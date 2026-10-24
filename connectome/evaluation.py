"""Global scoring, prediction caches, and complete-callback benchmarks."""
from __future__ import annotations

import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

from utils import BlockAccumulator, DataPoint, scoring_accumulator, add_scored_sequence, METRIC_ID
from .runtime import OnnxBackend, StreamingPredictor, apply_affinity, session_options


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def environment():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "architecture": platform.machine(), "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "peak_rss_bytes": int(rss if sys.platform == "darwin" else rss * 1024)}


def evaluate(dataset_path, model_path, groups=None, mode="run", cache_path=None,
             report_path=None, postprocess=None, bulk_model=None, progress=True, *, metric_id=METRIC_ID):
    from .data import read_sequence
    from .calibration import PredictionCache, apply_calibration
    parquet = pq.ParquetFile(dataset_path)
    groups = list(range(parquet.num_row_groups)) if groups is None else list(map(int, groups))
    if len(groups) != len(set(groups)) or not groups:
        raise ValueError("evaluation needs unique, nonempty row groups")
    start = time.perf_counter()
    predictor = None
    session = None
    if bulk_model:
        session = ort.InferenceSession(str(bulk_model), sess_options=session_options(),
                                       providers=["CPUExecutionProvider"])
        shapes = [x.shape for x in session.get_inputs()[1:]]
        widths = [int(s[-1]) for s in shapes]
    else:
        predictor = StreamingPredictor(OnnxBackend(model_path, mode), postprocess, capture_raw=True)
    startup = time.perf_counter() - start
    provenance = {"model_sha256": file_hash(model_path), "mode": "offline_sequence" if bulk_model else mode,
                  "dataset_path": str(Path(dataset_path).resolve()),
                  "dataset_size": Path(dataset_path).stat().st_size,
                  "environment": environment(), "raw_unclipped": True}
    if bulk_model:
        provenance["bulk_model_sha256"] = file_hash(bulk_model)
    configuration = {"mode": provenance["mode"], "model_sha256": provenance["model_sha256"],
                     "bulk_model_sha256": provenance.get("bulk_model_sha256"),
                     "runtime_sha256": file_hash(Path(__file__).with_name("runtime.py")),
                     "feature_representation": "original_112_fp32",
                     "onnxruntime": ort.__version__, "threads": 1}
    provenance["configuration"] = configuration
    provenance["configuration_sha256"] = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache = None
    if cache_path:
        ids = [int(parquet.read_row_group(g, columns=["seq_ix"])["seq_ix"][0].as_py()) for g in groups]
        cache = PredictionCache.create(cache_path, groups, ids, provenance)
    accumulator = scoring_accumulator(metric_id)
    rows = 0
    per_sequence = []
    for index, group in enumerate(groups):
        sequence = read_sequence(parquet, group)
        if session is not None:
            outputs = session.run(None, {"features": sequence.features[None, :, :],
                                         "hidden_0": np.zeros((1, 1, widths[0]), np.float32),
                                         "hidden_1": np.zeros((1, 1, widths[1]), np.float32)})
            raw = np.asarray(outputs[0][0], np.float32)
            if raw.shape != (20_000, 2) or not np.isfinite(raw).all():
                raise ValueError("invalid offline sequence predictions")
            predicted = apply_calibration(raw, **(postprocess or {})) if postprocess else raw
        else:
            raw = np.empty((20_000, 2), np.float32)
            predicted = np.full((20_000, 2), np.nan, np.float32)
            for step, features in enumerate(sequence.features):
                result = predictor.predict(DataPoint(sequence.seq_ix, step, bool(sequence.need[step]), features))
                raw[step] = predictor.last_raw_prediction
                if sequence.need[step]:
                    if result is None or result.shape != (2,) or not np.isfinite(result).all():
                        raise ValueError("invalid required prediction")
                    predicted[step] = result
                elif result is not None:
                    raise ValueError("warm-up must return None")
        if sequence.mask is None:
            raise ValueError("official evaluation requires validation scoring mask")
        per_sequence.append(add_scored_sequence(accumulator, sequence.targets, predicted, sequence.mask & sequence.need,
                                                row_group=group, seq_ix=sequence.seq_ix))
        rows += 20_000
        if cache:
            cache.write(index, raw)
        if progress and ((index + 1) % 32 == 0 or index == len(groups)-1):
            elapsed = time.perf_counter() - start
            has_rows = accumulator.selected_rows > 0
            if hasattr(accumulator, 'scores'):
                has_rows = bool(accumulator.scores)
            score = f"{accumulator.result()['weighted_pearson']:.6f}" if has_rows else "pending scoring rows"
            print(f"evaluate {index+1}/{len(groups)} sequences; {rows/elapsed:.0f} rows/s; "
                  f"WP={score}", flush=True)
    if cache:
        cache.finalize()
    result = accumulator.result()
    elapsed = time.perf_counter() - start
    result.update(provenance=provenance, rows_seen=rows, seconds=elapsed,
                  rows_per_second=rows/elapsed, startup_seconds=startup,
                  per_sequence=per_sequence, environment=environment(),
                  postprocess=postprocess or {},
                  execution="offline_sequence" if bulk_model else "row_callback",
                  judge_runtime_verified=False)
    if report_path:
        save_json(report_path, result)
    return result


def benchmark(dataset_path, model_path, groups, modes=("run",), repetitions=3,
              native_weights=None, native_library=None, affinity=False, report_path=None):
    from .data import read_sequence
    from .native import NativeBackend
    parquet = pq.ParquetFile(dataset_path)
    sequences = [read_sequence(parquet, int(g)) for g in groups]
    if not sequences or repetitions < 1:
        raise ValueError("benchmark requires sequences and positive repetition count")
    affinity_result = apply_affinity(affinity)
    runs = []
    for mode in modes:
        durations = []
        startups = []
        per_sequence = []
        for rep in range(repetitions):
            start = time.perf_counter()
            if mode.startswith("native_"):
                if not native_weights or not native_library:
                    raise ValueError("native benchmark needs weights and library")
                backend = NativeBackend(native_weights, native_library, layout=mode.split("_", 1)[1])
            else:
                backend = OnnxBackend(model_path, mode)
            predictor = StreamingPredictor(backend, strict=False)
            startups.append(time.perf_counter()-start)
            # Warm the runtime; reset before the timed complete sequences.
            for row in sequences[0].features[:256]:
                backend.step(row)
            backend.reset()
            start = time.perf_counter()
            for sequence in sequences:
                seq_start = time.perf_counter()
                for step, row in enumerate(sequence.features):
                    predictor.predict(DataPoint(sequence.seq_ix, step, step >= 99, row))
                per_sequence.append(time.perf_counter()-seq_start)
            duration = time.perf_counter()-start
            durations.append(duration)
            backend.close()
            print(f"benchmark {mode} {rep+1}/{repetitions}: {len(sequences)*20000/duration:.0f} callbacks/s", flush=True)
        rows = len(sequences)*20_000
        runs.append({"mode": mode, "repetitions": repetitions, "rows_per_repetition": rows,
                     "seconds": durations, "median_seconds": float(np.median(durations)),
                     "median_callbacks_per_second": rows/float(np.median(durations)),
                     "mean_microseconds_per_callback": float(np.mean(durations))*1e6/rows,
                     "startup_seconds": startups,
                     "sequence_p50_seconds": float(np.quantile(per_sequence, .5)),
                     "sequence_p95_seconds": float(np.quantile(per_sequence, .95))})
    result = {"model_sha256": file_hash(model_path), "groups": list(map(int, groups)),
              "environment": environment(), "affinity": affinity_result, "runs": runs,
              "timed_domain": "complete Python callback; sequences preloaded; startup separate",
              "judge_runtime_verified": False,
              "hidden_test_rows": None}
    if native_library:
        result["native_library_sha256"] = file_hash(native_library)
    if native_weights:
        result["native_weights_sha256"] = file_hash(native_weights)
    if report_path:
        save_json(report_path, result)
    return result


def check_parity(dataset_path, original_model, candidate_model=None, groups=(0,),
                 mode="run", native_weights=None, native_library=None, bulk_model=None,
                 report_path=None):
    from .data import read_sequence
    from .native import NativeBackend
    parquet = pq.ParquetFile(dataset_path)
    reference = OnnxBackend(original_model, "reference")
    candidate = (NativeBackend(native_weights, native_library, layout=mode.split("_", 1)[1])
                 if mode.startswith("native_") else OnnxBackend(candidate_model or original_model, mode))
    bulk = (ort.InferenceSession(str(bulk_model), sess_options=session_options(), providers=["CPUExecutionProvider"])
            if bulk_model else None)
    max_output_error = 0.0
    max_state_error = 0.0
    bitwise = True
    max_bulk_error = 0.0
    per_sequence = []
    for group in groups:
        sequence = read_sequence(parquet, int(group))
        reference.reset()
        candidate.reset()
        outputs = np.empty((20_000, 2), np.float32)
        trial = np.empty_like(outputs)
        reference_states = [np.empty((20_000, *state.shape), np.float32)
                            for state in reference.hidden_states()]
        trial_states = [np.empty_like(state) for state in reference_states]
        for step, row in enumerate(sequence.features):
            expected, actual = reference.step(row), candidate.step(row)
            outputs[step], trial[step] = expected, actual
            for layer, (h_expected, h_actual) in enumerate(zip(reference.hidden_states(), candidate.hidden_states())):
                reference_states[layer][step], trial_states[layer][step] = h_expected, h_actual
        # Compare every row in bounded, per-sequence buffers. This avoids tens
        # of thousands of Python assertion calls while retaining all states.
        max_output_error = max(max_output_error, float(np.max(np.abs(outputs-trial))))
        bitwise &= np.array_equal(outputs, trial)
        np.testing.assert_allclose(trial, outputs, atol=1e-6, rtol=1e-5,
                                   err_msg=f"output mismatch group={group}")
        for layer, (expected, actual) in enumerate(zip(reference_states, trial_states)):
            max_state_error = max(max_state_error, float(np.max(np.abs(expected-actual))))
            bitwise &= np.array_equal(expected, actual)
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5,
                                       err_msg=f"hidden mismatch group={group} layer={layer}")
        if bulk:
            value = bulk.run(None, {"features": sequence.features[None],
                                    "hidden_0": np.zeros((1, 1, 128), np.float32),
                                    "hidden_1": np.zeros((1, 1, 128), np.float32)})
            max_bulk_error = max(max_bulk_error, float(np.max(np.abs(value[0][0]-outputs))))
            np.testing.assert_allclose(value[0][0], outputs, atol=1e-6, rtol=1e-5)
            for actual, expected in zip(value[1:], reference.hidden_states()):
                np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        reference_acc, trial_acc = BlockAccumulator(), BlockAccumulator()
        reference_acc.add(sequence.targets, outputs, sequence.need & sequence.mask)
        trial_acc.add(sequence.targets, trial, sequence.need & sequence.mask)
        per_sequence.append({"row_group": int(group),
                             "reference": reference_acc.result()["weighted_pearson"] if reference_acc.scores else None,
                             "candidate": trial_acc.result()["weighted_pearson"] if trial_acc.scores else None})
        print(f"parity sequence {group}: output max {max_output_error:.3g}; hidden max {max_state_error:.3g}", flush=True)
    result = {"passed": True, "mode": mode, "groups": list(map(int, groups)),
              "bitwise_equal": bool(bitwise), "max_output_absolute_error": max_output_error,
              "max_hidden_absolute_error": max_state_error, "max_bulk_absolute_error": max_bulk_error,
              "rtol": 1e-5, "atol": 1e-6, "per_sequence": per_sequence,
              "environment": environment(), "original_sha256": file_hash(original_model)}
    result["candidate_sha256"] = file_hash(candidate_model or original_model)
    if bulk_model:
        result["bulk_model_sha256"] = file_hash(bulk_model)
    if report_path:
        save_json(report_path, result)
    return result
