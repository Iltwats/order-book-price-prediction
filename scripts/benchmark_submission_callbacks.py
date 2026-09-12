"""Compare the published callback itself with a configured strict candidate.

This is separate from ORT API microbenchmarks: the reference is loaded directly
from baseline/solution.py, and every timed call uses the public DataPoint API.
"""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow.parquet as pq

from connectome.data import read_sequence
from connectome.evaluation import environment
from connectome.runtime import apply_affinity, load_predictor
from utils import BlockAccumulator, DataPoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def benchmark_solutions(dataset_path, baseline_solution, candidate_config, groups,
                        repetitions=3, affinity=False):
    baseline_solution, candidate_config = Path(baseline_solution), Path(candidate_config)
    config = json.loads(candidate_config.read_text())
    if config.get("strict_inputs", True) is not True:
        raise ValueError("the selected candidate config must enable strict_inputs=True")
    specification = importlib.util.spec_from_file_location("connectome_published_benchmark", baseline_solution)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    factories = {"published_baseline": module.PredictionModel,
                 "selected_candidate_strict": lambda: load_predictor(candidate_config)}
    parquet = pq.ParquetFile(dataset_path)
    load_start = time.perf_counter()
    sequences = [read_sequence(parquet, int(group)) for group in groups]
    if not sequences or repetitions < 1:
        raise ValueError("benchmark requires complete sequences and positive repetitions")
    load_seconds = time.perf_counter() - load_start
    affinity_result = apply_affinity(affinity)
    records = {name: [] for name in factories}
    ordered_runs = []
    total_start = time.perf_counter()
    for repetition in range(repetitions):
        # Balance run order rather than timing every baseline run first.
        order = list(factories) if repetition % 2 == 0 else list(reversed(factories))
        for name in order:
            startup_start = time.perf_counter()
            predictor = factories[name]()
            startup_seconds = time.perf_counter() - startup_start
            if name == "selected_candidate_strict" and not predictor.strict:
                raise AssertionError("candidate input validation was disabled")
            # A distinct warm-up sequence id lets BOTH actual callback classes
            # reset through their normal public contract at the first timed row.
            warm_id = int(sequences[0].seq_ix) ^ 0x7FFFFFFF
            for step, row in enumerate(sequences[0].features[:256]):
                predictor.predict(DataPoint(warm_id, step, step >= 99, row))
            durations = []
            last = None
            start = time.perf_counter()
            for sequence in sequences:
                sequence_start = time.perf_counter()
                for step, row in enumerate(sequence.features):
                    last = predictor.predict(DataPoint(sequence.seq_ix, step, step >= 99, row))
                durations.append(time.perf_counter() - sequence_start)
            seconds = time.perf_counter() - start
            if last is None or np.asarray(last).shape != (2,) or not np.isfinite(last).all():
                raise AssertionError("callback did not produce valid final predictions")
            rows = 20_000 * len(sequences)
            record = {"repetition": repetition + 1, "seconds": seconds,
                      "callbacks_per_second": rows / seconds, "startup_seconds": startup_seconds,
                      "sequence_seconds": durations}
            records[name].append(record)
            ordered_runs.append({"name": name, **record})
            print(f"solutions {name} {repetition+1}/{repetitions}: {rows / seconds:.0f} callbacks/s", flush=True)
            backend = getattr(predictor, "backend", None)
            if backend is not None:
                backend.close()
            del predictor
            gc.collect()
    summaries = {}
    for name, values in records.items():
        seconds = [value["seconds"] for value in values]
        sequence_seconds = [duration for value in values for duration in value["sequence_seconds"]]
        summaries[name] = {
            "seconds": seconds, "median_seconds": float(np.median(seconds)),
            "median_callbacks_per_second": rows / float(np.median(seconds)),
            "mean_microseconds_per_callback": float(np.mean(seconds)) * 1e6 / rows,
            "startup_seconds": [value["startup_seconds"] for value in values],
            "sequence_p50_seconds": float(np.quantile(sequence_seconds, 0.5)),
            "sequence_p95_seconds": float(np.quantile(sequence_seconds, 0.95)),
        }
    ratio = summaries["published_baseline"]["median_seconds"] / summaries["selected_candidate_strict"]["median_seconds"]
    model_path = candidate_config.parent / config["model"]
    return {
        "reference": "actual baseline/solution.py PredictionModel.predict",
        "baseline_solution_sha256": digest(baseline_solution),
        "baseline_model_sha256": digest(baseline_solution.parent / "baseline.onnx"),
        "candidate_config": config, "candidate_config_sha256": digest(candidate_config),
        "candidate_model_sha256": digest(model_path), "strict_inputs": True,
        "groups": list(map(int, groups)), "repetitions": repetitions, "rows_per_repetition": rows,
        "ordered_runs": ordered_runs, "summaries": summaries,
        "candidate_throughput_ratio_vs_published": ratio,
        "load_seconds_excluded": load_seconds, "seconds": time.perf_counter() - total_start,
        "affinity": affinity_result, "environment": environment(),
        "timed_domain": "actual complete Python DataPoint callback; data preloaded; startup and warm-up separate",
        "hardware_counters": {"collected": False, "perf_available": shutil.which("perf") is not None,
                              "reason": "No hardware cache/performance counters were collected; the inference image does not install perf"},
        "host_exclusive": False, "judge_runtime_verified": False, "hidden_test_rows": None,
    }


def evaluate_streaming_candidate(dataset_path, candidate_config, groups, affinity=False):
    """Score one complete sequence at a time using the production configuration.

    End-to-end timing includes startup, Parquet reads, strict public callbacks,
    prediction storage for the current sequence, checks, and official scoring.
    It deliberately excludes prediction caches and the offline sequence graph.
    """
    candidate_config = Path(candidate_config)
    config = json.loads(candidate_config.read_text())
    if config.get("strict_inputs", True) is not True:
        raise ValueError("streamed validation requires strict_inputs=True")
    groups = list(map(int, groups))
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("streamed validation needs unique nonempty groups")
    affinity_result = apply_affinity(affinity)
    start = time.perf_counter()
    predictor = load_predictor(candidate_config)
    startup_seconds = time.perf_counter() - start
    if not predictor.strict:
        raise AssertionError("candidate input validation was disabled")
    parquet = pq.ParquetFile(dataset_path)
    accumulator = BlockAccumulator()
    per_sequence = []
    try:
        for index, group in enumerate(groups):
            sequence_start = time.perf_counter()
            sequence = read_sequence(parquet, group)
            predictions = np.full((len(sequence.features), 2), np.nan, np.float32)
            for step, row in enumerate(sequence.features):
                need = bool(sequence.need[step])
                output = predictor.predict(DataPoint(sequence.seq_ix, step, need, row))
                if need:
                    if output is None or output.shape != (2,) or not np.isfinite(output).all():
                        raise AssertionError("invalid required callback output")
                    predictions[step] = output
                elif output is not None:
                    raise AssertionError("warm-up must return None")
            if sequence.mask is None:
                raise ValueError("official streamed scoring requires validation masks")
            before = len(accumulator.scores)
            accumulator.add(sequence.targets, predictions, sequence.mask & sequence.need)
            eligible = len(accumulator.scores) > before
            scores = accumulator.scores[-1] if eligible else [None, None]
            per_sequence.append({"row_group": group, "seq_ix": sequence.seq_ix,
                                 "seconds": time.perf_counter() - sequence_start,
                                 "eligible": eligible, "t0": scores[0], "t1": scores[1],
                                 "weighted_pearson": float(np.mean(scores)) if eligible else None})
            if (index + 1) % 32 == 0 or index == len(groups) - 1:
                elapsed = time.perf_counter() - start
                print(f"stream {index+1}/{len(groups)} complete sequences; {(index+1)*20000/elapsed:.0f} rows/s", flush=True)
        result = accumulator.result()
    finally:
        predictor.backend.close()
    seconds = time.perf_counter() - start
    rows = len(groups) * 20_000
    result.update({
        "candidate_config": config, "candidate_config_sha256": digest(candidate_config),
        "candidate_model_sha256": digest(candidate_config.parent / config["model"]),
        "strict_inputs": True, "capture_raw": False, "prediction_cache": False,
        "execution": "production_row_callback_streaming", "groups": groups,
        "rows_seen": rows, "seconds": seconds, "rows_per_second": rows / seconds,
        "startup_seconds": startup_seconds, "per_sequence": per_sequence,
        "affinity": affinity_result, "environment": environment(),
        "timed_domain": "end-to-end: startup, one sequence Parquet read at a time, strict configured callback, validation and official scoring; no prediction cache",
        "peak_rss_caveat": "process lifetime high-water mark; includes Python/runtime/dataset/scorer overhead",
        "host_exclusive": False, "judge_runtime_verified": False, "hidden_test_rows": None,
    })
    return result
