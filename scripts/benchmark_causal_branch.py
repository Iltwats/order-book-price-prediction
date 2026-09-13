"""Unscored dynamics engineering benchmark, default eight smoke sequences.

Run inside the already prepared offline inference image, for example:
  python scripts/benchmark_causal_branch.py --output artifacts/causal_benchmark \
    --require-linux-limits

Each timed arm runs in a fresh process for interpretable RSS. Three cyclic run
orders counterbalance four arms; perfect position balance would require four
repetitions. This script neither launches containers nor accesses score labels.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARMS = ("published_baseline", "selected_incumbent", "zero_dynamics", "nonzero_dynamics")
SELECTED_SHA = "d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_orders():
    return [ARMS[offset:] + ARMS[:offset] for offset in range(3)]


def select_groups(manifest, requested=None, count=8):
    smoke = manifest["splits"]["smoke"]
    groups = smoke[:count] if requested is None else list(requested)
    if (not 1 <= len(groups) <= 32 or len(groups) != len(set(groups))
            or any(type(g) is not int for g in groups) or not set(groups) <= set(smoke)
            or set(groups) & set(manifest["splits"]["confirmation"])):
        raise ValueError("benchmark requires unique complete smoke groups, excluding confirmation")
    if requested is None and (not 1 <= count <= 32 or len(groups) != count):
        raise ValueError("requested smoke count is unavailable")
    return groups


def load_features(dataset, groups, expected_ids):
    """Read only metadata and features; labels and scoring masks stay unread."""
    import numpy as np
    import pyarrow.parquet as pq
    from utils import FEATURE_COLUMNS, validate_sequence
    sequences = []
    columns = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS]
    parquet = pq.ParquetFile(dataset)
    try:
        for group in groups:
            table = parquet.read_row_group(group, columns=columns, use_threads=False)
            if any(table[name].null_count for name in columns):
                raise ValueError("null benchmark feature or metadata value")
            seq_ix, need = validate_sequence(table)
            if seq_ix != expected_ids[str(group)]:
                raise ValueError("sequence ID differs from frozen manifest")
            features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(np.float32, copy=False)
            if not np.isfinite(features).all():
                raise ValueError("nonfinite benchmark features")
            sequences.append(SimpleNamespace(row_group=group, seq_ix=seq_ix, need=need, features=features))
    finally:
        parquet.close()
    return sequences


def rss():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = None
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                current = int(line.split()[1]) * 1024
                break
    return {"current_bytes": current, "process_peak_bytes": int(peak if sys.platform == "darwin" else peak * 1024)}


def constraints(required):
    from scripts.run_linux_checks import constraint_evidence, MEMORY_LIMIT_BYTES
    evidence = constraint_evidence()
    if required:
        cpu = (evidence.get("cpu.max") or "").split()
        if (sys.platform != "linux" or len(cpu) != 2 or cpu[0] == "max"
                or int(cpu[0]) <= 0 or int(cpu[0]) != int(cpu[1])
                or evidence.get("memory.max") != str(MEMORY_LIMIT_BYTES)
                or evidence.get("memory.swap.max") != "0"
                or evidence["active_non_loopback_interfaces"]):
            raise RuntimeError("requires Linux, one CPU quota, 16 GB, no swap and offline networking")
    return evidence


def factory(name, specification):
    from connectome.runtime import StreamingPredictor, load_predictor
    from connectome.causal_branch import DynamicsBackend
    if name == "published_baseline":
        spec = importlib.util.spec_from_file_location("actual_published_causal_benchmark", specification["baseline_solution"])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.PredictionModel
    if name == "selected_incumbent":
        return lambda: load_predictor(specification["selected_config"])
    return lambda: StreamingPredictor(DynamicsBackend(specification["fixtures"][name]["model_path"]), strict=True)


def verify_runtime(predictor, name):
    session = predictor.session if name == "published_baseline" else predictor.backend.session
    options = session.get_session_options()
    if (session.get_providers() != ["CPUExecutionProvider"]
            or options.intra_op_num_threads != 1 or options.inter_op_num_threads != 1):
        raise AssertionError("all arms must use the single-thread CPU runtime")
    if name != "published_baseline" and (not predictor.strict or predictor.capture_raw):
        raise AssertionError("timed candidates must use strict checks without raw-prediction capture")


def verify_ownership(saved):
    if any(value.tobytes() != original for value, original in saved):
        raise AssertionError("a retained prediction was overwritten by later callbacks")


def zero_parity(specification, sequences):
    """Strict raw-output/GRU-state comparisons on every real row, before timing."""
    from utils import DataPoint
    import numpy as np
    selected = factory("selected_incumbent", specification)()
    zero = factory("zero_dynamics", specification)()
    verify_runtime(selected, "selected_incumbent")
    verify_runtime(zero, "zero_dynamics")
    selected.capture_raw = zero.capture_raw = True  # untimed correctness only
    rows, saved = 0, []
    prediction_hash = hashlib.sha256()
    for sequence in sequences:
        for step, features in enumerate(sequence.features):
            point = DataPoint(sequence.seq_ix, step, bool(sequence.need[step]), features)
            expected, actual = selected.predict(point), zero.predict(point)
            if (actual is None) != (expected is None):
                raise AssertionError("zero fixture changed warm-up behavior")
            if zero.last_raw_prediction.tobytes() != selected.last_raw_prediction.tobytes():
                raise AssertionError(f"zero fixture raw prediction mismatch: group={sequence.row_group}, step={step}")
            prediction_hash.update(zero.last_raw_prediction.tobytes())
            for a, b in zip(zero.backend.hidden_states(), selected.backend.hidden_states()):
                if a.tobytes() != b.tobytes():
                    raise AssertionError("zero fixture changed a recurrent state")
            if step == 0:
                state = zero.backend.auxiliary_state()
                if state[-1] != 1 or state[:16].tobytes() != state[16:32].tobytes():
                    raise AssertionError("auxiliary state did not reset to the first observed projection")
            if step == 99:
                saved.extend((value, value.tobytes()) for value in (expected, actual))
            rows += 1
        if not np.isfinite(zero.backend.auxiliary_state()).all():
            raise AssertionError("nonfinite auxiliary state")
    verify_ownership(saved)
    return {"passed": True, "bitwise_equal": True, "rows": rows,
            "groups": specification["groups"], "all_row_hidden_states_checked": True,
            "output_ownership_checked": True, "raw_prediction_sha256": prediction_hash.hexdigest(),
            "selected_model_sha256": specification["selected_model_sha256"],
            "zero_model_sha256": specification["fixtures"]["zero_dynamics"]["model_sha256"]}


def timed_callback(name, specification, sequences):
    import numpy as np
    from utils import DataPoint
    construct = factory(name, specification)
    memory = {"before_startup": rss()}
    start = time.perf_counter()
    predictor = construct()
    startup = time.perf_counter() - start
    verify_runtime(predictor, name)
    memory["after_startup"] = rss()
    warm_id = -1
    while warm_id in {s.seq_ix for s in sequences}:
        warm_id -= 1
    saved = []
    for step, features in enumerate(sequences[0].features[:256]):
        value = predictor.predict(DataPoint(warm_id, step, step >= 99, features))
        if step == 99:
            saved.append((value, value.tobytes()))
    memory["after_warmup"] = rss()
    durations, last = [], None
    start = time.perf_counter()
    for sequence in sequences:
        sequence_start = time.perf_counter()
        for step, features in enumerate(sequence.features):
            last = predictor.predict(DataPoint(sequence.seq_ix, step, bool(sequence.need[step]), features))
        durations.append(time.perf_counter() - sequence_start)
    seconds = time.perf_counter() - start
    memory["after_callbacks"] = rss()
    verify_ownership(saved)
    if last is None or np.asarray(last).shape != (2,) or not np.isfinite(last).all():
        raise AssertionError("invalid final required prediction")
    rows = sum(len(s.features) for s in sequences)
    backend = getattr(predictor, "backend", None)
    if backend is not None:
        backend.close()
    return {"name": name, "seconds": seconds, "rows": rows,
            "callbacks_per_second": rows / seconds, "startup_seconds": startup,
            "sequence_seconds": durations, "rss": memory,
            "validation_policy": ("published_baseline_checks" if name == "published_baseline"
                                  else "current_strict_wrapper"),
            "current_strict_wrapper": name != "published_baseline",
            "output_ownership_checked": True}


def worker(args):
    from connectome.data import file_fingerprint
    from connectome.evaluation import environment
    specification = json.loads(args.spec.read_text())
    for path, expected in specification["artifact_sha256"].items():
        if digest(path) != expected:
            raise ValueError("benchmark artifact changed: " + path)
    if file_fingerprint(specification["dataset"]) != specification["dataset_fingerprint"]:
        raise ValueError("benchmark dataset fingerprint changed")
    evidence = constraints(specification["require_linux_limits"])
    before_load = rss()
    start = time.perf_counter()
    sequences = load_features(specification["dataset"], specification["groups"], specification["sequence_ids"])
    load_seconds = time.perf_counter() - start
    after_load = rss()
    result = (zero_parity(specification, sequences) if args.worker == "parity" else
              timed_callback(args.worker, specification, sequences))
    result.update(load_seconds_excluded=load_seconds, rss_before_load=before_load,
                  rss_after_load=after_load, environment=environment(), constraints=evidence,
                  scoring_performed=False, judge_runtime_verified=False)
    save_new(args.worker_output, result)


def summarize(records):
    import numpy as np
    result = {}
    for name in ARMS:
        values = [row for row in records if row["name"] == name]
        if len(values) != 3 or len({row["rows"] for row in values}) != 1:
            raise ValueError("each arm requires three repetitions over identical row counts")
        seconds = [row["seconds"] for row in values]
        result[name] = {"seconds": seconds, "median_seconds": float(np.median(seconds)),
                        "median_callbacks_per_second": values[0]["rows"] / float(np.median(seconds)),
                        "median_microseconds_per_callback": float(np.median(seconds))*1e6/values[0]["rows"],
                        "startup_seconds": [row["startup_seconds"] for row in values],
                        "peak_rss_bytes": [row["rss"]["after_callbacks"]["process_peak_bytes"] for row in values]}
    return result


def benchmark(args):
    from connectome.causal_branch import export_dynamics, make_fixture
    from connectome.data import file_fingerprint
    manifest = json.loads(args.manifest.read_text())
    groups = select_groups(manifest, args.groups, args.smoke_count)
    config = json.loads(args.selected_config.read_text())
    if (config.get("backend") != "onnx" or config.get("onnx_mode") != "iobinding"
            or config.get("strict_inputs") is not True or config.get("postprocess") != {}
            or config.get("affinity", False)):
        raise ValueError("benchmark requires the unchanged strict ONNX incumbent configuration")
    selected = args.selected_config.resolve().parent / config["model"]
    if digest(selected) != SELECTED_SHA:
        raise ValueError("selected model differs from the frozen incumbent")
    data_fp = file_fingerprint(args.data)
    if data_fp != manifest["datasets"]["valid"]["fingerprint"]:
        raise ValueError("data must match the frozen validation fingerprint")
    constraints(args.require_linux_limits)
    args.output.mkdir(parents=True, exist_ok=True)
    spec_path = args.output / "specification.json"
    if spec_path.exists() or (args.output / "benchmark_report.json").exists():
        raise FileExistsError("use a new benchmark output directory; prior results cannot be overwritten")
    fixtures = {}
    for name, nonzero in (("zero_dynamics", False), ("nonzero_dynamics", True)):
        fixtures[name] = export_dynamics(selected, args.output / f"{name}.onnx",
                                        make_fixture(seed=20260912, nonzero=nonzero))
    source_paths = [Path(__file__), ROOT / "connectome/causal_branch.py", ROOT / "connectome/runtime.py",
                    ROOT / "connectome/model_io.py", ROOT / "connectome/data.py", ROOT / "connectome/evaluation.py",
                    ROOT / "utils.py", ROOT / "scripts/run_linux_checks.py"]
    baseline = ROOT / "baseline/solution.py"
    artifacts = [*source_paths, baseline, baseline.parent / "baseline.onnx", selected, args.selected_config,
                 args.manifest, *(Path(item["model_path"]) for item in fixtures.values())]
    specification = {"groups": groups, "repetitions": 3, "run_orders": run_orders(),
                     "dataset": str(args.data.resolve()), "dataset_fingerprint": data_fp,
                     "sequence_ids": {str(row["row_group"]): row["seq_ix"] for row in manifest["datasets"]["valid"]["groups"] if row["row_group"] in groups},
                     "selected_config": str(args.selected_config.resolve()), "selected_model_sha256": SELECTED_SHA,
                     "baseline_solution": str(baseline), "fixtures": fixtures,
                     "artifact_sha256": {str(path.resolve()): digest(path) for path in artifacts},
                     "require_linux_limits": args.require_linux_limits,
                     "fixture_note": "Untrained synthetic zero/nonzero output fixtures with identity normalization; no statistics were fitted.",
                     "normalization_timing_limit": "ORT may remove identity normalization; nonidentity training normalization must be remeasured."}
    save_new(spec_path, specification)
    def launch(role, label):
        result_path = args.output / f"{label}.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", role,
                   "--spec", str(spec_path.resolve()), "--worker-output", str(result_path.resolve())]
        with (args.output / f"{label}.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return json.loads(result_path.read_text())
    # A separate correctness process cannot contaminate timed per-arm RSS.
    parity = launch("parity", "zero_parity")
    if not parity["passed"]:
        raise AssertionError("zero identity gate failed before timing")
    records = []
    for repetition, order in enumerate(run_orders(), 1):
        for name in order:
            gc.collect()
            record = launch(name, f"rep{repetition}_{name}")
            record["repetition"] = repetition
            records.append(record)
            print(f"{name} {repetition}/3: {record['callbacks_per_second']:.0f} callbacks/s", flush=True)
    summaries = summarize(records)
    branch_seconds = summaries["nonzero_dynamics"]["median_seconds"]
    report = {"completed": True, "specification_sha256": digest(spec_path), "zero_parity": parity,
              "groups": groups, "repetitions": 3, "rows_per_repetition": len(groups)*20_000,
              "run_orders": run_orders(), "ordered_runs": records, "summaries": summaries,
              "nonzero_throughput_ratio_vs_actual_baseline": summaries["published_baseline"]["median_seconds"] / branch_seconds,
              "nonzero_throughput_ratio_vs_selected": summaries["selected_incumbent"]["median_seconds"] / branch_seconds,
              "timed_domain": "Complete public DataPoint callback, including each arm's own validation/copies/state updates; features preloaded, startup and 256-row warm-up excluded.",
              "rss_scope": "Fresh process per repetition/arm; peak includes imports, feature preload, startup and callbacks. Current Linux RSS sampled outside timed loops.",
              "order_balance": "Three cyclic rotations, one occurrence of each arm per repetition and three distinct positions per arm; not perfect four-position balance.",
              "fixture_note": specification["fixture_note"], "normalization_timing_limit": specification["normalization_timing_limit"],
              "scoring_performed": False, "candidate_registration_performed": False,
              "profiled": False, "host_exclusive": False, "judge_runtime_verified": False,
              "hidden_test_rows": None, "promotion_certified": False}
    save_new(args.output / "benchmark_report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "datasets/valid.parquet")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/manifest.json")
    parser.add_argument("--selected-config", type=Path, default=ROOT / "submission_config.json")
    parser.add_argument("--groups", nargs="+", type=int)
    parser.add_argument("--smoke-count", type=int, default=8)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-linux-limits", action="store_true")
    parser.add_argument("--worker", choices=["parity", *ARMS], help=argparse.SUPPRESS)
    parser.add_argument("--spec", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"  # this process and its workers only; before numerical imports
    if args.worker:
        if args.spec is None or args.worker_output is None:
            parser.error("worker requires its frozen specification and output path")
        worker(args)
    else:
        if args.output is None:
            parser.error("--output is required")
        benchmark(args)


if __name__ == "__main__":
    main()
