"""Direct-correlation upper-GRU trials within the unchanged round-two cap."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connectome import experiments
from connectome import head_cycle as cycle
from connectome.calibration import paired_bootstrap
from connectome.evaluation import evaluate as score_model, file_hash, save_json
from connectome.recurrent_parity import check_parity, require_parity
from connectome.correlation_training_bridge import (prepare_candidate_protocol, run_diagnostics,
    require_bridge_acceptance, require_candidate_acceptance)
from connectome.correlation_gru import (DEFAULT_PARAMETERS, correlation_parameters,
    train_correlation_gru, resume_correlation_gru)

BRIDGE = cycle.ROUND / "correlation_training_bridge/report.json"
POLICY = cycle.ROUND / "correlation_gru_policy.json"
ARMS = {"correlation_gru_lr1e-5": 1e-5, "correlation_gru_lr3e-5": 3e-5}


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    names = ("connectome/partial_gru.py", "connectome/ort_training_bridge.py", "connectome/numerical_bridge.py",
             "connectome/recurrent_parity.py", "connectome/conversion_audit.py", "connectome/selection.py",
             "connectome/correlation_gru.py", "connectome/recurrent_objectives.py", "connectome/correlation_training_bridge.py")
    return list(dict.fromkeys([str(p) for p in cycle._sources()] + [str(ROOT / name) for name in names] + [
        str(Path(__file__).resolve()), str(BRIDGE), str(BRIDGE.with_name("protocol.json")), str(POLICY),
        str(cycle.ROUND / "recurrent_mse_results.json"), str(cycle.ROUND / "registry_after_mse.json")]))


def declare():
    require_bridge_acceptance(BRIDGE, cycle.SOURCE)
    policy = read(POLICY)
    groups = read(cycle.ANCHOR)["specification"]["groups"][:256]
    if (policy["groups"] != groups or policy["arms"] != ARMS or policy["seed"] != 20260912
            or policy["epochs"] != 1 or policy["parameters"] != correlation_parameters()
            or policy["caps"] != {"screen": 12, "dev": 4}):
        raise ValueError("objective procedure differs from the prospective trial allocation")
    for artifact in policy["source_artifacts"]:
        experiments._check_artifact(artifact)
    previous = read(cycle.ROUND / "recurrent_mse_results.json")
    if len(previous["candidates"]) != 2 or any(row["comparison"]["ci_high"] >= 0 for row in previous["candidates"]):
        raise ValueError("this conditional objective refinement requires the recorded negative MSE results")
    registry = cycle.ROUND / "registry.json"
    record, _ = experiments._load(registry)
    if record["caps"] != policy["caps"]:
        raise ValueError("existing round cap must remain unchanged")
    for candidate_id, rate in ARMS.items():
        directory = cycle.ROUND / candidate_id
        spec = {
            "hypothesis": "Improve recurrent correlation using full-sequence clipped TRAIN Pearson upstream gradients and exact current-student ONNX forward history",
            "training_groups": groups,
            "training_subset_reason": "Same first 256 groups of the frozen 1024 training anchor as completed MSE trials; prospectively change the objective only",
            "seed": 20260912, "epochs": 1, "parameters": correlation_parameters(learning_rate=rate),
            "teacher_lineage": [{"model_sha256": file_hash(cycle.SOURCE),
                                 "kind": "unchanged_selected_original_onnx_raw_causal_predictions"}],
            "config": {"model": "model.onnx", "backend": "onnx", "onnx_mode": "iobinding",
                       "strict_inputs": True, "affinity": False, "postprocess": {}},
            "config_path": str(directory / "config.json"), "training_report_path": str(directory / "report.json"),
            "source_paths": sources(),
        }
        experiments.register_candidate(registry, candidate_id, spec)
    print("Declared two correlation-objective trials; unchanged 12-screen/4-dev caps.", flush=True)


def fit(candidate_id, *, resume=False):
    _, _, entry = cycle._entry(candidate_id)
    if entry["evaluations"]:
        raise ValueError("cannot fit or resume after evaluation has been reserved")
    require_bridge_acceptance(BRIDGE, cycle.SOURCE)
    spec = entry["spec"]
    directory = Path(spec["config_path"]).parent
    existing = directory / "report.json"
    if existing.exists() and read(existing).get("complete"):
        if not resume:
            raise FileExistsError("complete fit already exists; resume may only finish its unscored checks")
        report = read(existing)
    elif resume:
        report = resume_correlation_gru(directory / "checkpoint.pt")
    else:
        report = train_correlation_gru(cycle.SOURCE, cycle.TRAIN, directory,
            group_indices=spec["training_groups"], epochs=spec["epochs"], seed=spec["seed"],
            **{k: spec["parameters"][k] for k in DEFAULT_PARAMETERS})
    if not report["complete"] or report["model_sha256"] != file_hash(directory / "model.onnx"):
        raise ValueError("training must finish and match its exported model before scoring")
    save_json(spec["config_path"], spec["config"])
    bridge_directory = directory / "bridge"
    protocol = bridge_directory / "protocol.json"
    if not protocol.exists():
        protocol = Path(prepare_candidate_protocol(directory / "model.onnx", BRIDGE, bridge_directory))
    if not (bridge_directory / "report.json").exists():
        run_diagnostics(protocol)
    require_candidate_acceptance(directory / "bridge/report.json", directory / "model.onnx")
    return report


def evaluate(candidate_id, phase):
    _, manifest, entry = cycle._entry(candidate_id)
    if phase not in {"screen", "dev"}:
        raise ValueError("only the declared screen/dev domains are allowed")
    directory = Path(entry["spec"]["config_path"]).parent
    require_candidate_acceptance(directory / "bridge/report.json", directory / "model.onnx")
    report = read(directory / "report.json")
    if (report["model_sha256"] != file_hash(directory / "model.onnx")
            or not report["lower_recurrence_bitwise_equal"]):
        raise ValueError("training/export evidence mismatch")
    output = directory / f"{phase}_report.json"
    if output.exists():
        raise FileExistsError("evaluation result already exists")
    registry = cycle.ROUND / "registry.json"
    experiments.reserve_evaluation(registry, candidate_id, phase)
    try:
        parity_path = directory / "parity.json"
        if phase == "screen":
            with (directory / "parity.log").open("w") as log, redirect_stdout(log):
                check_parity(cycle.VALID, directory / "model.onnx", groups=manifest["splits"]["smoke"],
                             mode="iobinding", bulk_model=directory / "sequence.onnx", report_path=parity_path)
        parity = require_parity(parity_path, cycle.VALID, directory / "model.onnx",
                                groups=manifest["splits"]["smoke"], bulk_model=directory / "sequence.onnx")
        if (not parity["passed"] or not parity["bitwise_equal"] or not parity["bulk_bitwise_equal"] or
                parity["original_sha256"] != file_hash(directory / "model.onnx") or
                parity["bulk_model_sha256"] != file_hash(directory / "sequence.onnx")):
            raise ValueError("matching bitwise callback/bulk parity is required before scoring")
        print(f"{candidate_id}: matching bitwise parity passed", flush=True)
        result = score_model(cycle.VALID, directory / "model.onnx", groups=manifest["splits"][phase],
                             mode="iobinding", report_path=output, bulk_model=directory / "sequence.onnx")
        receipt = experiments.record_result(registry, candidate_id, phase, output)
        incumbent = read(cycle.INCUMBENT_REPORT)
        groups = set(manifest["splits"][phase])
        incumbent["per_sequence"] = [row for row in incumbent["per_sequence"] if row["row_group"] in groups]
        comparison = paired_bootstrap(result, incumbent, n_resamples=10000, seed=20260913)
        save_json(directory / f"{phase}_comparison.json", comparison)
        print(f"{candidate_id} {phase}: WP={receipt['weighted_pearson']:.9f}, "
              f"delta={receipt['delta']:+.9f}, CI=[{comparison['ci_low']:+.9f},{comparison['ci_high']:+.9f}]", flush=True)
        return result
    except Exception as exc:
        current = read(registry)["candidates"][candidate_id]["evaluations"][phase]
        if current["status"] == "reserved":
            experiments.record_result(registry, candidate_id, phase, status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("declare")
    for name in ("fit", "resume", "screen", "dev", "run"):
        child = sub.add_parser(name)
        child.add_argument("candidate", choices=list(ARMS))
    args = parser.parse_args()
    if args.command == "declare":
        declare()
    elif args.command in {"fit", "resume", "run"}:
        fit(args.candidate, resume=args.command == "resume")
        if args.command == "run":
            evaluate(args.candidate, "screen")
    else:
        evaluate(args.candidate, args.command)
