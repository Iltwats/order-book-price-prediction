"""Measured, predeclared upper-GRU experiments within the existing round cap."""
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
from connectome.evaluation import check_parity, environment, evaluate as score_model, file_hash, save_json
from connectome.ort_training_bridge import (prepare_candidate_protocol, run_diagnostics,
    require_bridge_acceptance, require_candidate_acceptance)
from connectome.partial_gru import (DEFAULT_PARAMETERS, partial_parameters,
    train_partial_gru, resume_partial_gru)

BRIDGE = cycle.ROUND / "ort_training_bridge/report.json"
PILOT = cycle.ROUND / "partial_gru_pilot16"
ARMS = {"partial_gru_lr1e-5": 1e-5, "partial_gru_lr3e-5": 3e-5}


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return list(dict.fromkeys([str(p) for p in cycle._sources()] + [
        str(ROOT / "connectome/partial_gru.py"), str(ROOT / "connectome/ort_training_bridge.py"),
        str(ROOT / "connectome/numerical_bridge.py"),
        str(ROOT / "connectome/conversion_audit.py"), str(ROOT / "connectome/selection.py"),
        str(Path(__file__).resolve()), str(BRIDGE), str(BRIDGE.with_name("protocol.json"))]))


def pilot():
    require_bridge_acceptance(BRIDGE, cycle.SOURCE)
    groups = read(cycle.ANCHOR)["specification"]["groups"][:16]
    receipt = PILOT.with_suffix(".declaration.json")
    if receipt.exists():
        raise FileExistsError("timing pilot already declared; use its checkpoint instead of redrawing")
    save_json(receipt, {"purpose": "16-sequence timing/implementation pilot; no validation score or candidate selection",
                       "groups": groups, "epochs": 1, "seed": 20260912,
                       "parameters": partial_parameters(),
                       "source_artifacts": [experiments._artifact(p) for p in sources()]})
    report = train_partial_gru(cycle.SOURCE, cycle.TRAIN, PILOT, group_indices=groups)
    save_json(PILOT / "timing_projection.json", {
        "measured_sequences": 16, "measured_seconds": report["seconds"],
        "projected_seconds_per_256_sequence_configuration": report["seconds"] * 16,
        "projected_seconds_two_configurations": report["seconds"] * 32,
        "method": "linear planning estimate only; not a measured larger training run or inference timing",
        "peak_rss_bytes": environment()["peak_rss_bytes"],
    })
    return report


def declare():
    require_bridge_acceptance(BRIDGE, cycle.SOURCE)
    pilot_report = read(PILOT / "report.json")
    pilot_declaration = read(PILOT.with_suffix(".declaration.json"))
    for source in pilot_declaration["source_artifacts"]:
        experiments._check_artifact(source)
    for path, expected in pilot_report["specification"]["implementation"].items():
        if file_hash(path) != expected:
            raise ValueError("pilot training implementation changed before declaration")
    groups = read(cycle.ANCHOR)["specification"]["groups"][:256]
    if (not pilot_report["complete"] or pilot_report["specification"]["groups"] != groups[:16]
            or pilot_declaration["groups"] != groups[:16]
            or pilot_report["specification"]["epochs"] != 1 or pilot_report["specification"]["seed"] != 20260912
            or pilot_report["specification"]["parameters"] != partial_parameters()
            or pilot_declaration["parameters"] != partial_parameters()
            or pilot_report["specification"]["source"]["sha256"] != file_hash(cycle.SOURCE)
            or pilot_report["model_sha256"] != file_hash(PILOT / "model.onnx")
            or not pilot_report["lower_recurrence_bitwise_equal"]):
        raise ValueError("complete matching 16-sequence timing pilot required")
    registry = cycle.ROUND / "registry.json"
    record, _ = experiments._load(registry)
    snapshot = cycle.ROUND / "registry_after_heads.json"
    if not snapshot.exists():
        if len(record["candidates"]) != 6:
            raise ValueError("archive the head-stage registry before adding recurrent experiments")
        with snapshot.open("xb") as stream:
            stream.write(registry.read_bytes())
    for candidate_id, rate in ARMS.items():
        directory = cycle.ROUND / candidate_id
        parameters = partial_parameters(learning_rate=rate)
        spec = {
            "hypothesis": "Improve upper recurrent representation using current-student ORT forward history, local 256-row Torch gradients and an unchanged selected teacher",
            "training_groups": groups, "training_subset_reason": "Preplanned first 256 groups of the frozen affine1024 anchor for one-epoch recurrent screening",
            "seed": 20260912, "epochs": 1, "parameters": parameters,
            "teacher_lineage": [{"model_sha256": file_hash(cycle.SOURCE),
                                 "kind": "original_onnx_raw_selected_predictions_from_current_and_past_features"}],
            "config": {"model": "model.onnx", "backend": "onnx", "onnx_mode": "iobinding",
                       "strict_inputs": True, "affinity": False, "postprocess": {}},
            "config_path": str(directory / "config.json"),
            "training_report_path": str(directory / "report.json"),
            "source_paths": sources() + [str(PILOT / "report.json"), str(PILOT.with_suffix(".declaration.json"))],
        }
        experiments.register_candidate(registry, candidate_id, spec)
    print("Declared two upper-GRU learning rates; existing 12-screen/4-dev caps unchanged.", flush=True)


def fit(candidate_id, *, resume=False):
    _, _, entry = cycle._entry(candidate_id)
    require_bridge_acceptance(BRIDGE, cycle.SOURCE)
    spec = entry["spec"]
    directory = Path(spec["config_path"]).parent
    if resume:
        report = resume_partial_gru(directory / "checkpoint.pt")
    else:
        report = train_partial_gru(cycle.SOURCE, cycle.TRAIN, directory,
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
        parity = read(parity_path)
        if (not parity["passed"] or not parity["bitwise_equal"] or
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
    sub.add_parser("pilot")
    sub.add_parser("declare")
    for name in ("fit", "resume", "screen", "dev", "run"):
        child = sub.add_parser(name)
        child.add_argument("candidate", choices=list(ARMS))
    args = parser.parse_args()
    if args.command == "pilot":
        pilot()
    elif args.command == "declare":
        declare()
    elif args.command in {"fit", "resume", "run"}:
        fit(args.candidate, resume=args.command == "resume")
        if args.command == "run":
            evaluate(args.candidate, "screen")
    else:
        evaluate(args.candidate, args.command)
