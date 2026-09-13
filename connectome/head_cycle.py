"""Bounded round-two readout workflow: python -m connectome.head_cycle --help."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path

from . import experiments
from .calibration import paired_bootstrap
from .correlation import extract_moments, fit_correlation_head, load_moments
from .evaluation import check_parity, evaluate, file_hash, save_json
from .model_io import export_sequence_model

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "artifacts/round2"
SOURCE = ROOT / "artifacts/selected/model.onnx"
TRAIN = ROOT / "datasets/train.parquet"
VALID = ROOT / "datasets/valid.parquet"
MANIFEST = ROOT / "artifacts/manifest.json"
ANCHOR = ROOT / "artifacts/training_pilot1024/affine_p1e-2/report.json"
INCUMBENT_REPORT = ROOT / "artifacts/selected/dev_report.json"
ARMS = {
    "affine_mse": ("outputs", "mse", 1e-2),
    "affine_rho": ("outputs", "correlation", 1e-2),
    "head_rho_p1e-4": ("hidden", "correlation", 1e-4),
    "head_rho_p1e-3": ("hidden", "correlation", 1e-3),
    "head_rho_p1e-2": ("hidden", "correlation", 1e-2),
}


def _sources():
    return [ROOT / name for name in (
        "connectome/head_cycle.py", "connectome/correlation.py", "connectome/training.py",
        "connectome/model_io.py", "connectome/evaluation.py", "connectome/runtime.py",
        "connectome/experiments.py", "connectome/data.py", "connectome/calibration.py", "utils.py")]


def _read(path):
    return json.loads(Path(path).read_text())


def initialize():
    registry = ROUND / "registry.json"
    experiments.create_registry(registry, manifest_path=MANIFEST,
                                incumbent_config_path=ROOT / "submission_config.json",
                                incumbent_report_path=INCUMBENT_REPORT,
                                training_report_path=ANCHOR, source_paths=_sources())
    groups = _read(ANCHOR)["specification"]["groups"]
    for candidate_id, (representation, objective, penalty) in ARMS.items():
        directory = ROUND / candidate_id
        parameters = {"representation": representation, "objective": objective, "penalty": penalty,
                      "max_iterations": 200, "threads": 1, "offset_policy": "training_centered_selected"}
        spec = {"hypothesis": f"Selected {representation} readout optimized for {objective}; anchored centered offset",
                "training_groups": groups, "seed": 20260912, "epochs": 0,
                "parameters": parameters, "teacher_lineage": [],
                "config": {"model": "model.onnx", "backend": "onnx", "onnx_mode": "iobinding",
                           "strict_inputs": True, "affinity": False, "postprocess": {}},
                "config_path": str(directory / "config.json"), "training_report_path": str(directory / "report.json"),
                "source_paths": _sources()}
        # JSON signatures require strings, not Path objects.
        spec["source_paths"] = list(map(str, spec["source_paths"]))
        experiments.register_candidate(registry, candidate_id, spec)
    print(f"Declared {len(ARMS)} readout configurations; global caps remain 12 screen / 4 full dev.")


def _entry(candidate_id):
    record, manifest = experiments._load(ROUND / "registry.json")
    entry = record["candidates"][candidate_id]
    for source in entry["source_artifacts"]:
        experiments._check_artifact(source)
    return record, manifest, entry


def fit(candidate_id, moments_path=None):
    record, manifest, entry = _entry(candidate_id)
    spec = entry["spec"]
    path = moments_path or ROUND / "moments1024"
    _, _, _, metadata = load_moments(path, model_path=SOURCE, train_path=TRAIN, groups=spec["training_groups"])
    if metadata["training_data"] != manifest["datasets"]["train"]["fingerprint"]:
        raise ValueError("training statistics do not match round-two data")
    parameters = {k: v for k, v in spec["parameters"].items() if k != "offset_policy"}
    directory = Path(spec["config_path"]).parent
    report = fit_correlation_head(SOURCE, path, directory, seed=spec["seed"], **parameters)
    save_json(spec["config_path"], spec["config"])
    export_sequence_model(report["model_path"], directory / "sequence.onnx")
    print(f"fit {candidate_id}: training surrogate rho {report['optimization']['training_unclipped_rho']:.6f}; "
          f"{report['seconds']:.1f}s (not development WP)", flush=True)
    return report


def run_evaluation(candidate_id, phase):
    _, manifest, entry = _entry(candidate_id)
    directory = Path(entry["spec"]["config_path"]).parent
    output = directory / f"{phase}_report.json"
    if output.exists():
        raise FileExistsError("evaluation result already exists")
    if phase not in ("screen", "dev"):
        raise ValueError("only screen/dev allowed")
    registry = ROUND / "registry.json"
    experiments.reserve_evaluation(registry, candidate_id, phase)
    try:
        parity_path = directory / "parity.json"
        if phase == "screen":
            with (directory / "parity.log").open("w") as log, redirect_stdout(log):
                check_parity(VALID, directory / "model.onnx", groups=manifest["splits"]["smoke"],
                             mode="iobinding", bulk_model=directory / "sequence.onnx", report_path=parity_path)
            print(f"{candidate_id}: callback/reference and bulk parity passed 32 full sequences", flush=True)
        parity = _read(parity_path)
        if (not parity["passed"] or parity["original_sha256"] != file_hash(directory / "model.onnx")
                or parity["bulk_model_sha256"] != file_hash(directory / "sequence.onnx")):
            raise ValueError("bulk parity does not match evaluated model")
        result = evaluate(VALID, directory / "model.onnx", groups=manifest["splits"][phase],
                          mode="iobinding", report_path=output, bulk_model=directory / "sequence.onnx")
        receipt = experiments.record_result(registry, candidate_id, phase, output)
        incumbent = _read(INCUMBENT_REPORT)
        chosen = set(manifest["splits"][phase])
        incumbent["per_sequence"] = [row for row in incumbent["per_sequence"] if row["row_group"] in chosen]
        comparison = paired_bootstrap(result, incumbent, n_resamples=10000, seed=20260913)
        save_json(directory / f"{phase}_comparison.json", comparison)
        print(f"{candidate_id} {phase}: WP={receipt['weighted_pearson']:.9f}, "
              f"delta={receipt['delta']:+.9f}, CI=[{comparison['ci_low']:+.9f},{comparison['ci_high']:+.9f}]", flush=True)
        return result
    except Exception as exc:
        current = _read(registry)["candidates"][candidate_id]["evaluations"][phase]
        if current["status"] == "reserved":
            experiments.record_result(registry, candidate_id, phase, status="failed", reason=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    stats = sub.add_parser("moments")
    stats.add_argument("--limit", type=int, choices=[32, 1024], default=1024)
    stats.add_argument("--output", type=Path)
    for command in ("fit", "screen", "dev", "run"):
        child = sub.add_parser(command)
        child.add_argument("candidate", choices=list(ARMS))
    sub.add_parser("run-all")
    args = parser.parse_args()
    if args.command == "init":
        initialize()
    elif args.command == "moments":
        groups = _read(ANCHOR)["specification"]["groups"][:args.limit]
        extract_moments(SOURCE, TRAIN, groups, args.output or ROUND / f"moments{args.limit}")
    elif args.command == "fit":
        fit(args.candidate)
    elif args.command in ("screen", "dev"):
        run_evaluation(args.candidate, args.command)
    elif args.command == "run":
        fit(args.candidate)
        run_evaluation(args.candidate, "screen")
    else:
        for candidate in ARMS:
            fit(candidate)
            run_evaluation(candidate, "screen")


if __name__ == "__main__":
    main()
