"""Prospective TRAIN-only correlation-gradient audit, separate from MSE evidence.

Prepare freezes this objective and its implementation before any dataset read.
Initial checks cover eight fixed TRAIN sequences with a two-sequence stop stage;
candidate checks cover the first two. No training or selection runs here.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch

from . import numerical_bridge as old, ort_training_bridge as forward
from . import recurrent_objectives as objective
from .conversion_audit import export_original_graph
from .training import OnnxSequenceEncoder, load_torch_model

ROOT = old.ROOT


def procedure():
    return {"version": "correlation_ort_forward_gradient.v1", "objective": objective.objective_parameters(),
        "gradient": dict(objective.GRADIENT_TOLERANCE), "gradient_chunk_starts": [0, 19744],
        "teacher_weight": .1, "chunk_size": 256,
        "forward": "current-student ORT full trajectory; Torch chunk-local gradient and exact ORT forward proxies",
        "new_evidence": "correlation-specific full-sequence coefficients and FP32/FP64 gradients; MSE gradient evidence is not relabeled",
        "scope": "TRAIN-only approximation check; not masked WP or a model improvement score"}


def prepare_protocol(output_dir, *, model_path=ROOT / "artifacts/selected/model.onnx",
                     manifest_path=ROOT / "artifacts/manifest.json",
                     anchor_path=ROOT / "artifacts/training_pilot1024/affine_p1e-2/report.json",
                     forward_bridge_report=ROOT / "artifacts/round2/ort_training_bridge/report.json",
                     initial_bridge_report=None):
    model_path, manifest_path, anchor_path, forward_bridge_report = map(Path,
        (model_path, manifest_path, anchor_path, forward_bridge_report))
    forward_evidence = old._read(forward_bridge_report)
    teacher = Path(old._read(forward_evidence["protocol_path"])["model_path"])
    forward.require_bridge_acceptance(forward_bridge_report, teacher)
    if initial_bridge_report is None and model_path.resolve() != teacher.resolve():
        raise ValueError("initial correlation bridge requires the accepted selected source")
    if initial_bridge_report is not None:
        require_bridge_acceptance(initial_bridge_report, teacher)
    manifest, anchor = old._read(manifest_path), old._read(anchor_path)
    train, groups = manifest["datasets"]["train"], anchor["specification"]["groups"]
    if (len(groups) != 1024 or len(set(groups)) != 1024 or not set(groups) <= set(manifest["splits"]["train"]) or
            Path(anchor["specification"]["data"]["path"]).resolve() != Path(train["path"]).resolve()):
        raise ValueError("frozen 1024-group TRAIN anchor required")
    sources = [Path(__file__), Path(objective.__file__), Path(forward.__file__), Path(old.__file__),
        *[ROOT / name for name in ("connectome/correlation_gru.py", "connectome/partial_gru.py", "connectome/training.py",
          "connectome/conversion_audit.py", "connectome/model_io.py", "connectome/data.py", "connectome/evaluation.py", "utils.py")],
        model_path, teacher, manifest_path, anchor_path, forward_bridge_report, Path(forward_evidence["protocol_path"])]
    if initial_bridge_report is not None:
        sources.extend((Path(initial_bridge_report), Path(old._read(initial_bridge_report)["protocol_path"])))
    count = 2 if initial_bridge_report is not None else 8
    protocol = {"procedure": procedure(), "budget": copy.deepcopy(forward.BUDGET),
        "model_path": str(model_path.resolve()), "teacher_path": str(teacher.resolve()),
        "train_path": train["path"], "train_fingerprint": train["fingerprint"], "groups": groups[:count],
        "stages": [2] if count == 2 else [2, 8], "mode": "candidate_export" if count == 2 else "initial_source",
        "expected_sequence_ids": {str(g["row_group"]): g["seq_ix"] for g in train["groups"] if g["row_group"] in groups[:count]},
        "artifacts": {str(p.resolve()): old._sha(p) for p in sources},
        "forward_bridge_report": str(forward_bridge_report.resolve()),
        "strict_status": "Historical strict and conventional carried-state failures remain unchanged",
        "completion": f"{count} complete fixed TRAIN sequences; stop on first local/equation/gradient failure"}
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    old._write(output / "protocol.json", protocol, exclusive=True)
    return str(output / "protocol.json")


def prepare_candidate_protocol(model_path, initial_bridge_report, output_dir, **kwargs):
    return prepare_protocol(output_dir, model_path=model_path, initial_bridge_report=initial_bridge_report, **kwargs)


def _verify_protocol(path):
    protocol = old._read(path)
    stages = [2, 8] if protocol["mode"] == "initial_source" else [2]
    if (protocol["mode"] not in {"initial_source", "candidate_export"} or protocol["procedure"] != procedure() or
            protocol["budget"] != forward.BUDGET or protocol["stages"] != stages or len(protocol["groups"]) != stages[-1]):
        raise ValueError("changed prospective correlation-gradient procedure")
    for path, digest in protocol["artifacts"].items():
        if old._sha(path) != digest:
            raise ValueError(f"preregistered artifact changed: {path}")
    forward.require_bridge_acceptance(protocol["forward_bridge_report"], protocol["teacher_path"])
    return protocol


def run_diagnostics(protocol_path):
    from .data import file_fingerprint, iter_sequences
    from .model_io import load_weights
    protocol_path = Path(protocol_path).resolve()
    protocol, output = _verify_protocol(protocol_path), protocol_path.parent
    if (output / "report.json").exists():
        raise FileExistsError("preserve existing correlation-gradient result")
    if file_fingerprint(protocol["train_path"]) != protocol["train_fingerprint"]:
        raise ValueError("training dataset changed")
    weights, teacher_weights = load_weights(protocol["model_path"]), load_weights(protocol["teacher_path"])
    if any(weights[k].tobytes() != teacher_weights[k].tobytes() for k in ("w0", "r0", "b0")):
        raise ValueError("partial recurrence requires unchanged lower GRU")
    model = load_torch_model(protocol["model_path"])
    exported = export_original_graph(model, protocol["model_path"], output / "zero_update.onnx")
    if old._sha(exported) != old._sha(protocol["model_path"]):
        raise ValueError("actual model must survive byte-identical original-graph export")
    torch.set_num_threads(1)
    encoder = OnnxSequenceEncoder(exported, output / "encoder.onnx")
    teacher = encoder if old._sha(protocol["model_path"]) == old._sha(protocol["teacher_path"]) else OnnxSequenceEncoder(protocol["teacher_path"], output / "teacher.onnx")
    started, sequences = time.monotonic(), []
    report = {"protocol_path": str(protocol_path), "protocol_sha256": old._sha(protocol_path),
        "source_sha256": old._sha(protocol["model_path"]), "zero_update_sha256": old._sha(exported),
        "teacher_sha256": old._sha(protocol["teacher_path"]), "procedure": protocol["procedure"], "budget": protocol["budget"],
        "mode": protocol["mode"], "sequences": sequences, "stages": [], "encoder": encoder.provenance,
        "approximate_training_accepted": False, "approximate_candidate_export_accepted": False,
        "torch": str(torch.__version__), "threads": 1, "strict_historical_status": protocol["strict_status"]}
    for sequence in iter_sequences(protocol["train_path"], protocol["groups"]):
        if sequence.seq_ix != protocol["expected_sequence_ids"][str(sequence.row_group)] or sequence.mask is not None:
            raise ValueError("unexpected TRAIN sequence or validation mask")
        reference = encoder(sequence.features)
        teacher_prediction = reference[2] if teacher is encoder else teacher(sequence.features)[2]
        item = {"row_group": sequence.row_group, "seq_ix": sequence.seq_ix, "rows": len(sequence.features),
            "equations": old.fp64_equation_spotcheck(model, sequence.features, reference)}
        sequences.append(item)
        if item["equations"]["passed"]:
            paths, item["exact_forward"] = forward.local_paths(model, reference)
            item["branches"] = {}
            for name, values in paths.items():
                limits = forward.BUDGET["common_state" if name == "common_state" else "carried_state"]
                metrics = {key: old.error_metrics(a, b, limits) for key, a, b in zip(("h0", "h1", "prediction"), values, reference)}
                item["branches"][name] = {"outputs": metrics, "wp": old.wp_comparison(sequence, values[2], reference[2])
                    if all(m["finite"] for m in metrics.values()) else {"eligible": False, "drift": None, "passed": False}}
            objective_result = objective.correlation_objective(torch.from_numpy(reference[2])[:, None],
                torch.from_numpy(teacher_prediction)[:, None], torch.from_numpy(sequence.targets)[:, None])
            item["gradients"] = [objective.gradient_probe(model, reference, teacher_prediction, sequence.targets,
                                     start=start, objective_result=objective_result)
                                 for start in procedure()["gradient_chunk_starts"]]
        # Only generic coverage/local-error aggregation is reused; every VJP
        # above was recomputed using the new objective's full-sequence residual.
        report.update(summary=forward._summary(sequences), completed_sequences=len(sequences), seconds=time.monotonic() - started)
        local_ok = (item["equations"]["passed"] and item.get("exact_forward", False) and
            len(item.get("gradients", [])) == 2 and all(g["passed"] for g in item["gradients"]) and
            all(all(m["approximate_passed"] for m in b["outputs"].values()) and b["wp"].get("passed", True)
                for b in item.get("branches", {}).values()))
        if len(sequences) in protocol["stages"]:
            report["stages"].append({"sequences": len(sequences), "summary": report["summary"]})
        if not local_ok or len(sequences) in protocol["stages"] and not report["summary"]["approximate_passed"]:
            report["stop_reason"] = "Correlation-specific gradient/local audit failed; no automatic threshold changes"
            old._write(output / "report.json", report)
            return report
        old._write(output / "report.json", report)
        print(f"Correlation-gradient TRAIN {len(sequences)}/{len(protocol['groups'])}: passed", flush=True)
    accepted = len(sequences) == len(protocol["groups"]) and report["summary"]["approximate_passed"]
    report["approximate_training_accepted"] = accepted and protocol["mode"] == "initial_source"
    report["approximate_candidate_export_accepted"] = accepted and protocol["mode"] == "candidate_export"
    old._write(output / "report.json", report)
    return report


def _require(report_path, model_path, mode):
    report = old._read(report_path)
    protocol = _verify_protocol(report["protocol_path"])
    count, flag = (8, "approximate_training_accepted") if mode == "initial_source" else (2, "approximate_candidate_export_accepted")
    if (report["mode"] != mode or protocol["mode"] != mode or report["procedure"] != procedure() or
            report["budget"] != forward.BUDGET or report["protocol_sha256"] != old._sha(report["protocol_path"]) or
            report["source_sha256"] != old._sha(model_path) or report["zero_update_sha256"] != old._sha(model_path) or
            Path(model_path).resolve() != Path(protocol["model_path"]).resolve() or not report[flag] or
            report["completed_sequences"] != count or len(report["sequences"]) != count or
            [s["row_group"] for s in report["sequences"]] != protocol["groups"] or not forward._summary(report["sequences"])["approximate_passed"]):
        raise ValueError("complete accepted correlation-gradient bridge required for this exact model/mode")
    for s in report["sequences"]:
        if (s["rows"] != 20000 or s["seq_ix"] != protocol["expected_sequence_ids"][str(s["row_group"])] or
                set(s["branches"]) != set(forward.BRANCHES) or [g["start"] for g in s["gradients"]] != [0, 19744]):
            raise ValueError("incomplete correlation-gradient coverage")
        for g in s["gradients"]:
            if (g["rows"] != 256 or g["objective_report"]["objective"] != objective.objective_parameters() or
                    set(g["gradient_tensors"]) != set(forward.GRADIENT_NAMES) or any(
                        not v["finite"] or v["values"] != forward.GRADIENT_SIZES[key] or v["violations"] or
                        not 0 <= v["max_tolerance_ratio"] <= 1. for key, v in g["gradient_tensors"].items())):
                raise ValueError("missing correlation-specific gradient evidence")
        for name, branch in s["branches"].items():
            limits = forward.BUDGET["common_state" if name == "common_state" else "carried_state"]
            if set(branch["outputs"]) != {"h0", "h1", "prediction"} or any(
                    not v["finite"] or v["values"] != 20000 * (2 if key == "prediction" else 128) or
                    not 0 <= v["normalized_max"] <= limits["normalized_max"] or
                    not 0 <= v["normalized_rms"] <= limits["normalized_rms"] for key, v in branch["outputs"].items()):
                raise ValueError("missing local numerical evidence")
    return report


def require_bridge_acceptance(report_path, source_path):
    return _require(report_path, source_path, "initial_source")


def require_candidate_acceptance(report_path, candidate_model_path):
    return _require(report_path, candidate_model_path, "candidate_export")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--output", required=True)
    sub.add_parser("run").add_argument("--protocol", required=True)
    candidate = sub.add_parser("prepare-candidate")
    for argument in ("model", "initial-bridge", "output"):
        candidate.add_argument(f"--{argument}", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_protocol(args.output)); return 0
    if args.command == "prepare-candidate":
        print(prepare_candidate_protocol(args.model, args.initial_bridge, args.output)); return 0
    report = run_diagnostics(args.protocol)
    print(json.dumps({"training_accepted": report["approximate_training_accepted"],
                      "candidate_accepted": report["approximate_candidate_export_accepted"], "summary": report["summary"]}, indent=2))
    return 0 if report["approximate_training_accepted"] or report["approximate_candidate_export_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
