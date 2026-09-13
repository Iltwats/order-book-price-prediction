"""ORT forward history with bounded Torch local gradients: a separate bridge.

The failed conventional FP32 bridge stays failed. This procedure validates
only the training path whose forward trajectory is the current student's ORT
trajectory and whose gradient is truncated into ORT-reanchored 256-row chunks.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from . import numerical_bridge as old
from .conversion_audit import export_original_graph
from .training import OnnxSequenceEncoder, load_torch_model

ROOT = old.ROOT
BUDGET = copy.deepcopy(old.BUDGET)
PROCEDURE = {
    "version": "ort_forward_local_gradient.v1", "chunk_size": 256,
    "gradient_chunk_starts": [0, 19744],
    "teacher_weight": .1,
    "forward": "Current-student original-graph ORT carries all 20000 rows; lower GRU remains frozen",
    "gradient": {"atol": 1e-6, "rtol": 1e-4},
    "backward": "Torch upper GRU starts each 256-row chunk from detached current-student ORT prior h1",
    "state_proxy": "ort_u + (torch_u - torch_u.detach())",
    "prediction_proxy": "ort_p + (head(state_proxy) - head(state_proxy).detach())",
    "gradient_loss": "sequence-target mean clipped-label weighted MSE plus .1 raw selected-ORT-teacher weighted MSE",
    "gradient_residual": "FP64 analytic loss coefficients computed at FP32 ORT outputs, rounded once to FP32 then shared exactly across precisions",
    "limits": "Unchanged old common-state limits; unchanged carried-state limits applied within reanchored chunks",
    "design_basis": "New path motivated by the already-inspected failed TRAIN group2309; not an independent statistical holdout",
    "scope": "Local surrogate-gradient approximation, not exact ORT derivatives or a strict conventional-parity pass",
}
BRANCHES = ("common_state", "reanchored256")
GRADIENT_NAMES = ("gru1.weight_ih_l0", "gru1.weight_hh_l0", "gru1.bias_ih_l0", "gru1.bias_hh_l0", "head.weight", "head.bias")
GRADIENT_SIZES = dict(zip(GRADIENT_NAMES, (49152, 49152, 384, 384, 256, 2)))


def exact_forward_proxy(model, lower, initial, ort_upper, ort_prediction):
    """Return exact ORT forward values with a Torch local surrogate derivative."""
    from .partial_gru import matched_forward_prediction
    upper, _ = model.gru1(lower.detach(), initial.detach())
    proxy_upper = ort_upper.detach() + (upper - upper.detach())
    proxy_prediction = matched_forward_prediction(model.head, upper, ort_upper, ort_prediction)
    return proxy_upper, proxy_prediction, upper


def loss_coefficients(targets, ort_prediction, teacher, *, start, rows):
    """Fixed upstream derivative of the declared weighted MSE at ORT outputs.

    A scalar linear surrogate with these coefficients has the exact same local
    parameter gradient as the weighted MSE at the specified forward values.
    Warm-up and zero-target rows have zero derivative; target means include
    both sides, even when one side has zero total weight.
    """
    y, p, t = (np.asarray(value, dtype=np.float64) for value in (targets, ort_prediction, teacher))
    if y.ndim != 2 or y.shape[1] != 2 or p.shape != y.shape or t.shape != y.shape:
        raise ValueError("matching full-sequence [rows,2] labels/student/teacher required")
    if start < 0 or rows < 1 or start + rows > len(y) or not all(np.isfinite(v).all() for v in (y, p, t)):
        raise ValueError("invalid or nonfinite gradient probe inputs")
    clipped = np.clip(y, -2., 2.)
    weights = np.abs(clipped)
    weights[:99] = 0.
    denominator = np.maximum(weights.sum(0), 1e-12)
    # derivative 2 * residual, averaged over two target sides => factor one.
    coefficient = weights / denominator * ((p - clipped) + PROCEDURE["teacher_weight"] * (p - t))
    return coefficient[start:start + rows].astype(np.float32)


def gradient_probe(model, reference, teacher, targets, *, start):
    rows = min(PROCEDURE["chunk_size"], len(targets) - start)
    coefficient = loss_coefficients(targets, reference[2], teacher, start=start, rows=rows)
    results, forwards = {}, {}
    for dtype in (torch.float32, torch.float64):
        local = copy.deepcopy(model).to(dtype=dtype)
        for parameter in local.gru0.parameters():
            parameter.requires_grad_(False)
        lower = torch.from_numpy(reference[0][start:start + rows]).to(dtype=dtype)[:, None]
        ort_upper = torch.from_numpy(reference[1][start:start + rows]).to(dtype=dtype)[:, None]
        ort_prediction = torch.from_numpy(reference[2][start:start + rows]).to(dtype=dtype)[:, None]
        initial = (torch.zeros(1, 1, model.hidden_size, dtype=dtype) if start == 0 else
                   torch.from_numpy(reference[1][start - 1]).to(dtype=dtype)[None, None])
        proxy_upper, proxy_prediction, _ = exact_forward_proxy(local, lower, initial, ort_upper, ort_prediction)
        forwards[str(dtype)] = bool(torch.equal(proxy_upper, ort_upper) and torch.equal(proxy_prediction, ort_prediction)
                                  and torch.isfinite(proxy_upper).all() and torch.isfinite(proxy_prediction).all())
        scalar = (proxy_prediction * torch.from_numpy(coefficient).to(dtype=dtype)[:, None]).sum()
        parameters = dict(local.named_parameters())
        gradients = torch.autograd.grad(scalar, [parameters[name] for name in GRADIENT_NAMES])
        results[str(dtype)] = {name: value.detach().numpy().astype(np.float64) for name, value in zip(GRADIENT_NAMES, gradients)}
    comparisons = {}
    for name in GRADIENT_NAMES:
        a, b = results["torch.float32"][name], results["torch.float64"][name]
        error = np.abs(a - b)
        tolerance = PROCEDURE["gradient"]["atol"] + PROCEDURE["gradient"]["rtol"] * np.abs(b)
        finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
        comparisons[name] = {"values": int(a.size), "finite": finite,
            "max_absolute_error": float(error.max()) if finite else None,
            "rms_error": float(np.sqrt(np.mean(error ** 2))) if finite else None,
            "reference_max": float(np.abs(b).max()) if finite else None,
            "max_tolerance_ratio": float((error / tolerance).max()) if finite else None,
            "violations": int(np.count_nonzero(error > tolerance)) if finite else int(a.size),
            "passed": finite and bool(np.all(error <= tolerance))}
    return {"start": start, "rows": rows, "identical_forward_values": forwards,
            "gradient_tensors": comparisons, "coefficient_sha256": hashlib.sha256(coefficient.tobytes()).hexdigest(),
            "passed": all(forwards.values()) and all(value["passed"] for value in comparisons.values())}


def local_paths(model, reference):
    """Uncorrected local arithmetic is checked even though training corrects it."""
    h0, h1, prediction = reference
    common, chunked = np.empty_like(h1), np.empty_like(h1)
    previous = np.concatenate((np.zeros_like(h1[:1]), h1[:-1]))
    exact = True
    with torch.inference_mode():
        for start in range(0, len(h0), 256):
            end = min(len(h0), start + 256)
            input_rows = torch.from_numpy(h0[start:end])
            initial = torch.from_numpy(previous[start])[None, None]
            common[start:end] = model.gru1(input_rows[None], torch.from_numpy(previous[start:end])[None])[0][0].numpy()
            ort_u, ort_p = torch.from_numpy(h1[start:end])[:, None], torch.from_numpy(prediction[start:end])[:, None]
            proxy_u, proxy_p, actual = exact_forward_proxy(model, input_rows[:, None], initial, ort_u, ort_p)
            exact = exact and bool(torch.equal(proxy_u, ort_u) and torch.equal(proxy_p, ort_p))
            chunked[start:end] = actual[:, 0].numpy()
        return {"common_state": (h0, common, model.head(torch.from_numpy(common)).numpy()),
                "reanchored256": (h0, chunked, model.head(torch.from_numpy(chunked)).numpy())}, exact


def prepare_protocol(output_dir, *, model_path=ROOT / "artifacts/selected/model.onnx",
                     manifest_path=ROOT / "artifacts/manifest.json",
                     anchor_path=ROOT / "artifacts/training_pilot1024/affine_p1e-2/report.json",
                     conversion_dir=ROOT / "artifacts/round2/conversion", initial_bridge_report=None):
    """Write new procedure and unchanged limits before any Parquet access."""
    model_path, manifest_path, anchor_path = map(Path, (model_path, manifest_path, anchor_path))
    teacher = model_path
    if initial_bridge_report is not None:
        prior = old._read(initial_bridge_report)
        teacher = Path(old._read(prior["protocol_path"])["model_path"])
        require_bridge_acceptance(initial_bridge_report, teacher)
    evidence = old._check_prerequisites(teacher, conversion_dir)
    manifest, anchor = old._read(manifest_path), old._read(anchor_path)
    train, groups = manifest["datasets"]["train"], anchor["specification"]["groups"]
    if (len(groups) != 1024 or len(set(groups)) != 1024 or not set(groups) <= set(manifest["splits"]["train"]) or
            Path(anchor["specification"]["data"]["path"]).resolve() != Path(train["path"]).resolve()):
        raise ValueError("frozen 1024-group TRAIN anchor required")
    files = [Path(__file__), Path(old.__file__), ROOT / "connectome/training.py", ROOT / "connectome/model_io.py",
             ROOT / "connectome/partial_gru.py",
             ROOT / "connectome/conversion_audit.py", ROOT / "connectome/data.py", ROOT / "connectome/evaluation.py",
             ROOT / "utils.py", model_path, teacher, manifest_path, anchor_path, *evidence,
             ROOT / "artifacts/round2/recurrent_bridge/report.json"]
    if initial_bridge_report is not None:
        files.extend((Path(initial_bridge_report), Path(prior["protocol_path"])))
    count = 2 if initial_bridge_report is not None else 8
    protocol = {"procedure": copy.deepcopy(PROCEDURE), "budget": copy.deepcopy(BUDGET),
        "model_path": str(model_path.resolve()), "teacher_path": str(teacher.resolve()),
        "train_path": train["path"], "train_fingerprint": train["fingerprint"], "groups": groups[:count],
        "stages": [2] if count == 2 else [2, 8], "mode": "candidate_export" if count == 2 else "initial_source",
        "expected_sequence_ids": {str(g["row_group"]): g["seq_ix"] for g in train["groups"] if g["row_group"] in groups[:count]},
        "artifacts": {str(p.resolve()): old._sha(p) for p in files},
        "strict_status": "Both historical strict and conventional approximate carried-state bridges remain failed",
        "scope": "Fixed TRAIN-only ORT-forward/local-gradient procedure; no optimizer or deployment approval"}
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    old._write(output / "protocol.json", protocol, exclusive=True)
    return str(output / "protocol.json")


def prepare_candidate_protocol(model_path, initial_bridge_report, output_dir, **kwargs):
    return prepare_protocol(output_dir, model_path=model_path, initial_bridge_report=initial_bridge_report, **kwargs)


def _verify_protocol(path):
    protocol = old._read(path)
    stages = [2, 8] if protocol["mode"] == "initial_source" else [2]
    if (protocol["mode"] not in {"initial_source", "candidate_export"} or protocol["procedure"] != PROCEDURE or
            protocol["budget"] != BUDGET or protocol["stages"] != stages or len(protocol["groups"]) != stages[-1]):
        raise ValueError("protocol differs from frozen ORT-forward procedure or numerical limits")
    for path, digest in protocol["artifacts"].items():
        if old._sha(path) != digest:
            raise ValueError(f"preregistered artifact changed: {path}")
    return protocol


def _summary(sequences):
    branches = {}
    for name in BRANCHES:
        entries = [s["branches"][name] for s in sequences if "branches" in s]
        drifts = [e["wp"]["drift"] for e in entries if e["wp"]["eligible"]]
        means = np.mean(drifts, axis=0).tolist() if drifts else None
        branches[name] = {"mean_target_wp_drift": means,
            "approximate_passed": len(entries) == len(sequences) and bool(drifts) and
                all(m["approximate_passed"] for e in entries for m in e["outputs"].values()) and
                all(e["wp"].get("passed", True) for e in entries) and max(map(abs, means)) <= BUDGET["wp"]["max_mean_target_drift"],
            "strict_passed": bool(entries) and all(m["strict_passed"] for e in entries for m in e["outputs"].values())}
    return {"branches": branches, "approximate_passed": bool(sequences) and
            all(s["equations"]["passed"] and s.get("exact_forward", False) and
                len(s.get("gradients", [])) == 2 and all(g["passed"] for g in s["gradients"]) for s in sequences) and
            all(b["approximate_passed"] for b in branches.values()),
            "strict_passed": bool(sequences) and all(b["strict_passed"] for b in branches.values())}


def run_diagnostics(protocol_path):
    from .data import file_fingerprint, iter_sequences
    from .model_io import load_weights
    protocol_path = Path(protocol_path).resolve()
    protocol, output = _verify_protocol(protocol_path), protocol_path.parent
    if (output / "report.json").exists():
        raise FileExistsError("preserve existing bridge result")
    if file_fingerprint(protocol["train_path"]) != protocol["train_fingerprint"]:
        raise ValueError("training dataset fingerprint changed")
    model = load_torch_model(protocol["model_path"])
    source_weights, teacher_weights = load_weights(protocol["model_path"]), load_weights(protocol["teacher_path"])
    if any(source_weights[k].tobytes() != teacher_weights[k].tobytes() for k in ("w0", "r0", "b0")):
        raise ValueError("ORT-forward partial-training procedure requires unchanged lower GRU")
    exported = export_original_graph(model, protocol["model_path"], output / "zero_update.onnx")
    if old._sha(exported) != old._sha(protocol["model_path"]):
        raise ValueError("actual candidate weights do not survive byte-identical roundtrip")
    torch.set_num_threads(1)
    encoder = OnnxSequenceEncoder(exported, output / "encoder.onnx")
    teacher = encoder if old._sha(protocol["model_path"]) == old._sha(protocol["teacher_path"]) else OnnxSequenceEncoder(protocol["teacher_path"], output / "teacher.onnx")
    started, sequences = time.monotonic(), []
    report = {"protocol_path": str(protocol_path), "protocol_sha256": old._sha(protocol_path),
        "source_sha256": old._sha(protocol["model_path"]), "teacher_sha256": old._sha(protocol["teacher_path"]),
        "zero_update_sha256": old._sha(exported), "budget": protocol["budget"], "procedure": protocol["procedure"],
        "mode": protocol["mode"], "sequences": sequences, "encoder": encoder.provenance,
        "approximate_training_accepted": False, "approximate_candidate_export_accepted": False,
        "strict_historical_status": protocol["strict_status"], "torch": str(torch.__version__), "threads": 1, "stages": []}
    for sequence in iter_sequences(protocol["train_path"], protocol["groups"]):
        if sequence.seq_ix != protocol["expected_sequence_ids"][str(sequence.row_group)] or sequence.mask is not None:
            raise ValueError("sequence differs from frozen TRAIN identity/schema")
        reference = encoder(sequence.features)
        teacher_prediction = reference[2] if teacher is encoder else teacher(sequence.features)[2]
        item = {"row_group": sequence.row_group, "seq_ix": sequence.seq_ix, "rows": len(sequence.features),
            "equations": old.fp64_equation_spotcheck(model, sequence.features, reference),
            "ort_student_domain": old.domain_statistics(sequence, reference[2])}
        sequences.append(item)
        if item["equations"]["passed"]:
            paths, item["exact_forward"] = local_paths(model, reference)
            item["branches"] = {}
            for name, values in paths.items():
                limits = BUDGET["common_state" if name == "common_state" else "carried_state"]
                metrics = {key: old.error_metrics(a, b, limits) for key, a, b in zip(("h0", "h1", "prediction"), values, reference)}
                item["branches"][name] = {"outputs": metrics,
                    "wp": old.wp_comparison(sequence, values[2], reference[2]) if all(m["finite"] for m in metrics.values()) else
                          {"eligible": False, "drift": None, "passed": False}}
            item["gradients"] = [gradient_probe(model, reference, teacher_prediction, sequence.targets, start=start)
                                 for start in PROCEDURE["gradient_chunk_starts"]]
        report.update(summary=_summary(sequences), completed_sequences=len(sequences), seconds=time.monotonic() - started)
        local_ok = item["equations"]["passed"] and item.get("exact_forward", False)
        # Continuation is kept explicit to avoid treating an unmeasured gradient as passed.
        local_ok = (local_ok and len(item.get("gradients", [])) == 2 and all(g["passed"] for g in item["gradients"]) and
                    all(all(m["approximate_passed"] for m in b["outputs"].values()) and b["wp"].get("passed", True)
                        for b in item.get("branches", {}).values()))
        if len(sequences) in protocol["stages"]:
            report["stages"].append({"sequences": len(sequences), "summary": report["summary"]})
        if not local_ok or len(sequences) in protocol["stages"] and not report["summary"]["approximate_passed"]:
            report["stop_reason"] = "New local numerical/gradient procedure failed; no automatic tolerance change"
            old._write(output / "report.json", report)
            return report
        old._write(output / "report.json", report)
        print(f"ORT-forward bridge TRAIN {len(sequences)}/{len(protocol['groups'])}: local/gradient checks passed", flush=True)
    accepted = len(sequences) == len(protocol["groups"]) and report["summary"]["approximate_passed"]
    report["approximate_training_accepted"] = accepted and protocol["mode"] == "initial_source"
    report["approximate_candidate_export_accepted"] = accepted and protocol["mode"] == "candidate_export"
    old._write(output / "report.json", report)
    return report


def _require(report_path, model_path, mode):
    report = old._read(report_path)
    protocol = _verify_protocol(report["protocol_path"])
    count, flag = (8, "approximate_training_accepted") if mode == "initial_source" else (2, "approximate_candidate_export_accepted")
    if (report["mode"] != mode or protocol["mode"] != mode or report["procedure"] != PROCEDURE or report["budget"] != BUDGET or
            report["protocol_sha256"] != old._sha(report["protocol_path"]) or
            report["source_sha256"] != old._sha(model_path) or report["zero_update_sha256"] != old._sha(model_path) or
            Path(model_path).resolve() != Path(protocol["model_path"]).resolve() or not report[flag] or
            report["completed_sequences"] != count or len(report["sequences"]) != count or
            [s["row_group"] for s in report["sequences"]] != protocol["groups"] or not _summary(report["sequences"])["approximate_passed"]):
        raise ValueError("complete accepted ORT-forward bridge required for this exact model/mode")
    for s in report["sequences"]:
        if (s["rows"] != 20000 or s["seq_ix"] != protocol["expected_sequence_ids"][str(s["row_group"])] or
                set(s["branches"]) != set(BRANCHES) or [g["start"] for g in s["gradients"]] != PROCEDURE["gradient_chunk_starts"]):
            raise ValueError("incomplete sequence/local-gradient coverage")
        for gradient in s["gradients"]:
            if gradient["rows"] != 256 or set(gradient["gradient_tensors"]) != set(GRADIENT_NAMES) or any(
                    not g["finite"] or g["values"] != GRADIENT_SIZES[name] or g["violations"] or
                    not 0 <= g["max_tolerance_ratio"] <= 1. for name, g in gradient["gradient_tensors"].items()):
                raise ValueError("missing or failed current gradient evidence")
        for name, branch in s["branches"].items():
            limits = BUDGET["common_state" if name == "common_state" else "carried_state"]
            if set(branch["outputs"]) != {"h0", "h1", "prediction"} or any(
                    not m["finite"] or m["values"] != 20000 * (2 if key == "prediction" else 128) or
                    not 0 <= m["normalized_max"] <= limits["normalized_max"] or
                    not 0 <= m["normalized_rms"] <= limits["normalized_rms"] for key, m in branch["outputs"].items()):
                raise ValueError("missing or failed local arithmetic evidence")
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
