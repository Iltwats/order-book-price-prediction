"""Prospective TRAIN-only numerical diagnostics, never a strict parity override.

Prepare records the engineering budget and all inputs before any dataset read.
Run stops on failed prerequisites, equations, or the first failed sequence.
An accepted bridge permits investigating approximate Torch training only;
each changed model still needs original-graph export and deployment validation.
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

from .conversion_audit import export_original_graph, reference_gru
from .training import OnnxSequenceEncoder, load_torch_model

ROOT = Path(__file__).resolve().parents[1]
EPSILON = float(np.finfo(np.float32).eps)
BUDGET = {
    "version": "approximate_training_bridge.v1",
    "strict": {"atol": 1e-6, "rtol": 1e-5},
    "common_state": {"normalized_max": 128 * EPSILON, "normalized_rms": 8 * EPSILON},
    "carried_state": {"normalized_max": 512 * EPSILON, "normalized_rms": 32 * EPSILON},
    "wp": {"max_sequence_target_drift": 2e-5, "max_mean_target_drift": 2e-6},
    "fp64_equation": {"atol": 1e-12, "rtol": 1e-10},
    "normalization": "per coordinate max(1, reference RMS); RMS bound applies to every coordinate",
    "rationale": (
        "Prospective engineering limits informed by the existing roughly 1e-5 FP32 discrepancy; "
        "not formal roundoff bounds or a claim that strict parity passes. Common-state limits are "
        "tighter because recurrence cannot accumulate error. State scale stays one for bounded "
        "GRU states; the RMS floor prevents unstable relative errors near zero. WP limits constrain "
        "task-level drift, per sequence as well as per-target means to prevent cancellation. "
        "These diagnostics use training data only and do not establish generalization or release fitness."
    ),
}
BRANCHES = ("full_torch", "common_state", "upper_on_ort_lower")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _write(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def error_metrics(actual, reference, limits):
    """Per-coordinate scaling with independent historical strict diagnostics."""
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape or actual.ndim != 2 or actual.size == 0:
        raise ValueError("nonempty matching [rows, coordinates] arrays required")
    if not np.isfinite(actual).all() or not np.isfinite(reference).all():
        return {"finite": False, "approximate_passed": False, "strict_passed": False}
    expected = reference.astype(np.float64)
    error = np.abs(actual.astype(np.float64) - expected)
    scale = np.maximum(1., np.sqrt(np.mean(expected ** 2, axis=0)))
    normalized = error / scale
    normalized_max = float(normalized.max())
    normalized_rms = float(np.sqrt(np.mean(normalized ** 2, axis=0)).max())
    tolerance = BUDGET["strict"]["atol"] + BUDGET["strict"]["rtol"] * np.abs(expected)
    violations = int(np.count_nonzero(error > tolerance))
    return {"finite": True, "values": int(error.size),
            "max_absolute_error": float(error.max()), "rms_error": float(np.sqrt(np.mean(error ** 2))),
            "normalized_max": normalized_max, "normalized_rms": normalized_rms,
            "reference_scale_min": float(scale.min()), "reference_scale_max": float(scale.max()),
            "strict_violations": violations, "strict_max_tolerance_ratio": float((error / tolerance).max()),
            "strict_passed": violations == 0,
            "approximate_passed": normalized_max <= limits["normalized_max"] and
                                  normalized_rms <= limits["normalized_rms"]}


def _check_prerequisites(model_path, conversion_dir):
    conversion_dir = Path(conversion_dir)
    report_path = conversion_dir / "selected/report.json"
    gradient_path = conversion_dir / "fp64_gradients.json"
    report, gradients = _read(report_path), _read(gradient_path)
    if (report["source_sha256"] != _sha(model_path) or not report["passed"] or
            not report["artifact_bytes_identical"] or not report["structure_identical"] or
            report["sequence_count"] != 32 or not all(p["bitwise_equal"] for p in report["parameters"].values())):
        raise ValueError("selected model requires its complete byte-identical conversion evidence")
    if (not gradients["passed"] or len(gradients["cases"]) != 3 or
            not all(c["passed"] and all(c["finite_difference"].values()) for c in gradients["cases"])):
        raise ValueError("equation/gradient prerequisite failed; stop, do not change tolerance")
    strict = report["strict_fp32_audit"]
    if _sha(strict["path"]) != strict["sha256"]:
        raise ValueError("historical strict audit changed")
    from . import conversion_audit
    if gradients["audit_code_sha256"] != _sha(conversion_audit.__file__):
        raise ValueError("gradient audit does not match current conversion code")
    return [report_path, gradient_path, Path(strict["path"])]


def prepare_protocol(output_dir, *, model_path=ROOT / "artifacts/selected/model.onnx",
                     manifest_path=ROOT / "artifacts/manifest.json",
                     anchor_path=ROOT / "artifacts/training_pilot1024/affine_p1e-2/report.json",
                     conversion_dir=ROOT / "artifacts/round2/conversion", initial_bridge_report=None):
    """Freeze constants, eight TRAIN groups and artifacts without opening Parquet."""
    model_path, manifest_path, anchor_path = map(Path, (model_path, manifest_path, anchor_path))
    base_model = model_path
    if initial_bridge_report is not None:
        previous = _read(initial_bridge_report)
        base_model = Path(_read(previous["protocol_path"])["model_path"])
        require_bridge_acceptance(initial_bridge_report, base_model)
    evidence = _check_prerequisites(base_model, conversion_dir)
    manifest, anchor = _read(manifest_path), _read(anchor_path)
    train = manifest["datasets"]["train"]
    groups = anchor["specification"]["groups"]
    if len(groups) != 1024 or len(set(groups)) != 1024 or not set(groups) <= set(manifest["splits"]["train"]):
        raise ValueError("expected the frozen 1024-group training anchor")
    if Path(anchor["specification"]["data"]["path"]).resolve() != Path(train["path"]).resolve():
        raise ValueError("training anchor and manifest disagree")
    from . import training, model_io, data
    sources = [Path(__file__), Path(training.__file__), Path(model_io.__file__), Path(data.__file__),
               ROOT / "connectome/conversion_audit.py", ROOT / "connectome/evaluation.py", ROOT / "utils.py",
               model_path, base_model, manifest_path, anchor_path, *evidence]
    if initial_bridge_report is not None:
        sources.extend((Path(initial_bridge_report), Path(previous["protocol_path"])))
    count = 2 if initial_bridge_report is not None else 8
    protocol = {"budget": copy.deepcopy(BUDGET), "model_path": str(model_path.resolve()),
                "train_path": str(Path(train["path"]).resolve()), "train_fingerprint": train["fingerprint"],
                "groups": groups[:count], "stages": [2] if count == 2 else [2, 8], "common_state_batch_size": 256,
                "mode": "candidate_export" if count == 2 else "initial_source",
                "initial_bridge_report": str(Path(initial_bridge_report).resolve()) if count == 2 else None,
                "expected_sequence_ids": {str(g["row_group"]): g["seq_ix"] for g in train["groups"]
                                          if g["row_group"] in groups[:count]},
                "artifacts": {str(p.resolve()): _sha(p) for p in sources},
                "scope": "TRAIN-only numerical diagnostics; no optimizer or validation/confirmation data reads",
                "strict_status": "historical failed strict FP32 gate is unchanged and remains separate",
                "fp64_spotcheck": "64 evenly spaced rows per sequence; actual GRU128 weights, common ORT states",
                "completion": f"{count} sequences required; stop on first failure; strict parity remains separate"}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocol.json"
    _write(path, protocol, exclusive=True)
    return str(path)


def _verify_protocol(path):
    protocol = _read(path)
    stages = [2, 8] if protocol["mode"] == "initial_source" else [2]
    if (protocol["mode"] not in {"initial_source", "candidate_export"} or protocol["budget"] != BUDGET or
            protocol["stages"] != stages or len(protocol["groups"]) != stages[-1]):
        raise ValueError("protocol differs from the frozen acceptance procedure")
    for artifact, expected in protocol["artifacts"].items():
        if _sha(artifact) != expected:
            raise ValueError(f"preregistered artifact changed: {artifact}")
    return protocol


def prepare_candidate_protocol(model_path, initial_bridge_report, output_dir, **kwargs):
    """Two fixed TRAIN groups on changed weights; requires accepted source bridge.

    Reconstructing an exported candidate checks deployment/training arithmetic.
    The trainer must separately verify its actual pre-export parameter arrays.
    This convenience wrapper performs no inference and reads no dataset rows.
    """
    return prepare_protocol(output_dir, model_path=model_path,
                            initial_bridge_report=initial_bridge_report, **kwargs)


def torch_paths(model, features, reference, *, batch_size=256):
    """Compare whole sequence, independent batched steps, and planned upper-only path.

    Common-state steps are independent rows packed as a batch, not a recurrent
    batch sequence. Both prior states are always the original ORT states. The
    upper-only path carries Torch upper state but consumes original ORT h0.
    """
    x = torch.from_numpy(features)
    lower, upper, _ = reference
    with torch.inference_mode():
        full_lower, full_upper, _, _ = model.encode(x[:, None])
        full_prediction = model.head(full_upper)
        hybrid_upper, _ = model.gru1(torch.from_numpy(lower)[:, None])
        hybrid_prediction = model.head(hybrid_upper)
        common = [np.empty_like(value) for value in reference]
        previous = [np.concatenate((np.zeros_like(s[:1]), s[:-1]), axis=0) for s in (lower, upper)]
        for start in range(0, len(x), batch_size):
            end = min(len(x), start + batch_size)
            states = [torch.from_numpy(s[start:end])[None] for s in previous]
            h0, h1, _, _ = model.encode(x[start:end][None], *states)
            for destination, value in zip(common, (h0[0], h1[0], model.head(h1)[0])):
                destination[start:end] = value.numpy()
        unpack = lambda values: tuple(value[:, 0].numpy() for value in values)
        return {"full_torch": unpack((full_lower, full_upper, full_prediction)),
                "common_state": tuple(common),
                "upper_on_ort_lower": (lower, *unpack((hybrid_upper, hybrid_prediction)))}


def fp64_equation_spotcheck(model, features, reference):
    """Actual full-width parameters, independent equations at nonzero ORT states."""
    indices = np.linspace(0, len(features) - 1, min(64, len(features)), dtype=int)
    double = copy.deepcopy(model).double()
    x = torch.from_numpy(features[indices]).double()[None]
    previous = [np.concatenate((np.zeros_like(s[:1]), s[:-1]), axis=0) for s in reference[:2]]
    states = [torch.from_numpy(s[indices]).double()[None] for s in previous]
    with torch.inference_mode():
        actual0, actual1, _, _ = double.encode(x, *states)
        expected0, _ = reference_gru(x, states[0], *double.gru0.parameters())
        expected1, _ = reference_gru(expected0, states[1], *double.gru1.parameters())
        comparisons = {}
        for name, actual, expected in zip(("h0", "h1", "prediction"),
                (actual0, actual1, double.head(actual1)), (expected0, expected1, double.head(expected1))):
            a, b = actual.numpy(), expected.numpy()
            comparisons[name] = {"max_absolute_error": float(np.abs(a - b).max()),
                                 "passed": bool(np.allclose(a, b, **BUDGET["fp64_equation"]))}
    return {"rows": indices.tolist(), "outputs": comparisons,
            "passed": all(item["passed"] for item in comparisons.values())}


def wp_comparison(sequence, prediction, reference):
    from utils import BlockAccumulator
    if sequence.mask is not None:
        raise ValueError("numerical bridge requires TRAIN schema without a validation mask")
    actual, expected = BlockAccumulator(), BlockAccumulator()
    actual.add(sequence.targets, prediction, sequence.need)
    expected.add(sequence.targets, reference, sequence.need)
    if not actual.scores:
        return {"eligible": False, "drift": None}
    a, b = actual.result(), expected.result()
    drift = [a[name] - b[name] for name in ("t0", "t1")]
    return {"eligible": True, "torch": a, "ort": b, "drift": drift,
            "passed": max(map(abs, drift)) <= BUDGET["wp"]["max_sequence_target_drift"]}


def domain_statistics(sequence, prediction):
    """TRAIN-only target/teacher scale and prediction-clipping effect, no fitting."""
    if sequence.mask is not None:
        raise ValueError("TRAIN domain statistics must not use a validation mask")
    from utils import weighted_pearson
    target = sequence.targets[sequence.need].astype(np.float64)
    prediction = np.asarray(prediction)[sequence.need].astype(np.float64)
    clipped_target, clipped_prediction = np.clip(target, -2., 2.), np.clip(prediction, -2., 2.)
    eligible = all(np.any(target[:, side] > 0) and np.any(target[:, side] < 0) for side in range(2))
    outputs = {}
    for side in range(2):
        t, p = clipped_target[:, side], prediction[:, side]
        weight = np.abs(t)
        total = max(float(weight.sum()), 1e-12)
        centered_t, centered_p = t - np.sum(t * weight) / total, p - np.sum(p * weight) / total
        std_t, std_p = np.sqrt(np.sum(weight * centered_t ** 2) / total), np.sqrt(np.sum(weight * centered_p ** 2) / total)
        raw_wp = (float(np.clip(np.sum(weight * centered_t * centered_p) / total / (std_t * std_p), -1., 1.))
                  if weight.sum() >= 1e-8 and min(std_t, std_p) > 1e-8 else 0.)
        outputs[f"t{side}"] = {
            "target_clip_fraction": float(np.mean(np.abs(target[:, side]) > 2.)),
            "prediction_clip_fraction": float(np.mean(np.abs(p) > 2.)),
            "target_rms": float(np.sqrt(np.mean(target[:, side] ** 2))),
            "clipped_target_rms": float(np.sqrt(np.mean(t ** 2))),
            "prediction_rms": float(np.sqrt(np.mean(p ** 2))),
            "clipped_prediction_rms": float(np.sqrt(np.mean(clipped_prediction[:, side] ** 2))),
            "raw_prediction_wp": raw_wp if eligible else None,
            "clipped_prediction_wp": weighted_pearson(t, p) if eligible else None}
    return {"domain": "all TRAIN post-warmup rows; not comparable to masked development scores",
            "weighting": "both WP variants clip targets and weight by absolute clipped target; only prediction clipping differs",
            "rows": int(sequence.need.sum()), "eligible": eligible, "targets": outputs}


def summarize(sequences):
    outputs = {}
    for branch in BRANCHES:
        rows = [s["branches"][branch] for s in sequences if "branches" in s]
        drifts = [r["wp"]["drift"] for r in rows if r["wp"]["eligible"]]
        mean = np.mean(drifts, axis=0).tolist() if drifts else None
        outputs[branch] = {"eligible_sequences": len(drifts), "mean_target_wp_drift": mean,
            "strict_passed": bool(rows) and all(v["strict_passed"] for r in rows for v in r["outputs"].values()),
            "approximate_passed": bool(rows) and bool(drifts) and
                all(v["approximate_passed"] for r in rows for v in r["outputs"].values()) and
                all(r["wp"].get("passed", True) for r in rows) and
                max(map(abs, mean)) <= BUDGET["wp"]["max_mean_target_drift"]}
    return {"branches": outputs,
            "approximate_passed": bool(sequences) and all(s["equations"]["passed"] for s in sequences) and
                                  all(x["approximate_passed"] for x in outputs.values()),
            "strict_passed": bool(sequences) and all(x["strict_passed"] for x in outputs.values())}


def run_diagnostics(protocol_path):
    """Run serially after CPU scheduling; no data access before protocol verification."""
    from .data import file_fingerprint, iter_sequences
    protocol_path = Path(protocol_path).resolve()
    protocol = _verify_protocol(protocol_path)
    output = protocol_path.parent
    if (output / "report.json").exists():
        raise FileExistsError("diagnostic report already exists; preserve its result")
    if file_fingerprint(protocol["train_path"]) != protocol["train_fingerprint"]:
        raise ValueError("training dataset fingerprint changed")
    torch.set_num_threads(1)
    model = load_torch_model(protocol["model_path"])
    exported = export_original_graph(model, protocol["model_path"], output / "zero_update.onnx")
    if _sha(exported) != _sha(protocol["model_path"]):
        raise ValueError("zero-update original-graph export lost exact parameter/graph identity")
    encoder = OnnxSequenceEncoder(exported, output / "encoder.onnx")
    started, sequences = time.monotonic(), []
    report = {"protocol_path": str(protocol_path), "protocol_sha256": _sha(protocol_path),
              "budget": protocol["budget"], "source_sha256": _sha(protocol["model_path"]),
              "zero_update_sha256": _sha(exported), "encoder": encoder.provenance,
              "torch": str(torch.__version__), "threads": 1, "sequences": sequences,
              "strict_historical_status": protocol["strict_status"], "stages": [], "mode": protocol["mode"],
              "approximate_candidate_export_accepted": False}
    for sequence in iter_sequences(protocol["train_path"], protocol["groups"]):
        if sequence.seq_ix != protocol["expected_sequence_ids"][str(sequence.row_group)]:
            raise ValueError("training sequence ID differs from frozen manifest")
        reference = encoder(sequence.features)
        equations = fp64_equation_spotcheck(model, sequence.features, reference)
        item = {"row_group": sequence.row_group, "seq_ix": sequence.seq_ix,
                "rows": len(sequence.features), "equations": equations,
                "ort_teacher_domain": domain_statistics(sequence, reference[2])}
        sequences.append(item)
        if equations["passed"]:
            item["branches"] = {}
            for branch, values in torch_paths(model, sequence.features, reference,
                        batch_size=protocol["common_state_batch_size"]).items():
                limits = BUDGET["common_state" if branch == "common_state" else "carried_state"]
                metrics = {name: error_metrics(a, b, limits) for name, a, b in
                           zip(("h0", "h1", "prediction"), values, reference)}
                item["branches"][branch] = {"outputs": metrics,
                    "wp": wp_comparison(sequence, values[2], reference[2]) if
                          all(m["finite"] for m in metrics.values()) else
                          {"eligible": False, "drift": None, "passed": False}}
        report.update(summary=summarize(sequences), seconds=time.monotonic() - started,
                      completed_sequences=len(sequences),
                      approximate_training_accepted=False)
        if len(sequences) in protocol["stages"]:
            report["stages"].append({"sequences": len(sequences), "summary": report["summary"]})
        # The mean drift constraint is applied at declared stages; every sequence
        # must independently satisfy equations, elementwise/RMS and sequence WP.
        local_passed = equations["passed"] and all(
            all(m["approximate_passed"] for m in b["outputs"].values()) and b["wp"].get("passed", True)
            for b in item.get("branches", {}).values())
        stage_passed = len(sequences) not in protocol["stages"] or report["summary"]["approximate_passed"]
        if not local_passed or not stage_passed:
            report["stop_reason"] = "prospective numerical/equation budget failed; investigate, do not relax"
            _write(output / "report.json", report)
            return report
        _write(output / "report.json", report)
        print(f"Numerical bridge TRAIN sequence {len(sequences)}/{len(protocol['groups'])}: approximate checks passed; "
              f"strict={report['summary']['strict_passed']}", flush=True)
    accepted = len(sequences) == len(protocol["groups"]) and report["summary"]["approximate_passed"]
    report["approximate_training_accepted"] = protocol["mode"] == "initial_source" and accepted
    report["approximate_candidate_export_accepted"] = protocol["mode"] == "candidate_export" and accepted
    report["scope"] = "Training implementation only; changed exported candidates need new numerical, score and release gates"
    _write(output / "report.json", report)
    return report


def require_bridge_acceptance(report_path, source_path):
    """Gate a subsequent training experiment without claiming strict equivalence."""
    return _require_acceptance(report_path, source_path, mode="initial_source")


def require_candidate_acceptance(report_path, candidate_model_path):
    """Gate screening of this exact exported candidate; strict status is separate."""
    return _require_acceptance(report_path, candidate_model_path, mode="candidate_export")


def _require_acceptance(report_path, source_path, *, mode):
    report = _read(report_path)
    protocol_path = report["protocol_path"]
    protocol = _verify_protocol(protocol_path)
    count = 8 if mode == "initial_source" else 2
    flag = "approximate_training_accepted" if count == 8 else "approximate_candidate_export_accepted"
    if (report["protocol_sha256"] != _sha(protocol_path) or report["source_sha256"] != _sha(source_path) or
            report["zero_update_sha256"] != _sha(source_path) or
            Path(source_path).resolve() != Path(protocol["model_path"]).resolve() or
            protocol["mode"] != mode or report["mode"] != mode or report["budget"] != BUDGET or
            not report[flag] or report["completed_sequences"] != count or
            len(report["sequences"]) != count or
            [s["row_group"] for s in report["sequences"]] != protocol["groups"] or
            any(s["rows"] != 20000 or s["seq_ix"] != protocol["expected_sequence_ids"][str(s["row_group"])]
                for s in report["sequences"]) or
            not summarize(report["sequences"])["approximate_passed"]):
        raise ValueError("complete accepted numerical bridge required for this source and mode")
    for sequence in report["sequences"]:
        if set(sequence["branches"]) != set(BRANCHES):
            raise ValueError("missing training-path numerical evidence")
        for branch, evidence in sequence["branches"].items():
            limits = BUDGET["common_state" if branch == "common_state" else "carried_state"]
            if set(evidence["outputs"]) != {"h0", "h1", "prediction"}:
                raise ValueError("missing hidden-state or prediction evidence")
            for name, metric in evidence["outputs"].items():
                if (not metric["finite"] or metric["values"] != 20000 * (2 if name == "prediction" else 128) or
                        not 0 <= metric["normalized_max"] <= limits["normalized_max"] or
                        not 0 <= metric["normalized_rms"] <= limits["normalized_rms"]):
                    raise ValueError("recorded numerical error exceeds current budget or lacks full coverage")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare").add_argument("--output", required=True)
    sub.add_parser("run").add_argument("--protocol", required=True)
    candidate = sub.add_parser("prepare-candidate")
    candidate.add_argument("--model", required=True)
    candidate.add_argument("--initial-bridge", required=True)
    candidate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare_protocol(args.output))
        return 0
    if args.command == "prepare-candidate":
        print(prepare_candidate_protocol(args.model, args.initial_bridge, args.output))
        return 0
    report = run_diagnostics(args.protocol)
    print(json.dumps({"approximate_training_accepted": report["approximate_training_accepted"],
                      "summary": report["summary"]}, indent=2))
    return 0 if report["approximate_training_accepted"] or report["approximate_candidate_export_accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
