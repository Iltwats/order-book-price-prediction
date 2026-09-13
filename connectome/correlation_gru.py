"""Separately declared clipped-correlation upper-GRU training.

The frozen ORT-forward machinery supplies current-student trajectories and
local Torch Jacobians. A full-sequence nonlinear objective supplies detached
FP32 upstream derivatives once per batch; chunkwise backward uses their dot
product with the exact-forward proxy. That dot product is never a WP score or
the reported nonlinear loss. No frozen training source is modified here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from . import partial_gru as base
from . import recurrent_objectives as objective
from .partial_gru import (_encode_batch, _validated_groups, _LocalDiagnostics,
    OnnxSequenceEncoder, export_original_graph, export_sequence_model, load_weights,
    file_hash, file_fingerprint, save_json, load_torch_model, seed_everything,
    _atomic_checkpoint, _environment, _rng_state, _restore_rng)

FORMAT = "connectome.correlation-gru.ort-forward.v1"
DEFAULT_PARAMETERS = dict(base.DEFAULT_PARAMETERS)


def correlation_parameters(**overrides):
    """Canonical declaration, including the independently specified objective."""
    values = base.partial_parameters(**overrides)
    values["objective"] = objective.objective_parameters()
    values["backward_coefficients"] = "full_sequence_objective_detached_fp32_once_per_batch; chunk_dot_product_without_further_mean"
    return values


def _sources():
    return {**base._source_hashes(), str(Path(__file__).resolve()): file_hash(__file__),
            str(Path(objective.__file__).resolve()): file_hash(objective.__file__)}


def _read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != FORMAT:
        raise ValueError("not a correlation-GRU checkpoint")
    return state


def train_correlation_gru(model_path, train_path, output_dir, *, group_indices,
                          epochs=1, seed=20260912, resume_from=None, max_updates=None, **overrides):
    """Train only on the declared complete training sequences; never select here."""
    started = time.monotonic()
    parameters = correlation_parameters(**overrides)
    if type(epochs) is not int or not 1 <= epochs <= 3 or type(seed) is not int or seed < 0:
        raise ValueError("predeclared epochs 1..3 and nonnegative integer seed required")
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive absolute count")
    model_path, train_path, output_dir = (Path(p).resolve() for p in (model_path, train_path, output_dir))
    specification = {"method": "correlation_gru", "groups": _validated_groups(train_path, group_indices),
        "seed": seed, "epochs": epochs, "parameters": parameters,
        "source": {"path": str(model_path), "sha256": file_hash(model_path)},
        "data": {"path": str(train_path), **file_fingerprint(train_path)}, "implementation": _sources()}
    groups = specification["groups"]
    previous = _read_checkpoint(resume_from) if resume_from is not None else None
    if previous is None and output_dir.exists():
        raise FileExistsError("new correlation-GRU output directory already exists")
    if previous is not None and previous.get("specification") != specification:
        raise ValueError("resume specification changed (objective, data, source, groups, budget, parameters or code)")
    seed_everything(seed, parameters["threads"])
    model = load_torch_model(model_path, device="cpu")
    if model.hidden_size != 128 or model.head_state != "last":
        raise ValueError("original GRU128 architecture required")
    for parameter in model.gru0.parameters():
        parameter.requires_grad_(False)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not name.startswith(("gru1.", "head.")) for name in trainable):
        raise ValueError("only upper GRU and head may train")
    anchor = {name: p.detach().clone() for name, p in trainable.items()}
    lower_original = {name: p.detach().clone() for name, p in model.gru0.state_dict().items()}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
    cursor, history = {"epoch": 0, "position": 0, "updates": 0, "sequences_seen": 0}, []
    if previous is not None:
        model.load_state_dict(previous["model_state"])
        if any(not torch.equal(p, lower_original[name]) for name, p in model.gru0.state_dict().items()):
            raise ValueError("checkpoint changed frozen lower GRU")
        optimizer.load_state_dict(previous["optimizer_state"])
        cursor, history = dict(previous["cursor"]), list(previous["history"])
        _restore_rng(previous["rng"])
        if (not 0 <= cursor["epoch"] <= epochs or not 0 <= cursor["position"] < len(groups)
                or cursor["epoch"] == epochs and cursor["position"] != 0 or cursor["updates"] != len(history)
                or max_updates is not None and max_updates < cursor["updates"]):
            raise ValueError("invalid resumed cursor or update budget")
    output_dir.mkdir(parents=True, exist_ok=True)
    teacher_encoder = OnnxSequenceEncoder(model_path, output_dir / "source_encoder.onnx")
    if previous is not None and ({k: v for k, v in previous["encoder"].items() if k != "path"} !=
                                  {k: v for k, v in teacher_encoder.provenance.items() if k != "path"}):
        raise ValueError("resume teacher encoder changed")
    checkpoint_path = output_dir / "checkpoint.pt"

    def save_checkpoint():
        _atomic_checkpoint(checkpoint_path, {"format": FORMAT, "specification": specification,
            "encoder": teacher_encoder.provenance, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "cursor": dict(cursor), "history": list(history), "rng": _rng_state()})

    save_checkpoint()
    model.train()
    while cursor["epoch"] < epochs and (max_updates is None or cursor["updates"] < max_updates):
        epoch = cursor["epoch"]
        order = np.random.default_rng(seed+epoch).permutation(groups).tolist()
        batch_groups = order[cursor["position"]:cursor["position"]+parameters["batch_size"]]
        if file_hash(model_path) != specification["source"]["sha256"]:
            raise ValueError("selected source changed during fitting")
        versions = {name: p._version for name, p in trainable.items()}
        snapshot = export_original_graph(model, model_path, output_dir / "current_student.onnx")
        student = OnnxSequenceEncoder(snapshot, output_dir / "current_student_encoder.onnx")
        provenance = {k: v for k, v in student.provenance.items() if k != "path"}
        lower, upper, prediction, teacher, target, sequence_ids = _encode_batch(teacher_encoder, student, train_path, batch_groups)
        del student
        actual_loss, coefficients, objective_report = objective.correlation_objective(
            prediction, teacher, target, teacher_weight=parameters["teacher_weight"], warmup=parameters["warmup_rows"])
        if (not np.isfinite(actual_loss) or coefficients.shape != prediction.shape or coefficients.dtype != torch.float32
                or coefficients.device.type != "cpu" or coefficients.requires_grad or not torch.isfinite(coefficients).all()):
            raise FloatingPointError("invalid full-sequence objective or upstream coefficients; last checkpoint retained")
        coefficient_hash = hashlib.sha256(coefficients.contiguous().numpy().tobytes()).hexdigest()
        optimizer.zero_grad(set_to_none=True)
        diagnostics, linear_scalar = _LocalDiagnostics(), 0.
        for start in range(0, len(lower), parameters["chunk_size"]):
            end = min(start+parameters["chunk_size"], len(lower))
            proxy, raw_upper, raw_prediction = base.reanchored_upper_chunk(model, lower, upper, prediction, start=start, end=end)
            diagnostics.add(raw_upper, raw_prediction, upper[start:end], prediction[start:end])
            if not torch.equal(proxy.detach(), prediction[start:end]):
                raise FloatingPointError("matched forward values differ from current ORT student")
            scalar = (proxy*coefficients[start:end]).sum()  # Already normalized over all rows/sequences/targets.
            if not torch.isfinite(scalar):
                raise FloatingPointError("nonfinite local backward scalar; last checkpoint retained")
            scalar.backward()
            linear_scalar += float(scalar.detach())
        local_report = diagnostics.finish(sequence_ids)
        if any(not row[name]["passed"] for row in local_report for name in ("upper", "prediction")):
            save_json(output_dir / "failed_local_diagnostic.json", {"update": cursor["updates"]+1,
                "student_model_sha256": file_hash(snapshot), "limits": base.LOCAL_DIAGNOSTIC_LIMITS, "sequences": local_report})
            raise FloatingPointError("raw local Torch/ORT diagnostic exceeded unchanged limits; last checkpoint retained")
        if any(p._version != versions[name] for name, p in trainable.items()):
            raise AssertionError("student parameters changed before batch ended")
        anchor_loss = base.relative_anchor(trainable, anchor)
        (parameters["anchor_weight"]*anchor_loss).backward()
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), parameters["gradient_clip"])
        if not torch.isfinite(norm):
            raise FloatingPointError("nonfinite gradient; last checkpoint retained")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite parameters; last checkpoint retained")
        if any(not torch.equal(p, lower_original[name]) for name, p in model.gru0.state_dict().items()):
            raise AssertionError("frozen lower parameters changed")
        if file_fingerprint(train_path) != {k: v for k, v in specification["data"].items() if k != "path"}:
            raise ValueError("training data changed during fitting")
        cursor["updates"] += 1; cursor["sequences_seen"] += len(batch_groups); cursor["position"] += len(batch_groups)
        history.append({"epoch": epoch, "update": cursor["updates"], "groups": batch_groups, "sequence_ids": sequence_ids,
            "objective_loss": actual_loss, "objective_components": objective_report,
            "linear_backward_scalar_not_loss": linear_scalar, "coefficients_sha256": coefficient_hash,
            "relative_anchor": float(anchor_loss.detach()), "gradient_norm": float(norm),
            "student_model_sha256": file_hash(snapshot), "student_encoder": provenance,
            "teacher_model_sha256": specification["source"]["sha256"], "local_diagnostics": local_report})
        if cursor["position"] == len(groups):
            cursor["epoch"] += 1; cursor["position"] = 0
        save_checkpoint()
        print(f"correlation GRU {cursor['updates']} batches / {cursor['sequences_seen']} sequences; "
              f"loss {actual_loss:.6f}; {time.monotonic()-started:.1f}s", flush=True)
    model.eval()
    if file_hash(model_path) != specification["source"]["sha256"]:
        raise ValueError("selected source changed before export")
    exported = export_original_graph(model, model_path, output_dir / "model.onnx")
    sequence_path = export_sequence_model(exported, output_dir / "sequence.onnx")["path"]
    original, actual = load_weights(model_path), load_weights(exported)
    lower_equal = all(original[k].tobytes() == actual[k].tobytes() for k in ("w0", "r0", "b0"))
    if not lower_equal:
        raise AssertionError("export changed frozen lower weights")
    report = {"method": "correlation_gru", "complete": cursor["epoch"] >= epochs, "specification": specification,
        "model_path": exported, "model_sha256": file_hash(exported), "sequence_model_path": sequence_path,
        "sequence_model_sha256": file_hash(sequence_path), "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path), "cursor": cursor, "history": history,
        "encoder": teacher_encoder.provenance, "teacher_model_sha256": file_hash(model_path),
        "lower_recurrence_bitwise_equal": lower_equal, "forward_method": parameters["forward"],
        "gradient_method": parameters["gradient"], "local_diagnostic_limits": base.LOCAL_DIAGNOSTIC_LIMITS,
        "training_metric": "full-sequence mean(1-clipped-weighted-Pearson) + raw selected-teacher weighted MSE + relative parameter anchor; training-side degeneracy/masking differ from official eligibility",
        "deployment_gate": "separate objective declaration, local-gradient bridge, candidate original-graph checks and actual clipped development WP required",
        "seconds": time.monotonic()-started, "environment": _environment()}
    save_json(output_dir / "report.json", report)
    return report


def resume_correlation_gru(checkpoint_path, output_dir=None, *, max_updates=None):
    """Resume only the exact objective, source, groups, seed and total budget."""
    spec = _read_checkpoint(checkpoint_path)["specification"]
    return train_correlation_gru(spec["source"]["path"], spec["data"]["path"],
        output_dir or Path(checkpoint_path).resolve().parent, group_indices=spec["groups"], epochs=spec["epochs"], seed=spec["seed"],
        resume_from=checkpoint_path, max_updates=max_updates, **{k: spec["parameters"][k] for k in DEFAULT_PARAMETERS})
