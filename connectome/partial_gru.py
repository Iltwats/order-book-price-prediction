"""ORT-forward upper-GRU/head training with local Torch TBPTT gradients.

The current student runs full causal sequences in its original ONNX graph;
Torch chunks supply approximate local derivatives from exact ORT boundaries.
The selected original ONNX teacher and lower parameters remain unchanged.
This is a distinct training method, not a pass of the failed carried-Torch
bridge. Importing this module performs no training or numerical acceptance.
"""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch

from .conversion_audit import export_original_graph
from .data import file_fingerprint
from .evaluation import file_hash, save_json
from .model_io import export_sequence_model, load_weights
from .training import (OnnxSequenceEncoder, _atomic_checkpoint, _environment,
                       _iter_sequences, _restore_rng, _rng_state,
                       load_torch_model, seed_everything)
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, WARMUP

FORMAT = "connectome.partial-gru.ort-forward.v2"
LOCAL_DIAGNOSTIC_LIMITS = {
    "normalized_max": float(512 * np.finfo(np.float32).eps),
    "normalized_rms": float(32 * np.finfo(np.float32).eps),
}
DEFAULT_PARAMETERS = {
    "batch_size": 8, "chunk_size": 256, "learning_rate": 1e-5,
    "teacher_weight": .1, "anchor_weight": 1e-3, "weight_decay": 1e-4,
    "gradient_clip": 1., "threads": 1,
}
LOSS_POLICY = {
    "dtype": "float32", "device": "cpu", "scope": "upper_gru_and_head",
    "forward": "current_student_original_onnx_full_sequence_snapshot_per_batch",
    "gradient": "torch_chunk_local_jacobian_with_ort_hidden_and_prediction_forward_corrections",
    "chunk_initial_state": "detached_current_student_ort_upper_at_start_minus_one_or_zero",
    "local_diagnostics": "uncorrected_torch_upper_and_prediction_per_sequence_coordinate_rms_before_update",
    "local_diagnostic_limits": LOCAL_DIAGNOSTIC_LIMITS,
    "teacher": "unchanged_source_original_onnx_raw_predictions",
    "teacher_normalization": "same_clipped_label_absolute_weights_and_full_sequence_target_denominators_as_supervision",
    "anchor_normalization": "joint_squared_distance_over_selected_upper_and_head_squared_norm_once_per_batch",
    "warmup_rows": WARMUP, "training_mask": "all_post_warmup_rows_no_sign_eligibility_filter",
}


def partial_parameters(**overrides):
    """Return the exact JSON-compatible parameter declaration used in reports."""
    unknown = set(overrides) - set(DEFAULT_PARAMETERS)
    if unknown:
        raise ValueError(f"unknown partial-GRU parameters: {sorted(unknown)}")
    values = {**DEFAULT_PARAMETERS, **overrides}
    for key in ("batch_size", "chunk_size", "threads"):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{key} must be a positive integer")
        values[key] = int(value)
    for key in ("learning_rate", "teacher_weight", "anchor_weight", "weight_decay", "gradient_clip"):
        value = float(values[key])
        if not np.isfinite(value) or value < 0 or key in {"learning_rate", "gradient_clip"} and value == 0:
            raise ValueError(f"invalid {key}")
        values[key] = value
    return {**values, **LOSS_POLICY, "local_diagnostic_limits": dict(LOCAL_DIAGNOSTIC_LIMITS)}


def _validated_groups(path, groups):
    parquet = pq.ParquetFile(path)
    names = parquet.schema_arrow.names
    if "is_scored" in names:
        raise ValueError("partial-GRU fitting requires training data without is_scored")
    required = {"seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS}
    if not required <= set(names):
        raise ValueError("missing supplied training columns")
    groups = list(groups)
    if (not groups or any(isinstance(g, bool) or not isinstance(g, (int, np.integer)) for g in groups)
            or len(set(groups)) != len(groups) or min(groups) < 0 or max(groups) >= parquet.num_row_groups):
        raise ValueError("groups must be distinct existing integer row-group indices")
    return [int(group) for group in groups]


def _source_hashes():
    root = Path(__file__).resolve().parent
    paths = [Path(__file__), root / "training.py", root / "conversion_audit.py",
             root / "model_io.py", root / "data.py", root / "evaluation.py", root.parent / "utils.py"]
    return {str(path.resolve()): file_hash(path) for path in paths}


def weighted_chunk_losses(prediction, teacher, target, denominators, *, start_step):
    """Additive label/teacher losses, each averaged over sequence-target pairs.

Both use |clip(label,-2,2)| / full-sequence post-warm-up target weight.
The teacher is the raw calibrated selected prediction. Predictions are not
clipped inside either MSE; zero-label rows have zero loss weight. Empty-weight
target sides contribute zero and remain in the batch mean.
"""
    if prediction.ndim != 3 or prediction.shape[-1] != 2 or teacher.shape != prediction.shape or target.shape != prediction.shape:
        raise ValueError("prediction, teacher and target must have matching [time,batch,2] shapes")
    if denominators.shape != prediction.shape[1:] or start_step < 0:
        raise ValueError("invalid sequence denominators or chunk position")
    clipped = target.clamp(-2., 2.)
    mask = (torch.arange(len(target), device=target.device) + start_step >= WARMUP)[:, None, None]
    weights = clipped.abs() * mask
    divisor = denominators.clamp_min(1e-12)
    supervised = ((weights * (prediction-clipped).square()).sum(0) / divisor).mean()
    matching = ((weights * (prediction-teacher.detach()).square()).sum(0) / divisor).mean()
    return supervised, matching


def relative_anchor(parameters, reference):
    """One joint relative distance, independent of chunk and parameter counts."""
    numerator = sum((p-reference[name]).square().sum() for name, p in parameters.items())
    denominator = sum(p.square().sum() for p in reference.values()).clamp_min(1e-12)
    return numerator / denominator


def matched_forward_prediction(head, torch_hidden, ort_hidden, ort_prediction):
    """Use exact ORT forward values and local Torch derivatives, with one head call.

    Parenthesized self-subtractions must not be reassociated. This is an
    approximate Jacobian, not differentiation through the ORT implementation.
    ORT tensors are detached even when a caller supplies gradient-enabled data.
    """
    if torch_hidden.shape != ort_hidden.shape or ort_prediction.shape != (*ort_hidden.shape[:-1], 2):
        raise ValueError("matched forward tensors have incompatible shapes")
    hidden_proxy = ort_hidden.detach() + (torch_hidden - torch_hidden.detach())
    local_prediction = head(hidden_proxy)
    return ort_prediction.detach() + (local_prediction - local_prediction.detach())


def reanchored_upper_chunk(model, lower, ort_upper, ort_prediction, *, start, end):
    """Return corrected predictions and uncorrected diagnostics for one chunk."""
    if not 0 <= start < end <= len(lower) or ort_upper.shape[:2] != lower.shape[:2]:
        raise ValueError("invalid reanchored chunk bounds or states")
    initial = (ort_upper[start-1:start].detach() if start else
               lower.new_zeros((1, lower.shape[1], ort_upper.shape[2])))
    upper, _unused_final_state = model.gru1(lower[start:end].detach(), initial)
    prediction = matched_forward_prediction(model.head, upper, ort_upper[start:end], ort_prediction[start:end])
    with torch.no_grad():
        raw_prediction = model.head(upper.detach())
    return prediction, upper, raw_prediction


def _encode_batch(teacher_encoder, student_encoder, train_path, groups):
    # Preallocate after seeing the first sequence: no unbounded hidden cache or
    # list-to-stack duplication. Each encoder result is copied before reuse.
    arrays, ids, rows = None, [], None
    for seq_id, x, y in _iter_sequences(train_path, groups):
        if (x.ndim != 2 or x.shape[1] != 112 or len(x) <= WARMUP
                or y.shape != (len(x), 2) or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError("invalid or nonfinite training sequence")
        if rows is None:
            rows = len(x)
            arrays = [np.empty((rows, len(groups), width), np.float32) for width in (128, 128, 2, 2, 2)]
        if len(x) != rows or len(ids) >= len(groups):
            raise ValueError("incomplete, duplicate-ID or unequal-length training batch")
        h0, h1, prediction = student_encoder(x)
        for value, width in ((h0, 128), (h1, 128), (prediction, 2)):
            if value.shape != (rows, width) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("original ONNX student returned invalid lower states, upper states or predictions")
        slot = len(ids)
        for destination, value in zip(arrays[:3], (h0, h1, prediction)):
            destination[:, slot] = value
        teacher_lower, _, teacher = teacher_encoder(x)
        if (teacher.shape != (rows, 2) or teacher.dtype != np.float32 or not np.isfinite(teacher).all()
                or not np.array_equal(teacher_lower, arrays[0][:, slot])):
            raise ValueError("selected ONNX teacher is invalid or frozen lower forward values changed")
        arrays[3][:, slot], arrays[4][:, slot] = teacher, y
        ids.append(int(seq_id))
    if len(ids) != len(groups) or len(set(ids)) != len(ids):
        raise ValueError("incomplete, duplicate-ID or unequal-length training batch")
    return (*[torch.from_numpy(value) for value in arrays], ids)


class _LocalDiagnostics:
    """Bounded accumulation of uncorrected errors; no corrected-value tautology."""

    def __init__(self):
        self.statistics = {}
        self.rows = 0

    def add(self, upper, prediction, reference_upper, reference_prediction):
        for name, actual, reference in (("upper", upper, reference_upper),
                                         ("prediction", prediction, reference_prediction)):
            actual, reference = (v.detach().numpy().astype(np.float64) for v in (actual, reference))
            if not np.isfinite(actual).all() or not np.isfinite(reference).all():
                raise FloatingPointError("nonfinite raw local Torch/ORT diagnostic; last checkpoint retained")
            error = actual - reference
            maximum, squared, scale_squared = np.abs(error).max(0), (error**2).sum(0), (reference**2).sum(0)
            if name not in self.statistics:
                self.statistics[name] = [maximum, squared, scale_squared]
            else:
                prior = self.statistics[name]
                prior[0] = np.maximum(prior[0], maximum); prior[1] += squared; prior[2] += scale_squared
        self.rows += len(upper)

    def finish(self, sequence_ids):
        result = []
        for slot, sequence_id in enumerate(sequence_ids):
            entry = {"sequence_id": sequence_id, "rows": self.rows}
            for name, (maximum, squared, scale_squared) in self.statistics.items():
                scale = np.maximum(1., np.sqrt(scale_squared[slot] / self.rows))
                peak = float((maximum[slot] / scale).max())
                rms = float((np.sqrt(squared[slot] / self.rows) / scale).max())
                entry[name] = {"normalized_max": peak, "normalized_rms": rms,
                               "passed": peak <= LOCAL_DIAGNOSTIC_LIMITS["normalized_max"] and
                                         rms <= LOCAL_DIAGNOSTIC_LIMITS["normalized_rms"]}
            result.append(entry)
        return result


def _read_checkpoint(path):
    # Trusted local checkpoints only; this is not a third-party model loader.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or value.get("format") != FORMAT:
        raise ValueError("not a partial-GRU checkpoint")
    return value


def train_partial_gru(model_path, train_path, output_dir, *, group_indices, epochs=1,
                      seed=20260912, resume_from=None, max_updates=None, **overrides):
    """Train upper GRU/head; pause/resume only at complete sequence-batch boundaries.

    ``epochs`` is the predeclared total (1..3); resume cannot change it or any
    hyperparameter. ``max_updates`` is an optional absolute update count used
    for bounded pauses. No validation rows, selection or evaluation run here.
    """
    started = time.monotonic()
    parameters = partial_parameters(**overrides)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 3:
        raise ValueError("epochs must be a predeclared integer from 1 through 3")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if max_updates is not None and (isinstance(max_updates, bool) or not isinstance(max_updates, int) or max_updates < 1):
        raise ValueError("max_updates must be a positive absolute update count")
    model_path, train_path, output_dir = (Path(p).resolve() for p in (model_path, train_path, output_dir))
    groups = _validated_groups(train_path, group_indices)
    specification = {
        "method": "partial_gru", "groups": groups, "seed": seed, "epochs": epochs,
        "parameters": parameters, "source": {"path": str(model_path), "sha256": file_hash(model_path)},
        "data": {"path": str(train_path), **file_fingerprint(train_path)},
        "implementation": _source_hashes(),
    }
    previous = _read_checkpoint(resume_from) if resume_from is not None else None
    if previous is None and output_dir.exists():
        raise FileExistsError("new partial-GRU output directory already exists")
    if previous is not None and previous.get("specification") != specification:
        raise ValueError("resume specification changed (source, data, groups, budget, parameters or code)")
    seed_everything(seed, parameters["threads"])
    model = load_torch_model(model_path, device="cpu")
    if model.hidden_size != 128 or model.head_state != "last":
        raise ValueError("partial-GRU training requires the original GRU128 architecture")
    for p in model.gru0.parameters():
        p.requires_grad_(False)
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any(not name.startswith(("gru1.", "head.")) for name in trainable):
        raise ValueError("only upper-GRU and head parameters may train")
    reference = {name: p.detach().clone() for name, p in trainable.items()}
    lower_reference = {name: p.detach().clone() for name, p in model.gru0.state_dict().items()}
    optimizer = torch.optim.AdamW(list(trainable.values()), lr=parameters["learning_rate"],
                                 weight_decay=parameters["weight_decay"])
    cursor = {"epoch": 0, "position": 0, "updates": 0, "sequences_seen": 0}
    history = []
    if previous is not None:
        model.load_state_dict(previous["model_state"])
        if any(not torch.equal(p, lower_reference[name]) for name, p in model.gru0.state_dict().items()):
            raise ValueError("checkpoint altered the frozen lower GRU")
        optimizer.load_state_dict(previous["optimizer_state"])
        cursor, history = dict(previous["cursor"]), list(previous["history"])
        _restore_rng(previous["rng"])
        if (not 0 <= cursor["epoch"] <= epochs or not 0 <= cursor["position"] < len(groups)
                or cursor["epoch"] == epochs and cursor["position"] != 0
                or cursor["updates"] != len(history)
                or max_updates is not None and max_updates < cursor["updates"]):
            raise ValueError("invalid resumed cursor or update budget")
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = OnnxSequenceEncoder(model_path, output_dir / "source_encoder.onnx")
    if previous is not None and previous["encoder"] != encoder.provenance:
        # Path can change when resuming into a new output directory; content cannot.
        old = {k: v for k, v in previous["encoder"].items() if k != "path"}
        new = {k: v for k, v in encoder.provenance.items() if k != "path"}
        if old != new:
            raise ValueError("resume encoder implementation or source changed")

    def checkpoint():
        return {"format": FORMAT, "specification": specification, "encoder": encoder.provenance,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "cursor": dict(cursor), "history": list(history), "rng": _rng_state()}

    checkpoint_path = output_dir / "checkpoint.pt"
    _atomic_checkpoint(checkpoint_path, checkpoint())
    model.train()
    while cursor["epoch"] < epochs and (max_updates is None or cursor["updates"] < max_updates):
        epoch = cursor["epoch"]
        order = np.random.default_rng(seed + epoch).permutation(groups).tolist()
        batch_groups = order[cursor["position"]:cursor["position"] + parameters["batch_size"]]
        if file_hash(model_path) != specification["source"]["sha256"]:
            raise ValueError("selected teacher/source model changed during partial-GRU fitting")
        parameter_versions = {name: p._version for name, p in trainable.items()}
        # One immutable current-student snapshot supplies every forward row in
        # this batch. It is regenerated only after the preceding optimizer step.
        snapshot_path = export_original_graph(model, model_path, output_dir / "current_student.onnx")
        snapshot_sha = file_hash(snapshot_path)
        student_encoder = OnnxSequenceEncoder(snapshot_path, output_dir / "current_student_encoder.onnx")
        student_provenance = {k: v for k, v in student_encoder.provenance.items() if k != "path"}
        lower, ort_upper, ort_prediction, teacher, target, sequence_ids = _encode_batch(
            encoder, student_encoder, train_path, batch_groups,
        )
        del student_encoder  # Release the ORT session before the next snapshot.
        denominators = target[WARMUP:].clamp(-2., 2.).abs().sum(0)
        optimizer.zero_grad(set_to_none=True)
        supervised_total, teacher_total = 0., 0.
        diagnostics = _LocalDiagnostics()
        for start in range(0, len(lower), parameters["chunk_size"]):
            end = min(start + parameters["chunk_size"], len(lower))
            prediction, raw_upper, raw_prediction = reanchored_upper_chunk(
                model, lower, ort_upper, ort_prediction, start=start, end=end,
            )
            diagnostics.add(raw_upper, raw_prediction, ort_upper[start:end], ort_prediction[start:end])
            if not torch.equal(prediction.detach(), ort_prediction[start:end]):
                raise FloatingPointError("matched prediction forward values differ from the current ORT student")
            supervised, matching = weighted_chunk_losses(
                prediction, teacher[start:end], target[start:end], denominators, start_step=start,
            )
            loss = supervised + parameters["teacher_weight"] * matching
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite partial-GRU batch loss; last checkpoint retained")
            loss.backward()
            supervised_total += float(supervised.detach()); teacher_total += float(matching.detach())
        local_report = diagnostics.finish(sequence_ids)
        if any(not entry[name]["passed"] for entry in local_report for name in ("upper", "prediction")):
            save_json(output_dir / "failed_local_diagnostic.json", {
                "update": cursor["updates"] + 1, "student_model_sha256": snapshot_sha,
                "limits": LOCAL_DIAGNOSTIC_LIMITS, "sequences": local_report,
            })
            raise FloatingPointError("raw local Torch/ORT diagnostic exceeded unchanged limits; last checkpoint retained")
        if any(p._version != parameter_versions[name] for name, p in trainable.items()):
            raise AssertionError("student parameters changed before completing the sequence batch")
        anchor = relative_anchor(trainable, reference)
        (parameters["anchor_weight"] * anchor).backward()  # Exactly once per sequence batch.
        gradient_norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), parameters["gradient_clip"])
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite partial-GRU gradient; last checkpoint retained")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("nonfinite partial-GRU parameters; last checkpoint retained")
        if any(not torch.equal(p, lower_reference[name]) for name, p in model.gru0.state_dict().items()):
            raise AssertionError("frozen lower GRU changed during optimization")
        if file_fingerprint(train_path) != {k: v for k, v in specification["data"].items() if k != "path"}:
            raise ValueError("training data changed during partial-GRU fitting")
        cursor["updates"] += 1
        cursor["sequences_seen"] += len(batch_groups)
        cursor["position"] += len(batch_groups)
        history.append({"epoch": epoch, "update": cursor["updates"], "groups": batch_groups,
                        "sequence_ids": sequence_ids, "supervised_loss": supervised_total,
                        "teacher_loss": teacher_total, "relative_anchor": float(anchor.detach()),
                        "gradient_norm": float(gradient_norm), "student_model_sha256": snapshot_sha,
                        "student_encoder": student_provenance,
                        "teacher_model_sha256": specification["source"]["sha256"],
                        "local_diagnostics": local_report})
        if cursor["position"] == len(groups):
            cursor["epoch"] += 1; cursor["position"] = 0
        _atomic_checkpoint(checkpoint_path, checkpoint())
        print(f"partial GRU {cursor['updates']} batches / {cursor['sequences_seen']} sequences; "
              f"{time.monotonic()-started:.1f}s", flush=True)
    model.eval()
    if file_hash(model_path) != specification["source"]["sha256"]:
        raise ValueError("selected teacher/source model changed before final export")
    exported = export_original_graph(model, model_path, output_dir / "model.onnx")
    sequence_metadata = export_sequence_model(exported, output_dir / "sequence.onnx")
    sequence_model = sequence_metadata["path"]
    original, actual = load_weights(model_path), load_weights(exported)
    lower_equal = all(original[k].tobytes() == actual[k].tobytes() for k in ("w0", "r0", "b0"))
    if not lower_equal:
        raise AssertionError("original-graph export altered frozen lower weights")
    report = {"method": "partial_gru", "complete": cursor["epoch"] >= epochs,
              "specification": specification, "model_path": exported, "model_sha256": file_hash(exported),
              "sequence_model_path": sequence_model, "sequence_model_sha256": file_hash(sequence_model),
              "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": file_hash(checkpoint_path),
              "cursor": cursor, "history": history, "encoder": encoder.provenance,
              "lower_recurrence_bitwise_equal": lower_equal, "teacher_model_sha256": file_hash(model_path),
              "forward_method": LOSS_POLICY["forward"], "gradient_method": LOSS_POLICY["gradient"],
              "local_diagnostic_limits": LOCAL_DIAGNOSTIC_LIMITS,
              "training_metric": "sequence-target mean weighted label MSE + weighted raw selected-teacher MSE + relative parameter anchor",
              "deployment_gate": "separately accepted ORT-forward/local-gradient bridge, candidate original-graph checks and clipped development WP still required; prior carried-Torch bridge remains failed",
              "seconds": time.monotonic()-started, "environment": _environment()}
    save_json(output_dir / "report.json", report)
    return report


def resume_partial_gru(checkpoint_path, output_dir=None, *, max_updates=None):
    """Resume the exact declared run; ``max_updates`` remains an absolute bound."""
    state = _read_checkpoint(checkpoint_path)
    spec = state["specification"]
    return train_partial_gru(
        spec["source"]["path"], spec["data"]["path"], output_dir or Path(checkpoint_path).resolve().parent,
        group_indices=spec["groups"], epochs=spec["epochs"], seed=spec["seed"],
        resume_from=checkpoint_path, max_updates=max_updates,
        **{key: spec["parameters"][key] for key in DEFAULT_PARAMETERS},
    )
