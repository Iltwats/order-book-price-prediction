"""Full-sequence TRAIN correlation loss and detached coefficients for chunk VJPs.

This is an unmasked training surrogate, not the official masked/eligible-block
WP. Future labels enter loss statistics only; this module performs no model
forward pass, state update, optimization, dataset read, or candidate selection.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
import torch

CLIP = 2.
WEIGHT_FLOOR = 1e-8
STD_FLOOR = 1e-8
TEACHER_DENOMINATOR_FLOOR = 1e-12
GRADIENT_TOLERANCE = {"atol": 1e-6, "rtol": 1e-4}


def objective_parameters():
    """Fresh JSON-compatible declaration; mutating it cannot change behavior."""
    return {"version": "full_sequence_clipped_correlation.v1",
        "label_clip": [-2., 2.], "prediction_clip": [-2., 2.],
        "prediction_clip_derivative": "one strictly inside (-2,2); zero at endpoints and outside",
        "rho_bound": [-1., 1.], "rho_bound_derivative": "zero strictly outside [-1,1] for numerical roundoff safeguard",
        "weight": "absolute clipped target; normalized separately per complete sequence and target",
        "weight_sum_floor": WEIGHT_FLOOR, "std_floor": STD_FLOOR,
        "degenerate_policy": "rho0 and zero correlation gradient when weight<floor or either std<=floor; side retained with correlation loss1",
        "aggregation": "equal mean over all batch*2 sequence-target sides, including degenerate and one-sign sides",
        "domain": "all TRAIN post-warmup rows; no validation mask or sign-eligibility filtering",
        "teacher": "raw selected-ORT predictions, unclipped weighted MSE with same label weights and full-sequence denominator",
        "teacher_denominator_floor": TEACHER_DENOMINATOR_FLOOR,
        "accumulation": "FP64; current ORT FP32 input values preserved; FP64 inputs accepted for derivative audits",
        "coefficient": "detached FP32 d(total mean loss)/d(raw prediction), already includes teacher weight and batch*2 mean",
        "chunk_rule": "compute once per complete batch; sum(proxy_prediction*coefficient_slice), no extra mean or scaling",
        "gradient_scale": "correlation contains inverse prediction standard deviation; previous MSE learning rates are not equivalent",
        "state_scope": "future labels used only in training loss; no feature/state/callback mutation"}


@torch.no_grad()
def correlation_objective(prediction, teacher, targets, *, teacher_weight=.1, warmup=99):
    """Return (actual_loss_float, detached_FP32_coefficients[T,B,2], report).

    For normalized weights q and centered y,p, d rho/d p_i is
    q_i/(std_y*std_p) * (y_i - covariance/variance_p * p_i).
    Clip derivatives and the equal sequence-target mean are then applied.
    Teacher MSE remains active for a constant prediction when target weights
    exist, providing an escape from the zero correlation-gradient guard.
    """
    values = (prediction, teacher, targets)
    if any(not isinstance(v, torch.Tensor) or v.device.type != "cpu" or
           v.dtype not in (torch.float32, torch.float64) for v in values):
        raise ValueError("matching CPU FP32 tensors required; FP64 is supported for derivative audits")
    if prediction.ndim != 3 or prediction.shape[-1] != 2 or prediction.shape[1] < 1 or any(v.shape != prediction.shape for v in values):
        raise ValueError("matching [time,batch,2] tensors required")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < len(prediction):
        raise ValueError("warmup must leave at least one training row")
    teacher_weight = float(teacher_weight)
    if not np.isfinite(teacher_weight) or teacher_weight < 0 or not all(torch.isfinite(v).all() for v in values):
        raise ValueError("finite inputs and nonnegative finite teacher weight required")
    raw, teacher64, label = [v.detach().double() for v in values]
    y, p = label.clamp(-CLIP, CLIP), raw.clamp(-CLIP, CLIP)
    weight = y.abs()
    weight[:warmup] = 0.
    total = weight.sum(0)
    q = weight / torch.where(total > 0., total, torch.ones_like(total))
    yc = y - (q * y).sum(0)
    pc = p - (q * p).sum(0)
    covariance = (q * yc * pc).sum(0)
    vy, vp = (q * yc.square()).sum(0), (q * pc.square()).sum(0)
    sy, sp = vy.sqrt(), vp.sqrt()
    valid = (total >= WEIGHT_FLOOR) & (sy > STD_FLOOR) & (sp > STD_FLOOR)
    denominator = torch.where(valid, sy * sp, torch.ones_like(sy))
    raw_rho = covariance / denominator
    rho = torch.where(valid, raw_rho.clamp(-1., 1.), torch.zeros_like(raw_rho))
    safe_vp = torch.where(valid, vp, torch.ones_like(vp))
    derivative = q / denominator * (yc - covariance / safe_vp * pc)
    # Clamping rho is a numerical safeguard, matching the scorer. Its
    # derivative is zero if roundoff puts raw rho strictly outside [-1,1].
    correlation_active = valid & (raw_rho.abs() <= 1.)
    derivative *= correlation_active & (raw.abs() < CLIP)
    divisor = total.clamp_min(TEACHER_DENOMINATOR_FLOOR)
    teacher_per_side = (weight * (raw - teacher64).square()).sum(0) / divisor
    side_count = prediction.shape[1] * 2
    correlation_coefficient = -derivative / side_count
    teacher_coefficient = teacher_weight * 2. * weight / divisor * (raw - teacher64) / side_count
    coefficient64 = correlation_coefficient + teacher_coefficient
    coefficient = coefficient64.float().detach()
    correlation_loss = (1. - rho).mean()
    teacher_loss = teacher_per_side.mean()
    loss = correlation_loss + teacher_weight * teacher_loss
    if not torch.isfinite(loss) or not torch.isfinite(coefficient64).all() or not torch.isfinite(coefficient).all():
        raise FloatingPointError("nonfinite loss or coefficient; no coefficient clipping or skipped sides")

    def scale(value):
        return {"max_absolute": float(value.abs().max()), "l2": float(value.square().sum().sqrt()),
                "per_sequence_target_l1": value.abs().sum(0).tolist()}

    report = {"objective": objective_parameters(), "loss": float(loss), "correlation_loss": float(correlation_loss),
        "teacher_loss": float(teacher_loss), "teacher_weight": teacher_weight, "mean_rho": float(rho.mean()),
        "rho_per_sequence_target": rho.tolist(), "target_weight_sum": total.tolist(),
        "normalized_weight_sum": q.sum(0).tolist(), "prediction_std_after_clip": sp.tolist(),
        "correlation_active": correlation_active.tolist(), "degenerate_sides": int((~valid).sum()),
        "degenerate_reasons": {"low_weight": (total < WEIGHT_FLOOR).tolist(),
            "target_std": (sy <= STD_FLOOR).tolist(), "prediction_std": (sp <= STD_FLOOR).tolist()},
        "prediction_at_or_outside_clip_fraction": (raw[warmup:].abs() >= CLIP).double().mean(0).tolist(),
        "warmup": warmup, "rows_per_sequence": len(prediction), "batch_size": prediction.shape[1],
        "sequence_target_denominator": side_count, "gradient_scale": {
            "correlation": scale(correlation_coefficient), "teacher": scale(teacher_coefficient),
            "total": scale(coefficient64)}}
    return float(loss), coefficient, report


def gradient_probe(model, reference, teacher, targets, *, start, objective_result=None):
    """New-objective FP32/FP64 upper/head VJP comparison; no optimizer step.

    The complete sequence determines coefficients once, including future labels
    for training only. Both precision paths receive the identical rounded FP32
    coefficients and unchanged current-student ORT forward residuals/states.
    """
    from .ort_training_bridge import exact_forward_proxy, GRADIENT_NAMES
    if not 0 <= start <= len(targets) - 256:
        raise ValueError("gradient probe requires a complete 256-row chunk")
    if objective_result is None:
        objective_result = correlation_objective(torch.from_numpy(reference[2])[:, None],
            torch.from_numpy(teacher)[:, None], torch.from_numpy(targets)[:, None])
    loss, coefficients, objective_report = objective_result
    if (coefficients.shape != (len(targets), 1, 2) or coefficients.dtype != torch.float32 or
            coefficients.requires_grad or not torch.isfinite(coefficients).all() or
            objective_report["objective"] != objective_parameters()):
        raise ValueError("invalid full-sequence coefficient snapshot")
    local_coefficient = coefficients[start:start + 256]
    results, forwards = {}, {}
    for dtype in (torch.float32, torch.float64):
        local = copy.deepcopy(model).to(dtype=dtype)
        lower, ort_upper, ort_prediction = [torch.from_numpy(a[start:start + 256]).to(dtype=dtype)[:, None] for a in reference]
        initial = (torch.zeros(1, 1, model.hidden_size, dtype=dtype) if start == 0 else
                   torch.from_numpy(reference[1][start - 1]).to(dtype=dtype)[None, None])
        proxy_upper, proxy_prediction, _ = exact_forward_proxy(local, lower, initial, ort_upper, ort_prediction)
        forwards[str(dtype)] = bool(torch.equal(proxy_upper, ort_upper) and torch.equal(proxy_prediction, ort_prediction))
        scalar = (proxy_prediction * local_coefficient.to(dtype=dtype)).sum()
        parameters = dict(local.named_parameters())
        derivatives = torch.autograd.grad(scalar, [parameters[name] for name in GRADIENT_NAMES])
        results[str(dtype)] = {name: value.detach().double().numpy() for name, value in zip(GRADIENT_NAMES, derivatives)}
    comparisons = {}
    for name in GRADIENT_NAMES:
        actual, expected = results["torch.float32"][name], results["torch.float64"][name]
        error = np.abs(actual - expected)
        tolerance = GRADIENT_TOLERANCE["atol"] + GRADIENT_TOLERANCE["rtol"] * np.abs(expected)
        finite = bool(np.isfinite(actual).all() and np.isfinite(expected).all())
        comparisons[name] = {"values": int(actual.size), "finite": finite,
            "max_absolute_error": float(error.max()) if finite else None,
            "rms_error": float(np.sqrt(np.mean(error ** 2))) if finite else None,
            "max_tolerance_ratio": float((error / tolerance).max()) if finite else None,
            "violations": int(np.count_nonzero(error > tolerance)) if finite else int(actual.size),
            "passed": finite and bool(np.all(error <= tolerance))}
    return {"start": start, "rows": 256, "objective_loss": loss, "objective_report": objective_report,
        "identical_forward_values": forwards, "gradient_tensors": comparisons,
        "coefficient_sha256": hashlib.sha256(coefficients.numpy().tobytes()).hexdigest(),
        "passed": all(forwards.values()) and all(g["passed"] for g in comparisons.values())}
