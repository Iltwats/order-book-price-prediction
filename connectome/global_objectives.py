"""Batch-pooled Global-WP surrogate and detached coefficients for local VJPs.

The official metric pools selected rows across an entire evaluated dataset.
This TRAIN objective pools all post-warm-up rows in one sequence batch. It is
therefore a batch surrogate, not the full-corpus or hidden-mask score. Labels
enter objective statistics only; this module never runs or mutates a model.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

CLIP = 2.
WEIGHT_FLOOR = 1e-8
STD_FLOOR = 1e-8
TEACHER_DENOMINATOR_FLOOR = 1e-12
GRADIENT_TOLERANCE = {"atol": 1e-6, "rtol": 1e-4}
POOL_AXES = (0, 1)


def objective_parameters():
    """Return a fresh, JSON-compatible declaration of this distinct objective."""
    return {"version": "batch_pooled_global_correlation.v1",
        "label_clip": [-2., 2.], "prediction_clip": [-2., 2.],
        "prediction_clip_derivative": "one strictly inside (-2,2); zero at endpoints and outside",
        "rho_bound": [-1., 1.],
        "rho_bound_derivative": "zero strictly outside [-1,1] for numerical roundoff safeguard",
        "weight": "absolute clipped target; one denominator per target over time and batch",
        "weight_sum_floor": WEIGHT_FLOOR, "std_floor": STD_FLOOR,
        "degenerate_policy": "rho0 and zero correlation gradient when pooled weight<floor or either pooled std<=floor; target retained with correlation loss1",
        "aggregation": "pool time and batch rows per target; equal mean over two global target losses",
        "domain": "all TRAIN post-warmup rows; no validation mask or per-sequence sign eligibility",
        "surrogate_scope": "batch-pooled correlation is a surrogate for complete-corpus Global WP; batches do not yield additive correlation losses",
        "teacher": "raw selected-ORT predictions; unclipped weighted MSE with the same pooled label weights and one denominator per target",
        "teacher_denominator_floor": TEACHER_DENOMINATOR_FLOOR,
        "accumulation": "FP64; current ORT FP32 values preserved; FP64 inputs accepted for derivative audits",
        "coefficient": "detached FP32 d(total mean loss)/d(raw prediction), includes teacher weight and two-target mean only",
        "chunk_rule": "compute once over the complete sequence batch; sum(proxy_prediction*coefficient_slice), no extra mean or scaling",
        "gradient_scale": "correlation contains inverse pooled prediction standard deviation; legacy per-sequence learning rates require reassessment",
        "state_scope": "future labels used only in training loss; no feature/state/callback mutation"}


@torch.no_grad()
def correlation_objective(prediction, teacher, targets, *, teacher_weight=.1, warmup=99):
    """Return nonlinear loss, detached FP32 coefficients [T,B,2], and report.

    Center once per target over both time and batch. For normalized weights q,
    d rho/d p_i = q_i/(std_y*std_p) * (yc_i - covariance/variance_p * pc_i).
    The derivative then includes clipping and a division by two targets. It
    does not divide by batch size: batch normalization is already in q.
    """
    values = (prediction, teacher, targets)
    if any(not isinstance(value, torch.Tensor) or value.device.type != "cpu" or
           value.dtype not in (torch.float32, torch.float64) for value in values):
        raise ValueError("matching CPU FP32 tensors required; FP64 is supported for derivative audits")
    if (prediction.ndim != 3 or prediction.shape[-1] != 2 or prediction.shape[1] < 1
            or any(value.shape != prediction.shape for value in values)):
        raise ValueError("matching [time,batch,2] tensors required")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < len(prediction):
        raise ValueError("warmup must leave at least one training row")
    teacher_weight = float(teacher_weight)
    if (not np.isfinite(teacher_weight) or teacher_weight < 0
            or not all(torch.isfinite(value).all() for value in values)):
        raise ValueError("finite inputs and nonnegative finite teacher weight required")

    raw, teacher64, label = [value.detach().double() for value in values]
    y, p = label.clamp(-CLIP, CLIP), raw.clamp(-CLIP, CLIP)
    weight = y.abs()
    weight[:warmup] = 0.
    total = weight.sum(POOL_AXES)
    q = weight / torch.where(total > 0., total, torch.ones_like(total))
    yc = y - (q * y).sum(POOL_AXES)
    pc = p - (q * p).sum(POOL_AXES)
    covariance = (q * yc * pc).sum(POOL_AXES)
    vy, vp = (q * yc.square()).sum(POOL_AXES), (q * pc.square()).sum(POOL_AXES)
    sy, sp = vy.sqrt(), vp.sqrt()
    valid = (total >= WEIGHT_FLOOR) & (sy > STD_FLOOR) & (sp > STD_FLOOR)
    denominator = torch.where(valid, sy * sp, torch.ones_like(sy))
    raw_rho = covariance / denominator
    rho = torch.where(valid, raw_rho.clamp(-1., 1.), torch.zeros_like(raw_rho))
    safe_vp = torch.where(valid, vp, torch.ones_like(vp))
    derivative = q / denominator * (yc - covariance / safe_vp * pc)
    correlation_active = valid & (raw_rho.abs() <= 1.)
    derivative *= correlation_active & (raw.abs() < CLIP)

    divisor = total.clamp_min(TEACHER_DENOMINATOR_FLOOR)
    teacher_per_target = (weight * (raw - teacher64).square()).sum(POOL_AXES) / divisor
    correlation_coefficient = -derivative / 2.
    teacher_coefficient = teacher_weight * weight / divisor * (raw - teacher64)
    coefficient64 = correlation_coefficient + teacher_coefficient
    coefficient = coefficient64.float().detach()
    correlation_loss, teacher_loss = (1. - rho).mean(), teacher_per_target.mean()
    loss = correlation_loss + teacher_weight * teacher_loss
    if (not torch.isfinite(loss) or not torch.isfinite(coefficient64).all()
            or not torch.isfinite(coefficient).all()):
        raise FloatingPointError("nonfinite loss or coefficient; no coefficient clipping or skipped targets")

    def scale(value):
        return {"max_absolute": float(value.abs().max()),
                "l2": float(value.square().sum().sqrt()),
                "per_target_l1": value.abs().sum(POOL_AXES).tolist()}

    report = {"objective": objective_parameters(), "loss": float(loss),
        "correlation_loss": float(correlation_loss), "teacher_loss": float(teacher_loss),
        "teacher_loss_per_target": teacher_per_target.tolist(), "teacher_weight": teacher_weight,
        "mean_rho": float(rho.mean()), "rho_per_target": rho.tolist(),
        "target_weight_sum": total.tolist(), "normalized_weight_sum": q.sum(POOL_AXES).tolist(),
        "target_std_after_clip": sy.tolist(), "prediction_std_after_clip": sp.tolist(),
        "correlation_active": correlation_active.tolist(), "degenerate_targets": int((~valid).sum()),
        "degenerate_reasons": {"low_weight": (total < WEIGHT_FLOOR).tolist(),
            "target_std": (sy <= STD_FLOOR).tolist(), "prediction_std": (sp <= STD_FLOOR).tolist()},
        "prediction_at_or_outside_clip_fraction": (raw[warmup:].abs() >= CLIP).double().mean(POOL_AXES).tolist(),
        "warmup": warmup, "rows_per_sequence": len(prediction), "batch_size": prediction.shape[1],
        "pooled_post_warmup_rows": (len(prediction)-warmup)*prediction.shape[1], "target_denominator": 2,
        "coefficient_sha256": hashlib.sha256(coefficient.contiguous().numpy().tobytes()).hexdigest(),
        "gradient_scale": {"correlation": scale(correlation_coefficient),
                           "teacher": scale(teacher_coefficient), "total": scale(coefficient64)}}
    return float(loss), coefficient, report
