"""Pooled, clipped Global Weighted Pearson and paired cluster uncertainty.

The refreshed official starter reference specifies pooling selected rows across
sequences with centered weighted moments and 1e-8 weight/std floors.
This module has no dataset or callback imports; callers enforce warm-up and
finite predictions on every required row before passing scoring-mask rows.
"""
from __future__ import annotations

import numpy as np

METRIC_ID = "global_wp_clip2_v1"
CLIP = 2.0
WEIGHT_FLOOR = 1e-8
STD_FLOOR = 1e-8
STATISTIC_COLUMNS = ("sum_w", "mean_y", "mean_p", "M2y", "M2p", "C")


def sufficient_statistics(targets, predictions):
    """Return centered FP64 [target, statistic] moments from selected [rows, 2]."""
    y, p = (np.asarray(value, dtype=np.float32) for value in (targets, predictions))
    if y.ndim != 2 or y.shape[-1] != 2 or p.shape != y.shape:
        raise ValueError("targets and predictions must have matching [rows, 2] shapes")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite selected target or prediction")
    y, p = (np.clip(value, -CLIP, CLIP).astype(np.float64) for value in (y, p))
    w = np.abs(y)
    total = w.sum(0)
    denominator = np.where(total > 0, total, 1.)
    mean_y, mean_p = (w * y).sum(0) / denominator, (w * p).sum(0) / denominator
    yc, pc = y - mean_y, p - mean_p
    return np.stack([total, mean_y, mean_p, (w * yc * yc).sum(0),
                     (w * pc * pc).sum(0), (w * yc * pc).sum(0)], axis=-1)


def merge_statistics(statistics, axis=0):
    """Pool centered [..., groups, ..., 2, 6] moments along a leading axis.

    The weighted parallel-variance merge preserves small within-group variance
    even when means are large relative to the differences being measured.
    """
    stats = np.asarray(statistics, dtype=np.float64)
    if stats.ndim < 3 or stats.shape[-2:] != (2, 6) or not np.isfinite(stats).all():
        raise ValueError("finite grouped statistics with trailing shape [2, 6] required")
    if not isinstance(axis, (int, np.integer)) or isinstance(axis, bool):
        raise ValueError("one leading integer merge axis required")
    axis = int(axis) if axis >= 0 else stats.ndim + int(axis)
    if not 0 <= axis < stats.ndim - 2:
        raise ValueError("merge axis must precede the target/statistic axes")
    w = stats[..., 0]
    if np.any(w < 0):
        raise ValueError("negative target-weight sum")
    total = w.sum(axis)
    denominator = np.where(total > 0, total, 1.)
    mean_y = (w * stats[..., 1]).sum(axis) / denominator
    mean_p = (w * stats[..., 2]).sum(axis) / denominator
    dy = stats[..., 1] - np.expand_dims(mean_y, axis)
    dp = stats[..., 2] - np.expand_dims(mean_p, axis)
    return np.stack([total, mean_y, mean_p,
                     (stats[..., 3] + w * dy * dy).sum(axis),
                     (stats[..., 4] + w * dp * dp).sum(axis),
                     (stats[..., 5] + w * dy * dp).sum(axis)], axis=-1)


def score_statistics(statistics):
    """Return two correlations from [..., 2, 6] pooled sufficient statistics.

    Zero-weight or globally constant sides contribute zero. A sequence with
    one-sign labels or constant predictions still contributes its moments.
    """
    stats = np.asarray(statistics, dtype=np.float64)
    if stats.ndim < 2 or stats.shape[-2:] != (2, 6) or not np.isfinite(stats).all():
        raise ValueError("finite statistics with trailing shape [2, 6] required")
    total = stats[..., 0]
    if np.any(total < 0):
        raise ValueError("negative target-weight sum")
    divisor = np.where(total >= WEIGHT_FLOOR, total, 1.0)
    covariance = stats[..., 5] / divisor
    variance_y = np.maximum(stats[..., 3] / divisor, 0.0)
    variance_p = np.maximum(stats[..., 4] / divisor, 0.0)
    sy, sp = np.sqrt(variance_y), np.sqrt(variance_p)
    valid = (total >= WEIGHT_FLOOR) & (sy > STD_FLOOR) & (sp > STD_FLOOR)
    denominator = np.where(valid, sy * sp, 1.0)
    return np.where(valid, np.clip(covariance / denominator, -1.0, 1.0), 0.0)


class GlobalAccumulator:
    """Bounded-memory aggregation; add returns recombinable block statistics."""

    def __init__(self):
        self.statistics = np.zeros((2, 6), dtype=np.float64)
        self.blocks = 0
        self.selected_rows = 0
        self.blocks_with_selected_rows = 0

    def add(self, targets, predictions, mask):
        y, p = (np.asarray(value, dtype=np.float32) for value in (targets, predictions))
        selected = np.asarray(mask, dtype=bool)
        if y.ndim != 2 or y.shape[-1] != 2 or p.shape != y.shape or selected.shape != (len(y),):
            raise ValueError("matching [rows, 2] targets/predictions and [rows] mask required")
        if not np.isfinite(y).all():
            raise ValueError("nonfinite target")
        stats = sufficient_statistics(y[selected], p[selected])
        count = int(selected.sum())
        self.statistics = merge_statistics(np.stack([self.statistics, stats]))
        self.blocks += 1
        self.selected_rows += count
        self.blocks_with_selected_rows += int(count > 0)
        return {"selected_rows": count, "sufficient_statistics": stats.tolist()}

    def result(self):
        if self.selected_rows == 0:
            raise ValueError("no selected scoring rows")
        scores = score_statistics(self.statistics)
        return {"metric_id": METRIC_ID, "t0": float(scores[0]), "t1": float(scores[1]),
                "weighted_pearson": float(scores.mean()), "blocks": self.blocks,
                "blocks_with_selected_rows": self.blocks_with_selected_rows,
                "empty_mask_blocks": self.blocks - self.blocks_with_selected_rows,
                "selected_rows": self.selected_rows, "effective_scored_rows": self.selected_rows,
                "metric_clip": [-CLIP, CLIP],
                "aggregation": "pool all selected rows across sequences per target, then mean two target correlations",
                "statistic_columns": list(STATISTIC_COLUMNS),
                "sufficient_statistics": self.statistics.tolist()}


def _indexed_report(report):
    if not isinstance(report, dict) or report.get("metric_id") != METRIC_ID:
        raise ValueError("matching Global WP metric identifiers required")
    entries = report.get("per_sequence", [])
    if not entries:
        raise ValueError("global comparison needs per-sequence sufficient statistics")
    indexed = {}
    for row in entries:
        seq = row.get("seq_ix")
        if not isinstance(seq, (int, np.integer)) or isinstance(seq, bool) or seq in indexed:
            raise ValueError("distinct integer sequence IDs required")
        stats = np.asarray(row.get("sufficient_statistics"), dtype=np.float64)
        if stats.shape != (2, 6) or not np.isfinite(stats).all() or np.any(stats[:, 0] < 0):
            raise ValueError("invalid sequence sufficient statistics")
        rows = row.get("selected_rows")
        if not isinstance(rows, (int, np.integer)) or isinstance(rows, bool) or rows < 0:
            raise ValueError("nonnegative selected-row count required")
        if rows == 0 and np.any(stats != 0):
            raise ValueError("empty sequence mask has nonzero statistics")
        indexed[int(seq)] = (row, stats)
    total_rows = sum(row[0]["selected_rows"] for row in indexed.values())
    if total_rows == 0 or report.get("selected_rows", total_rows) != total_rows:
        raise ValueError("selected-row count differs from sequence statistics")
    combined = merge_statistics(np.stack([value[1] for _, value in sorted(indexed.items())]))
    if "sufficient_statistics" in report:
        declared = np.asarray(report["sufficient_statistics"], dtype=np.float64)
        if declared.shape != (2, 6) or not np.allclose(declared, combined, atol=1e-8, rtol=1e-12):
            raise ValueError("pooled statistics differ from sequence statistics")
    score = float(score_statistics(combined).mean())
    if "weighted_pearson" in report and not np.isclose(report["weighted_pearson"], score, atol=1e-10, rtol=0):
        raise ValueError("reported global score differs from pooled statistics")
    return indexed


def _dataset_identity(report):
    provenance = report.get("provenance", {})
    return {**{key: provenance[key] for key in ("dataset_path", "dataset_size", "dataset_fingerprint", "dataset_sha256")
               if key in provenance}, **({"dataset": report["dataset"]} if "dataset" in report else {})}


def paired_global_bootstrap(candidate, baseline, *, n_resamples=10000, seed=20260912,
                            batch_size=128, confidence=.95):
    """Percentile interval for paired Global WP changes using sequence clusters.

    Resample complete sequences together, sum their moments, and recompute each
    global correlation. This does not average sequence-correlation differences.
    """
    if (type(n_resamples) is not int or n_resamples < 1 or type(batch_size) is not int
            or batch_size < 1 or not np.isfinite(confidence) or not 0 < confidence < 1):
        raise ValueError("invalid global bootstrap parameters")
    cand, base = _indexed_report(candidate), _indexed_report(baseline)
    if cand.keys() != base.keys():
        raise ValueError("paired global comparison needs exactly the same sequences")
    if _dataset_identity(candidate) != _dataset_identity(baseline):
        raise ValueError("dataset identity differs between paired reports")
    ids = sorted(cand)
    for seq in ids:
        a, b = cand[seq], base[seq]
        if a[0]["selected_rows"] != b[0]["selected_rows"] or a[0].get("row_group") != b[0].get("row_group"):
            raise ValueError("paired sequence rows or row groups differ")
        if not np.array_equal(a[1][:, [0, 1, 3]], b[1][:, [0, 1, 3]]):
            raise ValueError("paired target moments differ; labels or scoring masks changed")
    a, b = (np.stack([report[seq][1] for seq in ids]) for report in (cand, base))
    score_a, score_b = (float(score_statistics(merge_statistics(value)).mean()) for value in (a, b))
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, len(ids), size=(stop - start, len(ids)))
        deltas[start:stop] = (score_statistics(merge_statistics(a[indices], axis=1)).mean(-1)
                              - score_statistics(merge_statistics(b[indices], axis=1)).mean(-1))
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(deltas, [tail, 1.0 - tail])
    return {"metric_id": METRIC_ID, "delta": score_a - score_b,
            "ci_low": float(low), "ci_high": float(high), "confidence": confidence,
            "candidate_wp": score_a, "baseline_wp": score_b, "sequences": len(ids),
            "n_resamples": n_resamples, "seed": seed,
            "method": "paired sequence-cluster percentile bootstrap with pooled global recomputation",
            "assumption": "sequences are sufficiently independent; model-selection bias is not corrected"}
