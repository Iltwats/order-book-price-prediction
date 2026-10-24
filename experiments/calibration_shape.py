"""Describe what a scored-row calibrator learns: its response along slices of (p0, p1).

    .venv/bin/python -m experiments.calibration_shape
"""
import json

import numpy as np
import torch

from experiments.devdata import load_predictions, load_targets
from experiments.output_calibration import apply, fit_mlp

torch.set_num_threads(8)
raw, groups = load_predictions()
raw = np.asarray(raw, np.float32)
targets, scored = load_targets(groups, store="artifacts/experiments/dev_targets.npz")
need = np.zeros_like(scored)
need[:, 99:] = True
p, y = raw[scored], np.clip(targets[scored], -2, 2)
params = fit_mlp(p, y, hidden=64, epochs=10, seed=0)
report = {"input_quantiles": {}, "slices": {}, "rows": {}}
for k in range(2):
    report["input_quantiles"][f"p{k}"] = dict(zip(["1%", "10%", "50%", "90%", "99%"],
                                                  np.quantile(p[:, k], [.01, .1, .5, .9, .99]).round(3).tolist()))
grid = np.linspace(-1.5, 1.5, 13, dtype=np.float32)
for name, direction in {"p0 only": (1, 0), "p1 only": (0, 1), "p0 = -p1": (1, -1), "p0 = p1": (1, 1)}.items():
    pts = (grid[:, None] * np.asarray(direction, np.float32)[None]).astype(np.float32)
    out = apply(params, pts)
    report["slices"][name] = [[float(a), float(b), float(c), float(d)] for (a, b), (c, d) in zip(pts, out)]
# How do scored and unscored rows differ for the same predictions?  |y|-weighted slope of y on p.
for label, rows in (("scored", scored), ("unscored", need & ~scored)):
    pp, yy = raw[rows], np.clip(targets[rows], -2, 2)
    w = np.abs(yy)
    report["rows"][label] = {
        "count": int(rows.sum()),
        "weighted_slope": [float((w[:, k] * pp[:, k] * yy[:, k]).sum() / (w[:, k] * pp[:, k] ** 2).sum()) for k in range(2)],
        "pred_std": pp.std(0).round(4).tolist(), "target_std": yy.std(0).round(4).tolist(),
        "mean_abs_target": np.abs(yy).mean(0).round(4).tolist(),
        "zero_target_share": (yy == 0).mean(0).round(4).tolist()}
    for k in range(2):
        edges = np.quantile(np.abs(pp[:, k]), [0, .5, .8, .95, .99, 1])
        idx = np.clip(np.searchsorted(edges, np.abs(pp[:, k])) - 1, 0, 4)
        report["rows"][label][f"slope_by_abs_p{k}_bucket"] = [
            float((w[idx == b, k] * pp[idx == b, k] * yy[idx == b, k]).sum() / (w[idx == b, k] * pp[idx == b, k] ** 2).sum())
            for b in range(5)]
print(json.dumps(report, indent=1))
