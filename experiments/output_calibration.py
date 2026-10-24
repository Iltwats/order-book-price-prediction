"""Cross-fitted output calibration of cached development predictions.

Each arm maps the model's two outputs p to p + f(p) and is fitted on four fifths
of the development sequences, then scored on the held-out fifth. The loss is the
pooled |y|-weighted MSE on clipped targets. No model inference runs here.

    .venv/bin/python -m experiments.output_calibration --output artifacts/experiments/output_calibration
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.devdata import (DEFAULT_CACHE, DEFAULT_VALID, WARMUP, folds, global_wp, load_predictions,
                                 load_targets, paired_bootstrap, sequence_statistics)

ARMS = {
    "affine_scored": dict(kind="affine", rows="scored"),
    "mlp32_scored_seed0": dict(kind="mlp", rows="scored", hidden=32, epochs=3, seed=0),
    "mlp32_scored_seed1": dict(kind="mlp", rows="scored", hidden=32, epochs=3, seed=1),
    "mlp32_scored_seed2": dict(kind="mlp", rows="scored", hidden=32, epochs=3, seed=2),
    "mlp32_all_rows": dict(kind="mlp", rows="all", hidden=32, epochs=1, seed=0),
    "mlp32_separable_scored": dict(kind="mlp", rows="scored", hidden=32, epochs=3, seed=0, separable=True),
    "mlp64_scored_10epochs": dict(kind="mlp", rows="scored", hidden=64, epochs=10, seed=0),
    # Follow-up sweep: is the larger 10-epoch gain seed-stable, and is it capacity or training length?
    "mlp64_scored_10epochs_seed1": dict(kind="mlp", rows="scored", hidden=64, epochs=10, seed=1),
    "mlp64_scored_10epochs_seed2": dict(kind="mlp", rows="scored", hidden=64, epochs=10, seed=2),
    "mlp32_scored_10epochs": dict(kind="mlp", rows="scored", hidden=32, epochs=10, seed=0),
    "mlp64_scored_3epochs": dict(kind="mlp", rows="scored", hidden=64, epochs=3, seed=0),
    "mlp64_scored_30epochs": dict(kind="mlp", rows="scored", hidden=64, epochs=30, seed=0),
    "mlp64_scored_30epochs_seed1": dict(kind="mlp", rows="scored", hidden=64, epochs=30, seed=1),
    "mlp64_scored_30epochs_seed2": dict(kind="mlp", rows="scored", hidden=64, epochs=30, seed=2),
    # Inputs clipped to the metric's +/-2 range, so the network never extrapolates beyond its support.
    "mlp64_scored_30epochs_clip2": dict(kind="mlp", rows="scored", hidden=64, epochs=30, seed=0, clip=2.0),
    "mlp64_scored_30epochs_clip2_seed1": dict(kind="mlp", rows="scored", hidden=64, epochs=30, seed=1, clip=2.0),
    # Nonparametric ceiling for any function of the two outputs: shrunk 2-D binned residual means.
    "binned40_scored": dict(kind="binned", rows="scored", bins=40),
    "binned80_scored": dict(kind="binned", rows="scored", bins=80),
    "binned40_all_rows": dict(kind="binned", rows="all", bins=40),
}
PRIMARY = "mlp32_scored_seed0"


def fit_affine(p, y):
    """|y|-weighted least squares of each clipped target on [1, p0, p1]."""
    w = np.abs(y).astype(np.float64)
    design = np.column_stack([np.ones(len(p)), p]).astype(np.float64)
    coefficients = [np.linalg.solve(design.T @ (w[:, k:k + 1] * design), design.T @ (w[:, k] * y[:, k]))
                    for k in range(2)]
    return {"kind": "affine", "coefficients": np.stack(coefficients)}


def fit_binned(p, y, bins=40, shrink=200.0):
    """p + |y|-weighted mean residual in quantile bins of (p0, p1), shrunk toward zero in thin bins."""
    edges = [np.quantile(p[:, k], np.linspace(0, 1, bins + 1)[1:-1]) for k in range(2)]
    cell = np.searchsorted(edges[0], p[:, 0]) * bins + np.searchsorted(edges[1], p[:, 1])
    w = np.abs(y).astype(np.float64)
    weight = np.stack([np.bincount(cell, w[:, k], bins * bins) for k in range(2)], 1)
    residual = np.stack([np.bincount(cell, w[:, k] * (y[:, k] - p[:, k]), bins * bins) for k in range(2)], 1)
    mean_weight = w.mean(0)
    return {"kind": "binned", "edges": edges, "bins": bins,
            "table": (residual / (weight + shrink * mean_weight)).astype(np.float32)}


def fit_mlp(p, y, hidden=32, epochs=3, seed=0, separable=False, batch=8192, lr=1e-3, clip=None):
    """Residual 2->hidden->2 GELU network; zero-initialised output keeps p at the start."""
    torch.manual_seed(seed)
    if clip is not None:
        p = np.clip(p, -clip, clip)
    mu, sd = p.mean(0), p.std(0) + 1e-6
    x, target = torch.from_numpy((p - mu) / sd), torch.from_numpy(y)
    base = torch.from_numpy(p)
    first, second = torch.nn.Linear(2, hidden), torch.nn.Linear(hidden, 2)
    torch.nn.init.zeros_(second.weight)
    torch.nn.init.zeros_(second.bias)
    mask_in = torch.ones(hidden, 2)
    if separable:  # first half of the units sees and feeds only target 0, second half only target 1
        mask_in[:hidden // 2, 1] = 0
        mask_in[hidden // 2:, 0] = 0
    mask_out = mask_in.T.clone()
    optimizer = torch.optim.Adam([*first.parameters(), *second.parameters()], lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for index in np.array_split(rng.permutation(len(p)), max(1, len(p) // batch)):
            index = torch.from_numpy(index)
            hidden_out = torch.nn.functional.gelu(x[index] @ (first.weight * mask_in).T + first.bias)
            out = base[index] + hidden_out @ (second.weight * mask_out).T + second.bias
            w = target[index].abs()
            loss = (w * (out - target[index]) ** 2).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        return {"kind": "mlp", "clip": clip, "mu": mu, "sd": sd,
                "w1": (first.weight * mask_in).numpy().copy(), "b1": first.bias.numpy().copy(),
                "w2": (second.weight * mask_out).numpy().copy(), "b2": second.bias.numpy().copy()}


def apply(params, p):
    """Apply a fitted calibrator to [rows, 2] predictions in bounded chunks."""
    if params["kind"] == "affine":
        c = params["coefficients"]
        return (c[:, 0] + p @ c[:, 1:].T).astype(np.float32)
    if params["kind"] == "binned":
        e, n = params["edges"], params["bins"]
        return p + params["table"][np.searchsorted(e[0], p[:, 0]) * n + np.searchsorted(e[1], p[:, 1])]
    if params.get("clip") is not None:
        p = np.clip(p, -params["clip"], params["clip"])
    out = np.empty_like(p)
    w1, b1, w2, b2 = (torch.from_numpy(params[k]) for k in ("w1", "b1", "w2", "b2"))
    for start in range(0, len(p), 1 << 20):
        chunk = p[start:start + (1 << 20)]
        x = torch.from_numpy((chunk - params["mu"]) / params["sd"])
        with torch.no_grad():
            out[start:start + len(chunk)] = chunk + (torch.nn.functional.gelu(x @ w1.T + b1) @ w2.T + b2).numpy()
    return out


def fit(arm, p, y):
    if arm["kind"] == "affine":
        return fit_affine(p, y)
    if arm["kind"] == "binned":
        return fit_binned(p, y, arm["bins"])
    return fit_mlp(p, y, arm["hidden"], arm["epochs"], arm["seed"], arm.get("separable", False), clip=arm.get("clip"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--output", default="artifacts/experiments/output_calibration")
    parser.add_argument("--arms", default=",".join(list(ARMS)[:7]))
    parser.add_argument("--tag", default="", help="suffix for results files of a follow-up run")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()
    raw, groups = load_predictions(args.cache)
    raw = np.asarray(raw, np.float32)
    targets, scored = load_targets(groups, args.valid, store=output.parent / "dev_targets.npz")
    need = np.zeros_like(scored)
    need[:, WARMUP:] = True
    unscored = need & ~scored
    clipped = np.clip(targets, -2, 2)
    fold = folds(len(groups))
    print(f"loaded {len(groups)} sequences, {scored.sum():,} scored rows ({time.time() - start:.0f}s)", flush=True)

    stats = {"raw": (sequence_statistics(targets, raw, scored), sequence_statistics(targets, raw, unscored))}
    for name in args.arms.split(","):
        arm, began = ARMS[name], time.time()
        rows = scored if arm["rows"] == "scored" else need
        predicted = raw.copy()
        for k in range(5):
            train = fold != k
            params = fit(arm, raw[train][rows[train]], clipped[train][rows[train]])
            test = np.nonzero(fold == k)[0]
            predicted[test] = apply(params, raw[test].reshape(-1, 2)).reshape(len(test), 20_000, 2)
        stats[name] = (sequence_statistics(targets, predicted, scored), sequence_statistics(targets, predicted, unscored))
        print(f"{name:24s} scored {global_wp(stats[name][0]):.5f}  unscored {global_wp(stats[name][1]):.5f}"
              f"  ({time.time() - began:.0f}s)", flush=True)

    results = {"cache": str(args.cache), "sequences": len(groups), "scored_rows": int(scored.sum()),
               "folds": "5 by sequence, seed 20261024", "arms": {}}
    for name, (sc, un) in stats.items():
        entry = {"config": ARMS.get(name, {}), "scored_wp": global_wp(sc), "unscored_wp": global_wp(un)}
        if name != "raw":
            entry["vs_raw"] = paired_bootstrap(sc, stats["raw"][0])
            entry["unscored_vs_raw"] = global_wp(un) - global_wp(stats["raw"][1])
            if PRIMARY in stats and name != PRIMARY:
                entry["vs_primary"] = paired_bootstrap(sc, stats[PRIMARY][0])
        results["arms"][name] = entry
        if "vs_raw" in entry:
            d = entry["vs_raw"]
            print(f"{name:24s} vs raw {d['delta']:+.5f} [{d['ci_low']:+.5f}, {d['ci_high']:+.5f}]")
    (output / f"results{args.tag}.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(output / f"sequence_statistics{args.tag}.npz", **{f"{k}__{part}": v[i] for k, v in stats.items()
                                                   for i, part in enumerate(("scored", "unscored"))})
    print(f"saved {output / f'results{args.tag}.json'} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
