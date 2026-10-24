"""Does fast/slow causal context add signal on top of the cached model outputs?

Stage `features` builds causal context at every scored row and a 5% sample of
unscored rows: six input-group activity series, |p| and p, each smoothed with
exponential half-lives of 16 and 1,024 rows, plus a0-a7. Stage `fit` compares
cross-fitted |y|-weighted linear and residual-MLP corrections with and without
that context. This tests the multiscale causal-summary hypothesis without
the GRU hidden state or end-to-end training.

    .venv/bin/python -m experiments.context_probe features
    .venv/bin/python -m experiments.context_probe fit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from connectome.global_metric import sufficient_statistics
from experiments.devdata import (DEFAULT_CACHE, DEFAULT_VALID, WARMUP, folds, global_wp, load_predictions,
                                 paired_bootstrap)
from utils import FEATURE_COLUMNS

GROUPS = [(0, 22), (22, 44), (44, 52), (52, 74), (74, 96), (96, 104)]  # i0/i1 price, volume, trade columns
HALF_LIVES = (16, 1024)


def build_features(cache, valid, output):
    predictions, groups = load_predictions(cache)
    parquet = pq.ParquetFile(valid)
    rng = np.random.default_rng(7)
    out = {k: [] for k in ("y", "p", "ema", "a", "seq", "step", "scored")}
    start = time.time()
    for first in range(0, len(groups), 100):
        index = range(first, min(first + 100, len(groups)))
        x = np.stack([np.column_stack([parquet.read_row_group(groups[i], columns=FEATURE_COLUMNS)[c].to_numpy()
                                       for c in FEATURE_COLUMNS]) for i in index]).astype(np.float32)
        tables = [parquet.read_row_group(groups[i], columns=["t0", "t1", "is_scored"]) for i in index]
        y = np.stack([np.c_[t["t0"].to_numpy(), t["t1"].to_numpy()] for t in tables]).astype(np.float32)
        scored = np.stack([t["is_scored"].to_numpy() for t in tables])
        scored[:, :WARMUP] = False
        p = np.asarray(predictions[first:first + len(index)], dtype=np.float32)
        change = np.abs(np.diff(x, axis=1, prepend=x[:, :1]))
        activity = np.log1p(np.stack([change[..., a:b].mean(-1) for a, b in GROUPS], -1))
        series = np.concatenate([activity, np.abs(p), p], -1)
        smoothed = []
        for half_life in HALF_LIVES:
            alpha = 1 - 2 ** (-1 / half_life)
            mean, ema = series[:, 0].copy(), np.empty_like(series)
            for t in range(series.shape[1]):
                mean += alpha * (series[:, t] - mean)
                ema[:, t] = mean
            smoothed.append(ema)
        ema = np.concatenate(smoothed, -1)
        sample = ~scored & (np.arange(20_000) >= WARMUP) & (rng.random(scored.shape) < 0.05)
        for rows, flag in ((scored, True), (sample, False)):
            b, t = np.nonzero(rows)
            for key, value in (("y", y[b, t]), ("p", p[b, t]), ("ema", ema[b, t]), ("a", x[b, t, 104:112]),
                               ("seq", b + first), ("step", t), ("scored", np.full(len(b), flag))):
                out[key].append(value)
        print(f"{index[-1] + 1}/{len(groups)} sequences, {time.time() - start:.0f}s", flush=True)
    np.savez(output / "ctx_dev.npz", **{k: np.concatenate(v) for k, v in out.items()})


def linear_arm(d, fold, k, parts):
    """Ridge-stabilised |y|-weighted least squares with context and context x prediction terms."""
    p, y, sc = d["p"], np.clip(d["y"], -2, 2), d["scored"]
    cols = [p[:, :1], p[:, 1:], np.ones((len(p), 1))]
    for z in parts:
        cols += [z, z * p[:, k:k + 1]]
    x = np.concatenate(cols, 1)
    pred = np.empty(len(x))
    for f in range(5):
        train, test = sc & (fold != f), fold == f
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-9
        mu[:3], sd[:3] = 0, 1
        xs = (x - mu) / sd
        w = np.abs(y[train, k])
        gram = (xs[train] * w[:, None]).T @ xs[train] + 1e-3 * w.sum() / len(w) * np.eye(x.shape[1])
        pred[test] = xs[test] @ np.linalg.solve(gram, (xs[train] * w[:, None]).T @ y[train, k])
    return pred


def mlp_arm(d, fold, parts):
    """Residual 32-unit GELU network on standardised [p, context], zero-initialised output."""
    torch.manual_seed(0)
    p, y, sc = d["p"].astype(np.float32), np.clip(d["y"], -2, 2).astype(np.float32), d["scored"]
    x = np.concatenate([p, *parts], 1).astype(np.float32)
    pred = np.empty_like(p)
    for f in range(5):
        train, test = sc & (fold != f), fold == f
        mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
        xt, yt, pt = torch.from_numpy((x - mu) / sd), torch.from_numpy(y), torch.from_numpy(p)
        net = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 32), torch.nn.GELU(), torch.nn.Linear(32, 2))
        torch.nn.init.zeros_(net[2].weight)
        torch.nn.init.zeros_(net[2].bias)
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        index = np.nonzero(train)[0]
        for epoch in range(3):
            for b in np.array_split(np.random.default_rng(epoch + f).permutation(index), len(index) // 8192):
                out, w = pt[b] + net(xt[b]), yt[b].abs()
                loss = (w * (out - yt[b]) ** 2).sum() / w.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            pred[test] = (pt[test] + net(xt[test])).numpy()
    return pred


def fit(output):
    d = dict(np.load(output / "ctx_dev.npz"))
    n = int(d["seq"].max()) + 1
    fold = folds(n)[d["seq"]]
    fast, slow, a = d["ema"][:, :10], d["ema"][:, 10:], d["a"]
    arms = {"raw": d["p"]}
    for name, parts in {"linear_affine_control": [], "linear_fast": [fast], "linear_slow": [slow],
                        "linear_both": [fast, slow], "linear_both_a": [fast, slow, a]}.items():
        arms[name] = np.stack([linear_arm(d, fold, k, parts) for k in range(2)], 1)
    for name, parts in {"mlp_control": [], "mlp_fast": [fast], "mlp_slow": [slow],
                        "mlp_fast_slow_a": [fast, slow, a], "mlp_a": [a]}.items():
        arms[name] = mlp_arm(d, fold, parts)
    sc = d["scored"]

    def stats(pred):
        out = np.zeros((n, 2, 6))
        ids = d["seq"][sc]
        for s in np.unique(ids):
            rows = ids == s
            out[s] = sufficient_statistics(d["y"][sc][rows], pred[sc][rows])
        return out

    stats_by_arm = {name: stats(pred) for name, pred in arms.items()}
    results = {}
    for name, pred in arms.items():
        results[name] = {"scored_wp": global_wp(stats_by_arm[name]),
                         "unscored_sample_wp": global_wp(sufficient_statistics(d["y"][~sc], pred[~sc])[None])}
        for control in ("raw", "linear_affine_control" if name.startswith("linear") else "mlp_control"):
            if control != name and name != "raw":
                results[name][f"vs_{control}"] = paired_bootstrap(stats_by_arm[name], stats_by_arm[control])
        print(name, json.dumps(results[name]), flush=True)
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["features", "fit"])
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--output", default="artifacts/experiments/context_probe")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    build_features(args.cache, args.valid, output) if args.stage == "features" else fit(output)


if __name__ == "__main__":
    main()
