"""Is current book/trade information missing from the GRU state? A frozen-state head probe.

Hypothesis (2026-10-25): t0 and t1 share a movement signal D = (t0 - t1) / 2 plus
a common adjustment C = (t0 + t1) / 2 that depends on the current market state.
Snapshot regressions on TRAIN found current i0 price-like and trade features
predictive of C, while the GRU's C is weak in the large-tick class. If the
recurrent state does not keep that current-row information, a head that also
sees the current features should beat an equal head that sees only the state.

Stage `features` runs the frozen correlation1024 GRU (Torch import of its
model.onnx) over every dev sequence and keeps, at scored rows, the upper-layer
state (128), the 112 current features, the model's prediction and the targets.

Stage `fit` cross-fits (5 folds by sequence) residual 64-unit GELU heads on top
of the model's prediction with |y|-weighted MSE on clipped targets, on scored
rows, early-stopped on WP of held-out training sequences. All arms share width,
step budget, batch size, learning rate and stopping rule:
  state       h
  state_i0    h + current i0 price-like (22) and trade (8) features
  state_all   h + all 112 current features
  output, output_i0, output_all
              the same with the model's two outputs (clipped to +/-2) in place
              of h, which cannot identify a sequence the way h can
It reports pooled, per-target and per-tick-class WP, paired bootstraps against
the state-only head, and median within-sequence correlations of D and C.

    .venv/bin/python -m experiments.state_head_probe features
    .venv/bin/python -m experiments.state_head_probe fit
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from connectome.global_metric import merge_statistics, score_statistics, sufficient_statistics
from connectome.training import load_torch_model
from experiments.data_semantics import spacing
from experiments.devdata import (DEFAULT_CACHE, DEFAULT_VALID, folds, global_wp, load_predictions, load_targets,
                                 paired_bootstrap)
from utils import FEATURE_COLUMNS

MODEL = Path("models/correlation1024_seed20260912/model.onnx")
I0_PRICE_TRADE = [*range(0, 22), *range(44, 52)]
ARMS = {"state": ("state", None), "state_i0": ("state", I0_PRICE_TRADE), "state_all": ("state", list(range(112))),
        "output": ("output", None), "output_i0": ("output", I0_PRICE_TRADE), "output_all": ("output", list(range(112)))}


def build_features(cache, valid, output, batch=50):
    predictions, groups = load_predictions(cache)
    targets, scored = load_targets(groups, valid, store="artifacts/experiments/dev_targets.npz")
    model = load_torch_model(MODEL).eval()
    parquet = pq.ParquetFile(valid)
    rows = int(scored.sum())
    state = np.lib.format.open_memmap(output / "state.npy", "w+", np.float32, (rows, 128))
    current = np.lib.format.open_memmap(output / "current.npy", "w+", np.float32, (rows, 112))
    p, seq = np.empty((rows, 2), np.float32), np.empty(rows, np.int32)
    at, worst, start = 0, 0.0, time.time()
    for first in range(0, len(groups), batch):
        index = range(first, min(first + batch, len(groups)))
        x = np.stack([np.column_stack([parquet.read_row_group(groups[i], columns=FEATURE_COLUMNS)[c].to_numpy()
                                       for c in FEATURE_COLUMNS]) for i in index]).astype(np.float32)
        with torch.no_grad():
            _, upper, _, _ = model.encode(torch.from_numpy(x).transpose(0, 1))  # GRU input is [time, batch, features]
            predicted = model.head(upper).transpose(0, 1).numpy()
            upper = upper.transpose(0, 1).numpy()
        for j, i in enumerate(index):
            keep = scored[i]
            n = int(keep.sum())
            state[at:at + n], current[at:at + n], p[at:at + n], seq[at:at + n] = upper[j][keep], x[j][keep], predicted[j][keep], i
            worst = max(worst, float(np.abs(predicted[j][keep] - predictions[i][keep]).max()) if n else 0.0)
            at += n
        print(f"{index[-1] + 1}/{len(groups)} sequences, {time.time() - start:.0f}s, max |torch - cache| {worst:.2e}",
              flush=True)
    state.flush()
    current.flush()
    classes = np.round([spacing(t) for t in targets], 4)
    np.savez(output / "rows.npz", p=p, y=targets[scored], seq=seq, classes=classes, max_abs_diff_vs_cache=worst)


def rows_wp(y, pred, seq):
    """Global WP of rows that are grouped by sequence in ascending order."""
    edges = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
    return global_wp(sequence_stats(y, pred, zip(edges[:-1], edges[1:])))


def fit_head(state, current, arm, p, y, seq, train, test, seed, hidden=64, max_epochs=3, batch=8192, lr=1e-3,
             every=32, patience=8):
    """Residual head q + W2 gelu(W1 z + b1) + b2, q = clip(p, -2, 2), on standardised z = [state or q, current[:, columns]].

    The base is clipped like the shipped calibrator's: the metric clips at 2, and
    an unclipped base let the ~0.3% of rows with |p| > 2 dominate the loss (first
    early-stopped run: the outputs-only head fell below the raw model).

    Rows within a sequence are nearly duplicates and the state identifies its
    sequence, so an unstopped head memorises training-sequence levels (first
    run: held-out WP fell 0.648 -> 0.629 in four epochs while training WP rose).
    One fifth of the training sequences is therefore held out; WP on them is
    checked every `every` steps, and the best checkpoint (step 0 = the
    unchanged model) is kept.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sequences = np.unique(seq[train])
    held = np.isin(seq[train], rng.choice(sequences, len(sequences) // 5, replace=False))
    fitting, inner = train[~held], train[held]

    base, columns = arm
    q = np.clip(p, -2, 2)

    def inputs(index):
        parts = [state[index] if base == "state" else q[index]]
        return np.concatenate(parts + ([current[index][:, columns]] if columns else []), 1)

    sample = inputs(np.sort(rng.choice(fitting, min(len(fitting), 200_000), replace=False)))
    mu, sd = sample.mean(0), sample.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(len(mu), hidden), torch.nn.GELU(), torch.nn.Linear(hidden, 2))
    torch.nn.init.zeros_(net[2].weight)
    torch.nn.init.zeros_(net[2].bias)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    target = np.clip(y, -2, 2)

    def predict(index):
        out = np.empty((len(index), 2), np.float32)
        with torch.no_grad():
            for start in range(0, len(index), 1 << 18):
                part = index[start:start + (1 << 18)]
                out[start:start + len(part)] = q[part] + net(torch.from_numpy((inputs(part) - mu) / sd)).numpy()
        return out

    best = {"step": 0, "inner_wp": rows_wp(y[inner], predict(inner), seq[inner]),
            "weights": {k: v.clone() for k, v in net.state_dict().items()}}
    start_wp, step, stale = best["inner_wp"], 0, 0
    for _ in range(max_epochs):
        for index in np.array_split(rng.permutation(fitting), max(1, len(fitting) // batch)):
            index = np.sort(index)
            out = torch.from_numpy(q[index]) + net(torch.from_numpy((inputs(index) - mu) / sd))
            t = torch.from_numpy(target[index])
            w = t.abs()
            loss = (w * (out - t) ** 2).sum() / w.sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            if step % every == 0:
                score = rows_wp(y[inner], predict(inner), seq[inner])
                if score > best["inner_wp"]:
                    best.update(step=step, inner_wp=score, weights={k: v.clone() for k, v in net.state_dict().items()})
                    stale = 0
                else:
                    stale += 1
            if stale >= patience:
                break
        if stale >= patience:
            break
    net.load_state_dict(best["weights"])
    return predict(test), {"best_step": best["step"], "steps_run": step, "inner_wp_start": start_wp,
                           "inner_wp_best": best["inner_wp"]}


def sequence_stats(y, pred, bounds):
    return np.stack([sufficient_statistics(y[a:b], pred[a:b]) for a, b in bounds])


def component_correlations(y, pred, bounds, classes, minimum=50):
    """Median within-sequence correlation of D = (t0 - t1)/2 and C = (t0 + t1)/2, clipped, per tick class."""
    y, pred = np.clip(y, -2, 2), np.clip(pred, -2, 2)
    out = {}
    for name, sign in (("D", -1), ("C", 1)):
        truth, guess = (y[:, 0] + sign * y[:, 1]) / 2, (pred[:, 0] + sign * pred[:, 1]) / 2
        corr = np.array([np.corrcoef(truth[a:b], guess[a:b])[0, 1]
                         if b - a >= minimum and truth[a:b].std() > 0 and guess[a:b].std() > 0 else np.nan
                         for a, b in bounds])
        out[name] = {f"{c:.4f}": float(np.nanmedian(corr[classes == c])) for c in np.unique(classes)}
    return out


def fit(output, seeds, arms):
    rows = np.load(output / "rows.npz")
    p, y, seq, classes = rows["p"], rows["y"], rows["seq"], rows["classes"]
    state, current = np.load(output / "state.npy", mmap_mode="r"), np.load(output / "current.npy", mmap_mode="r")
    state, current = np.ascontiguousarray(state), np.ascontiguousarray(current)
    n = len(classes)
    edges = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]
    bounds = np.zeros((n, 2), int)
    bounds[seq[edges[:-1]]] = np.c_[edges[:-1], edges[1:]]  # sequences without scored rows keep (0, 0)
    fold = folds(n)[seq]
    print(f"{len(seq):,} scored rows, max |torch - cache| {float(rows['max_abs_diff_vs_cache']):.2e}", flush=True)

    predictions, stopping = {"raw": p}, {}
    for name in arms:
        for seed in seeds:
            began, out, key = time.time(), np.empty_like(p), f"{name}_seed{seed}"
            stopping[key] = []
            for k in range(5):
                test = np.flatnonzero(fold == k)
                out[test], info = fit_head(state, current, ARMS[name], p, y, seq, np.flatnonzero(fold != k), test, seed)
                stopping[key].append(info)
                print(f"  {key} fold {k}: {json.dumps(info)}", flush=True)
            predictions[key] = out
            print(f"{key}: WP {global_wp(sequence_stats(y, out, bounds)):.5f} ({time.time() - began:.0f}s)", flush=True)

    stats = {name: sequence_stats(y, pred, bounds) for name, pred in predictions.items()}
    results = {"scored_rows": int(len(seq)), "max_abs_diff_vs_cache": float(rows["max_abs_diff_vs_cache"]),
               "folds": "5 by sequence, seed 20261024",
               "head": "residual 64-unit GELU, Adam 1e-3, batch 8192, at most 3 epochs; early stopping on WP of 20% "
                       "held-out training sequences, checked every 32 steps, patience 8",
               "arms": {}}
    for name, s in stats.items():
        entry = {"wp": global_wp(s), "wp_per_target": score_statistics(merge_statistics(s)).tolist(),
                 "wp_by_class": {f"{c:.4f}": global_wp(s[classes == c]) for c in np.unique(classes)},
                 "component_correlation": component_correlations(y, predictions[name], bounds, classes)}
        if name != "raw":
            entry["early_stopping"] = stopping[name]
            entry["vs_raw"] = paired_bootstrap(s, stats["raw"])
            control = f"{ARMS[name.rsplit('_seed', 1)[0]][0]}_seed{name.rsplit('seed', 1)[1]}"
            if control in stats and control != name:
                entry["vs_same_base_head"] = paired_bootstrap(s, stats[control])
                entry["vs_same_base_head_by_class"] = {f"{c:.4f}": paired_bootstrap(s[classes == c], stats[control][classes == c])
                                                   for c in np.unique(classes)}
        results["arms"][name] = entry
        print(name, json.dumps({k: v for k, v in entry.items() if k not in ("vs_same_base_head_by_class", "early_stopping")}),
              flush=True)
    tag = "_".join(map(str, seeds)) + ("" if set(arms) == set(ARMS) else "_" + "_".join(arms))
    (output / f"results_seed{tag}.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(output / f"sequence_statistics_seed{tag}.npz", **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["features", "fit"])
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--output", default="artifacts/experiments/state_head_probe")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.stage == "features":
        build_features(args.cache, args.valid, output)
    else:
        fit(output, [int(s) for s in args.seeds.split(",")], args.arms.split(","))


if __name__ == "__main__":
    main()
