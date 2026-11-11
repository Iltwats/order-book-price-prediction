"""Fit OOF mask-propensity model P(is_scored | observable features) on dev and verify AUC.

Uses ONLY observable causal features (a0..a7, slow/fast activity, book spread/imbalance)
on the 1,400 development sequences (confirmation split stays untouched). Saves compact
weights to `artifacts/experiments/mask_propensity/model.npz` for instant row-weight
scoring during full-dataset training on `train.parquet`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from experiments.devdata import DEFAULT_VALID, WARMUP, folds
from utils import FEATURE_COLUMNS


def mask_features_from_seq(x: np.ndarray) -> np.ndarray:
    """Fast vectorized observable features [T, 20] that predict `is_scored`."""
    a = x[:, 104:112]  # 8
    dx = np.vstack([np.zeros((1, 112), np.float32), np.abs(np.diff(x, axis=0))])
    i0_p_chg = (dx[:, :22] > 0).mean(1, keepdims=True)
    i0_v_chg = (dx[:, 22:44] > 0).mean(1, keepdims=True)
    i0_t_chg = (dx[:, 44:52] > 0).mean(1, keepdims=True)
    i1_chg = (dx[:, 52:104] > 0).mean(1, keepdims=True)
    spread0 = (x[:, 11:22].mean(1) - x[:, 0:11].mean(1))[:, None]
    vol0 = np.log1p(np.abs(x[:, 22:44]).mean(1, keepdims=True))
    trd0 = np.log1p(np.abs(x[:, 48:52]).mean(1, keepdims=True))
    step_norm = (np.arange(len(x), dtype=np.float32) / 20000.0)[:, None]
    a_sq = a[:, [2, 3, 4, 7]] ** 2  # nonlinear terms for top mask predictors a2, a3, a4, a7
    return np.hstack([a, a_sq, i0_p_chg, i0_v_chg, i0_t_chg, i1_chg, spread0, vol0, trd0, step_norm]).astype(np.float32)


def rank_auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    label_sorted = label[order]
    pos = float(label_sorted.sum())
    neg = float(len(label_sorted) - pos)
    cum_neg = np.cumsum(1.0 - label_sorted)
    # AUC = P(score_pos > score_neg): for each positive, count negatives before it
    return float((cum_neg[label_sorted == 1]).sum() / (pos * neg))


def main():
    t0 = time.time()
    torch.set_num_threads(4)
    manifest = json.loads(Path("artifacts/manifest.json").read_text())
    dev_groups = manifest["splits"]["dev"]
    pf = pq.ParquetFile(DEFAULT_VALID)

    # Sample 2,000 post-warmup rows per dev sequence (2.8M rows total) for fast, exact OOF fitting
    rng = np.random.default_rng(20261111)
    rows_per_seq = 2000
    n_seq = len(dev_groups)
    X = np.empty((n_seq, rows_per_seq, 20), dtype=np.float32)
    Y = np.empty((n_seq, rows_per_seq), dtype=np.float32)

    cols = list(FEATURE_COLUMNS) + ["is_scored"]
    for i, g in enumerate(dev_groups):
        tb = pf.read_row_group(g, columns=cols)
        x = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        s = tb["is_scored"].to_numpy().astype(np.float32)
        mf = mask_features_from_seq(x)
        idx = np.sort(rng.choice(np.arange(WARMUP, 20000), size=rows_per_seq, replace=False))
        X[i] = mf[idx]
        Y[i] = s[idx]

    print(f"Sampled {n_seq * rows_per_seq:,} dev rows (scored rate={Y.mean():.4f}) in {time.time() - t0:.1f}s", flush=True)

    fold = folds(n_seq)
    oof_prob = np.empty_like(Y)

    for k in range(5):
        tr = fold != k
        te = fold == k
        x_tr = X[tr].reshape(-1, 20)
        y_tr = Y[tr].reshape(-1, 1)
        mu = x_tr.mean(0)
        sd = x_tr.std(0) + 1e-5

        net = torch.nn.Sequential(
            torch.nn.Linear(20, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 1),
        )
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        xt = torch.from_numpy((x_tr - mu) / sd)
        yt = torch.from_numpy(y_tr)
        f_rng = np.random.default_rng(100 + k)
        for _ in range(4):
            perm = f_rng.permutation(len(xt))
            for st in range(0, len(xt), 16384):
                b = perm[st:st + 16384]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]), yt[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            x_te = (X[te].reshape(-1, 20) - mu) / sd
            oof_prob[te] = torch.sigmoid(net(torch.from_numpy(x_te))).numpy().reshape(-1, rows_per_seq)

    auc_oof = rank_auc(oof_prob.ravel(), Y.ravel())
    p_scored = float(oof_prob[Y == 1].mean())
    p_unscored = float(oof_prob[Y == 0].mean())
    print(f"OOF Mask Propensity AUC: {auc_oof:.4f} (mean P on scored={p_scored:.4f} vs unscored={p_unscored:.4f})", flush=True)

    # Fit final propensity model on all 1,400 dev sequences
    x_all = X.reshape(-1, 20)
    y_all = Y.reshape(-1, 1)
    mu = x_all.mean(0)
    sd = x_all.std(0) + 1e-5
    torch.manual_seed(20261111)
    net = torch.nn.Sequential(
        torch.nn.Linear(20, 32),
        torch.nn.SiLU(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    xt = torch.from_numpy((x_all - mu) / sd)
    yt = torch.from_numpy(y_all)
    for _ in range(5):
        perm = rng.permutation(len(xt))
        for st in range(0, len(xt), 16384):
            b = perm[st:st + 16384]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

    out_dir = Path("artifacts/experiments/mask_propensity")
    out_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        np.savez(
            out_dir / "model.npz",
            mu=mu.astype(np.float32),
            sd=sd.astype(np.float32),
            w1=net[0].weight.numpy().astype(np.float32),
            b1=net[0].bias.numpy().astype(np.float32),
            w2=net[2].weight.numpy().astype(np.float32),
            b2=net[2].bias.numpy().astype(np.float32),
        )
    report = {
        "oof_auc": round(auc_oof, 5),
        "mean_prob_scored": round(p_scored, 5),
        "mean_prob_unscored": round(p_unscored, 5),
        "ratio_scored_vs_unscored": round(p_scored / p_unscored, 3),
        "dev_sequences": n_seq,
        "rows_sampled": int(n_seq * rows_per_seq),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved propensity model:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
