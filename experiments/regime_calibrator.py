"""Causal sequence feature extractor and regime-aware head/calibrator OOF evaluation.

Strictly causal within each 20,000-step sequence (resetting at step 0, updating
on every warm-up and prediction step) and evaluated with 5-fold sequence-grouped
cross-fitting (fold seed 20261024) against both raw `correlation1024` (0.67209)
and the shipped `mlp64_clip2` calibrator (0.67392).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from connectome.global_metric import merge_statistics, score_statistics, sufficient_statistics
from experiments.devdata import (DEFAULT_CACHE, DEFAULT_VALID, WARMUP, folds, global_wp,
                                 load_predictions, load_targets, paired_bootstrap)
from utils import FEATURE_COLUMNS

EMA_C_CODE = r"""
#include <cmath>
#include <cstddef>

extern "C" {
void causal_ema_2d(const float* __restrict x, float* __restrict out,
                   int T, int D, float alpha) {
    const float one_m = 1.0f - alpha;
    for (int d = 0; d < D; ++d) out[d] = x[d];
    for (int t = 1; t < T; ++t) {
        const float* xr = x + t * D;
        const float* pr = out + (t - 1) * D;
        float* yr = out + t * D;
        for (int d = 0; d < D; ++d) {
            yr[d] = one_m * pr[d] + alpha * xr[d];
        }
    }
}
}
"""


def _load_ema_lib():
    td = tempfile.mkdtemp(prefix="connectome-ema-")
    src = Path(td) / "ema.cpp"
    ext = ".dylib" if Path("/System/Library").exists() else ".so"
    lib_path = Path(td) / f"libema{ext}"
    src.write_text(EMA_C_CODE)
    link = ["-dynamiclib"] if ext == ".dylib" else ["-shared", "-fPIC"]
    subprocess.run(["c++", "-std=c++17", "-O3", *link, str(src), "-o", str(lib_path)], check=True)
    lib = ctypes.CDLL(str(lib_path))
    lib.causal_ema_2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_float]
    return lib


_EMA_LIB = None


def causal_ema(x: np.ndarray, half_life: float) -> np.ndarray:
    global _EMA_LIB
    if _EMA_LIB is None:
        _EMA_LIB = _load_ema_lib()
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    alpha = float(1.0 - 2.0 ** (-1.0 / half_life))
    _EMA_LIB.causal_ema_2d(x.ctypes.data, out.ctypes.data, x.shape[0], x.shape[1], alpha)
    return out


def sequence_causal_features(x: np.ndarray, p: np.ndarray | None = None) -> np.ndarray:
    """Compute strictly causal per-row features [T, F] from [T, 112] features (and optional [T, 2] raw preds).

    Every statistic at row t uses ONLY rows 0..t of the same sequence.
    """
    T = len(x)
    i0_bp, i0_ap = x[:, 0:11], x[:, 11:22]
    i0_bv, i0_av = x[:, 22:33], x[:, 33:44]
    i0_dp, i0_dv = x[:, 44:48], x[:, 48:52]
    i1_bp, i1_ap = x[:, 52:63], x[:, 63:74]
    i1_bv, i1_av = x[:, 74:85], x[:, 85:96]
    i1_dp, i1_dv = x[:, 96:100], x[:, 100:104]
    a = x[:, 104:112]

    # Permutation-invariant order-book & trade snapshot [T, 32]
    i0_bv_sum = np.abs(i0_bv).sum(1)
    i0_av_sum = np.abs(i0_av).sum(1)
    i1_bv_sum = np.abs(i1_bv).sum(1)
    i1_av_sum = np.abs(i1_av).sum(1)

    snap = np.column_stack([
        a,                                                                    # 0..7   (8)
        i0_bp.mean(1), i0_bp.std(1), i0_bp.max(1), i0_bp.min(1),              # 8..11  (4)
        i0_ap.mean(1), i0_ap.std(1), i0_ap.max(1), i0_ap.min(1),              # 12..15 (4)
        i0_ap.mean(1) - i0_bp.mean(1),                                        # 16     (1)
        i0_bv.mean(1), i0_bv.max(1), i0_av.mean(1), i0_av.max(1),             # 17..20 (4)
        (i0_bv.sum(1) - i0_av.sum(1)) / (i0_bv_sum + i0_av_sum + 1e-5),       # 21     (1)
        i0_dp, i0_dv,                                                         # 22..29 (8)
        i1_bp.mean(1), i1_ap.mean(1),                                         # 30..31 (2)
        (i1_bv.sum(1) - i1_av.sum(1)) / (i1_bv_sum + i1_av_sum + 1e-5),       # 32     (1)
        i1_dp.mean(1), i1_dv.mean(1),                                         # 33..34 (2)
    ]).astype(np.float32)  # [T, 35]

    # 1-step change on key dynamic channels (a + i0/i1 means/imbalance/trades)
    dx1 = np.vstack([np.zeros((1, snap.shape[1]), np.float32), np.diff(snap, axis=0)])
    # Multi-scale causal EMAs of snapshot
    ema16 = causal_ema(snap, 16.0)
    ema128 = causal_ema(snap, 128.0)
    ema1024 = causal_ema(snap, 1024.0)

    dev16 = snap - ema16
    dev128 = snap - ema128
    slow_trend = ema16 - ema1024

    # Causal sequence-level running fingerprint from step 0..t (identifies tick regime without future leak)
    steps = np.arange(1, T + 1, dtype=np.float32)[:, None]
    run_a_mean = np.cumsum(a.astype(np.float64), axis=0) / steps
    p_diff = np.vstack([np.zeros((1, 22), np.float32), np.abs(np.diff(x[:, :22], axis=0))])
    any_book_move = (p_diff > 0).any(1).astype(np.float64)[:, None]
    move_mag = p_diff.sum(1).astype(np.float64)[:, None] / np.maximum((p_diff > 0).sum(1)[:, None], 1.0)
    run_move_rate = np.cumsum(any_book_move, axis=0) / steps
    run_move_mag = np.cumsum(move_mag, axis=0) / np.maximum(np.cumsum(any_book_move, axis=0), 1.0)
    step_feat = np.log1p(np.arange(T, dtype=np.float32))[:, None] / 10.0

    regime = np.hstack([
        run_a_mean.astype(np.float32),     # 8
        run_move_rate.astype(np.float32),  # 1
        np.log1p(run_move_mag).astype(np.float32),  # 1
        ema1024[:, :8],                    # 8 (slow a0..a7)
        step_feat,                         # 1
    ])  # [T, 19]

    parts = [snap, dx1[:, 8:], dev16[:, 8:], dev128[:, 8:], slow_trend[:, 8:], regime]
    if p is not None:
        q = np.clip(p, -2.0, 2.0).astype(np.float32)
        p_abs = np.abs(q)
        p_ema8 = causal_ema(q, 8.0)
        p_ema64 = causal_ema(q, 64.0)
        p_ema512 = causal_ema(q, 512.0)
        pa_ema64 = causal_ema(p_abs, 64.0)
        pa_ema512 = causal_ema(p_abs, 512.0)
        p_feats = np.column_stack([
            q,
            q[:, 0:1] - q[:, 1:2],
            q[:, 0:1] + q[:, 1:2],
            p_abs,
            q - p_ema8,
            q - p_ema64,
            p_ema8 - p_ema512,
            pa_ema64,
            pa_ema512,
        ]).astype(np.float32)  # [T, 16]
        parts = [p_feats] + parts
    return np.hstack(parts)


def build_cache(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    raw_preds, groups = load_predictions(DEFAULT_CACHE)
    raw_preds = np.asarray(raw_preds, np.float32)
    targets, scored = load_targets(groups, DEFAULT_VALID, store="artifacts/experiments/dev_targets.npz")
    pf = pq.ParquetFile(DEFAULT_VALID)
    total_rows = int(scored.sum())

    # Probe feature count on sequence 0
    tb0 = pf.read_row_group(groups[0], columns=FEATURE_COLUMNS)
    x0 = np.column_stack([tb0[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    f0 = sequence_causal_features(x0, raw_preds[0])
    feat_dim = f0.shape[1]
    print(f"Extracting {feat_dim} causal features across {len(groups)} dev sequences ({total_rows:,} scored rows)...", flush=True)

    feats_mm = np.lib.format.open_memmap(output / "causal_scored_features.npy", "w+", np.float32, (total_rows, feat_dim))
    at = 0
    t0 = time.time()
    for idx, g in enumerate(groups):
        tb = pf.read_row_group(g, columns=FEATURE_COLUMNS)
        x = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        f = sequence_causal_features(x, raw_preds[idx])
        keep = scored[idx]
        n = int(keep.sum())
        feats_mm[at:at + n] = f[keep]
        at += n
        if (idx + 1) % 200 == 0 or idx + 1 == len(groups):
            print(f"  {idx + 1}/{len(groups)} sequences ({time.time() - t0:.1f}s)", flush=True)
    feats_mm.flush()


class ResidualCalibratorNet(torch.nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128, dropout: float = 0.15):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.LayerNorm(hidden // 2),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden // 2, 2),
        )
        torch.nn.init.zeros_(self.net[-1].weight)
        torch.nn.init.zeros_(self.net[-1].bias)

    def forward(self, z):
        return self.net(z)


def run_oof(output: Path, threads: int = 6):
    torch.set_num_threads(threads)
    rows = np.load("artifacts/experiments/state_head_probe/rows.npz")
    p = rows["p"]
    y = rows["y"]
    seq = rows["seq"]
    classes = rows["classes"]
    n_seq = len(classes)
    feats = np.load(output / "causal_scored_features.npy", mmap_mode="r")
    state = np.load("artifacts/experiments/state_head_probe/state.npy", mmap_mode="r")

    edges = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]

    def seq_stats(pred):
        return np.stack([sufficient_statistics(y[edges[s]:edges[s + 1]], pred[edges[s]:edges[s + 1]])
                         for s in range(n_seq)])

    raw_st = seq_stats(p)
    shipped_st = np.load("artifacts/experiments/output_calibration/sequence_statistics_clip2.npz")[
        "mlp64_scored_30epochs_clip2__scored"
    ]
    print(f"Raw WP:     {global_wp(raw_st):.5f}")
    print(f"Shipped WP: {global_wp(shipped_st):.5f}")

    fold = folds(n_seq)
    q = np.clip(p, -2.0, 2.0)
    yc = np.clip(y, -2.0, 2.0)

    # Feature slices inside causal_scored_features (178 cols):
    # 0..16: p_feats (16)
    # 16..51: snap (35: 8 a + 27 book/trades)
    # 51..159: dx1, dev16, dev128, slow_trend (4 * 27 = 108)
    # 159..178: regime (19)
    configs = [
        ("causal_snap_regime_mlp128", False, list(range(51)) + list(range(159, 178)), 128, 0.15, 1e-3, 5e-4, 12),
        ("causal_full_mlp128", False, list(range(feats.shape[1])), 128, 0.20, 1e-3, 1e-3, 12),
        ("causal_full_plus_state_mlp128", True, list(range(feats.shape[1])), 128, 0.25, 8e-4, 2e-3, 8),
    ]

    results = {}
    for name, use_state, cols, hidden, drop, lr, wd, max_epochs in configs:
        t0 = time.time()
        oof = np.empty_like(p)
        for k in range(5):
            outer_tr_seqs = np.nonzero(fold != k)[0]
            rng = np.random.default_rng(20261111 + k)
            val_seqs = rng.choice(outer_tr_seqs, size=len(outer_tr_seqs) // 6, replace=False)
            fit_seqs = np.setdiff1d(outer_tr_seqs, val_seqs)

            fit_mask = np.isin(seq, fit_seqs)
            val_mask = np.isin(seq, val_seqs)
            te_mask = fold[seq] == k

            def get_z(mask):
                z = np.asarray(feats[mask][:, cols], dtype=np.float32)
                if use_state:
                    z = np.hstack([z, np.asarray(state[mask], dtype=np.float32)])
                return z

            z_fit = get_z(fit_mask)
            mu = z_fit.mean(0)
            sd = z_fit.std(0) + 1e-5
            z_fit = (z_fit - mu) / sd
            z_val = (get_z(val_mask) - mu) / sd
            z_te = (get_z(te_mask) - mu) / sd

            q_fit, y_fit = q[fit_mask], yc[fit_mask]
            q_val, y_val, seq_val = q[val_mask], y[val_mask], seq[val_mask]
            val_edges = np.r_[0, np.flatnonzero(np.diff(seq_val)) + 1, len(seq_val)]

            def eval_val(pred_val):
                st_v = np.stack([sufficient_statistics(y_val[val_edges[i]:val_edges[i + 1]],
                                                       pred_val[val_edges[i]:val_edges[i + 1]])
                                 for i in range(len(val_edges) - 1)])
                return global_wp(st_v)

            # Ensemble 2 seeds per fold for variance reduction + stable early stopping
            fold_pred = np.zeros((te_mask.sum(), 2), dtype=np.float32)
            for s_idx in range(2):
                torch.manual_seed(100 * k + s_idx)
                s_rng = np.random.default_rng(100 * k + s_idx)
                net = ResidualCalibratorNet(z_fit.shape[1], hidden=hidden, dropout=drop)
                opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
                best_wp = eval_val(q_val)
                best_state = {kk: vv.clone() for kk, vv in net.state_dict().items()}

                z_fit_t = torch.from_numpy(z_fit)
                q_fit_t = torch.from_numpy(q_fit)
                y_fit_t = torch.from_numpy(y_fit)
                w_fit_t = y_fit_t.abs()
                z_val_t = torch.from_numpy(z_val)

                for ep in range(max_epochs):
                    net.train()
                    perm = s_rng.permutation(len(z_fit))
                    for st_i in range(0, len(z_fit), 16384):
                        idx = perm[st_i:st_i + 16384]
                        pred = q_fit_t[idx] + net(z_fit_t[idx])
                        w = w_fit_t[idx]
                        loss = (w * (pred - y_fit_t[idx]) ** 2).sum() / w.sum()
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    sched.step()
                    net.eval()
                    with torch.no_grad():
                        pv = q_val + net(z_val_t).numpy()
                    v_wp = eval_val(pv)
                    if v_wp > best_wp:
                        best_wp = v_wp
                        best_state = {kk: vv.clone() for kk, vv in net.state_dict().items()}

                net.load_state_dict(best_state)
                net.eval()
                with torch.no_grad():
                    fold_pred += 0.5 * (q[te_mask] + net(torch.from_numpy(z_te)).numpy())
            oof[te_mask] = fold_pred

        st = seq_stats(oof)
        vs_raw = paired_bootstrap(st, raw_st)
        vs_ship = paired_bootstrap(st, shipped_st)
        by_class = {str(c): round(global_wp(st[classes == c]), 5) for c in np.unique(classes)}
        entry = {
            "wp": global_wp(st),
            "vs_raw": vs_raw,
            "vs_shipped": vs_ship,
            "by_class": by_class,
            "seconds": round(time.time() - t0, 1),
        }
        results[name] = entry
        print(f"{name:32s}: WP={entry['wp']:.5f} | vs_raw={vs_raw['delta']:+.5f} | "
              f"vs_shipped={vs_ship['delta']:+.5f} [{vs_ship['ci_low']:+.5f}, {vs_ship['ci_high']:+.5f}] | "
              f"by_class={by_class} ({entry['seconds']}s)", flush=True)

    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["features", "fit", "all"])
    parser.add_argument("--output", default="artifacts/experiments/regime_calibrator")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    out = Path(args.output)
    if args.stage in ("features", "all"):
        build_cache(out)
    if args.stage in ("fit", "all"):
        run_oof(out, threads=args.threads)


if __name__ == "__main__":
    main()
