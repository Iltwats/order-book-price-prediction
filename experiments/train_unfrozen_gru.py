"""End-to-end unfrozen multi-layer training on all 10,607 sequences with mask & class weighting.

Key breakthroughs over legacy `gpu_training.py`:
1. Unfreezes ALL layers (`gru0`, `gru1`, and readout + current-row snapshot skip connection).
2. Zero teacher leash (`teacher_weight=0`) and zero parameter anchor (`anchor_weight=0`).
3. Objective combines:
   - Pooled `|y|`-weighted MSE on `clip(y, -2, 2)` (learns exact Bayes-optimal cross-sequence scale)
   - Pooled Global Weighted Pearson correlation loss across the sequence batch (`B=64` sequences)
4. Mask-propensity row weighting (`alpha + (1 - alpha) * P(is_scored | x_t)`) so gradient capacity
   concentrates on scored-like high-activity rows rather than quiet unscored rows.
5. Stratified class-mix sampling matching validation's 4-tick-class distribution.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from connectome.global_metric import merge_statistics, score_statistics, sufficient_statistics
from connectome.training import load_torch_model, seed_everything
from experiments.devdata import DEFAULT_VALID, WARMUP, global_wp, load_targets
from utils import FEATURE_COLUMNS, TARGET_COLUMNS


def compute_book_skip_torch(x: torch.Tensor) -> torch.Tensor:
    """Fast differentiable/exact snapshot features from raw [..., 112] tensor -> [..., 38].

    Permutation-invariant across the 11 unordered bid/ask columns, plus price-sorted best quotes!
    """
    i0_bp = x[..., 0:11]
    i0_ap = x[..., 11:22]
    i0_bv = x[..., 22:33]
    i0_av = x[..., 33:44]
    i0_dp = x[..., 44:48]
    i0_dv = x[..., 48:52]
    i1_bp = x[..., 52:63]
    i1_ap = x[..., 63:74]
    i1_bv = x[..., 74:85]
    i1_av = x[..., 85:96]
    i1_dp = x[..., 96:100]
    i1_dv = x[..., 100:104]
    a = x[..., 104:112]

    i0_bp_mean = i0_bp.mean(dim=-1, keepdim=True)
    i0_bp_std = i0_bp.std(dim=-1, unbiased=False, keepdim=True)
    i0_bp_max = i0_bp.amax(dim=-1, keepdim=True)
    i0_bp_min = i0_bp.amin(dim=-1, keepdim=True)

    i0_ap_mean = i0_ap.mean(dim=-1, keepdim=True)
    i0_ap_std = i0_ap.std(dim=-1, unbiased=False, keepdim=True)
    i0_ap_max = i0_ap.amax(dim=-1, keepdim=True)
    i0_ap_min = i0_ap.amin(dim=-1, keepdim=True)

    i0_bv_sum = i0_bv.abs().sum(dim=-1, keepdim=True)
    i0_av_sum = i0_av.abs().sum(dim=-1, keepdim=True)
    i0_imb = (i0_bv.sum(dim=-1, keepdim=True) - i0_av.sum(dim=-1, keepdim=True)) / (i0_bv_sum + i0_av_sum + 1e-5)

    i1_bv_sum = i1_bv.abs().sum(dim=-1, keepdim=True)
    i1_av_sum = i1_av.abs().sum(dim=-1, keepdim=True)
    i1_imb = (i1_bv.sum(dim=-1, keepdim=True) - i1_av.sum(dim=-1, keepdim=True)) / (i1_bv_sum + i1_av_sum + 1e-5)

    return torch.cat([
        i0_bp_mean, i0_bp_std, i0_bp_max, i0_bp_min,
        i0_ap_mean, i0_ap_std, i0_ap_max, i0_ap_min,
        i0_ap_mean - i0_bp_mean,
        i0_bv.mean(dim=-1, keepdim=True), i0_bv.amax(dim=-1, keepdim=True),
        i0_av.mean(dim=-1, keepdim=True), i0_av.amax(dim=-1, keepdim=True),
        i0_imb,
        i0_dp, i0_dv,
        i1_bp.mean(dim=-1, keepdim=True), i1_ap.mean(dim=-1, keepdim=True),
        i1_imb,
        i1_dp.mean(dim=-1, keepdim=True), i1_dv.mean(dim=-1, keepdim=True),
        a,
    ], dim=-1)  # 4 + 4 + 1 + 4 + 1 + 8 + 2 + 1 + 2 + 8 = 35 dims


def compute_snap40_torch(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Exact 40-dim snapshot features that achieved 0.67695 WP (+0.00486 vs raw, +0.00303 vs shipped) OOF."""
    snap27 = compute_book_skip_torch(x)[..., :27]
    bp_s, _ = torch.sort(x[..., 0:11], dim=-1)
    ap_s, _ = torch.sort(x[..., 11:22], dim=-1)
    best_q = torch.cat([
        bp_s[..., 10:11], bp_s[..., 9:10], bp_s[..., 5:6],
        ap_s[..., 0:1], ap_s[..., 1:2], ap_s[..., 5:6],
        ap_s[..., 0:1] - bp_s[..., 10:11],
    ], dim=-1)  # 7
    q_feats = torch.cat([
        q,
        q[..., 0:1] - q[..., 1:2],
        q[..., 0:1] + q[..., 1:2],
        q.abs(),
    ], dim=-1)  # 6
    return torch.cat([q_feats, snap27, best_q], dim=-1)  # 40 dims!


class FullSkipGRU(nn.Module):
    """Unfrozen 2-layer GRU initialized from `correlation1024` + standardized 40-dim snapshot skip MLP."""

    def __init__(self, base_onnx: str | Path, skip_hidden: int = 96):
        super().__init__()
        base = load_torch_model(base_onnx, device="cpu")
        self.hidden_size = base.hidden_size
        self.gru0 = nn.GRU(112, self.hidden_size, batch_first=True)
        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.gru0.load_state_dict(base.gru0.state_dict())
        self.gru1.load_state_dict(base.gru1.state_dict())
        self.head = copy.deepcopy(base.head)

        # Fixed corpus standardization buffers (never per-row LayerNorm!)
        self.register_buffer("snap_mu", torch.zeros(40))
        self.register_buffer("snap_sd", torch.ones(40))
        self.skip_mlp = nn.Sequential(
            nn.Linear(40, skip_hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(skip_hidden, 2),
        )
        nn.init.zeros_(self.skip_mlp[-1].weight)
        nn.init.zeros_(self.skip_mlp[-1].bias)

    def initial_states(self, batch_size: int, device=None):
        dev = device or next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.hidden_size, device=dev),
                torch.zeros(1, batch_size, self.hidden_size, device=dev))

    def forward(self, x: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor):
        out0, nh0 = self.gru0(x, h0)
        out1, nh1 = self.gru1(out0, h1)
        base_p = self.head(out1)
        q = base_p.clamp(-2.0, 2.0)
        z = (compute_snap40_torch(x, q) - self.snap_mu) / self.snap_sd
        pred = q + self.skip_mlp(z)
        return pred, nh0, nh1


class MaskPropensityScorer:
    def __init__(self, npz_path: str | Path = "artifacts/experiments/mask_propensity/model.npz"):
        d = np.load(npz_path)
        self.mu = d["mu"]
        self.sd = d["sd"]
        self.w1 = d["w1"]
        self.b1 = d["b1"]
        self.w2 = d["w2"]
        self.b2 = d["b2"]

    def row_weights(self, x: np.ndarray, power: float = 2.5, top_q: float = 0.12) -> np.ndarray:
        """Compute sharp mask-propensity weights [T] concentrating strictly on scored-like rows!

        Diagnostic proved: on `train.parquet`, baseline WP is 0.500 on all rows,
        0.649 on top 10% propensity rows, and 0.689 on top 5% propensity rows.
        Any positive floor on the 90% unscored rows drags the model DOWN toward 0.47 WP!
        """
        a = x[:, 104:112]
        dx = np.vstack([np.zeros((1, 112), np.float32), np.abs(np.diff(x, axis=0))])
        mf = np.hstack([
            a,
            a[:, [2, 3, 4, 7]] ** 2,
            (dx[:, :22] > 0).mean(1, keepdims=True),
            (dx[:, 22:44] > 0).mean(1, keepdims=True),
            (dx[:, 44:52] > 0).mean(1, keepdims=True),
            (dx[:, 52:104] > 0).mean(1, keepdims=True),
            (x[:, 11:22].mean(1) - x[:, 0:11].mean(1))[:, None],
            np.log1p(np.abs(x[:, 22:44]).mean(1, keepdims=True)),
            np.log1p(np.abs(x[:, 48:52]).mean(1, keepdims=True)),
            (np.arange(len(x), dtype=np.float32) / 20000.0)[:, None],
        ]).astype(np.float32)
        z = (mf - self.mu) / self.sd
        h = z @ self.w1.T + self.b1
        h = h / (1.0 + np.exp(-np.clip(h, -20, 20)))  # SiLU
        logit = (h @ self.w2.T + self.b2).ravel()
        prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -20, 20)))
        prob[:WARMUP] = 0.0
        # Pure power-law / thresholded propensity weight: zero floor!
        # Keep top ~15% rows plus smooth P^power tail so 85%+ of weight is on true scored-like bursts
        thresh = np.quantile(prob[WARMUP:], 1.0 - top_q) * 0.5
        w = np.where(prob >= thresh, (prob / 0.20) ** power, 0.0)
        w[:WARMUP] = 0.0
        return w.astype(np.float32)


def load_batch_parallel(parquet_path: str | Path, groups: list[int], propensity: MaskPropensityScorer, executor: ThreadPoolExecutor):
    cols = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)

    def _read_one(g: int):
        pf = pq.ParquetFile(parquet_path)
        tb = pf.read_row_group(int(g), columns=cols, use_threads=False)
        x = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([tb[c].to_numpy() for c in TARGET_COLUMNS]).astype(np.float32)
        mw = propensity.row_weights(x)
        return x, y, mw

    items = list(executor.map(_read_one, groups))
    x = np.stack([it[0] for it in items], axis=0)
    y = np.stack([it[1] for it in items], axis=0)
    mw = np.stack([it[2] for it in items], axis=0)
    return x, y, mw


def exact_global_wp_coefficients(pred_full: torch.Tensor, target_full: torch.Tensor,
                                 row_w_full: torch.Tensor, mse_weight: float = 0.5):
    """Compute exact full-batch [B, 20000, 2] Global WP + |y|-weighted MSE loss and dL/d(pred).

    Computes moments ONCE across all B * 20,000 rows in FP64 so chunked TBPTT has zero
    chunk-boundary bias!
    """
    raw = pred_full.detach().double()
    yc = target_full.detach().double().clamp(-2.0, 2.0)
    pc = raw.clamp(-2.0, 2.0)
    w = yc.abs() * row_w_full.detach().double().unsqueeze(-1)  # [B, T, 2]
    w[:, :WARMUP, :] = 0.0
    total = w.sum(dim=(0, 1)).clamp_min(1e-8)  # [2]
    q = w / total

    # 1. Exact pooled Global WP correlation gradient across all B * 20,000 rows
    ym = (q * yc).sum(dim=(0, 1))
    pm = (q * pc).sum(dim=(0, 1))
    yd = yc - ym
    pd = pc - pm
    cov = (q * yd * pd).sum(dim=(0, 1))
    vy = (q * yd.square()).sum(dim=(0, 1)).clamp_min(1e-12)
    vp = (q * pd.square()).sum(dim=(0, 1)).clamp_min(1e-12)
    sy, sp = vy.sqrt(), vp.sqrt()
    rho = (cov / (sy * sp)).clamp(-1.0, 1.0)
    d_rho = (q / (sy * sp)) * (yd - (cov / vp) * pd) * (raw.abs() < 2.0)
    grad_wp = -d_rho / 2.0  # mean over 2 targets

    # 2. Exact pooled |y|-weighted MSE gradient (pulls raw predictions onto optimal cross-sequence scale!)
    mse_per_t = (w * (raw - yc).square()).sum(dim=(0, 1)) / total
    grad_mse = (2.0 * q * (raw - yc)) / 2.0

    coeff = (grad_wp + mse_weight * grad_mse).float()
    total_loss = float((1.0 - rho).mean() + mse_weight * mse_per_t.mean())
    return total_loss, coeff, float(rho.mean()), float(mse_per_t.mean())


@torch.no_grad()
def evaluate_model_on_split(model: nn.Module, groups: list[int], targets: np.ndarray, scored: np.ndarray,
                            valid_path: str | Path = DEFAULT_VALID, batch_size: int = 64) -> dict:
    model.eval()
    pf = pq.ParquetFile(valid_path)
    preds = np.empty((len(groups), 20000, 2), dtype=np.float32)
    for start in range(0, len(groups), batch_size):
        end = min(start + batch_size, len(groups))
        xs = []
        for g in groups[start:end]:
            tb = pf.read_row_group(int(g), columns=FEATURE_COLUMNS)
            xs.append(np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32))
        xt = torch.from_numpy(np.stack(xs, axis=0))
        h0, h1 = model.initial_states(len(xs))
        # Chunked inference for low memory
        outs = []
        for t0 in range(0, 20000, 2500):
            out, h0, h1 = model(xt[:, t0:t0 + 2500], h0, h1)
            outs.append(out.numpy())
        preds[start:end] = np.concatenate(outs, axis=1)

    stats = np.stack([sufficient_statistics(targets[i][scored[i]], preds[i][scored[i]]) for i in range(len(groups))])
    per_t = score_statistics(merge_statistics(stats))
    return {"wp": float(per_t.mean()), "t0": float(per_t[0]), "t1": float(per_t[1]), "stats": stats, "preds": preds}


def stratified_batches(train_classes: np.ndarray, dev_classes: np.ndarray, batch_size: int, n_batches: int, seed: int):
    """Draw stratified batches of `train.parquet` sequences matching validation's 4-tick-class proportions."""
    rng = np.random.default_rng(seed)
    unique_c = np.unique(dev_classes)
    dev_shares = np.array([(dev_classes == c).mean() for c in unique_c])
    counts = np.round(dev_shares * batch_size).astype(int)
    counts[-1] = batch_size - counts[:-1].sum()

    pools = {c: rng.permutation(np.nonzero(train_classes == c)[0]).tolist() for c in unique_c}
    cursors = {c: 0 for c in unique_c}
    batches = []
    for _ in range(n_batches):
        batch = []
        for c, k in zip(unique_c, counts):
            pool = pools[c]
            cur = cursors[c]
            if cur + k > len(pool):
                pool[:] = rng.permutation(pool).tolist()
                cur = 0
            batch.extend(pool[cur:cur + k])
            cursors[c] = cur + k
        rng.shuffle(batch)
        batches.append(batch)
    return batches


def train_unfrozen(args):
    torch.set_num_threads(args.threads)
    seed_everything(args.seed, args.threads)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path("artifacts/manifest.json").read_text())
    screen_groups = manifest["splits"]["screen"]
    dev_groups = manifest["splits"]["dev"]
    dev_targets, dev_scored = load_targets(dev_groups, DEFAULT_VALID, store="artifacts/experiments/dev_targets.npz")
    dev_group_to_idx = {g: i for i, g in enumerate(dev_groups)}
    screen_idx = [dev_group_to_idx[g] for g in screen_groups]
    screen_targets = dev_targets[screen_idx]
    screen_scored = dev_scored[screen_idx]

    cls_npz = np.load("artifacts/experiments/data_semantics/class_labels.npz")
    train_classes = cls_npz["train"]
    dev_classes = cls_npz["dev"]
    batches = stratified_batches(train_classes, dev_classes, args.batch_size, args.steps, args.seed)

    model = FullSkipGRU(args.base_model, skip_hidden=args.skip_hidden)
    propensity = MaskPropensityScorer()

    # Estimate fixed corpus snap40 mean/std on first 64 train sequences (weighted by propensity)
    with ThreadPoolExecutor(max_workers=4) as init_pool:
        x_cal, _, mw_cal = load_batch_parallel(args.train, batches[0], propensity, init_pool)
    with torch.no_grad():
        xt_cal = torch.from_numpy(x_cal)
        o0, _ = model.gru0(xt_cal, torch.zeros(1, len(x_cal), model.hidden_size))
        o1, _ = model.gru1(o0, torch.zeros(1, len(x_cal), model.hidden_size))
        q_cal = model.head(o1).clamp(-2.0, 2.0)
        s40 = compute_snap40_torch(xt_cal, q_cal)  # [B, T, 40]
        w_cal = torch.from_numpy(mw_cal).unsqueeze(-1)
        w_cal_sum = w_cal.sum().clamp_min(1.0)
        mu40 = (s40 * w_cal).sum(dim=(0, 1)) / w_cal_sum
        sd40 = (((s40 - mu40).square() * w_cal).sum(dim=(0, 1)) / w_cal_sum).sqrt() + 1e-5
        model.snap_mu.copy_(mu40)
        model.snap_sd.copy_(sd40)

    # Differential learning rates: gentle enough not to destroy pretrained features on step 1,
    # strong enough (10x-30x old v2.1) to genuinely adapt gru0, gru1, and the skip readout!
    optimizer = torch.optim.AdamW([
        {"params": model.gru0.parameters(), "lr": args.lr_gru0, "weight_decay": 1e-4},
        {"params": model.gru1.parameters(), "lr": args.lr_gru1, "weight_decay": 1e-4},
        {"params": model.head.parameters(), "lr": args.lr_head, "weight_decay": 1e-4},
        {"params": model.skip_mlp.parameters(), "lr": args.lr_skip, "weight_decay": 5e-4},
    ])

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    init_screen = evaluate_model_on_split(model, screen_groups, screen_targets, screen_scored)
    print(f"Step 0/{args.steps} | Screen WP: {init_screen['wp']:.5f} (t0={init_screen['t0']:.5f}, t1={init_screen['t1']:.5f})", flush=True)

    best_screen_wp = init_screen["wp"]
    best_state = copy.deepcopy(model.state_dict())
    history = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut = pool.submit(load_batch_parallel, args.train, batches[0], propensity, pool)
        for step in range(args.steps):
            t_step = time.time()
            x_np, y_np, mw_np = fut.result()
            if step + 1 < args.steps:
                fut = pool.submit(load_batch_parallel, args.train, batches[step + 1], propensity, pool)

            x_t = torch.from_numpy(x_np)
            y_t = torch.from_numpy(y_np)
            mw_t = torch.from_numpy(mw_np)

            # Pass 1 (no_grad): run full 20,000-step sequence to record exact boundary states & predictions
            chunk = args.tbptt
            model.eval()
            with torch.no_grad():
                h0, h1 = model.initial_states(len(batches[step]))
                boundary_states = []
                full_preds = []
                for c_start in range(0, 20000, chunk):
                    c_end = min(c_start + chunk, 20000)
                    boundary_states.append((h0.clone(), h1.clone()))
                    pred_c, h0, h1 = model(x_t[:, c_start:c_end], h0, h1)
                    full_preds.append(pred_c)
                pred_full = torch.cat(full_preds, dim=1)
                _, coeff_full, train_rho, train_mse = exact_global_wp_coefficients(
                    pred_full, y_t, mw_t, mse_weight=args.mse_weight
                )

            # Pass 2 (grad): exact VJP backward per chunk from exact boundary states
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for idx_c, c_start in enumerate(range(0, 20000, chunk)):
                c_end = min(c_start + chunk, 20000)
                if c_end <= WARMUP:
                    continue
                c_coeff = coeff_full[:, c_start:c_end]
                if c_coeff.abs().sum() == 0:
                    continue
                bh0, bh1 = boundary_states[idx_c]
                pred_c, _, _ = model(x_t[:, c_start:c_end], bh0, bh1)
                (pred_c * c_coeff).sum().backward()

            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            optimizer.step()
            scheduler.step()

            rec = {
                "step": step + 1,
                "train_rho": round(train_rho, 5),
                "train_mse": round(train_mse, 5),
                "grad_norm": round(gnorm, 4),
                "sec": round(time.time() - t_step, 2),
            }
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                sc = evaluate_model_on_split(model, screen_groups, screen_targets, screen_scored)
                rec["screen_wp"] = sc["wp"]
                print(f"Step {step + 1:3d}/{args.steps} | train_rho={rec['train_rho']:.4f} mse={rec['train_mse']:.4f} "
                      f"gnorm={gnorm:.3f} | Screen WP={sc['wp']:.5f} (delta={sc['wp'] - init_screen['wp']:+.5f}) "
                      f"[{time.time() - t_start:.0f}s]", flush=True)
                if sc["wp"] > best_screen_wp:
                    best_screen_wp = sc["wp"]
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save({"model_state": best_state, "step": step + 1, "screen_wp": best_screen_wp},
                               out_dir / "best_model.pt")
            else:
                if (step + 1) % 5 == 0:
                    print(f"Step {step + 1:3d}/{args.steps} | train_rho={rec['train_rho']:.4f} mse={rec['train_mse']:.4f} "
                          f"gnorm={gnorm:.3f} ({rec['sec']}s/step)", flush=True)
            history.append(rec)

    # Full 1,400-sequence dev evaluation on best screen checkpoint
    model.load_state_dict(best_state)
    print("Evaluating best checkpoint on full 1,400-sequence dev split...", flush=True)
    dev_eval = evaluate_model_on_split(model, dev_groups, dev_targets, dev_scored)
    by_class = {str(c): round(global_wp(dev_eval["stats"][dev_classes == c]), 5) for c in np.unique(dev_classes)}
    np.save(out_dir / "dev_predictions.npy", dev_eval["preds"])
    np.save(out_dir / "dev_statistics.npy", dev_eval["stats"])
    torch.save({"model_state": best_state, "args": vars(args), "dev_wp": dev_eval["wp"], "by_class": by_class},
               out_dir / "final_checkpoint.pt")
    summary = {
        "init_screen_wp": init_screen["wp"],
        "best_screen_wp": best_screen_wp,
        "dev_wp": dev_eval["wp"],
        "dev_t0": dev_eval["t0"],
        "dev_t1": dev_eval["t1"],
        "dev_by_class": by_class,
        "total_seconds": round(time.time() - t_start, 1),
        "args": vars(args),
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE Full Dev WP: {dev_eval['wp']:.5f} (t0={dev_eval['t0']:.5f}, t1={dev_eval['t1']:.5f}) | by_class={by_class}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="datasets/train.parquet")
    p.add_argument("--base-model", default="artifacts/round2/correlation_gru1024_seed20260912/model.onnx")
    p.add_argument("--output", default="artifacts/experiments/track_a_unfrozen_v1")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--warmup-steps", type=int, default=15)
    p.add_argument("--tbptt", type=int, default=1000)
    p.add_argument("--lr-gru0", type=float, default=1.5e-4)
    p.add_argument("--lr-gru1", type=float, default=2.5e-4)
    p.add_argument("--lr-head", type=float, default=5e-4)
    p.add_argument("--lr-skip", type=float, default=1e-3)
    p.add_argument("--skip-hidden", type=int, default=64)
    p.add_argument("--mse-weight", type=float, default=0.35)
    p.add_argument("--eval-every", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261111)
    p.add_argument("--threads", type=int, default=6)
    train_unfrozen(p.parse_args())


if __name__ == "__main__":
    main()
