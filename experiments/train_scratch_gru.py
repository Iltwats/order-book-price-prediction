"""Train a diverse Scratch 2-Layer Sorted-LOB GRU on CPU while MPS trains the Gated TCN.

Why run this in parallel:
1. Uses a completely different inductive bias from BOTH `baseline.onnx` (which reads raw unordered
   columns 0..111 with no quote sorting or order-flow diffs) and `MultiScaleGatedTCN` (finite-horizon
   dilated convs):
   - Infinite-horizon recurrent state (`GRU(169 -> 144) -> GRU(144 -> 144)`) fed with the exact
     169-channel price-sorted L11 depth + aligned volumes + multi-lag order-flow tensor!
   - Direct instantaneous skip readout from stem + `gru0` + `gru1` (`144 * 3 -> 2`).
2. Runs on 6 CPU threads (~12s/step at B=64) completely independent of the MPS GPU!
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
from connectome.training import seed_everything
from experiments.devdata import DEFAULT_VALID, WARMUP, global_wp, load_targets
from experiments.train_scratch_tcn import TICK_CLASSES, build_lob_engineered_features_torch, load_batch_tcn
from experiments.train_unfrozen_gru import MaskPropensityScorer, exact_global_wp_coefficients, stratified_batches
from utils import FEATURE_COLUMNS


class ScratchSortedGRU(nn.Module):
    """2-Layer Residual GRU on 169 price-sorted L11 + order-flow channels."""

    def __init__(self, in_channels: int = 169, hidden: int = 144):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.register_buffer("in_mu", torch.zeros(1, 1, in_channels))
        self.register_buffer("in_sd", torch.ones(1, 1, in_channels))

        self.stem = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
        )
        self.gru0 = nn.GRU(hidden, hidden, batch_first=True)
        self.gru1 = nn.GRU(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )

    def initial_states(self, batch_size: int, device=None):
        dev = device or next(self.parameters()).device
        return (torch.zeros(1, batch_size, self.hidden, device=dev),
                torch.zeros(1, batch_size, self.hidden, device=dev))

    def forward_features(self, feat_btf: torch.Tensor, h0: torch.Tensor, h1: torch.Tensor):
        # feat_btf: [B, T, 169]
        z = (feat_btf - self.in_mu) / self.in_sd
        s = self.stem(z)
        o0, nh0 = self.gru0(s, h0)
        o1, nh1 = self.gru1(o0 + s, h1)
        pred = self.head(torch.cat([s, o0, o1], dim=-1))
        return pred, nh0, nh1


@torch.no_grad()
def evaluate_scratch_gru(model: ScratchSortedGRU, groups: list[int], targets: np.ndarray,
                         scored: np.ndarray, valid_path: str | Path = DEFAULT_VALID,
                         batch_size: int = 64) -> dict:
    model.eval()
    pf = pq.ParquetFile(valid_path)
    preds = np.empty((len(groups), 20000, 2), dtype=np.float32)
    for start in range(0, len(groups), batch_size):
        end = min(start + batch_size, len(groups))
        xs = []
        for g in groups[start:end]:
            tb = pf.read_row_group(int(g), columns=FEATURE_COLUMNS)
            xs.append(np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32))
        feat_bct = build_lob_engineered_features_torch(torch.from_numpy(np.stack(xs, axis=0)))
        feat_btf = feat_bct.transpose(1, 2).contiguous()
        h0, h1 = model.initial_states(len(xs))
        outs = []
        for t0 in range(0, 20000, 2500):
            p, h0, h1 = model.forward_features(feat_btf[:, t0:t0 + 2500], h0, h1)
            outs.append(p.numpy())
        preds[start:end] = np.concatenate(outs, axis=1)
    stats = np.stack([sufficient_statistics(targets[i][scored[i]], preds[i][scored[i]]) for i in range(len(groups))])
    per_t = score_statistics(merge_statistics(stats))
    return {"wp": float(per_t.mean()), "t0": float(per_t[0]), "t1": float(per_t[1]), "stats": stats, "preds": preds}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="datasets/train.parquet")
    p.add_argument("--output", default="artifacts/experiments/track_c_scratch_sorted_gru_v1")
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=220, help="220 * 64 = 14,080 sequences (~1.33 passes over all 10,607)")
    p.add_argument("--warmup-steps", type=int, default=15)
    p.add_argument("--tbptt", type=int, default=1000)
    p.add_argument("--lr", type=float, default=2.5e-3)
    p.add_argument("--mse-weight", type=float, default=0.5)
    p.add_argument("--mask-power", type=float, default=1.5)
    p.add_argument("--mask-floor", type=float, default=0.15)
    p.add_argument("--eval-every", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--resume", default="", help="Optional checkpoint path (`best_model.pt` or `final_checkpoint.pt`) to warm-start from")
    args = p.parse_args()

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
    propensity = MaskPropensityScorer()
    model = ScratchSortedGRU(hidden=args.hidden)
    print(f"Initialized ScratchSortedGRU (hidden={args.hidden}, {sum(x.numel() for x in model.parameters()):,} params)", flush=True)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        print(f"Resumed weights (including fixed in_mu/in_sd) from {args.resume}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=3) as init_pool:
            f_cal, _, _, _ = load_batch_tcn(args.train, batches[0], train_classes, propensity, init_pool)
        with torch.no_grad():
            f_btf = f_cal.transpose(1, 2).contiguous()
            mu = f_btf.mean(dim=(0, 1), keepdim=True)
            sd = f_btf.std(dim=(0, 1), unbiased=False, keepdim=True) + 1e-4
            model.in_mu.copy_(mu)
            model.in_sd.copy_(sd)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_screen_wp = -1.0
    if args.resume:
        init_sc = evaluate_scratch_gru(model, screen_groups, screen_targets, screen_scored)
        best_screen_wp = init_sc["wp"]
        print(f"[GRU] Resumed Screen WP={best_screen_wp:.5f}", flush=True)
    best_state = copy.deepcopy(model.state_dict())
    history = []
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=3) as pool:
        fut = pool.submit(load_batch_tcn, args.train, batches[0], train_classes, propensity, pool,
                          args.mask_power, args.mask_floor)
        for step in range(args.steps):
            t_step = time.time()
            feat_bct, y_b2t_np, mw_np, _ = fut.result()
            if step + 1 < args.steps:
                fut = pool.submit(load_batch_tcn, args.train, batches[step + 1], train_classes, propensity, pool,
                                  args.mask_power, args.mask_floor)

            feat_btf = feat_bct.transpose(1, 2).contiguous()
            y_bt2 = torch.from_numpy(np.ascontiguousarray(y_b2t_np.transpose(0, 2, 1)))
            mw_t = torch.from_numpy(mw_np)

            # Pass 1: exact 20,000-step trajectory + exact batch-pooled Global WP + MSE coefficients
            chunk = args.tbptt
            model.eval()
            with torch.no_grad():
                h0, h1 = model.initial_states(len(batches[step]))
                b_states, f_preds = [], []
                for c_s in range(0, 20000, chunk):
                    c_e = min(c_s + chunk, 20000)
                    b_states.append((h0.clone(), h1.clone()))
                    pc, h0, h1 = model.forward_features(feat_btf[:, c_s:c_e], h0, h1)
                    f_preds.append(pc)
                pred_full = torch.cat(f_preds, dim=1)
                _, coeff_full, tr_rho, tr_mse = exact_global_wp_coefficients(
                    pred_full, y_bt2, mw_t, mse_weight=args.mse_weight
                )

            # Pass 2: chunked TBPTT backward
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for ic, c_s in enumerate(range(0, 20000, chunk)):
                c_e = min(c_s + chunk, 20000)
                if c_e <= WARMUP:
                    continue
                cc = coeff_full[:, c_s:c_e]
                if cc.abs().sum() == 0:
                    continue
                bh0, bh1 = b_states[ic]
                pc, _, _ = model.forward_features(feat_btf[:, c_s:c_e], bh0, bh1)
                (pc * cc).sum().backward()

            gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5))
            optimizer.step()
            scheduler.step()

            rec = {
                "step": step + 1,
                "train_rho": round(tr_rho, 5),
                "train_mse": round(tr_mse, 5),
                "grad_norm": round(gnorm, 4),
                "sec": round(time.time() - t_step, 2),
            }
            if (step + 1) % 10 == 0 and (step + 1) % args.eval_every != 0:
                print(f"[GRU] Step {step + 1:3d}/{args.steps} | train_rho={tr_rho:.4f} mse={tr_mse:.4f} "
                      f"gnorm={gnorm:.3f} ({rec['sec']:.2f}s/step)", flush=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                sc = evaluate_scratch_gru(model, screen_groups, screen_targets, screen_scored)
                rec["screen_wp"] = sc["wp"]
                print(f"[GRU] Step {step + 1:3d}/{args.steps} | train_rho={tr_rho:.4f} mse={tr_mse:.4f} "
                      f"| Screen WP={sc['wp']:.5f} (t0={sc['t0']:.5f}, t1={sc['t1']:.5f}) "
                      f"[{time.time() - t_start:.0f}s]", flush=True)
                if sc["wp"] > best_screen_wp:
                    best_screen_wp = sc["wp"]
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save({"model_state": best_state, "step": step + 1, "screen_wp": best_screen_wp},
                               out_dir / "best_model.pt")
            history.append(rec)

    model.load_state_dict(best_state)
    print(f"Evaluating best ScratchSortedGRU (screen WP={best_screen_wp:.5f}) on all 1,400 dev sequences...", flush=True)
    dev_eval = evaluate_scratch_gru(model, dev_groups, dev_targets, dev_scored)
    by_class = {str(c): round(global_wp(dev_eval["stats"][dev_classes == c]), 5) for c in np.unique(dev_classes)}

    inc_preds = np.load("artifacts/global_wp/rescore_v1/correlation1024_seed20260912.cache/predictions.npy", mmap_mode="r")
    blend_wps = {}
    for w_m in (0.15, 0.25, 0.35, 0.50):
        blended = (1.0 - w_m) * inc_preds + w_m * dev_eval["preds"]
        st_b = np.stack([sufficient_statistics(dev_targets[i][dev_scored[i]], blended[i][dev_scored[i]])
                         for i in range(len(dev_groups))])
        blend_wps[str(w_m)] = round(global_wp(st_b), 5)

    np.save(out_dir / "dev_predictions.npy", dev_eval["preds"])
    np.save(out_dir / "dev_statistics.npy", dev_eval["stats"])
    torch.save({"model_state": best_state, "args": vars(args), "dev_wp": dev_eval["wp"], "by_class": by_class},
               out_dir / "final_checkpoint.pt")
    summary = {
        "best_screen_wp": best_screen_wp,
        "dev_wp": dev_eval["wp"],
        "dev_t0": dev_eval["t0"],
        "dev_t1": dev_eval["t1"],
        "dev_by_class": by_class,
        "raw_blend_with_correlation1024": blend_wps,
        "total_seconds": round(time.time() - t_start, 1),
        "args": vars(args),
        "history": history,
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"DONE ScratchSortedGRU Dev WP: {dev_eval['wp']:.5f} | by_class={by_class} | raw_blends={blend_wps}", flush=True)


if __name__ == "__main__":
    main()
