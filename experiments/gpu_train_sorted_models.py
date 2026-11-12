"""Cloud GPU (CUDA / cuDNN) high-speed trainer for full-convergence Sorted-LOB ensembles.

Designed for Runpod / GCP / Lambda single-GPU pods (RTX 4090 / L40S / A100) via
`cloud/Dockerfile.scratch`.

Supports three high-leverage training arms out of the box:
1. `--arch sgru` (default `hidden=160` or `192`): 8-10 full epochs (`2,800-3,300` updates at `B=32/64`)
   of `ScratchSortedGRU` on all 10,607 `train.parquet` sequences with Polyak EMA weight averaging.
2. `--arch slstm` (`ScratchSortedLSTM`): orthogonal 2-layer Residual LSTM cell-state highway
   (`[i, f, g, o]`) over the exact 169 causal sorted-LOB + multi-lag order-flow channels.
3. `--mode large_tick` (`0.4685 + 0.6569` specialist): trains exclusively on the 8,044 large-tick
   sequences in `train.parquet` (48.5% of scored weight, current bottleneck at ~0.41 WP).
4. `--include-valid-split {none,dev,all}`: optional final-day `train.parquet + valid.parquet`
   12,480-sequence refit using exact ground-truth `is_scored` on `valid.parquet` rows!

Features:
- GPU-accelerated `build_lob_engineered_features_torch` + `cuDNN` fused GRU/LSTM kernels
  (~25-40x faster than CPU).
- Atomic checkpointing after every `--save-every` steps + clean `SIGTERM`/`SIGINT` handling
  for Spot/Runpod preemption (`--resume` continues seamlessly).
- Weight EMA (`--ema-decay 0.995`) evaluated automatically on `screen` / `dev` when `valid.parquet`
  is mounted, or exported directly as a lightweight checkpoint (`~2-4 MB`) to bring back locally.
"""
from __future__ import annotations

import os

# Must be set before torch initializes CuBLAS when deterministic mode is enabled
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import copy
import json
import math
import random
import signal
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn

from connectome.global_metric import merge_statistics, score_statistics, sufficient_statistics
from experiments.train_scratch_gru import ScratchSortedGRU
from experiments.train_scratch_tcn import TICK_CLASSES, build_lob_engineered_features_torch
from experiments.train_unfrozen_gru import MaskPropensityScorer, exact_global_wp_coefficients, stratified_batches
from utils import FEATURE_COLUMNS, TARGET_COLUMNS

WARMUP = 99


def seed_gpu_runner(seed: int, threads: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(max(1, threads))
    # Keep deterministic_algorithms opt-in on CUDA: torch.gather backward / cuDNN RNN
    # with benchmark=True can otherwise raise RuntimeError on newer PyTorch/CUDA builds!
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


class ScratchSortedLSTM(nn.Module):
    """Orthogonal 2-Layer Residual LSTM on 169 price-sorted L11 + order-flow channels.

    Matches `TorchSortedLSTM` in `native/ensemble_engine.cpp` bit-for-bit (`[i, f, g, o]` gate order).
    """

    def __init__(self, in_channels: int = 169, hidden: int = 160):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.register_buffer("in_mu", torch.zeros(1, 1, in_channels))
        self.register_buffer("in_sd", torch.ones(1, 1, in_channels))

        self.stem = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.GELU(),
        )
        self.lstm0 = nn.LSTM(hidden, hidden, batch_first=True)
        self.lstm1 = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        # Initialize forget-gate bias to +1.0 for long-horizon memory retention
        with torch.no_grad():
            self.lstm0.bias_ih_l0[hidden:2 * hidden].fill_(0.5)
            self.lstm0.bias_hh_l0[hidden:2 * hidden].fill_(0.5)
            self.lstm1.bias_ih_l0[hidden:2 * hidden].fill_(0.5)
            self.lstm1.bias_hh_l0[hidden:2 * hidden].fill_(0.5)

    def initial_states(self, batch_size: int, device=None):
        dev = device or next(self.parameters()).device
        z = lambda: torch.zeros(1, batch_size, self.hidden, device=dev)
        return ((z(), z()), (z(), z()))

    def forward_features(self, feat_btf: torch.Tensor, s0: tuple[torch.Tensor, torch.Tensor],
                         s1: tuple[torch.Tensor, torch.Tensor]):
        z = (feat_btf - self.in_mu) / self.in_sd
        s = self.stem(z)
        o0, ns0 = self.lstm0(s, s0)
        o1, ns1 = self.lstm1(o0 + s, s1)
        pred = self.head(torch.cat([s, o0, o1], dim=-1))
        return pred, ns0, ns1


def clone_state(state):
    if isinstance(state, tuple):
        return tuple(clone_state(x) for x in state)
    return state.detach().clone()


@torch.no_grad()
def update_ema(ema_model: nn.Module, online_model: nn.Module, decay: float):
    for p_ema, p_on in zip(ema_model.parameters(), online_model.parameters()):
        p_ema.data.mul_(decay).add_(p_on.data, alpha=1.0 - decay)
    for b_ema, b_on in zip(ema_model.buffers(), online_model.buffers()):
        b_ema.data.copy_(b_on.data)


def build_model(arch: str, hidden: int, device: torch.device) -> nn.Module:
    if arch == "sgru":
        return ScratchSortedGRU(in_channels=169, hidden=hidden).to(device)
    if arch == "slstm":
        return ScratchSortedLSTM(in_channels=169, hidden=hidden).to(device)
    raise ValueError(f"Unknown arch: {arch}")


import threading

_THREAD_LOCAL_PQ = threading.local()


def get_cached_parquet(path: str) -> pq.ParquetFile:
    """Thread-local ParquetFile cache: opened ONCE per worker thread (`pre_buffer=False` avoids Arrow ReadRangeCache bugs!)."""
    cache = getattr(_THREAD_LOCAL_PQ, "files", None)
    if cache is None:
        cache = {}
        _THREAD_LOCAL_PQ.files = cache
    pf = cache.get(path)
    if pf is None:
        pf = pq.ParquetFile(path, pre_buffer=False)
        cache[path] = pf
    return pf


class TorchMaskPropensityScorer:
    """Fast batched CUDA/PyTorch propensity scorer (~2 ms per [64, 20000, 112] batch vs ~8s on CPU!)."""

    def __init__(self, prop: MaskPropensityScorer, device: torch.device):
        self.device = device
        self.mu = torch.from_numpy(prop.mu.astype(np.float32)).to(device)
        self.sd = torch.from_numpy(prop.sd.astype(np.float32)).to(device)
        self.w1_t = torch.from_numpy(prop.w1.astype(np.float32).T.copy()).to(device)
        self.b1 = torch.from_numpy(prop.b1.astype(np.float32)).to(device)
        self.w2_t = torch.from_numpy(prop.w2.astype(np.float32).T.copy()).to(device)
        self.b2 = torch.from_numpy(prop.b2.astype(np.float32)).to(device)
        self.step_norm = (torch.arange(20000, dtype=torch.float32, device=device) / 20000.0).view(1, 20000, 1)

    @torch.no_grad()
    def batch_weights(self, x_btf: torch.Tensor, mask_power: float, mask_floor: float) -> torch.Tensor:
        # x_btf: [B, 20000, 112] on device
        B, T, _ = x_btf.shape
        a = x_btf[:, :, 104:112]
        a_sq = x_btf[:, :, [106, 107, 108, 111]].square()
        dx = torch.zeros_like(x_btf)
        dx[:, 1:, :] = (x_btf[:, 1:, :] - x_btf[:, :-1, :]).abs()
        c0 = (dx[:, :, :22] > 0).float().mean(dim=-1, keepdim=True)
        c1 = (dx[:, :, 22:44] > 0).float().mean(dim=-1, keepdim=True)
        c2 = (dx[:, :, 44:52] > 0).float().mean(dim=-1, keepdim=True)
        c3 = (dx[:, :, 52:104] > 0).float().mean(dim=-1, keepdim=True)
        spr = (x_btf[:, :, 11:22].mean(dim=-1) - x_btf[:, :, 0:11].mean(dim=-1)).unsqueeze(-1)
        lv = torch.log1p(x_btf[:, :, 22:44].abs().mean(dim=-1, keepdim=True))
        lt = torch.log1p(x_btf[:, :, 48:52].abs().mean(dim=-1, keepdim=True))
        st = self.step_norm[:, :T, :].expand(B, T, 1)

        mf = torch.cat([a, a_sq, c0, c1, c2, c3, spr, lv, lt, st], dim=-1)
        z = (mf - self.mu) / self.sd
        h = z @ self.w1_t + self.b1
        h = h * torch.sigmoid(h.clamp(-20.0, 20.0))
        logit = (h @ self.w2_t + self.b2).squeeze(-1)
        prob = torch.sigmoid(logit.clamp(-20.0, 20.0))
        mw = mask_floor + (1.0 - mask_floor) * ((prob / 0.12) ** mask_power)
        mw[:, :WARMUP] = 0.0
        return mw


def load_raw_sequence_batch(items_spec: list[tuple[str, int, bool]], executor: ThreadPoolExecutor):
    """Read raw [B, 20000, 112] features, [B, 20000, 2] targets, and optional exact `is_scored` masks in parallel.

    1. Caches `pq.ParquetFile` across the entire process (never reparses 10,607-row-group Parquet footer!).
    2. Fills preallocated contiguous `[B, 20000, 112]` buffers directly without `np.column_stack` copies.
    3. Leaves propensity MLP math for 2-ms batched GPU execution!
    """
    B = len(items_spec)
    x_out = np.empty((B, 20000, 112), dtype=np.float32)
    y_out = np.empty((B, 20000, 2), dtype=np.float32)
    exact_mw = np.empty((B, 20000), dtype=np.float32)
    has_exact = np.zeros(B, dtype=bool)
    base_cols = list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)

    def _read_into(idx_and_spec: tuple[int, tuple[str, int, bool]]):
        b_i, (p_path, g, exact_mask) = idx_and_spec
        pf = get_cached_parquet(p_path)
        cols = base_cols + (["is_scored"] if exact_mask else [])
        tb = pf.read_row_group(int(g), columns=cols, use_threads=False)
        xb = x_out[b_i]
        for c_i in range(112):
            xb[:, c_i] = tb.column(c_i).to_numpy(zero_copy_only=False)
        y_out[b_i, :, 0] = tb.column(112).to_numpy(zero_copy_only=False)
        y_out[b_i, :, 1] = tb.column(113).to_numpy(zero_copy_only=False)
        if exact_mask:
            m = tb.column(114).to_numpy(zero_copy_only=False).astype(np.float32)
            m[:WARMUP] = 0.0
            exact_mw[b_i] = m
            has_exact[b_i] = True

    list(executor.map(_read_into, enumerate(items_spec)))
    return x_out, y_out, exact_mw, has_exact


def build_training_schedule(train_path: str, valid_path: str, include_valid_split: str,
                            mode: str, batch_size: int, steps: int, seed: int,
                            manifest: dict, train_classes: np.ndarray, dev_classes: np.ndarray):
    rng = np.random.default_rng(seed)
    if mode == "large_tick":
        # Dedicated specialist on 0.4685 + 0.6569 (8,044 train sequences; 0.6569 weighted 65% / 0.4685 35%)
        idx_46 = np.flatnonzero(np.isclose(train_classes, 0.4685))
        idx_65 = np.flatnonzero(np.isclose(train_classes, 0.6569))
        n_65 = max(1, int(round(batch_size * 0.65)))
        n_46 = max(1, batch_size - n_65)
        batches = []
        p46, p65 = rng.permutation(idx_46), rng.permutation(idx_65)
        c46 = c65 = 0
        for _ in range(steps):
            if c46 + n_46 > len(p46):
                p46 = rng.permutation(idx_46)
                c46 = 0
            if c65 + n_65 > len(p65):
                p65 = rng.permutation(idx_65)
                c65 = 0
            g_ids = np.concatenate([p65[c65:c65 + n_65], p46[c46:c46 + n_46]])
            c65 += n_65
            c46 += n_46
            rng.shuffle(g_ids)
            batches.append([(train_path, int(g), False) for g in g_ids])
        return batches

    # Generalist stratified across all 4 tick classes matching validation mix
    raw_batches = stratified_batches(train_classes, dev_classes, batch_size, steps, seed)
    if include_valid_split == "none":
        return [[(train_path, g, False) for g in b] for b in raw_batches]

    # Final-day train+valid refit: mix in ~15% exact-mask validation sequences per batch!
    if include_valid_split == "dev":
        val_groups = list(manifest["splits"]["dev"])
    elif include_valid_split == "all":
        val_groups = sorted(list(manifest["splits"]["dev"]) + list(manifest["splits"]["confirmation"]))
    else:
        raise ValueError(f"Unsupported include_valid_split: {include_valid_split}")

    n_val = max(2, batch_size // 6)
    val_perm = rng.permutation(val_groups)
    v_cur = 0
    mixed = []
    for b in raw_batches:
        if v_cur + n_val > len(val_perm):
            val_perm = rng.permutation(val_groups)
            v_cur = 0
        vg = val_perm[v_cur:v_cur + n_val]
        v_cur += n_val
        spec = [(train_path, g, False) for g in b[:-n_val]] + [(valid_path, int(g), True) for g in vg]
        rng.shuffle(spec)
        mixed.append(spec)
    return mixed


@torch.no_grad()
def evaluate_model_gpu(model: nn.Module, groups: list[int], targets: np.ndarray,
                       scored: np.ndarray, valid_path: str | Path, device: torch.device,
                       batch_size: int = 64, executor: ThreadPoolExecutor | None = None) -> dict:
    model.eval()
    preds = np.empty((len(groups), 20000, 2), dtype=np.float32)
    x_buf = np.empty((batch_size, 20000, 112), dtype=np.float32)
    v_path_str = str(valid_path)

    def _read_eval(idx_g: tuple[int, int]):
        b_i, g = idx_g
        pf = get_cached_parquet(v_path_str)
        tb = pf.read_row_group(int(g), columns=FEATURE_COLUMNS, use_threads=False)
        xb = x_buf[b_i]
        for c_i in range(112):
            xb[:, c_i] = tb.column(c_i).to_numpy(zero_copy_only=False)

    for start in range(0, len(groups), batch_size):
        end = min(start + batch_size, len(groups))
        cur_b = end - start
        pairs = list(enumerate(groups[start:end]))
        if executor is not None:
            list(executor.map(_read_eval, pairs))
        else:
            for item in pairs:
                _read_eval(item)
        x_t = torch.from_numpy(x_buf[:cur_b]).to(device, non_blocking=True)
        feat_btf = build_lob_engineered_features_torch(x_t).transpose(1, 2).contiguous()
        s0, s1 = model.initial_states(cur_b, device=device)
        outs = []
        for t0 in range(0, 20000, 4000):
            p, s0, s1 = model.forward_features(feat_btf[:, t0:t0 + 4000], s0, s1)
            outs.append(p.cpu().numpy())
        preds[start:end] = np.concatenate(outs, axis=1)
    stats = np.stack([sufficient_statistics(targets[i][scored[i]], preds[i][scored[i]]) for i in range(len(groups))])
    per_t = score_statistics(merge_statistics(stats))
    return {"wp": float(per_t.mean()), "t0": float(per_t[0]), "t1": float(per_t[1]), "stats": stats, "preds": preds}


def atomic_save(obj: dict, path: Path):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(obj, tmp)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description="Cloud GPU trainer for Sorted-LOB GRU/LSTM & Large-Tick Specialists")
    p.add_argument("--train", default="datasets/train.parquet")
    p.add_argument("--valid", default="", help="Optional path to valid.parquet for live screen/dev eval or final refit")
    p.add_argument("--output", required=True)
    p.add_argument("--arch", choices=("sgru", "slstm"), default="sgru")
    p.add_argument("--mode", choices=("generalist", "large_tick"), default="generalist")
    p.add_argument("--include-valid-split", choices=("none", "dev", "all"), default="none")
    p.add_argument("--hidden", type=int, default=160)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=1600, help="1600 * 64 = 102,400 seqs (~9.65 full passes over 10,607)")
    p.add_argument("--warmup-steps", type=int, default=60)
    p.add_argument("--tbptt", type=int, default=2000)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--mse-weight", type=float, default=0.5)
    p.add_argument("--mask-power", type=float, default=1.5)
    p.add_argument("--mask-floor", type=float, default=0.10)
    p.add_argument("--ema-decay", type=float, default=0.995)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=20261112)
    p.add_argument("--deterministic", action="store_true", help="Strict deterministic CUDA ops (slower)")
    p.add_argument("--resume", action="store_true", help="Resume from <output>/latest_checkpoint.pt if present")
    args = p.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested (--device cuda) but torch.cuda.is_available() is False!")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = not args.deterministic

    seed_gpu_runner(args.seed, max(1, args.workers), deterministic=args.deterministic)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    stop_flag = {"stop": False}

    def _handle_sig(signum, _frame):
        print(f"[SIGNAL {signum}] Clean stop requested; will save checkpoint after current step.", flush=True)
        stop_flag["stop"] = True

    signal.signal(signal.SIGTERM, _handle_sig)
    signal.signal(signal.SIGINT, _handle_sig)

    manifest = json.loads(Path("artifacts/manifest.json").read_text())
    cls_npz = np.load("artifacts/experiments/data_semantics/class_labels.npz")
    train_classes = cls_npz["train"]
    dev_classes = cls_npz["dev"]

    has_valid = bool(args.valid) and Path(args.valid).exists()
    if args.include_valid_split != "none" and not has_valid:
        raise ValueError("--include-valid-split requires --valid pointing to valid.parquet")

    screen_groups = manifest["splits"]["screen"]
    dev_groups = manifest["splits"]["dev"]
    dev_targets = dev_scored = screen_targets = screen_scored = None
    if has_valid:
        from experiments.devdata import load_targets
        # Share dev_targets_cache.npz in out_dir.parent so Arm 2 & Arm 3 reuse Arm 1's cached targets instantly!
        shared_cache = out_dir.parent / "dev_targets_cache.npz"
        dev_targets, dev_scored = load_targets(dev_groups, args.valid, store=shared_cache)
        g2i = {g: i for i, g in enumerate(dev_groups)}
        s_idx = [g2i[g] for g in screen_groups]
        screen_targets = dev_targets[s_idx]
        screen_scored = dev_scored[s_idx]
        if args.mode == "large_tick":
            # Evaluate screen strictly on large-tick sequences (0.4685 + 0.6569) when training specialist!
            sc_classes = dev_classes[s_idx]
            lt_mask = (sc_classes >= 0.45)
            screen_groups = [screen_groups[i] for i in range(len(screen_groups)) if lt_mask[i]]
            screen_targets = screen_targets[lt_mask]
            screen_scored = screen_scored[lt_mask]

    batches = build_training_schedule(
        args.train, args.valid, args.include_valid_split, args.mode,
        args.batch_size, args.steps, args.seed, manifest, train_classes, dev_classes
    )
    propensity = MaskPropensityScorer("artifacts/experiments/mask_propensity/model.npz")
    gpu_propensity = TorchMaskPropensityScorer(propensity, device)

    model = build_model(args.arch, args.hidden, device)
    ema_model = copy.deepcopy(model)
    n_params = sum(x.numel() for x in model.parameters())
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)
    print(f"Initialized {args.arch.upper()} (mode={args.mode}, hidden={args.hidden}, {n_params:,} params) on {gpu_name}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return float(step + 1) / float(max(1, args.warmup_steps))
        prog = float(step - args.warmup_steps) / float(max(1, args.steps - args.warmup_steps))
        return 0.03 + 0.97 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    best_screen_wp = -1.0
    best_state = None
    history = []
    latest_ckpt_path = out_dir / "latest_checkpoint.pt"

    if args.resume and latest_ckpt_path.exists():
        ckpt = torch.load(latest_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["online_state"])
        ema_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_step = int(ckpt["step"])
        best_screen_wp = float(ckpt.get("best_screen_wp", -1.0))
        best_state = ckpt.get("best_state", copy.deepcopy(ema_model.state_dict()))
        history = ckpt.get("history", [])
        print(f"Resumed at step {start_step}/{args.steps} (best_screen_wp={best_screen_wp:.5f})", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as init_pool:
            x_cal, _, _, _ = load_raw_sequence_batch(batches[0], init_pool)
        with torch.no_grad():
            x_t = torch.from_numpy(x_cal).to(device)
            f_btf = build_lob_engineered_features_torch(x_t).transpose(1, 2).contiguous()
            mu = f_btf.mean(dim=(0, 1), keepdim=True)
            sd = f_btf.std(dim=(0, 1), unbiased=False, keepdim=True) + 1e-4
            model.in_mu.copy_(mu)
            model.in_sd.copy_(sd)
            ema_model.load_state_dict(model.state_dict())
        best_state = copy.deepcopy(ema_model.state_dict())

    t_start = time.time()
    if start_step < args.steps:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            fut = pool.submit(load_raw_sequence_batch, batches[start_step], pool)
            for step in range(start_step, args.steps):
                t_step = time.time()
                x_np, y_np, exact_mw_np, has_exact_np = fut.result()
                if step + 1 < args.steps and not stop_flag["stop"]:
                    fut = pool.submit(load_raw_sequence_batch, batches[step + 1], pool)

                with torch.no_grad():
                    x_gpu = torch.from_numpy(x_np).to(device, non_blocking=True)
                    mw_t = gpu_propensity.batch_weights(x_gpu, args.mask_power, args.mask_floor)
                    if has_exact_np.any():
                        ex_idx = torch.from_numpy(np.flatnonzero(has_exact_np)).to(device)
                        mw_t[ex_idx] = torch.from_numpy(exact_mw_np[has_exact_np]).to(device, non_blocking=True)
                    feat_btf = build_lob_engineered_features_torch(x_gpu).transpose(1, 2).contiguous()
                    del x_gpu
                    y_bt2 = torch.from_numpy(y_np).to(device, non_blocking=True)

                    # Pass 1: exact 20,000-step trajectory + exact pooled Global WP + MSE coefficients
                    chunk = args.tbptt
                    model.eval()
                    s0, s1 = model.initial_states(len(batches[step]), device=device)
                    b_states, f_preds = [], []
                    for c_s in range(0, 20000, chunk):
                        c_e = min(c_s + chunk, 20000)
                        b_states.append((clone_state(s0), clone_state(s1)))
                        pc, s0, s1 = model.forward_features(feat_btf[:, c_s:c_e], s0, s1)
                        f_preds.append(pc)
                    pred_full = torch.cat(f_preds, dim=1)
                    _, coeff_full, tr_rho, tr_mse = exact_global_wp_coefficients(
                        pred_full, y_bt2, mw_t, mse_weight=args.mse_weight
                    )
                    coeff_full = coeff_full.to(device=device, dtype=torch.float32)

                # Pass 2: chunked TBPTT backward on CUDA
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for ic, c_s in enumerate(range(0, 20000, chunk)):
                    c_e = min(c_s + chunk, 20000)
                    if c_e <= WARMUP:
                        continue
                    cc = coeff_full[:, c_s:c_e]
                    if cc.abs().sum() == 0:
                        continue
                    bs0, bs1 = b_states[ic]
                    pc, _, _ = model.forward_features(feat_btf[:, c_s:c_e], bs0, bs1)
                    (pc * cc).sum().backward()

                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5))
                optimizer.step()
                scheduler.step()
                update_ema(ema_model, model, args.ema_decay)

                rec = {
                    "step": step + 1,
                    "lr": round(float(scheduler.get_last_lr()[0]), 7),
                    "train_rho": round(tr_rho, 5),
                    "train_mse": round(tr_mse, 5),
                    "grad_norm": round(gnorm, 4),
                    "sec": round(time.time() - t_step, 2),
                }
                if (step + 1) <= 5 or ((step + 1) % 20 == 0 and (step + 1) % args.eval_every != 0):
                    print(f"[{args.arch.upper()}] Step {step + 1:4d}/{args.steps} | rho={tr_rho:.4f} mse={tr_mse:.4f} "
                          f"gnorm={gnorm:.3f} lr={rec['lr']:.2e} ({rec['sec']:.2f}s/step)", flush=True)

                if has_valid and args.include_valid_split == "none" and (
                    (step + 1) % args.eval_every == 0 or step + 1 == args.steps
                ):
                    sc = evaluate_model_gpu(
                        ema_model, screen_groups, screen_targets, screen_scored, args.valid, device, executor=pool
                    )
                    rec["screen_wp_ema"] = sc["wp"]
                    print(f"[{args.arch.upper()}] Step {step + 1:4d}/{args.steps} | Screen WP (EMA)={sc['wp']:.5f} "
                          f"(t0={sc['t0']:.5f}, t1={sc['t1']:.5f}) [{time.time() - t_start:.0f}s]", flush=True)
                    if sc["wp"] > best_screen_wp:
                        best_screen_wp = sc["wp"]
                        best_state = {k: v.detach().cpu() for k, v in ema_model.state_dict().items()}
                        atomic_save({
                            "arch": args.arch,
                            "mode": args.mode,
                            "hidden": args.hidden,
                            "model_state": best_state,
                            "step": step + 1,
                            "screen_wp": best_screen_wp,
                            "args": vars(args),
                        }, out_dir / "best_model.pt")
                elif not has_valid or args.include_valid_split != "none":
                    # Without validation split, track latest EMA weights as selected state
                    best_state = {k: v.detach().cpu() for k, v in ema_model.state_dict().items()}

                history.append(rec)

                if (step + 1) % args.save_every == 0 or step + 1 == args.steps or stop_flag["stop"]:
                    atomic_save({
                        "arch": args.arch,
                        "mode": args.mode,
                        "hidden": args.hidden,
                        "step": step + 1,
                        "model_state": {k: v.detach().cpu() for k, v in ema_model.state_dict().items()},
                        "online_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        "best_state": {k: v.detach().cpu() for k, v in best_state.items()} if best_state is not None else None,
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "best_screen_wp": best_screen_wp,
                        "args": vars(args),
                        "history": history,
                    }, latest_ckpt_path)

                if stop_flag["stop"]:
                    print(f"Stopped cleanly at step {step + 1}; resume with `--resume`.", flush=True)
                    return

    # Finalize and evaluate on all 1,400 dev sequences if valid.parquet is present
    ema_cpu_state = {k: v.detach().cpu() for k, v in (best_state if best_state is not None else ema_model.state_dict()).items()}
    ema_model.load_state_dict(ema_cpu_state)

    # Save final_checkpoint.pt BEFORE running full 1,400-seq dev evaluation so weights are persisted immediately!
    atomic_save({
        "arch": args.arch,
        "mode": args.mode,
        "hidden": args.hidden,
        "model_state": ema_cpu_state,
        "args": vars(args),
        "best_screen_wp": best_screen_wp,
    }, out_dir / "final_checkpoint.pt")

    summary = {
        "arch": args.arch,
        "mode": args.mode,
        "hidden": args.hidden,
        "include_valid_split": args.include_valid_split,
        "best_screen_wp": best_screen_wp,
        "total_seconds": round(time.time() - t_start, 1),
        "gpu": gpu_name,
        "args": vars(args),
        "history": history,
    }

    if has_valid and args.include_valid_split == "none":
        print("Evaluating selected EMA model on all 1,400 dev sequences...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as eval_pool:
            dev_eval = evaluate_model_gpu(
                ema_model, dev_groups, dev_targets, dev_scored, args.valid, device, executor=eval_pool
            )
        by_class = {str(c): round(float(score_statistics(merge_statistics(dev_eval["stats"][dev_classes == c])).mean()), 5)
                    for c in np.unique(dev_classes)}
        np.save(out_dir / "dev_predictions.npy", dev_eval["preds"])
        np.save(out_dir / "dev_statistics.npy", dev_eval["stats"])
        summary.update({
            "dev_wp": dev_eval["wp"],
            "dev_t0": dev_eval["t0"],
            "dev_t1": dev_eval["t1"],
            "dev_by_class": by_class,
        })
        print(f"DONE {args.arch.upper()} ({args.mode}) Dev WP: {dev_eval['wp']:.5f} | by_class={by_class}", flush=True)

    atomic_save({
        "arch": args.arch,
        "mode": args.mode,
        "hidden": args.hidden,
        "model_state": ema_cpu_state,
        "args": vars(args),
        "summary": {k: v for k, v in summary.items() if k != "history"},
    }, out_dir / "final_checkpoint.pt")
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Saved final checkpoint to {out_dir / 'final_checkpoint.pt'}", flush=True)


if __name__ == "__main__":
    main()
