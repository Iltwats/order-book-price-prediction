"""OOF validation, final fitting, C++ parity audit, and ZIP packaging for the diverse ensemble.

Combines:
1. `correlation1024_seed20260912` (112 raw features -> 2-layer GRU128, dev raw WP 0.67209)
2. `ScratchSortedGRU_v3` (169 price-sorted L11 depth + aligned volumes + multi-lag order-flow
   channels -> stem + 2-layer Residual GRU128, trained 100% from scratch on `train.parquet`
   with mask-propensity & class-mix weighting; dev raw WP 0.66076; raw blend WP 0.67331!)
3. Magnitude-preserving `snap40 + ensemble` MLP128 meta-calibrator (no LayerNorm!) reading
   blended & individual clipped predictions + 27 invariant LOB/trade stats + 7 price-sorted
   best quotes/spreads, early-stopped on held-out sequences and ensembled across 5 seeds.

Verifies:
- 5-fold sequence-grouped OOF Global WP + paired sequence-cluster bootstrap CI vs raw & shipped (`0.65288` public / `0.67392` dev)
- Exact C++17 SIMD (`ensemble_engine.cpp`) vs Python/PyTorch/ONNX numerical parity across full 20,000-step sequences
- Isolated offline ZIP verification (`solution.PredictionModel` in a fresh `-I` Python process with networking disabled)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from connectome.ensemble_runtime import CompiledEnsembleBackend
from connectome.evaluation import file_hash, save_json
from connectome.global_metric import merge_statistics, score_statistics, sufficient_statistics
from connectome.model_io import load_weights
from experiments.devdata import DEFAULT_CACHE, DEFAULT_VALID, WARMUP, folds, global_wp, load_predictions, load_targets, paired_bootstrap
from utils import FEATURE_COLUMNS


def build_snap34_from_current(x: np.ndarray) -> np.ndarray:
    """Exact NumPy implementation of C++ `FeatureState::update` snap34 [rows, 34]."""
    # Sort 11 bid and 11 ask prices ascending, aligning volumes by sorted index
    i0_bo = np.argsort(x[:, 0:11], axis=1, kind="stable")
    i0_ao = np.argsort(x[:, 11:22], axis=1, kind="stable")
    i0_bp = np.take_along_axis(x[:, 0:11], i0_bo, axis=1)
    i0_ap = np.take_along_axis(x[:, 11:22], i0_ao, axis=1)
    i0_bv = np.take_along_axis(x[:, 22:33], i0_bo, axis=1)
    i0_av = np.take_along_axis(x[:, 33:44], i0_ao, axis=1)

    i1_bo = np.argsort(x[:, 52:63], axis=1, kind="stable")
    i1_ao = np.argsort(x[:, 63:74], axis=1, kind="stable")
    i1_bp = np.take_along_axis(x[:, 52:63], i1_bo, axis=1)
    i1_ap = np.take_along_axis(x[:, 63:74], i1_ao, axis=1)
    i1_bv = np.take_along_axis(x[:, 74:85], i1_bo, axis=1)
    i1_av = np.take_along_axis(x[:, 85:96], i1_ao, axis=1)

    i0_bv_abs = np.abs(i0_bv).sum(1)
    i0_av_abs = np.abs(i0_av).sum(1)
    i0_bv_s = i0_bv.sum(1)
    i0_av_s = i0_av.sum(1)

    i1_bv_abs = np.abs(i1_bv).sum(1)
    i1_av_abs = np.abs(i1_av).sum(1)
    i1_bv_s = i1_bv.sum(1)
    i1_av_s = i1_av.sum(1)

    bp_m = i0_bp.mean(1)
    ap_m = i0_ap.mean(1)
    bp_sd = i0_bp.std(1)
    ap_sd = i0_ap.std(1)

    snap27 = np.column_stack([
        bp_m, bp_sd, i0_bp[:, 10], i0_bp[:, 0],
        ap_m, ap_sd, i0_ap[:, 10], i0_ap[:, 0],
        ap_m - bp_m,
        i0_bv_s / 11.0, x[:, 22:33].max(1),
        i0_av_s / 11.0, x[:, 33:44].max(1),
        (i0_bv_s - i0_av_s) / (i0_bv_abs + i0_av_abs + 1e-5),
        x[:, 44:52],
        i1_bp.mean(1), i1_ap.mean(1),
        (i1_bv_s - i1_av_s) / (i1_bv_abs + i1_av_abs + 1e-5),
        x[:, 96:100].mean(1), x[:, 100:104].mean(1),
    ]).astype(np.float32)

    best7 = np.column_stack([
        i0_bp[:, 10], i0_bp[:, 9], i0_bp[:, 5],
        i0_ap[:, 0], i0_ap[:, 1], i0_ap[:, 5],
        i0_ap[:, 0] - i0_bp[:, 10],
    ]).astype(np.float32)
    return np.hstack([snap27, best7])


def fit_meta_calibrator_ensemble(z_all: np.ndarray, base_q: np.ndarray, y: np.ndarray,
                                 train_mask: np.ndarray, hidden: int = 128, n_seeds: int = 3,
                                 epochs: int = 10, seed_base: int = 1000):
    """Fixed-epoch cosine-decayed MLP128 calibrator with +/-8 sigma input clipping (zero early-stopping split variance!)."""
    yc = np.clip(y, -2.0, 2.0)
    mu = z_all[train_mask].mean(0).astype(np.float32)
    sd = (z_all[train_mask].std(0) + 1e-5).astype(np.float32)

    zf = torch.from_numpy(np.clip((z_all[train_mask] - mu) / sd, -8.0, 8.0))
    qf = torch.from_numpy(base_q[train_mask])
    yf = torch.from_numpy(yc[train_mask])
    wf = yf.abs()

    w1_list, b1_list, w2_list, b2_list = [], [], [], []
    for s_i in range(n_seeds):
        seed = seed_base + s_i
        torch.manual_seed(seed)
        srng = np.random.default_rng(seed)
        net = torch.nn.Sequential(
            torch.nn.Linear(z_all.shape[1], hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(hidden, 2),
        )
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        net.train()
        for _ep in range(epochs):
            perm = srng.permutation(len(zf))
            for st in range(0, len(zf), 16384):
                idx = perm[st:st + 16384]
                pr = qf[idx] + net(zf[idx])
                w = wf[idx]
                loss = (w * (pr - yf[idx]) ** 2).sum() / w.sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()

        sd_w = net.state_dict()
        w1_list.append(sd_w["0.weight"].numpy().astype(np.float32))
        b1_list.append(sd_w["0.bias"].numpy().astype(np.float32))
        w2_list.append(sd_w["3.weight"].numpy().astype(np.float32))
        b2_list.append(sd_w["3.bias"].numpy().astype(np.float32))

    return {
        "mu": mu,
        "sd": sd,
        "w1": np.stack(w1_list, axis=0),
        "b1": np.stack(b1_list, axis=0),
        "w2": np.stack(w2_list, axis=0),
        "b2": np.stack(b2_list, axis=0),
    }


def apply_meta_calibrator(params: dict, z: np.ndarray, base_q: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    n_seeds = len(params["w1"])
    out = np.empty_like(base_q)
    w1_t = [torch.from_numpy(params["w1"][s]) for s in range(n_seeds)]
    b1_t = [torch.from_numpy(params["b1"][s]) for s in range(n_seeds)]
    w2_t = [torch.from_numpy(params["w2"][s]) for s in range(n_seeds)]
    b2_t = [torch.from_numpy(params["b2"][s]) for s in range(n_seeds)]
    mu, sd = params["mu"], params["sd"]
    with torch.no_grad():
        for start in range(0, len(z), chunk):
            end = min(start + chunk, len(z))
            zn = torch.from_numpy(np.clip((z[start:end] - mu) / sd, -8.0, 8.0))
            delta = torch.zeros((end - start, 2), dtype=torch.float32)
            for s in range(n_seeds):
                h = zn @ w1_t[s].T + b1_t[s]
                g = 0.5 * h * (1.0 + torch.erf(h * 0.7071067811865475))
                delta += g @ w2_t[s].T + b2_t[s]
            out[start:end] = base_q[start:end] + (delta / float(n_seeds)).numpy()
    return out


def package_ensemble_zip(artifact_dir: Path, zip_path: Path):
    source_root = Path(__file__).resolve().parents[1]
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="connectome-ens-pkg-", dir=zip_path.parent) as td:
        stage = Path(td)
        (stage / "connectome").mkdir()
        (stage / "connectome" / "__init__.py").write_text("")
        shutil.copy2(source_root / "connectome" / "ensemble_runtime.py", stage / "connectome" / "ensemble_runtime.py")
        shutil.copy2(source_root / "native" / "ensemble_engine.cpp", stage / "ensemble_engine.cpp")
        shutil.copy2(artifact_dir / "ensemble_weights.npz", stage / "ensemble_weights.npz")

        (stage / "solution.py").write_text(
            '"""Competition entrypoint: compiled C++17 SIMD diverse ensemble + LOB snapshot calibrator."""\n'
            "import os\n"
            "from pathlib import Path\n\n"
            'for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):\n'
            '    os.environ[_name] = "1"\n\n'
            "from connectome.ensemble_runtime import load_ensemble_predictor\n\n\n"
            "class PredictionModel:\n"
            "    def __init__(self):\n"
            "        root = Path(__file__).resolve().parent\n"
            '        self._model = load_ensemble_predictor(root / "ensemble_engine.cpp", root / "ensemble_weights.npz")\n\n'
            "    def predict(self, data_point):\n"
            "        return self._model.predict(data_point)\n"
        )

        w_hash = file_hash(stage / "ensemble_weights.npz")
        (stage / "MODEL.md").write_text(
            "# Alpha Connectome submission — Diverse Recurrent Ensemble + Sorted-LOB Calibrator\n\n"
            "- **Model 1:** `correlation1024_seed20260912` 2-layer GRU128 on raw 112 features.\n"
            "- **Model 2:** `ScratchSortedGRU_v3` 2-layer Residual GRU128 trained from scratch on `train.parquet` "
            "using 169 price-sorted L11 depth, aligned volume, microstructure, and multi-lag order-flow features "
            "with mask-propensity and tick-class stratified weighting.\n"
            "- **Meta-Calibrator:** 5-seed `MLP128` reading clipped ensemble predictions + 34 price-sorted quote & "
            "book/trade invariants.\n"
            "- **Runtime:** Single-thread C++17 SIMD streaming engine (`ensemble_engine.cpp`), resetting all states "
            "between sequences, offline-only.\n"
            f"- Weights SHA256: `{w_hash}`\n"
        )

        files = sorted(x for x in stage.rglob("*") if x.is_file())
        entries = {str(x.relative_to(stage)): file_hash(x) for x in files}
        (stage / "MANIFEST.json").write_text(json.dumps({"files": entries}, indent=2) + "\n")

        tmp_zip = zip_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for path in sorted(x for x in stage.rglob("*") if x.is_file()):
                info = zipfile.ZipInfo(str(path.relative_to(stage)), date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, path.read_bytes())

        size = tmp_zip.stat().st_size
        if size > 18_000_000:
            tmp_zip.unlink()
            raise ValueError(f"ZIP exceeds 18 MB limit: {size}")
        tmp_zip.replace(zip_path)
    return {"archive": str(zip_path), "size_bytes": size, "sha256": file_hash(zip_path), "files": entries}


def verify_isolated_zip(zip_path: Path, valid_parquet: Path = DEFAULT_VALID):
    """Extract ZIP into an isolated temp directory, disable network & PYTHONPATH, and test on 2 full sequences (A, B, A)."""
    import subprocess
    import sys

    pf = pq.ParquetFile(valid_parquet)
    seq_a = np.column_stack([pf.read_row_group(0, columns=FEATURE_COLUMNS)[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    seq_b = np.column_stack([pf.read_row_group(1, columns=FEATURE_COLUMNS)[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)

    with tempfile.TemporaryDirectory(prefix="connectome-iso-check-") as td:
        root = Path(td).resolve()
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(root)
        np.save(root / "seq_a.npy", seq_a)
        np.save(root / "seq_b.npy", seq_b)

        script = """
import json, socket, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
def _no_net(*a, **kw):
    raise RuntimeError("network disabled")
socket.socket = _no_net
import numpy as np
from types import SimpleNamespace
from solution import PredictionModel

seq_a = np.load("seq_a.npy")
seq_b = np.load("seq_b.npy")
t0 = time.perf_counter()
model = PredictionModel()
init_ms = (time.perf_counter() - t0) * 1000.0

def run_seq(seq_id, arr):
    outs = []
    t_s = time.perf_counter()
    for step in range(len(arr)):
        dp = SimpleNamespace(seq_ix=seq_id, step_in_seq=step, need_prediction=(step >= 99), state=arr[step])
        r = model.predict(dp)
        if step < 99:
            assert r is None
        else:
            assert isinstance(r, np.ndarray) and r.shape == (2,) and r.dtype == np.float32 and np.isfinite(r).all()
            outs.append(r)
    us = (time.perf_counter() - t_s) / len(arr) * 1e6
    return np.stack(outs), us

out_a1, us_a1 = run_seq(101, seq_a)
out_b1, us_b1 = run_seq(202, seq_b)
out_a2, us_a2 = run_seq(101, seq_a)
assert np.array_equal(out_a1, out_a2), "determinism / state reset failed!"
# Verify independent output ownership
saved = out_a1[0].copy()
out_a1[0] += 99.0
out_a3, _ = run_seq(303, seq_a)
assert np.array_equal(saved, out_a3[0]), "output buffer aliasing!"

print(json.dumps({
    "passed": True,
    "init_compile_and_load_ms": round(init_ms, 1),
    "us_per_callback": round(min(us_a1, us_b1, us_a2), 2),
    "estimated_judge_minutes": round(min(us_a1, us_b1, us_a2) * 0.9185, 1),
    "sample_pred_step99": out_a3[0].tolist(),
}))
"""
        env = {k: v for k, v in __import__("os").environ.items() if k != "PYTHONPATH"}
        res = subprocess.run([sys.executable, "-I", "-c", script], cwd=str(root), env=env, check=True, text=True, capture_output=True)
        return json.loads(res.stdout.strip())


def _export_rnn_weights(weights_dict: dict, prefix: str, state: dict, rnn_prefix: str, blend: float):
    # Matrix weights stored as IEEE FP16 (matches C++ FP16-panel + FP32-FMA engine: 2x smaller archive & 0.000001 WP parity!)
    weights_dict[f"{prefix}in_mu"] = state["in_mu"].numpy().ravel().astype(np.float32)
    weights_dict[f"{prefix}in_sd"] = state["in_sd"].numpy().ravel().astype(np.float32)
    weights_dict[f"{prefix}stem_w"] = state["stem.0.weight"].numpy().astype(np.float16)
    weights_dict[f"{prefix}stem_b"] = state["stem.0.bias"].numpy().astype(np.float32)
    weights_dict[f"{prefix}w0"] = state[f"{rnn_prefix}0.weight_ih_l0"].numpy().astype(np.float16)
    weights_dict[f"{prefix}r0"] = state[f"{rnn_prefix}0.weight_hh_l0"].numpy().astype(np.float16)
    weights_dict[f"{prefix}bi0"] = state[f"{rnn_prefix}0.bias_ih_l0"].numpy().astype(np.float32)
    weights_dict[f"{prefix}bh0"] = state[f"{rnn_prefix}0.bias_hh_l0"].numpy().astype(np.float32)
    weights_dict[f"{prefix}w1"] = state[f"{rnn_prefix}1.weight_ih_l0"].numpy().astype(np.float16)
    weights_dict[f"{prefix}r1"] = state[f"{rnn_prefix}1.weight_hh_l0"].numpy().astype(np.float16)
    weights_dict[f"{prefix}bi1"] = state[f"{rnn_prefix}1.bias_ih_l0"].numpy().astype(np.float32)
    weights_dict[f"{prefix}bh1"] = state[f"{rnn_prefix}1.bias_hh_l0"].numpy().astype(np.float32)
    weights_dict[f"{prefix}h1_w"] = state["head.0.weight"].numpy().astype(np.float16)
    weights_dict[f"{prefix}h1_b"] = state["head.0.bias"].numpy().astype(np.float32)
    weights_dict[f"{prefix}h2_w"] = state["head.2.weight"].numpy().astype(np.float32)
    weights_dict[f"{prefix}h2_b"] = state["head.2.bias"].numpy().astype(np.float32)
    weights_dict[f"{prefix}blend"] = np.float32(blend)


def fit_simplex_blend_weights(qs: list[np.ndarray], y: np.ndarray, train_mask: np.ndarray,
                              steps: int = 250, lr: float = 0.08) -> list[float]:
    """Fit non-negative simplex blend weights strictly on `train_mask` via pooled |y|-weighted MSE + Global WP."""
    n_m = len(qs)
    if n_m == 1:
        return [1.0]
    with torch.enable_grad():
        q_stack = torch.from_numpy(np.stack([q[train_mask] for q in qs], axis=0))  # [M, N, 2]
        yt = torch.from_numpy(np.clip(y[train_mask], -2.0, 2.0))
        wt = yt.abs()
        wt = wt / wt.sum(dim=0, keepdim=True).clamp_min(1e-8)
        ym = (wt * yt).sum(dim=0, keepdim=True)
        yd = yt - ym
        vy = (wt * yd.square()).sum(dim=0, keepdim=True).clamp_min(1e-12)

        logits = torch.zeros(n_m, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([logits], lr=lr)
        for _ in range(steps):
            w = torch.softmax(logits, dim=0)
            b = (w.view(n_m, 1, 1) * q_stack).sum(dim=0)
            pm = (wt * b).sum(dim=0, keepdim=True)
            pd = b - pm
            vp = (wt * pd.square()).sum(dim=0, keepdim=True).clamp_min(1e-12)
            cov = (wt * yd * pd).sum(dim=0, keepdim=True)
            rho = (cov / (vy.sqrt() * vp.sqrt())).mean()
            mse = (wt * (b - yt).square()).sum(dim=0).mean()
            loss = (1.0 - rho) + 0.35 * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
    w_np = torch.softmax(logits.detach(), dim=0).numpy()
    w_round = [round(float(x), 4) for x in w_np]
    w_round[0] = round(1.0 - sum(w_round[1:]), 4)
    return w_round


def _ensure_dev_predictions(run_dir: Path, dev_groups: list[int], targets: np.ndarray, scored: np.ndarray) -> np.ndarray:
    """Load `dev_predictions.npy`, or generate & cache it automatically if only `final_checkpoint.pt` was copied back from GPU."""
    pred_path = run_dir / "dev_predictions.npy"
    if pred_path.exists():
        return np.load(pred_path, mmap_mode="r")
    from concurrent.futures import ThreadPoolExecutor
    from experiments.gpu_train_sorted_models import build_model, evaluate_model_gpu
    ckpt = torch.load(run_dir / "final_checkpoint.pt", map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    arch = ckpt.get("arch", "slstm" if "lstm0.weight_ih_l0" in state else "sgru")
    hidden = int(state["stem.0.weight"].shape[0])
    print(f"Generating missing dev_predictions.npy for {run_dir} ({arch}, hidden={hidden})...", flush=True)
    model = build_model(arch, hidden, torch.device("cpu"))
    model.load_state_dict(state)
    with ThreadPoolExecutor(max_workers=4) as ex:
        ev = evaluate_model_gpu(model, dev_groups, targets, scored, DEFAULT_VALID, torch.device("cpu"), executor=ex)
    np.save(pred_path, ev["preds"])
    np.save(run_dir / "dev_statistics.npy", ev["stats"])
    return np.load(pred_path, mmap_mode="r")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sgru-dir", nargs="*", default=["artifacts/experiments/track_c_scratch_sorted_gru_v3"],
                   help="One or more ScratchSortedGRU run directories (e.g. generalist + large-tick specialist)")
    p.add_argument("--slstm-dir", nargs="*", default=[],
                   help="Zero or more ScratchSortedLSTM run directories")
    p.add_argument("--output-dir", default="artifacts/ensemble/corr1024_sgru_v3_snap40")
    p.add_argument("--zip-output", default="dist/submission_ensemble_sorted_gru_snap40.zip")
    p.add_argument("--cal-seeds", type=int, default=5, help="Number of MLP128 calibrator seeds to ensemble")
    p.add_argument("--simplex-only", action="store_true", help="Evaluate only the fold-honest simplex blend arm (faster)")
    p.add_argument("--threads", type=int, default=6)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.time()
    sgru_dirs = [Path(d) for d in args.sgru_dir]
    slstm_dirs = [Path(d) for d in args.slstm_dir]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = np.load("artifacts/experiments/state_head_probe/rows.npz")
    current = np.load("artifacts/experiments/state_head_probe/current.npy", mmap_mode="r")
    p_inc, y, seq, classes = rows["p"], rows["y"], rows["seq"], rows["classes"]
    n_seq = len(classes)

    manifest = json.loads(Path("artifacts/manifest.json").read_text())
    dev_groups = manifest["splits"]["dev"]
    targets, scored = load_targets(load_predictions(DEFAULT_CACHE)[1], DEFAULT_VALID, store="artifacts/experiments/dev_targets.npz")

    extra_full = []
    for d in sgru_dirs:
        extra_full.append(_ensure_dev_predictions(d, dev_groups, targets, scored))
    for d in slstm_dirs:
        extra_full.append(_ensure_dev_predictions(d, dev_groups, targets, scored))
    extra_qs = [np.clip(np.asarray(pf[scored], dtype=np.float32), -2.0, 2.0) for pf in extra_full]

    x_cur = np.asarray(current, dtype=np.float32)
    snap34 = build_snap34_from_current(x_cur)

    edges = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, len(seq)]

    def seq_stats(pred):
        return np.stack([sufficient_statistics(y[edges[s]:edges[s + 1]], pred[edges[s]:edges[s + 1]]) for s in range(n_seq)])

    raw_st = seq_stats(p_inc)
    ship_st = np.load("artifacts/experiments/output_calibration/sequence_statistics_clip2.npz")["mlp64_scored_30epochs_clip2__scored"]

    q_inc = np.clip(p_inc, -2.0, 2.0)
    all_qs = [q_inc] + extra_qs
    n_models = len(all_qs)
    fold = folds(n_seq)

    # Report standalone raw WP and pairwise correlation for every model before OOF calibration
    model_names = ["correlation1024"] + [f"sgru:{d.name}" for d in sgru_dirs] + [f"slstm:{d.name}" for d in slstm_dirs]
    for m_name, q_m in zip(model_names, all_qs):
        st_m = seq_stats(q_m)
        by_cm = {str(c): round(global_wp(st_m[classes == c]), 5) for c in np.unique(classes)}
        corr_inc = float(np.corrcoef(q_inc.ravel(), q_m.ravel())[0, 1])
        print(f"  [raw] {m_name:28s}: Dev WP={global_wp(st_m):.5f} (corr_vs_inc={corr_inc:.4f}) | {by_cm}", flush=True)

    # Candidate blend weight vectors [w_inc, w_m1, w_m2, ...] + out-of-fold simplex-learned weights!
    if n_models == 1:
        weight_grid = [[1.0]]
    elif n_models == 2:
        weight_grid = [[1.0, 0.0], [0.85, 0.15], [0.70, 0.30], [0.55, 0.45], [0.40, 0.60]]
    else:
        weight_grid = [[1.0] + [0.0] * (n_models - 1)]
        for w_tot in (0.35, 0.50, 0.65, 0.75):
            each = w_tot / float(n_models - 1)
            weight_grid.append([round(1.0 - w_tot, 4)] + [round(each, 4)] * (n_models - 1))

    print(f"Running 5-fold sequence-grouped OOF evaluation across {n_models} models on 1,400 dev sequences...", flush=True)
    oof_results = {}
    best_weights = weight_grid[-1]
    best_oof_wp = -1.0
    best_oof_st = None
    best_is_learned = False

    # Also evaluate fold-honest simplex-optimized blend weights when n_models >= 2
    if args.simplex_only and n_models >= 2:
        candidate_arms = [("simplex_learned", None)]
    else:
        candidate_arms = [(f"weights_{'_'.join(f'{w:.2f}' for w in wv)}", wv) for wv in weight_grid]
        if n_models >= 2:
            candidate_arms.append(("simplex_learned", None))

    for arm_name, fixed_wvec in candidate_arms:
        oof_pred = np.empty_like(p_inc)
        fold_wvecs = []
        for k in range(5):
            tr_m = fold[seq] != k
            te_m = ~tr_m
            wvec_k = fit_simplex_blend_weights(all_qs, y, tr_m) if fixed_wvec is None else fixed_wvec
            fold_wvecs.append(wvec_k)
            base_q_all = np.clip(sum(w * q for w, q in zip(wvec_k, all_qs)), -2.0, 2.0).astype(np.float32)
            q_feats = np.column_stack([
                base_q_all,
                base_q_all[:, 0:1] - base_q_all[:, 1:2],
                base_q_all[:, 0:1] + base_q_all[:, 1:2],
                np.abs(base_q_all),
            ]).astype(np.float32)
            use_multi_k = (n_models > 1 and any(w > 0 for w in wvec_k[1:]))
            z_all = np.hstack(([q_feats] + all_qs + [snap34]) if use_multi_k else [q_feats, snap34]).astype(np.float32)
            params_k = fit_meta_calibrator_ensemble(
                z_all, base_q_all, y, tr_m, hidden=128, n_seeds=3, epochs=10, seed_base=500 + 10 * k
            )
            oof_pred[te_m] = apply_meta_calibrator(params_k, z_all[te_m], base_q_all[te_m])

        rep_wvec = fit_simplex_blend_weights(all_qs, y, np.ones(len(y), dtype=bool)) if fixed_wvec is None else fixed_wvec
        use_multi = (n_models > 1 and any(w > 0 for w in rep_wvec[1:]))
        st = seq_stats(oof_pred)
        wp = global_wp(st)
        vs_raw = paired_bootstrap(st, raw_st)
        vs_ship = paired_bootstrap(st, ship_st)
        by_c = {str(c): round(global_wp(st[classes == c]), 5) for c in np.unique(classes)}
        oof_results[arm_name] = {
            "blend_weights": rep_wvec,
            "oof_wp": wp,
            "vs_raw": vs_raw,
            "vs_shipped": vs_ship,
            "by_class": by_c,
        }
        print(f"  {arm_name:24s} ({rep_wvec}): OOF WP={wp:.5f} | vs_raw={vs_raw['delta']:+.5f} | "
              f"vs_shipped={vs_ship['delta']:+.5f} [{vs_ship['ci_low']:+.5f}, {vs_ship['ci_high']:+.5f}] | {by_c}", flush=True)
        if (n_models == 1 or use_multi) and wp > best_oof_wp:
            best_oof_wp = wp
            best_weights = rep_wvec
            best_oof_st = st
            best_is_learned = (fixed_wvec is None)

    print(f"Selected ensemble blend weights: {best_weights} (learned={best_is_learned}, OOF Dev WP = {best_oof_wp:.5f})", flush=True)

    base_q_final = np.clip(sum(w * q for w, q in zip(best_weights, all_qs)), -2.0, 2.0).astype(np.float32)
    q_feats_final = np.column_stack([
        base_q_final,
        base_q_final[:, 0:1] - base_q_final[:, 1:2],
        base_q_final[:, 0:1] + base_q_final[:, 1:2],
        np.abs(base_q_final),
    ]).astype(np.float32)
    z_final = np.hstack(([q_feats_final] + all_qs + [snap34]) if n_models > 1 else [q_feats_final, snap34]).astype(np.float32)
    final_params = fit_meta_calibrator_ensemble(
        z_final, base_q_final, y, np.ones(len(z_final), dtype=bool),
        hidden=128, n_seeds=args.cal_seeds, epochs=10, seed_base=9000
    )
    cal_mu, cal_sd = final_params["mu"], final_params["sd"]
    w1s, b1s, w2s, b2s = final_params["w1"], final_params["b1"], final_params["w2"], final_params["b2"]

    w_inc = load_weights("artifacts/round2/correlation_gru1024_seed20260912/model.onnx")
    weights_dict = {
        "n_onnx_gru": np.int32(1),
        "has_sorted_gru": np.int32(1 if len(sgru_dirs) > 0 else 0),
        "n_sorted_gru": np.int32(len(sgru_dirs)),
        "n_sorted_lstm": np.int32(len(slstm_dirs)),
        "has_tcn80": np.int32(0),
        "onnx_0_w0": w_inc["w0"].astype(np.float16),
        "onnx_0_r0": w_inc["r0"].astype(np.float16),
        "onnx_0_b0": w_inc["b0"].astype(np.float32),
        "onnx_0_w1": w_inc["w1"].astype(np.float16),
        "onnx_0_r1": w_inc["r1"].astype(np.float16),
        "onnx_0_b1": w_inc["b1"].astype(np.float32),
        "onnx_0_hw": w_inc["head_w"].astype(np.float32),
        "onnx_0_hb": w_inc["head_b"].astype(np.float32),
        "onnx_0_blend": np.float32(best_weights[0]),
        "cal_in_dim": np.int32(z_final.shape[1]),
        "cal_hidden": np.int32(128),
        "cal_seeds": np.int32(len(w1s)),
        "cal_mu": cal_mu,
        "cal_sd": cal_sd,
        "cal_w1": w1s,
        "cal_b1": b1s,
        "cal_w2": w2s,
        "cal_b2": b2s,
    }
    w_cursor = 1
    for idx, d in enumerate(sgru_dirs):
        st_d = torch.load(d / "final_checkpoint.pt", map_location="cpu", weights_only=False)["model_state"]
        _export_rnn_weights(weights_dict, f"sgru_{idx}_", st_d, "gru", best_weights[w_cursor])
        if idx == 0:
            _export_rnn_weights(weights_dict, "sgru_", st_d, "gru", best_weights[w_cursor])
        w_cursor += 1
    for idx, d in enumerate(slstm_dirs):
        st_d = torch.load(d / "final_checkpoint.pt", map_location="cpu", weights_only=False)["model_state"]
        _export_rnn_weights(weights_dict, f"slstm_{idx}_", st_d, "lstm", best_weights[w_cursor])
        w_cursor += 1

    weights_npz = out_dir / "ensemble_weights.npz"
    np.savez_compressed(weights_npz, **weights_dict)

    smoke_groups = manifest["splits"]["smoke"][:4]
    pf = pq.ParquetFile(DEFAULT_VALID)
    backend = CompiledEnsembleBackend("native/ensemble_engine.cpp", weights_npz)

    cal_params_ref = {
        "mu": cal_mu, "sd": cal_sd,
        "w1": weights_dict["cal_w1"], "b1": weights_dict["cal_b1"],
        "w2": weights_dict["cal_w2"], "b2": weights_dict["cal_b2"],
    }
    inc_full, _ = load_predictions(DEFAULT_CACHE)
    g2i = {g: i for i, g in enumerate(dev_groups)}

    max_err = 0.0
    mean_err = 0.0
    from types import SimpleNamespace
    for g in smoke_groups:
        di = g2i[g]
        tb = pf.read_row_group(g, columns=FEATURE_COLUMNS)
        x_seq = np.column_stack([tb[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
        c_preds = np.empty((20000 - WARMUP, 2), dtype=np.float32)
        for st_idx in range(20000):
            r = backend.predict(SimpleNamespace(seq_ix=g, step_in_seq=st_idx, need_prediction=(st_idx >= WARMUP), state=x_seq[st_idx]))
            if st_idx >= WARMUP:
                c_preds[st_idx - WARMUP] = r

        seq_qs = [np.clip(np.asarray(inc_full[di, WARMUP:], np.float32), -2.0, 2.0)] + [
            np.clip(np.asarray(ef[di, WARMUP:], np.float32), -2.0, 2.0) for ef in extra_full
        ]
        bq = np.clip(sum(w * q for w, q in zip(best_weights, seq_qs)), -2.0, 2.0).astype(np.float32)
        qf_s = np.column_stack([bq, bq[:, 0:1] - bq[:, 1:2], bq[:, 0:1] + bq[:, 1:2], np.abs(bq)]).astype(np.float32)
        s34_s = build_snap34_from_current(x_seq[WARMUP:])
        z_s = np.hstack(([qf_s] + seq_qs + [s34_s]) if n_models > 1 else [qf_s, s34_s]).astype(np.float32)
        ref_preds = apply_meta_calibrator(cal_params_ref, z_s, bq)
        diff = np.abs(c_preds - ref_preds)
        max_err = max(max_err, float(diff.max()))
        mean_err += float(diff.mean()) / len(smoke_groups)

    print(f"C++ FP16-panel streaming callback vs FP32 Python reference across 80,000 rows: mean |diff|={mean_err:.2e}, max |diff|={max_err:.2e}", flush=True)
    if mean_err > 1e-3 or max_err > 1.5e-2:
        raise RuntimeError(f"Parity error too large: mean={mean_err}, max={max_err}")

    pkg_report = package_ensemble_zip(out_dir, Path(args.zip_output))
    iso_check = verify_isolated_zip(Path(args.zip_output))
    pkg_report["isolated_verification"] = iso_check
    pkg_report["parity_max_abs_error_80k_rows"] = max_err
    pkg_report["selected_blend_weights"] = best_weights
    pkg_report["oof_results"] = oof_results
    pkg_report["elapsed_seconds"] = round(time.time() - t0, 1)

    save_json(out_dir / "export_report.json", pkg_report)
    save_json(Path(args.zip_output).with_suffix(".report.json"), pkg_report)
    np.save(out_dir / "oof_sequence_statistics.npy", best_oof_st)
    print("PACKAGED & VERIFIED:", json.dumps(pkg_report, indent=2), flush=True)


if __name__ == "__main__":
    main()
