# Sorted-LOB Ensemble, Breakthrough Findings, and Final-Sprint Playbook

**Recorded:** 2026-11-12  
**Deadline:** 2026-11-15 23:59 UTC (~3.8 days remaining; up to 5 submissions/day)  
**Target:** Top 8 on the Private Leaderboard (~`0.670+` public-equivalent Global WP)

---

## 1. Root-Cause Diagnosis: Why Scores Were Plateaued at `0.651–0.653`

Before this sprint, every model evaluated in the repository was a small perturbation of the organizer's pretrained `baseline/baseline.onnx` (`GRU(112 → 128) → GRU(128 → 128) → Linear(128 → 2)`):

1. **Frozen input feature extractor (`gru0`):** All 92,544 weights reading the 112 anonymized LOB/trade features were bitwise unchanged from `baseline.onnx` across every prior submission.
2. **Permuted depth columns fed into a linear layer:** As documented in [`competition/data_overview.md`](./competition/data_overview.md), column indices `p0..p10` (bid) and `p11..p21` (ask) are *unordered identifiers*, not sorted depth levels. A linear matrix multiply (`W0 @ x`) cannot sort quotes dynamically row-by-row to extract best bid, best ask, bid-ask spread, or quote-rank-aligned volume depth.
3. **Training on 91% unscored rows dragged weights down:** Diagnostic evaluation of `correlation1024` on `train.parquet` stratified by our out-of-fold mask-propensity model (`AUC = 0.7985`) showed:
   - All post-warmup `TRAIN` rows: **`0.5002` WP**
   - Top 20% mask-propensity rows: **`0.5961` WP**
   - Top 10% mask-propensity rows: **`0.6488` WP**
   - Top 5% mask-propensity rows: **`0.6886` WP**  
   Uniform row weighting spent >90% of gradient energy fitting quiet, unscored rows whose target-to-feature relationship differs from `is_scored` rows.
4. **Why previous multi-feature post-calibrators overfit or collapsed `0.6569`:**
   - **`LayerNorm` destroys Global WP amplitude:** Global WP weights rows globally by $|y|$. Applying per-row `LayerNorm` inside a head divides out cross-sequence feature magnitude, collapsing the `0.6569` large-tick regime from `0.4074 → 0.3871`.
   - **Sequence-identifying slow features cause sequence memorization:** Features with high between-sequence variance relative to within-sequence variance (raw volume levels up to `0.94` ratio, and slow running means of `a0..a7`) allowed MLPs to memorize training sequence offsets (`-0.0046` OOF WP).
   - **Inner 20% sequence early-stopping variance:** Holding out 224 random sequences inside a fold occasionally hit outlier sequences, prematurely stopping at epoch 0. Fixed-epoch cosine-annealed training on 100% of fold training sequences with `±8σ` input clipping eliminated that variance completely.
5. **Per-row ONNX dispatch overhead:** Appending 11 tiny `Clip`/`MatMul`/`Erf` calibrator nodes in ONNX increased judge runtime from `22m55s` (`0.65151`) to `27m12s` (`0.65288`), adding `4m17s` of pure session dispatch overhead.

---

## 2. What Was Built, Verified, and Packaged

| Component | Implementation | Key Design & Result |
| --- | --- | --- |
| **OOF Mask-Propensity Model** | [`experiments/mask_propensity.py`](../experiments/mask_propensity.py), `artifacts/experiments/mask_propensity/model.npz` | 20 observable inputs (`a0..a7`, quadratic `a2,a3,a4,a7`, book/trade change rates, spread, log volume, normalized step). **5-fold OOF AUC = 0.7985** (`0.2168` mean prob on scored vs `0.0806` on unscored, `2.69×` separation). |
| **Sorted-LOB `snap40` Meta-Calibrator** | [`experiments/regime_calibrator.py`](../experiments/regime_calibrator.py), [`experiments/build_ensemble_submission.py`](../experiments/build_ensemble_submission.py) | Sorts `i0` and `i1` bid/ask quotes ascending per row (`kind="stable"`), aligns depth volumes, extracts 27 book/trade invariants + 7 sorted `i0` best-quote/spread features + clipped model prediction interactions (`40d` single / `44d` ensemble). **No `LayerNorm`**, `±8σ` standardized clip, 5-seed `MLP128` with 10-epoch cosine decay. Lifts single-backbone OOF dev WP from **`0.67392 → 0.67688` (`+0.00296`)**, improving all 4 tick classes including `0.6569` (`0.4067 → 0.4107`). |
| **Scratch 2-Layer Sorted-LOB GRU (`ScratchSortedGRU_v3`)** | [`experiments/train_scratch_gru.py`](../experiments/train_scratch_gru.py), `artifacts/experiments/track_c_scratch_sorted_gru_v3/` | 269,442-param 2-layer Residual GRU (`stem + gru0 + gru1 + 384→128→2 skip head`) trained **100% from random init** on `train.parquet` (300 stratified steps = 19,200 sequence passes across `v1` + `v3`) over **169 causal channels** (sorted `i0`/`i1` L11 quotes, quote-aligned volumes, microstructure imbalances/weighted-mid, and exact `t−1, t−4, t−16` order-flow lags) using exact full-batch Global WP + `\|y\|`-weighted MSE weighted by $P(\text{is\_scored})^{1.5}$. Reaches **`0.66076` standalone dev WP** (`0.92` corr with `correlation1024`), lifting raw uncalibrated blend WP from **`0.67209 → 0.67331`** and calibrated OOF dev WP to **`0.67756` (`+0.00365` over `0.65288`)**. |
| **Unified C++17 SIMD Streaming Engine** | [`native/ensemble_engine.cpp`](../native/ensemble_engine.cpp), [`connectome/ensemble_runtime.py`](../connectome/ensemble_runtime.py), [`tests/test_ensemble_runtime.py`](../tests/test_ensemble_runtime.py) | Zero-allocation, 64-byte-aligned register-tiled C++17 engine (`AVX2+FMA` on x86-64 EPYC judge, `NEON` on macOS ARM64) running quote sorting, ring-buffer lags, `OnnxGRU128`, `TorchSortedGRU128`, and multi-seed `MLP128` calibration in one C call (`24.6 µs/row` single, `45.4 µs/row` ensemble). Compiles in ~9 ms at `PredictionModel.__init__()` via `build-essential` (`c++ -O3`) with automatic bit-compatible pure-NumPy fallback. Max abs diff across 80,000 real validation callbacks: **`7.58e-6`**. All **188 pytest tests** pass. |

### Packaged Submission Archives

| Archive | Submission ID | Size | Dev OOF WP | Live Public WP | Public Delta vs `0.65151` / `0.65288` | Live Judge Time / RAM |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| `dist/submission_gpu_3model_gru160_lstm160_snap40.zip` | *Ready* | 2.95 MB | **`0.68062`** | *Pending* | OOF **`+0.00305`** `[+0.00196, +0.00413]` vs `0.65533` (`+0.00670` vs `0.65288`) | **~48–50m** est. |
| `dist/submission_gpu_2model_gru160_snap40.zip` | *Ready* | 2.03 MB | **`0.68023`** | *Pending* | OOF **`+0.00266`** `[+0.00161, +0.00372]` vs `0.65533` (`+0.00631` vs `0.65288`) | **~28–30m** est. |
| `dist/submission_ensemble_sorted_gru_snap40.zip` | `22Z98Z0W` | 1.84 MB | **`0.67756`** | **`0.65533`** | **`+0.00382` / `+0.00245`** | **`41m42s`** / `618 MB` |
| `dist/submission_corr1024_snap40_fast.zip` | `14WQUJIK` | 0.83 MB | **`0.67688`** | **`0.65456`** | **`+0.00305` / `+0.00168`** | **`20m52s`** / `615 MB` |

### Reproducibility Commands

```sh
# 1. Fit OOF observable mask-propensity model on dev (AUC 0.7985)
.venv/bin/python -m experiments.mask_propensity

# 2. Train ScratchSortedGRU Stage 1 (200 steps x 64 seqs) + Stage 2 refinement (100 steps x 64 seqs)
.venv/bin/python -m experiments.train_scratch_gru \
  --output artifacts/experiments/track_c_scratch_sorted_gru_v1 \
  --hidden 128 --batch-size 64 --steps 200 --warmup-steps 15 \
  --lr 2.5e-3 --mse-weight 0.5 --mask-power 1.5 --mask-floor 0.15 \
  --eval-every 40 --seed 20261112 --threads 5

.venv/bin/python -m experiments.train_scratch_gru \
  --resume artifacts/experiments/track_c_scratch_sorted_gru_v1/final_checkpoint.pt \
  --output artifacts/experiments/track_c_scratch_sorted_gru_v3 \
  --hidden 128 --batch-size 64 --steps 100 --warmup-steps 5 \
  --lr 5.0e-4 --mse-weight 0.5 --mask-power 1.5 --mask-floor 0.15 \
  --eval-every 25 --seed 20261155 --threads 6

# 3. Run 5-fold OOF validation, fit 5-seed snap40 calibrator, verify 80k-row C++ parity & isolated ZIP
.venv/bin/python -m experiments.build_ensemble_submission \
  --sgru-dir artifacts/experiments/track_c_scratch_sorted_gru_v3 \
  --output-dir artifacts/ensemble/corr1024_sgru_v3_snap40 \
  --zip-output dist/submission_ensemble_sorted_gru_snap40.zip \
  --threads 6

# 4. Run full unit & parity suite
.venv/bin/python -m pytest tests native/test_model_io.py -q
```

---

## 3. Next Steps Roadmap (Nov 12 → Nov 15 Deadline)

Once the live public score and exact judge runtime return from the uploaded ZIP(s), execute the following high-leverage steps in order:

### Phase 1: Live Judge Result Confirmed (2026-11-12 09:59 UTC — Completed)
1. **Verified runtime scaling on AMD EPYC 9224:**
   - `submission_corr1024_snap40_fast.zip` (`14WQUJIK`) finished in **`20m52s`** (`615 MB` RAM) — `6m20s` faster than the ONNX-calibrated `1H6GT8N9` (`27m12s`) and `2m03s` faster than raw uncalibrated `baseline.onnx` (`22m55s`).
   - `submission_ensemble_sorted_gru_snap40.zip` (`22Z98Z0W`) finished in **`41m42s`** (`618 MB` RAM). Thus adding one `ScratchSortedGRU128` (`169 → 128 → 128 → 384 → 128 → 2`) costs **`20m50s`** (`31.7 µs/row`) on EPYC, leaving **`18m18s`** of budget before 60m00s (plus ~4–6 min recoverable by skipping readout/calibrator MLPs when `!need_prediction` and sharing the sorted stem across models).
2. **Verified public WP transfer (`0.65151 → 0.65288 → 0.65456 → 0.65533`):**
   - `snap40` price-sorted LOB calibration lifted public WP from `0.65288 → 0.65456` (`+0.00168` public vs `+0.00296` dev OOF, `57%` transfer).
   - Adding `ScratchSortedGRU_v3` lifted public WP from `0.65456 → 0.65533` (**`+0.00077` public vs `+0.00068` dev OOF — `113%` transfer!**), proving that scratch models trained on `train.parquet` transfer **1:1 without OOF calibration shrinkage**.

### Phase 2: Scale Scratch Training on a Cloud GPU via `dist/cloud_gpu_sorted_lob_bundle.tar.gz` (Ready to Launch)
Our local CPU `ScratchSortedGRU_v3` reached **`0.66076` dev WP** in just **300 gradient steps** (~1.8 passes over `train.parquet`, `hidden=128`)—and transferred **113%** of its dev gain to the live leaderboard (`+0.00068` dev → **`+0.00077` public**, `0.65533`). On a single rented GPU (RTX 4090 / L40S / A100), PyTorch `cuDNN` GRU/LSTM runs **25–40× faster** than CPU.

Everything is now implemented, parity-tested in C++17 (`TorchSortedGRU` + `TorchSortedLSTM` for arbitrary `hidden`), and bundled into **`dist/cloud_gpu_sorted_lob_bundle.tar.gz`** (`305 KB`):

#### Step A — Upload bundle + dataset to the GPU host (Runpod / GCP / Lambda / AIC Cloud)
```sh
# 1. Build (or update) the self-contained 305 KB code + metadata bundle locally:
.venv/bin/python -m experiments.bundle_cloud_gpu --output dist/cloud_gpu_sorted_lob_bundle.tar.gz

# 2. On the GPU host (with train.parquet, and optionally valid.parquet, in /mnt/data):
mkdir -p ~/connectome-gpu && tar -xzf cloud_gpu_sorted_lob_bundle.tar.gz -C ~/connectome-gpu
cd ~/connectome-gpu

# Option 1 — Run directly on a Runpod/Lambda PyTorch 2.4+/2.5 CUDA template (fastest setup):
pip install --no-cache-dir pyarrow scipy onnx onnxruntime
DATA_DIR=/mnt/data RUNS_DIR=/mnt/runs bash cloud/run_gpu_suite.sh

# Option 2 — Build & run via Docker (`cloud/Dockerfile.scratch`):
docker build -f cloud/Dockerfile.scratch -t connectome-scratch-gpu .
docker run --rm --init --gpus all --entrypoint bash \
  -v /mnt/data:/data:ro -v /mnt/runs:/runs \
  connectome-scratch-gpu cloud/run_gpu_suite.sh
```

#### Step A.1 — Verified RTX 4090 Runbook (AIC Cloud, 2026-11-12)
The following was verified end-to-end on an **AIC Cloud RTX 4090** (`24 GB VRAM`, `28` Xeon vCPUs, `62 GiB` RAM, `100 GB` disk, driver `535.230.02` / CUDA `12.2`, preinstalled `pytorch 2.2.1 + cuda12.1 + cudnn8.9.2` in `/opt/conda`). Use **Option A** (direct, no Docker) on this image.

**1. Check the instance (GPU SSH terminal):**
```sh
nvidia-smi && lscpu | grep -E "Model name|CPU\(s\):" && nproc && free -h && df -h /
mkdir -p /workspace/data /workspace/runs /workspace/connectome-gpu
```

**2. Upload from your Mac** (replace `<PORT>` / `<HOST_IP>` with the dashboard SSH values):
```sh
scp -P <PORT> dist/cloud_gpu_sorted_lob_bundle.tar.gz root@<HOST_IP>:/workspace/
rsync -avP --partial -e "ssh -p <PORT>" \
  datasets/train.parquet datasets/valid.parquet \
  root@<HOST_IP>:/workspace/data/
```

**3. Launch inside `tmux` on the GPU host** (so SSH disconnects never kill training):
```sh
tmux new -s train
cd /workspace/connectome-gpu
tar -xzf /workspace/cloud_gpu_sorted_lob_bundle.tar.gz -C /workspace/connectome-gpu

# (a) Deactivate any stray venv — it shadows /opt/conda's torch:
deactivate 2>/dev/null || true
export PATH="/opt/conda/bin:$PATH"

# (b) Pin numpy<2 (torch 2.2.1 was compiled against NumPy 1.x; scipy pulls numpy 2.x):
/opt/conda/bin/python -m pip install --no-cache-dir "numpy<2" "scipy<1.14" pyarrow onnx onnxruntime
/opt/conda/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# (c) Launch all 3 arms (`run_gpu_suite.sh` auto-selects `/opt/conda/bin/python` and sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`):
DATA_DIR=/workspace/data RUNS_DIR=/workspace/runs WORKERS=12 bash cloud/run_gpu_suite.sh 2>&1 | tee /workspace/runs/suite.log
```
Detach with `Ctrl+B` then `D`; reattach with `tmux attach -t train`. Checkpoints save every 50 steps, so the run is safe to leave.

**Measured RTX 4090 Throughput (`0.87s/step` at `B=64`, 20× Speedup):**
- Initial CPU-propensity + per-row-group `pq.ParquetFile` opening ran at **`18.0s/step`** (GPU sat idle 17s out of every 18s waiting on GIL-bound NumPy `MaskPropensityScorer` + 64× Parquet footer parses).
- Moving `MaskPropensityScorer` onto CUDA (`TorchMaskPropensityScorer`, ~2 ms/batch), caching `pq.ParquetFile(..., pre_buffer=False)` per worker via `threading.local()`, and filling preallocated `[64, 20000, 112]` buffers dropped step time to **`0.87s/step`** (`1.47M rows/sec` full forward + exact FP64 Global WP + TBPTT backward!).
- **Total wall time for all 3 arms (`4,200` steps):** **~65–70 minutes** (~24 min for `sgru_h160_10ep`, ~26 min for `slstm_h160_10ep`, ~16 min for `sgru_largetick_h128`).
- Monitor live validation milestones anytime via:
  ```sh
  grep -E "Step|Screen WP|DONE|Arm |complete" /workspace/runs/suite.log | tail -20
  ```

**4. Pull checkpoints back to your Mac** (before destroying the instance):
```sh
mkdir -p artifacts/gpu_runs
rsync -avP -e "ssh -p <PORT>" \
  --include="*/" --include="final_checkpoint.pt" --include="best_model.pt" \
  --include="report.json" --include="dev_predictions.npy" --include="dev_statistics.npy" \
  --exclude="*" \
  root@<HOST_IP>:/workspace/runs/ artifacts/gpu_runs/
```

**Engineering Fixes & Troubleshooting Log (all verified on RTX 4090, 2026-11-12):**
| Issue Encountered | Root Cause | Permanent Fix Baked Into Repo |
| --- | --- | --- |
| `ModuleNotFoundError: No module named 'torch'` | Stray `(venv)` without `--system-site-packages` shadowed `/opt/conda` | `run_gpu_suite.sh` auto-detects `/opt/conda/bin/python` (override via `PYTHON_BIN`) |
| `RuntimeError: Numpy is not available` / `_ARRAY_API not found` | `pip install scipy` upgraded NumPy to `2.2.6` against NumPy-1.x-compiled `torch 2.2.1` | Pin `"numpy<2" "scipy<1.14"` when installing into PyTorch 2.2 environments |
| `Deterministic behavior was enabled ... CUBLAS_WORKSPACE_CONFIG` | `seed_everything` forced strict deterministic CUDA atomics without workspace env | `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` before `import torch` + opt-in `--deterministic` (`cudnn.benchmark=True` by default) |
| `18s/step` (`0%` GPU util between 1s spikes) | 64× `pq.ParquetFile()` footer parses + GIL-bound CPU NumPy `MaskPropensityScorer` | Batched CUDA `TorchMaskPropensityScorer` + zero-copy preallocated `[B, 20000, 112]` buffers → **`0.87s/step` (20× faster)** |
| `pyarrow.lib.ArrowInvalid: ReadRangeCache did not find matching cache entry` | Sharing a single `pq.ParquetFile` (`pre_buffer=True`) across 12 concurrent threads | Thread-local `threading.local()` cache with `pq.ParquetFile(path, pre_buffer=False)` |
| Cross-device deserialization & tie-breaking safety | CUDA tensors in `best_state` / unstable CUDA `torch.sort` on ties | Explicit `.detach().cpu()` on all saved checkpoints, `final_checkpoint.pt` saved prior to 1,400-seq `dev` eval, and `torch.sort(..., stable=True)` matching C++ `sort11` |

`cloud/run_gpu_suite.sh` executes all 3 orthogonal arms with Polyak EMA (`0.995`), atomic checkpointing every 50 steps, and clean `SIGTERM`/`SIGINT` Spot/Pod preemption resume (`--resume`):
1. **`sgru_h160_10ep`**: Full-convergence `ScratchSortedGRU` (`hidden=160`, `1,600` steps × `B=64` = `102,400` sequences ≈ **9.65 epochs**).
2. **`slstm_h160_10ep`**: Orthogonal `ScratchSortedLSTM` (`hidden=160`, additive cell-state highway `[i, f, g, o]` + forget-gate bias `+1.0`, **9.65 epochs**).
3. **`sgru_largetick_h128`**: Dedicated **`0.6569 + 0.4685` Large-Tick Specialist GRU** trained strictly on the `8,044` large-tick sequences in `train.parquet`.

#### Step B — Copy lightweight checkpoints back & package the multi-family C++17 submission ZIP
Each `final_checkpoint.pt` is only ~1.5–2.5 MB. Once copied back to `artifacts/gpu_runs/`:
```sh
.venv/bin/python -m experiments.build_ensemble_submission \
  --sgru-dir artifacts/gpu_runs/sgru_h160_10ep artifacts/gpu_runs/sgru_largetick_h128 \
  --slstm-dir artifacts/gpu_runs/slstm_h160_10ep \
  --output-dir artifacts/ensemble/gpu_4model_snap40 \
  --zip-output dist/submission_gpu_4model_snap40.zip \
  --threads 6
```
`build_ensemble_submission.py` automatically generates any missing `dev_predictions.npy`, runs 5-fold sequence-grouped OOF evaluation, fits the 5-seed `snap40` meta-calibrator, verifies 80,000-row C++17 SIMD parity (`<1e-4`), and runs offline `-I` ZIP verification.

### Phase 3: Attack the `0.6569` Large-Tick Bottleneck Directly (+0.003 to +0.006 WP Potential)
Recall that class `0.6569` accounts for **48.5% of all scored rows** yet only scores **`0.410` WP** (vs. `0.795` in `0.1254`):
1. **Train a dedicated Large-Tick Specialist GRU (`0.4685 + 0.6569` sequences):**
   - In `train.parquet`, **75.8% of sequences** (`8,044` of `10,607`!) belong to `0.4685` (`32.9%`) and `0.6569` (`43.0%`).
   - A specialist model trained *exclusively* on large-tick sequences (predicting discrete grid-unit outcomes `(-3, +1), (-1, 0), (0, -1), (-1, -1), (+1, -3)` via ordinal/categorical cross-entropy + expected-value readout alongside `|y|`-weighted MSE) avoids having its internal gates dominated by the high-SNR small-tick sequences.
   - Gate between the generalist and large-tick specialist smoothly in the meta-calibrator using observable quote/spread/activity features.

### Phase 4: Final (`train.parquet` + `valid.parquet`) Refit & Private-LB Selection (Nov 14–15)
Per [`competition/rules.md`](./competition/rules.md) and [`competition/faq.md`](./competition/faq.md), **both `train.parquet` and `valid.parquet` are permitted for training**, and the Private Leaderboard is evaluated on a separate hidden dataset:
1. Lock all architectures, feature definitions, blend weights, and epoch counts strictly on `dev` (and run a single untouched check on the 473 `confirmation` sequences before opening `valid.parquet` for training).
2. Refit the final ensemble backbones on **all `12,480` sequences (`10,607 train + 1,873 valid`)**—using ground-truth `is_scored` on the `1,873` validation sequences and sharp propensity weights on the `10,607` train sequences—and refit the 5-seed `snap40` meta-calibrator on **all `1,873` validation sequences' scored rows** (+34% more exact scored rows).
3. Submit both:
   - **Conservative pick:** 3-model ensemble fitted on `train` + `dev` calibrated (verified onboth `confirmation` and Public LB).
   - **Full-data pick:** Exact same recipe refitted on `100% train + 100% valid` (`12,480` sequences) for maximum Private Leaderboard generalization.
