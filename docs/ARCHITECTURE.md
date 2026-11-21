# Sorted-LOB Recurrent Ensemble & C++17 SIMD Inference Architecture

This document summarizes the technical findings, model architectures, GPU training pipeline, and C++17 SIMD inference engine behind the final competition solution.

---

## 1. Root-Cause Diagnosis: Breaking the Baseline Plateau (`0.651 → 0.657`)

Early iterations built on the starter-pack `baseline/baseline.onnx` (`GRU(112 → 128) → GRU(128 → 128) → Linear(128 → 2)`) plateaued around `0.651–0.653` public WP for five structural reasons:

1. **Frozen input feature extractor (`gru0`):** All 92,544 weights reading the 112 anonymized LOB/trade features were unchanged from `baseline.onnx`.
2. **Unordered depth columns fed into a linear layer:** Column indices `p0..p10` (bid) and `p11..p21` (ask) are unordered identifiers rather than sorted price levels. A fixed linear projection (`W0 @ x`) cannot sort quotes dynamically row-by-row to extract best bid, best ask, spread, or quote-rank-aligned depth.
3. **Training uniformly over ~91% unscored rows:** Evaluating `correlation1024` on `train.parquet` stratified by our out-of-fold mask-propensity model (`AUC = 0.7985`) revealed:
   - All post-warmup `TRAIN` rows: **`0.5002` WP**
   - Top 20% mask-propensity rows: **`0.5961` WP**
   - Top 10% mask-propensity rows: **`0.6488` WP**
   - Top 5% mask-propensity rows: **`0.6886` WP**  
   Uniform row weighting spent over 90% of gradient energy on quiet, unscored rows.
4. **Why naive multi-feature post-calibrators overfit or collapsed `0.6569`:**
   - **`LayerNorm` destroys Global WP amplitude:** Global WP weights rows globally by $|y|$. Per-row `LayerNorm` divides out cross-sequence feature magnitude, collapsing the `0.6569` large-tick class from `0.4074 → 0.3871`.
   - **Sequence-identifying slow features cause sequence memorization:** Features with high between-sequence variance (raw volume levels, slow running means of `a0..a7`) enabled MLPs to memorize training sequences (`-0.0046` OOF WP).
   - **Inner early-stopping split variance:** Fixed-epoch cosine-annealed training on 100% of fold training sequences with `±8σ` input clipping eliminated validation-split noise.
5. **Per-row ONNX dispatch overhead:** Appending 11 small `Clip`/`MatMul`/`Erf` calibrator nodes in ONNX added `4m17s` of per-row ORT dispatch overhead (`22m55s → 27m12s`).

---

## 2. Core Components

| Component | Implementation | Key Design & Result |
| --- | --- | --- |
| **OOF Mask-Propensity Model** | [`experiments/mask_propensity.py`](../experiments/mask_propensity.py), [`artifacts/experiments/mask_propensity/model.npz`](../artifacts/experiments/mask_propensity/model.npz) | 20 observable inputs (`a0..a7`, quadratic `a2,a3,a4,a7`, book/trade change rates, spread, log volume, normalized step). **5-fold OOF AUC = 0.7985** (`0.2168` mean prob on scored vs `0.0806` on unscored, `2.69×` separation). |
| **Sorted-LOB `snap40` Meta-Calibrator** | [`experiments/regime_calibrator.py`](../experiments/regime_calibrator.py), [`experiments/build_ensemble_submission.py`](../experiments/build_ensemble_submission.py) | Sorts `i0` and `i1` bid/ask quotes ascending per row (`kind="stable"`), aligns depth volumes, extracts 27 book/trade invariants + 7 sorted `i0` best-quote/spread features + clipped model prediction interactions (`40d` single / `44d–46d` ensemble). **No `LayerNorm`**, `±8σ` standardized clip, 5-seed `MLP128` with 10-epoch cosine decay. |
| **Scratch Sorted-LOB RNNs (`ScratchSortedGRU` & `ScratchSortedLSTM`)** | [`experiments/train_scratch_gru.py`](../experiments/train_scratch_gru.py), [`experiments/gpu_train_sorted_models.py`](../experiments/gpu_train_sorted_models.py) | 2-layer Residual GRU/LSTM (`hidden=160`) trained **100% from random init** on `train.parquet` over **169 causal channels** (sorted `i0`/`i1` L11 quotes, quote-aligned volumes, microstructure imbalances/weighted-mid, and exact `t−1, t−4, t−16` order-flow lags) using exact full-batch Global WP + `\|y\|`-weighted MSE weighted by $P(\text{is\_scored})^{1.5}$ and Polyak EMA (`0.995`). |
| **Unified C++17 SIMD Streaming Engine** | [`native/ensemble_engine.cpp`](../native/ensemble_engine.cpp), [`connectome/ensemble_runtime.py`](../connectome/ensemble_runtime.py) | Zero-allocation, 64-byte-aligned, FP16-panel register-tiled C++17 engine (`AVX2+F16C+FMA` on x86-64, `NEON` on ARM64) running quote sorting, ring-buffer lags, `OnnxGRU128`, `TorchSortedGRU`, `TorchSortedLSTM`, and multi-seed `MLP128` calibration in a single C call, with pure-NumPy fallback. |

---

## 3. Model Progression & Benchmarks

| Submission / Configuration | Blend Weights | Dev OOF WP | Public WP | Judge Time (1 vCPU) / RAM |
| --- | --- | ---: | ---: | ---: |
| `baseline/baseline.onnx` (starter baseline) | — | `0.61033` | — | — |
| `correlation1024_seed20260912` (single ONNX GRU128) | `[1.0]` | `0.67209` | `0.65151` | `22m55s` |
| `correlation1024` + ONNX `mlp64_clip2` calibrator | `[1.0]` | `0.67392` | `0.65288` | `27m12s` / `625 MB` |
| `correlation1024` + C++17 `snap40` calibrator (`14WQUJIK`) | `[1.0]` | `0.67688` | `0.65456` | `20m52s` / `615 MB` |
| CPU 2-model (`corr1024` + `ScratchSortedGRU-128`) (`22Z98Z0W`) | `[0.85, 0.15]` | `0.67756` | `0.65533` | `41m42s` / `618 MB` |
| **GPU 2-model (`corr1024` + `ScratchSortedGRU-160`) (`2QU7CNP2`)** | `[0.2663, 0.7337]` | `0.68023` | `0.65662` | `33m51s` / `664 MB` |
| **GPU 3-model (`corr1024` + `ScratchSortedGRU-160` + `ScratchSortedLSTM-160`) (`22GU0AJ2`)** | `[0.2137, 0.4331, 0.3532]` | **`0.68062`** | **`0.65694`** | `51m27s` / `656 MB` |

Note on 3-model vs 2-model:
- Standalone raw dev WP: `correlation1024` = `0.67209`, `ScratchSortedGRU-160` = `0.67499`, `ScratchSortedLSTM-160` = `0.67437`.
- `correlation1024` retains genuine prediction diversity against the scratch models (`r = 0.965` vs `r = 0.983` between GRU and LSTM), so keeping it in the blend adds `+0.0017` raw WP over GRU+LSTM alone.
- The 2-model GPU ensemble achieves `99.95%` of the 3-model score while running in `33m51s` instead of `51m27s`.

---

## 4. Cloud GPU Training Pipeline (`0.87s/step` on RTX 4090)

To train full-convergence `hidden=160` scratch models (~10 epochs over `10,607` sequences):

1. Build the self-contained training bundle (~300 KB, excludes datasets):
   ```sh
   python -m experiments.bundle_cloud_gpu --output dist/cloud_gpu_sorted_lob_bundle.tar.gz
   ```
2. On a CUDA host with `train.parquet` (and optionally `valid.parquet`) in `/data`:
   ```sh
   DATA_DIR=/data RUNS_DIR=/runs WORKERS=12 bash cloud/run_gpu_suite.sh
   ```
3. Or build and run with Docker (`cloud/Dockerfile.scratch`):
   ```sh
   docker build -f cloud/Dockerfile.scratch -t connectome-scratch-gpu .
   docker run --rm --init --gpus all --entrypoint bash \
     -v /path/to/data:/data:ro -v /path/to/runs:/runs \
     connectome-scratch-gpu cloud/run_gpu_suite.sh
   ```

Key GPU throughput optimizations implemented in [`experiments/gpu_train_sorted_models.py`](../experiments/gpu_train_sorted_models.py):
- **CUDA `TorchMaskPropensityScorer`:** Avoids GIL-bound NumPy scoring per batch (~2 ms/batch on GPU).
- **Thread-local Parquet caching (`pre_buffer=False`):** Prevents footer re-parsing and `ArrowInvalid` read-range race conditions across concurrent data-loader threads.
- **Preallocated `[B, 20000, 112]` pinned host buffers:** Dropped RTX 4090 step time from `18.0s/step` to **`0.87s/step`** (`1.47M rows/sec`).

---

## 5. Rebuilding & Packaging the Ensemble Locally

Once `final_checkpoint.pt` checkpoints are placed under `outputs/` (or another run directory):

```sh
# 1. Fit OOF observable mask-propensity model on dev (if regenerating from scratch)
python -m experiments.mask_propensity

# 2. Run 5-fold OOF validation, fit 5-seed snap40 calibrator, verify 80k-row C++ parity & isolated ZIP
python -m experiments.build_ensemble_submission \
  --sgru-dir outputs/sgru_h160_10ep \
  --slstm-dir outputs/slstm_h160_10ep \
  --output-dir artifacts/ensemble/gpu_3model_gru160_lstm160_snap40 \
  --zip-output dist/submission_gpu_3model_gru160_lstm160_snap40.zip \
  --simplex-only --threads 6
```

---

## 6. Summary of Earlier Phases & Archived Artifacts

During the open-source cleanup, ~400 intermediate JSON/log receipts (`artifacts/round2/`, `artifacts/linux_arm64/`, `artifacts/training_pilot*/`, `artifacts/training_smoke/`), intermediate `.zip` archives (`dist/`), and verbatim competition portal pages (`docs/competition/`) were removed to keep the repository lightweight and free of machine-local paths. Key findings from those earlier stages are preserved below and in [`docs/EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md):

- **Round 1 — Affine Head Folding & ONNX I/O Binding:** Fitting a 6-parameter ridge-regularized affine head (`penalty = 1e-2`) + `[1.25, 1.25]` scales on 1,024 training sequences and folding them into `baseline.onnx` kept recurrent weights bit-identical while lifting Global WP on dev from `0.61033 → 0.67173` (`0.65095` public) and improving single-core ONNX callback throughput by `+9.5%` via persistent I/O binding.
- **Round 2 — Sequence-Correlation Fine-Tuning (`correlation1024`):** Fine-tuning the upper GRU layer on 1,024 sequences (`seed = 20260912`, `lr = 3e-5`) reached **`0.67209` dev Global WP** (`0.65151` public). Its weights are preserved at [`models/correlation1024_seed20260912/model.onnx`](../models/correlation1024_seed20260912/model.onnx) as the first diverse member of the final ensemble.
- **Multiscale Context & Low-LR Coverage Probes:** Cross-fitted fast/slow activity EMAs (`half_lives = 16, 1024`) on cached predictions added no signal over an outputs-only residual MLP control (`0.67392` OOF dev WP, `0.65288` public). Similarly, anchored upper-GRU runs at `lr = 1e-5` (`coverage4096` vs `repeat1024`) moved predictions by `<1.3%`, motivating unanchored full-sequence scratch training on price-sorted LOB features.
- **Experimental FP32 C++ GRU (`native/connectome_native.cpp`):** Built and sanitizer-tested for single-backbone ONNX parity studies; superseded in deployment by [`native/ensemble_engine.cpp`](../native/ensemble_engine.cpp).

