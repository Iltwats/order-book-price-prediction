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

| Archive | SHA256 | Size | Dev OOF WP | `0.1254` | `0.2413` | `0.4685` | `0.6569` | Delta vs `0.65288` Shipped [95% CI] | Est. Judge Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| `dist/submission_ensemble_sorted_gru_snap40.zip` | `c59e453dacec...` | 1.84 MB | **`0.67756`** | `0.7955` | `0.7696` | `0.7212` | `0.4099` | **`+0.00365`** `[+0.00238, +0.00489]` | ~35.2 min |
| `dist/submission_corr1024_snap40_fast.zip` | `40c9a9470569...` | 0.83 MB | **`0.67688`** | `0.7935` | `0.7683` | `0.7202` | **`0.4107`** | **`+0.00296`** `[+0.00181, +0.00407]` | ~19.1 min |

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

### Phase 1: Interpret the Live Judge Result (Today, Nov 12)
1. **Verify runtime scaling on AMD EPYC 9224:**
   - Compare actual judge minutes against our M2 Pro projection (~35 min for the 2-model ensemble, ~19 min for single-backbone).
   - Because AMD EPYC 9224 has **AVX2/AVX-512 (256/512-bit FMA)** vs. M2 Pro's 128-bit NEON, `ensemble_engine.cpp` (`-march=native`) may run even faster on the judge (~24–30 min), confirming headroom for a **3- or 4-model ensemble** inside the 60-minute limit.
2. **Verify public WP transfer:**
   - Earlier `mlp64_clip2` transferred 75% of its dev OOF delta (`+0.00183` dev → `+0.00137` public, reaching `0.65288`).
   - A commensurate transfer on `+0.00365` dev OOF puts public WP around **`0.6556–0.6565`**, moving up ~200+ ranks immediately and validating sorted-LOB features + scratch ensembling on hidden data.

### Phase 2: Scale Scratch Training on a Cloud GPU via `cloud/Dockerfile` (+0.005 to +0.012 WP Potential)
Our local CPU `ScratchSortedGRU_v3` reached **`0.66076` dev WP** in just **300 gradient steps** (~1.8 passes over `train.parquet`, `hidden=128`)—and was still climbing (`0.623 → 0.641 → 0.652 → 0.658 → 0.663` screen WP). On a single rented GPU (RTX 4090 / A100 / L40S, ~$0.40–$1.20/hr for 3–4 hours), PyTorch `cuDNN` GRU runs **25–40× faster** than M2 Pro CPU:

1. **Train `ScratchSortedGRU` to full convergence (3,000+ steps, 8–10 epochs over all 10,607 sequences):**
   - Increase capacity to `hidden=192` or `hidden=256` with full-sequence TBPTT (`chunk=2000–4000`), cosine decay, weight EMA (Polyak/SWA averaging over final checkpoints), and multi-lag order-flow features (`t−1, t−4, t−16, t−64, t−256` causal EMAs).
   - Based on the local learning curve (`+0.040` WP from step 40 to step 300), full 10-epoch GPU convergence with `hidden=192` should push the scratch model from `0.6608` past **`0.674–0.678` standalone**, turning it from a 15%-weight helper into the primary `>50%`-weight backbone.
2. **Train a 3rd Orthogonal Model Family (`ScratchSortedLSTM` or `Conv-GRU` Hybrid):**
   - An LSTM cell `(h_t, c_t)` with an additive cell-state highway + local `kernel=5` causal depthwise Conv1d stem over the 169 sorted-LOB channels captures long-memory drift + exact short-tick bursts differently from a pure GRU.
   - Add `StreamingLSTM` support directly to `native/ensemble_engine.cpp` (~35 lines of C++; ~25 µs/row).

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
