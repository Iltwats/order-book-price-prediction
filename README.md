# Alpha Connectome — Solution & Step-by-Step Engineering Log

Solution codebase, pretrained checkpoints, and streaming C++17 SIMD inference engine for the **Wunder Fund Alpha Connectome** quantitative ML competition.

---

## 1. Problem Statement

The goal is causal, online prediction of two undisclosed future price-movement indicators (`t0`, `t1`) for trading instrument `i0` from an event-by-event market stream:

- **Inputs (`112` float32 features per row):** Two instruments (`i0`, which we predict, and `i1`, a correlated cross-instrument input), each providing 11 bid price-like columns (`p0..p10`), 11 ask price-like columns (`p11..p21`), 22 corresponding volume columns (`v0..v21`), 4 trade prices (`dp0..dp3`), and 4 trade volumes (`dv0..dv3`), plus 8 global context features (`a0..a7`).
- **Crucial data quirk — unordered depth columns:** Column indices `0..10` (bid) and `11..21` (ask) are *fixed component identifiers*, not sorted book depth levels (`p0` is not necessarily best bid).
- **Streaming causal interface:** Data is organized into independent **20,000-row sequences** (`10,607` train sequences = `212.1M` rows; `1,873` validation sequences = `37.5M` rows). Rows arrive one at a time via `PredictionModel.predict(data_point)`. Steps `0–98` are warm-up (`return None` after updating state), steps `99–19,999` require `[t0, t1]` predictions, and internal state must reset whenever `seq_ix` changes.
- **Hidden scoring mask & Global Weighted Pearson (WP):** Only ~9% of post-warm-up rows have `is_scored = True` (visible in validation, hidden at test time). Across all scored rows pooled globally across every sequence, targets and predictions are clipped to `[-2, 2]`, weighted by `|y|`, correlated separately for `t0` and `t1`, and averaged ([`METRIC.md`](METRIC.md)).
- **Execution limits:** Isolated Linux container (`python:3.11-slim-bookworm`), **1 vCPU** (AMD EPYC 9224), **16 GB RAM**, no GPU, no internet, 20 MB ZIP cap, and a **60-minute hard limit** for the entire `1,970`-sequence (`39.4M`-row) test set (~91 µs/row total budget including the judge's own Python harness).

---

## 2. Solution Overview

When each `DataPoint` arrives in `PredictionModel.predict`, the deployed pipeline executes four stages inside a single zero-allocation C++17 call ([`native/ensemble_engine.cpp`](native/ensemble_engine.cpp)):

1. **Dynamic LOB Quote Sorting & Ring-Buffer Order Flow (`169` causal channels + `34` snapshot features):**
   Sorts `i0` and `i1` bid/ask quotes ascending per row (stable insertion sort), gathers `v0..v21` in quote-rank order to align depth volumes with true price levels, computes best bid/ask, spread, weighted mid-price, book/trade imbalances, and maintains exact causal ring-buffer differences at lags `t−1`, `t−4`, and `t−16`.
2. **Diverse 3-Model Recurrent Backbone:**
   - **`correlation1024`** (`21.37%` weight): 2-layer `GRU128` operating on the raw 112 features, fine-tuned with sequence correlation loss from the organizer's pretrained weights.
   - **`ScratchSortedGRU-160`** (`43.31%` weight): 2-layer Residual `GRU160` (`413k` params) trained from random initialization on all `10,607` train sequences over the 169 sorted-LOB channels using exact batch-pooled Global WP + `|y|`-weighted MSE weighted by out-of-fold scoring-mask propensity $P(\text{is\_scored} \mid x)^{1.5}$.
   - **`ScratchSortedLSTM-160`** (`35.32%` weight): Orthogonal 2-layer Residual `LSTM160` (`550k` params, additive cell-state highway with `+1.0` forget-gate bias) trained from scratch under the same propensity-weighted Global WP objective with Polyak EMA (`0.995`).
3. **Magnitude-Preserving `snap40` Meta-Calibrator:**
   During warm-up (`!need_prediction`), recurrent states advance while readout heads are skipped. On required steps (`step >= 99`), a 5-seed `MLP128` residual head takes clipped individual/blended predictions, cross-target interactions (`q0 - q1`, `q0 + q1`, `|q|`), and 34 current sorted-LOB invariants (with `±8σ` standardization and **no `LayerNorm`**, preserving cross-sequence $|y|$ scale).
4. **FP16-Panel SIMD Execution (`AVX2+F16C+FMA` / `ARM NEON`):**
   Recurrent weight matrices are stored in cache-line-aligned IEEE FP16 panels and accumulated into FP32 registers via hardware FMA, cutting L2 cache bandwidth and archive size in half (`2.95 MB` ZIP, `51m27s` judge runtime for 3 models; `33m51s` for the 2-model variant) while matching PyTorch/ONNX reference WP within `1e-6`.

---

## 3. Step-by-Step Progression

Every change was tested on a frozen **1,400-sequence `dev` split** (`2.61M` scored rows; 5-fold sequence-grouped OOF with paired 2,000-resample cluster bootstrap CIs), holding `473` confirmation sequences apart. Full logs, negative results, and commands are in [`docs/EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md).

### Step 1 — Split Protocol & Output Scale Calibration (`0.61033 → 0.67173` dev | `0.65095` public)
Raw predictions from the starter `baseline/baseline.onnx` (`GRU128 × 2`) had standard deviation `2.4` against `0.9` for targets, causing 39% of predictions to hit the `±2` metric clip and distorting pooled Global WP. Fitting a 6-parameter ridge-regularized affine head (`penalty = 1e-2`) + `[1.25, 1.25]` scales on 1,024 train sequences and folding them into the final linear layer lifted dev Global WP by **`+0.0614`** with zero extra inference ops, while persistent ONNX I/O binding sped up callbacks by `+9.5%`.

### Step 2 — Sequence-Correlation Upper-GRU Fine-Tuning (`0.67209` dev | `0.65151` public)
Converting ONNX `[z, r, h]` gates into PyTorch `[r, z, n]` (`linear_before_reset=1`) exposed small FP32 kernel-rounding drift across 20,000-step recurrences. Using exact ONNX forward trajectories with local upper-GRU correlation gradients over 1,024 sequences produced `correlation1024_seed20260912` (`22m55s` on the judge). However, further perturbations around `baseline.onnx` (more seeds, MSE heads, low-LR `coverage4096` vs `repeat1024` runs, and fast/slow activity EMA probes) all plateaued within `0.671–0.672` dev WP.

### Step 3 — Data Semantics: Four Tick Regimes, Class-Mix Shift & the Scoring Mask
Inspecting target geometry and feature predictability ([`experiments/data_semantics.py`](experiments/data_semantics.py)) uncovered why models were stuck:
- **Exact tick grids:** Every sequence's targets are integer multiples of one of four spacings: `0.1254`, `0.2413`, `0.4685`, or `0.6569`. `t0` and `t1` are never both positive and never both zero (resembling long/short trade outcomes after paying the spread).
- **The large-tick bottleneck:** Class `0.6569` accounts for **48.5% of scored rows** yet scored only **`0.407` WP** (vs `0.72–0.79` in the three smaller-tick classes).
- **Train vs validation class-mix shift:** Small-tick classes (`0.1254` + `0.2413`) make up `24%` of `train.parquet` vs `40%` of `dev`, explaining most of the ~`0.02` level offset between dev and public leaderboard scores.
- **What `is_scored` actually selects:** Local book/trade jumps around step `t` barely predict `is_scored` (`AUC ≤ 0.54`), whereas slow context columns (`a3`, `a2`, `a4`) strongly predict it (`AUC 0.61–0.74`). Scored rows occur in high-activity regimes (`pred std ~ 1.0` vs `0.63` on unscored rows).

### Step 4 — Scored-Row Residual Calibration & ORT Overhead (`0.67392` dev | `0.65288` public)
`correlation1024` scored `0.672` WP on scored rows but only `0.470` on unscored rows. Fitting a clipped 64-unit residual MLP (`p -> p + f(clip(p, ±2))`) **strictly on scored rows** compressed extreme tails and used cross-target negative correlation to gain `+0.00183` OOF dev WP (`+0.00137` public). Fitting the exact same MLP on *all* rows hurt (`-0.0009`). However, appending just 11 small ONNX ops added **`4m17s`** of per-row ORT session dispatch overhead on the judge (`22m55s → 27m12s`), motivating a custom C++ runtime.

### Step 5 — Observable Mask-Propensity Scorer (`0.7985` OOF AUC)
To train backbones on `train.parquet` (where `is_scored` is absent) without wasting >90% of gradient energy on noisy unscored rows, we trained a 20-feature out-of-fold propensity MLP ([`experiments/mask_propensity.py`](experiments/mask_propensity.py)). Stratifying `correlation1024` on `train.parquet` by predicted propensity confirmed the diagnosis: `0.500` WP across all train rows vs **`0.689` WP** on the top 5% propensity rows.

### Step 6 — Decoding Anonymized LOB Permutations & `snap40` + C++17 Engine (`0.67688` dev | `0.65456` public)
Earlier snapshot heads overfit sequences (`0.638` WP) or collapsed `0.6569` (`0.407 → 0.387`). Three fixes unlocked `+0.00296` OOF dev WP ([`experiments/regime_calibrator.py`](experiments/regime_calibrator.py)):
1. **Row-wise quote sorting:** Sorting `p0..p10` and `p11..p21` per row decoded true best bid, best ask, spread, and rank-aligned depth volumes without sequence memorization.
2. **Removing `LayerNorm` and slow sequence identifiers:** `LayerNorm` divides out cross-sequence amplitude needed by $|y|$-weighted Global WP, while slow running means of `a0..a7` let MLPs memorize sequence IDs.
3. **Replacing ORT per-row calls with C++17 SIMD:** Compiling [`native/ensemble_engine.cpp`](native/ensemble_engine.cpp) cut judge runtime to **`20m52s`** — `6m20s` faster than the ONNX calibrator and `2m03s` faster than even raw `baseline.onnx`, while running quote sorting + 5-seed `MLP128` calibration on every row.

### Step 7 — Scratch 169-Channel GPU `SortedGRU-160` & `SortedLSTM-160` (`0.68062` dev | `0.65694` public)
Every prior submission still relied on the frozen first-layer weights of `baseline.onnx`. We trained completely independent recurrent architectures from random init over 169 causal sorted-LOB & multi-lag order-flow channels ([`experiments/train_scratch_gru.py`](experiments/train_scratch_gru.py), [`experiments/gpu_train_sorted_models.py`](experiments/gpu_train_sorted_models.py)):
- **CPU prototype (`ScratchSortedGRU-128`, 300 steps):** Reached `0.66076` standalone dev WP and lifted the 2-model public score from `0.65456 → 0.65533` (`+0.00077` public vs `+0.00068` dev — **1:1 out-of-sample transfer**).
- **20× GPU data-pipeline speedup (`18.0s → 0.87s/step` on RTX 4090):** Moving propensity scoring to CUDA, caching thread-local `ParquetFile(pre_buffer=False)` handles, and preallocating `[64, 20000, 112]` buffers trained 10-epoch `hidden=160` models in ~25 minutes each.
- **Scratch models surpass `correlation1024` standalone:** `ScratchSortedGRU-160` reached **`0.67499`** raw dev WP and `ScratchSortedLSTM-160` reached **`0.67437`** (both well above `correlation1024`'s `0.67209`).
- **FP16-panel AVX2/F16C C++17 kernel:** Packing recurrent matrices into stride-1 FP16 panels sped up the wider `hidden=160` 2-model ensemble by **`7m51s`** over the earlier FP32 `hidden=128` build (`41m42s → 33m51s`, **`0.65662` public**) and fit the full 3-model ensemble comfortably in **`51m27s` (`0.65694` public)**.

### Step 8 — Transfer & Ensemble Diversity Check
Post-submission analysis ([`experiments/transfer_check.py`](experiments/transfer_check.py)) showed that `correlation1024` (raw 112 features) correlates at `0.965` with either sorted-LOB scratch model, whereas `SortedGRU-160` and `SortedLSTM-160` correlate at `0.983` with each other — so retaining `correlation1024` at `21%` blend weight adds `+0.0017` raw WP over GRU+LSTM alone.

### Progression Summary

| Step / Model | Blend Weights | Dev Global WP | Public Global WP | 1-vCPU Judge Time / RAM |
| --- | --- | ---: | ---: | ---: |
| 0. Starter `baseline.onnx` (`GRU128 × 2`) | — | `0.61033` | — | — |
| 1. Affine-calibrated ONNX `GRU128` | `[1.0]` | `0.67173` | `0.65095` | ~`23m` |
| 2. `correlation1024` (sequence-corr upper-GRU fine-tune) | `[1.0]` | `0.67209` | `0.65151` | `22m55s` |
| 4. `correlation1024` + ONNX `mlp64_clip2` calibrator | `[1.0]` | `0.67392` | `0.65288` | `27m12s` / `625 MB` |
| 6. `correlation1024` + C++17 `snap40` sorted-LOB calibrator | `[1.0]` | `0.67688` | `0.65456` | `20m52s` / `615 MB` |
| 7a. CPU 2-model (`corr1024` + `ScratchSortedGRU-128`) | `[0.85, 0.15]` | `0.67756` | `0.65533` | `41m42s` / `618 MB` |
| **7b. GPU 2-model (`corr1024` + `ScratchSortedGRU-160`)** | `[0.2663, 0.7337]` | **`0.68023`** | **`0.65662`** | **`33m51s`** / `664 MB` |
| **7c. GPU 3-model (`corr1024` + `SGRU-160` + `SLSTM-160`)** | `[0.2137, 0.4331, 0.3532]` | **`0.68062`** | **`0.65694`** | **`51m27s`** / `656 MB` |

---

## 4. Quick Start

### Setup & Tests

Requires Python 3.10+ (tested on 3.11) and a C++17 compiler (`clang++` or `g++`; automatically falls back to pure NumPy if unavailable).

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m pytest -q
```

### Run Streaming Inference Out of the Box

Both the packaged FP16-panel ensemble weights and the standalone FP32 PyTorch/ONNX checkpoints are included under [`models/`](models/), with [`submission_config.json`](submission_config.json) selecting the 3-model GPU ensemble by default:

```python
import numpy as np
from solution import PredictionModel
from utils import DataPoint

model = PredictionModel()  # Compiles C++17 SIMD engine (~15 ms) and loads weights
rng = np.random.default_rng(0)

for step in range(105):
    dp = DataPoint(
        seq_ix=0,
        step_in_seq=step,
        need_prediction=(step >= 99),
        state=rng.normal(size=112).astype(np.float32),
    )
    pred = model.predict(dp)  # None on warm-up steps 0..98; float32 shape (2,) on steps >= 99
```

### Validate, Package, or Retrain (with `datasets/*.parquet`)

```sh
# Score solution.py end-to-end on datasets/valid.parquet
.venv/bin/python -c "from solution import PredictionModel; from utils import ScorerStepByStep; print(ScorerStepByStep('datasets/valid.parquet').score(PredictionModel()))"

# Build and verify an isolated, network-disabled submission ZIP
.venv/bin/python -m connectome package --config submission_config.json --output dist/submission.zip

# Train scratch SortedGRU-160 & SortedLSTM-160 on a CUDA GPU (~65 min on RTX 4090)
DATA_DIR=datasets RUNS_DIR=outputs WORKERS=12 bash cloud/run_gpu_suite.sh

# Rebuild OOF blend + 5-seed snap40 calibrator from the FP32 .pt checkpoints in models/
.venv/bin/python -m experiments.build_ensemble_submission --simplex-only --threads 6
```

---

## 5. Repository Layout

| Path | Description |
| --- | --- |
| [`solution.py`](solution.py), [`submission_config.json`](submission_config.json) | Streaming `PredictionModel` entrypoint and active model configuration |
| [`models/`](models/) | Packaged 3-model & 2-model `.npz` ensembles, FP32 PyTorch `.pt` checkpoints (`sgru_h160_10ep`, `slstm_h160_10ep`, `sgru_largetick_h128`), and `correlation1024` ONNX model |
| [`native/ensemble_engine.cpp`](native/ensemble_engine.cpp) | Zero-allocation C++17 SIMD (`AVX2+F16C+FMA` / `NEON`) streaming engine |
| [`connectome/`](connectome/) | Core library: `ensemble_runtime.py`, `runtime.py`, `global_metric.py`, `global_objectives.py`, `data.py`, `evaluation.py`, `calibration.py`, `training.py`, `gpu_training.py`, `packaging.py` |
| [`experiments/`](experiments/) | Scratch `SortedGRU` / `SortedLSTM` / `TCN` trainers, mask-propensity scorer, `snap40` calibrator, ensemble builder, data-semantics & transfer checks |
| [`cloud/`](cloud/) | CUDA Dockerfiles and `run_gpu_suite.sh` multi-arm GPU training launcher |
| [`baseline/`](baseline/) | Unmodified starter ONNX `GRU128` baseline for reference and parity tests |
| [`utils.py`](utils.py), [`METRIC.md`](METRIC.md) | `DataPoint` interface, step-by-step scorer, and Global Weighted Pearson specification |
| [`docs/`](docs/) | [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`EXPERIMENT_LOG.md`](docs/EXPERIMENT_LOG.md), [`GPU_TRAINING.md`](docs/GPU_TRAINING.md), [`training.md`](docs/training.md), [`linux_validation.md`](docs/linux_validation.md) |
| [`tests/`](tests/) | 190 unit, numerical parity, packaging isolation, and runtime regression tests |
| [`artifacts/`](artifacts/) | Frozen validation split manifest (`manifest.json`), sequence tick-class labels, and OOF mask-propensity weights |

---

## 6. Attribution & License

Released under the [MIT License](LICENSE) — Copyright (c) 2026 **Atul Sharma**.

If you use this code, model weights, feature engineering, or C++ SIMD runtime in your work, derivative projects, write-ups, or research, **please attribute Atul Sharma** and link back to this repository (see [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{sharma2026wunderlob,
  author = {Sharma, Atul},
  title = {Alpha Connectome: Sorted-LOB Recurrent Ensemble and C++17 SIMD Inference Engine},
  year = {2026},
  url = {https://github.com/Iltwats/wunder-lob},
  license = {MIT}
}
```


