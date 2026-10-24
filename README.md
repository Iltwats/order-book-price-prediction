# Alpha Connectome — leaderboard push

**Goal:** finish in the top 8 (paid places) of the Wunder Fund Alpha Connectome
private leaderboard. Final submissions close **2026-11-15 23:59 UTC**; team
creation/merging closes **2026-11-01 23:59 UTC**. Limit: 5 submissions per day.

## Where we stand

Snapshot from 2026-10-24. Public scores are Global WP on a sample of the hidden
test set; the private leaderboard uses a separate dataset.

| | Global WP |
| --- | ---: |
| Our best public: `submission_correlation1024.zip` (rank 635) | 0.65151 |
| Public rank 300 / 100 / 50 | 0.6572 / 0.6628 / 0.6653 |
| Public rank ~8 (last paid place) / rank 1 | ~0.670 / 0.6739 |
| Same model, our 1,400-sequence development split | 0.67209 |

Development scores run about 0.02 above public scores, so compare
improvements, not levels. Each +0.001 currently moves dozens of places. Full
rescore of existing models: [docs/GLOBAL_WP_RESULTS.md](docs/GLOBAL_WP_RESULTS.md).

## The task

Predict `t0` and `t1`, two future-movement indicators of instrument `i0`, from
112 features (two L11 order books with trades, plus 8 extra columns). Rows arrive
one at a time through `PredictionModel.predict`. Each sequence has 20,000 rows,
the first 99 are warm-up, and state resets between sequences.

**Metric — Global Weighted Pearson:** pool every scored row across all
sequences, clip targets and predictions to ±2, weight by |clipped target|,
compute one correlation per target, and average the two. About 9% of required
rows are scored; the test mask is hidden. See [METRIC.md](METRIC.md).

**Runtime:** 1 vCPU (x86-64 AMD EPYC 9224), 16 GB RAM, 60 minutes for the
1,970-sequence (39.4M-row) test set, 20 MB ZIP, no network. Our current
submissions take 22–23 minutes. Official pages: [docs/competition/](docs/competition/).

## What the data tells us

Measured on validation with the existing models. These shape the next experiments.

- **Scored rows are different.** The calibrated model scores 0.67 WP on scored
  rows but 0.47 on unscored rows. The mask is partly predictable from features
  (`a3` alone: AUC 0.735; `a2`: 0.685). Training treats all rows equally.
- **Magnitude across sequences matters.** Rescaling each sequence's predictions
  with hindsight would lift development WP from 0.672 to 0.694. The old
  per-sequence metric ignored this; Global WP does not.
- **Calibration was the big win so far.** Raw baseline outputs are far too large
  (standard deviation 2.4 against 0.9 for the targets; 39% get clipped). An affine
  head moved development WP from 0.610 to 0.672.
- **Fine-tuning the supplied GRU has plateaued.** Its calibrated and fine-tuned
  variants fall within 0.6712–0.6721 on development and 0.6510–0.6515 publicly.
- **Target structure.** Targets come in steps of about 0.657 (tick counts).
  Within sampled sequences, `t0` and `t1` are strongly negatively correlated
  (−0.49 to −0.93). In one sampled sequence, autocorrelation was about 0.96 at
  lag 1 and 0.46 at lag 100.
- **Unused runtime.** Submissions use 22–23 of 60 minutes, leaving roughly 2.5×
  compute for larger models or ensembles.
- **Scorer verified.** Our Global WP implementation reproduces the official
  baseline score of 0.617052 on the full validation set.

## Next experiments

| Experiment | Status |
| --- | --- |
| Fine-tune the upper GRU of correlation-1024 with the batch-pooled Global WP loss ([plan v2.1](docs/GPU_TRAINING.md)) | Ready; needs CUDA qualification on a rented GPU |
| Train new GRUs from scratch on all 10,607 train sequences with pooled \|y\|-weighted MSE on clipped targets (the loss whose minimizer maximizes Global WP) | Proposed |
| Mask-aware training: weight rows toward scored-like rows, or fine-tune on validation with the mask, holding out sequences and checking unscored-row WP | Proposed |
| Ensemble seeds/variants within the runtime budget; refit the final models on train + validation | Proposed |

The FAQ warns against overfitting the visible validation mask, and the private
test uses different data, so judge changes on held-out sequences, not only on
public scores.

## Repository map

| Path | Purpose |
| --- | --- |
| `solution.py`, `submission_config.json` | Submission entry point; the config selects the model. It currently selects `artifacts/selected/model.onnx` (calibrated affine model, public 0.65095). The best public archive was built from `artifacts/round2/correlation_gru1024_seed20260912/config.json`. |
| `utils.py`, `METRIC.md` | Official callback types and scoring reference; Global WP scorer (the legacy per-sequence scorer is kept for old reports) |
| `connectome/runtime.py`, `packaging.py` | Streaming ONNX predictor used at submission time; builds and verifies the ZIP offline |
| `connectome/data.py`, `evaluation.py`, `calibration.py`, `selection.py` | Parquet streaming and frozen splits, scoring and prediction caches, calibration search, versioned report checks |
| `connectome/global_metric.py`, `global_objectives.py` | Pooled Global WP with sequence-cluster bootstrap; batch-pooled training loss |
| `connectome/gpu_training.py` (+ `partial_gru.py`, `conversion_audit.py`, `recurrent_parity.py`) | Frozen-plan GPU runner for upper-GRU training with ONNX parity checks |
| `connectome/training.py`, `model_io.py` | Torch GRU import/export, head and affine fitting, ONNX inspection |
| `scripts/` | `rescore_global.py`, Linux single-CPU checks, callback benchmarks |
| `cloud/Dockerfile`, `scripts/Dockerfile` | GPU training image; offline inference-check image |
| `configs/` | Postprocessing configurations used by the rescore |
| `baseline/` | Official baseline, unchanged |
| `native/`, `connectome/native.py` | Experimental C++ GRU backend; failed its speed and accuracy gates and is not shipped |
| `docs/` | [Results](docs/GLOBAL_WP_RESULTS.md), [GPU training](docs/GPU_TRAINING.md), [training tools](docs/training.md), [Linux checks](docs/linux_validation.md), [competition pages](docs/competition/) |
| `artifacts/`, `dist/`, `datasets/` | Local outputs and data (gitignored; small evidence receipts are force-added) |

`artifacts/manifest.json` freezes the validation split: 1,400 development
sequences (with 32-sequence smoke and 256-sequence screen subsets) and 473
confirmation sequences that are already consumed.

## Common commands

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m pytest tests native/test_model_io.py -q

# Global WP of a model on the 256-sequence screen or the 1,400-sequence dev split
.venv/bin/python -m connectome evaluate --split screen --model path/to/model.onnx --output report.json
.venv/bin/python -m connectome evaluate --split dev --model path/to/model.onnx --output report.json

# Official-style end-to-end scoring of the baseline callback on full validation
.venv/bin/python baseline/solution.py --validation datasets/valid.parquet

# Build and verify a submission ZIP (does not upload)
.venv/bin/python -m connectome package --config submission_config.json --output dist/submission.zip
```

## History

Round-two experiments, research notes, the legacy per-sequence metric and old
plans were removed in the repository cleanup. They remain at the git tag
`pre-cleanup`, for example `git show pre-cleanup:docs/ROUND2_RESULTS.md`.
