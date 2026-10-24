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

Measured on validation with the existing models, with an additional bounded
TRAIN target check. These findings shape the next experiments.

- **Scored rows are different.** The calibrated model scores 0.67 WP on scored
  rows but 0.47 on unscored rows. The mask is partly predictable from features
  (`a3` alone: AUC 0.735; `a2`: 0.685). Scored rows are high-activity rows:
  prediction std is about 1.0 there against 0.63 elsewhere. Training treats all
  rows equally.
- **A scored-row calibration of the outputs helps.** A small network on the
  model's two outputs, fitted only on scored rows, gains +0.0018 out of fold
  (CI [+0.0010, +0.0030]). It compresses large predictions and uses each output
  to correct the other. Fitted on all rows, it hurts. See the
  [experiment log](docs/EXPERIMENT_LOG.md).
- **Magnitude across sequences matters.** Rescaling each sequence's predictions
  with hindsight would lift development WP from 0.672 to 0.694. The old
  per-sequence metric ignored this; Global WP does not.
- **Calibration was the big win so far.** Raw baseline outputs are far too large
  (standard deviation 2.4 against 0.9 for the targets; 39% get clipped). An affine
  head moved development WP from 0.610 to 0.672.
- **Fine-tuning the supplied GRU has plateaued.** Its calibrated and fine-tuned
  variants fall within 0.6712–0.6721 on development and 0.6510–0.6515 publicly.
- **Four tick classes.** Every sequence's targets are integer multiples of one
  of four spacings: 0.1254, 0.2413, 0.4685 or 0.6569. `t0` and `t1` are never
  both positive and never both zero; the common cells are (0, −1), (−1, 0) and
  (−1, −1) grid units. They look like long/short trade outcomes in ticks.
- **The largest class is where we lose.** The 0.6569 class has half the scored
  rows and within-class WP 0.41. The other classes score 0.72–0.79. A
  prediction that only knows the class level already scores 0.33 pooled.
- **TRAIN and validation differ in class mix.** The small-tick classes
  (0.1254 and 0.2413) make up 24% of TRAIN but 40% of validation. Reweighting dev
  to TRAIN's mix drops the same model from 0.672 to 0.626, so the score is
  sensitive to the unknown test mix.
- **The mask follows slow state, not events.** Book and trade changes around a
  row do not predict `is_scored` (AUC ≤ 0.54). `a3`, `a2` and `a4` do (AUC
  0.61–0.70). Scored rows come in short bursts. Details:
  [experiment log](docs/EXPERIMENT_LOG.md).
- **Features are a representation, not raw prices.** The price-like columns
  are stationary and O(1), and they encode the book's shape rather than its
  level. Linearly, future changes in them explain a median 3% or less of target
  variance.
- **Unused runtime.** Submissions use 22–23 of 60 minutes, leaving roughly 2.5×
  compute for larger models or ensembles.
- **Scorer verified.** Our Global WP implementation reproduces the official
  baseline score of 0.617052 on the full validation set.

## Next experiments

The [discussion and model improvement plan](docs/MULTISCALE_CONTEXT_PLAN.md)
records how we arrived at the next steps: complete the prepared coverage
comparison, test whether a longer backward window improves the current GRU,
then assess an explicit combination of recent changes and slower context.
The context experiments require separate plans. A CPU probe found no added
signal from fast/slow summaries on top of the current outputs.

Every experiment run, with commands, numbers and decisions, is recorded in
[docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

| Experiment | Status |
| --- | --- |
| Fine-tune the upper GRU of correlation-1024 with the batch-pooled Global WP loss ([plan v2.1](docs/GPU_TRAINING.md)) | Ready; needs CUDA qualification on a rented GPU |
| Compare 256-row and 1,024-row backward windows with matched training examples and updates | Proposed after the prepared comparison; preserves the inference architecture |
| Calibrate the two outputs on scored rows (64-unit network, inputs clipped to ±2), appended to the ONNX graph | Dev +0.0018 out of fold; packaged as `dist/submission_correlation1024_mlp64_clip2.zip`; awaiting public score |
| Add a small temporal correction and compare capacity control, fast-only, slow-only and combined summaries | Deprioritized: on cached outputs, fast/slow summaries added nothing over an outputs-only control ([log](docs/EXPERIMENT_LOG.md)) |
| Train new GRUs from scratch on all 10,607 train sequences with pooled \|y\|-weighted MSE on clipped targets (the loss whose minimizer maximizes Global WP) | Proposed |
| Check our assumptions against the official data overview: tick classes, class mix, mask, feature representation | Done ([log](docs/EXPERIMENT_LOG.md)): loss and calibration are sound; training ignores the class mix and the mask |
| Class- and mask-aware training: weight TRAIN sequences to validation's class mix and rows by a mask-propensity model (`a2`–`a6`); report WP per tick class | Next; CPU pilot first |
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
| `experiments/` | CPU experiments on cached development predictions: context probe, output calibration, calibrated ONNX export, latency breakdown, data-semantics checks ([log](docs/EXPERIMENT_LOG.md)) |
| `cloud/Dockerfile`, `scripts/Dockerfile` | GPU training image; offline inference-check image |
| `configs/` | Postprocessing configurations used by the rescore |
| `baseline/` | Official baseline, unchanged |
| `native/`, `connectome/native.py` | Experimental C++ GRU backend; failed its speed and accuracy gates and is not shipped |
| `docs/` | [Experiment log](docs/EXPERIMENT_LOG.md), [results](docs/GLOBAL_WP_RESULTS.md), [GPU training](docs/GPU_TRAINING.md), [training tools](docs/training.md), [Linux checks](docs/linux_validation.md), [competition pages](docs/competition/) |
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
