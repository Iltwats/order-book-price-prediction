# Experiment log

A dated record of every experiment run since the 2026-10-24 cleanup: the
question, what we ran, the numbers and the decision. Newest entries go at the
bottom. Each entry names the script that reproduces it.

Conventions, unless an entry says otherwise:

- **Dev** is the frozen 1,400-sequence development split in
  `artifacts/manifest.json`. The confirmation split stays closed.
- **Score** is Global WP on scored rows (`need_prediction AND is_scored`).
  Dev scores run about 0.02 above public scores, so compare differences.
- **Out of fold** means 5-fold cross-fitting by sequence (fold seed
  `20261024`): fit on 1,120 sequences, score the other 280, pool all five folds.
- **CI** is a paired sequence-cluster bootstrap 95% interval (2,000 resamples).
  It does not correct for choosing among arms.
- The starting model is `correlation1024_seed20260912` (public **0.65151**, dev
  **0.67209**). Experiments read its cached dev predictions from
  `artifacts/global_wp/rescore_v1/correlation1024_seed20260912.cache` and run
  on CPU; no GPU was used.

Scripts live in [`experiments/`](../experiments/); outputs go to
`artifacts/experiments/` (gitignored).

---

## 2026-10-24 — Do fast/slow summaries add signal? (CPU probe of plan hypothesis 2)

**Question.** [The context plan](MULTISCALE_CONTEXT_PLAN.md) proposes a residual
branch fed with fast and slow causal summaries. Before building runtime and
export support for it, do such summaries carry information the model's outputs
lack?

**Method.** `experiments/context_probe.py`. At every scored row (and a 5% sample
of unscored rows) we built causal features:

- six input-group activity series, log1p(mean |Δx|) for i0/i1 price, volume and
  trade columns, plus |p| and p;
- each smoothed with half-lives of 16 and 1,024 rows (the plan's timescales);
- the raw `a0`–`a7` columns.

Corrections of the model's two outputs were then fitted out of fold with and
without that context: |y|-weighted linear models and a residual 32-unit MLP.
The arms follow the plan's design (capacity control, fast, slow, both).

**Result.**

| Arm | Dev WP | vs its control |
| --- | ---: | ---: |
| Raw model | 0.67209 | — |
| Linear control (affine refit) | 0.67139 | — |
| Linear + fast | 0.67243 | +0.0010 |
| Linear + slow | 0.67121 | −0.0002 |
| Linear + both | 0.67199 | +0.0006 |
| Linear + both + `a*` | 0.67225 | +0.0009 (vs raw +0.0002, CI spans 0) |
| MLP control (outputs only) | **0.67339** | +0.0013 vs raw, CI [+0.0004, +0.0021] |
| MLP + fast | 0.67273 | −0.0007 |
| MLP + slow | 0.67179 | −0.0016 |
| MLP + fast + slow + `a*` | 0.67210 | −0.0013 |
| MLP + `a*` only (added 2026-10-25) | 0.67256 | −0.0008 |

The table shows the 2026-10-25 rerun of the repo script (`fit` stage on the saved
features). It reproduces the original scratch run exactly for the linear arms and
the MLP control. MLP context arms moved by at most 0.0003 because each arm now
reseeds its own initialization. The `a*`-only arm tests whether the
mask-predictive columns help the calibration; they do not.

**Conclusion.** The fast and slow summaries add nothing beyond what the model's
outputs already carry; every context arm is at or below its control. The only
gain came from the control itself: a nonlinear calibration of the two outputs.

**Limits.** The probe did not use the GRU hidden state and trained nothing end
to end, so it does not rule out a model that learns from these inputs. It does
remove the case for building the residual branch now.

---

## 2026-10-25 — Output calibration: is the gain real, and where does it come from?

**Question.** Is the +0.0013 from calibrating the two outputs stable, and what
does it learn?

**Method.** `experiments/output_calibration.py`, run on the full cached dev
predictions (27.9M required rows, 2.61M scored). Each arm maps the outputs p to
p + f(p), fitted out of fold with the pooled |y|-weighted MSE on clipped targets.
The run log and every arm's result are in
`artifacts/experiments/output_calibration/`.

### Step 1 — Stability and fitting domain

| Arm | Dev WP | vs raw [CI] | Unscored WP |
| --- | ---: | ---: | ---: |
| Raw model | 0.67209 | — | 0.47002 |
| Affine, fitted on scored rows | 0.67139 | −0.0007 [−0.0010, −0.0005] | 0.46908 |
| MLP 32 units, 3 epochs, scored rows, seeds 0/1/2 | 0.67343–0.67357 | +0.0013 to +0.0015, all CIs > 0 | 0.4656–0.4658 |
| Same MLP fitted on **all** required rows | 0.67117 | −0.0009 [−0.0015, −0.0003] | 0.47149 |
| Same MLP, each output depends only on its own input | 0.67244 | +0.0004 [+0.0000, +0.0006] | 0.46931 |

- The gain is seed-stable.
- **It exists only when the calibrator is fitted on scored rows.** Fitted on all
  rows, it improves unscored rows and hurts scored ones. Scored rows relate p to
  y differently from the rest, so this is a mask-specific calibration.
- Letting each output use the other output supplies most of the gain.

### Step 2 — Training length and capacity

| Arm (scored rows) | Dev WP | vs raw [CI] |
| --- | ---: | ---: |
| 64 units, 3 epochs | 0.67391 | +0.0018 [+0.0006, +0.0034] |
| 32 units, 10 epochs | 0.67429 | +0.0022 [+0.0007, +0.0045] |
| 64 units, 10 epochs, seeds 0/1/2 | 0.67436–0.67445 | +0.0023 to +0.0024 |
| 64 units, 30 epochs, seeds 0/1/2 | 0.67467–0.67470 | +0.0026 [+0.0010, +0.0052] |

Training length matters more than width. The network is still underfitting at
30 epochs: it has 322 parameters against 2.6M rows, so the limit is optimization,
not overfitting.

### Step 3 — Nonparametric check

Shrunk 2-D binned residual means (40 or 80 quantile bins per output) scored
0.67136–0.67185, below raw. Piecewise-constant bins are too coarse in the tails,
where the calibration acts, so binning gives no useful ceiling here.

### Step 4 — What the calibrator learns

`experiments/calibration_shape.py` (output in
`artifacts/experiments/output_calibration/shape.json`):

- **Scored rows are high-activity rows.** There, prediction std is 0.98–1.05
  against 0.63 on unscored rows, and |y| averages 0.70 against 0.58.
- **Tail compression.** On scored rows, the |y|-weighted slope of y0 on p0 falls
  from 0.87 (smallest half of |p0|) to 0.67 (top 1%). On unscored rows it falls
  only from 0.82 to 0.75. The calibrator maps p0 = 1.5 to about 1.08.
- **Cross-target term.** Each output pulls the other down (p0 = 1.5 alone gives
  p1′ ≈ −0.16), consistent with the negative t0/t1 correlation.
- **Agreeing signs shrink.** When both outputs share a sign, which is unusual
  given that negative correlation, both are pulled toward zero.

### Step 5 — Extrapolation and input clipping

The unclipped 64-unit calibrator behaved badly outside |p| > 2, the 0.6% of rows
beyond the training bulk. It flipped signs on about 7,900 dev rows; for example
(−4.2, −4.5) became (+2.9, −2.0). Hidden test data could hit that region more
often. The metric clips predictions at ±2 anyway, so we clipped the calibrator's
**input** to ±2. Raw outputs beyond ±2 already score as ±2, so the clip costs the
raw model nothing.

| Arm (64 units, 30 epochs, scored rows) | Dev WP | vs raw [CI] |
| --- | ---: | ---: |
| Unclipped | 0.67470 | +0.0026 [+0.0010, +0.0052] |
| Input clipped to ±2, seeds 0/1 | 0.67392–0.67397 | +0.0018 [+0.0010, +0.0030] |

- Unclipped minus clipped is +0.0008, CI [−0.0001, +0.0025].
- Leave-one-sequence-out shows that **one sequence supplies all of that +0.0008**.
- The clipped gain is broad: the top 20 sequences supply under half of it.

**Decision.** Ship the clipped calibrator: 64 units, 30 epochs, seed 0, inputs
clipped to ±2, fitted on all 1,400 dev sequences' scored rows. Its expected dev
gain is about +0.0018. On unscored rows it costs −0.0068 (0.47002 → 0.46327),
which matters only if the hidden test mask selects rows differently from
validation.

---

## 2026-10-25 — Export, verification and packaging

**Export.** `experiments/export_calibrated.py --hidden 64 --epochs 30 --seed 0 --clip 2
--output artifacts/calibrated/correlation1024_mlp64_clip2`.

- Fits the calibrator on all 2,612,518 dev scored rows.
- Appends it to both the deployment step model and the offline sequence model
  as standard ONNX nodes (`Clip`, `MatMul`, `Add`, `Erf`, `Mul`) after the
  original `prediction`. Opset 17 and IR 8 are unchanged, and so are
  `runtime.py`, `solution.py`, the config and packaging.
- Base model `15c3dad6…`; calibrated step model `9bab3466c2a6…`; sequence model
  `0eed778f0d02…`.

**Checks.**

| Check | Result |
| --- | --- |
| Calibrator graph vs Torch reference, all 27.9M dev rows | max abs error 2.1e-6 |
| Streaming step model (`connectome evaluate --split smoke`, 640,000 callbacks) vs cached raw outputs passed through the calibrator | max abs error 1.2e-6 |
| Output range over the whole [−2, 2]² input box | [−1.79, 1.99]; no outputs outside ±2 |
| Callback time, M2 Pro, `iobinding`, 4 sequences × 3 repetitions | 25.0 µs vs 22.2 µs (+13%) |
| Estimated judge runtime | about 25–26 min of 60, if the +13% carries over from the current 22–23 min |

**Archive.** `dist/submission_correlation1024_mlp64_clip2.zip`.

- Size: 724,099 bytes. SHA256: `b336b2327aadd55f9198782275e11da37827161edc1f7b8903f1de60b9c156a4`.
- Verified offline in a fresh process (resets, output ownership, no network).
- It differs from the archive that already ran on the judge
  (`submission_correlation1024.zip`, public 0.65151) only in `model.onnx` and
  `MODEL.md`. The runtime, `solution.py` and config are byte-identical.
- Checked against the [submission guide](competition/submission_guide.md) after
  extracting the archive. The test ran `solution.PredictionModel` from the archive
  in an isolated Python 3.11 process with networking disabled, on two full
  validation sequences (A, B, then A again; 60,000 callbacks).
  - `solution.py` is at the archive root and defines a no-argument
    `PredictionModel` with `predict(data_point)`.
  - Warm-up rows return `None`; every other row returns a finite float32 array
    of shape `(2,)`.
  - The state resets between sequences, and sequence A gave identical outputs on
    both runs.
  - Outputs match the reference to within 4.8e-7. Speed was 26.6 µs per
    callback including the Python loop.
  - Result: `artifacts/calibrated/correlation1024_mlp64_clip2/zip_check.json`.
- The Linux x86-64 container check was not rerun (Docker was not running). The runtime code is
  unchanged, and the new graph nodes are standard ONNX operators.

**Public result.** Pending upload. If the dev gain carries over, expect roughly
0.6533 (+0.0018). Record the score here when it arrives.

---

## Next (proposed, not started)

1. Upload `dist/submission_correlation1024_mlp64_clip2.zip` and record the public
   score above. A clear drop would mean the hidden mask differs from validation's,
   which also bears on any mask-aware training.
2. Run the v2.1 coverage comparison locally with `--device cpu`, about 75 minutes
   per 512-update arm, and score both arms on dev.
3. Any new base model needs its own calibrator: refit with
   `experiments/export_calibrated.py` on that model's dev predictions.
4. Because the scored-row relation differs from the rest, the next modeling lever
   is training that weights scored-like (high-activity) rows. The calibration
   only reshapes outputs after the fact.

---

## 2026-10-25 — Where does per-row time go?

**Question.** Can the callback get much faster, and what limits it?

**Method.** `experiments/latency_breakdown.py`, one validation sequence
(20,000 rows) on one thread, M2 Pro, calibrated model. Each figure is the best of
three passes. Output: `artifacts/experiments/latency/breakdown_m2.json`.

| Component | µs per row |
| --- | ---: |
| Complete callback (`StreamingPredictor` + ONNX step) | 25.4 |
| Python callback and input checks with a do-nothing backend | 2.0 |
| ONNX Runtime step calls alone | 22.5 |
| GRU arithmetic, the whole sequence in one ONNX call | 8.8 |
| ONNX Runtime per-call overhead (difference) | 13.7 |

**Findings.**

- More than half the time is ONNX Runtime's fixed cost per call, not
  arithmetic.
- The GRU reads all 192,258 FP32 weights (769 KB) every row. They do not fit in
  a core's L1 cache, so memory bandwidth sets the arithmetic floor at a few
  microseconds. The judge's EPYC 9224 has 1 MB of L2 per core, which holds them.
- Nanoseconds per row is not reachable with this model under a Python callback.
  The Python call alone costs about 2 µs, and the judge's own loop adds more:
  its 22–23 minute runs work out to about 34 µs per row in total.
- A SIMD native kernel that removes the ONNX Runtime per-call cost could plausibly
  reach about 6–10 µs per row. That is an estimate, not a measurement. The earlier
  C++ backend was slower than ONNX because it was built without SIMD flags.
- Speed does not change the score. It matters only if a larger model or an
  ensemble needs the runtime headroom; we currently use about 25 of 60 minutes.

---

## 2026-10-25 — Does our approach match what the data means?

**Question.** The [data overview](https://wundernn.io/connectome/docs/data_overview)
says `t0` and `t1` are undisclosed indicators of the future price movement of
`i0`; the 112 features are two L11 order books with trades plus `a0`–`a7`; and a
column index is an identifier, not a depth level. Do our working assumptions
(raw features into a GRU, |y|-weighted MSE, pooled calibration, mask-agnostic
training) fit that?

**Method.** `experiments/data_semantics.py`: dev targets with cached
`correlation1024` predictions; random dev samples of 12, 40 and 300 sequences
for the feature checks; the tick class of every TRAIN and validation sequence.
Output: `artifacts/experiments/data_semantics/{report.json,class_labels.npz,run.log}`.
Runtime about one minute.

### What the features are

- The price-like columns are not raw prices. Values are O(1) and stationary
  within a sequence. At step 0 the price block is nearly identical across
  sequences and even between `i0` and `i1` (e.g. `-1.434, 0.178, 0.417, -1.746, …`).
  The book goes through a fixed representation that encodes its shape rather
  than its price level.
- Targets are not a linear function of future changes in the 22 `i0` price
  columns: median R² is 0.01–0.03 at horizons from 1 to 2,000 rows.
  Mid-price, spread or queue imbalance cannot be computed by hand without first
  decoding the representation.
- Feeding these features into the GRU without normalisation is therefore
  reasonable; there is no scaling bug.

### What the targets are

Every sequence's targets are exact integer multiples of one of four spacings:
0.1254, 0.2413, 0.4685 or 0.6569. Below, a **tick class** is the set of
sequences sharing one spacing. On 2.6M required dev rows:

- `t0` and `t1` are **never both positive** (0 rows) and never both zero.
  Exactly one is zero on 43% of rows.
- The most common cells, in grid units, are (0, −1) 20%, (−1, 0) 20%,
  (−1, −1) 10%, (−3, +1) 5.7% and (+1, −3) 5.6%. Each target is positive on only
  20% of rows.
- This pattern looks like the outcomes of a long and a short trade measured in
  ticks, after paying the spread. The definition is not disclosed, so this is an
  interpretation.

### The four tick classes behave very differently

| Tick class | Dev seqs | Share of scored rows | Scored fraction | WP within class | Calibrator gain within class (out of fold) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1254 | 295 | 28.1% | 12.5% | 0.787 | +0.0019 |
| 0.2413 | 258 | 15.2% | 7.7% | 0.763 | +0.0030 |
| 0.4685 | 327 | 8.3% | 3.3% | 0.718 | +0.0010 |
| **0.6569** | **520** | **48.5%** | 12.2% | **0.407** | −0.0007 |

- **The large-tick class (0.6569) holds half the scored rows, and the model is
  weakest there.** Its predictions are under-dispersed (std 0.60 against 0.93
  for the targets) and biased about 0.19 too negative.
- **The shipped calibrator's gain comes from the three smaller-tick classes.**
  It slightly hurts inside the large-tick class. In the 0.1254 class the raw
  model predicts both outputs positive on 5.9% of scored rows; the calibrator
  halves that.
- **Much of the pooled score is class level.** A prediction that is constant
  within each class scores 0.330 pooled. An oracle that knows each sequence's
  weighted mean target scores 0.435.
- **Sensitivity.** Raising within-class WP by 0.02 moves the pooled score by
  +0.0063 (0.1254), +0.0031 (0.2413), +0.0014 (0.4685) or +0.0046 (0.6569).
  The large-tick class has the most room; the small-tick class has the most
  leverage per point.
- **The class is inferable without targets.** `a7` separates the large-tick
  class (median −0.3 to −0.4 against −2.3 to −2.8). A crude nearest-centroid on
  11 feature summaries identifies the class 65% of the time from the 99
  warm-up rows and 81% from the first 1,000 rows (majority class: 37%).

### TRAIN and validation have different class mixes

| Tick class | 0.1254 | 0.2413 | 0.4685 | 0.6569 |
| --- | ---: | ---: | ---: | ---: |
| TRAIN (10,607 seqs) | 12.3% | 11.9% | 32.9% | 43.0% |
| Validation (1,873) | 21.3% | 18.5% | 23.0% | 37.2% |
| Dev (1,400) | 21.1% | 18.4% | 23.4% | 37.1% |

Reweighting dev sequences to TRAIN's class mix lowers the same model from
**0.6721 to 0.6256** (calibrated: 0.6739 to 0.6272). Two consequences:

1. The score depends heavily on the test set's class mix, which is not
   published. A test mix between TRAIN's and validation's would explain the
   0.02 gap between dev and public scores. That is a hypothesis, not a
   measurement.
2. Training samples sequences uniformly from TRAIN, so it underweights the
   small-tick classes by about 1.7× relative to validation-like data. Those
   classes carry the most leverage per point.

### What the scoring mask follows

- Book and trade events around t do not predict `is_scored` (AUC 0.50–0.54
  for changes at t−1 … t+5). Neither does the future target size (0.55).
- It follows the slow features: `a3` AUC 0.70, `a2` 0.64, `a4` 0.61. Scored rows
  come in short bursts (median run 2 rows, 44% single rows), and the scored
  fraction varies by class from 3.3% to 12.5%.
- So the mask depends mostly on observable state, not on the future. The
  WP-optimal target function stays `E[|y|·y | x] / E[|y| | x]`, which is
  exactly what |y|-weighted MSE estimates. What the mask changes is where model capacity
  should go.

### Sign constraint as post-processing

Zeroing the smaller prediction whenever both are positive: raw +0.00008
[−0.00001, +0.00017]; calibrated −0.00004. Zeroing both: −0.0016 and −0.0007.
The constraint is real, but the calibrator already captures what
post-processing can get from it.

### Verdict on what we have been doing

| Practice | Verdict |
| --- | --- |
| MSE weighted by \|y\| on clipped targets, pooled across sequences | Correct: it estimates the WP-optimal prediction |
| Raw features into the GRU | Fine: features are already scaled and stationary |
| Scored-row output calibration | Correct, but the gain is in the small-tick classes, not where the model is weakest |
| Reporting only the pooled dev score | Blind spot: it hides the four classes and the class-mix risk |
| Training uniformly over TRAIN sequences and rows | Mismatched: class mix and mask both differ from the evaluated data |
| Context probe (generic activity EMAs) | Weak test: generic activity summaries and only the pooled score, not aimed at a class |
| Speed work | Irrelevant to the score, as recorded above |

**Decision.**
- Report WP by tick class in every future experiment.
- The next modeling experiment should change what the base model trains on:
  - weight TRAIN sequences to the validation class mix;
  - weight rows by a mask-propensity model fitted on validation (`a2`–`a6`);
  - consider validation sequences outside dev as extra training data.
- Post-processing has little left to give.
- Run a CPU pilot before any GPU run, so the GPU decision rests on measured
  evidence.
