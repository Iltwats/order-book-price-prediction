# Round-two readout experiments — 2026-09-13

The user-provided live leaderboard screenshot records **Atul at 0.6083, rank 26**, with one submission. It does not identify the uploaded ZIP hash. Public leaderboard WP and local development WP concern different evaluation data and must not be subtracted to claim improvement. The [screenshot receipt](../artifacts/round2/leaderboard_receipt.json) preserves this distinction.

## Implemented and measured

The selected supplied-weight GRU remains the starting point. This step implemented a frozen experiment registry, original-graph Torch-parameter export, independent equation/gradient checks, training-only weighted statistics, five regularized head fits, ONNX export, full-sequence callback checks, and paired score comparisons. All 192,000 recurrent parameters remain unchanged in every candidate. Corrections fold into the existing 258-parameter readout; the inference architecture and operation count stay the same.

Both selected and original models roundtrip into byte-identical ONNX files. FP64 equation and gradient checks pass. The earlier strict FP32 Torch/ORT mismatch remains failed and has not been relabelled; recurrent fine-tuning was therefore not started. See the [conversion report](ROUND2_CONVERSION.md) and [head method](ROUND2_HEAD_METHOD.md).

Statistics extraction streamed **1,024 training sequences / 20,480,000 rows** in **309.68 seconds**. There were 1,020 eligible training sequences. Reported peak process RSS was **1.57 GB** on the local M2 host. Covariance arrays occupy about 268 MB, with small additional vectors and metadata. The five fits then took approximately **0.76, 0.14, 4.86, 4.25 and 2.87 seconds**, respectively. The arrays, datasets and environments stay outside Git and the submission.

## Frozen 256-sequence screen

Every candidate uses the same training groups, selected parent, output-offset convention, scoring implementation and development screen. Intervals use 10,000 paired sequence-bootstrap resamples; they assume sufficiently independent sequences and do not correct for candidate-selection bias. These inspected development sequences are not a fresh holdout.

| Candidate | WP | Delta versus selected | 95% interval for delta | Winning sequences |
| --- | ---: | ---: | --- | ---: |
| Selected incumbent | 0.638916085 | — | — | — |
| Affine weighted-MSE control | 0.638702261 | −0.000213825 | [−0.000385966, −0.000045317] | 37.9% |
| Affine correlation | 0.638923648 | +0.000007563 | [−0.000170275, +0.000172128] | 51.2% |
| Hidden-state correlation, penalty 1e−4 | 0.636469442 | −0.002446644 | [−0.004889719, −0.000076080] | 42.2% |
| Hidden-state correlation, penalty 1e−3 | 0.636486413 | −0.002429672 | [−0.004842733, −0.000076629] | 42.2% |
| Hidden-state correlation, penalty 1e−2 | 0.637752649 | −0.001163437 | [−0.003157061, +0.000780612] | 46.1% |

The affine correlation fit was the sole positive screen result and advanced unchanged to 1,400-sequence development. Its tiny screening gain is not convincing evidence by itself. All three hidden-state corrections improved their training surrogate but lost official screening WP. Training overfit, different training/scoring row selection and prediction clipping are possible contributors; this experiment does not isolate their individual effects.

## Full development and release decision

The affine correlation finalist completed **1,400 sequences / 28,000,000 rows**, including 1,399 eligible blocks, in **273.31 seconds** using offline bulk evaluation.

| Model | Development WP | Target 0 | Target 1 |
| --- | ---: | ---: | ---: |
| Selected incumbent | 0.629255535 | 0.629335694 | 0.629175375 |
| Affine correlation | 0.629306913 | 0.629384603 | 0.629229224 |

The paired mean delta is **+0.000051378**, with a 95% bootstrap interval **[−0.000041751, +0.000141743]** and wins on 53.3% of eligible sequences. Both targets improve slightly on average, but the interval includes zero. This is weak development evidence, not an established public leaderboard gain.

A single deterministic sensitivity repeat was declared before its outcome: sample 512 of the original 1,024 training sequences without replacement using seed `20260913`, preserve their original order, and refit the identical affine objective and settings. It consumed one reserved robustness arm; screen and full development ran once after correctness passed, irrespective of the screen delta. The subset was not redrawn. This tests sensitivity to the fitting sample, not independent generalization; it overlaps the original training data and reuses the same inspected development set.

The repeat screened at **0.638916418**, delta **+0.000000332**, then reached development WP **0.629272806**, delta **+0.000017271** with interval **[−0.000003088, +0.000036346]**. Its 512-sequence extraction took 107.36 seconds and retained 509 eligible training sequences. Both full-development means are positive, but both intervals include zero. The observed gains remain small and uncertain. The 1,024-sequence fit is the highest-development-WP feasible candidate and has a separately validated **experimental ZIP**. The existing default model and release stay intact. No new public submission was made.

This completed six screens and two full-development evaluations, within the unchanged 12/4 caps. No confirmation evaluation or scale-grid expansion occurred. The remaining cycle retains two causal-feature arms, two conditional partial-GRU arms, one conditional clipped refinement and one robustness slot; only two full-development slots remain.

## Experimental package and CPU measurements

The new artifact is [`dist/submission_round2_affine_rho.zip`](../dist/submission_round2_affine_rho.zip), **722,478 bytes**. SHA-256: `26c00fed6e6ae0be0a42f67c84ecf554fe0d0afcffecb8755380588905b7fac6`. Its ONNX model SHA-256 is `4dd9a31de54098e46edc3d21e4262c94e57c0e0cff4c52397eb0a084424af82f`. It uses single-thread ONNX I/O binding, strict inputs and no affinity, native library or postprocessing filters. The current `dist/submission.zip` is not replaced.

Three unprofiled Linux repetitions used the same 32 complete sequences and balanced ordering against the actual original `baseline/solution.py` callback. Median throughput was **40,725 callbacks/s**, versus **36,858 callbacks/s** for the original baseline: **10.49% faster**. All three candidate runs exceeded the baseline runs. Median model-constructor startup was **2.18 ms**, measured separately; imports and whole-process startup are not included in that number. This retains the existing optimized-wrapper advantage; it is not evidence that changing head coefficients accelerated the GRU.

Sustained production callbacks streamed **256 complete sequences / 5,120,000 rows** in **137.75 seconds**, including Parquet reads, model startup, input checks and official scoring. Throughput was **37,168 rows/s**, with **370,515,968 bytes** peak process RSS. Linux screen WP was `0.6389236658`, only `1.79e-8` from the host's verified bulk score. The test collected no hardware cache counters. At this measured rate, 1,400 sequences project to about **12.6 minutes** and all 1,873 provided validation sequences to **16.8 minutes**; neither projection requires reopening confirmation.

The extracted ZIP passed a fresh offline Python 3.11 Linux process without Torch, including warm-up, repeated sequence resets, finite outputs and independent output ownership. The image used one CPU quota, 16 GB RAM and no networking; its host was shared and its architecture was ARM64. The hidden judge's architecture and test volume remain unknown, so these are development measurements rather than a judge-runtime guarantee. See the [machine-readable candidate receipt](../artifacts/round2/experimental_candidate.json) and [full results](../artifacts/round2/head_results.json).

## Verification and experiment history

The complete host suite passed **145 tests** with 12 existing Torch export warnings. The fresh Linux ARM64 inference environment passed **108 tests**, skipping four optional training modules because Torch is absent. Its container had one CPU quota, exactly 16,000,000,000 bytes of RAM, no additional swap and no network. Docker's image-tag lookup failed; the same installed image worked via its verified immutable ID, `sha256:be7a836b27b3eecd1c1e83b0675ff189c520678059526dbf77d90bbe20d4493f`. No image or dependency download was needed.

All six candidates, including the subset repeat, passed callback/reference and bulk prediction comparisons over the same **32 full sequences, 640,000 rows each**, with bitwise-equal outputs; wrapper hidden states were also bitwise equal. Tests cover warm-up, resets, causal behavior, deterministic results and output ownership. None of these checks establishes hidden-judge timing.

The first hidden-head fitting attempt failed before producing a model or reserving a screen because L-BFGS received a noncontiguous parameter layout. A single contiguous copy fixed it. The original source and registry are preserved, the three unscored arms received an explicit source amendment, and a regression test reproduces the former failure and verifies identical results for both layouts. A subsequent metadata repair preserves original declaration signatures while updating duplicate-detection signatures for the corrected source. Both repairs have dated, hashed receipts in the [registry](../artifacts/round2/registry.json); no candidate settings, completed results, trial caps or validation membership changed.

An independent review also noticed that the initial source list omitted `selection.py`, imported by the registry validator. A supplemental hash receipt records it, and it remained unchanged. The workflow's tolerance-based parity flag was supplemented by explicit inspection of all five `bitwise_equal` fields before the advancement decision.

The [preservation audit](../artifacts/round2/preserved_release_audit.json) checks the original model, selected model, root configuration, current ZIP and all six prior confirmation evidence files. Confirmation was not reevaluated. The current ZIP remains `dist/submission.zip`, SHA-256 `9d8862659bbe823532879708fa2765ac023d2cae8984501919e1293a022c8de4`, 722,475 bytes.

## Reproduction and next work

The commands used were `python -m connectome.head_cycle init`, `moments`, the five registered fit/screen operations, then `dev affine_rho`. `run-all` offers the uninterrupted five-arm sequence in a fresh round. The sensitivity check uses `python scripts/round2_repeat.py declare` followed by `run`. Existing outputs deliberately cannot be overwritten; preserve the completed registry and evidence. The initial layout failure and its explicit recovery are recorded above. All commands use the pinned `.venv` Python 3.11 environment. Linux verification uses `scripts/run_linux_checks.py` with phases `tests`, `solutions`, `stream` and `zip`, passing the immutable image ID above with `--image` and the candidate config/archive paths as appropriate.

The next model hypothesis in the [bounded plan](NEXT_ACTION_PLAN.md) is a small causal feature correction, beginning with an untrained branch's CPU overhead. It should add information beyond the existing readout, keep every original GRU input unchanged, and compare against the selected incumbent. Further correlation-head parameter searches are not supported by these screen results. Recurrent training still needs its explicit numerical acceptance procedure; external weights remain conditional on the earlier provenance and transfer checks.
