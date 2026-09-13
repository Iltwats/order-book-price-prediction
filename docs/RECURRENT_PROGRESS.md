# Recurrent model progress

Direct correlation training has produced a measured local improvement while reusing the supplied GRU. The stronger 256-sequence candidate reached **0.629469402 full-development WP**, versus **0.629255535** for the calibrated incumbent, a gain of **0.000213867**. An interim submission ZIP is available below. **The 0.75 live leaderboard target remains unmet; the two 1,024-sequence expansion runs and their results are pending.**

## What changed

Training starts from the selected calibrated checkpoint, which preserves the supplied two-layer GRU-128 recurrence. The lower layer remains frozen; the upper layer and head can learn. The selected original ONNX model also remains the unchanged teacher. No external weights or data enter this work.

The original strict Torch/ORT FP32 check remains failed. A separate conventional carried-state bridge also [failed on its first training sequence](../artifacts/round2/recurrent_bridge/report.json). The [same-sequence investigation](../artifacts/round2/recurrent_roundoff_diagnostic/report.json) found accumulated rounding in highly persistent hidden coordinates. Neither failure was relabeled and no numerical limits were widened.

The new procedure uses the **current student's original ONNX graph for the complete forward trajectory**. Torch computes approximate local gradients over 256-row chunks, each starting from the corresponding detached ONNX state. State and prediction corrections preserve ONNX forward values while passing local Torch derivatives backward. The [prospective protocol](../artifacts/round2/ort_training_bridge/protocol.json) checks uncorrected local errors as well as corrected values. This validates a bounded gradient approximation; it does not make Torch's derivatives exact derivatives of ORT.

## Completed evidence

The [new bridge report](../artifacts/round2/ort_training_bridge/report.json) covers eight fixed TRAIN sequences, 160,000 rows, with stop stages at two and eight sequences. All passed. Full-width FP64 equation checks and all 96 gradient-tensor comparisons passed, using fixed chunks beginning at rows 0 and 19,744 and gradient tolerances `atol=1e-6, rtol=1e-4`.

| Check | Largest observed error |
|---|---:|
| Local upper state, absolute | `1.1146e-5` |
| Local upper state, worst-coordinate RMS | `1.2832e-6` |
| Local prediction, absolute | `1.9073e-6` |
| FP32 versus FP64 gradient, absolute | `2.3960e-8` |

Corrected forward values matched ONNX throughout. Mean uncorrected local prediction WP drift was approximately `[-7.76e-9, -9.67e-9]` across the two targets. These are numerical comparisons on training data, not model improvement scores. Conventional strict statuses remain separate and failed.

The [pilot report](../artifacts/round2/partial_gru_pilot16/report.json) records one epoch over 16 fixed training sequences: two updates, batch size eight, chunk size 256, learning rate `1e-5`. It uses clipped-label weighted MSE, teacher weight `0.1`, relative parameter anchoring `0.001`, weight decay `0.0001`, and gradient clipping `1.0`. All 16 local checks passed before updates. Export preserved the lower recurrent weights bitwise. Checkpoints and the original-graph candidate export were produced; no validation score was taken for this timing pilot.

## Timing and limits

The pilot measured **13.158 seconds**, with one Torch CPU thread on macOS ARM64. The [timing receipt](../artifacts/round2/partial_gru_pilot16/timing_projection.json) reports process peak RSS of 2,170,109,952 bytes, approximately 2.02 GiB.

At the pilot stage, linear scaling estimated **210.53 seconds, about 3.51 minutes**, for one 256-sequence configuration, and **421.07 seconds, about 7.02 minutes**, for two. The completed training measurements are now recorded in the tables below. The earlier estimates excluded subsequent candidate numerical audits, validation, packaging and deployment testing. Neither training timing nor these estimates measure inference speed or establish the hidden judge's one-CPU runtime; its hardware and workload remain unknown.

Training uses all post-warmup training rows and never uses a validation scoring mask. Development selection must compare actual clipped WP against the selected incumbent, with the existing sequence eligibility rules. Previously consumed confirmation results cannot become independent again.

## Completed MSE trials

The [frozen results receipt](../artifacts/round2/recurrent_mse_results.json) records both completed one-epoch fits, candidate-specific numerical checks and byte-exact reference/callback/bulk checks on 32 full sequences. The source GRU's lower layer is unchanged. The same 256-sequence screen contains 256 eligible sequences; its incumbent WP is **0.638916085305**.

| Learning rate | Training seconds | Screen t0 | Screen t1 | Screen WP | Paired change | 95% sequence-bootstrap interval |
|---|---:|---:|---:|---:|---:|---|
| `1e-5` | 189.700 | 0.638362164 | 0.639261715 | 0.638811940 | -0.000104145 | [-0.000157997, -0.000049764] |
| `3e-5` | 190.325 | 0.638160133 | 0.639036530 | 0.638598331 | -0.000317754 | [-0.000474642, -0.000159095] |

Intervals use 10,000 paired sequence resamples, seed 20260913. They assume sufficiently independent sequences and do not correct for search bias. These negative results concern MSE training on this fixed subset and budget, not every possible recurrent model. Do not spend full-development slots or package either candidate. The original submission and the earlier affine experimental ZIP stay unchanged.

The host suite passed **210 tests**. The offline Linux one-CPU/16-GB suite passed **140 tests**, with eight Torch-dependent training modules intentionally skipped in the inference image. The separate [causal dynamics prototype](CAUSAL_BRANCH_RESULTS.md) failed the speed gate: nonzero throughput was 17.91% below the original baseline. It was not fitted or scored.

## Completed direct-correlation trials

The [completed screen receipt](../artifacts/round2/correlation_screen_results.json) records two prospectively declared one-epoch fits on the same first 256 training-anchor sequences, with seed 20260912. Both use batch eight, 256-row chunks, teacher strength 0.1, relative anchor 0.001 and the original inference graph. The lower pretrained GRU layer remains bitwise unchanged; learning updates the supplied model's upper layer and head. No external pretrained model or data was substituted.

The new objective computes clipped, sequence-weighted Pearson loss and its prediction gradient over each complete training sequence, then accumulates chunk gradients without changing parameters between chunks. Training uses all post-warmup rows, retains degenerate sides in the mean, and applies the declared clipping derivative. This remains a training surrogate: development uses the actual supplied mask and eligibility rules. A [separate objective-specific bridge](../artifacts/round2/correlation_training_bridge/report.json) records its numerical and gradient checks; earlier failed gates remain unchanged.

| Learning rate | Training seconds | Screen t0 | Screen t1 | Screen WP | Paired change | 95% sequence-bootstrap interval |
|---|---:|---:|---:|---:|---:|---|
| `1e-5` | 192.803 | 0.638508810 | 0.639451859 | 0.638980335 | +0.000064249 | [+0.000042167, +0.000086939] |
| `3e-5` | 193.374 | 0.638601365 | 0.639597554 | 0.639099460 | +0.000183374 | [+0.000119643, +0.000249044] |

Both comparisons use the same 256-sequence screen incumbent, **0.638916085305**. The stronger `3e-5` candidate advanced to [full development](../artifacts/round2/correlation_gru_lr3e-5/dev_report.json): 1,400 sequences processed, 1,399 eligible. Its target scores are **0.629507554** and **0.629431250**, averaging **0.629469402**.

The [paired development comparison](../artifacts/round2/correlation_gru_lr3e-5/dev_comparison.json) gives **+0.000213867**, with a 95% interval of **[+0.000182717, +0.000244814]**, against the calibrated incumbent's **0.629255535**. The candidate improved 71.62% of eligible sequences. These intervals use 10,000 paired sequence resamples, assume sufficiently independent sequences, and do not correct for adaptive search. The screen and full-development scores concern different subsets; neither is a live leaderboard score or new independent confirmation result.

## Interim submission

[Download submission_correlation256.zip](../dist/submission_correlation256.zip) contains the measured `correlation_gru_lr3e-5` model and is **722,519 bytes** compressed. The [package receipt](../artifacts/round2/correlation256_package_report.json) records successful Python 3.11 verification with networking disabled: 768 rows, two resets and independent output ownership. Its model hash matches the full-development candidate. The receipt's “full development pending” status was written before the completed development comparison above; the frozen receipt is preserved.

This is an interim experimental archive. No new live leaderboard result is established, and hidden-judge runtime remains unverified. The original submission stays available.

## 1,024-sequence expansion: pending

The [prospective expansion policy](../artifacts/round2/correlation_expansion_policy.json) fixes the first 1,024 training-anchor sequences, one epoch and learning rate `3e-5`, with seeds **20260912** and **20260913**. Each run starts independently from the selected pretrained source, rather than continuing the screened checkpoint. The unchanged selected model remains the teacher. These runs test broader training and seed sensitivity; they are not independent data holdouts.

Both training jobs are in progress; their completed training reports, numerical checks, screen results and any final development comparison are **pending**. The declared caps remain **12 screens / four full-development trials**. The completed correlation-256 stage brings usage to ten screens and three full-development trials; the policy allocates two final screens and at most one final development evaluation to the best feasible expansion. No expansion gain, revised archive or achievement of the 0.75 live target is claimed here. Confirmation remains consumed and has not been reused for these comparisons.
