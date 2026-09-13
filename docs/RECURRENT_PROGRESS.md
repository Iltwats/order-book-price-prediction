# Recurrent model progress

The supplied GRU now has a tested path for updating its upper recurrent layer and prediction head. An initial 16-sequence training pilot completed successfully. **No recurrent candidate score improvement or achievement of the 0.75 live leaderboard target has been measured in this report.** Later results remain pending below.

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

Linear scaling gives **210.53 seconds, about 3.51 minutes**, for one 256-sequence configuration, and **421.07 seconds, about 7.02 minutes**, for two. These are planning estimates from a small pilot, not measured larger runs. They exclude subsequent candidate numerical audits, validation, packaging and deployment testing. They also do not measure inference speed or establish the hidden judge's one-CPU runtime; its hardware and workload remain unknown.

Training uses all post-warmup training rows and never uses a validation scoring mask. Development selection must compare actual clipped WP against the selected incumbent, with the existing sequence eligibility rules. Previously consumed confirmation results cannot become independent again.

## Later results: pending

Registered candidate training, candidate-specific numerical checks, development WP, repeat evidence, Linux callback timing and any new live leaderboard result are **pending in this report**. Append measured results with their artifact paths when available; retain the incumbent until the applicable selection and release checks pass.
