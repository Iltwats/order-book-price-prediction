# Working target: at least 0.75 live WP

The target remains **0.75 or higher on the live leaderboard**. The [reported prior public score](../artifacts/round2/leaderboard_receipt.json) is **0.6083**, leaving a **0.1417** gap. The exact uploaded archive behind that prior score was not independently established. A local development score cannot close this live-score gap.

The strongest completed recurrent candidate, trained on 1,024 sequences, has full-development WP **0.6300190941**, versus **0.6292555348** for the calibrated incumbent: **+0.0007635593**. The [paired comparison](../artifacts/round2/correlation_gru1024_seed20260912/dev_comparison.json) gives a 95% interval **[+0.0006552369, +0.0008683899]** across 1,399 eligible sequences. The intermediate 256-sequence candidate scored **0.6294694019**. These are local results. **The 0.75 live target is unachieved.**

## Completed work

We continue to use the supplied two-layer GRU-128, initialized from its selected calibrated checkpoint. The lower recurrent layer stays bitwise unchanged; the upper layer and prediction head learn from supplied competition data. The unchanged selected model provides teacher predictions. No external weights, external data or paid hardware have been introduced.

| Approach | Completed outcome |
|---|---|
| Small causal dynamics branch | [Failed the speed gate](CAUSAL_BRANCH_RESULTS.md): nonzero-head throughput was 17.91% below the original baseline. It was not fitted or scored. |
| Upper-GRU/head weighted-MSE training | [Both 256-sequence trials regressed](../artifacts/round2/recurrent_mse_results.json) on the fixed screen, with negative paired intervals. Neither advanced to full development. |
| Direct clipped-correlation training | [Both 256-sequence trials improved](../artifacts/round2/correlation_screen_results.json) on the screen. The stronger `3e-5` 256-sequence candidate scored 0.629469402 on full development; the subsequent 1,024-sequence expansion scored 0.630019094. |

The two larger seeds reached screen WP **0.6395537025** and **0.6395466995**, compared with **0.6390994597** for the 256-sequence candidate and **0.6389160853** for the incumbent. The [latest experimental ZIP](../dist/submission_correlation1024.zip) contains the stronger seed's model. Its complete Linux callback was 13.4% faster than the original baseline at the median of three repetitions, and fresh offline extraction passed. These results establish neither a new live score nor hidden-judge runtime. See [Recurrent progress](RECURRENT_PROGRESS.md) for the evidence.

## Numerical method now in use

The historical strict Torch/ORT FP32 check and the conventional carried-Torch-state bridge **remain failed**. [Conversion documentation](ROUND2_CONVERSION.md), the [preserved failed bridge](../artifacts/round2/recurrent_bridge/report.json) and the [rounding investigation](../artifacts/round2/recurrent_roundoff_diagnostic/report.json) retain those results. Later tests did not relabel them or widen their limits.

The implemented method runs the current student's complete forward history in its original ONNX graph. Torch computes local gradients in 256-row chunks, starting each chunk from the corresponding detached ONNX state. Value corrections preserve ONNX forward states and predictions; derivatives remain an approximation. The [ORT-forward bridge](../artifacts/round2/ort_training_bridge/report.json) and the separate [correlation-gradient bridge](../artifacts/round2/correlation_training_bridge/report.json) passed their prescribed checks. Changed candidate weights must pass their own numerical and export checks before scoring.

## Completed cycle and next actions

Both 1,024-sequence runs completed under the [frozen policy](../artifacts/round2/correlation_expansion_policy.json), starting independently from the selected pretrained checkpoint. Both passed numerical and byte-exact export/callback checks. The stronger screen candidate, seed 20260912, passed three Linux CPU comparisons and sustained inference before using the final development allocation. Its final score supports the new experimental ZIP.

The original round-two caps are fully consumed: **12 screen evaluations and four full-development evaluations**. Confirmation remains permanently consumed. Do not reset this record to create more trials.

1. Use the packaged result for the next live submission when the user submits it. A live result is still required to assess public performance.
2. Run the prospectively frozen [TRAIN-only gradient diagnostic](CORRELATION_NEXT_EXPERIMENT.md): compare 256- and 1,024-row backward histories over identical supervised suffixes, then independently measure the frozen lower layer's gradient potential. This is a proposed diagnostic; it has not run.
3. Measure whether broader training remains beneficial. The consistent larger-data gains support this direction, but the experiment changed both available sequences and optimizer updates, so it does not isolate their effects or establish convergence.
4. Before another fitted candidate, declare a new bounded experiment and evaluation plan that acknowledges the already inspected development data. Prefer one evidence-supported change, such as longer backward windows or lower-layer adaptation. Preserve inference cost and numerical gates, and keep all current releases.

## Constraints and follow-on research

Training and development IDs remain disjoint. Training uses all post-warmup TRAIN rows without a validation mask, so its objective is a surrogate for the official masked domain. Confirmation is already consumed and has not been reused for these comparisons; a renamed split cannot restore independence. Sequence-bootstrap intervals assume sufficiently independent sequences and do not correct for adaptive selection.

Inference must remain offline Python 3.11 compatible, under one vCPU and 16 GB, within the 60-minute limit and below 20 MB compressed, targeting 18 MB for margin. Training is local with no paid hardware assumed. The judge's CPU, total workload and previous submission runtime remain unknown; local timings do not establish judge compliance. Preserve the supplied baseline, commit reviewable steps and exclude datasets, tensor caches and environments.

A longer credit-assignment diagnostic is a **follow-on investigation**, not a change to the completed training procedure. It should measure what the current 256-row gradient truncation misses before proposing longer chunks or another architecture. Any additional trained experiment requires a separately declared budget and evaluation plan. Report negative outcomes and uncertainty; do not repeatedly tune against live leaderboard feedback or promise that another architecture will reach 0.75.
