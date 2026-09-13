# Working target: at least 0.75 live WP

The user's target is **0.75 or higher on the live leaderboard**. The reported prior public result is **0.6083**; the gap is **0.1417**. The best new experimental head has local development WP **0.629306913**, just **0.000051378** above the frozen selected incumbent. Development and public test scores concern different data. The small head gain does not establish a route to 0.75, and there is currently no evidence that a particular architecture will reach that target.

## Change in emphasis

Stop allocating ordinary search effort to tiny affine or scale changes. Spend the next experiments on predictive representation and causal information. Continue using the supplied GRU and supplied competition data. A strong model may ultimately require more training data, a different architecture or distillation; choosing a target does not prove that the data support it.

1. **Train the upper recurrence.** Keep the lower GRU and raw preprocessing fixed. Obtain student states from the original ONNX graph with the current student weights, and teacher predictions from the unchanged selected model. Carry full forward history in ONNX; compute local gradients in 256-row Torch chunks reanchored to those states. This changes the representation while retaining the deployed architecture and callback operation count.
2. **Add causal information.** Prototype one small fused feature branch using cross-instrument lags or projected dynamics. Establish CPU overhead with a nonzero test head as well as zero-output identity before spending on fitting. A branch must reset its auxiliary state and use only current/past features.
3. **Expand a reproducible winner.** Use the existing 256-sequence recurrent pilot configurations first, then consider 1,024 training sequences, and eventually all 10,607 if improvement survives the declared checks and measured training time is practical. Expansion is a separately declared experiment, not an unrecorded replacement of a screened candidate.
4. **Use model diversity if justified.** A causal window model or smaller GRU student can be tested after the current round. Only complementary, useful predictions justify an ensemble; its combined callback must pass the CPU limit. External model weights remain subject to the earlier provenance findings, not a shortcut assumed to deliver 0.75.

## Immediate recurrent experiment

Use the first 256 groups of the existing 1,024-sequence training anchor. Freeze the two planned learning rates, `1e-5` and `3e-5`, before their outcomes. Each configuration runs one epoch, batch eight, 256-row chunks, FP32, gradient clipping 1.0 and AdamW decay `1e-4`. Teacher strength is `0.1`; the separate relative parameter anchor is `1e-3` toward the selected upper GRU/head and applies once per sequence-batch update. Label-weighted teacher matching uses the same full-sequence denominators as supervised loss, with warm-up excluded. Teacher outputs are the selected model's raw outputs, without another calibration or clipping step.

First measure a 16-sequence timing pilot without validation scoring. It is not a candidate-selection trial or a reason to tune unrecorded settings. Save atomic checkpoints and exact source/data/configuration hashes. Training remains local; the current environment reports ten logical CPUs and no available Torch MPS device. Inference still uses one CPU. No paid hardware is assumed.

## Prospective numerical bridge

The old strict FP32 Torch-versus-ORT check remains failed. The byte-identical inverse weight export and controlled FP64 equation/gradient checks passed, so a separately labelled approximate-training branch can be tested. All final scores still come from exported original-graph ONNX inference.

Before new training-data diagnostics, freeze these engineering acceptance limits. Normalize errors per coordinate by `max(1, reference RMS)`; this preserves absolute bounds for GRU states and avoids unstable relative errors near zero.

| Domain | Maximum normalized error | Normalized RMS error |
| --- | ---: | ---: |
| Carried full-sequence predictions/states, including exact-ORT-lower / Torch-upper training path | `512 × FP32 epsilon = 6.103515625e-5` | `32 × epsilon = 3.814697265625e-6` |
| Independent one-step outputs/states from common ORT prior states | `128 × epsilon = 1.52587890625e-5` | `8 × epsilon = 9.5367431640625e-7` |

Also require finite values, clipped training WP drift at most `2e-6` in the mean for each target and `2e-5` for every eligible sequence/target. Run two fixed training sequences first, then eight if they pass. These are prospective engineering budgets informed by the historical discrepancy, not a formal roundoff theorem or a replacement strict-parity pass. Stop on unexplained equation, mapping, derivative or threshold failures. Repeat the appropriate numerical check on changed weights before validation.

## Numerical result and a different training method

The prospective carried-state bridge above **failed on its first prescribed training sequence** (group 2309); it did not proceed to the other seven. Its protocol and failure remain in `artifacts/round2/recurrent_bridge/`. Actual-weight FP64 equations and independent common-state steps passed. Upper hidden-state maximum error was about `7.09e-5` and worst-coordinate RMS about `4.16e-5`, exceeding the unchanged budgets. Small prediction/WP drift does not reverse that failure.

A separate diagnosis on that same sequence, recorded in `artifacts/round2/recurrent_roundoff_diagnostic/report.json`, found a slowly changing upper coordinate whose median update gate was approximately `0.999998`. Signed rounding differences persist and accumulate there. Equivalent FP32 update formulas and a carried FP64 calculation also disagree with ORT, supporting an arithmetic-propagation cause rather than a gate mapping error. This diagnostic is evidence on one sequence, not general acceptance.

The next method therefore changes forward execution. Export the **current student** into the unchanged original graph before each sequence-batch, run its complete causal ORT history, and hold those weights fixed until that batch's optimizer update. Reanchor each 256-row Torch chunk to the student's exact ORT state immediately before the chunk. Use detached value corrections at the hidden sequence and prediction so that loss values and head inputs use ORT values, while local recurrent derivatives come from Torch. Keep the selected teacher unchanged. Discard each local Torch final state rather than carrying its accumulated error into the next chunk.

This is ORT-forward training with approximate local gradients; it is not strict cross-runtime arithmetic parity. A **separate prospective protocol** in `artifacts/round2/ort_training_bridge/` must pass on the original eight fixed training groups before training. Retain the numerical limits above for independent steps and raw reanchored chunks. Compare local upper-GRU/head gradients against FP64 on fixed real chunks starting at rows 0 and 19744, with `atol=1e-6`, `rtol=1e-4`, finite values and the same declared loss. Inspect raw Torch errors before any value correction. Repeat the candidate check on changed weights. Do not increase limits after observing failure.

The user reports that the leaderboard does not expose the prior submission's runtime. The measured original Linux callback therefore remains the speed gate until better platform timing is available.

## Evaluation discipline and completion criteria

Continue the same round-two registry, selected incumbent and original 12-screen / four-full-development caps. Six screens and two full-development slots remain after the head stage. The old confirmation set remains permanently consumed. Train/validation sequence IDs stay disjoint for these experiments; the supplied training file has no scoring mask, which makes its objective a surrogate for the masked validation domain.

The immediate deliverable is tested recurrent-training support, measured timing, the two declared training/evaluation outcomes when numerical and compute checks allow them, and the causal branch's concrete design/overhead path. Commit in stages and exclude datasets, tensor caches and environments. Preserve the existing and experimental ZIPs.

**The 0.75 target is reached only by a qualifying live result.** A local score or a predicted improvement cannot close that objective. Report negative results and uncertainty. A larger research round requires an explicit prospective budget and honest validation reuse; it must not reset the consumed confirmation history or repeatedly tune against public leaderboard feedback.
