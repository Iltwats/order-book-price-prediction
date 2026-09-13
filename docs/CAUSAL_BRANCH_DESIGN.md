# Two causal residual branches

Design date: 2026-09-13. This is a prospective implementation specification, with no branch training, scoring, or runtime measurements. The live target is WP ≥0.75; no available evidence establishes that these branches can reach it. The completed readout experiments produced very small development changes, so the next useful hypothesis is access to explicit input dynamics. The comparison remains the selected model `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45`, not the slightly improved experimental head.

Implement the two arms separately. Preserve the original 112-feature path, both GRUs, their 192,000 recurrent parameters, and the selected linear head. Compute `prediction_t = selected_prediction_t + residual_t` in the same synchronous ONNX call. The branch sees current/past supplied features only. Instrument indices identify stable columns; they do not identify ordered depth or physically paired prices and volumes. Lags and half-lives below are in event rows. [Data contract](data_overview.md), [current runtime](../connectome/runtime.py), [feature research](research/08_causal_features.md)

## Recommended arms

| Arm | Branch features and layers | Additional learned parameters | Analytical MACs per row | Auxiliary FP32 state |
| --- | --- | ---: | ---: | ---: |
| Cross-instrument lag | Two `52→8` projections; concatenate current i0/i1, i1 lags 1/4/16, eight extras; `48→16→2` ReLU MLP | 1,666 | 1,632 | 129 values |
| Dynamics | `112→16` projection; concatenate current projection, same-channel difference, EMA8 deviation; `48→16→2` ReLU MLP | 2,626 | 2,592 | 33 values |

Counts include projection/MLP biases as parameters, but MACs exclude elementwise operations, normalization, state movement and operator-launch overhead. These arithmetic increases are approximately 0.86% and 1.36% of the incumbent's 190,720 MACs. They are **not latency predictions**. At measured Linux rates, the allowance between selected and original callbacks is only about 2.49 microseconds per row. Extra ORT operators can consume that allowance despite small matrices. [Measured reference and gates](NEXT_ACTION_PLAN.md)

### Arm A: current i0 against short i1 history

Use `x0=x[0:52]`, `x1=x[52:104]`, and `a=x[104:112]` in documented numeric order. Standardize each branch input with frozen training statistics. Learn `u0=P0*z0+b0` and `u1=P1*z1+b1`, each with eight outputs. Define

`q_t = concat(u0_t, u1_t, u1_(t-1), u1_(t-4), u1_(t-16), standardized_a_t)`.

The first layer of the MLP learns combinations of current i0 and lagged i1 without assuming matching component semantics. Do not add multiplicative interactions, more lags, or i0 history as unregistered variants. The existing GRU already receives both instruments; the hypothesis is easier access to short cross-instrument history, not access to a new data source.

Incoming history is a newest-first `16×8` array containing `u1_(t-1)` through `u1_(t-16)`. Read rows 0, 3 and 15 before updating. Return `concat(u1_t, history[0:15])`. On the first row, replace all missing history with that row's observed `u1_t`; this is a specified causal left-padding rule. All three selected lags are real observations by the end of warm-up. A fixed shift of 128 floats is acceptable for the first implementation; do not introduce Python ring-buffer calls or dynamic indexing without measured benefit.

### Arm B: projected changes and departures from a causal mean

Learn `u_t=P*z_t+b`, with 16 outputs. Incoming state contains `u_prev` and `m_prev`. With `alpha=1−2^(−1/8)`, define

`delta_t = u_t − u_prev`

`innovation_t = u_t − m_prev`

`q_t = concat(u_t, delta_t, innovation_t)`

`m_t = m_prev + alpha*(u_t − m_prev)`.

Compute the innovation against the **previous** mean, then update the mean. On the first observed row, set both previous vectors to `u_t`, giving zero difference and innovation. Return `(u_t,m_t)`. Do not divide by a moving variance in this arm: square roots, extra state and adaptive scaling would introduce another hypothesis. This filters input representations, independently of the already explored smoothing of output predictions.

## Initialization, normalization and fitting

Initialize projections and first MLP weights with a fixed, declared Xavier seed; use zero biases. Initialize **only the final residual output layer** to zero. Zeroing every layer would prevent useful hidden-layer learning. The untrained model must reproduce the selected model's output and GRU states before any score is inspected.

Fit per-column mean and population variance using only the declared training groups, including their warm-up feature rows. Accumulate in FP64 with bounded row-group buffers. Use `scale=std` when `std≥1e-6`, otherwise `scale=1`; record constant/near-constant columns and the complete statistics hash. Apply this normalization to the branch only. Fit no validation-dependent normalization, per-sequence future statistics, or two-target inverse RevIN mapping. Do not add clipping or imputation without observed evidence and a prospective rule.

Fold fixed standardization into the projection at export: for row-vector convention, `P_fold[j,:]=P[j,:]/scale[j]` and `b_fold=b−mean@P_fold`. Keep the eight direct extras in arm A under their recorded affine normalization. Folding changes FP32 operation order: compare unfolded and folded branch calculations on full sequences and investigate differences against `atol=1e-6, rtol=1e-5`; never imply bitwise equality merely from algebra.

Proposed first fit for each arm: the same original 1,024 training groups, seed 20260912, one epoch, batches of eight complete sequences processed in 256-row chunks, AdamW learning rate `1e-3`, weight decay `1e-4`, and gradient clipping 1.0. Declare these values before fitting; they are a starting design, not tested settings. Optimize the existing sequence-normalized, target-weighted MSE against targets clipped to ±2, with a fixed 0.1 matching penalty to the actual selected output. Since the base is frozen, that matching penalty is the corresponding weighted squared residual. Use post-warm-up rows; never consume a validation scoring mask during fitting.

Generate base outputs with the selected **ONNX** model, not an unverified reconstructed Torch teacher. Stream or cache only the two FP32 outputs per row with hashes: 1,024 complete sequences require about 164 MB before metadata. Maintain branch state through each sequence, detach at chunk boundaries, and accumulate gradients before one optimizer update per complete sequence batch. Projection parameters consequently stay fixed while a sequence's history is generated. Reset between sequences. Chunk detachment truncates training gradients; it must not truncate forward history. No validation-driven checkpoint sweep is included in this one-epoch declaration.

## Explicit streaming and export contract

Keep the public no-argument `PredictionModel()` and `predict(DataPoint)` contract. The proposed graph interface is:

| Input | Output | FP32 shape |
| --- | --- | --- |
| `features` | `prediction` | `[1,1,112]` → `[1,1,2]` |
| `hidden_0` | `next_hidden_0` | `[1,1,128]` |
| `hidden_1` | `next_hidden_1` | `[1,1,128]` |
| `branch_state` | `next_branch_state` | `[1,1,A]`, A=129 or 33 |

The last auxiliary value is an initialized flag: reset to zero, return one. Use it to select first-observation initialization entirely inside the graph. The lag arm stores flattened history in the other 128 values; the dynamics arm stores previous projection and EMA in the other 32. State includes no target, sequence ID, clock time, or future row. Every call, including the 99 warm-up rows, advances all states.

Add new modules rather than editing frozen sources: a branch graph builder, a `BranchOnnxBackend`, training/cache utilities, and a branch evaluation runner. The existing `StreamingPredictor` accepts a backend through its constructor and can enforce ordering/warm-up/ownership unchanged. However, `OnnxBackend` and `load_predictor` currently accept exactly two recurrent states; they cannot load this graph unchanged. A separate candidate entrypoint must instantiate the branch backend and pass it to `StreamingPredictor`. Merely adding an auxiliary field to JSON will not change the current loader.

Create two persistent state banks and two CPU I/O bindings, including the auxiliary tensor. Flip banks after one ORT call; return an owned copy of the two predictions. Additional auxiliary bank storage is only 1,032 bytes for arm A or 264 bytes for arm B. Reuse allocated inputs, outputs and bindings. Keep single-thread sequential ORT, disabled spinning, CPU FP32, and affinity disabled. A startup fallback, if packaged, must support the **same branch graph and state**; falling back to the selected model after training would change the candidate. Runtime failure must raise rather than silently change models mid-sequence.

Build by copying the original ONNX graph and appending a residual path with ordinary `Slice`, `Concat`, `MatMul`/`Gemm`, `Add`, `Sub`, `Mul`, `Relu` and initialization-selection operators. Preserve original initializer bytes and the original raw-input path. Retain explicit state outputs so ORT cannot discard updates. Begin with the two small instrument projections in arm A; an unscored timing fixture may also compare one block-sparse dense projection to reduce operator count. This is an implementation-layout comparison, not permission to score additional fitted hypotheses.

The current sequence exporter, parity helper, evaluator and benchmark hard-code two-state handling. Do not pass a branch model through them unchanged. First make a new row-callback harness emitting the existing report schema. For faster full-sequence evaluation, create a separate export that computes projected lag shifts and causal EMA over a complete sequence; reset/pad exactly as above and verify against the row graph. An ONNX `Scan`/loop is a correctness option, not an assumed speed optimization. Freeze both step and bulk graph hashes in the new runner's parity receipt.

## Gates before spending evaluation slots

1. Check actual initializer bytes: all 192,000 recurrent values and the selected base-head values remain unchanged. An expanded graph must not weaken the baseline-specific native extractor; native execution is unsupported for these branches.
2. Verify a zero-residual model against the selected callback across the 32 complete smoke sequences: same recurrent states and required predictions, with bitwise checks for the unchanged path. Verify auxiliary state against an independent NumPy feature implementation, including first row and chunk boundaries.
3. Benchmark an unscored seeded **nonzero-head** fixture as well as the zero-output graph. A graph optimizer could prune zero-output MLP work, making zero-head timing unrealistically favorable. Hash the timing fixture and inspect executed operators when needed; do not evaluate its WP.
4. Exercise sequence resets, all warm-up rows, repeated identical input, alternating/extreme finite inputs, output ownership, and rejection of invalid steps/nonfinite input. Perturb future rows and confirm every earlier prediction/state is unchanged. Compare two interleaved model instances with isolated runs to detect shared buffers.
5. Compare trainable step and exported step with carried state; compare any bulk export across all 20,000 rows and final states. Record both tolerances and actual maximum error. Strict conversion evidence is separate from any prospectively justified approximate training bridge.
6. Measure three complete-callback Linux repetitions against the original baseline on identical sequences. Stop before fitting if the extra work consumes the available timing margin. Before release, repeat after training, measure streamed time/memory/startup, and validate the candidate package in a fresh offline process.

Only after these checks, reserve each arm's 256-sequence screen and record it in the existing registry. Any screen reservation, including a failed one, consumes its slot. Full-development results reuse inspected sequences and are not independent confirmation. The old confirmation ledger remains closed.

## Optional lower-layer readout control

`prediction = selected_prediction + C*lower_state`, with `C` initially zero, adds 256 MACs, 256 weights, and no filter state. It exposes the lower GRU state directly without changing recurrence. The existing `with_head_state(...,'both')` and frozen-head export offer a starting point, while runtime still uses two states. This control adds no raw information and is lower priority than the two input branches. It is a separately fitted hypothesis: register it only by prospectively replacing a remaining planned arm or in a later round. It is not a free extra score because it is inexpensive. [Training/export implementation](../connectome/training.py)

## Remaining registry requirements for upper-GRU training

At the completed-head audit, the registry contains six screens and two full-development evaluations, leaving six and two respectively under the unchanged 12/4 caps. Preserve the selected comparator, manifest and all consumed confirmation artifacts. Current sources are frozen; add a partial-training module, bridge and runner as new files and hash all imported implementation dependencies in each new declaration, including `selection.py`.

Use one candidate per learning rate (`1e-5`, `3e-5`), with the same declared 256 training groups, seed, one epoch, batch size eight, 256-row chunks, frozen lower GRU, clipping 1.0 and complete-batch optimizer update policy. Record `training_subset_reason` because 256 differs from the 1,024-group anchor. Declare teacher weighting/normalization and pretrained-parameter anchoring explicitly; the active implementation proposal uses teacher weight 0.1 and joint relative parameter anchor 0.001 once per whole sequence batch. AdamW decay toward zero is a separate setting. Teacher lineage may name only the selected supplied-data model in this registry.

Registration must precede creation of the candidate config and training report. The completed report must contain `complete=true` and `specification` fields `groups`, `seed`, `epochs`, exact `parameters`, `source.sha256`, and `data` with the training path, size and `mtime_ns`; also retain the full training fingerprint, teacher/cache hashes, bridge receipt and lower-layer byte-equality evidence. Record parameters even though the legacy validator permits their omission: an omitted field is not evidence of exact agreement. Reject unexpected defaults and record the actual learning rate, optimizer, loss normalization and update count.

The current registry checks deployment model hashes against evaluation receipts but does not itself prove that a training report's declared output model equals that file; the new runner must cross-check its output-model hash. Reserve before screening. A full-development reservation requires the same model, config and training-report hashes as the successful screen. Do not append benchmark fields to that frozen training report; put later evidence in separate artifacts. Failed reservations cannot be recycled. A new epoch, seed, postprocessing variant or changed checkpoint needs its own prospective candidate, and any source repair needs an explicit bounded amendment rather than silently changing old declarations.

The original round's budget remains shared across feature corrections, recurrent fits and robustness checks. A live target of 0.75 changes the ambition, not the meaning of reused development results or the remaining trial count. Expand training only on a measured, feasible candidate with credible improvement and a separately declared next budget.
