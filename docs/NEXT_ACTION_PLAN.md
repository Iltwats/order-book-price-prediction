# Next model improvement cycle

The first implementation should improve the current GRU readout, add a small source of causal information, and unlock conservative recurrent training. Preserve the released ONNX model and ZIP throughout. The supporting [research report](NEXT_MODEL_RESEARCH.md) explains the Hugging Face shortlist, eligibility checks and ten approaches.

## Fixed starting point

- Incumbent: `artifacts/selected/model.onnx`, SHA256 `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45`.
- Exact development WP: **0.629255535**. This is the score future experiments must beat.
- Existing Linux speed reference: selected **38,291 callbacks/s**, original **34,962 callbacks/s**. Streamed selected throughput is **36,249 rows/s**.
- Training: supplied training file only for this cycle. Feature order, target definitions, warm-up and state-reset behavior remain unchanged.
- Validation: existing 32-sequence correctness set, 256-sequence screen, then 1,400-sequence development set with 1,399 eligible blocks under the official mask.
- Confirmation: already consumed. Preserve its receipt and historical results; do not reset the ledger or create a renamed copy to regain access.

The tasks below define the bounded implementation cycle. On 2026-09-13, Step 1's registry and conversion checks were implemented and Step 2's five head candidates were declared before fitting and screening. Later steps remain conditional. The user-provided leaderboard screenshot records the previous submission at **0.6083, rank 26**; its exact uploaded ZIP hash is not established by the screenshot.

**Completed head stage:** five primary screens plus one fixed training-subset repeat, and two full-development evaluations. The best affine correlation head gained `0.000051378` development WP; its repeat gained `0.000017271`. Both 95% intervals include zero. See [actual results](ROUND2_RESULTS.md) and the [300-word explanation](ROUND2_SIMPLE.md). The candidate is experimental, and the existing default release remains available. Six screen slots and two full-development slots remain under the original caps.

## Step 1: establish round-two evidence and conversion checks

Create a new experiment registry separate from the old confirmation workflow. Each entry records hypothesis, parent model hash, code/config hash, training groups, validation groups, seed, teacher lineage, transformations, timing, score and failed gates. The incumbent comparator is the selected calibrated model. The current `compare_confirmation_once` logic uses the original raw baseline and cannot serve unchanged as the next-round selector.

Audit a zero-update conversion for both supplied and selected checkpoints: read ONNX weights into the trainable representation, apply the inverse gate mapping, and replace initializers in the same original graph. Require identical parameters, recurrent states and predictions under the same ORT runtime across 32 complete sequences. Then compare equations and gradients with an FP64 reference on small nontrivial cases.

Retain the existing failed FP32 Torch/ORT parity report. A separate approximate training branch needs a prospective, justified numerical budget after controlled diagnostics; it must never be relabelled as a strict pass. Stop on unexplained equation, mapping, derivative or long-sequence stability failures.

**Deliverables:** round-two registry, conversion/gradient reports, a measured 16-sequence neural training timing pilot when the numerical gate permits it. This stage changes no deployed weights.

## Step 2: optimize the prediction head for correlation

Use the current model's original ONNX recurrence to extract representations. Begin with the same 1,024 training sequences used for the selected affine fit, so a changed training objective can be distinguished from a changed dataset.

| Experiment | Configurations | Purpose |
| --- | ---: | --- |
| Two-output affine correction | 2 | Weighted-MSE control versus per-sequence correlation surrogate, both anchored to selected outputs |
| Final-state linear correction | 3 | Correlation surrogate with correction penalties `1e-4`, `1e-3`, `1e-2` |
| Conditional exact-clipped refinement | At most 1 | Refine only a promising readout using complete-sequence clipped correlation |

For the 128-state surrogate, accumulate target-specific per-sequence weighted covariance and cross-covariance in FP64. Never pool all sequences before computing correlation. Use a parameter anchor to control the scale ambiguity. Limit optimization to 200 deterministic iterations per candidate. Handle zero weights and variance explicitly.

**Implementation clarification, declared before scoring:** freeze the selected intercept in training-centered coordinates. Each target's center is the equal-sequence mean of its label-weighted training feature means. As coefficients change, fold `b = b0 + (beta0 - beta) @ center` into the one global output bias. This preserves the selected weighted training prediction mean and makes the anchor insensitive to feature translation. It does not estimate a bias separately for validation sequences. Both affine controls and all three hidden-state heads use this same convention; it adds no inference work. The complete method is documented in `ROUND2_HEAD_METHOD.md`.

The 1,024-sequence dense covariance pilot needs approximately 268 MB plus vectors and metadata. Time extraction on 32 sequences before extrapolating. This method avoids storing every hidden-state row, but covariance accumulation costs O(rows × width²); it is not as cheap as fitting six affine coefficients.

Training has no scoring mask, so its objective uses all eligible post-warm-up rows. Exact development WP uses the supplied mask, clips targets and predictions, and applies the official block eligibility rules. The moment method optimizes an unclipped-prediction surrogate. Exact clipping cannot be implemented merely by clipping the stored moments.

**Advance:** positive full-development delta against the incumbent, credible paired sequence behavior and no added callback work. Fold winning final-state linear weights into the existing head. **Stop:** surrogate loss improves while exact WP does not, variance degenerates, or improvement disappears in the prescribed repeat.

## Step 3: test two causal feature corrections

Keep the incumbent's recurrent inputs and preprocessing intact. Fit each correction from zero, with identical training groups and labels. Introduce only one feature hypothesis per model.

1. **Cross-instrument lag correction:** learn small projections of documented feature groups, retain a few event-row lags, and combine i1 history with current i0 features. No timestamps or physical price-level ordering are assumed.
2. **Dynamics correction:** current features, same-column differences and EMA8 deviations feed a small residual MLP. The proposed `336 → 32 → 2` version adds approximately 10.8k MACs per row; start smaller if the callback budget fails.

Initialize moving averages from observed first rows and reset all filters with the GRU. Fit branch normalization on training data. Do not apply 112-channel RevIN statistics directly to two anonymous output targets. Do not infer best quotes or real order-flow imbalance from undocumented column positions.

Export a jointly evaluated graph with an explicit state contract. The current runtime accepts exactly two GRU states; additional filter buffers require a validated adapter. Benchmark an untrained zero-output branch first to establish its overhead before spending time on fitting.

**Advance:** new information produces a positive incumbent-relative development gain and complete callback throughput remains at least the original baseline. **Stop:** extra Python/ORT calls or state copies exhaust the roughly 2.49-microsecond allowance. These feature experiments are distinct from the already unsuccessful smoothing of output predictions.

## Step 4: partially fine-tune the pretrained recurrence

Start from the selected affine-folded model. Freeze the lower GRU; train the upper GRU and head. Add parameter groups and verify the lower layer remains bitwise unchanged. An explicit penalty toward pretrained parameters is different from AdamW decay toward zero; document its normalization and apply it once per whole sequence-batch update, not once per chunk.

Use 256 training sequences, batch eight, 256-row chunks, gradient clipping 1.0, and one epoch. Compare learning rates `1e-5` and `3e-5` with the same chosen anchor convention and teacher weight 0.1. The teacher should represent the actual selected ONNX model, using a verified bounded cache or adapter. The current teacher implementation reconstructs Torch; claiming an exact ORT teacher requires additional work.

Keep recurrent state across chunks and detach gradients at boundaries. Forward history is longer than the backpropagation chunk. Accumulate the existing full-sequence-normalized supervised loss before one optimizer update per sequence batch. Teacher matching needs an explicit output-scale and weighting policy; do not silently match uncalibrated raw outputs.

**Expand only a winner:** first 1,024 training sequences, then consider 512-row chunks, both recurrent layers and additional epochs. Maximum three epochs for an expanded configuration. Use the measured timing pilot to budget CPU hours; the old one-sequence smoke run does not establish full-training time.

## Shared first-round trial cap

The ten research approaches are not ten independent hyperparameter sweeps. Limit the first implementation round to **12 candidate evaluations reaching the 256-sequence screen**:

- 2 affine objective controls;
- 3 correlation readouts;
- 2 causal-feature corrections;
- 2 partial-GRU learning rates;
- at most 1 clipped-correlation refinement;
- 2 reserved robustness evaluations for the strongest candidates: second training seeds for stochastic fits, or predeclared training-sequence subsamples for deterministic solvers.

The no-fit incumbent is a fixed control. Advance at most four frozen finalists, including robustness repeats, to full development. Do not add an output-scale grid in this round; selected initialization and explicit anchoring keep the comparison focused. Any change to the cap or replacement of a planned arm is recorded before seeing that arm's result. Detailed research memos describe additional possibilities; this shared cap takes precedence for the first implementation. Report both robustness results; do not discard an unfavorable repeat or call an identical deterministic rerun an independent replicate.

Use sequence-disjoint folds for fitting new heads, feature branches, teachers and stacking weights. The supplied encoder's unknown original training lineage prevents calling training-file folds globally unseen. All validation sequences have already been inspected. Nested development checks and seed repeats improve discipline but do not create a fresh holdout.

## Conditional second round: external transfer and smaller students

Only after the first round, register a separate small transfer budget:

| Order | Experiment | Required comparison and stop condition |
| --- | --- | --- |
| 1 | IBM Granite PatchTSMixer | Pretrained versus random initialization with the same causal adapter/head; stop unless transfer adds signal beyond both random initialization and incumbent |
| 2 | Chronos-2-Synth offline representations | Features-only trailing context plus a fitted two-target head; compare with raw-window and GRU-state controls before generating a large teacher cache |
| 3 | GRU96 or GRU64 distillation | Labels plus a verified eligible teacher; student must preserve useful improvement and satisfy actual callback timing |
| 4 | Cached TCN32 or minGRU64 | One architecture only if smaller GRU results motivate it; separate state/export adapter and train-versus-step parity required |

Pin model revisions, weight hashes, licenses and training lineage before adoption. TTM r1/r2/r2.1 weights and financial competition checkpoints remain excluded. TTM r3 is pending a sufficiently detailed corpus disclosure. No external teacher may consume historical true targets or future feature rows during prediction generation. Keep external teacher weights out of the submission when only the student is deployed.

For ensembles, assess residual complementarity using fold predictions. Multiple affine heads on the same recurrence can often be combined algebraically into one head; independent GRUs cannot. Clipping each member before blending generally prevents linear folding. Two GRU64 models have approximately 117k combined MACs versus the incumbent's 191k, but that arithmetic is not proof of lower callback time or higher score.

## Timing and release decision

On the existing measured Linux host, projections at the selected model's measured streamed rate are approximately **2.4 minutes for 256 sequences**, **12.9 minutes for development**, and **17.2 minutes for all provided validation**. These are planning estimates for comparable callbacks. Keep confirmation closed even though its runtime is easy to estimate.

The existing six-coefficient affine solver fitted 1,024 sequences in 216.8 seconds. Linear extrapolation suggests 14.5 minutes for 4,096 or 37.4 minutes for all 10,607; neither larger fit has been measured, and those numbers do not apply to covariance construction, neural training or foundation-model adaptation. Budget those jobs from representative pilots.

Select the best feasible incumbent-relative development candidate, report both target scores and paired sequence differences, and use 10,000 sequence-bootstrap resamples with explicit selection-bias limitations. A gain of about 0.001 is a proposed threshold for justifying substantial recurrent-training expansion, not a guaranteed outcome or a reason to discard a smaller robust gain from a free linear head.

Before release, verify warm-up, state reset, causality, determinism, finite outputs, output ownership and full-sequence export behavior. Measure three Linux one-CPU repetitions, streamed end-to-end time, memory and startup separately. Require no slowdown versus the original baseline until representative judge measurements establish additional headroom. Validate the extracted ZIP offline under 16 GB and target less than 18 MB compressed. No hidden-judge runtime guarantee is possible without its workload and CPU details.

Preserve datasets, caches and training environments outside Git and the ZIP. Commit implementation in small stages: protocol/conversion checks, head objective, causal branch, recurrent training support, then validated evidence and release. Do not alter the current release until the next candidate passes the applicable gates.
