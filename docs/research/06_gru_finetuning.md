# Approach 6: safely unlock pretrained GRU fine-tuning

Research date: 2026-09-12. Recommendation: investigate a trainable approximation while preserving exact ONNX deployment semantics. The supplied GRU remains the starting point; no new weights, training run or numerical pass is claimed here.

## What the current evidence establishes

[training.py](../../connectome/training.py) correctly reorders ONNX `[z,r,h]` into PyTorch `[r,z,n]`, preserves separate input/recurrent biases, and maps the head transpose. [PyTorch GRU](https://docs.pytorch.org/docs/2.5/generated/torch.nn.GRU.html) applies reset after the candidate recurrent affine operation, matching [ONNX `linear_before_reset=1`](https://onnx.ai/onnx/operators/onnx__GRU.html). Reset-before-matrix multiplication would be a different model.

The [recorded audit](../../artifacts/training_smoke/torch_strict_parity_audit.json) remains **failed**: 1,159/40,000 prediction values exceed `atol=1e-6, rtol=1e-5`, maximum absolute prediction error 1.19209e-5, and upper-state error 2.76566e-5. Common ORT previous states reduce prediction violations to one. The [training documentation](../training.md) records WP drift of only approximately 1.68e-7. This supports accumulated kernel rounding as an explanation; it does not prove every implementation detail correct.

[PyTorch's numerical-accuracy documentation](https://docs.pytorch.org/docs/2.5/notes/numerical_accuracy.html) explicitly distinguishes mathematical equivalence from bitwise equality, including batched versus sliced computations. Deterministic settings prevent certain nondeterministic algorithms; they cannot make different operation orders identical. Consequently, bitwise Torch/ORT equality is unnecessary for a model intentionally changing weights, but parameter mapping, gradients and deployed model quality still require independent verification.

## Experiment 1: establish the conversion and gradient contract

Use both original and selected affine-folded checkpoints without optimizer updates. Convert Torch tensors back through the inverse gate permutation into a clone of each **original ONNX graph**, retaining node order, attributes, names, shapes, dtype and preprocessing. Compare all initializer arrays bitwise. Run 32 complete development correctness sequences through original and roundtripped ORT callbacks; require predictions and both recurrent states bitwise equal. This tests conversion rather than simply copying the source file. Existing Torch re-export may alter graph structure; initializer replacement isolates that variable.

Separately implement an equation-level FP64 reference. Test nonzero initial states, distinct input/recurrent candidate biases, saturated gates and representative real inputs. Compare one-step and short unroll outputs and gradients with `nn.GRU`. Use [double-precision gradcheck](https://docs.pytorch.org/docs/2.5/notes/gradcheck.html) on a small GRU and directional finite differences on sampled directions in the actual GRU128. Check input, initial-state and recurrent-parameter derivatives, not only the head. Confirm chunking without detach agrees with a complete short unroll; detaching should remove only cross-boundary gradient paths.

**Stop gate:** any weight/graph mismatch or unexplained equation/gradient discrepancy blocks training. Keep the existing strict FP32 audit unchanged. For a separate approximate branch, prospectively document a conversion-error budget after FP64 and training-only stress diagnostics. A proposed score budget is absolute mean Torch-versus-deployed-ORT WP drift below `1e-5`, one hundredth of the proposed minimum worthwhile `0.001` improvement. Also record maximum/percentile state and prediction errors and reject unbounded accumulation/nonfinite outputs; score agreement alone is insufficient. Fix these thresholds before inspecting new development outcomes. This is a new acceptance protocol, never a retrospective parity pass.

## Experiment 2: unfreeze the upper GRU conservatively

Train the upper GRU and head while freezing the lower GRU. The current API only exposes all-recurrence versus head-only freezing, so a parameter-group extension is required. Verify frozen tensors remain bitwise unchanged after optimizer steps. Keep a frozen selected ONNX teacher to anchor the actual deployed function; the existing teacher uses reconstructed Torch and defaults to its source checkpoint.

Run a bounded six-arm pilot: two initializations—selected affine-folded weights and raw baseline—crossed with zero-decay, existing AdamW decay `1e-4`, and an explicit pretrained-anchor penalty. Use identical 256 training sequences, seed, one epoch, batch eight, chunks 256, recurrent learning rate `1e-5` and gradient clipping 1.0. Teacher matching initially stays fixed at 0.1. Give both starting points the same development-only calibration opportunity.

[AdamW](https://docs.pytorch.org/docs/2.5/generated/torch.optim.AdamW.html) shrinks parameters toward zero. For the anchor arm, disable that decay on transferred tensors and add `lambda * sum_l mean((theta_l-theta_start_l)^2)`, once per whole sequence-batch update, initially `lambda=1`. Log its gradient norm relative to supervised gradients to diagnose an ineffective penalty. Do not repeat the full penalty per chunk or equate its scale with AdamW's coefficient. [L2-SP](https://proceedings.mlr.press/v80/li18a.html) supports anchoring transferred parameters, but its experiments concern convolutional transfer, not evidence of gains on this GRU.

Prefer selected initialization because its output calibration is already useful; both recurrent initializations contain identical pretrained recurrence. Raw baseline remains a control for optimization path dependence. Anchor recurrence to supplied values and the head to its chosen starting values.

**Stop gate:** export each candidate into the validated original graph; score its actual ORT callback on 256 development sequences. Stop nonfinite runs, conversion-budget failures and candidates that do not improve over the selected model. Preserve every artifact and failed result.

## Experiment 3: longer credit assignment, then full recurrence

For the winning pilot only, compare chunks 256 and 512 at equal sequences, optimizer updates and seed; test 1,024 only if gradient-lag diagnostics or development gains justify it. Carry hidden state across chunks and detach at boundaries. Longer chunks extend gradient credit, not inference context; they increase activation memory approximately linearly. [Adaptive TBPTT research](https://proceedings.mlr.press/v115/aicher20a.html) explains truncation bias and motivates measuring gradient decay instead of assuming 256 is sufficient.

Then unfreeze both GRUs and screen learning rates `{1e-5,3e-5}` with teacher weights `{0,0.1}`. Keep the selected chunk/anchor convention fixed. Expand the winner from 256 to 1,024 training sequences before considering all available training sequences, measuring CPU training cost first. Extend only improving runs, to at most three epochs; repeat the winner with a second seed.

Retain existing full-sequence loss denominators and one optimizer update per sequence batch. Warm-up rows update state but have no supervised loss. Inputs never contain targets, future features, `is_scored` or sequence identifiers. Validate using the exact [masked, clipped, equal-sequence WP](../../METRIC.md), not training MSE.

**Release gate:** at least +0.001 full-development WP versus the selected incumbent, paired sequence uncertainty reported, conversion diagnostics passing, and complete callback throughput no slower than the original baseline. A fresh independent confirmation protocol is required before another holdout claim; the consumed confirmation remains closed. Keep the existing release available throughout.

Source audit: three Exa searches, 15 result slots, seven successfully fetched primary-source pages. Project files were read directly. Proposed thresholds and experiments are engineering choices, not published guarantees.
