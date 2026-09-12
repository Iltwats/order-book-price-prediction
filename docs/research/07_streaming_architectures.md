# Approach 7: lightweight causal streaming architectures

Verified 2026-09-12. Research only; the selected GRU, runtime, and release remain unchanged.

## Recommendation and constraint

Rank experiments as **GRU96/64 distillation, minGRU64, then a cached temporal-convolution student**. Treat diagonal SSM/LRU as a reserve experiment and Mamba/RWKV as lower priority. Every proposed replacement starts from a new architecture trained on supplied data; none inherits the incumbent's measured accuracy automatically.

The selected callback takes approximately 26.1 microseconds per row, versus 28.6 for the original callback. Thus the existing no-slower-than-original gate leaves only about **2.5 microseconds** for an added branch. Prefer a replacement or jointly exported model over multiple Python/ORT calls. Small arithmetic counts are necessary evidence, but ORT dispatch, copies, activation kernels, and state movement can dominate. [Measured incumbent](../../artifacts/IMPLEMENTATION_REPORT.md)

## Analytical size and arithmetic

For two standard GRU layers of width H with 112 inputs and a two-output head, dense MACs per row are `3H(112+H)+6H²+2H`; FP32 parameters including both bias vectors are `MACs+12H+2`.

| Candidate core | Parameters | Dense MACs/row | Recurrent state |
| --- | ---: | ---: | ---: |
| Existing GRU128 | 192,258 | 190,720 | 256 floats |
| GRU96 | 116,546 | 115,392 | 192 floats |
| GRU64 | 59,266 | 58,496 | 128 floats |
| Two-layer minGRU64 | 22,914 | 22,656 | 128 floats |
| Two-layer minLSTM64 | 34,306 | 33,920 | 128 floats |

The minimal-cell counts assume only the paper's input-dependent projections and two-output head, without expansion MLPs, normalization, or residual projections. minGRU uses `2H(112+H)+2H` MACs; minLSTM uses `3H(112+H)+2H`. Elementwise gating is additional. These are calculations, not benchmark results. GRU64 occupies approximately 237 KB in raw FP32 weights; package size is unlikely to bind before accuracy or callback speed.

minGRU removes hidden-state dependence from its gates, making `h=a(x)⊙previous_h+b(x)` amenable to parallel training. This changes its expressive behavior; deleting GRU recurrent gate matrices is not a valid exact conversion. Reported dramatic training speedups were GPU experiments and do not predict one-row CPU speed. [minRNN paper](https://arxiv.org/html/2410.01201v2)

## Architecture tradeoffs

**Cached TCN.** Causal dilated convolutions offer bounded memory and parallel training. A custom `112→32` projection, three depthwise kernel-3 layers with dilations `{1,4,16}`, each followed by `32→32` mixing, and a two-output head costs approximately 7,008 MACs per row. State stores 1,344 floats, roughly 5.4 KB; receptive field is 43 rows. Each input is processed once and intermediate activations are cached. Standard window-based code must be rewritten as a step graph; repeatedly convolving all historical rows loses this advantage. The original TCN code is MIT, but this depthwise design is our proposed adaptation. [Author TCN repository](https://github.com/locuslab/TCN)

**Diagonal SSM/S4D and LRU.** These replace dense hidden-to-hidden multiplication with elementwise recurrence, leaving input/output projections as major costs. For a simple diagonal recurrence with input width I, state N, and output D, projection cost is approximately `IN+ND` MACs plus O(N) state operations; practical blocks add nonlinear channel mixing. S4D/LRU initialization and normalization are central to their reported performance, not optional embellishments. Complex states should be exported as paired real FP32 tensors with explicit arithmetic. Train/step discretization must match, and eigenvalues near unity warrant long-sequence stability tests. [S4/S4D author repository](https://github.com/state-spaces/s4), [LRU paper](https://proceedings.mlr.press/v202/orvieto23a.html)

**Mamba/RWKV.** Both support recurrence with state independent of elapsed sequence length. However, constant state does not imply cheaper inference than our tiny fused GRU. Current Mamba documentation includes a pure-Python installation without its CUDA extension; saying it universally requires CUDA would be outdated. It still does not demonstrate a fast CPU ONNX callback at our dimensions. RWKV's language-model results likewise do not establish a financial forecasting advantage. Large public language weights are irrelevant to our feature interface; importing financially pretrained variants raises the separate provenance constraint. [Current Mamba repository](https://github.com/state-spaces/mamba), [RWKV paper](https://aclanthology.org/2023.findings-emnlp.936/)

## Required integration work

`OnnxBackend` currently requires exactly `features`, `hidden_0`, and `hidden_1`, with three FP32 outputs and fixed states shaped `[1,1,H]`. Width is flexible, so existing GRU64/96 exports fit. TCN buffers, complex SSM components, or extra convolution states need a dedicated validated state schema/adapter; the wrapper is not already a generic architecture loader. Preserve double-buffered ownership, deterministic reset, one synchronous call, and finite predictions. [Runtime](../../connectome/runtime.py)

The native extractor separately requires exactly two GRU128 nodes and the original head layout. Leave it unchanged for architecture experiments. New exports need their own structural checks and parity tests. The existing C++ backend's failed speed/parity gates give no reason to port every challenger into C++. [Model inspection](../../connectome/model_io.py)

## Three bounded experiments and stop gates

1. **GRU96, then GRU64:** use existing student tooling, labels plus the incumbent teacher, two seeds, and matched data budgets. Do not slice GRU128 weights and call it a faithful warm start. Prefer the smallest student retaining enough WP to enable a genuinely complementary ensemble within callback limits.
2. **minGRU64:** implement the published recurrence independently, train from scratch, compare labels-only against teacher matching, and prove parallel-training versus sequential-export parity. minLSTM is a follow-up only if this establishes value. The official minRNN repository has a CC BY-NC-SA 4.0 license; do not assume unrestricted code reuse. [Official license](https://github.com/BorealisAI/minRNNs/blob/main/LICENSE)
3. **Cached TCN32:** train the 43-row model above; expand its receptive field only if error analysis supports longer context. Its context fits entirely within the 99 warm-up rows. Longer-memory models also start from zero at every sequence; they cannot assume history before row zero.

Screen on frozen development subsets, then full development. New architecture parity compares its own training and exported implementations, not unchanged predictions against GRU128. Require causality, output ownership, all 20,000 steps, and three complete Linux timing repetitions. Drop candidates that fail accuracy or speed gates; preserve the consumed confirmation ledger. No score gain is guaranteed.

## Coverage

Five Exa searches returned 25 result slots. Eight primary external sources support this memo; repository content, abstracts, and selected paper sections were inspected rather than every full paper. No pretrained weights were downloaded.
