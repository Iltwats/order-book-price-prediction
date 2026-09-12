# Approach 4: compact limit-order-book architectures

Verified 2026-09-12. Research only; no external market weights downloaded or training run. Recommendations concern architectures trained on the supplied Connectome data.

## Decision

Prioritize a small causal feature/MLP correction, then a compact temporal-convolution student, then a low-rank bilinear challenger. Published LOB classification results justify experiments, not predicted improvements in our two-target WP. Current development WP is 0.629255535; the supplied GRU's recurrent weights remain unchanged. Its existing hidden-state head corrections lost screening, so the next correction should introduce information from explicit short-term dynamics rather than simply repeat a wider hidden-state head. [Local evidence](../../artifacts/IMPLEMENTATION_REPORT.md)

Our 112 features are anonymized stable components: index 0 is not a known best quote, indices are not ordered depth, and physical price-volume pairing cannot be inferred. Consequently, conventional spread, microprice, depth imbalance, and price-conditioned order-flow formulas are not automatically valid. Both targets are anonymous continuous indicators for instrument i0; neither up/down labels nor prior target values exist at inference. Preserve supplied feature order, use causal inputs, reset every sequence, and supervise only after warm-up. [Data specification](../data_overview.md), [official local metric](../../METRIC.md)

## What transfers from the literature

**DeepLOB** combines convolutions that exploit ordered book structure with temporal modeling through an LSTM. Its author implementation demonstrates FI-2010 data. Transfer learned feature extraction and multiscale temporal processing; replace spatial kernels based on adjacent physical levels with dense or documented-group projections. Do not assume FI-2010 weights fit our input semantics. [Paper](https://arxiv.org/abs/1808.03668), [author code](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)

**DeepOF/order-flow learning** found stationary order-flow representations more effective than raw books across 115 Nasdaq stocks. The transferable hypothesis is that changes and local deviations may be easier to learn than raw states. Here, same-column differences and deviations from causal averages are observable; calling them genuine exchange order flow would overstate their meaning. [Original study](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413)

**TABL and BiN-C(TABL)** separate feature and time mixing using bilinear projections, temporal attention, and input normalization. These have a stronger CPU rationale than a large transformer. BiN uses sample statistics across feature and temporal axes: a trailing window is causal when predicting its final row, but full-sequence normalization leaks future information. Avoid altering the pretrained GRU's original preprocessing; test normalization in a separately trained branch. [TABL](https://arxiv.org/abs/1712.00975), [BiN](https://arxiv.org/abs/2003.00598), [low-rank TABL](https://arxiv.org/abs/2107.06995)

**TransLOB/TLOB** provide convolution-plus-attention and dual feature/time attention designs. Their published tasks use three-class trend labels. More relevant initially, TLOB's authors also demonstrate a strong MLPLOB baseline, challenging the need for complex attention. Recomputed sliding-window attention is poorly matched to tens of millions of tiny callbacks unless carefully cached; a causal mask alone does not remove its cost. [TransLOB author code](https://github.com/jwallbridge/translob), [TLOB author code](https://github.com/LeonardoBerti00/TLOB)

**LOBFrame/HLOB** support comparisons across stocks and microstructure regimes; HLOB learns dependency structure rather than relying solely on a simple chain. This motivates train-only feature-covariance analysis, but deploying a graph model is lower priority than testing whether short-term features add residual signal. Their trading-oriented evaluation should not replace our supplied masked, clipped, equal-sequence WP. [LOBFrame paper](https://arxiv.org/abs/2403.09267), [HLOB paper](https://arxiv.org/html/2405.18938)

## Code and checkpoint provenance

| Source | Verified source status | Practical use |
| --- | --- | --- |
| [TLOB](https://github.com/LeonardoBerti00/TLOB) | Repository identifies MIT license | Useful MLP/bilinear implementation reference; verify copied files' notices. |
| [deepOBs](https://github.com/lorenzolucchese/deepOBs) | BSD-3-Clause; DeepLOB/DeepOF/deepVOL research pipeline | Useful model/reference code, trained anew locally. |
| [LOBFrame](https://github.com/FinancialComputingUCL/LOBFrame) | Explicit CC BY-NC-ND 4.0 | Treat as research reference; do not assume permission to distribute a modified submission implementation. |
| Original DeepLOB, TransLOB, [TABL](https://github.com/viebboy/TABL) | No explicit license identified in the extracted repository pages | License unresolved; independently implement paper ideas instead of copying code. |

No checkpoint from these financial studies is established as eligible for Connectome. Public code availability and weight eligibility are separate; the rules' market-data exclusion still applies. [Provenance memo](01_rules_provenance.md)

## Three ranked experiments

**1. Zero-initialized dynamics correction.** Keep the deployed GRU intact. Feed `[x, x−previous_x, x−EMA8(x)]`, 336 values, into a `336→32→2` ReLU MLP. Initialize its final layer to zero; fit against residual error with sequence-normalized weighted MSE and regularization. Initialize causal averages from the first row; use training-only scaling for this branch. Export into the same ONNX callback. Analytically this adds 10,816 multiply-accumulates per row and approximately 44 KB of FP32 weights, about 5.7% of the GRU's matrix arithmetic. It adds new short-term inputs, unlike the previous losing hidden-state-only heads. Identity remains the control.

**2. Cached causal temporal-convolution student.** Project `112→32`, then apply three depthwise temporal kernel-3 blocks with dilations `{1,4,16}`, each followed by `32→32` pointwise mixing, and a two-output head. The receptive field spans 43 rows. Train on labels, optionally matching the current eligible teacher; compare standalone performance before considering an ensemble. Analytical cost is approximately 7,008 matrix/convolution MACs per row, with about 5.4 KB of dilation state. Reuse each layer's past activations; reevaluating every complete window would defeat this estimate. This adapts DeepLOB's temporal idea without imposing physical depth adjacency.

**3. Small bilinear window model.** Cache a train-normalized `112→16` projection for the last 32 rows; mix time `32→8`, then predict through `128→16→2`. Start without attention, then add rank-4 temporal attention. Analytical cost is roughly 7,968 MACs per row before attention, or 12,064 with the described low-rank attention, excluding softmax and normalization. Test window-based BiN separately: changing normalization of historical rows can invalidate projection caching and increase work substantially. This is a new compact adaptation, not a claim to reproduce published C(TABL).

All compute figures are analytical proposals, not measured latency. Evaluate newly fitted components using sequence-disjoint training folds and then full development, and require three complete-callback Linux repetitions. Such training folds are untouched only by our new fitting: the supplied encoder's pretraining lineage is unknown and may include those rows. Preserve the existing confirmation ledger; do not reuse its 473 sequences as a new holdout. None of these experiments has a guaranteed score gain.

## Coverage

Four Exa searches returned **40 result slots / 39 distinct URLs**. Primary repository pages, paper abstracts, and selected paper sections were fetched; this was not a full-paper read of every result. Third-party summaries were excluded from the substantive evidence. No runtime files, datasets, or releases changed.
