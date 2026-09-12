# Approach 8: causal features and data-centered improvement

Verified 2026-09-12. These are bounded hypotheses, not measured gains. No new dataset profiling, training, or runtime changes were performed.

## Starting point

Our supplied 112 columns describe two instruments, but stable component indices are not ordered depth. Neither best quotes nor exchange timestamps are identified; the two output indicators are anonymous. Use event lags measured in rows, never invented milliseconds. Preserve every raw input and the supplied GRU's preprocessing. Additional features belong in a separately fitted residual branch or a new student, not silently substituted into pretrained inputs. Prior output EMA and hidden-state-only heads lost screening; the proposals below add different information. [Data overview](../data_overview.md), [current results](../../artifacts/IMPLEMENTATION_REPORT.md)

The literature supports testing stationary representations and cross-instrument histories, but not assuming they work here. Deep order-flow research finds transformations useful on real order books, while cross-impact research distinguishes contemporaneous fit from predictive lagged information. Our same-column differences are measurable; exchange order-flow imbalance formulas are not justified by anonymous indices. [Stationary-input study](https://onlinelibrary.wiley.com/doi/10.1111/mafi.12413), [cross-impact study](https://arxiv.org/html/2112.13213)

## Four small ablations

### 1. Learned cross-instrument lag branch

Learn separate projections `u0_t=P0*x0_t`, `u1_t=P1*x1_t`, each `52→8`, with training-only normalization. Build a 72-dimensional vector from both projected instruments at lags `{0,1,4,16}` plus the eight additional raw columns. Fit a zero-initialized `72→16→2` residual MLP. First compare current-only against lagged inputs; then optionally add elementwise projected interactions `u0_t⊙u1_(t−lag)`.

Separate projections avoid assuming that identically numbered columns across instruments share a physical unit or scale. The original GRU already sees both instruments, so the hypothesis is easier access to short lead-lag patterns, not newly discovered data. Analytical cost is approximately 2,016 MACs per row before optional interactions, plus ring-buffer operations. Store at most 17 projected rows and reset at sequence boundaries. This is small enough to benchmark as one fused graph, but not proven to fit the roughly 2.5-microsecond overhead budget.

### 2. Normalized innovations and group summaries

On 16 projected channels, maintain causal means `m_t^h=m_(t−1)^h+α_h(u_t−m_(t−1)^h)`, where `α_h=1−2^(−1/h)` and `h∈{2,8,32}`. Compare fast-minus-slow means and normalized innovations `(u_t−m_(t−1)^8)/sqrt(v_(t−1)+epsilon)` against plain deltas. Update variance causally after computing the feature; initialize from training statistics and clamp small negative numerical variance to zero. This examines intensity and timescale disagreement, rather than smoothing predictions.

Train-fixed scaling can be folded into learned input projections. Adaptive statistics add O(16×3) updates per row; actual cost includes square roots and export overhead. Start with the linear fast-minus-slow variant, adding variance normalization only if useful.

For documented price-like, volume-like, and trade groups, mean and dispersion of **per-column standardized values** are valid numerical summaries, not physical liquidity totals. Retain individual channels because group averaging discards stable identities. Do not sort price-like columns into presumed depths, divide anonymous volumes by their sum, or assume permutation invariance. Within-column ranks against training quantiles are possible, but add lookup cost and may discard amplitude information; defer them unless diagnostics support the hypothesis.

RevIN removes and restores instance statistics, and its official implementation requires matching input/output feature counts. Our 112 inputs and two unrelated targets have no justified reverse mapping. Likewise, full-sequence BiN/RevIN statistics would leak future rows; a strictly trailing window is required. These restrictions favor simple branch-only normalization. [RevIN author implementation](https://github.com/ts-kim/RevIN), [BiN paper](https://arxiv.org/abs/2003.00598)

### 3. Training-only future-feature auxiliary objective

Train a small causal student to predict the supplied targets while also predicting future projected feature changes:

`L = L_supervised + λ * mean_(τ∈{1,4,16}) ||Aτ*h_t − (u_(t+τ)−u_t)||²`.

Future features appear only as training labels, never encoder inputs or recurrent state. Exclude horizons crossing a sequence end. Fit the projection on the training split only, and remove auxiliary heads from the exported graph, making their inference overhead zero. Compare `λ=0` with one small auxiliary weight before broad tuning. This may encourage useful dynamics or distract from the real targets; both outcomes are plausible.

TS2Vec demonstrates hierarchical representation learning and exposes causal sliding inference. It does not justify calling arbitrary full-window representations causal. TS-TCC's official augmentation code includes temporal segment permutation and random scaling; these can destroy lead-lag and amplitude meaning here. Start without those augmentations. If adding masking, preserve time order, provide an explicit mask to the training encoder, and verify the unmasked deployment path. [TS2Vec author code](https://github.com/zhihanyue/ts2vec), [TS-TCC paper](https://arxiv.org/abs/2106.14112), [actual augmentation code](https://github.com/emadeldeen24/TS-TCC/blob/main/dataloader/augmentations.py)

### 4. Tail coverage without changing the metric

Sample sequences uniformly, then stratify supervised rows by target sign and magnitude of **clipped** targets. Keep recurrent updates on every row. With nonuniform row sampling probability q, weight each retained loss by `1/q` and the original full-sequence weight denominator; otherwise oversampling changes the intended objective. Compare against uniform row selection at an equal supervised-row budget. Large raw targets beyond ±2 have the same clipped magnitude, so unlimited emphasis on extreme raw tails is inconsistent with the scorer. Existing training already uses clipped targets and sequence denominators; change sampling, not those safeguards. [Training code](../../connectome/training.py), [metric](../../METRIC.md)

## Acceptance and coverage

No missingness, outlier frequency, rank stability, or instrument lag has been measured in this research round. Do not add imputations or clipping based on invented statistics. Test future-input perturbations, sequence resets, streaming/offline feature equality, warm-up initialization, and independent output ownership. Refit every transform inside training folds and preserve the already consumed confirmation ledger.

Four Exa searches returned **20 result slots**. Primary abstracts, repository documentation, and selected code were inspected; these counts do not imply 20 full-paper reads. All latency figures above are analytical. Promote only after measured WP improvement and complete-callback Linux timing.
