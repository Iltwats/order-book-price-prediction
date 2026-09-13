# Round-two frozen-head method

Recorded 2026-09-13. This document describes the five declared experiments and their implementation. It contains no candidate scores or selection outcome. The executable declarations are in [head_cycle.py](../connectome/head_cycle.py); the immutable experiment records are in the local [round-two registry](../artifacts/round2/registry.json).

## Question and starting point

The experiment asks whether changing the readout objective can improve predictions while reusing every recurrent parameter in the selected two-layer GRU128. The source is [the selected model](../artifacts/selected/model.onnx), SHA256 `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45`. Its calibrated head, rather than the original unscaled head, is the reference for every correction.

All five arms use the same 1,024 training row groups recorded in the earlier affine1024 training report. Their final coefficients fold into the existing 128-to-2 linear head, preserving the original architecture and inference operation count. No new pretrained weights, recurrent fine-tuning, prediction filters, or external data enter this experiment.

## Causal feature extraction and weighted moments

The original ONNX graph supplies the two predictions and final-layer hidden states for each complete training sequence. It consumes every row, including steps 0–98, with zero initial states at sequence boundaries. Only steps 99 onward contribute to fitting. The encoder never receives labels as inputs. A complete-sequence export is an offline execution convenience; it must preserve causal callback behavior.

Two feature spaces are cached: the selected model's two raw predictions, and its 128-dimensional final-layer state. “Raw” here means before the metric's prediction clipping; the model's already-folded calibration remains present.

For training sequence `b`, target `k`, and eligible row `i`, define

\[
y_i=\operatorname{clip}(t_i,-2,2),\quad
w_i=|y_i|,\quad \widetilde w_i=w_i/\sum_jw_j.
\]

Targets are first represented as FP32, matching scoring inputs. Moment accumulation uses FP64. Let `μ_bk` and `m_bk` be the feature and target means under these normalized weights. Store

\[
C_{bk}=\sum_i\widetilde w_i(h_i-\mu_{bk})(h_i-\mu_{bk})^T,
\]

\[
v_{bk}=\sum_i\widetilde w_i(h_i-\mu_{bk})(y_i-m_{bk}),\qquad
q_{bk}=\sum_i\widetilde w_i(y_i-m_{bk})^2.
\]

The covariance is symmetrized to remove tiny numerical asymmetry. Every sequence retains separate statistics for each target; neither rows nor covariance matrices are pooled across sequences before computing correlation.

Training has no `is_scored` column, so all post-warm-up training rows define the declared fitting domain. A training block is excluded when either target lacks a positive or a negative value. A sign-eligible block remains included when a side has tiny variance; that side contributes zero correlation. Development uses the actual supplied scoring mask and official block eligibility rules. Training-domain correlation therefore differs from the official development metric even before considering prediction clipping. [Scoring reference](../METRIC.md)

Caching reduces repeated readout optimization to small matrices. The two-target 128-dimensional covariance matrices for 1,024 sequences occupy 268,435,456 bytes, with additional means, cross-vectors and metadata. A single sequence is processed at a time; no full-dataset hidden-state row cache is created. Extraction has `O(Nd²)` matrix work, while each cached optimization evaluation has `O(Bd²)` work. These complexity and storage calculations are not wall-time measurements.

## Prospective clarification: preserve a centered offset

The research memo initially proposed fixing the original raw intercept. Before candidate outcomes, implementation review replaced that convention with a **fixed offset in training-centered coordinates**, uniformly for all five arms. Holding a raw intercept fixed makes coefficient changes shift prediction means when features have nonzero means, and depends on the arbitrary choice of feature origin.

For each target, set a single constant center

\[
\mu_k=\frac1B\sum_b\mu_{bk}.
\]

With selected coefficients `β₀k,a₀k`, fit `β_k` and derive

\[
a_k=a_{0k}+(\beta_{0k}-\beta_k)^T\mu_k.
\]

Thus the prediction correction is `(β_k−β₀k)ᵀ(h−μ_k)`. It preserves the selected prediction's equal-sequence, target-weighted **training** mean. Identity coefficients reproduce the source. Translating feature coordinates leaves the represented correction unchanged.

The centers use training data only and disappear into one exported bias per target. There is no per-sequence inference centering, online target statistic, or validation-derived offset. This policy does not promise to preserve development prediction means or clipping patterns, and it does not optimize the intercept freely.

For two-output arms, `β₀` is the identity matrix and `a₀` is zero in selected-output coordinates. The resulting transform is then composed with the selected head. A report's transform bias consequently differs from its final ONNX head bias; both coordinate meaning and exported bias are recorded.

## Objectives, regularization and solver

For unbounded linear predictions, sequence/target correlation follows exactly from the moments:

\[
\rho_{bk}(\beta_k)=
\frac{\beta_k^Tv_{bk}}{\sqrt{(\beta_k^TC_{bk}\beta_k)q_{bk}}}.
\]

The correlation objective averages `1−ρ` equally over sequences and targets. A side with prediction or target variance at most `10⁻¹⁶` contributes zero correlation, corresponding to the scorer's `10⁻⁸` standard-deviation threshold. Normalized moments make this threshold independent of sequence weight totals.

The MSE control uses the same centered parameterization and computes

\[
\mathrm{MSE}_{bk}=\beta_k^TC_{bk}\beta_k-2\beta_k^Tv_{bk}+q_{bk}
+(\beta_k^T\mu_{bk}+a_k-m_{bk})^2.
\]

Both objectives add the selected-coefficient anchor

\[
\lambda\,\frac12\sum_{k=0}^1
\frac{\|\beta_k-\beta_{0k}\|_2^2}{\max(\|\beta_{0k}\|_2^2,10^{-12})}.
\]

This relative penalty makes coefficient displacement interpretable across the two feature widths and discourages scale drift. It is not invariant to arbitrary feature rescaling. The same numerical penalty in MSE and correlation objectives also does not imply identical effective regularization strength because their data-loss units differ; the comparison is a bounded control, not definitive attribution of any change to loss alone.

Optimization uses FP64, one Torch compute thread, L-BFGS with strong-Wolfe line search, at most 200 optimizer iterations, and a requested evaluation budget of 400 closures. It retains the lowest finite penalized objective encountered, including the initial selected solution. Training reports record actual iterations and closure evaluations, anchor size, prediction variance, and unclipped training correlation. Better training loss does not constitute a selection result.

The first actual hidden-state fit exposed a storage-layout bug before producing a model or reserving a screen: the cached transposed head was Fortran-contiguous, but L-BFGS flattens gradients with `view`. The optimized parameter now receives one contiguous copy. A regression test requires identical coefficients, bias and complete optimization reports for C- and Fortran-order inputs. The previous code and registry are archived under `artifacts/round2/source_v1/` and `registry_before_layout_fix.json`. The guarded recovery script `scripts/amend_round2_layout.py` records the reason and updates only the three unfitted arms to the corrected source hash. The two completed affine screens retain their original code and results. No fitting settings, training groups, trial caps or validation membership changed.

## Five declared arms

| Candidate | Features | Data loss | Anchor penalty |
| --- | --- | --- | ---: |
| `affine_mse` | Two selected outputs | Weighted MSE | `1e−2` |
| `affine_rho` | Two selected outputs | `1−ρ` | `1e−2` |
| `head_rho_p1e-4` | 128 final hidden states | `1−ρ` | `1e−4` |
| `head_rho_p1e-3` | 128 final hidden states | `1−ρ` | `1e−3` |
| `head_rho_p1e-2` | 128 final hidden states | `1−ρ` | `1e−2` |

The declared seed is `20260912`; the deterministic readout solver does not run stochastic epochs. All arms use ONNX Runtime I/O binding, strict inputs, disabled affinity and empty postprocessing. There is no implicit scale or EMA sweep.

## Selection and verification limits

Clipping predictions to `[-2,2]` is nonlinear. These moments cannot recover exact clipped WP for arbitrary coefficients; correlation optimization is an **unclipped-prediction surrogate**. Final FP32 export rounding also means optimized FP64 readout values are not themselves the deployed predictions.

Each candidate therefore needs callback/bulk parity followed by exact clipped evaluation on the frozen 256-sequence development screen. Only prospectively reserved full-development evaluations may follow. The registry caps this round at 12 screen and four full-development reservations, counts failures, freezes source/configuration/model provenance, and permanently excludes confirmation. [Registry procedure](round2_registry.md)

Moment files are hashed; source model, ordered groups, feature contract and training fingerprint are checked when loading. Training fingerprints are fast size/mtime/edge-content change detectors, not complete dataset cryptographic hashes. Extraction also checks for dataset changes during its pass. Synthetic integration tests exercise excluded-block packing, tampering, invalid targets and both export coordinate systems. Every recurrent initializer must remain byte-identical, and meaningful score improvement and Linux callback performance still require measured evidence before promotion. [Math tests](../tests/test_correlation.py), [integration tests](../tests/test_correlation_integration.py)
