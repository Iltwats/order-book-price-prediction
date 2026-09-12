# Approach 05: train the readout for sequence correlation

Research date: 2026-09-12. **The next head experiment should change the objective before increasing model capacity.** Quarter-scale calibration explains 98.39% of the observed development improvement over the original baseline. The selected affine head adds only +0.000693146 WP beyond that control, and larger weighted-MSE heads lost. This supports testing metric alignment, not assuming useful recurrent features are exhausted. Start every experiment from the selected head and preserve its original ONNX encoder. [Implementation evidence](../../artifacts/IMPLEMENTATION_REPORT.md), [analytical checks](analysis_checks.json)

## What the current objective misses

The existing trainer already clips targets, uses their absolute values as weights, normalizes by each sequence's target-specific total weight, and accumulates minibatch gradients before one sequence update. It fits a correction around the supplied source artifact. Thus “add sequence weighting” alone is not new. Its remaining mismatch is squared-error regression versus **equal-sequence, equal-target correlation after clipping predictions**. Starting from the original uncalibrated model would also discard a strong current solution. [Training implementation](../../connectome/training.py)

For unbounded predictions, weighted correlation ignores positive scale and constant offset; MSE penalizes both. Official clipping makes scale and offset matter again. This explains why shrinking the original predictions can change WP substantially without improving the recurrent representation. GAR research derives a linear-time negative-Pearson loss from normalized pairwise differences, supporting correlation-oriented training without quadratic row-pair enumeration. Its authors explicitly study clean regression settings, however; its benchmark gains do not establish financial-series gains here. [Official metric](../../METRIC.md), [GAR, ICML 2025](https://proceedings.mlr.press/v267/zhu25x.html), [paper derivation](https://arxiv.org/html/2402.06104v6)

## A reusable sufficient-statistics surrogate

For eligible training sequence `b`, target `k`, let `h_i` be the frozen 128-dimensional final GRU state, `y_i = clip(t_i,-2,2)`, and `w_i=|y_i|`. Use post-warm-up rows. Compute weighted means, center `H` and `y`, and retain:

\[
C_{bk}=H_c^T\operatorname{diag}(w)H_c,\quad
v_{bk}=H_c^T\operatorname{diag}(w)y_c,\quad
q_{bk}=y_c^T\operatorname{diag}(w)y_c.
\]

For the linear readout `p_i = h_iᵀβ_k + a_k`, its **unclipped-prediction** correlation is

\[
\rho_{bk}(\beta_k)=\frac{\beta_k^Tv_{bk}}
{\sqrt{(\beta_k^TC_{bk}\beta_k)q_{bk}}}.
\]

The common weight normalization cancels. Its gradient is

\[
\nabla_\beta\rho=\frac{v}{\sqrt{(\beta^TC\beta)q}}
-\rho\frac{C\beta}{\beta^TC\beta}.
\]

Optimize the mean of `1−ρ` separately per sequence and target; **do not pool covariance matrices before taking correlation**, because correlation of pooled data is a different objective. The selected intercept cancels from this surrogate, so freeze it initially. Parameterize `β=β_selected+δ`; regularize corrections toward selected coefficients, preferably using training-state variance to make the penalty insensitive to hidden-feature units. A relative coefficient penalty is a simpler first control. Preserve scale near the selected head to avoid flat scale directions and tiny prediction variance.

This is a task-specific algebraic derivation. A 20,000-row synthetic check matches direct computation within `4.51e−17`; it does not prove accuracy improvements. Official prediction clipping is nonlinear and **invalidates this shortcut**. Exact clipped WP must select exported candidates. [Check record](analysis_checks.json)

## Memory, computation and causality

Full two-target FP64 `C` matrices for all 10,607 training sequences require **2,780,561,408 bytes**; vectors add 21.7 MB. Symmetric packing reduces matrices to 1.40 GB. A 1,024-sequence pilot needs 268 MB of dense matrices. Compare fairly: storing all final-layer FP32 states costs 108.6 GB; both layers cost 217.2 GB. Expanding covariance features to both states would quadruple matrices to 11.12 GB, so defer it.

Extract states one complete sequence at a time, including every warm-up row, then accumulate statistics in FP64. One final-state sequence is only 10.24 MB. Persist sequence statistics with encoder/data/feature/mask hashes and exclude them from Git. Statistics construction is `O(Nd²)` and may be expensive; cached optimization is `O(Bd²)` per iteration. These are memory estimates, not measured training times. Existing affine timing cannot predict covariance construction speed.

Training has no `is_scored` column: use all post-warm-up rows as the declared training surrogate. Determine sequence eligibility from both targets having positive and negative values on that training selection. Development uses the actual supplied scoring mask and official eligibility rules. Never infer an unavailable test mask. No target, future row, or sequence ID enters the encoder; whole-sequence supervised fitting is allowed because deployed predictions remain causal. Handle zero weights/variance explicitly and report skipped training blocks. [Data contract](../data_overview.md), [metric thresholds](../../METRIC.md)

## Three bounded experiments

1. **Two-output objective control.** On the same 1,024 training sequences, use selected predictions as two features. Fit one identity-anchored weighted-MSE affine correction and one correlation-surrogate correction with penalty `1e−2`. Keep selected predictions as the no-fit control. This isolates the loss change cheaply; do not mix it with more data, a larger encoder or another EMA search.
2. **128-state correlation readout.** Reuse those sequences and frozen states. Initialize at the selected head; test relative correction penalties `{1e−4,1e−3,1e−2}`, at most 200 deterministic optimizer iterations each, checking finite variance. Screen all three on the existing 256 development sequences. Advance only the strongest exact-WP candidate to full development.
3. **Clipped-correlation refinement.** Only if experiment 2 helps, take its winner and optimize actual sequence correlation with hard-clipped predictions, using full post-warm-up rows from 256 training sequences for at most two passes. Compare pure clipped correlation with a small `0.05` normalized prediction-anchor penalty; preserve the same regularization. This keeps trial count bounded.

For experiment 3, random row subsets estimate a nonlinear ratio and are biased for full-sequence correlation. Prefer complete-sequence prediction/target vectors, then accumulate head gradients in chunks. Averaging 4,096-row correlations is not the full-sequence loss. Hard clipping gives zero derivative beyond saturation; monitor saturation fraction, stay near the calibrated head, and reject stalled/nonfinite fits. A smooth clipping approximation would be another surrogate, not exact WP.

Use identity scale during screening; permit one predeclared final scale check `{0.9,1,1.1}` independently per target for the single winning head. Stop if exact development WP does not improve over selected, paired sequence differences reverse, or gains vanish on a second training seed. Preserve the already-used confirmation ledger. Fold winning coefficients into the existing head, verify recurrent-state parity and clipping-aware WP, and require three Linux callback repetitions with no material slowdown. No experiment here needs new inference operations.

Search accounting: **two Exa searches, ten returned results**, followed by two primary-source fetches. External literature informs objective design only; reported competition values come from local evidence. No training, weight edits or confirmation evaluations were performed.
