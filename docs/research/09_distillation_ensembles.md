# Approach 09: complementary teachers and compact students

Research date: 2026-09-12. **Seek complementary prediction errors, then compress useful combinations.** Our selected GRU reaches development WP 0.629255535 with 38,291 callbacks/second; the original achieves 34,962. This leaves only about 2.5 microseconds per row under the existing no-slower-than-original gate. An ensemble is worthwhile only when it improves exact WP and fits that measured budget, or when it can teach a smaller deployable model. [Current evidence](../../artifacts/IMPLEMENTATION_REPORT.md)

Distillation can transfer an expensive ensemble into one network; Born-Again Networks show that identical-capacity students sometimes improve on teachers. Neither establishes gains for our regression task. Classification-temperature recipes do not directly apply to continuous outputs. [Original distillation paper](https://arxiv.org/abs/1503.02531), [Born-Again Networks](https://proceedings.mlr.press/v80/furlanello18a.html)

## Diversity and evaluation before training an ensemble

Generate candidate predictions on sequence-disjoint folds of **training data** for learned stacking. Fit new head corrections or teacher adapters on two folds and predict the third; rotate three times. Every sequence remains intact. Analyze weighted, centered residual correlations and paired per-sequence WP differences for each target. Low error correlation is only a diagnostic: noise can be diverse without useful signal. Actual blended clipped WP decides.

The supplied encoder's original training lineage is not known at individual-sequence granularity. Cross-fitting newly learned heads does not make that fixed encoder globally out-of-fold; document this limitation. Development sequences remain outside supervised fitting. Do not fit stacking coefficients on confirmation or reopen its consumed ledger. [Evaluation contract](../../METRIC.md), [head-objective proposal](05_metric_head.md)

For two promising predictors, first test a nonnegative target-specific blend `p_k=(1−α_k)p_incumbent,k+α_k p_candidate,k`, with `α_k∈{0,0.1,0.25,0.5,1}`. This is 25 exact metric combinations and contains both endpoints. Fix calibration before blending. Learned stacking adds selection freedom; use it only if this grid demonstrates complementary signal, and fit its constrained coefficients exclusively from cross-fitted training predictions with shrinkage toward the incumbent.

If predictors share exactly the same hidden states and have linear heads, constant blends fold into one head. Different recurrent encoders, nonlinear corrections or independent filters prevent this reduction. **Clipping before blending does not commute with blending before clipping**: predictions 4 and 0 averaged first produce 2, whereas clipping each first gives an average of 1. Choose one explicit pipeline and preserve it during scoring, export and distillation. The foldable version blends raw affine predictions before the scorer clips.

## CPU accounting

| Recurrent deployment | Dense MACs/row | Interpretation |
| --- | ---: | --- |
| One GRU128 | 190,720 | Measured incumbent |
| Two GRU64 | 116,992 | 61% of incumbent dense arithmetic |
| GRU96 plus GRU64 | 173,888 | 91% of incumbent dense arithmetic |
| Two GRU96 | 230,784 | Exceeds incumbent dense arithmetic |

One MAC is conventionally two floating-point operations; these counts omit nonlinearities, elementwise state operations and dispatch. Exporting two branches into one ONNX call can reduce Python overhead but requires independent states and structural validation. A lower MAC count does not guarantee faster callbacks. Measure all 20,000 rows, startup and peak memory in three Linux one-CPU repetitions. [Architecture calculations](07_streaming_architectures.md)

Do not average independently trained GRU weights: hidden-unit permutations can preserve functions while making coordinate-wise averages meaningless. SWA studies nearby checkpoints along one training trajectory; that remains a separate, lower-priority experiment. [SWA paper](https://arxiv.org/abs/1803.05407)

## Teacher validity and matching objective

Use the **selected calibrated GRU**, not the original unscaled model, as the first teacher. The current student tool reconstructs its teacher in Torch and uses unweighted raw prediction MSE; the label loss is sequence-normalized weighted MSE. Exact-incumbent distillation therefore needs verified original-ORT teacher predictions or an explicitly accepted reconstruction criterion. Existing Torch long-sequence parity is unresolved. This is an implementation requirement, not an already available guarantee. [Training code](../../connectome/training.py), [training notes](../training.md)

A useful regression objective is sequence/target mean weighted label loss plus `λ` times weighted teacher matching, using `w=|clip(y,-2,2)|` and full-sequence weight denominators. Define teacher scale constants from training only and record them. Compare matching raw calibrated teacher predictions against matching clipped teacher values; keep student outputs unbounded in the latter MSE to avoid saturation's zero gradients. Select by official clipped WP, not teacher agreement. Teacher matching on near-zero-label rows is a separate optional regularizer, not implicit metric weighting.

An external teacher must first learn a features-only adapter to our anonymous targets. Historical `t0/t1` and future covariates never exist at inference and must never enter teacher inputs. Use trailing observed-feature windows, with isolated sequence boundaries. `autogluon/chronos-2-synth` explicitly declares synthetic-only pretraining and is a conditional offline candidate; eligible ETTh1 PatchTSMixer is cheaper to investigate. TTM r1/r2 market-data weights remain unacceptable teachers even if omitted from the ZIP. [Synthetic card](https://huggingface.co/autogluon/chronos-2-synth), [provenance research](02_small_hf.md), [foundation research](03_foundation_hf.md)

## Three bounded experiments

1. **Foldable head blend:** combine incumbent and the strongest correlation-trained readout from approach 5. Run the 25-pair grid on the 256-sequence development screen; validate one winner on full development. Stop if neither a stable paired gain nor a complementary target improvement appears. No new backbone training is needed.
2. **GRU96/64 distillation:** on matched 1,024 training sequences, run GRU96 labels-only, raw-teacher `λ=0.1`, and clipped-teacher `λ=0.1`, at most three epochs with early stopping. Repeat the best recipe once at width 64 and repeat the winning seed. Test the two-small-student blend only if both are individually competitive. A deployable finalist must beat the incumbent; a faster but weaker student is research material, not automatic promotion.
3. **Diverse offline teacher:** conditionally adapt one clean external checkpoint on a bounded training subset. Benchmark extraction on four complete sequences before expansion. Stop if its learned readout loses to a matched raw-window control or adds no blend gain. Distill only a teacher/blend that improves development WP; keep the large checkpoint outside the submission and preserve provenance.

All students update state during warm-up, reset between sequences, and detach training state at chunk boundaries. Verify each new architecture against its own exported inference implementation. Preserve confirmation guards and require finite, owned outputs and causal predictions before release.

Search accounting: **two Exa searches, ten returned results**, plus three primary-source fetches. This memo proposes experiments; no models were trained, weights changed or confirmation scores generated.
