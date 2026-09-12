# Approach 10: adversarial review and round-two validation

Research date: 2026-09-12. The research supports three useful next experiments, but **no available local split is now a pristine independent confirmation set**. Ten research approaches should feed one controlled experiment registry, not ten unrestricted searches against the same development data.

## Corrections that matter

The [completed decision](../../artifacts/confirmation_decision.json) records the current candidate's +0.041328119 confirmation WP over the original model, with interval [0.037460151, 0.045373192]. That is historical evidence for the already frozen release. All 1,873 validation sequences have now been used either for development or confirmation. Renaming the 473 confirmation sequences, reshuffling them, changing a seed or creating another manifest cannot restore independence.

[selection.py](../../connectome/selection.py) correctly describes its ledger as a workflow safeguard, not a statistical guarantee or data access control. Its ledger is associated with the manifest path; do not create a new filename to evade it. Its confirmation comparator also explicitly requires the original unmodified baseline. Round two must compare against the **current calibrated incumbent**, development WP 0.629255535. Preserve the old function/receipt as historical machinery instead of silently repurposing its meaning.

The [LOB memo](04_lob_architectures.md) originally referred to “untouched training-derived validation folds.” Such folds can be untouched by our new fitting, but the supplied GRU's training lineage is not established here; it may already have seen those training rows. Call these **reserved round-two fitting folds**, not independent model holdouts. Out-of-fold predictions remain useful for stacking, but do not undo pretrained exposure.

[Dwork and colleagues](https://arxiv.org/abs/1506.02629) show why adaptive data reuse needs special controls. Their reusable-holdout algorithms deliberately constrain information release; a plain bootstrap or new random split does not implement those guarantees. [Cawley and Talbot](https://www.jmlr.org/papers/v11/cawley10a.html) show that optimizing a noisy selection criterion can produce substantial selection bias. Nested folds help estimate a fixed fitting procedure when properly isolated; retroactively nesting experiments does not erase analyst decisions already influenced by all validation results.

## A bounded prospective protocol

**Before new outcomes:** record the incumbent model/config/runtime hashes, dataset and feature-order fingerprints, exact groups, training budgets, optimizer settings, early-stop rules and comparison metric. Register failed runs and exported checkpoints as well as winners. Keep confirmation files and its access ledger unchanged. A round-two registry records new experiments; it does not declare new independent data.

Use the existing **32/256/1,400** hierarchy. The 32 complete sequences establish correctness, including all 20,000 recurrent updates; they do not justify a score claim. The 256 screen selects promising experiments. Evaluate at most four frozen finalists on full development. A defensible initial cap is twelve non-control screened configurations across the three priority families, then stop and reassess. Include postprocessing variants and epochs in the trial ledger; calling each one “the same model” does not remove selection.

Fit learned parameters, normalization constants, channel selection and mixing weights on training sequences. For ensembles, generate sequence-disjoint training out-of-fold predictions for newly fitted components. Alternatively use genuinely nested inner/outer development evaluation for the complete new fitting procedure, while retaining its already-inspected-data caveat. Do not estimate mixture weights on development and present their in-sample development gain as an independent test.

Predeclare the strongest feasible exact-development-WP candidate as the local selection rule, with the incumbent winning exact ties. Repeat a stochastic winner with a second training seed and report both results; do not select the better seed and discard the other. For deterministic sufficient-statistic solvers, rerunning identical arithmetic is not a second statistical replicate: use a predeclared training-sequence subsampling or regularization-stability check instead.

Only fresh eligible data with established separation can support a new independent holdout claim. A frozen platform/private evaluation is the practical candidate when available; repeated public leaderboard feedback itself becomes another adaptive selection source. Continuing local development is valid, provided results are labelled development evidence.

## Metric and uncertainty requirements

Use the exact [scoring reference](../../METRIC.md): clip targets and predictions to [-2,2], weight each target by its absolute clipped value, apply `need_prediction AND is_scored`, compute each side's weighted Pearson, then equally average sides and eligible sequences. Eligibility requires positive and negative target values on selected rows for **both** sides; if either fails, exclude the whole sequence. Keep the same eligible domain across comparisons. Invalid/nonfinite predictions are failures, not removable observations. Constant predictions score zero. Training has no `is_scored`; do not invent a test-mask proxy and mistake it for the official objective.

For every finalist, pair candidate and incumbent by identical sequence IDs. Bootstrap **whole sequence differences**, keeping the two targets together, with 10,000 resamples and a fixed reported seed. Report both targets, mean/median difference, win fraction and negative-tail behavior. More resamples reduce Monte Carlo noise; they do not increase the number of independent sequences or correct adaptive selection.

The existing [bootstrap implementation](../../connectome/calibration.py) explicitly assumes sufficiently independent sequences and disclaims search-bias correction. Treat its 95% interval as descriptive conditional uncertainty, not automatically a post-selection 95% guarantee. Rows within a sequence are highly dependent; resampling millions of rows independently would overstate precision. If cross-sequence day/instrument clustering were known, cluster-level sensitivity analysis would help. Those labels/order are not supplied, and shuffled `seq_ix` is not chronology. Do not invent temporal blocks. [Politis's review](https://doi.org/10.1214/ss/1063994977) explains why dependence changes bootstrap validity.

## Collective critique and ranked action

1. **Improve the frozen readout objective.** It addresses demonstrated calibration/metric mismatch while preserving recurrent weights and callback arithmetic. Its covariance shortcut represents unclipped correlation, not exact clipped WP; expensive O(rows × hidden-width²) extraction needs timing before a full-cache build.
2. **Test a tiny causal feature correction.** Same-column changes add observable information absent from failed hidden-state-only head pilots. Stable feature indices are not physical order-book depth. Benchmark the proposed branch immediately: approximately 2.5 microseconds of measured callback allowance can disappear in dispatch/copies despite small MAC counts.
3. **Unlock upper-GRU fine-tuning through an explicit conversion audit.** Keep the failed strict parity result. Verify equations/gradients and zero-update original-graph roundtrip; define any approximate-candidate acceptance prospectively. Pretrained anchoring is a hypothesis, not an established score gain.

PatchTSMixer and Chronos-2-Synth remain secondary transfer controls. Provenance, target-history mismatch, early context and teacher/student generalization are separate gates. Financial foundation weights are not cleaned by distillation. Smaller GRUs, minRNNs, TCNs and SSMs are alternatives, not ten independent confirmations. Published LOB classification accuracy, unrelated competition WP and foundation MASE rankings cannot be compared numerically with our WP.

Finally, require three unprofiled complete-callback Linux repetitions, sustained streamed evaluation, one CPU quota, 16 GB, disabled networking and the extracted offline ZIP. Retain no-slower-than-original unless representative judge timing changes that policy. Existing ARM64 shared-host measurements do not certify unknown judge architecture, workload or its 60-minute limit. No training, data inspection or confirmation mutation was performed for this memo.

Source audit: three Exa searches, fifteen result slots, four successfully fetched primary-source pages; local reports, selection code and available approach memos read directly.
