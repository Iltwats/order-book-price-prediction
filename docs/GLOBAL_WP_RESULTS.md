# Existing models under Global WP — 2026-10-24

The 1,024-sequence correlation model remains the strongest existing model on our fixed development split. The revised score changes several intermediate rankings. No competition submission was uploaded.

The predeclared comparison covers **12 existing configurations**, all **1,400 development sequences** (2,612,518 selected rows). Four models required new CPU ONNX bulk inference; the others used complete raw prediction caches. Confirmation was excluded. Each configuration was scored once, with no new calibration search.

| Existing configuration | Development Global WP | Change versus earlier selected |
| --- | ---: | ---: |
| correlation1024_seed20260912 | 0.672091945 | +0.000361845 |
| correlation1024_seed20260913 | 0.672078182 | +0.000348082 |
| correlation256 | 0.671835061 | +0.000104961 |
| affine1024_scaled | 0.671730101 | +0.000000000 |
| selected | 0.671730101 | +0.000000000 |
| affine256_scaled | 0.671611439 | -0.000118662 |
| round2_affine_rho | 0.671587454 | -0.000142646 |
| affine256_raw | 0.671177101 | -0.000552999 |
| affine1024_raw | 0.671163245 | -0.000566856 |
| baseline_quarter | 0.668736378 | -0.002993723 |
| baseline_half | 0.658188840 | -0.013541261 |
| baseline | 0.610331386 | -0.061398714 |

The winner's paired sequence-cluster 95% interval versus the earlier selected model is `[+0.000219533, +0.000492291]`. Clusters are sampled jointly and the pooled correlation is recomputed. This interval does not remove development selection bias. The two winning seeds are very close; no claim is made that one seed is generally better.

The earlier affine-correlation head now scores below the selected calibrated model locally. Its current public score is higher than that comparator, so the development ranking is not a guarantee of hidden-test ordering.

## Current public submission history

These values were read from the signed-in [submission page](https://wundernn.io/connectome/submit). They are hidden-data scores and are separate from the development table. Filenames and submission IDs were checked; uploaded ZIP checksums were not.

| Archive | Current public WP |
| --- | ---: |
| submission_correlation1024.zip | 0.65151 |
| submission_correlation256.zip | 0.65110 |
| submission_round2_affine_rho.zip | 0.65146 |
| submission.zip | 0.65095 |

The original local baseline scores 0.610331386 on this development subset. The organizer documents 0.617052 on complete validation; the domains differ, and confirmation was not reopened to reproduce that full-validation number.

## Training decision

Plan `3723823f45de3395f4c12962230645515c20fbd4362c438619dfd4e07d0c10c4` starts from `correlation1024_seed20260912` and uses the same model as the unchanged teacher. Deployment postprocessing is preserved. It uses the new batch-pooled Global-WP surrogate, rather than averaging sequence losses.

The two fixed arms compare four passes over the original 1,024 groups with one pass over 4,096 groups: 512 optimizer updates each, batch size 8, backward window 256, learning rate 1e-5, teacher weight 0.1. The reduced learning rate is a conservative starting hypothesis, not an optimum established by this rescore. The original 512 reserved TRAIN groups stay excluded. No new validation rows become independent merely because the metric changed.

A real eight-sequence CPU pilot completed one update with the new objective. Synthetic tests cover exact relocated resume and numerical derivatives. CUDA qualification, actual GPU speed/memory, Docker build, cloud execution and scores from newly trained candidates remain pending.

## Evidence

- [Frozen comparison policy](../artifacts/global_wp/rescore_v1/policy.json)
- [Complete ranking and clustered comparisons](../artifacts/global_wp/rescore_v1/ranking.json)
- [Post-run cache hashes, runtime and official-reference receipt](../artifacts/global_wp/rescore_v1/evidence.json)
- [Winner callback/bulk byte parity on 32 full development sequences](../artifacts/global_wp/rescore_v1/winner_parity32.json)
- [Frozen Global-WP training plan](../artifacts/gpu/global-wp-v2/plan.json)
- [GPU training guide](GPU_TRAINING.md)

Cache array hashes were recorded after scoring, not prospectively in the policy. They identify the observed files for reproduction. Cache metadata and model/data identities were checked before scoring.
