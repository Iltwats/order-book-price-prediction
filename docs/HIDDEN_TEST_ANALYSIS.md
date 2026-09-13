# What the hidden results tell us

Evidence updated 2026-09-13. The user supplied four public submission results, including **0.6092**, **22m55s**, and **MAX 624 MB** for `submission_correlation1024.zip`. This memo combines those observations, current official documentation and previously computed development scores. It performs no new model inference, training or dataset scan and does not reopen confirmation or the completed evaluation budget.

**The improvements transfer to public evaluation, but their size is small. The current evidence does not identify the hidden dates, instruments or scoring-mask mechanism.** The local/public difference is about 0.0208; the remaining public gap to 0.75 is 0.1408. Even the local score is approximately 0.12 short of the target. These arithmetic comparisons are not a causal decomposition of error.

## What is actually tested

The Public Leaderboard reports a sample of test data. Final ranking uses a **separate private hidden dataset**, re-evaluating selected previously scored submissions. The screenshot is public feedback, not a private-final score. The rules do not disclose random versus chronological selection or split proportions. [Official rules](https://wundernn.io/connectome/docs/rules)

Test sequences have the same format as validation: 20,000 consecutive rows, 99 warm-up rows, 112 FP32 input features, and two prediction outputs. Both targets describe future price movement of instrument **i0**; **i1** contributes input features. The organizer feeds rows to the callback sequentially. Each sequence starts with reset state, and only current/past features can be used. Instrument identities, dates and target horizons remain undisclosed. [Official data overview](https://wundernn.io/connectome/docs/data_overview)

The scoring mask `is_scored` is hidden and applied after inference. Every required row still needs predictions. Scoring clips predictions and targets to ±2, weights rows by absolute clipped targets, and gives each eligible sequence equal final weight. Mask generation is not described in the reviewed documentation. Matching file format does not prove matching statistical distributions or mask-selection mechanisms. [Official data overview](https://wundernn.io/connectome/docs/data_overview), [local scoring reference](../METRIC.md)

Physically, the code runs in an organizer-managed scoring Docker container based on `python:3.11-slim-bookworm`, with one vCPU, 16 GB RAM, no GPU/network, and a 60-minute inference limit. CPU architecture/model, cloud provider, geographical region, hidden row count and exact UI duration/memory accounting are not disclosed. We cannot recover those quantities from the displayed duration and local throughput. [Official submission guide](https://wundernn.io/connectome/docs/submission_guide)

## The four observed results

| Matching archive | Local full-development WP | Displayed public WP | Local minus public | Displayed duration |
|---|---:|---:|---:|---|
| `submission.zip` | 0.629256 | 0.6083 | 0.020956 | 22m46s |
| `submission_round2_affine_rho.zip` | 0.629307 | 0.6085 | 0.020807 | 22m08s |
| `submission_correlation256.zip` | 0.629469 | 0.6086 | 0.020869 | 23m11s |
| `submission_correlation1024.zip` | 0.630019 | 0.6092 | 0.020819 | 22m55s |

All four show MAX 624 MB. Filenames match our local archives; the screenshot does not independently verify their uploaded contents or checksums. Public scores have four displayed decimals, and the unrounded values and rounding method are unavailable. [Screenshot observation receipt](../artifacts/round2/public_submissions_20260913.json)

The local gain from the first to latest model is **+0.00076356**, compared with **+0.0009 displayed publicly**. The local/public offset spans only about 0.000149 across these four models. This is compatible with a common difference in sequence difficulty, selection or distribution; finite-sample variation and local selection effects also remain possible. Four related models on shared datasets are not independent confirmations. The evidence does not support fitting a score correction, estimating a hidden sampling distribution, or predicting a private score.

## Weak cases already exist locally

Reanalysis of the cached reports covers all 1,400 development sequences, with 1,399 eligible and confirmation excluded. The latest model's median sequence WP is **0.687714**, population standard deviation **0.207130**, and minimum **−0.427098**. Nine sequence scores are negative. Its weakest quarter averages **0.335232**; its strongest quarter averages **0.842250**. These are performance groups, not identified market regimes. [Reproducible cached diagnostic](../artifacts/round2/hidden_gap_diagnostic.json)

For paired comparisons, fix membership using the incumbent's scores:

| Incumbent performance group | Sequences | Incumbent mean WP | Latest mean WP | Mean gain |
|---|---:|---:|---:|---:|
| Weakest quarter | 350 | 0.333555 | 0.335250 | +0.001695 |
| Second quarter | 349 | 0.595275 | 0.595956 | +0.000680 |
| Third quarter | 350 | 0.746158 | 0.746523 | +0.000365 |
| Strongest quarter | 350 | 0.841936 | 0.842250 | +0.000314 |

The latest model improves **1,016 of 1,399 sequences (72.62%)**, with positive mean gains for both targets. The weakest incumbent quarter contributes about 55.5% of the net gain, yet its improvement remains only 0.001695. The problem is broader than a handful of catastrophic failures.

A hypothetical mixture illustrates why an aggregate score cannot identify hidden contents: moving about **4.1 percentage points** of weight from sequences averaging 0.84225 to sequences averaging 0.33523 would lower an average by roughly 0.02082, if everything else stayed fixed. This is arithmetic, **not an estimate of the hidden composition**. Many other mixtures or changes in within-group difficulty could produce the same score.

## Hypotheses worth testing

| Hypothesis | Evidence and limits | Falsifiable next check |
|---|---|---|
| Limited adaptation | Our winning fine-tune used 1,024 of 10,607 supplied TRAIN sequences, one epoch and 128 updates. The lower GRU stays frozen. Larger-data seeds improved, but both data coverage and update count increased. Original pretraining coverage is not established by these fine-tuning counts. | Reserve a whole-sequence holdout for future adaptation; separate broader data coverage from more optimizer updates. Measure an actual held-out learning curve. |
| Truncated gradients or frozen lower features | Backward credit assignment stops at 256-row boundaries. Full causal forward history already persists in ORT. A slow hidden coordinate motivates investigation but does not prove useful long-range gradients. | Run the separately proposed 256-versus-1,024 gradient-window comparison and lower-layer sensitivity probe on eight fixed TRAIN groups, without optimizer updates. |
| Different selected rows | TRAIN has no scoring mask; official validation/public WP uses one. The contribution of this mismatch is unknown. | Keep the clipped, equal-sequence objective. Use prespecified subset sensitivity as a diagnostic, never as a claimed reconstruction of the official mask. |
| Different feature regimes | Other activity levels, liquidity conditions, time periods or changes in i0/i1 relationships are plausible. No hidden-row evidence identifies any of them. | Report errors against prespecified causal feature proxies on a bounded TRAIN probe: same-column changes, volume-like feature magnitudes and cross-instrument change co-movement. Learn scales/thresholds only within the declared training procedure. |
| Callback or numerical fault | Full-sequence parity, state/reset checks, offline ZIP verification and successful public runs weigh against a major contract failure. They cannot exclude every distribution-specific issue. | Retain finite-output and sequence-contract checks; prioritize a runtime rewrite only when a failure is reproducible. |

Anonymous indices are not book-depth or price ordering; `p0` is not guaranteed to be best bid. Sequence IDs are shuffled and cannot identify chronological periods. Feature-derived activity/liquidity proxies must be labelled as proxies, rather than assumed physical measurements. [Official data overview](https://wundernn.io/connectome/docs/data_overview)

## Practical next action

First implement the existing [bounded gradient diagnostic](CORRELATION_NEXT_EXPERIMENT.md). Separately, prospectively select 128 additional whole TRAIN sequences, excluding those used in our prior fitting/diagnostics, for one fixed comparison of the incumbent and completed winner across declared feature proxies. This is a proposed analysis, not a completed scan. Do not train, tune thresholds or choose among additional candidates on this probe. Once inspected, it becomes development evidence.

Any future adaptation holdout should be disjoint from that probe and remain sealed until one candidate is frozen. Unknown supplied-model pretraining lineage means it can be held out from **our adaptation**, not proven unseen by the original pretrained model. The completed 12-screen/four-development record and consumed confirmation remain unchanged.

The runtime evidence now permits investigating modestly more expensive models, but the same-architecture options come first: more useful training, longer backward windows or lower-layer adaptation add no inference operations. At the reported 22m55s, a 45-minute target is **1.96 times** that duration, and the 60-minute limit is **2.62 times**. Twofold cost would suggest about 45m50s only under comparable workload and accounting; it is not measured scaling or a private-test guarantee. Do not infer CPU speed gains from the four durations: they are single runs with unknown measurement scope. The earlier causal branch's rejection remains its result under the old gate; revisiting it would need an explicit new performance budget and experiment.

Research coverage: three Exa searches requested 15 result slots. Four official Connectome pages were read in rendered form after script-heavy extraction results. Claims about Predictorium, including approximately 1,500 test sequences, were excluded because it is a different competition. [Source-review record](../artifacts/round2/hidden_test_source_review_20260913.json)
