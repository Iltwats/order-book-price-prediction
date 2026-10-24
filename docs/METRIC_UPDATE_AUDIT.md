# Official documentation audit — 2026-10-24

**Follow-up completed:** the refreshed official `METRIC.md` was recovered with a
bounded ZIP read; its thresholds and centered-moment merging are now verified.
Global scoring, versioned selection, the [12-model rescore](GLOBAL_WP_RESULTS.md)
and the [v2 GPU plan](GPU_TRAINING.md) are implemented. The signed-in submission
history now verifies **0.65151** for `submission_correlation1024.zip`.

The original findings below describe the state before that migration; the old
per-sequence plan remains superseded. CUDA qualification is still pending.

## Original audit findings

**The competition now scores Global Weighted Pearson. Our local scoring and
correlation-training code still implement the earlier per-sequence objective.
The new GPU handoff is suspended until that mismatch is corrected.**

The user reports a current public score of approximately **0.651**. The earlier
0.6092 figure is historical and must not be presented as the current result.
The account's current submission receipt and matching uploaded ZIP were not
independently verified in this audit.

## Verified changes

The current official pages were read in a rendered browser on October 24.
The search connector could not extract the JavaScript-rendered documentation.
The comparison below uses the existing local documentation and code as the
earlier snapshot; the official pages did not disclose a change date.

| Item | Earlier local record | Current official documentation |
| --- | --- | --- |
| Score aggregation | Weighted correlation per sequence/target, then an equal mean across eligible sequences | Pool all selected rows across the evaluated dataset; compute one weighted correlation per target and average those two values |
| Supplied baseline's complete-validation score | 0.589595 | 0.617052 |
| Test workload | Unknown in our earlier analysis | 1,970 sequences, 39,400,000 rows |
| Scoring CPU | Architecture/model undisclosed in the earlier reviewed pages | x86-64, AMD EPYC 9224; the submission allocation remains one vCPU |
| Validation setup | The supplied validation mask was our main scoring domain | Explicit permission to define additional splits/masks from supplied labeled data, with a warning about overfitting the visible mask |
| Teams | Local FAQ said the interface did not support team formation | Up to five people; creation/join/merge deadline November 1, 2026 at 23:59 UTC; shared five-submission daily limit |

Sources: [current data overview](https://wundernn.io/connectome/docs/data_overview),
[quick start](https://wundernn.io/connectome/docs/quick_start),
[FAQ](https://wundernn.io/connectome/docs/faq),
[submission guide](https://wundernn.io/connectome/docs/submission_guide),
[rules](https://wundernn.io/connectome/docs/rules),
[timeline](https://wundernn.io/connectome/docs/timeline).

The decisive wording in the data overview is:

> All selected moments are pooled across sequences; scores are not averaged by sequence.

The unchanged parts include 112 ordered features, two targets, 20,000 rows per
sequence, 99 warm-up rows, causal state updates and resets, the scoring mask,
clipping targets/predictions to ±2, absolute clipped-target weights, CPU-only
inference, 16 GB RAM, 60 minutes, a 20 MB ZIP and no runtime internet. Training
remains restricted to supplied data; the local FAQ already allowed using the
provided train and validation files, so that is not a newly discovered permission.

The current guide also exposes the scorer package ranges. They are not exact
installed versions: for example ORT is specified as `>=1.18,<2`. Reproduce the
x86-64 callback behavior before treating earlier ARM64 timings as judge evidence.
The CPU's 48 visible logical processors do not give a submission 48 vCPUs.

## Why the score can change without a model change

The earlier statistic is a mean of correlations calculated with separate
sequence means and variances. The new statistic uses one weighted mean,
variance and covariance per target across all selected rows. It is not a
weighted average of the old per-sequence correlations. Sequence offsets,
scales and total target weights can change model rankings under the new score.

This is a plausible explanation for the reported leaderboard movement, not
proof of when or how the organizer rescored a particular uploaded archive.
The updated baseline number does not establish whether its weights changed;
that requires comparing a refreshed starter-kit model hash. The download is
login-gated in the currently rendered quick-start page, and an HTTP HEAD request
to the previously published ZIP URL returned 403. Revised starter-kit files and
dataset hashes were not verified here. The web overview and FAQ agree
on global aggregation, but the exact updated scorer's degeneracy thresholds
still need the refreshed `METRIC.md`/`utils.py` reference.

## Effect on this repository

- `utils.BlockAccumulator`, `connectome/evaluation.py` and cached calibration
  scoring average sequence correlations and apply sequence eligibility. They
  are historical scorers, not the current competition reference.
- `connectome/recurrent_objectives.py` centers and normalizes each sequence
  separately and averages `batch_size × 2` sides. The new GPU runner reuses that
  objective. It must not be described as aligned with current official scoring.
- Selection validators, confidence intervals and head/moment training also rely
  on the old aggregation. A metric/version field must prevent mixing old and
  corrected reports. Cluster bootstrap should resample sequences and recompute
  pooled correlations from statistics, rather than average sequence-score deltas.
- The old requirement for positive and negative targets within every sequence
  cannot simply be carried into a scorer that pools all selected moments. Match
  the refreshed official reference before deciding any global degeneracy policy.

The original runtime, causality, state-reset and export-parity checks remain
useful. Earlier score rankings and improvement intervals remain historical
evidence under a different metric. Keep them intact and label the distinction.
The consumed confirmation set does not become an independent holdout again.

## Next work before GPU training

1. Obtain and compare the refreshed scorer reference without replacing the
   historical package. Implement versioned global accumulation and tests against
   direct concatenation, clipping, one-sign sequences and partition invariance.
2. Re-score a small, predeclared set of existing models on the same development
   domain. Raw complete prediction caches exist for the original baseline and
   selected calibrated model, so those need no new inference. The completed
   correlation-1024 winner has per-sequence reports but no matching dev prediction
   cache found; its global score needs a new bounded inference pass. Scalar
   sequence correlations alone cannot reconstruct the global score.
3. Reassess calibration and incumbent choice under Global WP. Then declare a
   pooled training objective and new learning-rate/teacher settings. Pooling a
   batch is still a training surrogate for pooling the complete corpus.
4. Freeze new run plans and rebuild the private cloud handoff only after those
   checks. The old plan is preserved with a superseded receipt, and its archive
   is renamed to identify the historical block-WP method.

No new validation labels/predictions were read, models trained, scores computed,
cloud machines provisioned or submissions uploaded during this audit.
