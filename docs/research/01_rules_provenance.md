# Approach 1: eligibility and pretrained-weight provenance

Verified 2026-09-12. Research only: no weights downloaded, training performed, code imported, or organizers contacted.

## Recommendation

Keep the supplied GRU as the principal starting checkpoint. No verified, directly compatible public Connectome checkpoint was found in this search. A closely named Hugging Face release exists, but it belongs to an earlier competition and was trained on that competition's market data. Its weights should not enter this submission without an explicit organizer interpretation permitting that provenance. Its architecture can inspire experiments trained on our supplied data.

Our present release already has strong provenance: all 192,000 recurrent parameters remain unchanged from the supplied GRU; the head was fitted on 1,024 training sequences. Confirmation WP is 0.640997995 versus 0.599669876 for the original baseline. Most improvement came from calibration; the larger affine fit was practically tied with the smaller fit. That evidence favors better objectives and recurrent learning over searching for a replacement checkpoint by name. [Local implementation report](../../artifacts/IMPLEMENTATION_REPORT.md)

## What the organizer actually says

The current rules permit public open-source pretrained models conditional on the phrase **“publicly available and do not contain external market data.”** Public availability alone is therefore insufficient. The page also prohibits outside training data, and prohibits sharing complete solutions, models, or detailed strategies during the competition, including private sharing between participants. Reading a repository does not establish permission to incorporate its weights. [Current competition rules](https://wundernn.io/connectome/docs/rules)

The live FAQ explicitly permits training with the supplied **train.parquet and valid.parquet**, while repeating the same pretrained-model condition. This clarifies that organizer eligibility and our validation discipline are separate concerns: using validation for training can be permitted, but it destroys its role as an independent performance estimate. Neither the rules nor FAQ defines “contain,” provides an approved checkpoint list, or grants a teacher/distillation exception. [Current FAQ](https://wundernn.io/connectome/docs/faq)

| Candidate provenance | Working disposition |
| --- | --- |
| Supplied baseline, retrained or fine-tuned using current supplied data | Proceed within the existing evaluation protocol. |
| Public open-source checkpoint with documented non-market pretraining | Potentially eligible under the written exception; verify the complete training lineage and license. |
| Checkpoint trained on stock prices, order books, trades, or another competition's financial observations | Exclude from our immediate experiments; no published exception was verified. |
| General time-series checkpoint with an incomplete corpus list | Unresolved; absence of a finance tag is not proof of no market data. |
| Model distilled from an ineligible or unverified financial teacher | Unresolved; distillation transfers learned behavior and does not establish clean provenance. |

The last two rows are research judgments, not organizer rulings. Similarly, synthetic data produced from a market-trained teacher is not automatically independent of its source. A synthetic-only checkpoint whose generator and parameter fitting never use outside market observations has a substantially clearer lineage, although the specific artifact still needs review.

## Hugging Face and repository findings

**msrishav/wunder-fund-lob-predictorium** is a real, MIT-labelled release with ten stateful ONNX GRUs. Its author identifies it as a portfolio project for the completed LOB Fairy Predictorium competition. Its interface has **32 inputs**, **1,000 steps**, **1,444 validation sequences**, and clipping at **±6**. The card reports local WP 0.2846 and explicitly says its models were trained on that competition's supplied data. Those weights therefore contain a different competition's market training information; the repeated names `t0`, `t1`, and `PredictionModel` do not make them current Connectome weights. Its numbers cannot be compared directly with ours. [Publisher model card](https://huggingface.co/msrishav/wunder-fund-lob-predictorium)

The matching source repository provides useful methodological ideas: train-fixed normalization, causal differences and moving averages, unidirectional GRUs, weighted MSE, and per-target ensemble selection. These are hypotheses to reimplement and test locally. Its author’s claim about a low causal ceiling for its second target must not be transferred to our targets; our existing second-target correlation already differs substantially. [Publisher repository](https://github.com/msrishav-28/wunder-fund-LOB-fairy-predictorium)

**aicheye/wundernn** addresses another earlier challenge: next-state vector prediction, 1,000-step sequences, and R² evaluation. It is not a matching two-target, masked-WP Connectome checkpoint. [Author repository](https://github.com/aicheye/wundernn)

Name collisions are widespread. **TuKoResearch/NoConnectomeGPT100M** is a language-model research artifact; **medarc/CortexMAE** is an fMRI masked autoencoder trained on Human Connectome Project observations. These are unrelated to this trading competition. Their presence in search results is not evidence of pretrained trading capability. [TuKoResearch model card](https://huggingface.co/TuKoResearch/NoConnectomeGPT100M), [MedARC model card](https://huggingface.co/medarc/CortexMAE)

The actual current quick start confirms 112 features, 20,000 rows per sequence, 10,607 training sequences, 1,873 validation sequences, and a stateful GRU baseline scoring 0.589595 WP. These facts provide a reliable fingerprint when screening purported competition artifacts. [Current quick start](https://wundernn.io/connectome/docs/quick_start)

## Next actions and unresolved questions

1. Record repository URL, immutable revision, weight checksum, license, corpus list, and upstream teachers for each serious candidate before downloading weights.
2. Favor architecture transfer and training on supplied data while checkpoint provenance remains unresolved.
3. Preserve the completed confirmation ledger. The already inspected 473 sequences cannot be presented as a fresh holdout in the next round.
4. If a financially pretrained candidate becomes compelling, prepare an organizer question about that exact checkpoint, its corpus, and direct versus teacher-only use. No clarification request has been sent.

## Search coverage and verification limits

Three distinct Exa queries returned **30 result slots, representing 19 unique URLs**; repeated retrievals were not counted again. Three domain-restricted fallback web queries returned **13 additional result entries**, with overlap. Six primary pages were read in full: live rules, FAQ, quick start, and the three publisher pages discussed above. Other model cards were inspected through returned source excerpts. Search-result counts are not thoroughly-read-source counts.

Exa initially returned Next.js scripts for the organizer pages, and ordinary web extraction was incomplete. All three organizer pages were subsequently verified in the rendered browser, including the visible rule clause and FAQ. The search did not exhaust the Hugging Face Hub, private repositories, or Discord. “No verified matching checkpoint found” is a bounded search result, not a claim that none exists.
