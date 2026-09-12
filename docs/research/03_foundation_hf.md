# Approach 3: Hugging Face time-series foundation weights

Research date: 2026-09-12. Recommendation: preserve our GRU as the production model. The strongest eligible-looking external checkpoint is **`autogluon/chronos-2-synth`**, for a bounded offline representation/teacher experiment, not direct submission. No researched checkpoint is a verified drop-in improvement for this task.

Our incumbent has 192,258 parameters, development WP 0.629255535, confirmation WP 0.640997995, a 0.722 MB ZIP and measured streamed throughput of 36,249 rows/second. These are local project measurements from [the implementation report](../../artifacts/IMPLEMENTATION_REPORT.md). Confirmation has already been consumed and must remain closed during this research round.

## Verified checkpoint inventory

Parameter counts below are rounded model-card figures; file sizes are the actual downloadable file sizes displayed by Hugging Face, rounded by its UI, **not measured ZIP sizes**. All listed safetensors are F32. Model-name numbers can differ from total stored parameters.

| Canonical Hugging Face checkpoint | Reported parameters | Published weight file | Weight license | Initial disposition |
| --- | ---: | ---: | --- | --- |
| [amazon/chronos-t5-tiny](https://huggingface.co/amazon/chronos-t5-tiny) | 8.39M | [33.6 MB](https://huggingface.co/amazon/chronos-t5-tiny/blob/main/model.safetensors) | Apache-2.0 | Univariate; provenance unresolved |
| [amazon/chronos-t5-mini](https://huggingface.co/amazon/chronos-t5-mini) | 20.5M | [81.8 MB](https://huggingface.co/amazon/chronos-t5-mini/tree/main) | Apache-2.0 | Larger, same mismatch |
| [amazon/chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) | 8.65M | [34.6 MB](https://huggingface.co/amazon/chronos-bolt-tiny/tree/main) | Apache-2.0 | Only remotely size-plausible after conversion |
| [amazon/chronos-bolt-mini](https://huggingface.co/amazon/chronos-bolt-mini) | 21.2M | [85 MB](https://huggingface.co/amazon/chronos-bolt-mini/blob/main/model.safetensors) | Apache-2.0 | Low priority |
| [amazon/chronos-2](https://huggingface.co/amazon/chronos-2) | 120M | [478 MB](https://huggingface.co/amazon/chronos-2/tree/main) | Apache-2.0 | Mixed-data provenance unresolved |
| [autogluon/chronos-2-small](https://huggingface.co/autogluon/chronos-2-small) | 27.9M | [112 MB](https://huggingface.co/autogluon/chronos-2-small/tree/main) | Apache-2.0 family | Small does not mean synthetic-only |
| [autogluon/chronos-2-synth](https://huggingface.co/autogluon/chronos-2-synth) | 120M | [476 MB](https://huggingface.co/autogluon/chronos-2-synth/blob/main/model.safetensors) | [Apache-2.0](https://huggingface.co/autogluon/chronos-2-synth/raw/main/README.md) | Best conditional offline teacher |
| [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch) | 200M backbone; optional ~30M head | [925 MB](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/main) | Apache-2.0 | Too large; market-data concern |
| [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch) | ~0.3B | [1.32 GB](https://huggingface.co/google/timesfm-3.0-pytorch/tree/main) | TimesFM Non-Commercial v1.0 | Exclude from planned submission |
| [Salesforce/moirai-1.1-R-small](https://huggingface.co/Salesforce/moirai-1.1-R-small) | 13.8M | [55.3 MB](https://huggingface.co/Salesforce/moirai-1.1-R-small/tree/main) | [CC-BY-NC-4.0](https://huggingface.co/Salesforce/moirai-1.1-R-small/raw/main/README.md) | LOTSA market-data issue |
| [Salesforce/moirai-2.0-R-small](https://huggingface.co/Salesforce/moirai-2.0-R-small) | 11.4M | [45.6 MB](https://huggingface.co/Salesforce/moirai-2.0-R-small/tree/main) | [CC-BY-NC-4.0](https://huggingface.co/Salesforce/moirai-2.0-R-small/raw/main/README.md) | Provenance/license unresolved |
| [AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small) | 37.9M | [152 MB](https://huggingface.co/AutonLab/MOMENT-1-small/blob/main/model.safetensors) | MIT | PILE market-data issue |
| [AutoML-org/TempoPFN](https://huggingface.co/AutoML-org/TempoPFN) | 38M advertised | [499 MB `.pth`](https://huggingface.co/AutoML-org/TempoPFN/blob/main/models/checkpoint_38M.pth) | Apache-2.0 | Synthetic-only, CUDA implementation |

TempoPFN's checkpoint contains FloatStorage, but its 499 MB payload is not reconciled with the advertised parameter count; optimizer/auxiliary contents were not inspected. No weights were downloaded. Older official TimesFM [1.0-200M](https://huggingface.co/google/timesfm-1.0-200m-pytorch) and [2.0-500M](https://huggingface.co/google/timesfm-2.0-500m-pytorch) exist; neither resolves size or target semantics. The [official repository](https://github.com/google-research/timesfm) identifies 2.5's smaller backbone and optional quantile head.

## Eligibility is independent of inference location

The supplied [rules](../rules.md) allow publicly available open-source pretrained models only when they do not contain external market data. An offline teacher must satisfy the same restriction: distillation is not a way to erase training provenance.

Chronos-T5 trains on real-data mixtures plus KernelSynth. Its [paper](https://arxiv.org/html/2403.07815v2) separates training and evaluation datasets; Exchange Rate appearing in the complete collection does **not** establish that it was trained on Exchange Rate. Training does include M4 portions with finance categories. That merits a checkpoint-specific audit, not an unsupported blanket accusation. Bolt's public card does not resolve its precise no-market-data eligibility.

GIFT-Eval Pretrain explicitly includes [bitcoin_with_missing](https://huggingface.co/datasets/Salesforce/GiftEvalPretrain/tree/main). TimesFM 2.5 lists that corpus without enumerating exclusions. Chronos-2, Chronos-2-small and Moirai-2 reference subsets; their exact market exclusions remain unverified. Moirai-2 additionally lists private Salesforce operational data. Moirai's [original LOTSA paper](https://arxiv.org/html/2402.02592v1) explicitly includes Bitcoin. MOMENT's [paper](https://arxiv.org/html/2402.03885v3) describes PILE finance/Exchange data and masked pretraining. These are poor eligibility bets compared with an explicitly synthetic checkpoint.

The [Chronos-2-Synth card](https://huggingface.co/autogluon/chronos-2-synth/raw/main/README.md) explicitly states exclusively synthetic univariate and multivariate training. This is the clearest positive provenance evidence found. TimesFM 3.0 is additionally unsuitable because its [new license](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE) restricts commercial/production use and distribution; it is not interchangeable with older Apache releases.

## Why forecasting quality does not imply competition quality

Our [input contract](../data_overview.md) exposes 112 current features, never historical `t0`/`t1` or future covariates. Targets are anonymous future-price indicators with undisclosed definitions. Chronos-T5/Bolt predict future values of an observed univariate series; they cannot know which feature transformation equals either target. Chronos-2 adds multivariate/past-covariate conditioning, but its normal interface still forecasts observed target histories. Its 8,192 context and 1,024 horizon capacity do not solve the missing target semantics. [Canonical interface](https://huggingface.co/amazon/chronos-2).

TimesFM 2.5's core is univariate with optional XReg covariate support; 3.0 adds variate attention. Moirai models accommodate multiple variates. MOMENT offers embeddings and tunable downstream heads. Those capabilities permit **supervised adaptation**, not automatic two-target inference. All need evaluation against WP, not imported MASE/quantile benchmark rankings.

Direct deployment is also unattractive. Tiny T5/Bolt converted to FP16 would have approximately 16.8/17.3 MB of tensor payload by arithmetic, leaving little packaging margin; conversion, ZIP size, CPU kernel support and score are untested. Every larger F32 checkpoint exceeds the cap before packaging. Transformer window recomputation and Python dependencies are serious risks relative to a stateful GRU's roughly 28-microsecond streamed row budget. Published GPU batch throughput cannot validate one-CPU callbacks. TempoPFN explicitly requires CUDA/Triton in its released implementation despite its recurrent architecture.

## Ranked next experiments

1. **Prioritize the supplied GRU.** Improve task-specific recurrent training and causal feature extraction before allocating substantial compute to external foundations. Current calibration already captures most demonstrated gains; foundation-model superiority here is a hypothesis.
2. **Budget one Chronos-2-Synth representation pilot.** Freeze the checkpoint revision and provenance. Feed only trailing observed-feature windows, initially 128 rows and 8–16 train-selected channels. Extract representations through a verified adapter, then fit a regularized two-output supervised head using supplied training labels. Compare against equally sized raw-window and GRU-state heads. Never provide historical targets or future features as model inputs, even during validation.
3. **Distill only after that teacher wins.** Use sequence-disjoint training folds to generate teacher outputs; train a small GRU96/128 student on labels plus teacher matching. Evaluate teacher and student on the 256-sequence development screen, then full development. Require a repeatable improvement and the existing CPU gate. Keep confirmation closed; obtain a new independent evaluation protocol before making another holdout claim.
4. **Use architectures without external weights where appropriate.** Synthetic multivariate mixing and masked reconstruction can inspire pretraining on supplied feature sequences. [TimePFN](https://github.com/egetaga/TimePFN) supplies MIT code and a synthetic-trained checkpoint, but its standard 96-to-96 feature forecasting task still needs a supervised adapter. No canonical HF checkpoint was verified. TempoPFN's linear-recurrence ideas are more relevant than its GPU-only binary.

Search audit: 9 Exa queries, 45 returned result slots; 46 distinct URLs fetched, 44 successfully extracted and 2 HF JSON API pages rejected by the crawler. Evidence is from canonical model cards, file listings, author repositories and papers. No model speed, competition-score gain, compressed size or unverified eligibility is claimed as measured.
