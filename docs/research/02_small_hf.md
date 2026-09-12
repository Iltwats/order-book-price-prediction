# Approach 02: genuinely small Hugging Face models

Research date: 2026-09-12. Goal: improve development WP above **0.629255535** while preserving causal, 112-input/two-output inference under one vCPU and the submission size limit. The supplied GRU already contains only 192,258 parameters and achieves 38,291 complete callbacks/second; foundation-model branding is not evidence of improvement. The previous confirmation result is historical evidence, not a reusable selection set. [Local implementation evidence](../../artifacts/IMPLEMENTATION_REPORT.md)

## Recommendation

Keep the supplied GRU as the main development path. Run **one bounded PatchTSMixer transfer-versus-random-initialization experiment** only after cheaper GRU improvements. It is the strongest clean-provenance, genuinely small pretrained candidate found here. Do not use TTM r1/r2/r2.1 weights, including as distillation teachers: their cards explicitly list the Bitcoin dataset. TTM r3 deserves a provenance inquiry, not an immediate training job. The competition permits publicly available pretrained models only when they contain no external market data. [Rules](../rules.md), [r1 card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1), [r2 card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)

## Actual checkpoints and footprints

Sizes below are **uncompressed weight-file bytes**, not estimated ZIP sizes. Counts describe stored tensor elements; active MoE parameter counts are insufficient for packaging. All entries have real weights except Tiny-TSM and the Google TSMixer code entry. Revision names matter.

| Exact Hugging Face ID / revision | Stored elements; dtype | Weight bytes | Provenance and disposition |
| --- | ---: | ---: | --- |
| `ibm-granite/granite-timeseries-patchtsmixer`, `main` | 196,144; FP32 | 789,328 | Apache-2.0; ETTh1 train only; conditional transfer pilot |
| `ibm-granite/granite-timeseries-ttm-r1`, `main` | 805,280; FP32 | 3,240,592 | Apache-2.0; Bitcoin included; exclude |
| `ibm-granite/granite-timeseries-ttm-r2`, `main` | 805,280; FP32 | 3,240,592 | Apache-2.0; Bitcoin included; exclude |
| Same r2 repository, `52-16-ft-r2.1` | 70,972; FP32 | 319,440 | Truly smaller than our GRU, but same prohibited corpus |
| Same r2 repository, `90-30-ft-r2.1` | 860,620; FP32 | 3,496,760 | Short context does not necessarily mean fewer parameters |
| `ibm-granite/granite-timeseries-ttm-r3`, `main` | 1,414,514; FP32 | 5,729,424 | Family Apache-2.0; selected GIFT corpora unspecified |
| Same r3 repository, `90-30-dec-90-lite-r3` | 432,144; FP32 | 1,799,520 | Smaller actual variant; provenance unresolved |
| Same r3 repository, `52-16-dec-52-lite-r3` | 476,796; FP32 | 1,978,064 | Fits 99-row warm-up context; provenance unresolved |
| `eulogik/nanoforecast-200k`, `main` | 676,108; FP32 | 2,712,760 | Apache-2.0; checkpoint lineage insufficiently documented |
| `Maple728/TimeMoE-50M`, `main` | 113,352,192; BF16 | 226,760,264 | Apache-2.0; unsuitable package/runtime scale |
| `hitachi-nlp/citras-fm`, `main` | 7,219,376; FP32 | 28,896,400 | CC BY-NC-SA weights; mixed corpus; defer |
| Tiny-TSM paper | 23M claimed; unverified file | Unverified | No public checkpoint identified |

Exact sizes/counts were read from primary Hugging Face APIs: [PatchTSMixer](https://huggingface.co/api/models/ibm-granite/granite-timeseries-patchtsmixer?blobs=true), [r1](https://huggingface.co/api/models/ibm-granite/granite-timeseries-ttm-r1?blobs=true), [r2](https://huggingface.co/api/models/ibm-granite/granite-timeseries-ttm-r2?blobs=true), [r3](https://huggingface.co/api/models/ibm-granite/granite-timeseries-ttm-r3?blobs=true), [NanoForecast](https://huggingface.co/api/models/eulogik/nanoforecast-200k?blobs=true), [TimeMoE](https://huggingface.co/api/models/Maple728/TimeMoE-50M?blobs=true), [CITRAS](https://huggingface.co/api/models/hitachi-nlp/citras-fm?blobs=true). **The revision API repeated main-branch parameter counts for differently sized branch files.** Branch counts above were independently summed from safetensors JSON headers using byte-range requests; no tensor weights were downloaded. Audited branch sources: [r2 52](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/52-16-ft-r2.1/model.safetensors), [r2 90](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2/blob/90-30-ft-r2.1/model.safetensors), [r3 90-lite](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3/blob/90-30-dec-90-lite-r3/model.safetensors), [r3 52-lite](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3/blob/52-16-dec-52-lite-r3/model.safetensors).

## What transfers, and what does not

**PatchTSMixer: moderate-to-high integration difficulty.** IBM's checkpoint forecasts seven electricity-transformer channels, 96 hours ahead from a 512-hour context. ETTh1 records electrical loads and oil temperature, not traded electricity prices. This is credible non-market provenance, but unrelated predictive semantics. Its `common_channel` backbone mixes patches/features with shared weights; it is not a pretrained 112-feature financial predictor. We must learn a two-target regression head and cross-channel aggregation exclusively on competition training sequences. The [official card](https://huggingface.co/ibm-granite/granite-timeseries-patchtsmixer), [config](https://huggingface.co/ibm-granite/granite-timeseries-patchtsmixer/raw/main/config.json), and [dataset origin](https://github.com/zhouhaoyi/ETDataset) support these distinctions.

A [community ONNX conversion](https://huggingface.co/onnx-community/granite-timeseries-patchtsmixer) already exists, reducing export uncertainty; it does not solve the new input/head shapes or establish callback latency. Running a 512-row window over 112 channels at every event can cost far more than maintaining two GRU states. Early required predictions begin with only 100 observed rows. Keep the current GRU until genuine context exists, or train a smaller-context architecture from scratch. Do not silently pad and claim pretrained equivalence. Google Research's [TSMixer](https://github.com/google-research/google-research/tree/master/tsmixer) is a separate implementation and training recipe; no reusable foundation checkpoint was verified there.

**TTM: high integration difficulty plus provenance blockers.** r1/r2 primarily learn future observed channel values, whereas our two anonymous targets never appear in inference inputs. Their standard-scaling and forecasting head assumptions need supervised adaptation. r3 introduces shorter-context Lite checkpoints, but its card identifies only selected GIFT-Eval/GiftEvalPretrain datasets and synthetic data, without listing the exact subset. A family-level Apache license does not establish market-free training. [r3 card](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3)

**Other compact claims need scrutiny.** NanoForecast's “200k” checkpoint actually stores 676,108 elements; its repository paper includes exchange-rate data in the family corpus, while this checkpoint's card omits precise lineage. CITRAS has real weights but needs historical target values and has mixed TSMixup/GIFT/synthetic training; its sub-0.1-second CPU claim is not comparable to our approximately 26-microsecond callback. Its code has a separate noncommercial license. Tiny-TSM's “primarily synthetic” training is not proof of exclusively synthetic provenance; its causal normalization is useful inspiration without adopting unverified weights. [NanoForecast paper](https://github.com/eulogik/NanoForecast/blob/main/deploy/paper.tex), [CITRAS paper](https://arxiv.org/html/2606.10798v2), [CITRAS card](https://huggingface.co/hitachi-nlp/citras-fm), [Tiny-TSM paper](https://arxiv.org/html/2511.19272v1)

## Ranked experiments and stop gates

1. **PatchTSMixer transfer control:** use the same small training subset, causal windows and two-output head for pretrained and randomly initialized backbones. Retain a GRU-only control. Screen on 256 development sequences; stop if transfer does not beat random initialization and the current GRU. Do not spend a full training sweep establishing that an unrelated checkpoint has no advantage.
2. **Compact architecture from scratch:** try 64/96-row windows, a learned 112-to-8/16 projection and small mixer blocks. This sacrifices direct pretrained compatibility but fits early context. Benchmark complete ONNX callbacks before expanding training; stop any deployable candidate slower than the original baseline without measured runtime headroom.
3. **Eligible offline teacher:** only if the clean transfer pilot produces stronger, complementary predictions, distill into existing GRU64/96 tools using competition data. Ineligible pretrained weights remain ineligible teachers. Discard the teacher from the ZIP, but preserve its provenance.
4. **TTM r3 follow-up:** request a revision-specific corpus manifest; stop weight adoption until all market-data exclusions are established. Do not treat missing disclosure as absence of market data.

Choose finalists on development paired sequence differences, preserve confirmation-access guards, verify causality/reset behavior, measure three Linux one-CPU repetitions and extracted ZIP bytes. No claimed foundation-model benchmark gain substitutes for local WP.

Search accounting: **six Exa searches, 45 returned results, 41 distinct URLs**, followed by primary cards/configs/APIs. Two Hugging Face index searches (`Tiny-TSM`, `TinyTSM`) returned zero results. No model weights, training datasets or executable third-party code were downloaded; only metadata and source documentation were read.
