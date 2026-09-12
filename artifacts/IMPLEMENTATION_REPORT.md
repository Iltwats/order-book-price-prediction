# Implementation and benchmark report

Recorded 2026-09-12. **Release: supplied GRU128 with a train-fitted affine head, calibration folded into that head, and single-thread ONNX Runtime I/O binding.** The final ZIP is [submission.zip](../dist/submission.zip), 722,475 bytes (0.722 MB). The original baseline remains available unchanged.

## Accuracy and final decision

| Split | Eligible / total sequences | Original WP | Selected WP | Difference |
| --- | ---: | ---: | ---: | ---: |
| Development | 1,399 / 1,400 | 0.586188044 | 0.629255535 | +0.043067491 |
| Confirmation, evaluated once after freezing | 473 / 473 | 0.599669876 | 0.640997995 | +0.041328119 |
| Combined validation, descriptive | 1,872 / 1,873 | 0.589594511 | 0.632222513 | +0.042628002 |

The confirmation improvement is **+0.041328119 WP**, with a paired 95% sequence-bootstrap interval **[+0.037460151, +0.045373192]**, using 10,000 resamples and seed 20260912. The candidate wins on 79.28% of confirmation sequences. The predeclared positive-mean-improvement rule selects the candidate. There was no subsequent confirmation tuning. Bootstrap uncertainty assumes sequences are sufficiently independent; it does not establish performance on an unknown hidden distribution. Evidence: [confirmation decision](confirmation_decision.json), [frozen configuration](frozen_candidate.json), and [confirmation report](selected/confirmation_report.json).

| Target | Original confirmation | Selected confirmation | Original full validation | Selected full validation |
| --- | ---: | ---: | ---: | ---: |
| t0 | 0.583218786 | 0.643710738 | 0.571491710 | 0.632967850 |
| t1 | 0.616120966 | 0.638285252 | 0.607697312 | 0.631477176 |

The original full score reproduced at 0.589594511, only -4.89e-07 from the rounded published 0.589595 and within the 5e-6 investigation threshold. Full validation contains 37,460,000 rows. The combined selected score merges the saved disjoint development and confirmation reports, weighted equally by eligible sequence. It is not another holdout experiment: development was used for selection. Original baseline confirmation was evaluated earlier solely as the fixed control for baseline reproduction; candidate confirmation was opened only after the model was frozen.

## What was selected and why

All **192,000 recurrent parameters** are bitwise unchanged from the supplied model. Six affine coefficients were fitted using all post-warm-up rows from 1,024 selected training sequences (20,480,000 input rows), with sequence-normalized weighted MSE against clipped targets and ridge penalty 0.01 toward identity. Both output scales are 1.25. Affine coefficients and scales are folded into the existing 128-to-2 head, so the deployed architecture, operation count and 192,258 total stored parameters remain unchanged. Preprocessing and feature order are unchanged. Only supplied competition data was used, on local CPU. [Weight provenance](selected/provenance.json) and [training report](training_pilot1024/affine_p1e-2/report.json) record exact groups, source hashes and settings.

| Screened configuration | Calibrated screen WP | Full development WP |
| --- | ---: | ---: |
| Original baseline, unchanged | 0.595017563 | 0.586188044 |
| Original baseline, both scales 0.5 | 0.629899826 | 0.620368401 |
| Original baseline, both scales 0.25 | 0.638185271 | 0.628562389 |
| Affine fit, 256 training sequences, scales 1.25 / 1.5 | 0.638910223 | 0.629237395 |
| Affine fit, 1,024 training sequences, scales 1.25 / 1.25 | 0.638916086 | 0.629255535 |
| Last-state residual heads, three penalties | approximately 0.61620 | screened out |
| Both-state residual head, affine warm start | 0.637656178 | screened out |

The initial predeclared scale search won at its lower boundary, 0.5. A separately recorded screen-only refinement tested 25 pairs from {0.0625, 0.125, 0.25, 0.375, 0.5} and selected the interior value 0.25. This is an explicit extension of the original search, and shows that most of the score gain is achievable through cheap calibration. EMA candidates did not win. Affine1024 exceeds this stronger quarter-scale control by 0.000693146 on development, with paired interval [+0.000469399, +0.000946672]. This incremental advantage was not separately tested against quarter-scale on confirmation; the frozen confirmation comparator is the original baseline.

The two affine fits are practically tied: 1024-minus-256 development difference is +0.000018140, interval [-0.000124243, +0.000156984]. Highest measured development WP selects 1024; these results do not show a statistically established benefit from using more training sequences. Last-state pilots tested penalties 1e-4, 1e-3 and 1e-2 with zero-initialized corrections, learning rate 1e-3 and 4,096-row minibatches. The both-state pilot started from the better affine head and retained original ONNX recurrent operations. It also lost on screening. No full-training-set sweep, ensemble or further training expansion was undertaken. [Final development ranking](development_selection_final.json) and [paired shortlist comparisons](development_shortlist_comparison.json) retain the selection evidence.

## Measured CPU performance

Final production tests ran in Linux ARM64 Docker with Python 3.11.16, NumPy 2.2.6 and ONNX Runtime 1.23.2, one CPU quota, **exactly 16,000,000,000 bytes RAM**, no extra swap and no networking. They ran on the shared physical M2 Pro host and did not reserve an exclusive core.

| Measurement | Original published callback | Selected strict callback |
| --- | ---: | ---: |
| Median complete callbacks/second | 34,962 | 38,291 |
| Median seconds for 640,000 callbacks | 18.305398 | 16.714034 |
| Measured throughput gain | — | **9.52%** |

Each of three unprofiled repetitions used the same 32 complete sequences, including warm-up and resets, and alternated baseline/candidate order. The reference was the actual supplied `baseline/solution.py`, and the selected callback retained finite-input validation. Constructor startup was measured separately (candidate approximately 2–4 ms after dependency imports); Parquet loading was excluded from this callback timing. The preloaded benchmark peak RSS was 638.4 MB. [Complete callback measurements](linux_arm64/solutions_32seq_unpinned.json).

The production streamed evaluator processed **5,120,000 rows in 141.246 seconds**, or **36,249 rows/second**, including model construction, Parquet loading, strict callbacks, validation and scoring. It kept only one sequence in memory, used no prediction cache, and peaked at **370.0 MB RSS**. Linux screen WP was 0.638916098502, closely matching the corresponding host score. [Streamed evaluation](linux_arm64/stream_256seq.json).

The broader API study separately tested cached calls, persistent I/O binding, OrtValue vectors, and the simplified graph, with 32 sequences and three repetitions per case. Its best original-graph mode was I/O binding. Simplification preserved numerical results but did not show a repeatable speed advantage; the production model retains the original graph. Some later vector timings were variable under concurrent host work. Historical matrix tests used 16 GiB (17.18 GB); the final production tests above used the corrected decimal 16 GB limit. [Full Linux methodology, repetitions and caveats](linux_arm64/MEASUREMENTS.md).

## C++, cache layout and affinity

The C++17 backend is implemented with a C ABI and Python `ctypes`, 64-byte-aligned owned buffers, one-time weight packing and allocation, and standard FP32 sigmoid/tanh. It explicitly implements ONNX [z,r,h] gate order and `linear_before_reset=1`, preserving separate input/recurrent candidate biases. These equations are specified in the [ONNX GRU reference](https://onnx.ai/onnx/operators/onnx__GRU.html). There is no per-step native heap allocation, worker pool, fast-math or host-specific ISA flag. Native startup failure can choose ONNX before inference; runtime errors never silently reset state or return zeros.

Native release and ASan/UBSan checks pass. However, the full 20,000-row Linux parity audit failed the initial atol=1e-6 / rtol=1e-5 criterion: maximum prediction error 7.39098e-5 and hidden-state error 6.31213e-5. Explicit alignment did not change these errors. Aligned row/packed layouts measured 11,251 / 32,252 callbacks/s in the short three-repeat diagnostic, both below the final ONNX callback. Native therefore fails both the numerical gate and the required 10% speedup promotion gate. Its source and compiled Linux ARM64 library are delivered as experimental artifacts; no native library is included in the release ZIP. [Native implementation](../native/README.md) and [aligned Linux evidence](linux_arm64/align64/summary_prepare_32seq.json).

The packed traversal substantially improved native speed, but hardware cache counters were unavailable; no cache-miss count of zero is claimed. The affinity experiment read the allowed CPU set and pinned only its isolated process to one permitted logical CPU. Measurements were variable, so deployment affinity remains disabled. Pinning does not increase a CPU quota or grant exclusive core ownership, as explained by the [Docker CPU controls](https://docs.docker.com/engine/containers/resource_constraints/).

## Verification and implemented conditional work

- Host suite: **89 passed**. Linux inference suite: **79 passed**, with the separate Torch training module skipped because Torch is absent from the inference image. The nine training tests passed on the host. [Host log](pytest_host.log).
- Selected I/O binding versus reference: bitwise equal predictions and both hidden states at every step of 32 complete sequences, including 163,840,000 hidden-state values. Offline sequence output also passed. [Parity report](selected/parity_smoke32.json).
- Folding scales into the head passed the original combined absolute/relative tolerances; maximum absolute difference was 2.86102e-6, without relaxing thresholds. A fresh full-development evaluation of the folded artifact changed WP by only 8.79e-11. No old report was relabeled with the new model hash. [Folding audit](selected/fold_parity.json).
- Warm-up updates, sequence resets, future-input causality, determinism, finite outputs, output ownership, invalid model schema, startup-only native fallback, excluded metric blocks, cache provenance, packaging isolation and confirmation safeguards are tested.
- The fresh extracted ZIP passed offline in Python 3.11 on the host and Linux. Linux verified the exact SHA256 now present at `dist/submission.zip`; renaming from the candidate ZIP did not change bytes. No compilation, dependency download or training occurs inside the submission. [Archive verification](linux_arm64/fresh_zip.json).

Training/export tools for recurrent fine-tuning and GRU64/96 students, teacher matching and deterministic checkpoint/resume are implemented and exercised on bounded real-data smoke runs. Full recurrent fine-tuning remains unpromoted because the recovered Torch implementation does not pass the original long-sequence FP32 tolerance. Exact parameter reconstruction and common-previous-state diagnostics support kernel rounding amplified by recurrence; this was investigated and reported, not silently accepted. Frozen-head fitting instead uses original ONNX states and passes bitwise parity. [Training implementation and conditional epoch-selection workflow](../docs/training.md).

Generic quantization remains deferred: the [pinned ORT 1.23.2 registry](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/python/tools/quantization/registry.py) does not provide a GRU quantizer. Smaller students or ensembles were not needed to satisfy the measured baseline-speed gate, and were not promoted from smoke experiments.

## Release boundaries and reproduction

The archive is below the 18 MB engineering target and 20 MB stated cap. Final Linux inference used far less than 16 GB. **The hidden test size and CPU architecture/speed remain unknown, so the 60-minute judge limit is not certified.** At this measured streamed rate, a simple constant-rate projection allows approximately 97.87 million rows in 45 minutes, or 130.50 million in 60 minutes. These are projections on this host, not judge guarantees, and cannot be transferred directly to Linux x86 or different hardware.

Use [IMPLEMENTATION.md](../IMPLEMENTATION.md) for setup, audit, evaluation, calibration, training and packaging commands. The frozen split and confirmation ledger must remain intact. Rebuilding the selected archive is safe with `.venv/bin/python -m connectome package`; candidate selection must not reopen confirmation in this completed research round. This report can be regenerated from saved evidence with `.venv/bin/python scripts/finalize_report.py`.

- Selected model: [model.onnx](selected/model.onnx), SHA256 `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45`.
- Final archive: [submission.zip](../dist/submission.zip), SHA256 `9d8862659bbe823532879708fa2765ac023d2cae8984501919e1293a022c8de4`.
- Original fallback archive: [baseline_fallback.zip](../dist/baseline_fallback.zip).
- Machine-readable release record: [release.json](release.json).
- Frozen data/evaluation manifest: [manifest.json](manifest.json).
