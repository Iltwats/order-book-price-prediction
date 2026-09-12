# Implemented pretrained GRU workflow

The implementation starts from the supplied **112 → GRU-128 → GRU-128 → linear-2** model. `baseline/` remains unchanged. The production callback is `solution.PredictionModel()`; research, training and packaging are available through `python -m connectome`. See [the measured results](artifacts/IMPLEMENTATION_REPORT.md) for the selected model, accuracy, performance, and release limitations.

## Environment and reproducibility

Use Python 3.11. Inference requirements are pinned in `requirements.txt`; training/export/test dependencies are pinned in `requirements-train.txt`. Training dependencies are never copied into the submission.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-train.txt
.venv/bin/python -m pytest tests native/test_model_io.py -q
.venv/bin/python -m connectome --help
```

The installed local environment uses Python 3.11.16. A dependency image is built once with networking; all Linux benchmark and verification containers run offline. The judge never installs or compiles anything. [Linux runner instructions](docs/linux_validation.md) describe Docker limits, image provenance and the architecture distinction.

## Data and scoring

`connectome/data.py` reads complete Parquet row groups, validates the 112-column feature order and metadata, and streams one sequence at a time. The audit checks all sequence IDs and train/validation overlap. Full feature validation occurs whenever a sequence is consumed; the initial audit samples feature payloads rather than loading the complete dataset.

```sh
.venv/bin/python -m connectome audit --output artifacts/manifest.json
.venv/bin/python -m connectome prepare --output artifacts/models
```

`artifacts/manifest.json` freezes 1,400 development and 473 confirmation groups with seed `20260912`. Previously inspected groups remain in development. Its nested 32- and 256-sequence subsets serve correctness and screening. Do not regenerate this manifest during an experiment round. Dataset fingerprints include size, metadata and sampled content hashes; they are explicitly not full 34 GB content checksums.

The evaluator imports the supplied `BlockAccumulator` directly. It clips predictions and targets to ±2, uses absolute clipped-target weights, masks required/scored rows, excludes ineligible sequences, and averages both targets and eligible sequences equally. It saves every sequence's score and uses paired sequence bootstrap intervals with 10,000 resamples. Training uses a sequence-normalized weighted-MSE surrogate; training loss is not the competition metric.

Raw prediction caches contain every row, including warm-up, are bound to model/runtime provenance, and become readable only when finalized. Calibration searches read labels and predictions without re-running the GRU. Confirmation is reserved for the frozen winner; see the CLI's freeze and confirmation safeguards.

## ONNX execution and validation

`connectome/runtime.py` implements four modes: original output lookup (`reference`), cached names/dictionary (`run`), persistent CPU I/O binding (`iobinding`), and the pinned OrtValue vector API (`ortvaluevector`). The two binding variants alternate owned hidden-state buffers. Every returned prediction owns stable storage. Every input advances both GRUs, warm-up returns `None`, and any new sequence resets recurrent and filter state.

The fixed-step graph removes redundant shape/sequence-output operations while preserving the GRU equations and weights. The separate dynamic-sequence graph accelerates **offline evaluation only**. Its rows/second are not callback throughput or a claim about judge runtime. Validate the exact artifact pair before using the offline graph:

```sh
.venv/bin/python -m connectome parity --split smoke --original baseline/baseline.onnx --model artifacts/models/simplified.onnx --bulk-model artifacts/models/sequence.onnx --mode ortvaluevector --output artifacts/parity_baseline.json
.venv/bin/python -m connectome evaluate --split dev --bulk-model artifacts/models/sequence.onnx --parity-report artifacts/parity_baseline.json --cache artifacts/cache/reproduced_baseline_dev --output artifacts/reproduced_baseline_dev.json
.venv/bin/python -m connectome calibrate --split screen --cache artifacts/cache/reproduced_baseline_dev --output artifacts/reproduced_calibration.json
```

Use an unused cache directory for a fresh run. For ordinary streamed callback evaluation, omit `--bulk-model` and `--parity-report`. `check_parity` compares predictions and both hidden states at every step using bounded per-sequence buffers. Wrapper-only changes must also remain bitwise equal. Arithmetic rewrites start at `atol=1e-6, rtol=1e-5`; failures remain failures until investigated.

## Native experiment

`native/connectome_native.cpp` provides a C++17 model context with a C ABI, called once per row through `ctypes`. Its `create/reset/step/destroy` functions own aligned packed weights, hidden states and scratch space. ONNX gates are explicitly `[z,r,h]`, with `linear_before_reset=1`; Torch's different gate order is handled only by the offline training importer. Python transfers a contiguous FP32 feature pointer synchronously and returns an owned two-value array.

The implementation uses standard sigmoid/tanh, portable release flags, no fast-math, no `-march=native`, no worker threads and no per-row heap allocation. Row-oriented and gate-packed weight layouts are measured independently. Native startup incompatibility can select ONNX before any input is consumed; runtime errors propagate instead of silently switching state or returning zeros.

```sh
.venv/bin/python native/build.py --help
.venv/bin/python native/verify_backend.py --help
.venv/bin/python scripts/run_linux_checks.py --build --phase prepare
.venv/bin/python scripts/run_linux_checks.py --phase benchmark --smoke-count 32 --repetitions 3
```

The native backend requires both strict numerical checks and a repeatable ≥10% complete-callback speedup to be promoted. It currently remains experimental. Do not include an unverified architecture library in a release. Build and sanitizer evidence is under `artifacts/native` and `artifacts/linux_arm64`.

Affinity reads the process's allowed CPU set and selects one permitted logical CPU only inside a benchmark/submission process. Unsupported or denied affinity is nonfatal. Deployment affinity is off by default; it neither reserves a physical core nor changes the CPU quota. Cache counters are reported as unavailable when profiling is unavailable, not as zero misses.

## Reusing pretrained weights for score improvements

The first search tests per-target scales `{0.5,0.75,1,1.25,1.5,2}` and identity plus EMA half-lives `{2,8,32}` with blends `{0.1,0.25}`. EMA consumes warm-up rows and resets by sequence. Identity wins numerical ties. `connectome/transforms.py` folds compatible scale/affine changes into the final layer without adding inference operations; each folded model still requires parity checks because FP32 reassociation can matter.

Frozen-head training uses the **original ONNX GRUs** as the default encoder, exposing both hidden-state sequences one group at a time. This avoids a full hidden-state cache and avoids importing Torch's recurrent numerical differences into an otherwise frozen model. Last-state head exports replace only the original final-layer tensors. Both-state heads use a separately exported graph and remain ONNX-only.

```sh
.venv/bin/python -m connectome fit-affine --max-sequences 256 --penalty 0.01 --output artifacts/my_affine
.venv/bin/python -m connectome fit-head --max-sequences 256 --head-state last --penalty 0.001 --learning-rate 0.001 --minibatch-size 4096 --output artifacts/my_head
.venv/bin/python -m connectome fit-head --max-sequences 256 --head-state both --penalty 0.001 --output artifacts/my_both_head
```

The affine fit is a residual ridge regression shrunk toward identity. The AdamW head starts with zero correction and keeps both GRUs frozen. Minibatches bound activation memory; loss denominators are computed over full post-warm-up sequences. The Python API also provides a closed-form residual-head ridge fit. Omit the sequence limit only when intentionally fitting the full training set. Reports record actual groups, seeds, sampling and optimizer settings.

## Conditional recurrent fine-tuning and students

The trainable Torch importer exactly maps the supplied weights and gate biases. Its local long-sequence FP32 results do not yet satisfy the strict recurrent parity gate. Full recurrent fine-tuning is therefore not a production release candidate in this round. The working training/export paths have bounded real-data smoke tests, exact checkpoint-resume tests, and explicit parity diagnostics.

The tools support FP32 carried-state truncated backpropagation, detached state between chunks, full-sequence loss denominators, gradient clipping 1.0, weight decay `1e-4`, teacher matching, and width-64/96 students. These commands illustrate bounded runs **after resolving the parity prerequisite**:

```sh
.venv/bin/python -m connectome fine-tune --max-sequences 32 --epochs 1 --batch-size 8 --chunk-size 256 --learning-rate 0.00001 --teacher-weight 0.1 --output artifacts/my_finetune
.venv/bin/python -m connectome distill --max-sequences 32 --epochs 1 --hidden-size 96 --batch-size 8 --chunk-size 256 --output artifacts/my_student
```

For the planned recurrent grid, compare learning rates `{1e-5,3e-5}` and teacher penalties `{0,0.1}`, retain the best development epoch within three epochs, and repeat the winning setup with a second seed. No training loss alone qualifies a model for release. Students and ensembles require separate callback feasibility measurements. Generic GRU quantization remains deferred because the pinned ORT quantization registry lacks a GRU quantizer.

## Release

`submission_config.json` selects the model/backend and postprocessing. Freeze the chosen configuration only after development comparison and performance checks. Evaluate the frozen winner once on all confirmation groups, compare paired sequence differences with the original baseline, and keep the original baseline if mean improvement is nonpositive. No tuning follows confirmation within this round.

```sh
.venv/bin/python -m connectome freeze --help
.venv/bin/python -m connectome package --config submission_config.json --output dist/submission.zip
```

Packaging copies inference code, selected weights, configuration and file hashes only. It verifies an extracted archive in a fresh isolated Python process with networking disabled, checking warm-up, resets, determinism, output ownership and native-startup fallback. It publishes a replacement ZIP only after verification succeeds and rejects archives above the 18,000,000-byte engineering limit. The delivered ZIP is not uploaded by these tools.

The unknown hidden-test volume and CPU architecture prevent claiming the 60-minute judge limit has been proved. Linux ARM64 measurements on this Mac, the exact container limits, peak memory and callback rates are reported separately. The runtime projection must stay below 45 minutes once platform workload information becomes available.

## Primary technical references

- [ONNX GRU equations and gate ordering](https://onnx.ai/onnx/operators/onnx__GRU.html)
- [ONNX Runtime graph optimizations](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
- [ONNX Runtime I/O binding](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)
- [Pinned ORT 1.23.2 quantization registry](https://raw.githubusercontent.com/microsoft/onnxruntime/v1.23.2/onnxruntime/python/tools/quantization/registry.py)
- [Python 3.11 affinity APIs](https://docs.python.org/3.11/library/os.html#os.sched_setaffinity)
- [Docker CPU and memory controls](https://docs.docker.com/engine/containers/resource_constraints/)
- [PyTorch GRU definition](https://docs.pytorch.org/docs/2.5/generated/torch.nn.GRU.html)
