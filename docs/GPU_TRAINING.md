# Global-WP GPU training handoff

The existing 1,024-sequence correlation model remains our development winner
after rescoring with **Global Weighted Pearson**. The new GPU plan starts from
that model and uses it as the fixed teacher. It tests whether training on more
sequences helps more than repeating the existing subset.

The scorer pools every selected row across sequences, computes one clipped
weighted correlation per target, and averages those two correlations. Earlier
per-sequence scores belong to a different metric. See the
[metric reference](../METRIC.md).

## Starting point from the rescore

These scores use the same **1,400 development sequences** and official scoring
mask. They are not public leaderboard scores or new training results.

| Existing model | Development Global WP |
| --- | ---: |
| Correlation-1024, seed 20260912 | **0.672091945** |
| Correlation-1024, seed 20260913 | 0.672078182 |
| Earlier selected calibrated model | 0.671730101 |
| Round-two affine correlation head | 0.671587454 |

The winning model gains **0.000361845** over the earlier selected model. Its
paired sequence-cluster bootstrap interval is `[+0.000219533, +0.000492291]`.
Those intervals recompute pooled correlations and do not remove model-selection
bias. The two closely scored 1,024-sequence runs do not establish that one seed
is generally better.

The complete ranking and model/configuration hashes are saved in
`artifacts/global_wp/rescore_v1/ranking.json`. The signed-in submission history checked on October 24 shows **0.65151** for
`submission_correlation1024.zip`. This concerns hidden data and remains separate
from these development measurements. The consumed confirmation set stays closed.

## Training method and allocation

`connectome/gpu_training.py` uses plan format
`connectome.gpu-training.global-wp.v2`. CPU ONNX Runtime generates the current
student's complete causal trajectories. `connectome/global_objectives.py` pools
all post-warm-up rows over both time and batch, separately for each target, and
calculates the loss and upstream coefficients in FP64. Torch computes local
upper-GRU/head gradients on CUDA in FP32. The lower GRU stays frozen, and export
retains the original CPU inference graph.

Official Global WP pools the whole evaluated dataset under its scoring mask.
The loss pools one TRAIN batch without an official mask, so it remains a
training surrogate. Correlation is not additive across batches. The coefficients
already include the two-target mean; chunk backward passes sum their
contributions without another batch or chunk normalization.

Preparation verifies the completed Global-WP ranking and reports, then freezes
the winner's weights and postprocessing. The current winner uses identity
postprocessing. Supported per-target scales are preserved through the loss chain
rule and each run's exported `config.json`.

Both arms use batch size 8, a 256-row backward window, learning rate `1e-5`,
teacher weight `0.1`, relative parameter-anchor weight `0.001`, weight decay
`1e-4`, and gradient clipping at 1. The lower learning rate is a conservative
starting choice for the revised loss, not a validated optimum.

| Arm | Distinct fitting sequences | Passes | Optimizer updates | Purpose |
| --- | ---: | ---: | ---: | --- |
| `repeat1024` | 1,024 | 4 | 512 | Measure more updates on the existing subset |
| `coverage4096` | 4,096 | 1 | 512 | Measure broader coverage at equal update count |

The larger arm contains the same 1,024 groups plus 3,072 others. The plan
preserves the earlier 512-sequence TRAIN reserve and excludes it from both arms.
Recorded fitting groups, earlier GPU plans and run histories are inventoried.
This cannot prove that every old diagnostic was captured or that the supplied
pretrained model never saw the reserve. Keep it sealed while choosing settings.

Qualify the device first, then run a bounded timing pilot. If both pass, run the
two declared arms. The plan permits one existing-screen evaluation and one
full-development evaluation per completed arm, with no epoch or postprocessing
search. It does not reopen the old round or confirmation ledger. Longer backward
windows and lower-layer adaptation require a separate declared experiment.

## Prepare and package locally

Use Python 3.11 and the existing environment:

```sh
.venv/bin/python -m connectome.gpu_training prepare \
  --train datasets/train.parquet \
  --ranking artifacts/global_wp/rescore_v1/ranking.json \
  --output artifacts/gpu/global-wp-v2.1

.venv/bin/python -m connectome.gpu_training bundle \
  --plan artifacts/gpu/global-wp-v2.1 \
  --output dist/gpu_training_global_wp_v2.1.tar.gz
```

Preparation freezes the ranking, model hashes, full training-file SHA256,
relative source hashes, objective version, seed, allocations and reserved groups.
Hashing the roughly 27 GiB training file streams it rather than loading it into
RAM. Preparation reads metadata and completed rescore reports, and performs no
training or fresh validation evaluation.

The private archive contains source, pinned requirements, Dockerfile, this
guide, the plan, selection evidence, and the initial/teacher ONNX models. It
excludes datasets, credentials, Git history and prediction caches. Transfer
`train.parquet` separately to your own machine or private storage. The fitting
job does not need `valid.parquet`.

The current plan is `artifacts/gpu/global-wp-v2.1`, plan SHA256
`1c3a926d2a420ec70528b2517039112d05183b52f4087a08d302e08fc396a091`. It replaced
`global-wp-v2` after the repository cleanup removed unused modules, which changes
the frozen source hashes. Settings, reserve and `repeat1024` are identical;
`coverage4096` keeps the same 1,024 base groups but redrew 1,566 of its 3,072
added groups, because `prepare` counts the earlier plan's allocations as
exposure. Neither v2 arm had been trained. The legacy per-sequence plan under
`artifacts/gpu/coverage-v1` is rejected by the current runner.

## Launch a single-GPU pilot

Use a Linux x86-64 NVIDIA host with a working driver and NVIDIA Container
Toolkit. Start with a low-cost single-GPU machine, about 16 GB host RAM and at
least 80 GB disk for data, packages, image and outputs. This is a starting
allocation, not a measured requirement. Measure GPU memory and total runtime
before buying a larger machine or committing to a long run.

```sh
mkdir -p connectome-training
tar -xzf gpu_training_global_wp_v2.1.tar.gz -C connectome-training
cd connectome-training
docker build -f cloud/Dockerfile -t connectome-train:global-wp-v2.1 .
```

The image uses Python 3.11, the pinned project requirements, and the official
[PyTorch 2.5.1 CUDA 12.4 wheel index](https://pytorch.org/get-started/previous-versions/).
Building requires network access. Once the image, plan and dataset are present,
fitting runs offline. Keep credentials on the host, outside the image. Dataset
and cache directories are excluded from the build context.

These examples use `/mnt/connectome/data/train.parquet` and persistent output
storage at `/mnt/connectome/runs`. Adjust the host paths to the attached disk or
volume, and mount the bundle's plan read-only:

```sh
docker run --rm --init --gpus all --network none \
  -v "$PWD/plan:/plan:ro" \
  -v /mnt/connectome/data:/data:ro \
  -v /mnt/connectome/runs:/runs \
  connectome-train:global-wp-v2.1 qualify \
  --plan /plan --train /data/train.parquet \
  --arm coverage4096 --output /runs/qualification

docker run --rm --init --gpus all --network none \
  -v "$PWD/plan:/plan:ro" \
  -v /mnt/connectome/data:/data:ro \
  -v /mnt/connectome/runs:/runs \
  connectome-train:global-wp-v2.1 train \
  --plan /plan --train /data/train.parquet \
  --arm coverage4096 --output /runs/coverage4096 --max-updates 1
```

Qualification performs no optimizer update. It checks uncorrected local hidden
states and outputs against the existing error limits, then compares device
parameter gradients with CPU gradients on the same batch and window. Corrected
forward equality alone is insufficient. AMP and TF32 are disabled. A passing
receipt covers the tested runtime, weights, batch and window; it does not resolve
full carried-Torch versus ONNX parity.

The first training update includes qualification. `report.json` records elapsed
time, rows per second, peak allocated CUDA memory, runtime versions and hashes.
Startup timing also includes full-file verification. Use a bounded multi-update
pilot for a steadier estimate. CPU ONNX trajectory generation remains part of
every update, so faster CUDA kernels may have little effect on total runtime.
GPU speedup is currently unmeasured.

To resume the same arm, run the training command again and replace its last line
with:

```sh
  --arm coverage4096 --output /runs/coverage4096 --resume
```

Run `repeat1024` in its own output directory. `--max-updates` is an absolute
total, not an additional count. `--max-seconds` stops another batch from starting
after a soft time budget; file verification, one batch and export can exceed it.
Each invocation requalifies its first batch before updating. Change settings
only through a new frozen plan. `--device cpu` exercises local plumbing; the
default requires CUDA and does not silently fall back to CPU.

## GCP and lower-cost hosts

For GCP, attach a persistent disk configured to survive instance deletion and
keep another checkpoint copy outside the VM. Spot capacity is an option after
a successful pilot. Its default shutdown period is best effort and up to
30 seconds, so final shutdown syncing cannot be the only backup. See
[GCP Spot VMs](https://docs.cloud.google.com/compute/docs/instances/spot) and
[shutdown scripts](https://docs.cloud.google.com/compute/docs/shutdownscript).

Use the same container layout on a rented CUDA host such as Runpod. A network
volume persists independently of a Pod. Container storage is temporary, while
a Pod volume is retained across stops but deleted with the Pod. See
[Runpod storage](https://docs.runpod.io/pods/storage/types). Check availability
and current total costs when selecting the host; this guide makes no price
estimate. Include CPU, storage and transfers when comparing pilot costs.

The runner atomically saves model, optimizer, RNG, cursor and history after
each completed batch. SIGTERM/SIGINT requests a stop without committing a partial
batch. A forced kill can lose the current batch; resume replays it from the last
checkpoint. Use storage with POSIX locks and atomic rename, and periodically
copy completed checkpoint files to private backup storage from the host.

Content hashes permit moving the plan, dataset and run directory even when paths
or mtimes change. The receiving checkout must have identical source bytes. CPU
resume equality is covered by tests. Different GPU hardware or CUDA/cuDNN
versions can change arithmetic, so cross-hardware bitwise equality is not
promised; sessions record the runtime and requalify it. Load only checkpoints
produced by this runner.

## Return and evaluate the trained artifacts

Copy both run directories back. Each contains `checkpoint.pt`,
`qualification.json`, `report.json`, `model.onnx`, the offline-only
`sequence.onnx`, and the matching deployment `config.json`.

Compare each completed arm under the frozen Global-WP evaluation allocation.
Require complete-sequence callback/bulk parity, common masked scoring domains,
pooled sequence-cluster uncertainty and Linux single-CPU callback timing. Do not
select by training loss or reuse consumed confirmation. Training GPU throughput
does not establish CPU submission feasibility. The runner does not upload a
competition submission.

## Validation status

The Global-WP development rescore is complete. The pooled loss has independent
autograd and finite-difference tests. CPU runner tests cover resume after
relocation, frozen lower weights, changed inputs, reserved groups, qualification
failure, concurrent writers and interrupted batches. The session receipt will
record the bounded real-data CPU pilot separately.

CUDA qualification, measured GPU speedup, Docker image build, target-host cloud
execution and any score improvement from the new pooled-loss training remain
pending. Read `artifacts/gpu/VALIDATION.md` for the checks actually completed in
this checkout before launching a cloud job.
