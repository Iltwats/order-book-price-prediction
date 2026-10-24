# Latest validation

The current plan is `global-wp-v2.1`; its checks are recorded in [global-wp-v2.1/VALIDATION.md](global-wp-v2.1/VALIDATION.md). The Global-WP migration, rescore and original v2 checks are in [global-wp-v2/VALIDATION.md](global-wp-v2/VALIDATION.md). Earlier per-sequence evidence below is historical.

# GPU training structure validation — 2026-10-24

Completed:

- `python -m pytest tests native/test_model_io.py -q`: **256 passed, 1 skipped**, 12 existing Torch tracing/export warnings. The skipped test requires a real CUDA device.
- Synthetic CPU integration: exact interrupted/resumed model equality after relocating the plan, dataset and checkpoint; frozen lower GRU; changed plan/code/data rejection; reserved-group exclusion; qualification failure retaining the checkpoint; concurrent writer lock; interruption; private archive allowlist.
- Frozen plan: `944bb8f5b8ed7597ffc8ae801eec52fef38ac7fb325ecce713a5ed3eb1387503`. Full training file SHA256 recorded; 29,240,219,011 bytes streamed, not loaded into RAM. Two matched arms: 1024 × 4 passes and 4096 × 1 pass, 512 updates each; 512 reserved TRAIN groups.
- Real TRAIN CPU qualification passed for 8 full sequences (160,000 rows), using the selected 1024-sequence model and unchanged teacher. Actual CPU-to-CPU parameter gradient comparison was exact; this is not CUDA evidence.
- One real TRAIN CPU optimizer update completed: source lower weights remain byte-equal and upper/head weights changed. Total invocation 27.78 seconds, first batch including qualification 8.20 seconds. These are CPU smoke timings, not GPU throughput estimates.
- Real TRAIN callback/bulk parity passed, bitwise, over one complete 20,000-row sequence (row group 8384), including callback hidden states and bulk final states. No score computed.
- Private `dist/gpu_training_bundle.tar.gz` built. Extraction into a new temporary directory verified all source and frozen-plan hashes. 34 explicitly selected files; datasets, credentials, historical artifacts and Git metadata excluded.
- CLI help, Python compilation and `git diff --check` passed. Existing submission configuration and selected/released artifacts were not changed.

Pending:

- Actual CUDA raw-state/output and parameter-gradient qualification.
- GPU memory, end-to-end speed and provider cost pilot.
- Docker image build and NVIDIA target-host execution. Local Docker daemon access was denied by the execution sandbox, so no successful build is claimed.
- New prospectively bounded validation comparison, wider CPU callback/bulk parity coverage, Linux single-CPU callback feasibility and any leaderboard improvement.

This work adds training structure. It does not reopen round two, reuse consumed confirmation, provision a cloud VM, or upload a new submission. The eight-sequence CPU pilot is an engineering check, not a candidate-selection result.

## Superseded after official metric audit

The official docs checked on 2026-10-24 now specify Global WP. The plan and cloud archive described above use the old per-sequence objective. Their technical checks remain historical evidence; they do not qualify the workflow for current scoring. The GPU CLI now blocks prepare/bundle/train/qualify before work starts. See `docs/METRIC_UPDATE_AUDIT.md` and `coverage-v1/SUPERSEDED.json`. No new validation scores or training were run in this audit.
