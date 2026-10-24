# Global-WP v2 validation — 2026-10-24

Completed:

- Refreshed official `METRIC.md` recovered from a bounded 2 MiB prefix read of the public ZIP. Verified Global WP, no per-sequence eligibility, clipped absolute-target weights, FP32-compatible input/FP64 metric, 1e-8 floors and centered weighted-moment merging.
- Current default callback/cache/offline/calibration scorers use `global_wp_clip2_v1`. Historical block scoring remains explicitly available. Selectors, bootstrap and confirmation guards reject mixed metric versions.
- Full suite: **312 passed, 1 skipped**, 12 existing Torch tracing/export warnings. The skipped CUDA integration test requires a real GPU.
- Declared and completed one fixed comparison of 12 existing configurations on 1,400 development sequences. Confirmation excluded; no new calibration or epoch search. Best development Global WP **0.6720919452834274**, correlation1024_seed20260912.
- Paired sequence-cluster bootstrap versus earlier selected model: delta +0.000361844718973, 95% interval [+0.000219532901, +0.000492291084]. Development selection bias is not removed.
- Winner callback/bulk bitwise parity passed on all 32 frozen smoke sequences, including all callback hidden states and bulk final states (640,000 rows). Four inference-only rescore paths each passed an initial full-sequence parity check.
- New pooled objective passed independent FP64 autograd, finite-difference, warm-up, clipping, one-sign and pooling/normalization checks. Synthetic training resumes exactly after relocation, including optimizer and RNG state.
- Real TRAIN CPU pilot completed 1 update on 8 complete sequences (160,000 rows); objective version is `batch_pooled_global_correlation.v1`. Frozen lower weights stayed byte-equal; upper/head weights changed. Pilot export callback/bulk byte parity passed on one full TRAIN sequence. No pilot validation score was computed.
- Real pilot invocation: 29.26 s including full-file verification; first batch including CPU qualification: 8.56 s. These are CPU plumbing timings, not GPU speed estimates.
- Preserved the original 512 reserved TRAIN groups and closed confirmation. New plan and checkpoint format reject historical per-sequence plans; selection receipt/model/code/content hashes are bound.
- Plan `3723823f45de3395f4c12962230645515c20fbd4362c438619dfd4e07d0c10c4`; initialization/teacher SHA256 `15c3dad6c0f9964df7489c5dc8c3b5d5774907d6dc33ac80d8fe5533a61f6cde`. Teacher and initialization are identical and come from the Global-WP winner.
- Private bundle `gpu_training_global_wp_v2.tar.gz`, 1,595,619 bytes, SHA256 `082346a5a2b7430879849319245631adabbf8e7a1484bcd4c62785a86ea3c503`. Includes frozen plan/models, selection receipt, source and current handoff guide; excludes datasets, credentials and Git history.

Pending:

- Real CUDA raw-state/output and parameter-gradient qualification.
- Measured GPU memory, end-to-end speed and total provider cost.
- Docker build and Linux NVIDIA target-host/cloud execution.
- Completed training arms, their prospectively bounded development comparisons, Linux single-CPU callback feasibility and any new public/private score.

The signed-in site now displays public WP **0.65151** for `submission_correlation1024.zip`. No submission was uploaded. Existing submission configuration and historical experiment artifacts remain unchanged. Post-run cache hash/runtime evidence is in `artifacts/global_wp/rescore_v1/evidence.json`; cached array hashes were recorded after scoring rather than prospectively.
