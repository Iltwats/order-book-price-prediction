# Global-WP v2.1 plan validation — 2026-10-24

Plan `1c3a926d2a420ec70528b2517039112d05183b52f4087a08d302e08fc396a091`
replaces `global-wp-v2` after the repository cleanup removed ten unused
`connectome` modules. The removed files changed the frozen code identity; no
remaining source file changed.

Completed:

- `python -m pytest tests native/test_model_io.py -q`: **186 passed, 1 skipped**
  (the skipped test needs CUDA). The 126 removed tests covered only deleted modules.
- Plan comparison with v2: objective, parameters, seed, initial/teacher models,
  selection, training-file identity and the 512 reserved groups are identical.
  `repeat1024` is identical. `coverage4096` keeps the same 1,024 base groups and
  1,506 of its 3,072 added groups; 1,566 were redrawn because `prepare` now counts
  the untrained v2 allocations and the one-update CPU pilot as exposure. The
  reserve remains disjoint from both arms.
- Bundle `dist/gpu_training_global_wp_v2.1.tar.gz` (1,568,056 bytes, SHA256
  `cbd54311a598dd5d39747955828e6ab6c43951280640e0012c18bd38e6b7b7b4`). Extraction
  into a new directory matched all 21 frozen source hashes, with no unlisted
  Python files.
- Real TRAIN CPU qualification (`--arm coverage4096 --device cpu`) **passed**:
  8 full sequences, upper-GRU and head gradients exactly equal on CPU,
  28.8 seconds including startup. This is not CUDA evidence.
- Submission packaging from `submission_config.json` still builds and verifies
  the same eight-file archive offline.

Pending: CUDA qualification, GPU speed and memory, Docker image build,
cloud execution and any score from newly trained candidates.
