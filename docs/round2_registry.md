# Round-two experiment registry

This registry is separate from the permanently consumed confirmation workflow.
It fixes the selected calibrated GRU as comparator and allows only the existing
screen and full-development splits. It does not create an independent holdout.

```python
from connectome.experiments import (
    create_registry, register_candidate, reserve_evaluation, record_result,
)

registry = "artifacts/round2/registry.json"
create_registry(
    registry,
    manifest_path="artifacts/manifest.json",
    incumbent_config_path="submission_config.json",
    incumbent_report_path="artifacts/selected/dev_report.json",
    training_report_path="artifacts/training_pilot1024/affine_p1e-2/report.json",
    source_paths=["connectome/experiments.py", "connectome/runtime.py"],
)

# Declare before creating the candidate's configuration and training report.
spec = {
    "hypothesis": "Selected-output correlation correction",
    "training_groups": [...],  # exact groups from the affine1024 anchor report
    "seed": 20260913,
    "epochs": 0,  # closed-form/deterministic solver; neural epochs are 1..3
    "parameters": {"objective": "correlation", "max_iterations": 200,
                   "penalty": 0.01},
    "teacher_lineage": [],
    "config": {"model": "model.onnx", "backend": "onnx",
               "onnx_mode": "iobinding", "postprocess": {}},
    "config_path": "artifacts/round2/correlation/config.json",
    "training_report_path": "artifacts/round2/correlation/training.json",
    "source_paths": ["connectome/runtime.py", "connectome/experiments.py"],
    # Include every training/export/evaluation implementation used by this arm.
}
register_candidate(registry, "correlation", spec)

# Fit/export the candidate, write its exact declared config and complete report.
# Reserve immediately before running the evaluator, never after reading WP.
reserve_evaluation(registry, "correlation", "screen")
# ... evaluate exactly manifest['splits']['screen'] into screen.json ...
record_result(registry, "correlation", "screen", "artifacts/round2/correlation/screen.json")

# Only a successful screen permits a full-development reservation.
reserve_evaluation(registry, "correlation", "dev")
# ... evaluate exactly manifest['splits']['dev'] into dev.json ...
record_result(registry, "correlation", "dev", "artifacts/round2/correlation/dev.json")
```

The training report must record `complete: true` and a `specification` with
`groups`, `seed`, `source: {sha256: selected_model_hash}`, and
`data: {path, size, mtime_ns}` matching the frozen training file. Record `epochs`
for neural fits. If `parameters` is present it must equal the declared solver
configuration. New training code should always include it. Source hashes are
captured at registration and checked before evaluation and result recording.
The model, configuration, and training report are frozen on reservation; they
must remain identical between screen and full development.

Each stochastic seed or postprocess setting is a distinct candidate and uses
the shared cap. There is no implicit scale search. An alternate training subset
requires a prospective `training_subset_reason`. The registry rejects duplicate
declarations under another name/output path. Head optimization is limited to
200 iterations, and neural fits to at most three epochs.

An evaluation exception still consumes its reservation:

```python
record_result(registry, "correlation", "screen", status="failed",
              reason="Nonfinite prediction at sequence 123, step 456")
```

Failures can have no report, but require a reason and cannot be overwritten or
retried under the same reservation. A recorded `passed` status means the
evaluation completed with validated evidence; it does not mean the candidate
improved WP or passed deployment gates. The registry records incumbent-relative
delta and timing; it never certifies the hidden judge's runtime.

The shared maximum is 12 screen reservations and four full-development
reservations, including failed evaluations. File locking protects concurrent
reservations, and atomic replacement prevents partial JSON writes. The registry
is an accidental-misuse safeguard, not a security boundary against deliberately
editing files or creating independent registries. Keep one registry for this
round and preserve its lock file and all original evidence.
