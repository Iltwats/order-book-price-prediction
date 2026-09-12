# Offline training and model reuse

For the current Global-WP objective and cloud workflow, use the
[Global-WP GPU handoff](GPU_TRAINING.md). It starts from the development rescore
winner and pools time and batch for each target. The earlier methods below
remain available for historical reproduction; their sequence-normalized losses
are distinct surrogates, and their old numerical receipts do not qualify new code.

Use Python 3.11 and the pinned `requirements-train.txt`. Training is separate
from the inference-only submission. The default seed is `20260912`; every
training function is bounded by an explicit sequence limit and writes a model,
checkpoint, and JSON report under its output directory. A report's `complete`
flag means the requested bounded run completed, not that all 10,607 training
sequences were used or that the model should be submitted.

## Frozen recurrence: preferred current path

`fit_head` defaults to AdamW with learning rate `1e-3`, 4,096-row minibatches,
and a zero-initialized residual readout. Gradients accumulate for a whole
sequence before an update. Decay penalizes the residual weights, so the
pretrained head remains the reference. Test penalties `1e-4`, `1e-3`, and
`1e-2` on the same selected training sequences. `optimizer="ridge"` is a
separate, explicit closed-form alternative.

```python
from connectome.training import fit_head, fit_affine

head = fit_head(
    "baseline/baseline.onnx", "datasets/train.parquet", "artifacts/head_run",
    head_state="last", penalty=1e-3, max_sequences=256,
    encoder="onnx", threads=1,
)
affine = fit_affine(
    "baseline/baseline.onnx", "datasets/train.parquet", "artifacts/affine_run",
    penalty=1e-2, max_sequences=256, rows_per_sequence=None,
    encoder="onnx", threads=1,
)
```

The default `encoder="onnx"` clones the original graph for offline sequence
processing and exposes each GRU's existing output sequence. It retains the
original ORT recurrence, with one compute thread. The encoder must pass
complete-sequence callback parity before bulk outputs are trusted. Reports
record the source and encoder graph hashes and runtime version. On the first
real 20,000-row audit sequence, all predictions and all 5,120,000 hidden-state
values were bitwise equal to the original callback.

A last-state head or affine fit changes only the original ONNX head tensors
at export. It remains compatible with the fixed GRU128 native runtime and the
original graph's offline sequence exporter. `head_state="both"` reads the
current states of both layers; its 256-input readout is ONNX-only with the
current strict native weight format. No current row can read a future state.

## Reconstructed GRU: implemented, with an unresolved strict numerical gate

`load_torch_model` recovers both GRUs and the head from ONNX. It reorders ONNX
`z,r,h` gates into PyTorch `r,z,n` order, preserves separate input/recurrent
biases, and requires the original `linear_before_reset=1` semantics. A fixed
ONNX export reproduces the original parameter arrays bit for bit.

That parameter equivalence does not imply identical runtime arithmetic.
The recorded real-sequence audit had 1,159 of 40,000 prediction values outside
the initial `atol=1e-6, rtol=1e-5` gate, despite a WP difference of only
`1.68e-7`. Supplying the original ORT previous states to each PyTorch step
reduced this to one prediction violation. These observations support small
kernel-rounding differences amplified by recurrence; they do not constitute
a strict parity pass. Frozen fitting therefore uses original ONNX states by
default. Full recurrent fine-tuning remains an experimental branch until the
strict gate is resolved or an explicitly justified numerical criterion is
adopted. See `artifacts/training_smoke/torch_strict_parity_audit.json` for the
complete value/state counts and tolerance ratios.

`fine_tune` and `train_student(hidden_size=64 or 96)` are working, smoke-tested
implementations. They use stateful truncated backpropagation, detaching states
at chunk boundaries and resetting at sequence boundaries. Supervised loss
uses clipped targets and absolute clipped-target weights; each denominator
is computed over the full sequence after step 98. An optimizer update follows
the whole sequence batch, so chunk size does not reweight quiet periods.
Teacher matching is optional through `teacher_weight` (alias
`distill_weight`). Neither training surrogate replaces the exact validation
WP. Student exports are ONNX-only under the current native format.

## Bounded neural selection after resolving the gate

Use separate output directories for each learning-rate/teacher-weight/epoch
combination. The APIs accept `learning_rate`, `teacher_weight`, `batch_size`,
`chunk_size`, `max_sequences`, `group_indices`, and `seed`; checkpoint files
retain optimizer state, cursor, selection, data fingerprints, RNG state, and
environment versions. A small planned grid is learning rates `1e-5, 3e-5`
and teacher weights `0, 0.1`. Start each combination from the same original
weights, limit it to one epoch, and extend only promising combinations to at
most three epochs. This grid is a proposed experiment, not an established
optimum.

```python
from connectome.training import fine_tune, resume_training
from connectome.evaluation import evaluate

# Run only after the reconstruction's numerical acceptance gate is resolved.
run = fine_tune(
    "baseline/baseline.onnx", "datasets/train.parquet", "artifacts/grid/lr1e-5_tw0.1/epoch1",
    max_sequences=256, epochs=1, learning_rate=1e-5, teacher_weight=0.1,
)
score = evaluate("datasets/valid.parquet", run["model_path"], groups=development_groups)
# Keep the best exact development-WP artifact; do not select by training loss.
continued = resume_training(
    run["checkpoint_path"], "artifacts/grid/lr1e-5_tw0.1/epoch2", epochs=2,
)
```

Here `development_groups` is the frozen development split from the manifest,
not the confirmation split. Preserve every epoch's files, score each epoch,
stop a combination after a non-improving epoch, and never extend this bounded
grid past epoch three. Choose the checkpoint with the best exact development
WP across the permitted combinations, then apply the project's final-selection
and confirmation guard. `resume_training(..., epochs=N)` interprets `N` as
the total desired epoch count. `max_updates` is likewise a total update bound;
linear fitting's `max_new_sequences` bounds additional sequences for that call.

`export_onnx(..., dynamic=True)` emits a time-major offline model with inputs
`[time,batch,112]`. It is different from the original graph's batch-first
offline clone; do not interchange their input layouts. Default training
exports retain the fixed one-row judge interface.
