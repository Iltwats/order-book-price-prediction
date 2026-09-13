# Round-two conversion audit

Both the original supplied model and the selected calibrated model passed a zero-update Torch-parameter export into their original ONNX graphs. Each exported file is byte-identical to its own source. Independent FP64 equation and gradient checks also passed. **The existing strict Torch-versus-ORT FP32 parity check remains failed; these results do not override it.**

## Actual model roundtrips

[`export_original_graph`](../connectome/conversion_audit.py) reads the reconstructed Torch parameters and writes all eight ONNX weight groups: input weights, recurrent weights and combined biases for both GRUs, plus head weights and bias. It reverses the Torch `[r,z,n]` to ONNX `[z,r,h]` permutation and transposes the head. This is a real parameter conversion, not a source-file copy. Unit tests change every Torch parameter and verify the resulting ONNX arrays.

The exporter preserves graph nodes, attributes, tensor metadata and FP32 types, including `linear_before_reset=1`. It rejects incompatible architectures, nonfinite or non-FP32 parameters and source-overwrite aliases. All 192,258 parameters were bitwise preserved in both zero-update exports.

| Source | Source and exported SHA-256 | Development coverage | Result |
|---|---|---|---|
| Original | `321ec67b93f3c5dde2c0901ca4532072dad5f65753c2e6519e93ea3555f6a6fd` | 32 complete sequences; 640,000 rows | File, parameters, states and predictions bitwise equal |
| Selected | `d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45` | Same 32 sequences; 640,000 rows | File, parameters, states and predictions bitwise equal |

For each source, the audit compared 163,840,000 hidden-state values and 1,280,000 predictions, including warm-up rows, with zero discrepancies. It used the existing original-graph bulk ORT encoder to compare source and roundtrip over full 20,000-row sequences. This was not another full Python callback benchmark or a new Torch-versus-ORT comparison. Existing bulk/callback verification remains in [`test_training.py`](../tests/test_training.py) and the [selected parity report](../artifacts/selected/parity_smoke32.json).

The [original report](../artifacts/round2/conversion/original/report.json) and [selected report](../artifacts/round2/conversion/selected/report.json) record model, code and manifest hashes, sequence IDs, per-output comparisons and encoder provenance. The [summary](../artifacts/round2/conversion/summary.json) records 12.28 and 12.15 seconds respectively. These are local conversion-audit timings, not callback throughput or judge-runtime projections. Confirmation data was not used.

## Equation and gradient evidence

The [FP64 report](../artifacts/round2/conversion/fp64_gradients.json) compares an explicit per-gate GRU reference against Torch on three controlled cases: one step, five steps, and five steps with saturated gates. Each uses width two, three input features, batch size two, nonzero initial state and distinct input/recurrent candidate biases. Input, initial-state and all four parameter gradients agree; both implementations pass finite-difference checks.

Maximum absolute forward and gradient errors were `2.22e-16` and `6.66e-16`. Prespecified tolerances were `atol=1e-12, rtol=1e-10` for forward values and `atol=1e-10, rtol=1e-8` for gradients. Finite differences used `eps=1e-6, atol=1e-6, rtol=1e-4`. These small controlled cases check equations and differentiation; they do not establish FP32 equivalence of the full GRU-128 over long sequences.

## Remaining gate and reproduction

The unchanged [strict FP32 audit](../artifacts/training_smoke/torch_strict_parity_audit.json) still reports 1,159 prediction violations across 40,000 values on sequence `256678`, at `atol=1e-6, rtol=1e-5`; maximum prediction error is `1.1921e-5`. Hidden-state comparisons also fail. Different floating-point kernels remain a plausible explanation, not a demonstrated strict pass. No tolerance was relaxed. Recurrent fine-tuning needs a separately justified prospective acceptance procedure or resolution of that discrepancy. Frozen-head fitting can use the original ORT hidden states directly.

From the repository root, in the pinned environment:

```sh
.venv/bin/python -m pytest -q tests/test_conversion_audit.py tests/test_training.py
.venv/bin/python -m connectome.conversion_audit \
  --manifest artifacts/manifest.json \
  --output artifacts/round2/conversion_recheck
```

The completed test run passed all 22 tests. The separate output directory preserves the recorded audit. Generated ONNX files and reports remain under ignored artifacts; neither supplied model was changed.
