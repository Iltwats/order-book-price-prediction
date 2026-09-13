"""Synthetic-testable dynamics residual, appended to the supplied ONNX graph.

The original feature path, nodes and initializers are preserved. This module
does not train a prediction model, read datasets, or register score candidates.
Its four-input graph needs DynamicsBackend; the existing two-state loader is
deliberately unchanged. Normalization callers must supply training-only rows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np

WIDTH = 16
AUX_WIDTH = 33
ALPHA = np.float32(-np.expm1(-np.log(2.0) / 8.0))
CONTRACT = "connectome.dynamics.v1"
INPUT_NAMES = ["features", "hidden_0", "hidden_1", "branch_state"]
OUTPUT_NAMES = ["branch_prediction", "next_hidden_0", "next_hidden_1", "next_branch_state"]
INPUT_SHAPES = [(1, 1, 112), (1, 1, 128), (1, 1, 128), (1, 1, AUX_WIDTH)]
OUTPUT_SHAPES = [(1, 1, 2), *INPUT_SHAPES[1:]]


def _array(value, shape, name):
    result = np.array(value, dtype=np.float32, order="C", copy=True)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite FP32 with shape {shape}")
    result.setflags(write=False)
    return result


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class TrainingNormalization:
    mean: np.ndarray
    scale: np.ndarray
    provenance: dict = field(default_factory=lambda: {"kind": "identity", "rows": 0})

    def __post_init__(self):
        object.__setattr__(self, "mean", _array(self.mean, (112,), "mean"))
        object.__setattr__(self, "scale", _array(self.scale, (112,), "scale"))
        if np.any(self.scale <= 0):
            raise ValueError("normalization scales must be positive")
        object.__setattr__(self, "provenance", json.loads(json.dumps(self.provenance, allow_nan=False)))

    @classmethod
    def identity(cls):
        return cls(np.zeros(112, np.float32), np.ones(112, np.float32))

    def record(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(),
                "provenance": self.provenance,
                "statistics_sha256": hashlib.sha256(self.mean.tobytes() + self.scale.tobytes()).hexdigest()}


def fit_training_normalization(batches, *, split, groups, data_fingerprint):
    """Bounded FP64 moment fit over declared training feature batches [rows,112].

    No labels or validation masks are accepted. The caller is responsible for
    proving that the supplied batches belong to these training groups; this
    function records that declaration rather than guessing file provenance.
    """
    groups = list(groups)
    if split != "train":
        raise ValueError("normalization may be fitted only on the training split")
    if not groups or any(type(g) is not int or g < 0 for g in groups) or len(set(groups)) != len(groups):
        raise ValueError("declare unique nonnegative training groups")
    if not isinstance(data_fingerprint, dict) or not data_fingerprint:
        raise ValueError("training data fingerprint is required")
    count = 0
    mean = np.zeros(112, np.float64)
    m2 = np.zeros_like(mean)
    for batch in batches:
        x = np.asarray(batch, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 112 or not np.isfinite(x).all():
            raise ValueError("normalization requires finite [rows,112] feature batches")
        if len(x) == 0:
            continue
        batch_mean = x.mean(axis=0)
        centered = x - batch_mean
        batch_m2 = np.einsum("ij,ij->j", centered, centered)
        combined = count + len(x)
        difference = batch_mean - mean
        m2 += batch_m2 + difference * difference * (count * len(x) / combined)
        mean += difference * (len(x) / combined)
        count = combined
    if count == 0:
        raise ValueError("normalization requires at least one training row")
    std = np.sqrt(np.maximum(m2 / count, 0))
    small = std < 1e-6
    scale = np.where(small, 1.0, std)
    return TrainingNormalization(mean, scale, {
        "kind": "training_population_moments", "split": "train", "rows": count,
        "groups": groups, "data_fingerprint": data_fingerprint,
        "near_constant_columns": np.flatnonzero(small).tolist(), "std_floor": 1e-6,
        "near_constant_scale": 1.0, "accumulation_dtype": "float64"})


@dataclass(frozen=True)
class DynamicsParameters:
    projection_weight: np.ndarray
    projection_bias: np.ndarray
    hidden_weight: np.ndarray
    hidden_bias: np.ndarray
    output_weight: np.ndarray
    output_bias: np.ndarray
    normalization: TrainingNormalization = field(default_factory=TrainingNormalization.identity)

    def __post_init__(self):
        shapes = {"projection_weight": (112, 16), "projection_bias": (16,),
                  "hidden_weight": (48, 16), "hidden_bias": (16,),
                  "output_weight": (16, 2), "output_bias": (2,)}
        for name, shape in shapes.items():
            object.__setattr__(self, name, _array(getattr(self, name), shape, name))
        if not isinstance(self.normalization, TrainingNormalization):
            raise ValueError("normalization must be a TrainingNormalization")


def make_fixture(*, seed=20260912, nonzero=False, normalization=None):
    """Deterministic untrained fixture; never a fitted or scored candidate.

    Only the output layer is zero in the identity fixture. A small nonzero
    output fixture keeps residual MLP work observable during later benchmarks.
    """
    rng = np.random.default_rng(seed)
    def xavier(inputs, outputs):
        bound = np.sqrt(6.0 / (inputs + outputs))
        return rng.uniform(-bound, bound, (inputs, outputs)).astype(np.float32)
    projection, hidden = xavier(112, 16), xavier(48, 16)
    output = xavier(16, 2) * np.float32(0.05) if nonzero else np.zeros((16, 2), np.float32)
    return DynamicsParameters(projection, np.zeros(16, np.float32), hidden,
                              np.zeros(16, np.float32), output, np.zeros(2, np.float32),
                              normalization or TrainingNormalization.identity())


def folded_projection(parameters):
    """FP32 affine folding; arithmetic rounding is verified, not assumed exact."""
    normalization = parameters.normalization
    weight = np.ascontiguousarray(parameters.projection_weight / normalization.scale[:, None])
    bias = np.ascontiguousarray(parameters.projection_bias - normalization.mean @ weight)
    if not np.isfinite(weight).all() or not np.isfinite(bias).all():
        raise ValueError("normalization folding produced nonfinite parameters")
    return weight, bias


class DynamicsReference:
    """Independent NumPy branch recurrence; step returns only the residual."""

    def __init__(self, parameters, *, folded=False):
        self.parameters = parameters
        self.folded = folded
        self.weight, self.bias = (folded_projection(parameters) if folded else
                                 (parameters.projection_weight, parameters.projection_bias))
        self.state = np.zeros(AUX_WIDTH, np.float32)
        self.last_features = None

    def reset(self):
        self.state.fill(0)
        self.last_features = None

    def step(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.shape != (112,) or not np.isfinite(x).all():
            raise ValueError("reference expects finite [112] features")
        p = self.parameters
        if not self.folded:
            x = (x - p.normalization.mean) / p.normalization.scale
        u = x @ self.weight + self.bias
        previous = self.state[:16] if self.state[-1] > 0.5 else u
        mean = self.state[16:32] if self.state[-1] > 0.5 else u
        delta, innovation = u - previous, u - mean
        q = np.concatenate((u, delta, innovation))
        next_mean = mean + ALPHA * innovation
        hidden = np.maximum(q @ p.hidden_weight + p.hidden_bias, np.float32(0))
        residual = hidden @ p.output_weight + p.output_bias
        self.state[:16], self.state[16:32], self.state[32] = u, next_mean, 1
        self.last_features = q.copy()
        return residual.copy()


def normalization_folding_diagnostic(parameters, rows, *, atol=1e-6, rtol=1e-5):
    """Expose FP32 folding differences instead of silently loosening parity.

    FP64 algebra checks isolate coefficient/layout errors from operation-order
    rounding. Passing this bounded diagnostic is not full-sequence validation.
    """
    if atol < 0 or rtol < 0 or not np.isfinite([atol, rtol]).all():
        raise ValueError("folding tolerances must be nonnegative and finite")
    folded, unfolded = DynamicsReference(parameters, folded=True), DynamicsReference(parameters, folded=False)
    p = parameters
    mean, scale = p.normalization.mean.astype(np.float64), p.normalization.scale.astype(np.float64)
    projection, bias = p.projection_weight.astype(np.float64), p.projection_bias.astype(np.float64)
    weight64 = projection / scale[:, None]
    bias64 = bias - mean @ weight64
    result = {"rows": 0, "atol": atol, "rtol": rtol, "passed": True,
              "fp64_algebra_passed": True, "max_fp64_algebra_error": 0.0,
              "max_state_absolute_error": 0.0, "max_residual_absolute_error": 0.0,
              "state_rows_outside_tolerance": 0, "residual_rows_outside_tolerance": 0}
    for row in rows:
        actual, expected = folded.step(row), unfolded.step(row)
        x64 = np.asarray(row, dtype=np.float64)
        normalized64, raw64 = ((x64 - mean) / scale) @ projection + bias, x64 @ weight64 + bias64
        algebra_ok = np.allclose(normalized64, raw64, atol=1e-12, rtol=1e-12)
        state_ok = np.allclose(folded.state, unfolded.state, atol=atol, rtol=rtol)
        residual_ok = np.allclose(actual, expected, atol=atol, rtol=rtol)
        result["rows"] += 1
        result["fp64_algebra_passed"] &= bool(algebra_ok)
        result["passed"] &= bool(algebra_ok and state_ok and residual_ok)
        result["max_fp64_algebra_error"] = max(result["max_fp64_algebra_error"], float(np.max(np.abs(normalized64-raw64))))
        result["max_state_absolute_error"] = max(result["max_state_absolute_error"], float(np.max(np.abs(folded.state-unfolded.state))))
        result["max_residual_absolute_error"] = max(result["max_residual_absolute_error"], float(np.max(np.abs(actual-expected))))
        result["state_rows_outside_tolerance"] += int(not state_ok)
        result["residual_rows_outside_tolerance"] += int(not residual_ok)
    if result["rows"] == 0:
        raise ValueError("folding diagnostic requires at least one feature row")
    return result


def export_dynamics(source, destination, parameters, *, fold_normalization=False):
    """Append a fixed-step dynamics branch; never overwrite source/destination.

    Every original node and initializer stays byte-identical. The original
    prediction value remains internal; the public output is branch_prediction.
    Folding is opt-in: FP32 rearrangement can fail strict auxiliary-state parity.
    Train using the same formulation that will be exported, or validate the
    rewrite prospectively. The conservative default preserves normalization.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from .model_io import load_weights

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or destination.exists():
        raise ValueError("branch export requires a new destination")
    load_weights(source)  # strict supplied-architecture/gate/schema validation
    model = onnx.load_model(str(source), load_external_data=False)
    original_nodes = [node.SerializeToString() for node in model.graph.node]
    original_initializers = [tensor.SerializeToString() for tensor in model.graph.initializer]
    if ([item.name for item in model.graph.input] != INPUT_NAMES[:3]
            or [item.name for item in model.graph.output] != ["prediction", *OUTPUT_NAMES[1:3]]):
        raise ValueError("branch export requires the original three-output contract")
    existing = {name for node in model.graph.node for name in [*node.input, *node.output]}
    existing.update(item.name for item in [*model.graph.initializer, *model.graph.input, *model.graph.output])
    if any(name.startswith("__dynamics_") for name in existing) or any(
            name in existing for name in ("branch_state", "branch_prediction", "next_branch_state")):
        raise ValueError("branch namespace already exists")
    initializers, nodes = [], []
    def constant(name, value, dtype=np.float32):
        full = "__dynamics_" + name
        initializers.append(numpy_helper.from_array(np.asarray(value, dtype=dtype), name=full))
        return full
    def node(op, inputs, name, **attributes):
        full = "__dynamics_" + name
        nodes.append(helper.make_node(op, inputs, [full], name=full, **attributes))
        return full

    p = parameters
    features = "features"
    if fold_normalization:
        projection_weight, projection_bias = folded_projection(p)
    else:
        centered = node("Sub", [features, constant("normalization_mean", p.normalization.mean)], "centered")
        features = node("Div", [centered, constant("normalization_scale", p.normalization.scale)], "normalized")
        projection_weight, projection_bias = p.projection_weight, p.projection_bias
    product = node("MatMul", [features, constant("projection_weight", projection_weight)], "projection_product")
    u = node("Add", [product, constant("projection_bias", projection_bias)], "u")
    axis = constant("state_axis", [2], np.int64)
    boundaries = {n: constant(f"offset_{n}", [n], np.int64) for n in (0, 16, 32, 33)}
    previous = node("Slice", ["branch_state", boundaries[0], boundaries[16], axis], "previous_input")
    mean = node("Slice", ["branch_state", boundaries[16], boundaries[32], axis], "mean_input")
    flag = node("Slice", ["branch_state", boundaries[32], boundaries[33], axis], "flag")
    initialized = node("Greater", [flag, constant("flag_threshold", 0.5)], "initialized")
    previous = node("Where", [initialized, previous, u], "previous")
    mean = node("Where", [initialized, mean, u], "mean")
    delta = node("Sub", [u, previous], "delta")
    innovation = node("Sub", [u, mean], "innovation")
    q = node("Concat", [u, delta, innovation], "features", axis=2)
    update = node("Mul", [innovation, constant("alpha", ALPHA)], "ema_update")
    next_mean = node("Add", [mean, update], "next_mean")
    nodes.append(helper.make_node("Concat", [u, next_mean, constant("initialized_flag", [[[1.0]]])],
                                  ["next_branch_state"], name="__dynamics_next_state", axis=2))
    product = node("MatMul", [q, constant("hidden_weight", p.hidden_weight)], "hidden_product")
    hidden = node("Add", [product, constant("hidden_bias", p.hidden_bias)], "hidden_affine")
    hidden = node("Relu", [hidden], "hidden")
    product = node("MatMul", [hidden, constant("output_weight", p.output_weight)], "output_product")
    residual = node("Add", [product, constant("output_bias", p.output_bias)], "residual")
    nodes.append(helper.make_node("Add", ["prediction", residual], ["branch_prediction"],
                                  name="__dynamics_prediction"))
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(initializers)
    model.graph.input.append(helper.make_tensor_value_info("branch_state", TensorProto.FLOAT, [1, 1, AUX_WIDTH]))
    model.graph.output[0].name = "branch_prediction"
    model.graph.output.append(helper.make_tensor_value_info("next_branch_state", TensorProto.FLOAT, [1, 1, AUX_WIDTH]))
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.update({"causal_branch_contract": CONTRACT,
                     "causal_branch_source_sha256": _sha(source),
                     "causal_branch_normalization": json.dumps(p.normalization.record(), sort_keys=True, allow_nan=False),
                     "causal_branch_normalization_folded": json.dumps(bool(fold_normalization))})
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model)
    if ([node.SerializeToString() for node in model.graph.node[:len(original_nodes)]] != original_nodes
            or [tensor.SerializeToString() for tensor in model.graph.initializer[:len(original_initializers)]] != original_initializers):
        raise ValueError("export changed the original graph")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(model.SerializeToString())
    return {"model_path": str(destination), "model_sha256": _sha(destination),
            "source_sha256": _sha(source), "contract": CONTRACT,
            "original_nodes_and_initializers_unchanged": True,
            "auxiliary_fp32_values": AUX_WIDTH, "additional_parameters": 2626,
            "analytical_macs_per_row": 2592, "normalization_folded": bool(fold_normalization),
            "normalization": p.normalization.record(), "export_only": True}


class DynamicsBackend:
    """One CPU ORT call with persistent two-bank buffers and owned predictions.

    StreamingPredictor validates sequence order and input finiteness; step
    accepts its contiguous FP32 [112] array. A runtime error is propagated.
    """

    def __init__(self, model_path, mode="iobinding"):
        import onnxruntime as ort
        from .runtime import session_options
        if mode not in {"iobinding", "run"}:
            raise ValueError("dynamics mode must be iobinding or run")
        self.mode = mode
        self.session = ort.InferenceSession(str(model_path), sess_options=session_options(),
                                            providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if [x.name for x in inputs] != INPUT_NAMES or [x.name for x in outputs] != OUTPUT_NAMES:
            raise ValueError("expected the dynamics four-input/four-output state contract")
        if (any(item.type != "tensor(float)" for item in [*inputs, *outputs])
                or [tuple(item.shape) for item in inputs] != INPUT_SHAPES
                or [tuple(item.shape) for item in outputs] != OUTPUT_SHAPES):
            raise ValueError("dynamics inputs and outputs must have fixed FP32 shapes")
        if self.session.get_modelmeta().custom_metadata_map.get("causal_branch_contract") != CONTRACT:
            raise ValueError("missing dynamics model contract metadata")
        self.features = np.empty(INPUT_SHAPES[0], np.float32)
        self.feature_row = self.features.reshape(112)
        self.output = np.empty(OUTPUT_SHAPES[0], np.float32)
        self.output_row = self.output.reshape(2)
        self.banks = [[np.zeros(shape, np.float32) for shape in INPUT_SHAPES[1:]] for _ in range(2)]
        self.bank = 0
        self.bindings = []
        self.feed = dict(zip(INPUT_NAMES, [self.features, *self.banks[0]]))
        if mode == "iobinding":
            for index in range(2):
                binding = self.session.io_binding()
                binding.bind_cpu_input("features", self.features)
                for name, array in zip(INPUT_NAMES[1:], self.banks[index]):
                    binding.bind_cpu_input(name, array)
                for name, array in zip(OUTPUT_NAMES, [self.output, *self.banks[1-index]]):
                    binding.bind_output(name, "cpu", 0, np.float32, array.shape, array.ctypes.data)
                self.bindings.append(binding)

    def reset(self):
        for bank in self.banks:
            for state in bank:
                state.fill(0)
        self.bank = 0
        for name, state in zip(INPUT_NAMES[1:], self.banks[0]):
            self.feed[name] = state

    def step(self, features):
        if not isinstance(features, np.ndarray) or features.shape != (112,) or features.dtype != np.float32:
            raise ValueError("dynamics step expects an FP32 [112] array")
        np.copyto(self.feature_row, features)
        if self.mode == "iobinding":
            self.session.run_with_iobinding(self.bindings[self.bank])
        else:
            values = self.session.run(OUTPUT_NAMES, self.feed)
            np.copyto(self.output, values[0])
            for target, value in zip(self.banks[1-self.bank], values[1:]):
                np.copyto(target, value)
        self.bank = 1 - self.bank
        if self.mode == "run":
            for name, state in zip(INPUT_NAMES[1:], self.banks[self.bank]):
                self.feed[name] = state
        return self.output_row.copy()

    def hidden_states(self):
        return tuple(state.copy() for state in self.banks[self.bank][:2])

    def auxiliary_state(self):
        return self.banks[self.bank][2].reshape(AUX_WIDTH).copy()

    def close(self):
        pass
