"""Small inference-only module; no training, ONNX parser, or dataset imports."""
from __future__ import annotations

import json
import os
import platform
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort


def session_options():
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


def apply_affinity(enabled=False):
    """Restrict only this process to one allowed CPU, when explicitly selected."""
    result = {"requested": bool(enabled), "applied": False}
    if not enabled:
        return result
    try:
        allowed = sorted(os.sched_getaffinity(0))
        if not allowed:
            raise OSError("empty allowed CPU set")
        os.sched_setaffinity(0, {allowed[0]})
        result.update(applied=True, cpu=allowed[0], original_allowed=allowed)
    except (AttributeError, OSError) as exc:
        result["reason"] = str(exc)
    return result


class OnnxBackend:
    """One synchronous, causal step; all returned outputs have stable ownership."""

    def __init__(self, model_path, mode="run"):
        if mode not in {"run", "reference", "iobinding", "ortvaluevector"}:
            raise ValueError(f"unknown ONNX mode: {mode}")
        self.mode = mode
        self.session = ort.InferenceSession(str(model_path), sess_options=session_options(),
                                            providers=["CPUExecutionProvider"])
        self.output_names = [x.name for x in self.session.get_outputs()]
        self.input_names = [x.name for x in self.session.get_inputs()]
        if self.input_names != ["features", "hidden_0", "hidden_1"] or len(self.output_names) != 3:
            raise ValueError("expected features + two hidden states, and three outputs")
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        shape0, shape1 = inputs[1].shape, inputs[2].shape
        if any(x.type != "tensor(float)" for x in [*inputs, *outputs]):
            raise ValueError("deployment inputs and outputs must be FP32")
        if inputs[0].shape != [1, 1, 112] or outputs[0].shape != [1, 1, 2]:
            raise ValueError("deployment expects features [1,1,112] and predictions [1,1,2]")
        if (any(len(s) != 3 or s[:2] != [1, 1] for s in [shape0, shape1]) or
                any(not isinstance(d, int) or d <= 0 for d in shape0 + shape1)):
            raise ValueError("deployment hidden state shapes must be fixed")
        if outputs[1].shape != shape0 or outputs[2].shape != shape1:
            raise ValueError("deployment hidden input/output shapes must match")
        self.states = [np.zeros(shape0, np.float32), np.zeros(shape1, np.float32)]
        self.feed = dict(zip(self.input_names, [None, *self.states]))
        self.features = np.empty((1, 1, 112), np.float32)
        self.output = np.empty((1, 1, 2), np.float32)
        if mode in {"iobinding", "ortvaluevector"}:
            self.banks = [[np.zeros(shape0, np.float32), np.zeros(shape1, np.float32)]
                          for _ in range(2)]
            self.bank = 0
            if mode == "iobinding":
                self.bindings = []
                for source in range(2):
                    binding = self.session.io_binding()
                    binding.bind_cpu_input("features", self.features)
                    for name, array in zip(self.input_names[1:], self.banks[source]):
                        binding.bind_cpu_input(name, array)
                    for name, array in zip(self.output_names, [self.output, *self.banks[1-source]]):
                        binding.bind_output(name, "cpu", 0, np.float32, array.shape, array.ctypes.data)
                    self.bindings.append(binding)
            else:
                # Use ORT's pinned low-level vector types only for this optional mode.
                from onnxruntime.capi import _pybind_state as C
                self.run_options = ort.RunOptions()
                self._ort_values = []  # retain NumPy owners and OrtValue wrappers
                self.vectors = []
                for source in range(2):
                    feeds, fetches = C.OrtValueVector(), C.OrtValueVector()
                    for array in [self.features, *self.banks[source]]:
                        value = ort.OrtValue.ortvalue_from_numpy(array)
                        self._ort_values.append(value)
                        feeds.push_back(value._ortvalue)
                    for array in [self.output, *self.banks[1-source]]:
                        value = ort.OrtValue.ortvalue_from_numpy(array)
                        self._ort_values.append(value)
                        fetches.push_back(value._ortvalue)
                    devices = [C.OrtDevice(C.OrtDevice.cpu(), C.OrtDevice.default_memory(), 0)] * 3
                    self.vectors.append((feeds, fetches, devices))

    def reset(self):
        if hasattr(self, "banks"):
            for bank in self.banks:
                for state in bank:
                    state.fill(0)
            self.bank = 0
            self.states = self.banks[0]
        else:
            for state in self.states:
                state.fill(0)
        for name, state in zip(self.input_names[1:], self.states):
            self.feed[name] = state

    def step(self, features):
        if self.mode in {"run", "reference"}:
            self.feed["features"] = features.reshape(1, 1, 112)
            names = None if self.mode == "reference" else self.output_names
            prediction, h0, h1 = self.session.run(names, self.feed)
            self.states = [h0, h1]
            self.feed["hidden_0"], self.feed["hidden_1"] = h0, h1
            return prediction.reshape(2).copy()
        np.copyto(self.features.reshape(112), features)
        if self.mode == "iobinding":
            self.session.run_with_iobinding(self.bindings[self.bank])
        else:
            feeds, fetches, devices = self.vectors[self.bank]
            self.session.run_with_ortvaluevector(self.run_options, self.input_names, feeds,
                                                self.output_names, fetches, devices)
        self.bank = 1 - self.bank
        self.states = self.banks[self.bank]
        return self.output.reshape(2).copy()

    def hidden_states(self):
        return tuple(x.copy() for x in self.states)

    def close(self):
        pass


class StreamingPredictor:
    def __init__(self, backend, postprocess=None, strict=True, capture_raw=False):
        self.backend = backend
        self.strict = strict
        self.capture_raw = capture_raw
        self.sequence = None
        self.previous_step = None
        self.last_raw_prediction = None
        self.ema = None
        config = postprocess or {}
        self.scales = np.asarray(config.get("scales", [1, 1]), dtype=np.float32)
        if self.scales.shape != (2,) or not np.isfinite(self.scales).all():
            raise ValueError("invalid output scales")
        affine = config.get("affine") or {}
        self.matrix = np.asarray(affine.get("matrix", np.eye(2)), np.float32)
        self.bias = np.asarray(affine.get("bias", [0, 0]), np.float32)
        if self.matrix.shape != (2, 2) or self.bias.shape != (2,) or not np.isfinite(self.matrix).all() or not np.isfinite(self.bias).all():
            raise ValueError("invalid output affine transform")
        self.scale = not np.array_equal(self.scales, [1, 1])
        self.affine = bool(affine)
        self.transform = self.scale or self.affine
        self.blend = np.broadcast_to(np.asarray(config.get("ema_blend", 0), np.float32), (2,)).copy()
        half = config.get("ema_half_life")
        if not np.isfinite(self.blend).all() or np.any((self.blend < 0) | (self.blend > 1)):
            raise ValueError("EMA blend must be between zero and one")
        self.alpha = None
        if self.blend.any():
            if half is None:
                raise ValueError("EMA half life required for nonzero blend")
            half = np.broadcast_to(np.asarray(half, np.float64), (2,))
            if not np.isfinite(half).all() or np.any(half <= 0):
                raise ValueError("EMA half life must be positive")
            self.alpha = (-np.expm1(-np.log(2.0) / half)).astype(np.float32)

    def predict(self, data_point):
        seq, step = int(data_point.seq_ix), int(data_point.step_in_seq)
        need = bool(data_point.need_prediction)
        if step < 0 or step >= 20_000 or need != (step >= 99):
            raise ValueError("invalid sequence step or warm-up flag")
        if seq != self.sequence:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self.backend.reset()
            self.sequence, self.previous_step, self.ema = seq, None, None
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        features = np.asarray(data_point.state, dtype=np.float32)
        if features.shape != (112,):
            raise ValueError("expected 112 input features")
        if self.strict and not np.isfinite(features).all():
            raise ValueError("nonfinite input features")
        features = np.ascontiguousarray(features)
        raw = self.backend.step(features)
        if raw.shape != (2,) or not np.isfinite(raw).all():
            raise ValueError("model returned invalid predictions")
        self.previous_step = step
        if self.capture_raw:
            self.last_raw_prediction = raw.copy()
        prediction = raw * self.scales if self.scale else raw
        if self.affine:
            prediction = prediction @ self.matrix.T + self.bias
        if self.alpha is not None:
            if self.ema is None:
                self.ema = prediction.copy()
            else:
                self.ema += self.alpha * (prediction - self.ema)
            prediction = (1 - self.blend) * prediction + self.blend * self.ema
        if (self.transform or self.alpha is not None) and not np.isfinite(prediction).all():
            raise ValueError("postprocessing returned nonfinite predictions")
        return prediction if need else None


def load_predictor(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    root = config_path.parent
    apply_affinity(config.get("affinity", False))
    selected = config.get("backend", "onnx")
    if selected == "ensemble":
        from .ensemble_runtime import load_ensemble_predictor
        return load_ensemble_predictor(root / config["cpp_source"], root / config["weights"])
    if selected not in {"onnx", "native"}:
        raise ValueError("backend must be onnx, native, or ensemble")
    backend = None
    if selected == "native":
        try:
            from .native import NativeBackend
            key = f"{platform.system().lower()}-{platform.machine().lower()}"
            library = config.get("native_libraries", {}).get(key)
            if not library:
                raise OSError(f"no native library for {key}")
            backend = NativeBackend(root / config["native_weights"], root / library,
                                    layout=config.get("native_layout", "packed"))
        except (OSError, RuntimeError, ValueError, KeyError, ImportError, AttributeError) as exc:
            # This is startup-only. Never replace an already-advanced backend.
            warnings.warn(f"Native backend unavailable; using validated ONNX: {exc}", RuntimeWarning)
    if backend is None:
        backend = OnnxBackend(root / config["model"], config.get("onnx_mode", "run"))
    return StreamingPredictor(backend, config.get("postprocess"), strict=config.get("strict_inputs", True))
