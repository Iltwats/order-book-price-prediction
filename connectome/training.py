"""Bounded, reproducible training and conversion for the Connectome stream.

The original ONNX model is the initialization and fallback.  Training uses only
provided labels, processes complete sequences, and never uses ``is_scored`` as
an input.  Public training functions return paths plus a JSON-serializable
report; none runs at import time.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
from typing import Iterable, Iterator

import numpy as np
import pyarrow
import pyarrow.parquet as pq
import torch
from torch import nn

from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, TARGET_COLUMNS, WARMUP, validate_sequence

DEFAULT_SEED = 20260912
CHECKPOINT_FORMAT = "connectome.training.v1"


class StreamingGRU(nn.Module):
    """Two separate forward GRUs with the baseline's external state contract.

    Features are [time, batch, 112]; each state is [1, batch, hidden_size].
    ``head_state='both'`` reads concatenated layer outputs, not future states.
    """

    def __init__(self, hidden_size: int = 128, head_state: str = "last"):
        super().__init__()
        if hidden_size <= 0 or head_state not in {"last", "both"}:
            raise ValueError("positive hidden_size and head_state='last' or 'both' required")
        self.hidden_size = int(hidden_size)
        self.head_state = head_state
        self.gru0 = nn.GRU(112, self.hidden_size)
        self.gru1 = nn.GRU(self.hidden_size, self.hidden_size)
        self.head = nn.Linear(self.hidden_size * (2 if head_state == "both" else 1), 2)

    @property
    def config(self) -> dict:
        return {"hidden_size": self.hidden_size, "head_state": self.head_state}

    def initial_states(self, batch_size: int, *, device=None, dtype=None):
        parameter = next(self.parameters())
        kwargs = {"device": device or parameter.device, "dtype": dtype or parameter.dtype}
        return (torch.zeros(1, batch_size, self.hidden_size, **kwargs),
                torch.zeros(1, batch_size, self.hidden_size, **kwargs))

    def encode(self, features, hidden_0=None, hidden_1=None):
        lower, next_0 = self.gru0(features, hidden_0)
        upper, next_1 = self.gru1(lower, hidden_1)
        return lower, upper, next_0, next_1

    def forward(self, features, hidden_0=None, hidden_1=None):
        lower, upper, next_0, next_1 = self.encode(features, hidden_0, hidden_1)
        encoded = torch.cat((lower, upper), dim=-1) if self.head_state == "both" else upper
        return self.head(encoded), next_0, next_1


def seed_everything(seed: int = DEFAULT_SEED, threads: int = 1) -> None:
    if threads < 1:
        raise ValueError("threads must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def _reorder_gates(array: np.ndarray, hidden_size: int) -> np.ndarray:
    """ONNX z,r,h -> PyTorch r,z,n, preserving remaining dimensions."""
    array = np.asarray(array)
    if array.shape[0] != 3 * hidden_size:
        raise ValueError("invalid GRU gate tensor")
    return np.concatenate((array[hidden_size:2 * hidden_size],
                           array[:hidden_size], array[2 * hidden_size:]), axis=0)


def model_from_weights(weights: dict, *, device: str = "cpu") -> StreamingGRU:
    """Reconstruct weights already validated by ``model_io.load_weights``.

    The loader must reject reverse/bidirectional GRUs and LBR != 1.  This
    function also validates all tensor shapes and finite values.
    """
    expected = {"w0", "r0", "b0", "w1", "r1", "b1", "head_w", "head_b"}
    if not expected.issubset(weights):
        raise ValueError(f"missing model weights: {sorted(expected - set(weights))}")
    hidden_size = int(np.asarray(weights["r0"]).shape[-1])
    head_w = np.asarray(weights["head_w"])
    if head_w.shape not in {(hidden_size, 2), (2 * hidden_size, 2)}:
        raise ValueError("head_w must read the upper state or both layer states")
    model = StreamingGRU(hidden_size, "both" if head_w.shape[0] == 2 * hidden_size else "last")
    with torch.no_grad():
        for index, layer in enumerate((model.gru0, model.gru1)):
            input_size = 112 if index == 0 else hidden_size
            w = np.asarray(weights[f"w{index}"], dtype=np.float32)
            r = np.asarray(weights[f"r{index}"], dtype=np.float32)
            b = np.asarray(weights[f"b{index}"], dtype=np.float32)
            # model_io removes the singleton direction axis.  Also accept
            # original initializer tensors for direct conversion tests.
            if w.ndim == 2:
                w = w[None]
            if r.ndim == 2:
                r = r[None]
            if b.ndim == 1:
                b = b[None]
            if w.shape != (1, 3 * hidden_size, input_size) or r.shape != (1, 3 * hidden_size, hidden_size) or b.shape != (1, 6 * hidden_size):
                raise ValueError(f"invalid GRU {index} weight shapes")
            if not all(np.isfinite(v).all() for v in (w, r, b)):
                raise ValueError("nonfinite model weights")
            layer.weight_ih_l0.copy_(torch.from_numpy(_reorder_gates(w[0], hidden_size)))
            layer.weight_hh_l0.copy_(torch.from_numpy(_reorder_gates(r[0], hidden_size)))
            layer.bias_ih_l0.copy_(torch.from_numpy(_reorder_gates(b[0, :3 * hidden_size], hidden_size)))
            layer.bias_hh_l0.copy_(torch.from_numpy(_reorder_gates(b[0, 3 * hidden_size:], hidden_size)))
        head_b = np.asarray(weights["head_b"], dtype=np.float32)
        if head_b.shape != (2,) or not np.isfinite(head_w).all() or not np.isfinite(head_b).all():
            raise ValueError("invalid head weights")
        model.head.weight.copy_(torch.from_numpy(head_w.astype(np.float32).T.copy()))
        model.head.bias.copy_(torch.from_numpy(head_b.copy()))
    return model.to(device).eval()


def load_torch_model(model_path: str | Path, *, device: str = "cpu") -> StreamingGRU:
    from connectome.model_io import load_weights
    return model_from_weights(load_weights(model_path), device=device)


def with_head_state(model: StreamingGRU, head_state: str) -> StreamingGRU:
    if head_state == model.head_state:
        return copy.deepcopy(model)
    if model.head_state != "last" or head_state != "both":
        raise ValueError("only a last-state head can be expanded to both states")
    result = StreamingGRU(model.hidden_size, "both").to(next(model.parameters()).device)
    result.gru0.load_state_dict(model.gru0.state_dict())
    result.gru1.load_state_dict(model.gru1.state_dict())
    with torch.no_grad():
        result.head.weight.zero_()
        result.head.weight[:, model.hidden_size:].copy_(model.head.weight)
        result.head.bias.copy_(model.head.bias)
    return result.eval()


def export_onnx(model: StreamingGRU, path: str | Path, *, dynamic: bool = False) -> str:
    """Export FP32 opset 17 with explicit state; fixed one-step by default.

    ``dynamic=True`` is an offline training/evaluation convenience with
    time-major [time,batch,112] inputs. Fixed GRU128/last-head exports remain
    compatible with the strict native weight extractor.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    exported = copy.deepcopy(model).cpu().float().eval()
    features = torch.zeros(1, 1, 112, dtype=torch.float32)
    states = exported.initial_states(1)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.onnx.export(
            exported, (features, *states), str(temporary),
            input_names=["features", "hidden_0", "hidden_1"],
            output_names=["prediction", "next_hidden_0", "next_hidden_1"],
            dynamic_axes=({"features": {0: "time", 1: "batch"},
                          "hidden_0": {1: "batch"}, "hidden_1": {1: "batch"},
                          "prediction": {0: "time", 1: "batch"},
                          "next_hidden_0": {1: "batch"}, "next_hidden_1": {1: "batch"}} if dynamic else None),
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        import onnx
        graph = onnx.load(str(temporary))
        graph.ir_version = 8
        onnx.checker.check_model(graph)
        onnx.save(graph, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def export_frozen_head(source_path, model: StreamingGRU, destination, *, dynamic: bool = False) -> str:
    """Replace only the original ONNX readout tensors, preserving its graph.

    Frozen fits need no reconstructed recurrent implementation at deployment.
    For a both-state readout, concatenate the lower GRU's complete output
    sequence with the original upper-state head input. ``dynamic=True`` is a
    batch-first offline export, including for the ONNX-only both-state head.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    graph = onnx.load(str(source_path))
    matmul = next(node for node in graph.graph.node if node.op_type == "MatMul")
    add = next(node for node in graph.graph.node if node.op_type == "Add")
    bias_name = next(name for name in add.input if name != matmul.output[0])
    replacements = {matmul.input[1]: model.head.weight.detach().cpu().numpy().T.copy(),
                    bias_name: model.head.bias.detach().cpu().numpy().copy()}
    replaced = set()
    for initializer in graph.graph.initializer:
        if initializer.name in replacements:
            name = initializer.name
            initializer.CopyFrom(numpy_helper.from_array(replacements[name], name=name))
            replaced.add(name)
    if replaced != set(replacements):
        raise ValueError("original readout tensors must be direct initializers")
    if model.head_state == "both":
        lower = next(node for node in graph.graph.node if node.op_type == "GRU" and node.input[5] == "hidden_0")
        if not lower.output[0]:
            raise ValueError("both-state readout requires the complete lower GRU output")
        producers = {name: node for node in graph.graph.node for name in node.output if name}
        upper = producers.get(matmul.input[0])
        if upper is None or upper.op_type != "Transpose" or list(helper.get_attribute_value(next(a for a in upper.attribute if a.name == "perm"))) != [1, 0, 2]:
            raise ValueError("both-state frozen export requires the original batch-first head layout")
        axes_name = "frozen_head_squeeze_axes"
        graph.graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name))
        nodes = [helper.make_node("Squeeze", [lower.output[0], axes_name], ["frozen_head_lower_time"]),
                 helper.make_node("Transpose", ["frozen_head_lower_time"], ["frozen_head_lower_batch"], perm=[1, 0, 2]),
                 helper.make_node("Concat", ["frozen_head_lower_batch", matmul.input[0]], ["frozen_head_both"], axis=2)]
        insertion = next(index for index, node in enumerate(graph.graph.node) if node is matmul or node == matmul)
        for offset, node in enumerate(nodes):
            graph.graph.node.insert(insertion + offset, node)
        matmul.input[0] = "frozen_head_both"
    if dynamic:
        shapes = {"features": ("batch", "time", 112), "prediction": ("batch", "time", 2),
                  "hidden_0": (1, "batch", model.hidden_size), "hidden_1": (1, "batch", model.hidden_size),
                  "next_hidden_0": (1, "batch", model.hidden_size), "next_hidden_1": (1, "batch", model.hidden_size)}
        for value in [*graph.graph.input, *graph.graph.output]:
            value.CopyFrom(helper.make_tensor_value_info(value.name, TensorProto.FLOAT, shapes[value.name]))
        del graph.graph.value_info[:]
        graph = onnx.shape_inference.infer_shapes(graph, strict_mode=True)
    onnx.checker.check_model(graph)
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    onnx.save(graph, str(temporary))
    os.replace(temporary, destination)
    return str(destination)


def _fingerprint(path: str | Path, *, content: bool = False) -> dict:
    path = Path(path).resolve()
    stat = path.stat()
    result = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if content:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _groups(path, max_sequences: int | None, seed: int, group_indices: Iterable[int] | None) -> list[int]:
    count = pq.ParquetFile(path).num_row_groups
    if max_sequences is not None and max_sequences <= 0:
        raise ValueError("max_sequences must be positive or None")
    if group_indices is None:
        groups = np.random.default_rng(seed).permutation(count).tolist()
    else:
        groups = [int(group) for group in group_indices]
        if len(set(groups)) != len(groups) or any(group < 0 or group >= count for group in groups):
            raise ValueError("group_indices must be distinct existing row-group indices")
    if max_sequences is not None:
        groups = groups[:max_sequences]
    if not groups:
        raise ValueError("no training sequences selected")
    return groups


def _iter_sequences(path, groups: Iterable[int]) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    parquet = pq.ParquetFile(path)
    required = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    if not set(required).issubset(parquet.schema_arrow.names):
        raise ValueError("training requires the labeled 112-feature sequence schema")
    for group in groups:
        table = parquet.read_row_group(group, columns=required, use_threads=False)
        seq, _ = validate_sequence(table)
        x = np.column_stack([table[column].to_numpy() for column in FEATURE_COLUMNS]).astype(np.float32)
        y = np.column_stack([table[column].to_numpy() for column in TARGET_COLUMNS]).astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"nonfinite training data in sequence {seq}")
        yield seq, x, y


def _atomic_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save({**state, "environment": _environment()}, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checkpoint(path: str | Path) -> dict:
    # These are local training checkpoints generated by this module, not an
    # interchange format for untrusted third-party artifacts.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or state.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("not a Connectome training checkpoint")
    return state


def _environment() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform(),
            "torch": str(torch.__version__), "numpy": np.__version__,
            "pyarrow": pyarrow.__version__, "torch_cpu_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def _check_resume(state: dict, specification: dict) -> None:
    old = dict(state.get("specification", {}))
    if old.get("method") in {"head", "affine", "head_adamw"}:
        old.setdefault("encoder", "torch")
    if old != specification:
        changed = sorted(key for key in set(old) | set(specification) if old.get(key) != specification.get(key))
        raise ValueError(f"resume configuration changed: {', '.join(changed)}")


def _finish(model, output_dir, report: dict, checkpoint: dict) -> dict:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    spec = checkpoint["specification"]
    if spec["method"] in {"head", "head_adamw", "affine"}:
        model_path = export_frozen_head(spec["source"]["path"], model, output_dir / "model.onnx")
    else:
        model_path = export_onnx(model, output_dir / "model.onnx")
    checkpoint_path = output_dir / "checkpoint.pt"
    _atomic_checkpoint(checkpoint_path, checkpoint)
    report = {**report, "model_path": model_path, "checkpoint_path": str(checkpoint_path),
              "report_path": str(output_dir / "report.json"),
              "model_config": model.config, "parameter_count": sum(p.numel() for p in model.parameters()),
              "environment": _environment(),
              "training_metric": "sequence-normalized clipped-target weighted MSE surrogate; not validation WP"}
    destination = output_dir / "report.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)
    return report


class OnnxSequenceEncoder:
    """Offline original-graph ORT encoder exposing both causal state sequences.

    It clones the original batch-first graph through model_io, then exposes
    each GRU's existing sequence output. No GRU or readout arithmetic changes.
    Bulk-vs-callback parity must still be measured on real complete sequences.
    """

    def __init__(self, model_path, destination):
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper
        from connectome.model_io import export_sequence_model
        destination = Path(destination).resolve()
        export_sequence_model(model_path, destination)
        graph = onnx.load(str(destination))
        nodes = {node.input[5]: node for node in graph.graph.node if node.op_type == "GRU"}
        axes_name = "training_encoder_squeeze_axes"
        graph.graph.initializer.append(numpy_helper.from_array(np.array([1], dtype=np.int64), name=axes_name))
        for index in range(2):
            node = nodes[f"hidden_{index}"]
            if not node.output[0]:
                raise ValueError("offline encoder needs original GRU sequence outputs")
            squeezed, exposed = f"training_encoder_squeezed_{index}", f"training_encoder_h{index}"
            graph.graph.node.extend([
                helper.make_node("Squeeze", [node.output[0], axes_name], [squeezed]),
                helper.make_node("Transpose", [squeezed], [exposed], perm=[1, 0, 2]),
            ])
            graph.graph.output.append(helper.make_tensor_value_info(exposed, TensorProto.FLOAT, ["batch", "time", 128]))
        onnx.checker.check_model(graph)
        onnx.save(graph, str(destination))
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(destination), sess_options=options, providers=["CPUExecutionProvider"])
        self.provenance = {"kind": "original_onnx_bulk", "path": str(destination),
                           "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                           "source_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
                           "onnxruntime": ort.__version__, "compute_threads": 1}

    def __call__(self, x):
        predictions, lower, upper = self.session.run(
            ["prediction", "training_encoder_h0", "training_encoder_h1"],
            {"features": np.asarray(x, dtype=np.float32)[None],
             "hidden_0": np.zeros((1, 1, 128), dtype=np.float32),
             "hidden_1": np.zeros((1, 1, 128), dtype=np.float32)})
        return lower[0], upper[0], predictions[0]


def _encoder_backend(model_path, output_dir, encoder):
    if encoder == "onnx":
        backend = OnnxSequenceEncoder(model_path, Path(output_dir) / "encoder.onnx")
        return backend, backend.provenance
    if encoder == "torch":
        return None, {"kind": "reconstructed_torch", "torch": str(torch.__version__),
                      "source_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest()}
    raise ValueError("encoder must be 'onnx' or 'torch'")


def _encoded_sequence(model, x: np.ndarray, *, chunk_size: int, encoder=None):
    if encoder is not None:
        return encoder(x)
    device = next(model.parameters()).device
    states = model.initial_states(1)
    lower_parts, upper_parts, prediction_parts = [], [], []
    with torch.no_grad():
        for start in range(0, len(x), chunk_size):
            features = torch.from_numpy(x[start:start + chunk_size, None]).to(device)
            lower, upper, *states = model.encode(features, *states)
            encoded = torch.cat((lower, upper), -1) if model.head_state == "both" else upper
            prediction_parts.append(model.head(encoded)[:, 0].cpu().numpy())
            lower_parts.append(lower[:, 0].cpu().numpy())
            upper_parts.append(upper[:, 0].cpu().numpy())
    return np.concatenate(lower_parts), np.concatenate(upper_parts), np.concatenate(prediction_parts)


def _fit_linear(model_path, train_path, output_dir, *, kind: str, head_state: str = "last",
                max_sequences: int | None = 32, group_indices: Iterable[int] | None = None,
                ridge: float = 1e-3, rows_per_sequence: int | None = None,
                chunk_size: int = 512, seed: int = DEFAULT_SEED, device: str = "cpu",
                threads: int = 1, resume_from: str | Path | None = None,
                max_new_sequences: int | None = None, encoder: str = "onnx", progress: bool = False) -> dict:
    if ridge < 0 or chunk_size <= 0 or rows_per_sequence is not None and rows_per_sequence <= 0:
        raise ValueError("ridge must be nonnegative and chunk/row limits positive")
    if max_new_sequences is not None and max_new_sequences <= 0:
        raise ValueError("max_new_sequences must be positive")
    if kind not in {"head", "affine"}:
        raise ValueError("unsupported linear fitting method")
    started = time.monotonic()
    seed_everything(seed, threads)
    source = load_torch_model(model_path, device=device)
    fitted = with_head_state(source, head_state) if kind == "head" else copy.deepcopy(source)
    encoding, encoder_provenance = _encoder_backend(model_path, output_dir, encoder)
    selected = _groups(train_path, max_sequences, seed, group_indices)
    specification = {"method": kind, "source": _fingerprint(model_path, content=True),
                     "data": _fingerprint(train_path), "groups": selected,
                     "head_state": head_state if kind == "head" else source.head_state,
                     "ridge": float(ridge), "rows_per_sequence": rows_per_sequence,
                     "chunk_size": chunk_size, "seed": seed, "encoder": encoder}
    width = (source.hidden_size * (2 if head_state == "both" else 1)) if kind == "head" else 2
    grams = np.zeros((2, width + 1, width + 1), dtype=np.float64)
    right = np.zeros((2, width + 1), dtype=np.float64)
    completed = 0
    sampled_rows = 0
    sequence_ids = []
    if resume_from is not None:
        checkpoint = _read_checkpoint(resume_from)
        _check_resume(checkpoint, specification)
        grams, right = checkpoint["grams"], checkpoint["right"]
        completed, sampled_rows = checkpoint["completed_sequences"], checkpoint["sampled_rows"]
        sequence_ids = checkpoint["sequence_ids"]
    output_dir = Path(output_dir).resolve()
    end = len(selected) if max_new_sequences is None else min(len(selected), completed + max_new_sequences)
    for position, (seq, x, y) in enumerate(_iter_sequences(train_path, selected[completed:end]), start=completed):
        lower, upper, base = _encoded_sequence(source, x, chunk_size=chunk_size, encoder=encoding)
        if kind == "affine":
            representation = base
        else:
            representation = np.concatenate((lower, upper), axis=1) if head_state == "both" else upper
        available = np.arange(WARMUP, len(x))
        retained = available
        if rows_per_sequence is not None and len(available) > rows_per_sequence:
            retained = np.random.default_rng(seed + int(selected[position])).choice(available, rows_per_sequence, replace=False)
        inverse_probability = len(available) / len(retained)
        design = np.column_stack((representation[retained], np.ones(len(retained)))).astype(np.float64)
        clipped = np.clip(y, -2.0, 2.0).astype(np.float64)
        total_weights = np.abs(clipped[WARMUP:]).sum(axis=0)
        for target in range(2):
            if total_weights[target] <= 0:
                continue
            weights = np.abs(clipped[retained, target]) * inverse_probability / total_weights[target]
            residual = clipped[retained, target] - base[retained, target].astype(np.float64)
            grams[target] += design.T @ (weights[:, None] * design)
            right[target] += design.T @ (weights * residual)
        completed = position + 1
        sampled_rows += len(retained)
        sequence_ids.append(int(seq))
        checkpoint = {"format": CHECKPOINT_FORMAT, "specification": specification,
                      "grams": grams, "right": right, "completed_sequences": completed,
                      "sampled_rows": sampled_rows, "sequence_ids": sequence_ids}
        _atomic_checkpoint(output_dir / "checkpoint.pt", checkpoint)
        if progress and (completed % 32 == 0 or completed == end):
            print(f"{kind} fit {completed}/{len(selected)} sequences; {time.monotonic()-started:.1f}s", flush=True)
    if completed == 0:
        raise ValueError("no sequences available to fit")
    corrections = []
    for target in range(2):
        matrix = grams[target] / completed + ridge * np.eye(width + 1)
        rhs = right[target] / completed
        corrections.append(np.linalg.lstsq(matrix, rhs, rcond=None)[0])
    correction = np.stack(corrections).astype(np.float32)
    with torch.no_grad():
        coefficients = torch.from_numpy(correction).to(device)
        if kind == "head":
            fitted.head.weight.add_(coefficients[:, :-1])
            fitted.head.bias.add_(coefficients[:, -1])
        else:
            # Fuse p_new = p + D p + d into the original head, avoiding a
            # second inference call or runtime postprocessing layer.
            fitted.head.weight.copy_(source.head.weight + coefficients[:, :2] @ source.head.weight)
            fitted.head.bias.copy_(source.head.bias + coefficients[:, :2] @ source.head.bias + coefficients[:, 2])
    checkpoint = {"format": CHECKPOINT_FORMAT, "specification": specification,
                  "grams": grams, "right": right, "completed_sequences": completed,
                  "sampled_rows": sampled_rows, "sequence_ids": sequence_ids,
                  "model_config": fitted.config, "model_state": fitted.cpu().state_dict()}
    return _finish(fitted, output_dir,
                   {"method": kind, "seed": seed, "completed_sequences": completed,
                    "planned_sequences": len(selected), "complete": completed == len(selected),
                    "sampled_rows": sampled_rows, "ridge": ridge,
                    "encoder": encoder_provenance,
                    "seconds": time.monotonic() - started, "specification": specification}, checkpoint)


def _fit_head_adamw(model_path, train_path, output_dir, *, head_state: str = "last",
                     max_sequences: int | None = 32, group_indices: Iterable[int] | None = None,
                     penalty: float = 1e-3, learning_rate: float = 1e-3,
                     minibatch_size: int = 4096, epochs: int = 1,
                     rows_per_sequence: int | None = None, chunk_size: int = 512,
                     seed: int = DEFAULT_SEED, device: str = "cpu", threads: int = 1,
                     resume_from: str | Path | None = None,
                     max_new_sequences: int | None = None, encoder: str = "onnx", progress: bool = False) -> dict:
    if penalty < 0 or learning_rate <= 0 or minibatch_size <= 0 or epochs <= 0 or chunk_size <= 0:
        raise ValueError("invalid AdamW head hyperparameters")
    if rows_per_sequence is not None and rows_per_sequence <= 0 or max_new_sequences is not None and max_new_sequences <= 0:
        raise ValueError("row and sequence limits must be positive")
    started = time.monotonic()
    seed_everything(seed, threads)
    source = load_torch_model(model_path, device=device)
    fitted = with_head_state(source, head_state)
    encoding, encoder_provenance = _encoder_backend(model_path, output_dir, encoder)
    selected = _groups(train_path, max_sequences, seed, group_indices)
    delta = nn.Linear(fitted.head.in_features, 2).to(device)
    nn.init.zeros_(delta.weight)
    nn.init.zeros_(delta.bias)
    optimizer = torch.optim.AdamW(delta.parameters(), lr=learning_rate, weight_decay=penalty)
    specification = {"method": "head_adamw", "source": _fingerprint(model_path, content=True),
                     "data": _fingerprint(train_path), "groups": selected, "head_state": head_state,
                     "penalty": penalty, "learning_rate": learning_rate, "minibatch_size": minibatch_size,
                     "rows_per_sequence": rows_per_sequence, "chunk_size": chunk_size, "seed": seed, "encoder": encoder}
    cursor = {"epoch": 0, "position": 0, "updates": 0, "sequences_seen": 0}
    history = []
    if resume_from is not None:
        previous = _read_checkpoint(resume_from)
        _check_resume(previous, specification)
        delta.load_state_dict(previous["delta_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        cursor, history = previous["cursor"], previous["history"]
        _restore_rng(previous["rng"])
    output_dir = Path(output_dir).resolve()

    def checkpoint():
        return {"format": CHECKPOINT_FORMAT, "specification": specification,
                "delta_state": delta.state_dict(), "optimizer_state": optimizer.state_dict(),
                "cursor": dict(cursor), "history": list(history), "rng": _rng_state()}

    _atomic_checkpoint(output_dir / "checkpoint.pt", checkpoint())
    processed = 0
    while cursor["epoch"] < epochs and (max_new_sequences is None or processed < max_new_sequences):
        epoch = cursor["epoch"]
        order = np.random.default_rng(seed + epoch).permutation(selected).tolist()
        group = order[cursor["position"]]
        seq, x, y = next(_iter_sequences(train_path, [group]))
        lower, upper, base = _encoded_sequence(source, x, chunk_size=chunk_size, encoder=encoding)
        representation = np.concatenate((lower, upper), axis=1) if head_state == "both" else upper
        rows = np.arange(WARMUP, len(x))
        if rows_per_sequence is not None and len(rows) > rows_per_sequence:
            rows = np.random.default_rng(seed + group + epoch * len(selected)).choice(rows, rows_per_sequence, replace=False)
        inverse_probability = (len(x) - WARMUP) / len(rows)
        clipped = np.clip(y, -2, 2)
        denominators = torch.from_numpy(np.abs(clipped[WARMUP:]).sum(axis=0)).to(device).clamp_min(1e-12)
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        # Minibatches bound activation memory. Gradients accumulate for one
        # sequence, preserving its full post-warm-up weight denominator.
        for start in range(0, len(rows), minibatch_size):
            chosen = rows[start:start + minibatch_size]
            encoded = torch.from_numpy(representation[chosen]).to(device)
            truth = torch.from_numpy(clipped[chosen]).to(device)
            predicted = torch.from_numpy(base[chosen]).to(device) + delta(encoded)
            loss = ((truth.abs() * (predicted - truth).square()).sum(dim=0) * inverse_probability / denominators).mean()
            loss.backward()
            total_loss += float(loss.detach())
        norm = float(torch.nn.utils.clip_grad_norm_(delta.parameters(), 1.0))
        if not np.isfinite(norm):
            raise FloatingPointError("nonfinite frozen-head gradient")
        optimizer.step()
        if not all(torch.isfinite(parameter).all().item() for parameter in delta.parameters()):
            raise FloatingPointError("nonfinite frozen-head parameters after optimizer step")
        processed += 1
        cursor["position"] += 1
        cursor["updates"] += 1
        cursor["sequences_seen"] += 1
        history.append({"epoch": epoch, "sequence_id": int(seq), "supervised_loss": total_loss,
                        "sampled_rows": len(rows), "gradient_norm": norm})
        if cursor["position"] == len(selected):
            cursor["epoch"] += 1
            cursor["position"] = 0
        _atomic_checkpoint(output_dir / "checkpoint.pt", checkpoint())
        if progress and (processed % 32 == 0 or cursor["epoch"] >= epochs):
            print(f"AdamW head {cursor['sequences_seen']}/{epochs*len(selected)} sequences; {time.monotonic()-started:.1f}s", flush=True)
    with torch.no_grad():
        fitted.head.weight.add_(delta.weight)
        fitted.head.bias.add_(delta.bias)
    state = checkpoint()
    state.update(model_config=fitted.config, model_state=fitted.state_dict())
    return _finish(fitted, output_dir,
                   {"method": "head_adamw", "optimizer": "AdamW", "seed": seed,
                    "complete": cursor["epoch"] >= epochs, "requested_epochs": epochs,
                    "cursor": cursor, "history": history, "specification": specification,
                    "encoder": encoder_provenance,
                    "seconds": time.monotonic() - started}, state)


def fit_head(model_path, train_path, output_dir, *, head_state: str = "last",
             optimizer: str = "adamw", learning_rate: float = 1e-3,
             minibatch_size: int = 4096, epochs: int = 1, penalty: float = 1e-3,
             ridge: float | None = None, **kwargs) -> dict:
    """Fit a frozen recurrent model's residual head, fused into one ONNX model.

    Accepts max_sequences, group_indices, ridge, rows_per_sequence, chunk_size,
    seed, device, threads, resume_from, and max_new_sequences as keyword args.
    AdamW is the default, with learning rate 1e-3 and row minibatches of 4096.
    Its decay penalizes the zero-initialized residual, preserving the original
    head as the reference. ``optimizer='ridge'`` selects a deterministic
    closed-form CPU alternative. ``ridge`` overrides ``penalty`` when given.
    ``max_new_sequences`` provides a resumable bounded run.
    """
    strength = penalty if ridge is None else ridge
    if optimizer == "ridge":
        return _fit_linear(model_path, train_path, output_dir, kind="head", head_state=head_state,
                           ridge=strength, **kwargs)
    if optimizer != "adamw":
        raise ValueError("optimizer must be 'adamw' or 'ridge'")
    return _fit_head_adamw(model_path, train_path, output_dir, head_state=head_state,
                           penalty=strength, learning_rate=learning_rate,
                           minibatch_size=minibatch_size, epochs=epochs, **kwargs)


def fit_affine(model_path, train_path, output_dir, *, ridge: float = 1e-3,
               penalty: float | None = None, **kwargs) -> dict:
    """Fit a 2x2 affine residual output map and fuse it into the original head."""
    return _fit_linear(model_path, train_path, output_dir, kind="affine",
                       ridge=ridge if penalty is None else penalty, **kwargs)


def sequence_weight_denominators(targets: torch.Tensor) -> torch.Tensor:
    """Full-sequence denominators [batch, 2], excluding the 99 warm-up rows."""
    if targets.ndim != 3 or targets.shape[-1] != 2 or targets.shape[0] <= WARMUP:
        raise ValueError("targets must have shape [time > 99, batch, 2]")
    return targets[WARMUP:].clamp(-2.0, 2.0).abs().sum(dim=0)


def weighted_sequence_chunk_loss(prediction: torch.Tensor, target: torch.Tensor,
                                 denominators: torch.Tensor, *, start_step: int) -> torch.Tensor:
    """A chunk's additive contribution to the mean full-sequence target loss.

    Denominators must be computed once over the entire sequence, not the chunk.
    Prediction values are unbounded in this MSE surrogate; clipping them would
    introduce zero gradients outside the metric boundary.
    """
    if prediction.shape != target.shape or target.ndim != 3 or target.shape[-1] != 2:
        raise ValueError("matching [time, batch, 2] tensors required")
    if denominators.shape != target.shape[1:]:
        raise ValueError("denominators must have shape [batch, 2]")
    clipped = target.clamp(-2.0, 2.0)
    keep = (torch.arange(len(target), device=target.device) + start_step >= WARMUP)[:, None, None]
    weights = clipped.abs() * keep
    per_side = (weights * (prediction - clipped).square()).sum(dim=0) / denominators.clamp_min(1e-12)
    return per_side.mean()


def _rng_state() -> dict:
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def _restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _train_neural(model_path, train_path, output_dir, *, method: str,
                  hidden_size: int = 96, head_only: bool = False,
                  max_sequences: int | None = 32, group_indices: Iterable[int] | None = None,
                  epochs: int = 1, batch_size: int = 8, chunk_size: int = 256,
                  learning_rate: float = 1e-5, weight_decay: float = 1e-4,
                  distill_weight: float = 0.1, gradient_clip: float = 1.0,
                  teacher_path: str | Path | None = None, seed: int = DEFAULT_SEED,
                  device: str = "cpu", threads: int = 1,
                  resume_from: str | Path | None = None, max_updates: int | None = None) -> dict:
    if method not in {"fine_tune", "student"}:
        raise ValueError("unknown neural training method")
    if epochs <= 0 or batch_size <= 0 or chunk_size <= 0 or learning_rate <= 0 or weight_decay < 0 or distill_weight < 0 or gradient_clip <= 0:
        raise ValueError("invalid training hyperparameters")
    if max_updates is not None and max_updates <= 0:
        raise ValueError("max_updates must be positive")
    started = time.monotonic()
    seed_everything(seed, threads)
    selected = _groups(train_path, max_sequences, seed, group_indices)
    source = load_torch_model(model_path, device=device)
    model = copy.deepcopy(source) if method == "fine_tune" else StreamingGRU(hidden_size).to(device)
    teacher = load_torch_model(teacher_path, device=device) if teacher_path is not None else source
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if head_only:
        for layer in (model.gru0, model.gru1):
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],
                                 lr=learning_rate, weight_decay=weight_decay)
    specification = {"method": method, "source": _fingerprint(model_path, content=True),
                     "teacher": _fingerprint(teacher_path, content=True) if teacher_path is not None else None,
                     "data": _fingerprint(train_path), "groups": selected,
                     "model_config": model.config, "head_only": head_only,
                     "batch_size": batch_size, "chunk_size": chunk_size,
                     "learning_rate": learning_rate, "weight_decay": weight_decay,
                     "distill_weight": distill_weight, "gradient_clip": gradient_clip, "seed": seed}
    cursor = {"epoch": 0, "position": 0, "updates": 0, "sequences_seen": 0}
    history = []
    if resume_from is not None:
        previous = _read_checkpoint(resume_from)
        _check_resume(previous, specification)
        model.load_state_dict(previous["model_state"])
        optimizer.load_state_dict(previous["optimizer_state"])
        cursor = previous["cursor"]
        history = previous["history"]
        _restore_rng(previous["rng"])
    output_dir = Path(output_dir).resolve()

    def checkpoint():
        return {"format": CHECKPOINT_FORMAT, "specification": specification,
                "model_config": model.config, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "cursor": dict(cursor),
                "history": list(history), "rng": _rng_state()}

    _atomic_checkpoint(output_dir / "checkpoint.pt", checkpoint())
    model.train()
    while cursor["epoch"] < epochs and (max_updates is None or cursor["updates"] < max_updates):
        epoch = cursor["epoch"]
        order = np.random.default_rng(seed + epoch).permutation(selected).tolist()
        start_position = cursor["position"]
        batch_groups = order[start_position:start_position + batch_size]
        sequences = list(_iter_sequences(train_path, batch_groups))
        x = torch.from_numpy(np.stack([sequence[1] for sequence in sequences], axis=1)).to(device)
        y = torch.from_numpy(np.stack([sequence[2] for sequence in sequences], axis=1)).to(device)
        denominators = sequence_weight_denominators(y)
        states = model.initial_states(len(sequences))
        teacher_states = teacher.initial_states(len(sequences))
        optimizer.zero_grad(set_to_none=True)
        supervised_value = 0.0
        distill_value = 0.0
        for start in range(0, len(x), chunk_size):
            end = min(start + chunk_size, len(x))
            prediction, *states = model(x[start:end], *states)
            supervised = weighted_sequence_chunk_loss(prediction, y[start:end], denominators, start_step=start)
            loss = supervised
            supervised_value += float(supervised.detach())
            if distill_weight:
                with torch.no_grad():
                    teacher_prediction, *teacher_states = teacher(x[start:end], *teacher_states)
                keep_start = max(0, WARMUP - start)
                difference = prediction[keep_start:] - teacher_prediction[keep_start:]
                match = difference.square().sum() / ((len(x) - WARMUP) * len(sequences) * 2)
                loss = loss + distill_weight * match
                distill_value += float(match.detach())
            loss.backward()
            states = [state.detach() for state in states]
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip))
        if not np.isfinite(gradient_norm):
            raise FloatingPointError("nonfinite gradient; last completed batch is checkpointed")
        optimizer.step()
        if not all(torch.isfinite(parameter).all().item() for parameter in model.parameters()):
            raise FloatingPointError("nonfinite model parameters after optimizer step")
        cursor["updates"] += 1
        cursor["sequences_seen"] += len(sequences)
        cursor["position"] += len(sequences)
        history.append({"epoch": epoch, "update": cursor["updates"],
                        "sequence_ids": [int(sequence[0]) for sequence in sequences],
                        "supervised_loss": supervised_value, "distillation_loss": distill_value,
                        "gradient_norm": gradient_norm})
        if cursor["position"] == len(selected):
            cursor["epoch"] += 1
            cursor["position"] = 0
        _atomic_checkpoint(output_dir / "checkpoint.pt", checkpoint())
    model.eval()
    return _finish(model, output_dir,
                   {"method": method, "seed": seed, "complete": cursor["epoch"] >= epochs,
                    "requested_epochs": epochs, "cursor": cursor, "history": history,
                    "seconds": time.monotonic() - started, "specification": specification}, checkpoint())


def fine_tune(model_path, train_path, output_dir, *, distill_weight: float = 0.1,
               teacher_weight: float | None = None, **kwargs) -> dict:
    """Fine-tune recovered GRU weights with carried, detached TBPTT state.

    One optimizer update follows a whole sequence batch.  Chunk losses use
    full-sequence denominators, so chunking does not reweight quiet periods.
    Checkpoints are resumable at batch boundaries via ``resume_from``.
    """
    return _train_neural(model_path, train_path, output_dir, method="fine_tune",
                         distill_weight=distill_weight if teacher_weight is None else teacher_weight, **kwargs)


def train_student(model_path, train_path, output_dir, *, hidden_size: int = 96,
                  learning_rate: float = 1e-3, distill_weight: float = 1.0,
                  teacher_weight: float | None = None, **kwargs) -> dict:
    """Train a compact GRU with labels and the supplied model as teacher."""
    return _train_neural(model_path, train_path, output_dir, method="student", hidden_size=hidden_size,
                         learning_rate=learning_rate,
                         distill_weight=distill_weight if teacher_weight is None else teacher_weight, **kwargs)


def resume_training(checkpoint_path, output_dir=None, *, epochs: int | None = None,
                    max_updates: int | None = None, max_new_sequences: int | None = None,
                    device: str = "cpu", threads: int = 1) -> dict:
    """Resume using saved inputs, sequence selection, seed and hyperparameters.

    Neural ``epochs`` is the total desired epoch count (default: one additional
    epoch beyond the current completed count). ``max_updates`` is also total,
    not additional. Linear fits continue their originally selected groups.
    """
    previous = _read_checkpoint(checkpoint_path)
    spec = previous["specification"]
    output_dir = output_dir or Path(checkpoint_path).resolve().parent
    common = dict(model_path=spec["source"]["path"], train_path=spec["data"]["path"],
                  output_dir=output_dir, group_indices=spec["groups"], max_sequences=None,
                  seed=spec["seed"], chunk_size=spec["chunk_size"], device=device,
                  threads=threads, resume_from=checkpoint_path)
    if spec["method"] in {"head", "affine"}:
        return _fit_linear(**common, kind=spec["method"], head_state=spec["head_state"],
                           ridge=spec["ridge"], rows_per_sequence=spec["rows_per_sequence"],
                           encoder=spec.get("encoder", "torch"),
                           max_new_sequences=max_new_sequences)
    if spec["method"] == "head_adamw":
        return _fit_head_adamw(**common, head_state=spec["head_state"], penalty=spec["penalty"],
                               learning_rate=spec["learning_rate"], minibatch_size=spec["minibatch_size"],
                               rows_per_sequence=spec["rows_per_sequence"], max_new_sequences=max_new_sequences,
                               encoder=spec.get("encoder", "torch"),
                               epochs=epochs if epochs is not None else previous["cursor"]["epoch"] + 1)
    kwargs = {key: spec[key] for key in ("head_only", "batch_size", "learning_rate", "weight_decay", "distill_weight", "gradient_clip")}
    kwargs.update(hidden_size=spec["model_config"]["hidden_size"],
                  teacher_path=spec["teacher"]["path"] if spec["teacher"] else None,
                  epochs=epochs if epochs is not None else previous["cursor"]["epoch"] + 1,
                  max_updates=max_updates)
    return _train_neural(**common, method=spec["method"], **kwargs)
