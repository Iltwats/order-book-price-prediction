"""Offline ONNX inspection, strict baseline weight extraction, and simplification.

The native weight format is one C-contiguous float32 vector in WEIGHT_SHAPES
insertion order. GRUs use ONNX z,r,h gates and linear_before_reset=1. Runtime
submission code can load the resulting .npy without importing this module/ONNX.
"""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference

WEIGHT_SHAPES = {
    "w0": (384, 112), "r0": (384, 128), "b0": (768,),
    "w1": (384, 128), "r1": (384, 128), "b1": (768,),
    "head_w": (128, 2), "head_b": (2,),
}
WEIGHT_COUNT = sum(int(np.prod(shape)) for shape in WEIGHT_SHAPES.values())
INPUT_SHAPES = {"features": (1, 1, 112), "hidden_0": (1, 1, 128), "hidden_1": (1, 1, 128)}
OUTPUT_SHAPES = {"prediction": (1, 1, 2), "next_hidden_0": (1, 1, 128), "next_hidden_1": (1, 1, 128)}


def _shape(value):
    tensor = value.type.tensor_type
    return tuple(dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                 for dim in tensor.shape.dim)


def _read(path):
    model = onnx.load_model(str(path), load_external_data=False)
    if any(tensor.external_data for tensor in model.graph.initializer):
        raise ValueError("external ONNX tensor data is not supported by this weight exporter")
    onnx.checker.check_model(model)
    return model


def _attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _validate_and_extract(model):
    if len(model.opset_import) != 1 or model.opset_import[0].domain not in {"", "ai.onnx"}:
        raise ValueError("expected a single standard ONNX operator domain")
    if not 14 <= model.opset_import[0].version <= 17:
        raise ValueError("supported model opsets are 14 through 17")
    for values, expected in ((model.graph.input, INPUT_SHAPES), (model.graph.output, OUTPUT_SHAPES)):
        if {value.name for value in values} != set(expected):
            raise ValueError(f"unexpected graph input/output names; expected {tuple(expected)}")
        for value in values:
            if value.type.tensor_type.elem_type != TensorProto.FLOAT or _shape(value) != expected[value.name]:
                raise ValueError(f"expected fixed FP32 shape {expected[value.name]} for {value.name}")
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    producers = {output: node for node in graph.node for output in node.output if output}
    shapes = {value.name: _shape(value) for value in [*graph.input, *graph.output, *graph.value_info]}
    tensors = {tensor.name: np.asarray(numpy_helper.to_array(tensor)) for tensor in graph.initializer}
    allowed = {"GRU", "MatMul", "Add", "Constant", "Transpose", "Squeeze", "Unsqueeze", "Reshape", "Identity"}
    if any(node.domain or node.op_type not in allowed for node in graph.node):
        raise ValueError("graph contains unsupported operations")
    for node in graph.node:
        if node.op_type == "Constant":
            attributes = _attributes(node)
            if set(attributes) != {"value"}:
                raise ValueError("only tensor-valued Constant nodes are supported")
            tensors[node.output[0]] = np.asarray(numpy_helper.to_array(attributes["value"]))

    used_nodes = set()

    def tensor(name):
        node = producers.get(name)
        if node is not None and node.op_type == "Identity":
            used_nodes.add(node.name or node.output[0])
            return tensor(node.input[0])
        if name not in tensors:
            raise ValueError(f"expected constant tensor {name}")
        if node is not None:
            used_nodes.add(node.name or node.output[0])
        return tensors[name]

    def unwrap(name):
        seen = set()
        while name in producers and producers[name].op_type in {"Identity", "Transpose", "Squeeze", "Unsqueeze", "Reshape"}:
            if name in seen:
                raise ValueError("cycle in shape operations")
            seen.add(name)
            node = producers[name]
            used_nodes.add(node.name or node.output[0])
            source = node.input[0]
            before, after = shapes.get(source), shapes.get(name)
            if before is None or after is None or any(not isinstance(d, int) or d < 1 for d in (*before, *after)):
                raise ValueError("shape-only path must have known positive dimensions")
            if np.prod(before) != np.prod(after):
                raise ValueError("shape-only path changes tensor element count")
            if node.op_type == "Transpose":
                permutation = _attributes(node).get("perm", list(reversed(range(len(before)))))
                # Permuting singleton axes is harmless; preserve the order of all
                # non-singleton axes so flattening is unchanged.
                if [axis for axis in permutation if before[axis] != 1] != [i for i, d in enumerate(before) if d != 1]:
                    raise ValueError("transpose changes data order")
            for auxiliary in node.input[1:]:
                if auxiliary:
                    tensor(auxiliary)
            name = source
        return name

    grus = [node for node in graph.node if node.op_type == "GRU"]
    if len(grus) != 2:
        raise ValueError("expected exactly two GRU nodes")
    by_hidden = {node.input[5]: node for node in grus if len(node.input) == 6}
    if set(by_hidden) != {"hidden_0", "hidden_1"}:
        raise ValueError("GRU initial-state wiring does not match the callback")
    grus = [by_hidden["hidden_0"], by_hidden["hidden_1"]]
    weights = {}
    for index, node in enumerate(grus):
        used_nodes.add(node.name or node.output[0])
        attributes = _attributes(node)
        if (attributes.get("hidden_size") != 128 or attributes.get("linear_before_reset", 0) != 1 or
                attributes.get("direction", b"forward") != b"forward" or attributes.get("layout", 0) != 0):
            raise ValueError("expected forward GRU128, layout=0, linear_before_reset=1")
        if set(attributes) - {"hidden_size", "linear_before_reset", "direction", "layout", "activations"}:
            raise ValueError("custom GRU clipping/activation parameters are unsupported")
        if attributes.get("activations", [b"Sigmoid", b"Tanh"]) != [b"Sigmoid", b"Tanh"]:
            raise ValueError("expected standard Sigmoid/Tanh GRU activations")
        if node.input[4] or len(node.output) != 2:
            raise ValueError("sequence lengths must be omitted and final hidden output must be present")
        if unwrap(f"next_hidden_{index}") != node.output[1]:
            raise ValueError("final hidden output wiring does not match")
        source = unwrap(node.input[0])
        if index == 0 and source != "features":
            raise ValueError("first GRU does not consume the unmodified feature vector")
        if index == 1 and source not in {output for output in grus[0].output if output}:
            raise ValueError("second GRU does not consume the first GRU output")
        for key, input_index in ((f"w{index}", 1), (f"r{index}", 2), (f"b{index}", 3)):
            value = tensor(node.input[input_index])
            expected = (1, *WEIGHT_SHAPES[key])
            if value.dtype != np.float32 or value.shape != expected:
                raise ValueError(f"{key} must have shape {expected} and dtype float32")
            weights[key] = np.ascontiguousarray(value[0])
    matmuls = [node for node in graph.node if node.op_type == "MatMul"]
    adds = [node for node in graph.node if node.op_type == "Add"]
    if len(matmuls) != 1 or len(adds) != 1:
        raise ValueError("expected a single MatMul plus Add output head")
    matmul, add = matmuls[0], adds[0]
    used_nodes.update((matmul.name or matmul.output[0], add.name or add.output[0]))
    if unwrap(matmul.input[0]) not in {output for output in grus[1].output if output}:
        raise ValueError("head must consume only the second GRU output")
    weights["head_w"] = tensor(matmul.input[1])
    if unwrap("prediction") != add.output[0]:
        raise ValueError("prediction must be the unmodified linear head output")
    head_inputs = [unwrap(name) for name in add.input]
    if head_inputs.count(matmul.output[0]) != 1:
        raise ValueError("head bias Add does not consume the MatMul")
    bias_index = 1 - head_inputs.index(matmul.output[0])
    weights["head_b"] = tensor(add.input[bias_index])
    for key, value in weights.items():
        if value.dtype != np.float32 or value.shape != WEIGHT_SHAPES[key] or not np.isfinite(value).all():
            raise ValueError(f"invalid finite FP32 weights for {key}: expected {WEIGHT_SHAPES[key]}")
        weights[key] = np.array(value, dtype=np.float32, copy=True, order="C")
    # Unused Constant nodes are benign export debris; other dead operations can
    # hide a topology mismatch and are deliberately rejected.
    if any((node.name or node.output[0]) not in used_nodes and node.op_type != "Constant" for node in graph.node):
        raise ValueError("graph contains additional operations outside the supported network")
    return weights


def load_weights(path):
    """Return owned FP32 arrays with standardized WEIGHT_SHAPES keys."""
    return _validate_and_extract(_read(path))


def export_native_weights(source, destination=None):
    """Flatten validated weights, optionally writing a .npy at destination.

    source may be an ONNX path or a standardized weight dictionary. The return
    value always owns a contiguous one-dimensional FP32 array of 192258 scalars.
    """
    weights = load_weights(source) if not isinstance(source, dict) else source
    if set(weights) != set(WEIGHT_SHAPES):
        raise ValueError("weight dictionary keys do not match the native format")
    parts = []
    for key, expected in WEIGHT_SHAPES.items():
        value = np.asarray(weights[key])
        if value.dtype != np.float32 or value.shape != expected or not np.isfinite(value).all():
            raise ValueError(f"{key} must contain finite float32 values with shape {expected}")
        parts.append(value.reshape(-1))
    flat = np.ascontiguousarray(np.concatenate(parts), dtype=np.float32)
    if destination is not None:
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("wb") as handle:
            np.save(handle, flat, allow_pickle=False)
    return flat


def simplify_model(source, destination):
    """Create the fixed-one-step 2-GRU/MatMul/Add graph with original FP32 weights."""
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("simplification must not overwrite the source model")
    weights = load_weights(source)
    initializers = []
    for name, array in weights.items():
        array = array[None, ...] if name in {"w0", "r0", "b0", "w1", "r1", "b1"} else array
        initializers.append(numpy_helper.from_array(array, name=name))
    nodes = [
        helper.make_node("GRU", ["features", "w0", "r0", "b0", "", "hidden_0"],
                         ["", "next_hidden_0"], name="gru0", hidden_size=128, linear_before_reset=1),
        helper.make_node("GRU", ["next_hidden_0", "w1", "r1", "b1", "", "hidden_1"],
                         ["", "next_hidden_1"], name="gru1", hidden_size=128, linear_before_reset=1),
        helper.make_node("MatMul", ["next_hidden_1", "head_w"], ["head_product"], name="readout_matmul"),
        helper.make_node("Add", ["head_product", "head_b"], ["prediction"], name="readout_add"),
    ]
    graph = helper.make_graph(nodes, "connectome_fixed_step",
                              [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in INPUT_SHAPES.items()],
                              [helper.make_tensor_value_info(name, TensorProto.FLOAT, shape) for name, shape in OUTPUT_SHAPES.items()],
                              initializers)
    model = helper.make_model(graph, producer_name="connectome.fixed_step", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.doc_string = "Fixed sequence length=1, batch=1; forward GRU Y_h replaces redundant sequence outputs."
    onnx.checker.check_model(model)
    _validate_and_extract(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination))
    return inspect_model(destination)


def export_sequence_model(source, destination):
    """Offline-only batch/time export preserving the original graph's nodes.

    Features/predictions become batch-first dynamic sequences. This model is for
    accelerated offline evaluation or state extraction, never a replacement for
    the judge's fixed-step callback. Compare full sequences against row-by-row
    inference before trusting bulk results.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("sequence export must not overwrite its source")
    model = _read(source)
    _validate_and_extract(model)
    producers = {output: node for node in model.graph.node for output in node.output if output}
    first_gru = next(node for node in model.graph.node if node.op_type == "GRU" and node.input[5] == "hidden_0")
    first_transpose = producers.get(first_gru.input[0])
    head = next(node for node in model.graph.node if node.op_type == "MatMul")
    last_transpose = producers.get(head.input[0])
    if (first_transpose is None or first_transpose.op_type != "Transpose" or
            list(first_transpose.input) != ["features"] or
            list(_attributes(first_transpose).get("perm", [])) != [1, 0, 2] or
            last_transpose is None or last_transpose.op_type != "Transpose" or
            list(_attributes(last_transpose).get("perm", [])) != [1, 0, 2]):
        raise ValueError("sequence export requires the original batch-first Transpose/GRU/Transpose layout")
    dynamic_shapes = {
        "features": ("batch", "time", 112), "prediction": ("batch", "time", 2),
        "hidden_0": (1, "batch", 128), "hidden_1": (1, "batch", 128),
        "next_hidden_0": (1, "batch", 128), "next_hidden_1": (1, "batch", 128),
    }
    for value in [*model.graph.input, *model.graph.output]:
        value.CopyFrom(helper.make_tensor_value_info(value.name, TensorProto.FLOAT, dynamic_shapes[value.name]))
    del model.graph.value_info[:]
    model.doc_string += "\nOFFLINE ONLY: dynamic batch-first sequences; verify row-by-row parity before bulk evaluation."
    model = shape_inference.infer_shapes(model, strict_mode=True)
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(destination))
    return inspect_model(destination)


def inspect_model(path):
    """Return JSON-compatible graph metadata, artifact hash and native support."""
    path = Path(path)
    model = _read(path)
    def describe(value):
        return {"name": value.name, "shape": list(_shape(value)),
                "dtype": TensorProto.DataType.Name(value.type.tensor_type.elem_type)}
    metadata = {
        "path": str(path.resolve()), "sha256": sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size, "ir_version": model.ir_version,
        "producer_name": model.producer_name, "producer_version": model.producer_version,
        "opsets": [{"domain": item.domain, "version": item.version} for item in model.opset_import],
        "inputs": [describe(value) for value in model.graph.input],
        "outputs": [describe(value) for value in model.graph.output],
        "node_count": len(model.graph.node),
        "operator_counts": dict(Counter(node.op_type for node in model.graph.node)),
        "initializer_count": len(model.graph.initializer),
        "initializer_scalars": sum(int(np.prod(tensor.dims)) for tensor in model.graph.initializer),
    }
    try:
        _validate_and_extract(model)
        metadata["native_supported"] = True
        metadata["native_unsupported_reason"] = None
    except ValueError as error:
        metadata["native_supported"] = False
        metadata["native_unsupported_reason"] = str(error)
    return metadata
