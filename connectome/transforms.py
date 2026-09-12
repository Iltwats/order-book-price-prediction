"""Fold an output calibration into an ONNX linear head, without changing GRUs."""
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

from .evaluation import file_hash, save_json


def fold_output_transform(source, destination, *, scales=(1, 1), affine=None):
    """Preserve the graph and recurrent tensors; only replace final head constants.

    Floating-point reassociation can change predictions slightly. Validate each
    exported candidate before selection; a mathematical identity is not parity.
    EMA is causal state and cannot be folded into a static linear head.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("export to a separate path to preserve the source model")
    model = onnx.load(str(source))
    initializers = {value.name: value for value in model.graph.initializer}
    matmuls = [node for node in model.graph.node if node.op_type == "MatMul"]
    if len(matmuls) != 1:
        raise ValueError("expected exactly one final linear MatMul")
    matmul = matmuls[0]
    adds = [node for node in model.graph.node
            if node.op_type == "Add" and matmul.output[0] in node.input]
    if len(adds) != 1:
        raise ValueError("expected one linear head bias Add")
    add = adds[0]
    bias_name = next(name for name in add.input if name != matmul.output[0])
    if matmul.input[1] not in initializers or bias_name not in initializers:
        raise ValueError("head weights and bias must be direct initializers")
    weight_tensor, bias_tensor = initializers[matmul.input[1]], initializers[bias_name]
    weights = numpy_helper.to_array(weight_tensor)
    bias = numpy_helper.to_array(bias_tensor)
    scale = np.asarray(scales, np.float32)
    affine = affine or {}
    matrix = np.asarray(affine.get("matrix", np.eye(2)), np.float32)
    shift = np.asarray(affine.get("bias", [0, 0]), np.float32)
    if (weights.ndim != 2 or weights.shape[1] != 2 or bias.shape != (2,)
            or weights.dtype != np.float32 or bias.dtype != np.float32):
        raise ValueError("expected an FP32 two-output linear head")
    if scale.shape != (2,) or matrix.shape != (2, 2) or shift.shape != (2,):
        raise ValueError("invalid output calibration shapes")
    if not all(np.isfinite(x).all() for x in [weights, bias, scale, matrix, shift]):
        raise ValueError("calibration and weights must be finite")
    # Apply in the same algebraic order as calibration.apply_calibration.
    new_weights = (weights * scale) @ matrix.T
    new_bias = (bias * scale) @ matrix.T + shift
    weight_tensor.CopyFrom(numpy_helper.from_array(new_weights.astype(np.float32), weight_tensor.name))
    bias_tensor.CopyFrom(numpy_helper.from_array(new_bias.astype(np.float32), bias_tensor.name))
    onnx.checker.check_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(destination))
    result = {"source_sha256": file_hash(source), "model_sha256": file_hash(destination),
              "scales": scale.tolist(), "affine": affine,
              "changed_initializers": [weight_tensor.name, bias_tensor.name],
              "recurrent_weights_unchanged": True, "additional_operations": 0,
              "parity_required": True}
    save_json(destination.with_suffix(".transform.json"), result)
    return result
