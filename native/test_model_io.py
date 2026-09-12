"""Meaningful structural guards for exporting the supplied pretrained graph."""
from pathlib import Path

import numpy as np
import onnx
import pytest

from connectome.model_io import (
    WEIGHT_COUNT, export_native_weights, export_sequence_model, inspect_model,
    load_weights, simplify_model,
)

BASELINE = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"


def test_weight_roundtrip_and_four_node_simplification(tmp_path):
    destination = tmp_path / "simplified.onnx"
    metadata = simplify_model(BASELINE, destination)
    assert metadata["operator_counts"] == {"GRU": 2, "MatMul": 1, "Add": 1}
    assert metadata["native_supported"]
    original = load_weights(BASELINE)
    simplified = load_weights(destination)
    assert original.keys() == simplified.keys()
    for key in original:
        np.testing.assert_array_equal(original[key], simplified[key])
    weights_file = tmp_path / "weights.data"
    flat = export_native_weights(original, weights_file)
    assert flat.shape == (WEIGHT_COUNT,)
    assert flat.flags.c_contiguous and flat.flags.owndata
    np.testing.assert_array_equal(np.load(weights_file, allow_pickle=False), flat)


def test_rejects_wrong_reset_equation(tmp_path):
    model = onnx.load(BASELINE)
    gru = next(node for node in model.graph.node if node.op_type == "GRU")
    attribute = next(value for value in gru.attribute if value.name == "linear_before_reset")
    attribute.i = 0
    destination = tmp_path / "wrong_reset.onnx"
    onnx.save(model, destination)
    with pytest.raises(ValueError, match="linear_before_reset"):
        load_weights(destination)
    assert not inspect_model(destination)["native_supported"]


def test_rejects_head_on_wrong_layer(tmp_path):
    model = onnx.load(BASELINE)
    head = next(node for node in model.graph.node if node.op_type == "MatMul")
    head.input[0] = "next_hidden_0"
    destination = tmp_path / "wrong_head.onnx"
    onnx.save(model, destination)
    with pytest.raises(ValueError, match="second GRU"):
        load_weights(destination)


def test_rejects_nonfinite_or_wrong_weight_shapes():
    weights = load_weights(BASELINE)
    weights["head_b"][0] = np.nan
    with pytest.raises(ValueError, match="head_b"):
        export_native_weights(weights)
    weights = load_weights(BASELINE)
    weights["head_w"] = np.zeros((256, 2), np.float32)
    with pytest.raises(ValueError, match="head_w"):
        export_native_weights(weights)


def test_sequence_export_keeps_original_nodes_and_is_not_native(tmp_path):
    destination = tmp_path / "offline_sequence.onnx"
    metadata = export_sequence_model(BASELINE, destination)
    original = onnx.load(BASELINE)
    exported = onnx.load(destination)
    assert [node.SerializeToString() for node in exported.graph.node] == [node.SerializeToString() for node in original.graph.node]
    assert metadata["inputs"][0]["shape"] == ["batch", "time", 112]
    assert metadata["outputs"][0]["shape"] == ["batch", "time", 2]
    assert not metadata["native_supported"]
    simplified = tmp_path / "simple.onnx"
    simplify_model(BASELINE, simplified)
    with pytest.raises(ValueError, match="original batch-first"):
        export_sequence_model(simplified, tmp_path / "invalid_sequence.onnx")


def test_export_never_overwrites_source(tmp_path):
    source = tmp_path / "source.onnx"
    source.write_bytes(BASELINE.read_bytes())
    with pytest.raises(ValueError, match="overwrite"):
        simplify_model(source, source)
    with pytest.raises(ValueError, match="overwrite"):
        export_sequence_model(source, source)
    assert source.read_bytes() == BASELINE.read_bytes()
