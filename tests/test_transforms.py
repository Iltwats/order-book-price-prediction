from pathlib import Path

import numpy as np
import onnx
import pytest

from connectome.runtime import OnnxBackend
from connectome.transforms import fold_output_transform


BASELINE = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"


def test_folded_half_scale_has_exact_predictions_and_unchanged_gru(tmp_path):
    destination = tmp_path / "half.onnx"
    result = fold_output_transform(BASELINE, destination, scales=[.5, .5])
    before, after = onnx.load(BASELINE), onnx.load(destination)
    assert [n.SerializeToString() for n in before.graph.node] == [n.SerializeToString() for n in after.graph.node]
    changed = set(result["changed_initializers"])
    for first, second in zip(before.graph.initializer, after.graph.initializer):
        if first.name not in changed:
            assert first.SerializeToString() == second.SerializeToString()
    original, folded = OnnxBackend(BASELINE), OnnxBackend(destination)
    for features in np.random.default_rng(19).normal(size=(300, 112)).astype(np.float32):
        np.testing.assert_array_equal(folded.step(features), original.step(features) * .5)
        for actual, expected in zip(folded.hidden_states(), original.hidden_states()):
            np.testing.assert_array_equal(actual, expected)


def test_transform_cannot_overwrite_source_or_accept_nonfinite_scale(tmp_path):
    with pytest.raises(ValueError, match="separate path"):
        fold_output_transform(BASELINE, BASELINE)
    with pytest.raises(ValueError, match="finite"):
        fold_output_transform(BASELINE, tmp_path / "bad.onnx", scales=[np.nan, 1])
