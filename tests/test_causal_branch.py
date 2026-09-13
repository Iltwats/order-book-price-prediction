"""Small synthetic-only dynamics graph/stream contract tests; no dataset scans."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import onnx
import pytest

from connectome.causal_branch import (
    ALPHA, AUX_WIDTH, DynamicsBackend, DynamicsReference, TrainingNormalization,
    export_dynamics, fit_training_normalization, folded_projection, make_fixture,
    normalization_folding_diagnostic,
)
from connectome.runtime import OnnxBackend, StreamingPredictor
from utils import DataPoint

BASELINE = Path(__file__).resolve().parents[1] / "baseline/baseline.onnx"


def synthetic_rows(seed=81, count=173):
    return np.random.default_rng(seed).normal(0.1, 0.3, (count, 112)).astype(np.float32)


@pytest.fixture(scope="module")
def graphs(tmp_path_factory):
    directory = tmp_path_factory.mktemp("dynamics")
    rows = synthetic_rows(count=201)
    normalization = fit_training_normalization(
        (rows[:60], rows[60:]), split="train", groups=[0],
        data_fingerprint={"kind": "synthetic_test_fixture", "seed": 81})
    parameters = make_fixture(seed=13, nonzero=True, normalization=normalization)
    identity = replace(parameters, output_weight=np.zeros((16, 2), np.float32))
    paths = {}
    for name, value, folded in [("zero", identity, True), ("nonzero", parameters, True),
                                ("unfolded", parameters, False)]:
        path = directory / f"{name}.onnx"
        export_dynamics(BASELINE, path, value, fold_normalization=folded)
        paths[name] = path
    return parameters, identity, paths


def assert_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def test_training_normalization_is_bounded_population_fit_and_records_origin():
    rows = synthetic_rows(count=100)
    rows[:, 4] = np.float32(7)
    normalization = fit_training_normalization(
        (rows[start:start+17] for start in range(0, len(rows), 17)), split="train",
        groups=[8, 3], data_fingerprint={"kind": "synthetic", "version": 1})
    expected = rows.astype(np.float64)
    np.testing.assert_allclose(normalization.mean, expected.mean(0), atol=1e-7, rtol=1e-6)
    std = expected.std(0)
    np.testing.assert_allclose(normalization.scale, np.where(std < 1e-6, 1, std), atol=1e-7, rtol=1e-6)
    assert normalization.provenance["near_constant_columns"] == [4]
    assert normalization.provenance["groups"] == [8, 3]
    assert normalization.provenance["rows"] == 100
    assert len(normalization.record()["statistics_sha256"]) == 64


@pytest.mark.parametrize("options,error", [
    ({"split": "valid"}, "training split"),
    ({"groups": [1, 1]}, "unique"),
    ({"data_fingerprint": {}}, "fingerprint"),
])
def test_normalization_rejects_invalid_declared_provenance(options, error):
    arguments = {"split": "train", "groups": [1], "data_fingerprint": {"kind": "synthetic"}}
    arguments.update(options)
    with pytest.raises(ValueError, match=error):
        fit_training_normalization([synthetic_rows(count=5)], **arguments)


def test_normalization_rejects_nonfinite_rows_and_empty_input():
    arguments = {"split": "train", "groups": [0], "data_fingerprint": {"kind": "synthetic"}}
    bad = synthetic_rows(count=2)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_training_normalization([bad], **arguments)
    with pytest.raises(ValueError, match="at least one"):
        fit_training_normalization([], **arguments)
    with pytest.raises(ValueError, match="positive"):
        TrainingNormalization(np.zeros(112), np.zeros(112))


def test_fixtures_are_deterministic_and_parameter_storage_is_owned():
    first, second = make_fixture(seed=7, nonzero=True), make_fixture(seed=7, nonzero=True)
    for name in ("projection_weight", "hidden_weight", "output_weight"):
        assert_bits(getattr(first, name), getattr(second, name))
    assert np.count_nonzero(first.output_weight) > 0
    zero = make_fixture(seed=7)
    assert np.count_nonzero(zero.output_weight) == 0
    assert np.count_nonzero(zero.projection_weight) > 0
    supplied = np.ones((16, 2), np.float32)
    copied = replace(first, output_weight=supplied)
    supplied.fill(0)
    assert np.all(copied.output_weight == 1)
    with pytest.raises(ValueError):
        copied.output_weight[0, 0] = 99


def test_reference_uses_previous_mean_and_observed_first_row():
    projection = np.zeros((112, 16), np.float32)
    projection[:16] = np.eye(16, dtype=np.float32)
    reference = DynamicsReference(replace(make_fixture(), projection_weight=projection))
    first = np.ones(112, np.float32)
    reference.step(first)
    assert_bits(reference.last_features[16:], np.zeros(32, np.float32))
    assert_bits(reference.state[:32], np.ones(32, np.float32))
    assert reference.state[32] == 1
    reference.step(first * 3)
    assert_bits(reference.last_features[16:], np.full(32, 2, np.float32))
    assert_bits(reference.state[16:32], np.full(16, np.float32(1) + ALPHA*np.float32(2), np.float32))
    reference.reset()
    assert_bits(reference.state, np.zeros(AUX_WIDTH, np.float32))


def test_export_preserves_original_graph_and_exposes_only_four_inputs(graphs, tmp_path):
    _, identity, paths = graphs
    original = onnx.load(BASELINE)
    augmented = onnx.load(paths["zero"])
    count = len(original.graph.node)
    assert [n.SerializeToString() for n in augmented.graph.node[:count]] == [n.SerializeToString() for n in original.graph.node]
    count = len(original.graph.initializer)
    assert [n.SerializeToString() for n in augmented.graph.initializer[:count]] == [n.SerializeToString() for n in original.graph.initializer]
    assert [v.name for v in augmented.graph.input] == ["features", "hidden_0", "hidden_1", "branch_state"]
    with pytest.raises(ValueError, match="new destination"):
        export_dynamics(BASELINE, paths["zero"], identity)
    with pytest.raises(ValueError, match="new destination"):
        export_dynamics(BASELINE, BASELINE, identity)


@pytest.mark.parametrize("mode", ["iobinding", "run"])
def test_zero_head_is_bitwise_baseline_with_warmup_resets_and_owned_outputs(graphs, mode):
    _, _, paths = graphs
    backend = DynamicsBackend(paths["zero"], mode)
    original_backend = OnnxBackend(BASELINE, "iobinding")
    model, original = StreamingPredictor(backend), StreamingPredictor(original_backend)
    rows = synthetic_rows()
    retained = []
    for sequence in (3, 9, 3):
        for step, row in enumerate(rows):
            point = DataPoint(sequence, step, step >= 99, row)
            actual, expected = model.predict(point), original.predict(point)
            if step < 99:
                assert actual is expected is None
            else:
                assert_bits(actual, expected)
                retained.append((actual, actual.copy()))
            for a, b in zip(backend.hidden_states(), original_backend.hidden_states()):
                assert_bits(a, b)
    assert all(a.tobytes() == b.tobytes() for a, b in retained)
    assert not np.shares_memory(retained[0][0], retained[-1][0])
    state = backend.auxiliary_state()
    state.fill(0)
    assert backend.auxiliary_state()[-1] == 1


@pytest.mark.parametrize("graph_name,folded", [("nonzero", True), ("unfolded", False)])
def test_nonzero_graph_matches_numpy_reference_and_original_recurrence(graphs, graph_name, folded):
    parameters, _, paths = graphs
    branch, original = DynamicsBackend(paths[graph_name]), OnnxBackend(BASELINE, "iobinding")
    reference = DynamicsReference(parameters, folded=folded)
    observed_residual = []
    for row in synthetic_rows(seed=104, count=301):
        expected_base = original.step(row)
        expected_residual = reference.step(row)
        actual = branch.step(row)
        np.testing.assert_allclose(actual, expected_base + expected_residual, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(branch.auxiliary_state(), reference.state, atol=1e-6, rtol=1e-5)
        for a, b in zip(branch.hidden_states(), original.hidden_states()):
            assert_bits(a, b)
        observed_residual.append(actual - expected_base)
    assert np.max(np.abs(observed_residual)) > 0.01


def test_projection_folding_diagnostic_distinguishes_algebra_from_fp32_rounding(graphs):
    parameters, _, _ = graphs
    weight, bias = folded_projection(parameters)
    assert weight.dtype == bias.dtype == np.float32
    # This fixture exposed a near-zero state cancellation exceeding the strict
    # FP32 tolerance on the development host. Preserve the failure as a result,
    # not a relaxed assertion or a claim that folding always passes parity.
    report = normalization_folding_diagnostic(parameters, synthetic_rows(seed=72, count=139))
    assert report["rows"] == 139 and report["fp64_algebra_passed"]
    assert report["atol"] == 1e-6 and report["rtol"] == 1e-5
    assert report["max_fp64_algebra_error"] < 1e-12
    failures = report["state_rows_outside_tolerance"] + report["residual_rows_outside_tolerance"]
    assert report["passed"] == (failures == 0)
    exact = normalization_folding_diagnostic(make_fixture(nonzero=True), synthetic_rows(count=3))
    assert exact["passed"] and exact["max_state_absolute_error"] == 0


def run_backend(path, rows):
    backend = DynamicsBackend(path)
    outputs, states = [], []
    for row in rows:
        outputs.append(backend.step(row))
        states.append(backend.auxiliary_state())
    return np.asarray(outputs), np.asarray(states)


def test_future_suffix_cannot_change_earlier_predictions_or_filter_states(graphs):
    _, _, paths = graphs
    rows = synthetic_rows(count=151)
    changed = rows.copy()
    changed[99:] *= -4
    expected, expected_states = run_backend(paths["nonzero"], rows)
    actual, actual_states = run_backend(paths["nonzero"], changed)
    assert_bits(actual[:99], expected[:99])
    assert_bits(actual_states[:99], expected_states[:99])
    assert np.max(np.abs(actual[99:] - expected[99:])) > 0


def test_interleaved_instances_have_isolated_buffers_and_reset_all_banks(graphs):
    _, _, paths = graphs
    rows_a, rows_b = synthetic_rows(seed=19, count=87), synthetic_rows(seed=71, count=87)
    expected_a, states_a = run_backend(paths["nonzero"], rows_a)
    expected_b, states_b = run_backend(paths["nonzero"], rows_b)
    a, b = DynamicsBackend(paths["nonzero"]), DynamicsBackend(paths["nonzero"])
    for index, (row_a, row_b) in enumerate(zip(rows_a, rows_b)):
        assert_bits(a.step(row_a), expected_a[index])
        assert_bits(b.step(row_b), expected_b[index])
        assert_bits(a.auxiliary_state(), states_a[index])
        assert_bits(b.auxiliary_state(), states_b[index])
    a.reset()
    assert all(np.count_nonzero(state) == 0 for bank in a.banks for state in bank)
    for index, row in enumerate(rows_a):
        assert_bits(a.step(row), expected_a[index])


def test_backend_rejects_wrong_schema_and_propagates_runtime_failure(graphs, monkeypatch):
    _, _, paths = graphs
    with pytest.raises(ValueError, match="four-input"):
        DynamicsBackend(BASELINE)
    backend = DynamicsBackend(paths["nonzero"])
    before = backend.bank
    def broken(_):
        raise RuntimeError("synthetic runtime error")
    monkeypatch.setattr(backend.session, "run_with_iobinding", broken)
    with pytest.raises(RuntimeError, match="synthetic runtime"):
        backend.step(np.ones(112, np.float32))
    assert backend.bank == before


def test_streaming_validation_rejects_nonfinite_before_branch_advances(graphs):
    _, _, paths = graphs
    backend = DynamicsBackend(paths["nonzero"])
    model = StreamingPredictor(backend)
    with pytest.raises(ValueError, match="nonfinite"):
        model.predict(DataPoint(7, 0, False, np.full(112, np.nan, np.float32)))
    assert_bits(backend.auxiliary_state(), np.zeros(AUX_WIDTH, np.float32))
