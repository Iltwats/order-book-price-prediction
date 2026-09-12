"""Judge-facing causal contract and original-weight inference regression tests."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest

from connectome.calibration import apply_calibration
from connectome.runtime import OnnxBackend, StreamingPredictor, load_predictor
from utils import DataPoint

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "baseline.onnx"
MODES = ("reference", "run", "iobinding", "ortvaluevector")


class SumBackend:
    def __init__(self):
        self.state = np.zeros(2, np.float32)
        self.calls = self.resets = 0

    def reset(self):
        self.state.fill(0)
        self.resets += 1

    def step(self, features):
        self.calls += 1
        self.state += features[:2]
        return self.state.copy()


class IdentityBackend(SumBackend):
    def step(self, features):
        self.calls += 1
        return features[:2].copy()


def point(seq, step, features=None, *, need=None):
    return DataPoint(seq, step, step >= 99 if need is None else need,
                     np.ones(112, np.float32) if features is None else features)


def original_wrapper():
    spec = importlib.util.spec_from_file_location("unaltered_connectome_baseline", ROOT / "baseline" / "solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.PredictionModel()


def test_warmup_updates_state_and_resets_on_revisited_sequence():
    backend = SumBackend()
    model = StreamingPredictor(backend)
    for seq in [7, 8, 7]:
        for step in range(99):
            assert model.predict(point(seq, step)) is None
        result = model.predict(point(seq, 99, np.zeros(112, np.float32)))
        np.testing.assert_array_equal(result, [99, 99])
    assert backend.calls == 300
    assert backend.resets == 3


@pytest.mark.parametrize("bad_point,error", [
    (point(1, -1), "step"), (point(1, 20000), "step"),
    (point(1, 0, need=True), "warm-up"), (point(1, 99, need=False), "warm-up"),
    (point(1, 1), "step zero"),
    (point(1, 0, np.zeros(111, np.float32)), "112"),
    (point(1, 0, np.full(112, np.nan, np.float32)), "nonfinite"),
    (point(1, 0, np.full(112, np.inf, np.float32)), "nonfinite"),
])
def test_invalid_inputs_rejected_before_backend_advances(bad_point, error):
    backend = SumBackend()
    model = StreamingPredictor(backend, strict=True)
    with pytest.raises(ValueError, match=error):
        model.predict(bad_point)
    assert backend.calls == 0


@pytest.mark.parametrize("next_step", [0, 2])
def test_repeated_or_skipped_rows_rejected(next_step):
    backend = SumBackend()
    model = StreamingPredictor(backend)
    model.predict(point(3, 0))
    with pytest.raises(ValueError, match="sequence order"):
        model.predict(point(3, next_step))
    assert backend.calls == 1


@pytest.mark.parametrize("bad_output", [np.array([np.nan, 0], np.float32),
                                       np.array([np.inf, 0], np.float32),
                                       np.zeros((1, 2), np.float32)])
def test_nonfinite_or_wrong_shape_backend_output_fails_even_during_warmup(bad_output):
    backend = SumBackend()
    backend.step = lambda features: bad_output
    with pytest.raises(ValueError, match="invalid predictions"):
        StreamingPredictor(backend).predict(point(3, 0))


@pytest.mark.parametrize("mode", MODES)
def test_all_ort_modes_match_unmodified_wrapper_and_outputs_remain_owned(mode):
    original = original_wrapper()
    backend = OnnxBackend(BASELINE, mode)
    model = StreamingPredictor(backend)
    features = np.random.default_rng(313).normal(0, 0.5, size=(173, 112)).astype(np.float32)
    retained = []
    for seq in (7, 8, 7):
        for step, row in enumerate(features):
            datapoint = point(seq, step, row)
            expected, observed = original.predict(datapoint), model.predict(datapoint)
            if step < 99:
                assert expected is None and observed is None
            else:
                np.testing.assert_array_equal(observed, expected)
                retained.append((observed, observed.copy()))
            for actual, reference in zip(backend.hidden_states(), (original.hidden_0, original.hidden_1)):
                np.testing.assert_array_equal(actual, reference)
    for actual, snapshot in retained:
        np.testing.assert_array_equal(actual, snapshot)
    assert not np.shares_memory(retained[0][0], retained[-1][0])


@pytest.mark.parametrize("mode", MODES)
def test_future_suffix_does_not_change_earlier_recurrent_outputs(mode):
    features = np.random.default_rng(814).normal(0, 0.3, size=(190, 112)).astype(np.float32)
    changed = features.copy()
    changed[150:] *= -7
    def run(rows):
        model = StreamingPredictor(OnnxBackend(BASELINE, mode))
        return np.array([value for step, row in enumerate(rows)
                         if (value := model.predict(point(33, step, row))) is not None])
    original, modified = run(features), run(changed)
    np.testing.assert_array_equal(original[:150-99], modified[:150-99])
    assert np.max(np.abs(original[150-99:] - modified[150-99:])) > 0


@pytest.mark.parametrize("config", [
    {}, {"scales": [0.75, 1.25]},
    {"ema_half_life": 32, "ema_blend": 0.25},
    {"ema_half_life": [2, 8], "ema_blend": [0.1, 0.25]},
    {"scales": [0.75, 1.25], "affine": {"matrix": [[0.89, 0.37], [-0.16, 0.97]], "bias": [0.13, -0.24]},
     "ema_half_life": [2, 8], "ema_blend": [0.1, 0.25]},
])
def test_offline_calibration_matches_runtime_float32_tolerance_and_resets(config):
    features = np.random.default_rng(8).normal(size=(400, 112)).astype(np.float32)
    expected = apply_calibration(features[:, :2], **config)[99:]
    model = StreamingPredictor(IdentityBackend(), config)
    for seq in (1, 2):
        observed = np.array([value for step, row in enumerate(features)
                             if (value := model.predict(point(seq, step, row))) is not None])
        # Matrix fusion/batched BLAS can change FP32 rounding, but never causal state.
        np.testing.assert_allclose(observed, expected, atol=1e-6, rtol=1e-6)


def bad_shape_model(path, *, feature_shape=(1, 1, 111), feature_dtype=TensorProto.FLOAT,
                    output_shape=(1, 1, 2)):
    inputs = [helper.make_tensor_value_info("features", feature_dtype, list(feature_shape)),
              helper.make_tensor_value_info("hidden_0", TensorProto.FLOAT, [1, 1, 128]),
              helper.make_tensor_value_info("hidden_1", TensorProto.FLOAT, [1, 1, 128])]
    output = helper.make_tensor("prediction_constant", TensorProto.FLOAT, list(output_shape),
                                np.zeros(output_shape, np.float32).ravel().tolist())
    nodes = [helper.make_node("Constant", [], ["predictions"], value=output),
             helper.make_node("Identity", ["hidden_0"], ["next_hidden_0"]),
             helper.make_node("Identity", ["hidden_1"], ["next_hidden_1"])]
    outputs = [helper.make_tensor_value_info("predictions", TensorProto.FLOAT, list(output_shape)),
               helper.make_tensor_value_info("next_hidden_0", TensorProto.FLOAT, [1, 1, 128]),
               helper.make_tensor_value_info("next_hidden_1", TensorProto.FLOAT, [1, 1, 128])]
    model = helper.make_model(helper.make_graph(nodes, "shape_contract", inputs, outputs),
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=9)
    onnx.checker.check_model(model)
    onnx.save(model, path)


@pytest.mark.parametrize("options", [
    {"feature_shape": (1, 1, 111)},
    {"feature_shape": (1, 1, 112), "feature_dtype": TensorProto.DOUBLE},
    {"feature_shape": (1, 1, 112), "output_shape": (1, 1, 3)},
])
def test_wrong_model_schema_fails_at_startup(tmp_path, options):
    path = tmp_path / "wrong.onnx"
    bad_shape_model(path, **options)
    with pytest.raises(ValueError):
        OnnxBackend(path)


def test_native_missing_symbol_falls_back_at_startup_only(tmp_path, monkeypatch):
    import json
    import platform
    from connectome import native
    key = f"{platform.system().lower()}-{platform.machine().lower()}"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": "native", "model": str(BASELINE),
                                  "native_weights": "weights.npy", "native_libraries": {key: "missing-symbols.so"}}))
    def unavailable(*args, **kwargs):
        raise AttributeError("native library has no cc_abi_version symbol")
    monkeypatch.setattr(native, "NativeBackend", unavailable)
    with pytest.warns(RuntimeWarning, match="using validated ONNX"):
        model = load_predictor(config)
    assert isinstance(model.backend, OnnxBackend)


def test_native_error_after_stream_begins_is_not_replaced_by_zero_state_fallback(tmp_path, monkeypatch):
    import json
    import platform
    from connectome import native
    key = f"{platform.system().lower()}-{platform.machine().lower()}"
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"backend": "native", "model": str(BASELINE),
                                  "native_weights": "weights.npy", "native_libraries": {key: "available.so"}}))
    backend = SumBackend()
    monkeypatch.setattr(native, "NativeBackend", lambda *args, **kwargs: backend)
    model = load_predictor(config)
    for step in range(100):
        model.predict(point(1, step))
    def fail(features):
        raise RuntimeError("native failed after state advanced")
    backend.step = fail
    with pytest.raises(RuntimeError, match="after state advanced"):
        model.predict(point(1, 100))
    assert model.backend is backend


def mocked_labeled_sequence(group, *, eligible=False):
    targets = np.ones((20000, 2), np.float32)
    if eligible:
        targets[::2] = -1
    need = np.arange(20000) >= 99
    return SimpleNamespace(seq_ix=1000 + group, row_group=group,
                           features=np.zeros((20000, 112), np.float32), targets=targets,
                           need=need, mask=need.copy())


def test_evaluation_progress_can_precede_first_eligible_block(tmp_path, monkeypatch, capsys):
    from connectome import data, evaluation
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"mocked dataset")
    excluded = mocked_labeled_sequence(0)
    included = mocked_labeled_sequence(32, eligible=True)
    def read_sequence(parquet, group):
        source = included if group == 32 else excluded
        return SimpleNamespace(**{**source.__dict__, "seq_ix": 1000 + group, "row_group": group})
    monkeypatch.setattr(data, "read_sequence", read_sequence)
    monkeypatch.setattr(evaluation.pq, "ParquetFile", lambda path: SimpleNamespace(num_row_groups=33))
    class BulkSession:
        def get_inputs(self):
            return [SimpleNamespace(shape=[1, None, 112]), *[SimpleNamespace(shape=[1, 1, 128]) for _ in range(2)]]
        def run(self, *args, **kwargs):
            return [np.zeros((1, 20000, 2), np.float32), *[np.zeros((1, 1, 128), np.float32) for _ in range(2)]]
    monkeypatch.setattr(evaluation.ort, "InferenceSession", lambda *args, **kwargs: BulkSession())
    result = evaluation.evaluate(dataset, BASELINE, groups=range(33), bulk_model=BASELINE, progress=True)
    assert result["blocks"] == 33 and result["blocks_with_selected_rows"] == 33
    assert result['metric_id'] == 'global_wp_clip2_v1'
    assert result["weighted_pearson"] == 0


def test_inference_parity_does_not_require_metric_eligible_block(tmp_path, monkeypatch):
    from connectome import data, evaluation
    sequence = mocked_labeled_sequence(0)
    monkeypatch.setattr(data, "read_sequence", lambda parquet, group: sequence)
    monkeypatch.setattr(evaluation.pq, "ParquetFile", lambda path: object())
    class ZeroBackend:
        def __init__(self, *args, **kwargs):
            pass
        def reset(self):
            pass
        def step(self, row):
            return np.zeros(2, np.float32)
        def hidden_states(self):
            return (np.zeros((1, 1, 1), np.float32),) * 2
    monkeypatch.setattr(evaluation, "OnnxBackend", ZeroBackend)
    result = evaluation.check_parity(tmp_path / "mock.parquet", BASELINE, groups=[0])
    assert result["passed"]
    assert result["per_sequence"][0]["reference"] is None
    assert result["per_sequence"][0]["candidate"] is None
