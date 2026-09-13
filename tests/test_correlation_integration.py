"""Cache and ONNX-export checks using synthetic rows, never competition data."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import onnxruntime as ort
import pytest

pytest.importorskip("torch", reason="optional offline training dependency")

from connectome import correlation
from connectome.model_io import load_weights
from utils import WARMUP


BASELINE = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"
RECURRENT = ("w0", "r0", "b0", "w1", "r1", "b1")


@pytest.fixture
def synthetic_source(tmp_path, monkeypatch):
    rng = np.random.default_rng(735)
    weights = load_weights(BASELINE)
    train = tmp_path / "synthetic_training.parquet"
    train.write_bytes(b"synthetic training fingerprint; parquet reader is mocked")
    groups = [7, 2]
    samples = {}
    calls = []
    for group in groups:
        x = rng.normal(0, .2, (123, 112)).astype(np.float32)
        y = np.column_stack((np.linspace(-1, 1, 123), np.linspace(2, -2, 123))).astype(np.float32)
        y[WARMUP:] = np.column_stack((np.linspace(-1, 1, 24), np.linspace(2, -2, 24)))
        if group == 7:
            y = np.abs(y)  # Excluded first, so the retained block must pack into slot zero.
        samples[group] = (1000 + group, x, y)

    def encoded(x):
        upper = np.pad(x, ((0, 0), (0, 16))).astype(np.float32)
        predicted = upper @ weights["head_w"] + weights["head_b"]
        return upper * np.float32(.5), upper, predicted

    class FakeEncoder:
        def __init__(self, model_path, destination):
            assert Path(model_path) == BASELINE
            self.provenance = {"kind": "synthetic-test-encoder", "path": str(destination)}

        def __call__(self, x):
            calls.append(x.copy())
            return encoded(x)

    monkeypatch.setattr(correlation, "OnnxSequenceEncoder", FakeEncoder)
    monkeypatch.setattr(correlation, "_validate_groups", lambda path, requested: list(requested))
    monkeypatch.setattr(correlation, "_iter_sequences", lambda path, requested: (samples[g] for g in requested))

    def extract(name="moments"):
        path = tmp_path / name
        manifest = correlation.extract_moments(BASELINE, train, groups, path, progress=False)
        return path, manifest

    return SimpleNamespace(train=train, groups=groups, samples=samples, calls=calls,
                           encoded=encoded, extract=extract, weights=weights)


def test_cache_packs_eligible_prefix_and_preserves_read_only_moments(synthetic_source):
    source = synthetic_source
    path, manifest = source.extract()
    assert manifest["included_groups"] == [2]
    assert manifest["excluded"] == [{"row_group": 7, "seq_ix": 1007}]
    assert manifest["seq_ids"] == [1002]
    assert manifest["eligible_sequences"] == 1
    assert [len(x) for x in source.calls] == [123, 123]  # Encoder consumes warm-up too.
    _, x, y = source.samples[2]
    _, upper, predicted = source.encoded(x)
    for representation, features in (("hidden", upper), ("outputs", predicted)):
        moments, beta, bias, _ = correlation.load_moments(
            path, representation=representation, model_path=BASELINE,
            groups=source.groups, train_path=source.train,
        )
        expected = correlation.sequence_moments(features[WARMUP:], y[WARMUP:])
        for key, value in moments.items():
            assert len(value) == 1
            np.testing.assert_allclose(value[0], expected[key], atol=1e-14, rtol=1e-12)
            assert not value.flags.writeable
        if representation == "hidden":
            np.testing.assert_array_equal(beta, source.weights["head_w"].T)
            np.testing.assert_array_equal(bias, source.weights["head_b"])
        else:
            np.testing.assert_array_equal(beta, np.eye(2))
            np.testing.assert_array_equal(bias, np.zeros(2))
    with pytest.raises(FileExistsError, match="already exists"):
        source.extract()


def test_cache_rejects_model_group_path_fingerprint_and_content_changes(synthetic_source, tmp_path):
    source = synthetic_source
    path, _ = source.extract()
    changed_model = tmp_path / "changed.onnx"
    changed_model.write_bytes(BASELINE.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="model hash"):
        correlation.load_moments(path, model_path=changed_model)
    with pytest.raises(ValueError, match="training groups"):
        correlation.load_moments(path, groups=list(reversed(source.groups)))
    with pytest.raises(ValueError, match="dataset path"):
        correlation.load_moments(path, train_path=tmp_path / "other.parquet")
    array = np.load(path / "covariance.npy", mmap_mode="r+")
    array[0, 0, 0, 0] += 1
    array.flush()
    del array
    with pytest.raises(ValueError, match="content hash"):
        correlation.load_moments(path)
    source.train.write_bytes(b"training data changed")
    with pytest.raises(ValueError, match="dataset fingerprint"):
        correlation.load_moments(path)


def test_extraction_rejects_nonfinite_targets_without_completing_cache(synthetic_source, tmp_path):
    source = synthetic_source
    source.samples[2][2][WARMUP, 0] = np.inf
    with pytest.raises(ValueError, match="nonfinite training targets"):
        source.extract()
    manifest = json.loads((tmp_path / "moments" / "manifest.json").read_text())
    assert manifest["complete"] is False
    with pytest.raises(ValueError, match="incomplete"):
        correlation.load_moments(tmp_path / "moments")


def test_extraction_rejects_training_mutation_during_pass(synthetic_source, tmp_path, monkeypatch):
    source = synthetic_source

    def changing_sequences(path, groups):
        for group in groups:
            yield source.samples[group]
        source.train.write_bytes(source.train.read_bytes() + b" modified during extraction")

    monkeypatch.setattr(correlation, "_iter_sequences", changing_sequences)
    with pytest.raises(ValueError, match="changed during moment extraction"):
        source.extract()
    manifest = json.loads((tmp_path / "moments" / "manifest.json").read_text())
    assert manifest["complete"] is False


def cpu_session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("representation", ["outputs", "hidden"])
@pytest.mark.parametrize("correction", [False, True], ids=["identity", "corrected"])
def test_export_folds_head_preserves_recurrence_and_matches_runtime(
    synthetic_source, tmp_path, monkeypatch, representation, correction,
):
    source = synthetic_source
    path, _ = source.extract()
    captured = {}

    def fixed_optimizer(moments, initial_beta, initial_bias, **kwargs):
        beta = initial_beta.copy()
        if correction:
            if representation == "outputs":
                beta = np.array([[.9, .1], [-.2, 1.1]], dtype=np.float64)
            else:
                beta += np.linspace(-.015, .015, beta.size).reshape(beta.shape)
        center = np.asarray(moments["feature_mean"]).mean(0)
        bias = initial_bias + ((initial_beta - beta) * center).sum(-1)
        captured.update(beta=beta.copy(), bias=bias.copy())
        return beta, bias, {"test_optimizer": True}

    monkeypatch.setattr(correlation, "optimize_moments", fixed_optimizer)
    report = correlation.fit_correlation_head(
        BASELINE, path, tmp_path / "candidate", representation=representation,
    )
    fitted = load_weights(report["model_path"])
    assert report["recurrence_bitwise_equal"] is True
    assert report["complete"] is True
    for key in RECURRENT:
        assert fitted[key].dtype == source.weights[key].dtype
        assert fitted[key].tobytes() == source.weights[key].tobytes()
    if representation == "outputs":
        expected_w = (captured["beta"] @ source.weights["head_w"].T.astype(np.float64)).T.astype(np.float32)
        expected_b = (captured["beta"] @ source.weights["head_b"].astype(np.float64) + captured["bias"]).astype(np.float32)
    else:
        expected_w = captured["beta"].T.astype(np.float32)
        expected_b = captured["bias"].astype(np.float32)
    np.testing.assert_array_equal(fitted["head_w"], expected_w)
    np.testing.assert_array_equal(fitted["head_b"], expected_b)
    if not correction:
        for key in source.weights:
            np.testing.assert_array_equal(fitted[key], source.weights[key])

    rng = np.random.default_rng(207)
    inputs = {
        "features": rng.normal(0, .2, (1, 1, 112)).astype(np.float32),
        "hidden_0": rng.normal(0, .2, (1, 1, 128)).astype(np.float32),
        "hidden_1": rng.normal(0, .2, (1, 1, 128)).astype(np.float32),
    }
    original = cpu_session(BASELINE).run(None, inputs)
    actual = cpu_session(report["model_path"]).run(None, inputs)
    for old_state, new_state in zip(original[1:], actual[1:]):
        np.testing.assert_array_equal(old_state, new_state)
    if representation == "outputs":
        expected_prediction = original[0].astype(np.float64) @ captured["beta"].T + captured["bias"]
    else:
        expected_prediction = original[2].astype(np.float64) @ captured["beta"].T + captured["bias"]
    np.testing.assert_allclose(actual[0], expected_prediction, atol=1e-6, rtol=1e-5)
    if not correction:
        np.testing.assert_array_equal(actual[0], original[0])
