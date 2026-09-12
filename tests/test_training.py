from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

torch = pytest.importorskip("torch", reason="optional offline training dependency")

from connectome import training
from connectome.model_io import load_weights

BASELINE = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def test_reconstruction_matches_original_stream_and_nonzero_states():
    training.seed_everything()
    model = training.load_torch_model(BASELINE)
    original = session(BASELINE)
    rng = np.random.default_rng(143)
    features = rng.normal(0, 0.5, (167, 1, 112)).astype(np.float32)
    initial = [rng.normal(0, 0.2, (1, 1, 128)).astype(np.float32) for _ in range(2)]
    states = [state.copy() for state in initial]
    predictions = []
    for row in features:
        prediction, *states = original.run(None, {"features": row[None], "hidden_0": states[0], "hidden_1": states[1]})
        predictions.append(prediction)
    with torch.no_grad():
        actual, *final = model(torch.from_numpy(features), *[torch.from_numpy(state) for state in initial])
    np.testing.assert_allclose(actual.numpy(), np.concatenate(predictions), rtol=2e-5, atol=2e-6)
    for observed, expected in zip(final, states):
        np.testing.assert_allclose(observed.numpy(), expected, rtol=2e-5, atol=2e-6)


def test_fixed_export_preserves_weights_and_dynamic_export_matches(tmp_path):
    training.seed_everything()
    model = training.load_torch_model(BASELINE)
    fixed = training.export_onnx(model, tmp_path / "fixed.onnx")
    for key, original in load_weights(BASELINE).items():
        np.testing.assert_array_equal(load_weights(fixed)[key], original)
    dynamic = training.export_onnx(model, tmp_path / "dynamic.onnx", dynamic=True)
    features = torch.randn(137, 3, 112) * 0.2
    initial = model.initial_states(3)
    with torch.no_grad():
        expected = model(features, *initial)
    actual = session(dynamic).run(None, {"features": features.numpy(), "hidden_0": initial[0].numpy(), "hidden_1": initial[1].numpy()})
    for observed, reference in zip(actual, expected):
        np.testing.assert_allclose(observed, reference.numpy(), atol=2e-6, rtol=2e-5)


def test_both_state_expansion_preserves_prediction(tmp_path):
    training.seed_everything()
    original = training.load_torch_model(BASELINE)
    expanded = training.with_head_state(original, "both")
    x = torch.randn(103, 2, 112)
    with torch.no_grad():
        reference = original(x)
        actual = expanded(x)
    for expected, observed in zip(reference, actual):
        torch.testing.assert_close(expected, observed, atol=1e-6, rtol=1e-5)
    path = training.export_onnx(expanded, tmp_path / "both.onnx", dynamic=True)
    initial = expanded.initial_states(2)
    results = session(path).run(None, {"features": x.numpy(), "hidden_0": initial[0].numpy(), "hidden_1": initial[1].numpy()})
    np.testing.assert_allclose(results[0], actual[0].numpy(), atol=2e-6, rtol=2e-5)


def test_original_onnx_encoder_preserves_every_stream_state(tmp_path):
    encoder = training.OnnxSequenceEncoder(BASELINE, tmp_path / "encoder.onnx")
    rng = np.random.default_rng(182)
    x = rng.normal(0, 0.2, (127, 112)).astype(np.float32)
    lower, upper, predictions = encoder(x)
    original = session(BASELINE)
    states = [np.zeros((1, 1, 128), np.float32) for _ in range(2)]
    for step, row in enumerate(x):
        output, *states = original.run(None, {"features": row.reshape(1, 1, 112),
                                              "hidden_0": states[0], "hidden_1": states[1]})
        np.testing.assert_array_equal(predictions[step], output[0, 0])
        np.testing.assert_array_equal(lower[step], states[0][0, 0])
        np.testing.assert_array_equal(upper[step], states[1][0, 0])


def test_frozen_both_head_bulk_matches_stream_with_nonzero_lower_weights(tmp_path):
    training.seed_everything()
    model = training.with_head_state(training.load_torch_model(BASELINE), "both")
    # A nonzero lower-state readout detects an accidental use of final Y_h
    # (future state) in place of the lower layer's complete output sequence.
    with torch.no_grad():
        model.head.weight[:, :128].normal_(0, 0.1)
    fixed = session(training.export_frozen_head(BASELINE, model, tmp_path / "fixed.onnx"))
    bulk = session(training.export_frozen_head(BASELINE, model, tmp_path / "bulk.onnx", dynamic=True))
    rng = np.random.default_rng(131)
    x = rng.normal(0, 0.2, (1, 137, 112)).astype(np.float32)
    initial = [rng.normal(0, 0.1, (1, 1, 128)).astype(np.float32) for _ in range(2)]
    expected = bulk.run(None, {"features": x, "hidden_0": initial[0], "hidden_1": initial[1]})
    states = [state.copy() for state in initial]
    predictions = []
    for row in x[0]:
        output, *states = fixed.run(None, {"features": row.reshape(1, 1, 112),
                                           "hidden_0": states[0], "hidden_1": states[1]})
        predictions.append(output)
    np.testing.assert_array_equal(np.concatenate(predictions, axis=1), expected[0])
    for actual, reference in zip(states, expected[1:]):
        np.testing.assert_array_equal(actual, reference)


def test_chunk_loss_matches_full_sequence_value_and_gradient():
    training.seed_everything()
    target = torch.randn(167, 3, 2) * 4
    target[99:120, 0] = 0
    prediction = torch.randn(167, 3, 2, requires_grad=True)
    denominators = training.sequence_weight_denominators(target)
    full = training.weighted_sequence_chunk_loss(prediction, target, denominators, start_step=0)
    full_gradient, = torch.autograd.grad(full, prediction, retain_graph=True)
    chunks = sum(training.weighted_sequence_chunk_loss(prediction[start:start + 37], target[start:start + 37],
                                                       denominators, start_step=start)
                 for start in range(0, len(target), 37))
    chunk_gradient, = torch.autograd.grad(chunks, prediction)
    torch.testing.assert_close(full, chunks)
    torch.testing.assert_close(full_gradient, chunk_gradient)
    assert torch.count_nonzero(chunk_gradient[:99]) == 0
    # A zero-weight sequence/target contributes zero instead of NaN.
    zero = torch.zeros(167, 1, 2)
    loss = training.weighted_sequence_chunk_loss(torch.ones_like(zero), zero,
                                                 training.sequence_weight_denominators(zero), start_step=0)
    assert loss.item() == 0


@pytest.fixture
def tiny_sequences(monkeypatch, tmp_path):
    marker = tmp_path / "training_fixture.parquet"
    marker.write_text("mocked sequence reader; no real parquet I/O")
    rng = np.random.default_rng(554)
    sequences = {}
    for group in range(2):
        x = rng.normal(0, 0.2, (123, 112)).astype(np.float32)
        y = np.column_stack((x[:, 0] + 0.1, x[:, 1] - 0.1)).astype(np.float32)
        sequences[group] = (group + 41, x, y)
    monkeypatch.setattr(training, "_groups", lambda path, maximum, seed, groups: list(groups) if groups is not None else [0, 1])
    monkeypatch.setattr(training, "_iter_sequences", lambda path, groups: (sequences[int(group)] for group in groups))
    return marker, sequences


def test_student_training_resume_matches_uninterrupted(tiny_sequences, tmp_path):
    marker, _ = tiny_sequences
    options = dict(hidden_size=64, epochs=1, batch_size=1, chunk_size=37, threads=1, seed=815)
    full = training.train_student(BASELINE, marker, tmp_path / "full", **options)
    partial = training.train_student(BASELINE, marker, tmp_path / "partial", max_updates=1, **options)
    assert not partial["complete"]
    resumed = training.resume_training(partial["checkpoint_path"], tmp_path / "resumed", epochs=1, threads=1)
    assert resumed["complete"]
    state_full = training._read_checkpoint(full["checkpoint_path"])
    state_resumed = training._read_checkpoint(resumed["checkpoint_path"])
    for key, value in state_full["model_state"].items():
        torch.testing.assert_close(value, state_resumed["model_state"][key], atol=0, rtol=0)
    assert state_full["history"] == state_resumed["history"]
    assert resumed["model_config"]["hidden_size"] == 64


def test_fine_tune_head_only_and_resume_rejects_changed_data(tiny_sequences, tmp_path):
    marker, _ = tiny_sequences
    before = training.load_torch_model(BASELINE)
    report = training.fine_tune(BASELINE, marker, tmp_path / "fine", head_only=True,
                                epochs=1, batch_size=2, chunk_size=43, learning_rate=1e-3, threads=1)
    checkpoint = training._read_checkpoint(report["checkpoint_path"])
    for name, value in before.state_dict().items():
        if name.startswith("gru"):
            torch.testing.assert_close(value, checkpoint["model_state"][name], atol=0, rtol=0)
    assert not torch.equal(before.head.weight, checkpoint["model_state"]["head.weight"])
    marker.write_text("changed input")
    with pytest.raises(ValueError, match="data"):
        training.resume_training(report["checkpoint_path"], epochs=2)


def test_adamw_head_resume_and_fused_ridge_affine(tiny_sequences, tmp_path):
    marker, _ = tiny_sequences
    common = dict(head_state="both", minibatch_size=7, chunk_size=41, epochs=1, threads=1)
    full = training.fit_head(BASELINE, marker, tmp_path / "head_full", **common)
    partial = training.fit_head(BASELINE, marker, tmp_path / "head_partial", max_new_sequences=1, **common)
    resumed = training.resume_training(partial["checkpoint_path"], tmp_path / "head_resumed", epochs=1)
    assert full["optimizer"] == "AdamW" and resumed["complete"]
    original_state = training._read_checkpoint(full["checkpoint_path"])["model_state"]
    resumed_state = training._read_checkpoint(resumed["checkpoint_path"])["model_state"]
    for name, value in original_state.items():
        torch.testing.assert_close(value, resumed_state[name], atol=0, rtol=0)
    ridge = training.fit_head(BASELINE, marker, tmp_path / "ridge", optimizer="ridge", rows_per_sequence=None,
                              chunk_size=41, max_new_sequences=1)
    ridge_resumed = training.resume_training(ridge["checkpoint_path"], tmp_path / "ridge_resumed")
    assert ridge_resumed["complete"]
    affine = training.fit_affine(BASELINE, marker, tmp_path / "affine", rows_per_sequence=None, chunk_size=41)
    # Both ordinary-head fits preserve a native-compatible topology.
    assert load_weights(ridge_resumed["model_path"])["head_w"].shape == (128, 2)
    assert load_weights(affine["model_path"])["head_w"].shape == (128, 2)
