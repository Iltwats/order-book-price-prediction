"""Synthetic-only objective-driven training and exact-resume integration."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="optional offline training dependency")

from connectome import correlation_gru as training
from connectome import partial_gru as base
from connectome.model_io import load_weights

SOURCE = Path(__file__).resolve().parents[1] / "artifacts/selected/model.onnx"


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"synthetic train reader is replaced; no competition rows")
    rng = np.random.default_rng(714)
    sequences = {}
    for group in range(3):
        x = rng.normal(0, .2, (123, 112)).astype(np.float32)
        y = rng.normal(0, .4, (123, 2)).astype(np.float32)
        if group == 0:
            y = np.abs(y)  # Training's declared all-side policy has no sign filter.
        sequences[group] = (600+group, x, y)
    monkeypatch.setattr(training, "_validated_groups", lambda path, groups: list(groups))
    monkeypatch.setattr(base, "_iter_sequences", lambda path, groups: (sequences[g] for g in groups))
    original_loader = training.load_torch_model

    def loader(*args, **kwargs):
        model = original_loader(*args, **kwargs)
        def forbidden(*_):
            raise AssertionError("frozen lower Torch GRU must never run")
        model.gru0.register_forward_pre_hook(forbidden)
        return model

    monkeypatch.setattr(training, "load_torch_model", loader)
    def forbidden_mse(*args, **kwargs):
        raise AssertionError("correlation run must not call the old label-MSE objective")
    monkeypatch.setattr(base, "weighted_chunk_losses", forbidden_mse)
    return SimpleNamespace(path=path, sequences=sequences)


def run(synthetic, destination, **kwargs):
    return training.train_correlation_gru(SOURCE, synthetic.path, destination, group_indices=[0, 1, 2],
        epochs=1, learning_rate=1e-4, chunk_size=64, batch_size=2, **kwargs)


def test_real_ort_objective_training_preserves_lower_and_resumes_exactly(synthetic, tmp_path):
    full = run(synthetic, tmp_path / "full")
    assert full["complete"] and full["method"] == "correlation_gru"
    assert full["specification"]["method"] == "correlation_gru"
    assert full["specification"]["parameters"] == training.correlation_parameters(learning_rate=1e-4, chunk_size=64, batch_size=2)
    assert full["cursor"] == {"epoch": 1, "position": 0, "updates": 2, "sequences_seen": 3}
    original, fitted = load_weights(SOURCE), load_weights(full["model_path"])
    assert all(original[key].tobytes() == fitted[key].tobytes() for key in ("w0", "r0", "b0"))
    assert any(original[key].tobytes() != fitted[key].tobytes() for key in ("w1", "r1", "b1", "head_w", "head_b"))
    assert full["history"][0]["student_model_sha256"] == training.file_hash(SOURCE)
    assert full["history"][1]["student_model_sha256"] != training.file_hash(SOURCE)
    for row in full["history"]:
        assert "objective_components" in row and "objective_loss" in row
        assert "linear_backward_scalar_not_loss" in row and "supervised_loss" not in row
        assert row["teacher_model_sha256"] == training.file_hash(SOURCE)
        assert all(sequence[name]["passed"] for sequence in row["local_diagnostics"] for name in ("upper", "prediction"))
    paused = run(synthetic, tmp_path / "paused", max_updates=1)
    assert not paused["complete"]
    resumed = training.resume_correlation_gru(paused["checkpoint_path"])
    assert resumed["complete"] and resumed["history"] == full["history"] and resumed["cursor"] == full["cursor"]
    assert training.file_hash(resumed["model_path"]) == full["model_sha256"]


def test_full_objective_called_once_per_batch_and_coefficients_never_renormalized(synthetic, tmp_path, monkeypatch):
    calls, expected_scalars, anchors = [], [], []
    actual_anchor = base.relative_anchor

    def count_anchor(*args):
        anchors.append(1)
        return actual_anchor(*args)

    def known_objective(prediction, teacher, targets, *, teacher_weight, warmup):
        calls.append((prediction.shape, teacher_weight, warmup))
        coefficient = torch.full_like(prediction, .003)
        coefficient[:warmup] = 0
        # Sum in exactly the declared chunks to isolate unwanted extra means.
        expected_scalars.append(sum(float((prediction[start:start+64]*coefficient[start:start+64]).sum())
                                    for start in range(0, len(prediction), 64)))
        return 7.25, coefficient, {"test_objective": True}

    monkeypatch.setattr(base, "relative_anchor", count_anchor)
    monkeypatch.setattr(training.objective, "correlation_objective", known_objective)
    report = run(synthetic, tmp_path / "known")
    assert calls == [(torch.Size([123, 2, 2]), .1, 99), (torch.Size([123, 1, 2]), .1, 99)]
    assert len(anchors) == 2  # One anchor per complete sequence batch, not per chunk.
    assert [r["objective_loss"] for r in report["history"]] == [7.25, 7.25]
    assert [r["linear_backward_scalar_not_loss"] for r in report["history"]] == expected_scalars


def test_resume_rejects_changed_objective_and_frozen_lower(synthetic, tmp_path):
    paused = run(synthetic, tmp_path / "paused", max_updates=1)
    state = training._read_checkpoint(paused["checkpoint_path"])
    state["specification"]["parameters"]["objective"]["tampered"] = True
    altered = tmp_path / "altered.pt"
    torch.save(state, altered)
    with pytest.raises(ValueError, match="specification changed"):
        training.resume_correlation_gru(altered, tmp_path / "wrong_objective")
    state = training._read_checkpoint(paused["checkpoint_path"])
    state["model_state"]["gru0.weight_ih_l0"][0, 0] += 1
    torch.save(state, altered)
    with pytest.raises(ValueError, match="frozen lower"):
        training.resume_correlation_gru(altered, tmp_path / "wrong_lower")


def test_nonfinite_upstream_gradient_fails_before_first_update(synthetic, tmp_path, monkeypatch):
    def broken(prediction, *args, **kwargs):
        return .5, torch.full_like(prediction, float("nan")), {}
    monkeypatch.setattr(training.objective, "correlation_objective", broken)
    with pytest.raises(FloatingPointError, match="upstream coefficients"):
        run(synthetic, tmp_path / "bad")
    checkpoint = training._read_checkpoint(tmp_path / "bad/checkpoint.pt")
    assert checkpoint["cursor"]["updates"] == 0
    assert not (tmp_path / "bad/model.onnx").exists()
