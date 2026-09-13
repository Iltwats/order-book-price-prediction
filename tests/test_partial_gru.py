"""Synthetic-only tests for upper-layer training, teacher weights and resume."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
torch = pytest.importorskip("torch", reason="optional offline training dependency")

from connectome import partial_gru
from connectome.model_io import load_weights
from utils import FEATURE_COLUMNS, TARGET_COLUMNS, WARMUP

BASELINE = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"


def test_chunk_losses_equal_full_weighted_loss_and_ignore_warmup_teacher_gradients():
    torch.manual_seed(49)
    target = torch.randn(123, 2, 2)
    target[:, 1, 1] = 0  # No division-by-zero or exclusion of the other sides.
    prediction = torch.randn(123, 2, 2, requires_grad=True)
    teacher = torch.randn(123, 2, 2, requires_grad=True)
    denominators = target[WARMUP:].clamp(-2, 2).abs().sum(0)
    truth = target[WARMUP:].clamp(-2, 2)
    w = truth.abs() / denominators.clamp_min(1e-12)
    direct_label = (w * (prediction[WARMUP:] - truth).square()).sum(0).mean()
    direct_teacher = (w * (prediction[WARMUP:] - teacher[WARMUP:].detach()).square()).sum(0).mean()
    label, match = 0., 0.
    for start in range(0, len(target), 31):
        a, b = partial_gru.weighted_chunk_losses(
            prediction[start:start+31], teacher[start:start+31], target[start:start+31],
            denominators, start_step=start,
        )
        label, match = label+a, match+b
    torch.testing.assert_close(label, direct_label, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(match, direct_teacher, atol=1e-6, rtol=1e-6)
    actual_gradient, = torch.autograd.grad(label + .1*match, prediction, retain_graph=True)
    expected_gradient, = torch.autograd.grad(direct_label + .1*direct_teacher, prediction)
    torch.testing.assert_close(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-6)
    assert not actual_gradient[:WARMUP].count_nonzero()
    assert teacher.grad is None
    changed = target.clone(); changed[:WARMUP] = 1e10
    a, b = partial_gru.weighted_chunk_losses(prediction, teacher, changed, denominators, start_step=0)
    torch.testing.assert_close(a, label, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(b, match, atol=1e-6, rtol=1e-6)


def test_joint_relative_anchor_value_and_gradient():
    p = torch.tensor([1.25, 1.5], requires_grad=True)
    reference = {"upper": torch.tensor([1., 2.]), "head": torch.tensor([3.])}
    head = torch.tensor([3.5], requires_grad=True)
    anchor = partial_gru.relative_anchor({"upper": p, "head": head}, reference)
    torch.testing.assert_close(anchor, torch.tensor((.25**2 + .5**2 + .5**2)/14))
    anchor.backward()
    torch.testing.assert_close(p.grad, torch.tensor([.5, -1.])/14)
    torch.testing.assert_close(head.grad, torch.tensor([1.])/14)


def test_matched_forward_vjp_uses_exact_ort_loss_hidden_values_and_detached_teacher():
    torch.manual_seed(923)
    head = torch.nn.Linear(3, 2).double()
    hidden = torch.randn(13, 2, 3, dtype=torch.float64, requires_grad=True)
    ort_hidden = torch.randn(13, 2, 3, dtype=torch.float64, requires_grad=True)
    ort_prediction = torch.randn(13, 2, 2, dtype=torch.float64, requires_grad=True)
    teacher = torch.randn(13, 2, 2, dtype=torch.float64, requires_grad=True)
    target = torch.randn(13, 2, 2, dtype=torch.float64) * 3
    denominator = target.clamp(-2, 2).abs().sum(0)
    calls = []
    head.register_forward_hook(lambda *_: calls.append(1))
    prediction = partial_gru.matched_forward_prediction(head, hidden, ort_hidden, ort_prediction)
    assert torch.equal(prediction.detach(), ort_prediction.detach())
    assert len(calls) == 1
    label, match = partial_gru.weighted_chunk_losses(prediction, teacher, target, denominator, start_step=WARMUP)
    clipped, p, t = (v.detach().numpy() for v in (target.clamp(-2, 2), ort_prediction, teacher))
    w = np.abs(clipped) / denominator.numpy()
    expected_loss = np.sum(w*((p-clipped)**2 + .1*(p-t)**2))/4
    np.testing.assert_allclose(float((label+.1*match).detach()), expected_loss, rtol=1e-14)
    alpha = 2*w*((p-clipped)+.1*(p-t))/4
    expected_weight = np.einsum("tbo,tbi->oi", alpha, ort_hidden.detach().numpy())
    expected_hidden = alpha @ head.weight.detach().numpy()
    (label+.1*match).backward()
    np.testing.assert_allclose(head.weight.grad, expected_weight, atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(head.bias.grad, alpha.sum((0, 1)), atol=1e-14, rtol=1e-13)
    np.testing.assert_allclose(hidden.grad, expected_hidden, atol=1e-14, rtol=1e-13)
    assert ort_hidden.grad is ort_prediction.grad is teacher.grad is None


def test_chunks_start_from_detached_ort_history_and_never_carry_torch_final_state():
    class RecordingGRU(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gain = torch.nn.Parameter(torch.tensor(.2))
            self.initials = []

        def forward(self, x, initial):
            self.initials.append((initial.detach().clone(), initial.requires_grad))
            values = initial + self.gain*x.cumsum(0)
            return values, values[-1:]

    torch.manual_seed(33)
    model = SimpleNamespace(gru1=RecordingGRU(), head=torch.nn.Linear(3, 2))
    lower = torch.randn(8, 2, 3, requires_grad=True)
    ort_upper = torch.randn(8, 2, 3, requires_grad=True)
    ort_prediction = torch.randn(8, 2, 2, requires_grad=True)
    first, raw, _ = partial_gru.reanchored_upper_chunk(model, lower, ort_upper, ort_prediction, start=0, end=4)
    first.sum().backward()
    second, _, _ = partial_gru.reanchored_upper_chunk(model, lower, ort_upper, ort_prediction, start=4, end=8)
    second.sum().backward()  # Independent autograd segments need no retained graph.
    assert torch.equal(first.detach(), ort_prediction[:4].detach())
    assert torch.equal(second.detach(), ort_prediction[4:].detach())
    assert not model.gru1.initials[0][0].count_nonzero()
    assert torch.equal(model.gru1.initials[1][0], ort_upper[3:4].detach())
    assert not torch.equal(model.gru1.initials[1][0], raw[-1:].detach())
    assert all(not requires_grad for _, requires_grad in model.gru1.initials)
    assert lower.grad is ort_upper.grad is ort_prediction.grad is None
    assert model.gru1.gain.grad is not None


@pytest.fixture
def synthetic_training(tmp_path, monkeypatch):
    train = tmp_path / "train.parquet"
    train.write_bytes(b"synthetic training file; row reader mocked")
    rng = np.random.default_rng(514)
    source = load_weights(BASELINE)
    samples = {}
    calls, anchor_calls, encoder_sources = [], [], []
    for group in range(3):
        x = rng.normal(0, .2, (123, 112)).astype(np.float32)
        x[:, 0] += group * .1
        y = rng.normal(0, .4, (123, 2)).astype(np.float32)
        if group == 0:
            y = np.abs(y)  # Training must not silently exclude one-sign sequences.
        samples[group] = (300+group, x, y)

    class Encoder:
        def __init__(self, model_path, destination):
            self.provenance = {"kind": "synthetic_test", "source_sha256": partial_gru.file_hash(model_path),
                               "path": str(destination), "compute_threads": 1}
            self.h0 = np.zeros((123, 128), dtype=np.float32)
            self.h1 = np.zeros((123, 128), dtype=np.float32)
            self.teacher = np.zeros((123, 2), dtype=np.float32)
            self.model = original_loader(model_path)
            encoder_sources.append((str(destination), self.provenance["source_sha256"]))

        def __call__(self, x):
            calls.append(x.copy())
            self.h0.fill(0); self.h0[:, :112] = x
            with torch.no_grad():
                h1, _ = self.model.gru1(torch.from_numpy(self.h0)[:, None])
                self.h1[:] = h1[:, 0].numpy()
                self.teacher[:] = self.model.head(h1)[:, 0].numpy()
            # Deliberately reused arrays exercise per-sequence output ownership.
            return self.h0, self.h1, self.teacher

    original_loader = partial_gru.load_torch_model

    def guarded_loader(*args, **kwargs):
        model = original_loader(*args, **kwargs)

        def forbidden_lower(*_):
            raise AssertionError("lower Torch GRU must not run")

        model.gru0.register_forward_pre_hook(forbidden_lower)
        return model

    original_anchor = partial_gru.relative_anchor

    def counted_anchor(*args):
        anchor_calls.append(1)
        return original_anchor(*args)

    monkeypatch.setattr(partial_gru, "OnnxSequenceEncoder", Encoder)
    monkeypatch.setattr(partial_gru, "load_torch_model", guarded_loader)
    monkeypatch.setattr(partial_gru, "relative_anchor", counted_anchor)
    monkeypatch.setattr(partial_gru, "_validated_groups", lambda path, groups: list(groups))
    monkeypatch.setattr(partial_gru, "_iter_sequences", lambda path, groups: (samples[g] for g in groups))
    return SimpleNamespace(train=train, samples=samples, source=source, calls=calls,
                           anchor_calls=anchor_calls, encoder=Encoder, encoder_sources=encoder_sources)


def run_synthetic(source, output, **kwargs):
    return partial_gru.train_partial_gru(
        BASELINE, source.train, output, group_indices=[0, 1, 2], epochs=1,
        learning_rate=1e-3, batch_size=2, chunk_size=64, **kwargs,
    )


def test_encoded_batch_copies_original_teacher_buffers_and_consumes_all_rows(synthetic_training):
    source = synthetic_training
    encoder = source.encoder(BASELINE, "test")
    student = source.encoder(BASELINE, "student")
    lower, upper, prediction, teacher, target, ids = partial_gru._encode_batch(encoder, student, source.train, [0, 1])
    assert ids == [300, 301]
    assert [len(x) for x in source.calls] == [123, 123, 123, 123]
    np.testing.assert_array_equal(lower[:, 0, :112], source.samples[0][1])
    np.testing.assert_array_equal(lower[:, 1, :112], source.samples[1][1])
    _, expected_upper, expected = encoder(source.samples[0][1])
    np.testing.assert_array_equal(teacher[:, 0], expected)
    np.testing.assert_array_equal(prediction[:, 0], expected)
    np.testing.assert_array_equal(upper[:, 0], expected_upper)
    np.testing.assert_array_equal(target[:, 0], source.samples[0][2])


def test_partial_training_freezes_lower_exports_and_resumes_exactly(synthetic_training, tmp_path):
    source = synthetic_training
    full = run_synthetic(source, tmp_path / "full")
    assert full["complete"] and full["cursor"]["updates"] == 2
    assert len(source.anchor_calls) == 2  # Not once per chunk.
    assert full["cursor"]["sequences_seen"] == 3
    assert full["specification"]["parameters"] == partial_gru.partial_parameters(
        learning_rate=1e-3, batch_size=2, chunk_size=64,
    )
    fitted = load_weights(full["model_path"])
    for key in ("w0", "r0", "b0"):
        assert fitted[key].tobytes() == source.source[key].tobytes()
    assert any(fitted[k].tobytes() != source.source[k].tobytes() for k in ("w1", "r1", "b1", "head_w", "head_b"))
    assert Path(full["sequence_model_path"]).exists()
    baseline_sha = partial_gru.file_hash(BASELINE)
    assert full["history"][0]["student_model_sha256"] == baseline_sha
    assert full["history"][1]["student_model_sha256"] != baseline_sha
    assert all(entry["teacher_model_sha256"] == baseline_sha for entry in full["history"])
    assert all(sha == baseline_sha for path, sha in source.encoder_sources if path.endswith("source_encoder.onnx"))
    for entry in full["history"]:
        assert entry["student_encoder"]["source_sha256"] == entry["student_model_sha256"]
        assert all(seq[name]["passed"] for seq in entry["local_diagnostics"] for name in ("upper", "prediction"))
    paused = run_synthetic(source, tmp_path / "paused", max_updates=1)
    assert not paused["complete"] and paused["cursor"]["updates"] == 1
    resumed = partial_gru.resume_partial_gru(paused["checkpoint_path"])
    assert resumed["complete"] and resumed["cursor"] == full["cursor"]
    assert resumed["history"] == full["history"]
    for key, value in fitted.items():
        assert load_weights(resumed["model_path"])[key].tobytes() == value.tobytes()


def test_resume_rejects_changed_data_settings_and_frozen_lower(synthetic_training, tmp_path):
    source = synthetic_training
    paused = run_synthetic(source, tmp_path / "paused", max_updates=1)
    with pytest.raises(ValueError, match="specification changed"):
        partial_gru.train_partial_gru(
            BASELINE, source.train, tmp_path / "wrong", group_indices=[0, 1, 2],
            resume_from=paused["checkpoint_path"], learning_rate=3e-5, batch_size=2, chunk_size=64,
        )
    checkpoint = torch.load(paused["checkpoint_path"], weights_only=False)
    checkpoint["model_state"]["gru0.weight_ih_l0"][0, 0] += .5
    tampered = tmp_path / "altered.pt"
    torch.save(checkpoint, tampered)
    with pytest.raises(ValueError, match="frozen lower"):
        partial_gru.resume_partial_gru(tampered, tmp_path / "altered_run")
    source.train.write_bytes(b"changed training contents")
    with pytest.raises(ValueError, match="specification changed"):
        partial_gru.resume_partial_gru(paused["checkpoint_path"])


def test_nonfinite_encoder_fails_before_update_preserving_checkpoint(synthetic_training, tmp_path, monkeypatch):
    source = synthetic_training
    original = source.encoder.__call__

    def broken(self, x):
        lower, upper, teacher = original(self, x)
        lower[0, 0] = np.inf
        return lower, upper, teacher

    monkeypatch.setattr(source.encoder, "__call__", broken)
    with pytest.raises(ValueError, match="invalid lower states"):
        run_synthetic(source, tmp_path / "bad")
    checkpoint = partial_gru._read_checkpoint(tmp_path / "bad" / "checkpoint.pt")
    assert checkpoint["cursor"]["updates"] == 0
    assert not (tmp_path / "bad" / "model.onnx").exists()


def test_corrected_forward_cannot_hide_failed_raw_local_diagnostics(synthetic_training, tmp_path, monkeypatch):
    source = synthetic_training
    original = source.encoder.__call__

    def altered(self, x):
        lower, upper, prediction = original(self, x)
        upper[:] += .01
        return lower, upper, prediction

    monkeypatch.setattr(source.encoder, "__call__", altered)
    with pytest.raises(FloatingPointError, match="raw local Torch/ORT diagnostic"):
        run_synthetic(source, tmp_path / "local_failure")
    checkpoint = partial_gru._read_checkpoint(tmp_path / "local_failure" / "checkpoint.pt")
    assert checkpoint["cursor"]["updates"] == 0
    assert (tmp_path / "local_failure" / "failed_local_diagnostic.json").exists()
    assert not (tmp_path / "local_failure" / "model.onnx").exists()


def test_training_schema_groups_and_parameter_guards(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"is_scored": [True]}), path)
    with pytest.raises(ValueError, match="without is_scored"):
        partial_gru._validated_groups(path, [0])
    fields = ["seq_ix", "step_in_seq", "need_prediction", *FEATURE_COLUMNS, *TARGET_COLUMNS]
    pq.write_table(pa.table({name: [0., 0., 0., 0.] for name in fields}), path, row_group_size=2)
    for groups in ([0, 0], [2], [-1], [0.5], []):
        with pytest.raises(ValueError, match="distinct existing"):
            partial_gru._validated_groups(path, groups)
    assert partial_gru._validated_groups(path, [1, 0]) == [1, 0]
    for values in ({"chunk_size": 0}, {"teacher_weight": -1}, {"learning_rate": np.nan}, {"anchor_weight": -1}):
        with pytest.raises(ValueError):
            partial_gru.partial_parameters(**values)
