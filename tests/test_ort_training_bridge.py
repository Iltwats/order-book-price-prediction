import copy
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connectome import ort_training_bridge as bridge
from connectome.partial_gru import weighted_chunk_losses
from connectome.training import StreamingGRU


@pytest.fixture(autouse=True)
def bounded_torch():
    torch.set_num_threads(1)


def test_proxy_forward_matches_ort_and_derivatives_reach_all_upper_and_head_parameters():
    with torch.random.fork_rng():
        torch.manual_seed(196)
        model = StreamingGRU(3).double()
        lower = torch.randn(11, 1, 3, dtype=torch.float64)
        initial = torch.randn(1, 1, 3, dtype=torch.float64, requires_grad=True)
        ort_u = torch.randn(11, 1, 3, dtype=torch.float64, requires_grad=True)
        ort_p = torch.randn(11, 1, 2, dtype=torch.float64, requires_grad=True)
        proxy_u, proxy_p, local = bridge.exact_forward_proxy(model, lower, initial, ort_u, ort_p)
        assert torch.equal(proxy_u, ort_u)
        assert torch.equal(proxy_p, ort_p)
        proxy_p.square().sum().backward()
        for name, parameter in model.named_parameters():
            if name.startswith("gru0"):
                assert parameter.grad is None
            else:
                assert parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        assert initial.grad is None and ort_u.grad is None and ort_p.grad is None


def test_fixed_coefficients_match_declared_weighted_label_and_teacher_mse_derivative():
    rng = np.random.default_rng(836)
    target, student, teacher = [rng.normal(size=(400, 2)).astype(np.float32) for _ in range(3)]
    target[:, 1] = 0
    coefficient = bridge.loss_coefficients(target, student, teacher, start=0, rows=256)
    target_t, teacher_t = torch.from_numpy(target).double(), torch.from_numpy(teacher).double()
    prediction = torch.from_numpy(student[:256]).double().requires_grad_()
    denominator = target_t[99:].clamp(-2., 2.).abs().sum(0)[None]
    label, matching = weighted_chunk_losses(prediction[:, None], teacher_t[:256, None], target_t[:256, None], denominator, start_step=0)
    gradient = torch.autograd.grad(label + .1 * matching, prediction)[0]
    np.testing.assert_array_equal(coefficient, gradient.numpy().astype(np.float32))
    assert not coefficient[:99].any() and not coefficient[:, 1].any()


def test_real_shape_local_gradient_procedure_on_synthetic_values():
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(893)
        model = StreamingGRU(3)
        lower = torch.randn(400, 3)
        upper = model.gru1(lower[:, None])[0][:, 0]
        prediction = model.head(upper)
        reference = tuple(x.numpy() for x in (lower, upper, prediction))
        target = np.random.default_rng(736).normal(size=(400, 2)).astype(np.float32)
    # Outside no_grad: both precisions must actually differentiate all tensors.
    for start in (0, 128):
        report = bridge.gradient_probe(model, reference, reference[2] + .2, target, start=start)
        assert report["passed"], report
        assert set(report["gradient_tensors"]) == set(bridge.GRADIENT_NAMES)
        assert all(report["identical_forward_values"].values())


def test_gradient_probe_has_no_forward_or_threshold_shortcut(monkeypatch):
    with torch.no_grad():
        model = StreamingGRU(3)
        lower = torch.zeros(300, 3)
        upper = model.gru1(lower[:, None])[0][:, 0]
        reference = (lower.numpy(), upper.numpy(), model.head(upper).numpy())
    original = bridge.exact_forward_proxy

    def wrong_gradient(model, *args):
        u, p, actual = original(model, *args)
        if next(model.parameters()).dtype == torch.float32:
            p = p.detach() + 2 * (p - p.detach())
        return u, p, actual

    monkeypatch.setattr(bridge, "exact_forward_proxy", wrong_gradient)
    report = bridge.gradient_probe(model, reference, reference[2], np.ones((300, 2), np.float32), start=0)
    assert all(report["identical_forward_values"].values())
    assert not report["passed"]


def test_summary_does_not_turn_exact_forward_values_into_gradient_approval():
    branches = {name: {"outputs": {"prediction": {"approximate_passed": True, "strict_passed": False}},
                       "wp": {"eligible": True, "drift": [0., 0.], "passed": True}} for name in bridge.BRANCHES}
    s = {"branches": branches, "equations": {"passed": True}, "exact_forward": True,
         "gradients": [{"passed": True}, {"passed": False}]}
    assert not bridge._summary([s])["approximate_passed"]
    s["gradients"][1]["passed"] = True
    assert bridge._summary([s])["approximate_passed"]
    assert not bridge._summary([s])["strict_passed"]


def test_protocol_preserves_old_failure_and_declares_new_path_without_data_reads(tmp_path, monkeypatch):
    from connectome import data
    failure = bridge.ROOT / "artifacts/round2/recurrent_bridge/report.json"
    before = failure.read_bytes()
    monkeypatch.setattr(data, "iter_sequences", lambda *a, **k: pytest.fail("prepare must not read TRAIN rows"))
    monkeypatch.setattr(data, "file_fingerprint", lambda *a, **k: pytest.fail("prepare must not open datasets"))
    path = bridge.prepare_protocol(tmp_path)
    protocol = bridge._verify_protocol(path)
    assert protocol["groups"] == [2309, 9078, 4507, 3165, 3508, 2523, 1866, 5932]
    assert protocol["budget"] == bridge.old.BUDGET
    assert protocol["procedure"]["gradient"] == {"atol": 1e-6, "rtol": 1e-4}
    assert protocol["procedure"]["gradient_chunk_starts"] == [0, 19744]
    assert failure.read_bytes() == before
    assert not json.loads(before)["approximate_training_accepted"]
    with pytest.raises(FileExistsError):
        bridge.prepare_protocol(tmp_path)
    protocol["procedure"]["gradient"]["atol"] *= 2
    Path(path).write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="frozen"):
        bridge._verify_protocol(path)


def test_candidate_requires_new_initial_acceptance_and_keeps_two_fixed_groups(tmp_path, monkeypatch):
    initial = Path(bridge.prepare_protocol(tmp_path / "initial"))
    initial_report = initial.parent / "report.json"
    initial_report.write_text(json.dumps({"protocol_path": str(initial)}))
    called = []
    monkeypatch.setattr(bridge, "require_bridge_acceptance", lambda *args: called.append(args))
    path = bridge.prepare_candidate_protocol(bridge.ROOT / "artifacts/selected/model.onnx", initial_report, tmp_path / "candidate")
    protocol = bridge._verify_protocol(path)
    assert called and protocol["groups"] == [2309, 9078] and protocol["stages"] == [2]
    assert protocol["mode"] == "candidate_export"


def test_candidate_acceptance_requires_full_gradient_coverage_and_exact_model(tmp_path, monkeypatch):
    initial = Path(bridge.prepare_protocol(tmp_path / "initial"))
    prior = initial.parent / "report.json"
    prior.write_text(json.dumps({"protocol_path": str(initial)}))
    monkeypatch.setattr(bridge, "require_bridge_acceptance", lambda *a: None)
    model = tmp_path / "model.onnx"
    model.write_bytes((bridge.ROOT / "artifacts/selected/model.onnx").read_bytes())
    path = Path(bridge.prepare_candidate_protocol(model, prior, tmp_path / "candidate"))
    protocol = bridge._verify_protocol(path)
    sequences = []
    for group in protocol["groups"]:
        outputs = {key: {"values": 20000 * width, "finite": True, "normalized_max": 1e-6,
                        "normalized_rms": 1e-7, "approximate_passed": True, "strict_passed": False}
                   for key, width in (("h0", 128), ("h1", 128), ("prediction", 2))}
        gradients = [{"start": start, "rows": 256, "passed": True,
                      "gradient_tensors": {key: {"values": size, "finite": True, "violations": 0,
                                                  "max_tolerance_ratio": .01} for key, size in bridge.GRADIENT_SIZES.items()}}
                     for start in bridge.PROCEDURE["gradient_chunk_starts"]]
        sequences.append({"row_group": group, "seq_ix": protocol["expected_sequence_ids"][str(group)],
            "rows": 20000, "equations": {"passed": True}, "exact_forward": True, "gradients": gradients,
            "branches": {name: {"outputs": outputs, "wp": {"eligible": True, "drift": [0., 0.], "passed": True}}
                         for name in bridge.BRANCHES}})
    report = {"protocol_path": str(path), "protocol_sha256": bridge.old._sha(path),
        "source_sha256": bridge.old._sha(model), "zero_update_sha256": bridge.old._sha(model),
        "mode": "candidate_export", "procedure": bridge.PROCEDURE, "budget": bridge.BUDGET,
        "approximate_candidate_export_accepted": True, "completed_sequences": 2, "sequences": sequences}
    report_path = path.parent / "report.json"
    bridge.old._write(report_path, report)
    assert bridge.require_candidate_acceptance(report_path, model)
    assert not bridge._summary(sequences)["strict_passed"]
    report["sequences"][0]["gradients"][0]["gradient_tensors"]["head.weight"]["values"] = 2
    bridge.old._write(report_path, report)
    with pytest.raises(ValueError, match="gradient evidence"):
        bridge.require_candidate_acceptance(report_path, model)
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        bridge.require_candidate_acceptance(report_path, model)
