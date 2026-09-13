from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional offline training dependency")

from connectome import numerical_bridge as bridge
from connectome.data import Sequence
from connectome.training import StreamingGRU


@pytest.fixture(autouse=True)
def bounded_torch():
    torch.set_num_threads(1)


def test_approximate_acceptance_never_relabels_strict_failure():
    reference = np.zeros((16, 2), np.float32)
    actual = reference.copy()
    actual[0, 0] = 1e-5
    result = bridge.error_metrics(actual, reference, bridge.BUDGET["carried_state"])
    assert result["approximate_passed"]
    assert not result["strict_passed"]
    assert result["strict_violations"] == 1
    assert result["reference_scale_min"] == 1.


def test_rms_checks_each_coordinate_and_does_not_hide_persistent_drift():
    reference = np.zeros((20, 128), np.float32)
    actual = reference.copy()
    actual[:, 0] = 1e-5
    result = bridge.error_metrics(actual, reference, bridge.BUDGET["carried_state"])
    assert result["normalized_max"] < bridge.BUDGET["carried_state"]["normalized_max"]
    assert result["normalized_rms"] > bridge.BUDGET["carried_state"]["normalized_rms"]
    assert not result["approximate_passed"]


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_nonfinite_rejected_without_nan_report_values(bad):
    result = bridge.error_metrics(np.array([[bad]]), np.zeros((1, 1)), bridge.BUDGET["carried_state"])
    assert not result["approximate_passed"]
    json.dumps(result, allow_nan=False)


def test_common_state_rows_are_independent_and_upper_uses_ort_lower():
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(121)
        model = StreamingGRU(3).double()
        features = torch.randn(9, 112, dtype=torch.float64)
        h0, h1, _, _ = model.encode(features[:, None])
        reference = (h0[:, 0].numpy(), h1[:, 0].numpy(), model.head(h1)[:, 0].numpy())
        paths = bridge.torch_paths(model, features.numpy(), reference, batch_size=4)
        for branch in paths.values():
            for value, expected in zip(branch, reference):
                np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-10)
        changed = [value.copy() for value in reference]
        changed[0][3] += 0.3
        altered = bridge.torch_paths(model, features.numpy(), changed, batch_size=4)
        assert np.array_equal(altered["common_state"][0][:4], paths["common_state"][0][:4])
        assert not np.array_equal(altered["common_state"][0][4], paths["common_state"][0][4])
        assert np.array_equal(altered["upper_on_ort_lower"][0], changed[0])
        assert not np.array_equal(altered["upper_on_ort_lower"][1][3:], paths["upper_on_ort_lower"][1][3:])
        assert bridge.fp64_equation_spotcheck(model, features.numpy(), reference)["passed"]


def _sequence():
    target = np.resize(np.array([[-3., 0.5], [1., -3.], [0.4, 0.8]], np.float32), (20000, 2))
    need = np.arange(20000) >= 99
    return Sequence(1, 7, np.zeros((20000, 112), np.float32), target, need)


def test_training_domain_clipping_mask_and_wp_are_explicit():
    sequence = _sequence()
    prediction = sequence.targets * np.array([2., 0.7], np.float32)
    stats = bridge.domain_statistics(sequence, prediction)
    assert stats["rows"] == 19901
    assert stats["targets"]["t0"]["prediction_clip_fraction"] > 0
    assert stats["targets"]["t0"]["target_rms"] > stats["targets"]["t0"]["clipped_target_rms"]
    from utils import weighted_pearson
    assert stats["targets"]["t0"]["clipped_prediction_wp"] == weighted_pearson(
        sequence.targets[sequence.need, 0], prediction[sequence.need, 0])
    assert stats["targets"]["t0"]["raw_prediction_wp"] != stats["targets"]["t0"]["clipped_prediction_wp"]
    assert bridge.wp_comparison(sequence, prediction, prediction)["drift"] == [0., 0.]
    sequence = replace(sequence, mask=sequence.need)
    with pytest.raises(ValueError, match="TRAIN"):
        bridge.domain_statistics(sequence, prediction)
    with pytest.raises(ValueError, match="TRAIN"):
        bridge.wp_comparison(sequence, prediction, prediction)


def test_paired_wp_cancellation_does_not_hide_single_sequence_failure():
    metric = {"approximate_passed": True, "strict_passed": False}
    sequences = []
    for drift in (3e-5, -3e-5):
        entry = {"outputs": {"prediction": metric}, "wp": {"eligible": True, "drift": [drift, 0.], "passed": False}}
        sequences.append({"equations": {"passed": True}, "branches": {name: entry for name in bridge.BRANCHES}})
    summary = bridge.summarize(sequences)
    assert summary["branches"]["full_torch"]["mean_target_wp_drift"] == [0., 0.]
    assert not summary["approximate_passed"]
    assert not summary["strict_passed"]


def test_protocol_prepares_without_reading_datasets_and_is_immutable(tmp_path, monkeypatch):
    from connectome import data
    monkeypatch.setattr(data, "iter_sequences", lambda *a, **k: pytest.fail("prepare must not read rows"))
    monkeypatch.setattr(data, "file_fingerprint", lambda *a, **k: pytest.fail("prepare must not open dataset"))
    protocol_path = bridge.prepare_protocol(tmp_path)
    protocol = bridge._verify_protocol(protocol_path)
    assert protocol["groups"] == [2309, 9078, 4507, 3165, 3508, 2523, 1866, 5932]
    assert protocol["stages"] == [2, 8]
    assert protocol["mode"] == "initial_source"
    with pytest.raises(FileExistsError):
        bridge.prepare_protocol(tmp_path)
    protocol["budget"]["carried_state"]["normalized_max"] *= 2
    Path(protocol_path).write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="frozen"):
        bridge._verify_protocol(protocol_path)


def test_candidate_preparation_requires_source_acceptance_and_fixes_two_groups(tmp_path, monkeypatch):
    initial = Path(bridge.prepare_protocol(tmp_path / "initial"))
    report = initial.parent / "report.json"
    report.write_text(json.dumps({"protocol_path": str(initial)}))
    called = []
    monkeypatch.setattr(bridge, "require_bridge_acceptance", lambda *args: called.append(args))
    candidate = bridge.prepare_candidate_protocol(bridge.ROOT / "artifacts/selected/model.onnx", report, tmp_path / "candidate")
    protocol = bridge._verify_protocol(candidate)
    assert called
    assert protocol["groups"] == [2309, 9078]
    assert protocol["stages"] == [2]
    assert protocol["mode"] == "candidate_export"


def test_equation_failure_prevents_approximate_acceptance():
    sequences = [{"equations": {"passed": False}}]
    assert not bridge.summarize(sequences)["approximate_passed"]


def test_candidate_acceptance_checks_exact_model_coverage_and_keeps_strict_separate(tmp_path, monkeypatch):
    initial = Path(bridge.prepare_protocol(tmp_path / "initial"))
    initial_report = initial.parent / "report.json"
    initial_report.write_text(json.dumps({"protocol_path": str(initial)}))
    monkeypatch.setattr(bridge, "require_bridge_acceptance", lambda *args: None)
    candidate = tmp_path / "model.onnx"
    candidate.write_bytes((bridge.ROOT / "artifacts/selected/model.onnx").read_bytes())
    path = Path(bridge.prepare_candidate_protocol(candidate, initial_report, tmp_path / "candidate"))
    protocol = bridge._read(path)
    sequences = []
    for group in protocol["groups"]:
        outputs = {name: {"values": 20000 * width, "finite": True,
                         "normalized_max": 1e-6, "normalized_rms": 1e-7,
                         "approximate_passed": True, "strict_passed": False}
                   for name, width in (("h0", 128), ("h1", 128), ("prediction", 2))}
        sequences.append({"row_group": group, "seq_ix": protocol["expected_sequence_ids"][str(group)],
            "rows": 20000, "equations": {"passed": True},
            "branches": {b: {"outputs": outputs, "wp": {"eligible": True, "drift": [0., 0.], "passed": True}}
                         for b in bridge.BRANCHES}})
    report = {"protocol_path": str(path), "protocol_sha256": bridge._sha(path), "source_sha256": bridge._sha(candidate),
              "zero_update_sha256": bridge._sha(candidate), "mode": "candidate_export", "budget": bridge.BUDGET,
              "approximate_candidate_export_accepted": True, "completed_sequences": 2, "sequences": sequences}
    report_path = path.parent / "report.json"
    bridge._write(report_path, report)
    assert bridge.require_candidate_acceptance(report_path, candidate)
    assert not bridge.summarize(report["sequences"])["strict_passed"]
    report["sequences"][0]["rows"] = 19
    bridge._write(report_path, report)
    with pytest.raises(ValueError, match="complete"):
        bridge.require_candidate_acceptance(report_path, candidate)
    report["sequences"][0]["rows"] = 20000
    report["sequences"][0]["branches"]["full_torch"]["outputs"]["h1"]["normalized_max"] = 1.
    bridge._write(report_path, report)
    with pytest.raises(ValueError, match="budget"):
        bridge.require_candidate_acceptance(report_path, candidate)
    candidate.write_bytes(candidate.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        bridge.require_candidate_acceptance(report_path, candidate)
