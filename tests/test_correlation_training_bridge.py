import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from connectome import correlation_training_bridge as bridge


def test_new_protocol_records_correlation_objective_without_touching_data_or_old_evidence(tmp_path, monkeypatch):
    from connectome import data
    evidence = bridge.ROOT / "artifacts/round2/ort_training_bridge/report.json"
    before = evidence.read_bytes()
    monkeypatch.setattr(data, "iter_sequences", lambda *a, **k: pytest.fail("no dataset rows before registration"))
    monkeypatch.setattr(data, "file_fingerprint", lambda *a, **k: pytest.fail("prepare must not open dataset"))
    path = bridge.prepare_protocol(tmp_path)
    protocol = bridge._verify_protocol(path)
    assert protocol["groups"] == [2309, 9078, 4507, 3165, 3508, 2523, 1866, 5932]
    assert protocol["stages"] == [2, 8]
    assert protocol["procedure"]["objective"] == bridge.objective.objective_parameters()
    assert protocol["budget"] == bridge.forward.BUDGET
    assert evidence.read_bytes() == before
    with pytest.raises(FileExistsError):
        bridge.prepare_protocol(tmp_path)
    protocol["procedure"]["gradient"]["atol"] *= 2
    Path(path).write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="changed prospective"):
        bridge._verify_protocol(path)


def test_candidate_acceptance_cannot_reuse_mse_gradient_identity(tmp_path, monkeypatch):
    initial = Path(bridge.prepare_protocol(tmp_path / "initial"))
    initial_report = initial.parent / "report.json"
    initial_report.write_text(json.dumps({"protocol_path": str(initial)}))
    calls = []
    monkeypatch.setattr(bridge, "require_bridge_acceptance", lambda *a: calls.append(a))
    model = tmp_path / "model.onnx"
    model.write_bytes((bridge.ROOT / "artifacts/selected/model.onnx").read_bytes())
    path = Path(bridge.prepare_candidate_protocol(model, initial_report, tmp_path / "candidate"))
    protocol = bridge._verify_protocol(path)
    assert calls and protocol["groups"] == [2309, 9078]
    sequences = []
    for group in protocol["groups"]:
        outputs = {key: {"values": 20000 * width, "finite": True, "normalized_max": 1e-6,
                        "normalized_rms": 1e-7, "approximate_passed": True, "strict_passed": False}
                   for key, width in (("h0", 128), ("h1", 128), ("prediction", 2))}
        gradients = [{"start": start, "rows": 256, "passed": True,
            "objective_report": {"objective": bridge.objective.objective_parameters()},
            "gradient_tensors": {key: {"values": size, "finite": True, "violations": 0,
                "max_tolerance_ratio": .01} for key, size in bridge.forward.GRADIENT_SIZES.items()}}
            for start in (0, 19744)]
        sequences.append({"row_group": group, "seq_ix": protocol["expected_sequence_ids"][str(group)],
            "rows": 20000, "equations": {"passed": True}, "exact_forward": True, "gradients": gradients,
            "branches": {name: {"outputs": outputs, "wp": {"eligible": True, "drift": [0., 0.], "passed": True}}
                for name in bridge.forward.BRANCHES}})
    report = {"protocol_path": str(path), "protocol_sha256": bridge.old._sha(path),
        "source_sha256": bridge.old._sha(model), "zero_update_sha256": bridge.old._sha(model),
        "procedure": bridge.procedure(), "budget": bridge.forward.BUDGET,
        "mode": "candidate_export", "approximate_candidate_export_accepted": True,
        "completed_sequences": 2, "sequences": sequences}
    report_path = path.parent / "report.json"
    bridge.old._write(report_path, report)
    assert bridge.require_candidate_acceptance(report_path, model)
    assert not bridge.forward._summary(sequences)["strict_passed"]
    report["sequences"][0]["gradients"][0]["objective_report"]["objective"] = {"version": "mse"}
    bridge.old._write(report_path, report)
    with pytest.raises(ValueError, match="correlation-specific"):
        bridge.require_candidate_acceptance(report_path, model)
    model.write_bytes(model.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        bridge.require_candidate_acceptance(report_path, model)
