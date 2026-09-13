"""Round-two guards: fixed incumbent, prospective caps, and evidence identity."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import pytest

from connectome import experiments as ex
from connectome.data import file_fingerprint
from connectome.evaluation import file_hash


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def setup(tmp_path, monkeypatch):
    train, valid = tmp_path / "train.parquet", tmp_path / "valid.parquet"
    train.write_bytes(b"training fixture")
    valid.write_bytes(b"validation fixture")
    model = tmp_path / "incumbent.onnx"
    model.write_bytes(b"selected fixture")
    monkeypatch.setattr(ex, "INCUMBENT_SHA256", file_hash(model))
    config = write(tmp_path / "incumbent.json", {"model": model.name, "postprocess": {}})
    code = tmp_path / "source.py"
    code.write_text("x = 1\n")
    manifest = {"version": 1, "splits": {"train": [0, 1, 2, 3], "screen": [0, 1],
                "smoke": [0], "dev": [0, 1, 2], "confirmation": [3]},
                "datasets": {"train": {"path": str(train), "fingerprint": file_fingerprint(train)},
                "valid": {"path": str(valid), "fingerprint": file_fingerprint(valid),
                          "groups": [{"row_group": i, "seq_ix": 100 + i} for i in range(4)]}}}
    manifest_path = write(tmp_path / "manifest.json", manifest)
    report = evaluation(model, valid, [0, 1, 2], 0.4)
    report_path = write(tmp_path / "incumbent_report.json", report)
    data = {"path": str(train), "size": train.stat().st_size, "mtime_ns": train.stat().st_mtime_ns}
    anchor_path = write(tmp_path / "anchor.json", {"specification": {"groups": [0, 1], "data": data}})
    options = dict(manifest_path=manifest_path, incumbent_config_path=config,
                   incumbent_report_path=report_path, training_report_path=anchor_path, source_paths=[code])
    return {"root": tmp_path, "options": options, "model": model, "code": code,
            "valid": valid, "data": data, "registry": tmp_path / "registry.json"}


def evaluation(model, valid, groups, score=0.5):
    return {"weighted_pearson": score, "t0": score, "t1": score, "metric_clip": [-2., 2.],
            "postprocess": {}, "seconds": 1.0, "rows_per_second": 100.,
            "provenance": {"model_sha256": file_hash(model), "dataset_path": str(valid),
                           "dataset_size": valid.stat().st_size},
            "per_sequence": [{"row_group": i, "seq_ix": 100+i, "eligible": True,
                              "t0": score, "t1": score, "weighted_pearson": score} for i in groups]}


def spec_for(setup, name, seed=1):
    out = setup["root"] / name
    return {"hypothesis": "correlation head", "training_groups": [0, 1], "seed": seed,
            "epochs": 0, "parameters": {"penalty": .01, "max_iterations": 200},
            "teacher_lineage": [], "config": {"model": "model.onnx", "postprocess": {}},
            "config_path": str(out / "config.json"), "training_report_path": str(out / "training.json"),
            "source_paths": [str(setup["code"])]}


def materialize(setup, spec):
    path = write(spec["config_path"], spec["config"])
    model = path.parent / spec["config"]["model"]
    model.write_bytes(f"candidate {spec['seed']}".encode())
    write(spec["training_report_path"], {"complete": True, "specification": {
        "groups": spec["training_groups"], "seed": spec["seed"], "epochs": spec["epochs"],
        "source": {"sha256": file_hash(setup["model"])}, "data": setup["data"]}})
    return model


def add_candidate(setup, name="a", seed=1):
    spec = spec_for(setup, name, seed)
    ex.register_candidate(setup["registry"], name, spec)
    return spec, materialize(setup, spec)


def test_create_pins_selected_comparator_and_is_exclusive(setup, monkeypatch):
    monkeypatch.setattr(ex, "INCUMBENT_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="selected incumbent"):
        ex.create_registry(setup["registry"], **setup["options"])
    monkeypatch.setattr(ex, "INCUMBENT_SHA256", file_hash(setup["model"]))
    record = ex.create_registry(setup["registry"], **setup["options"])
    assert record["confirmation_policy"] == "permanently_excluded"
    assert record["caps"] == {"screen": 12, "dev": 4}
    before = setup["registry"].read_bytes()
    with pytest.raises(FileExistsError):
        ex.create_registry(setup["registry"], **setup["options"])
    assert setup["registry"].read_bytes() == before


@pytest.mark.parametrize("caps", [(13, 4), (12, 5), (0, 1), (2, 3), (True, 1)])
def test_invalid_caps_rejected(setup, caps):
    with pytest.raises(ValueError, match="caps"):
        ex.create_registry(setup["registry"], **setup["options"], screen_cap=caps[0], dev_cap=caps[1])


def test_declared_before_fitting_and_no_alias_duplicates(setup):
    ex.create_registry(setup["registry"], **setup["options"])
    spec = spec_for(setup, "a")
    ex.register_candidate(setup["registry"], "a", spec)
    with pytest.raises(ValueError, match="already registered"):
        ex.register_candidate(setup["registry"], "a", spec)
    with pytest.raises(ValueError, match="duplicate candidate"):
        ex.register_candidate(setup["registry"], "renamed", spec_for(setup, "renamed"))
    late = spec_for(setup, "late", seed=2)
    materialize(setup, late)
    with pytest.raises(ValueError, match="declare before fitting"):
        ex.register_candidate(setup["registry"], "late", late)


def test_groups_and_teacher_are_prospective(setup):
    ex.create_registry(setup["registry"], **setup["options"])
    spec = spec_for(setup, "a")
    spec["training_groups"] = [0]
    with pytest.raises(ValueError, match="subset_reason"):
        ex.register_candidate(setup["registry"], "a", spec)
    spec["training_subset_reason"] = "predeclared deterministic robustness subsample"
    spec["teacher_lineage"] = [{"model_sha256": "0" * 64}]
    with pytest.raises(ValueError, match="teacher"):
        ex.register_candidate(setup["registry"], "a", spec)
    spec["teacher_lineage"] = []
    ex.register_candidate(setup["registry"], "a", spec)


def test_successful_screen_dev_uses_incumbent_and_cannot_overwrite(setup):
    ex.create_registry(setup["registry"], **setup["options"])
    _, model = add_candidate(setup)
    report = write(setup["root"] / "screen.json", evaluation(model, setup["valid"], [0, 1]))
    with pytest.raises(ValueError, match="unused reservation"):
        ex.record_result(setup["registry"], "a", "screen", report)
    ex.reserve_evaluation(setup["registry"], "a", "screen")
    result = ex.record_result(setup["registry"], "a", "screen", report)
    assert result["delta"] == pytest.approx(.1)
    assert result["incumbent_wp"] == .4
    with pytest.raises(ValueError, match="overwrite"):
        ex.record_result(setup["registry"], "a", "screen", report)
    ex.reserve_evaluation(setup["registry"], "a", "dev")
    dev = write(setup["root"] / "dev.json", evaluation(model, setup["valid"], [0, 1, 2], .51))
    assert ex.record_result(setup["registry"], "a", "dev", dev)["delta"] == pytest.approx(.11)


@pytest.mark.parametrize("change", ["groups", "seq_ix", "model", "postprocess", "dataset", "clip", "target", "target_mean", "eligibility"])
def test_result_identity_and_metric_guards_are_transactional(setup, change):
    ex.create_registry(setup["registry"], **setup["options"])
    _, model = add_candidate(setup)
    ex.reserve_evaluation(setup["registry"], "a", "screen")
    report = evaluation(model, setup["valid"], [0, 1])
    if change == "groups": report["per_sequence"][1]["row_group"] = 3
    if change == "seq_ix": report["per_sequence"][1]["seq_ix"] = 900
    if change == "model": report["provenance"]["model_sha256"] = "0" * 64
    if change == "postprocess": report["postprocess"] = {"scales": [2, 2]}
    if change == "dataset": report["provenance"]["dataset_size"] += 1
    if change == "clip": report["metric_clip"] = [-6., 6.]
    if change == "target": report["per_sequence"][0]["t0"] = .2
    if change == "target_mean": report["t0"] = .9
    if change == "eligibility": report["per_sequence"][0]["eligible"] = False
    path = write(setup["root"] / "bad.json", report)
    before = setup["registry"].read_bytes()
    with pytest.raises(ValueError):
        ex.record_result(setup["registry"], "a", "screen", path)
    assert setup["registry"].read_bytes() == before


@pytest.mark.parametrize("change", ["source", "training_groups", "training_seed", "config", "data"])
def test_changed_candidate_artifacts_rejected_before_reservation(setup, change):
    ex.create_registry(setup["registry"], **setup["options"])
    spec, _ = add_candidate(setup)
    if change == "source": setup["code"].write_text("x = 2\n")
    elif change == "config": write(spec["config_path"], {**spec["config"], "postprocess": {"scales": [2, 2]}})
    elif change == "data": Path(setup["data"]["path"]).write_bytes(b"changed")
    else:
        training = json.loads(Path(spec["training_report_path"]).read_text())
        training["specification"]["groups" if change == "training_groups" else "seed"] = [0] if change == "training_groups" else 2
        write(spec["training_report_path"], training)
    with pytest.raises(ValueError):
        ex.reserve_evaluation(setup["registry"], "a", "screen")
    assert not json.loads(setup["registry"].read_text())["candidates"]["a"]["evaluations"]


def test_failed_screen_consumes_slot_without_report_and_blocks_dev(setup):
    ex.create_registry(setup["registry"], **setup["options"], screen_cap=1, dev_cap=1)
    add_candidate(setup)
    add_candidate(setup, "b", 2)
    ex.reserve_evaluation(setup["registry"], "a", "screen")
    failure = ex.record_result(setup["registry"], "a", "screen", status="failed", reason="callback failed at row 100")
    assert failure["status"] == "failed" and "report" not in failure
    with pytest.raises(ValueError, match="cap exhausted"):
        ex.reserve_evaluation(setup["registry"], "b", "screen")
    with pytest.raises(ValueError, match="successful screen"):
        ex.reserve_evaluation(setup["registry"], "a", "dev")
    with pytest.raises(ValueError, match="already reserved"):
        ex.reserve_evaluation(setup["registry"], "a", "screen")


def test_no_confirmation_or_unknown_phases(setup):
    ex.create_registry(setup["registry"], **setup["options"])
    add_candidate(setup)
    for phase in ("confirmation", "full", "renamed_confirmation"):
        with pytest.raises(ValueError): ex.reserve_evaluation(setup["registry"], "a", phase)
        with pytest.raises(ValueError): ex.record_result(setup["registry"], "a", phase, status="failed", reason="x")


def test_concurrent_reservations_share_one_cap(setup):
    ex.create_registry(setup["registry"], **setup["options"], screen_cap=1, dev_cap=1)
    add_candidate(setup, "a", 1)
    add_candidate(setup, "b", 2)
    def reserve(name):
        try:
            ex.reserve_evaluation(setup["registry"], name, "screen")
            return "reserved"
        except ValueError as error:
            assert "cap exhausted" in str(error)
            return "capped"
    with ThreadPoolExecutor(2) as pool:
        assert sorted(pool.map(reserve, ["a", "b"])) == ["capped", "reserved"]
    record = json.loads(setup["registry"].read_text())
    assert sum("screen" in c["evaluations"] for c in record["candidates"].values()) == 1


def test_dev_cap_and_frozen_artifact_between_phases(setup):
    ex.create_registry(setup["registry"], **setup["options"], screen_cap=2, dev_cap=1)
    for name, seed in (("a", 1), ("b", 2)):
        _, model = add_candidate(setup, name, seed)
        ex.reserve_evaluation(setup["registry"], name, "screen")
        report = write(setup["root"] / f"{name}_screen.json", evaluation(model, setup["valid"], [0, 1]))
        ex.record_result(setup["registry"], name, "screen", report)
    ex.reserve_evaluation(setup["registry"], "a", "dev")
    with pytest.raises(ValueError, match="cap exhausted"):
        ex.reserve_evaluation(setup["registry"], "b", "dev")
    (setup["root"] / "a" / "model.onnx").write_bytes(b"changed after reservation")
    bad = write(setup["root"] / "a_dev.json", evaluation(setup["root"] / "a" / "model.onnx", setup["valid"], [0, 1, 2]))
    with pytest.raises(ValueError, match="frozen artifact changed"):
        ex.record_result(setup["registry"], "a", "dev", bad)


def test_iteration_cap_and_training_parameter_drift(setup):
    ex.create_registry(setup["registry"], **setup["options"])
    spec = spec_for(setup, "a")
    spec["parameters"]["max_iterations"] = 201
    with pytest.raises(ValueError, match="200 iterations"):
        ex.register_candidate(setup["registry"], "a", spec)
    spec["parameters"]["max_iterations"] = 200
    ex.register_candidate(setup["registry"], "a", spec)
    materialize(setup, spec)
    training = json.loads(Path(spec["training_report_path"]).read_text())
    training["specification"]["parameters"] = {"penalty": 1.0}
    write(spec["training_report_path"], training)
    with pytest.raises(ValueError, match="parameters differ"):
        ex.reserve_evaluation(setup["registry"], "a", "screen")
