import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from connectome import __main__ as cli
from connectome.data import file_fingerprint
from connectome.evaluation import file_hash
from connectome.selection import (claim_confirmation, compare_confirmation_once, freeze_candidate,
                                  select_development_candidate, validate_frozen)

BASELINE = Path(__file__).resolve().parents[1] / "baseline" / "baseline.onnx"


def write_json(path, value):
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def setup(tmp_path):
    dataset = tmp_path / "valid.parquet"
    dataset.write_bytes(b"manifest-only fixture; never opened as parquet")
    manifest = write_json(tmp_path / "manifest.json", {
        "version": 1, "datasets": {"valid": {"path": str(dataset), "fingerprint": file_fingerprint(dataset),
          "num_row_groups": 4, "groups": [{"row_group": g, "seq_ix": 11 + g, "rows": 20000} for g in range(4)]}},
        "splits": {"dev": [0, 1], "screen": [0], "smoke": [0], "confirmation": [2, 3]}})
    model = tmp_path / "candidate.onnx"
    shutil.copyfile(BASELINE, model)
    settings = {"model": str(model), "backend": "onnx", "onnx_mode": "run", "postprocess": {"scales": [0.5, 0.5]}}
    config = write_json(tmp_path / "candidate.json", settings)
    postprocess = write_json(tmp_path / "postprocess.json", settings["postprocess"])
    inference = tmp_path / "runtime_fixture.py"
    inference.write_text("version = 1\n")
    return SimpleNamespace(root=tmp_path, dataset=dataset, manifest=manifest, model=model,
                           config=config, postprocess=postprocess, inference=inference,
                           frozen=tmp_path / "frozen.json", ledger=tmp_path / "manifest.confirmation-access.json")


def freeze(s, path=None):
    return freeze_candidate(s.config, path or s.frozen, manifest_path=s.manifest, inference_paths=[s.inference])


def report(s, path, scores, groups=(2, 3), **provenance):
    result = {"weighted_pearson": float(np.mean(scores)),
              "provenance": {"model_sha256": file_hash(s.model), **provenance},
              "postprocess": json.loads(s.config.read_text()).get("postprocess", {}),
              "per_sequence": [{"row_group": g, "seq_ix": 11 + g, "eligible": True,
                                 "weighted_pearson": score, "t0": score, "t1": score}
                                for g, score in zip(groups, scores)]}
    return write_json(path, result)


def claim(s, report_path=None, frozen_path=None, **kwargs):
    return claim_confirmation(frozen_path or s.frozen, model_path=s.model, groups=kwargs.pop("groups", [2, 3]),
                              report_path=report_path or s.root / "candidate_confirmation.json",
                              postprocess={"scales": [0.5, 0.5]}, **kwargs)


def test_development_selection_uses_all_dev_sequences_and_explicit_feasibility(setup):
    s = setup
    candidates = []
    for name, scores, feasible in [("baseline", [0.2, 0.3], True), ("winner", [0.4, 0.3], True),
                                    ("too_slow", [0.9, 0.9], False)]:
        path = report(s, s.root / f"{name}.json", scores, groups=[0, 1])
        candidates.append({"name": name, "config_path": str(s.config), "dev_report_path": str(path),
                           "feasible": feasible, "feasibility_reason": "measured engineering assessment; judge runtime unknown"})
    result = select_development_candidate(candidates, manifest_path=s.manifest)
    assert result["selected"]["name"] == "winner"
    assert result["confirmation_used"] is False
    partial = json.loads(Path(candidates[1]["dev_report_path"]).read_text())
    partial["per_sequence"] = partial["per_sequence"][:1]
    write_json(Path(candidates[1]["dev_report_path"]), partial)
    with pytest.raises(ValueError, match="complete dev"):
        select_development_candidate(candidates, manifest_path=s.manifest)


@pytest.mark.parametrize("changed,error", [("postprocess", "postprocess differs"),
                                           ("untracked", "tracked postprocess"),
                                           ("model", "does not match candidate model")])
def test_selection_cannot_relabel_postprocessing_or_replaced_model_evidence(setup, changed, error):
    s = setup
    path = report(s, s.root / "dev.json", [0.3, 0.4], groups=[0, 1])
    evidence = json.loads(path.read_text())
    if changed == "postprocess":
        evidence["postprocess"] = {"scales": [0.75, 0.5]}
    elif changed == "untracked":
        evidence["postprocess"] = None
        evidence["postprocess_tracked"] = False
    else:
        evidence["provenance"]["model_sha256"] = "another_source_before_deployment_folding"
    write_json(path, evidence)
    candidate = {"name": "candidate", "config_path": str(s.config), "dev_report_path": str(path),
                 "feasible": True, "feasibility_reason": "measured local resource limits"}
    with pytest.raises(ValueError, match=error):
        select_development_candidate([candidate], manifest_path=s.manifest)
    with pytest.raises(ValueError, match=error):
        freeze_candidate(s.config, s.frozen, manifest_path=s.manifest, development_report=path,
                          inference_paths=[s.inference])
    assert not s.frozen.exists()


def test_freeze_binds_inference_but_not_unrelated_training_files(setup):
    s = setup
    frozen = freeze(s)
    assert frozen["model_sha256"] == file_hash(s.model)
    (s.root / "training.py").write_text("changed unrelated training code\n")
    assert validate_frozen(s.frozen)["freeze_sha256"] == frozen["freeze_sha256"]
    s.inference.write_text("version = 2\n")
    with pytest.raises(ValueError, match="inference artifact changed"):
        validate_frozen(s.frozen)


def test_selection_cannot_compare_different_eligible_block_domains(setup):
    s = setup
    paths = [report(s, s.root / f"report{i}.json", [0.3, 0.4], groups=[0, 1]) for i in range(2)]
    changed = json.loads(paths[1].read_text())
    changed["per_sequence"][0]["eligible"] = False
    changed["weighted_pearson"] = 0.4
    write_json(paths[1], changed)
    candidates = [{"name": str(i), "config_path": str(s.config), "dev_report_path": str(path),
                   "feasible": True, "feasibility_reason": "local engineering checks"} for i, path in enumerate(paths)]
    with pytest.raises(ValueError, match="different eligible scoring domains"):
        select_development_candidate(candidates, manifest_path=s.manifest)


@pytest.mark.parametrize("changed", ["config", "model", "manifest"])
def test_freeze_rejects_changed_config_model_or_manifest(setup, changed):
    s = setup
    freeze(s)
    path = getattr(s, changed)
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="changed after freezing"):
        validate_frozen(s.frozen)


def test_global_manifest_receipt_prevents_refreeze_filename_bypass(setup):
    s = setup
    freeze(s)
    other = s.root / "another_frozen_filename.json"
    freeze(s, other)
    claim(s)
    with pytest.raises(ValueError, match="already opened"):
        claim(s, frozen_path=other)
    with pytest.raises(ValueError, match="already opened"):
        freeze(s, s.root / "third_filename.json")


def test_partial_confirmation_and_changed_postprocessing_do_not_consume_receipt(setup):
    s = setup
    freeze(s)
    with pytest.raises(ValueError, match="exact full"):
        claim(s, groups=[2])
    with pytest.raises(ValueError, match="postprocessing differs"):
        claim_confirmation(s.frozen, model_path=s.model, groups=[2, 3], report_path=s.root / "result.json")
    assert not s.ledger.exists()


@pytest.mark.parametrize("scores,selected,positive_ci", [([0.5, 0.1], "candidate", False),
                                                         ([0.1, 0.2], "baseline", False),
                                                         ([0.2, 0.3], "baseline", False)])
def test_confirmation_mean_policy_bootstrap_and_single_decision(setup, scores, selected, positive_ci):
    s = setup
    freeze(s)
    candidate = report(s, s.root / "candidate_confirmation.json", scores)
    baseline = report(s, s.root / "baseline_confirmation.json", [0.2, 0.3])
    control = json.loads(baseline.read_text())
    control["postprocess"] = {}
    write_json(baseline, control)
    claim(s, candidate)
    result = compare_confirmation_once(s.frozen, candidate, baseline, s.root / "decision.json", n_resamples=1000)
    assert result["selected"] == selected
    assert result["positive_ci_evidence"] == positive_ci
    assert result["no_further_confirmation_tuning"] is True
    with pytest.raises(ValueError, match="already completed"):
        compare_confirmation_once(s.frozen, candidate, baseline, s.root / "another_decision.json")


def test_cli_invalid_bulk_hash_preflight_does_not_burn_confirmation(setup):
    s = setup
    freeze(s)
    parity = write_json(s.root / "parity.json", {"passed": True, "original_sha256": file_hash(s.model),
                                                 "bulk_model_sha256": "incorrect"})
    with pytest.raises(ValueError, match="parity report"):
        cli.main(["evaluate", "--dataset", str(s.dataset), "--manifest", str(s.manifest),
                  "--split", "confirmation", "--model", str(s.model), "--postprocess", str(s.postprocess),
                  "--frozen-config", str(s.frozen), "--bulk-model", str(BASELINE), "--parity-report", str(parity),
                  "--output", str(s.root / "result.json")])
    assert not s.ledger.exists()


def test_cli_candidate_confirmation_is_claimed_after_valid_preflight(setup, monkeypatch):
    from connectome import evaluation
    s = setup
    freeze(s)
    seen = []
    def evaluate(*args, **kwargs):
        assert s.ledger.exists()
        seen.append(args)
        return {"weighted_pearson": 0.5}
    monkeypatch.setattr(evaluation, "evaluate", evaluate)
    cli.main(["evaluate", "--dataset", str(s.dataset), "--manifest", str(s.manifest),
              "--split", "confirmation", "--model", str(s.model), "--postprocess", str(s.postprocess),
              "--frozen-config", str(s.frozen), "--output", str(s.root / "result.json")])
    assert seen[0][2] == [2, 3]


def test_cli_calibration_cannot_use_confirmation_even_with_a_freeze(setup):
    s = setup
    freeze(s)
    with pytest.raises(ValueError, match="reserved"):
        cli.main(["calibrate", "--dataset", str(s.dataset), "--manifest", str(s.manifest),
                  "--split", "confirmation", "--model", str(s.model), "--cache", "never_opened",
                  "--frozen-config", str(s.frozen)])


def test_bulk_execution_and_prediction_cache_record_actual_bulk_hash(tmp_path, monkeypatch):
    from connectome import data, evaluation
    from connectome.calibration import PredictionCache
    dataset = tmp_path / "mock.parquet"
    dataset.write_bytes(b"mock")
    bulk_model = tmp_path / "bulk.onnx"
    bulk_model.write_bytes(b"distinct mocked bulk model bytes")
    need = np.arange(20000) >= 99
    targets = np.ones((20000, 2), np.float32)
    targets[::2] = -1
    sequence = SimpleNamespace(seq_ix=12, features=np.zeros((20000, 112), np.float32), targets=targets,
                               need=need, mask=need)
    class Scalar:
        def as_py(self):
            return 12
    class FakeParquet:
        num_row_groups = 1
        def read_row_group(self, *args, **kwargs):
            return {"seq_ix": [Scalar()]}
    class BulkSession:
        def get_inputs(self):
            return [SimpleNamespace(shape=[1, None, 112]), *[SimpleNamespace(shape=[1, 1, 128]) for _ in range(2)]]
        def run(self, *args, **kwargs):
            return [targets[None], *[np.zeros((1, 1, 128), np.float32) for _ in range(2)]]
    monkeypatch.setattr(data, "read_sequence", lambda *args: sequence)
    monkeypatch.setattr(evaluation.pq, "ParquetFile", lambda *args: FakeParquet())
    monkeypatch.setattr(evaluation.ort, "InferenceSession", lambda *args, **kwargs: BulkSession())
    result = evaluation.evaluate(dataset, BASELINE, groups=[0], bulk_model=bulk_model,
                                 cache_path=tmp_path / "cache", progress=False)
    assert result["provenance"]["bulk_model_sha256"] == file_hash(bulk_model)
    assert result["provenance"]["model_sha256"] == file_hash(BASELINE)
    assert PredictionCache.open(tmp_path / "cache").metadata["provenance"]["bulk_model_sha256"] == file_hash(bulk_model)
