"""Temporary-artifact policy/registry guards; no datasets or training runs."""
import copy
import json
from types import SimpleNamespace

import pytest

from scripts import expand_correlation_gru as expansion


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def context(tmp_path, monkeypatch):
    source, anchor, code = [tmp_path / name for name in ("selected.onnx", "anchor.json", "implementation.py")]
    source.write_bytes(b"selected pretrained model fixture")
    code.write_text("# frozen implementation fixture\n")
    groups = list(range(1024))
    write(anchor, {"specification": {"groups": groups}})
    monkeypatch.setattr(expansion, "core", SimpleNamespace(ROUND=tmp_path, SOURCE=source, ANCHOR=anchor))
    monkeypatch.setattr(expansion, "RESULTS", tmp_path / "results.json")
    monkeypatch.setattr(expansion, "SNAPSHOT", tmp_path / "snapshot.json")
    monkeypatch.setattr(expansion, "POLICY", tmp_path / "policy.json")
    source_hash = expansion.runner.file_hash(source)
    monkeypatch.setattr(expansion.experiments, "INCUMBENT_SHA256", source_hash)
    monkeypatch.setattr(expansion.runner, "require_bridge_acceptance", lambda *args: None)
    monkeypatch.setattr(expansion.runner, "sources", lambda: [str(code)])
    record = {"caps": dict(expansion.CAPS), "default_training_groups": groups, "candidates": {}}
    rows = []
    for index, (candidate, rate) in enumerate(expansion.runner.ARMS.items()):
        directory = tmp_path / candidate
        wp = .64 + index*.001
        comparison = {"delta": .001, "ci_low": .0005, "ci_high": .0015, "confidence": .95,
                      "n_resamples": 10000, "sequences": 256, "candidate_wp": wp}
        for filename, contents in (("config.json", {}), ("report.json", {}), ("model.onnx", {}),
                                   ("screen_report.json", {"weighted_pearson": wp}), ("screen_comparison.json", comparison)):
            write(directory / filename, contents)
        artifact = expansion.experiments._artifact
        record["candidates"][candidate] = {
            "spec": {"config_path": str(directory / "config.json"), "parameters": expansion.runner.correlation_parameters(learning_rate=rate),
                     "training_groups": groups[:256], "epochs": 1, "seed": 20260912, "parent_model_sha256": source_hash},
            "source_artifacts": [artifact(code)],
            "evaluations": {"screen": {"status": "passed", "weighted_pearson": wp,
                "artifacts": {name: artifact(directory / filename) for name, filename in
                              (("config", "config.json"), ("training_report", "report.json"), ("model", "model.onnx"))},
                "report": artifact(directory / "screen_report.json")}}}
        rows.append({"candidate": candidate, "wp": wp, "comparison": comparison})
    for index in range(8):
        record["candidates"][f"earlier{index}"] = {"evaluations": {"screen": {"status": "passed"}}}
    for index in range(2):
        record["candidates"][f"earlier{index}"]["evaluations"]["dev"] = {"status": "passed"}
    write(expansion.SNAPSHOT, record)
    summary = {"candidates": rows, "registry_sha256": expansion.runner.file_hash(expansion.SNAPSHOT)}
    write(expansion.RESULTS, summary)
    monkeypatch.setattr(expansion.experiments, "_load", lambda _: (record, {}))
    calls = []

    def register(path, candidate, spec):
        calls.append(candidate)
        copied = copy.deepcopy(spec)
        sources = [expansion.experiments._artifact(p) for p in copied.pop("source_paths")]
        record["candidates"][candidate] = {"spec": copied, "source_artifacts": sources, "evaluations": {}}

    monkeypatch.setattr(expansion.experiments, "register_candidate", register)
    return SimpleNamespace(record=record, summary=summary, code=code, calls=calls, source_hash=source_hash)


def test_freezes_exact_two_seeds_from_selected_source_and_all_anchor_groups(context):
    policy = expansion.prepare()
    assert policy["selected_screen_winner"]["candidate"] == "correlation_gru_lr3e-5"
    assert policy["parameters"]["learning_rate"] == 3e-5
    assert policy["groups"] == list(range(1024))
    assert policy["source"]["sha256"] == policy["teacher_sha256"] == context.source_hash
    assert policy["consumed_at_prepare"] == {"screen": 10, "dev": 2}
    # The incumbent 256-group winner may consume its third dev slot meanwhile.
    context.record["candidates"]["correlation_gru_lr3e-5"]["evaluations"]["dev"] = {"status": "reserved"}
    specs = expansion.declare()
    assert set(specs) == set(expansion.ARMS) and context.calls == list(expansion.ARMS)
    assert {value["seed"] for value in specs.values()} == {20260912, 20260913}
    assert all(value["parent_model_sha256"] == context.source_hash and value["epochs"] == 1
               and value["training_groups"] == list(range(1024)) for value in specs.values())
    assert all(value["teacher_lineage"][0]["model_sha256"] == context.source_hash for value in specs.values())
    expansion.declare()  # Recovering declaration is idempotent, never a third trial.
    assert context.calls == list(expansion.ARMS)
    with pytest.raises(FileExistsError, match="preserve"):
        expansion.prepare()


def test_rejects_nonpositive_ci_and_no_policy_is_written(context):
    row = context.summary["candidates"][0]
    row["comparison"]["ci_low"] = -.0001
    write(expansion.RESULTS, context.summary)
    comparison_path = expansion.core.ROUND / row["candidate"] / "screen_comparison.json"
    write(comparison_path, row["comparison"])
    with pytest.raises(ValueError, match="95% intervals"):
        expansion.prepare()
    assert not expansion.POLICY.exists() and context.calls == []


def test_rejects_changed_source_evidence_and_output_before_declaration(context):
    expansion.prepare()
    context.code.write_text("changed implementation")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        expansion.declare()
    assert context.calls == []


def test_rejects_used_screen_capacity_or_existing_expansion_artifact(context):
    context.record["candidates"]["extra"] = {"evaluations": {"screen": {"status": "failed"}}}
    with pytest.raises(ValueError, match="two final screen slots"):
        expansion.prepare()
    del context.record["candidates"]["extra"]
    (expansion.core.ROUND / next(iter(expansion.ARMS))).mkdir()
    with pytest.raises(ValueError, match="before either candidate"):
        expansion.prepare()
    assert not expansion.POLICY.exists()
