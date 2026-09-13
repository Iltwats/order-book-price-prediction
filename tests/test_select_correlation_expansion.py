"""Synthetic selector evidence only: no datasets, model loading, or evaluations."""
import hashlib
import json

import pytest

from scripts import select_correlation_expansion as selector


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def artifact(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def context(tmp_path, monkeypatch):
    source = tmp_path / "selected.onnx"
    source.write_bytes(b"selected fixture")
    incumbent = artifact(source)["sha256"]
    monkeypatch.setattr(selector, "INCUMBENT", incumbent)
    code = tmp_path / "frozen.py"
    code.write_text("# frozen fixture\n")
    groups, train_groups = list(range(256)), list(range(1024))
    rows = lambda wp: [{"row_group": g, "eligible": True, "t0": wp, "t1": wp, "weighted_pearson": wp} for g in groups]
    manifest_path, baseline_path, anchor = [tmp_path / n for n in ("manifest.json", "incumbent.json", "anchor.json")]
    datasets = {name: {"path": str(tmp_path / f"{name}.parquet"), "fingerprint": {"size_bytes": 100, "mtime_ns": 123}}
                for name in ("train", "valid")}
    write(manifest_path, {"splits": {"screen": groups, "train": train_groups, "confirmation": [1200]}, "datasets": datasets})
    write(baseline_path, {"per_sequence": rows(.63)})
    write(anchor, {"groups": train_groups})
    registry = {"registry_sha256": "", "confirmation_policy": "permanently_excluded", "caps": dict(selector.CAPS),
                "default_training_groups": train_groups, "manifest": artifact(manifest_path), "training_anchor": artifact(anchor),
                "incumbent": {"model": artifact(source), "report": artifact(baseline_path)}, "candidates": {}}
    policy_path, registry_path, output = [tmp_path / n for n in ("policy.json", "registry.json", "winner.json")]
    parameters = {"learning_rate": 3e-5}

    def signature(entry):
        spec = entry["spec"]
        normal = {k: v for k, v in spec.items() if k not in {"hypothesis", "config_path", "training_report_path", "training_subset_reason"}}
        normal["config"] = {k: v for k, v in spec["config"].items() if k != "model"}
        normal["source_hashes"] = sorted(a["sha256"] for a in entry["source_artifacts"])
        entry["signature"] = selector.digest(normal)

    def candidate(name, wp, seed=20260912, expansion=False):
        directory = tmp_path / name
        directory.mkdir(exist_ok=True)
        model, bulk = directory / "model.onnx", directory / "sequence.onnx"
        model.write_bytes(name.encode()); bulk.write_bytes((name + " bulk").encode())
        config = {"model": "model.onnx", "postprocess": {}}
        spec = {"hypothesis": "fixture", "training_groups": train_groups if expansion else groups,
                "seed": seed, "epochs": 1, "parameters": parameters, "parent_model_sha256": incumbent,
                "config": config, "config_path": str(directory / "config.json"), "training_report_path": str(directory / "report.json")}
        write(directory / "config.json", config)
        training = {"complete": True, "model_sha256": artifact(model)["sha256"], "lower_recurrence_bitwise_equal": True,
                    "teacher_model_sha256": incumbent, "sequence_model_path": str(bulk), "sequence_model_sha256": artifact(bulk)["sha256"],
                    "specification": {"source": artifact(source), "groups": spec["training_groups"], "seed": seed,
                                      "epochs": 1, "parameters": parameters,
                                      "data": {"path": datasets["train"]["path"], "size": 100, "mtime_ns": 123}}}
        write(directory / "report.json", training)
        report = {"per_sequence": rows(wp), "weighted_pearson": wp, "metric_clip": [-2., 2.], "postprocess": {},
                  "provenance": {"model_sha256": artifact(model)["sha256"], "bulk_model_sha256": artifact(bulk)["sha256"],
                                 "dataset_path": datasets["valid"]["path"], "dataset_size": 100}}
        write(directory / "screen_report.json", report)
        write(directory / "screen_comparison.json", {"candidate_wp": wp, "baseline_wp": .63, "delta": wp-.63,
                "confidence": .95, "n_resamples": 10000, "seed": 20260913, "sequences": 256,
                "ci_low": wp-.631, "ci_high": wp-.629})
        receipt = {"status": "passed", "groups": groups, "incumbent_sha256": incumbent,
                   "weighted_pearson": wp, "incumbent_wp": .63, "delta": wp-.63,
                   "artifacts": {"model": artifact(model), "config": artifact(directory / "config.json"),
                                 "training_report": artifact(directory / "report.json")},
                   "report": artifact(directory / "screen_report.json")}
        sources = [artifact(code)] + ([artifact(policy_path)] if expansion else [])
        entry = {"spec": spec, "source_artifacts": sources, "evaluations": {"screen": receipt}}
        signature(entry)
        registry["candidates"][name] = entry
        return entry

    prior_name = "correlation_gru_lr3e-5"
    candidate(prior_name, .64)
    policy = {"version": "correlation_gru_1024_expansion.v1", "arms": dict(selector.ARMS), "caps": dict(selector.CAPS),
              "epochs": 1, "parameters": parameters, "groups": train_groups, "source": artifact(source),
              "teacher_sha256": incumbent, "source_artifacts": [artifact(code)],
              "selected_screen_winner": {"candidate": prior_name, "screen_wp": .64}}
    write(policy_path, policy)
    for index, (name, seed) in enumerate(selector.ARMS.items()):
        candidate(name, .65+index*.001, seed, True)
    for i in range(9):
        evaluations = {"screen": {"status": "failed"}}
        if i < 3:
            evaluations["dev"] = {"status": "passed"}
        registry["candidates"][f"old{i}"] = {"evaluations": evaluations}

    def save():
        registry["registry_sha256"] = selector.digest({k: v for k, v in registry.items() if k != "registry_sha256"})
        write(registry_path, registry)

    def run():
        save()
        return selector.select(output, registry_path=registry_path, policy_path=policy_path)

    return {"run": run, "save": save, "registry": registry, "candidate": candidate, "output": output,
            "code": code, "registry_path": registry_path, "policy_path": policy_path, "root": tmp_path}


def test_selects_highest_and_freezes_evidence_without_mutating_registry(context):
    context["save"]()
    before = context["registry_path"].read_bytes()
    result = context["run"]()
    assert result["state"] == "selected_for_performance_not_dev_approved"
    assert result["winner"]["candidate"] == "correlation_gru1024_seed20260913"
    assert result["dev_approved"] is False and result["consumed"] == {"screen": 12, "dev": 3}
    assert result["registry_snapshot"] == context["registry"]
    assert context["registry_path"].read_bytes() == before
    assert artifact(context["policy_path"]) in result["evidence"]
    with pytest.raises(FileExistsError, match="no overwrite"):
        context["run"]()


def test_requires_both_screens_complete_even_if_first_is_positive(context):
    context["registry"]["candidates"][next(iter(selector.ARMS))]["evaluations"]["screen"]["status"] = "reserved"
    with pytest.raises(ValueError, match="both expansion screens must be complete"):
        context["run"]()
    assert not context["output"].exists()


@pytest.mark.parametrize("scores", [(.62, .625), (.639, .64)])
def test_negative_or_lower_than_prior_is_no_candidate(context, scores):
    for (name, seed), score in zip(selector.ARMS.items(), scores):
        context["candidate"](name, score, seed, True)
    result = context["run"]()
    assert result["state"] == "no_candidate" and result["winner"] is None
    assert not any(row["eligible_for_performance"] for row in result["candidates"])


@pytest.mark.parametrize("key,value", [("baseline_wp", .62), ("delta", .9)])
def test_rejects_paired_comparison_against_wrong_baseline(context, key, value):
    path = context["root"] / next(iter(selector.ARMS)) / "screen_comparison.json"
    data = json.loads(path.read_text()); data[key] = value; write(path, data)
    with pytest.raises(ValueError, match="paired comparison identity"):
        context["run"]()
    assert not context["output"].exists()


def test_stale_source_or_model_evidence_is_rejected(context):
    context["code"].write_text("changed")
    with pytest.raises(ValueError, match="stale evidence"):
        context["run"]()


def test_last_dev_must_still_be_available(context):
    context["registry"]["candidates"]["old4"]["evaluations"]["dev"] = {"status": "reserved"}
    with pytest.raises(ValueError, match="exactly three dev"):
        context["run"]()


def test_changed_training_groups_cannot_be_presented_as_expansion(context):
    entry = context["registry"]["candidates"][next(iter(selector.ARMS))]
    entry["spec"]["training_groups"] = list(range(256))
    with pytest.raises(ValueError, match="signature differs"):
        context["run"]()
