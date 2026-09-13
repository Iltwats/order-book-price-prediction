"""Round-two trial accounting, separate from the consumed confirmation ledger.

These guards prevent accidental reuse and overwrite, not deliberate file edits.
Declare candidates before fitting; reserve each evaluation before running it.
Failed reservations consume their trial slot and may be recorded without a report.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import uuid

from .data import file_fingerprint
from .evaluation import file_hash
from .selection import _validate_report

INCUMBENT_SHA256 = "d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45"
PHASES = {"screen": "screen", "dev": "dev"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_hash(path)}


def _check_artifact(item):
    if file_hash(item["path"]) != item["sha256"]:
        raise ValueError(f"frozen artifact changed: {item['path']}")


@contextmanager
def _lock(path):
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name(path.name + ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield path
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _write(path, value, *, create=False):
    value["registry_sha256"] = _digest({k: v for k, v in value.items() if k != "registry_sha256"})
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if create:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _groups(values, allowed, label):
    if not isinstance(values, list) or not values or any(type(g) is not int for g in values):
        raise ValueError(f"{label} must be a nonempty integer list")
    if len(values) != len(set(values)) or not set(values) <= set(allowed):
        raise ValueError(f"{label} contains duplicates or undeclared groups")
    return values


def _manifest(record):
    _check_artifact(record["manifest"])
    manifest = _read(record["manifest"]["path"])
    for name in ("train", "valid"):
        item = manifest["datasets"][name]
        if file_fingerprint(item["path"]) != item["fingerprint"]:
            raise ValueError(f"{name} dataset fingerprint changed")
    return manifest


def _load(path):
    record = _read(path)
    if record.get("version") != 1 or record.get("registry_sha256") != _digest({k: v for k, v in record.items() if k != "registry_sha256"}):
        raise ValueError("registry contents changed or version is unsupported")
    if record.get("confirmation_policy") != "permanently_excluded":
        raise ValueError("confirmation must remain permanently excluded")
    for item in record["incumbent"].values():
        if isinstance(item, dict) and "path" in item:
            _check_artifact(item)
    _check_artifact(record["training_anchor"])
    return record, _manifest(record)


def _validate_data_report(data, manifest):
    expected = manifest["datasets"]["train"]
    fp = expected["fingerprint"]
    size = data.get("size", data.get("size_bytes"))
    if (Path(data.get("path", "")).resolve() != Path(expected["path"]).resolve()
            or size != fp["size_bytes"] or data.get("mtime_ns") != fp["mtime_ns"]):
        raise ValueError("training report data does not match frozen training file")


def create_registry(path, *, manifest_path, incumbent_config_path, incumbent_report_path,
                    training_report_path, source_paths, screen_cap=12, dev_cap=4):
    """Create a fixed round with at most 12 screens and four full-dev runs.

    training_report_path is the selected affine1024 training report, anchoring
    the first readout experiments to its exact groups. It is not candidate data.
    """
    if (type(screen_cap) is not int or type(dev_cap) is not int
            or not 1 <= screen_cap <= 12 or not 1 <= dev_cap <= min(4, screen_cap)):
        raise ValueError("caps must satisfy 1 <= dev <= 4 and dev <= screen <= 12")
    manifest = _read(manifest_path)
    confirmation = set(manifest["splits"]["confirmation"])
    for split in ("screen", "dev", "smoke"):
        groups = _groups(manifest["splits"][split], manifest["splits"]["dev"], split)
        if set(groups) & confirmation:
            raise ValueError("development splits overlap confirmation")
    config_path = Path(incumbent_config_path).resolve()
    config = _read(config_path)
    model = _artifact(config_path.parent / config["model"])
    if model["sha256"] != INCUMBENT_SHA256:
        raise ValueError("round two comparator must be the selected incumbent, not the raw baseline")
    report = _read(incumbent_report_path)
    wp = _validate_report(report, manifest, "dev")
    if report.get("provenance", {}).get("model_sha256") != INCUMBENT_SHA256 or report.get("postprocess") != config.get("postprocess", {}):
        raise ValueError("incumbent report model/postprocess mismatch")
    training = _read(training_report_path)
    anchor = training["specification"]
    groups = _groups(anchor["groups"], manifest["splits"]["train"], "training groups")
    _validate_data_report(anchor["data"], manifest)
    sources = [_artifact(p) for p in source_paths]
    if not sources:
        raise ValueError("source_paths must record implementation files")
    record = {"version": 1, "created_at": _now(), "confirmation_policy": "permanently_excluded",
              "caps": {"screen": screen_cap, "dev": dev_cap}, "manifest": _artifact(manifest_path),
              "dataset_fingerprints": {k: manifest["datasets"][k]["fingerprint"] for k in ("train", "valid")},
              "incumbent": {"model": model, "config": _artifact(config_path),
                            "report": _artifact(incumbent_report_path), "weighted_pearson": wp},
              "training_anchor": _artifact(training_report_path), "default_training_groups": groups,
              "source_artifacts": sources, "candidates": {}}
    _manifest(record)
    with _lock(path) as destination:
        _write(destination, record, create=True)
    return record


def register_candidate(path, candidate_id, spec):
    """Predeclare an arm before training, including its eventual output paths.

    Required spec keys: hypothesis, training_groups, seed, epochs, parameters,
    teacher_lineage, config (deployment JSON), config_path,
    training_report_path, source_paths. Optional: training_subset_reason.
    Artifacts at the output paths need not exist until reserve_evaluation.
    Every postprocess variant or stochastic seed requires a separate candidate.
    """
    if not isinstance(candidate_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", candidate_id):
        raise ValueError("invalid candidate identifier")
    required = {"hypothesis", "training_groups", "seed", "epochs", "parameters", "teacher_lineage",
                "config", "config_path", "training_report_path", "source_paths"}
    if not isinstance(spec, dict) or not required <= spec.keys():
        raise ValueError(f"candidate spec requires {sorted(required)}")
    spec = json.loads(json.dumps(spec, allow_nan=False))
    if (not isinstance(spec["hypothesis"], str) or not spec["hypothesis"].strip()
            or type(spec["seed"]) is not int or type(spec["epochs"]) is not int or not 0 <= spec["epochs"] <= 3
            or not isinstance(spec["parameters"], dict) or not isinstance(spec["teacher_lineage"], list)):
        raise ValueError("invalid hypothesis, seed, epochs, parameters, or teacher lineage")
    iterations = spec["parameters"].get("max_iterations")
    if iterations is not None and (type(iterations) is not int or not 1 <= iterations <= 200):
        raise ValueError("head optimization is capped at 200 iterations")
    config = spec["config"]
    if (not isinstance(config, dict) or config.get("backend", "onnx") != "onnx"
            or not isinstance(config.get("model"), str) or not isinstance(config.get("postprocess", {}), dict)):
        raise ValueError("candidate must declare an ONNX deployment config and one postprocess variant")
    with _lock(path) as destination:
        record, manifest = _load(destination)
        if candidate_id in record["candidates"]:
            raise ValueError("candidate id already registered")
        groups = _groups(spec["training_groups"], manifest["splits"]["train"], "training groups")
        if groups != record["default_training_groups"] and not spec.get("training_subset_reason"):
            raise ValueError("changed training groups require a prospective training_subset_reason")
        parent = spec.setdefault("parent_model_sha256", INCUMBENT_SHA256)
        if parent != INCUMBENT_SHA256:
            raise ValueError("candidate must use selected incumbent as parent/comparator")
        for teacher in spec["teacher_lineage"]:
            if not isinstance(teacher, dict) or teacher.get("model_sha256") != INCUMBENT_SHA256:
                raise ValueError("this round permits only the selected supplied-data teacher")
        sources = [_artifact(p) for p in spec.pop("source_paths")]
        if not sources:
            raise ValueError("candidate must freeze its source files")
        for key in ("config_path", "training_report_path"):
            spec[key] = str(Path(spec[key]).resolve())
        if Path(spec["config_path"]).exists() or Path(spec["training_report_path"]).exists():
            raise ValueError("candidate output paths already exist; declare before fitting")
        normalized = {k: v for k, v in spec.items() if k not in {"hypothesis", "config_path", "training_report_path", "training_subset_reason"}}
        normalized["config"] = {k: v for k, v in config.items() if k != "model"}
        normalized["source_hashes"] = sorted(s["sha256"] for s in sources)
        signature = _digest(normalized)
        if any(c["signature"] == signature for c in record["candidates"].values()):
            raise ValueError("duplicate candidate configuration under another id/path")
        entry = {"declared_at": _now(), "spec": spec, "signature": signature,
                 "source_artifacts": sources, "evaluations": {}}
        record["candidates"][candidate_id] = entry
        _write(destination, record)
        return entry


def _candidate_artifacts(entry, manifest):
    spec = entry["spec"]
    for source in entry["source_artifacts"]:
        _check_artifact(source)
    config_path = Path(spec["config_path"])
    config = _read(config_path)
    if config != spec["config"]:
        raise ValueError("deployment config differs from declaration")
    training = _read(spec["training_report_path"])
    details = training.get("specification", training)
    groups = details.get("groups", training.get("training_groups"))
    if groups != spec["training_groups"]:
        raise ValueError("candidate training groups differ from declaration")
    if training.get("complete") is not True:
        raise ValueError("candidate training report must be complete")
    if details.get("seed", training.get("seed")) != spec["seed"]:
        raise ValueError("candidate training seed differs from declaration")
    if details.get("epochs", training.get("epochs", spec["epochs"])) != spec["epochs"]:
        raise ValueError("candidate training epochs differ from declaration")
    # Older closed-form reports have no epoch field. Stochastic fits must record
    # it explicitly rather than inheriting a declared value as evidence.
    if spec["epochs"] > 0 and "epochs" not in details and "epochs" not in training:
        raise ValueError("candidate training report must record completed epochs")
    measured_parameters = details.get("parameters", training.get("parameters"))
    if measured_parameters is not None and measured_parameters != spec["parameters"]:
        raise ValueError("candidate training parameters differ from declaration")
    source_sha = details.get("source", {}).get("sha256", training.get("source_sha256"))
    if source_sha != spec["parent_model_sha256"]:
        raise ValueError("candidate training report must identify the selected parent")
    _validate_data_report(details.get("data", {}), manifest)
    return {"model": _artifact(config_path.parent / config["model"]),
            "config": _artifact(config_path), "training_report": _artifact(spec["training_report_path"])}


def reserve_evaluation(path, candidate_id, phase):
    """Consume one immutable screen/dev slot before inspecting its result."""
    if phase not in PHASES:
        raise ValueError("only screen and dev are permitted; confirmation is permanently excluded")
    with _lock(path) as destination:
        record, manifest = _load(destination)
        entry = record["candidates"][candidate_id]
        if phase in entry["evaluations"]:
            raise ValueError("evaluation already reserved; failures do not release slots")
        if phase == "dev" and entry["evaluations"].get("screen", {}).get("status") != "passed":
            raise ValueError("full dev requires a recorded successful screen")
        used = sum(phase in item["evaluations"] for item in record["candidates"].values())
        if used >= record["caps"][phase]:
            raise ValueError(f"{phase} trial cap exhausted")
        artifacts = _candidate_artifacts(entry, manifest)
        if phase == "dev" and artifacts != entry["evaluations"]["screen"]["artifacts"]:
            raise ValueError("candidate artifacts changed after screening")
        reservation = {"reserved_at": _now(), "status": "reserved", "groups": manifest["splits"][PHASES[phase]],
                       "artifacts": artifacts, "incumbent_sha256": INCUMBENT_SHA256}
        entry["evaluations"][phase] = reservation
        _write(destination, record)
        return reservation


def record_result(path, candidate_id, phase, report_path=None, *, status="passed", reason=None):
    """Attach one result to a reservation; record failures even without output."""
    if phase not in PHASES or status not in {"passed", "failed"}:
        raise ValueError("invalid evaluation phase/status")
    if status == "failed" and (not isinstance(reason, str) or not reason.strip()):
        raise ValueError("failed result requires a reason")
    if status == "passed" and report_path is None:
        raise ValueError("passed result requires an evaluation report")
    with _lock(path) as destination:
        record, manifest = _load(destination)
        entry = record["candidates"][candidate_id]
        reservation = entry["evaluations"].get(phase)
        if not reservation or reservation["status"] != "reserved":
            raise ValueError("result requires an unused reservation; no overwrite allowed")
        result = {"status": status, "recorded_at": _now(), "reason": reason}
        if report_path is not None:
            for item in reservation["artifacts"].values():
                _check_artifact(item)
            for item in entry["source_artifacts"]:
                _check_artifact(item)
            report = _read(report_path)
            wp = _validate_report(report, manifest, PHASES[phase])
            provenance = report.get("provenance", {})
            if provenance.get("model_sha256") != reservation["artifacts"]["model"]["sha256"]:
                raise ValueError("evaluation report model differs from reservation")
            if report.get("postprocess") != entry["spec"]["config"].get("postprocess", {}) or report.get("postprocess_tracked") is False:
                raise ValueError("evaluation report postprocess differs or is untracked")
            valid = manifest["datasets"]["valid"]
            if (Path(provenance.get("dataset_path", "")).resolve() != Path(valid["path"]).resolve()
                    or provenance.get("dataset_size") != valid["fingerprint"]["size_bytes"]):
                raise ValueError("evaluation report validation dataset differs")
            if report.get("metric_clip") != [-2.0, 2.0]:
                raise ValueError("evaluation must use official clipping")
            incumbent = _read(record["incumbent"]["report"]["path"])
            controls = {r["row_group"]: r for r in incumbent["per_sequence"]}
            for row in report["per_sequence"]:
                if row["eligible"] != controls[row["row_group"]]["eligible"]:
                    raise ValueError("candidate eligibility differs from incumbent")
                if row["eligible"] and (not all(math.isfinite(row[k]) and abs(row[k]) <= 1 for k in ("t0", "t1"))
                        or not math.isclose((row["t0"] + row["t1"])/2, row["weighted_pearson"], abs_tol=1e-10, rel_tol=0)):
                    raise ValueError("per-target scores do not match sequence WP")
            eligible = [controls[g]["weighted_pearson"] for g in reservation["groups"] if controls[g]["eligible"]]
            base_wp = sum(eligible)/len(eligible)
            for target in ("t0", "t1"):
                values = [row[target] for row in report["per_sequence"] if row["eligible"]]
                if not isinstance(report.get(target), (int, float)) or not math.isclose(
                        report[target], sum(values)/len(values), abs_tol=1e-10, rel_tol=0):
                    raise ValueError("reported per-target mean differs from sequence scores")
            result.update(report=_artifact(report_path), weighted_pearson=wp, incumbent_wp=base_wp,
                          delta=wp-base_wp, t0=report.get("t0"), t1=report.get("t1"),
                          seconds=report.get("seconds"), rows_per_second=report.get("rows_per_second"),
                          environment=report.get("environment"), judge_runtime_verified=False)
        reservation.update(result)
        _write(destination, record)
        return reservation
