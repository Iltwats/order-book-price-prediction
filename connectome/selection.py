"""Development selection and one immutable, complete confirmation comparison.

These are workflow safeguards against accidental reuse, not access controls on
locally available data. A new research round requires an explicit new protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from .calibration import paired_bootstrap
from .evaluation import file_hash
from .global_metric import (METRIC_ID as GLOBAL_METRIC_ID, STATISTIC_COLUMNS,
                            merge_statistics, score_statistics, paired_global_bootstrap)
from utils import LEGACY_METRIC_ID

ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return json.loads(Path(path).read_text())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _exclusive_json(path, value):
    """Publish a complete JSON file without ever replacing an existing artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _ledger(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    return manifest_path.with_name(manifest_path.stem + ".confirmation-access.json")


def _report(path, key=None):
    value = _read(path)
    if key:
        value = value[key]
    elif "per_sequence" not in value and "best" in value:
        value = value["best"]
    return value


def _metric_id(report):
    metric = report.get("metric_id", LEGACY_METRIC_ID)
    if metric == "v2_wp_block_clip2_v1":
        metric = LEGACY_METRIC_ID
    if metric not in {LEGACY_METRIC_ID, GLOBAL_METRIC_ID}:
        raise ValueError("unsupported report metric identifier")
    return metric


def _global_domain(report):
    """Selected-row counts and label moments bind candidates to one domain."""
    return {entry["row_group"]: (entry["selected_rows"],
                tuple(np.asarray(entry["sufficient_statistics"], np.float64)[:, [0, 1, 3]].ravel()))
            for entry in report["per_sequence"]}


def _check_dataset_identity(first, second):
    left, right = first.get("provenance", {}), second.get("provenance", {})
    for key in ("dataset_path", "dataset_size", "dataset_fingerprint", "dataset_sha256"):
        if key in left and key in right and left[key] != right[key]:
            raise ValueError("dataset provenance differs between candidate reports")
    if "dataset" in first and "dataset" in second and first["dataset"] != second["dataset"]:
        raise ValueError("dataset provenance differs between candidate reports")


def _validate_global_report(report, entries):
    statistics, counts = [], []
    for entry in entries:
        count = entry.get("selected_rows")
        if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
                or not 0 <= count <= 19901):
            raise ValueError("global report needs valid per-sequence selected-row counts")
        stats = np.asarray(entry.get("sufficient_statistics"), np.float64)
        if (stats.shape != (2, 6) or not np.isfinite(stats).all() or np.any(stats[:, 0] < 0)
                or np.any(stats[:, 0] > 2 * count + 1e-8)
                or count == 0 and np.any(stats != 0)):
            raise ValueError("invalid per-sequence global sufficient statistics")
        tolerance = 1e-8 + 1e-12 * stats[:, 0]
        if (np.any(np.abs(stats[:, 1:3]) > 2 + 1e-12) or np.any(stats[:, 3:5] < 0)
                or np.any(stats[:, 3:5] > 4 * stats[:, 0, None] + tolerance[:, None])
                or np.any(np.abs(stats[:, 5]) > np.sqrt(stats[:, 3] * stats[:, 4]) + tolerance)
                or np.any(stats[stats[:, 0] == 0] != 0)):
            raise ValueError("invalid centered global moment invariants")
        statistics.append(stats)
        counts.append(int(count))
    if (sum(counts) == 0 or isinstance(report.get("selected_rows"), bool)
            or not isinstance(report.get("selected_rows"), (int, np.integer))
            or report["selected_rows"] != sum(counts)):
        raise ValueError("global selected-row count differs from sequence statistics")
    for key, expected in (("blocks", len(entries)), ("effective_scored_rows", sum(counts)),
                          ("blocks_with_selected_rows", sum(count > 0 for count in counts)),
                          ("empty_mask_blocks", sum(count == 0 for count in counts))):
        if key in report and report[key] != expected:
            raise ValueError(f"global report {key} count differs from sequence statistics")
    combined = merge_statistics(np.stack(statistics), axis=0)
    if "statistic_columns" in report and report["statistic_columns"] != list(STATISTIC_COLUMNS):
        raise ValueError("global sufficient statistic columns differ from the centered schema")
    declared = np.asarray(report.get("sufficient_statistics"), np.float64)
    if (declared.shape != (2, 6) or not np.isfinite(declared).all()
            or not np.allclose(declared, combined, atol=1e-8, rtol=1e-12)):
        raise ValueError("global pooled statistics differ from per-sequence statistics")
    scores = score_statistics(combined)
    wp = float(scores.mean())
    if not np.isfinite(report.get("weighted_pearson", np.nan)) or not np.isclose(report["weighted_pearson"], wp, atol=1e-10, rtol=0):
        raise ValueError("reported Global WP differs from pooled sufficient statistics")
    for key, expected in zip(("t0", "t1"), scores):
        if key in report and (not np.isfinite(report[key]) or not np.isclose(report[key], expected, atol=1e-10, rtol=0)):
            raise ValueError("reported global target correlation differs from pooled statistics")
    return wp


def _validate_report(report, manifest, split):
    entries = report.get("per_sequence", [])
    groups = [item["row_group"] for item in entries]
    if any(isinstance(group, bool) or not isinstance(group, (int, np.integer)) for group in groups):
        raise ValueError("report row groups must be integers")
    if len(set(groups)) != len(groups) or set(groups) != set(manifest["splits"][split]):
        raise ValueError(f"report must contain exactly the complete {split} sequence set")
    expected = {g["row_group"]: g["seq_ix"] for g in manifest["datasets"]["valid"]["groups"]}
    for item in entries:
        if (isinstance(item["seq_ix"], bool) or not isinstance(item["seq_ix"], (int, np.integer))
                or item["seq_ix"] != expected[item["row_group"]]):
            raise ValueError("report sequence identifier differs from manifest")
    if _metric_id(report) == GLOBAL_METRIC_ID:
        return _validate_global_report(report, entries)
    scores = np.asarray([item["weighted_pearson"] for item in entries if item["eligible"]], np.float64)
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("report contains no eligible or finite sequence scores")
    if np.any(np.abs(scores) > 1):
        raise ValueError("report correlations must be in [-1, 1]")
    if not np.isfinite(report["weighted_pearson"]) or not np.isclose(scores.mean(), report["weighted_pearson"], rtol=0, atol=1e-10):
        raise ValueError("reported WP does not equal the mean eligible sequence score")
    return float(scores.mean())


def _validate_candidate_mapping(report, settings, model_sha):
    recorded_sha = report.get("provenance", {}).get("model_sha256")
    if recorded_sha != model_sha:
        raise ValueError("development report does not match candidate model; new/folded artifacts require matching evaluation evidence")
    if "postprocess" not in report or report["postprocess"] is None or report.get("postprocess_tracked") is False:
        raise ValueError("development report must record a tracked postprocess configuration")
    if report["postprocess"] != settings.get("postprocess", {}):
        raise ValueError("development report postprocess differs from candidate configuration")


def select_development_candidate(candidates, *, manifest_path, output_path=None):
    """Choose highest complete-development WP among explicitly feasible candidates.

    Each entry supplies name, config_path, dev_report_path, feasible (bool), and
    feasibility_reason. Include the original baseline as a candidate. Feasibility
    is a caller's documented engineering assessment; unknown judge runtime must
    not be represented as verified merely by setting this flag.
    """
    manifest = _read(manifest_path)
    ranking, names, domain, first_report, metric = [], set(), None, None, None
    for item in candidates:
        if item["name"] in names:
            raise ValueError("candidate names must be unique")
        names.add(item["name"])
        if not isinstance(item.get("feasible"), bool) or not item.get("feasibility_reason"):
            raise ValueError("each candidate needs an explicit feasibility assessment and reason")
        path = Path(item["dev_report_path"]).resolve()
        report = _report(path, item.get("report_key"))
        wp = _validate_report(report, manifest, "dev")
        report_metric = _metric_id(report)
        if metric is not None and report_metric != metric:
            raise ValueError("development candidates mix legacy and global metric reports")
        current_domain = (_global_domain(report) if report_metric == GLOBAL_METRIC_ID else
                          {entry["row_group"]: bool(entry["eligible"]) for entry in report["per_sequence"]})
        if domain is not None and current_domain != domain:
            raise ValueError("development candidates have different global scoring domains" if report_metric == GLOBAL_METRIC_ID else
                             "development candidates have different eligible scoring domains")
        if first_report is not None and report_metric == GLOBAL_METRIC_ID:
            _check_dataset_identity(first_report, report)
        domain, metric = current_domain, report_metric
        if first_report is None:
            first_report = report
        config_path = Path(item["config_path"]).resolve()
        settings = _read(config_path)
        model_path = (config_path.parent / settings["model"]).resolve()
        model_sha = file_hash(model_path)
        _validate_candidate_mapping(report, settings, model_sha)
        ranking.append({"name": item["name"], "weighted_pearson": wp, "metric_id": report_metric,
                        "feasible": item["feasible"], "feasibility_reason": item["feasibility_reason"],
                        "config_path": str(config_path), "config_sha256": file_hash(config_path),
                        "model_sha256": model_sha, "dev_report_path": str(path),
                        "dev_report_sha256": file_hash(path)})
    available = [item for item in ranking if item["feasible"]]
    if not available:
        raise ValueError("no feasible development candidate")
    # Stable ordering makes an explicitly first baseline win an exact tie.
    selected = max(available, key=lambda item: item["weighted_pearson"])
    result = {"version": 1, "phase": "development_selection", "selected": selected, "metric_id": metric,
              "ranking": sorted(ranking, key=lambda item: -item["weighted_pearson"]),
              "manifest_sha256": file_hash(manifest_path), "confirmation_used": False,
              "policy": "highest development WP among documented feasible candidates; no confirmation tuning"}
    if output_path is not None:
        _exclusive_json(output_path, result)
    return result


def freeze_candidate(config_path, output_path, *, manifest_path="artifacts/manifest.json",
                     development_report=None, inference_paths=None):
    manifest_path = Path(manifest_path).resolve()
    ledger = _ledger(manifest_path)
    if ledger.exists():
        raise ValueError("confirmation already opened for this manifest; another freeze would reuse the holdout")
    manifest = _read(manifest_path)
    config_path = Path(config_path).resolve()
    settings = _read(config_path)
    model_path = (config_path.parent / settings["model"]).resolve()
    if inference_paths is None:
        inference_paths = [ROOT / "solution.py", ROOT / "connectome" / "__init__.py", ROOT / "connectome" / "runtime.py"]
        if settings.get("backend", "onnx") == "native":
            inference_paths.append(ROOT / "connectome" / "native.py")
    artifacts = {str(Path(path).resolve()): file_hash(path) for path in inference_paths}
    if settings.get("backend", "onnx") == "native":
        for name in [settings["native_weights"], *settings.get("native_libraries", {}).values()]:
            path = (config_path.parent / name).resolve()
            artifacts[str(path)] = file_hash(path)
    frozen = {"version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "metric_id": LEGACY_METRIC_ID,
              "config_path": str(config_path), "config_sha256": file_hash(config_path),
              "model_path": str(model_path), "model_sha256": file_hash(model_path),
              "manifest_path": str(manifest_path), "manifest_sha256": file_hash(manifest_path),
              "confirmation_groups": list(manifest["splits"]["confirmation"]),
              "confirmation_ledger": str(ledger), "inference_artifact_sha256": artifacts,
              "policy": "one full confirmation comparison; positive mean paired delta selects candidate, otherwise baseline"}
    if development_report is not None:
        path = Path(development_report).resolve()
        evidence = _report(path)
        _validate_report(evidence, manifest, "dev")
        _validate_candidate_mapping(evidence, settings, frozen["model_sha256"])
        frozen["metric_id"] = _metric_id(evidence)
        if frozen["metric_id"] == GLOBAL_METRIC_ID:
            frozen["policy"] = "one full confirmation comparison; positive pooled Global WP delta selects candidate, otherwise baseline"
        frozen["development_report"] = {"path": str(path), "sha256": file_hash(path)}
    frozen["freeze_sha256"] = _digest(frozen)
    _exclusive_json(output_path, frozen)
    return frozen


def validate_frozen(frozen_path):
    frozen = _read(frozen_path)
    original_digest = frozen.get("freeze_sha256")
    if frozen.get("version") != 1 or original_digest != _digest({k: v for k, v in frozen.items() if k != "freeze_sha256"}):
        raise ValueError("frozen candidate record changed or is unsupported")
    for path_key, hash_key in [("config_path", "config_sha256"), ("model_path", "model_sha256"),
                                ("manifest_path", "manifest_sha256")]:
        if file_hash(frozen[path_key]) != frozen[hash_key]:
            raise ValueError(f"{path_key} changed after freezing")
    for path, expected in frozen["inference_artifact_sha256"].items():
        if file_hash(path) != expected:
            raise ValueError(f"inference artifact changed after freezing: {path}")
    if "development_report" in frozen:
        item = frozen["development_report"]
        if file_hash(item["path"]) != item["sha256"]:
            raise ValueError("development evidence changed after freezing")
    return frozen


def claim_confirmation(frozen_path, *, model_path, groups, report_path, postprocess=None,
                       mode="run", bulk_model=None, parity_report=None, metric_id=None):
    """Call only after all CLI/model/parity preflight succeeds, immediately before inference."""
    frozen = validate_frozen(frozen_path)
    frozen_metric = _metric_id(frozen)
    if metric_id is not None and _metric_id({"metric_id": metric_id}) != frozen_metric:
        raise ValueError("confirmation metric differs from frozen development evidence")
    groups = [int(g) for g in groups]
    if len(set(groups)) != len(groups) or set(groups) != set(frozen["confirmation_groups"]):
        raise ValueError("candidate confirmation must use the exact full confirmation split")
    if file_hash(model_path) != frozen["model_sha256"]:
        raise ValueError("model differs from frozen candidate")
    settings = _read(frozen["config_path"])
    if (postprocess or {}) != settings.get("postprocess", {}):
        raise ValueError("postprocessing differs from frozen candidate")
    if settings.get("backend", "onnx") != "onnx":
        raise ValueError("native confirmation requires a separately validated native evaluation path")
    if bulk_model is None and mode != settings.get("onnx_mode", "run"):
        raise ValueError("runtime mode differs from frozen candidate")
    receipt = {"version": 1, "freeze_sha256": frozen["freeze_sha256"], "metric_id": frozen_metric,
               "frozen_path": str(Path(frozen_path).resolve()), "model_sha256": frozen["model_sha256"],
               "groups": groups, "report_path": str(Path(report_path).resolve()),
               "opened_at": datetime.now(timezone.utc).isoformat()}
    if bulk_model is not None:
        parity = _read(parity_report) if parity_report else {}
        bulk_sha = file_hash(bulk_model)
        if (not parity.get("passed") or parity.get("original_sha256") != frozen["model_sha256"]
                or parity.get("bulk_model_sha256") != bulk_sha):
            raise ValueError("bulk parity report does not validate the frozen candidate")
        receipt["bulk_model_sha256"] = bulk_sha
        receipt["parity_report_sha256"] = file_hash(parity_report)
    try:
        _exclusive_json(frozen["confirmation_ledger"], receipt)
    except FileExistsError as exc:
        raise ValueError("confirmation already opened for this manifest; inspect the saved report") from exc
    return receipt


def compare_confirmation_once(frozen_path, candidate_report, baseline_report, output_path,
                              *, n_resamples=10000, seed=20260912):
    """Compare one full confirmation domain under the frozen metric."""
    frozen = validate_frozen(frozen_path)
    receipt_path = Path(frozen["confirmation_ledger"])
    receipt = _read(receipt_path)
    if receipt["freeze_sha256"] != frozen["freeze_sha256"]:
        raise ValueError("confirmation was opened for another frozen candidate")
    if Path(candidate_report).resolve() != Path(receipt["report_path"]):
        raise ValueError("candidate report is not the report reserved before confirmation")
    candidate, baseline = _report(candidate_report), _report(baseline_report)
    metric = _metric_id(frozen)
    if (_metric_id(candidate) != metric or _metric_id(baseline) != metric
            or _metric_id(receipt) != metric):
        raise ValueError("confirmation reports and receipt must match the frozen metric")
    manifest = _read(frozen["manifest_path"])
    _validate_report(candidate, manifest, "confirmation")
    _validate_report(baseline, manifest, "confirmation")
    if candidate.get("provenance", {}).get("model_sha256") != frozen["model_sha256"]:
        raise ValueError("confirmation report model differs from frozen candidate")
    if candidate.get("postprocess") != _read(frozen["config_path"]).get("postprocess", {}):
        raise ValueError("confirmation report postprocessing differs from frozen candidate")
    if baseline.get("provenance", {}).get("model_sha256") != file_hash(ROOT / "baseline" / "baseline.onnx"):
        raise ValueError("baseline confirmation report must use the original control model")
    if baseline.get("postprocess", {}) != {}:
        raise ValueError("confirmation baseline report must be the unmodified original control")
    if receipt.get("bulk_model_sha256") != candidate.get("provenance", {}).get("bulk_model_sha256"):
        raise ValueError("confirmation bulk artifact differs from reserved execution")
    if metric == GLOBAL_METRIC_ID:
        if _global_domain(candidate) != _global_domain(baseline):
            raise ValueError("confirmation candidates have different global scoring domains")
        _check_dataset_identity(candidate, baseline)
    bootstrap = paired_global_bootstrap if metric == GLOBAL_METRIC_ID else paired_bootstrap
    comparison = bootstrap(candidate, baseline, n_resamples=n_resamples, seed=seed)
    result = {"version": 1, "phase": "final_confirmation_decision", "metric_id": metric,
              "freeze_sha256": frozen["freeze_sha256"],
              "selected": "candidate" if comparison["delta"] > 0 else "baseline",
              "policy": "positive mean paired confirmation delta selects candidate; zero/negative retains original baseline",
              "bootstrap": comparison, "positive_ci_evidence": comparison["ci_low"] > 0,
              "candidate_report": str(Path(candidate_report).resolve()),
              "candidate_report_sha256": file_hash(candidate_report),
              "baseline_report": str(Path(baseline_report).resolve()),
              "baseline_report_sha256": file_hash(baseline_report),
              "no_further_confirmation_tuning": True}
    if metric == GLOBAL_METRIC_ID:
        result["policy"] = "positive pooled Global WP confirmation delta selects candidate; zero/negative retains original baseline"
    decision_ledger = receipt_path.with_name(receipt_path.stem + ".decision.json")
    try:
        _exclusive_json(decision_ledger, result)
    except FileExistsError as exc:
        raise ValueError("confirmation comparison already completed; inspect its saved decision") from exc
    _exclusive_json(output_path, result)
    return result
