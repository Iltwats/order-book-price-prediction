"""Freeze a performance-test candidate after both final expansion screens finish."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / "artifacts/round2"
INCUMBENT = "d920986b8c42b256c652e9f695e6ca52af1be7c3ecc88cd0596de41ed5653a45"
ARMS = {"correlation_gru1024_seed20260912": 20260912,
        "correlation_gru1024_seed20260913": 20260913}
CAPS = {"screen": 12, "dev": 4}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def close(a, b):
    return isinstance(a, (int, float)) and isinstance(b, (int, float)) and math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=0, abs_tol=1e-12)


@contextmanager
def registry_lock(path):
    with path.with_name(path.name + ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def select(output_path, *, registry_path=ROUND / "registry.json",
           policy_path=ROUND / "correlation_expansion_policy.json"):
    """Write one immutable receipt; never fit, benchmark, score, or reserve dev."""
    output_path, registry_path = Path(output_path), Path(registry_path)
    if output_path.exists():
        raise FileExistsError("selection receipt already exists; no overwrite")
    with registry_lock(registry_path):
        evidence = {}

        def artifact(path, expected=None):
            path = Path(path).resolve()
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if expected is not None and sha != expected:
                raise ValueError(f"stale evidence: {path}")
            evidence[str(path)] = {"path": str(path), "sha256": sha}
            return evidence[str(path)]

        def check(ref):
            return artifact(ref["path"], ref["sha256"])

        registry_ref = artifact(registry_path)
        registry = read(registry_path)
        if (registry.get("registry_sha256") != digest({k: v for k, v in registry.items() if k != "registry_sha256"})
                or registry.get("confirmation_policy") != "permanently_excluded" or registry["caps"] != CAPS):
            raise ValueError("registry integrity, confirmation boundary, or caps changed")
        used = {phase: sum(phase in row["evaluations"] for row in registry["candidates"].values()) for phase in CAPS}
        if used != {"screen": 12, "dev": 3}:
            raise ValueError("selection requires twelve consumed screens and exactly three dev slots")
        policy_ref = artifact(policy_path)
        policy = read(policy_path)
        if (policy["version"] != "correlation_gru_1024_expansion.v1" or policy["arms"] != ARMS
                or policy["caps"] != CAPS or policy["epochs"] != 1
                or policy["source"]["sha256"] != INCUMBENT or policy["teacher_sha256"] != INCUMBENT
                or policy["groups"] != registry["default_training_groups"] or len(set(policy["groups"])) != 1024):
            raise ValueError("expansion policy contract changed")
        for ref in [policy["source"], *policy["source_artifacts"], registry["manifest"], registry["training_anchor"]]:
            check(ref)
        for ref in registry["incumbent"].values():
            if isinstance(ref, dict) and "path" in ref:
                check(ref)
        if registry["incumbent"]["model"]["sha256"] != INCUMBENT:
            raise ValueError("wrong incumbent")
        manifest = read(registry["manifest"]["path"])
        groups = manifest["splits"]["screen"]
        if len(groups) != 256 or len(set(groups)) != 256 or set(groups) & set(manifest["splits"]["confirmation"]):
            raise ValueError("wrong screen or confirmation overlap")
        baseline = read(registry["incumbent"]["report"]["path"])
        controls = {row["row_group"]: row for row in baseline["per_sequence"]}
        baseline_wp = sum(controls[g]["weighted_pearson"] for g in groups if controls[g]["eligible"]) / sum(controls[g]["eligible"] for g in groups)

        def screen(candidate, *, expansion=False):
            entry = registry["candidates"][candidate]
            spec = entry["spec"]
            normalized = {k: v for k, v in spec.items() if k not in {"hypothesis", "config_path", "training_report_path", "training_subset_reason"}}
            normalized["config"] = {k: v for k, v in spec["config"].items() if k != "model"}
            normalized["source_hashes"] = sorted(a["sha256"] for a in entry["source_artifacts"])
            if digest(normalized) != entry["signature"]:
                raise ValueError("candidate signature differs")
            for ref in entry["source_artifacts"]:
                check(ref)
            if expansion and (policy_ref not in entry["source_artifacts"] or spec["training_groups"] != policy["groups"]
                    or spec["seed"] != ARMS[candidate] or spec["epochs"] != 1 or spec["parameters"] != policy["parameters"]):
                raise ValueError("expansion declaration differs from policy")
            if not set(spec["training_groups"]) <= set(manifest["splits"]["train"]) or spec["parent_model_sha256"] != INCUMBENT:
                raise ValueError("wrong training groups or source")
            receipt = entry["evaluations"].get("screen", {})
            if receipt.get("status") not in {"passed", "failed"}:
                raise ValueError("both expansion screens must be complete")
            if receipt["groups"] != groups or receipt["incumbent_sha256"] != INCUMBENT:
                raise ValueError("screen reservation differs")
            refs = receipt["artifacts"]
            for ref in refs.values():
                check(ref)
            if (Path(refs["config"]["path"]).resolve() != Path(spec["config_path"]).resolve()
                    or Path(refs["training_report"]["path"]).resolve() != Path(spec["training_report_path"]).resolve()
                    or read(spec["config_path"]) != spec["config"]):
                raise ValueError("config/report declaration differs")
            training = read(refs["training_report"]["path"])
            details = training["specification"]
            data, expected = details["data"], manifest["datasets"]["train"]
            if (not training["complete"] or training["model_sha256"] != refs["model"]["sha256"]
                    or not training["lower_recurrence_bitwise_equal"] or training["teacher_model_sha256"] != INCUMBENT
                    or details["source"]["sha256"] != INCUMBENT or details["groups"] != spec["training_groups"]
                    or details["seed"] != spec["seed"] or details["epochs"] != spec["epochs"] or details["parameters"] != spec["parameters"]
                    or Path(data["path"]).resolve() != Path(expected["path"]).resolve()
                    or data.get("size", data.get("size_bytes")) != expected["fingerprint"]["size_bytes"]
                    or data["mtime_ns"] != expected["fingerprint"]["mtime_ns"]):
                raise ValueError("training report differs from frozen declaration")
            artifact(training["sequence_model_path"], training["sequence_model_sha256"])
            if receipt["status"] == "failed":
                return {"candidate": candidate, "status": "failed", "eligible_for_performance": False}
            check(receipt["report"])
            report = read(receipt["report"]["path"])
            rows = report["per_sequence"]
            if len(rows) != len(groups) or {row["row_group"] for row in rows} != set(groups):
                raise ValueError("screen report groups differ")
            for row in rows:
                if row["eligible"] != controls[row["row_group"]]["eligible"] or (row["eligible"] and not close(row["weighted_pearson"], (row["t0"] + row["t1"])/2)):
                    raise ValueError("screen eligibility or row scores differ")
            wp = sum(row["weighted_pearson"] for row in rows if row["eligible"]) / sum(row["eligible"] for row in rows)
            provenance = report["provenance"]
            valid = manifest["datasets"]["valid"]
            if (not close(wp, report["weighted_pearson"]) or not close(wp, receipt["weighted_pearson"])
                    or not close(baseline_wp, receipt["incumbent_wp"]) or not close(wp-baseline_wp, receipt["delta"])
                    or report["metric_clip"] != [-2., 2.] or report["postprocess"] != spec["config"].get("postprocess", {})
                    or provenance["model_sha256"] != refs["model"]["sha256"]
                    or provenance["bulk_model_sha256"] != training["sequence_model_sha256"]
                    or Path(provenance["dataset_path"]).resolve() != Path(valid["path"]).resolve()
                    or provenance["dataset_size"] != valid["fingerprint"]["size_bytes"]):
                raise ValueError("screen score/provenance differs")
            comparison_path = Path(spec["config_path"]).parent / "screen_comparison.json"
            artifact(comparison_path)
            comparison = read(comparison_path)
            if (not close(comparison["candidate_wp"], wp) or not close(comparison["baseline_wp"], baseline_wp)
                    or not close(comparison["delta"], wp-baseline_wp)
                    or comparison["confidence"] != .95 or comparison["n_resamples"] != 10000 or comparison["seed"] != 20260913
                    or comparison["sequences"] != sum(row["eligible"] for row in rows)
                    or not all(math.isfinite(comparison[k]) for k in ("ci_low", "ci_high")) or comparison["ci_low"] > comparison["ci_high"]):
                raise ValueError("paired comparison identity differs from incumbent/receipt")
            return {"candidate": candidate, "status": "passed", "screen_wp": wp,
                    "model": refs["model"], "comparison": comparison}

        prior_name = policy["selected_screen_winner"]["candidate"]
        prior = screen(prior_name)
        if prior["status"] != "passed" or not close(prior["screen_wp"], policy["selected_screen_winner"]["screen_wp"]):
            raise ValueError("prior selected winner differs from policy")
        rows = [screen(name, expansion=True) for name in ARMS]
        for row in rows:
            row["eligible_for_performance"] = row["status"] == "passed" and row["screen_wp"] > max(prior["screen_wp"], baseline_wp)
        eligible = [row for row in rows if row["eligible_for_performance"]]
        winner = max(eligible, key=lambda row: (row["screen_wp"], row["candidate"])) if eligible else None
        artifact(__file__)
        result = {"version": "correlation_expansion_selection.v1", "created_at": datetime.now(timezone.utc).isoformat(),
                  "state": "selected_for_performance_not_dev_approved" if winner else "no_candidate",
                  "winner": winner, "candidates": rows, "prior_256_winner": prior,
                  "selection_rule": "highest screen WP strictly above incumbent and prior 256 winner; ties use candidate id",
                  "caps": CAPS, "consumed": used, "registry": registry_ref, "registry_snapshot": registry,
                  "policy": policy_ref, "evidence": list(evidence.values()),
                  "release_feasibility": "pending Linux complete-callback performance gate",
                  "dev_approved": False, "registry_mutated": False,
                  "uncertainty": "screen-selected; comparison intervals do not correct search bias; confirmation remains consumed"}
        # Recheck all evidence before an exclusive write; current registry is locked.
        for ref in list(evidence.values()):
            check(ref)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(f".{output_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output_path)  # Atomic publish; an existing receipt wins.
        finally:
            temporary.unlink(missing_ok=True)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=ROUND / "registry.json")
    parser.add_argument("--policy", type=Path, default=ROUND / "correlation_expansion_policy.json")
    args = parser.parse_args()
    print(select(args.output, registry_path=args.registry, policy_path=args.policy)["state"])
