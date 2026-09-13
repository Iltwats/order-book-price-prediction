"""Freeze and declare the final two 1024-sequence correlation-GRU screens."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import correlation_gru_cycle as runner

core, experiments = runner.cycle, runner.experiments
RESULTS = core.ROUND / "correlation_screen_results.json"
SNAPSHOT = core.ROUND / "registry_after_correlation_screen.json"
POLICY = core.ROUND / "correlation_expansion_policy.json"
ARMS = {"correlation_gru1024_seed20260912": 20260912,
        "correlation_gru1024_seed20260913": 20260913}
CAPS = {"screen": 12, "dev": 4}
VERSION = "correlation_gru_1024_expansion.v1"


def read(path):
    return json.loads(Path(path).read_text())


def _capacity(record):
    used = {phase: sum(phase in row["evaluations"] for row in record["candidates"].values()) for phase in CAPS}
    if record["caps"] != CAPS or used["screen"] != 10 or used["dev"] > 3:
        raise ValueError("exactly two final screen slots and at least one dev slot must remain under unchanged caps")
    return used


def _selection(record):
    summary = read(RESULTS)
    rows = summary["candidates"]
    if len(rows) != 2 or {row["candidate"] for row in rows} != set(runner.ARMS):
        raise ValueError("exactly the two completed 256-sequence correlation screens are required")
    if summary["registry_sha256"] != runner.file_hash(SNAPSHOT):
        raise ValueError("screen summary does not match its frozen registry snapshot")
    paths, validated = [RESULTS, SNAPSHOT], []
    for row in rows:
        entry = record["candidates"][row["candidate"]]
        spec = entry["spec"]
        if (spec["parameters"] != runner.correlation_parameters(learning_rate=runner.ARMS[row["candidate"]])
                or spec["training_groups"] != record["default_training_groups"][:256]
                or spec["epochs"] != 1 or spec["seed"] != 20260912
                or spec["parent_model_sha256"] != experiments.INCUMBENT_SHA256):
            raise ValueError("prior screens must match the frozen 256-group objective trials")
        receipt = entry["evaluations"].get("screen", {})
        if receipt.get("status") != "passed":
            raise ValueError("successful frozen screens required before expansion")
        for artifact in [*entry["source_artifacts"], *receipt["artifacts"].values(), receipt["report"]]:
            experiments._check_artifact(artifact)
            paths.append(Path(artifact["path"]))
        directory = Path(entry["spec"]["config_path"]).parent
        comparison_path = directory / "screen_comparison.json"
        comparison, report = read(comparison_path), read(receipt["report"]["path"])
        if (row["comparison"] != comparison or not all(math.isfinite(comparison[key]) and comparison[key] > 0
                for key in ("delta", "ci_low", "ci_high")) or comparison["ci_low"] > comparison["ci_high"]
                or comparison.get("confidence") != .95 or comparison.get("n_resamples") != 10000
                or comparison.get("sequences") != 256
                or not math.isclose(row["wp"], report["weighted_pearson"], rel_tol=0, abs_tol=1e-12)
                or not math.isclose(row["wp"], receipt["weighted_pearson"], rel_tol=0, abs_tol=1e-12)
                or not math.isclose(row["wp"], comparison["candidate_wp"], rel_tol=0, abs_tol=1e-12)):
            raise ValueError("both matching screen means and 95% intervals must improve over the incumbent")
        paths.append(comparison_path)
        validated.append({"candidate": row["candidate"], "screen_wp": row["wp"],
                          "comparison": comparison, "learning_rate": entry["spec"]["parameters"]["learning_rate"]})
    winner = max(validated, key=lambda row: (row["screen_wp"], row["candidate"]))
    if winner["learning_rate"] != 3e-5:
        raise ValueError("observed winner differs from the requested lr3e-5 expansion")
    return winner, validated, paths


def _outputs_absent(record):
    for candidate in ARMS:
        if candidate in record["candidates"] or (core.ROUND / candidate).exists():
            raise ValueError("expansion must be frozen before either candidate is registered or fitted")


def prepare():
    """Freeze one immutable policy; this command performs no fitting or scoring."""
    if POLICY.exists():
        raise FileExistsError("preserve existing expansion policy")
    runner.require_bridge_acceptance(runner.BRIDGE, core.SOURCE)
    record, _ = experiments._load(core.ROUND / "registry.json")
    used = _capacity(record)
    _outputs_absent(record)
    winner, screens, evidence = _selection(record)
    groups = read(core.ANCHOR)["specification"]["groups"]
    if len(groups) != 1024 or len(set(groups)) != 1024 or groups != record["default_training_groups"]:
        raise ValueError("the complete unchanged 1024-group training anchor is required")
    source_hash = runner.file_hash(core.SOURCE)
    if source_hash != experiments.INCUMBENT_SHA256:
        raise ValueError("both expansions must initialize from the selected pretrained source")
    paths = list(dict.fromkeys([*runner.sources(), *map(str, evidence), str(core.SOURCE), str(core.ANCHOR), str(Path(__file__).resolve())]))
    policy = {"version": VERSION, "groups": groups, "arms": ARMS, "epochs": 1, "caps": CAPS,
        "parameters": runner.correlation_parameters(learning_rate=winner["learning_rate"]),
        "selected_screen_winner": winner, "screen_evidence": screens,
        "source": experiments._artifact(core.SOURCE), "teacher_sha256": source_hash,
        "initialization": "each seed starts independently from selected pretrained source; no screened checkpoint continuation",
        "selection_rule": "highest positive prior screen; both prior 95% intervals strictly positive; no expansion outcome inspected",
        "trial_allocation": "two final screens; existing 256 winner may use third dev; at most one best feasible expansion uses final dev",
        "consumed_at_prepare": used,
        "source_artifacts": [experiments._artifact(path) for path in paths]}
    with POLICY.open("x") as stream:
        json.dump(policy, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return policy


def _policy():
    policy = read(POLICY)
    if (policy["version"] != VERSION or policy["arms"] != ARMS or policy["epochs"] != 1 or policy["caps"] != CAPS
            or policy["parameters"] != runner.correlation_parameters(learning_rate=3e-5)
            or policy["groups"] != read(core.ANCHOR)["specification"]["groups"]
            or len(policy["groups"]) != 1024 or len(set(policy["groups"])) != 1024
            or policy["source"]["sha256"] != experiments.INCUMBENT_SHA256
            or Path(policy["source"]["path"]).resolve() != Path(core.SOURCE).resolve()
            or policy["teacher_sha256"] != experiments.INCUMBENT_SHA256):
        raise ValueError("expansion policy differs from the fixed source, objective, seeds or budget")
    for artifact in [policy["source"], *policy["source_artifacts"]]:
        experiments._check_artifact(artifact)
    return policy


def declare():
    policy = _policy()
    runner.require_bridge_acceptance(runner.BRIDGE, core.SOURCE)
    record, _ = experiments._load(core.ROUND / "registry.json")
    _capacity(record)
    specs = {}
    for candidate, seed in ARMS.items():
        directory = core.ROUND / candidate
        specs[candidate] = {"hypothesis": "Expand the selected correlation training configuration to all 1024 anchor sequences; independently test the two predeclared seeds",
            "training_groups": policy["groups"], "seed": seed, "epochs": 1, "parameters": policy["parameters"],
            "parent_model_sha256": experiments.INCUMBENT_SHA256,
            "teacher_lineage": [{"model_sha256": policy["teacher_sha256"], "kind": "unchanged_selected_original_onnx_raw_causal_predictions"}],
            "config": {"model": "model.onnx", "backend": "onnx", "onnx_mode": "iobinding", "strict_inputs": True, "affinity": False, "postprocess": {}},
            "config_path": str(directory / "config.json"), "training_report_path": str(directory / "report.json"),
            "source_paths": [a["path"] for a in policy["source_artifacts"]] + [str(POLICY)]}
    for candidate, spec in specs.items():
        if candidate in record["candidates"]:
            existing = record["candidates"][candidate]
            if existing["spec"] != {k: v for k, v in spec.items() if k != "source_paths"}:
                raise ValueError("existing expansion declaration differs; no overwrite")
            for artifact in existing["source_artifacts"]:
                experiments._check_artifact(artifact)
        elif (core.ROUND / candidate).exists():
            raise ValueError("candidate output directory already exists before declaration")
    for candidate, spec in specs.items():
        if candidate not in record["candidates"]:
            experiments.register_candidate(core.ROUND / "registry.json", candidate, spec)
    return specs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "declare"):
        sub.add_parser(name)
    for name in ("fit", "resume", "screen", "dev", "run"):
        sub.add_parser(name).add_argument("candidate", choices=list(ARMS))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "declare":
        declare()
    elif args.command in {"fit", "resume", "run"}:
        _policy()
        runner.fit(args.candidate, resume=args.command == "resume")
        if args.command == "run":
            runner.evaluate(args.candidate, "screen")
    else:
        _policy()
        runner.evaluate(args.candidate, args.command)
