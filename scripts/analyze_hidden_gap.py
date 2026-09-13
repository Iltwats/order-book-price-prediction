"""Describe cached full-development scores; never read datasets or run models."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORTS = {
    "incumbent": ROOT / "artifacts/selected/dev_report.json",
    "affine_rho": ROOT / "artifacts/round2/affine_rho/dev_report.json",
    "correlation256": ROOT / "artifacts/round2/correlation_gru_lr3e-5/dev_report.json",
    "correlation1024": ROOT / "artifacts/round2/correlation_gru1024_seed20260912/dev_report.json",
}
PUBLIC = {
    "incumbent": {"score": .6083, "runtime_seconds": 1366, "memory_mb_displayed": 624},
    "affine_rho": {"score": .6085, "runtime_seconds": 1328, "memory_mb_displayed": 624},
    "correlation256": {"score": .6086, "runtime_seconds": 1391, "memory_mb_displayed": 624},
    "correlation1024": {"score": .6092, "runtime_seconds": 1375, "memory_mb_displayed": 624},
}


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def describe(values):
    values = np.asarray(values, np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("nonempty finite one-dimensional cached scores required")
    quantiles = np.quantile(values, [0, .01, .05, .1, .25, .5, .75, .9, .95, .99, 1], method="linear")
    bottom, top = values <= quantiles[4], values >= quantiles[6]
    return {"count": len(values), "mean": float(values.mean()), "std_population": float(values.std(ddof=0)),
        "quantiles_linear": dict(zip(("min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"), map(float, quantiles))),
        "negative_count": int((values < 0).sum()), "zero_count": int((values == 0).sum()),
        "bottom_quartile_mean": float(values[bottom].mean()), "bottom_quartile_count": int(bottom.sum()),
        "top_quartile_mean": float(values[top].mean()), "top_quartile_count": int(top.sum())}


def change(values):
    values = np.asarray(values, np.float64)
    result = describe(values)
    result.update(improved_count=int((values > 0).sum()), worsened_count=int((values < 0).sum()),
                  unchanged_count=int((values == 0).sum()), improved_fraction=float((values > 0).mean()))
    result["magnitude_counts"] = {str(threshold): {
        "gain_above": int((values > threshold).sum()), "loss_below_negative": int((values < -threshold).sum())}
        for threshold in (1e-5, 1e-4, 5e-4, 1e-3, 5e-3)}
    return result


def _cached_rows(report, dev, confirmation):
    rows = report["per_sequence"]
    groups = [row["row_group"] for row in rows]
    if len(groups) != len(set(groups)) or set(groups) != dev or set(groups) & confirmation:
        raise ValueError("cached report must cover full dev exactly and exclude confirmation")
    if len({row["seq_ix"] for row in rows}) != len(rows):
        raise ValueError("cached sequence IDs must be unique")
    for row in rows:
        if row["eligible"]:
            values = np.array([row["t0"], row["t1"], row["weighted_pearson"]], np.float64)
            if (not np.isfinite(values).all() or np.any(np.abs(values) > 1)
                    or not np.isclose(values[:2].mean(), values[2], rtol=0, atol=1e-12)):
                raise ValueError("invalid cached per-target or sequence WP")
    return {row["row_group"]: row for row in rows}


def analyze(report_paths, public, manifest_path):
    if "incumbent" not in report_paths or set(report_paths) != set(public):
        raise ValueError("an incumbent and matching public labels are required")
    manifest = read(manifest_path)
    dev, confirmation = map(set, (manifest["splits"]["dev"], manifest["splits"]["confirmation"]))
    if dev & confirmation:
        raise ValueError("manifest dev/confirmation overlap")
    reports = {name: read(path) for name, path in report_paths.items()}
    rows = {name: _cached_rows(report, dev, confirmation) for name, report in reports.items()}
    baseline = rows["incumbent"]
    # IDs only join existing scores. Quartile membership uses baseline WP alone.
    order = [g for g in manifest["splits"]["dev"] if baseline[g]["eligible"]]
    arrays = {}
    for name, cached in rows.items():
        if any((cached[g]["seq_ix"], cached[g]["eligible"]) != (baseline[g]["seq_ix"], baseline[g]["eligible"]) for g in dev):
            raise ValueError("candidate sequence identity/eligibility differs from incumbent")
        arrays[name] = {key: np.array([cached[g][key] for g in order], np.float64)
                        for key in ("weighted_pearson", "t0", "t1")}
        for key, values in arrays[name].items():
            if not np.isclose(values.mean(), reports[name][key], rtol=0, atol=1e-12):
                raise ValueError("reported aggregate differs from cached sequence means")
    cuts = np.quantile(arrays["incumbent"]["weighted_pearson"], [.25, .5, .75], method="linear")
    membership = np.searchsorted(cuts, arrays["incumbent"]["weighted_pearson"], side="right")
    distributions = {name: {key: describe(values) for key, values in target.items()} for name, target in arrays.items()}
    pairs = {}
    for name, target in arrays.items():
        if name == "incumbent":
            continue
        delta = {key: values-arrays["incumbent"][key] for key, values in target.items()}
        quarters = []
        for quarter in range(4):
            mask = membership == quarter
            if not mask.any():
                raise ValueError("empty baseline quartile; tied-score partition needs separate design")
            quarters.append({"baseline_quartile": quarter+1, "count": int(mask.sum()),
                "baseline_wp_mean": float(arrays["incumbent"]["weighted_pearson"][mask].mean()),
                "candidate_wp_mean": float(target["weighted_pearson"][mask].mean()),
                "contribution_to_overall_mean_delta": float(delta["weighted_pearson"][mask].sum()/len(order)),
                "changes": {key: change(values[mask]) for key, values in delta.items()}})
        pairs[name] = {"changes": {key: change(values) for key, values in delta.items()}, "by_baseline_wp_quartile": quarters}
    public_rows = []
    for name, source in public.items():
        score = float(source["score"])
        if not np.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("finite public score required")
        local = distributions[name]["weighted_pearson"]["mean"]
        local_gain = local-distributions["incumbent"]["weighted_pearson"]["mean"]
        public_gain = score-public["incumbent"]["score"]
        public_rows.append({"model": name, **source, "local_dev_wp": local,
            "local_minus_public": local-score, "local_gain_over_incumbent": local_gain,
            "public_gain_over_incumbent": public_gain, "public_minus_local_gain": public_gain-local_gain})
    offsets = np.array([row["local_minus_public"] for row in public_rows])
    return {"kind": "cached_dev_and_user_reported_public_gap_diagnostic", "new_model_inference": False,
        "new_dataset_rows_read": 0, "confirmation_used": False, "registry_changed": False,
        "sources": {"reports": {name: artifact(path) for name, path in report_paths.items()},
                    "manifest": artifact(manifest_path), "analysis_script": artifact(__file__)},
        "method": {"std": "population, ddof=0", "quantiles": "NumPy linear",
            "own_quartiles": "at/below own p25 and at/above own p75; ties retained and counts reported",
            "paired_quartiles": "incumbent WP p25/p50/p75; ties at a cut enter the higher quartile",
            "grouping": "descriptive baseline-score strata on already inspected dev; sequence IDs only join records",
            "public_provenance": "four displayed scores/runtime/memory values relayed by user through parent task; four score decimals"},
        "coverage": {"full_dev_sequences": len(dev), "eligible_sequences": len(order), "excluded_sequences": len(dev)-len(order)},
        "baseline_quartile_cuts": cuts.tolist(), "distributions": distributions, "paired_vs_incumbent": pairs,
        "public_comparison": public_rows,
        "offset_summary": {"mean": float(offsets.mean()), "std_population": float(offsets.std()),
                           "min": float(offsets.min()), "max": float(offsets.max()), "range": float(np.ptp(offsets))},
        "limitations": [
            "Public test size, sequence membership, eligibility and scoring-mask distribution are unknown.",
            "Four closely related models evaluated on shared datasets are not four independent estimates of domain shift.",
            "A near-constant offset is compatible with a shared domain/mask difference but does not identify its cause or permit a validated score correction.",
            "Public scores have four displayed decimals; small differences are affected by display precision.",
            "Development strata were already inspected and selected against; none is a new holdout or confirmation set.",
            "Descriptive paired improvements do not guarantee future gains or the live 0.75 target.",
        ]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/round2/hidden_gap_diagnostic.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("preserve existing diagnostic")
    result = analyze(REPORTS, PUBLIC, ROOT / "artifacts/manifest.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"coverage": result["coverage"], "offset_summary": result["offset_summary"],
                      "current_changes": result["paired_vs_incumbent"]["correlation1024"]["changes"]}, indent=2))
