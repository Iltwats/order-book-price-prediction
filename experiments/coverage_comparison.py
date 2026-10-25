"""Compare v2.1 coverage/repeat arms with the incumbent, raw and calibrated.

Every model, the incumbent included, gets the same calibration as the shipped
submission: the residual 64-unit network on its two outputs, inputs clipped to
+/-2, 30 epochs, fitted on scored rows and cross-fitted with the shared dev
folds (`experiments.output_calibration`, arm `mlp64_scored_30epochs_clip2`).

Reported for each model: pooled WP, each target, each tick class and the
256-sequence screen, raw and calibrated; how far its raw outputs moved from the
incumbent's; and paired sequence-bootstrap differences
- against the incumbent (raw vs raw, calibrated vs calibrated), and
- coverage4096 against repeat1024 of the same seed.

    .venv/bin/python -m experiments.coverage_comparison \
        --model coverage_s0=artifacts/gpu/cpu-v2.1/coverage4096/dev_cache ...
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from connectome.global_metric import merge_statistics, score_statistics
from experiments.data_semantics import spacing
from experiments.devdata import (DEFAULT_CACHE, ROOT, folds, global_wp, load_predictions, load_targets,
                                 paired_bootstrap, sequence_statistics)
from experiments.output_calibration import ARMS, apply, fit

CALIBRATOR = "mlp64_scored_30epochs_clip2"


def calibrate(raw, clipped, scored, fold):
    predicted = raw.copy()
    for k in range(5):
        train = fold != k
        params = fit(ARMS[CALIBRATOR], raw[train][scored[train]], clipped[train][scored[train]])
        test = np.nonzero(fold == k)[0]
        predicted[test] = apply(params, raw[test].reshape(-1, 2)).reshape(len(test), 20_000, 2)
    return predicted


def summary(stats, classes, screen):
    per_target = score_statistics(merge_statistics(stats))
    return {"wp": global_wp(stats), "t0": float(per_target[0]), "t1": float(per_target[1]),
            "screen": global_wp(stats[screen]),
            "by_class": {str(c): global_wp(stats[classes == c]) for c in np.unique(classes)}}


def movement(new, old, scored):
    a, b = new[scored], old[scored]
    return {"corr_with_incumbent": [float(np.corrcoef(a[:, j], b[:, j])[0, 1]) for j in range(2)],
            "mean_abs_change": float(np.abs(a - b).mean()),
            "mean_abs_change_over_std": (np.abs(a - b).mean(0) / b.std(0)).tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", action="append", required=True, help="name=cache directory")
    parser.add_argument("--output", default="artifacts/experiments/coverage_comparison")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()

    incumbent, groups = load_predictions(DEFAULT_CACHE)
    incumbent = np.asarray(incumbent, np.float32)
    targets, scored = load_targets(groups, store=ROOT / "artifacts/experiments/dev_targets.npz")
    clipped = np.clip(targets, -2, 2)
    fold = folds(len(groups))
    classes = np.array([round(spacing(t), 4) for t in targets])
    splits = json.loads((ROOT / "artifacts/manifest.json").read_text())["splits"]
    screen = np.isin(groups, splits["screen"])

    models = {"incumbent": incumbent}
    for item in args.model:
        name, path = item.split("=", 1)
        raw, model_groups = load_predictions(path)
        if list(model_groups) != list(groups):
            raise ValueError(f"{name}: cache groups differ from the incumbent's")
        models[name] = np.asarray(raw, np.float32)

    stats, results = {}, {"calibrator": CALIBRATOR, "sequences": len(groups), "models": {}}
    for name, raw in models.items():
        began = time.time()
        calibrated = calibrate(raw, clipped, scored, fold)
        stats[name] = {"raw": sequence_statistics(targets, raw, scored),
                       "calibrated": sequence_statistics(targets, calibrated, scored)}
        entry = {kind: summary(s, classes, screen) for kind, s in stats[name].items()}
        if name != "incumbent":
            entry["movement"] = movement(raw, incumbent, scored)
            entry["vs_incumbent"] = {kind: paired_bootstrap(stats[name][kind], stats["incumbent"][kind])
                                     for kind in ("raw", "calibrated")}
        results["models"][name] = entry
        print(f"{name:14s} raw {entry['raw']['wp']:.5f}  calibrated {entry['calibrated']['wp']:.5f}"
              f"  ({time.time() - began:.0f}s)", flush=True)

    results["coverage_vs_repeat"] = {}
    for name in models:
        if name.startswith("coverage"):
            partner = name.replace("coverage", "repeat")
            if partner in stats:
                results["coverage_vs_repeat"][name] = {kind: paired_bootstrap(stats[name][kind], stats[partner][kind])
                                                       for kind in ("raw", "calibrated")}
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    np.savez(output / "sequence_statistics.npz",
             **{f"{name}__{kind}": s for name, kinds in stats.items() for kind, s in kinds.items()})
    print(json.dumps({k: v for k, v in results.items() if k != "models"}, indent=2))
    print(f"saved {output / 'results.json'} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
