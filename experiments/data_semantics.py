"""Check our modelling assumptions against what the data overview says the columns mean.

The overview (wundernn.io/connectome/docs/data_overview) says t0 and t1 are
undisclosed indicators of the future price movement of i0, and the features are
two L11 order books plus trades plus a0-a7, with column indices that are
identifiers rather than depth levels. This script measures:

  1. target structure: per-sequence grid spacing ("tick classes") and sign
     constraints between t0 and t1;
  2. whether targets are a linear function of future changes in the i0 price block;
  3. what predicts the hidden scoring mask: book/trade events around t, a0-a7,
     or the future target itself;
  4. how the current model (correlation1024, cached development predictions)
     and the shipped clip2 calibrator score inside each tick class, and how
     much the pooled score moves when one class improves;
  5. the class mix of every TRAIN and validation sequence, and the dev score
     after reweighting sequences to TRAIN's mix;
  6. whether the class can be inferred from features without targets;
  7. whether zeroing predictions that break `never both positive` helps.

    .venv/bin/python -m experiments.data_semantics
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from experiments.devdata import (DEFAULT_CACHE, DEFAULT_VALID, WARMUP, global_wp, load_predictions, load_targets,
                                 paired_bootstrap, sequence_statistics)
from experiments.output_calibration import apply
from utils import FEATURE_COLUMNS

CALIBRATOR = Path("artifacts/calibrated/correlation1024_mlp64_clip2/calibrator.npz")
OOF_STATISTICS = Path("artifacts/experiments/output_calibration/sequence_statistics_clip2.npz")
HORIZONS = (1, 2, 5, 10, 20, 50, 100, 200, 400, 1000, 2000)


def spacing(y):
    """Smallest non-zero |target| in a sequence: every target is an integer multiple of it."""
    nonzero = np.abs(y[y != 0])
    return float(nonzero.min()) if len(nonzero) else float("nan")


def auc(score, label):
    """Rank AUC with tied scores sharing their average rank."""
    _, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inverse]
    positives = label.sum()
    return float((ranks[label].sum() - positives * (positives + 1) / 2) / (positives * (len(label) - positives)))


def read_sequence(parquet, group, columns=None):
    table = parquet.read_row_group(group, columns=columns)
    features = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]) if columns is None else None
    return table, features


def target_structure(targets, scored, classes):
    need = np.zeros_like(scored)
    need[:, WARMUP:] = True
    y = targets[need]
    total = y.sum(1)
    units = np.round(y / np.repeat(classes, need.sum(1))[:, None]).astype(int)
    cells, counts = np.unique(np.clip(units, -4, 4), axis=0, return_counts=True)
    top = np.argsort(-counts)[:8]
    return {
        "both_positive": float(((y[:, 0] > 0) & (y[:, 1] > 0)).mean()),
        "both_zero": float((y == 0).all(1).mean()),
        "exactly_one_zero": float(((y == 0).sum(1) == 1).mean()),
        "sum_positive": float((total > 0).mean()),
        "positive_share": (y > 0).mean(0).tolist(),
        "top_cells_in_grid_units": {f"({a:+d},{b:+d})": round(float(c / len(y)), 4) for (a, b), c in
                                    zip(cells[top], counts[top])},
    }


def linear_future_check(parquet, groups):
    """R^2 of each target on the change of the 22 i0 price columns over h rows, per sequence."""
    r2 = np.zeros((len(groups), len(HORIZONS), 2))
    for i, group in enumerate(groups):
        table, x = read_sequence(parquet, group)
        y = np.c_[table["t0"].to_numpy(), table["t1"].to_numpy()].astype(np.float64)
        for j, h in enumerate(HORIZONS):
            design = np.c_[x[h:, :22] - x[:-h, :22], np.ones(len(x) - h)]
            residual = y[:-h] - design @ np.linalg.lstsq(design, y[:-h], rcond=None)[0]
            r2[i, j] = 1 - residual.var(0) / y[:-h].var(0)
    return {"sequences": len(groups), "horizons": list(HORIZONS),
            "median_r2": np.median(r2, 0).round(4).T.tolist(), "max_r2": r2.max(0).round(4).T.tolist()}


def mask_predictors(parquet, groups):
    """AUC for is_scored of event indicators around t, a0-a7 and (for reference) the future target."""
    columns, label, runs = {}, [], []
    for group in groups:
        table, x = read_sequence(parquet, group)
        s = table["is_scored"].to_numpy()
        y = np.c_[table["t0"].to_numpy(), table["t1"].to_numpy()]
        change = np.vstack([np.zeros((1, 112), bool), np.diff(x, axis=0) != 0])
        events = change[:, :52].sum(1)
        values = {"i0 book changed at t": change[:, :44].any(1), "i0 trades changed at t": change[:, 44:52].any(1),
                  "i1 book changed at t": change[:, 52:96].any(1), "i1 trades changed at t": change[:, 96:104].any(1)}
        for k in (-1, 0, 1, 2, 5):
            values[f"i0 columns changed at t{k:+d}"] = np.roll(events, -k)
        values.update({f"a{j}": x[:, 104 + j] for j in range(8)})
        values["future |t0|+|t1| (not available)"] = np.abs(y).sum(1)
        values["future t0 or t1 == 0 (not available)"] = (y == 0).any(1)
        keep = slice(WARMUP + 1, len(x) - 5)
        for name, value in values.items():
            columns.setdefault(name, []).append(np.asarray(value, np.float64)[keep])
        label.append(s[keep])
        edges = np.flatnonzero(np.diff(np.r_[0, s.astype(int), 0]))
        runs.append(edges[1::2] - edges[::2])
    label, runs = np.concatenate(label), np.concatenate(runs)
    result = {}
    for name, value in columns.items():
        a = auc(np.concatenate(value), label)
        result[name] = {"auc": round(max(a, 1 - a), 4), "direction": "+" if a >= .5 else "-"}
    return {"sequences": len(groups), "scored_fraction": float(label.mean()), "auc": result,
            "run_length": {"median": float(np.median(runs)), "p90": float(np.percentile(runs, 90)),
                           "share_single_rows": float((runs == 1).mean())}}


def label_classes(path):
    parquet = pq.ParquetFile(path)
    return np.array([round(spacing(np.c_[t["t0"].to_numpy(), t["t1"].to_numpy()]), 4) for t in
                     (parquet.read_row_group(g, columns=["t0", "t1"]) for g in range(parquet.metadata.num_row_groups))])


def class_mix(train, valid, dev_classes, raw_stats, oof_stats):
    """Class shares per split, and dev WP after reweighting sequences to another split's class mix."""
    classes = np.unique(dev_classes)

    def share(labels):
        return np.array([(labels == c).mean() for c in classes])

    def reweighted(stats, target):
        ratio = (target / share(dev_classes))[np.searchsorted(classes, dev_classes)]
        stats = stats.copy()
        stats[..., [0, 3, 4, 5]] *= ratio[:, None, None]  # weight sums and centred moments scale; means do not
        return global_wp(stats)

    shares = {"train": share(train), "valid": share(valid), "dev": share(dev_classes)}
    return {"classes": classes.tolist(), "sequences": {"train": len(train), "valid": len(valid)},
            "share": {k: v.round(4).tolist() for k, v in shares.items()},
            "dev_wp_at_mix": {k: {"raw": reweighted(raw_stats, v), "calibrated_out_of_fold": reweighted(oof_stats, v)}
                              for k, v in shares.items()}}


def class_identifiability(parquet, groups, classes):
    """Leave-one-out nearest-centroid accuracy of the tick class from simple feature summaries."""
    summaries = {"warm-up (99 rows)": [], "first 1,000 rows": []}
    for group in groups:
        _, x = read_sequence(parquet, group)
        for name, rows in (("warm-up (99 rows)", x[:99]), ("first 1,000 rows", x[:1000])):
            d = np.diff(rows.astype(np.float64), axis=0)
            moved = np.abs(d[:, :22][d[:, :22] != 0])
            summaries[name].append([(d[:, :44] != 0).any(1).mean(), np.median(moved) if len(moved) else 0,
                                    *rows[:, 104:112].mean(0), rows[:, :22].std(0).mean()])
    out = {"sequences": len(groups), "majority_share": float(max(np.unique(classes, return_counts=True)[1]) / len(classes))}
    for name, rows in summaries.items():
        z = np.asarray(rows)
        z = (z - z.mean(0)) / (z.std(0) + 1e-9)
        hits = 0
        for i in range(len(z)):
            others = np.arange(len(z)) != i
            centroids = {c: z[others & (classes == c)].mean(0) for c in np.unique(classes)}
            hits += min(centroids, key=lambda c: ((z[i] - centroids[c]) ** 2).sum()) == classes[i]
        out[name] = {"accuracy": hits / len(z),
                     "a7_median_by_class": {f"{c:.4f}": float(np.median(np.asarray(rows)[classes == c, 9]))
                                            for c in np.unique(classes)}}
    return out


def sign_projection(targets, scored, predictions, calibrated):
    """Parameter-free use of `t0, t1 never both positive`: when both predictions are positive, zero one or both."""
    out = {}
    for name, base in (("raw", predictions), ("calibrated_in_sample", calibrated)):
        reference = sequence_statistics(targets, base, scored)
        both = (base[..., 0] > 0) & (base[..., 1] > 0)
        smaller = np.argmin(base, -1)
        for mode in ("zero_smaller", "zero_both"):
            projected = base.copy()
            for k in range(2):
                projected[..., k] = np.where(both & ((smaller == k) | (mode == "zero_both")), 0, base[..., k])
            out[f"{name}_{mode}"] = paired_bootstrap(sequence_statistics(targets, projected, scored), reference)
    return out


def class_scores(targets, scored, predictions, calibrated, oof, classes):
    raw_stats = sequence_statistics(targets, predictions, scored)
    cal_stats = sequence_statistics(targets, calibrated, scored)
    out = {"pooled": {"raw": global_wp(raw_stats), "calibrated_in_sample": global_wp(cal_stats),
                      "calibrated_out_of_fold": global_wp(oof["mlp64_scored_30epochs_clip2__scored"])}}
    for c in np.unique(classes):
        m = classes == c
        y, p, q = targets[m][scored[m]], predictions[m][scored[m]], calibrated[m][scored[m]]
        w = np.abs(np.clip(y, -2, 2))
        out[f"{c:.4f}"] = {
            "sequences": int(m.sum()), "share_of_scored_rows": float(scored[m].sum() / scored.sum()),
            "scored_fraction": float(scored[m].sum() / (m.sum() * (20_000 - WARMUP))),
            "wp_within_raw": global_wp(raw_stats[m]),
            "wp_within_calibrated_out_of_fold": global_wp(oof["mlp64_scored_30epochs_clip2__scored"][m]),
            "oof_gain_within": global_wp(oof["mlp64_scored_30epochs_clip2__scored"][m]) - global_wp(oof["raw__scored"][m]),
            "weighted_mean_target": ((w * np.clip(y, -2, 2)).sum(0) / w.sum(0)).tolist(),
            "weighted_mean_raw_prediction": ((w * np.clip(p, -2, 2)).sum(0) / w.sum(0)).tolist(),
            "std_target": y.std(0).tolist(), "std_raw_prediction": p.std(0).tolist(),
            "both_predictions_positive_raw": float(((p[:, 0] > 0) & (p[:, 1] > 0)).mean()),
            "both_predictions_positive_calibrated": float(((q[:, 0] > 0) & (q[:, 1] > 0)).mean()),
        }
    return out


def level_predictors(targets, scored, classes):
    """Pooled WP of predictors that know only a level: per tick class, or (oracle) per sequence."""
    clipped = np.clip(targets, -2, 2)
    w = np.abs(clipped) * scored[..., None]
    per_sequence = (w * clipped).sum(1) / np.maximum(w.sum(1), 1e-9)
    per_class = np.zeros_like(per_sequence)
    for c in np.unique(classes):
        m = classes == c
        per_class[m] = (w[m] * clipped[m]).sum((0, 1)) / w[m].sum((0, 1))
    return {name: global_wp(sequence_statistics(targets, np.broadcast_to(level[:, None, :], targets.shape), scored))
            for name, level in (("class_level_only", per_class), ("sequence_level_only_oracle", per_sequence))}


def sensitivity(targets, scored, predictions, classes, lift=0.02):
    """Pooled WP change when one class's within-class WP rises by `lift` (blend toward its targets)."""
    base_stats = sequence_statistics(targets, predictions, scored)
    base = global_wp(base_stats)
    out = {}
    for c in np.unique(classes):
        m = np.nonzero(classes == c)[0]
        goal = global_wp(base_stats[m]) + lift
        low, high = 0.0, 1.0
        for _ in range(20):
            alpha = (low + high) / 2
            blended = (1 - alpha) * predictions[m] + alpha * np.clip(targets[m], -2, 2)
            stats = sequence_statistics(targets[m], blended, scored[m])
            low, high = (alpha, high) if global_wp(stats) < goal else (low, alpha)
        merged = base_stats.copy()
        merged[m] = stats
        out[f"{c:.4f}"] = {"alpha": alpha, "pooled_delta": global_wp(merged) - base}
    return {"within_class_lift": lift, "pooled_delta_by_class": out}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--train", default="datasets/train.parquet")
    parser.add_argument("--output", default="artifacts/experiments/data_semantics")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.time()

    def log(message):
        print(f"[{time.time() - start:5.0f}s] {message}", flush=True)

    predictions, groups = load_predictions(args.cache)
    predictions = np.asarray(predictions, np.float32)
    targets, scored = load_targets(groups, args.valid, store="artifacts/experiments/dev_targets.npz")
    classes = np.array([spacing(t) for t in targets])
    classes = np.round(classes, 4)
    params = {k: v for k, v in np.load(CALIBRATOR).items()}
    params.update(kind="mlp", clip=float(params["clip"]))
    calibrated = apply(params, predictions.reshape(-1, 2)).reshape(predictions.shape)
    oof = dict(np.load(OOF_STATISTICS))
    parquet = pq.ParquetFile(args.valid)
    rng = np.random.default_rng(20261025)
    report = {"development_sequences": len(groups),
              "classes": dict(zip(*[x.tolist() for x in np.unique(classes, return_counts=True)]))}
    log(f"loaded {len(groups)} development sequences; classes {report['classes']}")
    report["target_structure"] = target_structure(targets, scored, classes)
    log(f"target structure {json.dumps(report['target_structure'])}")
    labels = {"train": label_classes(args.train), "valid": label_classes(args.valid)}
    np.savez(output / "class_labels.npz", **labels)
    report["class_mix"] = class_mix(labels["train"], labels["valid"], classes,
                                    sequence_statistics(targets, predictions, scored),
                                    oof["mlp64_scored_30epochs_clip2__scored"])
    log(f"class mix {json.dumps(report['class_mix'], indent=1)}")
    pick = rng.choice(len(groups), 300, replace=False)
    report["class_identifiability"] = class_identifiability(parquet, [groups[i] for i in pick], classes[pick])
    log(f"class identifiability {json.dumps(report['class_identifiability'], indent=1)}")
    report["sign_projection"] = sign_projection(targets, scored, predictions, calibrated)
    log(f"sign projection {json.dumps(report['sign_projection'], indent=1)}")
    report["linear_future_check"] = linear_future_check(parquet, [groups[i] for i in rng.choice(len(groups), 12, replace=False)])
    log(f"linear future check {report['linear_future_check']}")
    report["mask_predictors"] = mask_predictors(parquet, [groups[i] for i in rng.choice(len(groups), 40, replace=False)])
    log(f"mask predictors {json.dumps(report['mask_predictors'], indent=1)}")
    report["class_scores"] = class_scores(targets, scored, predictions, calibrated, oof, classes)
    log(f"class scores {json.dumps(report['class_scores'], indent=1)}")
    report["level_predictors"] = level_predictors(targets, scored, classes)
    log(f"level predictors {report['level_predictors']}")
    report["sensitivity"] = sensitivity(targets, scored, predictions, classes)
    log(f"sensitivity {report['sensitivity']}")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(f"saved {output / 'report.json'}")


if __name__ == "__main__":
    main()
