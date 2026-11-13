"""Does the dev-to-public class-mix shift explain why only ~64% of dev gains reach public?

Three checks on cached dev artifacts (no training, no confirmation data):

1. Reweight each submission's per-sequence OOF statistics from the dev class mix
   toward the TRAIN class mix. Fit the one mixing parameter on the raw
   `correlation1024` public score, then predict the other four public scores.
2. Paired sequence bootstrap of the 3-model vs 2-model GPU ensemble.
3. Prediction correlation and raw-blend ablations for the three ensemble members.

Run: .venv/bin/python -m experiments.transfer_check
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from connectome.global_metric import merge_statistics, score_statistics

CLASSES = np.array([0.1254, 0.2413, 0.4685, 0.6569])
ENS = Path("artifacts/ensemble")
OUT = Path("artifacts/experiments/transfer_check/results.json")


def wp(stats: np.ndarray, weights: np.ndarray | None = None) -> float:
    s = stats.copy()
    if weights is not None:
        for j in (0, 3, 4, 5):  # sum_w and the centered second moments scale with the weight
            s[..., j] = s[..., j] * weights[:, None]
    return float(score_statistics(merge_statistics(s)).mean())


def pooled_wp(p: np.ndarray, y: np.ndarray) -> float:
    w = np.abs(y)
    out = []
    for t in range(2):
        ww, pp, yy = w[:, t], p[:, t], y[:, t]
        mp, my = (ww * pp).sum() / ww.sum(), (ww * yy).sum() / ww.sum()
        cov = (ww * (pp - mp) * (yy - my)).sum()
        out.append(cov / np.sqrt((ww * (pp - mp) ** 2).sum() * (ww * (yy - my) ** 2).sum()))
    return float(np.mean(out))


def main() -> None:
    labels = np.load("artifacts/experiments/data_semantics/class_labels.npz")
    cls = lambda a: np.argmin(np.abs(a[:, None] - CLASSES[None]), axis=1)
    dev_cls, train_cls = cls(labels["dev"]), cls(labels["train"])
    f_dev = np.bincount(dev_cls, minlength=4) / len(dev_cls)
    f_train = np.bincount(train_cls, minlength=4) / len(train_cls)
    to_train = (f_train / f_dev)[dev_cls]
    mix = lambda a: (1.0 - a) + a * to_train

    cal = np.load("artifacts/experiments/output_calibration/sequence_statistics_clip2.npz")
    subs = {
        "raw correlation1024": (cal["raw__scored"], 0.65151),
        "mlp64_clip2 1H6GT8N9": (cal["mlp64_scored_30epochs_clip2__scored"], 0.65288),
        "cpu 2-model 22Z98Z0W": (np.load(ENS / "corr1024_sgru_v3_snap40/oof_sequence_statistics.npy"), 0.65533),
        "gpu 2-model 2QU7CNP2": (np.load(ENS / "gpu_2model_gru160_snap40/oof_sequence_statistics.npy"), 0.65662),
        "gpu 3-model 22GU0AJ2": (np.load(ENS / "gpu_3model_gru160_lstm160_snap40/oof_sequence_statistics.npy"), 0.65694),
    }
    raw, raw_public = subs["raw correlation1024"]

    lo, hi = 0.0, 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if wp(raw, mix(mid)) > raw_public else (lo, mid)
    alpha = lo
    result = {"dev_mix": f_dev.round(4).tolist(), "train_mix": f_train.round(4).tolist(),
              "alpha_fit_on_raw": round(alpha, 4), "submissions": {}}

    print(f"dev class mix   {f_dev.round(4)}\ntrain class mix {f_train.round(4)}")
    print(f"alpha fitted on raw correlation1024 = {alpha:.3f} (0 = dev mix, 1 = TRAIN mix)\n")
    print(f"{'submission':22s} {'dev':>8s} {'mix@a':>8s} {'public':>8s} | {'d_dev':>8s} {'d_mix':>8s} {'d_trainmix':>10s} {'d_public':>8s}")
    base_dev, base_mix, base_train = wp(raw), wp(raw, mix(alpha)), wp(raw, to_train)
    for name, (st, public) in subs.items():
        row = {"dev": wp(st), "mix_at_alpha": wp(st, mix(alpha)), "train_mix": wp(st, to_train), "public": public}
        row.update(d_dev=row["dev"] - base_dev, d_mix=row["mix_at_alpha"] - base_mix,
                   d_train_mix=row["train_mix"] - base_train, d_public=public - raw_public)
        result["submissions"][name] = {k: round(v, 5) for k, v in row.items()}
        print(f"{name:22s} {row['dev']:8.5f} {row['mix_at_alpha']:8.5f} {public:8.5f} | "
              f"{row['d_dev']:+8.5f} {row['d_mix']:+8.5f} {row['d_train_mix']:+10.5f} {row['d_public']:+8.5f}")

    rng = np.random.default_rng(20261113)
    g3, g2 = subs["gpu 3-model 22GU0AJ2"][0], subs["gpu 2-model 2QU7CNP2"][0]
    deltas, levels = [], []
    for _ in range(4000):
        i = rng.integers(0, len(g3), len(g3))
        a = wp(g3[i])
        deltas.append(a - wp(g2[i]))
        levels.append(a)
    deltas = np.array(deltas)
    ci = np.percentile(deltas, [2.5, 97.5])
    result["gpu3_minus_gpu2"] = {"mean": round(float(deltas.mean()), 5), "ci95": ci.round(5).tolist(),
                                 "share_not_positive": round(float((deltas <= 0).mean()), 4)}
    result["level_sd_1400_sequences"] = round(float(np.std(levels)), 5)
    print(f"\n3-model minus 2-model on dev: {deltas.mean():+.5f}, 95% CI [{ci[0]:+.5f}, {ci[1]:+.5f}], "
          f"share <= 0: {(deltas <= 0).mean():.3f}")
    print(f"sd of the WP level over resampled 1,400-sequence sets: {np.std(levels):.5f}")

    dt = np.load("artifacts/experiments/dev_targets.npz")
    scored = dt["scored"].astype(bool)
    y = np.clip(dt["targets"][scored], -2, 2).astype(np.float64)
    load = lambda p: np.clip(np.load(p, mmap_mode="r")[scored], -2, 2).astype(np.float64)
    corr = load("artifacts/global_wp/rescore_v1/correlation1024_seed20260912.cache/predictions.npy")
    gru = load("outputs/sgru_h160_10ep/dev_predictions.npy")
    lstm = load("outputs/slstm_h160_10ep/dev_predictions.npy")
    pairs = {"corr1024~gru": (corr, gru), "corr1024~lstm": (corr, lstm), "gru~lstm": (gru, lstm)}
    result["prediction_correlation"] = {
        k: round(float(np.mean([np.corrcoef(a[:, t], b[:, t])[0, 1] for t in range(2)])), 4) for k, (a, b) in pairs.items()}
    blends = {
        "corr1024": corr, "gru": gru, "lstm": lstm,
        "gru+lstm 50/50 (no corr1024)": 0.5 * gru + 0.5 * lstm,
        "corr1024+gru (shipped 2-model weights)": 0.2663 * corr + 0.7337 * gru,
        "all three (shipped 3-model weights)": 0.2137 * corr + 0.4331 * gru + 0.3532 * lstm,
    }
    result["raw_blend_dev_wp"] = {k: round(pooled_wp(v, y), 5) for k, v in blends.items()}
    print("\nprediction correlation on scored rows:", result["prediction_correlation"])
    print("raw (uncalibrated) dev WP:")
    for k, v in result["raw_blend_dev_wp"].items():
        print(f"  {k:40s} {v:.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
