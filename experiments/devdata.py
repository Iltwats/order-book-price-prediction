"""Shared loaders for CPU experiments on the frozen 1,400-sequence development split."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from connectome.global_metric import merge_statistics, score_statistics, sufficient_statistics

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = ROOT / "artifacts/global_wp/rescore_v1/correlation1024_seed20260912.cache"
DEFAULT_VALID = ROOT / "datasets/valid.parquet"
WARMUP = 99
FOLD_SEED = 20261024


def load_predictions(cache_dir=DEFAULT_CACHE):
    """Cached raw [sequences, 20000, 2] predictions and their validation row groups."""
    cache_dir = Path(cache_dir)
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    if not metadata.get("complete"):
        raise ValueError(f"incomplete prediction cache: {cache_dir}")
    return np.load(cache_dir / "predictions.npy", mmap_mode="r"), metadata["groups"]


def load_targets(groups, valid=DEFAULT_VALID, store=None):
    """Targets [sequences, 20000, 2] and the scoring mask with warm-up rows removed."""
    if store is not None and Path(store).exists():
        saved = np.load(store)
        if saved["groups"].tolist() == list(groups):
            return saved["targets"], saved["scored"]
    parquet = pq.ParquetFile(valid)
    targets = np.empty((len(groups), 20_000, 2), np.float32)
    scored = np.empty((len(groups), 20_000), bool)
    for index, group in enumerate(groups):
        table = parquet.read_row_group(group, columns=["t0", "t1", "is_scored"])
        targets[index, :, 0] = table["t0"].to_numpy()
        targets[index, :, 1] = table["t1"].to_numpy()
        scored[index] = table["is_scored"].to_numpy()
    scored[:, :WARMUP] = False
    if store is not None:
        Path(store).parent.mkdir(parents=True, exist_ok=True)
        np.savez(store, groups=np.asarray(groups), targets=targets, scored=scored)
    return targets, scored


def folds(n_sequences, k=5, seed=FOLD_SEED):
    """Sequence-level fold labels shared by all cross-fitted experiments."""
    return np.random.default_rng(seed).permutation(n_sequences) % k


def sequence_statistics(targets, predictions, rows):
    """Per-sequence Global WP moments [sequences, 2, 6] over the selected rows."""
    return np.stack([sufficient_statistics(targets[s][rows[s]], predictions[s][rows[s]])
                     for s in range(len(targets))])


def global_wp(statistics):
    return float(score_statistics(merge_statistics(statistics)).mean())


def paired_bootstrap(candidate, baseline, n_resamples=2000, seed=1):
    """Sequence-cluster percentile interval for a pooled Global WP difference."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(0, n_resamples, 100):
        index = rng.integers(0, len(candidate), size=(100, len(candidate)))
        deltas.append(score_statistics(merge_statistics(candidate[index], axis=1)).mean(-1)
                      - score_statistics(merge_statistics(baseline[index], axis=1)).mean(-1))
    deltas = np.concatenate(deltas)
    return {"delta": global_wp(candidate) - global_wp(baseline),
            "ci_low": float(np.quantile(deltas, .025)), "ci_high": float(np.quantile(deltas, .975))}
