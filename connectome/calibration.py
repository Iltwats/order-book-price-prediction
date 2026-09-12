"""Small prediction caches, causal calibration, and exact paired evaluation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil

import numpy as np

from connectome.data import iter_labels
from utils import BlockAccumulator, SEQUENCE_LENGTH, WARMUP


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class PredictionCache:
    """Memory-mapped raw float32 outputs; incomplete caches cannot be opened.

    Every warm-up output is required, even though the judge callback returns None
    there. They are needed for postprocessors that update state during warm-up.
    """

    def __init__(self, directory: Path, metadata: dict, predictions: np.ndarray, writable: bool):
        self.directory, self.metadata, self.predictions = directory, metadata, predictions
        self.groups = list(metadata["groups"])
        self.seq_ids = list(metadata["seq_ids"])
        self._index = {group: i for i, group in enumerate(self.groups)}
        self._writable = writable
        self._written = np.zeros(len(self.groups), dtype=bool) if writable else np.ones(len(self.groups), dtype=bool)

    @classmethod
    def create(cls, directory: str | Path, groups, seq_ids, provenance: dict) -> "PredictionCache":
        directory = Path(directory)
        groups, seq_ids = [int(g) for g in groups], [int(s) for s in seq_ids]
        if not groups or len(groups) != len(seq_ids) or len(set(groups)) != len(groups) or len(set(seq_ids)) != len(seq_ids):
            raise ValueError("cache requires matching, nonempty, unique row groups and sequence IDs")
        if directory.exists():
            raise FileExistsError(f"cache directory already exists: {directory}")
        directory.parent.mkdir(parents=True, exist_ok=True)
        size = len(groups) * SEQUENCE_LENGTH * 2 * np.dtype(np.float32).itemsize
        if shutil.disk_usage(directory.parent).free < size + 64 * 1024**2:
            raise OSError(f"insufficient free disk for prediction cache ({size} bytes plus headroom)")
        # Validate provenance serialization before creating any output files.
        json.dumps(provenance)
        directory.mkdir()
        metadata = {"version": 1, "complete": False, "groups": groups, "seq_ids": seq_ids,
                    "shape": [len(groups), SEQUENCE_LENGTH, 2], "dtype": "float32",
                    "raw_warmup_outputs_required": True, "provenance": provenance,
                    "created_at": datetime.now(timezone.utc).isoformat()}
        _atomic_json(directory / "metadata.json", metadata)
        predictions = np.lib.format.open_memmap(directory / "predictions.npy", mode="w+",
                                                dtype=np.float32, shape=tuple(metadata["shape"]))
        return cls(directory, metadata, predictions, True)

    @classmethod
    def open(cls, directory: str | Path) -> "PredictionCache":
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text())
        if metadata.get("version") != 1 or metadata.get("complete") is not True:
            raise ValueError("prediction cache is incomplete or has an unsupported version")
        predictions = np.load(directory / "predictions.npy", mmap_mode="r", allow_pickle=False)
        if predictions.shape != tuple(metadata["shape"]) or predictions.dtype != np.float32:
            raise ValueError("prediction cache shape or dtype does not match metadata")
        if metadata["shape"] != [len(metadata["groups"]), SEQUENCE_LENGTH, 2]:
            raise ValueError("invalid sequence shape in prediction cache")
        if len(metadata["seq_ids"]) != len(metadata["groups"]) or len(set(metadata["groups"])) != len(metadata["groups"]):
            raise ValueError("invalid cache sequence mapping")
        return cls(directory, metadata, predictions, False)

    def write(self, index: int, predictions: np.ndarray) -> None:
        index = int(index)
        if not self._writable:
            raise ValueError("cache is read-only or already finalized")
        if not 0 <= index < len(self.groups):
            raise IndexError("cache index out of range")
        if self._written[index]:
            raise ValueError("cache sequence has already been written")
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.shape != (SEQUENCE_LENGTH, 2) or not np.isfinite(predictions).all():
            raise ValueError("cache requires two finite raw outputs on every row, including warm-up")
        self.predictions[index] = predictions
        self._written[index] = True

    def read_group(self, row_group: int) -> np.ndarray:
        index = self._index[int(row_group)]
        if not self._written[index]:
            raise ValueError("requested cache sequence has not been written")
        return self.predictions[index]

    def finalize(self) -> "PredictionCache":
        if not self._writable or not self._written.all():
            raise ValueError("cannot finalize an incomplete or already finalized cache")
        self.predictions.flush()
        self.metadata["complete"] = True
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        _atomic_json(self.directory / "metadata.json", self.metadata)
        self._writable = False
        return self


@dataclass(frozen=True)
class AffineCalibration:
    # Column-vector convention: output = matrix @ input + bias.
    matrix: list[list[float]]
    bias: list[float]
    ridge: float = 0.01
    shrink: float = 0.25
    source_split: str = "train"
    fit_sequences: int = 0
    fit_rows: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def _pair(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 0:
        result = np.repeat(result, 2)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite scalar or length-2 array")
    return result


def apply_calibration(predictions: np.ndarray, *, scales=(1.0, 1.0),
                      affine: AffineCalibration | dict | None = None,
                      ema_half_life=None, ema_blend=0.0) -> np.ndarray:
    """Scale -> affine -> causal EMA, starting from the first warm-up row.

    Call separately for each complete sequence, never on concatenated sequences.
    Float32 state/output agrees with the intended online postprocessing callback.
    """
    predictions = np.asarray(predictions, dtype=np.float32)
    if predictions.ndim != 2 or predictions.shape[1] != 2 or len(predictions) == 0 or not np.isfinite(predictions).all():
        raise ValueError("expected finite [rows, 2] raw predictions including warm-up")
    transformed = predictions * _pair(scales, "scales").astype(np.float32)
    if affine is not None:
        affine = affine.to_dict() if isinstance(affine, AffineCalibration) else affine
        matrix = np.asarray(affine["matrix"], dtype=np.float32)
        bias = np.asarray(affine["bias"], dtype=np.float32)
        if matrix.shape != (2, 2) or bias.shape != (2,) or not np.isfinite(matrix).all() or not np.isfinite(bias).all():
            raise ValueError("affine requires a finite 2x2 matrix and length-2 bias")
        transformed = transformed @ matrix.T + bias
    blend = _pair(ema_blend, "ema_blend").astype(np.float32)
    if np.any((blend < 0) | (blend > 1)):
        raise ValueError("EMA blend must be between zero and one")
    if not np.any(blend):
        if not np.isfinite(transformed).all():
            raise ValueError("calibration produced nonfinite values")
        return transformed.astype(np.float32, copy=False)
    if ema_half_life is None:
        raise ValueError("positive EMA blend requires an EMA half-life")
    half_life = _pair(ema_half_life, "ema_half_life")
    if np.any(half_life <= 0):
        raise ValueError("EMA half-life must be positive")
    alpha = (-np.expm1(-np.log(2.0) / half_life)).astype(np.float32)
    state = transformed[0].copy()
    output = np.empty_like(transformed)
    for index, value in enumerate(transformed):
        if index:
            state += alpha * (value - state)
        output[index] = (1.0 - blend) * value + blend * state
    if not np.isfinite(output).all():
        raise ValueError("calibration produced nonfinite values")
    return output


def scale_grid(values=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0)) -> list[dict]:
    """Independent per-target scales; includes the identity for the default grid."""
    return [{"scales": list(pair)} for pair in itertools.product(values, repeat=2)]


def ema_grid(half_lives=(2.0, 8.0, 32.0), blends=(0.0, 0.1, 0.25)) -> list[dict]:
    result = [{}] if 0.0 in blends else []
    result += [{"ema_half_life": half, "ema_blend": blend}
               for half, blend in itertools.product(half_lives, blends) if blend != 0.0]
    return result


def fit_affine_ridge(pairs, *, source_split: str, ridge: float = 0.01,
                     shrink: float = 0.25) -> AffineCalibration:
    """Fit a two-output weighted ridge model using TRAIN sequences only.

    `pairs` yields (raw_predictions, raw_targets), both [20000, 2]. Warm-up
    rows are excluded from fitting. Labels are clipped only for this surrogate
    objective. Target-absolute weights are normalized per sequence and target.
    A ridge prior and final shrinkage both pull the transform toward identity.
    The source_split guard supplements, rather than replaces, caller provenance.
    """
    if source_split != "train":
        raise ValueError("affine fitting is permitted on the train split only")
    if not np.isfinite(ridge) or ridge <= 0 or not np.isfinite(shrink) or not 0 <= shrink <= 1:
        raise ValueError("require finite ridge > 0 and shrink in [0, 1]")
    xx, xy = np.zeros((2, 3, 3)), np.zeros((2, 3))
    count = np.zeros(2, dtype=np.int64)
    sequences = rows = 0
    for predictions, targets in pairs:
        predictions, targets = np.asarray(predictions, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        if predictions.shape != (SEQUENCE_LENGTH, 2) or targets.shape != predictions.shape:
            raise ValueError("affine fitting requires complete 20k prediction/target sequence pairs")
        if not np.isfinite(predictions).all() or not np.isfinite(targets).all():
            raise ValueError("nonfinite values in affine training data")
        x = np.column_stack((predictions[WARMUP:].astype(np.float64), np.ones(SEQUENCE_LENGTH - WARMUP)))
        y = np.clip(targets[WARMUP:], -2.0, 2.0).astype(np.float64)
        for side in range(2):
            weights = np.abs(y[:, side])
            total = weights.sum()
            if total < 1e-8:
                continue
            weights /= total
            xx[side] += x.T @ (weights[:, None] * x)
            xy[side] += x.T @ (weights * y[:, side])
            count[side] += 1
        sequences += 1
        rows += SEQUENCE_LENGTH - WARMUP
    if not sequences or np.any(count == 0):
        raise ValueError("insufficient nonzero target weight for affine fitting")
    prior = np.column_stack((np.eye(2), np.zeros(2)))
    coefficients = np.empty((2, 3), dtype=np.float64)
    penalty = ridge * np.eye(3)
    for side in range(2):
        coefficients[side] = np.linalg.solve(xx[side] / count[side] + penalty,
                                             xy[side] / count[side] + penalty @ prior[side])
    coefficients = prior + shrink * (coefficients - prior)
    return AffineCalibration(coefficients[:, :2].tolist(), coefficients[:, 2].tolist(),
                             ridge, shrink, source_split, sequences, rows)


def evaluate_cache(dataset_path: str | Path, cache: PredictionCache, groups=None,
                   transform=None, *, postprocess: dict | None = None) -> dict:
    """Reference block scorer over a caller-selected split; no implicit confirmation reads."""
    selected_groups = cache.groups if groups is None else [int(g) for g in groups]
    if len(set(selected_groups)) != len(selected_groups):
        raise ValueError("duplicate groups in evaluation")
    if transform is not None and postprocess is not None:
        raise ValueError("provide either a tracked postprocess config or a custom transform")
    accumulator, per_sequence = BlockAccumulator(), []
    for sequence in iter_labels(dataset_path, selected_groups):
        if sequence.mask is None:
            raise ValueError("evaluation requires a labeled validation scoring mask")
        index = cache._index[sequence.row_group]
        if cache.seq_ids[index] != sequence.seq_ix:
            raise ValueError("cache sequence ID does not match dataset")
        predictions = cache.read_group(sequence.row_group)
        if transform is not None:
            predictions = transform(predictions)
        elif postprocess:
            predictions = apply_calibration(predictions, **postprocess)
        before = len(accumulator.scores)
        mask = sequence.mask & sequence.need
        accumulator.add(sequence.targets, predictions, mask)
        eligible = len(accumulator.scores) > before
        score = accumulator.scores[-1] if eligible else None
        per_sequence.append({"row_group": sequence.row_group, "seq_ix": sequence.seq_ix,
                             "eligible": eligible, "selected_rows": int(mask.sum()),
                             "t0": score[0] if eligible else None,
                             "t1": score[1] if eligible else None,
                             "weighted_pearson": float(np.mean(score)) if eligible else None})
    return {**accumulator.result(), "rows_seen": len(selected_groups) * SEQUENCE_LENGTH,
            "predictions_required": len(selected_groups) * (SEQUENCE_LENGTH - WARMUP),
            "provenance": {**cache.metadata["provenance"], "evaluation_source": "prediction_cache",
                           "cache_path": str(cache.directory.resolve())},
            "postprocess": (postprocess or {}) if transform is None else None,
            "postprocess_tracked": transform is None,
            "per_sequence": per_sequence}


def _paired_scores(candidate, baseline) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(candidate, dict) and isinstance(baseline, dict):
        def indexed(report):
            entries = report["per_sequence"]
            result = {item["seq_ix"]: item for item in entries}
            if len(result) != len(entries):
                raise ValueError("duplicate sequence IDs in paired comparison")
            return result
        cand, base = indexed(candidate), indexed(baseline)
        if cand.keys() != base.keys():
            raise ValueError("paired comparison requires exactly the same sequences")
        ids = sorted(cand)
        if any(cand[s]["eligible"] != base[s]["eligible"] for s in ids):
            raise ValueError("eligibility differs between paired reports")
        ids = [s for s in ids if cand[s]["eligible"]]
        candidate = [cand[s]["weighted_pearson"] for s in ids]
        baseline = [base[s]["weighted_pearson"] for s in ids]
    a, b = np.asarray(candidate, dtype=np.float64), np.asarray(baseline, dtype=np.float64)
    if a.shape != b.shape or a.ndim not in (1, 2) or (a.ndim == 2 and a.shape[1] != 2) or len(a) == 0:
        raise ValueError("paired scores must be matching nonempty [sequences] or [sequences, 2] arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite paired scores")
    return (a.mean(axis=1), b.mean(axis=1)) if a.ndim == 2 else (a, b)


def paired_bootstrap(candidate, baseline, *, n_resamples: int = 10000,
                     seed: int = 20260912, batch_size: int = 256,
                     confidence: float = 0.95) -> dict:
    """Percentile CI of mean paired sequence deltas, with bounded resampling memory."""
    a, b = _paired_scores(candidate, baseline)
    if n_resamples <= 0 or batch_size <= 0 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap parameters")
    delta = a - b
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, len(delta), size=(stop - start, len(delta)))
        means[start:stop] = delta[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return {"delta": float(delta.mean()), "ci_low": float(low), "ci_high": float(high),
            "confidence": confidence, "candidate_wp": float(a.mean()), "baseline_wp": float(b.mean()),
            "sequences": len(delta), "win_fraction": float(np.mean(delta > 0)),
            "n_resamples": n_resamples, "seed": seed, "method": "paired sequence percentile bootstrap",
            "assumption": "sequences are sufficiently independent; search bias is not corrected"}


def _shared_ema_paths(raw: np.ndarray, half_lives) -> np.ndarray:
    """Compute all EMA states in one causal Python pass over a sequence."""
    alpha = (-np.expm1(-np.log(2.0) / np.asarray(half_lives, dtype=np.float64))).astype(np.float32)
    state = np.repeat(raw[:1], len(half_lives), axis=0)
    smoothed = np.empty((len(raw), len(half_lives), 2), dtype=np.float32)
    smoothed[0] = state
    for step in range(1, len(raw)):
        state += alpha[:, None] * (raw[step] - state)
        smoothed[step] = state
    return smoothed


def search_calibration(dataset_path: str | Path, cache: PredictionCache, groups,
                       output_path: str | Path, *,
                       scale_values=(0.5, 0.75, 1.0, 1.25, 1.5, 2.0),
                       half_lives=(2.0, 8.0, 32.0), blends=(0.0, 0.1, 0.25),
                       bootstrap_resamples: int = 10000, seed: int = 20260912) -> dict:
    """One labels pass for a predeclared independent scale/EMA screening grid.

    The default is 42 unique candidates: identity, 35 other per-target scales,
    and six EMA half-life/blend settings. This is deliberately not the Cartesian
    product of scale and smoothing grids. Only explicitly passed groups are read.
    Identity wins numerical ties. Development search CIs do not correct selection
    bias; freeze the winner before a separate confirmation comparison.
    """
    groups = [int(g) for g in groups]
    if not groups or len(set(groups)) != len(groups) or any(g not in cache._index for g in groups):
        raise ValueError("search requires unique, nonempty groups present in the cache")
    scales = [_pair(pair["scales"], "scales") for pair in scale_grid(scale_values)]
    if any(np.any(pair <= 0) for pair in scales):
        raise ValueError("scale search values must be positive")
    halves = [float(h) for h in half_lives]
    if len(set(halves)) != len(halves) or any(not np.isfinite(h) or h <= 0 for h in halves):
        raise ValueError("EMA half-lives must be unique, finite and positive")
    if any(not np.isfinite(b) or not 0 <= b <= 1 for b in blends):
        raise ValueError("EMA blends must be finite and between zero and one")
    configurations = [("identity", {})]
    configurations.extend((f"scale_{s[0]:g}_{s[1]:g}", {"scales": s.tolist()})
                          for s in scales if not np.array_equal(s, [1.0, 1.0]))
    configurations.extend((f"ema_{config['ema_half_life']:g}_{config['ema_blend']:g}", config)
                          for config in ema_grid(halves, blends) if config)
    names = [name for name, _ in configurations]
    if len(set(names)) != len(names):
        raise ValueError("search grid contains duplicate candidates")
    accumulators = [BlockAccumulator() for _ in configurations]
    per_sequence = [[] for _ in configurations]
    half_index = {half: index for index, half in enumerate(halves)}
    for sequence in iter_labels(dataset_path, groups):
        if sequence.mask is None:
            raise ValueError("calibration search requires the validation scoring mask")
        if cache.seq_ids[cache._index[sequence.row_group]] != sequence.seq_ix:
            raise ValueError("cache sequence ID does not match dataset")
        raw = np.asarray(cache.read_group(sequence.row_group), dtype=np.float32)
        if not np.isfinite(raw).all():
            raise ValueError("calibration search requires raw finite warm-up outputs")
        smoothed = _shared_ema_paths(raw, halves) if halves and any(b != 0 for b in blends) else None
        mask = sequence.mask & sequence.need
        for index, (_, config) in enumerate(configurations):
            if "scales" in config:
                predictions = raw * np.asarray(config["scales"], dtype=np.float32)
            elif "ema_half_life" in config:
                blend = np.float32(config["ema_blend"])
                predictions = ((np.float32(1.0) - blend) * raw
                               + blend * smoothed[:, half_index[config["ema_half_life"]], :])
            else:
                predictions = raw
            accumulator = accumulators[index]
            before = len(accumulator.scores)
            accumulator.add(sequence.targets, predictions, mask)
            eligible = len(accumulator.scores) > before
            score = accumulator.scores[-1] if eligible else None
            per_sequence[index].append({"row_group": sequence.row_group, "seq_ix": sequence.seq_ix,
                                        "eligible": eligible, "selected_rows": int(mask.sum()),
                                        "t0": score[0] if eligible else None,
                                        "t1": score[1] if eligible else None,
                                        "weighted_pearson": float(np.mean(score)) if eligible else None})
    reports = [{**acc.result(), "rows_seen": len(groups) * SEQUENCE_LENGTH,
                "predictions_required": len(groups) * (SEQUENCE_LENGTH - WARMUP),
                "provenance": {**cache.metadata["provenance"], "evaluation_source": "prediction_cache",
                               "cache_path": str(cache.directory.resolve())},
                "postprocess": configurations[index][1], "postprocess_tracked": True,
                "per_sequence": per_sequence[index]}
               for index, acc in enumerate(accumulators)]
    winner = 0
    for index, report in enumerate(reports[1:], start=1):
        if report["weighted_pearson"] > reports[winner]["weighted_pearson"] + 1e-12:
            winner = index
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    postprocess_path = output_path.with_name(output_path.stem + ".postprocess.json")
    result = {"version": 1, "dataset": str(Path(dataset_path).resolve()),
              "cache": str(cache.directory.resolve()), "cache_provenance": cache.metadata["provenance"],
              "groups": groups, "candidates_evaluated": len(configurations),
              "best_name": configurations[winner][0], "best_postprocess": configurations[winner][1],
              "best_postprocess_path": str(postprocess_path.resolve()),
              "baseline": reports[0], "best": reports[winner],
              "paired_bootstrap": paired_bootstrap(reports[winner], reports[0],
                                                    n_resamples=bootstrap_resamples, seed=seed),
              "candidates": [{"name": name, "postprocess": config, **report}
                             for (name, config), report in zip(configurations, reports)],
              "selection_note": "Development grid selection; bootstrap does not correct search bias. Freeze before confirmation."}
    _atomic_json(postprocess_path, result["best_postprocess"])
    _atomic_json(output_path, result)
    return result
