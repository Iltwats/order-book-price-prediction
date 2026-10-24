"""Synthetic checks of current scorer defaults and explicit historical scoring."""
from types import SimpleNamespace

import numpy as np
import pytest

from connectome import calibration
from connectome.calibration import PredictionCache, evaluate_cache, paired_bootstrap
from connectome.global_metric import GlobalAccumulator, METRIC_ID
from utils import (BlockAccumulator, LEGACY_METRIC_ID, SEQUENCE_LENGTH, WARMUP,
                   add_scored_sequence, scoring_accumulator, weighted_pearson)


def synthetic_sequences():
    base = np.resize(np.array([-1., -.4, .4, 1.], np.float32), SEQUENCE_LENGTH)
    targets = np.column_stack([base, .8 * base])
    need = np.arange(SEQUENCE_LENGTH) >= WARMUP
    return [SimpleNamespace(row_group=group, seq_ix=500 + group,
                            targets=targets.copy(), need=need.copy(), mask=need.copy())
            for group in range(2)]


def test_default_accumulator_is_global_and_legacy_requires_identifier():
    assert isinstance(scoring_accumulator(), GlobalAccumulator)
    assert isinstance(scoring_accumulator(LEGACY_METRIC_ID), BlockAccumulator)
    with pytest.raises(ValueError, match="unsupported"):
        scoring_accumulator("unknown")


def test_cached_evaluation_defaults_global_with_explicit_legacy_reproduction(tmp_path, monkeypatch):
    sequences = synthetic_sequences()
    cache = PredictionCache.create(tmp_path / "cache", [0, 1], [500, 501], {"dataset_path": "synthetic"})
    predictions = [sequence.targets + offset for sequence, offset in zip(sequences, (.8, -.8))]
    for index, prediction in enumerate(predictions):
        cache.write(index, prediction)
    cache.finalize()
    monkeypatch.setattr(calibration, "iter_labels", lambda path, groups: (sequences[group] for group in groups))
    global_report = evaluate_cache("synthetic", cache)
    historical = evaluate_cache("synthetic", cache, metric_id=LEGACY_METRIC_ID)
    y, p = [np.concatenate([value[WARMUP:] for value in values])
            for values in ([sequence.targets for sequence in sequences], predictions)]
    expected = np.mean([weighted_pearson(y[:, side], p[:, side]) for side in range(2)])
    assert global_report["metric_id"] == METRIC_ID
    assert global_report["weighted_pearson"] == pytest.approx(expected, abs=1e-13)
    assert historical["metric_id"] == LEGACY_METRIC_ID
    assert historical["weighted_pearson"] == pytest.approx(1., abs=1e-13)
    assert global_report["weighted_pearson"] < historical["weighted_pearson"]
    assert all("sufficient_statistics" in row for row in global_report["per_sequence"])
    with pytest.raises(ValueError, match="different scoring metric"):
        paired_bootstrap(global_report, historical, n_resamples=10)


def test_required_unscored_predictions_are_checked_and_warmup_nan_is_allowed():
    sequence = synthetic_sequences()[0]
    prediction = sequence.targets.copy()
    prediction[:WARMUP] = np.nan
    mask = np.zeros(SEQUENCE_LENGTH, bool)
    mask[WARMUP] = True
    acc = GlobalAccumulator()
    row = add_scored_sequence(acc, sequence.targets, prediction, mask, row_group=0, seq_ix=500)
    assert row["selected_rows"] == 1
    prediction[WARMUP + 1] = np.nan  # Required, although excluded from the scoring mask.
    with pytest.raises(ValueError, match="including unscored"):
        add_scored_sequence(acc, sequence.targets, prediction, mask, row_group=1, seq_ix=501)


def test_warmup_cannot_be_selected_by_a_scoring_mask():
    sequence = synthetic_sequences()[0]
    mask = sequence.mask.copy()
    mask[0] = True
    with pytest.raises(ValueError, match="warm-up"):
        add_scored_sequence(GlobalAccumulator(), sequence.targets, sequence.targets, mask,
                            row_group=0, seq_ix=500)
