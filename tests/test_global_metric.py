import copy

import numpy as np
import pytest

from connectome.global_metric import (GlobalAccumulator, METRIC_ID, paired_global_bootstrap,
                                      merge_statistics, score_statistics, sufficient_statistics)


def direct(y, p):
    y, p = [np.clip(np.asarray(v, np.float32), -2, 2).astype(np.float64) for v in (y, p)]
    result = []
    for target in range(2):
        a, b = y[:, target], p[:, target]
        w = np.abs(a)
        if w.sum() < 1e-8:
            result.append(0.)
            continue
        a, b = a - np.sum(w * a) / w.sum(), b - np.sum(w * b) / w.sum()
        sy, sp = np.sqrt(np.sum(w * a * a) / w.sum()), np.sqrt(np.sum(w * b * b) / w.sum())
        result.append(0. if min(sy, sp) <= 1e-8 else np.sum(w * a * b) / w.sum() / sy / sp)
    return np.clip(result, -1, 1)


def test_pooled_statistics_match_direct_concatenation_and_clipping():
    rng = np.random.default_rng(826)
    y, p = rng.normal(size=(275, 2)) * 3, rng.normal(size=(275, 2)) * 4
    mask = rng.random(275) > .3
    acc = GlobalAccumulator()
    for rows in np.array_split(np.arange(275), 7):
        acc.add(y[rows], p[rows], mask[rows])
    report = acc.result()
    expected = direct(y[mask], p[mask])
    np.testing.assert_allclose([report["t0"], report["t1"]], expected, atol=1e-14, rtol=0)
    assert report["selected_rows"] == int(mask.sum())
    assert report["effective_scored_rows"] == int(mask.sum())
    assert report["metric_id"] == METRIC_ID


def test_one_sign_and_constant_blocks_contribute_to_global_correlation():
    positive = np.array([[.2, .4], [.5, 1.], [1., 1.5]], np.float32)
    negative = -positive
    acc = GlobalAccumulator()
    acc.add(positive, np.ones_like(positive), np.ones(3, bool))
    acc.add(negative, -np.ones_like(negative), np.ones(3, bool))
    report = acc.result()
    np.testing.assert_allclose([report["t0"], report["t1"]], direct(
        np.concatenate([positive, negative]), np.concatenate([np.ones_like(positive), -np.ones_like(negative)])), atol=1e-14)
    assert report["weighted_pearson"] > .9
    assert report["blocks_with_selected_rows"] == 2


def test_equal_sequence_correlations_can_have_different_pooled_result():
    y = np.array([[-1., -.8], [-.4, -.3], [.4, .3], [1., .8]], np.float32)
    shifts = [y + .8, y - .8]
    assert all(np.allclose(direct(y, p), 1.) for p in shifts)
    stats = merge_statistics([sufficient_statistics(y, p) for p in shifts])
    assert score_statistics(stats).mean() < .9
    np.testing.assert_allclose(score_statistics(stats), direct(np.concatenate([y, y]), np.concatenate(shifts)), atol=1e-14)


def test_partition_and_sequence_order_do_not_change_the_estimand():
    rng = np.random.default_rng(97)
    y, p = rng.uniform(-2, 2, (800, 2)), rng.uniform(-2, 2, (800, 2))
    expected = sufficient_statistics(y, p)
    groups = np.array_split(np.arange(800), 31)
    total = merge_statistics([sufficient_statistics(y[g], p[g]) for g in groups[::-1]])
    np.testing.assert_allclose(score_statistics(total), score_statistics(expected), atol=1e-14, rtol=0)


def test_centered_merge_preserves_near_constant_float32_variance():
    rng = np.random.default_rng(17)
    y = rng.normal(size=(20000, 2)).astype(np.float32)
    p = (1.1 + rng.normal(0, 1e-7, y.shape)).astype(np.float32)
    pieces = np.array_split(np.arange(len(y)), 31)
    merged = merge_statistics([sufficient_statistics(y[g], p[g]) for g in pieces])
    np.testing.assert_allclose(score_statistics(merged), direct(y, p), atol=1e-10, rtol=0)
    constant = np.full_like(p, 1.1)
    merged = merge_statistics([sufficient_statistics(y[g], constant[g]) for g in pieces])
    assert not score_statistics(merged).any()


def test_vectorized_centered_merge_matches_individual_merges():
    rng = np.random.default_rng(128)
    groups = [sufficient_statistics(rng.normal(size=(13, 2)), rng.normal(size=(13, 2))) for _ in range(4)]
    stacked = np.stack([groups, groups[::-1]])
    np.testing.assert_allclose(merge_statistics(stacked, axis=1),
                               np.stack([merge_statistics(groups), merge_statistics(groups[::-1])]), atol=0, rtol=0)


def test_empty_masks_and_globally_degenerate_sides():
    acc = GlobalAccumulator()
    zero = np.zeros((3, 2), np.float32)
    assert acc.add(zero, zero, np.zeros(3, bool))["sufficient_statistics"] == np.zeros((2, 6)).tolist()
    with pytest.raises(ValueError, match="no selected"):
        acc.result()
    acc.add(zero, zero, np.ones(3, bool))
    assert acc.result()["weighted_pearson"] == 0.
    y = np.array([[-1., -.5], [1., .5], [.2, .1]], np.float32)
    assert not score_statistics(sufficient_statistics(y, np.ones_like(y))).any()


def test_selected_predictions_only_and_invalid_inputs():
    acc = GlobalAccumulator()
    y, p = np.ones((2, 2), np.float32), np.ones((2, 2), np.float32)
    p[0] = np.nan  # Root callback wrapper validates every required prediction.
    acc.add(y, p, [False, True])
    with pytest.raises(ValueError, match="nonfinite selected"):
        acc.add(y, p, [True, True])
    y[0] = np.nan
    with pytest.raises(ValueError, match="nonfinite target"):
        acc.add(y, p, [False, True])
    with pytest.raises(ValueError, match="shape"):
        sufficient_statistics(np.ones((2, 3)), np.ones((2, 3)))


def reports():
    rng = np.random.default_rng(629)
    left, right = GlobalAccumulator(), GlobalAccumulator()
    rows_a, rows_b = [], []
    for group in range(5):
        y = rng.normal(size=(20 + group * 5, 2)).astype(np.float32)
        p = .5 * y + rng.normal(0, .5, y.shape).astype(np.float32)
        q = .5 * y + rng.normal(0, .7, y.shape).astype(np.float32)
        mask = np.ones(len(y), bool)
        rows_a.append({"seq_ix": group + 100, "row_group": group, **left.add(y, p, mask)})
        rows_b.append({"seq_ix": group + 100, "row_group": group, **right.add(y, q, mask)})
    identity = {"dataset_path": "synthetic", "dataset_size": 100}
    return ({**left.result(), "per_sequence": rows_a, "provenance": identity},
            {**right.result(), "per_sequence": rows_b, "provenance": identity})


def test_cluster_bootstrap_recomputes_global_and_is_batch_invariant():
    candidate, baseline = reports()
    result = paired_global_bootstrap(candidate, baseline, n_resamples=397, batch_size=17)
    other = paired_global_bootstrap(candidate, baseline, n_resamples=397, batch_size=103)
    assert result == other
    assert result["delta"] == pytest.approx(candidate["weighted_pearson"] - baseline["weighted_pearson"], abs=1e-14)
    assert "win_fraction" not in result
    identical = paired_global_bootstrap(candidate, candidate, n_resamples=99)
    assert identical["ci_low"] == identical["ci_high"] == identical["delta"] == 0.


@pytest.mark.parametrize("change, match", [
    (lambda r: r.update(metric_id="old"), "metric identifiers"),
    (lambda r: r["provenance"].update(dataset_path="different"), "dataset identity"),
    (lambda r: r["per_sequence"][0].update(seq_ix=101), "distinct integer sequence"),
    (lambda r: r["per_sequence"][0].update(selected_rows=1), "selected-row count"),
    (lambda r: r["per_sequence"][0]["sufficient_statistics"][0].__setitem__(1, 2.), "pooled statistics"),
])
def test_paired_comparison_rejects_mismatched_evidence(change, match):
    candidate, baseline = reports()
    broken = copy.deepcopy(baseline)
    change(broken)
    with pytest.raises(ValueError, match=match):
        paired_global_bootstrap(candidate, broken, n_resamples=20)


def test_target_moment_mismatch_rejected_even_with_coherent_pooled_report():
    candidate, baseline = reports()
    baseline["per_sequence"][0]["sufficient_statistics"][0][1] += .1
    baseline.pop("sufficient_statistics")
    baseline.pop("weighted_pearson")
    with pytest.raises(ValueError, match="target moments"):
        paired_global_bootstrap(candidate, baseline, n_resamples=20)
