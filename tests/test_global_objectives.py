import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connectome.global_objectives import correlation_objective, objective_parameters


def _arrays(rows=19, batch=3):
    rng = np.random.default_rng(269)
    return [torch.from_numpy(rng.normal(size=(rows, batch, 2))) for _ in range(3)]


def _reference_loss(prediction, teacher, targets, *, warmup=2, teacher_weight=.1):
    # Independent implementation concatenates post-warm-up rows before reducing.
    raw = prediction[warmup:].reshape(-1, 2)
    prior = teacher[warmup:].reshape(-1, 2)
    y = targets[warmup:].reshape(-1, 2).clamp(-2., 2.)
    p = raw.clamp(-2., 2.)
    w = y.abs()
    q = w / w.sum(0)
    centered_y, centered_p = y - (q*y).sum(0), p - (q*p).sum(0)
    rho = (q*centered_y*centered_p).sum(0) / (
        (q*centered_y.square()).sum(0) * (q*centered_p.square()).sum(0)).sqrt()
    teacher_mse = (w*(raw-prior).square()).sum(0) / w.sum(0)
    return (1.-rho.clamp(-1., 1.)).mean() + teacher_weight*teacher_mse.mean()


def test_pooled_coefficients_match_independent_fp64_autograd():
    prediction, teacher, targets = _arrays()
    prediction.requires_grad_()
    reference = _reference_loss(prediction, teacher, targets)
    gradient = torch.autograd.grad(reference, prediction)[0]
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=2)
    assert loss == pytest.approx(float(reference.detach()), abs=1e-14)
    np.testing.assert_allclose(coefficient.numpy(), gradient.numpy(), atol=2e-8, rtol=1e-6)
    assert coefficient.dtype == torch.float32 and not coefficient.requires_grad
    assert report["target_denominator"] == 2
    assert len(report["rho_per_target"]) == 2
    assert report["pooled_post_warmup_rows"] == 51
    json.dumps(report, allow_nan=False)


def test_finite_differences_include_shared_centering_across_sequences():
    prediction, teacher, targets = _arrays(rows=11)
    prediction.clamp_(-1.5, 1.5)
    _, coefficient, _ = correlation_objective(prediction, teacher, targets, warmup=2)
    for index in ((0, 0, 0), (2, 0, 1), (3, 1, 0), (7, 2, 1), (10, 1, 0)):
        plus, minus = prediction.clone(), prediction.clone()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        a = correlation_objective(plus, teacher, targets, warmup=2)[0]
        b = correlation_objective(minus, teacher, targets, warmup=2)[0]
        assert float(coefficient[index]) == pytest.approx((a-b)/2e-5, abs=2e-8, rel=1e-6)


def test_batch_partition_and_order_do_not_reweight_rows():
    prediction, teacher, targets = _arrays(rows=16, batch=4)
    original = [value.clone() for value in (prediction, teacher, targets)]
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=2)
    flattened = [value[2:].reshape(-1, 1, 2) for value in (prediction, teacher, targets)]
    flat_loss, flat_coefficient, flat_report = correlation_objective(*flattened, warmup=0)
    assert flat_loss == pytest.approx(loss, abs=1e-14)
    np.testing.assert_allclose(flat_coefficient, coefficient[2:].reshape(-1, 1, 2), atol=1e-9, rtol=1e-6)
    np.testing.assert_allclose(flat_report["rho_per_target"], report["rho_per_target"], atol=1e-14)
    reordered = [value[:, [3, 0, 2, 1]] for value in (prediction, teacher, targets)]
    new_loss, new_coefficient, _ = correlation_objective(*reordered, warmup=2)
    assert new_loss == pytest.approx(loss, abs=1e-14)
    np.testing.assert_allclose(new_coefficient, coefficient[:, [3, 0, 2, 1]], atol=1e-9)
    for actual, expected in zip((prediction, teacher, targets), original):
        assert torch.equal(actual, expected)


def test_offsets_and_unequal_weights_differ_from_mean_sequence_objective():
    from connectome.recurrent_objectives import correlation_objective as old_objective
    prediction, teacher, targets = _arrays(rows=21, batch=2)
    prediction = prediction.clamp(-.2, .2)
    targets = targets.clamp(-.2, .2)
    prediction[:, 1] += 1.
    targets[:, 1] -= 1.
    pooled_loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=2, teacher_weight=0.)
    old_loss, old_coefficient, _ = old_objective(prediction, teacher, targets, warmup=2, teacher_weight=0.)
    assert abs(pooled_loss-old_loss) > .5
    assert not torch.allclose(coefficient, old_coefficient)
    assert report["target_weight_sum"][0] > targets[2:, 0, 0].abs().sum()*5


def test_warmup_is_excluded_for_every_sequence():
    prediction, teacher, targets = _arrays()
    baseline, coefficients, _ = correlation_objective(prediction, teacher, targets, warmup=2)
    altered = [value.clone() for value in (prediction, teacher, targets)]
    for value in altered:
        value[:2] = 999.
    actual, changed_coefficients, _ = correlation_objective(*altered, warmup=2)
    assert actual == pytest.approx(baseline, abs=1e-14)
    assert not coefficients[:2].any()
    assert torch.equal(coefficients, changed_coefficients)


def test_constant_one_sign_sequences_are_retained_in_global_statistics():
    y = torch.tensor([-.8, .5], dtype=torch.float64)[None, :, None].expand(8, 2, 2).clone()
    p = y.clone()
    loss, coefficient, report = correlation_objective(p, p, y, warmup=0, teacher_weight=0.)
    assert loss == pytest.approx(0., abs=1e-14)
    assert report["rho_per_target"] == pytest.approx([1., 1.], abs=1e-14)
    assert report["degenerate_targets"] == 0
    assert torch.isfinite(coefficient).all()


def test_clipped_correlation_has_zero_endpoint_derivatives_but_raw_teacher_remains_active():
    p = torch.tensor([-3., -2., -1., .5, 1., 2., 3., 0., .2], dtype=torch.float64)[:, None, None].repeat(1, 3, 2)
    y = torch.tensor([.2, 1., -.4, -.2, .6, -1., .8, -.3, .5], dtype=torch.float64)[:, None, None].repeat(1, 3, 2)
    teacher = p+1.
    _, correlation, _ = correlation_objective(p, teacher, y, warmup=0, teacher_weight=0.)
    clipped = p.abs() >= 2.
    assert not correlation[clipped].any()
    assert correlation[~clipped].abs().sum() > 0
    _, combined, _ = correlation_objective(p, teacher, y, warmup=0)
    expected = -.1*y.abs()/y.abs().sum((0, 1))
    np.testing.assert_allclose(combined[clipped], expected[clipped], atol=1e-9, rtol=1e-6)


def test_degenerate_global_targets_keep_two_target_denominator_and_teacher_escape():
    prediction, teacher, targets = _arrays(rows=20, batch=2)
    targets[:, :, 0] = 0.
    prediction[:, :, 1] = .3
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=2)
    assert report["degenerate_targets"] == 2
    assert report["correlation_loss"] == 1.
    assert report["target_denominator"] == 2
    assert not coefficient[:, :, 0].any()
    assert coefficient[:, :, 1].abs().sum() > 0
    assert loss >= 1.
    no_teacher, correlation_only, _ = correlation_objective(prediction, teacher, targets, warmup=2, teacher_weight=0.)
    assert no_teacher == 1. and not correlation_only.any()


def test_weight_floor_uses_pooled_weight_and_zero_targets_give_loss_one():
    prediction, teacher, targets = _arrays(rows=10, batch=4)
    targets[:] = 1e-12
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=0, teacher_weight=0.)
    assert loss == 1. and not coefficient.any() and report["degenerate_targets"] == 2
    targets.zero_()
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=0)
    assert loss == 1. and not coefficient.any()
    assert report["normalized_weight_sum"] == [0., 0.]


def test_specification_is_independent_and_distinct_from_legacy():
    from connectome.recurrent_objectives import objective_parameters as old_parameters
    spec = objective_parameters()
    spec["label_clip"][0] = -99
    assert objective_parameters()["label_clip"] == [-2., 2.]
    assert spec["version"] == "batch_pooled_global_correlation.v1"
    assert spec["version"] != old_parameters()["version"]
    assert "surrogate" in spec["surrogate_scope"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_values_rejected(bad):
    prediction, teacher, targets = _arrays()
    prediction[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        correlation_objective(prediction, teacher, targets, warmup=2)


@pytest.mark.parametrize("warmup", [True, -1, 19])
def test_invalid_warmup_rejected(warmup):
    with pytest.raises(ValueError, match="warmup"):
        correlation_objective(*_arrays(), warmup=warmup)
