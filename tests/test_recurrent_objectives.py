import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connectome.recurrent_objectives import correlation_objective, objective_parameters, gradient_probe


def _reference_loss(prediction, teacher, targets, *, warmup=2, teacher_weight=.1):
    y, p = targets.clamp(-2., 2.), prediction.clamp(-2., 2.)
    w = y.abs().clone()
    w[:warmup] = 0.
    total = w.sum(0)
    q = w / total
    yc, pc = y - (q * y).sum(0), p - (q * p).sum(0)
    rho = (q * yc * pc).sum(0) / ((q * yc.square()).sum(0) * (q * pc.square()).sum(0)).sqrt()
    match = (w * (prediction - teacher).square()).sum(0) / total.clamp_min(1e-12)
    return (1. - rho.clamp(-1., 1.)).mean() + teacher_weight * match.mean()


def _arrays(rows=19, batch=2):
    rng = np.random.default_rng(855)
    return [torch.from_numpy(rng.normal(size=(rows, batch, 2))) for _ in range(3)]


def test_analytic_coefficients_match_independent_fp64_autograd():
    prediction, teacher, targets = _arrays()
    prediction.requires_grad_()
    expected = _reference_loss(prediction, teacher, targets)
    gradient = torch.autograd.grad(expected, prediction)[0]
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=2)
    assert loss == pytest.approx(float(expected.detach()), abs=1e-14)
    np.testing.assert_allclose(coefficient.numpy(), gradient.numpy(), atol=2e-8, rtol=1e-6)
    assert coefficient.dtype == torch.float32 and not coefficient.requires_grad
    assert report["sequence_target_denominator"] == 4


def test_finite_differences_away_from_clip_boundaries():
    prediction, teacher, targets = _arrays(rows=11)
    prediction = prediction.clamp(-1.5, 1.5)
    loss, coefficient, _ = correlation_objective(prediction, teacher, targets, warmup=2)
    for index in [(0, 0, 1), (2, 0, 0), (3, 1, 1), (6, 0, 1), (10, 1, 0)]:
        plus, minus = prediction.clone(), prediction.clone()
        plus[index] += 1e-5
        minus[index] -= 1e-5
        a = correlation_objective(plus, teacher, targets, warmup=2)[0]
        b = correlation_objective(minus, teacher, targets, warmup=2)[0]
        assert float(coefficient[index]) == pytest.approx((a - b) / 2e-5, abs=2e-8, rel=1e-6)


def test_clip_endpoints_zero_correlation_derivative_but_teacher_stays_active():
    p = torch.tensor([-3., -2., -1., .5, 1., 2., 3., 0., .2], dtype=torch.float64)[:, None, None].repeat(1, 1, 2)
    y = torch.tensor([.2, 1., -.4, -.2, .6, -1., .8, -.3, .5], dtype=torch.float64)[:, None, None].repeat(1, 1, 2)
    teacher = p + 1.
    _, correlation, _ = correlation_objective(p, teacher, y, warmup=0, teacher_weight=0.)
    clipped = p.abs() >= 2.
    assert not correlation[clipped].any()
    assert correlation[~clipped].abs().sum() > 0
    _, combined, _ = correlation_objective(p, teacher, y, warmup=0)
    expected_teacher = -.1 * y.abs() / y.abs().sum(0)
    np.testing.assert_allclose(combined[clipped], expected_teacher[clipped], atol=1e-9, rtol=1e-6)


def test_zero_variance_sides_stay_in_mean_and_teacher_can_escape_constant_predictions():
    prediction, teacher, targets = _arrays(rows=20)
    targets[:, 0, 0] = 0.
    targets[:, 0, 1] = 1.
    targets[:, 1, 0] = torch.linspace(.1, 1., 20)  # One sign; still a training side.
    prediction[:, 1, 1] = .3
    loss, coefficient, report = correlation_objective(prediction, teacher, targets, warmup=2)
    assert report["degenerate_sides"] == 3
    assert report["correlation_active"] == [[False, False], [True, False]]
    assert report["sequence_target_denominator"] == 4
    assert report["correlation_loss"] == pytest.approx(1. - report["rho_per_sequence_target"][1][0] / 4)
    assert not coefficient[:, 0, 0].any()
    assert coefficient[:, 1, 1].abs().sum() > 0  # Raw teacher loss supplies a gradient.
    _, correlation_only, _ = correlation_objective(prediction, teacher, targets, warmup=2, teacher_weight=0.)
    assert not correlation_only[:, 1, 1].any()


def test_floors_match_scorer_and_all_zero_loss_is_one():
    p, teacher, y = _arrays(rows=10, batch=1)
    y[:] = 1e-12
    loss, coefficient, report = correlation_objective(p, teacher, y, warmup=0, teacher_weight=0.)
    assert loss == 1. and not coefficient.any() and report["degenerate_sides"] == 2
    y.zero_()
    loss, coefficient, _ = correlation_objective(p, teacher, y, warmup=0)
    assert loss == 1. and not coefficient.any()


def test_fp32_point_rho_matches_supplied_per_target_scorer():
    from utils import weighted_pearson
    p, teacher, y = [value.float() for value in _arrays(rows=143)]
    _, _, report = correlation_objective(p, teacher, y)
    for batch in range(2):
        for side in range(2):
            expected = weighted_pearson(y[99:, batch, side], p[99:, batch, side])
            assert report["rho_per_sequence_target"][batch][side] == pytest.approx(expected, abs=2e-15)


def test_warmup_permutations_and_equal_sequence_normalization():
    p, t, y = _arrays(rows=21)
    before = [value.clone() for value in (p, t, y)]
    loss, coefficient, _ = correlation_objective(p, t, y, warmup=2)
    altered = [value.clone() for value in (p, t, y)]
    for v in altered:
        v[:2] = 999.
    assert correlation_objective(*altered, warmup=2)[0] == pytest.approx(loss, abs=1e-14)
    assert not coefficient[:2].any()
    permutation = torch.tensor([0, 1, *range(20, 1, -1)])
    shuffled_loss, shuffled_coefficient, _ = correlation_objective(p[permutation], t[permutation], y[permutation], warmup=2)
    assert shuffled_loss == pytest.approx(loss, abs=1e-14)
    np.testing.assert_allclose(shuffled_coefficient, coefficient[permutation], atol=1e-9, rtol=1e-6)
    _, reverse_coefficient, _ = correlation_objective(p[:, [1, 0]], t[:, [1, 0]], y[:, [1, 0]], warmup=2)
    np.testing.assert_allclose(reverse_coefficient, coefficient[:, [1, 0]], atol=1e-9)
    separate = [correlation_objective(p[:, b:b+1], t[:, b:b+1], y[:, b:b+1], warmup=2) for b in range(2)]
    assert loss == pytest.approx(sum(result[0] for result in separate) / 2, abs=1e-14)
    for b, result in enumerate(separate):
        np.testing.assert_allclose(coefficient[:, b:b+1], result[1] / 2, atol=1e-9)
    for actual, expected in zip((p, t, y), before):
        assert torch.equal(actual, expected)


def test_future_labels_change_training_coefficients_but_never_model_forward_or_state():
    torch.set_num_threads(1)
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(834)
        layer = torch.nn.GRU(3, 2)
        x = torch.randn(120, 1, 3)
        p, state = layer(x)
        p_before, state_before = p.clone(), state.clone()
        y = torch.randn_like(p)
    _, c0, _ = correlation_objective(p, p, y)
    y[-1] *= -4.
    _, c1, _ = correlation_objective(p, p, y)
    assert not torch.equal(c0[99:-1], c1[99:-1])
    assert torch.equal(p, p_before) and torch.equal(state, state_before)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_inputs_rejected(bad):
    p, t, y = _arrays()
    p[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        correlation_objective(p, t, y, warmup=2)


def test_specification_is_copied_and_gradient_probe_is_objective_specific():
    from connectome.training import StreamingGRU
    spec = objective_parameters()
    spec["label_clip"][0] = -99
    assert objective_parameters()["label_clip"] == [-2., 2.]
    torch.set_num_threads(1)
    with torch.random.fork_rng(), torch.no_grad():
        torch.manual_seed(854)
        model = StreamingGRU(3)
        h0 = torch.randn(400, 3)
        h1 = model.gru1(h0[:, None])[0][:, 0]
        p = model.head(h1)
        reference = tuple(v.numpy() for v in (h0, h1, p))
        targets = torch.randn(400, 2).numpy()
    result = gradient_probe(model, reference, reference[2], targets, start=128)
    assert result["passed"], result
    assert result["objective_report"]["objective"]["version"] == objective_parameters()["version"]
    assert len(result["gradient_tensors"]) == 6
