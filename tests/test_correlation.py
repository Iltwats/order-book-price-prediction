from pathlib import Path

import numpy as np
import pytest
torch = pytest.importorskip("torch")

from connectome.correlation import sequence_moments, moment_loss, optimize_moments, _validate_groups


def batch_moments(samples):
    rows = [sequence_moments(h, y) for h, y in samples]
    return {key: np.stack([r[key] for r in rows]) for key in rows[0]}


def tensors(moments):
    return {key: torch.tensor(value, dtype=torch.float64) for key, value in moments.items()}


def test_moments_match_independent_row_loss_and_gradients():
    rng = np.random.default_rng(918)
    samples = [(rng.normal(size=(n, 4)) + b, rng.normal(size=(n, 2)).astype(np.float32)*s)
               for b, n, s in [(0, 117, .3), (3, 409, 3)]]
    m = tensors(batch_moments(samples))
    beta0 = torch.tensor(rng.normal(size=(2, 4)), dtype=torch.float64)
    bias0 = torch.tensor([.3, -.5], dtype=torch.float64)
    beta = (beta0 + .1).requires_grad_()
    center = m["feature_mean"].mean(0)
    for objective in ("correlation", "mse"):
        value, details = moment_loss(beta, m, beta0, bias0, objective=objective, penalty=.01)
        row_losses = []
        for h, y in samples:
            h = torch.tensor(h, dtype=torch.float64)
            y = torch.tensor(np.clip(y, -2, 2), dtype=torch.float64)
            for k in range(2):
                w = y[:,k].abs(); w = w/w.sum()
                p = h@beta[k] + bias0[k] + (beta0[k]-beta[k])@center[k]
                yc, pc = y[:,k]-w@y[:,k], p-w@p
                rho = (w*yc*pc).sum()/torch.sqrt((w*yc.square()).sum()*(w*pc.square()).sum())
                row_losses.append(1-rho if objective=="correlation" else (w*(p-y[:,k]).square()).sum())
        direct = torch.stack(row_losses).mean() + .01*((beta-beta0).square().sum(-1)/beta0.square().sum(-1)).mean()
        torch.testing.assert_close(value, direct, atol=1e-12, rtol=1e-12)
        grad, = torch.autograd.grad(value, beta, retain_graph=True)
        other, = torch.autograd.grad(direct, beta, retain_graph=True)
        torch.testing.assert_close(grad, other, atol=1e-12, rtol=1e-12)
    # Global training offset is preserved, separately for each target.
    before = (m["feature_mean"]*beta0[None]).sum(-1)+bias0
    after = (m["feature_mean"]*beta[None]).sum(-1)+details["bias"]
    torch.testing.assert_close(before.mean(0), after.mean(0), atol=1e-12, rtol=1e-12)


def test_weighted_moments_do_not_pool_or_count_rows_as_sequences():
    rng = np.random.default_rng(17)
    h, y = rng.normal(size=(200, 5)), rng.normal(size=(200, 2)).astype(np.float32)
    a = sequence_moments(h,y)
    b = sequence_moments(np.repeat(h,3,axis=0),np.repeat(y,3,axis=0))
    for key in a:
        np.testing.assert_allclose(a[key],b[key],atol=1e-14,rtol=1e-12)
    assert not np.allclose(a["covariance"][0],a["covariance"][1])


def test_centered_parameterization_is_translation_invariant():
    rng=np.random.default_rng(12)
    h,y=rng.normal(size=(200,3)),rng.normal(size=(200,2)).astype(np.float32)
    shift=np.array([11.,-24.,3.])
    b0=torch.tensor([[.2,.3,-.1],[-.2,.4,.8]],dtype=torch.float64)
    a0=torch.tensor([.2,-.4],dtype=torch.float64)
    b=(b0+.12).requires_grad_()
    for objective in ("correlation","mse"):
        x,d=moment_loss(b,tensors(batch_moments([(h,y)])),b0,a0,objective=objective)
        z,e=moment_loss(b,tensors(batch_moments([(h+shift,y)])),b0,a0-b0@torch.tensor(shift),objective=objective)
        torch.testing.assert_close(x,z,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(h@b.detach().numpy().T+d["bias"].detach().numpy(),
                                   (h+shift)@b.detach().numpy().T+e["bias"].detach().numpy(),atol=1e-12)


def test_eligibility_finite_inputs_and_constant_predictions():
    h=np.ones((30,2)); y=np.column_stack((np.linspace(-1,1,30),np.linspace(2,-2,30))).astype(np.float32)
    assert sequence_moments(h,np.abs(y)) is None
    assert sequence_moments(h,np.zeros_like(y)) is None
    tiny=y.copy();tiny[:,0]*=1e-12
    assert sequence_moments(h,tiny) is not None  # Retain block with tiny-weight side.
    bad=y.copy();bad[0,0]=np.inf
    with pytest.raises(ValueError,match="nonfinite"):
        sequence_moments(h,bad)
    m=batch_moments([(h,y)])
    b=np.eye(2);a=np.zeros(2)
    fitted,_,report=optimize_moments(m,b,a,max_iterations=5)
    assert report["training_unclipped_rho"]==0
    assert report["zero_variance_sides"]==2
    np.testing.assert_array_equal(fitted,b)
    m["cross"][0,0,0]=np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        optimize_moments(m,b,a)


def test_correlation_fit_recovers_signal_with_no_validation_inputs():
    rng=np.random.default_rng(314)
    samples=[]
    for _ in range(4):
        h=rng.normal(size=(300,3))
        y=np.column_stack((h[:,1],-h[:,2])).astype(np.float32)
        samples.append((h,y))
    initial=np.array([[.5,.1,0],[.5,0,-.1]])
    fitted,bias,report=optimize_moments(batch_moments(samples),initial,np.zeros(2),penalty=1e-3,max_iterations=60)
    assert report["training_unclipped_rho"]>report["initial_training_unclipped_rho"]+.3
    assert report["final_loss"]<=report["initial_loss"]
    assert np.isfinite(fitted).all() and np.isfinite(bias).all()


def test_fortran_order_initial_head_matches_contiguous_optimizer_result():
    """Cached ONNX head transpose must not give LBFGS a strided parameter."""
    rng = np.random.default_rng(136)
    samples = []
    for _ in range(3):
        h = rng.normal(size=(80, 4))
        y = np.column_stack((h[:, 1] - .2*h[:, 2],
                             -.6*h[:, 3] + .1*h[:, 0])).astype(np.float32)
        samples.append((h, y))
    moments = batch_moments(samples)
    initial = np.array([[.5, .125, -.25, .0625],
                        [-.25, .125, .25, -.125]], dtype=np.float64)
    fortran_initial = np.asfortranarray(initial)
    assert fortran_initial.flags.f_contiguous
    assert not fortran_initial.flags.c_contiguous
    bias = np.array([.125, -.25], dtype=np.float64)
    expected_beta, expected_bias, expected_report = optimize_moments(
        moments, initial, bias, max_iterations=12,
    )
    actual_beta, actual_bias, actual_report = optimize_moments(
        moments, fortran_initial, bias, max_iterations=12,
    )
    np.testing.assert_array_equal(actual_beta, expected_beta)
    np.testing.assert_array_equal(actual_bias, expected_bias)
    assert actual_report == expected_report
    assert actual_report["optimizer_iterations"] > 0
    assert actual_report["final_loss"] < actual_report["initial_loss"]


def test_validation_table_and_duplicate_groups_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    path=tmp_path/"sample.parquet"
    pq.write_table(pa.table({"is_scored":[True]}),path)
    with pytest.raises(ValueError,match="training data"):
        _validate_groups(path,[0])
    pq.write_table(pa.table({"seq_ix":[1,2]}),path,row_group_size=1)
    with pytest.raises(ValueError,match="distinct"):
        _validate_groups(path,[0,0])
    with pytest.raises(ValueError,match="distinct"):
        _validate_groups(path,[2])
