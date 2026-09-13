import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from connectome.training import StreamingGRU
from scripts.diagnose_recurrent_roundoff import explicit_upper, reanchored_upper, _duration


def test_state_update_variants_have_same_fp64_equations():
    torch.set_num_threads(1)
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(822)
        layer = torch.nn.GRU(3, 2).double()
        x = torch.randn(7, 3, dtype=torch.float64)
        expected = layer(x[:, None])[0][:, 0].numpy()
        for formula in ("onnx", "candidate_centered", "previous_centered"):
            observed = explicit_upper(layer, x, formula=formula)
            np.testing.assert_allclose(observed, expected, atol=1e-12, rtol=1e-10)


def test_reanchoring_uses_prior_state_at_chunk_boundary_only():
    torch.set_num_threads(1)
    with torch.random.fork_rng(), torch.inference_mode():
        torch.manual_seed(821)
        model = StreamingGRU(3)
        x = torch.randn(10, 3)
        reference = model.gru1(x[:, None])[0][:, 0].numpy()
        actual = reanchored_upper(model, x.numpy(), reference, chunk=4)
        altered_reference = reference.copy()
        altered_reference[3] += 0.1
        altered = reanchored_upper(model, x.numpy(), altered_reference, chunk=4)
        assert np.array_equal(actual[:4], altered[:4])
        assert not np.array_equal(actual[4:8], altered[4:8])
        assert np.array_equal(actual[8:], altered[8:])
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)


def test_duration_reports_sustained_errors():
    assert _duration(np.array([False, True, True, False, True])) == {
        "rows": 3, "fraction": 0.6, "longest_run": 2, "runs": 2}
