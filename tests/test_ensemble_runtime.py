"""Tests for CompiledEnsembleBackend, FallbackPythonEnsembleBackend, and C++ parity."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from connectome.ensemble_runtime import CompiledEnsembleBackend, FallbackPythonEnsembleBackend, load_ensemble_predictor

ROOT = Path(__file__).resolve().parents[1]
CPP_SOURCE = ROOT / "native" / "ensemble_engine.cpp"


@pytest.fixture
def dummy_weights(tmp_path):
    rng = np.random.default_rng(123)
    npz_path = tmp_path / "ensemble_weights.npz"
    np.savez_compressed(
        npz_path,
        n_onnx_gru=np.int32(1),
        has_sorted_gru=np.int32(1),
        has_tcn80=np.int32(0),
        onnx_0_w0=rng.normal(0, 0.05, (384, 112)).astype(np.float32),
        onnx_0_r0=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        onnx_0_b0=rng.normal(0, 0.02, (768,)).astype(np.float32),
        onnx_0_w1=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        onnx_0_r1=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        onnx_0_b1=rng.normal(0, 0.02, (768,)).astype(np.float32),
        onnx_0_hw=rng.normal(0, 0.05, (128, 2)).astype(np.float32),
        onnx_0_hb=rng.normal(0, 0.02, (2,)).astype(np.float32),
        onnx_0_blend=np.float32(0.85),
        sgru_in_mu=rng.normal(0, 0.1, (169,)).astype(np.float32),
        sgru_in_sd=rng.uniform(0.5, 1.5, (169,)).astype(np.float32),
        sgru_stem_w=rng.normal(0, 0.05, (128, 169)).astype(np.float32),
        sgru_stem_b=rng.normal(0, 0.02, (128,)).astype(np.float32),
        sgru_w0=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        sgru_r0=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        sgru_bi0=rng.normal(0, 0.02, (384,)).astype(np.float32),
        sgru_bh0=rng.normal(0, 0.02, (384,)).astype(np.float32),
        sgru_w1=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        sgru_r1=rng.normal(0, 0.05, (384, 128)).astype(np.float32),
        sgru_bi1=rng.normal(0, 0.02, (384,)).astype(np.float32),
        sgru_bh1=rng.normal(0, 0.02, (384,)).astype(np.float32),
        sgru_h1_w=rng.normal(0, 0.05, (128, 384)).astype(np.float32),
        sgru_h1_b=rng.normal(0, 0.02, (128,)).astype(np.float32),
        sgru_h2_w=rng.normal(0, 0.05, (2, 128)).astype(np.float32),
        sgru_h2_b=rng.normal(0, 0.02, (2,)).astype(np.float32),
        sgru_blend=np.float32(0.15),
        cal_in_dim=np.int32(44),
        cal_hidden=np.int32(32),
        cal_seeds=np.int32(2),
        cal_mu=rng.normal(0, 0.1, (44,)).astype(np.float32),
        cal_sd=rng.uniform(0.5, 1.5, (44,)).astype(np.float32),
        cal_w1=rng.normal(0, 0.05, (2, 32, 44)).astype(np.float32),
        cal_b1=rng.normal(0, 0.02, (2, 32)).astype(np.float32),
        cal_w2=rng.normal(0, 0.05, (2, 2, 32)).astype(np.float32),
        cal_b2=rng.normal(0, 0.02, (2, 2)).astype(np.float32),
    )
    return npz_path


def test_compiled_matches_python_fallback_and_resets_cleanly(dummy_weights):
    cpp_backend = CompiledEnsembleBackend(CPP_SOURCE, dummy_weights)
    py_backend = FallbackPythonEnsembleBackend(dummy_weights)
    rng = np.random.default_rng(77)
    seq_a = rng.normal(0, 0.5, (120, 112)).astype(np.float32)
    seq_b = rng.normal(0, 0.5, (120, 112)).astype(np.float32)

    out_a1_cpp, out_a1_py = [], []
    for step in range(120):
        dp = SimpleNamespace(seq_ix=10, step_in_seq=step, need_prediction=(step >= 99), state=seq_a[step])
        rc = cpp_backend.predict(dp)
        rp = py_backend.predict(dp)
        if step < 99:
            assert rc is None and rp is None
        else:
            assert rc.shape == (2,) and rc.dtype == np.float32
            assert np.allclose(rc, rp, atol=2e-6, rtol=1e-5)
            out_a1_cpp.append(rc)
            out_a1_py.append(rp)

    # Interleave a second sequence, then replay sequence A to verify exact state reset
    for step in range(120):
        dp = SimpleNamespace(seq_ix=20, step_in_seq=step, need_prediction=(step >= 99), state=seq_b[step])
        cpp_backend.predict(dp)

    out_a2_cpp = []
    for step in range(120):
        dp = SimpleNamespace(seq_ix=30, step_in_seq=step, need_prediction=(step >= 99), state=seq_a[step])
        rc = cpp_backend.predict(dp)
        if step >= 99:
            out_a2_cpp.append(rc)

    assert np.array_equal(np.stack(out_a1_cpp), np.stack(out_a2_cpp))


def test_rejects_nonfinite_inputs_and_fallback_works_when_source_missing(dummy_weights, tmp_path):
    pred = load_ensemble_predictor(tmp_path / "nonexistent.cpp", dummy_weights)
    assert isinstance(pred, FallbackPythonEnsembleBackend)
    bad = np.zeros(112, dtype=np.float32)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        pred.predict(SimpleNamespace(seq_ix=1, step_in_seq=0, need_prediction=False, state=bad))
