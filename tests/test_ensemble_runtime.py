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
            assert np.allclose(rc, rp, atol=2e-3, rtol=2e-3)
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


def test_v2_variable_hidden_sgru_and_slstm_match_pytorch_and_fallback(tmp_path):
    import torch
    from experiments.gpu_train_sorted_models import ScratchSortedLSTM
    from experiments.train_scratch_gru import ScratchSortedGRU
    from experiments.train_scratch_tcn import build_lob_engineered_features_torch

    torch.manual_seed(42)
    sgru = ScratchSortedGRU(hidden=160).eval()
    slstm = ScratchSortedLSTM(hidden=144).eval()
    rng = np.random.default_rng(4242)
    sgru.in_mu.copy_(torch.from_numpy(rng.normal(0, 0.1, (1, 1, 169)).astype(np.float32)))
    sgru.in_sd.copy_(torch.from_numpy(rng.uniform(0.6, 1.4, (1, 1, 169)).astype(np.float32)))
    slstm.in_mu.copy_(torch.from_numpy(rng.normal(0, 0.1, (1, 1, 169)).astype(np.float32)))
    slstm.in_sd.copy_(torch.from_numpy(rng.uniform(0.6, 1.4, (1, 1, 169)).astype(np.float32)))

    def _pack(d, prefix, state, rnn_name, blend):
        d[f"{prefix}in_mu"] = state["in_mu"].numpy().ravel().astype(np.float32)
        d[f"{prefix}in_sd"] = state["in_sd"].numpy().ravel().astype(np.float32)
        d[f"{prefix}stem_w"] = state["stem.0.weight"].numpy().astype(np.float32)
        d[f"{prefix}stem_b"] = state["stem.0.bias"].numpy().astype(np.float32)
        d[f"{prefix}w0"] = state[f"{rnn_name}0.weight_ih_l0"].numpy().astype(np.float32)
        d[f"{prefix}r0"] = state[f"{rnn_name}0.weight_hh_l0"].numpy().astype(np.float32)
        d[f"{prefix}bi0"] = state[f"{rnn_name}0.bias_ih_l0"].numpy().astype(np.float32)
        d[f"{prefix}bh0"] = state[f"{rnn_name}0.bias_hh_l0"].numpy().astype(np.float32)
        d[f"{prefix}w1"] = state[f"{rnn_name}1.weight_ih_l0"].numpy().astype(np.float32)
        d[f"{prefix}r1"] = state[f"{rnn_name}1.weight_hh_l0"].numpy().astype(np.float32)
        d[f"{prefix}bi1"] = state[f"{rnn_name}1.bias_ih_l0"].numpy().astype(np.float32)
        d[f"{prefix}bh1"] = state[f"{rnn_name}1.bias_hh_l0"].numpy().astype(np.float32)
        d[f"{prefix}h1_w"] = state["head.0.weight"].numpy().astype(np.float32)
        d[f"{prefix}h1_b"] = state["head.0.bias"].numpy().astype(np.float32)
        d[f"{prefix}h2_w"] = state["head.2.weight"].numpy().astype(np.float32)
        d[f"{prefix}h2_b"] = state["head.2.bias"].numpy().astype(np.float32)
        d[f"{prefix}blend"] = np.float32(blend)

    payload = {
        "n_onnx_gru": np.int32(0),
        "has_sorted_gru": np.int32(1),
        "n_sorted_gru": np.int32(1),
        "n_sorted_lstm": np.int32(1),
        "has_tcn80": np.int32(0),
        "cal_seeds": np.int32(0),
    }
    _pack(payload, "sgru_0_", sgru.state_dict(), "gru", 0.55)
    _pack(payload, "slstm_0_", slstm.state_dict(), "lstm", 0.45)
    npz_path = tmp_path / "v2_weights.npz"
    np.savez_compressed(npz_path, **payload)

    cpp_b = CompiledEnsembleBackend(CPP_SOURCE, npz_path)
    py_b = FallbackPythonEnsembleBackend(npz_path)

    seq = rng.normal(0, 0.5, (115, 112)).astype(np.float32)
    with torch.no_grad():
        feat_btf = build_lob_engineered_features_torch(torch.from_numpy(seq[None])).transpose(1, 2).contiguous()
        pg, _, _ = sgru.forward_features(feat_btf, *sgru.initial_states(1))
        pl, _, _ = slstm.forward_features(feat_btf, *slstm.initial_states(1))
        expected = np.clip(0.55 * np.clip(pg[0, 99:].numpy(), -2, 2) + 0.45 * np.clip(pl[0, 99:].numpy(), -2, 2), -2, 2)

    cpp_outs, py_outs = [], []
    for st in range(115):
        dp = SimpleNamespace(seq_ix=1, step_in_seq=st, need_prediction=(st >= 99), state=seq[st])
        rc = cpp_b.predict(dp)
        rp = py_b.predict(dp)
        if st >= 99:
            cpp_outs.append(rc)
            py_outs.append(rp)

    cpp_arr = np.stack(cpp_outs)
    py_arr = np.stack(py_outs)
    assert np.allclose(cpp_arr, expected, atol=2e-3, rtol=2e-3)
    assert np.allclose(py_arr, expected, atol=3e-6, rtol=1e-5)


def test_rejects_nonfinite_inputs_and_fallback_works_when_source_missing(dummy_weights, tmp_path):
    pred = load_ensemble_predictor(tmp_path / "nonexistent.cpp", dummy_weights)
    assert isinstance(pred, FallbackPythonEnsembleBackend)
    bad = np.zeros(112, dtype=np.float32)
    bad[5] = np.nan
    with pytest.raises(ValueError):
        pred.predict(SimpleNamespace(seq_ix=1, step_in_seq=0, need_prediction=False, state=bad))
