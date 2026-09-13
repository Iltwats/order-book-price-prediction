from pathlib import Path

import numpy as np
import onnx
import pytest

torch = pytest.importorskip("torch", reason="optional offline training dependency")

from connectome.conversion_audit import (audit_fp64_gradients, export_original_graph,
                                         reference_gru, _difference, _structure_hash)
from connectome.model_io import load_weights
from connectome.training import load_torch_model, StreamingGRU

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline/baseline.onnx"


@pytest.mark.parametrize("source", [BASELINE, ROOT / "artifacts/selected/model.onnx"])
def test_zero_update_replaces_parameters_without_changing_graph(source, tmp_path):
    model = load_torch_model(source)
    destination = export_original_graph(model, source, tmp_path / "roundtrip.onnx")
    assert Path(destination).read_bytes() == source.read_bytes()
    assert _structure_hash(onnx.load(destination)) == _structure_hash(onnx.load(source))
    assert set(load_weights(destination)) == set(load_weights(source))


def test_export_changes_all_eight_tensors_from_torch_values(tmp_path):
    model = load_torch_model(BASELINE)
    originals = load_weights(BASELINE)
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters()):
            parameter.add_((index + 1) * 0.001)
    destination = export_original_graph(model, BASELINE, tmp_path / "changed.onnx")
    restored = load_torch_model(destination)
    for actual, expected in zip(restored.parameters(), model.parameters()):
        assert torch.equal(actual, expected)
    assert all(not np.array_equal(value, originals[key]) for key, value in load_weights(destination).items())
    assert _structure_hash(onnx.load(destination)) == _structure_hash(onnx.load(BASELINE))


def test_export_preserves_tensor_metadata(tmp_path):
    source = tmp_path / "metadata.onnx"
    graph = onnx.load(BASELINE)
    graph.graph.initializer[0].doc_string = "keep this tensor documentation"
    onnx.save(graph, source)
    destination = export_original_graph(load_torch_model(source), source, tmp_path / "output.onnx")
    assert source.read_bytes() == Path(destination).read_bytes()


@pytest.mark.parametrize("kind", ["same", "alias", "half", "nonfinite", "width", "head"])
def test_export_rejects_incompatible_or_unsafe_requests(kind, tmp_path):
    model = load_torch_model(BASELINE)
    destination = tmp_path / "output.onnx"
    if kind == "same":
        destination = BASELINE
    elif kind == "alias":
        destination.symlink_to(BASELINE)
    elif kind == "half":
        model.half()
    elif kind == "nonfinite":
        with torch.no_grad():
            model.gru1.bias_hh_l0[0] = float("nan")
    elif kind == "width":
        model = StreamingGRU(96)
    elif kind == "head":
        model = StreamingGRU(128, "both")
    before = BASELINE.read_bytes()
    with pytest.raises(ValueError):
        export_original_graph(model, BASELINE, destination)
    assert BASELINE.read_bytes() == before


def test_fp64_equations_gradients_and_full_finite_differences():
    torch.set_num_threads(1)
    report = audit_fp64_gradients()
    assert report["passed"], report
    assert len(report["cases"]) == 3
    assert all(len(case["gradients"]) == 6 for case in report["cases"])


def test_chunked_reference_carries_gradient_until_explicit_detach():
    torch.manual_seed(95)
    layer = torch.nn.GRU(3, 2).double()
    x = torch.randn(7, 1, 3, dtype=torch.float64, requires_grad=True)
    h = torch.randn(1, 1, 2, dtype=torch.float64, requires_grad=True)
    parameters = tuple(layer.parameters())
    full, _ = reference_gru(x, h, *parameters)
    prefix, state = reference_gru(x[:3], h, *parameters)
    suffix, _ = reference_gru(x[3:], state, *parameters)
    joined = torch.cat((prefix, suffix))
    torch.testing.assert_close(joined, full, atol=1e-12, rtol=1e-10)
    for a, b in zip(torch.autograd.grad(full[-1].sum(), (x, h, *parameters)),
                    torch.autograd.grad(joined[-1].sum(), (x, h, *parameters))):
        torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-10)
    _, state = reference_gru(x[:3], h, *parameters)
    detached, _ = reference_gru(x[3:], state.detach(), *parameters)
    gradient = torch.autograd.grad(detached[-1].sum(), x)[0]
    assert torch.equal(gradient[:3], torch.zeros_like(gradient[:3]))
    assert torch.count_nonzero(gradient[3:]) > 0


def test_bitwise_comparison_rejects_signed_zero_and_nonfinite_values():
    assert not _difference(np.array([-0.0], np.float32), np.array([0.0], np.float32))["passed"]
    assert not _difference(np.array([np.inf]), np.array([np.inf]))["passed"]
