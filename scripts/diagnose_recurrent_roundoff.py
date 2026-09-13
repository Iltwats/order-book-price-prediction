"""Investigate the already-failed TRAIN sequence; no training or gate changes.

Run from the repository root with ``python -m scripts.diagnose_recurrent_roundoff``.
This is explanatory evidence after a failed test, not a new acceptance test.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from connectome import numerical_bridge as bridge
from connectome.data import file_fingerprint, iter_sequences
from connectome.training import OnnxSequenceEncoder, load_torch_model


def explicit_upper(layer, inputs, *, formula="onnx"):
    """Same packed affine/sigmoid/tanh operations; only state algebra differs."""
    if formula not in {"onnx", "candidate_centered", "previous_centered"}:
        raise ValueError("unknown state-update formula")
    with torch.inference_mode():
        input_gates = F.linear(inputs, layer.weight_ih_l0, layer.bias_ih_l0)
        state = torch.zeros(layer.hidden_size, dtype=inputs.dtype)
        output = torch.empty((len(inputs), layer.hidden_size), dtype=inputs.dtype)
        for index, gates in enumerate(input_gates):
            recurrent = F.linear(state, layer.weight_hh_l0, layer.bias_hh_l0)
            ir, iz, inn = gates.chunk(3)
            hr, hz, hn = recurrent.chunk(3)
            reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            if formula == "onnx":
                state = (1. - update) * candidate + update * state
            elif formula == "candidate_centered":
                state = candidate + update * (state - candidate)
            else:
                state = state + (1. - update) * (candidate - state)
            output[index] = state
    return output.numpy()


def reanchored_upper(model, inputs, reference, chunk=256):
    """New diagnostic path: ORT h1 immediately before each chunk, ORT h0 inputs."""
    output = np.empty_like(reference)
    with torch.inference_mode():
        for start in range(0, len(inputs), chunk):
            initial = torch.zeros(1, 1, model.hidden_size) if start == 0 else torch.from_numpy(reference[start - 1])[None, None]
            observed, _ = model.gru1(torch.from_numpy(inputs[start:start + chunk])[:, None], initial)
            output[start:start + chunk] = observed[:, 0].numpy()
    return output


def _stats(error):
    error = np.asarray(error, dtype=np.float64)
    return {"max_absolute": float(np.abs(error).max()), "rms": float(np.sqrt(np.mean(error ** 2))),
            "mean_signed": float(error.mean())}


def _duration(mask):
    indices = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    durations = indices[1::2] - indices[::2]
    return {"rows": int(mask.sum()), "fraction": float(np.mean(mask)),
            "longest_run": int(durations.max(initial=0)), "runs": int(len(durations))}


def profile_states(observed, reference, coords):
    error = observed.astype(np.float64) - reference.astype(np.float64)
    coordinate_rms = np.sqrt(np.mean(error ** 2, axis=0))
    segments = {}
    for size in (256, 1024):
        segments[str(size)] = [{"start": start, "rows": len(error[start:start + size]),
            "all": _stats(error[start:start + size]),
            "coordinates": {str(c): _stats(error[start:start + size, c]) for c in coords}}
            for start in range(0, len(error), size)]
    coordinate_reports = {}
    for c in coords:
        e, h = error[:, c], reference[:, c]
        coordinate_reports[str(c)] = {**_stats(e), "first": float(e[0]), "last": float(e[-1]),
            "reference_range": [float(h.min()), float(h.max())],
            "strict_exceedance": _duration(np.abs(e) > 1e-6 + 1e-5 * np.abs(h)),
            "new_rms_budget_exceedance": _duration(np.abs(e) > bridge.BUDGET["carried_state"]["normalized_rms"]),
            "positive_error_fraction": float(np.mean(e > 0)),
            "error_increment": _stats(np.diff(e))}
    return {"all": _stats(error), "worst_coordinate_rms": float(coordinate_rms.max()),
            "worst_coordinate": int(coordinate_rms.argmax()), "coordinates": coordinate_reports,
            "segments": segments}


def gate_retention(layer, inputs, reference, coords):
    previous = np.concatenate((np.zeros_like(reference[:1]), reference[:-1]))
    with torch.inference_mode():
        double = copy.deepcopy(layer).double()
        input_gates = F.linear(torch.from_numpy(inputs).double(), double.weight_ih_l0, double.bias_ih_l0)
        recurrent = F.linear(torch.from_numpy(previous).double(), double.weight_hh_l0, double.bias_hh_l0)
        _, iz, _ = input_gates.chunk(3, dim=-1)
        _, hz, _ = recurrent.chunk(3, dim=-1)
        z = torch.sigmoid(iz + hz).numpy()
        fp32_gates = F.linear(torch.from_numpy(inputs), layer.weight_ih_l0, layer.bias_ih_l0)
        fp32_recurrent = F.linear(torch.from_numpy(previous), layer.weight_hh_l0, layer.bias_hh_l0)
        z32 = torch.sigmoid(fp32_gates[:, layer.hidden_size:2 * layer.hidden_size] +
                           fp32_recurrent[:, layer.hidden_size:2 * layer.hidden_size]).numpy()
    return {str(c): {"z_quantiles": np.quantile(z[:, c], [0, .01, .5, .99, 1]).tolist(),
            "one_minus_z_quantiles": np.quantile(1. - z[:, c], [0, .01, .5, .99, 1]).tolist(),
            "fp32_exact_one": _duration(z32[:, c] == 1.),
            "z_gt_0_9999": _duration(z[:, c] > .9999),
            "z_gt_0_999999": _duration(z[:, c] > .999999),
            "product_z_256_quantiles": np.quantile(
                [np.prod(z[start:start + 256, c]) for start in range(0, len(z), 256)], [0, .5, 1]).tolist()}
            for c in coords}


def readout_projection(model, observed, reference, prediction, reference_prediction, coords):
    weight = model.head.weight.detach().numpy().astype(np.float64)
    error = observed.astype(np.float64) - reference.astype(np.float64)
    projected = error @ weight.T
    selected = error[:, coords] @ weight[:, coords].T
    actual = prediction.astype(np.float64) - reference_prediction.astype(np.float64)
    return {"all_state_projection": [_stats(projected[:, side]) for side in range(2)],
            "selected_coordinates_projection": [_stats(selected[:, side]) for side in range(2)],
            "head_rounding_remainder": [_stats(actual[:, side] - projected[:, side]) for side in range(2)],
            "per_coordinate": {str(c): {"head_weights": weight[:, c].tolist(),
                "target_contribution": [_stats(error[:, c] * weight[side, c]) for side in range(2)]} for c in coords},
            "note": "Signed projections may cancel; coordinate contribution magnitudes are not additive percentages"}


def perturbation_probes(layer, inputs, reference, observed, coords):
    double = copy.deepcopy(layer).double()
    previous = np.concatenate((np.zeros_like(reference[:1]), reference[:-1]))
    error = observed.astype(np.float64) - reference.astype(np.float64)
    rows = sorted(set([0, 255, 1023, 9999, 19999, int(np.unravel_index(np.abs(error).argmax(), error.shape)[0])]))
    reports = []
    for row in rows:
        x = torch.from_numpy(inputs[row]).double()[None, None]
        state = torch.from_numpy(previous[row]).double().requires_grad_()

        def forward(h):
            return double(x, h[None, None])[0][0, 0]

        jacobian = torch.autograd.functional.jacobian(forward, state).detach().numpy()
        e = torch.from_numpy(error[row - 1] if row else np.zeros(layer.hidden_size))
        with torch.inference_mode():
            exact_delta = (forward(state + e) - forward(state)).numpy()
        linear_delta = jacobian @ e.numpy()
        reports.append({"row": row, "jacobian_diagonal": {str(c): float(jacobian[c, c]) for c in coords},
            "jacobian_row_l1": {str(c): float(np.abs(jacobian[c]).sum()) for c in coords},
            "spectral_norm": float(np.linalg.norm(jacobian, ord=2)),
            "existing_perturbation_norm": float(np.linalg.norm(e.numpy())),
            "propagated_perturbation_norm": float(np.linalg.norm(exact_delta)),
            "linearization_remainder": _stats(exact_delta - linear_delta)})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-report", default="artifacts/round2/recurrent_bridge/report.json")
    parser.add_argument("--output", default="artifacts/round2/recurrent_roundoff_diagnostic")
    args = parser.parse_args()
    failed = bridge._read(args.failed_report)
    protocol = bridge._verify_protocol(failed["protocol_path"])
    if failed["completed_sequences"] != 1 or failed["approximate_training_accepted"]:
        raise ValueError("expected preserved initial failure on exactly one TRAIN sequence")
    group = failed["sequences"][0]["row_group"]
    if group != protocol["groups"][0] or group != 2309:
        raise ValueError("diagnostic is restricted to the same failed TRAIN group")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    registration = {"kind": "post_failure_explanatory_diagnostic_not_acceptance",
        "group": group, "failed_report_sha256": bridge._sha(args.failed_report),
        "frozen_bridge_sha256": bridge._sha(bridge.__file__), "script_sha256": bridge._sha(__file__),
        "unchanged_budget": bridge.BUDGET,
        "new_path": "upper Torch GRU fed ORT h0; initial h1 reanchored to ORT immediately before every 256-row chunk",
        "scope": "One previously failed TRAIN sequence only; no optimizer, other groups, validation or gate changes"}
    bridge._write(output / "diagnostic_spec.json", registration, exclusive=True)
    if file_fingerprint(protocol["train_path"]) != protocol["train_fingerprint"]:
        raise ValueError("training dataset changed")
    torch.set_num_threads(1)
    model = load_torch_model(protocol["model_path"])
    encoder = OnnxSequenceEncoder(protocol["model_path"], output / "encoder.onnx")
    started = time.monotonic()
    sequence = next(iter_sequences(protocol["train_path"], [group]))
    if sequence.seq_ix != failed["sequences"][0]["seq_ix"]:
        raise ValueError("failed TRAIN sequence identity changed")
    reference = encoder(sequence.features)
    paths = bridge.torch_paths(model, sequence.features, reference)
    upper = paths["upper_on_ort_lower"][1]
    ranking = np.argsort(np.sqrt(np.mean((upper.astype(np.float64) - reference[1]) ** 2, axis=0)))[::-1]
    coords = ranking[:8].tolist()
    report = {"specification": registration, "seq_ix": sequence.seq_ix, "rows": len(sequence.features),
              "top_h1_coordinates_by_rms": coords, "experiments": {}, "claim": "diagnostic only; original bridge remains failed"}
    with torch.inference_mode():
        double = copy.deepcopy(model).double()
        fp64_upper, _ = double.gru1(torch.from_numpy(reference[0]).double()[:, None])
        fp64_upper = fp64_upper[:, 0].numpy()
        reanchored = reanchored_upper(model, reference[0], reference[1])
        additions = {"fp64_carried_upper": fp64_upper, "ort_reanchored256": reanchored}
        for formula in ("onnx", "candidate_centered", "previous_centered"):
            additions[f"explicit_fp32_{formula}"] = explicit_upper(model.gru1, torch.from_numpy(reference[0]), formula=formula)
        for name, h1 in additions.items():
            prediction = F.linear(torch.from_numpy(h1), double.head.weight, double.head.bias).numpy() if h1.dtype == np.float64 else model.head(torch.from_numpy(h1)).numpy()
            paths[name] = (reference[0], h1, prediction)
    for name, values in paths.items():
        h1, prediction = values[1:]
        limits = bridge.BUDGET["common_state" if name == "common_state" else "carried_state"]
        report["experiments"][name] = {"h1_vs_ort": bridge.error_metrics(h1, reference[1], limits),
            "prediction_vs_ort": bridge.error_metrics(prediction, reference[2], limits),
            "h1_vs_fp64_carried": _stats(h1.astype(np.float64) - fp64_upper),
            "wp": bridge.wp_comparison(sequence, prediction, reference[2]),
            "state_profile": profile_states(h1, reference[1], coords),
            "readout_projection": readout_projection(model, h1, reference[1], prediction, reference[2], coords)}
    report["gate_retention_at_common_ort_states"] = gate_retention(model.gru1, reference[0], reference[1], coords)
    report["jacobian_probes_at_common_ort_states"] = perturbation_probes(model.gru1, reference[0], reference[1], upper, coords)
    report["seconds"] = time.monotonic() - started
    if bridge._sha(args.failed_report) != registration["failed_report_sha256"] or bridge._sha(bridge.__file__) != registration["frozen_bridge_sha256"]:
        raise ValueError("preserved bridge evidence changed during diagnostic")
    bridge._write(output / "report.json", report)
    print(json.dumps({"coordinates": coords, "seconds": report["seconds"],
        "experiments": {name: {"h1": result["h1_vs_ort"], "prediction": result["prediction_vs_ort"], "wp": result["wp"]["drift"]}
                        for name, result in report["experiments"].items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
