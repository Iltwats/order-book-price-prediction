"""Fit the output calibrator on all development scored rows and append it to ONNX graphs.

The calibrator q + W2 gelu(W1 (q - mu) / sd + b1) + b2, with q = clip(p, -c, c), is folded into a few
standard ONNX nodes after the existing `prediction` output, so the streaming
runtime, solution.py and packaging stay unchanged. Writes the deployment step
model, the offline sequence model, the fitted parameters and a verification
report comparing ONNX outputs with the Torch reference on every cached
development row.

    .venv/bin/python -m experiments.export_calibrated --hidden 64 --epochs 30 --seed 0 --clip 2 \
        --output artifacts/calibrated/correlation1024_mlp64_clip2
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

from connectome.evaluation import file_hash
from experiments.devdata import DEFAULT_CACHE, DEFAULT_VALID, WARMUP, load_predictions, load_targets
from experiments.output_calibration import apply, fit_mlp

BASE = Path("artifacts/round2/correlation_gru1024_seed20260912")


def append_calibrator(source, target, params):
    """Rename the producer of `prediction` and recompute `prediction` through the calibrator."""
    model = onnx.load(source)
    graph = model.graph
    if [o.name for o in graph.output][0] != "prediction":
        raise ValueError("expected `prediction` as the first graph output")
    for node in graph.node:
        node.output[:] = ["cal_raw_prediction" if name == "prediction" else name for name in node.output]
    scale = params["w1"] / params["sd"][None, :]                      # [hidden, 2]
    constants = {
        "cal_w1": scale.T.astype(np.float32),                         # [2, hidden]
        "cal_b1": (params["b1"] - scale @ params["mu"]).astype(np.float32),
        "cal_w2": params["w2"].T.astype(np.float32),                  # [hidden, 2]
        "cal_b2": params["b2"].astype(np.float32),
        "cal_inv_sqrt2": np.asarray(1 / np.sqrt(2), np.float32),
        "cal_one": np.asarray(1, np.float32),
        "cal_half": np.asarray(.5, np.float32),
        "cal_low": np.asarray(-params["clip"], np.float32),
        "cal_high": np.asarray(params["clip"], np.float32),
    }
    graph.initializer.extend(numpy_helper.from_array(v, k) for k, v in constants.items())
    graph.node.extend([
        helper.make_node("Clip", ["cal_raw_prediction", "cal_low", "cal_high"], ["cal_input"]),
        helper.make_node("MatMul", ["cal_input", "cal_w1"], ["cal_h0"]),
        helper.make_node("Add", ["cal_h0", "cal_b1"], ["cal_h"]),
        helper.make_node("Mul", ["cal_h", "cal_inv_sqrt2"], ["cal_scaled"]),
        helper.make_node("Erf", ["cal_scaled"], ["cal_erf"]),
        helper.make_node("Add", ["cal_erf", "cal_one"], ["cal_erf1"]),
        helper.make_node("Mul", ["cal_h", "cal_erf1"], ["cal_gelu2"]),
        helper.make_node("Mul", ["cal_gelu2", "cal_half"], ["cal_gelu"]),
        helper.make_node("MatMul", ["cal_gelu", "cal_w2"], ["cal_d0"]),
        helper.make_node("Add", ["cal_d0", "cal_b2"], ["cal_delta"]),
        helper.make_node("Add", ["cal_input", "cal_delta"], ["prediction"]),
    ])
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, target)


def calibrator_only(params, path):
    """A standalone [rows, 2] -> [rows, 2] graph with the same nodes, for bulk verification."""
    graph = helper.make_graph([helper.make_node("Identity", ["raw"], ["prediction"])], "calibrator",
                              [helper.make_tensor_value_info("raw", onnx.TensorProto.FLOAT, ["rows", 2])],
                              [helper.make_tensor_value_info("prediction", onnx.TensorProto.FLOAT, ["rows", 2])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, path)
    append_calibrator(path, path, params)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--valid", default=str(DEFAULT_VALID))
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clip", type=float, default=2.0, help="calibrator input range; the metric clips at 2")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base, output = Path(args.base), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    raw, groups = load_predictions(args.cache)
    raw = np.asarray(raw, np.float32)
    targets, scored = load_targets(groups, args.valid, store="artifacts/experiments/dev_targets.npz")
    params = fit_mlp(raw[scored], np.clip(targets[scored], -2, 2), args.hidden, args.epochs, args.seed,
                     clip=args.clip)
    np.savez(output / "calibrator.npz", **{k: v for k, v in params.items() if k != "kind"})

    append_calibrator(base / "model.onnx", output / "model.onnx", params)
    append_calibrator(base / "sequence.onnx", output / "sequence.onnx", params)
    calibrator_only(params, output / "calibrator.onnx")
    config = json.loads((base / "config.json").read_text())
    config["model"] = "model.onnx"
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # Every cached development row through the calibrator graph versus the Torch reference.
    session = ort.InferenceSession(str(output / "calibrator.onnx"), providers=["CPUExecutionProvider"])
    flat = raw[:, WARMUP:].reshape(-1, 2)
    reference = apply(params, flat)
    onnx_out = np.concatenate([session.run(None, {"raw": flat[i:i + (1 << 20)]})[0]
                               for i in range(0, len(flat), 1 << 20)])
    report = {
        "fit": {"rows": int(scored.sum()), "sequences": len(groups), "hidden": args.hidden,
                "epochs": args.epochs, "seed": args.seed, "input_clip": args.clip, "loss": "pooled |y|-weighted MSE, clipped targets",
                "fit_domain": "development scored rows (need_prediction AND is_scored)"},
        "base_model_sha256": file_hash(base / "model.onnx"),
        "model_sha256": file_hash(output / "model.onnx"),
        "sequence_sha256": file_hash(output / "sequence.onnx"),
        "verification": {"rows": len(flat), "max_abs_error_vs_torch": float(np.abs(onnx_out - reference).max()),
                         "mean_abs_change": float(np.abs(reference - flat).mean()),
                         "max_abs_change": float(np.abs(reference - flat).max())},
    }
    (output / "export.json").write_text(json.dumps(report, indent=2) + "\n")
    shutil.copy2(__file__, output / "export_calibrated.py")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
