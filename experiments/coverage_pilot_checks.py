"""Health checks for a partially trained v2.1 arm, without reading validation.

The coverage pilot is the first updates of the frozen `coverage4096` arm, run
on CPU with `connectome.gpu_training train --device cpu --max-updates N`. This
script checks that run before it is resumed to its full 512 updates:

- training stayed finite and the device qualification passed;
- the lower GRU is bit-identical to the initial model and the upper GRU/head moved;
- the exported ONNX reloads, the row-by-row callback matches the bulk sequence
  model bitwise, and both match Torch with the checkpoint's weights;
- on unused TRAIN sequences (outside both arms, the reserve and every recorded
  exposure), predictions stay close to the initial model's rather than collapsing.

It reads only TRAIN features; no targets, no validation data.

    .venv/bin/python -m experiments.coverage_pilot_checks \
        --run artifacts/gpu/cpu-v2.1/coverage4096 --output artifacts/experiments/coverage_pilot/report_32.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch

from connectome.data import read_sequence
from connectome.model_io import export_sequence_model, load_weights
from connectome.runtime import OnnxBackend
from connectome.training import load_torch_model

WARMUP = 99
LOWER = ("w0", "r0", "b0")


def bulk(path, x):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    zeros = np.zeros((1, 1, 128), dtype=np.float32)
    return session.run(["prediction"], {"features": x[None], "hidden_0": zeros, "hidden_1": zeros})[0][0]


def callback(path, x):
    backend = OnnxBackend(path, mode="iobinding")
    backend.reset()
    return np.stack([backend.step(row) for row in x])


def torch_predictions(initial, state, x):
    model = load_torch_model(initial).eval()
    model.load_state_dict(state)
    with torch.no_grad():
        prediction, _, _ = model(torch.from_numpy(x)[:, None])  # [time, batch, features]
    return prediction[:, 0].numpy()


def unused_groups(plan, train, count, seed):
    used = set(plan["reserved_train_groups"])
    used.update(g for arm in plan["arms"].values() for g in arm["groups"])
    used.update(g for entry in plan["historical_exposure_inventory"] for g in entry["groups"])
    available = sorted(set(range(pq.ParquetFile(train).num_row_groups)) - used)
    return sorted(np.random.default_rng(seed).choice(available, count, replace=False).tolist()), len(available)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--plan", default="artifacts/gpu/global-wp-v2.1")
    parser.add_argument("--train", default="datasets/train.parquet")
    parser.add_argument("--sequences", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run, plan_dir = Path(args.run), Path(args.plan)
    plan = json.loads((plan_dir / "plan.json").read_text())
    report = json.loads((run / "report.json").read_text())
    qualification = json.loads((run / "qualification.json").read_text())
    initial = plan_dir / "initial.onnx"

    history = report["history"]
    losses = [h["objective_loss"] for h in history]
    training = {
        "updates": report["cursor"]["updates"], "sequences_seen": report["cursor"]["sequences_seen"],
        "qualification_passed": qualification["passed"],
        "all_finite": all(math.isfinite(v) for h in history for v in (h["objective_loss"], h["gradient_norm"])),
        "loss_first_last_8": [float(np.mean(losses[:8])), float(np.mean(losses[-8:]))],
        "gradient_norm_median_max": [float(np.median([h["gradient_norm"] for h in history])),
                                     max(h["gradient_norm"] for h in history)],
        "batch_seconds_median": float(np.median([h["batch_seconds"] for h in history])),
    }

    before, after = load_weights(initial), load_weights(run / "model.onnx")
    weights = {
        "lower_bitwise_equal": all(before[k].tobytes() == after[k].tobytes() for k in LOWER),
        "relative_change": {k: float(np.linalg.norm(after[k] - before[k]) / np.linalg.norm(before[k]))
                            for k in before if k not in LOWER},
    }

    state = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)["model_state"]
    initial_sequence = run.parent / "initial_sequence.onnx"
    if not initial_sequence.exists():
        export_sequence_model(initial, initial_sequence)
    groups, available = unused_groups(plan, args.train, args.sequences, args.seed)
    parquet = pq.ParquetFile(args.train)
    sequences = []
    for group in groups:
        x = read_sequence(parquet, group).features
        new, old = bulk(run / "sequence.onnx", x), bulk(initial_sequence, x)
        streamed, reference = callback(run / "model.onnx", x), torch_predictions(initial, state, x)
        new, old = new[WARMUP:], old[WARMUP:]
        sequences.append({
            "group": group,
            "callback_equals_bulk_bitwise": bool(np.array_equal(streamed[WARMUP:], new)),
            "max_abs_onnx_minus_torch": float(np.abs(new - reference[WARMUP:]).max()),
            "corr_with_initial": [float(np.corrcoef(new[:, j], old[:, j])[0, 1]) for j in range(2)],
            "std_ratio_to_initial": (new.std(0) / old.std(0)).tolist(),
            "mean_abs_change": float(np.abs(new - old).mean()),
            "max_abs_change": float(np.abs(new - old).max()),
            "initial_std": old.std(0).tolist(),
        })
        print(json.dumps(sequences[-1]), flush=True)

    result = {"run": str(run), "plan_sha256": plan["plan_sha256"], "training": training, "weights": weights,
              "unused_train_groups_available": available, "sequences": sequences}
    result["passed"] = bool(
        training["qualification_passed"] and training["all_finite"] and weights["lower_bitwise_equal"]
        and all(v > 0 for v in weights["relative_change"].values())
        and all(s["callback_equals_bulk_bitwise"] and s["max_abs_onnx_minus_torch"] < 1e-4
                and min(s["corr_with_initial"]) > 0.9 and 0.8 < min(s["std_ratio_to_initial"])
                and max(s["std_ratio_to_initial"]) < 1.25 for s in sequences))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("training", "weights", "passed")}, indent=2))


if __name__ == "__main__":
    main()
