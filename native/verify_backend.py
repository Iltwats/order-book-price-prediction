#!/usr/bin/env python3
"""Audit native and simplified-graph parity on complete real sequences.

This records numerical evidence; it never promotes a candidate or interprets
mixed verification-loop elapsed times as deployment benchmarks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

from connectome.model_io import export_native_weights, inspect_model, simplify_model
from connectome.native import NativeBackend
from utils import FEATURE_COLUMNS, SEQUENCE_LENGTH, validate_sequence


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "valid.parquet")
    parser.add_argument("--model", type=Path, default=ROOT / "baseline" / "baseline.onnx")
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--groups", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "native" / "parity.json")
    args = parser.parse_args()
    artifact_dir = args.output.resolve().parent
    artifact_dir.mkdir(parents=True, exist_ok=True)
    weights_path = artifact_dir / "weights.npy"
    simplified_path = artifact_dir / "simplified.onnx"
    export_native_weights(args.model, weights_path)
    simplify_model(args.model, simplified_path)
    original = session(args.model)
    simplified = session(simplified_path)
    parquet = pq.ParquetFile(args.dataset)
    report = {
        "reference": inspect_model(args.model), "simplified": inspect_model(simplified_path),
        "library": str(args.library.resolve()), "ort_version": ort.__version__,
        "groups": [], "rtol": 1e-5, "atol": 1e-6,
        "promoted": False, "timings_are_deployment_benchmarks": False,
    }
    output_names = ["prediction", "next_hidden_0", "next_hidden_1"]
    with NativeBackend(weights_path, args.library, "packed") as packed, NativeBackend(weights_path, args.library, "row") as row:
        # Exercise owned output behavior and invalid shapes independently of data.
        owned = packed.step(np.ones(112, dtype=np.float32))
        snapshot = owned.copy()
        packed.step(np.zeros(112, dtype=np.float32))
        if not owned.flags.owndata or not np.array_equal(owned, snapshot):
            raise AssertionError("native outputs do not own stable memory")
        for group in args.groups:
            table = parquet.read_row_group(group, use_threads=False)
            sequence_id, _ = validate_sequence(table)
            features = np.column_stack([table[name].to_numpy() for name in FEATURE_COLUMNS]).astype(np.float32)
            first = np.zeros((1, 1, 128), np.float32)
            second = np.zeros_like(first)
            simple_first, simple_second = first.copy(), second.copy()
            packed.reset()
            row.reset()
            result = {
                "group": group, "seq_ix": sequence_id, "rows": SEQUENCE_LENGTH,
                "native_prediction_max_abs": 0.0, "native_hidden_max_abs": 0.0,
                "native_prediction_tolerance_failures": 0, "native_hidden_tolerance_failures": 0,
                "simplified_prediction_max_abs": 0.0, "simplified_hidden_max_abs": 0.0,
                "simplified_bitwise_equal": True, "native_layout_bitwise_equal": True,
            }
            start = time.perf_counter()
            for step, feature in enumerate(features):
                expected, first, second = original.run(output_names, {"features": feature.reshape(1, 1, 112), "hidden_0": first, "hidden_1": second})
                actual_simple, simple_first, simple_second = simplified.run(output_names, {"features": feature.reshape(1, 1, 112), "hidden_0": simple_first, "hidden_1": simple_second})
                actual, actual_row = packed.step(feature), row.step(feature)
                native_hidden = packed.hidden_states()
                expected = expected.reshape(2)
                result["native_prediction_max_abs"] = max(result["native_prediction_max_abs"], float(np.max(np.abs(actual - expected))))
                result["native_prediction_tolerance_failures"] += int(not np.allclose(actual, expected, rtol=1e-5, atol=1e-6))
                for candidate, reference in zip(native_hidden, (first, second)):
                    result["native_hidden_max_abs"] = max(result["native_hidden_max_abs"], float(np.max(np.abs(candidate - reference))))
                    result["native_hidden_tolerance_failures"] += int(not np.allclose(candidate, reference, rtol=1e-5, atol=1e-6))
                result["native_layout_bitwise_equal"] &= bool(np.array_equal(actual, actual_row))
                result["simplified_prediction_max_abs"] = max(result["simplified_prediction_max_abs"], float(np.max(np.abs(actual_simple.reshape(2) - expected))))
                for candidate, reference in zip((actual_simple.reshape(2), simple_first, simple_second), (expected, first, second)):
                    result["simplified_bitwise_equal"] &= bool(np.array_equal(candidate, reference))
                for candidate, reference in zip((simple_first, simple_second), (first, second)):
                    result["simplified_hidden_max_abs"] = max(result["simplified_hidden_max_abs"], float(np.max(np.abs(candidate - reference))))
            result["verification_seconds"] = time.perf_counter() - start
            report["groups"].append(result)
            print(json.dumps(result), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
