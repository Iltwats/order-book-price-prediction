"""Where does per-row callback time go? Splits it into Python, ONNX Runtime and GRU arithmetic.

Runs one validation sequence (20,000 rows) on one thread:
  1. the complete solution callback (what the judge times),
  2. the same callback with a do-nothing backend (Python and validation overhead),
  3. ONNX Runtime step calls alone (per-call overhead plus arithmetic),
  4. the offline sequence model over all rows in one call (arithmetic with no per-call overhead).

    OMP_NUM_THREADS=1 .venv/bin/python -m experiments.latency_breakdown --model artifacts/calibrated/correlation1024_mlp64_clip2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq

from connectome.runtime import OnnxBackend, StreamingPredictor, session_options
from utils import FEATURE_COLUMNS


class NullBackend:
    def __init__(self):
        self.out = np.zeros(2, np.float32)

    def reset(self):
        pass

    def step(self, features):
        return self.out.copy()


def per_row(fn, rows, repeats=3):
    """Best of several timed passes, in microseconds per row."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best / rows * 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="directory with model.onnx and sequence.onnx")
    parser.add_argument("--valid", default="datasets/valid.parquet")
    parser.add_argument("--group", type=int, default=0)
    args = parser.parse_args()
    model = Path(args.model)
    table = pq.ParquetFile(args.valid).read_row_group(args.group, columns=FEATURE_COLUMNS)
    x = np.column_stack([table[c].to_numpy() for c in FEATURE_COLUMNS]).astype(np.float32)
    points = [SimpleNamespace(seq_ix=7, step_in_seq=t, need_prediction=t >= 99, state=x[t]) for t in range(len(x))]

    def callbacks(predictor):
        def run():
            predictor.sequence = None
            for point in points:
                predictor.predict(point)
        return run

    backend = OnnxBackend(model / "model.onnx", "iobinding")

    def steps():
        backend.reset()
        for row in x:
            backend.step(row)

    sequence = ort.InferenceSession(str(model / "sequence.onnx"), sess_options=session_options(),
                                    providers=["CPUExecutionProvider"])
    feed = {"features": x[None], "hidden_0": np.zeros((1, 1, 128), np.float32),
            "hidden_1": np.zeros((1, 1, 128), np.float32)}
    result = {
        "full_callback_us": per_row(callbacks(StreamingPredictor(backend)), len(x)),
        "python_callback_overhead_us": per_row(callbacks(StreamingPredictor(NullBackend())), len(x)),
        "onnx_step_only_us": per_row(steps, len(x)),
        "gru_arithmetic_one_call_us": per_row(lambda: sequence.run(None, feed), len(x)),
        "rows": len(x), "onnxruntime": ort.__version__, "threads": 1,
    }
    result["onnx_call_overhead_us"] = result["onnx_step_only_us"] - result["gru_arithmetic_one_call_us"]
    print(json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in result.items()}, indent=1))


if __name__ == "__main__":
    main()
