"""Parity check and the single development evaluation of a completed v2.1 arm.

This follows the plan's evaluation policy: one full-development evaluation per
completed arm (the 256-sequence screen is a subset of it and is read from the
same cache), no epoch or postprocessing search, and no confirmation access.

1. Byte parity on the 32 smoke sequences: row-by-row callback against the bulk
   sequence model (`connectome.recurrent_parity`), as for the incumbent.
2. Bulk evaluation on the 1,400 development sequences, writing a raw-prediction
   cache for the calibration and comparison step.

    .venv/bin/python -m experiments.coverage_evaluate --run artifacts/gpu/cpu-v2.1/coverage4096
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from connectome.evaluation import evaluate
from connectome.recurrent_parity import check_parity

ROOT = Path(__file__).resolve().parents[1]
VALID = ROOT / "datasets/valid.parquet"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    run = Path(args.run)
    report = json.loads((run / "report.json").read_text())
    if not report["complete"]:
        raise ValueError("evaluate completed arms only")
    for name in ("parity.json", "dev_report.json", "dev_cache"):
        if (run / name).exists():
            raise FileExistsError(f"{run / name} exists; each arm is evaluated once")
    splits = json.loads((ROOT / "artifacts/manifest.json").read_text())["splits"]
    model, sequence = run / "model.onnx", run / "sequence.onnx"
    parity = check_parity(VALID, model, groups=splits["smoke"], mode="iobinding", bulk_model=sequence,
                          report_path=run / "parity.json")
    if not parity["passed"]:
        raise ValueError("callback/bulk parity failed")
    result = evaluate(VALID, model, splits["dev"], "iobinding", run / "dev_cache", run / "dev_report.json",
                      None, sequence)
    print(json.dumps({"run": str(run), "weighted_pearson": result["weighted_pearson"],
                      "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
