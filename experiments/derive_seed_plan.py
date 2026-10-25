"""Copy a frozen v2 training plan with only its seed changed.

`connectome.gpu_training prepare` would redraw the added coverage groups,
because it counts every earlier plan's allocations as exposure. A second seed
of the same experiment must keep the same groups, reserve, models and settings,
and change only the batch order (`seed + epoch`) and the Torch seed. The copy is
re-digested so the runner's `load_plan` integrity checks still apply.

    .venv/bin/python -m experiments.derive_seed_plan --plan artifacts/gpu/global-wp-v2.1 \
        --seed 20261124 --output artifacts/gpu/global-wp-v2.1-seed20261124
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from connectome.gpu_training import digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--plan", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source, destination = Path(args.plan), Path(args.output)
    plan = read_json(source / "plan.json")
    if plan["plan_sha256"] != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("source plan changed")
    epochs = max(arm["epochs"] for arm in plan["arms"].values())
    if abs(args.seed - plan["seed"]) < epochs:
        raise ValueError("epoch orders use seed + epoch; choose a seed that cannot reuse an epoch order")
    destination.mkdir(parents=True)  # never overwrite a frozen plan
    for name in ("initial.onnx", "teacher.onnx", "selection.json"):
        shutil.copy2(source / name, destination / name)
    derived = {k: v for k, v in plan.items() if k != "plan_sha256"}
    derived["seed"] = args.seed
    derived["derived_from"] = {"plan_sha256": plan["plan_sha256"], "change": "seed only"}
    derived["plan_sha256"] = digest(derived)
    write_json(destination / "plan.json", derived)
    print(derived["plan_sha256"])


if __name__ == "__main__":
    main()
