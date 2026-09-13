"""One predeclared training-subset sensitivity check of the round-two finalist."""
from copy import deepcopy
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from connectome import experiments
from connectome import head_cycle as cycle
from connectome.correlation import extract_moments

CANDIDATE = "affine_rho_subset512"
MOMENTS = cycle.ROUND / "moments_repeat512"


def declare():
    registry = cycle.ROUND / "registry.json"
    record, _ = experiments._load(registry)
    parent = record["candidates"]["affine_rho"]
    if parent["evaluations"]["dev"]["delta"] <= 0:
        raise ValueError("repeat requires the completed positive full-development finalist")
    original = parent["spec"]["training_groups"]
    indices = sorted(np.random.default_rng(20260913).choice(len(original), 512, replace=False).tolist())
    spec = deepcopy(parent["spec"])
    spec.update(
        hypothesis="Training-subset sensitivity of the affine correlation finalist; same objective and hyperparameters",
        training_groups=[original[i] for i in indices], seed=20260913,
        training_subset_reason=("One prespecified draw of 512 of the original 1024 training sequences without replacement, "
                               "seed 20260913, preserving original group order. This is an overlapping training-subset "
                               "sensitivity check, not an independent holdout or a new stochastic optimizer seed."),
        robustness_of="affine_rho",
        robustness_protocol="Run screen and full development once regardless of screen delta if correctness passes; do not redraw the subset.",
        config_path=str(cycle.ROUND / CANDIDATE / "config.json"),
        training_report_path=str(cycle.ROUND / CANDIDATE / "report.json"),
        source_paths=[str(p) for p in cycle._sources()] + [str(Path(__file__).resolve()), str(ROOT / "connectome/selection.py")],
    )
    experiments.register_candidate(registry, CANDIDATE, spec)
    print("Declared one 512-sequence repeat, one screen and one full-development check; original caps unchanged.", flush=True)


def run():
    _, _, entry = cycle._entry(CANDIDATE)
    extract_moments(cycle.SOURCE, cycle.TRAIN, entry["spec"]["training_groups"], MOMENTS)
    cycle.fit(CANDIDATE, moments_path=MOMENTS)
    cycle.run_evaluation(CANDIDATE, "screen")
    cycle.run_evaluation(CANDIDATE, "dev")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["declare", "run"])
    args = parser.parse_args()
    declare() if args.command == "declare" else run()
