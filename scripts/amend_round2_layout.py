"""Record the observed, pre-screen LBFGS layout repair without resetting trials.

One-time recovery for the archived round-two registry. Existing evaluated fits
retain their original source hash through a byte-identical source snapshot.
Only the three unfitted hidden-state arms receive the corrected source hash.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connectome import experiments as ex


def main():
    registry = ROOT / "artifacts/round2/registry.json"
    backup = registry.with_name("registry_before_layout_fix.json")
    original = ROOT / "artifacts/round2/source_v1/correlation.py"
    corrected = ROOT / "connectome/correlation.py"
    old_text = original.read_text()
    before = "    beta = beta0.clone().requires_grad_()\n"
    after = ("    # Cached ONNX head transposes can be Fortran-contiguous. LBFGS flattens\n"
             "    # gradients with view(), so its optimized parameter must be contiguous.\n"
             "    beta = beta0.contiguous().clone().requires_grad_()\n")
    if old_text.count(before) != 1 or corrected.read_text() != old_text.replace(before, after):
        raise ValueError("recovery permits only the documented contiguous-parameter repair")
    pending = {"head_rho_p1e-4", "head_rho_p1e-3", "head_rho_p1e-2"}
    with ex._lock(registry) as destination:
        if destination.read_bytes() != backup.read_bytes():
            raise ValueError("registry changed since the recorded fitting failure")
        record, _ = ex._load(destination)
        for candidate_id in pending:
            entry = record["candidates"][candidate_id]
            if entry["evaluations"] or Path(entry["spec"]["config_path"]).exists() or Path(entry["spec"]["training_report_path"]).exists():
                raise ValueError("cannot revise an evaluated or completed fit")
        old_hash = ex.file_hash(original)
        new_hash = ex.file_hash(corrected)

        def revise(sources, use_corrected=False):
            matched = 0
            for item in sources:
                if item["path"] == str(corrected):
                    if item["sha256"] != old_hash:
                        raise ValueError("archived source does not match original declaration")
                    matched += 1
                    if use_corrected:
                        item["sha256"] = new_hash
                    else:
                        item["original_path"] = item["path"]
                        item["path"] = str(original)
                ex._check_artifact(item)
            if matched != 1:
                raise ValueError("expected one correlation source per declaration")

        revise(record["source_artifacts"])
        for candidate_id, entry in record["candidates"].items():
            revise(entry["source_artifacts"], candidate_id in pending)
        event = {
            "at": ex._now(), "type": "pre_screen_training_bugfix",
            "reason": "First hidden-state fit failed in LBFGS gradient view before producing a model or validation score; cached head transpose has noncontiguous strides.",
            "failure": "RuntimeError: view size is not compatible with input tensor's size and stride",
            "failed_fit": "head_rho_p1e-4", "revised_candidates": sorted(pending),
            "change": "Make optimized head parameter contiguous; objective, values, data, seed, solver settings and validation groups unchanged.",
            "previous_registry": ex._artifact(backup), "previous_source": ex._artifact(original),
            "corrected_source": ex._artifact(corrected), "recovery_script": ex._artifact(__file__),
            "supplemental_validation_source": ex._artifact(ROOT / "connectome/selection.py"),
            "completed_evaluations_preserved": True, "caps_unchanged": True,
            "screen_slot_consumed_by_fit_failure": False,
            "slot_reason": "No evaluation reservation or validation output existed for the failed fit."
        }
        record.setdefault("amendments", []).append(event)
        ex._write(destination, record)
    print("Recorded layout repair for three unscored arms; two completed screens and all caps preserved.")


if __name__ == "__main__":
    main()
