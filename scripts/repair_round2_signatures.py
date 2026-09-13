"""Repair signatures after the recorded layout amendment, preserving declarations.

Run once under the registry lock. Repeated runs verify the repair and do nothing.
No training settings, sources, reservations, results, or trial caps are changed.
"""
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from connectome import experiments as ex


ARMS = {"head_rho_p1e-4", "head_rho_p1e-3", "head_rho_p1e-2"}
REPAIR_TYPE = "candidate_signature_metadata_repair"


def signature(entry, *, corrected_path=None, original_hash=None):
    """Match register_candidate's canonical fields, optionally restoring v1 code."""
    spec = entry["spec"]
    normalized = {k: v for k, v in spec.items() if k not in {
        "hypothesis", "config_path", "training_report_path", "training_subset_reason"}}
    normalized["config"] = {k: v for k, v in spec["config"].items() if k != "model"}
    sources = deepcopy(entry["source_artifacts"])
    if corrected_path is not None:
        matches = [item for item in sources if item["path"] == corrected_path]
        if len(matches) != 1:
            raise ValueError("expected exactly one amended correlation source")
        matches[0]["sha256"] = original_hash
    normalized["source_hashes"] = sorted(item["sha256"] for item in sources)
    return ex._digest(normalized)


def main():
    registry = ROOT / "artifacts/round2/registry.json"
    with ex._lock(registry) as destination:
        record, _ = ex._load(destination)
        fixes = [item for item in record.get("amendments", [])
                 if item["type"] == "pre_screen_training_bugfix"]
        if len(fixes) != 1 or set(fixes[0]["revised_candidates"]) != ARMS:
            raise ValueError("repair requires the exact recorded three-arm layout amendment")
        fix = fixes[0]
        for key in ("previous_registry", "previous_source", "corrected_source"):
            ex._check_artifact(fix[key])
        archived = ex._read(fix["previous_registry"]["path"])
        if archived["registry_sha256"] != ex._digest({
                k: v for k, v in archived.items() if k != "registry_sha256"}):
            raise ValueError("archived registry digest is invalid")

        changes = {}
        for candidate_id in sorted(ARMS):
            entry = record["candidates"][candidate_id]
            prior = archived["candidates"][candidate_id]
            if entry["spec"] != prior["spec"]:
                raise ValueError("candidate settings changed since the layout amendment")
            for source in entry["source_artifacts"]:
                ex._check_artifact(source)
            old = signature(entry, corrected_path=fix["corrected_source"]["path"],
                            original_hash=fix["previous_source"]["sha256"])
            if prior["signature"] != old or signature(prior) != old:
                raise ValueError("original declaration signature does not match archived sources")
            changes[candidate_id] = {"declaration_signature": old,
                                     "signature": signature(entry)}

        receipts = [item for item in record.get("amendments", [])
                    if item["type"] == REPAIR_TYPE]
        if receipts:
            if len(receipts) != 1 or receipts[0]["candidate_signatures"] != changes:
                raise ValueError("existing signature repair does not match current declarations")
            ex._check_artifact(receipts[0]["repair_script"])
            for candidate_id, values in changes.items():
                entry = record["candidates"][candidate_id]
                if any(entry.get(key) != value for key, value in values.items()):
                    raise ValueError("previously repaired candidate signature changed")
            print("Signature repair already recorded and verified; no changes.")
            return

        before = deepcopy(record)
        for candidate_id, values in changes.items():
            entry = record["candidates"][candidate_id]
            if "declaration_signature" in entry or entry["signature"] != values["declaration_signature"]:
                raise ValueError("repair applies only to unchanged original declaration signatures")
            entry.update(values)

        # Verify the active duplicate-detection keys for every arm, including
        # completed affine fits whose source snapshot retains the original hash.
        signatures = []
        for entry in record["candidates"].values():
            if entry["signature"] != signature(entry):
                raise ValueError("a candidate signature does not match its current source hashes")
            signatures.append(entry["signature"])
        if len(set(signatures)) != len(signatures):
            raise ValueError("duplicate current candidate configurations detected")

        unchanged = deepcopy(record)
        for candidate_id in ARMS:
            unchanged["candidates"][candidate_id].pop("declaration_signature")
            unchanged["candidates"][candidate_id]["signature"] = before["candidates"][candidate_id]["signature"]
        if unchanged != before:
            raise ValueError("unexpected change outside candidate signature metadata")
        record["amendments"].append({
            "at": ex._now(), "type": REPAIR_TYPE,
            "reason": "Recompute duplicate-detection signatures after the recorded contiguous-parameter repair; preserve original declarations separately.",
            "candidate_signatures": changes,
            "repair_script": ex._artifact(__file__),
            "previous_registry_sha256": before["registry_sha256"],
            "layout_amendment_at": fix["at"],
            "sources_specs_reservations_results_caps_unchanged": True,
        })
        ex._write(destination, record)
    print("Repaired three active signatures; original declarations and all evaluation evidence preserved.")


if __name__ == "__main__":
    main()
