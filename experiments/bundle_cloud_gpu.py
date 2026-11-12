"""Package a self-contained Cloud GPU training bundle (`dist/cloud_gpu_sorted_lob_bundle.tar.gz`).

Everything needed on a rented CUDA host (Runpod / GCP / Lambda) EXCEPT `train.parquet`
(and optional `valid.parquet`), keeping the archive ~1.5 MB!

Includes:
- `cloud/Dockerfile.scratch` + `cloud/run_gpu_suite.sh`
- `connectome/`, `experiments/`, `native/ensemble_engine.cpp`, `utils.py`, `requirements*.txt`
- Small non-secret metadata required for stratified / mask-propensity training:
  - `artifacts/manifest.json` (split row-group indices)
  - `artifacts/experiments/data_semantics/class_labels.npz` (tick-class labels per row group)
  - `artifacts/experiments/mask_propensity/model.npz` (OOF observable `is_scored` scorer)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from pathlib import Path

REQUIRED_FILES = [
    "requirements.txt",
    "requirements-train.txt",
    "utils.py",
    "cloud/Dockerfile.scratch",
    "cloud/run_gpu_suite.sh",
    "native/ensemble_engine.cpp",
    "artifacts/manifest.json",
    "artifacts/experiments/data_semantics/class_labels.npz",
    "artifacts/experiments/mask_propensity/model.npz",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_cloud_bundle(output_path: str | Path) -> dict:
    root = Path(__file__).resolve().parents[1]
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    rel_paths: list[str] = list(REQUIRED_FILES)
    for pkg in ("connectome", "experiments"):
        for py in sorted((root / pkg).glob("*.py")):
            rel_paths.append(str(py.relative_to(root)))

    missing = [p for p in rel_paths if not (root / p).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files for cloud bundle: {missing}")

    manifest_entries = {}
    tmp_out = out.with_suffix(".tmp.tar.gz")
    with tarfile.open(tmp_out, "w:gz") as tar:
        for rel in sorted(set(rel_paths)):
            src = root / rel
            data = src.read_bytes()
            manifest_entries[rel] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name=rel)
            info.size = len(data)
            info.mtime = 1700000000
            info.mode = 0o755 if rel.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))

        bundle_meta = json.dumps({"files": manifest_entries}, indent=2).encode("utf-8")
        meta_info = tarfile.TarInfo(name="BUNDLE_MANIFEST.json")
        meta_info.size = len(bundle_meta)
        meta_info.mtime = 1700000000
        meta_info.mode = 0o644
        tar.addfile(meta_info, io.BytesIO(bundle_meta))

    tmp_out.replace(out)
    return {
        "bundle": str(out),
        "size_bytes": out.stat().st_size,
        "sha256": sha256_file(out),
        "file_count": len(manifest_entries),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="dist/cloud_gpu_sorted_lob_bundle.tar.gz")
    args = p.parse_args()
    report = build_cloud_bundle(args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
