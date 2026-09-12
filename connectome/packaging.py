"""Build a small inference-only ZIP and verify it in a fresh process."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import numpy as np

from .evaluation import file_hash, save_json


def package_submission(config_path, output_path, *, verify=True, smoke_features=None):
    config_path, output_path = Path(config_path).resolve(), Path(output_path).resolve()
    config = json.loads(config_path.read_text())
    source_root = Path(__file__).resolve().parents[1]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_source = (config_path.parent / config["model"]).resolve()
    if not model_source.is_file():
        raise FileNotFoundError(model_source)
    with tempfile.TemporaryDirectory(prefix="connectome-package-", dir=output_path.parent) as temporary:
        stage = Path(temporary)
        (stage / "connectome").mkdir()
        for name in ("__init__.py", "runtime.py", "native.py"):
            shutil.copy2(source_root / "connectome" / name, stage / "connectome" / name)
        shutil.copy2(source_root / "solution.py", stage / "solution.py")
        shutil.copy2(model_source, stage / "model.onnx")
        config["model"] = "model.onnx"
        if config.get("backend") == "native":
            weights = (config_path.parent / config["native_weights"]).resolve()
            shutil.copy2(weights, stage / "weights.npy")
            config["native_weights"] = "weights.npy"
            libs = {}
            for platform_key, value in config.get("native_libraries", {}).items():
                source = (config_path.parent / value).resolve()
                if not source.is_file():
                    raise FileNotFoundError(source)
                name = f"native/{platform_key}/{source.name}"
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                libs[platform_key] = name
            config["native_libraries"] = libs
        else:
            for name in ("native_weights", "native_libraries", "native_layout"):
                config.pop(name, None)
        (stage / "submission_config.json").write_text(json.dumps(config, indent=2) + "\n")
        (stage / "MODEL.md").write_text(
            "# Alpha Connectome submission\n\n"
            "Uses supplied competition weights and the two-target streaming callback.\n"
            "All recurrent state and filters reset between sequences.\n"
            "Inference is offline, CPU-only, with one runtime compute thread.\n"
            f"Model SHA256: `{file_hash(stage / 'model.onnx')}`\n"
            "The hidden test volume/CPU is unspecified; runtime compliance needs platform evidence.\n")
        files = sorted(x for x in stage.rglob("*") if x.is_file())
        entries = {str(x.relative_to(stage)): file_hash(x) for x in files}
        (stage / "MANIFEST.json").write_text(json.dumps({"files": entries}, indent=2) + "\n")
        temp_zip = output_path.with_suffix(".tmp.zip")
        with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(x for x in stage.rglob("*") if x.is_file()):
                info = zipfile.ZipInfo(str(path.relative_to(stage)), date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, path.read_bytes())
        size = temp_zip.stat().st_size
        if size > 18_000_000:
            temp_zip.unlink()
            raise ValueError(f"submission exceeds 18 MB engineering limit: {size}")
        result = {"archive": str(output_path), "size_bytes": size,
                  "sha256": file_hash(temp_zip), "files": entries, "verified": False,
                  "judge_runtime_verified": False}
        try:
            if verify:
                result["verification"] = verify_submission(temp_zip, smoke_features=smoke_features)
                result["verified"] = True
            # Never replace a verified release with an unverified candidate.
            temp_zip.replace(output_path)
        finally:
            temp_zip.unlink(missing_ok=True)
        save_json(output_path.with_suffix(".report.json"), result)
    return result


def verify_submission(archive_path, *, smoke_features=None):
    """No imports from the repository are allowed in the extracted smoke process."""
    archive_path = Path(archive_path).resolve()
    with tempfile.TemporaryDirectory(prefix="connectome-verify-") as temporary:
        root = Path(temporary).resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                target = (root / entry.filename).resolve()
                if root not in target.parents:
                    raise ValueError("unsafe archive path")
            if archive.testzip() is not None:
                raise ValueError("archive integrity failure")
            archive.extractall(root)
        if smoke_features is not None:
            features = np.asarray(smoke_features, np.float32)
            if features.ndim != 2 or features.shape[1] != 112 or len(features) < 100:
                raise ValueError("smoke features must contain at least 100 rows of 112 features")
            np.save(root / "smoke.npy", features)
        code = '''
import json, socket, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
def no_network(*a, **kw):
    raise RuntimeError("network disabled in submission verification")
socket.socket = no_network
import numpy as np
from types import SimpleNamespace
from solution import PredictionModel
features = np.load("smoke.npy") if Path("smoke.npy").exists() else np.random.default_rng(0).normal(size=(256,112)).astype("float32")
model = PredictionModel()
all_outputs = []
held = []
for seq in (7, 8, 7):
    values = []
    for step, row in enumerate(features):
        out = model.predict(SimpleNamespace(seq_ix=seq,step_in_seq=step,need_prediction=step>=99,state=row))
        if step < 99:
            assert out is None
        else:
            assert out.shape == (2,) and np.isfinite(out).all()
            held.append((out,out.copy()))
            values.append(out.copy())
    all_outputs.append(np.asarray(values))
np.testing.assert_array_equal(all_outputs[0],all_outputs[1])
np.testing.assert_array_equal(all_outputs[1],all_outputs[2])
for actual, original in held:
    np.testing.assert_array_equal(actual,original)
print(json.dumps({"passed":True,"rows":len(features)*3,"resets":2,"independent_output_ownership":True,"network_disabled":True,"python":sys.version.split()[0]}))
'''
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run([sys.executable, "-I", "-c", code], cwd=root, env=env,
                                   capture_output=True, text=True, timeout=180)
        if completed.returncode:
            raise RuntimeError(f"extracted submission failed: {completed.stderr}\n{completed.stdout}")
        return json.loads(completed.stdout.strip().splitlines()[-1])
