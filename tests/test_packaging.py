"""Extracted archive tests: clean imports, startup fallback and callback ownership."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np
import pytest

from connectome.packaging import package_submission, verify_submission

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline" / "baseline.onnx"


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps({"model": str(BASELINE), "backend": "onnx", "onnx_mode": "run",
                                "postprocess": {"ema_half_life": 8, "ema_blend": 0.1}}))
    return path


@pytest.fixture
def smoke_features():
    return np.random.default_rng(143).normal(0, 0.2, size=(107, 112)).astype(np.float32)


def rewrite_entry(source, destination, name, content):
    with zipfile.ZipFile(source) as original:
        files = {entry.filename: original.read(entry.filename) for entry in original.infolist()}
    files[name] = content.encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, data in files.items():
            archive.writestr(filename, data)


def test_package_verifies_in_fresh_isolated_process_without_pythonpath(config_path, smoke_features,
                                                                     tmp_path, monkeypatch):
    trap = tmp_path / "poison_pythonpath"
    trap.mkdir()
    (trap / "sitecustomize.py").write_text("raise RuntimeError('PYTHONPATH leaked into verification')\n")
    monkeypatch.setenv("PYTHONPATH", str(trap))
    calls = []
    original_run = subprocess.run
    def recorded_run(command, **kwargs):
        calls.append((command, kwargs))
        return original_run(command, **kwargs)
    monkeypatch.setattr(subprocess, "run", recorded_run)
    report = package_submission(config_path, tmp_path / "submission.zip", smoke_features=smoke_features)
    assert report["verified"] is True
    assert report["judge_runtime_verified"] is False
    verification = report["verification"]
    assert verification["passed"] and verification["network_disabled"]
    assert verification["independent_output_ownership"] and verification["resets"] == 2
    assert verification["rows"] == 3 * len(smoke_features)
    assert len(calls) == 1 and "-I" in calls[0][0]
    assert "PYTHONPATH" not in calls[0][1]["env"]
    assert Path(calls[0][1]["cwd"]) != ROOT
    with zipfile.ZipFile(report["archive"]) as archive:
        names = set(archive.namelist())
        assert {"solution.py", "model.onnx", "submission_config.json", "MANIFEST.json"} <= names
        assert not any(name.endswith(("data.py", "calibration.py", "training.py", "utils.py")) for name in names)
        configuration = json.loads(archive.read("submission_config.json"))
        assert configuration["model"] == "model.onnx"
        manifest = json.loads(archive.read("MANIFEST.json"))
        for filename, expected_hash in manifest["files"].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == expected_hash


def test_unsupported_native_platform_falls_back_to_packaged_onnx_offline(tmp_path, smoke_features):
    weights = tmp_path / "weights.npy"
    np.save(weights, np.zeros(192258, np.float32))
    # Only a deliberately unsupported architecture is advertised. The local
    # verifier must never try to load this non-native file before falling back.
    other_library = tmp_path / "other-architecture.so"
    other_library.write_bytes(b"unloadable on this architecture")
    config = tmp_path / "native.json"
    config.write_text(json.dumps({"backend": "native", "model": str(BASELINE),
                                  "native_weights": str(weights),
                                  "native_libraries": {"unsupported-os-architecture": str(other_library)},
                                  "postprocess": {"scales": [0.75, 1.25]}}))
    report = package_submission(config, tmp_path / "native_submission.zip", smoke_features=smoke_features)
    assert report["verified"] and report["verification"]["passed"]
    with zipfile.ZipFile(report["archive"]) as archive:
        settings = json.loads(archive.read("submission_config.json"))
        assert settings["native_weights"] == "weights.npy"
        assert settings["model"] == "model.onnx"
        assert all(value.startswith("native/") for value in settings["native_libraries"].values())


@pytest.mark.parametrize("prefix,error", [
    ("import utils\n", "No module named 'utils'"),
    ("import socket\nsocket.socket()\n", "network disabled"),
])
def test_verifier_rejects_repository_import_leak_and_network_use(config_path, tmp_path, prefix, error):
    report = package_submission(config_path, tmp_path / "valid.zip", verify=False)
    with zipfile.ZipFile(report["archive"]) as archive:
        original = archive.read("solution.py").decode()
    broken = tmp_path / "broken.zip"
    rewrite_entry(report["archive"], broken, "solution.py", prefix + original)
    with pytest.raises(RuntimeError, match=error):
        verify_submission(broken)


def test_packaging_is_deterministic_and_paths_are_relative(config_path, tmp_path):
    first = package_submission(config_path, tmp_path / "first.zip", verify=False)
    second = package_submission(config_path, tmp_path / "second.zip", verify=False)
    assert first["sha256"] == second["sha256"]
    assert first["size_bytes"] < 18_000_000
    with zipfile.ZipFile(first["archive"]) as archive:
        assert all(not Path(name).is_absolute() and ".." not in Path(name).parts for name in archive.namelist())


def test_failed_replacement_cannot_keep_a_stale_verified_report(config_path, smoke_features, tmp_path):
    output = tmp_path / "submission.zip"
    first = package_submission(config_path, output, smoke_features=smoke_features)
    original_archive = output.read_bytes()
    configuration = json.loads(config_path.read_text())
    configuration["postprocess"] = {"ema_half_life": 0, "ema_blend": 0.1}
    config_path.write_text(json.dumps(configuration))
    with pytest.raises(RuntimeError, match="half life"):
        package_submission(config_path, output, smoke_features=smoke_features)
    if output.read_bytes() == original_archive:
        assert json.loads(output.with_suffix(".report.json").read_text())["sha256"] == first["sha256"]
    else:
        report = json.loads(output.with_suffix(".report.json").read_text())
        assert report["verified"] is False
        assert report["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()


def test_verifier_rejects_unsafe_zip_entry_before_extraction(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.py", "raise RuntimeError('must never be extracted')")
    with pytest.raises(ValueError, match="unsafe archive path"):
        verify_submission(archive)
