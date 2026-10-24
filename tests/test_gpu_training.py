"""Portable runner integration on synthetic rows, with actual ONNX trajectories."""
import copy
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from connectome import gpu_training as training
from connectome import partial_gru as base
from connectome.model_io import load_weights

# The distributed baseline is sufficient; tests do not require local artifacts.
SOURCE = Path(__file__).resolve().parents[1] / "baseline/baseline.onnx"


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    path = tmp_path / "train.parquet"
    path.write_bytes(b"synthetic; reader replaced, no competition rows")
    rng = np.random.default_rng(921)
    sequences = {group: (100 + group, rng.normal(0, .2, (123, 112)).astype(np.float32),
                          rng.normal(0, .4, (123, 2)).astype(np.float32)) for group in range(3)}
    monkeypatch.setattr(base, "_validated_groups", lambda path, groups: list(groups))
    monkeypatch.setattr(base, "_iter_sequences", lambda path, groups: (sequences[g] for g in groups))
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    for name in ("initial.onnx", "teacher.onnx"):
        shutil.copyfile(SOURCE, plan_dir / name)
    parameters = dict(base.DEFAULT_PARAMETERS, batch_size=2, chunk_size=64, learning_rate=1e-4)
    plan = {"format": training.FORMAT, "metric_id": training.METRIC_ID, "parameters": parameters, "seed": 921,
            "arms": {"test": {"groups": [0, 1, 2], "epochs": 1}},
            "code": training.code_identity(), "objective": training.objective_parameters(),
            "reserved_train_groups": [3], "train": training.content_identity(path),
            "initial": training.content_identity(SOURCE), "teacher": training.content_identity(SOURCE)}
    plan["plan_sha256"] = training.digest(plan)
    training.write_json(plan_dir / "plan.json", plan)
    return SimpleNamespace(path=path, plan_dir=plan_dir, plan=plan, sequences=sequences)


def run(data, directory, **kwargs):
    return training.train(data.plan_dir, data.path, directory, arm="test", device="cpu", **kwargs)


def test_exact_resume_after_relocating_plan_data_and_checkpoint(synthetic, tmp_path):
    full = run(synthetic, tmp_path / "full")
    paused = run(synthetic, tmp_path / "paused", max_updates=1)
    assert not paused["complete"]
    remote = tmp_path / "different-machine"
    shutil.copytree(synthetic.plan_dir, remote / "plan")
    shutil.copytree(tmp_path / "paused", remote / "run")
    shutil.copyfile(synthetic.path, remote / "copied.parquet")
    resumed = training.train(remote / "plan", remote / "copied.parquet", remote / "run",
                             arm="test", device="cpu", resume=True)
    assert resumed["complete"] and resumed["cursor"] == full["cursor"]
    assert resumed["model_sha256"] == full["model_sha256"]
    for actual, expected in zip(resumed["history"], full["history"]):
        assert {k: v for k, v in actual.items() if k != "batch_seconds"} == {
            k: v for k, v in expected.items() if k != "batch_seconds"}
    original, fitted = load_weights(SOURCE), load_weights(tmp_path / "full/model.onnx")
    assert all(original[key].tobytes() == fitted[key].tobytes() for key in ("w0", "r0", "b0"))
    assert any(original[key].tobytes() != fitted[key].tobytes() for key in ("w1", "r1", "b1", "head_w", "head_b"))


def test_content_or_plan_mutation_rejected(synthetic, tmp_path):
    synthetic.path.write_bytes(b"different content")
    with pytest.raises(ValueError, match="train content changed"):
        run(synthetic, tmp_path / "bad-data")
    plan = copy.deepcopy(synthetic.plan)
    plan["parameters"]["learning_rate"] = .1
    training.write_json(synthetic.plan_dir / "plan.json", plan)
    with pytest.raises(ValueError, match="frozen plan changed"):
        run(synthetic, tmp_path / "bad-plan")


def test_changed_code_and_reserved_group_rejected(synthetic, tmp_path, monkeypatch):
    monkeypatch.setattr(training, "code_identity", lambda: {"changed": "code"})
    with pytest.raises(ValueError, match="source code"):
        run(synthetic, tmp_path / "bad-code")
    monkeypatch.setattr(training, "code_identity", lambda: synthetic.plan["code"])
    plan = copy.deepcopy(synthetic.plan)
    plan["reserved_train_groups"] = [2]
    plan["plan_sha256"] = training.digest({k: v for k, v in plan.items() if k != "plan_sha256"})
    training.write_json(synthetic.plan_dir / "plan.json", plan)
    with pytest.raises(ValueError, match="overlaps reserved"):
        run(synthetic, tmp_path / "overlap")


def test_qualification_does_not_commit_optimizer_update(synthetic, tmp_path):
    result = run(synthetic, tmp_path / "qualification", qualification_only=True)
    assert result["passed"]
    assert not (tmp_path / "qualification/checkpoint.pt").exists()
    assert result["current_model_sha256"] == training.content_identity(SOURCE)["sha256"]


@pytest.mark.parametrize("options", [{"stop_requested": lambda: True}, {"max_seconds": 1e-12}])
def test_qualification_without_a_batch_has_explicit_incomplete_status(synthetic, tmp_path, options):
    result = run(synthetic, tmp_path / "qualification", qualification_only=True, **options)
    assert not result["passed"] and result["status"] == "not_run"
    assert not (tmp_path / "qualification/model.onnx").exists()


def test_failed_gradient_qualification_preserves_initial_checkpoint(synthetic, tmp_path, monkeypatch):
    original = training.qualify
    def failed(*args, **kwargs):
        report = original(*args, **kwargs)
        report["passed"] = False
        return report
    monkeypatch.setattr(training, "qualify", failed)
    destination = tmp_path / "failed"
    with pytest.raises(FloatingPointError, match="failed qualification"):
        run(synthetic, destination)
    state = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=False)
    assert state["cursor"]["updates"] == 0
    assert not (destination / "model.onnx").exists()


def test_stop_during_backward_discards_uncommitted_batch(synthetic, tmp_path):
    calls = 0
    def stopped():
        nonlocal calls
        calls += 1
        return calls > 1
    result = run(synthetic, tmp_path / "interrupted", stop_requested=stopped)
    assert result["cursor"]["updates"] == 0
    assert result["model_sha256"] == training.content_identity(SOURCE)["sha256"]


def test_unavailable_cuda_fails_before_data_read(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested but unavailable"):
        training.train(tmp_path / "absent", tmp_path / "absent", tmp_path / "run", arm="test")


def test_concurrent_run_lock_rejected(tmp_path):
    with training.run_lock(tmp_path / "run"):
        with pytest.raises(RuntimeError, match="another process"):
            with training.run_lock(tmp_path / "run"):
                pass


def test_bundle_contains_only_explicit_source_and_model_files(synthetic, tmp_path):
    import tarfile
    output = tmp_path / "bundle.tar.gz"
    result = training.bundle(synthetic.plan_dir, output)
    with tarfile.open(output) as archive:
        names = archive.getnames()
        assert "plan/initial.onnx" in names and "cloud/Dockerfile" in names
        assert "connectome/gpu_training.py" in names
        assert not any(name.startswith(("datasets/", ".git/", ".venv/", "artifacts/")) for name in names)
    assert result["identity"] == training.content_identity(output)
    with pytest.raises(FileExistsError):
        training.bundle(synthetic.plan_dir, output)


@pytest.mark.parametrize('command', ['bundle', 'train', 'qualify'])
def test_cli_blocks_outdated_handoff_before_any_work(command, tmp_path, monkeypatch, capsys):
    import sys
    plan_dir = tmp_path / 'old-plan'
    plan_dir.mkdir()
    training.write_json(plan_dir / 'plan.json', {'format':'connectome.gpu-training.v1'})
    arguments = [command, '--plan', str(plan_dir), '--output', str(tmp_path / 'unused')]
    if command != 'bundle':
        arguments += ['--train', 'absent', '--arm', 'coverage4096']
    monkeypatch.setattr(sys, "argv", ["gpu_training", *arguments])
    def forbidden(*args, **kwargs):
        raise AssertionError("outdated rollout must stop before data reads or work")
    for name in ("prepare", "bundle", "train"):
        monkeypatch.setattr(training, name, forbidden)
    with pytest.raises(SystemExit) as exc:
        training.main()
    assert exc.value.code == 2
    assert "Global WP" in capsys.readouterr().err


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA qualification requires a real GPU")
def test_cuda_qualification_on_real_device(synthetic, tmp_path):
    result = training.train(synthetic.plan_dir, synthetic.path, tmp_path / "cuda",
                            arm="test", device="cuda", qualification_only=True)
    assert result["passed"] and result["runtime"]["gpu"]
