"""Portable, training-only ORT-forward / CUDA-backward experiment runner.

The deployment trajectory and nonlinear objective stay on CPU. Only local
upper-GRU Jacobians run on the selected Torch device. No validation reads,
leaderboard calls, infrastructure provisioning or submission changes occur.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tarfile
import time
import uuid

import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import torch

from . import partial_gru as base
from .conversion_audit import export_original_graph
from .data import _check_schema
from .model_io import export_sequence_model, load_weights
from .global_objectives import correlation_objective, objective_parameters, GRADIENT_TOLERANCE
from .global_metric import METRIC_ID
from .training import OnnxSequenceEncoder, load_torch_model, seed_everything, _rng_state, _restore_rng

ROOT = Path(__file__).resolve().parents[1]
FORMAT = "connectome.gpu-training.global-wp.v2"
DEFAULT_ANCHOR = "artifacts/round2/correlation_gru1024_seed20260912/report.json"
DEFAULT_RANKING = "artifacts/global_wp/rescore_v1/ranking.json"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def content_identity(path):
    """Full streaming SHA256, independent of filename and copied-file mtime."""
    path = Path(path)
    before = path.stat()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {path}")
    return {"size_bytes": before.st_size, "sha256": hasher.hexdigest()}


def code_identity():
    paths = [*sorted((ROOT / "connectome").glob("*.py")), ROOT / "utils.py",
             ROOT / "requirements.txt", ROOT / "requirements-train.txt",
             ROOT / "cloud/Dockerfile"]
    return {str(p.relative_to(ROOT)): content_identity(p)["sha256"] for p in paths}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def prepare(train_path, destination, *, initial=None, teacher=None,
            anchor=DEFAULT_ANCHOR, ranking=DEFAULT_RANKING, seed=20261024, reserved_count=512):
    """Freeze two matched-update arms before any new labels are inspected.

    Reserved TRAIN groups are excluded from fitting by this runner. Historical
    exposure is inventoried conservatively from report specifications; this
    does not prove they were unseen by the supplied pretrained model.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("use a new plan directory; preserve frozen plans")
    selection = read_json(ranking)
    if (selection.get('metric_id') != METRIC_ID or selection.get('confirmation_used') is not False
            or not selection.get('ranking') or selection.get('selected') != selection['ranking'][0]
            or selection['selected'] != max(selection['ranking'], key=lambda row: row['weighted_pearson'])):
        raise ValueError('a completed, development-only Global WP ranking is required')
    from .selection import _validate_report
    manifest = read_json(ROOT / 'artifacts/manifest.json')
    for row in selection['ranking']:
        report_path = ROOT / row['report']
        report = read_json(report_path)
        if content_identity(report_path)['sha256'] != row['report_sha256'] or report.get('metric_id') != METRIC_ID:
            raise ValueError('rescore report changed or uses a historical metric')
        if not np.isclose(_validate_report(report, manifest, 'dev'), row['weighted_pearson'], rtol=0, atol=1e-12):
            raise ValueError('ranking differs from verified development statistics')
    winner = selection['selected']
    initial = Path(initial or ROOT / winner['model'])
    teacher = Path(teacher or initial)
    if (content_identity(initial)['sha256'] != winner['model_sha256']
            or content_identity(teacher) != content_identity(initial)):
        raise ValueError('initialize and anchor the teacher to the Global WP development winner')
    postprocess = winner['postprocess']
    if set(postprocess) - {'scales'}:
        raise ValueError('this training plan supports identity or per-target scaling only')
    scales = np.asarray(postprocess.get('scales', [1., 1.]), dtype=np.float32)
    if scales.shape != (2,) or not np.isfinite(scales).all() or np.any(scales <= 0):
        raise ValueError('positive finite per-target scales required')
    with pq.ParquetFile(train_path) as parquet:
        _check_schema(parquet)
        if "is_scored" in parquet.schema_arrow.names:
            raise ValueError("prepare accepts supplied TRAIN only, without is_scored")
        count = parquet.num_row_groups
    anchor_report = read_json(anchor)
    groups = base._validated_groups(train_path, anchor_report["specification"]["groups"])
    if len(groups) != 1024 or not anchor_report.get("complete"):
        raise ValueError("a completed 1024-sequence training anchor is required")
    before, fixed_teacher = load_weights(initial), load_weights(teacher)
    if any(before[key].tobytes() != fixed_teacher[key].tobytes() for key in ("w0", "r0", "b0")):
        raise ValueError("initial model and teacher must share the frozen lower GRU")
    exposed, inventory = set(groups), []
    # Read metadata only. Never inspect cached predictions or validation labels.
    reports = sorted((ROOT / "artifacts").rglob("report.json"))
    reports += sorted((ROOT / "artifacts").rglob("plan.json"))
    reports = list(dict.fromkeys([Path(anchor).resolve(), *reports]))
    for path in reports:
        document = read_json(path)
        previous = document.get("specification", {}).get("groups", [])
        if str(document.get("format", '')).startswith('connectome.gpu-training.') and "arms" in document:
            previous = sorted({g for arm in document["arms"].values() for g in arm["groups"]})
        elif str(document.get("format", '')).startswith('connectome.gpu-training.') and "history" in document:
            previous = sorted({g for batch in document["history"] for g in batch["groups"]})
        if isinstance(previous, list) and previous and all(type(g) is int for g in previous):
            if not set(previous) <= set(range(count)):
                raise ValueError(f"historical training groups out of range: {path}")
            exposed.update(previous)
            inventory.append({"report": path.name if not path.is_relative_to(ROOT) else str(path.relative_to(ROOT)),
                              "sha256": content_identity(path)["sha256"], "groups": previous})
    if type(reserved_count) is not int or reserved_count < 1:
        raise ValueError("reserved_count must be a positive integer")
    rng = np.random.default_rng(seed)
    unexposed = rng.permutation(sorted(set(range(count)) - exposed)).tolist()
    if len(unexposed) < reserved_count:
        raise ValueError("insufficient groups outside recorded historical exposure")
    previous_reserve = ROOT / 'artifacts/gpu/coverage-v1/plan.json'
    reserved = (read_json(previous_reserve)['reserved_train_groups'] if previous_reserve.exists() else unexposed[:reserved_count])
    if len(reserved) != reserved_count or set(reserved) & exposed:
        raise ValueError('preserved adaptation reserve conflicts with known exposure')
    available = set(range(count)) - set(reserved) - set(groups)
    extended = groups + rng.permutation(sorted(available)).tolist()[:3072]
    if len(extended) != 4096:
        raise ValueError("at least 4096 fitting groups are required")
    parameters = dict(base.DEFAULT_PARAMETERS)
    parameters["learning_rate"] = 1e-5
    arms = {
        "repeat1024": {"groups": groups, "epochs": 4},
        "coverage4096": {"groups": extended, "epochs": 1},
    }
    plan = {"format": FORMAT, "metric_id": METRIC_ID, "seed": seed, "parameters": parameters, "arms": arms,
            "postprocess": postprocess,
            "selection": {'ranking_sha256': content_identity(ranking)['sha256'],
                          'policy_sha256': selection['policy_sha256'], 'winner': winner,
                          'rationale': 'initialize and teacher-anchor at best predeclared existing configuration under development Global WP; new pooled-loss learning rate is conservative and unvalidated'},
            "initial": content_identity(initial), "teacher": content_identity(teacher),
            "train": content_identity(train_path), "code": code_identity(),
            "objective": objective_parameters(), "reserved_train_groups": reserved,
            "historical_exposure_inventory": inventory,
            "reserved_policy": "excluded from new adaptation; inventory is not proof of no prior exposure; pretrained lineage unknown",
            "evaluation_policy": "training_only; score each completed arm once on existing screen, both once on full development; no postprocess or epoch search; confirmation permanently excluded",
            "method": "current_student_CPU_ORT_full_sequence_forward; local_FP32_Torch_upper_GRU_and_head_VJP; CPU_FP64_objective"}
    plan["plan_sha256"] = digest(plan)
    destination.mkdir(parents=True)
    for name, path in (("initial.onnx", initial), ("teacher.onnx", teacher)):
        (destination / name).write_bytes(Path(path).read_bytes())
    write_json(destination / "plan.json", plan)
    write_json(destination / "selection.json", selection)
    return plan


def load_plan(directory, train_path):
    directory = Path(directory).resolve()
    plan = read_json(directory / "plan.json")
    if plan.get("format") != FORMAT or plan.get('metric_id') != METRIC_ID:
        raise ValueError('obsolete GPU plan: current training requires Global WP format v2')
    if plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"}):
        raise ValueError("frozen plan changed or format is unsupported")
    if plan["code"] != code_identity() or plan["objective"] != objective_parameters():
        raise ValueError("training source code or objective changed since preparation")
    if 'selection' in plan and content_identity(directory / 'selection.json')['sha256'] != plan['selection']['ranking_sha256']:
        raise ValueError('Global WP selection receipt changed')
    for key, path in (("initial", directory / "initial.onnx"), ("teacher", directory / "teacher.onnx"), ("train", train_path)):
        if content_identity(path) != plan[key]:
            raise ValueError(f"{key} content changed")
    reserved = set(plan["reserved_train_groups"])
    for arm in plan["arms"].values():
        base._validated_groups(train_path, arm["groups"])
        if reserved & set(arm["groups"]):
            raise ValueError("training arm overlaps reserved TRAIN groups")
    return plan


def bundle(plan_dir, output):
    """Create a private source/model handoff; datasets and credentials excluded."""
    plan_dir, output = Path(plan_dir).resolve(), Path(output).resolve()
    plan = read_json(plan_dir / "plan.json")
    if (plan.get("format") != FORMAT or plan.get('metric_id') != METRIC_ID
            or plan.get("plan_sha256") != digest({k: v for k, v in plan.items() if k != "plan_sha256"})
            or plan["code"] != code_identity()):
        raise ValueError("plan or source code changed")
    files = {name: ROOT / name for name in plan["code"]}
    files.update({".dockerignore": ROOT / ".dockerignore", "docs/GPU_TRAINING.md": ROOT / "docs/GPU_TRAINING.md"})
    files["plan/plan.json"] = plan_dir / "plan.json"
    if (plan_dir / 'selection.json').exists():
        if content_identity(plan_dir / 'selection.json')['sha256'] != plan['selection']['ranking_sha256']:
            raise ValueError('Global WP selection receipt changed')
        files['plan/selection.json'] = plan_dir / 'selection.json'
    for key, name in (("initial", "initial.onnx"), ("teacher", "teacher.onnx")):
        path = plan_dir / name
        if content_identity(path) != plan[key]:
            raise ValueError(f"{key} content changed")
        files[f"plan/{name}"] = path
    if output.exists():
        raise FileExistsError("preserve existing handoff archives")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            for name, path in sorted(files.items()):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"bundle source must be a regular file: {path}")
                archive.add(path, arcname=name, recursive=False)
        # Exclusive publication prevents a second bundler replacing this result.
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(output), "identity": content_identity(output), "plan_sha256": plan["plan_sha256"]}


def runtime(device):
    selected = torch.device(device)
    if selected.type not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
    if selected.type == "cuda":
        # Required by deterministic CUDA matrix operations before CUDA startup.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if selected.index is None:
            selected = torch.device("cuda:0")
        torch.cuda.set_device(selected)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {"device": str(selected), "gpu": torch.cuda.get_device_name(selected) if selected.type == "cuda" else None,
            "torch": str(torch.__version__), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "onnxruntime": ort.__version__,
            "numpy": np.__version__, "precision": "FP32; no AMP or TF32"}


def backward(model, arrays, coefficients, sequence_ids, parameters):
    """Bounded device tensors and unchanged uncorrected local error checks."""
    lower, upper, prediction = arrays[:3]
    device = next(model.parameters()).device
    lower, upper, prediction, coefficients = [x.to(device) for x in (lower, upper, prediction, coefficients)]
    model.zero_grad(set_to_none=True)
    diagnostics = base._LocalDiagnostics()
    for start in range(0, len(lower), parameters["chunk_size"]):
        end = min(start + parameters["chunk_size"], len(lower))
        proxy, raw_upper, raw_prediction = base.reanchored_upper_chunk(model, lower, upper, prediction, start=start, end=end)
        diagnostics.add(raw_upper.detach().cpu(), raw_prediction.cpu(), upper[start:end].cpu(), prediction[start:end].cpu())
        if not torch.equal(proxy.detach(), prediction[start:end]):
            raise FloatingPointError("corrected forward differs from current ORT student")
        scalar = (proxy * coefficients[start:end]).sum()
        if not torch.isfinite(scalar):
            raise FloatingPointError("nonfinite backward scalar")
        scalar.backward()
    report = diagnostics.finish(sequence_ids)
    if any(not row[name]["passed"] for row in report for name in ("upper", "prediction")):
        raise FloatingPointError("raw local Torch/ORT errors exceed unchanged diagnostic limits")
    gradients = {name: p.grad.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}
    if not all(torch.isfinite(value).all() for value in gradients.values()):
        raise FloatingPointError("nonfinite local gradient")
    return gradients, report


def qualify(model, arrays, coefficients, sequence_ids, parameters):
    """Compare actual device VJPs with CPU on the same complete batch/window.

    Corrections alone are tautological; raw hidden/output diagnostics and
    independent device-to-CPU parameter gradients must also pass.
    """
    cpu = copy.deepcopy(model).cpu()
    reference, _ = backward(cpu, arrays, coefficients, sequence_ids, parameters)
    actual, diagnostics = backward(model, arrays, coefficients, sequence_ids, parameters)
    rows = {}
    for name in reference:
        difference = (actual[name] - reference[name]).abs()
        limit = GRADIENT_TOLERANCE["atol"] + GRADIENT_TOLERANCE["rtol"] * reference[name].abs()
        rows[name] = {"max_abs": float(difference.max()), "max_tolerance_ratio": float((difference / limit).max()),
                      "passed": bool((difference <= limit).all())}
    result = {"passed": all(row["passed"] for row in rows.values()), "gradient_tolerance": GRADIENT_TOLERANCE,
              "local_limits": base.LOCAL_DIAGNOSTIC_LIMITS, "gradients": rows, "local_diagnostics": diagnostics,
              "scope": "this runtime, current weights, batch and backward window; not full carried-Torch parity"}
    return result


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process owns this run directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def checkpoint(path, value):
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def train(plan_dir, train_path, output_dir, *, arm, device="cuda", resume=False,
          max_updates=None, max_seconds=None, qualification_only=False, stop_requested=lambda: False):
    """Resume from completed batch boundaries, even after path/mtime relocation."""
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive absolute update count")
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("max_seconds must be positive")
    environment = runtime(device)  # Fail before scanning 27 GB when CUDA is absent.
    started = time.monotonic()
    plan_dir, output_dir = Path(plan_dir).resolve(), Path(output_dir).resolve()
    plan = load_plan(plan_dir, train_path)
    if arm not in plan["arms"]:
        raise ValueError("unknown frozen training arm")
    parameters, allocation = plan["parameters"], plan["arms"][arm]
    seed_everything(plan["seed"], parameters["threads"])
    initial, teacher_path = plan_dir / "initial.onnx", plan_dir / "teacher.onnx"
    with run_lock(output_dir):
        checkpoint_path = output_dir / "checkpoint.pt"
        if not resume and any(p.name != "run.lock" for p in output_dir.iterdir()):
            raise FileExistsError("use an empty run directory or --resume")
        model = load_torch_model(initial, device=device)
        model.gru0.requires_grad_(False)
        model.train()
        trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
        anchor = {name: p.detach().clone() for name, p in trainable.items()}
        frozen = {name: p.detach().cpu().clone() for name, p in model.gru0.state_dict().items()}
        optimizer = torch.optim.AdamW(list(trainable.values()), lr=parameters["learning_rate"], weight_decay=parameters["weight_decay"])
        cursor = {"epoch": 0, "position": 0, "updates": 0, "sequences_seen": 0}
        history, sessions = [], []
        if resume:
            # Only our own trusted checkpoints; never load third-party pickles.
            previous = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (previous.get("format") != FORMAT or previous.get("plan_sha256") != plan["plan_sha256"]
                    or previous.get("arm") != arm):
                raise ValueError("checkpoint belongs to a different plan or arm")
            model.load_state_dict(previous["model_state"])
            optimizer.load_state_dict(previous["optimizer_state"])
            cursor, history, sessions = previous["cursor"], previous["history"], previous["sessions"]
            _restore_rng(previous["rng"])
        if (not 0 <= cursor["epoch"] <= allocation["epochs"] or not 0 <= cursor["position"] < len(allocation["groups"])
                or cursor["epoch"] == allocation["epochs"] and cursor["position"] != 0
                or cursor["updates"] != len(history) or max_updates is not None and max_updates < cursor["updates"]):
            raise ValueError("invalid checkpoint cursor or update budget")
        if any(not torch.equal(p.detach().cpu(), frozen[name]) for name, p in model.gru0.state_dict().items()):
            raise ValueError("checkpoint changed frozen lower GRU")
        sessions.append(environment)

        def save():
            checkpoint(checkpoint_path, {"format": FORMAT, "plan_sha256": plan["plan_sha256"], "arm": arm,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "cursor": dict(cursor), "history": list(history), "sessions": sessions, "rng": _rng_state()})

        if not qualification_only:
            save()
        teacher = OnnxSequenceEncoder(teacher_path, output_dir / "teacher_encoder.onnx")
        qualified = False
        session_updates, session_rows = 0, 0
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        while cursor["epoch"] < allocation["epochs"]:
            if (stop_requested() or max_updates is not None and cursor["updates"] >= max_updates
                    or max_seconds is not None and time.monotonic() - started >= max_seconds):
                break
            batch_started = time.monotonic()
            order = np.random.default_rng(plan["seed"] + cursor["epoch"]).permutation(allocation["groups"]).tolist()
            groups = order[cursor["position"]:cursor["position"] + parameters["batch_size"]]
            snapshot = export_original_graph(model, initial, output_dir / "current_student.onnx")
            student = OnnxSequenceEncoder(snapshot, output_dir / "student_encoder.onnx")
            arrays = base._encode_batch(teacher, student, train_path, groups)
            del student
            lower, upper, prediction, matching, target, sequence_ids = arrays
            scales = torch.tensor(plan.get('postprocess', {}).get('scales', [1., 1.]), dtype=torch.float32)
            loss, coefficients, components = correlation_objective(prediction * scales, matching * scales, target,
                                                                    teacher_weight=parameters["teacher_weight"])
            coefficients = coefficients * scales  # Chain rule to the raw ONNX output; preserve deployment scaling.
            if not qualified:
                evidence = qualify(model, arrays, coefficients, sequence_ids, parameters)
                evidence.update(runtime=environment, plan_sha256=plan["plan_sha256"], arm=arm,
                                current_model_sha256=content_identity(snapshot)["sha256"], groups=groups)
                write_json(output_dir / "qualification.json", evidence)
                if not evidence["passed"]:
                    raise FloatingPointError("device/CPU gradients failed qualification; no update committed")
                qualified = True
                diagnostics = evidence["local_diagnostics"]
            else:
                _, diagnostics = backward(model, arrays, coefficients, sequence_ids, parameters)
            if qualification_only:
                return evidence
            # A signal during qualification or backward discards this uncommitted batch.
            if stop_requested():
                break
            anchor_loss = base.relative_anchor(trainable, anchor)
            (parameters["anchor_weight"] * anchor_loss).backward()
            norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), parameters["gradient_clip"])
            if not torch.isfinite(norm):
                raise FloatingPointError("nonfinite gradient; last committed checkpoint retained")
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("nonfinite parameters; last committed checkpoint retained")
            if any(not torch.equal(p.detach().cpu(), frozen[name]) for name, p in model.gru0.state_dict().items()):
                raise AssertionError("frozen lower GRU changed")
            cursor["updates"] += 1
            cursor["sequences_seen"] += len(groups)
            cursor["position"] += len(groups)
            history.append({"epoch": cursor["epoch"], "update": cursor["updates"], "groups": groups,
                            "objective_loss": loss, "objective_components": components,
                            "gradient_norm": float(norm), "local_diagnostics": diagnostics,
                            "raw_coefficients_sha256": hashlib.sha256(coefficients.contiguous().numpy().tobytes()).hexdigest(),
                            "batch_seconds": time.monotonic() - batch_started})
            if cursor["position"] == len(allocation["groups"]):
                cursor["epoch"] += 1
                cursor["position"] = 0
            save()
            session_updates += 1
            session_rows += len(lower) * len(groups)
            print(json.dumps({"update": cursor["updates"], "epoch": cursor["epoch"], "loss": loss,
                              "batch_seconds": history[-1]["batch_seconds"]}), flush=True)
        if qualification_only:
            result = {"passed": False, "status": "not_run", "runtime": environment,
                      "plan_sha256": plan["plan_sha256"], "arm": arm,
                      "reason": "no batch qualified before interruption, time/update budget or completion"}
            write_json(output_dir / "qualification.json", result)
            return result
        exported = export_original_graph(model, initial, output_dir / "model.onnx")
        export_sequence_model(exported, output_dir / "sequence.onnx")
        write_json(output_dir / 'config.json', {'model':'model.onnx', 'backend':'onnx', 'onnx_mode':'iobinding',
                   'strict_inputs':True, 'affinity':False, 'postprocess':plan.get('postprocess', {})})
        elapsed = time.monotonic() - started
        report = {"format": FORMAT, "metric_id": METRIC_ID, "objective": objective_parameters(),
                  "postprocess": plan.get('postprocess', {}), "plan_sha256": plan["plan_sha256"], "arm": arm,
                  "complete": cursor["epoch"] >= allocation["epochs"], "cursor": cursor, "history": history,
                  "sessions": sessions, "model_sha256": content_identity(exported)["sha256"],
                  "checkpoint_sha256": content_identity(checkpoint_path)["sha256"],
                  "elapsed_seconds": elapsed, "session_updates": session_updates,
                  "session_rows_per_second": session_rows / elapsed, "session_rows": session_rows,
                  "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if device.startswith("cuda") else None,
                  "deployment_status": "experimental; CPU callback/bulk parity, new declared validation and callback feasibility pending"}
        write_json(output_dir / "report.json", report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare", help="freeze models, content hashes and training allocations")
    prepare_parser.add_argument("--train", default="datasets/train.parquet")
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.add_argument("--initial")
    prepare_parser.add_argument("--teacher")
    prepare_parser.add_argument("--anchor", default=DEFAULT_ANCHOR)
    prepare_parser.add_argument('--ranking', default=DEFAULT_RANKING)
    bundle_parser = commands.add_parser("bundle", help="package private source and frozen models, excluding datasets")
    bundle_parser.add_argument("--plan", required=True)
    bundle_parser.add_argument("--output", required=True)
    for name in ("train", "qualify"):
        child = commands.add_parser(name)
        child.add_argument("--plan", required=True)
        child.add_argument("--train", required=True)
        child.add_argument("--output", required=True)
        child.add_argument("--arm", required=True, choices=("repeat1024", "coverage4096"))
        child.add_argument("--device", default="cuda")
        child.add_argument("--resume", action="store_true")
        child.add_argument("--max-updates", type=int)
        child.add_argument("--max-seconds", type=float)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command != 'prepare':
        header = read_json(Path(args['plan']) / 'plan.json')
        if header.get('format') != FORMAT or header.get('metric_id') != METRIC_ID:
            parser.error('obsolete GPU plan: current training requires Global WP format v2')
    if command == "prepare":
        result = prepare(args.pop("train"), args.pop("output"), **args)
        print(json.dumps({"plan_sha256": result["plan_sha256"], "arms": list(result["arms"])}))
        return
    if command == "bundle":
        print(json.dumps(bundle(args["plan"], args["output"])))
        return
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    result = train(args.pop("plan"), args.pop("train"), args.pop("output"),
                   qualification_only=command == "qualify", stop_requested=lambda: stopped, **args)
    print(json.dumps({k: v for k, v in result.items() if k in {"complete", "passed", "cursor", "elapsed_seconds", "session_rows_per_second"}}))
    if stopped:
        raise SystemExit(143)
    if command == "qualify" and not result.get("passed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
