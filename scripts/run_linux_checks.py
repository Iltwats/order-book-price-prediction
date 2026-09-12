#!/usr/bin/env python3
"""Build/run the offline Linux ARM64 inference verification environment.

Run on the host with --build to create the dependency image, then launch a
1-CPU/16-GB/network-disabled container. --phase prepare builds/tests native code
and audits complete-group parity. --phase benchmark compares all callback modes
on a fixed manifest subset. Container results are Linux ARM64 measurements on
this host, not verified measurements on the hidden judge CPU.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "connectome-inference:py311-ort1232-arm64"
MEMORY_LIMIT_BYTES = 16_000_000_000


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def run_logged(command, log_path, cwd=ROOT):
    print("RUN", " ".join(map(str, command)), flush=True)
    start = time.perf_counter()
    with Path(log_path).open("w") as log:
        process = subprocess.Popen(list(map(str, command)), cwd=cwd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        returncode = process.wait()
    result = {"command": list(map(str, command)), "returncode": returncode,
              "seconds": time.perf_counter() - start, "log": str(log_path)}
    if returncode:
        error = subprocess.CalledProcessError(returncode, command)
        error.check_result = result
        raise error
    return result


def constraint_evidence():
    result = {"platform": platform.platform(), "architecture": platform.machine(),
              "python": platform.python_version(), "judge_runtime_verified": False,
              "host_exclusive": False, "native_promoted": False,
              "hardware_counters": {"collected": False,
                                    "reason": "No cache/performance counters were collected; perf is not installed in this inference image"}}
    for name in ("cpu.max", "memory.max", "memory.swap.max", "cpuset.cpus.effective"):
        path = Path("/sys/fs/cgroup") / name
        result[name] = path.read_text().strip() if path.exists() else None
    interfaces = {}
    for _, name in socket.if_nameindex():
        flags_path = Path("/sys/class/net") / name / "flags"
        flags = int(flags_path.read_text().strip(), 16) if flags_path.exists() else None
        interfaces[name] = {"flags": flags, "up": bool(flags & 1) if flags is not None else None}
    result["network_interfaces"] = interfaces
    # Docker Desktop's kernel exposes dormant tunnel devices even in an isolated
    # 'none' network. IFF_UP distinguishes those unconnected placeholders from an
    # actual interface. No socket to an external address is opened by this check.
    result["active_non_loopback_interfaces"] = [name for name, info in interfaces.items()
                                                if name != "lo" and info["up"] is not False]
    result["utc"] = datetime.now(timezone.utc).isoformat()
    return result


def inside(args):
    if platform.system() != "Linux" or platform.machine() not in {"aarch64", "arm64"}:
        raise RuntimeError("this run is explicitly scoped to Linux ARM64")
    sys.path.insert(0, str(ROOT))
    import importlib.util
    import onnx
    import onnxruntime as ort
    from connectome.evaluation import benchmark
    from connectome.model_io import export_native_weights, simplify_model

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = constraint_evidence()
    evidence.update(onnx=onnx.__version__, onnxruntime=ort.__version__,
                    torch_installed=importlib.util.find_spec("torch") is not None)
    evidence["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [ROOT / "connectome" / "runtime.py", ROOT / "connectome" / "evaluation.py",
                                              ROOT / "connectome" / "native.py", ROOT / "scripts" / "run_linux_checks.py"]}
    save(output / "environment.json", evidence)
    if evidence["cpu.max"] != "100000 100000":
        raise RuntimeError(f"expected one CPU quota; observed {evidence['cpu.max']}")
    if evidence["memory.max"] != str(MEMORY_LIMIT_BYTES):
        raise RuntimeError(f"expected {MEMORY_LIMIT_BYTES} byte RAM limit; observed {evidence['memory.max']}")
    if evidence["memory.swap.max"] != "0":
        raise RuntimeError(f"expected no additional swap; observed {evidence['memory.swap.max']}")
    if evidence["active_non_loopback_interfaces"]:
        raise RuntimeError("expected --network none with no active non-loopback interfaces")
    if evidence["torch_installed"]:
        raise RuntimeError("inference test image must not contain Torch")
    start = time.perf_counter()
    summary = {"environment": evidence, "phase": args.phase, "checks": [],
               "benchmark_reports": [], "native_promoted": False}
    library = output / "native" / "libconnectome_native.so"
    models = output / "models"
    models.mkdir(exist_ok=True)
    weights = models / "weights.npy"
    simplified = models / "simplified.onnx"
    original = ROOT / "baseline" / "baseline.onnx"
    dataset = ROOT / "datasets" / "valid.parquet"
    export_native_weights(original, weights)
    simplify_model(original, simplified)
    try:
        if args.phase in {"all", "prepare"}:
            summary["checks"].append(run_logged(
                [sys.executable, "native/build.py", "--output", library.parent, "--test", "--sanitize"],
                output / "native_build.log"))
        if args.phase in {"all", "prepare", "tests"}:
            test_command = [sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", "tests", "native/test_model_io.py"]
            # Keep host training coverage independent; older working-tree versions
            # of the training tests imported Torch unconditionally at collection.
            training_test = ROOT / "tests" / "test_training.py"
            if training_test.exists() and "importorskip" not in training_test.read_text():
                test_command += ["--ignore=tests/test_training.py"]
                summary["training_test_exclusion"] = "Torch is deliberately absent from the inference image"
            summary["checks"].append(run_logged(test_command, output / "pytest.log"))
        if args.phase in {"all", "prepare", "parity"}:
            if not library.is_file():
                raise FileNotFoundError("run --phase prepare before parity verification")
            verification = output / "verification"
            verification.mkdir(exist_ok=True)
            summary["checks"].append(run_logged(
                [sys.executable, "native/verify_backend.py", "--library", library,
                 "--dataset", dataset, "--groups", "0", "--output", verification / "parity_group0.json"],
                output / "parity_group0.log"))
            parity = json.loads((verification / "parity_group0.json").read_text())
            summary["native_accuracy_gate_passed"] = all(
                item["native_prediction_tolerance_failures"] == 0 and item["native_hidden_tolerance_failures"] == 0
                for item in parity["groups"])
            summary["simplified_bitwise_equal"] = all(item["simplified_bitwise_equal"] for item in parity["groups"])
        if args.phase in {"all", "benchmark", "solutions"}:
            # Only the immutable row-group membership is read. Host-specific data
            # paths recorded in the manifest are intentionally not resolved here.
            manifest_bytes = args.manifest.read_bytes()
            manifest = json.loads(manifest_bytes)
            groups = list(map(int, manifest["splits"]["smoke"]))[:args.smoke_count]
            if len(groups) != args.smoke_count or len(set(groups)) != len(groups):
                raise ValueError("manifest does not provide the requested unique smoke groups")
            summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            summary["groups"] = groups
            summary["repetitions"] = args.repetitions
        if args.phase == "stream":
            if args.candidate_config is None:
                raise ValueError("--candidate-config is required for streamed validation")
            from scripts.benchmark_submission_callbacks import evaluate_streaming_candidate
            manifest_bytes = args.manifest.read_bytes()
            manifest = json.loads(manifest_bytes)
            groups = list(map(int, manifest["splits"]["screen"]))[:args.stream_count]
            if len(groups) != args.stream_count or len(set(groups)) != len(groups):
                raise ValueError("manifest does not provide the requested unique screen groups")
            summary["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
            summary["groups"] = groups
            report = evaluate_streaming_candidate(dataset, args.candidate_config, groups, affinity=args.affinity)
            report["linux_constraints"] = evidence
            report_path = output / f"stream_{args.stream_count}seq{'_affinity' if args.affinity else ''}.json"
            save(report_path, report)
            summary["benchmark_reports"].append(str(report_path))
        if args.phase == "zip":
            if args.archive is None:
                raise ValueError("--archive is required for fresh-process ZIP validation")
            import pyarrow.parquet as pq
            from connectome.data import read_sequence
            from connectome.packaging import verify_submission
            features = read_sequence(pq.ParquetFile(dataset), 0).features[:256]
            check_start = time.perf_counter()
            report = verify_submission(args.archive, smoke_features=features)
            report.update(archive=str(args.archive), archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                          archive_bytes=args.archive.stat().st_size, seconds=time.perf_counter()-check_start,
                          linux_constraints=evidence)
            report_path = output / "fresh_zip.json"
            save(report_path, report)
            summary["benchmark_reports"].append(str(report_path))
        if args.phase == "solutions":
            if args.candidate_config is None:
                raise ValueError("--candidate-config is required for the actual solutions comparison")
            from scripts.benchmark_submission_callbacks import benchmark_solutions
            report = benchmark_solutions(dataset, ROOT / "baseline" / "solution.py", args.candidate_config,
                                         groups, repetitions=args.repetitions, affinity=args.affinity)
            report["linux_constraints"] = evidence
            report["native_promoted"] = False
            suffix = "affinity" if args.affinity else "unpinned"
            report_path = output / f"solutions_{args.smoke_count}seq_{suffix}.json"
            save(report_path, report)
            summary["benchmark_reports"].append(str(report_path))
        if args.phase in {"all", "benchmark"}:
            if not library.is_file():
                raise FileNotFoundError("run --phase prepare before benchmarking")
            results = {"original": [], "simplified": []}
            configurations = [("original", original, ["reference", "run", "iobinding", "ortvaluevector", "native_row", "native_packed"]),
                              ("simplified", simplified, ["reference", "run", "iobinding", "ortvaluevector"])]
            for label, model, modes in configurations:
                for mode in modes:
                    if args.modes is not None and mode not in args.modes:
                        continue
                    report = benchmark(dataset, model, groups, modes=[mode], repetitions=args.repetitions,
                                       native_weights=weights, native_library=library, affinity=args.affinity)
                    report["linux_constraints"] = evidence
                    report["native_promoted"] = False
                    report["comparison_caveat"] = "Linux ARM64 on a shared M2 host; quota-limited, not an exclusive or known judge CPU"
                    suffix = "_affinity" if args.affinity else ""
                    report_path = output / f"benchmark_{args.smoke_count}seq_{label}_{mode}{suffix}.json"
                    save(report_path, report)
                    summary["benchmark_reports"].append(str(report_path))
                    results[label].extend(report["runs"])
                    summary["seconds"] = time.perf_counter() - start
                    save(output / f"summary_{phase_label(args)}_{args.smoke_count}seq.json", summary)
            summary["results"] = results
        summary["seconds"] = time.perf_counter() - start
        summary["completed"] = True
    except Exception as error:
        summary["seconds"] = time.perf_counter() - start
        summary["completed"] = False
        summary["error"] = str(error)
        if hasattr(error, "check_result"):
            summary["failed_check"] = error.check_result
        save(output / f"summary_{phase_label(args)}_{run_count(args)}seq.json", summary)
        raise
    save(output / f"summary_{phase_label(args)}_{run_count(args)}seq.json", summary)
    print(json.dumps({"completed": True, "seconds": summary["seconds"], "output": str(output),
                      "native_promoted": False}), flush=True)


def host(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # A nested writable output mount keeps all other workspace/data files read-only.
    output_relative = output.relative_to(ROOT)
    container_output = Path("/workspace") / output_relative
    manifest_relative = args.manifest.resolve().relative_to(ROOT)
    host_report = {"host_platform": platform.platform(), "host_architecture": platform.machine(),
                   "image": args.image, "phase": args.phase, "native_promoted": False}
    start = time.perf_counter()
    if args.build:
        host_report["build"] = run_logged(
            ["docker", "build", "--platform", "linux/arm64", "--progress", "plain",
             "-t", args.image, "-f", "scripts/Dockerfile", "."], output / "image_build.log")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    host_report["image_id"] = image["Id"]
    host_report["image_architecture"] = image["Architecture"]
    name = f"connectome-linux-arm64-{uuid.uuid4().hex[:10]}"
    host_report["container_name"] = name
    cidfile = output / f"container_{phase_label(args)}_{run_count(args)}seq.cid"
    if cidfile.exists():
        cidfile.unlink()
    command = ["docker", "run", "--rm", "--name", name, "--cidfile", str(cidfile),
               "--platform", "linux/arm64", "--cpus", "1", "--memory", str(MEMORY_LIMIT_BYTES),
               "--memory-swap", str(MEMORY_LIMIT_BYTES),
               "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=1g",
               "--user", f"{os.getuid()}:{os.getgid()}",
               "--mount", f"type=bind,src={ROOT},dst=/workspace,readonly",
               "--mount", f"type=bind,src={output},dst={container_output}",
               "--workdir", "/workspace", args.image, "python", "scripts/run_linux_checks.py", "--inside",
               "--phase", args.phase, "--output", str(container_output), "--manifest", str(Path("/workspace") / manifest_relative),
               "--smoke-count", str(args.smoke_count), "--repetitions", str(args.repetitions)]
    command += ["--stream-count", str(args.stream_count)]
    if args.candidate_config is not None:
        candidate_relative = args.candidate_config.resolve().relative_to(ROOT)
        command += ["--candidate-config", str(Path("/workspace") / candidate_relative)]
    if args.archive is not None:
        archive_relative = args.archive.resolve().relative_to(ROOT)
        command += ["--archive", str(Path("/workspace") / archive_relative)]
    if args.affinity:
        command.append("--affinity")
    if args.modes is not None:
        command += ["--modes", *args.modes]
    save(output / f"host_{phase_label(args)}_{run_count(args)}seq.json", host_report)
    try:
        host_report["run"] = run_logged(command, output / f"container_{phase_label(args)}_{run_count(args)}seq.log")
        host_report["completed"] = True
    except subprocess.CalledProcessError as error:
        host_report["run"] = getattr(error, "check_result", {})
        host_report["completed"] = False
        raise
    finally:
        host_report["seconds_including_build"] = time.perf_counter() - start
        host_report["container_id"] = cidfile.read_text().strip() if cidfile.exists() else None
        save(output / f"host_{phase_label(args)}_{run_count(args)}seq.json", host_report)


def phase_label(args):
    return args.phase + ("_affinity" if args.affinity else "")


def run_count(args):
    return args.stream_count if args.phase == "stream" else args.smoke_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--image", default=IMAGE)
    parser.add_argument("--phase", choices=["all", "prepare", "tests", "parity", "benchmark", "solutions", "stream", "zip"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "linux_arm64")
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts" / "manifest.json")
    parser.add_argument("--smoke-count", type=int, default=32)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--candidate-config", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--stream-count", type=int, default=256)
    parser.add_argument("--affinity", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=["reference", "run", "iobinding", "ortvaluevector", "native_row", "native_packed"],
                        help="Restrict the backend benchmark; does not affect solution comparisons")
    args = parser.parse_args()
    if not 1 <= args.smoke_count <= 32 or not 1 <= args.stream_count <= 256 or args.repetitions < 1:
        parser.error("smoke-count must be 1..32, stream-count must be 1..256 and repetitions must be positive")
    inside(args) if args.inside else host(args)


if __name__ == "__main__":
    main()
