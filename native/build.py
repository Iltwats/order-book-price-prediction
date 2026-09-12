#!/usr/bin/env python3
"""Build the local native candidate with the system C++17 compiler.

No packages are installed. Linux produces .so; macOS produces .dylib. Artifacts
are specific to the current OS/architecture; a macOS build is not deployable to
the Linux judge. --sanitize runs a separate ASan/UBSan reference harness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "artifacts" / "native")
    parser.add_argument("--compiler", default=os.environ.get("CXX", "c++"))
    parser.add_argument("--test", action="store_true", help="run the independent C++ reference harness")
    parser.add_argument("--sanitize", action="store_true", help="also run ASan/UBSan harness")
    args = parser.parse_args()
    system = platform.system()
    if system not in {"Darwin", "Linux"}:
        parser.error("this build script supports macOS and Linux")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = shlex.split(args.compiler)
    source = root / "native" / "connectome_native.cpp"
    harness = root / "native" / "sanitizer_test.cpp"
    common = ["-std=c++17", "-Wall", "-Wextra", "-Wpedantic", "-fno-fast-math", "-ffp-contract=off"]
    release = [*common, "-O3", "-DNDEBUG", "-fvisibility=hidden"]
    extension = ".dylib" if system == "Darwin" else ".so"
    library = output / f"libconnectome_native{extension}"
    link = ["-dynamiclib"] if system == "Darwin" else ["-shared", "-fPIC"]
    command = [*compiler, *release, *link, str(source), "-o", str(library)]
    subprocess.run(command, check=True)
    checks = {}
    for enabled, name, flags in (
        (args.test, "reference", [*common, "-O3"]),
        (args.sanitize, "sanitized", [*common, "-O1", "-g", "-fno-omit-frame-pointer", "-fsanitize=address,undefined", "-fno-sanitize-recover=all"]),
    ):
        if not enabled:
            continue
        binary = output / f"native_test_{name}"
        subprocess.run([*compiler, *flags, str(source), str(harness), "-o", str(binary)], check=True)
        result = subprocess.run([str(binary)], check=True, text=True, capture_output=True)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        checks[name] = {"passed": True, "output": result.stdout.strip()}
    manifest = {
        "library_path": str(library), "os": system, "architecture": platform.machine(),
        "compiler": subprocess.run([*compiler, "--version"], check=True, text=True, capture_output=True).stdout.strip(),
        "command": command, "fast_math": False, "native_isa_flags": False,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "abi_version": 1, "weight_count": 192258, "checks": checks,
        "context_buffer_alignment_bytes": 64,
    }
    manifest_path = output / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Built {library}")
    print(f"Build evidence: {manifest_path}")


if __name__ == "__main__":
    main()
