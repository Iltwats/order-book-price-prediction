"""Self-contained inference engine for compiled C++ SIMD ensemble + ONNX/PyTorch fallback.

Placed inside `connectome/` so `packaging.py` can bundle it into the standalone
submission ZIP without external repository dependencies.

At `PredictionModel.__init__()`:
1. Compiles `ensemble_engine.cpp` with `c++ -std=c++17 -O3 -march=native` (~0.3s in
   the judge's `python:3.11-slim-bookworm` container which installs `build-essential`).
2. Loads all ensemble weights (`ensemble_weights.npz`) into 64-byte-aligned C++ memory.
3. Falls back cleanly to a pure-NumPy/ONNX implementation if `c++` is ever absent.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class CompiledEnsembleBackend:
    """Zero-allocation single-thread C++17 SIMD streaming predictor."""

    def __init__(self, cpp_source: str | Path, weights_npz: str | Path):
        self._ctx = None
        self._lib = None
        cpp_source = Path(cpp_source).resolve()
        weights_npz = Path(weights_npz).resolve()

        ext = ".dylib" if platform.system() == "Darwin" else ".so"
        src_hash = hashlib.sha256(cpp_source.read_bytes()).hexdigest()[:16]
        cache_dir = Path(tempfile.gettempdir()) / f"connectome_ens_{os.getuid()}_{src_hash}"
        cache_dir.mkdir(parents=True, exist_ok=True)
        lib_path = cache_dir / f"libensemble{ext}"

        if not lib_path.exists():
            tmp_lib = cache_dir / f"libensemble.{os.getpid()}{ext}"
            compiler = os.environ.get("CXX", "c++")
            isa_flag = "-mcpu=native" if platform.system() == "Darwin" else "-march=native"
            link_flags = ["-dynamiclib"] if platform.system() == "Darwin" else ["-shared", "-fPIC"]
            cmd = [compiler, "-std=c++17", "-O3", isa_flag, "-DNDEBUG", *link_flags, str(cpp_source), "-o", str(tmp_lib)]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError:
                # Portable fallback without -march=native
                cmd_portable = [compiler, "-std=c++17", "-O3", "-DNDEBUG", *link_flags, str(cpp_source), "-o", str(tmp_lib)]
                subprocess.run(cmd_portable, check=True, capture_output=True, text=True)
            os.replace(tmp_lib, lib_path)

        lib = ctypes.CDLL(str(lib_path))
        lib.ens_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.ens_create.restype = ctypes.c_void_p
        lib.ens_load_onnx_gru.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 8 + [ctypes.c_float]
        lib.ens_load_sorted_gru.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 16 + [ctypes.c_float]
        lib.ens_load_tcn80.argtypes = [ctypes.c_void_p] + [ctypes.c_void_p] * 12 + [ctypes.c_float]
        lib.ens_load_calibrator.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p] * 6
        lib.ens_reset.argtypes = [ctypes.c_void_p]
        lib.ens_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        lib.ens_step.restype = ctypes.c_int
        lib.ens_destroy.argtypes = [ctypes.c_void_p]
        self._lib = lib

        data = np.load(weights_npz, allow_pickle=False)
        n_onnx = int(data["n_onnx_gru"])
        has_sgru = int(data["has_sorted_gru"])
        has_tcn = int(data["has_tcn80"])
        ctx = lib.ens_create(n_onnx, has_sgru, has_tcn)
        if not ctx:
            raise RuntimeError("failed to allocate C++ ensemble context")
        self._ctx = ctx

        def c_arr(key: str) -> np.ndarray:
            return np.ascontiguousarray(data[key], dtype=np.float32)

        for idx in range(n_onnx):
            arrs = [c_arr(f"onnx_{idx}_{k}") for k in ("w0", "r0", "b0", "w1", "r1", "b1", "hw", "hb")]
            bw = float(data[f"onnx_{idx}_blend"])
            lib.ens_load_onnx_gru(ctx, idx, *[a.ctypes.data for a in arrs], bw)

        if has_sgru:
            keys = ("in_mu", "in_sd", "stem_w", "stem_b",
                    "w0", "r0", "bi0", "bh0", "w1", "r1", "bi1", "bh1",
                    "h1_w", "h1_b", "h2_w", "h2_b")
            arrs = [c_arr(f"sgru_{k}") for k in keys]
            bw = float(data["sgru_blend"])
            lib.ens_load_sorted_gru(ctx, *[a.ctypes.data for a in arrs], bw)

        if has_tcn:
            keys = ("in_mu", "in_sd", "stem_w", "stem_b",
                    "conv_w", "conv_b", "proj_w", "proj_b",
                    "h1_w", "h1_b", "h2_w", "h2_b")
            arrs = [c_arr(f"tcn_{k}") for k in keys]
            bw = float(data["tcn_blend"])
            lib.ens_load_tcn80(ctx, *[a.ctypes.data for a in arrs], bw)

        cal_seeds = int(data["cal_seeds"])
        if cal_seeds > 0:
            cal_in = int(data["cal_in_dim"])
            cal_h = int(data["cal_hidden"])
            cal_arrs = [c_arr(f"cal_{k}") for k in ("mu", "sd", "w1", "b1", "w2", "b2")]
            lib.ens_load_calibrator(ctx, cal_in, cal_h, cal_seeds, *[a.ctypes.data for a in cal_arrs])

        self._step_fn = lib.ens_step
        self._reset_fn = lib.ens_reset
        self.sequence = None
        self.previous_step = None

    def reset(self):
        if self._ctx is None:
            raise RuntimeError("ensemble backend closed")
        self._reset_fn(self._ctx)

    def predict(self, data_point) -> np.ndarray | None:
        seq = int(data_point.seq_ix)
        step = int(data_point.step_in_seq)
        need = bool(data_point.need_prediction)
        if step < 0 or step >= 20_000 or need != (step >= 99):
            raise ValueError("invalid sequence step or warm-up flag")
        if seq != self.sequence:
            if step != 0:
                raise ValueError("a new sequence must start at step zero")
            self._reset_fn(self._ctx)
            self.sequence = seq
            self.previous_step = None
        if self.previous_step is not None and step != self.previous_step + 1:
            raise ValueError("rows must arrive in sequence order")
        features = np.ascontiguousarray(data_point.state, dtype=np.float32)
        if features.shape != (112,):
            raise ValueError("expected 112 input features")
        out = np.empty(2, dtype=np.float32)
        rc = self._step_fn(self._ctx, features.ctypes.data, 1 if need else 0, out.ctypes.data)
        if rc != 0:
            raise ValueError(f"nonfinite input or prediction in C++ ensemble engine (code {rc})")
        self.previous_step = step
        return out if need else None

    def close(self):
        if self._ctx is not None and self._lib is not None:
            self._lib.ens_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()
