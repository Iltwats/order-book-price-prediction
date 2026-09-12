"""NumPy/ctypes-only runtime for the experimental FP32 native GRU backend.

The native implementation evaluates the pretrained equations, but floating-point
reduction/activation implementations differ from ORT. It is not automatically
eligible for submission: run recurrent parity and full validation first.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

WEIGHT_COUNT = 192_258


class NativeBackend:
    """Stateful single-thread backend. Instances are not safe for concurrent use."""

    def __init__(self, weights_path, library_path, layout="packed"):
        self._context = None
        if layout not in {"packed", "row"}:
            raise ValueError("layout must be 'packed' or 'row'")
        self.layout = layout
        self.weights_path = Path(weights_path).resolve()
        self.library_path = Path(library_path).resolve()
        values = np.load(self.weights_path, allow_pickle=False)
        if values.dtype != np.dtype(np.float32) or values.shape != (WEIGHT_COUNT,):
            raise ValueError("native weights must be a flat float32 array of 192258 values")
        values = np.ascontiguousarray(values)
        self._library = ctypes.CDLL(str(self.library_path))
        library = self._library
        library.cc_abi_version.argtypes = []
        library.cc_abi_version.restype = ctypes.c_int
        if library.cc_abi_version() != 1:
            raise RuntimeError("unsupported native backend ABI")
        library.cc_create.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        library.cc_create.restype = ctypes.c_void_p
        library.cc_reset.argtypes = [ctypes.c_void_p]
        library.cc_reset.restype = ctypes.c_int
        library.cc_step.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                    ctypes.c_void_p, ctypes.c_size_t]
        library.cc_step.restype = ctypes.c_int
        library.cc_hidden.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                      ctypes.c_void_p, ctypes.c_size_t]
        library.cc_hidden.restype = ctypes.c_int
        library.cc_destroy.argtypes = [ctypes.c_void_p]
        library.cc_destroy.restype = None
        library.cc_last_error.argtypes = []
        library.cc_last_error.restype = ctypes.c_char_p
        self._context = library.cc_create(values.ctypes.data, values.size, int(layout == "packed"))
        if not self._context:
            raise RuntimeError(self._error())
        self._step = library.cc_step

    def _error(self):
        message = self._library.cc_last_error()
        return message.decode("utf-8") if message else "native backend error"

    def _require_open(self):
        if self._context is None:
            raise RuntimeError("native backend is closed")

    def reset(self):
        self._require_open()
        if self._library.cc_reset(self._context):
            raise RuntimeError(self._error())

    def step(self, features):
        self._require_open()
        features = np.asarray(features, dtype=np.float32)
        if features.shape != (112,):
            raise ValueError("expected features with shape (112,)")
        features = np.ascontiguousarray(features)
        # Every result owns its memory; future calls cannot mutate retained outputs.
        prediction = np.empty(2, dtype=np.float32)
        if self._step(self._context, features.ctypes.data, 112, prediction.ctypes.data, 2):
            raise RuntimeError(self._error())
        return prediction

    def hidden_states(self):
        self._require_open()
        first = np.empty((1, 1, 128), dtype=np.float32)
        second = np.empty_like(first)
        if self._library.cc_hidden(self._context, first.ctypes.data, 128, second.ctypes.data, 128):
            raise RuntimeError(self._error())
        return first, second

    def close(self):
        if self._context is not None:
            self._library.cc_destroy(self._context)
            self._context = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()
