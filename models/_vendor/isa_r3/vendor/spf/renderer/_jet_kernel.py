"""Optional C fused jet emitter; explicit portable fallback needs no C build."""

from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np


class JetStats(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        (name, ctypes.c_uint64) for name in ("evaluated", "retained", "low", "high", "tiles", "culled")
    ] + [
        ("max_log", ctypes.c_int64),
        ("integer_samples", ctypes.c_uint64),
        ("fractional_samples", ctypes.c_uint64),
        ("max_output", ctypes.c_double),
    ]


@lru_cache(maxsize=1)
def kernel() -> Any:
    root = Path(__file__).resolve().parents[3]
    suffix = "dylib" if sys.platform == "darwin" else "so"
    explicit = os.environ.get("SPF_SFR_LIBRARY")
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            root / "native" / f"libspf_sfr.{suffix}",
            root / "native" / "build" / f"libspf_sfr.{suffix}",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        lib = ctypes.CDLL(str(path))
        version = lib.spf_sfr_jet_abi_version
        version.argtypes = []
        version.restype = ctypes.c_int
        if version() != 2:
            raise RuntimeError("SFR jet kernel ABI mismatch; rebuild the native target")
        fn = lib.spf_sfr_jet_project
        ptr = ctypes.c_void_p
        fn.argtypes = [ptr, ptr] + [ctypes.c_int] * 9 + [ctypes.c_double] * 7 + [ptr] * 3 + [ctypes.c_int] + [ptr] * 4
        fn.restype = None
        return fn
    if explicit:
        raise RuntimeError(f"requested SFR kernel library does not exist: {explicit}")
    return None


def pointer(array: np.ndarray | None) -> ctypes.c_void_p | None:
    return None if array is None else ctypes.c_void_p(array.ctypes.data)


def coefficients(jet: Any) -> np.ndarray:
    return np.asarray([jet.value, jet.grad_x, jet.grad_y, jet.hess_xx, jet.hess_xy, jet.hess_yy], dtype=np.float64)


class JetCommand(ctypes.Structure):
    """Versioned native command, with output offsets relative to a bound buffer."""

    _fields_: ClassVar[list[tuple[str, Any]]] = (
        [(name, ctypes.c_void_p) for name in ("lc", "pc")]
        + [
            (name, ctypes.c_int)
            for name in ("height", "width", "row_stride", "out_row_stride", "out_pixel_stride", "x0", "x1", "y0", "y1")
        ]
        + [
            (name, ctypes.c_double)
            for name in ("ox", "oy", "lscale", "pscale", "threshold", "lerror", "residual_upper")
        ]
        + [(name, ctypes.c_void_p) for name in ("lr", "pr", "lut")]
        + [("semantic", ctypes.c_int)]
        + [(name, ctypes.c_size_t) for name in ("out_offset", "log_offset", "phase_offset")]
    )


@lru_cache(maxsize=1)
def batch_kernel() -> Any:
    fn = kernel()
    if fn is None:
        return None
    # ctypes functions retain their library owner; fetch the batch entry from
    # that same loaded image so environment overrides cannot mix ABIs.
    for owner in fn._objects.values():
        if isinstance(owner, ctypes.CDLL):
            batch: Any = owner.spf_sfr_jet_batch
            batch.argtypes = [ctypes.POINTER(JetCommand), ctypes.c_size_t] + [ctypes.c_void_p] * 4
            batch.restype = None
            return batch
    raise RuntimeError("SFR native library owner unavailable")


@lru_cache(maxsize=1)
def steady_batch_kernel() -> Any:
    """Prepared output-only batch entry with frozen diagnostic accounting."""

    fn = kernel()
    if fn is None:
        return None
    for owner in fn._objects.values():
        if isinstance(owner, ctypes.CDLL):
            try:
                result: Any = owner.spf_sfr_jet_batch_steady
            except AttributeError:
                return None
            result.argtypes = [ctypes.POINTER(JetCommand), ctypes.c_size_t, ctypes.c_void_p]
            result.restype = None
            return result
    raise RuntimeError("SFR native library owner unavailable")


@lru_cache(maxsize=8)
def native_function(name: str, argtypes: tuple[Any, ...]) -> Any:
    fn = kernel()
    if fn is None:
        return None
    for owner in fn._objects.values():
        if isinstance(owner, ctypes.CDLL):
            result = getattr(owner, name)
            result.argtypes = list(argtypes)
            result.restype = None
            return result
    raise RuntimeError("SFR native library owner unavailable")
