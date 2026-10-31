"""C-backed Goertzel single-bin DFT.

Mirrors :mod:`spf.fft.goertzel`. The C entry points take raw pointers; these
wrappers marshal NumPy arrays into ctypes buffers and raise on invalid
length / bin index (matching the pure-Python side's validation).
"""

from __future__ import annotations

import ctypes
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from spf.native._libspf32 import load_library

__all__ = [
    "goertzel_bin_fp32_c",
    "goertzel_bin_spf32",
    "goertzel_bins_spf32",
]


def _validate_n_k(n: int, k: int) -> None:
    if n < 2:
        raise ValueError(f"Goertzel requires N >= 2; got N={n}")
    if not (0 <= int(k) < n):
        raise ValueError(f"bin index k must be in [0, {n}); got k={k}")


def goertzel_bin_fp32_c(x_re: npt.ArrayLike,
                        x_im: npt.ArrayLike | None,
                        k: int) -> complex:
    """Goertzel one bin on raw FP32 input. Returns a Python complex.

    ``x_im`` may be ``None`` for a real-input Goertzel. This is the symmetric
    partner of :func:`spf.fft.goertzel.goertzel_bin_fp32` but operates on
    pre-separated FP32 lanes so it can be used to probe the C kernel's
    arithmetic error without going through SPF-32 encode/decode.
    """
    lib = load_library()
    xr = np.ascontiguousarray(np.asarray(x_re), dtype=np.float32)
    if xr.ndim != 1:
        raise ValueError(f"x_re must be 1-D; got shape {xr.shape}")
    n = int(xr.shape[0])
    _validate_n_k(n, int(k))
    if x_im is None:
        xi_ptr: ctypes._Pointer[ctypes.c_float] | None = None
        xi_arr: np.ndarray | None = None
    else:
        xi_arr = np.ascontiguousarray(np.asarray(x_im), dtype=np.float32)
        if xi_arr.shape != xr.shape:
            raise ValueError(
                f"x_im shape {xi_arr.shape} must match x_re shape {xr.shape}"
            )
        xi_ptr = xi_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_re = ctypes.c_float(0.0)
    out_im = ctypes.c_float(0.0)
    lib.spf32_goertzel_bin_fp32(
        xr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        xi_ptr,
        ctypes.c_size_t(n),
        ctypes.c_int64(int(k)),
        ctypes.byref(out_re),
        ctypes.byref(out_im),
    )
    return complex(out_re.value, out_im.value)


def goertzel_bin_spf32(x_spf32: npt.ArrayLike, k: int) -> np.uint32:
    """One Goertzel bin with SPF-32 on both ends. Returns a single SPF-32 word."""
    lib = load_library()
    arr = np.ascontiguousarray(np.asarray(x_spf32), dtype=np.uint32)
    if arr.ndim != 1:
        raise ValueError(f"x_spf32 must be 1-D; got shape {arr.shape}")
    n = int(arr.shape[0])
    _validate_n_k(n, int(k))
    return np.uint32(
        lib.spf32_goertzel_bin(
            arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            ctypes.c_size_t(n),
            ctypes.c_int64(int(k)),
        )
    )


def goertzel_bins_spf32(x_spf32: npt.ArrayLike, ks: Iterable[int]) -> np.ndarray:
    """K-bin Goertzel with SPF-32 boundaries. Returns ``ndarray[uint32]``.

    Shares the O(N) SPF-32 decode pass across all bins — that is the scalar
    ROI of the sparse-bin pitch.
    """
    lib = load_library()
    arr = np.ascontiguousarray(np.asarray(x_spf32), dtype=np.uint32)
    if arr.ndim != 1:
        raise ValueError(f"x_spf32 must be 1-D; got shape {arr.shape}")
    n = int(arr.shape[0])
    if n < 2:
        raise ValueError(f"Goertzel requires N >= 2; got N={n}")
    ks_arr = np.ascontiguousarray(np.asarray(list(ks)), dtype=np.int64)
    if ks_arr.ndim != 1:
        raise ValueError(f"ks must be 1-D; got shape {ks_arr.shape}")
    for k in ks_arr:
        if not (0 <= int(k) < n):
            raise ValueError(f"bin index {int(k)} must be in [0, {n})")
    out = np.empty(ks_arr.size, dtype=np.uint32)
    scratch_re = np.empty(n, dtype=np.float32)
    scratch_im = np.empty(n, dtype=np.float32)
    rc = int(lib.spf32_goertzel_bins(
        arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n),
        ks_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.c_size_t(ks_arr.size),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        scratch_re.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        scratch_im.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    ))
    if rc != 0:
        raise RuntimeError(f"spf32_goertzel_bins returned {rc}")
    return out
