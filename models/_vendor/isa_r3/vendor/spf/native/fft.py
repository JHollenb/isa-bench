"""C-backed SPF-32 FFT driver.

Mirrors :mod:`spf.fft.spf32_fft`. The C entry points take raw pointers;
these wrappers allocate the output array, call through ctypes, and raise
on invalid length (matching the Python side's ``_validate_length``).
"""

from __future__ import annotations

import ctypes

import numpy as np
import numpy.typing as npt

from spf.native._libspf32 import load_library

__all__ = ["fft", "ifft"]

Word = npt.NDArray[np.uint32]


def _run(entry: str, x: npt.ArrayLike) -> Word:
    lib = load_library()
    arr = np.ascontiguousarray(np.asarray(x), dtype=np.uint32)
    if arr.ndim != 1:
        raise ValueError(f"FFT input must be 1-D; got shape {arr.shape}")
    n = int(arr.shape[0])
    if n < 2 or (n & (n - 1)) != 0 or n > (1 << 16):
        raise ValueError(f"FFT length N must be a power of 2 in [2, 2^16]; got N={n}")
    out = np.empty(n, dtype=np.uint32)
    in_ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    rc = int(getattr(lib, entry)(in_ptr, out_ptr, ctypes.c_int(n)))
    if rc != 0:
        raise RuntimeError(f"{entry} returned {rc} (invalid length or allocation failure)")
    return out


def fft(x: npt.ArrayLike) -> Word:
    """C port of :func:`spf.fft.spf32_fft.fft`."""
    return _run("spf32_fft", x)


def ifft(x: npt.ArrayLike) -> Word:
    """C port of :func:`spf.fft.spf32_fft.ifft`."""
    return _run("spf32_ifft", x)
