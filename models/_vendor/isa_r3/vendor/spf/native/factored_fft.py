"""C-backed factored tan-tanh SPF-32 FFT (the ``profile="v02_factored"`` C path).

Binds ``spf32_factored_fft_rows`` from ``libspf_ocean_packed`` (source
``native/src/scalar/spf32/spf32_factored_add.c``, declared in
``native/include/spf32_factored_add.h``).  The library is optional; branch on
:func:`available` before calling.  The pure-Python reference is
:func:`spf.fft.factored.spf_fft_radix2_factored` and is word-identical to this
(``tests/native_bindings/test_factored_add_native.py``,
``tests/fft/test_factored_fft.py``).

Only the forward transform has a C entry point; the inverse is the exact
conjugate + ``1/N`` log shift wrapper on the Python side
(:func:`spf.fft.factored.spf_ifft_radix2_factored`), whose per-word arithmetic
(conj, log-shift) is already C-parity-tested elsewhere.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["FACTORED_FFT_AVAILABLE", "available", "fft", "rom_version"]

Word = npt.NDArray[np.uint32]

_PRECISION = {"fast": 0, "balanced": 1, "accurate": 2, "exact": 3}


@lru_cache(maxsize=1)
def _library() -> Any:
    from spf.renderer import _ocean_display_native as ocean

    if not ocean.available():
        return None
    lib = ctypes.CDLL(str(ocean.library_path()))
    lib.spf32_factored_fft_rows.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    lib.spf32_factored_fft_rows.restype = ctypes.c_int
    lib.spf32_factored_rom_version.restype = ctypes.c_char_p
    return lib


def available() -> bool:
    """True when ``libspf_ocean_packed`` (with the factored FFT) is loadable."""

    try:
        return _library() is not None
    except (OSError, AttributeError):  # unloadable or stale build lacking the symbol
        return False


FACTORED_FFT_AVAILABLE: bool = available()


def rom_version() -> str:
    """Return the C factored-ROM version string (for parity gating)."""

    lib = _library()
    if lib is None:
        raise RuntimeError("libspf_ocean_packed not built (make -C native)")
    return str(lib.spf32_factored_rom_version().decode())


def fft(x: npt.ArrayLike, *, precision: str = "balanced") -> Word:
    """C factored radix-2 forward FFT of a 1-D or batched (``..., N``) word plane.

    Mirrors :func:`spf.fft.factored.spf_fft_radix2_factored`.  Each length-N row
    (power of two, ``N <= SPF32_FB3_MAX_N``) is transformed; leading batch axes
    are preserved.
    """

    lib = _library()
    if lib is None:
        raise RuntimeError("libspf_ocean_packed not built (make -C native)")
    arr = np.ascontiguousarray(np.asarray(x), dtype=np.uint32)
    if arr.ndim < 1:
        raise ValueError(f"factored FFT input must have >= 1 dimension; got shape {arr.shape}")
    n = int(arr.shape[-1])
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"factored FFT length N must be a power of 2 >= 2; got N={n}")
    rows = arr.reshape(-1, n)
    rows = np.ascontiguousarray(rows, dtype=np.uint32)
    out = np.empty_like(rows)
    in_ptr = rows.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    out_ptr = out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
    rc = int(
        lib.spf32_factored_fft_rows(
            in_ptr, out_ptr, ctypes.c_size_t(rows.shape[0]), ctypes.c_size_t(n),
            ctypes.c_int(_PRECISION.get(precision, 1)), None,
        )
    )
    if rc != 0:
        raise RuntimeError(f"spf32_factored_fft_rows returned {rc} (invalid length or allocation failure)")
    return out.reshape(arr.shape)
