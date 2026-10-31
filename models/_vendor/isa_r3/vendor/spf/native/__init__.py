"""C-backed SPF-32 ops via ctypes bindings to ``libspf32``.

Loaded lazily so ``import spf.native`` succeeds even when the shared library
hasn't been built. Consumers should branch on :data:`LIBSPF_AVAILABLE`
before using the C entry points; the pure-Python reference at
:mod:`spf.scalar.spf32` and :mod:`spf.fft.spf32_fft` is always available.

Build: ``make -C native`` from the repo root. Override the library path via the
``SPF_LIBRARY`` environment variable when installing.
"""

from __future__ import annotations

from spf.native import fb7
from spf.native._libspf32 import LIBSPF_AVAILABLE, LIBSPF_PATH, load_library
from spf.native.fnt import LIBFERMAT_NTT_AVAILABLE, LIBFERMAT_NTT_PATH
from spf.native.fp32ref import LIBFP32REF_AVAILABLE, LIBFP32REF_PATH

__all__ = [
    "LIBFERMAT_NTT_AVAILABLE",
    "LIBFERMAT_NTT_PATH",
    "LIBFP32REF_AVAILABLE",
    "LIBFP32REF_PATH",
    "LIBSPF_AVAILABLE",
    "LIBSPF_PATH",
    "fb7",
    "load_library",
]
