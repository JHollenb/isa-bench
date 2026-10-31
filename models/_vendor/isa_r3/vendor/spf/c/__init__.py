"""Compatibility facade for the renamed :mod:`spf.native` package."""

from __future__ import annotations

from spf.native import fb7
from spf.native._lib import LIBSPF_AVAILABLE, LIBSPF_PATH, load_library
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
