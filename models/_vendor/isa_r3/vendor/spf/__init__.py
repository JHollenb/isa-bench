"""Spectral Phase Format reference implementation."""

from spf import block, convert, dsp, fdwm, fnt, pa
from spf.array import SPFArray, asarray, kria_session, register_backend
from spf.scalar import spf16, spf32, spf64

from . import renderer, v02

spf32_fmt: str = "spf32"
"""Dtype sentinel for :func:`asarray`; pass as ``dtype=spf.spf32_fmt``."""

__all__ = [
    "SPFArray",
    "asarray",
    "block",
    "convert",
    "dsp",
    "fdwm",
    "fnt",
    "kria_session",
    "pa",
    "register_backend",
    "renderer",
    "spf16",
    "spf32",
    "spf32_fmt",
    "spf64",
    "v02",
]
