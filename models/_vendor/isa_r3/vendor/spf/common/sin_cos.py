"""Mode-dispatched sin+cos wrapper. Default = SPF integer-phase deg-7 bit-trick."""

from __future__ import annotations

import ctypes

from spf.common.modes import resolve_mode
from spf.native._libspf32 import load_library


def sin_cos(theta_rad: float, mode: str | int = "default") -> tuple[float, float]:
    """Return (sin(theta_rad), cos(theta_rad)) under the given mode.

    INTPHASE_* modes quantize theta to the SPF 16-bit phase encoding
    (natural mod-2π wrap).
    """
    lib = load_library()
    s = ctypes.c_double(0.0)
    c = ctypes.c_double(0.0)
    lib.spf_sin_cos(float(theta_rad), ctypes.byref(s), ctypes.byref(c), resolve_mode(mode))
    return s.value, c.value
