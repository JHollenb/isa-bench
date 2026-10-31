"""Mode-dispatched atan2 wrapper. Default = SPF octant degree-5 bit-trick."""

from __future__ import annotations

from spf.common.modes import resolve_mode
from spf.native._libspf32 import load_library


def atan2(y: float, x: float, mode: str | int = "default") -> float:
    """Compute atan2(y, x) under the given mode (see spf.common.modes)."""
    lib = load_library()
    return float(lib.spf_atan2(float(y), float(x), resolve_mode(mode)))
