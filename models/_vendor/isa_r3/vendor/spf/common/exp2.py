"""Mode-dispatched exp2 wrapper. Default = SPF argred deg-4 bit-trick."""

from __future__ import annotations

from spf.common.modes import resolve_mode
from spf.native._libspf32 import load_library


def exp2(x: float, mode: str | int = "default") -> float:
    """Compute exp2(x) under the given mode (see spf.common.modes)."""
    lib = load_library()
    return float(lib.spf_exp2(float(x), resolve_mode(mode)))
