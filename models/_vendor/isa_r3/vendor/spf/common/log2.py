"""Mode-dispatched log2 wrapper. Default = SPF Mitchell deg-4 bit-trick."""

from __future__ import annotations

from spf.common.modes import resolve_mode
from spf.native._libspf32 import load_library


def log2(x: float, mode: str | int = "default") -> float:
    """Compute log2(x) under the given mode (see spf.common.modes)."""
    lib = load_library()
    return float(lib.spf_log2(float(x), resolve_mode(mode)))
