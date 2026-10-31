"""Common transcendental wrapper mode enum.

Mirrors `native/include/spf_common_modes.h`. Keep in lockstep with the C
header — the integer values are the ABI between Python and libspf32.
"""

from __future__ import annotations

import enum


class CommonMode(enum.IntEnum):
    DEFAULT     = 0
    LIBM        = 1
    PADE        = 2
    OCTANT_D5   = 3
    OCTANT_D7   = 4
    SCHRAUDOLPH = 5
    MITCHELL_D3 = 6
    MITCHELL_D4 = 7
    ARGRED_D3   = 8
    ARGRED_D4   = 9
    INTPHASE_D5 = 10
    INTPHASE_D7 = 11


_NAME_TO_MODE: dict[str, CommonMode] = {m.name.lower(): m for m in CommonMode}


def resolve_mode(name_or_int: str | int | CommonMode) -> int:
    """Map a mode name / int to its integer value."""
    if isinstance(name_or_int, CommonMode):
        return int(name_or_int)
    if isinstance(name_or_int, int):
        return int(name_or_int)
    key = str(name_or_int).strip().lower()
    if key not in _NAME_TO_MODE:
        raise ValueError(
            f"unknown common mode {name_or_int!r}; "
            f"want one of {sorted(_NAME_TO_MODE)}"
        )
    return int(_NAME_TO_MODE[key])
