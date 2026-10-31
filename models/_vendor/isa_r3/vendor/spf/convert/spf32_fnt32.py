"""SPF-32 ↔ FNT-32 real-integer bridge (spec §11, table row line 763).

SPF-32 stores a complex number; FNT-32 operates in the integer ring
``Z_{2^32+1}`` — the bridge only makes sense for real scalars. The spec row
in §11.1 reads "decode to float, round to int, ~12 cycles" but gives no C
reference, so this module locks in the following conventions:

**Forward** (``spf32_to_fnt32``):

1. ``spf32.decode(w)`` → complex128.
2. Take ``Re(z)`` (the imaginary part is discarded, consistent with the
   "decode to float" phrasing in the spec).
3. Round to the nearest integer (Python's banker's rounding via ``np.rint``).
4. Reduce mod ``F = 2**32 + 1``. Negatives map to their canonical ring
   representative in ``[0, F)``.

**Reverse** (``fnt32_to_spf32``):

1. Map ring element ``r`` to its short (signed) representative: if
   ``r > F/2``, it represents ``r - F``.
2. Cast to ``complex(x, 0)`` and encode via ``spf32.encode``.

**Overflow behaviour.** SPF-32's representable magnitude range is
``|z| ∈ [2⁻¹²⁸, 2¹²⁸⁻ᵉ]`` (§2.1), far wider than the integer range we care
about here, so magnitude clamping is not a concern. Integer inputs outside
``(-F/2, F/2]`` are silently reduced mod F — that's the natural ring
behavior and matches how an NTT pipeline would handle them.

The bridge is not a round-trip over arbitrary complex values — by design it
projects to the real axis. Round-trip exactness holds for integer-valued
real scalars in ``(-F/2, F/2]`` whose magnitude is SPF-32 representable
(Q7.8 log-magnitude gives ~0.4% relative error; we test exactness only for
``|x| ≤ 2²⁰`` which stays well within the log-magnitude ULP floor).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fnt import F
from spf.scalar import spf32

__all__ = ["fnt32_to_spf32", "spf32_to_fnt32"]


def spf32_to_fnt32(w: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Convert SPF-32 word(s) to FNT-32 ring element(s).

    Projects to the real axis (imaginary part discarded), rounds to the
    nearest integer, and reduces mod ``F = 2^32 + 1``. Returns ``uint64``
    because ``F`` itself needs 33 bits.
    """
    z = spf32.decode(w)
    x = np.rint(np.real(z)).astype(np.int64)
    # Python int mod handles negatives correctly; numpy mod on int64 also does.
    r = np.mod(x, np.int64(F)).astype(np.uint64)
    return r


def fnt32_to_spf32(r: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Convert FNT-32 ring element(s) to SPF-32 word(s) on the real axis.

    Maps ``r`` to its short signed representative in ``(-F/2, F/2]``, casts
    to a real complex value, and encodes via :func:`spf32.encode`.
    """
    r_arr = np.asarray(r, dtype=np.uint64)
    half = np.uint64(F // 2)
    # r > F/2 ⇒ signed value is r - F; else r itself. F ≤ 2^33, so int64 suffices.
    signed = np.where(
        r_arr > half,
        r_arr.astype(np.int64) - np.int64(F),
        r_arr.astype(np.int64),
    )
    z = signed.astype(np.float64) + 0j
    return spf32.encode(z)
