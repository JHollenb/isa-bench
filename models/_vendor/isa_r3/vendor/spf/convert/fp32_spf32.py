"""FP32 ↔ SPF-32 conversion (spec §11.2).

Thin named wrapper over :mod:`spf.scalar.spf32` ``encode`` / ``decode`` so the
cross-format layer has one symbol per direction. The underlying encode takes
complex-valued input and returns a packed ``uint32`` word; decode is its
inverse returning ``complex128``.

Spec cost: ~10 cycles encode, ~8 cycles decode (C implementation at §11.2).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar import spf32

__all__ = ["fp32_to_spf32", "spf32_to_fp32"]


def fp32_to_spf32(z: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Encode FP32/complex64 values into SPF-32 words.

    Accepts any array-like of real or complex floats (scalar or array);
    reals are promoted via ``complex(x, 0)``. Returns ``ndarray[uint32]``
    of the same shape.
    """
    return spf32.encode(z)


def spf32_to_fp32(w: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Decode SPF-32 words to complex128.

    Return dtype is ``complex128`` (same as ``spf32.decode``) — cast to
    ``complex64`` at the call site if FP32 storage is required.
    """
    return spf32.decode(w)
