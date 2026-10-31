"""SPF-128 scalar format public surface.

Spec: docs/spf-formats-and-test-plans.md §5. Thin wrapper binding
:data:`spf.scalar._constants.SPF128` into every op from :mod:`spf.scalar._ops`.

Bit layout: Q31.32 L in the upper 64 bits, unsigned 64-bit Θ in the lower.
Dynamic range r ∈ [2⁻²¹⁴⁷⁴⁸³⁶⁴⁸, 2²¹⁴⁷⁴⁸³⁶⁴⁷·…]. Phase resolution
2π / 2⁶⁴ ≈ 3.4 × 10⁻¹⁹ rad. Twiddle factor ``W_N^k`` is exact when N divides 2⁶⁴.

**Word storage.** numpy has no 128-bit integer dtype, so SPF-128 words are
arbitrary-precision Python ints carried in ``object`` arrays — pure Python, no C,
slow by design. Unlike the SPF-16/32/64 wrappers (which down-cast to a fixed
unsigned dtype), this module returns ``object`` arrays unchanged; casting to a
narrower numpy integer would silently truncate the 128-bit word.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar import _ops
from spf.scalar._constants import SPF128

FORMAT = SPF128

__all__ = [
    "FORMAT",
    "abs2",
    "add",
    "butterfly",
    "conj",
    "decode",
    "div",
    "encode",
    "mul",
    "neg",
    "sub",
    "twiddle",
]

Word = npt.NDArray[np.object_]
Complex = npt.NDArray[np.complex128]


def _obj(w: npt.NDArray[np.generic]) -> Word:
    """Coerce an op result into an ``object`` array of Python-int words."""
    return np.asarray(w, dtype=object)


def encode(z: npt.ArrayLike) -> Word:
    """Encode complex values into SPF-128 words (object array of Python ints)."""
    return _obj(_ops.encode(z, FORMAT))


def decode(w: npt.ArrayLike) -> Complex:
    """Decode SPF-128 words back to complex128."""
    return _ops.decode(w, FORMAT)


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-128 multiply: ``L_out = L_a + L_b``, ``Θ_out = Θ_a + Θ_b`` (mod 2⁶⁴)."""
    return _obj(_ops.mul(a, b, FORMAT))


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-128 divide: ``L_out = L_a − L_b``, ``Θ_out = Θ_a − Θ_b``."""
    return _obj(_ops.div(a, b, FORMAT))


def conj(a: npt.ArrayLike) -> Word:
    """Complex conjugate: L unchanged, Θ ↦ (−Θ) mod 2⁶⁴."""
    return _obj(_ops.conj(a, FORMAT))


def abs2(a: npt.ArrayLike) -> Word:
    """|z|² as a positive-real SPF-128 word: ``L_out = 2·L_a``, Θ_out = 0."""
    return _obj(_ops.abs2(a, FORMAT))


def neg(a: npt.ArrayLike) -> Word:
    """Unary negation: Θ ↦ Θ + π (XOR with 2⁶³)."""
    return _obj(_ops.neg(a, FORMAT))


def twiddle(k: npt.ArrayLike, n: int) -> Word:
    """DFT twiddle ``W_N^k = e^(2πi·k/N)`` as a unit-magnitude SPF-128 word."""
    return _obj(_ops.twiddle(k, n, FORMAT))


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-128 complex addition (Sb()-based; spec §2.4).

    The Sb() core evaluates in FP64, so ADD does not preserve the full 64-bit
    L/Θ precision the exact integer ops (mul/div/conj/abs2/twiddle) do — it is
    accurate to FP64, not to the format's ULP. Magnitude/phase still round-trip
    to the ~10⁻¹⁵ relative floor of double precision.
    """
    return _obj(_ops.add(a, b, FORMAT))


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """Complex subtraction: ``a − b = a + (−b)``."""
    return _obj(_ops.sub(a, b, FORMAT))


def butterfly(a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike) -> tuple[Word, Word]:
    """Radix-2 DIT butterfly: ``a' = a + W·b``, ``b' = a − W·b``."""
    top, bot = _ops.butterfly(a, b, w, FORMAT)
    return _obj(top), _obj(bot)
