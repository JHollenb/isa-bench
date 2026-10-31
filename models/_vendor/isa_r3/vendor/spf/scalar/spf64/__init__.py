"""SPF-64 scalar format public surface.

Spec: docs/spf-formats-and-test-plans.md §4. Thin wrapper binding
:data:`spf.scalar._constants.SPF64` into every op from :mod:`spf.scalar._ops`.

Bit layout: Q15.16 L in the upper 32 bits, unsigned 32-bit Θ in the lower.
Dynamic range r ∈ [2⁻³²⁷⁶⁷·⁹⁹⁹⁹⁸, 2³²⁷⁶⁷·⁹⁹⁹⁹⁸]. Phase resolution
2π / 2³² ≈ 1.46 × 10⁻⁹ rad — matches AD9914-grade DDS. Twiddle factor
``W_N^k`` is exact when N divides 2³².
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar import _ops
from spf.scalar._constants import SPF64

FORMAT = SPF64

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

Word = npt.NDArray[np.uint64]
Complex = npt.NDArray[np.complex128]


def encode(z: npt.ArrayLike) -> Word:
    """Encode complex values into SPF-64 words."""
    return _ops.encode(z, FORMAT).astype(np.uint64, copy=False)


def decode(w: npt.ArrayLike) -> Complex:
    """Decode SPF-64 words back to complex128."""
    return _ops.decode(w, FORMAT)


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-64 multiply: ``L_out = L_a + L_b``, ``Θ_out = Θ_a + Θ_b`` (mod 2³²)."""
    return _ops.mul(a, b, FORMAT).astype(np.uint64, copy=False)


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-64 divide: ``L_out = L_a − L_b``, ``Θ_out = Θ_a − Θ_b``."""
    return _ops.div(a, b, FORMAT).astype(np.uint64, copy=False)


def conj(a: npt.ArrayLike) -> Word:
    """Complex conjugate: L unchanged, Θ ↦ (−Θ) mod 2³²."""
    return _ops.conj(a, FORMAT).astype(np.uint64, copy=False)


def abs2(a: npt.ArrayLike) -> Word:
    """|z|² as a positive-real SPF-64 word: ``L_out = 2·L_a``, Θ_out = 0."""
    return _ops.abs2(a, FORMAT).astype(np.uint64, copy=False)


def neg(a: npt.ArrayLike) -> Word:
    """Unary negation: Θ ↦ Θ + π (XOR with 0x80000000)."""
    return _ops.neg(a, FORMAT).astype(np.uint64, copy=False)


def twiddle(k: npt.ArrayLike, n: int) -> Word:
    """DFT twiddle ``W_N^k = e^(2πi·k/N)`` as a unit-magnitude SPF-64 word."""
    return _ops.twiddle(k, n, FORMAT).astype(np.uint64, copy=False)


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-64 complex addition (Sb()-based; spec §2.4)."""
    return _ops.add(a, b, FORMAT).astype(np.uint64, copy=False)


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """Complex subtraction: ``a − b = a + (−b)``."""
    return _ops.sub(a, b, FORMAT).astype(np.uint64, copy=False)


def butterfly(a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike) -> tuple[Word, Word]:
    """Radix-2 DIT butterfly: ``a' = a + W·b``, ``b' = a − W·b``."""
    top, bot = _ops.butterfly(a, b, w, FORMAT)
    return top.astype(np.uint64, copy=False), bot.astype(np.uint64, copy=False)
