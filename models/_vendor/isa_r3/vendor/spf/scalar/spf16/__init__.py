"""SPF-16 scalar format public surface.

Spec: docs/spf-formats-and-test-plans.md §3. Thin wrapper binding
:data:`spf.scalar._constants.SPF16` into every op from :mod:`spf.scalar._ops`.

Bit layout: Q3.4 L in the upper 8 bits, unsigned 8-bit Θ in the lower.
Dynamic range r ∈ [2⁻⁷·⁹³⁷⁵, 2⁷·⁹³⁷⁵] ≈ [0.0041, 245]. Twiddle factor
``W_N^k`` is exact when N divides 2⁸ = 256.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar import _ops
from spf.scalar._constants import SPF16

FORMAT = SPF16

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

Word = npt.NDArray[np.uint16]
Complex = npt.NDArray[np.complex128]


def encode(z: npt.ArrayLike) -> Word:
    """Encode complex values into SPF-16 words."""
    return _ops.encode(z, FORMAT).astype(np.uint16, copy=False)


def decode(w: npt.ArrayLike) -> Complex:
    """Decode SPF-16 words back to complex128."""
    return _ops.decode(w, FORMAT)


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-16 multiply: ``L_out = L_a + L_b``, ``Θ_out = Θ_a + Θ_b`` (mod 2⁸)."""
    return _ops.mul(a, b, FORMAT).astype(np.uint16, copy=False)


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-16 divide: ``L_out = L_a − L_b``, ``Θ_out = Θ_a − Θ_b``."""
    return _ops.div(a, b, FORMAT).astype(np.uint16, copy=False)


def conj(a: npt.ArrayLike) -> Word:
    """Complex conjugate: L unchanged, Θ ↦ (−Θ) mod 2⁸."""
    return _ops.conj(a, FORMAT).astype(np.uint16, copy=False)


def abs2(a: npt.ArrayLike) -> Word:
    """|z|² as a positive-real SPF-16 word: ``L_out = 2·L_a``, Θ_out = 0."""
    return _ops.abs2(a, FORMAT).astype(np.uint16, copy=False)


def neg(a: npt.ArrayLike) -> Word:
    """Unary negation: Θ ↦ Θ + π (XOR lower-8 with 0x80)."""
    return _ops.neg(a, FORMAT).astype(np.uint16, copy=False)


def twiddle(k: npt.ArrayLike, n: int) -> Word:
    """DFT twiddle ``W_N^k = e^(2πi·k/N)`` as a unit-magnitude SPF-16 word."""
    return _ops.twiddle(k, n, FORMAT).astype(np.uint16, copy=False)


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-16 complex addition (Sb()-based; spec §2.4)."""
    return _ops.add(a, b, FORMAT).astype(np.uint16, copy=False)


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """Complex subtraction: ``a − b = a + (−b)``."""
    return _ops.sub(a, b, FORMAT).astype(np.uint16, copy=False)


def butterfly(a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike) -> tuple[Word, Word]:
    """Radix-2 DIT butterfly: ``a' = a + W·b``, ``b' = a − W·b``."""
    top, bot = _ops.butterfly(a, b, w, FORMAT)
    return top.astype(np.uint16, copy=False), bot.astype(np.uint16, copy=False)
