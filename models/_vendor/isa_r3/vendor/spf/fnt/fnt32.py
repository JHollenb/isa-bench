"""FNT-32 scalar ring operations over ``Z_F``, ``F = 2**32 + 1``.

Spec: docs/spf-formats-and-test-plans.md §10.

All operations are vectorized over ``ndarray[uint64]``. Storage is uint64
because ``F`` itself is outside the uint32 range and because the twiddle
bit-rotation trick needs a 64-bit intermediate. The public contract is that
inputs are in ``[0, F)`` and outputs stay in ``[0, F)``.

``twiddle_mul(a, k)`` diverges from spec §10.3 in two places:

1. The spec's cyclic-shift formulation ``(a << k) | (a >> (32 - k))`` is wrong
   when the top ``k`` bits of ``a`` are nonzero: the rotation re-injects those
   bits into the low word after they've been "counted" in the high word, so
   the subsequent ``lo - hi`` reduction double-cancels them and gives zero.
   Example: ``a = 2^31, k = 1`` should yield ``2·2^31 mod F = F - 1``, but the
   spec's code returns 0. The correct reduction is a plain left shift
   ``s = a << k`` in uint64, then ``lo = s & 0xFFFFFFFF``, ``hi = s >> 32``,
   ``result = lo - hi mod F``. Plain shift is simpler *and* correct.
2. The spec reduces ``k`` modulo 32, but element 2 has order **64** in
   ``Z_F`` (because ``2^32 ≡ -1``, not 1). The spec's own T4.1 test iterates
   ``k`` up to 32 and expects ``twiddle_mul(a, 32) == -a mod F``, which the
   mod-32 code cannot produce. We reduce modulo 64 and fold ``k ≥ 32`` via
   ``2^k = -2^(k-32) mod F``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fnt._constants import _UINT64_32, _UINT64_F, _UINT64_LOW32_MASK, F

__all__ = ["F", "add", "butterfly", "neg", "sub", "twiddle_mul"]

U64 = npt.NDArray[np.uint64]


def _as_u64(x: np.ndarray) -> U64:
    if x.dtype != np.uint64:
        raise ValueError(f"FNT-32 inputs must be ndarray[uint64]; got dtype {x.dtype}")
    return x


def add(a: np.ndarray, b: np.ndarray) -> U64:
    """Return ``(a + b) mod F``, vectorized."""
    a_u = _as_u64(a)
    b_u = _as_u64(b)
    s = a_u + b_u
    # Branchless modular reduction: subtract F where the sum is >= F.
    # Sum never exceeds 2*(F-1) = 2^33, well within uint64 range.
    overflow = (s >= _UINT64_F).astype(np.uint64) * _UINT64_F
    return (s - overflow).astype(np.uint64)


def sub(a: np.ndarray, b: np.ndarray) -> U64:
    """Return ``(a - b) mod F``, vectorized."""
    a_u = _as_u64(a)
    b_u = _as_u64(b)
    # a + (F - b) mod F, avoiding unsigned underflow
    underflow = (a_u < b_u).astype(np.uint64) * _UINT64_F
    return (a_u + underflow - b_u).astype(np.uint64)


def neg(a: np.ndarray) -> U64:
    """Return ``(-a) mod F``. ``neg(0) == 0``; otherwise ``F - a``."""
    a_u = _as_u64(a)
    nonzero = (a_u != 0).astype(np.uint64)
    out: U64 = ((_UINT64_F - a_u) * nonzero).astype(np.uint64)
    return out


def _twiddle_mul_sub32(a: U64, k: int) -> U64:
    """Multiply by ``2^k mod F`` for ``0 <= k <= 31``.

    For ``a`` in ``[0, F)`` and ``k <= 31``, ``a << k`` fits in uint64
    (max ``(F-1) * 2^31 = 2^63``). Then ``a * 2^k mod F`` reduces via the
    identity ``2^32 ≡ -1``: ``shifted = hi * 2^32 + lo ≡ lo - hi (mod F)``.
    """
    shifted = a << np.uint64(k)
    lo = shifted & _UINT64_LOW32_MASK
    hi = shifted >> _UINT64_32
    return sub(lo, hi)


def twiddle_mul(a: np.ndarray, k: int) -> U64:
    """Return ``(a * 2^k) mod F``, vectorized over ``a``.

    ``k`` is reduced modulo 64. For ``32 <= k < 64`` we use the identity
    ``2^k = -2^(k-32) mod F``.
    """
    a_u = _as_u64(a)
    k = int(k) & 63
    if k == 0:
        return a_u.copy()
    if k >= 32:
        return neg(_twiddle_mul_sub32(a_u, k - 32))
    return _twiddle_mul_sub32(a_u, k)


def butterfly(a: np.ndarray, b: np.ndarray, k: int) -> tuple[U64, U64]:
    """Compute ``(a + w·b, a - w·b)`` where ``w = 2^k mod F``."""
    wb = twiddle_mul(b, k)
    return add(a, wb), sub(a, wb)
