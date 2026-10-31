"""FNT-32 forward and inverse Number Theoretic Transform (radix-2 DIT).

Spec: docs/spf-formats-and-test-plans.md §10.2 and §13.4.

``ntt(x)`` is the forward NTT; ``intt(X)`` is its inverse. Both operate on
``ndarray[uint64]`` of length ``N ∈ VALID_N = {2, 4, 8, 16, 32, 64}`` —
the divisors of 64, which is the order of 2 in ``Z_F``.

The algorithm is standard decimation-in-time: bit-reverse the input, then
``log2(N)`` stages of butterflies with twiddle power ``k_j = j · 64/m`` at
stage of size ``m`` and column ``j``. Inverse uses ``k_j = -j · 64/m (mod 64)``
plus a final scaling by ``N^(-1) = 2^(64 - log2(N)) mod F`` (exact because
every valid ``N`` is a power of 2 within the order-64 cycle).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fnt._constants import TWIDDLE_ORDER, VALID_N
from spf.fnt.fnt32 import add, sub, twiddle_mul

__all__ = ["intt", "ntt"]

U64 = npt.NDArray[np.uint64]

_BIT_REV_CACHE: dict[int, npt.NDArray[np.intp]] = {}


def _bit_reversal(n: int) -> npt.NDArray[np.intp]:
    cached = _BIT_REV_CACHE.get(n)
    if cached is not None:
        return cached
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.intp)
    rev = np.zeros(n, dtype=np.intp)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    _BIT_REV_CACHE[n] = rev
    return rev


def _validate(x: np.ndarray) -> int:
    if x.ndim != 1:
        raise ValueError(f"FNT-32 NTT input must be 1-D; got shape {x.shape}")
    if x.dtype != np.uint64:
        raise ValueError(f"FNT-32 NTT input must be ndarray[uint64]; got dtype {x.dtype}")
    n = int(x.shape[0])
    if n not in VALID_N:
        raise ValueError(f"FNT-32 NTT length N must be in {VALID_N}; got N={n}")
    return n


def _dit(x: U64, sign: int) -> U64:
    """Run radix-2 DIT with signed twiddle powers; sign ∈ {+1, -1}."""
    n = x.shape[0]
    buf: U64 = x[_bit_reversal(n)].astype(np.uint64, copy=True)
    log2n = n.bit_length() - 1
    for s in range(log2n):
        m = 2 << s
        half = m >> 1
        num_blocks = n // m
        stride = TWIDDLE_ORDER // m  # == 64 // m
        view = buf.reshape(num_blocks, m)
        for j in range(half):
            k = (sign * j * stride) & 63
            wb = twiddle_mul(view[:, half + j], k)
            new_top = add(view[:, j], wb)
            new_bot = sub(view[:, j], wb)
            view[:, j] = new_top
            view[:, half + j] = new_bot
    return buf


def ntt(x: np.ndarray) -> U64:
    """Forward NTT over ``Z_F``. ``x`` is ``ndarray[uint64]`` with ``|x| ∈ VALID_N``."""
    _validate(x)
    return _dit(x.astype(np.uint64, copy=False), sign=+1)


def intt(x: np.ndarray) -> U64:
    """Inverse NTT over ``Z_F``.

    Uses conjugate twiddles then multiplies by ``N^(-1) = 2^(64 - log2(N))``.
    """
    n = _validate(x)
    transformed = _dit(x.astype(np.uint64, copy=False), sign=-1)
    log2n = n.bit_length() - 1
    return twiddle_mul(transformed, TWIDDLE_ORDER - log2n)
