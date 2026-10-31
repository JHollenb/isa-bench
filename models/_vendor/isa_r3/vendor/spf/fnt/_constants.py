"""FNT-32 ring and NTT parameters (spec §10, §13.4).

The Fermat modulus ``F = 2**32 + 1 = 4 294 967 297`` is the fifth Fermat number
``F_5``. It is *composite* (641 × 6 700 417), so ``Z_F`` is a ring, not a
field. For the NTT we only rely on the identity ``2^32 ≡ -1 (mod F)``, which
makes 2 a primitive 2·32 = 64-th root of unity — enough for N ∈ {2, 4, 8, 16,
32, 64} transforms.
"""

from __future__ import annotations

from typing import Final

import numpy as np

# Fermat modulus F_5 = 2**32 + 1.
F: Final[int] = (1 << 32) + 1

# Element 2 has order 64 in Z_F: 2**32 ≡ -1 ≠ 1 (mod F), 2**64 ≡ 1 (mod F).
# This is the upper bound on N for a single-level FNT-32 transform.
TWIDDLE_ORDER: Final[int] = 64

# N for which ω_N = 2**(TWIDDLE_ORDER // N) mod F is a primitive N-th root of
# unity in Z_F. Must divide TWIDDLE_ORDER.
VALID_N: Final[tuple[int, ...]] = (2, 4, 8, 16, 32, 64)

# np.uint64 scalar handles (shifts and bitwise ops on ndarrays want uint64 args).
_UINT64_32: Final[np.uint64] = np.uint64(32)
_UINT64_F: Final[np.uint64] = np.uint64(F)
_UINT64_LOW32_MASK: Final[np.uint64] = np.uint64(0xFFFFFFFF)
