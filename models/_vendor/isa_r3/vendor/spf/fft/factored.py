"""Opt-in factored tan-tanh radix-2 FFT profile for SPF-32.

This is the ``profile="v02_factored"`` transform: the same v0.2 radix-2 DIT
schedule as :func:`spf.tensor.v02_planes.spf_fft_radix2` (bit reversal, then
stages of size 2, 4, ...), but with the FB3-Hybrid butterfly swapped for the
**factored tan-tanh pair adder** (:func:`spf.tensor.factored_add`).  Nothing
here is reimplemented: the bit-reversal, the stage index/twiddle table
(:func:`spf.tensor.v02_planes._stage_indices`), the per-element phase shift
(:func:`spf.tensor.v02_planes.words_phase_shift`) and the conjugate
(:func:`spf.tensor.v02_planes.words_conj`) are reused unchanged; only the pair
adder differs.  The C counterpart is ``spf32_factored_fft_rows`` in
``native/src/scalar/spf32/spf32_factored_add.c`` (built into
``libspf_ocean_packed``; bound in :mod:`spf.native.factored_fft`), and the
plane path here is word-identical to it (see
``tests/native_bindings/test_factored_add_native.py`` and
``tests/fft/test_factored_fft.py``).

Status: **CANDIDATE, not conforming** (Algorithm Reference v0.2 §0 rule 4).
FB3-Hybrid (``profile="v02"``) remains the conforming default; this profile is
selected only when a caller asks for ``"v02_factored"`` and nothing in the
library selects it on its own.

Error budget (per Algorithm Reference v0.2 §11 and the factored adder's
declared ``<= 1 tick`` L/Theta budget):

    The exact phase classes (equal magnitude, same/opposite phase, quadrature,
    zero, NaN) delegate to the FB3-Hybrid adder and stay **bit-identical**, so a
    butterfly with no non-trivial magnitude/phase gap adds no error beyond FB3.
    The generic case carries at most one tick of L and one tick of Theta per
    output word versus the exact local-frame reference.  Over the
    ``log2(N)`` DIT stages plus the input encode, the per-output relative error
    is bounded by

        (log2(N) + 1) * hypot(2**(1/512) - 1, pi / 65536)
        = (log2(N) + 1) * hypot(1.35472e-3, 4.79369e-5)

    i.e. (log2(N) + 1) half-tick roundings, each combining the SPF-32
    log-magnitude quantization (``2**(1/(2*L_SCALE)) - 1``) and the phase
    quantization (``pi / PHASE_MOD``) in quadrature.  MEASURED rel-L2 stays well
    inside this because only ~0.01-0.05 % of generic words move by a full tick
    (see ``tests/fft/test_factored_fft.py`` and the factored adder blog post
    2026-09-30-135722).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from spf.tensor import core as _core
from spf.tensor import v02
from spf.tensor.factored_add import spf_pair_addsub_factored, spf_pair_addsub_factored_planes
from spf.tensor.v02 import SPF32_L_SCALE, SPF32_L_ZERO, SPF32_WORD_NAN, spf_phase_shift
from spf.tensor.v02_planes import _clamp_finite, _stage_indices, words_conj, words_phase_shift

__all__ = [
    "FACTORED_FFT_HALF_TICK",
    "factored_fft_error_budget",
    "spf_fft_radix2_factored",
    "spf_fft_radix2_factored_reference",
    "spf_ifft_radix2_factored",
]

Word = npt.NDArray[np.uint32]

# hypot(2**(1/512) - 1, pi / 65536) — one half-tick rounding of an SPF-32 word.
FACTORED_FFT_HALF_TICK: float = math.hypot(2.0 ** (1.0 / 512.0) - 1.0, math.pi / 65536.0)


def factored_fft_error_budget(n: int) -> float:
    """Design per-output relative-error bound for an N-point factored FFT.

    ``(log2(N) + 1) * hypot(1.35472e-3, pi/65536)`` — see the module docstring.
    """

    return (math.log2(n) + 1.0) * FACTORED_FFT_HALF_TICK


def spf_fft_radix2_factored(
    values: npt.ArrayLike,
    precision: Any = "balanced",
) -> np.ndarray:
    """Forward SPF-32 radix-2 FFT using the factored tan-tanh butterfly.

    The last axis (power-of-two length) is transformed and leading batch axes
    are retained.  Word-identical to the scalar reference
    :func:`spf_fft_radix2_factored_reference` and to the C
    ``spf32_factored_fft_rows``.  This is the ``profile="v02_factored"`` path of
    :func:`spf.fft.spf32_fft.fft`.  Error budget: see the module docstring /
    :func:`FACTORED_FFT_ERROR_BUDGET`.
    """

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_fft_radix2_factored requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_fft_radix2_factored requires power-of-two N, got N={n}")
    if n > v02.SPF32_INTERNAL_PHASE_MOD:
        raise ValueError("spf_fft_radix2_factored requires N <= the internal phase accumulator modulus")
    rows = array.reshape(-1, n)
    working = rows[:, v02._bit_reverse_indices(n)].copy()
    block = 2
    while block <= n:
        first, second, twiddle = _stage_indices(n, block)
        a = working[:, first]
        b = words_phase_shift(working[:, second], twiddle[None, :])
        y0, y1 = spf_pair_addsub_factored_planes(a, b, precision)
        working[:, first] = y0
        working[:, second] = y1
        block *= 2
    return working.reshape(array.shape)


def spf_ifft_radix2_factored(
    values: npt.ArrayLike,
    precision: Any = "balanced",
) -> np.ndarray:
    """Inverse of :func:`spf_fft_radix2_factored`.

    ``ifft(X) = conj(fft(conj(X))) / N``; the ``1/N`` step is an exact log
    shift (``L -= log2(N) * L_SCALE``), preserving the zero/NaN sentinels.  Same
    conjugate + shift logic as :func:`spf.tensor.v02_planes.spf_ifft_radix2`,
    only the forward transform differs.
    """

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_ifft_radix2_factored requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_ifft_radix2_factored requires power-of-two N, got N={n}")
    transformed = words_conj(spf_fft_radix2_factored(words_conj(array), precision))
    log_shift = int(math.log2(n)) * SPF32_L_SCALE
    L = np.asarray(_core.unpack_L(transformed), dtype=np.int64)
    T = np.asarray(_core.unpack_theta(transformed), dtype=np.int64)
    keep = (L == SPF32_L_ZERO) | (transformed == np.uint32(SPF32_WORD_NAN))
    scaled = np.asarray(
        _core.pack_words(_clamp_finite(L - log_shift).astype(np.int16), T.astype(np.uint16))
    )
    return np.asarray(np.where(keep, transformed, scaled), dtype=np.uint32)


def _twiddle_code(offset: int, block: int) -> int:
    """Integer RNE twiddle (matches the C ``twiddle_code``); float-free oracle."""

    step = -((1 << 32) // block)
    wide = offset * step
    quotient = wide >> 16
    remainder = wide - (quotient << 16)
    half = 1 << 15
    if remainder > half or (remainder == half and (quotient & 1)):
        quotient += 1
    return quotient & 0xFFFF


def spf_fft_radix2_factored_reference(
    values: npt.ArrayLike,
    precision: Any = "balanced",
) -> np.ndarray:
    """Scalar per-word forward factored FFT — the bit-identity oracle.

    Mirrors the C ``spf32_factored_fft_rows`` exactly (integer twiddle,
    :func:`spf.tensor.factored_add.spf_pair_addsub_factored` per butterfly).
    """

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_fft_radix2_factored requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_fft_radix2_factored requires power-of-two N, got N={n}")
    rows = array.reshape(-1, n)
    output = np.empty_like(rows)
    rev = v02._bit_reverse_indices(n)
    for index, row in enumerate(rows):
        work = [int(w) for w in np.asarray(row, dtype=np.uint32)[rev]]
        block = 2
        while block <= n:
            half = block // 2
            for base in range(0, n, block):
                for offset in range(half):
                    twiddle = _twiddle_code(offset, block)
                    first = base + offset
                    second = first + half
                    rotated = int(spf_phase_shift(work[second], twiddle))
                    y0, y1 = spf_pair_addsub_factored(work[first], rotated, precision)
                    work[first] = y0
                    work[second] = y1
            block *= 2
        output[index] = np.asarray(work, dtype=np.uint32)
    return output.reshape(array.shape)
