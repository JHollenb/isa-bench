"""SPF-32 ↔ SPF-16 array conversion (cross-precision, integer-only).

Boundary path used by the mixed-precision DSP pipelines: SPF-32 is the
outer storage / equalization precision (Q7.8 L, 16-bit Θ) and SPF-16
(Q3.4 L, 8-bit Θ) is the narrower inner precision used for compute-bound
stages such as FFT64 where every word doubles the off-chip bandwidth.

Both directions are pure integer fixed-point operations on the packed
words:

* ``spf32_to_spf16``: rebracket L from Q7.8 to Q3.4 with symmetric
  round-half-away-from-zero (``+8`` bias on the magnitude before the
  ``>>4`` shift, applied to both signs), saturate at the SPF-16 finite
  range, drop the low byte of Θ16 with ``+128`` round-half-up plus
  natural ``mod 256`` wrap, and preserve the three sentinel classes
  (zero / directional inf / canonical NaN).
* ``spf16_to_spf32``: sign-extend the L code from int8 to int16 and
  scale by 16 (Q3.4 → Q7.8); shift Θ8 left by 8 bits; preserve sentinels.

Rounding policy:

* **L (signed Q7.8 → Q3.4)** uses **symmetric round-half-away-from-zero**
  — the ``+8`` magnitude bias is applied to both signs so SPF-32 codes
  ±8, ±24, ±40, … all round one bin away from zero (toward larger
  magnitude). This keeps the cast symmetric across the L = 0 axis and
  avoids the toward-zero bias an asymmetric rule would introduce on
  negative L codes.

* **Θ (unsigned 16-bit → unsigned 8-bit)** uses **circular round-half-up**
  — ``(theta + 128) >> 8 mod 256``. Θ is unsigned and wraps modulo
  ``2π``, so "away from zero" has no geometric meaning; round-half-up
  is the standard unsigned-integer rounding rule, and the natural
  ``mod 256`` wrap covers the Θ16 = 0xFF80 boundary that would
  otherwise carry into a 9th bit.

Both rules deliberately differ from the spf32 / spf16 *encoders*,
which use ``np.rint`` (banker rounding) on float64 magnitudes. This
function operates on already-quantized integer codes; the simpler
integer rounding rules above keep the cast a pure shift-and-mask
operation suitable for hardware. Residual DC bias on the L axis is
bounded by 1 L ULP and is exercised in the test
``test_l_rounding_no_dc_bias_on_uniform_grid``.

Conversion is lossy in one direction (downcast) and lossless in the
other (upcast then re-downcast hits the original SPF-16 word). No FP
arithmetic, no Sb() involvement.

Sentinel rules (critical, hardware will copy this verbatim):

* SPF-32 zero word ``0x80000000`` ↔ SPF-16 zero word ``0x8000``.
* SPF-32 NaN ``0x7FFFFFFF`` ↔ SPF-16 NaN ``0x7FFF``.
* SPF-32 directional inf (L = 0x7FFF, Θ ≠ 0xFFFF) → SPF-16 directional
  inf (L = 0x7F, Θ = round(Θ16/256) mod 256). Edge case: if rounding
  lands on Θ8 = 0xFF the result would alias the SPF-16 NaN code, so we
  snap that single bin to 0xFE to preserve the inf classification.
* SPF-16 directional inf (L = 0x7F, Θ < 0xFF) → SPF-32 directional inf
  (L = 0x7FFF, Θ16 = Θ8 << 8).

Spec: docs/spf-formats-and-test-plans.md §2 (SPF-32), §3 (SPF-16),
§13 (sentinels). This module sits alongside the FP↔SPF converters in
:mod:`spf.convert`; experimental SPF-16 hardware (see PR #197 baked
``spf16_core_pl`` integer overlay) will copy the same bit-level rules.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar._constants import SPF16, SPF32

__all__ = ["spf16_to_spf32", "spf32_to_spf16"]


_SPF32_L_CODE_INF: int = SPF32.l_code_inf  # 32767
_SPF32_L_CODE_ZERO: int = SPF32.l_code_zero  # -32768
_SPF32_THETA_NAN: int = SPF32.theta_nan_payload  # 0xFFFF
_SPF32_THETA_MASK: int = SPF32.word_mask_theta  # 0xFFFF

_SPF16_L_CODE_INF: int = SPF16.l_code_inf  # 127
_SPF16_L_CODE_ZERO: int = SPF16.l_code_zero  # -128
_SPF16_L_MAX_FINITE: int = SPF16.l_code_max_finite  # 126
_SPF16_L_MIN_FINITE: int = SPF16.l_code_min_finite  # -127
_SPF16_THETA_NAN: int = SPF16.theta_nan_payload  # 0xFF
_SPF16_WORD_ZERO: int = SPF16.word_zero  # 0x8000
_SPF16_WORD_NAN: int = SPF16.word_nan  # 0x7FFF
_SPF32_WORD_ZERO: int = SPF32.word_zero  # 0x80000000
_SPF32_WORD_NAN: int = SPF32.word_nan  # 0x7FFFFFFF


def spf32_to_spf16(w: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Downcast packed SPF-32 words to packed SPF-16 words.

    Parameters
    ----------
    w : array-like of uint32
        SPF-32 words (Q7.8 L | 16-bit Θ).

    Returns
    -------
    ndarray[uint16]
        SPF-16 words of identical shape (Q3.4 L | 8-bit Θ).

    Notes
    -----
    Lossy: L drops 4 fraction bits and 4 integer-range bits (saturates
    at ±7.9375 in log₂), Θ drops the low byte. Sentinels are routed
    intact; non-sentinel values that would alias the NaN code after
    rounding are snapped one bin away.
    """
    w_arr = np.asarray(w, dtype=np.uint32)

    l_signed = (w_arr >> 16).astype(np.uint16).view(np.int16).astype(np.int32)
    theta_u = (w_arr & _SPF32_THETA_MASK).astype(np.int32)

    is_zero = l_signed == _SPF32_L_CODE_ZERO
    is_inf_or_nan = l_signed == _SPF32_L_CODE_INF
    is_nan = is_inf_or_nan & (theta_u == _SPF32_THETA_NAN)
    is_inf = is_inf_or_nan & ~is_nan

    # Finite path: symmetric round-half-away-from-zero on L.
    # L Q7.8 → Q3.4: divide by 16 with the +8 magnitude bias applied to
    # BOTH signs, so ±8, ±24, ±40, … all round one step away from zero
    # (toward larger magnitude). Saturate to SPF-16 finite range.
    l_round = np.where(
        l_signed >= 0,
        (l_signed + 8) >> 4,
        -(((-l_signed) + 8) >> 4),
    ).astype(np.int32)
    l_q34 = np.clip(l_round, _SPF16_L_MIN_FINITE, _SPF16_L_MAX_FINITE).astype(
        np.int32
    )

    # Θ16 → Θ8: unsigned circular round-half-up; the +128 bias plus
    # natural ``& 0xFF`` wrap maps a half-step at Θ16 = 0xFF80 (just
    # below 2π) to Θ8 = 0 (= 0 rad), which is the right phase wrap.
    theta_8 = ((theta_u + 128) >> 8) & 0xFF

    # Pack finite words: signed int8 L into low byte (mod 256), then shift.
    l_byte = (l_q34 & 0xFF).astype(np.uint16)
    finite_word = ((l_byte << 8) | theta_8.astype(np.uint16)).astype(np.uint16)

    # Directional inf: L = 0x7F, Θ8 = round(Θ16/256). Avoid colliding
    # with the canonical NaN code (L=0x7F, Θ=0xFF) by snapping Θ8=0xFF
    # → 0xFE for inf-class words only. Direction is preserved at SPF-16
    # phase resolution (~1.4° per Θ8 ULP).
    inf_theta_8 = np.where(theta_8 == 0xFF, np.int32(0xFE), theta_8)
    inf_word = (
        (np.uint16(_SPF16_L_CODE_INF & 0xFF) << 8) | inf_theta_8.astype(np.uint16)
    ).astype(np.uint16)

    out = np.where(is_zero, np.uint16(_SPF16_WORD_ZERO), finite_word)
    out = np.where(is_inf, inf_word, out)
    out = np.where(is_nan, np.uint16(_SPF16_WORD_NAN), out)
    return out.astype(np.uint16, copy=False)


def spf16_to_spf32(w: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Upcast packed SPF-16 words to packed SPF-32 words.

    Parameters
    ----------
    w : array-like of uint16
        SPF-16 words (Q3.4 L | 8-bit Θ).

    Returns
    -------
    ndarray[uint32]
        SPF-32 words of identical shape (Q7.8 L | 16-bit Θ).

    Notes
    -----
    Lossless: ``spf32_to_spf16(spf16_to_spf32(x)) == x`` byte-for-byte.
    The decoded complex value rounds to the SPF-32 grid but holds the
    full SPF-16 precision (which is coarser).
    """
    w_arr = np.asarray(w, dtype=np.uint16)

    l_signed = (w_arr >> 8).astype(np.uint8).view(np.int8).astype(np.int32)
    theta_8 = (w_arr & 0xFF).astype(np.int32)

    is_zero = l_signed == _SPF16_L_CODE_ZERO
    is_inf_or_nan = l_signed == _SPF16_L_CODE_INF
    is_nan = is_inf_or_nan & (theta_8 == _SPF16_THETA_NAN)
    is_inf = is_inf_or_nan & ~is_nan

    # Finite: scale L by 16 (Q3.4 → Q7.8), Θ8 by 256.
    l_q78 = (l_signed * 16).astype(np.int32)
    theta_16 = (theta_8 << 8).astype(np.int32)
    l_byte = (l_q78 & 0xFFFF).astype(np.uint32)
    finite_word = ((l_byte << 16) | theta_16.astype(np.uint32)).astype(np.uint32)

    # Directional inf: SPF-32 L = 0x7FFF, Θ16 = Θ8 << 8.
    inf_word = (
        (np.uint32(_SPF32_L_CODE_INF & 0xFFFF) << 16)
        | theta_16.astype(np.uint32)
    ).astype(np.uint32)

    out = np.where(is_zero, np.uint32(_SPF32_WORD_ZERO), finite_word)
    out = np.where(is_inf, inf_word, out)
    out = np.where(is_nan, np.uint32(_SPF32_WORD_NAN), out)
    return out.astype(np.uint32, copy=False)
