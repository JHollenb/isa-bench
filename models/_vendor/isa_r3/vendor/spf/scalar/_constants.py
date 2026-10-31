"""SPF bit-layout constants and sentinel codes.

Spec: docs/spf-formats-and-test-plans.md §2 (SPF-32), §3 (SPF-16), §4 (SPF-64),
§13.1–13.2 (sentinels).

The three SPF scalar formats share one structure: a log-polar complex number
stored in a word that concatenates a signed fixed-point log-magnitude L (upper
``l_bits``) with an unsigned phase Θ (lower ``theta_bits``). The :class:`Format`
dataclass makes the per-format dtypes, scales, and sentinel codes explicit so
the core ops in :mod:`spf.scalar._ops` can generalize over the whole family.

Q-format naming: "Q7.8" = 7 integer bits + 8 fraction bits + 1 sign bit = 16
bits total (ARM convention). SPF-16 is Q3.4 in 8 bits; SPF-64 is Q15.16 in 32
bits.

**SPF-16 spec typo.** Spec §3.2 prints ``L=0x40 (−8, sentinel)`` for the zero
sentinel, but 0x40 as int8 is +64 (Q3.4 = 4.0), not −8. The family convention
is "zero = most-negative L code"; the prose value (−8, smallest representable
L) matches the int8 code ``0x80`` (−128 → Q3.4 = −8). We adopt ``0x80`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Format:
    """SPF scalar-format descriptor.

    A single instance captures every bit-level property of a format: dtypes for
    the packed word and its L/Θ fields, fixed-point scales, and the sentinel
    codes for zero / inf / NaN. Operators in :mod:`spf.scalar._ops` take a
    ``Format`` as a parameter so the same Python code drives SPF-16/32/64.
    """

    name: str  # e.g., "SPF-32"
    total_bits: int  # 16, 32, 64
    l_bits: int  # width of the L field (incl. sign bit)
    theta_bits: int  # width of the Θ field
    l_int_bits: int  # L integer bits, excluding sign
    l_frac_bits: int  # L fraction bits
    l_scale: int  # 2**l_frac_bits
    theta_scale: int  # 2**theta_bits
    l_ulp: float  # 1 / l_scale
    theta_ulp: float  # 1 / theta_scale (fractional-circle units)

    # numpy dtypes used for array-level packing/unpacking.
    word_dtype: Any
    l_signed_dtype: Any
    l_unsigned_dtype: Any
    theta_dtype: Any

    # Sentinel codes on the L field (signed-integer view).
    l_code_zero: int  # most-negative value: zero sentinel
    l_code_inf: int  # most-positive value: inf/NaN sentinel
    l_code_max_finite: int  # l_code_inf - 1
    l_code_min_finite: int  # -(most-negative + 1), one above zero sentinel
    theta_nan_payload: int  # Θ = all-ones alongside l_code_inf → NaN

    # Packed-word sentinels (integer values; callers wrap with word_dtype).
    word_zero: int
    word_nan: int
    word_mask_theta: int  # (1 << theta_bits) - 1


def _make_format(
    *,
    name: str,
    total_bits: int,
    l_bits: int,
    theta_bits: int,
    l_int_bits: int,
    l_frac_bits: int,
    word_dtype: Any,
    l_signed_dtype: Any,
    l_unsigned_dtype: Any,
    theta_dtype: Any,
) -> Format:
    """Build a :class:`Format` from its primitive parameters.

    Derives the scale factors, sentinel codes, and packed-word sentinels by
    construction so every field is internally consistent and there is a single
    source of truth for the bit layout.
    """
    assert l_bits + theta_bits == total_bits
    assert l_int_bits + l_frac_bits + 1 == l_bits  # +1 for sign

    l_scale = 1 << l_frac_bits
    theta_scale = 1 << theta_bits
    l_half = 1 << (l_bits - 1)
    l_code_zero = -l_half  # e.g., int8 −128 = 0x80
    l_code_inf = l_half - 1  # e.g., int8 +127 = 0x7F
    l_code_max_finite = l_code_inf - 1
    l_code_min_finite = -(l_half - 1)
    theta_nan_payload = theta_scale - 1

    l_bits_mask = (1 << l_bits) - 1
    word_mask_theta = theta_scale - 1
    word_zero = ((l_code_zero & l_bits_mask) << theta_bits) | 0
    word_nan = ((l_code_inf & l_bits_mask) << theta_bits) | theta_nan_payload

    return Format(
        name=name,
        total_bits=total_bits,
        l_bits=l_bits,
        theta_bits=theta_bits,
        l_int_bits=l_int_bits,
        l_frac_bits=l_frac_bits,
        l_scale=l_scale,
        theta_scale=theta_scale,
        l_ulp=1.0 / l_scale,
        theta_ulp=1.0 / theta_scale,
        word_dtype=word_dtype,
        l_signed_dtype=l_signed_dtype,
        l_unsigned_dtype=l_unsigned_dtype,
        theta_dtype=theta_dtype,
        l_code_zero=l_code_zero,
        l_code_inf=l_code_inf,
        l_code_max_finite=l_code_max_finite,
        l_code_min_finite=l_code_min_finite,
        theta_nan_payload=theta_nan_payload,
        word_zero=word_zero,
        word_nan=word_nan,
        word_mask_theta=word_mask_theta,
    )


# ── SPF-16 (spec §3) ─────────────────────────────────────────────────────────
# Q3.4 L (8 bits), 8-bit Θ. Dynamic range r ∈ [2⁻⁷·⁹³⁷⁵, 2⁷·⁹³⁷⁵] ≈ [0.004, 245].
# Twiddle factor W_N^k exact for N | 256.
SPF16: Format = _make_format(
    name="SPF-16",
    total_bits=16,
    l_bits=8,
    theta_bits=8,
    l_int_bits=3,
    l_frac_bits=4,
    word_dtype=np.uint16,
    l_signed_dtype=np.int8,
    l_unsigned_dtype=np.uint8,
    theta_dtype=np.uint8,
)

# ── SPF-32 (spec §2) ─────────────────────────────────────────────────────────
# Q7.8 L (16 bits), 16-bit Θ. Dynamic range ≈ [2⁻¹²⁷·⁹⁹⁶, 2¹²⁷·⁹⁹⁶].
# Twiddle exact for N | 2¹⁶.
SPF32: Format = _make_format(
    name="SPF-32",
    total_bits=32,
    l_bits=16,
    theta_bits=16,
    l_int_bits=7,
    l_frac_bits=8,
    word_dtype=np.uint32,
    l_signed_dtype=np.int16,
    l_unsigned_dtype=np.uint16,
    theta_dtype=np.uint16,
)

# ── SPF-64 (spec §4) ─────────────────────────────────────────────────────────
# Q15.16 L (32 bits), 32-bit Θ. Phase resolution 2π / 2³² ≈ 1.46×10⁻⁹ rad,
# matching AD9914 DDS. Twiddle exact for N | 2³².
SPF64: Format = _make_format(
    name="SPF-64",
    total_bits=64,
    l_bits=32,
    theta_bits=32,
    l_int_bits=15,
    l_frac_bits=16,
    word_dtype=np.uint64,
    l_signed_dtype=np.int32,
    l_unsigned_dtype=np.uint32,
    theta_dtype=np.uint32,
)

# ── SPF-128 (spec §5) ────────────────────────────────────────────────────────
# Q31.32 L (64 bits), 64-bit Θ. Phase resolution 2π / 2⁶⁴ ≈ 3.4×10⁻¹⁹ rad;
# twiddle exact for N | 2⁶⁴. The 128-bit word has **no numpy dtype** (numpy tops
# out at uint64), so SPF-128 words are stored as arbitrary-precision Python ints
# in ``object`` arrays. The L/Θ *fields* still map to real dtypes (int64 / uint64)
# — only the packed word is wide — so the exact integer ops keep full precision by
# routing their accumulators through Python int. Pure-Python (no C); slow by
# design. See :mod:`spf.scalar._ops` (the ``total_bits > 64`` branches).
SPF128: Format = _make_format(
    name="SPF-128",
    total_bits=128,
    l_bits=64,
    theta_bits=64,
    l_int_bits=31,
    l_frac_bits=32,
    word_dtype=object,
    l_signed_dtype=np.int64,
    l_unsigned_dtype=np.uint64,
    theta_dtype=np.uint64,
)

__all__ = [
    "L_CODE_INF",
    "L_CODE_MAX_FINITE",
    "L_CODE_MIN_FINITE",
    "L_CODE_ZERO",
    "L_FRAC_BITS",
    "L_INT_BITS",
    "L_SCALE",
    "L_ULP",
    "SB_LUT_DL_MAX",
    "SB_LUT_RES_DL",
    "SB_LUT_RES_DTHETA",
    "SB_WEDGE_DL",
    "SB_WEDGE_DTHETA_FROM_PI",
    "SB_WEDGE_LUT_NU",
    "SB_WEDGE_LUT_NV",
    "SB_WEDGE_LUT_RHO_MIN",
    "SPF16",
    "SPF32",
    "SPF64",
    "SPF128",
    "THETA_3PI_HALF",
    "THETA_BITS",
    "THETA_NAN_PAYLOAD",
    "THETA_PI",
    "THETA_PI_HALF",
    "THETA_SCALE",
    "THETA_ULP",
    "THETA_ZERO",
    "WORD_NAN",
    "WORD_NEG_I",
    "WORD_NEG_ONE",
    "WORD_ONE",
    "WORD_POS_I",
    "WORD_POS_INF",
    "WORD_ZERO",
    "Format",
]

# ── Legacy SPF-32 module-level aliases ───────────────────────────────────────
# Retained so existing imports (`from spf.scalar._constants import L_SCALE`, …)
# keep working without touching the 13 modules that use them. New code should
# prefer ``SPF32`` / ``Format`` fields.

L_INT_BITS: int = SPF32.l_int_bits
L_FRAC_BITS: int = SPF32.l_frac_bits
L_SCALE: int = SPF32.l_scale
L_ULP: float = SPF32.l_ulp

L_CODE_ZERO: int = SPF32.l_code_zero
L_CODE_INF: int = SPF32.l_code_inf
L_CODE_MAX_FINITE: int = SPF32.l_code_max_finite
L_CODE_MIN_FINITE: int = SPF32.l_code_min_finite

THETA_BITS: int = SPF32.theta_bits
THETA_SCALE: int = SPF32.theta_scale
THETA_ULP: float = SPF32.theta_ulp

THETA_ZERO: int = 0x0000  # 0 rad
THETA_PI_HALF: int = 0x4000  # π/2
THETA_PI: int = 0x8000  # π
THETA_3PI_HALF: int = 0xC000  # 3π/2
THETA_NAN_PAYLOAD: int = SPF32.theta_nan_payload

# ── Full-word sentinels (SPF-32, L in upper 16 bits, Θ in lower) ────────────

WORD_ONE: np.uint32 = np.uint32(0x00000000)  # +1 + 0i
WORD_NEG_ONE: np.uint32 = np.uint32(0x00008000)  # −1 + 0i
WORD_POS_I: np.uint32 = np.uint32(0x00004000)  # 0 + 1i
WORD_NEG_I: np.uint32 = np.uint32(0x0000C000)  # 0 − 1i
WORD_ZERO: np.uint32 = np.uint32(SPF32.word_zero)  # z = 0
WORD_POS_INF: np.uint32 = np.uint32(0x7FFF0000)  # +∞ at Θ=0
WORD_NAN: np.uint32 = np.uint32(SPF32.word_nan)  # canonical NaN

# ── Sb() LUT defaults (format-agnostic; inputs are physical units) ──────────
#
# The LUT operates on (ΔL in log₂, ΔΘ in radians) — physical units — so the
# same table and cancellation-wedge serve all three formats. See _sb.py.

SB_LUT_RES_DL: int = 4096
SB_LUT_RES_DTHETA: int = 512
SB_LUT_DL_MAX: float = 32.0
SB_WEDGE_DL: float = 1.0
SB_WEDGE_DTHETA_FROM_PI: float = 0.5

# ── Wedge-specific LUT (Phase 1 Sb replacement) ─────────────────────────────
#
# Grid over (u = |ΔL| ∈ [0, SB_WEDGE_DL], v = π − |ΔΘ| ∈ [0, SB_WEDGE_DTHETA_FROM_PI]).
# Bilinear FP32 interpolation; the log-singularity at (u, v) = (0, 0) is routed
# to sb_exact via a polar-radius fallback when √((u·ln2)² + v²) < SB_WEDGE_LUT_RHO_MIN.
#
# The 192×96 × 2 table geometry is selected in docs/research/sb/sb-wedge-design.md:
#   max |δL error| ≈ 1.16e-4 (3% of L ULP), max |δΘ error| ≈ 8.3e-5 (86% of Θ ULP)
# over 5M random wedge samples + 513×257 regular grid validation, FP32 arithmetic.
SB_WEDGE_LUT_NU: int = 192
SB_WEDGE_LUT_NV: int = 96
SB_WEDGE_LUT_RHO_MIN: float = 0.20
