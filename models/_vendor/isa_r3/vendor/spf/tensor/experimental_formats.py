"""Opt-in research layouts over the existing Format and SPFBlockV3 carriers.

Canonical SPF32 is unchanged. Arithmetic delegates to existing scalar operators;
these are code-space NumPy kernels, not a new C adder or a production format.
"""

from __future__ import annotations

import numpy as np

from spf.scalar._constants import _make_format


def _layout(name: str, magnitude: int, phase: int, integer: int = 7):
    return _make_format(
        name=name,
        total_bits=32,
        l_bits=magnitude,
        theta_bits=phase,
        l_int_bits=integer,
        l_frac_bits=magnitude - integer - 1,
        word_dtype=np.uint32,
        l_signed_dtype=np.int32 if magnitude > 16 else np.int16,
        l_unsigned_dtype=np.uint32 if magnitude > 16 else np.uint16,
        theta_dtype=np.uint32 if phase > 16 else np.uint16,
    )


M18_P14 = _layout("SPF-32-M18-P14", 18, 14)
M19_P13 = _layout("SPF-32-M19-P13", 19, 13)
M20_P12 = _layout("SPF-32-M20-P12", 20, 12)
Q3_12_P16 = _layout("SPF-32-Q3.12-P16", 16, 16, 3)
# Phase-heavy flat word: 16x finer phase, about +/-8 log2 magnitude units.
# This descriptor has no serialized SPFBlockV3 representation ID or C ABI.
Q3_8_P20 = _layout("SPF-32-Q3.8-P20", 12, 20, 3)


def convert_words(words, source, target):
    """Integer RNE conversion at an explicit existing-carrier boundary."""
    from .core import pack_words, unpack_L, unpack_theta

    L, T = unpack_L(words, source).astype(np.int64), unpack_theta(words, source).astype(np.int64)
    zero = source.l_code_zero == L
    nan = (source.l_code_inf == L) & (source.theta_nan_payload == T)

    def rescale(code, shift):
        if shift >= 0:
            return code << shift
        divisor = 1 << -shift
        quotient, remainder = np.divmod(code, divisor)
        return quotient + ((remainder * 2 > divisor) | ((remainder * 2 == divisor) & (quotient & 1)))

    L = np.where(source.l_code_inf == L, source.l_code_max_finite, L)
    L = rescale(L, target.l_frac_bits - source.l_frac_bits)
    T = rescale(T, target.theta_bits - source.theta_bits) & target.word_mask_theta
    L = np.clip(L, target.l_code_min_finite, target.l_code_max_finite)
    L = np.where(zero, target.l_code_zero, np.where(nan, target.l_code_inf, L))
    T = np.where(zero, 0, np.where(nan, target.theta_nan_payload, T))
    return pack_words(L, T, target)
