"""OFDM modulation / demodulation and FIR filter helpers.

Promoted from ``benchmarks/dsp/_helpers.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fft import fft_spf32, ifft_spf32
from spf.scalar import spf32

__all__ = [
    "demodulate",
    "demodulate_fp",
    "demodulate_spf",
    "fir_filter_fp",
    "fir_filter_spf",
    "modulate",
    "modulate_fp",
    "modulate_spf",
]

Word = npt.NDArray[np.uint32]
Complex = npt.NDArray[np.complex128]


def _profile_name(profile: object | None) -> str | None:
    if profile is None:
        return None
    raw = getattr(profile, "value", profile)
    return str(raw).strip().lower().replace("-", "_")


def _is_v02_profile(profile: object | None) -> bool:
    return _profile_name(profile) in {"v02", "spf_v02", "strict_v02", "fb3_hybrid"}


def _validate_fp_request(profile: object | None, precision: object) -> None:
    """Reject SPF-only controls instead of silently ignoring them on FP data."""

    if profile is not None:
        raise ValueError(
            f"profile={profile!r} is only supported for packed uint32 SPF inputs; "
            "pass spf=True with SPF words to select the v0.2 profile"
        )
    if precision != "balanced":
        raise ValueError(
            f"precision={precision!r} is only supported for packed uint32 SPF inputs"
        )


def modulate_spf(symbols: Word, *, profile: str | None = None, precision: str = "balanced") -> Word:
    """OFDM modulator with selectable legacy or v0.2 SPF-32 profile."""
    return ifft_spf32(symbols, profile=profile, precision=precision)


def demodulate_spf(signal: Word, *, profile: str | None = None, precision: str = "balanced") -> Word:
    """OFDM demodulator with selectable legacy or v0.2 SPF-32 profile."""
    return fft_spf32(signal, profile=profile, precision=precision)


def modulate_fp(symbols: Complex) -> Complex:
    """FP OFDM modulator: ``np.fft.ifft`` on subcarrier vector."""
    return np.fft.ifft(symbols)


def demodulate_fp(signal: Complex) -> Complex:
    """FP OFDM demodulator: ``np.fft.fft`` on time-domain vector."""
    return np.fft.fft(signal)


def modulate(
    symbols: Word | Complex,
    *,
    spf: bool | None = None,
    profile: str | None = None,
    precision: str = "balanced",
) -> Word | Complex:
    """Unified OFDM modulator; SPF calls may select the v0.2 profile."""
    use_spf = (np.asarray(symbols).dtype == np.uint32) if spf is None else spf
    if use_spf:
        return modulate_spf(
            np.asarray(symbols, dtype=np.uint32), profile=profile, precision=precision
        )
    _validate_fp_request(profile, precision)
    return modulate_fp(np.asarray(symbols, dtype=np.complex128))


def demodulate(
    signal: Word | Complex,
    *,
    spf: bool | None = None,
    profile: str | None = None,
    precision: str = "balanced",
) -> Word | Complex:
    """Unified OFDM demodulator; SPF calls may select the v0.2 profile."""
    use_spf = (np.asarray(signal).dtype == np.uint32) if spf is None else spf
    if use_spf:
        return demodulate_spf(
            np.asarray(signal, dtype=np.uint32), profile=profile, precision=precision
        )
    _validate_fp_request(profile, precision)
    return demodulate_fp(np.asarray(signal, dtype=np.complex128))


def fir_filter_spf(x: Word, h: Word, *, profile: str | None = None, precision: str = "balanced") -> Word:
    """Circular convolution in SPF-32 via a selectable FFT profile."""
    X = fft_spf32(x, profile=profile, precision=precision)
    H = fft_spf32(h, profile=profile, precision=precision)
    if _is_v02_profile(profile):
        from spf.tensor.v02 import NaNProfile, spf_mul

        Y = np.asarray(
            spf_mul(X, H, profile=NaNProfile.STRICT_V02), dtype=np.uint32
        )
    else:
        Y = spf32.mul(X, H)
    return ifft_spf32(Y, profile=profile, precision=precision)


def fir_filter_fp(x: Complex, h: Complex) -> Complex:
    """FP circular convolution via ``np.fft``."""
    return np.fft.ifft(np.fft.fft(x) * np.fft.fft(h))
