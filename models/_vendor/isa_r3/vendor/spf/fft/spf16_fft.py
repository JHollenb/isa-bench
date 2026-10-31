"""Radix-2 DIT FFT driver for SPF-16.

Spec: docs/spf-formats-and-test-plans.md §12 T1.4 applied to SPF-16 (§3).
Roadmap: Task 1.2.

Thin binding of :mod:`spf.fft._driver` to :mod:`spf.scalar.spf16`. Per-stage
error budget: ``ε_stage ≈ 2·(2⁻³ + 2⁻⁷) ≈ 0.266`` (format ULP derived, 16×
looser than SPF-32).

**Dynamic range cap.** SPF-16's finite range is r ∈ [2⁻⁷·⁹³⁷⁵, 2⁷·⁹³⁷⁵] ≈
[0.004, 245]. Random complex-normal inputs at N = 1024 produce spectrum
magnitudes up to ~1024, exceeding range. The driver caps N at 256 (spec-sanctioned
twiddle-exact range) and expects callers to normalize inputs by ``1/√N`` so
the peak spectrum amplitude stays inside finite range.

Supported sizes: powers of two in [2, 256]. Twiddle exact for all N | 256.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fft import _driver
from spf.scalar import spf16
from spf.scalar._constants import SPF16

__all__ = ["fft", "ifft"]

Word = npt.NDArray[np.uint16]

_MIN_N: int = 2
_MAX_N: int = 256


def fft(x: Word) -> Word:
    """Forward SPF-16 N-point DFT via radix-2 decimation-in-time.

    Parameters
    ----------
    x : ndarray[uint16]
        Length-N input of SPF-16 words. N must be a power of 2 in [2, 256].
        Caller should normalize amplitudes so the spectrum stays below 2⁷·⁹⁴.

    Returns
    -------
    ndarray[uint16]
        Length-N DFT in natural bin order. Input is not mutated.
    """
    out = _driver.forward_fft(
        x, scalar=spf16, fmt=SPF16, min_n=_MIN_N, max_n=_MAX_N
    )
    return out.astype(np.uint16, copy=False)


def ifft(x: Word) -> Word:
    """Inverse SPF-16 N-point DFT via ``(1/N) · conj(fft(conj(X)))``."""
    out = _driver.inverse_fft(
        x, scalar=spf16, fmt=SPF16, min_n=_MIN_N, max_n=_MAX_N
    )
    return out.astype(np.uint16, copy=False)
