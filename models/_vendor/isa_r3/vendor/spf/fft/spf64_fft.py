"""Radix-2 DIT FFT driver for SPF-64.

Spec: docs/spf-formats-and-test-plans.md §12 T1.4 applied to SPF-64 (§4).
Roadmap: Task 1.2.

Thin binding of :mod:`spf.fft._driver` to :mod:`spf.scalar.spf64`. Per-stage
error budget: ``ε_stage ≈ 2·(2⁻¹⁶ + 2⁻³¹) ≈ 3.1e-5``. Twiddles are exact for
every power-of-two N dividing 2³², so the driver's cap is a test-surface
choice only.

Supported sizes: powers of two in [2, 1024].
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fft import _driver
from spf.scalar import spf64
from spf.scalar._constants import SPF64

__all__ = ["fft", "ifft"]

Word = npt.NDArray[np.uint64]

_MIN_N: int = 2
_MAX_N: int = 1024


def fft(x: Word) -> Word:
    """Forward SPF-64 N-point DFT via radix-2 decimation-in-time.

    Parameters
    ----------
    x : ndarray[uint64]
        Length-N input of SPF-64 words. N must be a power of 2 in [2, 1024].

    Returns
    -------
    ndarray[uint64]
        Length-N DFT in natural bin order.
    """
    out = _driver.forward_fft(
        x, scalar=spf64, fmt=SPF64, min_n=_MIN_N, max_n=_MAX_N
    )
    return out.astype(np.uint64, copy=False)


def ifft(x: Word) -> Word:
    """Inverse SPF-64 N-point DFT via ``(1/N) · conj(fft(conj(X)))``."""
    out = _driver.inverse_fft(
        x, scalar=spf64, fmt=SPF64, min_n=_MIN_N, max_n=_MAX_N
    )
    return out.astype(np.uint64, copy=False)
