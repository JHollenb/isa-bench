"""Frequency-domain equalization: ZF and LS pilot-based channel estimation.

Promoted from ``benchmarks/dsp/_helpers.py``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar import spf32

__all__ = [
    "ls_channel_estimate",
    "ls_channel_estimate_fp",
    "ls_channel_estimate_spf",
    "zf_equalize",
    "zf_equalize_fp",
    "zf_equalize_spf",
]

Word = npt.NDArray[np.uint32]
Complex = npt.NDArray[np.complex128]


def zf_equalize_spf(rx_spectrum: Word, channel_spectrum: Word) -> Word:
    """Zero-forcing equalizer: ``Y = RX / H`` via ``spf32.div`` (2 integer subtracts/element)."""
    return spf32.div(rx_spectrum, channel_spectrum)


def zf_equalize_fp(rx_spectrum: Complex, channel_spectrum: Complex) -> Complex:
    """FP ZF equalizer: element-wise complex division."""
    return rx_spectrum / channel_spectrum


def zf_equalize(
    rx_spectrum: Word | Complex,
    channel_spectrum: Word | Complex,
    *,
    spf: bool | None = None,
) -> Word | Complex:
    """Unified ZF equalizer.  Auto-detects SPF vs FP from dtype if ``spf=None``."""
    use_spf = (np.asarray(rx_spectrum).dtype == np.uint32) if spf is None else spf
    if use_spf:
        return zf_equalize_spf(
            np.asarray(rx_spectrum, dtype=np.uint32),
            np.asarray(channel_spectrum, dtype=np.uint32),
        )
    return zf_equalize_fp(
        np.asarray(rx_spectrum, dtype=np.complex128),
        np.asarray(channel_spectrum, dtype=np.complex128),
    )


def ls_channel_estimate_spf(
    rx_pilot_spectrum: Word, known_pilot_spectrum: Word
) -> Word:
    """LS channel estimate ``H_hat[k] = RX[k] / pilot[k]`` (SPF-32, 2 integer subtracts/element).

    ``rx_pilot_spectrum`` may be batched ``(n_blocks, n_sub)``; ``known_pilot_spectrum``
    broadcasts as ``(n_sub,)``.
    """
    return spf32.div(rx_pilot_spectrum, known_pilot_spectrum)


def ls_channel_estimate_fp(
    rx_pilot_spectrum: Complex, known_pilot_spectrum: Complex
) -> Complex:
    """FP LS channel estimate: element-wise complex division."""
    return rx_pilot_spectrum / known_pilot_spectrum


def ls_channel_estimate(
    rx_pilot_spectrum: Word | Complex,
    known_pilot_spectrum: Word | Complex,
    *,
    spf: bool | None = None,
) -> Word | Complex:
    """Unified LS channel estimator.  Auto-detects SPF vs FP from dtype if ``spf=None``."""
    use_spf = (np.asarray(rx_pilot_spectrum).dtype == np.uint32) if spf is None else spf
    if use_spf:
        return ls_channel_estimate_spf(
            np.asarray(rx_pilot_spectrum, dtype=np.uint32),
            np.asarray(known_pilot_spectrum, dtype=np.uint32),
        )
    return ls_channel_estimate_fp(
        np.asarray(rx_pilot_spectrum, dtype=np.complex128),
        np.asarray(known_pilot_spectrum, dtype=np.complex128),
    )
