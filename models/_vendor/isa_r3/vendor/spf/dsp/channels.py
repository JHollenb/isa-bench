"""Channel models: AWGN and Rayleigh multipath.

Promoted from ``benchmarks/dsp/_helpers.py``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

__all__ = [
    "apply_freq_channel",
    "awgn_noise",
    "noise_variance_for_snr",
    "rayleigh_multipath",
]

Complex = npt.NDArray[np.complex128]


def noise_variance_for_snr(
    signal_power: float | npt.NDArray[np.floating],
    snr_db: float,
) -> float | npt.NDArray[np.floating]:
    """Complex-noise variance σ² to achieve ``snr_db`` for the given signal power."""
    scale: float = 10.0 ** (snr_db / 10.0)
    if isinstance(signal_power, np.ndarray):
        arr: npt.NDArray[np.floating] = signal_power / scale
        return arr
    return float(signal_power) / scale


def awgn_noise(
    shape: tuple[int, ...],
    noise_var: float,
    rng: np.random.Generator,
) -> Complex:
    """Draw complex AWGN with total variance ``noise_var`` (σ²/2 per real component)."""
    scale = math.sqrt(noise_var / 2.0)
    return np.array(
        rng.standard_normal(shape) + 1j * rng.standard_normal(shape), dtype=np.complex128
    ) * scale


def rayleigh_multipath(
    n_sub: int,
    n_taps: int,
    rng: np.random.Generator,
    *,
    rms_delay_taps: float | None = None,
    n_realizations: int = 1,
) -> Complex:
    """Frequency response ``H[k]`` of an L-tap Rayleigh multipath channel.

    Taps are independent zero-mean complex Gaussians with exponential power
    decay; the frequency response is obtained by zero-padding to ``n_sub``
    and FFT.  Total channel power is normalised to 1.

    Parameters
    ----------
    n_sub : int
        Number of subcarriers (= FFT size N).
    n_taps : int
        Number of channel taps in [1, n_sub].
    rng : np.random.Generator
        Random generator for reproducibility.
    rms_delay_taps : float, optional
        Exponential decay constant in tap units.  Default ``n_taps / 3``.
    n_realizations : int
        If > 1, returns shape ``(n_realizations, n_sub)``; default 1 → ``(n_sub,)``.
    """
    if n_taps < 1 or n_taps > n_sub:
        raise ValueError(f"n_taps must be in [1, n_sub={n_sub}], got {n_taps}")
    if n_realizations < 1:
        raise ValueError(f"n_realizations must be >= 1, got {n_realizations}")
    decay = rms_delay_taps if rms_delay_taps is not None else max(n_taps / 3.0, 1.0)
    tap_powers = np.exp(-np.arange(n_taps) / decay)
    tap_powers /= tap_powers.sum()
    tap_shape: tuple[int, ...] = (n_taps,) if n_realizations == 1 else (n_realizations, n_taps)
    taps = (
        rng.standard_normal(tap_shape) + 1j * rng.standard_normal(tap_shape)
    ) * np.sqrt(tap_powers / 2.0)
    h_shape: tuple[int, ...] = (n_sub,) if n_realizations == 1 else (n_realizations, n_sub)
    h = np.zeros(h_shape, dtype=np.complex128)
    h[..., :n_taps] = taps
    return np.fft.fft(h, axis=-1)


def apply_freq_channel(time_signal: Complex, H: Complex) -> Complex:
    """Apply a frequency-domain channel: ``IFFT(H · FFT(s))``."""
    return np.fft.ifft(H * np.fft.fft(time_signal))
