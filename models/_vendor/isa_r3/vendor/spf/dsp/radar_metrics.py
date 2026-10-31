"""Radar pulse-compression quality metrics: peak, sidelobe, processing gain.

These complement :mod:`spf.dsp.metrics` (BER / EVM) with the range-profile
figures of merit a radar engineer grades a matched filter on:

- peak range-bin localization (:func:`peak_bin`, :func:`range_error_bins`),
- range sidelobe quality (:func:`pslr_db`, :func:`islr_db`),
- peak SNR / measured processing gain (:func:`peak_snr_db`),
- SPF-vs-fp64 range-profile fidelity (:func:`range_profile_nmse_db`).

Sidelobe metrics operate on a real, non-negative *power* profile
``P[n] = |y[n]|²``; use :func:`power_profile` to derive it from a complex range
profile. Range-profile EVM is the general complex EVM in :func:`spf.dsp.metrics.evm`
and is not re-defined here.

Oracle: the matched filter of an unwindowed (rectangular-amplitude) LFM chirp
compresses to a ``sinc`` whose first range sidelobe sits at PSLR ≈ −13.26 dB —
the standard textbook value, used to unit-test :func:`pslr_db`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "islr_db",
    "mainlobe_bounds",
    "peak_bin",
    "peak_snr_db",
    "power_profile",
    "pslr_db",
    "range_error_bins",
    "range_profile_nmse_db",
]


def power_profile(y: npt.NDArray[np.complex128]) -> npt.NDArray[np.float64]:
    """Return ``|y[n]|²`` as a float64 power profile."""
    arr = np.asarray(y, dtype=np.complex128).ravel()
    return (arr.real * arr.real + arr.imag * arr.imag).astype(np.float64)


def peak_bin(power: npt.NDArray[np.float64]) -> int:
    """Index of the maximum-power bin."""
    p = np.asarray(power, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("power profile is empty")
    return int(np.argmax(p))


def mainlobe_bounds(
    power: npt.NDArray[np.float64], *, peak: int | None = None
) -> tuple[int, int]:
    """Inclusive ``[lo, hi]`` bin bounds of the mainlobe around the peak.

    Walks down from the peak in each direction until the power stops decreasing
    (the first null / valley). The returned bounds are those valley bins, so the
    mainlobe occupies ``[lo, hi]`` inclusive and the sidelobe region is its
    complement.
    """
    p = np.asarray(power, dtype=np.float64).ravel()
    n = p.size
    if n == 0:
        raise ValueError("power profile is empty")
    pk = peak_bin(p) if peak is None else int(peak)

    hi = pk
    while hi + 1 < n and p[hi + 1] < p[hi]:
        hi += 1
    lo = pk
    while lo - 1 >= 0 and p[lo - 1] < p[lo]:
        lo -= 1
    return lo, hi


def pslr_db(
    power: npt.NDArray[np.float64],
    *,
    peak: int | None = None,
    mainlobe: tuple[int, int] | None = None,
) -> float:
    """Peak-to-sidelobe ratio in dB: ``10·log10(max_sidelobe / peak)``.

    Negative; closer to zero is worse. The mainlobe is auto-detected via
    :func:`mainlobe_bounds` unless an explicit ``mainlobe`` ``(lo, hi)`` is given.
    """
    p = np.asarray(power, dtype=np.float64).ravel()
    pk = peak_bin(p) if peak is None else int(peak)
    lo, hi = mainlobe_bounds(p, peak=pk) if mainlobe is None else mainlobe
    mask = np.ones(p.size, dtype=bool)
    mask[lo : hi + 1] = False
    side = p[mask]
    peak_power = float(p[pk])
    if side.size == 0 or peak_power <= 0.0:
        return float("-inf")
    max_side = float(side.max())
    if max_side <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(max_side / peak_power))


def islr_db(
    power: npt.NDArray[np.float64],
    *,
    peak: int | None = None,
    mainlobe: tuple[int, int] | None = None,
) -> float:
    """Integrated sidelobe ratio in dB: ``10·log10(Σ_sidelobe / Σ_mainlobe)``."""
    p = np.asarray(power, dtype=np.float64).ravel()
    pk = peak_bin(p) if peak is None else int(peak)
    lo, hi = mainlobe_bounds(p, peak=pk) if mainlobe is None else mainlobe
    main_sum = float(p[lo : hi + 1].sum())
    side_sum = float(p.sum()) - main_sum
    if main_sum <= 0.0:
        return float("inf")
    if side_sum <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(side_sum / main_sum))


def peak_snr_db(
    power: npt.NDArray[np.float64],
    *,
    peak: int | None = None,
    guard: int | None = None,
    noise_floor: float | None = None,
) -> float:
    """Peak-to-noise-floor ratio in dB.

    The noise floor defaults to the *median* power outside a guard window
    around the peak (median is robust to sidelobes and other targets). Pass an
    explicit ``noise_floor`` (e.g. ``σ²``) to compare against a known level.
    """
    p = np.asarray(power, dtype=np.float64).ravel()
    n = p.size
    pk = peak_bin(p) if peak is None else int(peak)
    peak_power = float(p[pk])
    if noise_floor is None:
        if guard is None:
            lo, hi = mainlobe_bounds(p, peak=pk)
            guard = max(hi - pk, pk - lo) + 1
        mask = np.ones(n, dtype=bool)
        mask[max(0, pk - guard) : min(n, pk + guard + 1)] = False
        floor_vals = p[mask]
        if floor_vals.size == 0:
            return float("inf")
        noise_floor = float(np.median(floor_vals))
    if noise_floor <= 0.0 or peak_power <= 0.0:
        return float("inf")
    return 10.0 * float(np.log10(peak_power / noise_floor))


def range_error_bins(
    power: npt.NDArray[np.float64], true_bin: int, *, peak: int | None = None
) -> int:
    """Signed peak-bin error ``peak − true_bin`` (bins)."""
    pk = peak_bin(power) if peak is None else int(peak)
    return int(pk - int(true_bin))


def range_profile_nmse_db(
    spf_profile: npt.NDArray[np.complex128],
    ref_profile: npt.NDArray[np.complex128],
) -> float:
    """Normalized MSE in dB of an SPF range profile vs a reference profile.

    ``10·log10( Σ|spf − ref|² / Σ|ref|² )``. Equal to ``20·log10(EVM)``.
    """
    spf = np.asarray(spf_profile, dtype=np.complex128).ravel()
    ref = np.asarray(ref_profile, dtype=np.complex128).ravel()
    if spf.shape != ref.shape:
        raise ValueError(f"Shape mismatch: spf={spf.shape}, ref={ref.shape}")
    err = float(np.sum(np.abs(spf - ref) ** 2))
    sig = float(np.sum(np.abs(ref) ** 2))
    if sig == 0.0:
        return float("-inf") if err == 0.0 else float("inf")
    if err == 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(err / sig))
