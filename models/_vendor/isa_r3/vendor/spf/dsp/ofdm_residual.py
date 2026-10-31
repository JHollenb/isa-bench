"""Common-phase-error and timing-ramp residual correction for sOFDM RX.

CPE and timing-ramp estimation run entirely on the SPF ``theta`` field (a u16
phase tick); the data path never decodes a payload bin to Cartesian floats.
``apply_cpe_correction`` and ``apply_timing_correction`` subtract integer-tick
offsets from the ``theta`` field of every SPF word, then repack — SPF-native
phase arithmetic with no spf_mul required.

Circular mean ranges over pilot phase residuals using ``np.exp(j*radians)``
unit vectors. The radians come from the SPF tick field, never from a decode
of the payload magnitude, so ``OFDMRXCounters.cartesian_decode_calls`` stays
exactly at the pilot count contributed by the RX-2 channel estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_pilots import PilotPattern
from spf.dsp.ofdm_rx_plan import OFDMRXPlan
from spf.tensor import pack_words, unpack_L, unpack_theta

_PHASE_MOD = 1 << 16
_PHASE_HALF = 1 << 15
_TWOPI = 2.0 * np.pi


def _radians_to_ticks(radians: float | npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
    return np.round(np.asarray(radians) * _PHASE_MOD / _TWOPI).astype(np.int64)


def _ticks_to_radians(ticks: int | npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    return np.asarray(ticks, dtype=np.float64) * (_TWOPI / _PHASE_MOD)


def _signed_ticks(unsigned_diff: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
    return ((unsigned_diff + _PHASE_HALF) % _PHASE_MOD) - _PHASE_HALF


@dataclass(frozen=True, slots=True)
class CPEEstimate:
    """Per-symbol common-phase-error estimate (in u16 tick units)."""

    theta_per_symbol_ticks: npt.NDArray[np.int32]
    theta_radians: npt.NDArray[np.float64]
    n_pilots_used: int
    method: Literal["circular_mean"]


@dataclass(frozen=True, slots=True)
class TimingEstimate:
    """Per-symbol timing residual: affine ramp θ[k] = offset + slope·k (in ticks)."""

    slope_ticks_per_bin: float
    offset_ticks: float
    tau_samples_per_symbol: float
    rms_residual_ticks: float
    n_pilots_used: int


def _pilot_positions_in_active(
    active: npt.NDArray[np.uint32],
    pilot_subcarriers: npt.NDArray[np.uint32],
) -> npt.NDArray[np.int64]:
    """Return the index of each pilot subcarrier within the active grid."""
    pos = np.searchsorted(active, pilot_subcarriers)
    if np.any(pos >= active.size) or np.any(active[pos] != pilot_subcarriers):
        raise ValueError("pilot subcarriers must be a subset of plan.active_subcarriers")
    return pos.astype(np.int64)


def _pilot_residual_ticks(
    *,
    rx_words: npt.NDArray[np.uint32],
    pilots: PilotPattern,
    plan: OFDMRXPlan,
    h_words_active: npt.NDArray[np.uint32],
    symbol_idx: int,
) -> npt.NDArray[np.int64]:
    """Return signed per-pilot phase residuals (in ticks) for one OFDM symbol.

    Residual = theta(y) - theta(h_estimate) - theta(pilot_value), computed via
    u16 modular subtraction and wrapped to the signed range.
    """
    active = plan.active_subcarriers
    pilot_pos = _pilot_positions_in_active(active, pilots.subcarrier_indices)
    y_theta = unpack_theta(rx_words[0, symbol_idx, pilots.subcarrier_indices, 0]).astype(np.int64)
    h_theta = unpack_theta(h_words_active[pilot_pos]).astype(np.int64)
    pilot_theta = _radians_to_ticks(np.angle(pilots.values)) % _PHASE_MOD
    return _signed_ticks(y_theta - h_theta - pilot_theta)


def estimate_cpe_from_pilots(
    rx_words: npt.NDArray[np.uint32],
    pilots: PilotPattern,
    *,
    plan: OFDMRXPlan,
    h_words_active: npt.NDArray[np.uint32],
    n_symbols: int,
) -> CPEEstimate:
    """Estimate per-symbol common-phase-error as the circular mean of pilot residuals.

    Returns a tick-quantized CPE for every OFDM symbol in the frame; symbols
    that don't carry pilots inherit the latest available estimate (block-pilot
    semantics). The estimator uses unit-vector sums on the residual radians —
    no payload Cartesian decode.
    """
    theta_per_symbol = np.zeros(int(n_symbols), dtype=np.int32)
    pilot_symbols = sorted({int(s) for s in pilots.symbol_indices.tolist()})
    if pilot_symbols[-1] >= int(n_symbols):
        raise ValueError(
            f"pilot symbol_idx={pilot_symbols[-1]} out of range for frame "
            f"with {n_symbols} symbols"
        )
    pilot_set = set(pilot_symbols)
    pilot_radians_by_symbol: dict[int, float] = {}
    for sym_idx in pilot_symbols:
        residual_ticks = _pilot_residual_ticks(
            rx_words=rx_words,
            pilots=pilots,
            plan=plan,
            h_words_active=h_words_active,
            symbol_idx=sym_idx,
        )
        radians = _ticks_to_radians(residual_ticks)
        mean_angle = float(np.angle(np.mean(np.exp(1j * radians))))
        pilot_radians_by_symbol[sym_idx] = mean_angle
        theta_per_symbol[sym_idx] = round(mean_angle * _PHASE_MOD / _TWOPI)
    # Fill non-pilot symbols with the most recent prior estimate, using the
    # first available pilot estimate for symbols before the first pilot.
    last = int(theta_per_symbol[pilot_symbols[0]])
    for s in range(int(n_symbols)):
        if s in pilot_set:
            last = int(theta_per_symbol[s])
        else:
            theta_per_symbol[s] = last
    return CPEEstimate(
        theta_per_symbol_ticks=theta_per_symbol,
        theta_radians=_ticks_to_radians(theta_per_symbol.astype(np.int64)),
        n_pilots_used=pilots.n_pilots,
        method="circular_mean",
    )


def estimate_timing_ramp(
    rx_words: npt.NDArray[np.uint32],
    pilots: PilotPattern,
    *,
    plan: OFDMRXPlan,
    h_words_active: npt.NDArray[np.uint32],
) -> TimingEstimate:
    """Fit an affine theta ramp θ_residual[k] = offset + slope·k across pilots.

    The fit happens on the first pilot symbol; multi-symbol timing tracking
    is reserved for a later commit. Unwrapping uses ``np.unwrap`` on the
    pilot residual radians (no payload Cartesian decode).
    """
    sym_idx = int(pilots.symbol_indices[0])
    residual_ticks = _pilot_residual_ticks(
        rx_words=rx_words,
        pilots=pilots,
        plan=plan,
        h_words_active=h_words_active,
        symbol_idx=sym_idx,
    )
    pilot_sc = pilots.subcarrier_indices.astype(np.float64)
    radians = _ticks_to_radians(residual_ticks)
    unwrapped = np.unwrap(radians) if radians.size > 1 else radians
    if pilot_sc.size < 2:
        slope_rad = 0.0
        offset_rad = float(unwrapped[0]) if unwrapped.size else 0.0
        rms_residual_ticks = 0.0
    else:
        coeffs = np.polyfit(pilot_sc, unwrapped, 1)
        slope_rad = float(coeffs[0])
        offset_rad = float(coeffs[1])
        fit = coeffs[1] + coeffs[0] * pilot_sc
        rms_rad = float(np.sqrt(np.mean((unwrapped - fit) ** 2)))
        rms_residual_ticks = rms_rad * _PHASE_MOD / _TWOPI
    slope_ticks = slope_rad * _PHASE_MOD / _TWOPI
    offset_ticks = offset_rad * _PHASE_MOD / _TWOPI
    tau_samples = slope_rad * plan.n_fft / _TWOPI
    return TimingEstimate(
        slope_ticks_per_bin=slope_ticks,
        offset_ticks=offset_ticks,
        tau_samples_per_symbol=float(tau_samples),
        rms_residual_ticks=float(rms_residual_ticks),
        n_pilots_used=pilots.n_pilots,
    )


def apply_cpe_correction(
    y_words: npt.NDArray[np.uint32],
    cpe: CPEEstimate,
    *,
    plan: OFDMRXPlan,
    n_symbols: int,
) -> npt.NDArray[np.uint32]:
    """Subtract per-symbol CPE from y_words theta field. SPF-native."""
    n_active = plan.active_subcarriers.size
    if y_words.size != n_active * n_symbols:
        raise ValueError(
            f"y_words size {y_words.size} != n_active*n_symbols ({n_active * n_symbols})"
        )
    out = y_words.astype(np.uint32, copy=True)
    L = unpack_L(out).astype(np.int16)
    theta = unpack_theta(out).astype(np.int64)
    for sym_idx in range(int(n_symbols)):
        cpe_ticks = int(cpe.theta_per_symbol_ticks[sym_idx])
        sl = slice(sym_idx * n_active, (sym_idx + 1) * n_active)
        theta[sl] = (theta[sl] - cpe_ticks) % _PHASE_MOD
    return pack_words(L, theta.astype(np.uint16))


def apply_timing_correction(
    y_words: npt.NDArray[np.uint32],
    timing: TimingEstimate,
    *,
    plan: OFDMRXPlan,
    n_symbols: int,
) -> npt.NDArray[np.uint32]:
    """Subtract affine theta ramp from y_words. SPF-native, no spf_mul required."""
    active = plan.active_subcarriers.astype(np.int64)
    n_active = active.size
    if y_words.size != n_active * n_symbols:
        raise ValueError(
            f"y_words size {y_words.size} != n_active*n_symbols ({n_active * n_symbols})"
        )
    ramp = np.round(timing.offset_ticks + timing.slope_ticks_per_bin * active).astype(np.int64)
    out = y_words.astype(np.uint32, copy=True)
    L = unpack_L(out).astype(np.int16)
    theta = unpack_theta(out).astype(np.int64)
    for sym_idx in range(int(n_symbols)):
        sl = slice(sym_idx * n_active, (sym_idx + 1) * n_active)
        theta[sl] = (theta[sl] - ramp) % _PHASE_MOD
    return pack_words(L, theta.astype(np.uint16))


__all__ = [
    "CPEEstimate",
    "TimingEstimate",
    "apply_cpe_correction",
    "apply_timing_correction",
    "estimate_cpe_from_pilots",
    "estimate_timing_ramp",
]
