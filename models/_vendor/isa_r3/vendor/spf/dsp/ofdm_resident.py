"""SPF-resident SISO OFDM operator path — future optimization track.

This module is the **future optimization path** designed for sparse
hardware promotion (KV260 / PL overlay): active sub-carriers live in
resident SPF-grid form across the symbol and FFT folds into the
twiddle-add path. It is parity-validated against the compat RX but is
**not** part of the current hardware-compatibility proof. The
canonical RX for the cross-link 2×2 matrix is
``spf.dsp.ofdm_receiver.receive_sofdm_frame``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from spf.dsp.modulation import psk_demodulate_spf
from spf.dsp.ofdm_compat import Modulation, grid_from_bits
from spf.dsp.ofdm_grid import OFDMBinClass, OFDMResourceGrid
from spf.dsp.ofdm_pilots import (
    ChannelEstimate,
    InterpMode,
    PilotPattern,
    estimate_channel_from_pilots,
)
from spf.dsp.ofdm_residual import (
    CPEEstimate,
    TimingEstimate,
    apply_cpe_correction,
    apply_timing_correction,
    estimate_cpe_from_pilots,
    estimate_timing_ramp,
)
from spf.dsp.ofdm_rx_plan import (
    ChannelPolicy,
    CPEPolicy,
    OFDMRXCounters,
    OFDMRXPlan,
    TimingPolicy,
    _MutableCounters,
)
from spf.native import spf32 as c_spf32
from spf.tensor import SPF32_WORD_ZERO, encode_words, spf_mul_words


@dataclass(frozen=True, slots=True)
class ResidentOFDMCounters:
    """Hot-path structural counters for resident SPF OFDM."""

    boundary_encodes: int = 0
    boundary_decodes: int = 0
    reducer_calls: int = 0
    fb7_butterfly_calls: int = 0
    cartesian_decodes: int = 0
    spf_mul_calls: int = 0
    qam_logproj_calls: int = 0
    numpy_fft_calls: int = 0
    boundary_imports: int = 0


@dataclass(frozen=True, slots=True)
class SISOResidentOFDMResult:
    """Outputs from the resident spectral OFDM path."""

    tx_bits: npt.NDArray[np.uint8]
    rx_bits: npt.NDArray[np.uint8]
    tx_grid: OFDMResourceGrid
    rx_grid: OFDMResourceGrid
    equalized_words: npt.NDArray[np.uint32]
    confidence: npt.NDArray[np.uint8] | None
    counters: ResidentOFDMCounters
    stages: tuple[str, ...]
    demap: str

    @property
    def bit_error_rate(self) -> float:
        return float(np.count_nonzero(self.tx_bits != self.rx_bits)) / float(self.tx_bits.size)


def run_siso_resident_ofdm(
    bits: npt.ArrayLike,
    *,
    modulation: Modulation,
    n_fft: int,
    channel_h: npt.ArrayLike | None = None,
    active_subcarriers: npt.ArrayLike | None = None,
    qam16_threshold_L: int = 256,
) -> SISOResidentOFDMResult:
    """Run a diagonal SPF-resident SISO OFDM symbol path.

    ``channel_h`` is accepted as setup/configuration data and encoded once
    before the resident hot path. The hot path itself operates on packed SPF
    words only.
    """
    tx_bits = np.ascontiguousarray(np.asarray(bits), dtype=np.uint8).ravel()
    tx_grid = grid_from_bits(
        tx_bits,
        modulation=modulation,
        n_fft=n_fft,
        active_subcarriers=active_subcarriers,
    )
    active = tx_grid.active_subcarriers()
    h = _channel(n_fft, channel_h, active)
    h_words = encode_words(h[active])
    h_inv_words = encode_words(1.0 / h[active])
    tx_words = tx_grid.words[0, 0, active, 0]
    rx_active = spf_mul_words(tx_words, h_words)
    equalized = spf_mul_words(rx_active, h_inv_words)
    if modulation == "qpsk":
        rx_bits = psk_demodulate_spf(equalized, 4)
        confidence = None
        demap = "qpsk_phase"
        qam_calls = 0
    else:
        rx_bits, confidence = c_spf32.qam16_logproj_eq_demod_fused(
            rx_active,
            h_inv_words,
            threshold_L=qam16_threshold_L,
        )
        demap = "qam16_logproj"
        qam_calls = 1
    rx_words = tx_grid.words.copy()
    rx_words[0, 0, active, 0] = rx_active
    return SISOResidentOFDMResult(
        tx_bits=tx_bits,
        rx_bits=rx_bits,
        tx_grid=tx_grid,
        rx_grid=OFDMResourceGrid(rx_words, tx_grid.bin_class.copy()),
        equalized_words=equalized,
        confidence=confidence,
        counters=ResidentOFDMCounters(spf_mul_calls=2, qam_logproj_calls=qam_calls),
        stages=("spf_bins", "spf_channel_multiply", "spf_equalize", demap),
        demap=demap,
    )


def _channel(
    n_fft: int,
    channel_h: npt.ArrayLike | None,
    active: npt.NDArray[np.uint32] | None = None,
) -> npt.NDArray[np.complex128]:
    if channel_h is None:
        return np.ones(n_fft, dtype=np.complex128)
    h = np.ascontiguousarray(np.asarray(channel_h), dtype=np.complex128).ravel()
    if h.size != n_fft:
        raise ValueError(f"channel_h length {h.size} != n_fft {n_fft}")
    check = h[active] if active is not None else h
    if np.any(check == 0.0):
        raise ValueError("channel_h must not contain zero response at active subcarriers")
    return h


@dataclass(frozen=True, slots=True)
class ResidentRXResult:
    """Outputs from the resident-bins receive path.

    The function contract guarantees ``counters.numpy_fft_calls == 0`` and
    ``counters.cartesian_decode_calls == 0``; pilot-side Cartesian decodes
    used during channel estimation are tracked in
    ``counters.pilot_cartesian_decode_calls`` so the hot-path-clean
    invariant remains observable.
    """

    rx_bits: npt.NDArray[np.uint8]
    equalized_words: npt.NDArray[np.uint32]
    confidence: npt.NDArray[np.uint8] | None
    counters: OFDMRXCounters
    plan: OFDMRXPlan
    demap: str
    stages: tuple[str, ...]
    channel_estimate: ChannelEstimate | None = None
    cpe: CPEEstimate | None = None
    timing: TimingEstimate | None = None
    bit_error_rate: float | None = None


def _bits_per_symbol_resident(modulation: Modulation) -> int:
    if modulation == "qpsk":
        return 2
    if modulation == "qam16":
        return 4
    bad = cast(Literal["unsupported"], modulation)
    raise ValueError(f"unsupported modulation {bad!r}")


def _payload_ber_resident(
    *,
    rx_bits: npt.NDArray[np.uint8],
    tx_bits: npt.ArrayLike | None,
    plan: OFDMRXPlan,
    active: npt.NDArray[np.uint32],
    pilots: PilotPattern | None,
    n_symbols: int,
) -> float | None:
    if tx_bits is None:
        return None
    tx_arr = np.ascontiguousarray(np.asarray(tx_bits), dtype=np.uint8).ravel()
    if pilots is None:
        if rx_bits.size != tx_arr.size:
            raise ValueError(
                f"tx_bits size {tx_arr.size} does not match rx_bits size {rx_bits.size}"
            )
        if tx_arr.size == 0:
            return 0.0
        return float(np.count_nonzero(tx_arr != rx_bits)) / float(tx_arr.size)
    bps = _bits_per_symbol_resident(plan.modulation)
    pilot_set = {int(x) for x in pilots.subcarrier_indices.tolist()}
    pilot_symbols = {int(x) for x in pilots.symbol_indices.tolist()}
    mask_parts = []
    for sym_idx in range(n_symbols):
        is_data_sc = np.asarray(
            [
                not (sym_idx in pilot_symbols and int(k) in pilot_set)
                for k in active.tolist()
            ],
            dtype=bool,
        )
        mask_parts.append(np.repeat(is_data_sc, bps))
    mask = np.concatenate(mask_parts) if mask_parts else np.empty(0, dtype=bool)
    data_rx_bits = rx_bits[mask]
    if data_rx_bits.size != tx_arr.size:
        raise ValueError(
            f"tx_bits size {tx_arr.size} does not match data-only rx_bits "
            f"size {data_rx_bits.size}"
        )
    if tx_arr.size == 0:
        return 0.0
    return float(np.count_nonzero(tx_arr != data_rx_bits)) / float(tx_arr.size)


def _resident_demap(
    y_words: npt.NDArray[np.uint32],
    h_inv_words_tiled: npt.NDArray[np.uint32],
    equalized_words: npt.NDArray[np.uint32],
    modulation: Modulation,
    *,
    threshold_l: int,
    cpe_theta: int,
    phase_bits: int,
    agc_l_offset: int,
    counters: _MutableCounters,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None, str]:
    if modulation == "qpsk":
        counters.qpsk_phase_demap_calls += 1
        return psk_demodulate_spf(equalized_words, 4), None, "qpsk_phase"
    if modulation == "qam16":
        bits, confidence = c_spf32.qam16_logproj_eq_demod_fused(
            y_words,
            h_inv_words_tiled,
            cpe_theta=cpe_theta,
            agc_L_offset=agc_l_offset,
            threshold_L=threshold_l,
            phase_bits=phase_bits,
        )
        counters.qam_logproj_calls += int(y_words.size)
        return bits, confidence, "qam16_logproj"
    bad = cast(Literal["unsupported"], modulation)
    raise ValueError(f"unsupported modulation {bad!r}")


def _build_rx_grid(
    y_words_by_symbol: npt.NDArray[np.uint32],
    plan: OFDMRXPlan,
    pilots: PilotPattern | None,
) -> OFDMResourceGrid:
    n_symbols, _n_active = y_words_by_symbol.shape
    active = plan.active_subcarriers
    shape = (1, n_symbols, plan.n_fft, 1)
    words = np.full(shape, SPF32_WORD_ZERO, dtype=np.uint32)
    classes = np.full(shape, int(OFDMBinClass.GUARD), dtype=np.int16)
    classes[:, :, 0, :] = int(OFDMBinClass.DC_NULL)
    classes[0, :, active, 0] = int(OFDMBinClass.DATA)
    for s in range(n_symbols):
        words[0, s, active, 0] = y_words_by_symbol[s]
    if pilots is not None:
        for sym_idx in pilots.symbol_indices.tolist():
            if int(sym_idx) >= n_symbols:
                raise ValueError(
                    f"pilot symbol_idx={int(sym_idx)} out of range for frame "
                    f"with {n_symbols} symbols"
                )
            classes[0, int(sym_idx), pilots.subcarrier_indices, 0] = int(OFDMBinClass.PILOT)
    return OFDMResourceGrid(words, classes)


def receive_sofdm_resident_bins(
    y_words_by_symbol: npt.ArrayLike,
    plan: OFDMRXPlan,
    *,
    h_inv_words: npt.ArrayLike | None = None,
    h_words_active: npt.ArrayLike | None = None,
    pilots: PilotPattern | None = None,
    tx_bits: npt.ArrayLike | None = None,
    pilot_interp_mode: InterpMode = "spf_native_logphase_interp",
) -> ResidentRXResult:
    """Run equalize + demap from already-spectral SPF bins.

    This entrypoint never receives IQ samples and never calls ``np.fft.*``;
    the function asserts ``numpy_fft_calls == 0`` and
    ``cartesian_decode_calls == 0`` on return. Pilot-channel-estimate
    Cartesian decodes are routed to ``pilot_cartesian_decode_calls`` so the
    hot-path-clean invariant remains testable when ``channel_policy`` is
    ``PILOT_ESTIMATED``.
    """
    y_by_symbol = np.ascontiguousarray(
        np.asarray(y_words_by_symbol, dtype=np.uint32)
    )
    if y_by_symbol.ndim != 2:
        raise ValueError("y_words_by_symbol must be shape (n_symbols, n_active)")
    n_symbols, n_active = y_by_symbol.shape
    if n_active != plan.active_subcarriers.size:
        raise ValueError(
            f"n_active={n_active} does not match plan.active_subcarriers.size="
            f"{plan.active_subcarriers.size}"
        )
    counters = _MutableCounters()
    active = plan.active_subcarriers
    stages: list[str] = ["spf_bins"]

    rx_grid = _build_rx_grid(y_by_symbol, plan, pilots)

    channel_estimate: ChannelEstimate | None = None
    if h_inv_words is not None:
        h_inv_active = np.ascontiguousarray(np.asarray(h_inv_words, dtype=np.uint32)).ravel()
        if h_inv_active.size != n_active:
            raise ValueError(
                f"h_inv_words size {h_inv_active.size} != n_active {n_active}"
            )
        h_words_a = (
            np.ascontiguousarray(np.asarray(h_words_active, dtype=np.uint32)).ravel()
            if h_words_active is not None
            else np.empty(0, dtype=np.uint32)
        )
        stages.append("h_inv_known")
    elif plan.channel_policy is ChannelPolicy.PILOT_ESTIMATED:
        if pilots is None:
            raise ValueError("PILOT_ESTIMATED channel_policy requires a PilotPattern")
        channel_estimate = estimate_channel_from_pilots(
            rx_grid, pilots, plan=plan, mode=pilot_interp_mode
        )
        # Pilot decodes are setup-side, not hot-path, so route them to the
        # pilot-specific counter.
        counters.pilot_cartesian_decode_calls += pilots.n_pilots
        counters.spf_encode_calls += 2
        h_inv_active = channel_estimate.h_inv_words
        h_words_a = channel_estimate.h_words
        stages.append(
            "pilot_estimate_h_spf_native"
            if pilot_interp_mode == "spf_native_logphase_interp"
            else "pilot_estimate_h_compat"
        )
    elif plan.channel_policy is ChannelPolicy.KNOWN and plan.channel_h is not None:
        h_active_complex = plan.channel_h[active]
        if np.any(h_active_complex == 0.0):
            raise ValueError("plan.channel_h has zero response at an active subcarrier")
        h_inv_active = encode_words(1.0 / h_active_complex)
        h_words_a = encode_words(h_active_complex)
        counters.spf_encode_calls += 2
        stages.append("h_known_encode")
    else:
        raise ValueError(
            "receive_sofdm_resident_bins requires h_inv_words, "
            "channel_policy=PILOT_ESTIMATED with pilots, or KNOWN+channel_h"
        )
    counters.equalizer_inverts += 1

    cpe: CPEEstimate | None = None
    timing: TimingEstimate | None = None
    y_words = y_by_symbol.reshape(-1)
    if plan.timing_policy is TimingPolicy.PILOT_PHASE_SLOPE:
        if pilots is None:
            raise ValueError("PILOT_PHASE_SLOPE requires a PilotPattern")
        if h_words_a.size == 0:
            raise ValueError("timing estimator requires h_words_active")
        timing = estimate_timing_ramp(rx_grid.words, pilots, plan=plan, h_words_active=h_words_a)
        y_words = apply_timing_correction(y_words, timing, plan=plan, n_symbols=n_symbols)
        stages.append("timing_correction")
    if plan.cpe_policy is CPEPolicy.PILOT_COMMON_PHASE:
        if pilots is None:
            raise ValueError("PILOT_COMMON_PHASE requires a PilotPattern")
        if h_words_a.size == 0:
            raise ValueError("CPE estimator requires h_words_active")
        cpe = estimate_cpe_from_pilots(
            rx_grid.words, pilots, plan=plan, h_words_active=h_words_a, n_symbols=n_symbols
        )
        y_words = apply_cpe_correction(y_words, cpe, plan=plan, n_symbols=n_symbols)
        stages.append("cpe_correction")

    h_inv_tiled = np.tile(h_inv_active, n_symbols)
    equalized = spf_mul_words(y_words, h_inv_tiled)
    counters.bump_spf_mul_operational()
    rx_bits, confidence, demap = _resident_demap(
        y_words,
        h_inv_tiled,
        equalized,
        plan.modulation,
        threshold_l=plan.qam16_threshold_l,
        cpe_theta=int(plan.cpe_theta) & 0xFFFF,
        phase_bits=plan.phase_bits,
        agc_l_offset=plan.agc_l_offset,
        counters=counters,
    )
    stages.append(demap)

    bit_error_rate = _payload_ber_resident(
        rx_bits=rx_bits,
        tx_bits=tx_bits,
        plan=plan,
        active=active,
        pilots=pilots,
        n_symbols=n_symbols,
    )

    frozen = counters.freeze()
    if frozen.numpy_fft_calls != 0:
        raise RuntimeError(
            f"resident path must not invoke numpy_fft (got {frozen.numpy_fft_calls})"
        )
    if frozen.cartesian_decode_calls != 0:
        raise RuntimeError(
            "resident hot path must not Cartesian-decode payload bins "
            f"(got {frozen.cartesian_decode_calls})"
        )

    return ResidentRXResult(
        rx_bits=rx_bits,
        equalized_words=equalized,
        confidence=confidence,
        counters=frozen,
        plan=plan,
        demap=demap,
        stages=tuple(stages),
        channel_estimate=channel_estimate,
        cpe=cpe,
        timing=timing,
        bit_error_rate=bit_error_rate,
    )


__all__ = [
    "ResidentOFDMCounters",
    "ResidentRXResult",
    "SISOResidentOFDMResult",
    "receive_sofdm_resident_bins",
    "run_siso_resident_ofdm",
]
