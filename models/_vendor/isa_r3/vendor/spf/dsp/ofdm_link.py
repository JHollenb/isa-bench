"""Fixture-driven sOFDM TX/DAC/ADC/RX link harness."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias, TypeVar, cast

import numpy as np
import numpy.typing as npt

from spf.dsp.fft_backend import fft_rows
from spf.dsp.metrics import evm
from spf.dsp.metrics import evm_db as evm_db_metric
from spf.dsp.modulation import (
    psk_demodulate_fp,
    psk_modulate_spf,
    qam16_demodulate_fp,
    qam16_modulate_fp,
    qam16_modulate_spf,
)
from spf.dsp.ofdm_compat import Modulation, apply_diagonal_active_words
from spf.dsp.ofdm_grid import OFDMBinClass, OFDMResourceGrid
from spf.dsp.ofdm_receiver import (
    ADCConfig,
    ADCResult,
    OFDMReceiverConfig,  # noqa: F401  retained for downstream importers
    SOFDMReceiverResult,
    receive_sofdm_frame,
    simulate_iq_adc,
)
from spf.dsp.ofdm_rx_plan import (
    ChannelPolicy,
    OFDMRXPlan,
)
from spf.dsp.ofdm_synth import export_iq16_ifft, export_iq16_nco
from spf.tensor import SPF32_WORD_ZERO, SynthPlan, SynthStats, decode_words, encode_words

DacMode: TypeAlias = Literal["ifft", "nco"]
MetricValue: TypeAlias = str | int | float | bool | None
MetricsPayload: TypeAlias = dict[str, object]
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class OFDMTXConfig:
    """Deterministic sOFDM TX frame configuration."""

    n_fft: int
    n_symbols: int
    ncp: int
    modulation: Modulation
    active_subcarriers: npt.NDArray[np.uint32]
    seed: int = 20260521


@dataclass(frozen=True, slots=True)
class DACConfig:
    """Simulated normal-DAC export policy."""

    mode: DacMode = "ifft"
    sparse_threshold: int = 96


@dataclass(frozen=True, slots=True)
class SOFDMLoadConfig:
    """End-to-end sOFDM link load."""

    scenario: str
    tx: OFDMTXConfig
    channel_h: npt.NDArray[np.complex128]
    dac: DACConfig = field(default_factory=DACConfig)
    adc: ADCConfig = field(default_factory=ADCConfig)
    sample_rate_hz: float = 1_000_000.0
    evm_max: float = 0.08
    require_bit_exact: bool = True


@dataclass(frozen=True, slots=True)
class SOFDMTXFrame:
    """Generated TX bits and resource grid."""

    bits: npt.NDArray[np.uint8]
    grid: OFDMResourceGrid
    active_symbols: npt.NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class DACResult:
    """DAC waveform and export telemetry."""

    iq: npt.NDArray[np.int16]
    mode: DacMode
    output_format: str
    sample_count: int
    papr: float
    clip_count: int
    active_tones: int
    ifft_calls: int
    nco_tones: int


@dataclass(frozen=True, slots=True)
class SOFDMLinkMetrics:
    """Structured metrics for a TX/DAC/ADC/RX link run."""

    load: dict[str, MetricValue]
    correctness: dict[str, MetricValue]
    speed: MetricsPayload
    dac: dict[str, MetricValue]
    adc: dict[str, MetricValue]
    rx: dict[str, MetricValue]
    baseline: MetricsPayload = field(default_factory=dict)
    comparison: dict[str, MetricValue] = field(default_factory=dict)

    def to_json_dict(self) -> MetricsPayload:
        return {
            "load": self.load,
            "correctness": self.correctness,
            "speed": self.speed,
            "dac": self.dac,
            "adc": self.adc,
            "rx": self.rx,
            "baseline": self.baseline,
            "comparison": self.comparison,
        }


@dataclass(frozen=True, slots=True)
class SOFDMLinkResult:
    """Full artifacts from one sOFDM link load."""

    config: SOFDMLoadConfig
    tx: SOFDMTXFrame
    channel_grid: OFDMResourceGrid
    dac: DACResult
    adc: ADCResult
    receiver: SOFDMReceiverResult
    baseline: BaselineOFDMResult
    metrics: SOFDMLinkMetrics


@dataclass(frozen=True, slots=True)
class BaselineOFDMResult:
    """Basic complex OFDM TX/DAC/ADC/RX baseline artifacts and metrics."""

    tx_bits: npt.NDArray[np.uint8]
    rx_bits: npt.NDArray[np.uint8]
    tx_symbols: npt.NDArray[np.complex128]
    equalized_symbols: npt.NDArray[np.complex128]
    dac: ADCResult
    adc: ADCResult
    metrics: MetricsPayload


TXKind: TypeAlias = Literal["fp", "sofdm"]
RXKind: TypeAlias = Literal["fp", "sofdm"]


@dataclass(frozen=True, slots=True)
class CrossLinkCell:
    """One (TX backend × RX backend) cell of the cross-link matrix."""

    tx_kind: TXKind
    rx_kind: RXKind
    rx_bits: npt.NDArray[np.uint8]
    equalized_symbols: npt.NDArray[np.complex128]
    bit_errors: int
    bits_total: int
    ber: float
    evm: float | None
    evm_db: float | None
    snr_db_estimate: float | None
    tx_ns: int
    channel_ns: int
    dac_ns: int
    adc_ns: int
    rx_ns: int
    total_ns: int


@dataclass(frozen=True, slots=True)
class CrossLinkIdentity:
    """Cross-RX identity: same boundary IQ → same rx_bits across RX backends?"""

    fp_tx_rx_bits_match: bool
    fp_tx_rx_bit_diff: int
    sofdm_tx_rx_bits_match: bool
    sofdm_tx_rx_bit_diff: int


@dataclass(frozen=True, slots=True)
class CrossLinkMatrix:
    """Four-cell TX/RX backend matrix sharing one bit stream and channel."""

    config: SOFDMLoadConfig
    tx_bits: npt.NDArray[np.uint8]
    fp_post_adc: npt.NDArray[np.complex128]
    sofdm_post_adc: npt.NDArray[np.complex128]
    fp_tx_fp_rx: CrossLinkCell
    fp_tx_sofdm_rx: CrossLinkCell
    sofdm_tx_fp_rx: CrossLinkCell
    sofdm_tx_sofdm_rx: CrossLinkCell
    identity: CrossLinkIdentity

    def cells(self) -> tuple[CrossLinkCell, ...]:
        return (
            self.fp_tx_fp_rx,
            self.fp_tx_sofdm_rx,
            self.sofdm_tx_fp_rx,
            self.sofdm_tx_sofdm_rx,
        )

    def to_json_dict(self) -> MetricsPayload:
        return {
            "scenario": self.config.scenario,
            "cells": [_cell_to_metrics(c) for c in self.cells()],
            "identity": {
                "fp_tx_rx_bits_match": self.identity.fp_tx_rx_bits_match,
                "fp_tx_rx_bit_diff": self.identity.fp_tx_rx_bit_diff,
                "sofdm_tx_rx_bits_match": self.identity.sofdm_tx_rx_bits_match,
                "sofdm_tx_rx_bit_diff": self.identity.sofdm_tx_rx_bit_diff,
            },
        }


def build_sofdm_tx_frame(
    config: OFDMTXConfig,
    *,
    collect_reference_symbols: bool = True,
) -> SOFDMTXFrame:
    """Build a deterministic multi-symbol sOFDM resource grid."""
    _validate_tx_config(config)
    bits_per_symbol = _bits_per_symbol(config.modulation)
    bits_per_ofdm_symbol = int(config.active_subcarriers.size * bits_per_symbol)
    rng = np.random.default_rng(config.seed)
    bits = rng.integers(
        0,
        2,
        size=config.n_symbols * bits_per_ofdm_symbol,
        dtype=np.uint8,
    )
    shape = (1, config.n_symbols, config.n_fft, 1)
    words = np.full(shape, SPF32_WORD_ZERO, dtype=np.uint32)
    classes = np.full(shape, int(OFDMBinClass.GUARD), dtype=np.int16)
    classes[:, :, 0, :] = int(OFDMBinClass.DC_NULL)

    active_words = []
    for symbol in range(config.n_symbols):
        start = symbol * bits_per_ofdm_symbol
        stop = start + bits_per_ofdm_symbol
        symbol_words = _modulate(bits[start:stop], config.modulation)
        words[0, symbol, config.active_subcarriers, 0] = symbol_words
        classes[0, symbol, config.active_subcarriers, 0] = int(OFDMBinClass.DATA)
        active_words.append(symbol_words)

    # Hot path: classes / words were just assembled with the active mask
    # one line above. Skip the full ``validate()`` (n_isin × n_bin_classes)
    # in favour of the trusted constructor.
    grid = OFDMResourceGrid.from_trusted(words, classes)
    active = np.concatenate(active_words)
    active_symbols = decode_words(active) if collect_reference_symbols else np.empty(0, dtype=np.complex128)
    return SOFDMTXFrame(bits=bits, grid=grid, active_symbols=active_symbols)


def apply_known_channel(
    grid: OFDMResourceGrid,
    channel_h: npt.ArrayLike,
    *,
    active_subcarriers: npt.ArrayLike | None = None,
) -> OFDMResourceGrid:
    """Apply a known diagonal frequency-domain channel to active bins.

    Pass ``active_subcarriers`` to skip the grid scan on the hot path when
    the active set is already known from config.
    """
    h = _channel(grid.shape[2], channel_h)
    h_words = encode_words(h)
    active = (
        grid.active_subcarriers()
        if active_subcarriers is None
        else np.ascontiguousarray(np.asarray(active_subcarriers), dtype=np.uint32)
    )
    return apply_diagonal_active_words(
        grid,
        active,
        h_words,
        trusted=True,
    )


def simulate_iq_dac(
    grid: OFDMResourceGrid,
    *,
    ncp: int,
    config: DACConfig | None = None,
    validate_grid: bool = True,
) -> DACResult:
    """Export a resource grid through the configured simulated DAC boundary."""
    dac_config = DACConfig() if config is None else config
    if dac_config.mode not in ("ifft", "nco"):
        raise ValueError(f"unsupported DAC mode {dac_config.mode!r}")
    plan = SynthPlan(
        nfft=grid.shape[2],
        ncp=ncp,
        n_symbols=grid.shape[1],
        sparse_threshold=dac_config.sparse_threshold,
    )
    if dac_config.mode == "ifft":
        iq, stats = export_iq16_ifft(grid, plan, validate=validate_grid)
    else:
        iq, stats = export_iq16_nco(grid, plan)
    return _dac_result(iq, stats, dac_config.mode)


def run_sofdm_link_load(
    config: SOFDMLoadConfig,
    *,
    diagnostics: bool = True,
) -> SOFDMLinkResult:
    """Run TX -> channel -> DAC -> ADC -> RX and collect correctness/speed metrics."""
    validate_sofdm_link_load(config)

    tx_ns, tx = _time_ns(
        lambda: build_sofdm_tx_frame(
            config.tx,
            collect_reference_symbols=diagnostics,
        )
    )
    channel_ns, channel_grid = _time_ns(
        lambda: apply_known_channel(tx.grid, config.channel_h, active_subcarriers=config.tx.active_subcarriers)
    )
    dac_ns, dac = _time_ns(
        lambda: simulate_iq_dac(
            channel_grid,
            ncp=config.tx.ncp,
            config=config.dac,
            validate_grid=False,
        )
    )
    adc_ns, adc = _time_ns(lambda: simulate_iq_adc(dac.iq, config.adc))
    rx_cfg = OFDMRXPlan(
        n_fft=config.tx.n_fft,
        ncp=config.tx.ncp,
        modulation=config.tx.modulation,
        active_subcarriers=config.tx.active_subcarriers,
        channel_policy=ChannelPolicy.KNOWN,
        channel_h=config.channel_h,
    )
    rx_ns, rx = _time_ns(
        lambda: receive_sofdm_frame(
            adc.samples,
            rx_cfg,
            tx_bits=tx.bits,
            reference_symbols=tx.active_symbols if diagnostics else None,
            collect_equalized_symbols=diagnostics,
            collect_evm=diagnostics,
        )
    )
    if diagnostics:
        baseline = _time_ns(lambda: run_basic_ofdm_baseline(config, tx.bits))[1]
    else:
        baseline = _empty_baseline_result(tx.bits)
    candidate_ns = tx_ns + channel_ns + dac_ns + adc_ns + rx_ns
    metrics = _metrics(
        config,
        tx,
        dac,
        adc,
        rx,
        tx_ns,
        channel_ns,
        dac_ns,
        adc_ns,
        rx_ns,
        candidate_ns,
        baseline,
        diagnostics_enabled=diagnostics,
        diagnostic_total_ns=candidate_ns if diagnostics else None,
    )
    return SOFDMLinkResult(
        config=config,
        tx=tx,
        channel_grid=channel_grid,
        dac=dac,
        adc=adc,
        receiver=rx,
        baseline=baseline,
        metrics=metrics,
    )


def run_basic_ofdm_baseline(
    config: SOFDMLoadConfig,
    bits: npt.ArrayLike,
) -> BaselineOFDMResult:
    """Run ordinary complex OFDM TX -> DAC -> ADC -> RX for comparison."""
    validate_sofdm_link_load(config)
    tx_bits = np.ascontiguousarray(np.asarray(bits), dtype=np.uint8).ravel()
    n_fft = config.tx.n_fft
    n_symbols = config.tx.n_symbols
    active = config.tx.active_subcarriers
    bits_per_symbol = _bits_per_symbol(config.tx.modulation)
    bits_per_ofdm_symbol = int(active.size * bits_per_symbol)
    if tx_bits.size != n_symbols * bits_per_ofdm_symbol:
        raise ValueError("baseline bits length does not match load dimensions")

    total0 = time.perf_counter_ns()
    tx0 = time.perf_counter_ns()
    tx_freq = np.zeros((n_symbols, n_fft), dtype=np.complex128)
    tx_symbols = np.empty((n_symbols, active.size), dtype=np.complex128)
    for symbol in range(n_symbols):
        start = symbol * bits_per_ofdm_symbol
        stop = start + bits_per_ofdm_symbol
        symbol_values = _modulate_fp(tx_bits[start:stop], config.tx.modulation)
        tx_freq[symbol, active] = symbol_values
        tx_symbols[symbol] = symbol_values
    tx_ns = time.perf_counter_ns() - tx0

    channel0 = time.perf_counter_ns()
    channel_freq = tx_freq * config.channel_h.reshape(1, n_fft)
    channel_ns = time.perf_counter_ns() - channel0

    dac0 = time.perf_counter_ns()
    useful = np.fft.ifft(channel_freq, axis=1)
    with_cp = np.concatenate([useful[:, -config.tx.ncp :], useful], axis=1) if config.tx.ncp else useful
    # Baseline DAC must match the candidate SynthPlan IQ16 quantizer law so
    # candidate-vs-baseline differences reflect OFDM path differences, not DAC
    # quantizer implementation differences.
    baseline_iq16, baseline_clips = _quantize_iq16_synth_law(with_cp.reshape(-1))
    baseline_samples = _iq16_to_complex_unit(baseline_iq16)
    baseline_dac = ADCResult(
        samples=baseline_samples,
        quantized_iq=baseline_iq16,
        bit_depth=16,
        full_scale=1.0,
        clipped_count=baseline_clips,
        sample_count=int(baseline_samples.size),
        stages=("iq16_synth_quantize",),
        clipped_fraction=(float(baseline_clips) / float(2 * baseline_samples.size) if baseline_samples.size else 0.0),
    )
    dac_ns = time.perf_counter_ns() - dac0

    adc0 = time.perf_counter_ns()
    baseline_adc = simulate_iq_adc(baseline_dac.samples, config.adc)
    adc_ns = time.perf_counter_ns() - adc0

    rx0 = time.perf_counter_ns()
    rx_symbols_cp = baseline_adc.samples.reshape(n_symbols, n_fft + config.tx.ncp)
    rx_freq = np.fft.fft(rx_symbols_cp[:, config.tx.ncp :], axis=1)
    equalized = rx_freq[:, active] / config.channel_h[active].reshape(1, active.size)
    agc = _baseline_agc(equalized, config.tx.modulation)
    equalized = equalized * agc
    rx_bits = _demodulate_fp(equalized.reshape(-1), config.tx.modulation)
    rx_ns = time.perf_counter_ns() - rx0
    total_ns = time.perf_counter_ns() - total0

    metrics = _baseline_metrics(
        config,
        tx_bits,
        rx_bits,
        tx_symbols.reshape(-1),
        equalized.reshape(-1),
        baseline_dac,
        baseline_adc,
        tx_ns,
        channel_ns,
        dac_ns,
        adc_ns,
        rx_ns,
        total_ns,
    )
    return BaselineOFDMResult(
        tx_bits=tx_bits,
        rx_bits=rx_bits,
        tx_symbols=tx_symbols.reshape(-1),
        equalized_symbols=equalized.reshape(-1),
        dac=baseline_dac,
        adc=baseline_adc,
        metrics=metrics,
    )


def run_cross_link_matrix(
    config: SOFDMLoadConfig,
    *,
    diagnostics: bool = True,
) -> CrossLinkMatrix:
    """Run TX-fp/TX-spf × RX-fp/RX-spf with shared bits, channel, DAC/ADC config.

    Both TX backends drive the same bit stream (sourced from ``build_sofdm_tx_frame``)
    so that cross-cell ``rx_bits`` equality is meaningful. Each TX backend emits its
    own post-ADC IQ stream; both RX backends are run on each stream, producing four
    cells plus an identity check (same TX IQ ⇒ same rx_bits across RX backends).
    """
    validate_sofdm_link_load(config)

    s_tx_ns, sofdm_tx = _time_ns(
        lambda: build_sofdm_tx_frame(
            config.tx,
            collect_reference_symbols=diagnostics,
        )
    )
    s_channel_ns, sofdm_channel_grid = _time_ns(
        lambda: apply_known_channel(sofdm_tx.grid, config.channel_h, active_subcarriers=config.tx.active_subcarriers)
    )
    s_dac_ns, sofdm_dac = _time_ns(
        lambda: simulate_iq_dac(
            sofdm_channel_grid,
            ncp=config.tx.ncp,
            config=config.dac,
            validate_grid=False,
        )
    )
    s_adc_ns, sofdm_adc = _time_ns(lambda: simulate_iq_adc(sofdm_dac.iq, config.adc))

    fp_tx_ns, (fp_tx_symbols, fp_tx_freq) = _time_ns(
        lambda: _fp_tx_synth(config, sofdm_tx.bits, collect_reference_symbols=diagnostics)
    )
    fp_channel_ns, fp_channel_freq = _time_ns(lambda: fp_tx_freq * config.channel_h.reshape(1, config.tx.n_fft))
    fp_dac_ns, fp_dac_samples = _time_ns(lambda: _fp_dac_iq(config, fp_channel_freq))
    fp_adc_ns, fp_adc = _time_ns(lambda: simulate_iq_adc(fp_dac_samples, config.adc))

    plan = OFDMRXPlan(
        n_fft=config.tx.n_fft,
        ncp=config.tx.ncp,
        modulation=config.tx.modulation,
        active_subcarriers=config.tx.active_subcarriers,
        channel_policy=ChannelPolicy.KNOWN,
        channel_h=config.channel_h,
    )

    fp_tx_fp_rx = _cell_fp_rx(
        tx_kind="fp",
        config=config,
        samples=fp_adc.samples,
        tx_bits=sofdm_tx.bits,
        tx_symbols=fp_tx_symbols,
        tx_ns=fp_tx_ns,
        channel_ns=fp_channel_ns,
        dac_ns=fp_dac_ns,
        adc_ns=fp_adc_ns,
        diagnostics=diagnostics,
    )
    fp_tx_sofdm_rx = _cell_sofdm_rx(
        tx_kind="fp",
        config=config,
        plan=plan,
        samples=fp_adc.samples,
        tx_bits=sofdm_tx.bits,
        tx_symbols=fp_tx_symbols,
        tx_ns=fp_tx_ns,
        channel_ns=fp_channel_ns,
        dac_ns=fp_dac_ns,
        adc_ns=fp_adc_ns,
        diagnostics=diagnostics,
    )
    sofdm_tx_fp_rx = _cell_fp_rx(
        tx_kind="sofdm",
        config=config,
        samples=sofdm_adc.samples,
        tx_bits=sofdm_tx.bits,
        tx_symbols=sofdm_tx.active_symbols,
        tx_ns=s_tx_ns,
        channel_ns=s_channel_ns,
        dac_ns=s_dac_ns,
        adc_ns=s_adc_ns,
        diagnostics=diagnostics,
    )
    sofdm_tx_sofdm_rx = _cell_sofdm_rx(
        tx_kind="sofdm",
        config=config,
        plan=plan,
        samples=sofdm_adc.samples,
        tx_bits=sofdm_tx.bits,
        tx_symbols=sofdm_tx.active_symbols,
        tx_ns=s_tx_ns,
        channel_ns=s_channel_ns,
        dac_ns=s_dac_ns,
        adc_ns=s_adc_ns,
        diagnostics=diagnostics,
    )

    identity = CrossLinkIdentity(
        fp_tx_rx_bits_match=bool(np.array_equal(fp_tx_fp_rx.rx_bits, fp_tx_sofdm_rx.rx_bits)),
        fp_tx_rx_bit_diff=int(np.count_nonzero(fp_tx_fp_rx.rx_bits != fp_tx_sofdm_rx.rx_bits)),
        sofdm_tx_rx_bits_match=bool(np.array_equal(sofdm_tx_fp_rx.rx_bits, sofdm_tx_sofdm_rx.rx_bits)),
        sofdm_tx_rx_bit_diff=int(np.count_nonzero(sofdm_tx_fp_rx.rx_bits != sofdm_tx_sofdm_rx.rx_bits)),
    )

    return CrossLinkMatrix(
        config=config,
        tx_bits=sofdm_tx.bits,
        fp_post_adc=np.ascontiguousarray(fp_adc.samples, dtype=np.complex128),
        sofdm_post_adc=np.ascontiguousarray(sofdm_adc.samples, dtype=np.complex128),
        fp_tx_fp_rx=fp_tx_fp_rx,
        fp_tx_sofdm_rx=fp_tx_sofdm_rx,
        sofdm_tx_fp_rx=sofdm_tx_fp_rx,
        sofdm_tx_sofdm_rx=sofdm_tx_sofdm_rx,
        identity=identity,
    )


def write_cross_link_metrics(result: CrossLinkMatrix, path: str | Path) -> Path:
    """Write cross-matrix metrics JSON and return the output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def _fp_tx_synth(
    config: SOFDMLoadConfig,
    bits: npt.NDArray[np.uint8],
    *,
    collect_reference_symbols: bool = True,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """Build per-active-bin fp symbols and a zero-padded freq-domain grid."""
    n_fft = config.tx.n_fft
    n_symbols = config.tx.n_symbols
    active = config.tx.active_subcarriers
    bits_per_symbol = _bits_per_symbol(config.tx.modulation)
    bits_per_ofdm_symbol = int(active.size * bits_per_symbol)
    if bits.size != n_symbols * bits_per_ofdm_symbol:
        raise ValueError("bits length does not match TX configuration")
    tx_freq = np.zeros((n_symbols, n_fft), dtype=np.complex128)
    tx_symbols: npt.NDArray[np.complex128] | None = (
        np.empty((n_symbols, active.size), dtype=np.complex128)
        if collect_reference_symbols
        else None
    )
    for symbol in range(n_symbols):
        start = symbol * bits_per_ofdm_symbol
        stop = start + bits_per_ofdm_symbol
        symbol_values = _modulate_fp(bits[start:stop], config.tx.modulation)
        tx_freq[symbol, active] = symbol_values
        if tx_symbols is not None:
            tx_symbols[symbol] = symbol_values
    symbols = (
        tx_symbols.reshape(-1) if tx_symbols is not None else np.empty(0, dtype=np.complex128)
    )
    return symbols, tx_freq


def _fp_dac_iq(
    config: SOFDMLoadConfig,
    channel_freq: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Apply IFFT + CP + IQ16 synth quantization like ``run_basic_ofdm_baseline``."""
    if config.dac.mode != "ifft":
        raise ValueError("fp TX cross-link path only supports DAC mode 'ifft'")
    useful = np.fft.ifft(channel_freq, axis=1)
    with_cp = (
        np.concatenate([useful[:, -config.tx.ncp :], useful], axis=1)
        if config.tx.ncp
        else useful
    )
    iq16, _clips = _quantize_iq16_synth_law(with_cp.reshape(-1))
    return _iq16_to_complex_unit(iq16)


def _cell_fp_rx(
    *,
    tx_kind: TXKind,
    config: SOFDMLoadConfig,
    samples: npt.NDArray[np.complex128],
    tx_bits: npt.NDArray[np.uint8],
    tx_symbols: npt.NDArray[np.complex128],
    tx_ns: int,
    channel_ns: int,
    dac_ns: int,
    adc_ns: int,
    diagnostics: bool,
) -> CrossLinkCell:
    n_fft = config.tx.n_fft
    n_symbols = config.tx.n_symbols
    active = config.tx.active_subcarriers
    rx0 = time.perf_counter_ns()
    rx_symbols_cp = samples.reshape(n_symbols, n_fft + config.tx.ncp)
    rx_freq = fft_rows(rx_symbols_cp[:, config.tx.ncp :])  # numpy / KV260 FFT64 overlay toggle
    equalized = rx_freq[:, active] / config.channel_h[active].reshape(1, active.size)
    agc = _baseline_agc(equalized, config.tx.modulation)
    equalized = equalized * agc
    rx_bits = _demodulate_fp(equalized.reshape(-1), config.tx.modulation)
    rx_ns = time.perf_counter_ns() - rx0
    return _build_cell(
        tx_kind=tx_kind,
        rx_kind="fp",
        tx_bits=tx_bits,
        rx_bits=rx_bits,
        tx_symbols=tx_symbols,
        equalized_symbols=equalized.reshape(-1) if diagnostics else np.empty(0, dtype=np.complex128),
        tx_ns=tx_ns,
        channel_ns=channel_ns,
        dac_ns=dac_ns,
        adc_ns=adc_ns,
        rx_ns=rx_ns,
    )


def _cell_sofdm_rx(
    *,
    tx_kind: TXKind,
    config: SOFDMLoadConfig,
    plan: OFDMRXPlan,
    samples: npt.NDArray[np.complex128],
    tx_bits: npt.NDArray[np.uint8],
    tx_symbols: npt.NDArray[np.complex128],
    tx_ns: int,
    channel_ns: int,
    dac_ns: int,
    adc_ns: int,
    diagnostics: bool,
) -> CrossLinkCell:
    rx_ns, rx = _time_ns(
        lambda: receive_sofdm_frame(
            samples,
            plan,
            tx_bits=tx_bits,
            reference_symbols=tx_symbols if diagnostics else None,
            collect_equalized_symbols=diagnostics,
            collect_evm=diagnostics,
        )
    )
    return _build_cell(
        tx_kind=tx_kind,
        rx_kind="sofdm",
        tx_bits=tx_bits,
        rx_bits=rx.rx_bits,
        tx_symbols=tx_symbols,
        equalized_symbols=rx.equalized_symbols,
        tx_ns=tx_ns,
        channel_ns=channel_ns,
        dac_ns=dac_ns,
        adc_ns=adc_ns,
        rx_ns=rx_ns,
    )


def _build_cell(
    *,
    tx_kind: TXKind,
    rx_kind: RXKind,
    tx_bits: npt.NDArray[np.uint8],
    rx_bits: npt.NDArray[np.uint8],
    tx_symbols: npt.NDArray[np.complex128],
    equalized_symbols: npt.NDArray[np.complex128],
    tx_ns: int,
    channel_ns: int,
    dac_ns: int,
    adc_ns: int,
    rx_ns: int,
) -> CrossLinkCell:
    bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
    bits_total = int(tx_bits.size)
    evm_value = float(evm(tx_symbols, equalized_symbols)) if tx_symbols.size else None
    evm_db_value = _evm_db(tx_symbols, equalized_symbols) if tx_symbols.size else None
    snr_db = _snr_db_from_evm(evm_value)
    total_ns = tx_ns + channel_ns + dac_ns + adc_ns + rx_ns
    return CrossLinkCell(
        tx_kind=tx_kind,
        rx_kind=rx_kind,
        rx_bits=np.ascontiguousarray(rx_bits, dtype=np.uint8),
        equalized_symbols=np.ascontiguousarray(equalized_symbols, dtype=np.complex128),
        bit_errors=bit_errors,
        bits_total=bits_total,
        ber=float(bit_errors / bits_total) if bits_total else 0.0,
        evm=evm_value,
        evm_db=evm_db_value,
        snr_db_estimate=snr_db,
        tx_ns=tx_ns,
        channel_ns=channel_ns,
        dac_ns=dac_ns,
        adc_ns=adc_ns,
        rx_ns=rx_ns,
        total_ns=total_ns,
    )


def _cell_to_metrics(cell: CrossLinkCell) -> MetricsPayload:
    total_summary = _timing_summary_from_samples([cell.total_ns], warmup_iterations=0)
    return {
        "tx_kind": cell.tx_kind,
        "rx_kind": cell.rx_kind,
        "bit_exact": cell.bit_errors == 0,
        "bit_errors": cell.bit_errors,
        "bits_total": cell.bits_total,
        "ber": cell.ber,
        "evm": cell.evm,
        "evm_db": cell.evm_db,
        "snr_db_estimate": cell.snr_db_estimate,
        "tx_ns": cell.tx_ns,
        "channel_ns": cell.channel_ns,
        "dac_ns": cell.dac_ns,
        "adc_ns": cell.adc_ns,
        "rx_ns": cell.rx_ns,
        "total_ns": cell.total_ns,
        "hot_path_total_ns": cell.total_ns,
        "diagnostic_total_ns": None if cell.evm is None else cell.total_ns,
        "diagnostics_enabled": cell.evm is not None,
        "timing_summary": total_summary,
        "stage_summary": {
            "tx": _timing_summary_from_samples([cell.tx_ns], warmup_iterations=0),
            "channel": _timing_summary_from_samples([cell.channel_ns], warmup_iterations=0),
            "dac": _timing_summary_from_samples([cell.dac_ns], warmup_iterations=0),
            "adc": _timing_summary_from_samples([cell.adc_ns], warmup_iterations=0),
            "rx": _timing_summary_from_samples([cell.rx_ns], warmup_iterations=0),
            "total": total_summary,
        },
    }


def load_sofdm_link_config(path: str | Path) -> SOFDMLoadConfig:
    """Load and validate a JSON sOFDM link fixture."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("sOFDM link fixture must be a JSON object")
    config = sofdm_link_config_from_mapping(raw)
    validate_sofdm_link_load(config)
    return config


def sofdm_link_config_from_mapping(data: Mapping[str, object]) -> SOFDMLoadConfig:
    """Build a link config from a decoded fixture mapping."""
    scenario = _required_str(data, "scenario")
    n_fft = _required_int(data, "n_fft")
    n_symbols = _required_int(data, "n_symbols")
    ncp = _required_int(data, "ncp")
    modulation = _modulation(_required_str(data, "modulation"))
    seed = _optional_int(data, "seed", 20260521)
    active = _active_from_mapping(data, n_fft)
    channel_h = _channel_from_mapping(data, n_fft)
    dac = _dac_from_mapping(_optional_mapping(data, "dac"))
    adc = _adc_from_mapping(_optional_mapping(data, "adc"))
    acceptance = _optional_mapping(data, "acceptance")
    sample_rate_hz = _optional_float(data, "sample_rate_hz", 1_000_000.0)
    return SOFDMLoadConfig(
        scenario=scenario,
        tx=OFDMTXConfig(
            n_fft=n_fft,
            n_symbols=n_symbols,
            ncp=ncp,
            modulation=modulation,
            active_subcarriers=active,
            seed=seed,
        ),
        channel_h=channel_h,
        dac=dac,
        adc=adc,
        sample_rate_hz=sample_rate_hz,
        evm_max=0.08 if acceptance is None else _optional_float(acceptance, "evm_max", 0.08),
        require_bit_exact=_optional_bool(acceptance, "require_bit_exact", True),
    )


def validate_sofdm_link_load(config: SOFDMLoadConfig) -> None:
    """Validate cross-field constraints for a link load."""
    if not config.scenario:
        raise ValueError("scenario must not be empty")
    _validate_tx_config(config.tx)
    _channel(config.tx.n_fft, config.channel_h)
    if config.dac.mode not in ("ifft", "nco"):
        raise ValueError(f"unsupported DAC mode {config.dac.mode!r}")
    if config.adc.resample_up != 1 or config.adc.resample_down != 1:
        raise ValueError("ADC resampling is unsupported by fixed-size OFDM link loads")
    if config.sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive")
    if config.evm_max <= 0.0:
        raise ValueError("evm_max must be positive")


def write_sofdm_link_metrics(result: SOFDMLinkResult, path: str | Path) -> Path:
    """Write link metrics JSON and return the output path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result.metrics.to_json_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out


def _time_ns(fn: Callable[[], T]) -> tuple[int, T]:
    start = time.perf_counter_ns()
    result = fn()
    return time.perf_counter_ns() - start, result


def _metrics(
    config: SOFDMLoadConfig,
    tx: SOFDMTXFrame,
    dac: DACResult,
    adc: ADCResult,
    rx: SOFDMReceiverResult,
    tx_ns: int,
    channel_ns: int,
    dac_ns: int,
    adc_ns: int,
    rx_ns: int,
    candidate_ns: int,
    baseline: BaselineOFDMResult,
    *,
    diagnostics_enabled: bool,
    diagnostic_total_ns: int | None,
) -> SOFDMLinkMetrics:
    bit_errors = int(np.count_nonzero(tx.bits != rx.rx_bits))
    bits_total = int(tx.bits.size)
    bit_exact = bit_errors == 0
    active_bins_total = int(config.tx.n_symbols * config.tx.active_subcarriers.size)
    confidence_min = None if rx.confidence is None else int(np.min(rx.confidence))
    confidence_mean = None if rx.confidence is None else float(np.mean(rx.confidence))
    evm_value = None if rx.evm is None else float(rx.evm)
    evm_db_value = (
        None
        if evm_value is None or tx.active_symbols.size == 0 or rx.equalized_symbols.size == 0
        else _evm_db(tx.active_symbols, rx.equalized_symbols)
    )
    return SOFDMLinkMetrics(
        load={
            "scenario": config.scenario,
            "n_fft": config.tx.n_fft,
            "n_symbols": config.tx.n_symbols,
            "ncp": config.tx.ncp,
            "modulation": config.tx.modulation,
            "sample_rate_hz": config.sample_rate_hz,
            "subcarrier_spacing_hz": _subcarrier_spacing_hz(config),
            "occupied_active_bandwidth_hz": _occupied_active_bandwidth_hz(config),
            "active_bins_per_symbol": int(config.tx.active_subcarriers.size),
            "active_bins_total": active_bins_total,
            "active_tone_utilization": _active_tone_utilization(config),
            "seed": config.tx.seed,
        },
        correctness={
            "bit_exact": bit_exact,
            "bit_errors": bit_errors,
            "bits_total": bits_total,
            "ber": float(bit_errors / bits_total) if bits_total else 0.0,
            "evm": evm_value,
            "evm_db": evm_db_value,
            "snr_db_estimate": _snr_db_from_evm(evm_value),
            "evm_max": config.evm_max,
            "passes_evm": (
                bool(evm_value is not None and evm_value <= config.evm_max)
                if diagnostics_enabled
                else None
            ),
            "passes_acceptance": (
                bool(
                    (bit_exact or not config.require_bit_exact)
                    and evm_value is not None
                    and evm_value <= config.evm_max
                )
                if diagnostics_enabled
                else bool(bit_exact or not config.require_bit_exact)
            ),
        },
        speed={
            "tx_ns": tx_ns,
            "channel_ns": channel_ns,
            "dac_ns": dac_ns,
            "adc_ns": adc_ns,
            "rx_ns": rx_ns,
            "total_ns": candidate_ns,
            "hot_path_total_ns": candidate_ns,
            "diagnostic_total_ns": diagnostic_total_ns,
            "diagnostics_enabled": diagnostics_enabled,
            "ns_per_ofdm_symbol": _ratio(candidate_ns, config.tx.n_symbols),
            "ns_per_active_bin": _ratio(candidate_ns, active_bins_total),
            "ns_per_bit": _ratio(candidate_ns, bits_total),
            "symbols_per_s": _rate(config.tx.n_symbols, candidate_ns),
            "bits_per_s": _rate(bits_total, candidate_ns),
            "mbits_per_s": _mbits_per_s(bits_total, candidate_ns),
            "effective_mbits_per_s": _effective_mbits_per_s(
                bits_total,
                dac.sample_count,
                config.sample_rate_hz,
            ),
            "payload_bits_per_sample": _ratio(bits_total, dac.sample_count),
            "spectral_efficiency_bits_per_s_hz": _spectral_efficiency_bits_per_s_hz(
                config,
                bits_total,
                dac.sample_count,
            ),
            "stage_percent": _stage_percent(
                tx_ns=tx_ns,
                channel_ns=channel_ns,
                dac_ns=dac_ns,
                adc_ns=adc_ns,
                rx_ns=rx_ns,
                total_ns=candidate_ns,
            ),
            "stage_summary": {
                "tx": _timing_summary_from_samples([tx_ns], warmup_iterations=0),
                "channel": _timing_summary_from_samples([channel_ns], warmup_iterations=0),
                "dac": _timing_summary_from_samples([dac_ns], warmup_iterations=0),
                "adc": _timing_summary_from_samples([adc_ns], warmup_iterations=0),
                "rx": _timing_summary_from_samples([rx_ns], warmup_iterations=0),
                "total": _timing_summary_from_samples([candidate_ns], warmup_iterations=0),
            },
        },
        dac={
            "mode": dac.mode,
            "output_format": dac.output_format,
            "iq16_samples": dac.sample_count,
            "papr": dac.papr,
            "clip_count": dac.clip_count,
            "clipping_ratio": _iq_clip_ratio(dac.clip_count, dac.sample_count),
            "active_tones": dac.active_tones,
            "ifft_calls": dac.ifft_calls,
            "nco_tones": dac.nco_tones,
        },
        adc={
            "bit_depth": adc.bit_depth,
            "full_scale": adc.full_scale,
            "clipped_count": adc.clipped_count,
            "clipped_fraction": adc.clipped_fraction,
            "clipping_ratio": _iq_clip_ratio(adc.clipped_count, adc.sample_count),
            "sample_count": adc.sample_count,
            "fir_taps_used": adc.fir_taps_used,
            "awgn_snr_db_estimated": adc.awgn_snr_db_estimated,
            "post_quantize_evm_db": adc.post_quantize_evm_db,
            "stages": ",".join(adc.stages),
            "stage_timings_us": ",".join(f"{v:.3f}" for v in adc.stage_timings_us),
        },
        rx={
            "demap": rx.demap,
            "agc_gain": rx.agc_gain,
            "confidence_min": confidence_min,
            "confidence_mean": confidence_mean,
            "stages": ",".join(rx.stages),
        },
        baseline=baseline.metrics,
        comparison=_comparison_metrics(
            candidate_bit_errors=bit_errors,
            candidate_evm=evm_value,
            candidate_total_ns=candidate_ns,
            baseline=baseline,
        ),
    )


def _empty_baseline_result(tx_bits: npt.NDArray[np.uint8]) -> BaselineOFDMResult:
    empty_complex = np.empty(0, dtype=np.complex128)
    empty_iq = np.empty(0, dtype=np.int16)
    empty_adc = ADCResult(
        samples=empty_complex,
        quantized_iq=empty_iq,
        bit_depth=0,
        full_scale=0.0,
        clipped_count=0,
        sample_count=0,
        stages=(),
    )
    return BaselineOFDMResult(
        tx_bits=np.ascontiguousarray(tx_bits, dtype=np.uint8),
        rx_bits=np.empty(0, dtype=np.uint8),
        tx_symbols=empty_complex,
        equalized_symbols=empty_complex,
        dac=empty_adc,
        adc=empty_adc,
        metrics={},
    )


def _baseline_metrics(
    config: SOFDMLoadConfig,
    tx_bits: npt.NDArray[np.uint8],
    rx_bits: npt.NDArray[np.uint8],
    tx_symbols: npt.NDArray[np.complex128],
    equalized_symbols: npt.NDArray[np.complex128],
    dac: ADCResult,
    adc: ADCResult,
    tx_ns: int,
    channel_ns: int,
    dac_ns: int,
    adc_ns: int,
    rx_ns: int,
    total_ns: int,
) -> MetricsPayload:
    bit_errors = int(np.count_nonzero(tx_bits != rx_bits))
    bits_total = int(tx_bits.size)
    active_bins_total = int(config.tx.n_symbols * config.tx.active_subcarriers.size)
    evm_value = float(evm(tx_symbols, equalized_symbols))
    evm_db_value = _evm_db(tx_symbols, equalized_symbols)
    return {
        "correctness": {
            "bit_exact": bit_errors == 0,
            "bit_errors": bit_errors,
            "bits_total": bits_total,
            "ber": float(bit_errors / bits_total) if bits_total else 0.0,
            "evm": evm_value,
            "evm_db": evm_db_value,
            "snr_db_estimate": _snr_db_from_evm(evm_value),
            "passes_evm": evm_value <= config.evm_max,
        },
        "speed": {
            "tx_ns": tx_ns,
            "channel_ns": channel_ns,
            "dac_ns": dac_ns,
            "adc_ns": adc_ns,
            "rx_ns": rx_ns,
            "total_ns": total_ns,
            "ns_per_ofdm_symbol": _ratio(total_ns, config.tx.n_symbols),
            "ns_per_active_bin": _ratio(total_ns, active_bins_total),
            "ns_per_bit": _ratio(total_ns, bits_total),
            "symbols_per_s": _rate(config.tx.n_symbols, total_ns),
            "bits_per_s": _rate(bits_total, total_ns),
            "mbits_per_s": _mbits_per_s(bits_total, total_ns),
            "effective_mbits_per_s": _effective_mbits_per_s(
                bits_total,
                dac.sample_count,
                config.sample_rate_hz,
            ),
            "payload_bits_per_sample": _ratio(bits_total, dac.sample_count),
            "spectral_efficiency_bits_per_s_hz": _spectral_efficiency_bits_per_s_hz(
                config,
                bits_total,
                dac.sample_count,
            ),
            "stage_percent": _stage_percent(
                tx_ns=tx_ns,
                channel_ns=channel_ns,
                dac_ns=dac_ns,
                adc_ns=adc_ns,
                rx_ns=rx_ns,
                total_ns=total_ns,
            ),
        },
        "dac": {
            "mode": "basic_ifft",
            "output_format": "IQ_INT16",
            "sample_count": dac.sample_count,
            "clipped_count": dac.clipped_count,
            "clipping_ratio": _iq_clip_ratio(dac.clipped_count, dac.sample_count),
        },
        "adc": {
            "bit_depth": adc.bit_depth,
            "full_scale": adc.full_scale,
            "sample_count": adc.sample_count,
            "clipped_count": adc.clipped_count,
            "clipping_ratio": _iq_clip_ratio(adc.clipped_count, adc.sample_count),
            "stages": ",".join(adc.stages),
        },
        "rx": {
            "demap": f"{config.tx.modulation}_fp_hard",
            "stages": "cp_remove,numpy_fft,complex_zf_equalize,fp_hard_demap",
        },
    }


def _comparison_metrics(
    *,
    candidate_bit_errors: int,
    candidate_evm: float | None,
    candidate_total_ns: int,
    baseline: BaselineOFDMResult,
) -> dict[str, MetricValue]:
    if not baseline.metrics:
        return {
            "candidate": "sofdm",
            "baseline": "basic_ofdm_fp",
            "baseline_enabled": False,
            "both_bit_exact": None,
            "bit_error_delta": None,
            "evm_delta": None,
            "total_ns_delta": None,
            "candidate_to_baseline_total_ns_ratio": None,
        }
    base_correctness = cast(dict[str, MetricValue], baseline.metrics["correctness"])
    base_speed = cast(dict[str, MetricValue], baseline.metrics["speed"])
    baseline_total_ns = int(base_speed["total_ns"] or 0)
    baseline_evm = float(base_correctness["evm"] or 0.0)
    return {
        "candidate": "sofdm",
        "baseline": "basic_ofdm_fp",
        "both_bit_exact": bool(candidate_bit_errors == 0 and bool(base_correctness["bit_exact"])),
        "bit_error_delta": int(candidate_bit_errors - int(base_correctness["bit_errors"] or 0)),
        "evm_delta": None if candidate_evm is None else float(candidate_evm - baseline_evm),
        "total_ns_delta": int(candidate_total_ns - baseline_total_ns),
        "candidate_to_baseline_total_ns_ratio": _ratio(candidate_total_ns, baseline_total_ns),
    }


def _dac_result(
    iq: npt.NDArray[np.int16],
    stats: SynthStats,
    mode: DacMode,
) -> DACResult:
    return DACResult(
        iq=np.ascontiguousarray(iq, dtype=np.int16),
        mode=mode,
        output_format="IQ_INT16",
        sample_count=int(iq.size // 2),
        papr=float(stats.papr),
        clip_count=int(stats.clip_count),
        active_tones=int(stats.active_tones),
        ifft_calls=int(stats.ifft_calls),
        nco_tones=int(stats.nco_tones),
    )


def _quantize_iq16_synth_law(
    samples: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.int16], int]:
    arr = np.ascontiguousarray(np.asarray(samples, dtype=np.complex128)).reshape(-1)
    if arr.size == 0:
        return np.empty(0, dtype=np.int16), 0
    peak = float(max(np.max(np.abs(arr.real)), np.max(np.abs(arr.imag)), 1e-30))
    scale = 30000.0 / peak
    q = np.empty((arr.size, 2), dtype=np.float64)
    q[:, 0] = np.rint(arr.real * scale)
    q[:, 1] = np.rint(arr.imag * scale)
    clipped = np.logical_or(q > 32767.0, q < -32768.0)
    clips = int(np.count_nonzero(clipped))
    q_i16 = np.clip(q, -32768.0, 32767.0).astype(np.int16)
    return q_i16.reshape(-1), clips


def _iq16_to_complex_unit(iq16: npt.NDArray[np.int16]) -> npt.NDArray[np.complex128]:
    flat = np.ascontiguousarray(np.asarray(iq16, dtype=np.int16)).reshape(-1)
    if flat.size % 2:
        raise ValueError("interleaved IQ16 array must have an even length")
    pairs = flat.astype(np.float64).reshape(-1, 2) / 32767.0
    return np.ascontiguousarray(pairs[:, 0] + 1j * pairs[:, 1], dtype=np.complex128)


def _validate_tx_config(config: OFDMTXConfig) -> None:
    if config.n_fft < 4 or config.n_fft & (config.n_fft - 1):
        raise ValueError("n_fft must be a power of two >= 4")
    if config.n_symbols <= 0:
        raise ValueError("n_symbols must be positive")
    if config.ncp < 0 or config.ncp > config.n_fft:
        raise ValueError("ncp must be in [0, n_fft]")
    active = config.active_subcarriers
    if active.size == 0:
        raise ValueError("active_subcarriers must not be empty")
    if np.any(active < 0) or np.any(active == 0) or np.any(active >= config.n_fft):
        raise ValueError("active_subcarriers must be non-DC bins inside n_fft")
    if not np.all(active[1:] > active[:-1]):
        raise ValueError("active_subcarriers must be strictly increasing")
    _bits_per_symbol(config.modulation)


def _modulate(bits: npt.NDArray[np.uint8], modulation: Modulation) -> npt.NDArray[np.uint32]:
    if modulation == "qpsk":
        return psk_modulate_spf(bits, 4)
    if modulation == "qam16":
        return qam16_modulate_spf(bits)
    raise ValueError(f"unsupported modulation {modulation!r}")


def _modulate_fp(bits: npt.NDArray[np.uint8], modulation: Modulation) -> npt.NDArray[np.complex128]:
    if modulation == "qpsk":
        return decode_words(psk_modulate_spf(bits, 4))
    if modulation == "qam16":
        return qam16_modulate_fp(bits)
    raise ValueError(f"unsupported modulation {modulation!r}")


def _demodulate_fp(
    symbols: npt.NDArray[np.complex128],
    modulation: Modulation,
) -> npt.NDArray[np.uint8]:
    if modulation == "qpsk":
        return psk_demodulate_fp(symbols, 4)
    if modulation == "qam16":
        return qam16_demodulate_fp(symbols)
    raise ValueError(f"unsupported modulation {modulation!r}")


def _baseline_agc(symbols: npt.NDArray[np.complex128], modulation: Modulation) -> float:
    observed = float(np.sqrt(np.mean(np.abs(symbols) ** 2)))
    if observed <= 0.0:
        return 1.0
    target = 1.0 if modulation == "qpsk" else np.sqrt(10.0)
    return float(target / observed)


def _bits_per_symbol(modulation: Modulation) -> int:
    if modulation == "qpsk":
        return 2
    if modulation == "qam16":
        return 4
    raise ValueError(f"unsupported modulation {modulation!r}")


def _channel(n_fft: int, channel_h: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    h = np.ascontiguousarray(np.asarray(channel_h), dtype=np.complex128).ravel()
    if h.size != n_fft:
        raise ValueError(f"channel_h length {h.size} != n_fft {n_fft}")
    if np.any(h == 0.0):
        raise ValueError("channel_h must not contain zero subcarriers")
    return h


def _active_from_mapping(data: Mapping[str, object], n_fft: int) -> npt.NDArray[np.uint32]:
    if "active_subcarriers" in data:
        return _int_array(_required_sequence(data, "active_subcarriers"), "active_subcarriers")
    active_bins = _required_int(data, "active_bins")
    active_stride = _required_int(data, "active_stride")
    if active_bins <= 0 or active_stride <= 0:
        raise ValueError("active_bins and active_stride must be positive")
    bins = [k for k in range(1, n_fft) if k % active_stride == 0]
    if active_bins > len(bins):
        raise ValueError("active_bins exceeds bins available for active_stride")
    return np.asarray(bins[:active_bins], dtype=np.uint32)


def _channel_from_mapping(
    data: Mapping[str, object],
    n_fft: int,
) -> npt.NDArray[np.complex128]:
    if "channel_h" in data:
        return _complex_array(_required_sequence(data, "channel_h"), "channel_h")
    taps_raw = _required_sequence(data, "channel_taps")
    taps = _complex_array(taps_raw, "channel_taps")
    impulse = np.zeros(n_fft, dtype=np.complex128)
    if taps.size > n_fft:
        raise ValueError("channel_taps length must not exceed n_fft")
    impulse[: taps.size] = taps
    return np.ascontiguousarray(np.fft.fft(impulse), dtype=np.complex128)


def _dac_from_mapping(data: Mapping[str, object] | None) -> DACConfig:
    if data is None:
        return DACConfig()
    return DACConfig(
        mode=_dac_mode(str(data.get("mode", "ifft"))),
        sparse_threshold=_optional_int(data, "sparse_threshold", 96),
    )


def _adc_from_mapping(data: Mapping[str, object] | None) -> ADCConfig:
    if data is None:
        return ADCConfig()
    return ADCConfig(
        bit_depth=_optional_int(data, "bit_depth", 12),
        full_scale=_optional_float(data, "full_scale", 1.0),
        snr_db=_optional_float_or_none(data, "snr_db"),
        resample_up=_optional_int(data, "resample_up", 1),
        resample_down=_optional_int(data, "resample_down", 1),
        fir_taps=tuple(float(v) for v in _optional_float_sequence(data, "fir_taps")),
        seed=_optional_int(data, "seed", 20260521),
    )


def _modulation(value: str) -> Modulation:
    if value in {"qpsk", "qam16"}:
        return cast(Modulation, value)
    raise ValueError(f"unsupported modulation {value!r}")


def _dac_mode(value: str) -> DacMode:
    if value in {"ifft", "nco"}:
        return cast(DacMode, value)
    raise ValueError(f"unsupported DAC mode {value!r}")


def _required_str(data: Mapping[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _required_int(data: Mapping[str, object], key: str) -> int:
    value = data.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_int(data: Mapping[str, object], key: str, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def _optional_float(data: Mapping[str, object], key: str, default: float) -> float:
    value = data.get(key, default)
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric")
    return float(value)


def _optional_float_or_none(data: Mapping[str, object], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"{key} must be numeric or null")
    return float(value)


def _optional_bool(data: Mapping[str, object] | None, key: str, default: bool) -> bool:
    if data is None:
        return default
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be boolean")
    return value


def _optional_mapping(data: Mapping[str, object], key: str) -> Mapping[str, object] | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _required_sequence(data: Mapping[str, object], key: str) -> Sequence[object]:
    value = data.get(key)
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{key} must be an array")
    return value


def _optional_float_sequence(data: Mapping[str, object], key: str) -> Sequence[float]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{key} must be an array")
    out: list[float] = []
    for item in value:
        if not isinstance(item, int | float):
            raise ValueError(f"{key} entries must be numeric")
        out.append(float(item))
    return out


def _int_array(values: Sequence[object], key: str) -> npt.NDArray[np.uint32]:
    out: list[int] = []
    for item in values:
        if not isinstance(item, int):
            raise ValueError(f"{key} entries must be integers")
        out.append(item)
    return np.asarray(out, dtype=np.uint32)


def _complex_array(values: Sequence[object], key: str) -> npt.NDArray[np.complex128]:
    out: list[complex] = []
    for item in values:
        if not isinstance(item, Sequence) or isinstance(item, str | bytes) or len(item) != 2:
            raise ValueError(f"{key} entries must be [real, imag] pairs")
        real, imag = item
        if not isinstance(real, int | float) or not isinstance(imag, int | float):
            raise ValueError(f"{key} entries must be numeric [real, imag] pairs")
        out.append(complex(float(real), float(imag)))
    return np.asarray(out, dtype=np.complex128)


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _rate(count: int, elapsed_ns: int) -> float:
    return float(count * 1_000_000_000.0 / elapsed_ns) if elapsed_ns > 0 else 0.0


def _mbits_per_s(bits_total: int, elapsed_ns: int) -> float:
    return _rate(bits_total, elapsed_ns) / 1_000_000.0


def _effective_mbits_per_s(bits_total: int, sample_count: int, sample_rate_hz: float) -> float:
    if sample_count <= 0:
        return 0.0
    return float(bits_total * sample_rate_hz / sample_count / 1_000_000.0)


def _subcarrier_spacing_hz(config: SOFDMLoadConfig) -> float:
    return float(config.sample_rate_hz / config.tx.n_fft)


def _occupied_active_bandwidth_hz(config: SOFDMLoadConfig) -> float:
    return float(config.tx.active_subcarriers.size * _subcarrier_spacing_hz(config))


def _spectral_efficiency_bits_per_s_hz(
    config: SOFDMLoadConfig,
    bits_total: int,
    sample_count: int,
) -> float:
    bandwidth_hz = _occupied_active_bandwidth_hz(config)
    if sample_count <= 0 or bandwidth_hz <= 0.0:
        return 0.0
    effective_bits_per_s = float(bits_total * config.sample_rate_hz / sample_count)
    return float(effective_bits_per_s / bandwidth_hz)


def _active_tone_utilization(config: SOFDMLoadConfig) -> float:
    return float(config.tx.active_subcarriers.size / config.tx.n_fft)


def _stage_percent(
    *,
    tx_ns: int,
    channel_ns: int,
    dac_ns: int,
    adc_ns: int,
    rx_ns: int,
    total_ns: int,
) -> dict[str, float]:
    return {
        "tx": _percent(tx_ns, total_ns),
        "channel": _percent(channel_ns, total_ns),
        "dac": _percent(dac_ns, total_ns),
        "adc": _percent(adc_ns, total_ns),
        "rx": _percent(rx_ns, total_ns),
    }


def _percent(part: int, whole: int) -> float:
    return float(part * 100.0 / whole) if whole > 0 else 0.0


def _timing_summary_from_samples(
    samples_ns: Sequence[int],
    *,
    warmup_iterations: int,
) -> MetricsPayload:
    arr = np.asarray(samples_ns, dtype=np.float64)
    if arr.size == 0:
        return {
            "min_ns": 0,
            "p50_ns": 0,
            "p95_ns": 0,
            "mean_ns": 0.0,
            "std_ns": 0.0,
            "iterations": 0,
            "warmup_iterations": warmup_iterations,
        }
    return {
        "min_ns": int(np.min(arr)),
        "p50_ns": int(np.percentile(arr, 50)),
        "p95_ns": int(np.percentile(arr, 95)),
        "mean_ns": float(np.mean(arr)),
        "std_ns": float(np.std(arr)),
        "iterations": int(arr.size),
        "warmup_iterations": int(warmup_iterations),
    }


def _iq_clip_ratio(clipped_count: int, complex_sample_count: int) -> float:
    return _ratio(clipped_count, complex_sample_count * 2)


def _evm_db(
    tx_symbols: npt.NDArray[np.complex128],
    rx_symbols: npt.NDArray[np.complex128],
) -> float | None:
    value = evm_db_metric(tx_symbols, rx_symbols)
    return float(value) if np.isfinite(value) else None


def _snr_db_from_evm(evm_value: float | None) -> float | None:
    if evm_value is None or evm_value <= 0.0:
        return None
    return float(-20.0 * np.log10(evm_value))


__all__ = [
    "BaselineOFDMResult",
    "DACConfig",
    "DACResult",
    "DacMode",
    "OFDMTXConfig",
    "SOFDMLinkMetrics",
    "SOFDMLinkResult",
    "SOFDMLoadConfig",
    "SOFDMTXFrame",
    "apply_known_channel",
    "build_sofdm_tx_frame",
    "load_sofdm_link_config",
    "run_basic_ofdm_baseline",
    "run_sofdm_link_load",
    "simulate_iq_dac",
    "sofdm_link_config_from_mapping",
    "validate_sofdm_link_load",
    "write_sofdm_link_metrics",
]
