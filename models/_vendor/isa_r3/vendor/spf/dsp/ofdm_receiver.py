"""sOFDM receive path with a simulated IQ ADC front end."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal  # type: ignore[import-untyped]

from spf.dsp.fft_backend import fft_rows
from spf.dsp.metrics import evm
from spf.dsp.modulation import psk_demodulate_spf, qam16_demodulate_fp
from spf.dsp.ofdm_compat import Modulation, apply_diagonal_words, grid_from_bits
from spf.dsp.ofdm_grid import OFDMBinClass, OFDMResourceGrid
from spf.dsp.ofdm_pilots import (
    ChannelEstimate,
    PilotPattern,
    estimate_channel_from_pilots,
    tx_frame_with_pilots,
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
    from_receiver_config,
)
from spf.dsp.ofdm_synth import export_iq16_ifft
from spf.dsp.ofdm_v02 import RXProfile, spf_word_equalize, spf_word_fft
from spf.native import spf32 as c_spf32
from spf.tensor import SPF32_WORD_ZERO, SynthPlan, decode_words, encode_words, spf_mul_words

ADCInput = npt.NDArray[np.int16] | npt.NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class ADCConfig:
    """Configuration for the simulated IQ ADC front end."""

    bit_depth: int = 12
    full_scale: float = 1.0
    snr_db: float | None = None
    resample_up: int = 1
    resample_down: int = 1
    fir_taps: tuple[float, ...] = ()
    seed: int = 20260521


@dataclass(frozen=True, slots=True)
class ADCResult:
    """Quantized ADC output plus telemetry.

    ``clipped_fraction`` is ``clipped_count / sample_count`` (0.0 if no
    samples). ``awgn_snr_db_estimated`` is measured from the AWGN-injected
    samples and is ``None`` when no noise was added. ``post_quantize_evm_db``
    reports the loss from quantization alone (post-AWGN signal vs dequantized
    samples), useful for the bit-depth sweep. ``stage_timings_us`` carries the
    wall-clock cost of each ``stages`` entry so per-stage latency can be
    surfaced in load-mode JSON.
    """

    samples: npt.NDArray[np.complex128]
    quantized_iq: npt.NDArray[np.int16]
    bit_depth: int
    full_scale: float
    clipped_count: int
    sample_count: int
    stages: tuple[str, ...]
    clipped_fraction: float = 0.0
    awgn_snr_db_estimated: float | None = None
    fir_taps_used: int = 0
    post_quantize_evm_db: float | None = None
    stage_timings_us: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class OFDMReceiverConfig:
    """Configuration for a known-channel sOFDM receiver."""

    n_fft: int
    ncp: int
    modulation: Modulation
    active_subcarriers: npt.NDArray[np.uint32]
    channel_h: npt.NDArray[np.complex128] | None = None
    qam16_threshold_l: int = 256
    agc: bool = True


@dataclass(frozen=True, slots=True)
class SOFDMReceiverResult:
    """Outputs from receive-side CP removal, FFT, SPF equalization, and demap.

    ``counters`` reports structural hot-path counts (numpy FFTs, Cartesian
    decodes, SPF multiplies, log-projection demaps, …) so claim-ledger
    evidence docs can cite the boundary surface from a single field. ``plan``
    is the resolved :class:`OFDMRXPlan` (legacy ``OFDMReceiverConfig`` callers
    are converted on entry).
    """

    rx_bits: npt.NDArray[np.uint8]
    rx_grid: OFDMResourceGrid
    equalized_words: npt.NDArray[np.uint32]
    equalized_symbols: npt.NDArray[np.complex128]
    confidence: npt.NDArray[np.uint8] | None
    bit_error_rate: float | None
    evm: float | None
    demap: str
    agc_gain: float
    stages: tuple[str, ...]
    counters: OFDMRXCounters
    plan: OFDMRXPlan
    observed_words: npt.NDArray[np.uint32]
    channel_estimate: ChannelEstimate | None = None
    cpe: CPEEstimate | None = None
    timing: TimingEstimate | None = None
    profile: str = "numpy"
    precision: str = "balanced"


@dataclass(frozen=True, slots=True)
class SOFDMADCRoundTripResult:
    """End-to-end sOFDM TX export, ADC, and receiver result."""

    tx_bits: npt.NDArray[np.uint8]
    tx_grid: OFDMResourceGrid
    channel_grid: OFDMResourceGrid
    adc: ADCResult
    receiver: SOFDMReceiverResult


def simulate_iq_adc(samples: npt.ArrayLike, config: ADCConfig | None = None) -> ADCResult:
    """Run complex samples through a deterministic IQ ADC model.

    The input may be interleaved IQ16 output from :func:`export_iq16_ifft` or a
    complex baseband vector. Optional FIR filtering and rational resampling use
    SciPy's production DSP primitives; quantization is signed uniform I/Q.
    """
    adc_config = ADCConfig() if config is None else config
    _validate_adc_config(adc_config)
    t_ingress = time.perf_counter()
    x = _as_complex_samples(samples)
    stages: list[str] = ["iq_ingress"]
    stage_timings: list[float] = [(time.perf_counter() - t_ingress) * 1e6]

    fir_taps_used = 0
    if adc_config.fir_taps:
        t0 = time.perf_counter()
        taps = np.asarray(adc_config.fir_taps, dtype=np.float64)
        fir_taps_used = int(taps.size)
        x = np.asarray(scipy_signal.lfilter(taps, np.asarray([1.0]), x), dtype=np.complex128)
        stages.append("scipy_lfilter")
        stage_timings.append((time.perf_counter() - t0) * 1e6)

    if adc_config.resample_up != 1 or adc_config.resample_down != 1:
        t0 = time.perf_counter()
        x = np.asarray(
            scipy_signal.resample_poly(x, adc_config.resample_up, adc_config.resample_down),
            dtype=np.complex128,
        )
        stages.append("scipy_resample_poly")
        stage_timings.append((time.perf_counter() - t0) * 1e6)

    awgn_snr_db_estimated: float | None = None
    if adc_config.snr_db is not None:
        t0 = time.perf_counter()
        x_pre_noise = x.copy()
        x = _add_awgn(x, adc_config.snr_db, adc_config.seed)
        signal_power = float(np.mean(np.abs(x_pre_noise) ** 2))
        noise_power = float(np.mean(np.abs(x - x_pre_noise) ** 2))
        if signal_power > 0.0 and noise_power > 0.0:
            awgn_snr_db_estimated = 10.0 * float(np.log10(signal_power / noise_power))
        stages.append("awgn")
        stage_timings.append((time.perf_counter() - t0) * 1e6)

    t0 = time.perf_counter()
    quantized, clipped = _quantize_iq(x, adc_config.bit_depth, adc_config.full_scale)
    dequantized = _dequantize_iq(quantized, adc_config.bit_depth, adc_config.full_scale)
    stages.append("signed_iq_quantize")
    stage_timings.append((time.perf_counter() - t0) * 1e6)
    pre_q_power = float(np.mean(np.abs(x) ** 2))
    q_error_power = float(np.mean(np.abs(dequantized - x) ** 2))
    post_quantize_evm_db: float | None = None
    if pre_q_power > 0.0 and q_error_power > 0.0:
        post_quantize_evm_db = 10.0 * float(np.log10(q_error_power / pre_q_power))
    sample_count = int(dequantized.size)
    # clipped counts I and Q scalars separately (2 per complex sample).
    clipped_fraction = float(clipped) / float(2 * sample_count) if sample_count else 0.0
    return ADCResult(
        samples=dequantized,
        quantized_iq=quantized,
        bit_depth=adc_config.bit_depth,
        full_scale=adc_config.full_scale,
        clipped_count=clipped,
        sample_count=sample_count,
        stages=tuple(stages),
        clipped_fraction=clipped_fraction,
        awgn_snr_db_estimated=awgn_snr_db_estimated,
        fir_taps_used=fir_taps_used,
        post_quantize_evm_db=post_quantize_evm_db,
        stage_timings_us=tuple(stage_timings),
    )


def receive_sofdm_frame(
    adc_samples: npt.ArrayLike,
    config: OFDMReceiverConfig | OFDMRXPlan,
    *,
    tx_bits: npt.ArrayLike | None = None,
    reference_symbols: npt.ArrayLike | None = None,
    pilots: PilotPattern | None = None,
    collect_equalized_symbols: bool = True,
    collect_evm: bool = True,
    profile: RXProfile = "numpy",
    precision: str = "balanced",
) -> SOFDMReceiverResult:
    """Receive one or more CP-prefixed sOFDM symbols from complex ADC samples.

    ``config`` accepts either the new :class:`OFDMRXPlan` (preferred) or the
    legacy :class:`OFDMReceiverConfig` (emits a one-shot
    :class:`DeprecationWarning`). When ``plan.channel_policy`` is
    :data:`ChannelPolicy.PILOT_ESTIMATED`, ``pilots`` is required and the
    channel response is estimated from the pilot bins; payload BER (if
    ``tx_bits`` is supplied) excludes pilot subcarriers.

    ``profile`` selects the spectral transform (Algorithm Reference v0.2 §12):

    ``"numpy"`` (default, historical)
        :func:`spf.dsp.fft_backend.fft_rows` on complex samples, then encode to
        SPF-32 and equalize in the word domain.  ``numpy_fft_calls`` counts it.
    ``"fb7"``
        the v0.1 ``LEGACY_FB7_POLY7`` fused half-angle butterfly on packed
        SPF-32 words (:func:`spf.fft.spf32_fft.fft`, C-backed when libspf32 is
        importable).
    ``"v02"``
        the normative v0.2 profile: ``X = spf_fft_radix2(x, precision)`` on
        packed words followed by ``Y[k] = spf_mul(X[k], h_inv[k])``.  No FFT
        crosses a Cartesian boundary; ``numpy_fft_calls`` stays 0 and
        ``spf_fft_calls`` counts the transformed symbols.

    ``precision`` is the v0.2 pair-adder precision mode and is read only by
    ``profile="v02"``.
    """
    if collect_evm and not collect_equalized_symbols:
        raise ValueError("collect_evm=True requires collect_equalized_symbols=True")
    if profile not in ("numpy", "fb7", "v02"):
        raise ValueError(f"unknown RX profile {profile!r}; expected 'numpy', 'fb7', or 'v02'")
    plan = config if isinstance(config, OFDMRXPlan) else from_receiver_config(config)
    if plan.channel_policy not in (ChannelPolicy.KNOWN, ChannelPolicy.PILOT_ESTIMATED):
        raise NotImplementedError(f"channel_policy={plan.channel_policy.name} is not implemented yet")
    if plan.channel_policy is ChannelPolicy.PILOT_ESTIMATED and pilots is None:
        raise ValueError("PILOT_ESTIMATED channel_policy requires a PilotPattern")
    counters = _MutableCounters()
    active = _active_indices(plan.n_fft, plan.active_subcarriers)
    # Validate the known channel up front so we surface a clear error before
    # consuming the ADC samples or building the spectral grid.
    known_h_active: npt.NDArray[np.complex128] | None = None
    if plan.channel_policy is ChannelPolicy.KNOWN:
        known_h = _channel(plan.n_fft, plan.channel_h, active)
        known_h_active = known_h[active]
    samples = _as_complex_samples(adc_samples)
    symbols = _reshape_cp_symbols(samples, plan.n_fft, plan.ncp)
    n_symbols = symbols.shape[0]

    useful = symbols[:, plan.ncp :]
    rx_active: npt.NDArray[np.complex128] | None
    if profile == "numpy":
        freq = fft_rows(useful)  # numpy by default; KV260 FFT64 overlay under overlay_fft_active
        counters.numpy_fft_calls += n_symbols
        rx_active = np.ascontiguousarray(freq[:, active], dtype=np.complex128)
        y_words = encode_words(rx_active.reshape(-1))
        counters.spf_encode_calls += 1
        fft_stage = "numpy_fft"
    else:
        # §12 word-domain datapath: encode once at the ADC boundary, then the
        # whole transform runs on packed SPF-32 words.
        time_words = encode_words(np.ascontiguousarray(useful, dtype=np.complex128))
        counters.spf_encode_calls += 1
        bins = spf_word_fft(time_words, profile=profile, precision=precision)
        counters.spf_fft_calls += n_symbols
        y_words = np.ascontiguousarray(bins[:, active].reshape(-1), dtype=np.uint32)
        # rx_active is only needed by AGC; decoding it is diagnostic, never
        # part of the datapath, so it stays behind the AGC switch.
        rx_active = None
        fft_stage = "spf_v02_fft" if profile == "v02" else "spf_fb7_fft"

    rx_words = np.full((1, n_symbols, plan.n_fft, 1), SPF32_WORD_ZERO, dtype=np.uint32)
    y_by_symbol = y_words.reshape(n_symbols, active.size)
    for symbol in range(n_symbols):
        rx_words[0, symbol, active, 0] = y_by_symbol[symbol]
    rx_grid = OFDMResourceGrid(rx_words, _bin_classes(n_symbols, plan.n_fft, active, pilots))

    channel_estimate: ChannelEstimate | None = None
    if plan.channel_policy is ChannelPolicy.PILOT_ESTIMATED:
        assert pilots is not None
        channel_estimate = estimate_channel_from_pilots(rx_grid, pilots, plan=plan, mode="compat")
        counters.cartesian_decode_calls += pilots.n_pilots
        counters.spf_decode_calls += 1
        counters.spf_encode_calls += 2  # h + h_inv
        h_active = channel_estimate.h_active
        h_words_active = channel_estimate.h_words
    else:
        assert known_h_active is not None
        h_active = known_h_active
        h_words_active = encode_words(known_h_active)
        counters.spf_encode_calls += 1

    if plan.agc:
        if rx_active is None:
            rx_active = decode_words(y_words).reshape(n_symbols, active.size)
            counters.spf_decode_calls += 1
        agc_gain = _agc_gain(rx_active, h_active, plan.modulation)
        counters.agc_applied = True
    else:
        agc_gain = 1.0
    h_inv_words = encode_words((1.0 / h_active) * agc_gain)
    counters.equalizer_inverts += 1
    counters.spf_encode_calls += 1

    cpe: CPEEstimate | None = None
    timing: TimingEstimate | None = None
    y_corrected = y_words
    residual_policy_set = (
        plan.cpe_policy is CPEPolicy.PILOT_COMMON_PHASE or plan.timing_policy is TimingPolicy.PILOT_PHASE_SLOPE
    )
    if residual_policy_set and pilots is None:
        raise ValueError("PILOT_COMMON_PHASE / PILOT_PHASE_SLOPE policies require a PilotPattern")
    if plan.timing_policy is TimingPolicy.PILOT_PHASE_SLOPE:
        assert pilots is not None
        timing = estimate_timing_ramp(rx_grid.words, pilots, plan=plan, h_words_active=h_words_active)
        y_corrected = apply_timing_correction(y_corrected, timing, plan=plan, n_symbols=n_symbols)
    if plan.cpe_policy is CPEPolicy.PILOT_COMMON_PHASE:
        assert pilots is not None
        cpe = estimate_cpe_from_pilots(
            rx_grid.words, pilots, plan=plan, h_words_active=h_words_active, n_symbols=n_symbols
        )
        y_corrected = apply_cpe_correction(y_corrected, cpe, plan=plan, n_symbols=n_symbols)

    h_words = np.tile(h_inv_words, n_symbols)
    # Only pass plan.cpe_theta to the native demap when no pilot CPE was
    # applied at the y_words layer; otherwise the kernel would double-apply.
    native_cpe = 0 if cpe is not None else int(plan.cpe_theta) & 0xFFFF
    equalized: npt.NDArray[np.uint32] | None = None
    if plan.modulation == "qpsk":
        equalized = spf_word_equalize(y_corrected, h_words, profile=profile)
        counters.bump_spf_mul_operational()
    rx_bits, confidence, demap, demap_equalized = _demap_words(
        y_corrected,
        h_words,
        plan.modulation,
        threshold_l=plan.qam16_threshold_l,
        counters=counters,
        cpe_theta=native_cpe,
        phase_bits=plan.phase_bits,
        agc_l_offset=plan.agc_l_offset,
        equalized_words=equalized,
    )
    if equalized is None:
        equalized = demap_equalized

    if collect_equalized_symbols:
        if equalized is None:
            equalized = spf_word_equalize(y_corrected, h_words, profile=profile)
            counters.bump_spf_mul_diagnostic()
        eq_symbols = decode_words(equalized)
        counters.spf_decode_calls += 1
        equalized_out = equalized
    else:
        eq_symbols = np.empty(0, dtype=np.complex128)
        equalized_out = np.empty(0, dtype=np.uint32)
    bit_error_rate = _payload_bit_error_rate(
        rx_bits=rx_bits,
        tx_bits=tx_bits,
        plan=plan,
        active=active,
        pilots=pilots,
        n_symbols=n_symbols,
    )
    ref = None if reference_symbols is None else np.asarray(reference_symbols, dtype=np.complex128)
    evm_value = None if ref is None or not collect_evm else evm(ref.ravel(), eq_symbols)
    return SOFDMReceiverResult(
        rx_bits=rx_bits,
        rx_grid=rx_grid,
        equalized_words=equalized_out,
        equalized_symbols=eq_symbols,
        confidence=confidence,
        bit_error_rate=bit_error_rate,
        evm=evm_value,
        demap=demap,
        agc_gain=float(agc_gain),
        stages=(
            ("cp_remove", fft_stage, "spf_encode", "spf_equalize", demap)
            if profile == "numpy"
            else ("cp_remove", "spf_encode", fft_stage, "spf_equalize", demap)
        ),
        counters=counters.freeze(),
        plan=plan,
        observed_words=y_words,
        channel_estimate=channel_estimate,
        cpe=cpe,
        timing=timing,
        profile=profile,
        precision=precision,
    )


def run_sofdm_adc_roundtrip(
    bits: npt.ArrayLike,
    *,
    receiver: OFDMReceiverConfig | OFDMRXPlan,
    adc: ADCConfig | None = None,
    channel_h: npt.ArrayLike | None = None,
    pilots: PilotPattern | None = None,
    profile: RXProfile = "numpy",
    precision: str = "balanced",
) -> SOFDMADCRoundTripResult:
    """Build an sOFDM grid, export IQ16, quantize through ADC, and receive it.

    ``receiver`` accepts either :class:`OFDMRXPlan` (preferred) or the legacy
    :class:`OFDMReceiverConfig`. Internal stages run against the plan view so
    the round-trip does not double-emit the deprecation warning.

    When ``pilots`` is supplied the TX grid is built via
    :func:`spf.dsp.ofdm_pilots.tx_frame_with_pilots` and ``bits`` is the
    data-only bit stream (length must match the non-pilot active subcarriers).
    ``channel_h`` overrides ``plan.channel_h`` for the actual TX-side channel
    when the receiver runs under ``PILOT_ESTIMATED`` — useful for tests that
    want to exercise estimation without giving the RX side a known answer.

    ``profile`` / ``precision`` are forwarded to :func:`receive_sofdm_frame`
    (Algorithm Reference v0.2 §12 RX profile selection).
    """
    plan = receiver if isinstance(receiver, OFDMRXPlan) else from_receiver_config(receiver)
    adc_config = adc if adc is not None else ADCConfig()
    if adc_config.resample_up != adc_config.resample_down:
        raise ValueError(
            "run_sofdm_adc_roundtrip does not support rate-changing resampling "
            f"(resample_up={adc_config.resample_up}, resample_down={adc_config.resample_down}); "
            "call simulate_iq_adc and receive_sofdm_frame separately"
        )
    tx_bits = np.ascontiguousarray(np.asarray(bits), dtype=np.uint8).ravel()
    if pilots is None:
        tx_grid = grid_from_bits(
            tx_bits,
            modulation=plan.modulation,
            n_fft=plan.n_fft,
            active_subcarriers=plan.active_subcarriers,
        )
    else:
        tx_grid = tx_frame_with_pilots(
            tx_bits,
            pilots,
            modulation=plan.modulation,
            n_fft=plan.n_fft,
            active_subcarriers=plan.active_subcarriers,
        )
    active = tx_grid.active_subcarriers()
    effective_h = channel_h if channel_h is not None else plan.channel_h
    h = _channel(plan.n_fft, effective_h, active)
    channel_words = np.full(tx_grid.shape, SPF32_WORD_ZERO, dtype=np.uint32)
    h_words = encode_words(h)
    for symbol in range(tx_grid.shape[1]):
        channel_words[0, symbol, :, 0] = h_words
    channel_grid = apply_diagonal_words(tx_grid, channel_words)
    waveform, _ = export_iq16_ifft(
        channel_grid,
        SynthPlan(nfft=plan.n_fft, ncp=plan.ncp, n_symbols=tx_grid.shape[1]),
    )
    adc_result = simulate_iq_adc(waveform, adc)
    ref_words = tx_grid.words[0, :, :, 0][:, tx_grid.active_subcarriers()]
    ref_symbols = decode_words(ref_words.reshape(-1))
    rx = receive_sofdm_frame(
        adc_result.samples,
        plan,
        tx_bits=tx_bits,
        reference_symbols=ref_symbols,
        pilots=pilots,
        profile=profile,
        precision=precision,
    )
    return SOFDMADCRoundTripResult(
        tx_bits=tx_bits,
        tx_grid=tx_grid,
        channel_grid=channel_grid,
        adc=adc_result,
        receiver=rx,
    )


def _validate_adc_config(config: ADCConfig) -> None:
    if not 2 <= config.bit_depth <= 16:
        raise ValueError(f"bit_depth must be in [2, 16], got {config.bit_depth}")
    if config.full_scale <= 0.0:
        raise ValueError("full_scale must be positive")
    if config.resample_up <= 0 or config.resample_down <= 0:
        raise ValueError("resample_up and resample_down must be positive")


def _as_complex_samples(samples: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    arr = np.asarray(samples)
    if np.iscomplexobj(arr):
        return np.ascontiguousarray(arr, dtype=np.complex128).ravel()
    real = np.ascontiguousarray(arr).ravel()
    if real.size % 2:
        raise ValueError("interleaved IQ samples must have an even length")
    if np.issubdtype(real.dtype, np.integer):
        scale = float(np.iinfo(real.dtype).max)
        pairs = real.astype(np.float64).reshape(-1, 2) / scale
    else:
        pairs = real.astype(np.float64).reshape(-1, 2)
    return np.ascontiguousarray(pairs[:, 0] + 1j * pairs[:, 1], dtype=np.complex128)


def _add_awgn(
    samples: npt.NDArray[np.complex128],
    snr_db: float,
    seed: int,
) -> npt.NDArray[np.complex128]:
    power = float(np.mean(np.abs(samples) ** 2))
    if power == 0.0:
        return samples.copy()
    noise_power = power / (10.0 ** (float(snr_db) / 10.0))
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(noise_power / 2.0)
    noise = rng.normal(0.0, sigma, samples.size) + 1j * rng.normal(0.0, sigma, samples.size)
    return np.ascontiguousarray(samples + noise, dtype=np.complex128)


def _quantize_iq(
    samples: npt.NDArray[np.complex128],
    bit_depth: int,
    full_scale: float,
) -> tuple[npt.NDArray[np.int16], int]:
    max_code = (1 << (bit_depth - 1)) - 1
    min_code = -(1 << (bit_depth - 1))
    # complex128 stores (re, im) interleaved as float64 pairs — alias via a
    # contiguous float view to avoid an extra copy for the (re, im) buffer.
    samples_c = np.ascontiguousarray(samples, dtype=np.complex128)
    pairs = samples_c.view(np.float64).reshape(-1, 2)
    scale = float(max_code) / float(full_scale)
    # One allocation for the scaled buffer; rint and clip both run in-place.
    scaled = np.empty(pairs.shape, dtype=np.float64)
    np.multiply(pairs, scale, out=scaled)
    np.rint(scaled, out=scaled)
    clipped = int(np.count_nonzero((scaled > max_code) | (scaled < min_code)))
    np.clip(scaled, min_code, max_code, out=scaled)
    quantized = np.empty(scaled.size, dtype=np.int16)
    np.copyto(quantized, scaled.reshape(-1), casting="unsafe")
    return quantized, clipped


def _dequantize_iq(
    quantized_iq: npt.NDArray[np.int16],
    bit_depth: int,
    full_scale: float,
) -> npt.NDArray[np.complex128]:
    max_code = float((1 << (bit_depth - 1)) - 1)
    scale = float(full_scale) / max_code
    n_samples = quantized_iq.size // 2
    # Pre-allocate the complex output; write the float-scaled int16 values
    # straight into its float64 interleaved view (re, im) rather than
    # allocating a separate pairs/complex array pair.
    out = np.empty(n_samples, dtype=np.complex128)
    out_view = out.view(np.float64).reshape(-1)
    np.multiply(quantized_iq, scale, out=out_view, dtype=np.float64, casting="unsafe")
    return out


def _reshape_cp_symbols(
    samples: npt.NDArray[np.complex128],
    n_fft: int,
    ncp: int,
) -> npt.NDArray[np.complex128]:
    if n_fft <= 0 or ncp < 0 or ncp > n_fft:
        raise ValueError("n_fft must be positive and ncp must be in [0, n_fft]")
    symbol_len = n_fft + ncp
    if samples.size < symbol_len or samples.size % symbol_len:
        raise ValueError(f"sample length {samples.size} must be a nonzero multiple of n_fft+ncp={symbol_len}")
    return np.ascontiguousarray(samples.reshape(samples.size // symbol_len, symbol_len))


def _active_indices(
    n_fft: int,
    active_subcarriers: npt.ArrayLike,
) -> npt.NDArray[np.uint32]:
    active = np.ascontiguousarray(np.asarray(active_subcarriers), dtype=np.uint32).ravel()
    if active.size == 0:
        raise ValueError("active_subcarriers must not be empty")
    if np.any(active == 0) or np.any(active >= n_fft):
        raise ValueError("active_subcarriers must be non-DC bins inside n_fft")
    if not np.all(active[1:] > active[:-1]):
        raise ValueError("active_subcarriers must be strictly increasing")
    return active


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


def _agc_gain(
    rx_active: npt.NDArray[np.complex128],
    h_active: npt.NDArray[np.complex128],
    modulation: Modulation,
) -> float:
    equalized = rx_active / h_active.reshape(1, -1)
    observed = float(np.sqrt(np.mean(np.abs(equalized) ** 2)))
    if observed <= 0.0:
        return 1.0
    target = 1.0 if modulation == "qpsk" else np.sqrt(10.0)
    return float(target / observed)


def _demap_words(
    y_words: npt.NDArray[np.uint32],
    h_inv_words: npt.NDArray[np.uint32],
    modulation: Modulation,
    *,
    threshold_l: int,
    counters: _MutableCounters,
    cpe_theta: int = 0,
    phase_bits: int = 10,
    agc_l_offset: int = 0,
    erasure_mask: npt.NDArray[np.uint8] | None = None,
    equalized_words: npt.NDArray[np.uint32] | None = None,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8] | None,
    str,
    npt.NDArray[np.uint32] | None,
]:
    if modulation == "qpsk":
        if equalized_words is None:
            raise ValueError("QPSK demap requires precomputed equalized_words")
        counters.qpsk_phase_demap_calls += 1
        return psk_demodulate_spf(equalized_words, 4), None, "qpsk_phase", equalized_words
    if modulation == "qam16":
        try:
            bits, confidence = c_spf32.qam16_logproj_eq_demod_fused(
                y_words,
                h_inv_words,
                cpe_theta=cpe_theta,
                agc_L_offset=agc_l_offset,
                threshold_L=threshold_l,
                phase_bits=phase_bits,
                erasure_mask=erasure_mask,
            )
            counters.qam_logproj_calls += int(y_words.size)
            return bits, confidence, "qam16_logproj", None
        except RuntimeError:
            equalized = spf_mul_words(y_words, h_inv_words)
            counters.bump_spf_mul_operational()
            counters.cartesian_decode_calls += int(equalized.size)
            counters.spf_decode_calls += 1
            counters.qam_logproj_calls += int(y_words.size)
            bits = qam16_demodulate_fp(decode_words(equalized))
            confidence = np.full(equalized.size, 255, dtype=np.uint8)
            return bits, confidence, "qam16_logproj_python_fallback", equalized
    bad = cast(Literal["unsupported"], modulation)
    raise ValueError(f"unsupported modulation {bad!r}")


def _bin_classes(
    n_symbols: int,
    n_fft: int,
    active: npt.NDArray[np.uint32],
    pilots: PilotPattern | None = None,
) -> npt.NDArray[np.int16]:
    classes = np.full((1, n_symbols, n_fft, 1), int(OFDMBinClass.GUARD), dtype=np.int16)
    classes[:, :, 0, :] = int(OFDMBinClass.DC_NULL)
    classes[0, :, active, 0] = int(OFDMBinClass.DATA)
    if pilots is not None:
        for sym_idx in pilots.symbol_indices.tolist():
            if int(sym_idx) >= n_symbols:
                raise ValueError(
                    f"pilot symbol_idx={int(sym_idx)} out of range for frame "
                    f"with {n_symbols} symbols"
                )
            classes[0, int(sym_idx), pilots.subcarrier_indices, 0] = int(OFDMBinClass.PILOT)
    return classes


def _bits_per_symbol(modulation: Modulation) -> int:
    if modulation == "qpsk":
        return 2
    if modulation == "qam16":
        return 4
    bad = cast(Literal["unsupported"], modulation)
    raise ValueError(f"unsupported modulation {bad!r}")


def _payload_bit_error_rate(
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
    bps = _bits_per_symbol(plan.modulation)
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


__all__ = [
    "ADCConfig",
    "ADCInput",
    "ADCResult",
    "OFDMRXCounters",
    "OFDMRXPlan",
    "OFDMReceiverConfig",
    "RXProfile",
    "SOFDMADCRoundTripResult",
    "SOFDMReceiverResult",
    "receive_sofdm_frame",
    "run_sofdm_adc_roundtrip",
    "simulate_iq_adc",
]
