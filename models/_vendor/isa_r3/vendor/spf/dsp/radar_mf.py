"""Radar matched-filter / pulse-compression: SPF hotpath arm + fp64 baseline.

Two arms consume the same :class:`spf.dsp.radar_mf_sim.RadarScene` boundary and
the same padded chirp reference, so accuracy/latency deltas isolate the
matched-filter path:

- **SPF arm** (:func:`run_spf_matched_filter`): ``encode_words → spf32 fft →
  spectral_matched_filter_fb7 (fused spectral multiply + IFFT) → log_power /
  log_os_cfar``. Detection runs in the log domain on SPF words with **no
  decode**; decode happens only for diagnostic accuracy grading.
- **fp64 arm** (:func:`run_fp_matched_filter`): ``np.fft.fft → X·conj(H) →
  np.fft.ifft → |·|²`` with an OS-CFAR mirror of the native kernel.

The native ``spectral_matched_filter_fb7`` multiplies ``x_fft ⊙ h_conj_fft``
(caller pre-conjugates the reference) then IFFTs internally; a pure-Python
``spf32`` multiply + IFFT fallback runs when libspf32 is unavailable so the
arms work everywhere (tighter parity asserted when native is present).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from spf.dsp import metrics, radar_metrics
from spf.dsp.kernels import (
    log_os_cfar,
    log_power,
    phase_slope_estimate,
    spectral_matched_filter_fb7,
)
from spf.dsp.radar_mf_sim import RadarScene
from spf.fft import spf32_fft
from spf.native._lib import LIBSPF_AVAILABLE
from spf.runtime import RuntimeTelemetry
from spf.scalar import spf32 as scalar_spf32
from spf.tensor import decode_words, encode_words

ArmKind = Literal["spf", "fp"]
RangingMode = Literal["ifft", "phase_slope"]
_PHASE_MOD = 65536  # 2^16 — full turn in SPF-32 Θ ticks

_L_ZERO_LOGPOWER = -65536  # 2 · SPF32_L_ZERO (0x8000 = -32768); log-power of a true zero


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """OS-CFAR parameters (log-power Q7.8 domain), matching the native kernel."""

    threshold_offset_q78: int = 1024  # +4 in log2(power) ≈ +12 dB over the rank cell
    guard: int = 2
    train: int = 8
    rank: int = 12


@dataclass(frozen=True, slots=True)
class RadarMFArmResult:
    """One matched-filter arm's range profile, detections, and quality metrics."""

    arm: ArmKind
    peak_bin: int
    log_power_q78: npt.NDArray[np.int32]
    detections: npt.NDArray[np.uint8] | None
    range_profile: npt.NDArray[np.complex128] | None  # decoded; None on hot path
    pslr_db: float | None
    islr_db: float | None
    peak_snr_db: float | None
    telemetry: RuntimeTelemetry

    def to_dict(self) -> dict[str, object]:
        return {
            "arm": self.arm,
            "backend": self.telemetry.backend,
            "peak_bin": self.peak_bin,
            "pslr_db": self.pslr_db,
            "islr_db": self.islr_db,
            "peak_snr_db": self.peak_snr_db,
            "n_detections": None if self.detections is None else int(self.detections.sum()),
            "stages": list(self.telemetry.stages),
        }


@dataclass(frozen=True, slots=True)
class RadarMFResult:
    """SPF arm vs fp64 baseline over one shared scene."""

    scenario: str
    n: int
    spf: RadarMFArmResult
    fp: RadarMFArmResult | None
    peak_bin_match: bool
    peak_bin_diff: int
    range_profile_nmse_db: float | None
    range_profile_evm: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "scenario": self.scenario,
            "n": self.n,
            "peak_bin_match": self.peak_bin_match,
            "peak_bin_diff": self.peak_bin_diff,
            "range_profile_nmse_db": self.range_profile_nmse_db,
            "range_profile_evm": self.range_profile_evm,
            "spf": self.spf.to_dict(),
            "fp": None if self.fp is None else self.fp.to_dict(),
        }


# --- reference (filter-design-time) prep --------------------------------------


def spf_matched_filter_reference(
    reference_chirp: npt.NDArray[np.complex128],
) -> npt.NDArray[np.uint32]:
    """Precompute the SPF conjugate-spectrum reference ``conj(fft(encode(h)))``."""
    h_words = encode_words(np.asarray(reference_chirp, dtype=np.complex128))
    h_fft = spf32_fft.fft(h_words)
    return scalar_spf32.conj(h_fft)


def fp_matched_filter_reference(
    reference_chirp: npt.NDArray[np.complex128],
) -> npt.NDArray[np.complex128]:
    """Precompute the fp64 conjugate-spectrum reference ``conj(fft(h))``."""
    return np.conj(np.fft.fft(np.asarray(reference_chirp, dtype=np.complex128)))


# --- detection helpers --------------------------------------------------------


def log_power_q78_from_words(words: npt.NDArray[np.uint32]) -> npt.NDArray[np.int32]:
    """log2(power) in Q7.8 ticks (= 2·L) from SPF words; native when available."""
    if LIBSPF_AVAILABLE:
        return log_power(words)
    arr = np.asarray(words, dtype=np.uint32)
    L = (arr >> np.uint32(16)).astype(np.int16).astype(np.int32)
    return 2 * L


def log_power_q78_from_complex(y: npt.NDArray[np.complex128]) -> npt.NDArray[np.int32]:
    """log2(|y|²) in Q7.8 ticks — the fp-arm analog of :func:`log_power`."""
    power = np.abs(np.asarray(y, dtype=np.complex128)) ** 2
    out = np.full(power.shape, _L_ZERO_LOGPOWER, dtype=np.int32)
    pos = power > 0.0
    out[pos] = np.round(256.0 * np.log2(power[pos])).astype(np.int32)
    return out


def os_cfar_q78(
    log_power_q78: npt.NDArray[np.int32], detector: DetectorConfig
) -> npt.NDArray[np.uint8]:
    """Vectorized ordered-statistic CFAR; bit-identical to ``spf32_log_os_cfar_array``.

    For every cell ``i`` with a full ``guard``/``train`` neighborhood, the rank-th
    smallest of the ``2·train`` training cells (lagging + leading, excluding guard)
    sets the noise estimate; detect when ``cell > noise + offset``. Cells without a
    full neighborhood stay 0, exactly as the native kernel.
    """
    cells = np.asarray(log_power_q78, dtype=np.int32).ravel()
    n = cells.size
    det = np.zeros(n, dtype=np.uint8)
    g, tr = detector.guard, detector.train
    win = tr * 2
    if win == 0 or n < 2 * (g + tr) + 1:
        return det
    rank = min(detector.rank, win - 1)
    centers = np.arange(tr + g, n - tr - g)
    offsets = np.concatenate([np.arange(-(g + tr), -g), np.arange(g + 1, g + tr + 1)])
    window = np.sort(cells[centers[:, None] + offsets[None, :]], axis=1)
    noise = window[:, rank]
    det[centers] = (cells[centers] > noise + detector.threshold_offset_q78).astype(np.uint8)
    return det


def _spf_detect(
    words: npt.NDArray[np.uint32], detector: DetectorConfig
) -> npt.NDArray[np.uint8]:
    if LIBSPF_AVAILABLE:
        return log_os_cfar(
            words,
            threshold_offset_q78=detector.threshold_offset_q78,
            guard=detector.guard,
            train=detector.train,
            rank=detector.rank,
        )
    return os_cfar_q78(log_power_q78_from_words(words), detector)


def _spf_matched_filter_core(
    x_fft: npt.NDArray[np.uint32],
    h_conj_fft: npt.NDArray[np.uint32],
    fb7_mode: str,
) -> npt.NDArray[np.uint32]:
    if LIBSPF_AVAILABLE:
        return spectral_matched_filter_fb7(x_fft, h_conj_fft, mode=fb7_mode)
    product = scalar_spf32.mul(x_fft, h_conj_fft)
    return spf32_fft.ifft(product)


# --- arms ---------------------------------------------------------------------


def run_spf_matched_filter(
    scene: RadarScene,
    *,
    h_conj_fft: npt.NDArray[np.uint32] | None = None,
    fb7_mode: str = "balanced",
    diagnostics: bool = True,
    detect: bool = True,
    detector: DetectorConfig | None = None,
) -> RadarMFArmResult:
    """SPF-hotpath matched filter: encode → fft → fused MF+IFFT → log-domain detect."""
    det_cfg = detector or DetectorConfig()
    ref = (
        h_conj_fft
        if h_conj_fft is not None
        else spf_matched_filter_reference(scene.reference_chirp)
    )
    x_words = encode_words(scene.echo_samples)
    x_fft = spf32_fft.fft(x_words)
    y_words = _spf_matched_filter_core(x_fft, ref, fb7_mode)
    lp = log_power_q78_from_words(y_words)
    peak = int(np.argmax(lp))
    detections = _spf_detect(y_words, det_cfg) if detect else None

    range_profile = None
    pslr = islr = psnr = None
    if diagnostics:
        range_profile = decode_words(y_words)
        power = radar_metrics.power_profile(range_profile)
        pslr = radar_metrics.pslr_db(power, peak=peak)
        islr = radar_metrics.islr_db(power, peak=peak)
        psnr = radar_metrics.peak_snr_db(power, peak=peak)

    telemetry = RuntimeTelemetry(
        backend="spf-native" if LIBSPF_AVAILABLE else "spf-python",
        stages=("encode", "fft", "mf_ifft", "detect"),
        diagnostics=diagnostics,
        details={"fb7_mode": fb7_mode, "n": int(scene.config.n)},
    )
    return RadarMFArmResult(
        arm="spf",
        peak_bin=peak,
        log_power_q78=lp,
        detections=detections,
        range_profile=range_profile,
        pslr_db=pslr,
        islr_db=islr,
        peak_snr_db=psnr,
        telemetry=telemetry,
    )


def run_fp_matched_filter(
    scene: RadarScene,
    *,
    h_conj_fft: npt.NDArray[np.complex128] | None = None,
    diagnostics: bool = True,
    detect: bool = True,
    detector: DetectorConfig | None = None,
) -> RadarMFArmResult:
    """fp64 baseline matched filter via numpy FFTs (the standard reference)."""
    det_cfg = detector or DetectorConfig()
    ref = (
        h_conj_fft
        if h_conj_fft is not None
        else fp_matched_filter_reference(scene.reference_chirp)
    )
    x_fft = np.fft.fft(scene.echo_samples)
    y = np.fft.ifft(x_fft * ref)
    power = radar_metrics.power_profile(y)
    peak = int(np.argmax(power))
    lp = log_power_q78_from_complex(y)
    detections = os_cfar_q78(lp, det_cfg) if detect else None

    range_profile = None
    pslr = islr = psnr = None
    if diagnostics:
        range_profile = y
        pslr = radar_metrics.pslr_db(power, peak=peak)
        islr = radar_metrics.islr_db(power, peak=peak)
        psnr = radar_metrics.peak_snr_db(power, peak=peak)

    telemetry = RuntimeTelemetry(
        backend="numpy-fp64",
        stages=("fft", "mf_ifft", "detect"),
        diagnostics=diagnostics,
        details={"n": int(scene.config.n)},
    )
    return RadarMFArmResult(
        arm="fp",
        peak_bin=peak,
        log_power_q78=lp,
        detections=detections,
        range_profile=range_profile,
        pslr_db=pslr,
        islr_db=islr,
        peak_snr_db=psnr,
        telemetry=telemetry,
    )


def run_radar_mf_compare(
    scene: RadarScene,
    *,
    fb7_mode: str = "balanced",
    diagnostics: bool = True,
    include_baseline: bool = True,
    detector: DetectorConfig | None = None,
) -> RadarMFResult:
    """Run both arms over a shared scene and compute cross-arm accuracy deltas."""
    spf_res = run_spf_matched_filter(
        scene, fb7_mode=fb7_mode, diagnostics=diagnostics, detector=detector
    )
    fp_res = (
        run_fp_matched_filter(scene, diagnostics=diagnostics, detector=detector)
        if include_baseline
        else None
    )

    peak_match = fp_res is not None and spf_res.peak_bin == fp_res.peak_bin
    peak_diff = (spf_res.peak_bin - fp_res.peak_bin) if fp_res is not None else 0

    nmse = evm_val = None
    if (
        fp_res is not None
        and diagnostics
        and spf_res.range_profile is not None
        and fp_res.range_profile is not None
    ):
        nmse = radar_metrics.range_profile_nmse_db(
            spf_res.range_profile, fp_res.range_profile
        )
        evm_val = metrics.evm(fp_res.range_profile, spf_res.range_profile)

    return RadarMFResult(
        scenario=scene.config.scenario,
        n=scene.config.n,
        spf=spf_res,
        fp=fp_res,
        peak_bin_match=peak_match,
        peak_bin_diff=peak_diff,
        range_profile_nmse_db=nmse,
        range_profile_evm=evm_val,
    )


# --- Reduction A: phase-slope ranging (eliminate the IFFT) -------------------


def _phase_slope_numpy(words: npt.NDArray[np.uint32]) -> tuple[int, int]:
    """Unweighted histogram-mode of per-bin Θ steps (no-native fallback)."""
    arr = np.asarray(words, dtype=np.uint32)
    theta = (arr & np.uint32(0xFFFF)).astype(np.int32)
    step = ((theta[1:] - theta[:-1]) & 0xFFFF).astype(np.int32)
    hist = np.bincount((step >> 8), minlength=256)
    mode_bin = int(np.argmax(hist))
    return mode_bin << 8, int(min(255, hist[mode_bin]))


def phase_slope_delay(product_words: npt.NDArray[np.uint32], n: int) -> tuple[int, int]:
    """Reduction A: recover target delay from the slope of ∠P[k]. No IFFT.

    For a single target ``P[k] = a·|S[k]|²·e^(−j2πkd/N)``, so the per-bin Θ step is
    constant ``= −(d/N)·2¹⁶ mod 2¹⁶``. The native (L-weighted) histogram-mode slope
    maps back to ``d = (2¹⁶ − slope)·N / 2¹⁶``. Returns ``(delay_bin, confidence)``.
    """
    if LIBSPF_AVAILABLE:
        slope, _intercept, conf = phase_slope_estimate(
            product_words, n_streams=1, stride=0, n_per_stream=int(n)
        )
        s, c = int(slope[0]), int(conf[0])
    else:
        s, c = _phase_slope_numpy(product_words)
    delay = round(((_PHASE_MOD - s) % _PHASE_MOD) * n / _PHASE_MOD) % n
    return delay, c


@dataclass(frozen=True, slots=True)
class ReducedMFResult:
    """Phase-slope ranging result + provenance."""

    ranging: RangingMode
    delay_est: int
    confidence: int
    ranging_path: str  # "phase_slope" | "ifft_fallback" | "ifft"
    range_profile: npt.NDArray[np.complex128] | None

    def to_dict(self) -> dict[str, object]:
        return {
            "ranging": self.ranging,
            "delay_est": self.delay_est,
            "confidence": self.confidence,
            "ranging_path": self.ranging_path,
        }


def run_spf_matched_filter_reduced(
    scene: RadarScene,
    *,
    h_conj_fft: npt.NDArray[np.uint32] | None = None,
    ranging: RangingMode = "ifft",
    conf_threshold: int = 8,
) -> ReducedMFResult:
    """Matched filter with Reduction A: phase-slope ranging (eliminate the IFFT).

    Computes the spectral product ``P`` explicitly so the delay can be read from
    ∠P. ``ranging="phase_slope"`` recovers the delay from the phase slope with
    **no IFFT**; if the slope confidence is below ``conf_threshold`` it falls back
    to the exact IFFT (confidence-gated), mirroring ``sparse_doppler``.
    ``ranging="ifft"`` is the baseline (full IFFT + argmax).
    """
    ref = (
        h_conj_fft
        if h_conj_fft is not None
        else spf_matched_filter_reference(scene.reference_chirp)
    )
    n = scene.config.n
    x_words = encode_words(scene.echo_samples)
    x_fft = spf32_fft.fft(x_words)
    product = scalar_spf32.mul(x_fft, ref)

    if ranging == "phase_slope":
        delay, conf = phase_slope_delay(product, n)
        if conf >= conf_threshold:
            return ReducedMFResult(ranging, delay, conf, "phase_slope", None)
        # low confidence -> exact IFFT fallback
        y = spf32_fft.ifft(product)
        peak = int(np.argmax(log_power_q78_from_words(y)))
        return ReducedMFResult(ranging, peak, conf, "ifft_fallback", decode_words(y))

    y = spf32_fft.ifft(product)
    peak = int(np.argmax(log_power_q78_from_words(y)))
    return ReducedMFResult(ranging, peak, 0, "ifft", decode_words(y))
