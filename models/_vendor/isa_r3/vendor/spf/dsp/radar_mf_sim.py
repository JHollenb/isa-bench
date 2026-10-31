"""Radar pulse-compression signal model: LFM chirp, multi-target echo, ADC.

Generates the *shared boundary* both matched-filter arms consume — a complex
baseband receive block of length ``N`` (power-of-2 ≤ 1024, the SPF-32 FFT cap)
containing delayed/scaled copies of a linear-FM (LFM) chirp replica plus
circular complex AWGN, optionally passed through a finite-bit ADC.

The SPF arm (:mod:`spf.dsp.radar_mf`) encodes this block into SPF-32 words; the
fp64 baseline arm consumes it directly. Both arms use the same padded
``reference_chirp`` as the matched-filter reference, so timing/accuracy deltas
isolate the matched-filter path.

Noise convention: per-sample input SNR is defined against the *mean pulse
power* of the strongest (reference-amplitude) target,
``SNR_in = a_ref²·mean(|s|²) / σ²``. The matched filter then delivers a coherent
processing gain of ``10·log10(Ns)`` (number of integrated chirp samples),
window-independent under this definition.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import signal as scipy_signal  # type: ignore[import-untyped]

from spf.dsp.ofdm_receiver import ADCConfig, simulate_iq_adc

_MIN_N = 2
_MAX_N = 1024
_WINDOWS = ("rect", "hamming", "taylor")


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One point target: integer range-bin delay and amplitude in dB."""

    delay_bins: int
    amplitude_db: float = 0.0


@dataclass(frozen=True, slots=True)
class AcceptanceSpec:
    """Pass/fail budget for the SPF arm vs the fp64 baseline."""

    evm_max: float = 0.02
    pslr_delta_max_db: float = 2.0
    peak_bin_exact: bool = True


@dataclass(frozen=True, slots=True)
class RadarSceneConfig:
    """Declarative radar pulse-compression scene (JSON-fixture backed)."""

    scenario: str
    n: int
    chirp_samples: int
    sample_rate_hz: float
    bandwidth_hz: float
    targets: tuple[TargetSpec, ...]
    snr_db: float | None = None
    window: str = "rect"
    seed: int = 20260528
    adc: ADCConfig | None = None
    acceptance: AcceptanceSpec = field(default_factory=AcceptanceSpec)

    @property
    def time_bandwidth(self) -> float:
        """Time-bandwidth product TB = B·T = B·(Ns/fs) (compression ratio)."""
        return self.bandwidth_hz * (self.chirp_samples / self.sample_rate_hz)


@dataclass(frozen=True, slots=True)
class RadarScene:
    """Materialized scene: the shared boundary + ground truth."""

    config: RadarSceneConfig
    reference_chirp: npt.NDArray[np.complex128]  # length N, zero-padded replica
    echo_samples: npt.NDArray[np.complex128]  # length N — SHARED BOUNDARY
    target_delays: npt.NDArray[np.int64]
    target_amplitudes: npt.NDArray[np.float64]
    noise_sigma: float


def _amplitude_window(name: str, n: int) -> npt.NDArray[np.float64]:
    if name == "rect":
        return np.ones(n, dtype=np.float64)
    if name == "hamming":
        return np.hamming(n).astype(np.float64)
    if name == "taylor":
        taylor = scipy_signal.windows.taylor(n, nbar=5, sll=35, norm=False)
        return np.asarray(taylor, dtype=np.float64)
    raise ValueError(f"unknown window {name!r}; expected one of {_WINDOWS}")


def generate_lfm_chirp(
    chirp_samples: int,
    *,
    sample_rate_hz: float,
    bandwidth_hz: float,
    window: str = "rect",
) -> npt.NDArray[np.complex128]:
    """Baseband LFM (linear-FM) chirp replica, length ``chirp_samples``.

    ``s[n] = w[n]·exp(j·π·(B/T)·t_n²)`` with ``t_n`` centered on the pulse and
    ``T = chirp_samples/fs``. ``w`` is the amplitude window (rect/hamming/taylor).
    """
    if chirp_samples < 1:
        raise ValueError("chirp_samples must be >= 1")
    if sample_rate_hz <= 0.0 or bandwidth_hz <= 0.0:
        raise ValueError("sample_rate_hz and bandwidth_hz must be positive")
    ns = int(chirp_samples)
    t = (np.arange(ns, dtype=np.float64) - (ns - 1) / 2.0) / sample_rate_hz
    pulse_t = ns / sample_rate_hz
    rate = bandwidth_hz / pulse_t  # chirp rate B/T (Hz/s)
    phase = np.pi * rate * t * t
    s = np.exp(1j * phase).astype(np.complex128)
    return np.asarray(s * _amplitude_window(window, ns), dtype=np.complex128)


def simulate_echo(
    reference_chirp: npt.NDArray[np.complex128],
    targets: tuple[TargetSpec, ...],
    *,
    n: int,
    snr_db: float | None,
    seed: int,
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.int64], npt.NDArray[np.float64], float]:
    """Place delayed/scaled chirp copies into an ``N``-sample window + AWGN.

    Returns ``(echo, delays, amplitudes, noise_sigma)``. ``snr_db=None`` yields a
    noise-free echo (σ=0) for pure-format accuracy measurement.
    """
    chirp = np.asarray(reference_chirp, dtype=np.complex128)
    ns = chirp.size
    delays = np.array([int(t.delay_bins) for t in targets], dtype=np.int64)
    amps = np.array([10.0 ** (t.amplitude_db / 20.0) for t in targets], dtype=np.float64)

    echo = np.zeros(n, dtype=np.complex128)
    for d, a in zip(delays, amps, strict=True):
        echo[d : d + ns] += a * chirp

    ref_amp = float(amps.max()) if amps.size else 1.0
    mean_pulse_power = float(np.mean(np.abs(chirp) ** 2))
    if snr_db is None:
        sigma = 0.0
    else:
        sigma = float(np.sqrt(ref_amp**2 * mean_pulse_power / (10.0 ** (snr_db / 10.0))))
        rng = np.random.default_rng(seed)
        noise = (sigma / np.sqrt(2.0)) * (
            rng.standard_normal(n) + 1j * rng.standard_normal(n)
        )
        echo = echo + noise.astype(np.complex128)
    return echo, delays, amps, sigma


def build_radar_scene(config: RadarSceneConfig) -> RadarScene:
    """Materialize a :class:`RadarScene` from a config (chirp + echo + ADC)."""
    validate_radar_scene_config(config)
    chirp = generate_lfm_chirp(
        config.chirp_samples,
        sample_rate_hz=config.sample_rate_hz,
        bandwidth_hz=config.bandwidth_hz,
        window=config.window,
    )
    echo, delays, amps, sigma = simulate_echo(
        chirp, config.targets, n=config.n, snr_db=config.snr_db, seed=config.seed
    )
    if config.adc is not None:
        # ADC owns bit-depth quantization only; radar AWGN is already injected,
        # so force the ADC's own noise off to avoid double-counting.
        adc_cfg = dataclasses.replace(config.adc, snr_db=None)
        echo = simulate_iq_adc(echo, adc_cfg).samples

    padded = np.zeros(config.n, dtype=np.complex128)
    padded[: chirp.size] = chirp
    return RadarScene(
        config=config,
        reference_chirp=padded,
        echo_samples=echo,
        target_delays=delays,
        target_amplitudes=amps,
        noise_sigma=sigma,
    )


def validate_radar_scene_config(config: RadarSceneConfig) -> None:
    """Validate FFT size, window, and that the echo stays linear (no wrap)."""
    n = config.n
    if n < _MIN_N or n > _MAX_N or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of two in [{_MIN_N}, {_MAX_N}], got {n}")
    if config.window not in _WINDOWS:
        raise ValueError(f"window must be one of {_WINDOWS}, got {config.window!r}")
    if not config.targets:
        raise ValueError("at least one target is required")
    ns = config.chirp_samples
    if ns < 1 or ns >= n:
        raise ValueError(f"chirp_samples must be in [1, n), got {ns} (n={n})")
    max_delay = max(int(t.delay_bins) for t in config.targets)
    min_delay = min(int(t.delay_bins) for t in config.targets)
    if min_delay < 0:
        raise ValueError("target delay_bins must be >= 0")
    if max_delay + ns > n:
        raise ValueError(
            f"chirp_samples + max delay ({ns}+{max_delay}) exceeds n={n}; "
            "echo would wrap (circular correlation). Increase n or reduce delay."
        )


def radar_scene_config_from_mapping(data: Mapping[str, Any]) -> RadarSceneConfig:
    """Build a :class:`RadarSceneConfig` from a parsed JSON mapping."""
    targets = tuple(
        TargetSpec(delay_bins=int(t["delay_bins"]), amplitude_db=float(t.get("amplitude_db", 0.0)))
        for t in data["targets"]
    )
    adc_obj = data.get("adc")
    adc = None
    if adc_obj is not None:
        adc = ADCConfig(
            bit_depth=int(adc_obj.get("bit_depth", 12)),
            full_scale=float(adc_obj.get("full_scale", 1.0)),
            seed=int(adc_obj.get("seed", 20260521)),
        )
    acc_obj = data.get("acceptance") or {}
    acceptance = AcceptanceSpec(
        evm_max=float(acc_obj.get("evm_max", 0.02)),
        pslr_delta_max_db=float(acc_obj.get("pslr_delta_max_db", 2.0)),
        peak_bin_exact=bool(acc_obj.get("peak_bin_exact", True)),
    )
    snr = data.get("snr_db")
    return RadarSceneConfig(
        scenario=str(data["scenario"]),
        n=int(data["n"]),
        chirp_samples=int(data["chirp_samples"]),
        sample_rate_hz=float(data["sample_rate_hz"]),
        bandwidth_hz=float(data["bandwidth_hz"]),
        targets=targets,
        snr_db=None if snr is None else float(snr),
        window=str(data.get("window", "rect")),
        seed=int(data.get("seed", 20260528)),
        adc=adc,
        acceptance=acceptance,
    )


def load_radar_scene_config(path: str | Path) -> RadarSceneConfig:
    """Load and parse a radar scene fixture JSON file."""
    with open(path) as f:
        data = json.load(f)
    return radar_scene_config_from_mapping(data)
