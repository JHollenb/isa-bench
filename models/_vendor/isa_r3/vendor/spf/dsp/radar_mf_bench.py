"""Repeated-run bench harness for the radar matched-filter arms.

Mirrors :mod:`spf.dsp.ofdm_bench`: per-stage :class:`StageStats`
(min/p50/p95/mean/std) over warmup+iterations, split into four explicit modes
so the claim ledger never mixes diagnostic and hot-path timings:

- :func:`run_compare_diagnostic` — SPF + fp arms, full accuracy metrics
- :func:`run_compare_hot_path`   — SPF + fp arms, timing only
- :func:`run_spf_diagnostic`     — SPF arm + accuracy
- :func:`run_spf_hot_path`       — SPF arm timing only (headline latency)

Per-stage SPF breakdown: ``encode`` (Cartesian→SPF boundary), ``fft``,
``mf_ifft`` (the fused ``spectral_matched_filter_fb7`` spectral multiply +
IFFT), ``detect`` (log-power + OS-CFAR, no decode), and ``total_hotpath``
(fft+mf_ifft+detect — the streaming per-pulse cost, encode excluded as a
boundary cost numpy does not pay either). The matched-filter reference spectrum
is filter-design-time prep, computed once outside every timed loop.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from spf.dsp.ofdm_bench import StageStats
from spf.dsp.radar_mf import (
    DetectorConfig,
    _spf_detect,
    _spf_matched_filter_core,
    fp_matched_filter_reference,
    log_power_q78_from_complex,
    log_power_q78_from_words,
    os_cfar_q78,
    run_radar_mf_compare,
    spf_matched_filter_reference,
)
from spf.dsp.radar_mf_sim import RadarScene
from spf.fft import spf32_fft
from spf.native._lib import LIBSPF_AVAILABLE
from spf.tensor import encode_words

_perf = time.perf_counter_ns


def _agg(samples: list[int], *, iterations: int, warmup: int) -> StageStats:
    arr = np.asarray(samples, dtype=np.int64)
    return StageStats(
        min_ns=int(arr.min()),
        p50_ns=int(np.percentile(arr, 50)),
        p95_ns=int(np.percentile(arr, 95)),
        mean_ns=float(arr.mean()),
        std_ns=float(arr.std(ddof=0)),
        iterations=iterations,
        warmup=warmup,
    )


@dataclass(frozen=True, slots=True)
class RadarMFStats:
    """Aggregate timing + accuracy for one radar matched-filter bench mode."""

    scenario: str
    n: int
    mode: str
    iterations: int
    warmup: int
    diagnostics: bool
    include_baseline: bool
    backend: str
    # SPF per-stage timings
    encode_ns: StageStats
    fft_ns: StageStats
    mf_ifft_ns: StageStats
    detect_ns: StageStats
    total_hotpath_ns: StageStats
    # fp baseline timings (None when include_baseline=False)
    baseline_fft_ns: StageStats | None
    baseline_mf_ifft_ns: StageStats | None
    baseline_detect_ns: StageStats | None
    baseline_total_ns: StageStats | None
    # accuracy (None unless diagnostics=True)
    peak_bin_match: bool | None = None
    peak_bin_diff: int | None = None
    range_profile_evm: float | None = None
    range_profile_nmse_db: float | None = None
    spf_pslr_db: float | None = None
    fp_pslr_db: float | None = None
    spf_peak_snr_db: float | None = None
    fp_peak_snr_db: float | None = None

    def to_dict(self) -> dict[str, object]:
        def _sd(s: StageStats | None) -> dict[str, float | int] | None:
            return None if s is None else s.to_dict()

        return {
            "scenario": self.scenario,
            "n": self.n,
            "mode": self.mode,
            "iterations": self.iterations,
            "warmup": self.warmup,
            "diagnostics": self.diagnostics,
            "include_baseline": self.include_baseline,
            "backend": self.backend,
            "encode_ns": _sd(self.encode_ns),
            "fft_ns": _sd(self.fft_ns),
            "mf_ifft_ns": _sd(self.mf_ifft_ns),
            "detect_ns": _sd(self.detect_ns),
            "total_hotpath_ns": _sd(self.total_hotpath_ns),
            "baseline_fft_ns": _sd(self.baseline_fft_ns),
            "baseline_mf_ifft_ns": _sd(self.baseline_mf_ifft_ns),
            "baseline_detect_ns": _sd(self.baseline_detect_ns),
            "baseline_total_ns": _sd(self.baseline_total_ns),
            "accuracy": {
                "peak_bin_match": self.peak_bin_match,
                "peak_bin_diff": self.peak_bin_diff,
                "range_profile_evm": self.range_profile_evm,
                "range_profile_nmse_db": self.range_profile_nmse_db,
                "spf_pslr_db": self.spf_pslr_db,
                "fp_pslr_db": self.fp_pslr_db,
                "spf_peak_snr_db": self.spf_peak_snr_db,
                "fp_peak_snr_db": self.fp_peak_snr_db,
            },
        }


def _spf_stage_sample(
    scene: RadarScene,
    h_conj: np.ndarray,
    fb7_mode: str,
    detector: DetectorConfig,
) -> dict[str, int]:
    out: dict[str, int] = {}
    t0 = _perf()
    x_words = encode_words(scene.echo_samples)
    out["encode"] = _perf() - t0
    t0 = _perf()
    x_fft = spf32_fft.fft(x_words)
    out["fft"] = _perf() - t0
    t0 = _perf()
    y_words = _spf_matched_filter_core(x_fft, h_conj, fb7_mode)
    out["mf_ifft"] = _perf() - t0
    t0 = _perf()
    log_power_q78_from_words(y_words)
    _spf_detect(y_words, detector)
    out["detect"] = _perf() - t0
    out["total_hotpath"] = out["fft"] + out["mf_ifft"] + out["detect"]
    return out


def _fp_stage_sample(
    scene: RadarScene, fp_ref: np.ndarray, detector: DetectorConfig
) -> dict[str, int]:
    out: dict[str, int] = {}
    t0 = _perf()
    x_fft = np.fft.fft(scene.echo_samples)
    out["fft"] = _perf() - t0
    t0 = _perf()
    y = np.fft.ifft(x_fft * fp_ref)
    out["mf_ifft"] = _perf() - t0
    t0 = _perf()
    lp = log_power_q78_from_complex(y)
    os_cfar_q78(lp, detector)
    out["detect"] = _perf() - t0
    out["total"] = out["fft"] + out["mf_ifft"] + out["detect"]
    return out


def _collect(
    sample_fn: Callable[[], dict[str, int]], keys: tuple[str, ...], *, iterations: int, warmup: int
) -> dict[str, StageStats]:
    for _ in range(warmup):
        sample_fn()
    series: dict[str, list[int]] = {k: [] for k in keys}
    for _ in range(iterations):
        s = sample_fn()
        for k in keys:
            series[k].append(s[k])
    return {k: _agg(series[k], iterations=iterations, warmup=warmup) for k in keys}


def _run_mode(
    scene: RadarScene,
    *,
    mode: str,
    iterations: int,
    warmup: int,
    diagnostics: bool,
    include_baseline: bool,
    fb7_mode: str = "balanced",
    detector: DetectorConfig | None = None,
) -> RadarMFStats:
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    det = detector or DetectorConfig()
    h_conj = spf_matched_filter_reference(scene.reference_chirp)  # untimed prep

    spf_stats = _collect(
        lambda: _spf_stage_sample(scene, h_conj, fb7_mode, det),
        ("encode", "fft", "mf_ifft", "detect", "total_hotpath"),
        iterations=iterations,
        warmup=warmup,
    )

    base: dict[str, StageStats] | None = None
    if include_baseline:
        fp_ref = fp_matched_filter_reference(scene.reference_chirp)  # untimed prep
        base = _collect(
            lambda: _fp_stage_sample(scene, fp_ref, det),
            ("fft", "mf_ifft", "detect", "total"),
            iterations=iterations,
            warmup=warmup,
        )

    acc = (
        run_radar_mf_compare(
            scene, fb7_mode=fb7_mode, diagnostics=True,
            include_baseline=include_baseline, detector=det,
        )
        if diagnostics
        else None
    )

    return RadarMFStats(
        scenario=scene.config.scenario,
        n=scene.config.n,
        mode=mode,
        iterations=iterations,
        warmup=warmup,
        diagnostics=diagnostics,
        include_baseline=include_baseline,
        backend="spf-native" if LIBSPF_AVAILABLE else "spf-python",
        encode_ns=spf_stats["encode"],
        fft_ns=spf_stats["fft"],
        mf_ifft_ns=spf_stats["mf_ifft"],
        detect_ns=spf_stats["detect"],
        total_hotpath_ns=spf_stats["total_hotpath"],
        baseline_fft_ns=None if base is None else base["fft"],
        baseline_mf_ifft_ns=None if base is None else base["mf_ifft"],
        baseline_detect_ns=None if base is None else base["detect"],
        baseline_total_ns=None if base is None else base["total"],
        peak_bin_match=None if acc is None else acc.peak_bin_match,
        peak_bin_diff=None if acc is None else acc.peak_bin_diff,
        range_profile_evm=None if acc is None else acc.range_profile_evm,
        range_profile_nmse_db=None if acc is None else acc.range_profile_nmse_db,
        spf_pslr_db=None if acc is None else acc.spf.pslr_db,
        fp_pslr_db=None if acc is None or acc.fp is None else acc.fp.pslr_db,
        spf_peak_snr_db=None if acc is None else acc.spf.peak_snr_db,
        fp_peak_snr_db=None if acc is None or acc.fp is None else acc.fp.peak_snr_db,
    )


def run_compare_diagnostic(
    scene: RadarScene, *, iterations: int = 10, warmup: int = 2, **kw: object
) -> RadarMFStats:
    return _run_mode(
        scene, mode="compare_diagnostic", iterations=iterations, warmup=warmup,
        diagnostics=True, include_baseline=True, **kw,  # type: ignore[arg-type]
    )


def run_compare_hot_path(
    scene: RadarScene, *, iterations: int = 200, warmup: int = 5, **kw: object
) -> RadarMFStats:
    return _run_mode(
        scene, mode="compare_hot_path", iterations=iterations, warmup=warmup,
        diagnostics=False, include_baseline=True, **kw,  # type: ignore[arg-type]
    )


def run_spf_diagnostic(
    scene: RadarScene, *, iterations: int = 10, warmup: int = 2, **kw: object
) -> RadarMFStats:
    return _run_mode(
        scene, mode="spf_diagnostic", iterations=iterations, warmup=warmup,
        diagnostics=True, include_baseline=False, **kw,  # type: ignore[arg-type]
    )


def run_spf_hot_path(
    scene: RadarScene, *, iterations: int = 200, warmup: int = 5, **kw: object
) -> RadarMFStats:
    return _run_mode(
        scene, mode="spf_hot_path", iterations=iterations, warmup=warmup,
        diagnostics=False, include_baseline=False, **kw,  # type: ignore[arg-type]
    )


MODE_RUNNERS: dict[str, Callable[..., RadarMFStats]] = {
    "compare_diagnostic": run_compare_diagnostic,
    "compare_hot_path": run_compare_hot_path,
    "spf_diagnostic": run_spf_diagnostic,
    "spf_hot_path": run_spf_hot_path,
}
