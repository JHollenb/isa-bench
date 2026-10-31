"""SPF-resident sparse radar runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from spf.native import spf32 as c_spf32
from spf.runtime import RuntimeTelemetry

DetectionMode = Literal["threshold", "cfar"]


@dataclass(frozen=True, slots=True)
class RadarRuntimeResult:
    """Sparse radar product output."""

    detections: npt.NDArray[np.uint8]
    doppler_theta: npt.NDArray[np.uint16]
    confidence: npt.NDArray[np.uint8]
    fallback_mask: npt.NDArray[np.uint8]
    fallback_indices: npt.NDArray[np.uint16] | None
    fallback_bins: npt.NDArray[np.uint16] | None
    fallback_spectra: npt.NDArray[np.uint32] | None
    telemetry: RuntimeTelemetry


def sparse_doppler(
    slow_time_words: npt.ArrayLike,
    *,
    n_range_bins: int,
    n_chirps: int,
    threshold_log2_power_q78: int = 0,
    detection_mode: DetectionMode = "threshold",
    range_profile_words: npt.ArrayLike | None = None,
    cfar_threshold_offset_q78: int = 0,
    cfar_guard: int = 0,
    cfar_train: int = 1,
    cfar_rank: int = 1,
    fallback_confidence_threshold: int = 1,
    enable_fb7_fallback: bool = False,
    fallback_candidate_bins: npt.ArrayLike | None = None,
    fb7_mode: str | int = "balanced",
) -> RadarRuntimeResult:
    """Run sparse phase-first radar detection and Doppler estimation.

    The detector gates one range-profile word per range bin. If no explicit
    range profile is provided, the first chirp from each range bin is used.
    Doppler is estimated for all range bins, while ``fallback_mask`` marks
    detections whose phase-difference confidence is below the threshold.
    """
    cube = np.ascontiguousarray(np.asarray(slow_time_words), dtype=np.uint32).ravel()
    expected = int(n_range_bins) * int(n_chirps)
    if cube.size != expected:
        raise ValueError(f"slow_time_words length {cube.size} != n_range_bins*n_chirps {expected}")
    if n_range_bins <= 0 or n_chirps <= 1:
        raise ValueError("n_range_bins must be positive and n_chirps must be greater than 1")

    if range_profile_words is None:
        gate_words = cube.reshape(n_range_bins, n_chirps)[:, 0]
    else:
        gate_words = np.ascontiguousarray(np.asarray(range_profile_words), dtype=np.uint32).ravel()
        if gate_words.size != n_range_bins:
            raise ValueError(
                f"range_profile_words length {gate_words.size} != n_range_bins {n_range_bins}"
            )

    if detection_mode == "threshold":
        detections = c_spf32.logpower_threshold(gate_words, threshold_log2_power_q78)
        detect_stage = "logpower_threshold"
    elif detection_mode == "cfar":
        detections = c_spf32.log_os_cfar(
            gate_words,
            threshold_offset_q78=cfar_threshold_offset_q78,
            guard=cfar_guard,
            train=cfar_train,
            rank=cfar_rank,
        )
        detect_stage = "log_os_cfar"
    else:
        raise ValueError(f"unknown detection_mode {detection_mode!r}")

    doppler, confidence = c_spf32.doppler_phase_diff_estimate(
        cube, n_range_bins=n_range_bins, n_chirps=n_chirps
    )
    fallback_mask = (
        (detections != 0) & (confidence < int(fallback_confidence_threshold))
    ).astype(np.uint8)
    fallback_bins = None
    fallback_spectra = None
    fallback_indices_out = None
    fallback_stage: tuple[str, ...] = ()
    if enable_fb7_fallback:
        if fallback_candidate_bins is None:
            fallback_bins = np.arange(n_chirps, dtype=np.uint16)
        else:
            fallback_bins = np.ascontiguousarray(
                np.asarray(fallback_candidate_bins), dtype=np.uint16
            ).ravel()
        rows = cube.reshape(n_range_bins, n_chirps)
        fallback_indices = np.flatnonzero(fallback_mask)
        fallback_indices_out = fallback_indices.astype(np.uint16)
        fallback_spectra = np.empty(
            (fallback_indices.size, fallback_bins.size), dtype=np.uint32
        )
        for out_idx, range_idx in enumerate(fallback_indices):
            fallback_spectra[out_idx] = c_spf32.candidate_fft_bins_fb7(
                rows[int(range_idx)], fallback_bins, fb7_mode
            )
        fallback_stage = ("candidate_fft_bins_fb7",)
    return RadarRuntimeResult(
        detections=detections,
        doppler_theta=doppler,
        confidence=confidence,
        fallback_mask=fallback_mask,
        fallback_indices=fallback_indices_out,
        fallback_bins=fallback_bins,
        fallback_spectra=fallback_spectra,
        telemetry=RuntimeTelemetry(
            backend="native",
            stages=(detect_stage, "doppler_phase_diff_estimate", *fallback_stage),
            details={
                "n_range_bins": n_range_bins,
                "n_chirps": n_chirps,
                "fallback_count": int(np.count_nonzero(fallback_mask)),
                "fb7_mode": fb7_mode,
            },
        ),
    )
