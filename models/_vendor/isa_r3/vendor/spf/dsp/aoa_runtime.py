"""SPF-resident AoA/localization runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spf.native import spf32 as c_spf32
from spf.runtime import RuntimeTelemetry


@dataclass(frozen=True, slots=True)
class AoARuntimeResult:
    """Pair/interferometric AoA product output."""

    phase_delta: npt.NDArray[np.uint16]
    angle: npt.NDArray[np.int16]
    valid: npt.NDArray[np.uint8]
    confidence: npt.NDArray[np.uint8]
    telemetry: RuntimeTelemetry


def estimate_pair(
    ant_a_words: npt.ArrayLike,
    ant_b_words: npt.ArrayLike,
    theta_to_angle_lut: npt.ArrayLike,
    *,
    calibration_phase_offset: int | npt.ArrayLike = 0,
    min_confidence: int = 1,
) -> AoARuntimeResult:
    """Estimate AoA from one antenna pair without decoded Cartesian products."""
    phase_delta, confidence = c_spf32.aoa_pair_phase(ant_a_words, ant_b_words)
    calibration = np.asarray(calibration_phase_offset)
    if calibration.shape != ():
        calibration = np.broadcast_to(calibration.astype(np.uint16), phase_delta.shape)
    phase_delta = (
        phase_delta.astype(np.uint32) + calibration.astype(np.uint32)
    ).astype(np.uint16)
    angle, valid = c_spf32.aoa_lut(phase_delta, confidence, theta_to_angle_lut)
    valid = (valid & (confidence >= int(min_confidence))).astype(np.uint8)
    return AoARuntimeResult(
        phase_delta=phase_delta,
        angle=angle,
        valid=valid,
        confidence=confidence,
        telemetry=RuntimeTelemetry(
            backend="native",
            stages=("aoa_pair_phase", "aoa_lut"),
            details={
                "calibration_phase_offset": (
                    int(calibration) & 0xFFFF if calibration.shape == () else "array"
                ),
                "min_confidence": int(min_confidence),
                "valid_count": int(np.count_nonzero(valid)),
            },
        ),
    )
