"""SPF-resident beamforming/MIMO composition helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spf.native import spf32 as c_spf32
from spf.runtime import RuntimeTelemetry


@dataclass(frozen=True, slots=True)
class BeamRuntimeResult:
    """Beamforming product output."""

    beam_words: npt.NDArray[np.uint32]
    log_power_q78: npt.NDArray[np.int32] | None
    telemetry: RuntimeTelemetry


def beamform_reduce(
    antenna_words: npt.ArrayLike,
    *,
    gain_l_offsets: npt.ArrayLike | None = None,
    phase_offsets: npt.ArrayLike | None = None,
    mode: str | int = "balanced",
    emit_log_power: bool = True,
) -> BeamRuntimeResult:
    """Apply per-antenna gain/phase weights and FB7-reduce each sample."""
    words = np.ascontiguousarray(np.asarray(antenna_words), dtype=np.uint32)
    if words.ndim != 2:
        raise ValueError(
            f"antenna_words must have shape (n_antennas, n_samples), got {words.shape}"
        )
    n_antennas, n_samples = words.shape
    gain = (
        np.zeros(n_antennas, dtype=np.int16)
        if gain_l_offsets is None
        else np.ascontiguousarray(np.asarray(gain_l_offsets), dtype=np.int16).ravel()
    )
    phase = (
        np.zeros(n_antennas, dtype=np.uint16)
        if phase_offsets is None
        else np.ascontiguousarray(np.asarray(phase_offsets), dtype=np.uint16).ravel()
    )
    if gain.size != n_antennas:
        raise ValueError(f"gain_l_offsets length {gain.size} != n_antennas {n_antennas}")
    if phase.size != n_antennas:
        raise ValueError(f"phase_offsets length {phase.size} != n_antennas {n_antennas}")

    beam = c_spf32.beamform_fused(
        words, gain_l_offsets=gain, phase_offsets=phase, mode=mode
    )

    log_power = c_spf32.log_power(beam) if emit_log_power else None
    return BeamRuntimeResult(
        beam_words=beam,
        log_power_q78=log_power,
        telemetry=RuntimeTelemetry(
            backend="native",
            stages=("beamform_fused",),
            details={
                "n_antennas": int(n_antennas),
                "n_samples": int(n_samples),
                "fb7_mode": mode,
            },
        ),
    )
