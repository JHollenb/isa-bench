"""Rank-1 SPF beamspace sOFDM helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_grid import OFDMResourceGrid
from spf.tensor import SPFError, decode_words, encode_words, spf_mul_words


@dataclass(frozen=True, slots=True)
class BeamCodebook:
    """Packed SPF beam weights indexed by codebook entry."""

    weights: npt.NDArray[np.uint32]

    def __post_init__(self) -> None:
        arr = np.ascontiguousarray(np.asarray(self.weights), dtype=np.uint32)
        if arr.ndim != 3:
            raise SPFError("BeamCodebook weights must have shape [beam, antenna, subcarrier]")
        object.__setattr__(self, "weights", arr)

    def select(self, beam_id: int) -> npt.NDArray[np.uint32]:
        if beam_id < 0 or beam_id >= self.weights.shape[0]:
            raise SPFError("beam_id out of range")
        return np.ascontiguousarray(self.weights[beam_id], dtype=np.uint32)


def rank1_beamform_grid(
    stream_grid: OFDMResourceGrid,
    beam_weights: npt.ArrayLike,
) -> OFDMResourceGrid:
    """Apply ``X_ant[a,k] = S[k] * W[a,k]`` without digital antenna summation."""
    stream_grid.validate()
    if stream_grid.shape[3] != 1:
        raise SPFError("rank1 beamforming expects a single stream input grid")
    weights = np.ascontiguousarray(np.asarray(beam_weights), dtype=np.uint32)
    if weights.ndim != 2:
        raise SPFError("beam_weights must have shape [antenna, subcarrier]")
    n_ant, n_sub = weights.shape
    if n_sub != stream_grid.shape[2]:
        raise SPFError("beam_weights subcarrier count must match grid")
    logical_w = stream_grid.logical_words()
    logical_c = stream_grid.logical_bin_class()
    out = np.repeat(logical_w, n_ant, axis=3)
    classes = np.repeat(logical_c, n_ant, axis=3)
    for ant in range(n_ant):
        out[:, :, :, ant] = spf_mul_words(out[:, :, :, ant], weights[ant].reshape(1, 1, n_sub))
    return OFDMResourceGrid(
        out,
        classes,
        origin=(0, 0, 0, 0),
        reducer_id=stream_grid.reducer_id,
    )


def effective_channel(
    channel_ant: npt.ArrayLike,
    beam_weights: npt.ArrayLike,
) -> npt.NDArray[np.complex128]:
    """Reference setup-side effective channel ``sum_a H[a,k] * W[a,k]``."""
    h = np.ascontiguousarray(np.asarray(channel_ant), dtype=np.complex128)
    w = decode_words(np.asarray(beam_weights, dtype=np.uint32))
    if h.shape != w.shape:
        raise SPFError("channel_ant and beam_weights must have matching [antenna, subcarrier]")
    out: npt.NDArray[np.complex128] = np.asarray(np.sum(h * w, axis=0), dtype=np.complex128)
    return out


def dft_beam_codebook(n_antennas: int, n_subcarriers: int) -> BeamCodebook:
    """Create a small phase-only DFT beam codebook for tests and smoke paths."""
    if n_antennas <= 0 or n_subcarriers <= 0:
        raise SPFError("n_antennas and n_subcarriers must be positive")
    beams = np.empty((n_antennas, n_antennas, n_subcarriers), dtype=np.complex128)
    ant = np.arange(n_antennas, dtype=np.float64).reshape(n_antennas, 1)
    for beam in range(n_antennas):
        phase = 2.0 * np.pi * beam * ant / n_antennas
        beams[beam] = np.exp(1j * phase)
    return BeamCodebook(encode_words(beams))


__all__ = [
    "BeamCodebook",
    "dft_beam_codebook",
    "effective_channel",
    "rank1_beamform_grid",
]
