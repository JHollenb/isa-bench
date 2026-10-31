"""Compatibility SISO OFDM path over the SPF resource grid.

This module intentionally uses conventional NumPy IFFT/FFT transform islands.
It proves resource-grid, channel/equalization, and demap semantics before the
resident spectral-DAC path tries to remove boundary transforms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from spf.dsp.metrics import evm
from spf.dsp.modulation import psk_demodulate_spf, psk_modulate_spf, qam16_modulate_spf
from spf.dsp.ofdm_grid import OFDMBinClass, OFDMResourceGrid
from spf.native import spf32 as c_spf32
from spf.tensor import SPF32_WORD_ZERO, decode_words, encode_words, spf_mul_words

Modulation = Literal["qpsk", "qam16"]


@dataclass(frozen=True, slots=True)
class SISOCompatOFDMResult:
    """Outputs and telemetry from the compatibility OFDM round trip."""

    tx_bits: npt.NDArray[np.uint8]
    rx_bits: npt.NDArray[np.uint8]
    tx_grid: OFDMResourceGrid
    rx_grid: OFDMResourceGrid
    equalized_words: npt.NDArray[np.uint32]
    confidence: npt.NDArray[np.uint8] | None
    bit_error_rate: float
    evm: float
    stages: tuple[str, ...]
    demap: str


def default_active_subcarriers(n_fft: int) -> npt.NDArray[np.uint32]:
    """Return all non-DC subcarriers for a compact SISO smoke frame."""
    if n_fft < 4 or n_fft & (n_fft - 1):
        raise ValueError(f"n_fft must be a power of two >= 4, got {n_fft}")
    return np.asarray([k for k in range(n_fft) if k != 0], dtype=np.uint32)


def grid_from_bits(
    bits: npt.ArrayLike,
    *,
    modulation: Modulation,
    n_fft: int,
    active_subcarriers: npt.ArrayLike | None = None,
) -> OFDMResourceGrid:
    """Map bits into a one-frame, one-symbol SPF OFDM resource grid."""
    tx_bits = np.ascontiguousarray(np.asarray(bits), dtype=np.uint8).ravel()
    active = _active_indices(n_fft, active_subcarriers)
    symbols = _modulate_bits(tx_bits, modulation)
    if symbols.size != active.size:
        raise ValueError(
            f"{modulation} bit stream maps to {symbols.size} symbols, but active grid has {active.size} subcarriers"
        )
    shape = (1, 1, n_fft, 1)
    words = np.full(shape, SPF32_WORD_ZERO, dtype=np.uint32)
    classes = np.full(shape, int(OFDMBinClass.GUARD), dtype=np.int16)
    classes[:, :, 0, :] = int(OFDMBinClass.DC_NULL)
    words[0, 0, active, 0] = symbols
    classes[0, 0, active, 0] = int(OFDMBinClass.DATA)
    return OFDMResourceGrid.from_trusted(words, classes)


def apply_diagonal_words(
    grid: OFDMResourceGrid,
    weights: npt.ArrayLike,
) -> OFDMResourceGrid:
    """Apply an elementwise SPF diagonal operator to active data/pilot bins.

    ``weights`` is interpreted in logical grid coordinates.
    """
    weight_words = np.ascontiguousarray(np.asarray(weights), dtype=np.uint32).reshape(grid.shape)
    logical_words = grid.logical_words()
    logical_classes = grid.logical_bin_class()
    out_logical = logical_words.copy()
    active = np.isin(
        logical_classes,
        np.asarray([int(OFDMBinClass.DATA), int(OFDMBinClass.PILOT)], dtype=np.int16),
    )
    out_logical[active] = spf_mul_words(out_logical[active], weight_words[active])
    out = out_logical
    for axis, offset in enumerate(grid.origin):
        if offset:
            out = np.roll(out, shift=offset, axis=axis)
    # ``grid`` already passed validation when it was constructed; this operator
    # only changes active words and preserves the physical bin-class layout.
    return OFDMResourceGrid.from_trusted(
        out,
        grid.bin_class.copy(),
        origin=grid.origin,
        reducer_id=grid.reducer_id,
    )


def apply_diagonal_active_words(
    grid: OFDMResourceGrid,
    active_subcarriers: npt.ArrayLike,
    h_words: npt.ArrayLike,
    *,
    trusted: bool = False,
) -> OFDMResourceGrid:
    """Apply diagonal SPF weights only at logical active subcarriers.

    ``h_words`` may be full-FFT length or exactly ``len(active_subcarriers)``.
    """
    if not trusted:
        grid.validate()
    active = np.ascontiguousarray(np.asarray(active_subcarriers), dtype=np.uint32).ravel()
    if active.size == 0:
        raise ValueError("active_subcarriers must not be empty")
    if np.any(active >= grid.shape[2]):
        raise ValueError("active_subcarriers contains an index outside grid")
    if active.size and not np.all(active[1:] > active[:-1]):
        raise ValueError("active_subcarriers must be strictly increasing")
    weights = np.ascontiguousarray(np.asarray(h_words), dtype=np.uint32).ravel()
    if weights.size == grid.shape[2]:
        active_weights = weights[active]
    elif weights.size == active.size:
        active_weights = weights
    else:
        raise ValueError(
            f"h_words length {weights.size} must equal n_fft={grid.shape[2]} or active_subcarriers length={active.size}"
        )
    if trusted and grid.origin == (0, 0, 0, 0):
        # Hot-path: zero-origin grid — direct word indexing, no logical roll or isin scan.
        active_i = active.astype(np.intp)
        out = grid.words.copy()
        for symbol in range(grid.shape[1]):
            out[0, symbol, active_i, 0] = spf_mul_words(
                out[0, symbol, active_i, 0],
                active_weights,
            )
        return OFDMResourceGrid.from_trusted(
            out,
            grid.bin_class.copy(),
            origin=grid.origin,
            reducer_id=grid.reducer_id,
        )
    # Shifted-grid path: operate in logical frame then re-apply origin roll.
    logical_words = grid.logical_words()
    logical_classes = grid.logical_bin_class()
    out_logical = logical_words.copy()
    active_classes = np.asarray([int(OFDMBinClass.DATA), int(OFDMBinClass.PILOT)], dtype=np.int16)
    for idx, subcarrier in enumerate(active):
        sc = int(subcarrier)
        mask = np.isin(logical_classes[:, :, sc, :], active_classes)
        out_logical[:, :, sc, :][mask] = spf_mul_words(
            out_logical[:, :, sc, :][mask],
            np.full(np.count_nonzero(mask), active_weights[idx], dtype=np.uint32),
        )
    out = out_logical
    for axis, offset in enumerate(grid.origin):
        if offset:
            out = np.roll(out, shift=offset, axis=axis)
    return OFDMResourceGrid.from_trusted(
        out,
        grid.bin_class.copy(),
        origin=grid.origin,
        reducer_id=grid.reducer_id,
    )


def run_siso_compat_ofdm(
    bits: npt.ArrayLike,
    *,
    modulation: Modulation,
    n_fft: int,
    channel_h: npt.ArrayLike | None = None,
    active_subcarriers: npt.ArrayLike | None = None,
    qam16_threshold_L: int = 256,
) -> SISOCompatOFDMResult:
    """Run one compatibility OFDM frame through SPF spectral grid semantics."""
    tx_grid = grid_from_bits(
        bits,
        modulation=modulation,
        n_fft=n_fft,
        active_subcarriers=active_subcarriers,
    )
    active = tx_grid.active_subcarriers()
    h = _channel(n_fft, channel_h, active)
    tx_freq = decode_words(tx_grid.words.reshape(n_fft))
    tx_time = np.fft.ifft(tx_freq)
    rx_time = np.fft.ifft(h * np.fft.fft(tx_time))
    rx_freq = np.fft.fft(rx_time)
    rx_words = np.full((1, 1, n_fft, 1), SPF32_WORD_ZERO, dtype=np.uint32)
    rx_words[0, 0, active, 0] = encode_words(rx_freq[active])
    rx_grid = OFDMResourceGrid.from_trusted(rx_words, tx_grid.bin_class.copy())
    h_inv_words = encode_words(1.0 / h[active])
    y_words = rx_grid.words[0, 0, active, 0]
    if modulation == "qpsk":
        equalized = spf_mul_words(y_words, h_inv_words)
        rx_bits = psk_demodulate_spf(equalized, 4)
        confidence = None
        demap = "qpsk_phase"
    else:
        rx_bits, confidence = c_spf32.qam16_logproj_eq_demod_fused(
            y_words,
            h_inv_words,
            threshold_L=qam16_threshold_L,
        )
        equalized = spf_mul_words(y_words, h_inv_words)
        demap = "qam16_logproj"
    tx_bits = np.ascontiguousarray(np.asarray(bits), dtype=np.uint8).ravel()
    tx_symbols = decode_words(tx_grid.words[0, 0, active, 0])
    rx_symbols = decode_words(equalized)
    return SISOCompatOFDMResult(
        tx_bits=tx_bits,
        rx_bits=rx_bits,
        tx_grid=tx_grid,
        rx_grid=rx_grid,
        equalized_words=equalized,
        confidence=confidence,
        bit_error_rate=float(np.count_nonzero(tx_bits != rx_bits)) / float(tx_bits.size),
        evm=evm(tx_symbols, rx_symbols),
        stages=(
            "bits_to_spf_lookup",
            "spf_resource_grid",
            "compat_ifft_island",
            "channel_model",
            "compat_fft_island",
            "spf_equalize",
            demap,
        ),
        demap=demap,
    )


def _active_indices(n_fft: int, active_subcarriers: npt.ArrayLike | None) -> npt.NDArray[np.uint32]:
    active = (
        default_active_subcarriers(n_fft)
        if active_subcarriers is None
        else np.ascontiguousarray(np.asarray(active_subcarriers), dtype=np.uint32).ravel()
    )
    if active.size == 0:
        raise ValueError("active_subcarriers must not be empty")
    if np.any(active >= n_fft):
        raise ValueError("active_subcarriers contains an index outside n_fft")
    if np.any(active == 0):
        raise ValueError("DC subcarrier 0 is reserved for DC_NULL")
    if active.size and not np.all(active[1:] > active[:-1]):
        raise ValueError("active_subcarriers must be strictly increasing")
    return active


def _modulate_bits(bits: npt.NDArray[np.uint8], modulation: Modulation) -> npt.NDArray[np.uint32]:
    if modulation == "qpsk":
        return psk_modulate_spf(bits, 4)
    if modulation == "qam16":
        return qam16_modulate_spf(bits)
    raise ValueError(f"unsupported modulation {modulation!r}")


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


__all__ = [
    "Modulation",
    "SISOCompatOFDMResult",
    "apply_diagonal_active_words",
    "apply_diagonal_words",
    "default_active_subcarriers",
    "grid_from_bits",
    "run_siso_compat_ofdm",
]
