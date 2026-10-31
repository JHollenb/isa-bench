"""DSP measurement metrics: BER and EVM.

These are not in ``benchmarks/dsp/_helpers.py`` — they were computed inline
in each demo runner.  Promoting them here makes them reusable across demos and
examples without copy-paste.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "ber",
    "evm",
    "evm_db",
    "max_magnitude_relative_error",
    "max_phase_error",
    "ser_from_bits",
]


def ber(
    tx_bits: npt.NDArray[np.uint8],
    rx_bits: npt.NDArray[np.uint8],
) -> float:
    """Bit error rate.

    Parameters
    ----------
    tx_bits, rx_bits : ndarray[uint8]
        Transmitted and received bit streams; must be the same length.

    Returns
    -------
    float
        Fraction of differing bits in ``[0, 1]``.
    """
    tx = np.asarray(tx_bits, dtype=np.uint8)
    rx = np.asarray(rx_bits, dtype=np.uint8)
    if tx.shape != rx.shape:
        raise ValueError(f"Shape mismatch: tx={tx.shape}, rx={rx.shape}")
    n = tx.size
    if n == 0:
        return 0.0
    return float(np.count_nonzero(tx != rx)) / n


def evm(
    tx_symbols: npt.NDArray[np.complex128],
    rx_symbols: npt.NDArray[np.complex128],
) -> float:
    """Error vector magnitude (EVM) as a linear ratio.

    EVM = sqrt( mean(|rx - tx|²) / mean(|tx|²) ).

    Parameters
    ----------
    tx_symbols : ndarray[complex128]
        Reference (transmitted) complex symbols.
    rx_symbols : ndarray[complex128]
        Received / equalized complex symbols; same shape as ``tx_symbols``.

    Returns
    -------
    float
        EVM in [0, ∞); 0 = perfect, 1 = error power = signal power.
    """
    tx = np.asarray(tx_symbols, dtype=np.complex128).ravel()
    rx = np.asarray(rx_symbols, dtype=np.complex128).ravel()
    if tx.shape != rx.shape:
        raise ValueError(f"Shape mismatch: tx={tx.shape}, rx={rx.shape}")
    signal_power = float(np.mean(np.abs(tx) ** 2))
    error_power = float(np.mean(np.abs(rx - tx) ** 2))
    if signal_power == 0.0:
        return 0.0 if error_power == 0.0 else float("inf")
    return float(np.sqrt(error_power / signal_power))


def evm_db(
    tx_symbols: npt.NDArray[np.complex128],
    rx_symbols: npt.NDArray[np.complex128],
) -> float:
    """EVM in dB: ``20 · log10(evm(tx, rx))``."""
    e = evm(tx_symbols, rx_symbols)
    if e == 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(e))


def _paired(
    tx_symbols: npt.NDArray[np.complex128],
    rx_symbols: npt.NDArray[np.complex128],
) -> tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    tx = np.asarray(tx_symbols, dtype=np.complex128).ravel()
    rx = np.asarray(rx_symbols, dtype=np.complex128).ravel()
    if tx.shape != rx.shape:
        raise ValueError(f"Shape mismatch: tx={tx.shape}, rx={rx.shape}")
    return tx, rx


def max_phase_error(
    tx_symbols: npt.NDArray[np.complex128],
    rx_symbols: npt.NDArray[np.complex128],
) -> float:
    """Largest wrapped phase error in radians (Algorithm Reference v0.2 §12).

    Computed as ``max |arg(rx · conj(tx))|``, which wraps into ``(-π, π]``
    without a modular subtraction of two separately unwrapped angles. Bins
    where either symbol is zero carry no phase and are skipped; an all-zero
    pair returns ``0.0``.
    """
    tx, rx = _paired(tx_symbols, rx_symbols)
    live = (tx != 0) & (rx != 0)
    if not np.any(live):
        return 0.0
    return float(np.max(np.abs(np.angle(rx[live] * np.conj(tx[live])))))


def max_magnitude_relative_error(
    tx_symbols: npt.NDArray[np.complex128],
    rx_symbols: npt.NDArray[np.complex128],
) -> float:
    """Largest relative bin-magnitude error (Algorithm Reference v0.2 §12).

    ``max |(|rx| - |tx|)| / |tx|`` over bins with ``tx != 0``. Zero reference
    bins are skipped (the relative error is undefined there); an all-zero
    reference returns ``0.0``.
    """
    tx, rx = _paired(tx_symbols, rx_symbols)
    live = tx != 0
    if not np.any(live):
        return 0.0
    ref = np.abs(tx[live])
    return float(np.max(np.abs(np.abs(rx[live]) - ref) / ref))


def ser_from_bits(
    tx_bits: npt.NDArray[np.uint8],
    rx_bits: npt.NDArray[np.uint8],
    *,
    bits_per_symbol: int,
) -> float:
    """Symbol error rate from two bit streams.

    A symbol is in error when any of its ``bits_per_symbol`` consecutive bits
    differ. The streams must be the same length and a whole number of symbols.
    """
    if bits_per_symbol <= 0:
        raise ValueError(f"bits_per_symbol must be positive, got {bits_per_symbol}")
    tx = np.asarray(tx_bits, dtype=np.uint8).ravel()
    rx = np.asarray(rx_bits, dtype=np.uint8).ravel()
    if tx.shape != rx.shape:
        raise ValueError(f"Shape mismatch: tx={tx.shape}, rx={rx.shape}")
    if tx.size == 0:
        return 0.0
    if tx.size % bits_per_symbol:
        raise ValueError(f"bit stream length {tx.size} is not a multiple of bits_per_symbol={bits_per_symbol}")
    errors = (tx != rx).reshape(-1, bits_per_symbol)
    return float(np.count_nonzero(np.any(errors, axis=1))) / float(errors.shape[0])
