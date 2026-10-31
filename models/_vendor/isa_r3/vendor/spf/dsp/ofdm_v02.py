"""§12 OFDM validation profile driven by the v0.2 FB3-Hybrid transform.

Algorithm Reference v0.2 §12 states the profile as

.. code-block:: text

    ofdm_rx_spf64(time_spf32[64], h_inv_spf32[64], precision):
        x = copy_or_downcast_to_internal_format(time_spf32)
        X = spf_fft_radix2(x, precision)
        for k in 0..63:
            Y[k] = spf_mul(X[k], h_inv_spf32[k])
        return Y

:func:`ofdm_rx_spf_v02` is that datapath verbatim: packed SPF-32 words in,
:func:`spf.tensor.v02_planes.spf_fft_radix2` (the plane form of the normative
v0.2 radix-2 schedule) for the transform, :func:`spf.tensor.v02.spf_mul` for
the per-bin zero-forcing multiply, packed words out.  Nothing in the datapath
touches a Cartesian float array; the only decodes happen in
:func:`ofdm_v02_metrics`, which is measurement, not signal processing.

Two sibling profiles exist for the §11.7 four-way comparison:

``numpy``
    :func:`spf.dsp.ofdm_receiver.receive_sofdm_frame`'s historical path — a
    NumPy/scipy FFT on complex samples followed by an SPF equalize.
``fb7``
    the v0.1 fused half-angle butterfly (:mod:`spf.dsp.ofdm_fb7`,
    :mod:`spf.fft.spf32_fft` with its ``LEGACY_FB7_POLY7`` default), SPF-native
    but not the v0.2 transform.
``v02``
    this module.

The profile switch on the public receiver is
``receive_sofdm_frame(..., profile="v02")``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from spf.dsp.metrics import (
    ber,
    evm,
    max_magnitude_relative_error,
    max_phase_error,
    ser_from_bits,
)
from spf.tensor import decode_words
from spf.tensor import v02 as _v02
from spf.tensor import v02_planes as _v02_planes

__all__ = [
    "OfdmV02Metrics",
    "OfdmV02Result",
    "RXProfile",
    "ofdm_rx_spf_v02",
    "ofdm_v02_metrics",
    "spf_word_equalize",
    "spf_word_fft",
]

RXProfile = Literal["numpy", "fb7", "v02"]

_PROFILES: tuple[str, ...] = ("numpy", "fb7", "v02")


@dataclass(frozen=True, slots=True)
class OfdmV02Result:
    """Output of :func:`ofdm_rx_spf_v02`.

    ``raw_bins`` is ``X = spf_fft_radix2(x, precision)``; ``equalized`` is
    ``Y[k] = spf_mul(X[k], h_inv[k])``.  Both are packed SPF-32 ``uint32``
    words with the input's shape.  ``fft_calls`` counts transformed rows and
    ``spf_mul_calls`` counts equalized bins, so the datapath's structural cost
    is reportable without re-instrumenting a caller (design v0.2 §4
    invariant 8: conversion, fallback, and materialized-element counts are
    reported).  ``cartesian_conversions`` is zero by construction on this
    path — there is no float FFT and no decode inside it.
    """

    raw_bins: npt.NDArray[np.uint32]
    equalized: npt.NDArray[np.uint32]
    precision: str
    fft_calls: int
    spf_mul_calls: int
    cartesian_conversions: int = 0


@dataclass(frozen=True, slots=True)
class OfdmV02Metrics:
    """§12 validation metrics for one equalized frame.

    ``max_phase_error_rad`` and ``max_magnitude_relative_error`` are the §12
    "phase error" and "bin magnitude error" rows; ``evm`` and
    ``symbol_error_rate`` are the §12 EVM and SER rows.  ``bit_error_rate`` is
    ``None`` when no bit streams were supplied.
    """

    evm: float
    max_phase_error_rad: float
    max_magnitude_relative_error: float
    symbol_error_rate: float
    bit_error_rate: float | None = None


def _as_words(values: npt.ArrayLike, name: str) -> npt.NDArray[np.uint32]:
    array = np.ascontiguousarray(np.asarray(values), dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError(f"{name} must be a 1-D or batched array of SPF-32 words")
    n = int(array.shape[-1])
    if n < 2 or (n & (n - 1)) != 0:
        raise ValueError(f"{name} last axis must be a power of two >= 2; got N={n}")
    return array


def spf_word_fft(
    time_words: npt.ArrayLike,
    *,
    profile: RXProfile = "v02",
    precision: Any = "balanced",
) -> npt.NDArray[np.uint32]:
    """Transform packed SPF-32 words on the last axis under ``profile``.

    ``profile="v02"`` is the normative FB3-Hybrid radix-2 schedule
    (:func:`spf.tensor.v02_planes.spf_fft_radix2`); ``profile="fb7"`` is the
    legacy fused half-angle butterfly (:func:`spf.fft.spf32_fft.fft`, which
    routes to C when libspf32 is importable).  ``profile="numpy"`` is rejected
    here: that profile has no word-domain transform, it decodes to Cartesian
    first, and the receiver handles it separately.
    """
    array = _as_words(time_words, "time_words")
    if profile == "v02":
        return np.asarray(_v02_planes.spf_fft_radix2(array, precision), dtype=np.uint32)
    if profile == "fb7":
        from spf.fft import spf32_fft

        return np.asarray(spf32_fft.fft(array), dtype=np.uint32)
    if profile == "numpy":
        raise ValueError("profile='numpy' has no word-domain FFT; use receive_sofdm_frame")
    raise ValueError(f"unknown profile {profile!r}; expected one of {_PROFILES}")


def spf_word_equalize(
    bins: npt.ArrayLike,
    h_inv_words: npt.ArrayLike,
    *,
    profile: RXProfile = "v02",
) -> npt.NDArray[np.uint32]:
    """Per-bin zero-forcing multiply ``Y[k] = X[k] · H⁻¹[k]`` on packed words.

    ``profile="v02"`` uses :func:`spf.tensor.v02.spf_mul` (the §5.1 typed op
    under ``NaNProfile.STRICT_V02``), which is what §12 names.  Every other
    profile keeps the historical :func:`spf.tensor.spf_mul_words` plane
    multiply; the two agree bit-for-bit on finite non-zero words and differ
    only in sentinel handling.
    """
    a = np.asarray(bins, dtype=np.uint32)
    b = np.asarray(h_inv_words, dtype=np.uint32)
    if profile == "v02":
        return np.asarray(_v02.spf_mul(a, b), dtype=np.uint32)
    from spf.tensor import spf_mul_words

    return np.asarray(spf_mul_words(a, b), dtype=np.uint32)


def ofdm_rx_spf_v02(
    time_words: npt.ArrayLike,
    h_inv_words: npt.ArrayLike,
    *,
    precision: Any = "balanced",
) -> OfdmV02Result:
    """Algorithm Reference v0.2 §12 receive datapath, SPF-native end to end.

    Parameters
    ----------
    time_words : array-like
        Time-domain samples as packed SPF-32 words. The last axis (length
        ``N``, a power of two) is transformed; leading axes are OFDM symbols
        or any other batch dimension and are preserved.
    h_inv_words : array-like
        Pre-computed channel inverse ``1/H[k]`` as packed SPF-32 words,
        broadcastable against ``time_words``.
    precision : str | PrecisionMode
        v0.2 precision mode for the FB3-Hybrid pair adder ("fast",
        "balanced", "accurate", "exact").

    Returns
    -------
    OfdmV02Result
        ``raw_bins`` = ``spf_fft_radix2(x)``, ``equalized`` =
        ``spf_mul(X, h_inv)``, plus the structural counts.
    """
    x = _as_words(time_words, "time_words")
    h_inv = np.ascontiguousarray(np.asarray(h_inv_words), dtype=np.uint32)
    n = int(x.shape[-1])
    if h_inv.shape[-1] != n:
        raise ValueError(f"h_inv_words last axis {h_inv.shape[-1]} != time_words last axis {n}")
    try:
        np.broadcast_shapes(x.shape, h_inv.shape)
    except ValueError as exc:  # pragma: no cover - message is the value
        raise ValueError(f"h_inv_words shape {h_inv.shape} does not broadcast to {x.shape}") from exc

    bins = np.asarray(_v02_planes.spf_fft_radix2(x, precision), dtype=np.uint32)
    equalized = spf_word_equalize(bins, h_inv, profile="v02")
    return OfdmV02Result(
        raw_bins=bins,
        equalized=equalized,
        precision=_precision_label(precision),
        fft_calls=int(x.size // n),
        spf_mul_calls=int(equalized.size),
        cartesian_conversions=0,
    )


def _precision_label(precision: Any) -> str:
    if precision is None:
        return "balanced"
    name = getattr(precision, "name", None)
    return str(name).lower() if name is not None else str(precision).lower()


def ofdm_v02_metrics(
    reference_symbols: npt.ArrayLike,
    equalized_words: npt.ArrayLike,
    *,
    tx_bits: npt.ArrayLike | None = None,
    rx_bits: npt.ArrayLike | None = None,
    bits_per_symbol: int = 2,
) -> OfdmV02Metrics:
    """Score one equalized word frame against its transmitted symbols (§12).

    The decode here is measurement only: the receive datapath in
    :func:`ofdm_rx_spf_v02` never leaves the word domain.  ``reference_symbols``
    is the transmitted constellation; ``equalized_words`` is
    ``OfdmV02Result.equalized`` restricted to the same bins.

    ``symbol_error_rate`` comes from ``tx_bits``/``rx_bits`` when both are
    given (a symbol is in error when any of its ``bits_per_symbol`` bits
    differ); otherwise it is the nearest-reference-symbol decision rate over
    the reference alphabet, which is the §12 SER for a known-grid frame.
    """
    ref = np.asarray(reference_symbols, dtype=np.complex128).ravel()
    got = decode_words(np.asarray(equalized_words, dtype=np.uint32)).ravel()
    if ref.shape != got.shape:
        raise ValueError(f"reference_symbols {ref.shape} != equalized_words {got.shape}")

    bit_error_rate: float | None = None
    if tx_bits is not None and rx_bits is not None:
        tx = np.asarray(tx_bits, dtype=np.uint8).ravel()
        rx = np.asarray(rx_bits, dtype=np.uint8).ravel()
        bit_error_rate = ber(tx, rx)
        symbol_error_rate = ser_from_bits(tx, rx, bits_per_symbol=bits_per_symbol)
    else:
        symbol_error_rate = _ser_nearest(ref, got)

    return OfdmV02Metrics(
        evm=evm(ref, got),
        max_phase_error_rad=max_phase_error(ref, got),
        max_magnitude_relative_error=max_magnitude_relative_error(ref, got),
        symbol_error_rate=symbol_error_rate,
        bit_error_rate=bit_error_rate,
    )


def _ser_nearest(ref: npt.NDArray[np.complex128], got: npt.NDArray[np.complex128]) -> float:
    """Symbol error rate by nearest-neighbour decision over the reference alphabet."""
    if ref.size == 0:
        return 0.0
    alphabet = np.unique(np.round(ref, 12))
    if alphabet.size < 2:
        return 0.0
    distance = np.abs(got[:, None] - alphabet[None, :])
    decided = alphabet[np.argmin(distance, axis=1)]
    return float(np.count_nonzero(decided != np.round(ref, 12))) / float(ref.size)
