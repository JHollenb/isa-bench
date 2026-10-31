"""OFDM RX wired around the SPFv2 FB7 fused FFT (Algorithm Reference §7).

Pipeline shape:

    time-domain SPF-32  -> FB7 fused FFT (SPF-32 throughout)
                        -> SPF-32 zero-forcing via precomputed H^-1

The SPF-32 SPFv2 surface lives in :mod:`spf.fft.fused` (pure Python) and
:mod:`spf.native.fb7` (C-backed). Both produce the same SPF-32 spectrum
word-for-word at the per-mode tolerance; the C path is ~50–200× faster on
hot loops thanks to the libspf32 inner loop.

This module intentionally keeps SPF-32 end-to-end. The mixed-precision
variant (SPF-32 outer, SPF-16 inner FFT) lives on the `research` branch
under ``spf_experimental.dsp.mixed_precision`` and is not promoted to
``spf.*`` — promoting that variant to FB7 requires an SPF-16 FB7 Python
port (HLS already has it in ``hw/hls/common/spf16_fb7_scalar.hpp``).
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, assert_never

import numpy as np
import numpy.typing as npt

from spf.fft.fused import (
    DOMINANT_GAP_BALANCED_Q78,
    DOMINANT_GAP_FAST_Q78,
    SPF,
    pack_spf32,
    sincos_halfangle_poly7,
    spf_butterfly_fused,
    spf_fft_fused,
    unpack_words,
)
from spf.native.fb7 import fft as fb7_c_fft
from spf.scalar import spf32

_FB7Butterfly = Callable[[SPF, SPF, int], tuple[SPF, SPF]]

Mode = Literal["fast", "balanced", "accurate"]


@dataclass(frozen=True, slots=True)
class OfdmFB7Result:
    """Output of :func:`ofdm_rx_fused_poly7`.

    ``equalized`` is the SPF-32 spectrum after zero-forcing equalization;
    ``raw_bins`` is the SPF-32 spectrum before equalization. Both are stored
    as packed ``uint32`` SPF-32 words.
    """

    raw_bins: npt.NDArray[np.uint32]
    equalized: npt.NDArray[np.uint32]


def _python_butterfly_for_mode(mode: Mode) -> _FB7Butterfly:
    """Return a Python FB7 butterfly that matches C-side semantics for ``mode``.

    The C entry point sets a different dominant-gap threshold for each mode
    (§8 of the SPF Algorithm Reference): FAST uses 8·L_SCALE, BALANCED uses
    12·L_SCALE, ACCURATE disables the bypass. The Python backend must mirror
    that mapping bin-for-bin or mode sweeps silently misreport one mode as
    another.
    """
    if mode == "fast":
        gap_q78: int | None = DOMINANT_GAP_FAST_Q78
    elif mode == "balanced":
        gap_q78 = DOMINANT_GAP_BALANCED_Q78
    elif mode == "accurate":
        gap_q78 = None
    else:
        # mypy will flag this branch if Mode ever gains a member without a
        # matching elif above; the AssertionError is the runtime backstop for
        # untyped callers that pass strings outside Mode.
        assert_never(mode)

    def _bfly(u: SPF, v: SPF, tw: int) -> tuple[SPF, SPF]:
        return spf_butterfly_fused(
            u, v, tw,
            sincos_halfangle_fn=sincos_halfangle_poly7,
            log2_fn=math.log2,
            atan2_fn=math.atan2,
            exp2_fn=math.exp2,
            dominant_gap_q78=gap_q78,
        )

    return _bfly


def _fft_spf32_via_python(
    time_words: npt.NDArray[np.uint32], mode: Mode
) -> npt.NDArray[np.uint32]:
    bfly_fn = _python_butterfly_for_mode(mode)
    xs = unpack_words([int(w) for w in time_words])
    ys = spf_fft_fused(xs, bfly_fn=bfly_fn)
    return np.array([pack_spf32(y) for y in ys], dtype=np.uint32)


def _fft_spf32_via_c(
    time_words: npt.NDArray[np.uint32], mode: Mode
) -> npt.NDArray[np.uint32]:
    return fb7_c_fft(time_words, mode=mode)


def ofdm_rx_fused_poly7(
    time_words: npt.ArrayLike,
    h_inv_words: npt.ArrayLike,
    *,
    mode: Mode = "balanced",
    backend: Literal["python", "c"] = "c",
) -> OfdmFB7Result:
    """OFDM RX driven by the FB7 fused FFT (Algorithm Reference §7).

    Parameters
    ----------
    time_words : array-like
        Time-domain samples, length ``N`` (a power of two), as packed SPF-32
        ``uint32`` words.
    h_inv_words : array-like
        Pre-computed channel inverse, length ``N``, as packed SPF-32 words.
    mode : {"fast", "balanced", "accurate"}
        FB7 mode (§8). ``"balanced"`` is the default; ``"accurate"`` disables
        the dominant-gap bypass while keeping the shared polynomial finish.
    backend : {"python", "c"}
        Where to run the FFT. ``"c"`` uses the libspf32 FB7 entry point;
        ``"python"`` uses :func:`spf_experimental.fft.fused_butterfly.spf_fft_fused`
        for parity testing.

    Returns
    -------
    OfdmFB7Result
        ``raw_bins`` = FB7 FFT output; ``equalized`` = ``raw_bins · H_inv``
        in SPF-32 (per-bin complex multiply, no Sb call).
    """
    time_arr = np.ascontiguousarray(np.asarray(time_words), dtype=np.uint32)
    h_arr = np.ascontiguousarray(np.asarray(h_inv_words), dtype=np.uint32)
    n = time_arr.size
    if n < 2 or n & (n - 1):
        raise ValueError(f"time_words length must be a power of two; got {n}")
    if h_arr.size != n:
        raise ValueError(
            f"h_inv_words length {h_arr.size} != time_words length {n}"
        )

    if backend == "python":
        bins = _fft_spf32_via_python(time_arr, mode)
    elif backend == "c":
        bins = _fft_spf32_via_c(time_arr, mode)
    else:
        raise ValueError(f"unknown backend {backend!r}")

    equalized = spf32.mul(bins, h_arr)
    return OfdmFB7Result(raw_bins=bins, equalized=equalized)
