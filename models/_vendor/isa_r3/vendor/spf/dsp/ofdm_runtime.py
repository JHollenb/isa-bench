"""SPF-resident OFDM product receive paths.

These helpers are the first runtime/product layer over the existing native DSP
kernels. They keep the product path in SPF form through FFT, channel inverse,
and demap, and only materialize equalized SPF bins when diagnostics are
explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_fb7 import Mode, _fft_spf32_via_python
from spf.native import spf32 as c_spf32
from spf.native.fb7 import fft as fb7_c_fft
from spf.runtime import Domain, Layout, Provenance, RuntimeTelemetry, SPFBuffer, SPFMeta

Backend = Literal["c", "python"]


@dataclass(frozen=True, slots=True)
class OfdmRuntimeResult:
    """Product output from an SPF-resident OFDM receive path."""

    bits: npt.NDArray[np.uint8] | None = None
    symbols: npt.NDArray[np.uint8] | None = None
    llr: npt.NDArray[np.int8] | None = None
    confidence: npt.NDArray[np.uint8] | None = None
    erasure: npt.NDArray[np.uint8] | None = None
    telemetry: RuntimeTelemetry | None = None
    raw_bins: npt.NDArray[np.uint32] | None = None
    equalized: npt.NDArray[np.uint32] | None = None


def _as_words(x: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return np.ascontiguousarray(np.asarray(x), dtype=np.uint32).ravel()


def _strip_cp(time_words: npt.ArrayLike, cyclic_prefix_len: int) -> npt.NDArray[np.uint32]:
    words = _as_words(time_words)
    if cyclic_prefix_len < 0:
        raise ValueError(f"cyclic_prefix_len must be non-negative, got {cyclic_prefix_len}")
    if cyclic_prefix_len:
        if cyclic_prefix_len >= words.size:
            raise ValueError(
                f"cyclic_prefix_len {cyclic_prefix_len} must be smaller than "
                f"symbol length {words.size}"
            )
        words = words[cyclic_prefix_len:]
    n = words.size
    if n < 2 or n & (n - 1):
        raise ValueError(f"OFDM symbol length after CP removal must be a power of two; got {n}")
    return np.ascontiguousarray(words, dtype=np.uint32)


def _fft_words(
    time_words: npt.NDArray[np.uint32], *, mode: Mode, backend: Backend
) -> npt.NDArray[np.uint32]:
    if backend == "c":
        return fb7_c_fft(time_words, mode=mode)
    if backend == "python":
        return _fft_spf32_via_python(time_words, mode)
    raise ValueError(f"unknown backend {backend!r}")


def _prep_erasure(erasure_mask: npt.ArrayLike | None, n: int) -> npt.NDArray[np.uint8] | None:
    if erasure_mask is None:
        return None
    erasure = np.ascontiguousarray(np.asarray(erasure_mask), dtype=np.uint8).ravel()
    if erasure.size != n:
        raise ValueError(f"erasure_mask length {erasure.size} != OFDM bin count {n}")
    return erasure


def _estimate_cpe_if_requested(
    raw_bins: npt.NDArray[np.uint32],
    h_inv_words: npt.NDArray[np.uint32],
    cpe_theta: int,
    *,
    pilot_bins: npt.ArrayLike | None,
    pilot_symbols: npt.ArrayLike | None,
    pilot_mask: npt.ArrayLike | None,
) -> tuple[int, tuple[str, ...], dict[str, Any]]:
    if pilot_bins is None and pilot_symbols is None:
        correction = int(cpe_theta) & 0xFFFF
        return correction, (), {
            "estimated_cpe_residual_theta": None,
            "cpe_correction_theta": correction,
            "pilot_count": 0,
            "pilot_active_count": 0,
        }
    if pilot_bins is None or pilot_symbols is None:
        raise ValueError("pilot_bins and pilot_symbols must be provided together")
    bins = np.ascontiguousarray(np.asarray(pilot_bins), dtype=np.int64).ravel()
    if np.any((bins < 0) | (bins >= raw_bins.size)):
        raise ValueError("pilot_bins contains an index outside the OFDM symbol")
    known = _as_words(pilot_symbols)
    if known.size != bins.size:
        raise ValueError(f"pilot_symbols length {known.size} != pilot_bins length {bins.size}")
    mask = None
    if pilot_mask is not None:
        mask = np.ascontiguousarray(np.asarray(pilot_mask), dtype=np.uint8).ravel()
        if mask.size != bins.size:
            raise ValueError(f"pilot_mask length {mask.size} != pilot_bins length {bins.size}")
    active_count = int(np.count_nonzero(mask)) if mask is not None else int(bins.size)
    estimated = c_spf32.ofdm_cpe_estimate_from_pilots(
        raw_bins[bins], h_inv_words[bins], known, mask
    )
    # The demap kernels apply cpe_theta as a correction. The pilot estimator
    # returns the measured residual phase, so product receive uses -estimate.
    correction = (-estimated) & 0xFFFF
    return correction, ("cpe_estimate",), {
        "estimated_cpe_residual_theta": int(estimated) & 0xFFFF,
        "cpe_correction_theta": correction,
        "pilot_count": int(bins.size),
        "pilot_active_count": active_count,
    }


def _telemetry(
    *,
    backend: Backend,
    stages: tuple[str, ...],
    diagnostics: bool,
    materialized_equalized: bool,
    mode: Mode,
    cyclic_prefix_len: int,
    cpe_theta: int,
    cpe_details: dict[str, Any] | None = None,
) -> RuntimeTelemetry:
    details = {
        "fb7_mode": mode,
        "cyclic_prefix_len": cyclic_prefix_len,
        "cpe_theta": int(cpe_theta) & 0xFFFF,
        "cpe_correction_theta": int(cpe_theta) & 0xFFFF,
    }
    if cpe_details is not None:
        details.update(cpe_details)
    return RuntimeTelemetry(
        backend=backend,
        stages=stages,
        materialized_equalized=materialized_equalized,
        diagnostics=diagnostics,
        details=details,
    )


def _diagnostic_equalized(
    raw_bins: npt.NDArray[np.uint32],
    h_inv_words: npt.NDArray[np.uint32],
    cpe_theta: int,
    diagnostics: bool,
) -> npt.NDArray[np.uint32] | None:
    if not diagnostics:
        return None
    equalized = c_spf32.mul(raw_bins, h_inv_words)
    return c_spf32.phase_rotate_const(equalized, cpe_theta)


def receive_qpsk(
    time_words: npt.ArrayLike | SPFBuffer,
    h_inv_words: npt.ArrayLike,
    *,
    cpe_theta: int = 0,
    cyclic_prefix_len: int = 0,
    mode: Mode = "balanced",
    backend: Backend = "c",
    erasure_mask: npt.ArrayLike | None = None,
    pilot_bins: npt.ArrayLike | None = None,
    pilot_symbols: npt.ArrayLike | None = None,
    pilot_mask: npt.ArrayLike | None = None,
    diagnostics: bool = False,
) -> OfdmRuntimeResult:
    """Receive one SPF OFDM symbol and emit QPSK hard bits.

    ``time_words`` is a time-domain SPF-32 OFDM symbol, optionally with a
    cyclic prefix. ``h_inv_words`` is the per-bin channel inverse. Product mode
    emits bits, confidence, erasure, and telemetry. Raw FFT bins and equalized
    bins are returned only when ``diagnostics=True``.
    """
    source = time_words
    words = source.words if isinstance(source, SPFBuffer) else source
    time_arr = _strip_cp(words, cyclic_prefix_len)
    h_inv = _as_words(h_inv_words)
    if h_inv.size != time_arr.size:
        raise ValueError(f"h_inv_words length {h_inv.size} != OFDM bin count {time_arr.size}")
    erasure = _prep_erasure(erasure_mask, time_arr.size)
    raw_bins = _fft_words(time_arr, mode=mode, backend=backend)
    cpe, cpe_stages, cpe_details = _estimate_cpe_if_requested(
        raw_bins,
        h_inv,
        cpe_theta,
        pilot_bins=pilot_bins,
        pilot_symbols=pilot_symbols,
        pilot_mask=pilot_mask,
    )
    bits, confidence = c_spf32.ofdm_qpsk_eq_demod_fused(
        raw_bins, h_inv, cpe, erasure_mask=erasure
    )
    equalized = _diagnostic_equalized(raw_bins, h_inv, cpe, diagnostics)
    stages = ("cp_remove", "fb7_fft", *cpe_stages, "qpsk_fused_demap")
    return OfdmRuntimeResult(
        bits=bits,
        confidence=confidence,
        erasure=erasure,
        raw_bins=raw_bins if diagnostics else None,
        equalized=equalized,
        telemetry=_telemetry(
            backend=backend,
            stages=stages,
            diagnostics=diagnostics,
            materialized_equalized=equalized is not None,
            mode=mode,
            cyclic_prefix_len=cyclic_prefix_len,
            cpe_theta=cpe,
            cpe_details=cpe_details,
        ),
    )


def receive_qam_lut(
    time_words: npt.ArrayLike | SPFBuffer,
    h_inv_words: npt.ArrayLike,
    *,
    lut_symbol: npt.ArrayLike,
    lut_confidence: npt.ArrayLike,
    l_shift: int,
    theta_shift: int,
    l_min: int,
    l_max: int,
    agc_L_offset: int = 0,
    cpe_theta: int = 0,
    cyclic_prefix_len: int = 0,
    mode: Mode = "balanced",
    backend: Backend = "c",
    erasure_mask: npt.ArrayLike | None = None,
    diagnostics: bool = False,
) -> OfdmRuntimeResult:
    """Receive one SPF OFDM symbol and emit constellation LUT symbols."""
    words = time_words.words if isinstance(time_words, SPFBuffer) else time_words
    time_arr = _strip_cp(words, cyclic_prefix_len)
    h_inv = _as_words(h_inv_words)
    if h_inv.size != time_arr.size:
        raise ValueError(f"h_inv_words length {h_inv.size} != OFDM bin count {time_arr.size}")
    erasure = _prep_erasure(erasure_mask, time_arr.size)
    raw_bins = _fft_words(time_arr, mode=mode, backend=backend)
    symbols, confidence = c_spf32.constellation_lut_demap_fused(
        raw_bins,
        h_inv,
        lut_symbol,
        lut_confidence,
        agc_L_offset=agc_L_offset,
        cpe_theta=cpe_theta,
        l_shift=l_shift,
        theta_shift=theta_shift,
        l_min=l_min,
        l_max=l_max,
        erasure_mask=erasure,
    )
    equalized = _diagnostic_equalized(raw_bins, h_inv, cpe_theta, diagnostics)
    return OfdmRuntimeResult(
        symbols=symbols,
        confidence=confidence,
        erasure=erasure,
        raw_bins=raw_bins if diagnostics else None,
        equalized=equalized,
        telemetry=_telemetry(
            backend=backend,
            stages=("cp_remove", "fb7_fft", "constellation_lut_demap"),
            diagnostics=diagnostics,
            materialized_equalized=equalized is not None,
            mode=mode,
            cyclic_prefix_len=cyclic_prefix_len,
            cpe_theta=cpe_theta,
        ),
    )


def receive_qam_llr(
    time_words: npt.ArrayLike | SPFBuffer,
    h_inv_words: npt.ArrayLike,
    *,
    lut_llr: npt.ArrayLike,
    bits_per_symbol: int,
    l_shift: int,
    theta_shift: int,
    l_min: int,
    l_max: int,
    agc_L_offset: int = 0,
    cpe_theta: int = 0,
    cyclic_prefix_len: int = 0,
    mode: Mode = "balanced",
    backend: Backend = "c",
    erasure_mask: npt.ArrayLike | None = None,
    diagnostics: bool = False,
) -> OfdmRuntimeResult:
    """Receive one SPF OFDM symbol and emit constellation LUT LLRs."""
    words = time_words.words if isinstance(time_words, SPFBuffer) else time_words
    time_arr = _strip_cp(words, cyclic_prefix_len)
    h_inv = _as_words(h_inv_words)
    if h_inv.size != time_arr.size:
        raise ValueError(f"h_inv_words length {h_inv.size} != OFDM bin count {time_arr.size}")
    erasure = _prep_erasure(erasure_mask, time_arr.size)
    raw_bins = _fft_words(time_arr, mode=mode, backend=backend)
    llr = c_spf32.constellation_lut_llr_fused(
        raw_bins,
        h_inv,
        lut_llr,
        bits_per_symbol,
        agc_L_offset=agc_L_offset,
        cpe_theta=cpe_theta,
        l_shift=l_shift,
        theta_shift=theta_shift,
        l_min=l_min,
        l_max=l_max,
        erasure_mask=erasure,
    )
    equalized = _diagnostic_equalized(raw_bins, h_inv, cpe_theta, diagnostics)
    return OfdmRuntimeResult(
        llr=llr,
        erasure=erasure,
        raw_bins=raw_bins if diagnostics else None,
        equalized=equalized,
        telemetry=_telemetry(
            backend=backend,
            stages=("cp_remove", "fb7_fft", "constellation_lut_llr"),
            diagnostics=diagnostics,
            materialized_equalized=equalized is not None,
            mode=mode,
            cyclic_prefix_len=cyclic_prefix_len,
            cpe_theta=cpe_theta,
        ),
    )


def buffer_from_words(
    words: npt.ArrayLike,
    *,
    domain: Domain = Domain.TIME,
    backend: str = "native",
    source: str = "ofdm_runtime",
) -> SPFBuffer:
    """Create a zero-copy-oriented SPFBuffer from packed words."""
    return SPFBuffer(
        words=np.ascontiguousarray(np.asarray(words), dtype=np.uint32),
        meta=SPFMeta(),
        layout=Layout.CONTIGUOUS,
        domain=domain,
        provenance=Provenance(source=source, backend=backend),
        backend=backend,
    )
