"""Phase-only HDC/FHRR side-path helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.tensor import SPFError, circular_diff_u16

SPF32_PHASE_MOD = 1 << 16
SPF32_PHASE_MASK = SPF32_PHASE_MOD - 1


def fhrr_bind(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """FHRR binding: elementwise phase addition."""
    aa, bb = np.broadcast_arrays(np.asarray(a, dtype=np.uint16), np.asarray(b, dtype=np.uint16))
    out: npt.NDArray[np.uint16] = (
        (aa.astype(np.uint32) + bb.astype(np.uint32)) & SPF32_PHASE_MASK
    ).astype(np.uint16)
    return out


def fhrr_unbind(bound: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """FHRR unbinding: elementwise phase subtraction."""
    aa, bb = np.broadcast_arrays(
        np.asarray(bound, dtype=np.uint16), np.asarray(b, dtype=np.uint16)
    )
    out: npt.NDArray[np.uint16] = (
        (aa.astype(np.uint32) - bb.astype(np.uint32)) & SPF32_PHASE_MASK
    ).astype(np.uint16)
    return out


def fhrr_conjugate(a: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """FHRR conjugate: phase negation."""
    aa = np.asarray(a, dtype=np.uint16)
    out: npt.NDArray[np.uint16] = ((-aa.astype(np.int64)) & SPF32_PHASE_MASK).astype(np.uint16)
    return out


def cosine_similarity_i64(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    scale: int = 32767,
) -> int:
    """Cosine similarity over phase deltas, accumulated in int64."""
    aa = np.asarray(a, dtype=np.uint16).ravel()
    bb = np.asarray(b, dtype=np.uint16).ravel()
    if aa.shape != bb.shape:
        raise SPFError("FHRR similarity shape mismatch")
    delta = np.asarray(circular_diff_u16(aa, bb), dtype=np.int32)
    angle = delta.astype(np.float64) * (2.0 * np.pi / SPF32_PHASE_MOD)
    terms = np.rint(np.cos(angle) * int(scale)).astype(np.int64)
    return int(np.sum(terms, dtype=np.int64))


def bundle_pairwise_phase_average(vectors: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Bundle phase vectors by pairwise signed circular averaging."""
    live = _vectors(vectors).astype(np.uint32)
    while live.shape[0] > 1:
        if live.shape[0] % 2:
            tail = live[-1:, :]
            live = live[:-1, :]
        else:
            tail = np.empty((0, live.shape[1]), dtype=np.uint32)
        a = live[0::2]
        b = live[1::2]
        delta = np.asarray(circular_diff_u16(b, a), dtype=np.int64)
        avg = (a.astype(np.int64) + delta // 2) & SPF32_PHASE_MASK
        live = np.vstack([avg.astype(np.uint32), tail])
    out: npt.NDArray[np.uint16] = live[0].astype(np.uint16)
    return out


def bundle_circular_mean(vectors: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Bundle phase vectors by FP64 circular mean reference."""
    vv = _vectors(vectors)
    angle = vv.astype(np.float64) * (2.0 * np.pi / SPF32_PHASE_MOD)
    mean = np.mean(np.exp(1j * angle), axis=0)
    theta = np.mod(np.angle(mean), 2.0 * np.pi)
    out: npt.NDArray[np.uint16] = (
        np.rint(theta * SPF32_PHASE_MOD / (2.0 * np.pi)).astype(np.uint32).astype(np.uint16)
    )
    return out


def _vectors(vectors: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    vv = np.ascontiguousarray(np.asarray(vectors), dtype=np.uint16)
    if vv.ndim != 2:
        raise SPFError("FHRR bundling expects [n_vectors, dims]")
    if vv.shape[0] == 0 or vv.shape[1] == 0:
        raise SPFError("FHRR bundling expects non-empty vectors")
    return vv


__all__ = [
    "bundle_circular_mean",
    "bundle_pairwise_phase_average",
    "cosine_similarity_i64",
    "fhrr_bind",
    "fhrr_conjugate",
    "fhrr_unbind",
]
