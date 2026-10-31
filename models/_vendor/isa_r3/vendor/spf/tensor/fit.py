"""Small SPFTensor compiler fit helpers.

These helpers are intentionally thin wrappers around the registered v3 block
system.  They provide the metrics and residual accounting needed by external
model exporters without making those exporters depend on lower-level codec
details.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base_models import fit_separable_phase_rank1
from .core import SPF32_PHASE_MASK, SPF32_PHASE_MOD
from .residual_codecs import NoneCodec, RawInt16Codec
from .v3 import BaseModelKind, BaseModelRef, ResidualCodecKind, SPFBlockV3

_TWO_PI = 2.0 * np.pi
_PHASE_TO_RAD = _TWO_PI / SPF32_PHASE_MOD


def circular_phase_rmse(theta_a: np.ndarray, theta_b: np.ndarray) -> float:
    """Circular RMSE between two SPF-32 phase-code planes, in radians."""
    a = np.asarray(theta_a, dtype=np.float64)
    b = np.asarray(theta_b, dtype=np.float64)
    err = np.angle(np.exp(1j * (a - b) * _PHASE_TO_RAD))
    return float(np.sqrt(np.mean(err * err))) if err.size else 0.0


def _phase_delta_u16(target: np.ndarray, base: np.ndarray) -> np.ndarray:
    """Signed shortest-ring delta from ``base`` to ``target`` in u16 phase units."""
    raw = (
        np.asarray(target, dtype=np.int64)
        - np.asarray(base, dtype=np.int64)
        + (SPF32_PHASE_MOD // 2)
    ) & SPF32_PHASE_MASK
    return raw - (SPF32_PHASE_MOD // 2)


def fit_separable_phase_block(
    theta_u16: np.ndarray,
    *,
    axes: tuple[int, int] = (0, 1),
    dims: tuple[int, ...] | None = None,
    residual_policy: str = "auto",
) -> SPFBlockV3:
    """Fit a rank-1 ``SEPARABLE_PHASE`` block from an encoded phase plane.

    ``residual_policy``:
      ``"none"`` requires a residual-free fit,
      ``"auto"`` stores no residual when exact and otherwise stores RAW_INT16,
      ``"raw_int16"`` always stores RAW_INT16 residuals.
    """
    theta = np.asarray(theta_u16, dtype=np.int64) & SPF32_PHASE_MASK
    if theta.ndim != 2:
        raise ValueError("fit_separable_phase_block expects a 2-D phase plane")
    block_dims = tuple(int(d) for d in (dims or theta.shape))
    if block_dims != tuple(theta.shape):
        raise ValueError(f"dims {block_dims!r} do not match theta shape {theta.shape!r}")
    axis_i, axis_j = (int(axes[0]), int(axes[1]))
    params = fit_separable_phase_rank1(theta, axis_i=axis_i, axis_j=axis_j)
    base = BaseModelRef(int(BaseModelKind.SEPARABLE_PHASE), params)
    candidate = SPFBlockV3(dims=block_dims, base=base, residual=NoneCodec.encode())
    _, theta_fit = candidate.reconstruct()
    dtheta = _phase_delta_u16(theta, theta_fit).reshape(-1)
    exact = bool(np.all(dtheta == 0))
    if residual_policy not in {"none", "auto", "raw_int16"}:
        raise ValueError("residual_policy must be 'none', 'auto', or 'raw_int16'")
    if residual_policy == "none" and not exact:
        raise ValueError("SEPARABLE_PHASE fit is not residual-free")
    if residual_policy == "raw_int16" or (residual_policy == "auto" and not exact):
        residual = RawInt16Codec.encode(np.zeros(theta.size, dtype=np.int64), dtheta)
    else:
        residual = NoneCodec.encode()
    block = SPFBlockV3(dims=block_dims, base=base, residual=residual)
    _, theta_recon = block.reconstruct()
    block.meta = {
        "fit_kind": "SEPARABLE_PHASE",
        "phase_rmse_rad": circular_phase_rmse(theta, theta_recon),
        "residual_bits": 0
        if block.residual.codec_id == int(ResidualCodecKind.NONE)
        else int(theta.size * 32),
        "residual_free": block.residual.codec_id == int(ResidualCodecKind.NONE),
    }
    return block


def fit_separable_logmag_block(
    logmag: np.ndarray,
    *,
    axes: tuple[int, int] = (0, 1),
    dims: tuple[int, ...] | None = None,
) -> SPFBlockV3:
    """Encode a rank-1 separable log-magnitude block by least squares.

    This is a conservative MVP helper for exporters that already have an
    approximately rank-1 log-magnitude plane.  The exact dense residual is left
    to callers for now; PAN uses real-scalar dense blocks for non-phase params.
    """
    arr = np.asarray(logmag, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("fit_separable_logmag_block expects a 2-D plane")
    block_dims = tuple(int(d) for d in (dims or arr.shape))
    u, s, vh = np.linalg.svd(arr, full_matrices=False)
    scale = np.sqrt(s[0])
    b = np.rint(u[:, 0] * scale).astype(np.int64)
    g = np.rint(vh[0, :] * scale).astype(np.int64)
    # The dominant singular vectors are unit-norm, so for a plane whose entries
    # are O(1) the per-element products round to all-zeros — a silently dead
    # block. Fail loud rather than return one; an exporter that genuinely wants
    # the integer-grid fit must pre-scale the plane into a representable range.
    if np.any(arr) and not (np.any(b) and np.any(g)):
        raise ValueError(
            "fit_separable_logmag_block: rank-1 integer fit rounded to zero "
            "(plane magnitudes too small for the int64 grid); pre-scale before fitting"
        )
    params: dict[str, Any] = {
        "logmag_terms": [
            {"axis_i": int(axes[0]), "axis_j": int(axes[1]), "b": b, "g": g}
        ]
    }
    block = SPFBlockV3(
        dims=block_dims,
        base=BaseModelRef(int(BaseModelKind.SEPARABLE_LOGMAG), params),
        residual=NoneCodec.encode(),
    )
    # Fit-quality signal — the phase fitter records ``phase_rmse_rad``; without
    # the equivalent here a caller gets no error feedback on this lossy
    # SVD->int encode. The NONE-residual block reconstructs to exactly this
    # integer rank-1 outer product, so the recorded rel-L2 *is* the fit error.
    recon = np.outer(b, g).astype(np.float64)
    denom = float(np.linalg.norm(arr))
    block.meta = {
        "fit_kind": "SEPARABLE_LOGMAG",
        "logmag_rel_l2": float(np.linalg.norm(arr - recon) / denom) if denom else 0.0,
        "residual_free": True,
        "residual_bits": 0,
    }
    return block


__all__ = [
    "circular_phase_rmse",
    "fit_separable_logmag_block",
    "fit_separable_phase_block",
]
