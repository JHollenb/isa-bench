"""Python reference implementations for the 'gauge_norm' and 'l_gate' ops.

Log-domain gauge control for phase-native network blocks. Both ops act on the
L plane only (Θ is never touched), which is the point: in SPF the magnitude
gauge is an *additive* degree of freedom, so normalisation is an integer
subtract and gating is an integer compare — no divide, no sqrt, no float in
the deployable path (the ``rms`` tier mirrors ``softmax_lse``'s fp64-internal
exact path for its row statistic only).

``gauge_norm`` — subtract a per-row L statistic (the gauge) and add learned
per-channel L offsets:

* ``linf`` tier: statistic = max finite L (tropical, exact integer).
* ``gml``  tier: statistic = mean finite L code, round-half-away (exact
  integer subtract; after it the row's mean log-gain is 0 within one code —
  the edge-of-chaos pin).
* ``rms``  tier: statistic = ``(1/2)·log2(mean |z|^2)`` in L codes, computed
  fp64-internal with a max pivot (no overflow), mirroring the RMSNorm
  magnitude semantics.

``l_gate`` — tropical magnitude gate: finite words whose L falls below a
per-channel threshold become ``WORD_ZERO`` (the ReLU analogue; a comparator,
not an arithmetic op).

Sentinel discipline (both ops): zero words (``L=0x8000``) and NaN words
(``L=0x7FFF``) pass through unchanged and are excluded from statistics.

These are called by :class:`~spf.tensor.v2.ExecutionPlan`'s ``execute()``
dispatcher; like :mod:`spf.tensor.attention_ops` they operate purely on numpy
arrays and import nothing from outside the spf.tensor package.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .core import (
    SPF32_L_FRAC_BITS,
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    SPF32_WORD_ZERO,
    pack_words,
    unpack_L,
    unpack_theta,
)

__all__ = ["execute_gauge_norm", "execute_l_gate"]

# NaN sentinel L code (canonical NaN word is L=0x7FFF, Θ=0xFFFF); kept local —
# mirrors the private constant in spf.tensor.v2.
_L_NAN = np.int16(0x7FFF)

_TIERS = ("linf", "gml", "rms")


def _half_away_round(x: npt.ArrayLike) -> np.ndarray:
    """Vectorised round-half-away (mirrors C ``llround``)."""
    a = np.asarray(x, dtype=np.float64)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _finite_mask(L: np.ndarray) -> np.ndarray:
    """True where the word is finite (not the zero / NaN sentinel)."""
    return (L != SPF32_L_ZERO) & (L != _L_NAN)


def _row_stat_codes(L: np.ndarray, finite: np.ndarray, tier: str, axis: int) -> np.ndarray:
    """Per-row gauge statistic in L codes (float64, keepdims along ``axis``).

    Rows with no finite entry get a NaN statistic; callers must leave those
    rows untouched.
    """
    L64 = L.astype(np.float64)
    count = finite.sum(axis=axis, keepdims=True).astype(np.float64)
    empty = count == 0.0
    # -inf is the neutral element for the max over finite entries; plain
    # (non-nan) reductions keep all-sentinel rows warning-free.
    maxed = np.max(np.where(finite, L64, -np.inf), axis=axis, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        if tier == "linf":
            stat = maxed
        elif tier == "gml":
            total = np.where(finite, L64, 0.0).sum(axis=axis, keepdims=True)
            stat = _half_away_round(total / count)
        elif tier == "rms":
            # (1/2)·log2(mean 2^(2·L/scale)) in code units, max-pivoted so the
            # fp64 exp2 never overflows (L codes span ±32766 → ±128 in log2).
            scale = float(1 << SPF32_L_FRAC_BITS)
            twoL = 2.0 * L64  # code units, scale unchanged
            m = 2.0 * maxed
            terms = np.where(finite, np.exp2((twoL - np.where(empty, 0.0, m)) / scale), 0.0)
            mean = terms.sum(axis=axis, keepdims=True) / count
            stat = _half_away_round(0.5 * (m + np.log2(mean) * scale))
        else:  # pragma: no cover - guarded by execute_gauge_norm
            raise ValueError(f"gauge_norm: unknown tier {tier!r}")
    return np.where(empty, np.nan, stat)


def execute_gauge_norm(
    words: npt.ArrayLike,
    *,
    tier: str = "gml",
    axis: int = -1,
    gamma_codes: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Python reference for the ``gauge_norm`` execution op.

    ``L_out = clip(L - stat_row + gamma, L_MIN_FINITE, L_MAX_FINITE)``; Θ is
    untouched; sentinel words pass through and are excluded from ``stat_row``.

    Parameters
    ----------
    words:
        uint32 array of packed SPF-32 words, any shape with the channel axis
        given by ``axis``.
    tier:
        ``"linf"`` (max-L), ``"gml"`` (mean-L, exact integer), or ``"rms"``
        (log-RMS of magnitudes, fp64-internal statistic).
    axis:
        The channel axis the statistic is taken over (default last).
    gamma_codes:
        Optional learned per-channel L offsets in L-code units (int, length =
        size of ``axis``); broadcast over the remaining axes and added after
        the subtract (the RMSNorm ``γ`` analogue, free in SPF).
    """
    if tier not in _TIERS:
        raise ValueError(f"gauge_norm: tier must be one of {_TIERS}, got {tier!r}")
    arr = np.asarray(words, dtype=np.uint32)
    L = unpack_L(arr)
    T = unpack_theta(arr)
    finite = _finite_mask(L)

    stat = _row_stat_codes(L, finite, tier, axis)

    delta = np.where(np.isnan(stat), 0.0, stat)  # all-sentinel rows: no shift
    L_new64 = L.astype(np.int64) - delta.astype(np.int64)

    if gamma_codes is not None:
        gamma = np.asarray(gamma_codes, dtype=np.int64)
        if gamma.ndim != 1 or gamma.shape[0] != arr.shape[axis]:
            raise ValueError(
                f"gauge_norm: gamma_codes must be 1-D of length {arr.shape[axis]}, "
                f"got shape {gamma.shape}"
            )
        shape = [1] * arr.ndim
        shape[axis] = gamma.shape[0]
        L_new64 = L_new64 + gamma.reshape(shape)

    L_new = np.clip(L_new64, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE).astype(np.int16)
    out = pack_words(L_new, T)
    return np.where(finite, out, arr).astype(np.uint32)


def execute_l_gate(
    words: npt.ArrayLike,
    *,
    tau_codes: npt.ArrayLike,
    axis: int = -1,
) -> np.ndarray:
    """Python reference for the ``l_gate`` execution op.

    Finite words with ``L < tau`` become ``WORD_ZERO``; everything else
    (including the NaN sentinel) passes through unchanged. ``tau_codes`` is a
    scalar or per-channel array of L-code thresholds along ``axis``.
    """
    arr = np.asarray(words, dtype=np.uint32)
    L = unpack_L(arr)
    finite = _finite_mask(L)

    tau = np.asarray(tau_codes, dtype=np.int64)
    if tau.ndim == 0:
        tau_b = tau
    elif tau.ndim == 1:
        if tau.shape[0] != arr.shape[axis]:
            raise ValueError(
                f"l_gate: tau_codes must be scalar or 1-D of length {arr.shape[axis]}, "
                f"got shape {tau.shape}"
            )
        shape = [1] * arr.ndim
        shape[axis] = tau.shape[0]
        tau_b = tau.reshape(shape)
    else:
        raise ValueError(f"l_gate: tau_codes must be scalar or 1-D, got ndim={tau.ndim}")

    gated = finite & (L.astype(np.int64) < tau_b)
    return np.where(gated, np.uint32(int(SPF32_WORD_ZERO)), arr).astype(np.uint32)


# Re-export for callers building gamma/tau code grids from float gauges.
def l_codes_from_float(x: npt.ArrayLike) -> np.ndarray:
    """Convert float log2-gauge values to integer L codes (round-half-away)."""
    codes = _half_away_round(np.asarray(x, dtype=np.float64) * float(1 << SPF32_L_FRAC_BITS))
    return np.clip(codes, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE).astype(np.int16)
