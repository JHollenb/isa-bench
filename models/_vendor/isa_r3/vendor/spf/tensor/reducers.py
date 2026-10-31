"""Legacy domain-routed SPF tensor reductions — the Stage 5 backend.

The established ``AUTO`` and ``PHASE_BINNED`` policies live here for ABI and
golden compatibility.  The v0.2 reference reducer is in ``spf.tensor.v02``;
new callers should select ``ReducerPolicy.V02`` or call ``reduce_words_v02``.

A tensor contraction ``C[i,k] = Σ_j A[i,j] ⊗ B[j,k]`` is two ops with opposite
cost in SPF: the elementwise **product** is free (L and Θ add), but the
**reduction over j** is the entire SPF addition wall. The wall is not paid the
same way everywhere, so this module selects an *accumulation domain* per
reduction from cheap stats, dispatches to the matching backend, and enforces
the cancellation/zero invariants in a domain-independent guard layer.

Three domains (``AccumulationDomain``):

* ``TROPICAL``  — max-plus: the result is the dominant term. Exact when one
  term dominates (large ``dom_gap``); ~0 expensive ops. Order-0 only here;
  the order-1 within-window tail is deferred (ship-order step 3).
* ``LOGPOLAR``  — native FB7 ⊕ folded as a balanced pairwise tree. Reuses the
  normative ``spf.scalar.spf32.add`` (FB7 poly7 ACCURATE) **unchanged** — it
  never reintroduces an Sb-LUT (see the ``spf-add-path`` skill). Benign for
  short, incoherent reductions; error grows ~√N from per-step requantization.
* ``CARTESIAN`` — decode in the pivoted ``(L − pivot_L)`` frame so ``|z| ≤ 1``
  (no FP64 overflow on high-L blocks), FP64 Kahan-accumulate, encode once.
  Error ≈ one-encode ulp, independent of N. The exact fallback.

This is the experimental tensor reference model (T7–T14), not part of the
normative T1–T6 spec; the taxonomy is pre-normative pending the
``benchmarks/tensor/reducer_nufft_lane`` validation. Pure Python — no native C.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from spf.scalar import spf32

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_SCALE,
    SPF32_L_ZERO,
    SPF32_PHASE_MASK,
    SPF32_PHASE_MOD,
    SPF32_WORD_ZERO,
    AccumulationDomain,
    PrecisionMode,
    ReducerPolicy,
    RoutingPolicy,
    SPFError,
    decode_words,
    encode_words,
    pack_words,
    unpack_L,
    unpack_theta,
)

if TYPE_CHECKING:
    from .core import ReducerSpec
    from .v2 import BlockMetadata

__all__ = [
    "Accumulator",
    "CartesianAccumulator",
    "GuardConfig",
    "LogPolarAccumulator",
    "ReductionResult",
    "ReductionStats",
    "RouteConfig",
    "TropicalAccumulator",
    "compute_stats",
    "domain_cost",
    "domain_error",
    "guard_cfg",
    "reduce_axis",
    "reduce_stream",
    "route",
    "route_cfg",
    "spec_cost",
    "spec_error",
]

_TWO_PI = 2.0 * math.pi
_ZERO_CODE = int(SPF32_L_ZERO)

# Expensive-op-equivalent unit costs (Stage-5 §9): C_add > C_dec > C_enc.
C_ADD = 3.0  # full FB7 correction (exp2 + log2 + sincos)
C_DEC = 2.0  # one decode (exp2 + sincos)
C_ENC = 1.0  # one encode (log2 + atan2)
# Order-of-magnitude relative ulp for the planner error estimate. SPF-32 carries
# 8 fractional L bits; this is a planner weight, not a measured bound.
_ULP = 2.0**-8


# --------------------------------------------------------------------------- #
# Stats — the cheap metadata the router reads (Stage-5 §1).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReductionStats:
    """O(N) (or O(1) from a block header) reduction metadata, all in log2 bits.

    ``N``        number of finite (non-zero) terms;
    ``pivot_L``  ``max(L)`` — the tropical / Cartesian block-float pivot;
    ``dLrange``  ``max(L) − min(L)`` — dynamic range (tropical norm);
    ``dom_gap``  ``max(L) − 2nd-max(L)`` — q of the worst add (``+inf`` if N<2);
    ``Rbar``     ``|Σ exp(iΘ)| / N`` in ``[0,1]`` — phase coherence.
    """

    N: int
    pivot_L: float
    dLrange: float
    dom_gap: float
    Rbar: float


def compute_stats(words: npt.ArrayLike) -> ReductionStats:
    """Single-pass :class:`ReductionStats` from a packed SPF word stream."""
    ww = np.asarray(words, dtype=np.uint32).reshape(-1)
    L = unpack_L(ww)
    finite = L != SPF32_L_ZERO
    n = int(np.count_nonzero(finite))
    if n == 0:
        return ReductionStats(N=0, pivot_L=-math.inf, dLrange=0.0, dom_gap=math.inf, Rbar=0.0)
    Lf = L[finite].astype(np.float64) / SPF32_L_SCALE
    lmax = float(Lf.max())
    lmin = float(Lf.min())
    if n == 1:
        dom_gap = math.inf
    else:
        # 2nd-largest via partition (O(N)); equal maxima -> dom_gap 0 (no single dominant).
        second = float(np.partition(Lf, -2)[-2])
        dom_gap = lmax - second
    angle = unpack_theta(ww)[finite].astype(np.float64) * (_TWO_PI / SPF32_PHASE_MOD)
    rbar = float(np.abs(np.sum(np.exp(1j * angle))) / n)
    return ReductionStats(N=n, pivot_L=lmax, dLrange=lmax - lmin, dom_gap=dom_gap, Rbar=rbar)


def _compute_stats_rows(words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized :class:`ReductionStats` fields for a row-major batch of streams."""
    if words.ndim != 2:
        raise SPFError("_compute_stats_rows expects a 2-D word matrix")
    n_rows = words.shape[0]
    L = unpack_L(words)
    finite = L != SPF32_L_ZERO
    n = np.count_nonzero(finite, axis=1).astype(np.int64)
    if words.shape[1] == 0:
        return (
            n,
            np.full(n_rows, -math.inf, dtype=np.float64),
            np.zeros(n_rows, dtype=np.float64),
            np.full(n_rows, math.inf, dtype=np.float64),
            np.zeros(n_rows, dtype=np.float64),
        )

    L_codes = L.astype(np.int64)
    masked_hi = np.where(finite, L_codes, np.int64(SPF32_L_MIN_FINITE) - 1)
    masked_lo = np.where(finite, L_codes, np.int64(SPF32_L_MAX_FINITE) + 1)
    lmax_code = masked_hi.max(axis=1)
    lmin_code = masked_lo.min(axis=1)

    pivot_L = np.where(n > 0, lmax_code.astype(np.float64) / SPF32_L_SCALE, -math.inf)
    dLrange = np.where(
        n > 0,
        (lmax_code - lmin_code).astype(np.float64) / SPF32_L_SCALE,
        0.0,
    )

    if words.shape[1] >= 2:
        top2 = np.partition(masked_hi, -2, axis=1)[:, -2:]
        top2.sort(axis=1)
        dom_gap = (top2[:, 1] - top2[:, 0]).astype(np.float64) / SPF32_L_SCALE
    else:
        dom_gap = np.full(n_rows, math.inf, dtype=np.float64)
    dom_gap = np.where(n < 2, math.inf, dom_gap)

    angle = unpack_theta(words).astype(np.float64) * (_TWO_PI / SPF32_PHASE_MOD)
    zsum = np.sum(np.where(finite, np.exp(1j * angle), 0.0 + 0.0j), axis=1)
    Rbar = np.divide(
        np.abs(zsum),
        n,
        out=np.zeros(n_rows, dtype=np.float64),
        where=n > 0,
    )
    return n, pivot_L, dLrange, dom_gap, Rbar


def from_block_metadata(meta: BlockMetadata) -> ReductionStats:
    """Adapt a v2 :class:`BlockMetadata` to :class:`ReductionStats` (O(1) header path).

    Reads only the five fields ``finite_count``, ``pivot_L``, ``dynamic_range``,
    ``dominant_gap`` and ``phase_coherence`` (a TYPE_CHECKING-only import keeps the
    runtime decoupled from ``v2``).
    """
    n = int(meta.finite_count)
    if n == 0:
        return ReductionStats(N=0, pivot_L=-math.inf, dLrange=0.0, dom_gap=math.inf, Rbar=0.0)
    dom_gap = math.inf if n < 2 else float(meta.dominant_gap)
    return ReductionStats(
        N=n,
        pivot_L=float(meta.pivot_L),
        dLrange=float(meta.dynamic_range),
        dom_gap=dom_gap,
        Rbar=float(meta.phase_coherence),
    )


# --------------------------------------------------------------------------- #
# Routing (Stage-5 §3.1).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RouteConfig:
    tropical_gap: float  # dom_gap (bits) at/above which one term dominates
    short_n: int  # N at/below which a benign sum stays in LOGPOLAR
    cancel_coherence_max: float  # coherence at/below which a near-equal-mag sum cancels
    cancel_dL: float  # dynamic range (bits) below which the cancel test applies


def _mode_alias(mode: PrecisionMode) -> PrecisionMode:
    """EXACT aliases ACCURATE (both force the exact CARTESIAN domain)."""
    return PrecisionMode.ACCURATE if mode == PrecisionMode.EXACT else PrecisionMode(mode)


def route_cfg(mode: PrecisionMode) -> RouteConfig:
    m = _mode_alias(mode)
    if m == PrecisionMode.FAST:
        return RouteConfig(tropical_gap=6.0, short_n=16, cancel_coherence_max=0.30, cancel_dL=1.0)
    if m == PrecisionMode.BALANCED:
        return RouteConfig(tropical_gap=8.0, short_n=8, cancel_coherence_max=0.50, cancel_dL=2.0)
    # ACCURATE / EXACT: tropical disabled, short-circuit disabled -> always CARTESIAN.
    return RouteConfig(tropical_gap=math.inf, short_n=0, cancel_coherence_max=1.0, cancel_dL=math.inf)


def route(stats: ReductionStats, mode: PrecisionMode) -> AccumulationDomain:
    """Select the cheapest sufficient accumulation domain from ``stats`` (Stage-5 §3.1).

    The whole-tail bound ``(N-1) * 2**(-dom_gap)`` selects a budgeted
    TROPICAL approximation. Otherwise a
    short reduction stays in native LOGPOLAR *unless* it is cancellation-prone, in
    which case it goes to exact CARTESIAN; long reductions also go to CARTESIAN.

    Note — corrected cancellation predicate: the design doc framed *high* coherence
    (``Rbar`` near 1) ∧ small ``dLrange`` as the cancellation signal. That is
    inverted (the same inversion fixed in the Stage-4 ``cancellation_risk`` screen):
    high coherence is aligned/reinforcing phases — the numerically easy case; the sum
    stays large. Cancellation needs *opposing* phases, i.e. **low** ``Rbar`` with
    near-equal magnitude, where ``|Σz|`` collapses far below ``Σ|z|`` and only the
    exact Cartesian pass keeps significance. This matches the spec cancellation
    wedge and the Stage-4 screen (``Rbar ≤ 0.5 ∧ dLrange < 1.0``).
    """
    cfg = route_cfg(mode)
    if stats.N == 0:
        return AccumulationDomain.CARTESIAN
    if math.isfinite(cfg.tropical_gap) and stats.dom_gap >= cfg.tropical_gap + math.log2(max(stats.N - 1, 1)):
        return AccumulationDomain.TROPICAL
    cancellation_prone = stats.Rbar <= cfg.cancel_coherence_max and stats.dLrange < cfg.cancel_dL
    if cfg.short_n >= stats.N and not cancellation_prone:
        return AccumulationDomain.LOGPOLAR
    return AccumulationDomain.CARTESIAN


# --------------------------------------------------------------------------- #
# Guard layer (domain-independent invariants — always on, Stage-5 §6).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GuardConfig:
    dom_bypass_gap: float  # legacy diagnostic; no authority to discard finite terms
    cancel_bits: float  # legacy diagnostic threshold; never authority to erase a result
    phase_unreliable_bits: float  # result this far below pivot -> PHASE_UNRELIABLE flag


def guard_cfg(mode: PrecisionMode) -> GuardConfig:
    # Calibrated to SPF-32 resolution (~2^-8 magnitude, ~2^-16 phase). Exact
    # cancellation is caught by the encoder's zero (|acc|==0); ``cancel_bits`` is
    # retained for compatibility, with no authority to erase a small result,
    # while ``phase_unreliable_bits`` sits in the reachable deep-cancellation band
    # where the residual phase is rounding-dominated and the flag is meaningful.
    m = _mode_alias(mode)
    if m == PrecisionMode.FAST:
        return GuardConfig(dom_bypass_gap=24.0, cancel_bits=40.0, phase_unreliable_bits=8.0)
    if m == PrecisionMode.BALANCED:
        return GuardConfig(dom_bypass_gap=32.0, cancel_bits=48.0, phase_unreliable_bits=12.0)
    # ACCURATE / EXACT: legacy thresholds remain diagnostic only.
    return GuardConfig(dom_bypass_gap=40.0, cancel_bits=52.0, phase_unreliable_bits=16.0)


def _apply_dom_bypass(words: np.ndarray, gap_bits: float) -> np.ndarray:
    """Remove only exact zeros; a pivot-relative tail may survive cancellation.

    Lossy dominance belongs to the routed TROPICAL consumer with a whole-tail
    bound. A global cutoff inside other reducers cannot certify output error.
    ``gap_bits`` remains a compatibility argument.
    """
    return words[unpack_L(words) != _ZERO_CODE]


def _guarded_result(word: int, pivot_L: float, cfg: GuardConfig) -> tuple[np.uint32, bool]:
    """Apply the cancellation / phase-reliability invariants to a finalized word."""
    w = int(word)
    lc = int(unpack_L(np.asarray([w], dtype=np.uint32))[0])
    if lc == _ZERO_CODE:
        return SPF32_WORD_ZERO, False
    res_bits = lc / SPF32_L_SCALE
    below = pivot_L - res_bits  # how far the result sits below the dominant term
    if below >= cfg.phase_unreliable_bits:
        return np.uint32(w), True
    return np.uint32(w), False


def _apply_dom_bypass_mask_rows(words: np.ndarray, gap_bits: float) -> np.ndarray:
    """Keep all nonzero terms; do not erase a cancellation residual."""
    return unpack_L(words) != _ZERO_CODE


def _guarded_result_rows(words: np.ndarray, pivot_L: np.ndarray, cfg: GuardConfig) -> np.ndarray:
    """Vectorized cancellation / phase-reliability guard. The advisory flag is dropped."""
    out = np.asarray(words, dtype=np.uint32).copy()
    lc = unpack_L(out).astype(np.int64)
    zero = lc == _ZERO_CODE
    out[zero] = SPF32_WORD_ZERO
    return out


def _route_rows(
    n: np.ndarray,
    dLrange: np.ndarray,
    dom_gap: np.ndarray,
    Rbar: np.ndarray,
    mode: PrecisionMode,
) -> np.ndarray:
    """Vectorized :func:`route`, returning int-valued :class:`AccumulationDomain`s."""
    cfg = route_cfg(mode)
    domain = np.full(n.shape, int(AccumulationDomain.CARTESIAN), dtype=np.int64)
    tropical = math.isfinite(cfg.tropical_gap) & (n > 0) & (dom_gap >= cfg.tropical_gap + np.log2(np.maximum(n - 1, 1)))
    domain[tropical] = int(AccumulationDomain.TROPICAL)
    cancellation_prone = (Rbar <= cfg.cancel_coherence_max) & (dLrange < cfg.cancel_dL)
    logpolar = (n > 0) & ~tropical & (cfg.short_n >= n) & ~cancellation_prone
    domain[logpolar] = int(AccumulationDomain.LOGPOLAR)
    return domain


# --------------------------------------------------------------------------- #
# Backend kernels (single source of truth for the fold math).
# --------------------------------------------------------------------------- #


def _finite_words(words: np.ndarray) -> np.ndarray:
    L = unpack_L(words)
    return words[L != SPF32_L_ZERO]


def _tropical_reduce(words: np.ndarray) -> np.uint32:
    """Order-0 max-plus: return the dominant (largest-L) term, else ZERO."""
    L = unpack_L(words)
    finite = L != SPF32_L_ZERO
    if not finite.any():
        return SPF32_WORD_ZERO
    masked = np.where(finite, L.astype(np.int64), np.int64(SPF32_L_MIN_FINITE) - 1)
    return np.uint32(words[int(np.argmax(masked))])


def _logpolar_reduce(words: np.ndarray) -> tuple[np.uint32, int]:
    """Drop-disabled balanced pairwise FB7 fold via vectorized ``spf32.add``.

    Returns ``(word, n_log_adds)``. No tropical-drop — every finite term folds,
    so this is the honest always-LOGPOLAR baseline the NUFFT lane needs.
    """
    live = _finite_words(words).astype(np.uint32)
    if live.size == 0:
        return SPF32_WORD_ZERO, 0
    if live.size == 1:
        return np.uint32(live[0]), 0
    n_adds = 0
    while live.size > 1:
        if live.size & 1:
            tail = live[-1:]
            live = live[:-1]
        else:
            tail = np.empty(0, dtype=np.uint32)
        merged = spf32.add(live[0::2], live[1::2])
        n_adds += int(live.size // 2)
        live = np.concatenate([np.asarray(merged, dtype=np.uint32), tail])
    return np.uint32(live[0]), n_adds


def _unit_phasors(theta: np.ndarray) -> np.ndarray:
    """Exact antipodal symmetry and named axes in the discrete phase ring."""
    codes = np.asarray(theta, dtype=np.uint32)
    half = codes & 32767
    result = np.exp(1j * half.astype(np.float64) * (_TWO_PI / SPF32_PHASE_MOD))
    result = np.where(half == 0, 1.0 + 0j, result)
    result = np.where(half == 16384, 1j, result)
    return np.where((codes & 32768) != 0, -result, result)


def _cartesian_reduce(words: np.ndarray) -> np.uint32:
    """Pivoted Kahan: decode in the ``(L − pivot)`` frame, sum, encode once."""
    L = unpack_L(words)
    finite = L != SPF32_L_ZERO
    if not finite.any():
        return SPF32_WORD_ZERO
    Lc = L[finite].astype(np.int64)
    pivot = int(Lc.max())
    # Pivoted L codes are <= 0; floor at the finite minimum so they never alias
    # the zero sentinel. Terms below 2^-127 of the pivot are FP64-negligible.
    Lp = (Lc - pivot).astype(np.float64) / SPF32_L_SCALE
    mag = np.exp2(Lp)  # |z| in (0, 1]
    z = mag * _unit_phasors(unpack_theta(words)[finite])
    s = _kahan_sum(z)
    smag = abs(s)
    if smag == 0.0:
        return SPF32_WORD_ZERO
    # Finite-saturating encode, identical to the reference encoder
    # (core.encode_complex via _sat_l16): a magnitude overflow saturates to the
    # largest finite L, it does NOT emit the +inf sentinel (0x7FFF) — that code is
    # reserved for explicit infinities, never for an encode overflow. Pivoting keeps
    # |z| ≤ 1 so log2(smag) ≤ log2(N); the upper clamp is unreachable for any N that
    # does not overflow the reference encoder either.
    l_code = round(math.log2(smag) * SPF32_L_SCALE) + pivot
    l_code = max(SPF32_L_MIN_FINITE, min(SPF32_L_MAX_FINITE, l_code))
    t_code = round((math.atan2(s.imag, s.real) % _TWO_PI) * (SPF32_PHASE_MOD / _TWO_PI))
    return np.uint32(
        pack_words(np.asarray(l_code, dtype=np.int16), np.asarray(t_code & SPF32_PHASE_MASK, dtype=np.uint16))
    )


def _tropical_reduce_rows(words: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Row-batched max-plus reduce over already guarded/bypassed terms."""
    out = np.full(words.shape[0], SPF32_WORD_ZERO, dtype=np.uint32)
    if words.shape[0] == 0 or words.shape[1] == 0:
        return out
    L = unpack_L(words).astype(np.int64)
    masked = np.where(keep, L, np.int64(SPF32_L_MIN_FINITE) - 1)
    any_keep = keep.any(axis=1)
    if np.any(any_keep):
        row_idx = np.flatnonzero(any_keep)
        col_idx = np.argmax(masked[row_idx], axis=1)
        out[row_idx] = words[row_idx, col_idx]
    return out


def _logpolar_fold_matrix(live: np.ndarray, *, integer_only: bool = False) -> np.ndarray:
    """Balanced FB7 fold for a matrix whose rows all have the same live length."""
    if live.shape[1] == 0:
        return np.full(live.shape[0], SPF32_WORD_ZERO, dtype=np.uint32)
    cur = np.asarray(live, dtype=np.uint32)
    while cur.shape[1] > 1:
        if cur.shape[1] & 1:
            tail = cur[:, -1:]
            body = cur[:, :-1]
        else:
            tail = np.empty((cur.shape[0], 0), dtype=np.uint32)
            body = cur
        if integer_only:
            from .factored_add import spf_pair_addsub_factored_planes
            merged, _ = spf_pair_addsub_factored_planes(body[:, 0::2], body[:, 1::2], integer_only=True)
        else:
            merged = spf32.add(body[:, 0::2], body[:, 1::2])
        cur = np.concatenate([np.asarray(merged, dtype=np.uint32), tail], axis=1)
    return cur[:, 0].astype(np.uint32, copy=False)


def _logpolar_reduce_rows(words: np.ndarray, keep: np.ndarray, *, integer_only: bool = False) -> np.ndarray:
    """Row-batched LOGPOLAR reduce, compacting each row before the pairwise fold."""
    out = np.full(words.shape[0], SPF32_WORD_ZERO, dtype=np.uint32)
    counts = np.count_nonzero(keep, axis=1)
    for cnt in np.unique(counts):
        c = int(cnt)
        if c == 0:
            continue
        rows = np.flatnonzero(counts == c)
        sub_keep = keep[rows]
        sub_words = words[rows]
        compact = np.empty((rows.size, c), dtype=np.uint32)
        rr, cc = np.nonzero(sub_keep)
        pos = np.cumsum(sub_keep, axis=1)[rr, cc] - 1
        compact[rr, pos] = sub_words[rr, cc]
        out[rows] = compact[:, 0] if c == 1 else _logpolar_fold_matrix(compact, integer_only=integer_only)
    return out


def _cartesian_reduce_rows(words: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Row-batched pivoted Cartesian reduce; matches :func:`_cartesian_reduce` per row."""
    out = np.full(words.shape[0], SPF32_WORD_ZERO, dtype=np.uint32)
    if words.shape[0] == 0 or words.shape[1] == 0:
        return out
    any_keep = keep.any(axis=1)
    if not np.any(any_keep):
        return out
    sub_words = words[any_keep]
    sub_keep = keep[any_keep]
    L = unpack_L(sub_words).astype(np.int64)
    pivot = np.max(np.where(sub_keep, L, np.int64(SPF32_L_MIN_FINITE)), axis=1)
    Lp = (L - pivot[:, None]).astype(np.float64) / SPF32_L_SCALE
    mag = np.where(sub_keep, np.exp2(Lp), 0.0)
    s = np.sum(mag * _unit_phasors(unpack_theta(sub_words)), axis=1)
    smag = np.abs(s)
    zero = smag == 0.0
    l_code = np.rint(np.log2(np.where(zero, 1.0, smag)) * SPF32_L_SCALE).astype(np.int64) + pivot
    l_code = np.clip(l_code, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE).astype(np.int16)
    t_code = np.rint((np.angle(s) % _TWO_PI) * (SPF32_PHASE_MOD / _TWO_PI)).astype(np.int64)
    t_code = (t_code & SPF32_PHASE_MASK).astype(np.uint16)
    l_code = np.where(zero, SPF32_L_ZERO, l_code).astype(np.int16)
    t_code = np.where(zero, 0, t_code).astype(np.uint16)
    out[np.flatnonzero(any_keep)] = pack_words(l_code, t_code)
    return out


def _kahan_sum(values: np.ndarray) -> complex:
    """Pairwise (numpy) complex summation in FP64.

    Replaces a scalar Kahan loop: numpy's pairwise reduction is vectorized
    (NEON/Accelerate on this host) and accurate to O(log N) ulp — ample headroom
    below the integer re-encode in ``_cartesian_reduce``. Summation order differs
    from Kahan, so low-bit CARTESIAN reduction words can shift by a unit; the
    reduction goldens were regenerated for this change.
    """
    return complex(np.sum(np.asarray(values, dtype=np.complex128)))


_BACKENDS = {
    AccumulationDomain.TROPICAL: lambda w: (_tropical_reduce(w), 0),
    AccumulationDomain.LOGPOLAR: _logpolar_reduce,
    AccumulationDomain.CARTESIAN: lambda w: (_cartesian_reduce(w), 0),
}


def _reduce_rows_auto(rows: np.ndarray, spec: ReducerSpec) -> np.ndarray:
    """Batched :func:`reduce_stream` word path for ``ReducerPolicy.AUTO`` specs."""
    n, pivot_L, dLrange, dom_gap, Rbar = _compute_stats_rows(rows)
    out = np.full(rows.shape[0], SPF32_WORD_ZERO, dtype=np.uint32)
    nonzero = n > 0
    if not np.any(nonzero):
        return out

    mode = spec.precision_mode
    if spec.routing == RoutingPolicy.PINNED:
        if spec.domain is None:
            raise SPFError("PINNED routing requires an explicit ReducerSpec.domain")
        domain = np.full(rows.shape[0], int(AccumulationDomain(spec.domain)), dtype=np.int64)
    elif spec.routing == RoutingPolicy.STATIC:
        domain = _route_rows(n, dLrange, dom_gap, Rbar, mode)
    else:
        raise SPFError("ADAPTIVE routing is deferred (ship-order step 3); use STATIC or PINNED")

    gcfg = guard_cfg(mode)
    keep = _apply_dom_bypass_mask_rows(rows, gcfg.dom_bypass_gap)
    for dom in AccumulationDomain:
        sel = nonzero & (domain == int(dom))
        if not np.any(sel):
            continue
        if dom == AccumulationDomain.TROPICAL:
            words = _tropical_reduce_rows(rows[sel], keep[sel])
        elif dom == AccumulationDomain.LOGPOLAR:
            words = _logpolar_reduce_rows(rows[sel], keep[sel])
        else:
            words = _cartesian_reduce_rows(rows[sel], keep[sel])
        out[sel] = _guarded_result_rows(words, pivot_L[sel], gcfg)
    return out


def _reduce_rows_cartesian_kahan(rows: np.ndarray) -> np.ndarray:
    """Batched legacy CARTESIAN_KAHAN reducer used by :func:`core.reduce_words`."""
    if rows.shape[1] == 0:
        return np.full(rows.shape[0], SPF32_WORD_ZERO, dtype=np.uint32)
    summed = np.sum(decode_words(rows), axis=1)
    return encode_words(summed).astype(np.uint32, copy=False)


def _fallback_reduce_rows(rows: np.ndarray, spec: ReducerSpec) -> np.ndarray:
    """Compatibility path for reducer policies that do not yet have a row kernel."""
    from .core import reduce_words

    out = np.empty(rows.shape[0], dtype=np.uint32)
    for i in range(rows.shape[0]):
        out[i] = reduce_words(rows[i], spec)
    return out


def reduce_axis(
    terms: npt.ArrayLike,
    spec: ReducerSpec,
    axis: int | None = None,
) -> np.ndarray | np.uint32:
    """Reduce packed SPF words over ``axis`` with row-batched kernels.

    This is byte-for-byte equivalent to calling :func:`core.reduce_words` on each
    output slice, but avoids one Python reducer dispatch per row for the common
    CARTESIAN_KAHAN and AUTO domains.
    """
    arr = np.asarray(terms, dtype=np.uint32)
    if axis is None:
        rows = arr.reshape(1, -1)
        out_shape: tuple[int, ...] = ()
    else:
        ax = int(axis)
        moved = np.moveaxis(arr, ax, -1)
        out_shape = moved.shape[:-1]
        rows = moved.reshape(-1, moved.shape[-1])

    if spec.policy == ReducerPolicy.AUTO:
        flat = _reduce_rows_auto(rows, spec)
    elif spec.policy == ReducerPolicy.PHASE_BINNED:
        flat = _reduce_rows_phase_binned(rows, spec)
    elif spec.policy == ReducerPolicy.CARTESIAN_KAHAN:
        flat = _reduce_rows_cartesian_kahan(rows)
    else:
        flat = _fallback_reduce_rows(rows, spec)

    if axis is None:
        return np.uint32(flat[0])
    return flat.reshape(out_shape)


# --------------------------------------------------------------------------- #
# Streaming accumulator API (Stage-5 §4) — thin buffering shims over the
# backend kernels above, kept for design fidelity and direct unit testing.
# --------------------------------------------------------------------------- #


class Accumulator:
    """Common accumulator interface: ``reset`` / ``add`` / ``result``."""

    domain: AccumulationDomain

    def __init__(self) -> None:
        self._buf: list[int] = []

    def reset(self) -> None:
        self._buf = []

    def add(self, word: int) -> None:
        self._buf.append(int(word))

    def _words(self) -> np.ndarray:
        return np.asarray(self._buf, dtype=np.uint32)

    def result(self) -> np.uint32:
        if not self._buf:
            return SPF32_WORD_ZERO
        word, _ = _BACKENDS[self.domain](self._words())
        return word


class TropicalAccumulator(Accumulator):
    domain = AccumulationDomain.TROPICAL


class LogPolarAccumulator(Accumulator):
    domain = AccumulationDomain.LOGPOLAR

    def n_log_adds(self) -> int:
        if not self._buf:
            return 0
        return _logpolar_reduce(self._words())[1]


class CartesianAccumulator(Accumulator):
    domain = AccumulationDomain.CARTESIAN


# --------------------------------------------------------------------------- #
# Result + driver (Stage-5 §7).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ReductionResult:
    """One reduction's outcome plus the out-of-band advisory flag.

    ``phase_unreliable`` lives here because a uint32 SPF word has no spare bit —
    callers that want it must use :func:`reduce_stream` / ``reduce_words_result``.

    ``fp_boundary_decodes`` is the count of terms decoded to FP64 — the expensive
    SPF→Cartesian boundary the phase-binned policy is built to shrink. It is
    populated only by ``ReducerPolicy.PHASE_BINNED`` (the cross-bin combine plus
    any within-bin CARTESIAN bins); every other path leaves it 0.
    """

    word: np.uint32
    domain: AccumulationDomain
    phase_unreliable: bool
    n_log_adds: int
    fp_boundary_decodes: int = 0


def _as_word_array(terms: npt.ArrayLike) -> np.ndarray:
    if isinstance(terms, np.ndarray):
        return terms.astype(np.uint32, copy=False).reshape(-1)
    return np.fromiter((int(t) for t in terms), dtype=np.uint32)


def reduce_stream(
    term_iter: npt.ArrayLike,
    spec: ReducerSpec,
    stats: ReductionStats | None = None,
) -> ReductionResult:
    """Driver (Stage-5 §7): route, guard-bypass, dispatch, guard-finalize."""
    words = _as_word_array(term_iter)
    if words.size == 0:
        return ReductionResult(SPF32_WORD_ZERO, AccumulationDomain.CARTESIAN, False, 0)
    st = stats if stats is not None else compute_stats(words)
    if st.N == 0:  # all-zero stream: nothing to add
        return ReductionResult(SPF32_WORD_ZERO, AccumulationDomain.CARTESIAN, False, 0)

    if spec.policy == ReducerPolicy.PHASE_BINNED:
        return _phase_binned_reduce(words, spec, st)

    mode = spec.precision_mode
    if spec.routing == RoutingPolicy.PINNED:
        if spec.domain is None:
            raise SPFError("PINNED routing requires an explicit ReducerSpec.domain")
        domain = AccumulationDomain(spec.domain)
    elif spec.routing == RoutingPolicy.STATIC:
        domain = route(st, mode)
    else:  # ADAPTIVE
        raise SPFError("ADAPTIVE routing is deferred (ship-order step 3); use STATIC or PINNED")

    gcfg = guard_cfg(mode)
    kept = _apply_dom_bypass(words, gcfg.dom_bypass_gap)
    word, n_adds = _BACKENDS[domain](kept)
    word, flag = _guarded_result(word, st.pivot_L, gcfg)
    return ReductionResult(word=word, domain=domain, phase_unreliable=flag, n_log_adds=n_adds)


# --------------------------------------------------------------------------- #
# Phase-binned reduction (Stage-5 phase-coherence split, ReducerPolicy.PHASE_BINNED).
# --------------------------------------------------------------------------- #


def _auto_subspec(spec: ReducerSpec) -> ReducerSpec:
    """The AUTO/STATIC spec each phase bin is reduced through (route()+guard)."""
    from .core import ReducerSpec as _ReducerSpec  # core is fully loaded by call time

    return _ReducerSpec(
        policy=ReducerPolicy.AUTO,
        precision_mode=spec.precision_mode,
        routing=RoutingPolicy.STATIC,
    )


def _phase_bins(theta: np.ndarray, n_bins: int) -> np.ndarray:
    """Map u16 phase codes to ``n_bins`` contiguous arcs (matches _phase_histogram_reduce).

    Arcs wrap automatically with the u16 phase, so terms straddling the 0/2π seam
    land in bins 0 and ``n_bins-1`` (correct — they are *not* seam-merged; an
    optional dominant-phase grid rotation that would merge them is deferred).
    """
    return (theta.astype(np.int64) * int(n_bins)) // SPF32_PHASE_MOD


def _phase_binned_reduce(words: np.ndarray, spec: ReducerSpec, st: ReductionStats) -> ReductionResult:
    """Bin finite terms by Θ, reduce each bin via route()+guard, combine ≤B with CARTESIAN.

    The phase-coherence split keeps every cheap FB7 add inside a coherent arc (its
    accurate regime) and confines the FP64 boundary to the ≤B bin-sum words plus
    any bin the router still sends to CARTESIAN. ``n_log_adds`` is the honest sum
    of within-bin FB7 folds (the cross-bin combine is CARTESIAN → 0 FB7 adds);
    ``fp_boundary_decodes`` counts the terms actually decoded to FP64.
    """
    b = int(spec.n_phase_bins)
    L = unpack_L(words)
    fwords = words[L != SPF32_L_ZERO]
    if fwords.size == 0:  # defensive: st.N>0 implies finite terms exist
        return ReductionResult(SPF32_WORD_ZERO, AccumulationDomain.CARTESIAN, False, 0, 0)

    sub = _auto_subspec(spec)
    gcfg = guard_cfg(spec.precision_mode)
    binidx = _phase_bins(unpack_theta(fwords), b)

    bin_results: list[tuple[int, AccumulationDomain]] = []
    n_log_adds = 0
    fp_decodes = 0
    for bb in np.unique(binidx):
        members = fwords[binidx == bb]
        res = reduce_stream(members, sub)
        if int(res.word) == int(SPF32_WORD_ZERO):
            continue  # bin cancelled to the zero sentinel — contributes nothing
        bin_results.append((int(res.word), res.domain))
        n_log_adds += res.n_log_adds
        if res.domain == AccumulationDomain.CARTESIAN:
            fp_decodes += int(members.size)  # this bin decoded its own terms to FP64

    n_occ = len(bin_results)
    if n_occ == 0:
        return ReductionResult(SPF32_WORD_ZERO, AccumulationDomain.CARTESIAN, False, 0, fp_decodes)
    if n_occ == 1:
        # One occupied bin → one cheap reduction, no cross-bin pass. The lone bin
        # holds every finite term, so its pivot is the stream pivot and its guard
        # result is already final (re-guard here is idempotent, recovers the flag).
        w0, d0 = bin_results[0]
        word, flag = _guarded_result(w0, st.pivot_L, gcfg)
        return ReductionResult(word, d0, flag, n_log_adds, fp_decodes)

    # Cross-bin combine: the one guaranteed FP64 pass over the ≤B bin-sum words.
    fp_decodes += n_occ
    bin_w = np.asarray([w for w, _ in bin_results], dtype=np.uint32)
    combined = _cartesian_reduce(bin_w)
    word, flag = _guarded_result(combined, st.pivot_L, gcfg)
    return ReductionResult(word, AccumulationDomain.CARTESIAN, flag, n_log_adds, fp_decodes)


def _reduce_rows_phase_binned(rows: np.ndarray, spec: ReducerSpec) -> np.ndarray:
    """Batched phase-binned reduce; byte-identical to :func:`_phase_binned_reduce` per row.

    Each row's finite terms are scattered into a ``(R*B, N)`` bin-matrix (one row
    per (row, bin) pair, members kept at their original columns; zero-sentinel pad
    is inert to every kernel). The bins reduce through the shared AUTO row path
    :func:`_reduce_rows_auto`, then the ≤B bin sums combine per row with the same
    pivoted CARTESIAN kernel the scalar driver uses — so the two paths fold the
    same values in the same order and agree to the bit.
    """
    r, n = rows.shape
    b = int(spec.n_phase_bins)
    out = np.full(r, SPF32_WORD_ZERO, dtype=np.uint32)
    if r == 0 or n == 0:
        return out

    theta = unpack_theta(rows)
    finite = unpack_L(rows) != SPF32_L_ZERO
    binidx = _phase_bins(theta, b)

    # Scatter members to (row*B + bin); distinct original columns never collide.
    binmat = np.full((r * b, n), SPF32_WORD_ZERO, dtype=np.uint32)
    rr, cc = np.nonzero(finite)
    binmat[rr * b + binidx[rr, cc], cc] = rows[rr, cc]

    binsum = _reduce_rows_auto(binmat, _auto_subspec(spec)).reshape(r, b)
    occupied = binsum != np.uint32(SPF32_WORD_ZERO)
    n_occ = np.count_nonzero(occupied, axis=1)

    single = n_occ == 1
    if np.any(single):
        col = np.argmax(occupied[single], axis=1)
        out[np.flatnonzero(single)] = binsum[np.flatnonzero(single), col]

    multi = n_occ >= 2
    if np.any(multi):
        gcfg = guard_cfg(spec.precision_mode)
        _, pivot_L, _, _, _ = _compute_stats_rows(rows)
        rows_m = np.flatnonzero(multi)
        combined = _cartesian_reduce_rows(binsum[rows_m], occupied[rows_m])
        out[rows_m] = _guarded_result_rows(combined, pivot_L[rows_m], gcfg)
    return out


# --------------------------------------------------------------------------- #
# Planner cost / error contracts (Stage-5 §9).
# --------------------------------------------------------------------------- #


def domain_cost(domain: AccumulationDomain, stats: ReductionStats) -> float:
    """Expensive-op-equivalent cost of reducing ``stats.N`` terms in ``domain``."""
    if domain == AccumulationDomain.TROPICAL:
        return 0.0  # order-0
    if domain == AccumulationDomain.LOGPOLAR:
        return max(stats.N - 1, 0) * C_ADD
    return stats.N * C_DEC + C_ENC  # CARTESIAN


def domain_error(domain: AccumulationDomain, stats: ReductionStats) -> float:
    """Relative-error estimate (planner weight) for ``domain`` on ``stats``."""
    if domain == AccumulationDomain.TROPICAL:
        # Bound relative to the retained pivot, then use the reverse triangle
        # inequality for relative error against a possibly cancelling sum.
        tail = 0.0 if math.isinf(stats.dom_gap) else max(stats.N - 1, 0) * 2.0 ** (-stats.dom_gap)
        return tail / (1.0 - tail) if tail < 1.0 else math.inf
    if domain == AccumulationDomain.LOGPOLAR:
        return math.sqrt(max(stats.N, 1)) * _ULP
    return _ULP  # CARTESIAN — N-independent


def _phase_binned_cost(spec: ReducerSpec, stats: ReductionStats) -> float:
    """Within-bin FB7 folds + a ≤B-term CARTESIAN combine (Stage-5 §9).

    ``occ ≈ min(B, N)`` occupied bins fold ``N − occ`` terms cheaply, then the
    occupied bin sums decode once each (``occ`` decodes) plus one encode. Bin
    occupancy is data-dependent; this assumes the worst-case fully-spread case.
    """
    occ = min(int(spec.n_phase_bins), max(stats.N, 0))
    return max(stats.N - occ, 0) * C_ADD + occ * C_DEC + C_ENC


def _phase_binned_error(spec: ReducerSpec, stats: ReductionStats) -> float:
    """Approximate, data-dependent error weight for PHASE_BINNED.

    Deliberately *not* CARTESIAN's N-independent ``_ULP``: phase-binning re-encodes
    each within-bin sum before the cross-bin combine, so its error depends on how
    terms distribute across arcs and can exceed single-shot CARTESIAN on
    cancellation-dominated sums. The √ terms are an order-of-magnitude planner
    weight (within-bin requant ~√(N/occ), cross-bin combine ~√occ), not a bound.
    """
    occ = max(min(int(spec.n_phase_bins), max(stats.N, 1)), 1)
    return _ULP * (math.sqrt(max(stats.N / occ, 1.0)) + math.sqrt(occ))


def _resolve_domain(spec: ReducerSpec, stats: ReductionStats) -> AccumulationDomain:
    if spec.policy != ReducerPolicy.AUTO:
        return AccumulationDomain.CARTESIAN
    if spec.routing == RoutingPolicy.PINNED and spec.domain is not None:
        return AccumulationDomain(spec.domain)
    return route(stats, spec.precision_mode)


def spec_cost(spec: ReducerSpec, stats: object) -> float:
    if not isinstance(stats, ReductionStats):
        raise SPFError("spec.cost expects a ReductionStats")
    if spec.policy == ReducerPolicy.PHASE_BINNED:
        return _phase_binned_cost(spec, stats)
    return domain_cost(_resolve_domain(spec, stats), stats)


def spec_error(spec: ReducerSpec, stats: object) -> float:
    if not isinstance(stats, ReductionStats):
        raise SPFError("spec.error expects a ReductionStats")
    if spec.policy == ReducerPolicy.PHASE_BINNED:
        return _phase_binned_error(spec, stats)
    return domain_error(_resolve_domain(spec, stats), stats)
