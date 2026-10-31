"""Factored tan-tanh SPF32 complex pair adder (blog post 2026-09-30-135722, finding 1).

This is a *table-only* pair adder: it evaluates ``(u + t, u - t)`` from five 1-D
integer ROMs glued by integer adds, with **zero** ``exp``/``log``/``sin``/``atan``
calls on the generic path.  It is an alternative to
:func:`spf.tensor.v02.spf_pair_addsub_hybrid` (the FB3-Hybrid butterfly) with the
same word contract.

Identity (DERIVED in the post, CHECKED here).  With ``A`` the larger operand,
``g = L_A - L_B >= 0``, ``q = 2^{-g/256}``, ``dTheta = cd(Theta_B, Theta_A)``,
``delta = 2 pi dTheta / 65536``, ``h = delta/2``::

    1 + q e^{i delta} = e^{ih} (1+q) cos h (1 - i y),   y = tanh(g ln2/512) tan h

Taking ``256 log2`` of each factor turns the 2-D correction into a sum of three
1-D pieces glued by one integer add (``eta = TH(g) +/- LT(dTheta)``)::

    dL+ = LP(g) + LC(dTheta) + F(eta+),   dTheta+ = dTheta/2 - s G(eta+)
    dL- = LP(g) + LS(dTheta) + F(eta-),   dTheta- = dTheta/2 + s (G(eta-) - 16384)

where ``LS = LC + LT``, ``F(eta) = 128 log2(1 + 2^{2 eta}) `` and
``G(eta) = (65536/2pi) atan(2^eta)`` (post B.2).  All tables are integer, in
guard units (``2^-GV`` ticks); see :mod:`spf.tensor.factored_add_rom`.

Exact phase classes (same/opposite phase, equal magnitude, quadrature, zero, NaN)
are dispatched to :func:`spf.tensor.v02._pair_local` and the canonical
``logadd_pos`` / ``logsub_pos`` Lambda+/- ROMs, so those stay **bit-exact** with
the FB3-Hybrid adder.  The factored tables carry the generic case *and* the
near-cancellation case (delta near pi), which is accurate here because the huge
opposite-signed pieces (``LC -> -inf``, ``LT -> +inf``) are integers that cancel
exactly -- no two nearly-equal reals are ever subtracted.

The final phase word is rounded RNE on the *total* half-tick phase (the parity of
``Theta_A`` matters at a tie), matching ``spf.tensor.v02._rne_half_ticks``.

Status: **CANDIDATE, not conforming** (Algorithm Reference v0.2 §0 rule 4). The
generic path is not word-identical to the exact local-frame reference. Declared
error budget: at most 1 tick in L and 1 tick in Theta per output word. MEASURED
(``scripts/bench_factored_add_accuracy.py``, >=200k pairs per case): L words off by
one tick on 0.004-0.019 % of outputs, Theta words on 0.034-0.053 %, never more than
one tick, near-cancellation no worse than generic. Exact classes are bit-exact.
Nothing in the library selects this adder by default; the FB3-Hybrid
(:func:`spf.tensor.v02.spf_pair_addsub_hybrid`) remains the conforming path.
Speed (MEASURED, Beast x86_64 gcc 13 -O3, mrun job-2f3fd3b9dfd4): the C version is
1.53x faster than C FB3-Hybrid per pair and 1.48-1.53x per radix-2 FFT
(``benchmarks/factored_add/RESULTS-beast.md``). ROM: 173,144 int32 entries (676 KiB).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import v02
from .factored_add_rom import (
    FACTORED_FN,
    FACTORED_GI,
    FACTORED_GMAX,
    FACTORED_GN,
    FACTORED_GV,
    FACTORED_LC,
    FACTORED_LP,
    FACTORED_LT,
    FACTORED_N_NODE,
    FACTORED_ROM_VERSION,
    FACTORED_TH,
)
from .v02 import (
    SPF32_PHASE_BITS,
    SPF32_PHASE_HALF,
    SPF32_PHASE_MASK,
    SPF32_PHASE_QUARTER,
    SPF32_WORD_NAN,
    PerformanceCounters,
    _coerce_word,
    _pack_scalar,
    _pair_local,
    _word_parts,
    circular_diff_u16,
    is_nan,
    is_zero,
    spf_neg,
)

# --- table geometry ---------------------------------------------------------
_GV = FACTORED_GV
_GI = FACTORED_GI
_D = _GV - _GI  # guard units per coupling node step
_STEP = 1 << _D
_HALF_STEP = _STEP >> 1
_NODE_MAX = FACTORED_N_NODE - 2  # last interpolation cell start index
_GUARD = 1 << _GV
_GUARD_HALF = 1 << (_GV - 1)
_G_HALF_TICKS = 16384  # pi/2 in phase ticks
_G_HALF_GUARD = _G_HALF_TICKS << _GV

# numpy views of the ROM tables (int64 to keep interpolation products exact)
_LP = np.asarray(FACTORED_LP, dtype=np.int64)
_TH = np.asarray(FACTORED_TH, dtype=np.int64)
_LC = np.asarray(FACTORED_LC, dtype=np.int64)
_LT = np.asarray(FACTORED_LT, dtype=np.int64)
_FN = np.asarray(FACTORED_FN, dtype=np.int64)
_GN = np.asarray(FACTORED_GN, dtype=np.int64)

_EXACT_DTHETA = frozenset({0, -SPF32_PHASE_HALF, SPF32_PHASE_QUARTER, -SPF32_PHASE_QUARTER})

FACTORED_ADD_VERSION = FACTORED_ROM_VERSION

__all__ = [
    "FACTORED_ADD_VERSION",
    "spf_pair_addsub_factored",
    "spf_pair_addsub_factored_planes",
]


# --- scalar integer kernel (mirrors the C implementation bit-for-bit) -------
def _interp(table: Any, magnitude: int) -> int:
    """Rounded integer linear interpolation of a coupling node table.

    ``magnitude`` is ``|eta|`` in guard units; the return is in guard units.
    """

    index = magnitude >> _D
    if index >= _NODE_MAX:
        return int(table[_NODE_MAX + 1])  # clamp to the far node
    frac = magnitude - (index << _D)  # in [0, _STEP)
    lo = int(table[index])
    hi = int(table[index + 1])
    return lo + (((hi - lo) * frac + _HALF_STEP) >> _D)


def _f_eval(eta: int) -> int:
    if eta <= 0:
        return _interp(_FN, -eta)
    return eta + _interp(_FN, eta)  # F(eta) = eta + F(-eta)


def _g_eval(eta: int) -> int:
    if eta <= 0:
        return _interp(_GN, -eta)
    return _G_HALF_GUARD - _interp(_GN, eta)  # G(eta) = 16384 - G(-eta)


def _rne_from_guard(value: int) -> int:
    """Round ``value`` (guard units) to whole ticks, RNE ties-to-even."""

    quotient = value >> _GV  # floor, correct for negatives (arithmetic shift)
    remainder = value - (quotient << _GV)  # in [0, 2^GV)
    if remainder > _GUARD_HALF or (remainder == _GUARD_HALF and (quotient & 1)):
        quotient += 1
    return quotient


def _factored_outputs(A_L: int, A_T: int, gap: int, dT: int) -> tuple[int, int]:
    """Return ``(plus_word, minus_local_word)`` for the generic factored path.

    ``A`` is the larger operand; ``minus_local`` is ``A - B`` (the caller negates
    it when ``t`` was the larger operand).  Assumes ``gap >= 1`` and ``dT`` is not
    an exact phase class.
    """

    a = dT if dT >= 0 else -dT
    s = 1 if dT > 0 else -1
    lp = int(_LP[gap])
    th = int(_TH[gap])
    lc = int(_LC[a])
    lt = int(_LT[a])
    eta_p = th + lt
    eta_m = th - lt
    dhalf = dT << (_GV - 1)  # dTheta/2 in guard units (exact: dT even after shift)
    base_l = A_L << _GV
    base_t = A_T << _GV

    plus_l = v02._clamp_finite_l(_rne_from_guard(base_l + lp + lc + _f_eval(eta_p)))
    plus_t = _rne_from_guard(base_t + dhalf - s * _g_eval(eta_p)) & SPF32_PHASE_MASK
    plus = _pack_scalar(plus_l, plus_t)

    ls = lc + lt
    minus_l = v02._clamp_finite_l(_rne_from_guard(base_l + lp + ls + _f_eval(eta_m)))
    minus_t = _rne_from_guard(base_t + dhalf + s * (_g_eval(eta_m) - _G_HALF_GUARD)) & SPF32_PHASE_MASK
    minus_local = _pack_scalar(minus_l, minus_t)
    return plus, minus_local


def spf_pair_addsub_factored(
    u: int | Any,
    t: int | Any,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
    integer_only: bool = False,
) -> tuple[int, int]:
    """Return ``(u + t, u - t)`` with the factored tan-tanh table adder.

    Same word contract as :func:`spf.tensor.v02.spf_pair_addsub_hybrid`.  The
    generic and near-cancellation cases use the integer tables (zero
    transcendentals); the exact phase classes reuse the v0.2 exact paths so they
    stay bit-identical to FB3-Hybrid.  ``precision`` only steers the exact-class
    delegation (the factored tables are fixed); it is accepted for API parity.

    ``integer_only=True`` selects an explicitly experimental all-ROM consumer:
    equal-magnitude and quadrature lanes avoid floating exact-path evaluation.
    It has its own scalar/vector word contract, not a normative FB7 parity or
    arbitrary-fold error certificate. The native C factored kernel retains the
    default profile; this candidate knob is currently a Python consumer.
    """

    if counters is not None:
        counters.complex_pair_adds += 1
    wu, wt = _coerce_word(u), _coerce_word(t)
    if is_nan(wu) or is_nan(wt):
        return SPF32_WORD_NAN, SPF32_WORD_NAN
    if is_zero(wu):
        return wt, int(spf_neg(wt))
    if is_zero(wt):
        return wu, wu
    lu, tu = _word_parts(wu)
    lt_, tt = _word_parts(wt)
    if lu >= lt_:
        A_L, A_T, B_L, B_T = lu, tu, lt_, tt
        wA = wu
        u_is_A = True
    else:
        A_L, A_T, B_L, B_T = lt_, tt, lu, tu
        wA = wt
        u_is_A = False
    gap = A_L - B_L
    dT = circular_diff_u16(B_T, A_T)

    if integer_only and dT not in (0, -SPF32_PHASE_HALF):
        if gap == 0:
            a = abs(dT)
            lp = 256 << _GV
            plus = _pack_scalar(
                v02._clamp_finite_l(_rne_from_guard((A_L << _GV) + lp + int(_LC[a]))),
                v02._rne_half_ticks(2 * A_T + dT) & SPF32_PHASE_MASK,
            )
            minus_local = _pack_scalar(
                v02._clamp_finite_l(_rne_from_guard((A_L << _GV) + lp + int(_LC[a]) + int(_LT[a]))),
                v02._rne_half_ticks(2 * A_T + dT + (-SPF32_PHASE_HALF if dT > 0 else SPF32_PHASE_HALF))
                & SPF32_PHASE_MASK,
            )
        elif gap > FACTORED_GMAX:
            plus = minus_local = wA
        else:
            plus, minus_local = _factored_outputs(A_L, A_T, gap, dT)
    elif gap == 0 or dT in _EXACT_DTHETA:
        # Exact phase classes -> v0.2 exact paths (Lambda+/- ROMs, half-angle).
        plus = _pair_local(A_L, A_T, B_L, B_T, 1, precision)
        minus_local = _pair_local(A_L, A_T, B_L, B_T, -1, precision)
    elif gap > FACTORED_GMAX:
        # Big-gap shortcut: both corrections are < 0.5 tick, so both outputs are A.
        plus = wA
        minus_local = wA
    else:
        plus, minus_local = _factored_outputs(A_L, A_T, gap, dT)

    minus = minus_local if u_is_A else int(spf_neg(minus_local))
    return plus, minus


pair_addsub_factored = spf_pair_addsub_factored


# --- vectorized plane kernel ------------------------------------------------
def _interp_vec(table: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    index = np.minimum(magnitude >> _D, _NODE_MAX)
    frac = magnitude - (index << _D)
    lo = table[index]
    hi = table[index + 1]
    return lo + (((hi - lo) * frac + _HALF_STEP) >> _D)


def _f_eval_vec(eta: np.ndarray) -> np.ndarray:
    neg = eta <= 0
    out = np.where(neg, _interp_vec(_FN, np.where(neg, -eta, 0)), 0)
    pos = ~neg
    out = np.where(pos, eta + _interp_vec(_FN, np.where(pos, eta, 0)), out)
    return out


def _g_eval_vec(eta: np.ndarray) -> np.ndarray:
    neg = eta <= 0
    out = np.where(neg, _interp_vec(_GN, np.where(neg, -eta, 0)), 0)
    pos = ~neg
    out = np.where(pos, _G_HALF_GUARD - _interp_vec(_GN, np.where(pos, eta, 0)), out)
    return out


def _rne_from_guard_vec(value: np.ndarray) -> np.ndarray:
    quotient = value >> _GV
    remainder = value - (quotient << _GV)
    bump = (remainder > _GUARD_HALF) | ((remainder == _GUARD_HALF) & ((quotient & 1) == 1))
    return quotient + bump.astype(np.int64)


def spf_pair_addsub_factored_planes(
    u: Any,
    t: Any,
    precision: Any = "balanced",
    *,
    integer_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized :func:`spf_pair_addsub_factored` over packed-word planes.

    Broadcasts ``u`` and ``t`` (uint32 word arrays), returns ``(plus, minus)``
    word planes.  The generic/near-cancellation elements use the vectorized
    tables; exact-class / big-gap / sentinel elements defer to the scalar kernel
    so the result is word-identical to elementwise
    :func:`spf_pair_addsub_factored`.
    """

    wu = np.asarray(v02._word_array(u, preserve_raw=False), dtype=np.uint32)
    wt = np.asarray(v02._word_array(t, preserve_raw=False), dtype=np.uint32)
    wu, wt = np.broadcast_arrays(wu, wt)
    shape = wu.shape
    wu = wu.reshape(-1)
    wt = wt.reshape(-1)
    n = wu.size

    lu = v02._core.unpack_L(wu).astype(np.int64)
    tu = v02._core.unpack_theta(wu).astype(np.int64)
    lt = v02._core.unpack_L(wt).astype(np.int64)
    tt = v02._core.unpack_theta(wt).astype(np.int64)

    nan = (lu == v02.SPF32_NAN_L) | (lt == v02.SPF32_NAN_L)
    zero = (lu == v02.SPF32_L_ZERO) | (lt == v02.SPF32_L_ZERO)

    u_is_A = lu >= lt
    A_L = np.where(u_is_A, lu, lt)
    A_T = np.where(u_is_A, tu, tt)
    B_L = np.where(u_is_A, lt, lu)
    B_T = np.where(u_is_A, tt, tu)
    gap = A_L - B_L
    dT = ((B_T - A_T + SPF32_PHASE_HALF) & SPF32_PHASE_MASK) - SPF32_PHASE_HALF
    a = np.abs(dT)

    exact = (
        nan
        | zero
        | (gap == 0)
        | (dT == 0)
        | (dT == -SPF32_PHASE_HALF)
        | (a == SPF32_PHASE_QUARTER)
        | (gap > FACTORED_GMAX)
    )
    generic = ~exact

    plus = np.empty(n, dtype=np.uint32)
    minus = np.empty(n, dtype=np.uint32)

    if integer_only:
        # Sentinel and big-gap lanes are exact and need no scalar callbacks.
        ids = np.flatnonzero(nan)
        plus[ids] = minus[ids] = SPF32_WORD_NAN
        exact[ids] = False
        ids = np.flatnonzero(zero & ~nan)
        plus[ids] = np.where(lu[ids] == v02.SPF32_L_ZERO, wt[ids], wu[ids])
        local = np.where(lu[ids] == v02.SPF32_L_ZERO, _neg_word_vec(wt[ids]), wu[ids])
        minus[ids] = np.where((lu[ids] == v02.SPF32_L_ZERO) & (lt[ids] == v02.SPF32_L_ZERO), v02.SPF32_WORD_ZERO, local)
        exact[ids] = False
        ids = np.flatnonzero((gap > FACTORED_GMAX) & ~nan & ~zero)
        dominant = np.where(u_is_A[ids], wu[ids], wt[ids])
        plus[ids] = dominant
        minus[ids] = np.where(u_is_A[ids], dominant, _neg_word_vec(dominant))
        exact[ids] = False
        # Reuse the canonical Lambda ROMs, vectorized. Do not scalar-loop over
        # coherent streams (the dominant retained-memory workload).
        collinear = ~nan & ~zero & ((dT == 0) | (dT == -SPF32_PHASE_HALF))
        ids = np.flatnonzero(collinear)
        if ids.size:
            gaps = gap[ids]
            gp = np.minimum(gaps, v02.LOGADD_G_ZERO - 1)
            lp = np.where(gaps < v02.LOGADD_G_ZERO, np.asarray(v02.LOGADD_PLUS, np.int64)[gp], 0)
            lm = np.where(gaps < v02.LOGADD_G_ZERO, np.asarray(v02.LOGADD_MINUS, np.int64)[gp], 0)
            sumword = _pack_words_vec(np.clip(A_L[ids] + lp, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE), A_T[ids])
            diffword = _pack_words_vec(np.clip(A_L[ids] + lm, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE), A_T[ids])
            diffword = np.where(gaps == 0, v02.SPF32_WORD_ZERO, diffword).astype(np.uint32)
            opposite = dT[ids] != 0
            plus[ids] = np.where(opposite, diffword, sumword)
            local_minus = np.where(opposite, sumword, diffword)
            minus[ids] = np.where(
                local_minus == v02.SPF32_WORD_ZERO,
                v02.SPF32_WORD_ZERO,
                np.where(u_is_A[ids], local_minus, _neg_word_vec(local_minus)),
            )
            exact[ids] = False
        ids = np.flatnonzero((gap == 0) & ~nan & ~zero & (dT != 0) & (dT != -SPF32_PHASE_HALF))
        if ids.size:
            base_l = A_L[ids] << _GV
            lc, lt_t = _LC[a[ids]], _LT[a[ids]]
            plus[ids] = _pack_words_vec(
                np.clip(
                    _rne_from_guard_vec(base_l + (256 << _GV) + lc), v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE
                ),
                _rne_from_guard_vec((2 * A_T[ids] + dT[ids]) << (_GV - 1)),
            )
            minus_local = _pack_words_vec(
                np.clip(
                    _rne_from_guard_vec(base_l + (256 << _GV) + lc + lt_t),
                    v02.SPF32_L_MIN_FINITE,
                    v02.SPF32_L_MAX_FINITE,
                ),
                _rne_from_guard_vec(
                    (2 * A_T[ids] + dT[ids] + np.where(dT[ids] > 0, -SPF32_PHASE_HALF, SPF32_PHASE_HALF)) << (_GV - 1)
                ),
            )
            minus[ids] = np.where(u_is_A[ids], minus_local, _neg_word_vec(minus_local))
            exact[ids] = False
        quadrature = (a == SPF32_PHASE_QUARTER) & (gap != 0) & (gap <= FACTORED_GMAX) & ~nan & ~zero
        generic |= quadrature
        exact[quadrature] = False

    if generic.any():
        gA_L = A_L[generic]
        gA_T = A_T[generic]
        gdT = dT[generic]
        ga = a[generic]
        gg = gap[generic]
        gu_is_A = u_is_A[generic]
        s = np.where(gdT > 0, 1, -1).astype(np.int64)
        lp = _LP[gg]
        th = _TH[gg]
        lc = _LC[ga]
        lt_t = _LT[ga]
        eta_p = th + lt_t
        eta_m = th - lt_t
        dhalf = gdT << (_GV - 1)
        base_l = gA_L << _GV
        base_t = gA_T << _GV
        plus_l = np.clip(
            _rne_from_guard_vec(base_l + lp + lc + _f_eval_vec(eta_p)), v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE
        )
        plus_t = _rne_from_guard_vec(base_t + dhalf - s * _g_eval_vec(eta_p)) & SPF32_PHASE_MASK
        ls = lc + lt_t
        minus_l = np.clip(
            _rne_from_guard_vec(base_l + lp + ls + _f_eval_vec(eta_m)), v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE
        )
        minus_t = _rne_from_guard_vec(base_t + dhalf + s * (_g_eval_vec(eta_m) - _G_HALF_GUARD)) & SPF32_PHASE_MASK
        plus_w = _pack_words_vec(plus_l, plus_t)
        minus_local_w = _pack_words_vec(minus_l, minus_t)
        # u - t = (A - B) when u is A, else -(A - B)
        minus_w = np.where(gu_is_A, minus_local_w, _neg_word_vec(minus_local_w))
        plus[generic] = plus_w
        minus[generic] = minus_w

    if exact.any():
        idx = np.nonzero(exact)[0]
        for i in idx:
            p, m = spf_pair_addsub_factored(int(wu[i]), int(wt[i]), precision, integer_only=integer_only)
            plus[i] = p
            minus[i] = m

    return plus.reshape(shape), minus.reshape(shape)


def _pack_words_vec(L: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pack finite (L, T) planes to words (no sentinel inputs on this path)."""

    return ((L.astype(np.int64) & 0xFFFF) << SPF32_PHASE_BITS | (T & SPF32_PHASE_MASK)).astype(np.uint32)


def _neg_word_vec(words: np.ndarray) -> np.ndarray:
    L = v02._core.unpack_L(words).astype(np.int64)
    T = (v02._core.unpack_theta(words).astype(np.int64) + SPF32_PHASE_HALF) & SPF32_PHASE_MASK
    return _pack_words_vec(L, T)
