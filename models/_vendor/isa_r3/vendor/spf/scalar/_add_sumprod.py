"""Sum-to-product SPF complex addition — alternative to the Sb() path.

Reference: `docs/research/spf-extended-research.md` §"The crown jewel: reverse
prosthaphaeresis". Starting from the identity

    r_a·e^{iA} + r_b·e^{iB}
        = e^{i(A+B)/2} · ((r_a + r_b)·cos((A−B)/2)
                         + i·(r_a − r_b)·sin((A−B)/2))

and factoring out the larger operand ``r_big = r_a`` (after swapping so
``r_a ≥ r_b``) with ``s = r_b/r_a = 2^{−ΔL}``:

    r_a·e^{iA} + r_b·e^{iB}
        = r_a · e^{i(A+B)/2} · ((1 + s)·cos(Δ/2) + i·(1 − s)·sin(Δ/2))

Letting ``compound = (1+s)·cos(Δ/2) + i·(1−s)·sin(Δ/2)``:

    L_out = L_a + log₂|compound|
    Θ_out = (Θ_a + Θ_b)/2 + arg(compound)        (mod 2π)

where ``(Θ_a + Θ_b)/2`` is the circular midpoint along the shortest arc,
computed as ``Θ_a − Δ/2`` to avoid the naive-midpoint wrap pitfall.

Mathematically equivalent to :func:`spf.scalar._ops.add`. Numerically it
swaps one transcendental pattern (log₂ over ``1 + 2s·cos θ + s²``, plus
``atan2``) for another (``hypot`` + ``atan2`` over the half-angle
compound). The half-angle form has better conditioning at |Δθ|≈π
(``sin(θ/2) ≈ 1`` and ``cos(θ/2) ≈ 0`` are both well-conditioned there,
whereas Sb() evaluates ``sin(θ) ≈ 0`` and ``cos(θ) ≈ −1``). Cycle-cost
comparison is the subject of the Task 3.4 bench.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar._constants import Format
from spf.scalar._ops import (
    _canon_l,
    _classify,
    _classify_v02,
    _clip_finite,
    _is_v02,
    _pack,
    _unpack,
    _word_nan,
    _word_zero,
)

__all__ = ["add_sumprod"]


def add_sumprod(
    a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format
) -> npt.NDArray[np.unsignedinteger]:
    """SPF complex addition via the sum-to-product identity.

    Parameters
    ----------
    a, b : array-like of ``fmt.word_dtype``
        Packed SPF words.
    fmt : Format
        SPF format (SPF-16 / SPF-32 / SPF-64).

    Returns
    -------
    ndarray of ``fmt.word_dtype``
        Packed ``a + b``, matching the semantics of :func:`spf.scalar._ops.add`
        up to round-off in the L/Θ requantization.

    Notes
    -----
    Sentinel propagation is identical to Sb-based ADD. For SPF-32 that is the
    AlgRef v0.2 §7.1 cascade (NaN, then ZERO as the additive identity; §2.2
    defines no INF class and §2.4 canonicalizes the reserved ``L == 32767``
    encoding into the finite range on the way in). The other widths keep their
    v0.1 ``NaN > inf > zero > finite`` cascade — see
    :data:`spf.scalar._ops._V02_FORMAT_NAMES`.

    The function deliberately accepts slightly different quantization round-off
    from Sb-based ADD — that divergence is the target of the accuracy-comparison
    bench, not a bug. The §7.1 exact-cancellation branch is NOT part of that
    budget: it is normative, and without it the factored compound evaluates
    ``cos(pi/2)`` as the fp64 noise floor and lands on a spurious finite L.
    """
    v02 = _is_v02(fmt)
    a_arr = np.asarray(a, dtype=fmt.word_dtype)
    b_arr = np.asarray(b, dtype=fmt.word_dtype)
    a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)

    la_raw, ta = _unpack(a_arr, fmt)
    lb_raw, tb = _unpack(b_arr, fmt)

    if v02:
        nan_a, zero_a = _classify_v02(la_raw, ta, fmt)
        nan_b, zero_b = _classify_v02(lb_raw, tb, fmt)
        inf_a = np.zeros_like(nan_a)
        inf_b = np.zeros_like(nan_b)
        # §2.3/§2.4: read the canonical L, so a reserved word behaves as
        # L_MAX_FINITE instead of selecting the v0.1 infinity branch.
        la = _canon_l(la_raw, fmt).astype(fmt.l_signed_dtype)
        lb = _canon_l(lb_raw, fmt).astype(fmt.l_signed_dtype)
    else:
        nan_a, inf_a, zero_a = _classify(la_raw, ta, fmt)
        nan_b, inf_b, zero_b = _classify(lb_raw, tb, fmt)
        la, lb = la_raw, lb_raw

    # Swap so the first operand has the larger L (matches _ops.add's convention
    # so sentinel handling and Δ sign choices line up).
    swap = la < lb
    la_big = np.where(swap, lb, la)
    ta_big = np.where(swap, tb, ta)
    lb_sm = np.where(swap, la, lb)
    tb_sm = np.where(swap, ta, tb)

    # ΔL = L_big − L_sm ≥ 0, log₂ units; s = 2^{−ΔL} ∈ (0, 1].
    dl = (la_big.astype(np.float64) - lb_sm.astype(np.float64)) / fmt.l_scale
    s = np.exp2(-dl)

    # Signed phase difference (A − B) in (−π, π], A = Θ_big, B = Θ_sm.
    dt_u = (
        (ta_big.astype(np.int64) - tb_sm.astype(np.int64)) & fmt.word_mask_theta
    ).astype(np.int64)
    half_circle = fmt.theta_scale >> 1
    dt_signed = np.where(dt_u > half_circle, dt_u - fmt.theta_scale, dt_u)
    a_minus_b = dt_signed.astype(np.float64) / fmt.theta_scale * (2.0 * np.pi)
    half_amb = a_minus_b * 0.5

    # Compound = (1 + s)·cos(Δ/2) + i·(1 − s)·sin(Δ/2).
    cos_half = np.cos(half_amb)
    sin_half = np.sin(half_amb)
    c_re = (1.0 + s) * cos_half
    c_im = (1.0 - s) * sin_half

    # Log-magnitude correction: log₂|compound|. At exact cancellation (s=1 and
    # |Δ|=π) both c_re and c_im are 0 → log₂(0) = −∞; routed below to the
    # L_CODE_MIN_FINITE fallback via np.isfinite.
    with np.errstate(divide="ignore", invalid="ignore"):
        compound_mag = np.hypot(c_re, c_im)
        log_compound = np.log2(compound_mag)

    # Phase correction: arg(compound). atan2(0, 0) = 0 by IEEE — fine for the
    # cancellation case because L_out is forced to the sentinel-adjacent floor
    # there anyway; the phase is conventional.
    arg_compound = np.arctan2(c_im, c_re)

    # Phase mean along the shortest arc: Θ_a − (A − B)/2 = (A + B)/2.
    # Doing it this way avoids the naive-midpoint wrap ambiguity when Θ_big and
    # Θ_sm straddle the 2π boundary (e.g. 0.1·2π and 0.9·2π).
    theta_big_fp = ta_big.astype(np.float64) / fmt.theta_scale * (2.0 * np.pi)
    phase_mean = theta_big_fp - half_amb
    theta_out_fp = phase_mean + arg_compound

    # Back into code units.
    l_out_code_fp = la_big.astype(np.float64) + log_compound * fmt.l_scale
    with np.errstate(invalid="ignore"):
        l_safe = np.where(
            np.isfinite(l_out_code_fp), l_out_code_fp, float(fmt.l_code_min_finite)
        )
        l_out_i64 = np.rint(l_safe).astype(np.int64)
    l_out_clipped = _clip_finite(l_out_i64, fmt)

    t_out_u = np.mod(
        np.rint(theta_out_fp / (2.0 * np.pi) * fmt.theta_scale).astype(np.int64),
        fmt.theta_scale,
    ).astype(fmt.theta_dtype)

    finite_word = _pack(l_out_clipped, t_out_u, fmt)

    # §7.1 exact cancellation: equal magnitudes a half turn apart sum to exactly
    # zero. Normative, not an accuracy allowance — see the docstring. The local
    # wrap test above is ``dt_u > half_circle``, so the half turn comes out as
    # ``+half_circle``; accept the §2.5 spelling ``-half_circle`` too.
    exact_cancel = (la_big == lb_sm) & (np.abs(dt_signed) == half_circle)

    # Sentinel overlays — identical to _ops.add.
    both_inf = inf_a & inf_b
    same_theta = ta == tb
    is_nan_out = nan_a | nan_b | (both_inf & ~same_theta)
    is_inf_out = (inf_a | inf_b) & ~is_nan_out

    inf_theta = np.where(inf_a, ta, tb)
    inf_l = np.full_like(la_big, fmt.l_code_inf, dtype=fmt.l_signed_dtype)
    inf_word = _pack(inf_l, inf_theta, fmt)

    only_a_zero = zero_a & ~zero_b
    only_b_zero = zero_b & ~zero_a
    both_zero = zero_a & zero_b

    result = finite_word
    result = np.where(exact_cancel, _word_zero(fmt), result)
    result = np.where(only_a_zero, b_arr, result)
    result = np.where(only_b_zero, a_arr, result)
    result = np.where(both_zero, _word_zero(fmt), result)
    result = np.where(is_inf_out, inf_word, result)
    result = np.where(is_nan_out, _word_nan(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)
