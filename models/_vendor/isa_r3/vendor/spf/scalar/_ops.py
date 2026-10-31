"""Format-parameterized SPF scalar operators.

Spec: docs/spf-formats-and-test-plans.md §2 (SPF-32 as the reference) plus §3
and §4 for SPF-16/64 widths. Every operation here takes a :class:`Format`
argument that pins dtypes, bit widths, and sentinel codes; the per-format
modules (:mod:`spf.scalar.spf16`, :mod:`.spf32`, :mod:`.spf64`) bind their
``FORMAT`` constant and expose the same public surface.

The three formats share the same L/Θ algebra — only the field widths and
sentinel codes change. MUL/DIV/CONJ/TWIDDLE/ABS2 are exact integer adds on the
L and Θ codes. ADD uses the exact FP64 Sb() formula (Arnold & Collange 2011)
— the Sb() LUT path was removed in Phase 5 of the SPFv2 promotion.

Integer intermediates use ``int64`` for SPF-16/32/64: it comfortably holds any
sum/diff of those formats' L or Θ codes (the widest is SPF-64 L at 32 bits, so
doubling stays under int64's 63-bit range). **SPF-128** (64-bit L, 64-bit Θ,
128-bit word) has no numpy word dtype and overflows int64 — its words are
arbitrary-precision Python ints in ``object`` arrays, and the exact integer ops
route their accumulators through Python int (the ``fmt.total_bits > 64``
branches in :func:`_acc_dtype`, :func:`_pack`, etc.). The SPF-16/32/64 numpy fast
path is unchanged.

Sentinel profile per format (v0.2 vs v0.1)
-----------------------------------------

**SPF-32 is v0.2.** ``SPF_Algorithm_Reference_v0.2.md`` §2.2 lists exactly two
sentinels — ZERO (``L = -32768``, word ``0x80000000``) and the single canonical
NaN word (``L = 32767`` **and** ``Θ = 65535``, word ``0x7FFFFFFF``). There is no
INF class. ``L == 32767`` with any other Θ is a *reserved* encoding that §2.4
canonicalizes to ``L_MAX_FINITE`` before an operation reads its fields (§2.3,
amended 2026-09-18), and §3.1 *saturates* an infinite (or FP64-overflowing)
magnitude to ``L_MAX_FINITE`` instead of minting a directional-infinity word.
This module is the numpy fallback behind :mod:`spf.scalar.spf32`; keeping it on
v0.1 made those entry points behave differently with and without ``libspf32``
(the C core moved to v0.2 in commit a51398f1). The reference implementations the
SPF-32 branches here are matched against, bit-for-bit, are
``native/src/scalar/spf32/core/spf32_ops.c`` (``spf32_encode`` / ``spf32_decode``
/ ``spf32_mul`` / ``spf32_div`` / ``spf32_conj`` / ``spf32_neg`` / ``spf32_abs2``)
and :mod:`spf.tensor.v02`.

**SPF-16 / SPF-64 / SPF-128 stay v0.1** — they keep the directional-infinity
class (``L == l_code_inf`` with ``Θ != theta_nan_payload`` decodes as
``inf·e^(iθ)``, ``encode(inf)`` mints it, and mul/div/abs2/add propagate it).
Reason: AlgRef v0.2 is written for SPF-32 only (every constant in §2.2 is
``SPF32_*``-prefixed) and the repo spec that does cover the other widths
(``docs/spf-formats-and-test-plans.md`` §3 SPF-16, §4 SPF-64, §4A SPF-128) still
*documents* the infinity sentinel — §4A.2's sentinel table lists "+∞ at Θ=0"
explicitly and §3/§4 inherit "same convention as §2". ``AGENTS.md``: code vs doc
disagree → doc wins until the doc changes. Widening the v0.2 profile to those
formats is a doc change first, a code change second; see :data:`_V02_FORMAT_NAMES`.

``add``'s sentinel cascade is v0.2 for SPF-32 too (§7.1 names only ``is_zero``
and ``is_nan``), but its *finite* path is the exact FP64 Sb() formula while the C
``spf32_add`` delegates to poly7/FB7, so the two agree on sentinels and differ by
up to one Q7.8 rounding step on finite operands — a pre-existing, documented
algorithmic difference (``tests/native_bindings/test_T1_c_crossvalidate.py``
§"T1.6 / T1.7"), not a sentinel divergence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from spf.scalar._constants import Format
from spf.scalar._fast_log import POLY_MAX_ABS_ERROR, fast_log2


def _sb_exact(
    dl: npt.NDArray[np.float64], dtheta: npt.NDArray[np.float64]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Direct FP64 evaluation of Sb(ΔL, ΔΘ) — local copy after _sb.py removal.

    Returns (delta_l_log2, delta_theta_rad). At the cancellation singularity
    (ΔL=0, ΔΘ=π) delta_l → −∞ and delta_theta is 0 by convention.
    """
    s = np.exp2(-dl)
    cos_t = np.cos(dtheta)
    sin_t = np.sin(dtheta)
    r2 = 1.0 + 2.0 * s * cos_t + s * s
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_l = 0.5 * np.log2(r2)
    delta_t = np.arctan2(s * sin_t, 1.0 + s * cos_t)
    return delta_l, delta_t


__all__ = [
    "abs2",
    "add",
    "butterfly",
    "conj",
    "decode",
    "div",
    "dot",
    "dot_nokahan",
    "encode",
    "mul",
    "neg",
    "sub",
    "twiddle",
]


#: Formats that follow the AlgRef v0.2 sentinel profile (ZERO + one NaN word, no
#: INF class, reserved ``L == l_code_inf`` canonicalized per §2.4). See the
#: module docstring for why only SPF-32 is listed.
_V02_FORMAT_NAMES: frozenset[str] = frozenset(
    {
        "SPF-32",
        "SPF-32-M18-P14",
        "SPF-32-M19-P13",
        "SPF-32-M20-P12",
        "SPF-32-Q3.12-P16",
        "SPF-32-Q3.8-P20",
    }
)


def _is_v02(fmt: Format) -> bool:
    """True when ``fmt`` follows the AlgRef v0.2 sentinel profile (§2.2/§2.4)."""
    return fmt.name in _V02_FORMAT_NAMES


def _is_wide(fmt: Format) -> bool:
    """True when the packed word exceeds 64 bits (SPF-128 → Python-int/object)."""
    return fmt.total_bits > 64


def _acc_dtype(fmt: Format) -> Any:
    """Integer-accumulator dtype for the exact ops: ``object`` (arbitrary-precision
    Python int) for wide formats, else ``int64``. SPF-128's L codes (±2⁶³) and Θ
    ring (2⁶⁴) outrun int64, so its accumulation runs in Python int."""
    return object if _is_wide(fmt) else np.int64


def _wide_scalar(value: int) -> npt.NDArray[Any]:
    """Wrap a wide (>64-bit) Python int as a 0-d ``object`` array.

    A bare 128-bit Python int passed to ``np.where(cond, value, object_array)``
    trips numpy's C-long coercion; a 0-d ``object`` array is broadcast cleanly and
    keeps the result ``object``."""
    a = np.empty((), dtype=object)
    a[()] = int(value)
    return a


def _word_zero(fmt: Format) -> Any:
    """Return the zero-sentinel word — a 0-d ``object`` array for wide words
    (np.where-safe), else a scalar of ``fmt.word_dtype``."""
    if fmt.word_dtype is object:
        return _wide_scalar(fmt.word_zero)
    return fmt.word_dtype(fmt.word_zero)


def _word_nan(fmt: Format) -> Any:
    """Return the NaN-sentinel word (0-d ``object`` array for wide words)."""
    if fmt.word_dtype is object:
        return _wide_scalar(fmt.word_nan)
    return fmt.word_dtype(fmt.word_nan)


def _l_bits_mask(fmt: Format) -> int:
    """Mask covering the L field (all ``l_bits`` low bits set)."""
    return (1 << fmt.l_bits) - 1


def _rint_to_int_array(fp: npt.ArrayLike, fmt: Format) -> npt.NDArray[Any]:
    """Round a float array to integer codes. For wide formats return Python ints
    (``object``) so a later clip caps them below int64's range without the
    overflow that ``float→int64`` would hit at codes near 2⁶³."""
    r = np.rint(np.asarray(fp, dtype=np.float64))
    if _is_wide(fmt):
        flat = [int(v) for v in np.atleast_1d(r).ravel().tolist()]
        return np.array(flat, dtype=object).reshape(r.shape)
    return r.astype(np.int64)


def _theta_code_from_float(fp: npt.ArrayLike, fmt: Format) -> npt.NDArray[Any]:
    """Round a float phase value into a Θ code in ``[0, theta_scale)``, cast to
    ``fmt.theta_dtype``. Wide formats reduce in Python int (the ``2⁶⁴`` ring
    overflows int64)."""
    r = np.rint(np.asarray(fp, dtype=np.float64))
    if _is_wide(fmt):
        flat = [int(v) % fmt.theta_scale for v in np.atleast_1d(r).ravel().tolist()]
        return np.array(flat, dtype=object).reshape(r.shape).astype(fmt.theta_dtype)
    return np.mod(r.astype(np.int64), fmt.theta_scale).astype(fmt.theta_dtype)


# ── Encode / decode ──────────────────────────────────────────────────────────


def encode(z: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.generic]:
    """Encode complex values into packed SPF words of the given format.

    Parameters
    ----------
    z : array-like of complex
        Input values. Coerced to complex128 internally.
    fmt : Format
        Target SPF format.

    Returns
    -------
    ndarray of ``fmt.word_dtype``
        Packed SPF words, same shape as ``z``.

    Notes
    -----
    SPF-32 follows AlgRef v0.2 §3.1: ``L = RNE(256·log2|z|)`` clamped to
    ``[L_MIN_FINITE, L_MAX_FINITE]``, and an infinite (or FP64-overflowing)
    magnitude *saturates* to ``L_MAX_FINITE`` — §2.2 has no infinity word. The
    other widths keep the v0.1 directional-infinity sentinel; see the module
    docstring.
    """
    z_arr = np.asarray(z, dtype=np.complex128)

    is_nan = np.isnan(z_arr.real) | np.isnan(z_arr.imag)
    is_inf = np.isinf(z_arr.real) | np.isinf(z_arr.imag)
    is_zero = z_arr == 0

    l_scale = fmt.l_scale
    theta_scale = fmt.theta_scale

    # fast_log2 is only defined on positive normals; route specials through the
    # sentinel overlays below. The errstate suppresses the associated warnings.
    # For formats whose L ULP is finer than fast_log2's polynomial floor
    # (~2⁻¹¹), fall back to FP64 ``np.log2``. SPF-64 needs this; SPF-16 uses
    # fast_log2 so the C-port encode path is exercised by its tests.
    #
    # SPF-32 must use np.log2: AlgRef v0.2 §3.1 pins the L code to *exact* RNE
    # on ``256·log2|z|`` (so the §3.3 bound ε_r = 2^(1/512) − 1 holds), which the
    # degree-3 minimax fast_log2 misses on 0.556 % of magnitudes. The C
    # reference ``spf32_encode`` uses libm ``log2`` + ``rint`` for the same
    # reason (2026-09-18 conformance audit, deviation #10); the fast poly now
    # lives behind ``spf32_encode_fast`` with a declared error budget.
    use_np_log2 = _is_v02(fmt) or fmt.l_ulp < POLY_MAX_ABS_ERROR
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.abs(z_arr)
        # Θ comes from atan2 in every branch, infinities included: C99 atan2 is
        # fully specified on infinities, so the direction survives saturation.
        theta = np.angle(z_arr)  # [-π, π]
        r_safe = np.where(np.isfinite(r) & (r > 0), r, 1.0)
        l_fp = np.log2(r_safe) if use_np_log2 else fast_log2(r_safe)
        l_code = _clip_finite(
            _rint_to_int_array(np.where(np.isfinite(l_fp), l_fp, 0.0) * l_scale, fmt),
            fmt,
        )
        theta_safe = np.where(np.isnan(theta), 0.0, theta)
        theta_pos = np.where(theta_safe < 0, theta_safe + 2 * np.pi, theta_safe)
        # ``θ·scale/2π``, not ``θ/2π·scale``: matches the C reference and
        # ``spf.tensor.core.encode_complex`` expression order bit-for-bit.
        theta_code = _theta_code_from_float(theta_pos * theta_scale / (2 * np.pi), fmt)

    if _is_v02(fmt):
        # §2.2 / §3.1: no infinity word. An infinite input — or a finite input
        # whose magnitude overflows FP64 hypot, e.g. complex(1e308, 1e308) —
        # saturates the L code to L_MAX_FINITE and keeps its atan2 direction.
        overflow = (is_inf | ~np.isfinite(r)) & ~is_nan
        l_code = np.where(overflow, fmt.l_code_max_finite, l_code).astype(fmt.l_signed_dtype)
        word = _pack(l_code, theta_code, fmt)
        word = np.where(is_zero, _word_zero(fmt), word)
        word = np.where(is_nan, _word_nan(fmt), word)
        return word.astype(fmt.word_dtype, copy=False)

    word = _pack(l_code, theta_code, fmt)

    word = np.where(is_zero, _word_zero(fmt), word)

    # v0.1 only: directional inf preserves the phase code so decode can
    # reconstruct direction.
    inf_l = np.full(l_code.shape, fmt.l_code_inf, dtype=fmt.l_signed_dtype)
    inf_word = _pack(inf_l, theta_code, fmt)
    word = np.where(is_inf & ~is_nan, inf_word, word)

    word = np.where(is_nan, _word_nan(fmt), word)

    return word.astype(fmt.word_dtype, copy=False)


def decode(w: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.complex128]:
    """Decode packed SPF words back to complex128.

    Parameters
    ----------
    w : array-like of ``fmt.word_dtype``
        Packed SPF words.
    fmt : Format
        Source SPF format.

    Returns
    -------
    ndarray of complex128
        Decoded values, same shape as ``w``. Zero sentinel → 0+0j; canonical
        NaN → nan+nanj. For SPF-32 (AlgRef v0.2 §3.2 + §2.4) a *reserved* word
        (``L == 32767``, ``Θ != 65535``) decodes as the finite ``L_MAX_FINITE``
        magnitude — there is no infinity output. The other widths keep the v0.1
        directional-inf sentinel (``inf·e^(iθ)``); see the module docstring.
    """
    w_arr = np.asarray(w, dtype=fmt.word_dtype)

    # np.asarray re-wraps the shift/mask: a 0-d ``object`` word unwraps to a bare
    # Python int under ``>>``/``&``, which has no ``.astype``.
    l_signed, theta_u = _unpack(w_arr, fmt)

    v02 = _is_v02(fmt)
    is_zero = l_signed == fmt.l_code_zero
    is_inf_or_nan = l_signed == fmt.l_code_inf
    is_nan = is_inf_or_nan & (theta_u == fmt.theta_nan_payload)
    is_inf = is_inf_or_nan & ~is_nan

    if v02:
        # §2.4: canonicalize before the fields are read, so a reserved word is a
        # finite L_MAX_FINITE operand rather than a second (directional-inf)
        # format. Mirrors ``spf32_canon_l`` in the C decode.
        l_fp = _canon_l(l_signed, fmt).astype(np.float64) / fmt.l_scale
    else:
        l_fp = l_signed.astype(np.float64) / fmt.l_scale
    theta_fp = theta_u.astype(np.float64) / fmt.theta_scale * (2 * np.pi)

    # exp2 avoids log2/exp round-trip loss. errstate suppresses the inf*0→NaN
    # warning from the eager np.where of the directional-inf branch, plus the
    # overflow from SPF-64's finite L range (max L ≈ 32767, exp2 overflows FP64
    # at 1024 — sentinel paths mask it out via np.where).
    with np.errstate(invalid="ignore", over="ignore"):
        r = np.exp2(l_fp)
        z = r * np.exp(1j * theta_fp)

    z = np.where(is_zero, 0 + 0j, z)
    if not v02:
        with np.errstate(invalid="ignore", over="ignore"):
            inf_z = np.inf * np.exp(1j * theta_fp)
        z = np.where(is_inf, inf_z, z)
    z = np.where(is_nan, complex(np.nan, np.nan), z)

    return z.astype(np.complex128, copy=False)


# ── Internal bit helpers ─────────────────────────────────────────────────────


def _unpack(w: npt.ArrayLike, fmt: Format) -> tuple[npt.NDArray[np.signedinteger], npt.NDArray[np.unsignedinteger]]:
    """Split packed SPF words into (signed L code, unsigned Θ code)."""
    w_arr = np.asarray(w, dtype=fmt.word_dtype)
    if fmt.l_bits != np.dtype(fmt.l_signed_dtype).itemsize * 8:
        from spf.tensor.core import unpack_L, unpack_theta

        return unpack_L(w_arr, fmt), unpack_theta(w_arr, fmt)
    l_u = np.asarray(w_arr >> fmt.theta_bits, dtype=fmt.l_unsigned_dtype)
    l_signed = l_u.view(fmt.l_signed_dtype).copy()
    theta_u = np.asarray(w_arr & fmt.word_mask_theta, dtype=fmt.theta_dtype)
    return l_signed, theta_u


def _pack(
    l_signed: npt.NDArray[np.signedinteger],
    theta_u: npt.NDArray[np.unsignedinteger],
    fmt: Format,
) -> npt.NDArray[np.unsignedinteger]:
    """Pack signed L and unsigned Θ back into a ``fmt.word_dtype`` word."""
    l_bits_mask = _l_bits_mask(fmt)
    if fmt.word_dtype is object:
        # Wide (SPF-128): assemble the 128-bit word in Python int. ``& l_bits_mask``
        # takes the two's-complement low ``l_bits`` of a (possibly negative) L code.
        l_wide = l_signed.astype(object) & l_bits_mask
        t_wide = theta_u.astype(object)
        return (l_wide << fmt.theta_bits) | t_wide  # type: ignore[no-any-return]
    l_wide = (l_signed.astype(np.int64) & l_bits_mask).astype(fmt.word_dtype)
    t_wide = theta_u.astype(fmt.word_dtype)
    return (l_wide << fmt.theta_bits) | t_wide  # type: ignore[no-any-return]


def _classify(
    l_signed: npt.NDArray[np.signedinteger],
    theta_u: npt.NDArray[np.unsignedinteger],
    fmt: Format,
) -> tuple[
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
    npt.NDArray[np.bool_],
]:
    """Return (is_nan, is_inf, is_zero) masks over input words."""
    is_inf_or_nan = l_signed == fmt.l_code_inf
    is_nan = is_inf_or_nan & (theta_u == fmt.theta_nan_payload)
    is_inf = is_inf_or_nan & ~is_nan
    is_zero = l_signed == fmt.l_code_zero
    return is_nan, is_inf, is_zero


def _classify_v02(
    l_signed: npt.NDArray[np.signedinteger],
    theta_u: npt.NDArray[np.unsignedinteger],
    fmt: Format,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Return (is_nan, is_zero) masks — AlgRef v0.2 §2.2 / §2.3.1.

    Exactly two sentinels: ZERO (``L == l_code_zero``) and the single canonical
    NaN word (``L == l_code_inf`` **and** ``Θ == theta_nan_payload``). There is
    no INF class; ``L == l_code_inf`` with any other Θ is a *reserved* encoding
    that :func:`_canon_l` folds into the finite range (§2.4). Mirrors
    ``spf32_classify_v02`` in ``native/src/scalar/spf32/core/spf32_internal.h``.
    """
    is_zero = l_signed == fmt.l_code_zero
    is_nan = (l_signed == fmt.l_code_inf) & (theta_u == fmt.theta_nan_payload)
    return is_nan, is_zero


def _canon_l(l_signed: npt.NDArray[np.signedinteger], fmt: Format) -> npt.NDArray[Any]:
    """AlgRef v0.2 §2.4 restricted to the L plane, in the accumulator dtype.

    A reserved ``L == l_code_inf`` enters an operation as ``l_code_max_finite``.
    Mirrors ``spf32_canon_l``. The ZERO code is left alone — every caller
    overlays the ZERO sentinel word on top of the arithmetic result.
    """
    return np.where(l_signed == fmt.l_code_inf, fmt.l_code_max_finite, l_signed).astype(_acc_dtype(fmt))


def _canon_words(w: npt.ArrayLike, fmt: Format) -> npt.NDArray[Any]:
    """AlgRef v0.2 §2.4 applied to packed words.

    §2.3 (amended 2026-09-18): "Operands of every §5–§7 operation are
    canonicalized (§2.4) before their fields are read … a reserved word with
    ``L == 32767`` and ``T != 65535`` enters an operation as ``L = 32766``."
    Used by :func:`add`, whose zero-passthrough returns an operand word whole.
    """
    w_arr = np.asarray(w, dtype=fmt.word_dtype)
    l_signed, theta_u = _unpack(w_arr, fmt)
    reserved = (l_signed == fmt.l_code_inf) & (theta_u != fmt.theta_nan_payload)
    if not np.any(reserved):
        return w_arr
    fixed = _pack(
        np.full(l_signed.shape, fmt.l_code_max_finite, dtype=fmt.l_signed_dtype),
        theta_u,
        fmt,
    )
    return np.where(reserved, fixed, w_arr).astype(fmt.word_dtype, copy=False)


def _clip_finite(l: npt.NDArray[np.signedinteger], fmt: Format) -> npt.NDArray[np.signedinteger]:
    """Clip L to the finite range, guaranteeing sentinel codes never appear."""
    lo, hi = fmt.l_code_min_finite, fmt.l_code_max_finite
    if _is_wide(fmt):
        # np.clip on ``object`` arrays is unreliable; clamp in Python int. The
        # clamped codes always fit ``l_signed_dtype`` (int64) by construction.
        arr = np.asarray(l)
        flat = [lo if v < lo else hi if v > hi else int(v) for v in np.atleast_1d(arr).ravel().tolist()]
        return np.array(flat, dtype=fmt.l_signed_dtype).reshape(arr.shape)
    clipped = np.clip(l, lo, hi).astype(fmt.l_signed_dtype)
    return clipped  # type: ignore[no-any-return]


# ── O(1) integer-domain operations (spec §2.4) ──────────────────────────────


def mul(a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """SPF multiply: ``L_out = L_a + L_b`` (saturated), ``Θ_out = Θ_a + Θ_b``.

    SPF-32 follows AlgRef v0.2 §5.1 — NaN is tested before ZERO, and there is no
    INF class, so a reserved operand canonicalizes (§2.4) into a finite
    ``L_MAX_FINITE`` factor. The other widths keep the v0.1 cascade.
    """
    la, ta = _unpack(a, fmt)
    lb, tb = _unpack(b, fmt)
    la, lb, ta, tb = np.broadcast_arrays(la, lb, ta, tb)

    acc = _acc_dtype(fmt)
    t_sum = ((ta.astype(acc) + tb.astype(acc)) & fmt.word_mask_theta).astype(fmt.theta_dtype)

    if _is_v02(fmt):
        # §5.1: `if is_nan(a) or is_nan(b): NAN` then `if is_zero(...): ZERO`.
        nan_a, zero_a = _classify_v02(la, ta, fmt)
        nan_b, zero_b = _classify_v02(lb, tb, fmt)
        l_sum = _clip_finite(_canon_l(la, fmt) + _canon_l(lb, fmt), fmt)
        result = _pack(l_sum, t_sum, fmt)
        is_nan_out = nan_a | nan_b
        is_zero_out = (zero_a | zero_b) & ~is_nan_out
        result = np.where(is_zero_out, _word_zero(fmt), result)
        result = np.where(is_nan_out, _word_nan(fmt), result)
        return result.astype(fmt.word_dtype, copy=False)

    l_sum = _clip_finite(la.astype(acc) + lb.astype(acc), fmt)
    finite_word = _pack(l_sum, t_sum, fmt)

    nan_a, inf_a, zero_a = _classify(la, ta, fmt)
    nan_b, inf_b, zero_b = _classify(lb, tb, fmt)
    is_nan_out = nan_a | nan_b | (inf_a & zero_b) | (zero_a & inf_b)
    is_zero_out = (zero_a | zero_b) & ~is_nan_out
    is_inf_out = (inf_a | inf_b) & ~is_nan_out & ~is_zero_out

    inf_word = _pack(np.full_like(l_sum, fmt.l_code_inf, dtype=fmt.l_signed_dtype), t_sum, fmt)

    result = np.where(is_inf_out, inf_word, finite_word)
    result = np.where(is_zero_out, _word_zero(fmt), result)
    result = np.where(is_nan_out, _word_nan(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)


def div(a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """SPF divide: ``L_out = L_a − L_b``, ``Θ_out = Θ_a − Θ_b``.

    SPF-32 follows AlgRef v0.2 §5.2, whose test order is load-bearing: NaN, then
    ``is_zero(a) → ZERO``, then ``is_zero(b) → NaN``. So ``ZERO / ZERO = ZERO``
    and ``finite / ZERO = NaN`` — not a directional infinity (§2.2 has none). The
    other widths keep the v0.1 cascade.
    """
    la, ta = _unpack(a, fmt)
    lb, tb = _unpack(b, fmt)
    la, lb, ta, tb = np.broadcast_arrays(la, lb, ta, tb)

    acc = _acc_dtype(fmt)
    t_diff = ((ta.astype(acc) - tb.astype(acc)) & fmt.word_mask_theta).astype(fmt.theta_dtype)

    if _is_v02(fmt):
        nan_a, zero_a = _classify_v02(la, ta, fmt)
        nan_b, zero_b = _classify_v02(lb, tb, fmt)
        l_diff = _clip_finite(_canon_l(la, fmt) - _canon_l(lb, fmt), fmt)
        result = _pack(l_diff, t_diff, fmt)
        nan_first = nan_a | nan_b
        is_zero_out = zero_a & ~nan_first
        is_nan_out = nan_first | (zero_b & ~zero_a & ~nan_first)
        result = np.where(is_zero_out, _word_zero(fmt), result)
        result = np.where(is_nan_out, _word_nan(fmt), result)
        return result.astype(fmt.word_dtype, copy=False)

    l_diff = _clip_finite(la.astype(acc) - lb.astype(acc), fmt)
    finite_word = _pack(l_diff, t_diff, fmt)

    nan_a, inf_a, zero_a = _classify(la, ta, fmt)
    nan_b, inf_b, zero_b = _classify(lb, tb, fmt)
    is_nan_out = nan_a | nan_b | (zero_a & zero_b) | (inf_a & inf_b)
    is_inf_out = ((inf_a & ~inf_b) | (~zero_a & zero_b)) & ~is_nan_out
    is_zero_out = ((zero_a & ~zero_b) | (~inf_a & inf_b)) & ~is_nan_out & ~is_inf_out

    inf_word = _pack(np.full_like(l_diff, fmt.l_code_inf, dtype=fmt.l_signed_dtype), t_diff, fmt)

    result = np.where(is_inf_out, inf_word, finite_word)
    result = np.where(is_zero_out, _word_zero(fmt), result)
    result = np.where(is_nan_out, _word_nan(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)


def conj(a: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """Complex conjugate: L unchanged, Θ ↦ (−Θ) mod ``2**theta_bits``.

    SPF-32 follows AlgRef v0.2 §5.3 (``if is_zero(a) or is_nan(a): return a``,
    else the reserved L canonicalizes per §2.4).
    """
    la, ta = _unpack(a, fmt)
    acc = _acc_dtype(fmt)
    t_neg = ((-ta.astype(acc)) & fmt.word_mask_theta).astype(fmt.theta_dtype)

    if _is_v02(fmt):
        nan_a, zero_a = _classify_v02(la, ta, fmt)
        result = _pack(_clip_finite(_canon_l(la, fmt), fmt), t_neg, fmt)
        result = np.where(zero_a, _word_zero(fmt), result)
        result = np.where(nan_a, _word_nan(fmt), result)
        return result.astype(fmt.word_dtype, copy=False)

    finite_word = _pack(la, t_neg, fmt)

    nan_a, _inf_a, zero_a = _classify(la, ta, fmt)
    result = np.where(nan_a, _word_nan(fmt), finite_word)
    result = np.where(zero_a, _word_zero(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)


def abs2(a: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """|z|² as a positive real SPF word: ``L_out = 2·L_a`` (saturated), Θ_out = 0.

    SPF-32 follows AlgRef v0.2 §5.6 in its compatibility spelling (SPF32 word
    with Θ = 0): ZERO and NaN return themselves, otherwise the reserved L
    canonicalizes (§2.4) before doubling.
    """
    la, ta = _unpack(a, fmt)
    zero_theta = np.zeros_like(la, dtype=fmt.theta_dtype)

    if _is_v02(fmt):
        nan_a, zero_a = _classify_v02(la, ta, fmt)
        result = _pack(_clip_finite(_canon_l(la, fmt) * 2, fmt), zero_theta, fmt)
        result = np.where(zero_a, _word_zero(fmt), result)
        result = np.where(nan_a, _word_nan(fmt), result)
        return result.astype(fmt.word_dtype, copy=False)

    l_dbl = _clip_finite(la.astype(_acc_dtype(fmt)) * 2, fmt)
    finite_word = _pack(l_dbl, zero_theta, fmt)

    nan_a, inf_a, zero_a = _classify(la, ta, fmt)
    inf_word = _pack(
        np.full_like(l_dbl, fmt.l_code_inf, dtype=fmt.l_signed_dtype),
        zero_theta,
        fmt,
    )
    result = np.where(inf_a, inf_word, finite_word)
    result = np.where(zero_a, _word_zero(fmt), result)
    result = np.where(nan_a, _word_nan(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)


def neg(a: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """Unary negation: Θ ↦ Θ + π (XOR the Θ field's sign bit). Sentinels preserved.

    π is exactly half a circle in the Θ encoding (``2**(theta_bits-1)``), so
    negation is a single XOR on the low ``theta_bits`` of the packed word.

    SPF-32 follows AlgRef v0.2 §5.4: ZERO and NaN return themselves, and the
    reserved L canonicalizes per §2.4 (so the XOR shortcut, which would preserve
    ``L == 32767``, is replaced by an explicit repack there).
    """
    a_arr = np.asarray(a, dtype=fmt.word_dtype)
    la, ta = _unpack(a_arr, fmt)

    if _is_v02(fmt):
        nan_a, zero_a = _classify_v02(la, ta, fmt)
        acc = _acc_dtype(fmt)
        half_turn = fmt.theta_scale >> 1
        t_half = ((ta.astype(acc) + half_turn) & fmt.word_mask_theta).astype(fmt.theta_dtype)
        result = _pack(_clip_finite(_canon_l(la, fmt), fmt), t_half, fmt)
        result = np.where(zero_a, _word_zero(fmt), result)
        result = np.where(nan_a, _word_nan(fmt), result)
        return result.astype(fmt.word_dtype, copy=False)

    nan_a, _inf_a, zero_a = _classify(la, ta, fmt)
    pi_bit = 1 << (fmt.theta_bits - 1)
    pi_mask = pi_bit if fmt.word_dtype is object else fmt.word_dtype(pi_bit)
    toggled = a_arr ^ pi_mask
    result = np.where(zero_a, _word_zero(fmt), toggled)
    result = np.where(nan_a, _word_nan(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)


def twiddle(k: npt.ArrayLike, n: int, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """DFT twiddle factor ``W_N^k = e^(2πi·k/N)`` as a unit-magnitude SPF word.

    Exact when ``N`` divides ``2**theta_bits`` (SPF-16: N | 256, SPF-32: N | 2¹⁶,
    SPF-64: N | 2³²). Otherwise quantized to the nearest Θ code; the residual
    phase error is bounded by ``π / 2**theta_bits``.
    """
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    k_arr = np.asarray(k, dtype=np.int64)
    if _is_wide(fmt):
        # Exact integer twiddle: ``k·2⁶⁴ / N`` in Python int (the float path loses
        # the low bits of a 64-bit Θ). Exact when N | 2⁶⁴, nearest-code otherwise.
        ks = k_arr.astype(object)
        theta_u = ((ks * fmt.theta_scale // n) % fmt.theta_scale).astype(fmt.theta_dtype)
    else:
        theta_code = np.mod(np.rint(k_arr * fmt.theta_scale / n).astype(np.int64), fmt.theta_scale)
        theta_u = theta_code.astype(fmt.theta_dtype)
    l_zero = np.zeros_like(theta_u, dtype=fmt.l_signed_dtype)
    return _pack(l_zero, theta_u, fmt).astype(fmt.word_dtype, copy=False)


# ── Complex ADD (Sb-based) and SUB ───────────────────────────────────────────


def add(a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """SPF complex addition via the exact FP64 Sb() formula (spec §2.4).

    ``a + b = a · (1 + 2^(−ΔL) · e^(iΔΘ))`` with ``ΔL = L_a − L_b ≥ 0``,
    ``ΔΘ = Θ_b − Θ_a``. Operands are swapped so the larger magnitude plays
    ``a``; the Sb() result ``(δL, δΘ)`` is in physical units (log₂, radians)
    and is requantized into the format's L/Θ codes.

    SPF-32 follows the AlgRef v0.2 §7.1 sentinel list — only ``is_zero`` and
    ``is_nan``. §2.2 defines no INF class, so a reserved word canonicalizes
    (§2.4) into an ordinary finite operand and flows into the Sb path; nothing
    here can emit ``L == l_code_inf`` with a non-NaN Θ. The other widths keep the
    v0.1 cascade (NaN > inf > zero > finite). The finite Sb arithmetic below is
    identical in both profiles.
    """
    v02 = _is_v02(fmt)
    a_arr = np.asarray(a, dtype=fmt.word_dtype)
    b_arr = np.asarray(b, dtype=fmt.word_dtype)
    a_arr, b_arr = np.broadcast_arrays(a_arr, b_arr)
    if v02:
        # §2.3 (amended): operands are canonicalized before their fields are
        # read — including the zero-passthrough branches, which return an
        # operand word whole.
        a_arr = _canon_words(a_arr, fmt)
        b_arr = _canon_words(b_arr, fmt)

    la, ta = _unpack(a_arr, fmt)
    lb, tb = _unpack(b_arr, fmt)

    if v02:
        # No INF class (§2.2), so inf_a / inf_b do not exist in this profile —
        # the overlay below is guarded on ``v02`` instead.
        nan_a, zero_a = _classify_v02(la, ta, fmt)
        nan_b, zero_b = _classify_v02(lb, tb, fmt)
    else:
        nan_a, inf_a, zero_a = _classify(la, ta, fmt)
        nan_b, inf_b, zero_b = _classify(lb, tb, fmt)

    # Swap so the first operand has the larger L. For sentinel pairs the
    # finite_word path produces garbage; overlaid below.
    swap = la < lb
    la_big = np.where(swap, lb, la)
    ta_big = np.where(swap, tb, ta)
    lb_sm = np.where(swap, la, lb)
    tb_sm = np.where(swap, ta, tb)

    # ΔL in log₂ units, always ≥ 0.
    dl = (la_big.astype(np.float64) - lb_sm.astype(np.float64)) / fmt.l_scale
    # ΔΘ in signed (−π, π] rad. Wrap the unsigned difference via signed midpoint.
    acc = _acc_dtype(fmt)
    dt_u = (tb_sm.astype(acc) - ta_big.astype(acc)) & fmt.word_mask_theta
    half_circle = fmt.theta_scale >> 1
    dt_signed = np.where(dt_u > half_circle, dt_u - fmt.theta_scale, dt_u)
    dtheta = dt_signed.astype(np.float64) / fmt.theta_scale * (2.0 * np.pi)

    with np.errstate(invalid="ignore", divide="ignore"):
        delta_l_log, delta_t_rad = _sb_exact(dl, dtheta)

    l_out_fp = la_big.astype(np.float64) + delta_l_log * fmt.l_scale
    t_out_fp = ta_big.astype(np.float64) + delta_t_rad / (2.0 * np.pi) * fmt.theta_scale

    # delta_l_log is −∞ at exact cancellation; route non-finite to L_CODE_MIN_FINITE.
    with np.errstate(invalid="ignore"):
        l_safe = np.where(np.isfinite(l_out_fp), l_out_fp, float(fmt.l_code_min_finite))
        l_out_codes = _rint_to_int_array(l_safe, fmt)

    # v0.2 §7.1 makes exact antipodal cancellation a *sentinel* result, tested
    # before the Sb core: "if gap == 0 and dT == -32768: return ZERO" (and again
    # "if mag == 0: return ZERO" after the local frame). v0.1 let it fall through
    # to the L_CODE_MIN_FINITE floor above, which is what the other widths keep.
    cancel_to_zero = (~np.isfinite(l_out_fp)) | ((dl == 0.0) & (dt_u == half_circle)) if v02 else None
    l_out_clipped = _clip_finite(l_out_codes, fmt)
    t_out_u = _theta_code_from_float(t_out_fp, fmt)

    finite_word = _pack(l_out_clipped, t_out_u, fmt)

    # Sentinel overlays. v0.2 (§7.1): NaN > zero-passthrough > finite — no INF
    # class, so nothing below can mint a directional-infinity word. v0.1 keeps
    # NaN > inf > zero > finite, where non-matched inf+inf → NaN because
    # different rays don't compose to a deterministic direction.
    only_a_zero = zero_a & ~zero_b
    only_b_zero = zero_b & ~zero_a
    both_zero = zero_a & zero_b

    result = finite_word
    if cancel_to_zero is not None:
        result = np.where(cancel_to_zero, _word_zero(fmt), result)
    result = np.where(only_a_zero, b_arr, result)
    result = np.where(only_b_zero, a_arr, result)
    result = np.where(both_zero, _word_zero(fmt), result)

    if v02:
        is_nan_out = nan_a | nan_b
    else:
        both_inf = inf_a & inf_b
        same_theta = ta == tb
        is_nan_out = nan_a | nan_b | (both_inf & ~same_theta)
        is_inf_out = (inf_a | inf_b) & ~is_nan_out

        inf_theta = np.where(inf_a, ta, tb)
        inf_l = np.full_like(la_big, fmt.l_code_inf, dtype=fmt.l_signed_dtype)
        inf_word = _pack(inf_l, inf_theta, fmt)
        result = np.where(is_inf_out, inf_word, result)

    result = np.where(is_nan_out, _word_nan(fmt), result)
    return result.astype(fmt.word_dtype, copy=False)


def sub(a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format) -> npt.NDArray[np.unsignedinteger]:
    """Complex subtraction: ``a − b = a + (−b)``."""
    return add(a, neg(b, fmt), fmt)


# ── FFT butterfly primitive (spec §2.4) ──────────────────────────────────────


def butterfly(
    a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike, fmt: Format
) -> tuple[
    npt.NDArray[np.unsignedinteger],
    npt.NDArray[np.unsignedinteger],
]:
    """Radix-2 DIT butterfly: ``a' = a + W·b``, ``b' = a − W·b``.

    The MUL is free (integer add on L and Θ); each of the two ADDs uses the
    exact FP64 Sb() formula. Sentinels propagate through mul/add/sub.
    """
    wb = mul(w, b, fmt)
    return add(a, wb, fmt), sub(a, wb, fmt)


# ── Dual-lane Cartesian dot product (Task 3.5) ───────────────────────────────


def dot(a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format) -> np.generic:
    """SPF dot product ``Σ a[i]·b[i]`` with Kahan-compensated FP64 accumulation.

    Multiplication stays in log-polar (one integer-add ``mul`` per element).
    Each product decodes once to complex128; the real and imaginary channels
    are summed with Kahan compensation, bounding the accumulator's relative
    error at O(1)·ε instead of O(N)·ε. The final sum is re-encoded to one SPF
    word. See ``docs/research/spf-extended-research.md`` §"Accumulator
    strategy" for the Cartesian-until-readout argument.

    Inputs are flattened; shapes must match. Empty input returns the zero
    sentinel.
    """
    a_arr = np.asarray(a, dtype=fmt.word_dtype).ravel()
    b_arr = np.asarray(b, dtype=fmt.word_dtype).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"dot: shape mismatch, got {a_arr.shape} and {b_arr.shape}")
    if a_arr.size == 0:
        return _word_zero(fmt)  # type: ignore[no-any-return]

    products = mul(a_arr, b_arr, fmt)
    decoded = decode(products, fmt)

    sum_re = 0.0
    sum_im = 0.0
    c_re = 0.0
    c_im = 0.0
    for v in decoded:
        y = v.real - c_re
        t = sum_re + y
        c_re = (t - sum_re) - y
        sum_re = t
        y = v.imag - c_im
        t = sum_im + y
        c_im = (t - sum_im) - y
        sum_im = t

    total = np.array([complex(sum_re, sum_im)], dtype=np.complex128)
    return encode(total, fmt)[0]  # type: ignore[no-any-return]


def dot_nokahan(a: npt.ArrayLike, b: npt.ArrayLike, fmt: Format) -> np.generic:
    """Same as :func:`dot` without Kahan compensation (plain numpy sum).

    Exists purely to quantify Kahan's contribution in the accuracy bench; do
    not use it in production paths. Error behaviour degrades to O(N)·ε.
    """
    a_arr = np.asarray(a, dtype=fmt.word_dtype).ravel()
    b_arr = np.asarray(b, dtype=fmt.word_dtype).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(f"dot_nokahan: shape mismatch, got {a_arr.shape} and {b_arr.shape}")
    if a_arr.size == 0:
        return _word_zero(fmt)  # type: ignore[no-any-return]

    products = mul(a_arr, b_arr, fmt)
    decoded = decode(products, fmt)
    total = np.array([decoded.sum()], dtype=np.complex128)
    return encode(total, fmt)[0]  # type: ignore[no-any-return]
