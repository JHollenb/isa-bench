"""SPF-32 scalar format public surface.

Spec: ``SPF_Algorithm_Reference_v0.2.md`` §2–§7 (the arithmetic authority) plus
``docs/spf-formats-and-test-plans.md`` §2 for the word layout. This module is a
thin wrapper that binds :data:`spf.scalar._constants.SPF32` into every op
exported from :mod:`spf.scalar._ops`. The family core lives in ``_ops``;
per-format modules exist so callers can write ``from spf.scalar import spf32``
and get a namespace of SPF-32-specific functions without passing a format
argument.

**One library switch.** Every op below routes through the module-level
``_c_spf32`` binding, captured once at import: it is the C library when
``libspf32`` is importable and ``None`` otherwise. Nothing re-reads
``LIBSPF_AVAILABLE`` inside a function body, so
``monkeypatch.setattr(spf.scalar.spf32, "_c_spf32", None)`` deterministically
forces the whole surface onto the numpy fallback — which is how
``tests/scalar/test_v02_numpy_fallback_parity.py`` proves the two paths agree
bit-for-bit. ``twiddle``, ``dot``, ``dot_nokahan``, ``add_poly7`` and
``sub_poly7`` are pure-Python by design and ignore the binding (``twiddle``'s
bit-identity with C is pinned by
``tests/native_bindings/test_T1_c_crossvalidate.py``).

**Sentinel profile: v0.2.** AlgRef §2.2 defines exactly two sentinels (ZERO and
the single NaN word ``0x7FFFFFFF``); ``L == 32767`` with ``Θ != 65535`` is a
*reserved* encoding that §2.4 canonicalizes to ``L_MAX_FINITE`` before an
operation reads it. ``add``/``sub``/``butterfly`` apply that canonicalization
here (:func:`_canon_reserved`) because the C ``spf32_add`` / FB7 family still
ships the v0.1 directional-infinity cascade internally; pre-canonicalizing makes
the C and numpy paths agree on every sentinel and reserved word. Their *finite*
paths still differ by up to one Q7.8 rounding step (C poly7/FB7 vs the Python
exact-Sb reference) — a documented algorithmic difference, see
``tests/native_bindings/test_T1_c_crossvalidate.py`` §"T1.6 / T1.7".
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.native._lib import LIBSPF_AVAILABLE
from spf.scalar import _ops
from spf.scalar._constants import SPF32

if LIBSPF_AVAILABLE:
    from spf.native import spf32 as _c_spf32
else:
    _c_spf32 = None  # type: ignore[assignment]

FORMAT = SPF32

__all__ = [
    "FORMAT",
    "abs2",
    "add",
    "add_hybrid",
    "add_poly7",
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
    "sub_hybrid",
    "sub_poly7",
    "twiddle",
]

Word = npt.NDArray[np.uint32]
Complex = npt.NDArray[np.complex128]

_RESERVED_L: int = SPF32.l_code_inf & 0xFFFF          # 0x7FFF
_L_MAX_FINITE_SHIFTED: int = (SPF32.l_code_max_finite & 0xFFFF) << 16


def _canon_reserved(w: np.ndarray) -> np.ndarray:
    """AlgRef v0.2 §2.4 on packed words: fold reserved ``L == 32767`` to L_MAX.

    §2.3 (amended 2026-09-18): "Operands of every §5–§7 operation are
    canonicalized (§2.4) before their fields are read … a reserved word with
    ``L == 32767`` and ``T != 65535`` enters an operation as ``L = 32766``."

    The C core (``spf32_canon_word`` / ``spf32_canon_l``) and
    :mod:`spf.scalar._ops` both apply this at op entry, so ``encode`` / ``decode``
    / ``mul`` / ``div`` / ``conj`` / ``neg`` / ``abs2`` / ``add`` / ``sub`` /
    ``butterfly`` need nothing here. :func:`add_poly7` does: it is a pure-Python
    reference built on :mod:`spf.fft.fused`, which takes already-canonical
    ``(L, Θ)`` pairs. Zero-cost when no reserved word is present.
    """
    l_hi = w >> np.uint32(16)
    reserved = (l_hi == np.uint32(_RESERVED_L)) & (
        (w & np.uint32(0xFFFF)) != np.uint32(0xFFFF)
    )
    if not reserved.any():
        return w
    fixed = (w & np.uint32(0xFFFF)) | np.uint32(_L_MAX_FINITE_SHIFTED)
    return np.ascontiguousarray(np.where(reserved, fixed, w).astype(np.uint32))


def _c_unary(entry: str, a: npt.ArrayLike) -> Word:
    """Call a C unary word op, preserving the input's shape.

    ``np.ascontiguousarray`` inside the ctypes layer promotes a 0-d array to 1-d,
    so reshape back: the numpy fallback keeps 0-d, and the two paths must be
    indistinguishable (``int(spf32.mul(w, w))`` on scalar words depends on it).
    """
    arr = np.asarray(a, dtype=np.uint32)
    out = getattr(_c_spf32, entry)(arr)
    return out.reshape(arr.shape).astype(np.uint32, copy=False)  # type: ignore[no-any-return]


def _c_binary(entry: str, a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """Call a C binary word op, preserving the broadcast shape (see :func:`_c_unary`)."""
    a_b, b_b = np.broadcast_arrays(
        np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    )
    out = getattr(_c_spf32, entry)(a_b, b_b)
    return out.reshape(a_b.shape).astype(np.uint32, copy=False)  # type: ignore[no-any-return]


def encode(z: npt.ArrayLike) -> Word:
    """Encode complex values into SPF-32 words.

    Routes through the C ``spf32_encode_array`` entry point when libspf32
    is available; falls back to the pure-numpy ``_ops.encode`` reference
    otherwise. Output shape matches ``np.asarray(z).shape``.
    """
    if _c_spf32 is not None:
        arr = np.asarray(z, dtype=np.complex128)
        out = _c_spf32.encode(arr)
        return out.reshape(arr.shape).astype(np.uint32, copy=False)

    return _ops.encode(z, FORMAT).astype(np.uint32, copy=False)


def decode(w: npt.ArrayLike) -> Complex:
    """Decode SPF-32 words back to complex128.

    Routes through the C ``spf32_decode_array`` entry point when libspf32
    is available; falls back to the pure-numpy ``_ops.decode`` reference
    otherwise.
    """
    if _c_spf32 is not None:
        arr = np.asarray(w, dtype=np.uint32)
        out = _c_spf32.decode(arr)
        return out.reshape(arr.shape)

    return _ops.decode(w, FORMAT)


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-32 multiply: ``L_out = L_a + L_b``, ``Θ_out = Θ_a + Θ_b`` (mod 2¹⁶).

    AlgRef v0.2 §5.1. Routes through C ``spf32_mul_array`` when available; the
    numpy fallback is bit-identical (sentinels and reserved words included).
    """
    if _c_spf32 is not None:
        return _c_binary("mul", a, b)
    return _ops.mul(a, b, FORMAT).astype(np.uint32, copy=False)


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-32 divide: ``L_out = L_a − L_b``, ``Θ_out = Θ_a − Θ_b``.

    AlgRef v0.2 §5.2 — ``ZERO/ZERO = ZERO`` and ``finite/ZERO = NaN``.
    """
    if _c_spf32 is not None:
        return _c_binary("div", a, b)
    return _ops.div(a, b, FORMAT).astype(np.uint32, copy=False)


def conj(a: npt.ArrayLike) -> Word:
    """Complex conjugate: L unchanged, Θ ↦ (−Θ) mod 2¹⁶ (AlgRef v0.2 §5.3)."""
    if _c_spf32 is not None:
        return _c_unary("conj", a)
    return _ops.conj(a, FORMAT).astype(np.uint32, copy=False)


def abs2(a: npt.ArrayLike) -> Word:
    """|z|² as a positive-real SPF-32 word: ``L_out = 2·L_a``, Θ_out = 0.

    AlgRef v0.2 §5.6 compatibility spelling.
    """
    if _c_spf32 is not None:
        return _c_unary("abs2", a)
    return _ops.abs2(a, FORMAT).astype(np.uint32, copy=False)


def neg(a: npt.ArrayLike) -> Word:
    """Unary negation: Θ ↦ Θ + π, a half-turn in phase (AlgRef v0.2 §5.4)."""
    if _c_spf32 is not None:
        return _c_unary("neg", a)
    return _ops.neg(a, FORMAT).astype(np.uint32, copy=False)


def twiddle(k: npt.ArrayLike, n: int) -> Word:
    """DFT twiddle ``W_N^k = e^(2πi·k/N)`` as a unit-magnitude SPF-32 word."""
    return _ops.twiddle(k, n, FORMAT).astype(np.uint32, copy=False)


def _sentinel_add_scalar(wa: int, wb: int) -> int | None:
    """Return the sentinel result for a scalar add, or None if it is ordinary.

    AlgRef v0.2 §7.1 lists exactly two sentinel classes, so this is short: NaN on
    either side → the canonical NaN word. §2.2 defines no INF class, so a
    reserved ``L == 32767`` / ``Θ != 0xFFFF`` word is **not** a sentinel — the
    caller canonicalizes it (:func:`_canon_reserved`) and it flows into the
    polynomial path as an ordinary ``L_MAX_FINITE`` operand. Zero operands also
    return None so the caller delegates to the fused butterfly, which already
    implements §7.1's zero passthrough.

    v0.1 split ``L == 32767`` into a directional infinity here (inf + finite =
    inf, opposite rays → NaN); that class is gone from the whole §6/§7 family
    (2026-09-18 conformance audit, deviation #9).
    """
    la = (wa >> 16) & 0xFFFF
    lb = (wb >> 16) & 0xFFFF
    if la != 0x7FFF and lb != 0x7FFF:
        return None
    if (la == 0x7FFF and (wa & 0xFFFF) == 0xFFFF) or (
        lb == 0x7FFF and (wb & 0xFFFF) == 0xFFFF
    ):
        return 0x7FFFFFFF  # WORD_NAN — §2.2's only non-zero sentinel
    return None  # reserved word: canonicalized by the caller, then ordinary


def _hybrid_pair(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    precision: str,
    use_native: bool | None,
) -> tuple[Word, Word]:
    """Delegate the opt-in pair to the existing FB3 word-plane consumer."""
    from spf.tensor.v02_planes import spf_pair_addsub_hybrid

    a_b, b_b = np.broadcast_arrays(np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32))
    native = use_native if use_native is not None else (None if _c_spf32 is not None else False)
    plus, minus = spf_pair_addsub_hybrid(
        np.ascontiguousarray(a_b), np.ascontiguousarray(b_b), precision, use_native=native
    )
    return (
        np.asarray(plus, dtype=np.uint32).reshape(a_b.shape),
        np.asarray(minus, dtype=np.uint32).reshape(a_b.shape),
    )


def add_hybrid(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    precision: str = "accurate",
    *,
    use_native: bool | None = None,
) -> Word:
    """Opt-in FB3-Hybrid addition (AlgRef v0.2 §7.4–7.5).

    Uses the existing native plane kernel when available and its NumPy
    reference otherwise. ``use_native=False`` explicitly selects the reference.
    This candidate has a different rounding contract from legacy FB7; default
    :func:`add`, :func:`sub`, and :func:`butterfly` retain their existing paths.
    """
    return _hybrid_pair(a, b, precision, use_native)[0]


def sub_hybrid(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    precision: str = "accurate",
    *,
    use_native: bool | None = None,
) -> Word:
    """Opt-in FB3-Hybrid subtraction; see :func:`add_hybrid`."""
    return _hybrid_pair(a, b, precision, use_native)[1]


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """SPF-32 complex addition (poly7 FB7 ACCURATE; SPFv2 normative §4.3).

    Routes through the C spf32_add_array entry point when libspf32 is
    available (applies full sentinel cascade then delegates to poly7 ACCURATE);
    falls back to the vectorized format-generic exact-Sb reference otherwise. Operands
    broadcast via NumPy's standard rules — output shape is the broadcast
    result of ``a`` and ``b``.

    AlgRef v0.2 §7.1 lists only ``is_zero`` and ``is_nan``; reserved words are
    canonicalized (§2.4) up front so the C path's internal v0.1 INF class can
    never fire and both paths agree on every sentinel.
    """
    a_b, b_b = np.broadcast_arrays(
        np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    )
    out_shape = a_b.shape
    a_c = np.ascontiguousarray(a_b)
    b_c = np.ascontiguousarray(b_b)

    if _c_spf32 is not None:
        return (
            _c_spf32._binary_word_prebroadcast("spf32_add_array", a_c, b_c)
            .reshape(out_shape)
            .astype(np.uint32, copy=False)
        )

    return _ops.add(a_c, b_c, FORMAT).reshape(out_shape).astype(np.uint32, copy=False)


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> Word:
    """Complex subtraction: ``a − b = a + (−b)`` (poly7 FB7 ACCURATE).

    Broadcasts via NumPy rules; output shape is the broadcast of ``a`` and ``b``.
    Reserved words canonicalize per AlgRef v0.2 §2.4 — see :func:`add`.
    """
    a_b, b_b = np.broadcast_arrays(
        np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    )
    out_shape = a_b.shape
    a_c = np.ascontiguousarray(a_b)
    b_c = np.ascontiguousarray(b_b)

    if _c_spf32 is not None:
        return (
            _c_spf32._binary_word_prebroadcast("spf32_sub_array", a_c, b_c)
            .reshape(out_shape)
            .astype(np.uint32, copy=False)
        )

    return _ops.sub(a_c, b_c, FORMAT).reshape(out_shape).astype(np.uint32, copy=False)


def butterfly(a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike) -> tuple[Word, Word]:
    """Radix-2 DIT butterfly: ``a' = a + W·b``, ``b' = a − W·b``.

    SPFv2 normative: routes through FB7 fused butterfly (§5.2 ACCURATE).
    Routes through the C array entry point when libspf32 is available
    (spf32_butterfly_fb7_array, ACCURATE mode); falls back to the vectorized
    format-generic exact-Sb reference otherwise. ``a``, ``b``, ``w`` broadcast via
    NumPy rules. Reserved words canonicalize per AlgRef v0.2 §2.4 — see
    :func:`add`.
    """
    a_b, b_b, w_b = np.broadcast_arrays(
        np.asarray(a, dtype=np.uint32),
        np.asarray(b, dtype=np.uint32),
        np.asarray(w, dtype=np.uint32),
    )
    out_shape = a_b.shape
    a_c = np.ascontiguousarray(a_b)
    b_c = np.ascontiguousarray(b_b)
    w_c = np.ascontiguousarray(w_b)

    if _c_spf32 is not None:
        from spf.native import fb7 as _c_fb7
        top, bot = _c_fb7.butterfly_array(a_c, b_c, w_c, mode="accurate")
        return (
            top.reshape(out_shape).astype(np.uint32, copy=False),
            bot.reshape(out_shape).astype(np.uint32, copy=False),
        )

    top, bot = _ops.butterfly(a_c, b_c, w_c, FORMAT)
    return (
        top.reshape(out_shape).astype(np.uint32, copy=False),
        bot.reshape(out_shape).astype(np.uint32, copy=False),
    )


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> np.uint32:
    """SPF-32 dot product: ``Σ a[i]·b[i]`` with Kahan-compensated FP64 accum."""
    return np.uint32(_ops.dot(a, b, FORMAT))


def dot_nokahan(a: npt.ArrayLike, b: npt.ArrayLike) -> np.uint32:
    """SPF-32 dot product without Kahan compensation (baseline for accuracy bench)."""
    return np.uint32(_ops.dot_nokahan(a, b, FORMAT))


def add_poly7(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    mode: str | int = "balanced",
) -> Word:
    """Poly7 ADD: Sb-free complex addition via FB7 polynomial kernels (§4.3).

    Pure-Python reference — uses :func:`spf.fft.fused.spf_butterfly_fb7_accurate`
    as the ACCURATE-mode reference path (dominant-gap bypass disabled, same
    polynomial finish stage as the other modes). FAST and BALANCED modes use
    :func:`spf.fft.fused.spf_butterfly_fb7_fast` with the corresponding
    ``dominant_gap_q78`` threshold.

    This is the spec-side reference; for performance-critical code use
    :func:`spf.native.spf32.add_poly7` (C-backed via ``spf32_add_poly7_array``).
    """
    from spf.fft.fused import (
        DOMINANT_GAP_BALANCED_Q78,
        DOMINANT_GAP_FAST_Q78,
        pack_spf32,
        spf_butterfly_fused,
        unpack_spf32,
    )

    if isinstance(mode, str):
        mode_key = mode.lower()
        if mode_key == "accurate":
            gap: int | None = None
        elif mode_key == "balanced":
            gap = DOMINANT_GAP_BALANCED_Q78
        elif mode_key == "fast":
            gap = DOMINANT_GAP_FAST_Q78
        else:
            raise ValueError(f"unknown poly7 mode {mode!r}")
    elif isinstance(mode, int):
        # 0=fast, 1=balanced, 2=accurate — mirrors SPF32_FB7_MODE_* constants
        if mode == 0:
            gap = DOMINANT_GAP_FAST_Q78
        elif mode == 1:
            gap = DOMINANT_GAP_BALANCED_Q78
        elif mode == 2:
            gap = None
        else:
            raise ValueError(f"unknown poly7 mode integer {mode!r}")
    else:
        raise TypeError(f"mode must be str or int, got {type(mode)!r}")

    a_b, b_b = np.broadcast_arrays(
        np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    )
    out_shape = a_b.shape
    # §2.4 at op entry — spf.fft.fused takes canonical (L, Θ) pairs and would
    # otherwise carry a reserved L == 32767 straight through the polynomial.
    a_arr = _canon_reserved(np.ascontiguousarray(a_b)).reshape(-1)
    b_arr = _canon_reserved(np.ascontiguousarray(b_b)).reshape(-1)
    out = np.empty_like(a_arr)
    for i, (wa, wb) in enumerate(zip(a_arr, b_arr, strict=False)):
        wa_int = int(wa)
        wb_int = int(wb)
        sentinel = _sentinel_add_scalar(wa_int, wb_int)
        if sentinel is not None:
            out[i] = sentinel
        else:
            sa = unpack_spf32(wa_int)
            sb = unpack_spf32(wb_int)
            result, _ = spf_butterfly_fused(sa, sb, 0, dominant_gap_q78=gap)
            out[i] = pack_spf32(result)
    return out.reshape(out_shape).astype(np.uint32, copy=False)


def sub_poly7(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    mode: str | int = "balanced",
) -> Word:
    """Poly7 SUB: equivalent to ``add_poly7(a, neg(b), mode)``.

    Pure-Python reference; see :func:`add_poly7` for mode semantics.
    """
    return add_poly7(a, neg(np.asarray(b, dtype=np.uint32)), mode)
