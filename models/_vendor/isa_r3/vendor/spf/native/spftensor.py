"""NumPy wrappers around ``libspftensor`` ctypes bindings.

Provides ergonomic numpy array in/out wrappers that:
- enforce C-contiguous layout via ``np.ascontiguousarray``
- pass array data via ``.ctypes.data_as``
- check return codes and raise on error
- skip cleanly when ``LIBSPFTENSOR_AVAILABLE`` is False

These wrappers are standalone — do NOT import from spf.tensor.reducers or
spf.tensor.core at module level (those modules use these wrappers, not the
other way round).

Domain/routing/precision constants mirror the C header #defines:

    SPFT_FMT_SPF32 = 2
    SPFT_DOMAIN_* in {NONE=-1, TROPICAL=0, LOGPOLAR=1, CARTESIAN=2}
    SPFT_ROUTE_* in {STATIC=0, ADAPTIVE=1, PINNED=2}
    SPFT_MODE_* in {FAST=0, BALANCED=1, ACCURATE=2, EXACT=3}
    SPFT_POLICY_AUTO = 10
    SPFT_SCORE_* in {F32=0, F64=1, L32=2}
    SPFT_MASK_* in {NONE=0, CAUSAL=1}
    SPFT_LSE_* in {EXACT=0, TROPICAL_CORR=1}
    SPFT_OPERAND_* in {WORDS=0, C128=1, F32REAL=2}
"""

from __future__ import annotations

import ctypes
from typing import cast

import numpy as np
import numpy.typing as npt

from ._libspftensor import (
    LIBSPFTENSOR_AVAILABLE,
    LIBSPFTENSOR_PATH,
    CContractOpts,
    CPhaseRouteOpts,
    CReducerSpec,
    CRowStats,
    CSoftmaxOpts,
    CTelemetry,
    load_library,
)

__all__ = [
    "LIBSPFTENSOR_AVAILABLE",
    "LIBSPFTENSOR_PATH",
    "SPFT_DOMAIN_CARTESIAN",
    "SPFT_DOMAIN_LOGPOLAR",
    "SPFT_DOMAIN_NONE",
    "SPFT_DOMAIN_TROPICAL",
    "SPFT_EADAPTIVE",
    "SPFT_EBADMASK",
    "SPFT_EBADSPEC",
    "SPFT_EFMT",
    "SPFT_EINVAL",
    "SPFT_ENOMEM",
    "SPFT_FMT_SPF32",
    "SPFT_LSE_EXACT",
    "SPFT_LSE_TROPICAL_CORR",
    "SPFT_MASK_CAUSAL",
    "SPFT_MASK_NONE",
    "SPFT_MODE_ACCURATE",
    "SPFT_MODE_BALANCED",
    "SPFT_MODE_EXACT",
    "SPFT_MODE_FAST",
    "SPFT_OK",
    "SPFT_OPERAND_C128",
    "SPFT_OPERAND_F32REAL",
    "SPFT_OPERAND_WORDS",
    "SPFT_POLICY_AUTO",
    "SPFT_ROUTE_ADAPTIVE",
    "SPFT_ROUTE_PINNED",
    "SPFT_ROUTE_STATIC",
    "SPFT_SCORE_F32",
    "SPFT_SCORE_F64",
    "SPFT_SCORE_L32",
    "CContractOpts",
    "CPhaseRouteOpts",
    "CReducerSpec",
    "CRowStats",
    "CSoftmaxOpts",
    "CTelemetry",
    "attention_route",
    "compute_stats_rows",
    "contract",
    "decode_real_f32",
    "encode_real_f32",
    "low_rank_apply",
    "lse_rows",
    "make_reducer_spec",
    "pack_planes",
    "phase_features",
    "phase_route",
    "phase_stats",
    "reduce_rows",
    "reduce_rows_planes",
    "route",
    "softmax_rows_f32",
    "softmax_rows_words",
    "unpack_planes",
]

# ---------------------------------------------------------------------------
# C constants (mirror of spftensor.h #defines)
# ---------------------------------------------------------------------------

SPFT_FMT_SPF32 = 2

SPFT_DOMAIN_NONE = -1
SPFT_DOMAIN_TROPICAL = 0
SPFT_DOMAIN_LOGPOLAR = 1
SPFT_DOMAIN_CARTESIAN = 2

SPFT_ROUTE_STATIC = 0
SPFT_ROUTE_ADAPTIVE = 1
SPFT_ROUTE_PINNED = 2

SPFT_MODE_FAST = 0
SPFT_MODE_BALANCED = 1
SPFT_MODE_ACCURATE = 2
SPFT_MODE_EXACT = 3

SPFT_POLICY_AUTO = 10

SPFT_SCORE_F32 = 0
SPFT_SCORE_F64 = 1
SPFT_SCORE_L32 = 2

SPFT_MASK_NONE = 0
SPFT_MASK_CAUSAL = 1

SPFT_LSE_EXACT = 0
SPFT_LSE_TROPICAL_CORR = 1

SPFT_OPERAND_WORDS = 0
SPFT_OPERAND_C128 = 1
SPFT_OPERAND_F32REAL = 2

# Error codes
SPFT_OK = 0
SPFT_EBADSPEC = -1
SPFT_EFMT = -2
SPFT_EADAPTIVE = -3
SPFT_ENOMEM = -4
SPFT_EBADMASK = -5
SPFT_EINVAL = -6

_ERROR_NAMES: dict[int, str] = {
    SPFT_EBADSPEC: "SPFT_EBADSPEC (unsupported policy / missing PINNED domain)",
    SPFT_EFMT: "SPFT_EFMT (fmt != SPFT_FMT_SPF32)",
    SPFT_EADAPTIVE: "SPFT_EADAPTIVE (RoutingPolicy.ADAPTIVE is deferred)",
    SPFT_ENOMEM: "SPFT_ENOMEM",
    SPFT_EBADMASK: "SPFT_EBADMASK (bad/missing mask configuration)",
    SPFT_EINVAL: "SPFT_EINVAL (bad dims / null pointer)",
}


def _check(rc: int, fn: str) -> None:
    """Raise RuntimeError if ``rc`` is not SPFT_OK."""
    if rc != SPFT_OK:
        name = _ERROR_NAMES.get(rc, f"error code {rc}")
        raise RuntimeError(f"{fn} returned {name}")


def _u32p(arr: np.ndarray) -> ctypes._Pointer[ctypes.c_uint32]:
    """Return a POINTER(c_uint32) into a contiguous uint32 array."""
    a = np.ascontiguousarray(arr, dtype=np.uint32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _i16p(arr: np.ndarray) -> ctypes._Pointer[ctypes.c_int16]:
    a = np.ascontiguousarray(arr, dtype=np.int16)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))


def _u16p(arr: np.ndarray) -> ctypes._Pointer[ctypes.c_uint16]:
    a = np.ascontiguousarray(arr, dtype=np.uint16)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _f32p(arr: np.ndarray) -> ctypes._Pointer[ctypes.c_float]:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _f64p(arr: np.ndarray) -> ctypes._Pointer[ctypes.c_double]:
    a = np.ascontiguousarray(arr, dtype=np.float64)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _i32p(arr: np.ndarray) -> ctypes._Pointer[ctypes.c_int32]:
    a = np.ascontiguousarray(arr, dtype=np.int32)
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_reducer_spec(
    *,
    fmt: int = SPFT_FMT_SPF32,
    policy: int = SPFT_POLICY_AUTO,
    precision_mode: int = SPFT_MODE_BALANCED,
    routing: int = SPFT_ROUTE_STATIC,
    domain: int = SPFT_DOMAIN_NONE,
) -> CReducerSpec:
    """Build a :class:`CReducerSpec` with named arguments."""
    spec = CReducerSpec()
    spec.fmt = fmt
    spec.policy = policy
    spec.precision_mode = precision_mode
    spec.routing = routing
    spec.domain = domain
    return spec


# ---------------------------------------------------------------------------
# 1. Batched domain-routed reducer
# ---------------------------------------------------------------------------


def reduce_rows(
    rows: npt.ArrayLike,
    spec: CReducerSpec,
    *,
    want_stats: bool = False,
    want_domain: bool = False,
    want_telemetry: bool = False,
) -> (
    np.ndarray
    | tuple[np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray]
    | tuple[np.ndarray, np.ndarray, np.ndarray, CTelemetry]
):
    """Call ``spft_reduce_rows`` and return the output word array.

    Parameters
    ----------
    rows:
        2-D array of packed SPF-32 words, shape ``(n_rows, row_len)``.
    spec:
        A :class:`CReducerSpec` describing routing, policy, precision.
    want_stats:
        If True, also return a ``(n_rows,)`` array of :class:`CRowStats`.
    want_domain:
        If True, also return a ``(n_rows,)`` uint8 array of chosen domains.
    want_telemetry:
        If True, also return a :class:`CTelemetry` aggregate.

    Returns
    -------
    out : np.ndarray of dtype uint32, shape ``(n_rows,)``
        Reduced output words.  Optional extras follow in order:
        stats_arr, domain_arr, telemetry.
    """
    lib = load_library()
    rows_arr = np.ascontiguousarray(rows, dtype=np.uint32)
    if rows_arr.ndim != 2:
        raise ValueError(f"rows must be 2-D, got shape {rows_arr.shape}")
    n_rows, row_len = rows_arr.shape
    out = np.empty(n_rows, dtype=np.uint32)

    domain_arr: np.ndarray | None = None

    stats_ptr: ctypes._Pointer[CRowStats] | None = None
    domain_ptr: ctypes._Pointer[ctypes.c_uint8] | None = None
    tel: CTelemetry | None = None
    tel_ptr = None

    if want_stats:
        # Use a ctypes Array of CRowStats
        cstats = (CRowStats * n_rows)()
        stats_ptr = ctypes.cast(cstats, ctypes.POINTER(CRowStats))
    if want_domain:
        domain_arr = np.empty(n_rows, dtype=np.uint8)
        domain_ptr = domain_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    if want_telemetry:
        tel = CTelemetry()
        tel_ptr = ctypes.byref(tel)

    rc = lib.spft_reduce_rows(
        rows_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),  # dense: stride == row_len
        ctypes.byref(spec),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        stats_ptr,
        domain_ptr,
        tel_ptr,
    )
    _check(rc, "spft_reduce_rows")

    if not (want_stats or want_domain or want_telemetry):
        return out

    extras: list[object] = [out]
    if want_stats:
        extras.append(cstats)
    if want_domain:
        extras.append(domain_arr)
    if want_telemetry:
        extras.append(tel)
    return tuple(extras)  # type: ignore[return-value]


def reduce_rows_planes(
    L: npt.ArrayLike,
    T: npt.ArrayLike,
    spec: CReducerSpec,
    *,
    want_stats: bool = False,
    want_domain: bool = False,
    want_telemetry: bool = False,
) -> np.ndarray:
    """Call ``spft_reduce_rows_planes`` (split SoA plane form).

    Parameters
    ----------
    L : int16 array, shape ``(n_rows, row_len)``
    T : uint16 array, shape ``(n_rows, row_len)``
    """
    lib = load_library()
    L_arr = np.ascontiguousarray(L, dtype=np.int16)
    T_arr = np.ascontiguousarray(T, dtype=np.uint16)
    if L_arr.shape != T_arr.shape or L_arr.ndim != 2:
        raise ValueError("L and T must be 2-D arrays with the same shape")
    n_rows, row_len = L_arr.shape
    out = np.empty(n_rows, dtype=np.uint32)

    stats_ptr = None
    domain_ptr = None
    tel_ptr = None
    cstats = None
    domain_arr = None
    tel = None

    if want_stats:
        cstats = (CRowStats * n_rows)()
        stats_ptr = ctypes.cast(cstats, ctypes.POINTER(CRowStats))
    if want_domain:
        domain_arr = np.empty(n_rows, dtype=np.uint8)
        domain_ptr = domain_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
    if want_telemetry:
        tel = CTelemetry()
        tel_ptr = ctypes.byref(tel)

    rc = lib.spft_reduce_rows_planes(
        L_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        T_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),  # dense stride
        ctypes.byref(spec),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        stats_ptr,
        domain_ptr,
        tel_ptr,
    )
    _check(rc, "spft_reduce_rows_planes")

    if not (want_stats or want_domain or want_telemetry):
        return out
    extras: list[object] = [out]
    if want_stats:
        extras.append(cstats)
    if want_domain:
        extras.append(domain_arr)
    if want_telemetry:
        extras.append(tel)
    return tuple(extras)  # type: ignore[return-value]


def compute_stats_rows(
    rows: npt.ArrayLike,
    *,
    want_rbar: bool = True,
) -> ctypes.Array[CRowStats]:
    """Call ``spft_compute_stats_rows``; return ``(n_rows,)`` CRowStats array.

    Parameters
    ----------
    rows : uint32 array, shape ``(n_rows, row_len)``
    want_rbar : bool
        Force the (lazy) phase-coherence pass — recommended for faithful routing.
    """
    lib = load_library()
    rows_arr = np.ascontiguousarray(rows, dtype=np.uint32)
    if rows_arr.ndim != 2:
        raise ValueError(f"rows must be 2-D, got shape {rows_arr.shape}")
    n_rows, row_len = rows_arr.shape
    cstats = (CRowStats * n_rows)()
    rc = lib.spft_compute_stats_rows(
        rows_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),
        ctypes.c_int(1 if want_rbar else 0),
        ctypes.cast(cstats, ctypes.POINTER(CRowStats)),
    )
    _check(rc, "spft_compute_stats_rows")
    return cast("ctypes.Array[CRowStats]", cstats)


def route(stats: CRowStats, precision_mode: int) -> int:
    """Call ``spft_route``; return SPFT_DOMAIN_* (>= 0) or raise on error."""
    lib = load_library()
    rc = lib.spft_route(ctypes.byref(stats), ctypes.c_int(precision_mode))
    if rc < 0:
        name = _ERROR_NAMES.get(rc, f"error code {rc}")
        raise RuntimeError(f"spft_route returned {name}")
    return cast(int, rc)


# ---------------------------------------------------------------------------
# 2. Softmax / LSE in the log domain
# ---------------------------------------------------------------------------


def lse_rows(
    scores: npt.ArrayLike,
    opts: CSoftmaxOpts,
    *,
    want_f64: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Call ``spft_lse_rows``; return int32 L-code array (and optionally float64).

    Parameters
    ----------
    scores : array, shape ``(n_rows, row_len)``
        Element type must match ``opts.score_kind``:
        F32 → float32, F64 → float64, L32 → int32.
    opts : CSoftmaxOpts
    want_f64 : bool
        Also return LSE as fp64 nats.
    """
    lib = load_library()
    if opts.score_kind == SPFT_SCORE_F32:
        sc_arr = np.ascontiguousarray(scores, dtype=np.float32)
    elif opts.score_kind == SPFT_SCORE_F64:
        sc_arr = np.ascontiguousarray(scores, dtype=np.float64)
    else:
        sc_arr = np.ascontiguousarray(scores, dtype=np.int32)
    if sc_arr.ndim != 2:
        raise ValueError("scores must be 2-D")
    n_rows, row_len = sc_arr.shape
    lse_l = np.empty(n_rows, dtype=np.int32)
    lse_f64 = np.empty(n_rows, dtype=np.float64) if want_f64 else None
    rc = lib.spft_lse_rows(
        sc_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),
        ctypes.byref(opts),
        lse_l.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        lse_f64.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) if lse_f64 is not None else None,
    )
    _check(rc, "spft_lse_rows")
    if want_f64:
        return lse_l, lse_f64  # type: ignore[return-value]
    return lse_l


def softmax_rows_f32(
    scores: npt.ArrayLike,
    opts: CSoftmaxOpts,
) -> np.ndarray:
    """Call ``spft_softmax_rows_f32``; return float32 weight array."""
    lib = load_library()
    if opts.score_kind == SPFT_SCORE_F32:
        sc_arr = np.ascontiguousarray(scores, dtype=np.float32)
    elif opts.score_kind == SPFT_SCORE_F64:
        sc_arr = np.ascontiguousarray(scores, dtype=np.float64)
    else:
        sc_arr = np.ascontiguousarray(scores, dtype=np.int32)
    if sc_arr.ndim != 2:
        raise ValueError("scores must be 2-D")
    n_rows, row_len = sc_arr.shape
    out = np.empty((n_rows, row_len), dtype=np.float32)
    rc = lib.spft_softmax_rows_f32(
        sc_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),
        ctypes.byref(opts),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    )
    _check(rc, "spft_softmax_rows_f32")
    return out


def softmax_rows_words(
    scores: npt.ArrayLike,
    opts: CSoftmaxOpts,
) -> np.ndarray:
    """Call ``spft_softmax_rows_words``; return packed SPF-32 word array."""
    lib = load_library()
    if opts.score_kind == SPFT_SCORE_F32:
        sc_arr = np.ascontiguousarray(scores, dtype=np.float32)
    elif opts.score_kind == SPFT_SCORE_F64:
        sc_arr = np.ascontiguousarray(scores, dtype=np.float64)
    else:
        sc_arr = np.ascontiguousarray(scores, dtype=np.int32)
    if sc_arr.ndim != 2:
        raise ValueError("scores must be 2-D")
    n_rows, row_len = sc_arr.shape
    out = np.empty((n_rows, row_len), dtype=np.uint32)
    rc = lib.spft_softmax_rows_words(
        sc_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),
        ctypes.byref(opts),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
    )
    _check(rc, "spft_softmax_rows_words")
    return out


# ---------------------------------------------------------------------------
# 3. Data×data contraction driver
# ---------------------------------------------------------------------------


def contract(
    A: npt.ArrayLike,
    B: npt.ArrayLike,
    spec: CReducerSpec,
    opts: CContractOpts,
    *,
    want_telemetry: bool = False,
) -> np.ndarray | tuple[np.ndarray, CTelemetry]:
    """Call ``spft_contract``; return C output words.

    Parameters
    ----------
    A : uint32 array, shape ``(M, K)`` (or compatible for a_kind != WORDS)
    B : uint32 array, shape ``(K, N)`` or ``(N, K)`` if transpose_b
    spec, opts : CReducerSpec, CContractOpts
    """
    lib = load_library()
    if opts.a_kind == SPFT_OPERAND_WORDS:
        A_arr = np.ascontiguousarray(A, dtype=np.uint32)
    elif opts.a_kind == SPFT_OPERAND_C128:
        A_arr = np.ascontiguousarray(A, dtype=np.complex128)
    else:
        A_arr = np.ascontiguousarray(A, dtype=np.float32)
    if A_arr.ndim != 2:
        raise ValueError("A must be 2-D")
    M, K = A_arr.shape[:2]

    if opts.b_kind == SPFT_OPERAND_WORDS:
        B_arr = np.ascontiguousarray(B, dtype=np.uint32)
    elif opts.b_kind == SPFT_OPERAND_C128:
        B_arr = np.ascontiguousarray(B, dtype=np.complex128)
    else:
        B_arr = np.ascontiguousarray(B, dtype=np.float32)
    if B_arr.ndim != 2:
        raise ValueError("B must be 2-D")
    N = B_arr.shape[0] if opts.transpose_b else B_arr.shape[1]

    out = np.empty((M, N), dtype=np.uint32)

    tel = CTelemetry() if want_telemetry else None
    tel_ptr = ctypes.byref(tel) if tel is not None else None

    rc = lib.spft_contract(
        A_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(M),
        ctypes.c_size_t(K),
        B_arr.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(N),
        ctypes.byref(spec),
        ctypes.byref(opts),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        tel_ptr,
    )
    _check(rc, "spft_contract")

    if want_telemetry:
        return out, tel  # type: ignore[return-value]
    return out


def low_rank_apply(
    U: npt.ArrayLike,
    lam: npt.ArrayLike,
    Vh: npt.ArrayLike,
    x: npt.ArrayLike,
    spec: CReducerSpec,
    *,
    want_telemetry: bool = False,
) -> np.ndarray | tuple[np.ndarray, CTelemetry]:
    """Call ``spft_low_rank_apply``; return y = U @ (diag(lambda) @ (Vh @ x))."""
    lib = load_library()
    U_arr = np.ascontiguousarray(U, dtype=np.uint32)
    lam_arr = np.ascontiguousarray(lam, dtype=np.uint32).reshape(-1)
    Vh_arr = np.ascontiguousarray(Vh, dtype=np.uint32)
    x_arr = np.ascontiguousarray(x, dtype=np.uint32).reshape(-1)
    if U_arr.ndim != 2 or Vh_arr.ndim != 2:
        raise ValueError("U and Vh must be 2-D")
    m, r = U_arr.shape
    r2, n = Vh_arr.shape
    if r != r2 or r != lam_arr.size or n != x_arr.size:
        raise ValueError("Shape mismatch in low_rank_apply")
    y = np.empty(m, dtype=np.uint32)
    tel = CTelemetry() if want_telemetry else None
    tel_ptr = ctypes.byref(tel) if tel is not None else None

    rc = lib.spft_low_rank_apply(
        U_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        lam_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        Vh_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(m),
        ctypes.c_size_t(r),
        ctypes.c_size_t(n),
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.byref(spec),
        y.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        tel_ptr,
    )
    _check(rc, "spft_low_rank_apply")
    if want_telemetry:
        return y, tel  # type: ignore[return-value]
    return y


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def unpack_planes(
    words: npt.ArrayLike,
) -> tuple[np.ndarray, np.ndarray]:
    """Call ``spft_unpack_planes``; return (L int16, T uint16) planes."""
    lib = load_library()
    w_arr = np.ascontiguousarray(words, dtype=np.uint32).reshape(-1)
    n = w_arr.size
    L_out = np.empty(n, dtype=np.int16)
    T_out = np.empty(n, dtype=np.uint16)
    lib.spft_unpack_planes(
        w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        L_out.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        T_out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_size_t(n),
    )
    return L_out, T_out


def pack_planes(
    L: npt.ArrayLike,
    T: npt.ArrayLike,
) -> np.ndarray:
    """Call ``spft_pack_planes``; return packed uint32 words."""
    lib = load_library()
    L_arr = np.ascontiguousarray(L, dtype=np.int16).reshape(-1)
    T_arr = np.ascontiguousarray(T, dtype=np.uint16).reshape(-1)
    if L_arr.size != T_arr.size:
        raise ValueError("L and T must have the same number of elements")
    n = L_arr.size
    out = np.empty(n, dtype=np.uint32)
    lib.spft_pack_planes(
        L_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
        T_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n),
    )
    return out


def decode_real_f32(words: npt.ArrayLike) -> np.ndarray:
    """Call ``spft_decode_real_f32``; return float32 real values."""
    lib = load_library()
    w_arr = np.ascontiguousarray(words, dtype=np.uint32).reshape(-1)
    out = np.empty(w_arr.size, dtype=np.float32)
    lib.spft_decode_real_f32(
        w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_size_t(w_arr.size),
    )
    return out


def encode_real_f32(x: npt.ArrayLike) -> np.ndarray:
    """Call ``spft_encode_real_f32``; return packed uint32 words."""
    lib = load_library()
    x_arr = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    out = np.empty(x_arr.size, dtype=np.uint32)
    lib.spft_encode_real_f32(
        x_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(x_arr.size),
    )
    return out


def phase_stats(H_mem: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Call ``spft_phase_stats``; return (mu, sd) float32 over the memory rows."""
    lib = load_library()
    Hm = np.ascontiguousarray(H_mem, dtype=np.float32)
    if Hm.ndim != 2:
        raise ValueError("H_mem must be 2-D (M, d)")
    M, d = Hm.shape
    mu = np.empty(d, dtype=np.float32)
    sd = np.empty(d, dtype=np.float32)
    rc = lib.spft_phase_stats(
        _f32p(Hm), ctypes.c_size_t(M), ctypes.c_size_t(d), _f32p(mu), _f32p(sd)
    )
    _check(rc, "spft_phase_stats")
    return mu, sd


def phase_features(
    H: npt.ArrayLike,
    mu: npt.ArrayLike,
    sd: npt.ArrayLike,
) -> np.ndarray:
    """Call ``spft_phase_features``; whiten + per-token L2 normalize."""
    lib = load_library()
    H_arr = np.ascontiguousarray(H, dtype=np.float32)
    if H_arr.ndim != 2:
        raise ValueError("H must be 2-D (T, d)")
    T, d = H_arr.shape
    mu_arr = np.ascontiguousarray(mu, dtype=np.float32).reshape(-1)
    sd_arr = np.ascontiguousarray(sd, dtype=np.float32).reshape(-1)
    if mu_arr.size != d or sd_arr.size != d:
        raise ValueError("mu / sd must have d elements")
    Q = np.empty((T, d), dtype=np.float32)
    rc = lib.spft_phase_features(
        _f32p(H_arr), ctypes.c_size_t(T), ctypes.c_size_t(d),
        _f32p(mu_arr), _f32p(sd_arr), _f32p(Q),
    )
    _check(rc, "spft_phase_features")
    return Q


def phase_route(
    H: npt.ArrayLike,
    H_mem: npt.ArrayLike,
    H_mem_out: npt.ArrayLike,
    spec: CReducerSpec,
    *,
    beta: float = 8.0,
    l_frac_bits: int = 16,
    telemetry: CTelemetry | None = None,
) -> np.ndarray:
    """Call ``spft_phase_route``: H_out = H + softmax(beta*Q.K^T) @ (Hmem_out - Hmem).

    Full-memory phase-keyed associative routing (PHASE_ROUTE_SPEC items 1+5).
    No rank/prototype compression by design (cl-0180/cl-0181).
    """
    lib = load_library()
    H_arr = np.ascontiguousarray(H, dtype=np.float32)
    Hm = np.ascontiguousarray(H_mem, dtype=np.float32)
    Hmo = np.ascontiguousarray(H_mem_out, dtype=np.float32)
    if H_arr.ndim != 2 or Hm.ndim != 2 or Hmo.shape != Hm.shape:
        raise ValueError("H (T,d), H_mem (M,d), H_mem_out (M,d) required")
    T, d = H_arr.shape
    M = Hm.shape[0]
    if Hm.shape[1] != d:
        raise ValueError("H and H_mem channel dims differ")
    opts = CPhaseRouteOpts(
        beta=float(beta), l_frac_bits=int(l_frac_bits), causal=0
    )
    out = np.empty((T, d), dtype=np.float32)
    rc = lib.spft_phase_route(
        _f32p(H_arr), ctypes.c_size_t(T), ctypes.c_size_t(d),
        _f32p(Hm), _f32p(Hmo), ctypes.c_size_t(M),
        ctypes.byref(spec), ctypes.byref(opts), _f32p(out),
        ctypes.byref(telemetry) if telemetry is not None else None,
    )
    _check(rc, "spft_phase_route")
    return out


def attention_route(
    q_words: npt.ArrayLike,
    k_words: npt.ArrayLike,
    v_words: npt.ArrayLike,
    spec: CReducerSpec,
    *,
    beta: float = 8.0,
    l_frac_bits: int = 16,
    causal: bool = False,
    telemetry: CTelemetry | None = None,
) -> np.ndarray:
    """Call ``spft_attention_route``: the fused phase-native route kernel.

    Words in, words out, no decode inside — the C mirror of
    ``spf.tensor.route_op.execute_route``. q/k are T×P / M×P phasor words
    (only Θ planes read); v is M×d_v full words. Full memory by design
    (no rank/prototype parameters, cl-0180/cl-0181).
    """
    lib = load_library()
    q = np.ascontiguousarray(q_words, dtype=np.uint32)
    k = np.ascontiguousarray(k_words, dtype=np.uint32)
    v = np.ascontiguousarray(v_words, dtype=np.uint32)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q (T,P), k (M,P), v (M,d_v) required")
    T, P = q.shape
    M, d_v = v.shape
    if k.shape != (M, P):
        raise ValueError(f"k shape {k.shape} != (M={M}, P={P})")
    opts = CPhaseRouteOpts(
        beta=float(beta), l_frac_bits=int(l_frac_bits), causal=int(causal)
    )
    out = np.empty((T, d_v), dtype=np.uint32)
    rc = lib.spft_attention_route(
        _u32p(q), ctypes.c_size_t(T), ctypes.c_size_t(P),
        _u32p(k), _u32p(v), ctypes.c_size_t(M), ctypes.c_size_t(d_v),
        ctypes.byref(spec), ctypes.byref(opts), _u32p(out),
        ctypes.byref(telemetry) if telemetry is not None else None,
    )
    _check(rc, "spft_attention_route")
    return out
