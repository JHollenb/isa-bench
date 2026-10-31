"""ctypes bridge to ``libspf32_planes``: the v0.2 typed plane kernels in C.

The C side (``native/src/scalar/spf32/planes/*.c``, header
``native/include/spf32_planes.h``) receives split ``int16 L`` / ``int8 sign`` /
``uint16 Theta`` / ``uint32 word`` planes -- the AlgRef §21.1 structure of
arrays -- and executes the §6 log-add/log-sub ROM arithmetic, the §5.7 log
products, §20 exposure/gamma, the §7.5 hybrid pair adder, the §9.2-9.4 pivot
accumulators, the §18 transport monoid and §13.3 jet generation without ever
forming a Cartesian plane.  Every kernel is bit-identical to the corresponding
:mod:`spf.tensor.v02_planes` operation on the same words;
``tests/v02/test_v02_planes_native_parity.py`` is the gate.

This module owns only the NumPy buffers and the ABI.  The library is optional
and is never compiled inside a timed call: build it with ``make -C native``
(CMake target ``spf32_planes``).  ``SPF32_PLANES_LIBRARY`` overrides the path
and ``SPF_PLANES_DISABLE=1`` forces the Python plane path (what the oracle
tests use).
"""

from __future__ import annotations

import ctypes
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

__all__ = [
    "ABI_VERSION",
    "PRECISION",
    "Reduction",
    "available",
    "integer_only",
    "jet_emit",
    "library",
    "library_path",
    "log1m",
    "log1p",
    "logadd_pos",
    "logadd_signed",
    "logpos_div",
    "logpos_exposure",
    "logpos_mul",
    "logpos_pow_int",
    "logpos_power",
    "logpos_reinhard",
    "logsigned_mul",
    "logsub_pos",
    "pair_addsub",
    "pair_addsub_diagnostics",
    "phase_compose",
    "reduce_complex",
    "reduce_logadd_pos",
    "reduce_logadd_signed",
    "reduce_logpos",
    "reduce_logsigned",
    "transport_compose",
    "transport_reduce",
    "transport_scan",
    "words_abs2",
    "words_conj",
    "words_div",
    "words_mul",
    "words_neg",
    "words_phase_shift",
]

ABI_VERSION = 1
PRECISION: dict[str, int] = {"fast": 0, "balanced": 1, "accurate": 2, "exact": 3}

_I16P = ctypes.POINTER(ctypes.c_int16)
_I8P = ctypes.POINTER(ctypes.c_int8)
_U8P = ctypes.POINTER(ctypes.c_uint8)
_U16P = ctypes.POINTER(ctypes.c_uint16)
_U32P = ctypes.POINTER(ctypes.c_uint32)
_I32P = ctypes.POINTER(ctypes.c_int32)
_I64P = ctypes.POINTER(ctypes.c_int64)
_F64P = ctypes.POINTER(ctypes.c_double)
_SIZE = ctypes.c_size_t


class Reduction(ctypes.Structure):
    """Mirror of ``spf32_planes_reduction_t``: one output row per element."""

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("value_L", _I16P),
        ("value_sign", _I8P),
        ("value_word", _U32P),
        ("absolute", _F64P),
        ("relative", _F64P),
        ("phase_error", _F64P),
        ("phase_reliable", _U8P),
        ("retained", _I64P),
        ("dropped", _I64P),
        ("bits", _I32P),
        ("saturated", _U8P),
        ("has_nan", _U8P),
    ]


def _suffix() -> str:
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform == "win32":
        return "dll"
    return "so"


def library_path() -> Path:
    explicit = os.environ.get("SPF32_PLANES_LIBRARY")
    if explicit:
        return Path(explicit)
    root = Path(__file__).resolve().parents[3]
    return root / "native" / f"libspf32_planes.{_suffix()}"


def _configure(lib: Any) -> Any:
    lib.spf32_planes_abi_version.argtypes = []
    lib.spf32_planes_abi_version.restype = ctypes.c_int
    lib.spf32_planes_integer_only.argtypes = []
    lib.spf32_planes_integer_only.restype = ctypes.c_int
    lib.spf32_planes_reduction_struct_size.argtypes = []
    lib.spf32_planes_reduction_struct_size.restype = ctypes.c_int
    if int(lib.spf32_planes_abi_version()) != ABI_VERSION:
        raise RuntimeError("plane kernel library ABI mismatch; rebuild native/ (make -C native)")
    if int(lib.spf32_planes_reduction_struct_size()) != ctypes.sizeof(Reduction):
        raise RuntimeError("plane reduction record mismatch; rebuild native/")
    binary = [_I16P, _I16P, _I16P, _SIZE]
    for name in (
        "spf32_planes_logadd_pos",
        "spf32_planes_logsub_pos",
        "spf32_planes_logpos_mul",
        "spf32_planes_logpos_div",
    ):
        getattr(lib, name).argtypes = list(binary)
    for name in ("spf32_planes_log1m", "spf32_planes_log1p", "spf32_planes_logpos_reinhard"):
        getattr(lib, name).argtypes = [_I16P, _I16P, _SIZE]
    lib.spf32_planes_logpos_exposure.argtypes = [_I16P, ctypes.c_int32, _I16P, _SIZE]
    lib.spf32_planes_logpos_pow_int.argtypes = [_I16P, ctypes.c_int32, _I16P, _SIZE]
    lib.spf32_planes_logpos_power.argtypes = [_I16P, ctypes.c_double, _I16P, _SIZE]
    for name in ("spf32_planes_logadd_signed", "spf32_planes_logsigned_mul"):
        getattr(lib, name).argtypes = [_I16P, _I8P, _I16P, _I8P, _I16P, _I8P, _SIZE]
    for name in ("spf32_planes_words_mul", "spf32_planes_words_div"):
        getattr(lib, name).argtypes = [_U32P, _U32P, _U32P, _SIZE, ctypes.c_int]
    for name in ("spf32_planes_words_conj", "spf32_planes_words_neg"):
        getattr(lib, name).argtypes = [_U32P, _U32P, _SIZE, ctypes.c_int]
    lib.spf32_planes_words_phase_shift.argtypes = [_U32P, _I32P, _SIZE, _U32P, _SIZE, ctypes.c_int]
    lib.spf32_planes_words_abs2.argtypes = [_U32P, _I16P, _SIZE]
    lib.spf32_planes_phase_compose.argtypes = [_U16P, _U16P, _U16P, _SIZE]
    lib.spf32_planes_transport_compose.argtypes = [_I16P] * 6 + [_SIZE]
    lib.spf32_planes_transport_reduce.argtypes = [_I16P, _I16P, _SIZE, _I16P, _I16P]
    lib.spf32_planes_transport_scan.argtypes = [_I16P, _I16P, _SIZE, _I16P, _I16P]
    lib.spf32_planes_reduce_logadd_pos.argtypes = [_I16P, _SIZE, _SIZE, _I16P]
    lib.spf32_planes_reduce_logadd_signed.argtypes = [_I16P, _I8P, _SIZE, _SIZE, _I16P, _I8P]
    reduction = ctypes.POINTER(Reduction)
    lib.spf32_planes_reduce_logpos.argtypes = [
        _I16P, _SIZE, _SIZE, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, reduction,
    ]
    lib.spf32_planes_reduce_logsigned.argtypes = [
        _I16P, _I8P, _SIZE, _SIZE, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, reduction,
    ]
    lib.spf32_planes_reduce_complex.argtypes = [
        _U32P, _SIZE, _SIZE, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_double, ctypes.c_double, reduction,
    ]
    lib.spf32_planes_pair_addsub.argtypes = [_U32P, _U32P, _U32P, _U32P, _SIZE, ctypes.c_int, ctypes.c_int]
    lib.spf32_planes_pair_addsub_diagnostics.argtypes = [
        _U32P, _U32P, _U32P, _U32P, _U8P, _U8P, _F64P, _F64P, _U8P, _U8P, _U8P,
        _SIZE, ctypes.c_int, ctypes.c_int,
    ]
    lib.spf32_planes_jet_emit.argtypes = [
        _F64P, _F64P, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_double,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int, _F64P, _F64P, _U8P, _I16P, _U16P, _U32P,
    ]
    for name in (
        "spf32_planes_logadd_pos", "spf32_planes_logsub_pos", "spf32_planes_log1m", "spf32_planes_log1p",
        "spf32_planes_logadd_signed", "spf32_planes_logsigned_mul", "spf32_planes_logpos_mul",
        "spf32_planes_logpos_div", "spf32_planes_logpos_exposure", "spf32_planes_logpos_pow_int",
        "spf32_planes_logpos_reinhard", "spf32_planes_logpos_power", "spf32_planes_words_mul",
        "spf32_planes_words_div", "spf32_planes_words_conj", "spf32_planes_words_neg",
        "spf32_planes_words_phase_shift", "spf32_planes_words_abs2", "spf32_planes_phase_compose",
        "spf32_planes_transport_compose", "spf32_planes_transport_reduce", "spf32_planes_transport_scan",
        "spf32_planes_reduce_logadd_pos", "spf32_planes_reduce_logadd_signed",
        "spf32_planes_reduce_logpos", "spf32_planes_reduce_logsigned", "spf32_planes_reduce_complex",
        "spf32_planes_pair_addsub", "spf32_planes_pair_addsub_diagnostics", "spf32_planes_jet_emit",
    ):
        getattr(lib, name).restype = ctypes.c_int
    return lib


@lru_cache(maxsize=1)
def library() -> Any:
    path = library_path()
    if not path.is_file():
        return None
    return _configure(ctypes.CDLL(str(path)))


def available() -> bool:
    """True when the plane kernel library is loadable (never compiles)."""

    if os.environ.get("SPF_PLANES_DISABLE"):
        return False
    try:
        return library() is not None
    except OSError:
        return False


def integer_only() -> bool:
    """True when the integer kernels were compiled as the §37.5 profile."""

    return bool(int(_require().spf32_planes_integer_only()))


def _require() -> Any:
    lib = library()
    if lib is None:
        raise RuntimeError("native plane kernel library is unavailable; run make -C native")
    return lib


class PlaneDomainError(ValueError):
    """The kernel rejected its inputs on the v0.2 domain rule (e.g. ``logsub`` a < b)."""


class PlaneUnsupportedError(RuntimeError):
    """The kernel cannot reproduce the Python path for these inputs (widen in Python)."""


def _raise(status: int, *, stage: str) -> None:
    if status == 0:
        return
    if status == -3:
        raise PlaneDomainError(f"plane {stage} rejected its inputs on the v0.2 domain rule")
    if status == -4:
        raise PlaneUnsupportedError(f"plane {stage} needs the widened Python accumulator")
    reason = {-1: "rejected its arguments", -2: "could not allocate scratch"}
    raise RuntimeError(f"plane {stage} {reason.get(status, f'failed with status {status}')}")


def _codes(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.int16))
    if array.dtype != np.int16:  # pragma: no cover - defensive
        raise TypeError(f"{name} must be int16 log codes")
    return array


def _signs(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(values, dtype=np.int8))


def _words(values: npt.ArrayLike, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be packed SPF32 words")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _ptr(array: np.ndarray | None, pointer: Any) -> Any:
    return None if array is None else array.ctypes.data_as(pointer)


# ---------------------------------------------------------------------------
# §6 LOG_POS / LOG_SIGNED plane arithmetic
# ---------------------------------------------------------------------------


def _binary_codes(name: str, a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    lib = _require()
    left = _codes(a, name="a")
    right = _codes(b, name="b")
    if left.shape != right.shape:
        left, right = (np.ascontiguousarray(x) for x in np.broadcast_arrays(left, right))
    out = np.empty(left.shape, dtype=np.int16)
    _raise(
        getattr(lib, name)(_ptr(left, _I16P), _ptr(right, _I16P), _ptr(out, _I16P), _SIZE(out.size)),
        stage=name,
    )
    return out


def _unary_codes(name: str, a: npt.ArrayLike) -> np.ndarray:
    lib = _require()
    left = _codes(a, name="a")
    out = np.empty(left.shape, dtype=np.int16)
    _raise(getattr(lib, name)(_ptr(left, _I16P), _ptr(out, _I16P), _SIZE(out.size)), stage=name)
    return out


def logadd_pos(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    return _binary_codes("spf32_planes_logadd_pos", a, b)


def logsub_pos(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    return _binary_codes("spf32_planes_logsub_pos", a, b)


def logpos_mul(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    return _binary_codes("spf32_planes_logpos_mul", a, b)


def logpos_div(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    return _binary_codes("spf32_planes_logpos_div", a, b)


def log1m(a: npt.ArrayLike) -> np.ndarray:
    return _unary_codes("spf32_planes_log1m", a)


def log1p(a: npt.ArrayLike) -> np.ndarray:
    return _unary_codes("spf32_planes_log1p", a)


def logpos_reinhard(a: npt.ArrayLike) -> np.ndarray:
    return _unary_codes("spf32_planes_logpos_reinhard", a)


def logpos_exposure(a: npt.ArrayLike, shift: int) -> np.ndarray:
    lib = _require()
    left = _codes(a, name="a")
    out = np.empty(left.shape, dtype=np.int16)
    _raise(
        lib.spf32_planes_logpos_exposure(
            _ptr(left, _I16P), ctypes.c_int32(int(shift)), _ptr(out, _I16P), _SIZE(out.size)
        ),
        stage="exposure",
    )
    return out


def logpos_pow_int(a: npt.ArrayLike, exponent: int) -> np.ndarray:
    lib = _require()
    left = _codes(a, name="a")
    out = np.empty(left.shape, dtype=np.int16)
    _raise(
        lib.spf32_planes_logpos_pow_int(
            _ptr(left, _I16P), ctypes.c_int32(int(exponent)), _ptr(out, _I16P), _SIZE(out.size)
        ),
        stage="pow_int",
    )
    return out


def logpos_power(a: npt.ArrayLike, exponent: float) -> np.ndarray:
    lib = _require()
    left = _codes(a, name="a")
    out = np.empty(left.shape, dtype=np.int16)
    _raise(
        lib.spf32_planes_logpos_power(
            _ptr(left, _I16P), ctypes.c_double(float(exponent)), _ptr(out, _I16P), _SIZE(out.size)
        ),
        stage="power",
    )
    return out


def _binary_signed(
    name: str, a_L: npt.ArrayLike, a_sign: npt.ArrayLike, b_L: npt.ArrayLike, b_sign: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    lib = _require()
    left, left_sign = _codes(a_L, name="a_L"), _signs(a_sign, name="a_sign")
    right, right_sign = _codes(b_L, name="b_L"), _signs(b_sign, name="b_sign")
    shape = np.broadcast_shapes(left.shape, left_sign.shape, right.shape, right_sign.shape)
    if (left.shape, left_sign.shape, right.shape, right_sign.shape) != (shape,) * 4:
        left = np.ascontiguousarray(np.broadcast_to(left, shape))
        left_sign = np.ascontiguousarray(np.broadcast_to(left_sign, shape))
        right = np.ascontiguousarray(np.broadcast_to(right, shape))
        right_sign = np.ascontiguousarray(np.broadcast_to(right_sign, shape))
    out_L = np.empty(left.shape, dtype=np.int16)
    out_sign = np.empty(left.shape, dtype=np.int8)
    _raise(
        getattr(lib, name)(
            _ptr(left, _I16P), _ptr(left_sign, _I8P), _ptr(right, _I16P), _ptr(right_sign, _I8P),
            _ptr(out_L, _I16P), _ptr(out_sign, _I8P), _SIZE(out_L.size),
        ),
        stage=name,
    )
    return out_L, out_sign


def logadd_signed(
    a_L: npt.ArrayLike, a_sign: npt.ArrayLike, b_L: npt.ArrayLike, b_sign: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    return _binary_signed("spf32_planes_logadd_signed", a_L, a_sign, b_L, b_sign)


def logsigned_mul(
    a_L: npt.ArrayLike, a_sign: npt.ArrayLike, b_L: npt.ArrayLike, b_sign: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    return _binary_signed("spf32_planes_logsigned_mul", a_L, a_sign, b_L, b_sign)


# ---------------------------------------------------------------------------
# SPF_COMPLEX word planes
# ---------------------------------------------------------------------------


def _binary_words(name: str, a: npt.ArrayLike, b: npt.ArrayLike, frozen: bool) -> np.ndarray:
    lib = _require()
    left, right = _words(a, name="a"), _words(b, name="b")
    if left.shape != right.shape:
        left, right = (np.ascontiguousarray(x) for x in np.broadcast_arrays(left, right))
    out = np.empty(left.shape, dtype=np.uint32)
    _raise(
        getattr(lib, name)(
            _ptr(left, _U32P), _ptr(right, _U32P), _ptr(out, _U32P), _SIZE(out.size), ctypes.c_int(int(frozen))
        ),
        stage=name,
    )
    return out


def words_mul(a: npt.ArrayLike, b: npt.ArrayLike, *, frozen: bool = False) -> np.ndarray:
    return _binary_words("spf32_planes_words_mul", a, b, frozen)


def words_div(a: npt.ArrayLike, b: npt.ArrayLike, *, frozen: bool = False) -> np.ndarray:
    return _binary_words("spf32_planes_words_div", a, b, frozen)


def _unary_words(name: str, a: npt.ArrayLike, frozen: bool) -> np.ndarray:
    lib = _require()
    left = _words(a, name="a")
    out = np.empty(left.shape, dtype=np.uint32)
    _raise(
        getattr(lib, name)(_ptr(left, _U32P), _ptr(out, _U32P), _SIZE(out.size), ctypes.c_int(int(frozen))),
        stage=name,
    )
    return out


def words_conj(a: npt.ArrayLike, *, frozen: bool = False) -> np.ndarray:
    return _unary_words("spf32_planes_words_conj", a, frozen)


def words_neg(a: npt.ArrayLike, *, frozen: bool = False) -> np.ndarray:
    return _unary_words("spf32_planes_words_neg", a, frozen)


def words_phase_shift(a: npt.ArrayLike, delta: npt.ArrayLike, *, frozen: bool = False) -> np.ndarray:
    lib = _require()
    left = _words(a, name="a")
    shift = np.asarray(delta, dtype=np.int64)
    if shift.ndim == 0:
        plane = np.ascontiguousarray(np.asarray([int(shift) & 0xFFFF], dtype=np.int32))
        stride = 0
    else:
        broadcast = np.broadcast_to(shift, left.shape)
        plane = np.ascontiguousarray((broadcast & 0xFFFF).astype(np.int32))
        stride = 1
    out = np.empty(left.shape, dtype=np.uint32)
    _raise(
        lib.spf32_planes_words_phase_shift(
            _ptr(left, _U32P), _ptr(plane, _I32P), _SIZE(stride), _ptr(out, _U32P), _SIZE(out.size),
            ctypes.c_int(int(frozen)),
        ),
        stage="phase shift",
    )
    return out


def words_abs2(a: npt.ArrayLike) -> np.ndarray:
    lib = _require()
    left = _words(a, name="a")
    out = np.empty(left.shape, dtype=np.int16)
    _raise(
        lib.spf32_planes_words_abs2(_ptr(left, _U32P), _ptr(out, _I16P), _SIZE(out.size)),
        stage="abs2",
    )
    return out


def phase_compose(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
    lib = _require()
    left = np.ascontiguousarray(np.asarray(a, dtype=np.uint16))
    right = np.ascontiguousarray(np.asarray(b, dtype=np.uint16))
    if left.shape != right.shape:
        left, right = (np.ascontiguousarray(x) for x in np.broadcast_arrays(left, right))
    out = np.empty(left.shape, dtype=np.uint16)
    _raise(
        lib.spf32_planes_phase_compose(_ptr(left, _U16P), _ptr(right, _U16P), _ptr(out, _U16P), _SIZE(out.size)),
        stage="phase compose",
    )
    return out


# ---------------------------------------------------------------------------
# §18 transport monoid
# ---------------------------------------------------------------------------


def transport_compose(
    T1: npt.ArrayLike, C1: npt.ArrayLike, T2: npt.ArrayLike, C2: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    lib = _require()
    planes = [_codes(x, name=name) for x, name in ((T1, "T1"), (C1, "C1"), (T2, "T2"), (C2, "C2"))]
    shape = np.broadcast_shapes(*(p.shape for p in planes))
    planes = [np.ascontiguousarray(np.broadcast_to(p, shape)) for p in planes]
    out_T = np.empty(shape, dtype=np.int16)
    out_C = np.empty(shape, dtype=np.int16)
    _raise(
        lib.spf32_planes_transport_compose(
            *(_ptr(p, _I16P) for p in planes), _ptr(out_T, _I16P), _ptr(out_C, _I16P), _SIZE(out_T.size)
        ),
        stage="transport compose",
    )
    return out_T, out_C


def transport_reduce(T: npt.ArrayLike, C: npt.ArrayLike) -> tuple[int, int]:
    """§18.2 order-preserving balanced tree over one segment sequence."""

    lib = _require()
    left, right = _codes(T, name="T"), _codes(C, name="C")
    out_T = np.empty((1,), dtype=np.int16)
    out_C = np.empty((1,), dtype=np.int16)
    _raise(
        lib.spf32_planes_transport_reduce(
            _ptr(left, _I16P), _ptr(right, _I16P), _SIZE(left.size), _ptr(out_T, _I16P), _ptr(out_C, _I16P)
        ),
        stage="transport reduce",
    )
    return int(out_T[0]), int(out_C[0])


def transport_scan(T: npt.ArrayLike, C: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive ordered prefix scan from the transport identity."""

    lib = _require()
    left, right = _codes(T, name="T"), _codes(C, name="C")
    prefix_T = np.empty(left.shape, dtype=np.int16)
    prefix_C = np.empty(left.shape, dtype=np.int16)
    _raise(
        lib.spf32_planes_transport_scan(
            _ptr(left, _I16P), _ptr(right, _I16P), _SIZE(left.size),
            _ptr(prefix_T, _I16P), _ptr(prefix_C, _I16P),
        ),
        stage="transport scan",
    )
    return prefix_T, prefix_C


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------


def reduce_logadd_pos(rows: npt.ArrayLike) -> np.ndarray:
    """§37.5 integer-only sequential ROM fold over the last axis."""

    lib = _require()
    array = _codes(rows, name="rows")
    array = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
    out = np.empty(array.shape[0], dtype=np.int16)
    _raise(
        lib.spf32_planes_reduce_logadd_pos(
            _ptr(array, _I16P), _SIZE(array.shape[0]), _SIZE(array.shape[1]), _ptr(out, _I16P)
        ),
        stage="reduce logadd",
    )
    return out


def reduce_logadd_signed(rows_L: npt.ArrayLike, rows_sign: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    lib = _require()
    array = _codes(rows_L, name="rows_L")
    signs = _signs(rows_sign, name="rows_sign")
    array = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
    signs = signs.reshape(array.shape)
    out_L = np.empty(array.shape[0], dtype=np.int16)
    out_sign = np.empty(array.shape[0], dtype=np.int8)
    _raise(
        lib.spf32_planes_reduce_logadd_signed(
            _ptr(array, _I16P), _ptr(signs, _I8P), _SIZE(array.shape[0]), _SIZE(array.shape[1]),
            _ptr(out_L, _I16P), _ptr(out_sign, _I8P),
        ),
        stage="reduce logadd signed",
    )
    return out_L, out_sign


def _reduction_buffers(count: int, *, kind: str) -> tuple[Reduction, dict[str, np.ndarray]]:
    buffers: dict[str, np.ndarray] = {
        "absolute": np.empty(count, dtype=np.float64),
        "relative": np.empty(count, dtype=np.float64),
        "phase_error": np.empty(count, dtype=np.float64),
        "phase_reliable": np.empty(count, dtype=np.uint8),
        "retained": np.empty(count, dtype=np.int64),
        "dropped": np.empty(count, dtype=np.int64),
        "bits": np.empty(count, dtype=np.int32),
        "saturated": np.empty(count, dtype=np.uint8),
        "has_nan": np.empty(count, dtype=np.uint8),
    }
    if kind == "SPF_COMPLEX":
        buffers["value_word"] = np.empty(count, dtype=np.uint32)
    else:
        buffers["value_L"] = np.empty(count, dtype=np.int16)
        if kind == "LOG_SIGNED":
            buffers["value_sign"] = np.empty(count, dtype=np.int8)
    record = Reduction()
    pointers = {
        "value_L": _I16P,
        "value_sign": _I8P,
        "value_word": _U32P,
        "absolute": _F64P,
        "relative": _F64P,
        "phase_error": _F64P,
        "phase_reliable": _U8P,
        "retained": _I64P,
        "dropped": _I64P,
        "bits": _I32P,
        "saturated": _U8P,
        "has_nan": _U8P,
    }
    for name, pointer in pointers.items():
        setattr(record, name, _ptr(buffers.get(name), pointer))
    return record, buffers


def _spec_arguments(
    precision: str | int, explicit_bits: int | None, accumulator_fraction_bits: int | None,
    budget_abs: float, budget_rel: float,
) -> tuple[int, int, int, float, float]:
    mode = PRECISION.get(str(precision).lower(), 1) if not isinstance(precision, int) else int(precision)
    return (
        mode,
        -1 if explicit_bits is None else int(explicit_bits),
        -1 if accumulator_fraction_bits is None else int(accumulator_fraction_bits),
        float(budget_abs),
        float(budget_rel),
    )


def _reduce(
    name: str, rows: np.ndarray, signs: np.ndarray | None, *, kind: str,
    precision: str | int, explicit_bits: int | None, accumulator_fraction_bits: int | None,
    budget_abs: float, budget_rel: float,
) -> dict[str, np.ndarray]:
    lib = _require()
    count, n = rows.shape
    record, buffers = _reduction_buffers(count, kind=kind)
    mode, explicit, accumulator, abs_budget, rel_budget = _spec_arguments(
        precision, explicit_bits, accumulator_fraction_bits, budget_abs, budget_rel
    )
    pointer = _U32P if kind == "SPF_COMPLEX" else _I16P
    arguments: list[Any] = [_ptr(rows, pointer)]
    if signs is not None:
        arguments.append(_ptr(signs, _I8P))
    arguments += [
        _SIZE(count), _SIZE(n), ctypes.c_int(mode), ctypes.c_int(explicit), ctypes.c_int(accumulator),
        ctypes.c_double(abs_budget), ctypes.c_double(rel_budget), ctypes.byref(record),
    ]
    _raise(getattr(lib, name)(*arguments), stage=name)
    return buffers


def reduce_logpos(
    rows: npt.ArrayLike, *, precision: str | int = "balanced", explicit_bits: int | None = None,
    accumulator_fraction_bits: int | None = None, budget_abs: float = 0.0, budget_rel: float = 0.0,
) -> dict[str, np.ndarray]:
    array = _codes(rows, name="rows")
    array = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
    return _reduce(
        "spf32_planes_reduce_logpos", array, None, kind="LOG_POS", precision=precision,
        explicit_bits=explicit_bits, accumulator_fraction_bits=accumulator_fraction_bits,
        budget_abs=budget_abs, budget_rel=budget_rel,
    )


def reduce_logsigned(
    rows_L: npt.ArrayLike, rows_sign: npt.ArrayLike, *, precision: str | int = "balanced",
    explicit_bits: int | None = None, accumulator_fraction_bits: int | None = None,
    budget_abs: float = 0.0, budget_rel: float = 0.0,
) -> dict[str, np.ndarray]:
    array = _codes(rows_L, name="rows_L")
    array = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
    signs = _signs(rows_sign, name="rows_sign").reshape(array.shape)
    return _reduce(
        "spf32_planes_reduce_logsigned", array, signs, kind="LOG_SIGNED", precision=precision,
        explicit_bits=explicit_bits, accumulator_fraction_bits=accumulator_fraction_bits,
        budget_abs=budget_abs, budget_rel=budget_rel,
    )


def reduce_complex(
    rows: npt.ArrayLike, *, precision: str | int = "balanced", explicit_bits: int | None = None,
    accumulator_fraction_bits: int | None = None, budget_abs: float = 0.0, budget_rel: float = 0.0,
) -> dict[str, np.ndarray]:
    array = _words(rows, name="rows")
    array = array.reshape(1, -1) if array.ndim == 1 else array.reshape(-1, array.shape[-1])
    return _reduce(
        "spf32_planes_reduce_complex", array, None, kind="SPF_COMPLEX", precision=precision,
        explicit_bits=explicit_bits, accumulator_fraction_bits=accumulator_fraction_bits,
        budget_abs=budget_abs, budget_rel=budget_rel,
    )


# ---------------------------------------------------------------------------
# §7.5 hybrid pair adder
# ---------------------------------------------------------------------------


def pair_addsub(
    u: npt.ArrayLike, t: npt.ArrayLike, precision: str | int = "balanced", *, threads: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    lib = _require()
    left, right = _words(u, name="u"), _words(t, name="t")
    if left.shape != right.shape:
        left, right = (np.ascontiguousarray(x) for x in np.broadcast_arrays(left, right))
    mode = PRECISION.get(str(precision).lower(), 1) if not isinstance(precision, int) else int(precision)
    plus = np.empty(left.shape, dtype=np.uint32)
    minus = np.empty(left.shape, dtype=np.uint32)
    _raise(
        lib.spf32_planes_pair_addsub(
            _ptr(left, _U32P), _ptr(right, _U32P), _ptr(plus, _U32P), _ptr(minus, _U32P),
            _SIZE(plus.size), ctypes.c_int(mode), ctypes.c_int(int(threads)),
        ),
        stage="pair addsub",
    )
    return plus, minus


def pair_addsub_diagnostics(
    u: npt.ArrayLike, t: npt.ArrayLike, precision: str | int = "balanced", *, threads: int = 0
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    lib = _require()
    left, right = _words(u, name="u"), _words(t, name="t")
    if left.shape != right.shape:
        left, right = (np.ascontiguousarray(x) for x in np.broadcast_arrays(left, right))
    mode = PRECISION.get(str(precision).lower(), 1) if not isinstance(precision, int) else int(precision)
    plus = np.empty(left.shape, dtype=np.uint32)
    minus = np.empty(left.shape, dtype=np.uint32)
    flags: dict[str, np.ndarray] = {
        "plus_reliable": np.empty(left.shape, dtype=np.uint8),
        "minus_reliable": np.empty(left.shape, dtype=np.uint8),
        "plus_error": np.empty(left.shape, dtype=np.float64),
        "minus_error": np.empty(left.shape, dtype=np.float64),
        "plus_cancelled": np.empty(left.shape, dtype=np.uint8),
        "minus_cancelled": np.empty(left.shape, dtype=np.uint8),
        "stable_branch": np.empty(left.shape, dtype=np.uint8),
    }
    _raise(
        lib.spf32_planes_pair_addsub_diagnostics(
            _ptr(left, _U32P), _ptr(right, _U32P), _ptr(plus, _U32P), _ptr(minus, _U32P),
            _ptr(flags["plus_reliable"], _U8P), _ptr(flags["minus_reliable"], _U8P),
            _ptr(flags["plus_error"], _F64P), _ptr(flags["minus_error"], _F64P),
            _ptr(flags["plus_cancelled"], _U8P), _ptr(flags["minus_cancelled"], _U8P),
            _ptr(flags["stable_branch"], _U8P),
            _SIZE(plus.size), ctypes.c_int(mode), ctypes.c_int(int(threads)),
        ),
        stage="pair diagnostics",
    )
    return plus, minus, flags


# ---------------------------------------------------------------------------
# §13.3 jet generation
# ---------------------------------------------------------------------------


def jet_emit(
    log_coefficients: npt.ArrayLike,
    phase_coefficients: npt.ArrayLike,
    shape: tuple[int, int],
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    guard_L: int,
    guard_T: int,
    phase_modulo: int = 0,
    log_residual: npt.ArrayLike | None = None,
    phase_residual: npt.ArrayLike | None = None,
    zero_mask: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-difference materialization plus emission quantization for one tile."""

    lib = _require()
    log_c = np.ascontiguousarray(np.asarray(log_coefficients, dtype=np.float64).reshape(-1))
    phase_c = np.ascontiguousarray(np.asarray(phase_coefficients, dtype=np.float64).reshape(-1))
    if log_c.size != 6 or phase_c.size != 6:
        raise ValueError("jet emission needs six coefficients per jet")
    height, width = (int(value) for value in shape)
    out_L = np.empty((height, width), dtype=np.int16)
    out_T = np.empty((height, width), dtype=np.uint16)
    words = np.empty((height, width), dtype=np.uint32)
    residual_log = None if log_residual is None else np.ascontiguousarray(
        np.asarray(log_residual, dtype=np.float64)
    )
    residual_phase = None if phase_residual is None else np.ascontiguousarray(
        np.asarray(phase_residual, dtype=np.float64)
    )
    mask = None if zero_mask is None else np.ascontiguousarray(np.asarray(zero_mask, dtype=bool).view(np.uint8))
    _raise(
        lib.spf32_planes_jet_emit(
            _ptr(log_c, _F64P), _ptr(phase_c, _F64P), ctypes.c_int(height), ctypes.c_int(width),
            ctypes.c_double(float(origin[0])), ctypes.c_double(float(origin[1])),
            ctypes.c_int64(int(phase_modulo)), ctypes.c_int(int(guard_L)), ctypes.c_int(int(guard_T)),
            _ptr(residual_log, _F64P), _ptr(residual_phase, _F64P), _ptr(mask, _U8P),
            _ptr(out_L, _I16P), _ptr(out_T, _U16P), _ptr(words, _U32P),
        ),
        stage="jet emit",
    )
    return out_L, out_T, words
