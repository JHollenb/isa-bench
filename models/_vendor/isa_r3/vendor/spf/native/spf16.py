"""C-backed SPF-16 scalar ops.

Mirrors the public surface of :mod:`spf.scalar.spf16` for the ops exposed
by ``libspf16``. Pure-Python code should prefer that module; this one
exists so benchmarks and cross-validation tests can exercise the
compiled C path. Raises at call time if ``libspf16`` isn't available —
import alone is safe.

ABI 6 surface: integer ops (mul, div, conj, abs2, neg, theta_sub) plus
Sb-dependent ops (add, sub, butterfly). encode/decode stay in Python.

Library resolution (in order):

1. ``$SPF16_LIBRARY`` environment variable — absolute path to the .so/.dylib.
2. ``native/libspf16.{dylib,so}`` relative to the repo root.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "LIBSPF16_AVAILABLE",
    "LIBSPF16_PATH",
    "abs2",
    "add",
    "butterfly",
    "conj",
    "div",
    "load_library",
    "mul",
    "neg",
    "phase8_bind",
    "phase8_conj",
    "phase8_similarity",
    "phase8_similarity_i64",
    "phase8_unbind",
    "phase_bind",
    "phase_conj",
    "phase_similarity",
    "phase_similarity_i64",
    "phase_unbind",
    "sub",
    "theta_sub",
]

_EXPECTED_ABI_VERSION = 6


def _candidate_paths() -> list[Path]:
    """Return candidate library paths in search order."""
    candidates: list[Path] = []
    env = os.environ.get("SPF16_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())

    # src/spf/native/spf16.py → src/spf/native → src/spf → src → repo_root
    repo_root = Path(__file__).resolve().parents[3]
    ext = "dylib" if sys.platform == "darwin" else "so"
    candidates.append(repo_root / "native" / f"libspf16.{ext}")
    return candidates


def _resolve() -> tuple[ctypes.CDLL | None, Path | None]:
    for path in _candidate_paths():
        if not path.is_file():
            continue
        try:
            handle = ctypes.CDLL(str(path))
        except OSError:
            continue
        return handle, path
    return None, None


def _register(lib: ctypes.CDLL) -> None:
    """Declare argtypes / restype for every exported libspf16 symbol.

    A missing registration here surfaces as a TypeError at call time, not
    a silent misread. Mirrored byte-for-byte by the duplicate registration
    in ``hw/platforms/kria/board/agent.py:_load_lib_spf16`` per the
    CLAUDE.md ABI parity rule.
    """
    lib.spf16_abi_version.argtypes = []
    lib.spf16_abi_version.restype = ctypes.c_int32

    lib.spf16_simd_active.argtypes = []
    lib.spf16_simd_active.restype = ctypes.c_int32

    for _name in ("spf16_const_l_scale", "spf16_const_theta_scale"):
        getattr(lib, _name).argtypes = []
        getattr(lib, _name).restype = ctypes.c_int32
    for _name in ("spf16_const_word_zero", "spf16_const_word_nan"):
        getattr(lib, _name).argtypes = []
        getattr(lib, _name).restype = ctypes.c_uint16

    # Scalar ops — closed on the integer word.
    for _name in ("spf16_mul", "spf16_div", "spf16_add", "spf16_sub"):
        getattr(lib, _name).argtypes = [ctypes.c_uint16, ctypes.c_uint16]
        getattr(lib, _name).restype = ctypes.c_uint16
    for _name in ("spf16_conj", "spf16_abs2", "spf16_neg"):
        getattr(lib, _name).argtypes = [ctypes.c_uint16]
        getattr(lib, _name).restype = ctypes.c_uint16
    lib.spf16_butterfly.argtypes = [
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.spf16_butterfly.restype = None

    # Vectorized array ops. All return void; per-element loop runs in C.
    for _name in ("spf16_conj_array", "spf16_abs2_array", "spf16_neg_array"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    for _name in (
        "spf16_mul_array",
        "spf16_div_array",
        "spf16_add_array",
        "spf16_sub_array",
        "spf16_theta_sub_array",
    ):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    lib.spf16_butterfly_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf16_butterfly_array.restype = None
    for _name in ("spf16_phase_bind_array", "spf16_phase_unbind_array"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    lib.spf16_phase_conj_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf16_phase_conj_array.restype = None
    lib.spf16_phase_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf16_phase_similarity.restype = ctypes.c_int32
    lib.spf16_phase_similarity_i64.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf16_phase_similarity_i64.restype = ctypes.c_int64
    for _name in ("spf8_phase_bind_array", "spf8_phase_unbind_array"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    lib.spf8_phase_conj_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf8_phase_conj_array.restype = None
    lib.spf8_phase_similarity.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf8_phase_similarity.restype = ctypes.c_int32
    lib.spf8_phase_similarity_i64.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf8_phase_similarity_i64.restype = ctypes.c_int64


_HANDLE, LIBSPF16_PATH = _resolve()

if _HANDLE is not None:
    # ABI gate before _register(): an older libspf16 on disk reports its
    # old version, and _register() may dereference symbols that didn't
    # exist then. Reading abi_version first lets us disable the handle
    # cleanly (LIBSPF16_AVAILABLE=False) instead of crashing import.
    _abi = int(_HANDLE.spf16_abi_version())
    if _abi != _EXPECTED_ABI_VERSION:
        _HANDLE = None
        LIBSPF16_PATH = None
    else:
        _register(_HANDLE)

LIBSPF16_AVAILABLE: bool = _HANDLE is not None


def load_library() -> ctypes.CDLL:
    """Return the loaded ``libspf16`` handle or raise."""
    if _HANDLE is None:
        raise RuntimeError(
            "libspf16 not available. Build it via `make -C native` (or "
            "`make -C native libspf16.so`) or set $SPF16_LIBRARY to an "
            "absolute path."
        )
    return _HANDLE


# ── ctypes pointer helpers ──────────────────────────────────────────────


def _u16_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _u8_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _prep_word(x: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x), dtype=np.uint16)


def _prep_phase8(x: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x), dtype=np.uint8)


def _unary_word(entry: str, w: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    lib = load_library()
    arr = _prep_word(w)
    out = np.empty_like(arr)
    getattr(lib, entry)(_u16_ptr(arr), _u16_ptr(out), ctypes.c_size_t(arr.size))
    return out


def _binary_word(
    entry: str, a: npt.ArrayLike, b: npt.ArrayLike
) -> npt.NDArray[np.uint16]:
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(a), _prep_word(b))
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    out = np.empty_like(a_arr)
    getattr(lib, entry)(
        _u16_ptr(a_arr),
        _u16_ptr(b_arr),
        _u16_ptr(out),
        ctypes.c_size_t(a_arr.size),
    )
    return out


# ── Public surface ──────────────────────────────────────────────────────


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.mul`."""
    return _binary_word("spf16_mul_array", a, b)


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.div`."""
    return _binary_word("spf16_div_array", a, b)


def conj(a: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.conj`."""
    return _unary_word("spf16_conj_array", a)


def abs2(a: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.abs2`."""
    return _unary_word("spf16_abs2_array", a)


def neg(a: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.neg`."""
    return _unary_word("spf16_neg_array", a)


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.add`.

    Uses libm sb_exact internally (no LUT). At SPF-16's Q3.4 / 8-bit Θ
    resolution this rounds inside the same grid bin as the Python
    ``sb_default`` reference for every non-singular input — the wedge
    LUT correction Python carries from libspf32 is below 1 SPF-16 ULP.
    Tests cross-validate within 1 L ULP / 1 Θ ULP.
    """
    return _binary_word("spf16_add_array", a, b)


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """C port of :func:`spf.scalar.spf16.sub`."""
    return _binary_word("spf16_sub_array", a, b)


def theta_sub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Phase-only subtraction: L=0, Theta=(Theta_a-Theta_b) mod 2^8."""
    return _binary_word("spf16_theta_sub_array", a, b)


def phase_bind(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Phase-only HDC bind: unit magnitude, theta=(a+b) mod 2^8."""
    return _binary_word("spf16_phase_bind_array", a, b)


def phase_unbind(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Phase-only HDC unbind: unit magnitude, theta=(a-b) mod 2^8."""
    return _binary_word("spf16_phase_unbind_array", a, b)


def phase_conj(a: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Phase-only conjugate: unit magnitude, theta=-theta mod 2^8."""
    return _unary_word("spf16_phase_conj_array", a)


def phase_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> int:
    """Integer phase similarity, saturated to int32 for ABI compatibility."""
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(a), _prep_word(b))
    a_arr = np.ascontiguousarray(a_arr).ravel()
    b_arr = np.ascontiguousarray(b_arr).ravel()
    return int(
        lib.spf16_phase_similarity(
            _u16_ptr(a_arr), _u16_ptr(b_arr), ctypes.c_size_t(a_arr.size)
        )
    )


def phase_similarity_i64(a: npt.ArrayLike, b: npt.ArrayLike) -> int:
    """Wide integer phase similarity, max 128 per matching lane."""
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(a), _prep_word(b))
    a_arr = np.ascontiguousarray(a_arr).ravel()
    b_arr = np.ascontiguousarray(b_arr).ravel()
    return int(
        lib.spf16_phase_similarity_i64(
            _u16_ptr(a_arr), _u16_ptr(b_arr), ctypes.c_size_t(a_arr.size)
        )
    )


def _binary_phase8(entry: str, a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_phase8(a), _prep_phase8(b))
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    out = np.empty_like(a_arr)
    getattr(lib, entry)(_u8_ptr(a_arr), _u8_ptr(b_arr), _u8_ptr(out), ctypes.c_size_t(a_arr.size))
    return out


def phase8_bind(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """SPF8 phase bind: theta=(a+b) mod 2^8."""
    return _binary_phase8("spf8_phase_bind_array", a, b)


def phase8_unbind(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """SPF8 phase unbind: theta=(a-b) mod 2^8."""
    return _binary_phase8("spf8_phase_unbind_array", a, b)


def phase8_conj(a: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """SPF8 phase conjugate: theta=-theta mod 2^8."""
    lib = load_library()
    arr = _prep_phase8(a)
    out = np.empty_like(arr)
    lib.spf8_phase_conj_array(_u8_ptr(arr), _u8_ptr(out), ctypes.c_size_t(arr.size))
    return out


def phase8_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> int:
    """Integer SPF8 phase similarity, saturated to int32 for compatibility."""
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_phase8(a), _prep_phase8(b))
    a_arr = np.ascontiguousarray(a_arr).ravel()
    b_arr = np.ascontiguousarray(b_arr).ravel()
    return int(
        lib.spf8_phase_similarity(
            _u8_ptr(a_arr), _u8_ptr(b_arr), ctypes.c_size_t(a_arr.size)
        )
    )


def phase8_similarity_i64(a: npt.ArrayLike, b: npt.ArrayLike) -> int:
    """Wide integer SPF8 phase similarity, max 128 per matching lane."""
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_phase8(a), _prep_phase8(b))
    a_arr = np.ascontiguousarray(a_arr).ravel()
    b_arr = np.ascontiguousarray(b_arr).ravel()
    return int(
        lib.spf8_phase_similarity_i64(
            _u8_ptr(a_arr), _u8_ptr(b_arr), ctypes.c_size_t(a_arr.size)
        )
    )


def butterfly(
    a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16]]:
    """C port of :func:`spf.scalar.spf16.butterfly`.

    Returns ``(top, bot) = (a + W·b, a − W·b)``.
    """
    lib = load_library()
    a_arr, b_arr, w_arr = np.broadcast_arrays(
        _prep_word(a), _prep_word(b), _prep_word(w)
    )
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    w_arr = np.ascontiguousarray(w_arr)
    top = np.empty_like(a_arr)
    bot = np.empty_like(a_arr)
    lib.spf16_butterfly_array(
        _u16_ptr(a_arr),
        _u16_ptr(b_arr),
        _u16_ptr(w_arr),
        _u16_ptr(top),
        _u16_ptr(bot),
        ctypes.c_size_t(a_arr.size),
    )
    return top, bot
