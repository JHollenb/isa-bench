"""ctypes wrapper around ``libfp32ref``.

Parallel to :mod:`spf.native._libspf32` + :mod:`spf.native.spf32` + :mod:`spf.native.fft`, but
for the scalar FP32 reference library instead of SPF-32. Only used by the
DSP bench harness — the format library and its tests do not depend on this.

Availability is signaled through :data:`LIBFP32REF_AVAILABLE`; missing
library makes every function here raise ``RuntimeError`` at call time
without affecting import of the wider ``spf`` package.
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
    "LIBFP32REF_AVAILABLE",
    "LIBFP32REF_PATH",
    "demod",
    "div",
    "dot",
    "fft",
    "ifft",
    "load_library",
    "mul",
    "pa_pe",
    "pa_tone",
    "twiddle",
]

_EXPECTED_ABI_VERSION = 2


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("FP32REF_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())
    # src/spf/native/fp32ref.py → src/spf/native → src/spf → src → repo_root
    repo_root = Path(__file__).resolve().parents[3]
    ext = "dylib" if sys.platform == "darwin" else "so"
    candidates.append(repo_root / "native" / f"libfp32ref.{ext}")
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
    lib.fp32ref_abi_version.argtypes = []
    lib.fp32ref_abi_version.restype = ctypes.c_int32

    for _name in ("fp32ref_mul_array", "fp32ref_div_array", "fp32ref_dot_array"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None

    for _name in ("fp32ref_fft", "fp32ref_ifft"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        getattr(lib, _name).restype = ctypes.c_int

    lib.fp32ref_twiddle_array.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.fp32ref_twiddle_array.restype = None

    lib.fp32ref_demod_array.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
    ]
    lib.fp32ref_demod_array.restype = None

    lib.fp32ref_pa_tone.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
        ctypes.c_size_t,
    ]
    lib.fp32ref_pa_tone.restype = None

    lib.fp32ref_pa_pe.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
    ]
    lib.fp32ref_pa_pe.restype = None


_HANDLE, LIBFP32REF_PATH = _resolve()

if _HANDLE is not None:
    # ABI gate before _register(): stale libfp32ref binaries without the
    # new pa_tone / pa_pe symbols would otherwise crash import in the
    # registration loop. Reading abi_version first keeps the fallback
    # path (LIBFP32REF_AVAILABLE=False) intact.
    _abi = int(_HANDLE.fp32ref_abi_version())
    if _abi != _EXPECTED_ABI_VERSION:
        _HANDLE = None
        LIBFP32REF_PATH = None
    else:
        _register(_HANDLE)

LIBFP32REF_AVAILABLE: bool = _HANDLE is not None


def load_library() -> ctypes.CDLL:
    if _HANDLE is None:
        raise RuntimeError(
            "libfp32ref is not available. Build it with `make -C native` from the "
            "repo root, or set FP32REF_LIBRARY to an alternate path."
        )
    return _HANDLE


# ── Helpers ─────────────────────────────────────────────────────────────────


def _f32_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i32_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _i64_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _as_complex64(x: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x), dtype=np.complex64)


def _view_f32_pairs(arr: np.ndarray) -> np.ndarray:
    """Interleaved-float view of a contiguous complex64 array (zero copy)."""
    return arr.view(np.float32).reshape(-1)


# ── Public ops ──────────────────────────────────────────────────────────────


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.complex64]:
    lib = load_library()
    a_arr = _as_complex64(a)
    b_arr = _as_complex64(np.broadcast_to(np.asarray(b), a_arr.shape))
    b_arr = np.ascontiguousarray(b_arr, dtype=np.complex64)
    out = np.empty_like(a_arr)
    lib.fp32ref_mul_array(
        _f32_ptr(_view_f32_pairs(a_arr)),
        _f32_ptr(_view_f32_pairs(b_arr)),
        _f32_ptr(_view_f32_pairs(out)),
        ctypes.c_size_t(a_arr.size),
    )
    return out


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.complex64]:
    lib = load_library()
    a_arr = _as_complex64(a)
    b_arr = _as_complex64(np.broadcast_to(np.asarray(b), a_arr.shape))
    b_arr = np.ascontiguousarray(b_arr, dtype=np.complex64)
    out = np.empty_like(a_arr)
    lib.fp32ref_div_array(
        _f32_ptr(_view_f32_pairs(a_arr)),
        _f32_ptr(_view_f32_pairs(b_arr)),
        _f32_ptr(_view_f32_pairs(out)),
        ctypes.c_size_t(a_arr.size),
    )
    return out


def _run_fft(entry: str, x: npt.ArrayLike) -> npt.NDArray[np.complex64]:
    lib = load_library()
    arr = _as_complex64(x)
    if arr.ndim != 1:
        raise ValueError(f"FFT input must be 1-D; got shape {arr.shape}")
    n = int(arr.shape[0])
    if n < 2 or (n & (n - 1)) != 0 or n > (1 << 16):
        raise ValueError(f"FFT length N must be a power of 2 in [2, 2^16]; got N={n}")
    out = np.empty(n, dtype=np.complex64)
    rc = int(
        getattr(lib, entry)(
            _f32_ptr(_view_f32_pairs(arr)),
            _f32_ptr(_view_f32_pairs(out)),
            ctypes.c_int(n),
        )
    )
    if rc != 0:
        raise RuntimeError(f"{entry} returned {rc} (invalid length or allocation failure)")
    return out


def fft(x: npt.ArrayLike) -> npt.NDArray[np.complex64]:
    return _run_fft("fp32ref_fft", x)


def ifft(x: npt.ArrayLike) -> npt.NDArray[np.complex64]:
    return _run_fft("fp32ref_ifft", x)


def twiddle(k: npt.ArrayLike, n: int) -> npt.NDArray[np.complex64]:
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    lib = load_library()
    k_arr = np.ascontiguousarray(np.asarray(k), dtype=np.int64)
    out = np.empty(k_arr.shape, dtype=np.complex64)
    lib.fp32ref_twiddle_array(
        _i64_ptr(k_arr),
        ctypes.c_int32(int(n)),
        _f32_ptr(_view_f32_pairs(out)),
        ctypes.c_size_t(k_arr.size),
    )
    return out


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> np.complex64:
    """Scalar FP32 complex dot product. Baseline for the Task 3.5 bench."""
    lib = load_library()
    a_arr = _as_complex64(np.asarray(a).ravel())
    b_arr = _as_complex64(np.asarray(b).ravel())
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"dot: shape mismatch, got {a_arr.shape} and {b_arr.shape}"
        )
    out = np.zeros(1, dtype=np.complex64)
    lib.fp32ref_dot_array(
        _f32_ptr(_view_f32_pairs(a_arr)),
        _f32_ptr(_view_f32_pairs(b_arr)),
        _f32_ptr(_view_f32_pairs(out)),
        ctypes.c_size_t(a_arr.size),
    )
    return out[0]  # type: ignore[no-any-return]


def demod(symbols: npt.ArrayLike, order: int) -> npt.NDArray[np.int32]:
    if order <= 0:
        raise ValueError(f"order must be positive, got {order}")
    lib = load_library()
    arr = _as_complex64(symbols)
    out = np.empty(arr.shape, dtype=np.int32)
    lib.fp32ref_demod_array(
        _f32_ptr(_view_f32_pairs(arr)),
        ctypes.c_int32(int(order)),
        _i32_ptr(out),
        ctypes.c_size_t(arr.size),
    )
    return out


def pa_tone(
    omega: float, n: int, phase0: float = 0.0
) -> npt.NDArray[np.complex64]:
    """Scalar FP32 DDS tone (sinf/cosf per step). Baseline for PA bench."""
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    lib = load_library()
    re = np.empty(n, dtype=np.float32)
    im = np.empty(n, dtype=np.float32)
    if n > 0:
        lib.fp32ref_pa_tone(
            _f32_ptr(re),
            _f32_ptr(im),
            ctypes.c_float(float(omega)),
            ctypes.c_float(float(phase0)),
            ctypes.c_size_t(n),
        )
    out = np.empty(n, dtype=np.complex64)
    out.real = re
    out.imag = im
    return out


def pa_pe(
    seq_len: int, d_model: int, base: float = 10000.0
) -> npt.NDArray[np.float32]:
    """Scalar FP32 sinusoidal PE (sinf/cosf per step). Baseline for PA bench."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError(f"d_model must be a positive even integer, got {d_model}")
    lib = load_library()
    out = np.empty((seq_len, d_model), dtype=np.float32)
    if seq_len > 0:
        lib.fp32ref_pa_pe(
            _f32_ptr(out),
            ctypes.c_int32(int(seq_len)),
            ctypes.c_int32(int(d_model)),
            ctypes.c_float(float(base)),
        )
    return out
