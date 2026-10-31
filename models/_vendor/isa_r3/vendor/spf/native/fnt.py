"""ctypes wrapper around ``libfermat_ntt``.

Parallel to :mod:`spf.native._libspf32` (SPF-32) and :mod:`spf.native.fp32ref` (soft-float
reference), but for the length-32 Fermat-NTT + Nussbaumer complex-convolution
primitive. Standalone module — no LUTs, no libm dependency on the hot path.

Availability through :data:`LIBFERMAT_NTT_AVAILABLE`. Missing library makes
every function here raise ``RuntimeError`` at call time without affecting
import of the wider ``spf`` package (or of :mod:`spf.fnt`, whose pure-
Python reference stays usable regardless).
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
    "LIBFERMAT_NTT_AVAILABLE",
    "LIBFERMAT_NTT_PATH",
    "fermat_conv_32",
    "fermat_ntt_32",
    "fermat_ntt_32_inv",
    "load_library",
]

FERMAT_NTT_N = 32


def _candidate_paths() -> list[Path]:
    candidates: list[Path] = []
    env = os.environ.get("FERMAT_NTT_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())
    # src/spf/native/fnt.py → src/spf/native → src/spf → src → repo_root
    repo_root = Path(__file__).resolve().parents[3]
    ext = "dylib" if sys.platform == "darwin" else "so"
    candidates.append(repo_root / "native" / f"libfermat_ntt.{ext}")
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
    for _name in ("fermat_ntt_32", "fermat_ntt_32_inv"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        getattr(lib, _name).restype = None
    lib.fermat_conv_32.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),  # a_re
        ctypes.POINTER(ctypes.c_uint64),  # a_im
        ctypes.POINTER(ctypes.c_uint64),  # b_re
        ctypes.POINTER(ctypes.c_uint64),  # b_im
        ctypes.POINTER(ctypes.c_uint64),  # c_re
        ctypes.POINTER(ctypes.c_uint64),  # c_im
    ]
    lib.fermat_conv_32.restype = None


_HANDLE, LIBFERMAT_NTT_PATH = _resolve()

if _HANDLE is not None:
    _register(_HANDLE)

LIBFERMAT_NTT_AVAILABLE: bool = _HANDLE is not None


def load_library() -> ctypes.CDLL:
    if _HANDLE is None:
        raise RuntimeError(
            "libfermat_ntt is not available. Build it with `make -C native` from the "
            "repo root, or set FERMAT_NTT_LIBRARY to an alternate path."
        )
    return _HANDLE


def _u64_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64))


def _as_u64_32(x: npt.ArrayLike, name: str) -> npt.NDArray[np.uint64]:
    arr = np.ascontiguousarray(np.asarray(x), dtype=np.uint64)
    if arr.shape != (FERMAT_NTT_N,):
        raise ValueError(
            f"{name} must be shape ({FERMAT_NTT_N},); got {arr.shape}"
        )
    return arr


def fermat_ntt_32(x: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Length-32 forward Fermat-NTT via the C primitive."""
    lib = load_library()
    in_arr = _as_u64_32(x, "fermat_ntt_32 input")
    out = np.empty(FERMAT_NTT_N, dtype=np.uint64)
    lib.fermat_ntt_32(_u64_ptr(in_arr), _u64_ptr(out))
    return out


def fermat_ntt_32_inv(x: npt.ArrayLike) -> npt.NDArray[np.uint64]:
    """Length-32 inverse Fermat-NTT (incl. N^-1 scaling) via C."""
    lib = load_library()
    in_arr = _as_u64_32(x, "fermat_ntt_32_inv input")
    out = np.empty(FERMAT_NTT_N, dtype=np.uint64)
    lib.fermat_ntt_32_inv(_u64_ptr(in_arr), _u64_ptr(out))
    return out


def fermat_conv_32(
    a_re: npt.ArrayLike,
    a_im: npt.ArrayLike,
    b_re: npt.ArrayLike,
    b_im: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint64], npt.NDArray[np.uint64]]:
    """Length-32 cyclic complex convolution via the C Nussbaumer path.

    Inputs must be length-32 arrays of uint64 with values in ``[0, F_5)``;
    the caller owns signed↔unsigned encoding (see
    :mod:`spf.fnt.nussbaumer.to_signed` / ``_pack_signed``).
    """
    lib = load_library()
    a_re_u = _as_u64_32(a_re, "a_re")
    a_im_u = _as_u64_32(a_im, "a_im")
    b_re_u = _as_u64_32(b_re, "b_re")
    b_im_u = _as_u64_32(b_im, "b_im")
    c_re = np.empty(FERMAT_NTT_N, dtype=np.uint64)
    c_im = np.empty(FERMAT_NTT_N, dtype=np.uint64)
    lib.fermat_conv_32(
        _u64_ptr(a_re_u),
        _u64_ptr(a_im_u),
        _u64_ptr(b_re_u),
        _u64_ptr(b_im_u),
        _u64_ptr(c_re),
        _u64_ptr(c_im),
    )
    return c_re, c_im
