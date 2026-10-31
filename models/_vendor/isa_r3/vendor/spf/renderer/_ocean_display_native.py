"""ctypes bridge to ``libspf_ocean_packed``: the ocean grid terminal on SPF32 words.

The C side (``native/src/scalar/spf32/sfr_ocean_packed.c``) receives the
prepared ocean carrier's Hermitian half plane as canonical ``uint32`` SPF32
words plus a small plane header, and executes phase advance, optional
frequency transfer, Hermitian completion, spectral derivative, the packed
inverse FFT (the v0.2 FB3-Hybrid butterfly ported to C in
``spf32_fb3_hybrid.c``; libspf32's FB7 Stockham stays selectable) and the
final real projection without ever forming a Cartesian coefficient plane.
This module only owns NumPy buffers and binds the ABI; every stage is also
exposed on its own so the Python word path in
:mod:`spf.renderer.ocean_terminal` / :func:`spf.renderer.continuity.packed_ifft2`
can be graded against it bit-for-bit (every word stage, the FB3 transform
included) or by tolerance (the FB7 transform).

The library is optional and never compiled inside a timed call.  Build it with
``make -C native`` (CMake target ``spf_ocean_packed``) or
``python -m spf.renderer._ocean_display_native``.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt

ABI_VERSION = 2
FFT_MODES = {"fast": 0, "balanced": 1, "accurate": 2, "exact": 3}
FFT_PROFILES = {"fb3": 0, "fb7": 1}
FFT_PROFILE_NAMES = {0: "spf32_fb3_hybrid", 1: "spf32_fb7_stockham"}
CONSUMER = "sfr_ocean_packed_words_c"
DISPLAY_CONSUMER = "sfr_ocean_display_packed_words_c"

_U32P = ctypes.POINTER(ctypes.c_uint32)
_U16P = ctypes.POINTER(ctypes.c_uint16)
_I64P = ctypes.POINTER(ctypes.c_int64)
_F32P = ctypes.POINTER(ctypes.c_float)


class PackedStats(ctypes.Structure):
    """Counters written by one packed grid call (mirrors ``sfr_ocean_packed_stats_t``)."""

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        (name, ctypes.c_uint64)
        for name in (
            "half_words_read",
            "phase_advances",
            "transfer_multiplies",
            "hermitian_conjugates",
            "derivative_log_muls",
            "derivative_zeroed",
            "saturated_log_codes",
            "fft_transforms",
            "fft_butterflies",
            "fft_direct_pairs",
            "fft_local_pairs",
            "fft_sentinel_pairs",
            "words_decoded",
            "sentinel_words",
            "nonfinite_outputs",
            "materialized_word_elements",
        )
    ] + [("arithmetic_boundary", ctypes.c_uint32), ("fft_profile", ctypes.c_uint32)]

    def to_dict(self) -> dict[str, int]:
        return {name: int(getattr(self, name)) for name, _ in self._fields_}


class PackedPlane(ctypes.Structure):
    """Carrier header for one Hermitian half plane (mirrors ``sfr_ocean_packed_plane_t``)."""

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("abi_version", ctypes.c_uint32),
        ("fft_mode", ctypes.c_uint32),
        ("rows", ctypes.c_int32),
        ("cols", ctypes.c_int32),
        ("normalize", ctypes.c_int32),
        ("fft_profile", ctypes.c_int32),
        ("threads", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("spacing_y", ctypes.c_double),
        ("spacing_x", ctypes.c_double),
    ]


def _suffix() -> str:
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform == "win32":
        return "dll"
    return "so"


def library_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    explicit = os.environ.get("SPF_OCEAN_PACKED_LIBRARY")
    if explicit:
        return Path(explicit)
    return root / "native" / f"libspf_ocean_packed.{_suffix()}"


def _configure(lib: Any) -> Any:
    for name in (
        "spf_sfr_ocean_packed_abi_version",
        "spf_sfr_ocean_packed_stats_size",
        "spf_sfr_ocean_packed_plane_size",
    ):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = ctypes.c_int
    if int(lib.spf_sfr_ocean_packed_abi_version()) != ABI_VERSION:
        raise RuntimeError("ocean packed library ABI mismatch; rebuild native/ (make -C native)")
    if int(lib.spf_sfr_ocean_packed_stats_size()) != ctypes.sizeof(PackedStats):
        raise RuntimeError("ocean packed stats record mismatch; rebuild native/")
    if int(lib.spf_sfr_ocean_packed_plane_size()) != ctypes.sizeof(PackedPlane):
        raise RuntimeError("ocean packed plane record mismatch; rebuild native/")
    plane = ctypes.POINTER(PackedPlane)
    stats = ctypes.POINTER(PackedStats)
    lib.spf_sfr_ocean_packed_phase_advance.argtypes = [_U32P, plane, _I64P, _U16P, ctypes.c_size_t, stats]
    lib.spf_sfr_ocean_packed_transfer.argtypes = [_U32P, plane, _U32P, stats]
    lib.spf_sfr_ocean_packed_complete.argtypes = [_U32P, plane, _U32P, stats]
    lib.spf_sfr_ocean_packed_derivative.argtypes = [_U32P, plane, ctypes.c_int32, _U32P, stats]
    lib.spf_sfr_ocean_packed_ifft2_words.argtypes = [_U32P, plane, _U32P, _U32P, stats]
    lib.spf_sfr_ocean_packed_ifft2_real.argtypes = [_U32P, plane, _U32P, _F32P, stats]
    lib.spf_sfr_ocean_packed_grid.argtypes = [
        _U32P, plane, _I64P, _U16P, ctypes.c_size_t, _U32P, _F32P, _F32P, _F32P, stats,
    ]
    for name in (
        "spf_sfr_ocean_packed_phase_advance",
        "spf_sfr_ocean_packed_transfer",
        "spf_sfr_ocean_packed_complete",
        "spf_sfr_ocean_packed_derivative",
        "spf_sfr_ocean_packed_ifft2_words",
        "spf_sfr_ocean_packed_ifft2_real",
        "spf_sfr_ocean_packed_grid",
    ):
        getattr(lib, name).restype = ctypes.c_int
    lib.spf32_fb3_default_threads.argtypes = []
    lib.spf32_fb3_default_threads.restype = ctypes.c_int
    lib.spf32_fb3_logadd_g_zero.argtypes = []
    lib.spf32_fb3_logadd_g_zero.restype = ctypes.c_int
    lib.spf32_fb3_logadd_plus.argtypes = []
    lib.spf32_fb3_logadd_plus.restype = ctypes.POINTER(ctypes.c_int32)
    lib.spf32_fb3_logadd_minus.argtypes = []
    lib.spf32_fb3_logadd_minus.restype = ctypes.POINTER(ctypes.c_int32)
    lib.spf32_fb3_hypot2.argtypes = [ctypes.c_double, ctypes.c_double]
    lib.spf32_fb3_hypot2.restype = ctypes.c_double
    return lib


@lru_cache(maxsize=1)
def library() -> Any:
    path = library_path()
    if not path.is_file():
        return None
    return _configure(ctypes.CDLL(str(path)))


def available() -> bool:
    """True when the packed ocean library is loadable (never compiles)."""

    if os.environ.get("SPF_OCEAN_PACKED_DISABLE"):
        return False
    try:
        return library() is not None
    except OSError:
        return False


def _require() -> Any:
    lib = library()
    if lib is None:
        raise RuntimeError("native packed ocean library is unavailable; run make -C native")
    return lib


def _raise(status: int, *, stage: str) -> None:
    if status == 0:
        return
    reason = {-1: "rejected its inputs", -2: "could not allocate scratch", -3: "packed FFT rejected the plane"}
    raise RuntimeError(f"packed ocean {stage} {reason.get(status, f'failed with status {status}')}")


def make_plane(
    shape: Sequence[int],
    *,
    spacing: Sequence[float],
    normalize: bool,
    fft_mode: str = "accurate",
    profile: str = "fb3",
    threads: int = 0,
) -> PackedPlane:
    """Plane header.  ``profile="fb3"`` (default) is the normative v0.2 transform,
    bit-identical to the Python oracle; ``"fb7"`` is the legacy libspf32
    transform.  ``threads=0`` lets the library pick (``SPF_FB3_THREADS`` or the
    processor count); results do not depend on the thread count."""

    rows, cols = (int(x) for x in shape)
    if len(tuple(shape)) != 2 or any(n < 2 or (n & (n - 1)) for n in (rows, cols)):
        raise ValueError("packed ocean plane requires a two-dimensional power-of-two shape")
    sy, sx = (float(x) for x in spacing)
    if not (math.isfinite(sy) and math.isfinite(sx) and sy > 0 and sx > 0):
        raise ValueError("packed ocean plane spacing must be finite and positive")
    if fft_mode not in FFT_MODES:
        raise ValueError(f"packed ocean fft_mode must be one of {sorted(FFT_MODES)}")
    if profile not in FFT_PROFILES:
        raise ValueError(f"packed ocean profile must be one of {sorted(FFT_PROFILES)}")
    if int(threads) < 0:
        raise ValueError("packed ocean threads must be non-negative (0 = auto)")
    return PackedPlane(
        ABI_VERSION,
        FFT_MODES[fft_mode],
        rows,
        cols,
        int(bool(normalize)),
        FFT_PROFILES[profile],
        int(threads),
        0,
        sy,
        sx,
    )


def plane_profile_name(plane: PackedPlane) -> str:
    """Report label for the transform a plane selects, e.g. ``spf32_fb3_hybrid_accurate``."""

    mode = {v: k for k, v in FFT_MODES.items()}.get(int(plane.fft_mode), str(int(plane.fft_mode)))
    return f"{FFT_PROFILE_NAMES.get(int(plane.fft_profile), 'unknown')}_{mode}"


def default_threads() -> int:
    """Row-parallel worker count the library uses for ``threads=0``."""

    return int(_require().spf32_fb3_default_threads())


def logadd_rom() -> tuple[np.ndarray, np.ndarray]:
    """The compiled v0.2 LOGADD ROMs (plus, minus) as int32 arrays, for parity checks."""

    lib = _require()
    g_zero = int(lib.spf32_fb3_logadd_g_zero())
    plus = np.ctypeslib.as_array(lib.spf32_fb3_logadd_plus(), shape=(g_zero,)).copy()
    minus = np.ctypeslib.as_array(lib.spf32_fb3_logadd_minus(), shape=(g_zero,)).copy()
    return plus, minus


def hypot2(x: float, y: float) -> float:
    """CPython's ``math.hypot(x, y)`` as the C port computes it (bit-identity probe)."""

    return float(_require().spf32_fb3_hypot2(float(x), float(y)))


def _words(value: npt.ArrayLike, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype.kind not in "iu":
        raise TypeError(f"{name} must be packed SPF32 words")
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    return np.ascontiguousarray(array, dtype=np.uint32)


def _half_shape(plane: PackedPlane) -> tuple[int, int]:
    return (int(plane.rows), int(plane.cols) // 2 + 1)


def _full_shape(plane: PackedPlane) -> tuple[int, int]:
    return (int(plane.rows), int(plane.cols))


def _ptr(array: np.ndarray, pointer: Any) -> Any:
    return array.ctypes.data_as(pointer)


def _modes(indices: Any, codes: Any) -> tuple[np.ndarray, np.ndarray]:
    index_array = np.ascontiguousarray(np.asarray(indices, dtype=np.int64).reshape(-1, 2))
    code_array = np.ascontiguousarray(np.asarray(codes, dtype=np.uint16).reshape(-1))
    if index_array.shape[0] != code_array.shape[0]:
        raise ValueError("packed ocean mode indices and phase codes must have one entry per mode")
    return index_array, code_array


def phase_advance(
    half: npt.ArrayLike, plane: PackedPlane, indices: Any, codes: Any
) -> tuple[np.ndarray, PackedStats]:
    """Stage 1: return a copy of ``half`` with the mode phase codes added (u16)."""

    lib = _require()
    words = _words(half, _half_shape(plane), name="half plane").copy()
    index_array, code_array = _modes(indices, codes)
    stats = PackedStats()
    _raise(
        lib.spf_sfr_ocean_packed_phase_advance(
            _ptr(words, _U32P), ctypes.byref(plane), _ptr(index_array, _I64P), _ptr(code_array, _U16P),
            ctypes.c_size_t(code_array.size), ctypes.byref(stats),
        ),
        stage="phase advance",
    )
    return words, stats


def transfer(
    half: npt.ArrayLike, plane: PackedPlane, transfer_half: npt.ArrayLike
) -> tuple[np.ndarray, PackedStats]:
    """Stage 2: SPF multiply of the half plane by transfer words."""

    lib = _require()
    words = _words(half, _half_shape(plane), name="half plane").copy()
    factor = _words(transfer_half, _half_shape(plane), name="transfer plane")
    stats = PackedStats()
    status = lib.spf_sfr_ocean_packed_transfer(
        _ptr(words, _U32P), ctypes.byref(plane), _ptr(factor, _U32P), ctypes.byref(stats)
    )
    _raise(status, stage="transfer")
    return words, stats


def complete(half: npt.ArrayLike, plane: PackedPlane) -> tuple[np.ndarray, PackedStats]:
    """Stage 3: Hermitian completion into the full packed plane."""

    lib = _require()
    words = _words(half, _half_shape(plane), name="half plane")
    full = np.empty(_full_shape(plane), dtype=np.uint32)
    stats = PackedStats()
    status = lib.spf_sfr_ocean_packed_complete(
        _ptr(words, _U32P), ctypes.byref(plane), _ptr(full, _U32P), ctypes.byref(stats)
    )
    _raise(status, stage="completion")
    return full, stats


def derivative(full: npt.ArrayLike, plane: PackedPlane, axis: int) -> tuple[np.ndarray, PackedStats]:
    """Stage 4: spectral derivative words along ``axis``."""

    lib = _require()
    words = _words(full, _full_shape(plane), name="full plane")
    out = np.empty_like(words)
    stats = PackedStats()
    _raise(
        lib.spf_sfr_ocean_packed_derivative(
            _ptr(words, _U32P), ctypes.byref(plane), ctypes.c_int32(int(axis)), _ptr(out, _U32P), ctypes.byref(stats)
        ),
        stage="derivative",
    )
    return out, stats


def ifft2_words(full: npt.ArrayLike, plane: PackedPlane) -> tuple[np.ndarray, PackedStats]:
    """Stage 5 alone: packed inverse 2-D FFT, result as packed words."""

    lib = _require()
    words = _words(full, _full_shape(plane), name="full plane")
    rows, cols = _full_shape(plane)
    scratch = np.empty(2 * rows * cols + max(rows, cols), dtype=np.uint32)
    out = np.empty((rows, cols), dtype=np.uint32)
    stats = PackedStats()
    _raise(
        lib.spf_sfr_ocean_packed_ifft2_words(
            _ptr(words, _U32P), ctypes.byref(plane), _ptr(scratch, _U32P), _ptr(out, _U32P), ctypes.byref(stats)
        ),
        stage="inverse FFT",
    )
    return out, stats


def ifft2_real(full: npt.ArrayLike, plane: PackedPlane) -> tuple[np.ndarray, PackedStats]:
    """Stages 5-6: packed inverse 2-D FFT and float32 real projection."""

    lib = _require()
    words = _words(full, _full_shape(plane), name="full plane")
    rows, cols = _full_shape(plane)
    scratch = np.empty(2 * rows * cols + max(rows, cols), dtype=np.uint32)
    out = np.empty((rows, cols), dtype=np.float32)
    stats = PackedStats()
    _raise(
        lib.spf_sfr_ocean_packed_ifft2_real(
            _ptr(words, _U32P), ctypes.byref(plane), _ptr(scratch, _U32P), _ptr(out, _F32P), ctypes.byref(stats)
        ),
        stage="inverse FFT",
    )
    return out, stats


def grid(
    half: npt.ArrayLike,
    plane: PackedPlane,
    *,
    need_height: bool = True,
    gradient_axes: Sequence[int] = (0, 1),
    indices: Any = None,
    codes: Any = None,
    transfer_half: npt.ArrayLike | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, PackedStats]:
    """Fused grid terminal: ``(height, grad_y, grad_x, stats)`` in float32."""

    lib = _require()
    words = _words(half, _half_shape(plane), name="half plane")
    axes = {int(a) for a in gradient_axes}
    if not need_height and not axes:
        raise ValueError("packed ocean grid must request height or a gradient")
    rows, cols = _full_shape(plane)
    height = np.empty((rows, cols), dtype=np.float32) if need_height else None
    grad_y = np.empty((rows, cols), dtype=np.float32) if 0 in axes else None
    grad_x = np.empty((rows, cols), dtype=np.float32) if 1 in axes else None
    if indices is not None and codes is not None:
        index_array, code_array = _modes(indices, codes)
    else:
        index_array = np.empty((0, 2), dtype=np.int64)
        code_array = np.empty(0, dtype=np.uint16)
    factor = None if transfer_half is None else _words(transfer_half, _half_shape(plane), name="transfer plane")
    stats = PackedStats()
    _raise(
        lib.spf_sfr_ocean_packed_grid(
            _ptr(words, _U32P),
            ctypes.byref(plane),
            _ptr(index_array, _I64P),
            _ptr(code_array, _U16P),
            ctypes.c_size_t(code_array.size),
            None if factor is None else _ptr(factor, _U32P),
            None if height is None else _ptr(height, _F32P),
            None if grad_y is None else _ptr(grad_y, _F32P),
            None if grad_x is None else _ptr(grad_x, _F32P),
            ctypes.byref(stats),
        ),
        stage="grid",
    )
    return height, grad_y, grad_x, stats


def compile_library(output: str | os.PathLike[str] | None = None) -> Path:
    """Build the packed ocean library against ``native/libspf32``."""

    root = Path(__file__).resolve().parents[3]
    native = root / "native"
    target = library_path() if output is None else Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        os.environ.get("CC", "cc"),
        "-std=c11",
        "-shared",
        "-fPIC",
        "-O3",
        "-ffp-contract=off",
        "-I",
        str(native / "include"),
        str(native / "src" / "scalar" / "spf32" / "sfr_ocean_packed.c"),
        str(native / "src" / "scalar" / "spf32" / "spf32_fb3_hybrid.c"),
        "-lpthread",
        "-L",
        str(native),
        "-lspf32",
        "-Wl,-rpath," + str(native),
        "-o",
        str(target),
        "-lm",
    ]
    subprocess.run(command, check=True)
    library.cache_clear()
    return target


def main() -> None:
    print(compile_library())


__all__ = [
    "ABI_VERSION",
    "CONSUMER",
    "DISPLAY_CONSUMER",
    "PackedPlane",
    "PackedStats",
    "available",
    "compile_library",
    "complete",
    "default_threads",
    "derivative",
    "grid",
    "hypot2",
    "ifft2_real",
    "ifft2_words",
    "library",
    "library_path",
    "logadd_rom",
    "make_plane",
    "phase_advance",
    "plane_profile_name",
    "transfer",
]


if __name__ == "__main__":
    main()
