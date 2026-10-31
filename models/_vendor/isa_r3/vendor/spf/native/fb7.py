"""C-backed FB7 fused butterfly + FFT.

Mirrors :mod:`spf.fft.fused` (Python reference) and
``native/src/spf32/fb7/spf32_fb7.c`` (C). The C symbols live in ``libspf32``.

Mode names match the C header constants in
``native/include/spf32.h``:

    "fast"      -> SPF32_FB7_MODE_FAST       (dominant_gap =  8 · L_SCALE)
    "balanced"  -> SPF32_FB7_MODE_BALANCED   (dominant_gap = 12 · L_SCALE)
    "accurate"  -> SPF32_FB7_MODE_ACCURATE   (bypass disabled; polynomial finish)

Reference: SPF Algorithm Reference §5.2 / §6.2 / §8.
"""
from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import numpy.typing as npt

from spf.native._libspf32 import load_library

__all__ = [
    "MODE_ACCURATE",
    "MODE_BALANCED",
    "MODE_FAST",
    "butterfly",
    "butterfly_phase_array",
    "fft",
    "fft64",
    "mode_from_name",
]

MODE_FAST = 0
MODE_BALANCED = 1
MODE_ACCURATE = 2

# poly_kind nibble (high 4 bits of the mode int) selects the inner
# polynomial / transcendental kernel stack. Mirrors FB7_POLY_KIND_*
# constants in native/src/spf32/fb7/spf32_fb7_polyvariants.h. The default
# kind = 0 (baseline degree-9 atan / degree-7 log2 / degree-4 exp2 in
# double) is the only stack the existing T1.x gates run against directly;
# degree-reduced and bit-trick variants are opt-in via the composite
# names below.
POLY_KIND_7D = 0
POLY_KIND_5D = 1
POLY_KIND_7F = 2
POLY_KIND_5F = 3
POLY_KIND_BITTRICK_FAST = 4
POLY_KIND_BITTRICK_BAL = 5
POLY_KIND_BITTRICK_ACC = 6

_MODE_NAMES = {
    "fast": MODE_FAST,
    "balanced": MODE_BALANCED,
    "accurate": MODE_ACCURATE,
}

# Composite names select both the dominant-gap bypass threshold AND the
# inner kernel stack. Plain "fast"/"balanced"/"accurate" stay tied to
# poly7D for back-compat with the existing accuracy gates; opt in to the
# bit-trick kernels with the explicit names here.
_COMPOSITE_MODE_NAMES = {
    "bittrick_fast":     MODE_FAST     | (POLY_KIND_BITTRICK_FAST << 4),
    "bittrick_balanced": MODE_BALANCED | (POLY_KIND_BITTRICK_BAL  << 4),
    "bittrick_accurate": MODE_ACCURATE | (POLY_KIND_BITTRICK_ACC  << 4),
    "poly5d_balanced":   MODE_BALANCED | (POLY_KIND_5D << 4),
    "poly5d_accurate":   MODE_ACCURATE | (POLY_KIND_5D << 4),
    "poly7f_balanced":   MODE_BALANCED | (POLY_KIND_7F << 4),
    "poly5f_balanced":   MODE_BALANCED | (POLY_KIND_5F << 4),
}

_SIGNATURES_BOUND = False


def mode_from_name(name: str | int) -> int:
    if isinstance(name, int):
        if name < 0 or name > 0xFF:
            raise ValueError(
                f"FB7 mode integer {name!r} out of packed byte range 0..0xFF"
            )
        base = name & 0x0F
        if base not in (MODE_FAST, MODE_BALANCED, MODE_ACCURATE):
            raise ValueError(
                f"FB7 mode integer {name!r} has invalid base nibble "
                f"{base:#x}; expected one of "
                f"{MODE_FAST:#x}/{MODE_BALANCED:#x}/{MODE_ACCURATE:#x}"
            )
        poly = (name >> 4) & 0x0F
        if poly > POLY_KIND_BITTRICK_ACC:
            raise ValueError(
                f"FB7 mode integer {name!r} has invalid poly_kind nibble "
                f"{poly}; native fb7_valid_mode accepts 0..6"
            )
        return name
    key = name.lower()
    if key in _MODE_NAMES:
        return _MODE_NAMES[key]
    if key in _COMPOSITE_MODE_NAMES:
        return _COMPOSITE_MODE_NAMES[key]
    expected = sorted(_MODE_NAMES) + sorted(_COMPOSITE_MODE_NAMES)
    raise ValueError(f"unknown FB7 mode {name!r}; expected one of {expected}")


def _bind_signatures(lib: Any) -> None:
    global _SIGNATURES_BOUND
    if _SIGNATURES_BOUND:
        return
    lib.spf32_butterfly_fb7.argtypes = [
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.spf32_butterfly_fb7.restype = None
    lib.spf32_butterfly_fb7_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_butterfly_fb7_array.restype = None
    lib.spf32_butterfly_fb7_phase_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint16,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_butterfly_fb7_phase_array.restype = None
    lib.spf32_fft_fb7.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.spf32_fft_fb7.restype = ctypes.c_int
    lib.spf32_fft64_fb7.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.spf32_fft64_fb7.restype = ctypes.c_int
    _SIGNATURES_BOUND = True


def _u32_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def butterfly(
    u: int | np.uint32,
    v: int | np.uint32,
    w: int | np.uint32,
    mode: str | int = "balanced",
) -> tuple[int, int]:
    """Scalar FB7 butterfly: returns (top, bot) where top = u + W·v, bot = u − W·v."""
    lib = load_library()
    _bind_signatures(lib)
    top = ctypes.c_uint32(0)
    bot = ctypes.c_uint32(0)
    lib.spf32_butterfly_fb7(
        ctypes.c_uint32(int(u)),
        ctypes.c_uint32(int(v)),
        ctypes.c_uint32(int(w)),
        ctypes.byref(top), ctypes.byref(bot),
        ctypes.c_int(mode_from_name(mode)),
    )
    return int(top.value), int(bot.value)


def butterfly_array(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    w: npt.ArrayLike,
    mode: str | int = "balanced",
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """Vectorized FB7 butterfly. Returns (top, bot) arrays of equal length."""
    lib = load_library()
    _bind_signatures(lib)
    u_a = np.ascontiguousarray(np.asarray(u), dtype=np.uint32)
    v_a = np.ascontiguousarray(np.asarray(v), dtype=np.uint32)
    w_a = np.ascontiguousarray(np.asarray(w), dtype=np.uint32)
    if not (u_a.size == v_a.size == w_a.size):
        raise ValueError("u, v, w must have the same length")
    top = np.empty(u_a.size, dtype=np.uint32)
    bot = np.empty(u_a.size, dtype=np.uint32)
    lib.spf32_butterfly_fb7_array(
        _u32_ptr(u_a), _u32_ptr(v_a), _u32_ptr(w_a),
        _u32_ptr(top), _u32_ptr(bot),
        ctypes.c_size_t(u_a.size),
        ctypes.c_int(mode_from_name(mode)),
    )
    return top, bot


def butterfly_phase_array(
    u: npt.ArrayLike,
    v: npt.ArrayLike,
    theta0: int,
    theta_step: int,
    mode: str | int = "balanced",
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """FB7 butterfly over a linear phase source instead of packed twiddle words."""
    lib = load_library()
    _bind_signatures(lib)
    u_a = np.ascontiguousarray(np.asarray(u), dtype=np.uint32)
    v_a = np.ascontiguousarray(np.asarray(v), dtype=np.uint32)
    if u_a.size != v_a.size:
        raise ValueError("u and v must have the same length")
    top = np.empty(u_a.size, dtype=np.uint32)
    bot = np.empty(u_a.size, dtype=np.uint32)
    lib.spf32_butterfly_fb7_phase_array(
        _u32_ptr(u_a), _u32_ptr(v_a),
        ctypes.c_uint16(int(theta0) & 0xFFFF),
        ctypes.c_int32(int(theta_step)),
        _u32_ptr(top), _u32_ptr(bot),
        ctypes.c_size_t(u_a.size),
        ctypes.c_int(mode_from_name(mode)),
    )
    return top, bot


def fft(
    spectrum_in: npt.ArrayLike,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """Radix-2 forward DFT using FB7 fused butterflies."""
    lib = load_library()
    _bind_signatures(lib)
    arr = np.ascontiguousarray(np.asarray(spectrum_in), dtype=np.uint32)
    n = arr.size
    if n < 2 or n & (n - 1):
        raise ValueError(f"length must be a power of two in [2, 65536]; got {n}")
    out = np.empty(n, dtype=np.uint32)
    rc = lib.spf32_fft_fb7(
        _u32_ptr(arr), _u32_ptr(out),
        ctypes.c_int(int(n)),
        ctypes.c_int(mode_from_name(mode)),
    )
    if rc != 0:
        raise RuntimeError(f"spf32_fft_fb7 returned {rc}")
    return out


def fft64(
    spectrum_in: npt.ArrayLike,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """N=64 FB7 FFT (matches the HLS spf32_fft64_fb7 kernel surface)."""
    arr = np.ascontiguousarray(np.asarray(spectrum_in), dtype=np.uint32)
    if arr.size != 64:
        raise ValueError(f"fft64 requires length 64; got {arr.size}")
    return fft(arr, mode=mode)
