"""Mode-dispatched scalar core wrappers for SPF-32.

Wraps the `spf32_*_v` C entry points declared in `native/include/spf32_core.h`.
Each function takes the packed mode byte (bits[3:0] = base mode, bits[7:4] =
poly_kind per spf32_fb7_polyvariants.h). `mode == 0` is bit-identical to the
legacy no-mode wrappers exported from `spf.scalar.spf32` directly.

Provided here as the canonical "core" entry point for callers above the
format layer that want to choose a variant by mode.
"""

from __future__ import annotations

import ctypes

from spf.native._libspf32 import load_library


def mul(a: int, b: int, mode: int = 0) -> int:
    return int(load_library().spf32_mul_v(int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF, int(mode)))


def div(a: int, b: int, mode: int = 0) -> int:
    return int(load_library().spf32_div_v(int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF, int(mode)))


def conj(a: int, mode: int = 0) -> int:
    return int(load_library().spf32_conj_v(int(a) & 0xFFFFFFFF, int(mode)))


def abs2(a: int, mode: int = 0) -> int:
    return int(load_library().spf32_abs2_v(int(a) & 0xFFFFFFFF, int(mode)))


def neg(a: int, mode: int = 0) -> int:
    return int(load_library().spf32_neg_v(int(a) & 0xFFFFFFFF, int(mode)))


def twiddle(k: int, n: int, mode: int = 0) -> int:
    return int(load_library().spf32_twiddle_v(int(k), int(n), int(mode)))


def add(a: int, b: int, mode: int = 0) -> int:
    return int(load_library().spf32_add_v(int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF, int(mode)))


def sub(a: int, b: int, mode: int = 0) -> int:
    return int(load_library().spf32_sub_v(int(a) & 0xFFFFFFFF, int(b) & 0xFFFFFFFF, int(mode)))


def butterfly(u: int, v: int, w: int, mode: int = 0) -> tuple[int, int]:
    top = ctypes.c_uint32(0)
    bot = ctypes.c_uint32(0)
    load_library().spf32_butterfly_v(
        int(u) & 0xFFFFFFFF, int(v) & 0xFFFFFFFF, int(w) & 0xFFFFFFFF,
        ctypes.byref(top), ctypes.byref(bot), int(mode),
    )
    return int(top.value), int(bot.value)
