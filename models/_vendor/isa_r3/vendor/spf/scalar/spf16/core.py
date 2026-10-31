"""Mode-dispatched scalar core wrappers for SPF-16.

Wraps the `spf16_*_v` C entry points declared in `native/include/spf16_core.h`.
SPF-16 has no variants today, so `mode` is currently ignored — the wrappers
exist for uniform call shape across format types.
"""

from __future__ import annotations

import ctypes
from typing import Any

from spf.native.spf16 import load_library


def _lib() -> Any:
    lib: Any = load_library()
    if not getattr(lib, "_spf16_v_registered", False):
        for op in ("conj", "abs2", "neg"):
            fn = getattr(lib, f"spf16_{op}_v")
            fn.argtypes = [ctypes.c_uint16, ctypes.c_int]
            fn.restype = ctypes.c_uint16
        for op in ("mul", "div", "add", "sub"):
            fn = getattr(lib, f"spf16_{op}_v")
            fn.argtypes = [ctypes.c_uint16, ctypes.c_uint16, ctypes.c_int]
            fn.restype = ctypes.c_uint16
        lib.spf16_butterfly_v.argtypes = [
            ctypes.c_uint16, ctypes.c_uint16, ctypes.c_uint16,
            ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_uint16),
            ctypes.c_int,
        ]
        lib.spf16_butterfly_v.restype = None
        lib._spf16_v_registered = True
    return lib


def conj(a: int, mode: int = 0) -> int:
    return int(_lib().spf16_conj_v(int(a) & 0xFFFF, int(mode)))


def abs2(a: int, mode: int = 0) -> int:
    return int(_lib().spf16_abs2_v(int(a) & 0xFFFF, int(mode)))


def neg(a: int, mode: int = 0) -> int:
    return int(_lib().spf16_neg_v(int(a) & 0xFFFF, int(mode)))


def mul(a: int, b: int, mode: int = 0) -> int:
    return int(_lib().spf16_mul_v(int(a) & 0xFFFF, int(b) & 0xFFFF, int(mode)))


def div(a: int, b: int, mode: int = 0) -> int:
    return int(_lib().spf16_div_v(int(a) & 0xFFFF, int(b) & 0xFFFF, int(mode)))


def add(a: int, b: int, mode: int = 0) -> int:
    return int(_lib().spf16_add_v(int(a) & 0xFFFF, int(b) & 0xFFFF, int(mode)))


def sub(a: int, b: int, mode: int = 0) -> int:
    return int(_lib().spf16_sub_v(int(a) & 0xFFFF, int(b) & 0xFFFF, int(mode)))


def butterfly(a: int, b: int, w: int, mode: int = 0) -> tuple[int, int]:
    top = ctypes.c_uint16(0)
    bot = ctypes.c_uint16(0)
    _lib().spf16_butterfly_v(
        int(a) & 0xFFFF, int(b) & 0xFFFF, int(w) & 0xFFFF,
        ctypes.byref(top), ctypes.byref(bot), int(mode),
    )
    return int(top.value), int(bot.value)
