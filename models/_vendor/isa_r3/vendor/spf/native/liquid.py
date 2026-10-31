"""ctypes signatures for the liquid-dsp FFT ABI used by the Kria agent.

The board agent loads liquid-dsp directly, outside ``libspf32``. Keeping the
signature mirror in ``src/spf/native`` lets the ABI parity gate compare the
host and board declarations even though the host package does not currently
ship a public liquid-dsp wrapper.
"""

from __future__ import annotations

import ctypes


def register_liquid_fft(lib: ctypes.CDLL) -> None:
    """Declare liquid-dsp FFT plan signatures on ``lib``."""
    fftplan_t = ctypes.c_void_p
    lib.fft_create_plan.argtypes = [
        ctypes.c_uint,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.fft_create_plan.restype = fftplan_t
    lib.fft_execute.argtypes = [fftplan_t]
    lib.fft_execute.restype = None
    lib.fft_destroy_plan.argtypes = [fftplan_t]
    lib.fft_destroy_plan.restype = None


__all__ = ["register_liquid_fft"]
