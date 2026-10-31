"""ctypes wrapper around ``libspf32``.

Resolves the library once at import time. Search order:

1. ``$SPF_LIBRARY`` environment variable — absolute path to the shared object.
2. ``native/libspf32.{dylib,so}`` relative to the repo root (four levels up
   from this file: ``src/spf/native/_lib.py`` → repo).

If neither path resolves to a loadable file, :data:`LIBSPF_AVAILABLE` stays
``False`` and :func:`load_library` raises ``RuntimeError``. The package
imports cleanly either way so the pure-Python reference path keeps working
without a C build.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

__all__ = ["LIBSPF_AVAILABLE", "LIBSPF_PATH", "CAccum", "CComplex", "load_library"]

_EXPECTED_ABI_VERSION = 24


class CComplex(ctypes.Structure):
    """Mirror of ``spf32_complex_t`` from ``native/include/spf32.h``.

    Passed by value into ``spf32_encode`` and returned by ``spf32_decode``.
    """

    _fields_ = [("re", ctypes.c_double), ("im", ctypes.c_double)]


# Mirrors SPF32_ACCUM_K_MAX_HARD in native/include/spf32.h. Keep in sync with the
# C header — a mismatch corrupts the accumulator struct layout. The smoke
# test ``test_c_python_struct_size`` pins this to sizeof(spf32_accum_t).
_ACCUM_K_MAX_HARD: int = 16


class CAccum(ctypes.Structure):
    """Mirror of ``spf32_accum_t`` from ``native/include/spf32.h`` (Test 4).

    Stack-allocatable via ``CAccum()``; pass by pointer into
    ``spf32_accum_*``. Field names mirror the C struct (they are nominally
    opaque on the C side; the Python parity tests read them to cross-check
    the queue state against the pure-Python ``LazyAccum``).
    """

    _fields_ = [
        ("k_cap", ctypes.c_uint8),
        ("k_len", ctypes.c_uint8),
        ("_reserved", ctypes.c_uint8 * 2),
        ("n_sb_calls", ctypes.c_uint32),
        ("n_drops", ctypes.c_uint32),
        ("queue", ctypes.c_uint32 * _ACCUM_K_MAX_HARD),
    ]


def _candidate_paths() -> list[Path]:
    """Return candidate library paths in search order."""
    candidates: list[Path] = []
    env = os.environ.get("SPF_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())

    # src/spf/native/_lib.py → src/spf/native → src/spf → src → repo_root
    repo_root = Path(__file__).resolve().parents[3]
    ext = "dylib" if sys.platform == "darwin" else "so"
    candidates.append(repo_root / "native" / f"libspf32.{ext}")
    return candidates


def _resolve() -> tuple[ctypes.CDLL | None, Path | None]:
    """Find and open the first usable candidate; return (handle, path) or (None, None)."""
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
    """Declare argtypes / restype for every exported symbol.

    Grown incrementally as each C port task lands. A missing registration
    here will surface as a TypeError at call time, not a silent misread.
    """
    lib.spf32_abi_version.argtypes = []
    lib.spf32_abi_version.restype = ctypes.c_int32

    # SIMD dispatch probe — returns the ISA backends actually wired into
    # the dispatch table at load time.
    lib.spf32_simd_active.argtypes = []
    lib.spf32_simd_active.restype = ctypes.c_int32

    # SPF_DATA_DIR resolver (cross-arch, cross-type). Returns getenv or
    # SPF_DATA_DIR_DEFAULT from native/include/spf_config.h. Tilde unresolved.
    lib.spf_data_dir.argtypes = []
    lib.spf_data_dir.restype = ctypes.c_char_p

    # Common transcendental wrappers (cross-arch, cross-type). Each takes a
    # mode enum from native/include/spf_common_modes.h. mode == 0 → SPF
    # bit-trick default per fn.
    lib.spf_atan2.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.spf_atan2.restype = ctypes.c_double
    lib.spf_log2.argtypes = [ctypes.c_double, ctypes.c_int]
    lib.spf_log2.restype = ctypes.c_double
    lib.spf_exp2.argtypes = [ctypes.c_double, ctypes.c_int]
    lib.spf_exp2.restype = ctypes.c_double
    lib.spf_sin_cos.argtypes = [
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lib.spf_sin_cos.restype = None

    # Core mode-dispatched wrappers (`_v` = variant-aware) from spf32_core.h.
    # mode == 0 → default hot path, bit-identical to legacy no-mode entries.
    lib.spf32_mul_v.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    lib.spf32_mul_v.restype = ctypes.c_uint32
    lib.spf32_div_v.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    lib.spf32_div_v.restype = ctypes.c_uint32
    lib.spf32_conj_v.argtypes = [ctypes.c_uint32, ctypes.c_int]
    lib.spf32_conj_v.restype = ctypes.c_uint32
    lib.spf32_abs2_v.argtypes = [ctypes.c_uint32, ctypes.c_int]
    lib.spf32_abs2_v.restype = ctypes.c_uint32
    lib.spf32_neg_v.argtypes = [ctypes.c_uint32, ctypes.c_int]
    lib.spf32_neg_v.restype = ctypes.c_uint32
    lib.spf32_twiddle_v.argtypes = [ctypes.c_int64, ctypes.c_int32, ctypes.c_int]
    lib.spf32_twiddle_v.restype = ctypes.c_uint32
    lib.spf32_add_v.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    lib.spf32_add_v.restype = ctypes.c_uint32
    lib.spf32_sub_v.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    lib.spf32_sub_v.restype = ctypes.c_uint32
    lib.spf32_butterfly_v.argtypes = [
        ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.spf32_butterfly_v.restype = None
    lib.spf32_fft_v.argtypes = [
        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int, ctypes.c_int,
    ]
    lib.spf32_fft_v.restype = ctypes.c_int

    # VOLK-style multi-variant dispatch (ABI v16).
    lib.spf32_dispatch_select.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.spf32_dispatch_select.restype = ctypes.c_int
    lib.spf32_dispatch_active.argtypes = [ctypes.c_char_p]
    lib.spf32_dispatch_active.restype = ctypes.c_char_p
    lib.spf32_dispatch_primitive_at.argtypes = [
        ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.spf32_dispatch_primitive_at.restype = ctypes.c_char_p
    lib.spf32_dispatch_variant_at.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    lib.spf32_dispatch_variant_at.restype = ctypes.c_char_p

    # Constants (Task #11).
    for name in (
        "spf32_const_l_scale",
        "spf32_const_theta_scale",
    ):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_int32
    for name in (
        "spf32_const_word_zero",
        "spf32_const_word_nan",
    ):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_uint32

    # fast_log2 (Task #11).
    lib.spf32_fast_log2.argtypes = [ctypes.c_double]
    lib.spf32_fast_log2.restype = ctypes.c_double

    # encode / decode (Task #12) — struct-by-value in/out.
    lib.spf32_encode.argtypes = [CComplex]
    lib.spf32_encode.restype = ctypes.c_uint32
    lib.spf32_decode.argtypes = [ctypes.c_uint32]
    lib.spf32_decode.restype = CComplex

    # Integer-closed scalar ops (Task #12).
    for _name in ("spf32_mul", "spf32_div"):
        getattr(lib, _name).argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        getattr(lib, _name).restype = ctypes.c_uint32
    for _name in ("spf32_conj", "spf32_abs2", "spf32_neg"):
        getattr(lib, _name).argtypes = [ctypes.c_uint32]
        getattr(lib, _name).restype = ctypes.c_uint32
    lib.spf32_twiddle.argtypes = [ctypes.c_int64, ctypes.c_int32]
    lib.spf32_twiddle.restype = ctypes.c_uint32

    # Retained legacy ABI add/sub/butterfly (poly7 / FB7). The v0.2 renderer's
    # preferred dense coherent profile is the portable FB3-Hybrid path.
    for _name in ("spf32_add", "spf32_sub"):
        getattr(lib, _name).argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        getattr(lib, _name).restype = ctypes.c_uint32
    lib.spf32_butterfly.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.spf32_butterfly.restype = None

    # FFT driver (Task #14) — pointer in, pointer out, rc on success/fail.
    # Declared with explicit attribute access (not getattr/loop) so the
    # abi-parity-check skill can parse the signature and grade it against
    # the agent-side duplicate in hw/platforms/kria/board/agent.py.
    lib.spf32_fft.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.spf32_fft.restype = ctypes.c_int
    lib.spf32_ifft.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
    ]
    lib.spf32_ifft.restype = ctypes.c_int

    # Vectorized array ops (Task #15). All return void; per-element loop is
    # run in C so benchmarks amortize the ~200 ns ctypes transition over N
    # elements.
    for _name in ("spf32_conj_array", "spf32_abs2_array", "spf32_neg_array"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    for _name in (
        "spf32_mul_array",
        "spf32_div_array",
        "spf32_add_array",
        "spf32_sub_array",
        "spf32_theta_sub_array",
    ):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    lib.spf32_twiddle_array.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_twiddle_array.restype = None
    # Radix-2 butterfly vectorized: one Python→C transition per call.
    # Shipped so the Kria host runner has a PS-C partner for the PL
    # spf32_butterfly_kernel equivalence grading.
    lib.spf32_butterfly_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # a
        ctypes.POINTER(ctypes.c_uint32),  # b
        ctypes.POINTER(ctypes.c_uint32),  # w
        ctypes.POINTER(ctypes.c_uint32),  # top
        ctypes.POINTER(ctypes.c_uint32),  # bot
        ctypes.c_size_t,                  # n
    ]
    lib.spf32_butterfly_array.restype = None
    lib.spf32_encode_array.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_encode_array.restype = None
    lib.spf32_decode_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    lib.spf32_decode_array.restype = None
    # Fast Cartesian↔SPF — poly7/BAM helpers shared with the FB7 butterfly.
    # Use only at the Cartesian boundary; never inside an SPF-native pipeline.
    lib.spf32_encode_fast.argtypes = [CComplex]
    lib.spf32_encode_fast.restype = ctypes.c_uint32
    lib.spf32_decode_fast.argtypes = [ctypes.c_uint32]
    lib.spf32_decode_fast.restype = CComplex
    lib.spf32_encode_fast_array.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_encode_fast_array.restype = None
    lib.spf32_decode_fast_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    lib.spf32_decode_fast_array.restype = None
    lib.spf32_fast_log2_array.argtypes = [
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_size_t,
    ]
    lib.spf32_fast_log2_array.restype = None

    # SPF-native OFDM/QPSK DSP kernels. These are phase/log-polar integer
    # kernels and intentionally return final receiver products directly.
    lib.spf32_ofdm_zf_equalize_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_ofdm_zf_equalize_array.restype = None
    lib.spf32_phase_diff_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_phase_diff_array.restype = None
    lib.spf32_qpsk_phase_demod_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_qpsk_phase_demod_array.restype = None
    lib.spf32_qpsk_phase_demod_conf_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_qpsk_phase_demod_conf_array.restype = None
    lib.spf32_ofdm_qpsk_rx_hotpath_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_ofdm_qpsk_rx_hotpath_array.restype = None
    lib.spf32_phase_rotate_const_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_phase_rotate_const_array.restype = None
    lib.spf32_phase_ramp_apply_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint16,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_phase_ramp_apply_array.restype = None
    lib.spf32_phase_ramp_nd_apply_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # in
        ctypes.POINTER(ctypes.c_uint32),  # dims
        ctypes.c_uint8,  # order
        ctypes.POINTER(ctypes.c_uint16),  # ramp_R (order*order, row-major)
        ctypes.POINTER(ctypes.c_uint16),  # ramp_b (order)
        ctypes.POINTER(ctypes.c_uint16),  # quadratic_ramp (order or NULL)
        ctypes.POINTER(ctypes.c_uint16),  # cubic_ramp (order or NULL)
        ctypes.POINTER(ctypes.c_uint16),  # phase_gradient (order or NULL, ABI v22)
        ctypes.POINTER(ctypes.c_uint16),  # mixed_ramp (order*(order-1)/2 or NULL, ABI v22)
        ctypes.POINTER(ctypes.c_uint16),  # cubic_mixed (2 or NULL, order 2 only, ABI v24)
        ctypes.c_int,  # nonlinear
        ctypes.POINTER(ctypes.c_uint32),  # out
        ctypes.c_size_t,  # n
    ]
    lib.spf32_phase_ramp_nd_apply_array.restype = None
    lib.spf32_gain_phase_apply_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int16,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_gain_phase_apply_array.restype = None
    lib.spf32_pairwise_phase_diff_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_pairwise_phase_diff_array.restype = None
    lib.spf32_log_power_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
    ]
    lib.spf32_log_power_array.restype = None
    lib.spf32_channel_estimate_ls_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_channel_estimate_ls_array.restype = None
    lib.spf32_channel_inverse_guard_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int16,
        ctypes.c_int16,
        ctypes.c_size_t,
    ]
    lib.spf32_channel_inverse_guard_array.restype = None
    lib.spf32_ofdm_qpsk_eq_demod_fused_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_ofdm_qpsk_eq_demod_fused_array.restype = None
    lib.spf32_ofdm_mpsk_eq_demod_fused_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint16,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_ofdm_mpsk_eq_demod_fused_array.restype = None
    lib.spf32_constellation_lut_demap_fused_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int16,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_int16,
        ctypes.c_int16,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_constellation_lut_demap_fused_array.restype = None
    lib.spf32_qam16_logproj_eq_demod_fused_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int16,
        ctypes.c_uint16,
        ctypes.c_int16,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_qam16_logproj_eq_demod_fused_array.restype = None
    lib.spf32_constellation_lut_llr_fused_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int16,
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_uint8,
        ctypes.c_int16,
        ctypes.c_int16,
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_constellation_lut_llr_fused_array.restype = None
    lib.spf32_ofdm_cpe_estimate_from_pilots.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_ofdm_cpe_estimate_from_pilots.restype = ctypes.c_uint16
    lib.spf32_pilot_interp_loglinear_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.spf32_pilot_interp_loglinear_array.restype = None
    lib.spf32_phase_slope_estimate_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.spf32_phase_slope_estimate_array.restype = None
    lib.spf32_differential_preamble_match_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_uint16,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.spf32_differential_preamble_match_array.restype = None
    lib.spf32_doppler_phase_diff_estimate_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.spf32_doppler_phase_diff_estimate_array.restype = None
    lib.spf32_aoa_pair_phase_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_aoa_pair_phase_array.restype = None
    lib.spf32_aoa_lut_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_aoa_lut_array.restype = None
    lib.spf32_logpower_threshold_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.spf32_logpower_threshold_array.restype = None
    lib.spf32_log_os_cfar_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int16,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.spf32_log_os_cfar_array.restype = None
    lib.spf32_fb7_reduce_tree_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_fb7_reduce_tree_array.restype = None
    lib.spf32_spectral_matched_filter_fb7.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.spf32_spectral_matched_filter_fb7.restype = ctypes.c_int
    lib.spf32_candidate_fft_bins_fb7.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_candidate_fft_bins_fb7.restype = ctypes.c_int
    lib.spf32_beamform_fused_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_beamform_fused_array.restype = None

    lib.runtime_radar_sparse_doppler.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_uint8,
        ctypes.c_int32,
        ctypes.c_int16,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_uint,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
        ctypes.c_size_t,
    ]
    lib.runtime_radar_sparse_doppler.restype = None
    lib.runtime_aoa_pair_lut.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_uint16,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_size_t,
    ]
    lib.runtime_aoa_pair_lut.restype = None
    lib.runtime_rf_native_ingress.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
    ]
    lib.runtime_rf_native_ingress.restype = None

    for _name in ("runtime_radar_sparse_doppler", "runtime_radar_sparse_doppler_scalar"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_uint8,
            ctypes.c_int32,
            ctypes.c_int16,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    for _name in ("runtime_aoa_pair_lut", "runtime_aoa_pair_lut_scalar"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_uint16,
            ctypes.c_uint8,
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None
    for _name in ("runtime_rf_native_ingress", "runtime_rf_native_ingress_scalar"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.POINTER(ctypes.c_uint16),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = None

    # Dot-product accumulators (Task 3.5). Return the packed accumulator as a
    # single uint32, not a pointer — so restype must be declared explicitly
    # (ctypes defaults scalars to c_int, which would truncate on 64-bit).
    for _name in ("spf32_dot", "spf32_dot_nokahan"):
        getattr(lib, _name).argtypes = [
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.c_size_t,
        ]
        getattr(lib, _name).restype = ctypes.c_uint32

    # Sb Test 7: Goertzel single-bin / K-bin DFT. Not spec-normative —
    # research kernel that bypasses Sb() by running FP32 Goertzel lanes.
    lib.spf32_goertzel_bin_fp32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spf32_goertzel_bin_fp32.restype = None
    lib.spf32_goertzel_bin.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int64,
    ]
    lib.spf32_goertzel_bin.restype = ctypes.c_uint32
    lib.spf32_goertzel_bins.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.spf32_goertzel_bins.restype = ctypes.c_int


    # Phase accumulator (Phase 2 demo). DDS tone + sinusoidal PE via
    # integer phase + 1024-entry sincos LUT.
    lib.spf32_pa_init.argtypes = []
    lib.spf32_pa_init.restype = None
    lib.spf32_pa_tone_sc.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_size_t,
    ]
    lib.spf32_pa_tone_sc.restype = None
    lib.spf32_pa_pe_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
    ]
    lib.spf32_pa_pe_f32.restype = None
    lib.spf32_pa_pe_f32_slice.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_float,
    ]
    lib.spf32_pa_pe_f32_slice.restype = None

    # Sb Test 4 — lazy multi-phasor tail accumulator.
    lib.spf32_accum_init.argtypes = [ctypes.POINTER(CAccum), ctypes.c_uint32]
    lib.spf32_accum_init.restype = None
    lib.spf32_accum_reset.argtypes = [ctypes.POINTER(CAccum)]
    lib.spf32_accum_reset.restype = None
    lib.spf32_accum_push.argtypes = [ctypes.POINTER(CAccum), ctypes.c_uint32]
    lib.spf32_accum_push.restype = None
    lib.spf32_accum_finalize.argtypes = [ctypes.POINTER(CAccum)]
    lib.spf32_accum_finalize.restype = ctypes.c_uint32
    for _name in (
        "spf32_accum_k_cap",
        "spf32_accum_k_len",
        "spf32_accum_sb_call_count",
        "spf32_accum_drop_count",
    ):
        getattr(lib, _name).argtypes = [ctypes.POINTER(CAccum)]
        getattr(lib, _name).restype = ctypes.c_uint32
    lib.spf32_accum_sum.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.spf32_accum_sum.restype = ctypes.c_uint32
    lib.spf32_accum_inner_product.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_uint32,
    ]
    lib.spf32_accum_inner_product.restype = ctypes.c_uint32
    lib.spf32_accum_lazy_enabled.argtypes = []
    lib.spf32_accum_lazy_enabled.restype = ctypes.c_int32

    # FB7 butterfly and FFT entry points (ABI 8, Phase 2 promotion). The
    # scalar butterfly is declared here; the full Python wrappers are in
    # spf.native.fb7 (which calls _bind_signatures on demand).
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

    # Poly7 ADD/SUB (§4.3) — Sb-free complex addition via shared FB7
    # polynomial kernels. Mode constants are SPF32_FB7_MODE_{FAST,BALANCED,ACCURATE}.
    lib.spf32_add_poly7.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    lib.spf32_add_poly7.restype = ctypes.c_uint32
    lib.spf32_add_poly7_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_add_poly7_array.restype = None
    lib.spf32_sub_poly7.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    lib.spf32_sub_poly7.restype = ctypes.c_uint32
    lib.spf32_sub_poly7_array.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.spf32_sub_poly7_array.restype = None

    # spf32_sb_cotransform_lookup removed in Phase 5 (Sb LUT retired).
    # spf32_add_cotransform now delegates to spf32_add_poly7(ACCURATE).

    # NOTE: spf32_phase_diff_array, spf32_ofdm_zf_equalize_array, and
    # spf32_qpsk_phase_demod_array are registered once in the ABI-v10 block above.
    # A verbatim duplicate registration previously lived here and was removed —
    # the signatures were identical, so it only risked masking a future divergence.

    # Level-4 OperatorBlock / TensorBlock / SynthPlan kernels (ABI v17).
    # Struct pointers are registered as c_void_p here so the spf.tensor
    # wrapper can own the ctypes structure definitions without making the
    # loader import spf.tensor at module load.
    lib.spf32_op_abi_version.argtypes = []
    lib.spf32_op_abi_version.restype = ctypes.c_int32
    lib.spf32_op_header_size.argtypes = []
    lib.spf32_op_header_size.restype = ctypes.c_size_t
    lib.spf32_tb_v2_header_size.argtypes = []
    lib.spf32_tb_v2_header_size.restype = ctypes.c_size_t
    lib.spf32_op_circular_diff_u16.argtypes = [ctypes.c_uint16, ctypes.c_uint16]
    lib.spf32_op_circular_diff_u16.restype = ctypes.c_int16
    lib.spf32_op_header_validate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.spf32_op_header_validate.restype = ctypes.c_int
    lib.spf32_op_total_elements.argtypes = [ctypes.c_void_p]
    lib.spf32_op_total_elements.restype = ctypes.c_size_t
    lib.spf32_tb_v2_profile_term_validate.argtypes = [ctypes.c_void_p]
    lib.spf32_tb_v2_profile_term_validate.restype = ctypes.c_int
    lib.spf32_tb_v2_ramp_det_mod2.argtypes = [ctypes.c_void_p]
    lib.spf32_tb_v2_ramp_det_mod2.restype = ctypes.c_int
    lib.spf32_tb_v2_header_validate.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.spf32_tb_v2_header_validate.restype = ctypes.c_int
    lib.spf32_tb_v2_from_operator_header.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.spf32_tb_v2_from_operator_header.restype = ctypes.c_int
    lib.spf32_tb_v2_total_elements.argtypes = [ctypes.c_void_p]
    lib.spf32_tb_v2_total_elements.restype = ctypes.c_size_t
    lib.spf32_op_affine_reconstruct.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_op_affine_reconstruct.restype = ctypes.c_int
    lib.spf32_tb_v2_affine_reconstruct.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_tb_v2_affine_reconstruct.restype = ctypes.c_int
    lib.spf32_op_elementwise_mul_planes.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_int16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_op_elementwise_mul_planes.restype = None
    lib.spf32_op_elementwise_mul_words.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
    ]
    lib.spf32_op_elementwise_mul_words.restype = None
    lib.spf32_op_hdc_bind_phase_u16.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_op_hdc_bind_phase_u16.restype = None
    lib.spf32_op_hdc_unbind_phase_u16.argtypes = [
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
    ]
    lib.spf32_op_hdc_unbind_phase_u16.restype = None
    lib.spf32_op_lr_apply_words.argtypes = [
        ctypes.c_uint8,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.spf32_op_lr_apply_words.restype = ctypes.c_int
    lib.spf32_synth_choose_mode.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint16),
    ]
    lib.spf32_synth_choose_mode.restype = ctypes.c_int
    lib.spf32_synth_export_tone_controls.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.spf32_synth_export_tone_controls.restype = ctypes.c_int
    lib.spf32_synth_nco_iq16.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.spf32_synth_nco_iq16.restype = ctypes.c_int
    lib.spf32_synth_ifft_iq16.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.spf32_synth_ifft_iq16.restype = ctypes.c_int
    lib.spf32_synth_auto_iq16.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    lib.spf32_synth_auto_iq16.restype = ctypes.c_int



_HANDLE, LIBSPF_PATH = _resolve()

if _HANDLE is not None:
    # ABI gate runs before _register(): an older libspf32 on disk still
    # reports its old ABI, but _register() now dereferences symbols that
    # didn't exist then. Reading abi_version first lets us disable the
    # handle cleanly (LIBSPF_AVAILABLE=False) instead of crashing import
    # with an undefined-symbol AttributeError inside _register().
    _abi = int(_HANDLE.spf32_abi_version())
    if _abi != _EXPECTED_ABI_VERSION:
        _HANDLE = None
        LIBSPF_PATH = None
    else:
        _register(_HANDLE)

LIBSPF_AVAILABLE: bool = _HANDLE is not None


def load_library() -> ctypes.CDLL:
    """Return the loaded ``libspf32`` handle, or raise if unavailable."""
    if _HANDLE is None:
        raise RuntimeError(
            "libspf32 is not available. Build it with `make -C native` from the "
            "repo root, or set SPF_LIBRARY to an alternate path."
        )
    return _HANDLE
