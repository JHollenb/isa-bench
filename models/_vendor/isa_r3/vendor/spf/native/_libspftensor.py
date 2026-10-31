"""ctypes wrapper around ``libspftensor``.

Resolves the library once at import time. Search order:

1. ``$SPFTENSOR_LIBRARY`` environment variable — absolute path to the shared object.
2. ``native/libspftensor.{dylib,so}`` relative to the repo root (four levels up
   from this file: ``src/spf/native/_libspftensor.py`` → repo).

If neither path resolves to a loadable file, :data:`LIBSPFTENSOR_AVAILABLE` stays
``False`` and :func:`load_library` raises ``RuntimeError``. The package imports
cleanly either way so the pure-Python reference path keeps working without a C build.

ABI version is checked at load time: ``spftensor_abi_version()`` must equal
``_EXPECTED_ABI_VERSION``. A mismatch disables the handle and leaves
``LIBSPFTENSOR_AVAILABLE = False`` — a loud failure is better than a silent
mismatch with wrong structs.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

__all__ = [
    "LIBSPFTENSOR_AVAILABLE",
    "LIBSPFTENSOR_PATH",
    "CByteMassHeader",
    "CContractOpts",
    "CPhaseRouteOpts",
    "CReducerSpec",
    "CRowStats",
    "CSoftmaxOpts",
    "CTelemetry",
    "load_library",
]

# Mirror of SPFTENSOR_ABI_VERSION in native/include/spftensor.h.
# Bump here when the header is bumped; sync_abi.py will catch drift.
_EXPECTED_ABI_VERSION = 7

SPFT_BASE_PHASE_CLOCK_RECORDS = 0x0106
SPFT_BASE_BYTE_MASS_PLANE = 0x0107


# ---------------------------------------------------------------------------
# ctypes.Structure mirrors for spftensor.h structs
# ---------------------------------------------------------------------------


class CReducerSpec(ctypes.Structure):
    """Mirror of ``spft_reducer_spec_t``.

    Field order and types must be EXACT — layout is asserted by a parity test.

        typedef struct {
            uint8_t fmt;
            uint8_t policy;
            uint8_t precision_mode;
            uint8_t routing;
            int8_t  domain;
            uint8_t reserved[3];
        } spft_reducer_spec_t;
    """

    _fields_ = [
        ("fmt", ctypes.c_uint8),
        ("policy", ctypes.c_uint8),
        ("precision_mode", ctypes.c_uint8),
        ("routing", ctypes.c_uint8),
        ("domain", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class CRowStats(ctypes.Structure):
    """Mirror of ``spft_row_stats_t``.

    typedef struct {
        int32_t n;
        int32_t pivot_l;
        int32_t dlrange_l;
        int32_t dom_gap_l;
        float   rbar;
    } spft_row_stats_t;
    """

    _fields_ = [
        ("n", ctypes.c_int32),
        ("pivot_l", ctypes.c_int32),
        ("dlrange_l", ctypes.c_int32),
        ("dom_gap_l", ctypes.c_int32),
        ("rbar", ctypes.c_float),
    ]


class CTelemetry(ctypes.Structure):
    """Mirror of ``spft_telemetry_t``.

    typedef struct {
        uint64_t rows_tropical;
        uint64_t rows_logpolar;
        uint64_t rows_cartesian;
        uint64_t rows_empty;
        uint64_t guard_zeroed;
        uint64_t fb7_adds;
    } spft_telemetry_t;
    """

    _fields_ = [
        ("rows_tropical", ctypes.c_uint64),
        ("rows_logpolar", ctypes.c_uint64),
        ("rows_cartesian", ctypes.c_uint64),
        ("rows_empty", ctypes.c_uint64),
        ("guard_zeroed", ctypes.c_uint64),
        ("fb7_adds", ctypes.c_uint64),
    ]


class CSoftmaxOpts(ctypes.Structure):
    """Mirror of ``spft_softmax_opts_t``.

    typedef struct {
        uint8_t score_kind;
        uint8_t mask;
        uint8_t lse_kind;
        uint8_t l_frac_bits;
        float   beta;
    } spft_softmax_opts_t;
    """

    _fields_ = [
        ("score_kind", ctypes.c_uint8),
        ("mask", ctypes.c_uint8),
        ("lse_kind", ctypes.c_uint8),
        ("l_frac_bits", ctypes.c_uint8),
        ("beta", ctypes.c_float),
    ]


class CContractOpts(ctypes.Structure):
    """Mirror of ``spft_contract_opts_t``.

    typedef struct {
        uint8_t transpose_b;
        uint8_t causal;
        uint8_t a_kind;
        uint8_t b_kind;
    } spft_contract_opts_t;
    """

    _fields_ = [
        ("transpose_b", ctypes.c_uint8),
        ("causal", ctypes.c_uint8),
        ("a_kind", ctypes.c_uint8),
        ("b_kind", ctypes.c_uint8),
    ]


class CPhaseRouteOpts(ctypes.Structure):
    """Mirror of ``spft_phase_route_opts_t``.

    typedef struct {
        float   beta;
        uint8_t l_frac_bits;
        uint8_t causal;
        uint8_t reserved[2];
    } spft_phase_route_opts_t;
    """

    _fields_ = [
        ("beta", ctypes.c_float),
        ("l_frac_bits", ctypes.c_uint8),
        ("causal", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 2),
    ]


class CByteMassHeader(ctypes.Structure):
    """Mirror of ``spft_byte_mass_header_t`` in ``spftensor.h``."""

    _fields_ = [
        ("profile_id", ctypes.c_uint16),
        ("q", ctypes.c_uint8),
        ("bit_order", ctypes.c_uint8),
        ("stop_law", ctypes.c_uint8),
        ("mass_bits", ctypes.c_uint8),
        ("mass_bytes", ctypes.c_uint8),
        ("reserved0", ctypes.c_uint8),
        ("node_count", ctypes.c_uint16),
        ("context_count", ctypes.c_uint32),
        ("context_limit", ctypes.c_uint32),
        ("max_output_length", ctypes.c_uint32),
        ("score_min", ctypes.c_int32),
        ("score_max", ctypes.c_int32),
        ("full_scale", ctypes.c_uint64),
        ("missing_sentinel", ctypes.c_uint64),
        ("invalid_sentinel", ctypes.c_uint64),
        ("stop_min_mass", ctypes.c_uint64),
        ("context_ids_offset", ctypes.c_uint32),
        ("valid_offset", ctypes.c_uint32),
        ("missing_offset", ctypes.c_uint32),
        ("masses_offset", ctypes.c_uint32),
        ("stop_masses_offset", ctypes.c_uint32),
        ("payload_bytes", ctypes.c_uint32),
        ("rom_id", ctypes.c_char * 64),
    ]


# ---------------------------------------------------------------------------
# Library resolution
# ---------------------------------------------------------------------------


def _candidate_paths() -> list[Path]:
    """Return candidate library paths in search order."""
    candidates: list[Path] = []
    env = os.environ.get("SPFTENSOR_LIBRARY")
    if env:
        candidates.append(Path(env).expanduser())

    # src/spf/native/_libspftensor.py → src/spf/native → src/spf → src → repo_root
    repo_root = Path(__file__).resolve().parents[3]
    ext = "dylib" if sys.platform == "darwin" else "so"
    candidates.append(repo_root / "native" / f"libspftensor.{ext}")
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
    """Declare argtypes / restype for every exported symbol in spftensor.h.

    A missing registration will surface as a TypeError at call time, not a silent
    misread. All struct pointer args use the concrete ctypes.Structure subclasses
    so the caller doesn't need to manage field packing manually.
    """
    # ABI version probe — registered first so the pre-register check can call it.
    lib.spftensor_abi_version.argtypes = []
    lib.spftensor_abi_version.restype = ctypes.c_int32
    lib.spft_phase_clock_transport.argtypes = [
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int16),
    ]
    lib.spft_phase_clock_transport.restype = ctypes.c_int32

    lib.spft_byte_mass_header_validate.argtypes = [
        ctypes.POINTER(CByteMassHeader),
        ctypes.c_size_t,
    ]
    lib.spft_byte_mass_header_validate.restype = ctypes.c_int32
    lib.spft_byte_mass_plane_validate.argtypes = [
        ctypes.POINTER(CByteMassHeader),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_void_p,
    ]
    lib.spft_byte_mass_plane_validate.restype = ctypes.c_int32

    # -----------------------------------------------------------------------
    # 1. Batched domain-routed reducer
    # -----------------------------------------------------------------------

    lib.spft_reduce_rows.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # rows
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # row_len
        ctypes.c_size_t,  # row_stride
        ctypes.POINTER(CReducerSpec),  # spec
        ctypes.POINTER(ctypes.c_uint32),  # out
        ctypes.POINTER(CRowStats),  # stats_out (nullable)
        ctypes.POINTER(ctypes.c_uint8),  # domain_out (nullable)
        ctypes.POINTER(CTelemetry),  # telemetry (nullable)
    ]
    lib.spft_reduce_rows.restype = ctypes.c_int

    lib.spft_reduce_rows_planes.argtypes = [
        ctypes.POINTER(ctypes.c_int16),  # L
        ctypes.POINTER(ctypes.c_uint16),  # T
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # row_len
        ctypes.c_size_t,  # row_stride
        ctypes.POINTER(CReducerSpec),  # spec
        ctypes.POINTER(ctypes.c_uint32),  # out
        ctypes.POINTER(CRowStats),  # stats_out (nullable)
        ctypes.POINTER(ctypes.c_uint8),  # domain_out (nullable)
        ctypes.POINTER(CTelemetry),  # telemetry (nullable)
    ]
    lib.spft_reduce_rows_planes.restype = ctypes.c_int

    lib.spft_compute_stats_rows.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # rows
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # row_len
        ctypes.c_size_t,  # row_stride
        ctypes.c_int,  # want_rbar
        ctypes.POINTER(CRowStats),  # stats_out
    ]
    lib.spft_compute_stats_rows.restype = ctypes.c_int

    lib.spft_route.argtypes = [
        ctypes.POINTER(CRowStats),  # stats
        ctypes.c_int,  # precision_mode
    ]
    lib.spft_route.restype = ctypes.c_int

    # -----------------------------------------------------------------------
    # 2. Softmax / LSE in the log domain
    # -----------------------------------------------------------------------

    lib.spft_lse_rows.argtypes = [
        ctypes.c_void_p,  # scores (type depends on score_kind)
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # row_len
        ctypes.c_size_t,  # row_stride
        ctypes.POINTER(CSoftmaxOpts),  # opts
        ctypes.POINTER(ctypes.c_int32),  # lse_l_out
        ctypes.POINTER(ctypes.c_double),  # lse_f64_out (nullable)
    ]
    lib.spft_lse_rows.restype = ctypes.c_int

    lib.spft_softmax_rows_f32.argtypes = [
        ctypes.c_void_p,  # scores
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # row_len
        ctypes.c_size_t,  # row_stride
        ctypes.POINTER(CSoftmaxOpts),  # opts
        ctypes.POINTER(ctypes.c_float),  # weights_out
    ]
    lib.spft_softmax_rows_f32.restype = ctypes.c_int

    lib.spft_softmax_rows_words.argtypes = [
        ctypes.c_void_p,  # scores
        ctypes.c_size_t,  # n_rows
        ctypes.c_size_t,  # row_len
        ctypes.c_size_t,  # row_stride
        ctypes.POINTER(CSoftmaxOpts),  # opts
        ctypes.POINTER(ctypes.c_uint32),  # words_out
    ]
    lib.spft_softmax_rows_words.restype = ctypes.c_int

    # -----------------------------------------------------------------------
    # 3. Data×data contraction driver
    # -----------------------------------------------------------------------

    lib.spft_contract.argtypes = [
        ctypes.c_void_p,  # A (M×K)
        ctypes.c_size_t,  # M
        ctypes.c_size_t,  # K
        ctypes.c_void_p,  # B (K×N or N×K)
        ctypes.c_size_t,  # N
        ctypes.POINTER(CReducerSpec),  # spec
        ctypes.POINTER(CContractOpts),  # opts
        ctypes.POINTER(ctypes.c_uint32),  # C output
        ctypes.POINTER(CTelemetry),  # telemetry (nullable)
    ]
    lib.spft_contract.restype = ctypes.c_int

    lib.spft_low_rank_apply.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # U (m×r)
        ctypes.POINTER(ctypes.c_uint32),  # lambda (r)
        ctypes.POINTER(ctypes.c_uint32),  # Vh (r×n)
        ctypes.c_size_t,  # m
        ctypes.c_size_t,  # r
        ctypes.c_size_t,  # n
        ctypes.POINTER(ctypes.c_uint32),  # x (n)
        ctypes.POINTER(CReducerSpec),  # spec
        ctypes.POINTER(ctypes.c_uint32),  # y (m)
        ctypes.POINTER(CTelemetry),  # telemetry (nullable)
    ]
    lib.spft_low_rank_apply.restype = ctypes.c_int

    # -----------------------------------------------------------------------
    # 4. Phase route (full-memory phase-keyed associative routing)
    # -----------------------------------------------------------------------

    lib.spft_phase_stats.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # H_mem (M×d)
        ctypes.c_size_t,  # M
        ctypes.c_size_t,  # d
        ctypes.POINTER(ctypes.c_float),  # mu out (d)
        ctypes.POINTER(ctypes.c_float),  # sd out (d)
    ]
    lib.spft_phase_stats.restype = ctypes.c_int

    lib.spft_phase_features.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # H (T×d)
        ctypes.c_size_t,  # T
        ctypes.c_size_t,  # d
        ctypes.POINTER(ctypes.c_float),  # mu (d)
        ctypes.POINTER(ctypes.c_float),  # sd (d)
        ctypes.POINTER(ctypes.c_float),  # Q out (T×d)
    ]
    lib.spft_phase_features.restype = ctypes.c_int

    lib.spft_phase_route.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # H (T×d)
        ctypes.c_size_t,  # T
        ctypes.c_size_t,  # d
        ctypes.POINTER(ctypes.c_float),  # H_mem (M×d)
        ctypes.POINTER(ctypes.c_float),  # H_mem_out (M×d)
        ctypes.c_size_t,  # M
        ctypes.POINTER(CReducerSpec),  # spec
        ctypes.POINTER(CPhaseRouteOpts),  # opts
        ctypes.POINTER(ctypes.c_float),  # H_out (T×d)
        ctypes.POINTER(CTelemetry),  # telemetry (nullable)
    ]
    lib.spft_phase_route.restype = ctypes.c_int

    lib.spft_attention_route.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # q_words (T×P)
        ctypes.c_size_t,  # T
        ctypes.c_size_t,  # P
        ctypes.POINTER(ctypes.c_uint32),  # k_words (M×P)
        ctypes.POINTER(ctypes.c_uint32),  # v_words (M×d_v)
        ctypes.c_size_t,  # M
        ctypes.c_size_t,  # d_v
        ctypes.POINTER(CReducerSpec),  # spec
        ctypes.POINTER(CPhaseRouteOpts),  # opts
        ctypes.POINTER(ctypes.c_uint32),  # delta_out (T×d_v)
        ctypes.POINTER(CTelemetry),  # telemetry (nullable)
    ]
    lib.spft_attention_route.restype = ctypes.c_int

    # -----------------------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------------------

    lib.spft_unpack_planes.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # w
        ctypes.POINTER(ctypes.c_int16),  # L
        ctypes.POINTER(ctypes.c_uint16),  # T
        ctypes.c_size_t,  # n
    ]
    lib.spft_unpack_planes.restype = None

    lib.spft_pack_planes.argtypes = [
        ctypes.POINTER(ctypes.c_int16),  # L
        ctypes.POINTER(ctypes.c_uint16),  # T
        ctypes.POINTER(ctypes.c_uint32),  # w
        ctypes.c_size_t,  # n
    ]
    lib.spft_pack_planes.restype = None

    lib.spft_decode_real_f32.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),  # w
        ctypes.POINTER(ctypes.c_float),  # out
        ctypes.c_size_t,  # n
    ]
    lib.spft_decode_real_f32.restype = None

    lib.spft_encode_real_f32.argtypes = [
        ctypes.POINTER(ctypes.c_float),  # x
        ctypes.POINTER(ctypes.c_uint32),  # out
        ctypes.c_size_t,  # n
    ]
    lib.spft_encode_real_f32.restype = None


_HANDLE, LIBSPFTENSOR_PATH = _resolve()

if _HANDLE is not None:
    # ABI gate runs before _register(): an older libspftensor on disk still
    # reports its old ABI, but _register() now dereferences symbols that
    # didn't exist then. Reading abi_version first lets us disable the
    # handle cleanly (LIBSPFTENSOR_AVAILABLE=False) instead of crashing import
    # with an undefined-symbol AttributeError inside _register().
    _HANDLE.spftensor_abi_version.argtypes = []
    _HANDLE.spftensor_abi_version.restype = ctypes.c_int32
    _abi = int(_HANDLE.spftensor_abi_version())
    if _abi != _EXPECTED_ABI_VERSION:
        import warnings

        warnings.warn(
            f"libspftensor ABI mismatch: expected {_EXPECTED_ABI_VERSION}, got {_abi}. Disabling native path.",
            RuntimeWarning,
            stacklevel=2,
        )
        _HANDLE = None
        LIBSPFTENSOR_PATH = None
    else:
        _register(_HANDLE)

LIBSPFTENSOR_AVAILABLE: bool = _HANDLE is not None


def load_library() -> ctypes.CDLL:
    """Return the loaded ``libspftensor`` handle, or raise if unavailable."""
    if _HANDLE is None:
        raise RuntimeError(
            "libspftensor is not available. Build it with `make -C native` from the "
            "repo root, or set SPFTENSOR_LIBRARY to an alternate path."
        )
    return _HANDLE
