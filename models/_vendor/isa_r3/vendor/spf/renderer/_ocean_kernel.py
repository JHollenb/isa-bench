"""Optional native fused ocean consumers.

The C boundary takes the actual SFR source representation: canonical packed
SPF32 words or float32 control coefficients, signed interleaved Fourier mode
indices, and current wide phase state.  It returns one shared height/gradient
triple per grid or query point.  The C implementation performs the structural
tile recurrence and derivative factors; this module only validates/owns NumPy
buffers and binds the ABI.

The library is optional and is never compiled inside a timed call.  Build it
with ``python -m spf.renderer._ocean_kernel`` or the native CMake target.  If
the library is absent, callers can use the existing reference ocean path;
there is no implicit complex128 materialization here.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Literal

import numpy as np
import numpy.typing as npt

# Keep the diagnostic contract portable across C implementations where
# ``long double`` may be either binary64 or extended precision.  The native
# point fallback uses a looser implementation-dependent bound when available,
# but this conservative binary64-safe cycle bound is the threshold at which a
# ledger stops claiming phase-code precision.
_SPF_POINT_CYCLE_LIMIT = 2.0**32


class OceanStats(ctypes.Structure):
    """C telemetry for one fused ocean call."""

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        (name, ctypes.c_uint64)
        for name in (
            "samples",
            "active_modes",
            "zero_modes",
            "phase_emissions",
            "trig_evaluations",
            "visible_samples",
            "skipped_samples",
            "invalid_modes",
            "nonfinite_outputs",
            "recurrence_reseeds",
            "cartesian_source_decodes",
            "term_preparations",
            "setup_exp2",
            "setup_trig",
            "phase_grid_steps",
            "phase_lut_evaluations",
            "pivot_updates",
        )
    ] + [
        ("phase_bits", ctypes.c_uint32),
        ("arithmetic_boundary", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("max_abs_height", ctypes.c_double),
        ("max_abs_gradient", ctypes.c_double),
        ("phase_quantization_bound", ctypes.c_double),
    ]

    def to_dict(self) -> dict[str, int | float]:
        """Return deterministic JSON-friendly telemetry."""

        return {name: getattr(self, name) for name, _ in self._fields_}


@dataclass(frozen=True, slots=True)
class OceanErrorLedger:
    """Absolute error accounting for the packed SPF signed reduction.

    The ledger is deliberately additive and remains meaningful near
    cancellation: it never converts the bound to a relative error.  Values
    are output-domain bounds after the requested normalization.  The phase
    terms cover the u16 emission and quarter-wave table interpolation; the
    fixed-point terms cover coefficient and Q30 product rounding.  The
    ``grid_recurrence_bound`` covers rounded Q32 spatial steps on non-power
    grids; it is zero for the exact u16 recurrence.  Signed accumulator
    additions are exact while they remain inside int64, which is reported
    explicitly by :attr:`int64_safe`.
    """

    mode_count: int
    active_modes: int
    invalid_modes: int
    phase_emission_bound: float
    phase_lut_bound: float
    coordinate_composition_bound: float
    grid_recurrence_bound: float
    q30_coefficient_bound: float
    q30_product_bound: float
    accumulator_rounding_bound: float
    final_float32_bound: float
    height_absolute_bound: float
    gradient_y_absolute_bound: float
    gradient_x_absolute_bound: float
    int64_safe: bool
    finite: bool

    def to_dict(self) -> dict[str, int | float | bool]:
        return {name: getattr(self, name) for name, _ in self.__dataclass_fields__.items()}


class OceanContext(ctypes.Structure):
    """Stable callback context layout mirrored by ``sfr_ocean.h``."""

    _fields_: ClassVar[list[tuple[str, Any]]] = [
        ("abi_version", ctypes.c_uint32),
        ("kind", ctypes.c_uint32),
        ("mode_count", ctypes.c_uint32),
        ("phase_bits", ctypes.c_uint32),
        ("height", ctypes.c_int32),
        ("width", ctypes.c_int32),
        ("normalize", ctypes.c_int32),
        ("reserved", ctypes.c_int32),
        ("spacing_y", ctypes.c_double),
        ("spacing_x", ctypes.c_double),
        ("origin_y", ctypes.c_double),
        ("origin_x", ctypes.c_double),
        ("footprint_sigma_y", ctypes.c_double),
        ("footprint_sigma_x", ctypes.c_double),
        ("words_or_re", ctypes.c_void_p),
        ("im", ctypes.POINTER(ctypes.c_float)),
        ("indices", ctypes.POINTER(ctypes.c_int64)),
        ("phases", ctypes.POINTER(ctypes.c_uint64)),
        ("increments", ctypes.POINTER(ctypes.c_uint64)),
        ("stats", ctypes.POINTER(OceanStats)),
        ("prepared_terms", ctypes.c_void_p),
        ("prepared_phase_hash", ctypes.c_uint64),
        ("prepared_valid", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
    ]


_U32P = ctypes.POINTER(ctypes.c_uint32)
_U8P = ctypes.POINTER(ctypes.c_uint8)
_U16P = ctypes.POINTER(ctypes.c_uint16)
_U64P = ctypes.POINTER(ctypes.c_uint64)
_I64P = ctypes.POINTER(ctypes.c_int64)
_F32P = ctypes.POINTER(ctypes.c_float)
_F64P = ctypes.POINTER(ctypes.c_double)
_STATS_P = ctypes.POINTER(OceanStats)
_VOIDP = ctypes.c_void_p


def _suffix() -> str:
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform == "win32":
        return "dll"
    return "so"


def library_path() -> Path:
    """Return the native ocean library path used by the lazy loader."""

    root = Path(__file__).resolve().parents[3]
    explicit = os.environ.get("SPF_OCEAN_LIBRARY")
    if explicit:
        return Path(explicit)
    suffix = _suffix()
    return root / "native" / f"libspf_ocean.{suffix}"


def _pointer(array: np.ndarray | None, pointer_type: Any) -> Any:
    if array is None:
        return pointer_type()
    return array.ctypes.data_as(pointer_type)


def _configure(lib: Any) -> Any:
    """Attach all argument declarations to one loaded library."""

    lib.spf_sfr_ocean_abi_version.argtypes = []
    lib.spf_sfr_ocean_abi_version.restype = ctypes.c_int
    for name in ("spf_sfr_ocean_stats_size", "spf_sfr_ocean_context_size"):
        fn = getattr(lib, name)
        fn.argtypes = []
        fn.restype = ctypes.c_int

    lib.spf_sfr_ocean_emit_phase_u16.argtypes = [_U64P, ctypes.c_uint32, _U16P, ctypes.c_size_t]
    lib.spf_sfr_ocean_emit_phase_u16.restype = ctypes.c_int

    grid_tail = [
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        _F32P,
        _F32P,
        _F32P,
        _STATS_P,
    ]
    lib.spf_sfr_ocean_grid_spf32.argtypes = [_U32P, _I64P, _U64P, _U64P, *grid_tail]
    lib.spf_sfr_ocean_grid_spf32.restype = ctypes.c_int
    lib.spf_sfr_ocean_grid_fp32.argtypes = [_F32P, _F32P, _I64P, _U64P, _U64P, *grid_tail]
    lib.spf_sfr_ocean_grid_fp32.restype = ctypes.c_int

    point_tail = [
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int32,
        ctypes.c_double,
        ctypes.c_double,
        _F64P,
        _U8P,
        ctypes.c_size_t,
        _F32P,
        _STATS_P,
    ]
    lib.spf_sfr_ocean_points_spf32.argtypes = [_U32P, _I64P, _U64P, _U64P, *point_tail]
    lib.spf_sfr_ocean_points_spf32.restype = ctypes.c_int
    lib.spf_sfr_ocean_points_fp32.argtypes = [_F32P, _F32P, _I64P, _U64P, _U64P, *point_tail]
    lib.spf_sfr_ocean_points_fp32.restype = ctypes.c_int

    for name in ("spf_sfr_ocean_sample_spf32", "spf_sfr_ocean_sample_fp32"):
        fn = getattr(lib, name)
        fn.argtypes = [_VOIDP, ctypes.c_double, ctypes.c_double, ctypes.c_int, _F64P]
        fn.restype = ctypes.c_int
    for name in (
        "spf_sfr_ocean_context_prepare",
        "spf_sfr_ocean_context_refresh",
    ):
        fn = getattr(lib, name)
        fn.argtypes = [_VOIDP]
        fn.restype = ctypes.c_int
    lib.spf_sfr_ocean_context_release.argtypes = [_VOIDP]
    lib.spf_sfr_ocean_context_release.restype = None
    return lib


@lru_cache(maxsize=1)
def library() -> Any:
    """Load the optional native library, returning ``None`` when absent."""

    path = library_path()
    if not path.is_file():
        if os.environ.get("SPF_OCEAN_LIBRARY"):
            raise RuntimeError(f"requested ocean library does not exist: {path}")
        return None
    lib = _configure(ctypes.CDLL(str(path)))
    if lib.spf_sfr_ocean_abi_version() != 1:
        raise RuntimeError("native ocean kernel ABI mismatch; rebuild the native target")
    if lib.spf_sfr_ocean_stats_size() != ctypes.sizeof(OceanStats):
        raise RuntimeError("native ocean stats ABI size mismatch; rebuild the native target")
    if lib.spf_sfr_ocean_context_size() != ctypes.sizeof(OceanContext):
        raise RuntimeError("native ocean context ABI size mismatch; rebuild the native target")
    return lib


def available() -> bool:
    """Return whether the compiled native ocean library is available."""

    return library() is not None


def _as_shape(shape: Sequence[int]) -> tuple[int, int]:
    values = tuple(int(value) for value in shape)
    if len(values) != 2 or any(value <= 0 for value in values):
        raise ValueError("native ocean shape must contain two positive dimensions")
    return values


def _as_pair(
    value: Sequence[float] | float,
    *,
    name: str,
    nonnegative: bool = False,
    allow_any: bool = False,
) -> tuple[float, float]:
    if isinstance(value, (int, float, np.integer, np.floating)):
        values: tuple[float, ...] = (float(value), float(value))
    else:
        values = tuple(float(item) for item in value)
    if len(values) != 2 or any(not math.isfinite(item) for item in values):
        raise ValueError(f"native ocean {name} must contain two finite values")
    if not allow_any and any(item < 0.0 if nonnegative else item <= 0.0 for item in values):
        qualifier = "non-negative" if nonnegative else "positive"
        raise ValueError(f"native ocean {name} must contain {qualifier} values")
    return (values[0], values[1])


def _as_indices(indices: npt.ArrayLike, mode_count: int | None = None) -> np.ndarray:
    array = np.asarray(indices, dtype=np.int64)
    if array.ndim == 1:
        if array.size != 2:
            raise ValueError("native ocean indices must have shape (K, 2)")
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("native ocean indices must have shape (K, 2)")
    if mode_count is not None and array.shape[0] != mode_count:
        raise ValueError("native ocean index count does not match coefficients")
    return np.ascontiguousarray(array, dtype=np.int64)


def _as_u64(values: npt.ArrayLike | None, count: int, *, name: str) -> np.ndarray:
    if values is None:
        return np.zeros(count, dtype=np.uint64)
    array = np.asarray(values)
    if array.ndim != 1 or array.size != count:
        raise ValueError(f"native ocean {name} must have shape (K,)")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError(f"native ocean {name} must be non-negative uint64 values")
    return np.ascontiguousarray(array, dtype=np.uint64)


def _as_words(words: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(words)
    if array.ndim != 1:
        array = array.reshape(-1)
    if array.dtype != np.uint32:
        raise TypeError("native SPF ocean words must be canonical uint32 values")
    return np.ascontiguousarray(array)


def _as_fp32_coefficients(coefficients: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(coefficients)
    if np.iscomplexobj(array):
        if array.ndim != 1:
            array = array.reshape(-1)
        return (
            np.ascontiguousarray(np.asarray(array.real, dtype=np.float32)),
            np.ascontiguousarray(np.asarray(array.imag, dtype=np.float32)),
        )
    if array.ndim == 2 and array.shape[1] == 2:
        return (
            np.ascontiguousarray(np.asarray(array[:, 0], dtype=np.float32)),
            np.ascontiguousarray(np.asarray(array[:, 1], dtype=np.float32)),
        )
    raise ValueError("FP32 ocean coefficients must be complex (K,) or real/imag (K, 2)")


def _as_points(points: npt.ArrayLike) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.size != 2:
            raise ValueError("native ocean points must have shape (M, 2)")
        array = array.reshape(1, 2)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("native ocean points must have shape (M, 2) in physical (y, x) order")
    if not np.all(np.isfinite(array)):
        raise ValueError("native ocean points must be finite")
    return np.ascontiguousarray(array, dtype=np.float64)


def _visible(values: npt.ArrayLike | None, count: int) -> np.ndarray | None:
    if values is None:
        return None
    array = np.asarray(values)
    if array.ndim != 1 or array.size != count:
        raise ValueError("native ocean visibility must have shape (M,)")
    return np.ascontiguousarray(array, dtype=np.uint8)


def _require_library() -> Any:
    lib = library()
    if lib is None:
        raise RuntimeError("native ocean library is unavailable; build libspf_ocean first")
    return lib


def _call_common(
    *,
    shape: Sequence[int],
    spacing: Sequence[float] | float,
    origin: Sequence[float] | float,
    normalize: bool,
    footprint: Sequence[float] | float,
    phase_bits: int,
) -> tuple[tuple[int, int], tuple[float, float], tuple[float, float], tuple[float, float], int]:
    full_shape = _as_shape(shape)
    spacing_pair = _as_pair(spacing, name="spacing")
    origin_pair = _as_pair(origin, name="origin", allow_any=True)
    footprint_pair = _as_pair(footprint, name="footprint", nonnegative=True)
    bits = int(phase_bits)
    if bits < 1 or bits > 64:
        raise ValueError("native ocean phase_bits must be in [1, 64]")
    return full_shape, spacing_pair, origin_pair, footprint_pair, bits


def emit_phase_codes(phases: npt.ArrayLike, phase_bits: int = 64) -> np.ndarray:
    """Emit canonical u16 phase codes from current wide phase state."""

    values = _as_u64(phases, np.asarray(phases).size, name="phases")
    output = np.empty(values.size, dtype=np.uint16)
    status = _require_library().spf_sfr_ocean_emit_phase_u16(
        _pointer(values, _U64P), int(phase_bits), _pointer(output, _U16P), values.size
    )
    if status:
        raise ValueError(f"native ocean phase emission failed with status {status}")
    return output


def spf_error_ledger(
    words: npt.ArrayLike,
    indices: npt.ArrayLike,
    shape: Sequence[int],
    *,
    spacing: Sequence[float] | float = (1.0, 1.0),
    origin: Sequence[float] | float = (0.0, 0.0),
    footprint: Sequence[float] | float = (0.0, 0.0),
    normalize: bool = True,
    coordinate_bound: float | None = None,
) -> OceanErrorLedger:
    """Compute an executable absolute-error ledger for packed SPF reduction.

    This is a diagnostic calculation at the Python boundary.  It does not
    decode a complex source plane and is independent of any rendered samples.
    The table bound is the declared quarter-wave interpolation bound, while
    the fixed-point terms conservatively charge one Q30 unit per mode
    coefficient and product.  If the source or supported mode count cannot
    produce a finite/signed-safe bound, ``finite`` or ``int64_safe`` is false.
    """

    packed = _as_words(words)
    mode_indices = _as_indices(indices, packed.size)
    full_shape, spacing_pair, origin_pair, footprint_pair, _bits = _call_common(
        shape=shape,
        spacing=spacing,
        origin=origin,
        normalize=normalize,
        footprint=footprint,
        phase_bits=64,
    )
    height, width = full_shape
    logs = (packed >> np.uint32(16)).astype(np.uint16).view(np.int16).astype(np.int64)
    zero = logs == -32768
    invalid = logs == 32767
    active = ~(zero | invalid)
    signed_y = np.where(mode_indices[:, 0] <= height // 2, mode_indices[:, 0], mode_indices[:, 0] - height)
    signed_x = np.where(mode_indices[:, 1] <= width // 2, mode_indices[:, 1], mode_indices[:, 1] - width)
    self_conjugate = np.isin(mode_indices[:, 0], (0, height // 2) if height % 2 == 0 else (0,)) & np.isin(
        mode_indices[:, 1], (0, width // 2) if width % 2 == 0 else (0,)
    )
    pair = np.where(self_conjugate, 1.0, 2.0)
    raw_dy = 2.0 * math.pi * signed_y / (height * spacing_pair[0])
    raw_dx = 2.0 * math.pi * signed_x / (width * spacing_pair[1])
    dy = raw_dy.copy()
    dx = raw_dx.copy()
    nyquist_y = (
        np.abs(signed_y) == height // 2
        if height % 2 == 0
        else np.zeros_like(signed_y, dtype=bool)
    )
    nyquist_x = (
        np.abs(signed_x) == width // 2
        if width % 2 == 0
        else np.zeros_like(signed_x, dtype=bool)
    )
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        amplitude = np.exp2(logs.astype(np.float64) / 256.0)
        footprint_gain = np.exp(-0.5 * ((raw_dy * footprint_pair[0]) ** 2 + (raw_dx * footprint_pair[1]) ** 2))
    amplitude = np.where(active, amplitude, 0.0)
    footprint_gain = np.where(active, footprint_gain, 0.0)
    normalizer = 1.0 / float(height * width) if normalize else 1.0
    height_gain = np.sum(np.abs(amplitude * footprint_gain * pair), dtype=np.float64) * normalizer
    # Fractional point queries use the canonical cosine extension on a
    # Nyquist axis, so their derivative gain is the raw physical wavenumber.
    # The grid consumer still zeros Nyquist derivatives; charge the larger
    # point-domain gain in this shared ledger.
    gradient_y_gain = np.sum(np.abs(amplitude * footprint_gain * pair * dy), dtype=np.float64) * normalizer
    gradient_x_gain = np.sum(np.abs(amplitude * footprint_gain * pair * dx), dtype=np.float64) * normalizer

    # The C pivot includes pair weight and the larger of the height/dy/dx
    # gains.  A one-code guard keeps this Python ledger conservative at ties.
    pivot_terms: list[float] = []
    for position in np.flatnonzero(active):
        base = float(logs[position]) + (256.0 if pair[position] == 2.0 else 0.0)
        candidates = [base]
        for factor in (dy[position], dx[position]):
            if factor != 0.0:
                candidates.append(base + 256.0 * math.log2(abs(float(factor))))
        pivot_terms.append(max(candidates))
    pivot = max(pivot_terms, default=0.0)
    if pivot_terms:
        # C adds one Q8 code so rounded derivative gains cannot exceed the
        # signed Q30 pivot by a tie-rounding accident.
        pivot += 1.0
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        pivot_scale = math.exp2(pivot / 256.0) * normalizer
    active_count = int(np.count_nonzero(active))
    q30_unit = 2.0**-30
    phase_lut_unit = math.pi**2 / (8.0 * 1024.0**2) + q30_unit
    # A generic term needs one phase table lookup.  Each Nyquist coordinate
    # adds an independently rounded cosine/sine factor; charge the largest
    # applicable product order across the supplied modes.
    phase_factor_count = int(np.max(1 + nyquist_y.astype(np.int64) + nyquist_x.astype(np.int64), initial=1))
    phase_emission = (math.pi / 65536.0) * phase_factor_count
    phase_lut = phase_lut_unit * phase_factor_count
    if coordinate_bound is None:
        # The native point ABI accepts physical coordinates, while this
        # diagnostic bound is expressed in sample-coordinate units.  Include
        # the absolute source origin in that default domain so cancellation
        # and phase-composition rounding are not understated for translated
        # fields.  Callers rendering a larger ray domain can pass its maximum
        # absolute sample-coordinate magnitude explicitly.
        coordinate_extent = max(
            abs(origin_pair[0]) / spacing_pair[0] + height,
            abs(origin_pair[1]) / spacing_pair[1] + width,
        )
    else:
        coordinate_extent = float(coordinate_bound)
        if not math.isfinite(coordinate_extent) or coordinate_extent < 0.0:
            raise ValueError("coordinate_bound must be finite and non-negative")
    coordinate_term = np.abs(signed_y) / float(height) + np.abs(signed_x) / float(width)
    coordinate_bound_per_mode = 8.0 * np.finfo(np.float64).eps * 2.0 * math.pi * coordinate_extent * coordinate_term
    nyquist_coordinate_term = 0.5
    nyquist_coordinate_bound = (
        8.0 * np.finfo(np.float64).eps * 2.0 * math.pi * coordinate_extent * nyquist_coordinate_term
    )
    coordinate_phase = float(np.max(coordinate_bound_per_mode[active], initial=0.0))
    if np.any(nyquist_y | nyquist_x):
        coordinate_phase = max(coordinate_phase, nyquist_coordinate_bound)
    point_cycle_extent = coordinate_extent * coordinate_term
    point_precision_supported = bool(
        np.all(point_cycle_extent[active] <= _SPF_POINT_CYCLE_LIMIT)
    )
    if not point_precision_supported:
        # The C fallback rejects this domain rather than reducing a phase
        # whose low bits are no longer representable.  Reflect that guard in
        # the executable ledger instead of returning a finite-looking bound.
        coordinate_phase = math.inf
    if math.isfinite(coordinate_phase):
        phase_height = float(height_gain * (phase_emission + phase_lut + coordinate_phase))
        phase_gradient_y = float(gradient_y_gain * (phase_emission + phase_lut + coordinate_phase))
        phase_gradient_x = float(gradient_x_gain * (phase_emission + phase_lut + coordinate_phase))
    else:
        # The native evaluator rejects the whole query domain when a phase
        # contribution falls outside the precision guard.  Keep every
        # observable explicitly unbounded, including zero-gain components,
        # rather than allowing 0*inf to manufacture NaN.
        phase_height = math.inf
        phase_gradient_y = math.inf
        phase_gradient_x = math.inf
    exact_u16_grid = (
        height > 0
        and width > 0
        and height <= 65536
        and width <= 65536
        and (height & (height - 1)) == 0
        and (width & (width - 1)) == 0
    )
    # Non-power-of-two SPF grids use rounded Q32 spatial steps.  The maximum
    # accumulated phase drift over one row/column is charged independently of
    # the final u16 emission.  Direct points do not incur this term, so this
    # remains a conservative shared ledger value.
    recurrence_phase = 0.0 if exact_u16_grid else math.pi * (height + width - 2) / 2.0**32
    recurrence_height = height_gain * recurrence_phase
    recurrence_gradient_y = gradient_y_gain * recurrence_phase
    recurrence_gradient_x = gradient_x_gain * recurrence_phase
    pivot_factor = math.exp2(1.0 / 256.0)
    q30_rounding = 3.0 * active_count * q30_unit * pivot_scale * pivot_factor
    # Signed additions are exact; this is the explicit zero rounding term.
    accumulator_rounding = 0.0
    product_rounding = 3.0 * active_count * q30_unit * pivot_scale * pivot_factor
    max_gain = max(height_gain, gradient_y_gain, gradient_x_gain)
    # Include every preceding absolute contribution, including non-power
    # grid recurrence drift, before applying the final float32 binade bound.
    pre_output_bound = max(
        phase_height + recurrence_height,
        phase_gradient_y + recurrence_gradient_y,
        phase_gradient_x + recurrence_gradient_x,
    ) + q30_rounding + product_rounding
    if not math.isfinite(max_gain) or not math.isfinite(pre_output_bound):
        final_float = 0.0 if max_gain == 0.0 and pre_output_bound == 0.0 else math.inf
    else:
        # A float32 round can cross a binade after the preceding absolute
        # error has been added.  Use the largest permitted relative error and
        # retain the subnormal floor explicitly.
        final_float = (2.0**-24) * (max_gain + pre_output_bound) + 2.0**-150
    height_bound = phase_height + recurrence_height + q30_rounding + product_rounding + final_float
    gradient_y_bound = phase_gradient_y + recurrence_gradient_y + q30_rounding + product_rounding + final_float
    gradient_x_bound = phase_gradient_x + recurrence_gradient_x + q30_rounding + product_rounding + final_float
    max_q30_coeff = math.ceil((1 << 30) * pivot_factor)
    int64_safe = active_count <= np.iinfo(np.int64).max // max_q30_coeff
    finite = bool(
        not np.any(invalid)
        and np.all(np.isfinite((amplitude[active], footprint_gain[active], dy[active], dx[active])))
        and point_precision_supported
        and all(math.isfinite(value) for value in (
            height_bound,
            gradient_y_bound,
            gradient_x_bound,
            pivot_scale,
        ))
    )
    return OceanErrorLedger(
        mode_count=int(packed.size),
        active_modes=active_count,
        invalid_modes=int(np.count_nonzero(invalid)),
        phase_emission_bound=phase_emission,
        phase_lut_bound=phase_lut,
        coordinate_composition_bound=coordinate_phase,
        grid_recurrence_bound=max(recurrence_height, recurrence_gradient_y, recurrence_gradient_x),
        q30_coefficient_bound=q30_rounding,
        q30_product_bound=product_rounding,
        accumulator_rounding_bound=accumulator_rounding,
        final_float32_bound=final_float,
        height_absolute_bound=float(height_bound),
        gradient_y_absolute_bound=float(gradient_y_bound),
        gradient_x_absolute_bound=float(gradient_x_bound),
        int64_safe=bool(int64_safe),
        finite=finite,
    )


def _raise_status(status: int, *, kind: str) -> None:
    if status == 0:
        return
    if status == -4:
        raise ValueError(f"native {kind} ocean input contains a non-finite or invalid mode")
    raise ValueError(f"native {kind} ocean kernel failed with status {status}")


def _grid_call(
    *,
    kind: Literal["spf32", "fp32"],
    source: tuple[np.ndarray, ...],
    indices: np.ndarray,
    phases: np.ndarray,
    increments: np.ndarray,
    shape: tuple[int, int],
    spacing: tuple[float, float],
    origin: tuple[float, float],
    normalize: bool,
    footprint: tuple[float, float],
    phase_bits: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], OceanStats]:
    lib = _require_library()
    height, width = shape
    size = height * width
    output = tuple(np.empty(size, dtype=np.float32) for _ in range(3))
    stats = OceanStats()
    function = getattr(lib, f"spf_sfr_ocean_grid_{kind}")
    status = function(
        *[_pointer(array, _U32P if kind == "spf32" else _F32P) for array in source],
        _pointer(indices, _I64P),
        _pointer(phases, _U64P),
        _pointer(increments, _U64P),
        len(indices),
        phase_bits,
        height,
        width,
        spacing[0],
        spacing[1],
        origin[0],
        origin[1],
        int(bool(normalize)),
        footprint[0],
        footprint[1],
        *[_pointer(array, _F32P) for array in output],
        ctypes.pointer(stats),
    )
    _raise_status(int(status), kind=kind)
    shaped: tuple[np.ndarray, np.ndarray, np.ndarray] = (
        output[0].reshape(shape),
        output[1].reshape(shape),
        output[2].reshape(shape),
    )
    return (shaped, stats)


def grid_spf32(
    words: npt.ArrayLike,
    indices: npt.ArrayLike,
    shape: Sequence[int],
    *,
    phases: npt.ArrayLike | None = None,
    increments: npt.ArrayLike | None = None,
    phase_bits: int = 64,
    spacing: Sequence[float] | float = (1.0, 1.0),
    origin: Sequence[float] | float = (0.0, 0.0),
    normalize: bool = True,
    footprint: Sequence[float] | float = (0.0, 0.0),
    return_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], OceanStats]:
    """Evaluate packed SPF32 ocean modes on a shared height/gradient grid."""

    packed = _as_words(words)
    mode_indices = _as_indices(indices, packed.size)
    phase_values = _as_u64(phases, packed.size, name="phases")
    increment_values = _as_u64(increments, packed.size, name="increments")
    common = _call_common(
        shape=shape,
        spacing=spacing,
        origin=origin,
        normalize=normalize,
        footprint=footprint,
        phase_bits=phase_bits,
    )
    output, stats = _grid_call(
        kind="spf32",
        source=(packed,),
        indices=mode_indices,
        phases=phase_values,
        increments=increment_values,
        shape=common[0],
        spacing=common[1],
        origin=common[2],
        normalize=normalize,
        footprint=common[3],
        phase_bits=common[4],
    )
    return (output, stats) if return_stats else output


def grid_fp32(
    coefficients: npt.ArrayLike,
    indices: npt.ArrayLike,
    shape: Sequence[int],
    *,
    phases: npt.ArrayLike | None = None,
    increments: npt.ArrayLike | None = None,
    phase_bits: int = 64,
    spacing: Sequence[float] | float = (1.0, 1.0),
    origin: Sequence[float] | float = (0.0, 0.0),
    normalize: bool = True,
    footprint: Sequence[float] | float = (0.0, 0.0),
    return_stats: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], OceanStats]:
    """Evaluate the equally structured FP32 ocean control."""

    coeff_re, coeff_im = _as_fp32_coefficients(coefficients)
    mode_indices = _as_indices(indices, coeff_re.size)
    phase_values = _as_u64(phases, coeff_re.size, name="phases")
    increment_values = _as_u64(increments, coeff_re.size, name="increments")
    common = _call_common(
        shape=shape,
        spacing=spacing,
        origin=origin,
        normalize=normalize,
        footprint=footprint,
        phase_bits=phase_bits,
    )
    output, stats = _grid_call(
        kind="fp32",
        source=(coeff_re, coeff_im),
        indices=mode_indices,
        phases=phase_values,
        increments=increment_values,
        shape=common[0],
        spacing=common[1],
        origin=common[2],
        normalize=normalize,
        footprint=common[3],
        phase_bits=common[4],
    )
    return (output, stats) if return_stats else output


def _points_call(
    *,
    kind: Literal["spf32", "fp32"],
    source: tuple[np.ndarray, ...],
    indices: np.ndarray,
    phases: np.ndarray,
    increments: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray | None,
    shape: tuple[int, int],
    spacing: tuple[float, float],
    origin: tuple[float, float],
    normalize: bool,
    footprint: tuple[float, float],
    phase_bits: int,
) -> tuple[np.ndarray, OceanStats]:
    lib = _require_library()
    height, width = shape
    output = np.empty((points.shape[0], 3), dtype=np.float32)
    stats = OceanStats()
    function = getattr(lib, f"spf_sfr_ocean_points_{kind}")
    source_types = [_U32P if kind == "spf32" else _F32P for _ in source]
    status = function(
        *[_pointer(array, pointer_type) for array, pointer_type in zip(source, source_types, strict=True)],
        _pointer(indices, _I64P),
        _pointer(phases, _U64P),
        _pointer(increments, _U64P),
        len(indices),
        phase_bits,
        height,
        width,
        spacing[0],
        spacing[1],
        origin[0],
        origin[1],
        int(bool(normalize)),
        footprint[0],
        footprint[1],
        _pointer(points, _F64P),
        _pointer(visible, _U8P),
        points.shape[0],
        _pointer(output, _F32P),
        ctypes.pointer(stats),
    )
    _raise_status(int(status), kind=kind)
    return output, stats


def points_spf32(
    words: npt.ArrayLike,
    indices: npt.ArrayLike,
    points: npt.ArrayLike,
    *,
    phases: npt.ArrayLike | None = None,
    increments: npt.ArrayLike | None = None,
    phase_bits: int = 64,
    shape: Sequence[int] = (1, 1),
    spacing: Sequence[float] | float = (1.0, 1.0),
    origin: Sequence[float] | float = (0.0, 0.0),
    normalize: bool = True,
    footprint: Sequence[float] | float = (0.0, 0.0),
    visible: npt.ArrayLike | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, OceanStats]:
    """Evaluate packed SPF32 modes at physical ``(y, x)`` query points."""

    packed = _as_words(words)
    mode_indices = _as_indices(indices, packed.size)
    query_points = _as_points(points)
    phase_values = _as_u64(phases, packed.size, name="phases")
    increment_values = _as_u64(increments, packed.size, name="increments")
    visibility = _visible(visible, query_points.shape[0])
    common = _call_common(
        shape=shape,
        spacing=spacing,
        origin=origin,
        normalize=normalize,
        footprint=footprint,
        phase_bits=phase_bits,
    )
    output, stats = _points_call(
        kind="spf32",
        source=(packed,),
        indices=mode_indices,
        phases=phase_values,
        increments=increment_values,
        points=query_points,
        visible=visibility,
        shape=common[0],
        spacing=common[1],
        origin=common[2],
        normalize=normalize,
        footprint=common[3],
        phase_bits=common[4],
    )
    return (output, stats) if return_stats else output


def points_fp32(
    coefficients: npt.ArrayLike,
    indices: npt.ArrayLike,
    points: npt.ArrayLike,
    *,
    phases: npt.ArrayLike | None = None,
    increments: npt.ArrayLike | None = None,
    phase_bits: int = 64,
    shape: Sequence[int] = (1, 1),
    spacing: Sequence[float] | float = (1.0, 1.0),
    origin: Sequence[float] | float = (0.0, 0.0),
    normalize: bool = True,
    footprint: Sequence[float] | float = (0.0, 0.0),
    visible: npt.ArrayLike | None = None,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, OceanStats]:
    """Evaluate the structured FP32 control at physical query points."""

    coeff_re, coeff_im = _as_fp32_coefficients(coefficients)
    mode_indices = _as_indices(indices, coeff_re.size)
    query_points = _as_points(points)
    phase_values = _as_u64(phases, coeff_re.size, name="phases")
    increment_values = _as_u64(increments, coeff_re.size, name="increments")
    visibility = _visible(visible, query_points.shape[0])
    common = _call_common(
        shape=shape,
        spacing=spacing,
        origin=origin,
        normalize=normalize,
        footprint=footprint,
        phase_bits=phase_bits,
    )
    output, stats = _points_call(
        kind="fp32",
        source=(coeff_re, coeff_im),
        indices=mode_indices,
        phases=phase_values,
        increments=increment_values,
        points=query_points,
        visible=visibility,
        shape=common[0],
        spacing=common[1],
        origin=common[2],
        normalize=normalize,
        footprint=common[3],
        phase_bits=common[4],
    )
    return (output, stats) if return_stats else output


@dataclass(slots=True)
class NativeOceanSampler:
    """Persistent native sampler retaining source and callback storage.

    Construct with exactly one of ``words`` or ``coefficients``.  The C
    callback consumes current ``phases`` from the retained uint64 array;
    :meth:`advance` is the only method that applies ``increments``.  The
    ``context_ptr`` and ``function_ptr`` properties are safe to hand to a
    native display terminal while this object remains alive.
    """

    shape: tuple[int, int]
    _jet_caches: Any
    _source_transfer_ledger: Any
    indices: np.ndarray
    spacing: tuple[float, float]
    origin: tuple[float, float]
    normalize: bool
    footprint: tuple[float, float]
    phase_bits: int
    words: np.ndarray | None
    coeff_re: np.ndarray | None
    coeff_im: np.ndarray | None
    phases: np.ndarray
    increments: np.ndarray
    _stats: OceanStats
    _context: OceanContext
    _library: Any
    _sample: Any

    def __init__(
        self,
        *,
        shape: Sequence[int],
        indices: npt.ArrayLike,
        words: npt.ArrayLike | None = None,
        coefficients: npt.ArrayLike | None = None,
        phases: npt.ArrayLike | None = None,
        increments: npt.ArrayLike | None = None,
        phase_bits: int = 64,
        spacing: Sequence[float] | float = (1.0, 1.0),
        origin: Sequence[float] | float = (0.0, 0.0),
        normalize: bool = True,
        footprint: Sequence[float] | float = (0.0, 0.0),
    ) -> None:
        if (words is None) == (coefficients is None):
            raise ValueError("NativeOceanSampler requires exactly one of words or coefficients")
        common = _call_common(
            shape=shape,
            spacing=spacing,
            origin=origin,
            normalize=normalize,
            footprint=footprint,
            phase_bits=phase_bits,
        )
        if words is not None:
            packed = _as_words(words).copy()
            coeff_re = coeff_im = None
            kind = 0
            count = packed.size
        else:
            packed = None
            coeff_re, coeff_im = _as_fp32_coefficients(coefficients)  # type: ignore[arg-type]
            coeff_re = coeff_re.copy()
            coeff_im = coeff_im.copy()
            kind = 1
            count = coeff_re.size
        mode_indices = _as_indices(indices, count).copy()
        phase_values = _as_u64(phases, count, name="phases").copy()
        increment_values = _as_u64(increments, count, name="increments").copy()
        # A prepared callback intentionally observes only phase mutations.
        # Freeze source words/coefficients and index metadata so edits cannot
        # silently diverge from the cached per-mode reduction.
        mode_indices.setflags(write=False)
        increment_values.setflags(write=False)
        if packed is not None:
            packed.setflags(write=False)
        else:
            assert coeff_re is not None and coeff_im is not None
            coeff_re.setflags(write=False)
            coeff_im.setflags(write=False)
        lib = _require_library()
        stats = OceanStats()
        context = OceanContext()
        context.abi_version = 1
        context.kind = kind
        context.mode_count = count
        context.phase_bits = common[4]
        context.height, context.width = common[0]
        context.normalize = int(bool(normalize))
        context.reserved = 0
        context.spacing_y, context.spacing_x = common[1]
        context.origin_y, context.origin_x = common[2]
        context.footprint_sigma_y, context.footprint_sigma_x = common[3]
        if kind == 0:
            assert packed is not None
            context.words_or_re = packed.ctypes.data
            context.im = _F32P()
            sample = lib.spf_sfr_ocean_sample_spf32
        else:
            assert coeff_re is not None and coeff_im is not None
            context.words_or_re = coeff_re.ctypes.data
            context.im = coeff_im.ctypes.data_as(_F32P)
            sample = lib.spf_sfr_ocean_sample_fp32
        context.indices = mode_indices.ctypes.data_as(_I64P)
        context.phases = phase_values.ctypes.data_as(_U64P)
        context.increments = increment_values.ctypes.data_as(_U64P)
        context.stats = ctypes.pointer(stats)
        context.prepared_terms = None
        context.prepared_phase_hash = 0
        context.prepared_valid = 0
        context.reserved2 = 0
        status = lib.spf_sfr_ocean_context_prepare(
            ctypes.cast(ctypes.pointer(context), _VOIDP)
        )
        _raise_status(int(status), kind="context preparation")
        self.shape = common[0]
        self.indices = mode_indices
        self.spacing = common[1]
        self.origin = common[2]
        self.normalize = bool(normalize)
        self.footprint = common[3]
        self.phase_bits = common[4]
        self.words = packed
        self.coeff_re = coeff_re
        self.coeff_im = coeff_im
        self.phases = phase_values
        self.increments = increment_values
        self._stats = stats
        self._context = context
        self._library = lib
        self._sample = sample
        # Optional support-sized lowering certificate supplied by a retained
        # FilteredBlockView.  It is folded into error_ledger() so display
        # query wrappers (including NativeOceanJetSampler) retain the source
        # transfer error without knowing about the terminal's view object.
        self._source_transfer_ledger = None

    @classmethod
    def from_spf32(cls, words: npt.ArrayLike, indices: npt.ArrayLike, **kwargs: Any) -> NativeOceanSampler:
        return cls(words=words, indices=indices, **kwargs)

    @classmethod
    def from_fp32(
        cls, coefficients: npt.ArrayLike, indices: npt.ArrayLike, **kwargs: Any
    ) -> NativeOceanSampler:
        return cls(coefficients=coefficients, indices=indices, **kwargs)

    @property
    def context_ptr(self) -> ctypes.c_void_p:
        """Opaque context pointer for the native display callback ABI."""

        return ctypes.cast(ctypes.pointer(self._context), ctypes.c_void_p)

    @property
    def function_ptr(self) -> ctypes.c_void_p:
        """Function pointer for ``callback(void*, y, x, need, out[3])``."""

        return ctypes.cast(self._sample, ctypes.c_void_p)

    @property
    def callback(self) -> Any:
        """Configured ctypes callback function retaining its shared library."""

        return self._sample

    @property
    def stats(self) -> OceanStats:
        return self._stats

    @property
    def mode_count(self) -> int:
        return int(self.indices.shape[0])

    def phase_codes(self) -> np.ndarray:
        """Return canonical emitted phase codes for current state."""

        return emit_phase_codes(self.phases, self.phase_bits)

    def refresh(self) -> None:
        """Rebuild cached per-mode terms after an external phase mutation."""

        status = self._library.spf_sfr_ocean_context_refresh(self.context_ptr)
        _raise_status(int(status), kind="context refresh")
        # Invalidate derived local coefficients even for reset/advance replay
        # ending at the same phase. Animation timings must charge rebuilding
        # the current field representation, not reuse a previous trial's jets.
        for cache in getattr(self, "_jet_caches", ()):
            cache.invalidate()

    def mode_state(self) -> dict[str, np.ndarray | int]:
        """Expose retained words/coefficients, indices, and wide phase state."""

        result: dict[str, np.ndarray | int] = {
            "indices": self.indices,
            "phases": self.phases,
            "increments": self.increments,
            "phase_bits": self.phase_bits,
        }
        if self.words is not None:
            result["words"] = self.words
        else:
            result["coeff_re"] = self.coeff_re if self.coeff_re is not None else np.empty(0, np.float32)
            result["coeff_im"] = self.coeff_im if self.coeff_im is not None else np.empty(0, np.float32)
        return result

    def error_ledger(self, *, coordinate_bound: float | None = None) -> OceanErrorLedger:
        """Return the packed signed-reduction absolute-error ledger."""

        if self.words is None:
            raise TypeError("error_ledger is defined for packed SPF32 samplers")
        base = spf_error_ledger(
            self.words,
            self.indices,
            self.shape,
            spacing=self.spacing,
            origin=self.origin,
            footprint=self.footprint,
            normalize=self.normalize,
            coordinate_bound=coordinate_bound,
        )
        extra = getattr(self, "_source_transfer_ledger", None)
        if extra is None:
            return base
        return replace(
            base,
            height_absolute_bound=base.height_absolute_bound + float(extra.height_absolute_bound),
            gradient_y_absolute_bound=base.gradient_y_absolute_bound + float(extra.gradient_y_absolute_bound),
            gradient_x_absolute_bound=base.gradient_x_absolute_bound + float(extra.gradient_x_absolute_bound),
            finite=bool(base.finite and math.isfinite(float(extra.output_error_bound))),
        )

    def advance(self, steps: int = 1) -> np.ndarray:
        """Advance retained master phases modulo their declared width."""

        if isinstance(steps, (bool, np.bool_)):
            raise TypeError("native ocean phase steps must be an integer")
        integer = int(steps)
        if isinstance(steps, (float, np.floating)) and float(steps) != integer:
            raise TypeError("native ocean phase steps must be an integer")
        bits = self.phase_bits
        modulus = 1 << bits
        step = integer % modulus
        if self.mode_count:
            delta = np.multiply(self.increments, np.uint64(step), dtype=np.uint64)
            np.add(self.phases, delta, out=self.phases, dtype=np.uint64)
            if bits < 64:
                self.phases &= np.uint64(modulus - 1)
        self.refresh()
        return self.phases

    def close(self) -> None:
        """Release native per-mode cache while retaining Python source arrays."""

        context = getattr(self, "_context", None)
        library_object = getattr(self, "_library", None)
        if context is None or library_object is None:
            return
        if int(context.prepared_valid):
            library_object.spf_sfr_ocean_context_release(self.context_ptr)

    def __enter__(self) -> NativeOceanSampler:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    def sample(self, y: float, x: float, *, need_gradients: bool = True) -> np.ndarray:
        """Evaluate one physical ``(y, x)`` query through the C callback."""

        output = np.empty(3, dtype=np.float64)
        status = self._sample(
            self.context_ptr,
            float(y),
            float(x),
            int(bool(need_gradients)),
            _pointer(output, _F64P),
        )
        _raise_status(int(status), kind="callback")
        return output

    def grid(
        self, *, return_stats: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], OceanStats]:
        """Evaluate the full grid through the matching native source path."""

        if self.words is not None:
            return grid_spf32(
                self.words,
                self.indices,
                self.shape,
                phases=self.phases,
                increments=self.increments,
                phase_bits=self.phase_bits,
                spacing=self.spacing,
                origin=self.origin,
                normalize=self.normalize,
                footprint=self.footprint,
                return_stats=return_stats,
            )
        assert self.coeff_re is not None and self.coeff_im is not None
        coefficients = np.column_stack((self.coeff_re, self.coeff_im))
        return grid_fp32(
            coefficients,
            self.indices,
            self.shape,
            phases=self.phases,
            increments=self.increments,
            phase_bits=self.phase_bits,
            spacing=self.spacing,
            origin=self.origin,
            normalize=self.normalize,
            footprint=self.footprint,
            return_stats=return_stats,
        )

    def points(
        self,
        points: npt.ArrayLike,
        *,
        visible: npt.ArrayLike | None = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, OceanStats]:
        """Evaluate visible physical ``(y, x)`` points through native C."""

        if self.words is not None:
            return points_spf32(
                self.words,
                self.indices,
                points,
                phases=self.phases,
                increments=self.increments,
                phase_bits=self.phase_bits,
                shape=self.shape,
                spacing=self.spacing,
                origin=self.origin,
                normalize=self.normalize,
                footprint=self.footprint,
                visible=visible,
                return_stats=return_stats,
            )
        assert self.coeff_re is not None and self.coeff_im is not None
        coefficients = np.column_stack((self.coeff_re, self.coeff_im))
        return points_fp32(
            coefficients,
            self.indices,
            points,
            phases=self.phases,
            increments=self.increments,
            phase_bits=self.phase_bits,
            shape=self.shape,
            spacing=self.spacing,
            origin=self.origin,
            normalize=self.normalize,
            footprint=self.footprint,
            visible=visible,
            return_stats=return_stats,
        )


def compile_library(output: str | os.PathLike[str] | None = None) -> Path:
    """Build the standalone native ocean shared library once."""

    root = Path(__file__).resolve().parents[3]
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
        str(root / "native" / "include"),
        str(root / "native" / "src" / "scalar" / "spf32" / "sfr_ocean.c"),
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
    "NativeOceanSampler",
    "OceanContext",
    "OceanErrorLedger",
    "OceanStats",
    "available",
    "compile_library",
    "emit_phase_codes",
    "grid_fp32",
    "grid_spf32",
    "library",
    "library_path",
    "points_fp32",
    "points_spf32",
    "spf_error_ledger",
]


if __name__ == "__main__":
    main()
