"""Joint, structure preserving ocean terminals.

``PreparedOcean`` owns an immutable carrier and mutable wide phase state.  A
frame walks the selected source once and emits height and both physical
gradients together.  Sparse block inputs stay as addressed SPF words until the
reference terminal reaches its numeric output boundary; dense FFT inputs are
materialised only after the planner selected that route.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf.renderer.carriers import decode_words
from spf.renderer.ocean_carrier import OceanCarrier
from spf.renderer.ocean_planner import (
    OceanExecutionPlan,
    OceanOutputRequest,
    bind_ocean_execution,
)
from spf.renderer.ocean_planner import _carrier_source_transfer as _source_transfer_operations
from spf.renderer.ocean_transfer import (
    FilteredSourceLowering,
    lower_filtered_source,
    lower_spf32_gaussian_footprint,
)
from spf.tensor import v02


@dataclass(frozen=True, slots=True)
class OceanFieldResult:
    """Height/gradient result from one joint ocean terminal invocation."""

    height: np.ndarray | None
    grad_x: np.ndarray | None
    grad_y: np.ndarray | None
    report: Any
    output_dtype: str
    points: np.ndarray | None = None
    footprint: Any = None
    operations: tuple[Any, ...] = ()
    trace: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    @property
    def gradients(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        return self.grad_x, self.grad_y

    @property
    def values(self) -> np.ndarray | None:
        return self.height


@dataclass(slots=True)
class _Counters:
    source_reads: int = 0
    packed_words_read: int = 0
    phase_updates: int = 0
    phase_emissions: int = 0
    term_evaluations: int = 0
    real_adds: int = 0
    complex_adds: int = 0
    fft_calls: int = 0
    dense_carrier_elements: int = 0
    materialized_intermediate_elements: int = 0
    materialized_output_elements: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    output_fields: int = 0
    reduction_operations: int = 0
    fallback_conversions: int = 0
    trace: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = []

    def record(self, opcode: str, **values: Any) -> None:
        assert self.trace is not None
        self.trace.append({"opcode": opcode, **values})

    def invoke(self, function: Callable[..., Any], executor: str, *args: Any, **kwargs: Any) -> Any:
        """Invoke and trace the same callable, independently of its planned name."""

        kernel = f"{function.__module__}.{function.__qualname__}"
        result = function(*args, **kwargs)
        self.record("BoundExecution", executor=executor, kernel=kernel)
        return result


def _dtype(value: Any, default: Any = np.float32) -> np.dtype:
    result = np.dtype(default if value is None else value)
    if result.kind != "f" or result.itemsize not in {4, 8}:
        raise TypeError("ocean output_dtype must be numpy.float32 or numpy.float64")
    return cast(np.dtype[Any], result)


def _spacing(shape: tuple[int, ...], value: Any) -> tuple[float, ...]:
    if value is None:
        return (1.0,) * len(shape)
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(len(shape), float(array), dtype=np.float64)
    if array.shape != (len(shape),) or np.any(~np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("ocean sample spacing must be finite, positive, and match the shape")
    return tuple(float(x) for x in array)


def _validate_points(points: Any, dimension: int) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim == 1:
        if array.size != dimension:
            raise ValueError("ocean point coordinates must have trailing dimension equal to the field rank")
        array = array.reshape(1, dimension)
    if array.ndim != 2 or array.shape[1] != dimension:
        raise ValueError("ocean point coordinates must have shape (Q, dimension)")
    if not np.all(np.isfinite(array)):
        raise ValueError("ocean point coordinates must be finite")
    return array


def _signed_frequency(index: int, size: int) -> int:
    return int(index if index <= size // 2 else index - size)


def _footprint_factor(
    footprint: Any, index: Sequence[int], shape: tuple[int, ...], spacing: tuple[float, ...]
) -> float:
    if footprint is None:
        return 1.0
    covariance = getattr(footprint, "covariance", None)
    if covariance is None and isinstance(footprint, Mapping):
        covariance = footprint.get("covariance")
    if covariance is not None:
        matrix = _validated_covariance(covariance, len(shape))
        frequencies = np.asarray(
            [_signed_frequency(i, n) / (n * d) for i, n, d in zip(index, shape, spacing, strict=True)]
        )
        # GaussianFootprint uses [fx, fy] while an ocean array uses (y, x).
        if len(shape) == 2:
            frequencies = frequencies[::-1]
        exponent = -2.0 * math.pi**2 * float(frequencies @ matrix @ frequencies)
        return float(math.exp(exponent)) if exponent > -745.0 else 0.0
    transfer = getattr(footprint, "frequency_transfer", None)
    if callable(transfer):
        frequency = tuple(_signed_frequency(i, n) / (n * d) for i, n, d in zip(index, shape, spacing, strict=True))
        value = transfer(frequency)
        return float(value)
    raise TypeError("ocean footprint must provide covariance or frequency_transfer(frequency)")


def _validated_covariance(covariance: Any, dimension: int) -> np.ndarray:
    """Validate a Gaussian footprint covariance before any route consumes it."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"ocean footprint covariance must be finite {dimension}x{dimension}")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("ocean footprint covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix), initial=0.0) < -1.0e-12:
        raise ValueError("ocean footprint covariance must be positive semidefinite")
    return matrix


def _cache_value(value: Any) -> Any:
    """Make the small footprint descriptors safe and value keyed for caching."""

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return ("array", array.shape, array.dtype.str, array.tobytes())
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(sorted((str(key), _cache_value(item)) for key, item in value.items())),
        )
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, tuple(_cache_value(item) for item in value))
    if callable(value):
        return ("callable", type(value).__module__, type(value).__qualname__, id(value))
    try:
        hash(value)
    except TypeError:
        return (type(value).__module__, type(value).__qualname__, repr(value))
    return value


def _footprint_cache_key(footprint: Any) -> Any:
    """Key prepared analytic consumers by footprint values, never object id alone."""

    if footprint is None:
        return None
    if isinstance(footprint, Mapping):
        return _cache_value(footprint)
    covariance = getattr(footprint, "covariance", None)
    if covariance is not None:
        return (type(footprint).__module__, type(footprint).__qualname__, _cache_value(np.asarray(covariance)))
    transfer = getattr(footprint, "frequency_transfer", None)
    if callable(transfer):
        return (type(footprint).__module__, type(footprint).__qualname__, _cache_value(transfer))
    return _cache_value(footprint)


def _view_offsets(carrier: OceanCarrier) -> tuple[float, float]:
    """Read lazy BlockView offsets retained by the structural carrier."""

    view = carrier.source_view
    if view is None:
        return 0.0, 0.0
    log_offset = float(getattr(view, "log_offset", 0.0))
    phase_offset = float(getattr(view, "phase_offset", 0.0))
    if not np.isfinite(log_offset) or not np.isfinite(phase_offset):
        raise ValueError("ocean block offsets must be finite")
    return log_offset, phase_offset


def _view_has_fractional_offsets(carrier: OceanCarrier) -> bool:
    """Whether a BlockView offset needs the numeric reference boundary.

    The name is retained for compatibility with the existing terminal code;
    out-of-range integral log shifts have the same wide semantics as
    fractional offsets and are included here.
    """

    log_offset, phase_offset = _view_offsets(carrier)
    if log_offset != round(log_offset) or phase_offset != round(phase_offset):
        return True
    if carrier.source_kind != "block_view" or carrier.source_view is None or log_offset == 0.0:
        return False
    from spf.renderer.ocean_carrier import _view_offsets_require_numeric

    view = carrier.source_view
    addresses = np.asarray(view.active_addresses(), dtype=np.int64).reshape(-1)
    if not addresses.size:
        return False
    words = view.sample_words(addresses, phase=True)
    return _view_offsets_require_numeric(words, view)


def _view_numeric_factor(carrier: OceanCarrier) -> complex:
    """Return the exact numeric factor for a non-integral lazy view offset."""

    log_offset, phase_offset = _view_offsets(carrier)
    return complex(
        math.exp2(log_offset / 256.0) * math.cos(phase_offset * (2.0 * math.pi / v02.SPF32_PHASE_MOD)),
        math.exp2(log_offset / 256.0) * math.sin(phase_offset * (2.0 * math.pi / v02.SPF32_PHASE_MOD)),
    )


def _phase_value_to_code(value: Any) -> int:
    """Emit one legacy phase value into the native u16 code domain."""

    from spf.renderer.ocean import WidePhaseAccumulator

    if isinstance(value, WidePhaseAccumulator):
        return int(value.phase_code) & v02.SPF32_PHASE_MASK
    if isinstance(value, v02.Phase):
        return int(value.T) & v02.SPF32_PHASE_MASK
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return int(value) & v02.SPF32_PHASE_MASK
    radians = float(value)
    if not math.isfinite(radians):
        raise ValueError("ocean phase values must be finite")
    return round(radians * v02.SPF32_PHASE_MOD / (2.0 * math.pi)) & v02.SPF32_PHASE_MASK


def _dirichlet_factor(values: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the real periodic Dirichlet kernel and its derivative.

    A constant Hermitian half-plane extends to a constant full spectrum.  Its
    continuous real interpolant is therefore the cosine Dirichlet kernel,
    including ``cos(pi*q)`` for an even Nyquist axis.  The sinc forms below
    avoid constructing all frequency modes and the local series supplies the
    removable limit at integer periods.
    """

    coordinates = np.asarray(values, dtype=np.float64)
    reduced = coordinates - float(size) * np.rint(coordinates / float(size))
    denominator = np.sin(np.pi * reduced / float(size))
    regular = np.abs(denominator) > 1.0e-7
    result = np.empty_like(reduced)
    derivative = np.empty_like(reduced)
    q = reduced[regular]
    if size % 2:
        numerator = np.sin(np.pi * q)
        denominator_regular = denominator[regular]
        result[regular] = numerator / denominator_regular
        derivative[regular] = (
            np.pi * np.cos(np.pi * q) * denominator_regular
            - (np.pi / float(size)) * numerator * np.cos(np.pi * q / float(size))
        ) / (denominator_regular * denominator_regular)
    else:
        numerator = np.sin(np.pi * q)
        denominator_regular = denominator[regular]
        cotangent = np.cos(np.pi * q / float(size)) / denominator_regular
        result[regular] = numerator * cotangent
        derivative[regular] = (
            np.pi * np.cos(np.pi * q) * cotangent - (np.pi / float(size)) * numerator / denominator_regular**2
        )
    near = ~regular
    q_near = reduced[near]
    # The first non-zero Taylor coefficient is sufficient for the tiny
    # removable neighbourhood and is much more accurate than subtracting two
    # nearly equal trigonometric quotients there.
    q2 = q_near * q_near
    if size % 2:
        coefficient = 1.0 / 3.0 * (1.0 - 1.0 / float(size * size))
    else:
        coefficient = 1.0 / 3.0 + 2.0 / (3.0 * float(size * size))
    result[near] = float(size) * (1.0 - 0.5 * np.pi**2 * coefficient * q2)
    derivative[near] = -float(size) * np.pi**2 * coefficient * q_near
    return np.asarray(result, dtype=np.float64), np.asarray(derivative, dtype=np.float64)


def _geometric_factor(values: Any, size: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``sum(exp(i*2π*k*q/N))`` and its q derivative for k=1..count."""

    coordinates = np.asarray(values, dtype=np.float64)
    if count <= 0:
        zeros = np.zeros_like(coordinates, dtype=np.complex128)
        return zeros, zeros.copy()
    reduced = coordinates - float(size) * np.rint(coordinates / float(size))
    denominator = np.sin(np.pi * reduced / float(size))
    numerator = np.sin(np.pi * float(count) * reduced / float(size))
    regular = np.abs(denominator) > 1.0e-8
    ratio = np.empty_like(reduced)
    ratio[regular] = numerator[regular] / denominator[regular]
    ratio[~regular] = (
        float(count)
        * np.sinc(float(count) * reduced[~regular] / float(size))
        / np.sinc(reduced[~regular] / float(size))
    )
    center = 0.5 * (count + 1.0)
    phase = np.exp(1j * (2.0 * np.pi * center / float(size)) * reduced)
    result = phase * ratio
    derivative_ratio = np.zeros_like(reduced)
    derivative_ratio[regular] = (
        (np.pi * float(count) / float(size))
        * np.cos(np.pi * float(count) * reduced[regular] / float(size))
        * denominator[regular]
        - (np.pi / float(size)) * numerator[regular] * np.cos(np.pi * reduced[regular] / float(size))
    ) / (denominator[regular] * denominator[regular])
    derivative = phase * (derivative_ratio + 1j * (2.0 * np.pi * center / float(size)) * ratio)
    return np.asarray(result, dtype=np.complex128), np.asarray(derivative, dtype=np.complex128)


def _mode_base(o: Any, value: Any) -> complex:
    return cast(complex, o._coefficient_complex(value))


def _emit_phase_codes(values: np.ndarray, bits: int) -> np.ndarray:
    """Emit the canonical u16 view of homogeneous uint64 phase state.

    The master timeline is kept in a uint64 array for the common case.  This
    vectorized emitter has the same round-to-nearest-even rule as
    :attr:`WidePhaseAccumulator.phase_code`, while avoiding an object-array
    conversion on every frame.  Widths outside uint64 are handled by the
    compatibility path in :class:`PreparedOcean`.
    """

    array = np.asarray(values, dtype=np.uint64)
    width = int(bits)
    if width < 1 or width > 64:
        raise ValueError("wide phase width must be in [1, 64]")
    if width == 16:
        return np.asarray(array & np.uint64(0xFFFF), dtype=np.uint16)
    if width > 16:
        shift = width - 16
        divisor = 1 << shift
        quotient = array >> np.uint64(shift)
        remainder = array & np.uint64(divisor - 1)
        twice = remainder * np.uint64(2)
        round_up = (twice > np.uint64(divisor)) | (
            (twice == np.uint64(divisor)) & ((quotient & np.uint64(1)) == np.uint64(1))
        )
        return np.asarray((quotient + round_up.astype(np.uint64)) & np.uint64(0xFFFF), dtype=np.uint16)
    return np.asarray((array << np.uint64(16 - width)) & np.uint64(0xFFFF), dtype=np.uint16)


def _decode_words_for_dtype(words: np.ndarray, complex_dtype: np.dtype) -> np.ndarray:
    """Decode packed words directly into the requested complex width.

    ``decode_words`` is the normative complex128 compatibility decoder.  A
    float32 dense route must not allocate a full complex128 plane and cast it
    after the FFT was already selected, so the narrow branch decodes one word
    at a time into a complex64 destination.
    """

    packed = np.asarray(words, dtype=np.uint32)
    if complex_dtype == np.dtype(np.complex128):
        return np.asarray(decode_words(packed), dtype=complex_dtype)
    # Decode one leading-axis slab at a time: the complex128 temporary is
    # bounded by a single row, never the whole plane.  (Measured 2026-09-17:
    # the previous one-word-at-a-time loop cost 1.4 s per 256² half plane.)
    result = np.empty(packed.shape, dtype=complex_dtype)
    flat_out = result.reshape(packed.shape[0] if packed.ndim else 1, -1)
    flat_in = packed.reshape(packed.shape[0] if packed.ndim else 1, -1)
    for row in range(flat_in.shape[0]):
        flat_out[row] = np.asarray(decode_words(flat_in[row]), dtype=complex_dtype)
    return result


def _compose_half_words(
    words: np.ndarray, indices: np.ndarray, phase_codes: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray:
    """Place phase-shifted mode words into a Hermitian SPF32 half plane.

    A mode whose last index lies past the Nyquist column is stored at its
    partner as a conjugate; a destination on an endpoint column (0 or N/2)
    mirrors its conjugate onto the ``-ky`` row.  This is the Python word
    oracle for ``spf_sfr_ocean_packed_phase_advance`` in the C terminal.
    """

    from spf.renderer import ocean as o

    half = np.full(o._half_shape(shape), np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32)
    for index, word, phase_code in zip(indices, words, phase_codes, strict=True):
        shifted = int(v02.spf_phase_shift(int(word), int(phase_code)))
        index_tuple = tuple(int(x) for x in index)
        partner = o._partner(index_tuple, shape)
        if int(index[-1]) > shape[-1] // 2:
            destination = partner
            stored = int(v02.spf_conj(shifted))
        else:
            destination = index_tuple
            stored = shifted
        half[destination] = np.uint32(stored)
        if o._endpoint_last(destination, shape) and destination != o._partner(destination, shape):
            half[o._partner(destination, shape)] = np.uint32(v02.spf_conj(stored))
    return half


def _half_destinations(indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Flat half-plane addresses written by ``_compose_half_words`` for each mode."""

    rows, cols = int(shape[0]), int(shape[1])
    index_array = np.asarray(indices, dtype=np.int64).reshape(-1, 2)
    upper = index_array[:, 1] > cols // 2
    dy = np.where(upper, (rows - index_array[:, 0]) % rows, index_array[:, 0])
    dx = np.where(upper, (cols - index_array[:, 1]) % cols, index_array[:, 1])
    return dy * (cols // 2 + 1) + dx


def _hermitian_complete_words(half: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Complete a packed half plane into the full Hermitian plane (Python word oracle)."""

    from spf.renderer import ocean as o

    full = np.full(shape, np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32)
    full[..., : shape[-1] // 2 + 1] = half
    for index in np.ndindex(shape):
        if index[-1] <= shape[-1] // 2:
            continue
        partner = o._partner(index, shape)
        full[index] = np.uint32(v02.spf_conj(int(full[partner])))
    return full


def _derivative_words(full: np.ndarray, shape: tuple[int, ...], spacing: tuple[float, ...], axis: int) -> np.ndarray:
    """Spectral derivative as a packed log shift and quarter turn (Python word oracle)."""

    derivative = np.empty_like(full)
    for index in np.ndindex(shape):
        freq = _signed_frequency(index[axis], shape[axis])
        if shape[axis] % 2 == 0 and abs(freq) == shape[axis] // 2:
            derivative[index] = np.uint32(v02.SPF32_WORD_ZERO)
            continue
        magnitude = abs(2.0 * math.pi * freq / (shape[axis] * spacing[axis]))
        source_word = int(full[index])
        if magnitude == 0.0 or v02.is_zero(source_word):
            derivative[index] = np.uint32(v02.SPF32_WORD_ZERO)
            continue
        log_shift = round(256.0 * math.log2(magnitude))
        log_code = int(v02.unpack_L(source_word)) + log_shift
        phase = int(v02.unpack_T(source_word)) + (v02.SPF32_PHASE_QUARTER if freq > 0 else 3 * v02.SPF32_PHASE_QUARTER)
        log_code = max(v02.SPF32_L_MIN_FINITE, min(v02.SPF32_L_MAX_FINITE, log_code))
        derivative[index] = np.uint32((log_code & 0xFFFF) << 16 | (phase & 0xFFFF))
    return derivative


class PreparedOcean:
    """Owned carrier plus a joint height/gradient execution terminal."""

    def __init__(
        self,
        shape: tuple[int, ...],
        half: np.ndarray | None,
        plan: Any,
        *,
        normalize: bool,
        output_kind: Any,
        backend: Any,
        tolerance: float,
        modes: Any = (),
        terms: Any = (),
        carrier: OceanCarrier | None = None,
        spacing: Sequence[float] | None = None,
    ) -> None:
        from spf.renderer import ocean as o

        self.shape = tuple(int(x) for x in shape)
        self.plan: OceanExecutionPlan = plan
        self.normalize = bool(normalize)
        self.output_kind = output_kind
        self.backend = backend
        self.tolerance = float(tolerance)
        self.spacing = _spacing(self.shape, spacing)
        self.half_shape = o._half_shape(self.shape)
        self.half = None if half is None else np.array(half, copy=True)
        if self.half is not None:
            self.half.setflags(write=False)
        self.modes = tuple(modes)
        self.terms = tuple(terms)
        if carrier is None:
            carrier = OceanCarrier.from_inputs(
                self.shape,
                half_spectrum=self.half,
                coefficients=tuple(t[1] for t in self.terms) if self.terms else None,
                indices=tuple(t[0] for t in self.terms) if self.terms else None,
                modes=self.modes,
            )
        self.carrier = carrier
        self.indices: tuple[tuple[int, ...], ...] = tuple(carrier.index_values)
        self._mode_indices = tuple(o._normalise_index(m.index, self.shape) for m in self.modes)
        self._mode_base = tuple(_mode_base(o, m.coefficient) for m in self.modes)
        self.base_coefficients = np.asarray(self._mode_base, dtype=np.complex128)
        if not self.modes and self.terms:
            self.indices = tuple(t[0] for t in self.terms)
            self.base_coefficients = np.asarray([complex(t[1]) for t in self.terms], dtype=np.complex128)
        mode_bits = tuple(int(m.phase_accumulator.bits) for m in self.modes)
        self.bits = 0 if self.modes and len(set(mode_bits)) != 1 else (mode_bits[0] if self.modes else 64)
        # A homogeneous <=64-bit mode set owns a compact uint64 timeline.
        # OceanMode objects remain a compatibility view and are synchronized
        # only at explicit API boundaries (``sync_phases``/``set_phase_state``).
        self._phase_fastpath = bool(self.modes and self.bits <= 64 and self.bits >= 1)
        self._phase_values_u64 = (
            np.asarray([int(m.phase_accumulator.phase) for m in self.modes], dtype=np.uint64)
            if self._phase_fastpath
            else None
        )
        self._phase_increments_u64 = (
            np.asarray([int(m.phase_accumulator.increment) for m in self.modes], dtype=np.uint64)
            if self._phase_fastpath
            else None
        )
        if self._phase_fastpath:
            # These arrays are intentionally writable for the historical
            # ``prepared.phases[:] = ...`` benchmark/control API.  They are
            # the authoritative timeline for execution.
            self.phases = self._phase_values_u64
            self.increments = self._phase_increments_u64
        else:
            self.phases = np.asarray([m.phase_accumulator.phase for m in self.modes], dtype=object)
            self.increments = np.asarray([m.phase_accumulator.increment for m in self.modes], dtype=object)
        self.wraps = np.zeros(len(self.modes), dtype=np.int64)
        self._pending_phase_updates = 0
        self._native_sampler: Any | None = None
        self._native_sampler_key: tuple[Any, ...] | None = None
        self._native_transfer_lowering: FilteredSourceLowering | None = None
        self._native_transfer_lowering_key: tuple[Any, ...] | None = None
        self._native_footprint_lowering: FilteredSourceLowering | None = None
        self._native_footprint_lowering_key: tuple[Any, ...] | None = None
        # Analytic carriers retain their source object and are prepared lazily
        # at the canonical analytic boundary.  Keep that prepared consumer on
        # the terminal so a field/image query does not repeat preparation.
        self._analytic_prepared_cache: dict[tuple[Any, ...], Any] = {}
        # Packed C grid terminal state: the static half plane it advances, the
        # caller's native preference for the current frame, and which executor
        # (``"c"`` or ``"python"``) actually ran the last packed route.
        self._packed_base_cache: tuple[Any, np.ndarray] | None = None
        self._packed_use_native: bool | None = None
        self._packed_executor: str | None = None
        self.weights = np.asarray(
            [1 if o._is_self_conjugate(i, self.shape) else 2 for i in self.indices], dtype=np.float64
        )
        self._last_report: Any = None
        self.report = o._ocean_render_report(
            backend=getattr(backend, "name", str(backend)),
            shape=self.shape,
            method=self.plan.method,
            modes=self.plan.active_modes,
            fft_count=0,
            fallback=self.plan.fallback,
            diagnostics={
                "prepared": True,
                "joint_terminal": True,
                "execution_class": getattr(
                    getattr(backend, "capabilities", None), "execution_class", "reference_python"
                ),
                "hardware_native": False,
                "backend_execution_provenance": "prepared_joint_ocean_consumer",
                "requested_method": self.plan.requested_method,
                "route": self.plan.route,
                "profile": self.plan.profile,
                "consumer": self._consumer_name(),
                "output_dtype": "float64",
                "cached_spatial_pixels": 0,
                "carrier_bytes": int(self.carrier.metadata.get("carrier_nbytes", 0)),
                "dense_carrier_allocated": self.half is not None,
                "structure_preserved": self.plan.structure_preserved,
                "source_metadata": dict(self.carrier.metadata),
                "structural_program": None if self.plan.program is None else self.plan.program.to_dict(),
            },
        )
        # Keep the metadata-only source choice separate from the request-bound
        # plan.  Rebinding each frame starts from the same immutable recipe and
        # therefore cannot accumulate display/native route decisions.
        self._candidate_plan = plan

    def _consumer_name(self) -> str:
        return {
            "sparse": "packed_sparse_reference"
            if self.carrier.source_kind == "block_view"
            else "separable_real_fourier_reference",
            "irfft": "scipy_irfftn_reference",
            "fb3": "continuity_packed_ifft2_reference",
            "analytic": "analytic_atom_joint_reference",
        }.get(self.plan.method, "ocean_reference")

    @property
    def structural_program(self) -> Any:
        return self.plan.program

    @property
    def layout(self) -> Any:
        """Expose the analytic raster origin used by the terminal fallback.

        A public ``PreparedOcean`` otherwise has no layout object because its
        packed routes synthesize from the zero-origin array.  Spectral jet
        tiles carry a physical source origin, however, and the display must
        receive that same origin when it consumes the explicit grid fallback.
        """

        if self.carrier.source_kind != "analytic":
            return None
        payload = self.carrier.analytic_payload
        if payload is None:
            return None
        from spf.renderer.model import Field, SampleLayout

        tile_source = payload
        if isinstance(payload, Field):
            tile_source = getattr(payload.representation, "payload", payload)
        if type(tile_source).__name__ == "SpectralJetTile2D":
            tile_origin = getattr(tile_source, "origin", (0.0, 0.0))
            return SampleLayout(
                self.shape,
                origin=(float(tile_origin[1]), float(tile_origin[0])),
                spacing=self.spacing,
            )
        analytic = self._canonical_analytic_prepared(output_dtype=np.float64)
        return getattr(analytic, "layout", None)

    @property
    def operations(self) -> tuple[Any, ...]:
        return self.plan.operations

    def bind_execution(
        self,
        request: OceanOutputRequest | Mapping[str, Any],
        *,
        native_available: bool = False,
        use_native: bool | None = None,
    ) -> OceanExecutionPlan:
        """Bind a complete output request and make it the terminal plan."""

        if not isinstance(request, OceanOutputRequest):
            request = OceanOutputRequest(**dict(request))
        native_preference = request.native if use_native is None else use_native
        native_supported = self.native_queries_supported
        if request.footprint is not None and native_preference is not False:
            native_supported = native_supported and self.native_footprint_supported(request.footprint)
        backend_name = request.backend
        backend_allows_native = backend_name in {"cpu", "numpy"} or bool(
            getattr(getattr(self.backend, "capabilities", None), "native", False)
        )
        bound = bind_ocean_execution(
            self.carrier,
            self._candidate_plan,
            request,
            native_supported=native_supported,
            native_available=native_available,
            backend_allows_native=backend_allows_native,
            use_native=use_native,
            native_representation=self._native_representation(),
        )
        self.plan = bound
        return bound

    def bind_display_execution(
        self,
        resolution: tuple[int, int],
        intersections: int,
        *,
        sampling: str = "auto",
        footprint: Any = None,
        output_dtype: Any = np.float32,
    ) -> tuple[str, OceanExecutionPlan, Mapping[str, Any], str | None]:
        """Choose display sampling and bind that choice to this terminal."""

        selected_sampling = str(sampling).strip().lower().replace("-", "_")
        selected_sampling = {"query": "queries", "points": "queries"}.get(selected_sampling, selected_sampling)
        if selected_sampling not in {"auto", "queries", "grid"}:
            raise ValueError("ocean image sampling must be auto, queries, or grid")
        if len(resolution) != 2 or any(int(value) != value or int(value) < 1 for value in resolution):
            raise ValueError("display resolution must be two positive integers")
        grid_samples = int(math.prod(self.shape))
        query_samples = int(math.prod(resolution)) * (int(intersections) + 1)
        active = max(1, int(self._candidate_plan.active_modes))
        grid_cost = min(float(self._candidate_plan.sparse_cost), float(self._candidate_plan.dense_cost))
        query_cost = float(query_samples * active)
        costs: Mapping[str, Any] = MappingProxyType(
            {
                "grid_samples": grid_samples,
                "query_samples_upper_bound": query_samples,
                "active_modes": active,
                "grid_operation_estimate": grid_cost,
                "query_operation_estimate": query_cost,
                "model": "operation_count_heuristic_not_measured_runtime",
            }
        )
        has_query = self.native_queries_supported
        footprint_supported = self.native_footprint_supported(footprint)
        backend_name = str(getattr(self.backend, "name", self.backend)).lower().replace("-", "_")
        reference_requested = backend_name in {"reference", "python", "cpu_reference"}
        requested_method = self._candidate_plan.requested_method
        query_candidate = bool(
            has_query
            and footprint_supported
            and not reference_requested
            and requested_method not in {"irfft", "fb3", "dense", "fft", "dense_irfft", "packed", "v02"}
        )
        fallback_reason = "native_query_footprint_unavailable" if not footprint_supported else None
        if backend_name == "cuda":
            # Only the retained-query display has a CUDA terminal. A grid
            # heuristic must not select a CPU display or unavailable cuFFT.
            if selected_sampling == "grid" or not query_candidate:
                raise ValueError("CUDA ocean images require the retained SPF32 query display")
            selected_sampling = "queries"
            costs = MappingProxyType({
                **dict(costs),
                "model": "available_cuda_display_consumer",
                "selection_reason": "retained query display is the implemented CUDA image terminal",
            })
        elif selected_sampling == "auto":
            selected_sampling = "queries" if query_candidate and query_cost < grid_cost else "grid"
        elif selected_sampling == "queries" and not query_candidate:
            raise ValueError("this ocean plan has no native compact point consumer")
        request = OceanOutputRequest(
            sampling="points" if selected_sampling == "queries" else "grid",
            outputs=("height", "grad_y", "grad_x"),
            footprint=footprint,
            backend=backend_name,
            output_dtype=str(_dtype(output_dtype, np.float32)),
            point_count=query_samples if selected_sampling == "queries" else grid_samples,
        )
        bound = self.bind_execution(request, native_available=True)
        return selected_sampling, bound, costs, fallback_reason

    @property
    def phase_state(self) -> tuple[Any, ...]:
        if self._phase_fastpath:
            assert self._phase_values_u64 is not None and self._phase_increments_u64 is not None
            from spf.renderer.ocean import WidePhaseAccumulator

            return tuple(
                WidePhaseAccumulator(int(phase), int(increment), self.bits)
                for phase, increment in zip(self._phase_values_u64, self._phase_increments_u64, strict=True)
            )
        return tuple(m.phase_accumulator.copy() for m in self.modes)

    @property
    def wide_phases(self) -> np.ndarray:
        """Current owned wide phases without exposing compatibility objects."""

        if self._phase_fastpath:
            assert self._phase_values_u64 is not None
            return self._phase_values_u64.copy()
        return np.asarray([m.phase_accumulator.copy() for m in self.modes], dtype=object)

    def _phase_codes(self) -> np.ndarray:
        """Return current emitted phase codes for all owned modes."""

        if self._phase_fastpath:
            assert self._phase_values_u64 is not None
            return _emit_phase_codes(self._phase_values_u64, self.bits)
        return np.asarray([mode.phase_code for mode in self.modes], dtype=np.uint16)

    def _current_mode_coefficients(self) -> np.ndarray:
        """Compose mode coefficients from the authoritative phase array."""

        if not self.modes:
            return np.empty(0, dtype=np.complex128)
        phase_codes = self._phase_codes()
        angles = phase_codes.astype(np.float64) * (2.0 * math.pi / 65536.0)
        result = np.asarray(self.base_coefficients * (np.cos(angles) + 1j * np.sin(angles)), dtype=np.complex128)
        if self.carrier.source_kind == "block_view" and _view_has_fractional_offsets(self.carrier):
            result *= _view_numeric_factor(self.carrier)
        return result

    def _refresh_native_phase(self, native: Any) -> None:
        """Refresh only phase dependent native state when the ABI provides it."""

        refresh_phase = getattr(native, "refresh_phase", None)
        if callable(refresh_phase):
            refresh_phase()
        else:
            # Compatibility with the first native sampler ABI.  New kernels
            # should expose refresh_phase() so static geometry, derivative
            # factors, and the SPF pivot remain cached between frames.
            native.refresh()

    @property
    def packed_coefficients(self) -> np.ndarray:
        return self.carrier.packed_coefficients

    def _native_representation(self) -> str:
        """Return the native source lane without decoding a dense carrier."""

        if self.carrier.source_kind == "block_view":
            return "spf32"
        values = tuple(mode.coefficient for mode in self.modes) if self.modes else self.carrier.coefficient_values
        if self.carrier.source_kind == "half_spectrum":
            return "spf32"
        if values and all(isinstance(value, (int, np.integer, v02.SPF32Value)) for value in values):
            return "spf32"
        return "fp32"

    def _lowered_source_transfer(self) -> FilteredSourceLowering | None:
        """Lower a retained filtered support once for packed native consumers.

        The lowering is support-sized and cached with the immutable source
        view.  Temporal wide phases remain native sampler state; only the
        static source transfer is emitted here.
        """

        if not _source_transfer_operations(self.carrier):
            return None
        view = self.carrier.source_view
        if view is None or not bool(getattr(view, "frequency_domain", False)):
            raise ValueError("native filtered source lowering requires a frequency-domain source view")
        key = (
            id(view),
            tuple(self.shape),
            tuple(self.spacing),
            bool(self.normalize),
            tuple(self.carrier.index_values),
            bool(self.carrier.half_view),
        )
        cached = self._native_transfer_lowering
        if cached is not None and self._native_transfer_lowering_key == key:
            return cached
        indices, _carrier_words, _flags = self.carrier.materialize_sparse_packed()
        # Re-read the canonical representative addresses through the view's
        # raw source accessor.  ``materialize_sparse_packed`` may apply an
        # integral lazy offset for ordinary packed consumers; FilteredBlockView
        # must see its own offsets in ``planes`` exactly once.
        source_addresses = np.asarray(
            np.ravel_multi_index(
                tuple(np.asarray(indices[:, axis], dtype=np.int64) for axis in range(indices.shape[1])),
                tuple(int(value) for value in view.shape),
            ),
            dtype=np.int64,
        ).reshape(-1)
        source_words = np.asarray(view.sample_words(source_addresses, phase=True), dtype=np.uint32).reshape(-1)
        lowering = lower_filtered_source(
            view,
            indices,
            shape=self.shape,
            spacing=self.spacing,
            normalize=self.normalize,
            # The carrier has already applied any legacy narrow phase state
            # and integral view offset at its packed boundary.  The retained
            # FilteredBlockView then contributes its transfer exactly once.
            source_words=source_words,
        )
        self._native_transfer_lowering = lowering
        self._native_transfer_lowering_key = key
        return lowering

    def _native_gaussian_footprint_lowering(self, footprint: Any) -> FilteredSourceLowering | None:
        """Lower a full covariance for the packed SPF32 native consumer.

        The C ABI has only separable ``(sigma_y, sigma_x)`` arguments.  A
        covariance with a cross term is therefore composed into the addressed
        SPF32 words once, before the native call, and the ABI receives zero
        sigmas.  This path is deliberately unavailable to FP32 consumers so
        that their native/reference contract remains unchanged.
        """

        if footprint is None:
            return None
        covariance = getattr(footprint, "covariance", None)
        if covariance is None and isinstance(footprint, Mapping):
            covariance = footprint.get("covariance")
        if covariance is None:
            return None
        matrix = _validated_covariance(covariance, len(self.shape))
        if len(self.shape) != 2 or (float(matrix[0, 1]) == 0.0 and float(matrix[1, 0]) == 0.0):
            return None
        if self._native_representation() != "spf32":
            return None
        source_lowering = self._lowered_source_transfer()
        key = (
            _footprint_cache_key(footprint),
            tuple(self.shape),
            tuple(self.spacing),
            bool(self.normalize),
            None if source_lowering is None else id(source_lowering),
            tuple(self.carrier.index_values),
            bool(self.carrier.half_view),
        )
        cached = self._native_footprint_lowering
        if cached is not None and self._native_footprint_lowering_key == key:
            return cached

        if source_lowering is not None:
            lowering = lower_spf32_gaussian_footprint(
                source_lowering.indices,
                source_lowering.words,
                shape=self.shape,
                spacing=self.spacing,
                normalize=self.normalize,
                covariance=matrix,
                source_lowering=source_lowering,
            )
        else:
            # Keep sparse-mode narrow phase state in the native phase buffer,
            # exactly as the ordinary SPF32 path does.  The Gaussian is real,
            # so it modifies only the log plane and does not consume that
            # temporal/narrow phase twice.
            if not self.modes and self.carrier.source_kind == "sparse_modes" and self.carrier.phase_values:
                from spf.renderer.ocean_carrier import _word_from_value

                indices = np.asarray(self.carrier.index_values, dtype=np.int64)
                words = np.asarray(
                    [_word_from_value(value) for value in self.carrier.coefficient_values],
                    dtype=np.uint32,
                )
            else:
                indices, words, _ = self.carrier.materialize_sparse_packed()
            lowering = lower_spf32_gaussian_footprint(
                indices,
                words,
                shape=self.shape,
                spacing=self.spacing,
                normalize=self.normalize,
                covariance=matrix,
            )
        self._native_footprint_lowering = lowering
        self._native_footprint_lowering_key = key
        return lowering

    @property
    def native_queries_supported(self) -> bool:
        """Whether this selected route has a compact native query carrier."""

        # Analytic atoms/jets have a separate C callback ABI.  Do not make
        # them pretend to be sparse SPF carriers merely to pass this probe.
        # The canonical consumer owns chart/support/residual validation.
        if self.carrier.source_kind == "analytic":
            if str(getattr(self.backend, "name", self.backend)).lower() == "cuda":
                return False
            analytic = self._canonical_analytic_prepared(output_dtype=np.float64)
            return bool(getattr(analytic, "native_queries_supported", False))

        # A retained frequency transfer is lowered to addressed SPF32 words
        # at the native boundary.  Probe that support-sized lowering here so
        # malformed/out-of-range planes remain reference eligible without
        # claiming that native would silently drop the transfer.
        lowered_transfer = None
        if _source_transfer_operations(self.carrier):
            try:
                lowered_transfer = self._lowered_source_transfer()
            except (TypeError, ValueError, OverflowError):
                return False

        support_count = int(self.carrier.active_count)
        if lowered_transfer is not None:
            support_count = int(lowered_transfer.active_count)
        # Native query cost is proportional to the addressed mode walk, not
        # merely to the field shape.  A dense half spectrum must therefore
        # stay on its dense/analytic consumer even when its array happens to
        # be two-dimensional; only a bounded compact support is eligible for
        # the direct query ABI.
        compact_support = support_count > 0 and support_count <= min(4096, max(1, self.carrier.potential_elements // 8))
        backend_name = str(getattr(self.backend, "name", self.backend)).lower().replace("-", "_")
        if backend_name == "cuda" and self._native_representation() != "spf32":
            return False
        return bool(
            len(self.shape) == 2
            and self.carrier.source_kind != "analytic"
            and (self.carrier.source_kind != "block_view" or int(self.carrier.metadata.get("scalar_bits", 32)) == 32)
            and (
                self.carrier.metadata.get("closed_form") != "constant_half"
                or bool(_source_transfer_operations(self.carrier))
            )
            and (
                self.plan.diagnostics.get("structure_route") != "diagonal_collapsed"
                or bool(_source_transfer_operations(self.carrier))
            )
            and (not _view_has_fractional_offsets(self.carrier) or lowered_transfer is not None)
            and compact_support
            and (self.modes or self.carrier.source_kind in {"block_view", "sparse_modes", "half_spectrum"})
        )

    def native_sampling_capability(self) -> Mapping[str, Any]:
        """Describe native query eligibility without touching coefficient storage."""

        if self.carrier.source_kind == "analytic":
            analytic = self._canonical_analytic_prepared(output_dtype=np.float64)
            capability = getattr(analytic, "native_sampling_capability", None)
            if callable(capability):
                return MappingProxyType(
                    {
                        **dict(capability()),
                        "terminal_route": self.plan.route,
                        "coordinate_contract": "physical_yx",
                    }
                )

        return MappingProxyType(
            {
                "supported": self.native_queries_supported,
                "representation": self._native_representation() if self.native_queries_supported else None,
                "route": self.plan.route,
                "reason": (
                    "closed constant half-plane has a direct Dirichlet consumer"
                    if self.carrier.metadata.get("closed_form") == "constant_half"
                    and not _source_transfer_operations(self.carrier)
                    else "retained source frequency transfer lowered to addressed SPF32"
                    if _source_transfer_operations(self.carrier) and self.native_queries_supported
                    else "normalized source frequency transfer cannot be lowered for native SPF32"
                    if _source_transfer_operations(self.carrier)
                    else "addressed support exceeds the bounded native query mode budget"
                    if int(self.carrier.active_count) > min(4096, max(1, self.carrier.potential_elements // 8))
                    else "BlockView lazy offset requires a numeric wide composition"
                    if _view_has_fractional_offsets(self.carrier)
                    else "native sampler requires a 2-D sparse or packed route"
                    if not self.native_queries_supported
                    else "compact addressed carrier available"
                ),
            }
        )

    def native_footprint_supported(self, footprint: Any = None) -> bool:
        """Return whether the fused native sampler can consume ``footprint``.

        Diagonal Gaussian covariance uses the native ABI sigma pair.  A full
        covariance with a cross term is lowered into addressed SPF32 words so
        the native ABI can receive zero sigmas; the FP32 lane remains limited
        to its existing diagonal contract.  Invalid covariance is rejected at
        the shared validation boundary instead of being silently clamped.
        """

        if self.carrier.source_kind == "analytic":
            analytic = self._canonical_analytic_prepared(output_dtype=np.float64)
            if not bool(getattr(analytic, "native_queries_supported", False)):
                return False
            if footprint is None:
                return True
            covariance = getattr(footprint, "covariance", None)
            transfer = getattr(footprint, "frequency_transfer", None)
            if isinstance(footprint, Mapping):
                covariance = footprint.get("covariance", covariance)
                transfer = footprint.get("frequency_transfer", transfer)
            if covariance is None:
                if callable(transfer):
                    return False
                raise TypeError("native ocean footprint must provide covariance or frequency_transfer(frequency)")
            # Preparation validates the covariance and atom convolution.  A
            # separate canonical object is cached for this footprint, while
            # the source without a residual remains eligible for C queries.
            candidate = self._canonical_analytic_prepared(footprint=footprint, output_dtype=np.float64)
            return bool(getattr(candidate, "native_queries_supported", False))

        if footprint is None:
            return (
                bool(self.native_queries_supported)
                if _source_transfer_operations(self.carrier)
                else True
            )
        covariance = getattr(footprint, "covariance", None)
        transfer = getattr(footprint, "frequency_transfer", None)
        if isinstance(footprint, Mapping):
            covariance = footprint.get("covariance", covariance)
            transfer = footprint.get("frequency_transfer", transfer)
        if covariance is not None:
            matrix = _validated_covariance(covariance, 2)
            if not self.native_queries_supported:
                return False
            diagonal = bool(float(matrix[0, 1]) == 0.0 and float(matrix[1, 0]) == 0.0)
            if diagonal:
                return True
            # Full covariance composition is a packed SPF32 feature.  The
            # helper also validates source planes and keeps a value-keyed
            # support-sized cache, so this probe describes the actual native
            # carrier that the sampler would consume.
            if self._native_representation() != "spf32":
                return False
            try:
                return self._native_gaussian_footprint_lowering(footprint) is not None
            except (TypeError, ValueError, OverflowError):
                return False
        if callable(transfer):
            return False
        raise TypeError("native ocean footprint must provide covariance or frequency_transfer(frequency)")

    def native_sampler(self, *, footprint: Any = None) -> Any:
        """Return the optional fused native sampler while retaining its ABI context.

        The returned object owns the packed/index/phase buffers and can be
        handed directly to the native display callback.  Construction is
        explicit: a missing library raises instead of relabeling the Python
        consumer as SPF native.
        """

        if self.carrier.source_kind == "analytic":
            if str(getattr(self.backend, "name", self.backend)).lower() == "cuda":
                raise ValueError("CUDA ocean has no analytic atom/jet consumer; packed SPF32 support is required")
            if not self.native_footprint_supported(footprint):
                raise ValueError("native analytic sampler is unavailable for the selected source/footprint")
            analytic = self._canonical_analytic_prepared(footprint=footprint, output_dtype=np.float64)
            analytic_key: tuple[Any, ...] = ("analytic", id(analytic), _footprint_cache_key(footprint))
            cached = self._native_sampler
            if cached is not None and self._native_sampler_key == analytic_key:
                return cached
            sampler = analytic.native_sampler(footprint=footprint)
            self._native_sampler = sampler
            self._native_sampler_key = analytic_key
            return sampler

        if not self.native_queries_supported:
            raise ValueError("native ocean sampler is unavailable for the selected route/carrier")
        if len(self.shape) != 2:
            raise ValueError("native ocean sampler currently requires a two-dimensional field")
        if self.bits == 0 or self.bits > 64:
            raise ValueError("native ocean sampler supports wide phase state up to 64 bits")
        backend_name = str(getattr(self.backend, "name", self.backend)).lower().replace("-", "_")
        representation = self._native_representation()
        if backend_name == "cuda" and representation != "spf32":
            raise ValueError("CUDA ocean sampler accepts only packed SPF32 source words")

        if not self.native_footprint_supported(footprint):
            raise ValueError("native ocean sampler cannot consume the selected footprint")
        covariance = getattr(footprint, "covariance", None)
        if covariance is None and isinstance(footprint, Mapping):
            covariance = footprint.get("covariance")
        source_lowering = self._lowered_source_transfer()
        footprint_lowering = self._native_gaussian_footprint_lowering(footprint)
        if footprint_lowering is not None:
            # The full covariance has already been composed into the emitted
            # words.  Passing diagonal projections here would apply a second
            # Gaussian and silently change the requested transfer.
            footprint_pair = (0.0, 0.0)
        elif covariance is None:
            footprint_pair = (0.0, 0.0)
        else:
            matrix = _validated_covariance(covariance, 2)
            # Public covariance is in (x, y) order; the native context is
            # explicitly (sigma_y, sigma_x).
            footprint_pair = (
                math.sqrt(max(0.0, float(matrix[1, 1]))),
                math.sqrt(max(0.0, float(matrix[0, 0]))),
            )
        lowering = footprint_lowering or source_lowering
        key = (
            backend_name,
            representation,
            tuple(self.spacing),
            footprint_pair,
            self.normalize,
            None if source_lowering is None else id(source_lowering),
            None if footprint_lowering is None else id(footprint_lowering),
            _footprint_cache_key(footprint),
        )
        cached = self._native_sampler
        if cached is not None and self._native_sampler_key == key:
            # ``phases`` remains a writable compatibility view for existing
            # callers (including frame benchmarks that reset it directly).
            # The native ABI owns a copy, so reconcile that view at the
            # sampler boundary before returning the cached context.  This is
            # one O(K) comparison per frame, outside the per-pixel callback,
            # and avoids rebuilding static packed/index/derivative state.
            if self.modes:
                if self._phase_fastpath:
                    assert self._phase_values_u64 is not None
                    desired_phases = self._phase_values_u64
                else:
                    desired_phases = np.asarray(
                        [int(mode.phase_accumulator.phase) for mode in self.modes], dtype=np.uint64
                    )
                if not np.array_equal(cached.phases, desired_phases):
                    cached.phases[...] = desired_phases
                    self._refresh_native_phase(cached)
            return cached
        if representation == "spf32":
            if lowering is not None:
                # The retained view has already composed source and transfer
                # planes into canonical SPF32 words.  Keep the support
                # addressed and let the native sampler own only temporal
                # phase state.
                indices = np.asarray(lowering.indices, dtype=np.int64)
                words = np.asarray(lowering.words, dtype=np.uint32)
            elif not self.modes and self.carrier.source_kind == "sparse_modes" and self.carrier.phase_values:
                # ``materialize_sparse_packed`` applies legacy narrow phase
                # values to its words.  Keep the base word untouched here so
                # the native context applies that phase exactly once.
                from spf.renderer.ocean_carrier import _word_from_value

                indices = np.asarray(self.carrier.index_values, dtype=np.int64)
                words = np.asarray(
                    [_word_from_value(value) for value in self.carrier.coefficient_values],
                    dtype=np.uint32,
                )
            else:
                indices, words, _ = self.carrier.materialize_sparse_packed()
        else:
            indices = np.asarray(
                self._mode_indices if self.modes else self.carrier.index_values,
                dtype=np.int64,
            )
            words = np.empty(0, dtype=np.uint32)
        native_phase_bits = max(1, self.bits)
        if self.modes:
            if self._phase_fastpath:
                assert self._phase_values_u64 is not None and self._phase_increments_u64 is not None
                # Keep the sampler and the prepared terminal on one phase
                # buffer.  A reset/advance therefore updates native state
                # without rebuilding a mode object or copying an object array.
                phase_values = self._phase_values_u64
                increments = self._phase_increments_u64
            else:
                phase_values = np.asarray([int(mode.phase_accumulator.phase) for mode in self.modes], dtype=np.uint64)
                increments = np.asarray([int(mode.phase_accumulator.increment) for mode in self.modes], dtype=np.uint64)
        elif self.carrier.phase_values and source_lowering is None:
            native_phase_bits = 16
            phase_values = np.asarray(
                [_phase_value_to_code(value) for value in self.carrier.phase_values], dtype=np.uint64
            )
            increments = np.zeros(len(indices), dtype=np.uint64)
        else:
            phase_values = np.zeros(len(indices), dtype=np.uint64)
            increments = np.zeros(len(indices), dtype=np.uint64)
        if len(phase_values) != len(indices):
            raise ValueError("native sampler mode state does not match packed carrier support")
        spacing = self.spacing
        if backend_name == "cuda":
            from spf.renderer._ocean_cuda import CUDAOceanSampler

            sampler = CUDAOceanSampler.from_spf32(
                words,
                indices,
                shape=self.shape,
                phases=phase_values,
                increments=increments,
                phase_bits=native_phase_bits,
                spacing=spacing,
                normalize=self.normalize,
                footprint=footprint_pair,
            )
            sampler._timeline_owner = "prepared_ocean_host_uint64_then_support_sized_device_refresh"
        elif representation == "spf32":
            from spf.renderer._ocean_kernel import NativeOceanSampler

            sampler = NativeOceanSampler.from_spf32(
                words,
                indices,
                shape=self.shape,
                phases=phase_values,
                increments=increments,
                phase_bits=native_phase_bits,
                spacing=spacing,
                normalize=self.normalize,
                footprint=footprint_pair,
            )
        else:
            from spf.renderer import ocean as o
            from spf.renderer._ocean_kernel import NativeOceanSampler

            # Native FP32 consumes unrotated source coefficients plus the
            # retained wide phase array.  Rotating here would apply temporal
            # phase twice (the reference terminal does compose it here).
            values = np.asarray(
                self._mode_base
                if self.modes
                else [o._coefficient_complex(value) for value in self.carrier.coefficient_values],
                dtype=np.complex128,
            )
            sampler = NativeOceanSampler.from_fp32(
                values,
                indices,
                shape=self.shape,
                phases=phase_values,
                increments=increments,
                phase_bits=native_phase_bits,
                spacing=spacing,
                normalize=self.normalize,
                footprint=footprint_pair,
            )
        self._native_sampler = sampler
        self._native_sampler_key = key
        if lowering is not None:
            # NativeOceanSampler folds this support-sized source/footprint
            # certificate into error_ledger(), including display-side jet
            # wrappers.  For a full covariance this is the combined lowering;
            # the native sigma pair is zero and contributes no duplicate gain.
            sampler._source_transfer_ledger = lowering
        return sampler

    def derivative_operators(self, *, spacing: Any = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Expose the exact packed/log derivative operators for the carrier."""

        return self.carrier.derivative_operators(self.spacing if spacing is None else spacing)

    def advance(self, steps: int = 1) -> None:
        if isinstance(steps, (bool, np.bool_)) or int(steps) != steps:
            raise TypeError("wide phase steps must be an integer")
        steps = int(steps)
        if not self.modes or steps == 0:
            return
        if self._phase_fastpath:
            assert self._phase_values_u64 is not None and self._phase_increments_u64 is not None
            phases = self._phase_values_u64
            increments = self._phase_increments_u64
            bits = int(self.bits)
            mask = (1 << bits) - 1
            if steps in {1, -1}:
                before = phases.copy()
                if steps == 1:
                    np.add(phases, increments, out=phases)
                    wrapped = phases < before if bits == 64 else before > np.uint64(mask) - increments
                else:
                    np.subtract(phases, increments, out=phases)
                    wrapped = phases > before if bits == 64 else before < increments
                if bits < 64:
                    phases &= np.uint64(mask)
                self.wraps += np.asarray(wrapped, dtype=np.int64)
            else:
                before = phases.copy()
                delta = increments * np.uint64(steps & mask)
                np.add(phases, delta, out=phases)
                if bits < 64:
                    phases &= np.uint64(mask)
                modulus = 1 << bits
                # Larger jumps are deliberately an exact slow path.  The
                # storage and common one-step update remain uint64-native.
                for position, (old, increment) in enumerate(zip(before, increments, strict=True)):
                    total = int(old) + int(increment) * steps
                    self.wraps[position] += abs(total // modulus - int(old) // modulus)
            self._pending_phase_updates += abs(steps) * len(self.modes)
            native = getattr(self, "_native_sampler", None)
            if native is not None:
                if getattr(native, "phases", None) is not phases:
                    native.phases[...] = phases
                # The native ABI refreshes retained phase state in place;
                # geometry, derivative factors, and SPF pivot data stay in
                # the sampler's persistent context.
                self._refresh_native_phase(native)
            return
        for position, mode in enumerate(self.modes):
            before = mode.phase_accumulator.wraps
            mode.advance(steps)
            self.wraps[position] += mode.phase_accumulator.wraps - before
        self._pending_phase_updates += abs(steps) * len(self.modes)
        native = getattr(self, "_native_sampler", None)
        if native is not None:
            native.advance(steps)
        self.phases = np.asarray([m.phase_accumulator.phase for m in self.modes], dtype=object)

    def set_phase_state(self, phases: Any = 0) -> None:
        """Set the owned master phases and clear frame/wrap accounting.

        The prepared object owns a private timeline, so mutating a caller's
        original :class:`OceanMode` cannot reset it.  This explicit boundary
        is also used by benchmark harnesses between timed trials.
        """

        if not self.modes:
            self._pending_phase_updates = 0
            return
        if np.isscalar(phases):
            values = [phases] * len(self.modes)
        else:
            values = list(phases)
            if len(values) != len(self.modes):
                raise ValueError("ocean phase state count does not match owned modes")
        if self._phase_fastpath:
            assert self._phase_values_u64 is not None
            mask = (1 << self.bits) - 1
            for position, (mode, value) in enumerate(zip(self.modes, values, strict=True)):
                raw_value: Any = getattr(value, "phase", getattr(value, "T", value))
                if isinstance(raw_value, (bool, np.bool_)) or int(raw_value) != raw_value:
                    raise TypeError("ocean phases must be integral wide phase ticks")
                phase = int(raw_value) & mask
                self._phase_values_u64[position] = np.uint64(phase)
                mode.phase_accumulator.phase = phase
                mode.phase_accumulator._wraps = 0
            self.wraps[:] = 0
            native = self._native_sampler
            if native is not None:
                if getattr(native, "phases", None) is not self._phase_values_u64:
                    native.phases[...] = self._phase_values_u64
                self._refresh_native_phase(native)
            self._pending_phase_updates = 0
            return
        for mode, value in zip(self.modes, values, strict=True):
            raw_value = getattr(value, "phase", getattr(value, "T", value))
            if isinstance(raw_value, (bool, np.bool_)) or int(raw_value) != raw_value:
                raise TypeError("ocean phases must be integral wide phase ticks")
            mode.phase_accumulator.phase = int(raw_value) & mode.phase_accumulator.mask
            mode.phase_accumulator._wraps = 0
        self.wraps[:] = 0
        self.phases = np.asarray([mode.phase_accumulator.phase for mode in self.modes], dtype=object)
        native = self._native_sampler
        if native is not None:
            native.phases[:] = np.asarray([mode.phase_accumulator.phase for mode in self.modes], dtype=np.uint64)
            self._refresh_native_phase(native)
        self._pending_phase_updates = 0

    def reset(self) -> None:
        """Reset the owned timeline to phase zero for a fresh frame trial."""

        self.set_phase_state(0)

    def sync_phases(self, modes: Any) -> None:
        for position, mode in enumerate(modes):
            if position >= len(self.modes):
                break
            if self._phase_fastpath:
                assert self._phase_values_u64 is not None
                phase = int(self._phase_values_u64[position])
                self.modes[position].phase_accumulator.phase = phase
                mode.phase_accumulator.phase = phase
                self.modes[position].phase_accumulator._wraps += int(self.wraps[position])
                mode.phase_accumulator._wraps += int(self.wraps[position])
                continue
            source = self.modes[position].phase_accumulator
            target = mode.phase_accumulator
            target.phase = source.phase
            target._wraps += int(self.wraps[position])
        self.wraps[:] = 0

    def _source_values(self, indices: np.ndarray, words: np.ndarray, counters: _Counters) -> np.ndarray:
        """Project source words through a normalized frequency transfer.

        ``OceanCarrier.materialize_sparse_packed`` intentionally knows only
        about addressed storage.  A ``FilteredBlockView`` therefore returns
        the source words and leaves its charted transfer for the selected
        numeric consumer.  Read the raw words again at this boundary so the
        view applies source headers, transfer, and masks exactly once.
        """

        transfer_operations = _source_transfer_operations(self.carrier)
        if not transfer_operations:
            return np.asarray(decode_words(words), dtype=np.complex128)
        view = self.carrier.source_view
        if view is None or not bool(getattr(view, "frequency_domain", False)):
            raise ValueError(
                "normalized spatial block filtering has no packed ocean consumer; "
                "select an explicit spatial materialization"
            )
        linear = np.asarray(
            np.ravel_multi_index(
                tuple(np.asarray(indices[:, axis], dtype=np.int64) for axis in range(indices.shape[1])),
                tuple(int(value) for value in view.shape),
            ),
            dtype=np.int64,
        ).reshape(-1)
        raw_words = np.asarray(view.sample_words(linear, phase=True), dtype=np.uint32).reshape(-1)
        values = np.asarray(view.project_words(raw_words, linear), dtype=np.complex128).reshape(-1)
        counters.record(
            "SourceTransfer",
            operations=transfer_operations,
            domain="frequency",
            addresses=int(linear.size),
            coordinate_chart=type(getattr(view, "chart", None)).__name__,
        )
        return values

    def _dynamic_sparse(self, counters: _Counters) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return (indices, coefficients, weights, words) for one frame."""

        from spf.renderer import ocean as o

        if self.modes:
            indices = np.asarray(self._mode_indices, dtype=np.int64)
            _, words, flags = self.carrier.materialize_sparse_packed()
            if _source_transfer_operations(self.carrier):
                # The normalized source transfer is static; compose the
                # owned wide phase after the transfer at the same numeric
                # boundary as the ordinary mode path.
                values = self._source_values(indices, words, counters)
                phase_codes = self._phase_codes()
                angles = phase_codes.astype(np.float64) * (2.0 * math.pi / 65536.0)
                values *= np.cos(angles) + 1j * np.sin(angles)
            else:
                values = self._current_mode_coefficients()
            counters.packed_words_read += int(words.size)
            counters.bytes_read += int(words.nbytes)
            counters.phase_emissions += len(self.modes)
            counters.source_reads += len(indices)
            counters.record("PhaseCompose", modes=len(self.modes), master_bits=self.bits, emitted_bits=16)
            return indices, values, np.where(flags, 1.0, 2.0), words
        if self.carrier.source_kind == "sparse_modes":
            indices = np.asarray(self.carrier.index_values, dtype=np.int64)
            values = np.asarray(
                [
                    o._coefficient_complex(value, self.carrier.phase_values[i] if self.carrier.phase_values else None)
                    for i, value in enumerate(self.carrier.coefficient_values)
                ],
                dtype=np.complex128,
            )
            _, words, flags = self.carrier.materialize_sparse_packed()
            counters.packed_words_read += int(words.size)
            counters.bytes_read += int(words.nbytes)
            counters.source_reads += len(indices)
            counters.record("AddressSupport", source="sparse_modes", count=len(indices))
            return indices, values, np.where(flags, 1.0, 2.0), words
        if self.carrier.source_kind == "half_spectrum" and self.half is not None:
            terms = o._terms_from_half(self.half, self.shape)
            indices = np.asarray([term[0] for term in terms], dtype=np.int64)
            values = np.asarray([term[1] for term in terms], dtype=np.complex128)
            flags = np.asarray([term[2] for term in terms], dtype=bool)
            counters.record("AddressSupport", source="half_spectrum", count=len(indices))
            counters.source_reads += len(indices)
            return indices, values, np.where(flags, 1.0, 2.0), np.empty(0, dtype=np.uint32)
        indices, words, flags = self.carrier.materialize_sparse_packed()
        values = self._source_values(indices, words, counters)
        if _view_has_fractional_offsets(self.carrier) and not _source_transfer_operations(self.carrier):
            # BlockView.project_words preserves fractional exposure/phase
            # offsets in numeric space.  Keep the source packed through the
            # addressed read, then apply that exact factor only at this
            # reference consumer boundary; native packed samplers are gated
            # off for this representation.
            values *= _view_numeric_factor(self.carrier)
            counters.record(
                "ViewOffsetCompose",
                domain="numeric_reference",
                log_offset=float(getattr(self.carrier.source_view, "log_offset", 0.0)),
                phase_offset=float(getattr(self.carrier.source_view, "phase_offset", 0.0)),
            )
        counters.packed_words_read += int(words.size)
        counters.bytes_read += int(words.nbytes)
        counters.source_reads += len(indices)
        counters.fallback_conversions += 1
        counters.record("PackedDecodeReference", count=int(words.size), destination="numeric_terminal")
        return indices, values, np.where(flags, 1.0, 2.0), words

    def _diagonal_terms(self, counters: _Counters) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Use the canonical addressed reader for either full or half support.

        Centralization retains masks, lazy offsets, wide timelines and the
        carrier's Hermitian validation without a second coordinate convention.
        """
        if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
            return None
        indices, coefficients, weights, _ = self._dynamic_sparse(counters)
        if indices.size and not bool(np.all(indices[:, 0] == indices[:, 1])):
            return None
        from spf.renderer import ocean as o

        for index, coefficient in zip(indices, coefficients, strict=True):
            if o._is_self_conjugate(tuple(int(value) for value in index), self.shape) and abs(coefficient.imag) > 1e-12:
                raise o.HermitianValidationError("diagonal self-conjugate coefficient must remain real")
        return indices, coefficients, weights

    def _diagonal_joint(
        self,
        *,
        points: np.ndarray | None,
        need_height: bool,
        gradient_axes: tuple[int, ...],
        output_dtype: np.dtype,
        counters: _Counters,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None] | None:
        """Collapse a square diagonal spectrum to a periodic one-dimensional ring.

        The grid evaluates K*N characters once, then copies cyclic slices into
        the requested N*N outputs. No per-mode two-dimensional basis is formed.
        Physical off-grid queries use the canonical real Nyquist extension.
        """

        terms = self._diagonal_terms(counters)
        if terms is None:
            return None
        indices, coefficients, weights = terms
        size = self.shape[0]
        count = points.shape[0] if points is not None else int(math.prod(self.shape))
        basis_count = count if points is not None else size
        height_basis = np.zeros(basis_count, dtype=np.float64) if need_height else None
        gradient_basis = {axis: np.zeros(basis_count, dtype=np.float64) for axis in gradient_axes}
        normalization = float(math.prod(self.shape)) if self.normalize else 1.0
        if points is not None:
            qy = points[:, 0] / self.spacing[0]
            qx = points[:, 1] / self.spacing[1]
            coordinate_sum = qy + qx
        else:
            coordinate_sum = np.arange(size, dtype=np.float64)
        for index, coefficient, weight in zip(indices, coefficients, weights, strict=True):
            frequency = _signed_frequency(int(index[0]), size)
            scale = weight / normalization
            nyquist = size % 2 == 0 and abs(frequency) == size // 2
            if nyquist and points is not None:
                # The self-conjugate corner has a product of real cosine
                # factors, not cos(pi*(qx+qy)); they agree only on the lattice.
                cy, cx = np.cos(np.pi * qy), np.cos(np.pi * qx)
                if height_basis is not None:
                    height_basis += scale * coefficient.real * cy * cx
                for axis in gradient_axes:
                    derivative = (
                        -np.pi * np.sin(np.pi * qy) * cx / self.spacing[0]
                        if axis == 0
                        else -np.pi * cy * np.sin(np.pi * qx) / self.spacing[1]
                    )
                    gradient_basis[axis] += scale * coefficient.real * derivative
            else:
                angle = (2.0 * math.pi * frequency / size) * coordinate_sum
                cosine, sine = np.cos(angle), np.sin(angle)
                if height_basis is not None:
                    height_basis += scale * (coefficient.real * cosine - coefficient.imag * sine)
                if not nyquist:
                    imaginary_part = coefficient.real * sine + coefficient.imag * cosine
                    for axis in gradient_axes:
                        multiplier = 2.0 * math.pi * frequency / (size * self.spacing[axis])
                        gradient_basis[axis] -= scale * multiplier * imaginary_part
            counters.term_evaluations += int(basis_count)

        def emit(basis: np.ndarray | None) -> np.ndarray | None:
            if basis is None:
                return None
            if points is not None:
                return np.asarray(basis, dtype=output_dtype)
            output = np.empty(self.shape, dtype=output_dtype)
            for row in range(size):
                output[row, : size - row] = basis[row:]
                output[row, size - row :] = basis[:row]
            return output

        height = emit(height_basis)
        gradients = {axis: emit(basis) for axis, basis in gradient_basis.items()}
        counters.output_fields = int(need_height) + len(gradient_axes)
        counters.materialized_output_elements += int(count * counters.output_fields)
        # Maximum live numeric scratch estimate, not measured DRAM traffic.
        counters.materialized_intermediate_elements += int(basis_count * (8 + counters.output_fields))
        additions = int(max(0, len(indices) - 1) * basis_count * counters.output_fields)
        counters.real_adds += additions
        counters.reduction_operations += additions
        counters.record(
            "DiagonalCollapsedSynthesis",
            terms=len(indices),
            points=int(count),
            basis_points=int(basis_count),
            term_evaluations=int(len(indices) * basis_count),
            collapsed_coordinate="y/spacing_y + x/spacing_x",
            grid_emission="cyclic_row_slices" if points is None else "physical_point_queries",
            scratch_scope="linear_basis_estimate",
            shared_phase_basis=True,
        )
        return height, gradients.get(1), gradients.get(0)

    def _grid_coordinates(self) -> tuple[Any, ...]:
        if len(self.shape) == 1:
            return (np.arange(self.shape[0], dtype=np.float64),)
        if len(self.shape) == 2:
            return (
                np.arange(self.shape[0], dtype=np.float64)[:, None],
                np.arange(self.shape[1], dtype=np.float64)[None, :],
            )
        return tuple(np.indices(self.shape, dtype=np.float64)[axis] for axis in range(len(self.shape)))

    def _sparse_joint(
        self,
        *,
        points: np.ndarray | None,
        need_height: bool,
        gradient_axes: tuple[int, ...],
        footprint: Any,
        output_dtype: np.dtype,
        counters: _Counters,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        if (
            self.carrier.metadata.get("support_pattern") == "diagonal"
            and not _source_transfer_operations(self.carrier)
            and footprint is None
        ):
            diagonal = self._diagonal_joint(
                points=points,
                need_height=need_height,
                gradient_axes=gradient_axes,
                output_dtype=output_dtype,
                counters=counters,
            )
            if diagonal is not None:
                return diagonal
        closed_form = self.carrier.metadata.get("closed_form")
        if closed_form == "constant_half" and not _source_transfer_operations(self.carrier) and footprint is None:
            value = complex(self.carrier.metadata.get("closed_form_value", 0.0))
            # A constant Hermitian half-plane is a full constant Fourier
            # plane.  Its sampled height is an impulse, but its continuous
            # periodic derivatives are Dirichlet-kernel derivatives and are
            # non-zero away from the impulse.  Keep the separable structure
            # and evaluate one kernel per axis instead of enumerating modes.
            if abs(value.imag) > max(1.0e-12, abs(value.real) * 1.0e-12):
                # The metadata recognizer only marks valid real constants;
                # keep malformed manually-constructed carriers on the normal
                # validation path rather than silently discarding phase.
                closed_form = None
            else:
                if len(self.shape) != 2:
                    # The closed 2-D half-plane formula below handles the
                    # endpoint orbit convention explicitly.  Keep higher
                    # ranks on the validated sparse path until a matching
                    # multidimensional endpoint consumer exists.
                    closed_form = None
                else:
                    reshape_y: tuple[int, int] | None
                    reshape_x: tuple[int, int] | None
                    if points is None:
                        q_y = np.arange(self.shape[0], dtype=np.float64)
                        q_x = np.arange(self.shape[1], dtype=np.float64)
                        reshape_y, reshape_x = (self.shape[0], 1), (1, self.shape[1])
                    else:
                        q_y = points[:, 0] / self.spacing[0]
                        q_x = points[:, 1] / self.spacing[1]
                        reshape_y = reshape_x = None
                    factor_y, derivative_y = _dirichlet_factor(q_y, self.shape[0])
                    factor_x, derivative_x = _dirichlet_factor(q_x, self.shape[1])
                    scale = value.real / (float(math.prod(self.shape)) if self.normalize else 1.0)
                    if points is None:
                        factor_y = factor_y.reshape(reshape_y)
                        derivative_y = derivative_y.reshape(reshape_y)
                        factor_x = factor_x.reshape(reshape_x)
                        derivative_x = derivative_x.reshape(reshape_x)
                    product = factor_y * factor_x
                count = points.shape[0] if points is not None else int(math.prod(self.shape))
                out_shape = (count,) if points is not None else self.shape
                height = (
                    np.asarray((scale * product).real, dtype=output_dtype).reshape(out_shape) if need_height else None
                )
                gradients: dict[int, np.ndarray] = {}
                for axis in gradient_axes:
                    if axis == 0:
                        component = derivative_y * factor_x / self.spacing[0]
                    elif axis == 1:
                        component = factor_y * derivative_x / self.spacing[1]
                    else:
                        continue
                    gradients[axis] = np.asarray(scale * component, dtype=output_dtype).reshape(out_shape)
                counters.materialized_output_elements += int(count * (int(need_height) + len(gradient_axes)))
                counters.output_fields = int(need_height) + len(gradient_axes)
                counters.record(
                    "ClosedFormDirichlet",
                    value=value.real,
                    points=int(count),
                    dimensions=tuple(int(size) for size in self.shape),
                    shared_gradient=True,
                    stable_sinc_limits=True,
                )
                grad_x = gradients.get(1, gradients.get(0)) if len(self.shape) >= 2 else gradients.get(0)
                grad_y = gradients.get(0) if len(self.shape) >= 2 else None
                return height, grad_x, grad_y
        indices, coefficients, weights, _ = self._dynamic_sparse(counters)
        count = (
            1
            if points is not None and points.shape[0] == 1
            else (points.shape[0] if points is not None else int(math.prod(self.shape)))
        )
        out_shape = (count,) if points is not None else self.shape
        height = np.zeros(out_shape, dtype=output_dtype) if need_height else None
        gradients = {axis: np.zeros(out_shape, dtype=output_dtype) for axis in gradient_axes}
        coords = points if points is not None else None
        grid = None if points is not None else self._grid_coordinates()
        normalization = float(math.prod(self.shape)) if self.normalize else 1.0
        if gradient_axes:
            log_shifts, phase_shifts, zero_flags = self.derivative_operators()
            counters.record(
                "DerivativeLogMul",
                axes=tuple(int(axis) for axis in gradient_axes),
                log_shifts=tuple(tuple(int(x) for x in log_shifts[axis]) for axis in gradient_axes),
                phase_shifts=tuple(tuple(int(x) for x in phase_shifts[axis]) for axis in gradient_axes),
                zero_counts=tuple(int(np.count_nonzero(zero_flags[axis])) for axis in gradient_axes),
                exact_quarter_turn=True,
                execution="numeric_projection_after_packed_decode",
            )
        for _position, (index, coefficient, weight) in enumerate(zip(indices, coefficients, weights, strict=True)):
            idx = tuple(int(x) for x in index)
            frequency = tuple(_signed_frequency(i, n) for i, n in zip(idx, self.shape, strict=True))
            factor = _footprint_factor(footprint, idx, self.shape, self.spacing)
            coefficient = coefficient * factor
            nyquist_axes: tuple[int, ...] = ()
            if coords is not None:
                # Point coordinates are physical ``(y, x, ...)`` positions;
                # grid coordinates below are converted to the same units.
                nyquist_axes = tuple(
                    axis
                    for axis, (freq, size) in enumerate(zip(frequency, self.shape, strict=True))
                    if size % 2 == 0 and abs(freq) == size // 2
                )
                if nyquist_axes:
                    # The real periodic interpolant chooses cos(pi*q) on an
                    # even Nyquist axis.  Integer samples cannot distinguish
                    # exp(+/-i*pi*q), but fractional queries can; factor all
                    # Nyquist axes separately and apply their product rule.
                    ordinary_frequency = np.asarray(
                        [0 if axis in nyquist_axes else freq for axis, freq in enumerate(frequency)],
                        dtype=np.float64,
                    )
                    ordinary_angle = (
                        2.0
                        * math.pi
                        * (
                            coords
                            @ (
                                ordinary_frequency
                                / (
                                    np.asarray(self.shape, dtype=np.float64)
                                    * np.asarray(self.spacing, dtype=np.float64)
                                )
                            )
                        )
                    )
                    ordinary_cosine = np.cos(ordinary_angle)
                    ordinary_sine = np.sin(ordinary_angle)
                    ordinary_real = coefficient.real * ordinary_cosine - coefficient.imag * ordinary_sine
                    ordinary_imaginary = coefficient.real * ordinary_sine + coefficient.imag * ordinary_cosine
                    nyquist_factor = np.ones(count, dtype=np.float64)
                    for axis in nyquist_axes:
                        nyquist_factor *= np.cos(math.pi * coords[:, axis] / self.spacing[axis])
                    real_part = ordinary_real * nyquist_factor
                    imaginary_part = ordinary_imaginary
                else:
                    angle = (
                        2.0
                        * math.pi
                        * (
                            coords
                            @ (
                                np.asarray(frequency, dtype=np.float64)
                                / (
                                    np.asarray(self.shape, dtype=np.float64)
                                    * np.asarray(self.spacing, dtype=np.float64)
                                )
                            )
                        )
                    )
                    cosine = np.cos(angle)
                    sine = np.sin(angle)
                    real_part = coefficient.real * cosine - coefficient.imag * sine
                    imaginary_part = coefficient.real * sine + coefficient.imag * cosine
            else:
                angle = np.zeros(self.shape, dtype=np.float64)
                # Grid coordinates are integer sample indices.  Their phase
                # is independent of physical sample spacing; spacing enters
                # only the derivative multiplier below.  Point coordinates
                # use physical units in the branch above.
                assert grid is not None
                for axis, (freq, size) in enumerate(zip(frequency, self.shape, strict=True)):
                    angle = angle + (2.0 * math.pi * freq / size) * grid[axis]
            if coords is None:
                cosine = np.cos(angle)
                sine = np.sin(angle)
                real_part = coefficient.real * cosine - coefficient.imag * sine
            if height is not None:
                height += (weight / normalization) * real_part
            for axis in gradient_axes:
                freq = frequency[axis]
                if coords is not None and nyquist_axes:
                    if axis in nyquist_axes:
                        derivative_factor = (
                            -math.pi / self.spacing[axis] * np.sin(math.pi * coords[:, axis] / self.spacing[axis])
                        )
                        for other_axis in nyquist_axes:
                            if other_axis != axis:
                                derivative_factor *= np.cos(math.pi * coords[:, other_axis] / self.spacing[other_axis])
                        gradients[axis] += (weight / normalization) * ordinary_real * derivative_factor
                    else:
                        multiplier = 2.0 * math.pi * freq / (self.shape[axis] * self.spacing[axis])
                        gradients[axis] += (weight / normalization) * (-multiplier * imaginary_part * nyquist_factor)
                    continue
                if self.shape[axis] % 2 == 0 and abs(freq) == self.shape[axis] // 2:
                    continue
                multiplier = 2.0 * math.pi * freq / (self.shape[axis] * self.spacing[axis])
                if coords is None:
                    imaginary_part = coefficient.real * sine + coefficient.imag * cosine
                gradients[axis] += (weight / normalization) * (-multiplier * imaginary_part)
            counters.term_evaluations += int(count)
        counters.materialized_output_elements += int(count * (int(need_height) + len(gradient_axes)))
        counters.materialized_intermediate_elements += int(count * (3 + len(gradient_axes)))
        counters.output_fields = int(need_height) + len(gradient_axes)
        # Each term emits one real contribution per requested field and query
        # into the shared accumulators.  ``real_adds`` reflects that concrete
        # loop work (including the first add into a zero buffer); the
        # reduction lower bound excludes that initial term but retains the
        # full Q×field geometry.  No complex reductions are performed after
        # the coefficient's real projection.
        counters.real_adds += int(len(indices) * count * counters.output_fields)
        counters.reduction_operations += int(max(0, len(indices) - 1) * count * max(1, counters.output_fields))
        counters.record(
            "ReduceSigned",
            terms=len(indices),
            outputs=int(counters.output_fields),
            points=int(count),
            shared_phase_basis=True,
        )
        grad_x = gradients.get(1, gradients.get(0)) if len(self.shape) >= 2 else gradients.get(0)
        grad_y = gradients.get(0) if len(self.shape) >= 2 else None
        return height, grad_x, grad_y

    def _dense_half(self, counters: _Counters, output_dtype: np.dtype) -> np.ndarray:
        complex_dtype = np.dtype(np.complex64 if output_dtype == np.dtype(np.float32) else np.complex128)
        if self.half is not None:
            owned = np.asarray(self.half)
            if owned.dtype.kind in "iu":
                # Owned packed SPF32 words: decoding to the dense FFT boundary is a
                # counted conversion, never a numeric cast of the word bits
                # (regression 2026-09-17: the cast produced ~2^31 coefficients).
                packed = np.asarray(owned, dtype=np.uint32)
                counters.record("AddressSupport", source="owned_half_words", count=int(packed.size))
                counters.bytes_read += int(packed.nbytes)
                counters.fallback_conversions += 1
                counters.record(
                    "PackedDecodeReference",
                    count=int(packed.size),
                    destination="dense_fft_boundary",
                    dtype=str(complex_dtype),
                )
                return _decode_words_for_dtype(packed, complex_dtype)
            counters.record("AddressSupport", source="owned_half_spectrum", count=int(owned.size))
            return np.asarray(owned, dtype=complex_dtype).copy()
        if self.carrier.source_kind == "sparse_modes":
            # Numeric authored modes retain their exact compatibility values;
            # the dense conversion is charged here, after route selection.
            from spf.renderer import ocean as o

            if self.modes:
                mode_values = self._current_mode_coefficients()
                indices = self._mode_indices
            else:
                mode_values = np.asarray(
                    [
                        o._coefficient_complex(
                            value, self.carrier.phase_values[i] if self.carrier.phase_values else None
                        )
                        for i, value in enumerate(self.carrier.coefficient_values)
                    ],
                    dtype=np.complex128,
                )
                indices = self.carrier.index_values
            # Build the selected dense carrier directly at its requested
            # complex width.  ``build_half_spectrum`` is intentionally a
            # complex128 compatibility API; using it here would allocate a
            # full wide plane before casting a float32 request.
            terms = o._sparse_terms(self.shape, indices, mode_values, tolerance=self.tolerance)
            half = np.zeros(self.half_shape, dtype=complex_dtype)
            for index, coefficient, _ in terms:
                partner = o._partner(index, self.shape)
                if index[-1] <= self.shape[-1] // 2:
                    destination, stored = index, coefficient
                else:
                    destination, stored = partner, np.conj(coefficient)
                half[destination] = np.asarray(stored, dtype=complex_dtype)
                if o._endpoint_last(destination, self.shape) and destination != o._partner(destination, self.shape):
                    half[o._partner(destination, self.shape)] = np.asarray(np.conj(stored), dtype=complex_dtype)
            counters.record(
                "HermitianSynthesis",
                consumer="compatibility_half_builder",
                elements=int(half.size),
                dtype=str(complex_dtype),
            )
            counters.materialized_intermediate_elements += int(half.size)
            return half
        if _source_transfer_operations(self.carrier):
            view = self.carrier.source_view
            if view is None or not bool(getattr(view, "frequency_domain", False)):
                raise ValueError(
                    "normalized spatial block filtering has no dense ocean consumer; "
                    "select an explicit spatial materialization"
                )
            source_indices, values, _weights, _words = self._dynamic_sparse(counters)
            half = np.zeros(self.half_shape, dtype=complex_dtype)
            from spf.renderer import ocean as o

            for index, value in zip(source_indices, values, strict=True):
                index_tuple = tuple(int(item) for item in index)
                if self.carrier.half_view:
                    destination, stored = index_tuple, value
                elif index_tuple[-1] > self.shape[-1] // 2:
                    destination, stored = o._partner(index_tuple, self.shape), np.conj(value)
                else:
                    destination, stored = index_tuple, value
                half[destination] = np.asarray(stored, dtype=complex_dtype)
                partner = o._partner(destination, self.shape)
                if o._endpoint_last(destination, self.shape) and destination != partner:
                    half[partner] = np.asarray(np.conj(stored), dtype=complex_dtype)
            counters.dense_carrier_elements += int(half.size)
            counters.materialized_intermediate_elements += int(half.size)
            counters.record(
                "HermitianSynthesis",
                consumer="filtered_block_dense_boundary",
                elements=int(half.size),
                dtype=str(complex_dtype),
                source_transfer=_source_transfer_operations(self.carrier),
            )
            return half
        if self.carrier.source_kind == "block_view" and self.modes:
            # A block-backed timeline is bound only after the metadata route
            # is selected.  Keep the addressed SPF words packed through this
            # boundary and compose the current wide phases before the dense
            # reference FFT; decoding a static half plane here would freeze
            # the frame at preparation time.
            packed = self._current_packed_half()
            counters.dense_carrier_elements += int(packed.size)
            counters.packed_words_read += int(packed.size)
            counters.bytes_read += int(packed.nbytes)
            counters.phase_emissions += len(self.modes)
            counters.source_reads += int(self.carrier.active_count)
            counters.record(
                "HermitianSynthesis",
                consumer="block_packed_dense_boundary",
                elements=int(packed.size),
                dtype=str(complex_dtype),
                phase_composed=True,
            )
            counters.materialized_intermediate_elements += int(packed.size)
            decoded = _decode_words_for_dtype(packed, complex_dtype)
            if _view_has_fractional_offsets(self.carrier):
                decoded *= np.asarray(_view_numeric_factor(self.carrier), dtype=complex_dtype)
                counters.record("ViewOffsetCompose", domain="numeric_dense_boundary")
            return decoded
        packed = self.carrier.materialize_half_packed()
        counters.dense_carrier_elements += int(packed.size)
        counters.packed_words_read += int(packed.size)
        counters.bytes_read += int(packed.nbytes)
        counters.fallback_conversions += 1
        counters.record(
            "PackedDecodeReference", count=int(packed.size), destination="dense_fft_boundary", dtype=str(complex_dtype)
        )
        decoded = _decode_words_for_dtype(packed, complex_dtype)
        if _view_has_fractional_offsets(self.carrier):
            decoded *= np.asarray(_view_numeric_factor(self.carrier), dtype=complex_dtype)
            counters.record("ViewOffsetCompose", domain="numeric_dense_boundary")
        return decoded

    def _dense_joint(
        self,
        *,
        points: np.ndarray | None = None,
        need_height: bool,
        gradient_axes: tuple[int, ...],
        footprint: Any,
        output_dtype: np.dtype,
        counters: _Counters,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        half = self._dense_half(counters, output_dtype)
        if footprint is not None:
            covariance = getattr(footprint, "covariance", None)
            if covariance is None and isinstance(footprint, Mapping):
                covariance = footprint.get("covariance")
            if covariance is not None and len(self.shape) == 2:
                fy = fft.fftfreq(self.shape[0], d=self.spacing[0])
                fx = fft.rfftfreq(self.shape[1], d=self.spacing[1])
                yy, xx = np.meshgrid(fy, fx, indexing="ij")
                frequencies = np.stack((xx, yy), axis=-1)
                matrix = _validated_covariance(covariance, 2)
                half *= np.exp(-2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", frequencies, matrix, frequencies))
                counters.materialized_intermediate_elements += int(half.size)
                counters.record("FootprintTransfer", domain="dense_frequency", elements=int(half.size))
            else:
                raise TypeError("dense ocean footprint requires a Gaussian covariance on a 2-D field")
        height = None
        if need_height:
            height = np.asarray(fft.irfftn(half, s=self.shape, workers=1), dtype=output_dtype)
            counters.fft_calls += 1
        gradients: dict[int, np.ndarray] = {}
        for axis in gradient_axes:
            freq = (
                fft.rfftfreq(self.shape[axis], d=self.spacing[axis])
                if axis == len(self.shape) - 1
                else fft.fftfreq(self.shape[axis], d=self.spacing[axis])
            )
            if self.shape[axis] % 2 == 0:
                freq[self.shape[axis] // 2] = 0.0
            dims = [1] * len(self.shape)
            dims[axis] = freq.size
            if output_dtype == np.dtype(np.float32):
                # Keep the derivative multiplier in the same complex width;
                # a float64 scalar would silently promote the carrier back to
                # complex128 before SciPy sees it.
                multiplier: np.ndarray = np.asarray(freq.reshape(dims), dtype=np.float32) * np.float32(2.0 * math.pi)
                multiplier = multiplier.astype(np.complex64) * np.complex64(1j)
            else:
                multiplier = np.asarray(freq.reshape(dims), dtype=np.float64) * (2j * math.pi)
            transformed = np.asarray(half * multiplier, dtype=half.dtype)
            gradients[axis] = np.asarray(fft.irfftn(transformed, s=self.shape, workers=1), dtype=output_dtype)
            counters.fft_calls += 1
        if not self.normalize:
            scale = math.prod(self.shape)
            if height is not None:
                height *= scale
            for value in gradients.values():
                value *= scale
        counters.record(
            "HermitianSynthesis",
            consumer="scipy_irfftn",
            fft_calls=counters.fft_calls,
            carrier_dtype=str(half.dtype),
            output_dtype=str(output_dtype),
        )
        counters.materialized_intermediate_elements += int(half.size * (1 + len(gradient_axes)))
        counters.materialized_output_elements += int(math.prod(self.shape) * (int(need_height) + len(gradient_axes)))
        counters.materialized_intermediate_elements += int(math.prod(self.shape) + half.size * len(gradient_axes))
        counters.output_fields = int(need_height) + len(gradient_axes)
        grad_x = gradients.get(1, gradients.get(0)) if len(self.shape) >= 2 else gradients.get(0)
        grad_y = gradients.get(0) if len(self.shape) >= 2 else None
        if points is not None:
            scaled_points = points / np.asarray(self.spacing, dtype=np.float64)
            rounded = np.rint(scaled_points)
            if not np.allclose(scaled_points, rounded, rtol=0.0, atol=1.0e-12):
                raise ValueError("dense ocean point queries require coordinates on the physical sample-spacing grid")
            selector = rounded.astype(np.int64)
            if np.any(selector < 0) or np.any(selector >= np.asarray(self.shape)):
                raise ValueError("dense ocean point query must lie inside the field extent")
            addresses = tuple(selector[:, axis] for axis in range(len(self.shape)))
            height, grad_x, grad_y = (
                None if value is None else np.asarray(value[addresses], dtype=output_dtype).reshape(-1)
                for value in (height, grad_x, grad_y)
            )
        return height, grad_x, grad_y

    def _packed_mode_support(self) -> tuple[np.ndarray, np.ndarray, bool] | None:
        """Return ``(indices, base_words, temporal)`` for the packed mode composition.

        ``None`` means the carrier owns a static half plane with no addressed
        mode list (``materialize_half_packed`` is the whole composition).
        """

        lowering = self._lowered_source_transfer()
        if lowering is not None:
            indices = np.asarray(lowering.indices, dtype=np.int64)
            words = np.asarray(lowering.words, dtype=np.uint32)
            if self.modes and (
                len(self._mode_indices) != len(indices)
                or not np.array_equal(np.asarray(self._mode_indices, dtype=np.int64), indices)
            ):
                raise ValueError("filtered source transfer support does not match temporal mode state")
            return indices, words, bool(self.modes)
        if not self.modes:
            return None
        _, base_words, _ = self.carrier.materialize_sparse_packed()
        return np.asarray(self._mode_indices, dtype=np.int64), np.asarray(base_words, dtype=np.uint32), True

    def _compose_packed_half(self, phase_codes: np.ndarray | None) -> np.ndarray:
        """Compose addressed mode words (with ``phase_codes``) into an SPF32 half plane."""

        support = self._packed_mode_support()
        if support is None:
            return np.asarray(self.carrier.materialize_half_packed(), dtype=np.uint32).copy()
        indices, words, _temporal = support
        codes = np.zeros(len(indices), dtype=np.uint16) if phase_codes is None else np.asarray(phase_codes)
        return _compose_half_words(words, indices, codes, self.shape)

    def _packed_base_half(self) -> np.ndarray:
        """Static (phase code 0) half plane; the C terminal adds the temporal phases itself."""

        lowering = self._lowered_source_transfer()
        key = (None if lowering is None else id(lowering), self.carrier.source_kind)
        cached = self._packed_base_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        base = self._compose_packed_half(None)
        base.setflags(write=False)
        self._packed_base_cache = (key, base)
        return base

    def _current_packed_half(self) -> np.ndarray:
        """Compose the current temporal phases into an SPF32 half plane."""

        support = self._packed_mode_support()
        if support is None:
            return self._compose_packed_half(None)
        indices, _words, temporal = support
        return self._compose_packed_half(self._phase_codes() if temporal else np.zeros(len(indices), dtype=np.uint16))

    def _packed_native_available(self, use_native: bool | None) -> bool:
        """Whether the C packed grid terminal may execute this route."""

        if use_native is False:
            return False
        from spf.renderer import _ocean_display_native as native

        return native.available()

    def _packed_fb3(
        self,
        *,
        points: np.ndarray | None = None,
        need_height: bool,
        gradient_axes: tuple[int, ...],
        footprint: Any = None,
        output_dtype: np.dtype,
        counters: _Counters,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        from spf.renderer.continuity import packed_ifft2

        if points is not None:
            raise ValueError("packed FB3 ocean queries require a display-side packed query consumer")
        if _view_has_fractional_offsets(self.carrier):
            raise ValueError(
                "packed FB3 cannot preserve fractional BlockView offsets; use the sparse or dense reference route"
            )

        # ``materialize_half_packed`` is intentionally a static carrier cache;
        # temporal modes use a fresh packed phase composition at this boundary.
        transfer_lowering = self._lowered_source_transfer()
        native_route = self._packed_native_available(self._packed_use_native)
        mode_indices: np.ndarray | None = None
        mode_codes: np.ndarray | None = None
        support = self._packed_mode_support()
        if native_route and support is not None and support[2]:
            indices = support[0]
            destinations = _half_destinations(indices, self.shape)
            if np.unique(destinations).size == destinations.size:
                # The C terminal adds the emitted u16 phase codes to the static
                # plane itself.  A duplicated destination would accumulate in C
                # where Python overwrites, so that degenerate list composes here.
                half = self._packed_base_half()
                mode_indices, mode_codes = indices, self._phase_codes()
            else:
                half = self._current_packed_half()
        else:
            half = self._current_packed_half()
        # FB3 is an explicit packed dense route.  The half plane produced at
        # this boundary, the completed full plane below, and each derivative
        # packed plane are real materialized carriers and must be visible in
        # the execution ledger.  This is distinct from the sparse terminal,
        # which never allocates an N-element coefficient plane.
        counters.dense_carrier_elements += int(half.size)
        counters.materialized_intermediate_elements += int(half.size)
        if self.modes:
            counters.phase_emissions += len(self.modes)
            counters.record("PhaseCompose", modes=len(self.modes), master_bits=self.bits, emitted_bits=16)
        counters.source_reads += int(self.carrier.active_count)
        counters.packed_words_read += int(self.carrier.active_count)
        counters.bytes_read += int(self.carrier.active_count) * np.dtype(np.uint32).itemsize
        counters.record("AddressSupport", source=self.carrier.source_kind, count=int(self.carrier.active_count))
        if transfer_lowering is not None:
            counters.record(
                "SourceTransfer",
                operations=transfer_lowering.operations,
                domain="frequency",
                addresses=int(transfer_lowering.active_count),
                dense_intermediate_elements=0,
                coefficient_error_bound=float(transfer_lowering.coefficient_error_bound),
            )
        if footprint is not None:
            covariance = getattr(footprint, "covariance", None)
            if covariance is None and isinstance(footprint, Mapping):
                covariance = footprint.get("covariance")
            if covariance is None:
                raise TypeError("packed FB3 footprint requires a Gaussian covariance")
            matrix = _validated_covariance(covariance, 2)
            yy, xx = np.indices(half.shape, dtype=np.float64)
            fy = np.where(yy <= self.shape[0] // 2, yy, yy - self.shape[0]) / (self.shape[0] * self.spacing[0])
            fx = xx / (self.shape[1] * self.spacing[1])
            frequencies = np.stack((fx, fy), axis=-1)
            transfer = np.exp(-2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", frequencies, matrix, frequencies))
            from spf.renderer.ocean_carrier import _word_from_value

            transfer_words = np.asarray(
                [_word_from_value(v02.LogPos.from_value(float(value))) for value in transfer.reshape(-1)],
                dtype=np.uint32,
            ).reshape(half.shape)
            # Positive Gaussian transfer changes only the log magnitude.  SPF
            # multiplication preserves the packed phase and applies the
            # transfer before Hermitian completion and differentiation.
            if not native_route:
                filtered = np.empty_like(half)
                for address in np.ndindex(half.shape):
                    filtered[address] = np.uint32(v02.spf_mul(int(half[address]), int(transfer_words[address])))
                half = filtered
            counters.materialized_intermediate_elements += int(half.size)
            counters.record(
                "FootprintTransfer",
                domain="packed_frequency",
                elements=int(half.size),
                executor="c" if native_route else "python",
            )
        else:
            transfer_words = None
        if native_route:
            return self._packed_native_grid(
                half,
                need_height=need_height,
                gradient_axes=gradient_axes,
                output_dtype=output_dtype,
                counters=counters,
                mode_indices=mode_indices,
                mode_codes=mode_codes,
                transfer_words=transfer_words,
            )
        self._packed_executor = "python"
        full = _hermitian_complete_words(half, self.shape)
        counters.dense_carrier_elements += int(full.size)
        counters.materialized_intermediate_elements += int(full.size)
        counters.packed_words_read += int(full.size)
        counters.bytes_read += int(full.nbytes)
        counters.record(
            "HermitianSynthesis",
            consumer="continuity.packed_ifft2",
            packed=True,
            materialized_elements=int(full.size),
        )

        def run(words: np.ndarray) -> np.ndarray:
            counters.fft_calls += 1
            spatial = packed_ifft2(words, implementation="v02", precision="accurate")
            return np.asarray(decode_words(spatial).real, dtype=output_dtype)

        height = run(full) if need_height else None
        gradients: dict[int, np.ndarray] = {}
        for axis in gradient_axes:
            derivative = _derivative_words(full, self.shape, self.spacing, axis)
            counters.dense_carrier_elements += int(derivative.size)
            counters.materialized_intermediate_elements += int(derivative.size)
            counters.record("DerivativeLogMul", axis=int(axis), packed=True, materialized_elements=int(derivative.size))
            gradients[axis] = run(derivative)
        transform_count = int(need_height) + len(gradient_axes)
        transform_stages = sum(max(1, int(math.log2(size))) for size in self.shape)
        transform_operations = int(full.size * transform_stages * transform_count)
        counters.real_adds += transform_operations
        counters.reduction_operations += transform_operations
        counters.record(
            "PackedTransformAccounting",
            operation_counts={"radix2_stage_work_lower_bound": transform_operations},
            counter_semantics="packed transform lower bound; implementation counters are not exposed by FB3",
        )
        if not self.normalize:
            scale = math.prod(self.shape)
            if height is not None:
                height *= scale
            for value in gradients.values():
                value *= scale
        counters.materialized_output_elements += int(math.prod(self.shape) * (int(need_height) + len(gradient_axes)))
        counters.output_fields = int(need_height) + len(gradient_axes)
        grad_x = gradients.get(1, gradients.get(0)) if len(self.shape) >= 2 else gradients.get(0)
        grad_y = gradients.get(0) if len(self.shape) >= 2 else None
        return height, grad_x, grad_y

    def _packed_native_grid(
        self,
        half: np.ndarray,
        *,
        need_height: bool,
        gradient_axes: tuple[int, ...],
        output_dtype: np.dtype,
        counters: _Counters,
        mode_indices: np.ndarray | None,
        mode_codes: np.ndarray | None,
        transfer_words: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Execute the packed dense route in C (``libspf_ocean_packed``).

        Every word stage (phase advance, transfer, Hermitian completion,
        derivative, and the v0.2 FB3-Hybrid inverse FFT) is bit-identical to
        the Python oracle in this module / ``continuity.packed_ifft2``; the
        real projection is the only float boundary.  Counters mirror the
        Python route so a report reads the same regardless of executor.
        """

        from spf.renderer import _ocean_display_native as native

        plane = native.make_plane(self.shape, spacing=self.spacing, normalize=self.normalize)
        height32, grad_y32, grad_x32, stats = native.grid(
            half,
            plane,
            need_height=bool(need_height),
            gradient_axes=gradient_axes,
            indices=mode_indices,
            codes=mode_codes,
            transfer_half=transfer_words,
        )
        elements = int(math.prod(self.shape))
        word_bytes = elements * np.dtype(np.uint32).itemsize
        counters.dense_carrier_elements += elements
        counters.materialized_intermediate_elements += elements
        counters.packed_words_read += elements
        counters.bytes_read += word_bytes
        counters.record(
            "HermitianSynthesis",
            consumer=native.CONSUMER,
            packed=True,
            materialized_elements=elements,
            executor="c",
        )
        for axis in gradient_axes:
            counters.dense_carrier_elements += elements
            counters.materialized_intermediate_elements += elements
            counters.record(
                "DerivativeLogMul", axis=int(axis), packed=True, materialized_elements=elements, executor="c"
            )
        transform_count = int(need_height) + len(gradient_axes)
        counters.fft_calls += transform_count
        transform_stages = sum(max(1, int(math.log2(size))) for size in self.shape)
        transform_operations = int(elements * transform_stages * transform_count)
        counters.real_adds += transform_operations
        counters.reduction_operations += transform_operations
        stats_dict = stats.to_dict()
        fft_profile = native.plane_profile_name(plane)
        counters.record(
            "PackedTransformAccounting",
            operation_counts={
                "radix2_stage_work_lower_bound": transform_operations,
                "fft_butterflies": int(stats_dict["fft_butterflies"]),
                "fft_transforms": int(stats_dict["fft_transforms"]),
                "fft_direct_pairs": int(stats_dict["fft_direct_pairs"]),
                "fft_local_pairs": int(stats_dict["fft_local_pairs"]),
                "fft_sentinel_pairs": int(stats_dict["fft_sentinel_pairs"]),
            },
            counter_semantics="packed transform lower bound plus native butterfly counts",
        )
        counters.record(
            "NativePackedOcean",
            consumer=native.CONSUMER,
            stats=stats_dict,
            fft_profile=fft_profile,
            fft_threads=int(plane.threads) or native.default_threads(),
            transform_oracle=(
                "spf.renderer.continuity.packed_ifft2 (v02 FB3-Hybrid, bit-identical)"
                if int(plane.fft_profile) == native.FFT_PROFILES["fb3"]
                else "spf.tensor.v02.spf_ifft_radix2 (FB3-Hybrid, tolerance graded)"
            ),
            phase_advance="c_uint16_add" if mode_codes is not None else "python_composed_half",
            cartesian_intermediate_elements=0,
        )
        counters.materialized_output_elements += int(elements * transform_count)
        counters.output_fields = transform_count
        self._packed_executor = "c"

        def cast(value: np.ndarray | None) -> np.ndarray | None:
            return None if value is None else np.asarray(value, dtype=output_dtype)

        return cast(height32), cast(grad_x32), cast(grad_y32)

    def _canonical_analytic_prepared(
        self,
        *,
        footprint: Any = None,
        output_dtype: Any = np.float64,
        layout_sampling: bool = False,
    ) -> Any:
        """Return the cached canonical consumer for an analytic carrier.

        ``PreparedOcean`` is the public compatibility terminal, while
        ``PreparedAnalyticOcean`` owns the source semantics.  Keep this bridge
        lazy and keyed by the source/footprint/output contract so point/image
        consumers reuse the same compact preparation without a grid emission.
        ``layout_sampling`` is part of the key because terminal raster spacing
        is an override for a Field chart; physical point/native queries keep
        the Field chart unchanged.
        """

        payload = self.carrier.analytic_payload
        if payload is None:
            raise ValueError("analytic ocean route has no analytic payload")
        dtype = np.dtype(output_dtype)
        key = (
            id(payload),
            _footprint_cache_key(footprint),
            dtype.str,
            bool(layout_sampling),
            tuple(self.shape),
            tuple(self.spacing),
        )
        cached = self._analytic_prepared_cache.get(key)
        if cached is not None:
            return cached

        # A canonical prepared analytic source already carries its chart,
        # support, residual ledger, and layout.  Reuse it directly instead of
        # wrapping it in another preparation.
        analytic = payload if callable(getattr(payload, "execute_fields", None)) else None
        if analytic is None:
            from spf.renderer.model import Field
            from spf.renderer.ocean_analytic import prepare_analytic_ocean

            field_source = payload if isinstance(payload, Field) else None
            spacing_override: tuple[float, float] | None = None
            if field_source is not None and layout_sampling and any(value != 1.0 for value in self.spacing):
                # Terminal spacing is in (row/y, column/x) order; the
                # canonical chart constructor performs the corresponding
                # raster-to-(x,y) conversion while retaining Field metadata.
                spacing_override = (float(self.spacing[0]), float(self.spacing[1]))
            analytic_kwargs: dict[str, Any] = {
                "shape": self.shape,
                "output_dtype": dtype,
                "output_format": "real",
                "footprint": footprint,
            }
            if spacing_override is not None:
                analytic_kwargs["spacing"] = spacing_override
            analytic = prepare_analytic_ocean(payload, **analytic_kwargs)

        self._analytic_prepared_cache[key] = analytic
        return analytic

    def _analytic_joint(
        self,
        *,
        points: np.ndarray | None,
        footprint: Any = None,
        need_height: bool,
        gradient_axes: tuple[int, ...],
        output_dtype: np.dtype,
        counters: _Counters,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Consume the compact analytic carrier through the canonical route.

        ``PreparedOcean`` keeps the historical ocean coordinate contract:
        point queries are physical ``(y, x)`` and layout arrays are ordered
        ``(row=y, column=x)``.  The analytic consumer uses named physical
        ``(x, y)`` points and a ``SampleLayout``.  Keep that translation at
        this boundary so atoms, fields, guarded jets, support, footprints,
        and residual policies all share one implementation.
        """

        payload = self.carrier.analytic_payload
        if payload is None:
            raise ValueError("analytic ocean route has no analytic payload")
        if len(self.shape) != 2:
            raise ValueError("analytic ocean consumers require a two-dimensional shape")

        from spf.renderer.model import Field, SampleLayout

        field_source = payload if isinstance(payload, Field) else None
        tile_source = payload
        if field_source is not None:
            tile_source = getattr(field_source.representation, "payload", payload)
        is_tile = type(tile_source).__name__ == "SpectralJetTile2D"
        layout: Any = None
        out_shape: tuple[int, ...]
        if points is None and field_source is None:
            origin = (0.0, 0.0)
            if is_tile:
                tile_origin = getattr(tile_source, "origin", (0.0, 0.0))
                origin = (float(tile_origin[1]), float(tile_origin[0]))
            layout = SampleLayout(self.shape, origin=origin, spacing=self.spacing)

        analytic = self._canonical_analytic_prepared(
            footprint=footprint,
            output_dtype=output_dtype,
            layout_sampling=points is None,
        )

        # Save a per-call delta of the canonical counters.  The analytic
        # consumer owns the actual atom/jet evaluations; terminal counters
        # expose those measurements instead of inventing byte or DRAM totals.
        canonical_counters = getattr(analytic, "counters", None)
        before_counts: dict[str, int] = {}
        if canonical_counters is not None:
            as_dict = getattr(canonical_counters, "as_dict", None)
            if callable(as_dict):
                before_counts = {str(key): int(value) for key, value in as_dict().items()}

        if points is None:
            frame = cast(
                Any,
                analytic.execute_fields(
                    layout=layout,
                    output_dtype=output_dtype,
                    footprint=footprint,
                ),
            )
            out_shape = self.shape
            sample_count = int(math.prod(out_shape))
        else:
            # Public PreparedOcean points are physical (y, x).  The canonical
            # physical evaluator takes (x, y), so reverse exactly once.
            physical_xy = np.asarray(points[:, ::-1], dtype=np.float64)
            frame = cast(
                Any,
                analytic.execute_fields(
                    points=physical_xy,
                    output_dtype=output_dtype,
                    footprint=footprint,
                ),
            )
            out_shape = (points.shape[0],)
            sample_count = int(points.shape[0])

        after_counts: dict[str, int] = {}
        if canonical_counters is not None:
            as_dict = getattr(canonical_counters, "as_dict", None)
            if callable(as_dict):
                after_counts = {str(key): int(value) for key, value in as_dict().items()}
        counter_delta = {
            key: max(0, int(value) - int(before_counts.get(key, 0))) for key, value in after_counts.items()
        }
        counters.term_evaluations += int(
            counter_delta.get("atom_evaluations", 0) or counter_delta.get("jet_samples", 0)
        )
        counters.complex_adds += int(counter_delta.get("complex_pair_adds", 0))
        counters.reduction_operations += int(counter_delta.get("pivot_reductions", 0))
        counters.fallback_conversions += int(counter_delta.get("fallback_conversions", 0))

        requested_fields = int(need_height) + int(0 in gradient_axes) + int(1 in gradient_axes)
        # The canonical frame allocates height, both gradients, and a residual
        # channel.  Requested fields are terminal outputs; the rest remains
        # bounded scratch and is charged as an intermediate allocation.
        counters.materialized_output_elements += int(sample_count * requested_fields)
        counters.materialized_intermediate_elements += int(sample_count * max(0, 4 - requested_fields))
        counters.output_fields = requested_fields

        report = getattr(frame, "report", None)
        report_ledger = getattr(report, "ledger", None)
        report_diagnostics = getattr(report, "diagnostics", {})
        ledger_fields = None
        if report_ledger is not None:
            ledger_fields = {
                name: float(getattr(report_ledger, name))
                for name in (
                    "source_quantization",
                    "representation_fit",
                    "pruning",
                    "reducer",
                    "transform",
                    "materialization",
                    "display",
                )
                if hasattr(report_ledger, name)
            }
        counters.record(
            "AnalyticAtomJet",
            source_type=type(payload).__name__,
            points=sample_count,
            shared_gradient=True,
            coordinate_contract="terminal_physical_yx_to_analytic_physical_xy",
            canonical_counters=counter_delta,
            canonical_error_ledger=ledger_fields,
            canonical_diagnostics=dict(report_diagnostics),
        )

        height = np.asarray(frame.height, dtype=output_dtype).reshape(out_shape) if need_height else None
        grad_x = np.asarray(frame.gradient_x, dtype=output_dtype).reshape(out_shape) if 1 in gradient_axes else None
        grad_y = np.asarray(frame.gradient_y, dtype=output_dtype).reshape(out_shape) if 0 in gradient_axes else None
        return height, grad_x, grad_y

    def _render_report(
        self,
        counters: _Counters,
        output_dtype: np.dtype,
        *,
        points: np.ndarray | None,
        footprint: Any,
        profile: str | None = None,
        consumer: str | None = None,
        native: bool = False,
        ledger: v02.ErrorLedger | None = None,
        footprint_lowering: FilteredSourceLowering | None = None,
    ) -> Any:
        from spf.renderer import ocean as o

        counters.bytes_written = int(counters.materialized_output_elements * output_dtype.itemsize)
        diagnostics = {
            "prepared": True,
            "joint_terminal": True,
            "bound": self.plan.bound,
            "bound_request": None if self.plan.request is None else self.plan.request.to_dict(),
            "bound_executor": self.plan.executor,
            "selected_kernel": self.plan.kernel,
            "route": self.plan.route,
            "requested_method": self.plan.requested_method,
            "selected_method": self.plan.method,
            "profile": profile or self.plan.profile,
            "consumer": consumer or self._consumer_name(),
            "output_dtype": str(output_dtype),
            "actual_operations": tuple(counters.trace or ()),
            "operation_trace": tuple(counters.trace or ()),
            "trace_matches_bound_kernel": bool(
                self.plan.kernel is None
                or any(
                    item.get("opcode") == "BoundExecution" and item.get("kernel") == self.plan.kernel
                    for item in (counters.trace or ())
                )
            ),
            "actual_counters": {
                "source_reads": counters.source_reads,
                "packed_words_read": counters.packed_words_read,
                "phase_updates": counters.phase_updates,
                "phase_emissions": counters.phase_emissions,
                "term_evaluations": counters.term_evaluations,
                "real_adds": counters.real_adds,
                "complex_adds": counters.complex_adds,
                "fft_calls": counters.fft_calls,
                "dense_carrier_elements": counters.dense_carrier_elements,
                "materialized_intermediate_elements": counters.materialized_intermediate_elements,
                "materialized_output_elements": counters.materialized_output_elements,
                "bytes_read": counters.bytes_read,
                "bytes_read_semantics": "logical_source_storage_bytes_not_total_dram_traffic",
                "bytes_written": counters.bytes_written,
                "reduction_operations": counters.reduction_operations,
                "fallback_conversions": counters.fallback_conversions,
            },
            "points": None if points is None else int(points.shape[0]),
            "footprint": None if footprint is None else type(footprint).__name__,
            "structure_preserved": self.plan.structure_preserved,
            "source_metadata": dict(self.carrier.metadata),
            "structural_program": None if self.plan.program is None else self.plan.program.to_dict(),
            "cost_model": self.plan.diagnostics.get("cost_model"),
            "dense_carrier_allocated": bool(counters.dense_carrier_elements),
            "hardware_native": bool(native),
            "byte_count_semantics": "logical_source_storage_bytes_not_total_dram_traffic",
        }
        lowering = self._native_transfer_lowering
        if lowering is not None:
            diagnostics["source_transfer_lowering"] = lowering.to_dict()
            diagnostics["source_transfer_error_bound"] = float(lowering.output_error_bound)
        if footprint_lowering is not None:
            diagnostics["native_footprint_lowering"] = footprint_lowering.to_dict()
            diagnostics["native_footprint_error_bound"] = float(footprint_lowering.output_error_bound)
        report = o._ocean_render_report(
            backend=getattr(self.backend, "name", str(self.backend)),
            shape=self.shape if points is None else (int(points.shape[0]),),
            method=self.plan.method,
            modes=self.plan.active_modes,
            fft_count=counters.fft_calls,
            fallback=self.plan.fallback,
            diagnostics=diagnostics,
            materialized_elements=counters.materialized_output_elements,
            potential_elements=int(math.prod(self.shape) if points is None else points.shape[0]),
            complex_adds=counters.complex_adds,
            field_operations=counters.output_fields,
            reduction_operations=counters.reduction_operations,
            bytes_read=counters.bytes_read,
            bytes_written=counters.bytes_written,
            materialized_intermediate_elements=counters.materialized_intermediate_elements,
            output_dtype=str(output_dtype),
            profile=profile or self.plan.profile,
            consumer=consumer or self._consumer_name(),
            ledger=ledger,
        )
        self._last_report = report
        self.report = report
        return report

    @staticmethod
    def _analytic_native_points(sampler: Any, points: np.ndarray, *, need_gradients: bool) -> np.ndarray:
        return np.asarray(
            [sampler.sample(float(point[0]), float(point[1]), need_gradients=need_gradients) for point in points],
            dtype=np.float64,
        ).reshape(-1, 3)

    def execute_fields(
        self,
        *,
        need_height: bool = True,
        need_gradients: bool = True,
        points: Any = None,
        footprint: Any = None,
        output_dtype: Any = np.float32,
        spacing: Any = None,
        return_report: bool = False,
        gradient_axes: Sequence[int] | None = None,
        use_native: bool | None = None,
    ) -> OceanFieldResult | tuple[OceanFieldResult, Any]:
        dtype = _dtype(output_dtype, np.float32)
        if spacing is not None:
            self.spacing = _spacing(self.shape, spacing)
        point_array = None if points is None else _validate_points(points, len(self.shape))
        axes = (
            ()
            if not need_gradients
            else tuple(range(len(self.shape)))
            if gradient_axes is None
            else tuple(int(x) % len(self.shape) for x in gradient_axes)
        )
        if any(axis < 0 or axis >= len(self.shape) for axis in axes):
            raise ValueError("ocean gradient axis is outside the field shape")
        selected_outputs: list[str] = []
        if need_height:
            selected_outputs.append("height")
        if need_gradients and 0 in axes:
            selected_outputs.append("grad_y")
        if need_gradients and len(self.shape) >= 2 and 1 in axes:
            selected_outputs.append("grad_x")
        if not selected_outputs:
            raise ValueError("ocean execution must request height or at least one gradient")
        backend_name = str(getattr(self.backend, "name", self.backend)).lower().replace("-", "_")
        output_request = OceanOutputRequest(
            sampling="points" if point_array is not None else "grid",
            outputs=tuple(selected_outputs),
            footprint=footprint,
            backend=backend_name,
            output_dtype=str(dtype),
            point_count=None if point_array is None else int(point_array.shape[0]),
            native=use_native,
        )
        counters = _Counters()
        counters.phase_updates = int(self._pending_phase_updates)
        self._pending_phase_updates = 0
        # The planner owns route selection.  An optimistic availability bit
        # keeps library loading out of preparation; a missing library causes
        # one explicit rebind to the reference executor below.
        cuda_requested = backend_name == "cuda"
        if cuda_requested and use_native is True and self.carrier.source_kind == "analytic":
            raise ValueError(
                "CUDA ocean has no analytic atom/jet consumer; native ocean consumer is unavailable; "
                "packed SPF32 support is required"
            )
        bound = self.bind_execution(output_request, native_available=True, use_native=use_native)
        if cuda_requested and not bound.native:
            raise ValueError("CUDA ocean backend requires a compact packed-SPF32 native route for this request")
        height: np.ndarray | None = None
        grad_x: np.ndarray | None = None
        grad_y: np.ndarray | None = None
        native_used = False
        native_profile = None
        native_consumer = None
        native_footprint_lowering: FilteredSourceLowering | None = None
        execution_ledger: v02.ErrorLedger | None = None
        if bound.native:
            try:
                sampler = self.native_sampler(footprint=footprint)
            except (ImportError, ModuleNotFoundError, OSError, RuntimeError):
                # An explicit CUDA backend is a placement/consumer request,
                # so a missing compiler, runtime, or kernel cannot silently
                # relabel the following CPU reference call as CUDA.
                if use_native is True or cuda_requested:
                    raise
                sampler = None
            if sampler is not None:
                requested_gradient_y = bool(need_gradients and 0 in axes)
                requested_gradient_x = bool(need_gradients and len(self.shape) >= 2 and 1 in axes)
                requested_fields = int(need_height) + int(requested_gradient_y) + int(requested_gradient_x)
                if self.carrier.source_kind == "analytic":
                    if point_array is None:
                        raise RuntimeError("analytic native ocean callbacks are point-only")
                    native_array = counters.invoke(
                        self._analytic_native_points,
                        str(bound.executor),
                        sampler,
                        point_array,
                        need_gradients=bool(need_gradients),
                    )
                    height = native_array[:, 0].astype(dtype, copy=False) if need_height else None
                    grad_y = native_array[:, 1].astype(dtype, copy=False) if requested_gradient_y else None
                    grad_x = native_array[:, 2].astype(dtype, copy=False) if requested_gradient_x else None
                    sample_count = int(point_array.shape[0])
                    native_stats = sampler.stats
                    stats_dict = native_stats.to_dict()
                    active_modes = int(stats_dict.get("source_terms", 0))
                    counters.source_reads = active_modes
                    counters.packed_words_read = 0
                    counters.bytes_read = 0
                    counters.phase_emissions = 0
                    counters.term_evaluations = int(stats_dict.get("samples", sample_count))
                    counters.real_adds = max(0, counters.term_evaluations - sample_count)
                    counters.reduction_operations = counters.real_adds
                else:
                    if point_array is None:
                        native_result, native_stats = counters.invoke(
                            sampler.grid, str(bound.executor), return_stats=True
                        )
                        height = np.asarray(native_result[0], dtype=dtype) if need_height else None
                        grad_y = np.asarray(native_result[1], dtype=dtype) if requested_gradient_y else None
                        grad_x = np.asarray(native_result[2], dtype=dtype) if requested_gradient_x else None
                        sample_count = int(math.prod(self.shape))
                    else:
                        native_result, native_stats = counters.invoke(
                            sampler.points, str(bound.executor), point_array, return_stats=True
                        )
                        native_array = np.asarray(native_result, dtype=dtype)
                        height = native_array[:, 0] if need_height else None
                        grad_y = native_array[:, 1] if requested_gradient_y else None
                        grad_x = native_array[:, 2] if requested_gradient_x else None
                        sample_count = int(point_array.shape[0])
                    counters.source_reads = int(native_stats.active_modes)
                    counters.packed_words_read = int(native_stats.active_modes)
                    counters.bytes_read = int(native_stats.active_modes * 4)
                    counters.phase_emissions = int(native_stats.phase_emissions)
                    stats_dict = native_stats.to_dict()
                    # Integer phase grids charge LUT evaluations and
                    # recurrence steps rather than libm trig calls.
                    phase_evaluations = int(
                        stats_dict.get("phase_lut_evaluations", 0)
                        + stats_dict.get("phase_grid_steps", 0)
                        + stats_dict.get("trig_evaluations", 0)
                    )
                    logical_terms = int(native_stats.active_modes) * int(sample_count)
                    counters.term_evaluations = max(phase_evaluations, logical_terms)
                    counters.real_adds = (
                        max(0, int(native_stats.active_modes) - 1)
                        * int(sample_count)
                        * max(1, int(need_height) + 2 * int(need_gradients))
                    )
                    counters.reduction_operations = counters.real_adds
                    try:
                        native_ledger = sampler.error_ledger()
                        native_bound = max(
                            float(native_ledger.height_absolute_bound),
                            float(native_ledger.gradient_y_absolute_bound),
                            float(native_ledger.gradient_x_absolute_bound),
                        )
                        lowering = self._native_transfer_lowering
                        native_footprint_lowering = self._native_gaussian_footprint_lowering(footprint)
                        active_lowering = native_footprint_lowering or lowering
                        lowering_bound = (
                            0.0 if active_lowering is None else float(active_lowering.output_error_bound)
                        )
                        execution_ledger = v02.ErrorLedger(
                            source_quantization=lowering_bound,
                            reducer=max(0.0, native_bound - lowering_bound),
                        )
                        counters.record("NativeErrorLedger", **native_ledger.to_dict())
                    except (AttributeError, TypeError, ValueError, OverflowError):
                        # The sampler remains the authoritative consumer; an
                        # unavailable diagnostic ledger keeps the report's
                        # uncertainty explicit through the default boundary.
                        execution_ledger = None
                    lowering = self._native_transfer_lowering
                    if lowering is not None:
                        counters.record(
                            "SourceTransfer",
                            operations=lowering.operations,
                            domain="frequency",
                            addresses=int(lowering.active_count),
                            dense_intermediate_elements=0,
                            coefficient_error_bound=float(lowering.coefficient_error_bound),
                        )
                    if native_footprint_lowering is not None:
                        counters.record(
                            "FootprintTransfer",
                            operations=native_footprint_lowering.operations,
                            domain="packed_frequency",
                            addresses=int(native_footprint_lowering.active_count),
                            dense_intermediate_elements=0,
                            covariance_composed_once=True,
                            coefficient_error_bound=float(native_footprint_lowering.coefficient_error_bound),
                        )
                counters.materialized_output_elements = int(sample_count * requested_fields)
                counters.materialized_intermediate_elements += int(sample_count * max(0, 3 - requested_fields))
                counters.output_fields = requested_fields
                counters.record(
                    "NativeFusedOcean",
                    stats=stats_dict,
                    outputs=tuple(
                        name
                        for name, selected in (
                            ("height", bool(need_height)),
                            ("grad_y", requested_gradient_y),
                            ("grad_x", requested_gradient_x),
                        )
                        if selected
                    ),
                    native_emitted_planes=("height", "grad_y", "grad_x"),
                    unrequested_planes_materialized=max(0, 3 - requested_fields),
                    operation_counts={
                        "phase_lut_evaluations": int(stats_dict.get("phase_lut_evaluations", 0)),
                        "phase_grid_steps": int(stats_dict.get("phase_grid_steps", 0)),
                        "trig_evaluations": int(stats_dict.get("trig_evaluations", 0)),
                        "term_evaluations": int(counters.term_evaluations),
                        "accumulator_additions_lower_bound": int(counters.real_adds),
                    },
                    counter_semantics="native counters plus logical lower bounds",
                )
                native_used = True
                native_profile = bound.profile
                native_consumer = bound.kernel
            else:
                # Rebind once so the report and trace describe the reference
                # invocation that actually follows the unavailable native
                # candidate.
                bound = self.bind_execution(output_request, native_available=False, use_native=False)
        if not native_used:
            # Resolve the selected route once, then invoke and trace that
            # exact bound callable. Each consumer owns its internal algorithm.
            consumers: dict[str, Callable[..., Any]] = {
                "analytic": self._analytic_joint,
                "irfft": self._dense_joint,
                "fb3": self._packed_fb3,
                "sparse": self._sparse_joint,
            }
            consumer = consumers[bound.method]
            self._packed_use_native = use_native
            self._packed_executor = None
            height, grad_x, grad_y = counters.invoke(
                consumer,
                str(bound.executor),
                points=point_array,
                footprint=footprint,
                need_height=bool(need_height),
                gradient_axes=axes,
                output_dtype=dtype,
                counters=counters,
            )
            if bound.method == "fb3" and self._packed_executor == "c":
                from spf.renderer import _ocean_display_native as packed_native

                # The packed words were consumed by libspf_ocean_packed; the
                # report names that C consumer, not the Python oracle.
                native_used = True
                native_profile = "packed_spf32_words_native_c"
                native_consumer = packed_native.CONSUMER
            if bound.method == "fb3":
                lowering = self._lowered_source_transfer()
                if lowering is not None:
                    # FB3's packed transform and derivative stages do not
                    # publish a finite executable certificate yet.  Retain
                    # the finite transfer lowering bound while leaving the
                    # transform component explicitly unbounded.
                    execution_ledger = v02.ErrorLedger(
                        source_quantization=float(lowering.output_error_bound),
                        transform=math.inf,
                    )
        counters.record(
            "Materialize",
            output_dtype=str(dtype),
            query=point_array is not None,
            outputs=tuple(selected_outputs),
            **(
                {"coordinate_contract": "terminal_physical_yx_to_analytic_physical_xy"}
                if self.plan.method == "analytic"
                else {}
            ),
        )
        report = self._render_report(
            counters,
            dtype,
            points=point_array,
            footprint=footprint,
            profile=native_profile,
            consumer=native_consumer,
            native=native_used,
            ledger=execution_ledger,
            footprint_lowering=native_footprint_lowering,
        )
        result = OceanFieldResult(
            height,
            grad_x,
            grad_y,
            report,
            str(dtype),
            None if point_array is None else point_array.copy(),
            footprint,
            tuple(self.operations),
            tuple(MappingProxyType(item) for item in (counters.trace or ())),
            MappingProxyType({"shape": self.shape, "spacing": self.spacing, "route": self.plan.route}),
        )
        return (result, report) if return_report else result

    def execute_samples(
        self, points: Any, *, output_dtype: Any = np.float32, footprint: Any = None, return_report: bool = False
    ) -> Any:
        result = self.execute_fields(
            need_height=True,
            need_gradients=False,
            points=points,
            output_dtype=output_dtype,
            footprint=footprint,
        )
        assert isinstance(result, OceanFieldResult)
        return (result.height, result.report) if return_report else result.height

    def execute(
        self,
        *,
        return_report: bool = False,
        derivative_axis: int | None = None,
        spacing: Any = None,
    ) -> Any:
        if derivative_axis is None:
            result = self.execute_fields(
                need_height=True,
                need_gradients=False,
                output_dtype=np.float64,
                spacing=spacing,
            )
            assert isinstance(result, OceanFieldResult)
            from spf.renderer import ocean as o

            output = o._format_sparse_output(np.asarray(result.height, dtype=np.float64), self.output_kind)
            return (output, result.report) if return_report else output
        axis = int(derivative_axis)
        if axis != derivative_axis or not -len(self.shape) <= axis < len(self.shape):
            raise ValueError("derivative axis is outside the ocean shape")
        axis %= len(self.shape)
        result = self.execute_fields(
            need_height=False,
            need_gradients=True,
            gradient_axes=(axis,),
            output_dtype=np.float64,
            spacing=spacing,
        )
        assert isinstance(result, OceanFieldResult)
        selected = result.grad_y if len(self.shape) >= 2 and axis == 0 else result.grad_x
        from spf.renderer import ocean as o

        output = o._format_sparse_output(np.asarray(selected, dtype=np.float64), self.output_kind)
        return (output, result.report) if return_report else output

    render = execute


def prepare_ocean(
    shape: Any,
    coefficients: Any = None,
    indices: Any = None,
    *,
    half_spectrum: Any = None,
    phases: Any = None,
    method: Any = "auto",
    normalize: bool | None = None,
    output_kind: Any = "native",
    tolerance: float = 1e-12,
    spacing: Any = None,
    backend: Any = "reference",
    allow_dense: bool = True,
    profile: str | None = None,
) -> PreparedOcean:
    from spf.renderer import ocean as o
    from spf.renderer.backends import get_backend

    selected_backend = get_backend(backend)
    selected_backend.require_available()
    modes: tuple[Any, ...] = ()
    if isinstance(shape, o.OceanSpectrum):
        spectrum = shape
        shape = spectrum.shape
        if normalize is None:
            normalize = spectrum.normalize
        if coefficients is None and indices is None and half_spectrum is None:
            modes = tuple(spectrum.modes)
            if modes:
                indices, coefficients = spectrum.indices, spectrum.coefficients
            else:
                indices, coefficients, phases = spectrum.indices, spectrum.coefficients, spectrum.phases or None
                half_spectrum = spectrum.half_spectrum
    # Preparation owns a private timeline.  The public OceanMode accumulators
    # are updated exactly once by sync_phases() at the API boundary; mutating
    # the caller's objects during each frame would double-count wraps.
    owned_modes = tuple(
        o.OceanMode(mode.index, mode.coefficient, mode.phase_accumulator.copy(), self_conjugate=mode.self_conjugate)
        for mode in modes
    )
    full_shape = o._normalise_shape(shape)
    if normalize is None:
        normalize = True
    if spacing is not None:
        _spacing(full_shape, spacing)
    carrier = OceanCarrier.from_inputs(
        full_shape,
        coefficients=coefficients,
        indices=indices,
        phases=phases,
        half_spectrum=half_spectrum,
        modes=owned_modes,
    )
    terms: tuple[Any, ...] = ()
    if carrier.source_kind == "sparse_modes":
        terms = o._sparse_terms(
            full_shape,
            carrier.index_values,
            carrier.coefficient_values,
            phases=carrier.phase_values or None,
            tolerance=tolerance,
        )
    from spf.renderer.ocean_planner import plan_ocean_carrier

    plan = plan_ocean_carrier(
        carrier,
        method=method,
        backend=selected_backend,
        allow_dense=allow_dense,
        profile=profile,
        need_gradients=True,
    )
    if carrier.source_kind == "block_view" and phases is not None:
        # Planning above used the block header/support certificate only.  Once
        # the route is fixed, bind the caller's temporal phase stream to the
        # addressed words.  This keeps the source view and its residual/error
        # metadata intact while giving every selected consumer the same owned
        # wide timeline.  The first addressed read is therefore a preparation
        # boundary operation, never planner-time coefficient expansion.
        from spf.renderer.ocean_carrier import _apply_integral_view_offsets, _validated_spf32_words

        source_view = carrier.source_view
        assert source_view is not None
        source_addresses = np.asarray(source_view.active_addresses(), dtype=np.int64).reshape(-1)
        source_words = _validated_spf32_words(
            source_view.sample_words(source_addresses, phase=True),
            source="addressed SPF32 view",
        ).reshape(-1)
        source_words = _apply_integral_view_offsets(source_words, source_view)
        phase_values = tuple(carrier.phase_values)
        if len(phase_values) != len(source_addresses):
            raise ValueError("ocean block phase state count must match addressed support count")
        source_shape = tuple(int(value) for value in source_view.shape)
        phase_by_index = {
            tuple(int(value) for value in np.unravel_index(int(address), source_shape)): phase_value
            for address, word, phase_value in zip(source_addresses, source_words, phase_values, strict=True)
            if int(word) != int(v02.SPF32_WORD_ZERO)
        }
        addresses, base_words, _ = carrier.materialize_sparse_packed()
        block_modes: list[o.OceanMode] = []
        for address, word in zip(addresses, base_words, strict=True):
            index = tuple(int(value) for value in address)
            try:
                phase_value = phase_by_index[index]
            except KeyError as exc:
                raise ValueError("ocean block phase state has no value for an addressed packed word") from exc
            if isinstance(phase_value, o.WidePhaseAccumulator):
                accumulator = phase_value.copy()
            elif isinstance(phase_value, v02.Phase):
                accumulator = o.WidePhaseAccumulator(int(phase_value.T), 0, v02.SPF32_PHASE_BITS)
            elif isinstance(phase_value, (int, np.integer)) and not isinstance(phase_value, (bool, np.bool_)):
                accumulator = o.WidePhaseAccumulator(int(phase_value), 0, v02.SPF32_PHASE_BITS)
            else:
                radians = o._phase_radians(phase_value)
                code = round(radians * v02.SPF32_PHASE_MOD / (2.0 * math.pi))
                accumulator = o.WidePhaseAccumulator(code, 0, v02.SPF32_PHASE_BITS)
            block_modes.append(o.OceanMode(index, int(word), phase=accumulator))
        carrier.modes = tuple(block_modes)
        owned_modes = tuple(block_modes)
        # Rebuild only the typed program after the phase inputs become known;
        # route selection and all cost decisions remain the metadata-only
        # result above.
        plan = plan_ocean_carrier(
            carrier,
            method=method,
            backend=selected_backend,
            allow_dense=allow_dense,
            profile=profile,
            need_gradients=True,
        )
    owned_half = None
    if carrier.source_kind == "half_spectrum" and carrier.half_spectrum is not None:
        owned_half = np.array(carrier.half_spectrum, copy=True)
    return PreparedOcean(
        full_shape,
        owned_half,
        plan,
        normalize=bool(normalize),
        output_kind=output_kind,
        backend=selected_backend,
        tolerance=tolerance,
        modes=owned_modes,
        terms=terms,
        carrier=carrier,
        spacing=spacing,
    )


__all__ = ["OceanFieldResult", "PreparedOcean", "prepare_ocean"]
