"""Ocean and real-Fourier rendering paths for SFR v0.2.

The ocean path deliberately keeps the control state and the field arithmetic
separate.  A mode owns a wide, modular phase accumulator; only a rounded u16
phase is emitted when an SPF value is requested.  A real spatial field owns a
Hermitian half spectrum.  Its negative-frequency partner is reconstructed at
the materialization boundary, or is consumed directly by the signed-real
sparse synthesizer.

The module is a reference implementation.  It uses NumPy for dense
``irfftn`` fallbacks and the already-tested typed primitives in
``spf.tensor.v02`` for phase codes, SPF words, and signed-real values.  The
reports intentionally expose which path was selected and whether a dense
fallback was needed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.renderer.backends import SFRBackend
from spf.renderer.carriers import decode_words, encode_words
from spf.renderer.ocean_carrier import OceanCarrier
from spf.renderer.ocean_planner import (
    OceanExecutionPlan,
    OceanInstruction,
    OceanProgram,
    plan_ocean,
    plan_ocean_carrier,
)
from spf.renderer.ocean_terminal import PreparedOcean, prepare_ocean
from spf.renderer.report import (
    CompilerPassRecord,
    RenderMetrics,
    RenderReport,
    relative_l2,
    temporal_difference_error,
)
from spf.tensor import v02

__all__ = [
    "HermitianHalfSpectrum",
    "HermitianValidation",
    "HermitianValidationError",
    "OceanCarrier",
    "OceanExecutionPlan",
    "OceanInstruction",
    "OceanMode",
    "OceanProgram",
    "OceanSpectrum",
    "OceanSynthesisPlan",
    "OceanTemporalReport",
    "PreparedOcean",
    "SynthesisMethod",
    "WidePhaseAccumulator",
    "build_half_spectrum",
    "dense_irfft",
    "derivative_quarter_turn",
    "hermitian_half_spectrum",
    "plan_ocean",
    "plan_ocean_carrier",
    "plan_ocean_synthesis",
    "plan_synthesis",
    "prepare_ocean",
    "render_ocean",
    "render_ocean_frames",
    "report_temporal",
    "sparse_derivative",
    "sparse_signed_real_synthesis",
    "sparse_synthesis",
    "synthesize_derivative",
    "synthesize_ocean",
    "temporal_report",
    "validate_half_spectrum",
    "validate_hermitian_half",
]


class HermitianValidationError(ValueError):
    """Raised when a half spectrum cannot represent a real field."""


class SynthesisMethod(StrEnum):
    """Materialization choices exposed by the ocean planner."""

    AUTO = "auto"
    SPARSE = "sparse"
    IRFFT = "irfft"
    ANALYTIC = "analytic"
    FB3 = "fb3"


def _normalise_shape(shape: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    values = tuple(int(value) for value in np.asarray(shape).reshape(-1).tolist())
    if not values or any(value <= 0 for value in values):
        raise ValueError("ocean shape must contain positive dimensions")
    return values


def _normalise_index(index: Sequence[int], shape: tuple[int, ...]) -> tuple[int, ...]:
    values_list: list[int] = []
    for raw in index:
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError("Fourier indices must be integral coordinates")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("Fourier indices must be integral coordinates") from exc
        if isinstance(raw, (float, np.floating)) and (not np.isfinite(raw) or float(raw) != value):
            raise ValueError("Fourier indices must be integral coordinates")
        values_list.append(value)
    values = tuple(values_list)
    if len(values) != len(shape):
        raise ValueError("Fourier index dimension does not match the ocean shape")
    return tuple(value % size for value, size in zip(values, shape, strict=True))


def _normalise_indices(indices: Any, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    # Keep the original scalar type until ``_normalise_index`` validates it.
    # Casting here would silently turn 1.5 into 1 (and NaN/inf into an
    # implementation-dependent integer), allowing malformed Fourier support
    # to enter both the legacy and prepared paths.
    array = np.asarray(indices)
    if array.ndim == 1:
        if array.size != len(shape):
            raise ValueError("Fourier indices must have shape (K, dimension)")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(shape):
        raise ValueError("Fourier indices must have shape (K, dimension)")
    return tuple(_normalise_index(row.tolist(), shape) for row in array)


def _half_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*shape[:-1], shape[-1] // 2 + 1)


def _partner(index: tuple[int, ...], shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple((-value) % size for value, size in zip(index, shape, strict=True))


def _is_self_conjugate(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(value == 0 or (size % 2 == 0 and value == size // 2) for value, size in zip(index, shape, strict=True))


def _endpoint_last(index: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return index[-1] == 0 or (shape[-1] % 2 == 0 and index[-1] == shape[-1] // 2)


def _phase_radians(value: Any) -> float:
    if isinstance(value, WidePhaseAccumulator):
        return value.phase_radians
    if isinstance(value, v02.Phase):
        return value.radians
    if isinstance(value, (int, np.integer)):
        return (int(value) & v02.SPF32_PHASE_MASK) * (2.0 * math.pi / v02.SPF32_PHASE_MOD)
    return float(value)


def _coefficient_complex(value: Any, phase: Any | None = None) -> complex:
    """Decode one coefficient while keeping typed log inputs explicit."""

    if isinstance(value, (v02.LogPos, v02.LogSigned)):
        result = complex(value.value, 0.0)
    elif isinstance(value, v02.SPF32Value):
        result = complex(decode_words(value.word))
    elif isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        result = complex(decode_words(int(value)))
    else:
        result = complex(value)
    if phase is not None:
        result *= complex(math.cos(_phase_radians(phase)), math.sin(_phase_radians(phase)))
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError("ocean Fourier coefficients must be finite")
    return result


def _coefficient_sequence(coefficients: Any, count: int | None = None) -> tuple[Any, ...]:
    if isinstance(coefficients, np.ndarray):
        values = tuple(coefficients.reshape(-1).tolist())
    elif isinstance(coefficients, (list, tuple)):
        values = tuple(coefficients)
    else:
        values = (coefficients,)
    if count is not None and len(values) != count:
        raise ValueError("coefficient count does not match Fourier index count")
    return values


@dataclass(slots=True)
class WidePhaseAccumulator:
    """A modular master phase with a narrow u16 emission view.

    ``phase`` and ``increment`` are integer ticks on a ``2**bits`` full-turn
    circle.  For widths above 16 the high bits are emitted (with
    round-to-nearest-even); widths below 16 are promoted to the u16 view.  The
    default 64-bit state is intentionally wider than the 32-bit internal
    accumulator used by the scalar v0.2 FFT.
    """

    phase: int = 0
    increment: int = 0
    bits: int = 64
    _wraps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        bits = int(self.bits)
        if bits < 1:
            raise ValueError("wide phase width must be positive")
        for name in ("phase", "increment"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)):
                raise TypeError(f"wide phase {name} must be an integer")
            try:
                integer = int(value)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"wide phase {name} must be an integer") from exc
            if isinstance(value, (float, np.floating)) and float(value) != integer:
                raise TypeError(f"wide phase {name} must be an integer")
            object.__setattr__(self, name, integer)
        object.__setattr__(self, "bits", bits)
        object.__setattr__(self, "phase", int(self.phase) & self.mask)
        object.__setattr__(self, "increment", int(self.increment) & self.mask)

    @property
    def modulus(self) -> int:
        return 1 << self.bits

    @property
    def mask(self) -> int:
        return self.modulus - 1

    @property
    def wraps(self) -> int:
        return int(self._wraps)

    @property
    def phase_code(self) -> int:
        if self.bits == v02.SPF32_PHASE_BITS:
            return int(self.phase) & v02.SPF32_PHASE_MASK
        if self.bits > v02.SPF32_PHASE_BITS:
            divisor = 1 << (self.bits - v02.SPF32_PHASE_BITS)
            quotient, remainder = divmod(self.phase, divisor)
            return (
                quotient + (2 * remainder > divisor or (2 * remainder == divisor and quotient % 2))
            ) & v02.SPF32_PHASE_MASK
        return (int(self.phase) << (v02.SPF32_PHASE_BITS - self.bits)) & v02.SPF32_PHASE_MASK

    @property
    def theta(self) -> int:
        """Alias for the emitted canonical phase code."""

        return self.phase_code

    @property
    def phase_radians(self) -> float:
        return self.phase_code * (2.0 * math.pi / v02.SPF32_PHASE_MOD)

    def emit(self) -> int:
        return self.phase_code

    def advance(self, steps: int = 1) -> int:
        """Advance by an integer number of ticks and return the new phase."""

        if isinstance(steps, (bool, np.bool_)):
            raise TypeError("wide phase steps must be an integer")
        integer = int(steps)
        if isinstance(steps, (float, np.floating)) and float(steps) != integer:
            raise TypeError("wide phase steps must be an integer")
        before_unwrapped = int(self.phase)
        delta = int(self.increment) * integer
        unwrapped = before_unwrapped + delta
        if delta:
            self._wraps += abs(unwrapped // self.modulus - before_unwrapped // self.modulus)
        self.phase = unwrapped & self.mask
        return self.phase

    step = advance

    def copy(self) -> WidePhaseAccumulator:
        result = WidePhaseAccumulator(self.phase, self.increment, self.bits)
        result._wraps = self._wraps
        return result


@dataclass(slots=True)
class OceanMode:
    """One Fourier mode with a persistent wide phase state."""

    index: tuple[int, ...]
    coefficient: Any
    phase: int | v02.Phase | WidePhaseAccumulator = 0
    phase_increment: int = 0
    phase_bits: int = 64
    self_conjugate: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", tuple(int(value) for value in self.index))
        if isinstance(self.phase, WidePhaseAccumulator):
            accumulator = self.phase
        else:
            raw = self.phase.T if isinstance(self.phase, v02.Phase) else int(self.phase)
            accumulator = WidePhaseAccumulator(raw, int(self.phase_increment), int(self.phase_bits))
        object.__setattr__(self, "phase", accumulator)
        if self.self_conjugate is not None:
            object.__setattr__(self, "self_conjugate", bool(self.self_conjugate))

    @property
    def phase_accumulator(self) -> WidePhaseAccumulator:
        return cast(WidePhaseAccumulator, self.phase)

    @property
    def phase_code(self) -> int:
        return self.phase_accumulator.phase_code

    @property
    def theta(self) -> int:
        return self.phase_code

    def complex_coefficient(self) -> complex:
        return _coefficient_complex(self.coefficient, self.phase_code)

    def advance(self, steps: int = 1) -> int:
        return self.phase_accumulator.advance(steps)


@dataclass(frozen=True, slots=True)
class HermitianValidation:
    """Evidence returned by optional half-spectrum validation."""

    full_shape: tuple[int, ...]
    half_shape: tuple[int, ...]
    max_symmetry_error: float
    dc_real: bool
    nyquist_real: bool
    checked_pairs: int
    valid: bool = True
    errors: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.valid)


def _as_half_complex(half_spectrum: Any) -> np.ndarray:
    from spf.renderer.carriers import complex_carrier

    try:
        output = complex_carrier(half_spectrum)
    except (TypeError, ValueError) as exc:
        raise TypeError("Hermitian half spectrum must contain numeric coefficients") from exc
    if output.ndim == 0:
        raise ValueError("Hermitian half spectrum must be an array")
    if not np.all(np.isfinite(output)):
        raise HermitianValidationError("Hermitian half spectrum contains non-finite coefficients")
    return output


def _validate_half_certificate(
    half_spectrum: Any,
    shape: Sequence[int] | None = None,
    *,
    tolerance: float = 1.0e-12,
) -> HermitianValidation:
    half = _as_half_complex(half_spectrum)
    if shape is None:
        if half.ndim == 0:
            raise ValueError("Hermitian half spectrum needs a positive dimension")
        inferred_last = max(1, 2 * (half.shape[-1] - 1))
        full_shape = (*half.shape[:-1], inferred_last)
    else:
        full_shape = _normalise_shape(shape)
    expected = _half_shape(full_shape)
    errors: list[str] = []
    if half.shape != expected:
        errors.append(f"half spectrum shape {half.shape} does not match expected {expected}")
    if errors:
        raise HermitianValidationError("; ".join(errors))
    if not math.isfinite(float(tolerance)) or tolerance < 0.0:
        raise ValueError("Hermitian validation tolerance must be finite and non-negative")
    max_error = 0.0
    dc_real = abs(float(np.imag(half[(0,) * half.ndim]))) <= tolerance
    if not dc_real:
        errors.append("DC coefficient must be real")
    nyquist_real = True
    endpoint_last_values = [0]
    if full_shape[-1] % 2 == 0:
        endpoint_last_values.append(full_shape[-1] // 2)
    checked = 0
    for last in endpoint_last_values:
        prefix_shape = full_shape[:-1]
        for prefix in np.ndindex(prefix_shape):
            index = (*prefix, last)
            partner = (*tuple(-value % size for value, size in zip(prefix, full_shape[:-1], strict=True)), last)
            if index > partner:
                continue
            difference = abs(half[index] - np.conj(half[partner]))
            max_error = max(max_error, float(difference))
            checked += 1
            if float(difference) > tolerance:
                errors.append(f"Hermitian endpoint pair {index}/{partner} is not conjugate")
            if index == partner and abs(float(np.imag(half[index]))) > tolerance:
                nyquist_real = False
    valid = not errors
    return HermitianValidation(
        full_shape,
        expected,
        max_error,
        dc_real,
        nyquist_real,
        checked,
        valid,
        tuple(errors),
    )


def validate_half_spectrum(
    half_spectrum: Any,
    shape: Sequence[int] | None = None,
    *,
    tolerance: float = 1.0e-12,
    return_certificate: bool = False,
) -> bool | HermitianValidation:
    """Validate shape, DC, Nyquist, and endpoint conjugacy contracts."""

    certificate = _validate_half_certificate(half_spectrum, shape, tolerance=tolerance)
    if not certificate.valid:
        raise HermitianValidationError("; ".join(certificate.errors))
    return certificate if return_certificate else True


validate_hermitian_half = validate_half_spectrum


def _sparse_terms(
    shape: tuple[int, ...],
    indices: Any,
    coefficients: Any,
    *,
    phases: Any = None,
    tolerance: float = 1e-12,
    allow_conjugate_duplicates: bool = False,
) -> tuple[Any, ...]:
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Hermitian validation tolerance must be finite and non-negative")
    normalised = _normalise_indices(indices, shape)
    values = _coefficient_sequence(coefficients, len(normalised))
    phase_values = None if phases is None else _coefficient_sequence(phases, len(normalised))
    seen: dict[tuple[int, ...], tuple[tuple[int, ...], complex]] = {}
    terms = []
    for position, (index, value) in enumerate(zip(normalised, values, strict=True)):
        coefficient = _coefficient_complex(value, None if phase_values is None else phase_values[position])
        partner = _partner(index, shape)
        orbit = min(index, partner)
        if orbit in seen:
            old_index, old_coefficient = seen[orbit]
            if (
                not allow_conjugate_duplicates
                or index != _partner(old_index, shape)
                or abs(coefficient - np.conj(old_coefficient)) > tolerance
            ):
                raise HermitianValidationError(f"duplicate Hermitian orbit for {index}")
            continue
        if index == partner and abs(coefficient.imag) > tolerance:
            raise HermitianValidationError(f"DC/Nyquist coefficient {index} must be real")
        seen[orbit] = (index, coefficient)
        terms.append((index, coefficient, index == partner))
    return tuple(terms)


def build_half_spectrum(
    shape: Sequence[int],
    indices: Any,
    coefficients: Any,
    *,
    phases: Any | None = None,
    tolerance: float = 1.0e-12,
    allow_conjugate_duplicates: bool = False,
) -> np.ndarray:
    """Build a NumPy ``rfftn`` half spectrum from unique sparse modes.

    Indices use the ordinary modular DFT convention.  Negative indices are
    accepted and canonicalized.  Each conjugate orbit must occur once; an
    optional duplicate mode is accepted only when it is the exact conjugate
    partner and ``allow_conjugate_duplicates`` is requested.
    """

    full_shape = _normalise_shape(shape)
    terms = _sparse_terms(
        full_shape,
        indices,
        coefficients,
        phases=phases,
        tolerance=tolerance,
        allow_conjugate_duplicates=allow_conjugate_duplicates,
    )
    half = np.zeros(_half_shape(full_shape), dtype=np.complex128)
    for index, coefficient, _ in terms:
        partner = _partner(index, full_shape)
        if index[-1] <= full_shape[-1] // 2:
            destination, stored = index, coefficient
        else:
            destination, stored = partner, np.conj(coefficient)
        if _endpoint_last(destination, full_shape) and destination != _partner(destination, full_shape):
            half[destination] = stored
            endpoint_partner = _partner(destination, full_shape)
            half[endpoint_partner] = np.conj(stored)
        else:
            half[destination] = stored
    validate_half_spectrum(half, full_shape, tolerance=tolerance)
    return half


def _terms_from_half(
    half_spectrum: Any, shape: Sequence[int] | None = None
) -> tuple[tuple[tuple[int, ...], complex, bool], ...]:
    half = _as_half_complex(half_spectrum)
    full_shape = _normalise_shape(shape) if shape is not None else (*half.shape[:-1], max(1, 2 * (half.shape[-1] - 1)))
    validate_half_spectrum(half, full_shape)
    terms: list[tuple[tuple[int, ...], complex, bool]] = []
    for address in np.argwhere(half != 0):
        index = tuple(int(value) for value in address)
        full_index = index
        if _endpoint_last(full_index, full_shape):
            partner = _partner(full_index, full_shape)
            if full_index > partner:
                continue
            terms.append((full_index, complex(half[index]), full_index == partner))
        else:
            terms.append((full_index, complex(half[index]), False))
    return tuple(terms)


def _numeric_sparse_synthesis(
    shape: tuple[int, ...],
    terms: Sequence[tuple[tuple[int, ...], complex, bool]],
    *,
    normalize: bool = True,
    spacing: Sequence[float] | None = None,
) -> np.ndarray:
    if spacing is not None:
        spacing_values = tuple(float(value) for value in spacing)
        if len(spacing_values) != len(shape) or any(
            value <= 0.0 or not math.isfinite(value) for value in spacing_values
        ):
            raise ValueError("ocean sample spacing must be positive and match the shape")
    coordinates = np.indices(shape, dtype=np.float64)
    result = np.zeros(shape, dtype=np.float64)
    for index, coefficient, self_conjugate in terms:
        angle = np.zeros(shape, dtype=np.float64)
        for axis, (frequency, size) in enumerate(zip(index, shape, strict=True)):
            signed_frequency = frequency if frequency <= size // 2 else frequency - size
            angle += (2.0 * math.pi * signed_frequency / size) * coordinates[axis]
        # Real Hermitian pair synthesis: evaluate the pair contribution
        # directly as a signed-real cosine/sine term.  This keeps the
        # half-spectrum compressed and avoids constructing a complex spatial
        # plane or relying on an implicit FFT fallback.
        term = coefficient.real * np.cos(angle) - coefficient.imag * np.sin(angle)
        result += term if self_conjugate else 2.0 * term
    if normalize:
        result /= float(math.prod(shape))
    return result


def _format_sparse_output(values: np.ndarray, output_kind: Any) -> Any:
    name = str(getattr(output_kind, "value", output_kind or "native")).lower().replace("-", "_")
    if name in {"native", "dense", "real", "float", "float64", "default"}:
        return values
    if name in {"log_signed", "logsigned", "signed", "signed_real"}:
        output = np.empty(values.shape, dtype=object)
        for index in np.ndindex(values.shape):
            output[index] = v02.LogSigned.from_value(float(values[index]))
        return output
    if name in {"spf", "spf32", "words", "word", "uint32"}:
        return np.asarray(encode_words(values), dtype=np.uint32)
    raise ValueError(f"unsupported sparse ocean output kind {output_kind!r}")


def sparse_signed_real_synthesis(
    coefficients: Any,
    indices: Any,
    shape: Sequence[int],
    *,
    phases: Any | None = None,
    normalize: bool = True,
    output_kind: Any = "native",
    spacing: Sequence[float] | None = None,
    tolerance: float = 1.0e-12,
    return_report: bool = False,
) -> Any:
    """Synthesize a real field directly from unique Hermitian modes."""

    return prepare_ocean(
        shape,
        coefficients,
        indices,
        phases=phases,
        method="sparse",
        normalize=normalize,
        output_kind=output_kind,
        tolerance=tolerance,
        spacing=spacing,
    ).execute(return_report=return_report)


sparse_synthesis = sparse_signed_real_synthesis


def dense_irfft(
    half_spectrum: Any,
    shape: Sequence[int] | None = None,
    *,
    axes: Sequence[int] | None = None,
    tolerance: float = 1.0e-12,
    output_kind: Any = "native",
    return_report: bool = False,
) -> Any:
    """Validate and materialize a Hermitian half spectrum through NumPy."""

    certificate = _validate_half_certificate(half_spectrum, shape, tolerance=tolerance)
    if not certificate.valid:
        raise HermitianValidationError("; ".join(certificate.errors))
    half = _as_half_complex(half_spectrum)
    transform_axes = tuple(range(len(certificate.full_shape))) if axes is None else tuple(axes)
    from scipy import fft  # type: ignore[import-untyped]

    result = fft.irfftn(half, s=certificate.full_shape, axes=transform_axes, workers=1)
    output = _format_sparse_output(np.asarray(result, dtype=np.float64), output_kind)
    report = _ocean_render_report(
        backend="reference",
        shape=certificate.full_shape,
        method=SynthesisMethod.IRFFT.value,
        modes=0,
        fft_count=1,
        fallback=None,
        diagnostics={
            "half_shape": certificate.half_shape,
            "max_symmetry_error": certificate.max_symmetry_error,
            "dc_real": certificate.dc_real,
            "nyquist_real": certificate.nyquist_real,
        },
    )
    return (output, report) if return_report else output


# The richer carrier plan is the public prepared-plan type.  Keep this name
# as an alias so callers that imported the v0.1 ``OceanSynthesisPlan`` still
# recognize plans returned by both the legacy planner and ``prepare_ocean``.
OceanSynthesisPlan = OceanExecutionPlan


def plan_synthesis(
    shape: Sequence[int],
    *,
    active_modes: int | None = None,
    indices: Any | None = None,
    half_spectrum: Any | None = None,
    method: str | SynthesisMethod = SynthesisMethod.AUTO,
    fft_constant: float = 1.0,
    allow_dense: bool = True,
) -> OceanSynthesisPlan:
    """Choose sparse synthesis or a dense ``irfft`` using transparent costs."""

    full_shape = _normalise_shape(shape)
    n = int(math.prod(full_shape))
    modes = int(active_modes if active_modes is not None else (len(indices) if indices is not None else 0))
    if modes < 0:
        raise ValueError("active mode count must be non-negative")
    constant = float(fft_constant)
    if not math.isfinite(constant) or constant <= 0.0:
        raise ValueError("fft_constant must be finite and positive")
    sparse_cost = float(n * modes)
    irfft_cost = float(constant * n * max(1.0, math.log2(max(2, n))))
    requested = str(getattr(method, "value", method)).lower().replace("-", "_")
    if requested in {"dense", "fft", "irfft", "dense_irfft"}:
        requested = SynthesisMethod.IRFFT.value
    elif requested in {"packed", "fb3", "fb3_hybrid", "v02"}:
        requested = SynthesisMethod.FB3.value
    elif requested not in {item.value for item in SynthesisMethod}:
        raise ValueError(f"unsupported ocean synthesis method {method!r}")
    if requested == SynthesisMethod.SPARSE.value:
        return OceanSynthesisPlan(full_shape, requested, modes, n, sparse_cost, irfft_cost, "explicit sparse request")
    if requested == SynthesisMethod.IRFFT.value:
        fallback = None if allow_dense else SynthesisMethod.SPARSE.value
        selected = requested if allow_dense else SynthesisMethod.SPARSE.value
        return OceanSynthesisPlan(
            full_shape,
            selected,
            modes,
            n,
            sparse_cost,
            irfft_cost,
            "explicit dense request" if allow_dense else "dense path disabled; sparse fallback",
            fallback,
        )
    if requested == SynthesisMethod.FB3.value:
        return OceanSynthesisPlan(
            full_shape,
            requested,
            modes,
            n,
            sparse_cost,
            irfft_cost,
            "explicit packed FB3 request",
        )
    if requested == SynthesisMethod.ANALYTIC.value:
        return OceanSynthesisPlan(
            full_shape,
            requested,
            modes,
            n,
            sparse_cost,
            irfft_cost,
            "explicit analytic request",
        )
    if allow_dense and (half_spectrum is not None or indices is not None) and irfft_cost < sparse_cost:
        return OceanSynthesisPlan(
            full_shape,
            SynthesisMethod.IRFFT.value,
            modes,
            n,
            sparse_cost,
            irfft_cost,
            "dense irfft has lower estimated cost",
        )
    return OceanSynthesisPlan(
        full_shape,
        SynthesisMethod.SPARSE.value,
        modes,
        n,
        sparse_cost,
        irfft_cost,
        "sparse signed-real path has lower estimated cost",
    )


plan_ocean_synthesis = plan_synthesis


def _ocean_render_report(
    *,
    backend: str,
    shape: tuple[int, ...],
    method: str,
    modes: int,
    fft_count: int,
    fallback: str | None,
    diagnostics: Mapping[str, Any] | None = None,
    temporal_error: float | None = None,
    materialized_elements: int | None = None,
    potential_elements: int | None = None,
    complex_adds: int | None = None,
    field_operations: int | None = None,
    reduction_operations: int | None = None,
    bytes_read: int | None = None,
    bytes_written: int | None = None,
    materialized_intermediate_elements: int | None = None,
    output_dtype: str | None = None,
    profile: str | None = None,
    consumer: str | None = None,
    ledger: v02.ErrorLedger | None = None,
) -> RenderReport:
    detail = dict(diagnostics or {})
    route = str(method)
    representation = {
        SynthesisMethod.IRFFT.value: "hermitian_half_cartesian_reference",
        "fb3": "packed_spf32_fb3_reference",
        "analytic": "analytic_atom_joint_reference",
        SynthesisMethod.SPARSE.value: "sparse_signed_real_reference",
    }.get(route, route)
    if profile is not None:
        representation = str(profile)
    n = int(math.prod(shape))
    output_elements = n if materialized_elements is None else int(materialized_elements)
    potential = n if potential_elements is None else int(potential_elements)
    intermediate = 0 if materialized_intermediate_elements is None else int(materialized_intermediate_elements)
    # A sparse program has one logical term evaluation for every output point;
    # the actual intermediate allocation is the scratch peak, recorded by the
    # terminal.  Dense paths charge their owned half carrier separately.
    potential_intermediate = int(n * max(0, modes)) if route == SynthesisMethod.SPARSE.value else int(n)
    detail.setdefault("output_dtype", output_dtype or "float64")
    detail.setdefault("profile", profile or route)
    detail.setdefault("consumer", consumer or route)
    detail.setdefault("hardware_native", False)
    detail.setdefault("numerical_error_status", "not_certified" if ledger is None else "explicit_ledger")
    if ledger is None:
        # A missing transform/rounding certificate is not a zero error bound.
        # Native point consumers publish their separate per-observable ledger;
        # reference and FB3 reports must retain their uncertainty explicitly.
        ledger = v02.ErrorLedger(materialization=math.inf)
    metrics = RenderMetrics(
        backend=backend,
        workload="ocean",
        resolution=shape,
        representation=representation,
        precision=output_dtype or "float64",
        potential_elements=potential,
        materialized_elements=output_elements,
        atoms_considered=modes,
        complex_adds=max(0, modes - 1) if complex_adds is None else int(complex_adds),
        fft_count=fft_count,
        field_operations=1 if field_operations is None else int(field_operations),
        reduction_operations=(1 if modes else 0) if reduction_operations is None else int(reduction_operations),
        temporal_error=temporal_error,
        bytes_read=0 if bytes_read is None else int(bytes_read),
        bytes_written=0 if bytes_written is None else int(bytes_written),
        potential_intermediate_elements=potential_intermediate,
        materialized_intermediate_elements=intermediate,
        terminal_output_elements=output_elements,
        terminal_output_bytes=output_elements * (4 if output_dtype == "float32" else 8),
        materialized_intermediate_bytes=intermediate * (4 if output_dtype == "float32" else 8),
        logical_peak_bytes=intermediate * (4 if output_dtype == "float32" else 8),
        fallback_conversions=int(detail.get("actual_counters", {}).get("fallback_conversions", 0)),
    )
    passes = (
        CompilerPassRecord("hermitian_validate", 1, 1, notes=("DC/Nyquist and endpoint conjugacy checked",)),
        CompilerPassRecord("ocean_materialize", 1, 1, notes=(method,)),
    )
    fallbacks = () if fallback is None else (fallback,)
    capabilities = ["wide_phase", "hermitian", route, "joint_observable_terminal"]
    if bool(detail.get("hardware_native", False)):
        capabilities.append("spf_native_fused" if str(profile or "").startswith("spf32") else "fp32_native_fused")
    return RenderReport(
        metrics,
        ledger=ledger,
        passes=passes,
        fallbacks=fallbacks,
        capabilities_used=tuple(capabilities),
        diagnostics=MappingProxyType(detail),
    )


def synthesize_ocean(
    shape: Sequence[int] | OceanSpectrum,
    coefficients: Any | None = None,
    indices: Any | None = None,
    *,
    half_spectrum: Any | None = None,
    phases: Any | None = None,
    method: str | SynthesisMethod = SynthesisMethod.AUTO,
    normalize: bool | None = None,
    output_kind: Any = "native",
    tolerance: float = 1.0e-12,
    spacing: Sequence[float] | None = None,
    backend: str | SFRBackend = "reference",
    return_report: bool = False,
    allow_dense: bool = True,
) -> Any:
    """Render an ocean spectrum with a sparse/irfft planner and fallback."""

    prepared = prepare_ocean(
        shape,
        coefficients,
        indices,
        half_spectrum=half_spectrum,
        phases=phases,
        method=method,
        normalize=normalize,
        output_kind=output_kind,
        tolerance=tolerance,
        spacing=spacing,
        backend=backend,
        allow_dense=allow_dense,
    )
    return prepared.execute(return_report=return_report)


render_ocean = synthesize_ocean


def derivative_quarter_turn(value: Any) -> Any:
    """Apply the v0.2 quarter-turn phase used by spectral derivatives."""

    if isinstance(value, v02.Phase):
        return v02.Phase(value.T + v02.SPF32_PHASE_QUARTER)
    if isinstance(value, WidePhaseAccumulator):
        result = value.copy()
        result.phase = (result.phase + (1 << max(0, result.bits - 2))) & result.mask
        return result
    if isinstance(value, v02.SPF32Value):
        return v02.spf_phase_shift(value.word, v02.SPF32_PHASE_QUARTER)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return v02.spf_phase_shift(int(value), v02.SPF32_PHASE_QUARTER)
    if isinstance(value, v02.LogPos):
        return v02.LogSigned(value.L, 1)
    if isinstance(value, v02.LogSigned):
        return v02.LogSigned(value.L, -value.sign if not value.is_zero and not value.is_nan else value.sign)
    return complex(value) * 1j


def synthesize_derivative(
    coefficients: Any,
    indices: Any,
    shape: Sequence[int],
    *,
    axis: int = 0,
    phases: Any | None = None,
    spacing: Sequence[float] | float | None = None,
    method: str | SynthesisMethod = SynthesisMethod.SPARSE,
    output_kind: Any = "native",
    return_report: bool = False,
) -> Any:
    """Synthesize one spatial derivative using a spectral quarter-turn."""

    prepared = prepare_ocean(shape, coefficients, indices, phases=phases, method=method, output_kind=output_kind)
    result, report = prepared.execute(derivative_axis=axis, spacing=spacing, return_report=True)
    selected_axis = int(axis) % len(prepared.shape)
    diagnostics = dict(report.diagnostics)
    diagnostics.update({"derivative_axis": selected_axis, "quarter_turn": v02.SPF32_PHASE_QUARTER})
    report = RenderReport(
        report.metrics,
        ledger=report.ledger,
        passes=report.passes,
        decisions=report.decisions,
        fallbacks=report.fallbacks,
        capabilities_used=(*report.capabilities_used, "derivative_quarter_turn"),
        diagnostics=diagnostics,
    )
    return (result, report) if return_report else result


sparse_derivative = synthesize_derivative


@dataclass(frozen=True, slots=True)
class OceanTemporalReport:
    """Temporal quality evidence for an ocean frame sequence."""

    frames: int
    dt: float
    temporal_error: float
    max_frame_error: float
    phase_error_bound: float
    phase_wraps: int
    method: str
    report: RenderReport | None = None

    @property
    def metrics(self) -> RenderMetrics | None:
        return None if self.report is None else self.report.metrics

    @property
    def error_bound(self) -> float:
        return self.phase_error_bound

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "dt": self.dt,
            "temporal_error": self.temporal_error,
            "max_frame_error": self.max_frame_error,
            "phase_error_bound": self.phase_error_bound,
            "phase_wraps": self.phase_wraps,
            "method": self.method,
            "render_report": None if self.report is None else self.report.to_dict(),
        }


def temporal_report(
    actual: Sequence[Any],
    reference: Sequence[Any] | None = None,
    *,
    dt: float = 1.0,
    phase_wraps: int = 0,
    method: str = SynthesisMethod.SPARSE.value,
    report: RenderReport | None = None,
) -> OceanTemporalReport:
    """Report temporal-difference error without collapsing frame evidence."""

    actual_values = tuple(actual)
    if not actual_values:
        raise ValueError("temporal report requires at least one frame")
    delta_t = float(dt)
    if not math.isfinite(delta_t) or delta_t <= 0.0:
        raise ValueError("temporal dt must be finite and positive")
    if reference is None:
        error = 0.0
        frame_errors = [0.0]
    else:
        reference_values = tuple(reference)
        error = temporal_difference_error(actual_values, reference_values)
        frame_errors = [relative_l2(left, right) for left, right in zip(actual_values, reference_values, strict=True)]
    result_report = report
    if result_report is not None:
        metrics = result_report.metrics
        metrics.temporal_error = float(error)
    return OceanTemporalReport(
        frames=len(actual_values),
        dt=delta_t,
        temporal_error=float(error),
        max_frame_error=float(max(frame_errors)),
        phase_error_bound=math.pi / v02.SPF32_PHASE_MOD,
        phase_wraps=int(phase_wraps),
        method=str(method),
        report=result_report,
    )


report_temporal = temporal_report


@dataclass(slots=True)
class OceanSpectrum:
    """Persistent sparse or half-spectrum ocean state."""

    shape: tuple[int, ...]
    coefficients: tuple[Any, ...] = ()
    indices: tuple[tuple[int, ...], ...] = ()
    phases: tuple[Any, ...] = ()
    half_spectrum: np.ndarray | None = None
    modes: tuple[OceanMode, ...] = ()
    spacing: tuple[float, ...] | None = None
    normalize: bool = True
    _wraps: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        full_shape = _normalise_shape(self.shape)
        object.__setattr__(self, "shape", full_shape)
        if self.spacing is not None:
            spacing = tuple(float(value) for value in self.spacing)
            if len(spacing) != len(full_shape) or any(value <= 0.0 or not math.isfinite(value) for value in spacing):
                raise ValueError("ocean spacing must be positive and match the shape")
            object.__setattr__(self, "spacing", spacing)
        mode_values = tuple(self.modes)
        for mode in mode_values:
            if not isinstance(mode, OceanMode):
                raise TypeError("OceanSpectrum modes must be OceanMode values")
            _normalise_index(mode.index, full_shape)
        object.__setattr__(self, "modes", mode_values)
        if mode_values:
            object.__setattr__(self, "indices", tuple(_normalise_index(mode.index, full_shape) for mode in mode_values))
            object.__setattr__(self, "coefficients", tuple(mode.coefficient for mode in mode_values))
            object.__setattr__(self, "phases", ())
            object.__setattr__(self, "half_spectrum", None)
        else:
            normalised_indices = () if not self.indices else _normalise_indices(self.indices, full_shape)
            coefficients = (
                () if not self.coefficients else _coefficient_sequence(self.coefficients, len(normalised_indices))
            )
            phases = () if not self.phases else _coefficient_sequence(self.phases, len(normalised_indices))
            if self.half_spectrum is not None:
                half = _as_half_complex(self.half_spectrum)
                validate_half_spectrum(half, full_shape)
                object.__setattr__(self, "half_spectrum", half)
            object.__setattr__(self, "indices", normalised_indices)
            object.__setattr__(self, "coefficients", coefficients)
            object.__setattr__(self, "phases", phases)

    @classmethod
    def from_sparse(
        cls,
        shape: Sequence[int],
        indices: Any,
        coefficients: Any,
        *,
        phases: Any | None = None,
        spacing: Sequence[float] | None = None,
        normalize: bool = True,
    ) -> OceanSpectrum:
        full_shape = _normalise_shape(shape)
        normalised_indices = _normalise_indices(indices, full_shape)
        return cls(
            full_shape,
            _coefficient_sequence(coefficients, len(normalised_indices)),
            normalised_indices,
            () if phases is None else _coefficient_sequence(phases, len(normalised_indices)),
            spacing=None if spacing is None else tuple(spacing),
            normalize=normalize,
        )

    @classmethod
    def from_half_spectrum(
        cls,
        half_spectrum: Any,
        shape: Sequence[int] | None = None,
        *,
        tolerance: float = 1.0e-12,
    ) -> OceanSpectrum:
        certificate = _validate_half_certificate(half_spectrum, shape, tolerance=tolerance)
        return cls(certificate.full_shape, half_spectrum=_as_half_complex(half_spectrum))

    @property
    def phase_accumulators(self) -> tuple[WidePhaseAccumulator, ...]:
        return tuple(mode.phase_accumulator for mode in self.modes)

    @property
    def wrap_count(self) -> int:
        return int(self._wraps + sum(mode.phase_accumulator.wraps for mode in self.modes))

    def validate(self, *, tolerance: float = 1.0e-12) -> HermitianValidation | bool:
        if self.half_spectrum is not None:
            return validate_half_spectrum(self.half_spectrum, self.shape, tolerance=tolerance, return_certificate=True)
        if self.indices:
            build_half_spectrum(
                self.shape, self.indices, self.coefficients, phases=self.phases or None, tolerance=tolerance
            )
        return True

    def current_sparse(self) -> tuple[tuple[tuple[int, ...], ...], tuple[complex, ...]]:
        if self.modes:
            indices = tuple(_normalise_index(mode.index, self.shape) for mode in self.modes)
            return indices, tuple(mode.complex_coefficient() for mode in self.modes)
        if self.half_spectrum is not None:
            terms = _terms_from_half(self.half_spectrum, self.shape)
            return tuple(item[0] for item in terms), tuple(item[1] for item in terms)
        values = _coefficient_sequence(self.coefficients, len(self.indices))
        phase_values = self.phases or None
        return self.indices, tuple(
            _coefficient_complex(value, None if phase_values is None else phase_values[position])
            for position, value in enumerate(values)
        )

    def half(self, *, tolerance: float = 1.0e-12) -> np.ndarray:
        if self.modes:
            indices, coefficients = self.current_sparse()
            return build_half_spectrum(self.shape, indices, coefficients, tolerance=tolerance)
        if self.half_spectrum is not None:
            validate_half_spectrum(self.half_spectrum, self.shape, tolerance=tolerance)
            return np.array(self.half_spectrum, copy=True)
        return build_half_spectrum(
            self.shape, self.indices, self.coefficients, phases=self.phases or None, tolerance=tolerance
        )

    def prepare(self, **options: Any) -> Any:
        return prepare_ocean(self, **options)

    def render(
        self, *, method: str | SynthesisMethod = SynthesisMethod.AUTO, return_report: bool = False, **options: Any
    ) -> Any:
        return synthesize_ocean(self, method=method, return_report=return_report, **options)

    def derivative(
        self,
        axis: int = 0,
        *,
        method: str | SynthesisMethod = SynthesisMethod.SPARSE,
        return_report: bool = False,
        **options: Any,
    ) -> Any:
        spacing = options.pop("spacing", self.spacing)
        return self.prepare(method=method, **options).execute(
            derivative_axis=axis, spacing=spacing, return_report=return_report
        )

    def advance(self, steps: int = 1) -> tuple[int, ...]:
        if not self.modes:
            return ()
        return tuple(mode.advance(steps) for mode in self.modes)

    step = advance

    def frames(
        self,
        count: int,
        *,
        steps: int = 1,
        method: str | SynthesisMethod = SynthesisMethod.AUTO,
        return_report: bool = False,
        reference: Sequence[Any] | None = None,
        **options: Any,
    ) -> Any:
        frame_count = int(count)
        if frame_count <= 0:
            raise ValueError("ocean frame count must be positive")
        values: list[np.ndarray] = []
        reports: list[RenderReport] = []
        prepared = self.prepare(method=method, **options)
        try:
            for frame_index in range(frame_count):
                if frame_index:
                    prepared.advance(steps)
                value, report = prepared.execute(return_report=True)
                values.append(np.asarray(value))
                reports.append(report)
        finally:
            # Preserve the public mutable timeline while vectorizing intermediate frames.
            prepared.sync_phases(self.modes)
        stack = np.stack(values, axis=0)
        base_report = reports[-1]
        temporal = temporal_report(
            values,
            reference,
            dt=float(steps),
            phase_wraps=self.wrap_count,
            method=str(getattr(method, "value", method)),
            report=base_report,
        )
        if return_report:
            return stack, temporal
        return stack


def hermitian_half_spectrum(*args: Any, **kwargs: Any) -> HermitianHalfSpectrum:
    """Construct the immutable convenience wrapper for a half spectrum."""

    return HermitianHalfSpectrum(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class HermitianHalfSpectrum:
    """Validated immutable wrapper around an ``rfftn`` half spectrum."""

    values: np.ndarray
    shape: tuple[int, ...] | None = None
    tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        values = _as_half_complex(self.values)
        certificate = _validate_half_certificate(values, self.shape, tolerance=self.tolerance)
        if not certificate.valid:
            raise HermitianValidationError("; ".join(certificate.errors))
        values = np.array(values, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "shape", certificate.full_shape)

    @property
    def full_shape(self) -> tuple[int, ...]:
        return cast(tuple[int, ...], self.shape)

    @property
    def half_shape(self) -> tuple[int, ...]:
        return self.values.shape

    def validate(self) -> HermitianValidation:
        return _validate_half_certificate(self.values, self.full_shape, tolerance=self.tolerance)

    def prepare(self, **options: Any) -> Any:
        return prepare_ocean(self.full_shape, half_spectrum=self.values, **options)

    def irfft(self, **kwargs: Any) -> np.ndarray:
        options = dict(kwargs)
        shape = self.full_shape
        options.setdefault("axes", tuple(range(len(shape))))
        from scipy import fft

        return np.asarray(fft.irfftn(self.values, s=shape, **options), dtype=np.float64)

    def sparse(self, **kwargs: Any) -> np.ndarray:
        shape = self.full_shape
        return np.asarray(prepare_ocean(shape, half_spectrum=self.values, method="sparse", **kwargs).execute())


def render_ocean_frames(
    spectrum: OceanSpectrum,
    count: int,
    *,
    steps: int = 1,
    method: str | SynthesisMethod = SynthesisMethod.AUTO,
    return_report: bool = False,
    reference: Sequence[Any] | None = None,
    **options: Any,
) -> Any:
    return spectrum.frames(
        count,
        steps=steps,
        method=method,
        return_report=return_report,
        reference=reference,
        **options,
    )
