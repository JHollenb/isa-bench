"""Source-aware analytic consumers for the SPF ocean renderer.

The terminal in :mod:`spf.renderer.ocean_terminal` is intentionally a dense
reference path.  This module is the compact-source path: Gaussian atoms,
quadratic jets, and sparse addressed Fourier carriers are evaluated directly
at the requested points.  Height and both spatial derivatives are accumulated
from the same value evaluation, so a renderer never has to make three
independent source traversals or reconstruct a dense FFT plane.

The implementation is deliberately conservative at representation boundaries.
An atom mixture residual is carried in the error ledger, a broad addressed
carrier is refused instead of being silently densified, and deep-water (or
other non-quadratic) dispersion does not claim Gaussian closure without an
explicit residual policy.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.tensor import v02
from spf.tensor.v02 import (
    AtomMixture,
    ErrorLedger,
    GaussianChirpAtom,
    Jet2D,
    PerformanceCounters,
    atom_inverse_fourier,
    atom_mul,
)

from .blocks import BlockView
from .carriers import as_block_view
from .model import CoordinateChart, Domain, Field, Region, SampleLayout, Support
from .report import RenderMetrics, RenderReport

_TAU = 2.0 * math.pi
_DEFAULT_CHUNK = 65_536
_DEFAULT_HERMITIAN_TOLERANCE = 5.0e-10


class AnalyticOceanError(ValueError):
    """Raised when a source cannot be consumed without an invalid materialization."""


@dataclass(frozen=True, slots=True)
class OceanAnalyticFrame:
    """One analytic ocean sample result.

    ``gradient_x`` and ``gradient_y`` are physical coordinate derivatives.
    Arrays use the requested point/layout shape.  ``residual`` is optional
    pointwise residual information supplied by a source evaluator; a scalar
    representation bound is available on ``report.ledger``.
    """

    height: np.ndarray
    gradient_x: np.ndarray
    gradient_y: np.ndarray
    report: RenderReport | None = None
    mask: np.ndarray | None = None
    residual: np.ndarray | None = None

    def __post_init__(self) -> None:
        height = np.asarray(self.height)
        gradient_x = np.asarray(self.gradient_x)
        gradient_y = np.asarray(self.gradient_y)
        if height.shape != gradient_x.shape or height.shape != gradient_y.shape:
            raise ValueError("height and gradient arrays must have identical shapes")
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "gradient_x", gradient_x)
        object.__setattr__(self, "gradient_y", gradient_y)
        if self.mask is not None:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != height.shape:
                raise ValueError("ocean result mask shape must match height")
            object.__setattr__(self, "mask", mask)
        if self.residual is not None:
            residual = np.asarray(self.residual)
            if residual.shape != height.shape:
                raise ValueError("ocean result residual shape must match height")
            object.__setattr__(self, "residual", residual)

    @property
    def values(self) -> np.ndarray:
        """Compatibility alias for the height field."""

        return self.height

    @property
    def dx(self) -> np.ndarray:
        return self.gradient_x

    @property
    def dy(self) -> np.ndarray:
        return self.gradient_y

    @property
    def grad_x(self) -> np.ndarray:
        return self.gradient_x

    @property
    def grad_y(self) -> np.ndarray:
        return self.gradient_y

    @property
    def slopes(self) -> tuple[np.ndarray, np.ndarray]:
        return self.gradient_x, self.gradient_y

    @property
    def output_dtype(self) -> str:
        return str(self.height.dtype)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.height, dtype=dtype)

    def __iter__(self) -> Iterator[np.ndarray]:
        return iter((self.height, self.gradient_x, self.gradient_y))


@dataclass(frozen=True, slots=True)
class OceanAnalyticPlan:
    """Immutable description of a compact analytic ocean consumer."""

    representation: str
    source_count: int
    source_shape: tuple[int, int] | None
    compact_source: bool
    normalization: float
    dtype: str
    temporal_closed: bool
    temporal_residual_bound: float
    error: ErrorLedger = field(default_factory=ErrorLedger)
    source_domain: str = "SPATIAL_2D"
    capabilities: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if int(self.source_count) < 0:
            raise ValueError("source_count must be non-negative")
        object.__setattr__(self, "source_count", int(self.source_count))
        if self.source_shape is None:
            normalized_shape = None
        else:
            raw_shape = tuple(int(x) for x in self.source_shape)
            if len(raw_shape) != 2:
                raise ValueError("analytic ocean source_shape must be two-dimensional")
            normalized_shape = (raw_shape[0], raw_shape[1])
        object.__setattr__(self, "source_shape", normalized_shape)
        object.__setattr__(self, "normalization", float(self.normalization))
        object.__setattr__(self, "temporal_residual_bound", float(self.temporal_residual_bound))
        object.__setattr__(self, "capabilities", tuple(str(value) for value in self.capabilities))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def kind(self) -> str:
        return self.representation

    @property
    def error_bound(self) -> float:
        return self.error.total_bound

    @property
    def no_dense_materialization(self) -> bool:
        return bool(self.compact_source)


@dataclass(frozen=True, slots=True)
class _DispersionSpec:
    kind: str
    center: np.ndarray
    value: float
    gradient: np.ndarray
    hessian: np.ndarray
    sign: float
    evaluator: Callable[[np.ndarray], np.ndarray] | None
    residual_bound_per_time: float
    residual_certified: bool = False


@dataclass(slots=True)
class _SparseMode:
    index: tuple[int, int]
    base: complex
    phase_ticks: int = 0
    phase_increment: int = 0
    phase_bits: int = 16
    pair_weight: float = 2.0
    external: Any = None
    time_phase: float = 0.0
    angular_frequency: float | None = None
    sign: float = -1.0

    def phase_radians(self) -> float:
        if self.external is not None:
            accumulator = getattr(self.external, "phase_accumulator", None)
            if accumulator is not None:
                emitted = getattr(accumulator, "phase_radians", None)
                if emitted is not None and not callable(emitted):
                    # WidePhaseAccumulator.phase_radians is deliberately the
                    # u16 emission angle, regardless of its wider state.
                    return float(emitted) + self.time_phase
                raw_phase = getattr(accumulator, "phase", None)
                if raw_phase is not None:
                    bits = int(getattr(accumulator, "bits", self.phase_bits))
                    return (int(raw_phase) & ((1 << bits) - 1)) * _TAU / float(1 << bits) + self.time_phase
                code = getattr(accumulator, "phase_code", 0)
                return int(code) * _TAU / float(1 << 16) + self.time_phase
            phase = getattr(self.external, "phase", None)
            if phase is not None:
                if hasattr(phase, "radians"):
                    return float(phase.radians) + self.time_phase
                return int(phase) * _TAU / float(1 << self.phase_bits) + self.time_phase
        return (
            (self.phase_ticks & ((1 << self.phase_bits) - 1)) * _TAU / float(1 << self.phase_bits)
        ) + self.time_phase

    def current(self) -> complex:
        phase = self.phase_radians()
        return self.base * complex(math.cos(phase), math.sin(phase))

    def advance(self, steps: int = 1, dt: float | None = None) -> None:
        count = int(steps)
        if count < 0:
            raise ValueError("steps must be non-negative")
        if self.external is not None and hasattr(self.external, "advance"):
            for _ in range(count):
                self.external.advance()
        else:
            modulus = 1 << self.phase_bits
            self.phase_ticks = (self.phase_ticks + count * self.phase_increment) % modulus
        if dt is not None and self.angular_frequency is not None:
            self.time_phase += self.sign * float(self.angular_frequency) * float(dt) * count


@dataclass(slots=True)
class _AtomSource:
    atoms: tuple[GaussianChirpAtom, ...]
    residual: Any = None
    source_domain: str = "SPATIAL_2D"
    base_fourier_atoms: tuple[GaussianChirpAtom, ...] = ()
    temporal: _DispersionSpec | None = None


@dataclass(frozen=True, slots=True)
class _JetResidualSource:
    """A Jet2D plus a residual carried by its owning Field representation."""

    jet: Jet2D
    residual: Any


@dataclass(frozen=True, slots=True)
class _SpectralJetResidualSource:
    """A spectral jet tile plus an additional Field representation residual."""

    tile: Any
    residual: Any


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _as_complex(value: Any) -> complex:
    """Decode one ordinary scalar or packed SPF scalar without expanding arrays."""

    if isinstance(value, v02.SPF32Value):
        return complex(v02.decode_complex(value.word))
    if isinstance(value, (v02.LogPos, v02.LogSigned)):
        return complex(value.value) * (1 if not isinstance(value, v02.LogSigned) else value.sign)
    if isinstance(value, v02.Phase):
        return complex(math.cos(value.radians), math.sin(value.radians))
    if isinstance(value, (int, np.integer)):
        return complex(v02.decode_complex(int(value)))
    return complex(value)


def _array_complex(values: Any) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in "iu":
        return np.asarray(v02.decode_complex(array.astype(np.uint32, copy=False)), dtype=np.complex128)
    if array.dtype.kind == "O":
        return np.asarray([_as_complex(value) for value in array.reshape(-1)], dtype=np.complex128).reshape(array.shape)
    return np.asarray(array, dtype=np.complex128)


def _phase_radians(value: Any, phase_bits: int = 16) -> float:
    if value is None:
        return 0.0
    if isinstance(value, v02.Phase):
        return float(value.radians)
    if hasattr(value, "radians"):
        return float(value.radians)
    if isinstance(value, (int, np.integer)):
        return int(value) * _TAU / float(1 << phase_bits)
    return float(value)


def _phase_ticks(value: Any, phase_bits: int) -> int:
    if value is None:
        return 0
    if isinstance(value, v02.Phase):
        radians = value.radians
        return round(radians * (1 << phase_bits) / _TAU)
    if isinstance(value, (int, np.integer)):
        # Existing SPF ocean modes use u16 phase words.  A wider accumulator
        # scales that word to its requested modulus without changing its angle.
        return (
            int(value) << max(0, phase_bits - 16)
            if phase_bits >= 16
            else round(int(value) / float(1 << (16 - phase_bits)))
        )
    return round(float(value) * (1 << phase_bits) / _TAU)


def _read_field_source(
    source: Any,
) -> tuple[Any, str, CoordinateChart | None, Support | None, ErrorLedger, Mapping[str, Any]]:
    if isinstance(source, Field):
        from spf.renderer.normalization import complete_field_ledger, require_closed_field

        source = require_closed_field(source, consumer="analytic ocean")
        representation = source.representation
        payload = representation.payload
        if representation.residual is not None:
            if isinstance(payload, Jet2D):
                # Jet2D is immutable and intentionally has no residual slot.
                # Keep the representation residual in an explicit wrapper so
                # the point consumer evaluates it instead of silently losing
                # it at Field ingress.
                payload = _JetResidualSource(payload, representation.residual)
            elif _is_spectral_jet_tile(payload):
                # The tile already owns log/phase residual streams; retain a
                # further Field-level residual as a separate explicit source.
                payload = _SpectralJetResidualSource(payload, representation.residual)
            else:
                raise AnalyticOceanError("analytic ocean source requires an explicit representation residual consumer")
        domain = str(source.domain)
        chart = source.chart
        support = source.support
        metadata = dict(source.metadata)
        metadata.update(representation.metadata)
        return payload, domain, chart, support, complete_field_ledger(source), metadata
    if isinstance(source, AtomMixture):
        from spf.renderer.normalization import complete_field_ledger

        return source, "", None, None, complete_field_ledger(Field.from_value(source)), {}
    return source, "", None, None, ErrorLedger(), {}


def _source_atoms(source: Any) -> tuple[tuple[GaussianChirpAtom, ...], Any, str]:
    if isinstance(source, GaussianChirpAtom):
        return (source,), None, "SPATIAL_2D"
    if isinstance(source, AtomMixture):
        return tuple(source.atoms), source.residual, "SPATIAL_2D"
    if isinstance(source, Jet2D):
        return (), None, "JET_2D"
    atoms = getattr(source, "atoms", None)
    if atoms is not None and all(isinstance(atom, GaussianChirpAtom) for atom in atoms):
        return tuple(atoms), getattr(source, "residual", None), "SPATIAL_2D"
    return (), None, ""


def _is_spectral_jet_tile(source: Any) -> bool:
    """Check the paired guarded spectral-jet type without an import cycle."""

    try:
        from .jets import SpectralJetTile2D
    except ImportError:
        return False
    return isinstance(source, SpectralJetTile2D)


def _spectral_jet_source(source: Any) -> tuple[Any, Any]:
    if isinstance(source, _SpectralJetResidualSource):
        return source.tile, source.residual
    return source, None


def _chart_matrix(chart: CoordinateChart | None) -> tuple[np.ndarray, np.ndarray]:
    if chart is None:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    if chart.dimension != 2:
        raise AnalyticOceanError("analytic ocean consumers require a two-dimensional coordinate chart")
    basis = np.asarray(chart.basis, dtype=np.float64)
    scale = np.asarray(chart.scale, dtype=np.float64)
    spacing = np.asarray(chart.sample_spacing, dtype=np.float64)
    return basis @ np.diag(scale * spacing), np.asarray(chart.origin, dtype=np.float64)


def _chart_coordinates(points_xy: np.ndarray, chart: CoordinateChart | None) -> np.ndarray:
    matrix, origin = _chart_matrix(chart)
    return np.asarray(points_xy @ matrix.T + origin, dtype=np.float64)


def _chart_jacobian(chart: CoordinateChart | None) -> np.ndarray:
    matrix, _ = _chart_matrix(chart)
    return matrix


def _field_points(points_xy: np.ndarray, chart: CoordinateChart | None) -> np.ndarray:
    """Map output sample indices into the source's physical chart."""

    return _chart_coordinates(points_xy, chart)


def _coerce_layout(layout: Any, shape: tuple[int, int] | None) -> SampleLayout:
    if isinstance(layout, SampleLayout):
        if len(layout.shape) != 2:
            raise AnalyticOceanError("ocean layouts must be two-dimensional")
        return layout
    if layout is None:
        if shape is None:
            raise AnalyticOceanError("a two-dimensional shape, layout, or points is required")
        return SampleLayout(tuple(shape))
    if isinstance(layout, Mapping):
        return SampleLayout(
            tuple(layout["shape"]), tuple(layout.get("origin", (0.0, 0.0))), tuple(layout.get("spacing", (1.0, 1.0)))
        )
    return SampleLayout(tuple(int(value) for value in layout))


def _coerce_shape_2d(shape: Sequence[int] | None) -> tuple[int, int] | None:
    """Normalize an ingress shape while retaining the two-dimensional contract."""

    if shape is None:
        return None
    values = tuple(int(value) for value in shape)
    if len(values) != 2:
        raise AnalyticOceanError("analytic ocean shape must be two-dimensional")
    return values[0], values[1]


def _coerce_points(points: Any) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(points, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 2:
        raise AnalyticOceanError("points must have trailing shape (2,), ordered as (x, y)")
    if not np.all(np.isfinite(array)):
        raise AnalyticOceanError("points must be finite")
    return array.reshape(-1, 2), array.shape[:-1]


def _region_mask(points: np.ndarray, support: Support | Region | None) -> np.ndarray | None:
    if support is None:
        return None
    if isinstance(support, Region):
        region = support
        kind = "AABB"
    else:
        kind = str(support.kind)
        if kind.endswith("EMPTY"):
            return np.zeros(points.shape[0], dtype=bool)
        if kind.endswith("FULL") or kind.endswith("UNKNOWN") or kind.endswith("FREQUENCY_BAND"):
            return None
        if not kind.endswith("AABB"):
            raise AnalyticOceanError(f"analytic ocean support kind {kind!r} has no point-mask evaluator")
        nested_region = support.region
        if nested_region is None:
            raise AnalyticOceanError("AABB ocean support requires a region")
        region = nested_region
    if region.dimension != 2:
        raise AnalyticOceanError("ocean support must be two-dimensional")
    lower = np.asarray(region.lower, dtype=np.float64)
    upper = np.asarray(region.upper, dtype=np.float64)
    return np.asarray(np.all((points >= lower) & (points <= upper), axis=1), dtype=bool)


def _fit_callable_quadratic(
    evaluator: Callable[[np.ndarray], Any], center_value: Any
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit local Taylor coefficients for a callable dispersion diagnostic."""

    center_array = np.asarray(center_value, dtype=np.float64).reshape(2)
    step = 1.0e-4
    f0 = float(np.asarray(evaluator(center_array)).reshape(-1)[0])
    f_plus = np.asarray([evaluator(center_array + np.eye(2)[axis] * step) for axis in range(2)], dtype=float).reshape(2)
    f_minus = np.asarray([evaluator(center_array - np.eye(2)[axis] * step) for axis in range(2)], dtype=float).reshape(
        2
    )
    gradient = (f_plus - f_minus) / (2.0 * step)
    hessian = np.zeros((2, 2), dtype=np.float64)
    for axis in range(2):
        hessian[axis, axis] = (f_plus[axis] - 2.0 * f0 + f_minus[axis]) / (step * step)
    f_xy = float(np.asarray(evaluator(center_array + np.array([step, step]))).reshape(-1)[0])
    f_xmy = float(np.asarray(evaluator(center_array + np.array([step, -step]))).reshape(-1)[0])
    f_mxy = float(np.asarray(evaluator(center_array + np.array([-step, step]))).reshape(-1)[0])
    f_mxmy = float(np.asarray(evaluator(center_array - np.array([step, step]))).reshape(-1)[0])
    hessian[0, 1] = hessian[1, 0] = (f_xy - f_xmy - f_mxy + f_mxmy) / (4.0 * step * step)
    return f0, gradient, hessian


def _quadratic_dispersion(
    dispersion: Any,
    *,
    center: Sequence[float] | None = None,
    sign: float = -1.0,
    support: Any = None,
) -> _DispersionSpec | None:
    if dispersion is None:
        return None
    selected_sign = float(sign)
    certified_residual: float | None = None
    third_derivative_bound: float | None = None
    if isinstance(dispersion, Mapping):
        kind = str(dispersion.get("kind", dispersion.get("type", "quadratic"))).lower()
        selected_sign = float(dispersion.get("sign", selected_sign))
        center_value = dispersion.get("center", center if center is not None else (0.0, 0.0))
        value = dispersion.get("value", dispersion.get("omega", 0.0))
        gradient = dispersion.get("gradient", dispersion.get("grad", (0.0, 0.0)))
        hessian = dispersion.get("hessian", dispersion.get("hess", np.zeros((2, 2))))
        evaluator = dispersion.get("evaluate", dispersion.get("fn"))
        raw_residual = dispersion.get("residual_bound", dispersion.get("certified_residual_bound"))
        if raw_residual is not None:
            certified_residual = float(raw_residual)
        raw_third = dispersion.get("third_derivative_bound")
        if raw_third is not None:
            third_derivative_bound = float(raw_third)
        if (
            callable(evaluator)
            and kind not in {"quadratic", "quad", "polynomial2", "second_order"}
            and not any(key in dispersion for key in ("value", "omega", "gradient", "grad", "hessian", "hess"))
        ):
            value, gradient, hessian = _fit_callable_quadratic(evaluator, center_value)
    elif all(hasattr(dispersion, name) for name in ("value", "gradient", "hessian")):
        kind = str(getattr(dispersion, "kind", "quadratic")).lower()
        center_value = getattr(dispersion, "center", center if center is not None else (0.0, 0.0))
        value = dispersion.value
        gradient = dispersion.gradient
        hessian = dispersion.hessian
        evaluator = getattr(dispersion, "evaluate", None)
        selected_sign = float(getattr(dispersion, "sign", selected_sign))
        raw_residual = getattr(dispersion, "residual_bound", None)
        if raw_residual is not None:
            certified_residual = float(raw_residual)
        raw_third = getattr(dispersion, "third_derivative_bound", None)
        if raw_third is not None:
            third_derivative_bound = float(raw_third)
    elif _is_sequence(dispersion) and len(dispersion) >= 3:
        kind = "quadratic"
        center_value = center if center is not None else (0.0, 0.0)
        value, gradient, hessian = dispersion[:3]
        evaluator = None
    elif callable(dispersion):
        kind = "nonquadratic"
        center_value = center if center is not None else (0.0, 0.0)
        f0, gradient, hessian = _fit_callable_quadratic(dispersion, center_value)
        evaluator = dispersion
        value = f0
    else:
        raise AnalyticOceanError("dispersion must be a quadratic descriptor, sequence, object, or callable")

    center_array = np.asarray(center_value, dtype=np.float64).reshape(-1)
    gradient_array = np.asarray(gradient, dtype=np.float64).reshape(-1)
    hessian_array = np.asarray(hessian, dtype=np.float64)
    if center_array.size != 2 or gradient_array.size != 2 or hessian_array.shape != (2, 2):
        raise AnalyticOceanError("dispersion descriptors must contain two-dimensional center/gradient/hessian")
    if not np.allclose(hessian_array, hessian_array.T):
        raise AnalyticOceanError("dispersion hessian must be symmetric")

    # A supplied callable is explicitly non-closed, even when its local
    # Taylor coefficients happen to be zero.  A descriptor marked quadratic
    # is the exact closure contract.
    nonquadratic = kind not in {"quadratic", "quad", "polynomial2", "second_order"} or (
        evaluator is not None and callable(dispersion)
    )
    residual_per_time = 0.0
    if nonquadratic:
        if evaluator is None or not callable(evaluator):
            raise AnalyticOceanError("non-quadratic dispersion requires a callable evaluator")
        if certified_residual is not None:
            if not math.isfinite(certified_residual) or certified_residual < 0.0:
                raise AnalyticOceanError("certified dispersion residual bound must be finite and non-negative")
            residual_per_time = certified_residual
        elif third_derivative_bound is not None and support is not None:
            if not math.isfinite(third_derivative_bound) or third_derivative_bound < 0.0:
                raise AnalyticOceanError("third derivative bound must be finite and non-negative")
            radius = _support_radius(support, center_array)
            residual_per_time = third_derivative_bound * radius**3 / 6.0
        else:
            # A finite maximum over sample points is a diagnostic, not a
            # continuum certificate.  Keep this route explicitly unbounded
            # until the caller supplies a certified residual contract.
            residual_per_time = math.inf
    return _DispersionSpec(
        kind="nonquadratic" if nonquadratic else "quadratic",
        center=center_array,
        value=float(value),
        gradient=gradient_array,
        hessian=hessian_array,
        sign=selected_sign,
        evaluator=evaluator if callable(evaluator) else None,
        residual_bound_per_time=float(residual_per_time),
        residual_certified=bool(nonquadratic and math.isfinite(residual_per_time)),
    )


def _dispersion_sampled_residual(
    evaluator: Callable[[np.ndarray], Any],
    center: np.ndarray,
    value: float,
    gradient: np.ndarray,
    hessian: np.ndarray,
    support: Any,
) -> float:
    """Diagnostic grid residual; never used as a continuum error certificate."""
    if support is None:
        return math.inf
    if isinstance(support, Region):
        lower, upper = np.asarray(support.lower), np.asarray(support.upper)
    elif isinstance(support, Mapping) and "lower" in support and "upper" in support:
        lower, upper = np.asarray(support["lower"], dtype=float), np.asarray(support["upper"], dtype=float)
    else:
        array = np.asarray(support, dtype=float)
        if array.shape == (2, 2):
            lower, upper = array[0], array[1]
        else:
            return math.inf
    if lower.shape != (2,) or upper.shape != (2,):
        return math.inf
    axes = [np.linspace(lower[axis], upper[axis], 5) for axis in range(2)]
    residual = 0.0
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    try:
        exact = np.asarray(evaluator(points), dtype=float).reshape(-1)
    except Exception:
        exact = np.asarray([float(np.asarray(evaluator(point)).reshape(-1)[0]) for point in points])
    delta = points - center
    approx = value + delta @ gradient + 0.5 * np.einsum("...i,ij,...j->...", delta, hessian, delta)
    if exact.size != approx.size or not np.all(np.isfinite(exact)):
        return math.inf
    residual = float(np.max(np.abs(exact - approx)))
    return residual


def _support_radius(support: Any, center: np.ndarray) -> float:
    """Return a Euclidean box radius for a certified Taylor remainder."""

    if isinstance(support, Region):
        lower = np.asarray(support.lower, dtype=np.float64)
        upper = np.asarray(support.upper, dtype=np.float64)
    elif isinstance(support, Mapping) and "lower" in support and "upper" in support:
        lower = np.asarray(support["lower"], dtype=np.float64)
        upper = np.asarray(support["upper"], dtype=np.float64)
    else:
        array = np.asarray(support, dtype=np.float64)
        if array.shape != (2, 2):
            return math.inf
        lower, upper = array[0], array[1]
    if lower.shape != (2,) or upper.shape != (2,):
        return math.inf
    corner_distance = np.maximum(np.abs(lower - center), np.abs(upper - center))
    return float(np.linalg.norm(corner_distance))


def _phase_atom(spec: _DispersionSpec, time: float) -> GaussianChirpAtom:
    """Build ``exp(sign*i*time*omega(k))`` in atom coordinates."""

    constant = spec.value - spec.gradient @ spec.center + 0.5 * spec.center @ spec.hessian @ spec.center
    kappa = spec.sign * 1j * float(time) * constant
    h = spec.sign * 1j * float(time) * (spec.gradient - spec.hessian @ spec.center)
    C = -spec.sign * 1j * float(time) * spec.hessian
    # This atom is a pure phase and is intentionally formal/nondecaying.
    return GaussianChirpAtom(kappa, h, C, allow_nondecaying=True)


def _apply_dispersion_to_atoms(
    atoms: Sequence[GaussianChirpAtom],
    domain: str,
    spec: _DispersionSpec | None,
    time: float,
) -> tuple[tuple[GaussianChirpAtom, ...], tuple[GaussianChirpAtom, ...]]:
    if not atoms:
        return (), ()
    if domain not in {str(Domain.FOURIER_2D), "FOURIER_2D", ""}:
        if spec is not None:
            raise AnalyticOceanError("temporal dispersion is defined on Fourier atoms, not spatial atoms")
        return tuple(atoms), ()
    try:
        fourier_atoms = tuple(atoms)
        if spec is None or time == 0.0:
            spatial = tuple(atom_inverse_fourier(atom) for atom in fourier_atoms)
            return spatial, fourier_atoms
        phase = _phase_atom(spec, time)
        evolved = tuple(atom_mul(atom, phase) for atom in fourier_atoms)
        spatial = tuple(atom_inverse_fourier(atom) for atom in evolved)
        return spatial, fourier_atoms
    except ValueError as exc:
        raise AnalyticOceanError(
            f"Fourier Gaussian atom is not integrable for analytic inverse transform: {exc}"
        ) from exc


def _canonical_indices(index: Sequence[int], shape: tuple[int, int]) -> tuple[tuple[int, int], bool]:
    row, col = (int(index[0]) % shape[0], int(index[1]) % shape[1])
    partner = ((-row) % shape[0], (-col) % shape[1])
    canonical = min((row, col), partner)
    return canonical, (row, col) != canonical


def _sparse_modes_from_arrays(
    indices: Any,
    coefficients: Any,
    shape: tuple[int, int],
    *,
    phases: Any = None,
    phase_increments: Any = None,
    phase_bits: int = 16,
    normalize: bool = True,
    tolerance: float = _DEFAULT_HERMITIAN_TOLERANCE,
) -> tuple[_SparseMode, ...]:
    raw_indices = np.asarray(indices, dtype=np.int64)
    if raw_indices.ndim == 1 and raw_indices.size == 2:
        raw_indices = raw_indices.reshape(1, 2)
    if raw_indices.ndim != 2 or raw_indices.shape[1] != 2:
        raise AnalyticOceanError("sparse ocean indices must have shape (K, 2), ordered (row, column)")
    values = _array_complex(coefficients).reshape(-1)
    if values.size != raw_indices.shape[0]:
        raise AnalyticOceanError("sparse ocean coefficients and indices have different lengths")
    phase_values = [None] * values.size if phases is None else list(np.asarray(phases, dtype=object).reshape(-1))
    increment_values = (
        [None] * values.size
        if phase_increments is None
        else list(np.asarray(phase_increments, dtype=object).reshape(-1))
    )
    if len(phase_values) != values.size or len(increment_values) != values.size:
        raise AnalyticOceanError("sparse ocean phase arrays must match the number of indices")
    grouped: dict[tuple[int, int], _SparseMode] = {}
    for position, (index, value) in enumerate(zip(raw_indices, values, strict=True)):
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise AnalyticOceanError("sparse ocean coefficients must be finite")
        canonical, conjugated = _canonical_indices(index, shape)
        phase = _phase_ticks(phase_values[position], phase_bits)
        increment = _phase_ticks(increment_values[position], phase_bits)
        current = complex(value)
        if conjugated:
            current = current.conjugate()
            phase = -phase
            increment = -increment
        existing = grouped.get(canonical)
        if existing is not None:
            expected = existing.base * complex(math.cos(existing.phase_radians()), math.sin(existing.phase_radians()))
            incoming = current * complex(
                math.cos(phase * _TAU / (1 << phase_bits)), math.sin(phase * _TAU / (1 << phase_bits))
            )
            if abs(expected - incoming) > tolerance * max(1.0, abs(expected), abs(incoming)):
                raise AnalyticOceanError(f"Hermitian sparse pair {canonical} has inconsistent coefficients")
            continue
        self_conjugate = canonical == ((-canonical[0]) % shape[0], (-canonical[1]) % shape[1])
        if self_conjugate:
            angle = phase * _TAU / float(1 << phase_bits)
            evolved = current * complex(math.cos(angle), math.sin(angle))
            if abs(evolved.imag) > tolerance * max(1.0, abs(evolved)):
                raise AnalyticOceanError(f"self-conjugate ocean mode {canonical} must be real")
            if increment % (1 << max(0, phase_bits - 1)) != 0:
                raise AnalyticOceanError(
                    f"self-conjugate ocean mode {canonical} cannot carry a complex phase increment"
                )
            current = complex(evolved.real, 0.0)
            phase = 0
        grouped[canonical] = _SparseMode(
            canonical,
            current,
            phase_ticks=phase,
            phase_increment=increment,
            phase_bits=int(phase_bits),
            pair_weight=1.0 if self_conjugate else 2.0,
        )
    factor = 1.0 / float(shape[0] * shape[1]) if normalize else 1.0
    for mode in grouped.values():
        mode.base *= factor
    return tuple(grouped.values())


def _block_sparse_modes(
    block: BlockView,
    shape: tuple[int, int],
    *,
    normalize: bool,
    max_sparse_modes: int | None,
    tolerance: float,
) -> tuple[_SparseMode, ...]:
    if str(block.observable).lower() == "intensity":
        raise AnalyticOceanError("intensity blocks do not carry coherent Fourier phases for an ocean height")
    if tuple(block.shape) != tuple(shape):
        raise AnalyticOceanError(f"addressed Fourier source shape {block.shape} does not match ocean shape {shape}")
    active = np.asarray(block.active_addresses(), dtype=np.int64).reshape(-1)
    if max_sparse_modes is not None and active.size > int(max_sparse_modes):
        raise AnalyticOceanError(
            f"addressed Fourier source has {active.size} active modes; "
            f"compact analytic ocean limit is {max_sparse_modes}"
        )
    if active.size == 0:
        return ()
    words = block.sample_words(active, phase=block.observable != "intensity")
    values = _array_complex(block.project_words(words, active))
    indices = np.column_stack(np.unravel_index(active, shape))
    return _sparse_modes_from_arrays(indices, values, shape, normalize=normalize, tolerance=tolerance)


def _extract_sparse_source(
    source: Any, shape: tuple[int, int] | None
) -> tuple[Any, Any, Any, Any, tuple[int, ...] | None, Mapping[str, Any]]:
    metadata: dict[str, Any] = {}
    source_shape: tuple[int, ...] | None
    modes = tuple(getattr(source, "modes", ()) or ())
    if modes:
        source_shape = shape or tuple(int(x) for x in source.shape)
        indices = []
        values = []
        phases = []
        increments = []
        metadata["mode_objects"] = modes
        for mode in modes:
            indices.append(tuple(mode.index))
            values.append(getattr(mode, "coefficient", getattr(mode, "amplitude", 0.0)))
            accumulator = getattr(mode, "phase_accumulator", None)
            if accumulator is not None:
                phase = getattr(accumulator, "phase_code", getattr(accumulator, "phase", 0))
                increment = getattr(accumulator, "increment", getattr(mode, "phase_increment", 0))
            else:
                phase = getattr(mode, "phase", 0)
                increment = getattr(mode, "phase_increment", 0)
            phases.append(phase)
            increments.append(increment)
        return np.asarray(indices), np.asarray(values, dtype=object), phases, increments, source_shape, metadata
    index_values = getattr(source, "indices", None)
    coefficient_values = getattr(source, "coefficients", None)
    if index_values is not None and coefficient_values is not None:
        if shape is not None:
            source_shape = shape
        elif getattr(source, "shape", None) is not None:
            source_shape = tuple(int(x) for x in source.shape)
        else:
            source_shape = None
        if source_shape is None:
            raise AnalyticOceanError("sparse ocean arrays require shape")
        source_phases = getattr(source, "phases", None)
        source_increments = getattr(source, "phase_increments", None)
        if source_phases is not None and len(tuple(source_phases)) == 0:
            source_phases = None
        if source_increments is not None and len(tuple(source_increments)) == 0:
            source_increments = None
        return (
            index_values,
            coefficient_values,
            source_phases,
            source_increments,
            tuple(source_shape),
            metadata,
        )
    return None, None, None, None, shape, metadata


def _footprint_covariance(footprint: Any) -> np.ndarray | None:
    """Read a Gaussian footprint covariance without sampling the footprint."""

    if footprint is None:
        return None
    if isinstance(footprint, GaussianChirpAtom):
        if footprint.dimension != 2:
            raise AnalyticOceanError("ocean Gaussian footprints must be two-dimensional")
        real_C = np.real(footprint.C)
        try:
            return np.linalg.inv(real_C)
        except np.linalg.LinAlgError as exc:
            raise AnalyticOceanError("Gaussian footprint covariance requires an invertible real precision") from exc
    if isinstance(footprint, Mapping):
        covariance = footprint.get("covariance", footprint.get("sigma", footprint.get("variance")))
        if covariance is None and "radius" in footprint:
            covariance = np.eye(2) * float(footprint["radius"]) ** 2
    elif hasattr(footprint, "covariance"):
        covariance = footprint.covariance
    else:
        covariance = footprint
    array = np.asarray(covariance, dtype=np.float64)
    if array.ndim == 0:
        array = np.eye(2, dtype=np.float64) * float(array) ** 2
    if array.shape != (2, 2) or not np.all(np.isfinite(array)):
        raise AnalyticOceanError("ocean footprint covariance must be a finite 2x2 matrix")
    if not np.allclose(array, array.T):
        raise AnalyticOceanError("ocean footprint covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(array))) < -1.0e-12:
        raise AnalyticOceanError("ocean footprint covariance must be positive semidefinite")
    return array


def _owned_footprint(footprint: Any) -> Any:
    """Snapshot the small footprint descriptor owned by a prepared consumer."""
    if footprint is None:
        return None
    if isinstance(footprint, GaussianChirpAtom):
        return Field.from_value(footprint).representation.payload
    covariance = _footprint_covariance(footprint)
    assert covariance is not None
    covariance = covariance.copy()
    covariance.setflags(write=False)
    return MappingProxyType({"covariance": covariance})


def _filter_atoms(atoms: Sequence[GaussianChirpAtom], footprint: Any) -> tuple[GaussianChirpAtom, ...]:
    """Apply the same footprint closure as a deferred Field Gaussian filter."""
    if footprint is None or not atoms:
        return tuple(atoms)
    if isinstance(footprint, GaussianChirpAtom):
        return tuple(atom.convolve(footprint) for atom in atoms)
    from spf.renderer.model import FieldOperation
    from spf.renderer.normalization import _apply_atom_operation

    covariance = _footprint_covariance(footprint)
    field = Field.from_value(atoms[0], domain=Domain.SPATIAL_2D)
    operation = FieldOperation("gaussian_filter", {"covariance": covariance})
    return tuple(_apply_atom_operation(atom, operation, field) for atom in atoms)


def _evaluate_residual(
    residual: Any, points: np.ndarray, *, allow_layout_gradient: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Evaluate an optional residual and return value/gradient/unknown status."""

    if residual is None:
        zeros = np.zeros(points.shape[0], dtype=np.complex128)
        return zeros, zeros.copy(), zeros.copy(), False
    from spf.renderer.normalization import _ResidualSum, _ScaledResidual

    if isinstance(residual, _ScaledResidual):
        value, grad_x, grad_y, unknown = _evaluate_residual(
            residual.residual, points, allow_layout_gradient=allow_layout_gradient
        )
        return residual.scale_values(value), residual.scale_values(grad_x), residual.scale_values(grad_y), unknown
    if isinstance(residual, _ResidualSum):
        values = np.zeros(points.shape[0], dtype=np.complex128)
        gradient_x_values, gradient_y_values = values.copy(), values.copy()
        unknown_gradient = False
        for item in residual.items:
            value, item_x, item_y, unknown = _evaluate_residual(
                item, points, allow_layout_gradient=allow_layout_gradient
            )
            values += value
            gradient_x_values += item_x
            gradient_y_values += item_y
            unknown_gradient |= unknown
        return values, gradient_x_values, gradient_y_values, unknown_gradient
    evaluator = residual
    gradient_x = gradient_y = None
    if isinstance(residual, Mapping):
        evaluator = residual.get("evaluate", residual.get("values", residual.get("value")))
        gradient_x = residual.get("gradient_x", residual.get("dx"))
        gradient_y = residual.get("gradient_y", residual.get("dy"))
    if allow_layout_gradient and hasattr(evaluator, "layout_gradients"):
        values, gradient_x_value, gradient_y_value, unknown = evaluator.layout_gradients(points)
        return (
            np.asarray(values, dtype=np.complex128).reshape(-1),
            np.asarray(gradient_x_value, dtype=np.complex128).reshape(-1),
            np.asarray(gradient_y_value, dtype=np.complex128).reshape(-1),
            bool(unknown),
        )
    if hasattr(evaluator, "sample_flat") and hasattr(evaluator, "shape"):
        # SparseResidual is an addressed source, not a callable field.  It is
        # legal for an integer layout query and must be consumed through its
        # bounded sample API; continuous coordinates are rejected explicitly.
        residual_shape = tuple(int(value) for value in evaluator.shape)
        if len(residual_shape) != 2:
            raise AnalyticOceanError("addressed residuals for ocean sampling must be two-dimensional")
        local = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rounded = np.rint(local)
        if np.any(~np.isclose(local, rounded, rtol=0.0, atol=1.0e-10)):
            raise AnalyticOceanError(
                "addressed residuals require integer sample coordinates; use a continuous residual evaluator"
            )
        row, column = rounded[:, 1].astype(np.int64), rounded[:, 0].astype(np.int64)
        inside = (row >= 0) & (row < residual_shape[0]) & (column >= 0) & (column < residual_shape[1])
        values = np.zeros(local.shape[0], dtype=np.complex128)
        if np.any(inside):
            flat = np.ravel_multi_index((row[inside], column[inside]), residual_shape)
            values[inside] = _array_complex(evaluator.sample_flat(flat)).reshape(-1)
    elif callable(evaluator):
        try:
            values = _array_complex(evaluator(points)).reshape(-1)
        except Exception:
            values = np.asarray([_as_complex(evaluator(point)) for point in points], dtype=np.complex128)
    else:
        array = _array_complex(evaluator)
        if array.ndim == 0:
            values = np.full(points.shape[0], complex(array), dtype=np.complex128)
        elif array.size == points.shape[0]:
            values = array.reshape(-1)
        else:
            raise AnalyticOceanError(
                "a residual array can be consumed only when it already matches the requested point count; "
                "a dense residual grid is not implicitly materialized"
            )

    def gradient_component(component: Any) -> np.ndarray | None:
        if component is None:
            return None
        if callable(component):
            try:
                return _array_complex(component(points)).reshape(-1)
            except Exception:
                return np.asarray([_as_complex(component(point)) for point in points], dtype=np.complex128)
        array = _array_complex(component)
        if array.ndim == 0:
            return np.full(points.shape[0], complex(array), dtype=np.complex128)
        if array.size == points.shape[0]:
            return array.reshape(-1)
        raise AnalyticOceanError("residual gradient array does not match requested points")

    gx = gradient_component(gradient_x)
    gy = gradient_component(gradient_y)
    unknown = gx is None or gy is None
    if gx is None:
        gx = np.zeros(points.shape[0], dtype=np.complex128)
    if gy is None:
        gy = np.zeros(points.shape[0], dtype=np.complex128)
    return values, gx, gy, bool(unknown and np.any(np.abs(values) > 0.0))


def _sparse_phase_geometry(
    mode: _SparseMode,
    shape: tuple[int, int],
    points_xy: np.ndarray,
    chart: CoordinateChart | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return phase and its physical x/y derivatives for one lattice mode."""

    matrix, _ = _chart_matrix(chart)
    try:
        inverse_transpose = np.linalg.inv(matrix).T
    except np.linalg.LinAlgError as exc:
        raise AnalyticOceanError("coordinate chart basis must be invertible for Fourier gradients") from exc
    row, column = mode.index
    # Sparse source addresses are (row, column), while public points are
    # (x, y).  A chart maps output sample coordinates to physical coordinates;
    # the inverse transpose maps phase derivatives into physical units.
    lattice_gradient = _TAU * np.asarray((column / shape[1], row / shape[0]), dtype=np.float64)
    physical_gradient = inverse_transpose @ lattice_gradient
    phase = _TAU * (points_xy @ np.asarray((column / shape[1], row / shape[0]), dtype=np.float64))
    return phase, np.full(points_xy.shape[0], physical_gradient[0]), np.full(points_xy.shape[0], physical_gradient[1])


def _mode_frequency(mode: _SparseMode, shape: tuple[int, int]) -> np.ndarray:
    row, column = mode.index
    signed_row = row if row <= shape[0] // 2 else row - shape[0]
    signed_column = column if column <= shape[1] // 2 else column - shape[1]
    return np.asarray((signed_column / shape[1], signed_row / shape[0]), dtype=np.float64)


def _configure_sparse_temporal(
    modes: Sequence[_SparseMode], shape: tuple[int, int], spec: _DispersionSpec | None, time: float
) -> None:
    if spec is None:
        return
    for mode in modes:
        frequency = _mode_frequency(mode, shape)
        if spec.evaluator is not None and spec.kind == "nonquadratic":
            omega = float(np.asarray(spec.evaluator(frequency)).reshape(-1)[0])
        else:
            delta = frequency - spec.center
            omega = float(spec.value + delta @ spec.gradient + 0.5 * delta @ spec.hessian @ delta)
        if not math.isfinite(omega):
            raise AnalyticOceanError("dispersion evaluated to a non-finite mode frequency")
        mode.sign = spec.sign
        # A supplied integer phase increment is the source's explicit timeline
        # and remains authoritative.  Otherwise a scalar mode is exactly
        # closed under any dispersion, including deep-water sqrt(|k|).
        if mode.phase_increment == 0:
            mode.angular_frequency = omega
        mode.time_phase = spec.sign * omega * float(time)


def _jet_values_and_gradients(jet: Jet2D, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kind = str(jet.value_kind).lower()
    if kind in {"log", "log_pos", "log_signed", "phase"}:
        raise AnalyticOceanError(f"Jet2D value kind {jet.value_kind!r} needs a typed emitter before ocean sampling")
    x, y = points[:, 0], points[:, 1]
    value = (
        complex(jet.value)
        + complex(jet.grad_x) * x
        + complex(jet.grad_y) * y
        + 0.5 * complex(jet.hess_xx) * x * x
        + complex(jet.hess_xy) * x * y
        + 0.5 * complex(jet.hess_yy) * y * y
    )
    gradient_x = complex(jet.grad_x) + complex(jet.hess_xx) * x + complex(jet.hess_xy) * y
    gradient_y = complex(jet.grad_y) + complex(jet.hess_xy) * x + complex(jet.hess_yy) * y
    return value, gradient_x, gradient_y


def _spectral_jet_atom(tile: Any) -> GaussianChirpAtom:
    """Lower a guarded log/phase tile to one complex quadratic atom.

    ``SpectralJetTile2D`` stores widened integer-like coefficients.  The
    conversion is exact in the declared widened arithmetic: log coefficients
    become natural-log coefficients and phase coefficients become radians;
    the resulting complex quadratic polynomial is then represented by the
    ordinary Gaussian-chirp atom formula.  No packed plane or tile emission
    is needed.
    """

    log_jet = tile.log_magnitude
    phase_jet = tile.phase
    log_scale = math.log(2.0) / (float(v02.SPF32_L_SCALE) * float(1 << int(log_jet.guard_L)))
    phase_scale = _TAU / (float(v02.SPF32_PHASE_MOD) * float(1 << int(phase_jet.guard_T)))
    log_coefficients = (
        np.asarray(
            (
                log_jet.value,
                log_jet.grad_x,
                log_jet.grad_y,
                log_jet.hess_xx,
                log_jet.hess_xy,
                log_jet.hess_yy,
            ),
            dtype=np.float64,
        )
        * log_scale
    )
    phase_coefficients = (
        np.asarray(
            (
                phase_jet.value,
                phase_jet.grad_x,
                phase_jet.grad_y,
                phase_jet.hess_xx,
                phase_jet.hess_xy,
                phase_jet.hess_yy,
            ),
            dtype=np.float64,
        )
        * phase_scale
    )
    complex_coefficients = log_coefficients + 1j * phase_coefficients
    hessian = np.asarray(
        (
            (complex_coefficients[3], complex_coefficients[4]),
            (complex_coefficients[4], complex_coefficients[5]),
        ),
        dtype=np.complex128,
    )
    # The atom's convention is exp(kappa + h^T x - 1/2 x^T C x), whereas a
    # spectral jet stores +1/2 x^T H x.
    return GaussianChirpAtom(
        complex(complex_coefficients[0]),
        np.asarray(complex_coefficients[1:3], dtype=np.complex128),
        -hessian,
        allow_nondecaying=True,
    )


class _SpectralJetResidualEvaluator:
    """Addressed SpectralJet residual with a layout-only derivative policy.

    The residual stores corrections at integer tile samples, so it has no
    mathematically defined continuous derivative by itself.  Point queries
    therefore remain rejected whenever a correction is needed.  A raster
    layout can opt into the explicit discrete policy below: centered
    differences at interior samples and one-sided differences at tile edges,
    with one source-coordinate sample as the step.  This evaluates adjacent
    addressed samples directly and never creates a dense residual plane.  The
    resulting slopes are finite render outputs with an ``unknown`` error
    status in the report rather than certified continuous derivatives.
    """

    layout_gradient_policy = "addressed_sample_finite_difference_not_certified"

    def __init__(self, tile: Any, base_atom: GaussianChirpAtom, extra_residual: Any = None) -> None:
        self.tile = tile
        self.base_atom = base_atom
        self.extra_residual = extra_residual
        self.log_residual = tile.log_residual
        self.phase_residual = tile.phase_residual
        self.zero_mask = None if tile.zero_mask is None else np.asarray(tile.zero_mask, dtype=bool).reshape(-1)
        self.rows, self.columns = int(tile.shape[0]), int(tile.shape[1])
        self.origin_x, self.origin_y = float(tile.origin[0]), float(tile.origin[1])
        self.log_scale = math.log(2.0) / (float(v02.SPF32_L_SCALE) * float(1 << int(tile.log_magnitude.guard_L)))
        self.phase_scale = _TAU / (float(v02.SPF32_PHASE_MOD) * float(1 << int(tile.phase.guard_T)))

    def __call__(self, points: np.ndarray) -> np.ndarray:
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        local_x = array[:, 0] - self.origin_x
        local_y = array[:, 1] - self.origin_y
        rounded_x = np.rint(local_x)
        rounded_y = np.rint(local_y)
        valid_grid = np.isclose(local_x, rounded_x, rtol=0.0, atol=1.0e-10) & np.isclose(
            local_y, rounded_y, rtol=0.0, atol=1.0e-10
        )
        inside = (rounded_x >= 0.0) & (rounded_x < self.columns) & (rounded_y >= 0.0) & (rounded_y < self.rows)
        if np.any(~valid_grid & inside):
            raise AnalyticOceanError(
                "SpectralJetTile2D residuals are addressed at integer tile samples; "
                "continuous residual queries require an explicit residual evaluator"
            )
        values = np.zeros(array.shape[0], dtype=np.complex128)
        active = valid_grid & inside
        if not np.any(active):
            return values
        flat = (rounded_y[active].astype(np.int64) * self.columns + rounded_x[active].astype(np.int64)).astype(np.int64)
        base = np.asarray(self.base_atom.evaluate(array[active]), dtype=np.complex128).reshape(-1)
        correction = np.ones(flat.size, dtype=np.complex128)
        if self.log_residual is not None:
            correction *= np.exp(np.asarray(self.log_residual.sample_flat(flat), dtype=np.float64) * self.log_scale)
        if self.phase_residual is not None:
            correction *= np.exp(
                1j * np.asarray(self.phase_residual.sample_flat(flat), dtype=np.float64) * self.phase_scale
            )
        corrected = base * (correction - 1.0)
        if self.zero_mask is not None:
            zero = self.zero_mask[flat]
            corrected[zero] = -base[zero]
        values[active] = corrected
        if self.extra_residual is not None:
            extra, _extra_x, _extra_y, _unknown = _evaluate_residual(self.extra_residual, array)
            values += np.asarray(extra, dtype=np.complex128)
        return values

    def layout_gradients(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Return discrete residual ``(value, dx, dy, unknown)`` for raster samples."""

        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        local_x = np.rint(array[:, 0] - self.origin_x).astype(np.int64)
        local_y = np.rint(array[:, 1] - self.origin_y).astype(np.int64)
        exact = np.isclose(array[:, 0] - self.origin_x, local_x, rtol=0.0, atol=1.0e-10) & np.isclose(
            array[:, 1] - self.origin_y, local_y, rtol=0.0, atol=1.0e-10
        )
        inside = (local_x >= 0) & (local_x < self.columns) & (local_y >= 0) & (local_y < self.rows)
        if np.any(~exact & inside):
            raise AnalyticOceanError(
                "SpectralJetTile2D residual gradients require integer layout samples; "
                "continuous residual queries require an explicit residual evaluator"
            )
        values = self(array)
        gradient_x = np.zeros(values.shape, dtype=np.complex128)
        gradient_y = np.zeros(values.shape, dtype=np.complex128)
        active = exact & inside
        if not np.any(active):
            return values, gradient_x, gradient_y, False
        active_indices = np.flatnonzero(active)
        for axis, count, output in ((0, self.columns, gradient_x), (1, self.rows, gradient_y)):
            coordinate = local_x if axis == 0 else local_y
            plus_points = array.copy()
            minus_points = array.copy()
            plus_points[:, axis] += 1.0
            minus_points[:, axis] -= 1.0
            plus = self(plus_points[active_indices])
            minus = self(minus_points[active_indices])
            current = values[active_indices]
            at_lower = coordinate[active_indices] <= 0
            at_upper = coordinate[active_indices] >= count - 1
            estimate = (plus - minus) * 0.5
            estimate[at_lower] = plus[at_lower] - current[at_lower]
            estimate[at_upper] = current[at_upper] - minus[at_upper]
            output[active_indices] = estimate
        return values, gradient_x, gradient_y, False


def _spectral_jet_residual(
    tile: Any, base_atom: GaussianChirpAtom, extra_residual: Any = None
) -> _SpectralJetResidualEvaluator:
    """Build the addressed residual evaluator without decoding a dense tile."""

    return _SpectralJetResidualEvaluator(tile, base_atom, extra_residual)


def _jet_footprint(
    jet: Jet2D, points: np.ndarray, covariance: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values, gradient_x, gradient_y = _jet_values_and_gradients(jet, points)
    if covariance is None:
        return values, gradient_x, gradient_y
    cxx, cxy, cyy = covariance[0, 0], covariance[0, 1], covariance[1, 1]
    correction = 0.5 * (complex(jet.hess_xx) * cxx + 2.0 * complex(jet.hess_xy) * cxy + complex(jet.hess_yy) * cyy)
    return values + correction, gradient_x, gradient_y


@dataclass(slots=True)
class PreparedAnalyticOcean:
    """Prepared point/tile consumer for compact analytic ocean sources."""

    plan: OceanAnalyticPlan
    shape: tuple[int, int] | None = None
    chart: CoordinateChart | None = None
    layout: SampleLayout | None = None
    source: Any = None
    atom_source: _AtomSource | None = None
    jet: Jet2D | None = None
    jet_residual: Any = None
    sparse_modes: tuple[_SparseMode, ...] = ()
    support: Support | Region | None = None
    visible: Any = None
    footprint: Any = None
    footprint_covariance: np.ndarray | None = None
    output_dtype: np.dtype = field(default_factory=lambda: np.dtype(np.float64))
    output_format: str = "real"
    chunk_size: int = _DEFAULT_CHUNK
    tolerance: float = _DEFAULT_HERMITIAN_TOLERANCE
    allow_temporal_approximation: bool = False
    time: float = 0.0
    default_dt: float | None = None
    counters: PerformanceCounters = field(default_factory=PerformanceCounters)
    _base_spatial_atoms: tuple[GaussianChirpAtom, ...] = ()
    report: RenderReport | None = field(default=None, init=False)

    @property
    def prepared_plan(self) -> OceanAnalyticPlan:
        return self.plan

    @property
    def temporal_closed(self) -> bool:
        return bool(self.plan.temporal_closed)

    @property
    def representation(self) -> str:
        return self.plan.representation

    @property
    def spacing(self) -> tuple[float, float]:
        """Return spacing in ocean array order ``(row/y, column/x)``.

        ``CoordinateChart`` stores coordinates in ``(x, y)`` order, while
        the ocean display and layout APIs expose a raster as
        ``(rows, columns)``.  Keeping this conversion at the property avoids
        a caller accidentally comparing chart spacing to raster spacing.
        """

        if self.layout is not None:
            return (float(self.layout.spacing[0]), float(self.layout.spacing[1]))
        if self.chart is None or self.chart.dimension != 2:
            return (1.0, 1.0)
        return (float(self.chart.sample_spacing[1]), float(self.chart.sample_spacing[0]))

    def _layout_chunk(self, layout: SampleLayout, start: int, stop: int) -> np.ndarray:
        rows, columns = layout.shape
        linear = np.arange(start, stop, dtype=np.int64)
        if layout.order == "F":
            row = linear % rows
            column = linear // rows
        else:
            row = linear // columns
            column = linear % columns
        x = layout.origin[1] + layout.spacing[1] * column
        y = layout.origin[0] + layout.spacing[0] * row
        return np.column_stack((x, y)).astype(np.float64, copy=False)

    def _evaluate_atoms(
        self, points_xy: np.ndarray, *, allow_layout_gradient: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        if self.atom_source is None:
            zeros = np.zeros(points_xy.shape[0], dtype=np.complex128)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy(), False
        coordinates = _field_points(points_xy, self.chart)
        values = np.zeros(points_xy.shape[0], dtype=np.complex128)
        gradient_x = np.zeros_like(values)
        gradient_y = np.zeros_like(values)
        for atom in self.atom_source.atoms:
            atom_values = np.asarray(atom.evaluate(coordinates, counters=self.counters), dtype=np.complex128).reshape(
                -1
            )
            factor = np.asarray(atom.h)[None, :] - coordinates @ np.asarray(atom.C).T
            values += atom_values
            gradient_x += atom_values * factor[:, 0]
            gradient_y += atom_values * factor[:, 1]
        residual, residual_x, residual_y, unknown = _evaluate_residual(
            self.atom_source.residual, coordinates, allow_layout_gradient=allow_layout_gradient
        )
        values += residual
        gradient_x += residual_x
        gradient_y += residual_y
        return values, gradient_x, gradient_y, np.abs(residual), unknown

    def _evaluate_sparse(self, points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        values = np.zeros(points_xy.shape[0], dtype=np.complex128)
        gradient_x = np.zeros_like(values)
        gradient_y = np.zeros_like(values)
        for mode in self.sparse_modes:
            phase, derivative_x, derivative_y = _sparse_phase_geometry(
                mode, self.plan.source_shape or self.shape or (1, 1), points_xy, self.chart
            )
            term = mode.pair_weight * mode.current() * np.exp(1j * phase)
            if self.footprint_covariance is not None:
                covariance = self.footprint_covariance
                # The Fourier transform of a normalized Gaussian footprint.
                physical_gradient = np.column_stack((derivative_x, derivative_y))
                attenuation = np.exp(
                    -0.5 * np.einsum("...i,ij,...j->...", physical_gradient, covariance, physical_gradient)
                )
                term *= attenuation
            values += term
            gradient_x += 1j * derivative_x * term
            gradient_y += 1j * derivative_y * term
        zeros = np.zeros(points_xy.shape[0], dtype=np.float64)
        return values, gradient_x, gradient_y, zeros, False

    def _evaluate_jet(
        self, points_xy: np.ndarray, *, allow_layout_gradient: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        if self.jet is None:
            zeros = np.zeros(points_xy.shape[0], dtype=np.complex128)
            return zeros, zeros.copy(), zeros.copy(), zeros.copy(), False
        coordinates = _field_points(points_xy, self.chart)
        values, gradient_x, gradient_y = _jet_footprint(self.jet, coordinates, self.footprint_covariance)
        residual, residual_x, residual_y, unknown = _evaluate_residual(
            self.jet_residual, coordinates, allow_layout_gradient=allow_layout_gradient
        )
        values += residual
        gradient_x += residual_x
        gradient_y += residual_y
        return values, gradient_x, gradient_y, np.abs(residual), unknown

    def _evaluate_chunk(
        self, points_xy: np.ndarray, *, allow_layout_gradient: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, np.ndarray | None]:
        if self.representation in {"atom", "mixture", "gaussian_chirp_atom", "atom_mixture"}:
            values, gradient_x, gradient_y, residual, unknown = self._evaluate_atoms(
                points_xy, allow_layout_gradient=allow_layout_gradient
            )
        elif self.representation in {"jet", "quadratic_jet"}:
            values, gradient_x, gradient_y, residual, unknown = self._evaluate_jet(
                points_xy, allow_layout_gradient=allow_layout_gradient
            )
        elif self.representation in {"sparse", "sparse_fourier", "hermitian_sparse", "sparse_block"}:
            values, gradient_x, gradient_y, residual, unknown = self._evaluate_sparse(points_xy)
        else:
            raise AnalyticOceanError(f"unsupported analytic ocean representation {self.representation!r}")
        support_mask = _region_mask(_chart_coordinates(points_xy, self.chart), self.support)
        if support_mask is not None:
            values = np.where(support_mask, values, 0.0)
            gradient_x = np.where(support_mask, gradient_x, 0.0)
            gradient_y = np.where(support_mask, gradient_y, 0.0)
            residual = np.where(support_mask, residual, 0.0)
        if self.output_format in {"real", "native", "height", "signed_real"}:
            imag_leak = np.maximum(
                np.maximum(np.abs(values.imag), np.abs(gradient_x.imag)),
                np.abs(gradient_y.imag),
            )
            return values.real, gradient_x.real, gradient_y.real, residual, unknown, imag_leak
        return values, gradient_x, gradient_y, residual, unknown, None

    def _visible_mask(self, output_shape: tuple[int, ...]) -> np.ndarray | None:
        if self.visible is None:
            return None
        mask = np.asarray(self.visible, dtype=bool)
        if mask.shape != output_shape:
            if mask.size != int(math.prod(output_shape)):
                raise AnalyticOceanError("visible mask shape does not match ocean output")
            mask = mask.reshape(output_shape)
        return mask.reshape(-1)

    def _layout_gradient_policy(self) -> str | None:
        """Return the explicit policy used for addressed layout residuals."""

        candidates: list[Any] = []
        if self.atom_source is not None:
            candidates.append(self.atom_source.residual)
        candidates.append(self.jet_residual)
        for candidate in candidates:
            policy = getattr(candidate, "layout_gradient_policy", None)
            if isinstance(policy, str):
                return policy
        return None

    def execute(
        self,
        points: Any = None,
        *,
        layout: SampleLayout | Sequence[int] | Mapping[str, Any] | None = None,
        visible: Any = None,
        footprint: Any = None,
        derivative_axis: int | None = None,
        spacing: Any = None,
        return_report: bool = False,
    ) -> Any:
        """Evaluate height and both gradients at points or a visible tile.

        Point arrays are ordered ``(x, y)``.  A layout uses shape order
        ``(rows, columns)`` and is streamed in bounded chunks; no full source
        grid or FFT intermediate is constructed.
        """

        if footprint is not None and footprint is not self.footprint:
            # A per-call footprint is legal for covariance footprints.  Atom
            # convolution is prepared once and therefore must be supplied at
            # preparation time.
            covariance = _footprint_covariance(footprint)
        else:
            covariance = self.footprint_covariance
        if points is not None:
            flat_points, output_shape = _coerce_points(points)
            sample_layout = None
        else:
            sample_layout = _coerce_layout(self.layout if layout is None else layout, self.shape)
            output_shape = tuple(sample_layout.shape)
            flat_points = None
        count = int(math.prod(output_shape))
        if flat_points is None:
            flat_points = np.empty((0, 2), dtype=np.float64)
        layout_gradient_policy = self._layout_gradient_policy() if points is None else None
        if points is not None and self._layout_gradient_policy() is not None:
            raise AnalyticOceanError(
                "SpectralJetTile2D addressed residuals support layout sampling only; "
                "point queries require a continuous residual evaluator with derivatives"
            )
        dtype = np.dtype(self.output_dtype)
        is_complex = self.output_format in {"complex", "coherent_complex"} or dtype.kind == "c"
        if is_complex and dtype.kind != "c":
            dtype = np.dtype(np.complex128)
        result_dtype = np.complex128 if is_complex else np.float64
        height = np.empty(count, dtype=result_dtype)
        gradient_x = np.empty_like(height)
        gradient_y = np.empty_like(height)
        residual = np.empty(count, dtype=np.float64)
        unknown_gradient = False
        max_imaginary_leakage = 0.0
        masks = self._visible_mask(output_shape) if visible is None else np.asarray(visible, dtype=bool).reshape(-1)
        if masks is not None and masks.size != count:
            raise AnalyticOceanError("visible mask does not match requested output")
        old_covariance = self.footprint_covariance
        old_atoms = None if self.atom_source is None else self.atom_source.atoms
        try:
            if covariance is not old_covariance:
                self.footprint_covariance = covariance
                if (
                    self.atom_source is not None
                    and self.atom_source.atoms
                    and footprint is not None
                    and footprint is not self.footprint
                ):
                    # The prepared atom tuple may already include the
                    # preparation footprint.  A per-call footprint starts
                    # from the current unfiltered spatial frame so filtering
                    # is applied exactly once for this execution.
                    base_atoms = self._base_spatial_atoms or self.atom_source.atoms
                    try:
                        self.atom_source.atoms = _filter_atoms(base_atoms, footprint)
                    except ValueError as exc:
                        raise AnalyticOceanError(f"analytic footprint convolution failed: {exc}") from exc
            step = max(1, int(self.chunk_size))
            for start in range(0, count, step):
                stop = min(count, start + step)
                if points is not None:
                    chunk_points = flat_points[start:stop]
                else:
                    if sample_layout is None:
                        raise AnalyticOceanError("an ocean layout is required when points are omitted")
                    chunk_points = self._layout_chunk(sample_layout, start, stop)
                values, gx, gy, chunk_residual, unknown, leakage = self._evaluate_chunk(
                    chunk_points, allow_layout_gradient=points is None
                )
                if is_complex:
                    height[start:stop] = np.asarray(values, dtype=np.complex128)
                    gradient_x[start:stop] = np.asarray(gx, dtype=np.complex128)
                    gradient_y[start:stop] = np.asarray(gy, dtype=np.complex128)
                else:
                    height[start:stop] = np.asarray(values, dtype=np.float64)
                    gradient_x[start:stop] = np.asarray(gx, dtype=np.float64)
                    gradient_y[start:stop] = np.asarray(gy, dtype=np.float64)
                residual[start:stop] = np.asarray(chunk_residual, dtype=np.float64)
                unknown_gradient = unknown_gradient or unknown
                if leakage is not None and leakage.size:
                    max_imaginary_leakage = max(max_imaginary_leakage, float(np.max(leakage)))
        finally:
            self.footprint_covariance = old_covariance
            if self.atom_source is not None and old_atoms is not None:
                self.atom_source.atoms = old_atoms
        if masks is not None:
            height[~masks] = 0
            gradient_x[~masks] = 0
            gradient_y[~masks] = 0
            residual[~masks] = 0
        if unknown_gradient:
            # An omitted residual derivative is a declared limitation.  NaNs
            # make accidental use visible to the native consumer and report.
            gradient_x[np.abs(residual) > 0.0] = np.nan
            gradient_y[np.abs(residual) > 0.0] = np.nan
        height = height.astype(dtype, copy=False).reshape(output_shape)
        gradient_x = gradient_x.astype(dtype, copy=False).reshape(output_shape)
        gradient_y = gradient_y.astype(dtype, copy=False).reshape(output_shape)
        residual = residual.reshape(output_shape)
        ledger = self.plan.error
        if unknown_gradient:
            ledger = ledger.plus(ErrorLedger(materialization=math.inf))
        if layout_gradient_policy is not None:
            # The finite differences are a declared raster policy.  They are
            # useful for a complete image fallback but do not certify a
            # continuous derivative between addressed samples.
            ledger = ledger.plus(ErrorLedger(display=math.inf))
        report = self._report(
            output_shape,
            ledger,
            max_imaginary_leakage,
            gradient_policy=layout_gradient_policy,
        )
        self.report = report
        frame = OceanAnalyticFrame(
            height,
            gradient_x,
            gradient_y,
            report=report,
            mask=masks.reshape(output_shape) if masks is not None else None,
            residual=residual,
        )
        if derivative_axis is not None:
            axis = int(derivative_axis)
            if axis not in {0, 1, -1, -2}:
                raise AnalyticOceanError("derivative axis must be 0/1 or -2/-1")
            result = frame.gradient_y if axis in {0, -2} else frame.gradient_x
            if spacing is not None:
                spacing_array = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (2,))
                if np.any(~np.isfinite(spacing_array)) or np.any(spacing_array <= 0.0):
                    raise AnalyticOceanError("derivative spacing must be finite and positive")
                result = result / spacing_array[0 if axis in {0, -2} else 1]
            return (result, report) if return_report else result
        return (frame, report) if return_report else frame

    def execute_fields(
        self,
        points: Any = None,
        *,
        layout: SampleLayout | Sequence[int] | Mapping[str, Any] | None = None,
        need_height: bool = True,
        need_gradients: bool = True,
        gradient_axes: Sequence[int] | None = None,
        output_dtype: Any = None,
        visible: Any = None,
        footprint: Any = None,
        return_report: bool = False,
    ) -> OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport]:
        """Vectorized tiled-renderer entry point for physical (x,y) points.

        ``need_height``/``need_gradients`` and ``gradient_axes`` are accepted
        for compatibility with the dense terminal.  The compact consumer
        computes the joint fields in one pass; callers selecting a subset can
        ignore the extra arrays on the returned frame.
        """

        del need_height, need_gradients, gradient_axes
        if output_dtype is None:
            return cast(
                OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport],
                self.execute_physical(points, visible=visible, footprint=footprint, return_report=return_report)
                if points is not None
                else self.execute(layout=layout, visible=visible, footprint=footprint, return_report=return_report),
            )
        original_dtype = self.output_dtype
        self.output_dtype = np.dtype(output_dtype)
        try:
            result = (
                self.execute_physical(points, visible=visible, footprint=footprint, return_report=return_report)
                if points is not None
                else self.execute(layout=layout, visible=visible, footprint=footprint, return_report=return_report)
            )
        finally:
            self.output_dtype = original_dtype
        return cast(OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport], result)

    def ocean_fields(self, points: Any, *, gradient_axes: Sequence[int] | None = None) -> tuple[np.ndarray, ...]:
        """Compatibility producer for the terminal analytic carrier.

        The historical terminal passes coordinates as ``(row=y, column=x)``
        and expects ``(height, grad_y, grad_x)``.  The public tiled API above
        uses physical ``(x, y)`` points and named gradients.
        """

        array = np.asarray(points, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 2:
            raise AnalyticOceanError("ocean_fields points must have trailing shape (y, x)")
        xy = array[..., ::-1]
        frame = cast(OceanAnalyticFrame, self.execute_physical(xy))
        axes = (0, 1) if gradient_axes is None else tuple(int(axis) % 2 for axis in gradient_axes)
        outputs: list[np.ndarray] = [frame.height]
        for axis in axes:
            outputs.append(frame.gradient_y if axis == 0 else frame.gradient_x)
        return tuple(outputs)

    def execute_physical(
        self,
        points: Any,
        *,
        visible: Any = None,
        footprint: Any = None,
        return_report: bool = False,
    ) -> OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport]:
        """Evaluate physical (x,y) points, accounting for the coordinate chart."""

        flat, output_shape = _coerce_points(points)
        matrix, origin = _chart_matrix(self.chart)
        try:
            chart_points = (flat - origin) @ np.linalg.inv(matrix).T
        except np.linalg.LinAlgError as exc:
            raise AnalyticOceanError("coordinate chart basis must be invertible for physical queries") from exc
        result = self.execute(
            chart_points.reshape((*output_shape, 2)), visible=visible, footprint=footprint, return_report=return_report
        )
        return cast(OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport], result)

    def execute_tile(
        self,
        layout: SampleLayout | Sequence[int] | Mapping[str, Any],
        *,
        visible: Any = None,
        footprint: Any = None,
        return_report: bool = False,
    ) -> OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport]:
        """Evaluate one streamed visible tile/layout."""

        return cast(
            OceanAnalyticFrame | tuple[OceanAnalyticFrame, RenderReport],
            self.execute(layout=layout, visible=visible, footprint=footprint, return_report=return_report),
        )

    render_tile = execute_tile

    def sample_point(
        self, x: float, y: float, *, need_gradients: bool = True
    ) -> tuple[float | complex, float | complex, float | complex]:
        """Evaluate one point in the native callback order ``(h, dy, dx)``."""

        frame = cast(
            OceanAnalyticFrame,
            self.execute_physical(np.asarray([[float(x), float(y)]], dtype=np.float64)),
        )
        if need_gradients:
            return (
                frame.height.reshape(-1)[0].item(),
                frame.gradient_y.reshape(-1)[0].item(),
                frame.gradient_x.reshape(-1)[0].item(),
            )
        return frame.height.reshape(-1)[0].item(), 0.0, 0.0

    sample = sample_point

    def native_sample(
        self, y: float, x: float, *, need_gradients: bool = True, footprint: Any = None
    ) -> tuple[float | complex, float | complex, float | complex]:
        """Native callback spelling: arguments ``(y, x)`, result ``(h, dy, dx)``."""

        if footprint is None:
            return self.sample_point(float(x), float(y), need_gradients=need_gradients)
        frame = cast(
            OceanAnalyticFrame,
            self.execute_physical(
                np.asarray([[float(x), float(y)]], dtype=np.float64),
                footprint=footprint,
            ),
        )
        if need_gradients:
            return (
                frame.height.reshape(-1)[0].item(),
                frame.gradient_y.reshape(-1)[0].item(),
                frame.gradient_x.reshape(-1)[0].item(),
            )
        return frame.height.reshape(-1)[0].item(), 0.0, 0.0

    def native_sampler(self, *, footprint: Any = None, cache_capacity: int = 4096) -> Any:
        """Return a bounded no-grid callback adapter for the native display.

        The adapter delegates every compact-source query to this prepared
        consumer, preserving its chart, support, footprint, and residual
        policy.  It is deliberately separate from the packed SPF kernel: an
        atom/jet source has no legal packed-plane representation to hand to
        that kernel without changing the source semantics.
        """

        from .ocean_analytic_native import AnalyticNativeSampler

        return AnalyticNativeSampler(self, footprint=footprint, cache_capacity=cache_capacity)

    @property
    def native_queries_supported(self) -> bool:
        """Whether this compact representation has a point callback path."""

        if not self.plan.compact_source or self.representation not in {
            "atom",
            "mixture",
            "gaussian_chirp_atom",
            "atom_mixture",
            "jet",
            "quadratic_jet",
        }:
            return False
        if self.support is not None:
            support_kind = str(getattr(self.support, "kind", "")).upper()
            if not isinstance(self.support, Region) and not (
                support_kind.endswith("FULL")
                or support_kind.endswith("EMPTY")
                or support_kind.endswith("AABB")
                or support_kind.endswith("UNKNOWN")
                or support_kind.endswith("FREQUENCY_BAND")
            ):
                return False
        if not math.isfinite(float(self.plan.temporal_residual_bound)):
            return False
        if self.output_format not in {"real", "native", "height", "signed_real"} or (
            np.dtype(self.output_dtype).kind == "c"
        ):
            return False
        if (
            self.representation in {"jet", "quadratic_jet"}
            and self.jet is not None
            and str(self.jet.value_kind).lower() not in {"native", "real", "complex"}
        ):
            return False
        if self.representation in {"atom", "mixture", "gaussian_chirp_atom", "atom_mixture"} and (
            self.atom_source is None or not self.atom_source.atoms
        ):
            return False
        return (self.atom_source is None or self.atom_source.residual is None) and self.jet_residual is None

    def native_sampling_capability(self) -> Mapping[str, Any]:
        """Describe the analytic callback route without materializing data."""

        support_supported = True
        support_kind = ""
        if self.support is not None:
            support_kind = str(getattr(self.support, "kind", "")).upper()
            support_supported = isinstance(self.support, Region) or support_kind.endswith(
                ("FULL", "EMPTY", "AABB", "UNKNOWN", "FREQUENCY_BAND")
            )
        if (self.atom_source is not None and self.atom_source.residual is not None) or self.jet_residual is not None:
            reason = "residual requires explicit vectorized evaluator"
        elif not support_supported:
            reason = f"support kind {support_kind!r} has no compiled point mask"
        elif not math.isfinite(float(self.plan.temporal_residual_bound)):
            reason = "temporal approximation has no certified finite residual bound"
        elif self.output_format not in {"real", "native", "height", "signed_real"} or (
            np.dtype(self.output_dtype).kind == "c"
        ):
            reason = "native callback emits real height and gradients only"
        elif (
            self.representation in {"jet", "quadratic_jet"}
            and self.jet is not None
            and str(self.jet.value_kind).lower() not in {"native", "real", "complex"}
        ):
            reason = f"Jet2D value kind {self.jet.value_kind!r} needs a typed emitter"
        else:
            reason = (
                "compact analytic source has a direct point consumer"
                if self.native_queries_supported
                else "selected representation has no compact point consumer"
            )
        return MappingProxyType(
            {
                "supported": self.native_queries_supported,
                "representation": self.representation if self.native_queries_supported else None,
                "route": "analytic",
                "coordinates": "physical_yx",
                "gradients": "height,gradient_y,gradient_x",
                "reason": reason,
            }
        )

    def _report(
        self,
        output_shape: tuple[int, ...],
        ledger: ErrorLedger,
        imaginary_leakage: float,
        *,
        gradient_policy: str | None = None,
    ) -> RenderReport:
        output_elements = int(math.prod(output_shape))
        output_dtype = np.dtype(self.output_dtype)
        terminal_output_elements = output_elements * 3
        terminal_output_bytes = int(terminal_output_elements * output_dtype.itemsize)
        # ``execute`` owns the three terminal arrays for the requested output
        # and allocates bounded chunk scratch.  The Python evaluator does not
        # instrument allocator/DRAM traffic, so keep those measurements
        # separate from the structural no-dense-source claim.
        chunk_elements = min(output_elements, max(1, int(self.chunk_size)))
        scratch_bytes = int(
            chunk_elements * (2 * np.dtype(np.float64).itemsize)  # physical point coordinates
            + chunk_elements * 3 * np.dtype(np.complex128).itemsize  # shared value/gradient accumulators
            + chunk_elements * np.dtype(np.float64).itemsize  # residual ledger values
            + chunk_elements * np.dtype(np.bool_).itemsize  # support/visibility mask
        )
        logical_peak_bytes = terminal_output_bytes + scratch_bytes
        metric = RenderMetrics(
            backend="analytic",
            workload="ocean",
            resolution=tuple(output_shape),
            representation=self.plan.representation,
            precision=str(self.output_dtype),
            error_budget=ledger.total_bound,
            # Source reads and allocator peak are intentionally unmeasured at
            # this Python boundary.  ``diagnostics`` carries bounded logical
            # scratch information without presenting it as DRAM telemetry.
            bytes_read=0,
            bytes_written=terminal_output_bytes,
            peak_memory=0,
            potential_elements=terminal_output_elements,
            materialized_elements=terminal_output_elements,
            atoms_considered=self.plan.source_count
            if "atom" in self.plan.representation or self.plan.representation == "jet"
            else 0,
            complex_adds=max(0, self.plan.source_count - 1) * output_elements,
            pivot_reductions=output_elements if self.plan.source_count else 0,
            fft_count=0,
            field_operations=1,
            reduction_operations=output_elements,
            temporal_error=self.plan.temporal_residual_bound,
            potential_intermediate_elements=0,
            materialized_intermediate_elements=0,
            terminal_output_elements=terminal_output_elements,
            potential_intermediate_bytes=0,
            materialized_intermediate_bytes=0,
            terminal_output_bytes=terminal_output_bytes,
            logical_peak_bytes=logical_peak_bytes,
        )
        diagnostics = {
            "source_count": self.plan.source_count,
            "source_shape": self.plan.source_shape,
            "compact_source": self.plan.compact_source,
            "dense_source_materializations": 0,
            "fft_count": 0,
            "gradient_coordinates": "physical",
            "imaginary_leakage": imaginary_leakage,
            "temporal_closed": self.plan.temporal_closed,
            "temporal_residual_bound": self.plan.temporal_residual_bound,
            "native_sampler_order": "height, gradient_y, gradient_x",
            "source_materialization": "none",
            "source_read_provenance": "unmeasured",
            "allocator_peak_provenance": "unmeasured",
            "logical_peak_provenance": "bounded_chunk_estimate",
            "logical_peak_scratch_bytes": scratch_bytes,
            "error_bound_semantics": "representation_and_declared_approximation_only",
            "numerical_evaluation_error": "not_certified",
            "gradient_policy": gradient_policy or "analytic_source_gradient",
            "gradient_error_status": "not_certified" if gradient_policy is not None else "analytic",
        }
        return RenderReport(metric, ledger=ledger, capabilities_used=self.plan.capabilities, diagnostics=diagnostics)

    def advance(self, steps: int = 1, dt: float | None = None) -> PreparedAnalyticOcean:
        count = int(steps)
        if count < 0:
            raise ValueError("steps must be non-negative")
        delta = self.default_dt if dt is None else float(dt)
        if delta is None:
            delta = 1.0
        if not math.isfinite(delta):
            raise ValueError("dt must be finite")
        if not self.plan.temporal_closed and not self.allow_temporal_approximation:
            raise AnalyticOceanError(
                "non-quadratic dispersion has no exact Gaussian closure; supply allow_temporal_approximation=True "
                "to continue with the recorded residual"
            )
        self.time += count * delta
        for mode in self.sparse_modes:
            mode.advance(count, dt=delta)
        if self.atom_source is not None and self.atom_source.base_fourier_atoms:
            spatial, _ = _apply_dispersion_to_atoms(
                self.atom_source.base_fourier_atoms,
                self.atom_source.source_domain,
                self.atom_source.temporal,
                self.time,
            )
            if self.footprint is not None:
                try:
                    spatial = _filter_atoms(spatial, self.footprint)
                except ValueError as exc:
                    raise AnalyticOceanError(f"analytic footprint convolution failed during advance: {exc}") from exc
            self.atom_source.atoms = tuple(spatial)
            temporal = self.atom_source.temporal
            if temporal is not None and temporal.kind == "nonquadratic":
                bound = 0.0 if self.time == 0.0 else abs(self.time) * temporal.residual_bound_per_time
                self.plan = OceanAnalyticPlan(
                    representation=self.plan.representation,
                    source_count=self.plan.source_count,
                    source_shape=self.plan.source_shape,
                    compact_source=self.plan.compact_source,
                    normalization=self.plan.normalization,
                    dtype=self.plan.dtype,
                    temporal_closed=self.plan.temporal_closed,
                    temporal_residual_bound=bound,
                    error=ErrorLedger(
                        source_quantization=self.plan.error.source_quantization,
                        representation_fit=self.plan.error.representation_fit,
                        pruning=self.plan.error.pruning,
                        reducer=self.plan.error.reducer,
                        transform=bound,
                        materialization=self.plan.error.materialization,
                        display=self.plan.error.display,
                        stages=self.plan.error.stages,
                    ),
                    source_domain=self.plan.source_domain,
                    capabilities=self.plan.capabilities,
                    metadata=self.plan.metadata,
                )
        return self

    def sync_phases(self, modes: Sequence[Any]) -> None:
        """Copy owned sparse phase state back at an explicit mutable boundary."""

        owned: dict[tuple[int, int], _SparseMode] = {
            mode.index: mode for mode in self.sparse_modes if mode.external is not None
        }
        for target in modes:
            target_index = tuple(int(value) for value in target.index)
            if len(target_index) != 2:
                continue
            source = owned.get((target_index[0], target_index[1]))
            if source is None:
                continue
            source_accumulator = getattr(source.external, "phase_accumulator", None)
            target_accumulator = getattr(target, "phase_accumulator", None)
            if source_accumulator is None or target_accumulator is None:
                continue
            target_accumulator.phase = int(source_accumulator.phase)

    step = advance
    render = execute


AnalyticOceanPrepared = PreparedAnalyticOcean
OceanFieldResult = OceanAnalyticFrame


def prepare_analytic_ocean(
    source: Any = None,
    shape: Sequence[int] | None = None,
    *positional: Any,
    coefficients: Any = None,
    indices: Any = None,
    phases: Any = None,
    phase_increments: Any = None,
    phase_bits: int = 16,
    normalize: bool | None = None,
    source_domain: str | Domain | None = None,
    domain: str | Domain | None = None,
    chart: CoordinateChart | None = None,
    layout: SampleLayout | Sequence[int] | Mapping[str, Any] | None = None,
    support: Support | Region | None = None,
    visible: Any = None,
    footprint: Any = None,
    spacing: Any = None,
    time: float = 0.0,
    dt: float | None = None,
    dispersion: Any = None,
    dispersion_support: Any = None,
    allow_temporal_approximation: bool = False,
    max_sparse_modes: int | None = 65_536,
    output_dtype: Any = np.float64,
    output_format: str = "real",
    chunk_size: int = _DEFAULT_CHUNK,
    tolerance: float = _DEFAULT_HERMITIAN_TOLERANCE,
    strict: bool = True,
    **_: Any,
) -> PreparedAnalyticOcean:
    """Prepare a compact analytic ocean source.

    ``indices`` are Fourier addresses ordered ``(row, column)`` and
    ``coefficients`` are complex amplitudes.  A lone address implies its
    Hermitian partner; supplying both requires conjugate consistency.  A
    ``Field`` preserves its chart, support, representation ledger, and
    residual metadata.  Dense arrays are rejected because this consumer is
    explicitly the no-dense/no-FFT path.
    """

    if positional:
        # Accept the legacy ``prepare_ocean(shape, coefficients, indices)``
        # positional spelling when the analytic route is selected explicitly.
        if len(positional) > 2 or coefficients is not None or indices is not None:
            raise TypeError("analytic ocean positional form is (shape, coefficients, indices[, phases])")
        legacy_shape, legacy_coefficients = source, shape
        source, shape, coefficients, indices = None, legacy_shape, legacy_coefficients, positional[0]
        if len(positional) == 2:
            phases = positional[1]
    payload, field_domain, field_chart, field_support, field_ledger, raw_metadata = _read_field_source(source)
    metadata: dict[str, Any] = dict(raw_metadata)
    if normalize is None:
        normalize = bool(getattr(payload, "normalize", True))
    selected_source_domain = source_domain if source_domain is not None else domain
    source_spacing = getattr(payload, "spacing", None)
    if chart is None and field_chart is None and source_spacing is not None and len(tuple(source_spacing)) == 2:
        # OceanSpectrum stores spacing in source-array order (row, column),
        # while the public chart uses (x, y).
        selected_chart = CoordinateChart.identity(2, spacing=(float(source_spacing[1]), float(source_spacing[0])))
    else:
        selected_chart = chart or field_chart or CoordinateChart.identity(2)
    if selected_chart.dimension == 0:
        # ``Field.from_value(Jet2D(...))`` predates Jet2D dimension inference
        # and may carry a zero-dimensional identity chart.  The analytic
        # ocean contract is explicitly two-dimensional, so repair only this
        # unambiguous legacy default at ingress.
        selected_chart = CoordinateChart.identity(2)
    if selected_chart.dimension != 2:
        raise AnalyticOceanError("analytic ocean sources require a two-dimensional coordinate chart")
    if spacing is not None:
        spacing_array = np.asarray(spacing, dtype=np.float64).reshape(-1)
        if spacing_array.size != 2 or not np.all(np.isfinite(spacing_array)) or np.any(spacing_array <= 0.0):
            raise AnalyticOceanError("ocean spacing must contain two finite positive values")
        # Public ocean spacing follows raster order (row/y, column/x).
        selected_chart = CoordinateChart(
            origin=selected_chart.origin,
            basis=selected_chart.basis,
            scale=selected_chart.scale,
            periodicity=selected_chart.periodicity,
            sample_spacing=(float(spacing_array[1]), float(spacing_array[0])),
            frequency_convention=selected_chart.frequency_convention,
        )
    selected_support = support if support is not None else field_support
    footprint = _owned_footprint(footprint)
    selected_shape = _coerce_shape_2d(shape)
    if layout is not None:
        layout_object = _coerce_layout(layout, selected_shape)
        if selected_shape is not None and tuple(layout_object.shape) != selected_shape:
            raise AnalyticOceanError("layout shape and ocean shape must match")
        if selected_shape is None:
            selected_shape = _coerce_shape_2d(layout_object.shape)
    else:
        layout_object = None

    # Explicit sparse arrays take precedence over an OceanSpectrum adapter.
    sparse_payload = payload
    sparse_shape: tuple[int, ...] | None
    if indices is not None or coefficients is not None:
        if indices is None or coefficients is None:
            raise AnalyticOceanError("indices and coefficients must be supplied together")
        sparse_indices, sparse_coefficients, sparse_phases, sparse_increments = (
            indices,
            coefficients,
            phases,
            phase_increments,
        )
        source_metadata: Mapping[str, Any] = {}
        if selected_shape is None:
            raise AnalyticOceanError("sparse indices require shape")
        sparse_shape = selected_shape
    else:
        sparse_indices, sparse_coefficients, sparse_phases, sparse_increments, sparse_shape, source_metadata = (
            _extract_sparse_source(payload, selected_shape)
        )
        if sparse_shape is not None and selected_shape is None:
            selected_shape = _coerce_shape_2d(sparse_shape)

    # Addressed blocks are compact only when their active address set is
    # bounded.  The block view is sampled at those addresses and is never
    # passed through complex_carrier/reconstruct/FFT.
    block = payload if isinstance(payload, BlockView) else as_block_view(payload)
    jet_residual = None
    modes: tuple[_SparseMode, ...] = ()
    base_spatial: tuple[GaussianChirpAtom, ...] = ()
    if block is not None:
        if selected_shape is None:
            block_shape = _coerce_shape_2d(block.shape)
            if block_shape is None:
                raise AnalyticOceanError("addressed ocean blocks must be two-dimensional")
            selected_shape = block_shape
        if selected_shape is None:
            raise AnalyticOceanError("addressed ocean blocks require a two-dimensional shape")
        modes = _block_sparse_modes(
            block, selected_shape, normalize=normalize, max_sparse_modes=max_sparse_modes, tolerance=tolerance
        )
        representation = "sparse_block"
        source_domain = str(Domain.FOURIER_2D)
        source_count = len(modes)
        atom_source = None
        jet = None
        base_spatial = ()
        temporal_spec = (
            _quadratic_dispersion(dispersion, support=dispersion_support) if dispersion is not None else None
        )
    elif sparse_indices is not None and sparse_coefficients is not None:
        if sparse_shape is None:
            raise AnalyticOceanError("sparse ocean source requires shape")
        selected_shape = _coerce_shape_2d(sparse_shape)
        if selected_shape is None:
            raise AnalyticOceanError("sparse ocean source shape must be two-dimensional")
        modes = _sparse_modes_from_arrays(
            sparse_indices,
            sparse_coefficients,
            selected_shape,
            phases=sparse_phases,
            phase_increments=sparse_increments,
            phase_bits=phase_bits,
            normalize=normalize,
            tolerance=tolerance,
        )
        # Preserve WidePhaseAccumulator/OceanMode objects when an adapter
        # supplied them.  The external accumulator remains the source of truth
        # for subsequent calls to advance().
        external_modes = source_metadata.get("mode_objects", ())
        if external_modes:
            external_by_index: dict[tuple[int, int], Any] = {}
            for item in external_modes:
                clone = copy.copy(item)
                accumulator = getattr(item, "phase_accumulator", None)
                if accumulator is not None and hasattr(accumulator, "copy"):
                    clone.phase = accumulator.copy()
                raw_index = tuple(int(value) for value in item.index)
                if len(raw_index) != 2:
                    raise AnalyticOceanError("sparse ocean mode indices must be two-dimensional")
                external_by_index[(raw_index[0], raw_index[1])] = clone
            for mode in modes:
                external = external_by_index.get(mode.index)
                if external is None:
                    external = external_by_index.get(
                        ((-mode.index[0]) % selected_shape[0], (-mode.index[1]) % selected_shape[1])
                    )
                mode.external = external
        representation = "sparse"
        source_domain = str(Domain.FOURIER_2D)
        source_count = len(modes)
        atom_source = None
        jet = None
        base_spatial = ()
        temporal_spec = (
            _quadratic_dispersion(dispersion, support=dispersion_support) if dispersion is not None else None
        )
    elif _is_spectral_jet_tile(payload) or isinstance(payload, _SpectralJetResidualSource):
        tile, field_residual = _spectral_jet_source(payload)
        tile_shape = _coerce_shape_2d(tile.shape)
        if tile_shape is None:
            raise AnalyticOceanError("spectral jet tile shape must be two-dimensional")
        if selected_shape is None:
            selected_shape = tile_shape
        if selected_shape != tile_shape:
            raise AnalyticOceanError("spectral jet tile shape and ocean shape must match")
        if footprint is not None and (
            tile.log_residual is not None or tile.phase_residual is not None or field_residual is not None
        ):
            raise AnalyticOceanError(
                "a spectral jet residual cannot be convolved by the compact footprint route; "
                "use an explicit residual-filter consumer"
            )
        spectral_atom = _spectral_jet_atom(tile)
        filtered_atom = spectral_atom
        if footprint is not None:
            try:
                filtered_atom = _filter_atoms((spectral_atom,), footprint)[0]
            except ValueError as exc:
                raise AnalyticOceanError(f"spectral jet footprint convolution failed: {exc}") from exc
        residual_active = any(
            residual is not None and float(getattr(residual, "max_abs_value", 0.0)) > 0.0
            for residual in (tile.log_residual, tile.phase_residual)
        )
        zero_mask_active = tile.zero_mask is not None and bool(np.any(tile.zero_mask))
        residual_evaluator: Callable[[np.ndarray], np.ndarray] | None = (
            _spectral_jet_residual(tile, spectral_atom, field_residual)
            if residual_active or zero_mask_active or field_residual is not None
            else None
        )
        representation = "atom"
        source_count = 1
        source_domain = str(Domain.SPATIAL_2D)
        atom_source = _AtomSource((filtered_atom,), residual_evaluator, source_domain)
        jet = None
        modes = ()
        temporal_spec = None
        base_spatial = (spectral_atom,)
        tile_diagnostics = getattr(tile, "diagnostics", None)
        if tile_diagnostics is not None:
            dropped_bound = float(getattr(tile_diagnostics, "max_complex_error", 0.0))
        else:
            dropped_bound = 0.0
        metadata.update(
            {
                "spectral_jet_tile": True,
                "tile_shape": tuple(int(value) for value in tile.shape),
                "tile_origin": tuple(float(value) for value in tile.origin),
                "tile_log_guard_bits": int(tile.log_magnitude.guard_L),
                "tile_phase_guard_bits": int(tile.phase.guard_T),
                "tile_log_residual": tile.log_residual is not None,
                "tile_phase_residual": tile.phase_residual is not None,
                "tile_zero_mask": zero_mask_active,
                "tile_residual_route": "addressed_vectorized_fallback" if residual_evaluator is not None else "none",
            }
        )
        if selected_support is None:
            selected_support = Support.aabb(
                (float(tile.origin[0]), float(tile.origin[1])),
                (
                    float(tile.origin[0]) + int(tile.shape[1]) - 1.0,
                    float(tile.origin[1]) + int(tile.shape[0]) - 1.0,
                ),
            )
        if layout_object is None:
            # Tile origins are source coordinates.  Preserve them for the
            # default raster fallback so a nonzero-origin tile is not queried
            # at an unrelated zero-origin plane.
            layout_object = SampleLayout(
                tile_shape,
                origin=(float(tile.origin[1]), float(tile.origin[0])),
            )
    elif isinstance(payload, (Jet2D, _JetResidualSource)):
        if selected_shape is None and layout_object is not None:
            selected_shape = _coerce_shape_2d(layout_object.shape)
        representation = "jet"
        source_count = 1
        source_domain = str(Domain.SPATIAL_2D)
        jet = payload.jet if isinstance(payload, _JetResidualSource) else payload
        jet_residual = payload.residual if isinstance(payload, _JetResidualSource) else None
        atom_source = None
        modes = ()
        base_spatial = ()
        temporal_spec = None
    else:
        atoms, residual, inferred_domain = _source_atoms(payload)
        if not atoms and not isinstance(payload, AtomMixture):
            if isinstance(payload, np.ndarray) or (hasattr(payload, "shape") and not isinstance(payload, (str, bytes))):
                raise AnalyticOceanError(
                    "dense ocean arrays belong to the dense terminal; analytic preparation requires atoms, "
                    "jets, or sparse addresses"
                )
            raise AnalyticOceanError(f"unsupported analytic ocean source {type(payload).__name__}")
        source_domain = str(selected_source_domain or field_domain or inferred_domain)
        if source_domain in {"", "Domain.CUSTOM"}:
            source_domain = str(Domain.SPATIAL_2D)
        if source_domain not in {str(Domain.SPATIAL_2D), str(Domain.FOURIER_2D), "SPATIAL_2D", "FOURIER_2D"}:
            raise AnalyticOceanError(
                f"analytic ocean source domain must be SPATIAL_2D or FOURIER_2D, got {source_domain!r}"
            )
        temporal_spec = (
            _quadratic_dispersion(dispersion, support=dispersion_support) if dispersion is not None else None
        )
        if (
            temporal_spec is not None
            and temporal_spec.kind == "nonquadratic"
            and time != 0.0
            and not allow_temporal_approximation
        ):
            raise AnalyticOceanError(
                "non-quadratic dispersion breaks Gaussian closure; provide a bounded dispersion support and "
                "allow_temporal_approximation=True to use the local quadratic with residual"
            )
        unfiltered_spatial_atoms, fourier_atoms = _apply_dispersion_to_atoms(
            tuple(atoms), source_domain, temporal_spec, float(time)
        )
        spatial_atoms = unfiltered_spatial_atoms
        if footprint is not None:
            try:
                spatial_atoms = _filter_atoms(spatial_atoms, footprint)
            except ValueError as exc:
                raise AnalyticOceanError(f"Gaussian footprint requires integrable source atoms: {exc}") from exc
        if isinstance(payload, AtomMixture):
            residual = payload.residual
            # Residual and pruning uncertainty already entered through the
            # complete source ledger, for both Field and raw mixture ingress.
            dropped_bound = 0.0
        else:
            dropped_bound = 0.0
        atom_source = _AtomSource(tuple(spatial_atoms), residual, source_domain, tuple(fourier_atoms), temporal_spec)
        representation = "atom" if len(spatial_atoms) == 1 else "mixture"
        source_count = len(spatial_atoms)
        jet = None
        modes = ()
        # Keep the unfiltered frame so a later per-call footprint can be
        # applied once, and so temporal advance can refresh the native
        # context from the current source parameters.
        base_spatial = tuple(unfiltered_spatial_atoms)
    if modes and temporal_spec is not None:
        _configure_sparse_temporal(modes, selected_shape or (1, 1), temporal_spec, float(time))
    temporal_closed = temporal_spec is None or temporal_spec.kind == "quadratic" or representation.startswith("sparse")
    temporal_bound = 0.0
    if (
        temporal_spec is not None
        and temporal_spec.kind == "nonquadratic"
        and not representation.startswith("sparse")
        and float(time) != 0.0
    ):
        temporal_bound = abs(float(time)) * temporal_spec.residual_bound_per_time
    dropped_bound = float(locals().get("dropped_bound", 0.0))
    ledger = field_ledger.plus(ErrorLedger(representation_fit=dropped_bound, transform=temporal_bound))
    covariance = _footprint_covariance(footprint)
    plan = OceanAnalyticPlan(
        representation=representation,
        source_count=source_count,
        source_shape=selected_shape,
        compact_source=True,
        normalization=1.0 / math.prod(selected_shape)
        if normalize and selected_shape is not None and representation.startswith("sparse")
        else 1.0,
        dtype=np.dtype(output_dtype).name,
        temporal_closed=temporal_closed,
        temporal_residual_bound=temporal_bound,
        error=ledger,
        source_domain=source_domain,
        capabilities=(
            "direct_points",
            "joint_height_gradients",
            "no_dense_fft",
            "footprint_sampling",
            "chart_physical_gradients",
            *(() if not metadata.get("spectral_jet_tile") else ("spectral_jet_to_atom",)),
        ),
        metadata={
            **metadata,
            "source_type": type(payload).__name__,
            "strict": strict,
            "temporal_residual_units": "absolute_value_bound_or_phase_radians",
            "temporal_residual_certified": temporal_spec is None or temporal_spec.residual_certified,
            "sampled_dispersion_residual_is_diagnostic": True,
        },
    )
    prepared = PreparedAnalyticOcean(
        plan=plan,
        shape=selected_shape,
        chart=selected_chart,
        layout=layout_object,
        source=payload,
        atom_source=atom_source,
        jet=jet,
        jet_residual=locals().get("jet_residual", None),
        sparse_modes=tuple(modes),
        support=selected_support,
        visible=visible,
        footprint=footprint,
        footprint_covariance=covariance,
        output_dtype=np.dtype(output_dtype),
        output_format=str(output_format).lower().replace("-", "_"),
        chunk_size=max(1, int(chunk_size)),
        tolerance=float(tolerance),
        allow_temporal_approximation=bool(allow_temporal_approximation),
        time=float(time),
        default_dt=dt,
        _base_spatial_atoms=base_spatial,
    )
    return prepared


prepare_ocean_analytic = prepare_analytic_ocean
prepare_ocean = prepare_analytic_ocean


__all__ = [
    "AnalyticOceanError",
    "AnalyticOceanPrepared",
    "OceanAnalyticFrame",
    "OceanAnalyticPlan",
    "OceanFieldResult",
    "PreparedAnalyticOcean",
    "prepare_analytic_ocean",
    "prepare_ocean",
    "prepare_ocean_analytic",
]
