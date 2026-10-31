"""Typed optical-depth and ordered transport paths for SFR v0.2.

This module keeps volume state in the two algebras used by the renderer
design: non-negative optical depth for attenuation and an ordered affine
transport map ``x -> C + T*x`` for compositing.  Density sources can remain
analytic (Gaussian atoms), separable, or dense until a ray actually requests
samples.  All culling and early termination is accompanied by a conservative
bound in the returned :class:`~spf.renderer.report.RenderReport`.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.renderer.backends import SFRBackend, get_backend
from spf.renderer.carriers import decode_words, encode_words
from spf.renderer.model import Field, Region
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport
from spf.tensor import v02

__all__ = [
    "BrickCullCertificate",
    "BrickCullResult",
    "DenseDensity",
    "DensityField",
    "DensityKind",
    "OpticalDepth",
    "OpticalDepthSegment",
    "QuadratureResult",
    "RGBTransport",
    "TransportRGB",
    "VolumeBrick",
    "VolumeSegment",
    "constant_density_closed_form",
    "constant_density_reference",
    "constant_density_segment",
    "constant_density_transport",
    "cull_bricks",
    "cull_bricks_result",
    "early_terminate_transport",
    "heterogeneous_segments",
    "heterogeneous_volume_reference",
    "integrate_density",
    "make_density_segment",
    "ordered_transport_reduce",
    "ordered_transport_scan",
    "quadrature_optical_depth",
    "reduce_transport_rgb",
    "render_constant_density",
    "render_heterogeneous",
    "render_volume",
    "render_volume_planes",
    "sample_atom_density",
    "sample_dense_density",
    "sample_density",
    "sample_separable_density",
    "transport_compose_rgb",
    "transport_identity",
    "transport_scan",
    "transport_tree",
]


def _finite_nonnegative(name: str, value: Any, *, allow_inf: bool = False) -> float:
    number = float(value)
    if math.isnan(number) or number < 0.0 or (not allow_inf and not math.isfinite(number)):
        suffix = " finite" if not allow_inf else ""
        raise ValueError(f"{name} must be a non-negative{suffix} value")
    return number


def _normalise_point(value: Any, *, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.size == 0 or not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must be a finite non-empty point")
    return point


def _ray_points(start: Any, end: Any) -> tuple[np.ndarray, np.ndarray, float]:
    left = _normalise_point(start, name="ray start")
    right = _normalise_point(end, name="ray end")
    if left.shape != right.shape:
        raise ValueError("ray start and end must have the same dimension")
    delta = right - left
    length = float(np.linalg.norm(delta))
    if not math.isfinite(length) or length < 0.0:
        raise ValueError("ray length must be finite and non-negative")
    return left, right, length


@dataclass(frozen=True, slots=True)
class OpticalDepth:
    """A typed non-negative optical-depth scalar ``tau``."""

    value: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_nonnegative("optical depth", self.value, allow_inf=True))

    @property
    def tau(self) -> float:
        return self.value

    @property
    def transmittance(self) -> float:
        return 0.0 if math.isinf(self.value) else float(math.exp(-self.value))

    @property
    def log_transmittance(self) -> v02.LogPos:
        if math.isinf(self.value):
            return v02.LogPos(v02.SPF32_L_MIN_FINITE)
        if self.value == 0.0:
            return v02.LogPos(0)
        code = round(-self.value * v02.SPF32_L_SCALE / math.log(2.0))
        if code <= v02.SPF32_L_ZERO:
            code = v02.SPF32_L_MIN_FINITE
        return v02.LogPos(code)

    def __float__(self) -> float:
        return self.value

    def __add__(self, other: OpticalDepth | float) -> OpticalDepth:
        return OpticalDepth(self.value + float(other))


class DensityKind(StrEnum):
    """Supported density source families."""

    CONSTANT = "constant"
    ATOM = "atom"
    MIXTURE = "mixture"
    SEPARABLE = "separable"
    DENSE = "dense"
    BLOCK = "block"
    CALLABLE = "callable"


def _decode_scalar(value: Any) -> complex | float:
    if isinstance(value, OpticalDepth):
        return value.value
    if isinstance(value, v02.LogPos):
        return value.value
    if isinstance(value, v02.LogSigned):
        return value.value
    if isinstance(value, v02.SPF32Value):
        return complex(decode_words(value.word))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return complex(decode_words(int(value)))
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return value.item() if isinstance(value, np.generic) else value
    return complex(value)


def _validated_density(values: Any, *, tolerance: float = 1.0e-12) -> np.ndarray:
    raw = np.asarray(values)
    if raw.dtype == object:
        output = np.empty(raw.shape, dtype=np.float64)
        for index in np.ndindex(raw.shape):
            value = _decode_scalar(raw[index])
            if isinstance(value, complex):
                if abs(value.imag) > tolerance:
                    raise ValueError("density samples must be real and non-negative")
                value = value.real
            output[index] = float(value)
    else:
        if np.iscomplexobj(raw):
            if np.any(np.abs(np.imag(raw)) > tolerance):
                raise ValueError("density samples must be real and non-negative")
            raw = np.real(raw)
        try:
            output = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("density samples must be numeric") from exc
    if np.any(~np.isfinite(output)) or np.any(output < 0.0):
        raise ValueError("density samples must be finite and non-negative")
    return output


def _project_block_real(
    block: Any,
    linear: np.ndarray,
    *,
    value_kind: str | None = None,
    tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Project addressed words to real positive samples without rebuilding a plane.

    The density/material consumers use this helper at their own scalar
    boundary.  It deliberately asks the block for phase-bearing words so a
    complex source cannot silently cross into a positive field.  ``LOG_POS``
    blocks are already positive magnitudes; all other sources must have a
    numerically real projection.  V3 SPF16/64/128 words use their registered
    format rather than being narrowed to SPF32.
    """

    addresses = np.asarray(linear, dtype=np.int64)
    positive_observable = value_kind in {"log_pos", "radiance", "incoherent_radiance", "optical_depth", "opacity"}
    words = np.asarray(block.sample_words(addresses, phase=not positive_observable))
    projected = np.asarray(block.project_words(words, addresses))
    if projected.dtype.kind in "iu":
        projected = decode_words(projected)
    if np.iscomplexobj(projected):
        imaginary = np.abs(np.imag(projected))
        # A block is sampled at an explicit real-valued consumer boundary.
        # Never discard a coherent phase here; callers must first request an
        # intensity field, which changes the block observable to a positive
        # one before this helper is reached.
        if np.any(imaginary > tolerance):
            raise ValueError("block density samples must be real; convert coherent data to intensity explicitly")
        projected = np.real(projected)
    return np.asarray(projected, dtype=np.float64)


def _scale_block_residual(residual: Any, factor: float) -> Any:
    """Scale a representation residual through a positive exposure."""

    gain = float(factor)
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError("block residual exposure gain must be finite and non-negative")
    if gain == 0.0:
        return 0.0
    from spf.renderer.normalization import _ResidualSum, _ScaledResidual
    from spf.renderer.residuals import SparseResidual

    if isinstance(residual, _ResidualSum):
        return _ResidualSum(tuple(_scale_block_residual(item, gain) for item in residual.items))
    if isinstance(residual, _ScaledResidual):
        return replace(residual, exposure=math.fsum((residual.exposure, math.log2(gain))))
    if isinstance(residual, SparseResidual):
        return replace(residual, scale=residual.scale * gain, error_bound=residual.error_bound * gain)
    if callable(residual):
        return lambda points, residual=residual, gain=gain: residual(points) * gain
    return np.asarray(residual) * gain if np.asarray(residual).ndim else float(residual) * gain


def _residual_direct_value(residual: Any, points: np.ndarray) -> np.ndarray | None:
    """Evaluate only physical-point residual terms for a block query.

    Lattice residuals are sampled at each multilinear corner.  Keeping that
    distinction explicit prevents a callable residual from being accidentally
    indexed by the query ordinal (which is unrelated to its physical point).
    """

    from spf.renderer.normalization import _ResidualSum, _ScaledResidual
    from spf.renderer.residuals import SparseResidual

    if isinstance(residual, _ResidualSum):
        values: np.ndarray | None = None
        for item in residual.items:
            item_values = _residual_direct_value(item, points)
            if item_values is None:
                continue
            values = item_values if values is None else values + item_values
        return values
    if isinstance(residual, _ScaledResidual):
        values = _residual_direct_value(residual.residual, points)
        return None if values is None else residual.scale_values(values)
    if isinstance(residual, SparseResidual):
        return None
    if callable(residual):
        # The residual stream owns its carrier; widening it here promoted every
        # real residual to a Cartesian pair plane for nothing.
        values = np.asarray(v02._residual_value(residual, points)).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError("block representation residual does not match the query points")
        return values
    # Scalars and arrays matching the block extent are lattice residuals.
    # They are handled at the corner address, where interpolation is defined.
    return None


def _real_residual(values: Any, *, tolerance: float) -> np.ndarray:
    """Project a residual at a positive consumer boundary without dropping phase."""

    array = np.asarray(values)
    if np.iscomplexobj(array):
        imaginary = np.abs(np.imag(array))
        if np.any(imaginary > tolerance):
            raise ValueError("block density residual must be real at a positive consumer boundary")
        array = np.real(array)
    return np.asarray(array, dtype=np.float64)


def _residual_lattice_or_direct(
    residual: Any,
    linear: np.ndarray,
    shape: tuple[int, ...],
    points: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Split a residual into lattice-addressed and physical-point terms."""

    from spf.renderer.normalization import _ResidualSum, _ScaledResidual
    from spf.renderer.residuals import SparseResidual

    if isinstance(residual, _ResidualSum):
        # Accumulate in each term's own carrier (no pre-seeded Cartesian plane):
        # a real residual sum stays real all the way to the consumer.
        lattice: np.ndarray | None = None
        direct: np.ndarray | None = None
        for item in residual.items:
            item_lattice, item_direct = _residual_lattice_or_direct(item, linear, shape, points)
            if item_lattice is not None:
                lattice = item_lattice if lattice is None else lattice + item_lattice
            if item_direct is not None:
                direct = item_direct if direct is None else direct + item_direct
        return lattice, direct
    if isinstance(residual, _ScaledResidual):
        scaled_lattice, scaled_direct = _residual_lattice_or_direct(residual.residual, linear, shape, points)
        if scaled_lattice is not None:
            scaled_lattice = residual.scale_values(scaled_lattice)
        if scaled_direct is not None:
            scaled_direct = residual.scale_values(scaled_direct)
        return scaled_lattice, scaled_direct
    if isinstance(residual, SparseResidual):
        if tuple(int(value) for value in residual.shape) != shape:
            raise ValueError("block representation residual shape must match its source block")
        return np.asarray(residual.sample_flat(linear)), None
    if callable(residual):
        values = np.asarray(v02._residual_value(residual, points)).reshape(-1)
        if values.size != points.shape[0]:
            raise ValueError("block representation residual does not match the query points")
        return None, values
    raw = np.asarray(residual)
    if raw.ndim == 0 or raw.size == 1:
        return np.full(linear.shape, raw.reshape(-1)[0], dtype=raw.dtype), None
    if raw.shape != shape:
        raise ValueError("block representation residual must be scalar or match its source block shape")
    return np.asarray(raw.reshape(-1)[linear]), None


def _sample_block_density(
    block: Any,
    points: Any,
    *,
    value_kind: str | None = None,
    boundary: str = "zero",
    tolerance: float = 1.0e-12,
    residual: Any | None = None,
    residual_points: Any | None = None,
) -> float | np.ndarray:
    """Multilinearly sample a block through addressed word reads.

    Block headers and residuals are evaluated only at the lattice corners
    touched by the query.  This keeps ray quadrature and material UV queries
    on the structural path while matching :class:`DenseDensity`'s coordinate
    and zero/clamp boundary semantics.
    """

    selected_boundary = str(boundary).lower().replace("-", "_")
    if selected_boundary not in {"zero", "clamp"}:
        raise ValueError("block density boundary must be 'zero' or 'clamp'")
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim == 1:
        if raw.shape != (len(block.shape),):
            raise ValueError("block density point has the wrong dimension")
        point_array = raw.reshape(1, -1)
        scalar = True
        output_shape: tuple[int, ...] = ()
    else:
        if raw.ndim == 0 or raw.shape[-1] != len(block.shape):
            raise ValueError("block density points have the wrong trailing dimension")
        point_array = raw.reshape(-1, len(block.shape))
        scalar = False
        output_shape = raw.shape[:-1]
    if not np.all(np.isfinite(point_array)):
        raise ValueError("block density points must be finite")
    extent = np.asarray(block.shape, dtype=np.int64) - 1
    inside = np.all((point_array >= 0.0) & (point_array <= extent), axis=1)
    clipped = np.clip(point_array, 0.0, extent)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, extent)
    weight = clipped - lower
    result = np.zeros(point_array.shape[0], dtype=np.float64)
    residual_direct: np.ndarray | None = None
    physical: np.ndarray | None = None
    if residual is not None:
        physical = point_array if residual_points is None else np.asarray(residual_points, dtype=np.float64)
        if physical.ndim == 1:
            physical = physical.reshape(1, -1)
        if physical.shape != point_array.shape:
            raise ValueError("block residual query points must match the density query shape")
        residual_direct = _residual_direct_value(residual, physical)
    for corner in np.ndindex((2,) * len(block.shape)):
        indices = tuple(np.where(corner[axis], upper[:, axis], lower[:, axis]) for axis in range(len(block.shape)))
        linear = np.ravel_multi_index(indices, tuple(int(value) for value in block.shape))
        samples = _project_block_real(block, linear, value_kind=value_kind, tolerance=tolerance)
        if residual is not None:
            corner_residual, _ = _residual_lattice_or_direct(
                residual,
                linear,
                tuple(int(value) for value in block.shape),
                cast(np.ndarray, physical),
            )
            if corner_residual is not None:
                samples = samples + _real_residual(corner_residual, tolerance=tolerance)
        factor = np.ones(point_array.shape[0], dtype=np.float64)
        for axis, bit in enumerate(corner):
            factor *= weight[:, axis] if bit else 1.0 - weight[:, axis]
        result += factor * samples
    if residual_direct is not None:
        result += _real_residual(residual_direct, tolerance=tolerance)
    if selected_boundary == "zero":
        result[~inside] = 0.0
    if scalar:
        return float(result[0])
    return result.reshape(output_shape)


@dataclass(frozen=True, slots=True)
class DenseDensity:
    """A non-negative dense density grid with multilinear sampling."""

    values: Any
    origin: tuple[float, ...] = ()
    spacing: tuple[float, ...] = ()
    boundary: str = "zero"

    def __post_init__(self) -> None:
        values = _validated_density(self.values)
        if values.ndim == 0 or values.size == 0:
            raise ValueError("dense density must have at least one sample")
        dimension = values.ndim
        origin = tuple(float(value) for value in (self.origin or (0.0,) * dimension))
        spacing = tuple(float(value) for value in (self.spacing or (1.0,) * dimension))
        if len(origin) != dimension or len(spacing) != dimension:
            raise ValueError("dense density origin and spacing must match its dimension")
        if any(not math.isfinite(value) for value in origin) or any(
            value <= 0.0 or not math.isfinite(value) for value in spacing
        ):
            raise ValueError("dense density origin must be finite and spacing positive")
        boundary = str(self.boundary).lower().replace("-", "_")
        if boundary not in {"zero", "clamp"}:
            raise ValueError("dense density boundary must be 'zero' or 'clamp'")
        values = np.array(values, dtype=np.float64, copy=True)
        values.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "boundary", boundary)

    @property
    def dimension(self) -> int:
        return int(self.values.ndim)

    def sample(self, points: Any) -> float | np.ndarray:
        raw_points = np.asarray(points, dtype=np.float64)
        if raw_points.ndim == 1:
            if raw_points.shape != (self.dimension,):
                raise ValueError("dense density point has the wrong dimension")
            point_array = raw_points.reshape(1, -1)
            scalar = True
        else:
            if raw_points.shape[-1] != self.dimension:
                raise ValueError("dense density points have the wrong trailing dimension")
            point_array = raw_points.reshape(-1, self.dimension)
            scalar = False
        coordinates = (point_array - np.asarray(self.origin)) / np.asarray(self.spacing)
        inside = np.all((coordinates >= 0.0) & (coordinates <= (np.asarray(self.values.shape) - 1)), axis=1)
        if self.boundary == "clamp":
            clipped = np.clip(coordinates, 0.0, np.asarray(self.values.shape) - 1)
        else:
            clipped = np.clip(coordinates, 0.0, np.asarray(self.values.shape) - 1)
        lower = np.floor(clipped).astype(np.int64)
        upper = np.minimum(lower + 1, np.asarray(self.values.shape) - 1)
        weight = clipped - lower
        result = np.zeros(point_array.shape[0], dtype=np.float64)
        for corner in np.ndindex((2,) * self.dimension):
            index = tuple(np.where(corner[axis], upper[:, axis], lower[:, axis]) for axis in range(self.dimension))
            factor = np.ones(point_array.shape[0], dtype=np.float64)
            for axis, bit in enumerate(corner):
                factor *= weight[:, axis] if bit else 1.0 - weight[:, axis]
            result += factor * self.values[index]
        if self.boundary == "zero":
            result[~inside] = 0.0
        if scalar:
            return float(result[0])
        return result.reshape(raw_points.shape[:-1])

    __call__ = sample


def _infer_density_kind(source: Any) -> DensityKind:
    # Block-backed fields are an addressed density source.  Keep this before
    # the callable/constant checks so DensityField retains its chart metadata
    # instead of wrapping a structural carrier in DenseDensity (which would
    # necessarily force a full reconstruction).
    from spf.renderer.carriers import as_block_view

    if as_block_view(source) is not None or (
        isinstance(source, Field) and as_block_view(source.representation.payload) is not None
    ):
        return DensityKind.BLOCK
    if isinstance(source, (DenseDensity, np.ndarray)):
        return DensityKind.DENSE
    if isinstance(source, v02.GaussianChirpAtom):
        return DensityKind.ATOM
    if isinstance(source, v02.AtomMixture):
        return DensityKind.MIXTURE
    if isinstance(source, v02.SeparableField):
        return DensityKind.SEPARABLE
    if callable(source):
        return DensityKind.CALLABLE
    return DensityKind.CONSTANT


@dataclass(frozen=True, slots=True)
class DensityField:
    """Typed density source with explicit dense-grid sampling metadata."""

    source: Any
    kind: DensityKind | str | None = None
    origin: tuple[float, ...] = ()
    spacing: tuple[float, ...] = ()
    boundary: str = "zero"

    def __post_init__(self) -> None:
        selected = (
            _infer_density_kind(self.source)
            if self.kind is None
            else DensityKind(str(self.kind).lower().replace("-", "_"))
        )
        source = self.source
        if selected is DensityKind.DENSE and not isinstance(source, DenseDensity):
            source = DenseDensity(source, self.origin, self.spacing, self.boundary)
        if selected is DensityKind.BLOCK:
            from spf.renderer.carriers import as_block_view

            payload = source.representation.payload if isinstance(source, Field) else source
            block = as_block_view(payload)
            if block is None:
                raise TypeError("block density kind requires a TensorBlock, SPFBlock, V3 block, or SPF kernel")
            dimension = len(block.shape)
            origin = tuple(float(value) for value in (self.origin or (0.0,) * dimension))
            spacing = tuple(float(value) for value in (self.spacing or (1.0,) * dimension))
            if len(origin) != dimension or len(spacing) != dimension:
                raise ValueError("block density origin and spacing must match its dimension")
            if any(not math.isfinite(value) for value in origin) or any(
                value <= 0.0 or not math.isfinite(value) for value in spacing
            ):
                raise ValueError("block density origin must be finite and spacing positive")
            object.__setattr__(self, "origin", origin)
            object.__setattr__(self, "spacing", spacing)
            boundary = str(self.boundary).lower().replace("-", "_")
            if boundary not in {"zero", "clamp"}:
                raise ValueError("block density boundary must be 'zero' or 'clamp'")
            object.__setattr__(self, "boundary", boundary)
        object.__setattr__(self, "kind", selected)
        object.__setattr__(self, "source", source)

    def sample(self, points: Any, *, tolerance: float = 1.0e-12) -> float | np.ndarray:
        return sample_density(self, points, tolerance=tolerance)

    __call__ = sample


def _call_density(source: Any, points: np.ndarray) -> Any:
    if callable(source):
        try:
            return source(points)
        except (TypeError, ValueError, IndexError):
            if points.ndim == 1:
                return source(points)
            return np.asarray([source(point) for point in points])
    if hasattr(source, "evaluate"):
        try:
            return source.evaluate(points)
        except (TypeError, ValueError):
            if points.ndim == 1:
                return source.evaluate(points)
            return np.asarray([source.evaluate(point) for point in points])
    raise TypeError(f"unsupported density source {type(source).__name__}")


def _prepared_density(source: Any) -> Any:
    """Compile field operations once before quadrature; retain physical coordinates."""
    if isinstance(source, np.ndarray):
        return DenseDensity(source)
    if not isinstance(source, Field):
        return source
    from spf.renderer.model import MaterializationRequest

    # Parameter-only closed operations (exposure, masking, and compatible
    # header/filter closures) are folded into the addressed view first.  A
    # block field with such a prefix must not be routed through field_carrier,
    # whose generic pending-operation path materializes a whole plane.
    from spf.renderer.normalization import normalize_field

    normalized = normalize_field(source).field
    payload = normalized.representation.payload
    from spf.renderer.carriers import as_block_view

    block = as_block_view(payload)
    from spf.renderer.block_filter import FilteredBlockView

    # Spatial convolution is an explicit periodic/filter terminal. Prepare it
    # once for this density ray, then sample the resulting regular grid at all
    # quadrature points. Calling FilteredBlockView.sample_words at every
    # multilinear corner would either fail closed or repeat a full FFT.
    if isinstance(block, FilteredBlockView) and not block.frequency_domain:
        from spf.renderer.block_filter import PeriodicBlockExecution

        if normalized.representation.residual is not None:
            raise ValueError(
                "block density spatial filtering with a representation residual requires "
                "an explicit residual filter terminal"
            )
        try:
            filtered = np.asarray(PeriodicBlockExecution(block).execute())
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "block density spatial filtering requires its explicit periodic filter terminal"
            ) from exc
        if np.iscomplexobj(filtered):
            if np.any(np.abs(filtered.imag) > 1.0e-12 * np.maximum(1.0, np.abs(filtered.real))):
                raise ValueError("filtered block density must be real at the optical-depth boundary")
            filtered = filtered.real
        payload = _validated_density(filtered)
        block = None
    if block is not None and normalized.operations:
        if normalized.representation.residual is not None:
            pending = ", ".join(operation.op for operation in normalized.operations)
            raise ValueError(
                "block density residual cannot cross deferred operations without an explicit residual terminal: "
                f"{pending}"
            )
        from spf.renderer.carriers import field_carrier

        payload = field_carrier(normalized)
        block = as_block_view(payload)
    elif block is None and normalized.operations:
        from spf.renderer.carriers import field_carrier

        payload = field_carrier(normalized)
    if block is not None:
        chart = normalized.chart
        if chart.dimension not in {0, len(block.shape)}:
            raise ValueError("block density chart dimension must match the block extent")
        if chart.dimension == 0:
            chart = type(source.chart).identity(len(block.shape))
        matrix = np.asarray(chart.basis, dtype=np.float64) @ np.diag(
            np.asarray(chart.scale, dtype=np.float64) * np.asarray(chart.sample_spacing, dtype=np.float64)
        )
        if matrix.shape != (len(block.shape), len(block.shape)) or np.linalg.matrix_rank(matrix) != len(block.shape):
            raise ValueError("block density chart must be an invertible square mapping")
        inverse = np.linalg.inv(matrix)
        origin = np.asarray(chart.origin, dtype=np.float64)
        value_kind = str(normalized.value_kind)
        support_kind = str(getattr(normalized.support.kind, "value", normalized.support.kind)).lower()
        support_region = normalized.support.region
        residual = normalized.representation.residual

        def sample_block(points: Any) -> Any:
            coordinates = np.asarray(points, dtype=np.float64)
            if coordinates.ndim == 0:
                coordinates = coordinates.reshape(1)
            if coordinates.shape[-1] != len(block.shape):
                raise ValueError("block density points have the wrong trailing dimension")
            point_shape = coordinates.shape[:-1]
            flat_coordinates = coordinates.reshape(-1, len(block.shape))
            if support_kind == "empty":
                return np.zeros(point_shape, dtype=np.float64) if point_shape else 0.0
            support_inside = (
                np.ones(flat_coordinates.shape[0], dtype=bool)
                if support_region is None
                else np.asarray(
                    [support_region.contains(point.tolist()) for point in flat_coordinates], dtype=bool
                )
            )
            local = (coordinates - origin) @ inverse.T
            sampled = np.asarray(
                _sample_block_density(
                    block,
                    local,
                    value_kind=value_kind,
                    residual=residual,
                    residual_points=flat_coordinates,
                )
            )
            if support_region is not None:
                sampled = np.where(support_inside.reshape(point_shape), sampled, 0.0)
            return float(sampled) if sampled.ndim == 0 else sampled

        return sample_block
    if isinstance(payload, np.ndarray):
        if payload.dtype.kind in "iu" and str(normalized.value_kind) in {"spf_complex", "words"}:
            payload = decode_words(payload)
        dense = DenseDensity(payload)
        chart = normalized.chart
        if chart.dimension:
            if chart.dimension != dense.dimension:
                raise ValueError("dense density chart dimension must match the payload extent")
            inverse = np.linalg.inv(np.asarray(chart.basis))
            factors = np.asarray(chart.scale) * np.asarray(chart.sample_spacing)
            support_kind = str(getattr(normalized.support.kind, "value", normalized.support.kind)).lower()
            support_region = normalized.support.region

            def sample(points: Any) -> Any:
                coordinates = np.asarray(points, dtype=np.float64)
                local = ((coordinates - chart.origin) @ inverse.T) / factors
                values = dense.sample(local)
                if support_kind == "empty":
                    return np.zeros_like(values, dtype=np.float64)
                if support_region is None:
                    return values
                point_array = coordinates.reshape(-1, chart.dimension)
                inside = np.asarray(
                    [support_region.contains(point.tolist()) for point in point_array], dtype=bool
                )
                if np.asarray(values).ndim == 0:
                    return values if bool(inside[0]) else 0.0
                return np.where(inside.reshape(np.asarray(values).shape), values, 0.0)

            return sample
        return dense

    def sample_field(points: Any) -> Any:
        return normalized.materialize(
            MaterializationRequest(region=cast(Any, {"points": np.asarray(points)}), output_format="complex")
        )

    return sample_field


def _constant_density(source: Any) -> bool:
    from spf.renderer.carriers import as_block_view

    return (
        not isinstance(
            source,
            (Field, np.ndarray, DenseDensity, DensityField, v02.GaussianChirpAtom, v02.AtomMixture, v02.SeparableField),
        )
        and as_block_view(source) is None
        and not callable(source)
    )


def sample_density(
    source: Any,
    points: Any,
    *,
    origin: Sequence[float] | None = None,
    spacing: Sequence[float] | None = None,
    boundary: str = "zero",
    tolerance: float = 1.0e-12,
) -> float | np.ndarray:
    """Sample a constant, atom, separable, callable, or dense density field."""

    result: Any
    if isinstance(source, Field):
        return sample_density(_prepared_density(source), points, tolerance=tolerance)
    from spf.renderer.carriers import as_block_view

    block = as_block_view(source)
    if block is not None:
        coordinates = np.asarray(points, dtype=np.float64)
        if coordinates.ndim == 0:
            coordinates = coordinates.reshape(1)
        if coordinates.shape[-1] != len(block.shape):
            raise ValueError("block density points have the wrong trailing dimension")
        if origin is not None or spacing is not None:
            selected_origin = (
                np.zeros(len(block.shape), dtype=np.float64)
                if origin is None
                else np.asarray(origin, dtype=float)
            )
            selected_spacing = (
                np.ones(len(block.shape), dtype=np.float64)
                if spacing is None
                else np.asarray(spacing, dtype=float)
            )
            if selected_origin.shape != (len(block.shape),) or selected_spacing.shape != (len(block.shape),):
                raise ValueError("block density origin and spacing must match the block extent")
            if (
                np.any(~np.isfinite(selected_origin))
                or np.any(~np.isfinite(selected_spacing))
                or np.any(selected_spacing <= 0)
            ):
                raise ValueError("block density origin must be finite and spacing positive")
            coordinates = (coordinates - selected_origin) / selected_spacing
        result = _sample_block_density(block, coordinates, value_kind=None, boundary=boundary, tolerance=tolerance)
        checked = _validated_density(result, tolerance=tolerance)
        return float(checked) if checked.ndim == 0 else checked
    if isinstance(source, DensityField):
        if source.kind is DensityKind.DENSE:
            result = source.source.sample(points)
        elif source.kind is DensityKind.BLOCK:
            if isinstance(source.source, Field):
                # A Field owns its chart, support, and representation ledger;
                # do not apply DensityField's default lattice metadata a
                # second time or discard that custody at the ray boundary.
                result = sample_density(source.source, points, tolerance=tolerance)
            else:
                result = sample_density(
                    source.source,
                    points,
                    origin=source.origin or None,
                    spacing=source.spacing or None,
                    boundary=source.boundary,
                    tolerance=tolerance,
                )
        else:
            result = sample_density(source.source, points, tolerance=tolerance)
        return (
            _validated_density(result, tolerance=tolerance).item()
            if np.asarray(result).ndim == 0
            else _validated_density(result, tolerance=tolerance)
        )
    if isinstance(source, DenseDensity):
        result = source.sample(points)
        return (
            _validated_density(result, tolerance=tolerance).item()
            if np.asarray(result).ndim == 0
            else _validated_density(result, tolerance=tolerance)
        )
    if isinstance(source, np.ndarray):
        dense = DenseDensity(
            source, () if origin is None else tuple(origin), () if spacing is None else tuple(spacing), boundary
        )
        return dense.sample(points)
    raw_points = np.asarray(points, dtype=np.float64)
    if raw_points.ndim == 0:
        raw_points = raw_points.reshape(1)
    if isinstance(source, (v02.GaussianChirpAtom, v02.AtomMixture)):
        result = source.evaluate(raw_points)
    elif isinstance(source, v02.SeparableField):
        from spf.renderer.separable_terminal import evaluate_grid

        result, _, _ = evaluate_grid(source, raw_points.reshape(-1, raw_points.shape[-1]), kind=str(source.value_kind))
        if np.asarray(result).dtype.kind in "iu":
            result = decode_words(result)
        result = np.asarray(result).reshape(raw_points.shape[:-1])
    elif callable(source):
        result = _call_density(source, raw_points)
    else:
        value = _decode_scalar(source)
        result = np.full(raw_points.shape[:-1] if raw_points.ndim > 1 else (), value)
    checked = _validated_density(result, tolerance=tolerance)
    return float(checked) if checked.ndim == 0 else checked


def sample_atom_density(atom: v02.GaussianChirpAtom, points: Any, **kwargs: Any) -> Any:
    return sample_density(atom, points, **kwargs)


def sample_separable_density(field: v02.SeparableField, points: Any, **kwargs: Any) -> Any:
    return sample_density(field, points, **kwargs)


def sample_dense_density(values: Any, points: Any, **kwargs: Any) -> Any:
    return sample_density(values, points, **kwargs)


def _scale_error_ledger(ledger: v02.ErrorLedger, scale: float) -> v02.ErrorLedger:
    factor = abs(float(scale))
    return v02.ErrorLedger(
        **{
            name: float(getattr(ledger, name)) * factor
            for name in (
                "source_quantization",
                "representation_fit",
                "pruning",
                "reducer",
                "transform",
                "materialization",
                "display",
            )
        },
        stages=ledger.stages,
    )


def _density_error_ledger(source: Any, *, scale: float = 1.0) -> v02.ErrorLedger:
    if isinstance(source, DensityField):
        return _density_error_ledger(source.source, scale=scale)
    if isinstance(source, Field):
        from spf.renderer.normalization import complete_field_ledger

        return _scale_error_ledger(complete_field_ledger(source), scale)
    bound = float(getattr(source, "error_bound", 0.0))
    if bound == 0.0:
        return v02.ErrorLedger()
    return v02.ErrorLedger(source_quantization=bound * abs(float(scale)))


@dataclass(frozen=True, slots=True)
class QuadratureResult:
    """Optical-depth quadrature value and an order-doubling diagnostic bound."""

    value: float
    error_bound: float
    samples: int
    order: int
    method: str = "gauss_legendre"
    quadrature_error_bound: float | None = None
    source_error_bound: float = 0.0
    source_ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _finite_nonnegative("quadrature optical depth", self.value, allow_inf=True))
        object.__setattr__(
            self, "error_bound", _finite_nonnegative("quadrature error bound", self.error_bound, allow_inf=True)
        )
        if int(self.samples) < 0 or int(self.order) <= 0:
            raise ValueError("quadrature samples/order must be positive")
        object.__setattr__(self, "samples", int(self.samples))
        object.__setattr__(self, "order", int(self.order))
        source_error = _finite_nonnegative("quadrature source error bound", self.source_error_bound, allow_inf=True)
        quadrature_error = self.error_bound if self.quadrature_error_bound is None else self.quadrature_error_bound
        quadrature_error = _finite_nonnegative("quadrature estimate error bound", quadrature_error, allow_inf=True)
        if not isinstance(self.source_ledger, v02.ErrorLedger):
            raise TypeError("quadrature source_ledger must be an ErrorLedger")
        ledger_bound = float(self.source_ledger.total_bound)
        if math.isnan(ledger_bound) or ledger_bound < 0.0:
            raise ValueError("quadrature source_ledger must have a non-negative bound")
        if source_error < ledger_bound:
            source_error = ledger_bound
        object.__setattr__(self, "source_error_bound", source_error)
        object.__setattr__(self, "quadrature_error_bound", quadrature_error)

    @property
    def tau(self) -> float:
        return self.value

    @property
    def ledger(self) -> v02.ErrorLedger:
        source = self.source_ledger
        if source.total_bound == 0.0 and self.source_error_bound:
            source = v02.ErrorLedger(source_quantization=self.source_error_bound)
        return source.plus(v02.ErrorLedger(materialization=cast(float, self.quadrature_error_bound)))

    def __float__(self) -> float:
        return self.value


@lru_cache(maxsize=32)
def _quadrature_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def quadrature_optical_depth(
    density: Any,
    start: Any,
    end: Any,
    *,
    order: int = 8,
    quadrature_order: int | None = None,
    return_result: bool = False,
    tolerance: float = 1.0e-12,
) -> float | QuadratureResult:
    """Integrate ``sigma_t`` along a ray with Gauss--Legendre quadrature."""

    selected_order = int(order if quadrature_order is None else quadrature_order)
    if selected_order <= 0:
        raise ValueError("quadrature order must be positive")
    left, right, length = _ray_points(start, end)
    if length == 0.0:
        result = QuadratureResult(0.0, 0.0, 0, selected_order)
        return result if return_result else result.value
    if _constant_density(density):
        value = _validated_density(_decode_scalar(density), tolerance=tolerance)
        scalar_value = float(value)
        result = QuadratureResult(scalar_value * length, 0.0, 0, selected_order, "constant_closed_form")
        return result if return_result else result.value
    nodes, weights = _quadrature_rule(selected_order)
    points = (left + 0.5 * (nodes[:, None] + 1.0) * (right - left)).astype(np.float64)
    values = np.asarray(sample_density(density, points, tolerance=tolerance), dtype=np.float64).reshape(-1)
    estimate = float(0.5 * length * np.dot(weights, values))
    high_order = max(selected_order + 1, 2 * selected_order)
    high_nodes, high_weights = _quadrature_rule(high_order)
    high_points = left + 0.5 * (high_nodes[:, None] + 1.0) * (right - left)
    high_values = np.asarray(sample_density(density, high_points, tolerance=tolerance), dtype=np.float64).reshape(-1)
    high_estimate = float(0.5 * length * np.dot(high_weights, high_values))
    quadrature_error = abs(high_estimate - estimate) + np.finfo(np.float64).eps * max(1.0, abs(high_estimate))
    source_ledger = _density_error_ledger(density, scale=length)
    source_error = source_ledger.total_bound
    error = quadrature_error + source_error
    result = QuadratureResult(
        high_estimate,
        error,
        selected_order + high_order,
        selected_order,
        "gauss_legendre_order_doubling",
        quadrature_error_bound=quadrature_error,
        source_error_bound=source_error,
        source_ledger=source_ledger,
    )
    return result if return_result else result.value


integrate_density = quadrature_optical_depth


def _channels(value: Any) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return (value.item(),)
        return tuple(value.reshape(-1).tolist())
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _numeric(value: Any) -> complex | float:
    decoded = _decode_scalar(value)
    if isinstance(decoded, np.generic):
        return cast(complex | float, decoded.item())
    return decoded


def _family(value: Any) -> str:
    if isinstance(value, v02.LogPos):
        return "log_pos"
    if isinstance(value, v02.LogSigned):
        return "log_signed"
    if isinstance(value, v02.SPF32Value) or (
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
    ):
        return "spf_complex"
    return "native"


def _typed_mul(left: Any, right: Any) -> Any:
    if isinstance(left, v02.LogPos) and isinstance(right, v02.LogPos):
        return v02.logpos_mul(left, right)
    if isinstance(left, v02.LogSigned) and isinstance(right, v02.LogSigned):
        if left.is_nan or right.is_nan:
            return v02.LogSigned.nan()
        if left.is_zero or right.is_zero:
            return v02.LogSigned.zero()
        return v02.LogSigned(left.L + right.L, left.sign * right.sign)
    if isinstance(left, v02.LogPos) and isinstance(right, v02.LogSigned):
        return _typed_mul(v02.LogSigned(left.L, 1), right)
    if isinstance(left, v02.LogSigned) and isinstance(right, v02.LogPos):
        return _typed_mul(left, v02.LogSigned(right.L, 1))
    if isinstance(left, v02.SPF32Value):
        left = left.word
    if isinstance(right, v02.SPF32Value):
        right = right.word
    if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
        return v02.spf_mul(int(left), int(right))
    if _family(left) != "native" or _family(right) != "native":
        raise TypeError("transport components must use matching typed or native representations")
    return left * right


def _typed_add(left: Any, right: Any) -> Any:
    if isinstance(left, v02.LogPos) and isinstance(right, v02.LogPos):
        return v02.logadd_pos(left, right)
    if isinstance(left, v02.LogSigned) and isinstance(right, v02.LogSigned):
        return v02.logadd_signed(left, right)
    if isinstance(left, v02.LogPos) and isinstance(right, v02.LogSigned):
        return v02.logadd_signed(v02.LogSigned(left.L, 1), right)
    if isinstance(left, v02.LogSigned) and isinstance(right, v02.LogPos):
        return v02.logadd_signed(left, v02.LogSigned(right.L, 1))
    if isinstance(left, v02.SPF32Value):
        left = left.word
    if isinstance(right, v02.SPF32Value):
        right = right.word
    if isinstance(left, (int, np.integer)) and isinstance(right, (int, np.integer)):
        return v02.spf_add_complex_reference(int(left), int(right))
    if _family(left) != "native" or _family(right) != "native":
        raise TypeError("transport components must use matching typed or native representations")
    return left + right


def _coerce_transport_operand(value: Any, family: str) -> Any:
    """Cross an explicit native-to-transport representation boundary."""

    if _family(value) == family:
        return value
    if _family(value) != "native":
        raise TypeError("transport components must use matching typed or native representations")
    numeric = _numeric(value)
    if family == "log_pos":
        if isinstance(numeric, complex) and abs(numeric.imag) > 1.0e-12:
            raise TypeError("positive-log transport cannot consume a complex background")
        real = float(numeric.real if isinstance(numeric, complex) else numeric)
        if real < 0.0:
            raise ValueError("positive-log transport cannot consume a negative background")
        return v02.LogPos.from_value(real)
    if family == "log_signed":
        if isinstance(numeric, complex) and abs(numeric.imag) > 1.0e-12:
            raise TypeError("signed-log transport cannot consume a complex background")
        return v02.LogSigned.from_value(float(numeric.real if isinstance(numeric, complex) else numeric))
    if family == "spf_complex":
        return int(encode_words(complex(numeric)))
    return value


def _max_abs(value: Any) -> float:
    numeric = np.asarray([_numeric(item) for item in _channels(value)], dtype=np.complex128)
    return float(np.max(np.abs(numeric))) if numeric.size else 0.0


def _transport_ledger(items: Sequence[TransportRGB], total_error: float) -> v02.ErrorLedger:
    """Preserve input error categories while adding only a missing output gap."""

    base = v02.ErrorLedger()
    for item in items:
        base = base.plus(item.ledger)
    if math.isnan(base.total_bound) or math.isnan(total_error):
        gap = math.nan
    elif math.isinf(base.total_bound) or base.total_bound >= total_error:
        gap = 0.0
    else:
        gap = total_error - base.total_bound
    stage = v02.ErrorLedger(reducer=gap)
    return replace(base, reducer=base.reducer + gap, stages=(*base.stages, stage))


@dataclass(frozen=True, slots=True)
class TransportRGB:
    """An RGB/channel affine transport map ``x -> C + T*x``."""

    T: Any
    C: Any
    error_bound_T: float = 0.0  # noqa: N815 - transport-map notation
    error_bound_C: float = 0.0  # noqa: N815 - transport-map notation
    ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)
    # When present, this is the exact optical depth represented by T.  It is
    # an internal consumer certificate: SPF32 log words remain available at
    # the terminal boundary, while ordered volume composition can add tau in
    # native precision and perform one exponential at the end.  This avoids
    # quantizing every small ray segment independently into an all-zero q8
    # log word.  A caller-supplied certificate is checked against T within
    # the advertised transmittance error before a reducer may consume it.
    optical_depth: tuple[float, ...] | None = field(default=None, kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        transmittance = _channels(self.T)
        contribution = _channels(self.C)
        channels = max(len(transmittance), len(contribution))
        if channels == 0:
            raise ValueError("transport T and C must have a positive channel count")
        if len(transmittance) == 1 and channels > 1:
            transmittance = transmittance * channels
        if len(contribution) == 1 and channels > 1:
            contribution = contribution * channels
        if len(transmittance) != len(contribution):
            raise ValueError("transport T and C channel counts are not broadcast-compatible")
        for value in transmittance:
            if isinstance(value, v02.LogSigned) and not value.is_zero and value.sign < 0:
                raise ValueError("transport transmittance must be non-negative")
            numeric = _numeric(value)
            if isinstance(numeric, complex):
                if abs(numeric.imag) > 1.0e-12 or numeric.real < 0.0 or not math.isfinite(numeric.real):
                    raise ValueError("transport transmittance must be finite and non-negative")
            elif math.isnan(float(numeric)) or float(numeric) < 0.0:
                raise ValueError("transport transmittance must be non-negative")
        for name in ("error_bound_T", "error_bound_C"):
            _finite_nonnegative(name, getattr(self, name), allow_inf=True)
        if not isinstance(self.ledger, v02.ErrorLedger):
            raise TypeError("TransportRGB.ledger must be an ErrorLedger")
        exact_tau = self.optical_depth
        if exact_tau is not None:
            tau_values = _channels(exact_tau)
            if len(tau_values) == 1 and channels > 1:
                tau_values = tau_values * channels
            if len(tau_values) != channels:
                raise ValueError("transport optical_depth channel count does not match T/C")
            normalized_tau = tuple(
                _finite_nonnegative("transport optical depth", value, allow_inf=True) for value in tau_values
            )
            exact_tau = normalized_tau
            # ``optical_depth`` authorizes the guarded reducer to replace the
            # stored T product with exp(-sum(tau)).  Do not allow an
            # arbitrary public TransportRGB construction to attach a
            # contradictory certificate and silently change the affine map.
            # One ulp is admitted for a direct native construction; any
            # larger discrepancy must be covered by error_bound_T.  Compare
            # through the monotone log interval so validation does not add an
            # exponential to the guarded terminal reduction.
            transmittance_error = float(self.error_bound_T)
            if not math.isinf(transmittance_error):
                epsilon = np.finfo(np.float64).eps
                for value, tau_value in zip(transmittance, exact_tau, strict=True):
                    actual = complex(_numeric(value))
                    if not math.isfinite(actual.real):
                        raise ValueError("transport optical_depth certificate disagrees with T")
                    # Scale the native arithmetic allowance at the actual
                    # transmittance.  An epsilon at unity would make T=0
                    # appear compatible with a small, but still representable,
                    # exp(-tau) value.
                    tolerance = transmittance_error + max(epsilon * abs(actual), math.ulp(abs(actual)))
                    if abs(actual.imag) > tolerance:
                        raise ValueError("transport optical_depth certificate disagrees with T")
                    # Squaring a subnormal/very small tolerance would erase
                    # its interval and reject a valid underflowed exp(-tau).
                    radius = tolerance if actual.imag == 0.0 else tolerance * math.sqrt(
                        max(0.0, 1.0 - (actual.imag / tolerance) ** 2)
                    )
                    lower = max(0.0, actual.real - radius)
                    upper = actual.real + radius
                    lower_tau = 0.0 if upper >= 1.0 else math.inf if upper <= 0.0 else -math.log(upper)
                    upper_tau = math.inf if lower <= 0.0 else -math.log(lower)
                    if tau_value < lower_tau or tau_value > upper_tau:
                        raise ValueError("transport optical_depth certificate disagrees with T")
        object.__setattr__(self, "T", tuple(transmittance))
        object.__setattr__(self, "C", tuple(contribution))
        object.__setattr__(self, "error_bound_T", float(self.error_bound_T))
        object.__setattr__(self, "error_bound_C", float(self.error_bound_C))
        object.__setattr__(self, "optical_depth", exact_tau)

    @property
    def channels(self) -> int:
        return len(self.T)

    @property
    def transmittance(self) -> Any:
        return self.T[0] if self.channels == 1 else self.T

    @property
    def contribution(self) -> Any:
        return self.C[0] if self.channels == 1 else self.C

    @property
    def absolute_error_bound(self) -> float:
        return self.error_bound_T + self.error_bound_C

    @property
    def error_bound(self) -> float:
        return self.absolute_error_bound

    def numeric(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray([_numeric(value) for value in self.T]),
            np.asarray([_numeric(value) for value in self.C]),
        )

    def apply(self, background: Any = 0.0) -> Any:
        background_values = _channels(background)
        if len(background_values) not in {1, self.channels}:
            raise ValueError("background channel count does not match transport")
        values = []
        for index, (transmittance, contribution) in enumerate(zip(self.T, self.C, strict=True)):
            back = background_values[0] if len(background_values) == 1 else background_values[index]
            family = _family(transmittance)
            if family != "native":
                back = _coerce_transport_operand(back, family)
            values.append(_typed_add(contribution, _typed_mul(transmittance, back)))
        return (
            values[0]
            if self.channels == 1
            else np.asarray(values, dtype=object if any(_family(value) != "native" for value in values) else None)
        )

    def compose(self, back: TransportRGB) -> TransportRGB:
        return transport_compose_rgb(self, back)

    def to_transport(self) -> v02.Transport:
        if self.channels != 1:
            raise ValueError("to_transport is scalar-only; use TransportRGB for RGB")
        return v02.Transport(
            self.T[0],
            self.C[0],
            error_bound_T=self.error_bound_T,
            error_bound_C=self.error_bound_C,
            ledger=self.ledger,
        )

    @classmethod
    def from_transport(cls, value: v02.Transport) -> TransportRGB:
        if not isinstance(value, v02.Transport):
            raise TypeError("from_transport expects a tensor.v02 Transport")
        return cls(value.T, value.C, value.error_bound_T, value.error_bound_C, value.ledger)


RGBTransport = TransportRGB


_VOLUME_OPERATION = "volume"
_VOLUME_EXECUTION_CLASS = "volume_cpu_reference_transport"
_VOLUME_CALLABLE = "spf.renderer.volume.render_volume"


def _backend_request_name(backend: str | SFRBackend) -> str:
    requested = getattr(backend, "name", backend)
    return str(requested).strip().lower().replace("-", "_")


def _validate_volume_backend(selected_backend: SFRBackend) -> None:
    """Reject native selections that have no volume terminal.

    The volume implementation is a portable Python/NumPy semantic path.  A
    selected native adapter must advertise an actual volume operation before
    it can be used, otherwise a request such as CUDA would silently execute
    on the host while reporting the adapter's hardware class.
    """

    selected_backend.require_available(_VOLUME_OPERATION)
    capabilities = selected_backend.capabilities
    if capabilities.native and not capabilities.supports_operation(_VOLUME_OPERATION):
        reason = capabilities.unsupported_reasons.get(_VOLUME_OPERATION)
        detail = f": {reason}" if reason else ""
        raise NotImplementedError(
            f"SFR backend {capabilities.name!r} does not support {_VOLUME_OPERATION!r}{detail}"
        )


def _volume_provenance(
    selected_backend: SFRBackend,
    *,
    requested_backend: str,
    actual_callable: str,
) -> dict[str, Any]:
    """Describe selected adapter metadata separately from the executed path."""

    capabilities = selected_backend.capabilities
    return {
        "requested_backend": requested_backend,
        "selected_backend": selected_backend.name,
        "selected_backend_native": bool(capabilities.native),
        "selected_backend_execution_class": capabilities.execution_class,
        # Compatibility keys describe the selected adapter; actual_* and
        # hardware_native describe the callable that produced this report.
        "backend_native": bool(capabilities.native),
        "backend_execution_class": capabilities.execution_class,
        "execution_class": _VOLUME_EXECUTION_CLASS,
        "hardware_native": False,
        "actual_execution_class": _VOLUME_EXECUTION_CLASS,
        "actual_callable": actual_callable,
        "backend_execution_provenance": _VOLUME_EXECUTION_CLASS,
    }


def transport_identity(channels: int = 3, *, representation: str = "native") -> TransportRGB:
    count = int(channels)
    if count <= 0:
        raise ValueError("transport identity needs a positive channel count")
    name = str(representation).lower().replace("-", "_")
    if name in {"log", "log_pos", "spf", "typed"}:
        one: Any = v02.LogPos(0)
        zero: Any = v02.LogPos.zero()
    else:
        one, zero = 1.0, 0.0
    return TransportRGB((one,) * count, (zero,) * count, optical_depth=(0.0,) * count)


def _coerce_transport(value: Any) -> TransportRGB:
    if isinstance(value, TransportRGB):
        return value
    if isinstance(value, v02.Transport):
        return TransportRGB.from_transport(value)
    converter = getattr(value, "to_transport", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, TransportRGB):
            return converted
        if isinstance(converted, v02.Transport):
            return TransportRGB.from_transport(converted)
        raise TypeError("transport conversion did not return a supported transport type")
    if isinstance(value, Mapping):
        t = value.get("T", value.get("transmittance"))
        c = value.get("C", value.get("contribution"))
        if t is None or c is None:
            raise ValueError("transport mapping requires T/transmittance and C/contribution")
        return TransportRGB(t, c)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return TransportRGB(value[0], value[1])
    raise TypeError(f"expected a transport segment, received {type(value).__name__}")


def _guarded_transport_compose(first: TransportRGB, second: TransportRGB) -> TransportRGB:
    """Compose transports whose transmittances carry exact optical depth.

    The affine law is still ordered and channel-wise.  The only special case
    is the transmittance product: adding the certified tau values before the
    exponential keeps a long heterogeneous ray from repeatedly rounding a
    sub-q8 increment to zero.  Contributions stay in the ordinary native
    value domain because they are already at the material/emission boundary.
    """

    if first.optical_depth is None or second.optical_depth is None:
        raise ValueError("guarded optical-depth composition requires certificates on both operands")
    if first.channels != second.channels:
        raise ValueError("transport channel counts must match")
    first_tau = np.asarray(first.optical_depth, dtype=np.float64)
    second_tau = np.asarray(second.optical_depth, dtype=np.float64)
    total_tau = np.asarray(
        [math.fsum((float(left), float(right))) for left, right in zip(first_tau, second_tau, strict=True)],
        dtype=np.float64,
    )
    exact_t = np.asarray(
        [0.0 if math.isinf(value) else math.exp(-float(value)) for value in total_tau], dtype=np.float64
    )
    # T enters only through its magnitude envelope, so the transmittance
    # channels stay a real envelope instead of a Cartesian pair plane.
    first_t_magnitude = np.asarray([abs(_numeric(value)) for value in first.T], dtype=np.float64)
    # Same carrier-preserving read as ``TransportRGB.numeric``: a real emission
    # spectrum stays real instead of being widened to a Cartesian pair plane.
    first_c = np.asarray([_numeric(value) for value in first.C])
    second_c = np.asarray([_numeric(value) for value in second.C])
    first_transmittance = np.asarray(
        [0.0 if math.isinf(value) else math.exp(-float(value)) for value in first_tau], dtype=np.float64
    )
    exact_c = first_c + first_transmittance * second_c
    # Native arithmetic has no SPF log quantization here, but retain a
    # conservative roundoff certificate so reports distinguish this guarded
    # path from an exact symbolic identity.
    scale_t = max(float(np.max(np.abs(exact_t))), 1.0) if exact_t.size else 1.0
    scale_c = max(float(np.max(np.abs(exact_c))), 1.0) if exact_c.size else 1.0
    eps = np.finfo(np.float64).eps
    tau_scale = max(
        (abs(float(value)) for value in total_tau if math.isfinite(float(value))),
        default=1.0,
    )
    own_t = eps * max(scale_t, tau_scale)
    own_c = eps * scale_c
    first_t_abs = float(np.max(first_t_magnitude)) if first_t_magnitude.size else 0.0
    second_t_magnitude = np.asarray([abs(_numeric(value)) for value in second.T], dtype=np.float64)
    second_t_abs = float(np.max(second_t_magnitude)) if second_t_magnitude.size else 0.0
    second_c_abs = float(np.max(np.abs(second_c))) if second_c.size else 0.0
    total_t = (
        own_t
        + second_t_abs * first.error_bound_T
        + first_t_abs * second.error_bound_T
        + first.error_bound_T * second.error_bound_T
    )
    total_c = (
        own_c
        + first.error_bound_C
        + second_c_abs * first.error_bound_T
        + first_t_abs * second.error_bound_C
        + first.error_bound_T * second.error_bound_C
    )
    ledger = _transport_ledger((first, second), total_t + total_c)
    transmittance: tuple[float, ...] = tuple(float(value) for value in exact_t)
    contribution: tuple[complex | float, ...] = tuple(
        value.real if abs(value.imag) <= 1.0e-15 else value for value in exact_c
    )
    return TransportRGB(
        transmittance,
        contribution,
        total_t,
        total_c,
        ledger,
        optical_depth=tuple(float(value) for value in total_tau),
    )


def _guarded_transport_tree(items: Sequence[TransportRGB]) -> TransportRGB:
    """Reduce exact-τ carriers in one terminal consumer."""

    if not items:
        return transport_identity(3)
    from spf.renderer.transport_terminal import reduce_guarded_optical_depth

    tau = np.asarray([item.optical_depth for item in items], dtype=np.float64)
    # ``reduce_guarded_optical_depth`` normalizes the contribution carrier at
    # its own boundary; casting here built the same plane twice.
    contribution = np.asarray([[_numeric(value) for value in item.C] for item in items])
    error_t = np.asarray([item.error_bound_T for item in items], dtype=np.float64)
    error_c = np.asarray([item.error_bound_C for item in items], dtype=np.float64)
    return cast(
        TransportRGB,
        reduce_guarded_optical_depth(
            tau,
            contribution,
            error_t,
            error_c,
            ledgers=tuple(item.ledger for item in items),
        ),
    )


def transport_compose_rgb(front: Any, back: Any) -> TransportRGB:
    """Compose two transports in front-to-back order."""

    first, second = _coerce_transport(front), _coerce_transport(back)
    if first.channels != second.channels:
        raise ValueError("transport channel counts must match")
    if first.optical_depth is not None and second.optical_depth is not None:
        return _guarded_transport_compose(first, second)
    transmittance: list[Any] = []
    contribution: list[Any] = []
    first_t, first_c = first.numeric()
    second_t, second_c = second.numeric()
    exact_t_values = first_t * second_t
    exact_c_values = first_c + first_t * second_c
    for left_t, left_c, right_t, right_c in zip(first.T, first.C, second.T, second.C, strict=True):
        product_t = _typed_mul(left_t, right_t)
        product_c = _typed_mul(left_t, right_c)
        transmittance.append(product_t)
        contribution.append(_typed_add(left_c, product_c))
    # Read back in the composed values' own carrier, as ``numeric()`` does, so a
    # typed real transport is compared without a Cartesian widening.
    actual_t = np.asarray([_numeric(value) for value in transmittance])
    actual_c = np.asarray([_numeric(value) for value in contribution])
    own_t = float(np.max(np.abs(actual_t - exact_t_values))) if actual_t.size else 0.0
    own_c = float(np.max(np.abs(actual_c - exact_c_values))) if actual_c.size else 0.0
    scale_t = max(float(np.max(np.abs(exact_t_values))), 1.0)
    scale_c = max(float(np.max(np.abs(exact_c_values))), 1.0)
    if own_t == 0.0 and _family(first.T[0]) == "native" and _family(second.T[0]) == "native":
        own_t = np.finfo(np.float64).eps * scale_t
    if own_c == 0.0 and _family(first.C[0]) == "native" and _family(second.C[0]) == "native":
        own_c = np.finfo(np.float64).eps * scale_c
    first_t_abs = np.max(np.abs(first_t)) if first_t.size else 0.0
    second_t_abs = np.max(np.abs(second_t)) if second_t.size else 0.0
    second_c_abs = np.max(np.abs(second_c)) if second_c.size else 0.0
    total_t = (
        own_t
        + second_t_abs * first.error_bound_T
        + first_t_abs * second.error_bound_T
        + first.error_bound_T * second.error_bound_T
    )
    total_c = (
        own_c
        + first.error_bound_C
        + second_c_abs * first.error_bound_T
        + first_t_abs * second.error_bound_C
        + first.error_bound_T * second.error_bound_C
    )
    stage = v02.ErrorLedger(reducer=total_t + total_c)
    ledger = v02.ErrorLedger(
        reducer=total_t + total_c,
        stages=first.ledger.stages + second.ledger.stages + (stage,),
    )
    return TransportRGB(transmittance, contribution, total_t, total_c, ledger)


def _native_transport_tree(items: Sequence[TransportRGB]) -> TransportRGB:
    """Batch each ordered tree level, carrying per-node error bounds."""
    t = np.asarray([item.T for item in items])
    c = np.asarray([item.C for item in items])
    et = np.asarray([item.error_bound_T for item in items], dtype=np.float64)
    ec = np.asarray([item.error_bound_C for item in items], dtype=np.float64)
    if t.ndim != 2 or c.shape != t.shape:
        raise ValueError("transport channel counts must match")
    stages = tuple(stage for item in items for stage in item.ledger.stages)
    return _reduce_transport_arrays(t, c, et, ec, stages)


def _reduce_transport_arrays(
    t: np.ndarray, c: np.ndarray, et: np.ndarray, ec: np.ndarray, stages: tuple[Any, ...] = ()
) -> TransportRGB:
    eps = np.finfo(np.float64).eps
    while t.shape[0] > 1:
        pairs = t.shape[0] // 2
        a, b = slice(0, pairs * 2, 2), slice(1, pairs * 2, 2)
        ta, tb, ca, cb = t[a], t[b], c[a], c[b]
        nt, nc = ta * tb, ca + ta * cb
        at, bt, bc = (np.max(np.abs(x), axis=1) for x in (ta, tb, cb))
        net = eps * np.maximum(1, np.max(np.abs(nt), axis=1)) + bt * et[a] + at * et[b] + et[a] * et[b]
        nec = eps * np.maximum(1, np.max(np.abs(nc), axis=1)) + ec[a] + bc * et[a] + at * ec[b] + et[a] * ec[b]
        if t.shape[0] % 2:
            nt, nc = np.concatenate((nt, t[-1:])), np.concatenate((nc, c[-1:]))
            net, nec = np.r_[net, et[-1]], np.r_[nec, ec[-1]]
        t, c, et, ec = nt, nc, net, nec
    ledger = v02.ErrorLedger(reducer=float(et[0] + ec[0]), stages=stages)
    return TransportRGB(t[0], c[0], float(et[0]), float(ec[0]), ledger)


def transport_tree(segments: Sequence[Any], *, channels: int | None = None) -> TransportRGB:
    """Reduce an ordered sequence through a pairwise associative tree."""

    items = [_coerce_transport(value) for value in segments]
    if not items:
        return transport_identity(3 if channels is None else channels)
    if len(items) > 1 and all(item.optical_depth is not None for item in items):
        return _guarded_transport_tree(items)
    if len(items) > 1 and all(_family(x) == "native" for item in items for x in (*item.T, *item.C)):
        return _native_transport_tree(items)
    if len(items) > 1 and all(isinstance(x, v02.LogPos) and not x.is_nan for item in items for x in (*item.T, *item.C)):
        return cast(TransportRGB, PreparedVolume(items).execute(output_kind="transport"))
    current = items
    while len(current) > 1:
        next_level: list[TransportRGB] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
            else:
                next_level.append(transport_compose_rgb(current[index], current[index + 1]))
        current = next_level
    return current[0]


ordered_transport_reduce = transport_tree


def transport_scan(segments: Sequence[Any], *, inclusive: bool = True) -> tuple[TransportRGB, ...]:
    """Return order-preserving inclusive or exclusive transport prefixes."""

    items = [_coerce_transport(value) for value in segments]
    if not items:
        return ()
    result: list[TransportRGB] = []
    prefix = transport_identity(items[0].channels)
    for item in items:
        if inclusive:
            prefix = transport_compose_rgb(prefix, item)
            result.append(prefix)
        else:
            result.append(prefix)
            prefix = transport_compose_rgb(prefix, item)
    return tuple(result)


ordered_transport_scan = transport_scan


def reduce_transport_rgb(
    segments: Sequence[Any],
    *,
    channels: int | None = None,
    backend: str | SFRBackend = "reference",
    return_report: bool = False,
) -> Any:
    selected_backend = get_backend(backend)
    _validate_volume_backend(selected_backend)
    total = transport_tree(segments, channels=channels)
    provenance = _volume_provenance(
        selected_backend,
        requested_backend=_backend_request_name(backend),
        actual_callable="spf.renderer.volume.reduce_transport_rgb",
    )
    report = _volume_report(
        backend=selected_backend.name,
        segments=len(segments),
        bricks_considered=0,
        bricks_culled=0,
        early_terminated=False,
        quadrature_samples=0,
        method="transport_tree",
        ledger=total.ledger,
        diagnostics={"ordered": True, "channels": total.channels, **provenance},
    )
    return (total, report) if return_report else total


@dataclass(frozen=True, slots=True)
class OpticalDepthSegment:
    """One typed optical-depth segment and its affine transport."""

    tau: OpticalDepth | float
    contribution: Any = 0.0
    transmission: Any | None = None
    contribution_bound: float | None = None
    error_bound: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger, kw_only=True)
    _implicit_transmission: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        optical_depth = self.tau if isinstance(self.tau, OpticalDepth) else OpticalDepth(self.tau)
        implicit_transmission = self.transmission is None
        transmission = optical_depth.transmittance if implicit_transmission else self.transmission
        values = _channels(transmission)
        for value in values:
            if isinstance(value, v02.LogPos) and value.is_nan:
                raise ValueError("segment transmittance cannot be NaN")
        if self.contribution_bound is not None:
            _finite_nonnegative("segment contribution_bound", self.contribution_bound, allow_inf=True)
        _finite_nonnegative("segment error_bound", self.error_bound, allow_inf=True)
        if not isinstance(self.ledger, v02.ErrorLedger):
            raise TypeError("segment ledger must be an ErrorLedger")
        ledger = self.ledger
        if ledger.total_bound == 0.0 and self.error_bound:
            ledger = ledger.plus(v02.ErrorLedger(materialization=self.error_bound))
        object.__setattr__(self, "tau", optical_depth)
        object.__setattr__(self, "transmission", transmission)
        object.__setattr__(self, "error_bound", float(self.error_bound))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "ledger", ledger)
        object.__setattr__(self, "_implicit_transmission", implicit_transmission)

    @property
    def optical_depth(self) -> OpticalDepth:
        return cast(OpticalDepth, self.tau)

    @property
    def T(self) -> Any:  # noqa: N802 - transport-map notation
        return self.transmission

    @property
    def C(self) -> Any:  # noqa: N802 - transport-map notation
        return self.contribution

    @property
    def value_kind(self) -> str:
        return "OPTICAL_DEPTH"

    @property
    def log_transmittance(self) -> v02.LogPos:
        return self.optical_depth.log_transmittance

    @property
    def transport(self) -> TransportRGB:
        bound = self.contribution_bound
        if bound is None:
            bound = _max_abs(self.contribution)
        return TransportRGB(
            self.transmission,
            self.contribution,
            self.error_bound,
            self.error_bound,
            self.ledger,
            optical_depth=(float(self.optical_depth),) * len(_channels(self.transmission))
            if self._implicit_transmission
            else None,
        )

    def to_transport(self) -> TransportRGB:
        return self.transport

    @classmethod
    def from_constant_density(
        cls,
        sigma_t: float,
        length: float,
        emission: Any = 0.0,
        *,
        source_is_density: bool = True,
        error_bound: float = 0.0,
        ledger: v02.ErrorLedger | None = None,
    ) -> OpticalDepthSegment:
        sigma = _finite_nonnegative("sigma_t", sigma_t)
        distance = _finite_nonnegative("segment length", length)
        tau = sigma * distance
        one_minus_t = -math.expm1(-tau) if math.isfinite(tau) else 1.0
        scale = distance if sigma == 0.0 else (one_minus_t / sigma if source_is_density else one_minus_t)
        source_values = _channels(emission)
        contribution = source_values[0] * scale if len(source_values) == 1 else np.asarray(source_values) * scale
        bound = _max_abs(contribution)
        return cls(
            OpticalDepth(tau),
            contribution,
            contribution_bound=bound,
            error_bound=error_bound,
            ledger=v02.ErrorLedger() if ledger is None else ledger,
            metadata={
                "constant_density": True,
                "sigma_t": sigma,
                "length": distance,
                "source_is_density": source_is_density,
            },
        )

    @classmethod
    def from_density(
        cls,
        density: Any,
        start: Any,
        end: Any,
        emission: Any = 0.0,
        *,
        quadrature_order: int = 8,
        source_is_density: bool = True,
    ) -> OpticalDepthSegment:
        result = quadrature_optical_depth(density, start, end, order=quadrature_order, return_result=True)
        if not isinstance(result, QuadratureResult):  # pragma: no cover - literal True contract
            raise RuntimeError("quadrature did not return its certificate")
        _, _, length = _ray_points(start, end)
        mean_density = 0.0 if length == 0.0 else result.value / length
        return cls.from_constant_density(
            mean_density,
            length,
            emission,
            source_is_density=source_is_density,
            error_bound=result.error_bound,
            ledger=result.ledger,
        )


VolumeSegment = OpticalDepthSegment


def constant_density_segment(*args: Any, **kwargs: Any) -> OpticalDepthSegment:
    return OpticalDepthSegment.from_constant_density(*args, **kwargs)


def make_density_segment(
    density: Any,
    start: Any,
    end: Any,
    emission: Any = 0.0,
    **kwargs: Any,
) -> OpticalDepthSegment:
    return OpticalDepthSegment.from_density(density, start, end, emission, **kwargs)


def constant_density_transport(
    sigma_t: float,
    length: float,
    emission: Any = 0.0,
    *,
    source_is_density: bool = True,
) -> TransportRGB:
    return constant_density_segment(sigma_t, length, emission, source_is_density=source_is_density).transport


def constant_density_reference(
    sigma_t: float,
    length: float,
    emission: Any = 0.0,
    background: Any = 0.0,
    *,
    source_is_density: bool = True,
    return_transport: bool = False,
) -> Any:
    transport = constant_density_transport(sigma_t, length, emission, source_is_density=source_is_density)
    return transport if return_transport else transport.apply(background)


render_constant_density = constant_density_reference
constant_density_closed_form = constant_density_reference


@dataclass(frozen=True, slots=True)
class VolumeBrick:
    """A bounded density brick used for certified culling."""

    bounds: Region | Any
    density: Any = 0.0
    emission: Any = 0.0
    length: float | None = None
    max_density: float | None = None
    contribution_bound: float | None = None
    start: Any | None = None
    end: Any | None = None
    source_is_density: bool = True
    brick_id: str | int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        region = self.bounds if isinstance(self.bounds, Region) else _region_from_bounds(self.bounds)
        if self.length is not None:
            _finite_nonnegative("brick length", self.length)
        if self.max_density is not None:
            _finite_nonnegative("brick max_density", self.max_density, allow_inf=True)
        if self.contribution_bound is not None:
            _finite_nonnegative("brick contribution_bound", self.contribution_bound, allow_inf=True)
        object.__setattr__(self, "bounds", region)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def region(self) -> Region:
        return self.bounds

    @property
    def dimension(self) -> int:
        return self.bounds.dimension

    @property
    def path_length(self) -> float:
        if self.length is not None:
            return float(self.length)
        return float(np.linalg.norm(np.asarray(self.bounds.extent, dtype=np.float64)))

    @property
    def density_bound(self) -> float:
        if self.max_density is not None:
            return float(self.max_density)
        source = self.density
        if isinstance(source, DensityField):
            source = source.source
        if isinstance(source, (int, float, np.integer, np.floating)):
            return _finite_nonnegative("brick density", source)
        if isinstance(source, v02.GaussianChirpAtom):
            return float(v02.atom_bound(source, (self.bounds.lower, self.bounds.upper)))
        if isinstance(source, v02.AtomMixture):
            return float(
                math.fsum(float(v02.atom_bound(atom, (self.bounds.lower, self.bounds.upper))) for atom in source.atoms)
            )
        if isinstance(source, DenseDensity):
            return float(np.max(source.values))
        if isinstance(source, np.ndarray):
            return float(np.max(_validated_density(source)))
        return math.inf

    @property
    def upper_bound(self) -> float:
        if self.contribution_bound is not None:
            return float(self.contribution_bound)
        source_bound = _max_abs(self.emission)
        return self.density_bound * self.path_length * (source_bound if source_bound else 1.0)

    def cull_bound(self) -> float:
        return self.upper_bound


def _region_from_bounds(bounds: Any) -> Region:
    if isinstance(bounds, Mapping):
        lower = bounds.get("lower", bounds.get("lo"))
        upper = bounds.get("upper", bounds.get("hi"))
        if lower is None or upper is None:
            raise ValueError("volume bounds mapping requires lower/upper or lo/hi")
        return Region(tuple(lower), tuple(upper))
    if isinstance(bounds, (tuple, list)) and len(bounds) == 2:
        return Region(tuple(bounds[0]), tuple(bounds[1]))
    array = np.asarray(bounds, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 2:
        return Region(tuple(array[:, 0]), tuple(array[:, 1]))
    if array.ndim == 2 and array.shape[0] == 2:
        return Region(tuple(array[0]), tuple(array[1]))
    raise ValueError("volume bounds must be (lower, upper), (d,2), or (2,d)")


@dataclass(frozen=True, slots=True)
class BrickCullCertificate:
    """Conservative evidence for bricks omitted before ray sampling."""

    considered: int
    retained: int
    culled: int
    dropped_bound: float
    budget: float
    certified: bool
    reason: str = "bound_sum_within_budget"
    ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)

    @property
    def error_bound(self) -> float:
        return self.dropped_bound

    @property
    def absolute_error_bound(self) -> float:
        return self.dropped_bound


@dataclass(frozen=True, slots=True)
class BrickCullResult:
    """Kept/cut bricks plus their immutable culling certificate."""

    kept: tuple[VolumeBrick, ...]
    culled: tuple[VolumeBrick, ...]
    certificate: BrickCullCertificate

    def __iter__(self) -> Iterator[tuple[VolumeBrick, ...] | BrickCullCertificate]:
        yield self.kept
        yield self.certificate


def cull_bricks_result(bricks: Sequence[VolumeBrick], error_budget: float) -> BrickCullResult:
    items = tuple(brick if isinstance(brick, VolumeBrick) else VolumeBrick(**brick) for brick in bricks)
    budget = _finite_nonnegative("brick cull error_budget", error_budget, allow_inf=True)
    kept: list[VolumeBrick] = []
    culled: list[VolumeBrick] = []
    dropped_bound = 0.0
    for brick in items:
        bound = float(brick.upper_bound)
        if bound == 0.0 or (math.isfinite(bound) and dropped_bound + bound <= budget):
            culled.append(brick)
            dropped_bound += bound
        else:
            kept.append(brick)
    certificate = BrickCullCertificate(
        len(items),
        len(kept),
        len(culled),
        dropped_bound,
        budget,
        bool(math.isfinite(dropped_bound) and dropped_bound <= budget + np.finfo(np.float64).eps * max(1.0, budget)),
        ledger=v02.ErrorLedger(pruning=dropped_bound),
    )
    return BrickCullResult(tuple(kept), tuple(culled), certificate)


def cull_bricks(
    bricks: Sequence[VolumeBrick],
    error_budget: float,
    *,
    return_certificate: bool = False,
) -> Any:
    result = cull_bricks_result(bricks, error_budget)
    return (list(result.kept), result.certificate) if return_certificate else list(result.kept)


def _segment_bound(value: Any) -> float:
    if isinstance(value, OpticalDepthSegment):
        if value.contribution_bound is not None:
            return float(value.contribution_bound)
        return _max_abs(value.contribution)
    if isinstance(value, TransportRGB):
        return _max_abs(value.C)
    return _max_abs(_coerce_transport(value).C)


def early_terminate_transport(
    segments: Sequence[Any],
    *,
    threshold: float,
    background_bound: float = 1.0,
) -> tuple[tuple[Any, ...], float, bool]:
    """Skip a suffix only when its attenuated contribution has a bound."""

    limit = _finite_nonnegative("early termination threshold", threshold, allow_inf=True)
    back_bound = _finite_nonnegative("background bound", background_bound, allow_inf=True)
    items = tuple(segments)
    if not items:
        return (), 0.0, False
    prefix_t = 1.0
    for index, item in enumerate(items):
        segment = _coerce_transport(item)
        prefix_t *= _max_abs(segment.T)
        suffix = items[index + 1 :]
        tail = sum((_segment_bound(value) for value in suffix), 0.0) + back_bound
        impact = prefix_t * tail
        if math.isfinite(impact) and impact <= limit:
            return items[: index + 1], impact, True
    return items, 0.0, False


def _volume_report(
    *,
    backend: str,
    segments: int,
    bricks_considered: int,
    bricks_culled: int,
    early_terminated: bool,
    quadrature_samples: int,
    method: str,
    ledger: v02.ErrorLedger,
    diagnostics: Mapping[str, Any],
) -> RenderReport:
    metrics = RenderMetrics(
        backend=backend,
        workload="volume",
        representation="affine_transport",
        precision="balanced",
        potential_elements=segments + bricks_considered,
        materialized_elements=segments,
        atoms_considered=bricks_considered,
        atoms_pruned=bricks_culled,
        transport_compositions=max(0, segments - 1),
        reduction_operations=1 if segments else 0,
        field_operations=max(1, segments),
    )
    data = dict(diagnostics)
    data.update(
        {
            "bricks_considered": bricks_considered,
            "bricks_culled": bricks_culled,
            "early_terminated": early_terminated,
            "quadrature_samples": quadrature_samples,
            "method": method,
            "error_bound": ledger.total_bound,
        }
    )
    return RenderReport(
        metrics,
        ledger=ledger,
        passes=(
            CompilerPassRecord("density_to_optical_depth", bricks_considered, segments),
            CompilerPassRecord("ordered_transport_tree", segments, 1),
        ),
        fallbacks=() if method != "dense_fallback" else ("dense_fallback",),
        capabilities_used=("optical_depth", "affine_transport", "ordered_tree"),
        diagnostics=data,
    )


def heterogeneous_segments(
    density: Any,
    start: Any,
    end: Any,
    emission: Any = 0.0,
    *,
    steps: int = 32,
    quadrature_order: int = 4,
    source_is_density: bool = True,
    return_report: bool = False,
) -> Any:
    """Create local constant-density segments for a heterogeneous ray."""

    count = int(steps)
    if count <= 0:
        raise ValueError("heterogeneous volume steps must be positive")
    left, right, length = _ray_points(start, end)
    if length == 0.0:
        result: tuple[OpticalDepthSegment, ...] = ()
        return (result, QuadratureResult(0.0, 0.0, 0, quadrature_order)) if return_report else result
    source_error_ledger = _density_error_ledger(density)
    source_error = source_error_ledger.total_bound
    density = _prepared_density(density)
    order = int(quadrature_order)
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    high_order = max(order + 1, 2 * order)
    origins = left + (np.arange(count) / count)[:, None] * (right - left)
    direction = (right - left) / count
    segment_length = length / count
    estimates = []
    for q in (order, high_order):
        nodes, weights = _quadrature_rule(q)
        points = origins[:, None, :] + 0.5 * (nodes[None, :, None] + 1) * direction
        sampled = np.asarray(sample_density(density, points.reshape(-1, left.size))).reshape(count, q)
        estimates.append(0.5 * segment_length * (sampled @ weights))
    low, high = estimates
    quadrature_errors = np.abs(high - low) + np.finfo(np.float64).eps * np.maximum(1, np.abs(high))
    source_errors = np.full(count, segment_length * source_error, dtype=np.float64)
    errors = quadrature_errors + source_errors
    constant = _constant_density(density)
    if constant:
        quadrature_errors[:] = 0.0
        errors[:] = 0
    segments = [
        OpticalDepthSegment.from_constant_density(
            float(tau / segment_length),
            segment_length,
            emission,
            source_is_density=source_is_density,
            error_bound=float(error) * max(1.0, _max_abs(emission)),
        )
        for tau, error in zip(high, errors, strict=True)
    ]
    total_samples = 0 if constant else count * (order + high_order)
    total_error = float(np.sum(errors))
    report = QuadratureResult(
        sum(float(segment.tau) for segment in segments),
        total_error,
        total_samples,
        quadrature_order,
        "piecewise_gauss_legendre",
        quadrature_error_bound=float(np.sum(quadrature_errors)),
        source_error_bound=float(np.sum(source_errors)),
        source_ledger=_scale_error_ledger(source_error_ledger, length),
    )
    return (tuple(segments), report) if return_report else tuple(segments)


def render_heterogeneous(
    density: Any,
    start: Any,
    end: Any,
    emission: Any = 0.0,
    background: Any = 0.0,
    *,
    steps: int = 32,
    quadrature_order: int = 4,
    source_is_density: bool = True,
    early_termination: float | None = None,
    backend: str | SFRBackend = "reference",
    return_report: bool = False,
) -> Any:
    segments, quadrature = heterogeneous_segments(
        density,
        start,
        end,
        emission,
        steps=steps,
        quadrature_order=quadrature_order,
        source_is_density=source_is_density,
        return_report=True,
    )
    value = render_volume(
        segments,
        background=background,
        early_termination=early_termination,
        backend=backend,
        return_report=return_report,
    )
    if not return_report:
        return value
    rendered, report = value
    diagnostics = dict(report.diagnostics)
    diagnostics.update(
        {
            "constant_density": False,
            "heterogeneous": True,
            "quadrature_error": quadrature.error_bound,
            "quadrature_samples": quadrature.samples,
        }
    )
    report = RenderReport(
        report.metrics,
        ledger=report.ledger.plus(quadrature.ledger),
        passes=report.passes,
        decisions=report.decisions,
        fallbacks=report.fallbacks,
        capabilities_used=(*report.capabilities_used, "quadrature"),
        diagnostics=diagnostics,
    )
    return rendered, report


heterogeneous_volume_reference = render_heterogeneous


# ---------------------------------------------------------------------------
# Word-plane volume terminal (design §25 on LOG_POS planes)
# ---------------------------------------------------------------------------


def _density_block(density: Any) -> tuple[Any, Any]:
    """Return ``(block_view, owner_field)`` for a block-backed density source."""

    from spf.renderer.carriers import as_block_view

    if isinstance(density, DensityField):
        density = density.source
    owner = density if isinstance(density, Field) else None
    payload = density.representation.payload if owner is not None else density
    if owner is not None:
        from spf.renderer.normalization import normalize_field

        normalized = normalize_field(owner).field
        if normalized.operations:
            pending = ", ".join(operation.op for operation in normalized.operations)
            raise TypeError(f"block density has unsupported deferred operations at the volume boundary: {pending}")
        if normalized.representation.residual is not None:
            raise TypeError("block density residuals have no word-plane volume consumer; use render_volume")
        payload = normalized.representation.payload
    block = as_block_view(payload)
    if block is None:
        raise TypeError("render_volume_planes requires a TensorBlock/SPFBlock density carrier")
    return block, owner


def _corner_density_codes(block: Any, points: np.ndarray, *, boundary: str) -> tuple[np.ndarray, np.ndarray]:
    """Lattice-corner LOG_POS codes and multilinear weights for query ``points``.

    Words are read at the touched corners through ``sample_words`` and never
    decoded; the interpolation weights are geometry (float) and are encoded
    as positive factors.  Returns ``(codes (P, corners), weights (P, corners))``.
    """

    from spf.tensor.v02_planes import LogSignedPlane

    dimension = len(block.shape)
    extent = np.asarray(block.shape, dtype=np.int64) - 1
    inside = np.all((points >= 0.0) & (points <= extent), axis=1)
    clipped = np.clip(points, 0.0, extent)
    lower = np.floor(clipped).astype(np.int64)
    upper = np.minimum(lower + 1, extent)
    frac = clipped - lower
    corners = list(np.ndindex((2,) * dimension))
    codes = np.empty((points.shape[0], len(corners)), dtype=np.int64)
    weights = np.empty((points.shape[0], len(corners)), dtype=np.float64)
    for column, corner in enumerate(corners):
        indices = tuple(np.where(corner[axis], upper[:, axis], lower[:, axis]) for axis in range(dimension))
        linear = np.ravel_multi_index(indices, tuple(int(value) for value in block.shape))
        words = np.asarray(block.sample_words(linear, phase=False))
        projected = np.asarray(block.project_words(words, linear))
        if projected.dtype.kind not in "iu":
            raise TypeError("block density header offsets are not closed over words; use render_volume")
        signed = LogSignedPlane.from_words(np.asarray(projected, dtype=np.uint32))
        if np.any(signed.is_nan) or np.any(signed.sign < 0):
            raise ValueError("block density words must be finite and non-negative")
        codes[:, column] = np.where(signed.sign == 0, v02.SPF32_L_ZERO, signed.L.astype(np.int64))
        factor = np.ones(points.shape[0], dtype=np.float64)
        for axis, bit in enumerate(corner):
            factor *= frac[:, axis] if bit else 1.0 - frac[:, axis]
        weights[:, column] = factor
    if boundary == "zero":
        weights[~inside] = 0.0
    return codes, weights


def render_volume_planes(
    density: Any,
    start: Any,
    end: Any,
    emission: Any = 0.0,
    background: Any = 0.0,
    *,
    steps: int = 32,
    quadrature_order: int = 4,
    source_is_density: bool = True,
    boundary: str = "zero",
    output_kind: str = "native",
    backend: str | SFRBackend = "reference",
    return_report: bool = False,
) -> Any:
    """Heterogeneous ray on word planes: density words → τ plane → transmittance ROM → ordered composition.

    The block density is read at the lattice corners each quadrature node
    touches (packed words, never decoded).  Multilinear and Gauss--Legendre
    weights are encoded as positive factors and every optical depth is a
    positive pivot reduction (:func:`spf.tensor.v02_planes.reduce_logpos`) of
    ``weight * density`` terms.  Transmittance codes come from the optical-depth
    ROM in :mod:`spf.renderer.transport_terminal`, the emission scale
    ``(1 - T) / sigma`` from ``LogPosPlane.log1m``/``div``, and the ordered
    front-to-back composition is :func:`spf.renderer.transport_terminal.reduce_logpos`
    on the int code planes.  The certificate uses the same order-doubling
    envelope as :func:`heterogeneous_segments`, evaluated through the
    magnitude ROM.
    """

    from spf.renderer.transport_terminal import display_values, optical_depth_to_transmittance, reduce_logpos
    from spf.tensor.v02_planes import LogPosPlane
    from spf.tensor.v02_planes import reduce_logpos as reduce_logpos_plane

    requested_backend = _backend_request_name(backend)
    selected_backend = get_backend(backend)
    _validate_volume_backend(selected_backend)
    count = int(steps)
    if count <= 0:
        raise ValueError("heterogeneous volume steps must be positive")
    order = int(quadrature_order)
    if order <= 0:
        raise ValueError("quadrature order must be positive")
    selected_boundary = str(boundary).lower()
    if selected_boundary not in {"zero", "clamp"}:
        raise ValueError("block density boundary must be 'zero' or 'clamp'")
    emission_values = np.asarray(_channels(emission), dtype=np.float64).reshape(-1)
    if np.any(emission_values < 0.0) or not np.all(np.isfinite(emission_values)):
        raise ValueError("render_volume_planes requires finite non-negative emission")
    channels = max(3, emission_values.size) if emission_values.size != 1 else 3
    if emission_values.size == 1:
        emission_values = np.repeat(emission_values, channels)
    block, _owner = _density_block(density)
    source_ledger = _density_error_ledger(density)
    left, right, length = _ray_points(start, end)
    if left.size != len(block.shape):
        raise ValueError("ray dimension must match the block extent")
    origins = left + (np.arange(count) / count)[:, None] * (right - left)
    direction = (right - left) / count
    segment_length = length / count
    high_order = max(order + 1, 2 * order)
    tau_planes: list[LogPosPlane] = []
    counters = v02.PerformanceCounters()
    for q in (order, high_order):
        nodes, weights = _quadrature_rule(q)
        points = (origins[:, None, :] + 0.5 * (nodes[None, :, None] + 1.0) * direction).reshape(-1, left.size)
        codes, corner_weights = _corner_density_codes(block, points, boundary=selected_boundary)
        node_weights = np.repeat(0.5 * segment_length * weights, count).reshape(q, count).T.reshape(-1)
        factors = LogPosPlane.from_values(corner_weights * node_weights[:, None])
        terms = LogPosPlane(codes).mul(factors)
        reduction = reduce_logpos_plane(LogPosPlane(terms.L.reshape(count, -1)), axis=1)
        tau_planes.append(cast(LogPosPlane, reduction.value))
        counters.pivot_reductions += int(reduction.counters.pivot_reductions)
        counters.aligned_mantissas += int(reduction.counters.aligned_mantissas)
        counters.bytes_read += int(codes.nbytes)
    low_tau, tau = tau_planes
    low_envelope = display_values(low_tau)
    high_envelope = display_values(tau)
    quadrature_errors = np.abs(high_envelope - low_envelope) + np.finfo(np.float64).eps * np.maximum(1.0, high_envelope)
    source_error = float(source_ledger.total_bound)
    errors = quadrature_errors + segment_length * source_error
    transmittance = optical_depth_to_transmittance(tau)
    one_minus_t = transmittance.log1m()
    length_plane = LogPosPlane.from_values(np.full(count, segment_length))
    if source_is_density:
        sigma = tau.div(length_plane)
        scale = one_minus_t.div(sigma)
        scale = LogPosPlane(np.where(tau.is_zero, length_plane.L, scale.L))
    else:
        scale = one_minus_t
    emission_plane = LogPosPlane.from_values(emission_values)
    contribution = LogPosPlane(scale.L[:, None]).mul(LogPosPlane(np.broadcast_to(emission_plane.L, (count, channels))))
    t_codes = np.broadcast_to(transmittance.L.astype(np.int64)[:, None], (count, channels))
    emission_scale = max(1.0, float(np.max(emission_values)))
    total = reduce_logpos(t_codes, contribution.L.astype(np.int64), errors, errors * emission_scale)
    if str(output_kind).lower().replace("-", "_") in {"transport", "affine_transport"}:
        value: Any = total
    else:
        value = total.apply(background)
    ledger = total.ledger.plus(_scale_error_ledger(source_ledger, length))
    if not return_report:
        return value
    report = _volume_report(
        backend=selected_backend.name,
        segments=count,
        bricks_considered=0,
        bricks_culled=0,
        early_terminated=False,
        quadrature_samples=count * (order + high_order),
        method="plane_optical_depth",
        ledger=ledger,
        diagnostics={
            **_volume_provenance(
                selected_backend,
                requested_backend=requested_backend,
                actual_callable="spf.renderer.volume.render_volume_planes",
            ),
            "constant_density": False,
            "heterogeneous": True,
            "consumer": "spf.renderer.transport_terminal.reduce_logpos",
            "density_consumer": "spf.tensor.v02_planes.reduce_logpos",
            "block_native_arithmetic": True,
            "quadrature_error": float(np.sum(quadrature_errors)),
            "quadrature_samples": count * (order + high_order),
            "channels": channels,
            "pivot_reductions": int(counters.pivot_reductions),
            "aligned_mantissas": int(counters.aligned_mantissas),
            "display_projection": "magnitude_rom",
        },
    )
    report.metrics.fallback_conversions = 0
    report.metrics.pivot_reductions = int(counters.pivot_reductions) + channels
    return value, report


class PreparedVolume:
    """Owned ordered transport carriers; compose anew for each background.

    Preparation packs segment objects into channel arrays. No composited color
    or total transport is cached. Typed logarithmic/SPF objects keep their
    normative reducer when no array specialization is available.
    """

    def __init__(self, segments: Sequence[Any], *, channels: int = 3) -> None:
        self.items = tuple(_coerce_transport(value) for value in segments)
        self.channels = self.items[0].channels if self.items else int(channels)
        if self.channels < 1 or any(x.channels != self.channels for x in self.items):
            raise ValueError("transport channel counts must match and be positive")
        self.native = all(_family(x) == "native" for item in self.items for x in (*item.T, *item.C))
        self.guarded = bool(self.items) and all(item.optical_depth is not None for item in self.items)
        self.logpos = bool(self.items) and all(
            isinstance(x, v02.LogPos) and not x.is_nan for item in self.items for x in (*item.T, *item.C)
        )
        if (self.native or self.logpos) and not self.guarded:
            self.t = np.asarray([x.T for x in self.items]).reshape(-1, self.channels)
            self.c = np.asarray([x.C for x in self.items]).reshape(-1, self.channels)
            self.et = np.asarray([x.error_bound_T for x in self.items])
            self.ec = np.asarray([x.error_bound_C for x in self.items])
            if self.logpos:
                self.t = np.asarray([[x.L for x in item.T] for item in self.items], dtype=np.int64)
                self.c = np.asarray([[x.L for x in item.C] for item in self.items], dtype=np.int64)
            self.stages = tuple(stage for x in self.items for stage in x.ledger.stages)
            for array in (self.t, self.c, self.et, self.ec):
                array.setflags(write=False)

    def execute(
        self,
        background: Any = 0.0,
        *,
        output_kind: str = "native",
        backend: str | SFRBackend = "reference",
        return_report: bool = False,
    ) -> Any:
        selected = get_backend(backend)
        _validate_volume_backend(selected)
        if not self.items:
            total = transport_identity(self.channels)
        elif self.guarded:
            total = transport_tree(self.items)
        elif self.logpos:
            from spf.renderer.transport_terminal import reduce_logpos

            total = reduce_logpos(self.t, self.c, self.et, self.ec, self.stages)
        elif self.native:
            total = _reduce_transport_arrays(self.t, self.c, self.et, self.ec, self.stages)
        else:
            total = transport_tree(self.items)
        output = total if output_kind in {"transport", "affine_transport"} else total.apply(background)
        if not return_report:
            return output
        report = _volume_report(
            backend=selected.name,
            segments=len(self.items),
            bricks_considered=0,
            bricks_culled=0,
            early_terminated=False,
            quadrature_samples=0,
                method=(
                    "prepared_guarded_optical_depth"
                    if self.guarded
                    else "prepared_ordered_arrays"
                    if self.native or self.logpos
                    else "prepared_typed_tree"
                ),
            ledger=total.ledger,
            diagnostics={
                "prepared": True,
                "cached_composites": 0,
                "consumer": "ordered_affine_transport",
                "native_channel_arrays": self.native,
                "logpos_channel_arrays": self.logpos,
                "guarded_optical_depth": self.guarded,
                "channels": self.channels,
                **_volume_provenance(
                    selected,
                    requested_backend=_backend_request_name(backend),
                    actual_callable="spf.renderer.volume.PreparedVolume.execute",
                ),
            },
        )
        return output, report

    render = execute


def prepare_volume(segments: Sequence[Any], *, channels: int = 3) -> PreparedVolume:
    return PreparedVolume(segments, channels=channels)


def render_volume(
    segments: Sequence[Any] | PreparedVolume | None = None,
    *,
    bricks: Sequence[VolumeBrick] | None = None,
    density: Any | None = None,
    start: Any | None = None,
    end: Any | None = None,
    emission: Any = 0.0,
    background: Any = 0.0,
    steps: int = 32,
    quadrature_order: int = 4,
    source_is_density: bool = True,
    cull_budget: float = 0.0,
    early_termination: float | None = None,
    error_budget: float | None = None,
    output_kind: str = "native",
    backend: str | SFRBackend = "reference",
    return_report: bool = False,
) -> Any:
    """Render explicit segments, bricks, or one heterogeneous density ray."""

    requested_backend = _backend_request_name(backend)
    selected_backend = get_backend(backend)
    _validate_volume_backend(selected_backend)
    if isinstance(segments, PreparedVolume):
        if bricks or density is not None or early_termination is not None or error_budget is not None or cull_budget:
            # Continuation/culling controls retain their ordinary consumer and ledger.
            segments = segments.items
        else:
            return segments.execute(
                background, output_kind=output_kind, backend=selected_backend, return_report=return_report
            )
    all_bricks = tuple(bricks or ())
    cull_certificate: BrickCullCertificate | None = None
    if error_budget is not None:
        cull_budget = float(error_budget)
    if all_bricks:
        kept, cull_certificate = cull_bricks(all_bricks, cull_budget, return_certificate=True)
        generated: list[OpticalDepthSegment] = []
        for brick in kept:
            if brick.start is not None and brick.end is not None:
                generated.append(
                    OpticalDepthSegment.from_density(
                        brick.density,
                        brick.start,
                        brick.end,
                        brick.emission,
                        quadrature_order=quadrature_order,
                        source_is_density=brick.source_is_density,
                    )
                )
            else:
                density_bound = brick.density_bound
                if not math.isfinite(density_bound):
                    raise ValueError("a brick without a ray must provide finite max_density")
                generated.append(
                    OpticalDepthSegment.from_constant_density(
                        density_bound,
                        brick.path_length,
                        brick.emission,
                        source_is_density=brick.source_is_density,
                    )
                )
        selected_segments: Sequence[Any] = generated
        quadrature_samples = 0
    elif density is not None:
        if start is None or end is None:
            raise ValueError("density rendering requires start and end points")
        selected_segments, quad = heterogeneous_segments(
            density,
            start,
            end,
            emission,
            steps=steps,
            quadrature_order=quadrature_order,
            source_is_density=source_is_density,
            return_report=True,
        )
        quadrature_samples = quad.samples
    else:
        selected_segments = tuple(segments or ())
        quadrature_samples = 0
    items = tuple(selected_segments)
    terminated = False
    termination_bound = 0.0
    if early_termination is not None:
        background_bound = _max_abs(background)
        items, termination_bound, terminated = early_terminate_transport(
            items, threshold=early_termination, background_bound=background_bound
        )
    transports = [_coerce_transport(item) for item in items]
    total = transport_tree(
        transports, channels=len(_channels(background)) if not transports else transports[0].channels
    )
    if str(output_kind).lower().replace("-", "_") in {"transport", "affine_transport"}:
        value = total
    else:
        value = total.apply(background)
    ledger = total.ledger
    if cull_certificate is not None:
        ledger = ledger.plus(cull_certificate.ledger)
    if termination_bound:
        ledger = ledger.plus(v02.ErrorLedger(pruning=termination_bound))
    report = _volume_report(
        backend=selected_backend.name,
        segments=len(items),
        bricks_considered=len(all_bricks),
        bricks_culled=0 if cull_certificate is None else cull_certificate.culled,
        early_terminated=terminated,
        quadrature_samples=quadrature_samples,
        method="dense_fallback" if density is not None and isinstance(density, np.ndarray) else "reference",
        ledger=ledger,
        diagnostics={
            **_volume_provenance(
                selected_backend,
                requested_backend=requested_backend,
                actual_callable=_VOLUME_CALLABLE,
            ),
            "constant_density": density is None and not all_bricks,
            "heterogeneous": density is not None or bool(all_bricks),
            "termination_bound": termination_bound,
            "within_error_budget": error_budget is None or ledger.total_bound <= float(error_budget),
            "channels": total.channels,
        },
    )
    if return_report:
        return value, report
    return value
