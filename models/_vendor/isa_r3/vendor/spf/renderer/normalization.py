"""Shared parameter-only normalization for all renderer Field consumers.

This module owns structured operation algebra and the legacy consumed-operation
handoff. It never samples a field. Consumers receive a complete Field with only
its unexecuted suffix, preserving carriers, support, residuals, and error custody.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from spf.renderer.model import (
    Domain,
    Field,
    FieldOperation,
    FieldRepresentation,
    RepresentationKind,
    Support,
    SupportKind,
)
from spf.renderer.residuals import SparseResidual
from spf.tensor import v02

_TAU = 2.0 * math.pi
_POSITIVE_KINDS = {"log_pos", "radiance", "incoherent_radiance"}
_SIGNED_KINDS = {"log_signed"}


def _normal_name(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower().replace("-", "_").replace(" ", "_")


def _representation_kind(value: RepresentationKind | str) -> RepresentationKind:
    if isinstance(value, RepresentationKind):
        return value
    return RepresentationKind(str(value).strip().upper().replace("-", "_").split(".")[-1])


class MaterializationError(ValueError):
    """Raised when a field cannot be materialized under its typed contract."""


@dataclass(frozen=True, slots=True)
class _ResidualSum:
    """Internal complete-field adapter for payload and representation residuals."""

    items: tuple[Any, ...]

    def __call__(self, points: Any) -> Any:
        return sum((v02._residual_value(item, points) for item in self.items), 0)

    @property
    def error_bound(self) -> float:
        return math.fsum(float(getattr(item, "error_bound", 0.0)) for item in self.items)

    @property
    def nbytes(self) -> int:
        return sum(int(getattr(item, "nbytes", 0)) for item in self.items)


def _combine_residuals(*items: Any) -> Any:
    retained: list[Any] = []
    for item in items:
        if item is None:
            continue
        if isinstance(item, _ResidualSum):
            retained.extend(item.items)
        else:
            retained.append(item)
    if not retained:
        return None
    if len(retained) == 1:
        return retained[0]
    return _ResidualSum(tuple(retained))


def _consume_representation_operations(field: Field) -> tuple[Field, tuple[int, ...]]:
    """Remove planner-consumed deferred operations exactly once.

    A representation returned by the planner may already include a proven
    operation prefix (or a non-prefix set after a rewrite).  The indices are
    part of the representation contract so duplicate operation names remain
    unambiguous.  Clearing the marker on the returned Field prevents a second
    runtime entry from consuming the same operations again.
    """

    consumed = tuple(field.representation.consumed_operation_indices)
    if not consumed:
        metadata_indices = field.representation.metadata.get("consumed_operation_indices", ())
        consumed = tuple(int(index) for index in metadata_indices)
    if not consumed:
        return field, ()
    count = len(field.operations)
    if any(index < 0 or index >= count for index in consumed):
        raise MaterializationError(
            f"representation consumed operation index outside field chain: {consumed!r} / {count}"
        )
    consumed_set = set(consumed)
    pending = tuple(operation for index, operation in enumerate(field.operations) if index not in consumed_set)
    metadata = {
        key: value for key, value in field.representation.metadata.items() if key != "consumed_operation_indices"
    }
    representation = replace(field.representation, consumed_operation_indices=(), metadata=metadata)
    applied = replace(field, operations=tuple(field.operations[index] for index in sorted(consumed_set)))
    return replace(
        field, operations=pending, representation=representation, support=_support_after_operations(applied)
    ), consumed


def _scale_ledger(ledger: v02.ErrorLedger, factor: float) -> v02.ErrorLedger:
    magnitude = abs(float(factor))
    return v02.ErrorLedger(
        source_quantization=ledger.source_quantization * magnitude,
        representation_fit=ledger.representation_fit * magnitude,
        pruning=ledger.pruning * magnitude,
        reducer=ledger.reducer * magnitude,
        transform=ledger.transform * magnitude,
        materialization=ledger.materialization * magnitude,
        display=ledger.display * magnitude,
        stages=ledger.stages,
    )


def _field_ledger_after_operations(field: Field) -> v02.ErrorLedger:
    """Propagate each error contribution at its final scale, in operation order.

    Wide opposite exposures can cancel before emission. Scaling error through
    intermediate Cartesian magnitudes would overflow or erase a small bound.
    """

    from spf.renderer.blocks import BlockView

    field, _ = _consume_representation_operations(field)

    periodic_block = isinstance(field.representation.payload, BlockView) and field.domain in {
        Domain.SPATIAL_1D,
        Domain.SPATIAL_2D,
        Domain.SPATIAL_3D,
    }
    spectral_segment = False
    gains: list[float] = []
    for operation in field.operations:
        gain = 0.0
        if operation.op == "mask":
            spectral_segment = False
        if (
            periodic_block
            and not spectral_segment
            and (
                operation.op
                in {
                    "translate",
                    "gaussian_filter",
                    "glare",
                    "fresnel",
                    "spectral_lens",
                    "lod",
                }
                or (
                    operation.op in {"phase", "phase_shift"}
                    and np.asarray(
                        operation.params.get("dT", operation.params.get("phase", operation.params.get("delta", 0)))
                    ).ndim
                    > 0
                )
            )
        ):
            # For the normalized discrete inverse, ||h||_1 <= sqrt(N) max|H|
            # by Parseval and Cauchy–Schwarz. Gaussian/glare/LOD are <= 1;
            # translation/propagation have unit modulus. Exposure is below.
            gain += 0.5 * math.log2(math.prod(field.representation.payload.shape))
            spectral_segment = True
        if operation.op == "exposure":
            exposure = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
            if not math.isfinite(exposure):
                raise MaterializationError("exposure must be finite")
            gain += exposure
        gains.append(gain)
    ledger = _scale_ledger_log2(field.representation.error, math.fsum(gains))
    for index, operation in enumerate(field.operations):
        ledger = ledger.plus(_scale_ledger_log2(operation.error, math.fsum(gains[index + 1 :])))
    return ledger


def _scale_ledger_log2(ledger: v02.ErrorLedger, gain: float) -> v02.ErrorLedger:
    if gain == 0.0:
        return ledger
    integer = math.floor(gain)
    fraction = gain - integer

    def scale(value: float) -> float:
        if value == 0.0 or not math.isfinite(value):
            return value
        mantissa, exponent = math.frexp(value)
        try:
            result = math.ldexp(mantissa * 2.0**fraction, exponent + integer)
        except OverflowError:
            return math.inf
        # A positive error remains positive even below the representable range.
        # Fractional gain emission rounds upward to retain a conservative bound.
        return float(np.nextafter(result, math.inf)) if result == 0.0 or fraction else result

    return v02.ErrorLedger(
        **{
            name: scale(float(getattr(ledger, name)))
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


def _support_after_operations(field: Field) -> Support:
    support = field.support
    if support.kind is SupportKind.EMPTY:
        return support
    for operation in field.operations:
        if (
            operation.op == "translate"
            and field.domain
            in {
                Domain.SPATIAL_1D,
                Domain.SPATIAL_2D,
                Domain.SPATIAL_3D,
            }
            and support.kind is SupportKind.AABB
            and support.region is not None
        ):
            offset = _offset(operation.params, support.region.dimension)
            support = Support.aabb(
                tuple(value + delta for value, delta in zip(support.region.lower, offset, strict=True)),
                tuple(value + delta for value, delta in zip(support.region.upper, offset, strict=True)),
            )
        elif operation.op == "gaussian_filter" and field.domain in {
            Domain.SPATIAL_1D,
            Domain.SPATIAL_2D,
            Domain.SPATIAL_3D,
        }:
            # Gaussian support is infinite.  Finite support requires a later
            # epsilon-certified truncation, not reuse of the source AABB.
            support = Support.full()
        elif operation.op in {"rotate", "affine"} and support.kind is SupportKind.AABB and support.region is not None:
            # A rotated or sheared box is not a box; claiming the source AABB
            # would be unsound (design §17.2-17.3).  A diagonal map with a
            # non-negative scale does stay axis-aligned, so keep that case.
            region = support.region
            matrix, shift = _affine_params(operation.params, region.dimension)
            if np.count_nonzero(matrix - np.diag(np.diagonal(matrix))) or np.any(np.diagonal(matrix) <= 0.0):
                support = Support.full()
            else:
                inverse = 1.0 / np.diagonal(matrix)
                lower = (np.asarray(region.lower) - shift) * inverse
                upper = (np.asarray(region.upper) - shift) * inverse
                support = Support.aabb(tuple(float(v) for v in lower), tuple(float(v) for v in upper))
    return support


def _offset(params: Mapping[str, Any], dimension: int) -> np.ndarray:
    """Return a translation offset in AXIS ORDER (design §17.1, fixed 2026-09-18).

    ``offset`` is already axis order.  The legacy ``dx``/``dy``/``dz`` spelling
    names the reader's axes (x = columns), so it is reversed here.
    """

    raw = params.get("offset")
    if raw is None:
        named = tuple(value for value in (params.get("dx"), params.get("dy"), params.get("dz")) if value is not None)
        raw = named[::-1]
    vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    if vector.shape != (dimension,):
        raise MaterializationError("translation offset dimension does not match the field")
    if not np.all(np.isfinite(vector)):
        raise MaterializationError("translation offset must be finite")
    return vector


def _affine_params(params: Mapping[str, Any], dimension: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the AXIS-ORDER ``(A, b)`` of a §17.2 rotation or §17.3 affine.

    Both operations are the substitution ``g(y) = f(A y + b)`` that
    :func:`spf.tensor.v02.atom_affine` and :func:`spf.tensor.v02.jet_transform`
    implement, so ``affine(I, -d)`` is exactly ``translate(d)``.  A rotation is
    given by ``angle`` (radians, in the array's own axis frame) and has
    ``|det A| = 1``, so its §17.3 log-magnitude shift is zero.
    """

    angle = params.get("angle")
    if params.get("matrix") is None and angle is not None:
        value = float(angle)
        if dimension != 2 or not math.isfinite(value):
            raise MaterializationError("a rotation angle requires a finite value and a two-dimensional field")
        cos, sin = math.cos(value), math.sin(value)
        matrix = np.array([[cos, -sin], [sin, cos]], dtype=np.float64)
    else:
        matrix = np.asarray(params.get("matrix"), dtype=np.float64)
    shift = np.asarray(params.get("offset", (0.0,) * dimension), dtype=np.float64).reshape(-1)
    if matrix.shape != (dimension, dimension) or shift.shape != (dimension,):
        raise MaterializationError("affine transform dimensions do not match the field")
    if not (np.all(np.isfinite(matrix)) and np.all(np.isfinite(shift))):
        raise MaterializationError("affine transform must be finite")
    if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
        raise MaterializationError("affine transform matrix must be invertible")
    return matrix, shift


def _matrix_param(params: Mapping[str, Any], key: str, dimension: int) -> np.ndarray:
    matrix = np.asarray(params.get(key), dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise MaterializationError(f"{key} must have shape ({dimension}, {dimension})")
    return matrix


def _gaussian_transfer_atom(covariance: Any, dimension: int) -> v02.GaussianChirpAtom:
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise MaterializationError("Gaussian filter covariance dimension does not match the atom")
    if not np.allclose(matrix, matrix.T):
        raise MaterializationError("Gaussian filter covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) < -1.0e-12:
        raise MaterializationError("Gaussian filter covariance must be positive semidefinite")
    # H(k)=exp(-2*pi^2*k^T Sigma*k) and the atom exponent is
    # -1/2*k^T C*k, hence C=4*pi^2*Sigma.
    return v02.GaussianChirpAtom.from_arrays(
        0.0j,
        np.zeros(dimension, dtype=np.complex128),
        (4.0 * math.pi**2 * matrix).astype(np.complex128),
    )


def _lens_atom(params: Mapping[str, Any], dimension: int) -> v02.GaussianChirpAtom:
    focal_raw = params.get("focal_length")
    wavelength_raw = params.get("wavelength")
    if focal_raw is None or wavelength_raw is None:
        raise MaterializationError("lens phase requires focal_length and wavelength")
    focal = float(focal_raw)
    wavelength = float(wavelength_raw)
    if focal == 0.0 or wavelength <= 0.0 or not math.isfinite(focal) or not math.isfinite(wavelength):
        raise MaterializationError("lens phase requires a finite nonzero focal length and positive wavelength")
    matrix = 1j * (2.0 * math.pi / (wavelength * focal)) * np.eye(dimension, dtype=np.complex128)
    return v02.GaussianChirpAtom.from_arrays(0.0j, np.zeros(dimension, dtype=np.complex128), matrix)


@dataclass(frozen=True, slots=True)
class _ScaledResidual:
    """Retain a residual gain in log space until its numeric consumer."""

    residual: Any
    exposure: float

    def scale_values(self, values: Any) -> np.ndarray:
        values = np.asarray(values, dtype=np.complex128)
        integer = math.floor(self.exposure)
        fraction = 2.0 ** (self.exposure - integer)
        exponent = max(-(2**30), min(2**30, integer))
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            result = np.empty_like(values)
            result.real = np.ldexp(values.real * fraction, exponent)
            result.imag = np.ldexp(values.imag * fraction, exponent)
        return np.asarray(result, dtype=np.complex128)

    def __call__(self, points: Any) -> Any:
        return self.scale_values(v02._residual_value(self.residual, points))

    @property
    def error_bound(self) -> float:
        bound = float(getattr(self.residual, "error_bound", 0.0))
        return float(_scale_ledger_log2(v02.ErrorLedger(representation_fit=bound), self.exposure).representation_fit)

    @property
    def nbytes(self) -> int:
        return int(getattr(self.residual, "nbytes", 0)) + 8


def _transform_mixture_residual(
    residual: Any,
    operation: FieldOperation,
    field: Field,
    dimension: int,
) -> tuple[Any, bool]:
    """Transform a mixture residual without changing the represented field.

    Atom closure only applies to the whole mixture when the residual follows
    the same operator.  Returning ``False`` forces the operation (including
    every atom) through the dense suffix instead of partially transforming the
    analytic component and silently leaving a stale residual behind.
    """

    if residual is None:
        return None, True
    if isinstance(residual, _ResidualSum):
        transformed: list[Any] = []
        for item in residual.items:
            value, supported = _transform_mixture_residual(item, operation, field, dimension)
            if not supported:
                return residual, False
            transformed.append(value)
        return _ResidualSum(tuple(transformed)), True
    op = operation.op
    if op == "exposure":
        ev = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
        if not math.isfinite(ev):
            raise MaterializationError("exposure must be finite")
        if isinstance(residual, _ScaledResidual):
            return _ScaledResidual(residual.residual, math.fsum((residual.exposure, ev))), True
        return _ScaledResidual(residual, ev), True
    if isinstance(residual, _ScaledResidual):
        transformed, supported = _transform_mixture_residual(residual.residual, operation, field, dimension)
        return (_ScaledResidual(transformed, residual.exposure), True) if supported else (residual, False)
    if not callable(residual):
        # An uncharted dense/sparse residual has no coordinate contract here.
        # It must be sampled together with the atoms before a geometric or
        # filtering operation is applied.
        return residual, False

    if op == "translate":
        offset = _offset(operation.params, dimension)
        if field.domain in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
            bound = offset.copy()
            return lambda points, residual=residual, bound=bound: residual(np.asarray(points) - bound), True
        if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
            bound = offset.copy()

            def translated_frequency(points: Any, residual: Any = residual, bound: np.ndarray = bound) -> Any:
                coordinates = np.asarray(points, dtype=np.float64)
                phase = -_TAU * np.einsum("...i,i->...", coordinates, bound)
                return residual(coordinates) * np.exp(1j * phase)

            return translated_frequency, True
        return residual, False
    if op == "gaussian_filter":
        # Frequency-space filtering is pointwise.  Spatial convolution of an
        # arbitrary residual is not closed and therefore belongs to the dense
        # fallback for the *entire* mixture.
        if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
            covariance = _matrix_param(operation.params, "covariance", dimension)
            bound_covariance = covariance.copy()

            def filtered_frequency(
                points: Any,
                residual: Any = residual,
                covariance: np.ndarray = bound_covariance,
            ) -> Any:
                coordinates = np.asarray(points, dtype=np.float64)
                exponent = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", coordinates, covariance, coordinates)
                return residual(coordinates) * np.exp(exponent)

            return filtered_frequency, True
        native_filter = getattr(residual, "gaussian_filter", None)
        if callable(native_filter):
            return native_filter(_matrix_param(operation.params, "covariance", dimension)), True
        return residual, False
    if op == "lens_phase":
        focal_raw = operation.params.get("focal_length")
        wavelength_raw = operation.params.get("wavelength")
        if focal_raw is None or wavelength_raw is None:
            raise MaterializationError("lens phase requires focal_length and wavelength")
        focal = float(focal_raw)
        wavelength = float(wavelength_raw)
        if focal == 0.0 or wavelength <= 0.0 or not math.isfinite(focal) or not math.isfinite(wavelength):
            raise MaterializationError("lens phase requires a finite nonzero focal length and positive wavelength")

        def lensed(points: Any, residual: Any = residual, focal: float = focal, wavelength: float = wavelength) -> Any:
            coordinates = np.asarray(points, dtype=np.float64)
            phase = -math.pi * np.sum(coordinates * coordinates, axis=-1) / (wavelength * focal)
            return residual(coordinates) * np.exp(1j * phase)

        return lensed, True
    return residual, False


def _apply_atom_operation(
    atom: v02.GaussianChirpAtom, operation: FieldOperation, field: Field
) -> v02.GaussianChirpAtom:
    op = operation.op
    if op in {"fresnel", "spectral_lens", "phase", "phase_shift"}:
        return _spectral_atom_terms(atom, operation, field)[0]
    if op == "translate":
        vector = _offset(operation.params, atom.dimension)
        if field.domain in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
            return v02.atom_affine(atom, np.eye(atom.dimension), -vector)
        if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
            phase_ramp = v02.GaussianChirpAtom.from_arrays(
                0.0j,
                -1j * 2.0 * math.pi * vector,
                np.zeros((atom.dimension, atom.dimension), dtype=np.complex128),
            )
            return v02.atom_mul(atom, phase_ramp)
        raise MaterializationError(
            f"translation is not defined for domain {getattr(field.domain, 'value', field.domain)}"
        )
    if op in {"rotate", "affine"}:
        matrix, shift = _affine_params(operation.params, atom.dimension)
        if field.domain in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
            # Design §17.2/§17.3 g(y) = f(A y + b): an exact parameter change.
            # The 1/|det A| amplitude belongs to the spectrum, so a spatial
            # resample carries no magnitude shift.
            return v02.atom_affine(atom, matrix, shift)
        if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
            # G(k) = |det A|^-1 exp(i 2 pi (A^-1 b) . k) F(A^-T k) (§17.3).
            inverse = np.linalg.inv(matrix)
            remapped = v02.atom_affine(atom, inverse.T, np.zeros(atom.dimension))
            factor = v02.GaussianChirpAtom.from_arrays(
                complex(-math.log(abs(float(np.linalg.det(matrix))))),
                (inverse @ shift) * complex(0.0, _TAU),
                np.zeros((atom.dimension, atom.dimension)),
                allow_nondecaying=True,
            )
            return v02.atom_mul(remapped, factor)
        raise MaterializationError(
            f"affine transform is not defined for domain {getattr(field.domain, 'value', field.domain)}"
        )
    if op == "gaussian_filter":
        covariance = np.asarray(operation.params["covariance"], dtype=np.float64)
        if field.domain in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
            if not np.any(covariance):
                return atom
            # A singular footprint remains closed: the *source* precision is
            # invertible, so no inverse of the footprint covariance is needed.
            return v02.atom_inverse_fourier(
                v02.atom_mul(v02.atom_fourier(atom), _gaussian_transfer_atom(covariance, atom.dimension))
            )
        if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
            return v02.atom_mul(atom, _gaussian_transfer_atom(covariance, atom.dimension))
        raise MaterializationError(
            f"Gaussian filtering is not defined for domain {getattr(field.domain, 'value', field.domain)}"
        )
    if op == "exposure":
        exposure = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
        if not math.isfinite(exposure):
            raise MaterializationError("exposure must be finite")
        return v02.GaussianChirpAtom.from_arrays(
            atom.kappa + exposure * math.log(2.0), atom.h, atom.C, allow_nondecaying=atom.allow_nondecaying
        )
    if op == "lens_phase":
        return v02.atom_mul(atom, _lens_atom(operation.params, atom.dimension))
    raise MaterializationError(f"operation {op!r} is not closed over GaussianChirpAtom")


def _spectral_atom_terms(
    atom: v02.GaussianChirpAtom, operation: FieldOperation, field: Field
) -> tuple[v02.GaussianChirpAtom, ...]:
    from spf.renderer.block_filter import FOURIER, SPATIAL
    from spf.renderer.transfers import SpectralTransfer

    if field.domain not in FOURIER | SPATIAL:
        raise MaterializationError("spectral atom filtering requires a spatial or Fourier domain")
    transfer = SpectralTransfer.identity(atom.dimension).apply(operation)
    if transfer.phase_residuals or transfer.windows:
        raise MaterializationError("sampled transfer requires an explicit discrete field boundary")
    if operation.op in {"phase", "phase_shift"}:
        return (v02.atom_mul(atom, transfer.atom),)
    if operation.op in {"fresnel", "spectral_lens"} and field.representation.error.total_bound:
        raise MaterializationError("propagating uncertain continuous atoms requires an integrable error envelope")
    spatial = field.domain in SPATIAL
    terms: tuple[v02.GaussianChirpAtom, ...] = (
        v02.atom_mul(v02.atom_fourier(atom) if spatial else atom, transfer.atom),
    )
    for mixture in transfer.mixtures:
        terms = tuple(v02.atom_mul(a, b) for a in terms for b in mixture.atoms)
    return tuple(v02.atom_inverse_fourier(a) for a in terms) if spatial else terms


def _apply_jet_operation(jet: v02.Jet2D, operation: FieldOperation) -> v02.Jet2D:
    op = operation.op
    if op == "translate":
        vector = _offset(operation.params, 2)
        return v02.jet_transform(jet, np.eye(2), -vector)
    if op == "exposure":
        exposure = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
        if not math.isfinite(exposure):
            raise MaterializationError("exposure must be finite")
        factor = 2.0**exposure
        if jet.value_kind in {"log", "log_pos", "log_signed"}:
            delta = exposure * v02.SPF32_L_SCALE * (1 << jet.guard_L)
            return replace(jet, value=jet.value + delta)
        if jet.value_kind == "phase":
            raise MaterializationError("exposure is not defined for a phase-only jet")
        return replace(
            jet,
            value=jet.value * factor,
            grad_x=jet.grad_x * factor,
            grad_y=jet.grad_y * factor,
            hess_xx=jet.hess_xx * factor,
            hess_xy=jet.hess_xy * factor,
            hess_yy=jet.hess_yy * factor,
        )
    raise MaterializationError(f"operation {op!r} is not closed over Jet2D")


def _wrap_separable_factor(factor: Any, operation: str, parameter: Any, kind: str) -> Any:
    if operation == "translate":
        offset = float(parameter)
        if callable(factor):
            return lambda coordinate, factor=factor, offset=offset: factor(coordinate - offset)
        return factor
    if operation == "exposure":
        scale = float(parameter)
        if callable(factor):

            def scaled(coordinate: Any, factor: Any = factor, scale: float = scale, kind: str = kind) -> Any:
                value = factor(coordinate)
                if kind in _POSITIVE_KINDS and isinstance(value, v02.LogPos):
                    return v02.LogPos.from_value(value.value * scale)
                if kind in _SIGNED_KINDS and isinstance(value, v02.LogSigned):
                    return v02.LogSigned.from_value(value.value * scale)
                return value * scale

            return scaled
        if kind in _POSITIVE_KINDS and isinstance(factor, v02.LogPos):
            return v02.LogPos.from_value(factor.value * scale)
        if kind in _SIGNED_KINDS and isinstance(factor, v02.LogSigned):
            return v02.LogSigned.from_value(factor.value * scale)
        return factor * scale
    return factor


def _prepare_structured(
    field: Field, payload: Any, region: Any, budget: float
) -> tuple[Any, tuple[FieldOperation, ...], list[str], int, int, list[str]]:
    """Apply the analytic operation prefix and return the dense suffix."""

    kind = field.representation.kind
    operations = tuple(field.operations)
    analytic: list[str] = []
    notes: list[str] = []
    atoms_considered = 0
    atoms_pruned = 0
    current: Any

    kind = _representation_kind(kind)
    from spf.renderer.block_filter import FILTER_OPS, FilteredBlockView
    from spf.renderer.blocks import BlockView

    if isinstance(payload, BlockView):
        current = payload
        for index, operation in enumerate(operations):
            try:
                analytic_filter = operation.op in {
                    "translate",
                    "gaussian_filter",
                    "glare",
                    "fresnel",
                    "spectral_lens",
                    "lod",
                }
                sampled_phase = (
                    operation.op in {"phase", "phase_shift"}
                    and np.asarray(
                        operation.params.get("dT", operation.params.get("phase", operation.params.get("delta", 0)))
                    ).ndim
                    > 0
                )
                if (analytic_filter or sampled_phase) and not isinstance(current, FilteredBlockView):
                    current = FilteredBlockView.from_field(current, field)
                current = current.apply(operation)
            except ValueError as exc:
                if operation.op in FILTER_OPS:
                    raise MaterializationError(str(exc)) from exc
                return current, operations[index:], analytic, 0, 0, notes
            analytic.append(operation.op)
            notes.append(
                f"block analytic transfer: {operation.op}"
                if isinstance(current, FilteredBlockView)
                else f"block header closure: {operation.op}"
            )
        return current, (), analytic, 0, 0, notes
    if kind is RepresentationKind.GAUSSIAN_CHIRP_ATOM:
        if not isinstance(payload, v02.GaussianChirpAtom):
            raise TypeError("GAUSSIAN_CHIRP_ATOM payload must be GaussianChirpAtom")
        current = payload
        for index, operation in enumerate(operations):
            try:
                if operation.op == "glare":
                    mixture = v02.AtomMixture(_spectral_atom_terms(current, operation, field))
                    next_field = replace(
                        field,
                        operations=operations[index + 1 :],
                        representation=replace(
                            field.representation, kind=RepresentationKind.ATOM_MIXTURE, payload=mixture
                        ),
                    )
                    result, pending, rest, considered, pruned, tail = _prepare_structured(
                        next_field, mixture, region, budget
                    )
                    return (
                        result,
                        pending,
                        [*analytic, operation.op, *rest],
                        considered,
                        pruned,
                        [*notes, "analytic glare mixture", *tail],
                    )
                current = _apply_atom_operation(current, operation, field)
            except MaterializationError:
                return current, operations[index:], analytic, atoms_considered, atoms_pruned, notes
            analytic.append(operation.op)
            notes.append(f"analytic closure: {operation.op}")
        return current, (), analytic, atoms_considered, atoms_pruned, notes

    if kind is RepresentationKind.ATOM_MIXTURE:
        if not isinstance(payload, v02.AtomMixture):
            raise TypeError("ATOM_MIXTURE payload must be AtomMixture")
        current = payload
        atoms_considered = len(current.atoms)
        if current.dropped_bound:
            atoms_pruned = 1
        dropped_exposure = 0.0

        def emit_bound(value: v02.AtomMixture) -> v02.AtomMixture:
            bound = _scale_ledger_log2(v02.ErrorLedger(pruning=value.dropped_bound), dropped_exposure).pruning
            if not math.isfinite(bound):
                raise MaterializationError("mixture pruning bound exceeds float64 at the consumer boundary")
            return v02.AtomMixture(value.atoms, value.residual, bound)

        for index, operation in enumerate(operations):
            op = operation.op
            if op in {"glare", "fresnel", "spectral_lens", "phase", "phase_shift"} and current.residual is None:
                if current.dropped_bound and op in {"fresnel", "spectral_lens"}:
                    raise MaterializationError(
                        "propagating uncertain continuous atoms requires an integrable error envelope"
                    )
                current = v02.AtomMixture(
                    tuple(b for a in current.atoms for b in _spectral_atom_terms(a, operation, field)),
                    dropped_bound=current.dropped_bound,
                )
                analytic.append(op)
                notes.append(f"analytic spectral closure: {op}")
                continue
            if op in {"translate", "gaussian_filter", "exposure", "lens_phase", "rotate", "affine"}:
                dimension = current.atoms[0].dimension if current.atoms else field.chart.dimension
                residual, residual_supported = _transform_mixture_residual(
                    current.residual, operation, field, dimension
                )
                if not residual_supported:
                    if op == "gaussian_filter" and callable(current.residual):
                        raise MaterializationError(
                            "spatial Gaussian filtering of an unbounded callable residual requires "
                            "a residual.gaussian_filter(covariance) implementation or a bounded dense residual"
                        )
                    notes.append(f"whole-mixture dense fallback: residual is not closed under {op}")
                    return emit_bound(current), operations[index:], analytic, atoms_considered, atoms_pruned, notes
                transformed: list[v02.GaussianChirpAtom] = []
                try:
                    for atom in current.atoms:
                        transformed.append(_apply_atom_operation(atom, operation, field))
                except MaterializationError:
                    return emit_bound(current), operations[index:], analytic, atoms_considered, atoms_pruned, notes
                if op == "exposure":
                    ev = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
                    dropped_exposure = math.fsum((dropped_exposure, ev))
                current = v02.AtomMixture(tuple(transformed), residual, current.dropped_bound)
                analytic.append(op)
                notes.append(f"analytic closure: {op}")
                continue
            if op == "merge_equal":
                current = v02.mixture_merge_equal(current)
                analytic.append(op)
                notes.append("exact common-shape atom merge")
                atoms_considered = len(current.atoms)
                continue
            if op in {"prune", "mixture_prune", "cull", "mixture_cull"}:
                operation_budget = float(operation.params.get("error_budget", budget))
                if not math.isfinite(operation_budget) or operation_budget < 0.0:
                    raise MaterializationError("mixture pruning requires a finite non-negative error budget")
                prune_budget = min(operation_budget, budget)
                prune_region = operation.params.get("region", region)
                if prune_region is None:
                    return emit_bound(current), operations[index:], analytic, atoms_considered, atoms_pruned, notes
                current = emit_bound(current)
                dropped_exposure = 0.0
                before = len(current.atoms)
                current = v02.mixture_prune(current, prune_region, prune_budget)
                atoms_pruned += max(0, before - len(current.atoms))
                analytic.append(op)
                notes.append(f"bound-certified mixture prune: {before - len(current.atoms)} atoms")
                continue
            return emit_bound(current), operations[index:], analytic, atoms_considered, atoms_pruned, notes
        return emit_bound(current), (), analytic, atoms_considered, atoms_pruned, notes

    if kind is RepresentationKind.QUADRATIC_JET:
        if not isinstance(payload, v02.Jet2D):
            # SpectralJetTile2D is a useful dense-ready representation but is
            # not a single Jet2D and therefore has no analytic operation
            # closure at this boundary.
            return payload, operations, analytic, atoms_considered, atoms_pruned, notes
        current = payload
        for index, operation in enumerate(operations):
            try:
                current = _apply_jet_operation(current, operation)
            except MaterializationError:
                return current, operations[index:], analytic, atoms_considered, atoms_pruned, notes
            analytic.append(operation.op)
            notes.append(f"analytic closure: {operation.op}")
        return current, (), analytic, atoms_considered, atoms_pruned, notes

    if kind is RepresentationKind.SEPARABLE:
        if not isinstance(payload, v02.SeparableField):
            raise TypeError("SEPARABLE payload must be SeparableField")
        current = payload
        kind_name = _normal_name(current.value_kind)
        for index, operation in enumerate(operations):
            op = operation.op
            if op == "translate":
                vector = _offset(operation.params, current.dimension)
                rows = tuple(
                    tuple(
                        _wrap_separable_factor(factor, "translate", vector[axis], kind_name)
                        for axis, factor in enumerate(row)
                    )
                    for row in current.factors
                )
                current = v02.SeparableField(rows, current.value_kind)
                analytic.append(op)
                notes.append("analytic closure: separable coordinate shift")
                continue
            if op == "exposure":
                exposure = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
                scale = 2.0**exposure
                rows = tuple(
                    tuple(
                        _wrap_separable_factor(factor, "exposure", scale if axis == 0 else 1.0, kind_name)
                        for axis, factor in enumerate(row)
                    )
                    for row in current.factors
                )
                current = v02.SeparableField(rows, current.value_kind)
                analytic.append(op)
                notes.append("analytic closure: separable gain")
                continue
            return current, operations[index:], analytic, atoms_considered, atoms_pruned, notes
        return current, (), analytic, atoms_considered, atoms_pruned, notes

    return payload, operations, analytic, atoms_considered, atoms_pruned, notes


@dataclass(frozen=True, slots=True)
class FieldNormalization:
    """A complete field with only its unexecuted operation suffix.

    Indices describe the *input* for legacy planner diagnostics only. Execution
    uses ``field`` and never interprets these indices again.
    """

    field: Field
    consumed_indices: tuple[int, ...] = ()
    analytic_operations: tuple[str, ...] = ()
    atoms_considered: int = 0
    atoms_pruned: int = 0
    notes: tuple[str, ...] = ()


def select_representation(field: Field, representation: FieldRepresentation) -> FieldNormalization:
    """Bind a planner candidate and acknowledge its applied operations once."""
    selected, consumed = _consume_representation_operations(replace(field, representation=representation))
    return FieldNormalization(selected, consumed)


def normalize_field(field: Field, *, region: Any = None, budget: float = math.inf) -> FieldNormalization:
    """Fold a representation-closed prefix without allocating sampled fields.

    Operation order, the whole residual, support, and absolute error follow the
    same prefix. Unsupported operations remain explicit. Calling this again on
    the result cannot repeat the already applied operations.
    """
    if not isinstance(field, Field):
        raise TypeError("normalize_field expects a renderer.model.Field")
    original = field
    field, existing = _consume_representation_operations(field)
    remaining_indices = tuple(index for index in range(len(original.operations)) if index not in set(existing))
    payload = field.representation.payload
    residual = field.representation.residual
    kind = field.representation.kind
    if residual is not None and isinstance(payload, (v02.GaussianChirpAtom, v02.AtomMixture)):
        payload = (
            v02.AtomMixture(payload.atoms, _combine_residuals(payload.residual, residual), payload.dropped_bound)
            if isinstance(payload, v02.AtomMixture)
            else v02.AtomMixture((payload,), residual)
        )
        residual = None
        kind = RepresentationKind.ATOM_MIXTURE
        field = replace(field, representation=replace(field.representation, payload=payload, residual=None, kind=kind))
    if not field.operations:
        return FieldNormalization(field, existing)

    # A representation-level residual must admit the same prefix. Keep a
    # non-closed operation pending for the whole field, never just its residual.
    operations = tuple(field.operations)
    eligible = len(operations)
    residual_prefix = [residual]
    if residual is not None:
        for index, operation in enumerate(operations):
            transformed, supported = _transform_mixture_residual(
                residual_prefix[-1], operation, field, field.chart.dimension
            )
            if not supported:
                eligible = index
                break
            residual_prefix.append(transformed)
    working = replace(field, operations=operations[:eligible])
    prepared, pending_prefix, analytic, considered, pruned, notes = _prepare_structured(
        working, payload, region, float(budget)
    )
    count = eligible - len(pending_prefix)
    pending = operations[count:]
    if isinstance(prepared, v02.AtomMixture):
        kind = RepresentationKind.ATOM_MIXTURE
    applied = replace(field, operations=operations[:count])
    representation = replace(
        field.representation,
        kind=kind,
        payload=prepared,
        residual=residual_prefix[count] if residual is not None else None,
        error=_field_ledger_after_operations(applied),
        consumed_operation_indices=(),
        byte_size=None if count else field.representation.byte_size,
    )
    representation = inherit_representation_error(representation, complete_field_ledger(applied))
    normalized = replace(
        field, representation=representation, operations=pending, support=_support_after_operations(applied)
    )
    consumed = tuple(sorted((*existing, *remaining_indices[:count])))
    return FieldNormalization(normalized, consumed, tuple(analytic), considered, pruned, tuple(notes))


def require_closed_field(field: Field, *, consumer: str) -> Field:
    """Normalize at a structured-only API boundary, rejecting an open suffix."""
    normalized = normalize_field(field).field
    if normalized.operations:
        pending = ", ".join(operation.op for operation in normalized.operations)
        raise MaterializationError(f"{consumer} cannot consume deferred field operations: {pending}")
    return normalized


def _intrinsic_representation_error(representation: FieldRepresentation) -> v02.ErrorLedger:
    extra = 0.0
    payload = representation.payload
    residuals = [representation.residual]
    if isinstance(payload, SparseResidual):
        residuals.append(payload)
    if isinstance(payload, Mapping):
        for key in ("residual", "sparse"):
            if payload.get(key) is not None:
                residuals.append(payload[key])
    if isinstance(payload, v02.AtomMixture) and payload.residual is not None:
        residuals.append(payload.residual)
    for residual in residuals:
        if isinstance(residual, SparseResidual):
            extra += float(residual.error_bound)
        elif residual is not None:
            extra += float(getattr(residual, "error_bound", 0.0))
    if isinstance(payload, v02.AtomMixture):
        return v02.ErrorLedger(representation_fit=extra, pruning=float(payload.dropped_bound))
    return v02.ErrorLedger(representation_fit=extra)


def _apply_representation_error(field: Field, ledger: v02.ErrorLedger) -> v02.ErrorLedger:
    return ledger.plus(_intrinsic_representation_error(field.representation))


def inherit_representation_error(
    representation: FieldRepresentation, source_error: v02.ErrorLedger
) -> FieldRepresentation:
    """Carry source uncertainty that the output payload no longer retains."""
    present = representation.error.plus(_intrinsic_representation_error(representation))
    missing_values: dict[str, Any] = {
        name: 0.0
        if getattr(present, name) >= getattr(source_error, name)
        else getattr(source_error, name) - getattr(present, name)
        for name in (
            "source_quantization",
            "representation_fit",
            "pruning",
            "reducer",
            "transform",
            "materialization",
            "display",
        )
    }
    missing = v02.ErrorLedger(**missing_values)
    return replace(representation, error=representation.error.plus(missing))


def complete_field_ledger(field: Field) -> v02.ErrorLedger:
    """Include retained residual/pruning uncertainty before pending gains."""
    field, _ = _consume_representation_operations(field)
    ledger = _apply_representation_error(field, field.representation.error)
    return _field_ledger_after_operations(replace(field, representation=replace(field.representation, error=ledger)))
