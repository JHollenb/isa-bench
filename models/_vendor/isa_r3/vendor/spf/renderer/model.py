"""Canonical typed field model for the SFR v0.2 renderer.

The low-level :mod:`spf.renderer.v02` module is an executable operation graph.
This module supplies the metadata-rich field object described by the renderer
design: value algebra, physical semantics, coordinate chart, support,
representation, precision, and error contracts travel together.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from dataclasses import fields as dataclass_fields
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.renderer.v02 import FieldKind
from spf.tensor.v02 import AtomMixture, ErrorLedger, GaussianChirpAtom


class Domain(StrEnum):
    SPATIAL_1D = "SPATIAL_1D"
    SPATIAL_2D = "SPATIAL_2D"
    SPATIAL_3D = "SPATIAL_3D"
    FOURIER_1D = "FOURIER_1D"
    FOURIER_2D = "FOURIER_2D"
    FOURIER_3D = "FOURIER_3D"
    ANGULAR_S2 = "ANGULAR_S2"
    TEMPORAL = "TEMPORAL"
    WAVELENGTH = "WAVELENGTH"
    SCREEN_TILE = "SCREEN_TILE"
    CUSTOM = "CUSTOM"


class Semantic(StrEnum):
    RADIANCE = "RADIANCE"
    IRRADIANCE = "IRRADIANCE"
    TRANSMITTANCE = "TRANSMITTANCE"
    OPACITY = "OPACITY"
    OPTICAL_DEPTH = "OPTICAL_DEPTH"
    ALBEDO = "ALBEDO"
    MATERIAL_GAIN = "MATERIAL_GAIN"
    HEIGHT = "HEIGHT"
    DISPLACEMENT = "DISPLACEMENT"
    NORMAL_COMPONENT = "NORMAL_COMPONENT"
    COMPLEX_OPTICAL_AMPLITUDE = "COMPLEX_OPTICAL_AMPLITUDE"
    FILTER_TRANSFER = "FILTER_TRANSFER"
    DEPTH = "DEPTH"
    VISIBILITY = "VISIBILITY"
    GENERIC_FIELD = "GENERIC_FIELD"


class Coherence(StrEnum):
    INCOHERENT_POSITIVE = "INCOHERENT_POSITIVE"
    SIGNED_REAL = "SIGNED_REAL"
    COHERENT_COMPLEX = "COHERENT_COMPLEX"
    PHASE_ONLY = "PHASE_ONLY"
    NOT_APPLICABLE = "NOT_APPLICABLE"


def default_dense_representation(kind: Any, payload: Any) -> RepresentationKind:
    """Default representation label for an unstructured payload.

    Design v0.2 §7.3 (amended 2026-09-17): packed SPF words, typed log/phase
    planes and block views are ``DENSE_TYPED`` (DenseSPF32).  Only a
    Cartesian float/complex array bound to a ``DENSE``/``ANY`` field is a
    ``DENSE_CARTESIAN_REFERENCE`` import, and it keeps that honest label.
    """

    selected = kind if isinstance(kind, FieldKind) else FieldKind(str(kind).lower())
    if selected not in {FieldKind.DENSE, FieldKind.ANY}:
        return RepresentationKind.DENSE_TYPED
    array = payload if isinstance(payload, np.ndarray) else None
    if array is not None and array.dtype.kind in "iu":
        return RepresentationKind.DENSE_TYPED
    return RepresentationKind.DENSE_CARTESIAN_REFERENCE


class RepresentationKind(StrEnum):
    GAUSSIAN_CHIRP_ATOM = "GAUSSIAN_CHIRP_ATOM"
    SPHERICAL_GAUSSIAN_ATOM = "SPHERICAL_GAUSSIAN_ATOM"
    ATOM_MIXTURE = "ATOM_MIXTURE"
    QUADRATIC_JET = "QUADRATIC_JET"
    SEPARABLE = "SEPARABLE"
    SPARSE_RESIDUAL = "SPARSE_RESIDUAL"
    HERMITIAN_HALF = "HERMITIAN_HALF"
    DENSE_TYPED = "DENSE_TYPED"
    DENSE_CARTESIAN_REFERENCE = "DENSE_CARTESIAN_REFERENCE"


class MaterializationState(StrEnum):
    SYMBOLIC_ATOMS = "SYMBOLIC_ATOMS"
    ANALYTIC_ATOM = "ANALYTIC_ATOM"
    QUADRATIC_JET = "QUADRATIC_JET"
    SEPARABLE = "SEPARABLE"
    SPARSE_RESIDUAL = "SPARSE_RESIDUAL"
    HERMITIAN_HALF = "HERMITIAN_HALF"
    DENSE_TYPED = "DENSE_TYPED"
    DENSE_CARTESIAN_REFERENCE = "DENSE_CARTESIAN_REFERENCE"


class PrecisionMode(StrEnum):
    FAST = "FAST"
    BALANCED = "BALANCED"
    ACCURATE = "ACCURATE"
    EXACT = "EXACT"


class SupportKind(StrEnum):
    AABB = "AABB"
    ORIENTED_BOX = "ORIENTED_BOX"
    FREQUENCY_BAND = "FREQUENCY_BAND"
    SPHERE = "SPHERE"
    CONE = "CONE"
    LEVEL_SET = "LEVEL_SET"
    TILE_MASK = "TILE_MASK"
    FULL = "FULL"
    EMPTY = "EMPTY"
    UNKNOWN = "UNKNOWN"


class BackendResidency(StrEnum):
    HOST = "HOST"
    CPU = "CPU"
    GPU = "GPU"
    SPU = "SPU"
    EXTERNAL = "EXTERNAL"


class Exactness(StrEnum):
    EXACT = "EXACT"
    BOUNDED = "BOUNDED"
    APPROXIMATE = "APPROXIMATE"
    REFERENCE = "REFERENCE"


def _enum(value: Any, kind: type[StrEnum]) -> Any:
    if isinstance(value, kind):
        return value
    name = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return kind(name)
    except ValueError:
        return kind[name]


def _nonnegative(name: str, value: float, *, allow_inf: bool = True) -> float:
    result = float(value)
    if math.isnan(result) or result < 0.0 or (not allow_inf and not math.isfinite(result)):
        raise ValueError(f"{name} must be a non-negative{' finite' if not allow_inf else ''} value")
    return result


@dataclass(frozen=True, slots=True)
class ErrorContract:
    absolute_max: float = math.inf
    relative_max: float = math.inf
    phase_max: float = math.inf
    temporal_max: float = math.inf
    perceptual_target: str | None = None
    phase_required: bool = False

    def __post_init__(self) -> None:
        for name in ("absolute_max", "relative_max", "phase_max", "temporal_max"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        target = None if self.perceptual_target is None else str(self.perceptual_target).strip()
        object.__setattr__(self, "perceptual_target", target or None)
        object.__setattr__(self, "phase_required", bool(self.phase_required))

    def accepts(self, ledger: ErrorLedger, *, scale: float = 1.0, phase_error: float = 0.0) -> bool:
        total = float(ledger.total_bound)
        if not math.isfinite(total):
            return False
        denominator = max(abs(float(scale)), np.finfo(np.float64).tiny)
        return (
            total <= self.absolute_max
            and total / denominator <= self.relative_max
            and float(phase_error) <= self.phase_max
        )


@dataclass(frozen=True, slots=True)
class ErrorBudget:
    absolute: float = math.inf
    relative: float = math.inf
    phase: float = math.inf
    temporal: float = math.inf
    perceptual: str | None = None

    def __post_init__(self) -> None:
        for name in ("absolute", "relative", "phase", "temporal"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))

    def as_contract(self, *, phase_required: bool = False) -> ErrorContract:
        return ErrorContract(
            absolute_max=self.absolute,
            relative_max=self.relative,
            phase_max=self.phase,
            temporal_max=self.temporal,
            perceptual_target=self.perceptual,
            phase_required=phase_required,
        )


@dataclass(frozen=True, slots=True)
class PrecisionContract:
    mode: PrecisionMode | str = PrecisionMode.BALANCED
    log_guard_bits: int = 8
    phase_guard_bits: int = 8
    residual_log_bits: int = 12
    residual_phase_bits: int = 12
    accumulator_fraction_bits: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", _enum(self.mode, PrecisionMode))
        for name in ("log_guard_bits", "phase_guard_bits", "residual_log_bits", "residual_phase_bits"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if self.accumulator_fraction_bits is not None:
            bits = int(self.accumulator_fraction_bits)
            if bits < 0:
                raise ValueError("accumulator_fraction_bits must be non-negative")
            object.__setattr__(self, "accumulator_fraction_bits", bits)

    @classmethod
    def for_mode(cls, mode: PrecisionMode | str) -> PrecisionContract:
        selected = _enum(mode, PrecisionMode)
        if selected is PrecisionMode.FAST:
            return cls(selected, 6, 6, 8, 10, 20)
        if selected is PrecisionMode.ACCURATE:
            return cls(selected, 12, 12, 16, 16, 40)
        if selected is PrecisionMode.EXACT:
            return cls(selected, 16, 16, 16, 16, 52)
        return cls(selected, 8, 8, 12, 12, 32)


@dataclass(frozen=True, slots=True)
class CoordinateChart:
    origin: tuple[float, ...] = ()
    basis: tuple[tuple[float, ...], ...] = ()
    scale: tuple[float, ...] = ()
    periodicity: tuple[float | None, ...] = ()
    sample_spacing: tuple[float, ...] = ()
    frequency_convention: str = "cycles"

    def __post_init__(self) -> None:
        origin = tuple(float(value) for value in self.origin)
        dimension = len(origin)
        basis = self.basis or tuple(
            tuple(1.0 if row == column else 0.0 for column in range(dimension)) for row in range(dimension)
        )
        normalized_basis = tuple(tuple(float(value) for value in row) for row in basis)
        if any(len(row) != dimension for row in normalized_basis) or len(normalized_basis) != dimension:
            raise ValueError("coordinate chart basis must be square and match origin dimension")
        scale = tuple(float(value) for value in (self.scale or (1.0,) * dimension))
        spacing = tuple(float(value) for value in (self.sample_spacing or (1.0,) * dimension))
        periodicity = tuple(self.periodicity or (None,) * dimension)
        if len(scale) != dimension or len(spacing) != dimension or len(periodicity) != dimension:
            raise ValueError("coordinate chart vectors must share one dimension")
        if any(value == 0.0 or not math.isfinite(value) for value in scale):
            raise ValueError("coordinate chart scale values must be finite and nonzero")
        if any(value <= 0.0 or not math.isfinite(value) for value in spacing):
            raise ValueError("sample spacing values must be positive and finite")
        normalized_periodicity = tuple(
            None if value is None else _nonnegative("periodicity", value, allow_inf=False) for value in periodicity
        )
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "basis", normalized_basis)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "periodicity", normalized_periodicity)
        object.__setattr__(self, "sample_spacing", spacing)
        object.__setattr__(self, "frequency_convention", str(self.frequency_convention))

    @property
    def dimension(self) -> int:
        return len(self.origin)

    @classmethod
    def identity(cls, dimension: int, *, spacing: float | Sequence[float] = 1.0) -> CoordinateChart:
        size = int(dimension)
        if size < 0:
            raise ValueError("dimension must be non-negative")
        samples = (
            tuple(float(value) for value in spacing)
            if isinstance(spacing, Sequence) and not isinstance(spacing, (str, bytes))
            else (float(cast(float, spacing)),) * size
        )
        return cls(origin=(0.0,) * size, sample_spacing=samples)

    def coordinates(self, index: Sequence[float]) -> np.ndarray:
        point = np.asarray(tuple(index), dtype=np.float64)
        if point.shape != (self.dimension,):
            raise ValueError("coordinate index dimension does not match chart")
        basis = np.asarray(self.basis, dtype=np.float64)
        scaled = point * np.asarray(self.sample_spacing) * np.asarray(self.scale)
        return np.asarray(np.asarray(self.origin) + basis @ scaled, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Region:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        lower = tuple(float(value) for value in self.lower)
        upper = tuple(float(value) for value in self.upper)
        if not lower or len(lower) != len(upper):
            raise ValueError("region bounds must be non-empty and share one dimension")
        if any(not math.isfinite(value) for value in lower + upper):
            raise ValueError("region bounds must be finite")
        if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
            raise ValueError("region lower bounds must not exceed upper bounds")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def extent(self) -> tuple[float, ...]:
        return tuple(hi - lo for lo, hi in zip(self.lower, self.upper, strict=True))

    def intersects(self, other: Region) -> bool:
        if self.dimension != other.dimension:
            return False
        return all(
            left_hi >= right_lo and right_hi >= left_lo
            for left_lo, left_hi, right_lo, right_hi in zip(
                self.lower, self.upper, other.lower, other.upper, strict=True
            )
        )

    def contains(self, point: Sequence[float]) -> bool:
        values = tuple(float(value) for value in point)
        return len(values) == self.dimension and all(
            lo <= value <= hi for lo, hi, value in zip(self.lower, self.upper, values, strict=True)
        )


@dataclass(frozen=True, slots=True)
class Support:
    kind: SupportKind | str = SupportKind.UNKNOWN
    region: Region | None = None
    tiles: frozenset[tuple[int, ...]] = frozenset()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        selected = _enum(self.kind, SupportKind)
        tiles = frozenset(tuple(int(value) for value in index) for index in self.tiles)
        if selected is SupportKind.AABB and self.region is None:
            raise ValueError("AABB support requires a region")
        if selected is SupportKind.TILE_MASK and not tiles:
            raise ValueError("tile-mask support requires at least one tile")
        object.__setattr__(self, "kind", selected)
        object.__setattr__(self, "tiles", tiles)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def full(cls) -> Support:
        return cls(SupportKind.FULL)

    @classmethod
    def empty(cls) -> Support:
        return cls(SupportKind.EMPTY)

    @classmethod
    def aabb(cls, lower: Sequence[float], upper: Sequence[float]) -> Support:
        return cls(SupportKind.AABB, Region(tuple(lower), tuple(upper)))

    def intersects(self, region: Region) -> bool:
        if self.kind is SupportKind.EMPTY:
            return False
        if self.kind in {SupportKind.FULL, SupportKind.UNKNOWN}:
            return True
        if self.region is not None:
            return self.region.intersects(region)
        return True

    def intersection(self, other: Support) -> Support:
        if self.kind is SupportKind.EMPTY or other.kind is SupportKind.EMPTY:
            return Support.empty()
        if self.kind in {SupportKind.FULL, SupportKind.UNKNOWN}:
            return other
        if other.kind in {SupportKind.FULL, SupportKind.UNKNOWN}:
            return self
        if self.region is not None and other.region is not None:
            if not self.region.intersects(other.region):
                return Support.empty()
            lower = tuple(max(a, b) for a, b in zip(self.region.lower, other.region.lower, strict=True))
            upper = tuple(min(a, b) for a, b in zip(self.region.upper, other.region.upper, strict=True))
            return Support.aabb(lower, upper)
        if self.kind is SupportKind.TILE_MASK and other.kind is SupportKind.TILE_MASK:
            overlap = self.tiles & other.tiles
            return Support(SupportKind.TILE_MASK, tiles=overlap) if overlap else Support.empty()
        return Support(SupportKind.UNKNOWN)


def _payload_nbytes(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return len(payload)
    # Python numeric scalars are the header words of structured carriers
    # (e.g. Gaussian ``kappa`` and Jet2D coefficients/guards).  They do not
    # expose ``nbytes`` themselves, but their widened NumPy storage width is
    # part of the serialized representation.  Keep strings/enums and boolean
    # flags metadata zero-cost; the latter are not widened payload words.
    if isinstance(payload, (bool, np.bool_)):
        return 0
    if isinstance(payload, (int, float, complex, np.integer, np.floating, np.complexfloating)):
        return int(np.asarray(payload).nbytes)
    if isinstance(payload, Mapping):
        return sum(_payload_nbytes(key) + _payload_nbytes(value) for key, value in payload.items())
    if isinstance(payload, (tuple, list)):
        return sum(_payload_nbytes(value) for value in payload)
    # Structured payloads commonly use frozen slot dataclasses (Gaussian
    # atoms, jets, Hermitian carriers).  ``getattr(payload, 'nbytes', 0)``
    # silently reported zero for these values, which made representation
    # planning believe that an analytic carrier was free and that a dense
    # half-spectrum had no storage.  Walk declared dataclass fields so the
    # accounting remains structural without inspecting arbitrary objects.
    if is_dataclass(payload) and not isinstance(payload, type):
        return sum(_payload_nbytes(getattr(payload, item.name)) for item in dataclass_fields(payload))
    return int(getattr(payload, "nbytes", 0))


def _freeze_value(value: Any) -> Any:
    if isinstance(value, AtomMixture):
        return replace(
            value, atoms=tuple(_freeze_value(a) for a in value.atoms), residual=_freeze_value(value.residual)
        )
    if isinstance(value, GaussianChirpAtom):
        # A frozen dataclass alone does not protect its ndarray parameters.
        # Field preparation must own these arrays to keep cached plans valid.
        return replace(value, h=_freeze_value(value.h), C=_freeze_value(value.C))
    if isinstance(value, np.ndarray):
        result = value.copy()
        result.setflags(write=False)
        return result
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    return value


_STATE_BY_REPRESENTATION = {
    RepresentationKind.GAUSSIAN_CHIRP_ATOM: MaterializationState.ANALYTIC_ATOM,
    RepresentationKind.SPHERICAL_GAUSSIAN_ATOM: MaterializationState.ANALYTIC_ATOM,
    RepresentationKind.ATOM_MIXTURE: MaterializationState.SYMBOLIC_ATOMS,
    RepresentationKind.QUADRATIC_JET: MaterializationState.QUADRATIC_JET,
    RepresentationKind.SEPARABLE: MaterializationState.SEPARABLE,
    RepresentationKind.SPARSE_RESIDUAL: MaterializationState.SPARSE_RESIDUAL,
    RepresentationKind.HERMITIAN_HALF: MaterializationState.HERMITIAN_HALF,
    RepresentationKind.DENSE_TYPED: MaterializationState.DENSE_TYPED,
    RepresentationKind.DENSE_CARTESIAN_REFERENCE: MaterializationState.DENSE_CARTESIAN_REFERENCE,
}


@dataclass(frozen=True, slots=True)
class FieldRepresentation:
    kind: RepresentationKind | str
    payload: Any
    exactness: Exactness | str = Exactness.BOUNDED
    error: ErrorLedger = field(default_factory=ErrorLedger)
    byte_size: int | None = None
    evaluation_cost: float = 0.0
    transform_cost: float = 0.0
    materialization_cost: float = 0.0
    residual: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Indices into the owning ``Field.operations`` sequence that have already
    # been incorporated into this representation by a planner/compiler.
    # Indices (rather than operation names) keep repeated operations
    # unambiguous and form an explicit exactly-once handoff to the runtime.
    consumed_operation_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        selected = _enum(self.kind, RepresentationKind)
        exactness = _enum(self.exactness, Exactness)
        if not isinstance(self.error, ErrorLedger):
            raise TypeError("representation error must be an ErrorLedger")
        payload = _freeze_value(self.payload)
        residual = _freeze_value(self.residual)
        size = _payload_nbytes(payload) if self.byte_size is None else int(self.byte_size)
        if size < 0:
            raise ValueError("representation byte_size must be non-negative")
        consumed = tuple(int(index) for index in self.consumed_operation_indices)
        if any(index < 0 for index in consumed):
            raise ValueError("consumed operation indices must be non-negative")
        if len(set(consumed)) != len(consumed):
            raise ValueError("consumed operation indices must be unique")
        object.__setattr__(self, "kind", selected)
        object.__setattr__(self, "exactness", exactness)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "byte_size", size)
        object.__setattr__(self, "consumed_operation_indices", consumed)
        for name in ("evaluation_cost", "transform_cost", "materialization_cost"):
            object.__setattr__(self, name, _nonnegative(name, getattr(self, name)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def state(self) -> MaterializationState:
        return _STATE_BY_REPRESENTATION[cast(RepresentationKind, self.kind)]


@dataclass(frozen=True, slots=True)
class FieldOperation:
    op: str
    params: Mapping[str, Any] = field(default_factory=dict)
    error: ErrorLedger = field(default_factory=ErrorLedger)

    def __post_init__(self) -> None:
        operation = str(self.op).strip().lower().replace("-", "_")
        if not operation:
            raise ValueError("field operation name must be non-empty")
        if not isinstance(self.error, ErrorLedger):
            raise TypeError("field operation error must be an ErrorLedger")
        object.__setattr__(self, "op", operation)
        frozen = {key: _freeze_value(value) for key, value in self.params.items()}
        object.__setattr__(self, "params", MappingProxyType(frozen))


_DEFAULT_COHERENCE = {
    FieldKind.LOG_POS: Coherence.INCOHERENT_POSITIVE,
    FieldKind.RADIANCE: Coherence.INCOHERENT_POSITIVE,
    FieldKind.INCOHERENT_RADIANCE: Coherence.INCOHERENT_POSITIVE,
    FieldKind.LOG_SIGNED: Coherence.SIGNED_REAL,
    FieldKind.SPF_COMPLEX: Coherence.COHERENT_COMPLEX,
    FieldKind.PHASE: Coherence.PHASE_ONLY,
}


@dataclass(frozen=True, slots=True)
class Field:
    id: str
    domain: Domain | str
    semantic: Semantic | str
    coherence: Coherence | str
    value_kind: FieldKind | str
    representation: FieldRepresentation
    chart: CoordinateChart
    support: Support = field(default_factory=Support.full)
    error: ErrorContract = field(default_factory=ErrorContract)
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    operations: tuple[FieldOperation, ...] = ()
    residency: BackendResidency | str = BackendResidency.HOST
    inputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identifier = str(self.id).strip()
        if not identifier:
            raise ValueError("field id must be non-empty")
        kind = self.value_kind if isinstance(self.value_kind, FieldKind) else FieldKind(str(self.value_kind).lower())
        coherence = _enum(self.coherence, Coherence)
        expected = _DEFAULT_COHERENCE.get(kind)
        if expected is not None and coherence is not expected:
            raise ValueError(f"{kind.value} requires {expected.value} coherence, received {coherence.value}")
        if not isinstance(self.representation, FieldRepresentation):
            raise TypeError("field representation must be a FieldRepresentation")
        if not isinstance(self.chart, CoordinateChart):
            raise TypeError("field chart must be a CoordinateChart")
        if not isinstance(self.support, Support):
            raise TypeError("field support must be Support")
        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "domain", _enum(self.domain, Domain))
        object.__setattr__(self, "semantic", _enum(self.semantic, Semantic))
        object.__setattr__(self, "coherence", coherence)
        object.__setattr__(self, "value_kind", kind)
        object.__setattr__(self, "operations", tuple(self.operations))
        object.__setattr__(self, "residency", _enum(self.residency, BackendResidency))
        object.__setattr__(self, "inputs", tuple(str(value) for value in self.inputs))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_value(
        cls,
        value: Any,
        *,
        id: str = "field",
        value_kind: FieldKind | str = FieldKind.DENSE,
        domain: Domain | str = Domain.CUSTOM,
        semantic: Semantic | str = Semantic.GENERIC_FIELD,
        coherence: Coherence | str | None = None,
        chart: CoordinateChart | None = None,
        support: Support | None = None,
        error: ErrorContract | None = None,
        precision: PrecisionContract | None = None,
        representation: RepresentationKind | str | None = None,
    ) -> Field:
        from spf.renderer.blocks import BlockView
        from spf.renderer.carriers import as_block_view

        block = as_block_view(value)
        if block is not None:
            value = block
        if isinstance(value, BlockView) and value_kind == FieldKind.DENSE:
            value_kind = FieldKind.SPF_COMPLEX
        if isinstance(value, (GaussianChirpAtom, AtomMixture)):
            if value_kind == FieldKind.DENSE:
                value_kind = FieldKind.SPF_COMPLEX
            if representation is None:
                representation = (
                    RepresentationKind.GAUSSIAN_CHIRP_ATOM
                    if isinstance(value, GaussianChirpAtom)
                    else RepresentationKind.ATOM_MIXTURE
                )
            value = _freeze_value(value)
        kind = value_kind if isinstance(value_kind, FieldKind) else FieldKind(str(value_kind).lower())
        selected_coherence = coherence or _DEFAULT_COHERENCE.get(kind, Coherence.NOT_APPLICABLE)
        rep_kind = representation or default_dense_representation(kind, value)
        array_dimension = (
            value.dimension
            if isinstance(value, GaussianChirpAtom)
            else value.atoms[0].dimension
            if isinstance(value, AtomMixture) and value.atoms
            else len(value.shape)
            if isinstance(value, BlockView)
            else int(np.asarray(value).ndim)
            if isinstance(value, (np.ndarray, list, tuple))
            else 0
        )
        return cls(
            id=id,
            domain=domain,
            semantic=semantic,
            coherence=selected_coherence,
            value_kind=kind,
            representation=FieldRepresentation(
                rep_kind,
                value,
                byte_size=value.nbytes if isinstance(value, BlockView) else None,
                metadata={"shape": value.shape, "structural_block": True} if isinstance(value, BlockView) else {},
            ),
            chart=chart or CoordinateChart.identity(array_dimension),
            support=support or Support.full(),
            error=error or ErrorContract(),
            precision=precision or PrecisionContract(),
        )

    @classmethod
    def from_texture(
        cls,
        texture: Any,
        *,
        id: str = "texture",
        value_kind: FieldKind | str = FieldKind.DENSE,
        semantic: Semantic | str = Semantic.GENERIC_FIELD,
        coherence: Coherence | str | None = None,
        precision: PrecisionContract | None = None,
    ) -> Field:
        """Bind a :class:`SpectralTexture` without materializing it eagerly."""

        from spf.renderer.assets import SpectralTexture

        if not isinstance(texture, SpectralTexture):
            raise TypeError("Field.from_texture requires a SpectralTexture")
        kind = value_kind if isinstance(value_kind, FieldKind) else FieldKind(str(value_kind).lower())
        selected_coherence = coherence or _DEFAULT_COHERENCE.get(kind, Coherence.NOT_APPLICABLE)
        error_bound = float(texture.error_bound)
        return cls(
            id=id,
            domain=Domain.SPATIAL_2D,
            semantic=semantic,
            coherence=selected_coherence,
            value_kind=kind,
            representation=FieldRepresentation(
                RepresentationKind.DENSE_TYPED,
                texture,
                exactness=Exactness.BOUNDED if error_bound else Exactness.EXACT,
                error=ErrorLedger(representation_fit=error_bound),
                byte_size=texture.byte_size,
                metadata={
                    "shape": texture.shape,
                    "block_shape": texture.block_shape,
                    "asset_representations": tuple(item.value for item in texture.representation),
                    "asset_color_space": texture.color_space,
                },
            ),
            chart=CoordinateChart.identity(2),
            support=Support.aabb((0.0, 0.0), tuple(float(max(0, size - 1)) for size in reversed(texture.shape))),
            error=ErrorContract(absolute_max=error_bound, relative_max=math.inf),
            precision=precision or PrecisionContract(),
            metadata={"shape": texture.shape, "spectral_texture": True},
        )

    @property
    def state(self) -> MaterializationState:
        return self.representation.state

    @property
    def ledger(self) -> ErrorLedger:
        from spf.renderer.normalization import _field_ledger_after_operations

        return _field_ledger_after_operations(self)

    def with_operation(self, op: str, /, **params: Any) -> Field:
        return replace(self, operations=(*self.operations, FieldOperation(op, params)))

    def translate(
        self,
        *offset: float,
        dx: float | None = None,
        dy: float | None = None,
        dz: float | None = None,
    ) -> Field:
        vector = tuple(float(value) for value in offset)
        if not vector:
            supplied = tuple(value for value in (dx, dy, dz) if value is not None)
            vector = tuple(float(value) for value in supplied)
        if len(vector) != self.chart.dimension:
            raise ValueError("translation dimension must match the field chart")
        return self.with_operation("translate", offset=vector)

    def gaussian_filter(self, covariance: Any) -> Field:
        matrix = np.asarray(covariance, dtype=np.float64)
        if matrix.shape != (self.chart.dimension, self.chart.dimension):
            raise ValueError("filter covariance must match the field chart")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("filter covariance must be symmetric")
        if np.min(np.linalg.eigvalsh(matrix)) < -1.0e-12:
            raise ValueError("filter covariance must be positive semidefinite")
        return self.with_operation("gaussian_filter", covariance=matrix.copy())

    def expose(self, ev: float) -> Field:
        if self.coherence not in {Coherence.INCOHERENT_POSITIVE, Coherence.COHERENT_COMPLEX}:
            raise TypeError("exposure requires a positive or coherent-complex field")
        return self.with_operation("exposure", ev=float(ev))

    def lens_phase(self, *, focal_length: float, wavelength: float) -> Field:
        if self.coherence is not Coherence.COHERENT_COMPLEX:
            raise TypeError("lens phase requires a coherent-complex field")
        focal = float(focal_length)
        wave = float(wavelength)
        if focal == 0.0 or wave <= 0.0 or not math.isfinite(focal) or not math.isfinite(wave):
            raise ValueError("lens focal length must be finite/nonzero and wavelength positive")
        return self.with_operation("lens_phase", focal_length=focal, wavelength=wave)

    def bandlimit(self, cutoff: float, *, window: str = "raised_cosine") -> Field:
        return self.with_operation("bandlimit", cutoff=_nonnegative("cutoff", cutoff), window=str(window))

    def mask(self, mask: Any) -> Field:
        return self.with_operation("mask", mask=np.asarray(mask, dtype=bool).copy())

    def materialize(self, request: MaterializationRequest | None = None, **options: Any) -> Any:
        """Execute the deferred field program at one explicit boundary.

        The high-level design spells the requested value kind as ``output``;
        the dataclass uses ``output_kind`` to distinguish it from the carrier
        ``output_format``.  Accept both spellings here, plus convenient string
        precision modes and scalar absolute-error budgets.
        """

        from spf.renderer.compiler import materialize_field

        return_report = bool(options.pop("return_report", options.pop("report", False)))
        return_result = bool(options.pop("return_result", False))
        return_certificate = bool(options.pop("return_certificate", False))
        if request is not None and options:
            raise TypeError("pass either a MaterializationRequest or keyword options, not both")
        if "output" in options:
            if "output_kind" in options:
                raise TypeError("pass only one of output or output_kind")
            options["output_kind"] = options.pop("output")
        if isinstance(options.get("precision"), (str, PrecisionMode)):
            options["precision"] = PrecisionContract.for_mode(options["precision"])
        budget = options.get("error_budget")
        if budget is not None and not isinstance(budget, ErrorBudget):
            options["error_budget"] = ErrorBudget(absolute=float(budget))
        selected = request or MaterializationRequest(**options)
        return materialize_field(
            self,
            selected,
            return_report=return_report,
            return_result=return_result,
            return_certificate=return_certificate,
        )

    def prepare(self, request: MaterializationRequest | None = None) -> Any:
        """Prepare once, then execute repeatedly without replanning."""
        from spf.renderer.compiler import prepare_materialization

        return prepare_materialization(self, request)


@dataclass(frozen=True, slots=True)
class SampleLayout:
    shape: tuple[int, ...]
    origin: tuple[float, ...] = ()
    spacing: tuple[float, ...] = ()
    order: str = "C"

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        if not shape or any(value < 0 for value in shape):
            raise ValueError("sample layout shape must be non-empty and non-negative")
        origin = tuple(float(value) for value in (self.origin or (0.0,) * len(shape)))
        spacing = tuple(float(value) for value in (self.spacing or (1.0,) * len(shape)))
        if len(origin) != len(shape) or len(spacing) != len(shape):
            raise ValueError("sample layout vectors must match shape dimension")
        if any(value <= 0.0 or not math.isfinite(value) for value in spacing):
            raise ValueError("sample layout spacing must be positive and finite")
        order = str(self.order).upper()
        if order not in {"C", "F"}:
            raise ValueError("sample layout order must be C or F")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "order", order)

    @property
    def element_count(self) -> int:
        return int(math.prod(self.shape))

    def points(self) -> np.ndarray:
        axes = [
            origin + spacing * np.arange(size)
            for size, origin, spacing in zip(self.shape, self.origin, self.spacing, strict=True)
        ]
        mesh = np.meshgrid(*axes, indexing="ij")
        stacked = np.asarray(np.stack(mesh, axis=-1), dtype=np.float64)
        output_shape = (-1, len(self.shape))
        if self.order == "F":
            return np.asarray(stacked.reshape(output_shape, order="F"), dtype=np.float64)
        return np.asarray(stacked.reshape(output_shape, order="C"), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class MaterializationRequest:
    layout: SampleLayout | None = None
    region: Region | None = None
    output_kind: FieldKind | str = FieldKind.DENSE
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    backend: str = "reference"
    output_format: str = "native"

    def __post_init__(self) -> None:
        kind = self.output_kind if isinstance(self.output_kind, FieldKind) else FieldKind(str(self.output_kind).lower())
        object.__setattr__(self, "output_kind", kind)
        object.__setattr__(self, "backend", str(self.backend).lower().replace("-", "_"))
        object.__setattr__(self, "output_format", str(self.output_format).lower().replace("-", "_"))


__all__ = [
    "BackendResidency",
    "Coherence",
    "CoordinateChart",
    "Domain",
    "ErrorBudget",
    "ErrorContract",
    "Exactness",
    "Field",
    "FieldOperation",
    "FieldRepresentation",
    "MaterializationRequest",
    "MaterializationState",
    "PrecisionContract",
    "PrecisionMode",
    "Region",
    "RepresentationKind",
    "SampleLayout",
    "Semantic",
    "Support",
    "SupportKind",
]
