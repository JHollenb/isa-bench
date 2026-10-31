"""Native typed producer-to-consumer rendering for analytically closed fields.

The dense packed FFT is a required fallback, but it is not the preferred SFR
execution model.  This module implements the v0.2 path for a Gaussian--chirp
field: producers submit typed atom parameters, compatible operations remain
symbolic, the inverse Fourier transform is performed on the parameter block,
and only the requested tile is lowered to guarded quadratic jets.  Magnitude,
intensity, signed-real, and display projections never construct a Cartesian
complex intermediate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from enum import StrEnum
from functools import lru_cache
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.model import (
    Coherence,
    CoordinateChart,
    Domain,
    ErrorContract,
    Exactness,
    Field,
    FieldRepresentation,
    PrecisionContract,
    Region,
    RepresentationKind,
    Semantic,
    Support,
    SupportKind,
)
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision
from spf.renderer.v02 import FieldKind
from spf.tensor import v02, v02_planes


class OutputSemantic(StrEnum):
    """Physical observable requested from a native field."""

    MAGNITUDE = "magnitude"
    INTENSITY = "intensity"
    REAL_FIELD = "real_field"
    RADIANCE = "radiance"
    LOG_RADIANCE = "log_radiance"
    DISPLAY_RGB = "display_rgb"


def _output_semantic(value: OutputSemantic | str) -> OutputSemantic:
    if isinstance(value, OutputSemantic):
        return value
    name = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {"real": "real_field", "display": "display_rgb"}
    return OutputSemantic(aliases.get(name, name))


@dataclass(frozen=True, slots=True)
class NativeGrid2D:
    """A row-major 2-D sample grid with explicit physical coordinates.

    ``shape`` is ``(height, width)`` while ``origin`` and ``spacing`` are
    ``(x, y)``.  ``potential_shape`` names the dense parent field for an ROI;
    it is used only for materialization accounting.
    """

    shape: tuple[int, int]
    origin: tuple[float, float] = (0.0, 0.0)
    spacing: tuple[float, float] = (1.0, 1.0)
    potential_shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        origin = tuple(float(value) for value in self.origin)
        spacing = tuple(float(value) for value in self.spacing)
        potential = shape if self.potential_shape is None else tuple(int(value) for value in self.potential_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("native grid shape must contain two positive dimensions")
        if len(origin) != 2 or any(not math.isfinite(value) for value in origin):
            raise ValueError("native grid origin must contain two finite coordinates")
        if len(spacing) != 2 or any(value <= 0.0 or not math.isfinite(value) for value in spacing):
            raise ValueError("native grid spacing must contain two positive finite values")
        if len(potential) != 2 or any(value < actual for value, actual in zip(potential, shape, strict=True)):
            raise ValueError("potential shape must contain the requested grid")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "potential_shape", potential)

    @property
    def size(self) -> int:
        return int(math.prod(self.shape))

    @property
    def potential_size(self) -> int:
        return int(math.prod(self.potential_shape or self.shape))


def _continuity_metadata(
    *,
    native_inputs: int = 0,
    legacy_imports: int = 0,
    legacy_exports: int = 0,
    cartesian_intermediates: int = 0,
    native_packs: int = 0,
    native_quantizations: int = 0,
    symbolic_operations: int = 0,
    potential_dense_passes: int = 1,
    packed_atom_arithmetic: int = 0,
    packed_atom_decodes: int = 0,
    rom_logadds: int = 0,
    typed_pivot_reductions: int = 0,
    hermitian_syntheses: int = 0,
    footprint_integrations: int = 0,
) -> dict[str, int]:
    return {
        "native_input_count": int(native_inputs),
        "legacy_import_count": int(legacy_imports),
        "legacy_export_count": int(legacy_exports),
        "cartesian_intermediate_count": int(cartesian_intermediates),
        "native_pack_count": int(native_packs),
        "native_quantization_count": int(native_quantizations),
        "symbolic_operation_count": int(symbolic_operations),
        "potential_dense_passes": int(potential_dense_passes),
        "packed_atom_arithmetic_count": int(packed_atom_arithmetic),
        "packed_atom_decode_count": int(packed_atom_decodes),
        "rom_logadd_count": int(rom_logadds),
        "typed_pivot_reduction_count": int(typed_pivot_reductions),
        "hermitian_synthesis_count": int(hermitian_syntheses),
        "footprint_integration_count": int(footprint_integrations),
    }


def _field_continuity(field: Field) -> dict[str, int]:
    raw = field.metadata.get("native_continuity", {})
    base = _continuity_metadata()
    if isinstance(raw, dict) or hasattr(raw, "items"):
        for key, value in raw.items():
            if key in base:
                base[key] = int(value)
    return base


def _merge_continuity(left: Field, right: Field, *, operation: bool = True) -> dict[str, int]:
    a = _field_continuity(left)
    b = _field_continuity(right)
    merged = {key: a[key] + b[key] for key in a if key != "potential_dense_passes"}
    merged["symbolic_operation_count"] += int(operation)
    merged["potential_dense_passes"] = a["potential_dense_passes"] + b["potential_dense_passes"] + int(operation)
    return merged


@dataclass(frozen=True, slots=True)
class NativeField(Field):
    """A :class:`Field` whose constructor records a native typed boundary."""

    @classmethod
    def gaussian_chirp(
        cls,
        atom: v02.GaussianChirpAtom,
        *,
        id: str = "gaussian_chirp",
        domain: Domain | str = Domain.FOURIER_2D,
        semantic: Semantic | str | None = None,
        coherent: bool = True,
        chart: CoordinateChart | None = None,
        support: Support | None = None,
        error: ErrorContract | None = None,
        precision: PrecisionContract | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NativeField:
        if not isinstance(atom, v02.GaussianChirpAtom):
            raise TypeError("NativeField.gaussian_chirp requires GaussianChirpAtom")
        if atom.dimension != 2:
            raise ValueError("native Gaussian--chirp rendering currently requires a two-dimensional atom")
        coherence = Coherence.COHERENT_COMPLEX if coherent else Coherence.INCOHERENT_POSITIVE
        value_kind = FieldKind.SPF_COMPLEX if coherent else FieldKind.LOG_POS
        selected_semantic = (
            (Semantic.COMPLEX_OPTICAL_AMPLITUDE if coherent else Semantic.RADIANCE) if semantic is None else semantic
        )
        details = dict(metadata or {})
        details.setdefault("native_continuity", _continuity_metadata(native_inputs=1))
        return cls(
            id=id,
            domain=domain,
            semantic=selected_semantic,
            coherence=coherence,
            value_kind=value_kind,
            representation=FieldRepresentation(
                RepresentationKind.GAUSSIAN_CHIRP_ATOM,
                atom,
                exactness=Exactness.EXACT,
                byte_size=int((1 + atom.dimension + atom.dimension * atom.dimension) * 16),
            ),
            chart=chart or CoordinateChart.identity(2),
            support=support or Support.full(),
            error=error or ErrorContract(),
            precision=precision or PrecisionContract(),
            metadata=details,
        )

    @classmethod
    def atom_mixture(
        cls,
        mixture: v02.AtomMixture,
        *,
        id: str = "atom_mixture",
        domain: Domain | str = Domain.SPATIAL_2D,
        semantic: Semantic | str | None = None,
        coherent: bool = True,
        chart: CoordinateChart | None = None,
        support: Support | None = None,
        error: ErrorContract | None = None,
        precision: PrecisionContract | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NativeField:
        """Submit a symbolic atom sum without rasterizing its terms."""

        if not isinstance(mixture, v02.AtomMixture):
            raise TypeError("NativeField.atom_mixture requires AtomMixture")
        if any(atom.dimension != 2 for atom in mixture.atoms):
            raise ValueError("native atom mixtures currently require two-dimensional atoms")
        coherence = Coherence.COHERENT_COMPLEX if coherent else Coherence.INCOHERENT_POSITIVE
        value_kind = FieldKind.SPF_COMPLEX if coherent else FieldKind.LOG_POS
        selected_semantic = (
            (Semantic.COMPLEX_OPTICAL_AMPLITUDE if coherent else Semantic.RADIANCE) if semantic is None else semantic
        )
        details = dict(metadata or {})
        details.setdefault("native_continuity", _continuity_metadata(native_inputs=1))
        atom_bytes = len(mixture.atoms) * (1 + 2 + 4) * np.dtype(np.uint32).itemsize
        return cls(
            id=id,
            domain=domain,
            semantic=selected_semantic,
            coherence=coherence,
            value_kind=value_kind,
            representation=FieldRepresentation(
                RepresentationKind.ATOM_MIXTURE,
                mixture,
                exactness=Exactness.EXACT,
                byte_size=atom_bytes,
            ),
            chart=chart or CoordinateChart.identity(2),
            support=support or Support.full(),
            error=error or ErrorContract(),
            precision=precision or PrecisionContract(),
            metadata=details,
        )

    @classmethod
    def log_positive(
        cls,
        values: Any,
        *,
        id: str = "log_positive",
        semantic: Semantic | str = Semantic.RADIANCE,
        domain: Domain | str = Domain.SPATIAL_2D,
    ) -> NativeField:
        array = np.asarray(values, dtype=object if np.asarray(values).dtype == object else None)
        return cls._dense_native(array, id=id, kind=FieldKind.LOG_POS, semantic=semantic, domain=domain)

    @classmethod
    def phase(
        cls,
        values: Any,
        *,
        id: str = "phase",
        domain: Domain | str = Domain.FOURIER_2D,
    ) -> NativeField:
        return cls._dense_native(values, id=id, kind=FieldKind.PHASE, semantic=Semantic.GENERIC_FIELD, domain=domain)

    @classmethod
    def spf_complex(
        cls,
        words: Any,
        *,
        id: str = "spf_complex",
        domain: Domain | str = Domain.FOURIER_2D,
    ) -> NativeField:
        packed = np.asarray(words)
        if packed.dtype != np.uint32:
            raise TypeError("NativeField.spf_complex requires canonical uint32 SPF32 words")
        return cls._dense_native(
            packed,
            id=id,
            kind=FieldKind.SPF_COMPLEX,
            semantic=Semantic.COMPLEX_OPTICAL_AMPLITUDE,
            domain=domain,
            native_packs=1,
        )

    @classmethod
    def jet2d(
        cls,
        jet: v02.Jet2D,
        *,
        id: str = "jet2d",
        semantic: Semantic | str = Semantic.GENERIC_FIELD,
        domain: Domain | str = Domain.SPATIAL_2D,
    ) -> NativeField:
        if not isinstance(jet, v02.Jet2D):
            raise TypeError("NativeField.jet2d requires Jet2D")
        kind = FieldKind.PHASE if jet.value_kind == "phase" else FieldKind.LOG_POS
        coherence = Coherence.PHASE_ONLY if kind is FieldKind.PHASE else Coherence.INCOHERENT_POSITIVE
        return cls(
            id=id,
            domain=domain,
            semantic=semantic,
            coherence=coherence,
            value_kind=kind,
            representation=FieldRepresentation(RepresentationKind.QUADRATIC_JET, jet, exactness=Exactness.EXACT),
            chart=CoordinateChart.identity(2),
            metadata={"native_continuity": _continuity_metadata(native_inputs=1)},
        )

    @classmethod
    def hermitian_spectrum(
        cls,
        half_spectrum: Any,
        *,
        full_shape: tuple[int, int],
        id: str = "hermitian_spectrum",
    ) -> NativeField:
        payload = {"half_spectrum": np.asarray(half_spectrum), "shape": tuple(int(v) for v in full_shape)}
        return cls(
            id=id,
            domain=Domain.FOURIER_2D,
            semantic=Semantic.GENERIC_FIELD,
            coherence=Coherence.SIGNED_REAL,
            value_kind=FieldKind.LOG_SIGNED,
            representation=FieldRepresentation(RepresentationKind.HERMITIAN_HALF, payload),
            chart=CoordinateChart.identity(2),
            metadata={"native_continuity": _continuity_metadata(native_inputs=1)},
        )

    @classmethod
    def optical_depth(cls, values: Any, *, id: str = "optical_depth") -> NativeField:
        return cls._dense_native(
            values,
            id=id,
            kind=FieldKind.OPTICAL_DEPTH,
            semantic=Semantic.OPTICAL_DEPTH,
            domain=Domain.SPATIAL_2D,
        )

    @classmethod
    def _dense_native(
        cls,
        values: Any,
        *,
        id: str,
        kind: FieldKind,
        semantic: Semantic | str,
        domain: Domain | str,
        native_packs: int = 0,
    ) -> NativeField:
        coherence = {
            FieldKind.LOG_POS: Coherence.INCOHERENT_POSITIVE,
            FieldKind.PHASE: Coherence.PHASE_ONLY,
            FieldKind.SPF_COMPLEX: Coherence.COHERENT_COMPLEX,
            FieldKind.OPTICAL_DEPTH: Coherence.NOT_APPLICABLE,
        }[kind]
        array = np.asarray(values)
        return cls(
            id=id,
            domain=domain,
            semantic=semantic,
            coherence=coherence,
            value_kind=kind,
            representation=FieldRepresentation(RepresentationKind.DENSE_TYPED, array),
            chart=CoordinateChart.identity(max(0, array.ndim)),
            metadata={"native_continuity": _continuity_metadata(native_inputs=1, native_packs=native_packs)},
        )


def _atom_from_field(field: Field) -> v02.GaussianChirpAtom:
    if field.representation.kind is not RepresentationKind.GAUSSIAN_CHIRP_ATOM:
        raise TypeError("operation requires a Gaussian--chirp native field")
    atom = field.representation.payload
    if not isinstance(atom, v02.GaussianChirpAtom):
        raise TypeError("Gaussian--chirp representation has an invalid payload")
    return atom


def _replace_atom(
    field: NativeField,
    atom: v02.GaussianChirpAtom,
    *,
    op: str,
    domain: Domain | None = None,
) -> NativeField:
    continuity = _field_continuity(field)
    continuity["symbolic_operation_count"] += 1
    continuity["potential_dense_passes"] += 1
    metadata = dict(field.metadata)
    metadata["native_continuity"] = continuity
    operations = (*field.operations,)
    return replace(
        field,
        domain=field.domain if domain is None else domain,
        representation=replace(field.representation, payload=atom),
        operations=operations,
        metadata={**metadata, "last_native_operation": op},
    )


def _atom_log_jet(atom: v02.GaussianChirpAtom, grid: NativeGrid2D, guard_bits: int) -> v02.Jet2D:
    transformed = v02.atom_affine(atom, np.diag(grid.spacing), np.asarray(grid.origin))
    h = np.asarray(transformed.h, dtype=np.complex128)
    C = np.asarray(transformed.C, dtype=np.complex128)
    scale = v02.SPF32_L_SCALE * (1 << guard_bits) / math.log(2.0)
    return v02.Jet2D(
        value=round(float(np.real(transformed.kappa)) * scale),
        grad_x=round(float(np.real(h[0])) * scale),
        grad_y=round(float(np.real(h[1])) * scale),
        hess_xx=round(float(-np.real(C[0, 0])) * scale),
        hess_xy=round(float(-np.real(C[0, 1])) * scale),
        hess_yy=round(float(-np.real(C[1, 1])) * scale),
        guard_L=guard_bits,
        value_kind="log_pos",
    )


def _atom_phase_jet(atom: v02.GaussianChirpAtom, grid: NativeGrid2D, guard_bits: int) -> v02.Jet2D:
    transformed = v02.atom_affine(atom, np.diag(grid.spacing), np.asarray(grid.origin))
    h = np.asarray(transformed.h, dtype=np.complex128)
    C = np.asarray(transformed.C, dtype=np.complex128)
    scale = v02.SPF32_PHASE_MOD * (1 << guard_bits) / (2.0 * math.pi)
    return v02.Jet2D(
        value=round(float(np.imag(transformed.kappa)) * scale),
        grad_x=round(float(np.imag(h[0])) * scale),
        grad_y=round(float(np.imag(h[1])) * scale),
        hess_xx=round(float(-np.imag(C[0, 0])) * scale),
        hess_xy=round(float(-np.imag(C[0, 1])) * scale),
        hess_yy=round(float(-np.imag(C[1, 1])) * scale),
        guard_T=guard_bits,
        value_kind="phase",
    )


def _atom_jet_coefficients(
    atom: v02.GaussianChirpAtom,
    grid: NativeGrid2D,
    guard_bits: int,
    *,
    phase: bool,
) -> tuple[np.ndarray, float]:
    """Return exact guarded Jet2 coefficients before integer quantization."""

    transformed = v02.atom_affine(atom, np.diag(grid.spacing), np.asarray(grid.origin))
    h = np.asarray(transformed.h, dtype=np.complex128)
    C = np.asarray(transformed.C, dtype=np.complex128)
    if phase:
        scale = v02.SPF32_PHASE_MOD * (1 << guard_bits) / (2.0 * math.pi)
        coefficients = np.asarray(
            (
                np.imag(transformed.kappa),
                np.imag(h[0]),
                np.imag(h[1]),
                -np.imag(C[0, 0]),
                -np.imag(C[0, 1]),
                -np.imag(C[1, 1]),
            ),
            dtype=np.float64,
        )
    else:
        scale = v02.SPF32_L_SCALE * (1 << guard_bits) / math.log(2.0)
        coefficients = np.asarray(
            (
                np.real(transformed.kappa),
                np.real(h[0]),
                np.real(h[1]),
                -np.real(C[0, 0]),
                -np.real(C[0, 1]),
                -np.real(C[1, 1]),
            ),
            dtype=np.float64,
        )
    coefficients *= scale
    return coefficients, scale


def _jet_from_coefficients(
    coefficients: np.ndarray,
    guard_bits: int,
    *,
    phase: bool,
    already_rounded: bool = False,
) -> v02.Jet2D:
    rounded = (
        np.asarray(coefficients, dtype=np.int64)
        if already_rounded
        else np.rint(np.asarray(coefficients, dtype=np.float64)).astype(np.int64)
    )
    return v02.Jet2D(
        value=int(rounded[0]),
        grad_x=int(rounded[1]),
        grad_y=int(rounded[2]),
        hess_xx=int(rounded[3]),
        hess_xy=int(rounded[4]),
        hess_yy=int(rounded[5]),
        guard_T=guard_bits if phase else 0,
        guard_L=0 if phase else guard_bits,
        value_kind="phase" if phase else "log_pos",
    )


def _coefficient_error_bound(
    exact: np.ndarray,
    rounded: np.ndarray,
    shape: tuple[int, int],
    guard_bits: int,
    *,
    allocation_records: list[tuple[str, int, int]] | None = None,
    allocation_prefix: str = "jet",
) -> float:
    """Absolute stored-code error bound for one locally quantized tile."""

    height, width = shape
    x = float(max(0, width - 1))
    y = float(max(0, height - 1))
    rounded_float = np.asarray(rounded, dtype=np.float64)
    delta = np.asarray(exact, dtype=np.float64) - rounded_float
    error = np.abs(delta)
    if allocation_records is not None:
        allocation_records.extend(
            (
                (f"{allocation_prefix}_coefficient_delta", int(delta.size), int(delta.nbytes)),
                (f"{allocation_prefix}_coefficient_error", int(error.size), int(error.nbytes)),
                (
                    f"{allocation_prefix}_rounded_coefficients_float",
                    int(rounded_float.size),
                    int(rounded_float.nbytes),
                ),
            )
        )
    guarded = (
        error[0] + error[1] * x + error[2] * y + 0.5 * error[3] * x * x + error[4] * x * y + 0.5 * error[5] * y * y
    ) / float(1 << guard_bits)
    # Final guarded-to-storage RNE contributes at most half a stored code.
    return float(guarded + 0.5)


def _atom_projection_tiles(
    atom: v02.GaussianChirpAtom,
    grid: NativeGrid2D,
    log_guard: int,
    phase_guard: int,
    extent: tuple[int, int, int, int],
    include_phase: bool,
) -> Any:
    # Atoms contain arrays, so key by the small parameter state rather than
    # ndarray identity. A caller changing an array cannot reuse stale headers.
    return _cached_atom_projection_tiles(
        atom.kappa,
        tuple(np.asarray(atom.h).tolist()),
        tuple(np.asarray(atom.C).ravel().tolist()),
        atom.allow_nondecaying,
        grid,
        log_guard,
        phase_guard,
        extent,
        include_phase,
    )


@lru_cache(maxsize=32)
def _cached_atom_projection_tiles(
    kappa: complex,
    h: tuple[complex, ...],
    c: tuple[complex, ...],
    allow_nondecaying: bool,
    grid: NativeGrid2D,
    log_guard: int,
    phase_guard: int,
    extent: tuple[int, int, int, int],
    include_phase: bool,
) -> Any:
    """Prepare bounded local headers without allocating any sample plane."""
    from spf.renderer.jets import AdaptiveJetField, JetFitDiagnostics, SpectralJetTile2D

    atom = v02.GaussianChirpAtom(kappa, np.asarray(h), np.asarray(c).reshape(2, 2), allow_nondecaying)
    left, right, top, bottom = extent
    pending = [(left, top, right - left, bottom - top)] if right > left and bottom > top else []
    tiles, slices = [], []
    while pending:
        x0, y0, width, height = pending.pop()
        local = NativeGrid2D(
            (height, width),
            origin=(grid.origin[0] + x0 * grid.spacing[0], grid.origin[1] + y0 * grid.spacing[1]),
            spacing=grid.spacing,
        )
        exact, _ = _atom_jet_coefficients(atom, local, log_guard, phase=False)
        rounded = np.rint(exact).astype(np.int64)
        log_bound = _coefficient_error_bound(exact, rounded, local.shape, log_guard)
        phase_bound = 0.5
        phase_jet = v02.Jet2D(0, 0, 0, 0, 0, 0, guard_T=phase_guard, value_kind="phase")
        if include_phase:
            exact_phase, _ = _atom_jet_coefficients(atom, local, phase_guard, phase=True)
            rounded_phase = np.rint(exact_phase).astype(np.int64)
            phase_bound = _coefficient_error_bound(exact_phase, rounded_phase, local.shape, phase_guard)
            phase_jet = _jet_from_coefficients(rounded_phase, phase_guard, phase=True, already_rounded=True)
        if max(log_bound, phase_bound) > 1.0 and (width > 1 or height > 1):
            if width >= height and width > 1:
                middle = width // 2
                pending.extend(((x0 + middle, y0, width - middle, height), (x0, y0, middle, height)))
            else:
                middle = height // 2
                pending.extend(((x0, y0 + middle, width, height - middle), (x0, y0, width, middle)))
            continue
        tiles.append(
            SpectralJetTile2D(
                _jet_from_coefficients(rounded, log_guard, phase=False, already_rounded=True),
                phase_jet,
                local.shape,
                diagnostics=JetFitDiagnostics(
                    max_abs_log_error=(log_bound - 0.5) / v02.SPF32_L_SCALE,
                    max_phase_error=(phase_bound - 0.5) * 2 * math.pi / v02.SPF32_PHASE_MOD,
                ),
            )
        )
        slices.append((slice(y0, y0 + height), slice(x0, x0 + width)))
    return AdaptiveJetField(grid.shape, tuple(tiles), tuple(slices), 0.0)


def _evaluate_atom_jet_tiled(
    atom: v02.GaussianChirpAtom,
    grid: NativeGrid2D,
    guard_bits: int,
    *,
    phase: bool,
    maximum_code_error: float = 1.0,
    allocation_records: list[tuple[str, int, int]] | None = None,
    allocation_prefix: str = "jet",
) -> tuple[np.ndarray, float, int]:
    """Evaluate a quadratic with v0.2 adaptive Jet2 subdivision.

    A single header over a large tile can round a small Hessian to zero and
    flatten a broad Gaussian.  Re-centering child tiles preserves the local
    value and slope while the coefficient-error polynomial proves each tile's
    final stored-code error.  Every pixel is still generated by the normative
    forward-difference recurrence; no per-pixel polynomial is introduced.
    """

    def record(name: str, array: np.ndarray) -> None:
        if allocation_records is not None:
            value = np.asarray(array)
            allocation_records.append((f"{allocation_prefix}_{name}", int(value.size), int(value.nbytes)))

    output = np.empty(grid.shape, dtype=np.int64)
    record("output_plane", output)
    pending = [(0, 0, grid.shape[1], grid.shape[0])]
    maximum_bound = 0.0
    tile_count = 0
    while pending:
        x0, y0, width, height = pending.pop()
        local_grid = NativeGrid2D(
            (height, width),
            origin=(
                grid.origin[0] + x0 * grid.spacing[0],
                grid.origin[1] + y0 * grid.spacing[1],
            ),
            spacing=grid.spacing,
        )
        exact, _ = _atom_jet_coefficients(atom, local_grid, guard_bits, phase=phase)
        record("exact_coefficients", exact)
        rounded_float = np.rint(exact)
        record("rounded_coefficients_float", rounded_float)
        rounded = rounded_float.astype(np.int64)
        record("rounded_coefficients", rounded)
        bound = _coefficient_error_bound(
            exact,
            rounded,
            local_grid.shape,
            guard_bits,
            allocation_records=allocation_records,
            allocation_prefix=allocation_prefix,
        )
        if bound > maximum_code_error and (width > 1 or height > 1):
            if width >= height and width > 1:
                left = width // 2
                pending.append((x0 + left, y0, width - left, height))
                pending.append((x0, y0, left, height))
            else:
                top = height // 2
                pending.append((x0, y0 + top, width, height - top))
                pending.append((x0, y0, width, top))
            continue
        jet = _jet_from_coefficients(rounded, guard_bits, phase=phase, already_rounded=True)
        wide = jet_forward_differences_vectorized(
            jet,
            local_grid.shape,
            allocation_records=allocation_records,
            allocation_prefix=f"{allocation_prefix}_recurrence",
        )
        wide_float = wide.astype(np.float64)
        record("wide_float", wide_float)
        scaled = wide_float / float(1 << guard_bits)
        record("scaled_storage_codes", scaled)
        rounded_storage = np.rint(scaled)
        record("rounded_storage_codes", rounded_storage)
        storage = rounded_storage.astype(np.int64)
        record("stored_codes", storage)
        output[y0 : y0 + height, x0 : x0 + width] = storage
        maximum_bound = max(maximum_bound, bound)
        tile_count += 1
    return output, maximum_bound, tile_count


def jet_forward_differences_vectorized(
    jet: v02.Jet2D,
    shape: tuple[int, int],
    *,
    allocation_records: list[tuple[str, int, int]] | None = None,
    allocation_prefix: str = "jet_recurrence",
) -> np.ndarray:
    """Evaluate a 2-D quadratic using only recurrence additions after setup.

    NumPy's compiled ``add.accumulate`` supplies the row and column scans.  The
    routine is algebraically identical to the scalar recurrence in v0.2 section
    13.3 and retains integer headers as ``int64`` throughout.
    """

    height, width = (int(value) for value in shape)
    if height <= 0 or width <= 0:
        raise ValueError("jet shape must contain two positive dimensions")

    def record(name: str, array: np.ndarray) -> None:
        if allocation_records is not None:
            value = np.asarray(array)
            allocation_records.append((f"{allocation_prefix}_{name}", int(value.size), int(value.nbytes)))

    coefficients = np.asarray([jet.value, jet.grad_x, jet.grad_y, jet.hess_xx, jet.hess_xy, jet.hess_yy])
    record("coefficients", coefficients)
    integer_coefficients = np.issubdtype(coefficients.dtype, np.integer)
    half_step = integer_coefficients and (int(jet.hess_xx) % 2 != 0 or int(jet.hess_yy) % 2 != 0)
    # A quadratic's first forward difference includes H/2.  Hardware retains
    # this as one additional guard bit; float64 is the exact software carrier
    # for the bounded integer headers used here when that half-bit is present.
    dtype = np.float64 if half_step or not integer_coefficients else np.int64
    c, gx, gy, hxx, hxy, hyy = (np.asarray(value, dtype=dtype).item() for value in coefficients)

    row_values = np.empty(height, dtype=dtype)
    record("row_values", row_values)
    row_dx = np.empty(height, dtype=dtype)
    record("row_dx", row_dx)
    row_values[0] = c
    row_dx[0] = gx + 0.5 * hxx
    row_dy0 = gy + 0.5 * hyy
    if height > 1:
        dy_steps = np.empty(height - 1, dtype=dtype)
        record("dy_steps", dy_steps)
        dy_steps[0] = row_dy0
        if height > 2:
            dy_increments = np.full(height - 2, hyy, dtype=dtype)
            record("dy_increments", dy_increments)
            dy_prefix = np.add.accumulate(dy_increments)
            record("dy_prefix", dy_prefix)
            dy_tail = row_dy0 + dy_prefix
            record("dy_tail", dy_tail)
            dy_steps[1:] = dy_tail
        row_value_prefix = np.add.accumulate(dy_steps)
        record("row_value_prefix", row_value_prefix)
        row_value_tail = c + row_value_prefix
        record("row_value_tail", row_value_tail)
        row_values[1:] = row_value_tail
        row_dx_increments = np.full(height - 1, hxy, dtype=dtype)
        record("row_dx_increments", row_dx_increments)
        row_dx_prefix = np.add.accumulate(row_dx_increments)
        record("row_dx_prefix", row_dx_prefix)
        row_dx_tail = row_dx[0] + row_dx_prefix
        record("row_dx_tail", row_dx_tail)
        row_dx[1:] = row_dx_tail

    output = np.empty((height, width), dtype=dtype)
    record("output", output)
    output[:, 0] = row_values
    if width > 1:
        dx = np.empty((height, width - 1), dtype=dtype)
        record("dx", dx)
        dx[:, 0] = row_dx
        if width > 2:
            increments = np.full((height, width - 2), hxx, dtype=dtype)
            record("x_increments", increments)
            dx_prefix = np.add.accumulate(increments, axis=1)
            record("dx_prefix", dx_prefix)
            dx_tail = row_dx[:, None] + dx_prefix
            record("dx_tail", dx_tail)
            dx[:, 1:] = dx_tail
        output_prefix = np.add.accumulate(dx, axis=1)
        record("output_prefix", output_prefix)
        output_tail = row_values[:, None] + output_prefix
        record("output_tail", output_tail)
        output[:, 1:] = output_tail
    return output


def _certified_support_box(atom: v02.GaussianChirpAtom, magnitude_epsilon: float) -> Region | None:
    if magnitude_epsilon <= 0.0:
        return None
    real_c = np.real(atom.C)
    eigenvalues = np.linalg.eigvalsh(real_c)
    if float(np.min(eigenvalues)) <= 0.0:
        return None
    inverse = np.linalg.inv(real_c)
    real_h = np.real(atom.h)
    center = inverse @ real_h
    peak = float(np.real(atom.kappa) + 0.5 * real_h @ inverse @ real_h)
    threshold = math.log(magnitude_epsilon)
    if peak <= threshold:
        return Region(tuple(center), tuple(center))
    radius_squared = 2.0 * (peak - threshold)
    radii = np.sqrt(radius_squared * np.diag(inverse))
    return Region(tuple(center - radii), tuple(center + radii))


def _grid_support_slices(grid: NativeGrid2D, support: Region | None, *, empty: bool) -> tuple[slice, slice]:
    if empty:
        return slice(0, 0), slice(0, 0)
    if support is None:
        return slice(0, grid.shape[0]), slice(0, grid.shape[1])
    x0, y0 = grid.origin
    dx, dy = grid.spacing
    x_lo = max(0, math.ceil((support.lower[0] - x0) / dx))
    x_hi = min(grid.shape[1], math.floor((support.upper[0] - x0) / dx) + 1)
    y_lo = max(0, math.ceil((support.lower[1] - y0) / dy))
    y_hi = min(grid.shape[0], math.floor((support.upper[1] - y0) / dy) + 1)
    return slice(max(0, y_lo), max(0, y_hi)), slice(max(0, x_lo), max(0, x_hi))


_FINITE_LOG_CODES = np.arange(v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE + 1, dtype=np.int32)
_MAGNITUDE_LUT = np.exp2(_FINITE_LOG_CODES.astype(np.float64) / v02.SPF32_L_SCALE)
_INTENSITY_LUT = np.exp2(2.0 * _FINITE_LOG_CODES.astype(np.float64) / v02.SPF32_L_SCALE)


def _project_log_codes(
    log_codes: np.ndarray,
    semantic: OutputSemantic,
    zero: np.ndarray,
    *,
    coherent_display: bool = True,
) -> np.ndarray:
    values, _ = _project_log_codes_accounted(
        log_codes,
        semantic,
        zero,
        coherent_display=coherent_display,
    )
    return values


def _project_log_codes_accounted(
    log_codes: np.ndarray,
    semantic: OutputSemantic,
    zero: np.ndarray,
    *,
    coherent_display: bool = True,
) -> tuple[np.ndarray, tuple[tuple[str, int, int], ...]]:
    """Project a log plane and retain every logical temporary for accounting.

    The returned allocation records do not keep the temporary arrays alive, but
    their sizes describe materialized intermediate values that a backend must
    allocate.  This is intentionally separate from the terminal output array:
    the v0.2 M_A metric is about dense intermediate materialization.
    """

    clipped = np.clip(log_codes, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE)
    offset_codes = clipped - v02.SPF32_L_MIN_FINITE
    index = np.asarray(offset_codes, dtype=np.int64)
    workspace: list[tuple[str, int, int]] = [
        ("lut_clipped_codes", int(clipped.size), int(clipped.nbytes)),
        ("lut_offset_codes", int(offset_codes.size), int(offset_codes.nbytes)),
        ("lut_indices", int(index.size), int(index.nbytes)),
    ]
    if semantic in {OutputSemantic.MAGNITUDE, OutputSemantic.RADIANCE}:
        values = _MAGNITUDE_LUT[index]
        workspace.append(("magnitude_lut_values", int(values.size), int(values.nbytes)))
    elif semantic is OutputSemantic.LOG_RADIANCE:
        values = log_codes.astype(np.float64)
        workspace.append(("log_values_float64", int(values.size), int(values.nbytes)))
        values = values / v02.SPF32_L_SCALE
        workspace.append(("log_values_scaled", int(values.size), int(values.nbytes)))
    elif semantic is OutputSemantic.INTENSITY or (semantic is OutputSemantic.DISPLAY_RGB and coherent_display):
        values = _INTENSITY_LUT[index]
        workspace.append(("intensity_lut_values", int(values.size), int(values.nbytes)))
    elif semantic is OutputSemantic.DISPLAY_RGB:
        values = _MAGNITUDE_LUT[index]
        workspace.append(("display_magnitude_lut_values", int(values.size), int(values.nbytes)))
    else:
        raise AssertionError("real-field projection also requires phase")
    values = np.where(zero, -math.inf if semantic is OutputSemantic.LOG_RADIANCE else 0.0, values)
    workspace.append(("zero_masked_projection", int(values.size), int(values.nbytes)))
    if semantic is OutputSemantic.DISPLAY_RGB:
        mapped = values / (1.0 + values)
        workspace.append(("display_tone_map", int(mapped.size), int(mapped.nbytes)))
        srgb = np.where(mapped <= 0.0031308, 12.92 * mapped, 1.055 * np.power(mapped, 1.0 / 2.4) - 0.055)
        workspace.append(("display_srgb", int(srgb.size), int(srgb.nbytes)))
        clipped_srgb = np.clip(srgb, 0.0, 1.0)
        workspace.append(("display_clipped_srgb", int(clipped_srgb.size), int(clipped_srgb.nbytes)))
        display_scaled = clipped_srgb * 255.0
        workspace.append(("display_scaled_codes", int(display_scaled.size), int(display_scaled.nbytes)))
        rounded = np.rint(display_scaled)
        workspace.append(("display_rounded_codes", int(rounded.size), int(rounded.nbytes)))
        return np.asarray(rounded, dtype=np.uint8), tuple(workspace)
    return np.asarray(values, dtype=np.float32), tuple(workspace)


def _jet_rounding_code_bound(shape: tuple[int, int], guard_bits: int) -> float:
    """Bound terminal log/phase-code error from rounded quadratic headers.

    Each of the six Jet2 coefficients is rounded independently by at most half
    a guarded unit.  Evaluate the absolute coefficient-error polynomial at the
    far corner of the non-negative local tile, then include the final half-code
    RNE when the guarded recurrence is emitted.
    """

    height, width = shape
    if height <= 0 or width <= 0:
        return 0.0
    x = float(width - 1)
    y = float(height - 1)
    coefficient_weight = 1.0 + x + y + 0.5 * x * x + x * y + 0.5 * y * y
    return 0.5 + 0.5 * coefficient_weight / float(1 << guard_bits)


def _projection_error_ledger(
    values: np.ndarray,
    semantic: OutputSemantic,
    *,
    maximum_magnitude: float,
    log_code_error: float,
    phase_code_error: float,
    support_epsilon: float,
    pruned_elements: int,
    saturated_elements: int,
    maximum_output: float | None = None,
) -> v02.ErrorLedger:
    """Return a conservative absolute pointwise terminal certificate."""

    pruning = support_epsilon if pruned_elements > 0 else 0.0
    if pruned_elements > 0 and semantic is OutputSemantic.LOG_RADIANCE:
        pruning = math.inf
    if saturated_elements > 0:
        # Clipping discards an unbounded distance in the source exponent.  A
        # finite certificate would be false without an independent source
        # range bound, so range-stress rows deliberately report infinity.
        return v02.ErrorLedger(pruning=pruning, materialization=math.inf)
    max_output = (
        maximum_output
        if maximum_output is not None
        else max(abs(float(np.min(values, initial=0))), abs(float(np.max(values, initial=0))))
    )
    if semantic is OutputSemantic.LOG_RADIANCE:
        materialization = log_code_error / v02.SPF32_L_SCALE
    elif semantic is OutputSemantic.DISPLAY_RGB:
        # The byte projection includes tone-map and sRGB nonlinearities.  Keep
        # the portable certificate conservative until those stages expose a
        # tighter derivative/LUT certificate.
        materialization = 255.0 if values.size else 0.0
    else:
        exponent_multiplier = 2.0 if semantic is OutputSemantic.INTENSITY else 1.0
        magnitude_ratio = math.exp2(exponent_multiplier * log_code_error / v02.SPF32_L_SCALE)
        error_scale = max_output if semantic is not OutputSemantic.REAL_FIELD else maximum_magnitude
        materialization = error_scale * (magnitude_ratio - 1.0)
        if semantic is OutputSemantic.REAL_FIELD:
            phase_radians = phase_code_error * (2.0 * math.pi / v02.SPF32_PHASE_MOD)
            materialization += maximum_magnitude * magnitude_ratio * phase_radians
        materialization = float(materialization + float(np.finfo(np.float32).eps) * max(error_scale, maximum_magnitude))
    return v02.ErrorLedger(pruning=pruning, materialization=materialization)



def _combine_support(left: Support, right: Support) -> Support:
    """Intersect supports without treating UNKNOWN as proven full support."""

    if left.kind is SupportKind.EMPTY or right.kind is SupportKind.EMPTY:
        return Support.empty()
    if left.kind is SupportKind.UNKNOWN or right.kind is SupportKind.UNKNOWN:
        return Support(SupportKind.UNKNOWN)
    return left.intersection(right)


def _inverse_support_or_raise(support: Support) -> Support:
    """Return the support valid for a whole-space analytic inverse.

    A Gaussian atom is the inverse of an untruncated Gaussian spectrum.  A
    finite Fourier support instead produces a windowed inverse and therefore
    cannot be represented by the atom payload alone.  Keep the exact empty
    case, and force every other non-full case through an explicit fallback
    rather than silently widening its support.
    """

    if support.kind is SupportKind.EMPTY:
        return support
    if support.kind is not SupportKind.FULL:
        raise NotImplementedError(
            "analytic Gaussian inverse requires FULL Fourier support; "
            "finite/unknown support needs a residual or fallback"
        )
    return Support.full()


def _prepare_native_gaussian(field: Field) -> tuple[v02.GaussianChirpAtom, Support]:
    """Bind the shared normalized field to the single-Gaussian native kernel."""
    from spf.renderer.normalization import MaterializationError, require_closed_field

    try:
        normalized = require_closed_field(field, consumer="native Gaussian projection")
    except MaterializationError as exc:
        raise NotImplementedError(str(exc)) from exc
    payload = normalized.representation.payload
    if isinstance(payload, v02.AtomMixture):
        if len(payload.atoms) != 1 or payload.residual is not None or payload.dropped_bound:
            raise NotImplementedError(
                "native Gaussian projection supports only an exact one-atom mixture; "
                "multi-atom/residual mixtures need an explicit mixture consumer or fallback"
            )
        payload = payload.atoms[0]
    if normalized.representation.residual is not None:
        raise NotImplementedError("native Gaussian projection requires an explicit residual consumer")
    if not isinstance(payload, v02.GaussianChirpAtom):
        raise TypeError("native Gaussian projection requires a Gaussian atom")
    return payload, normalized.support


def _prepare_native_atom(field: Field) -> tuple[v02.GaussianChirpAtom, Support]:
    _atom_from_field(field)
    return _prepare_native_gaussian(field)


def _prepare_native_mixture(field: NativeField) -> tuple[v02.AtomMixture, Support]:
    """Bind a normalized symbolic mixture to the native plane reducer."""

    from spf.renderer.normalization import MaterializationError, require_closed_field

    try:
        normalized = require_closed_field(field, consumer="native atom-mixture projection")
    except MaterializationError as exc:
        raise NotImplementedError(str(exc)) from exc
    payload = normalized.representation.payload
    if not isinstance(payload, v02.AtomMixture):
        raise TypeError("ATOM_MIXTURE representation has an invalid payload")
    if normalized.representation.residual is not None or payload.residual is not None:
        raise NotImplementedError("native atom-mixture projection requires an explicit residual consumer")
    if any(atom.dimension != 2 for atom in payload.atoms):
        raise ValueError("native atom-mixture projection currently requires two-dimensional atoms")
    return payload, normalized.support


def _jet_recurrence(jet: v02.Jet2D, shape: tuple[int, int], *, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    if origin != (0.0, 0.0):
        jet = v02.jet_transform(jet, np.eye(2), np.asarray(origin))
    values = jet_forward_differences_vectorized(jet, shape)
    if str(jet.value_kind).lower() == "phase":
        values %= v02.SPF32_PHASE_MOD << jet.guard_T
    return values


def _jet_plane_codes(
    payload: Any,
    shape: tuple[int, int],
    grid_origin: tuple[float, float],
    *,
    include_phase: bool,
    allocation_records: list[tuple[str, int, int]] | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float, float, int, int]:
    """Materialize a selected paired Jet payload into native SPF planes.

    The returned arrays are *code* planes, not decoded Cartesian samples.  A
    selected jet is therefore consumed by the same log/LUT terminal path as an
    atom, without reconstructing an atom or calling ``program.render``.
    ``SpectralJetTile2D`` and ``AdaptiveJetField`` expose their guarded planes;
    a plain log ``Jet2D`` is accepted for magnitude/intensity consumers but
    cannot satisfy a real-field request because it carries no phase plane.
    """

    from spf.renderer.jets import AdaptiveJetField, SpectralJetTile2D

    height, width = (int(value) for value in shape)
    if height <= 0 or width <= 0:
        raise ValueError("quadratic jet projection requires a non-empty 2-D grid")

    def record(name: str, array: np.ndarray) -> None:
        if allocation_records is not None:
            value = np.asarray(array)
            allocation_records.append((name, int(value.size), int(value.nbytes)))

    def quantize_plane(values: np.ndarray, guard_bits: int, prefix: str) -> np.ndarray:
        scaled = np.asarray(values, dtype=np.float64) / float(1 << guard_bits)
        record(f"{prefix}_scaled_codes", scaled)
        rounded = np.rint(scaled)
        record(f"{prefix}_rounded_codes", rounded)
        codes = rounded.astype(np.int64)
        record(f"{prefix}_stored_codes", codes)
        return codes

    def convert_tile(tile: SpectralJetTile2D) -> tuple[np.ndarray, np.ndarray | None, float, float]:
        if tile.shape != shape:
            raise ValueError(
                f"selected quadratic-jet tile shape does not match the native grid ({tile.shape!r} != {shape!r})"
            )
        log_values = np.asarray(
            _jet_recurrence(tile.log_magnitude, tile.shape, origin=tile.origin),
            dtype=np.float64,
        )
        record("selected_log_recurrence", log_values)
        if tile.log_residual is not None:
            log_residual = tile.log_residual.to_dense()
            record("selected_log_residual", log_residual)
            log_values += log_residual
        log_guard = int(tile.log_magnitude.guard_L)
        log_codes = quantize_plane(log_values, log_guard, "selected_log")
        phase_codes = None
        if include_phase:
            phase_values = np.asarray(
                _jet_recurrence(tile.phase, tile.shape, origin=tile.origin),
                dtype=np.float64,
            )
            record("selected_phase_recurrence", phase_values)
            if tile.phase_residual is not None:
                phase_residual = tile.phase_residual.to_dense()
                record("selected_phase_residual", phase_residual)
                phase_values += phase_residual
            phase_guard = int(tile.phase.guard_T)
            phase_codes = quantize_plane(phase_values, phase_guard, "selected_phase")
        diagnostics = tile.diagnostics
        # ``SpectralJetTile2D`` reports log error in log2 units and phase error
        # in radians.  Convert to storage-code units and include the final
        # SPF32 RNE half-code, which is not part of ``planes()``.
        log_error = float(diagnostics.max_abs_log_error) * v02.SPF32_L_SCALE + 0.5
        phase_error = 0.0
        if include_phase:
            phase_error = float(diagnostics.max_phase_error) * v02.SPF32_PHASE_MOD / (2.0 * math.pi) + 0.5
        return log_codes, phase_codes, log_error, phase_error

    if isinstance(payload, SpectralJetTile2D):
        log_codes, phase_codes, log_error, phase_error = convert_tile(payload)
        return log_codes, phase_codes, log_error, phase_error, 1, int(include_phase)

    if isinstance(payload, AdaptiveJetField):
        if payload.shape != shape:
            raise ValueError(
                f"selected adaptive quadratic-jet shape does not match the native grid ({payload.shape!r} != {shape!r})"
            )
        log_codes = np.empty(shape, dtype=np.int64)
        record("adaptive_log_codes", log_codes)
        phase_codes = np.empty(shape, dtype=np.int64) if include_phase else None
        if phase_codes is not None:
            record("adaptive_phase_codes", phase_codes)
        max_log_error = 0.5
        max_phase_error = 0.5
        for tile, target in zip(payload.tiles, payload.slices, strict=True):
            log_values = np.asarray(
                _jet_recurrence(tile.log_magnitude, tile.shape, origin=tile.origin),
                dtype=np.float64,
            )
            record("adaptive_log_recurrence", log_values)
            if tile.log_residual is not None:
                log_residual = tile.log_residual.to_dense()
                record("adaptive_log_residual", log_residual)
                log_values += log_residual
            log_guard = int(tile.log_magnitude.guard_L)
            log_codes[target] = quantize_plane(log_values, log_guard, "adaptive_log")
            if include_phase:
                phase_values = np.asarray(
                    _jet_recurrence(tile.phase, tile.shape, origin=tile.origin),
                    dtype=np.float64,
                )
                record("adaptive_phase_recurrence", phase_values)
                if tile.phase_residual is not None:
                    phase_residual = tile.phase_residual.to_dense()
                    record("adaptive_phase_residual", phase_residual)
                    phase_values += phase_residual
                phase_guard = int(tile.phase.guard_T)
                assert phase_codes is not None
                phase_codes[target] = quantize_plane(phase_values, phase_guard, "adaptive_phase")
            diagnostics = tile.diagnostics
            max_log_error = max(
                max_log_error,
                float(diagnostics.max_abs_log_error) * v02.SPF32_L_SCALE + 0.5,
            )
            if include_phase:
                max_phase_error = max(
                    max_phase_error,
                    float(diagnostics.max_phase_error) * v02.SPF32_PHASE_MOD / (2.0 * math.pi) + 0.5,
                )
        return (
            log_codes,
            phase_codes,
            max_log_error,
            max_phase_error if include_phase else 0.0,
            len(payload.tiles),
            len(payload.tiles) if include_phase else 0,
        )

    if isinstance(payload, v02.Jet2D):
        kind = str(payload.value_kind).lower()
        if kind not in {"log", "log_pos"}:
            raise NotImplementedError(
                f"native quadratic-jet projection requires a paired log/phase jet; received {kind!r}"
            )
        log_values = _jet_recurrence(payload, shape, origin=grid_origin)
        record("plain_log_recurrence", log_values)
        log_codes = quantize_plane(np.asarray(log_values), int(payload.guard_L), "plain_log")
        log_error = _jet_rounding_code_bound(shape, int(payload.guard_L))
        return log_codes, None, log_error, 0.0, 1, 0

    if isinstance(payload, tuple) and len(payload) == 2 and all(isinstance(item, v02.Jet2D) for item in payload):
        log_jet, phase_jet = payload
        if str(log_jet.value_kind).lower() not in {"log", "log_pos"} or str(phase_jet.value_kind).lower() != "phase":
            raise TypeError("paired quadratic jets must be (log_pos, phase)")
        log_values = _jet_recurrence(log_jet, shape, origin=grid_origin)
        record("paired_log_recurrence", log_values)
        log_codes = quantize_plane(np.asarray(log_values), int(log_jet.guard_L), "paired_log")
        phase_codes = None
        if include_phase:
            phase_values = _jet_recurrence(phase_jet, shape, origin=grid_origin)
            record("paired_phase_recurrence", phase_values)
            phase_codes = quantize_plane(np.asarray(phase_values), int(phase_jet.guard_T), "paired_phase")
        return (
            log_codes,
            phase_codes,
            _jet_rounding_code_bound(shape, int(log_jet.guard_L)),
            _jet_rounding_code_bound(shape, int(phase_jet.guard_T)) if include_phase else 0.0,
            1,
            int(include_phase),
        )

    raise TypeError(f"QUADRATIC_JET representation has unsupported payload {type(payload).__name__}")


def _jet_payload_has_phase(payload: Any) -> bool:
    """Return whether a quadratic-jet payload carries a phase plane."""

    from spf.renderer.jets import AdaptiveJetField, SpectralJetTile2D

    return isinstance(payload, (SpectralJetTile2D, AdaptiveJetField)) or (
        isinstance(payload, tuple)
        and len(payload) == 2
        and all(isinstance(item, v02.Jet2D) for item in payload)
        and str(payload[1].value_kind).lower() == "phase"
    )


def _intersection_region(left: Region | None, right: Region | None) -> tuple[Region | None, bool]:
    if left is None:
        return right, False
    if right is None:
        return left, False
    if not left.intersects(right):
        return None, True
    return Region(
        tuple(max(a, b) for a, b in zip(left.lower, right.lower, strict=True)),
        tuple(min(a, b) for a, b in zip(left.upper, right.upper, strict=True)),
    ), False


def _infinite_ledger_like(ledger: v02.ErrorLedger) -> v02.ErrorLedger:
    """Mark every nonzero inherited component as unbounded without 0*inf NaNs."""

    return v02.ErrorLedger(
        **{
            name: math.inf if float(getattr(ledger, name)) != 0.0 else 0.0
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


def _scale_ledger_for_observable(ledger: v02.ErrorLedger, factor: float) -> v02.ErrorLedger:
    """Scale inherited ledger components while preserving their categories."""

    if factor == 0.0:
        return v02.ErrorLedger(stages=ledger.stages)
    if not math.isfinite(factor):
        return _infinite_ledger_like(ledger)
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


def _input_error_for_projection(field: Field, values: np.ndarray, semantic: OutputSemantic) -> v02.ErrorLedger:
    """Propagate inherited uncertainty into the requested observable.

    Returning a ledger (rather than a scalar stuffed into ``representation_fit``)
    preserves source/reducer/materialization custody even when every requested
    sample is pruned.
    """

    from spf.renderer.normalization import complete_field_ledger

    source_ledger = complete_field_ledger(field)
    amplitude_error = float(source_ledger.total_bound)
    if amplitude_error == 0.0:
        return v02.ErrorLedger(stages=source_ledger.stages)
    if not math.isfinite(amplitude_error):
        return _infinite_ledger_like(source_ledger)
    maximum = max(abs(float(np.min(values, initial=0))), abs(float(np.max(values, initial=0))))
    if semantic is OutputSemantic.INTENSITY:
        magnitude = math.sqrt(maximum)
        return _scale_ledger_for_observable(source_ledger, 2.0 * magnitude + amplitude_error)
    if semantic is OutputSemantic.LOG_RADIANCE:
        # Without a positive lower amplitude bound, additive source error has
        # no finite logarithmic image.
        return _infinite_ledger_like(source_ledger)
    if semantic is OutputSemantic.DISPLAY_RGB:
        return _scale_ledger_for_observable(source_ledger, 255.0)
    return _scale_ledger_for_observable(source_ledger, 1.0)


def _enforce_native_contract(
    field: Field,
    ledger: v02.ErrorLedger,
    values: np.ndarray,
    *,
    phase_error: float = 0.0,
) -> None:
    if (
        math.isinf(field.error.absolute_max)
        and math.isinf(field.error.relative_max)
        and math.isinf(field.error.phase_max)
        and not field.error.phase_required
    ):
        return
    scale = max(abs(float(np.min(values, initial=0))), abs(float(np.max(values, initial=0))))
    if not field.error.accepts(ledger, scale=scale, phase_error=phase_error):
        from spf.renderer.runtime import MaterializationError

        raise MaterializationError(
            "native projection exceeds field ErrorContract "
            f"(absolute={ledger.total_bound!r}, relative={ledger.total_bound / max(scale, np.finfo(float).tiny)!r}, "
            f"phase={phase_error!r})"
        )


def _phase_code_error_to_radians(code_error: float) -> float:
    """Convert the terminal SPF phase-code bound to radians."""

    if not math.isfinite(code_error):
        return math.inf
    return float(code_error) * (2.0 * math.pi / v02.SPF32_PHASE_MOD)


def _maximum_magnitude_from_log_codes(log_codes: np.ndarray) -> float:
    """Find a magnitude bound without allocating a second full-grid LUT array."""

    maximum_code = int(np.max(log_codes, initial=v02.SPF32_L_ZERO))
    if maximum_code <= v02.SPF32_L_ZERO:
        return 0.0
    maximum_code = min(maximum_code, v02.SPF32_L_MAX_FINITE)
    return float(math.exp2(maximum_code / v02.SPF32_L_SCALE))


@dataclass(frozen=True, slots=True)
class NativeProjectionResult:
    values: np.ndarray
    report: RenderReport
    log_codes: np.ndarray
    phase_codes: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class NativeIntegralResult:
    """Scalar result from an analytic structured-footprint consumer."""

    value: complex
    report: RenderReport


def _store_projection_output(result: NativeProjectionResult, output: np.ndarray | None) -> NativeProjectionResult:
    if output is None or result.values is output:
        return result
    if output.shape != result.values.shape or output.dtype != result.values.dtype:
        raise ValueError("terminal output shape/dtype mismatch")
    np.copyto(output, result.values)
    metrics = replace(result.report.metrics)
    metrics.materialized_intermediate_elements += result.values.size
    metrics.materialized_intermediate_bytes += result.values.nbytes
    metrics.materialized_elements += result.values.size
    metrics.logical_peak_bytes += output.nbytes
    metrics.peak_memory += output.nbytes
    metrics.bytes_written += output.nbytes
    diagnostics = {**result.report.diagnostics, "terminal_copy_bytes": output.nbytes}
    return replace(result, values=output, report=replace(result.report, metrics=metrics, diagnostics=diagnostics))


def _emit_project(*args: Any, _preparation: list[Any] | None = None, **kwargs: Any) -> Any:
    from spf.renderer.emission import emit_selected, prepare_emission

    if _preparation is None:
        return emit_selected(*args, **kwargs)
    plan = prepare_emission(*args, **kwargs)
    if plan is None:
        return emit_selected(*args, **kwargs)
    _preparation.append(plan)
    return plan.execute(kwargs.get("output"))


@dataclass(frozen=True)
class PreparedNativeProjection:
    """Static terminal certificate plus executable, owned native commands.

    A new source, header, support, precision or semantic needs a new plan.
    Runtime output storage can change without recompiling mathematical state.
    """

    emission: Any
    report: RenderReport

    def execute(self, *, output: np.ndarray | None = None, clear: bool = True) -> NativeProjectionResult:
        emitted = self.emission.execute(output, clear=clear)
        return NativeProjectionResult(emitted.values, self.report, emitted.logs, emitted.phases)


@dataclass(frozen=True)
class PreparedNativeRGB:
    """One output clear and one native submission for three prepared channels."""

    channels: tuple[PreparedNativeProjection, ...]
    commands: Any
    shape: tuple[int, ...]
    dtype: Any
    report: RenderReport

    @classmethod
    def prepare(cls, projections: Any) -> PreparedNativeRGB:
        import ctypes

        from spf.renderer._jet_kernel import JetCommand

        channels = tuple(getattr(item, "_prepared_execution", item) for item in projections)
        if len(channels) != 3 or any(not isinstance(item, PreparedNativeProjection) for item in channels):
            raise ValueError("RGB preparation requires three native prepared projections")
        templates = [item.emission.template for item in channels]
        shape, dtype = templates[0].values.shape, templates[0].values.dtype
        if any(
            t.values.shape != shape or t.values.dtype != dtype or t.logs.size or t.phases is not None for t in templates
        ):
            raise ValueError("RGB channels must share shape/dtype and disable diagnostic code planes")
        if any(item.emission.fill_value != channels[0].emission.fill_value for item in channels):
            raise ValueError("RGB channel background semantics must match")
        count = sum(len(item.emission.records) for item in channels)
        commands = (JetCommand * count)()
        offset = 0
        for channel, item in enumerate(channels):
            source = item.emission.native_commands
            for i in range(len(source)):
                command = commands[offset]
                ctypes.memmove(ctypes.byref(command), ctypes.byref(source[i]), ctypes.sizeof(JetCommand))
                command.out_offset = 3 * command.out_offset + channel * dtype.itemsize
                command.out_row_stride *= 3
                command.out_pixel_stride *= 3
                offset += 1
        reports = [item.report for item in channels]
        metrics = RenderMetrics()
        ledger = v02.ErrorLedger()
        for report in reports:
            metrics.add(report.metrics)
            ledger = ledger.plus(report.ledger)
        metrics.backend = reports[0].metrics.backend
        metrics.workload = "prepared_native_rgb"
        metrics.resolution = (*shape, 3)
        metrics.precision = reports[0].metrics.precision
        diagnostics = {
            "rgb_aggregated": True,
            "rgb_channel_count": 3,
            "native_submissions": 1,
            "output_clear_passes": 1,
            "prepared_commands": count,
            "channel_diagnostics": tuple(report.diagnostics for report in reports),
            "conversion_counts": {
                key: sum(int(r.diagnostics.get("conversion_counts", {}).get(key, 0)) for r in reports)
                for key in reports[0].diagnostics.get("conversion_counts", {})
            },
            "fft_forbidden": all(r.diagnostics.get("fft_forbidden", False) for r in reports),
            "rgb_direct_output": True,
            "peak_memory_status": "prepared_commands_and_single_shared_output; allocator_overhead_unmeasured",
            "pass_avoidance": metrics.materialization_avoidance,
            "potential_pass_elements": metrics.potential_intermediate_elements,
            "realized_pass_elements": metrics.materialized_intermediate_elements,
            "planner_selected": tuple(x for r in reports for x in r.diagnostics.get("planner_selected", ())),
            "executed_backend_ops": tuple(r.diagnostics.get("executed_backend_ops", ()) for r in reports),
            "compiler_original_ops": tuple(r.diagnostics.get("compiler_original_ops", ()) for r in reports),
            "compiler_optimized_ops": tuple(r.diagnostics.get("compiler_optimized_ops", ()) for r in reports),
        }
        for key in (
            "support_evaluated_elements",
            "evaluated_grid_elements",
            "retained_output_elements",
            "support_pruned_elements",
            "low_saturation_elements",
            "high_saturation_elements",
            "potential_intermediate_elements",
            "materialized_intermediate_elements",
            "terminal_output_elements",
            "potential_intermediate_bytes",
            "materialized_intermediate_bytes",
            "terminal_output_bytes",
        ):
            diagnostics[key] = sum(int(r.diagnostics.get(key, 0)) for r in reports)
        report = RenderReport(
            metrics,
            ledger,
            passes=tuple(x for r in reports for x in r.passes),
            decisions=tuple(x for r in reports for x in r.decisions),
            fallbacks=tuple(x for r in reports for x in r.fallbacks),
            capabilities_used=tuple(dict.fromkeys(x for r in reports for x in r.capabilities_used)),
            diagnostics=diagnostics,
        )
        return cls(channels, commands, (*shape, 3), dtype, report)

    def execute(self, *, output: np.ndarray | None = None) -> NativeProjectionResult:
        import ctypes

        from spf.renderer._jet_kernel import JetStats, pointer, steady_batch_kernel

        if output is None:
            output = np.empty(self.shape, dtype=self.dtype)
        if (
            output.shape != self.shape
            or output.dtype != self.dtype
            or not output.flags.c_contiguous
            or not output.flags.writeable
        ):
            raise ValueError("prepared RGB output must be a writable contiguous buffer of the prepared shape and dtype")
        output.fill(self.channels[0].emission.fill_value)
        steady = steady_batch_kernel()
        if steady is not None:
            steady(self.commands, len(self.commands), pointer(output))
        else:
            stats = JetStats(max_log=v02.SPF32_L_ZERO)
            self.channels[0].emission.function(
                self.commands, len(self.commands), pointer(output), None, None, ctypes.byref(stats)
            )
        return NativeProjectionResult(output, self.report, np.empty(0, dtype=np.int32))


class NativeRenderer:
    """Strict native-field renderer for v0.2 producer/consumer continuity."""

    def __init__(
        self,
        *,
        precision: PrecisionContract | None = None,
        strict_native: bool = True,
        retain_codes: bool = True,
        reference_planes: bool = False,
        packed_atom_arithmetic: bool = False,
    ) -> None:
        self.precision = precision or PrecisionContract()
        self.strict_native = bool(strict_native)
        self.retain_codes = bool(retain_codes)
        self.reference_planes = bool(reference_planes)
        self.packed_atom_arithmetic = bool(packed_atom_arithmetic)
        self._rgb_prepared: dict[Any, tuple[tuple[NativeField, ...], PreparedNativeRGB]] = {}

    submit_log_positive = staticmethod(NativeField.log_positive)
    submit_phase_field = staticmethod(NativeField.phase)
    submit_spf_complex = staticmethod(NativeField.spf_complex)
    submit_jet2d = staticmethod(NativeField.jet2d)
    submit_gaussian_chirp = staticmethod(NativeField.gaussian_chirp)
    submit_atom_mixture = staticmethod(NativeField.atom_mixture)
    submit_hermitian_spectrum = staticmethod(NativeField.hermitian_spectrum)
    submit_optical_depth = staticmethod(NativeField.optical_depth)

    def _closed_atom_product(
        self,
        left: v02.GaussianChirpAtom,
        right: v02.GaussianChirpAtom,
    ) -> tuple[v02.GaussianChirpAtom, int, int]:
        """Return a product and its packed-header accounting."""

        if not self.packed_atom_arithmetic:
            return v02.atom_mul(left, right), 0, 0
        packed = v02.PackedGaussianChirpAtom.from_atom(left).product(
            v02.PackedGaussianChirpAtom.from_atom(right)
        )
        return packed.to_atom(), 1, 1

    def compile_gaussian_product(
        self,
        source: NativeField,
        transfer: NativeField,
        grid: NativeGrid2D,
        *,
        semantic: OutputSemantic | str = OutputSemantic.INTENSITY,
        support_epsilon: float = 0.0,
    ) -> CompiledNativeGaussianProjection:
        """Compile one closed Gaussian product through compiler, planner, and backend."""

        from spf.renderer.compiler import CompilerOptions, FieldGraph, SFRNode, compile_graph
        from spf.renderer.model import ErrorBudget, SampleLayout

        if source.domain is not Domain.FOURIER_2D or transfer.domain is not Domain.FOURIER_2D:
            raise ValueError("compiled Gaussian product requires two FOURIER_2D inputs")
        source_atom, source_support = _prepare_native_atom(source)
        transfer_atom, transfer_support = _prepare_native_atom(transfer)
        product_support = _combine_support(source_support, transfer_support)
        # The graph's inverse node is a whole-space Gaussian closure.  Do not
        # let a finite/unknown frequency support disappear while lowering it.
        inverse_support = _inverse_support_or_raise(product_support)
        from spf.renderer.normalization import complete_field_ledger

        product_error = complete_field_ledger(source).plus(complete_field_ledger(transfer))
        metadata = {
            "value_kind": FieldKind.ATOM,
            "domain": Domain.FOURIER_2D,
            "semantic": Semantic.COMPLEX_OPTICAL_AMPLITUDE,
            "coherence": Coherence.COHERENT_COMPLEX,
            "chart": CoordinateChart.identity(2),
            "support": product_support,
            "ledger": product_error,
        }
        spatial_metadata = {**metadata, "domain": Domain.SPATIAL_2D}

        # Preserve each producer's typed operation chain in the compiler graph
        # so translation/exposure/filter operations are folded exactly once.
        # Prefixing avoids collisions when callers submit fields with the same
        # public id (the common default for anonymous atoms).
        def prefix_graph(graph: FieldGraph, prefix: str) -> tuple[tuple[SFRNode, ...], str]:
            mapping = {node.output: f"{prefix}{node.output}" for node in graph.nodes}
            nodes = tuple(
                node.with_updates(
                    output=mapping[node.output],
                    inputs=tuple(mapping.get(name, f"{prefix}{name}") for name in node.inputs),
                )
                for node in graph.nodes
            )
            output = mapping[graph.outputs[-1]]
            return nodes, output

        source_nodes, source_output = prefix_graph(FieldGraph.from_field(source), "source__")
        transfer_nodes, transfer_output = prefix_graph(FieldGraph.from_field(transfer), "transfer__")
        graph = FieldGraph.from_nodes(
            (
                *source_nodes,
                *transfer_nodes,
                SFRNode(
                    "v02_atom_product",
                    "frequency_product",
                    (source_output, transfer_output),
                    metadata=metadata,
                ),
                SFRNode(
                    "v02_atom_inverse_fourier",
                    "spatial_atom",
                    ("frequency_product",),
                    metadata=spatial_metadata,
                ),
            ),
            outputs=("spatial_atom",),
        )
        layout = SampleLayout(
            grid.shape,
            origin=grid.origin,
            spacing=grid.spacing,
        )
        region = Region(
            grid.origin,
            (
                grid.origin[0] + max(0, grid.shape[1] - 1) * grid.spacing[0],
                grid.origin[1] + max(0, grid.shape[0] - 1) * grid.spacing[1],
            ),
        )
        program = compile_graph(
            graph,
            options=CompilerOptions(
                backend="cpu",
                consumer_representations=(
                    RepresentationKind.GAUSSIAN_CHIRP_ATOM,
                    RepresentationKind.ATOM_MIXTURE,
                    RepresentationKind.QUADRATIC_JET,
                ),
                precision=self.precision,
                error_budget=ErrorBudget(),
                layout=layout,
                regions=(region,),
                allow_fallback=False,
            ),
        )
        # Compilation may select a representation other than the graph's raw
        # payload.  Carry that candidate into execution; calling
        # ``program.render`` here would execute the graph during compilation
        # and then silently discard the planner's selection.
        planner_candidates = () if program.materialization_plan is None else program.materialization_plan.selected
        selected_candidate = next(
            (candidate for candidate in planner_candidates if candidate.field_id == "spatial_atom"),
            None,
        )
        product_atom, _, _ = self._closed_atom_product(source_atom, transfer_atom)
        analytic_atom = v02.atom_inverse_fourier(product_atom)
        selected_representation = (
            selected_candidate.representation
            if selected_candidate is not None
            else FieldRepresentation(
                RepresentationKind.GAUSSIAN_CHIRP_ATOM,
                analytic_atom,
                exactness=Exactness.EXACT,
            )
        )
        compiled_atom = (
            selected_representation.payload
            if selected_representation.kind is RepresentationKind.GAUSSIAN_CHIRP_ATOM
            else analytic_atom
        )
        if not isinstance(compiled_atom, v02.GaussianChirpAtom):
            # ``spatial_atom`` is retained as a compatibility oracle for
            # callers inspecting the closed form, while execute() consumes
            # the selected representation and fails closed for unsupported
            # native payloads below.
            compiled_atom = analytic_atom
        return CompiledNativeGaussianProjection(
            program,
            compiled_atom,
            self,
            source,
            transfer,
            grid,
            _output_semantic(semantic),
            float(support_epsilon),
            inverse_support,
            selected_representation,
        )

    def multiply(self, left: NativeField, right: NativeField, *, id: str = "atom_product") -> NativeField:
        if left.domain != right.domain:
            raise ValueError("native atom product requires matching domains")
        left_atom, left_support = _prepare_native_atom(left)
        right_atom, right_support = _prepare_native_atom(right)
        atom, packed_operations, packed_decodes = self._closed_atom_product(left_atom, right_atom)
        metadata = dict(left.metadata)
        metadata["native_continuity"] = _merge_continuity(left, right)
        metadata["native_continuity"]["packed_atom_arithmetic_count"] += packed_operations
        metadata["native_continuity"]["packed_atom_decode_count"] += packed_decodes
        metadata["native_continuity"]["native_pack_count"] += 2 * packed_operations
        product_support = _combine_support(left_support, right_support)
        return NativeField(
            id=id,
            domain=left.domain,
            semantic=left.semantic,
            coherence=left.coherence,
            value_kind=left.value_kind,
            representation=FieldRepresentation(
                RepresentationKind.GAUSSIAN_CHIRP_ATOM,
                atom,
                exactness=Exactness.BOUNDED if packed_operations else Exactness.EXACT,
                byte_size=left.representation.byte_size,
            ),
            chart=left.chart,
            support=product_support,
            error=left.error,
            precision=left.precision,
            metadata={**metadata, "last_native_operation": "atom_product"},
        )

    def inverse_fourier(self, field: NativeField, *, id: str = "analytic_inverse_fourier") -> NativeField:
        domain = field.domain
        if domain is not Domain.FOURIER_2D:
            raise ValueError("analytic inverse Fourier transform requires a FOURIER_2D field")
        atom, support = _prepare_native_atom(field)
        inverse_support = _inverse_support_or_raise(support)
        transformed = _replace_atom(
            replace(field, id=id, operations=()),
            v02.atom_inverse_fourier(atom),
            op="atom_inverse_fourier",
            domain=Domain.SPATIAL_2D,
        )
        return replace(
            transformed,
            support=inverse_support,
        )

    def _finish_projection(
        self,
        field: NativeField,
        grid: NativeGrid2D,
        selected: OutputSemantic,
        *,
        declared_support: Support,
        support: Region | None,
        active_count: int,
        log_codes: np.ndarray,
        phase_codes: np.ndarray | None,
        log_code_error: float,
        phase_code_error: float,
        jet_tile_count: int,
        phase_jet_tile_count: int,
        low_saturation: int,
        high_saturation: int,
        intermediate_arrays: list[tuple[str, int, int]],
        source_representation: RepresentationKind,
        support_epsilon: float = 0.0,
        evaluated_count: int | None = None,
        atoms_considered: int = 1,
        pass_records: tuple[CompilerPassRecord, ...] | None = None,
        capabilities: tuple[str, ...] | None = None,
        workload: str = "gaussian_chirp_closed_form",
        projected_values: np.ndarray | None = None,
        maximum_magnitude_override: float | None = None,
        maximum_output_override: float | None = None,
        backend_name: str = "numpy_add_accumulate_reference",
        emission_diagnostics: dict[str, Any] | None = None,
        additional_ledger: v02.ErrorLedger | None = None,
        reduction_operations: int = 0,
        continuity_updates: dict[str, int] | None = None,
    ) -> NativeProjectionResult:
        """Finish a native projection from already-emitted SPF planes.

        Atom, one-atom-mixture, and paired-jet inputs all converge here.  The
        convergence is deliberately after plane emission: no selected payload
        is reconstructed as a Cartesian atom, and RGB callers can aggregate
        the same terminal accounting one channel at a time.
        """

        def track_intermediate(name: str, array: np.ndarray) -> None:
            value = np.asarray(array)
            intermediate_arrays.append((name, int(value.size), int(value.nbytes)))

        if projected_values is None:
            log_codes = np.asarray(log_codes, dtype=np.int32)
            zero = log_codes == v02.SPF32_L_ZERO
            track_intermediate("zero_mask", zero)
            if selected is OutputSemantic.REAL_FIELD:
                if phase_codes is None:
                    raise TypeError("REAL_FIELD projection requires a paired phase jet")
                phase_codes = np.asarray(phase_codes, dtype=np.uint16)
                track_intermediate("phase_codes", phase_codes)
                magnitude, lut_workspace = _project_log_codes_accounted(log_codes, OutputSemantic.MAGNITUDE, zero)
                intermediate_arrays.extend(lut_workspace)
                phase_float_codes = phase_codes.astype(np.float64)
                track_intermediate("phase_float_codes", phase_float_codes)
                phase_radians = phase_float_codes * (2.0 * math.pi / v02.SPF32_PHASE_MOD)
                track_intermediate("phase_radians", phase_radians)
                cosine = np.cos(phase_radians)
                track_intermediate("phase_cosine", cosine)
                real_values = magnitude * cosine
                track_intermediate("real_values_float64", real_values)
                values = np.asarray(real_values, dtype=np.float32)
            else:
                values, lut_workspace = _project_log_codes_accounted(
                    log_codes,
                    selected,
                    zero,
                    coherent_display=field.coherence is Coherence.COHERENT_COMPLEX,
                )
                intermediate_arrays.extend(lut_workspace)

        else:
            values = projected_values

        pruned_elements = grid.size - int(active_count)
        maximum_magnitude = (
            _maximum_magnitude_from_log_codes(log_codes)
            if maximum_magnitude_override is None
            else maximum_magnitude_override
        )
        ledger = _projection_error_ledger(
            values,
            selected,
            maximum_magnitude=maximum_magnitude,
            log_code_error=log_code_error,
            phase_code_error=phase_code_error,
            support_epsilon=float(support_epsilon),
            pruned_elements=pruned_elements,
            saturated_elements=int(low_saturation) + int(high_saturation),
        )
        if additional_ledger is not None:
            ledger = ledger.plus(additional_ledger)
        input_error = _input_error_for_projection(field, values, selected)
        ledger = ledger.plus(input_error)
        phase_error_radians = _phase_code_error_to_radians(phase_code_error)
        _enforce_native_contract(field, ledger, values, phase_error=phase_error_radians)
        continuity = _field_continuity(field)
        for key, value in (continuity_updates or {}).items():
            if key in continuity:
                continuity[key] += int(value)
        if field.operations:
            continuity["symbolic_operation_count"] += len(field.operations)
            continuity["potential_dense_passes"] += len(field.operations)
        continuity["native_quantization_count"] += 1
        conversion_counts = {
            "cartesian_ingress_conversions": continuity["legacy_import_count"],
            "cartesian_internal_conversions": continuity["cartesian_intermediate_count"],
            "cartesian_egress_conversions": continuity["legacy_export_count"],
            "legacy_adapter_calls": continuity["legacy_import_count"] + continuity["legacy_export_count"],
        }
        if self.strict_native and any(conversion_counts.values()):
            raise RuntimeError(f"native continuity violation: {conversion_counts}")

        intermediate_elements = int(sum(size for _, size, _ in intermediate_arrays))
        intermediate_bytes = int(sum(nbytes for _, _, nbytes in intermediate_arrays))
        terminal_elements = int(values.size)
        terminal_bytes = int(values.nbytes)
        potential = int(grid.potential_size)
        potential_passes = max(1, int(continuity["potential_dense_passes"]))
        potential_intermediate_elements = potential * potential_passes
        potential_intermediate_bytes = potential_intermediate_elements * int(np.dtype(np.complex64).itemsize)
        logical_peak_bytes = int(field.representation.byte_size or 0) + intermediate_bytes + terminal_bytes
        potential_pass_elements = potential_intermediate_elements
        metrics = RenderMetrics(
            backend=backend_name,
            workload=workload,
            resolution=grid.shape,
            representation=RepresentationKind.QUADRATIC_JET.value,
            precision=str(self.precision.mode).lower(),
            bytes_read=int(field.representation.byte_size or 0),
            bytes_written=terminal_bytes
            + (active_count * values.itemsize if projected_values is not None else 0)
            + (log_codes.nbytes + active_count * log_codes.itemsize if log_codes.size else 0)
            + (phase_codes.nbytes + active_count * phase_codes.itemsize if phase_codes is not None else 0),
            peak_memory=logical_peak_bytes,
            potential_elements=potential,
            materialized_elements=intermediate_elements + terminal_elements,
            potential_intermediate_elements=potential_intermediate_elements,
            materialized_intermediate_elements=intermediate_elements,
            terminal_output_elements=terminal_elements,
            potential_intermediate_bytes=potential_intermediate_bytes,
            materialized_intermediate_bytes=intermediate_bytes,
            terminal_output_bytes=terminal_bytes,
            logical_peak_bytes=logical_peak_bytes,
            atoms_considered=int(atoms_considered),
            atoms_pruned=int(active_count == 0),
            fft_count=0,
            fallback_conversions=0,
            field_operations=continuity["symbolic_operation_count"],
            reduction_operations=int(reduction_operations),
        )
        selected_name = source_representation.value
        evaluated_elements = int(active_count if evaluated_count is None else evaluated_count)
        if pass_records is None:
            pass_records = (
                CompilerPassRecord("native_selected_representation", 1, 1, notes=(selected_name,)),
                CompilerPassRecord("native_projection", 1, 1, notes=(selected.value,)),
            )
        if capabilities is None:
            capabilities = (
                "typed_native_input",
                "selected_representation_direct_execution",
                "jet2_forward_difference",
                f"project_{selected.value}",
            )
        report = RenderReport(
            metrics,
            ledger,
            passes=pass_records,
            decisions=(
                RepresentationDecision(
                    field.id,
                    repr(grid),
                    RepresentationKind.QUADRATIC_JET.value,
                    (selected_name, RepresentationKind.QUADRATIC_JET.value),
                    float(active_count),
                    float(ledger.total_bound),
                    "selected native representation emitted directly; dense FFT forbidden",
                ),
            ),
            capabilities_used=capabilities,
            diagnostics=MappingProxyType(
                {
                    "output_semantic": selected.value,
                    "source_representation_kind": selected_name,
                    "conversion_counts": conversion_counts,
                    "native_counts": continuity,
                    "materialization_count": 1,
                    "display_projection_count": int(selected is OutputSemantic.DISPLAY_RGB),
                    "magnitude_projection_count": int(selected is OutputSemantic.MAGNITUDE),
                    "intensity_projection_count": int(selected is OutputSemantic.INTENSITY),
                    "real_projection_count": int(selected is OutputSemantic.REAL_FIELD),
                    "radiance_projection_count": int(selected is OutputSemantic.RADIANCE),
                    "log_radiance_projection_count": int(selected is OutputSemantic.LOG_RADIANCE),
                    "support_epsilon": float(support_epsilon),
                    "support_region": None if support is None else {"lower": support.lower, "upper": support.upper},
                    "declared_support_kind": getattr(declared_support.kind, "value", str(declared_support.kind)),
                    "support_evaluated_elements": evaluated_elements,
                    "evaluated_grid_elements": evaluated_elements,
                    "retained_output_elements": int(active_count),
                    "support_pruned_elements": pruned_elements,
                    "requested_elements": grid.size,
                    "materialized_intermediate_elements": intermediate_elements,
                    "materialized_intermediate_bytes": intermediate_bytes,
                    "intermediate_allocations": tuple(intermediate_arrays),
                    "terminal_header_elements": int(active_count),
                    "terminal_output_elements": terminal_elements,
                    "terminal_output_bytes": terminal_bytes,
                    "potential_intermediate_elements": potential_intermediate_elements,
                    "potential_intermediate_bytes": potential_intermediate_bytes,
                    "potential_intermediate_dtype": "complex64",
                    "logical_peak_bytes": logical_peak_bytes,
                    "peak_memory_status": (
                        "conservative_upper_bound_from_explicit_numpy_allocations; "
                        "allocator_and_python_object_overhead_unmeasured"
                    ),
                    "potential_pass_elements": potential_pass_elements,
                    "realized_pass_elements": intermediate_elements,
                    "pass_avoidance": (
                        float(potential_pass_elements) / float(intermediate_elements)
                        if intermediate_elements
                        else math.inf
                    ),
                    "low_saturation_elements": int(low_saturation),
                    "high_saturation_elements": int(high_saturation),
                    "jet_tile_count": int(jet_tile_count),
                    "phase_jet_tile_count": int(phase_jet_tile_count),
                    "max_log_code_error_bound": float(log_code_error),
                    "max_phase_code_error_bound": float(phase_code_error),
                    "phase_error_bound": phase_error_radians,
                    "adaptive_jet_subdivision": jet_tile_count > 1 or phase_jet_tile_count > 1,
                    "fft_forbidden": True,
                    **(emission_diagnostics or {}),
                }
            ),
        )
        return NativeProjectionResult(values, report, log_codes, phase_codes)

    def _project_selected_jet(
        self,
        field: NativeField,
        grid: NativeGrid2D,
        *,
        semantic: OutputSemantic,
        support_epsilon: float,
        output: np.ndarray | None = None,
        _preparation: list[Any] | None = None,
    ) -> NativeProjectionResult:
        """Execute a planner-selected quadratic jet without atom reconstruction."""

        if field.representation.residual is not None:
            raise NotImplementedError(
                "selected quadratic jet has an external residual; use an explicit residual-aware consumer"
            )
        declared_support = field.support
        declared_region = declared_support.region if declared_support.kind is SupportKind.AABB else None
        support, disjoint = _intersection_region(None, declared_region)
        empty = bool(declared_support.kind is SupportKind.EMPTY or disjoint)
        squared = semantic is OutputSemantic.INTENSITY or (
            semantic is OutputSemantic.DISPLAY_RGB and field.coherence is Coherence.COHERENT_COMPLEX
        )
        emitted = (
            None
            if self.reference_planes
            else _emit_project(
                field.representation.payload,
                grid.shape,
                grid.origin,
                _grid_support_slices(grid, support, empty=empty),
                semantic=semantic.value,
                squared=squared,
                epsilon=support_epsilon,
                magnitude_lut=_MAGNITUDE_LUT,
                intensity_lut=_INTENSITY_LUT,
                retain_codes=self.retain_codes,
                output=output,
                _preparation=_preparation,
            )
        )
        if emitted is not None:
            stats = emitted.stats
            return self._finish_projection(
                field,
                grid,
                semantic,
                declared_support=declared_support,
                support=support,
                active_count=int(stats.retained),
                log_codes=emitted.logs,
                phase_codes=emitted.phases,
                log_code_error=emitted.log_error,
                phase_code_error=emitted.phase_error,
                jet_tile_count=emitted.tiles,
                phase_jet_tile_count=emitted.tiles if semantic is OutputSemantic.REAL_FIELD else 0,
                low_saturation=int(stats.low),
                high_saturation=int(stats.high),
                intermediate_arrays=list(emitted.allocations),
                source_representation=RepresentationKind.QUADRATIC_JET,
                support_epsilon=support_epsilon,
                evaluated_count=int(stats.evaluated),
                atoms_considered=0,
                projected_values=emitted.values,
                maximum_magnitude_override=0.0 if not stats.retained else math.exp2(stats.max_log / v02.SPF32_L_SCALE),
                maximum_output_override=stats.max_output,
                backend_name=emitted.backend,
                emission_diagnostics={
                    "execution_tiles": int(stats.tiles),
                    "integer_recurrence_samples": int(stats.integer_samples),
                    "fractional_recurrence_samples": int(stats.fractional_samples),
                    "output_scale": float(stats.max_output),
                    "bytes_written_model": "output_clear_plus_retained_stores_and_sidecars",
                    "culled_tiles": int(stats.culled),
                    "retained_codes": self.retain_codes,
                    "fused_terminal": emitted.backend == "c_fused_jet_terminal",
                    "bounded_tile_terminal": True,
                },
            )

        intermediate_arrays: list[tuple[str, int, int]] = []

        def track_intermediate(name: str, array: np.ndarray) -> None:
            value = np.asarray(array)
            intermediate_arrays.append((name, int(value.size), int(value.nbytes)))

        log_codes = np.full(grid.shape, v02.SPF32_L_ZERO, dtype=np.int32)
        track_intermediate("log_codes", log_codes)
        phase_codes: np.ndarray | None = None
        low_saturation = 0
        high_saturation = 0
        log_code_error = 0.0
        phase_code_error = 0.0
        jet_tile_count = 0
        phase_jet_tile_count = 0
        raw_log_codes: np.ndarray | None = None
        raw_phase_codes: np.ndarray | None = None
        evaluated_count = 0
        active_mask = np.zeros(grid.shape, dtype=bool) if empty else np.ones(grid.shape, dtype=bool)
        track_intermediate("active_mask", active_mask)
        if not empty:
            y_slice, x_slice = _grid_support_slices(grid, support, empty=False)
            support_mask = np.zeros(grid.shape, dtype=bool)
            track_intermediate("support_mask", support_mask)
            support_mask[y_slice, x_slice] = True
            active_mask &= support_mask
        if np.any(active_mask):
            raw_log_codes, raw_phase_codes, log_code_error, phase_code_error, jet_tile_count, phase_jet_tile_count = (
                _jet_plane_codes(
                    field.representation.payload,
                    grid.shape,
                    grid.origin,
                    include_phase=semantic is OutputSemantic.REAL_FIELD,
                    allocation_records=intermediate_arrays,
                )
            )
            evaluated_count = grid.size
            if support_epsilon > 0.0:
                squared_observable = semantic is OutputSemantic.INTENSITY or (
                    semantic is OutputSemantic.DISPLAY_RGB and field.coherence is Coherence.COHERENT_COMPLEX
                )
                magnitude_epsilon = math.sqrt(support_epsilon) if squared_observable else support_epsilon
                threshold_code = v02.SPF32_L_SCALE * math.log2(max(magnitude_epsilon, np.finfo(float).tiny))
                # The jet's log-plane error is in storage-code units.  Only
                # cull points whose *upper* possible magnitude is still below
                # the requested observable threshold; the remainder stays
                # materialized and therefore remains conservative.
                upper_log_codes = raw_log_codes + log_code_error
                track_intermediate("support_upper_log_codes", upper_log_codes)
                above_threshold = upper_log_codes > threshold_code
                track_intermediate("support_threshold_mask", above_threshold)
                active_mask &= above_threshold
            clipped = np.clip(raw_log_codes, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE)
            track_intermediate("clipped_log_codes", clipped)
            stored_log_codes = np.asarray(clipped, dtype=np.int32)
            track_intermediate("stored_log_codes", stored_log_codes)
            active_stored_log_codes = stored_log_codes[active_mask]
            track_intermediate("active_stored_log_codes", active_stored_log_codes)
            log_codes[active_mask] = active_stored_log_codes
            active_raw_log_codes = raw_log_codes[active_mask]
            track_intermediate("active_raw_log_codes", active_raw_log_codes)
            low_saturation_mask = active_raw_log_codes < v02.SPF32_L_MIN_FINITE
            high_saturation_mask = active_raw_log_codes > v02.SPF32_L_MAX_FINITE
            track_intermediate("low_saturation_mask", low_saturation_mask)
            track_intermediate("high_saturation_mask", high_saturation_mask)
            low_saturation = int(np.count_nonzero(low_saturation_mask))
            high_saturation = int(np.count_nonzero(high_saturation_mask))
            if raw_phase_codes is not None:
                phase_codes = np.zeros(grid.shape, dtype=np.uint16)
                track_intermediate("phase_codes", phase_codes)
                wrapped_phase_codes = raw_phase_codes & v02.SPF32_PHASE_MASK
                track_intermediate("wrapped_phase_codes", wrapped_phase_codes)
                stored_phase_codes = np.asarray(wrapped_phase_codes, dtype=np.uint16)
                track_intermediate("stored_phase_codes", stored_phase_codes)
                active_stored_phase_codes = stored_phase_codes[active_mask]
                track_intermediate("active_stored_phase_codes", active_stored_phase_codes)
                phase_codes[active_mask] = active_stored_phase_codes
        elif semantic is OutputSemantic.REAL_FIELD:
            # Ask the payload validator to distinguish a paired jet from a
            # plain log jet even when the declared support culls the tile.
            if _jet_payload_has_phase(field.representation.payload):
                phase_codes = np.zeros(grid.shape, dtype=np.uint16)
                track_intermediate("phase_codes", phase_codes)
                phase_code_error = 0.5
        active_count = int(np.count_nonzero(active_mask))
        result = self._finish_projection(
            field,
            grid,
            semantic,
            declared_support=declared_support,
            support=support,
            active_count=active_count,
            log_codes=log_codes,
            phase_codes=phase_codes,
            log_code_error=log_code_error,
            phase_code_error=phase_code_error,
            jet_tile_count=jet_tile_count,
            phase_jet_tile_count=phase_jet_tile_count,
            low_saturation=low_saturation,
            high_saturation=high_saturation,
            intermediate_arrays=intermediate_arrays,
            source_representation=RepresentationKind.QUADRATIC_JET,
            support_epsilon=support_epsilon,
            evaluated_count=evaluated_count,
            atoms_considered=0,
        )
        return _store_projection_output(result, output)

    def _project_incoherent_atom_mixture(
        self,
        field: NativeField,
        grid: NativeGrid2D,
        *,
        semantic: OutputSemantic,
        support_epsilon: float,
        output: np.ndarray | None = None,
    ) -> NativeProjectionResult:
        """Reduce positive atom magnitudes through the ROM-backed log path."""

        if semantic not in {
            OutputSemantic.MAGNITUDE,
            OutputSemantic.INTENSITY,
            OutputSemantic.RADIANCE,
            OutputSemantic.LOG_RADIANCE,
            OutputSemantic.DISPLAY_RGB,
        }:
            raise TypeError("incoherent atom mixtures do not provide a signed-real phase")
        mixture, declared_support = _prepare_native_mixture(field)
        declared_region = declared_support.region if declared_support.kind is SupportKind.AABB else None
        support, disjoint = _intersection_region(None, declared_region)
        empty = bool(declared_support.kind is SupportKind.EMPTY or disjoint)
        y_slice, x_slice = _grid_support_slices(grid, support, empty=empty)
        active_height = max(0, y_slice.stop - y_slice.start)
        active_width = max(0, x_slice.stop - x_slice.start)
        active_count = active_height * active_width
        local_grid = (
            NativeGrid2D(
                (active_height, active_width),
                origin=(
                    grid.origin[0] + x_slice.start * grid.spacing[0],
                    grid.origin[1] + y_slice.start * grid.spacing[1],
                ),
                spacing=grid.spacing,
            )
            if active_count
            else None
        )
        intermediate_arrays: list[tuple[str, int, int]] = []

        def track(name: str, array: np.ndarray) -> None:
            value = np.asarray(array)
            intermediate_arrays.append((name, int(value.size), int(value.nbytes)))

        planes: list[v02_planes.LogPosPlane] = []
        max_log_error = 0.0
        low_saturation = 0
        high_saturation = 0
        tile_count = 0
        if local_grid is not None:
            for atom_index, atom in enumerate(mixture.atoms):
                raw_log, log_error, tiles = _evaluate_atom_jet_tiled(
                    atom,
                    local_grid,
                    self.precision.log_guard_bits,
                    phase=False,
                    allocation_records=intermediate_arrays,
                    allocation_prefix=f"incoherent_{atom_index}_log_jet",
                )
                exponent_multiplier = 2 if semantic is OutputSemantic.INTENSITY else 1
                observable_log = raw_log * exponent_multiplier
                low_saturation += int(np.count_nonzero(observable_log < v02.SPF32_L_MIN_FINITE))
                high_saturation += int(np.count_nonzero(observable_log > v02.SPF32_L_MAX_FINITE))
                max_log_error = max(max_log_error, float(log_error) * exponent_multiplier)
                tile_count += int(tiles)
                stored_log = np.asarray(
                    np.clip(observable_log, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE), dtype=np.int16
                )
                codes = np.full(grid.shape, v02.SPF32_L_ZERO, dtype=np.int16)
                codes[y_slice, x_slice] = stored_log
                track(f"incoherent_{atom_index}_log_plane", codes)
                planes.append(v02_planes.LogPosPlane(codes))

        if planes:
            rom_plane = planes[0]
            for plane in planes[1:]:
                rom_plane = rom_plane.add(plane)
        else:
            rom_plane = v02_planes.LogPosPlane.zeros(grid.shape)
        rom_words = rom_plane.to_words()
        track("incoherent_rom_output_words", rom_words)

        # Run the typed pivot certificate over the same term planes.  The
        # consumed output remains the ROM fold; the pivot result supplies a
        # conservative cross-check for the fold's approximation.
        if planes:
            term_codes = np.stack(tuple(plane.L for plane in planes), axis=0)
            track("incoherent_log_term_stack", term_codes)
            pivot = v02_planes.reduce_logpos(
                v02_planes.LogPosPlane(term_codes),
                axis=0,
                spec=v02.ReductionSpec(precision_mode=self.precision.mode),
            )
            delta = float(
                np.max(
                    np.abs(rom_plane.to_values() - pivot.value.to_values()),
                    initial=0.0,
                )
            )
            pivot_error = float(pivot.ledger.total_bound) + delta + float(mixture.dropped_bound)
        else:
            pivot = None
            pivot_error = float(mixture.dropped_bound)
        reduced_log = np.asarray(rom_plane.L, dtype=np.int32)
        track("incoherent_reduced_log_codes", reduced_log)
        maximum_magnitude = _maximum_magnitude_from_log_codes(reduced_log)
        if semantic is OutputSemantic.LOG_RADIANCE and pivot_error:
            reducer_observable_error = math.inf
        elif semantic is OutputSemantic.INTENSITY:
            reducer_observable_error = pivot_error * (2.0 * maximum_magnitude + pivot_error)
        elif semantic is OutputSemantic.DISPLAY_RGB:
            reducer_observable_error = 255.0 if pivot_error else 0.0
        else:
            reducer_observable_error = pivot_error
        result = self._finish_projection(
            field,
            grid,
            semantic,
            declared_support=declared_support,
            support=support,
            active_count=active_count,
            log_codes=reduced_log,
            phase_codes=None,
            log_code_error=max_log_error,
            phase_code_error=0.0,
            jet_tile_count=tile_count,
            phase_jet_tile_count=0,
            low_saturation=low_saturation,
            high_saturation=high_saturation,
            intermediate_arrays=intermediate_arrays,
            source_representation=RepresentationKind.ATOM_MIXTURE,
            support_epsilon=support_epsilon,
            evaluated_count=active_count * len(mixture.atoms),
            atoms_considered=len(mixture.atoms),
            capabilities=(
                "typed_native_input",
                "symbolic_atom_mixture",
                "rom_logadd_plane_fold",
                "typed_pivot_reducer_certificate",
                f"project_{semantic.value}",
            ),
            workload="incoherent_atom_mixture_rom_reduction",
            maximum_magnitude_override=maximum_magnitude,
            backend_name="numpy_spf32_rom_logadd",
            emission_diagnostics={
                "mixture_atom_count": len(mixture.atoms),
                "mixture_dropped_bound": float(mixture.dropped_bound),
                "rom_logadd_count": max(0, len(planes) - 1) * grid.size,
                "rom_fold_output": True,
                "pivot_certificate_error": pivot_error,
                "pivot_certificate_method": None if pivot is None else pivot.method,
                "packed_word_term_elements": int(sum(plane.L.size for plane in planes)),
                "cartesian_term_intermediate": False,
            },
            additional_ledger=v02.ErrorLedger(reducer=reducer_observable_error),
            reduction_operations=max(0, len(planes) - 1),
            continuity_updates={
                "rom_logadd_count": max(0, len(planes) - 1) * grid.size,
                "typed_pivot_reduction_count": int(pivot is not None),
            },
        )
        return _store_projection_output(result, output)

    def _project_atom_mixture(
        self,
        field: NativeField,
        grid: NativeGrid2D,
        *,
        semantic: OutputSemantic,
        support_epsilon: float,
        output: np.ndarray | None = None,
    ) -> NativeProjectionResult:
        """Emit atom terms into packed word planes and reduce them natively."""

        if field.coherence is not Coherence.COHERENT_COMPLEX:
            return self._project_incoherent_atom_mixture(
                field,
                grid,
                semantic=semantic,
                support_epsilon=support_epsilon,
                output=output,
            )
        if semantic in {OutputSemantic.RADIANCE, OutputSemantic.LOG_RADIANCE}:
            raise TypeError(f"{semantic.value.upper()} projection requires an incoherent-positive field")
        mixture, declared_support = _prepare_native_mixture(field)
        declared_region = declared_support.region if declared_support.kind is SupportKind.AABB else None
        support, disjoint = _intersection_region(None, declared_region)
        empty = bool(declared_support.kind is SupportKind.EMPTY or disjoint)
        y_slice, x_slice = _grid_support_slices(grid, support, empty=empty)
        active_height = max(0, y_slice.stop - y_slice.start)
        active_width = max(0, x_slice.stop - x_slice.start)
        active_count = active_height * active_width
        local_grid = (
            NativeGrid2D(
                (active_height, active_width),
                origin=(
                    grid.origin[0] + x_slice.start * grid.spacing[0],
                    grid.origin[1] + y_slice.start * grid.spacing[1],
                ),
                spacing=grid.spacing,
            )
            if active_count
            else None
        )
        intermediate_arrays: list[tuple[str, int, int]] = []

        def track(name: str, array: np.ndarray) -> None:
            value = np.asarray(array)
            intermediate_arrays.append((name, int(value.size), int(value.nbytes)))

        term_words: list[np.ndarray] = []
        max_log_error = 0.0
        max_phase_error = 0.0
        low_saturation = 0
        high_saturation = 0
        tile_count = 0
        if local_grid is not None:
            for atom_index, atom in enumerate(mixture.atoms):
                raw_log, log_error, log_tiles = _evaluate_atom_jet_tiled(
                    atom,
                    local_grid,
                    self.precision.log_guard_bits,
                    phase=False,
                    allocation_records=intermediate_arrays,
                    allocation_prefix=f"mixture_{atom_index}_log_jet",
                )
                raw_phase, phase_error, phase_tiles = _evaluate_atom_jet_tiled(
                    atom,
                    local_grid,
                    self.precision.phase_guard_bits,
                    phase=True,
                    allocation_records=intermediate_arrays,
                    allocation_prefix=f"mixture_{atom_index}_phase_jet",
                )
                stored_log = np.asarray(
                    np.clip(raw_log, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE), dtype=np.int32
                )
                stored_phase = np.asarray(raw_phase & v02.SPF32_PHASE_MASK, dtype=np.uint16)
                low_saturation += int(np.count_nonzero(raw_log < v02.SPF32_L_MIN_FINITE))
                high_saturation += int(np.count_nonzero(raw_log > v02.SPF32_L_MAX_FINITE))
                max_log_error = max(max_log_error, float(log_error))
                max_phase_error = max(max_phase_error, float(phase_error))
                tile_count += max(int(log_tiles), int(phase_tiles))
                local_words = np.asarray(v02.pack_spf32(stored_log, stored_phase), dtype=np.uint32)
                track(f"mixture_{atom_index}_packed_words", local_words)
                words = np.full(grid.shape, v02.SPF32_WORD_ZERO, dtype=np.uint32)
                words[y_slice, x_slice] = local_words
                term_words.append(words)
                track(f"mixture_{atom_index}_word_plane", words)

        packed_terms = (
            np.stack(term_words, axis=0)
            if term_words
            else np.zeros((1, *grid.shape), dtype=np.uint32)
        )
        track("mixture_packed_term_stack", packed_terms)
        reduction = v02_planes.reduce_complex(
            packed_terms,
            axis=0,
            spec=v02.ReductionSpec(precision_mode=self.precision.mode),
        )
        reduced_words = np.asarray(reduction.value, dtype=np.uint32)
        track("mixture_reduced_words", reduced_words)
        reduced_log = np.asarray(v02.unpack_L(reduced_words), dtype=np.int32)
        track("mixture_reduced_log_codes", reduced_log)
        reduced_phase = np.asarray(v02.unpack_T(reduced_words), dtype=np.uint16)
        track("mixture_reduced_phase_codes", reduced_phase)
        phase_reduction_codes = (
            float(np.max(reduction.phase_error_bound, initial=0.0)) * v02.SPF32_PHASE_MOD / (2.0 * math.pi)
        )
        maximum_magnitude = _maximum_magnitude_from_log_codes(reduced_log)
        reducer_amplitude_error = float(reduction.ledger.total_bound)
        if mixture.dropped_bound:
            reducer_amplitude_error += float(mixture.dropped_bound)
        if semantic is OutputSemantic.INTENSITY:
            reducer_observable_error = reducer_amplitude_error * (
                2.0 * maximum_magnitude + reducer_amplitude_error
            )
        elif semantic is OutputSemantic.DISPLAY_RGB:
            reducer_observable_error = 255.0 if reducer_amplitude_error else 0.0
        else:
            reducer_observable_error = reducer_amplitude_error
        reducer_ledger = v02.ErrorLedger(reducer=reducer_observable_error)
        phase_codes = reduced_phase if semantic is OutputSemantic.REAL_FIELD else None
        result = self._finish_projection(
            field,
            grid,
            semantic,
            declared_support=declared_support,
            support=support,
            active_count=active_count,
            log_codes=reduced_log,
            phase_codes=phase_codes,
            log_code_error=max_log_error,
            phase_code_error=max_phase_error + phase_reduction_codes,
            jet_tile_count=tile_count,
            phase_jet_tile_count=tile_count,
            low_saturation=low_saturation,
            high_saturation=high_saturation,
            intermediate_arrays=intermediate_arrays,
            source_representation=RepresentationKind.ATOM_MIXTURE,
            support_epsilon=support_epsilon,
            evaluated_count=active_count * len(mixture.atoms),
            atoms_considered=len(mixture.atoms),
            capabilities=(
                "typed_native_input",
                "symbolic_atom_mixture",
                "spf32_word_plane_carrier",
                "typed_pivot_reducer",
                f"project_{semantic.value}",
            ),
            workload="symbolic_atom_mixture_word_reduction",
            maximum_magnitude_override=maximum_magnitude,
            backend_name="numpy_spf32_pivot_reducer",
            emission_diagnostics={
                "mixture_atom_count": len(mixture.atoms),
                "mixture_dropped_bound": float(mixture.dropped_bound),
                "reducer_method": reduction.method,
                "reducer_native": bool(reduction.native),
                "reducer_counters": reduction.counters.as_dict(),
                "reducer_phase_error_radians": float(np.max(reduction.phase_error_bound, initial=0.0)),
                "packed_word_term_elements": int(packed_terms.size),
                "cartesian_term_intermediate": False,
            },
            additional_ledger=reducer_ledger,
            reduction_operations=1,
            continuity_updates={"typed_pivot_reduction_count": 1},
        )
        return _store_projection_output(result, output)

    def project(
        self,
        field: NativeField,
        grid: NativeGrid2D,
        *,
        semantic: OutputSemantic | str,
        support_epsilon: float = 0.0,
        output: np.ndarray | None = None,
        _preparation: list[Any] | None = None,
    ) -> NativeProjectionResult:
        selected = _output_semantic(semantic)
        if field.domain is not Domain.SPATIAL_2D:
            raise ValueError("native projection requires a SPATIAL_2D field")
        representation_kind = field.representation.kind
        if representation_kind is RepresentationKind.QUADRATIC_JET:
            if selected in {OutputSemantic.RADIANCE, OutputSemantic.LOG_RADIANCE} and (
                field.coherence is not Coherence.INCOHERENT_POSITIVE
            ):
                raise TypeError(f"{selected.value.upper()} projection requires an incoherent-positive field")
            if selected is OutputSemantic.REAL_FIELD and field.coherence is not Coherence.COHERENT_COMPLEX:
                raise TypeError("REAL_FIELD projection requires a coherent-complex field")
            epsilon = float(support_epsilon)
            if epsilon < 0.0 or not math.isfinite(epsilon):
                raise ValueError("support epsilon must be finite and non-negative")
            return self._project_selected_jet(
                field,
                grid,
                semantic=selected,
                support_epsilon=epsilon,
                output=output,
                _preparation=_preparation,
            )
        if representation_kind is RepresentationKind.ATOM_MIXTURE:
            epsilon = float(support_epsilon)
            if epsilon < 0.0 or not math.isfinite(epsilon):
                raise ValueError("support epsilon must be finite and non-negative")
            if selected is OutputSemantic.REAL_FIELD and field.coherence is not Coherence.COHERENT_COMPLEX:
                raise TypeError("REAL_FIELD projection requires a coherent-complex field")
            return self._project_atom_mixture(
                field,
                grid,
                semantic=selected,
                support_epsilon=epsilon,
                output=output,
            )
        else:
            atom, declared_support = _prepare_native_atom(field)
        if selected in {OutputSemantic.RADIANCE, OutputSemantic.LOG_RADIANCE} and (
            field.coherence is not Coherence.INCOHERENT_POSITIVE
        ):
            raise TypeError(f"{selected.value.upper()} projection requires an incoherent-positive field")
        if selected is OutputSemantic.REAL_FIELD and field.coherence is not Coherence.COHERENT_COMPLEX:
            raise TypeError("REAL_FIELD projection requires a coherent-complex field")
        epsilon = float(support_epsilon)
        if epsilon < 0.0 or not math.isfinite(epsilon):
            raise ValueError("support epsilon must be finite and non-negative")
        squared_observable = selected is OutputSemantic.INTENSITY or (
            selected is OutputSemantic.DISPLAY_RGB and field.coherence is Coherence.COHERENT_COMPLEX
        )
        magnitude_epsilon = math.sqrt(epsilon) if squared_observable else epsilon
        analytic_support = _certified_support_box(atom, magnitude_epsilon)
        real_c = np.real(atom.C)
        inverse = (
            np.linalg.inv(real_c) if analytic_support is not None and np.min(np.linalg.eigvalsh(real_c)) > 0.0 else None
        )
        peak = (
            float(np.real(atom.kappa) + 0.5 * np.real(atom.h) @ inverse @ np.real(atom.h))
            if inverse is not None
            else math.inf
        )
        analytic_empty = bool(
            analytic_support is not None and peak <= math.log(max(magnitude_epsilon, np.finfo(float).tiny))
        )
        declared_region = declared_support.region if declared_support.kind is SupportKind.AABB else None
        support, disjoint = _intersection_region(analytic_support, declared_region)
        empty = bool(declared_support.kind is SupportKind.EMPTY or analytic_empty or disjoint)
        y_slice, x_slice = _grid_support_slices(grid, support, empty=empty)
        active_height = max(0, y_slice.stop - y_slice.start)
        active_width = max(0, x_slice.stop - x_slice.start)
        active_count = active_height * active_width

        if not self.reference_planes:
            headers = _atom_projection_tiles(
                atom,
                grid,
                self.precision.log_guard_bits,
                self.precision.phase_guard_bits,
                (x_slice.start, x_slice.stop, y_slice.start, y_slice.stop),
                selected is OutputSemantic.REAL_FIELD,
            )
            emitted = _emit_project(
                headers,
                grid.shape,
                grid.origin,
                (y_slice, x_slice),
                semantic=selected.value,
                squared=squared_observable,
                epsilon=0.0,
                magnitude_lut=_MAGNITUDE_LUT,
                intensity_lut=_INTENSITY_LUT,
                retain_codes=self.retain_codes,
                output=output,
                _preparation=_preparation,
            )
            if emitted is not None:
                stats = emitted.stats
                return self._finish_projection(
                    field,
                    grid,
                    selected,
                    declared_support=declared_support,
                    support=support,
                    active_count=int(stats.retained),
                    log_codes=emitted.logs,
                    phase_codes=emitted.phases,
                    log_code_error=emitted.log_error,
                    phase_code_error=emitted.phase_error,
                    jet_tile_count=emitted.tiles,
                    phase_jet_tile_count=emitted.tiles if selected is OutputSemantic.REAL_FIELD else 0,
                    low_saturation=int(stats.low),
                    high_saturation=int(stats.high),
                    intermediate_arrays=list(emitted.allocations),
                    source_representation=RepresentationKind(representation_kind),
                    support_epsilon=epsilon,
                    evaluated_count=int(stats.evaluated),
                    projected_values=emitted.values,
                    maximum_magnitude_override=0.0
                    if not stats.retained
                    else math.exp2(stats.max_log / v02.SPF32_L_SCALE),
                    maximum_output_override=stats.max_output,
                    backend_name=emitted.backend,
                    emission_diagnostics={
                        "execution_tiles": int(stats.tiles),
                        "integer_recurrence_samples": int(stats.integer_samples),
                        "fractional_recurrence_samples": int(stats.fractional_samples),
                        "output_scale": float(stats.max_output),
                        "bytes_written_model": "output_clear_plus_retained_stores_and_sidecars",
                        "culled_tiles": int(stats.culled),
                        "retained_codes": self.retain_codes,
                        "bounded_tile_terminal": True,
                        "fused_terminal": emitted.backend == "c_fused_jet_terminal",
                        "atom_headers_prepared": True,
                    },
                )

        # Track logical intermediate allocations explicitly.  Short-lived
        # NumPy arrays still count toward the v0.2 materialization metric.
        intermediate_arrays: list[tuple[str, int, int]] = []

        def track_intermediate(name: str, array: np.ndarray) -> None:
            value = np.asarray(array)
            intermediate_arrays.append((name, int(value.size), int(value.nbytes)))

        log_codes = np.full(grid.shape, v02.SPF32_L_ZERO, dtype=np.int32)
        track_intermediate("log_codes", log_codes)
        phase_codes: np.ndarray | None = None
        raw_log_codes = np.empty((0, 0), dtype=np.int64)
        log_code_error = 0.0
        phase_code_error = 0.0
        jet_tile_count = 0
        phase_jet_tile_count = 0
        if active_count:
            local_grid = NativeGrid2D(
                (active_height, active_width),
                origin=(
                    grid.origin[0] + x_slice.start * grid.spacing[0],
                    grid.origin[1] + y_slice.start * grid.spacing[1],
                ),
                spacing=grid.spacing,
            )
            raw_log_codes, log_code_error, jet_tile_count = _evaluate_atom_jet_tiled(
                atom,
                local_grid,
                self.precision.log_guard_bits,
                phase=False,
                allocation_records=intermediate_arrays,
                allocation_prefix="log_jet",
            )
            clipped = np.clip(raw_log_codes, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE)
            track_intermediate("clipped_log_codes", clipped)
            stored_log_codes = np.asarray(clipped, dtype=np.int32)
            track_intermediate("stored_log_codes", stored_log_codes)
            log_codes[y_slice, x_slice] = stored_log_codes

        low_saturation_mask = raw_log_codes < v02.SPF32_L_MIN_FINITE
        high_saturation_mask = raw_log_codes > v02.SPF32_L_MAX_FINITE
        track_intermediate("low_saturation_mask", low_saturation_mask)
        track_intermediate("high_saturation_mask", high_saturation_mask)
        low_saturation = int(np.count_nonzero(low_saturation_mask))
        high_saturation = int(np.count_nonzero(high_saturation_mask))

        zero = log_codes == v02.SPF32_L_ZERO
        track_intermediate("zero_mask", zero)
        if selected is OutputSemantic.REAL_FIELD:
            phase_codes = np.zeros(grid.shape, dtype=np.uint16)
            track_intermediate("phase_codes", phase_codes)
            if active_count:
                local_phase, phase_code_error, phase_jet_tile_count = _evaluate_atom_jet_tiled(
                    atom,
                    local_grid,
                    self.precision.phase_guard_bits,
                    phase=True,
                    allocation_records=intermediate_arrays,
                    allocation_prefix="phase_jet",
                )
                wrapped_phase_codes = local_phase & v02.SPF32_PHASE_MASK
                track_intermediate("wrapped_phase_codes", wrapped_phase_codes)
                stored_phase_codes = np.asarray(wrapped_phase_codes, dtype=np.uint16)
                track_intermediate("stored_phase_codes", stored_phase_codes)
                phase_codes[y_slice, x_slice] = stored_phase_codes
            magnitude, lut_workspace = _project_log_codes_accounted(log_codes, OutputSemantic.MAGNITUDE, zero)
            intermediate_arrays.extend(lut_workspace)
            phase_float_codes = phase_codes.astype(np.float64)
            track_intermediate("phase_float_codes", phase_float_codes)
            phase_radians = phase_float_codes * (2.0 * math.pi / v02.SPF32_PHASE_MOD)
            track_intermediate("phase_radians", phase_radians)
            cosine = np.cos(phase_radians)
            track_intermediate("phase_cosine", cosine)
            real_values = magnitude * cosine
            track_intermediate("real_values_float64", real_values)
            values = np.asarray(real_values, dtype=np.float32)
        else:
            values, lut_workspace = _project_log_codes_accounted(
                log_codes,
                selected,
                zero,
                coherent_display=field.coherence is Coherence.COHERENT_COMPLEX,
            )
            intermediate_arrays.extend(lut_workspace)

        # ``raw_log_codes`` and the local phase plane are accounted for above;
        # their buffers need not stay live while the certificate is built.
        del raw_log_codes
        if selected is OutputSemantic.REAL_FIELD and active_count:
            del local_phase
        pruned_elements = grid.size - active_count
        maximum_magnitude = _maximum_magnitude_from_log_codes(log_codes)
        ledger = _projection_error_ledger(
            values,
            selected,
            maximum_magnitude=maximum_magnitude,
            log_code_error=log_code_error,
            phase_code_error=phase_code_error,
            support_epsilon=epsilon,
            pruned_elements=pruned_elements,
            saturated_elements=low_saturation + high_saturation,
        )
        # Preserve inherited uncertainty even when support culling emits no
        # active samples, and retain its original ledger category.
        input_error = _input_error_for_projection(field, values, selected)
        ledger = ledger.plus(input_error)
        phase_error_radians = _phase_code_error_to_radians(phase_code_error)
        _enforce_native_contract(field, ledger, values, phase_error=phase_error_radians)
        continuity = _field_continuity(field)
        if field.operations:
            continuity["symbolic_operation_count"] += len(field.operations)
            continuity["potential_dense_passes"] += len(field.operations)
        continuity["native_quantization_count"] += 1
        conversion_counts = {
            "cartesian_ingress_conversions": continuity["legacy_import_count"],
            "cartesian_internal_conversions": continuity["cartesian_intermediate_count"],
            "cartesian_egress_conversions": continuity["legacy_export_count"],
            "legacy_adapter_calls": continuity["legacy_import_count"] + continuity["legacy_export_count"],
        }
        if self.strict_native and any(conversion_counts.values()):
            raise RuntimeError(f"native continuity violation: {conversion_counts}")

        # M_A is defined over dense intermediates, not the requested ROI.  The
        # legacy potential/materialized fields remain populated for consumers
        # that have not adopted the explicit split yet.
        intermediate_elements = int(sum(size for _, size, _ in intermediate_arrays))
        intermediate_bytes = int(sum(nbytes for _, _, nbytes in intermediate_arrays))
        terminal_elements = int(values.size)
        terminal_bytes = int(values.nbytes)
        potential = int(grid.potential_size)
        potential_passes = max(1, int(continuity["potential_dense_passes"]))
        potential_intermediate_elements = potential * potential_passes
        # The dense baseline represented by this numerator carries complex64
        # samples (8 bytes/scalar); terminal output and native intermediates
        # retain their actual dtypes below.
        potential_intermediate_bytes = potential_intermediate_elements * int(np.dtype(np.complex64).itemsize)
        logical_peak_bytes = int(field.representation.byte_size or 0) + intermediate_bytes + terminal_bytes
        potential_pass_elements = grid.size * potential_passes
        metrics = RenderMetrics(
            backend="numpy_add_accumulate_reference",
            workload="gaussian_chirp_closed_form",
            resolution=grid.shape,
            representation=RepresentationKind.QUADRATIC_JET.value,
            precision=str(self.precision.mode).lower(),
            bytes_read=int(field.representation.byte_size or 0),
            bytes_written=terminal_bytes,
            # This is a conservative logical live-set model: it includes the
            # transient log/LUT arrays listed below, not just returned pixels.
            peak_memory=logical_peak_bytes,
            potential_elements=potential,
            # Keep the old field truthful as total scalar values produced,
            # while M_A uses the explicit intermediate fields.
            materialized_elements=intermediate_elements + terminal_elements,
            potential_intermediate_elements=potential_intermediate_elements,
            materialized_intermediate_elements=intermediate_elements,
            terminal_output_elements=terminal_elements,
            potential_intermediate_bytes=potential_intermediate_bytes,
            materialized_intermediate_bytes=intermediate_bytes,
            terminal_output_bytes=terminal_bytes,
            logical_peak_bytes=logical_peak_bytes,
            atoms_considered=1,
            atoms_pruned=int(active_count == 0),
            fft_count=0,
            fallback_conversions=0,
            field_operations=continuity["symbolic_operation_count"],
            reduction_operations=0,
        )
        report = RenderReport(
            metrics,
            ledger,
            passes=(
                CompilerPassRecord("analytic_atom_closure", continuity["symbolic_operation_count"], 1),
                CompilerPassRecord("atom_to_log_jet", 1, 1),
                CompilerPassRecord("native_projection", 1, 1, notes=(selected.value,)),
            ),
            decisions=(
                RepresentationDecision(
                    field.id,
                    repr(grid),
                    RepresentationKind.QUADRATIC_JET.value,
                    (RepresentationKind.GAUSSIAN_CHIRP_ATOM.value, RepresentationKind.QUADRATIC_JET.value),
                    float(active_count),
                    float(ledger.total_bound),
                    "closed Gaussian--chirp lowered to a guarded log jet; dense FFT forbidden",
                ),
            ),
            capabilities_used=(
                "typed_native_input",
                "gaussian_chirp_product_closure",
                "analytic_inverse_fourier",
                "jet2_forward_difference",
                f"project_{selected.value}",
            ),
            diagnostics=MappingProxyType(
                {
                    "output_semantic": selected.value,
                    "conversion_counts": conversion_counts,
                    "native_counts": continuity,
                    "materialization_count": 1,
                    "display_projection_count": int(selected is OutputSemantic.DISPLAY_RGB),
                    "magnitude_projection_count": int(selected is OutputSemantic.MAGNITUDE),
                    "intensity_projection_count": int(selected is OutputSemantic.INTENSITY),
                    "real_projection_count": int(selected is OutputSemantic.REAL_FIELD),
                    "radiance_projection_count": int(selected is OutputSemantic.RADIANCE),
                    "log_radiance_projection_count": int(selected is OutputSemantic.LOG_RADIANCE),
                    "support_epsilon": epsilon,
                    "support_region": None if support is None else {"lower": support.lower, "upper": support.upper},
                    "declared_support_kind": getattr(declared_support.kind, "value", str(declared_support.kind)),
                    "support_evaluated_elements": active_count,
                    "support_pruned_elements": pruned_elements,
                    "requested_elements": grid.size,
                    "materialized_intermediate_elements": intermediate_elements,
                    "materialized_intermediate_bytes": intermediate_bytes,
                    "intermediate_allocations": tuple(intermediate_arrays),
                    "terminal_header_elements": active_count,
                    "terminal_output_elements": terminal_elements,
                    "terminal_output_bytes": terminal_bytes,
                    "potential_intermediate_elements": potential_intermediate_elements,
                    "potential_intermediate_bytes": potential_intermediate_bytes,
                    "potential_intermediate_dtype": "complex64",
                    "logical_peak_bytes": logical_peak_bytes,
                    "peak_memory_status": (
                        "conservative_upper_bound_from_explicit_numpy_allocations; "
                        "allocator_and_python_object_overhead_unmeasured"
                    ),
                    "potential_pass_elements": potential_pass_elements,
                    "realized_pass_elements": intermediate_elements,
                    "pass_avoidance": (
                        float(potential_pass_elements) / float(intermediate_elements)
                        if intermediate_elements
                        else math.inf
                    ),
                    "low_saturation_elements": low_saturation,
                    "high_saturation_elements": high_saturation,
                    "jet_tile_count": jet_tile_count,
                    "phase_jet_tile_count": phase_jet_tile_count,
                    "max_log_code_error_bound": log_code_error,
                    "max_phase_code_error_bound": phase_code_error,
                    "phase_error_bound": phase_error_radians,
                    "adaptive_jet_subdivision": jet_tile_count > 1 or phase_jet_tile_count > 1,
                    "fft_forbidden": True,
                }
            ),
        )
        return _store_projection_output(NativeProjectionResult(values, report, log_codes, phase_codes), output)

    def project_rgb(
        self,
        fields: Sequence[NativeField],
        grid: NativeGrid2D,
        *,
        semantic: OutputSemantic | str = OutputSemantic.INTENSITY,
        support_epsilon: float = 0.0,
        output: np.ndarray | None = None,
    ) -> NativeProjectionResult:
        """Project three typed channels and aggregate their accounting.

        Without diagnostic sidecars, channels write into one RGB buffer.
        Compatibility sidecars retain sequential channel planes followed by
        stack assembly; accounting includes their intermediate live set.
        """

        channels = tuple(fields)
        if len(channels) != 3:
            raise ValueError("project_rgb requires exactly three NativeField channels")
        direct = not self.retain_codes
        selected = _output_semantic(semantic)
        if direct and not self.reference_planes:
            from spf.renderer._jet_kernel import batch_kernel
            from spf.renderer.emission import kernel as selected_kernel

            if selected_kernel() is not None and batch_kernel() is not None:
                key = (tuple(id(channel) for channel in channels), grid, selected, support_epsilon, self.precision)
                if key not in self._rgb_prepared:
                    try:
                        prepared = self.prepare_rgb(channels, grid, semantic=selected, support_epsilon=support_epsilon)
                    except NotImplementedError:
                        # A valid projection can exceed the bounded native
                        # header range. Keep its declared portable consumer.
                        pass
                    else:
                        if len(self._rgb_prepared) >= 8:
                            self._rgb_prepared.pop(next(iter(self._rgb_prepared)))
                        self._rgb_prepared[key] = (channels, prepared)
                if key in self._rgb_prepared:
                    return self._rgb_prepared[key][1].execute(output=output)
        destination = (
            np.empty((*grid.shape, 3), dtype=np.uint8 if selected is OutputSemantic.DISPLAY_RGB else np.float32)
            if direct
            else None
        )
        results = tuple(
            self.project(
                field,
                grid,
                semantic=semantic,
                support_epsilon=support_epsilon,
                output=None if destination is None else destination[..., index],
            )
            for index, field in enumerate(channels)
        )
        values = np.stack(tuple(result.values for result in results), axis=-1) if destination is None else destination
        log_codes = (
            np.stack(tuple(result.log_codes for result in results), axis=-1)
            if self.retain_codes
            else np.empty(0, dtype=np.int32)
        )
        has_phase = any(result.phase_codes is not None for result in results)
        if has_phase and not all(result.phase_codes is not None for result in results):
            raise RuntimeError("RGB channels disagree on phase-plane materialization")
        phase_planes = tuple(np.asarray(result.phase_codes) for result in results if result.phase_codes is not None)
        phase_codes = np.stack(phase_planes, axis=-1) if has_phase else None

        metrics = RenderMetrics()
        for result in results:
            metrics.add(result.report.metrics)
        channel_terminal_elements = metrics.terminal_output_elements
        channel_terminal_bytes = metrics.terminal_output_bytes
        # The per-channel value planes are temporary inputs to RGB assembly.
        # The stacked log/phase sidecars returned by NativeProjectionResult are
        # additional materialized arrays and must not disappear from M_A or the
        # live-set model merely because they are diagnostic/native outputs.
        stacked_sidecar_elements = int(log_codes.size) + (0 if phase_codes is None else int(phase_codes.size))
        stacked_sidecar_bytes = int(log_codes.nbytes) + (0 if phase_codes is None else int(phase_codes.nbytes))
        metrics.materialized_intermediate_elements += (
            0 if direct else channel_terminal_elements
        ) + stacked_sidecar_elements
        metrics.materialized_intermediate_bytes += (0 if direct else channel_terminal_bytes) + stacked_sidecar_bytes
        metrics.terminal_output_elements = int(values.size)
        metrics.terminal_output_bytes = int(values.nbytes)
        metrics.materialized_elements = metrics.materialized_intermediate_elements + int(values.size)
        if not direct:
            metrics.bytes_written += int(values.nbytes)
        retained_bytes = 0
        sequential_peak = 0
        for result in results:
            sequential_peak = max(
                sequential_peak,
                retained_bytes + max(result.report.metrics.logical_peak_bytes, result.report.metrics.peak_memory),
            )
            retained_bytes += (
                int(result.values.nbytes)
                + int(result.log_codes.nbytes)
                + (0 if result.phase_codes is None else int(result.phase_codes.nbytes))
            )
        assembly_peak = retained_bytes + int(values.nbytes) + stacked_sidecar_bytes
        metrics.logical_peak_bytes = max(sequential_peak, assembly_peak)
        if direct:
            metrics.logical_peak_bytes = int(values.nbytes) + max(
                max(0, result.report.metrics.logical_peak_bytes - result.values.nbytes) for result in results
            )
        metrics.peak_memory = metrics.logical_peak_bytes
        selected = _output_semantic(semantic)
        metrics.backend = results[0].report.metrics.backend
        metrics.workload = "native_rgb_projection"
        metrics.resolution = (*grid.shape, 3)
        metrics.representation = f"RGB[{RepresentationKind.QUADRATIC_JET.value}]"
        metrics.precision = str(self.precision.mode).lower()
        metrics.error_budget = max(result.report.metrics.error_budget for result in results)

        ledger = v02.ErrorLedger()
        passes: list[CompilerPassRecord] = []
        decisions: list[RepresentationDecision] = []
        fallbacks: list[str] = []
        capabilities: list[str] = []
        channel_diagnostics: list[dict[str, Any]] = []
        aggregate_conversion_counts = {
            "cartesian_ingress_conversions": 0,
            "cartesian_internal_conversions": 0,
            "cartesian_egress_conversions": 0,
            "legacy_adapter_calls": 0,
        }
        aggregate_native_counts: dict[str, int] = {}
        for result in results:
            report = result.report
            ledger = ledger.plus(report.ledger)
            passes.extend(report.passes)
            decisions.extend(report.decisions)
            fallbacks.extend(report.fallbacks)
            for capability in report.capabilities_used:
                if capability not in capabilities:
                    capabilities.append(capability)
            diagnostics = dict(report.diagnostics)
            channel_diagnostics.append(diagnostics)
            for key, value in diagnostics.get("conversion_counts", {}).items():
                aggregate_conversion_counts[key] = aggregate_conversion_counts.get(key, 0) + int(value)
            for key, value in diagnostics.get("native_counts", {}).items():
                aggregate_native_counts[key] = aggregate_native_counts.get(key, 0) + int(value)

        aggregate_diagnostics: dict[str, Any] = {
            "output_semantic": selected.value,
            "rgb_aggregated": True,
            "rgb_channel_count": 3,
            "rgb_direct_output": direct,
            "channel_diagnostics": tuple(channel_diagnostics),
            "conversion_counts": aggregate_conversion_counts,
            "native_counts": aggregate_native_counts,
            "materialization_count": sum(int(item.get("materialization_count", 0)) for item in channel_diagnostics),
            "requested_elements": int(grid.size * 3),
            "support_evaluated_elements": sum(
                int(item.get("support_evaluated_elements", 0)) for item in channel_diagnostics
            ),
            "evaluated_grid_elements": sum(
                int(item.get("evaluated_grid_elements", item.get("support_evaluated_elements", 0)))
                for item in channel_diagnostics
            ),
            "retained_output_elements": sum(
                int(item.get("retained_output_elements", 0)) for item in channel_diagnostics
            ),
            "support_pruned_elements": sum(int(item.get("support_pruned_elements", 0)) for item in channel_diagnostics),
            "materialized_intermediate_elements": metrics.materialized_intermediate_elements,
            "materialized_intermediate_bytes": metrics.materialized_intermediate_bytes,
            "stacked_sidecar_elements": stacked_sidecar_elements,
            "stacked_sidecar_bytes": stacked_sidecar_bytes,
            "terminal_output_elements": metrics.terminal_output_elements,
            "terminal_output_bytes": metrics.terminal_output_bytes,
            "potential_intermediate_elements": metrics.potential_intermediate_elements,
            "potential_intermediate_bytes": metrics.potential_intermediate_bytes,
            "logical_peak_bytes": metrics.logical_peak_bytes,
            "peak_memory_status": (
                "conservative_upper_bound_including_channel_and_stacked_sidecars; "
                "allocator_and_python_object_overhead_unmeasured"
            ),
            "phase_error_bound": max(float(item.get("phase_error_bound", 0.0)) for item in channel_diagnostics),
            "fft_forbidden": all(bool(item.get("fft_forbidden", False)) for item in channel_diagnostics),
        }
        report = RenderReport(
            metrics,
            ledger,
            passes=tuple(passes),
            decisions=tuple(decisions),
            fallbacks=tuple(dict.fromkeys(fallbacks)),
            capabilities_used=tuple(capabilities),
            diagnostics=aggregate_diagnostics,
        )
        return _store_projection_output(NativeProjectionResult(values, report, log_codes, phase_codes), output)

    def prepare_rgb(self, fields: Sequence[NativeField], grid: NativeGrid2D, **options: Any) -> PreparedNativeRGB:
        return PreparedNativeRGB.prepare(tuple(self.prepare_projection(f, grid, **options) for f in fields))

    def prepare_projection(self, field: NativeField, grid: NativeGrid2D, **options: Any) -> PreparedNativeProjection:
        plans: list[Any] = []
        result = self.project(field, grid, _preparation=plans, **options)
        if len(plans) != 1:
            raise NotImplementedError("this projection requires the portable/reference consumer")
        return PreparedNativeProjection(plans[0], result.report)

    def integrate_footprint(
        self,
        field: NativeField,
        footprint: v02.GaussianChirpAtom,
    ) -> NativeIntegralResult:
        """Integrate a closed Gaussian atom against a Gaussian footprint."""

        if not isinstance(footprint, v02.GaussianChirpAtom) or footprint.dimension != 2:
            raise TypeError("analytic footprint integration requires a two-dimensional GaussianChirpAtom")
        atom, support = _prepare_native_atom(field)
        if support.kind is not SupportKind.FULL:
            raise NotImplementedError("analytic footprint integration requires FULL atom support")
        packed_operations = 0
        packed_decodes = 0
        if self.packed_atom_arithmetic:
            product_header = v02.PackedGaussianChirpAtom.from_atom(atom).product(
                v02.PackedGaussianChirpAtom.from_atom(footprint)
            )
            product = product_header.to_atom()
            packed_operations = 1
            packed_decodes = 1
        else:
            product = v02.atom_mul(atom, footprint)
        value = complex(v02.atom_integral(product))
        from spf.renderer.normalization import complete_field_ledger

        continuity = _field_continuity(field)
        continuity["symbolic_operation_count"] += 1
        continuity["potential_dense_passes"] += 1
        continuity["footprint_integration_count"] += 1
        continuity["packed_atom_arithmetic_count"] += packed_operations
        continuity["packed_atom_decode_count"] += packed_decodes
        continuity["native_pack_count"] += 2 * packed_operations
        ledger = complete_field_ledger(field)
        if packed_operations:
            # The word-header quantization is explicit, but its effect on a
            # whole-space integral is not bounded by the local projection
            # certificate.  Preserve the result while refusing a false finite
            # certificate for this optional mode.
            ledger = ledger.plus(v02.ErrorLedger(materialization=math.inf))
        metrics = RenderMetrics(
            backend="analytic_gaussian_integral",
            workload="analytic_footprint_integral",
            resolution=(),
            representation=RepresentationKind.GAUSSIAN_CHIRP_ATOM.value,
            precision=str(self.precision.mode).lower(),
            bytes_read=int(field.representation.byte_size or 0),
            bytes_written=16,
            terminal_output_elements=1,
            terminal_output_bytes=16,
            atoms_considered=1,
            field_operations=continuity["symbolic_operation_count"],
            reduction_operations=1,
            fft_count=0,
            fallback_conversions=0,
        )
        report = RenderReport(
            metrics,
            ledger,
            passes=(
                CompilerPassRecord("analytic_atom_product", 2, 1),
                CompilerPassRecord("analytic_footprint_integral", 1, 1),
            ),
            decisions=(
                RepresentationDecision(
                    field.id,
                    "whole_space",
                    RepresentationKind.GAUSSIAN_CHIRP_ATOM.value,
                    (RepresentationKind.GAUSSIAN_CHIRP_ATOM.value, "ANALYTIC_FOOTPRINT_INTEGRAL"),
                    1.0,
                    float(ledger.total_bound),
                    "Gaussian product and integral stayed symbolic; no sample plane was materialized",
                ),
            ),
            capabilities_used=(
                "typed_native_input",
                "gaussian_product_closure",
                "analytic_footprint_integration",
            ),
            diagnostics={
                "fft_forbidden": True,
                "cartesian_intermediate": False,
                "native_counts": continuity,
                "footprint_representation": RepresentationKind.GAUSSIAN_CHIRP_ATOM.value,
                "input_support_kind": getattr(support.kind, "value", str(support.kind)),
            },
        )
        return NativeIntegralResult(value, report)

    def synthesize_hermitian(
        self,
        field: NativeField,
        *,
        output: np.ndarray | None = None,
    ) -> NativeProjectionResult:
        """Synthesize a real field from a validated Hermitian half spectrum."""

        if field.representation.kind is not RepresentationKind.HERMITIAN_HALF:
            raise TypeError("Hermitian synthesis requires a HERMITIAN_HALF representation")
        payload = field.representation.payload
        if not isinstance(payload, Mapping) or "half_spectrum" not in payload:
            raise TypeError("HERMITIAN_HALF representation has an invalid payload")
        from spf.renderer.normalization import complete_field_ledger
        from spf.renderer.ocean import HermitianHalfSpectrum

        shape = tuple(int(value) for value in payload["shape"])
        half = HermitianHalfSpectrum(payload["half_spectrum"], shape)
        synthesized = np.asarray(half.sparse(), dtype=np.float64)
        continuity = _field_continuity(field)
        continuity["hermitian_synthesis_count"] += 1
        continuity["typed_pivot_reduction_count"] += 1
        ledger = complete_field_ledger(field)
        metrics = RenderMetrics(
            backend="spf32_hermitian_sparse",
            workload="hermitian_half_synthesis",
            resolution=shape,
            representation=RepresentationKind.HERMITIAN_HALF.value,
            precision=str(self.precision.mode).lower(),
            bytes_read=int(np.asarray(payload["half_spectrum"]).nbytes),
            bytes_written=int(synthesized.nbytes),
            materialized_elements=int(synthesized.size),
            terminal_output_elements=int(synthesized.size),
            terminal_output_bytes=int(synthesized.nbytes),
            atoms_considered=0,
            fft_count=0,
            fallback_conversions=0,
            field_operations=continuity["symbolic_operation_count"],
            reduction_operations=1,
        )
        report = RenderReport(
            metrics,
            ledger,
            passes=(CompilerPassRecord("hermitian_pair_index", 1, 1), CompilerPassRecord("hermitian_synthesis", 1, 1)),
            capabilities_used=(
                "typed_native_input",
                "hermitian_pair_synthesis",
                "signed_real_terminal",
            ),
            diagnostics={
                "fft_forbidden": True,
                "cartesian_intermediate": False,
                "native_counts": continuity,
                "half_shape": half.half_shape,
                "full_shape": half.full_shape,
                "hermitian_nonzero_coefficients": int(np.count_nonzero(half.values)),
                "terminal_output_elements": int(synthesized.size),
            },
        )
        result = NativeProjectionResult(synthesized, report, np.empty(0, dtype=np.int32))
        return _store_projection_output(result, output)

    project_hermitian = synthesize_hermitian

    def project_magnitude(self, field: NativeField, grid: NativeGrid2D, **options: Any) -> NativeProjectionResult:
        return self.project(field, grid, semantic=OutputSemantic.MAGNITUDE, **options)

    def project_intensity(self, field: NativeField, grid: NativeGrid2D, **options: Any) -> NativeProjectionResult:
        return self.project(field, grid, semantic=OutputSemantic.INTENSITY, **options)

    def project_real(self, field: NativeField, grid: NativeGrid2D, **options: Any) -> NativeProjectionResult:
        return self.project(field, grid, semantic=OutputSemantic.REAL_FIELD, **options)

    def project_radiance(self, field: NativeField, grid: NativeGrid2D, **options: Any) -> NativeProjectionResult:
        return self.project(field, grid, semantic=OutputSemantic.RADIANCE, **options)

    def project_log_radiance(
        self,
        field: NativeField,
        grid: NativeGrid2D,
        **options: Any,
    ) -> NativeProjectionResult:
        return self.project(field, grid, semantic=OutputSemantic.LOG_RADIANCE, **options)

    def project_display(self, field: NativeField, grid: NativeGrid2D, **options: Any) -> NativeProjectionResult:
        return self.project(field, grid, semantic=OutputSemantic.DISPLAY_RGB, **options)


@dataclass(frozen=True, slots=True)
class CompiledNativeGaussianProjection:
    """Executable public vertical slice for one analytically closed product."""

    program: Any
    spatial_atom: v02.GaussianChirpAtom
    renderer: NativeRenderer
    source: NativeField
    transfer: NativeField
    grid: NativeGrid2D
    semantic: OutputSemantic
    support_epsilon: float = 0.0
    support: Support = dataclass_field(default_factory=Support.full)
    spatial_representation: FieldRepresentation | None = None
    _prepared_execution: PreparedNativeProjection | None = dataclass_field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        plans: list[Any] = []
        result = self._execute_unprepared(_preparation=plans)
        if len(plans) == 1:
            object.__setattr__(self, "_prepared_execution", PreparedNativeProjection(plans[0], result.report))

    def execute(self, *, output: np.ndarray | None = None) -> NativeProjectionResult:
        if self._prepared_execution is not None:
            return self._prepared_execution.execute(output=output)
        return self._execute_unprepared(output=output)

    def _execute_unprepared(
        self, *, output: np.ndarray | None = None, _preparation: list[Any] | None = None
    ) -> NativeProjectionResult:
        continuity = _merge_continuity(self.source, self.transfer)
        deferred_count = len(self.source.operations) + len(self.transfer.operations)
        continuity["symbolic_operation_count"] += 1 + deferred_count
        continuity["potential_dense_passes"] += 1 + deferred_count
        representation = self.spatial_representation or FieldRepresentation(
            RepresentationKind.GAUSSIAN_CHIRP_ATOM,
            self.spatial_atom,
            exactness=Exactness.EXACT,
        )
        representation_kind = RepresentationKind(representation.kind)
        if representation_kind not in {
            RepresentationKind.GAUSSIAN_CHIRP_ATOM,
            RepresentationKind.ATOM_MIXTURE,
            RepresentationKind.QUADRATIC_JET,
        }:
            raise NotImplementedError(
                "compiled native Gaussian projection cannot execute selected "
                f"representation {representation_kind.value!r} yet"
            )
        if representation_kind is RepresentationKind.GAUSSIAN_CHIRP_ATOM and not isinstance(
            representation.payload, v02.GaussianChirpAtom
        ):
            raise TypeError("selected native Gaussian representation has an invalid payload")
        spatial = NativeField(
            id="compiled_spatial_representation",
            domain=Domain.SPATIAL_2D,
            semantic=Semantic.COMPLEX_OPTICAL_AMPLITUDE,
            coherence=Coherence.COHERENT_COMPLEX,
            value_kind=FieldKind.SPF_COMPLEX,
            representation=representation,
            chart=CoordinateChart.identity(2),
            support=self.support,
            metadata={"native_continuity": continuity},
        )
        projected = self.renderer.project(
            spatial,
            self.grid,
            semantic=self.semantic,
            support_epsilon=self.support_epsilon,
            output=output,
            _preparation=_preparation,
        )
        compiler_report = self.program.report
        planner_decisions = (
            () if self.program.materialization_plan is None else tuple(self.program.materialization_plan.decisions)
        )
        diagnostics = dict(projected.report.diagnostics)
        diagnostics.update(
            {
                "compiler_backend": self.program.backend,
                "compiler_original_nodes": compiler_report.diagnostics["original_nodes"],
                "compiler_optimized_nodes": compiler_report.diagnostics["optimized_nodes"],
                "compiler_original_ops": tuple(
                    (node.op, node.params.get("source_operation")) for node in self.program.original_graph.nodes
                ),
                "compiler_optimized_ops": tuple(node.op for node in self.program.nodes),
                "executed_backend_ops": (
                    "selected_representation",
                    representation_kind.value,
                    f"project_{self.semantic.value}",
                ),
                "steady_state_backend_ops": ("selected_representation", f"project_{self.semantic.value}"),
                "planner_selected": tuple(decision.selected for decision in planner_decisions),
                "planner_representation_consumed": representation_kind.value,
            }
        )
        report = RenderReport(
            metrics=replace(
                projected.report.metrics,
                workload="compiled_native_gaussian_projection",
            ),
            ledger=projected.report.ledger,
            passes=(*compiler_report.passes, *projected.report.passes),
            decisions=(*planner_decisions, *projected.report.decisions),
            fallbacks=(*compiler_report.fallbacks, *projected.report.fallbacks),
            capabilities_used=(
                *compiler_report.capabilities_used,
                *projected.report.capabilities_used,
                "public_compiler_planner_backend",
            ),
            diagnostics=diagnostics,
        )
        return NativeProjectionResult(projected.values, report, projected.log_codes, projected.phase_codes)


class Legacy:
    """Explicit Cartesian interoperability boundary; never used implicitly."""

    @staticmethod
    def import_complex(values: Any, *, id: str = "legacy_complex") -> NativeField:
        array = np.asarray(values)
        if not np.issubdtype(array.dtype, np.complexfloating):
            raise TypeError("Legacy.import_complex requires a Cartesian complex array")
        return NativeField(
            id=id,
            domain=Domain.FOURIER_2D,
            semantic=Semantic.COMPLEX_OPTICAL_AMPLITUDE,
            coherence=Coherence.COHERENT_COMPLEX,
            value_kind=FieldKind.DENSE,
            representation=FieldRepresentation(RepresentationKind.DENSE_CARTESIAN_REFERENCE, array),
            chart=CoordinateChart.identity(array.ndim),
            metadata={"native_continuity": _continuity_metadata(legacy_imports=1)},
        )

    @staticmethod
    def import_rgba(values: Any, *, id: str = "legacy_rgba") -> NativeField:
        array = np.asarray(values)
        if array.ndim < 2 or array.shape[-1] not in {3, 4}:
            raise ValueError("Legacy.import_rgba requires an RGB or RGBA array")
        return NativeField(
            id=id,
            domain=Domain.SPATIAL_2D,
            semantic=Semantic.RADIANCE,
            coherence=Coherence.INCOHERENT_POSITIVE,
            value_kind=FieldKind.RADIANCE,
            representation=FieldRepresentation(RepresentationKind.DENSE_CARTESIAN_REFERENCE, array),
            chart=CoordinateChart.identity(array.ndim - 1),
            metadata={"native_continuity": _continuity_metadata(legacy_imports=1)},
        )


def assert_native_continuity(report: RenderReport) -> None:
    """Fail if a supposedly native report contains any Cartesian conversion."""

    counts = report.diagnostics.get("conversion_counts", {})
    required = (
        "cartesian_ingress_conversions",
        "cartesian_internal_conversions",
        "cartesian_egress_conversions",
        "legacy_adapter_calls",
    )
    missing = [name for name in required if name not in counts]
    if missing:
        raise AssertionError(f"native report is missing continuity counters: {missing}")
    violations = {name: int(counts[name]) for name in required if int(counts[name]) != 0}
    if violations:
        raise AssertionError(f"native continuity violated: {violations}")


__all__ = [
    "CompiledNativeGaussianProjection",
    "Legacy",
    "NativeField",
    "NativeGrid2D",
    "NativeIntegralResult",
    "NativeProjectionResult",
    "NativeRenderer",
    "OutputSemantic",
    "assert_native_continuity",
    "jet_forward_differences_vectorized",
]
