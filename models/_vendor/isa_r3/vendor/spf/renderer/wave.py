"""Explicit coherent wave-optics operations for SFR v0.2.

The wave path is deliberately separate from ordinary radiance.  Its inputs
are :class:`CoherentWaveField` values (or an explicitly constructed coherent
array/atom), and its outputs remain coherent until :func:`intensity` is
called.  Gaussian--chirp atoms stay analytic through propagation and lens
products whenever their Fourier integrals converge.  Arbitrary dense masks
and dense fields use a deterministic FFT fallback and report the transforms
and conversions it required.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.renderer.carriers import dft_frequencies, encode_words
from spf.renderer.model import (
    Coherence,
    CoordinateChart,
    Domain,
    ErrorContract,
    Field,
    FieldRepresentation,
    RepresentationKind,
    Semantic,
)
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport
from spf.renderer.v02 import FieldKind
from spf.renderer.wave_terminal import WaveCarrier
from spf.tensor import v02


class WaveDomain(StrEnum):
    SPATIAL = "spatial"
    FREQUENCY = "frequency"


class WaveRepresentation(StrEnum):
    DENSE = "dense"
    GAUSSIAN_CHIRP_ATOM = "gaussian_chirp_atom"


@dataclass(frozen=True, slots=True)
class WaveOperation:
    name: str
    params: Mapping[str, Any] = field(default_factory=dict)
    phase_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name).strip().lower().replace("-", "_"))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))
        object.__setattr__(self, "phase_only", bool(self.phase_only))


@dataclass(frozen=True, slots=True)
class CoherentWaveField:
    """A coherent complex field in either spatial or Fourier coordinates."""

    payload: Any
    domain: WaveDomain | str = WaveDomain.SPATIAL
    shape: tuple[int, int] | None = None
    sample_spacing: tuple[float, float] = (1.0, 1.0)
    origin: tuple[float, float] = (0.0, 0.0)
    representation: WaveRepresentation | str | None = None
    operations: tuple[WaveOperation, ...] = ()
    report: RenderReport | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        domain = WaveDomain(str(getattr(self.domain, "value", self.domain)).lower())
        object.__setattr__(self, "domain", domain)
        spacing = tuple(float(value) for value in self.sample_spacing)
        origin = tuple(float(value) for value in self.origin)
        if len(spacing) != 2 or any(value <= 0.0 or not math.isfinite(value) for value in spacing):
            raise ValueError("wave sample_spacing must be a positive finite two-vector")
        if len(origin) != 2 or any(not math.isfinite(value) for value in origin):
            raise ValueError("wave origin must be a finite two-vector")
        object.__setattr__(self, "sample_spacing", spacing)
        object.__setattr__(self, "origin", origin)
        if isinstance(self.payload, v02.GaussianChirpAtom):
            if self.payload.dimension != 2:
                raise ValueError("coherent wave atoms must be two-dimensional")
            selected_representation = WaveRepresentation.GAUSSIAN_CHIRP_ATOM
            selected_shape = (1, 1) if self.shape is None else tuple(int(value) for value in self.shape)
        else:
            carrier = WaveCarrier.own(self.payload, domain.value)
            if len(carrier.shape) != 2:
                raise ValueError("coherent waves require a two-dimensional carrier")
            selected_representation = WaveRepresentation.DENSE
            selected_shape = carrier.shape
            object.__setattr__(self, "payload", carrier)
        if len(selected_shape) != 2 or any(value <= 0 for value in selected_shape):
            raise ValueError("coherent wave shape must contain two positive dimensions")
        if self.shape is not None and tuple(int(value) for value in self.shape) != selected_shape:
            raise ValueError("wave shape does not match the payload")
        object.__setattr__(self, "shape", selected_shape)
        if self.representation is not None:
            requested = str(getattr(self.representation, "value", self.representation)).lower().replace("-", "_")
            if requested not in {item.value for item in WaveRepresentation}:
                raise ValueError(f"unknown wave representation {self.representation!r}")
            selected_representation = WaveRepresentation(requested)
        object.__setattr__(self, "representation", selected_representation)
        object.__setattr__(self, "operations", tuple(self.operations))
        if self.report is not None and not isinstance(self.report, RenderReport):
            raise TypeError("wave report must be a RenderReport")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_array(
        cls,
        values: Any,
        *,
        domain: WaveDomain | str = WaveDomain.SPATIAL,
        sample_spacing: Sequence[float] = (1.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0),
        metadata: Mapping[str, Any] | None = None,
    ) -> CoherentWaveField:
        spacing = tuple(float(value) for value in sample_spacing)
        offset = tuple(float(value) for value in origin)
        if len(spacing) != 2 or len(offset) != 2:
            raise ValueError("wave sample_spacing and origin must be two-vectors")
        return cls(
            values,
            domain,
            None,
            (spacing[0], spacing[1]),
            (offset[0], offset[1]),
            None,
            (),
            None,
            metadata or {},
        )

    @classmethod
    def from_atom(
        cls,
        atom: v02.GaussianChirpAtom,
        shape: Sequence[int] = (1, 1),
        *,
        domain: WaveDomain | str = WaveDomain.SPATIAL,
        sample_spacing: Sequence[float] = (1.0, 1.0),
        origin: Sequence[float] = (0.0, 0.0),
        metadata: Mapping[str, Any] | None = None,
    ) -> CoherentWaveField:
        dimensions = tuple(int(value) for value in shape)
        spacing = tuple(float(value) for value in sample_spacing)
        offset = tuple(float(value) for value in origin)
        if len(dimensions) != 2 or len(spacing) != 2 or len(offset) != 2:
            raise ValueError("wave shape, sample_spacing, and origin must be two-vectors")
        return cls(
            atom,
            domain,
            (dimensions[0], dimensions[1]),
            (spacing[0], spacing[1]),
            (offset[0], offset[1]),
            WaveRepresentation.GAUSSIAN_CHIRP_ATOM,
            (),
            None,
            metadata or {},
        )

    @property
    def values(self) -> Any:
        return self.payload if self.is_analytic else self.materialize()

    @property
    def data(self) -> Any:
        return self.values

    @property
    def atom(self) -> v02.GaussianChirpAtom | None:
        return self.payload if self.representation is WaveRepresentation.GAUSSIAN_CHIRP_ATOM else None

    @property
    def dimensions(self) -> tuple[int, int]:
        return cast(tuple[int, int], self.shape)

    @property
    def wave_domain(self) -> WaveDomain:
        return cast(WaveDomain, self.domain)

    @property
    def representation_kind(self) -> WaveRepresentation:
        return cast(WaveRepresentation, self.representation)

    @property
    def coherence(self) -> Coherence:
        return Coherence.COHERENT_COMPLEX

    @property
    def is_analytic(self) -> bool:
        return self.representation_kind is WaveRepresentation.GAUSSIAN_CHIRP_ATOM

    def materialize(self) -> np.ndarray:
        if self.atom is None:
            return np.asarray(self.payload.materialize(self.wave_domain.value))
        points = _grid_points(self.dimensions, self.origin, self.sample_spacing)
        return np.asarray(self.atom.evaluate(points), dtype=np.complex128).reshape(self.dimensions)

    to_dense = materialize

    def spectrum(self) -> np.ndarray:
        if isinstance(self.payload, WaveCarrier):
            return np.asarray(self.payload.materialize("frequency"))
        values = self.materialize()
        return (
            values if self.wave_domain is WaveDomain.FREQUENCY else np.asarray(np.fft.fftn(values), dtype=np.complex128)
        )

    def to_field(self, *, field_id: str = "coherent_wave") -> Field:
        from spf.renderer.carriers import as_block_view

        block = as_block_view(self.payload.source) if isinstance(self.payload, WaveCarrier) else None
        if self.is_analytic:
            kind, payload = RepresentationKind.GAUSSIAN_CHIRP_ATOM, self.payload
        elif block is not None and self.payload.is_identity:
            # A block-backed wave stays a block: the field consumes it by
            # address (rule 1) instead of materializing a Cartesian plane.
            kind, payload = RepresentationKind.DENSE_TYPED, block
        else:
            kind, payload = RepresentationKind.DENSE_CARTESIAN_REFERENCE, self.materialize()
        representation = FieldRepresentation(
            kind,
            payload,
            exactness="EXACT",
            metadata={"shape": self.dimensions, "domain": self.wave_domain.value},
        )
        return Field(
            id=field_id,
            domain=Domain.SPATIAL_2D if self.wave_domain is WaveDomain.SPATIAL else Domain.FOURIER_2D,
            semantic=Semantic.COMPLEX_OPTICAL_AMPLITUDE,
            coherence=Coherence.COHERENT_COMPLEX,
            value_kind=FieldKind.SPF_COMPLEX,
            representation=representation,
            chart=CoordinateChart.identity(2, spacing=self.sample_spacing),
            error=ErrorContract(absolute_max=0.0, relative_max=0.0, phase_max=0.0, phase_required=True),
            metadata={"shape": self.dimensions},
        )

    def with_report(self, report: RenderReport, *operation: WaveOperation) -> CoherentWaveField:
        return replace(self, operations=(*self.operations, *operation), report=report)

    def intensity(self, *, output: str = "linear", return_report: bool = False) -> Any:
        return intensity(self, output=output, return_report=return_report)


WaveField = CoherentWaveField
CoherentField = CoherentWaveField


def _grid_points(
    shape: Sequence[int],
    origin: Sequence[float] = (0.0, 0.0),
    spacing: Sequence[float] = (1.0, 1.0),
) -> np.ndarray:
    height, width = (int(value) for value in shape)
    dx, dy = (float(value) for value in spacing)
    x0, y0 = (float(value) for value in origin)
    y, x = np.meshgrid(
        y0 + dy * np.arange(height, dtype=np.float64),
        x0 + dx * np.arange(width, dtype=np.float64),
        indexing="ij",
    )
    return np.stack((x, y), axis=-1)


def coherent(
    value: Any,
    *,
    domain: WaveDomain | str = WaveDomain.SPATIAL,
    sample_spacing: Sequence[float] = (1.0, 1.0),
    origin: Sequence[float] = (0.0, 0.0),
    shape: Sequence[int] | None = None,
) -> CoherentWaveField:
    """Perform the explicit edge into the coherent complex algebra."""

    if isinstance(value, Field):
        if value.coherence is not Coherence.COHERENT_COMPLEX:
            raise TypeError("ordinary/incoherent fields require an explicit physical coherence model")
        from spf.renderer.carriers import field_carrier

        value = field_carrier(value)
    if isinstance(value, v02.GaussianChirpAtom):
        return CoherentWaveField.from_atom(
            value, shape or (1, 1), domain=domain, sample_spacing=sample_spacing, origin=origin
        )
    return CoherentWaveField.from_array(value, domain=domain, sample_spacing=sample_spacing, origin=origin)


def _coerce_wave(value: Any) -> CoherentWaveField:
    if isinstance(value, CoherentWaveField):
        return value
    if isinstance(value, Field):
        if value.coherence is not Coherence.COHERENT_COMPLEX:
            raise TypeError("wave optics requires a COHERENT_COMPLEX field")
        from spf.renderer.carriers import field_carrier

        payload = field_carrier(value)
        domain = (
            WaveDomain.FREQUENCY
            if value.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}
            else WaveDomain.SPATIAL
        )
        shape = value.metadata.get("shape") if isinstance(value.metadata, Mapping) else None
        if isinstance(payload, v02.GaussianChirpAtom):
            if shape is None:
                raise ValueError("an atom Field requires metadata['shape'] for wave materialization")
            return CoherentWaveField.from_atom(
                payload,
                shape,
                domain=domain,
                sample_spacing=value.chart.sample_spacing or (1.0, 1.0),
                origin=value.chart.origin or (0.0, 0.0),
            )
        if value.chart.dimension and (
            value.chart.basis != CoordinateChart.identity(2).basis or value.chart.scale != (1.0, 1.0)
        ):
            from spf.renderer.carriers import carrier_shape
            from spf.renderer.model import MaterializationRequest, SampleLayout

            selected_shape = shape or carrier_shape(payload)
            payload = value.materialize(
                MaterializationRequest(layout=SampleLayout(selected_shape), output_format="complex")
            )
        return CoherentWaveField.from_array(
            payload,
            domain=domain,
            sample_spacing=value.chart.sample_spacing or (1.0, 1.0),
            origin=value.chart.origin or (0.0, 0.0),
        )
    if isinstance(value, v02.GaussianChirpAtom):
        return CoherentWaveField.from_atom(value)
    return CoherentWaveField.from_array(value)


def _frequency_grid(shape: tuple[int, int], spacing: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    fy = dft_frequencies(shape[0], spacing[1])
    fx = dft_frequencies(shape[1], spacing[0])
    return np.meshgrid(fy, fx, indexing="ij")


def _report(
    wave: CoherentWaveField,
    operation: str,
    *,
    analytic: bool,
    fft_count: int = 0,
    materialized: int = 0,
    fallback_conversions: int = 0,
    diagnostics: Mapping[str, Any] | None = None,
) -> RenderReport:
    notes = {
        "operation": operation,
        "analytic_atom_closure": analytic,
        "domain": wave.wave_domain.value,
        "coherence": Coherence.COHERENT_COMPLEX.value,
        "deferred_domain_conversion": not analytic,
        "phase_storage": "axis_factors" if not analytic else "atom_header",
    }
    if diagnostics:
        notes.update(diagnostics)
    metrics = RenderMetrics(
        backend="reference",
        workload="wave_optics",
        resolution=wave.dimensions,
        representation=wave.representation_kind.value,
        precision="complex128" if analytic else "dense_reference",
        potential_elements=int(np.prod(wave.dimensions)),
        materialized_elements=int(materialized),
        fft_count=int(fft_count),
        fallback_conversions=int(fallback_conversions),
        bytes_read=int(materialized * (16 if not analytic else 0)),
        bytes_written=int(materialized * (16 if not analytic else 0)),
        field_operations=1,
    )
    return RenderReport(
        metrics,
        v02.ErrorLedger(),
        passes=(CompilerPassRecord(operation, 1, 1, rewrites=("atom closure" if analytic else "dense FFT fallback",)),),
        capabilities_used=("COHERENT_COMPLEX", "GAUSSIAN_CHIRP" if analytic else "DENSE_FFT"),
        diagnostics=notes,
    )


def _propagation_atom(distance: float, wavelength: float) -> v02.GaussianChirpAtom:
    return v02.GaussianChirpAtom.from_arrays(
        1j * 2.0 * math.pi * distance / wavelength,
        np.zeros(2, dtype=np.complex128),
        1j * 2.0 * math.pi * wavelength * distance * np.eye(2, dtype=np.complex128),
    )


def _lens_atom(focal_length: float, wavelength: float) -> v02.GaussianChirpAtom:
    return v02.GaussianChirpAtom.from_arrays(
        0.0j,
        np.zeros(2, dtype=np.complex128),
        1j * 2.0 * math.pi / (wavelength * focal_length) * np.eye(2, dtype=np.complex128),
    )


def _check_wave_parameters(distance: float, wavelength: float) -> tuple[float, float]:
    z, wave = float(distance), float(wavelength)
    if not math.isfinite(z) or z == 0.0 or not math.isfinite(wave) or wave <= 0.0:
        raise ValueError("propagation requires finite nonzero distance and positive wavelength")
    return z, wave


def propagate(
    field: Any,
    distance: float,
    wavelength: float,
    *,
    return_report: bool = False,
) -> Any:
    """Apply the Fresnel transfer function to a coherent field."""

    wave = _coerce_wave(field)
    distance, wavelength = _check_wave_parameters(distance, wavelength)
    if wave.atom is not None:
        transfer = _propagation_atom(distance, wavelength)
        if wave.wave_domain is WaveDomain.FREQUENCY:
            atom = v02.atom_mul(wave.atom, transfer)
        else:
            frequency_atom = v02.atom_fourier(wave.atom)
            atom = v02.atom_inverse_fourier(v02.atom_mul(frequency_atom, transfer))
        result = CoherentWaveField.from_atom(
            atom,
            wave.dimensions,
            domain=wave.wave_domain,
            sample_spacing=wave.sample_spacing,
            origin=wave.origin,
            metadata=wave.metadata,
        )
        report = _report(wave, "propagate", analytic=True, diagnostics={"distance": distance, "wavelength": wavelength})
    else:
        carrier = wave.payload.to_domain("frequency")
        fy = dft_frequencies(wave.dimensions[0], wave.sample_spacing[1])
        fx = dft_frequencies(wave.dimensions[1], wave.sample_spacing[0])
        coefficient = -math.pi * wavelength * distance
        carrier = carrier.phase_shift(coefficient * fy * fy, coefficient * fx * fx, 2 * math.pi * distance / wavelength)
        fft_count = int(wave.payload.domain != "frequency")
        result = CoherentWaveField.from_array(
            carrier,
            domain=wave.wave_domain,
            sample_spacing=wave.sample_spacing,
            origin=wave.origin,
            metadata=wave.metadata,
        )
        report = _report(
            wave,
            "propagate",
            analytic=False,
            fft_count=fft_count,
            materialized=int(np.prod(wave.dimensions)),
            fallback_conversions=0,
            diagnostics={"distance": distance, "wavelength": wavelength},
        )
    result = result.with_report(
        report, WaveOperation("propagate", {"distance": distance, "wavelength": wavelength}, True)
    )
    return (result, report) if return_report else result


def _coerce_lens_parameters(focal_length: float, wavelength: float) -> tuple[float, float]:
    focal, wave = float(focal_length), float(wavelength)
    if not math.isfinite(focal) or focal == 0.0 or not math.isfinite(wave) or wave <= 0.0:
        raise ValueError("lens requires finite nonzero focal_length and positive wavelength")
    return focal, wave


def lens(
    field: Any,
    focal_length: float,
    wavelength: float,
    *,
    return_report: bool = False,
) -> Any:
    """Apply the thin-lens quadratic phase, preserving atom closure when possible."""

    wave = _coerce_wave(field)
    focal_length, wavelength = _coerce_lens_parameters(focal_length, wavelength)
    if wave.atom is not None:
        lens_factor = _lens_atom(focal_length, wavelength)
        if wave.wave_domain is WaveDomain.SPATIAL:
            atom = v02.atom_mul(wave.atom, lens_factor)
        else:
            spatial_atom = v02.atom_inverse_fourier(wave.atom)
            spatial_atom = v02.atom_mul(spatial_atom, lens_factor)
            atom = v02.atom_fourier(spatial_atom)
        result = CoherentWaveField.from_atom(
            atom,
            wave.dimensions,
            domain=wave.wave_domain,
            sample_spacing=wave.sample_spacing,
            origin=wave.origin,
            metadata=wave.metadata,
        )
        report = _report(
            wave, "lens", analytic=True, diagnostics={"focal_length": focal_length, "wavelength": wavelength}
        )
    else:
        carrier = wave.payload.to_domain("spatial")
        x = wave.origin[0] + wave.sample_spacing[0] * np.arange(wave.dimensions[1])
        y = wave.origin[1] + wave.sample_spacing[1] * np.arange(wave.dimensions[0])
        coefficient = -math.pi / (wavelength * focal_length)
        carrier = carrier.phase_shift(coefficient * y * y, coefficient * x * x)
        fft_count = int(wave.payload.domain != "spatial")
        result = CoherentWaveField.from_array(
            carrier,
            domain=wave.wave_domain,
            sample_spacing=wave.sample_spacing,
            origin=wave.origin,
            metadata=wave.metadata,
        )
        report = _report(
            wave,
            "lens",
            analytic=False,
            fft_count=fft_count,
            materialized=int(np.prod(wave.dimensions)),
            diagnostics={"focal_length": focal_length, "wavelength": wavelength},
        )
    result = result.with_report(
        report, WaveOperation("lens", {"focal_length": focal_length, "wavelength": wavelength}, True)
    )
    return (result, report) if return_report else result


def aperture(
    field: Any,
    mask: Any,
    *,
    return_report: bool = False,
) -> Any:
    """Multiply a coherent field by a spatial aperture mask."""

    wave = _coerce_wave(field)
    if isinstance(mask, v02.GaussianChirpAtom):
        if mask.dimension != 2:
            raise ValueError("aperture atom must be two-dimensional")
        if wave.atom is not None:
            if wave.wave_domain is WaveDomain.SPATIAL:
                atom = v02.atom_mul(wave.atom, mask)
            else:
                spatial_atom = v02.atom_inverse_fourier(wave.atom)
                atom = v02.atom_fourier(v02.atom_mul(spatial_atom, mask))
            result = CoherentWaveField.from_atom(
                atom,
                wave.dimensions,
                domain=wave.wave_domain,
                sample_spacing=wave.sample_spacing,
                origin=wave.origin,
                metadata=wave.metadata,
            )
            report = _report(wave, "aperture", analytic=True)
            result = result.with_report(report, WaveOperation("aperture", {"analytic": True}, False))
            return (result, report) if return_report else result
        mask_values = np.asarray(
            mask.evaluate(_grid_points(wave.dimensions, wave.origin, wave.sample_spacing)), dtype=np.complex128
        ).reshape(wave.dimensions)
    else:
        if callable(mask):
            mask_values = np.asarray(
                mask(_grid_points(wave.dimensions, wave.origin, wave.sample_spacing)), dtype=np.complex128
            )
        else:
            mask_values = np.asarray(mask)
        if mask_values.ndim == 0:
            mask_values = np.full(wave.dimensions, complex(mask_values), dtype=np.complex128)
        try:
            mask_values = np.broadcast_to(mask_values, wave.dimensions).astype(np.complex128)
        except ValueError as exc:
            raise ValueError("aperture mask cannot broadcast to the wave shape") from exc
    carrier = (
        WaveCarrier.own(wave.materialize(), wave.wave_domain.value) if wave.atom is not None else wave.payload
    ).to_domain("spatial")
    carrier = carrier.multiply(mask_values)
    fft_count = int(wave.wave_domain is WaveDomain.FREQUENCY)
    result = CoherentWaveField.from_array(
        carrier,
        domain=wave.wave_domain,
        sample_spacing=wave.sample_spacing,
        origin=wave.origin,
        metadata=wave.metadata,
    )
    report = _report(
        wave,
        "aperture",
        analytic=False,
        fft_count=fft_count,
        materialized=int(np.prod(wave.dimensions)),
        fallback_conversions=1 if wave.atom is not None else 0,
        diagnostics={"analytic_mask": False},
    )
    result = result.with_report(report, WaveOperation("aperture", {"analytic": False}, False))
    return (result, report) if return_report else result


def _log_pos_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        output[index] = v02.LogPos.from_value(float(array[index]))
    return output


def intensity(
    field: Any,
    *,
    output: str = "linear",
    return_report: bool = False,
) -> Any:
    """Demote a coherent field through magnitude-square.

    The phase plane is intentionally not evaluated for this terminal
    consumer.  The report records that pure phase was eliminated; callers
    that need interference must remain in :class:`CoherentWaveField`.
    """

    wave = _coerce_wave(field)
    if wave.atom is not None:
        atom = wave.atom
        intensity_atom = v02.GaussianChirpAtom.from_arrays(
            2.0 * atom.kappa.real,
            2.0 * np.real(atom.h),
            2.0 * np.real(atom.C),
        )
        if str(output).lower().replace("-", "_") in {"atom", "gaussian_chirp"}:
            report = _report(wave, "intensity", analytic=True, diagnostics={"terminal_phase_eliminated": True})
            return (intensity_atom, report) if return_report else intensity_atom
        evaluated = np.asarray(
            intensity_atom.evaluate(_grid_points(wave.dimensions, wave.origin, wave.sample_spacing)),
            dtype=np.complex128,
        )
        if np.any(np.abs(evaluated.imag) > 1.0e-12):
            raise ValueError("analytic intensity closure produced a non-real field")
        values = evaluated.real.reshape(wave.dimensions)
        analytic = True
    else:
        values = wave.payload.intensity(wave.wave_domain.value)
        intensity_atom = None
        analytic = False
    name = str(output).lower().replace("-", "_")
    result: Any
    if name in {"linear", "dense", "float", "float64", "native"}:
        result = np.asarray(values, dtype=np.float64)
    elif name in {"log", "log_pos", "logpos", "positive"}:
        result = _log_pos_array(values)
    elif name in {"atom", "gaussian_chirp"}:
        if intensity_atom is None:
            raise ValueError("dense intensity has no analytic atom representation")
        result = intensity_atom
    elif name in {"spf", "spf32"}:
        result = np.asarray(encode_words(values), dtype=np.uint32)
    else:
        raise ValueError(f"unsupported intensity output {output!r}")
    report = _report(
        wave,
        "intensity",
        analytic=analytic,
        materialized=0 if analytic else int(np.prod(wave.dimensions)),
        diagnostics={
            "terminal_phase_eliminated": analytic or wave.payload.domain == wave.wave_domain.value,
            "phase_observable": not analytic and wave.payload.domain != wave.wave_domain.value,
            "output_kind": name,
        },
    )
    if return_report:
        return result, report
    return result


def coherent_propagate(field: Any, distance: float, wavelength: float, **options: Any) -> Any:
    return propagate(field, distance, wavelength, **options)


def thin_lens(field: Any, focal_length: float, wavelength: float, **options: Any) -> Any:
    return lens(field, focal_length, wavelength, **options)


__all__ = [
    "CoherentField",
    "CoherentWaveField",
    "WaveDomain",
    "WaveField",
    "WaveOperation",
    "WaveRepresentation",
    "aperture",
    "coherent",
    "coherent_propagate",
    "intensity",
    "lens",
    "propagate",
    "thin_lens",
]
