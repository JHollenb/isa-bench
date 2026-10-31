"""Deterministic benchmark and quality-report harness for SFR v0.2.

The design requires every performance row to carry the representation,
materialization, reduction, quality, and fallback evidence alongside timing.
This module deliberately keeps timing orchestration small: workload functions
return their normal renderer value/report objects and the harness normalizes
them into the publication schema without inventing unavailable energy or LPIPS
measurements.
"""

from __future__ import annotations

import csv
import io
import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.report import RenderMetrics, RenderReport, psnr, relative_l2, ssim_global

REQUIRED_RESULT_COLUMNS = (
    "backend",
    "workload",
    "resolution",
    "representation",
    "precision",
    "error_budget",
    "runtime_ms",
    "energy_j",
    "bytes_read",
    "bytes_written",
    "peak_memory",
    "materialized_elements",
    "atoms_considered",
    "atoms_pruned",
    "positive_logadds",
    "complex_adds",
    "pivot_reductions",
    "transport_compositions",
    "fft_count",
    "fallback_conversions",
    "psnr",
    "ssim",
    "lpips",
    "temporal_error",
    # Added 2026-09-17 (blog Tier 3 item 12): DRAM traffic as the path accounted it
    # and average power when a sampler measured it. Both stay null when absent.
    "dram_traffic_bytes",
    "power_w",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """One independently timed renderer workload."""

    name: str
    workload: str
    runner: Callable[[], Any]
    backend: str = "reference"
    representation: str = "unknown"
    precision: str = "balanced"
    error_budget: float = math.inf
    reference: Any | Callable[[], Any] | None = None
    warmups: int = 1
    repeats: int = 3
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.name).strip() or not str(self.workload).strip():
            raise ValueError("benchmark name and workload must be non-empty")
        if not callable(self.runner):
            raise TypeError("benchmark runner must be callable")
        if int(self.warmups) < 0 or int(self.repeats) <= 0:
            raise ValueError("benchmark warmups must be non-negative and repeats positive")
        budget = float(self.error_budget)
        if math.isnan(budget) or budget < 0.0:
            raise ValueError("benchmark error_budget must be non-negative")
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "workload", str(self.workload))
        object.__setattr__(self, "backend", str(self.backend))
        object.__setattr__(self, "representation", str(self.representation))
        object.__setattr__(self, "precision", str(self.precision))
        object.__setattr__(self, "error_budget", budget)
        object.__setattr__(self, "warmups", int(self.warmups))
        object.__setattr__(self, "repeats", int(self.repeats))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class BenchmarkRow:
    """The required v0.2 result columns plus explanatory diagnostics."""

    name: str
    backend: str
    workload: str
    resolution: tuple[int, ...]
    representation: str
    precision: str
    error_budget: float
    runtime_ms: float
    energy_j: float | None
    bytes_read: int
    bytes_written: int
    peak_memory: int
    materialized_elements: int
    atoms_considered: int
    atoms_pruned: int
    positive_logadds: int
    complex_adds: int
    pivot_reductions: int
    transport_compositions: int
    fft_count: int
    fallback_conversions: int
    psnr: float | None
    ssim: float | None
    lpips: float | None
    temporal_error: float | None
    relative_l2: float | None = None
    potential_elements: int = 0
    field_operations: int = 0
    reduction_operations: int = 0
    fallbacks: tuple[str, ...] = ()
    samples_ms: tuple[float, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    power_w: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolution", tuple(int(value) for value in self.resolution))
        object.__setattr__(self, "fallbacks", tuple(str(value) for value in self.fallbacks))
        object.__setattr__(self, "samples_ms", tuple(float(value) for value in self.samples_ms))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def materialization_avoidance(self) -> float:
        if self.materialized_elements == 0:
            return math.inf if self.potential_elements else 1.0
        return self.potential_elements / self.materialized_elements

    @property
    def reduction_density(self) -> float:
        return 0.0 if self.field_operations == 0 else self.reduction_operations / self.field_operations

    @property
    def dram_traffic_bytes(self) -> int:
        return int(self.bytes_read) + int(self.bytes_written)

    def to_dict(self) -> dict[str, Any]:
        value = {name: getattr(self, name) for name in self.__dataclass_fields__}
        value["dram_traffic_bytes"] = self.dram_traffic_bytes
        value["resolution"] = list(self.resolution)
        value["fallbacks"] = list(self.fallbacks)
        value["samples_ms"] = list(self.samples_ms)
        value["metadata"] = dict(self.metadata)
        value["materialization_avoidance"] = self.materialization_avoidance
        value["reduction_density"] = self.reduction_density
        return value


@dataclass(frozen=True, slots=True)
class BenchmarkSuite:
    rows: tuple[BenchmarkRow, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": list(REQUIRED_RESULT_COLUMNS),
            "metadata": dict(self.metadata),
            "rows": [row.to_dict() for row in self.rows],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(_json_safe(self.to_dict()), indent=indent, sort_keys=True, allow_nan=False)

    def to_csv(self) -> str:
        records = [row.to_dict() for row in self.rows]
        if not records:
            return ""
        stream = io.StringIO()
        columns = list(records[0])
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in records:
            encoded = {
                key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value
                for key, value in record.items()
            }
            writer.writerow(encoded)
        return stream.getvalue()

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json() + "\n", encoding="utf-8")

    def save_csv(self, path: str | Path) -> None:
        Path(path).write_text(self.to_csv(), encoding="utf-8")


def _split_result(result: Any) -> tuple[Any, RenderReport | None]:
    if isinstance(result, np.ndarray):
        return result, None
    if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], RenderReport):
        return result[0], result[1]
    if isinstance(result, RenderReport):
        return None, result
    report = getattr(result, "report", None)
    if not isinstance(report, RenderReport):
        report = None
    for name in ("values", "value", "image", "color", "data"):
        if hasattr(result, name):
            return getattr(result, name), report
    return result, report


def _reference_value(value: Any | Callable[[], Any] | None) -> Any:
    if value is None:
        return None
    result = value() if callable(value) else value
    return _split_result(result)[0]


def _inferred_resolution(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[-1] in {1, 3, 4}:
        return (int(array.shape[1]), int(array.shape[0]))
    return tuple(int(item) for item in array.shape)


def _quality(value: Any, reference: Any) -> tuple[float | None, float | None, float | None]:
    if value is None or reference is None:
        return None, None, None
    actual = np.asarray(value)
    expected = np.asarray(reference)
    if actual.shape != expected.shape:
        raise ValueError(f"benchmark output shape {actual.shape} does not match reference {expected.shape}")
    l2 = relative_l2(actual, expected)
    if np.iscomplexobj(actual) or np.iscomplexobj(expected):
        actual_quality = np.abs(actual)
        expected_quality = np.abs(expected)
    else:
        actual_quality = actual
        expected_quality = expected
    expected_numeric = np.asarray(expected_quality, dtype=np.float64)
    dynamic = float(np.ptp(expected_numeric))
    if dynamic <= 0.0 or not math.isfinite(dynamic):
        dynamic = float(np.max(np.abs(expected_numeric))) if expected_numeric.size else 0.0
    if dynamic <= 0.0 or not math.isfinite(dynamic):
        dynamic = 1.0
    return (
        l2,
        psnr(actual_quality, expected_quality, data_range=dynamic),
        ssim_global(actual_quality, expected_quality, data_range=dynamic),
    )


def run_benchmark(case: BenchmarkCase) -> BenchmarkRow:
    """Run one case and return a complete, fixed-schema result row."""

    if not isinstance(case, BenchmarkCase):
        raise TypeError("run_benchmark expects BenchmarkCase")
    for _ in range(case.warmups):
        case.runner()
    timings: list[float] = []
    last: Any = None
    for _ in range(case.repeats):
        started = time.perf_counter_ns()
        last = case.runner()
        timings.append((time.perf_counter_ns() - started) / 1.0e6)
    payload, report = _split_result(last)
    reference = _reference_value(case.reference)
    l2, measured_psnr, measured_ssim = _quality(payload, reference)
    metrics = report.metrics if report is not None else RenderMetrics()
    resolution = metrics.resolution or _inferred_resolution(payload)
    return BenchmarkRow(
        name=case.name,
        backend=metrics.backend if report is not None else case.backend,
        workload=metrics.workload if report is not None and metrics.workload != "generic" else case.workload,
        resolution=resolution,
        representation=(
            metrics.representation
            if report is not None and metrics.representation != "unknown"
            else case.representation
        ),
        precision=metrics.precision if report is not None else case.precision,
        error_budget=metrics.error_budget if report is not None else case.error_budget,
        runtime_ms=float(statistics.median(timings)),
        energy_j=metrics.energy_j,
        power_w=metrics.power_w,
        bytes_read=int(metrics.bytes_read),
        bytes_written=int(metrics.bytes_written),
        peak_memory=int(metrics.peak_memory),
        materialized_elements=int(metrics.materialized_elements),
        atoms_considered=int(metrics.atoms_considered),
        atoms_pruned=int(metrics.atoms_pruned),
        positive_logadds=int(metrics.positive_logadds),
        complex_adds=int(metrics.complex_adds),
        pivot_reductions=int(metrics.pivot_reductions),
        transport_compositions=int(metrics.transport_compositions),
        fft_count=int(metrics.fft_count),
        fallback_conversions=int(metrics.fallback_conversions),
        psnr=measured_psnr if measured_psnr is not None else metrics.psnr,
        ssim=measured_ssim if measured_ssim is not None else metrics.ssim,
        lpips=metrics.lpips,
        temporal_error=metrics.temporal_error,
        relative_l2=l2 if l2 is not None else metrics.relative_l2,
        potential_elements=int(metrics.potential_elements),
        field_operations=int(metrics.field_operations),
        reduction_operations=int(metrics.reduction_operations),
        fallbacks=() if report is None else report.fallbacks,
        samples_ms=tuple(timings),
        metadata={**dict(case.metadata), "warmups": case.warmups, "repeats": case.repeats},
    )


def run_benchmarks(cases: Iterable[BenchmarkCase], *, metadata: Mapping[str, Any] | None = None) -> BenchmarkSuite:
    return BenchmarkSuite(tuple(run_benchmark(case) for case in cases), metadata or {})


def benchmark_filter_chain(chain: Any, *, warmups: int = 1, repeats: int = 3) -> BenchmarkSuite:
    """Run the v0.2 four-rung filter-chain baseline ladder."""

    from spf.renderer.filter_chain import FilterVariant

    reference = chain.run(FilterVariant.EAGER_FP32).values
    cases = tuple(
        BenchmarkCase(
            name=f"filter_chain/{variant.value}",
            workload="filter_chain",
            runner=partial(chain.run, variant),
            representation=variant.value,
            reference=reference,
            warmups=warmups,
            repeats=repeats,
            metadata={"baseline_rung": index + 1},
        )
        for index, variant in enumerate(FilterVariant)
    )
    return run_benchmarks(cases, metadata={"baseline_ladder": [item.value for item in FilterVariant]})


def _json_safe(value: Any) -> Any:
    """Convert non-finite diagnostics to explicit strings for strict JSON."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def benchmark_renderer_suite(
    *,
    size: int = 16,
    warmups: int = 0,
    repeats: int = 1,
) -> BenchmarkSuite:
    """Run the portable v0.2 workload matrix at a deterministic small size.

    The suite covers the four-rung filter baseline plus splats, ocean,
    volume, texture LOD, and coherent wave optics.  It is an engineering
    smoke suite, not a claim against mature external renderers; unavailable
    energy/LPIPS fields remain null and the high-entropy texture row is marked
    as an expected losing case.
    """

    count = int(size)
    if count < 8:
        raise ValueError("renderer benchmark size must be at least 8")
    if int(warmups) < 0 or int(repeats) <= 0:
        raise ValueError("benchmark warmups must be non-negative and repeats positive")

    from spf.renderer.assets import SpectralTexture
    from spf.renderer.filter_chain import FilterChain
    from spf.renderer.ocean import build_half_spectrum, synthesize_ocean
    from spf.renderer.report import CompilerPassRecord
    from spf.renderer.splats import Camera, GaussianSplat, rasterize_splats_reference, rasterize_splats_spf
    from spf.renderer.volume import constant_density_segment, render_volume
    from spf.renderer.wave import CoherentWaveField, propagate

    rng = np.random.default_rng(202)
    source = rng.normal(size=(count, count))
    filter_chain = (
        FilterChain.from_source(source)
        .translate(dx=0.37, dy=-0.22)
        .gaussian_filter(np.asarray(((0.3, 0.04), (0.04, 0.16))))
        .exposure(0.5)
    )
    filter_suite = benchmark_filter_chain(filter_chain, warmups=warmups, repeats=repeats)

    camera = Camera.orthographic(
        count,
        count,
        scale=float(count),
        position=(0.0, 0.0, 0.0),
        target=(0.0, 0.0, 1.0),
    )
    splats = (
        GaussianSplat((-1.2, 0.4, 1.0), np.diag((0.8, 0.5, 0.2)), 0.75, (0.9, 0.2, 0.1), 1),
        GaussianSplat((1.0, -0.2, 1.8), np.diag((0.6, 0.9, 0.2)), 0.55, (0.1, 0.5, 0.9), 2),
    )
    splat_reference = rasterize_splats_reference(splats, camera, tile_size=8).color

    shape = (count, count)
    ocean_half = build_half_spectrum(shape, ((0, 0), (1, 2), (3, 1)), (0.2, 0.7 + 0.1j, -0.2 + 0.4j))
    ocean_reference = synthesize_ocean(shape, half_spectrum=ocean_half, method="irfft")

    segments = tuple(
        constant_density_segment(
            0.08 + 0.015 * index,
            0.2,
            (0.05 + 0.01 * index, 0.025, 0.012),
        )
        for index in range(8)
    )
    volume_reference = render_volume(segments, background=(0.01, 0.01, 0.015))

    texture_source = rng.random(shape)
    texture = SpectralTexture.from_array(texture_source, block_shape=(8, 8), error_budget=2.0e-2)
    rows, columns = np.indices(shape, dtype=np.float64)
    coordinates = np.stack((columns / (count - 1), rows / (count - 1)), axis=-1)
    texture_reference = np.fft.ifftn(np.fft.fftn(texture_source) * texture.lod_window(0.35)).real

    def run_texture_lod() -> tuple[np.ndarray, RenderReport]:
        values = np.asarray(texture.sample(coordinates, lod_cutoff=0.35))
        metrics = RenderMetrics(
            backend="reference",
            workload="texture_lod",
            resolution=shape,
            representation="block_spectral_texture",
            precision="balanced",
            error_budget=2.0e-2,
            bytes_read=texture.byte_size,
            bytes_written=values.nbytes,
            peak_memory=values.nbytes,
            potential_elements=count * count,
            materialized_elements=count * count,
            fft_count=2,
            fallback_conversions=sum(
                str(getattr(block.representation, "value", block.representation)) == "DENSE_SPF"
                for block in texture.blocks
            ),
            field_operations=1,
        )
        return values, RenderReport(
            metrics,
            passes=(CompilerPassRecord("texture_lod", len(texture.blocks), len(texture.blocks)),),
            decisions=texture.candidate_report(),
            capabilities_used=("spectral_texture", "smooth_lod_window"),
            diagnostics={"hostile": True, "expected_to_lose": True, "input_entropy": "uniform_random"},
        )

    wave_source = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    wave = CoherentWaveField.from_array(wave_source, sample_spacing=(0.02, 0.02))

    def run_wave() -> tuple[np.ndarray, RenderReport]:
        result, report = propagate(wave, distance=0.4, wavelength=0.03, return_report=True)
        return result.materialize(), report

    wave_reference = run_wave()[0]
    cases = (
        BenchmarkCase(
            "gaussian_splats/fp64_reference",
            "gaussian_splats",
            lambda: rasterize_splats_reference(splats, camera, tile_size=8),
            representation="fp64_alpha_serial",
            reference=splat_reference,
            warmups=warmups,
            repeats=repeats,
            metadata={"baseline_rung": 1},
        ),
        BenchmarkCase(
            "gaussian_splats/sfr_structured",
            "gaussian_splats",
            lambda: rasterize_splats_spf(splats, camera, tile_size=8, compare_reference=False),
            representation="jet2_log_alpha_transport",
            reference=splat_reference,
            warmups=warmups,
            repeats=repeats,
            metadata={"baseline_rung": 4},
        ),
        BenchmarkCase(
            "ocean/irfft",
            "ocean",
            lambda: synthesize_ocean(shape, half_spectrum=ocean_half, method="irfft", return_report=True),
            representation="hermitian_half",
            reference=ocean_reference,
            warmups=warmups,
            repeats=repeats,
        ),
        BenchmarkCase(
            "ocean/sparse_signed_real",
            "ocean",
            lambda: synthesize_ocean(shape, half_spectrum=ocean_half, method="sparse", return_report=True),
            representation="sparse_signed_real",
            reference=ocean_reference,
            warmups=warmups,
            repeats=repeats,
        ),
        BenchmarkCase(
            "volume/ordered_transport",
            "volume",
            lambda: render_volume(segments, background=(0.01, 0.01, 0.015), return_report=True),
            representation="affine_transport",
            reference=volume_reference,
            warmups=warmups,
            repeats=repeats,
        ),
        BenchmarkCase(
            "texture_lod/high_entropy",
            "texture_lod",
            run_texture_lod,
            representation="block_spectral_texture",
            reference=texture_reference,
            warmups=warmups,
            repeats=repeats,
            metadata={"hostile": True, "expected_to_lose": True},
        ),
        BenchmarkCase(
            "wave_optics/dense_fresnel",
            "wave_optics",
            run_wave,
            representation="dense_coherent",
            reference=wave_reference,
            warmups=warmups,
            repeats=repeats,
        ),
    )
    workload_suite = run_benchmarks(cases)
    workloads = ("filter_chain", "gaussian_splats", "ocean", "volume", "texture_lod", "wave_optics")
    return BenchmarkSuite(
        (*filter_suite.rows, *workload_suite.rows),
        {
            "suite": "sfr_v02_portable",
            "workloads": list(workloads),
            "external_renderer_parity": "not_measured",
            "hostile_cases_included": ("texture_lod/high_entropy",),
        },
    )


def with_runtime(report: RenderReport, runtime_ms: float) -> RenderReport:
    """Return a report copy with externally measured runtime attached."""

    if not isinstance(report, RenderReport):
        raise TypeError("with_runtime expects RenderReport")
    return replace(report, metrics=replace(report.metrics, runtime_ms=float(runtime_ms)))


__all__ = [
    "REQUIRED_RESULT_COLUMNS",
    "BenchmarkCase",
    "BenchmarkRow",
    "BenchmarkSuite",
    "benchmark_filter_chain",
    "benchmark_renderer_suite",
    "run_benchmark",
    "run_benchmarks",
    "with_runtime",
]
