"""Measured SFR display planning and certified local jet consumers.

The cache stores local polynomial coefficients plus an exact sparse residual,
not ocean pixels. A wide-phase change invalidates its regions lazily. Planning
times actual source-to-display candidates on this host, outside frame timing.
FP32 controls do not enter this planner.
"""

from __future__ import annotations

import ctypes
import math
import time
import weakref
from contextlib import suppress
from dataclasses import replace
from typing import Any, ClassVar

import numpy as np

from spf.renderer._ocean_kernel import NativeOceanSampler, OceanErrorLedger, _as_points, _visible


class _JetStats(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        (name, ctypes.c_uint64)
        for name in (
            "jet_queries", "sparse_queries", "tile_builds", "residual_terms_evaluated",
            "resident_bytes", "approximate_modes", "residual_modes",
        )
    ] + [(name, ctypes.c_double) for name in ("height_bound", "dy_bound", "dx_bound")]


class NativeOceanJetSampler:
    """Own a bounded polynomial cache while retaining the packed source ABI."""

    def __init__(
        self, source: NativeOceanSampler, *, tiles: int = 256,
        degree: int = 3, absolute_budget: float = 1e-6, reuse_after: int = 2,
    ) -> None:
        if source.words is None:
            raise TypeError("local SPF jets require packed SPF source words")
        if not math.isfinite(absolute_budget) or absolute_budget <= 0:
            raise ValueError("jet absolute budget must be finite and positive")
        if isinstance(tiles, bool) or int(tiles) != tiles or not 1 <= tiles <= 512:
            raise ValueError("jet tile count must be an integer in [1, 512]")
        if degree not in (2, 3) or isinstance(degree, bool):
            raise ValueError("jet degree must be 2 or 3")
        if isinstance(reuse_after, bool) or int(reuse_after) != reuse_after or reuse_after < 0:
            raise ValueError("jet reuse threshold must be a nonnegative integer")
        self.source = source
        self.tiles, self.degree = int(tiles), int(degree)
        self.absolute_budget = float(absolute_budget)
        self.reuse_after = int(reuse_after)
        lib = source._library
        lib.spf_sfr_ocean_jet_create.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ]
        lib.spf_sfr_ocean_jet_create.restype = ctypes.c_void_p
        lib.spf_sfr_ocean_jet_release.argtypes = [ctypes.c_void_p]
        lib.spf_sfr_ocean_jet_release.restype = None
        lib.spf_sfr_ocean_jet_invalidate.argtypes = [ctypes.c_void_p]
        lib.spf_sfr_ocean_jet_invalidate.restype = None
        lib.spf_sfr_ocean_jet_stats.argtypes = [ctypes.c_void_p, ctypes.POINTER(_JetStats)]
        lib.spf_sfr_ocean_jet_stats.restype = ctypes.c_int
        lib.spf_sfr_ocean_sample_jet.argtypes = [
            ctypes.c_void_p, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_double),
        ]
        lib.spf_sfr_ocean_sample_jet.restype = ctypes.c_int
        lib.spf_sfr_ocean_jet_points.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t, ctypes.POINTER(ctypes.c_float),
        ]
        lib.spf_sfr_ocean_jet_points.restype = ctypes.c_int
        self._sample = lib.spf_sfr_ocean_sample_jet
        self._cache = lib.spf_sfr_ocean_jet_create(
            source.context_ptr, self.tiles, self.tiles, self.degree, self.absolute_budget, int(reuse_after),
        )
        if not self._cache:
            raise RuntimeError("native ocean jet preparation failed")
        if not hasattr(source, "_jet_caches"):
            source._jet_caches = weakref.WeakSet()
        source._jet_caches.add(self)

    def __getattr__(self, name: str) -> Any:
        if name == "source":
            raise AttributeError(name)
        return getattr(self.source, name)

    @property
    def context_ptr(self) -> ctypes.c_void_p:
        if not self._cache:
            raise RuntimeError("native ocean jet sampler is closed")
        return ctypes.c_void_p(self._cache)

    @property
    def function_ptr(self) -> ctypes.c_void_p:
        return ctypes.cast(self._sample, ctypes.c_void_p)

    @property
    def diagnostics(self) -> dict[str, Any]:
        stats = _JetStats()
        status = self.source._library.spf_sfr_ocean_jet_stats(self.context_ptr, ctypes.byref(stats))
        if status:
            raise RuntimeError("native ocean jet stats failed")
        return {
            "representation": "local_taylor_jet_with_sparse_residual",
            "degree": self.degree, "tiles_per_axis": self.tiles,
            "additional_absolute_budget": self.absolute_budget,
            "cached_output_pixels": 0,
            **{name: getattr(stats, name) for name, _ in stats._fields_},
        }

    def error_ledger(self, *, coordinate_bound: float | None = None) -> OceanErrorLedger:
        base = self.source.error_ledger(coordinate_bound=coordinate_bound)
        bounds = self.diagnostics
        return replace(
            base,
            height_absolute_bound=base.height_absolute_bound + bounds["height_bound"],
            gradient_y_absolute_bound=base.gradient_y_absolute_bound + bounds["dy_bound"],
            gradient_x_absolute_bound=base.gradient_x_absolute_bound + bounds["dx_bound"],
        )

    def sample(self, y: float, x: float, *, need_gradients: bool = True) -> np.ndarray:
        result = np.empty(3, np.float64)
        status = self._sample(self.context_ptr, y, x, int(need_gradients),
                              result.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if status:
            raise ValueError(f"native jet query failed ({status})")
        return result

    def points(self, points: Any, *, visible: Any = None, return_stats: bool = False) -> Any:
        points = _as_points(points)
        mask = _visible(visible, len(points))
        result = np.empty((len(points), 3), np.float32)
        status = self.source._library.spf_sfr_ocean_jet_points(
            self.context_ptr, points.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.c_uint8)() if mask is None else mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            len(points), result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        )
        if status:
            raise ValueError(f"native jet point batch failed ({status})")
        return (result, self.stats) if return_stats else result

    def close(self) -> None:
        if getattr(self, "_cache", None):
            self.source._library.spf_sfr_ocean_jet_release(self._cache)
            self._cache = None

    def invalidate(self) -> None:
        if self._cache:
            self.source._library.spf_sfr_ocean_jet_invalidate(self._cache)

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def calibrate_ocean_display(
    source: Any, display: Any, *, sampling: str, footprint: Any = None,
    absolute_budget: float = 1e-6,
) -> tuple[str, Any, dict[str, Any]] | None:
    """Measure legal SFR candidates through the actual final display consumer.

    This runs once during explicit image preparation. It does not modify phase
    state or optimize an FP32 source. Explicit grid/method requests remain hard
    constraints. The measured cost includes field synthesis and shading; each
    jet candidate starts cold, so its tile construction is charged.
    """
    if sampling == "grid" or not callable(getattr(source, "bind_execution", None)):
        return None
    if str(getattr(source.backend, "name", source.backend)) == "cuda":
        return _calibrate_cuda_display(source, display, footprint, absolute_budget)
    if str(getattr(source.backend, "name", source.backend)) not in {"cpu", "numpy"}:
        return None
    analytic = getattr(getattr(source, "carrier", None), "source_kind", None) == "analytic"
    if not analytic and getattr(source, "_native_representation", lambda: "")() != "spf32":
        return None
    if not source.native_queries_supported or not source.native_footprint_supported(footprint):
        return None
    requested = source._candidate_plan.requested_method
    if requested in {"irfft", "fb3", "dense", "fft", "dense_irfft", "packed", "v02"}:
        return None
    from spf.renderer.ocean_planner import OceanOutputRequest

    old_plan = source.plan
    query_count = math.prod(display.resolution) * (display.intersections + 1)
    request = OceanOutputRequest(sampling="points", outputs=("height", "grad_y", "grad_x"),
                                 footprint=footprint, backend="cpu", output_dtype="float32",
                                 point_count=query_count)
    source.bind_execution(request, native_available=True)
    base = source.native_sampler(footprint=footprint)
    if not analytic and (not isinstance(base, NativeOceanSampler) or base.words is None):
        source.plan = old_plan
        return None
    candidates: list[tuple[str, Any, dict[str, Any]]] = []
    started = time.perf_counter_ns()
    try:
        start = time.perf_counter_ns()
        display.execute(sampler=base)
        candidates.append(("queries", base, {
            "representation": "analytic_atom_or_jet" if analytic else "packed_sparse",
            "runtime_ms": (time.perf_counter_ns()-start)/1e6,
            "additional_error_bound": 0.0, "materialized_field_elements": 0,
        }))
        jet_available = not analytic and hasattr(base._library, "spf_sfr_ocean_jet_create")
        if jet_available and absolute_budget > 0:
            for tiles, degree in ((128, 3), (256, 3), (512, 2)):
                jet = NativeOceanJetSampler(base, tiles=tiles, degree=degree, absolute_budget=absolute_budget)
                descriptor = jet.diagnostics
                if descriptor["approximate_modes"] <= descriptor["residual_modes"]:
                    jet.close()
                    continue
                start = time.perf_counter_ns()
                display.execute(sampler=jet)
                candidates.append(("queries", jet, {
                    **jet.diagnostics, "runtime_ms": (time.perf_counter_ns()-start)/1e6,
                    "materialized_field_elements": 0,
                }))
        if sampling == "auto":
            # Use the selected exact source-grid algorithm and the same camera
            # and display size. No N log N extrapolation decides this boundary.
            source.plan = old_plan
            start = time.perf_counter_ns()
            fields = source.execute_fields(output_dtype=np.float32, footprint=footprint)
            display.execute(fields)
            candidates.append(("grid", None, {
                "representation": f"{source.plan.method}_grid",
                "runtime_ms": (time.perf_counter_ns()-start)/1e6,
                "additional_error_bound": 0.0,
                "materialized_field_elements": 3*math.prod(source.shape),
            }))
        selected = min(candidates, key=lambda candidate: candidate[2]["runtime_ms"])
        for _, sampler, _ in candidates:
            if isinstance(sampler, NativeOceanJetSampler) and sampler is not selected[1]:
                sampler.close()
        selected_sampling, sampler, descriptor = selected
        costs = {
            "model": "measured_source_to_display_on_current_host",
            "candidates": [row for _, _, row in candidates],
            "selected_representation": descriptor["representation"],
            "calibration_ms": (time.perf_counter_ns()-started)/1e6,
            "jet_absolute_budget": absolute_budget,
            "grid_samples": math.prod(source.shape), "query_samples_upper_bound": query_count,
            "phase_advance_during_calibration": 0, "cold_jet_tiles_charged": True,
            "native_jet_available": jet_available,
            "calibration_materialized_field_elements": sum(
                row["materialized_field_elements"] for _, _, row in candidates
            ),
        }
        return selected_sampling, sampler, costs
    finally:
        source.plan = old_plan


def _calibrate_cuda_display(
    source: Any, display: Any, footprint: Any, absolute_budget: float,
) -> tuple[str, Any, dict[str, Any]] | None:
    """Measure direct SPF and bounded Jet2 blocks at the same RGB consumer."""
    if not source.native_queries_supported or not source.native_footprint_supported(footprint):
        return None
    sampler = source.native_sampler(footprint=footprint)
    if not callable(getattr(sampler, "configure_query_blocks", None)):
        return None
    started = time.perf_counter_ns()
    sampler.configure_query_blocks(0)
    # Warm up extension/constants once; neither candidate pays compilation.
    display.execute(sampler=sampler)
    candidates = []
    for budget in (0.0, absolute_budget) if absolute_budget > 0 else (0.0,):
        descriptor = sampler.configure_query_blocks(budget)
        if budget > 0 and descriptor is None:
            continue
        raw_ms = []
        for _ in range(3):
            sampler.invalidate_query_blocks()
            sampler._torch.cuda.synchronize(sampler.device)
            start = time.perf_counter_ns()
            result = display.execute(sampler=sampler)
            raw_ms.append((time.perf_counter_ns()-start)/1e6)
            if descriptor is not None:
                actual_queries = result.diagnostics["height_queries"] + result.diagnostics["gradient_queries"]
                counted_queries = result.diagnostics["jet_queries"] + result.diagnostics["jet_sparse_fallback_queries"]
                if actual_queries != counted_queries:
                    raise RuntimeError("CUDA query-block counters do not cover the executed field queries")
        candidates.append({
            "representation": "packed_sparse" if descriptor is None else descriptor["representation"],
            "runtime_ms": float(np.median(raw_ms)), "raw_ms": raw_ms,
            "absolute_budget": budget, "query_blocks": descriptor,
            "materialized_field_elements": 0,
            "execution": {key: result.diagnostics.get(key) for key in
                          ("jet_queries", "jet_builds", "jet_term_preparations", "term_evaluations",
                           "jet_sparse_fallback_queries", "height_queries", "gradient_queries", "water_pixels")},
        })
    selected = min(candidates, key=lambda row: row["runtime_ms"])
    sampler.configure_query_blocks(selected["absolute_budget"])
    return "queries", sampler, {
        "model": "measured_source_to_display_on_current_device",
        "candidates": candidates, "selected_representation": selected["representation"],
        "calibration_ms": (time.perf_counter_ns()-started)/1e6,
        "jet_absolute_budget": absolute_budget,
        "cold_jet_tiles_charged": True, "phase_advance_during_calibration": 0,
        "calibration_materialized_field_elements": 0,
    }
