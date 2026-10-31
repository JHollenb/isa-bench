"""Transparent execution, planning, and quality reports for SFR v0.2."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.tensor.v02 import ErrorLedger


@dataclass(frozen=True, slots=True)
class RepresentationDecision:
    field_id: str
    region: str
    selected: str
    candidates: tuple[str, ...]
    predicted_cost: float
    predicted_error: float
    reason: str


@dataclass(frozen=True, slots=True)
class CompilerPassRecord:
    name: str
    input_nodes: int
    output_nodes: int
    rewrites: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(slots=True)
class RenderMetrics:
    backend: str = "reference"
    workload: str = "generic"
    resolution: tuple[int, ...] = ()
    representation: str = "unknown"
    precision: str = "balanced"
    error_budget: float = math.inf
    runtime_ms: float = 0.0
    energy_j: float | None = None
    # Average power over the timed interval when a hardware sampler measured it
    # (design §47.1 / §48.1 E_frame); ``None`` means no sampler, never an estimate.
    power_w: float | None = None
    bytes_read: int = 0
    bytes_written: int = 0
    peak_memory: int = 0
    potential_elements: int = 0
    materialized_elements: int = 0
    atoms_considered: int = 0
    atoms_pruned: int = 0
    positive_logadds: int = 0
    complex_adds: int = 0
    pivot_reductions: int = 0
    transport_compositions: int = 0
    fft_count: int = 0
    fallback_conversions: int = 0
    field_operations: int = 0
    reduction_operations: int = 0
    synchronization_depth: int = 0
    psnr: float | None = None
    ssim: float | None = None
    lpips: float | None = None
    relative_l2: float | None = None
    temporal_error: float | None = None
    # The legacy ``potential_elements``/``materialized_elements`` pair is
    # retained for report compatibility.  Structured backends should populate
    # these fields so M_A measures dense *intermediate* values, not merely the
    # size of a requested ROI.
    potential_intermediate_elements: int = 0
    materialized_intermediate_elements: int = 0
    terminal_output_elements: int = 0
    potential_intermediate_bytes: int = 0
    materialized_intermediate_bytes: int = 0
    terminal_output_bytes: int = 0
    logical_peak_bytes: int = 0

    @property
    def dram_traffic_bytes(self) -> int:
        """Bytes moved through memory as the path accounted them (``bytes_read + bytes_written``)."""

        return int(self.bytes_read) + int(self.bytes_written)

    @property
    def materialization_avoidance(self) -> float:
        potential = int(self.potential_intermediate_elements)
        materialized = int(self.materialized_intermediate_elements)
        if potential or materialized:
            if materialized == 0:
                return math.inf if potential else 1.0
            return float(potential) / float(materialized)
        if self.materialized_elements == 0:
            return math.inf if self.potential_elements else 1.0
        return float(self.potential_elements) / float(self.materialized_elements)

    @property
    def reduction_density(self) -> float:
        if self.field_operations == 0:
            return 0.0
        return float(self.reduction_operations) / float(self.field_operations)

    def add(self, other: RenderMetrics) -> None:
        for name in (
            "runtime_ms",
            "bytes_read",
            "bytes_written",
            "peak_memory",
            "potential_elements",
            "materialized_elements",
            "atoms_considered",
            "atoms_pruned",
            "positive_logadds",
            "complex_adds",
            "pivot_reductions",
            "transport_compositions",
            "fft_count",
            "fallback_conversions",
            "field_operations",
            "reduction_operations",
            "synchronization_depth",
            "potential_intermediate_elements",
            "materialized_intermediate_elements",
            "terminal_output_elements",
            "potential_intermediate_bytes",
            "materialized_intermediate_bytes",
            "terminal_output_bytes",
            "logical_peak_bytes",
        ):
            if name in {"peak_memory", "synchronization_depth", "logical_peak_bytes"}:
                setattr(self, name, max(int(getattr(self, name)), int(getattr(other, name))))
            else:
                setattr(self, name, getattr(self, name) + getattr(other, name))


@dataclass(frozen=True, slots=True)
class RenderReport:
    metrics: RenderMetrics
    ledger: ErrorLedger = field(default_factory=ErrorLedger)
    passes: tuple[CompilerPassRecord, ...] = ()
    decisions: tuple[RepresentationDecision, ...] = ()
    fallbacks: tuple[str, ...] = ()
    capabilities_used: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    # Estimate and measurement side by side (design §47 / blog Tier 2 item 8).
    # ``cost_estimate`` is what the cost model predicted for the executed plan,
    # ``cost_measured`` is what the executed path timed itself at, and
    # ``cost_source`` names the calibration that priced the estimate.
    cost_estimate: Mapping[str, Any] | None = None
    cost_measured: Mapping[str, Any] | None = None
    cost_source: str = "static_defaults"

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, RenderMetrics):
            raise TypeError("RenderReport.metrics must be RenderMetrics")
        if not isinstance(self.ledger, ErrorLedger):
            raise TypeError("RenderReport.ledger must be ErrorLedger")
        object.__setattr__(self, "passes", tuple(self.passes))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "fallbacks", tuple(str(value) for value in self.fallbacks))
        object.__setattr__(self, "capabilities_used", tuple(str(value) for value in self.capabilities_used))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        for name in ("cost_estimate", "cost_measured"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "cost_source", str(self.cost_source))

    def with_measured_cost(self, runtime_ms: float, **extra: Any) -> RenderReport:
        """Attach the executed path's own timing next to the estimate (also fills ``metrics.runtime_ms``)."""

        previous = dict(self.cost_measured) if self.cost_measured else {}
        measured = {**previous, "runtime_ms": float(runtime_ms), **extra}
        estimate = dict(self.cost_estimate) if self.cost_estimate else None
        if estimate is not None and estimate.get("estimated_ms") is not None and runtime_ms > 0.0:
            measured["estimate_to_measured_ratio"] = float(estimate["estimated_ms"]) / float(runtime_ms)
        metrics = replace(self.metrics, runtime_ms=float(runtime_ms))
        return replace(self, metrics=metrics, cost_measured=measured)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": asdict(self.metrics),
            "dram_traffic_bytes": self.metrics.dram_traffic_bytes,
            "materialization_avoidance": self.metrics.materialization_avoidance,
            "reduction_density": self.metrics.reduction_density,
            "error_ledger": {
                name: getattr(self.ledger, name)
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
            "error_bound": self.ledger.total_bound,
            "passes": [asdict(value) for value in self.passes],
            "decisions": [asdict(value) for value in self.decisions],
            "fallbacks": list(self.fallbacks),
            "capabilities_used": list(self.capabilities_used),
            "diagnostics": dict(self.diagnostics),
            "cost_estimate": None if self.cost_estimate is None else dict(self.cost_estimate),
            "cost_measured": None if self.cost_measured is None else dict(self.cost_measured),
            "cost_source": self.cost_source,
        }


def relative_l2(actual: Any, reference: Any) -> float:
    left = np.asarray(actual)
    right = np.asarray(reference)
    if left.shape != right.shape:
        raise ValueError("quality inputs must have identical shapes")
    denominator = max(float(np.linalg.norm(right.reshape(-1))), np.finfo(np.float64).tiny)
    return float(np.linalg.norm((left - right).reshape(-1)) / denominator)


def _quality_data_range(reference: np.ndarray, data_range: float | None) -> float:
    if data_range is not None:
        value = float(data_range)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("quality data_range must be positive and finite")
        return value
    observed = float(np.ptp(reference)) if reference.size else 0.0
    if observed > 0.0 and math.isfinite(observed):
        return observed
    magnitude = float(np.max(np.abs(reference))) if reference.size else 0.0
    return magnitude if magnitude > 0.0 and math.isfinite(magnitude) else 1.0


def psnr(actual: Any, reference: Any, *, data_range: float | None = None) -> float:
    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("quality inputs must have identical shapes")
    mse = float(np.mean(np.square(left - right))) if left.size else 0.0
    if mse == 0.0:
        return math.inf
    dynamic_range = _quality_data_range(right, data_range)
    return float(20.0 * math.log10(dynamic_range) - 10.0 * math.log10(mse))


def ssim_global(actual: Any, reference: Any, *, data_range: float | None = None) -> float:
    """Deterministic global SSIM diagnostic without an image-library dependency."""

    left = np.asarray(actual, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("quality inputs must have identical shapes")
    if left.size == 0:
        return 1.0
    dynamic_range = _quality_data_range(right, data_range)
    c1 = (0.01 * dynamic_range) ** 2
    c2 = (0.03 * dynamic_range) ** 2
    mean_left = float(np.mean(left))
    mean_right = float(np.mean(right))
    variance_left = float(np.mean(np.square(left - mean_left)))
    variance_right = float(np.mean(np.square(right - mean_right)))
    covariance = float(np.mean((left - mean_left) * (right - mean_right)))
    numerator = (2.0 * mean_left * mean_right + c1) * (2.0 * covariance + c2)
    denominator = (mean_left**2 + mean_right**2 + c1) * (variance_left + variance_right + c2)
    return 1.0 if denominator == 0.0 else float(numerator / denominator)


def ssim_local(
    actual: Any,
    reference: Any,
    *,
    data_range: float | None = None,
    window_size: int = 8,
) -> float:
    """Average deterministic SSIM over non-overlapping image windows."""

    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("quality inputs must have identical shapes")
    if left.ndim < 2:
        return ssim_global(left, right, data_range=data_range)
    window = int(window_size)
    if window <= 0:
        raise ValueError("window_size must be positive")
    dynamic_range = _quality_data_range(right, data_range)
    scores: list[float] = []
    for row in range(0, left.shape[0], window):
        for column in range(0, left.shape[1], window):
            slices = (slice(row, min(row + window, left.shape[0])), slice(column, min(column + window, left.shape[1])))
            scores.append(ssim_global(left[slices], right[slices], data_range=dynamic_range))
    return float(math.fsum(scores) / len(scores)) if scores else 1.0


def dark_region_rmse(
    actual: Any,
    reference: Any,
    *,
    data_range: float | None = None,
    dark_fraction: float = 0.05,
) -> float:
    """RMSE over pixels whose reference intensity is in the darkest band."""

    left = np.asarray(actual, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("quality inputs must have identical shapes")
    fraction = float(dark_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("dark_fraction must lie in [0, 1]")
    dynamic_range = _quality_data_range(right, data_range)
    intensity = np.max(np.abs(right), axis=-1) if right.ndim >= 3 else np.abs(right)
    threshold = float(np.min(intensity)) + fraction * dynamic_range if intensity.size else 0.0
    mask = intensity <= threshold
    if right.ndim >= 3:
        mask = np.broadcast_to(mask[..., None], right.shape)
    difference = left[mask] - right[mask]
    return float(np.sqrt(np.mean(np.square(difference)))) if difference.size else 0.0


def temporal_difference_error(actual: Sequence[Any], reference: Sequence[Any]) -> float:
    if len(actual) != len(reference):
        raise ValueError("temporal sequences must have equal length")
    if len(actual) < 2:
        return 0.0
    errors = []
    for index in range(len(actual) - 1):
        actual_delta = np.asarray(actual[index + 1]) - np.asarray(actual[index])
        reference_delta = np.asarray(reference[index + 1]) - np.asarray(reference[index])
        errors.append(float(np.linalg.norm((actual_delta - reference_delta).reshape(-1))))
    return float(math.fsum(errors) / len(errors))


__all__ = [
    "CompilerPassRecord",
    "RenderMetrics",
    "RenderReport",
    "RepresentationDecision",
    "dark_region_rmse",
    "psnr",
    "relative_l2",
    "ssim_global",
    "ssim_local",
    "temporal_difference_error",
]
