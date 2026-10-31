"""Quality and separately sampled resident-memory observations."""

from __future__ import annotations

import math
import platform
import resource
import threading
import time
import tracemalloc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore[import-untyped]

from spf.renderer.report import psnr as _report_psnr
from spf.renderer.report import relative_l2 as _report_relative_l2
from spf.renderer.report import ssim_local as _report_ssim_local
from spf.renderer.report import temporal_difference_error as _report_temporal_error


def display(values: Any) -> np.ndarray:
    """Fixed linear-to-display mapping shared by every arm; no per-arm scaling."""
    x = np.maximum(np.asarray(values).real, 0)
    return np.asarray((x / (1 + x)) ** (1 / 2.2), dtype=np.float32)


def window_ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Wang-style 11x11 Gaussian SSIM, population covariance, display range 1.

    The five-pixel border is excluded, matching the ordinary valid-window mean.
    Float64 accumulation avoids cancellation in almost-identical flat regions.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or min(a.shape) < 11:
        raise ValueError("SSIM requires matching 2D arrays at least 11x11")

    def blur(x: np.ndarray) -> np.ndarray:
        return np.asarray(gaussian_filter(x, sigma=1.5, truncate=3.5, mode="reflect"))

    ma, mb = blur(a), blur(b)
    va, vb = blur(a * a) - ma * ma, blur(b * b) - mb * mb
    cov = blur(a * b) - ma * mb
    score = ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2)) / ((ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2))
    return float(np.mean(score[5:-5, 5:-5]))


def quality(actual: Any, reference: Any) -> dict[str, Any]:
    a, b = np.asarray(actual), np.asarray(reference)
    if a.shape != b.shape or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("quality comparison requires matching finite outputs")
    error = float(np.linalg.norm((a - b).ravel()))
    norm = float(np.linalg.norm(b.ravel()))
    relative = error / norm if norm else (0.0 if not error else None)
    da, db = display(a), display(b)
    mse = float(np.mean((da.astype(float) - db) ** 2))
    return {
        "relative_l2": relative,
        "max_abs_error": float(np.max(np.abs(a - b))),
        "psnr": None if mse == 0 else float(-10 * np.log10(mse)),
        "psnr_status": "identical" if mse == 0 else "measured",
        "ssim": window_ssim(da, db) if a.ndim == 2 and min(a.shape) >= 11 else None,
        "lpips": None,
        "lpips_status": "not_measured",
        "quality_projection": "complex L2; real component through fixed Reinhard + gamma 1/2.2 for display",
        "ssim_definition": "11x11 Gaussian sigma=1.5; population covariance; range=1; valid-window mean",
    }


def sample_rss(function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """Profile a separate execution, never put this polling thread in a timing trial."""
    import psutil  # type: ignore[import-untyped]

    process = psutil.Process()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(0.002)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        result = function()
        peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        thread.join()
    return result, {
        "peak_memory": peak[0],
        "resident_before_profile": baseline,
        "peak_incremental_rss": max(0, peak[0] - baseline),
        "memory_instrument": "psutil process RSS, sampled every 2 ms on a separate execution; sampled lower bound",
        "memory_profile_ms": (time.perf_counter() - start) * 1000,
    }


# ---------------------------------------------------------------------------
# Design v0.2 §48 core metrics
#
# Every function here either delegates to the existing primitive in
# ``spf.renderer.report`` (PSNR, SSIM, relative L2, §51.6 temporal error) or
# computes a ratio over counters that some measured path already produced.
# Nothing in this block re-implements an arithmetic primitive, and nothing
# estimates a hardware quantity: a metric with no counter behind it is returned
# as ``None`` next to a ``*_status`` string naming why it is absent.
# ---------------------------------------------------------------------------

#: §48.4 numerator candidates, in the order they are preferred. A path that
#: already counted ``reduction_operations`` wins; otherwise the typed reducer
#: counters are summed, which is the same quantity the renderer accumulates
#: into ``reduction_operations`` (``spf/renderer/splats.py``).
REDUCTION_COUNTERS = ("reduction_operations",)
REDUCTION_SURROGATE_COUNTERS = ("positive_logadds", "signed_logadds", "pivot_reductions", "complex_adds")
FIELD_COUNTERS = ("field_operations",)

#: §49 fixed factors. ``B0/B1`` is the value-width ratio of an interleaved
#: complex FP32 coefficient (64 bit) to one SPF32 word (32 bit); it is a format
#: constant, not a measurement.
VALUE_WIDTH_RATIO = 2.0


def materialization_avoidance(potential: Any, realized: Any) -> float | None:
    """§48.3 ``M_A = N_potential / N_realized``; ``None`` when either count is absent."""

    try:
        n_potential = float(potential)
        n_realized = float(realized)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(n_potential) or n_potential < 0.0:
        return None
    if n_realized == 0.0:
        return math.inf if n_potential > 0.0 else 1.0
    if not math.isfinite(n_realized) or n_realized < 0.0:
        return None
    return n_potential / n_realized


def reduction_density(reduction_operations: Any, field_operations: Any) -> float | None:
    """§48.4 ``rho_R = N_reduction / N_field``; ``None`` when the field count is absent."""

    try:
        reductions = float(reduction_operations)
        fields = float(field_operations)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(reductions) and math.isfinite(fields)) or fields <= 0.0:
        return None
    return reductions / fields


def reduction_density_from_counters(counters: Mapping[str, Any]) -> tuple[float | None, str]:
    """``rho_R`` from whatever counters a row carries, plus the counter names used.

    Returns ``(value, provenance)``.  ``provenance`` is either a
    ``"<numerator>/<denominator>"`` string naming the measured counters or an
    ``"absent: ..."`` reason; it is never a guess.
    """

    if "reduction_density" in counters and counters["reduction_density"] is not None:
        value = reduction_density(counters["reduction_density"], 1.0)
        return value, "reduction_density (reported by the path)"
    fields = next((counters[name] for name in FIELD_COUNTERS if counters.get(name)), None)
    if fields is None:
        return None, "absent: no field-operation counter on this row"
    direct = next((counters[name] for name in REDUCTION_COUNTERS if counters.get(name) is not None), None)
    if direct is not None:
        return reduction_density(direct, fields), "reduction_operations/field_operations"
    parts: dict[str, float] = {}
    for name in REDUCTION_SURROGATE_COUNTERS:
        value = counters.get(name)
        if value:
            parts[name] = float(value)
    if not parts:
        return None, "absent: no reduction counter on this row"
    total = math.fsum(parts.values())
    return reduction_density(total, fields), "+".join(sorted(parts)) + "/field_operations"


def psnr(actual: Any, reference: Any, *, data_range: float | None = None) -> tuple[float | None, str]:
    """§48.5 PSNR; identical inputs return ``(None, "identical")`` rather than an unprintable inf."""

    value = _report_psnr(actual, reference, data_range=data_range)
    return (None, "identical") if not math.isfinite(value) else (float(value), "measured")


def ssim(actual: Any, reference: Any, *, data_range: float | None = None) -> tuple[float | None, str]:
    """§48.5 SSIM. ``scikit-image`` when installed, else the repo's own window SSIM."""

    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("SSIM inputs must have identical shapes")
    try:
        from skimage.metrics import structural_similarity
    except ImportError as exc:
        if a.ndim == 2 and min(a.shape) >= 11:
            return float(window_ssim(a, b)), "spf.sfrbench.metrics.window_ssim (11x11 Gaussian)"
        return float(_report_ssim_local(a, b, data_range=data_range)), f"spf.renderer.report.ssim_local ({exc})"
    span = data_range if data_range is not None else float(np.ptp(b)) or 1.0
    score = structural_similarity(b, a, data_range=span)  # type: ignore[no-untyped-call]
    return float(score), "skimage.metrics.structural_similarity"


def lpips(actual: Any, reference: Any) -> tuple[float | None, str]:
    """§48.5 LPIPS. Absent unless the ``lpips`` package is installed on this host."""

    try:
        import lpips as _lpips  # type: ignore[import-not-found]
        import torch
    except ImportError as exc:
        return None, f"absent: {exc.name or 'lpips'} is not installed on this host"

    def prepare(values: Any) -> Any:
        x = np.asarray(display(values), dtype=np.float32)
        if x.ndim == 2:
            x = np.repeat(x[None, ...], 3, axis=0)
        elif x.ndim == 3:
            x = np.moveaxis(x, -1, 0)[:3]
        return torch.from_numpy(np.ascontiguousarray(x * 2.0 - 1.0))[None, ...]

    net = _lpips.LPIPS(net="alex", verbose=False)
    with torch.no_grad():
        score = float(net(prepare(actual), prepare(reference)).item())
    return score, "lpips.LPIPS(net='alex') on the shared display mapping"


def relative_l2(actual: Any, reference: Any) -> float:
    """§48.5 relative L2, delegated to ``spf.renderer.report.relative_l2``."""

    return _report_relative_l2(actual, reference)


def temporal_error(actual: Sequence[Any], reference: Sequence[Any]) -> float:
    """§51.6 temporal difference error, delegated to ``spf.renderer.report``."""

    return _report_temporal_error(actual, reference)


def phase_error(actual: Any, reference: Any, *, magnitude_floor: float | None = None) -> tuple[float | None, str]:
    """§48.5 phase error for coherent paths: mean |wrapped Δθ| over supported samples.

    Only samples whose reference magnitude exceeds ``magnitude_floor`` (default
    ``1e-6`` of the reference peak) enter the mean: below the floor the phase of
    a complex value is not a meaningful quantity.
    """

    a = np.asarray(actual)
    b = np.asarray(reference)
    if a.shape != b.shape:
        raise ValueError("phase inputs must have identical shapes")
    if not (np.iscomplexobj(a) and np.iscomplexobj(b)):
        return None, "absent: path is not coherent (no complex phase on either side)"
    peak = float(np.max(np.abs(b))) if b.size else 0.0
    floor = float(magnitude_floor) if magnitude_floor is not None else 1.0e-6 * peak
    mask = np.abs(b) > floor
    if not bool(np.any(mask)):
        return None, "absent: no reference sample above the magnitude floor"
    delta = np.angle(a[mask]) - np.angle(b[mask])
    wrapped = np.abs((delta + np.pi) % (2.0 * np.pi) - np.pi)
    return float(np.mean(wrapped)), f"mean |wrapped dtheta| over {int(mask.sum())} samples above {floor:.3g}"


def _maxrss_bytes() -> int:
    """``ru_maxrss`` in bytes (Darwin reports bytes, Linux kilobytes)."""

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if platform.system() == "Darwin" else value * 1024


def peak_resident_memory(function: Callable[[], Any]) -> tuple[Any, dict[str, Any]]:
    """§48.2 peak resident memory and intermediate allocations for one untimed execution.

    ``ru_maxrss`` is a high-water mark, so only its *delta* over the call is
    attributable to the call, and even that is a lower bound (the allocator may
    reuse pages it already owns).  ``tracemalloc`` adds the Python-level peak of
    intermediate allocations, which is exact for NumPy buffers allocated through
    Python and blind to allocations inside native kernels.
    """

    before = _maxrss_bytes()
    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    baseline_current, baseline_peak = tracemalloc.get_traced_memory()
    tracemalloc.reset_peak()
    try:
        result = function()
        current, peak = tracemalloc.get_traced_memory()
    finally:
        if not already_tracing:
            tracemalloc.stop()
    after = _maxrss_bytes()
    return result, {
        "peak_rss_bytes": after,
        "peak_rss_delta_bytes": max(0, after - before),
        "peak_rss_mb": after / 1.0e6,
        "tracemalloc_peak_bytes": int(peak),
        "tracemalloc_current_bytes": int(current),
        "tracemalloc_baseline_bytes": int(baseline_current),
        "tracemalloc_baseline_peak_bytes": int(baseline_peak),
        "memory_instrument": (
            "resource.getrusage ru_maxrss high-water delta over one untimed execution plus the tracemalloc "
            "Python-level peak of intermediate allocations; both are lower bounds and neither is DRAM traffic"
        ),
    }


def bytes_moved(counters: Mapping[str, Any]) -> dict[str, Any]:
    """§48.2 ``B_frame``. Sums the accounted ``bytes_read``/``bytes_written``; never estimates DRAM."""

    read = counters.get("bytes_read")
    written = counters.get("bytes_written")
    if read is None and written is None:
        logical_read = counters.get("logical_bytes_read")
        logical_written = counters.get("logical_bytes_written")
        if logical_read is None and logical_written is None:
            return {
                "dram_bytes_per_frame": None,
                "dram_bytes_status": "absent: no byte counter on this row",
            }
        return {
            "dram_bytes_per_frame": None,
            "logical_bytes_per_frame": int(logical_read or 0) + int(logical_written or 0),
            "dram_bytes_status": (
                "absent: only logical boundary byte counts exist on this host; no DRAM performance counter "
                "is readable, so logical bytes are not reported as DRAM bytes/frame"
            ),
        }
    return {
        "dram_bytes_per_frame": int(read or 0) + int(written or 0),
        "dram_bytes_status": "accounted bytes_read + bytes_written as the executed path counted them",
    }


def core_metrics(
    row: Mapping[str, Any],
    *,
    potential_elements: Any = None,
    realized_elements: Any = None,
) -> dict[str, Any]:
    """Assemble the §48 block for one benchmark row from the counters it carries."""

    potential = potential_elements if potential_elements is not None else row.get("potential_elements")
    realized = realized_elements if realized_elements is not None else row.get("materialized_elements")
    rho, rho_provenance = reduction_density_from_counters(row)
    return {
        "m_a": materialization_avoidance(potential, realized),
        "m_a_potential_elements": None if potential is None else int(potential),
        "m_a_realized_elements": None if realized is None else int(realized),
        "rho_r": rho,
        "rho_r_provenance": rho_provenance,
        **bytes_moved(row),
        "psnr": row.get("psnr"),
        "ssim": row.get("ssim"),
        "lpips": row.get("lpips"),
        "relative_l2": row.get("relative_l2"),
        "temporal_error": row.get("temporal_error"),
        "peak_memory": row.get("peak_memory"),
    }
