"""Fused CUDA splat tile compositor (design v0.2 §36.3, Appendix B).

One CUDA kernel launch per frame composites every tile.  The kernel
(``_splat_cuda_kernels.cu``) runs one thread block per tile -- or per row group
of a tile, when the frame bins into too few tiles to fill the device -- and one
thread per pixel: the depth-sorted splats of the tile are streamed in chunks, the
log-alpha codes come from the §13.3 forward-difference recurrence on the
widened jet (or the ABI v23 header log polynomial when the jet is
header-exact), the alpha complement and the ordered ``C' = C + T*c``,
``T' = T*t`` transport run on integer L codes with the §6 LOGADD ROMs resident
in shared memory, and each pixel terminates as soon as its transmittance code
reaches ``L_ZERO`` (the remaining entries are then identities of the §18
transport monoid, so termination cannot change a word).

The output words are identical to
:func:`spf.renderer.splat_terminal.rasterize_planes` and to
:func:`spf.renderer._splat_cuda.rasterize`; ``tests/v02/test_sfr_splat_cuda.py``
grades that.  This module only plans (host, vectorized NumPy), uploads, launches
and projects the output words for display.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from spf.tensor import v02
from spf.tensor.logadd_rom import LOGADD_G_ZERO, LOGADD_MINUS, LOGADD_PLUS

if TYPE_CHECKING:
    from spf.renderer.splats import SplatRenderResult

_SOURCE = Path(__file__).with_name("_splat_cuda_kernels.cu")
_EXTENSION_NAME = "spf_sfr_splat_fused_v1"
_BUILD_LOCK = threading.RLock()
_EXTENSION: Any | None = None
_BUILD_ERROR: BaseException | None = None

_L_ZERO = int(v02.SPF32_L_ZERO)
_NAN_L = int(v02.SPF32_NAN_L)
_L_MIN = int(v02.SPF32_L_MIN_FINITE)
_L_MAX = int(v02.SPF32_L_MAX_FINITE)

#: The kernel serves tiles up to 16x16 (one thread per pixel in a 256-thread block).
MAX_TILE_DIM = 16

#: Terminal spellings that select this compositor.
TERMINALS = ("cuda_fused", "fused", "cuda_kernel", "device_fused")

CONSUMER = "spf.renderer._splat_cuda_fused.rasterize_tiles_fused"


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except Exception as exc:  # pragma: no cover - depends on host extras
        raise ImportError("the fused splat kernel requires the optional torch CUDA runtime") from exc
    return torch


def availability() -> tuple[bool, str | None]:
    """Whether the fused CUDA kernel can run here, and why not."""

    if str(os.environ.get("SPF_SFR_SPLAT_FUSED_DISABLE", "")).strip().lower() in {"1", "true", "yes"}:
        return False, "SPF_SFR_SPLAT_FUSED_DISABLE requested"
    if not _SOURCE.is_file():
        return False, f"fused splat kernel source is missing: {_SOURCE}"
    try:
        torch = _import_torch()
    except ImportError as exc:
        return False, str(exc)
    try:
        if not bool(torch.cuda.is_available()):
            return False, "torch reports no CUDA runtime"
    except Exception as exc:  # pragma: no cover - vendor runtime failure
        return False, f"CUDA runtime probe failed: {exc}"
    if _BUILD_ERROR is not None:
        return False, f"the fused splat kernel failed to build: {_BUILD_ERROR}"
    return True, None


def available() -> bool:
    return availability()[0]


def ensure_loaded() -> Any:
    """Build (JIT, cached under ``TORCH_EXTENSIONS_DIR``) and return the extension."""

    global _EXTENSION, _BUILD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _BUILD_ERROR is not None:
        raise RuntimeError("the fused splat CUDA kernel failed to build") from _BUILD_ERROR
    ok, reason = availability()
    if not ok:
        raise RuntimeError(f"the fused splat CUDA kernel is unavailable: {reason}")
    with _BUILD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        try:
            _import_torch()
            from torch.utils.cpp_extension import load  # type: ignore[import-not-found,unused-ignore]

            _EXTENSION = load(
                name=_EXTENSION_NAME,
                sources=[str(_SOURCE)],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr", "-fmad=false"],
                verbose=False,
            )
        except BaseException as exc:  # keep the build failure explicit
            _BUILD_ERROR = exc
            raise RuntimeError(f"failed to compile the fused splat CUDA kernel: {exc}") from exc
    return _EXTENSION


def extension_ready() -> tuple[bool, str | None]:
    """Whether the extension is built (building it on demand), and why not."""

    ok, reason = availability()
    if not ok:
        return False, reason
    try:
        ensure_loaded()
    except RuntimeError as exc:
        return False, str(exc)
    return True, None


def runtime_info() -> dict[str, Any]:
    ok, reason = availability()
    info: dict[str, Any] = {
        "available": bool(ok),
        "reason": reason,
        "kernel_source": str(_SOURCE),
        "kernel_sha256": hashlib.sha256(_SOURCE.read_bytes()).hexdigest() if _SOURCE.is_file() else None,
        "extension": _EXTENSION_NAME,
        "extension_loaded": _EXTENSION is not None,
        "max_tile_dim": MAX_TILE_DIM,
    }
    if not ok:
        return info
    try:
        torch = _import_torch()
        index = int(torch.cuda.current_device())
        info.update(
            {
                "torch_version": str(torch.__version__),
                "cuda_version": str(getattr(torch.version, "cuda", None)),
                "device_name": str(torch.cuda.get_device_name(index)),
                "device_capability": tuple(int(value) for value in torch.cuda.get_device_capability(index)),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        info["runtime_info_error"] = str(exc)
    return info


# ---------------------------------------------------------------------------
# Host plan: the same per-entry decisions ``_splat_cuda._plan_entry`` makes,
# vectorized over every (tile, splat) entry of the frame.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FusedPlan:
    """Flat, device-ready description of every tile entry in depth order."""

    tile_bounds: np.ndarray  # (T, 4) int32
    tile_offset: np.ndarray  # (T + 1,) int32
    flags: np.ndarray  # (E,) int32, bit 0 = header route
    ticks: np.ndarray  # (E, 6) int32
    jet: np.ndarray  # (E, 7) float64
    color_codes: np.ndarray  # (E, 3) int32 L codes
    color_signs: np.ndarray  # (E, 3) int32 signs
    depth: np.ndarray  # (E,) float64
    primitive_id: np.ndarray  # (E,) int64
    routes: dict[str, int]
    tile_pixels: int
    jet_pixels: int


def _jet_eval(coeff: np.ndarray, qx: float, qy: float) -> np.ndarray:
    """:func:`spf.tensor.v02.jet_eval` over stacked coefficients, add for add.

    The expression order is the one in ``jet_eval``; float64 is associative
    nowhere, so keeping it is what makes the device codes word-identical.
    """

    value = coeff[:, 0]
    grad_x = coeff[:, 1]
    grad_y = coeff[:, 2]
    hess_xx = coeff[:, 3]
    hess_xy = coeff[:, 4]
    hess_yy = coeff[:, 5]
    return value + grad_x * qx + grad_y * qy + 0.5 * hess_xx * qx * qx + hess_xy * qx * qy + 0.5 * hess_yy * qy * qy


def _header_ticks(coeff: np.ndarray, guard: np.ndarray, extent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(ticks, header_route)`` for :func:`spf.tensor.v02.jet_to_tensor_block_header`.

    ``bound == 0`` (the condition ``splat_terminal`` takes the header route on)
    holds exactly when every header coefficient with a non-zero monomial over
    the tile extent lands on a whole header tick and inside its int16 field.
    The raw tick values are exact in float64: the denominators are powers of
    two.  ``extent`` is ``(qy_max, qx_max)`` per entry.
    """

    denominator = np.stack([guard, guard, guard, 2.0 * guard, 2.0 * guard, guard], axis=1)
    order = np.stack(
        [coeff[:, 0], coeff[:, 1], coeff[:, 2], coeff[:, 3], coeff[:, 5], coeff[:, 4]], axis=1
    )  # c, gx, gy, hxx2, hyy2, hxy
    raw = order / denominator
    finite = np.all(np.isfinite(raw), axis=1)
    rounded = np.rint(np.where(finite[:, None], raw, 0.0))
    residual = np.abs(raw - rounded)
    qy_max = extent[:, 0].astype(np.float64)
    qx_max = extent[:, 1].astype(np.float64)
    monomial = np.stack(
        [np.ones_like(qx_max), qx_max, qy_max, qx_max * qx_max, qy_max * qy_max, qx_max * qy_max], axis=1
    )
    low = np.asarray([_L_MIN, -32768, -32768, -32768, -32768, -32768], dtype=np.float64)
    high = np.asarray([_L_MAX, 32767, 32767, 32767, 32767, 32767], dtype=np.float64)
    stored = np.clip(rounded, low, high)
    charged = (residual + np.abs(rounded - stored)) * monomial
    header = finite & np.all(charged == 0.0, axis=1)
    # Kernel order: c, gy, gx, hyy2, hxx2, hxy (header dim 0 is the row axis).
    ticks = np.stack(
        [stored[:, 0], stored[:, 2], stored[:, 1], stored[:, 4], stored[:, 3], stored[:, 5]], axis=1
    ).astype(np.int32)
    return ticks, header


def _canonical_signed(L: np.ndarray, sign: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``LogSignedPlane.__post_init__`` over stacked colour codes."""

    nan = L == _NAN_L
    zero = (L == _L_ZERO) | (sign == 0)
    codes = np.clip(L, _L_MIN, _L_MAX)
    codes = np.where(zero, _L_ZERO, codes)
    codes = np.where(nan, _NAN_L, codes)
    signs = np.where(sign > 0, 1, -1)
    signs = np.where(zero | nan, 0, signs)
    return codes.astype(np.int32), signs.astype(np.int32)


def plan(bins: Any, *, header_tolerance_ticks: float = 0.0) -> FusedPlan | None:
    """Flatten every non-empty tile of ``bins`` into device-ready arrays.

    Returns ``None`` when the scene is outside the kernel's envelope (a tile
    edge past :data:`MAX_TILE_DIM`, a non-real jet, or no entries at all), so
    the caller can fall back without a partial render.
    """

    if header_tolerance_ticks != 0.0:
        return None  # the kernel takes the header route only when it is word-exact
    tiles = [tile for tile in bins.tiles if tile.entries]
    if not tiles:
        return None
    bounds = np.asarray([tile.bounds for tile in tiles], dtype=np.int32)
    heights = bounds[:, 2] - bounds[:, 0]
    widths = bounds[:, 3] - bounds[:, 1]
    if int(heights.max()) > MAX_TILE_DIM or int(widths.max()) > MAX_TILE_DIM:
        return None
    counts = np.asarray([len(tile.entries) for tile in tiles], dtype=np.int64)
    offsets = np.zeros(len(tiles) + 1, dtype=np.int32)
    offsets[1:] = np.cumsum(counts).astype(np.int32)
    entries = [entry for tile in tiles for entry in tile.entries]
    total = len(entries)
    jets = [entry.spf_log_alpha_jet for entry in entries]
    try:
        coeff = np.asarray(
            [(j.value, j.grad_x, j.grad_y, j.hess_xx, j.hess_xy, j.hess_yy) for j in jets], dtype=np.float64
        )
    except (TypeError, ValueError):
        return None  # a non-real jet: keep the typed path
    guard = np.asarray([float(1 << int(j.guard_L)) for j in jets], dtype=np.float64)
    extent = np.repeat(np.stack([heights - 1, widths - 1], axis=1), counts, axis=0)

    ticks, header = _header_ticks(coeff, guard, extent)
    row_value = _jet_eval(coeff, 0.0, 0.0)
    row_dx = _jet_eval(coeff, 1.0, 0.0) - row_value
    row_dy = _jet_eval(coeff, 0.0, 1.0) - row_value
    jet = np.stack(
        [row_value, row_dx, row_dy, coeff[:, 3], coeff[:, 4], coeff[:, 5], guard], axis=1
    ).astype(np.float64)

    projected = [entry.projected for entry in entries]
    raw_colors = np.asarray([p.color_codes for p in projected], dtype=np.int64).reshape(total, 3, 2)
    color_codes, color_signs = _canonical_signed(raw_colors[:, :, 0], raw_colors[:, :, 1])
    depth = np.fromiter((entry.depth for entry in entries), dtype=np.float64, count=total)
    primitive_id = np.fromiter((p.primitive_id for p in projected), dtype=np.int64, count=total)
    header_count = int(np.count_nonzero(header))
    return FusedPlan(
        tile_bounds=np.ascontiguousarray(bounds, dtype=np.int32),
        tile_offset=offsets,
        flags=header.astype(np.int32),
        ticks=np.ascontiguousarray(ticks, dtype=np.int32),
        jet=np.ascontiguousarray(jet),
        color_codes=np.ascontiguousarray(color_codes, dtype=np.int32),
        color_signs=np.ascontiguousarray(color_signs, dtype=np.int32),
        depth=depth,
        primitive_id=primitive_id,
        routes={"header_log_jet": header_count, "jet_forward_differences": total - header_count},
        tile_pixels=int(np.sum(counts * heights.astype(np.int64) * widths.astype(np.int64))),
        jet_pixels=int(np.sum(np.repeat(heights.astype(np.int64) * widths.astype(np.int64), counts)[~header])),
    )


# ---------------------------------------------------------------------------
# The compositor
# ---------------------------------------------------------------------------


def rasterize(
    bins: Any,
    background: tuple[float, ...],
    depth_limit: Any,
    budget: float,
    early_terminate: bool,
    compare_reference: bool,
    *,
    device: str | None = None,
    header_tolerance_ticks: float = 0.0,
) -> SplatRenderResult | None:
    """Composite every tile in one fused CUDA launch; words identical to the CPU planes.

    Returns ``None`` (never a partial result) when the kernel cannot serve the
    request: no CUDA runtime, the extension does not build, a positive error
    budget with early termination (which keeps per-pixel tail certificates on
    the typed CPU compositor), or a scene outside the kernel's envelope.
    """

    from spf.renderer.splat_terminal import TypedWordView
    from spf.renderer.splats import (
        SplatRenderResult,
        _empty_outputs,
        _reference_core,
        _report,
        _splat_ingress_summary,
    )
    from spf.renderer.transport_terminal import display_values
    from spf.tensor.v02_planes import LogPosPlane, LogSignedPlane

    if early_terminate and budget > 0.0:
        return None
    ok, reason = availability()
    if not ok:
        return None
    try:
        extension = ensure_loaded()
    except RuntimeError:
        return None
    torch = _import_torch()
    resolved = torch.device(device) if device is not None else torch.device("cuda")
    if resolved.type != "cuda":
        return None

    plan_started = time.perf_counter()
    tiles = plan(bins, header_tolerance_ticks=header_tolerance_ticks)
    if tiles is None:
        return None
    plan_ms = 1000.0 * (time.perf_counter() - plan_started)

    started = time.perf_counter()
    H, W = bins.image_shape
    color, transmittance, alpha, depth, ids, coverage = _empty_outputs((H, W))
    background_plane = LogSignedPlane.from_values(np.asarray(background, dtype=np.float64))

    def upload(array: np.ndarray, dtype: Any) -> Any:
        return torch.as_tensor(np.ascontiguousarray(array), dtype=dtype).to(resolved, non_blocking=False)

    rom = np.stack(
        [np.asarray(LOGADD_PLUS, dtype=np.int16), np.asarray(LOGADD_MINUS, dtype=np.int16)], axis=0
    )
    d_rom = upload(rom, torch.int16)
    d_bounds = upload(tiles.tile_bounds, torch.int32)
    d_offset = upload(tiles.tile_offset, torch.int32)
    d_flags = upload(tiles.flags, torch.int32)
    d_ticks = upload(tiles.ticks, torch.int32)
    d_jet = upload(tiles.jet, torch.float64)
    d_color_L = upload(tiles.color_codes, torch.int32)
    d_color_S = upload(tiles.color_signs, torch.int32)
    d_depth = upload(tiles.depth, torch.float64)
    d_pid = upload(tiles.primitive_id, torch.int64)
    d_limit = (
        upload(np.asarray(depth_limit, dtype=np.float64), torch.float64)
        if depth_limit is not None
        else torch.empty(0, dtype=torch.float64, device=resolved)
    )
    d_bg_L = upload(np.asarray(background_plane.L, dtype=np.int32), torch.int32)
    d_bg_S = upload(np.asarray(background_plane.sign, dtype=np.int32), torch.int32)

    # The output planes carry SPF32 words: int16 L codes and int8 signs, the
    # widths ``LogPosPlane``/``LogSignedPlane`` store, so the readback moves the
    # word planes themselves rather than widened copies of them.
    out_color_L = d_bg_L.to(torch.int16).view(1, 1, 3).expand(H, W, 3).contiguous()
    out_color_S = d_bg_S.to(torch.int8).view(1, 1, 3).expand(H, W, 3).contiguous()
    out_trans = torch.zeros((H, W), dtype=torch.int16, device=resolved)
    out_depth = torch.full((H, W), math.inf, dtype=torch.float64, device=resolved)
    out_ids = torch.full((H, W), -1, dtype=torch.int64, device=resolved)
    out_counters = torch.zeros(4, dtype=torch.int64, device=resolved)

    torch.cuda.synchronize(resolved)
    upload_ms = 1000.0 * (time.perf_counter() - started)
    kernel_start = torch.cuda.Event(enable_timing=True)
    kernel_stop = torch.cuda.Event(enable_timing=True)
    kernel_start.record()
    extension.rasterize_tiles_fused(
        d_bounds,
        d_offset,
        d_flags,
        d_ticks,
        d_jet,
        d_color_L,
        d_color_S,
        d_depth,
        d_pid,
        d_limit,
        d_rom,
        d_bg_L,
        d_bg_S,
        out_color_L,
        out_color_S,
        out_trans,
        out_depth,
        out_ids,
        out_counters,
        int(W),
        bool(early_terminate),
    )
    kernel_stop.record()
    torch.cuda.synchronize(resolved)
    kernel_ms = float(kernel_start.elapsed_time(kernel_stop))

    download_started = time.perf_counter()
    color_plane = LogSignedPlane(out_color_L.cpu().numpy(), out_color_S.cpu().numpy())
    t_plane = LogPosPlane(out_trans.cpu().numpy())
    depth[...] = out_depth.cpu().numpy()
    ids[...] = out_ids.cpu().numpy()
    counter_values = out_counters.cpu().numpy()
    download_ms = 1000.0 * (time.perf_counter() - download_started)
    device_ms = 1000.0 * (time.perf_counter() - started)

    color_words = color_plane.to_words()
    twords = t_plane.to_words()
    color[...] = display_values(color_plane)
    transmittance[...] = display_values(t_plane)
    alpha[:] = 1.0 - transmittance
    coverage[:] = alpha

    composited = int(counter_values[0])
    skipped = int(counter_values[1])
    terminated = int(counter_values[2])
    counters = v02.PerformanceCounters()
    ingress = _splat_ingress_summary(bins)
    counters.fallback_conversions = int(ingress["source_conversion_count"])
    counters.jet_samples += tiles.jet_pixels
    counters.values_materialized += tiles.jet_pixels
    counters.transport_compositions += composited
    counters.positive_logadds += composited
    counters.signed_logadds += 3 * composited
    counters.log_ops += 4 * composited
    counters.bytes_read += int(
        tiles.jet.nbytes
        + tiles.ticks.nbytes
        + tiles.color_codes.nbytes
        + tiles.color_signs.nbytes
        + tiles.depth.nbytes
    )
    counters.bytes_written = int(color_words.nbytes + twords.nbytes)

    reference = (
        _reference_core(bins, background=cast(tuple[float, float, float], background), opaque_depth=depth_limit)[0]
        if compare_reference
        else None
    )
    index = int(torch.cuda.current_device() if resolved.index is None else resolved.index)
    info = {
        "device": str(resolved),
        "device_type": "cuda",
        "torch_version": str(torch.__version__),
        "cuda": True,
        "device_name": str(torch.cuda.get_device_name(index)),
        "cuda_version": str(getattr(torch.version, "cuda", None)),
    }
    report = _report(
        backend="cuda_fused",
        bins=bins,
        counters=counters,
        error_budget=budget,
        color=color,
        reference=reference,
        certificates=(),
        representation="spf_log_alpha_fused_tile_transport",
        background=background,
    )
    report = replace(
        report,
        diagnostics={
            **report.diagnostics,
            "execution_class": "device_cuda_fused",
            "actual_execution_class": "device_cuda_fused",
            "actual_callable": CONSUMER,
            "consumer": CONSUMER,
            "terminal": "cuda_fused",
            "hardware_native": True,
            "device": info,
            "device_ms": device_ms,
            "host_plan_ms": plan_ms,
            "upload_ms": upload_ms,
            "kernel_ms": kernel_ms,
            "download_ms": download_ms,
            "kernel_launches": 1,
            "launches_per_frame": 1,
            "tile_blocks": int(tiles.tile_bounds.shape[0]),
            "block_threads": MAX_TILE_DIM * MAX_TILE_DIM,
            "entry_pixels_composited": composited,
            "entry_pixels_skipped": skipped,
            "terminated_pixels": terminated,
            "early_terminate": bool(early_terminate),
            "block_native_arithmetic": bool(ingress["source_conversion_count"] == 0),
            "typed_objects": "lazy_diagnostic_view",
            "alpha_planes": len(tiles.flags),
            "alpha_routes": dict(tiles.routes),
            "display_projection": "magnitude_rom",
            "oracle": "spf.renderer.splat_terminal.rasterize_planes",
            "kernel_source": str(_SOURCE),
            "rom_words": 2 * int(LOGADD_G_ZERO),
            "unavailable_reason": reason,
        },
    )
    return SplatRenderResult(
        color,
        transmittance,
        alpha,
        depth,
        ids,
        coverage,
        report,
        bins,
        TypedWordView(color_words),
        TypedWordView(twords),
        color_words,
        twords,
        (),
    )


__all__ = [
    "CONSUMER",
    "MAX_TILE_DIM",
    "TERMINALS",
    "FusedPlan",
    "availability",
    "available",
    "ensure_loaded",
    "extension_ready",
    "plan",
    "rasterize",
    "runtime_info",
]
