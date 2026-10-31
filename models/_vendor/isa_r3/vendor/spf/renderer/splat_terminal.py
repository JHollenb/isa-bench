"""Bounded native splat transport; diagnostic Python objects are lazy views."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from spf.renderer._jet_kernel import coefficients, native_function, pointer
from spf.tensor import v02

if TYPE_CHECKING:
    from spf.renderer.splats import SplatRenderResult


class TypedWordView:
    """Compatibility view: typed scalar objects are created only when requested."""

    dtype = np.dtype(object)

    def __init__(self, words: np.ndarray):
        self.words = words
        self.shape = words.shape

    @staticmethod
    def scalar(word: Any) -> Any:
        item = v02.unpack_spf32(int(word))
        return v02.LogSigned(item.L, -1) if item.T == 32768 else v02.LogPos(item.L)

    def __getitem__(self, index: Any) -> Any:
        words = self.words[index]
        return self.scalar(words) if np.ndim(words) == 0 else np.asarray(TypedWordView(words))

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        result = np.empty(self.shape, dtype=object)
        for index in np.ndindex(self.shape):
            result[index] = self.scalar(self.words[index])
        return np.asarray(result, dtype=dtype)


class CertificateView(Sequence[Any]):
    def __init__(self, records: tuple[Any, ...]):
        self.records = records
        self.ends = np.cumsum([r[1].size for r in records])

    def __len__(self) -> int:
        return int(self.ends[-1]) if self.ends.size else 0

    def __getitem__(self, index: Any) -> Any:
        from spf.renderer.splats import EarlyTerminationCertificate

        if isinstance(index, slice):
            return tuple(self[i] for i in range(*index.indices(len(self))))
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        tile_index = int(np.searchsorted(self.ends, index, side="right"))
        tile, processed, bounds = self.records[tile_index]
        local = index - (0 if tile_index == 0 else int(self.ends[tile_index - 1]))
        row, col = np.unravel_index(local, processed.shape)
        used = int(processed[row, col])
        error = float(bounds[row, col])
        return EarlyTerminationCertificate(
            tile.index,
            (row + tile.bounds[0], col + tile.bounds[1]),
            used,
            len(tile.entries) - used,
            used < len(tile.entries),
            error,
            error,
        )


def _header_alpha_codes(entry: Any, shape: tuple[int, int], *, tolerance_ticks: float) -> np.ndarray | None:
    """Log-alpha codes from the entry's ``TensorBlockHeaderV2`` log polynomial, or ``None``.

    The header route (ABI v23 ``band_exp`` / ``log_quadratic`` / ``log_mixed``) is
    taken only when the certified rounding bound of the widened jet onto whole
    L ticks is within ``tolerance_ticks`` over the tile; ``0.0`` (the default)
    means header-exact, which keeps word identity with the typed path.  The
    polynomial is evaluated by the library relabel base
    (:class:`spf.tensor.base_models.StbAffinePolyModel`), the same evaluator a
    relabelled block uses, so the splat terminal consumes header structure
    rather than a renderer-private jet.
    """

    from spf.tensor.base_models import StbAffinePolyModel
    from spf.tensor.core import SPFError

    try:
        header, bound = entry.alpha_header(rounding="nearest")
    except (SPFError, ValueError):
        return None
    if bound > tolerance_ticks:
        return None
    active = np.arange(shape[0] * shape[1], dtype=np.int64)
    L, _ = StbAffinePolyModel().sample_planes({"header": header}, shape, (0, 0), active)
    return np.minimum(0, np.asarray(L, dtype=np.int64).reshape(shape))


def _alpha_code_planes(
    tile: Any,
    counters: v02.PerformanceCounters,
    *,
    header_tolerance_ticks: float = 0.0,
    routes: dict[str, int] | None = None,
) -> np.ndarray:
    """Log-alpha codes ``(entries, rows, cols)`` per entry, header structure first.

    Each entry is tried as ``TensorBlockHeaderV2`` log structure
    (:func:`_header_alpha_codes`); when the widened jet is not header-exact
    within ``header_tolerance_ticks`` the v0.2 ``jet_forward_differences``
    recurrence runs on the integer-tick jet and the codes are the rounded
    ticks divided by ``2**guard_L``.  Both are clamped to ``<= 0`` (alpha never
    exceeds one), exactly ``splats._logpos_from_wide``.  ``routes`` receives the
    per-route entry counts (``header_log_jet`` / ``jet_forward_differences``).
    """

    row0, col0, row1, col1 = tile.bounds
    shape = (row1 - row0, col1 - col0)
    codes = np.empty((len(tile.entries), *shape), dtype=np.int64)
    for index, entry in enumerate(tile.entries):
        from_header = _header_alpha_codes(entry, shape, tolerance_ticks=header_tolerance_ticks)
        if from_header is not None:
            codes[index] = from_header
            if routes is not None:
                routes["header_log_jet"] = routes.get("header_log_jet", 0) + 1
            continue
        wide = v02.jet_forward_differences(entry.spf_log_alpha_jet, shape, origin=(0.0, 0.0), counters=counters)
        guard = float(1 << int(entry.spf_log_alpha_jet.guard_L))
        codes[index] = np.minimum(0, np.rint(np.real(wide) / guard).astype(np.int64))
        if routes is not None:
            routes["jet_forward_differences"] = routes.get("jet_forward_differences", 0) + 1
    return codes


def rasterize_planes(
    bins: Any,
    background: tuple[float, ...],
    depth_limit: Any,
    budget: float,
    early_terminate: bool,
    compare_reference: bool,
) -> SplatRenderResult | None:
    """Composite every tile on LOG_POS/LOG_SIGNED word planes (design §27).

    Per tile the log-alpha jets are materialized as one ``(entries, h, w)``
    code plane, complemented with the ``LOGADD_MINUS`` ROM
    (``LogPosPlane.log1m``), multiplied by the colour codes and composed
    front-to-back with :func:`spf.renderer.transport_terminal.compose_front_to_back`
    (``C' = C + T*c``, ``T' = T*t``: the same operations
    :func:`spf.tensor.v02.transport_compose` applies segment by segment).
    Nothing is decoded: the float image is the magnitude-ROM display
    projection of the output words.

    Early termination needs per-pixel tail certificates and stays on the
    typed per-pixel path, so a positive budget returns ``None``.
    """

    from spf.renderer.splats import (
        SplatRenderResult,
        _empty_outputs,
        _reference_core,
        _report,
        _splat_ingress_summary,
    )
    from spf.renderer.transport_terminal import compose_front_to_back, display_values
    from spf.tensor.v02_planes import LogPosPlane, LogSignedPlane

    if early_terminate and budget > 0.0:
        return None
    shape = bins.image_shape
    color, transmittance, alpha, depth, ids, coverage = _empty_outputs(shape)
    background_plane = LogSignedPlane.from_values(np.asarray(background, dtype=np.float64))
    color_L = np.broadcast_to(background_plane.L, (*shape, 3)).copy()
    color_sign = np.broadcast_to(background_plane.sign, (*shape, 3)).copy()
    t_L = np.zeros(shape, dtype=np.int16)  # LOG_POS code 0 == unit transmittance
    counters = v02.PerformanceCounters()
    ingress = _splat_ingress_summary(bins)
    counters.fallback_conversions = int(ingress["source_conversion_count"])
    alpha_planes = 0
    alpha_routes: dict[str, int] = {"header_log_jet": 0, "jet_forward_differences": 0}
    for tile in bins.tiles:
        y0, x0, y1, x1 = tile.bounds
        tile_shape = (y1 - y0, x1 - x0)
        n = len(tile.entries)
        codes = _alpha_code_planes(tile, counters, routes=alpha_routes)
        if depth_limit is not None:
            depths = np.asarray([entry.depth for entry in tile.entries], dtype=np.float64)
            occluded = depths[:, None, None] >= np.asarray(depth_limit[y0:y1, x0:x1], dtype=np.float64)[None]
            codes = np.where(occluded, v02.SPF32_L_ZERO, codes)
        alpha_plane = LogPosPlane(codes)
        alpha_planes += n
        step_t = alpha_plane.log1m()
        color_codes = np.asarray([entry.projected.color_codes for entry in tile.entries], dtype=np.int64)
        colors = LogSignedPlane(color_codes[:, None, None, :, 0], color_codes[:, None, None, :, 1])
        step_c = LogSignedPlane(alpha_plane.L[..., None], np.ones((n, *tile_shape, 1), dtype=np.int8)).mul(colors)
        total_t, total_c = compose_front_to_back(step_t, step_c)
        out = total_c.add(background_plane.mul(LogPosPlane(total_t.L[..., None])))
        target = np.s_[y0:y1, x0:x1]
        color_L[target] = out.L
        color_sign[target] = out.sign
        t_L[target] = total_t.L
        live = alpha_plane.L != v02.SPF32_L_ZERO
        any_live = np.any(live, axis=0)
        first = np.argmax(live, axis=0)
        depths = np.asarray([entry.depth for entry in tile.entries], dtype=np.float64)
        primitive_ids = np.asarray([entry.projected.primitive_id for entry in tile.entries], dtype=np.int64)
        depth[target] = np.where(any_live, depths[first], depth[target])
        ids[target] = np.where(any_live, primitive_ids[first], ids[target])
        counters.transport_compositions += int(n * math.prod(tile_shape))
        counters.positive_logadds += int(n * math.prod(tile_shape))
        counters.signed_logadds += int(3 * n * math.prod(tile_shape))
        counters.log_ops += int(4 * n * math.prod(tile_shape))
        counters.bytes_read += int(codes.nbytes + color_codes.nbytes)
    color_plane = LogSignedPlane(color_L, color_sign)
    t_plane = LogPosPlane(t_L)
    color_words = color_plane.to_words()
    twords = t_plane.to_words()
    color[...] = display_values(color_plane)
    transmittance[...] = display_values(t_plane)
    alpha[:] = 1.0 - transmittance
    coverage[:] = alpha
    counters.bytes_written = int(color_words.nbytes + twords.nbytes)
    reference = (
        _reference_core(bins, background=cast(tuple[float, float, float], background), opaque_depth=depth_limit)[0]
        if compare_reference
        else None
    )
    report = _report(
        backend="plane_python",
        bins=bins,
        counters=counters,
        error_budget=budget,
        color=color,
        reference=reference,
        certificates=(),
        representation="spf_log_alpha_plane_transport",
        background=background,
    )
    report = replace(
        report,
        diagnostics={
            **report.diagnostics,
            "execution_class": "plane_python",
            "actual_execution_class": "plane_python",
            "actual_callable": "spf.renderer.splat_terminal.rasterize_planes",
            "consumer": "spf.renderer.transport_terminal.compose_front_to_back",
            "hardware_native": False,
            "block_native_arithmetic": bool(ingress["source_conversion_count"] == 0),
            "typed_objects": "lazy_diagnostic_view",
            "alpha_planes": alpha_planes,
            "alpha_routes": dict(alpha_routes),
            "display_projection": "magnitude_rom",
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


def rasterize_native(
    bins: Any,
    background: tuple[float, ...],
    depth_limit: Any,
    budget: float,
    early_terminate: bool,
    compare_reference: bool,
) -> SplatRenderResult | None:
    from spf.renderer.splats import (
        SplatRenderResult,
        _empty_outputs,
        _reference_core,
        _report,
        _splat_ingress_summary,
        _typed_background,
    )

    ptr = ctypes.c_void_p
    fn = native_function(
        "spf_sfr_transport_tile",
        tuple([ptr] * 6 + [ctypes.c_size_t, ctypes.c_int, ctypes.c_int] + [ptr, ptr, ctypes.c_double] + [ptr] * 8),
    )
    if fn is None:
        return None
    color, transmittance, alpha, depth, ids, coverage = _empty_outputs(bins.image_shape)
    bg = np.array([v02.pack_spf32(_typed_background(v)) for v in background], dtype=np.uint32)
    color_words = np.broadcast_to(bg, (*bins.image_shape, 3)).copy()
    twords = np.zeros(bins.image_shape, dtype=np.uint32)  # packed unity
    color[:] = [float(_typed_background(v).value) for v in background]
    transmittance.fill(1)
    records = []
    counters = v02.PerformanceCounters()
    counters.fallback_conversions = int(_splat_ingress_summary(bins)["source_conversion_count"])
    maximum_early = 0.0
    terminated = 0
    for tile in bins.tiles:
        y0, x0, y1, x1 = tile.bounds
        shape = (y1 - y0, x1 - x0)
        jets = np.asarray([coefficients(e.spf_log_alpha_jet) for e in tile.entries], dtype=np.float64)
        scales = np.asarray([2.0**-e.spf_log_alpha_jet.guard_L for e in tile.entries])
        colors = np.asarray(
            [[v02.pack_spf32(_typed_background(v)) for v in e.projected.color] for e in tile.entries], dtype=np.uint32
        )
        depths = np.asarray([e.depth for e in tile.entries], dtype=np.float64)
        primitive_ids = np.asarray([e.projected.primitive_id for e in tile.entries], dtype=np.int64)
        # Conservative suffix opacity/color bounds are independent of pixels.
        # Inflation includes alpha/color quantization; the report separately
        # accounts for the retained computation's quantizer error.
        terms = np.asarray([e.max_alpha * e.projected.max_color_abs for e in tile.entries]) * 1.01
        tail = np.r_[np.cumsum(terms[::-1])[::-1], 0.0] + max(map(abs, background))
        opaque = None if depth_limit is None else np.ascontiguousarray(depth_limit[y0:y1, x0:x1])
        outputs = [
            np.empty((*shape, 3)),
            np.empty(shape),
            np.empty((*shape, 3), dtype=np.uint32),
            np.empty(shape, dtype=np.uint32),
            np.empty(shape),
            np.empty(shape, dtype=np.int64),
            np.empty(shape, dtype=np.uint32),
            np.empty(shape),
        ]
        fn(
            *(pointer(a) for a in (jets, scales, colors, depths, primitive_ids, tail)),
            len(tile.entries),
            *shape,
            pointer(opaque),
            pointer(bg),
            budget if early_terminate else 0.0,
            *(pointer(a) for a in outputs),
        )
        c, t, cw, tw, d, identifier, processed, bounds = outputs
        target = np.s_[y0:y1, x0:x1]
        color[target], transmittance[target], color_words[target], twords[target] = c, t, cw, tw
        depth[target], ids[target] = d, identifier
        records.append((tile, processed, bounds))
        terminated += int(np.count_nonzero(processed < len(tile.entries)))
        maximum_early = max(maximum_early, float(bounds.max(initial=0)))
        counters.transport_compositions += int(processed.sum())
        counters.pivot_reductions += int(np.prod(shape)) * 3
        counters.bytes_read += sum(a.nbytes for a in (jets, scales, colors, depths, primitive_ids, tail))
    alpha[:] = 1 - transmittance
    coverage[:] = alpha
    counters.bytes_written = sum(
        a.nbytes for a in (color, transmittance, color_words, twords, alpha, depth, ids, coverage)
    )
    reference = (
        _reference_core(bins, background=cast(tuple[float, float, float], background), opaque_depth=depth_limit)[0]
        if compare_reference
        else None
    )
    report = _report(
        backend="native_cpu",
        bins=bins,
        counters=counters,
        error_budget=budget,
        color=color,
        reference=reference,
        certificates=(),
        representation="spf_log_alpha_pivot_transport",
        background=background,
    )
    report = replace(
        report,
        ledger=report.ledger.plus(v02.ErrorLedger(pruning=maximum_early)),
        diagnostics={
            **report.diagnostics,
            "execution_class": "native_cpu",
            "actual_execution_class": "native_cpu",
            "actual_callable": "spf.renderer.splat_terminal.rasterize_native",
            "hardware_native": True,
            "early_terminated_pixels": terminated,
            "max_early_termination_bound": maximum_early,
            "native_submissions": len(bins.tiles),
            "scratch_extent": (bins.tile_size, bins.tile_size),
            "typed_objects": "lazy_diagnostic_view",
            "alpha_planes": 0,
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
        CertificateView(tuple(records)),
    )
