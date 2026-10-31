"""Public Field terminal lowering shared with native projections and Scene bindings."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from spf.renderer.carriers import decode_words, encode_words
from spf.renderer.model import Coherence, CoordinateChart, Domain, RepresentationKind, SupportKind


def prepare_terminal(field: Any, request: Any) -> Any:
    """Select a fused consumer only when its sampling and error contracts apply.

    Arbitrary point queries, non-closed suffixes and typed object outputs retain
    the addressed/reference consumer. No coordinate or complex sample plane is
    constructed to prepare a regular-grid terminal.
    """
    from spf.renderer.block_filter import FilteredBlockView, prepare_filtered_terminal
    from spf.renderer.blocks import BlockView
    from spf.renderer.jets import AdaptiveJetField, SpectralJetTile2D
    from spf.renderer.native import NativeGrid2D, NativeRenderer
    from spf.renderer.runtime import MaterializationError
    from spf.tensor import v02

    if isinstance(field.representation.payload, FilteredBlockView):
        return prepare_filtered_terminal(field, request)
    if isinstance(field.representation.payload, BlockView):
        return prepare_block_terminal(field, request)
    from spf.renderer.mixture_terminal import prepare_atom_grid

    atom_grid = prepare_atom_grid(field, request)
    if atom_grid is not None:
        return atom_grid

    layout = request.layout
    if (
        field.operations
        or field.domain is not Domain.SPATIAL_2D
        or layout is None
        or len(layout.shape) != 2
        or layout.order != "C"
        or request.region is not None
        or field.representation.residual is not None
        or field.support.kind not in {SupportKind.FULL, SupportKind.EMPTY, SupportKind.AABB}
    ):
        return None
    if field.coherence is not Coherence.INCOHERENT_POSITIVE or request.output_format not in {"native", "real"}:
        return None
    if str(request.output_kind) not in {"dense", "any"}:
        return None
    payload = field.representation.payload
    if not isinstance(payload, (v02.GaussianChirpAtom, SpectralJetTile2D, AdaptiveJetField)):
        return None
    # A positive tag is a contract, not permission to discard source phase.
    if isinstance(payload, v02.GaussianChirpAtom):
        if complex(payload.kappa).imag or np.any(np.imag(payload.h)) or np.any(np.imag(payload.C)):
            return None
    elif any(
        any(
            float(getattr(tile.phase, name)) != 0
            for name in ("value", "grad_x", "grad_y", "hess_xx", "hess_xy", "hess_yy")
        )
        or (tile.phase_residual is not None and tile.phase_residual.max_abs_value > 0)
        for tile in (payload.tiles if isinstance(payload, AdaptiveJetField) else (payload,))
    ):
        return None
    chart = field.chart
    if field.representation.metadata.get("chart_baked") and field.support.kind in {SupportKind.FULL, SupportKind.EMPTY}:
        field = replace(field, chart=CoordinateChart.identity(2))
        chart = field.chart
    if chart.dimension and chart != CoordinateChart.identity(2):
        # Atom coordinates can be folded into the header without sampling.
        if not isinstance(payload, v02.GaussianChirpAtom) or field.support.kind is SupportKind.AABB:
            return None
        matrix = np.asarray(chart.basis) @ np.diag(np.asarray(chart.scale) * np.asarray(chart.sample_spacing))
        payload = v02.atom_affine(payload, matrix, np.asarray(chart.origin))
        field = replace(
            field, chart=CoordinateChart.identity(2), representation=replace(field.representation, payload=payload)
        )
    grid = NativeGrid2D(layout.shape, origin=layout.origin, spacing=layout.spacing)
    # Paired/adaptive tiles are already fitted in sample-index space.
    if field.representation.kind is RepresentationKind.QUADRATIC_JET:
        grid = NativeGrid2D(layout.shape)
    try:
        prepared = NativeRenderer(precision=request.precision, retain_codes=False).prepare_projection(
            field, grid, semantic="radiance"
        )
    except (NotImplementedError, MaterializationError):
        return None
    bound = float(prepared.report.ledger.total_bound)
    # Relative request checks use the native pointwise source bound, not a
    # guessed unit scale. The field contract was checked by prepare_projection.
    scale = max(float(prepared.report.diagnostics["output_scale"]), np.finfo(float).tiny)
    if (
        not math.isfinite(bound)
        or bound > request.error_budget.absolute
        or bound / scale > request.error_budget.relative
    ):
        return None
    return prepared


class PreparedBlockTerminal:
    """Compact code carriers and an address map, decoded only at output time."""

    def __init__(
        self, view: Any, request: Any, addresses: np.ndarray, words: np.ndarray, report: Any, dtype: Any
    ) -> None:
        self.view, self.request = view, request
        self.addresses, self.words, self.report = addresses, words, report
        self.dtype = np.dtype(dtype)
        self.logs = (words >> 16).astype(np.int16)
        self.nan = words == 0x7FFFFFFF
        # Header arithmetic remains wide. These are log codes, never cached pixels.
        self.exponents = (self.logs.astype(np.float64) + view.log_offset) / 128
        self.dense = addresses.size == request.layout.element_count and (
            addresses.size == 0
            or (addresses[0] == 0 and addresses[-1] == addresses.size - 1 and np.all(np.diff(addresses) == 1))
        )
        self.constant_intensity = (
            view.observable == "intensity"
            and request.output_format in {"native", "real", "complex"}
            and words.size > 0
            and not np.any(self.nan)
            and np.all(self.logs == self.logs[0])
        )
        if self.constant_intensity:
            self.logs, self.words = self.logs[:1].copy(), self.words[:1].copy()
            self.exponents, self.nan = self.exponents[:1].copy(), self.nan[:1].copy()
        if self.dense:
            self.addresses = np.empty(0, dtype=np.int64)
        carrier_bytes = sum(a.nbytes for a in (self.addresses, self.words, self.logs, self.nan, self.exponents))
        self.report = replace(
            report,
            metrics=replace(
                report.metrics,
                bytes_read=carrier_bytes,
                materialized_intermediate_elements=self.logs.size,
                materialized_intermediate_bytes=self.logs.size * 8,
                peak_memory=carrier_bytes + report.metrics.terminal_output_bytes + self.logs.size * 16,
                logical_peak_bytes=carrier_bytes + report.metrics.terminal_output_bytes + self.logs.size * 16,
            ),
            diagnostics={
                **report.diagnostics,
                "prepared_carrier_bytes": carrier_bytes,
                "decoded_code_count": int(self.logs.size),
                "constant_log_magnitude": self.constant_intensity,
            },
        )
        for array in (self.addresses, self.words, self.logs, self.nan, self.exponents):
            array.setflags(write=False)

    def execute(self) -> Any:
        from spf.renderer.native import NativeProjectionResult
        from spf.tensor import v02

        layout = self.request.layout
        fill = v02.SPF32_WORD_ZERO if self.dtype == np.uint32 else 0
        output = (
            np.empty(layout.shape, dtype=self.dtype, order=layout.order)
            if self.dense
            else np.full(layout.shape, fill, dtype=self.dtype, order=layout.order)
        )
        if self.constant_intensity:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                value = np.exp2(self.exponents[0])
            if self.dense:
                output.fill(value)
            else:
                output.ravel(order=layout.order)[self.addresses] = value
            return NativeProjectionResult(output, self.report, self.logs)
        if self.view.observable == "intensity" and self.request.output_format in {"native", "real", "complex"}:
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values = np.exp2(self.exponents)
            if np.any(self.nan):
                values[self.nan] = np.nan
        else:
            # The address mask was consumed in the map, so projecting needs no
            # per-frame membership join or dense source plane.
            values = self.view.project_words(self.words, self.addresses)
            if self.request.output_format in {"complex", "real"} and values.dtype == np.uint32:
                values = decode_words(values)
            if self.request.output_format == "real":
                values = np.real(values)
            elif self.request.output_format == "spf32" and values.dtype != np.uint32:
                values = encode_words(values)
            elif self.dtype.kind in "fc" and values.dtype == np.uint32:
                values = decode_words(values)
        if self.dense:
            output.ravel(order=layout.order)[:] = values
        else:
            output.ravel(order=layout.order)[self.addresses] = values
        return NativeProjectionResult(output, self.report, self.logs)


def prepare_block_terminal(field: Any, request: Any) -> PreparedBlockTerminal | None:
    """Compile support, residual codecs and scalar projections once for a grid.

    Preparation checks the existing semantic/error boundary on the active
    samples. Every execution allocates the requested output and decodes fresh
    values from immutable code carriers; it never caches a rendered image.
    """
    from spf.renderer import runtime
    from spf.renderer.model import SampleLayout
    from spf.renderer.report import RenderMetrics, RenderReport
    from spf.tensor import v02

    view, layout = field.representation.payload, request.layout
    folded: list[str] = []
    for operation in field.operations:
        # Closed header operations (exposure, phase shift, mask) compose on
        # the view without touching residual words; anything else is a
        # suffix for the runtime's word executor, so this fused terminal
        # declines instead of decoding.
        try:
            view = view.apply(operation)
        except (ValueError, KeyError, TypeError):
            return None
        folded.append(operation.op)
    if (
        field.representation.residual is not None
        or request.region is not None
        or layout is None
        or len(layout.shape) != len(view.shape)
        or field.support.kind not in {SupportKind.FULL, SupportKind.EMPTY}
        or field.chart != CoordinateChart.identity(len(view.shape))
        or view.scalar_bits != 32
        or request.output_format not in {"native", "real", "complex", "spf32"}
        or str(request.output_kind) not in {"dense", "any"}
    ):
        return None
    # Sampling non-integer coordinates is an error in the block contract. Keep
    # that validation in the generic path instead of rounding a fractional grid.
    if any(x != round(x) for x in (*layout.origin, *layout.spacing)):
        return None
    if any(
        o < 0 or (n and o + (n - 1) * step >= dim)
        for o, n, step, dim in zip(layout.origin, layout.shape, layout.spacing, view.shape, strict=True)
    ):
        return None
    source_addresses = (
        view.active_addresses() if field.support.kind is not SupportKind.EMPTY else np.empty(0, dtype=np.int64)
    )
    if (
        layout.shape == view.shape
        and not any(layout.origin)
        and all(s == 1 for s in layout.spacing)
        and layout.order == "C"
    ):
        destination = source_addresses.copy()
    else:
        coordinates = np.asarray(np.unravel_index(source_addresses, view.shape)).T
        grid = (coordinates - np.asarray(layout.origin)) / np.asarray(layout.spacing)
        selected = np.all((grid == np.rint(grid)) & (grid >= 0) & (grid < np.asarray(layout.shape)), axis=-1)
        source_addresses = source_addresses[selected]
        destination = np.ravel_multi_index(tuple(grid[selected].astype(np.int64).T), layout.shape, order=layout.order)
    words = view.sample_words(source_addresses, phase=view.observable != "intensity")
    live = (words >> 16) != 32768
    words, destination, source_addresses = words[live], destination[live], source_addresses[live]
    active_view = replace(view, mask_indices=None)
    sampled = active_view.project_words(words, source_addresses)
    # Reuse the authoritative formatter and budget checks on K active samples,
    # plus one inactive zero, which is needed for relative-scale/empty semantics.
    zero = v02.SPF32_WORD_ZERO if sampled.dtype == np.uint32 else 0
    sampled = np.concatenate((sampled, np.asarray([zero], dtype=sampled.dtype)))
    native_values, source_values, source_kind = runtime._semantic_source(field, sampled)
    context = runtime._SamplingContext(np.empty((0, 1)), (sampled.size,), "C", SampleLayout((sampled.size,)), None)
    formatted, observed, phase_error, format_ledger, _ = runtime._format_output(
        field,
        request,
        native_values,
        source_values,
        source_kind,
        context,
        (),
    )
    from spf.renderer.normalization import _field_ledger_after_operations

    ledger = runtime._apply_representation_error(field, _field_ledger_after_operations(field))
    ledger = runtime._ledger_add(
        ledger,
        source_quantization=format_ledger.source_quantization,
        materialization=format_ledger.materialization,
        display=format_ledger.display,
    )
    runtime._enforce_error_budget(field, request, ledger, source_values, phase_error)
    result = v02.MaterializationResult(formatted, max(observed, ledger.total_bound), 0.0, ledger)
    count, active = layout.element_count, words.size
    output_bytes = count * result.values.dtype.itemsize
    carrier_bytes = destination.nbytes + words.nbytes + active * (2 + 1 + 8)
    metrics = RenderMetrics(
        backend=request.backend,
        workload="block_projection",
        resolution=layout.shape,
        representation="DENSE_TYPED",
        precision=str(request.precision.mode),
        error_budget=request.error_budget.absolute,
        bytes_read=carrier_bytes,
        bytes_written=output_bytes,
        potential_elements=count,
        materialized_elements=active,
        potential_intermediate_elements=count,
        materialized_intermediate_elements=active,
        potential_intermediate_bytes=count * 16,
        materialized_intermediate_bytes=active * 8,
        terminal_output_elements=count,
        terminal_output_bytes=output_bytes,
        peak_memory=carrier_bytes + output_bytes + active * 16,
        logical_peak_bytes=carrier_bytes + output_bytes + active * 16,
    )
    report = RenderReport(
        metrics,
        ledger=ledger,
        capabilities_used=(
            "prepared_block_terminal",
            "addressed_block_consumer",
            "active_address_scatter",
            *(f"header_{op}" for op in folded),
            "intensity_from_log_codes" if view.observable == "intensity" else "packed_block_projection",
        ),
        diagnostics={
            "active_source_samples": int(active),
            "terminal_output_elements": count,
            "output_scale": runtime._max_abs(source_values),
            "dense_coordinate_queries": 0,
            "cached_pixels": 0,
            "prepared_carrier_bytes": carrier_bytes,
            "output_observed_error": result.absolute_error_bound,
            "scalar_bits": view.scalar_bits,
            "source_block_type": type(view).__name__,
            "preparation_sampling_api": "sample_words",
            "folded_header_operations": tuple(folded),
            "preparation_fallback_policy": (
                "addressed tensor model and residuals"
                if type(view).__name__ == "TensorBlockView"
                else "addressed built-ins; unsupported model, codec or expression uses its reference decoder"
            ),
            "byte_traffic_status": "logical_model",
        },
    )
    return PreparedBlockTerminal(active_view, request, destination, words, report, result.values.dtype)
