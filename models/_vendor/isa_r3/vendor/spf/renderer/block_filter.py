"""Compose analytic spectral operators with addressed SPF block carriers.

A frequency block keeps its packed source and adds full log/phase jets only at
requested addresses. Spatial blocks select exact characters, separable source axes, or matrix
diagonal synthesis before considering a dense periodic transform. The dense
fallback applies the compact operator without constructing a transfer plane.
No operation prefix constructs a Cartesian field or a dense transfer plane.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf.renderer.blocks import BlockView, V3BlockView
from spf.renderer.model import CoordinateChart, Domain, SampleLayout, SupportKind
from spf.renderer.transfers import SpectralTransfer

CHUNK = 65536
_C64 = np.dtype("complex64")
_C128 = np.dtype("complex128")
FOURIER = {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}
SPATIAL = {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}
FILTER_OPS = {
    "translate",
    "gaussian_filter",
    "exposure",
    "phase",
    "phase_shift",
    "glare",
    "fresnel",
    "spectral_lens",
    "lod",
}


def chart_matrix(chart: CoordinateChart, dimension: int) -> np.ndarray:
    if chart.dimension == 0:
        return np.eye(dimension)
    matrix = np.asarray(chart.basis) @ np.diag(np.asarray(chart.scale) * np.asarray(chart.sample_spacing))
    if matrix.shape != (dimension, dimension) or np.linalg.matrix_rank(matrix) != dimension:
        raise ValueError("block filter requires an invertible coordinate chart")
    return matrix


def source_planes(view: BlockView, words: np.ndarray, addresses: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    """Physical log2/radian state with sentinels, before Cartesian projection."""
    if isinstance(view, FilteredBlockView):
        return view.planes(words, addresses)
    from spf.scalar._constants import SPF32
    from spf.tensor.core import unpack_L, unpack_theta

    fmt = view.block.representation.format if isinstance(view, V3BlockView) else SPF32
    logs = unpack_L(words, fmt)
    zero, nan = logs == fmt.l_code_zero, words == fmt.word_nan
    magnitude = logs.astype(float) / fmt.l_scale + view.log_offset / 256
    phase = unpack_theta(words, fmt).astype(float) * (2 * math.pi / (1 << fmt.theta_bits))
    phase += view.phase_offset * (2 * math.pi / 65536)
    if view.observable == "intensity":
        magnitude *= 2
        phase = np.zeros_like(magnitude)
    if view.mask_indices is not None:
        zero = zero | ~np.isin(addresses, view.mask_indices)
    return magnitude, phase, zero, nan


_SPF32_RELATIVE_QUANTIZATION = (2.0 ** (1.0 / 512.0) - 1.0) + math.pi / 65536.0


def pack_planes(logs: np.ndarray, phase: np.ndarray, zero: Any, nan: Any) -> tuple[np.ndarray, float]:
    """Quantize physical log2/radian planes once into packed SPF32 words.

    This is the DenseSPF32 projection of an analytic transfer (design §7.3):
    ``L = rint(256·log2)``, ``Θ = rint(65536·φ/2π) mod 65536`` through
    :func:`spf.tensor.core.pack_words`.  Returns the words and the absolute
    quantization bound ``(2^(1/512) − 1 + π/65536)·max|value|``; the bound is
    infinite when a finite sample saturates the L range.
    """
    from spf.tensor import v02
    from spf.tensor.core import pack_words

    logs = np.asarray(logs, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    zero = np.broadcast_to(np.asarray(zero, dtype=bool), logs.shape)
    nan = np.broadcast_to(np.asarray(nan, dtype=bool), logs.shape)
    live = ~zero & ~nan & np.isfinite(logs)
    scaled = np.rint(logs * 256.0)
    saturated = live & ((scaled < v02.SPF32_L_MIN_FINITE) | (scaled > v02.SPF32_L_MAX_FINITE))
    L = np.clip(np.where(live, scaled, 0.0), v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE).astype(np.int16)
    T = (np.rint(phase * (65536.0 / (2.0 * math.pi))).astype(np.int64) & 65535).astype(np.uint16)
    words = np.asarray(pack_words(L, np.where(live, T, 0).astype(np.uint16)), dtype=np.uint32)
    words = np.where(live, words, np.uint32(v02.SPF32_WORD_ZERO))
    words = np.where(nan & ~zero, np.uint32(v02.SPF32_WORD_NAN), words)
    if np.any(saturated) or np.any(nan & ~zero):
        return np.asarray(words, dtype=np.uint32), math.inf
    if not np.any(live):
        return np.asarray(words, dtype=np.uint32), 0.0
    top = float(np.max(logs[live]))
    try:
        bound = math.exp2(top) * _SPF32_RELATIVE_QUANTIZATION
    except OverflowError:
        bound = math.inf
    return np.asarray(words, dtype=np.uint32), bound


def project_planes(logs: np.ndarray, phase: np.ndarray, zero: Any, nan: Any, *, real: bool = False) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        # Inactive words must remain zero even when a wide transfer overflows.
        magnitude = np.exp2(np.where(zero, -np.inf, logs))
        values = magnitude if real else magnitude * np.exp(1j * np.remainder(phase, 2 * math.pi))
    return np.where(nan & ~zero, np.nan, np.where(zero, 0, values))


@dataclass(frozen=True)
class FilteredBlockView(BlockView):
    source: BlockView
    transfer: SpectralTransfer
    frequency_domain: bool
    chart: CoordinateChart
    observable: str = "complex"
    log_offset: float = 0.0
    phase_offset: float = 0.0
    mask_indices: np.ndarray | None = None
    shape: tuple[int, ...] = dataclass_field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", self.source.shape)
        if self.source.log_offset or self.source.phase_offset:
            from spf.renderer.model import FieldOperation

            # Global factors commute with a linear filter. Absorb them before
            # a spatial source FFT, so opposite wide exposures can cancel
            # without overflowing an intermediate Cartesian source.
            source, transfer = self.source, self.transfer
            if source.log_offset:
                ev = source.log_offset / (128 if source.observable == "intensity" else 256)
                transfer = transfer.apply(FieldOperation("exposure", {"ev": ev}))
            if source.phase_offset:
                transfer = transfer.apply(FieldOperation("phase", {"dT": source.phase_offset}))
            object.__setattr__(self, "source", source._replace(log_offset=0.0, phase_offset=0.0))
            object.__setattr__(self, "transfer", transfer)

    @classmethod
    def from_field(cls, source: BlockView, field: Any) -> FilteredBlockView:
        if field.domain not in FOURIER | SPATIAL:
            raise ValueError("block filtering needs an explicit spatial or Fourier domain")
        chart_matrix(field.chart, len(source.shape))
        return cls(source, SpectralTransfer.identity(len(source.shape)), field.domain in FOURIER, field.chart)

    @property
    def scalar_bits(self) -> int:
        return self.source.scalar_bits

    @property
    def nbytes(self) -> int:
        return (
            self.source.nbytes + self.transfer.nbytes + (0 if self.mask_indices is None else self.mask_indices.nbytes)
        )

    def apply(self, operation: Any) -> BlockView:
        if operation.op == "mask":
            return super().apply(operation)
        if operation.op not in FILTER_OPS:
            raise ValueError(f"unsupported filtered block operation {operation.op!r}")
        if self.observable == "intensity" or self.mask_indices is not None:
            # Filtering after abs2/masking acts on a different field. Preserve
            # the stage boundary rather than commuting it through convolution.
            scalar_phase = (
                np.asarray(
                    operation.params.get("dT", operation.params.get("phase", operation.params.get("delta", 0)))
                ).ndim
                == 0
            )
            if operation.op == "exposure" or (operation.op in {"phase", "phase_shift"} and scalar_phase):
                return super().apply(operation)
            outer = FilteredBlockView(
                self, SpectralTransfer.identity(len(self.shape)), self.frequency_domain, self.chart
            )
            return outer.apply(operation)
        if operation.op == "lod" or (
            operation.op in {"phase", "phase_shift"}
            and np.asarray(
                operation.params.get("dT", operation.params.get("phase", operation.params.get("delta", 0)))
            ).ndim
        ):
            from spf.renderer.model import FieldOperation

            operation = FieldOperation(operation.op, {**operation.params, "shape": self.shape})
        return replace(self, transfer=self.transfer.apply(operation))

    def active_addresses(self) -> np.ndarray:
        addresses = self.source.active_addresses() if self.frequency_domain else np.arange(math.prod(self.shape))
        return np.intersect1d(addresses, self.mask_indices) if self.mask_indices is not None else addresses

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        if not self.frequency_domain:
            raise ValueError("spatial convolution has an explicit FFT terminal, not a packed pointwise sample")
        # These are the source carrier words. Wide operator headers are applied
        # by project_words, once, after all compositions and before encoding.
        return self.source.sample_words(linear, phase=phase)

    def coordinates(self, linear: np.ndarray, *, periodic: bool = False) -> np.ndarray:
        indices = np.asarray(np.unravel_index(linear, self.shape)).T.astype(float)
        matrix = chart_matrix(self.chart, len(self.shape))
        if periodic:
            # np.fft.fftfreq convention, including negative even-size Nyquist.
            dims = np.asarray(self.shape)
            indices = np.where(indices >= (dims + 1) // 2, indices - dims, indices) / dims
            return np.asarray(indices @ np.linalg.inv(matrix))
        origin = np.asarray(self.chart.origin) if self.chart.dimension else np.zeros(len(self.shape))
        return np.asarray(indices @ matrix.T + origin)

    def planes(
        self,
        words: np.ndarray,
        addresses: np.ndarray,
        *,
        periodic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, Any, Any]:
        logs, phases, zero, nan = source_planes(self.source, words, addresses)
        dl, dt = self.transfer.log_phase(
            self.coordinates(addresses, periodic=periodic),
            addresses,
            include_phase=self.observable != "intensity",
        )
        logs, phases = logs + dl, phases + dt
        if self.observable == "intensity":
            logs, phases = 2 * logs, np.zeros_like(phases)
        logs = logs + self.log_offset / (128 if self.observable == "intensity" else 256)
        phases = phases + self.phase_offset * (2 * math.pi / 65536)
        if self.mask_indices is not None:
            zero = zero | ~np.isin(addresses, self.mask_indices)
        return logs, phases, zero, nan

    def project_words(
        self,
        result: np.ndarray,
        linear: np.ndarray,
        *,
        periodic: bool = False,
    ) -> np.ndarray:
        shape = np.shape(linear)
        planes = self.planes(
            np.asarray(result).reshape(-1),
            np.asarray(linear).reshape(-1),
            periodic=periodic,
        )
        return project_planes(*planes, real=self.observable == "intensity").reshape(shape)

    def project_words_packed(self, result: np.ndarray, linear: np.ndarray) -> tuple[np.ndarray, float]:
        """Packed SPF32 projection of the transferred source words, with its bound."""
        if self.observable != "complex":
            raise ValueError("packed projection needs a complex observable")
        shape = np.shape(linear)
        planes = self.planes(np.asarray(result).reshape(-1), np.asarray(linear).reshape(-1))
        words, bound = pack_planes(*planes)
        return words.reshape(shape), bound

    def sample_packed(self, points: Any) -> tuple[np.ndarray, float] | None:
        """Frequency-domain lattice sample as packed words; ``None`` for a spatial expression."""
        if not self.frequency_domain or self.observable != "complex":
            return None
        points = np.asarray(points, float)
        shape = points.shape[:-1]
        points = points.reshape(-1, len(self.shape))
        matrix = chart_matrix(self.chart, len(self.shape))
        origin = np.asarray(self.chart.origin) if self.chart.dimension else np.zeros(len(self.shape))
        indices = (points - origin) @ np.linalg.inv(matrix).T
        rounded = np.rint(indices)
        if not np.all(np.isfinite(indices)) or not np.allclose(indices, rounded, atol=1e-9, rtol=0):
            raise ValueError("block filter samples must address the source lattice")
        addresses = np.ravel_multi_index(tuple(rounded.astype(np.int64).T), self.shape)
        words, bound = self.project_words_packed(self.sample_words(addresses), addresses)
        return words.reshape(shape), bound

    def sample(self, points: Any) -> np.ndarray:
        points = np.asarray(points, float)
        shape = points.shape[:-1]
        points = points.reshape(-1, len(self.shape))
        matrix = chart_matrix(self.chart, len(self.shape))
        origin = np.asarray(self.chart.origin) if self.chart.dimension else np.zeros(len(self.shape))
        indices = (points - origin) @ np.linalg.inv(matrix).T
        rounded = np.rint(indices)
        if not np.all(np.isfinite(indices)) or not np.allclose(indices, rounded, atol=1e-9, rtol=0):
            raise ValueError("block filter samples must address the source lattice")
        addresses = np.ravel_multi_index(tuple(rounded.astype(np.int64).T), self.shape)
        if self.frequency_domain:
            return self.project_words(
                self.sample_words(addresses, phase=self.observable != "intensity"), addresses
            ).reshape(shape)
        from spf.renderer.source_filter import plan_structured_source

        plan = plan_structured_source(self, SampleLayout((len(addresses), 1)))
        if plan is not None:
            return plan.execute(indices=rounded.astype(np.int64)).reshape(shape)
        return np.asarray(PeriodicBlockExecution(self).execute().reshape(-1)[addresses].reshape(shape))


def sample_source_words(view: BlockView) -> np.ndarray | None:
    """Packed plane of the active source with header offsets folded in.

    The inactive addresses hold the zero word.  Returns ``None`` when the
    view's wide (fractional or out-of-range) header state cannot be folded
    into 32-bit words, or when the view is a spatial filtered expression
    whose source is a transform boundary.
    """
    from spf.tensor import v02

    if isinstance(view, FilteredBlockView):
        return None
    if view.observable != "complex" or view.scalar_bits != 32:
        return None
    output = np.full(view.shape, np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32)
    active = view.active_addresses()
    for start in range(0, len(active), CHUNK):
        addresses = active[start : start + CHUNK]
        words = view.project_words(view.sample_words(addresses), addresses)
        if words.dtype != np.uint32:
            return None
        output.reshape(-1)[addresses] = words
    return output


def sample_source(view: BlockView) -> np.ndarray:
    """Cartesian source at the declared discrete FFT boundary (reference decode).

    The plane is built as packed words first; the decode below is the one
    Cartesian conversion this transform boundary owns.  Wide header state or
    intensity views keep the plane projection.
    """
    if isinstance(view, FilteredBlockView) and not view.frequency_domain:
        return PeriodicBlockExecution(view).execute()
    packed = sample_source_words(view)
    if packed is not None:
        from spf.renderer.carriers import decode_words

        return np.asarray(decode_words(packed))
    output = np.zeros(view.shape, complex)
    active = view.active_addresses()
    for start in range(0, len(active), CHUNK):
        addresses = active[start : start + CHUNK]
        words = view.sample_words(addresses, phase=view.observable != "intensity")
        output.reshape(-1)[addresses] = project_planes(*source_planes(view, words, addresses))
    return output


class PeriodicBlockExecution:
    """Prepared source transform, compact transfer, fresh spectral/spatial terminal."""

    preparation_fft_count: int

    def __init__(
        self,
        view: FilteredBlockView,
        *,
        hoist: bool = True,
        dtype: Any = _C128,
    ) -> None:
        from spf.renderer._filter_kernel import NativeTransfer

        self.view = view
        self.dtype = np.dtype(dtype)
        if self.dtype not in (_C64, _C128):
            raise ValueError("periodic block compute dtype must be complex64 or complex128")
        self.native = NativeTransfer(view.transfer, view.chart, periodic=True)
        self.native.prepare(view.shape)
        self.active_source_samples = len(view.source.active_addresses())
        if isinstance(view.source, FilteredBlockView) and not view.source.frequency_domain:
            nested = PeriodicBlockExecution(view.source, dtype=self.dtype)
            source = nested.execute()
            self.preparation_fft_count = nested.preparation_fft_count + 2
        else:
            source = sample_source(view.source)
            self.preparation_fft_count = 1
        source128 = np.asarray(source, dtype=_C128)
        stored_source = np.asarray(source128, dtype=self.dtype)
        difference = np.abs(source128 - stored_source.astype(_C128))
        self.source_projection_l1_error = float(np.sum(difference, dtype=np.float64))
        from spf.renderer.source_filter import _periodic_transfer_gain_bound

        self.source_projection_operator_gain = _periodic_transfer_gain_bound(view)
        self.source_projection_spectral_error = (
            self.source_projection_l1_error * self.source_projection_operator_gain
        )
        # For e in the spatial source, |IFFT(H FFT(e))|_inf is bounded by
        # sup|H| * |e|_1.  Keep the cast discrepancy attached to the exact
        # prepared operator instead of silently treating complex64 as exact.
        self.source_projection_error = self.source_projection_spectral_error
        self.spectrum = fft.fftn(stored_source, workers=1)
        self.spectrum.setflags(write=False)
        self.hoisted = False
        if hoist:
            self.spectrum = self.execute(spectrum=True)
            self.spectrum.setflags(write=False)
            self.hoisted = True
            self.native.seeds = np.empty(0, dtype=np.uint8)
            self.native.seed_shape = ()

    def execute(self, *, spectrum: bool = False, spectrum_intensity: bool = False) -> np.ndarray:
        view, total = self.view, math.prod(self.view.shape)
        filtered = (
            self.spectrum
            if self.hoisted
            else (
                self.native.grid(self.spectrum, intensity=spectrum_intensity)
                if self.native.supported
                else np.empty(
                    view.shape,
                    self.spectrum.real.dtype if spectrum_intensity else self.dtype,
                )
            )
        )
        for start in range(0, 0 if self.hoisted or self.native.supported else total, CHUNK):
            end = min(total, start + CHUNK)
            addresses = np.arange(start, end)
            logs, phase = view.transfer.log_phase(
                view.coordinates(addresses, periodic=True), addresses, include_phase=not spectrum_intensity
            )
            source = self.spectrum.reshape(-1)[start:end]
            if spectrum_intensity:
                filtered.reshape(-1)[start:end] = np.abs(source) ** 2 * np.exp2(2 * logs)
            else:
                filtered.reshape(-1)[start:end] = source * np.exp2(logs) * np.exp(1j * phase)
        if spectrum or spectrum_intensity:
            if self.hoisted:
                return np.abs(filtered) ** 2 if spectrum_intensity else filtered.copy()
            return filtered
        values = fft.ifftn(filtered, workers=1)
        if view.observable == "intensity":
            values = np.abs(values) ** 2
        if view.log_offset:
            values *= 2 ** (view.log_offset / (128 if view.observable == "intensity" else 256))
        if view.phase_offset:
            values *= np.exp(1j * view.phase_offset * (2 * math.pi / 65536))
        if view.mask_indices is not None:
            selected = np.zeros(total, bool)
            selected[view.mask_indices] = True
            values.reshape(-1)[~selected] = 0
        return np.asarray(values)


class PreparedFilteredBlockTerminal:
    """Shared compiler/runtime terminal; keeps operators analytic until output."""

    def __init__(self, field: Any, request: Any) -> None:
        from spf.renderer._filter_kernel import NativeTransfer
        from spf.renderer.source_filter import PreparedFourierSource, StructuredPeriodicSource, plan_structured_source

        self.field, self.request, self.view = field, request, field.representation.payload
        self.native = NativeTransfer(self.view.transfer, self.view.chart, periodic=False)
        self.native_words = (
            self.native.supported
            and self.view.scalar_bits == 32
            and not isinstance(self.view.source, FilteredBlockView)
        )
        layout = request.layout
        if any(x != round(x) for x in (*layout.origin, *layout.spacing)):
            raise ValueError("block filter output must address the source lattice")
        if any(
            o < 0 or (n and o + (n - 1) * s >= dim)
            for o, n, s, dim in zip(layout.origin, layout.shape, layout.spacing, self.view.shape, strict=True)
        ):
            raise ValueError("block filter output lies outside source extent")
        self.source_plan: PreparedFourierSource | StructuredPeriodicSource | None = None
        self.structured_fallback_reason: str | None = None
        if (
            self.view.frequency_domain
            and field.metadata.get("sfr_terminal_spatial", False)
            and field.support.kind is not SupportKind.EMPTY
        ):
            self.source_plan = PreparedFourierSource(
                self.view,
                layout,
                dtype=field.metadata.get("compute_dtype", "complex128"),
                periodic_transfer=bool(field.metadata.get("sfr_texture_block_ingress", False)),
            )
        if not self.view.frequency_domain and field.support.kind is not SupportKind.EMPTY:
            self.source_plan = plan_structured_source(
                self.view, layout, spectrum=field.metadata.get("sfr_terminal_spectral", False)
            )
            if self.source_plan is None:
                source_kind = getattr(getattr(self.view.source, "kernel", None), "expr", None)
                source_op = getattr(source_kind, "op", None)
                self.structured_fallback_reason = (
                    "low-rank factor projection finite SPF32 envelope is unproven; "
                    "exact periodic fallback"
                    if source_op == "LowRankOuter"
                    else "structured source proof unavailable or too expensive; exact periodic fallback"
                )
        # A factor FFT evaluates the retained U C V* values before the
        # canonical kernel rounds each matrix entry to SPF32.  Its source
        # projection bound is explicit on the plan; keep the exact addressed
        # periodic path when the requested absolute contract cannot admit it.
        if self.source_plan is not None:
            source_bound = (
                getattr(self.source_plan, "source_projection_spectral_error", 0.0)
                if field.metadata.get("sfr_terminal_spectral", False)
                else getattr(self.source_plan, "source_projection_error", 0.0)
            )
            allowed = min(float(request.error_budget.absolute), float(field.error.absolute_max))
            if (not math.isfinite(source_bound)) or source_bound > allowed:
                requested_dtype = np.dtype(field.metadata.get("compute_dtype", _C128))
                if requested_dtype == _C64 and isinstance(self.source_plan, PreparedFourierSource):
                    self.structured_fallback_reason = (
                        f"structured complex64 source projection bound {source_bound:.6g} exceeds "
                        f"absolute contract {allowed:.6g}; using complex128 terminal"
                    )
                    self.source_plan = PreparedFourierSource(
                        self.view,
                        layout,
                        dtype=_C128,
                        periodic_transfer=bool(field.metadata.get("sfr_texture_block_ingress", False)),
                    )
                else:
                    self.structured_fallback_reason = (
                        f"structured source projection bound {source_bound:.6g} exceeds "
                        f"absolute contract {allowed:.6g}"
                    )
                    self.source_plan = None
        if field.support.kind is SupportKind.EMPTY:
            self.periodic = None
            return
        self.periodic = (
            None
            if self.view.frequency_domain or self.source_plan is not None
            else PeriodicBlockExecution(
                self.view,
                hoist=not field.metadata.get("sfr_terminal_spectral", False),
                dtype=field.metadata.get("compute_dtype", "complex128"),
            )
        )
        if self.periodic is not None:
            periodic_bound = float(self.periodic.source_projection_error)
            allowed = min(float(request.error_budget.absolute), float(field.error.absolute_max))
            if (not math.isfinite(periodic_bound)) or periodic_bound > allowed:
                self.structured_fallback_reason = (
                    f"periodic complex64 source projection bound {periodic_bound:.6g} exceeds "
                    f"absolute contract {allowed:.6g}"
                )
                self.periodic = PeriodicBlockExecution(
                    self.view,
                    hoist=not field.metadata.get("sfr_terminal_spectral", False),
                    dtype=_C128,
                )
        requested_dtype = np.dtype(field.metadata.get("compute_dtype", _C128))
        effective_dtype = (
            self.periodic.dtype
            if self.periodic is not None
            else self.source_plan.dtype
            if self.source_plan is not None and hasattr(self.source_plan, "dtype")
            else _C128
        )
        if effective_dtype != requested_dtype:
            self.field = field = replace(
                field,
                metadata={
                    **dict(field.metadata),
                    "requested_compute_dtype": requested_dtype.name,
                    "compute_dtype": effective_dtype.name,
                    "compute_dtype_fallback": True,
                },
            )
        if self.periodic is None and self.source_plan is None:
            layout = request.layout
            active = (
                self.view.active_addresses()
                if field.support.kind is not SupportKind.EMPTY
                else np.empty(0, dtype=np.int64)
            )
            coordinates = np.asarray(np.unravel_index(active, self.view.shape)).T
            grid = (coordinates - np.asarray(layout.origin)) / np.asarray(layout.spacing)
            valid = np.all((grid == np.rint(grid)) & (grid >= 0) & (grid < np.asarray(layout.shape)), axis=-1)
            self.source_addresses = active[valid]
            self.destination = np.ravel_multi_index(
                tuple(grid[valid].astype(np.int64).T), layout.shape, order=layout.order
            )
            self.words = self.view.sample_words(self.source_addresses, phase=self.view.observable != "intensity")
            for array in (self.source_addresses, self.destination, self.words):
                array.setflags(write=False)

    def execute(self) -> Any:
        from spf.renderer import runtime
        from spf.renderer.native import NativeProjectionResult
        from spf.renderer.report import RenderMetrics, RenderReport

        view, layout = self.view, self.request.layout
        count = layout.element_count
        periodic = self.periodic.execute() if self.periodic is not None else None
        direct = self.source_plan.execute() if self.source_plan is not None else None
        full_layout = (
            layout.shape == view.shape and all(x == 0 for x in layout.origin) and all(x == 1 for x in layout.spacing)
        )
        values = (
            periodic
            if periodic is not None and full_layout and layout.order == "C"
            else np.zeros(layout.shape, float if view.observable == "intensity" else complex, order=layout.order)
        )
        if direct is not None:
            values = direct
        active_count = 0
        packed_bound = 0.0
        packed_output = (
            periodic is None
            and direct is None
            and view.observable == "complex"
            and runtime._format_name(self.request.output_format) == "spf32"
        )
        if packed_output:
            # Words requested: project the analytic transfer straight into
            # packed SPF32 words (one quantization, bounded) instead of a
            # Cartesian plane that the formatter would re-encode.
            from spf.tensor import v02

            values = np.full(layout.shape, np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32, order=layout.order)
            for start in range(0, 0 if self.field.support.kind is SupportKind.EMPTY else len(self.words), CHUNK):
                addresses = self.source_addresses[start : start + CHUNK]
                chunk, bound = view.project_words_packed(self.words[start : start + CHUNK], addresses)
                packed_bound = max(packed_bound, bound)
                values.ravel(order=layout.order)[self.destination[start : start + CHUNK]] = chunk
                active_count += len(addresses)
        if self.field.support.kind is not SupportKind.EMPTY and direct is None and not packed_output:
            native_words = periodic is None and self.native_words
            if native_words:
                self.native.words(self.words, self.source_addresses, self.destination, values, view)
                active_count = len(self.words)
            for start in range(
                0, 0 if native_words or values is periodic else len(self.words) if periodic is None else count, CHUNK
            ):
                if periodic is None:
                    addresses = self.source_addresses[start : start + CHUNK]
                    chunk = view.project_words(self.words[start : start + CHUNK], addresses)
                    values.ravel(order=layout.order)[self.destination[start : start + CHUNK]] = chunk
                    active_count += len(addresses)
                    continue
                end = min(count, start + CHUNK)
                output_indices = np.asarray(np.unravel_index(np.arange(start, end), layout.shape, order=layout.order)).T
                source_indices = np.asarray(layout.origin) + output_indices * np.asarray(layout.spacing)
                if np.any(source_indices != np.rint(source_indices)):
                    raise ValueError("block filter output must address the source lattice")
                source_addresses = np.ravel_multi_index(tuple(source_indices.astype(np.int64).T), view.shape)
                chunk = periodic.reshape(-1)[source_addresses]
                values.ravel(order=layout.order)[start:end] = chunk
        native, source, kind = runtime._semantic_source(self.field, values)
        context = runtime._SamplingContext(np.empty((0, len(view.shape))), layout.shape, layout.order, layout, None)
        output, observed, phase_error, format_ledger, _ = runtime._format_output(
            self.field, self.request, native, source, kind, context, ()
        )
        from spf.renderer.normalization import _field_ledger_after_operations

        ledger = runtime._apply_representation_error(self.field, _field_ledger_after_operations(self.field))
        if self.source_plan is not None:
            # Low-rank factor FFTs operate on the natural factor values while
            # the canonical SPF kernel rounds each matrix entry on projection.
            # Carry that bounded discrepancy through the same source
            # quantization ledger used by all other packed consumers.
            source_error = (
                getattr(self.source_plan, "source_projection_spectral_error", 0.0)
                if self.field.metadata.get("sfr_terminal_spectral", False)
                else getattr(self.source_plan, "source_projection_error", 0.0)
            )
            if source_error:
                ledger = runtime._ledger_add(ledger, source_quantization=source_error)
        if self.periodic is not None and self.periodic.source_projection_error:
            ledger = runtime._ledger_add(
                ledger,
                source_quantization=float(self.periodic.source_projection_error),
            )
        ledger = runtime._ledger_add(
            ledger,
            source_quantization=format_ledger.source_quantization + packed_bound,
            materialization=format_ledger.materialization,
            display=format_ledger.display,
        )
        source_scale = runtime._max_abs(source)
        runtime._enforce_error_budget(self.field, self.request, ledger, source, phase_error, scale=source_scale)
        packed_count = len(self.words) if hasattr(self, "words") else 0
        source_native = getattr(self.source_plan, "native", None)
        prepared_bytes = self.words.nbytes if packed_count else 0
        if self.periodic is not None:
            prepared_bytes = self.periodic.spectrum.nbytes
        carrier_elements = packed_count if self.periodic is None else 3 * math.prod(view.shape)
        if self.source_plan is not None:
            prepared_bytes = self.source_plan.nbytes
            source_itemsize = np.dtype(getattr(self.source_plan, "dtype", _C128)).itemsize
            carrier_elements = prepared_bytes // source_itemsize
        metrics = RenderMetrics(
            backend="cpu",
            workload="analytic_block_filter",
            representation="block_analytic_transfer",
            resolution=layout.shape,
            fft_count=self.source_plan.execution_fft_count
            if self.source_plan is not None
            else int(self.periodic is not None),
            field_operations=len(view.transfer.operations),
            terminal_output_elements=count,
            terminal_output_bytes=output.nbytes,
            potential_elements=count,
            potential_intermediate_elements=count * len(view.transfer.operations),
            materialized_intermediate_elements=carrier_elements,
            bytes_read=prepared_bytes + view.transfer.nbytes,
            bytes_written=output.nbytes,
        )
        report = RenderReport(
            metrics,
            ledger=ledger,
            capabilities_used=(
                "addressed_block_consumer",
                "analytic_spectral_transfer",
                "full_quadratic_log_phase_jets",
                *(("packed_transfer_projection",) if packed_output else ()),
                *(
                    ("native_packed_filter_consumer",)
                    if getattr(self.source_plan, "native_consumer_used", False)
                    else ()
                ),
            ),
            diagnostics={
                "analytic_operations": view.transfer.operations,
                "dense_operations": (),
                "operator_bytes": view.transfer.nbytes,
                "dense_transfer_elements": 0,
                "native_filter_consumer": bool(getattr(self.source_plan, "native_consumer_used", False))
                if self.source_plan is not None
                else self.native_words
                if self.periodic is None
                else self.periodic.native.supported,
                "native_filter_path": getattr(self.source_plan, "native_filter_path", None)
                if self.source_plan is not None
                else "PeriodicBlockExecution -> NativeTransfer.grid -> spf_sfr_filter_grid"
                if self.periodic is not None and self.periodic.native.supported
                else None,
                "native_filter_abi": 3
                if getattr(self.source_plan, "native_consumer_used", False)
                or (self.periodic is not None and self.periodic.native.supported)
                else None,
                "prepared_recurrence_bytes": (
                    int(source_native.seeds.nbytes)
                    if source_native is not None
                    else 0
                    if self.periodic is None
                    else int(self.periodic.native.seeds.nbytes)
                ),
                "spectral_prefix_hoisted": self.periodic is not None and self.periodic.hoisted,
                "cached_pixels": 0,
                "source_block_type": type(view.source).__name__,
                "active_source_samples": (
                    self.source_plan.active_source_samples
                    if self.source_plan is not None
                    else active_count
                    if self.periodic is None
                    else self.periodic.active_source_samples
                ),
                "prepared_packed_elements": (
                    getattr(self.source_plan, "mode_count", packed_count)
                    if getattr(self.source_plan, "native_consumer_used", False)
                    else packed_count
                ),
                "prepared_spectrum_elements": 0 if self.periodic is None else self.periodic.spectrum.size,
                "sampled_phase_elements": sum(a.size for a in view.transfer.phase_residuals),
                "transfer_chunk_limit": CHUNK,
                "byte_traffic_status": "logical carrier I/O; excludes FFT traffic and arithmetic scratch",
                "materialization_scope": "packed source or FFT carriers; excludes bounded arithmetic scratch",
                "source_plan": None if self.source_plan is None else self.source_plan.kind,
                "structured_fallback_reason": self.structured_fallback_reason,
                "source_projection_error": (
                    float(getattr(self.periodic, "source_projection_error", 0.0))
                    if self.source_plan is None
                    else float(getattr(self.source_plan, "source_projection_error", 0.0))
                ),
                "source_projection_spectral_error": (
                    float(getattr(self.periodic, "source_projection_spectral_error", 0.0))
                    if self.source_plan is None
                    else float(getattr(self.source_plan, "source_projection_spectral_error", 0.0))
                ),
                "source_projection_l1_error": (
                    float(getattr(self.periodic, "source_projection_l1_error", 0.0))
                    if self.source_plan is None
                    else float(getattr(self.source_plan, "source_projection_l1_error", 0.0))
                ),
                "source_projection_operator_gain": (
                    float(getattr(self.periodic, "source_projection_operator_gain", 1.0))
                    if self.source_plan is None
                    else float(getattr(self.source_plan, "source_projection_operator_gain", 1.0))
                ),
                "source_projection_semantics": (
                    "factorized natural UCV* versus canonical per-entry SPF32 projection"
                    if self.source_plan is not None and self.source_plan.kind == "matrix_low_rank_fft"
                    else "native packed SPF32 transfer to declared Cartesian inverse-FFT terminal"
                    if getattr(self.source_plan, "native_consumer_used", False)
                    else None
                ),
                "transform_arithmetic": (
                    "none"
                    if self.source_plan is not None
                    and self.source_plan.preparation_fft_count == self.source_plan.execution_fft_count == 0
                    else f"{np.dtype(getattr(self.source_plan, 'dtype', _C128)).name}_inverse_fft"
                    if self.source_plan is not None and self.source_plan.execution_fft_count
                    else f"{np.dtype(getattr(self.source_plan, 'dtype', _C128)).name}_axis_fft"
                    if self.source_plan is not None
                    else f"{self.periodic.dtype.name}_periodic_fft"
                    if self.periodic is not None
                    else "none"
                ),
                "source_mode_count": None if self.source_plan is None else self.source_plan.mode_count,
                "preparation_fft_count": (
                    self.source_plan.preparation_fft_count
                    if self.source_plan is not None
                    else 0
                    if self.periodic is None
                    else self.periodic.preparation_fft_count
                ),
                "fft_boundary": (
                    "discrete_inverse_terminal"
                    if self.source_plan is not None and self.source_plan.execution_fft_count
                    else "none"
                    if self.periodic is None
                    else "periodic_discrete_source_and_terminal"
                ),
                "output_observed_error": observed,
                "output_scale": source_scale,
                "memory_status": "not_measured; bounded transfer chunks plus explicit FFT/output arrays",
            },
        )
        return NativeProjectionResult(output, report, np.empty(0, np.int16))


def prepare_filtered_terminal(field: Any, request: Any) -> PreparedFilteredBlockTerminal | None:
    if (
        field.operations
        or request.region is not None
        or field.representation.residual is not None
        or field.support.kind not in {SupportKind.FULL, SupportKind.EMPTY}
    ):
        return None
    if request.layout is None:
        request = replace(request, layout=SampleLayout(field.representation.payload.shape))
    if len(request.layout.shape) != len(field.representation.payload.shape):
        return None
    return PreparedFilteredBlockTerminal(field, request)
