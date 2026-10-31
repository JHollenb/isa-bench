"""Source-aware finite-periodic synthesis from block structure.

No continuous Gaussian is substituted for a sampled periodic field. Exact
characters are proved in integer phase space; other separable axes retain all
DFT coefficients. The planner compares requested synthesis work with a dense
transform before allocating a two-dimensional Cartesian carrier.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf.renderer.blocks import KernelBlockView, TensorBlockView, V3BlockView
from spf.renderer.carriers import decode_words
from spf.tensor import v02
from spf.tensor.core import ActiveEncoding

_C64 = np.dtype("complex64")
_C128 = np.dtype("complex128")


def _character_mode(logs: np.ndarray, phases: np.ndarray) -> int | None:
    n = len(logs)
    step = int((phases[1] - phases[0]) % 65536) if n > 1 else 0
    if np.all(logs == logs[0]) and np.all((np.diff(phases) - step) % 65536 == 0) and step * n % 65536 == 0:
        return step * n // 65536
    return None


def _axis_spectrum(logs: np.ndarray, phases: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    n = len(logs)
    mode = _character_mode(logs, phases)
    if mode is not None:
        value = n * np.exp2(float(logs[0]) / 256) * np.exp(2j * np.pi * float(phases[0] % 65536) / 65536)
        return np.array([mode]), np.array([value]), 0
    values = np.exp2(logs / 256) * np.exp(2j * np.pi * (phases % 65536) / 65536)
    return np.arange(n), fft.fft(values, workers=1), 1


def _tensor_axes(source: TensorBlockView) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    if len(source._tensors) != 1 or source.mask_indices is not None or source.observable != "complex":
        return None
    tensor = source._tensors[0]
    h = tensor.header
    if h.active_encoding != ActiveEncoding.DENSE_ALL or tensor.outliers:
        return None
    # Residual separability is an exact property, checked once on the frozen
    # carrier, in bounded rows. Header bands and cubic terms are already axial.
    axes_l: list[np.ndarray] = []
    axes_t: list[np.ndarray] = []
    assert h.origin is not None and h.band_exp is not None and h.ramp_b is not None
    assert h.quadratic_ramp is not None and h.cubic_ramp is not None
    for stream, axes, modular in ((tensor.dL, axes_l, False), (tensor.dT, axes_t, True)):
        residual = np.asarray(stream, np.int16).reshape(h.dims)
        row = residual[0].astype(np.int64)
        col = residual[:, 0].astype(np.int64) - int(row[0])
        rows_per_chunk = max(1, 65536 // h.dims[1])
        for start in range(0, h.dims[0], rows_per_chunk):
            difference = (
                residual[start : start + rows_per_chunk].astype(np.int64)
                - col[start : start + rows_per_chunk, None]
                - row
            )
            if np.any(difference % 65536 if modular else difference):
                return None
        axes.extend((col, row))
    r = np.asarray(h.ramp_R, np.int64).sum(axis=0)
    for axis, n in enumerate(h.dims):
        q = (np.arange(n, dtype=np.int64) + h.origin[axis]) % n
        axes_l[axis] += q * h.band_exp[axis]
        axes_t[axis] += q * r[axis] + q * q * h.quadratic_ramp[axis] + q * q * q * h.cubic_ramp[axis]
        for band in tensor.bands:
            if band.dim == axis:
                selected = (q >= band.start) & (q < band.start + band.length)
                if band.effective_flags & 1:
                    axes_l[axis][selected] += int(band.L_delta)
                if band.effective_flags & 2:
                    axes_t[axis][selected] += int(band.T_delta)
    axes_l[0] += h.global_L_bias + h.coarse_exp
    axes_t[0] += h.global_T_bias + sum(h.ramp_b)
    # Saturation and the reserved zero/NaN codes break multiplicative closure.
    if (
        sum(int(x.min()) for x in axes_l) < v02.SPF32_L_MIN_FINITE
        or sum(int(x.max()) for x in axes_l) > v02.SPF32_L_MAX_FINITE
    ):
        return None
    # Balance the factors to avoid overflow in an axis whose opposing offset
    # cancels in the represented two-dimensional log magnitude.
    shift = int(axes_l[0][0])
    axes_l[0] -= shift
    axes_l[1] += shift
    if any(np.max(np.abs(x)) > 200000 for x in axes_l):
        return None
    return axes_l, axes_t


def _v3_tensor(source: V3BlockView) -> TensorBlockView | None:
    """Recognize only the width/model/codec whose algebra is the v2 algebra."""
    from spf.tensor.v2 import TensorBlock, TensorBlockHeaderV2
    from spf.tensor.v3 import BaseModelKind, ResidualCodecKind

    block = source.block
    if source.scalar_bits != 32 or block.residual.codec_id != int(ResidualCodecKind.RAW_INT16):
        return None
    if block.base.profile_id == int(BaseModelKind.STB_AFFINE_POLY):
        h = block.base.params["header"]
        active = block.base.params.get("active_indices")
        if h.dims != block.dims or (
            active is not None
            and (len(active) != math.prod(block.dims) or not np.all(np.diff(active) == 1) or active[0] != 0)
        ):
            return None
        bands = block.base.params.get("bands", ())
    elif block.base.profile_id == int(BaseModelKind.SEPARABLE_PHASE) and not block.base.params:
        h, bands = TensorBlockHeaderV2(dims=block.dims), ()
    else:
        return None
    logs, phase = block.residual.payload["dL"], block.residual.payload["dT"]
    if np.any(np.asarray(logs) < -32768) or np.any(np.asarray(logs) > 32767):
        return None
    tensor = TensorBlock(
        h, np.asarray(logs, np.int16), np.asarray(phase).astype(np.int16), bands=bands, outliers=block.outliers
    )
    return TensorBlockView(
        (tensor,), source.shape, source.observable, source.log_offset, source.phase_offset, source.mask_indices
    )


class StructuredPeriodicSource:
    """A factored spectrum and an addressed inverse-DFT consumer, never an N grid."""

    execution_fft_count = 0

    def __init__(
        self,
        view: Any,
        layout: Any,
        axes: Any = None,
        diagonal: Any = None,
        low_rank: Any = None,
    ) -> None:
        self.view, self.layout = view, layout
        self.preparation_fft_count = 0
        self.kind = "tensor_separable_axes" if axes is not None else "matrix_constant_diagonal"
        self.axes = None
        self.diagonal = diagonal
        self.diagonal_spectrum: np.ndarray | None = None
        self.low_rank_factors: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        # Low-rank factors are evaluated in their natural complex domain and
        # therefore do not include the per-entry SPF32 rounding performed by
        # SPFKernel.sample_words.  Retain a conservative source projection
        # bound so a structured approximation cannot masquerade as exact.
        self.source_projection_error = 0.0
        self.source_projection_spectral_error = 0.0
        self.source_projection_l1_error = 0.0
        self.source_projection_operator_gain = 1.0
        if axes is not None:
            self.axes = []
            for logs, phases in zip(*axes, strict=True):
                indices, values, calls = _axis_spectrum(logs, phases)
                self.axes.append((indices, values))
                self.preparation_fft_count += calls
            self.mode_count = math.prod(len(a[0]) for a in self.axes)
            if self.mode_count == 1:
                self.kind = "tensor_exact_character"
            self.nbytes = sum(a.nbytes + b.nbytes for a, b in self.axes)
        elif low_rank is not None:
            # A low-rank spatial matrix remains a sum of products after a
            # separable 2-D DFT.  Keep only the two one-dimensional spectra
            # and the rank coefficients; each requested 2-D frequency is
            # reduced at the mode boundary below.
            u_fft, v_fft, coefficients, *bound_values = low_rank
            self.low_rank_factors = (
                np.asarray(u_fft, dtype=np.complex128),
                np.asarray(v_fft, dtype=np.complex128),
                np.asarray(coefficients, dtype=np.complex128).reshape(-1),
            )
            self.mode_count = math.prod(view.shape)
            self.preparation_fft_count = 2
            self.nbytes = sum(item.nbytes for item in self.low_rank_factors)
            self.kind = "matrix_low_rank_fft"
            self.active_source_samples = sum(item.size for item in self.low_rank_factors)
            for item in self.low_rank_factors:
                item.setflags(write=False)
            coefficients = self.low_rank_factors[2]
            # The planner computes this from the decoded raw factor rows
            # before the two retained FFTs.  Do not spend hidden preparation
            # work reconstructing factors merely to certify the bound.
            l1_bound = float(bound_values[0]) if bound_values else math.inf
            # SPF32 uses 1/256 log2 ticks and 1/65536 phase ticks.  The
            # resulting relative magnitude error plus phase arc bounds every
            # independently projected matrix entry.
            quantization = (2.0 ** (1.0 / 512.0) - 1.0) + 2.0 ** (1.0 / 512.0) * math.pi / 65536.0
            total = math.prod(view.shape)
            header_gain = float(bound_values[1]) if len(bound_values) > 1 else 1.0
            underflow = 2.0 * math.exp2(v02.SPF32_L_MIN_FINITE / 256.0) * total * header_gain
            source_l1_error = quantization * l1_bound + underflow
            gain = _periodic_transfer_gain_bound(view)
            # ``source_l1_error`` bounds the sum of entry-wise source errors,
            # while a spatial inverse DFT with an arbitrary compact transfer
            # is an l-infinity convolution.  Its induced norm is bounded by
            # sqrt(N)*sup|H| via Parseval; dividing by N here would be false
            # for a phase-only fractional translation (and under-report
            # hostile cancellation).  Spectral terminals only need the
            # entry-sum bound.  Keep both contracts explicit in the ledger.
            self.source_projection_l1_error = source_l1_error
            self.source_projection_operator_gain = gain
            self.source_projection_spectral_error = source_l1_error * gain
            self.source_projection_error = source_l1_error * math.sqrt(max(total, 1)) * gain
            return
        elif np.asarray(diagonal).ndim == 0:
            self.mode_count = view.shape[0]
            self.nbytes = 16
        else:
            values = np.asarray(diagonal, dtype=np.complex128).reshape(-1)
            # For A[i,j] = d[i] 1[i=j], the 2-D DFT is
            # D[(p+q) mod n], where D is the 1-D DFT of d.  The vector and
            # its FFT are both O(n); no diagonal matrix is ever built.
            self.diagonal_spectrum = np.asarray(fft.fft(values, workers=1), dtype=np.complex128)
            self.diagonal_spectrum.setflags(write=False)
            self.mode_count = math.prod(view.shape)
            self.preparation_fft_count = 1
            self.nbytes = self.diagonal_spectrum.nbytes
            self.kind = "matrix_diagonal_fft"
        self.active_source_samples = 0 if axes is not None else view.shape[0]

    def modes(self, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
        if self.axes is not None:
            row, col = np.unravel_index(np.arange(start, end), tuple(len(x[0]) for x in self.axes))
            indices = self.axes[0][0][row] * self.view.shape[1] + self.axes[1][0][col]
            values = self.axes[0][1][row] * self.axes[1][1][col]
            return indices, values
        if self.diagonal_spectrum is not None:
            n = self.view.shape[1]
            flat = np.arange(start, end, dtype=np.int64)
            rows, columns = np.divmod(flat, n)
            return flat, self.diagonal_spectrum[(rows + columns) % n]
        if self.low_rank_factors is not None:
            u_fft, v_fft, coefficients = self.low_rank_factors
            n = self.view.shape[1]
            flat = np.arange(start, end, dtype=np.int64)
            rows, columns = np.divmod(flat, n)
            # DFT(conj(v))[q] = conj(DFT(v)[-q]).  Reduce the rank only for
            # this mode, retaining the low-rank carrier through synthesis.
            values = np.sum(
                u_fft[rows] * np.conj(v_fft[(-columns) % n]) * coefficients[None, :], axis=1
            )
            return flat, values
        n = self.view.shape[0]
        row = np.arange(start, end)
        return row * n + (-row % n), np.full(end - start, n * self.diagonal, complex)

    def execute(
        self, *, spectrum: bool = False, spectrum_intensity: bool = False, indices: np.ndarray | None = None
    ) -> np.ndarray:
        view, layout = self.view, self.layout
        count, total = layout.element_count, math.prod(view.shape)
        output = np.zeros(layout.shape, complex, order=layout.order)
        flat = output.ravel(order=layout.order)
        # Bound both the source scratch and each synthesis matrix, independent
        # of image size. No cached terminal pixels or dense transfer planes.
        mode_chunk = min(1024, self.mode_count)
        for start in range(0, self.mode_count, max(1, mode_chunk)):
            addresses, coefficients = self.modes(start, min(start + mode_chunk, self.mode_count))
            dl, dt = view.transfer.log_phase(view.coordinates(addresses, periodic=True), addresses)
            coefficients = coefficients * np.exp2(dl) * np.exp(1j * dt)
            if not spectrum and self.mode_count == 1 and indices is None:
                ky, kx = np.unravel_index(addresses[0], view.shape)
                if ky == kx == 0:
                    output.fill(coefficients[0] / total)
                else:
                    y = layout.origin[0] + np.arange(layout.shape[0]) * layout.spacing[0]
                    x = layout.origin[1] + np.arange(layout.shape[1]) * layout.spacing[1]
                    np.multiply(
                        np.exp(2j * np.pi * ky * y / view.shape[0])[:, None],
                        np.exp(2j * np.pi * kx * x / view.shape[1])[None, :] * (coefficients[0] / total),
                        out=output,
                    )
                continue
            if spectrum:
                coords = np.asarray(np.unravel_index(addresses, view.shape)).T
                grid = (coords - layout.origin) / np.asarray(layout.spacing)
                keep = np.all((grid == np.rint(grid)) & (grid >= 0) & (grid < layout.shape), axis=1)
                dest = np.ravel_multi_index(tuple(grid[keep].astype(np.int64).T), layout.shape, order=layout.order)
                flat[dest] = coefficients[keep]
                continue
            frequency = np.asarray(np.unravel_index(addresses, view.shape)).T / np.asarray(view.shape)
            query_chunk = max(1, 65536 // len(addresses))
            for first in range(0, count, query_chunk):
                last = min(first + query_chunk, count)
                if indices is None:
                    coords = np.asarray(np.unravel_index(np.arange(first, last), layout.shape, order=layout.order)).T
                    coords = np.asarray(layout.origin) + coords * np.asarray(layout.spacing)
                else:
                    coords = indices[first:last]
                flat[first:last] += np.exp(2j * np.pi * (coords @ frequency.T)) @ (coefficients / total)
        if spectrum:
            return np.abs(output) ** 2 if spectrum_intensity else output
        if view.observable == "intensity":
            output = np.abs(output) ** 2
        if view.log_offset:
            output *= 2 ** (view.log_offset / (128 if view.observable == "intensity" else 256))
        if view.phase_offset:
            output *= np.exp(2j * np.pi * view.phase_offset / 65536)
        if view.mask_indices is not None:
            for first in range(0, count, 65536):
                last = min(first + 65536, count)
                if indices is None:
                    coords = np.asarray(np.unravel_index(np.arange(first, last), layout.shape, order=layout.order)).T
                    coords = np.asarray(layout.origin) + coords * np.asarray(layout.spacing)
                else:
                    coords = indices[first:last]
                addresses = np.ravel_multi_index(tuple(coords.astype(np.int64).T), view.shape)
                output.ravel(order=layout.order)[first:last] *= np.isin(addresses, view.mask_indices)
        return output


def plan_structured_source(view: Any, layout: Any, *, spectrum: bool = False) -> StructuredPeriodicSource | None:
    if len(view.shape) != 2:
        return None
    source = view.source
    axes: Any = None
    diagonal: Any = None
    low_rank: Any = None
    if (
        isinstance(source, KernelBlockView)
        and source.kernel.expr.op == "Block"
        and source.kernel.expr.block is not None
    ):
        source = V3BlockView(
            source.kernel.expr.block,
            source.shape,
            source.observable,
            source.log_offset,
            source.phase_offset,
            source.mask_indices,
        )
    if isinstance(source, V3BlockView):
        source = _v3_tensor(source)
    if isinstance(source, TensorBlockView):
        axes = _tensor_axes(source)
        if axes is None:
            return None
    elif (
        isinstance(source, KernelBlockView)
        and source.mask_indices is None
        and source.observable == "complex"
    ):
        op = source.kernel.expr.op
        if op == "ClosedFormIdentity":
            if view.shape[0] != view.shape[1]:
                return None
            diagonal = _project_kernel_words(source, np.array([v02.pack_spf32(0, 0)]), np.array([0]))[0]
            if not np.isfinite(diagonal):
                return None
        elif op == "DiagonalSPF":
            if view.shape[0] != view.shape[1]:
                return None
            words = source.kernel.expr.args[0]["diag_words"]
            if not len(words):
                return None
            # The diagonal vector is stored separately from the matrix
            # address space.  Feed its true matrix addresses to the common
            # projector so any wide header state is applied at the matrix
            # element exactly once.
            size = int(view.shape[1])
            addresses: np.ndarray = np.arange(len(words), dtype=np.int64) * (size + 1)
            values = _project_kernel_words(source, words, addresses)
            if not np.all(np.isfinite(values)):
                return None
            # Preserve the zero-FFT fast path for a genuinely constant
            # diagonal.  ``array_equal`` is intentional: this is a carrier
            # identity, not a tolerance-based approximation.
            diagonal = (
                _project_kernel_words(source, np.asarray(words[:1]), np.array([0]))[0]
                if np.all(np.asarray(words) == words[0])
                else values
            )
        elif op == "LowRankOuter":
            params = source.kernel.expr.args[0]
            try:
                u_words = np.asarray(params["u_words"], dtype=np.uint32)
                v_words = np.asarray(params["v_words"], dtype=np.uint32)
                coeff_words = np.asarray(params["coeff_words"], dtype=np.uint32).reshape(-1)
                if u_words.ndim != 2 or v_words.ndim != 2 or u_words.shape[1] != v_words.shape[1]:
                    return None
                if u_words.shape[0] != view.shape[0] or v_words.shape[0] != view.shape[1]:
                    return None
                # U, V and C are independent SPF carriers.  Their packed
                # words must be decoded without applying a matrix header to
                # each factor; a wide source factor is applied once below.
                u_values = np.asarray(decode_words(u_words.reshape(-1)), dtype=np.complex128)
                v_values = np.asarray(decode_words(v_words.reshape(-1)), dtype=np.complex128)
                raw_coefficients = np.asarray(decode_words(coeff_words), dtype=np.complex128)
                source_factor = np.exp2(float(source.log_offset) / 256.0) * np.exp(
                    1j * 2.0 * math.pi * float(source.phase_offset) / 65536.0
                )
                coefficients = raw_coefficients * source_factor
                u_values = u_values.reshape(u_words.shape)
                v_values = v_values.reshape(v_words.shape)
                if not (
                    np.all(np.isfinite(u_values))
                    and np.all(np.isfinite(v_values))
                    and np.all(np.isfinite(coefficients))
                ):
                    return None
                # Every canonical matrix entry is rounded/clipped once by
                # ``SPFKernel.sample_words``.  The factor route is valid only
                # when a finite-code envelope proves that no reduced entry
                # can clip BEFORE a view's wide offset is applied. A negative
                # offset cannot undo clipping in the canonical source kernel.
                # This factor bound avoids expanding U C V* to inspect it.
                factor_terms = (
                    np.max(np.abs(u_values), axis=0)
                    * np.abs(raw_coefficients)
                    * np.max(np.abs(v_values), axis=0)
                )
                pointwise_bound = math.fsum(float(value) for value in factor_terms)
                finite_max = math.exp2(v02.SPF32_L_MAX_FINITE / 256.0)
                if not math.isfinite(pointwise_bound) or pointwise_bound > finite_max:
                    return None
                l1_bound = float(
                    np.sum(
                        np.sum(np.abs(u_values), axis=0)
                        * np.abs(coefficients)
                        * np.sum(np.abs(v_values), axis=0)
                    )
                )
                if not math.isfinite(l1_bound):
                    return None
                low_rank = (
                    fft.fft(u_values, axis=0, workers=1),
                    fft.fft(v_values, axis=0, workers=1),
                    coefficients,
                    l1_bound,
                    abs(source_factor),
                )
            except (KeyError, TypeError, ValueError):
                return None
        else:
            return None
    else:
        return None
    # Sparse characters are advantageous even for full output. Dense spectral
    # support is advantageous for a small query, not for a dense inverse DFT.
    modes = (
        math.prod(1 if _character_mode(l, t) is not None else len(l) for l, t in zip(*axes, strict=True))
        if axes is not None
        else math.prod(view.shape)
        if diagonal is not None and np.asarray(diagonal).ndim != 0
        else math.prod(view.shape)
        if low_rank is not None
        else view.shape[0]
    )
    work = modes * (1 if spectrum else layout.element_count)
    dense_work = 4 * math.prod(view.shape) * max(1, math.log2(math.prod(view.shape)))
    return (
        StructuredPeriodicSource(view, layout, axes, diagonal, low_rank)
        if work <= dense_work
        else None
    )


def _project_kernel_words(source: KernelBlockView, words: Any, addresses: np.ndarray) -> np.ndarray:
    """Decode a structural kernel vector without asking the kernel for a matrix."""

    projected = source.project_words(np.asarray(words, dtype=np.uint32), addresses)
    if np.asarray(projected).dtype == np.uint32:
        projected = decode_words(projected)
    return np.asarray(projected, dtype=np.complex128).reshape(-1)


def _periodic_transfer_gain_bound(view: Any) -> float:
    """Bound a compact transfer on the normalized periodic frequency box."""

    try:
        from spf.renderer.block_filter import chart_matrix

        matrix = chart_matrix(view.chart, len(view.shape))
        inverse = np.linalg.inv(matrix)
        radius = 0.5 * np.sum(np.abs(inverse), axis=0)
        region = {"lo": -radius, "hi": radius}
        gain = float(v02.atom_bound(view.transfer.atom, region))
        for mixture in view.transfer.mixtures:
            # ``dropped_bound`` is an additive spectral magnitude envelope for
            # pruned mixture atoms. Include it in the operator norm used to
            # scale the source projection error so pruning and source
            # quantization cannot jointly under-report the terminal bound.
            gain *= math.fsum(
                [*(float(v02.atom_bound(atom, region)) for atom in mixture.atoms), float(mixture.dropped_bound)]
            )
        return gain if math.isfinite(gain) else math.inf
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError, OverflowError):
        return math.inf


class PreparedFourierSource:
    """Keep active Fourier coefficients sparse through filtering and synthesis."""

    preparation_fft_count = 0

    def __init__(
        self,
        view: Any,
        layout: Any,
        *,
        dtype: Any = _C128,
        periodic_transfer: bool = False,
    ) -> None:
        self.view, self.layout = view, layout
        self.periodic_transfer = bool(periodic_transfer)
        self.dtype = np.dtype(dtype)
        if self.dtype not in (_C64, _C128):
            raise ValueError("Fourier source compute dtype must be complex64 or complex128")
        self.addresses = view.active_addresses()
        self.mode_count = len(self.addresses)
        self.active_source_samples = self.mode_count
        total = math.prod(view.shape)
        full_lattice = bool(
            self.mode_count == total
            and np.array_equal(self.addresses, np.arange(total, dtype=self.addresses.dtype))
        )
        self.execution_fft_count = int(self.mode_count * layout.element_count > 4 * total * max(1, math.log2(total)))
        self.dense = bool(full_lattice and self.execution_fft_count)
        self.kind = "fourier_dense_terminal" if self.execution_fft_count else "fourier_sparse_synthesis"
        self.native = None
        self.native_consumer_used = False
        self.native_filter_path: str | None = None
        self.coefficients: np.ndarray = np.empty(self.mode_count, dtype=self.dtype)
        projection_l1_error = 0.0
        projection_max_error = 0.0
        if (
            self.periodic_transfer
            and self.dense
            and len(view.shape) == 2
            and view.observable == "complex"
            and view.scalar_bits == 32
            and view.log_offset == 0
            and view.phase_offset == 0
        ):
            from spf.renderer._filter_kernel import NativeTransfer

            candidate = NativeTransfer(view.transfer, view.chart, periodic=True)
            if candidate.supported:
                candidate.prepare(view.shape)
                words = np.ascontiguousarray(
                    view.sample_words(self.addresses, phase=True).reshape(view.shape),
                    dtype=np.uint32,
                )
                stored_grid = candidate.packed_grid(words, dtype=self.dtype)
                if self.dtype == _C64:
                    # Certify the terminal float32 cast without evaluating the
                    # whole packed filter a second time in float64.  For a
                    # rounded finite component y=fl(x),
                    # |x-y| <= u/(1-u)|y| plus one subnormal quantum.  Summing
                    # the real/imaginary component bounds also bounds complex
                    # magnitude error.  Canonical source NaNs preserve their
                    # class and therefore do not contribute a numeric error.
                    real = np.asarray(stored_grid.real, dtype=np.float64)
                    imag = np.asarray(stored_grid.imag, dtype=np.float64)
                    finite = np.isfinite(real) & np.isfinite(imag)
                    source_nan = words == np.uint32(v02.SPF32_WORD_NAN)
                    matched_nan = source_nan & np.isnan(real) & np.isnan(imag)
                    if np.all(finite | matched_nan):
                        unit_roundoff = np.finfo(np.float32).eps / 2.0
                        factor = unit_roundoff / (1.0 - unit_roundoff)
                        cast_bound = factor * (np.abs(real) + np.abs(imag))
                        nonzero_source = (words >> np.uint32(16)) != np.uint32(0x8000)
                        cast_bound += np.where(
                            finite & nonzero_source,
                            2.0 * float(np.finfo(np.float32).smallest_subnormal),
                            0.0,
                        )
                        cast_bound = np.where(finite, cast_bound, 0.0)
                        projection_l1_error = float(np.sum(cast_bound, dtype=np.float64))
                        projection_max_error = float(np.max(cast_bound, initial=0.0))
                    else:
                        projection_l1_error = projection_max_error = math.inf
                self.coefficients = np.asarray(stored_grid, dtype=self.dtype).reshape(-1)
                self.native = candidate
                self.native_consumer_used = True
                self.native_filter_path = (
                    "PreparedFourierSource -> NativeTransfer.packed_grid -> "
                    "spf_sfr_filter_packed_grid"
                )
                self.kind = "fourier_native_packed_terminal"
        if not self.native_consumer_used:
            for start in range(0, self.mode_count, 65536):
                addresses = self.addresses[start : start + 65536]
                projected = np.asarray(
                    view.project_words(
                        view.sample_words(addresses),
                        addresses,
                        periodic=self.periodic_transfer,
                    ),
                    dtype=_C128,
                )
                stored = np.asarray(projected, dtype=self.dtype)
                difference = np.abs(projected - stored.astype(_C128))
                projection_l1_error += float(np.sum(difference, dtype=np.float64))
                projection_max_error = max(projection_max_error, float(np.max(difference, initial=0.0)))
                self.coefficients[start : start + len(addresses)] = stored
        self.source_projection_l1_error = projection_l1_error
        self.source_projection_spectral_error = projection_max_error
        self.source_projection_error = projection_l1_error / max(1, total)
        self.source_projection_operator_gain = (
            _periodic_transfer_gain_bound(view) if self.periodic_transfer else 1.0
        )
        if self.dense:
            self.addresses = np.empty(0, dtype=np.int64)
        self.addresses.setflags(write=False)
        self.coefficients.setflags(write=False)
        self.nbytes = (
            self.addresses.nbytes
            + self.coefficients.nbytes
            + (0 if self.native is None else self.native.nbytes)
        )

    def execute(self) -> np.ndarray:
        shape, layout = self.view.shape, self.layout
        if self.execution_fft_count:
            if self.dense:
                spectrum = self.coefficients.reshape(shape)
            else:
                spectrum = np.zeros(shape, dtype=self.dtype)
                spectrum.ravel()[self.addresses] = self.coefficients
            dense = fft.ifftn(spectrum, workers=1)
            slices = tuple(
                slice(int(o), int(o + n * s), int(s))
                for o, n, s in zip(layout.origin, layout.shape, layout.spacing, strict=True)
            )
            return np.array(dense[slices], order=layout.order, copy=True)
        output = np.zeros(layout.shape, complex, order=layout.order)
        flat = output.ravel(order=layout.order)
        for start in range(0, self.mode_count, 1024):
            addresses = self.addresses[start : start + 1024]
            frequency = np.asarray(np.unravel_index(addresses, shape)).T / np.asarray(shape)
            coefficients = self.coefficients[start : start + 1024] / math.prod(shape)
            chunk = max(1, 65536 // len(addresses))
            for first in range(0, layout.element_count, chunk):
                last = min(first + chunk, layout.element_count)
                coords = np.asarray(np.unravel_index(np.arange(first, last), layout.shape, order=layout.order)).T
                coords = np.asarray(layout.origin) + coords * np.asarray(layout.spacing)
                flat[first:last] += np.exp(2j * np.pi * (coords @ frequency.T)) @ coefficients
        return output
