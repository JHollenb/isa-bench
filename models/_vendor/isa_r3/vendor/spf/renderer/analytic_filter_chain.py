"""Dense-source ingress to the same compact spectral field program as blocks."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf.renderer._filter_kernel import NativeTransfer
from spf.renderer.block_filter import CHUNK, chart_matrix
from spf.renderer.model import CoordinateChart, FieldOperation
from spf.renderer.transfers import SpectralTransfer
from spf.tensor import v02


class PreparedAnalyticFilterChain:
    """Prepared source spectrum plus compact jets/mixtures; no dense diagonal."""

    def __init__(self, chain: Any, output: str) -> None:
        if output not in {"spatial", "dense", "frequency", "spectrum", "spectrum_intensity", "intensity"}:
            raise ValueError(f"unsupported filter-chain output {output!r}")
        self.chain, self.output = chain, output
        self.phase_eliminated = output in {"spectrum_intensity", "intensity"}
        self.dtype = np.dtype(chain.metadata.get("compute_dtype", "complex128"))
        if self.dtype not in (np.dtype("complex64"), np.dtype("complex128")):
            raise ValueError("filter compute_dtype must be complex64 or complex128")
        # Axis-order chart (design §17.1 convention): axis i carries spacing[i].
        # The user-facing (x, y) -> axis-order exchange happens once, in
        # ``FilterChain._field_operation``, not inside the chart basis.
        self.chart = chain.metadata.get("source_chart") or CoordinateChart.identity(2, spacing=chain.sample_spacing)
        self.transfer = SpectralTransfer.identity(2)
        for operation in chain.operations:
            op = chain._field_operation(operation)
            self.transfer = self.transfer.apply(FieldOperation(op.op, {**op.params, "shape": chain.shape}))
        self.native = NativeTransfer(self.transfer, self.chart, periodic=not chain.source_is_fourier)
        self.native.prepare(chain.shape)
        self.matrix = chart_matrix(self.chart, 2)
        # Raw numeric arrays already have a declared Cartesian boundary. Packed
        # SPF and texture ingress continue to use their semantic decoder.
        if isinstance(chain.source, np.ndarray) and chain.source.dtype.kind in "fc":
            source = np.asarray(chain.source, dtype=self.dtype)
        else:
            source = np.asarray(chain._source_values(), dtype=self.dtype)
        self.spectrum = source.copy() if chain.source_is_fourier else fft.fftn(source, workers=1)
        self.resample_bound = 0.0
        self.resample_conversions = 0
        if self.transfer.coordinate is not None:
            # Design §17.2/§17.3: a coordinate remap is not diagonal, so the
            # source spectrum is read at ``coordinate @ k``.  The |det A|^-1
            # amplitude already sits on the transfer's own atom.  Sampled
            # spectra have no closed form for that read, so this is the counted
            # reference resample with a stated bound.
            from spf.renderer import runtime

            basis = np.linalg.inv(self.matrix).T @ np.diag([1.0 / n for n in chain.shape])
            bins = np.linalg.inv(basis) @ self.transfer.coordinate @ basis
            self.spectrum, self.resample_bound = runtime._reference_affine_resample(
                self.spectrum, bins, np.zeros(2), (1.0, 1.0)
            )
            self.resample_conversions = 1
        self.spectrum.setflags(write=False)
        self.hoisted = output in {"spatial", "dense"}
        if self.hoisted:
            # The source and operator are immutable. Bind the whole spectral
            # prefix; each execution still performs the requested inverse FFT.
            # Spectral terminals do not take this path: never cache an output.
            self.spectrum = self._filtered()
            self.spectrum.setflags(write=False)
            self.native.seeds = np.empty(0, dtype=np.uint8)
            self.native.seed_shape = ()
        self.report = self._report()

    def _report(self) -> Any:
        from spf.renderer.filter_chain import FilterVariant

        chain, count = self.chain, math.prod(self.chain.shape)
        spectral = self.output in {"frequency", "spectrum", "spectrum_intensity", "intensity"}
        report = chain._report(
            FilterVariant.STRUCTURED,
            fft_count=int(not spectral),
            materialized=count,
            potential=count * max(1, len(chain.operations) + 1),
            bytes_read=self.spectrum.nbytes + self.transfer.nbytes,
            bytes_written=count * (self.dtype.itemsize // 2 if self.phase_eliminated else self.dtype.itemsize),
            fallback_conversions=self.resample_conversions,
            ledger=chain.metadata.get("source_error", v02.ErrorLedger()).plus(
                v02.ErrorLedger(transform=self.resample_bound)
            ),
            diagnostics={
                "structured": True,
                "reference_only": False,
                "structured_reference_dense": False,
                "dense_transfer_elements": 0,
                "source_carrier": f"dense_{self.dtype.name}_spectrum",
                "operator_carrier": "GaussianChirpAtom + full jets + positive mixtures",
                "spf_block_tensor_matrix_used": False,
                "analytic_operations": self.transfer.operations,
                "dense_operations": (),
                "operator_bytes": self.transfer.nbytes,
                "prepared_recurrence_bytes": self.native.seeds.nbytes,
                "spectral_prefix_hoisted": self.hoisted,
                "prepared_spectrum_state": "post-filter" if self.hoisted else "source",
                "cached_terminal_values": 0,
                "prepared": True,
                "preparation_fft_count": int(not chain.source_is_fourier),
                "cached_spatial_pixels": 0,
                "terminal_phase_eliminated": self.phase_eliminated,
                "native_filter_consumer": self.native.supported,
                "consumer": "native_quadratic_recurrence" if self.native.supported else "bounded_quadratic_transfer",
                "compute_dtype": self.dtype.name,
                "numeric_error_status": "source ledger retained; FFT/recurrence rounding externally measured",
                "byte_traffic_status": "logical carrier I/O; excludes FFT traffic and arithmetic scratch",
                "memory_status": "logical arrays; total resident peak not measured",
            },
        )
        report.metrics.backend = "cpu"
        report.metrics.representation = "analytic_transfer_dense_source"
        report.metrics.precision = self.dtype.name
        return replace(
            report,
            capabilities_used=(
                "analytic_spectral_transfer",
                "quadratic_recurrence" if self.native.supported else "bounded_transfer_evaluation",
            ),
        )

    def _filtered(self) -> np.ndarray:
        if self.native.supported:
            return self.native.grid(self.spectrum, intensity=self.phase_eliminated)
        shape = self.chain.shape
        dtype = self.spectrum.real.dtype if self.phase_eliminated else self.dtype
        output = np.empty(shape, dtype=dtype)
        total = math.prod(shape)
        inverse = np.linalg.inv(self.matrix)
        for start in range(0, total, CHUNK):
            addresses = np.arange(start, min(total, start + CHUNK))
            q = np.asarray(np.unravel_index(addresses, shape)).T.astype(float)
            if self.chain.source_is_fourier:
                points = q @ self.matrix.T + self.chart.origin
            else:
                dims = np.asarray(shape)
                q = np.where(q >= (dims + 1) // 2, q - dims, q) / dims
                points = q @ inverse
            logs, phase = self.transfer.log_phase(points, addresses, include_phase=not self.phase_eliminated)
            src = self.spectrum.reshape(-1)[start : start + len(addresses)]
            result = (
                np.abs(src) ** 2 * np.exp2(2 * logs)
                if self.phase_eliminated
                else src * np.exp2(logs) * np.exp(1j * phase)
            )
            output.reshape(-1)[start : start + len(addresses)] = result
        return output

    def execute(self) -> Any:
        from spf.renderer import runtime
        from spf.renderer.filter_chain import FilterResult, FilterVariant

        filtered = self.spectrum if self.hoisted else self._filtered()
        values = fft.ifftn(filtered, workers=1) if self.output in {"spatial", "dense"} else filtered
        if (
            "compute_dtype" not in self.chain.metadata
            and self.output in {"spatial", "dense"}
            and np.all(np.abs(values.imag) <= 1e-12)
        ):
            values = values.real
        report = replace(self.report, metrics=replace(self.report.metrics))
        ledger = report.ledger
        if ledger.total_bound:
            count = math.prod(self.chain.shape)
            gain = math.exp(complex(self.transfer.atom.kappa).real)
            if not self.chain.source_is_fourier:
                gain *= math.sqrt(count) if self.output in {"spatial", "dense"} else count
            ledger = runtime._scale_ledger(ledger, gain)
            if self.phase_eliminated:
                ledger = runtime._scale_ledger(ledger, 2 * math.sqrt(runtime._max_abs(values)) + ledger.total_bound)
            if not self.chain.error_budget.as_contract().accepts(ledger, scale=runtime._max_abs(values)):
                raise runtime.MaterializationError("filter chain exceeds requested ErrorBudget")
            report = replace(report, ledger=ledger)
        return FilterResult(values, report, FilterVariant.STRUCTURED)

    run = execute
