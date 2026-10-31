"""Continuous atom-source filtering: parameter transforms, never a grid FFT."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from spf.renderer.model import (
    CoordinateChart,
    Domain,
    Field,
    MaterializationRequest,
    SampleLayout,
    Support,
    SupportKind,
)
from spf.renderer.transfers import SpectralTransfer
from spf.tensor import v02


def is_atom_source(value: Any) -> bool:
    return isinstance(value, (v02.GaussianChirpAtom, v02.AtomMixture))


class PreparedAtomFilterChain:
    """Keep the source and all closed spectral operations as Gaussian atoms.

    Atom fields have continuous whole-space semantics. Packed/sample grids
    keep their periodic discrete semantics in the block/array consumers.
    """

    def __init__(self, chain: Any, output: str, layout: SampleLayout | None = None) -> None:
        if output not in {"spatial", "dense", "frequency", "spectrum", "spectrum_intensity", "intensity"}:
            raise ValueError(f"unsupported filter-chain output {output!r}")
        self.chain, self.output = chain, output
        from spf.renderer import runtime

        source = chain.source
        support = chain.metadata.get("source_support", Support.full())
        if support.kind not in {SupportKind.FULL, SupportKind.EMPTY}:
            raise ValueError(
                "a windowed atom requires an explicit sampled-field boundary; whole-space closure is invalid"
            )
        atoms = (source,) if isinstance(source, v02.GaussianChirpAtom) else source.atoms
        if isinstance(source, v02.AtomMixture) and source.residual is not None:
            raise ValueError("atom FilterChain requires an analytic mixture; bind its residual as a separate field")
        transfer = SpectralTransfer.identity(2)
        for operation in chain.operations:
            op = chain._field_operation(operation)
            transfer = transfer.apply(replace(op, params={**op.params, "shape": chain.shape}))
        if transfer.phase_residuals or transfer.windows:
            raise ValueError(
                "sampled spectral phase/LOD requires an explicit sampled-field boundary for an atom source"
            )
        spectral = output in {"frequency", "spectrum", "spectrum_intensity", "intensity"}
        direct_spatial = (
            not spectral
            and not chain.source_is_fourier
            and all(op.name.value in {"translate", "exposure", "phase", "rotate", "affine"} for op in chain.operations)
        )
        if direct_spatial:
            terms = tuple(atoms)
            source_field = Field.from_value(source, domain=Domain.SPATIAL_2D)
            for op in chain.operations:
                terms = tuple(runtime._apply_atom_operation(a, chain._field_operation(op), source_field) for a in terms)
        else:
            terms = tuple(atoms) if chain.source_is_fourier else tuple(v02.atom_fourier(a) for a in atoms)
            if transfer.coordinate is not None:
                # §17.2/§17.3: the source spectrum is read at ``coordinate @ k``.
                # For an atom that read is an exact parameter change, so the
                # continuous chain stays closed with no resampling at all.
                terms = tuple(v02.atom_affine(a, transfer.coordinate, np.zeros(2)) for a in terms)
            terms = tuple(v02.atom_mul(a, transfer.atom) for a in terms)
            for mixture in transfer.mixtures:
                terms = tuple(v02.atom_mul(a, b) for a in terms for b in mixture.atoms)
            if not spectral:
                terms = tuple(v02.atom_inverse_fourier(a) for a in terms)
        self.field = Field.from_value(
            v02.AtomMixture(terms),
            domain=Domain.FOURIER_2D if spectral else Domain.SPATIAL_2D,
            chart=chain.metadata.get("source_chart") or CoordinateChart.identity(2),
            support=support,
        )
        self.field = replace(
            self.field,
            representation=replace(
                self.field.representation, error=chain.metadata.get("source_error", v02.ErrorLedger())
            ),
        )
        ledger = self.field.representation.error
        if isinstance(source, v02.AtomMixture) and not chain.metadata.get("source_error_includes_payload"):
            ledger = ledger.plus(v02.ErrorLedger(pruning=source.dropped_bound))
        if ledger.total_bound:
            # A whole-space pointwise source bound alone does not bound its
            # Fourier integral or a unitary propagator in the infinity norm.
            # Do not manufacture a finite certificate from a finite raster N.
            unbounded = spectral != chain.source_is_fourier or any(
                op.name.value in {"fresnel", "lens_phase"} for op in chain.operations
            )
            if unbounded and (
                spectral
                or chain.source_is_fourier
                or any(op.name.value in {"fresnel", "lens_phase"} for op in chain.operations)
            ):
                raise ValueError("continuous transform of an uncertain atom requires an integrable error envelope")
            gain = math.exp(transfer.atom.kappa.real)
            ledger = runtime._scale_ledger(ledger, gain)
        self.field = replace(self.field, representation=replace(self.field.representation, error=ledger))
        self.layout = layout or SampleLayout(chain.shape, spacing=chain.sample_spacing)
        self.prepared = self.field.prepare(
            MaterializationRequest(
                layout=self.layout,
                output_format="complex",
                precision=chain.precision,
                error_budget=chain.error_budget,
            )
        )
        self.transfer = transfer
        self.terms = len(terms)

    def execute(self) -> Any:
        from spf.renderer import runtime
        from spf.renderer.filter_chain import FilterResult, FilterVariant

        result, report = self.prepared.execute(return_report=True)
        values = np.asarray(result.values if hasattr(result, "values") else result)
        ledger = report.ledger
        if self.output in {"intensity", "spectrum_intensity"}:
            ledger = runtime._scale_ledger(ledger, 2 * runtime._max_abs(values) + ledger.total_bound)
            values = np.abs(values) ** 2
        dtype = self.chain.metadata.get("compute_dtype")
        if dtype is not None:
            if dtype not in {"complex64", "complex128"}:
                raise ValueError("filter compute_dtype must be complex64 or complex128")
            converted = np.asarray(
                values,
                dtype=("float32" if dtype == "complex64" else "float64")
                if self.output in {"intensity", "spectrum_intensity"}
                else dtype,
            )
            ledger = ledger.plus(v02.ErrorLedger(materialization=runtime._max_abs(converted - values)))
            values = converted
        runtime._enforce_error_budget(self.field, self.prepared.request, ledger, values, 0.0)
        report = replace(
            report,
            ledger=ledger,
            diagnostics={
                **report.diagnostics,
                "source_carrier": type(self.chain.source).__name__,
                "consumer": "analytic_atom_filter_chain",
                "source_transform": "continuous_atom_parameters",
                "source_atoms": 1
                if isinstance(self.chain.source, v02.GaussianChirpAtom)
                else len(self.chain.source.atoms),
                "output_atoms": self.terms,
                "analytic_operations": self.transfer.operations,
                "dense_transfer_elements": 0,
                "preparation_fft_count": 0,
                "cached_terminal_values": 0,
            },
        )
        report.metrics.fft_count = 0
        report.metrics.representation = "analytic_atom_filter_chain"
        return FilterResult(values, report, FilterVariant.STRUCTURED)

    run = execute
