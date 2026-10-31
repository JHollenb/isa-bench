"""Batched atom header evaluation with a compensated, one-shot tile pivot."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

from spf.renderer._jet_kernel import native_function, pointer


def evaluate_atoms(atoms: Any, points: np.ndarray) -> np.ndarray | None:
    if points.ndim != 2 or points.shape[1] != 2 or any(a.dimension != 2 for a in atoms):
        return None
    ptr = ctypes.c_void_p
    fn = native_function("spf_sfr_atom_mixture", (ptr, ptr, ctypes.c_size_t, ptr, ctypes.c_size_t, ptr))
    if fn is None:
        return None
    headers = np.asarray(
        [[a.kappa, a.h[0], a.h[1], -a.C[0, 0], -a.C[0, 1], -a.C[1, 1]] for a in atoms], dtype=np.complex128
    ).reshape(-1, 6)
    log = np.ascontiguousarray(headers.real)
    phase = np.ascontiguousarray(headers.imag) if np.any(headers.imag) else None
    coordinates = np.ascontiguousarray(points, dtype=np.float64)
    output = np.empty(points.shape[0], dtype=np.complex128)
    fn(pointer(log), pointer(phase), len(atoms), pointer(coordinates), points.shape[0], pointer(output))
    return output


class PreparedAtomGridTerminal:
    """Parameter-only preparation; bounded points and coherent sums at output."""

    def __init__(self, field: Any, request: Any) -> None:
        from spf.tensor import v02

        self.field, self.request = field, request
        payload = field.representation.payload
        atoms = (payload,) if isinstance(payload, v02.GaussianChirpAtom) else payload.atoms
        chart = field.chart
        if chart.dimension:
            matrix = np.asarray(chart.basis) @ np.diag(np.asarray(chart.scale) * np.asarray(chart.sample_spacing))
            atoms = tuple(v02.atom_affine(a, matrix, np.asarray(chart.origin)) for a in atoms)
        self.atoms = atoms

    def execute(self) -> Any:
        from spf.renderer import runtime
        from spf.renderer.model import SupportKind
        from spf.renderer.native import NativeProjectionResult
        from spf.renderer.report import RenderMetrics, RenderReport

        layout = self.request.layout
        values = np.zeros(layout.shape, complex, order=layout.order)
        flat = values.ravel(order=layout.order)
        native = True
        for start in range(0, 0 if self.field.support.kind is SupportKind.EMPTY else layout.element_count, 65536):
            stop = min(start + 65536, layout.element_count)
            row, col = np.unravel_index(np.arange(start, stop), layout.shape, order=layout.order)
            # Axis order (design §17.1 convention, fixed 2026-09-18): coordinate
            # component i belongs to array axis i and carries spacing[i].
            points = np.column_stack(
                (layout.origin[0] + row * layout.spacing[0], layout.origin[1] + col * layout.spacing[1])
            )
            evaluated = evaluate_atoms(self.atoms, points)
            if evaluated is None:
                # Compensated coherent reduction preserves cancellation; no
                # atom-count-sized stack of raster planes is constructed.
                evaluated = np.zeros(len(points), complex)
                compensation = np.zeros(len(points), complex)
                for atom in self.atoms:
                    y = atom.evaluate(points) - compensation
                    updated = evaluated + y
                    compensation = (updated - evaluated) - y
                    evaluated = updated
                native = False
            flat[start:stop] = evaluated
        native_values, source, kind = runtime._semantic_source(self.field, values)
        context = runtime._SamplingContext(np.empty((0, 2)), layout.shape, layout.order, layout, None)
        output, observed, phase, format_ledger, _ = runtime._format_output(
            self.field, self.request, native_values, source, kind, context, ()
        )
        from spf.renderer.normalization import _field_ledger_after_operations

        ledger = runtime._apply_representation_error(self.field, _field_ledger_after_operations(self.field)).plus(
            format_ledger
        )
        runtime._enforce_error_budget(self.field, self.request, ledger, source, phase)
        report = RenderReport(
            RenderMetrics(
                backend="cpu",
                workload="analytic_atom_grid",
                resolution=layout.shape,
                representation="analytic_atom_grid",
                fft_count=0,
                atoms_considered=len(self.atoms),
                reduction_operations=layout.element_count if len(self.atoms) > 1 else 0,
                terminal_output_elements=layout.element_count,
                terminal_output_bytes=output.nbytes,
                materialized_elements=layout.element_count,
                bytes_read=len(self.atoms) * 112,
                bytes_written=output.nbytes,
            ),
            ledger,
            capabilities_used=(
                "analytic_atom_grid",
                "native_compensated_atom_pivot" if native else "compensated_atom_sum",
            ),
            diagnostics={
                "dense_transfer_elements": 0,
                "cached_pixels": 0,
                "coordinate_chunk_limit": 65536,
                "preparation_fft_count": 0,
                "output_observed_error": observed,
                "output_scale": runtime._max_abs(source),
                "native_atom_consumer": native,
                "transform_arithmetic": "none",
            },
        )
        return NativeProjectionResult(output, report, np.empty(0, np.int16))


def prepare_atom_grid(field: Any, request: Any) -> PreparedAtomGridTerminal | None:
    from spf.renderer.model import Domain, SupportKind
    from spf.tensor import v02

    payload, layout = field.representation.payload, request.layout
    if (
        field.operations
        or field.representation.residual is not None
        or request.region is not None
        or layout is None
        or len(layout.shape) != 2
        or field.support.kind not in {SupportKind.FULL, SupportKind.EMPTY}
        or field.domain not in {Domain.SPATIAL_2D, Domain.FOURIER_2D}
        or request.output_format != "complex"
    ):
        return None
    atoms = (
        (payload,)
        if isinstance(payload, v02.GaussianChirpAtom)
        else payload.atoms
        if isinstance(payload, v02.AtomMixture) and payload.residual is None
        else None
    )
    if atoms is None or any(a.dimension != 2 for a in atoms):
        return None
    return PreparedAtomGridTerminal(field, request)
