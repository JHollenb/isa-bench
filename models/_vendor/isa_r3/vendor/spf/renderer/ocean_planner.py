"""Typed planning for the public ocean carriers.

The planner is intentionally metadata-only.  A selected plan is a concrete
program made of typed operations; it is not a label attached after a dense
array has already been built.  ``OceanCarrier`` remains lazy until the chosen
consumer asks for its addressed words.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.ocean_carrier import OceanCarrier
from spf.renderer.ocean_transfer import filtered_view_operations

_OP_SIGNATURES: dict[str, tuple[tuple[int, int], tuple[str, ...]]] = {
    "AddressSupport": ((1, 1), ("read_structured_support",)),
    "SourceTransfer": ((1, 1), ("frequency_transfer",)),
    "PhaseCompose": ((2, 2), ("modular_phase_update", "round_at_emission")),
    "FootprintTransfer": ((2, 2), ("frequency_transfer",)),
    "DerivativeLogMul": ((2, 2), ("log_shift", "exact_quarter_turn", "shared_coefficient_emission")),
    "HermitianSynthesis": ((1, 2), ("pair_reduce", "real_output")),
    "ReduceSigned": ((1, 1), ("shared_phase_basis",)),
    "ClosedFormDirichlet": ((1, 2), ("direct_dirichlet_identity", "stable_sinc_limits", "real_output")),
    "DiagonalCollapsedSynthesis": (
        (1, 2),
        ("one_dimensional_support", "pair_reduce", "shared_phase_basis", "real_output"),
    ),
    "AnalyticAtomJet": ((1, 2), ("analytic_gradient", "preserve_residuals")),
    "Materialize": ((1, 1), ("output_dtype_boundary",)),
}

_VALUE_TYPES: dict[str, str] = {
    "carrier": "Carrier",
    "wide_phase": "WidePhaseState",
    "footprint": "Footprint",
    "derivative_operator": "DerivativeOperator",
    "analytic_carrier": "AnalyticCarrier",
    "indices": "IndexSet",
    "packed_coefficients": "PackedCoefficients",
    "phase_coefficients": "PackedCoefficients",
    "filtered_packed_coefficients": "PackedCoefficients",
    "source_filtered_coefficients": "PackedCoefficients",
    "derivative_terms": "DerivativeTerms",
    "signed_terms": "SignedTerms",
    "fields": "Fields",
    "output": "MaterializedFields",
}


@dataclass(frozen=True, slots=True)
class OceanInstruction:
    """One typed ocean IR operation and its declared effects."""

    opcode: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    effects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "opcode", str(self.opcode))
        object.__setattr__(self, "inputs", tuple(str(x) for x in self.inputs))
        object.__setattr__(self, "outputs", tuple(str(x) for x in self.outputs))
        object.__setattr__(self, "attributes", MappingProxyType(dict(self.attributes)))
        object.__setattr__(self, "effects", tuple(str(x) for x in self.effects))
        if self.opcode not in _OP_SIGNATURES:
            raise ValueError(f"unsupported ocean instruction opcode {self.opcode!r}")
        (minimum, maximum), required_effects = _OP_SIGNATURES[self.opcode]
        if not minimum <= len(self.inputs) <= maximum:
            raise ValueError(
                f"ocean instruction {self.opcode} expects {minimum}..{maximum} inputs, got {len(self.inputs)}"
            )
        expected_outputs = 2 if self.opcode == "AddressSupport" else 1
        if len(self.outputs) != expected_outputs:
            raise ValueError(f"ocean instruction {self.opcode} has invalid output arity")
        if len(set(self.outputs)) != len(self.outputs):
            raise ValueError(f"ocean instruction {self.opcode} has duplicate outputs")
        missing_effects = tuple(effect for effect in required_effects if effect not in self.effects)
        if missing_effects:
            raise ValueError(f"ocean instruction {self.opcode} omits declared effects {missing_effects!r}")
        allowed_outputs: dict[str, set[tuple[str, ...]]] = {
            "AddressSupport": {("indices", "packed_coefficients")},
            "SourceTransfer": {("source_filtered_coefficients",)},
            "PhaseCompose": {("phase_coefficients",)},
            "FootprintTransfer": {("filtered_packed_coefficients",)},
            "DerivativeLogMul": {("derivative_terms",)},
            "HermitianSynthesis": {("signed_terms",), ("fields",)},
            "ReduceSigned": {("fields",)},
            "ClosedFormDirichlet": {("fields",)},
            "DiagonalCollapsedSynthesis": {("fields",)},
            "AnalyticAtomJet": {("fields",)},
            "Materialize": {("output",)},
        }
        if self.outputs not in allowed_outputs[self.opcode]:
            raise ValueError(f"ocean instruction {self.opcode} has invalid output values {self.outputs!r}")


@dataclass(frozen=True, slots=True)
class OceanProgram:
    """Immutable structural program exposed to diagnostics and display code."""

    instructions: tuple[OceanInstruction, ...]
    input_carrier: str
    output_fields: tuple[str, ...]
    effects: tuple[str, ...] = ()
    input_types: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    source_metadata: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        # Validate the program as a small SSA graph.  This catches a planner
        # bug at construction time instead of leaving a descriptive trace that
        # the dispatcher cannot actually consume.
        external_types = {
            "carrier": "Carrier",
            "wide_phase": "WidePhaseState",
            "footprint": "Footprint",
            "derivative_operator": "DerivativeOperator",
            "analytic_carrier": "AnalyticCarrier",
        }
        declared_types = dict(self.input_types)
        unknown_declared = tuple(name for name in declared_types if name not in external_types)
        if unknown_declared:
            raise ValueError(f"ocean program declares unknown external values {unknown_declared!r}")
        for name, value_type in declared_types.items():
            if external_types[name] != value_type:
                raise ValueError(f"ocean external {name!r} has incompatible type {value_type!r}")
        available = set(external_types)
        value_types = dict(_VALUE_TYPES)
        for instruction in self.instructions:
            undeclared = tuple(
                name for name in instruction.inputs if name in external_types and name not in declared_types
            )
            if undeclared:
                raise ValueError(f"ocean program uses undeclared external values {undeclared!r}")
            missing = tuple(name for name in instruction.inputs if name not in available)
            if missing:
                raise ValueError(f"ocean program {instruction.opcode} has undefined inputs {missing!r}")
            duplicate = tuple(name for name in instruction.outputs if name in available)
            if duplicate:
                raise ValueError(f"ocean program {instruction.opcode} redefines SSA values {duplicate!r}")
            for output in instruction.outputs:
                if output not in value_types:
                    raise ValueError(f"ocean instruction {instruction.opcode} has unknown output value {output!r}")
            available.update(instruction.outputs)
            for output in instruction.outputs:
                value_types[output] = _VALUE_TYPES[output]
            if instruction.opcode == "AddressSupport" and value_types.get(instruction.inputs[0]) != "Carrier":
                raise ValueError("AddressSupport must consume a Carrier")
            if instruction.opcode == "PhaseCompose" and (
                value_types.get(instruction.inputs[0]) != "WidePhaseState"
                or value_types.get(instruction.inputs[1]) != "PackedCoefficients"
            ):
                raise ValueError("PhaseCompose must consume WidePhaseState and PackedCoefficients")
            if (
                instruction.opcode == "SourceTransfer"
                and value_types.get(instruction.inputs[0]) != "PackedCoefficients"
            ):
                raise ValueError("SourceTransfer must consume PackedCoefficients")
            if instruction.opcode == "FootprintTransfer" and (
                value_types.get(instruction.inputs[0]) != "PackedCoefficients"
                or value_types.get(instruction.inputs[1]) != "Footprint"
            ):
                raise ValueError(f"{instruction.opcode} must consume a Footprint")
            if instruction.opcode == "DerivativeLogMul" and (
                value_types.get(instruction.inputs[0]) != "PackedCoefficients"
                or value_types.get(instruction.inputs[1]) != "DerivativeOperator"
            ):
                raise ValueError("DerivativeLogMul must consume PackedCoefficients and DerivativeOperator")
            if instruction.opcode == "HermitianSynthesis" and (
                value_types.get(instruction.inputs[0]) != "PackedCoefficients"
                or (len(instruction.inputs) == 2 and value_types.get(instruction.inputs[1]) != "DerivativeTerms")
            ):
                raise ValueError("HermitianSynthesis must consume packed coefficients and optional derivative terms")
            if instruction.opcode in {"ClosedFormDirichlet", "DiagonalCollapsedSynthesis"} and (
                value_types.get(instruction.inputs[0]) != "PackedCoefficients"
                or (len(instruction.inputs) == 2 and value_types.get(instruction.inputs[1]) != "DerivativeTerms")
            ):
                raise ValueError(f"{instruction.opcode} has incompatible source types")
            if instruction.opcode == "ReduceSigned" and value_types.get(instruction.inputs[0]) != "SignedTerms":
                raise ValueError("ReduceSigned must consume SignedTerms")
            if instruction.opcode == "Materialize" and value_types.get(instruction.inputs[0]) != "Fields":
                raise ValueError("Materialize must consume Fields")
            if instruction.opcode == "AnalyticAtomJet" and value_types.get(instruction.inputs[0]) != "AnalyticCarrier":
                raise ValueError("AnalyticAtomJet must consume AnalyticCarrier")
            if (
                instruction.opcode == "AnalyticAtomJet"
                and len(instruction.inputs) == 2
                and value_types.get(instruction.inputs[1]) != "Footprint"
            ):
                raise ValueError("AnalyticAtomJet must consume an optional Footprint")
        if "output" not in available:
            raise ValueError("ocean program has no materialized output")
        object.__setattr__(self, "input_types", MappingProxyType(dict(declared_types)))
        object.__setattr__(self, "source_metadata", MappingProxyType(dict(self.source_metadata)))

    @property
    def opcodes(self) -> tuple[str, ...]:
        return tuple(item.opcode for item in self.instructions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_carrier": self.input_carrier,
            "output_fields": self.output_fields,
            "effects": self.effects,
            "input_types": dict(self.input_types),
            "source_metadata": dict(self.source_metadata),
            "instructions": [
                {
                    "opcode": item.opcode,
                    "inputs": item.inputs,
                    "outputs": item.outputs,
                    "attributes": dict(item.attributes),
                    "effects": item.effects,
                }
                for item in self.instructions
            ],
        }


def _normal_method(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower().replace("-", "_")


_OCEAN_OUTPUTS = ("height", "grad_y", "grad_x")


def _footprint_descriptor(value: Any) -> Any:
    """Return a stable, JSON-friendly description of an opaque footprint."""

    if value is None:
        return None
    covariance = getattr(value, "covariance", None)
    if covariance is None and isinstance(value, Mapping):
        covariance = value.get("covariance")
    if covariance is not None:
        try:
            return {
                "type": type(value).__name__,
                "covariance": tuple(tuple(float(item) for item in row) for row in np.asarray(covariance).tolist()),
            }
        except (TypeError, ValueError):
            # The terminal owns the detailed validation.  Planning should
            # still expose that an opaque value was supplied without trying
            # to serialize malformed user input.
            return {"type": type(value).__name__, "covariance": "unserializable"}
    return {"type": type(value).__name__}


@dataclass(frozen=True, slots=True)
class OceanOutputRequest:
    """Complete output contract consumed by one ocean execution decision.

    ``OceanCarrier`` planning can remain metadata-only, but the final route
    cannot be selected until the output shape, requested planes, footprint,
    backend, and output dtype are known.  ``point_count`` is sufficient for
    cost planning; the terminal retains the actual point coordinates.
    """

    sampling: str = "grid"
    outputs: tuple[str, ...] = _OCEAN_OUTPUTS
    footprint: Any = None
    backend: str = "reference"
    output_dtype: str = "float32"
    point_count: int | None = None
    native: bool | None = None

    def __post_init__(self) -> None:
        sampling = str(self.sampling).strip().lower().replace("-", "_")
        sampling = {"query": "points", "queries": "points"}.get(sampling, sampling)
        if sampling not in {"grid", "points"}:
            raise ValueError("ocean output sampling must be 'grid' or 'points'")
        raw_outputs = (self.outputs,) if isinstance(self.outputs, str) else tuple(self.outputs)
        outputs = tuple(str(value).strip().lower() for value in raw_outputs)
        if not outputs:
            raise ValueError("ocean output request must contain at least one output plane")
        unknown = tuple(value for value in outputs if value not in _OCEAN_OUTPUTS)
        if unknown:
            raise ValueError(f"unsupported ocean output planes {unknown!r}")
        if len(set(outputs)) != len(outputs):
            raise ValueError("ocean output request contains duplicate output planes")
        backend = str(getattr(self.backend, "name", self.backend)).strip().lower().replace("-", "_")
        if not backend:
            raise ValueError("ocean output backend must be non-empty")
        dtype = str(np.dtype(self.output_dtype))
        if dtype not in {"float32", "float64"}:
            raise TypeError("ocean output_dtype must be numpy.float32 or numpy.float64")
        if self.point_count is None:
            point_count = None
        else:
            if isinstance(self.point_count, (bool, np.bool_)) or int(self.point_count) != self.point_count:
                raise TypeError("ocean point_count must be an integer")
            point_count = int(self.point_count)
            if point_count < 0:
                raise ValueError("ocean point_count must be non-negative")
        if sampling == "points" and point_count is None:
            # A binding may not know the coordinates yet (the display callback
            # supplies them lazily).  Keep that contract explicit and use one
            # conservative point for the cost fallback rather than inventing a
            # dense field shape.
            point_count = 0
        if self.native not in (None, True, False):
            raise TypeError("ocean native preference must be True, False, or None")
        object.__setattr__(self, "sampling", sampling)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "output_dtype", dtype)
        object.__setattr__(self, "point_count", point_count)

    @property
    def query(self) -> bool:
        return self.sampling == "points"

    @property
    def need_height(self) -> bool:
        return "height" in self.outputs

    @property
    def need_gradients(self) -> bool:
        return "grad_y" in self.outputs or "grad_x" in self.outputs

    @property
    def gradient_axes(self) -> tuple[int, ...]:
        return tuple(axis for axis, name in ((0, "grad_y"), (1, "grad_x")) if name in self.outputs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampling": self.sampling,
            "outputs": self.outputs,
            "footprint": _footprint_descriptor(self.footprint),
            "backend": self.backend,
            "output_dtype": self.output_dtype,
            "point_count": self.point_count,
            "native": self.native,
        }


@dataclass(frozen=True, slots=True)
class OceanExecutionPlan:
    """Costed route selection, retaining the old synthesis-plan fields."""

    shape: tuple[int, ...]
    method: str
    active_modes: int
    potential_elements: int
    sparse_cost: float
    irfft_cost: float
    reason: str
    fallback: str | None = None
    route: str | None = None
    backend: str = "reference"
    profile: str = "sparse_reference"
    native: bool = False
    structure_preserved: bool = False
    source_kind: str = "sparse_modes"
    supports_queries: bool = True
    supports_footprints: bool = True
    packed_native: bool = False
    program: OceanProgram | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    request: OceanOutputRequest | None = None
    executor: str | None = None
    kernel: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(x) for x in self.shape))
        object.__setattr__(self, "method", str(self.method))
        object.__setattr__(self, "route", self.method if self.route is None else str(self.route))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))
        if self.request is not None and not isinstance(self.request, OceanOutputRequest):
            object.__setattr__(self, "request", OceanOutputRequest(**dict(self.request)))
        if self.executor is not None:
            object.__setattr__(self, "executor", str(self.executor))
        if self.kernel is not None:
            object.__setattr__(self, "kernel", str(self.kernel))

    @property
    def selected_method(self) -> str:
        return self.method

    @property
    def requested_method(self) -> str:
        """Original caller route before compatibility aliases were applied."""

        return str(self.diagnostics.get("requested_method", self.method))

    @property
    def dense_cost(self) -> float:
        return self.irfft_cost

    @property
    def structural_program(self) -> OceanProgram | None:
        return self.program

    @property
    def operations(self) -> tuple[OceanInstruction, ...]:
        return () if self.program is None else self.program.instructions

    @property
    def bound(self) -> bool:
        return self.request is not None and self.executor is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": self.shape,
            "method": self.method,
            "requested_method": self.requested_method,
            "route": self.route,
            "active_modes": self.active_modes,
            "potential_elements": self.potential_elements,
            "sparse_cost": self.sparse_cost,
            "irfft_cost": self.irfft_cost,
            "reason": self.reason,
            "fallback": self.fallback,
            "backend": self.backend,
            "profile": self.profile,
            "native": self.native,
            "structure_preserved": self.structure_preserved,
            "source_kind": self.source_kind,
            "supports_queries": self.supports_queries,
            "supports_footprints": self.supports_footprints,
            "packed_native": self.packed_native,
            "bound": self.bound,
            "request": None if self.request is None else self.request.to_dict(),
            "executor": self.executor,
            "kernel": self.kernel,
            "program": None if self.program is None else self.program.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _program_for(
    carrier: OceanCarrier,
    route: str,
    *,
    need_gradients: bool = True,
    query: bool = False,
    footprint: bool = False,
) -> OceanProgram:
    instructions: list[OceanInstruction] = []
    source_view = carrier.source_view
    source_transfer = _carrier_source_transfer(carrier)

    if route != "analytic":
        instructions.append(
            OceanInstruction(
                "AddressSupport",
                inputs=("carrier",),
                outputs=("indices", "packed_coefficients"),
                attributes={"source_kind": carrier.source_kind, "lazy": True},
                effects=("read_structured_support",),
            )
        )
    source_value = "packed_coefficients"
    if source_transfer and route != "analytic":
        instructions.append(
            OceanInstruction(
                "SourceTransfer",
                inputs=(source_value,),
                outputs=("source_filtered_coefficients",),
                attributes={
                    "domain": "frequency",
                    "operations": source_transfer,
                    "chart": getattr(getattr(source_view, "chart", None), "frequency_convention", None),
                    "preserve_error_budget": True,
                },
                effects=("frequency_transfer",),
            )
        )
        source_value = "source_filtered_coefficients"
    if carrier.modes:
        instructions.append(
            OceanInstruction(
                "PhaseCompose",
                inputs=("wide_phase", source_value),
                outputs=("phase_coefficients",),
                attributes={"emit_bits": 16, "master_bits": "per_mode"},
                effects=("modular_phase_update", "round_at_emission"),
            )
        )
        source_value = "phase_coefficients"
    if footprint and route != "analytic":
        instructions.append(
            OceanInstruction(
                "FootprintTransfer",
                inputs=(source_value, "footprint"),
                outputs=("filtered_packed_coefficients",),
                attributes={"domain": "frequency", "preserve_error_budget": True},
                effects=("frequency_transfer",),
            )
        )
        source_value = "filtered_packed_coefficients"
    if need_gradients and route != "analytic":
        instructions.append(
            OceanInstruction(
                "DerivativeLogMul",
                inputs=(source_value, "derivative_operator"),
                outputs=("derivative_terms",),
                attributes={"quarter_turn": 16384, "zero_frequency": "zero_sentinel"},
                effects=("log_shift", "exact_quarter_turn", "shared_coefficient_emission"),
            )
        )
    if route == "sparse":
        if (
            carrier.metadata.get("closed_form") == "constant_half"
            and not _carrier_source_transfer(carrier)
            and not footprint
        ):
            inputs = (source_value, "derivative_terms") if need_gradients else (source_value,)
            instructions.append(
                OceanInstruction(
                    "ClosedFormDirichlet",
                    inputs=inputs,
                    outputs=("fields",),
                    attributes={"query": query, "value": carrier.metadata.get("closed_form_value")},
                    effects=("direct_dirichlet_identity", "stable_sinc_limits", "real_output"),
                )
            )
        elif carrier.metadata.get("support_pattern") == "diagonal" and not source_transfer and not footprint:
            inputs = (source_value, "derivative_terms") if need_gradients else (source_value,)
            instructions.append(
                OceanInstruction(
                    "DiagonalCollapsedSynthesis",
                    inputs=inputs,
                    outputs=("fields",),
                    attributes={"query": query, "collapsed_axis": "y_plus_x"},
                    effects=("one_dimensional_support", "pair_reduce", "shared_phase_basis", "real_output"),
                )
            )
        else:
            synthesis_inputs = (source_value, "derivative_terms") if need_gradients else (source_value,)
            instructions.extend(
                (
                    OceanInstruction(
                        "HermitianSynthesis",
                        inputs=synthesis_inputs,
                        outputs=("signed_terms",),
                        attributes={"consumer": "signed_real", "query": query},
                        effects=("pair_reduce", "real_output"),
                    ),
                    OceanInstruction(
                        "ReduceSigned",
                        inputs=("signed_terms",),
                        outputs=("fields",),
                        attributes={"joint": True, "query": query},
                        effects=("shared_phase_basis",),
                    ),
                )
            )
    elif route == "analytic":
        analytic_inputs = ("analytic_carrier", "footprint") if footprint else ("analytic_carrier",)
        instructions.append(
            OceanInstruction(
                "AnalyticAtomJet",
                inputs=analytic_inputs,
                outputs=("fields",),
                attributes={
                    "query": query,
                    "outputs": ("height", "grad_x", "grad_y") if need_gradients else ("height",),
                },
                effects=("analytic_gradient", "preserve_residuals"),
            )
        )
    else:
        synthesis_inputs = (source_value, "derivative_terms") if need_gradients else (source_value,)
        instructions.append(
            OceanInstruction(
                "HermitianSynthesis",
                inputs=synthesis_inputs,
                outputs=("fields",),
                attributes={"consumer": "packed_fb3" if route == "fb3" else "scipy_irfftn", "query": query},
                effects=("pair_reduce", "real_output", "packed_boundary" if route == "fb3" else "dense_fft"),
            )
        )
    instructions.append(
        OceanInstruction(
            "Materialize",
            inputs=("fields",),
            outputs=("output",),
            attributes={"query": query},
            effects=("output_dtype_boundary",),
        )
    )
    return OceanProgram(
        tuple(instructions),
        input_carrier=carrier.source_kind,
        output_fields=("height", "grad_x", "grad_y") if need_gradients else ("height",),
        effects=("residuals_retained", "masks_retained", "outliers_retained", "error_budget_retained"),
        input_types=MappingProxyType(
            {
                **({"analytic_carrier": "AnalyticCarrier"} if route == "analytic" else {"carrier": "Carrier"}),
                **({"wide_phase": "WidePhaseState"} if carrier.modes else {}),
                **({"footprint": "Footprint"} if footprint else {}),
                **({"derivative_operator": "DerivativeOperator"} if need_gradients and route != "analytic" else {}),
            }
        ),
        source_metadata=MappingProxyType(
            {
                **dict(carrier.metadata),
                **(
                    {
                        "source_transfer": source_transfer,
                        "source_transfer_domain": "frequency",
                    }
                    if source_transfer
                    else {}
                ),
            }
        ),
    )


def _carrier_source_transfer(carrier: OceanCarrier) -> tuple[str, ...]:
    """Return the normalized frequency operations retained by a block view."""

    view = carrier.source_view
    if view is None or not bool(getattr(view, "frequency_domain", False)):
        return ()
    return filtered_view_operations(view)


def _compact_support(carrier: OceanCarrier) -> bool:
    """Whether an addressed source is small enough for the point consumer."""

    count = int(carrier.active_count)
    return bool(
        count > 0
        and count <= min(4096, max(1, carrier.potential_elements // 8))
        and carrier.source_kind in {"sparse_modes", "block_view", "half_spectrum"}
    )


def _bound_executor_name(
    carrier: OceanCarrier,
    route: str,
    request: OceanOutputRequest,
    *,
    native: bool,
    native_representation: str | None,
) -> tuple[str, str]:
    """Name the concrete callable and kernel for one request."""

    sampling = "points" if request.query else "grid"
    if native:
        if route == "analytic":
            return "native_analytic_points", "spf.renderer.ocean_terminal.PreparedOcean._analytic_native_points"
        representation = str(native_representation or ("spf32" if carrier.metadata.get("packed_storage") else "fp32"))
        representation = representation.lower().replace("-", "_")
        if representation not in {"spf32", "fp32"}:
            representation = "spf32" if carrier.metadata.get("packed_storage") else "fp32"
        backend_name = str(request.backend).lower().replace("-", "_")
        if backend_name == "cuda":
            return (
                f"cuda_{representation}_{sampling}",
                f"spf.renderer._ocean_cuda.CUDAOceanSampler.{sampling}",
            )
        return f"native_{representation}_{sampling}", f"spf.renderer._ocean_kernel.NativeOceanSampler.{sampling}"
    if route == "analytic":
        return f"analytic_atom_reference_{sampling}", "spf.renderer.ocean_terminal.PreparedOcean._analytic_joint"
    if route == "sparse":
        if (
            carrier.metadata.get("closed_form") == "constant_half"
            and not _carrier_source_transfer(carrier)
            and request.footprint is None
        ):
            return (
                f"closed_form_dirichlet_reference_{sampling}",
                "spf.renderer.ocean_terminal.PreparedOcean._sparse_joint",
            )
        if (
            carrier.metadata.get("support_pattern") == "diagonal"
            and not _carrier_source_transfer(carrier)
            and request.footprint is None
        ):
            return f"diagonal_collapsed_reference_{sampling}", "spf.renderer.ocean_terminal.PreparedOcean._sparse_joint"
        return f"sparse_reference_{sampling}", "spf.renderer.ocean_terminal.PreparedOcean._sparse_joint"
    if route == "fb3":
        return "continuity_packed_ifft2_reference_grid", "spf.renderer.ocean_terminal.PreparedOcean._packed_fb3"
    return f"scipy_irfftn_reference_{sampling}", "spf.renderer.ocean_terminal.PreparedOcean._dense_joint"


def bind_ocean_execution(
    carrier: OceanCarrier,
    candidate: OceanExecutionPlan,
    request: OceanOutputRequest | Mapping[str, Any],
    *,
    native_supported: bool = False,
    native_available: bool = False,
    backend_allows_native: bool | None = None,
    use_native: bool | None = None,
    native_representation: str | None = None,
) -> OceanExecutionPlan:
    """Bind one complete request to the callable the terminal will invoke."""

    if not isinstance(candidate, OceanExecutionPlan):
        raise TypeError("ocean execution binding requires an OceanExecutionPlan candidate")
    if not isinstance(request, OceanOutputRequest):
        request = OceanOutputRequest(**dict(request))
    if tuple(carrier.shape) != tuple(candidate.shape):
        raise ValueError("ocean execution candidate shape does not match its carrier")
    preferred_native = request.native if use_native is None else use_native
    if preferred_native not in (None, True, False):
        raise TypeError("ocean native preference must be True, False, or None")
    if backend_allows_native is None:
        backend_allows_native = request.backend in {"cpu", "numpy"}
    transfer_operations = _carrier_source_transfer(carrier)

    route = candidate.method
    selection_reason = "candidate route satisfies the output request"
    # A point request can use the addressed source even when the source-only
    # estimate preferred an FFT for a full raster.  Keep explicit irfft/fb3
    # requests explicit; only automatic planning is allowed to switch.
    if (
        request.query
        and candidate.requested_method == "auto"
        and route in {"irfft", "fb3"}
        and _compact_support(carrier)
    ):
        route = "sparse"
        selection_reason = "automatic point request selected compact addressed support"
    if request.query and route == "fb3":
        raise ValueError("packed FB3 ocean queries require a display-side packed query consumer")
    if route == "analytic" and not request.query:
        selection_reason = "analytic grid request uses the canonical tiled consumer"
    if route not in {"sparse", "irfft", "fb3", "analytic"}:
        raise ValueError(f"unsupported bound ocean route {route!r}")

    native_eligible = bool(
        preferred_native is not False
        and native_supported
        and native_available
        and backend_allows_native
        and request.output_dtype == "float32"
        and ((route == "analytic" and request.query) or route == "sparse")
    )
    if preferred_native is True and not native_eligible:
        reason = (
            "retained source transfer is unavailable to the native SPF32 boundary"
            if transfer_operations and not native_supported
            else "native ocean consumer is unavailable for the selected route, backend, footprint, or dtype"
        )
        raise ValueError(reason)
    native = native_eligible
    executor, kernel = _bound_executor_name(
        carrier,
        route,
        request,
        native=native,
        native_representation=native_representation,
    )
    if native:
        representation = "analytic" if route == "analytic" else str(native_representation or "fp32").lower()
        selected_profile = f"{representation}_native_fused"
    else:
        selected_profile = candidate.profile
    diagnostics = dict(candidate.diagnostics)
    diagnostics.update(
        {
            "bound_request": request.to_dict(),
            "bound_executor": executor,
            "selected_kernel": kernel,
            "program_role": "source_recipe",
            "selection_reason": selection_reason,
            "source_transfer_operations": transfer_operations,
        }
    )
    fallback = candidate.fallback
    if route != candidate.method and fallback is None:
        fallback = "compact_point_consumer"
    return replace(
        candidate,
        method=route,
        route=route,
        reason=f"{candidate.reason}; {selection_reason}",
        fallback=fallback,
        backend=request.backend,
        profile=selected_profile,
        native=native,
        supports_queries=route in {"sparse", "analytic"},
        supports_footprints=True,
        packed_native=bool(candidate.packed_native or native),
        # ``program`` is the source recipe.  The bound executor/kernel below
        # is the only execution identity; no second descriptive schedule is
        # manufactured for the request.
        program=candidate.program,
        diagnostics=MappingProxyType(diagnostics),
        request=request,
        executor=executor,
        kernel=kernel,
    )


def plan_ocean_carrier(
    carrier: OceanCarrier,
    *,
    method: Any = "auto",
    backend: Any = "reference",
    allow_dense: bool = True,
    profile: str | None = None,
    need_gradients: bool = True,
    query: bool = False,
    footprint: bool = False,
    request: OceanOutputRequest | Mapping[str, Any] | None = None,
) -> OceanExecutionPlan:
    """Plan an already-owned carrier without reading its coefficient stream."""

    # Preserve the caller's route request alongside the selected route.
    # Compatibility aliases remain useful, while display/query consumers
    # must distinguish an explicit request from an automatic selection.
    requested_method = _normal_method(method)
    requested = requested_method
    aliases = {"dense": "irfft", "fft": "irfft", "dense_irfft": "irfft", "packed": "fb3", "v02": "fb3"}
    requested = aliases.get(requested, requested)
    if requested not in {"auto", "sparse", "irfft", "analytic", "fb3"}:
        raise ValueError(f"unsupported ocean synthesis method {method!r}")
    backend_name = str(getattr(backend, "name", backend)).lower().replace("-", "_")
    n = carrier.potential_elements
    modes = int(carrier.active_count)
    # A joint terminal evaluates all requested observables in one source walk.
    # The factor is intentionally part of both candidates so legacy cost
    # comparisons retain their ordering while the plan reports the actual
    # terminal shape.
    observable_factor = 1.0 if not need_gradients else 1.15
    # Keep the reference route's conservative per-mode overhead explicit in
    # the heuristic.  Comparing only algebraic K versus log2(N) counts can
    # select sparse K=8 on million-pixel frames even when the dense route is
    # the better reference choice.  This is a planning heuristic, not a
    # claim about measured runtime; native profiles may supply their own
    # multiplier when they are selected explicitly.
    profile_name = str(profile or "").lower()
    sparse_multiplier = 1.0 if "native" in profile_name else 3.0
    sparse_cost = float(max(1, n) * max(0, modes) * observable_factor * sparse_multiplier)
    irfft_cost = float(max(1, n) * max(1.0, math.log2(max(2, n))) * observable_factor)
    closed_form = carrier.metadata.get("closed_form")
    source_transfer = bool(_carrier_source_transfer(carrier))
    # A retained frequency transfer can preserve addressed diagonal support
    # while changing each coefficient's magnitude/phase.  The collapsed
    # character shortcut bypasses that shared transfer boundary, so only use
    # its structural label when no source transfer remains.
    diagonal = carrier.metadata.get("support_pattern") == "diagonal" and not source_transfer
    if closed_form == "constant_half" and not source_transfer:
        sparse_cost = 1.0
    if diagonal and len(carrier.shape) == 2 and carrier.shape[0] == carrier.shape[1]:
        # A diagonal Fourier support is a one-dimensional character along
        # y+x; charge its support walk, not a hypothetical 2-D coefficient
        # plane.  The terminal still emits the requested 2-D output/query.
        sparse_cost = float(max(1, modes) * observable_factor * max(1, carrier.shape[0]))
    fallback: str | None = None
    if requested == "analytic":
        route = "analytic"
        reason = "explicit analytic atom/jet request"
    elif requested == "sparse":
        route = "sparse"
        reason = "explicit sparse request"
    elif requested == "irfft":
        route = "irfft" if allow_dense else "sparse"
        reason = "explicit dense request" if allow_dense else "dense path disabled; sparse fallback"
        fallback = None if allow_dense else "sparse"
    elif requested == "fb3":
        route = "fb3"
        reason = "explicit packed FB3 request"
    elif requested == "auto" and carrier.source_kind == "analytic":
        # Analytic atom/jet carriers already expose the closed evaluator and
        # its residual/error ledger.  Auto must keep that consumer instead of
        # attempting an unowned sparse or Cartesian realization.
        route = "analytic"
        reason = "analytic carrier has a direct atom/jet consumer"
    elif allow_dense and irfft_cost < sparse_cost:
        route = "irfft"
        reason = "dense irfft has lower estimated joint cost"
    else:
        route = "sparse"
        reason = "sparse signed-real path has lower estimated joint cost"
    if closed_form == "constant_half" and not source_transfer and requested == "auto":
        route = "sparse"
        reason = "constant Hermitian half-plane has a direct impulse consumer"
    if diagonal and requested == "auto" and route == "sparse":
        reason = "diagonal Fourier support collapses to a one-dimensional character"
    if route == "fb3" and (len(carrier.shape) != 2 or any(n & (n - 1) for n in carrier.shape)):
        raise ValueError("packed FB3 ocean route requires a two-dimensional power-of-two shape")
    if route == "analytic" and carrier.source_kind != "analytic":
        raise ValueError("analytic ocean route requires an explicit analytic carrier")
    selected_profile = (
        profile
        or {
            "sparse": "sparse_packed_reference"
            if carrier.source_kind == "block_view"
            else "sparse_signed_real_reference",
            "irfft": "cartesian_reference",
            "fb3": "packed_spf32_fb3_reference",
            "analytic": "analytic_atom_reference",
        }[route]
    )
    if closed_form == "constant_half" and not source_transfer and profile is None:
        selected_profile = "closed_form_dirichlet_reference"
    if diagonal and profile is None and route == "sparse":
        selected_profile = "diagonal_collapsed_reference"
    packed_native = bool(
        route in {"sparse", "fb3"}
        and carrier.metadata.get("packed_storage", False)
        and (carrier.source_kind != "block_view" or int(carrier.metadata.get("scalar_bits", 32)) == 32)
    )
    program = _program_for(carrier, route, need_gradients=need_gradients, query=query, footprint=footprint)
    diagnostics = {
        "joint_terminal": True,
        "planning_materialized_coefficients": 0,
        "planning_materialized_elements": 0,
        "carrier_active_count_hint": modes,
        "structure_route": "diagonal_collapsed"
        if diagonal
        else "constant_dirichlet"
        if closed_form == "constant_half" and not source_transfer
        else "generic",
        "source_metadata": dict(carrier.metadata),
        "profile": selected_profile,
        "requested_method": requested_method,
        "selected_method": route,
        "selected_source": {
            "source_kind": carrier.source_kind,
            "representation": "packed_spf32" if packed_native else "cartesian_or_structured_reference",
            "packed_native": packed_native,
            "route": route,
        },
        "cost_model": {
            "kind": "operation_count_heuristic_not_measured_runtime",
            "sparse_multiplier": sparse_multiplier,
            "observable_factor": observable_factor,
            "dense_transform": "N*log2(N)",
            "backend": backend_name,
            "profile_hint": profile,
        },
    }
    candidate = OceanExecutionPlan(
        carrier.shape,
        route,
        modes,
        n,
        sparse_cost,
        irfft_cost,
        reason,
        fallback,
        route=route,
        backend=backend_name,
        profile=selected_profile,
        native=False,
        structure_preserved=bool(
            carrier.source_kind in {"block_view", "analytic"} or closed_form == "constant_half" or diagonal
        ),
        source_kind=carrier.source_kind,
        supports_queries=True,
        supports_footprints=True,
        packed_native=packed_native,
        program=program,
        diagnostics=MappingProxyType(diagnostics),
    )
    if request is None:
        return candidate
    return bind_ocean_execution(carrier, candidate, request)


def plan_ocean(
    source: Any = None,
    *,
    shape: Any | None = None,
    coefficients: Any = None,
    indices: Any = None,
    phases: Any = None,
    half_spectrum: Any = None,
    method: Any = "auto",
    backend: Any = "reference",
    allow_dense: bool = True,
    **kwargs: Any,
) -> OceanExecutionPlan:
    """Convenience planner for sparse, half-spectrum, or addressed sources."""

    if half_spectrum is not None:
        source = half_spectrum
    if shape is None:
        view = None
        if source is not None:
            from spf.renderer.blocks import BlockView
            from spf.renderer.carriers import as_block_view

            view = source if isinstance(source, BlockView) else as_block_view(source)
        if view is not None:
            candidate_shape = tuple(int(x) for x in view.shape)
            shape = candidate_shape
        elif source is not None and hasattr(source, "shape"):
            candidate_shape = tuple(int(x) for x in source.shape)
            shape = (
                (*candidate_shape[:-1], max(1, 2 * (candidate_shape[-1] - 1)))
                if isinstance(source, np.ndarray) and candidate_shape and candidate_shape[-1] != 1
                else candidate_shape
            )
        elif indices is not None:
            raise ValueError("plan_ocean requires shape for sparse coefficients")
        else:
            raise ValueError("plan_ocean requires shape for an unshaped source")
    carrier = OceanCarrier.from_inputs(
        shape,
        half_spectrum=source if coefficients is None and indices is None else None,
        coefficients=coefficients,
        indices=indices,
        phases=phases,
    )
    return plan_ocean_carrier(carrier, method=method, backend=backend, allow_dense=allow_dense, **kwargs)


__all__ = [
    "OceanExecutionPlan",
    "OceanInstruction",
    "OceanOutputRequest",
    "OceanProgram",
    "bind_ocean_execution",
    "plan_ocean",
    "plan_ocean_carrier",
]
