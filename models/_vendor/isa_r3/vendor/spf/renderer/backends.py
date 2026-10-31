"""Portable executable backends for the SFR v0.2 field machine.

This module is deliberately explicit about the boundary between an executable
portable implementation and a native target that is not present.  The
reference backend is the semantic oracle, the CPU backend adds NumPy-backed
pointwise execution and the SPU backend is a functional Appendix-C simulator
with an integer cycle model.  The CUDA adapter is intentionally narrow: when
the optional CUDA runtime is present it owns compact packed-SPF32 ocean
synthesis, while generic scalar/FFT/display operations remain unsupported.
Metal, Vulkan, FPGA and physical-SPU adapters are capability objects which
fail closed; they never route work to the CPU merely because a caller
requested a different backend.
"""

from __future__ import annotations

import math
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.carriers import decode_words, encode_words
from spf.renderer.cost import (
    CostModel,
    PlanCost,
    cpu_cost_model,
    gpu_cost_model,
    reference_cost_model,
    spu_cost_model,
)
from spf.renderer.model import Field, MaterializationRequest, RepresentationKind
from spf.renderer.report import RenderMetrics, RenderReport
from spf.renderer.v02 import FieldIR, RenderPlan
from spf.renderer.v02 import execute_op as reference_execute_op
from spf.tensor import v02 as arithmetic
from spf.tensor.op_registry import OP_REGISTRY


def _normal_name(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower().replace("-", "_")


def _jsonable(value: Any) -> Any:
    """Convert an execution value to deterministic, JSON-safe data."""

    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
    if isinstance(value, (np.integer, np.floating)):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, arithmetic.LogPos):
        return {"type": "LogPos", "L": int(value.L)}
    if isinstance(value, arithmetic.LogSigned):
        return {"type": "LogSigned", "L": int(value.L), "sign": int(value.sign)}
    if isinstance(value, arithmetic.Phase):
        return {"type": "Phase", "T": int(value.T)}
    if isinstance(value, arithmetic.SPF32Value):
        return {"type": "SPF32Value", "L": int(value.L), "T": int(value.T)}
    if isinstance(value, arithmetic.Transport):
        return {"type": "Transport", "T": _jsonable(value.T), "C": _jsonable(value.C)}
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return repr(value)


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend has no executable runtime."""

    def __init__(
        self,
        message: str,
        *,
        backend: str | None = None,
        operation: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.backend = backend
        self.operation = operation
        self.reason = reason


_ALL_REPRESENTATIONS = frozenset(RepresentationKind)


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Immutable capability declaration for one SFR backend.

    ``supported_operations`` is a semantic capability set, not a promise that
    every operation is a vendor-native kernel.  The capability table exposed by
    :class:`SFRBackend` records that distinction through ``native`` and the
    backend status fields.  An unavailable backend has an empty capability set
    and a non-empty reason, so a request cannot accidentally fall through.
    """

    name: str
    available: bool
    native: bool
    deterministic: bool
    supported_operations: frozenset[str]
    supported_representations: frozenset[RepresentationKind]
    supports_float64_reference: bool = False
    supports_integer_only: bool = False
    supports_parallel_transport: bool = True
    max_dimension: int = 3
    reason_unavailable: str | None = None
    backend_version: str = "sfr-backend-0.2"
    isa_version: str = "sfr-isa-v0.2"
    status: str = "available"
    execution_class: str = "reference_python"
    unsupported_reasons: Mapping[str, str] = field(default_factory=dict)
    cost_source: str = "modeled"

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower().replace("-", "_")
        operations = frozenset(str(value) for value in self.supported_operations)
        representations: set[RepresentationKind] = set()
        for value in self.supported_representations:
            if isinstance(value, RepresentationKind):
                representations.add(value)
            else:
                representations.add(RepresentationKind(str(value).upper().replace("-", "_")))
        dimension = int(self.max_dimension)
        if dimension < 0:
            raise ValueError("backend max_dimension must be non-negative")
        available = bool(self.available)
        status = str(self.status).strip().lower().replace("-", "_")
        if available and status in {"", "unavailable", "missing"}:
            status = "available"
        if not available and status in {"", "available"}:
            status = "unavailable"
        reason = None if self.reason_unavailable is None else str(self.reason_unavailable).strip()
        if not available and not reason:
            reason = "no executable runtime is attached"
        execution_class = str(self.execution_class).strip().lower().replace("-", "_")
        if not execution_class:
            raise ValueError("backend execution_class must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "native", bool(self.native))
        object.__setattr__(self, "deterministic", bool(self.deterministic))
        object.__setattr__(self, "supported_operations", operations)
        object.__setattr__(self, "supported_representations", frozenset(representations))
        object.__setattr__(self, "max_dimension", dimension)
        object.__setattr__(self, "reason_unavailable", reason)
        object.__setattr__(self, "backend_version", str(self.backend_version))
        object.__setattr__(self, "isa_version", str(self.isa_version))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "execution_class", execution_class)
        object.__setattr__(
            self,
            "unsupported_reasons",
            MappingProxyType({str(key): str(value) for key, value in self.unsupported_reasons.items()}),
        )
        object.__setattr__(self, "cost_source", str(self.cost_source))

    def supports_operation(self, operation: str) -> bool:
        candidate = str(getattr(operation, "value", operation))
        if "*" in self.supported_operations or candidate in self.supported_operations:
            return True
        folded = candidate.lower()
        return any(item.lower() == folded for item in self.supported_operations)

    def supports_representation(self, representation: RepresentationKind | str) -> bool:
        if isinstance(representation, RepresentationKind):
            kind = representation
        else:
            raw = str(representation).strip().upper().replace("-", "_")
            try:
                kind = RepresentationKind(raw)
            except ValueError:
                kind = RepresentationKind[raw]
        return kind in self.supported_representations


@dataclass(frozen=True, slots=True)
class CapabilityRecord:
    """One serializable row in a backend capability table."""

    backend: str
    operation: str
    supported: bool
    available: bool
    native: bool
    deterministic: bool
    status: str
    reason: str | None = None
    backend_version: str = "sfr-backend-0.2"
    isa_version: str = "sfr-isa-v0.2"

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "operation": self.operation,
            "supported": bool(self.supported),
            "available": bool(self.available),
            "native": bool(self.native),
            "deterministic": bool(self.deterministic),
            "status": self.status,
            "reason": self.reason,
            "backend_version": self.backend_version,
            "isa_version": self.isa_version,
        }


@dataclass(frozen=True, slots=True)
class BackendCostRecord:
    """One modeled per-operation cost row.

    Costs are explicitly labeled as modeled.  Unavailable adapters carry
    ``None`` rather than a CPU estimate, which makes accidental fallback
    visible to callers and benchmark serializers.
    """

    backend: str
    operation: str
    cost: float | None
    source: str
    available: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "operation": self.operation,
            "cost": self.cost,
            "source": self.source,
            "available": bool(self.available),
            "reason": self.reason,
        }


_SURFACE_OPERATIONS = frozenset(
    {
        "logadd_pos",
        "logadd_signed",
        "logpos_mul",
        "logpos_div",
        "phase_add",
        "phase_sub",
        "phase_conj",
        "phase_compose",
        "spf_mul",
        "spf_div",
        "spf_conj",
        "spf_neg",
        "spf_abs2",
        "spf_pair_addsub",
        "atom_transform",
        "atom_bound",
        "atom_to_jet",
        "jet_materialize",
        "reduce_positive",
        "reduce_signed",
        "reduce_complex",
        "reduce_transport",
        "fft",
        "ifft",
        "materialize",
        "display",
        "execute_ir",
    }
)
_PORTABLE_OPERATIONS = frozenset(set(OP_REGISTRY) | set(_SURFACE_OPERATIONS))


class SFRBackend(ABC):
    """Unified backend interface for scalar, structured and display work."""

    capabilities: BackendCapabilities
    cost_model: CostModel

    def __init__(self, capabilities: BackendCapabilities, cost_model: CostModel) -> None:
        self.capabilities = capabilities
        self.cost_model = cost_model
        self.last_report: RenderReport | None = None

    @property
    def name(self) -> str:
        return self.capabilities.name

    def require_available(self, operation: str | None = None) -> None:
        if self.capabilities.available:
            return
        reason = self.capabilities.reason_unavailable or "no executable runtime is attached"
        suffix = f" for operation {operation!r}" if operation is not None else ""
        raise BackendUnavailableError(
            f"SFR backend {self.name!r}{suffix} is unavailable: {reason}",
            backend=self.name,
            operation=None if operation is None else str(operation),
            reason=reason,
        )

    def _require_operation(self, operation: str) -> None:
        self.require_available(operation)
        if not self.capabilities.supports_operation(operation):
            reason = self.capabilities.unsupported_reasons.get(str(operation))
            detail = f": {reason}" if reason else ""
            raise NotImplementedError(f"SFR backend {self.name!r} does not support {operation!r}{detail}")

    def capability_table(self) -> tuple[CapabilityRecord, ...]:
        operations = self.capabilities.supported_operations
        selected = _PORTABLE_OPERATIONS if "*" in operations else operations
        if not self.capabilities.available:
            selected = _PORTABLE_OPERATIONS
        return tuple(
            CapabilityRecord(
                backend=self.name,
                operation=operation,
                supported=self.capabilities.supports_operation(operation),
                available=self.capabilities.available,
                native=self.capabilities.native,
                deterministic=self.capabilities.deterministic,
                status=self.capabilities.status,
                reason=(None if self.capabilities.available else self.capabilities.reason_unavailable),
                backend_version=self.capabilities.backend_version,
                isa_version=self.capabilities.isa_version,
            )
            for operation in sorted(selected)
        )

    def cost_table(self) -> tuple[BackendCostRecord, ...]:
        selected = (
            _PORTABLE_OPERATIONS
            if "*" in self.capabilities.supported_operations
            else self.capabilities.supported_operations
        )
        if not self.capabilities.available:
            selected = _PORTABLE_OPERATIONS
        rows: list[BackendCostRecord] = []
        for operation in sorted(selected):
            if not self.capabilities.available:
                rows.append(
                    BackendCostRecord(
                        self.name,
                        operation,
                        None,
                        "unavailable",
                        False,
                        self.capabilities.reason_unavailable,
                    )
                )
                continue
            try:
                cost = float(self.cost_model.operation_cost(operation))
            except (AttributeError, TypeError, ValueError):
                cost = None
            rows.append(BackendCostRecord(self.name, operation, cost, self.capabilities.cost_source, True))
        return tuple(rows)

    def estimate_cost(self, ir: FieldIR, *, element_count: int = 1, bytes_moved: int = 0) -> PlanCost:
        self._require_operation("execute_ir")
        return self.cost_model.estimate_ir(ir, element_count=element_count, bytes_moved=bytes_moved)

    # ---- scalar and packed arithmetic ---------------------------------

    def logadd_pos(self, left: Any, right: Any) -> Any:
        self._require_operation("logadd_pos")
        return arithmetic.logadd_pos(left, right)

    def logadd_signed(self, left: Any, right: Any) -> Any:
        self._require_operation("logadd_signed")
        return arithmetic.logadd_signed(left, right)

    def logpos_mul(self, left: Any, right: Any) -> Any:
        self._require_operation("logpos_mul")
        return arithmetic.logpos_mul(left, right)

    def logpos_div(self, left: Any, right: Any) -> Any:
        self._require_operation("logpos_div")
        return arithmetic.logpos_div(left, right)

    def phase_add(self, left: Any, right: Any) -> arithmetic.Phase:
        self._require_operation("phase_add")
        return arithmetic.phase_compose(left, right)

    def phase_compose(self, left: Any, right: Any) -> arithmetic.Phase:
        self._require_operation("phase_compose")
        return arithmetic.phase_compose(left, right)

    def phase_sub(self, left: Any, right: Any) -> arithmetic.Phase:
        self._require_operation("phase_sub")
        left_code = left.T if isinstance(left, arithmetic.Phase) else int(left)
        right_code = right.T if isinstance(right, arithmetic.Phase) else int(right)
        return arithmetic.Phase(left_code - right_code)

    def phase_conj(self, value: Any) -> arithmetic.Phase:
        self._require_operation("phase_conj")
        code = value.T if isinstance(value, arithmetic.Phase) else int(value)
        return arithmetic.Phase(-code)

    def spf_mul(self, left: Any, right: Any) -> Any:
        self._require_operation("spf_mul")
        return arithmetic.spf_mul(left, right)

    def spf_div(self, left: Any, right: Any) -> Any:
        self._require_operation("spf_div")
        return arithmetic.spf_div(left, right)

    def spf_conj(self, value: Any) -> Any:
        self._require_operation("spf_conj")
        return arithmetic.spf_conj(value)

    def spf_neg(self, value: Any) -> Any:
        self._require_operation("spf_neg")
        return arithmetic.spf_neg(value)

    def spf_abs2(self, value: Any) -> Any:
        self._require_operation("spf_abs2")
        return arithmetic.spf_abs2(value)

    def spf_pair_addsub(self, left: Any, right: Any, spec: Any = "balanced") -> Any:
        self._require_operation("spf_pair_addsub")
        return arithmetic.spf_pair_addsub_hybrid(left, right, precision=spec)

    # ---- atoms, jets and typed reductions -----------------------------

    def atom_transform(self, atoms: Any, transform: Any) -> Any:
        self._require_operation("atom_transform")
        matrix, shift = transform
        if isinstance(atoms, arithmetic.GaussianChirpAtom):
            return arithmetic.atom_affine(atoms, matrix, shift)
        return tuple(arithmetic.atom_affine(atom, matrix, shift) for atom in atoms)

    def atom_bound(self, atoms: Any, regions: Any) -> Any:
        self._require_operation("atom_bound")
        if isinstance(atoms, arithmetic.GaussianChirpAtom):
            return arithmetic.atom_bound(atoms, regions)
        return tuple(arithmetic.atom_bound(atom, regions) for atom in atoms)

    @staticmethod
    def _jet_center(regions: Any) -> Any:
        if hasattr(regions, "center"):
            return regions.center
        if isinstance(regions, Mapping) and "center" in regions:
            return regions["center"]
        values = np.asarray(regions, dtype=object)
        if values.shape == (2,):
            return values
        if values.shape == (2, 2):
            return (np.asarray(values[0], dtype=float) + np.asarray(values[1], dtype=float)) / 2.0
        raise ValueError("atom-to-jet regions must provide a two-dimensional center")

    def atom_to_jet(self, atoms: Any, regions: Any) -> Any:
        self._require_operation("atom_to_jet")
        center = self._jet_center(regions)
        if isinstance(atoms, arithmetic.GaussianChirpAtom):
            return arithmetic.atom_to_jet(atoms, center)
        return tuple(arithmetic.atom_to_jet(atom, center) for atom in atoms)

    @staticmethod
    def _jet_shape(regions: Any) -> tuple[int, int]:
        if hasattr(regions, "shape") and not isinstance(regions, tuple):
            shape = regions.shape
        elif isinstance(regions, Mapping):
            shape = regions.get("shape")
        else:
            shape = regions
        values = tuple(int(value) for value in np.asarray(shape).reshape(-1))
        if len(values) != 2:
            raise ValueError("jet materialization requires a two-dimensional shape")
        return (values[0], values[1])

    def jet_materialize(
        self,
        jets: Any,
        regions: Any,
        *,
        origin: Sequence[float] = (0.0, 0.0),
        output_format: Any = None,
    ) -> Any:
        self._require_operation("jet_materialize")
        shape = self._jet_shape(regions)
        if isinstance(regions, Mapping):
            origin = tuple(float(value) for value in regions.get("origin", origin))
        origin_values = tuple(float(value) for value in origin)
        if len(origin_values) != 2:
            raise ValueError("jet materialization origin must be two-dimensional")
        origin_pair = (origin_values[0], origin_values[1])
        if isinstance(jets, arithmetic.Jet2D):
            return arithmetic.jet_materialize(
                jets,
                shape=shape,
                origin=origin_pair,
                output_format=output_format,
            )
        return tuple(
            arithmetic.jet_materialize(
                jet,
                shape=shape,
                origin=origin_pair,
                output_format=output_format,
            )
            for jet in jets
        )

    def reduce_positive(self, values: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_positive")
        return arithmetic.reduce_logpos(values, spec=spec)

    def reduce_signed(self, values: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_signed")
        return arithmetic.reduce_logsigned(values, spec=spec)

    def reduce_complex(self, values: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_complex")
        return arithmetic.reduce_complex(values, spec=spec)

    def reduce_transport(self, segments: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_transport")
        return arithmetic.reduce_transport(segments, spec=spec)

    def fft(self, field: Any, spec: Any = None) -> Any:
        """Execute the public transform interface for dense or packed fields."""

        self._require_operation("fft")
        payload = field.representation.payload if isinstance(field, Field) else field
        values = np.asarray(payload)
        if values.dtype == np.uint32:
            return arithmetic.spf_fft_radix2(values, precision=spec or "balanced")
        return np.asarray(np.fft.fftn(values), dtype=np.complex128)

    def ifft(self, field: Any, spec: Any = None) -> Any:
        """Execute the inverse public transform interface."""

        self._require_operation("ifft")
        payload = field.representation.payload if isinstance(field, Field) else field
        values = np.asarray(payload)
        if values.dtype == np.uint32:
            return arithmetic.spf_ifft_radix2(values, precision=spec or "balanced")
        return np.asarray(np.fft.ifftn(values), dtype=np.complex128)

    # ---- IR and materialization ---------------------------------------

    def execute_operation(
        self,
        operation: str,
        operands: Sequence[Any] = (),
        params: Mapping[str, Any] | None = None,
    ) -> Any:
        """Execute one registered IR operation through this backend boundary.

        The current portable IR handlers share the tested v0.2 semantic
        implementation.  The call is explicit and capability-checked; there
        is no catch-and-retry path to another backend.
        """

        self._require_operation(operation)
        name = str(operation)
        direct: dict[str, Callable[..., Any]] = {
            "logadd_pos": self.logadd_pos,
            "v02_logadd_pos": self.logadd_pos,
            "logadd_signed": self.logadd_signed,
            "v02_logadd_signed": self.logadd_signed,
            "logpos_mul": self.logpos_mul,
            "v02_log_mul": self.logpos_mul,
            "logpos_div": self.logpos_div,
            "v02_log_div": self.logpos_div,
            "phase_add": self.phase_add,
            "v02_phase_add": self.phase_add,
            "v02_phase_compose": self.phase_compose,
            "phase_sub": self.phase_sub,
            "v02_phase_sub": self.phase_sub,
            "phase_conj": self.phase_conj,
            "v02_phase_conj": self.phase_conj,
            "spf_mul": self.spf_mul,
            "v02_spf_mul": self.spf_mul,
            "spf_div": self.spf_div,
            "v02_spf_div": self.spf_div,
            "spf_conj": self.spf_conj,
            "v02_spf_conj": self.spf_conj,
            "spf_neg": self.spf_neg,
            "v02_spf_neg": self.spf_neg,
            "spf_abs2": self.spf_abs2,
            "v02_abs2": self.spf_abs2,
            "spf_pair_addsub": self.spf_pair_addsub,
            "v02_spf_pair_addsub": self.spf_pair_addsub,
        }
        if name in direct:
            if name in {"spf_pair_addsub", "v02_spf_pair_addsub"}:
                return direct[name](*operands, params.get("precision", "balanced") if params else "balanced")
            return direct[name](*operands)
        return reference_execute_op(name, operands, params)

    # Alias used by callers that treat a backend as an operation executor.
    execute_op = execute_operation

    def materialize_field(self, field: Any, request: Any, **options: Any) -> Any:
        """Execute a compiler-selected representation on this backend.

        This is the representation-consumer boundary, distinct from executing
        algebraic SSA nodes. Portable backends share the typed interpreter;
        specialized consumers may override it without bypassing preparation.
        """
        from spf.renderer.runtime import materialize_field

        self.require_available()
        return materialize_field(field, request, backend=self, **options)

    @staticmethod
    def _value_size(value: Any) -> int:
        try:
            return max(1, int(np.asarray(value).size))
        except (TypeError, ValueError):
            return 1

    def execute_ir(
        self,
        ir: FieldIR | RenderPlan,
        values: Mapping[str, Any],
    ) -> tuple[dict[str, Any], RenderReport]:
        self._require_operation("execute_ir")
        graph = ir.ir if isinstance(ir, RenderPlan) else ir
        if not isinstance(graph, FieldIR):
            raise TypeError("execute_ir expects FieldIR or RenderPlan")
        missing = [name for name in graph.inputs if name not in values]
        if missing:
            raise KeyError(f"render plan inputs are missing: {', '.join(missing)}")
        unsupported = sorted({item.op for item in graph.nodes if not self.capabilities.supports_operation(item.op)})
        if unsupported:
            raise NotImplementedError(
                f"SFR backend {self.name!r} does not support operations: {', '.join(unsupported)}"
            )
        graph.validate_values(values)
        environment: dict[str, Any] = dict(values)
        lowered_steps = graph.to_execution_plan().steps
        for item, step in zip(graph.nodes, lowered_steps, strict=True):
            operands = tuple(environment[name] for name in item.inputs)
            # Preserve the same typed-edge proof carried by reference
            # ExecutionPlan lowering.  Raw uint32 words are only accepted as
            # SPF complex values when a preceding explicit cast established
            # that semantic contract.
            environment[item.output] = self.execute_operation(item.op, operands, step.params)
        element_count = max((self._value_size(value) for value in values.values()), default=1)
        cost = self.cost_model.estimate_ir(graph, element_count=element_count)
        metrics = RenderMetrics(
            backend=self.name,
            workload="field_ir",
            resolution=(element_count,),
            representation="typed_ir",
            field_operations=len(graph.nodes),
            reduction_operations=int(cost.reductions),
            potential_elements=element_count,
            materialized_elements=int(cost.materialization),
            bytes_read=int(cost.bytes_moved),
            peak_memory=int(element_count * 8),
        )
        report = RenderReport(
            metrics,
            capabilities_used=tuple(sorted({item.op for item in graph.nodes})),
            diagnostics={
                "execution": self.capabilities.execution_class,
                "hardware_native": self.capabilities.native,
                "cost_model": self.cost_model.name,
                "backend_status": self.capabilities.status,
            },
        )
        self.last_report = report
        return environment, report

    @staticmethod
    def _layout_for(
        field: Field,
        request: MaterializationRequest,
        payload: Any,
    ) -> tuple[tuple[int, ...], tuple[float, ...]]:
        del field
        if request.layout is not None:
            return request.layout.shape, request.layout.origin
        if isinstance(payload, np.ndarray) and payload.ndim:
            return tuple(int(value) for value in payload.shape), (0.0,) * payload.ndim
        raise ValueError("materialization requires MaterializationRequest.layout for a symbolic field")

    @staticmethod
    def _native_scalar(value: Any) -> Any:
        if isinstance(value, arithmetic.LogPos):
            return value.value
        if isinstance(value, arithmetic.LogSigned):
            return value.value
        if isinstance(value, arithmetic.SPF32Value):
            return complex(decode_words(value.word))
        if isinstance(value, arithmetic.Phase):
            return value.T
        if isinstance(value, np.uint32):
            return complex(decode_words(int(value)))
        return value

    @staticmethod
    def _complex_scalar(value: Any) -> complex:
        if isinstance(value, arithmetic.LogPos):
            return complex(value.value)
        if isinstance(value, arithmetic.LogSigned):
            return complex(value.value)
        if isinstance(value, arithmetic.SPF32Value):
            return complex(decode_words(value.word))
        return complex(value)

    @staticmethod
    def _map_values(values: Any, function: Callable[[Any], Any], *, dtype: Any = object) -> Any:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return function(array.item())
        result = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            result[index] = function(array[index])
        if dtype is not object:
            return np.asarray(result, dtype=dtype)
        return result

    def _format_materialized(self, values: Any, output_format: Any) -> Any:
        name = _normal_name(output_format or "native")
        raw = np.asarray(values)
        if name in {"native", "default", "complex", "complex128", "dense"}:
            if raw.dtype == np.uint32:
                return np.asarray(decode_words(raw), dtype=np.complex128)
            if raw.dtype == object:
                mapped = self._map_values(values, self._native_scalar)
                try:
                    return np.asarray(mapped, dtype=np.complex128)
                except (TypeError, ValueError):
                    return mapped
            return np.array(values, copy=True)
        if name in {"real", "float", "float64"}:
            return np.asarray(
                self._map_values(values, lambda item: float(self._complex_scalar(item).real)),
                dtype=np.float64,
            )
        if name in {"spf", "spf32", "spf_complex", "word", "words", "uint32"}:
            if raw.dtype == np.uint32:
                return np.asarray(raw, dtype=np.uint32).copy()
            mapped = self._map_values(values, self._complex_scalar)
            return np.asarray(
                encode_words(np.asarray(mapped, dtype=np.complex128)),
                dtype=np.uint32,
            )
        if name in {"log_pos", "logpos", "positive", "positive_real"}:

            def to_log_pos(item: Any) -> arithmetic.LogPos:
                if isinstance(item, arithmetic.LogPos):
                    return item
                if isinstance(item, arithmetic.LogSigned):
                    if item.value < 0.0:
                        raise ValueError("LOG_POS conversion of negative LOG_SIGNED value is not implicit")
                    return arithmetic.LogPos.from_value(item.value)
                if isinstance(item, (arithmetic.SPF32Value, np.uint32)):
                    decoded = self._complex_scalar(item)
                    if abs(decoded.imag) > 1e-12:
                        raise ValueError(
                            "LOG_POS conversion of complex values requires an explicit magnitude operation"
                        )
                    item = decoded.real
                number = float(item)
                if number < 0.0:
                    raise ValueError("LOG_POS conversion requires non-negative values")
                return arithmetic.LogPos.from_value(number)

            return self._map_values(values, to_log_pos)
        if name in {"log_signed", "logsigned", "signed", "signed_real"}:

            def to_log_signed(item: Any) -> arithmetic.LogSigned:
                if isinstance(item, arithmetic.LogSigned):
                    return item
                if isinstance(item, arithmetic.LogPos):
                    return arithmetic.LogSigned(item.L, 1)
                value = self._complex_scalar(item)
                if abs(value.imag) > 1e-12:
                    raise ValueError("LOG_SIGNED conversion requires real values")
                return arithmetic.LogSigned.from_value(value.real)

            return self._map_values(values, to_log_signed)
        if name in {"phase", "phase_only"}:

            def to_phase(item: Any) -> arithmetic.Phase:
                if isinstance(item, arithmetic.Phase):
                    return item
                value = self._complex_scalar(item)
                if value == 0.0:
                    return arithmetic.Phase(0)
                return arithmetic.Phase(
                    round(math.atan2(value.imag, value.real) * arithmetic.SPF32_PHASE_MOD / (2.0 * math.pi))
                )

            return self._map_values(values, to_phase)
        if name in {"display", "display_code", "uint8", "byte"}:

            def to_display(item: Any) -> int:
                if isinstance(item, arithmetic.LogPos):
                    return arithmetic.display_code_from_log(item)
                if isinstance(item, arithmetic.LogSigned):
                    magnitude = abs(item.value)
                else:
                    magnitude = abs(self._complex_scalar(item))
                return arithmetic.display_code_from_log(arithmetic.LogPos.from_value(float(magnitude)))

            return np.asarray(self._map_values(values, to_display, dtype=np.uint8), dtype=np.uint8)
        raise ValueError(f"unsupported SFR materialization output_format {output_format!r}")

    def _materialize_payload(
        self,
        payload: Any,
        shape: tuple[int, ...],
        origin: tuple[float, ...],
        output_format: Any,
    ) -> Any:
        if isinstance(payload, arithmetic.Jet2D):
            if len(shape) != 2:
                raise ValueError("Jet2D materialization requires a two-dimensional layout")
            jet_format = (
                output_format if _normal_name(output_format) in {"spf", "spf32", "word", "words", "uint32"} else None
            )
            values = arithmetic.jet_materialize(
                payload,
                shape=shape,
                origin=(origin[0], origin[1]),
                output_format=jet_format,
            )
            return values if jet_format is not None else self._format_materialized(values, output_format)
        if isinstance(payload, arithmetic.GaussianChirpAtom):
            if len(origin) != payload.dimension or len(shape) != payload.dimension:
                raise ValueError("atom dimension does not match materialization layout")
            axes = [origin[axis] + np.arange(shape[axis], dtype=float) for axis in range(len(shape))]
            mesh = np.meshgrid(*axes, indexing="ij")
            points = np.stack(mesh, axis=-1)
            values = payload.evaluate(points)
            return self._format_materialized(values, output_format)
        if isinstance(payload, arithmetic.AtomMixture):
            return payload.materialize({"shape": shape, "origin": origin}, output_format=output_format)
        if isinstance(payload, arithmetic.SeparableField):
            axes = [origin[axis] + np.arange(shape[axis], dtype=float) for axis in range(len(shape))]
            mesh = np.meshgrid(*axes, indexing="ij")
            points = np.stack(mesh, axis=-1)
            output = np.empty(shape, dtype=object)
            for index in np.ndindex(shape):
                result = payload.evaluate(points[index])
                output[index] = getattr(result, "value", result)
            return self._format_materialized(output, output_format)
        if isinstance(payload, arithmetic.SphericalGaussian):
            raise ValueError("spherical materialization requires explicit unit-direction points")
        array = np.asarray(payload)
        if array.ndim == 0:
            values = np.full(shape, array.item())
        elif int(array.size) == int(math.prod(shape)):
            values = np.asarray(array).reshape(shape)
        else:
            raise ValueError(f"dense payload with {array.size} elements cannot fill layout {shape}")
        return self._format_materialized(values, output_format)

    def materialize(self, field: Field, request: MaterializationRequest) -> Any:
        self._require_operation("materialize")
        if not isinstance(field, Field):
            raise TypeError("materialize expects a typed Field")
        if not isinstance(request, MaterializationRequest):
            raise TypeError("materialize expects a MaterializationRequest")
        # The runtime owns the one-boundary policy and complete dense suffix;
        # routing through it keeps backend.materialize and Field.materialize
        # semantically identical without silently changing the backend.
        from spf.renderer.runtime import materialize_field

        return materialize_field(field, request, backend=self)

    def display(self, field: Field, request: MaterializationRequest | None = None) -> Any:
        self._require_operation("display")
        if request is None:
            request = MaterializationRequest(backend=self.name, output_format="display")
        selected = MaterializationRequest(
            layout=request.layout,
            region=request.region,
            output_kind=request.output_kind,
            precision=request.precision,
            error_budget=request.error_budget,
            backend=self.name,
            output_format="display",
        )
        return self.materialize(field, selected)


class ReferenceBackend(SFRBackend):
    """Correctness-first semantic implementation."""

    def __init__(self, *, name: str = "reference") -> None:
        super().__init__(
            BackendCapabilities(
                name=name,
                available=True,
                native=False,
                deterministic=True,
                supported_operations=_PORTABLE_OPERATIONS,
                supported_representations=_ALL_REPRESENTATIONS,
                supports_float64_reference=True,
                cost_source="reference_model",
                execution_class="reference_python",
            ),
            reference_cost_model(),
        )


def _word_array(value: Any) -> np.ndarray | None:
    if isinstance(value, (arithmetic.SPF32Value, arithmetic.LogPos, arithmetic.LogSigned)):
        return None
    if not isinstance(value, (np.ndarray, list, tuple)):
        return None
    array = np.asarray(value)
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        return None
    return np.asarray(array, dtype=np.uint32)


def _word_components(value: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(value, dtype=np.uint32)
    logs = np.asarray(arithmetic.unpack_L(array), dtype=np.int64)
    phases = np.asarray(arithmetic.unpack_T(array), dtype=np.int64)
    nan = (logs == arithmetic.SPF32_NAN_L) & (phases == arithmetic.SPF32_NAN_T)
    reserved = logs == arithmetic.SPF32_NAN_L
    logs = np.where(reserved & ~nan, arithmetic.SPF32_L_MAX_FINITE, logs)
    zero = logs == arithmetic.SPF32_L_ZERO
    phases = np.where(zero, 0, phases & arithmetic.SPF32_PHASE_MASK)
    return logs, phases, zero, nan


def _pack_word_components(
    logs: Any,
    phases: Any,
    *,
    zero: Any = False,
    nan: Any = False,
) -> np.ndarray:
    log_array = np.asarray(logs, dtype=np.int64)
    phase_array = np.asarray(phases, dtype=np.int64)
    log_array, phase_array = np.broadcast_arrays(log_array, phase_array)
    log_array = np.clip(log_array, arithmetic.SPF32_L_MIN_FINITE, arithmetic.SPF32_L_MAX_FINITE)
    phase_array = phase_array & arithmetic.SPF32_PHASE_MASK
    zero_array = np.broadcast_to(np.asarray(zero, dtype=bool), log_array.shape)
    nan_array = np.broadcast_to(np.asarray(nan, dtype=bool), log_array.shape)
    packed = ((log_array & 0xFFFF) << 16) | phase_array
    packed = np.where(zero_array & ~nan_array, arithmetic.SPF32_WORD_ZERO, packed)
    packed = np.where(nan_array, arithmetic.SPF32_WORD_NAN, packed)
    return np.asarray(packed, dtype=np.uint32)


class CPUBackend(SFRBackend):
    """Portable CPU profile with vectorized packed-word pointwise kernels."""

    def __init__(self) -> None:
        super().__init__(
            BackendCapabilities(
                name="cpu",
                available=True,
                native=False,
                deterministic=True,
                supported_operations=_PORTABLE_OPERATIONS,
                supported_representations=_ALL_REPRESENTATIONS,
                supports_float64_reference=True,
                cost_source="cpu_model",
                execution_class="portable_numpy",
            ),
            cpu_cost_model(),
        )

    @staticmethod
    def _finish_word_result(result: np.ndarray, left: Any, right: Any | None = None) -> Any:
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            return result
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            return result
        if result.ndim == 0:
            return int(result.item())
        return result

    def _vector_binary_words(self, left: Any, right: Any, operation: str) -> Any:
        left_array = _word_array(left)
        right_array = _word_array(right)
        if left_array is None or right_array is None:
            return None
        left_array, right_array = np.broadcast_arrays(left_array, right_array)
        left_l, left_t, left_zero, left_nan = _word_components(left_array)
        right_l, right_t, right_zero, right_nan = _word_components(right_array)
        nan = left_nan | right_nan
        zero = left_zero | right_zero
        if operation == "mul":
            logs = left_l + right_l
            phases = left_t + right_t
        elif operation == "div":
            logs = left_l - right_l
            phases = left_t - right_t
            zero = left_zero
            # ``v02.spf_div`` (scalar and plane paths) tests the numerator first,
            # so ``0/0`` is the zero sentinel, not NaN; ``_pack_word_components``
            # lets NaN win, so the zero numerator has to be excluded here.
            nan = nan | (right_zero & ~left_zero)
        else:
            raise ValueError(f"unknown vector word operation {operation!r}")
        result = _pack_word_components(logs, phases, zero=zero, nan=nan)
        return self._finish_word_result(result, left, right)

    def _vector_unary_words(self, value: Any, operation: str) -> Any:
        array = _word_array(value)
        if array is None:
            return None
        logs, phases, zero, nan = _word_components(array)
        if operation == "conj":
            phases = -phases
        elif operation == "neg":
            phases = phases + arithmetic.SPF32_PHASE_MASK // 2 + 1
        else:
            raise ValueError(f"unknown vector unary word operation {operation!r}")
        result = _pack_word_components(logs, phases, zero=zero, nan=nan)
        if isinstance(value, (np.ndarray, list, tuple)):
            return result
        return int(result.item()) if result.ndim == 0 else result

    def spf_mul(self, left: Any, right: Any) -> Any:
        self._require_operation("spf_mul")
        result = self._vector_binary_words(left, right, "mul")
        return arithmetic.spf_mul(left, right) if result is None else result

    def spf_div(self, left: Any, right: Any) -> Any:
        self._require_operation("spf_div")
        result = self._vector_binary_words(left, right, "div")
        return arithmetic.spf_div(left, right) if result is None else result

    def spf_conj(self, value: Any) -> Any:
        self._require_operation("spf_conj")
        result = self._vector_unary_words(value, "conj")
        return arithmetic.spf_conj(value) if result is None else result

    def spf_neg(self, value: Any) -> Any:
        self._require_operation("spf_neg")
        result = self._vector_unary_words(value, "neg")
        return arithmetic.spf_neg(value) if result is None else result

    def spf_abs2(self, value: Any) -> Any:
        self._require_operation("spf_abs2")
        array = _word_array(value)
        if array is None:
            return arithmetic.spf_abs2(value)
        logs, _, zero, nan = _word_components(array)
        output = np.empty(logs.shape, dtype=object)
        for index in np.ndindex(logs.shape):
            if bool(nan[index]):
                output[index] = arithmetic.LogPos.nan()
            elif bool(zero[index]):
                output[index] = arithmetic.LogPos.zero()
            else:
                output[index] = arithmetic.LogPos(
                    int(
                        np.clip(
                            2 * logs[index],
                            arithmetic.SPF32_L_MIN_FINITE,
                            arithmetic.SPF32_L_MAX_FINITE,
                        )
                    )
                )
        return output if isinstance(value, (np.ndarray, list, tuple)) else output.item()

    @staticmethod
    def _typed_binary(left: Any, right: Any, function: Callable[[Any, Any], Any]) -> Any:
        if not isinstance(left, (np.ndarray, list, tuple)) and not isinstance(right, (np.ndarray, list, tuple)):
            return function(left, right)
        left_array, right_array = np.broadcast_arrays(np.asarray(left, dtype=object), np.asarray(right, dtype=object))
        output = np.empty(left_array.shape, dtype=object)
        for index in np.ndindex(output.shape):
            output[index] = function(left_array[index], right_array[index])
        return output

    def logadd_pos(self, left: Any, right: Any) -> Any:
        self._require_operation("logadd_pos")
        return self._typed_binary(left, right, arithmetic.logadd_pos)

    def logadd_signed(self, left: Any, right: Any) -> Any:
        self._require_operation("logadd_signed")
        return self._typed_binary(left, right, arithmetic.logadd_signed)

    def spf_pair_addsub(self, left: Any, right: Any, spec: Any = "balanced") -> Any:
        self._require_operation("spf_pair_addsub")
        left_array = _word_array(left)
        right_array = _word_array(right)
        if left_array is None or right_array is None:
            return arithmetic.spf_pair_addsub_hybrid(left, right, precision=spec)
        left_array, right_array = np.broadcast_arrays(left_array, right_array)
        plus = np.empty(left_array.shape, dtype=np.uint32)
        minus = np.empty(left_array.shape, dtype=np.uint32)
        for index in np.ndindex(left_array.shape):
            plus[index], minus[index] = arithmetic.spf_pair_addsub_hybrid(
                int(left_array[index]), int(right_array[index]), precision=spec
            )
        if isinstance(left, (np.ndarray, list, tuple)) or isinstance(right, (np.ndarray, list, tuple)):
            return plus, minus
        return int(plus.item()), int(minus.item())

    def fft(self, field: Any, spec: Any = None) -> Any:
        self._require_operation("fft")
        if isinstance(field, Field):
            field = field.representation.payload
        values = np.asarray(field)
        if values.dtype == np.uint32:
            return arithmetic.spf_fft_radix2(values, precision=spec or "balanced")
        return np.fft.fftn(values)

    def ifft(self, field: Any, spec: Any = None) -> Any:
        self._require_operation("ifft")
        if isinstance(field, Field):
            field = field.representation.payload
        values = np.asarray(field)
        if values.dtype == np.uint32:
            return arithmetic.spf_ifft_radix2(values, precision=spec or "balanced")
        return np.fft.ifftn(values)


class SPUOpcode(StrEnum):
    """Appendix-C functional SPU instruction set."""

    LADD_SAT = "LADD.SAT"
    LSUB_SAT = "LSUB.SAT"
    TADD_MOD = "TADD.MOD"
    TSUB_MOD = "TSUB.MOD"
    LOGADD_ROM = "LOGADD.ROM"
    LOGSUB_ROM = "LOGSUB.ROM"
    SPF_MUL = "SPF.MUL"
    SPF_CONJ = "SPF.CONJ"
    SPF_PAIR = "SPF.PAIR"
    JET_SET = "JET.SET"
    JET_STEP = "JET.STEP"
    JET_NEXTROW = "JET.NEXTROW"
    PIVOT_MAX = "PIVOT.MAX"
    PIVOT_ADDPOS = "PIVOT.ADDPOS"
    PIVOT_ADDSIGN = "PIVOT.ADDSIGN"
    PIVOT_ADDXY = "PIVOT.ADDXY"
    PIVOT_FINAL = "PIVOT.FINAL"
    TRN_COMPOSE = "TRN.COMPOSE"
    TROP_MIN = "TROP.MIN"
    TROP_MAX = "TROP.MAX"
    BOUND_TEST = "BOUND.TEST"


@dataclass(frozen=True, slots=True)
class SPUInstruction:
    opcode: SPUOpcode
    dst: str | tuple[str, ...]
    sources: tuple[str, ...] = ()
    immediate: Any = None

    def __post_init__(self) -> None:
        if isinstance(self.opcode, SPUOpcode):
            opcode = self.opcode
        else:
            raw = str(self.opcode).strip().upper()
            try:
                opcode = SPUOpcode(raw)
            except ValueError:
                opcode = SPUOpcode[raw.replace(".", "_")]
        if isinstance(self.dst, str):
            destinations: str | tuple[str, ...] = self.dst
        else:
            destinations = tuple(str(value) for value in self.dst)
            if not destinations:
                raise ValueError("SPU instruction requires at least one destination")
        source_values = (self.sources,) if isinstance(self.sources, str) else tuple(self.sources)
        source_names = tuple(str(value) for value in source_values)
        if any(not value for value in source_names):
            raise ValueError("SPU source register names must be non-empty")
        object.__setattr__(self, "opcode", opcode)
        object.__setattr__(self, "dst", destinations)
        object.__setattr__(self, "sources", source_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "opcode": self.opcode.value,
            "dst": list(self.dst) if isinstance(self.dst, tuple) else self.dst,
            "sources": list(self.sources),
            "immediate": _jsonable(self.immediate),
        }


@dataclass(frozen=True, slots=True)
class SPUTraceEvent:
    sequence: int
    opcode: str
    cycles: int
    dst: str | tuple[str, ...]
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "opcode": self.opcode,
            "cycles": int(self.cycles),
            "dst": list(self.dst) if isinstance(self.dst, tuple) else self.dst,
            "sources": list(self.sources),
        }


class SPUPivot(int):
    """An integer-compatible pivot accumulator carrying retained terms."""

    values: list[Any]
    pivot_L: int  # noqa: N815 - Appendix-C ISA spelling

    def __new__(cls, pivot_L: int, values: Sequence[Any] = ()) -> SPUPivot:
        instance = int.__new__(cls, int(pivot_L))
        instance.values = list(values)
        instance.pivot_L = int(pivot_L)
        return instance


@dataclass(slots=True)
class SPUState:
    registers: dict[str, Any] = field(default_factory=dict)
    instruction_counts: dict[str, int] = field(default_factory=dict)
    cycles: int = 0
    trace: list[SPUTraceEvent] = field(default_factory=list)

    @property
    def instruction_count(self) -> int:
        return int(sum(self.instruction_counts.values()))

    @property
    def cycle_count(self) -> int:
        return int(self.cycles)

    def record(self, opcode: SPUOpcode, cycles: int, instruction: SPUInstruction | None = None) -> None:
        amount = int(cycles)
        if amount < 0:
            raise ValueError("SPU instruction cycles must be non-negative")
        self.instruction_counts[opcode.value] = self.instruction_counts.get(opcode.value, 0) + 1
        self.cycles += amount
        if instruction is not None:
            self.trace.append(
                SPUTraceEvent(
                    sequence=self.instruction_count - 1,
                    opcode=opcode.value,
                    cycles=amount,
                    dst=instruction.dst,
                    sources=instruction.sources,
                )
            )


class SPUSimulatorBackend(SFRBackend):
    """Functional Appendix-C SPU simulator with deterministic accounting."""

    # MEASURED, not modeled: these are the execution cycles the Amaranth RTL in
    # ``hw/spu/`` spends per micro-op, reported by ``hw.spu.sim
    # .measure_microop_cycles()`` on 2026-09-18 and pinned by
    # ``tests/v02/test_spu_cycle_model.py``.  "Execution" excludes the §38.1
    # sequencer's 2-cycle fetch/decode, which is per-instruction overhead of a
    # non-pipelined issue stage, not a cost of the op (``SEQUENCER_OVERHEAD``).
    #
    # What changed against the pre-RTL estimates:
    #   JET.SET       3 -> 12   ten §13.6 seeds stream one per cycle out of the
    #                           §38.8 header SRAM, plus one row-start
    #   PIVOT.MAX     1 -> 2    one issue cycle + the max-L scan cycle
    #   PIVOT.ADDPOS  1 -> 4    §8.5 exp2 ROM read, align, accumulate
    #   PIVOT.ADDSIGN 1 -> 4    same
    #   PIVOT.ADDXY   2 -> 4    mantissa, Q62 sin/cos products, accumulate
    #   PIVOT.FINAL   5 -> 23   leading-one detect + a 9-step binary search over
    #                           the exact log2 threshold ROM (36 for the complex
    #                           algebra, which also runs the 30-iteration CORDIC
    #                           -- see ``_CYCLES_PIVOT_FINAL_COMPLEX``)
    #   TRN.COMPOSE   4 -> 2    the products are combinational; only the
    #                           per-channel LOGADD ROM read costs a cycle
    #   SPF.PAIR      8 -> 96   there is no SPF.PAIR datapath in the RTL: §7.4's
    #                           hybrid adder is a float64 local frame, so it
    #                           lowers to two 2-term complex pivots
    #                           (``hw.spu.isa.lower_spf_pair``), and 96 is the
    #                           measured cost of that sequence
    _CYCLES = MappingProxyType(
        {
            SPUOpcode.LADD_SAT: 1,
            SPUOpcode.LSUB_SAT: 1,
            SPUOpcode.TADD_MOD: 1,
            SPUOpcode.TSUB_MOD: 1,
            SPUOpcode.LOGADD_ROM: 2,
            SPUOpcode.LOGSUB_ROM: 2,
            SPUOpcode.SPF_MUL: 1,
            SPUOpcode.SPF_CONJ: 1,
            SPUOpcode.SPF_PAIR: 96,
            SPUOpcode.JET_SET: 12,
            SPUOpcode.JET_STEP: 1,
            SPUOpcode.JET_NEXTROW: 2,
            SPUOpcode.PIVOT_MAX: 2,
            SPUOpcode.PIVOT_ADDPOS: 4,
            SPUOpcode.PIVOT_ADDSIGN: 4,
            SPUOpcode.PIVOT_ADDXY: 4,
            SPUOpcode.PIVOT_FINAL: 23,
            SPUOpcode.TRN_COMPOSE: 2,
            SPUOpcode.TROP_MIN: 1,
            SPUOpcode.TROP_MAX: 1,
            SPUOpcode.BOUND_TEST: 1,
        }
    )

    #: ``PIVOT.FINAL`` with the §9.4 complex algebra also runs the atan2 CORDIC.
    _CYCLES_PIVOT_FINAL_COMPLEX = 36
    #: Fetch + decode cycles the non-pipelined §38.1 sequencer adds per op.
    _SEQUENCER_OVERHEAD = 2

    def __init__(self) -> None:
        operations = frozenset(set(_PORTABLE_OPERATIONS) | {opcode.value for opcode in SPUOpcode})
        super().__init__(
            BackendCapabilities(
                name="spu_sim",
                available=True,
                native=False,
                deterministic=True,
                supported_operations=operations,
                supported_representations=_ALL_REPRESENTATIONS,
                supports_integer_only=True,
                cost_source="spu_instruction_model",
                execution_class="simulator",
            ),
            spu_cost_model(),
        )
        self.last_state = SPUState()

    def _observe(self, opcode: SPUOpcode) -> None:
        state = SPUState()
        state.record(opcode, self._CYCLES[opcode])
        self.last_state = state

    def logadd_pos(self, left: Any, right: Any) -> Any:
        value = super().logadd_pos(left, right)
        self._observe(SPUOpcode.LOGADD_ROM)
        return value

    def logadd_signed(self, left: Any, right: Any) -> Any:
        value = super().logadd_signed(left, right)
        self._observe(SPUOpcode.LOGADD_ROM)
        return value

    def spf_mul(self, left: Any, right: Any) -> Any:
        value = super().spf_mul(left, right)
        self._observe(SPUOpcode.SPF_MUL)
        return value

    def spf_conj(self, value: Any) -> Any:
        result = super().spf_conj(value)
        self._observe(SPUOpcode.SPF_CONJ)
        return result

    @staticmethod
    def _sources(state: SPUState, instruction: SPUInstruction) -> list[Any]:
        values: list[Any] = []
        for name in instruction.sources:
            if name not in state.registers:
                raise KeyError(f"SPU source register {name!r} is not defined")
            values.append(state.registers[name])
        return values

    @classmethod
    def _operand(cls, state: SPUState, instruction: SPUInstruction, index: int) -> Any:
        values = cls._sources(state, instruction)
        if index < len(values):
            return values[index]
        if index == 1 and instruction.immediate is not None:
            return instruction.immediate
        raise ValueError(f"{instruction.opcode.value} requires source operand {index}")

    @staticmethod
    def _write(state: SPUState, destination: str | tuple[str, ...], value: Any) -> None:
        if isinstance(destination, tuple):
            if not isinstance(value, (tuple, list)):
                raise ValueError("SPU multi-destination instruction must return a tuple/list")
            values = tuple(value)
            if len(destination) != len(values):
                raise ValueError("SPU multi-destination result arity mismatch")
            for name, item in zip(destination, values, strict=True):
                state.registers[name] = item
            return
        state.registers[destination] = value

    @staticmethod
    def _log_value(value: Any) -> int:
        if isinstance(value, (arithmetic.LogPos, arithmetic.LogSigned)):
            return int(value.L)
        if isinstance(value, arithmetic.RawLogCode):
            return int(value.code)
        if isinstance(value, arithmetic.SPF32Value):
            return int(value.L)
        return int(value)

    @classmethod
    def _pivot_items(cls, value: Any) -> list[Any]:
        if isinstance(value, SPUPivot):
            return list(value.values)
        if isinstance(value, Mapping) and "values" in value:
            return list(value["values"])
        if isinstance(value, (list, tuple, np.ndarray)):
            return list(np.asarray(value, dtype=object).reshape(-1))
        return [value]

    @classmethod
    def _pivot_value(cls, value: Any, *, signed: bool = False) -> Any:
        if signed:
            if isinstance(value, arithmetic.LogSigned):
                return value
            if isinstance(value, arithmetic.LogPos):
                return arithmetic.LogSigned(value.L, 1)
            if isinstance(value, arithmetic.RawLogCode):
                return arithmetic.LogSigned(value.code, value.sign)
            if isinstance(value, (tuple, list)) and len(value) == 2:
                return arithmetic.LogSigned(int(value[1]), int(value[0]))
            return arithmetic.LogSigned(int(value), 1)
        if isinstance(value, arithmetic.LogPos):
            return value
        if isinstance(value, arithmetic.RawLogCode):
            return arithmetic.LogPos(value.code)
        if isinstance(value, arithmetic.LogSigned):
            if value.sign < 0:
                raise TypeError("PIVOT.ADDPOS cannot consume a negative signed term")
            return arithmetic.LogPos(value.L)
        return arithmetic.LogPos(int(value))

    @classmethod
    def _new_pivot(cls, previous: Any, term: Any | None = None, *, signed: bool = False) -> SPUPivot:
        values = cls._pivot_items(previous)
        if term is not None:
            values.append(cls._pivot_value(term, signed=signed))
        valid = [
            cls._log_value(item)
            for item in values
            if not getattr(item, "is_nan", False) and not getattr(item, "is_zero", False)
        ]
        pivot = max(valid, default=arithmetic.SPF32_L_ZERO)
        return SPUPivot(pivot, values)

    @classmethod
    def _jet_state(cls, jet: arithmetic.Jet2D) -> dict[str, Any]:
        value = arithmetic.jet_eval(jet, 0.0, 0.0)
        next_value = arithmetic.jet_eval(jet, 1.0, 0.0)
        row_value = arithmetic.jet_eval(jet, 0.0, 1.0)
        return {
            "jet": jet,
            "x": 0,
            "y": 0,
            "value": value,
            "d1": next_value - value,
            "d2": jet.hess_xx,
            "row_dy": row_value - value,
        }

    @classmethod
    def _unwrap_jet_state(cls, value: Any) -> dict[str, Any]:
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], dict):
            value = value[1]
        if not isinstance(value, Mapping) or "jet" not in value:
            raise TypeError("SPU jet instruction requires a jet state")
        return dict(value)

    @staticmethod
    def _tropical_key(value: Any) -> tuple[int, float, str]:
        if isinstance(value, (arithmetic.LogPos, arithmetic.LogSigned)):
            return (0, float(value.L), type(value).__name__)
        if isinstance(value, arithmetic.SPF32Value):
            return (1, float(value.L), "SPF32Value")
        try:
            return (2, float(value), type(value).__name__)
        except (TypeError, ValueError):
            return (3, 0.0, repr(value))

    @staticmethod
    def _bound_number(value: Any) -> float:
        if isinstance(value, (arithmetic.LogPos, arithmetic.LogSigned)):
            return abs(float(value.value))
        if isinstance(value, arithmetic.ReductionResult):
            return SPUSimulatorBackend._bound_number(value.value)
        return abs(float(value))

    def execute_instruction(self, instruction: SPUInstruction, state: SPUState) -> None:
        if not isinstance(instruction, SPUInstruction):
            raise TypeError("execute_instruction expects SPUInstruction")
        self.require_available(instruction.opcode.value)
        op = instruction.opcode
        operands = self._sources(state, instruction)
        left = operands[0] if operands else None
        right = operands[1] if len(operands) > 1 else instruction.immediate
        value: Any

        if op is SPUOpcode.LADD_SAT:
            if left is None or right is None:
                raise ValueError("LADD.SAT requires two operands")
            value = int(np.clip(int(left) + int(right), arithmetic.SPF32_L_MIN_FINITE, arithmetic.SPF32_L_MAX_FINITE))
        elif op is SPUOpcode.LSUB_SAT:
            if left is None or right is None:
                raise ValueError("LSUB.SAT requires two operands")
            value = int(np.clip(int(left) - int(right), arithmetic.SPF32_L_MIN_FINITE, arithmetic.SPF32_L_MAX_FINITE))
        elif op is SPUOpcode.TADD_MOD:
            if left is None or right is None:
                raise ValueError("TADD.MOD requires two operands")
            value = (int(left) + int(right)) & arithmetic.SPF32_PHASE_MASK
        elif op is SPUOpcode.TSUB_MOD:
            if left is None or right is None:
                raise ValueError("TSUB.MOD requires two operands")
            value = (int(left) - int(right)) & arithmetic.SPF32_PHASE_MASK
        elif op is SPUOpcode.LOGADD_ROM:
            if left is None or right is None:
                raise ValueError("LOGADD.ROM requires two operands")
            value = arithmetic.logadd_pos(
                left.code if isinstance(left, arithmetic.RawLogCode) else left,
                right.code if isinstance(right, arithmetic.RawLogCode) else right,
            )
            if not isinstance(left, (arithmetic.LogPos, arithmetic.LogSigned)) and not isinstance(
                right, (arithmetic.LogPos, arithmetic.LogSigned)
            ):
                value = int(value.L)
        elif op is SPUOpcode.LOGSUB_ROM:
            if left is None or right is None:
                raise ValueError("LOGSUB.ROM requires two operands")
            value = arithmetic.logsub_pos(
                left.code if isinstance(left, arithmetic.RawLogCode) else left,
                right.code if isinstance(right, arithmetic.RawLogCode) else right,
            )
            if not isinstance(left, arithmetic.LogPos) and not isinstance(right, arithmetic.LogPos):
                value = int(value.L)
        elif op is SPUOpcode.SPF_MUL:
            if left is None or right is None:
                raise ValueError("SPF.MUL requires two operands")
            value = arithmetic.spf_mul(left, right)
        elif op is SPUOpcode.SPF_CONJ:
            if left is None:
                raise ValueError("SPF.CONJ requires one operand")
            value = arithmetic.spf_conj(left)
        elif op is SPUOpcode.SPF_PAIR:
            if left is None or right is None:
                raise ValueError("SPF.PAIR requires two operands")
            precision = instruction.immediate if isinstance(instruction.immediate, str) else "balanced"
            value = arithmetic.spf_pair_addsub_hybrid(left, right, precision=precision)
        elif op is SPUOpcode.JET_SET:
            jet = left if isinstance(left, arithmetic.Jet2D) else instruction.immediate
            if not isinstance(jet, arithmetic.Jet2D):
                raise TypeError("JET.SET requires a Jet2D source or immediate")
            value = self._jet_state(jet)
        elif op is SPUOpcode.JET_STEP:
            cursor = self._unwrap_jet_state(left)
            value = dict(cursor)
            emitted = value["value"]
            value["value"] = value["value"] + value["d1"]
            value["d1"] = value["d1"] + value["d2"]
            value["x"] = int(value["x"]) + 1
            value = (emitted, value)
        elif op is SPUOpcode.JET_NEXTROW:
            cursor = self._unwrap_jet_state(left)
            value = dict(cursor)
            value["y"] = int(value["y"]) + 1
            value["x"] = 0
            value["value"] = arithmetic.jet_eval(value["jet"], 0.0, value["y"])
            value["d1"] = arithmetic.jet_eval(value["jet"], 1.0, value["y"]) - value["value"]
            value["d2"] = value["jet"].hess_xx
        elif op is SPUOpcode.PIVOT_MAX:
            value = self._new_pivot(left)
        elif op is SPUOpcode.PIVOT_ADDPOS:
            value = self._new_pivot(left, right, signed=False)
        elif op is SPUOpcode.PIVOT_ADDSIGN:
            if len(operands) >= 3:
                term = arithmetic.LogSigned(int(operands[2]), int(operands[1]))
            elif len(operands) == 2 and isinstance(operands[1], (tuple, list)) and len(operands[1]) == 2:
                term = arithmetic.LogSigned(int(operands[1][1]), int(operands[1][0]))
            elif isinstance(instruction.immediate, (tuple, list)) and len(instruction.immediate) == 2:
                term = arithmetic.LogSigned(int(instruction.immediate[1]), int(instruction.immediate[0]))
            else:
                raise ValueError("PIVOT.ADDSIGN requires sign and log-code operands")
            value = self._new_pivot(left, term, signed=True)
        elif op is SPUOpcode.PIVOT_ADDXY:
            value = self._new_pivot(left, right, signed=False)
        elif op is SPUOpcode.PIVOT_FINAL:
            items = self._pivot_items(left)
            raw_kind = instruction.immediate
            if isinstance(raw_kind, Mapping):
                raw_kind = raw_kind.get("kind", "positive")
            kind = _normal_name(raw_kind or "positive")
            if kind in {"positive", "log_pos", "positive_real"}:
                typed = [self._pivot_value(item) for item in items]
                value = arithmetic.reduce_logpos(typed).value
            elif kind in {"signed", "log_signed", "signed_real"}:
                typed = [self._pivot_value(item, signed=True) for item in items]
                value = arithmetic.reduce_logsigned(typed).value
            elif kind in {"complex", "spf_complex", "spf"}:
                value = arithmetic.reduce_complex(items).value
            else:
                raise ValueError(f"unsupported PIVOT.FINAL kind {raw_kind!r}")
        elif op is SPUOpcode.TRN_COMPOSE:
            if left is None or right is None:
                raise ValueError("TRN.COMPOSE requires two operands")
            value = arithmetic.transport_compose(left, right)
        elif op in {SPUOpcode.TROP_MIN, SPUOpcode.TROP_MAX}:
            if len(operands) == 1 and isinstance(left, (list, tuple, np.ndarray)):
                values = list(np.asarray(left, dtype=object).reshape(-1))
            else:
                values = [left, right]
            if not values:
                raise ValueError(f"{op.value} requires at least one value")
            if any(isinstance(item, (arithmetic.LogPos, arithmetic.LogSigned)) and item.is_nan for item in values):
                value = math.nan
            else:
                selected = min if op is SPUOpcode.TROP_MIN else max
                value = selected(values, key=self._tropical_key)
        elif op is SPUOpcode.BOUND_TEST:
            if len(operands) >= 2:
                budget = self._bound_number(operands[1])
            elif instruction.immediate is not None:
                budget = float(instruction.immediate)
            else:
                raise ValueError("BOUND.TEST requires a budget")
            value = self._bound_number(left) <= budget
        else:  # pragma: no cover - defensive exhaustive enum guard
            raise NotImplementedError(f"unimplemented SPU opcode {op.value!r}")

        self._write(state, instruction.dst, value)
        state.record(op, self._CYCLES[op], instruction)

    def execute_program(
        self,
        instructions: Sequence[SPUInstruction],
        *,
        registers: Mapping[str, Any] | None = None,
        mode: str = "functional",
    ) -> SPUState:
        """Run an Appendix-C program.

        ``mode="functional"`` (default) executes on this class's own
        object-level simulator and charges :attr:`_CYCLES` per op.
        ``mode="rtl"`` assembles the program with :mod:`hw.spu.isa` and runs it
        on the Amaranth §38 core through :mod:`hw.spu.sim`, so ``state.cycles``
        becomes a *measured* cycle count rather than a model.  The RTL path only
        accepts the ops the Amaranth core decodes; anything else raises with the
        op named.
        """

        self.require_available()
        selected = _normal_name(mode)
        if selected == "rtl":
            return self._execute_program_rtl(instructions, registers=registers)
        if selected not in {"functional", "sim", "object"}:
            raise ValueError(f"unsupported SPU execution mode {mode!r}")
        state = SPUState(dict(registers or {}))
        for instruction in instructions:
            self.execute_instruction(instruction, state)
        self.last_state = state
        return state

    # -- Amaranth RTL delegation (hw/spu) ---------------------------------
    #
    # The RTL is a separate carrier: 64-bit architectural registers holding
    # split SPF planes, not Python objects.  These helpers are the only place
    # that translation lives.

    @staticmethod
    def _rtl_modules() -> tuple[Any, Any]:
        """Import ``hw.spu.isa`` / ``hw.spu.sim``; raises if the tree is absent."""

        import sys
        from pathlib import Path

        hardware_root = Path(__file__).resolve().parents[3] / "hw"
        if str(hardware_root) not in sys.path:
            sys.path.insert(0, str(hardware_root))
        from spu import isa as isa_module  # type: ignore[import-not-found]
        from spu import sim as sim_module

        return isa_module, sim_module

    @classmethod
    def rtl_available(cls) -> bool:
        """True when the Amaranth §38 model can be imported and simulated."""

        try:
            cls._rtl_modules()
        except Exception:  # pragma: no cover - amaranth not installed
            return False
        return True

    @classmethod
    def _rtl_register_value(cls, value: Any) -> int:
        """Encode one operand into the §38.2 register carrier."""

        _, sim_module = cls._rtl_modules()
        if isinstance(value, arithmetic.LogSigned):
            return int(sim_module.pack_log(value.L, value.sign))
        if isinstance(value, arithmetic.LogPos):
            return int(sim_module.pack_log(value.L, 0))
        if isinstance(value, arithmetic.RawLogCode):
            return int(sim_module.pack_log(value.code, value.sign if value.sign < 0 else 0))
        if isinstance(value, arithmetic.SPF32Value):
            return int(sim_module.pack_reg(value.word))
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(sim_module.pack_reg(int(value)))
        raise TypeError(f"{type(value).__name__} has no §38.2 register encoding")

    @classmethod
    def _rtl_lower(cls, instruction: SPUInstruction, slots: dict[str, int]) -> Any:
        """Translate one :class:`SPUInstruction` into an encoded RTL instruction."""

        isa_module, _ = cls._rtl_modules()
        unsupported = {SPUOpcode.SPF_PAIR}
        if instruction.opcode in unsupported:
            raise NotImplementedError(
                f"{instruction.opcode.value} has no Amaranth datapath; lower it with "
                "hw.spu.isa.lower_spf_pair (see hw/spu/README.md)"
            )
        micro = isa_module.MicroOp[instruction.opcode.name]
        destination = instruction.dst if isinstance(instruction.dst, str) else instruction.dst[0]
        sources = [slots[name] for name in instruction.sources]
        sources += [0] * (3 - len(sources))
        return isa_module.Instruction(
            micro,
            dst0=slots.setdefault(destination, len(slots) + 1),
            src0=sources[0],
            src1=sources[1],
            src2=sources[2],
        )

    def _execute_program_rtl(
        self,
        instructions: Sequence[SPUInstruction],
        *,
        registers: Mapping[str, Any] | None = None,
    ) -> SPUState:
        isa_module, sim_module = self._rtl_modules()
        slots: dict[str, int] = {}
        initial: dict[int, int] = {}
        for name, value in (registers or {}).items():
            index = slots.setdefault(str(name), len(slots) + 1)
            initial[index] = self._rtl_register_value(value)
        lowered = [self._rtl_lower(instruction, slots) for instruction in instructions]
        run = sim_module.run_program(lowered, registers=initial)
        state = SPUState()
        for name, index in slots.items():
            state.registers[name] = int(run.registers[index])
        for instruction in instructions:
            state.instruction_counts[instruction.opcode.value] = (
                state.instruction_counts.get(instruction.opcode.value, 0) + 1
            )
        state.cycles = int(run.execution_cycles)
        self.last_state = state
        del isa_module
        return state

    @classmethod
    def measured_cycle_table(cls) -> dict[str, dict[str, int]]:
        """``{mnemonic: {"model": n, "measured": n, "delta": n}}`` for every op.

        ``model`` is :attr:`_CYCLES`; ``measured`` comes from running the
        Amaranth §38 core.  ``tests/v02/test_spu_cycle_model.py`` prints this as
        a table and asserts the two agree.
        """

        _, sim_module = cls._rtl_modules()
        measured = sim_module.measure_microop_cycles()
        out: dict[str, dict[str, int]] = {}
        for opcode, model in cls._CYCLES.items():
            key = opcode.value
            if key not in measured:
                out[key] = {"model": int(model), "measured": -1, "delta": 0}
                continue
            value = int(measured[key]["execution"])
            out[key] = {"model": int(model), "measured": value, "delta": value - int(model)}
        if "PIVOT.FINAL/COMPLEX" in measured:
            value = int(measured["PIVOT.FINAL/COMPLEX"]["execution"])
            out["PIVOT.FINAL/COMPLEX"] = {
                "model": int(cls._CYCLES_PIVOT_FINAL_COMPLEX),
                "measured": value,
                "delta": value - int(cls._CYCLES_PIVOT_FINAL_COMPLEX),
            }
        return out


class UnavailableBackend(SFRBackend):
    """Explicit fail-closed adapter for a target without a runtime."""

    def __init__(self, name: str, reason: str) -> None:
        super().__init__(
            BackendCapabilities(
                name=name,
                available=False,
                native=False,
                deterministic=False,
                supported_operations=frozenset(),
                supported_representations=frozenset(),
                supports_parallel_transport=False,
                reason_unavailable=reason,
                status="unavailable",
                cost_source="unavailable",
                execution_class="unavailable",
            ),
            reference_cost_model(),
        )


class CUDABackend(SFRBackend):
    """Optional native CUDA backend for compact packed-SPF32 ocean synthesis.

    CUDA is probed without compiling at import time.  The first
    :class:`~spf.renderer._ocean_cuda.CUDAOceanSampler` construction compiles
    the extension lazily and reports build failures explicitly.  The
    capability set names only the ocean lanes implemented by that extension;
    generic SFR IR, FFT, and Cartesian/FP32 paths stay unsupported.
    """

    _OCEAN_OPERATIONS = frozenset(
        {
            "ocean_sfr32_grid",
            "ocean_sfr32_points",
            "ocean_sfr32_phase",
            "ocean_sfr32_display",
        }
    )

    def __init__(self) -> None:
        unsupported = {
            "execute_ir": "CUDA coverage is limited to compact packed-SPF32 ocean synthesis",
            "fft": "CUDA coverage does not include a cuFFT-backed SFR transform",
            "ifft": "CUDA coverage does not include a cuFFT-backed SFR transform",
            "display": "generic display conversion remains outside the CUDA ocean terminal",
            "fp32_ocean": "the CUDA backend intentionally has no FP32 source lane",
            "filtered_transfer": "normalized frequency-transfer consumption is not lowered here",
        }
        super().__init__(
            BackendCapabilities(
                name="cuda",
                available=False,
                native=False,
                deterministic=True,
                supported_operations=frozenset(),
                supported_representations=frozenset(),
                supports_float64_reference=False,
                supports_integer_only=False,
                supports_parallel_transport=False,
                max_dimension=2,
                reason_unavailable="CUDA SFR capability probe deferred until the backend is requested",
                backend_version="sfr-cuda-ocean-0.1",
                isa_version="sfr-isa-v0.2-cuda-ocean",
                status="deferred",
                execution_class="unavailable",
                unsupported_reasons=unsupported,
                cost_source="unavailable",
            ),
            CostModel(
                "cuda_ocean",
                operation_overrides={
                    "ocean_sfr32_grid": 1.0,
                    "ocean_sfr32_points": 1.0,
                    "ocean_sfr32_phase": 0.05,
                    "ocean_sfr32_display": 4.0,
                },
            ),
        )
        self._cuda_probed = False
        self._cuda_unsupported_reasons = unsupported

    @property
    def capabilities(self) -> BackendCapabilities:
        if not getattr(self, "_cuda_probed", False):
            self._probe_cuda()
        return self._capabilities

    @capabilities.setter
    def capabilities(self, value: BackendCapabilities) -> None:
        self._capabilities = value

    def _probe_cuda(self) -> None:
        if self._cuda_probed:
            return
        self._cuda_probed = True
        try:
            from spf.renderer import _ocean_cuda

            available, reason = _ocean_cuda.availability()
        except Exception as exc:  # pragma: no cover - optional import boundary
            available, reason = False, f"CUDA SFR probe failed: {exc}"
        self._capabilities = BackendCapabilities(
            name="cuda",
            available=available,
            native=available,
            deterministic=True,
            supported_operations=self._OCEAN_OPERATIONS if available else frozenset(),
            supported_representations=(
                frozenset({RepresentationKind.SPARSE_RESIDUAL, RepresentationKind.HERMITIAN_HALF})
                if available
                else frozenset()
            ),
            supports_float64_reference=False,
            supports_integer_only=False,
            supports_parallel_transport=False,
            max_dimension=2,
            reason_unavailable=reason,
            backend_version="sfr-cuda-ocean-0.1",
            isa_version="sfr-isa-v0.2-cuda-ocean",
            status="available_lazy_compile" if available else "unavailable",
            execution_class="cuda_ocean_fused" if available else "unavailable",
            unsupported_reasons=self._cuda_unsupported_reasons,
            cost_source="cuda_ocean_model" if available else "unavailable",
        )

    def runtime_info(self) -> Mapping[str, Any]:
        """Expose the CUDA probe without forcing extension compilation."""

        try:
            from spf.renderer import _ocean_cuda

            return MappingProxyType(dict(_ocean_cuda.runtime_info()))
        except Exception as exc:  # pragma: no cover - diagnostics boundary
            return MappingProxyType({"available": False, "reason": str(exc)})


_TORCH_PLANE_OPERATIONS = frozenset(
    {
        "logadd_pos",
        "logadd_signed",
        "logpos_mul",
        "logpos_div",
        "phase_add",
        "phase_sub",
        "phase_conj",
        "phase_compose",
        "spf_mul",
        "spf_div",
        "spf_conj",
        "spf_neg",
        "spf_abs2",
        "spf_pair_addsub",
        "reduce_positive",
        "reduce_signed",
        "reduce_complex",
        "fft",
        "ifft",
        "display",
        "execute_ir",
        "jet_materialize",
        "filter_chain_suffix",
        "lighting_transport",
        "volume_transport",
        "splat_transport",
        "transport_scan",
    }
)

_TORCH_UNSUPPORTED = {
    "materialize": "the torch backend consumes word planes; dense materialization stays a CPU boundary",
    "reduce_transport": "reduce_transport takes scalar Transport objects, not word planes",
    "atom_transform": "atom algebra is host-side symbolic work, not a device plane op",
    "atom_bound": "atom algebra is host-side symbolic work, not a device plane op",
    "atom_to_jet": "atom algebra is host-side symbolic work, not a device plane op",
}


class TorchBackend(SFRBackend):
    """Design §36 general device backend: word-identical integer planes on torch.

    Every operation runs through :mod:`spf.renderer._planes_torch`, whose words
    are asserted identical to the :mod:`spf.tensor.v02_planes` NumPy oracle
    (``tests/v02/test_sfr_planes_torch_oracle.py``).  The device is CUDA when
    ``torch.cuda.is_available()`` and CPU torch otherwise; the capability object
    says which, and ``native`` is true only on CUDA.  Nothing here decodes to a
    dense complex buffer, and the operations this backend cannot execute are
    declared unsupported with a reason instead of being routed to the CPU
    backend (§39).
    """

    _REPRESENTATIONS = frozenset(
        {
            RepresentationKind.DENSE_TYPED,
            RepresentationKind.SPARSE_RESIDUAL,
            RepresentationKind.HERMITIAN_HALF,
            RepresentationKind.QUADRATIC_JET,
        }
    )

    def __init__(self, *, name: str = "torch", device: str | None = None) -> None:
        self._name = str(name)
        self._requested_device = device
        self._ops: Any = None
        self._module: Any = None
        self._torch_probed = False
        self.host_device_transfers = 0
        # torch is *not* imported here: the registry builds every backend at
        # module import, and importing torch there would cost seconds and pull a
        # heavy optional dependency into every ``spf.renderer`` consumer.  The
        # probe runs on the first capability access, exactly like CUDABackend.
        super().__init__(
            BackendCapabilities(
                name=name,
                available=False,
                native=False,
                deterministic=True,
                supported_operations=frozenset(),
                supported_representations=frozenset(),
                supports_float64_reference=False,
                supports_integer_only=True,
                supports_parallel_transport=True,
                max_dimension=3,
                reason_unavailable="torch SFR plane probe deferred until the backend is requested",
                backend_version="sfr-torch-planes-0.2",
                isa_version="sfr-isa-v0.2",
                status="deferred",
                execution_class="unavailable",
                unsupported_reasons=_TORCH_UNSUPPORTED,
                cost_source="unavailable",
            ),
            cpu_cost_model(),
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        if not getattr(self, "_torch_probed", False):
            self._probe_torch()
        return self._capabilities

    @capabilities.setter
    def capabilities(self, value: BackendCapabilities) -> None:
        self._capabilities = value

    def _probe_torch(self) -> None:
        if self._torch_probed:
            return
        self._torch_probed = True
        device = self._requested_device
        try:
            from spf.renderer import _planes_torch

            self._module = _planes_torch
            available, reason = _planes_torch.availability()
            cuda_present = available and _planes_torch.cuda_available()
        except Exception as exc:  # pragma: no cover - optional import boundary
            available, reason, cuda_present = False, f"torch SFR probe failed: {exc}", False
        wants_cuda = device is not None and str(device).startswith("cuda")
        cuda = cuda_present if device is None else (cuda_present and wants_cuda)
        if wants_cuda and not cuda_present:
            # A pinned-CUDA adapter on a host without a CUDA runtime fails
            # closed; it never silently becomes the CPU torch profile.
            available = False
            reason = reason or "torch reports no CUDA runtime on this host"
        resolved = "cuda" if cuda else "cpu"
        self._capabilities = BackendCapabilities(
            name=self._name,
            available=available,
            # Vendor-native only on CUDA; CPU torch is a portable vectorized
            # profile, the same honesty the CPU backend applies to NumPy.
            native=bool(cuda),
            deterministic=True,
            supported_operations=_TORCH_PLANE_OPERATIONS if available else frozenset(),
            supported_representations=self._REPRESENTATIONS if available else frozenset(),
            supports_float64_reference=False,
            supports_integer_only=True,
            supports_parallel_transport=True,
            max_dimension=3,
            reason_unavailable=reason,
            backend_version="sfr-torch-planes-0.2",
            isa_version="sfr-isa-v0.2",
            status="available" if available else "unavailable",
            execution_class=(f"torch_int_planes_{resolved}" if available else "unavailable"),
            unsupported_reasons=_TORCH_UNSUPPORTED,
            cost_source=("gpu_model" if cuda else "cpu_model") if available else "unavailable",
        )
        self.cost_model = gpu_cost_model() if cuda else cpu_cost_model()

    # ---- device plumbing ----------------------------------------------
    def plane_ops(self) -> Any:
        """The :class:`~spf.renderer._planes_torch.TorchPlanes` for this backend."""

        self.require_available()
        if self._module is None:  # pragma: no cover - probe sets it
            from spf.renderer import _planes_torch

            self._module = _planes_torch
        if self._ops is None:
            self._ops = self._module.planes(self._requested_device)
        return self._ops

    @property
    def device(self) -> str:
        return str(self.plane_ops().device)

    def device_info(self) -> Mapping[str, Any]:
        self.require_available()
        info = dict(self._module.device_info(self._requested_device))
        info["rom_bytes"] = int(self._module.rom_bytes())
        ops = self.plane_ops()
        info["compiled_ops"] = list(getattr(ops, "compiled", ()))
        # What the ROMs actually cost on this device right now, not the host image
        # total: the tables upload lazily, so a run that never touches trig pays
        # only for the integer log ROM.
        info["rom_resident_bytes"] = int(ops.rom.resident_bytes)
        info["rom_resident_tables"] = list(ops.rom.resident)
        return MappingProxyType(info)

    # ---- word/code marshalling ----------------------------------------
    @staticmethod
    def _typed_codes(value: Any) -> tuple[np.ndarray, str] | None:
        """Extract the log codes (and kind) of a typed value or object array."""

        if isinstance(value, arithmetic.LogPos):
            return np.asarray(int(value.L), dtype=np.int64), "LOG_POS"
        if isinstance(value, arithmetic.LogSigned):
            return np.asarray([int(value.L), int(value.sign)], dtype=np.int64), "LOG_SIGNED"
        if isinstance(value, arithmetic.Phase):
            return np.asarray(int(value.T), dtype=np.int64), "PHASE"
        array = np.asarray(value, dtype=object) if isinstance(value, (list, tuple)) else None
        if array is None or array.size == 0:
            return None
        first = array.reshape(-1)[0]
        if isinstance(first, arithmetic.LogPos):
            return np.asarray([int(item.L) for item in array.reshape(-1)], dtype=np.int64).reshape(
                array.shape
            ), "LOG_POS"
        return None

    def _words_in(self, value: Any) -> tuple[Any, bool]:
        """Upload an SPF word scalar/plane to the device; ``True`` when it was scalar."""

        if isinstance(value, arithmetic.SPF32Value):
            array = np.asarray(int(value.word), dtype=np.uint32)
        else:
            array = np.asarray(value)
            if array.dtype == object or array.dtype.kind not in "iu":
                raise NotImplementedError(
                    f"SFR backend {self.name!r} executes packed SPF32 word planes; "
                    f"{type(value).__name__} is not a word carrier"
                )
            array = array.astype(np.uint32)
        scalar = array.ndim == 0
        ops = self.plane_ops()
        self.host_device_transfers += 1
        return ops.upload_words(np.atleast_1d(array)), scalar

    def _words_out(self, tensor: Any, scalar: bool, like: Any) -> Any:
        ops = self.plane_ops()
        self.host_device_transfers += 1
        words = ops.download_words(tensor)
        if scalar:
            value = int(np.asarray(words).reshape(-1)[0])
            if isinstance(like, arithmetic.SPF32Value):
                return arithmetic.SPF32Value(
                    int(arithmetic.unpack_L(value)), int(arithmetic.unpack_T(value))
                )
            return value
        return np.asarray(words, dtype=np.uint32)

    # ---- packed word arithmetic ---------------------------------------
    def _word_binary(self, left: Any, right: Any, kernel: str) -> Any:
        ops = self.plane_ops()
        a, a_scalar = self._words_in(left)
        b, b_scalar = self._words_in(right)
        result = getattr(ops, kernel)(a, b)
        return self._words_out(result, a_scalar and b_scalar, left)

    def spf_mul(self, left: Any, right: Any) -> Any:
        self._require_operation("spf_mul")
        return self._word_binary(left, right, "words_mul")

    def spf_div(self, left: Any, right: Any) -> Any:
        self._require_operation("spf_div")
        return self._word_binary(left, right, "words_div")

    def spf_conj(self, value: Any) -> Any:
        self._require_operation("spf_conj")
        ops = self.plane_ops()
        words, scalar = self._words_in(value)
        return self._words_out(ops.words_conj(words), scalar, value)

    def spf_neg(self, value: Any) -> Any:
        self._require_operation("spf_neg")
        ops = self.plane_ops()
        words, scalar = self._words_in(value)
        return self._words_out(ops.words_neg(words), scalar, value)

    def spf_abs2(self, value: Any) -> Any:
        self._require_operation("spf_abs2")
        ops = self.plane_ops()
        words, scalar = self._words_in(value)
        codes = ops.download_codes(ops.words_abs2(words))
        self.host_device_transfers += 1
        if scalar:
            return arithmetic.LogPos(int(np.asarray(codes).reshape(-1)[0]))
        flat = np.asarray(codes, dtype=np.int64)
        output = np.empty(flat.shape, dtype=object)
        for index in np.ndindex(flat.shape):
            output[index] = arithmetic.LogPos(int(flat[index]))
        return output

    def spf_pair_addsub(self, left: Any, right: Any, spec: Any = "balanced") -> Any:
        self._require_operation("spf_pair_addsub")
        ops = self.plane_ops()
        a, a_scalar = self._words_in(left)
        b, b_scalar = self._words_in(right)
        plus, minus = ops.spf_pair_addsub_hybrid(a, b, spec)
        scalar = a_scalar and b_scalar
        return (
            self._words_out(plus, scalar, left),
            self._words_out(minus, scalar, left),
        )

    # ---- LOG_POS / LOG_SIGNED / PHASE ---------------------------------
    def logadd_pos(self, left: Any, right: Any) -> Any:
        self._require_operation("logadd_pos")
        ops = self.plane_ops()
        if isinstance(left, arithmetic.LogPos) and isinstance(right, arithmetic.LogPos):
            codes = ops.logadd_pos(
                ops.upload_codes(np.asarray([left.L], dtype=np.int64)),
                ops.upload_codes(np.asarray([right.L], dtype=np.int64)),
            )
            return arithmetic.LogPos(int(ops.download_codes(codes)[0]))
        return self._code_binary(left, right, "logadd_pos")

    def logpos_mul(self, left: Any, right: Any) -> Any:
        self._require_operation("logpos_mul")
        if isinstance(left, arithmetic.LogPos) and isinstance(right, arithmetic.LogPos):
            ops = self.plane_ops()
            codes = ops.logpos_mul(
                ops.upload_codes(np.asarray([left.L], dtype=np.int64)),
                ops.upload_codes(np.asarray([right.L], dtype=np.int64)),
            )
            return arithmetic.LogPos(int(ops.download_codes(codes)[0]))
        return self._code_binary(left, right, "logpos_mul")

    def logpos_div(self, left: Any, right: Any) -> Any:
        self._require_operation("logpos_div")
        if isinstance(left, arithmetic.LogPos) and isinstance(right, arithmetic.LogPos):
            ops = self.plane_ops()
            codes = ops.logpos_div(
                ops.upload_codes(np.asarray([left.L], dtype=np.int64)),
                ops.upload_codes(np.asarray([right.L], dtype=np.int64)),
            )
            return arithmetic.LogPos(int(ops.download_codes(codes)[0]))
        return self._code_binary(left, right, "logpos_div")

    def _code_binary(self, left: Any, right: Any, kernel: str) -> Any:
        """Elementwise LOG_POS code-plane op on int code arrays."""

        ops = self.plane_ops()
        a = np.asarray(left, dtype=np.int64)
        b = np.asarray(right, dtype=np.int64)
        self.host_device_transfers += 2
        result = getattr(ops, kernel)(ops.upload_codes(a), ops.upload_codes(b))
        self.host_device_transfers += 1
        return np.asarray(ops.download_codes(result), dtype=np.int64)

    def logadd_signed(self, left: Any, right: Any) -> Any:
        self._require_operation("logadd_signed")
        ops = self.plane_ops()
        if isinstance(left, arithmetic.LogSigned) and isinstance(right, arithmetic.LogSigned):
            codes = ops.signed_add(
                (
                    ops.upload_codes(np.asarray([left.L], dtype=np.int64)),
                    ops.upload_codes(np.asarray([left.sign], dtype=np.int64)),
                ),
                (
                    ops.upload_codes(np.asarray([right.L], dtype=np.int64)),
                    ops.upload_codes(np.asarray([right.sign], dtype=np.int64)),
                ),
            )
            return arithmetic.LogSigned(
                int(ops.download_codes(codes[0])[0]), int(ops.download_codes(codes[1])[0])
            )
        raise NotImplementedError(
            f"SFR backend {self.name!r} adds LOG_SIGNED through (code, sign) planes; "
            "use signed_plane_add for arrays"
        )

    def signed_plane_add(self, left: tuple[Any, Any], right: tuple[Any, Any]) -> tuple[Any, Any]:
        """``logadd_signed`` on a ``(code, sign)`` plane pair (device-resident tensors)."""

        self._require_operation("logadd_signed")
        codes, signs = self.plane_ops().signed_add(left, right)
        return codes, signs

    def phase_compose(self, left: Any, right: Any) -> arithmetic.Phase:
        self._require_operation("phase_compose")
        ops = self.plane_ops()
        a = left.T if isinstance(left, arithmetic.Phase) else int(left)
        b = right.T if isinstance(right, arithmetic.Phase) else int(right)
        codes = ops.phase_compose(
            ops.upload_codes(np.asarray([a], dtype=np.int64)),
            ops.upload_codes(np.asarray([b], dtype=np.int64)),
        )
        return arithmetic.Phase(int(ops.download_codes(codes)[0]))

    def phase_add(self, left: Any, right: Any) -> arithmetic.Phase:
        self._require_operation("phase_add")
        return self.phase_compose(left, right)

    def phase_sub(self, left: Any, right: Any) -> arithmetic.Phase:
        self._require_operation("phase_sub")
        right_code = right.T if isinstance(right, arithmetic.Phase) else int(right)
        return self.phase_compose(left, -right_code)

    def phase_conj(self, value: Any) -> arithmetic.Phase:
        self._require_operation("phase_conj")
        ops = self.plane_ops()
        code = value.T if isinstance(value, arithmetic.Phase) else int(value)
        codes = ops.phase_neg(ops.upload_codes(np.asarray([code], dtype=np.int64)))
        return arithmetic.Phase(int(ops.download_codes(codes)[0]))

    # ---- transforms ----------------------------------------------------
    def fft(self, field: Any, spec: Any = None) -> Any:
        """Word-exact SPF radix-2 transform on the device (§36.6 native butterflies).

        There is no cuFFT/dense path here: a dense transform would be a
        ``REFERENCE_CARTESIAN`` conversion, so a non-word payload is refused.
        """

        self._require_operation("fft")
        return self._transform(field, spec, inverse=False)

    def ifft(self, field: Any, spec: Any = None) -> Any:
        self._require_operation("ifft")
        return self._transform(field, spec, inverse=True)

    def _transform(self, field: Any, spec: Any, *, inverse: bool) -> Any:
        payload = field.representation.payload if isinstance(field, Field) else field
        values = np.asarray(payload)
        if values.dtype != np.uint32:
            raise NotImplementedError(
                f"SFR backend {self.name!r} transforms packed SPF32 word planes only; a dense "
                f"{values.dtype} transform would be a REFERENCE_CARTESIAN conversion"
            )
        ops = self.plane_ops()
        self.host_device_transfers += 2
        words = ops.upload_words(values)
        precision = spec or "balanced"
        result = (
            ops.spf_ifft_radix2(words, precision) if inverse else ops.spf_fft_radix2(words, precision)
        )
        return np.asarray(ops.download_words(result), dtype=np.uint32)

    # ---- typed reductions ---------------------------------------------
    def reduce_positive(self, values: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_positive")
        return self._reduce(values, spec, kind="LOG_POS")

    def reduce_signed(self, values: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_signed")
        return self._reduce(values, spec, kind="LOG_SIGNED")

    def reduce_complex(self, values: Any, spec: Any = None) -> Any:
        self._require_operation("reduce_complex")
        return self._reduce(values, spec, kind="SPF_COMPLEX")

    def reduce_plane(self, values: Any, *, kind: str, axis: int | None = None, spec: Any = None) -> Any:
        """Axis-aware device reduction returning a ``PlaneReduction`` (§36.5)."""

        self._require_operation({"LOG_POS": "reduce_positive", "LOG_SIGNED": "reduce_signed"}.get(
            kind, "reduce_complex"
        ))
        return self._reduce_device(values, spec, kind=kind, axis=axis).to_plane_reduction()

    def _reduce_device(self, values: Any, spec: Any, *, kind: str, axis: int | None) -> Any:
        ops = self.plane_ops()
        self.host_device_transfers += 1
        if kind == "SPF_COMPLEX":
            words = np.asarray(values, dtype=np.uint32)
            return ops.reduce_complex(ops.upload_words(words), axis=axis, spec=spec)
        codes, signs = _log_codes_and_signs(values, kind)
        if kind == "LOG_POS":
            return ops.reduce_logpos(ops.upload_codes(codes), axis=axis, spec=spec)
        return ops.reduce_logsigned(
            (ops.upload_codes(codes), ops.upload_codes(signs)), axis=axis, spec=spec
        )

    def _reduce(self, values: Any, spec: Any, *, kind: str) -> Any:
        """Reduce a flat sequence and return the scalar ``ReductionResult`` shape."""

        try:
            reduction = self._reduce_device(values, spec, kind=kind, axis=None)
        except OverflowError as exc:
            # ``_sum_rows`` falls back to Python big ints above 62 accumulator
            # bits; that branch has no int64 device form, so it is a declared
            # unsupported case, not a silent CPU reroute.
            raise NotImplementedError(
                f"SFR backend {self.name!r} cannot run this reduction on the device: {exc}"
            ) from exc
        return reduction.to_plane_reduction().result(())

    # ---- renderer consumers -------------------------------------------
    def filter_chain_suffix(self, spectrum: Any, transfer: Any, *, spec: Any = None, axes: Any = None) -> Any:
        """§26 suffix on words: device spectral product, then the device SPF inverse transform."""

        self._require_operation("filter_chain_suffix")
        ops = self.plane_ops()
        counters = arithmetic.PerformanceCounters()
        self.host_device_transfers += 3
        result = ops.filter_chain_suffix(
            ops.upload_words(np.asarray(spectrum, dtype=np.uint32)),
            ops.upload_words(np.asarray(transfer, dtype=np.uint32)),
            precision=spec or "balanced",
            counters=counters,
            axes=axes,
        )
        words = np.asarray(ops.download_words(result), dtype=np.uint32)
        self.last_report = self.render_report(
            "filter_chain_suffix",
            resolution=tuple(int(v) for v in words.shape),
            counters=counters,
            representation="spf_word_plane_device",
            elements=int(words.size),
        )
        return words

    def lighting_transport(
        self, transmittance_codes: Any, contribution: tuple[Any, Any]
    ) -> tuple[Any, tuple[Any, Any]]:
        """Ordered lighting/visibility composition on LOG_POS/LOG_SIGNED code planes."""

        self._require_operation("lighting_transport")
        ops = self.plane_ops()
        total_t, total_c = ops.transport_compose_front_to_back(transmittance_codes, contribution)
        return total_t, total_c

    def volume_transport(self, optical_depth_codes: Any, contribution: tuple[Any, Any]) -> Any:
        """§16 volume march: optical depth -> transmittance ROM, then the ordered scan."""

        self._require_operation("volume_transport")
        ops = self.plane_ops()
        transmittance = ops.optical_depth_to_transmittance(optical_depth_codes)
        return ops.transport_compose_front_to_back(transmittance, contribution)

    def transport_scan(self, transmittance_codes: Any, contribution: tuple[Any, Any]) -> Any:
        """Per-segment ordered prefix ``(T_i, C_i)`` on the device."""

        self._require_operation("transport_scan")
        return self.plane_ops().transport_scan(transmittance_codes, contribution)

    def splat_transport(self, bins: Any, background: Any, **options: Any) -> Any:
        """§27.7 splat compositor: the existing device compositor on integer planes."""

        self._require_operation("splat_transport")
        from spf.renderer import _splat_cuda

        result = _splat_cuda.rasterize(
            bins,
            tuple(float(value) for value in background),
            options.pop("depth_limit", None),
            float(options.pop("budget", 0.0)),
            bool(options.pop("early_terminate", False)),
            bool(options.pop("compare_reference", False)),
            device=options.pop("device", self._requested_device),
            **options,
        )
        if result is None:
            raise NotImplementedError(
                f"SFR backend {self.name!r} splat compositor declined this request "
                "(early termination with a positive budget keeps per-pixel tail certificates on the CPU)"
            )
        self.last_report = result.report
        return result

    def jet_materialize(
        self,
        jets: Any,
        regions: Any,
        *,
        origin: Sequence[float] = (0.0, 0.0),
        output_format: Any = None,
    ) -> Any:
        """§13.3 forward-difference jet emission for a batch of tiles on the device."""

        self._require_operation("jet_materialize")
        shape = self._jet_shape(regions)
        if isinstance(regions, Mapping):
            origin = tuple(float(value) for value in regions.get("origin", origin))
        batch = (jets,) if isinstance(jets, arithmetic.Jet2D) else tuple(jets)
        kinds = {str(jet.value_kind) for jet in batch}
        if len(kinds) != 1:
            raise NotImplementedError("device jet emission needs one value kind per batch")
        kind = kinds.pop()
        ops = self.plane_ops()
        coefficients = np.asarray(
            [
                [
                    float(np.real(jet.value)),
                    float(np.real(jet.grad_x)),
                    float(np.real(jet.grad_y)),
                    float(np.real(jet.hess_xx)),
                    float(np.real(jet.hess_xy)),
                    float(np.real(jet.hess_yy)),
                ]
                for jet in batch
            ],
            dtype=np.float64,
        )
        torch = ops.torch
        modulo = (
            (int(arithmetic.SPF32_PHASE_MOD) << int(batch[0].guard_T)) if kind == "phase" else None
        )
        wide = ops.jet_forward_differences(
            torch.as_tensor(coefficients, dtype=torch.float64).to(ops.device),
            (int(shape[0]), int(shape[1])),
            (float(origin[0]), float(origin[1])),
            modulo=modulo,
        )
        self.host_device_transfers += 2
        name = _normal_name(output_format or "native")
        if name in {"spf", "spf32", "word", "words", "uint32"}:
            if kind not in {"log", "log_pos", "log_signed"}:
                raise NotImplementedError(
                    "device jet emission to words covers the log kinds; phase/complex kinds stay on the CPU"
                )
            words = ops.jet_emit_log_words(wide, int(batch[0].guard_L), int(batch[0].sign))
            emitted = np.asarray(ops.download_words(words), dtype=np.uint32)
            return emitted if len(batch) > 1 else emitted[0]
        values = np.asarray(wide.detach().cpu().numpy(), dtype=np.float64)
        return values if len(batch) > 1 else values[0]

    # ---- reports -------------------------------------------------------
    def render_report(
        self,
        workload: str,
        *,
        resolution: tuple[int, ...] = (),
        counters: Any = None,
        representation: str = "spf_word_plane_device",
        elements: int = 0,
        runtime_ms: float = 0.0,
        fallback_conversions: int | None = None,
    ) -> RenderReport:
        """A ``RenderReport`` naming the device and every conversion this backend made."""

        counts = counters if counters is not None else arithmetic.PerformanceCounters()
        conversions = (
            int(getattr(counts, "fallback_conversions", 0))
            if fallback_conversions is None
            else int(fallback_conversions)
        )
        info = dict(self.device_info())
        metrics = RenderMetrics(
            backend=self.name,
            workload=workload,
            resolution=resolution,
            representation=representation,
            runtime_ms=float(runtime_ms),
            bytes_read=int(getattr(counts, "bytes_read", 0)),
            bytes_written=int(getattr(counts, "bytes_written", 0)),
            potential_elements=int(elements),
            materialized_elements=0,
            terminal_output_elements=int(elements),
            positive_logadds=int(getattr(counts, "positive_logadds", 0)),
            complex_adds=int(getattr(counts, "complex_pair_adds", 0)),
            pivot_reductions=int(getattr(counts, "pivot_reductions", 0)),
            transport_compositions=int(getattr(counts, "transport_compositions", 0)),
            fft_count=int(getattr(counts, "fft_butterflies", 0)),
            fallback_conversions=conversions,
            peak_memory=int(info.get("peak_gpu_memory_bytes", 0) or 0),
        )
        report = RenderReport(
            metrics,
            capabilities_used=tuple(sorted(self.capabilities.supported_operations)),
            diagnostics={
                "execution_class": self.capabilities.execution_class,
                "actual_execution_class": self.capabilities.execution_class,
                "hardware_native": self.capabilities.native,
                "device": info,
                "consumer": self._module.CONSUMER,
                "oracle": self._module.ORACLE,
                "cost_model": self.cost_model.name,
                "backend_status": self.capabilities.status,
                "host_device_transfers": int(self.host_device_transfers),
                "fallback_conversions": conversions,
                "peak_gpu_memory_bytes": info.get("peak_gpu_memory_bytes"),
                "block_native_arithmetic": conversions == 0,
            },
        )
        self.last_report = report
        return report


def _log_codes_and_signs(values: Any, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract ``(codes, signs)`` from typed objects, planes or raw code arrays."""

    from spf.tensor import v02_planes as _planes

    if isinstance(values, _planes.LogPosPlane):
        return values.L.astype(np.int64), np.ones(values.L.shape, dtype=np.int64)
    if isinstance(values, _planes.LogSignedPlane):
        return values.L.astype(np.int64), values.sign.astype(np.int64)
    array = np.asarray(values, dtype=object)
    if array.size and isinstance(array.reshape(-1)[0], (arithmetic.LogPos, arithmetic.LogSigned)):
        plane = (
            _planes.LogPosPlane.from_scalars(array)
            if kind == "LOG_POS"
            else _planes.LogSignedPlane.from_scalars(array)
        )
        return _log_codes_and_signs(plane, kind)
    numeric = np.asarray(values)
    if numeric.dtype.kind == "f":
        plane = (
            _planes.LogPosPlane.from_values(numeric)
            if kind == "LOG_POS"
            else _planes.LogSignedPlane.from_values(numeric)
        )
        return _log_codes_and_signs(plane, kind)
    codes = numeric.astype(np.int64)
    return codes, np.where(codes == int(arithmetic.SPF32_L_ZERO), 0, 1).astype(np.int64)


_BACKENDS: dict[str, SFRBackend] = {
    "reference": ReferenceBackend(),
    "python": ReferenceBackend(),
    "cpu_reference": ReferenceBackend(),
    "cpu": CPUBackend(),
    "numpy": CPUBackend(),
    "spu_sim": SPUSimulatorBackend(),
    "spu_simulator": SPUSimulatorBackend(),
    "cuda": CUDABackend(),
    # §36 general device backend: integer SPF32 planes on torch, CUDA when the
    # runtime reports a device and CPU torch otherwise.  ``torch`` follows the
    # device torch picks; ``torch_cpu``/``torch_cuda`` pin it.
    "torch": TorchBackend(),
    "torch_cpu": TorchBackend(name="torch_cpu", device="cpu"),
    "torch_cuda": TorchBackend(name="torch_cuda", device="cuda"),
    "metal": UnavailableBackend("metal", "no compiled Metal SFR extension is installed"),
    "vulkan": UnavailableBackend("vulkan", "no compiled Vulkan SFR extension is installed"),
    "fpga": UnavailableBackend("fpga", "no FPGA bitstream/runtime is attached"),
    "spu": UnavailableBackend("spu", "no physical SPU runtime is attached; use spu_sim"),
}


def get_backend(name: str | SFRBackend) -> SFRBackend:
    if isinstance(name, SFRBackend):
        return name
    normalized = str(name).lower().replace("-", "_")
    if normalized not in _BACKENDS:
        raise KeyError(f"unknown SFR backend {name!r}; available names: {', '.join(sorted(_BACKENDS))}")
    return _BACKENDS[normalized]


def backend_capabilities() -> Mapping[str, BackendCapabilities]:
    return MappingProxyType({name: backend.capabilities for name, backend in _BACKENDS.items()})


def backend_capability_table() -> tuple[CapabilityRecord, ...]:
    rows: list[CapabilityRecord] = []
    for name in sorted(_BACKENDS):
        rows.extend(_BACKENDS[name].capability_table())
    return tuple(rows)


def backend_cost_table() -> tuple[BackendCostRecord, ...]:
    rows: list[BackendCostRecord] = []
    for name in sorted(_BACKENDS):
        rows.extend(_BACKENDS[name].cost_table())
    return tuple(rows)


__all__ = [
    "BackendCapabilities",
    "BackendCostRecord",
    "BackendUnavailableError",
    "CPUBackend",
    "CUDABackend",
    "CapabilityRecord",
    "ReferenceBackend",
    "SFRBackend",
    "SPUInstruction",
    "SPUOpcode",
    "SPUPivot",
    "SPUSimulatorBackend",
    "SPUState",
    "SPUTraceEvent",
    "TorchBackend",
    "UnavailableBackend",
    "backend_capabilities",
    "backend_capability_table",
    "backend_cost_table",
    "get_backend",
]
