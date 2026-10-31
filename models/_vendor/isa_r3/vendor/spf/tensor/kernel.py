"""Typed kernel view over ``SPFBlockV3``.

``SPFBlockMatrix`` is a lazy operator facade. It records axis variance,
structured algebra proofs, and explicit fallback lowering without becoming a new
dense storage format.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_NAN,
    SPF32_L_ZERO,
    SPF32_PHASE_MASK,
    SPF32_WORD_ZERO,
    AccumulationDomain,
    ActiveEncoding,
    ReducerPolicy,
    ReducerSpec,
    RoutingPolicy,
    SPFError,
    _coord_matrix,
    _sat_l16,
    decode_words,
    encode_words,
    pack_words,
    reduce_words,
    spf_conj_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)
from .v2 import ExecutionPlan, ExecutionStep, TensorBlockHeaderV2
from .v3 import (
    BaseModelKind,
    BaseModelRef,
    Representation,
    RepresentationRef,
    ResidualCodecKind,
    ResidualRef,
    SPFBlockV3,
    block_to_complex,
)

_AXIS_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_EINSUM_BRACKET_LABEL_RE = re.compile(r"\[([A-Za-z_][A-Za-z0-9_]*)\]")
_SPECTRAL_INVERTIBILITY_EPS = 1e-12


def _default_reducer() -> ReducerSpec:
    return ReducerSpec(
        policy=ReducerPolicy.AUTO,
        routing=RoutingPolicy.PINNED,
        domain=AccumulationDomain.CARTESIAN,
    )


def _require_contraction_reducer(reducer: ReducerSpec) -> ReducerSpec:
    reducer.require_contraction()
    return reducer


@dataclass(frozen=True)
class AxisSpec:
    name: str
    size: int
    role: str = "free"
    basis: str = "standard"
    group: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not _AXIS_RE.match(str(self.name)):
            raise SPFError(f"invalid axis name {self.name!r}")
        if int(self.size) < 0:
            raise SPFError("axis size must be non-negative")
        allowed_roles = {"free", "domain", "codomain", "batch", "contract", "frequency"}
        if self.role not in allowed_roles:
            raise SPFError(f"invalid axis role {self.role!r}")
        allowed_bases = {"standard", "fourier", "product_group", "learned", "algebra"}
        if self.basis not in allowed_bases:
            raise SPFError(f"invalid axis basis {self.basis!r}")
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "group", None if self.group is None else dict(self.group))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class KernelSignature:
    out_axes: tuple[str, ...]
    in_axes: tuple[str, ...]
    batch_axes: tuple[str, ...] = ()
    basis_in: tuple[str, ...] = ()
    basis_out: tuple[str, ...] = ()
    exactness: str = "quantized"
    reducer_policy: ReducerSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "out_axes", tuple(str(a) for a in self.out_axes))
        object.__setattr__(self, "in_axes", tuple(str(a) for a in self.in_axes))
        object.__setattr__(self, "batch_axes", tuple(str(a) for a in self.batch_axes))
        if self.exactness not in {"exact", "quantized", "approximate"}:
            raise SPFError(f"invalid exactness {self.exactness!r}")
        names = self.out_axes + self.in_axes + self.batch_axes
        if len(set(names)) != len(names):
            raise SPFError("kernel signature axes must be unique")
        if self.reducer_policy is not None and not isinstance(self.reducer_policy, ReducerSpec):
            object.__setattr__(self, "reducer_policy", ReducerSpec(policy=self.reducer_policy))


@dataclass(frozen=True)
class SPFProof:
    law: str
    exact: bool
    metadata_only: bool
    reducer_count: int
    cartesian_fallback_count: int
    error_bound_l: float | None = None
    error_bound_theta: float | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        reducer_count = int(self.reducer_count)
        cartesian_fallback_count = int(self.cartesian_fallback_count)
        if reducer_count < 0:
            raise SPFError("reducer_count must be non-negative")
        if cartesian_fallback_count < 0:
            raise SPFError("cartesian_fallback_count must be non-negative")
        if self.error_bound_l is not None and float(self.error_bound_l) < 0.0:
            raise SPFError("error_bound_l must be non-negative")
        if self.error_bound_theta is not None and float(self.error_bound_theta) < 0.0:
            raise SPFError("error_bound_theta must be non-negative")
        object.__setattr__(self, "reducer_count", reducer_count)
        object.__setattr__(self, "cartesian_fallback_count", cartesian_fallback_count)
        object.__setattr__(
            self,
            "error_bound_l",
            None if self.error_bound_l is None else float(self.error_bound_l),
        )
        object.__setattr__(
            self,
            "error_bound_theta",
            None if self.error_bound_theta is None else float(self.error_bound_theta),
        )
        object.__setattr__(self, "notes", tuple(str(n) for n in self.notes))


@dataclass(frozen=True)
class SPFExecutionPlanParity:
    expression_op: str
    proof_law: str
    plan_ops: tuple[str, ...]
    shape: tuple[int, ...]
    reference_word_digest: str
    word_exact: bool
    mismatched_word_count: int
    max_abs_delta: float
    rel_l2: float
    atol: float
    rtol: float
    passed: bool

    def __post_init__(self) -> None:
        shape = tuple(int(dim) for dim in self.shape)
        if any(dim < 0 for dim in shape):
            raise SPFError("ExecutionPlan parity shape dimensions must be non-negative")
        mismatched_word_count = int(self.mismatched_word_count)
        if mismatched_word_count < 0:
            raise SPFError("mismatched_word_count must be non-negative")
        max_abs_delta = float(self.max_abs_delta)
        rel_l2 = float(self.rel_l2)
        atol = float(self.atol)
        rtol = float(self.rtol)
        for name, value in (
            ("max_abs_delta", max_abs_delta),
            ("rel_l2", rel_l2),
            ("atol", atol),
            ("rtol", rtol),
        ):
            if value < 0.0:
                raise SPFError(f"{name} must be non-negative")
        object.__setattr__(self, "expression_op", str(self.expression_op))
        object.__setattr__(self, "proof_law", str(self.proof_law))
        object.__setattr__(self, "plan_ops", tuple(str(op) for op in self.plan_ops))
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "reference_word_digest", str(self.reference_word_digest))
        object.__setattr__(self, "word_exact", bool(self.word_exact))
        object.__setattr__(self, "mismatched_word_count", mismatched_word_count)
        object.__setattr__(self, "max_abs_delta", max_abs_delta)
        object.__setattr__(self, "rel_l2", rel_l2)
        object.__setattr__(self, "atol", atol)
        object.__setattr__(self, "rtol", rtol)
        object.__setattr__(self, "passed", bool(self.passed))


@dataclass(frozen=True)
class SPFValidationEvidence:
    """External validation evidence for one Python SPFBlockMatrix law.

    The implementation can prove local invariants, but it cannot infer that the
    caller has exercised deterministic fixtures, randomized dense-oracle cases,
    and refusal fixtures. Those are test-suite facts, so callers pass the
    evidence explicitly when checking whether a Python law is ready to port to C.
    """

    deterministic_fixture_count: int
    randomized_case_count: int
    refusal_case_count: int
    dense_oracle_parity: bool
    execution_plan_parity: SPFExecutionPlanParity | None = None
    c_port_implemented: bool = False
    python_c_parity: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "deterministic_fixture_count",
            "randomized_case_count",
            "refusal_case_count",
        ):
            value = int(getattr(self, field_name))
            if value < 0:
                raise SPFError(f"{field_name} must be non-negative")
            object.__setattr__(self, field_name, value)
        object.__setattr__(self, "dense_oracle_parity", bool(self.dense_oracle_parity))
        object.__setattr__(self, "c_port_implemented", bool(self.c_port_implemented))
        object.__setattr__(self, "python_c_parity", bool(self.python_c_parity))
        if self.execution_plan_parity is not None and not isinstance(
            self.execution_plan_parity,
            SPFExecutionPlanParity,
        ):
            raise SPFError("execution_plan_parity must be an SPFExecutionPlanParity report")


@dataclass(frozen=True)
class SPFCPortEligibility:
    expression_op: str
    proof_law: str
    python_law_ready_for_c: bool
    c_port_implemented: bool
    python_c_parity: bool
    native_pl_allowed: bool
    missing_requirements: tuple[str, ...]
    plan_ops: tuple[str, ...]
    reference_ops: tuple[str, ...]
    lowered_deployable: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression_op", str(self.expression_op))
        object.__setattr__(self, "proof_law", str(self.proof_law))
        object.__setattr__(self, "python_law_ready_for_c", bool(self.python_law_ready_for_c))
        object.__setattr__(self, "c_port_implemented", bool(self.c_port_implemented))
        object.__setattr__(self, "python_c_parity", bool(self.python_c_parity))
        object.__setattr__(self, "native_pl_allowed", bool(self.native_pl_allowed))
        object.__setattr__(
            self,
            "missing_requirements",
            tuple(str(requirement) for requirement in self.missing_requirements),
        )
        object.__setattr__(self, "plan_ops", tuple(str(op) for op in self.plan_ops))
        object.__setattr__(self, "reference_ops", tuple(str(op) for op in self.reference_ops))
        object.__setattr__(self, "lowered_deployable", bool(self.lowered_deployable))


@dataclass(frozen=True)
class SPFBlockExpr:
    op: str
    args: tuple[Any, ...] = ()
    block: SPFBlockV3 | None = None
    proof: SPFProof | None = None


def _axis_names(axes: Sequence[str | AxisSpec]) -> tuple[str, ...]:
    return tuple(a.name if isinstance(a, AxisSpec) else str(a) for a in axes)


def _axis_specs(
    axes: Sequence[str | AxisSpec],
    sizes: Sequence[int],
    role: str,
    *,
    basis: str = "standard",
) -> tuple[AxisSpec, ...]:
    if len(axes) != len(sizes):
        raise SPFError("axis count must match size count")
    out: list[AxisSpec] = []
    for axis, size in zip(axes, sizes, strict=True):
        if isinstance(axis, AxisSpec):
            if int(axis.size) != int(size):
                raise SPFError(f"axis {axis.name!r} size {axis.size} != block size {int(size)}")
            out.append(axis)
        else:
            out.append(AxisSpec(str(axis), int(size), role=role, basis=basis))
    return tuple(out)


def _axis_specs_with_optional_shape(
    axes: Sequence[str | AxisSpec],
    sizes: Sequence[int] | None,
    role: str,
    *,
    basis: str = "standard",
) -> tuple[AxisSpec, ...]:
    if sizes is None:
        if not all(isinstance(axis, AxisSpec) for axis in axes):
            raise SPFError("axis sizes are required when axes are not AxisSpec instances")
        return tuple(axes)  # type: ignore[return-value]
    return _axis_specs(axes, sizes, role, basis=basis)


def _check_axes_unique(axes: Sequence[AxisSpec]) -> None:
    names = [a.name for a in axes]
    if len(set(names)) != len(names):
        raise SPFError("kernel axes must be unique")


def _axis_map(axes: Sequence[AxisSpec]) -> dict[str, AxisSpec]:
    return {a.name: a for a in axes}


def _structural_view_block(
    dims: Sequence[int],
    structural_kind: str,
    params: Mapping[str, Any],
) -> SPFBlockV3:
    return SPFBlockV3(
        dims=tuple(int(dim) for dim in dims),
        representation=RepresentationRef(int(Representation.STRUCTURAL_VIEW)),
        base=BaseModelRef(
            int(BaseModelKind.STB_AFFINE_POLY),
            {"structural_kind": str(structural_kind), **dict(params)},
        ),
        residual=ResidualRef(int(ResidualCodecKind.NONE), ()),
    )


def _closed_form_zero_expr(
    shape: Sequence[int],
    proof: SPFProof,
    *,
    batch_rank: int = 0,
    out_rank: int = 1,
    in_rank: int = 1,
) -> SPFBlockExpr:
    ordered_shape = tuple(int(x) for x in shape)
    return SPFBlockExpr(
        "ClosedFormZero",
        block=_structural_view_block(
            ordered_shape,
            "closed_form_zero",
            {
                "shape": ordered_shape,
                "batch_rank": int(batch_rank),
                "out_rank": int(out_rank),
                "in_rank": int(in_rank),
            },
        ),
        proof=proof,
    )


def _closed_form_identity_expr(size: int, basis: str, proof: SPFProof) -> SPFBlockExpr:
    n = int(size)
    return SPFBlockExpr(
        "ClosedFormIdentity",
        block=_structural_view_block(
            (n, n),
            "closed_form_identity",
            {"size": n, "basis": basis},
        ),
        proof=proof,
    )


def _shape_for(sig: KernelSignature, axes: Sequence[AxisSpec]) -> tuple[int, ...]:
    amap = _axis_map(axes)
    return tuple(amap[name].size for name in sig.batch_axes + sig.out_axes + sig.in_axes)


def _ordered_axes(sig: KernelSignature, axes: Sequence[AxisSpec]) -> tuple[AxisSpec, ...]:
    amap = _axis_map(axes)
    return tuple(amap[name] for name in sig.batch_axes + sig.out_axes + sig.in_axes)


def _axis_names_from_sequence(axes: Sequence[str | AxisSpec]) -> tuple[str, ...]:
    return tuple(axis.name if isinstance(axis, AxisSpec) else str(axis) for axis in axes)


def _words_from_complex(z: np.ndarray) -> np.ndarray:
    return encode_words(np.asarray(z, dtype=np.complex128))


def _complex_from_words(words: Any) -> np.ndarray:
    return decode_words(np.asarray(words, dtype=np.uint32))


def _word_digest(words: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(words, dtype=np.uint32))
    h = hashlib.sha256()
    h.update(repr(tuple(int(x) for x in arr.shape)).encode("ascii"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _is_packed_spf_words(value: Any) -> bool:
    return np.asarray(value).dtype == np.uint32


def _as_spf_words(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.uint32:
        return np.asarray(arr, dtype=np.uint32).copy()
    return encode_words(np.asarray(arr, dtype=np.complex128))


def _as_complex_values(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.uint32:
        return _complex_from_words(arr)
    return np.asarray(arr, dtype=np.complex128)


def _stored_carrier_bytes(value: Any) -> int:
    from dataclasses import fields, is_dataclass

    seen: set[int] = set()

    def count(item: Any) -> int:
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            return int(item.nbytes)
        if isinstance(item, (bytes, bytearray)):
            return len(item)
        if isinstance(item, Mapping):
            return sum(count(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(count(v) for v in item)
        if is_dataclass(item):
            return sum(count(getattr(item, f.name)) for f in fields(item))
        return 0

    return count(value)


def _storage_accounting_for_expr(kernel: SPFKernel) -> dict[str, Any]:
    if kernel.expr.op == "BlockCirculantSpectrum":
        return _block_circulant_storage_accounting(kernel)
    return {}


def _require_matrix_dense(kernel: SPFKernel, op_name: str) -> np.ndarray:
    dense = _complex_from_words(kernel.materialize_words())
    if dense.ndim != 2:
        raise SPFError(f"{op_name} requires a matrix-shaped kernel")
    return dense


def _estimated_plan_steps(op: str) -> int:
    if op == "CirculantApply":
        return 3
    if op == "BlockCirculantApply":
        return 1
    if op in {
        "LazySum",
        "LazyHadamard",
        "Bind",
        "Pushforward",
        "LowRankApply",
        "PhaseConcordanceRoute",
        "ResidualPatch",
        "LazyCompose",
        "FallbackContract",
    }:
        return 1
    return 0


def _lowering_cost_accounting(kernel: SPFKernel) -> dict[str, Any]:
    try:
        from .operators import certify_plan
        from .rewrite import DEFAULT_COST_MODEL

        plan, _, _ = kernel.to_execution_plan(_materialize=False)
        cert = certify_plan(plan)
        return {
            "lowered_plan_cost": DEFAULT_COST_MODEL.plan_cost(plan),
            "lowered_plan_steps": len(plan.steps),
            "lowered_reducer_boundary_count": len(plan.reducer_boundaries),
            "lowered_reference_ops": cert.reference_ops,
            "lowered_deployable": bool(cert.deployable),
        }
    except Exception:
        return {
            "lowered_plan_cost": None,
            "lowered_plan_steps": None,
            "lowered_reducer_boundary_count": None,
            "lowered_reference_ops": (),
            "lowered_deployable": None,
        }


def _affine_ramp_execution_plan(
    kernel: SPFKernel, *, materialize: bool = True
) -> tuple[ExecutionPlan, dict[str, np.ndarray], SPFProof] | None:
    block = kernel.expr.block
    if block is None or int(block.base.profile_id) != int(BaseModelKind.STB_AFFINE_POLY):
        return None
    params = block.base.params
    header = params.get("header")
    if not isinstance(header, TensorBlockHeaderV2):
        return None
    if tuple(params.get("bands", ())) or int(block.residual.codec_id) != int(ResidualCodecKind.NONE):
        return None
    if header.active_encoding is not ActiveEncoding.DENSE_ALL:
        return None
    active = np.asarray(params.get("active_indices", ()), dtype=np.int64)
    if active.size != header.total_elements or (
        active.size and (active[0] != 0 or active[-1] != active.size - 1 or np.any(np.diff(active) != 1))
    ):
        return None
    order = int(header.order or 0)
    shape = tuple(int(dim) for dim in header.dims)
    base_words = np.empty(0, dtype=np.uint32)
    if materialize:
        coords = _coord_matrix(active, shape, header.origin, order)
        band_exp = np.asarray(header.band_exp, dtype=np.int64)
        l_val = np.int64(header.global_L_bias) + np.int64(header.coarse_exp)
        if order:
            l_val = l_val + coords @ band_exp
            if header.has_nonlinear_log:
                from .v2 import _mixed_ramp_phase

                l_val = l_val + (coords * coords) @ np.asarray(header.log_quadratic, dtype=np.int64)
                l_val = l_val + _mixed_ramp_phase(coords, header.log_mixed, order)
        l_plane = _sat_l16(np.asarray(l_val, dtype=np.int64)).reshape(shape)
        t_plane = np.full(shape, int(header.global_T_bias) & SPF32_PHASE_MASK, dtype=np.uint16)
        base_words = pack_words(l_plane, t_plane)
    proof = kernel.expr.proof or SPFProof("affine_ramp_view", False, True, 0, 0)
    plan = ExecutionPlan(
        steps=(
            ExecutionStep(
                "ramp",
                ("base",),
                ("out",),
                params={
                    "ramp_R": tuple(tuple(int(x) for x in row) for row in header.ramp_R),
                    "ramp_b": tuple(int(x) for x in header.ramp_b),
                    "quadratic_ramp": tuple(int(x) for x in header.quadratic_ramp),
                    "cubic_ramp": tuple(int(x) for x in header.cubic_ramp),
                    "phase_gradient": tuple(int(x) for x in header.phase_gradient),
                    "mixed_ramp": tuple(int(x) for x in header.mixed_ramp),
                    "cubic_mixed": tuple(int(x) for x in header.cubic_mixed),
                    "nonlinear": bool(header.has_nonlinear_phase),
                    "boundary": "affine_ramp",
                },
            ),
        )
    )
    return plan, {"base": base_words}, proof


def _closed_structural_execution_plan(
    kernel: SPFKernel,
    *,
    materialize: bool = True,
) -> tuple[ExecutionPlan, dict[str, np.ndarray], SPFProof] | None:
    proof = kernel.expr.proof
    if proof is None:
        return None
    if kernel.expr.op == "DiagonalSPF":
        diag_words = np.asarray(kernel.expr.args[0]["diag_words"], dtype=np.uint32).reshape(-1)
        identity_words = (
            encode_words(np.eye(int(diag_words.size), dtype=np.complex128))
            if materialize
            else np.empty(0, dtype=np.uint32)
        )
        plan = ExecutionPlan(
            steps=(
                ExecutionStep(
                    "diagonal_scale",
                    ("identity", "diag"),
                    ("out",),
                    params={"boundary": "diagonal_structural_view"},
                ),
            )
        )
        return plan, {"identity": identity_words, "diag": diag_words}, proof
    if kernel.expr.op == "Permutation":
        indices = np.asarray(kernel.expr.args[0]["indices"], dtype=np.int64).reshape(-1)
        identity_words = (
            encode_words(np.eye(int(indices.size), dtype=np.complex128))
            if materialize
            else np.empty(0, dtype=np.uint32)
        )
        plan = ExecutionPlan(
            steps=(
                ExecutionStep(
                    "gather",
                    ("identity",),
                    ("out",),
                    params={"indices": tuple(int(x) for x in indices), "axis": 0},
                ),
            )
        )
        return plan, {"identity": identity_words}, proof
    if kernel.expr.op == "Roll":
        params = kernel.expr.args[0]
        n = int(params["n"])
        indices = _roll_indices(n, int(params["shift"]))
        identity_words = encode_words(np.eye(n, dtype=np.complex128)) if materialize else np.empty(0, dtype=np.uint32)
        plan = ExecutionPlan(
            steps=(
                ExecutionStep(
                    "gather",
                    ("identity",),
                    ("out",),
                    params={"indices": tuple(int(x) for x in indices), "axis": 0, "boundary": "roll_structural_view"},
                ),
            )
        )
        return plan, {"identity": identity_words}, proof
    return None


_MATERIALIZED_REFERENCE_WITNESS_OPS = frozenset(
    {
        "MaterializedWords",
        "Block",
        "ClosedFormZero",
        "ClosedFormIdentity",
        "CirculantSpectrum",
        "Character",
        "ProductGroupCharacter",
        "BlockCirculantSpectrum",
        "LazyOuterSum",
        "NonAbelianCoefficientTable",
        "KroneckerProduct",
        "JacobianOperator",
        "IteratedOperator",
        "TTStructured",
    }
)


def _allows_materialized_reference_witness(kernel: SPFKernel) -> bool:
    if kernel.expr.op in _MATERIALIZED_REFERENCE_WITNESS_OPS:
        return True
    proof = kernel.expr.proof
    return bool(proof is not None and proof.metadata_only)


@dataclass(frozen=True)
class SPFKernel:
    expr: SPFBlockExpr
    axes: tuple[AxisSpec, ...]
    sig: KernelSignature

    def __post_init__(self) -> None:
        # Other expressions use an envelope as structural metadata, not as a
        # packed scalar carrier. Only these leaves execute block.to_words().
        if (
            self.expr.op in {"Block", "CirculantSpectrum"}
            and self.expr.block is not None
            and self.expr.block.representation.rep_id != int(Representation.POLAR_SPF32)
        ):
            raise SPFError(
                "SPFKernel execution requires POLAR_SPF32; render other widths directly as SPFBlockV3, "
                "or explicitly convert with a quantified error budget"
            )
        object.__setattr__(self, "axes", tuple(self.axes))
        _check_axes_unique(self.axes)
        names = {a.name for a in self.axes}
        missing = set(self.sig.out_axes + self.sig.in_axes + self.sig.batch_axes) - names
        if missing:
            raise SPFError(f"signature references missing axes {sorted(missing)!r}")

    @property
    def proof(self) -> SPFProof | None:
        return self.expr.proof

    def cost(self) -> dict[str, Any]:
        proof = self.expr.proof
        storage_accounting = {
            **_storage_accounting_for_expr(self),
            "stored_carrier_bytes": _stored_carrier_bytes(self),
            "terminal_output_elements": math.prod(_shape_for(self.sig, self.axes)),
            "terminal_output_bytes": 4 * math.prod(_shape_for(self.sig, self.axes)),
            "lowered_plan_cost_units": "sum_of_op_weights_not_shape_scaled",
        }
        if self.expr.op == "StructuralApply":
            storage_accounting["scalar_work_elements"] = math.prod(_shape_for(self.sig, self.axes))
        lowering_accounting = _lowering_cost_accounting(self)
        if proof is None:
            return {
                "expression_op": self.expr.op,
                "law": self.expr.op,
                "exact": False,
                "metadata_only": False,
                "exactness": self.sig.exactness,
                "reducer_count": 0,
                "cartesian_fallback_count": 0,
                "reducer_domain": None,
                "error_bound_l": None,
                "error_bound_theta": None,
                "notes": (),
                "estimated_plan_steps": _estimated_plan_steps(self.expr.op),
                **lowering_accounting,
                **storage_accounting,
            }
        reducer = self.sig.reducer_policy
        return {
            "expression_op": self.expr.op,
            "law": proof.law,
            "exact": bool(proof.exact),
            "metadata_only": bool(proof.metadata_only),
            "exactness": self.sig.exactness,
            "reducer_count": int(proof.reducer_count),
            "cartesian_fallback_count": int(proof.cartesian_fallback_count),
            "reducer_domain": None if reducer is None or reducer.domain is None else reducer.domain.name,
            "error_bound_l": proof.error_bound_l,
            "error_bound_theta": proof.error_bound_theta,
            "notes": proof.notes,
            "estimated_plan_steps": _estimated_plan_steps(self.expr.op),
            **lowering_accounting,
            **storage_accounting,
        }

    def reducer_budget(self) -> int:
        return int(self.expr.proof.reducer_count if self.expr.proof is not None else 0)

    def execution_plan_parity(
        self,
        *,
        atol: float = 2e-2,
        rtol: float = 2e-2,
    ) -> SPFExecutionPlanParity:
        plan, inputs, proof = self.to_execution_plan()
        result = plan.execute(inputs)
        if "out" not in result:
            raise SPFError("ExecutionPlan parity requires an 'out' result")
        got_words = np.asarray(result["out"], dtype=np.uint32)
        want_words = np.asarray(self.materialize_words(), dtype=np.uint32)
        if got_words.shape != want_words.shape:
            raise SPFError(f"ExecutionPlan parity shape mismatch {got_words.shape!r} != {want_words.shape!r}")
        word_equal = got_words == want_words
        got = decode_words(got_words)
        want = decode_words(want_words)
        delta = got - want
        max_abs_delta = float(np.max(np.abs(delta))) if delta.size else 0.0
        rel_l2 = float(np.linalg.norm(delta) / max(float(np.linalg.norm(want)), 1.0))
        return SPFExecutionPlanParity(
            expression_op=self.expr.op,
            proof_law=proof.law,
            plan_ops=tuple(step.op for step in plan.steps),
            shape=tuple(int(x) for x in got.shape),
            reference_word_digest=_word_digest(want_words),
            word_exact=bool(np.all(word_equal)),
            mismatched_word_count=int(np.count_nonzero(~word_equal)),
            max_abs_delta=max_abs_delta,
            rel_l2=rel_l2,
            atol=float(atol),
            rtol=float(rtol),
            passed=bool(np.allclose(got, want, atol=atol, rtol=rtol)),
        )

    def validate_execution_plan_parity(
        self,
        *,
        atol: float = 2e-2,
        rtol: float = 2e-2,
        require_word_exact: bool = False,
    ) -> SPFExecutionPlanParity:
        report = self.execution_plan_parity(atol=atol, rtol=rtol)
        if not report.passed:
            raise SPFError(
                "ExecutionPlan parity failed for "
                f"{report.expression_op}: rel_l2={report.rel_l2:.6g}, "
                f"max_abs_delta={report.max_abs_delta:.6g}"
            )
        if require_word_exact and not report.word_exact:
            raise SPFError(
                "ExecutionPlan word parity failed for "
                f"{report.expression_op}: mismatched_word_count={report.mismatched_word_count}"
            )
        return report

    def c_port_eligibility(
        self,
        evidence: SPFValidationEvidence,
        *,
        require_word_exact_plan: bool = False,
    ) -> SPFCPortEligibility:
        """Check the Python-law gate that must pass before writing a C port."""

        if not isinstance(evidence, SPFValidationEvidence):
            raise SPFError("c_port_eligibility evidence must be an SPFValidationEvidence report")

        missing: list[str] = []
        proof = self.expr.proof
        if proof is None:
            missing.append("proof_metadata")
            proof_law = self.expr.op
        else:
            proof_law = proof.law
            cost = self.cost()
            required_cost_keys = (
                "law",
                "exact",
                "metadata_only",
                "exactness",
                "reducer_count",
                "cartesian_fallback_count",
                "lowered_reference_ops",
                "lowered_deployable",
            )
            for key in required_cost_keys:
                if key not in cost:
                    missing.append(f"cost_field:{key}")
        if evidence.deterministic_fixture_count <= 0:
            missing.append("deterministic_fixtures")
        if evidence.randomized_case_count <= 0:
            missing.append("randomized_dense_oracle_cases")
        if evidence.refusal_case_count <= 0:
            missing.append("refusal_cases")
        if not evidence.dense_oracle_parity:
            missing.append("dense_oracle_parity")

        parity = evidence.execution_plan_parity
        if parity is None:
            parity = self.execution_plan_parity()
        if not parity.passed:
            missing.append("execution_plan_parity")
        if require_word_exact_plan and not parity.word_exact:
            missing.append("execution_plan_word_parity")
        if parity.expression_op != self.expr.op:
            missing.append("execution_plan_parity_expression")
        if parity.proof_law != proof_law:
            missing.append("execution_plan_parity_law")
        expected_shape = tuple(int(x) for x in self.materialize_words().shape)
        if parity.shape != expected_shape:
            missing.append("execution_plan_parity_shape")
        expected_digest = _word_digest(self.materialize_words())
        if parity.reference_word_digest != expected_digest:
            missing.append("execution_plan_parity_reference_digest")
        current_plan, _, _ = self.to_execution_plan()
        expected_plan_ops = tuple(step.op for step in current_plan.steps)
        if parity.plan_ops != expected_plan_ops:
            missing.append("execution_plan_parity_plan_ops")

        cost = self.cost()
        reference_ops = tuple(str(x) for x in cost.get("lowered_reference_ops", ()))
        lowered_deployable = bool(cost.get("lowered_deployable", False))
        python_ready = not missing
        if evidence.c_port_implemented and not evidence.python_c_parity:
            missing.append("python_c_parity")
        native_pl_allowed = bool(
            python_ready and evidence.c_port_implemented and evidence.python_c_parity and lowered_deployable
        )
        return SPFCPortEligibility(
            expression_op=self.expr.op,
            proof_law=proof_law,
            python_law_ready_for_c=python_ready,
            c_port_implemented=evidence.c_port_implemented,
            python_c_parity=evidence.python_c_parity,
            native_pl_allowed=native_pl_allowed,
            missing_requirements=tuple(missing),
            plan_ops=parity.plan_ops,
            reference_ops=reference_ops,
            lowered_deployable=lowered_deployable,
        )

    def explain(self) -> str:
        proof = self.expr.proof
        if proof is None:
            return f"{self.expr.op}: no proof metadata"
        detail = (
            f"{proof.law}: exact={proof.exact} metadata_only={proof.metadata_only} "
            f"reducers={proof.reducer_count} fallbacks={proof.cartesian_fallback_count}"
        )
        if proof.error_bound_l is not None:
            detail += f" error_bound_l={proof.error_bound_l:.6g}"
        if proof.error_bound_theta is not None:
            detail += f" error_bound_theta={proof.error_bound_theta:.6g}"
        if proof.notes:
            detail += f" notes={'; '.join(proof.notes)}"
        return detail

    def trace(self) -> complex:
        structural = _structural_trace(self)
        if structural is not None:
            return structural
        dense = _require_matrix_dense(self, "trace")
        if dense.shape[0] != dense.shape[1]:
            raise SPFError("trace requires a square matrix")
        return complex(np.trace(dense))

    def det(self) -> complex:
        structural = _structural_det(self)
        if structural is not None:
            return structural
        dense = _require_matrix_dense(self, "det")
        if dense.shape[0] != dense.shape[1]:
            raise SPFError("det requires a square matrix")
        return complex(np.linalg.det(dense))

    def spectral_norm(self) -> float:
        structural = _structural_spectral_norm(self)
        if structural is not None:
            return structural
        dense = _require_matrix_dense(self, "spectral_norm")
        if dense.size == 0:
            return 0.0
        return float(np.linalg.svd(dense, compute_uv=False)[0])

    def spectral_map(self, fn: Any) -> SPFKernel:
        if not callable(fn):
            raise SPFError("spectral_map expects a callable")
        if self.expr.op == "CirculantSpectrum":
            assert self.expr.block is not None
            spectrum = block_to_complex(self.expr.block).reshape(-1)
            mapped = np.asarray([fn(x) for x in spectrum], dtype=np.complex128)
            return SPFBlockMatrix.circulant_spectrum(
                mapped,
                out_axis=self.sig.out_axes[0],
                axis=self.sig.in_axes[0],
                reducer_policy=self.sig.reducer_policy,
            )
        if self.expr.op == "DiagonalSPF":
            diag = _complex_from_words(self.expr.args[0]["diag_words"]).reshape(-1)
            mapped = np.asarray([fn(x) for x in diag], dtype=np.complex128)
            return SPFBlockMatrix.diagonal(
                mapped,
                out_axis=self.sig.out_axes[0],
                axis=self.sig.in_axes[0],
                reducer_policy=self.sig.reducer_policy,
            )
        raise SPFError(f"spectral_map is not registered for {self.expr.op}")

    def apply(self, rhs: SPFKernel | np.ndarray) -> SPFKernel:
        if isinstance(rhs, SPFKernel):
            if self.expr.op == "BlockCirculantSpectrum":
                closed = _block_circulant_apply(self, rhs)
                if closed is not None:
                    return closed
            if self.sig.batch_axes != rhs.sig.batch_axes:
                raise SPFError("composition batch axes must match exactly")
            lhs_labels = self.sig.batch_axes + self.sig.out_axes + self.sig.in_axes
            rhs_labels = rhs.sig.batch_axes + rhs.sig.out_axes + rhs.sig.in_axes
            out_labels = self.sig.batch_axes + self.sig.out_axes
            return spfblock_einsum(
                f"{_format_einsum_labels(lhs_labels)},"
                f"{_format_einsum_labels(rhs_labels)}->"
                f"{_format_einsum_labels(out_labels)}",
                self,
                rhs,
                reducer=self.sig.reducer_policy or rhs.sig.reducer_policy,
            )
        arr = np.asarray(rhs)
        if arr.ndim != len(self.sig.in_axes):
            raise SPFError("array rhs rank must match kernel input rank")
        self_axes = _axis_map(self.axes)
        vec_axes = tuple(
            AxisSpec(
                name,
                int(size),
                role="codomain",
                basis=self_axes[name].basis,
                group=self_axes[name].group,
                metadata=self_axes[name].metadata,
            )
            for name, size in zip(self.sig.in_axes, arr.shape, strict=True)
        )
        vec = SPFKernel(
            SPFBlockExpr(
                "MaterializedWords",
                args=(_as_spf_words(arr),),
                proof=SPFProof("materialized_vector", False, False, 0, 0, notes=("quantized_realization",)),
            ),
            vec_axes,
            KernelSignature(
                out_axes=_axis_names(vec_axes),
                in_axes=(),
                exactness="quantized",
                reducer_policy=self.sig.reducer_policy,
            ),
        )
        return self.compose(vec)

    def pullback(
        self,
        axis_map: Mapping[str, str],
        *,
        axes: Sequence[AxisSpec] | None = None,
        out_axes: Sequence[str | AxisSpec] | None = None,
        in_axes: Sequence[str | AxisSpec] | None = None,
        batch_axes: Sequence[str | AxisSpec] | None = None,
    ) -> SPFKernel:
        if not isinstance(axis_map, Mapping):
            raise SPFError("pullback axis_map must be a mapping")
        source_names = self.sig.batch_axes + self.sig.out_axes + self.sig.in_axes
        source_set = set(source_names)
        unknown = set(axis_map) - source_set
        if unknown:
            raise SPFError(f"pullback axis map references unknown axes {sorted(unknown)!r}")
        mapped_names = tuple(str(axis_map.get(name, name)) for name in source_names)
        if len(set(mapped_names)) != len(mapped_names):
            raise SPFError("pullback target axis names must be unique")

        source_axis_map = _axis_map(self.axes)
        if axes is None:
            target_specs = tuple(
                AxisSpec(
                    str(axis_map.get(axis.name, axis.name)),
                    axis.size,
                    role=axis.role,
                    basis=axis.basis,
                    group=axis.group,
                    metadata=axis.metadata,
                )
                for axis in _ordered_axes(self.sig, self.axes)
            )
            batch_names = tuple(str(axis_map.get(name, name)) for name in self.sig.batch_axes)
            out_names = tuple(str(axis_map.get(name, name)) for name in self.sig.out_axes)
            in_names = tuple(str(axis_map.get(name, name)) for name in self.sig.in_axes)
        else:
            if not all(isinstance(axis, AxisSpec) for axis in axes):
                raise SPFError("pullback explicit axes must be AxisSpec instances")
            target_specs = tuple(axes)
            target_map = _axis_map(target_specs)
            missing = set(mapped_names) - set(target_map)
            if missing:
                raise SPFError(f"pullback target axes missing mapped names {sorted(missing)!r}")
            for source_name, target_name in zip(source_names, mapped_names, strict=True):
                source_axis = source_axis_map[source_name]
                target_axis = target_map[target_name]
                if source_axis.size != target_axis.size:
                    raise SPFError(
                        f"pullback axis size mismatch for {source_name!r}->{target_name!r}: "
                        f"{source_axis.size} != {target_axis.size}"
                    )
                if source_axis.basis != target_axis.basis:
                    raise SPFError(
                        f"pullback axis basis mismatch for {source_name!r}->{target_name!r}: "
                        f"{source_axis.basis!r} != {target_axis.basis!r}"
                    )
            if batch_axes is None and out_axes is None and in_axes is None:
                batch_names = tuple(axis.name for axis in target_specs if axis.role == "batch")
                out_names = tuple(axis.name for axis in target_specs if axis.role != "batch")
                in_names = ()
            else:
                batch_names = _axis_names_from_sequence(batch_axes or ())
                out_names = _axis_names_from_sequence(out_axes or ())
                in_names = _axis_names_from_sequence(in_axes or ())
        target_map = _axis_map(target_specs)
        missing_sig = set(batch_names + out_names + in_names) - set(target_map)
        if missing_sig:
            raise SPFError(f"pullback signature references missing target axes {sorted(missing_sig)!r}")
        ordered_specs = tuple(target_map[name] for name in batch_names + out_names + in_names)
        from .algebra import pullback_with_registry

        closed = pullback_with_registry(
            self,
            {str(k): str(v) for k, v in axis_map.items()},
            axes=ordered_specs,
            out_axes=out_names,
            in_axes=in_names,
            batch_axes=batch_names,
        )
        if closed is not None:
            return closed
        proof = SPFProof(
            "pullback_axis_view",
            False,
            True,
            0,
            0,
            notes=(f"axis_map={tuple(sorted((str(k), str(v)) for k, v in axis_map.items()))}",),
        )
        return SPFKernel(
            SPFBlockExpr(
                "Pullback",
                args=(
                    {
                        "source": self,
                        "axis_map": tuple(
                            (name, mapped) for name, mapped in zip(source_names, mapped_names, strict=True)
                        ),
                    },
                ),
                proof=proof,
            ),
            ordered_specs,
            KernelSignature(
                out_names,
                in_names,
                batch_axes=batch_names,
                reducer_policy=self.sig.reducer_policy,
            ),
        )

    def bind(self, rhs: SPFKernel, *, allow_fallback: bool = True) -> SPFKernel:
        if not isinstance(rhs, SPFKernel):
            raise SPFError("bind expects an SPFKernel")
        _validate_sum_compatible(self, rhs)
        from .algebra import bind_with_registry

        closed = bind_with_registry(self, rhs, allow_fallback=allow_fallback)
        if closed is not None:
            return closed
        closed = _closed_hadamard(self, rhs)
        if closed is not None and self.expr.op != "Pullback" and rhs.expr.op != "Pullback":
            return closed
        if not bool(allow_fallback):
            raise SPFError(f"no metadata-only bind law for {self.expr.op} and {rhs.expr.op}")
        batch_axes = tuple(_axis_map(self.axes)[name] for name in self.sig.batch_axes)
        out_axes = tuple(_axis_map(self.axes)[name] for name in self.sig.out_axes)
        in_axes = tuple(_axis_map(self.axes)[name] for name in self.sig.in_axes)
        return SPFKernel(
            SPFBlockExpr(
                "Bind",
                args=(self, rhs),
                proof=SPFProof(
                    "spf_bind_elementwise",
                    False,
                    False,
                    0,
                    0,
                    notes=("pullback operands share an explicit index space",),
                ),
            ),
            batch_axes + out_axes + in_axes,
            KernelSignature(
                tuple(axis.name for axis in out_axes),
                tuple(axis.name for axis in in_axes),
                batch_axes=tuple(axis.name for axis in batch_axes),
                reducer_policy=self.sig.reducer_policy or rhs.sig.reducer_policy,
            ),
        )

    def pushforward(
        self,
        reduce_axes: Sequence[str],
        *,
        out_axes: Sequence[str | AxisSpec] | None = None,
        in_axes: Sequence[str | AxisSpec] | None = None,
        batch_axes: Sequence[str | AxisSpec] | None = None,
        reducer: ReducerSpec | None = None,
    ) -> SPFKernel:
        reduce_names = tuple(str(axis) for axis in reduce_axes)
        if not reduce_names:
            raise SPFError("pushforward requires at least one reduce axis")
        ordered = _ordered_axes(self.sig, self.axes)
        ordered_names = tuple(axis.name for axis in ordered)
        missing = set(reduce_names) - set(ordered_names)
        if missing:
            raise SPFError(f"pushforward references missing axes {sorted(missing)!r}")
        if len(set(reduce_names)) != len(reduce_names):
            raise SPFError("pushforward reduce axes must be unique")
        remaining = tuple(axis for axis in ordered if axis.name not in set(reduce_names))
        remaining_map = _axis_map(remaining)
        batch_names = _axis_names_from_sequence(batch_axes or tuple(axis for axis in remaining if axis.role == "batch"))
        if out_axes is None and in_axes is None:
            out_names = tuple(axis.name for axis in remaining if axis.name not in set(batch_names))
            in_names = ()
        else:
            out_names = _axis_names_from_sequence(out_axes or ())
            in_names = _axis_names_from_sequence(in_axes or ())
        missing_sig = set(batch_names + out_names + in_names) - set(remaining_map)
        if missing_sig:
            raise SPFError(f"pushforward signature references missing remaining axes {sorted(missing_sig)!r}")
        target_order = batch_names + out_names + in_names
        if len(set(target_order)) != len(target_order):
            raise SPFError("pushforward output axes must be unique")
        remaining_order = tuple(axis.name for axis in remaining)
        if set(target_order) != set(remaining_order):
            raise SPFError("pushforward output axes must contain every non-reduced axis exactly once")
        output_permutation = tuple(remaining_order.index(name) for name in target_order)
        ordered_specs = tuple(remaining_map[name] for name in batch_names + out_names + in_names)
        reducer_spec = reducer or self.sig.reducer_policy or _default_reducer()
        reduce_indices = tuple(ordered_names.index(name) for name in reduce_names)
        from .algebra import pushforward_with_registry

        closed = pushforward_with_registry(
            self,
            reduce_names,
            out_axes=out_names,
            in_axes=in_names,
            batch_axes=batch_names,
            reducer=reducer_spec,
        )
        if closed is not None:
            return closed
        reducer_spec = _require_contraction_reducer(reducer_spec)
        cartesian_fallback_count = 1 if reducer_spec.domain == AccumulationDomain.CARTESIAN else 0
        proof = SPFProof(
            "pushforward_reducer_boundary",
            False,
            False,
            max(1, len(reduce_indices)),
            cartesian_fallback_count,
            notes=(f"reduce_axes={reduce_names}",),
        )
        return SPFKernel(
            SPFBlockExpr(
                "Pushforward",
                args=(
                    {
                        "source": self,
                        "reduce_axes": reduce_indices,
                        "reducer": reducer_spec,
                        "output_permutation": output_permutation,
                    },
                ),
                proof=proof,
            ),
            ordered_specs,
            KernelSignature(out_names, in_names, batch_axes=batch_names, reducer_policy=reducer_spec),
        )

    def recompress_low_rank(
        self,
        *,
        max_rank: int,
        max_rel_error: float | None = None,
        drop_tol: float = 0.0,
    ) -> SPFKernel:
        return _recompress_low_rank(self, max_rank=max_rank, max_rel_error=max_rel_error, drop_tol=drop_tol)

    def recompress(
        self,
        *,
        strategy: str = "low_rank",
        rank: int | None = None,
        max_rank: int | None = None,
        max_rel_error: float | None = None,
        drop_tol: float = 0.0,
        split_residual: bool = False,
    ) -> SPFKernel:
        if strategy == "circulant":
            if rank is not None or max_rank is not None:
                raise SPFError("circulant recompression does not accept rank or max_rank")
            if float(drop_tol) != 0.0:
                raise SPFError("circulant recompression does not accept drop_tol")
            return _recompress_circulant(
                self,
                max_rel_error=max_rel_error,
                split_residual=split_residual,
            )
        if strategy == "character":
            if rank is not None or max_rank is not None:
                raise SPFError("character recompression does not accept rank or max_rank")
            if float(drop_tol) != 0.0:
                raise SPFError("character recompression does not accept drop_tol")
            if split_residual:
                raise SPFError("split_residual is not supported for character recompression")
            return _recompress_character(self, max_rel_error=max_rel_error)
        if strategy != "low_rank":
            raise SPFError(f"unsupported recompression strategy {strategy!r}")
        if split_residual:
            raise SPFError("split_residual is only supported for circulant recompression")
        if rank is None and max_rank is None:
            raise SPFError("low-rank recompression requires rank or max_rank")
        if rank is not None and max_rank is not None and int(rank) != int(max_rank):
            raise SPFError("rank and max_rank must agree when both are provided")
        budget = int(max_rank if max_rank is not None else rank)
        return self.recompress_low_rank(
            max_rank=budget,
            max_rel_error=max_rel_error,
            drop_tol=drop_tol,
        )

    def __add__(self, rhs: SPFKernel) -> SPFKernel:
        if not isinstance(rhs, SPFKernel):
            return NotImplemented
        return lazy_sum((self, rhs), reducer=self.sig.reducer_policy or rhs.sig.reducer_policy)

    def __mul__(self, rhs: SPFKernel) -> SPFKernel:
        if not isinstance(rhs, SPFKernel):
            return NotImplemented
        return self.hadamard(rhs)

    def hadamard(self, rhs: SPFKernel, *, allow_fallback: bool = True) -> SPFKernel:
        if not isinstance(rhs, SPFKernel):
            raise SPFError("hadamard expects an SPFKernel")
        return lazy_hadamard(self, rhs, allow_fallback=allow_fallback)

    def __matmul__(self, rhs: SPFKernel) -> SPFKernel:
        if not isinstance(rhs, SPFKernel):
            raise SPFError("matmul expects an SPFKernel")
        self._validate_compose_axes(rhs)
        if self.expr.op == rhs.expr.op == "DiagonalSPF":
            return self.compose(rhs)
        contract_labels = tuple(f"__contract{idx}" for idx in range(len(self.sig.in_axes)))
        lhs_labels = self.sig.batch_axes + self.sig.out_axes + contract_labels
        rhs_labels = rhs.sig.batch_axes + contract_labels + rhs.sig.in_axes
        out_labels = self.sig.batch_axes + self.sig.out_axes + rhs.sig.in_axes
        return spfblock_einsum(
            f"{_format_einsum_labels(lhs_labels)},"
            f"{_format_einsum_labels(rhs_labels)}->"
            f"{_format_einsum_labels(out_labels)}",
            self,
            rhs,
            reducer=self.sig.reducer_policy or rhs.sig.reducer_policy,
        )

    def H(self) -> SPFKernel:
        from .algebra import adjoint_with_registry

        out = adjoint_with_registry(self)
        if out is not None:
            return out
        dense = _complex_from_words(self.materialize_words())
        batch_rank = len(self.sig.batch_axes)
        out_rank = len(self.sig.out_axes)
        in_rank = len(self.sig.in_axes)
        perm = (
            tuple(range(batch_rank))
            + tuple(range(batch_rank + out_rank, batch_rank + out_rank + in_rank))
            + tuple(range(batch_rank, batch_rank + out_rank))
        )
        dense = np.conj(np.transpose(dense, perm))
        batch_axes = tuple(_axis_map(self.axes)[name] for name in self.sig.batch_axes)
        return SPFBlockMatrix.dense(
            dense,
            out_axes=self.sig.in_axes,
            in_axes=self.sig.out_axes,
            batch_axes=batch_axes,
            reducer_policy=self.sig.reducer_policy,
        )

    def compose(self, rhs: SPFKernel, *, reducer: ReducerSpec | None = None) -> SPFKernel:
        from .algebra import compose_with_registry

        if not isinstance(rhs, SPFKernel):
            raise SPFError("compose expects an SPFKernel")
        self._validate_compose_axes(rhs)
        closed = _structural_vector_apply(self, rhs)
        if closed is None:
            closed = compose_with_registry(self, rhs)
        if closed is not None:
            return closed
        reducer_spec = reducer or self.sig.reducer_policy or rhs.sig.reducer_policy or _default_reducer()
        return _fallback_compose(self, rhs, _require_contraction_reducer(reducer_spec))

    def _validate_compose_axes(self, rhs: SPFKernel) -> None:
        if len(self.sig.in_axes) != len(rhs.sig.out_axes):
            raise SPFError("composition rank mismatch")
        if self.sig.batch_axes != rhs.sig.batch_axes:
            raise SPFError("composition batch axes must match exactly")
        left = _axis_map(self.axes)
        right = _axis_map(rhs.axes)
        for name in self.sig.batch_axes:
            a = left[name]
            b = right[name]
            if a.size != b.size:
                raise SPFError(f"batch axis size mismatch: {a.name}={a.size}, {b.name}={b.size}")
            if a.basis != b.basis:
                raise SPFError(f"batch axis basis mismatch: {a.basis!r} != {b.basis!r}")
        for a_name, b_name in zip(self.sig.in_axes, rhs.sig.out_axes, strict=True):
            a = left[a_name]
            b = right[b_name]
            if a.size != b.size:
                raise SPFError(f"contracted axis size mismatch: {a.name}={a.size}, {b.name}={b.size}")
            if a.basis != b.basis:
                raise SPFError(f"contracted axis basis mismatch: {a.basis!r} != {b.basis!r}")
            if a.metadata.get("frame_id") != b.metadata.get("frame_id"):
                raise SPFError("contracted axis frame_id mismatch")

    def to_execution_plan(self, *, _materialize: bool = True) -> tuple[ExecutionPlan, dict[str, np.ndarray], SPFProof]:
        """Lower execution, or inspect the same plan without realizing inputs.

        ``_materialize=False`` is private cost inspection: its placeholder
        inputs are deliberately empty and must never be executed.
        """
        if self.expr.op == "StructuralApply":
            params = self.expr.args[0]
            operator, vector = params["operator"], params["vector"]
            inputs = {"x": vector._plan_words(_materialize)}
            kind = operator.expr.op
            if kind == "DiagonalSPF":
                inputs["diagonal"] = operator.expr.args[0]["diag_words"]
                step = ExecutionStep("elementwise_mul", ("diagonal", "x"), ("out",))
            elif kind in {"Permutation", "Roll"}:
                indices = (
                    operator.expr.args[0]["indices"]
                    if kind == "Permutation"
                    else _roll_indices(
                        _axis_map(operator.axes)[operator.sig.out_axes[0]].size, int(operator.expr.args[0]["shift"])
                    )
                )
                step = ExecutionStep("gather", ("x",), ("out",), params={"indices": indices, "axis": 0})
            else:
                step = ExecutionStep("permute", ("x",), ("out",), params={"axes": (0,)})
            assert self.proof is not None
            return ExecutionPlan(steps=(step,)), inputs, self.proof
        if self.expr.op == "LazySum":
            terms, reducer = self.expr.args
            inputs = {f"term_{idx}": term._plan_words(_materialize) for idx, term in enumerate(terms)}
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "packed_sum" if reducer.domain == AccumulationDomain.LOGPOLAR else "decoded_sum_reference",
                        tuple(inputs),
                        ("out",),
                        reducer=reducer,
                        params={"term_count": len(terms)},
                    ),
                )
            )
            assert self.expr.proof is not None
            return plan, inputs, self.expr.proof
        if self.expr.op == "LazyHadamard":
            lhs, rhs = self.expr.args
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "elementwise_mul",
                        ("lhs", "rhs"),
                        ("out",),
                        params={"boundary": "lazy_hadamard"},
                    ),
                )
            )
            assert self.expr.proof is not None
            return plan, {"lhs": lhs._plan_words(_materialize), "rhs": rhs._plan_words(_materialize)}, self.expr.proof
        if self.expr.op == "Bind":
            lhs, rhs = self.expr.args
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "elementwise_mul",
                        ("lhs", "rhs"),
                        ("out",),
                        params={"boundary": "spf_bind"},
                    ),
                )
            )
            assert self.expr.proof is not None
            return plan, {"lhs": lhs._plan_words(_materialize), "rhs": rhs._plan_words(_materialize)}, self.expr.proof
        if self.expr.op == "Pushforward":
            params = self.expr.args[0]
            output_permutation = tuple(int(x) for x in params.get("output_permutation", ()))
            has_output_permutation = output_permutation and output_permutation != tuple(range(len(output_permutation)))
            reduce_axes = tuple(int(axis) for axis in params["reduce_axes"])
            if (
                params["reducer"].domain != AccumulationDomain.CARTESIAN
                and len(reduce_axes) == 1
                and not has_output_permutation
            ):
                plan = ExecutionPlan(
                    steps=(
                        ExecutionStep(
                            "reduce",
                            ("source",),
                            ("out",),
                            reducer=params["reducer"],
                            params={"axis": reduce_axes[0], "boundary": "pushforward_reduce"},
                        ),
                    )
                )
                assert self.expr.proof is not None
                return plan, {"source": params["source"]._plan_words(_materialize)}, self.expr.proof
            contract_output = "pushed" if has_output_permutation else "out"
            steps = [
                ExecutionStep(
                    "decoded_pushforward_reference",
                    ("source",),
                    (contract_output,),
                    reducer=params["reducer"],
                    params={"axes": params["reduce_axes"]},
                )
            ]
            if contract_output != "out":
                steps.append(
                    ExecutionStep(
                        "permute",
                        (contract_output,),
                        ("out",),
                        params={"axes": output_permutation},
                    )
                )
            plan = ExecutionPlan(steps=tuple(steps))
            assert self.expr.proof is not None
            return plan, {"source": params["source"]._plan_words(_materialize)}, self.expr.proof
        if self.expr.op == "CirculantApply":
            circulant, vector = self.expr.args
            assert circulant.expr.block is not None
            spectrum_words = (
                circulant.expr.block.to_words()
                if _materialize
                else np.broadcast_to(np.uint32(0), circulant.expr.block.dims)
            )
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep("fft", ("x",), ("X",), params={"n": int(spectrum_words.size)}),
                    ExecutionStep("diagonal_scale", ("X", "h"), ("Y",)),
                    ExecutionStep("ifft", ("Y",), ("out",), params={"n": int(spectrum_words.size)}),
                )
            )
            assert self.expr.proof is not None
            return plan, {"x": vector._plan_words(_materialize), "h": spectrum_words}, self.expr.proof
        if self.expr.op == "BlockCirculantApply":
            params = self.expr.args[0]
            operator = params["operator"]
            vector = params["vector"]
            block_params = _block_circulant_params(operator)
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "block_circulant_apply",
                        ("x", "spectrum"),
                        ("out",),
                        reducer=params["reducer"],
                        params={
                            "frequency_count": int(block_params["frequency_count"]),
                            "out_blocks": int(block_params["out_blocks"]),
                            "in_blocks": int(block_params["in_blocks"]),
                        },
                    ),
                )
            )
            assert self.expr.proof is not None
            return (
                plan,
                {
                    "x": vector._plan_words(_materialize),
                    "spectrum": encode_words(_block_circulant_spectrum(operator))
                    if _materialize
                    else np.empty(0, dtype=np.uint32),
                },
                self.expr.proof,
            )
        if self.expr.op == "WideLowRankApply":
            params = self.expr.args[0]
            inputs = {
                "x": params["vector"]._plan_words(_materialize),
                "left": params["left_words"],
                "right": params["right_words"],
                "coeff": params["coeff_words"],
            }
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "decoded_low_rank_scaled_reference",
                        tuple(inputs),
                        ("out",),
                        reducer=params["reducer"],
                        params={"coeff_log2_scale": params["coeff_log2_scale"]},
                    ),
                )
            )
            assert self.proof is not None
            return plan, inputs, self.proof
        if self.expr.op == "LowRankApply":
            params = self.expr.args[0]
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "low_rank_apply",
                        ("x",),
                        ("out",),
                        reducer=params["reducer"],
                        params={
                            "u_factors": params["u_factors"],
                            "v_factors": params["v_factors"],
                            "lambdas": params["lambdas"],
                        },
                    ),
                )
            )
            assert self.expr.proof is not None
            return plan, {"x": params["vector"]._plan_words(_materialize)}, self.expr.proof
        if self.expr.op == "PhaseConcordanceRoute":
            params = self.expr.args[0]
            q_param = params.get("q_words")
            k_param = params.get("k_words")
            v_param = params.get("v_words")
            q_words = np.zeros((0, 0), dtype=np.uint32) if q_param is None else np.asarray(q_param, dtype=np.uint32)
            k_words = np.zeros((0, 0), dtype=np.uint32) if k_param is None else np.asarray(k_param, dtype=np.uint32)
            v_words = (
                np.zeros_like(k_words, dtype=np.uint32) if v_param is None else np.asarray(v_param, dtype=np.uint32)
            )
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "route",
                        ("q_words", "k_words", "v_words"),
                        ("out",),
                        reducer=params["reducer"],
                        params={
                            "causal": bool(params["causal"]),
                            "beta": float(params["beta"]),
                            "l_frac_bits": int(params["l_frac_bits"]),
                            "boundary": "phase_concordance_transport",
                            "reducer_name": params["reducer_name"],
                        },
                    ),
                )
            )
            assert self.expr.proof is not None
            return plan, {"q_words": q_words, "k_words": k_words, "v_words": v_words}, self.expr.proof
        if self.expr.op == "ResidualPatch":
            params = self.expr.args[0]
            reducer = params["reducer"]
            if reducer.domain != AccumulationDomain.CARTESIAN:
                plan = ExecutionPlan(
                    steps=(
                        ExecutionStep(
                            "reduce",
                            ("terms",),
                            ("out",),
                            reducer=reducer,
                            params={"axis": 0, "boundary": "residual_patch"},
                        ),
                    )
                )
                assert self.expr.proof is not None
                return (
                    plan,
                    {
                        "terms": np.stack(
                            (
                                params["base"]._plan_words(_materialize),
                                params["patch"]._plan_words(_materialize),
                            ),
                            axis=0,
                        ),
                    },
                    self.expr.proof,
                )
            plan = ExecutionPlan(
                steps=(
                    ExecutionStep(
                        "decoded_sum_reference",
                        ("base", "patch"),
                        ("out",),
                        reducer=reducer,
                        params={"term_count": 2, "boundary": "residual_patch"},
                    ),
                )
            )
            assert self.expr.proof is not None
            return (
                plan,
                {
                    "base": params["base"]._plan_words(_materialize),
                    "patch": params["patch"]._plan_words(_materialize),
                },
                self.expr.proof,
            )
        if self.expr.op in {"LazyCompose", "FallbackContract"}:
            lhs, rhs, axes, reducer, *rest = self.expr.args
            output_permutation = None
            params_extra: dict[str, Any] = {}
            if rest:
                output_permutation = rest[0]
            if len(rest) > 1 and isinstance(rest[1], dict):
                params_extra = dict(rest[1])
            if (
                self.expr.op == "LazyCompose"
                and reducer.domain != AccumulationDomain.CARTESIAN
                and output_permutation is None
                and tuple(axes) == ((1,), (0,))
                and tuple(params_extra.get("batch_axes", ())) == ()
            ):
                plan = ExecutionPlan(
                    steps=(
                        ExecutionStep(
                            "contract",
                            ("lhs", "rhs"),
                            ("out",),
                            reducer=reducer,
                            params={
                                "transpose_b": False,
                                "causal": False,
                                "boundary": "lazy_compose_contract",
                            },
                        ),
                    )
                )
                assert self.expr.proof is not None
                return (
                    plan,
                    {"lhs": lhs._plan_words(_materialize), "rhs": rhs._plan_words(_materialize)},
                    self.expr.proof,
                )
            contract_output = "contracted" if output_permutation is not None else "out"
            steps = [
                ExecutionStep(
                    "decoded_contract_reference",
                    ("lhs", "rhs"),
                    (contract_output,),
                    reducer=reducer,
                    params={"axes": axes, **params_extra},
                )
            ]
            if output_permutation is not None:
                steps.append(
                    ExecutionStep(
                        "permute",
                        (contract_output,),
                        ("out",),
                        params={"axes": tuple(int(x) for x in output_permutation)},
                    )
                )
            plan = ExecutionPlan(steps=tuple(steps))
            assert self.expr.proof is not None
            return plan, {"lhs": lhs._plan_words(_materialize), "rhs": rhs._plan_words(_materialize)}, self.expr.proof
        if self.expr.op == "Block":
            lowered = _affine_ramp_execution_plan(self, materialize=_materialize)
            if lowered is not None:
                return lowered
        lowered_structural = _closed_structural_execution_plan(self, materialize=_materialize)
        if lowered_structural is not None:
            return lowered_structural
        proof = self.expr.proof or SPFProof(self.expr.op, False, False, 0, 0)
        if not _allows_materialized_reference_witness(self):
            raise SPFError(f"no explicit ExecutionPlan lowering for expression op {self.expr.op!r}")
        plan = ExecutionPlan(
            steps=(
                ExecutionStep(
                    "decoded_materialized_reference",
                    ("result",),
                    ("out",),
                    params={"expression_op": self.expr.op},
                ),
            )
        )
        return plan, {"result": self._plan_words(_materialize)}, proof

    def to_rewrite_context(
        self,
        plan: ExecutionPlan | None = None,
        inputs: Mapping[str, np.ndarray] | None = None,
    ) -> Any:
        """Build the plan-time rewrite context for this lowered kernel.

        Only coefficient-like inputs are exposed as constants. Runtime data
        inputs such as apply vectors remain ordinary environment values so the
        rewrite layer does not accidentally fold data-dependent paths.
        """

        if plan is None or inputs is None:
            plan, inputs, _ = self.to_execution_plan()
        del plan
        from .rewrite import RewriteContext

        constants: dict[str, np.ndarray] = {}
        if self.expr.op == "CirculantApply" and "h" in inputs:
            constants["h"] = np.asarray(inputs["h"], dtype=np.uint32)
        elif self.expr.op == "BlockCirculantApply" and "spectrum" in inputs:
            constants["spectrum"] = np.asarray(inputs["spectrum"], dtype=np.uint32)
        elif self.expr.op == "DiagonalSPF":
            for name in ("identity", "diag"):
                if name in inputs:
                    constants[name] = np.asarray(inputs[name], dtype=np.uint32)
        elif self.expr.op in {"Permutation", "Roll"} and "identity" in inputs:
            constants["identity"] = np.asarray(inputs["identity"], dtype=np.uint32)
        elif self.expr.op == "Block" and "base" in inputs:
            constants["base"] = np.asarray(inputs["base"], dtype=np.uint32)
        return RewriteContext(constants=constants)

    def to_execution_plan_with_context(self) -> tuple[ExecutionPlan, dict[str, np.ndarray], SPFProof, Any]:
        plan, inputs, proof = self.to_execution_plan()
        return plan, inputs, proof, self.to_rewrite_context(plan, inputs)

    def _plan_words(self, materialize: bool) -> np.ndarray:
        return self.materialize_words() if materialize else np.empty(0, dtype=np.uint32)

    def active_addresses(self) -> np.ndarray:
        """Enumerate structural support without constructing a dense matrix."""
        shape = _shape_for(self.sig, self.axes)
        op = self.expr.op
        if op == "ClosedFormZero":
            return np.empty(0, dtype=np.int64)
        if op == "StructuralApply":
            params = self.expr.args[0]
            operator, vector = params["operator"], params["vector"]
            support = vector.active_addresses()
            kind = operator.expr.op
            if kind == "Permutation":
                return np.flatnonzero(np.isin(operator.expr.args[0]["indices"], support))
            if kind == "Roll":
                return np.sort((support + int(operator.expr.args[0]["shift"])) % shape[0])
            # Diagonal multiplication follows the frozen TensorBlock profile:
            # absent source zeros are absorbing, including against a NaN word.
            # Keep a conservative support; zero diagonal entries may shrink it.
            return support
        if op in {"DiagonalSPF", "ClosedFormIdentity", "Permutation", "Roll"}:
            n = shape[0]
            rows = np.arange(n, dtype=np.int64)
            columns = (
                self.expr.args[0]["indices"]
                if op == "Permutation"
                else (rows - int(self.expr.args[0]["shift"])) % n
                if op == "Roll" and n
                else rows
            )
            return rows * shape[1] + columns
        if op == "Block" and self.expr.block is not None:
            return self.expr.block.active_addresses()
        return np.arange(math.prod(shape), dtype=np.int64)

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        """Evaluate addressed structural entries; preserve code-space operators."""
        shape = _shape_for(self.sig, self.axes)
        query = np.asarray(linear, dtype=np.int64)
        flat = query.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= math.prod(shape)):
            raise SPFError("kernel sample outside extent")
        op = self.expr.op
        if op == "Block" and self.expr.block is not None:
            return self.expr.block.sample_words(query, phase=phase)
        if op == "MaterializedWords":
            return np.asarray(self.expr.args[0], dtype=np.uint32).reshape(-1)[query]
        if op == "ClosedFormZero":
            return np.full(query.shape, np.uint32(0x80000000), dtype=np.uint32)
        if op == "StructuralApply":
            params = self.expr.args[0]
            operator, vector = params["operator"], params["vector"]
            kind = operator.expr.op
            addresses = (
                operator.expr.args[0]["indices"][query]
                if kind == "Permutation"
                else (query - int(operator.expr.args[0]["shift"])) % shape[0]
                if kind == "Roll"
                else query
            )
            words = vector.sample_words(addresses, phase=phase)
            if kind == "DiagonalSPF":
                return spf_mul_words(operator.expr.args[0]["diag_words"][query], words)
            return words
        if op in {"DiagonalSPF", "ClosedFormIdentity", "Permutation", "Roll"}:
            rows, cols = np.unravel_index(flat, shape)
            expected = (
                self.expr.args[0]["indices"][rows]
                if op == "Permutation"
                else (rows - int(self.expr.args[0]["shift"])) % shape[1]
                if op == "Roll"
                else rows
            )
            values = self.expr.args[0]["diag_words"][rows] if op == "DiagonalSPF" else np.uint32(0)
            return np.where(cols == expected, values, np.uint32(0x80000000)).reshape(query.shape)
        if op in {"LowRankOuter", "LazyOuterSum"}:
            rows, cols = np.unravel_index(flat, shape)
            params = self.expr.args[0]
            left = _complex_from_words(params["u_words" if op == "LowRankOuter" else "left_words"][rows])
            right = _complex_from_words(params["v_words" if op == "LowRankOuter" else "right_words"][cols])
            coeff = _outer_coefficients(params)
            values = np.sum((left * coeff if op == "LowRankOuter" else left @ coeff) * np.conj(right), axis=-1)
            return _words_from_complex(values).reshape(query.shape)
        if op == "Character":
            rows, cols = np.unravel_index(flat, shape)
            params = self.expr.args[0]
            n = int(params["N"])
            theta = (int(params["alpha"]) * rows + int(params["beta"]) * cols) % n
            values = complex(params.get("scale", 1)) * np.exp(2j * np.pi * theta / n)
            return _words_from_complex(values).reshape(query.shape)
        if op in {"LazyHadamard", "Bind"}:
            a, b = self.expr.args
            return spf_mul_words(a.sample_words(query), b.sample_words(query))
        if op == "KroneckerProduct":
            factors = self.expr.args[0]["factors"]
            rows, cols = np.unravel_index(flat, shape)
            values = np.ones(flat.size, dtype=np.complex128)
            for factor in reversed(factors):
                m, n = _shape_for(factor.sig, factor.axes)
                addresses = (rows % m) * n + cols % n
                values *= _complex_from_words(factor.sample_words(addresses))
                rows, cols = rows // m, cols // n
            return _words_from_complex(values).reshape(query.shape)
        # Contractions execute their declared plan; no invented closure from
        # the rank of a phase generator. Only the requested output is gathered.
        return self.materialize_words().reshape(-1)[query]

    def materialize_words(self) -> np.ndarray:
        op = self.expr.op
        if op == "StructuralApply":
            params = self.expr.args[0]
            operator, vector = params["operator"], params["vector"]
            words = vector.materialize_words()
            kind = operator.expr.op
            if kind == "DiagonalSPF":
                return spf_mul_words(operator.expr.args[0]["diag_words"], words)
            if kind == "Permutation":
                return np.take(words, operator.expr.args[0]["indices"], axis=0)
            if kind == "Roll":
                return np.roll(words, int(operator.expr.args[0]["shift"]), axis=0)
            return np.array(words, copy=True)
        if op in {"Block", "CirculantSpectrum"}:
            assert self.expr.block is not None
            block = self.expr.block
            if int(block.base.profile_id) == int(BaseModelKind.CIRCULANT_SPECTRUM):
                spectrum = block_to_complex(block).reshape(-1)
                return _words_from_complex(_circulant_matrix_from_spectrum(spectrum))
            return block.to_words()
        if op == "MaterializedWords":
            return np.asarray(self.expr.args[0], dtype=np.uint32)
        if op == "ClosedFormZero":
            return encode_words(np.zeros(_shape_for(self.sig, self.axes), dtype=np.complex128))
        if op == "ClosedFormIdentity":
            shape = _shape_for(self.sig, self.axes)
            if len(shape) != 2 or shape[0] != shape[1]:
                raise SPFError("ClosedFormIdentity materialization requires square matrix shape")
            return encode_words(np.eye(shape[0], dtype=np.complex128))
        if op == "Character":
            return _words_from_complex(_materialize_character(self.expr.args[0], _shape_for(self.sig, self.axes)))
        if op == "ProductGroupCharacter":
            return _words_from_complex(
                _materialize_product_character(self.expr.args[0], _shape_for(self.sig, self.axes))
            )
        if op == "QuadraticPhase":
            return _words_from_complex(_quadratic_phase_matrix(self.expr.args[0]))
        if op == "DiagonalSPF":
            diag = _complex_from_words(self.expr.args[0]["diag_words"]).reshape(-1)
            return _words_from_complex(np.diag(diag))
        if op == "Permutation":
            return _words_from_complex(_permutation_matrix(self.expr.args[0]["indices"]))
        if op == "Roll":
            params = self.expr.args[0]
            return _words_from_complex(_roll_matrix(int(params["n"]), int(params["shift"])))
        if op == "KroneckerProduct":
            factors = self.expr.args[0]["factors"]
            dense = _complex_from_words(factors[0].materialize_words())
            for factor in factors[1:]:
                dense = np.kron(dense, _complex_from_words(factor.materialize_words()))
            return _words_from_complex(dense)
        if op == "BlockCirculantSpectrum":
            spectrum = _block_circulant_spectrum(self)
            return _words_from_complex(_block_circulant_matrix_from_spectrum(spectrum))
        if op == "NonAbelianCoefficientTable":
            params = self.expr.args[0]
            coords = _complex_from_words(params["coords_words"])
            basis = _complex_from_words(params["basis_words"]).reshape(
                int(params["basis_count"]),
                int(params["matrix_size"]),
                int(params["matrix_size"]),
            )
            return _words_from_complex(np.tensordot(coords, basis, axes=(0, 0)))
        if op == "LowRankOuter":
            params = self.expr.args[0]
            u = _complex_from_words(params["u_words"])
            v = _complex_from_words(params["v_words"])
            coeff = _outer_coefficients(params)
            return _words_from_complex(u @ np.diag(coeff) @ np.conj(v).T)
        if op == "LazyOuterSum":
            params = self.expr.args[0]
            left = _complex_from_words(params["left_words"])
            coeff = _outer_coefficients(params)
            right = _complex_from_words(params["right_words"])
            return _words_from_complex(left @ coeff @ np.conj(right).T)
        if op == "LazySum":
            terms = self.expr.args[0]
            if not terms:
                raise SPFError("LazySum requires at least one term")
            dense = np.zeros(_shape_for(self.sig, self.axes), dtype=np.complex128)
            for term in terms:
                dense = dense + _complex_from_words(term.materialize_words())
            return _words_from_complex(dense)
        if op == "LazyHadamard":
            lhs, rhs = self.expr.args
            return spf_mul_words(lhs.materialize_words(), rhs.materialize_words())
        if op == "Pullback":
            return _materialize_pullback(self)
        if op == "Bind":
            lhs, rhs = self.expr.args
            return spf_mul_words(lhs.materialize_words(), rhs.materialize_words())
        if op == "Pushforward":
            params = self.expr.args[0]
            dense = _complex_from_words(params["source"].materialize_words())
            axes = tuple(int(axis) for axis in params["reduce_axes"])
            pushed = np.sum(dense, axis=axes)
            output_permutation = tuple(int(x) for x in params.get("output_permutation", ()))
            if output_permutation and output_permutation != tuple(range(len(output_permutation))):
                pushed = np.transpose(pushed, output_permutation)
            return _words_from_complex(pushed)
        if op == "CirculantApply":
            plan, inputs, _ = self.to_execution_plan()
            return np.asarray(plan.execute(inputs)["out"], dtype=np.uint32)
        if op == "BlockCirculantApply":
            plan, inputs, _ = self.to_execution_plan()
            return np.asarray(plan.execute(inputs)["out"], dtype=np.uint32)
        if op in {"LowRankApply", "WideLowRankApply"}:
            plan, inputs, _ = self.to_execution_plan()
            return np.asarray(plan.execute(inputs)["out"], dtype=np.uint32)
        if op == "PhaseConcordanceRoute":
            plan, inputs, _ = self.to_execution_plan()
            if not inputs["q_words"].size:
                return encode_words(np.zeros(_shape_for(self.sig, self.axes), dtype=np.complex128))
            return np.asarray(plan.execute(inputs)["out"], dtype=np.uint32)
        if op == "ResidualPatch":
            params = self.expr.args[0]
            base = _complex_from_words(params["base"].materialize_words())
            patch = _complex_from_words(params["patch"].materialize_words())
            return _words_from_complex(base + patch)
        if op == "JacobianOperator":
            params = self.expr.args[0]
            dense = _complex_from_words(params["matrix_words"])
            residual_words = params.get("residual_words")
            if residual_words is not None:
                dense = dense + _complex_from_words(residual_words)
            return _words_from_complex(dense)
        if op == "IteratedOperator":
            return self.expr.args[0]["operator"].materialize_words()
        if op == "TTStructured":
            return _words_from_complex(_materialize_tt_cores(self.expr.args[0]["core_words"]))
        if op == "GeometricRidge":
            from .header_algebra import materialize_geometric_ridge

            return materialize_geometric_ridge(self.expr.args[0])
        if op in {"LazyCompose", "FallbackContract"}:
            plan, inputs, _ = self.to_execution_plan()
            return np.asarray(plan.execute(inputs)["out"], dtype=np.uint32)
        raise SPFError(f"cannot materialize expression op {op!r}")


class SPFBlockMatrix(SPFKernel):
    @classmethod
    def from_block(
        cls,
        block: SPFBlockV3,
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        batch_axes: Sequence[str | AxisSpec] = (),
        **sig: Any,
    ) -> SPFBlockMatrix:
        proof = sig.pop("proof", None)
        if not isinstance(block, SPFBlockV3):
            raise SPFError("from_block block must be an SPFBlockV3")
        if len(block.dims) != len(batch_axes) + len(out_axes) + len(in_axes):
            raise SPFError("block dims must match batch_axes + out_axes + in_axes")
        batch_stop = len(batch_axes)
        out_stop = batch_stop + len(out_axes)
        batch_specs = _axis_specs(batch_axes, block.dims[:batch_stop], "batch")
        out_specs = _axis_specs(out_axes, block.dims[batch_stop:out_stop], "codomain")
        in_specs = _axis_specs(in_axes, block.dims[out_stop:], "domain")
        signature = KernelSignature(
            out_axes=_axis_names(out_specs),
            in_axes=_axis_names(in_specs),
            batch_axes=_axis_names(batch_specs),
            basis_in=tuple(a.basis for a in in_specs),
            basis_out=tuple(a.basis for a in out_specs),
            **sig,
        )
        default_proof = SPFProof(
            "block_view",
            signature.exactness == "exact",
            True,
            0,
            0,
        )
        return cls(
            SPFBlockExpr("Block", block=block, proof=proof or default_proof),
            batch_specs + out_specs + in_specs,
            signature,
        )

    @classmethod
    def separable_phase(
        cls,
        params: Mapping[str, Any],
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        shape: Sequence[int] | None = None,
        batch_axes: Sequence[str | AxisSpec] = (),
        origin: Sequence[int] = (),
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        return cls._separable_base_model(
            BaseModelKind.SEPARABLE_PHASE,
            params,
            out_axes=out_axes,
            in_axes=in_axes,
            shape=shape,
            batch_axes=batch_axes,
            origin=origin,
            exactness=exactness,
            reducer_policy=reducer_policy,
            proof=proof
            or SPFProof(
                "separable_phase_view",
                exactness == "exact",
                True,
                0,
                0,
                notes=() if exactness == "exact" else ("quantized_realization",),
            ),
        )

    @classmethod
    def separable_logmag(
        cls,
        params: Mapping[str, Any],
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        shape: Sequence[int] | None = None,
        batch_axes: Sequence[str | AxisSpec] = (),
        origin: Sequence[int] = (),
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        return cls._separable_base_model(
            BaseModelKind.SEPARABLE_LOGMAG,
            params,
            out_axes=out_axes,
            in_axes=in_axes,
            shape=shape,
            batch_axes=batch_axes,
            origin=origin,
            exactness=exactness,
            reducer_policy=reducer_policy,
            proof=proof
            or SPFProof(
                "separable_logmag_view",
                exactness == "exact",
                True,
                0,
                0,
                notes=() if exactness == "exact" else ("quantized_realization",),
            ),
        )

    @classmethod
    def clock(
        cls,
        n_ticks: int,
        omega: Sequence[int] | np.ndarray,
        *,
        out_axis: str = "time",
        in_axis: str = "channel",
        tick_origin: int = 0,
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        ticks = int(n_ticks)
        if ticks < 0:
            raise SPFError("clock n_ticks must be non-negative")
        freqs = np.asarray(omega, dtype=np.int64).reshape(-1)
        from .base_models import separable_phase_params

        params = separable_phase_params(
            axis_i=0,
            axis_j=1,
            a=np.arange(ticks, dtype=np.int64),
            omega=freqs,
            closed_form=True,
            a_origin=int(tick_origin),
        )
        return cls.separable_phase(
            params,
            out_axes=(AxisSpec(out_axis, ticks),),
            in_axes=(AxisSpec(in_axis, int(freqs.size)),),
            exactness=exactness,
            reducer_policy=reducer_policy,
            proof=SPFProof(
                "clock_phase_view",
                exactness == "exact",
                True,
                0,
                0,
                notes=() if exactness == "exact" else ("quantized_realization",),
            ),
        )

    @classmethod
    def rope(
        cls,
        n_positions: int,
        head_dim: int,
        *,
        out_axis: str = "position",
        in_axis: str = "channel",
        base: float = 10000.0,
        position_origin: int = 0,
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        positions = int(n_positions)
        width = int(head_dim)
        if positions < 0:
            raise SPFError("RoPE n_positions must be non-negative")
        if width < 0:
            raise SPFError("RoPE head_dim must be non-negative")
        from .base_models import rope_frequencies_u16, separable_phase_params

        omega = rope_frequencies_u16(width, base=base)
        params = separable_phase_params(
            axis_i=0,
            axis_j=1,
            a=np.arange(positions, dtype=np.int64),
            omega=omega,
            closed_form=True,
            a_origin=int(position_origin),
        )
        return cls.separable_phase(
            params,
            out_axes=(AxisSpec(out_axis, positions),),
            in_axes=(AxisSpec(in_axis, width),),
            exactness=exactness,
            reducer_policy=reducer_policy,
            proof=SPFProof(
                "rope_phase_view",
                exactness == "exact",
                True,
                0,
                0,
                notes=() if exactness == "exact" else ("quantized_realization",),
            ),
        )

    @classmethod
    def affine_ramp(
        cls,
        *,
        dims: Sequence[int],
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec] = (),
        batch_axes: Sequence[str | AxisSpec] = (),
        origin: Sequence[int] = (),
        global_L_bias: int = 0,
        global_T_bias: int = 0,
        coarse_exp: int = 0,
        band_exp: Sequence[int] = (),
        ramp_R: Sequence[Sequence[int]] | np.ndarray | None = None,
        ramp_b: Sequence[int] = (),
        quadratic_ramp: Sequence[int] = (),
        cubic_ramp: Sequence[int] = (),
        cubic_mixed: Sequence[int] = (),
        log_quadratic: Sequence[int] = (),
        log_mixed: Sequence[int] = (),
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        shape = tuple(int(dim) for dim in dims)
        if len(shape) != len(batch_axes) + len(out_axes) + len(in_axes):
            raise SPFError("affine_ramp dims must match batch_axes + out_axes + in_axes")
        header = TensorBlockHeaderV2(
            dims=shape,
            origin=tuple(int(x) for x in origin),
            global_L_bias=int(global_L_bias),
            global_T_bias=int(global_T_bias),
            coarse_exp=int(coarse_exp),
            band_exp=tuple(int(x) for x in band_exp),
            ramp_R=ramp_R,
            ramp_b=tuple(int(x) for x in ramp_b),
            quadratic_ramp=tuple(int(x) for x in quadratic_ramp),
            cubic_ramp=tuple(int(x) for x in cubic_ramp),
            cubic_mixed=tuple(int(x) for x in cubic_mixed),
            log_quadratic=tuple(int(x) for x in log_quadratic),
            log_mixed=tuple(int(x) for x in log_mixed),
        )
        block = SPFBlockV3(
            dims=shape,
            origin=tuple(header.origin),
            representation=RepresentationRef(int(Representation.POLAR_SPF32)),
            base=BaseModelRef(
                int(BaseModelKind.STB_AFFINE_POLY),
                {
                    "header": header,
                    "bands": (),
                    "active_indices": np.arange(header.total_elements, dtype=np.int64),
                },
            ),
            residual=ResidualRef(int(ResidualCodecKind.NONE), None),
        )
        return cls.from_block(
            block,
            batch_axes=batch_axes,
            out_axes=out_axes,
            in_axes=in_axes,
            exactness=exactness,
            reducer_policy=reducer_policy,
            proof=SPFProof(
                "affine_ramp_view",
                exactness == "exact",
                True,
                0,
                0,
                notes=() if exactness == "exact" else ("quantized_realization",),
            ),
        )

    @staticmethod
    def small_matmul(
        left: SPFKernel,
        right: SPFKernel,
        *,
        reducer: ReducerSpec | None = None,
    ) -> SPFKernel:
        if not isinstance(left, SPFKernel) or not isinstance(right, SPFKernel):
            raise SPFError("small_matmul expects SPFKernel operands")
        if len(left.sig.batch_axes) != 0 or len(right.sig.batch_axes) != 0:
            raise SPFError("small_matmul expects unbatched small kernels")
        if len(left.sig.out_axes) != 1 or len(left.sig.in_axes) != 1:
            raise SPFError("small_matmul expects rank-2 left kernel")
        if len(right.sig.out_axes) != 1 or len(right.sig.in_axes) != 1:
            raise SPFError("small_matmul expects rank-2 right kernel")
        return left.compose(right, reducer=reducer)

    @classmethod
    def _separable_base_model(
        cls,
        kind: BaseModelKind,
        params: Mapping[str, Any],
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        shape: Sequence[int] | None,
        batch_axes: Sequence[str | AxisSpec],
        origin: Sequence[int],
        exactness: str,
        reducer_policy: ReducerSpec | None,
        proof: SPFProof,
    ) -> SPFBlockMatrix:
        sizes = None if shape is None else tuple(int(x) for x in shape)
        if sizes is not None and len(sizes) != len(batch_axes) + len(out_axes) + len(in_axes):
            raise SPFError("separable shape must match batch_axes + out_axes + in_axes")
        batch_specs = _axis_specs_with_optional_shape(
            batch_axes,
            None if sizes is None else sizes[: len(batch_axes)],
            "batch",
        )
        out_start = len(batch_axes)
        out_stop = out_start + len(out_axes)
        out_specs = _axis_specs_with_optional_shape(
            out_axes,
            None if sizes is None else sizes[out_start:out_stop],
            "codomain",
        )
        in_specs = _axis_specs_with_optional_shape(
            in_axes,
            None if sizes is None else sizes[out_stop:],
            "domain",
        )
        dims = tuple(axis.size for axis in batch_specs + out_specs + in_specs)
        block = SPFBlockV3(
            dims=dims,
            origin=tuple(int(x) for x in origin),
            representation=RepresentationRef(int(Representation.POLAR_SPF32)),
            base=BaseModelRef(int(kind), dict(params)),
            residual=ResidualRef(int(ResidualCodecKind.NONE), None),
        )
        return cls.from_block(
            block,
            batch_axes=batch_specs,
            out_axes=out_specs,
            in_axes=in_specs,
            reducer_policy=reducer_policy,
            exactness=exactness,
            proof=proof,
        )

    @classmethod
    def from_words(
        cls,
        words: np.ndarray,
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        batch_axes: Sequence[str | AxisSpec] = (),
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        arr = np.asarray(words, dtype=np.uint32)
        if arr.ndim != len(batch_axes) + len(out_axes) + len(in_axes):
            raise SPFError("word array rank must match batch_axes + out_axes + in_axes")
        batch_specs = _axis_specs(batch_axes, arr.shape[: len(batch_axes)], "batch")
        out_start = len(batch_axes)
        out_stop = out_start + len(out_axes)
        out_specs = _axis_specs(out_axes, arr.shape[out_start:out_stop], "codomain")
        in_specs = _axis_specs(in_axes, arr.shape[out_stop:], "domain")
        return cls(
            SPFBlockExpr(
                "MaterializedWords",
                args=(arr.copy(),),
                proof=proof or SPFProof("materialized_words", False, False, 0, 0),
            ),
            batch_specs + out_specs + in_specs,
            KernelSignature(
                _axis_names(out_specs),
                _axis_names(in_specs),
                batch_axes=_axis_names(batch_specs),
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def dense(
        cls,
        values: np.ndarray,
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        batch_axes: Sequence[str | AxisSpec] = (),
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        return cls.from_words(
            _as_spf_words(values),
            out_axes=out_axes,
            in_axes=in_axes,
            batch_axes=batch_axes,
            reducer_policy=reducer_policy,
            proof=proof,
        )

    @classmethod
    def zero(
        cls,
        *,
        out_axes: Sequence[str | AxisSpec],
        in_axes: Sequence[str | AxisSpec],
        shape: Sequence[int] | None = None,
        batch_axes: Sequence[str | AxisSpec] = (),
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        sizes = None if shape is None else tuple(int(x) for x in shape)
        if sizes is not None and len(sizes) != len(batch_axes) + len(out_axes) + len(in_axes):
            raise SPFError("zero shape must match batch_axes + out_axes + in_axes")
        batch_specs = _axis_specs_with_optional_shape(
            batch_axes,
            None if sizes is None else sizes[: len(batch_axes)],
            "batch",
        )
        out_start = len(batch_axes)
        out_stop = out_start + len(out_axes)
        out_specs = _axis_specs_with_optional_shape(
            out_axes,
            None if sizes is None else sizes[out_start:out_stop],
            "codomain",
        )
        in_specs = _axis_specs_with_optional_shape(
            in_axes,
            None if sizes is None else sizes[out_stop:],
            "domain",
        )
        ordered_shape = tuple(axis.size for axis in batch_specs + out_specs + in_specs)
        return cls(
            _closed_form_zero_expr(
                ordered_shape,
                proof or SPFProof("closed_form_zero", True, True, 0, 0, error_bound_l=0.0),
                batch_rank=len(batch_specs),
                out_rank=len(out_specs),
                in_rank=len(in_specs),
            ),
            batch_specs + out_specs + in_specs,
            KernelSignature(
                _axis_names(out_specs),
                _axis_names(in_specs),
                batch_axes=_axis_names(batch_specs),
                exactness="exact",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def identity(
        cls,
        size: int,
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        basis: str = "standard",
        group: Mapping[str, Any] | None = None,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        n = int(size)
        if n < 0:
            raise SPFError("identity size must be non-negative")
        out = AxisSpec(out_axis, n, role="codomain", basis=basis, group=group)
        inn = AxisSpec(in_axis, n, role="domain", basis=basis, group=group)
        return cls(
            _closed_form_identity_expr(
                n,
                basis,
                proof or SPFProof("closed_form_identity", True, True, 0, 0),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=(basis,),
                basis_out=(basis,),
                exactness="exact",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def circulant_spectrum(
        cls,
        spectrum: np.ndarray,
        *,
        axis: str = "i",
        out_axis: str = "o",
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        from .spectral_surface import CirculantSpectralSurface

        if _is_packed_spf_words(spectrum):
            block = _block_from_spectrum_words(np.asarray(spectrum, dtype=np.uint32))
        else:
            block = CirculantSpectralSurface.from_spectrum(np.asarray(spectrum, dtype=np.complex128)).to_block()
        n = int(block.dims[0])
        out = AxisSpec(out_axis, n, role="codomain", basis="fourier")
        inn = AxisSpec(axis, n, role="domain", basis="fourier")
        return cls(
            SPFBlockExpr(
                "CirculantSpectrum",
                block=block,
                proof=proof
                or SPFProof(
                    "circulant_spectrum_view",
                    exactness == "exact",
                    True,
                    0,
                    0,
                    notes=() if exactness == "exact" else ("quantized_realization",),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=("fourier",),
                basis_out=("fourier",),
                exactness=exactness,
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def character(
        cls,
        *,
        N: int,
        alpha: int,
        beta: int,
        out_axis: str = "o",
        in_axis: str = "i",
        scale: complex = 1.0 + 0.0j,
        exact_group_semantics: bool = True,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        n = int(N)
        if n <= 0:
            raise SPFError("character modulus N must be positive")
        params = {
            "group": "Z_N",
            "N": n,
            "alpha": int(alpha) % n,
            "beta": int(beta) % n,
            "scale": complex(scale),
            "exact_group_semantics": bool(exact_group_semantics),
        }
        out = AxisSpec(
            out_axis,
            n,
            role="codomain",
            basis="product_group",
            group={"kind": "Z_N", "N": n},
        )
        inn = AxisSpec(
            in_axis,
            n,
            role="domain",
            basis="product_group",
            group={"kind": "Z_N", "N": n},
        )
        return cls(
            SPFBlockExpr(
                "Character",
                args=(params,),
                proof=proof
                or SPFProof(
                    "character_view",
                    bool(exact_group_semantics),
                    True,
                    0,
                    0,
                    notes=() if exact_group_semantics else ("quantized_group_semantics",),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=("product_group",),
                basis_out=("product_group",),
                exactness="exact" if exact_group_semantics else "quantized",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def quadratic_phase(
        cls,
        *,
        N: int,
        a: int,
        b: int,
        c: int,
        out_axis: str = "o",
        in_axis: str = "i",
        scale: complex = 1.0 + 0.0j,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        r"""A quadratic-phase block ``Q(a,b,c)_{ik} = ω^{a i² + b ik + c k²}``.

        ``ω = e^{2πi/N}`` and ``i, k`` run over ``Z_N`` (square, order ``N``). These
        are the discrete metaplectic / Weil-representation generators of the
        2026-09-30 matmul post §3: the DFT is ``Q(0, -1, 0)``, chirps are
        ``Q(a, 0, 0)``, and any product of them closes under
        :func:`_compose_quadratic_phase` (law 2). For ``N = 2^n`` every entry is an
        exact SPF32 word (phases are multiples of ``2^16 / N`` ticks); odd ``N`` is
        quantized because ``2^16 / N`` is not integral.
        """
        n = int(N)
        if n <= 0:
            raise SPFError("quadratic-phase modulus N must be positive")
        params: dict[str, Any] = {
            "kind": "plain",
            "N": n,
            "a": int(a) % n,
            "b": int(b) % n,
            "c": int(c) % n,
            "scale": complex(scale),
        }
        exact = _is_power_of_two(n) and complex(scale) == 1.0 + 0.0j
        return cls._quadratic_phase_node(
            params,
            AxisSpec(out_axis, n, role="codomain", basis="standard"),
            AxisSpec(in_axis, n, role="domain", basis="standard"),
            exact=exact,
            reducer_policy=reducer_policy,
            proof=proof
            or SPFProof(
                "quadratic_phase_view",
                exact,
                True,
                0,
                0,
                notes=() if exact else ("quantized_phase_lattice",),
            ),
        )

    @classmethod
    def _quadratic_phase_node(
        cls,
        params: Mapping[str, Any],
        out: AxisSpec,
        inn: AxisSpec,
        *,
        exact: bool,
        reducer_policy: ReducerSpec | None,
        proof: SPFProof,
    ) -> SPFBlockMatrix:
        return cls(
            SPFBlockExpr("QuadraticPhase", args=(dict(params),), proof=proof),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=(inn.basis,),
                basis_out=(out.basis,),
                exactness="exact" if exact else "quantized",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def product_character(
        cls,
        *,
        moduli: Sequence[int],
        out_frequency: Sequence[int],
        in_frequency: Sequence[int],
        out_axis: str = "o",
        in_axis: str = "i",
        scale: complex = 1.0 + 0.0j,
        exact_group_semantics: bool = True,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        mods = tuple(int(m) for m in moduli)
        if not mods or any(m <= 0 for m in mods):
            raise SPFError("product character moduli must be positive")
        out_freq = _mod_frequency(out_frequency, mods)
        in_freq = _mod_frequency(in_frequency, mods)
        size = int(np.prod(mods, dtype=np.int64))
        group = {"kind": "product_Z", "moduli": mods, "indexing": "no_carry"}
        params = {
            "group": "product_Z",
            "moduli": mods,
            "out_frequency": out_freq,
            "in_frequency": in_freq,
            "scale": complex(scale),
            "exact_group_semantics": bool(exact_group_semantics),
            "indexing": "no_carry",
        }
        out = AxisSpec(out_axis, size, role="codomain", basis="product_group", group=group)
        inn = AxisSpec(in_axis, size, role="domain", basis="product_group", group=group)
        return cls(
            SPFBlockExpr(
                "ProductGroupCharacter",
                args=(params,),
                proof=proof
                or SPFProof(
                    "product_group_character_view",
                    bool(exact_group_semantics),
                    True,
                    0,
                    0,
                    notes=("indexing=no_carry",)
                    if exact_group_semantics
                    else ("indexing=no_carry", "quantized_group_semantics"),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=("product_group",),
                basis_out=("product_group",),
                exactness="exact" if exact_group_semantics else "quantized",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def diagonal(
        cls,
        values: np.ndarray,
        *,
        axis: str = "i",
        out_axis: str = "o",
        basis: str = "standard",
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        diag_words = _as_spf_words(values).reshape(-1)
        n = int(diag_words.size)
        out = AxisSpec(out_axis, n, role="codomain", basis=basis)
        inn = AxisSpec(axis, n, role="domain", basis=basis)
        block = _structural_view_block(
            (n, n),
            "diagonal_spf",
            {
                "size": n,
                "basis": basis,
                "diag_shape": tuple(int(x) for x in diag_words.shape),
            },
        )
        return cls(
            SPFBlockExpr(
                "DiagonalSPF",
                args=({"diag_words": diag_words},),
                block=block,
                proof=proof
                or SPFProof(
                    "diagonal_view",
                    exactness == "exact",
                    True,
                    0,
                    0,
                    notes=() if exactness == "exact" else ("quantized_realization",),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=(inn.basis,),
                basis_out=(out.basis,),
                exactness=exactness,
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def gauge(
        cls,
        values: np.ndarray,
        *,
        axis: str = "i",
        out_axis: str = "o",
        basis: str = "standard",
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        return cls.diagonal(
            values,
            axis=axis,
            out_axis=out_axis,
            basis=basis,
            exactness=exactness,
            reducer_policy=reducer_policy,
            proof=SPFProof(
                "gauge_diagonal_view",
                exactness == "exact",
                True,
                0,
                0,
                notes=() if exactness == "exact" else ("quantized_realization",),
            ),
        )

    @classmethod
    def permutation(
        cls,
        indices: Sequence[int],
        *,
        axis: str = "i",
        out_axis: str = "o",
        basis: str = "standard",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        idx = _validate_permutation(indices)
        n = int(idx.size)
        out = AxisSpec(out_axis, n, role="codomain", basis=basis)
        inn = AxisSpec(axis, n, role="domain", basis=basis)
        block = _structural_view_block(
            (n, n),
            "permutation",
            {"size": n, "basis": basis, "indices_shape": tuple(int(x) for x in idx.shape)},
        )
        return cls(
            SPFBlockExpr(
                "Permutation",
                args=({"indices": idx, "basis": basis},),
                block=block,
                proof=SPFProof("permutation_view", True, True, 0, 0),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=(inn.basis,),
                basis_out=(out.basis,),
                exactness="exact",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def roll(
        cls,
        n: int,
        shift: int,
        *,
        axis: str = "i",
        out_axis: str = "o",
        basis: str = "standard",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        size = int(n)
        if size < 0:
            raise SPFError("roll size must be non-negative")
        out = AxisSpec(out_axis, size, role="codomain", basis=basis)
        inn = AxisSpec(axis, size, role="domain", basis=basis)
        normalized_shift = int(shift) % size if size else 0
        block = _structural_view_block(
            (size, size),
            "roll",
            {"size": size, "shift": normalized_shift, "basis": basis},
        )
        return cls(
            SPFBlockExpr(
                "Roll",
                args=({"n": size, "shift": normalized_shift, "basis": basis},),
                block=block,
                proof=SPFProof("roll_view", True, True, 0, 0),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=(inn.basis,),
                basis_out=(out.basis,),
                exactness="exact",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def kronecker(
        cls,
        factors: Sequence[SPFKernel],
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        fs = tuple(factors)
        if not fs:
            raise SPFError("kronecker requires at least one factor")
        for factor in fs:
            if not isinstance(factor, SPFKernel):
                raise SPFError("kronecker factors must be SPFKernel instances")
            if len(factor.sig.out_axes) != 1 or len(factor.sig.in_axes) != 1:
                raise SPFError("kronecker factors must be rank-2 kernels")
        out_size = int(np.prod([_axis_map(f.axes)[f.sig.out_axes[0]].size for f in fs], dtype=np.int64))
        in_size = int(np.prod([_axis_map(f.axes)[f.sig.in_axes[0]].size for f in fs], dtype=np.int64))
        out = AxisSpec(out_axis, out_size, role="codomain")
        inn = AxisSpec(in_axis, in_size, role="domain")
        reducer_count = sum(f.reducer_budget() for f in fs)
        exact = all(f.sig.exactness == "exact" and f.proof and f.proof.exact for f in fs)
        factor_shapes = tuple(
            (
                int(_axis_map(f.axes)[f.sig.out_axes[0]].size),
                int(_axis_map(f.axes)[f.sig.in_axes[0]].size),
            )
            for f in fs
        )
        block = _structural_view_block(
            (out_size, in_size),
            "kronecker_product",
            {
                "factor_count": len(fs),
                "factor_ops": tuple(str(f.expr.op) for f in fs),
                "factor_shapes": factor_shapes,
                "exact_factors": bool(exact),
            },
        )
        return cls(
            SPFBlockExpr(
                "KroneckerProduct",
                args=({"factors": fs},),
                block=block,
                proof=SPFProof("kronecker_view", exact, True, reducer_count, 0),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                exactness="exact" if exact else "quantized",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def block_circulant_spectrum(
        cls,
        spectrum: Any,
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        block_axes: tuple[str, str] = ("bo", "bi"),
        inner_structure: str = "dense",
        exactness: str = "quantized",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        allowed = {"dense", "diagonal", "low_rank", "sparse", "character"}
        if inner_structure not in allowed:
            raise SPFError(f"unsupported block-circulant inner structure {inner_structure!r}")
        if len(block_axes) != 2:
            raise SPFError("block-circulant block_axes must contain exactly two axis names")
        block_axis_out, block_axis_in = (str(block_axes[0]), str(block_axes[1]))
        if not _AXIS_RE.match(block_axis_out) or not _AXIS_RE.match(block_axis_in):
            raise SPFError("block-circulant block_axes must be valid axis names")
        if block_axis_out == block_axis_in:
            raise SPFError("block-circulant block_axes must be distinct")
        block_axes_tuple = (block_axis_out, block_axis_in)
        low_rank_params: dict[str, Any] | None = None
        sparse_params: dict[str, Any] | None = None
        character_params: dict[str, Any] | None = None
        spectrum_words: np.ndarray | None = None
        diagonal_words: np.ndarray | None = None
        if inner_structure == "low_rank":
            spec, low_rank_params = _block_circulant_low_rank_spectrum_and_params(spectrum)
        elif inner_structure == "sparse":
            spec, sparse_params = _block_circulant_sparse_spectrum_and_params(spectrum)
        elif inner_structure == "character":
            spec, character_params = _block_circulant_character_spectrum_and_params(spectrum)
        else:
            if _is_packed_spf_words(spectrum):
                spectrum_words = np.asarray(spectrum, dtype=np.uint32).copy()
                spec = _complex_from_words(spectrum_words)
            else:
                spec = np.asarray(spectrum, dtype=np.complex128)
        if inner_structure == "diagonal":
            if spec.ndim == 2:
                frequency_count, block_count = (int(x) for x in spec.shape)
                diag_values = spec.copy()
                if spectrum_words is not None:
                    diagonal_words = spectrum_words.copy()
                expanded = np.zeros(
                    (frequency_count, block_count, block_count),
                    dtype=np.complex128,
                )
                if block_count:
                    idx = np.arange(block_count)
                    expanded[:, idx, idx] = spec
                spec = expanded
            elif spec.ndim == 3:
                if spec.shape[1] != spec.shape[2]:
                    raise SPFError("diagonal block-circulant spectra must be square")
                off_diag = spec.copy()
                idx = np.arange(spec.shape[1])
                off_diag[:, idx, idx] = 0.0
                if np.any(np.abs(off_diag) > 1e-12):
                    raise SPFError("diagonal block-circulant spectrum contains off-diagonal entries")
                diag_values = np.diagonal(spec, axis1=1, axis2=2).copy()
                if spectrum_words is not None:
                    diagonal_words = np.diagonal(spectrum_words, axis1=1, axis2=2).copy()
            else:
                raise SPFError("diagonal block-circulant spectrum must have shape (frequency, block)")
        elif spec.ndim != 3:
            raise SPFError("block-circulant spectrum must have shape (frequency, out_block, in_block)")
        frequency_count, out_blocks, in_blocks = (int(x) for x in spec.shape)
        if frequency_count <= 0:
            raise SPFError("block-circulant frequency count must be positive")
        if out_blocks <= 0 or in_blocks <= 0:
            raise SPFError("block-circulant block dimensions must be positive")
        out = AxisSpec(
            out_axis,
            frequency_count * out_blocks,
            role="codomain",
            basis="fourier",
            metadata={
                "frequency_count": frequency_count,
                "block_count": out_blocks,
                "block_axis": block_axis_out,
                "block_axes": block_axes_tuple,
            },
        )
        inn = AxisSpec(
            in_axis,
            frequency_count * in_blocks,
            role="domain",
            basis="fourier",
            metadata={
                "frequency_count": frequency_count,
                "block_count": in_blocks,
                "block_axis": block_axis_in,
                "block_axes": block_axes_tuple,
            },
        )
        params = {
            "frequency_count": frequency_count,
            "out_blocks": out_blocks,
            "in_blocks": in_blocks,
            "block_axes": block_axes_tuple,
            "inner_structure": inner_structure,
        }
        if inner_structure == "dense":
            params["spectrum_words"] = (
                spectrum_words.reshape(spec.shape).copy() if spectrum_words is not None else encode_words(spec)
            )
        elif inner_structure == "diagonal":
            params["diagonal_words"] = diagonal_words if diagonal_words is not None else encode_words(diag_values)
        if low_rank_params is not None:
            params.update(low_rank_params)
        if sparse_params is not None:
            params.update(sparse_params)
        if character_params is not None:
            params.update(character_params)
        block = _structural_view_block(
            (frequency_count * out_blocks, frequency_count * in_blocks),
            "block_circulant_spectrum",
            {
                "frequency_count": frequency_count,
                "out_blocks": out_blocks,
                "in_blocks": in_blocks,
                "block_axes": block_axes_tuple,
                "inner_structure": inner_structure,
            },
        )
        return cls(
            SPFBlockExpr(
                "BlockCirculantSpectrum",
                args=(params,),
                block=block,
                proof=proof
                or SPFProof(
                    "block_circulant_spectrum_view",
                    exactness == "exact",
                    True,
                    0,
                    0,
                    notes=() if exactness == "exact" else ("quantized_realization",),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=("fourier",),
                basis_out=("fourier",),
                exactness=exactness,
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def nonabelian(
        cls,
        basis_ref: np.ndarray,
        coords: np.ndarray,
        multiplication_table: np.ndarray,
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        exact_table: bool = False,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        return cls.nonabelian_coefficients(
            coords,
            basis_ref,
            multiplication_table,
            out_axis=out_axis,
            in_axis=in_axis,
            exact_table=exact_table,
            reducer_policy=reducer_policy,
            proof=proof,
        )

    @classmethod
    def nonabelian_coefficients(
        cls,
        coords: np.ndarray,
        basis: np.ndarray,
        multiplication_table: np.ndarray,
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        exact_table: bool = False,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        cc = _as_complex_values(coords).reshape(-1)
        bb = _as_complex_values(basis)
        table = _as_complex_values(multiplication_table)
        _validate_nonabelian_table(cc, bb, table, exact_table=exact_table)
        n = int(bb.shape[1])
        out = AxisSpec(out_axis, n, role="codomain", basis="algebra")
        inn = AxisSpec(in_axis, n, role="domain", basis="algebra")
        params = {
            "coords_words": _as_spf_words(coords).reshape(cc.shape),
            "basis_words": _as_spf_words(basis).reshape(bb.shape),
            "table_words": _as_spf_words(multiplication_table).reshape(table.shape),
            "basis_count": int(cc.size),
            "matrix_size": n,
            "exact_table": bool(exact_table),
        }
        block = _structural_view_block(
            (n, n),
            "nonabelian_coefficient_table",
            {
                "basis_count": int(cc.size),
                "matrix_size": n,
                "exact_table": bool(exact_table),
                "coords_shape": tuple(int(x) for x in cc.shape),
                "basis_shape": tuple(int(x) for x in bb.shape),
                "table_shape": tuple(int(x) for x in table.shape),
            },
        )
        return cls(
            SPFBlockExpr(
                "NonAbelianCoefficientTable",
                args=(params,),
                block=block,
                proof=proof or SPFProof("nonabelian_coefficient_table_view", bool(exact_table), True, 0, 0),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=("algebra",),
                basis_out=("algebra",),
                exactness="exact" if exact_table else "quantized",
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def low_rank(
        cls,
        u: np.ndarray,
        v: np.ndarray,
        coeff: np.ndarray | None = None,
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        uu = _as_complex_values(u)
        vv = _as_complex_values(v)
        if uu.ndim != 2 or vv.ndim != 2:
            raise SPFError("low_rank factors must be rank-2 arrays")
        if uu.shape[1] != vv.shape[1]:
            raise SPFError("low_rank factors must have the same rank")
        cc = np.ones(uu.shape[1], dtype=np.complex128) if coeff is None else _as_complex_values(coeff)
        if cc.shape != (uu.shape[1],):
            raise SPFError("low_rank coeff must have shape (rank,)")
        params = {
            "u_words": _as_spf_words(u).reshape(uu.shape),
            "v_words": _as_spf_words(v).reshape(vv.shape),
            "coeff_words": encode_words(cc) if coeff is None else _as_spf_words(coeff).reshape(cc.shape),
        }
        out = AxisSpec(out_axis, int(uu.shape[0]), role="codomain")
        inn = AxisSpec(in_axis, int(vv.shape[0]), role="domain")
        block = _structural_view_block(
            (out.size, inn.size),
            "low_rank_outer",
            {
                "rank": int(uu.shape[1]),
                "u_shape": tuple(int(x) for x in uu.shape),
                "v_shape": tuple(int(x) for x in vv.shape),
                "coeff_shape": tuple(int(x) for x in cc.shape),
            },
        )
        return cls(
            SPFBlockExpr(
                "LowRankOuter",
                args=(params,),
                block=block,
                proof=proof or SPFProof("low_rank_view", False, False, 0, 0),
            ),
            (out, inn),
            KernelSignature((out.name,), (inn.name,), reducer_policy=reducer_policy),
        )

    @classmethod
    def residual_patch(
        cls,
        base: SPFKernel,
        patch: SPFKernel,
        *,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        if not isinstance(base, SPFKernel):
            raise SPFError("residual_patch base must be an SPFKernel")
        if not isinstance(patch, SPFKernel):
            raise SPFError("residual_patch patch must be an SPFKernel")
        if len(base.sig.out_axes) != 1 or len(base.sig.in_axes) != 1:
            raise SPFError("residual_patch currently requires rank-2 base kernels")
        if len(patch.sig.out_axes) != 1 or len(patch.sig.in_axes) != 1:
            raise SPFError("residual_patch currently requires rank-2 patch kernels")
        _validate_sum_compatible(base, patch)
        out = _axis_map(base.axes)[base.sig.out_axes[0]]
        inn = _axis_map(base.axes)[base.sig.in_axes[0]]
        reducer = reducer_policy or base.sig.reducer_policy or patch.sig.reducer_policy or _default_reducer()
        reducer_count = int(out.size * inn.size)
        return cls(
            SPFBlockExpr(
                "ResidualPatch",
                args=({"base": base, "patch": patch, "reducer": reducer},),
                proof=proof
                or SPFProof(
                    "residual_patch_reducer_boundary",
                    False,
                    False,
                    reducer_count,
                    0,
                    notes=("base plus residual patch is an explicit additive boundary",),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=base.sig.basis_in,
                basis_out=base.sig.basis_out,
                reducer_policy=reducer,
            ),
        )

    @classmethod
    def jacobian_operator(
        cls,
        matrix: np.ndarray,
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        provenance: Mapping[str, Any] | None = None,
        residual: np.ndarray | None = None,
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        dense = _as_complex_values(matrix)
        if dense.ndim != 2:
            raise SPFError("jacobian_operator matrix must be rank-2")
        residual_words = None
        if residual is not None:
            rr = _as_complex_values(residual)
            if rr.shape != dense.shape:
                raise SPFError("jacobian residual shape must match matrix")
            residual_words = _as_spf_words(residual).reshape(rr.shape)
        out = AxisSpec(out_axis, int(dense.shape[0]), role="codomain", basis="learned")
        inn = AxisSpec(in_axis, int(dense.shape[1]), role="domain", basis="learned")
        params = {
            "matrix_words": _as_spf_words(matrix).reshape(dense.shape),
            "provenance": dict(provenance or {}),
            "residual_words": residual_words,
        }
        block = _structural_view_block(
            dense.shape,
            "jacobian_operator",
            {
                "matrix_shape": tuple(int(x) for x in dense.shape),
                "has_residual": residual_words is not None,
                "provenance_keys": tuple(sorted(str(k) for k in dict(provenance or {}))),
            },
        )
        return cls(
            SPFBlockExpr(
                "JacobianOperator",
                args=(params,),
                block=block,
                proof=proof
                or SPFProof(
                    "jacobian_operator_view",
                    False,
                    False,
                    0,
                    0,
                    notes=tuple(f"{k}={v}" for k, v in sorted(dict(provenance or {}).items())),
                ),
            ),
            (out, inn),
            KernelSignature(
                (out.name,), (inn.name,), basis_in=("learned",), basis_out=("learned",), reducer_policy=reducer_policy
            ),
        )

    @classmethod
    def iterated_operator(
        cls,
        operator: SPFKernel,
        *,
        reducer_policy: ReducerSpec | None = None,
    ) -> SPFBlockMatrix:
        if not isinstance(operator, SPFKernel):
            raise SPFError("iterated_operator generator must be an SPFKernel")
        if len(operator.sig.out_axes) != 1 or len(operator.sig.in_axes) != 1:
            raise SPFError("iterated_operator requires a rank-2 generator")
        amap = _axis_map(operator.axes)
        out = amap[operator.sig.out_axes[0]]
        inn = amap[operator.sig.in_axes[0]]
        if out.size != inn.size:
            raise SPFError("iterated_operator generator must be square")
        block = _structural_view_block(
            (out.size, inn.size),
            "iterated_operator",
            {
                "operator_op": str(operator.expr.op),
                "operator_law": None if operator.proof is None else str(operator.proof.law),
                "size": int(out.size),
            },
        )
        return cls(
            SPFBlockExpr(
                "IteratedOperator",
                args=({"operator": operator},),
                block=block,
                proof=SPFProof("iterated_operator_view", False, True, operator.reducer_budget(), 0),
            ),
            (out, inn),
            KernelSignature(
                (out.name,),
                (inn.name,),
                basis_in=operator.sig.basis_in,
                basis_out=operator.sig.basis_out,
                reducer_policy=reducer_policy or operator.sig.reducer_policy,
            ),
        )

    @classmethod
    def tt_structured(
        cls,
        cores: Sequence[np.ndarray],
        *,
        out_axis: str = "o",
        in_axis: str = "i",
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockMatrix:
        core_words, out_size, in_size = _validate_and_encode_tt_cores(cores)
        out = AxisSpec(out_axis, out_size, role="codomain")
        inn = AxisSpec(in_axis, in_size, role="domain")
        core_shapes = tuple(tuple(int(x) for x in core.shape) for core in core_words)
        block = _structural_view_block(
            (out_size, in_size),
            "tt_structured",
            {
                "core_count": len(core_words),
                "core_shapes": core_shapes,
                "out_size": out_size,
                "in_size": in_size,
            },
        )
        return cls(
            SPFBlockExpr(
                "TTStructured",
                args=({"core_words": core_words},),
                block=block,
                proof=proof
                or SPFProof(
                    "tt_structured_view",
                    False,
                    True,
                    0,
                    0,
                    notes=(f"core_count={len(core_words)}",),
                ),
            ),
            (out, inn),
            KernelSignature((out.name,), (inn.name,), reducer_policy=reducer_policy),
        )

    def inverse(self) -> SPFKernel:
        from .algebra import inverse_with_registry

        out = inverse_with_registry(self)
        if out is not None:
            return out
        raise SPFError(f"no inverse law registered for {self.expr.op}")

    def power(self, k: int) -> SPFKernel:
        from .algebra import power_with_registry

        out = power_with_registry(self, int(k))
        if out is not None:
            return out
        raise SPFError(f"no power law registered for {self.expr.op}")


class SPFBlockVector(SPFKernel):
    @classmethod
    def from_words(
        cls,
        words: np.ndarray,
        *,
        axis: str = "i",
        role: str = "codomain",
        basis: str = "standard",
        batch_axes: Sequence[str | AxisSpec] = (),
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockVector:
        if role not in {"codomain", "domain"}:
            raise SPFError("SPFBlockVector role must be 'codomain' or 'domain'")
        arr = np.asarray(words, dtype=np.uint32)
        if arr.ndim != len(batch_axes) + 1:
            raise SPFError("vector word array rank must match batch_axes + one vector axis")
        batch_specs = _axis_specs(batch_axes, arr.shape[: len(batch_axes)], "batch")
        vector_axis = AxisSpec(axis, int(arr.shape[-1]), role=role, basis=basis)
        out_axes = (vector_axis,) if role == "codomain" else ()
        in_axes = (vector_axis,) if role == "domain" else ()
        return cls(
            SPFBlockExpr(
                "MaterializedWords",
                args=(arr.copy(),),
                proof=proof
                or SPFProof(
                    "materialized_vector",
                    False,
                    False,
                    0,
                    0,
                    notes=("quantized_realization",),
                ),
            ),
            (*batch_specs, vector_axis),
            KernelSignature(
                _axis_names(out_axes),
                _axis_names(in_axes),
                batch_axes=_axis_names(batch_specs),
                basis_in=tuple(axis.basis for axis in in_axes),
                basis_out=tuple(axis.basis for axis in out_axes),
                reducer_policy=reducer_policy,
            ),
        )

    @classmethod
    def dense(
        cls,
        values: np.ndarray,
        *,
        axis: str = "i",
        role: str = "codomain",
        basis: str = "standard",
        batch_axes: Sequence[str | AxisSpec] = (),
        reducer_policy: ReducerSpec | None = None,
        proof: SPFProof | None = None,
    ) -> SPFBlockVector:
        return cls.from_words(
            _as_spf_words(values),
            axis=axis,
            role=role,
            basis=basis,
            batch_axes=batch_axes,
            reducer_policy=reducer_policy,
            proof=proof,
        )


def lazy_sum(terms: Sequence[SPFKernel], reducer: ReducerSpec | None = None) -> SPFKernel:
    flat: list[SPFKernel] = []
    for term in terms:
        if not isinstance(term, SPFKernel):
            raise SPFError("LazySum terms must be SPFKernel instances")
        if term.expr.op == "LazySum":
            flat.extend(term.expr.args[0])
        else:
            flat.append(term)
    if not flat:
        raise SPFError("LazySum requires at least one term")
    first = flat[0]
    for term in flat[1:]:
        _validate_sum_compatible(first, term)
    batch_axes = tuple(_axis_map(first.axes)[name] for name in first.sig.batch_axes)
    out_axes = tuple(_axis_map(first.axes)[name] for name in first.sig.out_axes)
    in_axes = tuple(_axis_map(first.axes)[name] for name in first.sig.in_axes)
    shape = _shape_for(first.sig, first.axes)
    reducer_spec = reducer or first.sig.reducer_policy or ReducerSpec(
        policy=ReducerPolicy.AUTO, routing=RoutingPolicy.PINNED, domain=AccumulationDomain.LOGPOLAR,
    )
    reducer_count = int(np.prod(shape, dtype=np.int64)) if len(flat) > 1 else 0
    return SPFKernel(
        SPFBlockExpr(
            "LazySum",
            args=(tuple(flat), reducer_spec),
            proof=SPFProof(
                "lazy_sum_reducer_boundary",
                False,
                False,
                reducer_count,
                0,
                notes=(f"term_count={len(flat)}",),
            ),
        ),
        batch_axes + out_axes + in_axes,
        KernelSignature(
            tuple(a.name for a in out_axes),
            tuple(a.name for a in in_axes),
            batch_axes=tuple(a.name for a in batch_axes),
            basis_in=first.sig.basis_in,
            basis_out=first.sig.basis_out,
            reducer_policy=reducer_spec,
        ),
    )


def lazy_hadamard(a: SPFKernel, b: SPFKernel, *, allow_fallback: bool = True) -> SPFKernel:
    if not isinstance(a, SPFKernel) or not isinstance(b, SPFKernel):
        raise SPFError("LazyHadamard operands must be SPFKernel instances")
    _validate_sum_compatible(a, b)
    closed = _closed_hadamard(a, b)
    if closed is not None:
        return closed
    if not bool(allow_fallback):
        raise SPFError(f"no metadata-only Hadamard law for {a.expr.op} and {b.expr.op}")
    batch_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes)
    out_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes)
    in_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.in_axes)
    shape = _shape_for(a.sig, a.axes)
    return SPFKernel(
        SPFBlockExpr(
            "LazyHadamard",
            args=(a, b),
            proof=SPFProof(
                "elementwise_mul_fallback",
                False,
                False,
                0,
                0,
                notes=(f"elementwise_terms={int(np.prod(shape, dtype=np.int64))}",),
            ),
        ),
        batch_axes + out_axes + in_axes,
        KernelSignature(
            tuple(axis.name for axis in out_axes),
            tuple(axis.name for axis in in_axes),
            batch_axes=tuple(axis.name for axis in batch_axes),
            basis_in=a.sig.basis_in,
            basis_out=a.sig.basis_out,
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        ),
    )


def _closed_hadamard(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "Block" and b.expr.op == "Block":
        closed_block = _closed_block_hadamard(a, b)
        if closed_block is not None:
            return closed_block
    if a.expr.op == "Character" and b.expr.op == "Character":
        pa = _character_params(a)
        pb = _character_params(b)
        if int(pa["N"]) != int(pb["N"]):
            raise SPFError("character Hadamard bind requires matching moduli")
        if bool(pa.get("exact_group_semantics")) != bool(pb.get("exact_group_semantics")):
            return None
        n = int(pa["N"])
        return SPFBlockMatrix.character(
            N=n,
            alpha=int(pa["alpha"]) + int(pb["alpha"]),
            beta=int(pa["beta"]) + int(pb["beta"]),
            out_axis=a.sig.out_axes[0],
            in_axis=a.sig.in_axes[0],
            scale=complex(pa.get("scale", 1.0 + 0.0j)) * complex(pb.get("scale", 1.0 + 0.0j)),
            exact_group_semantics=bool(pa.get("exact_group_semantics")) and bool(pb.get("exact_group_semantics")),
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "ProductGroupCharacter" and b.expr.op == "ProductGroupCharacter":
        pa = _character_params(a)
        pb = _character_params(b)
        moduli = tuple(int(m) for m in pa["moduli"])
        if moduli != tuple(int(m) for m in pb["moduli"]):
            raise SPFError("product-character Hadamard bind requires matching moduli")
        if bool(pa.get("exact_group_semantics")) != bool(pb.get("exact_group_semantics")):
            return None
        return SPFBlockMatrix.product_character(
            moduli=moduli,
            out_frequency=tuple(int(x) + int(y) for x, y in zip(pa["out_frequency"], pb["out_frequency"], strict=True)),
            in_frequency=tuple(int(x) + int(y) for x, y in zip(pa["in_frequency"], pb["in_frequency"], strict=True)),
            out_axis=a.sig.out_axes[0],
            in_axis=a.sig.in_axes[0],
            scale=complex(pa.get("scale", 1.0 + 0.0j)) * complex(pb.get("scale", 1.0 + 0.0j)),
            exact_group_semantics=bool(pa.get("exact_group_semantics")) and bool(pb.get("exact_group_semantics")),
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "DiagonalSPF" and b.expr.op == "DiagonalSPF":
        if a.sig.exactness != b.sig.exactness:
            return None
        wa = np.asarray(a.expr.args[0]["diag_words"], dtype=np.uint32)
        wb = np.asarray(b.expr.args[0]["diag_words"], dtype=np.uint32)
        if _spf32_mul_l_saturates(wa, wb):
            return None
        out_words = spf_mul_words(wa, wb)
        return SPFBlockMatrix.diagonal(
            out_words,
            out_axis=a.sig.out_axes[0],
            axis=a.sig.in_axes[0],
            exactness=a.sig.exactness,
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
            proof=SPFProof(
                "diagonal_hadamard_metadata",
                a.sig.exactness == "exact",
                True,
                0,
                0,
                notes=() if a.sig.exactness == "exact" else ("quantized_realization",),
            ),
        )
    if a.expr.op == "MaterializedWords" and b.expr.op == "MaterializedWords":
        wa = np.asarray(a.expr.args[0], dtype=np.uint32)
        wb = np.asarray(b.expr.args[0], dtype=np.uint32)
        if _spf32_mul_l_saturates(wa, wb):
            return None
        batch_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes)
        out_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes)
        in_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.in_axes)
        return SPFBlockMatrix.from_words(
            spf_mul_words(wa, wb),
            out_axes=out_axes,
            in_axes=in_axes,
            batch_axes=batch_axes,
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
            proof=SPFProof("word_hadamard_bind", True, False, 0, 0),
        )
    return None


def _closed_block_hadamard(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    ba = a.expr.block
    bb = b.expr.block
    if ba is None or bb is None:
        return None
    if a.sig.exactness != b.sig.exactness:
        return None
    if ba.dims != bb.dims:
        raise SPFError("block Hadamard bind requires matching dimensions")
    if ba.origin != bb.origin:
        return None
    if ba.representation.rep_id != bb.representation.rep_id:
        return None
    if ba.residual.codec_id != int(ResidualCodecKind.NONE) or bb.residual.codec_id != int(ResidualCodecKind.NONE):
        return None
    if ba.outliers or bb.outliers:
        return None
    if int(ba.base.profile_id) != int(bb.base.profile_id):
        return None
    if int(ba.base.profile_id) not in {
        int(BaseModelKind.STB_AFFINE_POLY),
        int(BaseModelKind.SEPARABLE_PHASE),
        int(BaseModelKind.SEPARABLE_LOGMAG),
    }:
        return None
    if _spf32_mul_l_saturates(ba.to_words(), bb.to_words()):
        return None

    if int(ba.base.profile_id) == int(BaseModelKind.STB_AFFINE_POLY):
        params = _compose_stb_affine_poly_params(ba.base.params, bb.base.params)
        if params is None:
            return None
        law = "stb_affine_poly_hadamard_bind"
        kind = BaseModelKind.STB_AFFINE_POLY
    elif int(ba.base.profile_id) == int(BaseModelKind.SEPARABLE_PHASE):
        from .base_models import compose_separable_phase

        params = compose_separable_phase(ba.base.params, bb.base.params)
        law = "separable_phase_hadamard_bind"
        kind = BaseModelKind.SEPARABLE_PHASE
    else:
        params = _compose_separable_logmag_params(ba.base.params, bb.base.params)
        law = "separable_logmag_hadamard_bind"
        kind = BaseModelKind.SEPARABLE_LOGMAG
    block = SPFBlockV3(
        dims=ba.dims,
        origin=ba.origin,
        representation=RepresentationRef(ba.representation.rep_id),
        base=BaseModelRef(int(kind), params),
        residual=ResidualRef(int(ResidualCodecKind.NONE), None),
    )
    batch_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes)
    out_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes)
    in_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.in_axes)
    return SPFBlockMatrix.from_block(
        block,
        batch_axes=batch_axes,
        out_axes=out_axes,
        in_axes=in_axes,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        exactness=a.sig.exactness,
        proof=SPFProof(
            law,
            a.sig.exactness == "exact",
            True,
            0,
            0,
            notes=() if a.sig.exactness == "exact" else ("quantized_realization",),
        ),
    )


def _compose_stb_affine_poly_params(params_a: Mapping[str, Any], params_b: Mapping[str, Any]) -> dict[str, Any] | None:
    header_a = params_a.get("header")
    header_b = params_b.get("header")
    if not isinstance(header_a, TensorBlockHeaderV2) or not isinstance(header_b, TensorBlockHeaderV2):
        return None
    if tuple(header_a.dims) != tuple(header_b.dims):
        raise SPFError("STB affine Hadamard bind requires matching dimensions")
    if int(header_a.order or 0) != int(header_b.order or 0):
        raise SPFError("STB affine Hadamard bind requires matching order")
    if header_a.scalar_format_id != header_b.scalar_format_id:
        return None
    if header_a.active_encoding != header_b.active_encoding:
        return None
    if tuple(header_a.origin or ()) != tuple(header_b.origin or ()):
        return None
    active_a = np.asarray(params_a.get("active_indices", ()), dtype=np.int64)
    active_b = np.asarray(params_b.get("active_indices", ()), dtype=np.int64)
    if not np.array_equal(active_a, active_b):
        return None
    if tuple(params_a.get("bands", ())) or tuple(params_b.get("bands", ())):
        return None
    order = int(header_a.order or 0)
    try:
        header = replace(
            header_a,
            global_L_bias=int(header_a.global_L_bias) + int(header_b.global_L_bias),
            global_T_bias=(int(header_a.global_T_bias) + int(header_b.global_T_bias)) & SPF32_PHASE_MASK,
            coarse_exp=int(header_a.coarse_exp) + int(header_b.coarse_exp),
            # band_exp is int16 in the header (checked on construction since
            # 2026-09-18); an overflowing sum raises SPFError and the bind is declined.
            band_exp=tuple(int(x) + int(y) for x, y in zip(header_a.band_exp, header_b.band_exp, strict=True)),
            ramp_R=tuple(
                tuple(
                    (int(header_a.ramp_R[row][col]) + int(header_b.ramp_R[row][col])) & SPF32_PHASE_MASK
                    for col in range(order)
                )
                for row in range(order)
            ),
            ramp_b=tuple(
                (int(x) + int(y)) & SPF32_PHASE_MASK for x, y in zip(header_a.ramp_b, header_b.ramp_b, strict=True)
            ),
            quadratic_ramp=tuple(
                (int(x) + int(y)) & SPF32_PHASE_MASK
                for x, y in zip(header_a.quadratic_ramp, header_b.quadratic_ramp, strict=True)
            ),
            cubic_ramp=tuple(
                (int(x) + int(y)) & SPF32_PHASE_MASK
                for x, y in zip(header_a.cubic_ramp, header_b.cubic_ramp, strict=True)
            ),
            phase_gradient=tuple(
                (int(x) + int(y)) & SPF32_PHASE_MASK
                for x, y in zip(header_a.phase_gradient, header_b.phase_gradient, strict=True)
            ),
            mixed_ramp=tuple(
                (int(x) + int(y)) & SPF32_PHASE_MASK
                for x, y in zip(header_a.mixed_ramp, header_b.mixed_ramp, strict=True)
            ),
            # §13.6 mixed cubics add like every other phase coefficient: a
            # Hadamard bind sums the phase polynomials term by term.
            cubic_mixed=tuple(
                (int(x) + int(y)) & SPF32_PHASE_MASK
                for x, y in zip(header_a.cubic_mixed, header_b.cubic_mixed, strict=True)
            ),
            # Log-magnitude terms add on a Hadamard bind (L planes sum); int16
            # overflow raises SPFError below and the bind is declined.
            log_quadratic=tuple(
                int(x) + int(y) for x, y in zip(header_a.log_quadratic, header_b.log_quadratic, strict=True)
            ),
            log_mixed=tuple(int(x) + int(y) for x, y in zip(header_a.log_mixed, header_b.log_mixed, strict=True)),
        )
    except SPFError:
        return None
    return {"header": header, "bands": (), "active_indices": active_a}


def _compose_separable_logmag_params(params_a: Mapping[str, Any], params_b: Mapping[str, Any]) -> dict[str, Any]:
    for src in (params_a, params_b):
        if src.get("phase_terms") or src.get("phase_bias"):
            raise SPFError(
                "compose_separable_logmag: cannot compose a law carrying a phase term/bias "
                "(log-magnitude Hadamard product is logmag-only)"
            )
    result: dict[str, Any] = {
        "logmag_terms": list(params_a.get("logmag_terms", ())) + list(params_b.get("logmag_terms", ()))
    }
    bias_a = params_a.get("logmag_bias")
    bias_b = params_b.get("logmag_bias")
    if bias_a is not None and bias_b is not None:
        va = np.asarray(bias_a["vec"], dtype=np.int64)
        vb = np.asarray(bias_b["vec"], dtype=np.int64)
        if int(bias_a["axis"]) != int(bias_b["axis"]) or va.shape != vb.shape:
            raise SPFError("compose_separable_logmag: logmag_bias axis/length mismatch")
        result["logmag_bias"] = {"axis": int(bias_a["axis"]), "vec": va + vb}
    elif bias_a is not None:
        result["logmag_bias"] = {"axis": int(bias_a["axis"]), "vec": np.asarray(bias_a["vec"], dtype=np.int64)}
    elif bias_b is not None:
        result["logmag_bias"] = {"axis": int(bias_b["axis"]), "vec": np.asarray(bias_b["vec"], dtype=np.int64)}
    return result


def _spf32_mul_l_saturates(a_words: np.ndarray, b_words: np.ndarray) -> bool:
    a_l = np.asarray(unpack_L(a_words), dtype=np.int64)
    b_l = np.asarray(unpack_L(b_words), dtype=np.int64)
    a_b, b_b = np.broadcast_arrays(a_l, b_l)
    finite = (a_b != SPF32_L_ZERO) & (b_b != SPF32_L_ZERO)
    if not np.any(finite):
        return False
    summed = a_b[finite] + b_b[finite]
    return bool(np.any((summed < SPF32_L_MIN_FINITE) | (summed > SPF32_L_MAX_FINITE)))


def _validate_sum_compatible(a: SPFKernel, b: SPFKernel) -> None:
    if a.sig.batch_axes != b.sig.batch_axes or a.sig.out_axes != b.sig.out_axes or a.sig.in_axes != b.sig.in_axes:
        raise SPFError("LazySum terms must have identical batch/input/output axis order")
    amap = _axis_map(a.axes)
    bmap = _axis_map(b.axes)
    for name in a.sig.batch_axes + a.sig.out_axes + a.sig.in_axes:
        left = amap[name]
        right = bmap[name]
        if left.size != right.size:
            raise SPFError(f"LazySum axis size mismatch on {name!r}: {left.size} != {right.size}")
        if left.basis != right.basis:
            raise SPFError(f"LazySum axis basis mismatch on {name!r}: {left.basis!r} != {right.basis!r}")


def _parse_einsum_labels(term: str) -> tuple[str, ...]:
    if "[" not in term and "]" not in term:
        return tuple(term)
    matches = tuple(_EINSUM_BRACKET_LABEL_RE.finditer(term))
    if not matches or "".join(match.group(0) for match in matches) != term:
        raise SPFError("bracketed einsum labels must cover the entire term")
    return tuple(match.group(1) for match in matches)


def _format_einsum_labels(labels: Sequence[str]) -> str:
    return "".join(f"[{label}]" for label in labels)


def _split_einsum_operand_labels(
    operand: SPFKernel,
    labels: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], bool]:
    labels = tuple(str(label) for label in labels)
    batch_rank = len(operand.sig.batch_axes)
    out_rank = len(operand.sig.out_axes)
    in_rank = len(operand.sig.in_axes)
    operator_rank = out_rank + in_rank
    full_rank = batch_rank + operator_rank
    if len(labels) == full_rank:
        return (
            labels[:batch_rank],
            labels[batch_rank : batch_rank + out_rank],
            labels[batch_rank + out_rank :],
            True,
        )
    if len(labels) == operator_rank:
        return (), labels[:out_rank], labels[out_rank:], False
    raise SPFError("einsum term rank must match operand rank")


def _relabel_kernel_axes(
    kernel: SPFKernel,
    *,
    batch_axes: Sequence[str],
    out_axes: Sequence[str],
    in_axes: Sequence[str],
) -> SPFKernel:
    batch_names = tuple(str(axis) for axis in batch_axes)
    out_names = tuple(str(axis) for axis in out_axes)
    in_names = tuple(str(axis) for axis in in_axes)
    old_names = kernel.sig.batch_axes + kernel.sig.out_axes + kernel.sig.in_axes
    new_names = batch_names + out_names + in_names
    if len(old_names) != len(new_names):
        raise SPFError("einsum relabel rank mismatch")
    if len(set(new_names)) != len(new_names):
        raise SPFError("einsum relabel axes must be unique")
    if new_names == old_names:
        return kernel
    old_axes = _axis_map(kernel.axes)
    axes = tuple(
        AxisSpec(
            new_name,
            old_axes[old_name].size,
            role=old_axes[old_name].role,
            basis=old_axes[old_name].basis,
            group=old_axes[old_name].group,
            metadata=old_axes[old_name].metadata,
        )
        for old_name, new_name in zip(old_names, new_names, strict=True)
    )
    return SPFKernel(
        kernel.expr,
        axes,
        KernelSignature(
            out_names,
            in_names,
            batch_axes=batch_names,
            basis_in=kernel.sig.basis_in,
            basis_out=kernel.sig.basis_out,
            exactness=kernel.sig.exactness,
            reducer_policy=kernel.sig.reducer_policy,
        ),
    )


def spfblock_einsum(spec: str, *operands: SPFKernel, reducer: ReducerSpec | None = None) -> SPFKernel:
    if not isinstance(spec, str):
        raise SPFError("einsum spec must be a string")
    lhs, arrow, rhs = spec.replace(" ", "").partition("->")
    if arrow != "->" or not rhs:
        raise SPFError("einsum spec must contain '->'")
    term_strings = lhs.split(",")
    if len(term_strings) != len(operands):
        raise SPFError("einsum operand count mismatch")
    if any(not isinstance(operand, SPFKernel) for operand in operands):
        raise SPFError("einsum operands must be SPFKernel instances")
    terms = tuple(_parse_einsum_labels(term) for term in term_strings)
    rhs_labels = _parse_einsum_labels(rhs)
    if len(operands) == 2:
        a, b = operands
        a_labels = terms[0]
        b_labels = terms[1]
        if len(set(a_labels)) != len(a_labels) or len(set(b_labels)) != len(b_labels):
            raise SPFError("einsum input terms must not contain duplicate axes")
        if len(set(rhs_labels)) != len(rhs_labels):
            raise SPFError("einsum output term must not contain duplicate axes")
        a_batch_labels, a_out_labels, a_in_labels, a_has_batch_labels = _split_einsum_operand_labels(a, a_labels)
        b_batch_labels, b_out_labels, b_in_labels, b_has_batch_labels = _split_einsum_operand_labels(b, b_labels)
        if a.sig.batch_axes or b.sig.batch_axes:
            if not (a_has_batch_labels and b_has_batch_labels and a_batch_labels and b_batch_labels):
                raise SPFError("batched einsum requires explicit batch axis labels")
            if a_batch_labels != b_batch_labels:
                raise SPFError("batched einsum operands must use identical batch axis labels")
            if rhs_labels[: len(a_batch_labels)] != a_batch_labels:
                raise SPFError("batched einsum output must preserve batch axis labels as leading axes")
        batch_labels_match = a_batch_labels == b_batch_labels and a_batch_labels != ()
        if (
            len(a_in_labels) == len(b_out_labels)
            and a_in_labels == b_out_labels
            and rhs_labels
            in {
                a_out_labels + b_in_labels if not (a.sig.batch_axes or b.sig.batch_axes) else (),
                a_batch_labels + a_out_labels + b_in_labels if batch_labels_match else (),
            }
        ):
            relabeled_a = _relabel_kernel_axes(
                a,
                batch_axes=a_batch_labels,
                out_axes=a_out_labels,
                in_axes=a_in_labels,
            )
            relabeled_b = _relabel_kernel_axes(
                b,
                batch_axes=b_batch_labels,
                out_axes=b_out_labels,
                in_axes=b_in_labels,
            )
            return _relabel_kernel_axes(
                relabeled_a.compose(relabeled_b, reducer=reducer),
                batch_axes=a_batch_labels if batch_labels_match else (),
                out_axes=a_out_labels,
                in_axes=b_in_labels,
            )
        if terms[0] == terms[1] == rhs_labels:
            relabeled_a = _relabel_kernel_axes(
                a,
                batch_axes=a_batch_labels,
                out_axes=a_out_labels,
                in_axes=a_in_labels,
            )
            relabeled_b = _relabel_kernel_axes(
                b,
                batch_axes=b_batch_labels,
                out_axes=b_out_labels,
                in_axes=b_in_labels,
            )
            return relabeled_a.hadamard(relabeled_b)
        return _fallback_einsum_contract(a, b, terms[0], terms[1], rhs_labels, reducer)
    raise SPFError("only two-operand SPFBlockMatrix einsum is implemented")


def _fallback_einsum_contract(
    a: SPFKernel,
    b: SPFKernel,
    a_term: Sequence[str],
    b_term: Sequence[str],
    out_term: Sequence[str],
    reducer: ReducerSpec | None,
) -> SPFKernel:
    a_labels = tuple(str(label) for label in a_term)
    b_labels = tuple(str(label) for label in b_term)
    out_labels = tuple(str(label) for label in out_term)
    if len(set(a_labels)) != len(a_labels) or len(set(b_labels)) != len(b_labels):
        raise SPFError("einsum input terms must not contain duplicate axes")
    if len(set(out_labels)) != len(out_labels):
        raise SPFError("einsum output term must not contain duplicate axes")

    a_axis_names = a.sig.batch_axes + a.sig.out_axes + a.sig.in_axes
    b_axis_names = b.sig.batch_axes + b.sig.out_axes + b.sig.in_axes
    if len(a_labels) != len(a_axis_names) or len(b_labels) != len(b_axis_names):
        raise SPFError("einsum term rank must match operand rank")

    a_axes = tuple(_axis_map(a.axes)[name] for name in a_axis_names)
    b_axes = tuple(_axis_map(b.axes)[name] for name in b_axis_names)
    label_sources: dict[str, list[AxisSpec]] = {}
    for label, axis in zip(a_labels, a_axes, strict=True):
        label_sources.setdefault(label, []).append(axis)
    for label, axis in zip(b_labels, b_axes, strict=True):
        label_sources.setdefault(label, []).append(axis)

    output_label_set = set(out_labels)
    missing = output_label_set - set(label_sources)
    if missing:
        raise SPFError(f"einsum output references missing axes {sorted(missing)!r}")

    for label, axes in label_sources.items():
        if len(axes) == 2:
            if axes[0].size != axes[1].size:
                raise SPFError(f"einsum contracted axis size mismatch on {label!r}: {axes[0].size} != {axes[1].size}")
            if axes[0].basis != axes[1].basis:
                raise SPFError(
                    f"einsum contracted axis basis mismatch on {label!r}: {axes[0].basis!r} != {axes[1].basis!r}"
                )
            shared_batch_output = axes[0].role == axes[1].role == "batch" and label in output_label_set
            if label in output_label_set and not shared_batch_output:
                raise SPFError("einsum shared output axes require an explicit Hadamard bind")
        elif len(axes) == 1:
            if label not in output_label_set:
                raise SPFError("einsum cannot reduce an axis that appears in only one operand")
        else:
            raise SPFError("einsum supports at most two operands per axis label")

    b_label_set = set(b_labels)
    contract_labels = tuple(label for label in a_labels if label in b_label_set and label not in output_label_set)
    if not contract_labels:
        raise SPFError("einsum fallback requires at least one contracted axis")

    default_label_list = [label for label in a_labels if label not in contract_labels]
    default_label_list.extend(
        label for label in b_labels if label not in contract_labels and label not in set(default_label_list)
    )
    default_labels = tuple(default_label_list)
    if set(default_labels) != output_label_set or len(default_labels) != len(out_labels):
        raise SPFError("einsum output must contain every non-contracted axis exactly once")
    axis_by_output: dict[str, AxisSpec] = {}
    for label, axis in zip(a_labels, a_axes, strict=True):
        if label in output_label_set:
            axis_by_output[label] = axis
    for label, axis in zip(b_labels, b_axes, strict=True):
        if label in output_label_set:
            axis_by_output[label] = axis

    axis_by_label: dict[str, AxisSpec] = {}
    for label, axes in label_sources.items():
        source_axis = axes[0]
        axis_by_label[label] = AxisSpec(
            label,
            source_axis.size,
            role="contract" if label in contract_labels else "codomain",
            basis=source_axis.basis,
            group=source_axis.group,
            metadata=source_axis.metadata,
        )
    batch_output_labels = tuple(
        label
        for label in out_labels
        if label in label_sources and all(axis.role == "batch" for axis in label_sources[label])
    )
    operator_output_labels = tuple(label for label in out_labels if label not in set(batch_output_labels))
    shared_labels = out_labels + contract_labels
    shared_axes = tuple(axis_by_label[label] for label in shared_labels)
    out_axes = tuple(
        AxisSpec(
            label,
            axis_by_output[label].size,
            role="codomain",
            basis=axis_by_output[label].basis,
            group=axis_by_output[label].group,
            metadata=axis_by_output[label].metadata,
        )
        for label in operator_output_labels
    )
    reducer_spec = reducer or a.sig.reducer_policy or b.sig.reducer_policy or _default_reducer()
    a_axis_names = a.sig.batch_axes + a.sig.out_axes + a.sig.in_axes
    b_axis_names = b.sig.batch_axes + b.sig.out_axes + b.sig.in_axes
    left = a.pullback(
        {name: label for name, label in zip(a_axis_names, a_labels, strict=True)},
        axes=shared_axes,
        batch_axes=batch_output_labels,
        out_axes=tuple(label for label in shared_labels if label not in set(batch_output_labels)),
        in_axes=(),
    )
    right = b.pullback(
        {name: label for name, label in zip(b_axis_names, b_labels, strict=True)},
        axes=shared_axes,
        batch_axes=batch_output_labels,
        out_axes=tuple(label for label in shared_labels if label not in set(batch_output_labels)),
        in_axes=(),
    )
    return left.bind(right).pushforward(
        contract_labels,
        out_axes=tuple(axis.name for axis in out_axes),
        in_axes=(),
        batch_axes=batch_output_labels,
        reducer=reducer_spec,
    )


def _structural_vector_apply(operator: SPFKernel, vector: SPFKernel) -> SPFKernel | None:
    """Lower an operator at its vector consumer without realizing its matrix."""
    if operator.sig.batch_axes or vector.sig.in_axes or len(vector.sig.out_axes) != 1:
        return None
    if len(operator.sig.out_axes) != 1 or len(operator.sig.in_axes) != 1:
        return None
    op = operator.expr.op
    if op not in {"DiagonalSPF", "Permutation", "Roll", "ClosedFormIdentity"}:
        return None
    out = _axis_map(operator.axes)[operator.sig.out_axes[0]]
    params: dict[str, Any] = {"operator": operator, "vector": vector}
    proof = SPFProof(f"{op.lower()}_vector_apply", op != "DiagonalSPF", True, 0, 0)
    return SPFKernel(
        SPFBlockExpr("StructuralApply", args=(params,), proof=proof),
        (out,),
        KernelSignature((out.name,), (), exactness="quantized", reducer_policy=vector.sig.reducer_policy),
    )


def _fallback_compose(a: SPFKernel, b: SPFKernel, reducer: ReducerSpec) -> SPFKernel:
    batch_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes)
    out_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes)
    in_axes = tuple(_axis_map(b.axes)[name] for name in b.sig.in_axes)
    a_order = a.sig.batch_axes + a.sig.out_axes + a.sig.in_axes
    b_order = b.sig.batch_axes + b.sig.out_axes + b.sig.in_axes
    left_contract = tuple(a_order.index(name) for name in a.sig.in_axes)
    right_contract = tuple(b_order.index(name) for name in b.sig.out_axes)
    batch_pairs = tuple((a_order.index(name), b_order.index(name)) for name in a.sig.batch_axes)
    cartesian_fallback_count = 1 if reducer.domain == AccumulationDomain.CARTESIAN else 0
    proof = SPFProof(
        "decoded_contract_reference",
        False,
        False,
        len(left_contract) or 1,
        cartesian_fallback_count,
        notes=("fallback uses decoded dense oracle path with explicit ReducerSpec",),
    )
    return SPFKernel(
        SPFBlockExpr(
            "LazyCompose",
            args=(a, b, (left_contract, right_contract), reducer, None, {"batch_axes": batch_pairs}),
            proof=proof,
        ),
        batch_axes + out_axes + in_axes,
        KernelSignature(
            tuple(x.name for x in out_axes),
            tuple(x.name for x in in_axes),
            batch_axes=tuple(x.name for x in batch_axes),
            reducer_policy=reducer,
        ),
    )


def _pullback_source_and_axis_map(k: SPFKernel) -> tuple[SPFKernel, dict[str, str]] | None:
    if k.expr.op != "Pullback":
        return None
    params = k.expr.args[0]
    return params["source"], dict(params["axis_map"])


def _character_label_coeffs(k: SPFKernel) -> tuple[Mapping[str, Any], dict[str, int]] | None:
    pulled = _pullback_source_and_axis_map(k)
    if pulled is None:
        return None
    source, axis_map = pulled
    if source.expr.op != "Character":
        return None
    params = _character_params(source)
    if not bool(params.get("exact_group_semantics")):
        return None
    return params, {
        axis_map[source.sig.out_axes[0]]: int(params["alpha"]),
        axis_map[source.sig.in_axes[0]]: int(params["beta"]),
    }


def _product_character_label_coeffs(k: SPFKernel) -> tuple[Mapping[str, Any], dict[str, tuple[int, ...]]] | None:
    pulled = _pullback_source_and_axis_map(k)
    if pulled is None:
        return None
    source, axis_map = pulled
    if source.expr.op != "ProductGroupCharacter":
        return None
    params = _character_params(source)
    if not bool(params.get("exact_group_semantics")):
        return None
    return params, {
        axis_map[source.sig.out_axes[0]]: tuple(int(v) for v in params["out_frequency"]),
        axis_map[source.sig.in_axes[0]]: tuple(int(v) for v in params["in_frequency"]),
    }


def _closed_pushforward(
    k: SPFKernel,
    reduce_names: tuple[str, ...],
    out_names: tuple[str, ...],
    in_names: tuple[str, ...],
    batch_names: tuple[str, ...],
    reducer: ReducerSpec,
) -> SPFKernel | None:
    if k.expr.op != "Bind" or len(reduce_names) != 1 or batch_names or len(out_names) != 1 or len(in_names) != 1:
        return None
    lhs, rhs = k.expr.args
    character = _closed_character_pushforward(lhs, rhs, reduce_names[0], out_names[0], in_names[0], reducer)
    if character is not None:
        return character
    return _closed_product_character_pushforward(lhs, rhs, reduce_names[0], out_names[0], in_names[0], reducer)


def _closed_character_pushforward(
    lhs: SPFKernel,
    rhs: SPFKernel,
    reduce_name: str,
    out_name: str,
    in_name: str,
    reducer: ReducerSpec,
) -> SPFKernel | None:
    left = _character_label_coeffs(lhs)
    right = _character_label_coeffs(rhs)
    if left is None or right is None:
        return None
    lp, lc = left
    rp, rc = right
    if int(lp["N"]) != int(rp["N"]):
        raise SPFError("character pushforward moduli must match")
    n = int(lp["N"])
    coeffs: dict[str, int] = {}
    for source in (lc, rc):
        for label, value in source.items():
            coeffs[label] = coeffs.get(label, 0) + int(value)
    axis_map = _axis_map(lhs.axes)
    out_axis = axis_map[out_name]
    in_axis = axis_map[in_name]
    if coeffs.get(reduce_name, 0) % n != 0:
        return SPFKernel(
            _closed_form_zero_expr(
                (out_axis.size, in_axis.size),
                SPFProof("character_pushforward_orthogonality_zero", True, True, 0, 0),
            ),
            (out_axis, in_axis),
            KernelSignature((out_name,), (in_name,), exactness="exact", reducer_policy=reducer),
        )
    return SPFBlockMatrix.character(
        N=n,
        alpha=coeffs.get(out_name, 0),
        beta=coeffs.get(in_name, 0),
        out_axis=out_name,
        in_axis=in_name,
        scale=complex(lp.get("scale", 1.0 + 0.0j)) * complex(rp.get("scale", 1.0 + 0.0j)) * n,
        exact_group_semantics=True,
        reducer_policy=reducer,
        proof=SPFProof("character_pushforward_orthogonality", True, True, 0, 0),
    )


def _closed_product_character_pushforward(
    lhs: SPFKernel,
    rhs: SPFKernel,
    reduce_name: str,
    out_name: str,
    in_name: str,
    reducer: ReducerSpec,
) -> SPFKernel | None:
    left = _product_character_label_coeffs(lhs)
    right = _product_character_label_coeffs(rhs)
    if left is None or right is None:
        return None
    lp, lc = left
    rp, rc = right
    moduli = tuple(int(m) for m in lp["moduli"])
    if moduli != tuple(int(m) for m in rp["moduli"]):
        raise SPFError("product character pushforward moduli must match")
    zero = tuple(0 for _ in moduli)
    coeffs: dict[str, tuple[int, ...]] = {}
    for source in (lc, rc):
        for label, value in source.items():
            prev = coeffs.get(label, zero)
            coeffs[label] = tuple(int(a) + int(b) for a, b in zip(prev, value, strict=True))
    axis_map = _axis_map(lhs.axes)
    out_axis = axis_map[out_name]
    in_axis = axis_map[in_name]
    reduce_freq = coeffs.get(reduce_name, zero)
    if any(value % modulus != 0 for value, modulus in zip(reduce_freq, moduli, strict=True)):
        return SPFKernel(
            _closed_form_zero_expr(
                (out_axis.size, in_axis.size),
                SPFProof("product_group_pushforward_orthogonality_zero", True, True, 0, 0),
            ),
            (out_axis, in_axis),
            KernelSignature((out_name,), (in_name,), exactness="exact", reducer_policy=reducer),
        )
    return SPFBlockMatrix.product_character(
        moduli=moduli,
        out_frequency=coeffs.get(out_name, zero),
        in_frequency=coeffs.get(in_name, zero),
        out_axis=out_name,
        in_axis=in_name,
        scale=complex(lp.get("scale", 1.0 + 0.0j))
        * complex(rp.get("scale", 1.0 + 0.0j))
        * int(np.prod(moduli, dtype=np.int64)),
        exact_group_semantics=True,
        reducer_policy=reducer,
        proof=SPFProof(
            "product_group_pushforward_orthogonality",
            True,
            True,
            0,
            0,
            notes=("indexing=no_carry",),
        ),
    )


def _materialize_pullback(k: SPFKernel) -> np.ndarray:
    params = k.expr.args[0]
    source = params["source"]
    axis_map = dict(params["axis_map"])
    source_words = np.asarray(source.materialize_words(), dtype=np.uint32)
    source_order = source.sig.batch_axes + source.sig.out_axes + source.sig.in_axes
    target_order = k.sig.batch_axes + k.sig.out_axes + k.sig.in_axes
    mapped_names = tuple(str(axis_map[name]) for name in source_order)
    mapped_set = set(mapped_names)
    target_shape = tuple(_axis_map(k.axes)[name].size for name in target_order)
    perm = tuple(mapped_names.index(name) for name in target_order if name in mapped_set)
    transposed = np.transpose(source_words, perm) if perm != tuple(range(len(perm))) else source_words
    reshape_shape: list[int] = []
    transposed_axis = 0
    for name in target_order:
        if name in mapped_set:
            reshape_shape.append(int(transposed.shape[transposed_axis]))
            transposed_axis += 1
        else:
            reshape_shape.append(1)
    return np.broadcast_to(transposed.reshape(tuple(reshape_shape)), target_shape).copy()


def _circulant_matrix_from_spectrum(spectrum: np.ndarray) -> np.ndarray:
    n = int(np.asarray(spectrum).size)
    kernel = np.fft.ifft(np.asarray(spectrum, dtype=np.complex128))
    return np.stack([np.roll(kernel, k) for k in range(n)], axis=1)


def _block_circulant_matrix_from_spectrum(spectrum: np.ndarray) -> np.ndarray:
    spec = np.asarray(spectrum, dtype=np.complex128)
    if spec.ndim != 3:
        raise SPFError("block-circulant spectrum must have shape (frequency, out_block, in_block)")
    frequency_count, out_blocks, in_blocks = (int(x) for x in spec.shape)
    dense = np.zeros(
        (frequency_count * out_blocks, frequency_count * in_blocks),
        dtype=np.complex128,
    )
    for out_block in range(out_blocks):
        for in_block in range(in_blocks):
            circulant = _circulant_matrix_from_spectrum(spec[:, out_block, in_block])
            dense[out_block::out_blocks, in_block::in_blocks] = circulant
    return dense


def _block_circulant_apply(a: SPFKernel, vector: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "BlockCirculantSpectrum":
        return None
    if vector.sig.batch_axes or vector.sig.in_axes or len(vector.sig.out_axes) != 1:
        return None
    params = _block_circulant_params(a)
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(a.axes)[a.sig.in_axes[0]]
    vector_axis = _axis_map(vector.axes)[vector.sig.out_axes[0]]
    if vector_axis.size != in_axis.size:
        raise SPFError("block-circulant vector size mismatch")
    if vector_axis.basis != in_axis.basis:
        raise SPFError("block-circulant vector basis mismatch")
    reducer = a.sig.reducer_policy or vector.sig.reducer_policy or _default_reducer()
    frequency_count = int(params["frequency_count"])
    out_blocks = int(params["out_blocks"])
    in_blocks = int(params["in_blocks"])
    cartesian_fallback_count = 1 if reducer.domain == AccumulationDomain.CARTESIAN else 0
    return SPFKernel(
        SPFBlockExpr(
            "BlockCirculantApply",
            args=({"operator": a, "vector": vector, "reducer": reducer},),
            proof=SPFProof(
                "block_circulant_apply_plan",
                False,
                True,
                frequency_count * out_blocks,
                cartesian_fallback_count,
                notes=(
                    f"frequency_count={frequency_count}",
                    f"out_blocks={out_blocks}",
                    f"in_blocks={in_blocks}",
                    "fft over positions, per-frequency block matvec, inverse fft",
                ),
            ),
        ),
        (out_axis,),
        KernelSignature(
            (out_axis.name,),
            (),
            basis_out=(out_axis.basis,),
            reducer_policy=reducer,
        ),
    )


def _block_from_spectrum_words(words: np.ndarray) -> SPFBlockV3:
    arr = np.asarray(words, dtype=np.uint32).reshape(-1)
    return SPFBlockV3(
        dims=(int(arr.size),),
        representation=RepresentationRef(int(Representation.POLAR_SPF32)),
        base=BaseModelRef(
            int(BaseModelKind.CIRCULANT_SPECTRUM),
            {
                "spectrum_L": unpack_L(arr).astype(np.int64),
                "spectrum_T": unpack_theta(arr).astype(np.int64),
                "word_bits": 32,
            },
        ),
        residual=ResidualRef(int(ResidualCodecKind.NONE), None),
    )


def _is_power_of_two(n: int) -> bool:
    n = int(n)
    return n > 0 and (n & (n - 1)) == 0


def _gauss_sum(N: int, A: int) -> complex:
    r"""The full-period quadratic Gauss sum ``G_N(A) = Σ_{k∈Z_N} ω^{A k²}``.

    For ``N = 2^n`` with ``A`` odd this is the closed form
    ``G = (1 + i^A)·(2/A)^n·√N`` (``(2/A)`` the Jacobi symbol: ``+1`` for
    ``A ≡ ±1 mod 8``, ``-1`` for ``A ≡ ±3 mod 8``), which is ``0`` at ``n = 1`` and
    otherwise the exact SPF32 word ``L = 128(n+1)``, ``Θ ∈ {8192, 24576, 40960,
    57344}``. The sign factor only bites for odd ``n``. Every other case (odd
    ``N``, even ``A``) is the finite sum, quantized.
    """
    n = int(N)
    a = int(A) % n if n > 0 else 0
    if _is_power_of_two(n) and a % 2 == 1:
        e = n.bit_length() - 1
        if e == 1:
            return 0.0 + 0.0j
        jacobi_two = 1 if a % 8 in (1, 7) else -1
        unit = (1.0 + 1.0j) if a % 4 == 1 else (1.0 - 1.0j)
        return complex(unit * (jacobi_two**e) * math.sqrt(n))
    k = np.arange(n)
    return complex(np.sum(np.exp(2j * np.pi * ((a * k * k) % n) / n)))


def _quadratic_phase_matrix(params: Mapping[str, Any]) -> np.ndarray:
    """Reference materialization of a :meth:`SPFBlockMatrix.quadratic_phase` node.

    Builds the dense complex matrix (encoded to words by the caller, exactly as
    ``Character`` does); the closed *composition* never touches this path.
    """
    n = int(params["N"])
    kind = str(params.get("kind", "plain"))
    scale = complex(params.get("scale", 1.0 + 0.0j))
    i = np.arange(n)[:, None]
    j = np.arange(n)[None, :]
    if kind == "plain":
        a, b, c = int(params["a"]), int(params["b"]), int(params["c"])
        phase = (a * i * i + b * i * j + c * j * j) % n
        return scale * np.exp(2j * np.pi * phase / n)
    if kind == "metaplectic_2n":
        a, c = int(params["a"]), int(params["c"])
        ainv, bb, ee = int(params["Ainv"]), int(params["bb"]), int(params["ee"])
        big_b = bb * i + ee * j
        half = (big_b // 2) % n
        phase = (a * i * i + c * j * j - ainv * half * half) % n
        return np.where(big_b % 2 == 0, scale * np.exp(2j * np.pi * phase / n), 0.0)
    if kind == "degenerate_zero":
        a, c = int(params["a"]), int(params["c"])
        bb, ee = int(params["bb"]), int(params["ee"])
        big_b = (bb * i + ee * j) % n
        phase = (a * i * i + c * j * j) % n
        return np.where(big_b == 0, scale * np.exp(2j * np.pi * phase / n), 0.0)
    raise SPFError(f"unknown quadratic-phase kind {kind!r}")


def _compose_quadratic_phase(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    """Law 2/3 — ``Q(a,b,c) ∘ Q(d,e,f)`` of the discrete metaplectic group.

    Completing the square in the ``k``-sum ``Σ_k ω^{A k² + (b i + e j) k}`` with
    ``A = c + d`` gives, in ``O(1)`` modular integer ops:

    * odd ``N`` with ``gcd(A, N) = 1``: ``G_N(A) · Q(a-b²γ, -2beγ, f-e²γ)``,
      ``γ = (4A)⁻¹ mod N`` — a plain quadratic block (closed; chains collapse);
    * ``N = 2^n`` with ``A`` odd: the parity-masked block
      ``[b i + e j even] · G_N(A) · ω^{a i² + f j² − A⁻¹((b i + e j)/2)²}`` — exact
      in SPF32 (the scalar is an exact word);
    * ``A ≡ 0 (mod N)``: the scaled permutation ``N · [b i + e j ≡ 0] · ω^{a i²+f j²}``
      with chirp diagonals.

    Any other case (incomplete Gauss sum / non-invertible ``A``, or a non-plain
    operand from an earlier ``N = 2^n`` / degenerate product) declines so the dense
    fallback keeps the result correct.
    """
    if a.expr.op != "QuadraticPhase" or b.expr.op != "QuadraticPhase":
        return None
    pa = a.expr.args[0]
    pb = b.expr.args[0]
    if str(pa.get("kind")) != "plain" or str(pb.get("kind")) != "plain":
        return None
    n = int(pa["N"])
    if int(pb["N"]) != n:
        raise SPFError("quadratic-phase moduli must match")
    if n == 1:
        return None
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    reducer_policy = a.sig.reducer_policy or b.sig.reducer_policy
    a_i2, b_ik, c_k2 = int(pa["a"]), int(pa["b"]), int(pa["c"])
    d_k2, e_kj, f_j2 = int(pb["a"]), int(pb["b"]), int(pb["c"])
    A = (c_k2 + d_k2) % n
    scale = complex(pa.get("scale", 1.0)) * complex(pb.get("scale", 1.0))

    if A % n == 0:
        params = {
            "kind": "degenerate_zero",
            "N": n,
            "a": a_i2 % n,
            "c": f_j2 % n,
            "bb": b_ik % n,
            "ee": e_kj % n,
            "scale": scale * float(n),
        }
        exact = _is_power_of_two(n) and scale == 1.0 + 0.0j
        law = "quadratic_phase_compose_degenerate"
    elif n % 2 == 1 and math.gcd(A, n) == 1:
        gamma = pow((4 * A) % n, -1, n)
        params = {
            "kind": "plain",
            "N": n,
            "a": (a_i2 - b_ik * b_ik * gamma) % n,
            "b": (-2 * b_ik * e_kj * gamma) % n,
            "c": (f_j2 - e_kj * e_kj * gamma) % n,
            "scale": scale * _gauss_sum(n, A),
        }
        exact = False  # odd-N Gauss magnitude √N is not on the SPF32 L-lattice
        law = "quadratic_phase_compose_odd"
    elif _is_power_of_two(n) and n >= 4 and A % 2 == 1:
        # The parity mask comes from k -> k + N/2, which needs ω^{A N²/4} = 1,
        # i.e. N >= 4. At N = 2 the law is false (odd-B entries are 2, not 0),
        # so N = 2 declines to the dense fallback below.
        params = {
            "kind": "metaplectic_2n",
            "N": n,
            "a": a_i2 % n,
            "c": f_j2 % n,
            "Ainv": pow(A % n, -1, n),
            "bb": b_ik % n,
            "ee": e_kj % n,
            "scale": scale * _gauss_sum(n, A),
        }
        exact = scale == 1.0 + 0.0j
        law = "quadratic_phase_compose_2n"
    else:
        return None  # incomplete Gauss sum: decline to the dense fallback

    return SPFBlockMatrix._quadratic_phase_node(
        params,
        AxisSpec(out_axis.name, n, role="codomain", basis=out_axis.basis),
        AxisSpec(in_axis.name, n, role="domain", basis=in_axis.basis),
        exact=exact,
        reducer_policy=reducer_policy,
        proof=SPFProof(
            law,
            exact,
            True,
            0,
            0,
            notes=(f"N={n}", f"A={A}") if exact else (f"N={n}", f"A={A}", "quantized_scalar"),
        ),
    )


def _materialize_character(params: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 2:
        raise SPFError("character materialization requires matrix shape")
    n = int(params["N"])
    if shape != (n, n):
        raise SPFError("character shape must equal (N, N)")
    o = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    phase = (int(params["alpha"]) * o + int(params["beta"]) * i) % n
    return complex(params.get("scale", 1.0 + 0.0j)) * np.exp(2j * np.pi * phase / n)


def _mod_frequency(values: Sequence[int], moduli: Sequence[int]) -> tuple[int, ...]:
    vals = tuple(int(v) for v in values)
    mods = tuple(int(m) for m in moduli)
    if len(vals) != len(mods):
        raise SPFError("frequency rank must match product-group moduli")
    return tuple(v % m for v, m in zip(vals, mods, strict=True))


def _product_group_coords(moduli: tuple[int, ...]) -> np.ndarray:
    size = int(np.prod(moduli, dtype=np.int64))
    coords = np.empty((size, len(moduli)), dtype=np.int64)
    for idx in range(size):
        coords[idx, :] = np.unravel_index(idx, moduli)
    return coords


def _materialize_product_character(params: Mapping[str, Any], shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) != 2:
        raise SPFError("product character materialization requires matrix shape")
    moduli = tuple(int(m) for m in params["moduli"])
    size = int(np.prod(moduli, dtype=np.int64))
    if shape != (size, size):
        raise SPFError("product character shape must equal product-group size")
    out_freq = np.asarray(params["out_frequency"], dtype=np.int64)
    in_freq = np.asarray(params["in_frequency"], dtype=np.int64)
    coords = _product_group_coords(moduli)
    out_phase = np.zeros(size, dtype=np.float64)
    in_phase = np.zeros(size, dtype=np.float64)
    for dim, modulus in enumerate(moduli):
        out_phase += out_freq[dim] * coords[:, dim] / modulus
        in_phase += in_freq[dim] * coords[:, dim] / modulus
    phase = out_phase[:, None] + in_phase[None, :]
    return complex(params.get("scale", 1.0 + 0.0j)) * np.exp(2j * np.pi * phase)


def _require_square_signature(k: SPFKernel, op_name: str) -> tuple[int, int] | None:
    shape = _shape_for(k.sig, k.axes)
    if len(k.sig.batch_axes) != 0 or len(k.sig.out_axes) != 1 or len(k.sig.in_axes) != 1:
        return None
    out_n = int(shape[0])
    in_n = int(shape[1])
    if out_n != in_n:
        raise SPFError(f"{op_name} requires a square matrix")
    return out_n, in_n


def _structural_trace(k: SPFKernel) -> complex | None:
    if k.expr.op == "ClosedFormZero":
        square = _require_square_signature(k, "trace")
        return None if square is None else 0.0 + 0.0j
    if k.expr.op == "ClosedFormIdentity":
        square = _require_square_signature(k, "trace")
        return None if square is None else complex(square[0])
    if k.expr.op == "DiagonalSPF":
        _require_square_signature(k, "trace")
        return complex(np.sum(_complex_from_words(k.expr.args[0]["diag_words"])))
    if k.expr.op == "CirculantSpectrum":
        _require_square_signature(k, "trace")
        assert k.expr.block is not None
        return complex(np.sum(block_to_complex(k.expr.block).reshape(-1)))
    if k.expr.op == "BlockCirculantSpectrum":
        spectrum = _block_circulant_spectrum(k)
        if spectrum.shape[1] != spectrum.shape[2]:
            raise SPFError("trace requires a square matrix")
        return complex(np.trace(spectrum, axis1=1, axis2=2).sum())
    return None


def _structural_det(k: SPFKernel) -> complex | None:
    if k.expr.op == "ClosedFormZero":
        square = _require_square_signature(k, "det")
        return None if square is None else complex(1.0 if square[0] == 0 else 0.0)
    if k.expr.op == "ClosedFormIdentity":
        square = _require_square_signature(k, "det")
        return None if square is None else 1.0 + 0.0j
    if k.expr.op == "DiagonalSPF":
        _require_square_signature(k, "det")
        return complex(np.prod(_complex_from_words(k.expr.args[0]["diag_words"])))
    if k.expr.op == "CirculantSpectrum":
        _require_square_signature(k, "det")
        assert k.expr.block is not None
        return complex(np.prod(block_to_complex(k.expr.block).reshape(-1)))
    if k.expr.op == "BlockCirculantSpectrum":
        spectrum = _block_circulant_spectrum(k)
        if spectrum.shape[1] != spectrum.shape[2]:
            raise SPFError("det requires a square matrix")
        out = 1.0 + 0.0j
        for mat in spectrum:
            out *= complex(np.linalg.det(mat))
        return out
    return None


def _structural_spectral_norm(k: SPFKernel) -> float | None:
    if k.expr.op == "ClosedFormZero":
        return 0.0
    if k.expr.op == "ClosedFormIdentity":
        square = _require_square_signature(k, "spectral_norm")
        return None if square is None else (0.0 if square[0] == 0 else 1.0)
    if k.expr.op == "DiagonalSPF":
        diag = _complex_from_words(k.expr.args[0]["diag_words"]).reshape(-1)
        return 0.0 if diag.size == 0 else float(np.max(np.abs(diag)))
    if k.expr.op == "CirculantSpectrum":
        assert k.expr.block is not None
        spectrum = block_to_complex(k.expr.block).reshape(-1)
        return 0.0 if spectrum.size == 0 else float(np.max(np.abs(spectrum)))
    if k.expr.op == "BlockCirculantSpectrum":
        spectrum = _block_circulant_spectrum(k)
        if spectrum.size == 0:
            return 0.0
        return float(max(float(np.linalg.svd(mat, compute_uv=False)[0]) for mat in spectrum))
    return None


def _validate_permutation(indices: Sequence[int]) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = int(idx.size)
    if np.any(idx < 0) or np.any(idx >= n):
        raise SPFError("permutation indices out of range")
    if len(set(int(i) for i in idx)) != n:
        raise SPFError("permutation indices must be unique")
    return idx


def _permutation_matrix(indices: Sequence[int]) -> np.ndarray:
    idx = _validate_permutation(indices)
    mat = np.zeros((idx.size, idx.size), dtype=np.complex128)
    if idx.size:
        mat[np.arange(idx.size), idx] = 1.0
    return mat


def _roll_indices(n: int, shift: int) -> np.ndarray:
    size = int(n)
    if size < 0:
        raise SPFError("roll size must be non-negative")
    if size == 0:
        return np.zeros(0, dtype=np.int64)
    return (np.arange(size, dtype=np.int64) - int(shift)) % size


def _roll_matrix(n: int, shift: int) -> np.ndarray:
    return _permutation_matrix(_roll_indices(n, shift))


def _roll_spectrum(n: int, shift: int) -> np.ndarray:
    size = int(n)
    impulse = np.zeros(size, dtype=np.complex128)
    if size:
        impulse[int(shift) % size] = 1.0
    return np.fft.fft(impulse)


def _supports_resident_fft(n: int) -> bool:
    size = int(n)
    return 2 <= size <= 1024 and (size & (size - 1)) == 0


def _compose_circulant(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "CirculantSpectrum" and len(b.sig.out_axes) == 1 and not b.sig.in_axes:
        assert a.expr.block is not None
        n = int(a.expr.block.dims[0])
        if not _supports_resident_fft(n):
            return None
        out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
        return SPFKernel(
            SPFBlockExpr(
                "CirculantApply",
                args=(a, b),
                proof=SPFProof(
                    "circulant_apply_fft_diagonal_ifft",
                    False,
                    True,
                    0,
                    0,
                    notes=(f"n={n}",),
                ),
            ),
            (out_axis,),
            KernelSignature(
                (out_axis.name,),
                (),
                basis_out=(out_axis.basis,),
                reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
            ),
        )
    if a.expr.op != "CirculantSpectrum" or b.expr.op != "CirculantSpectrum":
        return None
    assert a.expr.block is not None and b.expr.block is not None
    if a.expr.block.dims != b.expr.block.dims:
        raise SPFError("circulant spectra must have matching sizes")
    words = spf_mul_words(a.expr.block.to_words(), b.expr.block.to_words())
    block = _block_from_spectrum_words(words)
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    return SPFBlockMatrix(
        SPFBlockExpr(
            "CirculantSpectrum",
            block=block,
            proof=SPFProof("circulant_compose", False, True, 0, 0),
        ),
        (out_axis, in_axis),
        KernelSignature(
            (out_axis.name,),
            (in_axis.name,),
            basis_in=("fourier",),
            basis_out=("fourier",),
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        ),
    )


def _block_circulant_params(k: SPFKernel) -> Mapping[str, Any]:
    return k.expr.args[0]


def _block_circulant_block_axes_from_params(params: Mapping[str, Any]) -> tuple[str, str]:
    raw = tuple(str(x) for x in params.get("block_axes", ("bo", "bi")))
    if len(raw) != 2:
        raise SPFError("block-circulant block_axes metadata must contain exactly two axis names")
    return (raw[0], raw[1])


def _block_circulant_spectrum(k: SPFKernel) -> np.ndarray:
    params = _block_circulant_params(k)
    frequency_count = int(params["frequency_count"])
    out_blocks = int(params["out_blocks"])
    in_blocks = int(params["in_blocks"])
    if "spectrum_words" in params:
        return _complex_from_words(params["spectrum_words"]).reshape(
            frequency_count,
            out_blocks,
            in_blocks,
        )
    inner_structure = str(params.get("inner_structure", "dense"))
    if inner_structure == "diagonal":
        diag = _complex_from_words(params["diagonal_words"]).reshape(frequency_count, out_blocks)
        if out_blocks != in_blocks:
            raise SPFError("diagonal block-circulant spectrum must be square")
        spectrum = np.zeros((frequency_count, out_blocks, in_blocks), dtype=np.complex128)
        idx = np.arange(out_blocks)
        spectrum[:, idx, idx] = diag
        return spectrum
    if inner_structure == "low_rank":
        u, v, coeff = _block_circulant_low_rank_factors(k)
        return np.einsum("for,fr,fir->foi", u, coeff, np.conj(v))
    if inner_structure == "sparse":
        indices, values = _block_circulant_sparse_entries(k)
        return _block_circulant_sparse_dense_from_entries(
            (frequency_count, out_blocks, in_blocks),
            indices,
            values,
        )
    if inner_structure == "character":
        n, alpha, beta, scale = _block_circulant_character_factors(k)
        out = np.arange(n, dtype=np.int64)
        inn = np.arange(n, dtype=np.int64)
        phase = (alpha[:, None, None] * out[None, :, None]) + (beta[:, None, None] * inn[None, None, :])
        return scale[:, None, None] * np.exp((2j * np.pi / n) * np.mod(phase, n))
    raise SPFError("block-circulant spectrum has no dense spectrum payload")


def _block_circulant_low_rank_spectrum_and_params(spectrum: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(spectrum, Mapping):
        if "u" not in spectrum or "v" not in spectrum:
            raise SPFError("low-rank block-circulant mapping requires u and v factors")
        u_src = spectrum.get("u")
        v_src = spectrum.get("v")
        u = _as_complex_values(u_src)
        v = _as_complex_values(v_src)
        coeff_src = spectrum.get("coeff")
    elif isinstance(spectrum, tuple) and len(spectrum) in {2, 3}:
        u_src = spectrum[0]
        v_src = spectrum[1]
        u = _as_complex_values(u_src)
        v = _as_complex_values(v_src)
        coeff_src = None if len(spectrum) == 2 else spectrum[2]
    else:
        raise SPFError("low-rank block-circulant spectrum must be a mapping or (u, v, coeff) tuple")
    if u.ndim != 3 or v.ndim != 3:
        raise SPFError("low-rank block-circulant factors must have shape (frequency, block, rank)")
    if u.shape[0] != v.shape[0] or u.shape[2] != v.shape[2]:
        raise SPFError("low-rank block-circulant factors must share frequency count and rank")
    frequency_count, out_blocks, rank = (int(x) for x in u.shape)
    in_blocks = int(v.shape[1])
    if coeff_src is None:
        coeff = np.ones((frequency_count, rank), dtype=np.complex128)
        coeff_words = encode_words(coeff)
    else:
        coeff = _as_complex_values(coeff_src)
        coeff_words = _as_spf_words(coeff_src).reshape(np.asarray(coeff_src).shape)
        if coeff.ndim == 1:
            if int(coeff.size) != rank:
                raise SPFError("low-rank block-circulant coeff length must match rank")
            coeff = np.broadcast_to(coeff.reshape(1, rank), (frequency_count, rank)).copy()
            coeff_words = np.broadcast_to(coeff_words.reshape(1, rank), (frequency_count, rank)).copy()
        elif coeff.shape != (frequency_count, rank):
            raise SPFError("low-rank block-circulant coeff must have shape (frequency, rank)")
    if frequency_count <= 0 or out_blocks <= 0 or in_blocks <= 0:
        raise SPFError("low-rank block-circulant dimensions must be positive")
    spec = np.einsum("for,fr,fir->foi", u, coeff, np.conj(v))
    return spec, {
        "u_words": _as_spf_words(u_src).reshape(u.shape),
        "v_words": _as_spf_words(v_src).reshape(v.shape),
        "coeff_words": coeff_words.reshape(coeff.shape),
        "rank": rank,
    }


def _block_circulant_low_rank_factors(k: SPFKernel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = _block_circulant_params(k)
    if str(params.get("inner_structure", "dense")) != "low_rank":
        raise SPFError("block-circulant spectrum is not low-rank")
    f = int(params["frequency_count"])
    out_blocks = int(params["out_blocks"])
    in_blocks = int(params["in_blocks"])
    rank = int(params["rank"])
    u = _complex_from_words(params["u_words"]).reshape(f, out_blocks, rank)
    v = _complex_from_words(params["v_words"]).reshape(f, in_blocks, rank)
    coeff = _outer_coefficients(params).reshape(f, rank)
    return u, v, coeff


def _block_circulant_sparse_spectrum_and_params(spectrum: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if isinstance(spectrum, Mapping):
        if "shape" not in spectrum or "indices" not in spectrum or "values" not in spectrum:
            raise SPFError("sparse block-circulant mapping requires shape, indices, and values")
        shape_src = tuple(int(x) for x in spectrum["shape"])
        if len(shape_src) != 3:
            raise SPFError("sparse block-circulant shape must be (frequency, out_block, in_block)")
        shape = (shape_src[0], shape_src[1], shape_src[2])
        indices = np.asarray(spectrum["indices"], dtype=np.int64)
        values = _as_complex_values(spectrum["values"])
        dense = _block_circulant_sparse_dense_from_entries(shape, indices, values)
        value_words = _as_spf_words(spectrum["values"]).reshape(values.shape)
        if any(int(x) <= 0 for x in dense.shape):
            raise SPFError("sparse block-circulant dimensions must be positive")
        return dense, _block_circulant_sparse_params_from_entries(shape, indices, value_words)
    else:
        dense_words = _as_spf_words(spectrum) if _is_packed_spf_words(spectrum) else None
        dense = _as_complex_values(spectrum)
        if dense.ndim != 3:
            raise SPFError("sparse block-circulant spectrum must have shape (frequency, out_block, in_block)")
        value_words = dense_words
    if any(int(x) <= 0 for x in dense.shape):
        raise SPFError("sparse block-circulant dimensions must be positive")
    return dense, _block_circulant_sparse_params_from_dense(dense, values_words=value_words)


def _block_circulant_sparse_params_from_entries(
    shape: tuple[int, int, int],
    indices: np.ndarray,
    value_words: np.ndarray,
) -> dict[str, Any]:
    _block_circulant_sparse_dense_from_entries(shape, indices, _complex_from_words(value_words).reshape(-1))
    return {
        "sparse_indices": np.asarray(indices, dtype=np.int64).reshape(-1, 3).copy(),
        "sparse_value_words": np.asarray(value_words, dtype=np.uint32).reshape(-1).copy(),
        "nnz": int(np.asarray(indices).reshape(-1, 3).shape[0]),
    }


def _block_circulant_sparse_dense_from_entries(
    shape: tuple[int, int, int],
    indices: np.ndarray,
    values: np.ndarray,
) -> np.ndarray:
    if any(int(x) <= 0 for x in shape):
        raise SPFError("sparse block-circulant dimensions must be positive")
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise SPFError("sparse block-circulant indices must have shape (nnz, 3)")
    if values.ndim != 1 or values.shape[0] != indices.shape[0]:
        raise SPFError("sparse block-circulant values must have shape (nnz,)")
    dense = np.zeros(shape, dtype=np.complex128)
    if indices.size == 0:
        return dense
    bounds = np.asarray(shape, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= bounds.reshape(1, 3)):
        raise SPFError("sparse block-circulant indices out of bounds")
    np.add.at(dense, (indices[:, 0], indices[:, 1], indices[:, 2]), values)
    return dense


def _block_circulant_sparse_params_from_dense(
    spectrum: np.ndarray,
    *,
    values_words: np.ndarray | None = None,
) -> dict[str, Any]:
    indices = np.argwhere(spectrum != 0)
    values = spectrum[tuple(indices.T)] if indices.size else np.zeros(0, dtype=np.complex128)
    if values_words is None:
        stored_words = encode_words(values)
    else:
        word_arr = np.asarray(values_words, dtype=np.uint32)
        if word_arr.shape == spectrum.shape:
            stored_words = word_arr[tuple(indices.T)] if indices.size else np.zeros(0, dtype=np.uint32)
        else:
            stored_words = word_arr.reshape(-1)
            if stored_words.shape != values.shape:
                raise SPFError("sparse block-circulant value words must match nnz")
    return {
        "sparse_indices": indices.astype(np.int64, copy=False),
        "sparse_value_words": stored_words,
        "nnz": int(indices.shape[0]),
    }


def _block_circulant_sparse_entries(k: SPFKernel) -> tuple[np.ndarray, np.ndarray]:
    params = _block_circulant_params(k)
    if str(params.get("inner_structure", "dense")) != "sparse":
        raise SPFError("block-circulant spectrum is not sparse")
    indices = np.asarray(params["sparse_indices"], dtype=np.int64).reshape(-1, 3)
    values = _complex_from_words(params["sparse_value_words"]).reshape(-1)
    return indices, values


def _block_circulant_character_spectrum_and_params(spectrum: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(spectrum, Mapping):
        raise SPFError("character block-circulant spectrum requires metadata mapping")
    if "N" not in spectrum or "alpha" not in spectrum or "beta" not in spectrum:
        raise SPFError("character block-circulant mapping requires N, alpha, and beta")
    n = int(spectrum["N"])
    if n <= 0:
        raise SPFError("character block-circulant modulus N must be positive")
    alpha = np.asarray(spectrum["alpha"], dtype=np.int64)
    beta = np.asarray(spectrum["beta"], dtype=np.int64)
    if alpha.ndim != 1 or beta.ndim != 1 or alpha.shape != beta.shape:
        raise SPFError("character block-circulant alpha and beta must have shape (frequency,)")
    frequency_count = int(alpha.shape[0])
    if frequency_count <= 0:
        raise SPFError("character block-circulant frequency count must be positive")
    scale_src = spectrum.get("scale", 1.0 + 0.0j)
    scale = _as_complex_values(scale_src)
    if scale.ndim == 0:
        scale = np.full(frequency_count, complex(scale.item()), dtype=np.complex128)
        scale_words = encode_words(scale)
    elif scale.shape != (frequency_count,):
        raise SPFError("character block-circulant scale must be scalar or have shape (frequency,)")
    else:
        scale_words = _as_spf_words(scale_src).reshape(scale.shape)
    alpha_mod = np.mod(alpha, n).astype(np.int64, copy=False)
    beta_mod = np.mod(beta, n).astype(np.int64, copy=False)
    out = np.arange(n, dtype=np.int64)
    inn = np.arange(n, dtype=np.int64)
    phase = (alpha_mod[:, None, None] * out[None, :, None]) + (beta_mod[:, None, None] * inn[None, None, :])
    spec = scale[:, None, None] * np.exp((2j * np.pi / n) * np.mod(phase, n))
    return spec, {
        "character_N": n,
        "character_alpha": tuple(int(x) for x in alpha_mod),
        "character_beta": tuple(int(x) for x in beta_mod),
        "character_scale_words": scale_words.reshape(scale.shape),
    }


def _block_circulant_character_factors(k: SPFKernel) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    params = _block_circulant_params(k)
    if str(params.get("inner_structure", "dense")) != "character":
        raise SPFError("block-circulant spectrum is not character-structured")
    n = int(params["character_N"])
    frequency_count = int(params["frequency_count"])
    alpha = np.asarray(params["character_alpha"], dtype=np.int64).reshape(frequency_count)
    beta = np.asarray(params["character_beta"], dtype=np.int64).reshape(frequency_count)
    scale = _complex_from_words(params["character_scale_words"]).reshape(frequency_count)
    return n, alpha, beta, scale


def _block_circulant_diagonal_entries(k: SPFKernel) -> np.ndarray:
    params = _block_circulant_params(k)
    if "diagonal_words" in params:
        return _complex_from_words(params["diagonal_words"]).reshape(
            int(params["frequency_count"]),
            int(params["out_blocks"]),
        )
    spectrum = _block_circulant_spectrum(k)
    if spectrum.shape[1] != spectrum.shape[2]:
        raise SPFError("diagonal block-circulant spectrum must be square")
    return np.diagonal(spectrum, axis1=1, axis2=2)


def _block_circulant_storage_accounting(k: SPFKernel) -> dict[str, Any]:
    params = _block_circulant_params(k)
    frequency_count = int(params["frequency_count"])
    out_blocks = int(params["out_blocks"])
    in_blocks = int(params["in_blocks"])
    spectral_dense_words = frequency_count * out_blocks * in_blocks
    logical_dense_words = (frequency_count * out_blocks) * (frequency_count * in_blocks)
    inner_structure = str(params.get("inner_structure", "dense"))
    metadata_entries = 0
    if inner_structure == "dense":
        stored_value_words = spectral_dense_words
    elif inner_structure == "diagonal":
        stored_value_words = frequency_count * out_blocks
    elif inner_structure == "low_rank":
        rank = int(params["rank"])
        stored_value_words = frequency_count * rank * (out_blocks + in_blocks + 1)
    elif inner_structure == "sparse":
        nnz = int(params["nnz"])
        stored_value_words = nnz
        metadata_entries = 3 * nnz
    elif inner_structure == "character":
        stored_value_words = frequency_count
        metadata_entries = 2 * frequency_count + 1
    else:
        stored_value_words = spectral_dense_words
    denominator = max(stored_value_words, 1)
    return {
        "logical_dense_words": logical_dense_words,
        "spectral_dense_words": spectral_dense_words,
        "stored_value_words": stored_value_words,
        "metadata_entries": metadata_entries,
        "block_axes": _block_circulant_block_axes_from_params(params),
        "logical_to_stored_word_ratio": float(logical_dense_words / denominator),
        "spectral_to_stored_word_ratio": float(spectral_dense_words / denominator),
    }


def _compose_block_circulant_sparse_sparse(a: SPFKernel, b: SPFKernel) -> tuple[dict[str, Any], int]:
    ia, va = _block_circulant_sparse_entries(a)
    ib, vb = _block_circulant_sparse_entries(b)
    pa = _block_circulant_params(a)
    pb = _block_circulant_params(b)
    shape = (int(pa["frequency_count"]), int(pa["out_blocks"]), int(pb["in_blocks"]))
    by_frequency_and_row: dict[tuple[int, int], list[tuple[int, complex]]] = {}
    for (f_b, m_b, i_b), value_b in zip(ib, vb, strict=True):
        by_frequency_and_row.setdefault((int(f_b), int(m_b)), []).append((int(i_b), complex(value_b)))
    accum: dict[tuple[int, int, int], complex] = {}
    reducer_count = 0
    for (f_a, o_a, m_a), value_a in zip(ia, va, strict=True):
        for i_b, value_b in by_frequency_and_row.get((int(f_a), int(m_a)), ()):
            key = (int(f_a), int(o_a), i_b)
            accum[key] = accum.get(key, 0.0 + 0.0j) + complex(value_a) * value_b
            reducer_count += 1
    if accum:
        items = sorted((key, value) for key, value in accum.items() if value != 0.0)
        indices = np.asarray([key for key, _ in items], dtype=np.int64).reshape(-1, 3)
        values = np.asarray([value for _, value in items], dtype=np.complex128)
    else:
        indices = np.zeros((0, 3), dtype=np.int64)
        values = np.zeros(0, dtype=np.complex128)
    return {"shape": shape, "indices": indices, "values": values}, reducer_count


def _scale_block_circulant_sparse(
    k: SPFKernel,
    scale: np.ndarray,
    *,
    side: str,
) -> dict[str, Any]:
    indices, values = _block_circulant_sparse_entries(k)
    params = _block_circulant_params(k)
    shape = (
        int(params["frequency_count"]),
        int(params["out_blocks"]),
        int(params["in_blocks"]),
    )
    if side == "left":
        scaled = values * scale[indices[:, 0], indices[:, 1]]
    elif side == "right":
        scaled = values * scale[indices[:, 0], indices[:, 2]]
    else:
        raise SPFError(f"unknown sparse scaling side {side!r}")
    return {"shape": shape, "indices": indices, "values": scaled}


def _compose_block_circulant(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    applied = _block_circulant_apply(a, b)
    if applied is not None:
        return applied
    if a.expr.op != "BlockCirculantSpectrum" or b.expr.op != "BlockCirculantSpectrum":
        return None
    pa = _block_circulant_params(a)
    pb = _block_circulant_params(b)
    if int(pa["frequency_count"]) != int(pb["frequency_count"]):
        raise SPFError("block-circulant frequency counts must match")
    if int(pa["in_blocks"]) != int(pb["out_blocks"]):
        raise SPFError("block-circulant inner block dimensions must match")
    inner_a = str(pa.get("inner_structure", "dense"))
    inner_b = str(pb.get("inner_structure", "dense"))
    if inner_a == "low_rank" and inner_b == "low_rank":
        ua, va, ca = _block_circulant_low_rank_factors(a)
        ub, vb, cb = _block_circulant_low_rank_factors(b)
        gram = np.einsum("fmr,fms->frs", np.conj(va), ub)
        coeff = (ca[:, :, None] * cb[:, None, :] * gram).reshape(ca.shape[0], -1)
        u = np.repeat(ua, ub.shape[2], axis=2)
        v = np.tile(vb, (1, 1, ua.shape[2]))
        spectrum = (u, v, coeff)
        reducer_count = int(pa["frequency_count"]) * int(ca.shape[1]) * int(cb.shape[1])
        inner_structure = "low_rank"
        law = "block_circulant_per_frequency_low_rank_matmul"
    elif inner_a == "character" and inner_b == "character":
        na, alpha_a, beta_a, scale_a = _block_circulant_character_factors(a)
        nb, alpha_b, beta_b, scale_b = _block_circulant_character_factors(b)
        if na != nb:
            raise SPFError("character block-circulant moduli must match")
        cancels = np.mod(beta_a + alpha_b, na) == 0
        scale = np.where(cancels, na * scale_a * scale_b, 0.0 + 0.0j)
        spectrum = {
            "N": na,
            "alpha": alpha_a,
            "beta": beta_b,
            "scale": scale,
        }
        reducer_count = 0
        inner_structure = "character"
        law = "block_circulant_per_frequency_character_orthogonality"
    elif inner_a == "sparse" and inner_b == "sparse":
        spectrum, reducer_count = _compose_block_circulant_sparse_sparse(a, b)
        inner_structure = "sparse"
        law = "block_circulant_per_frequency_sparse_matmul"
    elif inner_a == "diagonal" and inner_b == "diagonal":
        spectrum = _block_circulant_diagonal_entries(a) * _block_circulant_diagonal_entries(b)
        reducer_count = 0
        inner_structure = "diagonal"
        law = "block_circulant_per_frequency_diagonal_matmul"
    elif inner_a == "diagonal" and inner_b == "sparse":
        spectrum = _scale_block_circulant_sparse(
            b,
            _block_circulant_diagonal_entries(a),
            side="left",
        )
        reducer_count = 0
        inner_structure = "sparse"
        law = "block_circulant_per_frequency_diagonal_sparse_matmul"
    elif inner_a == "sparse" and inner_b == "diagonal":
        spectrum = _scale_block_circulant_sparse(
            a,
            _block_circulant_diagonal_entries(b),
            side="right",
        )
        reducer_count = 0
        inner_structure = "sparse"
        law = "block_circulant_per_frequency_sparse_diagonal_matmul"
    elif inner_a == "diagonal":
        diag = _block_circulant_diagonal_entries(a)
        sb = _block_circulant_spectrum(b)
        spectrum = diag[:, :, None] * sb
        reducer_count = 0
        inner_structure = "dense"
        law = "block_circulant_per_frequency_diagonal_dense_matmul"
    elif inner_b == "diagonal":
        sa = _block_circulant_spectrum(a)
        diag = _block_circulant_diagonal_entries(b)
        spectrum = sa * diag[:, None, :]
        reducer_count = 0
        inner_structure = "dense"
        law = "block_circulant_per_frequency_dense_diagonal_matmul"
    else:
        sa = _block_circulant_spectrum(a)
        sb = _block_circulant_spectrum(b)
        spectrum = np.einsum("fom,fmi->foi", sa, sb)
        reducer_count = int(pa["frequency_count"]) * int(pa["out_blocks"]) * int(pb["in_blocks"])
        inner_structure = "dense"
        law = "block_circulant_per_frequency_matmul"
    return SPFBlockMatrix.block_circulant_spectrum(
        spectrum,
        out_axis=a.sig.out_axes[0],
        in_axis=b.sig.in_axes[0],
        block_axes=(
            _block_circulant_block_axes_from_params(pa)[0],
            _block_circulant_block_axes_from_params(pb)[1],
        ),
        inner_structure=inner_structure,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        proof=SPFProof(
            law,
            False,
            True,
            reducer_count,
            0,
            notes=("per-frequency small matrix products over the block axis",),
        ),
    )


def _validate_nonabelian_table(
    coords: np.ndarray,
    basis: np.ndarray,
    table: np.ndarray,
    *,
    exact_table: bool,
) -> None:
    if basis.ndim != 3 or basis.shape[1] != basis.shape[2]:
        raise SPFError("non-abelian basis must have shape (d, n, n)")
    d = int(basis.shape[0])
    if coords.shape != (d,):
        raise SPFError("non-abelian coords length must match basis count")
    if table.shape != (d, d, d):
        raise SPFError("non-abelian multiplication table must have shape (d, d, d)")
    if exact_table:
        one_hot = np.isclose(table, 1.0 + 0.0j, atol=1e-12)
        zero = np.isclose(table, 0.0 + 0.0j, atol=1e-12)
        if not np.all(one_hot | zero) or not np.all(np.sum(one_hot, axis=2) == 1):
            raise SPFError("exact non-abelian multiplication table must be one-hot closed")
        product = np.argmax(one_hot, axis=2).astype(np.int64, copy=False)
        for a in range(d):
            for b in range(d):
                for c in range(d):
                    if int(product[int(product[a, b]), c]) != int(product[a, int(product[b, c])]):
                        raise SPFError("exact non-abelian multiplication table must be associative")
        identities = [e for e in range(d) if all(int(product[e, g]) == g and int(product[g, e]) == g for g in range(d))]
        if not identities:
            raise SPFError("exact non-abelian multiplication table must have a two-sided identity")
        identity = int(identities[0])
        for g in range(d):
            if not any(int(product[g, h]) == identity and int(product[h, g]) == identity for h in range(d)):
                raise SPFError("exact non-abelian multiplication table must have two-sided inverses")


def _nonabelian_params(k: SPFKernel) -> Mapping[str, Any]:
    return k.expr.args[0]


def _nonabelian_components(k: SPFKernel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = _nonabelian_params(k)
    d = int(params["basis_count"])
    n = int(params["matrix_size"])
    coords = _complex_from_words(params["coords_words"]).reshape(d)
    basis = _complex_from_words(params["basis_words"]).reshape(d, n, n)
    table = _complex_from_words(params["table_words"]).reshape(d, d, d)
    return coords, basis, table


def _compose_nonabelian(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "NonAbelianCoefficientTable" or b.expr.op != "NonAbelianCoefficientTable":
        return None
    ca, ba, ta = _nonabelian_components(a)
    cb, bb, tb = _nonabelian_components(b)
    if ca.shape != cb.shape or ba.shape != bb.shape or ta.shape != tb.shape:
        raise SPFError("non-abelian coefficient-table shapes must match")
    if not np.allclose(ba, bb, atol=2e-2, rtol=2e-2):
        raise SPFError("non-abelian bases must match")
    if not np.allclose(ta, tb, atol=2e-2, rtol=2e-2):
        raise SPFError("non-abelian multiplication tables must match")
    exact_a = bool(_nonabelian_params(a).get("exact_table"))
    exact_b = bool(_nonabelian_params(b).get("exact_table"))
    if exact_a != exact_b:
        raise SPFError("non-abelian exactness claims must match")
    exact_table = exact_a and exact_b
    coords = np.einsum("a,b,abc->c", ca, cb, ta)
    d = int(ca.size)
    reducer_count = d
    return SPFBlockMatrix.nonabelian_coefficients(
        coords,
        ba,
        ta,
        out_axis=a.sig.out_axes[0],
        in_axis=b.sig.in_axes[0],
        exact_table=exact_table,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        proof=SPFProof(
            "nonabelian_coefficient_table_compose",
            exact_table,
            True,
            reducer_count,
            0,
            notes=(
                f"each output coordinate reduces {d * d} coefficient products",
                "exact_finite_group_table" if exact_table else "fitted_approximate_operator_basis",
            ),
        ),
    )


def _require_square_block_circulant(a: SPFKernel) -> np.ndarray:
    spectrum = _block_circulant_spectrum(a)
    if spectrum.shape[1] != spectrum.shape[2]:
        raise SPFError("block-circulant inverse/power requires square per-frequency matrices")
    return spectrum


def _invert_block_circulant_spectrum(spectrum: np.ndarray) -> np.ndarray:
    out = np.empty_like(spectrum)
    for f, mat in enumerate(spectrum):
        if np.linalg.svd(mat, compute_uv=False)[-1] <= _SPECTRAL_INVERTIBILITY_EPS:
            raise SPFError("cannot invert block-circulant with singular frequency block")
        out[f] = np.linalg.inv(mat)
    return out


def _adjoint_block_circulant(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "BlockCirculantSpectrum":
        return None
    inner_structure = str(_block_circulant_params(a).get("inner_structure", "dense"))
    if inner_structure == "diagonal":
        spectrum = np.conj(_block_circulant_diagonal_entries(a))
        out_inner_structure = "diagonal"
    elif inner_structure == "low_rank":
        u, v, coeff = _block_circulant_low_rank_factors(a)
        spectrum = (v, u, np.conj(coeff))
        out_inner_structure = "low_rank"
    elif inner_structure == "sparse":
        indices, values = _block_circulant_sparse_entries(a)
        flipped = indices.copy()
        flipped[:, 1], flipped[:, 2] = indices[:, 2], indices[:, 1]
        params = _block_circulant_params(a)
        spectrum = {
            "shape": (
                int(params["frequency_count"]),
                int(params["in_blocks"]),
                int(params["out_blocks"]),
            ),
            "indices": flipped,
            "values": np.conj(values),
        }
        out_inner_structure = "sparse"
    elif inner_structure == "character":
        n, alpha, beta, scale = _block_circulant_character_factors(a)
        spectrum = {
            "N": n,
            "alpha": -beta,
            "beta": -alpha,
            "scale": np.conj(scale),
        }
        out_inner_structure = "character"
    else:
        spectrum = np.swapaxes(np.conj(_block_circulant_spectrum(a)), 1, 2)
        out_inner_structure = "dense"
    return SPFBlockMatrix.block_circulant_spectrum(
        spectrum,
        out_axis=a.sig.in_axes[0],
        in_axis=a.sig.out_axes[0],
        block_axes=(
            _block_circulant_block_axes_from_params(_block_circulant_params(a))[1],
            _block_circulant_block_axes_from_params(_block_circulant_params(a))[0],
        ),
        inner_structure=out_inner_structure,
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof("block_circulant_adjoint", False, True, 0, 0),
    )


def _inverse_block_circulant(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "BlockCirculantSpectrum":
        return None
    inner_structure = str(_block_circulant_params(a).get("inner_structure", "dense"))
    if inner_structure == "diagonal":
        diag = _block_circulant_diagonal_entries(a)
        if np.any(np.abs(diag) <= _SPECTRAL_INVERTIBILITY_EPS):
            raise SPFError("cannot invert block-circulant with singular frequency block")
        spectrum = 1.0 / diag
        out_inner_structure = "diagonal"
    else:
        spectrum = _invert_block_circulant_spectrum(_require_square_block_circulant(a))
        out_inner_structure = "dense"
    return SPFBlockMatrix.block_circulant_spectrum(
        spectrum,
        out_axis=a.sig.out_axes[0],
        in_axis=a.sig.in_axes[0],
        block_axes=_block_circulant_block_axes_from_params(_block_circulant_params(a)),
        inner_structure=out_inner_structure,
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof("block_circulant_inverse", False, True, 0, 0),
    )


def _power_block_circulant(a: SPFKernel, k: int) -> SPFKernel | None:
    if a.expr.op != "BlockCirculantSpectrum":
        return None
    inner_structure = str(_block_circulant_params(a).get("inner_structure", "dense"))
    if inner_structure == "diagonal":
        diag = _block_circulant_diagonal_entries(a)
        if int(k) < 0:
            if np.any(np.abs(diag) <= _SPECTRAL_INVERTIBILITY_EPS):
                raise SPFError("cannot invert block-circulant with singular frequency block")
            diag = 1.0 / diag
            exponent = -int(k)
        else:
            exponent = int(k)
        powered = np.ones_like(diag) if exponent == 0 else diag**exponent
        spectrum = powered
        out_inner_structure = "diagonal"
    else:
        spectrum = _require_square_block_circulant(a)
        if int(k) < 0:
            spectrum = _invert_block_circulant_spectrum(spectrum)
            exponent = -int(k)
        else:
            exponent = int(k)
        if exponent == 0:
            powered = np.broadcast_to(np.eye(spectrum.shape[1], dtype=np.complex128), spectrum.shape).copy()
        else:
            powered = np.stack([np.linalg.matrix_power(mat, exponent) for mat in spectrum], axis=0)
        spectrum = powered
        out_inner_structure = "dense"
    return SPFBlockMatrix.block_circulant_spectrum(
        spectrum,
        out_axis=a.sig.out_axes[0],
        in_axis=a.sig.in_axes[0],
        block_axes=_block_circulant_block_axes_from_params(_block_circulant_params(a)),
        inner_structure=out_inner_structure,
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof("block_circulant_power", False, True, 0, 0, notes=(f"k={int(k)}",)),
    )


def _diagonal_word_result(
    words: np.ndarray, out: AxisSpec, inn: AxisSpec, *, exactness: str,
    reducer: ReducerSpec | None, proof: SPFProof | None = None,
) -> SPFKernel:
    if out.name == inn.name:
        inn = replace(inn, name=f"{inn.name}__in")
    result = SPFBlockMatrix.diagonal(words, out_axis=out.name, axis=inn.name, basis=out.basis,
                                     exactness=exactness, reducer_policy=reducer, proof=proof)
    return replace(result, axes=(replace(out, role="codomain"), replace(inn, role="domain")))


def _compose_diagonal(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "DiagonalSPF" and b.expr.op == "DiagonalSPF":
        aw = a.expr.args[0]["diag_words"]
        bw = b.expr.args[0]["diag_words"]
        if np.any(unpack_L(aw) == SPF32_L_NAN) or np.any(unpack_L(bw) == SPF32_L_NAN):
            raise SPFError("cannot compose diagonal with non-finite entry")
        words = spf_mul_words(aw, bw)
        rates = unpack_L(aw).astype(np.int64) + unpack_L(bw).astype(np.int64)
        finite = (unpack_L(aw) != SPF32_L_ZERO) & (unpack_L(bw) != SPF32_L_ZERO)
        finite &= (unpack_L(aw) != SPF32_L_NAN) & (unpack_L(bw) != SPF32_L_NAN)
        in_range = not np.any(finite & ((rates < SPF32_L_MIN_FINITE) | (rates > SPF32_L_MAX_FINITE)))
        exact = bool(in_range and a.sig.exactness == b.sig.exactness == "exact")
        exactness = "exact" if exact else "quantized"
        if "approximate" in {a.sig.exactness, b.sig.exactness}:
            exactness = "approximate"
        return _diagonal_word_result(
            words, _axis_map(a.axes)[a.sig.out_axes[0]], _axis_map(b.axes)[b.sig.in_axes[0]],
            exactness=exactness,
            reducer=a.sig.reducer_policy or b.sig.reducer_policy,
            proof=SPFProof("diagonal_word_compose", exact, True, 0, 0,
                           notes=("coefficient_product; packed application boundaries remain distinct",)),
        )
    if a.expr.op == "DiagonalSPF" and b.expr.op == "LowRankOuter":
        diag = _complex_from_words(a.expr.args[0]["diag_words"]).reshape(-1)
        params = b.expr.args[0]
        u = _complex_from_words(params["u_words"])
        v = _complex_from_words(params["v_words"])
        coeff = _outer_coefficients(params)
        return SPFBlockMatrix.low_rank(
            diag[:, None] * u,
            v,
            coeff,
            out_axis=a.sig.out_axes[0],
            in_axis=b.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "LowRankOuter" and b.expr.op == "DiagonalSPF":
        params = a.expr.args[0]
        u = _complex_from_words(params["u_words"])
        v = _complex_from_words(params["v_words"])
        coeff = _outer_coefficients(params)
        diag = _complex_from_words(b.expr.args[0]["diag_words"]).reshape(-1)
        return SPFBlockMatrix.low_rank(
            u,
            np.conj(diag)[:, None] * v,
            coeff,
            out_axis=a.sig.out_axes[0],
            in_axis=b.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    return None


def _compose_permutation(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "Permutation" and b.expr.op == "Permutation":
        ia = np.asarray(a.expr.args[0]["indices"], dtype=np.int64)
        ib = np.asarray(b.expr.args[0]["indices"], dtype=np.int64)
        return SPFBlockMatrix.permutation(
            ib[ia],
            out_axis=a.sig.out_axes[0],
            axis=b.sig.in_axes[0],
            basis=str(a.expr.args[0].get("basis", "standard")),
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "Permutation" and b.expr.op == "LowRankOuter":
        idx = np.asarray(a.expr.args[0]["indices"], dtype=np.int64)
        params = b.expr.args[0]
        return SPFBlockMatrix.low_rank(
            _complex_from_words(params["u_words"])[idx],
            _complex_from_words(params["v_words"]),
            _outer_coefficients(params),
            out_axis=a.sig.out_axes[0],
            in_axis=b.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "LowRankOuter" and b.expr.op == "Permutation":
        idx = np.asarray(b.expr.args[0]["indices"], dtype=np.int64)
        inv = np.empty_like(idx)
        inv[idx] = np.arange(idx.size)
        params = a.expr.args[0]
        return SPFBlockMatrix.low_rank(
            _complex_from_words(params["u_words"]),
            _complex_from_words(params["v_words"])[inv],
            _outer_coefficients(params),
            out_axis=a.sig.out_axes[0],
            in_axis=b.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    return None


def _compose_roll(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "Roll" and b.expr.op == "Roll":
        pa = a.expr.args[0]
        pb = b.expr.args[0]
        if int(pa["n"]) != int(pb["n"]):
            raise SPFError("roll sizes must match")
        return SPFBlockMatrix.roll(
            int(pa["n"]),
            int(pa["shift"]) + int(pb["shift"]),
            out_axis=a.sig.out_axes[0],
            axis=b.sig.in_axes[0],
            basis=str(pa.get("basis", "standard")),
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "Roll" and b.expr.op == "CirculantSpectrum":
        assert b.expr.block is not None
        n = int(b.expr.block.dims[0])
        spectrum = block_to_complex(b.expr.block).reshape(-1) * _roll_spectrum(n, int(a.expr.args[0]["shift"]))
        return SPFBlockMatrix.circulant_spectrum(
            spectrum,
            out_axis=a.sig.out_axes[0],
            axis=b.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    if a.expr.op == "CirculantSpectrum" and b.expr.op == "Roll":
        assert a.expr.block is not None
        n = int(a.expr.block.dims[0])
        spectrum = block_to_complex(a.expr.block).reshape(-1) * _roll_spectrum(n, int(b.expr.args[0]["shift"]))
        return SPFBlockMatrix.circulant_spectrum(
            spectrum,
            out_axis=a.sig.out_axes[0],
            axis=b.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        )
    return None


def _character_params(k: SPFKernel) -> Mapping[str, Any]:
    return k.expr.args[0]


def _compose_character(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "Character" or b.expr.op != "Character":
        return None
    pa = _character_params(a)
    pb = _character_params(b)
    if int(pa["N"]) != int(pb["N"]):
        raise SPFError("character moduli must match")
    n = int(pa["N"])
    if not bool(pa.get("exact_group_semantics")) or not bool(pb.get("exact_group_semantics")):
        return None
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    if (int(pa["beta"]) + int(pb["alpha"])) % n != 0:
        return SPFKernel(
            _closed_form_zero_expr(
                (out_axis.size, in_axis.size),
                SPFProof("character_orthogonality_zero", True, True, 0, 0),
            ),
            (out_axis, in_axis),
            KernelSignature(
                (out_axis.name,),
                (in_axis.name,),
                exactness="exact",
                reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
            ),
        )
    return SPFBlockMatrix.character(
        N=n,
        alpha=int(pa["alpha"]),
        beta=int(pb["beta"]),
        out_axis=out_axis.name,
        in_axis=in_axis.name,
        scale=complex(pa.get("scale", 1.0 + 0.0j)) * complex(pb.get("scale", 1.0 + 0.0j)) * n,
        exact_group_semantics=True,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        proof=SPFProof(
            "character_compose_orthogonality",
            True,
            True,
            0,
            0,
            notes=(f"N={n}", "middle_frequency_cancelled"),
        ),
    )


def _compose_product_character(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "ProductGroupCharacter" or b.expr.op != "ProductGroupCharacter":
        return None
    pa = _character_params(a)
    pb = _character_params(b)
    moduli = tuple(int(m) for m in pa["moduli"])
    if moduli != tuple(int(m) for m in pb["moduli"]):
        raise SPFError("product character moduli must match")
    if not bool(pa.get("exact_group_semantics")) or not bool(pb.get("exact_group_semantics")):
        return None
    lhs = tuple(int(v) for v in pa["in_frequency"])
    rhs = tuple(int(v) for v in pb["out_frequency"])
    cancels = all((x + y) % m == 0 for x, y, m in zip(lhs, rhs, moduli, strict=True))
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    if not cancels:
        return SPFKernel(
            _closed_form_zero_expr(
                (out_axis.size, in_axis.size),
                SPFProof("product_group_orthogonality_zero", True, True, 0, 0),
            ),
            (out_axis, in_axis),
            KernelSignature(
                (out_axis.name,),
                (in_axis.name,),
                exactness="exact",
                reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
            ),
        )
    return SPFBlockMatrix.product_character(
        moduli=moduli,
        out_frequency=pa["out_frequency"],
        in_frequency=pb["in_frequency"],
        out_axis=out_axis.name,
        in_axis=in_axis.name,
        scale=complex(pa.get("scale", 1.0 + 0.0j))
        * complex(pb.get("scale", 1.0 + 0.0j))
        * int(np.prod(moduli, dtype=np.int64)),
        exact_group_semantics=True,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        proof=SPFProof(
            "product_group_compose_orthogonality",
            True,
            True,
            0,
            0,
            notes=("indexing=no_carry", "middle_frequency_cancelled"),
        ),
    )


def _encode_outer_coefficients(values: np.ndarray) -> dict[str, Any]:
    """Keep out-of-range Gram coefficients in SPF words plus exponent metadata."""
    magnitude = np.abs(values)
    finite = np.isfinite(magnitude) & (magnitude > 0)
    codes = np.zeros(magnitude.shape)
    codes[finite] = np.rint(np.log2(magnitude[finite]) * 256)
    if not np.any(finite & ((codes < SPF32_L_MIN_FINITE) | (codes > SPF32_L_MAX_FINITE))):
        return {"coeff_words": encode_words(values)}
    scales = np.where(finite, np.floor(codes / 256), 0).astype(np.int32)
    normalized = np.ldexp(values.real, -scales) + 1j * np.ldexp(values.imag, -scales)
    return {"coeff_words": encode_words(normalized), "coeff_log2_scale": scales}


def _outer_coefficients(params: Mapping[str, Any]) -> np.ndarray:
    values = _complex_from_words(params["coeff_words"])
    scales = params.get("coeff_log2_scale")
    if scales is None:
        return values
    return np.ldexp(values.real, scales) + 1j * np.ldexp(values.imag, scales)


def _compose_low_rank(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "LowRankOuter" and len(b.sig.out_axes) == 1 and not b.sig.in_axes:
        pa = a.expr.args[0]
        u = _complex_from_words(pa["u_words"])
        v = _complex_from_words(pa["v_words"])
        coeff = _complex_from_words(pa["coeff_words"])
        out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
        reducer = a.sig.reducer_policy or b.sig.reducer_policy or _default_reducer()
        rank = int(coeff.size)
        out_size = int(u.shape[0])
        return SPFKernel(
            SPFBlockExpr(
                "LowRankApply",
                args=(
                    {
                        "u_factors": encode_words(u.T),
                        "v_factors": encode_words(v.T),
                        "lambdas": encode_words(coeff),
                        "vector": b,
                        "reducer": reducer,
                    },
                ),
                proof=SPFProof(
                    "low_rank_apply_plan",
                    False,
                    False,
                    rank + out_size,
                    0,
                    notes=(f"rank={rank}", f"out_dim={out_size}"),
                ),
            ),
            (out_axis,),
            KernelSignature((out_axis.name,), (), reducer_policy=reducer),
        )
    if a.expr.op != "LowRankOuter" or b.expr.op != "LowRankOuter":
        return None
    pa = a.expr.args[0]
    pb = b.expr.args[0]
    va = _complex_from_words(pa["v_words"])
    ca = _complex_from_words(pa["coeff_words"])
    ub = _complex_from_words(pb["u_words"])
    cb = _complex_from_words(pb["coeff_words"])
    gram = np.conj(va).T @ ub
    coeff = ca[:, None] * gram * cb[None, :]
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    reducer_count = int(ca.size * cb.size)
    if np.all(coeff == 0):
        return SPFKernel(
            _closed_form_zero_expr(
                (out_axis.size, in_axis.size),
                SPFProof(
                    "low_rank_zero_gram_compose",
                    False,
                    False,
                    reducer_count,
                    0,
                    error_bound_l=0.0,
                    notes=(f"zero_gram_entries={reducer_count}",),
                ),
            ),
            (out_axis, in_axis),
            KernelSignature(
                (out_axis.name,),
                (in_axis.name,),
                reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy or _default_reducer(),
            ),
        )
    params = {"left_words": pa["u_words"], **_encode_outer_coefficients(coeff), "right_words": pb["v_words"]}
    return SPFKernel(
        SPFBlockExpr(
            "LazyOuterSum",
            args=(params,),
            proof=SPFProof("low_rank_lazy_outer_sum", False, False, reducer_count, 0),
        ),
        (out_axis, in_axis),
        KernelSignature(
            (out_axis.name,),
            (in_axis.name,),
            reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy or _default_reducer(),
        ),
    )


def _compose_lazy_outer_sum(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "LazyOuterSum" or len(b.sig.out_axes) != 1 or b.sig.in_axes:
        return None
    params = a.expr.args[0]
    if "coeff_log2_scale" in params:
        out = _axis_map(a.axes)[a.sig.out_axes[0]]
        reducer = a.sig.reducer_policy or b.sig.reducer_policy or _default_reducer()
        return SPFKernel(
            SPFBlockExpr(
                "WideLowRankApply",
                args=({**params, "vector": b, "reducer": reducer},),
                proof=SPFProof(
                    "scaled_outer_sum_apply",
                    False,
                    False,
                    a.reducer_budget() + out.size,
                    1,
                    notes=("wide_coefficient_exponent_metadata",),
                ),
            ),
            (out,),
            KernelSignature((out.name,), (), reducer_policy=reducer),
        )
    left = _complex_from_words(params["left_words"])
    coeff = _outer_coefficients(params)
    right = _complex_from_words(params["right_words"])
    if coeff.shape != (left.shape[1], right.shape[1]):
        raise SPFError("LazyOuterSum coefficient shape must match left/right ranks")
    rank_left, rank_right = (int(x) for x in coeff.shape)
    flat_rank = rank_left * rank_right
    flat_coeff = coeff.reshape(flat_rank)
    nonzero = flat_coeff != 0
    live_rank = int(np.count_nonzero(nonzero))
    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    reducer = a.sig.reducer_policy or b.sig.reducer_policy or _default_reducer()
    u_factors = np.repeat(left.T, rank_right, axis=0)[nonzero]
    v_factors = np.tile(right.T, (rank_left, 1))[nonzero]
    lambdas = flat_coeff[nonzero]
    notes = (f"flat_rank={live_rank}",)
    skipped = flat_rank - live_rank
    if skipped:
        notes = (*notes, f"skipped_zero_terms={skipped}")
    return SPFKernel(
        SPFBlockExpr(
            "LowRankApply",
            args=(
                {
                    "u_factors": encode_words(u_factors),
                    "v_factors": encode_words(v_factors),
                    "lambdas": encode_words(lambdas),
                    "vector": b,
                    "reducer": reducer,
                },
            ),
            proof=SPFProof(
                "lazy_outer_sum_low_rank_apply_plan",
                False,
                False,
                live_rank + int(out_axis.size),
                0,
                notes=notes,
            ),
        ),
        (out_axis,),
        KernelSignature((out_axis.name,), (), reducer_policy=reducer),
    )


def _adjoint_lazy_outer_sum(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "LazyOuterSum":
        return None
    params = a.expr.args[0]
    coeff = _outer_coefficients(params)
    out_axis = _axis_map(a.axes)[a.sig.in_axes[0]]
    in_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    return SPFKernel(
        SPFBlockExpr(
            "LazyOuterSum",
            args=(
                {
                    "left_words": params["right_words"],
                    **_encode_outer_coefficients(np.conj(coeff.T)),
                    "right_words": params["left_words"],
                },
            ),
            proof=SPFProof(
                "lazy_outer_sum_adjoint",
                False,
                False,
                a.reducer_budget(),
                0,
            ),
        ),
        (out_axis, in_axis),
        KernelSignature((out_axis.name,), (in_axis.name,), reducer_policy=a.sig.reducer_policy),
    )


def _recompress_low_rank(
    k: SPFKernel,
    *,
    max_rank: int,
    max_rel_error: float | None,
    drop_tol: float,
) -> SPFKernel:
    if k.expr.op not in {"LazyOuterSum", "LowRankOuter"}:
        raise SPFError(f"low-rank recompression requires LazyOuterSum or LowRankOuter, got {k.expr.op!r}")
    rank_budget = int(max_rank)
    if rank_budget < 0:
        raise SPFError("max_rank must be non-negative")
    if float(drop_tol) < 0.0:
        raise SPFError("drop_tol must be non-negative")
    if max_rel_error is not None and float(max_rel_error) < 0.0:
        raise SPFError("max_rel_error must be non-negative")

    dense = _complex_from_words(k.materialize_words())
    if dense.ndim != 2:
        raise SPFError("low-rank recompression requires matrix-shaped kernel")
    norm = float(np.linalg.norm(dense))
    if rank_budget == 0:
        rel_error = 0.0 if norm == 0.0 else 1.0
        if max_rel_error is None or rel_error <= float(max_rel_error):
            return _closed_form_zero_like(k, "low_rank_recompress_zero")
        raise SPFError("recompression rank budget cannot satisfy error bound")

    u_svd, s, vh = np.linalg.svd(dense, full_matrices=False)
    keep_by_tol = s > float(drop_tol)
    keep_count = min(rank_budget, int(np.sum(keep_by_tol)))
    if keep_count == 0:
        rel_error = 0.0 if norm == 0.0 else 1.0
        if max_rel_error is None or rel_error <= float(max_rel_error):
            return _closed_form_zero_like(k, "low_rank_recompress_zero")
        raise SPFError("recompression rank budget cannot satisfy error bound")

    residual = s[keep_count:]
    rel_error = float(np.linalg.norm(residual) / norm) if norm > 0.0 else 0.0
    if max_rel_error is not None and rel_error > float(max_rel_error):
        raise SPFError("recompression rank budget cannot satisfy error bound")

    left = u_svd[:, :keep_count]
    right = np.conj(vh[:keep_count, :].T)
    out_axis = _axis_map(k.axes)[k.sig.out_axes[0]]
    in_axis = _axis_map(k.axes)[k.sig.in_axes[0]]
    return SPFBlockMatrix.low_rank(
        left,
        right,
        s[:keep_count].astype(np.complex128),
        out_axis=out_axis.name,
        in_axis=in_axis.name,
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "low_rank_svd_recompression",
            False,
            False,
            0,
            0,
            error_bound_l=rel_error,
            notes=(
                f"requested_max_rank={rank_budget}",
                f"kept_rank={keep_count}",
                f"relative_frobenius_error={rel_error:.6g}",
            ),
        ),
    )


def _nearest_circulant_spectrum(dense: np.ndarray) -> np.ndarray:
    mat = np.asarray(dense, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise SPFError("circulant recompression requires a square matrix")
    n = int(mat.shape[0])
    if n == 0:
        raise SPFError("circulant recompression requires a non-empty matrix")
    cols = np.arange(n, dtype=np.int64)
    kernel = np.empty(n, dtype=np.complex128)
    for delta in range(n):
        kernel[delta] = np.mean(mat[(cols + delta) % n, cols])
    return np.fft.fft(kernel)


def _recompress_circulant(
    k: SPFKernel,
    *,
    max_rel_error: float | None,
    split_residual: bool,
) -> SPFKernel:
    if max_rel_error is not None and float(max_rel_error) < 0.0:
        raise SPFError("max_rel_error must be non-negative")
    if len(k.sig.batch_axes) != 0 or len(k.sig.out_axes) != 1 or len(k.sig.in_axes) != 1:
        raise SPFError("circulant recompression requires an unbatched rank-2 kernel")
    dense = _complex_from_words(k.materialize_words())
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise SPFError("circulant recompression requires a square matrix")

    out_axis = _axis_map(k.axes)[k.sig.out_axes[0]]
    in_axis = _axis_map(k.axes)[k.sig.in_axes[0]]
    spectrum = _nearest_circulant_spectrum(dense)
    fit = SPFBlockMatrix.circulant_spectrum(
        spectrum,
        out_axis=out_axis.name,
        axis=in_axis.name,
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "circulant_recompression_fit",
            False,
            False,
            0,
            0,
            notes=(f"size={int(dense.shape[0])}",),
        ),
    )
    fit_dense = _complex_from_words(fit.materialize_words())
    norm = float(np.linalg.norm(dense))
    rel_error = float(np.linalg.norm(dense - fit_dense) / norm) if norm > 0.0 else 0.0
    fit = SPFBlockMatrix.circulant_spectrum(
        spectrum,
        out_axis=out_axis.name,
        axis=in_axis.name,
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "circulant_recompression_fit",
            False,
            False,
            0,
            0,
            error_bound_l=rel_error,
            notes=(f"size={int(dense.shape[0])}",),
        ),
    )
    if max_rel_error is None or rel_error <= float(max_rel_error):
        return fit
    if not split_residual:
        raise SPFError("circulant recompression error bound cannot be satisfied")

    base_dense = _complex_from_words(fit.materialize_words())
    fit_out_axis = _axis_map(fit.axes)[fit.sig.out_axes[0]]
    fit_in_axis = _axis_map(fit.axes)[fit.sig.in_axes[0]]
    patch = SPFBlockMatrix.dense(
        dense - base_dense,
        out_axes=(fit_out_axis,),
        in_axes=(fit_in_axis,),
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "circulant_recompression_residual_words",
            False,
            False,
            0,
            0,
            notes=(f"base_rel_error={rel_error}",),
        ),
    )
    return SPFBlockMatrix.residual_patch(
        fit,
        patch,
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "circulant_residual_split",
            False,
            False,
            int(dense.shape[0] * dense.shape[1]),
            0,
            error_bound_l=0.0,
            notes=(
                f"base_rel_error={rel_error}",
                "base plus residual patch is an explicit additive boundary",
            ),
        ),
    )


def _best_character_fit(dense: np.ndarray) -> tuple[int, int, complex]:
    mat = np.asarray(dense, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise SPFError("character recompression requires a square matrix")
    n = int(mat.shape[0])
    if n == 0:
        raise SPFError("character recompression requires a non-empty matrix")
    rows = np.arange(n, dtype=np.float64)
    cols = np.arange(n, dtype=np.float64)
    best_alpha = 0
    best_beta = 0
    best_scale = 0.0 + 0.0j
    best_error = float("inf")
    norm = max(float(np.linalg.norm(mat)), 1.0)
    for alpha in range(n):
        row_phase = alpha * rows / n
        for beta in range(n):
            phi = np.exp(2j * np.pi * (row_phase[:, None] + (beta * cols / n)[None, :]))
            scale = complex(np.vdot(phi, mat) / np.vdot(phi, phi))
            error = float(np.linalg.norm(mat - scale * phi) / norm)
            if error < best_error:
                best_alpha = alpha
                best_beta = beta
                best_scale = scale
                best_error = error
    return best_alpha, best_beta, best_scale


def _recompress_character(k: SPFKernel, *, max_rel_error: float | None) -> SPFKernel:
    if max_rel_error is not None and float(max_rel_error) < 0.0:
        raise SPFError("max_rel_error must be non-negative")
    if len(k.sig.batch_axes) != 0 or len(k.sig.out_axes) != 1 or len(k.sig.in_axes) != 1:
        raise SPFError("character recompression requires an unbatched rank-2 kernel")
    dense = _complex_from_words(k.materialize_words())
    if dense.ndim != 2 or dense.shape[0] != dense.shape[1]:
        raise SPFError("character recompression requires a square matrix")

    out_axis = _axis_map(k.axes)[k.sig.out_axes[0]]
    in_axis = _axis_map(k.axes)[k.sig.in_axes[0]]
    alpha, beta, scale = _best_character_fit(dense)
    fit = SPFBlockMatrix.character(
        N=int(dense.shape[0]),
        alpha=alpha,
        beta=beta,
        out_axis=out_axis.name,
        in_axis=in_axis.name,
        scale=scale,
        exact_group_semantics=True,
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "character_recompression_fit",
            False,
            False,
            0,
            0,
            notes=(
                f"N={int(dense.shape[0])}",
                f"alpha={alpha}",
                f"beta={beta}",
            ),
        ),
    )
    fit_dense = _complex_from_words(fit.materialize_words())
    norm = float(np.linalg.norm(dense))
    rel_error = float(np.linalg.norm(dense - fit_dense) / norm) if norm > 0.0 else 0.0
    if max_rel_error is not None and rel_error > float(max_rel_error):
        raise SPFError("character recompression error bound cannot be satisfied")
    return SPFBlockMatrix.character(
        N=int(dense.shape[0]),
        alpha=alpha,
        beta=beta,
        out_axis=out_axis.name,
        in_axis=in_axis.name,
        scale=scale,
        exact_group_semantics=True,
        reducer_policy=k.sig.reducer_policy,
        proof=SPFProof(
            "character_recompression_fit",
            False,
            False,
            0,
            0,
            error_bound_l=rel_error,
            notes=(
                f"N={int(dense.shape[0])}",
                f"alpha={alpha}",
                f"beta={beta}",
            ),
        ),
    )


def _validate_and_encode_tt_cores(cores: Sequence[np.ndarray]) -> tuple[tuple[np.ndarray, ...], int, int]:
    encoded: list[np.ndarray] = []
    out_size = 1
    in_size = 1
    previous_right_rank: int | None = None
    for idx, core in enumerate(cores):
        arr = _as_complex_values(core)
        if arr.ndim != 4:
            raise SPFError("TT cores must have shape (left_rank, out_dim, in_dim, right_rank)")
        left_rank, out_dim, in_dim, right_rank = (int(x) for x in arr.shape)
        if out_dim <= 0 or in_dim <= 0:
            raise SPFError("TT core physical dimensions must be positive")
        if left_rank <= 0 or right_rank <= 0:
            raise SPFError("TT core ranks must be positive")
        if idx == 0 and left_rank != 1:
            raise SPFError("first TT core left rank must be 1")
        if previous_right_rank is not None and left_rank != previous_right_rank:
            raise SPFError("adjacent TT core ranks must match")
        previous_right_rank = right_rank
        out_size *= out_dim
        in_size *= in_dim
        encoded.append(_as_spf_words(core).reshape(arr.shape))
    if not encoded:
        raise SPFError("tt_structured requires at least one core")
    last_right_rank = int(np.asarray(cores[-1]).shape[3])
    if last_right_rank != 1:
        raise SPFError("last TT core right rank must be 1")
    return tuple(encoded), out_size, in_size


def _materialize_tt_cores(core_words: Sequence[np.ndarray]) -> np.ndarray:
    cores = tuple(_complex_from_words(words) for words in core_words)
    if not cores:
        raise SPFError("TTStructured requires at least one core")
    first = cores[0]
    acc = first[0, :, :, :]
    acc = acc.reshape(int(first.shape[1]), int(first.shape[2]), int(first.shape[3]))
    for core in cores[1:]:
        joined = np.einsum("abr,rois->aobis", acc, core)
        acc = joined.reshape(
            int(acc.shape[0] * core.shape[1]),
            int(acc.shape[1] * core.shape[2]),
            int(core.shape[3]),
        )
    if acc.shape[2] != 1:
        raise SPFError("TTStructured final rank must be 1")
    return acc[:, :, 0]


def _closed_form_zero_like(k: SPFKernel, law: str) -> SPFKernel:
    out_axis = _axis_map(k.axes)[k.sig.out_axes[0]]
    in_axis = _axis_map(k.axes)[k.sig.in_axes[0]]
    return SPFKernel(
        _closed_form_zero_expr(
            (out_axis.size, in_axis.size),
            SPFProof(law, True, True, 0, 0, error_bound_l=0.0),
        ),
        (out_axis, in_axis),
        KernelSignature((out_axis.name,), (in_axis.name,), reducer_policy=k.sig.reducer_policy),
    )


def _closed_form_zero_for_compose(a: SPFKernel, b: SPFKernel, law: str) -> SPFKernel:
    batch_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes)
    out_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes)
    in_axes = tuple(_axis_map(b.axes)[name] for name in b.sig.in_axes)
    return SPFBlockMatrix.zero(
        batch_axes=batch_axes,
        out_axes=out_axes,
        in_axes=in_axes,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        proof=SPFProof(law, True, True, 0, 0, error_bound_l=0.0),
    )


def _compose_closed_form_zero(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "ClosedFormZero" or b.expr.op == "ClosedFormZero":
        return _closed_form_zero_for_compose(a, b, "closed_form_zero_absorbing_compose")
    return None


def _compose_closed_form_identity(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op == "ClosedFormIdentity":
        if len(a.sig.out_axes) != 1 or len(a.sig.in_axes) != 1 or len(b.sig.out_axes) != 1:
            return None
        return b.pullback(
            {b.sig.out_axes[0]: a.sig.out_axes[0]},
            out_axes=a.sig.out_axes,
            in_axes=b.sig.in_axes,
        )
    if b.expr.op == "ClosedFormIdentity":
        if len(a.sig.in_axes) != 1 or len(b.sig.out_axes) != 1 or len(b.sig.in_axes) != 1:
            return None
        return a.pullback(
            {a.sig.in_axes[0]: b.sig.in_axes[0]},
            out_axes=a.sig.out_axes,
            in_axes=b.sig.in_axes,
        )
    return None


def _adjoint_lazy_sum(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "LazySum":
        return None
    terms, reducer = a.expr.args
    return lazy_sum(tuple(term.H() for term in terms), reducer=reducer)


def _adjoint_lazy_hadamard(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "LazyHadamard":
        return None
    lhs, rhs = a.expr.args
    batch_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes)
    out_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.in_axes)
    in_axes = tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes)
    return SPFKernel(
        SPFBlockExpr(
            "LazyHadamard",
            args=(lhs.H(), rhs.H()),
            proof=SPFProof(
                "lazy_hadamard_adjoint",
                False,
                False,
                0,
                0,
                notes=("adjoint distributed over elementwise SPF bind",),
            ),
        ),
        batch_axes + out_axes + in_axes,
        KernelSignature(
            tuple(axis.name for axis in out_axes),
            tuple(axis.name for axis in in_axes),
            batch_axes=tuple(axis.name for axis in batch_axes),
            basis_in=a.sig.basis_out,
            basis_out=a.sig.basis_in,
            reducer_policy=a.sig.reducer_policy,
        ),
    )


def _adjoint_residual_patch(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "ResidualPatch":
        return None
    params = a.expr.args[0]
    base_h = params["base"].H()
    patch_h = params["patch"].H()
    out = SPFBlockMatrix.residual_patch(base_h, patch_h, reducer_policy=params["reducer"])
    return SPFKernel(
        SPFBlockExpr(
            "ResidualPatch",
            args=out.expr.args,
            proof=SPFProof(
                "residual_patch_adjoint",
                False,
                False,
                out.reducer_budget(),
                0,
                notes=("adjoint distributed over base plus residual patch",),
            ),
        ),
        out.axes,
        out.sig,
    )


def _adjoint_jacobian_operator(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "JacobianOperator":
        return None
    params = a.expr.args[0]
    matrix = np.conj(_complex_from_words(params["matrix_words"]).T)
    residual_words = params.get("residual_words")
    residual = None if residual_words is None else np.conj(_complex_from_words(residual_words).T)
    provenance = dict(params.get("provenance", {}))
    provenance["adjoint_of"] = "jacobian_operator"
    return SPFBlockMatrix.jacobian_operator(
        matrix,
        out_axis=a.sig.in_axes[0],
        in_axis=a.sig.out_axes[0],
        provenance=provenance,
        residual=residual,
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof(
            "jacobian_operator_adjoint",
            False,
            False,
            a.reducer_budget(),
            0,
            notes=("adjoint preserves the local linear Jacobian representation",),
        ),
    )


def _adjoint_tt_structured(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "TTStructured":
        return None
    cores = tuple(np.swapaxes(np.conj(_complex_from_words(words)), 1, 2) for words in a.expr.args[0]["core_words"])
    return SPFBlockMatrix.tt_structured(
        cores,
        out_axis=a.sig.in_axes[0],
        in_axis=a.sig.out_axes[0],
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof(
            "tt_structured_adjoint",
            False,
            True,
            0,
            0,
            notes=("adjoint conjugates each TT core and swaps physical input/output dimensions",),
        ),
    )


def _compose_kronecker(a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "KroneckerProduct" or b.expr.op != "KroneckerProduct":
        return None
    fa = tuple(a.expr.args[0]["factors"])
    fb = tuple(b.expr.args[0]["factors"])
    if len(fa) != len(fb):
        raise SPFError("kronecker factor counts must match")
    for idx, (left_factor, right_factor) in enumerate(zip(fa, fb, strict=True)):
        left_axes = _axis_map(left_factor.axes)
        right_axes = _axis_map(right_factor.axes)
        for left_name, right_name in zip(left_factor.sig.in_axes, right_factor.sig.out_axes, strict=True):
            left_axis = left_axes[left_name]
            right_axis = right_axes[right_name]
            if left_axis.size != right_axis.size:
                raise SPFError(
                    "kronecker factor signature mismatch: "
                    f"factor {idx} contracts {left_axis.name}={left_axis.size} with "
                    f"{right_axis.name}={right_axis.size}"
                )
            if left_axis.basis != right_axis.basis:
                raise SPFError(
                    "kronecker factor signature mismatch: "
                    f"factor {idx} basis {left_axis.basis!r} != {right_axis.basis!r}"
                )
    composed = tuple(x.compose(y) for x, y in zip(fa, fb, strict=True))
    return SPFBlockMatrix.kronecker(
        composed,
        out_axis=a.sig.out_axes[0],
        in_axis=b.sig.in_axes[0],
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
    )


def _adjoint_character(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "Character":
        return None
    p = _character_params(a)
    return SPFBlockMatrix.character(
        N=int(p["N"]),
        alpha=-int(p["beta"]),
        beta=-int(p["alpha"]),
        out_axis=a.sig.in_axes[0],
        in_axis=a.sig.out_axes[0],
        scale=np.conj(complex(p.get("scale", 1.0 + 0.0j))),
        exact_group_semantics=bool(p.get("exact_group_semantics", True)),
        reducer_policy=a.sig.reducer_policy,
    )


def _adjoint_product_character(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "ProductGroupCharacter":
        return None
    p = _character_params(a)
    return SPFBlockMatrix.product_character(
        moduli=p["moduli"],
        out_frequency=tuple(-int(v) for v in p["in_frequency"]),
        in_frequency=tuple(-int(v) for v in p["out_frequency"]),
        out_axis=a.sig.in_axes[0],
        in_axis=a.sig.out_axes[0],
        scale=np.conj(complex(p.get("scale", 1.0 + 0.0j))),
        exact_group_semantics=bool(p.get("exact_group_semantics", True)),
        reducer_policy=a.sig.reducer_policy,
    )


def _adjoint_low_rank(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "LowRankOuter":
        return None
    p = a.expr.args[0]
    u = _complex_from_words(p["u_words"])
    v = _complex_from_words(p["v_words"])
    c = _complex_from_words(p["coeff_words"])
    return SPFBlockMatrix.low_rank(
        v,
        u,
        np.conj(c),
        out_axis=a.sig.in_axes[0],
        in_axis=a.sig.out_axes[0],
        reducer_policy=a.sig.reducer_policy,
    )


def _adjoint_circulant(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "CirculantSpectrum":
        return None
    assert a.expr.block is not None
    spectrum = np.conj(block_to_complex(a.expr.block).reshape(-1))
    return SPFBlockMatrix.circulant_spectrum(
        spectrum,
        out_axis=a.sig.in_axes[0],
        axis=a.sig.out_axes[0],
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof("circulant_adjoint", False, True, 0, 0),
    )


def _inverse_circulant(a: SPFKernel) -> SPFKernel | None:
    if a.expr.op != "CirculantSpectrum":
        return None
    assert a.expr.block is not None
    spectrum = block_to_complex(a.expr.block).reshape(-1)
    if np.any(np.abs(spectrum) <= _SPECTRAL_INVERTIBILITY_EPS):
        raise SPFError("cannot invert circulant with zero spectral bin")
    return SPFBlockMatrix.circulant_spectrum(
        1.0 / spectrum,
        axis=a.sig.in_axes[0],
        out_axis=a.sig.out_axes[0],
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof("circulant_inverse", False, True, 0, 0),
    )


def _power_circulant(a: SPFKernel, k: int) -> SPFKernel | None:
    if a.expr.op != "CirculantSpectrum":
        return None
    assert a.expr.block is not None
    spectrum = block_to_complex(a.expr.block).reshape(-1)
    if k < 0 and np.any(np.abs(spectrum) <= _SPECTRAL_INVERTIBILITY_EPS):
        raise SPFError("cannot raise circulant with zero spectral bin to negative power")
    powered = np.ones_like(spectrum) if k == 0 else spectrum ** int(k)
    return SPFBlockMatrix.circulant_spectrum(
        powered,
        axis=a.sig.in_axes[0],
        out_axis=a.sig.out_axes[0],
        reducer_policy=a.sig.reducer_policy,
        proof=SPFProof("circulant_power", False, True, 0, 0, notes=(f"k={int(k)}",)),
    )


class _CirculantAlgebra:
    kind = "CirculantSpectrum"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_circulant(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_circulant(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return _inverse_circulant(a)

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return _power_circulant(a, k)


class _BlockCirculantAlgebra:
    kind = "BlockCirculantSpectrum"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_block_circulant(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_block_circulant(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return _inverse_block_circulant(a)

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return _power_block_circulant(a, k)


class _NonAbelianCoefficientAlgebra:
    kind = "NonAbelianCoefficientTable"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_nonabelian(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _LazySumAlgebra:
    kind = "LazySum"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_lazy_sum(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _LazyHadamardAlgebra:
    kind = "LazyHadamard"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_lazy_hadamard(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _BindAlgebra:
    kind = "Bind"

    def pushforward(
        self,
        expr: SPFKernel,
        reduce_axes: tuple[str, ...],
        *,
        out_axes: tuple[str, ...],
        in_axes: tuple[str, ...],
        batch_axes: tuple[str, ...],
        reducer: ReducerSpec,
    ) -> SPFKernel | None:
        return _closed_pushforward(expr, reduce_axes, out_axes, in_axes, batch_axes, reducer)

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _ClosedFormZeroAlgebra:
    kind = "ClosedFormZero"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_closed_form_zero(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "ClosedFormZero":
            return None
        return SPFBlockMatrix.zero(
            batch_axes=tuple(_axis_map(a.axes)[name] for name in a.sig.batch_axes),
            out_axes=tuple(_axis_map(a.axes)[name] for name in a.sig.in_axes),
            in_axes=tuple(_axis_map(a.axes)[name] for name in a.sig.out_axes),
            reducer_policy=a.sig.reducer_policy,
            proof=SPFProof("closed_form_zero_adjoint", True, True, 0, 0, error_bound_l=0.0),
        )

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _ClosedFormIdentityAlgebra:
    kind = "ClosedFormIdentity"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_closed_form_identity(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "ClosedFormIdentity":
            return None
        out_axis = _axis_map(a.axes)[a.sig.in_axes[0]]
        in_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
        return SPFBlockMatrix.identity(
            out_axis.size,
            out_axis=out_axis.name,
            in_axis=in_axis.name,
            basis=out_axis.basis,
            group=out_axis.group,
            reducer_policy=a.sig.reducer_policy,
            proof=SPFProof("closed_form_identity_adjoint", True, True, 0, 0),
        )

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return a if a.expr.op == "ClosedFormIdentity" else None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return a if a.expr.op == "ClosedFormIdentity" else None


class _ResidualPatchAlgebra:
    kind = "ResidualPatch"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_residual_patch(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _JacobianOperatorAlgebra:
    kind = "JacobianOperator"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_jacobian_operator(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _DiagonalAlgebra:
    kind = "DiagonalSPF"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_diagonal(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "DiagonalSPF":
            return None
        return _diagonal_word_result(
            spf_conj_words(a.expr.args[0]["diag_words"]),
            _axis_map(a.axes)[a.sig.in_axes[0]], _axis_map(a.axes)[a.sig.out_axes[0]],
            exactness=a.sig.exactness,
            reducer=a.sig.reducer_policy,
        )

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "DiagonalSPF":
            return None
        words = a.expr.args[0]["diag_words"]
        rates = unpack_L(words)
        if np.any(rates == SPF32_L_ZERO):
            raise SPFError("cannot invert diagonal with zero entry")
        if np.any(rates == SPF32_L_NAN):
            raise SPFError("cannot invert diagonal with NaN entry")
        result, in_range = _spf_words_integer_power(words, -1)
        if not in_range:
            raise SPFError("diagonal reciprocal exceeds the finite SPF32 log range")
        return _diagonal_word_result(
            result, _axis_map(a.axes)[a.sig.in_axes[0]], _axis_map(a.axes)[a.sig.out_axes[0]],
            exactness=a.sig.exactness,
            reducer=a.sig.reducer_policy,
        )

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        if a.expr.op != "DiagonalSPF":
            return None
        words = a.expr.args[0]["diag_words"]
        if int(k) < 0 and np.any(unpack_L(words) == SPF32_L_ZERO):
            raise SPFError("cannot raise diagonal with zero entry to negative power")
        if int(k) != 0 and np.any(unpack_L(words) == SPF32_L_NAN):
            raise SPFError("cannot raise diagonal with NaN entry to nonzero power")
        result, in_range = _spf_words_integer_power(words, int(k))
        if int(k) == 0:
            result = pack_words(np.zeros(words.shape, dtype=np.int16), np.zeros(words.shape, dtype=np.uint16))
        if int(k) < 0 and not in_range:
            raise SPFError("negative diagonal power exceeds the finite SPF32 log range")
        exactness = a.sig.exactness if in_range or a.sig.exactness == "approximate" else "quantized"
        return _diagonal_word_result(
            result, _axis_map(a.axes)[a.sig.out_axes[0]], _axis_map(a.axes)[a.sig.in_axes[0]],
            exactness=exactness,
            reducer=a.sig.reducer_policy,
            proof=SPFProof("diagonal_integer_power", exactness == "exact", True, 0, 0,
                           notes=() if in_range else ("log_rate_saturation",)),
        )


class _PermutationAlgebra:
    kind = "Permutation"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_permutation(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "Permutation":
            return None
        idx = np.asarray(a.expr.args[0]["indices"], dtype=np.int64)
        inv = np.empty_like(idx)
        inv[idx] = np.arange(idx.size)
        return SPFBlockMatrix.permutation(
            inv,
            out_axis=a.sig.in_axes[0],
            axis=a.sig.out_axes[0],
            basis=str(a.expr.args[0].get("basis", "standard")),
            reducer_policy=a.sig.reducer_policy,
        )

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return self.adjoint(a)

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        if a.expr.op != "Permutation":
            return None
        idx = np.asarray(a.expr.args[0]["indices"], dtype=np.int64)
        if int(k) < 0:
            inv = np.empty_like(idx)
            inv[idx] = np.arange(idx.size)
            idx = inv
            k = -int(k)
        out = np.arange(idx.size, dtype=np.int64)
        for _ in range(int(k)):
            out = idx[out]
        return SPFBlockMatrix.permutation(
            out,
            out_axis=a.sig.out_axes[0],
            axis=a.sig.in_axes[0],
            basis=str(a.expr.args[0].get("basis", "standard")),
            reducer_policy=a.sig.reducer_policy,
        )


class _RollAlgebra:
    kind = "Roll"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_roll(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "Roll":
            return None
        params = a.expr.args[0]
        return SPFBlockMatrix.roll(
            int(params["n"]),
            -int(params["shift"]),
            out_axis=a.sig.in_axes[0],
            axis=a.sig.out_axes[0],
            basis=str(params.get("basis", "standard")),
            reducer_policy=a.sig.reducer_policy,
        )

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return self.adjoint(a)

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        if a.expr.op != "Roll":
            return None
        params = a.expr.args[0]
        return SPFBlockMatrix.roll(
            int(params["n"]),
            int(params["shift"]) * int(k),
            out_axis=a.sig.out_axes[0],
            axis=a.sig.in_axes[0],
            basis=str(params.get("basis", "standard")),
            reducer_policy=a.sig.reducer_policy,
        )


class _CharacterAlgebra:
    kind = "Character"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_character(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_character(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _QuadraticPhaseAlgebra:
    kind = "QuadraticPhase"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_quadratic_phase(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _ProductGroupCharacterAlgebra:
    kind = "ProductGroupCharacter"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_product_character(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_product_character(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _LowRankAlgebra:
    kind = "LowRankOuter"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_low_rank(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_low_rank(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _LazyOuterSumAlgebra:
    kind = "LazyOuterSum"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_lazy_outer_sum(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_lazy_outer_sum(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


class _KroneckerAlgebra:
    kind = "KroneckerProduct"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return _compose_kronecker(a, b)

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "KroneckerProduct":
            return None
        return SPFBlockMatrix.kronecker(
            tuple(f.H() for f in a.expr.args[0]["factors"]),
            out_axis=a.sig.in_axes[0],
            in_axis=a.sig.out_axes[0],
            reducer_policy=a.sig.reducer_policy,
        )

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        if a.expr.op != "KroneckerProduct":
            return None
        return SPFBlockMatrix.kronecker(
            tuple(f.inverse() for f in a.expr.args[0]["factors"]),
            out_axis=a.sig.out_axes[0],
            in_axis=a.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy,
        )

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        if a.expr.op != "KroneckerProduct":
            return None
        return SPFBlockMatrix.kronecker(
            tuple(f.power(int(k)) for f in a.expr.args[0]["factors"]),
            out_axis=a.sig.out_axes[0],
            in_axis=a.sig.in_axes[0],
            reducer_policy=a.sig.reducer_policy,
        )


def _spf_words_integer_power(words: np.ndarray, exponent: int) -> tuple[np.ndarray, bool]:
    """Raise each SPF32 word to an integer power by multiplying its integer rates.

    For a finite word ``(L, Θ)``, ``w**n`` is ``(n·L, n·Θ mod 2^16)`` — one integer
    multiply per mode, exact in SPF32 (no re-rounding) and identical word-for-word
    to ``n-1`` calls to :func:`spf_mul_words` (matmul post §7). Zero and NaN
    sentinels are preserved. ``L`` saturates to the finite range on overflow (the
    SPF32 ±128-octave dynamic-range limit); the returned flag is ``False`` when any
    finite mode saturated, so long rate chains that need SPF64 are reported as
    ``quantized`` rather than silently wrong.
    """
    exponent = int(exponent)
    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    L = unpack_L(packed).astype(np.int64)
    T = unpack_theta(packed).astype(np.int64)
    zero = int(SPF32_L_ZERO) == L
    nan = int(SPF32_L_NAN) == L
    finite = ~zero & ~nan
    if abs(exponent) <= np.iinfo(np.int64).max // 32768:
        L_wide = L * exponent
    else:
        # Every nonzero finite rate overflows the format at this exponent.
        # Keep arithmetic bounded even for arbitrary Python integer powers.
        L_wide = np.sign(L) * (1 if exponent > 0 else -1) * 32768
    Ln = _sat_l16(L_wide)
    Tn = ((T * (exponent & SPF32_PHASE_MASK)) & SPF32_PHASE_MASK).astype(np.uint16)
    saturated = bool(
        np.any(finite & ((L_wide < int(SPF32_L_MIN_FINITE)) | (L_wide > int(SPF32_L_MAX_FINITE))))
    )
    Ln = np.where(zero, np.int16(SPF32_L_ZERO), Ln)
    Ln = np.where(nan, np.int16(SPF32_L_NAN), Ln).astype(np.int16)
    Tn = np.where(nan, unpack_theta(packed), Tn).astype(np.uint16)
    return pack_words(Ln, Tn), (not saturated)


def _spf_words_reciprocal(words: np.ndarray) -> np.ndarray:
    """``1/w`` per SPF32 word: ``(-L, -Θ mod 2^16)`` (log-polar negation, exact)."""
    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    L = unpack_L(packed).astype(np.int64)
    T = unpack_theta(packed).astype(np.int64)
    zero = int(SPF32_L_ZERO) == L
    nan = int(SPF32_L_NAN) == L
    Lr = np.where(zero, np.int16(SPF32_L_ZERO), _sat_l16(-L)).astype(np.int16)
    Tr = ((-T) & SPF32_PHASE_MASK).astype(np.uint16)
    Tr = np.where(zero, np.uint16(0), Tr).astype(np.uint16)
    Lr = np.where(nan, np.int16(SPF32_L_NAN), Lr).astype(np.int16)
    Tr = np.where(nan, unpack_theta(packed), Tr).astype(np.uint16)
    return pack_words(Lr, Tr)


def _spf_geometric_depth_sum_words(
    lam_words: np.ndarray, count: int, reducer: ReducerSpec
) -> tuple[np.ndarray, bool, int] | None:
    """Per-mode ``Σ_{k<K} λ^k = (1-λ^K)/(1-λ)`` on exact SPF words (matmul post §2.1, §7).

    Each general mode is two SPF pair-adds (``1-λ^K`` and ``1-λ`` through the
    reducer boundary) and one exact word subtraction (``num · recip(den)``).
    ``λ=1`` returns ``K`` (the geometric limit) and an exactly-cancelling numerator
    (character orthogonality, ``λ^K==1`` with ``λ≠1``) returns the exact ZERO word.
    Returns ``(words, exact, reducer_count)``, or ``None`` when ``λ^K`` saturates.
    """
    count = int(count)
    lam = np.asarray(lam_words, dtype=np.uint32).reshape(-1)
    n = int(lam.size)
    one = np.uint32(encode_words(np.asarray(1.0 + 0.0j)))
    neg_one = np.uint32(encode_words(np.asarray(-1.0 + 0.0j)))
    lamK, in_range = _spf_words_integer_power(lam, count)
    if not in_range:
        # |λ|^K left the SPF32 ±128-octave range: the saturated λ^K would make the
        # numerator silently wrong, so the caller must take the dense fallback.
        return None
    is_one_lam = (unpack_L(lam) == 0) & (unpack_theta(lam) == 0)
    is_one_lamK = (unpack_L(lamK) == 0) & (unpack_theta(lamK) == 0)
    out = np.empty(n, dtype=np.uint32)
    exact = True
    reducer_count = 0
    for c in range(n):
        if bool(is_one_lam[c]):
            # λ == 1 ⇒ Σ = K. K is exact in SPF32 only when it is a power of two.
            out[c] = np.uint32(encode_words(np.asarray(float(count) + 0.0j)))
            if (count & (count - 1)) != 0 or count <= 0:
                exact = False
            continue
        if bool(is_one_lamK[c]):
            out[c] = SPF32_WORD_ZERO  # exact orthogonality zero
            continue
        num = np.uint32(
            reduce_words(np.array([one, np.uint32(spf_mul_words(lamK[c], neg_one))], dtype=np.uint32), reducer)
        )
        den = np.uint32(
            reduce_words(np.array([one, np.uint32(spf_mul_words(lam[c], neg_one))], dtype=np.uint32), reducer)
        )
        recip = np.uint32(_spf_words_reciprocal(np.array([den], dtype=np.uint32))[0])
        out[c] = np.uint32(spf_mul_words(np.array([num], dtype=np.uint32), np.array([recip], dtype=np.uint32))[0])
        reducer_count += 2
        exact = False
    return out, exact, reducer_count


def iterated_operator_depth_sum(operator: SPFKernel, count: int) -> SPFKernel:
    """Closed depth sum ``Σ_{k<K} A^k`` of an ``IteratedOperator`` (Law 9).

    When the generator is a ``DiagonalSPF`` (its diagonal *are* the modal rates,
    ``V = I``), the sum is the per-mode geometric law of
    :func:`_spf_geometric_depth_sum_words`. Otherwise the generator has no stored
    modal decomposition here, so the sum lowers to the dense reference
    ``Σ_k A^k`` — a counted Cartesian fallback, never silently structured.
    """
    if not isinstance(operator, SPFKernel) or operator.expr.op != "IteratedOperator":
        raise SPFError("iterated_operator_depth_sum requires an IteratedOperator kernel")
    count = int(count)
    if count < 1:
        raise SPFError("iterated_operator_depth_sum requires count >= 1")
    out_axis = _axis_map(operator.axes)[operator.sig.out_axes[0]]
    in_axis = _axis_map(operator.axes)[operator.sig.in_axes[0]]
    generator = operator.expr.args[0]["operator"]
    reducer = operator.sig.reducer_policy or _default_reducer()
    if generator.expr.op == "DiagonalSPF":
        lam_words = np.asarray(generator.expr.args[0]["diag_words"], dtype=np.uint32).reshape(-1)
        closed = _spf_geometric_depth_sum_words(lam_words, count, reducer)
    else:
        closed = None
    if closed is not None:
        summed, exact, reducer_count = closed
        return SPFBlockMatrix.diagonal(
            summed,
            out_axis=out_axis.name,
            axis=in_axis.name,
            basis=out_axis.basis,
            exactness="exact" if exact else "quantized",
            reducer_policy=operator.sig.reducer_policy,
            proof=SPFProof(
                "iterated_operator_depth_sum_geometric",
                exact,
                True,
                reducer_count,
                0,
                notes=(f"count={count}", "per_mode_geometric"),
            ),
        )
    dense = _complex_from_words(operator.materialize_words())
    acc = np.eye(dense.shape[0], dtype=np.complex128)
    total = np.zeros_like(dense)
    for _ in range(count):
        total = total + acc
        acc = acc @ dense
    return SPFBlockMatrix.dense(
        total,
        out_axes=(out_axis,),
        in_axes=(in_axis,),
        reducer_policy=operator.sig.reducer_policy,
        proof=SPFProof(
            "iterated_operator_depth_sum_dense_fallback",
            False,
            False,
            max(count - 1, 0),
            max(count - 1, 0),
            notes=(f"count={count}",),
        ),
    )


class _IteratedAlgebra:
    kind = "IteratedOperator"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        if a.expr.op != "IteratedOperator":
            return None
        exponent = int(k)
        if exponent < 0:
            raise SPFError("iterated_operator does not support negative powers without an inverse law")
        out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
        in_axis = _axis_map(a.axes)[a.sig.in_axes[0]]
        if exponent == 0:
            return SPFKernel(
                _closed_form_identity_expr(
                    out_axis.size,
                    out_axis.basis,
                    SPFProof("iterated_operator_power_zero_identity", True, True, 0, 0),
                ),
                (out_axis, in_axis),
                KernelSignature((out_axis.name,), (in_axis.name,), reducer_policy=a.sig.reducer_policy),
            )
        if exponent == 1:
            return a
        generator = a.expr.args[0]["operator"]
        if generator.expr.op == "DiagonalSPF":
            # Law 8: A = diag(λ) has V = I, so A^n multiplies each modal rate by n
            # (L·n, Θ·n mod 2^16) — an exact SPF32 word op, no dense matrix_power.
            diag_words = np.asarray(generator.expr.args[0]["diag_words"], dtype=np.uint32).reshape(-1)
            powered, exact = _spf_words_integer_power(diag_words, exponent)
            return SPFBlockMatrix.diagonal(
                powered,
                out_axis=out_axis.name,
                axis=in_axis.name,
                basis=out_axis.basis,
                exactness="exact" if exact else "quantized",
                reducer_policy=a.sig.reducer_policy,
                proof=SPFProof(
                    "iterated_operator_rate_power",
                    exact,
                    True,
                    0,
                    0,
                    notes=(f"power={exponent}", "per_mode_rate_multiply"),
                ),
            )
        dense = np.linalg.matrix_power(_complex_from_words(a.materialize_words()), exponent)
        return SPFBlockMatrix.dense(
            dense,
            out_axes=(out_axis,),
            in_axes=(in_axis,),
            reducer_policy=a.sig.reducer_policy,
            proof=SPFProof(
                "iterated_operator_dense_power_fallback",
                False,
                False,
                exponent - 1,
                exponent - 1,
                notes=(f"power={exponent}",),
            ),
        )


class _TTStructuredAlgebra:
    kind = "TTStructured"

    def compose(self, a: SPFKernel, b: SPFKernel) -> SPFKernel | None:
        return None

    def adjoint(self, a: SPFKernel) -> SPFKernel | None:
        return _adjoint_tt_structured(a)

    def inverse(self, a: SPFKernel) -> SPFKernel | None:
        return None

    def power(self, a: SPFKernel, k: int) -> SPFKernel | None:
        return None


def _register_default_algebras() -> None:
    from .algebra import register_structural_algebra
    from .header_algebra import GeometricRidgeAlgebra, STBBlockAlgebra

    register_structural_algebra(_CirculantAlgebra())
    register_structural_algebra(_BlockCirculantAlgebra())
    register_structural_algebra(_NonAbelianCoefficientAlgebra())
    register_structural_algebra(_LazySumAlgebra())
    register_structural_algebra(_LazyHadamardAlgebra())
    register_structural_algebra(_BindAlgebra())
    register_structural_algebra(_ClosedFormZeroAlgebra())
    register_structural_algebra(_ClosedFormIdentityAlgebra())
    register_structural_algebra(_ResidualPatchAlgebra())
    register_structural_algebra(_JacobianOperatorAlgebra())
    register_structural_algebra(_DiagonalAlgebra())
    register_structural_algebra(_PermutationAlgebra())
    register_structural_algebra(_RollAlgebra())
    register_structural_algebra(_CharacterAlgebra())
    register_structural_algebra(_QuadraticPhaseAlgebra())
    register_structural_algebra(_ProductGroupCharacterAlgebra())
    register_structural_algebra(_LowRankAlgebra())
    register_structural_algebra(_LazyOuterSumAlgebra())
    register_structural_algebra(_KroneckerAlgebra())
    register_structural_algebra(_IteratedAlgebra())
    register_structural_algebra(_TTStructuredAlgebra())
    register_structural_algebra(STBBlockAlgebra())
    register_structural_algebra(GeometricRidgeAlgebra())


_register_default_algebras()


__all__ = [
    "AxisSpec",
    "KernelSignature",
    "SPFBlockExpr",
    "SPFBlockMatrix",
    "SPFBlockVector",
    "SPFCPortEligibility",
    "SPFExecutionPlanParity",
    "SPFKernel",
    "SPFProof",
    "SPFValidationEvidence",
    "iterated_operator_depth_sum",
    "lazy_hadamard",
    "lazy_sum",
    "spfblock_einsum",
]
