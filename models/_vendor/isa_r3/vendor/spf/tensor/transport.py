"""Gauge-normalized SPF transport kernels."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .core import SPF32_L_MIN_FINITE, SPFError, pack_words
from .kernel import AxisSpec, KernelSignature, SPFBlockExpr, SPFBlockMatrix, SPFKernel, SPFProof, lazy_sum
from .policy import SPFExecutionPolicy
from .v2 import ExecutionPlan

MeasureKind = Literal[
    "counting",
    "causal_counting",
    "haar_finite_group",
    "frequency_bins",
    "page_cache",
    "learned_sparse",
]
GaugeKind = Literal[
    "none",
    "logsumexp",
    "tropical_max",
    "haar_exact",
    "analytic",
    "fourier",
    "standard",
    "product_group",
]
VALID_GAUGE_KINDS = frozenset(
    {
        "none",
        "logsumexp",
        "tropical_max",
        "haar_exact",
        "analytic",
        "fourier",
        "standard",
        "product_group",
    }
)
ActionKind = Literal[
    "identity",
    "u1_phase",
    "product_group_character",
    "diagonal_fourier",
    "low_rank",
    "dense_residual_fallback",
]


@dataclass(frozen=True)
class SPFMeasureSpec:
    kind: MeasureKind
    axes: tuple[str, ...]
    causal: bool = False
    group: Mapping[str, Any] | None = None
    mask: Any | None = None
    reducer_name: str | None = None
    reducer_domain: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(str(axis) for axis in self.axes))
        object.__setattr__(self, "group", None if self.group is None else dict(self.group))
        if self.kind == "causal_counting":
            object.__setattr__(self, "causal", True)


@dataclass(frozen=True)
class SPFGaugeSpec:
    axes: tuple[str, ...]
    kind: GaugeKind
    measure: SPFMeasureSpec
    exactness: str = "quantized"
    reducer: Any | None = None
    l_frac_bits: int = 8

    def __post_init__(self) -> None:
        if self.kind not in VALID_GAUGE_KINDS:
            raise SPFError(
                f"SPFGaugeSpec kind must be one of {sorted(VALID_GAUGE_KINDS)}, got {self.kind!r}"
            )
        if self.exactness not in {"exact", "quantized", "approximate"}:
            raise SPFError("SPFGaugeSpec exactness must be exact, quantized, or approximate")
        if int(self.l_frac_bits) <= 0:
            raise SPFError("SPFGaugeSpec l_frac_bits must be positive")
        object.__setattr__(self, "axes", tuple(str(axis) for axis in self.axes))
        object.__setattr__(self, "l_frac_bits", int(self.l_frac_bits))


@dataclass(frozen=True)
class SPFActionSpec:
    kind: ActionKind
    expr: SPFKernel | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata))


class SPFTransportKernel:
    """Typed transport facade over either an SPF kernel or an ExecutionPlan.

    The constructor intentionally accepts both the normalized keyword form used
    by the SMM runtime and the older positional test shape:
    ``(measure, gauge, action, kernel, plan, proof, lowering_kind)``.
    """

    def __init__(
        self,
        *args: Any,
        potential: SPFKernel | np.ndarray | None = None,
        action: SPFActionSpec | None = None,
        gauge: SPFGaugeSpec | None = None,
        source_axes: Sequence[AxisSpec] = (),
        target_axes: Sequence[AxisSpec] = (),
        fiber_in: Sequence[AxisSpec] = (),
        fiber_out: Sequence[AxisSpec] = (),
        proof: SPFProof | None = None,
        metadata: Mapping[str, Any] | None = None,
        measure: SPFMeasureSpec | None = None,
        kernel: SPFKernel | None = None,
        plan: ExecutionPlan | None = None,
        lowering_kind: str | None = None,
        policy: SPFExecutionPolicy | str | None = None,
        dense_decode_count: int = 0,
        cartesian_fallback_count: int = 0,
    ) -> None:
        if args:
            if len(args) != 7:
                raise SPFError("SPFTransportKernel positional form requires 7 arguments")
            measure, gauge, action, kernel, plan, proof, lowering_kind = args
            if not isinstance(measure, SPFMeasureSpec):
                raise SPFError("SPFTransportKernel positional measure must be SPFMeasureSpec")
            source_axes = _transport_axes_from_measure(measure, kernel)
            target_axes = _transport_target_axes(kernel)
        if action is None:
            raise SPFError("SPFTransportKernel requires an action")
        self.potential = potential
        self.action = action
        self.gauge = gauge
        self.source_axes = tuple(source_axes)
        self.target_axes = tuple(target_axes)
        self.fiber_in = tuple(fiber_in)
        self.fiber_out = tuple(fiber_out)
        self.proof = proof
        self.metadata = dict(metadata or {})
        self.measure = measure
        self.kernel = kernel
        self.plan = plan
        self.lowering_kind = lowering_kind or str(action.kind).upper()
        self.policy = _policy(policy)
        self.dense_decode_count = int(dense_decode_count)
        self.cartesian_fallback_count = int(cartesian_fallback_count)
        if not self.source_axes:
            raise SPFError("SPFTransportKernel requires at least one source axis")
        if not self.target_axes:
            raise SPFError("SPFTransportKernel requires at least one target axis")

    def cost(self) -> dict[str, Any]:
        kernel_cost = self.kernel.cost() if self.kernel is not None else {}
        reducer_count = 0
        if self.proof is not None:
            reducer_count = int(self.proof.reducer_count)
        elif self.kernel is not None and self.kernel.proof is not None:
            reducer_count = int(self.kernel.proof.reducer_count)
        if self.plan is not None:
            reducer_count += len(self.plan.reducer_boundaries)
        return {
            **kernel_cost,
            "lowering_kind": self.lowering_kind,
            "reducer_count": int(reducer_count),
            "cartesian_fallback_count": max(
                int(self.cartesian_fallback_count),
                int(kernel_cost.get("cartesian_fallback_count", 0)),
            ),
            "dense_decode_count": int(self.dense_decode_count),
        }

    def audit(self, policy: SPFExecutionPolicy | str | None = None) -> SPFPolicyAuditResult:
        return audit_spf_transport(self, policy=policy or self.policy)

    def recompress(
        self,
        *,
        max_rank: int | None = None,
        max_lazy_depth: int | None = None,
        residual_storage_bits: int | None = None,
    ) -> SPFTransportKernel:
        if self.kernel is None:
            return self
        if max_rank is not None and _transport_active_rank(self.kernel) > int(max_rank):
            raise SPFError("rank cap exceeded")
        if max_lazy_depth is not None and lazy_expression_depth(self.kernel) > int(max_lazy_depth):
            raise SPFError("lazy expression depth cap exceeded")
        if residual_storage_bits is not None and residual_storage_bits_for_kernel(self.kernel) > int(
            residual_storage_bits
        ):
            raise SPFError("residual storage budget exceeded")
        return self


@dataclass(frozen=True)
class TransportLoweringResult:
    kernel: SPFKernel
    lowering: str
    proof: SPFProof
    policy: SPFExecutionPolicy
    fallback_count: int = 0

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "lowering": self.lowering,
            "policy": self.policy.name,
            "fallback_count": int(self.fallback_count),
            "kernel_cost": self.kernel.cost(),
            "proof": {
                "law": self.proof.law,
                "exact": self.proof.exact,
                "metadata_only": self.proof.metadata_only,
                "reducer_count": self.proof.reducer_count,
                "cartesian_fallback_count": self.proof.cartesian_fallback_count,
                "notes": self.proof.notes,
            },
        }


@dataclass(frozen=True)
class SPFPolicyAuditResult:
    ok: bool
    failures: tuple[str, ...] = ()
    cost: Mapping[str, Any] = field(default_factory=dict)
    dense_decode_count: int = 0
    cartesian_fallback_count: int = 0

    def require_ok(self) -> None:
        if not self.ok:
            raise SPFError("; ".join(self.failures) or "SPF policy audit failed")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "failures": self.failures,
            "cost": dict(self.cost),
            "dense_decode_count": int(self.dense_decode_count),
            "cartesian_fallback_count": int(self.cartesian_fallback_count),
        }


SPFStrictAuditResult = SPFPolicyAuditResult
SPFStrictPolicy = SPFExecutionPolicy


TRANSPORT_PROOF_STAGES = (
    "pullback",
    "gauge",
    "action",
    "bind",
    "pushforward",
    "proof_law",
    "reducer_evidence",
    "fallback_decision",
)

TRANSPORT_LOWERING_REGISTRY_VERSION = "spf-transport-lowering-registry-v1"
STRICT_TRANSPORT_LOWERING_REGISTRY = (
    "phase_route",
    "relative_circulant",
    "causal_toeplitz_page",
    "product_character",
    "low_rank",
)


@dataclass(frozen=True)
class LegacyTransportPlan:
    action: SPFActionSpec
    measure: SPFMeasureSpec
    gauge: SPFGaugeSpec
    proof: SPFProof
    dense_decode_count: int = 0
    cartesian_fallback_count: int = 0
    strict: bool = False

    def cost(self) -> dict[str, Any]:
        return {
            "expression_op": "LegacyTransportPlan",
            "law": self.proof.law,
            "exact": self.proof.exact,
            "metadata_only": self.proof.metadata_only,
            "reducer_count": self.proof.reducer_count,
            "cartesian_fallback_count": self.cartesian_fallback_count,
            "dense_decode_count": self.dense_decode_count,
        }


def gauge_fix(
    potential: SPFKernel | np.ndarray | None,
    gauge: SPFGaugeSpec | None,
    *,
    target_axes: Sequence[AxisSpec],
    source_axes: Sequence[AxisSpec],
    policy: SPFExecutionPolicy | str | None = None,
) -> SPFKernel | None:
    exec_policy = _policy(policy)
    if gauge is None or gauge.kind == "none":
        if potential is None:
            return None
        if isinstance(potential, SPFKernel):
            exec_policy.validate_kernel(potential, purpose="ungauged transport potential")
            return potential
        return _potential_lq_to_probability_kernel(
            np.asarray(potential),
            gauge or SPFGaugeSpec(
                axes=tuple(axis.name for axis in source_axes),
                kind="none",
                measure=SPFMeasureSpec("counting", tuple(axis.name for axis in source_axes)),
            ),
            target_axes=target_axes,
            source_axes=source_axes,
            normalize=False,
        )
    if gauge.kind == "haar_exact":
        if not _has_exact_group_provenance(potential, target_axes, source_axes):
            raise SPFError("haar_exact gauge requires exact finite-group provenance")
        if isinstance(potential, SPFKernel):
            exec_policy.validate_kernel(potential, purpose="haar_exact transport potential")
            return potential
        return None
    if gauge.kind == "analytic":
        if isinstance(potential, SPFKernel):
            exec_policy.validate_kernel(potential, purpose="analytic transport potential")
            return potential
        raise SPFError("analytic gauge requires a structured SPFKernel potential")
    if not isinstance(potential, np.ndarray):
        if isinstance(potential, SPFKernel):
            raise SPFError(f"{gauge.kind} gauge over generic SPFKernel is not a v1 strict lowering")
        raise SPFError(f"{gauge.kind} gauge requires an integer L-domain potential")
    return _potential_lq_to_probability_kernel(
        potential,
        gauge,
        target_axes=target_axes,
        source_axes=source_axes,
        normalize=gauge.kind == "logsumexp",
        tropical=gauge.kind == "tropical_max",
    )


def lower_transport(
    transport: SPFTransportKernel | SPFActionSpec,
    measure: SPFMeasureSpec | None = None,
    gauge: SPFGaugeSpec | None = None,
    *,
    policy: SPFExecutionPolicy | str | None = None,
    strict: bool | None = None,
) -> TransportLoweringResult | SPFTransportKernel:
    if isinstance(transport, SPFActionSpec):
        return _lower_legacy_transport(transport, measure, gauge, policy=policy, strict=strict)
    exec_policy = _policy(policy)
    action = transport.action
    _validate_transport_gauge_policy(transport.gauge, exec_policy)
    if action.kind == "dense_residual_fallback" and exec_policy.forbid_dense_fallback:
        raise SPFError(f"{exec_policy.name} policy rejects dense_residual_fallback transport")
    _reject_low_rank_dense_softmax(action, exec_policy)
    P = gauge_fix(
        transport.potential,
        transport.gauge,
        target_axes=transport.target_axes,
        source_axes=transport.source_axes,
        policy=exec_policy,
    )
    A = action.expr
    if action.kind in {"diagonal_fourier", "u1_phase", "product_group_character", "low_rank"}:
        if A is None:
            raise SPFError(f"{action.kind} transport requires an SPFKernel action")
        exec_policy.validate_kernel(A, purpose=f"{action.kind} transport action")
        base = A if P is None else P.hadamard(A, allow_fallback=not exec_policy.forbid_dense_fallback)
        lowering = action.kind
    elif action.kind == "identity":
        if P is None:
            base = _identity_transport_kernel(
                target_axes=transport.target_axes,
                source_axes=transport.source_axes,
                policy=exec_policy,
            )
            lowering = "identity"
        else:
            base = P
            lowering = "gauge_measure"
    elif action.kind == "dense_residual_fallback":
        if A is None:
            raise SPFError("dense_residual_fallback requires an SPFKernel fallback action")
        exec_policy.validate_kernel(A, purpose="dense residual fallback")
        base = A if P is None else lazy_sum((P, A), reducer=exec_policy.default_reducer())
        lowering = "dense_residual_fallback"
    else:
        raise SPFError(f"unsupported transport action {action.kind!r}")
    proof = SPFProof(
        f"transport_{lowering}",
        bool(base.proof.exact) if base.proof is not None else False,
        bool(base.proof.metadata_only) if base.proof is not None else False,
        int(base.proof.reducer_count) if base.proof is not None else 0,
        int(base.proof.cartesian_fallback_count) if base.proof is not None else 0,
        notes=(
            f"gauge={None if transport.gauge is None else transport.gauge.kind}",
            f"action={action.kind}",
            f"reducer={_transport_reducer_name(transport)}",
            f"fallback={'present' if action.kind == 'dense_residual_fallback' else 'clean'}",
        ),
    )
    exec_policy.validate_kernel(base, purpose="lowered transport")
    return TransportLoweringResult(
        kernel=base,
        lowering=lowering,
        proof=proof,
        policy=exec_policy,
        fallback_count=1 if action.kind == "dense_residual_fallback" else 0,
    )


def apply_transport(
    transport: SPFTransportKernel,
    values: SPFKernel | np.ndarray,
    *,
    policy: SPFExecutionPolicy | str | None = None,
) -> SPFKernel:
    if not isinstance(values, SPFKernel):
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise SPFError("array transport values must be rank-1")
        axis_name = transport.source_axes[0].name if transport.source_axes else "i"
        values = SPFBlockMatrix.dense(arr, out_axes=(axis_name,), in_axes=())
    if transport.kernel is not None:
        return transport.kernel.apply(values)
    lowered = lower_transport(transport, policy=policy)
    return lowered.kernel.apply(values)


def compose_transport(
    left: SPFTransportKernel | TransportLoweringResult,
    right: SPFTransportKernel | TransportLoweringResult,
    *,
    policy: SPFExecutionPolicy | str | None = None,
) -> SPFTransportKernel:
    """Compose two typed transports as kernel morphisms."""

    exec_policy = _policy(policy)
    left_kernel, left_source, left_target, left_trace = _transport_kernel_and_axes(left, exec_policy)
    right_kernel, right_source, right_target, right_trace = _transport_kernel_and_axes(right, exec_policy)
    if len(left_source) != len(right_target):
        raise SPFError("transport composition source/target rank mismatch")
    for source_axis, target_axis in zip(left_source, right_target, strict=True):
        if source_axis.size != target_axis.size:
            raise SPFError("transport composition middle-axis size mismatch")
        if source_axis.basis != target_axis.basis:
            raise SPFError("transport composition middle-axis basis mismatch")
    reducer = exec_policy.default_reducer()
    composed = left_kernel.compose(right_kernel, reducer=reducer)
    proof = SPFProof(
        "transport_compose",
        bool(composed.proof.exact) if composed.proof is not None else False,
        bool(composed.proof.metadata_only) if composed.proof is not None else False,
        int(composed.proof.reducer_count) if composed.proof is not None else 0,
        int(composed.proof.cartesian_fallback_count) if composed.proof is not None else 0,
        notes=(
            "pullback:left_target_middle",
            "pullback:middle_right_source",
            "bind:spf_kernel_compose",
            "pushforward:middle_axis",
        ),
    )
    exec_policy.validate_kernel(composed, purpose="composed transport")
    return SPFTransportKernel(
        potential=None,
        action=SPFActionSpec("identity"),
        gauge=None,
        source_axes=right_source,
        target_axes=left_target,
        kernel=with_proof(composed, proof),
        proof=proof,
        policy=exec_policy,
        lowering_kind="COMPOSED_TRANSPORT",
        metadata={
            "proof_trace": (*left_trace, *right_trace, *transport_proof_trace(proof)),
            "left_lowering": _transport_lowering_name(left),
            "right_lowering": _transport_lowering_name(right),
        },
    )


def transport_proof_trace(
    transport: SPFTransportKernel | TransportLoweringResult | SPFProof,
) -> tuple[str, ...]:
    if isinstance(transport, SPFProof):
        return (
            f"law:{transport.law}",
            f"exact:{transport.exact}",
            f"metadata_only:{transport.metadata_only}",
            f"reducers:{transport.reducer_count}",
            *tuple(str(note) for note in transport.notes),
        )
    if isinstance(transport, TransportLoweringResult):
        proof = transport.proof
        return (
            "pullback:target_source_axes",
            f"gauge:{_proof_note_value(proof, 'gauge')}",
            f"action:{_proof_note_value(proof, 'action')}",
            "bind:measure_action",
            "pushforward:source_axes",
            *transport_proof_trace(proof),
        )
    trace = tuple(str(item) for item in transport.metadata.get("proof_trace", ()))
    if trace:
        return trace
    if transport.measure is not None and transport.gauge is not None and transport.proof is not None:
        return (
            "pullback:target_source_axes",
            f"gauge:{_proof_note_value(transport.proof, 'gauge')}",
            f"action:{_proof_note_value(transport.proof, 'action')}",
            "bind:measure_action",
            "pushforward:source_axes",
            *transport_proof_trace(transport.proof),
        )
    if transport.proof is not None:
        return transport_proof_trace(transport.proof)
    lowered = lower_transport(transport, policy=transport.policy)
    return transport_proof_trace(lowered)


def transport_proof_stage_coverage(
    transport: SPFTransportKernel | TransportLoweringResult,
) -> dict[str, bool]:
    """Return machine-readable proof coverage for a lowered transport."""

    trace = transport_proof_trace(transport)
    cost = _transport_cost(transport)
    reducer_count = int(cost.get("reducer_count", 0))
    fallback_count = max(
        int(cost.get("cartesian_fallback_count", 0)),
        int(cost.get("dense_decode_count", 0)),
        int(cost.get("fallback_count", 0)),
    )
    # A stage is only "covered" when the lowering actually produced its evidence.
    # The pullback/bind/pushforward stages describe the lowered SPF object itself,
    # so a degenerate lowering that yielded no kernel (e.g. an unsupported op that
    # fell through to a metadata-only proof) must report them MISSING rather than
    # being satisfied by the boilerplate trace strings emitted unconditionally.
    lowered = _transport_has_lowered_kernel(transport)
    return {
        "pullback": lowered and _trace_has(trace, "pullback:"),
        "gauge": (not _transport_has_gauge(transport)) or _trace_has(trace, "gauge:", "gauge="),
        "action": (not _transport_has_action(transport)) or _trace_has(trace, "action:", "action="),
        "bind": lowered and _trace_has(trace, "bind:"),
        "pushforward": lowered and _trace_has(trace, "pushforward:"),
        "proof_law": _trace_has(trace, "law:"),
        "reducer_evidence": reducer_count == 0 or _transport_has_reducer_evidence(transport, trace),
        "fallback_decision": fallback_count == 0
        or _trace_has(trace, "fallback:", "fallback=", "DENSE_REFERENCE", "dense_reference"),
    }


def transport_validation_report(
    transport: SPFTransportKernel | TransportLoweringResult,
    *,
    policy: SPFExecutionPolicy | str | None = None,
    require_policy_ok: bool | None = None,
) -> dict[str, Any]:
    """Validate one transport as a proof-carrying SPF transport record."""

    exec_policy = _transport_policy(transport, policy)
    coverage = transport_proof_stage_coverage(transport)
    missing = tuple(stage for stage in TRANSPORT_PROOF_STAGES if not coverage.get(stage, False))
    audit = audit_spf_transport(transport, policy=exec_policy)
    require_policy = (
        exec_policy.name in {"native", "strict"}
        if require_policy_ok is None
        else bool(require_policy_ok)
    )
    ok = not missing and (audit.ok or not require_policy)
    return {
        "status": "ok" if ok else "failed",
        "policy": exec_policy.name,
        "lowering": _transport_lowering_name(transport),
        "proof_stage_coverage": coverage,
        "missing_stages": missing,
        "trace": transport_proof_trace(transport),
        "cost": _transport_cost(transport),
        "policy_audit": audit.to_json_dict(),
    }


def transport_validation_suite_report(
    transports: Mapping[str, SPFTransportKernel | TransportLoweringResult],
    *,
    policy: SPFExecutionPolicy | str | None = None,
    require_policy_ok: bool | None = None,
) -> dict[str, Any]:
    """Validate a named transport suite for proof, reducer, and fallback coverage."""

    reports = {
        name: transport_validation_report(
            transport,
            policy=policy,
            require_policy_ok=require_policy_ok,
        )
        for name, transport in transports.items()
    }
    failed = tuple(name for name, report in sorted(reports.items()) if report["status"] != "ok")
    return {
        "status": "ok" if not failed else "failed",
        "transport_count": len(reports),
        "failed_transports": failed,
        "reports": reports,
    }


def registered_transport_lowering_suite(
    *,
    policy: SPFExecutionPolicy | str | None = "strict",
) -> dict[str, SPFTransportKernel | TransportLoweringResult]:
    """Build the current named SPF transport lowering suite.

    This is intentionally a small executable registry rather than a passive
    tuple of names. Callers get concrete lowered transports that can be audited
    for proof-stage coverage, reducer evidence, and fallback decisions.
    """

    exec_policy = _policy(policy)
    strict = exec_policy.name == "strict"
    measure = SPFMeasureSpec("registry_measure", ("i",), reducer_name="registry_logpolar_fold")
    gauge = SPFGaugeSpec("registry_gauge", "fourier", ("i",))
    return {
        "phase_route": lower_transport(
            SPFActionSpec("route", "phase_concordance_lse", {"beta": 6.0, "causal": True}),
            measure,
            gauge,
            policy=exec_policy,
            strict=strict,
        ),
        "relative_circulant": lower_transport(
            SPFActionSpec("relative", "relative_circulant", {"spectrum": np.ones(4)}),
            measure,
            gauge,
            policy=exec_policy,
            strict=strict,
        ),
        "causal_toeplitz_page": lower_transport(
            SPFActionSpec(
                "page_relative",
                "causal_relative_toeplitz",
                {
                    "spectrum": np.ones(8, dtype=np.complex128),
                    "page_size": 8,
                    "source_axis": "page_key_token",
                    "target_axis": "page_query_token",
                },
            ),
            SPFMeasureSpec("page_measure", ("page_key_token",), reducer_name="page_logpolar_fold"),
            SPFGaugeSpec("page_gauge", "fourier", ("page_key_token",)),
            policy=exec_policy,
            strict=strict,
        ),
        "product_character": lower_transport(
            SPFActionSpec(
                "push",
                "product_character_pushforward",
                {"moduli": (2, 3), "out_frequency": (1, 0), "in_frequency": (0, 1)},
            ),
            measure,
            SPFGaugeSpec("product_gauge", "product_group", ("i",)),
            policy=exec_policy,
            strict=strict,
        ),
        "low_rank": lower_transport(
            SPFActionSpec(
                "lr",
                "low_rank",
                {
                    "u": np.eye(3, 2, dtype=np.complex128),
                    "v": np.eye(3, 2, dtype=np.complex128),
                    "coeff": np.array([1.0, 0.5], dtype=np.complex128),
                },
            ),
            measure,
            SPFGaugeSpec("standard_gauge", "standard", ("i",)),
            policy=exec_policy,
            strict=strict,
        ),
    }


def transport_lowering_registry_report(
    *,
    policy: SPFExecutionPolicy | str | None = "strict",
) -> dict[str, Any]:
    """Return the versioned machine-readable SPF transport lowering registry."""

    exec_policy = _policy(policy)
    suite = registered_transport_lowering_suite(policy=exec_policy)
    validation = transport_validation_suite_report(suite, policy=exec_policy)
    lowerings = {
        name: {
            "lowering": _transport_lowering_name(transport),
            "proof_law": (
                transport.proof.law
                if isinstance(transport, TransportLoweringResult)
                else (None if transport.proof is None else transport.proof.law)
            ),
            "kernel_op": (
                transport.kernel.expr.op
                if isinstance(transport, TransportLoweringResult)
                else (None if transport.kernel is None else transport.kernel.expr.op)
            ),
            "cost": _transport_cost(transport),
            "proof_stage_coverage": transport_proof_stage_coverage(transport),
        }
        for name, transport in suite.items()
    }
    missing_registered = tuple(
        name for name in STRICT_TRANSPORT_LOWERING_REGISTRY if name not in lowerings
    )
    incomplete = tuple(
        name
        for name, item in sorted(lowerings.items())
        if not all(bool(value) for value in item["proof_stage_coverage"].values())
    )
    strict_clean = all(
        int(item["cost"].get("dense_decode_count", 0)) == 0
        and int(item["cost"].get("cartesian_fallback_count", 0)) == 0
        for item in lowerings.values()
    )
    ok = bool(
        validation["status"] == "ok"
        and not missing_registered
        and not incomplete
        and (exec_policy.name != "strict" or strict_clean)
    )
    return {
        "format": TRANSPORT_LOWERING_REGISTRY_VERSION,
        "policy": exec_policy.name,
        "status": "ok" if ok else "failed",
        "registered_lowering_count": len(lowerings),
        "required_lowerings": STRICT_TRANSPORT_LOWERING_REGISTRY,
        "missing_registered_lowerings": missing_registered,
        "incomplete_lowerings": incomplete,
        "strict_clean": bool(strict_clean),
        "lowerings": lowerings,
        "validation": validation,
    }


def audit_spf_kernel(
    kernel: SPFKernel,
    *,
    policy: SPFExecutionPolicy | str | None = None,
) -> SPFPolicyAuditResult:
    exec_policy = _policy(policy)
    failures: list[str] = []
    try:
        exec_policy.validate_kernel(kernel, purpose="kernel audit")
    except Exception as exc:
        failures.append(str(exc))
    cost = kernel.cost() if hasattr(kernel, "cost") else {}
    return SPFPolicyAuditResult(ok=not failures, failures=tuple(failures), cost=cost)


def audit_spf_transport(
    transport: SPFTransportKernel | TransportLoweringResult | LegacyTransportPlan,
    policy: SPFExecutionPolicy | str | None = None,
) -> SPFPolicyAuditResult:
    exec_policy = _policy(policy)
    if isinstance(transport, LegacyTransportPlan):
        failures: list[str] = []
        if exec_policy.forbid_decode and transport.dense_decode_count:
            failures.append("decode fallback present")
        if exec_policy.forbid_cartesian_reducer and transport.cartesian_fallback_count:
            failures.append("Cartesian fallback present")
        return SPFPolicyAuditResult(
            ok=not failures,
            failures=tuple(failures),
            cost=transport.cost(),
            dense_decode_count=transport.dense_decode_count,
            cartesian_fallback_count=transport.cartesian_fallback_count,
        )
    if isinstance(transport, TransportLoweringResult):
        return audit_spf_kernel(transport.kernel, policy=exec_policy)
    if isinstance(transport, SPFTransportKernel) and (
        transport.kernel is not None or transport.plan is not None
    ):
        failures: list[str] = []
        if transport.dense_decode_count or transport.cartesian_fallback_count:
            failures.append("dense/cartesian reference fallback present")
        if exec_policy.forbid_decode and transport.dense_decode_count:
            failures.append("decode fallback present")
        if exec_policy.forbid_cartesian_reducer and transport.cartesian_fallback_count:
            failures.append("Cartesian fallback present")
        if transport.kernel is not None:
            try:
                exec_policy.validate_kernel(transport.kernel, purpose="transport audit")
            except Exception as exc:
                failures.append(str(exc))
        return SPFPolicyAuditResult(
            ok=not failures,
            failures=tuple(failures),
            cost=transport.cost(),
            dense_decode_count=transport.dense_decode_count,
            cartesian_fallback_count=transport.cartesian_fallback_count,
        )
    try:
        lowered = lower_transport(transport, policy=exec_policy)
    except Exception as exc:
        return SPFPolicyAuditResult(ok=False, failures=(str(exc),), cost={})
    return audit_spf_kernel(lowered.kernel, policy=exec_policy)


def with_proof(kernel: SPFKernel, proof: SPFProof) -> SPFKernel:
    if not isinstance(kernel, SPFKernel):
        raise SPFError("with_proof expects an SPFKernel")
    return SPFKernel(
        SPFBlockExpr(kernel.expr.op, args=kernel.expr.args, block=kernel.expr.block, proof=proof),
        kernel.axes,
        kernel.sig,
    )


def _transport_kernel_and_axes(
    transport: SPFTransportKernel | TransportLoweringResult,
    policy: SPFExecutionPolicy,
) -> tuple[SPFKernel, tuple[AxisSpec, ...], tuple[AxisSpec, ...], tuple[str, ...]]:
    if isinstance(transport, TransportLoweringResult):
        kernel = transport.kernel
        source_axes = _kernel_axes(kernel, kernel.sig.in_axes)
        target_axes = _kernel_axes(kernel, kernel.sig.out_axes)
        return kernel, source_axes, target_axes, transport_proof_trace(transport)
    if transport.kernel is not None:
        kernel = transport.kernel
        return kernel, transport.source_axes, transport.target_axes, transport_proof_trace(transport)
    lowered = lower_transport(transport, policy=policy)
    return (
        lowered.kernel,
        _kernel_axes(lowered.kernel, lowered.kernel.sig.in_axes),
        _kernel_axes(lowered.kernel, lowered.kernel.sig.out_axes),
        transport_proof_trace(lowered),
    )


def _kernel_axes(kernel: SPFKernel, names: Sequence[str]) -> tuple[AxisSpec, ...]:
    axes = {axis.name: axis for axis in kernel.axes}
    return tuple(axes[name] for name in names)


def _transport_lowering_name(transport: SPFTransportKernel | TransportLoweringResult) -> str:
    if isinstance(transport, TransportLoweringResult):
        return transport.lowering
    return transport.lowering_kind


def _transport_policy(
    transport: SPFTransportKernel | TransportLoweringResult,
    policy: SPFExecutionPolicy | str | None,
) -> SPFExecutionPolicy:
    if policy is not None:
        return _policy(policy)
    if isinstance(transport, TransportLoweringResult):
        return transport.policy
    return transport.policy


def _transport_cost(transport: SPFTransportKernel | TransportLoweringResult) -> dict[str, Any]:
    if isinstance(transport, TransportLoweringResult):
        kernel_cost = transport.kernel.cost()
        return {
            **kernel_cost,
            "lowering_kind": transport.lowering,
            "reducer_count": int(transport.proof.reducer_count),
            "cartesian_fallback_count": max(
                int(kernel_cost.get("cartesian_fallback_count", 0)),
                int(transport.proof.cartesian_fallback_count),
            ),
            "dense_decode_count": int(kernel_cost.get("dense_decode_count", 0)),
            "fallback_count": int(transport.fallback_count),
        }
    return transport.cost()


def _trace_has(trace: Sequence[str], *prefixes_or_terms: str) -> bool:
    for item in trace:
        text = str(item)
        for prefix_or_term in prefixes_or_terms:
            if text.startswith(prefix_or_term) or prefix_or_term in text:
                return True
    return False


def _transport_has_lowered_kernel(
    transport: SPFTransportKernel | TransportLoweringResult,
) -> bool:
    """True only when the lowering produced a concrete SPF kernel.

    A ``TransportLoweringResult`` always carries a real kernel; a
    ``SPFTransportKernel`` may be degenerate (``kernel is None``) when an
    unsupported/failed lowering fell through to a metadata-only proof. The
    pullback/bind/pushforward proof stages describe that lowered object, so
    coverage must distinguish a real lowering from a bare proof record.
    """

    if isinstance(transport, TransportLoweringResult):
        return transport.kernel is not None
    return transport.kernel is not None


def _transport_has_gauge(transport: SPFTransportKernel | TransportLoweringResult) -> bool:
    if isinstance(transport, TransportLoweringResult):
        return True
    return transport.gauge is not None and transport.gauge.kind != "none"


def _transport_has_action(transport: SPFTransportKernel | TransportLoweringResult) -> bool:
    if isinstance(transport, TransportLoweringResult):
        return True
    return transport.action.kind != "identity"


def _transport_has_reducer_evidence(
    transport: SPFTransportKernel | TransportLoweringResult,
    trace: Sequence[str],
) -> bool:
    if _trace_has(trace, "reducer:", "reducer="):
        return True
    if isinstance(transport, TransportLoweringResult):
        return bool(transport.kernel.sig.reducer_policy)
    if transport.measure is not None and transport.measure.reducer_name:
        return True
    if transport.gauge is not None and transport.gauge.measure.reducer_name:
        return True
    if transport.kernel is not None and transport.kernel.sig.reducer_policy is not None:
        return True
    return bool(transport.plan is not None and transport.plan.reducer_boundaries)


def _validate_transport_gauge_policy(gauge: SPFGaugeSpec | None, policy: SPFExecutionPolicy) -> None:
    if gauge is None or not policy.require_named_reducers:
        return
    if gauge.kind not in {"logsumexp", "tropical_max"}:
        return
    reducer_name = getattr(gauge.measure, "reducer_name", None)
    if not reducer_name:
        raise SPFError(f"{policy.name} policy requires named reducer for transport gauge")


def _phase_concordance_route_proof(
    action: SPFActionSpec,
    measure: SPFMeasureSpec,
    gauge: SPFGaugeSpec,
) -> SPFProof:
    return SPFProof(
        "phase_concordance_logsumexp_transport",
        True,
        False,
        1,
        0,
        notes=(
            "pullback:q_theta,k_theta,value_fiber",
            "address:phase_concordance_lut",
            f"gauge={gauge.kind}",
            f"action={action.kind}",
            "bind:positive_measure_value",
            "pushforward:memory_axis",
            f"reducer={measure.reducer_name}",
            f"score_domain={action.metadata.get('score_domain', 'integer_theta_lut')}",
            f"causal={bool(action.metadata.get('causal', False))}",
            f"l_frac_bits={int(gauge.l_frac_bits)}",
        ),
    )


def _phase_concordance_route_kernel(
    action: SPFActionSpec,
    measure: SPFMeasureSpec,
    gauge: SPFGaugeSpec,
    *,
    reducer: Any,
    proof: SPFProof,
) -> SPFKernel:
    q_words = action.metadata.get("q_words")
    k_words = action.metadata.get("k_words")
    v_words = action.metadata.get("v_words")
    q_arr = None if q_words is None else np.asarray(q_words, dtype=np.uint32)
    k_arr = None if k_words is None else np.asarray(k_words, dtype=np.uint32)
    v_arr = None if v_words is None else np.asarray(v_words, dtype=np.uint32)
    target_size = int(action.metadata.get("target_size", 0))
    source_size = int(action.metadata.get("source_size", 0))
    channel_size = int(action.metadata.get("channel_size", 0))
    if q_arr is not None:
        if q_arr.ndim != 2:
            raise SPFError("phase_concordance_lse q_words must be rank-2")
        target_size = int(q_arr.shape[0])
        channel_size = int(q_arr.shape[1])
    if k_arr is not None:
        if k_arr.ndim != 2:
            raise SPFError("phase_concordance_lse k_words must be rank-2")
        source_size = int(k_arr.shape[0])
        if channel_size and int(k_arr.shape[1]) != channel_size:
            raise SPFError("phase_concordance_lse q/k channel mismatch")
        channel_size = int(k_arr.shape[1])
    if v_arr is not None:
        if v_arr.ndim != 2:
            raise SPFError("phase_concordance_lse v_words must be rank-2")
        if source_size and int(v_arr.shape[0]) != source_size:
            raise SPFError("phase_concordance_lse k/v source mismatch")
        source_size = int(v_arr.shape[0])
        if channel_size and int(v_arr.shape[1]) != channel_size:
            raise SPFError("phase_concordance_lse v channel mismatch")
        channel_size = int(v_arr.shape[1])
    source_name = measure.axes[0] if measure.axes else "memory"
    target_name = str(action.metadata.get("target_axis", "target"))
    params = {
        "q_words": q_arr,
        "k_words": k_arr,
        "v_words": v_arr,
        "beta": float(action.metadata.get("beta", 1.0)),
        "causal": bool(action.metadata.get("causal", False)),
        "l_frac_bits": int(gauge.l_frac_bits),
        "reducer": reducer,
        "reducer_name": measure.reducer_name,
        "channel_size": channel_size,
    }
    target_axis = AxisSpec(target_name, target_size, role="codomain")
    source_axis = AxisSpec(source_name, source_size, role="domain")
    return SPFKernel(
        SPFBlockExpr("PhaseConcordanceRoute", args=(params,), proof=proof),
        (target_axis, source_axis),
        KernelSignature(
            out_axes=(target_axis.name,),
            in_axes=(source_axis.name,),
            reducer_policy=reducer,
        ),
    )


def _proof_note_value(proof: SPFProof, key: str) -> str:
    prefix = f"{key}="
    for note in proof.notes:
        text = str(note)
        if text.startswith(prefix):
            return text[len(prefix) :]
    return "<none>"


def proof_law_counts(kernels: Sequence[SPFKernel]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for kernel in kernels:
        proof = kernel.proof
        law = kernel.expr.op if proof is None else proof.law
        counts[str(law)] = counts.get(str(law), 0) + 1
    return counts


def lazy_expression_depth(kernel: SPFKernel) -> int:
    op = kernel.expr.op
    if op in {"LazySum", "LazyHadamard", "Bind"}:
        children = kernel.expr.args[0] if op == "LazySum" else kernel.expr.args
        return 1 + max((lazy_expression_depth(child) for child in children if isinstance(child, SPFKernel)), default=0)
    return 1


def active_rank(kernel: SPFKernel) -> int:
    if kernel.expr.op == "LowRankOuter":
        params = kernel.expr.args[0]
        return int(params.get("rank", 1)) if isinstance(params, Mapping) else 1
    if kernel.expr.op == "LazyOuterSum":
        params = kernel.expr.args[0]
        terms = params.get("terms", ()) if isinstance(params, Mapping) else ()
        return len(terms)
    return 0


def residual_storage_bits_for_kernel(kernel: SPFKernel) -> int:
    cost = kernel.cost() if hasattr(kernel, "cost") else {}
    for key in ("residual_storage_bits", "storage_bits", "payload_bits"):
        if key in cost and cost[key] is not None:
            return int(cost[key])
    return 0


def _transport_axes_from_measure(
    measure: SPFMeasureSpec,
    kernel: SPFKernel | None,
) -> tuple[AxisSpec, ...]:
    if kernel is not None and kernel.sig.in_axes:
        axis_by_name = {axis.name: axis for axis in kernel.axes}
        return tuple(axis_by_name[name] for name in kernel.sig.in_axes)
    return tuple(AxisSpec(name, 0, role="domain") for name in measure.axes)


def _transport_target_axes(kernel: SPFKernel | None) -> tuple[AxisSpec, ...]:
    if kernel is not None and kernel.sig.out_axes:
        axis_by_name = {axis.name: axis for axis in kernel.axes}
        return tuple(axis_by_name[name] for name in kernel.sig.out_axes)
    return (AxisSpec("o", 0, role="codomain"),)


def _transport_active_rank(kernel: SPFKernel) -> int:
    if kernel.expr.op == "LowRankOuter":
        params = kernel.expr.args[0]
        coeff_words = params.get("coeff_words") if isinstance(params, Mapping) else None
        if coeff_words is not None:
            return int(np.asarray(coeff_words).size)
    return active_rank(kernel)


def _lower_legacy_transport(
    action: SPFActionSpec,
    measure: SPFMeasureSpec | None,
    gauge: SPFGaugeSpec | None,
    *,
    policy: SPFExecutionPolicy | str | None,
    strict: bool | None,
) -> SPFTransportKernel:
    exec_policy = (
        SPFExecutionPolicy.strict()
        if strict
        else (SPFExecutionPolicy.reference() if policy is None else _policy(policy))
    )
    if measure is None or gauge is None:
        raise SPFError("legacy lower_transport requires action, measure, and gauge")
    domain = (measure.reducer_domain or "").lower()
    if exec_policy.forbid_cartesian_reducer and domain == "cartesian":
        raise SPFError("strict policy forbids Cartesian reducer transport")
    if exec_policy.require_named_reducers and not measure.reducer_name:
        raise SPFError(f"{exec_policy.name} policy requires named reducer for transport")
    expr = str(action.expr)
    reducer = exec_policy.default_reducer()
    kernel: SPFKernel | None = None
    plan: ExecutionPlan | None = None
    dense_decode_count = 0
    cartesian_fallback_count = 0
    if expr == "dense" or str(action.kind) == "dense":
        if exec_policy.forbid_dense_fallback:
            raise SPFError("strict transport refuses dense fallback")
        values = np.asarray(action.metadata.get("values"))
        kernel = SPFBlockMatrix.dense(
            values,
            out_axes=("o",),
            in_axes=("i",),
            reducer_policy=reducer,
            proof=SPFProof("dense_reference_transport", False, False, 1, 1),
        )
        proof = SPFProof(
            "dense_reference_transport",
            False,
            False,
            1,
            1,
            notes=(
                f"gauge={gauge.kind}",
                f"action={action.kind}",
                "fallback:dense_reference",
                f"reducer={measure.reducer_name}",
            ),
        )
        kernel = with_proof(kernel, proof)
        lowering_kind = "DENSE_REFERENCE"
        dense_decode_count = 1
        cartesian_fallback_count = 1
    elif expr == "phase_concordance_lse":
        proof = _phase_concordance_route_proof(action, measure, gauge)
        kernel = _phase_concordance_route_kernel(
            action,
            measure,
            gauge,
            reducer=reducer,
            proof=proof,
        )
        lowering_kind = "PHASE_CONCORDANCE_TRANSPORT"
    elif expr == "relative_circulant":
        proof = SPFProof(
            "relative_circulant_spectrum_transport",
            False,
            True,
            0,
            0,
            notes=(f"reducer={measure.reducer_name}", "fallback=clean"),
        )
        kernel = with_proof(
            SPFBlockMatrix.circulant_spectrum(
                np.asarray(action.metadata.get("spectrum")),
                reducer_policy=reducer,
            ),
            proof,
        )
        lowering_kind = "CIRCULANT_SPECTRUM"
    elif expr == "causal_relative_toeplitz":
        spectrum = np.asarray(action.metadata.get("spectrum"))
        page_size = int(action.metadata.get("page_size", spectrum.shape[0] if spectrum.ndim else 0))
        if page_size <= 0:
            raise SPFError("causal_relative_toeplitz requires a positive page_size")
        if spectrum.ndim != 1:
            raise SPFError("causal_relative_toeplitz spectrum must be rank-1")
        if int(spectrum.shape[0]) != page_size:
            raise SPFError("causal_relative_toeplitz spectrum length must match page_size")
        proof = SPFProof(
            "causal_relative_toeplitz_page_transport",
            False,
            True,
            0,
            0,
            notes=(
                "pullback:page_origin_source_target",
                "bind:spectral_page_kernel_value",
                "pushforward:causal_overlap_save_page",
                f"page_size={page_size}",
                f"reducer={measure.reducer_name}",
                "fallback=clean",
            ),
        )
        kernel = with_proof(
            SPFBlockMatrix.circulant_spectrum(
                spectrum,
                axis=str(action.metadata.get("source_axis", "page_key_token")),
                out_axis=str(action.metadata.get("target_axis", "page_query_token")),
                reducer_policy=reducer,
            ),
            proof,
        )
        lowering_kind = "CAUSAL_TOEPLITZ_PAGE_SPECTRUM"
    elif expr == "product_character_pushforward":
        proof = SPFProof(
            "product_group_character_pushforward_transport",
            True,
            True,
            0,
            0,
            notes=(f"reducer={measure.reducer_name}", "fallback=clean"),
        )
        kernel = with_proof(
            SPFBlockMatrix.product_character(
                moduli=action.metadata.get("moduli", ()),
                out_frequency=action.metadata.get("out_frequency", ()),
                in_frequency=action.metadata.get("in_frequency", ()),
                reducer_policy=reducer,
                proof=proof,
            ),
            proof,
        )
        lowering_kind = "CLOSED_FORM_PUSHFORWARD"
    elif expr == "low_rank":
        _reject_low_rank_dense_softmax(action, exec_policy)
        u = np.asarray(action.metadata.get("u"))
        v = np.asarray(action.metadata.get("v"))
        coeff = action.metadata.get("coeff")
        rank = int(u.shape[1]) if u.ndim == 2 else 0
        proof = SPFProof(
            "low_rank_spf_gram_transport",
            False,
            False,
            rank,
            0,
            notes=(f"reducer={measure.reducer_name}", "fallback=clean"),
        )
        kernel = with_proof(
            SPFBlockMatrix.low_rank(
                u,
                v,
                None if coeff is None else np.asarray(coeff),
                reducer_policy=reducer,
                proof=proof,
            ),
            proof,
        )
        lowering_kind = "LOW_RANK_GRAM"
    else:
        proof = SPFProof(
            f"legacy_transport_{expr}",
            False,
            False,
            0,
            0,
            notes=(f"reducer={measure.reducer_name}", "fallback=clean"),
        )
        lowering_kind = str(expr).upper()
    return SPFTransportKernel(
        measure,
        gauge,
        action,
        kernel,
        plan,
        proof,
        lowering_kind,
        policy=exec_policy,
        dense_decode_count=dense_decode_count,
        cartesian_fallback_count=cartesian_fallback_count,
    )


def _policy(policy: SPFExecutionPolicy | str | None) -> SPFExecutionPolicy:
    if policy is None:
        return SPFExecutionPolicy.native()
    if isinstance(policy, SPFExecutionPolicy):
        return policy
    return SPFExecutionPolicy.for_name(str(policy))


def _reject_low_rank_dense_softmax(action: SPFActionSpec, policy: SPFExecutionPolicy) -> None:
    if str(action.kind) != "low_rank" and str(action.expr) != "low_rank":
        return
    metadata = action.metadata
    score_path = str(metadata.get("score_path", metadata.get("softmax_path", ""))).lower()
    dense_softmax = bool(metadata.get("dense_softmax", False)) or score_path in {
        "dense_softmax",
        "dense_f32_softmax",
        "dense_float_softmax",
    }
    if dense_softmax and (policy.forbid_dense_fallback or policy.forbid_scores_f32):
        raise SPFError(f"{policy.name} policy refuses low-rank score with dense softmax")


def _transport_reducer_name(transport: SPFTransportKernel) -> str:
    if transport.gauge is not None and transport.gauge.measure.reducer_name:
        return str(transport.gauge.measure.reducer_name)
    if transport.measure is not None and transport.measure.reducer_name:
        return str(transport.measure.reducer_name)
    if transport.kernel is not None and transport.kernel.sig.reducer_policy is not None:
        return str(getattr(transport.kernel.sig.reducer_policy, "name", None) or "kernel_reducer")
    return "<none>"


def _potential_lq_to_probability_kernel(
    potential: np.ndarray,
    gauge: SPFGaugeSpec,
    *,
    target_axes: Sequence[AxisSpec],
    source_axes: Sequence[AxisSpec],
    normalize: bool,
    tropical: bool = False,
) -> SPFBlockMatrix:
    lq = np.asarray(potential)
    if lq.ndim != len(target_axes) + len(source_axes):
        raise SPFError("potential rank must match target_axes + source_axes")
    if not np.issubdtype(lq.dtype, np.integer):
        raise SPFError("transport potential must be integer L-domain codes")
    lq = lq.astype(np.int64, copy=False)
    source_rank = len(source_axes)
    target_shape = tuple(axis.size for axis in target_axes)
    source_shape = tuple(axis.size for axis in source_axes)
    if tuple(lq.shape) != target_shape + source_shape:
        raise SPFError(f"potential shape {tuple(lq.shape)} does not match target/source axes")
    mask = _source_mask(lq.shape, target_axes, source_axes, gauge.measure)
    invalid = ~mask
    if tropical:
        valid_scores = np.where(mask, lq, np.iinfo(np.int64).min // 4)
        max_scores = np.max(valid_scores, axis=tuple(range(lq.ndim - source_rank, lq.ndim)), keepdims=True)
        keep = (valid_scores == max_scores) & mask
        out_l = np.where(keep, 0, SPF32_L_MIN_FINITE).astype(np.int16)
    elif normalize:
        scale = float(1 << gauge.l_frac_bits)
        scores = lq.astype(np.float64) / scale
        scores = np.where(mask, scores, -np.inf)
        max_scores = np.max(scores, axis=tuple(range(lq.ndim - source_rank, lq.ndim)), keepdims=True)
        shifted = np.where(np.isfinite(scores), scores - max_scores, -np.inf)
        denom = np.sum(np.exp2(shifted), axis=tuple(range(lq.ndim - source_rank, lq.ndim)), keepdims=True)
        logp = shifted - np.log2(denom)
        finite_logp = np.where(mask & np.isfinite(logp), logp, 0.0)
        out_l = np.rint(256.0 * finite_logp).astype(np.int64)
        out_l = np.where(invalid | ~np.isfinite(logp), SPF32_L_MIN_FINITE, out_l)
        out_l = np.clip(out_l, np.iinfo(np.int16).min, 0).astype(np.int16)
    else:
        out_l = np.rint(lq.astype(np.float64) * (256.0 / float(1 << gauge.l_frac_bits))).astype(np.int64)
        out_l = np.where(invalid, SPF32_L_MIN_FINITE, out_l)
        out_l = np.clip(out_l, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
    words = pack_words(out_l, np.zeros_like(out_l, dtype=np.uint16))
    proof = SPFProof(
        f"gauge_fix_{gauge.kind}",
        gauge.exactness == "exact",
        False,
        1 if normalize else 0,
        0,
        error_bound_l=1.0 if normalize else 0.0,
        notes=(f"measure={gauge.measure.kind}", f"l_frac_bits={gauge.l_frac_bits}"),
    )
    return SPFBlockMatrix.from_words(
        words,
        out_axes=target_axes,
        in_axes=source_axes,
        reducer_policy=gauge.reducer,
        proof=proof,
    )


def _source_mask(
    shape: tuple[int, ...],
    target_axes: Sequence[AxisSpec],
    source_axes: Sequence[AxisSpec],
    measure: SPFMeasureSpec,
) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    if measure.kind != "causal_counting" and not measure.causal:
        return mask
    if len(target_axes) != 1 or len(source_axes) != 1:
        raise SPFError("causal_counting v1 requires one target axis and one source axis")
    target = np.arange(target_axes[0].size, dtype=np.int64)[:, None]
    source = np.arange(source_axes[0].size, dtype=np.int64)[None, :]
    return source <= target


def _identity_transport_kernel(
    *,
    target_axes: Sequence[AxisSpec],
    source_axes: Sequence[AxisSpec],
    policy: SPFExecutionPolicy,
) -> SPFKernel:
    if len(target_axes) != 1 or len(source_axes) != 1:
        raise SPFError("identity transport v1 requires one target axis and one source axis")
    if target_axes[0].size != source_axes[0].size:
        raise SPFError("identity transport requires matching target/source sizes")
    return SPFBlockMatrix.identity(
        target_axes[0].size,
        out_axis=target_axes[0].name,
        in_axis=source_axes[0].name,
        basis=target_axes[0].basis,
        group=target_axes[0].group,
        reducer_policy=policy.default_reducer(),
    )


def _has_exact_group_provenance(
    potential: SPFKernel | np.ndarray | None,
    target_axes: Sequence[AxisSpec],
    source_axes: Sequence[AxisSpec],
) -> bool:
    if isinstance(potential, SPFKernel) and potential.sig.exactness == "exact":
        proof = potential.proof
        if proof is not None and proof.exact:
            return True
    axes = tuple(target_axes) + tuple(source_axes)
    return bool(axes and all(axis.basis == "product_group" and axis.group for axis in axes))


__all__ = [
    "STRICT_TRANSPORT_LOWERING_REGISTRY",
    "TRANSPORT_LOWERING_REGISTRY_VERSION",
    "TRANSPORT_PROOF_STAGES",
    "LegacyTransportPlan",
    "SPFActionSpec",
    "SPFGaugeSpec",
    "SPFMeasureSpec",
    "SPFPolicyAuditResult",
    "SPFStrictAuditResult",
    "SPFStrictPolicy",
    "SPFTransportKernel",
    "TransportLoweringResult",
    "active_rank",
    "apply_transport",
    "audit_spf_kernel",
    "audit_spf_transport",
    "compose_transport",
    "gauge_fix",
    "lazy_expression_depth",
    "lower_transport",
    "proof_law_counts",
    "registered_transport_lowering_suite",
    "residual_storage_bits_for_kernel",
    "transport_lowering_registry_report",
    "transport_proof_stage_coverage",
    "transport_proof_trace",
    "transport_validation_report",
    "transport_validation_suite_report",
    "with_proof",
]
