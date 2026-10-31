"""Graph-level PS/PL planner for PAN Fabric Runtime.

The primitive fabric runtime can call existing KV260 kernels. This module adds
the planning layer that decides where buffers should live across a typed SPF
operation graph, avoiding the per-op round trip that made standalone PL
``spf32.mul`` lose to the A53 library.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Optional, Sequence

from .fabric_runtime import FabricKernelSpec, FabricRegistry, KERNEL_CATALOG, active_xmutil_app


FABRIC_LOCATIONS = frozenset({"ps", "pl"})


@dataclass(frozen=True)
class FabricBufferSpec:
    name: str
    elements: int
    residency: str = "ps"

    @classmethod
    def from_json(cls, name: str, payload: Any) -> "FabricBufferSpec":
        if isinstance(payload, int):
            return cls(name=name, elements=int(payload), residency="ps")
        data = dict(payload)
        return cls(
            name=name,
            elements=int(data["elements"]),
            residency=str(data.get("residency", "ps")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "elements": self.elements,
            "residency": self.residency,
        }


@dataclass(frozen=True)
class FabricGraphOp:
    op_id: str
    op: str
    inputs: tuple[str, ...]
    output: str
    elements: int
    extra_outputs: tuple[str, ...] = ()

    @classmethod
    def from_json(cls, payload: dict[str, Any], index: int) -> "FabricGraphOp":
        outputs = tuple(str(value) for value in payload.get("outputs") or ())
        primary_output = str(payload.get("output") or (outputs[0] if outputs else ""))
        extra_outputs = tuple(value for value in outputs if value != primary_output)
        return cls(
            op_id=str(payload.get("id") or f"op{index}"),
            op=str(payload["op"]),
            inputs=tuple(str(value) for value in payload.get("inputs", ())),
            output=primary_output,
            elements=int(payload["elements"]),
            extra_outputs=extra_outputs,
        )

    @property
    def outputs(self) -> tuple[str, ...]:
        return (self.output, *self.extra_outputs)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "id": self.op_id,
            "op": self.op,
            "inputs": list(self.inputs),
            "output": self.output,
            "elements": self.elements,
        }
        if self.extra_outputs:
            payload["outputs"] = list(self.outputs)
        return payload


@dataclass(frozen=True)
class FabricGraph:
    buffers: dict[str, FabricBufferSpec]
    ops: tuple[FabricGraphOp, ...]
    outputs: tuple[str, ...]
    output_residency: str = "ps"
    schema: str = "pan_fabric_graph_v1"

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FabricGraph":
        buffers = {
            str(name): FabricBufferSpec.from_json(str(name), spec)
            for name, spec in dict(payload.get("buffers") or {}).items()
        }
        ops = tuple(FabricGraphOp.from_json(item, index) for index, item in enumerate(payload.get("ops") or ()))
        for op in ops:
            for name in op.inputs:
                buffers.setdefault(name, FabricBufferSpec(name=name, elements=op.elements, residency="ps"))
            for name in op.outputs:
                buffers.setdefault(name, FabricBufferSpec(name=name, elements=op.elements, residency="unknown"))
        outputs = tuple(str(value) for value in payload.get("outputs") or ((ops[-1].output,) if ops else ()))
        return cls(
            buffers=buffers,
            ops=ops,
            outputs=outputs,
            output_residency=str(payload.get("output_residency", "ps")),
            schema=str(payload.get("schema", "pan_fabric_graph_v1")),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "FabricGraph":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def linear(
        cls,
        ops: Sequence[str],
        *,
        elements: int,
        output_residency: str = "ps",
        input_residency: str = "ps",
    ) -> "FabricGraph":
        if not ops:
            raise ValueError("linear fabric graph needs at least one op")
        buffers: dict[str, FabricBufferSpec] = {
            "input0": FabricBufferSpec("input0", int(elements), input_residency),
        }
        graph_ops: list[FabricGraphOp] = []
        previous = "input0"
        side_index = 1
        for index, op in enumerate(ops):
            arity = int((KERNEL_CATALOG.get(_kernel_name_hint(op)) or {}).get("arity", 1))
            inputs = [previous]
            if arity >= 2:
                side_name = f"input{side_index}"
                side_index += 1
                buffers[side_name] = FabricBufferSpec(side_name, int(elements), input_residency)
                inputs.append(side_name)
            output = f"tmp{index}"
            buffers[output] = FabricBufferSpec(output, int(elements), "unknown")
            graph_ops.append(FabricGraphOp(f"op{index}", op, tuple(inputs), output, int(elements)))
            previous = output
        return cls(
            buffers=buffers,
            ops=tuple(graph_ops),
            outputs=(previous,),
            output_residency=output_residency,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "buffers": {name: spec.to_json() for name, spec in sorted(self.buffers.items())},
            "ops": [op.to_json() for op in self.ops],
            "outputs": list(self.outputs),
            "output_residency": self.output_residency,
        }


@dataclass(frozen=True)
class FabricCostModel:
    ps_seconds_per_word: dict[str, float] = field(default_factory=dict)
    ps_energy_j_per_word: dict[str, float] = field(default_factory=dict)
    pl_dispatch_seconds: dict[str, float] = field(default_factory=dict)
    pl_dispatch_energy_j: dict[str, float] = field(default_factory=dict)
    transfer_seconds_per_byte: float = 1.0e-8
    transfer_energy_j_per_byte: float = 4.0e-10
    default_ps_seconds_per_word: float = 8.0e-8
    default_ps_energy_j_per_word: float = 2.0e-8
    default_pl_dispatch_seconds: float = 3.6e-4
    default_pl_dispatch_energy_j: float = 1.35e-4

    @classmethod
    def measured_kv260_v0(cls) -> "FabricCostModel":
        return cls(
            ps_seconds_per_word={
                "spf32.mul": 1.0 / 29_544_624.27017326,
                "spf32.add": 1.0 / 35_000_000.0,
                "spf32.conj": 1.0 / 50_000_000.0,
                "spf32.abs2": 1.0 / 24_000_000.0,
                "spf32.fft64": 1.2e-5,
            },
            ps_energy_j_per_word={
                "spf32.mul": 0.7486438073382305 / (4096 * 20000),
                "spf32.add": 6.0e-9,
                "spf32.conj": 5.0e-9,
                "spf32.abs2": 1.2e-8,
                "spf32.fft64": 8.0e-7,
            },
            pl_dispatch_seconds={
                "spf32.mul": 3.5233469009399415e-4,
                "spf32.add": 3.5e-4,
                "spf32.conj": 3.5e-4,
                "spf32.abs2": 3.5e-4,
                "spf32.fft64": 3.5e-4,
                "spf32.fft64_fb7": 3.5e-4,
                "spf16.fft64_fb7": 3.5e-4,
            },
            pl_dispatch_energy_j={
                "spf32.mul": 2.679169471040368 / 20000,
                "spf32.add": 1.3e-4,
                "spf32.conj": 1.3e-4,
                "spf32.abs2": 1.3e-4,
                "spf32.fft64": 1.3e-4,
                "spf32.fft64_fb7": 1.3e-4,
                "spf16.fft64_fb7": 1.3e-4,
            },
        )

    def ps_seconds(self, op: str, elements: int) -> float:
        return float(elements) * self.ps_seconds_per_word.get(op, self.default_ps_seconds_per_word)

    def ps_energy(self, op: str, elements: int) -> float:
        return float(elements) * self.ps_energy_j_per_word.get(op, self.default_ps_energy_j_per_word)

    def pl_seconds(self, op: str, elements: int) -> float:
        return self.pl_dispatch_seconds.get(op, self.default_pl_dispatch_seconds)

    def pl_energy(self, op: str, elements: int) -> float:
        return self.pl_dispatch_energy_j.get(op, self.default_pl_dispatch_energy_j)


@dataclass(frozen=True)
class FabricPolicyHead:
    """Small linear policy head over cost/residency features.

    It is intentionally serializable and trainable later. Today the default
    weights implement the measured latency+energy objective.
    """

    weights: dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    policy_kind: str = "linear_neural_fabric_head_v1"
    policy_role: str = "rule_authority"
    promotion_status: str = "promoted"
    can_override_rules: bool = True
    min_confidence: float = 1.0
    heldout_metrics: dict[str, Any] = field(default_factory=dict)
    heads: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "FabricPolicyHead":
        return cls(
            weights={
                "latency_s": 1.0,
                "energy_j": 0.10,
                "transfer_bytes": 1.0e-10,
                "unsupported": 1.0e9,
                "keeps_residency": -2.5e-5,
                "active_app_match": -5.0e-5,
                "downstream_pl_affinity": -7.5e-5,
            }
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FabricPolicyHead":
        gate = dict(payload.get("gating") or {})
        return cls(
            weights={str(name): float(value) for name, value in dict(payload.get("weights") or {}).items()},
            bias=float(payload.get("bias", 0.0)),
            policy_kind=str(payload.get("policy_kind", "linear_neural_fabric_head_v1")),
            policy_role=str(payload.get("policy_role", gate.get("policy_role", "rule_authority"))),
            promotion_status=str(payload.get("promotion_status", gate.get("promotion_status", "promoted"))),
            can_override_rules=bool(payload.get("can_override_rules", gate.get("can_override_rule_scheduler", True))),
            min_confidence=float(payload.get("min_confidence", gate.get("min_confidence", 1.0))),
            heldout_metrics=dict(payload.get("heldout_metrics") or gate.get("heldout_metrics") or {}),
            heads=dict(payload.get("heads") or {}),
        )

    @classmethod
    def from_path(cls, path: str | Path) -> "FabricPolicyHead":
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))

    def score(self, features: dict[str, float]) -> float:
        return self.bias + sum(self.weights.get(name, 0.0) * float(value) for name, value in features.items())

    def can_override_rule_scheduler(self) -> bool:
        return bool(self.can_override_rules and self.promotion_status in {"promoted", "rule_authority"})

    def promotion_evaluation(
        self,
        metrics: dict[str, Any],
        *,
        min_samples: int = 1,
        max_latency_ratio: float = 1.0,
        max_transfer_ratio: float = 1.0,
        max_energy_ratio: float = 1.0,
        min_compile_priority_precision: float = 0.80,
    ) -> dict[str, Any]:
        latency_ratio = _ratio_metric(metrics, "latency_ratio_vs_rule", "neural_elapsed_seconds", "rule_elapsed_seconds")
        transfer_ratio = _ratio_metric(metrics, "transfer_bytes_ratio_vs_rule", "neural_transfer_bytes", "rule_transfer_bytes")
        energy_ratio = _ratio_metric(metrics, "energy_ratio_vs_rule", "neural_energy_j", "rule_energy_j")
        compile_precision = _float_or_none(metrics.get("compile_priority_precision"))
        sample_count = int(metrics.get("sample_count") or metrics.get("heldout_sample_count") or 0)
        confidence = _float_or_none(metrics.get("confidence"))
        energy_checked = bool(metrics.get("energy_checked"))
        checks = {
            "sample_count_sufficient": sample_count >= int(min_samples),
            "latency_beats_rule_scheduler": latency_ratio < float(max_latency_ratio),
            "transfer_bytes_not_worse": transfer_ratio <= float(max_transfer_ratio),
            "energy_not_worse": True if not energy_checked else energy_ratio <= float(max_energy_ratio),
            "validation_failures_zero": int(metrics.get("validation_failures") or 0) == 0,
            "output_residency_not_worse": int(metrics.get("output_residency_regressions") or 0) == 0,
            "materialization_regressions_zero": int(metrics.get("materialization_regressions") or 0) == 0,
            "compile_priority_precision_checked": (
                bool(metrics.get("compile_priority_precision_checked"))
                if compile_precision is None
                else compile_precision >= float(min_compile_priority_precision)
            ),
            "confidence_passed": True if confidence is None else confidence >= float(self.min_confidence),
        }
        promoted = all(checks.values())
        return {
            "schema": "pan_fabric_policy_promotion_evaluation_v1",
            "promoted": promoted,
            "checks": checks,
            "metrics": dict(metrics),
            "thresholds": {
                "min_samples": int(min_samples),
                "max_latency_ratio": float(max_latency_ratio),
                "max_transfer_ratio": float(max_transfer_ratio),
                "max_energy_ratio": float(max_energy_ratio),
                "min_compile_priority_precision": float(min_compile_priority_precision),
                "min_confidence": float(self.min_confidence),
            },
            "reason": "heldout_metrics_passed" if promoted else "heldout_metrics_failed",
        }

    def with_promotion_metrics(
        self,
        metrics: dict[str, Any],
        *,
        min_samples: int = 1,
        max_latency_ratio: float = 1.0,
        max_transfer_ratio: float = 1.0,
        max_energy_ratio: float = 1.0,
        min_compile_priority_precision: float = 0.80,
    ) -> "FabricPolicyHead":
        evaluation = self.promotion_evaluation(
            metrics,
            min_samples=min_samples,
            max_latency_ratio=max_latency_ratio,
            max_transfer_ratio=max_transfer_ratio,
            max_energy_ratio=max_energy_ratio,
            min_compile_priority_precision=min_compile_priority_precision,
        )
        heldout_metrics = dict(self.heldout_metrics)
        heldout_metrics.update(dict(metrics))
        heldout_metrics["status"] = "promoted" if evaluation["promoted"] else "rejected"
        heldout_metrics["promotion_evaluation"] = evaluation
        return replace(
            self,
            policy_role="neural_promoted" if evaluation["promoted"] else self.policy_role,
            promotion_status="promoted" if evaluation["promoted"] else "candidate",
            can_override_rules=bool(evaluation["promoted"]),
            heldout_metrics=heldout_metrics,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_kind": self.policy_kind,
            "policy_role": self.policy_role,
            "promotion_status": self.promotion_status,
            "can_override_rules": self.can_override_rules,
            "min_confidence": self.min_confidence,
            "heldout_metrics": dict(self.heldout_metrics),
            "heads": dict(self.heads),
            "gating": {
                "policy_role": self.policy_role,
                "promotion_status": self.promotion_status,
                "can_override_rule_scheduler": self.can_override_rule_scheduler(),
                "min_confidence": self.min_confidence,
                "heldout_metrics": dict(self.heldout_metrics),
            },
            "bias": self.bias,
            "weights": dict(sorted(self.weights.items())),
        }


@dataclass(frozen=True)
class FabricGraphPlanStep:
    action: str
    backend: str
    op_id: Optional[str] = None
    op: Optional[str] = None
    buffer: Optional[str] = None
    src: Optional[str] = None
    dst: Optional[str] = None
    kernel: Optional[FabricKernelSpec] = None
    reason: str = ""
    latency_s: float = 0.0
    energy_j: float = 0.0
    transfer_bytes: int = 0
    score: float = 0.0
    features: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "backend": self.backend,
            "op_id": self.op_id,
            "op": self.op,
            "buffer": self.buffer,
            "src": self.src,
            "dst": self.dst,
            "kernel": None if self.kernel is None else self.kernel.to_json(),
            "reason": self.reason,
            "latency_s": self.latency_s,
            "energy_j": self.energy_j,
            "transfer_bytes": self.transfer_bytes,
            "score": self.score,
            "features": dict(sorted(self.features.items())),
        }


@dataclass(frozen=True)
class FabricGraphPlan:
    graph: FabricGraph
    steps: tuple[FabricGraphPlanStep, ...]
    final_residency: dict[str, tuple[str, ...]]
    policy: FabricPolicyHead
    candidate_policy: Optional[FabricPolicyHead] = None
    cost_model: str = "measured_kv260_v0"
    schema: str = "pan_fabric_graph_plan_v1"

    @property
    def total_latency_s(self) -> float:
        return sum(step.latency_s for step in self.steps)

    @property
    def total_energy_j(self) -> float:
        return sum(step.energy_j for step in self.steps)

    @property
    def transfer_bytes(self) -> int:
        return sum(step.transfer_bytes for step in self.steps)

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "graph": self.graph.to_json(),
            "policy": self.policy.to_json(),
            "candidate_policy": None if self.candidate_policy is None else self.candidate_policy.to_json(),
            "cost_model": self.cost_model,
            "summary": {
                "step_count": len(self.steps),
                "run_ps_count": sum(1 for step in self.steps if step.action == "run_ps"),
                "run_pl_count": sum(1 for step in self.steps if step.action == "run_pl"),
                "transfer_count": sum(1 for step in self.steps if step.action == "transfer"),
                "total_latency_s": self.total_latency_s,
                "total_energy_j": self.total_energy_j,
                "transfer_bytes": self.transfer_bytes,
            },
            "steps": [step.to_json() for step in self.steps],
            "final_residency": {name: list(value) for name, value in sorted(self.final_residency.items())},
            "pl_regions": _pl_regions(self.steps),
        }


class FabricGraphPlanner:
    def __init__(
        self,
        registry: FabricRegistry,
        *,
        policy: Optional[FabricPolicyHead] = None,
        cost_model: Optional[FabricCostModel] = None,
        backend: str = "auto",
        active_app: Optional[str] = None,
    ):
        if backend not in {"auto", "ps", "pl"}:
            raise ValueError(f"unsupported fabric graph backend {backend!r}")
        self.registry = registry
        requested_policy = policy or FabricPolicyHead.default()
        self.candidate_policy = None if requested_policy.can_override_rule_scheduler() else requested_policy
        self.policy = requested_policy if requested_policy.can_override_rule_scheduler() else FabricPolicyHead.default()
        self.cost_model = cost_model or FabricCostModel.measured_kv260_v0()
        self.backend = backend
        self.active_app = active_xmutil_app() if active_app == "auto" else active_app

    def plan(self, graph: FabricGraph) -> FabricGraphPlan:
        residency = {name: _parse_residency(spec.residency) for name, spec in graph.buffers.items()}
        steps: list[FabricGraphPlanStep] = []
        consumers = _consumer_map(graph)
        for index, op in enumerate(graph.ops):
            ps_candidate = self._candidate(graph, op, "ps", residency, consumers, index)
            pl_candidate = self._candidate(graph, op, "pl", residency, consumers, index)
            candidates = [ps_candidate, pl_candidate]
            if self.backend == "ps":
                chosen = ps_candidate
            elif self.backend == "pl":
                chosen = pl_candidate
            else:
                chosen = min(candidates, key=lambda item: item.score)
            steps.extend(chosen.steps)
            residency = _apply_candidate_residency(residency, op, chosen.backend)
        for output in graph.outputs:
            want = graph.output_residency
            locations = residency.get(output, {"ps"})
            if want in FABRIC_LOCATIONS and want not in locations:
                spec = graph.buffers.get(output)
                elements = spec.elements if spec is not None else _output_elements(graph, output)
                steps.append(self._transfer_step(output, "pl" if want == "ps" else "ps", want, elements))
                residency.setdefault(output, set()).add(want)
        return FabricGraphPlan(
            graph=graph,
            steps=tuple(steps),
            final_residency={name: tuple(sorted(values)) for name, values in residency.items()},
            policy=self.policy,
            candidate_policy=self.candidate_policy,
        )

    def _candidate(
        self,
        graph: FabricGraph,
        op: FabricGraphOp,
        backend: str,
        residency: dict[str, set[str]],
        consumers: dict[str, list[int]],
        index: int,
    ) -> "_Candidate":
        steps: list[FabricGraphPlanStep] = []
        kernel = self.registry.select_kernel(op.op, active_app=self.active_app)
        unsupported = 0.0
        reason = "selected_by_policy"
        if backend == "pl" and kernel is None:
            unsupported = 1.0
            reason = "unsafe_pl_kernel" if self.registry.unsafe_specs_for_op(op.op) else "no_pl_kernel"
        if backend == "pl" and kernel is not None and kernel.fixed_elements is not None and kernel.fixed_elements != op.elements:
            unsupported = 1.0
            reason = "fixed_size_mismatch"
        target = backend
        transfer_bytes = 0
        transfer_latency = 0.0
        transfer_energy = 0.0
        if unsupported:
            latency = 1.0e6
            energy = 1.0e6
        else:
            for buffer in op.inputs:
                if target not in residency.get(buffer, {"ps"}):
                    elements = graph.buffers.get(buffer, FabricBufferSpec(buffer, op.elements)).elements
                    step = self._transfer_step(buffer, "pl" if target == "ps" else "ps", target, elements)
                    steps.append(step)
                    transfer_bytes += step.transfer_bytes
                    transfer_latency += step.latency_s
                    transfer_energy += step.energy_j
            if backend == "ps":
                latency = transfer_latency + self.cost_model.ps_seconds(op.op, op.elements)
                energy = transfer_energy + self.cost_model.ps_energy(op.op, op.elements)
            else:
                latency = transfer_latency + self.cost_model.pl_seconds(op.op, op.elements)
                energy = transfer_energy + self.cost_model.pl_energy(op.op, op.elements)
        output_consumers = consumers.get(op.output, [])
        downstream_pl_affinity = float(any(_op_has_pl_kernel(graph.ops[next_index], self.registry) for next_index in output_consumers))
        keeps_residency = float(bool(output_consumers) and backend == "pl" and downstream_pl_affinity)
        active_app_match = float(kernel is not None and self.active_app is not None and kernel.app_name == self.active_app)
        features = {
            "latency_s": latency,
            "energy_j": energy,
            "transfer_bytes": float(transfer_bytes),
            "unsupported": unsupported,
            "keeps_residency": keeps_residency,
            "active_app_match": active_app_match,
            "downstream_pl_affinity": downstream_pl_affinity if backend == "pl" else 0.0,
        }
        score = self.policy.score(features)
        action = "run_ps" if backend == "ps" else "run_pl"
        steps.append(
            FabricGraphPlanStep(
                action=action,
                backend=backend,
                op_id=op.op_id,
                op=op.op,
                kernel=kernel if backend == "pl" else None,
                reason=reason,
                latency_s=latency - transfer_latency,
                energy_j=energy - transfer_energy,
                score=score,
                features=features,
            )
        )
        return _Candidate(backend=backend, score=score, steps=tuple(steps))

    def _transfer_step(self, buffer: str, src: str, dst: str, elements: int) -> FabricGraphPlanStep:
        transfer_bytes = int(elements) * 4
        latency = transfer_bytes * self.cost_model.transfer_seconds_per_byte
        energy = transfer_bytes * self.cost_model.transfer_energy_j_per_byte
        return FabricGraphPlanStep(
            action="transfer",
            backend="fabric",
            buffer=buffer,
            src=src,
            dst=dst,
            reason=f"{src}_to_{dst}_for_residency",
            latency_s=latency,
            energy_j=energy,
            transfer_bytes=transfer_bytes,
        )


@dataclass(frozen=True)
class _Candidate:
    backend: str
    score: float
    steps: tuple[FabricGraphPlanStep, ...]


def _parse_residency(value: str) -> set[str]:
    if value == "both":
        return {"ps", "pl"}
    if value in FABRIC_LOCATIONS:
        return {value}
    return {"ps"}


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio_metric(metrics: dict[str, Any], ratio_key: str, numerator_key: str, denominator_key: str) -> float:
    direct = _float_or_none(metrics.get(ratio_key))
    if direct is not None:
        return direct
    numerator = _float_or_none(metrics.get(numerator_key))
    denominator = _float_or_none(metrics.get(denominator_key))
    if numerator is None or denominator is None:
        return float("inf")
    if denominator == 0.0:
        return 1.0 if numerator == 0.0 else float("inf")
    return numerator / denominator


def _apply_candidate_residency(
    residency: dict[str, set[str]],
    op: FabricGraphOp,
    backend: str,
) -> dict[str, set[str]]:
    updated = {name: set(values) for name, values in residency.items()}
    updated[op.output] = {backend}
    for name in op.inputs:
        updated.setdefault(name, {"ps"}).add(backend)
    return updated


def _consumer_map(graph: FabricGraph) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = {}
    for index, op in enumerate(graph.ops):
        for name in op.inputs:
            consumers.setdefault(name, []).append(index)
    return consumers


def _op_has_pl_kernel(op: FabricGraphOp, registry: FabricRegistry) -> bool:
    spec = registry.select_kernel(op.op)
    if spec is None:
        return False
    return spec.fixed_elements is None or spec.fixed_elements == op.elements


def _output_elements(graph: FabricGraph, output: str) -> int:
    for op in reversed(graph.ops):
        if op.output == output:
            return op.elements
    spec = graph.buffers.get(output)
    return 0 if spec is None else spec.elements


def _pl_regions(steps: Sequence[FabricGraphPlanStep]) -> list[dict[str, Any]]:
    regions: list[dict[str, Any]] = []
    current: Optional[dict[str, Any]] = None
    for index, step in enumerate(steps):
        if step.action != "run_pl" or step.kernel is None:
            if current is not None:
                regions.append(current)
                current = None
            continue
        app = step.kernel.app_name
        if current is None or current["app_name"] != app:
            if current is not None:
                regions.append(current)
            current = {"app_name": app, "start_step": index, "end_step": index, "ops": []}
        current["end_step"] = index
        current["ops"].append(step.op_id)
    if current is not None:
        regions.append(current)
    return regions


def _kernel_name_hint(op: str) -> str:
    if op.startswith("spf32."):
        return "spf32_" + op.split(".", 1)[1] + "_kernel"
    if op.startswith("spf16."):
        return "spf16_" + op.split(".", 1)[1] + "_kernel"
    return op.replace(".", "_") + "_kernel"
