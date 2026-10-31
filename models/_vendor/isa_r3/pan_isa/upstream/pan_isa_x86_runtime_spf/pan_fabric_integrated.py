"""Integrated PAN fabric observations for normal x86 runtime execution."""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from .fabric_graph import FabricBufferSpec, FabricGraph, FabricGraphOp


PAN_FABRIC_TRACE_SCHEMA = "pan_fabric_runtime_trace_v1"
PAN_FABRIC_TRAINING_SCHEMA = "pan_fabric_runtime_training_record_v1"
PAN_FABRIC_MODES = ("auto", "off", "force-ps", "force-pl", "backend")


def backend_for_fabric_mode(
    fabric_mode: str,
    *,
    requested_backend: str,
    pl_xclbin: str | None = None,
) -> str:
    if fabric_mode not in PAN_FABRIC_MODES:
        raise ValueError(f"unsupported PAN fabric mode {fabric_mode!r}")
    if fabric_mode == "backend":
        return requested_backend
    if fabric_mode in {"off", "force-ps"}:
        return "ps"
    if fabric_mode == "auto":
        return "auto"
    if fabric_mode == "force-pl":
        return "pl-xrt" if pl_xclbin else "pl-sim"
    return requested_backend


def trace_record_from_route(
    routed: Any,
    stop_reason: Any,
    *,
    fabric_mode: str,
    committed: bool,
    fabric_daemon_socket: str | None = None,
) -> dict[str, Any]:
    decision = routed.decision.to_json()
    proposal = routed.proposal
    placement_action = _placement_action(routed.decision.selected_backend, committed=committed)
    graph = graph_from_route(routed)
    graph_id = graph_id_from_route(routed, graph)
    input_value_ids = {name: f"{graph_id}:{name}" for name in _graph_input_names(graph)}
    output_value_ids = {name: f"{graph_id}:{name}" for name in graph.outputs}
    output_residency = _residency_for_backend(routed.decision.selected_backend, committed=committed)
    return {
        "schema": PAN_FABRIC_TRACE_SCHEMA,
        "fabric_mode": fabric_mode,
        "fabric_daemon_socket": fabric_daemon_socket,
        "graph_id": graph_id,
        "graph": graph.to_json(),
        "graph_ops": [op.op for op in graph.ops],
        "committed": bool(committed),
        "stop_kind": getattr(stop_reason, "kind", None),
        "stop_detail": getattr(stop_reason, "detail", None),
        "region_kind": _region_kind(proposal),
        "proposal_kind": proposal.kind,
        "instruction_name": proposal.instruction_name,
        "instruction_count": int(routed.instruction_count),
        "placement_action": placement_action,
        "input_value_ids": input_value_ids,
        "output_value_ids": output_value_ids,
        "residency_before": {name: "ps" for name in input_value_ids},
        "residency_after": {name: output_residency for name in output_value_ids},
        "materialization_reason": _materialization_reason(routed.decision.selected_backend, committed=committed),
        "scheduler_hint": None,
        "validation": {
            "ps_authoritative_commit": bool(committed),
            "result": "committed" if committed else "rejected_fail_closed",
        },
        "decision": decision,
        "features": fabric_features_from_route(routed),
        "label": {
            "selected_backend": routed.decision.selected_backend,
            "placement_action": placement_action,
            "elapsed_seconds": float(routed.decision.elapsed_seconds),
            "transfer_to_pl_bytes": int(routed.decision.transfer_to_pl_bytes),
            "transfer_from_pl_bytes": int(routed.decision.transfer_from_pl_bytes),
            "commit_ok": bool(committed),
        },
    }


def training_record_from_route(
    routed: Any,
    stop_reason: Any,
    *,
    fabric_mode: str,
    committed: bool,
    fabric_daemon_socket: str | None = None,
) -> dict[str, Any]:
    trace = trace_record_from_route(
        routed,
        stop_reason,
        fabric_mode=fabric_mode,
        committed=committed,
        fabric_daemon_socket=fabric_daemon_socket,
    )
    return training_record_from_trace(trace, routed, stop_reason)


def training_record_from_trace(
    trace: dict[str, Any],
    routed: Any,
    stop_reason: Any,
) -> dict[str, Any]:
    return {
        "schema": PAN_FABRIC_TRAINING_SCHEMA,
        "source": "pan_isa_x86_runtime_spf",
        "fabric_mode": trace["fabric_mode"],
        "fabric_daemon_socket": trace["fabric_daemon_socket"],
        "graph_id": trace["graph_id"],
        "graph": trace["graph"],
        "graph_ops": trace["graph_ops"],
        "input_value_ids": trace["input_value_ids"],
        "output_value_ids": trace["output_value_ids"],
        "materialization_reason": trace["materialization_reason"],
        "validation": trace["validation"],
        "region_kind": trace["region_kind"],
        "backend": routed.decision.selected_backend,
        "placement_action": trace["placement_action"],
        "reason": routed.decision.reason,
        "features": trace["features"],
        "observed_seconds": float(routed.decision.elapsed_seconds),
        "observed_transfer_bytes": int(routed.decision.transfer_to_pl_bytes)
        + int(routed.decision.transfer_from_pl_bytes),
        "committed": bool(trace["committed"]),
        "stop_kind": trace["stop_kind"],
    }


def graph_from_route(routed: Any) -> FabricGraph:
    """Represent one normal-runtime routed region as a typed PAN fabric graph."""

    op_names = _typed_ops_from_route(routed)
    elements = max(1, int(getattr(routed, "instruction_count", 1) or 1))
    buffers: dict[str, FabricBufferSpec] = {
        "input0": FabricBufferSpec("input0", elements, "ps"),
    }
    ops: list[FabricGraphOp] = []
    previous = "input0"
    for index, op_name in enumerate(op_names):
        output = f"tmp{index}"
        buffers[output] = FabricBufferSpec(output, elements, "unknown")
        ops.append(
            FabricGraphOp(
                op_id=f"rt{index}",
                op=op_name,
                inputs=(previous,),
                output=output,
                elements=elements,
            )
        )
        previous = output
    return FabricGraph(
        buffers=buffers,
        ops=tuple(ops),
        outputs=(previous,),
        output_residency=_residency_for_backend(routed.decision.selected_backend, committed=True),
    )


def graph_id_from_route(routed: Any, graph: FabricGraph) -> str:
    seed = {
        "rip": getattr(routed.decision, "rip", 0),
        "instruction_count": int(getattr(routed, "instruction_count", 1) or 1),
        "graph": graph.to_json(),
    }
    digest = hashlib.sha256(json.dumps(seed, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return f"graph_{digest}"


def fabric_features_from_route(routed: Any) -> dict[str, float]:
    decision = routed.decision
    proposal = routed.proposal
    instruction_count = int(decision.instruction_count or routed.instruction_count or 1)
    transfer_bytes = int(decision.transfer_to_pl_bytes) + int(decision.transfer_from_pl_bytes)
    return {
        "eligible": 1.0 if decision.eligible else 0.0,
        "instruction_count": float(instruction_count),
        "is_simd_block": 1.0 if proposal.instruction_name == "PL_SIMD_BLOCK" else 0.0,
        "is_ps": 1.0 if decision.selected_backend == "ps" else 0.0,
        "is_pl": 1.0 if decision.selected_backend.startswith("pl") else 0.0,
        "template_hit": 1.0 if decision.template_hit else 0.0,
        "template_seen_count": float(decision.template_seen_count),
        "repeat_group_size": float(decision.repeat_group_size),
        "repeat_count": float(decision.repeat_count),
        "transfer_bytes": float(transfer_bytes),
        "transfer_to_pl_bytes": float(decision.transfer_to_pl_bytes),
        "transfer_from_pl_bytes": float(decision.transfer_from_pl_bytes),
        "memory_read_count": float(len(getattr(proposal, "memory_reads", ()) or ())),
        "memory_write_count": float(len(getattr(proposal, "memory_writes", ()) or ())),
        "xmm_write_count": float(len(getattr(proposal, "xmm_writes", {}) or {})),
    }


def _typed_ops_from_route(routed: Any) -> list[str]:
    proposal = routed.proposal
    names = list(getattr(routed.decision, "instruction_names", ()) or ())
    if not names and getattr(proposal, "operands", None):
        for item in proposal.operands:
            if isinstance(item, dict) and item.get("instruction_name"):
                names.append(str(item["instruction_name"]))
    if not names:
        names = [str(getattr(proposal, "instruction_name", "unknown"))]
    prefix = "x86.simd" if _region_kind(proposal) == "x86_simd_transition_block" else "x86"
    return [f"{prefix}.{_sanitize_op_name(name)}" for name in names]


def _sanitize_op_name(name: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(name).strip()).strip("_").lower()
    return value or "unknown"


def _graph_input_names(graph: FabricGraph) -> tuple[str, ...]:
    produced = {op.output for op in graph.ops}
    consumed = {name for op in graph.ops for name in op.inputs}
    return tuple(sorted(consumed - produced)) or ("input0",)


def _residency_for_backend(selected_backend: str, *, committed: bool) -> str:
    if not committed:
        return "unknown"
    if selected_backend in {"daemon", "daemon_session_resident"}:
        return "daemon"
    if selected_backend == "catalog_core":
        return "catalog_core"
    return "ps"


def _materialization_reason(selected_backend: str, *, committed: bool) -> str | None:
    if not committed:
        return "fault_recovery"
    if selected_backend.startswith("pl"):
        return "validation"
    return None


def _region_kind(proposal: Any) -> str:
    if getattr(proposal, "instruction_name", "") == "PL_SIMD_BLOCK":
        return "x86_simd_transition_block"
    stage = getattr(proposal, "stage", "")
    if stage == "stage4_simd_x87":
        return "x86_simd_or_x87_instruction"
    if stage == "stage2_control_flow":
        return "x86_control_or_syscall"
    return "x86_instruction"


def _placement_action(selected_backend: str, *, committed: bool) -> str:
    if not committed:
        return "reject_fail_closed"
    if selected_backend == "ps":
        return "run_ps_now"
    if selected_backend == "pl-sim":
        return "run_simulated_fabric_now"
    if selected_backend == "pl-xrt":
        return "run_existing_pl_now"
    if selected_backend.startswith("pl"):
        return "run_pl_now"
    return "run_existing_runtime_now"
