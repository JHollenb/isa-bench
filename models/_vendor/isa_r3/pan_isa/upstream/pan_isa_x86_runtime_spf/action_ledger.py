from __future__ import annotations

from copy import deepcopy
from typing import Any

from pan_hardware_resolver.ledger import (
    ActionCommit,
    ActionInput,
    ActionLedgerRow,
    ActionProposal,
    ActionVerification,
    validate_ledger_row,
)

from .fabric_graph import FabricGraphPlan
from .transitions import TransitionProposal, validate_transition_proposal


FABRIC_ACTION_SCHEMA_VERSION = "pan-fabric-action-v1"
FOUNDRY_ACTION_SCHEMA_VERSION = "pan-foundry-action-v1"
ISA_TRANSITION_SCHEMA_VERSION = "pan-isa-transition-v1"

FABRIC_ACTIONS = {
    "run_ps_now",
    "run_daemon_resident_now",
    "run_catalog_core_now",
    "keep_resident",
    "materialize",
    "compile_better_for_next_time",
    "reject_fail_closed",
}

UNSAFE_PLAN_REASONS = {"unsafe_pl_kernel", "no_pl_kernel", "fixed_size_mismatch"}

FOUNDRY_ACTIONS = {
    "mine_graph_region",
    "generate_candidate",
    "queue_build",
    "ingest_built_artifact",
    "measure_artifact",
    "promote_artifact",
    "quarantine_artifact",
    "reject_fail_closed",
}

FOUNDRY_TRANSITIONS = {
    "mine_graph_region": {(None, "region"), ("observed", "region")},
    "generate_candidate": {("region", "candidate")},
    "queue_build": {("candidate", "queued")},
    "ingest_built_artifact": {("queued", "built"), ("building", "built")},
    "measure_artifact": {("built", "measured"), ("built", "failed")},
    "promote_artifact": {("measured", "promoted")},
    "quarantine_artifact": {
        ("candidate", "quarantined"),
        ("queued", "quarantined"),
        ("building", "quarantined"),
        ("built", "quarantined"),
        ("measured", "quarantined"),
        ("failed", "quarantined"),
        ("promoted", "quarantined"),
    },
}


def fabric_action_from_plan(plan: FabricGraphPlan, *, confidence: float = 1.0) -> dict[str, Any]:
    plan_json = plan.to_json()
    selected_action = _selected_action(plan_json)
    action_plan = _action_plan(selected_action, plan_json)
    return {
        "schema_version": FABRIC_ACTION_SCHEMA_VERSION,
        "selected_action": selected_action,
        "action_plan": action_plan,
        "graph": plan.graph.to_json(),
        "policy": plan.policy.to_json(),
        "candidate_policy": None if plan.candidate_policy is None else plan.candidate_policy.to_json(),
        "plan": plan_json,
        "confidence": float(confidence),
    }


def verify_fabric_action(
    state: dict[str, Any], constraints: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    reason = _fabric_rejection_reason(state, constraints, action)
    safe_action = None
    if reason is not None:
        safe_action = {
            "schema_version": FABRIC_ACTION_SCHEMA_VERSION,
            "selected_action": "reject_fail_closed",
            "action_plan": ["reject_fail_closed"],
            "reason": reason,
        }
    return {
        "ok": reason is None,
        "reason": reason,
        "verifier": "pan_isa_x86_runtime_spf.action_ledger.verify_fabric_action",
        "safe_action": safe_action,
        "rejected_action": None if reason is None else action.get("selected_action"),
        "rejected_slot": None,
    }


def commit_fabric_action(
    state: dict[str, Any], constraints: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    verification = verify_fabric_action(state, constraints, action)
    after_state = deepcopy(state)
    if verification["ok"]:
        after_state["last_fabric_action"] = {
            "selected_action": action["selected_action"],
            "action_plan": list(action["action_plan"]),
            "summary": dict(action["plan"].get("summary", {})),
            "final_residency": dict(action["plan"].get("final_residency", {})),
        }
        committed = deepcopy(action)
    else:
        after_state["last_fabric_action"] = verification["safe_action"]
        committed = deepcopy(verification["safe_action"])
    return {
        "ok": bool(verification["ok"]),
        "reason": verification["reason"],
        "verification": verification,
        "committed": committed,
        "before_state": deepcopy(state),
        "after_state": after_state,
    }


def fabric_action_ledger_row(
    state: dict[str, Any],
    goal: dict[str, Any],
    constraints: dict[str, Any],
    plan: FabricGraphPlan,
    *,
    measurement: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = fabric_action_from_plan(plan)
    commit = commit_fabric_action(state, constraints, action)
    row = ActionLedgerRow(
        action_family=FABRIC_ACTION_SCHEMA_VERSION,
        input=ActionInput(commit["before_state"], goal, constraints),
        proposal=ActionProposal(
            schema_version=FABRIC_ACTION_SCHEMA_VERSION,
            action_family=FABRIC_ACTION_SCHEMA_VERSION,
            op=action["selected_action"],
            payload=action,
            expected_after={"last_fabric_action": commit["after_state"]["last_fabric_action"]},
            confidence=float(action.get("confidence", 1.0)),
            provenance={"planner": "FabricGraphPlanner"},
        ),
        verification=ActionVerification(
            bool(commit["verification"]["ok"]),
            commit["verification"]["reason"],
            commit["verification"]["verifier"],
            commit["verification"],
        ),
        commit=ActionCommit(
            bool(commit["ok"]),
            commit["reason"],
            commit["committed"],
            commit["before_state"],
            commit["after_state"],
        ),
        measurement=measurement or {},
        lineage={"producer": "pan_isa_x86_runtime_spf.action_ledger", **(lineage or {})},
    )
    payload = row.to_dict()
    problems = validate_ledger_row(payload)
    if problems:
        raise ValueError("invalid fabric action ledger row: " + "; ".join(problems))
    return payload


def foundry_action_packet(
    selected_action: str,
    subject: dict[str, Any],
    *,
    before_status: str | None,
    after_status: str | None,
    target: str = "kv260",
    validation: dict[str, Any] | None = None,
    confidence: float = 1.0,
) -> dict[str, Any]:
    return {
        "schema_version": FOUNDRY_ACTION_SCHEMA_VERSION,
        "selected_action": selected_action,
        "subject": deepcopy(subject),
        "before_status": before_status,
        "after_status": after_status,
        "target": target,
        "validation": dict(validation or {}),
        "confidence": float(confidence),
    }


def verify_foundry_action(
    state: dict[str, Any], constraints: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    reason = _foundry_rejection_reason(state, constraints, action)
    safe_action = None
    if reason is not None:
        safe_action = foundry_action_packet(
            "reject_fail_closed",
            action.get("subject", {}),
            before_status=action.get("before_status"),
            after_status=None,
            target=str(action.get("target", constraints.get("target", "kv260"))),
            validation={"reason": reason},
        )
    return {
        "ok": reason is None,
        "reason": reason,
        "verifier": "pan_isa_x86_runtime_spf.action_ledger.verify_foundry_action",
        "safe_action": safe_action,
        "rejected_action": None if reason is None else action.get("selected_action"),
        "rejected_slot": None,
    }


def commit_foundry_action(
    state: dict[str, Any], constraints: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    verification = verify_foundry_action(state, constraints, action)
    after_state = deepcopy(state)
    if verification["ok"]:
        after_state["last_foundry_action"] = {
            "selected_action": action["selected_action"],
            "subject_id": _foundry_subject_id(action.get("subject", {})),
            "before_status": action.get("before_status"),
            "after_status": action.get("after_status"),
            "target": action.get("target"),
        }
        committed = deepcopy(action)
    else:
        after_state["last_foundry_action"] = verification["safe_action"]
        committed = deepcopy(verification["safe_action"])
    return {
        "ok": bool(verification["ok"]),
        "reason": verification["reason"],
        "verification": verification,
        "committed": committed,
        "before_state": deepcopy(state),
        "after_state": after_state,
    }


def foundry_action_ledger_row(
    state: dict[str, Any],
    goal: dict[str, Any],
    constraints: dict[str, Any],
    action: dict[str, Any],
    *,
    measurement: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    commit = commit_foundry_action(state, constraints, action)
    row = ActionLedgerRow(
        action_family=FOUNDRY_ACTION_SCHEMA_VERSION,
        input=ActionInput(commit["before_state"], goal, constraints),
        proposal=ActionProposal(
            schema_version=FOUNDRY_ACTION_SCHEMA_VERSION,
            action_family=FOUNDRY_ACTION_SCHEMA_VERSION,
            op=action["selected_action"],
            payload=action,
            expected_after={"last_foundry_action": commit["after_state"]["last_foundry_action"]},
            confidence=float(action.get("confidence", 1.0)),
            provenance={"foundry": "FabricFoundryCatalog"},
        ),
        verification=ActionVerification(
            bool(commit["verification"]["ok"]),
            commit["verification"]["reason"],
            commit["verification"]["verifier"],
            commit["verification"],
        ),
        commit=ActionCommit(
            bool(commit["ok"]),
            commit["reason"],
            commit["committed"],
            commit["before_state"],
            commit["after_state"],
        ),
        measurement=measurement or {},
        lineage={"producer": "pan_isa_x86_runtime_spf.action_ledger", **(lineage or {})},
    )
    payload = row.to_dict()
    problems = validate_ledger_row(payload)
    if problems:
        raise ValueError("invalid foundry action ledger row: " + "; ".join(problems))
    return payload


def isa_transition_action_packet(proposal: TransitionProposal, *, confidence: float | None = None) -> dict[str, Any]:
    proposal_json = proposal.to_json()
    return {
        "schema_version": ISA_TRANSITION_SCHEMA_VERSION,
        "kind": proposal.kind,
        "instruction": {
            "name": proposal.instruction_name,
            "hex": proposal.instruction_hex,
            "rip": hex(int(proposal.rip)),
            "next_rip": proposal_json.get("next_rip"),
            "stage": proposal.stage,
        },
        "proposal": proposal_json,
        "profile": proposal.profile or proposal.spf_manifest.get("target_profile") or proposal.spf_manifest.get("profile"),
        "confidence": float(proposal.confidence if confidence is None else confidence),
    }


def verify_isa_transition_action(
    state: dict[str, Any],
    constraints: dict[str, Any],
    action: dict[str, Any],
    proposal: TransitionProposal | None = None,
) -> dict[str, Any]:
    reason = _isa_transition_rejection_reason(state, constraints, action, proposal)
    safe_action = None
    if reason is not None:
        safe_action = {
            "schema_version": ISA_TRANSITION_SCHEMA_VERSION,
            "kind": "reject_fail_closed",
            "instruction": deepcopy(action.get("instruction", {})),
            "proposal": {},
            "profile": action.get("profile"),
            "confidence": 0.0,
            "reason": reason,
        }
    return {
        "ok": reason is None,
        "reason": reason,
        "verifier": "pan_isa_x86_runtime_spf.action_ledger.verify_isa_transition_action",
        "safe_action": safe_action,
        "rejected_action": None if reason is None else action.get("kind"),
        "rejected_slot": None,
    }


def commit_isa_transition_action(
    state: dict[str, Any],
    constraints: dict[str, Any],
    action: dict[str, Any],
    proposal: TransitionProposal | None = None,
) -> dict[str, Any]:
    verification = verify_isa_transition_action(state, constraints, action, proposal)
    after_state = deepcopy(state)
    if verification["ok"]:
        after_state["last_isa_transition"] = {
            "kind": action["kind"],
            "instruction": deepcopy(action["instruction"]),
            "profile": action.get("profile"),
            "confidence": float(action.get("confidence", 1.0)),
        }
        committed = deepcopy(action)
    else:
        after_state["last_isa_transition"] = verification["safe_action"]
        committed = deepcopy(verification["safe_action"])
    return {
        "ok": bool(verification["ok"]),
        "reason": verification["reason"],
        "verification": verification,
        "committed": committed,
        "before_state": deepcopy(state),
        "after_state": after_state,
    }


def isa_transition_ledger_row(
    state: dict[str, Any],
    goal: dict[str, Any],
    constraints: dict[str, Any],
    proposal: TransitionProposal,
    *,
    measurement: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action = isa_transition_action_packet(proposal)
    commit = commit_isa_transition_action(state, constraints, action, proposal)
    row = ActionLedgerRow(
        action_family=ISA_TRANSITION_SCHEMA_VERSION,
        input=ActionInput(commit["before_state"], goal, constraints),
        proposal=ActionProposal(
            schema_version=ISA_TRANSITION_SCHEMA_VERSION,
            action_family=ISA_TRANSITION_SCHEMA_VERSION,
            op=action["kind"],
            payload=action,
            expected_after={"last_isa_transition": commit["after_state"]["last_isa_transition"]},
            confidence=float(action.get("confidence", 1.0)),
            provenance={"runtime": "SPFPrimaryPanRuntime", "validator": "validate_transition_proposal"},
        ),
        verification=ActionVerification(
            bool(commit["verification"]["ok"]),
            commit["verification"]["reason"],
            commit["verification"]["verifier"],
            commit["verification"],
        ),
        commit=ActionCommit(
            bool(commit["ok"]),
            commit["reason"],
            commit["committed"],
            commit["before_state"],
            commit["after_state"],
        ),
        measurement=measurement or {},
        lineage={"producer": "pan_isa_x86_runtime_spf.action_ledger", **(lineage or {})},
    )
    payload = row.to_dict()
    problems = validate_ledger_row(payload)
    if problems:
        raise ValueError("invalid ISA transition action ledger row: " + "; ".join(problems))
    return payload


def _isa_transition_rejection_reason(
    state: dict[str, Any],
    constraints: dict[str, Any],
    action: dict[str, Any],
    proposal: TransitionProposal | None,
) -> str | None:
    if action.get("schema_version") != ISA_TRANSITION_SCHEMA_VERSION:
        return "schema_version_mismatch"
    proposal_json = action.get("proposal")
    if not isinstance(proposal_json, dict):
        return "proposal_missing"
    kind = str(action.get("kind", ""))
    if not kind:
        return "transition_kind_missing"
    if proposal_json.get("kind") is not None and proposal_json.get("kind") != kind:
        return "transition_kind_mismatch"
    instruction = action.get("instruction")
    if not isinstance(instruction, dict):
        return "instruction_missing"
    for field_name in ("name", "hex", "rip"):
        if not isinstance(instruction.get(field_name), str) or not instruction[field_name]:
            return f"instruction_{field_name}_missing"
    if action.get("confidence") is None:
        return "confidence_missing"
    try:
        confidence = float(action["confidence"])
    except (TypeError, ValueError):
        return "confidence_invalid"
    exact_confidence = float(constraints.get("exact_confidence", 1.0))
    if confidence < exact_confidence:
        return "confidence_below_exact_threshold"
    expected_profile = str(constraints.get("expected_profile", "long64-ring3"))
    action_profile = action.get("profile") or proposal_json.get("profile")
    if action_profile is not None and action_profile != expected_profile:
        return f"SPF transition profile {action_profile!r} does not match runtime profile {expected_profile!r}"
    if constraints.get("active_quarantine"):
        return "active_quarantine"
    if state.get("force_reject"):
        return "state_forced_reject"
    if proposal is None:
        return None
    stop_reason = validate_transition_proposal(
        proposal,
        expected_profile=expected_profile,
        exact_confidence=exact_confidence,
        require_learned=bool(constraints.get("require_learned", False)),
    )
    if stop_reason is None:
        return None
    return stop_reason.detail or stop_reason.kind


def _selected_action(plan_json: dict[str, Any]) -> str:
    steps = list(plan_json.get("steps", []))
    if any(step.get("reason") in UNSAFE_PLAN_REASONS for step in steps):
        return "reject_fail_closed"
    if any(step.get("action") == "run_pl" for step in steps):
        return "run_catalog_core_now"
    if any(step.get("action") == "run_ps" for step in steps):
        return "run_ps_now"
    return "reject_fail_closed"


def _action_plan(selected_action: str, plan_json: dict[str, Any]) -> list[str]:
    actions = [selected_action]
    final_residency = plan_json.get("final_residency", {})
    graph_outputs = set(plan_json.get("graph", {}).get("outputs", []))
    if any("pl" in set(final_residency.get(output, [])) for output in graph_outputs):
        actions.append("keep_resident")
    if any(step.get("action") == "transfer" and step.get("dst") == "ps" for step in plan_json.get("steps", [])):
        actions.append("materialize")
    if selected_action == "reject_fail_closed":
        return ["reject_fail_closed"]
    return actions


def _fabric_rejection_reason(
    state: dict[str, Any], constraints: dict[str, Any], action: dict[str, Any]
) -> str | None:
    if action.get("schema_version") != FABRIC_ACTION_SCHEMA_VERSION:
        return "schema_version_mismatch"
    selected_action = str(action.get("selected_action", ""))
    if selected_action not in FABRIC_ACTIONS:
        return "unknown_fabric_action"
    action_plan = list(action.get("action_plan", []))
    if not action_plan or action_plan[0] != selected_action:
        return "action_plan_mismatch"
    if constraints.get("active_quarantine") and selected_action != "reject_fail_closed":
        return "active_quarantine"
    graph = action.get("graph", {})
    if graph.get("schema") != "pan_fabric_graph_v1":
        return "graph_schema_mismatch"
    plan = action.get("plan", {})
    if plan.get("schema") != "pan_fabric_graph_plan_v1":
        return "plan_schema_mismatch"
    for step in plan.get("steps", []):
        if step.get("reason") in UNSAFE_PLAN_REASONS and selected_action != "reject_fail_closed":
            return str(step.get("reason"))
    candidate_policy = action.get("candidate_policy")
    if isinstance(candidate_policy, dict):
        gating = candidate_policy.get("gating", {})
        if gating.get("can_override_rule_scheduler") and action.get("policy", {}).get("policy_role") == "rule_authority":
            return "candidate_policy_authority_mismatch"
    if state.get("force_reject") and selected_action != "reject_fail_closed":
        return "state_forced_reject"
    return None


def _foundry_rejection_reason(
    state: dict[str, Any], constraints: dict[str, Any], action: dict[str, Any]
) -> str | None:
    if action.get("schema_version") != FOUNDRY_ACTION_SCHEMA_VERSION:
        return "schema_version_mismatch"
    selected_action = str(action.get("selected_action", ""))
    if selected_action not in FOUNDRY_ACTIONS:
        return "unknown_foundry_action"
    if selected_action == "reject_fail_closed":
        return None
    target = str(action.get("target", ""))
    expected_target = constraints.get("target")
    if expected_target is not None and target != str(expected_target):
        return "target_mismatch"
    if constraints.get("active_quarantine") and selected_action != "quarantine_artifact":
        return "active_quarantine"
    transition = (action.get("before_status"), action.get("after_status"))
    allowed = FOUNDRY_TRANSITIONS.get(selected_action, set())
    if transition not in allowed:
        return "illegal_foundry_transition"
    validation = dict(action.get("validation") or {})
    if selected_action == "promote_artifact":
        if not validation.get("measurement_ok"):
            return "promotion_measurement_missing"
        if validation.get("validation_failures", 1) != 0:
            return "promotion_validation_failures"
    if selected_action == "measure_artifact" and action.get("after_status") == "measured":
        if not validation.get("measurement_ok"):
            return "measurement_not_ok"
    if state.get("force_reject") and selected_action != "quarantine_artifact":
        return "state_forced_reject"
    return None


def _foundry_subject_id(subject: dict[str, Any]) -> str | None:
    for key in ("candidate_id", "region_id", "build_id", "artifact_id", "quarantine_id"):
        if key in subject:
            return str(subject[key])
    return None
