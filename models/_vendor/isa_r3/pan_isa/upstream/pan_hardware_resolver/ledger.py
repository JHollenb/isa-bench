from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .direct_action import ACTION_SCHEMA_VERSION, ActionPacket, CommitResult
from .microprogram import MICROPROGRAM_SCHEMA_VERSION, MicroprogramCommitResult, MicroprogramPacket


ACTION_ENVELOPE_SCHEMA_VERSION = "pan-action-envelope-v1"
ACTION_LEDGER_SCHEMA_VERSION = "pan-action-ledger-v1"


@dataclass(frozen=True)
class ActionInput:
    state: dict[str, Any]
    goal: dict[str, Any]
    constraints: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        state = deepcopy(self.state)
        goal = deepcopy(self.goal)
        constraints = deepcopy(self.constraints)
        return {
            "state": state,
            "goal": goal,
            "constraints": constraints,
            "state_hash": stable_hash(state),
            "goal_hash": stable_hash(goal),
            "constraints_hash": stable_hash(constraints),
        }


@dataclass(frozen=True)
class ActionProposal:
    schema_version: str
    action_family: str
    op: str
    payload: dict[str, Any]
    expected_after: dict[str, Any]
    confidence: float = 1.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "action_family": self.action_family,
            "op": self.op,
            "payload": deepcopy(self.payload),
            "expected_after": deepcopy(self.expected_after),
            "confidence": float(self.confidence),
            "provenance": deepcopy(self.provenance),
        }
        payload["proposal_hash"] = stable_hash(payload)
        return payload


@dataclass(frozen=True)
class ActionVerification:
    ok: bool
    reason: str | None
    verifier: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        body = dict(deepcopy(self.payload))
        body.update({"ok": bool(self.ok), "reason": self.reason, "verifier": self.verifier})
        body.setdefault("safe_action", None)
        body.setdefault("rejected_action", None)
        body.setdefault("rejected_slot", None)
        return body


@dataclass(frozen=True)
class ActionCommit:
    ok: bool
    reason: str | None
    committed: dict[str, Any]
    before_state: dict[str, Any]
    after_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        before_state = deepcopy(self.before_state)
        after_state = deepcopy(self.after_state)
        return {
            "ok": bool(self.ok),
            "reason": self.reason,
            "committed": deepcopy(self.committed),
            "before_state_hash": stable_hash(before_state),
            "after_state_hash": stable_hash(after_state),
            "after_state": after_state,
        }


@dataclass(frozen=True)
class ActionLedgerRow:
    action_family: str
    input: ActionInput
    proposal: ActionProposal
    verification: ActionVerification
    commit: ActionCommit
    measurement: dict[str, Any] = field(default_factory=dict)
    lineage: dict[str, Any] = field(default_factory=dict)

    def envelope(self) -> dict[str, Any]:
        input_payload = self.input.to_dict()
        proposal_payload = self.proposal.to_dict()
        verification_payload = self.verification.to_dict()
        commit_payload = self.commit.to_dict()
        measurement_payload = {
            "environment": deepcopy(self.measurement.get("environment", {})),
            "metrics": deepcopy(self.measurement.get("metrics", {})),
        }
        body = {
            "schema_version": ACTION_ENVELOPE_SCHEMA_VERSION,
            "action_family": self.action_family,
            "input": input_payload,
            "proposal": proposal_payload,
            "verification": verification_payload,
            "commit": commit_payload,
            "measurement": measurement_payload,
        }
        body["action_id"] = "act_" + short_hash(body)
        return body

    def to_dict(self) -> dict[str, Any]:
        envelope = self.envelope()
        body = {
            "schema_version": ACTION_LEDGER_SCHEMA_VERSION,
            "action_family": self.action_family,
            "envelope": envelope,
            "lineage": {"producer": "pan_hardware_resolver", **deepcopy(self.lineage)},
        }
        body["ledger_id"] = "ledger_" + short_hash(body)
        return body


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def short_hash(payload: Any, length: int = 16) -> str:
    return stable_hash(payload)[:length]


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def direct_action_ledger_row(
    goal: dict[str, Any],
    constraints: dict[str, Any],
    proposed_action: ActionPacket,
    result: CommitResult,
    *,
    measurement: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proposal_payload = proposed_action.to_dict()
    verification_payload = result.verification.to_dict()
    committed = result.committed_action.to_dict()
    row = ActionLedgerRow(
        action_family=ACTION_SCHEMA_VERSION,
        input=ActionInput(result.before_state, goal, constraints),
        proposal=ActionProposal(
            schema_version=ACTION_SCHEMA_VERSION,
            action_family=ACTION_SCHEMA_VERSION,
            op=proposed_action.op,
            payload=proposal_payload,
            expected_after=proposal_payload.get("expected_after", {}),
            confidence=float(proposal_payload.get("confidence", 1.0)),
            provenance={"oracle": "direct_action_oracle"},
        ),
        verification=ActionVerification(
            bool(result.verification.ok),
            result.verification.reason,
            "pan_hardware_resolver.direct_action.verify_action",
            verification_payload,
        ),
        commit=ActionCommit(bool(result.ok), result.reason, committed, result.before_state, result.after_state),
        measurement=measurement or {},
        lineage=lineage or {},
    )
    payload = row.to_dict()
    problems = validate_ledger_row(payload)
    if problems:
        raise ValueError("invalid direct action ledger row: " + "; ".join(problems))
    return payload


def microprogram_ledger_row(
    goal: dict[str, Any],
    constraints: dict[str, Any],
    proposed_program: MicroprogramPacket,
    result: MicroprogramCommitResult,
    *,
    measurement: dict[str, Any] | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    proposal_payload = proposed_program.to_dict()
    verification_payload = result.verification.to_dict()
    if result.committed_program is not None:
        committed = result.committed_program.to_dict()
    elif result.recovery_action is not None:
        committed = result.recovery_action.to_dict()
    else:
        committed = {}
    row = ActionLedgerRow(
        action_family=MICROPROGRAM_SCHEMA_VERSION,
        input=ActionInput(result.before_state, goal, constraints),
        proposal=ActionProposal(
            schema_version=MICROPROGRAM_SCHEMA_VERSION,
            action_family=MICROPROGRAM_SCHEMA_VERSION,
            op="MICROPROGRAM",
            payload=proposal_payload,
            expected_after=proposal_payload.get("expected_after", {}),
            confidence=float(proposal_payload.get("confidence", 1.0)),
            provenance={"oracle": "microprogram_oracle"},
        ),
        verification=ActionVerification(
            bool(result.verification.ok),
            result.verification.reason,
            "pan_hardware_resolver.microprogram.verify_microprogram",
            verification_payload,
        ),
        commit=ActionCommit(bool(result.ok), result.reason, committed, result.before_state, result.after_state),
        measurement=measurement or {},
        lineage=lineage or {},
    )
    payload = row.to_dict()
    problems = validate_ledger_row(payload)
    if problems:
        raise ValueError("invalid microprogram ledger row: " + "; ".join(problems))
    return payload


def validate_ledger_row(row: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    _require(row, "schema_version", ACTION_LEDGER_SCHEMA_VERSION, problems)
    _require_present(row, "ledger_id", problems)
    _require_present(row, "action_family", problems)
    _require_present(row, "envelope", problems)
    _require_present(row, "lineage", problems)
    envelope = row.get("envelope", {})
    if isinstance(envelope, dict):
        _require(envelope, "schema_version", ACTION_ENVELOPE_SCHEMA_VERSION, problems)
        _require_present(envelope, "action_id", problems)
        _require_present(envelope, "action_family", problems)
        _require_present(envelope, "input", problems)
        _require_present(envelope, "proposal", problems)
        _require_present(envelope, "verification", problems)
        _require_present(envelope, "commit", problems)
        _require_present(envelope, "measurement", problems)
        if row.get("action_family") != envelope.get("action_family"):
            problems.append("row action_family must match envelope action_family")
        _validate_input(envelope.get("input", {}), problems)
        _validate_proposal(envelope.get("proposal", {}), problems)
        _validate_verification(envelope.get("verification", {}), problems)
        _validate_commit(envelope.get("commit", {}), problems)
    lineage = row.get("lineage", {})
    if isinstance(lineage, dict):
        _require_present(lineage, "producer", problems)
    else:
        problems.append("lineage must be an object")
    return problems


def write_ledger_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            problems = validate_ledger_row(row)
            if problems:
                raise ValueError("invalid ledger row: " + "; ".join(problems))
            handle.write(canonical_json(row) + "\n")


def _validate_input(payload: Any, problems: list[str]) -> None:
    if not isinstance(payload, dict):
        problems.append("input must be an object")
        return
    for key in ("state", "goal", "constraints", "state_hash", "goal_hash", "constraints_hash"):
        _require_present(payload, key, problems)
    if payload.get("state_hash") != stable_hash(payload.get("state", {})):
        problems.append("input state_hash mismatch")
    if payload.get("goal_hash") != stable_hash(payload.get("goal", {})):
        problems.append("input goal_hash mismatch")
    if payload.get("constraints_hash") != stable_hash(payload.get("constraints", {})):
        problems.append("input constraints_hash mismatch")


def _validate_proposal(payload: Any, problems: list[str]) -> None:
    if not isinstance(payload, dict):
        problems.append("proposal must be an object")
        return
    for key in ("schema_version", "action_family", "op", "payload", "expected_after", "confidence", "proposal_hash"):
        _require_present(payload, key, problems)
    expected_hash_payload = dict(payload)
    proposal_hash = expected_hash_payload.pop("proposal_hash", None)
    if proposal_hash != stable_hash(expected_hash_payload):
        problems.append("proposal_hash mismatch")


def _validate_verification(payload: Any, problems: list[str]) -> None:
    if not isinstance(payload, dict):
        problems.append("verification must be an object")
        return
    for key in ("ok", "reason", "verifier"):
        _require_present(payload, key, problems)
    if not isinstance(payload.get("ok"), bool):
        problems.append("verification ok must be boolean")


def _validate_commit(payload: Any, problems: list[str]) -> None:
    if not isinstance(payload, dict):
        problems.append("commit must be an object")
        return
    for key in ("ok", "reason", "committed", "before_state_hash", "after_state_hash", "after_state"):
        _require_present(payload, key, problems)
    if not isinstance(payload.get("ok"), bool):
        problems.append("commit ok must be boolean")
    if payload.get("after_state_hash") != stable_hash(payload.get("after_state", {})):
        problems.append("commit after_state_hash mismatch")


def _require(payload: dict[str, Any], key: str, expected: Any, problems: list[str]) -> None:
    if payload.get(key) != expected:
        problems.append(f"{key} must be {expected!r}")


def _require_present(payload: dict[str, Any], key: str, problems: list[str]) -> None:
    if key not in payload:
        problems.append(f"missing {key}")
