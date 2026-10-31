from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .direct_action import (
    ACTION_SCHEMA_VERSION,
    ActionPacket,
    CommitResult,
    VerificationError,
    _as_int,
    _fault_active,
    _hex_key,
    _is_subset,
    _safe_action,
    commit_action,
    simulate_action,
    verify_action,
)


MICROPROGRAM_SCHEMA_VERSION = "pan-hw-microprogram-v1"

MICROPROGRAM_OPS = {
    "NOOP",
    "FAULT_HOLD",
    "WRITE_REG",
    "SET_PWM",
    "MOVE_AXIS",
    "DMA_COPY",
    "SET_NCO",
    "LOAD_FILTER",
    "SET_GAIN",
    "BUS_TX",
    "WAIT_EVENT",
    "CHECK_FLAG",
}

DEFAULT_SLOT_COUNT = 8
ALLOWED_GUARDS = {"always", "no_fault", "event_seen"}


@dataclass(frozen=True)
class MicroSlot:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    guard: str = "always"
    expected_delta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MicroSlot":
        return cls(
            op=str(payload.get("op", "")),
            params=dict(payload.get("params", {})),
            guard=str(payload.get("guard", "always")),
            expected_delta=dict(payload.get("expected_delta", payload.get("expected_after", {}))),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "params": deepcopy(self.params),
            "guard": self.guard,
            "expected_delta": deepcopy(self.expected_delta),
        }


@dataclass(frozen=True)
class MicroprogramPacket:
    program: list[MicroSlot]
    expected_after: dict[str, Any] = field(default_factory=dict)
    slot_count: int = DEFAULT_SLOT_COUNT
    confidence: float = 1.0
    schema_version: str = MICROPROGRAM_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MicroprogramPacket":
        program = [MicroSlot.from_dict(slot) for slot in payload.get("program", [])]
        slot_count = int(payload.get("slot_count", len(program) or DEFAULT_SLOT_COUNT))
        return cls(
            schema_version=str(payload.get("schema_version", MICROPROGRAM_SCHEMA_VERSION)),
            program=program,
            expected_after=dict(payload.get("expected_after", {})),
            slot_count=slot_count,
            confidence=float(payload.get("confidence", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "slot_count": self.slot_count,
            "program": [slot.to_dict() for slot in self.program],
            "expected_after": deepcopy(self.expected_after),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class MicroprogramVerificationResult:
    ok: bool
    reason: str | None = None
    rejected_slot: int | None = None
    safe_action: ActionPacket | None = None
    normalized_program: MicroprogramPacket | None = None
    expected_after_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "reason": self.reason,
            "rejected_slot": self.rejected_slot,
        }
        if self.safe_action is not None:
            payload["safe_action"] = self.safe_action.to_dict()
        if self.normalized_program is not None:
            payload["normalized_program"] = self.normalized_program.to_dict()
        if self.expected_after_state:
            payload["expected_after_state"] = deepcopy(self.expected_after_state)
        return payload


@dataclass(frozen=True)
class MicroprogramCommitResult:
    ok: bool
    reason: str | None
    proposed_program: MicroprogramPacket
    committed_program: MicroprogramPacket | None
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    verification: MicroprogramVerificationResult
    slot_after_states: list[dict[str, Any]] = field(default_factory=list)
    recovery_action: ActionPacket | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "reason": self.reason,
            "proposed_program": self.proposed_program.to_dict(),
            "committed_program": self.committed_program.to_dict() if self.committed_program is not None else None,
            "before_state": deepcopy(self.before_state),
            "after_state": deepcopy(self.after_state),
            "verification": self.verification.to_dict(),
            "slot_after_states": deepcopy(self.slot_after_states),
        }
        if self.recovery_action is not None:
            payload["recovery_action"] = self.recovery_action.to_dict()
        return payload


def microprogram_oracle(
    state: dict[str, Any], goal: dict[str, Any], constraints: dict[str, Any] | None = None
) -> MicroprogramPacket:
    constraints = constraints or {}
    slot_count = _slot_count(constraints, goal.get("slot_count"))
    if _fault_active(state, constraints):
        return _pad_program(
            [MicroSlot("FAULT_HOLD", {"reason_code": 1}, "always", {"safe_state": "fault_hold"})],
            slot_count,
            {"safe_state": "fault_hold"},
        )

    kind = str(goal.get("kind", "")).lower()
    if kind == "pll_switch":
        pll_device = str(goal.get("pll_device", "pll0"))
        pll_address = _as_int(goal.get("pll_address", "0x10"), "pll_address")
        pll_value = _as_int(goal.get("pll_value", "0x31"), "pll_value")
        pll_mask = _as_int(goal.get("pll_mask", "0xff"), "pll_mask")
        lock_event = str(goal.get("lock_event", f"{pll_device}_locked"))
        mux_device = str(goal.get("mux_device", "clock_mux"))
        mux_address = _as_int(goal.get("mux_address", "0x04"), "mux_address")
        mux_value = _as_int(goal.get("mux_value", "0x2"), "mux_value")
        mux_mask = _as_int(goal.get("mux_mask", "0x3"), "mux_mask")
        timeout = _as_int(goal.get("timeout_ticks", _default_event_timeout(constraints)), "timeout_ticks")
        clock_source = str(goal.get("clock_source", pll_device))
        slots = [
            MicroSlot(
                "WRITE_REG",
                {
                    "device": pll_device,
                    "address": _hex_key(pll_address),
                    "value": _hex_key(pll_value),
                    "mask": _hex_key(pll_mask),
                },
                "no_fault",
                {
                    "registers": {
                        pll_device: {
                            _hex_key(pll_address): _masked_register_value_from_slot(
                                state, constraints, pll_device, pll_address, pll_value, pll_mask
                            )
                        }
                    }
                },
            ),
            MicroSlot(
                "WAIT_EVENT",
                {"event": lock_event, "timeout_ticks": timeout},
                "no_fault",
                {"events": {lock_event: True}, "events_seen": [lock_event]},
            ),
            MicroSlot(
                "WRITE_REG",
                {
                    "device": mux_device,
                    "address": _hex_key(mux_address),
                    "value": _hex_key(mux_value),
                    "mask": _hex_key(mux_mask),
                },
                "event_seen",
                {"registers": {mux_device: {_hex_key(mux_address): mux_value & mux_mask}}, "clock_source": clock_source},
            ),
        ]
        return _pad_program(
            slots,
            slot_count,
            {
                "events": {lock_event: True},
                "clock_source": clock_source,
                "registers": {mux_device: {_hex_key(mux_address): mux_value & mux_mask}},
            },
        )
    if kind == "filter_dma_start":
        bank = str(goal["bank"])
        coefficient_hash = str(goal["coefficient_hash"])
        src = str(goal["src"])
        dst = str(goal["dst"])
        length = _as_int(goal["length"], "length")
        event = str(goal.get("event", "dma_complete"))
        timeout = _as_int(goal.get("timeout_ticks", _default_event_timeout(constraints)), "timeout_ticks")
        slots = [
            MicroSlot(
                "LOAD_FILTER",
                {"bank": bank, "coefficient_hash": coefficient_hash},
                "no_fault",
                {"filters": {bank: {"coefficient_hash": coefficient_hash}}},
            ),
            MicroSlot(
                "DMA_COPY",
                {"src": src, "dst": dst, "length": length, "flags": _as_int(goal.get("flags", 0), "flags")},
                "no_fault",
                {"dma": {"last_copy": {"src": src, "dst": dst}}},
            ),
            MicroSlot("WAIT_EVENT", {"event": event, "timeout_ticks": timeout}, "no_fault", {"events": {event: True}}),
        ]
        return _pad_program(slots, slot_count, {"events": {event: True}, "filters": {bank: {"coefficient_hash": coefficient_hash}}})

    raise ValueError(f"unsupported microprogram goal kind: {kind!r}")


def verify_microprogram(
    state: dict[str, Any], constraints: dict[str, Any] | None, packet: MicroprogramPacket | dict[str, Any]
) -> MicroprogramVerificationResult:
    constraints = constraints or {}
    program = MicroprogramPacket.from_dict(packet) if isinstance(packet, dict) else packet
    try:
        normalized = _normalize_microprogram(program, constraints)
        sim_state = deepcopy(state)
        seen_events = _seen_events(sim_state)
        total_timeout = 0

        for index, slot in enumerate(normalized.program):
            if not _guard_satisfied(slot.guard, sim_state, seen_events, constraints):
                raise _slot_error("guard_not_satisfied", index, slot.guard)
            if slot.op == "WAIT_EVENT":
                timeout = _as_int(slot.params["timeout_ticks"], "timeout_ticks")
                total_timeout += timeout
                _verify_timeout_budget(constraints, timeout, total_timeout, index)
                _verify_event_allowed(constraints, str(slot.params["event"]), index)
                sim_state = _simulate_wait_event(sim_state, str(slot.params["event"]))
            elif slot.op == "CHECK_FLAG":
                _verify_check_flag(sim_state, constraints, slot, index)
            else:
                action = ActionPacket(
                    schema_version=ACTION_SCHEMA_VERSION,
                    op=slot.op,
                    params=slot.params,
                    expected_after=slot.expected_delta,
                )
                result = verify_action(sim_state, constraints, action)
                if not result.ok or result.normalized_action is None:
                    raise _slot_error(result.reason or "invalid_action", index, slot.op)
                _verify_sequence_ordering(seen_events, result.normalized_action, constraints, index)
                sim_state = simulate_action(sim_state, result.normalized_action, constraints)

            seen_events = _seen_events(sim_state)
            if slot.expected_delta and not _is_subset(slot.expected_delta, sim_state):
                raise _slot_error("expected_delta_mismatch", index, slot.op)

        if normalized.expected_after and not _is_subset(normalized.expected_after, sim_state):
            raise VerificationError("expected_after_mismatch", "program final state")
        return MicroprogramVerificationResult(True, normalized_program=normalized, expected_after_state=sim_state)
    except _SlotVerificationError as exc:
        return MicroprogramVerificationResult(
            False,
            exc.reason,
            exc.slot_index,
            _safe_action(exc.reason, exc.detail),
        )
    except VerificationError as exc:
        return MicroprogramVerificationResult(False, exc.reason, None, _safe_action(exc.reason, exc.detail))


def commit_microprogram(
    state: dict[str, Any],
    constraints: dict[str, Any] | None,
    packet: MicroprogramPacket | dict[str, Any],
    *,
    commit_safe_action: bool = True,
) -> MicroprogramCommitResult:
    constraints = constraints or {}
    program = MicroprogramPacket.from_dict(packet) if isinstance(packet, dict) else packet
    before = deepcopy(state)
    verification = verify_microprogram(before, constraints, program)
    if not verification.ok or verification.normalized_program is None:
        safe = verification.safe_action or _safe_action(verification.reason or "invalid_params")
        after = simulate_action(before, safe, constraints) if commit_safe_action else before
        return MicroprogramCommitResult(
            False,
            verification.reason,
            program,
            None,
            before,
            after,
            verification,
            [],
            safe if commit_safe_action else None,
        )

    current = before
    slot_after_states: list[dict[str, Any]] = []
    seen_events = _seen_events(current)
    for index, slot in enumerate(verification.normalized_program.program):
        if not _guard_satisfied(slot.guard, current, seen_events, constraints):
            return _commit_recovery(program, verification, before, current, constraints, slot_after_states, "guard_not_satisfied")
        if slot.op == "WAIT_EVENT":
            current = _simulate_wait_event(current, str(slot.params["event"]))
        elif slot.op == "CHECK_FLAG":
            if not _check_flag_matches(current, str(slot.params["flag_id"]), bool(slot.params["expected_value"])):
                return _commit_recovery(program, verification, before, current, constraints, slot_after_states, "check_flag_mismatch")
        else:
            action = ActionPacket(op=slot.op, params=slot.params, expected_after=slot.expected_delta)
            action_result: CommitResult = commit_action(current, constraints, action)
            if not action_result.ok:
                return _commit_recovery(program, verification, before, current, constraints, slot_after_states, action_result.reason)
            current = action_result.after_state

        seen_events = _seen_events(current)
        if slot.expected_delta and not _is_subset(slot.expected_delta, current):
            return _commit_recovery(program, verification, before, current, constraints, slot_after_states, "expected_delta_mismatch")
        slot_after_states.append(deepcopy(current))

    if verification.normalized_program.expected_after and not _is_subset(verification.normalized_program.expected_after, current):
        return _commit_recovery(program, verification, before, current, constraints, slot_after_states, "expected_after_mismatch")
    return MicroprogramCommitResult(
        True,
        None,
        program,
        verification.normalized_program,
        before,
        current,
        verification,
        slot_after_states,
    )


def _normalize_microprogram(packet: MicroprogramPacket, constraints: dict[str, Any]) -> MicroprogramPacket:
    if packet.schema_version != MICROPROGRAM_SCHEMA_VERSION:
        raise VerificationError("schema_version_mismatch", packet.schema_version)
    expected_slots = _slot_count(constraints, packet.slot_count)
    if packet.slot_count != expected_slots:
        raise VerificationError("invalid_params", "slot_count")
    if len(packet.program) != expected_slots:
        raise VerificationError("invalid_params", "program must include explicit fixed slots")

    normalized_slots = []
    for index, slot in enumerate(packet.program):
        if slot.op not in MICROPROGRAM_OPS:
            raise _slot_error("unknown_op", index, slot.op)
        if slot.guard not in ALLOWED_GUARDS:
            raise _slot_error("invalid_guard", index, slot.guard)
        params = dict(slot.params)
        if slot.op == "WAIT_EVENT":
            params = {"event": str(params["event"]), "timeout_ticks": _as_int(params["timeout_ticks"], "timeout_ticks")}
        elif slot.op == "CHECK_FLAG":
            params = {"flag_id": str(params["flag_id"]), "expected_value": bool(params["expected_value"])}
        else:
            normalized = verify_action({}, constraints, ActionPacket(op=slot.op, params=params))
            if not normalized.ok or normalized.normalized_action is None:
                raise _slot_error(normalized.reason or "invalid_action", index, slot.op)
            params = normalized.normalized_action.params
        normalized_slots.append(MicroSlot(slot.op, params, slot.guard, deepcopy(slot.expected_delta)))
    return MicroprogramPacket(
        schema_version=packet.schema_version,
        program=normalized_slots,
        expected_after=deepcopy(packet.expected_after),
        slot_count=packet.slot_count,
        confidence=packet.confidence,
    )


def _pad_program(slots: list[MicroSlot], slot_count: int, expected_after: dict[str, Any]) -> MicroprogramPacket:
    padded = slots[:]
    while len(padded) < slot_count:
        padded.append(MicroSlot("NOOP", {}, "always", {}))
    if len(padded) > slot_count:
        raise ValueError("program exceeds fixed slot count")
    return MicroprogramPacket(program=padded, expected_after=expected_after, slot_count=slot_count)


def _slot_count(constraints: dict[str, Any], requested: Any = None) -> int:
    configured = constraints.get("microprogram", {}).get("slot_count")
    if configured is not None:
        return _as_int(configured, "slot_count")
    if requested is not None:
        return _as_int(requested, "slot_count")
    return DEFAULT_SLOT_COUNT


def _default_event_timeout(constraints: dict[str, Any]) -> int:
    events = constraints.get("events", {})
    return _as_int(events.get("default_timeout_ticks", 5000), "default_timeout_ticks")


def _simulate_wait_event(state: dict[str, Any], event: str) -> dict[str, Any]:
    after = deepcopy(state)
    events = after.setdefault("events", {})
    events[event] = True
    seen = list(after.get("events_seen", []))
    if event not in seen:
        seen.append(event)
    after["events_seen"] = seen
    if event.endswith("_locked"):
        after[event.removesuffix("_locked") + "_lock"] = True
    return after


def _seen_events(state: dict[str, Any]) -> set[str]:
    seen = {str(event) for event, value in state.get("events", {}).items() if value}
    seen.update(str(event) for event in state.get("events_seen", []))
    return seen


def _guard_satisfied(
    guard: str, state: dict[str, Any], seen_events: set[str], constraints: dict[str, Any]
) -> bool:
    if guard == "always":
        return True
    if guard == "no_fault":
        return not _fault_active(state, constraints)
    if guard == "event_seen":
        return bool(seen_events)
    return False


def _verify_timeout_budget(constraints: dict[str, Any], timeout: int, total_timeout: int, slot_index: int) -> None:
    events = constraints.get("events", {})
    if timeout <= 0:
        raise _slot_error("range_violation", slot_index, "timeout_ticks")
    max_timeout = _as_int(events.get("max_timeout_ticks", timeout), "max_timeout_ticks")
    if timeout > max_timeout:
        raise _slot_error("range_violation", slot_index, "timeout_ticks")
    watchdog = _as_int(events.get("watchdog_budget_ticks", total_timeout), "watchdog_budget_ticks")
    if total_timeout > watchdog:
        raise _slot_error("watchdog_budget_exceeded", slot_index, "timeout_ticks")


def _verify_event_allowed(constraints: dict[str, Any], event: str, slot_index: int) -> None:
    allowed = {str(item) for item in constraints.get("events", {}).get("allowed", [])}
    if allowed and event not in allowed:
        raise _slot_error("allowlist_violation", slot_index, event)


def _verify_check_flag(state: dict[str, Any], constraints: dict[str, Any], slot: MicroSlot, slot_index: int) -> None:
    flag_id = str(slot.params["flag_id"])
    allowed = {str(item) for item in constraints.get("flags", {}).get("allowed", [])}
    if allowed and flag_id not in allowed:
        raise _slot_error("allowlist_violation", slot_index, flag_id)
    if not _check_flag_matches(state, flag_id, bool(slot.params["expected_value"])):
        raise _slot_error("check_flag_mismatch", slot_index, flag_id)


def _check_flag_matches(state: dict[str, Any], flag_id: str, expected_value: bool) -> bool:
    flags = state.get("flags", {})
    if flag_id not in flags:
        return False
    return bool(flags[flag_id]) is expected_value


def _verify_sequence_ordering(
    seen_events: set[str], action: ActionPacket, constraints: dict[str, Any], slot_index: int
) -> None:
    for requirement in _ordering_requirements(constraints):
        if action.op != requirement.get("op"):
            continue
        params = action.params
        if "device" in requirement and str(params.get("device")) != str(requirement["device"]):
            continue
        if "address" in requirement and _as_int(params.get("address"), "address") != _as_int(requirement["address"], "address"):
            continue
        required_event = str(requirement["event"])
        if required_event not in seen_events:
            raise _slot_error("required_event_missing", slot_index, required_event)


def _ordering_requirements(constraints: dict[str, Any]) -> list[dict[str, Any]]:
    configured = constraints.get("microprogram", {}).get("required_event_before", [])
    if configured:
        return [dict(item) for item in configured]
    return [{"op": "WRITE_REG", "device": "clock_mux", "address": "0x04", "event": "pll0_locked"}]


def _masked_register_value_from_slot(
    state: dict[str, Any], constraints: dict[str, Any], device: str, address: int, value: int, mask: int
) -> int:
    from .direct_action import _masked_register_value

    return _masked_register_value(state, constraints, device, address, value, mask)


def _commit_recovery(
    proposed: MicroprogramPacket,
    verification: MicroprogramVerificationResult,
    before: dict[str, Any],
    current: dict[str, Any],
    constraints: dict[str, Any],
    slot_after_states: list[dict[str, Any]],
    reason: str | None,
) -> MicroprogramCommitResult:
    safe = _safe_action(reason or "invalid_params")
    after = simulate_action(current, safe, constraints)
    return MicroprogramCommitResult(
        False,
        reason,
        proposed,
        verification.normalized_program,
        before,
        after,
        verification,
        slot_after_states,
        safe,
    )


class _SlotVerificationError(VerificationError):
    def __init__(self, reason: str, slot_index: int, detail: str | None = None) -> None:
        self.slot_index = slot_index
        super().__init__(reason, detail)


def _slot_error(reason: str, slot_index: int, detail: str | None = None) -> _SlotVerificationError:
    return _SlotVerificationError(reason, slot_index, detail)
