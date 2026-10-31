from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


ACTION_SCHEMA_VERSION = "pan-hw-action-v1"

ACTION_OPS = {
    "NOOP",
    "SET_PWM",
    "MOVE_AXIS",
    "WRITE_REG",
    "DMA_COPY",
    "SET_NCO",
    "LOAD_FILTER",
    "SET_GAIN",
    "BUS_TX",
    "FAULT_HOLD",
}

SAFE_REASON_CODES = {
    "fault_active": 1,
    "schema_version_mismatch": 2,
    "unknown_op": 3,
    "invalid_params": 4,
    "range_violation": 5,
    "allowlist_violation": 6,
    "expected_after_mismatch": 7,
}


class VerificationError(ValueError):
    def __init__(self, reason: str, detail: str | None = None) -> None:
        self.reason = reason
        self.detail = detail
        message = reason if detail is None else f"{reason}: {detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ActionPacket:
    op: str
    params: dict[str, Any] = field(default_factory=dict)
    expected_after: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    schema_version: str = ACTION_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionPacket":
        return cls(
            schema_version=str(payload.get("schema_version", ACTION_SCHEMA_VERSION)),
            op=str(payload.get("op", "")),
            params=dict(payload.get("params", {})),
            expected_after=dict(payload.get("expected_after", {})),
            confidence=float(payload.get("confidence", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "op": self.op,
            "params": deepcopy(self.params),
            "expected_after": deepcopy(self.expected_after),
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    reason: str | None = None
    rejected_action: str | None = None
    safe_action: ActionPacket | None = None
    normalized_action: ActionPacket | None = None
    expected_after_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": self.ok,
            "reason": self.reason,
            "rejected_action": self.rejected_action,
        }
        if self.safe_action is not None:
            payload["safe_action"] = self.safe_action.to_dict()
        if self.normalized_action is not None:
            payload["normalized_action"] = self.normalized_action.to_dict()
        if self.expected_after_state:
            payload["expected_after_state"] = deepcopy(self.expected_after_state)
        return payload


@dataclass(frozen=True)
class CommitResult:
    ok: bool
    reason: str | None
    proposed_action: ActionPacket
    committed_action: ActionPacket
    before_state: dict[str, Any]
    after_state: dict[str, Any]
    verification: VerificationResult

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "proposed_action": self.proposed_action.to_dict(),
            "committed_action": self.committed_action.to_dict(),
            "before_state": deepcopy(self.before_state),
            "after_state": deepcopy(self.after_state),
            "verification": self.verification.to_dict(),
        }


def direct_action_oracle(
    state: dict[str, Any], goal: dict[str, Any], constraints: dict[str, Any] | None = None
) -> ActionPacket:
    constraints = constraints or {}
    if _fault_active(state, constraints):
        return _safe_action("fault_active", "input fault flag is active")

    kind = str(goal.get("kind", "")).lower()
    if kind == "set_pwm":
        channel = goal["channel"]
        duty_q16 = _as_int(goal["duty_q16"], "duty_q16")
        return ActionPacket(
            op="SET_PWM",
            params={"channel": channel, "duty_q16": duty_q16},
            expected_after={"pwm": {str(channel): {"duty_q16": duty_q16}}},
        )
    if kind == "move_axis":
        axis = str(goal["axis"])
        current = _axis_position(state, axis)
        if "delta_ticks" in goal:
            delta = _as_int(goal["delta_ticks"], "delta_ticks")
            target = current + delta
        else:
            target = _as_int(goal["target_position_ticks"], "target_position_ticks")
            delta = target - current
        axis_spec = _axis_spec(constraints, axis)
        velocity = _as_int(
            goal.get("velocity_ticks_per_s", axis_spec.get("default_velocity_ticks_per_s", 1)),
            "velocity_ticks_per_s",
        )
        accel = _as_int(
            goal.get("accel_ticks_per_s2", axis_spec.get("default_accel_ticks_per_s2", 1)),
            "accel_ticks_per_s2",
        )
        return ActionPacket(
            op="MOVE_AXIS",
            params={
                "axis": axis,
                "delta_ticks": delta,
                "velocity_ticks_per_s": velocity,
                "accel_ticks_per_s2": accel,
            },
            expected_after={
                "axes": {
                    axis: {
                        "target_position_ticks": target,
                        "trajectory_state": "armed",
                        "remaining_ticks": delta,
                    }
                }
            },
        )
    if kind == "write_reg":
        device = str(goal["device"])
        address = _as_int(goal["address"], "address")
        value = _as_int(goal["value"], "value")
        mask = _as_int(goal["mask"], "mask")
        return ActionPacket(
            op="WRITE_REG",
            params={"device": device, "address": _hex_key(address), "value": _hex_key(value), "mask": _hex_key(mask)},
            expected_after={
                "registers": {
                    device: {
                        _hex_key(address): _masked_register_value(state, constraints, device, address, value, mask)
                    }
                }
            },
        )
    if kind == "dma_copy":
        return ActionPacket(
            op="DMA_COPY",
            params={
                "src": str(goal["src"]),
                "dst": str(goal["dst"]),
                "length": _as_int(goal["length"], "length"),
                "flags": _as_int(goal.get("flags", 0), "flags"),
            },
            expected_after={"dma": {"last_copy": {"src": str(goal["src"]), "dst": str(goal["dst"])}}},
        )
    if kind == "set_nco":
        return ActionPacket(
            op="SET_NCO",
            params={
                "channel": str(goal["channel"]),
                "frequency_word": _as_int(goal["frequency_word"], "frequency_word"),
                "phase_word": _as_int(goal.get("phase_word", 0), "phase_word"),
            },
            expected_after={"nco": {str(goal["channel"]): {"frequency_word": _as_int(goal["frequency_word"], "frequency_word")}}},
        )
    if kind == "load_filter":
        bank = str(goal["bank"])
        coefficient_hash = str(goal["coefficient_hash"])
        return ActionPacket(
            op="LOAD_FILTER",
            params={"bank": bank, "coefficient_hash": coefficient_hash},
            expected_after={"filters": {bank: {"coefficient_hash": coefficient_hash}}},
        )
    if kind == "set_gain":
        channel = str(goal["channel"])
        gain_q8 = _as_int(goal["gain_q8"], "gain_q8")
        return ActionPacket(
            op="SET_GAIN",
            params={"channel": channel, "gain_q8": gain_q8},
            expected_after={"gain": {channel: {"gain_q8": gain_q8}}},
        )
    if kind == "bus_tx":
        bus = str(goal["bus"])
        device = str(goal["device"])
        payload_id = str(goal["payload_id"])
        return ActionPacket(
            op="BUS_TX",
            params={"bus": bus, "device": device, "payload_id": payload_id},
            expected_after={"bus": {"last_tx": {"bus": bus, "device": device, "payload_id": payload_id}}},
        )
    if kind == "fault_hold":
        return _safe_action("fault_active", str(goal.get("reason", "requested fault hold")))

    raise ValueError(f"unsupported direct action goal kind: {kind!r}")


def verify_action(
    state: dict[str, Any], constraints: dict[str, Any] | None, action: ActionPacket | dict[str, Any]
) -> VerificationResult:
    constraints = constraints or {}
    packet = ActionPacket.from_dict(action) if isinstance(action, dict) else action
    try:
        normalized = _normalize_action(packet, constraints)
        if _fault_active(state, constraints) and normalized.op != "FAULT_HOLD":
            raise VerificationError("fault_active", "fault flag forces FAULT_HOLD")
        if normalized.op not in ACTION_OPS:
            raise VerificationError("unknown_op", normalized.op)
        if normalized.op == "NOOP":
            _verify_noop(normalized)
        elif normalized.op == "FAULT_HOLD":
            _verify_fault_hold(normalized)
        elif normalized.op == "SET_PWM":
            _verify_set_pwm(constraints, normalized)
        elif normalized.op == "MOVE_AXIS":
            _verify_move_axis(state, constraints, normalized)
        elif normalized.op == "WRITE_REG":
            _verify_write_reg(constraints, normalized)
        elif normalized.op == "DMA_COPY":
            _verify_dma_copy(constraints, normalized)
        elif normalized.op == "SET_NCO":
            _verify_set_nco(constraints, normalized)
        elif normalized.op == "LOAD_FILTER":
            _verify_load_filter(constraints, normalized)
        elif normalized.op == "SET_GAIN":
            _verify_set_gain(constraints, normalized)
        elif normalized.op == "BUS_TX":
            _verify_bus_tx(constraints, normalized)

        after_state = simulate_action(state, normalized, constraints)
        if normalized.expected_after and not _is_subset(normalized.expected_after, after_state):
            raise VerificationError("expected_after_mismatch", "declared expected_after does not match simulator")
        return VerificationResult(ok=True, normalized_action=normalized, expected_after_state=after_state)
    except VerificationError as exc:
        return VerificationResult(
            ok=False,
            reason=exc.reason,
            rejected_action=packet.op,
            safe_action=_safe_action(exc.reason, exc.detail),
        )


def commit_action(
    state: dict[str, Any],
    constraints: dict[str, Any] | None,
    action: ActionPacket | dict[str, Any],
    *,
    commit_safe_action: bool = True,
) -> CommitResult:
    packet = ActionPacket.from_dict(action) if isinstance(action, dict) else action
    before = deepcopy(state)
    verification = verify_action(before, constraints, packet)
    if verification.ok and verification.normalized_action is not None:
        committed = verification.normalized_action
        after = simulate_action(before, committed, constraints or {})
        return CommitResult(True, None, packet, committed, before, after, verification)

    safe = verification.safe_action or _safe_action(verification.reason or "invalid_params")
    if commit_safe_action:
        after = simulate_action(before, safe, constraints or {})
        return CommitResult(False, verification.reason, packet, safe, before, after, verification)
    return CommitResult(False, verification.reason, packet, packet, before, before, verification)


def simulate_action(
    state: dict[str, Any], action: ActionPacket | dict[str, Any], constraints: dict[str, Any] | None = None
) -> dict[str, Any]:
    packet = ActionPacket.from_dict(action) if isinstance(action, dict) else action
    constraints = constraints or {}
    after = deepcopy(state)

    if packet.op == "NOOP":
        return after
    if packet.op == "FAULT_HOLD":
        after["safe_state"] = "fault_hold"
        after["fault_hold"] = {"reason_code": _as_int(packet.params.get("reason_code", 0), "reason_code")}
        return after
    if packet.op == "SET_PWM":
        channel = str(packet.params["channel"])
        after.setdefault("pwm", {}).setdefault(channel, {})["duty_q16"] = _as_int(packet.params["duty_q16"], "duty_q16")
        return after
    if packet.op == "MOVE_AXIS":
        axis = str(packet.params["axis"])
        delta = _as_int(packet.params["delta_ticks"], "delta_ticks")
        current = _axis_position(after, axis)
        axis_state = after.setdefault("axes", {}).setdefault(axis, {})
        axis_state["target_position_ticks"] = current + delta
        axis_state["trajectory_state"] = "armed"
        axis_state["remaining_ticks"] = delta
        axis_state["velocity_ticks_per_s"] = _as_int(packet.params["velocity_ticks_per_s"], "velocity_ticks_per_s")
        axis_state["accel_ticks_per_s2"] = _as_int(packet.params["accel_ticks_per_s2"], "accel_ticks_per_s2")
        return after
    if packet.op == "WRITE_REG":
        device = str(packet.params["device"])
        address = _as_int(packet.params["address"], "address")
        value = _as_int(packet.params["value"], "value")
        mask = _as_int(packet.params["mask"], "mask")
        new_value = _masked_register_value(after, constraints, device, address, value, mask)
        after.setdefault("registers", {}).setdefault(device, {})[_hex_key(address)] = new_value
        _apply_register_state_alias(after, constraints, device, address, new_value)
        return after
    if packet.op == "DMA_COPY":
        src = str(packet.params["src"])
        dst = str(packet.params["dst"])
        length = _as_int(packet.params["length"], "length")
        flags = _as_int(packet.params.get("flags", 0), "flags")
        after.setdefault("dma", {})["last_copy"] = {"src": src, "dst": dst, "length": length, "flags": flags}
        buffers = after.setdefault("dma_buffers", {})
        buffers.setdefault(dst, {})["last_write_from"] = src
        buffers.setdefault(dst, {})["last_write_length"] = length
        return after
    if packet.op == "SET_NCO":
        channel = str(packet.params["channel"])
        after.setdefault("nco", {}).setdefault(channel, {})["frequency_word"] = _as_int(
            packet.params["frequency_word"], "frequency_word"
        )
        after["nco"][channel]["phase_word"] = _as_int(packet.params.get("phase_word", 0), "phase_word")
        return after
    if packet.op == "LOAD_FILTER":
        bank = str(packet.params["bank"])
        after.setdefault("filters", {}).setdefault(bank, {})["coefficient_hash"] = str(packet.params["coefficient_hash"])
        return after
    if packet.op == "SET_GAIN":
        channel = str(packet.params["channel"])
        after.setdefault("gain", {}).setdefault(channel, {})["gain_q8"] = _as_int(packet.params["gain_q8"], "gain_q8")
        return after
    if packet.op == "BUS_TX":
        after.setdefault("bus", {})["last_tx"] = {
            "bus": str(packet.params["bus"]),
            "device": str(packet.params["device"]),
            "payload_id": str(packet.params["payload_id"]),
        }
        return after
    raise VerificationError("unknown_op", packet.op)


def _normalize_action(packet: ActionPacket, constraints: dict[str, Any]) -> ActionPacket:
    if packet.schema_version != ACTION_SCHEMA_VERSION:
        raise VerificationError("schema_version_mismatch", packet.schema_version)
    if packet.op not in ACTION_OPS:
        raise VerificationError("unknown_op", packet.op)

    params = dict(packet.params)
    if packet.op == "NOOP":
        params = {}
    elif packet.op == "FAULT_HOLD":
        params = {"reason_code": _as_int(params.get("reason_code", 0), "reason_code")}
    elif packet.op == "SET_PWM":
        params = {"channel": str(params["channel"]), "duty_q16": _as_int(params["duty_q16"], "duty_q16")}
    elif packet.op == "MOVE_AXIS":
        velocity = params.get("velocity_ticks_per_s", params.get("velocity"))
        accel = params.get("accel_ticks_per_s2", params.get("accel"))
        params = {
            "axis": str(params["axis"]),
            "delta_ticks": _as_int(params["delta_ticks"], "delta_ticks"),
            "velocity_ticks_per_s": _as_int(velocity, "velocity_ticks_per_s"),
            "accel_ticks_per_s2": _as_int(accel, "accel_ticks_per_s2"),
        }
    elif packet.op == "WRITE_REG":
        address = _as_int(params["address"], "address")
        params = {
            "device": str(params["device"]),
            "address": _hex_key(address),
            "value": _hex_key(_as_int(params["value"], "value")),
            "mask": _hex_key(_as_int(params["mask"], "mask")),
        }
    elif packet.op == "DMA_COPY":
        params = {
            "src": str(params["src"]),
            "dst": str(params["dst"]),
            "length": _as_int(params["length"], "length"),
            "flags": _as_int(params.get("flags", 0), "flags"),
        }
    elif packet.op == "SET_NCO":
        params = {
            "channel": str(params["channel"]),
            "frequency_word": _as_int(params["frequency_word"], "frequency_word"),
            "phase_word": _as_int(params.get("phase_word", 0), "phase_word"),
        }
    elif packet.op == "LOAD_FILTER":
        params = {"bank": str(params["bank"]), "coefficient_hash": str(params["coefficient_hash"])}
    elif packet.op == "SET_GAIN":
        params = {"channel": str(params["channel"]), "gain_q8": _as_int(params["gain_q8"], "gain_q8")}
    elif packet.op == "BUS_TX":
        params = {"bus": str(params["bus"]), "device": str(params["device"]), "payload_id": str(params["payload_id"])}

    return ActionPacket(
        op=packet.op,
        params=params,
        expected_after=deepcopy(packet.expected_after),
        confidence=packet.confidence,
        schema_version=packet.schema_version,
    )


def _verify_noop(packet: ActionPacket) -> None:
    if packet.params:
        raise VerificationError("invalid_params", "NOOP takes no params")


def _verify_fault_hold(packet: ActionPacket) -> None:
    reason_code = _as_int(packet.params.get("reason_code", 0), "reason_code")
    if reason_code < 0 or reason_code > 0xFFFF:
        raise VerificationError("range_violation", "reason_code")


def _verify_set_pwm(constraints: dict[str, Any], packet: ActionPacket) -> None:
    channel = str(packet.params["channel"])
    duty = _as_int(packet.params["duty_q16"], "duty_q16")
    spec = _pwm_spec(constraints, channel)
    min_duty = _as_int(spec.get("min_q16", 0), "min_q16")
    max_duty = _as_int(spec.get("max_q16", 65535), "max_q16")
    if duty < min_duty or duty > max_duty:
        raise VerificationError("range_violation", "duty_q16")


def _verify_move_axis(state: dict[str, Any], constraints: dict[str, Any], packet: ActionPacket) -> None:
    axis = str(packet.params["axis"])
    spec = _axis_spec(constraints, axis)
    delta = _as_int(packet.params["delta_ticks"], "delta_ticks")
    velocity = abs(_as_int(packet.params["velocity_ticks_per_s"], "velocity_ticks_per_s"))
    accel = abs(_as_int(packet.params["accel_ticks_per_s2"], "accel_ticks_per_s2"))
    current = _axis_position(state, axis)
    target = current + delta
    if target < _as_int(spec.get("min_position_ticks", -2**63), "min_position_ticks"):
        raise VerificationError("range_violation", "axis_min_position")
    if target > _as_int(spec.get("max_position_ticks", 2**63 - 1), "max_position_ticks"):
        raise VerificationError("range_violation", "axis_max_position")
    if velocity > _as_int(spec.get("max_velocity_ticks_per_s", 0), "max_velocity_ticks_per_s"):
        raise VerificationError("range_violation", "velocity_ticks_per_s")
    if accel > _as_int(spec.get("max_accel_ticks_per_s2", 0), "max_accel_ticks_per_s2"):
        raise VerificationError("range_violation", "accel_ticks_per_s2")
    if "max_delta_ticks" in spec and abs(delta) > _as_int(spec["max_delta_ticks"], "max_delta_ticks"):
        raise VerificationError("range_violation", "delta_ticks")


def _verify_write_reg(constraints: dict[str, Any], packet: ActionPacket) -> None:
    device = str(packet.params["device"])
    address = _as_int(packet.params["address"], "address")
    value = _as_int(packet.params["value"], "value")
    mask = _as_int(packet.params["mask"], "mask")
    spec = _register_address_spec(constraints, device, address)
    allowed_mask = _as_int(spec.get("mask", spec.get("allowed_mask", 0)), "allowed_mask")
    value_max = _as_int(spec.get("value_max", 0xFFFFFFFF), "value_max")
    if mask <= 0:
        raise VerificationError("invalid_params", "mask must be nonzero")
    if mask & ~allowed_mask:
        raise VerificationError("allowlist_violation", "mask")
    if value < 0 or value > value_max:
        raise VerificationError("range_violation", "value")


def _verify_dma_copy(constraints: dict[str, Any], packet: ActionPacket) -> None:
    dma = constraints.get("dma", {})
    buffers = dma.get("buffers", {})
    src = str(packet.params["src"])
    dst = str(packet.params["dst"])
    length = _as_int(packet.params["length"], "length")
    flags = _as_int(packet.params.get("flags", 0), "flags")
    if src not in buffers or dst not in buffers:
        raise VerificationError("allowlist_violation", "dma buffer")
    if "r" not in str(buffers[src].get("permissions", "")):
        raise VerificationError("allowlist_violation", "src permissions")
    if "w" not in str(buffers[dst].get("permissions", "")):
        raise VerificationError("allowlist_violation", "dst permissions")
    if length <= 0:
        raise VerificationError("range_violation", "length")
    max_length = _as_int(dma.get("max_length", min(_as_int(buffers[src]["length"], "src length"), _as_int(buffers[dst]["length"], "dst length"))), "max_length")
    if length > max_length or length > _as_int(buffers[src]["length"], "src length") or length > _as_int(buffers[dst]["length"], "dst length"):
        raise VerificationError("range_violation", "length")
    alignment = _as_int(dma.get("alignment", 1), "alignment")
    if alignment > 1 and length % alignment != 0:
        raise VerificationError("range_violation", "alignment")
    allowed_flags = _as_int(dma.get("allowed_flags", 0), "allowed_flags")
    if flags & ~allowed_flags:
        raise VerificationError("allowlist_violation", "flags")


def _verify_set_nco(constraints: dict[str, Any], packet: ActionPacket) -> None:
    channel = str(packet.params["channel"])
    spec = _nco_spec(constraints, channel)
    frequency_word = _as_int(packet.params["frequency_word"], "frequency_word")
    phase_word = _as_int(packet.params.get("phase_word", 0), "phase_word")
    if frequency_word < _as_int(spec.get("min_frequency_word", 0), "min_frequency_word"):
        raise VerificationError("range_violation", "frequency_word")
    if frequency_word > _as_int(spec.get("max_frequency_word", 2**64 - 1), "max_frequency_word"):
        raise VerificationError("range_violation", "frequency_word")
    if phase_word < 0 or phase_word > _as_int(spec.get("max_phase_word", 2**64 - 1), "max_phase_word"):
        raise VerificationError("range_violation", "phase_word")


def _verify_load_filter(constraints: dict[str, Any], packet: ActionPacket) -> None:
    bank = str(packet.params["bank"])
    spec = _filter_spec(constraints, bank)
    coefficient_hash = str(packet.params["coefficient_hash"])
    allowed = {str(item) for item in spec.get("allowed_hashes", [])}
    if allowed and coefficient_hash not in allowed:
        raise VerificationError("allowlist_violation", "coefficient_hash")


def _verify_set_gain(constraints: dict[str, Any], packet: ActionPacket) -> None:
    channel = str(packet.params["channel"])
    spec = _gain_spec(constraints, channel)
    gain = _as_int(packet.params["gain_q8"], "gain_q8")
    if gain < _as_int(spec.get("min_q8", 0), "min_q8"):
        raise VerificationError("range_violation", "gain_q8")
    if gain > _as_int(spec.get("max_q8", 0), "max_q8"):
        raise VerificationError("range_violation", "gain_q8")


def _verify_bus_tx(constraints: dict[str, Any], packet: ActionPacket) -> None:
    bus = str(packet.params["bus"])
    device = str(packet.params["device"])
    payload_id = str(packet.params["payload_id"])
    spec = _bus_spec(constraints, bus)
    allowed_devices = {str(item) for item in spec.get("devices", [])}
    allowed_payloads = {str(item) for item in spec.get("payloads", [])}
    if device not in allowed_devices:
        raise VerificationError("allowlist_violation", "bus device")
    if payload_id not in allowed_payloads:
        raise VerificationError("allowlist_violation", "payload_id")


def _safe_action(reason: str, detail: str | None = None) -> ActionPacket:
    return ActionPacket(
        op="FAULT_HOLD",
        params={"reason_code": SAFE_REASON_CODES.get(reason, SAFE_REASON_CODES["invalid_params"])},
        expected_after={"safe_state": "fault_hold"},
        confidence=1.0,
    )


def _fault_active(state: dict[str, Any], constraints: dict[str, Any]) -> bool:
    if constraints.get("force_fault_hold"):
        return True
    limit_flags = state.get("limit_flags", {})
    if isinstance(limit_flags, dict):
        for key in ("e_stop", "hard_limit", "thermal_fault", "overcurrent", "fault"):
            if bool(limit_flags.get(key, False)):
                return True
    faults = state.get("faults", {})
    if isinstance(faults, dict):
        return any(bool(value) for value in faults.values())
    if isinstance(faults, list):
        return bool(faults)
    return False


def _axis_position(state: dict[str, Any], axis: str) -> int:
    axes = state.get("axes", {})
    if axis in axes and isinstance(axes[axis], dict):
        return _as_int(axes[axis].get("position_ticks", 0), "position_ticks")
    positions = state.get("axis_positions", {})
    if axis in positions:
        return _as_int(positions[axis], "position_ticks")
    return 0


def _masked_register_value(
    state: dict[str, Any],
    constraints: dict[str, Any],
    device: str,
    address: int,
    value: int,
    mask: int,
) -> int:
    registers = state.get("registers", {})
    current = 0
    if device in registers and isinstance(registers[device], dict):
        current = _as_int(registers[device].get(_hex_key(address), registers[device].get(str(address), 0)), "register")
    spec = _maybe_register_address_spec(constraints, device, address)
    if spec is not None and "reset" in spec and current == 0:
        current = _as_int(spec["reset"], "register reset")
    return (current & ~mask) | (value & mask)


def _apply_register_state_alias(
    state: dict[str, Any], constraints: dict[str, Any], device: str, address: int, new_value: int
) -> None:
    spec = _maybe_register_address_spec(constraints, device, address)
    if not spec:
        return
    alias = spec.get("state_alias")
    if not isinstance(alias, dict):
        return
    field = alias.get("field")
    if not field:
        return
    alias_mask = _as_int(alias.get("mask", spec.get("mask", 0xFFFFFFFF)), "state alias mask")
    selected = new_value & alias_mask
    values = alias.get("values", {})
    mapped = _lookup_keyed(values, selected)
    if mapped is None:
        mapped = selected
    state[str(field)] = mapped


def _pwm_spec(constraints: dict[str, Any], channel: str) -> dict[str, Any]:
    channels = constraints.get("pwm_channels", constraints.get("pwm", {}).get("channels", {}))
    if channel not in channels:
        raise VerificationError("allowlist_violation", f"pwm channel {channel!r}")
    return dict(channels[channel])


def _axis_spec(constraints: dict[str, Any], axis: str) -> dict[str, Any]:
    axes = constraints.get("axes", constraints.get("motion", {}).get("axes", {}))
    if axis not in axes:
        raise VerificationError("allowlist_violation", f"axis {axis!r}")
    return dict(axes[axis])


def _register_address_spec(constraints: dict[str, Any], device: str, address: int) -> dict[str, Any]:
    spec = _maybe_register_address_spec(constraints, device, address)
    if spec is None:
        raise VerificationError("allowlist_violation", f"register {device}:{_hex_key(address)}")
    return spec


def _maybe_register_address_spec(constraints: dict[str, Any], device: str, address: int) -> dict[str, Any] | None:
    devices = constraints.get("register_devices", constraints.get("registers", {}).get("devices", {}))
    if device not in devices:
        return None
    device_spec = devices[device]
    addresses = device_spec.get("addresses", device_spec)
    spec = _lookup_keyed(addresses, address)
    return dict(spec) if isinstance(spec, dict) else None


def _nco_spec(constraints: dict[str, Any], channel: str) -> dict[str, Any]:
    channels = constraints.get("nco_channels", constraints.get("nco", {}).get("channels", {}))
    if channel not in channels:
        raise VerificationError("allowlist_violation", f"nco channel {channel!r}")
    return dict(channels[channel])


def _filter_spec(constraints: dict[str, Any], bank: str) -> dict[str, Any]:
    banks = constraints.get("filter_banks", constraints.get("filters", {}).get("banks", {}))
    if bank not in banks:
        raise VerificationError("allowlist_violation", f"filter bank {bank!r}")
    return dict(banks[bank])


def _gain_spec(constraints: dict[str, Any], channel: str) -> dict[str, Any]:
    channels = constraints.get("gain_channels", constraints.get("gain", {}).get("channels", {}))
    if channel not in channels:
        raise VerificationError("allowlist_violation", f"gain channel {channel!r}")
    return dict(channels[channel])


def _bus_spec(constraints: dict[str, Any], bus: str) -> dict[str, Any]:
    buses = constraints.get("buses", constraints.get("bus", {}).get("buses", {}))
    if bus not in buses:
        raise VerificationError("allowlist_violation", f"bus {bus!r}")
    return dict(buses[bus])


def _lookup_keyed(mapping: dict[Any, Any], key: int) -> Any:
    for candidate in (key, str(key), _hex_key(key), _hex_key(key).upper()):
        if candidate in mapping:
            return mapping[candidate]
    return None


def _as_int(value: Any, field_name: str) -> int:
    if value is None:
        raise VerificationError("invalid_params", field_name)
    if isinstance(value, bool):
        raise VerificationError("invalid_params", field_name)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError as exc:
            raise VerificationError("invalid_params", field_name) from exc
    raise VerificationError("invalid_params", field_name)


def _hex_key(value: int) -> str:
    return hex(value)


def _is_subset(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        return all(key in actual and _is_subset(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return expected == actual
    return expected == actual
