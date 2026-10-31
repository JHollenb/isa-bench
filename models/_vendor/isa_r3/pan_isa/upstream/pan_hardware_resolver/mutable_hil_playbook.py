from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from .ledger import stable_hash


PLAYBOOK_SCHEMA_VERSION = "pan-hw-mutable-hil-playbook-v1"
VALIDATION_SCHEMA_VERSION = "pan-hw-mutable-hil-playbook-validation-v1"

ALLOWED_ACTION_OPS = {
    "WRITE_REG",
    "SET_PWM",
    "MOVE_AXIS",
    "DMA_COPY",
    "SET_NCO",
    "LOAD_FILTER",
    "SET_GAIN",
    "BUS_TX",
}
ALLOWED_ACCESS_PATHS = {"synthetic", "userspace-device", "kernel-driver", "xrt-debug"}
RESTORE_METHODS = {
    "write_original_then_readback",
    "write_restore_value_then_readback",
    "driver_reset_then_readback",
    "reload_app_then_readback",
}
FAULT_RESPONSES = {
    "FAULT_HOLD",
    "restore_then_fault_hold",
    "fault_hold_and_stop",
}
REQUIRED_NEGATIVE_CONTROLS = {"invalid_mask", "readback_mismatch", "restore_mismatch", "fault_active"}
REQUIRED_EVIDENCE_FLAGS = {
    "requires_board_identity",
    "requires_before_after_hashes",
    "requires_ledger_replay",
    "requires_command_log",
    "requires_restore_log",
}


def synthetic_mutable_register_playbook() -> dict[str, Any]:
    return {
        "schema_version": PLAYBOOK_SCHEMA_VERSION,
        "target": {
            "id": "synthetic-debug-register",
            "description": "No-write template for validating mutable HIL readiness gates.",
            "board_model_contains": "*",
            "access_path": "synthetic",
            "mutable": True,
        },
        "arming": {
            "dry_run_default": True,
            "requires_human_arm": True,
            "max_write_count": 1,
        },
        "allowlist": {
            "operations": ["WRITE_REG"],
            "registers": [
                {
                    "device": "debug0",
                    "address": "0x10",
                    "mask": "0xff",
                    "readback_mask": "0xff",
                    "reset": "0x0",
                    "safe_values": ["0x0", "0x1"],
                }
            ],
        },
        "readback": {
            "required": True,
            "before_write": True,
            "after_write": True,
            "verify_expected_after": True,
            "max_attempts": 3,
            "settle_ms": 0,
        },
        "restore": {
            "required": True,
            "method": "write_original_then_readback",
            "verify_readback": True,
            "max_attempts": 3,
            "failure_action": "FAULT_HOLD",
        },
        "health_checks": {
            "pre": ["platform_identity", "target_idle"],
            "post": ["platform_identity", "target_idle", "restore_verified"],
        },
        "fault_response": {
            "on_verifier_reject": "FAULT_HOLD",
            "on_readback_mismatch": "restore_then_fault_hold",
            "on_restore_failure": "fault_hold_and_stop",
        },
        "negative_controls": [
            {"kind": "invalid_mask"},
            {"kind": "readback_mismatch"},
            {"kind": "restore_mismatch"},
            {"kind": "fault_active"},
        ],
        "evidence": {
            "requires_board_identity": True,
            "requires_before_after_hashes": True,
            "requires_ledger_replay": True,
            "requires_command_log": True,
            "requires_restore_log": True,
        },
    }


def kv260_sysfs_led_playbook(
    led_name: str = "vbus_det",
    *,
    brightness_path: str | None = None,
    max_brightness_path: str | None = None,
    trigger_path: str | None = None,
) -> dict[str, Any]:
    led = str(led_name)
    base = f"/sys/class/leds/{led}"
    return {
        "schema_version": PLAYBOOK_SCHEMA_VERSION,
        "target": {
            "id": f"kv260-sysfs-led-{led}",
            "description": "Benign KV260 sysfs LED brightness proof with readback and restore.",
            "board_model_contains": "KV260",
            "access_path": "userspace-device",
            "mutable": True,
            "raw_mmio": False,
            "userspace_device": {
                "kind": "sysfs-led",
                "led": led,
                "brightness_path": brightness_path or f"{base}/brightness",
                "max_brightness_path": max_brightness_path or f"{base}/max_brightness",
                "trigger_path": trigger_path or f"{base}/trigger",
                "required_active_trigger": "none",
            },
        },
        "arming": {
            "dry_run_default": True,
            "requires_human_arm": True,
            "max_write_count": 2,
        },
        "allowlist": {
            "operations": ["WRITE_REG"],
            "registers": [
                {
                    "device": f"sysfs-led:{led}",
                    "address": "0x0",
                    "mask": "0x1",
                    "readback_mask": "0x1",
                    "restore_from": "original",
                    "safe_values": ["0x0", "0x1"],
                }
            ],
        },
        "readback": {
            "required": True,
            "before_write": True,
            "after_write": True,
            "verify_expected_after": True,
            "max_attempts": 3,
            "settle_ms": 10,
        },
        "restore": {
            "required": True,
            "method": "write_original_then_readback",
            "verify_readback": True,
            "max_attempts": 3,
            "failure_action": "FAULT_HOLD",
        },
        "health_checks": {
            "pre": ["platform_identity", "target_idle", "sysfs_led_trigger_none"],
            "post": ["platform_identity", "target_idle", "restore_verified", "sysfs_led_trigger_unchanged"],
        },
        "fault_response": {
            "on_verifier_reject": "FAULT_HOLD",
            "on_readback_mismatch": "restore_then_fault_hold",
            "on_restore_failure": "fault_hold_and_stop",
        },
        "negative_controls": [
            {"kind": "invalid_mask"},
            {"kind": "readback_mismatch"},
            {"kind": "restore_mismatch"},
            {"kind": "fault_active"},
        ],
        "evidence": {
            "requires_board_identity": True,
            "requires_before_after_hashes": True,
            "requires_ledger_replay": True,
            "requires_command_log": True,
            "requires_restore_log": True,
        },
    }


def load_playbook(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(payload: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def validate_playbook(
    playbook: dict[str, Any],
    *,
    expected_model: str | None = None,
    platform_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    platform_payload = platform_info or _platform_info()
    problems: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    _validate_schema_version(playbook, problems)
    target = _expect_dict(playbook, "target", problems)
    target_id = str(target.get("id", "")).strip()
    if not target_id:
        _problem(problems, "target.id", "missing target id")
    access_path = str(target.get("access_path", "")).strip()
    if access_path not in ALLOWED_ACCESS_PATHS:
        _problem(problems, "target.access_path", "unsupported or unsafe access path")
    if bool(target.get("raw_mmio", False)) or access_path == "raw-mmio":
        _problem(problems, "target.access_path", "raw MMIO is not allowed by this preflight")
    if target.get("mutable") is not True:
        _problem(problems, "target.mutable", "mutable targets must opt in explicitly")
    _validate_model(target, expected_model, platform_payload, problems)

    _validate_arming(_expect_dict(playbook, "arming", problems), problems)
    _validate_allowlist(_expect_dict(playbook, "allowlist", problems), problems)
    _validate_readback(_expect_dict(playbook, "readback", problems), problems)
    _validate_restore(_expect_dict(playbook, "restore", problems), problems)
    _validate_health(_expect_dict(playbook, "health_checks", problems), problems)
    _validate_fault_response(_expect_dict(playbook, "fault_response", problems), problems)
    _validate_negative_controls(playbook.get("negative_controls"), problems)
    _validate_evidence(_expect_dict(playbook, "evidence", problems), problems)

    if access_path in {"kernel-driver", "xrt-debug"}:
        warnings.append(
            {
                "path": "target.access_path",
                "reason": "preflight validates the playbook only; a separate armed runner must prove exclusive access",
            }
        )
    if access_path == "userspace-device":
        _validate_userspace_device(target, problems)

    ok = not problems
    return {
        "schema": VALIDATION_SCHEMA_VERSION,
        "ok": ok,
        "playbook_hash": stable_hash(playbook),
        "playbook_schema": str(playbook.get("schema_version", "")),
        "target_id": target_id,
        "target_access_path": access_path,
        "platform": platform_payload,
        "expected_model": expected_model,
        "mutable_hil_ready": ok,
        "execution_authority": "preflight_only",
        "mutable_execution_authorized": False,
        "problems": problems,
        "warnings": warnings,
        "requirements": {
            "allowlist": "every writable surface must be listed with mask, readback mask, reset/restore basis, and safe values",
            "readback": "before and after readback must be mandatory and must verify expected_after",
            "restore": "restore must be mandatory, read back, bounded, and fail to FAULT_HOLD",
            "fault_response": "verifier, readback, and restore failures must land in a safe hold path",
            "evidence": "board identity, before/after hashes, replay, command log, and restore log are mandatory",
        },
    }


def _validate_schema_version(playbook: dict[str, Any], problems: list[dict[str, str]]) -> None:
    if playbook.get("schema_version") != PLAYBOOK_SCHEMA_VERSION:
        _problem(problems, "schema_version", f"expected {PLAYBOOK_SCHEMA_VERSION}")


def _validate_model(
    target: dict[str, Any],
    expected_model: str | None,
    platform_info: dict[str, Any],
    problems: list[dict[str, str]],
) -> None:
    model = str(platform_info.get("model", ""))
    if expected_model and expected_model.lower() not in model.lower():
        _problem(problems, "platform.model", f"expected model containing {expected_model!r}, got {model!r}")
    required = str(target.get("board_model_contains", "")).strip()
    if not required:
        _problem(problems, "target.board_model_contains", "missing board model guard")
    elif required != "*" and required.lower() not in model.lower():
        _problem(problems, "target.board_model_contains", f"target requires {required!r}, got {model!r}")


def _validate_arming(arming: dict[str, Any], problems: list[dict[str, str]]) -> None:
    if arming.get("dry_run_default") is not True:
        _problem(problems, "arming.dry_run_default", "mutable HIL must default to dry-run")
    if arming.get("requires_human_arm") is not True:
        _problem(problems, "arming.requires_human_arm", "mutable HIL must require explicit human arm")
    max_write_count = _as_int(arming.get("max_write_count"), "arming.max_write_count", problems)
    if max_write_count is not None and max_write_count <= 0:
        _problem(problems, "arming.max_write_count", "max_write_count must be positive")


def _validate_allowlist(allowlist: dict[str, Any], problems: list[dict[str, str]]) -> None:
    operations = allowlist.get("operations")
    if not isinstance(operations, list) or not operations:
        _problem(problems, "allowlist.operations", "must list at least one operation")
        operations = []
    for index, op in enumerate(operations):
        if str(op) not in ALLOWED_ACTION_OPS:
            _problem(problems, f"allowlist.operations[{index}]", "operation is not supported by the direct-action verifier")
    registers = allowlist.get("registers")
    if "WRITE_REG" in {str(op) for op in operations}:
        if not isinstance(registers, list) or not registers:
            _problem(problems, "allowlist.registers", "WRITE_REG requires at least one register allowlist entry")
            return
        for index, register in enumerate(registers):
            if not isinstance(register, dict):
                _problem(problems, f"allowlist.registers[{index}]", "register entry must be an object")
                continue
            _validate_register(register, f"allowlist.registers[{index}]", problems)


def _validate_userspace_device(target: dict[str, Any], problems: list[dict[str, str]]) -> None:
    device = target.get("userspace_device")
    if not isinstance(device, dict):
        _problem(problems, "target.userspace_device", "userspace-device targets must describe the device")
        return
    kind = str(device.get("kind", ""))
    if kind != "sysfs-led":
        _problem(problems, "target.userspace_device.kind", "only sysfs-led userspace devices are currently supported")
        return
    led = str(device.get("led", "")).strip()
    if not led or "/" in led:
        _problem(problems, "target.userspace_device.led", "LED name must be non-empty and must not contain path separators")
    for key in ("brightness_path", "max_brightness_path", "trigger_path"):
        value = str(device.get(key, "")).strip()
        if not value:
            _problem(problems, f"target.userspace_device.{key}", "missing sysfs path")
        elif not value.endswith("/" + key.replace("_path", "")):
            _problem(problems, f"target.userspace_device.{key}", "path must point to the expected sysfs LED file")
    if str(device.get("required_active_trigger", "")).strip() != "none":
        _problem(problems, "target.userspace_device.required_active_trigger", "real LED HIL requires active trigger none")


def _validate_register(register: dict[str, Any], path: str, problems: list[dict[str, str]]) -> None:
    for key in ("device", "address", "mask", "readback_mask"):
        if key not in register or str(register.get(key, "")).strip() == "":
            _problem(problems, f"{path}.{key}", "missing required register field")
    mask = _as_int(register.get("mask"), f"{path}.mask", problems)
    readback_mask = _as_int(register.get("readback_mask"), f"{path}.readback_mask", problems)
    if mask is not None and mask <= 0:
        _problem(problems, f"{path}.mask", "mask must be nonzero")
    if readback_mask is not None and readback_mask <= 0:
        _problem(problems, f"{path}.readback_mask", "readback_mask must be nonzero")
    if mask is not None and readback_mask is not None and mask & ~readback_mask:
        _problem(problems, f"{path}.readback_mask", "readback_mask must cover every writable bit")
    if "reset" not in register and "restore_value" not in register and register.get("restore_from") != "original":
        _problem(problems, path, "register must define reset, restore_value, or restore_from=original")
    safe_values = register.get("safe_values")
    if not isinstance(safe_values, list) or not safe_values:
        _problem(problems, f"{path}.safe_values", "must include at least one safe value")
    elif mask is not None:
        for value_index, value in enumerate(safe_values):
            parsed = _as_int(value, f"{path}.safe_values[{value_index}]", problems)
            if parsed is not None and parsed & ~mask:
                _problem(problems, f"{path}.safe_values[{value_index}]", "safe value sets bits outside mask")


def _validate_readback(readback: dict[str, Any], problems: list[dict[str, str]]) -> None:
    for key in ("required", "before_write", "after_write", "verify_expected_after"):
        if readback.get(key) is not True:
            _problem(problems, f"readback.{key}", "must be true")
    attempts = _as_int(readback.get("max_attempts"), "readback.max_attempts", problems)
    if attempts is not None and attempts <= 0:
        _problem(problems, "readback.max_attempts", "max_attempts must be positive")
    settle_ms = _as_int(readback.get("settle_ms", 0), "readback.settle_ms", problems)
    if settle_ms is not None and settle_ms < 0:
        _problem(problems, "readback.settle_ms", "settle_ms must be nonnegative")


def _validate_restore(restore: dict[str, Any], problems: list[dict[str, str]]) -> None:
    if restore.get("required") is not True:
        _problem(problems, "restore.required", "must be true")
    if str(restore.get("method", "")) not in RESTORE_METHODS:
        _problem(problems, "restore.method", "unsupported restore method")
    if restore.get("verify_readback") is not True:
        _problem(problems, "restore.verify_readback", "must be true")
    attempts = _as_int(restore.get("max_attempts"), "restore.max_attempts", problems)
    if attempts is not None and attempts <= 0:
        _problem(problems, "restore.max_attempts", "max_attempts must be positive")
    if str(restore.get("failure_action", "")) != "FAULT_HOLD":
        _problem(problems, "restore.failure_action", "restore failure must force FAULT_HOLD")


def _validate_health(health: dict[str, Any], problems: list[dict[str, str]]) -> None:
    pre = health.get("pre")
    post = health.get("post")
    if not isinstance(pre, list) or not pre:
        _problem(problems, "health_checks.pre", "must list pre-run health checks")
        pre = []
    if not isinstance(post, list) or not post:
        _problem(problems, "health_checks.post", "must list post-run health checks")
        post = []
    if "platform_identity" not in {str(item) for item in pre}:
        _problem(problems, "health_checks.pre", "must include platform_identity")
    if "restore_verified" not in {str(item) for item in post}:
        _problem(problems, "health_checks.post", "must include restore_verified")


def _validate_fault_response(fault_response: dict[str, Any], problems: list[dict[str, str]]) -> None:
    required = {
        "on_verifier_reject": "FAULT_HOLD",
        "on_readback_mismatch": "restore_then_fault_hold",
        "on_restore_failure": "fault_hold_and_stop",
    }
    for key, required_value in required.items():
        value = str(fault_response.get(key, ""))
        if value not in FAULT_RESPONSES:
            _problem(problems, f"fault_response.{key}", "unsupported fault response")
        elif value != required_value:
            _problem(problems, f"fault_response.{key}", f"expected {required_value}")


def _validate_negative_controls(payload: Any, problems: list[dict[str, str]]) -> None:
    if not isinstance(payload, list):
        _problem(problems, "negative_controls", "must be a list")
        return
    observed = {str(item.get("kind", "")) for item in payload if isinstance(item, dict)}
    missing = sorted(REQUIRED_NEGATIVE_CONTROLS - observed)
    if missing:
        _problem(problems, "negative_controls", "missing required controls: " + ", ".join(missing))


def _validate_evidence(evidence: dict[str, Any], problems: list[dict[str, str]]) -> None:
    for key in sorted(REQUIRED_EVIDENCE_FLAGS):
        if evidence.get(key) is not True:
            _problem(problems, f"evidence.{key}", "must be true")


def _expect_dict(payload: dict[str, Any], key: str, problems: list[dict[str, str]]) -> dict[str, Any]:
    value = payload.get(key)
    if isinstance(value, dict):
        return value
    _problem(problems, key, "must be an object")
    return {}


def _as_int(value: Any, path: str, problems: list[dict[str, str]]) -> int | None:
    try:
        if isinstance(value, str):
            return int(value, 0)
        return int(value)
    except (TypeError, ValueError):
        _problem(problems, path, "must be an integer")
        return None


def _problem(problems: list[dict[str, str]], path: str, reason: str) -> None:
    problems.append({"path": path, "reason": reason})


def _platform_info() -> dict[str, Any]:
    model = ""
    model_path = Path("/proc/device-tree/model")
    if model_path.exists():
        model = model_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "").strip()
    return {
        "model": model or platform.node(),
        "machine": platform.machine(),
        "system": platform.system(),
        "release": platform.release(),
        "python": platform.python_version(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a no-write mutable hardware HIL readiness playbook.")
    parser.add_argument("--playbook", help="Playbook JSON to validate.")
    parser.add_argument("--expect-model", help="Require the platform model string to contain this value.")
    parser.add_argument("--write", help="Write validation report JSON; stdout when omitted.")
    parser.add_argument("--write-template", help="Write a synthetic mutable-register playbook template before validation.")
    parser.add_argument("--write-kv260-sysfs-led-template", help="Write a KV260 sysfs LED playbook template before validation.")
    parser.add_argument("--led-name", default="vbus_det", help="LED name for --write-kv260-sysfs-led-template.")
    args = parser.parse_args(argv)

    if args.write_template:
        write_json(synthetic_mutable_register_playbook(), args.write_template)
    if args.write_kv260_sysfs_led_template:
        write_json(kv260_sysfs_led_playbook(args.led_name), args.write_kv260_sysfs_led_template)
    playbook_path = args.playbook or args.write_template or args.write_kv260_sysfs_led_template
    if not playbook_path:
        parser.error("--playbook, --write-template, or --write-kv260-sysfs-led-template is required")
    playbook = load_playbook(playbook_path)
    report = validate_playbook(playbook, expected_model=args.expect_model)
    write_json(report, args.write)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
