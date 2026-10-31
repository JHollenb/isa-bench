from __future__ import annotations

import argparse
import errno
import json
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Sequence

from .direct_action import ActionPacket, commit_action
from .ledger import direct_action_ledger_row, stable_hash, write_ledger_jsonl
from .mutable_hil_playbook import (
    PLAYBOOK_SCHEMA_VERSION,
    kv260_sysfs_led_playbook,
    load_playbook,
    synthetic_mutable_register_playbook,
    validate_playbook,
    write_json,
)
from .replay import replay_ledger_rows


RUNNER_SCHEMA_VERSION = "pan-hw-mutable-hil-runner-report-v1"
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def run_mutable_hil_playbook(
    playbook: dict[str, Any],
    *,
    arm_synthetic: bool = False,
    arm_real_sysfs_led: bool = False,
    allow_sudo: bool = False,
    allow_test_sysfs_paths: bool = False,
    command_runner: CommandRunner | None = None,
    expected_model: str | None = None,
    platform_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preflight = validate_playbook(playbook, expected_model=expected_model, platform_info=platform_info)
    base_report: dict[str, Any] = {
        "schema": RUNNER_SCHEMA_VERSION,
        "playbook_schema": str(playbook.get("schema_version", "")),
        "playbook_hash": stable_hash(playbook),
        "target_id": preflight.get("target_id", ""),
        "target_access_path": preflight.get("target_access_path", ""),
        "preflight": preflight,
        "unsafe_mmio": False,
        "real_device_execution_authorized": False,
        "synthetic_execution_authorized": False,
        "executed": False,
        "ledger_rows": 0,
        "restore": {"ok": False, "method": "not executed"},
        "health": {"ok": False, "checks": []},
        "command_log": [],
        "restore_log": [],
    }
    if not preflight["ok"]:
        return {**base_report, "ok": False, "reason": "preflight_failed", "problems": preflight["problems"]}
    if preflight.get("target_access_path") == "userspace-device":
        if arm_synthetic:
            return {
                **base_report,
                "ok": False,
                "reason": "unsupported_target_access_path",
                "problems": [{"path": "target.access_path", "reason": "synthetic arm cannot execute userspace-device targets"}],
            }
        if not arm_real_sysfs_led:
            return {
                **base_report,
                "ok": True,
                "reason": "dry_run",
                "health": {"ok": True, "checks": ["preflight"]},
            }
        run = _run_sysfs_led(
            playbook,
            allow_sudo=allow_sudo,
            allow_test_sysfs_paths=allow_test_sysfs_paths,
            command_runner=command_runner,
        )
        return {
            **base_report,
            "ok": bool(run["ok"]),
            "reason": run["reason"],
            "real_device_execution_authorized": True,
            "executed": True,
            "initial_state_hash": stable_hash(run["initial_state"]),
            "restored_state_hash": stable_hash(run["restored_state"]),
            "write_readback": run["write_readback"],
            "restore_readback": run["restore_readback"],
            "restore": run["restore"],
            "health": run["health"],
            "ledger_rows": len(run["ledger_rows"]),
            "ledger_replay": run["ledger_replay"],
            "ledger": run["ledger_rows"],
            "command_log": run["command_log"],
            "restore_log": run["restore_log"],
            "real_device": run["real_device"],
        }
    if preflight.get("target_access_path") != "synthetic":
        return {
            **base_report,
            "ok": False,
            "reason": "unsupported_target_access_path",
            "problems": [{"path": "target.access_path", "reason": "runner only supports synthetic access"}],
        }
    if not arm_synthetic:
        return {
            **base_report,
            "ok": True,
            "reason": "dry_run",
            "health": {"ok": True, "checks": ["preflight"]},
        }

    run = _run_synthetic_register(playbook)
    return {
        **base_report,
        "ok": bool(run["ok"]),
        "reason": run["reason"],
        "synthetic_execution_authorized": True,
        "executed": True,
        "initial_state_hash": stable_hash(run["initial_state"]),
        "restored_state_hash": stable_hash(run["restored_state"]),
        "write_readback": run["write_readback"],
        "restore_readback": run["restore_readback"],
        "restore": run["restore"],
        "health": run["health"],
        "ledger_rows": len(run["ledger_rows"]),
        "ledger_replay": run["ledger_replay"],
        "ledger": run["ledger_rows"],
    }


def _run_sysfs_led(
    playbook: dict[str, Any],
    *,
    allow_sudo: bool,
    allow_test_sysfs_paths: bool,
    command_runner: CommandRunner | None,
) -> dict[str, Any]:
    register = _first_register(playbook)
    target = dict(playbook.get("target") or {})
    device_info = dict(target.get("userspace_device") or {})
    paths = _sysfs_led_paths(device_info, allow_test_sysfs_paths=allow_test_sysfs_paths)
    readback_mask = _as_int(register["readback_mask"])
    mask = _as_int(register["mask"])
    device = str(register["device"])
    address = _hex_key(_as_int(register["address"]))

    command_log: list[dict[str, Any]] = []
    restore_log: list[dict[str, Any]] = []
    initial_physical = _read_sysfs_led(paths, command_log=command_log)
    original = int(initial_physical["brightness"]) & readback_mask
    max_brightness = int(initial_physical["max_brightness"])
    if max_brightness < 1:
        raise RuntimeError("sysfs LED max_brightness must be at least 1")
    if initial_physical["active_trigger"] != str(device_info.get("required_active_trigger", "none")):
        return _failed_sysfs_led_run(
            "sysfs_led_trigger_not_idle",
            paths=paths,
            initial=initial_physical,
            command_log=command_log,
            restore_log=restore_log,
        )

    write_value = min(_select_write_value(register, original, mask), max_brightness) & mask
    initial = {
        "registers": {device: {address: original}},
        "sysfs_led": {paths["led"]: initial_physical},
        "limit_flags": {},
        "readbacks": [],
    }
    constraints = {
        "registers": {
            "devices": {
                device: {
                    "addresses": {
                        address: {
                            "mask": _hex_key(mask),
                            "reset": _hex_key(original),
                            "value_max": _hex_key(max_brightness),
                        }
                    }
                }
            }
        }
    }
    before_readback = _readback(initial, device, address, readback_mask)
    write_action = ActionPacket(
        op="WRITE_REG",
        params={"device": device, "address": address, "value": _hex_key(write_value), "mask": _hex_key(mask)},
        expected_after={"registers": {device: {address: write_value}}},
    )
    write_goal = {
        "kind": "mutable_hil_write_sysfs_led",
        "device": device,
        "address": address,
        "value": _hex_key(write_value),
        "mask": _hex_key(mask),
        "led": paths["led"],
    }
    write_result = commit_action(initial, constraints, write_action)
    if not write_result.ok:
        return _failed_sysfs_led_run(
            "direct_action_verifier_rejected_write",
            paths=paths,
            initial=initial_physical,
            command_log=command_log,
            restore_log=restore_log,
            logical_result=write_result.to_dict(),
        )

    write_started = False
    try:
        _write_sysfs_value(paths["brightness"], write_value, allow_sudo=allow_sudo, command_runner=command_runner, command_log=command_log)
        write_started = True
        _settle(playbook)
        after_physical = _read_sysfs_led(paths, command_log=command_log)
        write_after = deepcopy(write_result.after_state)
        write_after["sysfs_led"] = {paths["led"]: after_physical}
        after_write_readback = _readback(write_after, device, address, readback_mask)

        invalid_mask = mask | _next_mask_bit(mask)
        invalid_action = ActionPacket(
            op="WRITE_REG",
            params={"device": device, "address": address, "value": _hex_key(write_value), "mask": _hex_key(invalid_mask)},
        )
        invalid_goal = {
            "kind": "mutable_hil_negative_invalid_mask",
            "device": device,
            "address": address,
            "value": _hex_key(write_value),
            "mask": _hex_key(invalid_mask),
            "led": paths["led"],
        }
        invalid_result = commit_action(initial, constraints, invalid_action)

        restore_action = ActionPacket(
            op="WRITE_REG",
            params={"device": device, "address": address, "value": _hex_key(original), "mask": _hex_key(mask)},
            expected_after={"registers": {device: {address: original}}},
        )
        restore_goal = {
            "kind": "mutable_hil_restore_sysfs_led",
            "device": device,
            "address": address,
            "value": _hex_key(original),
            "mask": _hex_key(mask),
            "led": paths["led"],
        }
        restore_result = commit_action(write_after, constraints, restore_action)
        _write_sysfs_value(
            paths["brightness"],
            original,
            allow_sudo=allow_sudo,
            command_runner=command_runner,
            command_log=command_log,
            restore_log=restore_log,
        )
        _settle(playbook)
        restored_physical = _read_sysfs_led(paths, command_log=command_log)
        restored = deepcopy(restore_result.after_state)
        restored["sysfs_led"] = {paths["led"]: restored_physical}
        restore_readback = _readback(restored, device, address, readback_mask)
    except Exception:
        if write_started:
            try:
                _write_sysfs_value(
                    paths["brightness"],
                    original,
                    allow_sudo=allow_sudo,
                    command_runner=command_runner,
                    command_log=command_log,
                    restore_log=restore_log,
                )
            except Exception as restore_exc:
                restore_log.append({"action": "restore_after_exception", "ok": False, "error": str(restore_exc)})
        raise

    rows = [
        direct_action_ledger_row(
            write_goal,
            constraints,
            write_action,
            write_result,
            measurement=_sysfs_led_measurement(
                paths,
                before_value=before_readback["value"],
                after_value=after_write_readback["value"],
                target_value=write_value,
                readback_ok=int(after_write_readback["value"] == (write_value & readback_mask)),
            ),
            lineage={"source": "mutable-hil-runner", "case": "real_sysfs_led_write"},
        ),
        direct_action_ledger_row(
            invalid_goal,
            constraints,
            invalid_action,
            invalid_result,
            measurement=_sysfs_led_measurement(paths, negative_control_invalid_mask=1),
            lineage={"source": "mutable-hil-runner", "case": "real_sysfs_led_invalid_mask"},
        ),
        direct_action_ledger_row(
            restore_goal,
            constraints,
            restore_action,
            restore_result,
            measurement=_sysfs_led_measurement(
                paths,
                after_restore_value=restore_readback["value"],
                restored_to_original=int(restore_readback["value"] == before_readback["value"]),
            ),
            lineage={"source": "mutable-hil-runner", "case": "real_sysfs_led_restore"},
        ),
    ]
    replay = replay_ledger_rows(rows)
    readback_ok = bool(write_result.ok and after_write_readback["value"] == (write_value & readback_mask))
    restore_ok = bool(restore_result.ok and restore_readback["value"] == before_readback["value"])
    trigger_ok = restored_physical["active_trigger"] == initial_physical["active_trigger"]
    negative_ok = bool(not invalid_result.ok and invalid_result.reason == "allowlist_violation")
    ok = bool(readback_ok and restore_ok and trigger_ok and negative_ok and replay["ok"])
    return {
        "ok": ok,
        "reason": None if ok else "real_sysfs_led_readback_restore_failed",
        "initial_state": initial,
        "restored_state": restored,
        "write_readback": {"before": before_readback, "after": after_write_readback, "ok": readback_ok},
        "restore_readback": {"after": restore_readback, "ok": restore_ok},
        "restore": {"ok": restore_ok and trigger_ok, "method": "write_original_then_readback"},
        "health": {
            "ok": ok,
            "checks": [
                "preflight",
                "active_trigger_none",
                "before_readback",
                "after_readback",
                "invalid_mask_negative",
                "restore_readback",
                "trigger_unchanged",
                "ledger_replay",
            ],
        },
        "ledger_rows": rows,
        "ledger_replay": replay,
        "command_log": command_log,
        "restore_log": restore_log,
        "real_device": {"kind": "sysfs-led", **paths, "max_brightness": max_brightness},
    }


def _failed_sysfs_led_run(
    reason: str,
    *,
    paths: dict[str, str],
    initial: dict[str, Any],
    command_log: list[dict[str, Any]],
    restore_log: list[dict[str, Any]],
    logical_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {"sysfs_led": {paths["led"]: initial}, "registers": {}, "limit_flags": {}}
    return {
        "ok": False,
        "reason": reason,
        "initial_state": state,
        "restored_state": state,
        "write_readback": {"ok": False, "before": None, "after": None},
        "restore_readback": {"ok": False, "after": None},
        "restore": {"ok": False, "method": "not executed"},
        "health": {"ok": False, "checks": ["preflight", "active_trigger_check"], "reason": reason},
        "ledger_rows": [],
        "ledger_replay": replay_ledger_rows([]),
        "command_log": command_log,
        "restore_log": restore_log,
        "real_device": {"kind": "sysfs-led", **paths},
        "logical_result": logical_result,
    }


def _sysfs_led_measurement(paths: dict[str, str], **metrics: Any) -> dict[str, Any]:
    payload_metrics = {"real_device": 1, "target_access_path": "userspace-device"}
    payload_metrics.update(metrics)
    return {
        "environment": {"userspace_device": {"kind": "sysfs-led", **deepcopy(paths)}},
        "metrics": payload_metrics,
    }


def _sysfs_led_paths(device_info: dict[str, Any], *, allow_test_sysfs_paths: bool) -> dict[str, str]:
    led = str(device_info.get("led") or "").strip()
    if not led or "/" in led:
        raise ValueError("sysfs LED name must be non-empty and must not contain path separators")
    paths = {
        "led": led,
        "brightness": str(device_info.get("brightness_path") or f"/sys/class/leds/{led}/brightness"),
        "max_brightness": str(device_info.get("max_brightness_path") or f"/sys/class/leds/{led}/max_brightness"),
        "trigger": str(device_info.get("trigger_path") or f"/sys/class/leds/{led}/trigger"),
    }
    if not allow_test_sysfs_paths:
        prefix = f"/sys/class/leds/{led}/"
        for key in ("brightness", "max_brightness", "trigger"):
            if not paths[key].startswith(prefix):
                raise ValueError(f"{key} path must stay under {prefix}")
    return paths


def _read_sysfs_led(paths: dict[str, str], *, command_log: list[dict[str, Any]]) -> dict[str, Any]:
    brightness_text = Path(paths["brightness"]).read_text(encoding="utf-8").strip()
    max_text = Path(paths["max_brightness"]).read_text(encoding="utf-8").strip()
    trigger_text = Path(paths["trigger"]).read_text(encoding="utf-8").strip()
    payload = {
        "brightness": int(brightness_text or "0"),
        "max_brightness": int(max_text or "0"),
        "trigger": trigger_text,
        "active_trigger": _active_trigger(trigger_text),
    }
    command_log.append({"action": "read_sysfs_led", "paths": dict(paths), "payload": payload, "ok": True})
    return payload


def _write_sysfs_value(
    path: str,
    value: int,
    *,
    allow_sudo: bool,
    command_runner: CommandRunner | None,
    command_log: list[dict[str, Any]],
    restore_log: list[dict[str, Any]] | None = None,
) -> None:
    destination = Path(path)
    entry = {"action": "write_sysfs_value", "path": str(destination), "value": int(value), "ok": False}
    try:
        destination.write_text(f"{int(value)}\n", encoding="utf-8")
        entry["method"] = "direct"
        entry["ok"] = True
    except OSError as exc:
        permission_denied = exc.errno in {errno.EACCES, errno.EPERM}
        if not permission_denied or not allow_sudo:
            entry["method"] = "direct"
            entry["error"] = "permission_denied" if permission_denied else str(exc)
            command_log.append(entry)
            if restore_log is not None:
                restore_log.append(dict(entry))
            raise
        runner = command_runner or subprocess.run
        argv = ["sudo", "tee", str(destination)]
        result = runner(argv, input=f"{int(value)}\n", stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
        entry["method"] = "sudo_tee"
        entry["argv"] = argv
        entry["returncode"] = int(result.returncode)
        entry["stdout"] = result.stdout
        entry["stderr"] = result.stderr
        entry["ok"] = result.returncode == 0
        if result.returncode != 0:
            command_log.append(entry)
            if restore_log is not None:
                restore_log.append(dict(entry))
            raise RuntimeError(f"failed to write {destination}: {result.stderr.strip()}")
    command_log.append(entry)
    if restore_log is not None:
        restore_log.append(dict(entry))


def _active_trigger(trigger_text: str) -> str:
    for token in trigger_text.split():
        if token.startswith("[") and token.endswith("]"):
            return token[1:-1]
    return ""


def _settle(playbook: dict[str, Any]) -> None:
    settle_ms = int((playbook.get("readback") or {}).get("settle_ms") or 0)
    if settle_ms > 0:
        time.sleep(settle_ms / 1000.0)


def ledger_rows_from_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("ledger", [])
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    return []


def _run_synthetic_register(playbook: dict[str, Any]) -> dict[str, Any]:
    register = _first_register(playbook)
    device = str(register["device"])
    address = _hex_key(_as_int(register["address"]))
    mask = _as_int(register["mask"])
    readback_mask = _as_int(register["readback_mask"])
    original = _initial_register_value(register)
    write_value = _select_write_value(register, original, mask)

    initial = {"registers": {device: {address: original}}, "limit_flags": {}, "readbacks": []}
    constraints = {
        "registers": {
            "devices": {
                device: {
                    "addresses": {
                        address: {
                            "mask": _hex_key(mask),
                            "reset": _hex_key(original),
                        }
                    }
                }
            }
        }
    }
    before_readback = _readback(initial, device, address, readback_mask)

    write_action = ActionPacket(
        op="WRITE_REG",
        params={"device": device, "address": address, "value": _hex_key(write_value), "mask": _hex_key(mask)},
        expected_after={"registers": {device: {address: write_value}}},
    )
    write_goal = {
        "kind": "mutable_hil_write_register",
        "device": device,
        "address": address,
        "value": _hex_key(write_value),
        "mask": _hex_key(mask),
    }
    write_result = commit_action(initial, constraints, write_action)
    write_after = write_result.after_state
    after_write_readback = _readback(write_after, device, address, readback_mask)

    invalid_mask = mask | _next_mask_bit(mask)
    invalid_action = ActionPacket(
        op="WRITE_REG",
        params={"device": device, "address": address, "value": _hex_key(write_value), "mask": _hex_key(invalid_mask)},
    )
    invalid_goal = {
        "kind": "mutable_hil_negative_invalid_mask",
        "device": device,
        "address": address,
        "value": _hex_key(write_value),
        "mask": _hex_key(invalid_mask),
    }
    invalid_result = commit_action(initial, constraints, invalid_action)

    restore_action = ActionPacket(
        op="WRITE_REG",
        params={"device": device, "address": address, "value": _hex_key(original), "mask": _hex_key(mask)},
        expected_after={"registers": {device: {address: original}}},
    )
    restore_goal = {
        "kind": "mutable_hil_restore_register",
        "device": device,
        "address": address,
        "value": _hex_key(original),
        "mask": _hex_key(mask),
    }
    restore_result = commit_action(write_after, constraints, restore_action)
    restored = restore_result.after_state
    restore_readback = _readback(restored, device, address, readback_mask)

    rows = [
        direct_action_ledger_row(
            write_goal,
            constraints,
            write_action,
            write_result,
            measurement={"readback": {"before": before_readback, "after": after_write_readback}, "synthetic": True},
            lineage={"source": "mutable-hil-runner", "case": "write"},
        ),
        direct_action_ledger_row(
            invalid_goal,
            constraints,
            invalid_action,
            invalid_result,
            measurement={"negative_control": "invalid_mask", "synthetic": True},
            lineage={"source": "mutable-hil-runner", "case": "invalid_mask"},
        ),
        direct_action_ledger_row(
            restore_goal,
            constraints,
            restore_action,
            restore_result,
            measurement={"readback": {"after_restore": restore_readback}, "synthetic": True},
            lineage={"source": "mutable-hil-runner", "case": "restore"},
        ),
    ]
    replay = replay_ledger_rows(rows)
    restore_ok = bool(restore_result.ok and restore_readback["value"] == before_readback["value"])
    readback_ok = bool(write_result.ok and after_write_readback["value"] == (write_value & readback_mask))
    negative_ok = bool(not invalid_result.ok and invalid_result.reason == "allowlist_violation")
    ok = bool(readback_ok and restore_ok and negative_ok and replay["ok"])
    return {
        "ok": ok,
        "reason": None if ok else "synthetic_readback_restore_failed",
        "initial_state": initial,
        "restored_state": restored,
        "write_readback": {
            "before": before_readback,
            "after": after_write_readback,
            "ok": readback_ok,
        },
        "restore_readback": {
            "after": restore_readback,
            "ok": restore_ok,
        },
        "restore": {
            "ok": restore_ok,
            "method": "write_original_then_readback",
        },
        "health": {
            "ok": ok,
            "checks": ["preflight", "before_readback", "after_readback", "invalid_mask_negative", "restore_readback", "ledger_replay"],
        },
        "ledger_rows": rows,
        "ledger_replay": replay,
    }


def _first_register(playbook: dict[str, Any]) -> dict[str, Any]:
    registers = playbook.get("allowlist", {}).get("registers", [])
    if not registers:
        raise ValueError("playbook does not define an allowlisted register")
    register = registers[0]
    if not isinstance(register, dict):
        raise ValueError("first register allowlist entry must be an object")
    return deepcopy(register)


def _initial_register_value(register: dict[str, Any]) -> int:
    if "reset" in register:
        return _as_int(register["reset"])
    if "restore_value" in register:
        return _as_int(register["restore_value"])
    return 0


def _select_write_value(register: dict[str, Any], original: int, mask: int) -> int:
    safe_values = register.get("safe_values", [])
    for value in safe_values:
        parsed = _as_int(value)
        if parsed != original:
            return parsed & mask
    return (original ^ (mask & -mask)) & mask


def _readback(state: dict[str, Any], device: str, address: str, readback_mask: int) -> dict[str, Any]:
    value = _as_int(state.get("registers", {}).get(device, {}).get(address, 0)) & readback_mask
    return {"device": device, "address": address, "mask": _hex_key(readback_mask), "value": value}


def _next_mask_bit(mask: int) -> int:
    bit = 1
    while mask & bit:
        bit <<= 1
    return bit


def _as_int(value: Any) -> int:
    if isinstance(value, str):
        return int(value, 0)
    return int(value)


def _hex_key(value: int) -> str:
    return hex(int(value))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a mutable HIL playbook proof with explicit arming.")
    parser.add_argument("--playbook", help="Playbook JSON to run.")
    parser.add_argument("--write-template", help="Write the synthetic playbook template before running it.")
    parser.add_argument("--write-kv260-sysfs-led-template", help="Write a KV260 sysfs LED playbook template before running it.")
    parser.add_argument("--led-name", default="vbus_det", help="LED name for --write-kv260-sysfs-led-template.")
    parser.add_argument("--expect-model", help="Require the platform model string to contain this value.")
    parser.add_argument("--arm-synthetic", action="store_true", help="Execute the synthetic readback/restore proof.")
    parser.add_argument("--arm-real-sysfs-led", action="store_true", help="Execute an explicitly allowlisted sysfs LED userspace-device target.")
    parser.add_argument("--allow-sudo", action="store_true", help="Allow sudo tee for root-owned sysfs LED brightness writes.")
    parser.add_argument("--write", help="Write runner report JSON; stdout when omitted.")
    parser.add_argument("--write-ledger-jsonl", help="Write emitted ledger rows when synthetic execution runs.")
    args = parser.parse_args(argv)

    if args.write_template:
        write_json(synthetic_mutable_register_playbook(), args.write_template)
    if args.write_kv260_sysfs_led_template:
        write_json(kv260_sysfs_led_playbook(args.led_name), args.write_kv260_sysfs_led_template)
    playbook_path = args.playbook or args.write_template or args.write_kv260_sysfs_led_template
    if not playbook_path:
        parser.error("--playbook, --write-template, or --write-kv260-sysfs-led-template is required")
    playbook = load_playbook(playbook_path)
    if playbook.get("schema_version") != PLAYBOOK_SCHEMA_VERSION:
        parser.error(f"playbook schema_version must be {PLAYBOOK_SCHEMA_VERSION}")
    report = run_mutable_hil_playbook(
        playbook,
        arm_synthetic=args.arm_synthetic,
        arm_real_sysfs_led=args.arm_real_sysfs_led,
        allow_sudo=args.allow_sudo,
        expected_model=args.expect_model,
    )
    rows = ledger_rows_from_report(report)
    if args.write_ledger_jsonl and rows:
        write_ledger_jsonl(rows, args.write_ledger_jsonl)
    write_json({key: value for key, value in report.items() if key != "ledger"}, args.write)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
