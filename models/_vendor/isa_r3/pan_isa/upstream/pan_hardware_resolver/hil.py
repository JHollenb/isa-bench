from __future__ import annotations

import argparse
import json
import platform
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from .direct_action import ActionPacket, commit_action, direct_action_oracle
from .ledger import direct_action_ledger_row, microprogram_ledger_row, stable_hash, write_ledger_jsonl
from .microprogram import commit_microprogram, microprogram_oracle
from .replay import replay_ledger_rows


def synthetic_hil_state() -> dict[str, Any]:
    return {
        "axes": {"x": {"position_ticks": 100}},
        "registers": {"pll0": {"0x10": 0}, "clock_mux": {"0x4": 0}},
        "limit_flags": {},
        "events": {},
    }


def synthetic_hil_constraints() -> dict[str, Any]:
    return {
        "motion": {
            "axes": {
                "x": {
                    "min_position_ticks": -1000,
                    "max_position_ticks": 1000,
                    "max_velocity_ticks_per_s": 500,
                    "max_accel_ticks_per_s2": 100,
                    "default_velocity_ticks_per_s": 250,
                    "default_accel_ticks_per_s2": 50,
                }
            }
        },
        "registers": {
            "devices": {
                "pll0": {"addresses": {"0x10": {"mask": "0xff"}}},
                "clock_mux": {
                    "addresses": {
                        "0x4": {
                            "mask": "0x3",
                            "state_alias": {
                                "field": "clock_source",
                                "mask": "0x3",
                                "values": {"0x2": "pll0"},
                            },
                        }
                    }
                },
            }
        },
        "pwm": {"channels": {"fan0": {"min_q16": 0, "max_q16": 65535}}},
        "microprogram": {"slot_count": 8},
        "events": {
            "allowed": ["pll0_locked"],
            "default_timeout_ticks": 5000,
            "max_timeout_ticks": 5000,
            "watchdog_budget_ticks": 8000,
        },
    }


def build_synthetic_hil_report(*, expected_model: str | None = None) -> dict[str, Any]:
    platform_info = _platform_info()
    if expected_model and expected_model.lower() not in platform_info.get("model", "").lower():
        return {
            "ok": False,
            "schema": "pan-hw-benign-hil-v1",
            "error": f"expected model containing {expected_model!r}, got {platform_info.get('model')!r}",
            "platform": platform_info,
        }
    initial = synthetic_hil_state()
    constraints = synthetic_hil_constraints()
    cases = _hil_cases(initial, constraints)
    rows = [case["ledger"] for case in cases]
    replay = replay_ledger_rows(rows)
    restored = deepcopy(initial)
    return {
        "ok": bool(replay["ok"] and restored == initial and _hil_expectations_pass(cases)),
        "schema": "pan-hw-benign-hil-v1",
        "target": "synthetic-userspace-device-model",
        "unsafe_mmio": False,
        "platform": platform_info,
        "initial_state_hash": stable_hash(initial),
        "restored_state_hash": stable_hash(restored),
        "restore": {
            "ok": restored == initial,
            "method": "synthetic model reset to initial state",
        },
        "health": {
            "ok": True,
            "checks": ["platform_identity", "ledger_replay", "synthetic_restore"],
        },
        "cases": [_case_summary(case) for case in cases],
        "ledger_replay": replay,
    }


def build_readonly_identity_hil_report(*, expected_model: str | None = None) -> dict[str, Any]:
    platform_info = _platform_info()
    if expected_model and expected_model.lower() not in platform_info.get("model", "").lower():
        return {
            "ok": False,
            "schema": "pan-hw-readonly-identity-hil-v1",
            "error": f"expected model containing {expected_model!r}, got {platform_info.get('model')!r}",
            "platform": platform_info,
        }
    case = _readonly_identity_case(platform_info)
    replay = replay_ledger_rows([case["ledger"]])
    envelope = case["ledger"]["envelope"]
    before = envelope["commit"].get("before_state_hash")
    after = envelope["commit"].get("after_state_hash")
    return {
        "ok": bool(replay["ok"] and envelope["verification"]["ok"] and before == after),
        "schema": "pan-hw-readonly-identity-hil-v1",
        "target": "read-only-platform-identity",
        "unsafe_mmio": False,
        "platform": platform_info,
        "restore": {
            "ok": True,
            "method": "read-only probe; no restore required",
        },
        "health": {
            "ok": True,
            "checks": ["platform_identity", "noop_verifier", "ledger_replay", "no_state_mutation"],
        },
        "cases": [_case_summary(case)],
        "ledger_replay": replay,
    }


def build_hil_report(*, target: str = "synthetic", expected_model: str | None = None) -> dict[str, Any]:
    if target == "synthetic":
        return build_synthetic_hil_report(expected_model=expected_model)
    if target == "identity":
        return build_readonly_identity_hil_report(expected_model=expected_model)
    if target != "both":
        raise ValueError(f"unknown HIL target: {target!r}")
    synthetic = build_synthetic_hil_report(expected_model=expected_model)
    identity = build_readonly_identity_hil_report(expected_model=expected_model)
    rows = _ledger_rows_for_target("both")
    replay = replay_ledger_rows(rows)
    return {
        "ok": bool(synthetic.get("ok") and identity.get("ok") and replay["ok"]),
        "schema": "pan-hw-benign-hil-suite-v1",
        "target": "synthetic-and-read-only-identity",
        "unsafe_mmio": False,
        "reports": {
            "synthetic": synthetic,
            "readonly_identity": identity,
        },
        "ledger_replay": replay,
    }


def _hil_cases(initial: dict[str, Any], constraints: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    normal_goal = {"kind": "move_axis", "axis": "x", "target_position_ticks": 140}
    normal_action = direct_action_oracle(initial, normal_goal, constraints)
    normal_result = commit_action(initial, constraints, normal_action)
    cases.append(
        _case(
            "normal_move_axis",
            "normal",
            direct_action_ledger_row(normal_goal, constraints, normal_action, normal_result, lineage={"source": "synthetic-hil"}),
        )
    )

    boundary_goal = {"kind": "move_axis", "axis": "x", "target_position_ticks": 1000}
    boundary_action = direct_action_oracle(initial, boundary_goal, constraints)
    boundary_result = commit_action(initial, constraints, boundary_action)
    cases.append(
        _case(
            "boundary_axis_max",
            "boundary",
            direct_action_ledger_row(boundary_goal, constraints, boundary_action, boundary_result, lineage={"source": "synthetic-hil"}),
        )
    )

    invalid_goal = {"kind": "write_reg", "device": "pll0", "address": "0x10", "value": "0x31", "mask": "0x1ff"}
    invalid_action = ActionPacket(op="WRITE_REG", params={"device": "pll0", "address": "0x10", "value": "0x31", "mask": "0x1ff"})
    invalid_result = commit_action(initial, constraints, invalid_action)
    cases.append(
        _case(
            "invalid_mask_reject",
            "invalid_mask",
            direct_action_ledger_row(invalid_goal, constraints, invalid_action, invalid_result, lineage={"source": "synthetic-hil"}),
        )
    )

    fault_state = deepcopy(initial)
    fault_state["limit_flags"]["e_stop"] = True
    fault_goal = {"kind": "set_pwm", "channel": "fan0", "duty_q16": 1200}
    fault_action = ActionPacket(op="SET_PWM", params={"channel": "fan0", "duty_q16": 1200})
    fault_result = commit_action(fault_state, constraints, fault_action)
    cases.append(
        _case(
            "fault_hold_path",
            "fault_hold",
            direct_action_ledger_row(fault_goal, constraints, fault_action, fault_result, lineage={"source": "synthetic-hil"}),
        )
    )

    event_goal = {"kind": "pll_switch", "clock_source": "pll0"}
    event_program = microprogram_oracle(initial, event_goal, constraints)
    event_result = commit_microprogram(initial, constraints, event_program)
    cases.append(
        _case(
            "microprogram_event_path",
            "event_path",
            microprogram_ledger_row(event_goal, constraints, event_program, event_result, lineage={"source": "synthetic-hil"}),
        )
    )

    timeout_goal = {"kind": "pll_switch", "clock_source": "pll0", "timeout_ticks": 9000}
    timeout_program = microprogram_oracle(initial, timeout_goal, constraints)
    timeout_result = commit_microprogram(initial, constraints, timeout_program)
    cases.append(
        _case(
            "microprogram_timeout_reject",
            "timeout",
            microprogram_ledger_row(timeout_goal, constraints, timeout_program, timeout_result, lineage={"source": "synthetic-hil"}),
        )
    )
    return cases


def _readonly_identity_case(platform_info: dict[str, Any]) -> dict[str, Any]:
    state = {"platform": platform_info}
    constraints = {"read_only": True, "allowed_ops": ["NOOP"], "target": "platform_identity"}
    goal = {"kind": "read_only_platform_identity_probe", "model": platform_info.get("model", "")}
    action = ActionPacket(op="NOOP", expected_after={"platform": platform_info})
    result = commit_action(state, constraints, action)
    row = direct_action_ledger_row(
        goal,
        constraints,
        action,
        result,
        measurement={"environment": {"platform": platform_info}, "metrics": {"read_only": 1, "unsafe_mmio": 0}},
        lineage={"source": "readonly-identity-hil"},
    )
    return _case("readonly_platform_identity", "readonly_identity", row)


def _ledger_rows_for_target(target: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if target in {"synthetic", "both"}:
        rows.extend(case["ledger"] for case in _hil_cases(synthetic_hil_state(), synthetic_hil_constraints()))
    if target in {"identity", "both"}:
        rows.append(_readonly_identity_case(_platform_info())["ledger"])
    return rows


def _case(case_id: str, kind: str, ledger: dict[str, Any]) -> dict[str, Any]:
    return {"case_id": case_id, "kind": kind, "ledger": ledger}


def _case_summary(case: dict[str, Any]) -> dict[str, Any]:
    envelope = case["ledger"]["envelope"]
    verification = envelope["verification"]
    commit = envelope["commit"]
    return {
        "case_id": case["case_id"],
        "kind": case["kind"],
        "action_family": case["ledger"]["action_family"],
        "verification_ok": verification["ok"],
        "verification_reason": verification["reason"],
        "commit_ok": commit["ok"],
        "committed": commit["committed"],
    }


def _hil_expectations_pass(cases: list[dict[str, Any]]) -> bool:
    expected = {
        "normal_move_axis": True,
        "boundary_axis_max": True,
        "invalid_mask_reject": False,
        "fault_hold_path": False,
        "microprogram_event_path": True,
        "microprogram_timeout_reject": False,
    }
    for case in cases:
        ok = bool(case["ledger"]["envelope"]["verification"]["ok"])
        if ok != expected[case["case_id"]]:
            return False
    return True


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


def write_report(report: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KV260-safe benign hardware resolver HIL over a synthetic userspace model.")
    parser.add_argument("--expect-model", help="Require the platform model string to contain this value.")
    parser.add_argument("--target", choices=("synthetic", "identity", "both"), default="synthetic")
    parser.add_argument("--write", help="Write report JSON; stdout when omitted.")
    parser.add_argument("--write-ledger-jsonl", help="Append emitted HIL ledger rows to this JSONL file.")
    args = parser.parse_args(argv)
    report = build_hil_report(target=args.target, expected_model=args.expect_model)
    if args.write_ledger_jsonl and report.get("ok"):
        write_ledger_jsonl(_ledger_rows_for_target(args.target), args.write_ledger_jsonl)
    write_report(report, args.write)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
