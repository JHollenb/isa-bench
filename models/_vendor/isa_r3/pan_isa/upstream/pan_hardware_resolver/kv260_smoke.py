from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .direct_action import commit_action, direct_action_oracle, verify_action
from .ledger import direct_action_ledger_row, microprogram_ledger_row, write_ledger_jsonl
from .microprogram import commit_microprogram, microprogram_oracle, verify_microprogram


def build_report() -> dict[str, Any]:
    direct_state = {"axes": {"x": {"position_ticks": 100}}, "limit_flags": {}}
    direct_constraints = {
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
        }
    }
    direct_action = direct_action_oracle(
        direct_state,
        {"kind": "move_axis", "axis": "x", "target_position_ticks": 140},
        direct_constraints,
    )
    direct_goal = {"kind": "move_axis", "axis": "x", "target_position_ticks": 140}
    direct_verification = verify_action(direct_state, direct_constraints, direct_action)
    direct_commit = commit_action(direct_state, direct_constraints, direct_action)

    micro_state = {
        "registers": {"pll0": {"0x10": 0}, "clock_mux": {"0x4": 0}},
        "limit_flags": {},
        "events": {},
    }
    micro_constraints = {
        "microprogram": {"slot_count": 8},
        "events": {
            "allowed": ["pll0_locked"],
            "default_timeout_ticks": 5000,
            "max_timeout_ticks": 5000,
            "watchdog_budget_ticks": 8000,
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
    }
    micro_goal = {"kind": "pll_switch", "clock_source": "pll0"}
    microprogram = microprogram_oracle(micro_state, micro_goal, micro_constraints)
    micro_verification = verify_microprogram(micro_state, micro_constraints, microprogram)
    micro_commit = commit_microprogram(micro_state, micro_constraints, microprogram)
    platform = _platform_report()
    direct_ledger = direct_action_ledger_row(
        direct_goal,
        direct_constraints,
        direct_action,
        direct_commit,
        measurement={
            "environment": {"runner": "pan_hardware_resolver.kv260_smoke", "platform": platform},
            "metrics": {"verification_ok": direct_verification.ok, "commit_ok": direct_commit.ok},
        },
        lineage={"source": "kv260-safe-smoke:direct-action"},
    )
    micro_ledger = microprogram_ledger_row(
        micro_goal,
        micro_constraints,
        microprogram,
        micro_commit,
        measurement={
            "environment": {"runner": "pan_hardware_resolver.kv260_smoke", "platform": platform},
            "metrics": {"verification_ok": micro_verification.ok, "commit_ok": micro_commit.ok},
        },
        lineage={"source": "kv260-safe-smoke:microprogram"},
    )

    report = {
        "platform": platform,
        "direct_action": {
            "ok": direct_verification.ok and direct_commit.ok,
            "action": direct_action.to_dict(),
            "verification": direct_verification.to_dict(),
            "commit": direct_commit.to_dict(),
            "ledger": direct_ledger,
        },
        "microprogram": {
            "ok": micro_verification.ok and micro_commit.ok,
            "program": microprogram.to_dict(),
            "verification": micro_verification.to_dict(),
            "commit": micro_commit.to_dict(),
            "ledger": micro_ledger,
        },
    }
    report["ok"] = bool(report["direct_action"]["ok"] and report["microprogram"]["ok"])
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run KV260-safe PAN hardware resolver smoke tests.")
    parser.add_argument(
        "--expect-model",
        help="Optional case-insensitive substring that must appear in /proc/device-tree/model.",
    )
    parser.add_argument("--write-ledger-jsonl", help="Append the two KV260-safe smoke ledger rows to this JSONL file.")
    args = parser.parse_args(argv)

    report = build_report()
    if args.expect_model:
        model = str(report["platform"].get("model", "")).lower()
        report["model_expectation_ok"] = args.expect_model.lower() in model
        report["ok"] = bool(report["ok"] and report["model_expectation_ok"])
    if args.write_ledger_jsonl:
        write_ledger_jsonl(
            [report["direct_action"]["ledger"], report["microprogram"]["ledger"]],
            args.write_ledger_jsonl,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def _platform_report() -> dict[str, Any]:
    return {
        "model": _read_text(Path("/proc/device-tree/model")),
        "compatible": _read_text(Path("/proc/device-tree/compatible")),
        "kernel": _read_text(Path("/proc/sys/kernel/osrelease")),
        "machine": _read_machine(),
    }


def _read_text(path: Path) -> str | None:
    try:
        return path.read_bytes().replace(b"\x00", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return None


def _read_machine() -> str | None:
    try:
        import platform

        return platform.machine()
    except OSError:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
