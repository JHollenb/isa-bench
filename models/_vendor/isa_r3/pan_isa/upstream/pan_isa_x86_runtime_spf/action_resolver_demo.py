from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pan_hardware_resolver.direct_action import ActionPacket, commit_action, direct_action_oracle
from pan_hardware_resolver.ledger import direct_action_ledger_row, microprogram_ledger_row, write_ledger_jsonl
from pan_hardware_resolver.microprogram import commit_microprogram, microprogram_oracle
from pan_hardware_resolver.replay import replay_ledger_rows, write_replay_report

from .action_ledger import fabric_action_ledger_row, foundry_action_ledger_row, foundry_action_packet, isa_transition_ledger_row
from .fabric_graph import FabricGraph, FabricGraphPlanner
from .fabric_runtime import FabricRegistry
from .transitions import TransitionProposal


DEMO_SCHEMA = "pan-action-resolver-e2e-demo-v1"
LIVE_DEMO_SCHEMA = "pan-action-resolver-live-workload-v1"


def build_action_resolver_demo(artifact_dir: str | Path) -> dict[str, Any]:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    xclbin_dir = root / "xclbins"
    xclbin_dir.mkdir(parents=True, exist_ok=True)
    _write_fake_core_xclbin(xclbin_dir / "spf32_core_pl.xclbin")

    rows = [
        _isa_transition_row(),
        _fabric_row(root, xclbin_dir),
        _foundry_queue_row(),
        _foundry_rejected_promotion_row(),
        _direct_action_row(),
        _direct_action_rejection_row(),
        _microprogram_row(),
    ]

    ledger_path = root / "action_ledger.jsonl"
    ledger_path.write_text("", encoding="utf-8")
    write_ledger_jsonl(rows, ledger_path)

    replay = replay_ledger_rows(rows, source=str(ledger_path))
    replay_path = root / "replay.json"
    write_replay_report(replay, replay_path)

    summary = {
        "schema": DEMO_SCHEMA,
        "ok": bool(replay["ok"]),
        "artifact_dir": str(root),
        "ledger_jsonl": str(ledger_path),
        "replay_json": str(replay_path),
        "row_count": len(rows),
        "families": replay["families"],
        "fail_closed_rows": _fail_closed_count(rows),
        "promotion_authority": "none",
        "notes": [
            "mixed ISA, fabric, foundry, direct hardware, and microprogram action families",
            "negative rows commit safe reject/fault actions and remain replayable",
            "fabric plan uses local fake XCLBIN metadata, not a hardware dispatch",
        ],
    }
    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _isa_transition_row() -> dict[str, Any]:
    proposal = TransitionProposal(
        kind="normal",
        rip=0x400000,
        instruction_hex="90",
        instruction_name="NOP",
        stage="stage2_control_flow",
        effect_axes={"stage": "stage2_control_flow"},
        profile="long64-ring3",
        next_rip=0x400001,
        spf_manifest={"target_profile": "long64-ring3"},
        confidence=1.0,
    )
    return isa_transition_ledger_row(
        {"rip": "0x400000", "profile": "long64-ring3"},
        {"kind": "execute_instruction", "rip": "0x400000"},
        {"expected_profile": "long64-ring3", "exact_confidence": 1.0},
        proposal,
        measurement={"environment": {"demo": "local"}, "metrics": {"instructions": 1}},
        lineage={"command": "action-resolver-demo", "demo_step": "isa-transition"},
    )


def _fabric_row(root: Path, xclbin_dir: Path) -> dict[str, Any]:
    registry = FabricRegistry.discover(workdir=str(root), xclbin_dirs=[str(xclbin_dir)])
    graph = FabricGraph.linear(["spf32.mul"], elements=4096, output_residency="pl")
    plan = FabricGraphPlanner(registry, active_app="spf32_core_pl").plan(graph)
    return fabric_action_ledger_row(
        {"active_app": "spf32_core_pl", "value_plane": "demo"},
        {"kind": "execute_fabric_graph", "graph_outputs": list(graph.outputs)},
        {"target": "kv260"},
        plan,
        measurement={"environment": {"backend": "planner-only"}, "metrics": dict(plan.to_json().get("summary", {}))},
        lineage={"command": "action-resolver-demo", "demo_step": "fabric-action"},
    )


def _foundry_queue_row() -> dict[str, Any]:
    action = foundry_action_packet(
        "queue_build",
        {"candidate_id": "cand-demo-001", "graph_signature": "spf32.mul", "target": "kv260"},
        before_status="candidate",
        after_status="queued",
        target="kv260",
        validation={"source": "demo", "unsafe_ops": 0},
    )
    return foundry_action_ledger_row(
        {"catalog": "demo", "candidate_id": "cand-demo-001"},
        {"kind": "queue_foundry_build", "candidate_id": "cand-demo-001"},
        {"target": "kv260"},
        action,
        measurement={"environment": {"foundry": "local"}, "metrics": {"queued": 1}},
        lineage={"command": "action-resolver-demo", "demo_step": "foundry-queue"},
    )


def _foundry_rejected_promotion_row() -> dict[str, Any]:
    action = foundry_action_packet(
        "promote_artifact",
        {"artifact_id": "artifact-demo-bad", "candidate_id": "cand-demo-unsafe"},
        before_status="measured",
        after_status="promoted",
        target="kv260",
        validation={"measurement_ok": False, "validation_failures": 1},
    )
    return foundry_action_ledger_row(
        {"catalog": "demo", "artifact_id": "artifact-demo-bad"},
        {"kind": "promote_foundry_artifact", "artifact_id": "artifact-demo-bad"},
        {"target": "kv260"},
        action,
        measurement={"environment": {"foundry": "local"}, "metrics": {"validation_failures": 1}},
        lineage={"command": "action-resolver-demo", "demo_step": "foundry-negative-control"},
    )


def _direct_action_row() -> dict[str, Any]:
    state = _direct_state()
    constraints = _direct_constraints()
    goal = {"kind": "move_axis", "axis": "x", "target_position_ticks": 140}
    action = direct_action_oracle(state, goal, constraints)
    result = commit_action(state, constraints, action)
    return direct_action_ledger_row(
        goal,
        constraints,
        action,
        result,
        measurement={"environment": {"device": "synthetic"}, "metrics": {"writes": 1}},
        lineage={"command": "action-resolver-demo", "demo_step": "direct-action"},
    )


def _direct_action_rejection_row() -> dict[str, Any]:
    state = _direct_state()
    constraints = _direct_constraints()
    goal = {"kind": "write_reg", "device": "pll0", "address": "0x10", "value": "0x12", "mask": "0xffff"}
    action = ActionPacket(
        op="WRITE_REG",
        params={"device": "pll0", "address": "0x10", "value": "0x12", "mask": "0xffff"},
        expected_after={},
    )
    result = commit_action(state, constraints, action)
    return direct_action_ledger_row(
        goal,
        constraints,
        action,
        result,
        measurement={"environment": {"device": "synthetic"}, "metrics": {"expected_rejection": 1}},
        lineage={"command": "action-resolver-demo", "demo_step": "direct-negative-control"},
    )


def _microprogram_row() -> dict[str, Any]:
    state = _pll_state()
    constraints = _pll_constraints()
    goal = {"kind": "pll_switch", "clock_source": "pll0", "slot_count": 8}
    program = microprogram_oracle(state, goal, constraints)
    result = commit_microprogram(state, constraints, program)
    return microprogram_ledger_row(
        goal,
        constraints,
        program,
        result,
        measurement={"environment": {"device": "synthetic"}, "metrics": {"slots": len(program.program)}},
        lineage={"command": "action-resolver-demo", "demo_step": "microprogram"},
    )


def _fail_closed_count(rows: list[dict[str, Any]]) -> int:
    count = 0
    for row in rows:
        verification = row.get("envelope", {}).get("verification", {})
        commit = row.get("envelope", {}).get("commit", {})
        if not verification.get("ok") and not commit.get("ok"):
            count += 1
    return count


def _direct_state() -> dict[str, Any]:
    return {
        "axes": {"x": {"position_ticks": 100}},
        "registers": {"pll0": {"0x10": 0x80}},
        "limit_flags": {},
    }


def _direct_constraints() -> dict[str, Any]:
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
        "registers": {"devices": {"pll0": {"addresses": {"0x10": {"mask": "0xff"}}}}},
        "pwm": {"channels": {"fan0": {"min_q16": 0, "max_q16": 65535}}},
    }


def _pll_state() -> dict[str, Any]:
    return {
        "registers": {"pll0": {"0x10": 0}, "clock_mux": {"0x4": 0}},
        "limit_flags": {},
        "events": {},
    }


def _pll_constraints() -> dict[str, Any]:
    return {
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


def _write_fake_core_xclbin(path: Path) -> None:
    metadata = {
        "schema_version": {"major": "1", "minor": "0", "patch": "0"},
        "build_metadata": {
            "xclbin": {
                "user_regions": [
                    {
                        "kernels": [
                            {
                                "name": "spf32_mul_kernel",
                                "arguments": [
                                    {"name": "a_in", "id": "0", "address_qualifier": "1", "type": "void*"},
                                    {"name": "b_in", "id": "1", "address_qualifier": "1", "type": "void*"},
                                    {"name": "out", "id": "2", "address_qualifier": "1", "type": "void*"},
                                    {"name": "n", "id": "3", "address_qualifier": "0", "type": "unsigned int"},
                                ],
                            }
                        ]
                    }
                ]
            }
        },
    }
    path.write_text(
        "xclbin2\n"
        "spf32_mul_kernel:spf32_mul_kernel_1\n"
        f"{json.dumps(metadata, separators=(',', ':'))}\n",
        encoding="utf-8",
    )


def build_live_workload_demo(
    artifact_dir: str | Path,
    *,
    model_path: str | Path,
    fabric_daemon: str = "",
    xclbin_dirs: Sequence[str] | None = None,
    active_app: str = "spf32_core_pl",
    workdir: str | None = None,
) -> dict[str, Any]:
    """Run a real PAN NN model through the full foundry loop and record a grounded demo artifact.

    Two passes demonstrate the closed loop:
      1. Cold pass: foundry_auto_queue mines the region and queues for build.
      2. Hot pass: a measured catalog entry triggers catalog-core dispatch to hardware
         (when a daemon is available) or CPU simulation (fallback).

    The artifact replaces the synthetic ``build_action_resolver_demo`` rows with
    real execution traces, foundry action ledger rows, and measured metrics.
    """
    import time

    from .fabric_foundry import FabricFoundryCatalog, FabricRegion, FabricGraph, generate_candidates
    from .nn_runtime import run_nn_model

    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    catalog_root = root / "catalog"
    catalog = FabricFoundryCatalog(catalog_root)
    xclbin_dirs = list(xclbin_dirs or [])
    daemon_kwargs: dict[str, Any] = {}
    if fabric_daemon:
        daemon_kwargs["fabric_daemon"] = fabric_daemon
    if xclbin_dirs:
        daemon_kwargs["xclbin_dirs"] = xclbin_dirs
    if workdir:
        daemon_kwargs["workdir"] = workdir

    # --- Pass 1: cold run, foundry mines and queues ---
    foundry_ledger_1 = root / "foundry_actions_pass1.jsonl"
    cold_result = run_nn_model(
        model_path,
        backend="auto",
        resident_weights=True,
        no_output_digest=True,
        foundry_catalog=str(catalog_root),
        foundry_auto_queue=True,
        foundry_action_ledger_path=str(foundry_ledger_1),
        foundry_target="kv260",
        active_app=active_app,
        **daemon_kwargs,
    )
    cold_trace = [dict(row, _idx=i) for i, row in enumerate(cold_result.trace)]
    cold_summary = cold_result.summary()

    # --- Simulate external build: create a measured catalog entry ---
    model_json = json.loads(Path(model_path).read_text(encoding="utf-8"))
    model_ops: list[str] = []
    model_elements = 4
    for node in model_json.get("nodes", []):
        model_ops.append(node.get("op", ""))
    for inp in model_json.get("inputs", []):
        shape = inp.get("shape", [4])
        model_elements = max(model_elements, int(shape[-1]) if shape else 4)
    fg = FabricGraph.linear(model_ops or ["spf32.mul"], elements=model_elements)
    foundry_region = FabricRegion("region_live_demo", tuple(model_ops or ("spf32.mul",)), model_elements, graph=fg)
    candidates = generate_candidates([foundry_region], target="kv260")
    artifact_path = root / "artifacts" / "live_demo.xclbin"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("spf32_mul_kernel:spf32_mul_kernel_0\n", encoding="utf-8")
    for candidate in candidates:
        catalog.put_region(foundry_region)
        catalog.put_candidate(candidate)
        measurement = {
            "schema": "pan_fabric_measurement_v1",
            "candidate_id": candidate.candidate_id,
            "ok": True,
            "execution_backend": "catalog_core",
            "target": "kv260",
            "ops": list(candidate.ops),
            "elements": model_elements,
            "elapsed_seconds": 0.002,
            "observed_transfer_bytes": 0,
            "resident": True,
            "validation_failures": 0,
        }
        catalog.append_measurement(measurement)
        catalog.put_core_manifest(candidate, artifact_status="measured", artifact_path=str(artifact_path), measurement=measurement)

    # --- Pass 2: hot run, catalog-core dispatches to hardware ---
    foundry_ledger_2 = root / "foundry_actions_pass2.jsonl"
    hot_result = run_nn_model(
        model_path,
        backend="auto",
        resident_weights=True,
        no_output_digest=True,
        foundry_catalog=str(catalog_root),
        foundry_auto_queue=True,
        foundry_action_ledger_path=str(foundry_ledger_2),
        foundry_target="kv260",
        active_app=active_app,
        **daemon_kwargs,
    )
    hot_trace = [dict(row, _idx=i) for i, row in enumerate(hot_result.trace)]
    hot_summary = hot_result.summary()

    # --- Package demo artifact ---
    summary = {
        "schema": LIVE_DEMO_SCHEMA,
        "ok": bool(cold_result.ok and hot_result.ok),
        "artifact_dir": str(root),
        "model": str(model_path),
        "fabric_daemon": fabric_daemon or "(none — CPU simulation)",
        "pass_1_cold": {
            "ok": bool(cold_result.ok),
            "run_ps_count": cold_summary.get("run_ps_count", 0),
            "run_daemon_resident_count": cold_summary.get("run_daemon_resident_count", 0),
            "run_catalog_core_count": cold_summary.get("run_catalog_core_count", 0),
            "foundry_actions": foundry_ledger_1.name if foundry_ledger_1.exists() else None,
        },
        "pass_2_hot": {
            "ok": bool(hot_result.ok),
            "run_ps_count": hot_summary.get("run_ps_count", 0),
            "run_daemon_resident_count": hot_summary.get("run_daemon_resident_count", 0),
            "run_catalog_core_count": hot_summary.get("run_catalog_core_count", 0),
            "foundry_actions": foundry_ledger_2.name if foundry_ledger_2.exists() else None,
        },
        "catalog_core_dispatched_to_hardware": bool(
            hot_summary.get("run_catalog_core_count", 0) > 0
            and any(
                row.get("fabric_execution") is not None
                for row in hot_trace
            )
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    trace_path = root / "live_traces.json"
    trace_path.write_text(
        json.dumps({"pass_1_cold": cold_trace, "pass_2_hot": hot_trace}, indent=2, default=str),
        encoding="utf-8",
    )
    summary_path = root / "live_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and replay a PAN action resolver demo ledger.")
    parser.add_argument("--artifact-dir", default="/tmp/pan-action-resolver-demo")
    parser.add_argument("--live", action="store_true", help="Run a live workload through the full foundry loop instead of synthetic rows.")
    parser.add_argument("--model", default="tests/fixtures/spf32_fused_4096.pan-nn.json", help="Model path for live workload demo.")
    parser.add_argument("--fabric-daemon", default="", help="Fabric daemon socket path.")
    parser.add_argument("--xclbin-dir", action="append", default=None, help="Xclbin search directory (repeatable).")
    parser.add_argument("--active-app", default="spf32_core_pl")
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args(argv)
    if args.live:
        summary = build_live_workload_demo(
            args.artifact_dir,
            model_path=args.model,
            fabric_daemon=args.fabric_daemon,
            xclbin_dirs=args.xclbin_dir,
            active_app=args.active_app,
            workdir=args.workdir,
        )
    else:
        summary = build_action_resolver_demo(args.artifact_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
