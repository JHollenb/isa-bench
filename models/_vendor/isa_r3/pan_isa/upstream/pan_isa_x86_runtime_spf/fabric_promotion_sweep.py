from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .fabric_graph import FabricPolicyHead
from .fabric_oracle_preflight import run_oracle_preflight
from .fabric_promotion_evidence import append_jsonl, build_promotion_evidence, write_json


SWEEP_SCHEMA = "pan-fabric-promotion-sweep-v1"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]
PreflightRunner = Callable[..., dict[str, Any]]


def run_promotion_sweep(
    *,
    ops: Sequence[str],
    elements: Sequence[int],
    artifact_dir: str | Path,
    workdir: str = "/home/ubuntu/spfr",
    xclbin_dirs: Sequence[str] = (),
    active_app: str = "spf32_core_pl",
    repeats: int = 1,
    source_commit: str | None = None,
    execute: bool = True,
    command_runner: CommandRunner | None = None,
    preflight_runner: PreflightRunner | None = None,
    min_samples: int = 1,
    max_latency_ratio: float = 1.0,
    max_transfer_ratio: float = 1.0,
    max_energy_ratio: float = 1.0,
    min_compile_priority_precision: float = 0.80,
    confidence: float = 0.85,
) -> dict[str, Any]:
    if not ops:
        raise ValueError("at least one op is required")
    if not elements:
        raise ValueError("at least one element count is required")
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    runner = command_runner or _run_command
    preflight = preflight_runner or run_oracle_preflight
    policy = _candidate_policy(confidence=confidence)
    policy_path = root / "candidate_policy.json"
    write_json(policy.to_json(), policy_path)

    preflight_reports = []
    runnable_pairs: list[tuple[str, int]] = []
    for count in sorted({int(value) for value in elements}):
        report = preflight(list(ops), elements=count, workdir=workdir)
        preflight_path = root / f"oracle_preflight_{count}.json"
        write_json(report, preflight_path)
        preflight_reports.append({"path": str(preflight_path), "report": report})
        ok_ops = {str(row["op"]) for row in report.get("ops", []) if row.get("ok")}
        runnable_pairs.extend((op, count) for op in ops if op in ok_ops)

    samples = []
    commands = []
    for op, count in runnable_pairs:
        sample = _sample_paths(root, op, count)
        rule_cmd = _fabric_exec_command(
            op=op,
            elements=count,
            output_json=sample["rule_json"],
            action_ledger=sample["rule_ledger"],
            policy_path=None,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            repeats=repeats,
            fabric_backend="ps",
            execute_backend="sim",
            input_residency="ps",
            output_residency="ps",
            no_output_digest=False,
        )
        neural_cmd = _fabric_exec_command(
            op=op,
            elements=count,
            output_json=sample["neural_json"],
            action_ledger=sample["neural_ledger"],
            policy_path=policy_path,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            repeats=repeats,
            fabric_backend="pl",
            execute_backend="xrt",
            input_residency="pl",
            output_residency="pl",
            no_output_digest=True,
        )
        rule_replay_cmd = _replay_command(sample["rule_ledger"], sample["rule_replay"])
        neural_replay_cmd = _replay_command(sample["neural_ledger"], sample["neural_replay"])
        commands.extend(
            [
                {"sample": sample["sample_id"], "kind": "rule", "argv": rule_cmd},
                {"sample": sample["sample_id"], "kind": "candidate", "argv": neural_cmd},
                {"sample": sample["sample_id"], "kind": "rule_replay", "argv": rule_replay_cmd},
                {"sample": sample["sample_id"], "kind": "candidate_replay", "argv": neural_replay_cmd},
            ]
        )
        if not execute:
            samples.append({**sample, "executed": False, "ok": None})
            continue
        results = [
            _run_and_record(runner, rule_cmd, sample["rule_stdout"], sample["rule_stderr"]),
            _run_and_record(runner, neural_cmd, sample["neural_stdout"], sample["neural_stderr"]),
            _run_and_record(runner, rule_replay_cmd, sample["rule_replay_stdout"], sample["rule_replay_stderr"]),
            _run_and_record(runner, neural_replay_cmd, sample["neural_replay_stdout"], sample["neural_replay_stderr"]),
        ]
        samples.append(
            {
                **sample,
                "executed": True,
                "ok": all(result["returncode"] == 0 for result in results),
                "results": results,
            }
        )

    executed_ok = [sample for sample in samples if sample.get("executed") and sample.get("ok")]
    evidence = None
    promotion_policy = None
    if execute and executed_ok:
        evidence = build_promotion_evidence(
            [sample["rule_json"] for sample in executed_ok],
            [sample["neural_json"] for sample in executed_ok],
            policy_path=policy_path,
            confidence=confidence,
            min_samples=min_samples,
            max_latency_ratio=max_latency_ratio,
            max_transfer_ratio=max_transfer_ratio,
            max_energy_ratio=max_energy_ratio,
            min_compile_priority_precision=min_compile_priority_precision,
        )
        write_json(evidence["metrics"], root / "heldout_metrics.json")
        write_json(evidence, root / "heldout_evidence.json")
        if evidence["negative_record"] is not None:
            append_jsonl([evidence["negative_record"]], root / "neural_negative.jsonl")
        promotion_policy = policy.with_promotion_metrics(
            evidence["metrics"],
            min_samples=min_samples,
            max_latency_ratio=max_latency_ratio,
            max_transfer_ratio=max_transfer_ratio,
            max_energy_ratio=max_energy_ratio,
            min_compile_priority_precision=min_compile_priority_precision,
        )
        write_json(promotion_policy.to_json(), root / "promotion_policy.json")

    manifest = {
        "schema": SWEEP_SCHEMA,
        "ok": bool((not execute) or (executed_ok and all(sample.get("ok") for sample in samples if sample.get("executed")))),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "artifact_dir": str(root),
        "ops": list(ops),
        "elements": [int(value) for value in elements],
        "workdir": workdir,
        "xclbin_dirs": list(xclbin_dirs),
        "active_app": active_app,
        "repeats": int(repeats),
        "execute": bool(execute),
        "promotion_thresholds": {
            "min_samples": int(min_samples),
            "max_latency_ratio": float(max_latency_ratio),
            "max_transfer_ratio": float(max_transfer_ratio),
            "max_energy_ratio": float(max_energy_ratio),
            "min_compile_priority_precision": float(min_compile_priority_precision),
            "confidence": float(confidence),
        },
        "safety": {
            "requires_oracle_preflight": True,
            "rule_backend": "ps/sim",
            "candidate_backend": "pl/xrt",
            "loads_pl_app": False,
            "promotion_authority_before_gate": False,
        },
        "preflight_reports": preflight_reports,
        "commands": commands,
        "samples": [_jsonable_sample(sample) for sample in samples],
        "evidence_path": str(root / "heldout_evidence.json") if evidence is not None else None,
        "promotion_allowed": bool(evidence and evidence["promotion_allowed"]),
        "promotion_evaluation": None if evidence is None else evidence["promotion_evaluation"],
        "negative_record_path": str(root / "neural_negative.jsonl") if evidence and evidence["negative_record"] else None,
        "promotion_policy_path": str(root / "promotion_policy.json") if promotion_policy is not None else None,
    }
    write_json(manifest, root / "promotion_sweep_manifest.json")
    return manifest


def _candidate_policy(*, confidence: float) -> FabricPolicyHead:
    base = FabricPolicyHead.default()
    return replace(
        base,
        policy_role="candidate",
        promotion_status="candidate",
        can_override_rules=False,
        min_confidence=float(confidence),
    )


def _fabric_exec_command(
    *,
    op: str,
    elements: int,
    output_json: Path,
    action_ledger: Path,
    policy_path: Path | None,
    workdir: str,
    xclbin_dirs: Sequence[str],
    active_app: str,
    repeats: int,
    fabric_backend: str,
    execute_backend: str,
    input_residency: str,
    output_residency: str,
    no_output_digest: bool,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "pan_isa_x86_runtime_spf",
        "fabric-exec",
        "--linear-ops",
        op,
        "--elements",
        str(int(elements)),
        "--fabric-backend",
        fabric_backend,
        "--execute-backend",
        execute_backend,
        "--input-residency",
        input_residency,
        "--output-residency",
        output_residency,
        "--active-app",
        active_app,
        "--workdir",
        workdir,
        "--repeats",
        str(int(repeats)),
        "--write-action-ledger-jsonl",
        str(action_ledger),
        "--write",
        str(output_json),
    ]
    for xclbin_dir in xclbin_dirs:
        argv.extend(["--xclbin-dir", str(xclbin_dir)])
    if policy_path is not None:
        argv.extend(["--policy", str(policy_path)])
    if no_output_digest:
        argv.append("--no-output-digest")
    return argv


def _replay_command(ledger_path: Path, replay_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "pan_hardware_resolver.replay",
        str(ledger_path),
        "--write",
        str(replay_path),
    ]


def _run_and_record(runner: CommandRunner, argv: Sequence[str], stdout_path: Path, stderr_path: Path) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    result = runner(list(argv))
    stdout_path.write_text(result.stdout or "", encoding="utf-8")
    stderr_path.write_text(result.stderr or "", encoding="utf-8")
    return {
        "argv": list(argv),
        "returncode": int(result.returncode),
        "stdout": str(result.stdout or ""),
        "stderr": str(result.stderr or ""),
    }


def _run_command(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)


def _sample_paths(root: Path, op: str, elements: int) -> dict[str, Any]:
    sample_id = f"{_safe_name(op)}_{int(elements)}"
    sample_root = root / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    return {
        "sample_id": sample_id,
        "op": op,
        "elements": int(elements),
        "sample_dir": str(sample_root),
        "rule_json": sample_root / "rule.json",
        "rule_ledger": sample_root / "rule_action_ledger.jsonl",
        "rule_replay": sample_root / "rule_replay.json",
        "rule_stdout": sample_root / "rule.stdout",
        "rule_stderr": sample_root / "rule.stderr",
        "rule_replay_stdout": sample_root / "rule_replay.stdout",
        "rule_replay_stderr": sample_root / "rule_replay.stderr",
        "neural_json": sample_root / "candidate.json",
        "neural_ledger": sample_root / "candidate_action_ledger.jsonl",
        "neural_replay": sample_root / "candidate_replay.json",
        "neural_stdout": sample_root / "candidate.stdout",
        "neural_stderr": sample_root / "candidate.stderr",
        "neural_replay_stdout": sample_root / "candidate_replay.stdout",
        "neural_replay_stderr": sample_root / "candidate_replay.stderr",
    }


def _jsonable_sample(sample: dict[str, Any]) -> dict[str, Any]:
    payload = {}
    for key, value in sample.items():
        if isinstance(value, Path):
            payload[key] = str(value)
        else:
            payload[key] = value
    return payload


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "sample"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a rule-preflighted PAN fabric promotion sweep.")
    parser.add_argument("--op", action="append", required=True, help="Typed SPF op to sweep; may be repeated.")
    parser.add_argument("--elements", action="append", type=int, help="Element count to sweep; may be repeated.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--workdir", default="/home/ubuntu/spfr")
    parser.add_argument("--xclbin-dir", action="append", default=[])
    parser.add_argument("--active-app", default="spf32_core_pl")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--source-commit")
    parser.add_argument("--dry-run", action="store_true", help="Plan commands and preflight without executing fabric runs.")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-latency-ratio", type=float, default=1.0)
    parser.add_argument("--max-transfer-ratio", type=float, default=1.0)
    parser.add_argument("--max-energy-ratio", type=float, default=1.0)
    parser.add_argument("--min-compile-priority-precision", type=float, default=0.80)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--write", help="Write sweep manifest JSON; stdout when omitted.")
    args = parser.parse_args(argv)
    manifest = run_promotion_sweep(
        ops=args.op,
        elements=args.elements or [4096],
        artifact_dir=args.artifact_dir,
        workdir=args.workdir,
        xclbin_dirs=args.xclbin_dir,
        active_app=args.active_app,
        repeats=args.repeats,
        source_commit=args.source_commit,
        execute=not bool(args.dry_run),
        min_samples=args.min_samples,
        max_latency_ratio=args.max_latency_ratio,
        max_transfer_ratio=args.max_transfer_ratio,
        max_energy_ratio=args.max_energy_ratio,
        min_compile_priority_precision=args.min_compile_priority_precision,
        confidence=args.confidence,
    )
    write_json(manifest, args.write)
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
