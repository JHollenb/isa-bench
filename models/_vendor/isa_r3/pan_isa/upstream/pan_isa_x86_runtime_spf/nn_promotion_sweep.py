from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from .fabric_promotion_evidence import append_jsonl, build_promotion_evidence, write_json


NN_SWEEP_SCHEMA = "pan-nn-promotion-sweep-v1"
NN_PROMOTION_SCOPE_CONTRACT_SCHEMA = "pan-nn-promotion-scope-contract-v1"
CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def run_nn_promotion_sweep(
    *,
    model_paths: Sequence[str | Path],
    artifact_dir: str | Path,
    input_paths: Sequence[str | Path | None] | None = None,
    fabric_daemon: str = "/tmp/pan-fabricd.sock",
    workdir: str = "/home/ubuntu/spfr",
    xclbin_dirs: Sequence[str] = (),
    active_app: str = "spf32_core_pl",
    fabric_backend: str = "pl",
    daemon_buffer_storage: str = "inline",
    foundry_catalog: str | None = None,
    foundry_auto_queue: bool = False,
    repeats: int = 1,
    promotion_warmup_runs: int = 1,
    resident_candidate_measurement: bool = False,
    resident_inputs: bool = False,
    feedback_outputs: Sequence[str] = (),
    required_neural_output_residency: str | None = None,
    source_commit: str | None = None,
    execute: bool = True,
    command_runner: CommandRunner | None = None,
    min_samples: int = 1,
    max_latency_ratio: float = 1.0,
    max_transfer_ratio: float = 1.0,
    max_energy_ratio: float = 1.0,
    min_compile_priority_precision: float = 0.80,
    confidence: float = 0.85,
    validation_sample_count: int | None = None,
) -> dict[str, Any]:
    if not model_paths:
        raise ValueError("at least one NN model path is required")
    promotion_warmup_runs = int(promotion_warmup_runs)
    repeats = int(repeats)
    if promotion_warmup_runs < 0:
        raise ValueError("promotion_warmup_runs must be non-negative")
    feedback_outputs = _normalize_feedback_outputs(feedback_outputs)
    state_feedback = bool(feedback_outputs)
    if (resident_inputs or state_feedback) and repeats > 1 and promotion_warmup_runs >= repeats:
        raise ValueError("promotion_warmup_runs must be smaller than repeats for steady-state promotion windows")
    inputs = _normalize_input_paths(model_paths, input_paths)
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    runner = command_runner or _run_command
    residency_goal = required_neural_output_residency
    if resident_candidate_measurement and not residency_goal:
        residency_goal = "daemon"
    scope_contract = _promotion_scope_contract(
        repeats=repeats,
        promotion_warmup_runs=promotion_warmup_runs,
        resident_inputs=resident_inputs,
        state_feedback=state_feedback,
        resident_candidate_measurement=resident_candidate_measurement,
    )

    samples = []
    commands = []
    for index, (model_path, input_path) in enumerate(zip(model_paths, inputs)):
        sample = _sample_paths(root, index, Path(model_path))
        rule_cmd = _nn_run_command(
            model_path=Path(model_path),
            input_path=None if input_path is None else Path(input_path),
            backend="ps",
            output_json=sample["rule_json"],
            trace_jsonl=sample["rule_trace"],
            training_jsonl=sample["rule_training"],
            fabric_daemon=fabric_daemon,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            fabric_backend=fabric_backend,
            daemon_buffer_storage=daemon_buffer_storage,
            foundry_catalog=foundry_catalog,
            foundry_auto_queue=False,
            repeats=repeats,
            promotion_warmup_runs=promotion_warmup_runs,
            resident_inputs=resident_inputs,
            feedback_outputs=feedback_outputs,
            validate_output=True,
            no_output_digest=False,
            validation_sample_count=validation_sample_count,
        )
        candidate_validation_cmd = None
        if resident_candidate_measurement:
            candidate_validation_cmd = _nn_run_command(
                model_path=Path(model_path),
                input_path=None if input_path is None else Path(input_path),
                backend="fabric-session",
                output_json=sample["candidate_validation_json"],
                trace_jsonl=sample["candidate_validation_trace"],
                training_jsonl=sample["candidate_validation_training"],
                fabric_daemon=fabric_daemon,
                workdir=workdir,
                xclbin_dirs=xclbin_dirs,
                active_app=active_app,
                fabric_backend=fabric_backend,
                daemon_buffer_storage=daemon_buffer_storage,
                foundry_catalog=foundry_catalog,
                foundry_auto_queue=foundry_auto_queue,
                repeats=repeats,
                promotion_warmup_runs=promotion_warmup_runs,
                resident_inputs=resident_inputs,
                feedback_outputs=feedback_outputs,
                validate_output=True,
                no_output_digest=False,
                validation_sample_count=validation_sample_count,
            )
        candidate_cmd = _nn_run_command(
            model_path=Path(model_path),
            input_path=None if input_path is None else Path(input_path),
            backend="fabric-session",
            output_json=sample["candidate_json"],
            trace_jsonl=sample["candidate_trace"],
            training_jsonl=sample["candidate_training"],
            fabric_daemon=fabric_daemon,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            fabric_backend=fabric_backend,
            daemon_buffer_storage=daemon_buffer_storage,
            foundry_catalog=foundry_catalog,
            foundry_auto_queue=foundry_auto_queue,
            repeats=repeats,
            promotion_warmup_runs=promotion_warmup_runs,
            resident_inputs=resident_inputs,
            feedback_outputs=feedback_outputs,
            validate_output=not bool(resident_candidate_measurement),
            no_output_digest=bool(resident_candidate_measurement),
            validation_sample_count=validation_sample_count,
        )
        commands.append({"sample": sample["sample_id"], "kind": "rule", "argv": rule_cmd})
        if candidate_validation_cmd is not None:
            commands.append({"sample": sample["sample_id"], "kind": "candidate_validation", "argv": candidate_validation_cmd})
        commands.append({"sample": sample["sample_id"], "kind": "candidate", "argv": candidate_cmd})
        if not execute:
            samples.append({**sample, "model": str(model_path), "inputs": None if input_path is None else str(input_path), "executed": False, "ok": None})
            continue
        results = [_run_and_record(runner, rule_cmd, sample["rule_stdout"], sample["rule_stderr"])]
        if candidate_validation_cmd is not None:
            results.append(
                _run_and_record(
                    runner,
                    candidate_validation_cmd,
                    sample["candidate_validation_stdout"],
                    sample["candidate_validation_stderr"],
                )
            )
        results.append(_run_and_record(runner, candidate_cmd, sample["candidate_stdout"], sample["candidate_stderr"]))
        samples.append(
            {
                **sample,
                "model": str(model_path),
                "inputs": None if input_path is None else str(input_path),
                "executed": True,
                "ok": all(result["returncode"] == 0 for result in results),
                "results": results,
            }
        )

    executed_ok = [sample for sample in samples if sample.get("executed") and sample.get("ok")]
    train_result = None
    evidence = None
    promotion_policy_path = root / "candidate_policy.json"
    training_report_path = root / "candidate_policy_training.json"
    if execute and executed_ok:
        train_cmd = _nn_train_policy_command(
            [sample["candidate_training"] for sample in executed_ok],
            policy_out=promotion_policy_path,
            report_out=training_report_path,
        )
        commands.append({"sample": "heldout", "kind": "candidate_policy_training", "argv": train_cmd})
        train_result = _run_and_record(runner, train_cmd, root / "candidate_policy_training.stdout", root / "candidate_policy_training.stderr")
        if train_result["returncode"] == 0:
            evidence = build_promotion_evidence(
                [sample["rule_json"] for sample in executed_ok],
                [sample["candidate_json"] for sample in executed_ok],
                neural_validation_execution_paths=(
                    [sample["candidate_validation_json"] for sample in executed_ok]
                    if resident_candidate_measurement
                    else None
                ),
                required_neural_output_residency=residency_goal,
                policy_path=promotion_policy_path,
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

    executed_samples = [sample for sample in samples if sample.get("executed")]
    manifest_ok = bool(
        (not execute)
        or (
            executed_ok
            and len(executed_ok) == len(executed_samples) == len(samples)
            and train_result is not None
            and train_result["returncode"] == 0
            and evidence is not None
        )
    )
    manifest = {
        "schema": NN_SWEEP_SCHEMA,
        "ok": manifest_ok,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "artifact_dir": str(root),
        "models": [str(path) for path in model_paths],
        "inputs": [None if path is None else str(path) for path in inputs],
        "repeats": int(repeats),
        "promotion_warmup_runs": int(promotion_warmup_runs),
        "promotion_scope": scope_contract["scope"],
        "promotion_scope_contract": scope_contract,
        "execute": bool(execute),
        "fabric_daemon": fabric_daemon,
        "workdir": workdir,
        "xclbin_dirs": list(xclbin_dirs),
        "active_app": active_app,
        "fabric_backend": fabric_backend,
        "daemon_buffer_storage": daemon_buffer_storage,
        "foundry_catalog": foundry_catalog,
        "foundry_auto_queue": bool(foundry_auto_queue),
        "validation_sample_count": validation_sample_count,
        "resident_candidate_measurement": bool(resident_candidate_measurement),
        "resident_inputs": bool(resident_inputs),
        "state_feedback": state_feedback,
        "feedback_outputs": list(feedback_outputs),
        "required_neural_output_residency": residency_goal,
        "promotion_thresholds": {
            "min_samples": int(min_samples),
            "max_latency_ratio": float(max_latency_ratio),
            "max_transfer_ratio": float(max_transfer_ratio),
            "max_energy_ratio": float(max_energy_ratio),
            "min_compile_priority_precision": float(min_compile_priority_precision),
            "confidence": float(confidence),
        },
        "safety": {
            "rule_backend": "ps",
            "candidate_backend": "fabric-session",
            "validate_outputs": True,
            "candidate_measurement_validates_outputs": not bool(resident_candidate_measurement),
            "separate_candidate_validation": bool(resident_candidate_measurement),
            "candidate_measurement_keeps_output_resident": bool(resident_candidate_measurement),
            "resident_inputs": bool(resident_inputs),
            "state_feedback": state_feedback,
            "feedback_outputs": list(feedback_outputs),
            "promotion_warmup_runs": int(promotion_warmup_runs),
            "promotion_authority_before_gate": False,
            "requires_existing_fabric_daemon_for_candidate": True,
        },
        "commands": commands,
        "samples": [_jsonable_sample(sample) for sample in samples],
        "candidate_policy_training": train_result,
        "candidate_policy_path": str(promotion_policy_path) if train_result and train_result["returncode"] == 0 else None,
        "candidate_policy_training_report": str(training_report_path) if train_result and train_result["returncode"] == 0 else None,
        "evidence_path": str(root / "heldout_evidence.json") if evidence is not None else None,
        "metrics_path": str(root / "heldout_metrics.json") if evidence is not None else None,
        "promotion_allowed": bool(evidence and evidence["promotion_allowed"]),
        "promotion_evaluation": None if evidence is None else evidence["promotion_evaluation"],
        "negative_record_path": str(root / "neural_negative.jsonl") if evidence and evidence["negative_record"] else None,
    }
    write_json(manifest, root / "nn_promotion_sweep_manifest.json")
    return manifest


def _normalize_input_paths(
    model_paths: Sequence[str | Path],
    input_paths: Sequence[str | Path | None] | None,
) -> list[str | Path | None]:
    if not input_paths:
        return [None for _ in model_paths]
    if len(input_paths) != len(model_paths):
        raise ValueError("--inputs must be omitted or repeated once per --model")
    return list(input_paths)


def _normalize_feedback_outputs(items: Sequence[str]) -> list[str]:
    feedback: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError("feedback_outputs entries must use INPUT=OUTPUT")
        input_name, output_name = (part.strip() for part in str(item).split("=", 1))
        if not input_name or not output_name:
            raise ValueError("feedback_outputs entries must use non-empty INPUT=OUTPUT names")
        if input_name in feedback and feedback[input_name] != output_name:
            raise ValueError(f"duplicate feedback input {input_name!r}")
        feedback[input_name] = output_name
    return [f"{name}={output}" for name, output in sorted(feedback.items())]


def _promotion_scope_contract(
    *,
    repeats: int,
    promotion_warmup_runs: int,
    resident_inputs: bool,
    state_feedback: bool,
    resident_candidate_measurement: bool,
) -> dict[str, Any]:
    requested_repeats = int(repeats)
    steady_state = bool((resident_inputs or state_feedback) and requested_repeats > 1)
    warmup_runs = int(promotion_warmup_runs) if steady_state else 0
    promotion_runs = max(0, requested_repeats - warmup_runs) if steady_state else requested_repeats
    if state_feedback and requested_repeats > 1:
        scope = "hot_cache_state_feedback" if warmup_runs > 1 else "steady_state_state_feedback"
    elif resident_inputs and requested_repeats > 1:
        scope = "hot_cache_resident_inputs" if warmup_runs > 1 else "steady_state_resident_inputs"
    elif requested_repeats > 1:
        scope = "repeated_cold_start"
    else:
        scope = "cold_start_single_run"
    return {
        "schema": NN_PROMOTION_SCOPE_CONTRACT_SCHEMA,
        "scope": scope,
        "resident_inputs": bool(resident_inputs),
        "state_feedback": bool(state_feedback),
        "requested_repeats": requested_repeats,
        "warmup_runs_excluded": warmup_runs,
        "promotion_window_start_run_index": warmup_runs,
        "promotion_window_completed_runs": promotion_runs,
        "separate_candidate_validation": bool(resident_candidate_measurement),
        "candidate_measurement_validates_outputs": not bool(resident_candidate_measurement),
    }


def _nn_run_command(
    *,
    model_path: Path,
    input_path: Path | None,
    backend: str,
    output_json: Path,
    trace_jsonl: Path,
    training_jsonl: Path,
    fabric_daemon: str,
    workdir: str,
    xclbin_dirs: Sequence[str],
    active_app: str,
    fabric_backend: str,
    daemon_buffer_storage: str,
    foundry_catalog: str | None,
    foundry_auto_queue: bool,
    repeats: int,
    promotion_warmup_runs: int,
    resident_inputs: bool,
    feedback_outputs: Sequence[str],
    validate_output: bool,
    no_output_digest: bool,
    validation_sample_count: int | None,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "pan_isa_x86_runtime_spf",
        "nn-run",
        "--model",
        str(model_path),
        "--backend",
        backend,
        "--fabric-daemon",
        fabric_daemon,
        "--workdir",
        workdir,
        "--active-app",
        active_app,
        "--fabric-backend",
        fabric_backend,
        "--daemon-buffer-storage",
        daemon_buffer_storage,
        "--repeats",
        str(int(repeats)),
        "--promotion-warmup-runs",
        str(int(promotion_warmup_runs)),
        "--include-trace",
        "--trace-jsonl",
        str(trace_jsonl),
        "--training-jsonl",
        str(training_jsonl),
        "--write",
        str(output_json),
    ]
    if validate_output:
        argv.append("--validate-output")
    if no_output_digest:
        argv.append("--no-output-digest")
    if resident_inputs:
        argv.append("--resident-inputs")
    for feedback_output in feedback_outputs:
        argv.extend(["--feedback-output", str(feedback_output)])
    if input_path is not None:
        argv.extend(["--inputs", str(input_path)])
    for xclbin_dir in xclbin_dirs:
        argv.extend(["--xclbin-dir", str(xclbin_dir)])
    if foundry_catalog is not None:
        argv.extend(["--foundry-catalog", str(foundry_catalog)])
    if foundry_auto_queue:
        argv.append("--foundry-auto-queue")
    if validate_output and validation_sample_count is not None:
        argv.extend(["--validation-sample-count", str(int(validation_sample_count))])
    return argv


def _nn_train_policy_command(
    training_paths: Sequence[Path],
    *,
    policy_out: Path,
    report_out: Path,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "pan_isa_x86_runtime_spf",
        "nn-train-policy",
        "--policy-out",
        str(policy_out),
        "--write",
        str(report_out),
        "--trace-jsonl",
    ]
    argv.extend(str(path) for path in training_paths)
    return argv


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


def _sample_paths(root: Path, index: int, model_path: Path) -> dict[str, Any]:
    sample_id = f"{int(index):03d}_{_safe_name(model_path.stem)}"
    sample_root = root / sample_id
    sample_root.mkdir(parents=True, exist_ok=True)
    return {
        "sample_id": sample_id,
        "sample_dir": str(sample_root),
        "rule_json": sample_root / "rule.json",
        "rule_trace": sample_root / "rule_trace.jsonl",
        "rule_training": sample_root / "rule_training.jsonl",
        "rule_stdout": sample_root / "rule.stdout",
        "rule_stderr": sample_root / "rule.stderr",
        "candidate_json": sample_root / "candidate.json",
        "candidate_trace": sample_root / "candidate_trace.jsonl",
        "candidate_training": sample_root / "candidate_training.jsonl",
        "candidate_stdout": sample_root / "candidate.stdout",
        "candidate_stderr": sample_root / "candidate.stderr",
        "candidate_validation_json": sample_root / "candidate_validation.json",
        "candidate_validation_trace": sample_root / "candidate_validation_trace.jsonl",
        "candidate_validation_training": sample_root / "candidate_validation_training.jsonl",
        "candidate_validation_stdout": sample_root / "candidate_validation.stdout",
        "candidate_validation_stderr": sample_root / "candidate_validation.stderr",
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
    parser = argparse.ArgumentParser(description="Run paired PAN NN rule/candidate promotion evidence sweeps.")
    parser.add_argument("--model", action="append", required=True, help="PAN NN JSON or ONNX model; repeat for held-out samples.")
    parser.add_argument("--inputs", action="append", help="Input JSON matching each --model; omit to use seeded inputs.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--fabric-daemon", default="/tmp/pan-fabricd.sock")
    parser.add_argument("--workdir", default="/home/ubuntu/spfr")
    parser.add_argument("--xclbin-dir", action="append", default=[])
    parser.add_argument("--active-app", default="spf32_core_pl")
    parser.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="pl")
    parser.add_argument("--daemon-buffer-storage", choices=("inline", "fd"), default="inline")
    parser.add_argument("--foundry-catalog")
    parser.add_argument("--foundry-auto-queue", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--promotion-warmup-runs", type=int, default=1)
    parser.add_argument("--resident-candidate-measurement", action="store_true", help="Measure the fabric candidate with --no-output-digest and require a separate validated candidate proof.")
    parser.add_argument("--resident-inputs", action="store_true", help="Retain identical graph inputs across repeated nn-run sessions.")
    parser.add_argument("--feedback-output", action="append", default=[], help="Retain a produced output as the next repeated run's input, formatted INPUT=OUTPUT.")
    parser.add_argument("--required-neural-output-residency", help="Residency the measured candidate outputs must satisfy; defaults to daemon with --resident-candidate-measurement.")
    parser.add_argument("--source-commit")
    parser.add_argument("--dry-run", action="store_true", help="Plan commands without executing NN runs.")
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-latency-ratio", type=float, default=1.0)
    parser.add_argument("--max-transfer-ratio", type=float, default=1.0)
    parser.add_argument("--max-energy-ratio", type=float, default=1.0)
    parser.add_argument("--min-compile-priority-precision", type=float, default=0.80)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--validation-sample-count", type=int)
    parser.add_argument("--write", help="Write sweep manifest JSON; stdout when omitted.")
    args = parser.parse_args(argv)
    manifest = run_nn_promotion_sweep(
        model_paths=args.model,
        input_paths=args.inputs,
        artifact_dir=args.artifact_dir,
        fabric_daemon=args.fabric_daemon,
        workdir=args.workdir,
        xclbin_dirs=args.xclbin_dir,
        active_app=args.active_app,
        fabric_backend=args.fabric_backend,
        daemon_buffer_storage=args.daemon_buffer_storage,
        foundry_catalog=args.foundry_catalog,
        foundry_auto_queue=bool(args.foundry_auto_queue),
        repeats=args.repeats,
        promotion_warmup_runs=args.promotion_warmup_runs,
        resident_candidate_measurement=bool(args.resident_candidate_measurement),
        resident_inputs=bool(args.resident_inputs),
        feedback_outputs=args.feedback_output,
        required_neural_output_residency=args.required_neural_output_residency,
        source_commit=args.source_commit,
        execute=not bool(args.dry_run),
        min_samples=args.min_samples,
        max_latency_ratio=args.max_latency_ratio,
        max_transfer_ratio=args.max_transfer_ratio,
        max_energy_ratio=args.max_energy_ratio,
        min_compile_priority_precision=args.min_compile_priority_precision,
        confidence=args.confidence,
        validation_sample_count=args.validation_sample_count,
    )
    write_json(manifest, args.write)
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
