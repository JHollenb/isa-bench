from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .fabric_promotion_evidence import write_json
from .nn_promotion_sweep import CommandRunner, run_nn_promotion_sweep


NN_SCOPE_MATRIX_SCHEMA = "pan-nn-promotion-scope-matrix-v1"
DEFAULT_SCOPES = (
    "cold_start_single_run",
    "steady_state_resident_inputs",
    "hot_cache_resident_inputs",
)


def build_nn_promotion_scope_matrix(
    *,
    model_paths: Sequence[str | Path],
    artifact_dir: str | Path,
    input_paths: Sequence[str | Path | None] | None = None,
    scopes: Sequence[str] = DEFAULT_SCOPES,
    feedback_outputs: Sequence[str] = (),
    fabric_daemon: str = "/tmp/pan-fabricd.sock",
    workdir: str = "/home/ubuntu/spfr",
    xclbin_dirs: Sequence[str] = (),
    active_app: str = "spf32_core_pl",
    fabric_backend: str = "pl",
    daemon_buffer_storage: str = "inline",
    foundry_catalog: str | None = None,
    foundry_auto_queue: bool = False,
    resident_candidate_measurement: bool = True,
    required_neural_output_residency: str | None = None,
    source_commit: str | None = None,
    validation_sample_count: int | None = None,
    min_samples: int = 1,
    max_latency_ratio: float = 1.0,
    max_transfer_ratio: float = 1.0,
    max_energy_ratio: float = 1.0,
    min_compile_priority_precision: float = 0.80,
    confidence: float = 0.85,
    execute: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if not scopes:
        raise ValueError("at least one promotion scope is required")
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries = []
    seen_scopes: set[str] = set()
    for scope in scopes:
        normalized_scope = str(scope)
        if normalized_scope in seen_scopes:
            raise ValueError(f"duplicate promotion scope {normalized_scope!r}")
        seen_scopes.add(normalized_scope)
        config = _scope_config(normalized_scope, feedback_outputs=feedback_outputs)
        scope_dir = root / _safe_name(normalized_scope)
        sweep = run_nn_promotion_sweep(
            model_paths=model_paths,
            input_paths=input_paths,
            artifact_dir=scope_dir,
            fabric_daemon=fabric_daemon,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            fabric_backend=fabric_backend,
            daemon_buffer_storage=daemon_buffer_storage,
            foundry_catalog=foundry_catalog,
            foundry_auto_queue=foundry_auto_queue,
            repeats=config["repeats"],
            promotion_warmup_runs=config["promotion_warmup_runs"],
            resident_candidate_measurement=resident_candidate_measurement,
            resident_inputs=config["resident_inputs"],
            feedback_outputs=config["feedback_outputs"],
            required_neural_output_residency=required_neural_output_residency,
            source_commit=source_commit,
            execute=bool(execute),
            command_runner=command_runner,
            min_samples=min_samples,
            max_latency_ratio=max_latency_ratio,
            max_transfer_ratio=max_transfer_ratio,
            max_energy_ratio=max_energy_ratio,
            min_compile_priority_precision=min_compile_priority_precision,
            confidence=confidence,
            validation_sample_count=validation_sample_count,
        )
        entries.append(
            {
                "scope": normalized_scope,
                "artifact_dir": str(scope_dir),
                "manifest_path": str(scope_dir / "nn_promotion_sweep_manifest.json"),
                "promotion_scope": sweep["promotion_scope"],
                "promotion_scope_contract": sweep["promotion_scope_contract"],
                "execute": bool(sweep["execute"]),
                "sweep_ok": bool(sweep["ok"]),
                "promotion_allowed": bool(sweep["promotion_allowed"]),
                "evidence_path": sweep["evidence_path"],
                "metrics_path": sweep["metrics_path"],
                "candidate_policy_path": sweep["candidate_policy_path"],
                "negative_record_path": sweep["negative_record_path"],
                "command_count": len(sweep["commands"]),
                "commands": sweep["commands"],
                "sample_count": len(sweep["samples"]),
            }
        )
    manifest = {
        "schema": NN_SCOPE_MATRIX_SCHEMA,
        "ok": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "artifact_dir": str(root),
        "execute": bool(execute),
        "models": [str(path) for path in model_paths],
        "inputs": None if input_paths is None else [None if path is None else str(path) for path in input_paths],
        "scopes": [entry["scope"] for entry in entries],
        "resident_candidate_measurement": bool(resident_candidate_measurement),
        "required_neural_output_residency": required_neural_output_residency,
        "validation_sample_count": validation_sample_count,
        "promotion_thresholds": {
            "min_samples": int(min_samples),
            "max_latency_ratio": float(max_latency_ratio),
            "max_transfer_ratio": float(max_transfer_ratio),
            "max_energy_ratio": float(max_energy_ratio),
            "min_compile_priority_precision": float(min_compile_priority_precision),
            "confidence": float(confidence),
        },
        "entries": entries,
    }
    manifest["ok"] = all(bool(entry["sweep_ok"]) for entry in entries) if execute else True
    write_json(manifest, root / "nn_promotion_scope_matrix.json")
    return manifest


def _scope_config(scope: str, *, feedback_outputs: Sequence[str]) -> dict[str, Any]:
    if scope == "cold_start_single_run":
        return {
            "repeats": 1,
            "promotion_warmup_runs": 0,
            "resident_inputs": False,
            "feedback_outputs": (),
        }
    if scope == "repeated_cold_start":
        return {
            "repeats": 4,
            "promotion_warmup_runs": 0,
            "resident_inputs": False,
            "feedback_outputs": (),
        }
    if scope == "steady_state_resident_inputs":
        return {
            "repeats": 3,
            "promotion_warmup_runs": 1,
            "resident_inputs": True,
            "feedback_outputs": (),
        }
    if scope == "hot_cache_resident_inputs":
        return {
            "repeats": 4,
            "promotion_warmup_runs": 2,
            "resident_inputs": True,
            "feedback_outputs": (),
        }
    if scope in {"steady_state_state_feedback", "hot_cache_state_feedback"}:
        if not feedback_outputs:
            raise ValueError(f"{scope} requires at least one feedback output mapping")
        return {
            "repeats": 3 if scope == "steady_state_state_feedback" else 4,
            "promotion_warmup_runs": 1 if scope == "steady_state_state_feedback" else 2,
            "resident_inputs": False,
            "feedback_outputs": tuple(feedback_outputs),
        }
    raise ValueError(f"unknown promotion scope {scope!r}")


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "scope"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write side-by-side NN promotion sweep manifests for several measurement scopes.")
    parser.add_argument("--model", action="append", required=True, help="PAN NN JSON or ONNX model; repeat for held-out samples.")
    parser.add_argument("--inputs", action="append", help="Input JSON matching each --model; omit to use seeded inputs.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--scope", action="append", default=[], help="Promotion scope to include; defaults to cold, steady resident-input, and hot-cache resident-input scopes.")
    parser.add_argument("--feedback-output", action="append", default=[], help="State-feedback mapping for decode scopes, formatted INPUT=OUTPUT.")
    parser.add_argument("--fabric-daemon", default="/tmp/pan-fabricd.sock")
    parser.add_argument("--workdir", default="/home/ubuntu/spfr")
    parser.add_argument("--xclbin-dir", action="append", default=[])
    parser.add_argument("--active-app", default="spf32_core_pl")
    parser.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="pl")
    parser.add_argument("--daemon-buffer-storage", choices=("inline", "fd"), default="inline")
    parser.add_argument("--foundry-catalog")
    parser.add_argument("--foundry-auto-queue", action="store_true")
    parser.add_argument("--materialized-candidate-measurement", dest="resident_candidate_measurement", action="store_false", help="Plan candidate measurement with validation/materialization instead of separate resident measurement.")
    parser.add_argument("--required-neural-output-residency")
    parser.add_argument("--source-commit")
    parser.add_argument("--validation-sample-count", type=int)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-latency-ratio", type=float, default=1.0)
    parser.add_argument("--max-transfer-ratio", type=float, default=1.0)
    parser.add_argument("--max-energy-ratio", type=float, default=1.0)
    parser.add_argument("--min-compile-priority-precision", type=float, default=0.80)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--execute", action="store_true", help="Execute each per-scope sweep; default writes dry-run command plans.")
    parser.add_argument("--write", help="Write matrix manifest JSON; stdout when omitted.")
    parser.set_defaults(resident_candidate_measurement=True)
    args = parser.parse_args(argv)
    manifest = build_nn_promotion_scope_matrix(
        model_paths=args.model,
        input_paths=args.inputs,
        artifact_dir=args.artifact_dir,
        scopes=tuple(args.scope) if args.scope else DEFAULT_SCOPES,
        feedback_outputs=args.feedback_output,
        fabric_daemon=args.fabric_daemon,
        workdir=args.workdir,
        xclbin_dirs=args.xclbin_dir,
        active_app=args.active_app,
        fabric_backend=args.fabric_backend,
        daemon_buffer_storage=args.daemon_buffer_storage,
        foundry_catalog=args.foundry_catalog,
        foundry_auto_queue=bool(args.foundry_auto_queue),
        resident_candidate_measurement=bool(args.resident_candidate_measurement),
        required_neural_output_residency=args.required_neural_output_residency,
        source_commit=args.source_commit,
        validation_sample_count=args.validation_sample_count,
        min_samples=args.min_samples,
        max_latency_ratio=args.max_latency_ratio,
        max_transfer_ratio=args.max_transfer_ratio,
        max_energy_ratio=args.max_energy_ratio,
        min_compile_priority_precision=args.min_compile_priority_precision,
        confidence=args.confidence,
        execute=bool(args.execute),
    )
    write_json(manifest, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
