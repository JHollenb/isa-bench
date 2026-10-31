from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .fabric_promotion_evidence import load_json, write_json
from .nn_promotion_policy_contract import (
    DEFAULT_REQUIRED_SCOPES,
    evaluate_nn_promotion_policy_contract,
)
from .nn_promotion_scope_matrix import (
    NN_SCOPE_MATRIX_SCHEMA,
    build_nn_promotion_scope_matrix,
)
from .nn_promotion_sweep import CommandRunner


NN_PROMOTION_PRODUCTION_BUNDLE_SCHEMA = "pan-nn-promotion-production-bundle-v1"
NN_PROMOTION_PRODUCTION_BUNDLE_SPEC_SCHEMA = "pan-nn-promotion-production-bundle-spec-v1"


def load_production_bundle_spec(path: str | Path) -> list[dict[str, Any]]:
    spec_path = Path(path)
    payload = load_json(spec_path)
    if payload.get("schema") not in {None, NN_PROMOTION_PRODUCTION_BUNDLE_SPEC_SCHEMA}:
        raise ValueError(f"unsupported production bundle spec schema {payload.get('schema')!r}")
    families = payload.get("families")
    if not isinstance(families, list) or not families:
        raise ValueError("production bundle spec requires a non-empty families list")
    return [_normalize_family_spec(item, base_dir=spec_path.parent) for item in families]


def build_nn_promotion_production_bundle(
    *,
    family_specs: Sequence[dict[str, Any]],
    artifact_dir: str | Path,
    default_scopes: Sequence[str] = DEFAULT_REQUIRED_SCOPES,
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
    min_passed_sweeps_per_scope: int = 1,
    min_passed_sweeps_per_graph_signature: int = 1,
    min_passed_sweeps_per_graph_family: int = 1,
    min_distinct_graph_signatures: int = 1,
    min_distinct_graph_families: int | None = None,
    execute: bool = False,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    if not family_specs:
        raise ValueError("at least one graph-family spec is required")
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)

    normalized_specs = [_normalize_family_spec(spec) for spec in family_specs]
    family_names = [spec["family"] for spec in normalized_specs]
    if len(set(family_names)) != len(family_names):
        raise ValueError("graph-family specs must use unique family names")

    required_scopes = _ordered_unique(scope for spec in normalized_specs for scope in spec["scopes"])
    if not required_scopes:
        required_scopes = list(default_scopes)
    required_graph_families = _ordered_unique(
        family for spec in normalized_specs for family in spec["required_graph_families"]
    )
    required_graph_signatures = _ordered_unique(
        signature for spec in normalized_specs for signature in spec["required_graph_signatures"]
    )
    min_family_count = int(min_distinct_graph_families) if min_distinct_graph_families is not None else max(1, len(required_graph_families))

    families: list[dict[str, Any]] = []
    family_scope_pairs: list[dict[str, Any]] = []
    sweep_manifest_paths: list[str] = []
    executed_matrix_paths: list[str] = []

    for spec in normalized_specs:
        family_dir = root / _safe_name(spec["family"])
        matrix = build_nn_promotion_scope_matrix(
            model_paths=spec["model_paths"],
            input_paths=spec["input_paths"],
            artifact_dir=family_dir,
            scopes=spec["scopes"],
            feedback_outputs=spec["feedback_outputs"],
            fabric_daemon=fabric_daemon,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            fabric_backend=fabric_backend,
            daemon_buffer_storage=daemon_buffer_storage,
            foundry_catalog=foundry_catalog,
            foundry_auto_queue=foundry_auto_queue,
            resident_candidate_measurement=resident_candidate_measurement,
            required_neural_output_residency=spec["required_neural_output_residency"] or required_neural_output_residency,
            source_commit=source_commit,
            validation_sample_count=(
                spec["validation_sample_count"]
                if spec["validation_sample_count"] is not None
                else validation_sample_count
            ),
            min_samples=min_samples,
            max_latency_ratio=max_latency_ratio,
            max_transfer_ratio=max_transfer_ratio,
            max_energy_ratio=max_energy_ratio,
            min_compile_priority_precision=min_compile_priority_precision,
            confidence=confidence,
            execute=execute,
            command_runner=command_runner,
        )
        matrix_path = family_dir / "nn_promotion_scope_matrix.json"
        executed_matrix_paths.append(str(matrix_path))
        entries = list(matrix["entries"])
        for entry in entries:
            sweep_manifest_paths.append(str(entry["manifest_path"]))
            status = _family_scope_status(entry, execute=execute)
            family_scope_pairs.append(
                {
                    "family": spec["family"],
                    "scope": entry["scope"],
                    "status": status,
                    "sweep_ok": bool(entry["sweep_ok"]),
                    "promotion_allowed": bool(entry["promotion_allowed"]),
                    "manifest_path": entry["manifest_path"],
                }
            )
        families.append(
            {
                "family": spec["family"],
                "required_graph_families": spec["required_graph_families"],
                "required_graph_signatures": spec["required_graph_signatures"],
                "scopes": list(spec["scopes"]),
                "model_count": len(spec["model_paths"]),
                "models": [str(path) for path in spec["model_paths"]],
                "inputs": [None if path is None else str(path) for path in spec["input_paths"]],
                "feedback_outputs": list(spec["feedback_outputs"]),
                "required_neural_output_residency": spec["required_neural_output_residency"] or required_neural_output_residency,
                "validation_sample_count": (
                    spec["validation_sample_count"]
                    if spec["validation_sample_count"] is not None
                    else validation_sample_count
                ),
                "matrix_path": str(matrix_path),
                "matrix_ok": bool(matrix["ok"]),
                "entries": entries,
            }
        )

    failed_family_scope_pairs = [
        {"family": pair["family"], "scope": pair["scope"], "status": pair["status"]}
        for pair in family_scope_pairs
        if execute and pair["status"] != "passed"
    ]
    scope_plan_path = _write_contract_scope_plan(root, required_scopes)
    contract_path = root / "production_policy_contract.json"
    contract_summary: dict[str, Any] | None = None
    if execute:
        contract = evaluate_nn_promotion_policy_contract(
            sweep_manifest_paths=sweep_manifest_paths,
            scope_matrix_paths=[scope_plan_path],
            required_scopes=required_scopes,
            required_graph_signatures=required_graph_signatures,
            required_graph_families=required_graph_families,
            min_passed_sweeps_per_scope=min_passed_sweeps_per_scope,
            min_passed_sweeps_per_graph_signature=min_passed_sweeps_per_graph_signature,
            min_passed_sweeps_per_graph_family=min_passed_sweeps_per_graph_family,
            min_distinct_graph_signatures=min_distinct_graph_signatures,
            min_distinct_graph_families=min_family_count,
            source_commit=source_commit,
        )
        write_json(contract, contract_path)
        contract_summary = {
            "production_authority_allowed": bool(contract["production_authority_allowed"]),
            "reason": contract["reason"],
            "failed_checks": list(contract["failed_checks"]),
            "missing_executed_scopes": list(contract["missing_executed_scopes"]),
            "missing_graph_signatures": list(contract["missing_graph_signatures"]),
            "missing_graph_families": list(contract["missing_graph_families"]),
            "passed_graph_signatures": list(contract["passed_graph_signatures"]),
            "passed_graph_families": list(contract["passed_graph_families"]),
        }
    contract_allowed = bool(contract_summary and contract_summary["production_authority_allowed"])
    production_allowed = bool(execute and contract_allowed and not failed_family_scope_pairs)
    manifest = {
        "schema": NN_PROMOTION_PRODUCTION_BUNDLE_SCHEMA,
        "ok": bool((not execute) or all(family["matrix_ok"] for family in families)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "artifact_dir": str(root),
        "execute": bool(execute),
        "production_authority_allowed": production_allowed,
        "reason": _bundle_reason(execute, contract_allowed, failed_family_scope_pairs),
        "required_scopes": required_scopes,
        "required_graph_families": required_graph_families,
        "required_graph_signatures": required_graph_signatures,
        "min_distinct_graph_signatures": int(min_distinct_graph_signatures),
        "min_distinct_graph_families": min_family_count,
        "promotion_thresholds": {
            "min_samples": int(min_samples),
            "max_latency_ratio": float(max_latency_ratio),
            "max_transfer_ratio": float(max_transfer_ratio),
            "max_energy_ratio": float(max_energy_ratio),
            "min_compile_priority_precision": float(min_compile_priority_precision),
            "confidence": float(confidence),
        },
        "families": families,
        "family_scope_pairs": family_scope_pairs,
        "failed_family_scope_pairs": failed_family_scope_pairs,
        "executed_matrix_paths": executed_matrix_paths,
        "contract_scope_plan_path": str(scope_plan_path),
        "contract_path": str(contract_path) if execute else None,
        "contract_summary": contract_summary,
    }
    write_json(manifest, root / "nn_promotion_production_bundle.json")
    return manifest


def _normalize_family_spec(spec: dict[str, Any], *, base_dir: Path | None = None) -> dict[str, Any]:
    if not isinstance(spec, dict):
        raise ValueError("family spec entries must be objects")
    family = str(spec.get("family") or spec.get("graph_family") or "").strip()
    if not family:
        raise ValueError("family spec requires family")
    if "model_paths" in spec:
        model_paths = [str(path) for path in (spec.get("model_paths") or ())]
        input_paths = [None if path is None else str(path) for path in (spec.get("input_paths") or ())]
        if len(input_paths) != len(model_paths):
            raise ValueError(f"family {family!r} input_paths must match model_paths")
    else:
        model_entries = spec.get("models")
        if not isinstance(model_entries, list) or not model_entries:
            raise ValueError(f"family {family!r} requires a non-empty models list")
        model_paths = []
        input_paths = []
        for entry in model_entries:
            model, inputs = _normalize_model_entry(entry, base_dir=base_dir)
            model_paths.append(model)
            input_paths.append(inputs)
    if not model_paths:
        raise ValueError(f"family {family!r} requires a non-empty models list")
    scopes = _ordered_unique(str(scope) for scope in (spec.get("scopes") or DEFAULT_REQUIRED_SCOPES))
    if not scopes:
        raise ValueError(f"family {family!r} requires at least one scope")
    required_graph_families = _ordered_unique(
        str(item) for item in (spec.get("required_graph_families") or [family]) if str(item)
    )
    required_graph_signatures = _ordered_unique(
        str(item) for item in (spec.get("required_graph_signatures") or []) if str(item)
    )
    validation_sample_count = spec.get("validation_sample_count")
    if validation_sample_count is not None:
        validation_sample_count = int(validation_sample_count)
    return {
        "family": family,
        "model_paths": model_paths,
        "input_paths": input_paths,
        "scopes": scopes,
        "feedback_outputs": tuple(str(item) for item in (spec.get("feedback_outputs") or ())),
        "required_graph_families": required_graph_families,
        "required_graph_signatures": required_graph_signatures,
        "required_neural_output_residency": spec.get("required_neural_output_residency"),
        "validation_sample_count": validation_sample_count,
    }


def _normalize_model_entry(entry: Any, *, base_dir: Path | None) -> tuple[str, str | None]:
    if isinstance(entry, str):
        return _resolve_path(entry, base_dir), None
    if not isinstance(entry, dict):
        raise ValueError("model entries must be strings or objects")
    model = entry.get("model")
    if not model:
        raise ValueError("model entry requires model")
    inputs = entry.get("inputs")
    return _resolve_path(str(model), base_dir), None if inputs is None else _resolve_path(str(inputs), base_dir)


def _resolve_path(value: str, base_dir: Path | None) -> str:
    path = Path(value)
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return str(path)


def _write_contract_scope_plan(root: Path, scopes: Sequence[str]) -> Path:
    path = root / "production_scope_contract_plan.json"
    payload = {
        "schema": NN_SCOPE_MATRIX_SCHEMA,
        "ok": True,
        "execute": False,
        "entries": [{"scope": scope, "command_count": 0} for scope in scopes],
    }
    write_json(payload, path)
    return path


def _family_scope_status(entry: dict[str, Any], *, execute: bool) -> str:
    if not execute:
        return "planned"
    if entry.get("sweep_ok") and entry.get("promotion_allowed"):
        return "passed"
    if not entry.get("sweep_ok"):
        return "failed_sweep"
    return "blocked_promotion"


def _bundle_reason(execute: bool, contract_allowed: bool, failed_pairs: Sequence[dict[str, Any]]) -> str:
    if not execute:
        return "dry_run_plan_only"
    if failed_pairs:
        return "family_scope_cross_product_failed"
    if not contract_allowed:
        return "production_promotion_contract_failed"
    return "production_promotion_contract_passed"


def _ordered_unique(items: Sequence[str] | Any) -> list[str]:
    seen: set[str] = set()
    values: list[str] = []
    for item in items:
        value = str(item)
        if not value or value in seen:
            continue
        seen.add(value)
        values.append(value)
    return values


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_") or "family"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package PAN NN production-promotion evidence across graph families and scopes.")
    parser.add_argument("--spec", required=True, help="Production bundle JSON spec.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--fabric-daemon", default="/tmp/pan-fabricd.sock")
    parser.add_argument("--workdir", default="/home/ubuntu/spfr")
    parser.add_argument("--xclbin-dir", action="append", default=[])
    parser.add_argument("--active-app", default="spf32_core_pl")
    parser.add_argument("--fabric-backend", choices=("auto", "ps", "pl"), default="pl")
    parser.add_argument("--daemon-buffer-storage", choices=("inline", "fd"), default="inline")
    parser.add_argument("--foundry-catalog")
    parser.add_argument("--foundry-auto-queue", action="store_true")
    parser.add_argument("--materialized-candidate-measurement", dest="resident_candidate_measurement", action="store_false")
    parser.add_argument("--required-neural-output-residency")
    parser.add_argument("--source-commit")
    parser.add_argument("--validation-sample-count", type=int)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-latency-ratio", type=float, default=1.0)
    parser.add_argument("--max-transfer-ratio", type=float, default=1.0)
    parser.add_argument("--max-energy-ratio", type=float, default=1.0)
    parser.add_argument("--min-compile-priority-precision", type=float, default=0.80)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--min-passed-sweeps-per-scope", type=int, default=1)
    parser.add_argument("--min-passed-sweeps-per-graph-signature", type=int, default=1)
    parser.add_argument("--min-passed-sweeps-per-graph-family", type=int, default=1)
    parser.add_argument("--min-distinct-graph-signatures", type=int, default=1)
    parser.add_argument("--min-distinct-graph-families", type=int)
    parser.add_argument("--execute", action="store_true", help="Execute all family/scope sweeps; default writes command plans.")
    parser.add_argument("--write", help="Write an additional copy of the bundle manifest.")
    parser.set_defaults(resident_candidate_measurement=True)
    args = parser.parse_args(argv)
    manifest = build_nn_promotion_production_bundle(
        family_specs=load_production_bundle_spec(args.spec),
        artifact_dir=args.artifact_dir,
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
        min_passed_sweeps_per_scope=args.min_passed_sweeps_per_scope,
        min_passed_sweeps_per_graph_signature=args.min_passed_sweeps_per_graph_signature,
        min_passed_sweeps_per_graph_family=args.min_passed_sweeps_per_graph_family,
        min_distinct_graph_signatures=args.min_distinct_graph_signatures,
        min_distinct_graph_families=args.min_distinct_graph_families,
        execute=bool(args.execute),
    )
    if args.write:
        write_json(manifest, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
