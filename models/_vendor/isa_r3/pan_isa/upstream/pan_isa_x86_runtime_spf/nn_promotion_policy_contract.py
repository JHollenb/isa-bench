from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .fabric_promotion_evidence import load_json, write_json
from .nn_promotion_scope_matrix import NN_SCOPE_MATRIX_SCHEMA
from .nn_promotion_sweep import NN_SWEEP_SCHEMA


NN_PROMOTION_POLICY_CONTRACT_SCHEMA = "pan-nn-promotion-policy-contract-v1"
DEFAULT_REQUIRED_SCOPES = (
    "cold_start_single_run",
    "steady_state_resident_inputs",
    "hot_cache_resident_inputs",
)
DEFAULT_REQUIRED_GRAPH_SIGNATURES: tuple[str, ...] = ()
DEFAULT_REQUIRED_GRAPH_FAMILIES: tuple[str, ...] = ()
REQUIRED_METRICS = (
    "sample_count",
    "latency_ratio_vs_rule",
    "transfer_bytes_ratio_vs_rule",
    "validation_failures",
    "output_residency_regressions",
    "materialization_regressions",
    "compile_priority_precision",
    "confidence",
)
REQUIRED_GATE_CHECKS = (
    "sample_count_sufficient",
    "latency_beats_rule_scheduler",
    "transfer_bytes_not_worse",
    "energy_not_worse",
    "validation_failures_zero",
    "output_residency_not_worse",
    "materialization_regressions_zero",
    "compile_priority_precision_checked",
    "confidence_passed",
)


def evaluate_nn_promotion_policy_contract(
    *,
    sweep_manifest_paths: Sequence[str | Path],
    scope_matrix_paths: Sequence[str | Path] = (),
    required_scopes: Sequence[str] = DEFAULT_REQUIRED_SCOPES,
    required_graph_signatures: Sequence[str] = DEFAULT_REQUIRED_GRAPH_SIGNATURES,
    required_graph_families: Sequence[str] = DEFAULT_REQUIRED_GRAPH_FAMILIES,
    require_scope_matrix: bool = True,
    min_passed_sweeps_per_scope: int = 1,
    min_passed_sweeps_per_graph_signature: int = 1,
    min_passed_sweeps_per_graph_family: int = 1,
    min_distinct_graph_signatures: int = 1,
    min_distinct_graph_families: int = 1,
    source_commit: str | None = None,
) -> dict[str, Any]:
    if not required_scopes:
        raise ValueError("at least one required promotion scope is required")
    if int(min_passed_sweeps_per_scope) < 1:
        raise ValueError("min_passed_sweeps_per_scope must be positive")
    if int(min_passed_sweeps_per_graph_signature) < 1:
        raise ValueError("min_passed_sweeps_per_graph_signature must be positive")
    if int(min_passed_sweeps_per_graph_family) < 1:
        raise ValueError("min_passed_sweeps_per_graph_family must be positive")
    if int(min_distinct_graph_signatures) < 1:
        raise ValueError("min_distinct_graph_signatures must be positive")
    if int(min_distinct_graph_families) < 1:
        raise ValueError("min_distinct_graph_families must be positive")
    required_scope_list = [str(scope) for scope in required_scopes]
    required_graph_signature_list = [str(signature) for signature in required_graph_signatures if str(signature)]
    required_graph_family_list = [str(family) for family in required_graph_families if str(family)]
    sweep_reports = [_load_sweep_report(path) for path in sweep_manifest_paths]
    matrix_reports = [_load_matrix_report(path) for path in scope_matrix_paths]
    passed_sweeps_by_scope = {
        scope: [
            report
            for report in sweep_reports
            if report["scope"] == scope and report["contract_ready"]
        ]
        for scope in required_scope_list
    }
    missing_executed_scopes = [
        scope
        for scope, reports in passed_sweeps_by_scope.items()
        if len(reports) < int(min_passed_sweeps_per_scope)
    ]
    passed_graph_signatures = sorted(
        {
            signature
            for report in sweep_reports
            if report["contract_ready"]
            for signature in report["graph_signatures"]
            if signature and signature != "unknown"
        }
    )
    passed_sweeps_by_graph_signature = {
        signature: [
            report
            for report in sweep_reports
            if report["contract_ready"] and signature in set(report["graph_signatures"])
        ]
        for signature in required_graph_signature_list
    }
    missing_graph_signatures = [
        signature
        for signature, reports in passed_sweeps_by_graph_signature.items()
        if len(reports) < int(min_passed_sweeps_per_graph_signature)
    ]
    passed_graph_families = sorted(
        {
            family
            for report in sweep_reports
            if report["contract_ready"]
            for family in report["graph_families"]
            if family and family != "unknown"
        }
    )
    passed_sweeps_by_graph_family = {
        family: [
            report
            for report in sweep_reports
            if report["contract_ready"] and family in set(report["graph_families"])
        ]
        for family in required_graph_family_list
    }
    missing_graph_families = [
        family
        for family, reports in passed_sweeps_by_graph_family.items()
        if len(reports) < int(min_passed_sweeps_per_graph_family)
    ]
    matrix_scopes = sorted({scope for report in matrix_reports for scope in report["scopes"]})
    missing_matrix_scopes = [
        scope for scope in required_scope_list if scope not in set(matrix_scopes)
    ]
    checks = {
        "required_scopes_have_executed_passing_sweeps": not missing_executed_scopes,
        "required_graph_signatures_have_executed_passing_sweeps": not missing_graph_signatures,
        "required_graph_families_have_executed_passing_sweeps": not missing_graph_families,
        "distinct_graph_signature_threshold_met": len(passed_graph_signatures) >= int(min_distinct_graph_signatures),
        "distinct_graph_family_threshold_met": len(passed_graph_families) >= int(min_distinct_graph_families),
        "all_sweeps_ok": bool(sweep_reports) and all(report["ok"] for report in sweep_reports),
        "all_sweeps_executed": bool(sweep_reports) and all(report["execute"] for report in sweep_reports),
        "all_sweeps_promotion_allowed": bool(sweep_reports)
        and all(report["promotion_allowed"] for report in sweep_reports),
        "all_promotion_gate_checks_pass": bool(sweep_reports)
        and all(report["promotion_gate_checks_pass"] for report in sweep_reports),
        "no_policy_authority_before_gate": bool(sweep_reports)
        and all(not report["policy_authoritative_before_promotion"] for report in sweep_reports),
        "validation_risk_heads_zero": bool(sweep_reports)
        and all(report["validation_risk_passed"] for report in sweep_reports),
        "required_metrics_present": bool(sweep_reports)
        and all(not report["missing_metrics"] for report in sweep_reports),
        "rule_oracle_preflight_declared": bool(sweep_reports)
        and all(report["rule_oracle_preflight_declared"] for report in sweep_reports),
        "residency_contract_declared": bool(sweep_reports)
        and all(report["residency_contract_declared"] for report in sweep_reports),
        "no_negative_records": bool(sweep_reports)
        and all(not report["negative_record_path"] for report in sweep_reports),
        "scope_matrix_manifests_ok": (
            (not require_scope_matrix)
            or (
                bool(matrix_reports)
                and all(report["schema_ok"] and report["ok"] and not report["execute"] for report in matrix_reports)
            )
        ),
        "scope_matrix_reports_required_scopes": (
            (not require_scope_matrix) or (bool(matrix_reports) and not missing_matrix_scopes)
        ),
    }
    allowed = all(checks.values())
    failed_checks = sorted(name for name, ok in checks.items() if not ok)
    manifest = {
        "schema": NN_PROMOTION_POLICY_CONTRACT_SCHEMA,
        "ok": True,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_commit": source_commit,
        "production_authority_allowed": allowed,
        "reason": "production_promotion_contract_passed" if allowed else "production_promotion_contract_failed",
        "required_scopes": required_scope_list,
        "required_graph_signatures": required_graph_signature_list,
        "required_graph_families": required_graph_family_list,
        "min_passed_sweeps_per_scope": int(min_passed_sweeps_per_scope),
        "min_passed_sweeps_per_graph_signature": int(min_passed_sweeps_per_graph_signature),
        "min_passed_sweeps_per_graph_family": int(min_passed_sweeps_per_graph_family),
        "min_distinct_graph_signatures": int(min_distinct_graph_signatures),
        "min_distinct_graph_families": int(min_distinct_graph_families),
        "require_scope_matrix": bool(require_scope_matrix),
        "checks": checks,
        "failed_checks": failed_checks,
        "missing_executed_scopes": missing_executed_scopes,
        "missing_graph_signatures": missing_graph_signatures,
        "missing_graph_families": missing_graph_families,
        "passed_graph_signatures": passed_graph_signatures,
        "passed_graph_families": passed_graph_families,
        "missing_matrix_scopes": missing_matrix_scopes,
        "matrix_scopes": matrix_scopes,
        "sweeps": sweep_reports,
        "scope_matrices": matrix_reports,
    }
    return manifest


def _load_sweep_report(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = load_json(manifest_path)
    schema_ok = payload.get("schema") == NN_SWEEP_SCHEMA
    evidence = _load_optional_json(manifest_path, payload.get("evidence_path"))
    policy = _load_optional_json(manifest_path, payload.get("candidate_policy_path"))
    metrics = _metrics_from_manifest(payload, evidence)
    graph_signatures = _graph_signatures_from_metrics(metrics)
    graph_families = _graph_families_from_metrics(metrics)
    evaluation = payload.get("promotion_evaluation") or {}
    gate_checks = dict(evaluation.get("checks") or {})
    missing_gate_checks = [name for name in REQUIRED_GATE_CHECKS if gate_checks.get(name) is not True]
    missing_metrics = [name for name in REQUIRED_METRICS if name not in metrics]
    validation_risk = _validation_risk_report(policy)
    rule_preflight = _rule_oracle_preflight_declared(payload)
    residency_declared = _residency_contract_declared(payload, metrics)
    ready = bool(
        schema_ok
        and payload.get("ok") is True
        and payload.get("execute") is True
        and payload.get("promotion_allowed") is True
        and not missing_gate_checks
        and not bool(evidence.get("policy_authoritative_before_promotion"))
        and validation_risk["passed"]
        and not missing_metrics
        and rule_preflight
        and residency_declared
        and not payload.get("negative_record_path")
    )
    return {
        "manifest_path": str(manifest_path),
        "schema_ok": schema_ok,
        "ok": bool(payload.get("ok")),
        "execute": bool(payload.get("execute")),
        "scope": str(payload.get("promotion_scope") or ""),
        "promotion_scope_contract": dict(payload.get("promotion_scope_contract") or {}),
        "promotion_allowed": bool(payload.get("promotion_allowed")),
        "promotion_gate_checks_pass": not missing_gate_checks,
        "missing_gate_checks": missing_gate_checks,
        "metrics": _metrics_summary(metrics),
        "graph_signatures": graph_signatures,
        "graph_families": graph_families,
        "missing_metrics": missing_metrics,
        "policy_path": payload.get("candidate_policy_path"),
        "policy_authoritative_before_promotion": bool(evidence.get("policy_authoritative_before_promotion")),
        "validation_risk": validation_risk,
        "validation_risk_passed": bool(validation_risk["passed"]),
        "rule_oracle_preflight_declared": rule_preflight,
        "residency_contract_declared": residency_declared,
        "required_neural_output_residency": payload.get("required_neural_output_residency"),
        "negative_record_path": payload.get("negative_record_path"),
        "contract_ready": ready,
    }


def _load_matrix_report(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    payload = load_json(manifest_path)
    entries = [entry for entry in payload.get("entries") or [] if isinstance(entry, dict)]
    return {
        "manifest_path": str(manifest_path),
        "schema_ok": payload.get("schema") == NN_SCOPE_MATRIX_SCHEMA,
        "ok": bool(payload.get("ok")),
        "execute": bool(payload.get("execute")),
        "scopes": [str(entry.get("scope") or "") for entry in entries],
        "entry_count": len(entries),
    }


def _load_optional_json(manifest_path: Path, path: Any) -> dict[str, Any]:
    if not path:
        return {}
    candidate = Path(str(path))
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    if not candidate.is_file():
        return {}
    return load_json(candidate)


def _metrics_from_manifest(payload: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    if isinstance(evidence.get("metrics"), dict):
        return dict(evidence["metrics"])
    evaluation = payload.get("promotion_evaluation") or {}
    if isinstance(evaluation.get("metrics"), dict):
        return dict(evaluation["metrics"])
    return {}


def _graph_signatures_from_metrics(metrics: dict[str, Any]) -> list[str]:
    signatures: set[str] = set()
    for split in metrics.get("splits") or ():
        if not isinstance(split, dict):
            continue
        signature = str(split.get("graph_signature") or "")
        if signature:
            signatures.add(signature)
    return sorted(signatures)


def _graph_families_from_metrics(metrics: dict[str, Any]) -> list[str]:
    families: set[str] = set()
    for split in metrics.get("splits") or ():
        if not isinstance(split, dict):
            continue
        family = str(split.get("graph_family") or "")
        if not family:
            family = _graph_family_from_signature(str(split.get("graph_signature") or ""))
        if family:
            families.add(family)
    return sorted(families)


def _graph_family_from_signature(graph_signature: str) -> str:
    ops = [op for op in str(graph_signature).split("+") if op]
    if not ops or ops == ["unknown"]:
        return "unknown"
    if all(op.startswith("spf32.") for op in ops):
        return "spf32_fused" if len(ops) > 1 else "spf32_single"
    if any(op in {"conv2d", "depthwise_conv2d", "batchnorm"} for op in ops):
        return "cnn_region"
    if any(op in {"kv_cache_append", "top_p_filter", "topk"} for op in ops):
        return "cache_decode"
    if any(op in {"attention_qkv", "attention_score", "attention_value", "rotary_embedding", "rmsnorm"} for op in ops):
        return "transformer_attention"
    if any(op in {"matmul", "gemm", "qlinear_matmul", "matmul_integer"} for op in ops):
        return "dense_linear"
    return "generic_nn"


def _validation_risk_report(policy: dict[str, Any]) -> dict[str, Any]:
    heads = dict(policy.get("heads") or {})
    risk = heads.get("validation_risk")
    if not isinstance(risk, dict):
        return {"present": False, "risk_rate": None, "failure_count": None, "passed": False}
    risk_rate = _float_or_none(risk.get("risk_rate"))
    failure_count = _int_or_none(risk.get("failure_count"))
    passed = risk_rate == 0.0 and (failure_count is None or failure_count == 0)
    return {
        "present": True,
        "risk_rate": risk_rate,
        "failure_count": failure_count,
        "passed": bool(passed),
    }


def _rule_oracle_preflight_declared(payload: dict[str, Any]) -> bool:
    safety = dict(payload.get("safety") or {})
    if safety.get("rule_backend") != "ps" or safety.get("validate_outputs") is not True:
        return False
    commands = [command for command in payload.get("commands") or [] if isinstance(command, dict)]
    for command in commands:
        if command.get("kind") != "rule":
            continue
        argv = [str(item) for item in command.get("argv") or []]
        if "--backend" not in argv or "--validate-output" not in argv:
            continue
        backend_index = argv.index("--backend") + 1
        if backend_index < len(argv) and argv[backend_index] == "ps":
            return True
    return False


def _residency_contract_declared(payload: dict[str, Any], metrics: dict[str, Any]) -> bool:
    if payload.get("required_neural_output_residency") in {None, ""}:
        return False
    return int(metrics.get("output_residency_regressions") or 0) == 0


def _metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {name: metrics.get(name) for name in REQUIRED_METRICS if name in metrics}


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate PAN NN production promotion policy contracts.")
    parser.add_argument("--sweep-manifest", action="append", default=[], help="Executed nn_promotion_sweep manifest.")
    parser.add_argument("--scope-matrix", action="append", default=[], help="nn_promotion_scope_matrix manifest.")
    parser.add_argument("--required-scope", action="append", default=[], help="Scope required for production authority.")
    parser.add_argument("--required-graph-signature", action="append", default=[], help="Graph signature required for production authority.")
    parser.add_argument("--required-graph-family", action="append", default=[], help="Graph family required for production authority.")
    parser.add_argument("--no-require-scope-matrix", dest="require_scope_matrix", action="store_false")
    parser.add_argument("--min-passed-sweeps-per-scope", type=int, default=1)
    parser.add_argument("--min-passed-sweeps-per-graph-signature", type=int, default=1)
    parser.add_argument("--min-passed-sweeps-per-graph-family", type=int, default=1)
    parser.add_argument("--min-distinct-graph-signatures", type=int, default=1)
    parser.add_argument("--min-distinct-graph-families", type=int, default=1)
    parser.add_argument("--source-commit")
    parser.add_argument("--write", help="Write contract JSON; stdout when omitted.")
    parser.set_defaults(require_scope_matrix=True)
    args = parser.parse_args(argv)
    manifest = evaluate_nn_promotion_policy_contract(
        sweep_manifest_paths=args.sweep_manifest,
        scope_matrix_paths=args.scope_matrix,
        required_scopes=tuple(args.required_scope) if args.required_scope else DEFAULT_REQUIRED_SCOPES,
        required_graph_signatures=tuple(args.required_graph_signature),
        required_graph_families=tuple(args.required_graph_family),
        require_scope_matrix=bool(args.require_scope_matrix),
        min_passed_sweeps_per_scope=args.min_passed_sweeps_per_scope,
        min_passed_sweeps_per_graph_signature=args.min_passed_sweeps_per_graph_signature,
        min_passed_sweeps_per_graph_family=args.min_passed_sweeps_per_graph_family,
        min_distinct_graph_signatures=args.min_distinct_graph_signatures,
        min_distinct_graph_families=args.min_distinct_graph_families,
        source_commit=args.source_commit,
    )
    write_json(manifest, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
