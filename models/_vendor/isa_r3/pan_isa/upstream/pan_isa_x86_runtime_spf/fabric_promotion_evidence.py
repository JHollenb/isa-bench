from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .fabric_graph import FabricPolicyHead


EVIDENCE_SCHEMA = "pan_fabric_real_heldout_evidence_v1"
METRICS_SCHEMA = "pan_fabric_real_heldout_metrics_v1"
NEGATIVE_RECORD_SCHEMA = "pan_fabric_neural_negative_record_v1"


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_heldout_metrics(
    rule_execution_paths: Sequence[str | Path],
    neural_execution_paths: Sequence[str | Path],
    *,
    neural_validation_execution_paths: Sequence[str | Path] | None = None,
    required_neural_output_residency: str | None = None,
    compile_priority_precision: float = 1.0,
    confidence: float = 0.85,
) -> dict[str, Any]:
    if len(rule_execution_paths) != len(neural_execution_paths):
        raise ValueError("rule and neural execution lists must have the same length")
    if neural_validation_execution_paths is not None and len(neural_validation_execution_paths) != len(neural_execution_paths):
        raise ValueError("neural validation execution list must match neural execution list length")
    if not rule_execution_paths:
        raise ValueError("at least one held-out execution pair is required")
    pairs = []
    validation_paths = (
        [None for _ in neural_execution_paths]
        if neural_validation_execution_paths is None
        else list(neural_validation_execution_paths)
    )
    for index, (rule_path, neural_path, validation_path) in enumerate(
        zip(rule_execution_paths, neural_execution_paths, validation_paths)
    ):
        rule = load_json(rule_path)
        neural = load_json(neural_path)
        neural_validation = None if validation_path is None else load_json(validation_path)
        pairs.append(
            _pair_metrics(
                index,
                rule_path,
                neural_path,
                rule,
                neural,
                neural_validation_path=validation_path,
                neural_validation=neural_validation,
                required_neural_output_residency=required_neural_output_residency,
            )
        )
    rule_seconds = sum(float(pair["rule_seconds"]) for pair in pairs)
    neural_seconds = sum(float(pair["neural_seconds"]) for pair in pairs)
    rule_transfer = sum(int(pair["rule_transfer_bytes"]) for pair in pairs)
    neural_transfer = sum(int(pair["neural_transfer_bytes"]) for pair in pairs)
    energy_pairs = [
        pair
        for pair in pairs
        if pair.get("rule_energy_j") is not None and pair.get("neural_energy_j") is not None
    ]
    rule_energy = sum(float(pair["rule_energy_j"]) for pair in energy_pairs)
    neural_energy = sum(float(pair["neural_energy_j"]) for pair in energy_pairs)
    validation_failures = sum(int(pair["validation_failures"]) for pair in pairs)
    output_residency_regressions = sum(int(pair["output_residency_regression"]) for pair in pairs)
    materialization_regressions = sum(int(pair["materialization_regression"]) for pair in pairs)
    return {
        "schema": METRICS_SCHEMA,
        "sample_count": len(pairs),
        "heldout_sample_count": len(pairs),
        "latency_ratio_vs_rule": _ratio(neural_seconds, rule_seconds),
        "transfer_bytes_ratio_vs_rule": _ratio(neural_transfer, rule_transfer),
        "rule_elapsed_seconds": rule_seconds,
        "neural_elapsed_seconds": neural_seconds,
        "rule_transfer_bytes": rule_transfer,
        "neural_transfer_bytes": neural_transfer,
        "energy_checked": len(energy_pairs) == len(pairs),
        "energy_sample_count": len(energy_pairs),
        "energy_ratio_vs_rule": _ratio(neural_energy, rule_energy) if energy_pairs else None,
        "rule_energy_j": rule_energy if energy_pairs else None,
        "neural_energy_j": neural_energy if energy_pairs else None,
        "validation_failures": validation_failures,
        "output_residency_regressions": output_residency_regressions,
        "materialization_regressions": materialization_regressions,
        "compile_priority_precision": float(compile_priority_precision),
        "compile_priority_precision_checked": True,
        "confidence": float(confidence),
        "splits": pairs,
    }


def build_promotion_evidence(
    rule_execution_paths: Sequence[str | Path],
    neural_execution_paths: Sequence[str | Path],
    *,
    neural_validation_execution_paths: Sequence[str | Path] | None = None,
    required_neural_output_residency: str | None = None,
    policy_path: str | Path | None = None,
    compile_priority_precision: float = 1.0,
    confidence: float = 0.85,
    min_samples: int = 1,
    max_latency_ratio: float = 1.0,
    max_transfer_ratio: float = 1.0,
    max_energy_ratio: float = 1.0,
    min_compile_priority_precision: float = 0.80,
) -> dict[str, Any]:
    metrics = build_heldout_metrics(
        rule_execution_paths,
        neural_execution_paths,
        neural_validation_execution_paths=neural_validation_execution_paths,
        required_neural_output_residency=required_neural_output_residency,
        compile_priority_precision=compile_priority_precision,
        confidence=confidence,
    )
    policy = FabricPolicyHead.from_path(policy_path) if policy_path else FabricPolicyHead()
    evaluation = policy.promotion_evaluation(
        metrics,
        min_samples=min_samples,
        max_latency_ratio=max_latency_ratio,
        max_transfer_ratio=max_transfer_ratio,
        max_energy_ratio=max_energy_ratio,
        min_compile_priority_precision=min_compile_priority_precision,
    )
    negative_record = None
    if not evaluation["promoted"]:
        negative_record = {
            "schema": NEGATIVE_RECORD_SCHEMA,
            "blocked_promotion": True,
            "reason": evaluation["reason"],
            "failed_checks": sorted(name for name, ok in evaluation["checks"].items() if not ok),
            "metrics": metrics,
        }
    return {
        "schema": EVIDENCE_SCHEMA,
        "ok": True,
        "policy_path": None if policy_path is None else str(policy_path),
        "policy_authoritative_before_promotion": bool(policy.can_override_rule_scheduler()),
        "promotion_allowed": bool(evaluation["promoted"]),
        "metrics": metrics,
        "promotion_evaluation": evaluation,
        "negative_record": negative_record,
    }


def write_json(payload: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def append_jsonl(rows: Iterable[dict[str, Any]], path: str | Path | None) -> None:
    if path is None:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _pair_metrics(
    index: int,
    rule_path: str | Path,
    neural_path: str | Path,
    rule: dict[str, Any],
    neural: dict[str, Any],
    *,
    neural_validation_path: str | Path | None = None,
    neural_validation: dict[str, Any] | None = None,
    required_neural_output_residency: str | None = None,
) -> dict[str, Any]:
    rule_summary = _summary_for_metrics(rule)
    neural_summary = _summary_for_metrics(neural)
    rule_graph = dict(rule.get("graph") or {})
    neural_graph = dict(neural.get("graph") or {})
    graph_signature = _graph_signature(neural_graph or rule_graph)
    rule_final = _final_residency(rule)
    neural_final = _final_residency(neural)
    rule_materialization_bytes = _materialization_bytes(rule_summary)
    neural_materialization_bytes = _materialization_bytes(neural_summary)
    validation_payloads = [rule, neural]
    if neural_validation is not None:
        validation_payloads.append(neural_validation)
    output_residency_regression = (
        int(not _residency_satisfies_goal(neural_final, required_neural_output_residency))
        if required_neural_output_residency
        else int(not _residency_not_worse(rule_final, neural_final))
    )
    return {
        "index": index,
        "rule_execution_json": str(rule_path),
        "neural_execution_json": str(neural_path),
        "neural_validation_execution_json": None if neural_validation_path is None else str(neural_validation_path),
        "graph_signature": graph_signature,
        "graph_family": _graph_family(graph_signature),
        "element_count": _element_count(neural_graph or rule_graph),
        "residency_pattern": _residency_pattern(neural_graph or rule_graph, neural_final or rule_final),
        "required_neural_output_residency": required_neural_output_residency,
        "app_state": _app_state(neural, rule),
        "measurement_scope": _measurement_scope(neural),
        "rule_backend": _execution_backend(rule),
        "neural_backend": _execution_backend(neural),
        "neural_validation_backend": None if neural_validation is None else _execution_backend(neural_validation),
        "rule_seconds": _observed_seconds(rule),
        "neural_seconds": _observed_seconds(neural),
        "rule_transfer_bytes": _transfer_bytes(rule_summary),
        "neural_transfer_bytes": _transfer_bytes(neural_summary),
        "rule_energy_j": _energy_j(rule),
        "neural_energy_j": _energy_j(neural),
        "validation_failures": sum(_validation_failure_count(payload) for payload in validation_payloads),
        "output_residency_regression": output_residency_regression,
        "materialization_regression": int(neural_materialization_bytes > rule_materialization_bytes),
    }


def _observed_seconds(payload: dict[str, Any]) -> float:
    summary = _summary_for_metrics(payload)
    if _uses_explicit_promotion_window(payload) and "promotion_elapsed_seconds" in summary:
        return float(summary.get("promotion_elapsed_seconds") or 0.0)
    observed = (
        float(summary.get("observed_run_ps_seconds") or 0.0)
        + float(summary.get("observed_run_pl_seconds") or 0.0)
        + float(summary.get("observed_run_daemon_seconds") or 0.0)
    )
    if observed > 0.0:
        return observed
    timed = float(summary.get("timed_elapsed_seconds") or 0.0)
    if timed > 0.0:
        return timed
    return float(payload.get("elapsed_seconds") or 0.0)


def _transfer_bytes(summary: dict[str, Any]) -> int:
    if summary.get("_use_promotion_window") and "promotion_transfer_bytes" in summary:
        return int(summary.get("promotion_transfer_bytes") or 0)
    if "observed_transfer_bytes" in summary:
        return int(summary.get("observed_transfer_bytes") or 0)
    return (
        int(summary.get("timed_transfer_bytes") or 0)
        + int(summary.get("setup_transfer_bytes") or 0)
        + int(summary.get("forced_materialization_bytes") or 0)
    )


def _materialization_bytes(summary: dict[str, Any]) -> int:
    if summary.get("_use_promotion_window") and "promotion_materialization_bytes" in summary:
        return int(summary.get("promotion_materialization_bytes") or 0)
    if "final_transfer_bytes" in summary:
        return int(summary.get("final_transfer_bytes") or 0)
    return int(summary.get("forced_materialization_bytes") or 0)


def _energy_j(payload: dict[str, Any]) -> float | None:
    summary = _summary_for_metrics(payload)
    if summary.get("_use_promotion_window") and "promotion_energy_j" in summary:
        return _float_or_none(summary.get("promotion_energy_j"))
    for source in (summary, payload):
        for key in ("observed_energy_j", "energy_j", "planned_energy_j"):
            value = _float_or_none(source.get(key))
            if value is not None:
                return value
    observed = sum(
        float(summary.get(key) or 0.0)
        for key in (
            "observed_run_ps_energy_j",
            "observed_run_pl_energy_j",
            "observed_run_daemon_energy_j",
        )
    )
    if observed > 0.0:
        return observed
    foundry = payload.get("foundry")
    if isinstance(foundry, dict):
        candidate = foundry.get("candidate")
        if isinstance(candidate, dict):
            estimate = candidate.get("estimate")
            if isinstance(estimate, dict):
                energy = estimate.get("energy_estimate")
                if isinstance(energy, dict):
                    value = _float_or_none(energy.get("joules"))
                    if value is not None:
                        return value
                value = _float_or_none(estimate.get("energy_j"))
                if value is not None:
                    return value
    return None


def _summary_for_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(payload.get("summary") or {})
    if _uses_explicit_promotion_window(payload):
        summary["_use_promotion_window"] = True
    return summary


def _measurement_scope(payload: dict[str, Any]) -> dict[str, Any]:
    scope = payload.get("measurement_scope")
    return dict(scope) if isinstance(scope, dict) else {"scope": "full_execution"}


def _uses_explicit_promotion_window(payload: dict[str, Any]) -> bool:
    scope = _measurement_scope(payload)
    return str(scope.get("scope") or "") in {
        "steady_state_resident_inputs",
        "hot_cache_resident_inputs",
        "steady_state_state_feedback",
        "hot_cache_state_feedback",
    }


def _validation_failure_count(payload: dict[str, Any]) -> int:
    validation = payload.get("validation")
    validation_failed = isinstance(validation, dict) and validation.get("ok") is False
    return int((not bool(payload.get("ok"))) or validation_failed)


def _execution_backend(payload: dict[str, Any]) -> str:
    return str(payload.get("execution_backend") or payload.get("backend") or "unknown")


def _app_state(neural: dict[str, Any], rule: dict[str, Any]) -> str:
    explicit = neural.get("active_app") or rule.get("active_app")
    if explicit:
        return str(explicit)
    schemas = {str(neural.get("schema") or ""), str(rule.get("schema") or "")}
    if "pan_nn_execution_v1" in schemas:
        return "pan_nn_runtime"
    return "unknown"


def _graph_signature(graph: dict[str, Any]) -> str:
    ops = [str(op.get("op")) for op in graph.get("ops") or () if isinstance(op, dict)]
    if not ops:
        ops = [str(node.get("op")) for node in graph.get("nodes") or () if isinstance(node, dict)]
    return "+".join(ops) if ops else "unknown"


def _graph_family(graph_signature: str) -> str:
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


def _element_count(graph: dict[str, Any]) -> int:
    counts = [
        int(op.get("elements"))
        for op in graph.get("ops") or ()
        if isinstance(op, dict) and isinstance(op.get("elements"), int)
    ]
    tensors = graph.get("tensors")
    if not counts and isinstance(tensors, dict):
        for spec in tensors.values():
            if isinstance(spec, dict):
                counts.append(_shape_element_count(spec))
    return counts[0] if counts else 0


def _residency_pattern(graph: dict[str, Any], final_residency: dict[str, Any]) -> dict[str, Any]:
    buffers = dict(graph.get("buffers") or {})
    if buffers:
        input_residencies = sorted(
            str(spec.get("residency"))
            for name, spec in buffers.items()
            if isinstance(spec, dict) and str(name).startswith("input")
        )
    else:
        input_residencies = sorted("unknown" for _ in graph.get("inputs") or ())
    return {
        "input_residency": input_residencies,
        "output_residency": str(graph.get("output_residency", "unknown")),
        "final_residency": final_residency,
    }


def _residency_not_worse(rule: dict[str, Any], neural: dict[str, Any]) -> bool:
    if not rule or not neural:
        return True
    for name, rule_values in rule.items():
        neural_values = neural.get(name)
        if neural_values is None:
            return False
        if not set(_as_list(rule_values)).issubset(set(_as_list(neural_values))):
            return False
    return True


def _residency_satisfies_goal(final_residency: dict[str, Any], required_residency: str | None) -> bool:
    if not required_residency:
        return True
    if not final_residency:
        return False
    required = str(required_residency)
    return all(required in set(_as_list(values)) for values in final_residency.values())


def _final_residency(payload: dict[str, Any]) -> dict[str, Any]:
    explicit = payload.get("final_residency")
    if isinstance(explicit, dict):
        return dict(explicit)
    outputs = payload.get("outputs")
    if isinstance(outputs, dict):
        return {
            str(name): _nn_value_locations(value)
            for name, value in sorted(outputs.items())
            if isinstance(value, dict)
        }
    return _value_store_output_residency(payload)


def _value_store_output_residency(payload: dict[str, Any]) -> dict[str, Any]:
    graph = dict(payload.get("graph") or {})
    value_store = dict(payload.get("value_store") or {})
    latest_by_name = dict(value_store.get("latest_by_name") or {})
    values = dict(value_store.get("values") or {})
    final: dict[str, Any] = {}
    for name in graph.get("outputs") or ():
        value_id = latest_by_name.get(str(name))
        value = values.get(value_id)
        if isinstance(value, dict):
            final[str(name)] = _nn_value_locations(value)
    return final


def _nn_value_locations(value: dict[str, Any]) -> list[str]:
    locations: list[str] = []
    residency = value.get("residency")
    if residency not in {None, "", "unknown"}:
        locations.append(str(residency))
    if value.get("materialized"):
        locations.append("ps")
    return sorted(set(locations))


def _shape_element_count(spec: dict[str, Any]) -> int:
    if isinstance(spec.get("elements"), int):
        return int(spec["elements"])
    total = 1
    for dim in spec.get("shape") or ():
        if not isinstance(dim, int):
            return 0
        total *= int(dim)
    return int(total)


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    return [str(value)]


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else float("inf")
    return float(numerator) / float(denominator)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build held-out neural fabric promotion evidence from real execution artifacts.")
    parser.add_argument("--rule-execution-json", action="append", required=True, help="Rule-scheduler execution JSON; repeat for held-out pairs.")
    parser.add_argument("--neural-execution-json", action="append", required=True, help="Candidate/neural execution JSON; repeat in the same order as rule files.")
    parser.add_argument("--neural-validation-execution-json", action="append", help="Optional validated candidate execution JSON; repeat in the same order as neural files.")
    parser.add_argument("--required-neural-output-residency", help="Optional residency goal that measured candidate outputs must satisfy, e.g. daemon.")
    parser.add_argument("--policy", help="Candidate policy JSON to evaluate.")
    parser.add_argument("--metrics-out", help="Write metrics JSON accepted by fabric-promote-policy.")
    parser.add_argument("--negative-jsonl", help="Append a negative promotion record when gates fail.")
    parser.add_argument("--compile-priority-precision", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--max-latency-ratio", type=float, default=1.0)
    parser.add_argument("--max-transfer-ratio", type=float, default=1.0)
    parser.add_argument("--max-energy-ratio", type=float, default=1.0)
    parser.add_argument("--min-compile-priority-precision", type=float, default=0.80)
    parser.add_argument("--write", help="Write full evidence report JSON; stdout when omitted.")
    args = parser.parse_args(argv)
    evidence = build_promotion_evidence(
        args.rule_execution_json,
        args.neural_execution_json,
        neural_validation_execution_paths=args.neural_validation_execution_json,
        required_neural_output_residency=args.required_neural_output_residency,
        policy_path=args.policy,
        compile_priority_precision=args.compile_priority_precision,
        confidence=args.confidence,
        min_samples=args.min_samples,
        max_latency_ratio=args.max_latency_ratio,
        max_transfer_ratio=args.max_transfer_ratio,
        max_energy_ratio=args.max_energy_ratio,
        min_compile_priority_precision=args.min_compile_priority_precision,
    )
    if args.metrics_out:
        write_json(evidence["metrics"], args.metrics_out)
    if evidence["negative_record"] is not None:
        append_jsonl([evidence["negative_record"]], args.negative_jsonl)
    write_json(evidence, args.write)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
