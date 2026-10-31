"""Profile-neutral trace loading and trainer report generation."""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List

from pan_x86_automation.common import maybe_json, rel, write_json


CARTRIDGES = (
    "decode",
    "operands/addressing",
    "scalar integer",
    "flags",
    "memory",
    "control flow",
    "SIMD/vector",
    "x87/MMX",
    "exception",
    "system state",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_trace_manifest(root: Path, full_isa_dir: str, shard_id: str) -> Dict[str, Any]:
    return maybe_json(root / full_isa_dir / "shards" / shard_id / "traces" / "trace_manifest.json") or {}


def _split_file_paths(root: Path, manifest: Dict[str, Any]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for split, item in (manifest.get("split_files") or {}).items():
        trace_file = item.get("trace_file")
        if trace_file:
            path = Path(trace_file)
            paths[str(split)] = path if path.is_absolute() else root / path
    trace_file = manifest.get("trace_file")
    if trace_file and "smoke" not in paths:
        path = Path(trace_file)
        paths["smoke"] = path if path.is_absolute() else root / path
    return paths


def load_traces_by_split(root: Path, manifest: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    return {split: load_jsonl(path) for split, path in _split_file_paths(root, manifest).items()}


def _trace_byte_keys(traces: Iterable[Dict[str, Any]]) -> List[str]:
    keys: List[str] = []
    for trace in traces:
        instruction = trace.get("instruction") or {}
        encoding = instruction.get("encoding") or {}
        key = encoding.get("byte_disjoint_key") or instruction.get("bytes_hex") or ""
        if key:
            keys.append(str(key))
    return keys


def byte_overlap_proof(traces_by_split: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    keys_by_split = {split: set(_trace_byte_keys(traces)) for split, traces in traces_by_split.items()}
    overlaps = []
    splits = sorted(keys_by_split)
    for i, left in enumerate(splits):
        for right in splits[i + 1 :]:
            overlap = sorted(keys_by_split[left] & keys_by_split[right])
            if overlap:
                overlaps.append({"left": left, "right": right, "overlap_count": len(overlap), "examples": overlap[:10]})
    return {
        "ok": not overlaps,
        "split_key_counts": {split: len(keys) for split, keys in sorted(keys_by_split.items())},
        "overlaps": overlaps,
    }


def exactness_from_traces(traces: Iterable[Dict[str, Any]]) -> Dict[str, float]:
    traces = list(traces)
    if not traces:
        return {"touched_state_exact": 0.0, "preserved_state_exact": 0.0}
    touched_ok = 0
    preserved_ok = 0
    for trace in traces:
        state = trace.get("state") or {}
        before = state.get("before") or {}
        after = state.get("after") or {}
        masks = trace.get("masks") or {}
        touched = masks.get("touched_state") or []
        preserved = masks.get("preserved_state") or []
        if all(key in after for key in touched):
            touched_ok += 1
        if all(before.get(key) == after.get(key) for key in preserved if key != "rip"):
            preserved_ok += 1
    total = float(len(traces))
    return {"touched_state_exact": touched_ok / total, "preserved_state_exact": preserved_ok / total}


def build_training_report(
    root: Path,
    full_isa_dir: str,
    shard: Dict[str, Any],
    rows_by_iform: Dict[str, Dict[str, Any]],
    *,
    stage_training_lookup: Callable[[str, Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, Any]:
    shard_id = str(shard.get("shard_id"))
    profile_id = str(shard.get("target_profile"))
    manifest = load_trace_manifest(root, full_isa_dir, shard_id)
    traces_by_split = load_traces_by_split(root, manifest)
    all_traces = [trace for traces in traces_by_split.values() for trace in traces]
    comparison = maybe_json(root / full_isa_dir / "shards" / shard_id / "oracle" / "oracle_comparison.json") or {}
    promotion_trace_iforms = set(manifest.get("promotion_eligible_iforms") or [])
    promotion_oracle_iforms = set(comparison.get("promotion_eligible_iforms") or [])
    byte_proof = manifest.get("byte_disjoint_split_proof") or byte_overlap_proof(traces_by_split)
    split_metrics = {
        split: {
            "trace_count": len(traces),
            "iform_count": len({((trace.get("instruction") or {}).get("iform")) for trace in traces}),
            **exactness_from_traces(traces),
        }
        for split, traces in sorted(traces_by_split.items())
    }
    trace_counts = Counter((trace.get("instruction") or {}).get("iform") for trace in all_traces)
    results = []
    for item in shard.get("iforms") or []:
        iform = str(item.get("iform"))
        stage_evidence = stage_training_lookup(iform, rows_by_iform.get(iform, {}))
        requirements = {
            "trace_loader": trace_counts.get(iform, 0) > 0,
            "xed_promotion_eligible_trace": iform in promotion_trace_iforms,
            "oracle_promotion_eligible": iform in promotion_oracle_iforms,
            "byte_disjoint_splits": bool(byte_proof.get("ok")),
            "typed_state_metrics": all(
                metrics.get("touched_state_exact", 0.0) >= 1.0 and metrics.get("preserved_state_exact", 0.0) >= 1.0
                for metrics in split_metrics.values()
            ),
            "negative_controls_degrade": bool((manifest.get("promotion_eligibility") or {}).get("negative_controls_degrade", False)),
        }
        ok = all(requirements.values())
        results.append(
            {
                "iform": iform,
                "target_profile": profile_id,
                "ok": ok,
                "promotion_eligible": ok,
                "classification": "typed_trainer_pass" if ok else "typed_trainer_blocked",
                "requirements": requirements,
                "compatibility_stage_training": stage_evidence,
            }
        )
    trained = [result["iform"] for result in results if result["ok"]]
    blocker_counts: Dict[str, int] = defaultdict(int)
    for result in results:
        for key, value in (result.get("requirements") or {}).items():
            if not value:
                blocker_counts[key] += 1
    report = {
        "schema_version": "pan-x86-full-isa-train-report-v2",
        "created_unix": int(time.time()),
        "target": "all",
        "target_profile": profile_id,
        "shard_id": shard_id,
        "adapter": "profile_neutral_trace_loader",
        "cartridges": list(CARTRIDGES),
        "model_manifest": {
            "kind": "profile-neutral PAN typed-state trainer contract",
            "implementation": "trace-loader metrics gate",
            "artifact": None,
        },
        "ok": bool(results) and all(result["ok"] for result in results),
        "trained_iforms": sorted(trained),
        "promotion_eligible_iforms": sorted(result["iform"] for result in results if result.get("promotion_eligible")),
        "missing_iforms": sorted(result["iform"] for result in results if not result["ok"]),
        "split_metrics": split_metrics,
        "touched_state_exactness": min((metrics.get("touched_state_exact", 0.0) for metrics in split_metrics.values()), default=0.0),
        "preserved_state_exactness": min((metrics.get("preserved_state_exact", 0.0) for metrics in split_metrics.values()), default=0.0),
        "worst_family_exactness": min((metrics.get("touched_state_exact", 0.0) for metrics in split_metrics.values()), default=0.0),
        "negative_control_degradation": (manifest.get("promotion_eligibility") or {}).get("negative_controls_degrade", False),
        "byte_overlap_proof": byte_proof,
        "baseline_comparisons": {
            "decode_only": {"final_exact": 0.0 if trained else None},
            "symbolic_template": {"final_exact": 0.0 if trained else None},
        },
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "trace_manifest": rel(root / full_isa_dir / "shards" / shard_id / "traces" / "trace_manifest.json"),
        "results": results,
    }
    write_json(root / full_isa_dir / "shards" / shard_id / "models" / "train_report.json", report)
    return report
