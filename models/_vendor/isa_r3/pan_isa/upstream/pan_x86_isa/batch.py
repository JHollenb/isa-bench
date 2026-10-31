"""Batch planning and status helpers for full-ISA automation."""
from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

from pan_x86_automation.common import maybe_json, rel, write_json

from pan_x86_isa import full_isa
from pan_x86_isa.profiles import PROFILE_IDS, resolve_profile


BATCH_ROOT = Path(full_isa.FULL_ISA_DIR) / "batches"


def plan_batch(
    root: Path,
    matrix: Dict[str, Any],
    *,
    target_profile: str,
    categories: Sequence[str],
    limit_per_category: int,
    seed: int,
) -> Dict[str, Any]:
    profile_id = resolve_profile(target_profile)
    classification = full_isa.load_or_build_classification(root, matrix)
    category_set = {item.upper() for item in categories}
    rows_by_category: Dict[str, List[Dict[str, Any]]] = {}
    for row in classification.get("records") or []:
        status = ((row.get("profile_statuses") or {}).get(profile_id) or {})
        if status.get("mode_status") != "executable":
            continue
        category = str(row.get("category") or "UNKNOWN")
        if category_set and category.upper() not in category_set:
            continue
        rows_by_category.setdefault(category, []).append(row)

    shards = []
    for category in sorted(rows_by_category):
        rows = sorted(rows_by_category[category], key=lambda row: str(row.get("iform") or ""))[:limit_per_category]
        if not rows:
            continue
        shard_plan = full_isa.plan_shards(
            root,
            matrix,
            target_profile=profile_id,
            category_filter=[category],
            iform_filter=[str(row.get("iform")) for row in rows],
            limit=len(rows),
            seed=seed,
        )
        shards.extend(shard_plan.get("shards") or [])

    batch_id = full_isa.stable_id(
        [profile_id, ",".join(sorted(category_set)), str(limit_per_category), str(seed)]
        + [str(shard.get("shard_id")) for shard in shards],
        "isa-batch",
    )
    return {
        "schema_version": "pan-x86-full-isa-batch-plan-v1",
        "target": "all",
        "target_profile": profile_id,
        "batch_id": batch_id,
        "created_unix": int(time.time()),
        "selection": {
            "categories": sorted(category_set),
            "limit_per_category": limit_per_category,
            "seed": seed,
        },
        "shards": shards,
        "status": "planned",
    }


def write_batch_plan(root: Path, batch_plan: Dict[str, Any]) -> None:
    batch_id = str(batch_plan["batch_id"])
    batch_dir = root / BATCH_ROOT / batch_id
    write_json(batch_dir / "batch_plan.json", batch_plan)
    for shard in batch_plan.get("shards") or []:
        write_json(root / full_isa.FULL_ISA_DIR / "shards" / shard["shard_id"] / "shard.json", shard)


def load_batch(root: Path, batch_id: str) -> Dict[str, Any]:
    batch = maybe_json(root / BATCH_ROOT / batch_id / "batch_plan.json")
    if not batch:
        raise FileNotFoundError(f"unknown batch: {batch_id}")
    return batch


def batch_status(root: Path, batch_id: str) -> Dict[str, Any]:
    batch = load_batch(root, batch_id)
    statuses = []
    for shard in batch.get("shards") or []:
        shard_id = str(shard.get("shard_id"))
        shard_dir = root / full_isa.FULL_ISA_DIR / "shards" / shard_id
        trace = maybe_json(shard_dir / "traces" / "trace_manifest.json") or {}
        qemu = maybe_json(shard_dir / "oracle" / "qemu_oracle_report.json") or {}
        bochs = maybe_json(shard_dir / "oracle" / "bochs_oracle_report.json") or {}
        comparison = maybe_json(shard_dir / "oracle" / "oracle_comparison.json") or {}
        train = maybe_json(shard_dir / "models" / "train_report.json") or {}
        promotion_refusals = []
        profile_id = str(shard.get("target_profile") or batch.get("target_profile") or "")
        promotion_dir = root / full_isa.FULL_ISA_DIR / "classification" / "promotions"
        for item in shard.get("iforms") or []:
            iform = str(item.get("iform"))
            path = promotion_dir / f"{iform}.{profile_id.replace('/', '_')}.json"
            promotion = maybe_json(path) or {}
            if promotion and not promotion.get("ok"):
                promotion_refusals.append(iform)
        state = "planned"
        failure_bucket = None
        failure_reasons: List[str] = []
        if trace:
            state = "traced"
            eligibility = trace.get("promotion_eligibility") or {}
            if not eligibility.get("helper_promotion_eligible", False):
                state = "encoder_blocked"
                failure_bucket = "encoder"
                failure_reasons.append("helper_not_promotion_eligible")
            if not eligibility.get("byte_disjoint_splits", True):
                state = "encoder_blocked"
                failure_bucket = "encoder"
                failure_reasons.append("byte_split_overlap")
        if qemu or bochs:
            state = "oracle_partial"
        if qemu.get("ok") and bochs.get("ok") and comparison.get("ok"):
            state = "oracle_compared"
        elif comparison:
            accepted = comparison.get("accepted_backend_blocker_iforms") or []
            state = "oracle_blocked"
            failure_bucket = failure_bucket or "oracle"
            if accepted:
                failure_reasons.append("backend_unsupported")
            else:
                failure_reasons.append("oracle_not_promotion_eligible")
        if train.get("ok"):
            state = "trained"
            failure_bucket = None
            failure_reasons = []
        elif train:
            state = "trainer_blocked"
            failure_bucket = failure_bucket or "trainer"
            for key, value in (train.get("blocker_counts") or {}).items():
                if value:
                    failure_reasons.append(key)
        if promotion_refusals:
            state = "promotion_refused"
            failure_bucket = failure_bucket or "promotion"
            failure_reasons.append("promotion_requirements_failed")
        statuses.append(
            {
                "shard_id": shard_id,
                "iform_count": len(shard.get("iforms") or []),
                "state": state,
                "failure_bucket": failure_bucket,
                "failure_reasons": sorted(set(failure_reasons)),
                "trace_count": trace.get("trace_count"),
                "qemu_ok": qemu.get("ok"),
                "bochs_ok": bochs.get("ok"),
                "comparison_ok": comparison.get("ok"),
                "train_ok": train.get("ok"),
            }
        )
    counts = Counter(item["state"] for item in statuses)
    report = {
        "schema_version": "pan-x86-full-isa-batch-status-v1",
        "target": "all",
        "target_profile": batch.get("target_profile"),
        "batch_id": batch_id,
        "ok": bool(statuses) and all(item["state"] == "trained" for item in statuses),
        "status_counts": dict(sorted(counts.items())),
        "failure_bucket_counts": dict(sorted(Counter(item["failure_bucket"] for item in statuses if item["failure_bucket"]).items())),
        "shards": statuses,
    }
    write_json(root / BATCH_ROOT / batch_id / "batch_status.json", report)
    return report


def failure_summary(root: Path, batch_id: str) -> Dict[str, Any]:
    status = batch_status(root, batch_id)
    failures = [item for item in status["shards"] if item["state"] != "trained"]
    reason_counts = Counter(reason for item in failures for reason in item.get("failure_reasons") or [])
    summary = {
        "schema_version": "pan-x86-full-isa-failure-summary-v1",
        "target": "all",
        "target_profile": status.get("target_profile"),
        "batch_id": batch_id,
        "ok": not failures,
        "failure_count": len(failures),
        "failure_states": dict(sorted(Counter(item["state"] for item in failures).items())),
        "failure_buckets": dict(sorted(Counter(item.get("failure_bucket") for item in failures if item.get("failure_bucket")).items())),
        "failure_reasons": dict(sorted(reason_counts.items())),
        "failures": failures,
    }
    write_json(root / BATCH_ROOT / batch_id / "failure_summary.json", summary)
    return summary
