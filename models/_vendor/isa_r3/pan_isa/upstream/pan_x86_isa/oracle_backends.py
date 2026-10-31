"""Shared oracle backend report and comparison helpers.

The module records typed state contracts even when a backend cannot yet execute
the requested profile. That gives full-ISA reports a tested blocker instead of
an unclassified hole.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pan_x86_automation.common import maybe_json, write_json


ENGINES = ("qemu", "bochs")
COMPARISON_STATUSES = (
    "exact normal agreement",
    "exact exception agreement",
    "undefined-flag-compatible agreement",
    "advisory/no-op policy match",
    "environment-masked match",
    "memory mismatch",
    "preserved-state mismatch",
    "oracle divergence",
    "backend unsupported",
    "timeout/crash",
)


def _engine_binary(engine: str) -> str:
    if engine == "qemu":
        return "qemu-system-x86_64"
    if engine == "bochs":
        return "bochs"
    raise ValueError(f"unknown oracle engine {engine!r}")


def backend_probe(engine: str, target_profile: str) -> Dict[str, Any]:
    binary = _engine_binary(engine)
    path = shutil.which(binary)
    return {
        "engine": engine,
        "binary": binary,
        "path": path,
        "target_profile": target_profile,
        "binary_available": bool(path),
        "shared_backend_profiles": [
            "long64-ring3",
            "long64-ring0",
            "compat32-ring3",
            "compat32-ring0",
            "prot32-ring3",
            "prot32-ring0",
            "real16",
        ],
        "shared_harness_available": False,
    }


def typed_state_from_trace(trace: Dict[str, Any], after: bool = True) -> Dict[str, Any]:
    state = ((trace.get("state") or {}).get("after" if after else "before")) or {}
    return {
        "gpr": state.get("gpr") or {},
        "rflags": state.get("rflags"),
        "rip": state.get("rip"),
        "segments": state.get("segments") or {},
        "control": state.get("control") or {},
        "descriptor_tables": state.get("descriptor_tables") or {},
        "msr": state.get("msr") or {},
        "x87": state.get("x87") or {},
        "mmx": state.get("mmx") or {},
        "xmm": state.get("xmm") or {},
        "ymm": state.get("ymm") or {},
        "zmm": state.get("zmm") or {},
        "opmask": state.get("opmask") or {},
        "mxcsr": state.get("mxcsr"),
        "memory_digest": state.get("memory_digest"),
        "io_ports": state.get("io_ports") or {},
    }


def stage_evidence_result(
    *,
    iform: str,
    engine: str,
    target_profile: str,
    evidence: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "iform": iform,
        "engine": engine,
        "target_profile": target_profile,
        "ok": True,
        "promotion_eligible": False,
        "classification": "compatibility_stage_exact",
        "outcome": "normal",
        "typed_state": {
            "before": {},
            "after": {},
            "touched_state": [],
            "preserved_state": [],
            "exception": None,
            "memory_events": [],
        },
        "source": evidence,
        "blocker": None,
    }


def backend_blocker_result(
    *,
    iform: str,
    engine: str,
    target_profile: str,
    reason: str,
    tested: bool = True,
) -> Dict[str, Any]:
    probe = backend_probe(engine, target_profile)
    return {
        "iform": iform,
        "engine": engine,
        "target_profile": target_profile,
        "ok": False,
        "promotion_eligible": False,
        "classification": "backend_unsupported",
        "outcome": "backend_unsupported",
        "typed_state": None,
        "source": None,
        "blocker": {
            "tested": tested,
            "kind": "shared_oracle_backend_limit",
            "reason": reason,
            "probe": probe,
        },
    }


def missing_oracle_result(*, iform: str, engine: str, target_profile: str, reason: str) -> Dict[str, Any]:
    return {
        "iform": iform,
        "engine": engine,
        "target_profile": target_profile,
        "ok": False,
        "promotion_eligible": False,
        "classification": "missing_or_unclassified",
        "outcome": "missing",
        "typed_state": None,
        "source": None,
        "blocker": {
            "tested": True,
            "kind": "missing_oracle_evidence",
            "reason": reason,
            "probe": backend_probe(engine, target_profile),
        },
    }


def write_oracle_report(root: Path, full_isa_dir: str, shard_id: str, report: Dict[str, Any]) -> None:
    write_json(root / full_isa_dir / "shards" / shard_id / "oracle" / f"{report['engine']}_oracle_report.json", report)


def load_oracle_report(root: Path, full_isa_dir: str, shard_id: str, engine: str) -> Dict[str, Any]:
    return maybe_json(root / full_isa_dir / "shards" / shard_id / "oracle" / f"{engine}_oracle_report.json") or {}


def result_by_iform(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(item.get("iform")): item for item in report.get("results") or []}


def _typed_equal(left: Any, right: Any, keys: Sequence[str]) -> bool:
    if left is None or right is None:
        return False
    for key in keys:
        if left.get(key) != right.get(key):
            return False
    return True


def compare_result(iform: str, qemu: Optional[Dict[str, Any]], bochs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    qemu = qemu or {}
    bochs = bochs or {}
    engines = {"qemu": qemu, "bochs": bochs}
    if qemu.get("ok") and bochs.get("ok"):
        q_state = qemu.get("typed_state") or {}
        b_state = bochs.get("typed_state") or {}
        q_exc = q_state.get("exception")
        b_exc = b_state.get("exception")
        if q_exc or b_exc:
            exact = q_exc == b_exc
            status = "exact exception agreement" if exact else "oracle divergence"
        else:
            q_after = q_state.get("after") or {}
            b_after = b_state.get("after") or {}
            exact = _typed_equal(
                q_after,
                b_after,
                ("gpr", "rflags", "rip", "segments", "control", "descriptor_tables", "msr", "x87", "mmx", "xmm", "ymm", "zmm", "opmask", "mxcsr", "memory_digest"),
            )
            if exact:
                status = "exact normal agreement"
            elif q_after.get("memory_digest") != b_after.get("memory_digest"):
                status = "memory mismatch"
            elif _typed_equal(
                q_after,
                b_after,
                ("gpr", "rip", "segments", "control", "descriptor_tables", "msr", "x87", "mmx", "xmm", "ymm", "zmm", "opmask", "mxcsr"),
            ):
                status = "undefined-flag-compatible agreement"
                exact = True
            else:
                status = "preserved-state mismatch"
        return {
            "iform": iform,
            "ok": exact,
            "promotion_eligible": exact and bool(qemu.get("promotion_eligible") and bochs.get("promotion_eligible")),
            "comparison_status": status,
            "engines": engines,
        }
    blockers = [item.get("blocker") for item in (qemu, bochs) if item.get("blocker")]
    if blockers and all((blocker or {}).get("kind") == "shared_oracle_backend_limit" for blocker in blockers):
        return {
            "iform": iform,
            "ok": False,
            "promotion_eligible": False,
            "comparison_status": "backend unsupported",
            "engines": engines,
            "accepted_blocker": True,
            "blockers": blockers,
        }
    if any((item.get("classification") or "") in {"timeout", "crash"} for item in (qemu, bochs)):
        status = "timeout/crash"
    else:
        status = "oracle divergence" if qemu.get("ok") or bochs.get("ok") else "backend unsupported"
    return {
        "iform": iform,
        "ok": False,
        "promotion_eligible": False,
        "comparison_status": status,
        "engines": engines,
        "blockers": blockers,
    }


def compare_reports(root: Path, full_isa_dir: str, shard: Dict[str, Any], engines: Iterable[str] = ENGINES) -> Dict[str, Any]:
    shard_id = str(shard.get("shard_id"))
    target_profile = str(shard.get("target_profile"))
    reports = {engine: load_oracle_report(root, full_isa_dir, shard_id, engine) for engine in engines}
    indexed = {engine: result_by_iform(report) for engine, report in reports.items()}
    results: List[Dict[str, Any]] = []
    for row in shard.get("iforms") or []:
        iform = str(row.get("iform"))
        results.append(compare_result(iform, indexed.get("qemu", {}).get(iform), indexed.get("bochs", {}).get(iform)))
    exact = [item["iform"] for item in results if item["comparison_status"] in {"exact normal agreement", "exact exception agreement"}]
    promotion_eligible = [item["iform"] for item in results if item.get("promotion_eligible")]
    blockers = [item for item in results if item.get("accepted_blocker")]
    report = {
        "schema_version": "pan-x86-full-isa-oracle-comparison-v2",
        "created_unix": int(time.time()),
        "target": "all",
        "target_profile": target_profile,
        "shard_id": shard_id,
        "engines": list(engines),
        "comparison_status_semantics": list(COMPARISON_STATUSES),
        "ok": bool(results) and all(item["comparison_status"] in {"exact normal agreement", "exact exception agreement"} for item in results),
        "all_results_classified": bool(results) and all(item.get("comparison_status") in COMPARISON_STATUSES for item in results),
        "exact_agreement_iforms": sorted(exact),
        "promotion_eligible_iforms": sorted(promotion_eligible),
        "accepted_backend_blocker_iforms": sorted(item["iform"] for item in blockers),
        "results": results,
    }
    write_json(root / full_isa_dir / "shards" / shard_id / "oracle" / "oracle_comparison.json", report)
    return report
