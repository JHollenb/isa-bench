"""All-profile XED-driven full-ISA framework for PAN x86 automation.

The first implementation is framework-first: it classifies every XED row across
execution profiles, plans shards, writes profile-neutral trace manifests, and
records conservative oracle/training/promotion evidence.  The existing user64
module remains the compatibility path for legacy artifacts.
"""
from __future__ import annotations

import json
import random
import time
import fcntl
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from pan_x86_automation.common import maybe_json, rel, validated_run_dir, write_json
from pan_x86_automation import full_isa_user64

from pan_x86_isa import encoder, operand_policy, oracle_backends, source_freeze, training
from pan_x86_isa.profiles import (
    LONG64_RING3,
    PROFILE_IDS,
    TARGET,
    USER64_ALIAS,
    all_profile_policy,
    profile_policy,
    resolve_profile,
)


FULL_ISA_DIR = "full_isa"
CLASSIFICATION_PATH = Path(FULL_ISA_DIR) / "classification" / "xed_profile_matrix.json"
POLICY_PATH = Path(FULL_ISA_DIR) / "classification" / "all_profile_policy.json"
PROMOTION_LEDGER_PATH = Path(FULL_ISA_DIR) / "classification" / "promotion_ledger.json"
SHARD_PLAN_PATH = Path(FULL_ISA_DIR) / "shards" / "shard_plan.json"
REPORT_PATH = Path(FULL_ISA_DIR) / "reports" / "full_isa_report.json"

PROFILE_MODE_STATUSES = (
    "executable",
    "exception_only",
    "validate_only",
    "environment_dependent",
    "vendor_specific",
    "unsupported_by_profile",
    "tooling_gap",
)
PROFILE_COVERAGE_STATUSES = (
    "unimplemented",
    "classified_non_executable",
    "operand_subset_learned_validated",
    "oracle_validated",
    "trained",
    "fully_validated",
)

ALL_EXECUTABLE_CATEGORIES = set(full_isa_user64.EXECUTABLE_CATEGORIES) | {
    "AVX",
    "AVX2",
    "AVX512",
    "BROADCAST",
    "COMPRESS",
    "EXPAND",
    "GATHER",
    "KMASK",
    "SCATTER",
    "VEX",
    "VFMA",
    "VAES",
}
LEGACY_EXECUTABLE_CATEGORIES = {"DECIMAL", "MMX", "X87_ALU", "3DNOW"}
SYSTEM_EXECUTABLE_CATEGORIES = set(full_isa_user64.PRIVILEGED_CATEGORIES) | {
    "SEGOP",
    "SEMAPHORE",
    "SMAP",
    "SMX",
    "VMFUNC",
}
VENDOR_EXTENSIONS = {"SVM", "XOP", "TBM"}


def parse_csv(value: Optional[str]) -> List[str]:
    return full_isa_user64.parse_csv(value)


def stable_id(parts: Iterable[str], prefix: str, size: int = 12) -> str:
    return full_isa_user64.stable_id(parts, prefix, size)


def profile_entry_key(iform: str, profile: str) -> str:
    return f"{iform}::{profile}"


def _is_ring0(profile: Dict[str, Any]) -> bool:
    return int(profile.get("ring", 3)) == 0


def _mode(profile: Dict[str, Any]) -> str:
    return str(profile.get("mode") or "")


def _is_64bit(profile: Dict[str, Any]) -> bool:
    return int(profile.get("operand_bits", 0)) == 64


def classify_row_for_profile(row: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    profile = profile_policy(profile_id)
    attrs = {str(a) for a in row.get("attributes") or []}
    category = str(row.get("category") or "")
    extension = str(row.get("extension") or "")
    iclass = str(row.get("iclass") or "")
    isa_set = str(row.get("isa_set") or "")
    required_overlays: List[str] = []

    if extension in VENDOR_EXTENSIONS or "AMDONLY" in attrs or category in full_isa_user64.VENDOR_CATEGORIES:
        required_overlays.append("system-amd")
    if extension in {"VMX", "SGX"} or category in {"VTX", "SGX"}:
        required_overlays.append("system-intel")
    if extension == "AVX" or category == "AVX":
        required_overlays.append("avx")
    if extension == "AVX2" or category == "AVX2":
        required_overlays.append("avx2")
    if extension.startswith("AVX512") or category.startswith("AVX512"):
        required_overlays.append("avx512")

    if iclass in full_isa_user64.EXCEPTION_ONLY_ICLASSES:
        return {
            "mode_status": "exception_only",
            "reason": "architectural interrupt/invalid-opcode form; normal result is exception capture",
            "required_overlays": required_overlays,
        }
    if iclass in full_isa_user64.ENVIRONMENT_ICLASSES or category in {"RDRAND", "RDSEED", "XSAVE", "XSAVEOPT"}:
        return {
            "mode_status": "environment_dependent",
            "reason": "depends on host-visible time/randomness/OS-managed xstate",
            "required_overlays": sorted(set(required_overlays + ["system-intel", "system-amd"])),
        }
    if category in full_isa_user64.VALIDATE_ONLY_CATEGORIES or "PREFETCH" in iclass:
        return {
            "mode_status": "validate_only",
            "reason": "advisory or implementation-dependent normal-state update",
            "required_overlays": required_overlays,
        }
    if "RING0" in attrs or category in SYSTEM_EXECUTABLE_CATEGORIES:
        if _is_ring0(profile):
            return {
                "mode_status": "executable",
                "reason": "system instruction executable in ring-0 profile",
                "required_overlays": sorted(set(required_overlays + ["system-intel", "system-amd"])),
            }
        return {
            "mode_status": "exception_only",
            "reason": "privileged instruction is represented by its ring-3 fault path in this profile",
            "required_overlays": required_overlays,
        }
    if category in LEGACY_EXECUTABLE_CATEGORIES:
        if _mode(profile) in {"real", "protected", "compatibility"} or category in {"MMX", "X87_ALU"}:
            return {
                "mode_status": "executable",
                "reason": "legacy/compatibility instruction covered by this execution profile",
                "required_overlays": required_overlays,
            }
        return {
            "mode_status": "unsupported_by_profile",
            "reason": "legacy form is not valid in this profile without a compatibility submode",
            "required_overlays": required_overlays,
        }
    if "INVALID" in attrs and not (_mode(profile) == "real" and isa_set in {"I86", "I186"}):
        return {
            "mode_status": "unsupported_by_profile",
            "reason": "XED marks this row invalid for the profile/mode",
            "required_overlays": required_overlays,
        }
    if category in ALL_EXECUTABLE_CATEGORIES or row.get("coverage_status") == "operand_subset_learned_validated":
        if _is_64bit(profile) or isa_set not in {"LONGMODE", "I64"}:
            return {
                "mode_status": "executable",
                "reason": "normal architectural candidate for this profile",
                "required_overlays": sorted(set(required_overlays)),
            }
    if extension in VENDOR_EXTENSIONS or required_overlays:
        return {
            "mode_status": "vendor_specific",
            "reason": "requires vendor or extended CPUID overlay before execution",
            "required_overlays": sorted(set(required_overlays)),
        }
    return {
        "mode_status": "tooling_gap",
        "reason": "classified but no executable policy has been authored for this row/profile yet",
        "required_overlays": required_overlays,
    }


def coverage_status(row: Dict[str, Any], profile_id: str, profile_status: Dict[str, Any]) -> str:
    if profile_status["mode_status"] != "executable":
        return "classified_non_executable"
    if profile_id == LONG64_RING3 and row.get("coverage_status") == "operand_subset_learned_validated":
        return "operand_subset_learned_validated"
    return "unimplemented"


def build_classification(root: Path, matrix: Dict[str, Any]) -> Dict[str, Any]:
    inventory = full_isa_user64.load_xed_inventory(root, matrix)
    oracle_index = full_isa_user64.build_oracle_index(root, matrix)
    records: List[Dict[str, Any]] = []
    profile_counts: Dict[str, Counter[str]] = {profile: Counter() for profile in PROFILE_IDS}
    coverage_counts: Dict[str, Counter[str]] = {profile: Counter() for profile in PROFILE_IDS}

    for raw in inventory.records:
        row = dict(raw)
        row.setdefault("attributes", [])
        statuses: Dict[str, Any] = {}
        for profile_id in PROFILE_IDS:
            entry = classify_row_for_profile(row, profile_id)
            entry["coverage_status"] = coverage_status(row, profile_id, entry)
            statuses[profile_id] = entry
            profile_counts[profile_id][entry["mode_status"]] += 1
            coverage_counts[profile_id][entry["coverage_status"]] += 1
        iform = str(row.get("iform"))
        row["target"] = TARGET
        row["profile_statuses"] = statuses
        row["existing_evidence"] = {
            LONG64_RING3: {
                "stage": full_isa_user64.stage_for_subset(row),
                "qemu": oracle_index.get(iform, {}).get("qemu"),
                "bochs": oracle_index.get(iform, {}).get("bochs"),
            }
        }
        records.append(row)

    unclassified_profile_entries = sum(
        count
        for profile_id in PROFILE_IDS
        for status, count in profile_counts[profile_id].items()
        if status not in PROFILE_MODE_STATUSES
    )
    profile_entries = len(records) * len(PROFILE_IDS)
    frozen_sources = source_freeze.load_source_freeze(root) or source_freeze.build_source_freeze(root, matrix)
    classification = {
        "schema_version": "pan-x86-full-isa-profile-matrix-v1",
        "created_unix": int(time.time()),
        "target": TARGET,
        "profiles": list(PROFILE_IDS),
        "policy": all_profile_policy(),
        "source_freeze": frozen_sources,
        "inventory_source": inventory.source,
        "profile_mode_status_semantics": {status: status for status in PROFILE_MODE_STATUSES},
        "profile_coverage_status_semantics": {status: status for status in PROFILE_COVERAGE_STATUSES},
        "summary": {
            "xed_rows": len(records),
            "target_profiles": len(PROFILE_IDS),
            "profile_entries": profile_entries,
            "classified_profile_entries": profile_entries - unclassified_profile_entries,
            "unclassified_profile_entries": unclassified_profile_entries,
            "executable_profile_entries": sum(c.get("executable", 0) for c in profile_counts.values()),
            "exception_profile_entries": sum(c.get("exception_only", 0) for c in profile_counts.values()),
            "profile_mode_status_counts": {
                profile: dict(sorted(profile_counts[profile].items())) for profile in PROFILE_IDS
            },
            "profile_coverage_status_counts": {
                profile: dict(sorted(coverage_counts[profile].items())) for profile in PROFILE_IDS
            },
        },
        "records": records,
    }
    return classification


def write_classification(root: Path, classification: Dict[str, Any]) -> None:
    write_json(root / CLASSIFICATION_PATH, classification)
    write_json(root / POLICY_PATH, classification["policy"])
    write_json(
        root / FULL_ISA_DIR / "classification" / "summary.json",
        {
            "target": TARGET,
            "summary": classification["summary"],
            "source_freeze": classification["source_freeze"],
            "inventory_source": classification["inventory_source"],
        },
    )


def load_or_build_classification(root: Path, matrix: Dict[str, Any]) -> Dict[str, Any]:
    return maybe_json(root / CLASSIFICATION_PATH) or build_classification(root, matrix)


def row_by_iform(classification: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row.get("iform")): row for row in classification.get("records") or []}


def sampling_axes(row: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    axes = full_isa_user64.sampling_axes(row)
    profile = profile_policy(profile_id)
    axes["target_profile"] = profile_id
    axes["mode"] = profile["mode"]
    axes["ring"] = profile["ring"]
    axes["operand_bits"] = profile["operand_bits"]
    axes["address_bits"] = profile["address_bits"]
    if profile["system_state"]:
        axes["system_state"] = ["cr0", "cr3", "cr4", "gdtr", "idtr", "msr"]
    if profile["mode"] == "real":
        axes["segments"] = ["cs", "ds", "es", "ss"]
    return axes


def shard_iform_record(row: Dict[str, Any], profile_id: str) -> Dict[str, Any]:
    policy = operand_policy.build_operand_policy(row, profile_id)
    return {
        "iform": row.get("iform"),
        "iclass": row.get("iclass"),
        "category": row.get("category"),
        "extension": row.get("extension"),
        "isa_set": row.get("isa_set"),
        "profile_status": (row.get("profile_statuses") or {}).get(profile_id),
        "existing_evidence": (row.get("existing_evidence") or {}).get(profile_id),
        "operand_policy": policy,
        "operand_policy_id": policy["policy_id"],
        "sampling_axes": sampling_axes(row, profile_id),
    }


def plan_shards(
    root: Path,
    matrix: Dict[str, Any],
    *,
    target_profile: str,
    category_filter: Sequence[str],
    iform_filter: Sequence[str],
    limit: int,
    seed: int,
) -> Dict[str, Any]:
    profile_id = resolve_profile(target_profile)
    classification = load_or_build_classification(root, matrix)
    categories = {c.upper() for c in category_filter}
    iforms = set(iform_filter)
    candidates: List[Dict[str, Any]] = []
    for row in classification.get("records") or []:
        status = ((row.get("profile_statuses") or {}).get(profile_id) or {})
        if status.get("mode_status") != "executable":
            continue
        if categories and str(row.get("category") or "").upper() not in categories:
            continue
        if iforms and row.get("iform") not in iforms:
            continue
        candidates.append(row)

    candidates.sort(
        key=lambda row: (
            ((row.get("profile_statuses") or {}).get(profile_id) or {}).get("coverage_status") != "operand_subset_learned_validated",
            str(row.get("category") or ""),
            str(row.get("extension") or ""),
            str(row.get("iform") or ""),
        )
    )
    selected = candidates[:limit]
    shard_id = stable_id(
        [TARGET, profile_id, ",".join(category_filter), ",".join(iform_filter), str(seed)]
        + [str(row.get("iform")) for row in selected],
        "isa-shard",
    )
    shard = {
        "schema_version": "pan-x86-full-isa-shard-plan-v1",
        "target": TARGET,
        "target_profile": profile_id,
        "shard_id": shard_id,
        "status": "planned",
        "created_unix": int(time.time()),
        "selection": {
            "category": list(category_filter),
            "iform": list(iform_filter),
            "limit": limit,
            "seed": seed,
        },
        "iforms": [shard_iform_record(row, profile_id) for row in selected],
    }
    return {
        "schema_version": "pan-x86-full-isa-shard-plan-index-v1",
        "target": TARGET,
        "target_profile": profile_id,
        "classification_summary": classification["summary"],
        "shards": [shard],
    }


def write_shard_plan(root: Path, plan: Dict[str, Any]) -> None:
    write_json(root / SHARD_PLAN_PATH, plan)
    for shard in plan.get("shards") or []:
        write_json(root / FULL_ISA_DIR / "shards" / shard["shard_id"] / "shard.json", shard)


def load_shard(root: Path, shard_id: str) -> Dict[str, Any]:
    direct = maybe_json(root / FULL_ISA_DIR / "shards" / shard_id / "shard.json")
    if direct:
        return direct
    plan = maybe_json(root / SHARD_PLAN_PATH) or {}
    for shard in plan.get("shards") or []:
        if shard.get("shard_id") == shard_id:
            return shard
    raise FileNotFoundError(f"unknown full-ISA shard: {shard_id}")


def base_state(profile_id: str, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)
    profile = profile_policy(profile_id)
    bits = int(profile["operand_bits"])
    mask = (1 << min(bits, 64)) - 1
    width = 16 if bits == 64 else 8 if bits == 32 else 4
    gprs = {
        name: f"0x{rng.getrandbits(min(bits, 64)) & mask:0{width}x}"
        for name in ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"]
    }
    gprs["rsp"] = profile["default_sp"]
    state: Dict[str, Any] = {
        "gpr": gprs,
        "rflags": "0x0000000000000002",
        "rip": profile["default_ip"],
        "segments": {"cs": "0x0008", "ds": "0x0010", "es": "0x0010", "ss": "0x0010"},
        "control": {},
        "descriptor_tables": {},
        "msr": {},
        "x87": {},
        "mmx": {},
        "xmm": {f"xmm{i}": f"{rng.getrandbits(128):032x}" for i in range(4)},
        "ymm": {},
        "zmm": {},
        "opmask": {},
        "mxcsr": "0x00001f80",
        "memory_digest": f"sha256:{rng.getrandbits(256):064x}",
        "io_ports": {},
    }
    if profile["system_state"]:
        state["control"] = {"cr0": "0x80010033", "cr3": "0x0000000000100000", "cr4": "0x00000660"}
        state["descriptor_tables"] = {"gdtr": "0x0000000000000000:0x003f", "idtr": "0x0000000000000000:0x0fff"}
        state["msr"] = {"efer": "0x0000000000000500" if profile["long_mode"] else "0x0000000000000000"}
    return state


def _mutate_hex_byte(bytes_hex: str) -> str:
    if len(bytes_hex) < 2:
        return bytes_hex
    first = int(bytes_hex[:2], 16) ^ 0x01
    return f"{first:02x}{bytes_hex[2:]}"


def _read_trace_file(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    traces: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                traces.append(json.loads(line))
    return traces


def _existing_split_files(root: Path, manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    split_files = dict(manifest.get("split_files") or {})
    if manifest.get("trace_file") and "smoke" not in split_files:
        split_files["smoke"] = {
            "trace_file": manifest["trace_file"],
            "trace_count": manifest.get("trace_count", 0),
        }
    return split_files


def _source_freeze_summary(root: Path, matrix: Dict[str, Any]) -> Dict[str, Any]:
    freeze = source_freeze.load_source_freeze(root) or source_freeze.build_source_freeze(root, matrix)
    refs = freeze.get("references") or {}
    pdf_failures = [
        name
        for name in ("intel_sdm", "amd_apm")
        if (refs.get(name) or {}).get("status") != "frozen"
    ]
    non_pdf_refs_ok = all(
        record.get("status") == "frozen"
        for name, record in refs.items()
        if record.get("required") and name not in {"intel_sdm", "amd_apm"}
    )
    tools_ok = all(record.get("status") == "frozen" for record in (freeze.get("tools") or {}).values() if record.get("required"))
    return {
        "schema_version": freeze.get("schema_version"),
        "path": rel(root / source_freeze.SOURCE_FREEZE_PATH),
        "ok": bool(freeze.get("ok")),
        "ok_excluding_pdf_acquisition": bool(non_pdf_refs_ok and tools_ok),
        "pdf_acquisition_failures": pdf_failures,
        "gate_failures": freeze.get("gate_failures") or [],
    }


def generate_traces(
    root: Path,
    matrix: Dict[str, Any],
    shard_id: str,
    cases_per_iform: int,
    seed: int,
    split: str = "smoke",
) -> Dict[str, Any]:
    shard = load_shard(root, shard_id)
    profile_id = resolve_profile(str(shard.get("target_profile") or LONG64_RING3))
    out_dir = root / FULL_ISA_DIR / "shards" / shard_id / "traces"
    traces: List[Dict[str, Any]] = []
    for row_index, row in enumerate(shard.get("iforms") or []):
        iform = str(row.get("iform"))
        policy = row.get("operand_policy") or operand_policy.build_operand_policy(row, profile_id)
        cases = operand_policy.build_operand_cases(
            row,
            profile_id,
            cpuid_profile="base",
            split=split,
            cases_per_iform=cases_per_iform,
            seed=seed + row_index * 1009,
        )
        for case_index, operand_case in enumerate(cases):
            encoding = encoder.encode_case(row, policy, root, matrix, operand_case)
            original_bytes_hex = str(encoding.get("bytes_hex") or "")
            bytes_hex = original_bytes_hex
            if split == "negative-control" and bytes_hex:
                bytes_hex = _mutate_hex_byte(bytes_hex)
                encoding = dict(encoding)
                encoding["status"] = "negative_control_mutation"
                encoding["original_bytes_hex"] = original_bytes_hex
                encoding["bytes_hex"] = bytes_hex
                encoding["xed_roundtrip_ok"] = False
                encoding["promotion_eligible"] = False
                encoding["exact_failure_reason"] = "negative_control_opcode_byte_flip"
            encoding["byte_disjoint_key"] = bytes_hex
            asm = str(encoding.get("asm") or "")
            case_seed = seed + row_index * 1009 + case_index
            before = base_state(profile_id, case_seed)
            after = json.loads(json.dumps(before))
            insn_len = len(bytes.fromhex(bytes_hex)) if bytes_hex else 0
            if insn_len:
                after["rip"] = f"0x{int(str(before['rip']), 16) + insn_len:016x}"
            traces.append(
                {
                    "schema_version": "pan-x86-trace-v2",
                    "trace_id": f"{shard_id}:{iform}:{case_index}",
                    "target": TARGET,
                    "target_profile": profile_id,
                    "shard_id": shard_id,
                    "source": {
                        "generator": "pan_x86_isa.full_isa.generate_traces",
                        "seed": case_seed,
                        "oracle_provenance": [],
                        "semantic_status": "promotion_candidate" if encoding.get("promotion_eligible") else "decode_operand_smoke",
                    },
                    "mode": profile_policy(profile_id),
                    "instruction": {
                        "iform": iform,
                        "iclass": row.get("iclass"),
                        "category": row.get("category"),
                        "extension": row.get("extension"),
                        "isa_set": row.get("isa_set"),
                        "bytes_hex": bytes_hex,
                        "asm": asm,
                        "encoding": encoding,
                        "operand_case": operand_case,
                    },
                    "state": {"before": before, "after": after},
                    "masks": {
                        "defined_flags": ["cf", "pf", "af", "zf", "sf", "of"],
                        "undefined_flags": [],
                        "touched_state": ["rip"] if bytes_hex else [],
                        "preserved_state": ["gpr", "segments", "control", "xmm", "mxcsr", "memory_digest"],
                        "ignored_state": [],
                    },
                    "memory_events": [],
                    "exception": None,
                }
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{split}_traces.jsonl"
    jsonl.write_text("\n".join(json.dumps(trace, sort_keys=True) for trace in traces) + "\n", encoding="utf-8")
    with (out_dir / "trace_manifest.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing_manifest = maybe_json(out_dir / "trace_manifest.json") or {}
        split_files = _existing_split_files(root, existing_manifest)
        split_files[split] = {"trace_file": rel(jsonl), "trace_count": len(traces)}
        traces_by_split: Dict[str, List[Dict[str, Any]]] = {}
        for split_name, split_record in split_files.items():
            trace_file = Path(str(split_record.get("trace_file") or ""))
            trace_path = trace_file if trace_file.is_absolute() else root / trace_file
            traces_by_split[split_name] = _read_trace_file(trace_path)
        all_traces = [trace for split_traces in traces_by_split.values() for trace in split_traces]
        byte_proof = training.byte_overlap_proof(traces_by_split)
        helper_manifest = encoder.load_helper_manifest(root)
        source_summary = _source_freeze_summary(root, matrix)
        encoding_status_counts = Counter(
            ((trace.get("instruction") or {}).get("encoding") or {}).get("status") for trace in all_traces
        )
        static_fixture_count = encoding_status_counts.get("static_fixture", 0)
        negative_traces = traces_by_split.get("negative-control") or []
        negative_controls_degrade = bool(negative_traces) and all(
            not (((trace.get("instruction") or {}).get("encoding") or {}).get("promotion_eligible")) for trace in negative_traces
        )
        manifest = {
            "schema_version": "pan-x86-trace-manifest-v2",
            "target": TARGET,
            "target_profile": profile_id,
            "shard_id": shard_id,
            "split": split,
            "trace_file": rel(jsonl),
            "trace_count": len(all_traces),
            "split_files": split_files,
            "cases_per_iform": cases_per_iform,
            "iforms": sorted({trace["instruction"]["iform"] for trace in all_traces}),
            "encoding_status_counts": dict(sorted(encoding_status_counts.items())),
            "operand_policy_ids": sorted(
                {str(trace["instruction"]["encoding"].get("operand_policy_id")) for trace in all_traces}
            ),
            "encoder_helper_manifest": {
                "path": rel(root / encoder.HELPER_MANIFEST_REL),
                "available": bool(helper_manifest),
                "source_sha256": helper_manifest.get("source_sha256"),
                "binary_sha256": helper_manifest.get("binary_sha256"),
                "xed_linked": bool(helper_manifest.get("xed_linked")),
                "promotion_eligible": bool(helper_manifest.get("promotion_eligible")),
                "implementation": helper_manifest.get("implementation"),
            },
            "source_freeze": source_summary,
            "byte_disjoint_split_proof": byte_proof,
            "xed_roundtrip_ok_iforms": sorted(
                {
                    trace["instruction"]["iform"]
                    for trace in all_traces
                    if trace["instruction"]["encoding"]["xed_roundtrip_ok"]
                }
            ),
            "promotion_eligible_iforms": sorted(
                {
                    trace["instruction"]["iform"]
                    for trace in all_traces
                    if trace["instruction"]["encoding"].get("promotion_eligible")
                }
            ),
            "promotion_eligibility": {
                "source_freeze_ok": source_summary["ok"],
                "source_freeze_ok_excluding_pdf_acquisition": source_summary["ok_excluding_pdf_acquisition"],
                "helper_promotion_eligible": bool(helper_manifest.get("promotion_eligible")),
                "byte_disjoint_splits": bool(byte_proof.get("ok")),
                "static_fixture_free": static_fixture_count == 0,
                "negative_controls_degrade": negative_controls_degrade,
            },
        }
        write_json(out_dir / "trace_manifest.json", manifest)
        write_json(
            out_dir / f"{split}_trace_manifest.json",
            {**manifest, "trace_count": len(traces), "trace_file": rel(jsonl)},
        )
        fcntl.flock(lock, fcntl.LOCK_UN)
    return manifest


def _stage_evidence(root: Path, matrix: Dict[str, Any], iform: str, engine: str, profile_id: str) -> Optional[Dict[str, Any]]:
    if profile_id != LONG64_RING3:
        return None
    return full_isa_user64.existing_stage_oracle_evidence(root, matrix, iform, engine)


def normalize_oracle_check(root: Path, matrix: Dict[str, Any], shard_id: str, engine: str) -> Dict[str, Any]:
    shard = load_shard(root, shard_id)
    profile_id = resolve_profile(str(shard.get("target_profile") or LONG64_RING3))
    results = []
    for row in shard.get("iforms") or []:
        iform = str(row.get("iform"))
        evidence = _stage_evidence(root, matrix, iform, engine, profile_id)
        if evidence and evidence.get("ok"):
            result = oracle_backends.stage_evidence_result(
                iform=iform,
                engine=engine,
                target_profile=profile_id,
                evidence=evidence,
            )
        elif profile_id == LONG64_RING3:
            result = oracle_backends.missing_oracle_result(
                iform=iform,
                engine=engine,
                target_profile=profile_id,
                reason="no existing stage oracle evidence and shared oracle execution is not configured",
            )
        else:
            result = oracle_backends.backend_blocker_result(
                iform=iform,
                engine=engine,
                target_profile=profile_id,
                reason="shared QEMU/Bochs injection and typed-state capture harness is not implemented for this profile",
            )
        results.append(result)
    exact = [result["iform"] for result in results if result["ok"]]
    promotion_eligible = [result["iform"] for result in results if result.get("promotion_eligible")]
    report = {
        "schema_version": "pan-x86-full-isa-oracle-report-v2",
        "target": TARGET,
        "target_profile": profile_id,
        "shard_id": shard_id,
        "engine": engine,
        "adapter": "shared_oracle_backend_adapter",
        "backend_probe": oracle_backends.backend_probe(engine, profile_id),
        "ok": len(exact) == len(results) and bool(results),
        "checked_iforms": len(results),
        "exact_iforms": sorted(exact),
        "promotion_eligible_iforms": sorted(promotion_eligible),
        "missing_iforms": sorted(result["iform"] for result in results if not result["ok"]),
        "blocker_iforms": sorted(result["iform"] for result in results if result.get("blocker")),
        "results": results,
    }
    oracle_backends.write_oracle_report(root, FULL_ISA_DIR, shard_id, report)
    return report


def compare_oracles(root: Path, shard_id: str) -> Dict[str, Any]:
    shard = load_shard(root, shard_id)
    return oracle_backends.compare_reports(root, FULL_ISA_DIR, shard)


def train_shard(root: Path, matrix: Dict[str, Any], shard_id: str) -> Dict[str, Any]:
    classification = load_or_build_classification(root, matrix)
    rows = row_by_iform(classification)
    shard = load_shard(root, shard_id)
    profile_id = resolve_profile(str(shard.get("target_profile") or LONG64_RING3))
    _ = profile_id

    def lookup(iform: str, row: Dict[str, Any]) -> Dict[str, Any]:
        if resolve_profile(str(shard.get("target_profile") or LONG64_RING3)) == LONG64_RING3:
            return full_isa_user64.stage_training_evidence(root, matrix, row)
        return {"ok": False, "reason": "no compatibility stage training exists for this profile"}

    return training.build_training_report(root, FULL_ISA_DIR, shard, rows, stage_training_lookup=lookup)


def load_promotion_ledger(root: Path) -> Dict[str, Any]:
    return maybe_json(root / PROMOTION_LEDGER_PATH) or {
        "schema_version": "pan-x86-full-isa-promotion-ledger-v1",
        "target": TARGET,
        "promotions": {},
    }


def shard_reports_for_iform(root: Path, iform: str, profile_id: str) -> Dict[str, Any]:
    reports: Dict[str, Any] = {}
    shard_root = root / FULL_ISA_DIR / "shards"
    if not shard_root.exists():
        return reports
    for shard_dir in sorted(shard_root.glob("isa-shard-*")):
        shard = maybe_json(shard_dir / "shard.json") or {}
        if shard.get("target_profile") != profile_id:
            continue
        manifest = maybe_json(shard_dir / "traces" / "trace_manifest.json") or {}
        if iform in set(manifest.get("iforms") or []):
            reports["trace_manifest"] = manifest
        train = maybe_json(shard_dir / "models" / "train_report.json") or {}
        if iform in set(train.get("trained_iforms") or []) or iform in set(train.get("missing_iforms") or []):
            reports["train"] = train
        comparison = maybe_json(shard_dir / "oracle" / "oracle_comparison.json") or {}
        comparison_iforms = {
            str(item.get("iform"))
            for item in comparison.get("results") or []
        } | set(comparison.get("exact_agreement_iforms") or [])
        if iform in comparison_iforms:
            reports["comparison"] = comparison
        for engine in ("qemu", "bochs"):
            oracle = maybe_json(shard_dir / "oracle" / f"{engine}_oracle_report.json") or {}
            if iform in set(oracle.get("exact_iforms") or []):
                reports[engine] = oracle
    return reports


def promote_iform(root: Path, matrix: Dict[str, Any], iform: str, target_profile: str) -> Dict[str, Any]:
    profile_id = resolve_profile(target_profile)
    classification = load_or_build_classification(root, matrix)
    row = row_by_iform(classification).get(iform)
    if row is None:
        raise ValueError(f"unknown iform: {iform}")
    profile_status = ((row.get("profile_statuses") or {}).get(profile_id) or {})
    shard_reports = shard_reports_for_iform(root, iform, profile_id)
    qemu = shard_reports.get("qemu") or _stage_evidence(root, matrix, iform, "qemu", profile_id)
    bochs = shard_reports.get("bochs") or _stage_evidence(root, matrix, iform, "bochs", profile_id)
    train = shard_reports.get("train")
    manifest = shard_reports.get("trace_manifest") or {}
    if train:
        train_ok = iform in set(train.get("promotion_eligible_iforms") or [])
        train_source: Any = {"kind": "shard_train_report", "report": train.get("shard_id")}
    else:
        train_source = {"ok": False, "reason": "no profile-neutral train report evidence"}
        train_ok = False
    comparison = shard_reports.get("comparison")
    two_oracle_ok = bool(
        comparison
        and iform in set(comparison.get("exact_agreement_iforms") or [])
        and iform in set(comparison.get("promotion_eligible_iforms") or [])
    )
    roundtrip_ok = bool(
        manifest
        and iform in set(manifest.get("xed_roundtrip_ok_iforms") or [])
        and iform in set(manifest.get("promotion_eligible_iforms") or [])
    )
    manifest_eligibility = manifest.get("promotion_eligibility") or {}
    train_byte_proof = (train or {}).get("byte_overlap_proof") or manifest.get("byte_disjoint_split_proof") or {}
    source_summary = manifest.get("source_freeze") or _source_freeze_summary(root, matrix)
    train_results = {
        str(result.get("iform")): result
        for result in (train or {}).get("results") or []
    }
    train_result = train_results.get(iform) or {}
    train_reqs = train_result.get("requirements") or {}
    requirements = {
        "classified_executable_profile": profile_status.get("mode_status") == "executable",
        "source_freeze_ok_excluding_pdf_acquisition": bool(source_summary.get("ok_excluding_pdf_acquisition")),
        "xed_roundtrip_promotion_eligible": roundtrip_ok,
        "byte_disjoint_splits": bool(train_byte_proof.get("ok")),
        "two_oracle_exact_or_classified_divergence": two_oracle_ok,
        "typed_trainer_pass": train_ok,
        "touched_state_exact": bool(train_reqs.get("typed_state_metrics")) and (train or {}).get("touched_state_exactness", 0.0) >= 1.0,
        "preserved_state_exact": bool(train_reqs.get("typed_state_metrics")) and (train or {}).get("preserved_state_exactness", 0.0) >= 1.0,
        "negative_controls_degrade": bool(manifest_eligibility.get("negative_controls_degrade")),
        "static_fixture_free": bool(manifest_eligibility.get("static_fixture_free")),
        "no_stale_policy_helper_oracle_artifacts": bool(
            manifest.get("encoder_helper_manifest", {}).get("source_sha256")
            and manifest.get("encoder_helper_manifest", {}).get("binary_sha256")
            and comparison
            and train
        ),
    }
    ok = all(requirements.values())
    status = f"fully_validated_{profile_id}" if ok else "promotion_refused"
    promotion = {
        "schema_version": "pan-x86-full-isa-promotion-v1",
        "target": TARGET,
        "target_profile": profile_id,
        "iform": iform,
        "ok": ok,
        "status": status,
        "requirements": requirements,
        "row": {
            "iclass": row.get("iclass"),
            "category": row.get("category"),
            "extension": row.get("extension"),
            "isa_set": row.get("isa_set"),
            "profile_status": profile_status,
        },
        "evidence": {
            "qemu": qemu,
            "bochs": bochs,
            "comparison": comparison,
            "train": train_source,
            "trace_manifest": manifest,
        },
    }
    out_dir = root / FULL_ISA_DIR / "classification" / "promotions"
    safe_profile = profile_id.replace("/", "_")
    write_json(out_dir / f"{iform}.{safe_profile}.json", promotion)
    ledger = load_promotion_ledger(root)
    key = profile_entry_key(iform, profile_id)
    if ok:
        ledger["promotions"][key] = {
            "iform": iform,
            "target_profile": profile_id,
            "status": status,
            "updated_unix": int(time.time()),
            "path": rel(out_dir / f"{iform}.{safe_profile}.json"),
        }
    else:
        ledger.setdefault("refusals", {})[key] = {
            "iform": iform,
            "target_profile": profile_id,
            "status": "promotion_refused",
            "updated_unix": int(time.time()),
            "path": rel(out_dir / f"{iform}.{safe_profile}.json"),
            "requirements": requirements,
        }
    write_json(root / PROMOTION_LEDGER_PATH, ledger)
    return promotion


def collect_tooling_blockers(root: Path) -> Dict[str, Any]:
    blockers: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    shard_root = root / FULL_ISA_DIR / "shards"
    if not shard_root.exists():
        return {"by_key": {}, "counts": {}}
    for shard_dir in sorted(shard_root.glob("isa-shard-*")):
        shard = maybe_json(shard_dir / "shard.json") or {}
        profile_id = str(shard.get("target_profile") or "")
        iforms = [str(item.get("iform")) for item in shard.get("iforms") or []]
        manifest = maybe_json(shard_dir / "traces" / "trace_manifest.json") or {}
        manifest_eligibility = manifest.get("promotion_eligibility") or {}
        if manifest and not manifest_eligibility.get("helper_promotion_eligible"):
            for iform in iforms:
                blockers[profile_entry_key(iform, profile_id)].append(
                    {
                        "kind": "encoder_helper_not_promotion_eligible",
                        "shard_id": shard.get("shard_id"),
                        "tested": True,
                        "detail": manifest.get("encoder_helper_manifest") or {},
                    }
                )
        comparison = maybe_json(shard_dir / "oracle" / "oracle_comparison.json") or {}
        for iform in comparison.get("accepted_backend_blocker_iforms") or []:
            blockers[profile_entry_key(str(iform), profile_id)].append(
                {
                    "kind": "oracle_backend_unsupported",
                    "shard_id": shard.get("shard_id"),
                    "tested": True,
                }
            )
        train = maybe_json(shard_dir / "models" / "train_report.json") or {}
        for result in train.get("results") or []:
            if result.get("ok"):
                continue
            failed_requirements = [
                key for key, value in (result.get("requirements") or {}).items() if not value
            ]
            blockers[profile_entry_key(str(result.get("iform")), profile_id)].append(
                {
                    "kind": "trainer_requirements_failed",
                    "shard_id": shard.get("shard_id"),
                    "tested": True,
                    "failed_requirements": failed_requirements,
                }
            )
    counts = Counter(blocker["kind"] for entries in blockers.values() for blocker in entries)
    return {
        "by_key": {key: entries for key, entries in sorted(blockers.items())},
        "counts": dict(sorted(counts.items())),
    }


def full_isa_report(root: Path, matrix: Dict[str, Any]) -> Dict[str, Any]:
    classification = load_or_build_classification(root, matrix)
    ledger = load_promotion_ledger(root)
    promotions = ledger.get("promotions") or {}
    tooling_blockers = collect_tooling_blockers(root)
    blocker_keys = set((tooling_blockers.get("by_key") or {}).keys())
    rows = classification.get("records") or []
    gap_counts: Dict[str, Counter[str]] = {profile: Counter() for profile in PROFILE_IDS}
    blocked_counts: Dict[str, Counter[str]] = {profile: Counter() for profile in PROFILE_IDS}
    mode_totals: Counter[str] = Counter()
    coverage_totals: Counter[str] = Counter()
    unblocked_executable = 0
    blocked_executable = 0
    for row in rows:
        for profile_id, status in (row.get("profile_statuses") or {}).items():
            mode_totals[str(status.get("mode_status"))] += 1
            coverage_totals[str(status.get("coverage_status"))] += 1
            key = profile_entry_key(str(row.get("iform")), profile_id)
            if status.get("mode_status") == "executable" and key not in promotions and key in blocker_keys:
                blocked_executable += 1
                blocked_counts[profile_id][str(row.get("category") or "UNKNOWN")] += 1
            elif status.get("mode_status") == "executable" and key not in promotions:
                unblocked_executable += 1
                gap_counts[profile_id][str(row.get("category") or "UNKNOWN")] += 1
    frozen_sources = source_freeze.load_source_freeze(root) or classification.get("source_freeze") or {}
    gate_failures = [
        name
        for name, failed in {
            "source_hashes": not bool(frozen_sources.get("ok")),
            "unclassified_profile_entries": classification["summary"]["unclassified_profile_entries"] != 0,
            "no_promotions": len(promotions) == 0,
            "unimplemented_executable_entries": unblocked_executable != 0,
            "tooling_gap_entries": mode_totals.get("tooling_gap", 0) != 0,
        }.items()
        if failed
    ]
    report = {
        "schema_version": "pan-x86-full-isa-report-v1",
        "target": TARGET,
        "ok": not gate_failures,
        "summary": {
            "xed_rows": classification["summary"]["xed_rows"],
            "target_profiles": classification["summary"]["target_profiles"],
            "profile_entries": classification["summary"]["profile_entries"],
            "classified_profile_entries": classification["summary"]["classified_profile_entries"],
            "unclassified_profile_entries": classification["summary"]["unclassified_profile_entries"],
            "executable_profile_entries": classification["summary"]["executable_profile_entries"],
            "exception_profile_entries": classification["summary"]["exception_profile_entries"],
            "fully_validated_profile_entries": len(promotions),
            "blocked_executable_profile_entries": blocked_executable,
            "unimplemented_executable_entries": unblocked_executable,
            "tooling_gap_entries": mode_totals.get("tooling_gap", 0),
            "classified_non_executable_entries": coverage_totals.get("classified_non_executable", 0),
        },
        "profile_mode_status_counts": classification["summary"]["profile_mode_status_counts"],
        "profile_coverage_status_counts": classification["summary"]["profile_coverage_status_counts"],
        "mode_status_totals": dict(sorted(mode_totals.items())),
        "coverage_status_totals": dict(sorted(coverage_totals.items())),
        "source_freeze": frozen_sources,
        "promotions": promotions,
        "tested_tooling_blockers": {
            "counts": tooling_blockers.get("counts") or {},
            "sample": dict(list((tooling_blockers.get("by_key") or {}).items())[:50]),
        },
        "top_remaining_executable_gaps": {
            profile: dict(counts.most_common(20)) for profile, counts in gap_counts.items()
        },
        "top_blocked_executable_entries": {
            profile: dict(counts.most_common(20)) for profile, counts in blocked_counts.items()
        },
        "gate_failures": gate_failures,
    }
    write_json(root / REPORT_PATH, report)
    return report


def target_is_user64(target: str) -> bool:
    return target in {USER64_ALIAS, LONG64_RING3}
