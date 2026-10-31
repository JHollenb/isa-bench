"""Deterministic operand-policy generation for full-ISA shards."""
from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List

from pan_x86_isa.profiles import profile_policy, resolve_profile


TRACE_SPLITS = ("smoke", "train", "validation", "long", "exception", "negative-control")
SPLIT_OFFSETS = {
    "smoke": 0,
    "train": 17,
    "validation": 37,
    "long": 59,
    "exception": 83,
    "negative-control": 109,
}


def stable_policy_id(policy: Dict[str, Any]) -> str:
    payload = json.dumps(policy, sort_keys=True, separators=(",", ":"))
    return "op-policy-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def stable_case_id(case: Dict[str, Any]) -> str:
    payload = json.dumps(case, sort_keys=True, separators=(",", ":"))
    return "op-case-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _register_pool(bits: int) -> List[str]:
    if bits == 16:
        return ["ax", "bx", "cx", "dx", "si", "di", "bp", "sp"]
    if bits == 32:
        return ["eax", "ebx", "ecx", "edx", "esi", "edi", "ebp", "esp"]
    return ["rax", "rbx", "rcx", "rdx", "rsi", "rdi", "rbp", "rsp"]


def _memory_axes(profile: Dict[str, Any]) -> Dict[str, Any]:
    bits = int(profile["address_bits"])
    base_regs = _register_pool(32 if bits <= 32 else 64)
    return {
        "base_registers": base_regs[:4],
        "index_registers": base_regs[1:5],
        "scales": [1, 2, 4, 8] if bits >= 32 else [1],
        "displacements": [0, 1, 7, 8, 15, 16, 127, 128],
        "alignments": ["aligned", "misaligned"],
        "sandbox_base": profile["state_defaults"]["memory_sandbox_base"],
        "sandbox_size": profile["state_defaults"]["memory_sandbox_size"],
    }


def _operand_shapes(iform: str) -> List[str]:
    if "MEM" in iform:
        if iform.startswith("MEM"):
            return ["memory/register", "memory/immediate"]
        return ["register/memory"]
    if "IMM" in iform:
        return ["register/immediate"]
    if any(token in iform for token in ("XMM", "YMM", "ZMM", "GPR", "MMX", "ST")):
        return ["register/register"]
    return ["implicit"]


def build_operand_policy(row: Dict[str, Any], target_profile: str) -> Dict[str, Any]:
    profile_id = resolve_profile(target_profile)
    profile = profile_policy(profile_id)
    iform = str(row.get("iform") or "")
    iclass = str(row.get("iclass") or "")
    category = str(row.get("category") or "")
    bits = int(profile["operand_bits"])
    policy: Dict[str, Any] = {
        "schema_version": "pan-x86-operand-policy-v1",
        "target_profile": profile_id,
        "iform": iform,
        "iclass": iclass,
        "category": category,
        "registers": {
            "gpr": _register_pool(bits),
            "xmm": [f"xmm{i}" for i in range(8 if bits < 64 else 16)],
            "ymm": [f"ymm{i}" for i in range(8 if bits < 64 else 16)],
            "zmm": [f"zmm{i}" for i in range(32)],
            "mmx": [f"mm{i}" for i in range(8)],
            "x87": [f"st{i}" for i in range(8)],
        },
        "immediates": [],
        "memory": None,
        "operand_shapes": _operand_shapes(iform),
        "prefixes": {
            "lock": "LOCK" in iform or "LOCK" in iclass,
            "rep": any(token in iform or token in iclass for token in ("REP", "REPE", "REPNE")),
            "operand_size_override": bits in (16, 32),
            "address_size_override": int(profile["address_bits"]) != bits,
        },
        "addressing_modes": [],
        "alignment_variants": ["natural", "unaligned"],
        "fault_axes": [],
        "environment_axes": [],
        "splits": {
            split: {
                "purpose": {
                    "smoke": "small deterministic command smoke",
                    "train": "model fitting",
                    "validation": "held-out exactness gate",
                    "long": "long/free-run exactness gate",
                    "exception": "fault and descriptor paths",
                    "negative-control": "must degrade under corrupted bytes or operands",
                }[split],
                "byte_disjoint_offset": SPLIT_OFFSETS[split],
            }
            for split in TRACE_SPLITS
        },
    }
    if "IMM" in iform:
        policy["immediates"] = [0, 1, -1, 0x7F, 0x80, 0xFF, 0xFFFF, 0xFFFFFFFF]
    if "MEM" in iform or "AGEN" in category:
        policy["memory"] = _memory_axes(profile)
        policy["addressing_modes"] = ["base", "base_index_scale", "disp_only"]
        policy["fault_axes"].extend(["alignment", "page_boundary", "canonical_address" if bits == 64 else "segment_limit"])
    if any(token in iform for token in ("DIV", "IDIV")):
        policy["fault_axes"].append("divide_error")
    if category in {"SYSTEM", "SYSCALL", "SYSRET", "VTX", "SVM"}:
        policy["fault_axes"].extend(["privilege", "descriptor", "control_register"])
    if iclass in {"CPUID", "RDRAND", "RDSEED", "RDTSC", "RDTSCP"}:
        policy["environment_axes"].append("host_visible_state")
    policy["policy_id"] = stable_policy_id(policy)
    return policy


def _pick(pool: List[str], index: int) -> str:
    if not pool:
        return ""
    return pool[index % len(pool)]


def _register_operands(policy: Dict[str, Any], index: int) -> Dict[str, str]:
    regs = policy.get("registers") or {}
    if "XMM" in str(policy.get("iform") or ""):
        pool = list(regs.get("xmm") or [])
    elif "YMM" in str(policy.get("iform") or ""):
        pool = list(regs.get("ymm") or [])
    elif "ZMM" in str(policy.get("iform") or ""):
        pool = list(regs.get("zmm") or [])
    elif "MM" in str(policy.get("iform") or ""):
        pool = list(regs.get("mmx") or [])
    else:
        pool = list(regs.get("gpr") or [])
    return {"dst": _pick(pool, index), "src": _pick(pool, index + 1)}


def _memory_operands(policy: Dict[str, Any], index: int) -> Dict[str, Any]:
    memory = policy.get("memory") or {}
    base = _pick(list(memory.get("base_registers") or []), index)
    index_reg = _pick(list(memory.get("index_registers") or []), index + 1)
    scales = list(memory.get("scales") or [1])
    displacements = list(memory.get("displacements") or [0])
    return {
        "base": base,
        "index": index_reg,
        "scale": scales[index % len(scales)],
        "disp": displacements[index % len(displacements)],
        "alignment": _pick(list(memory.get("alignments") or ["aligned"]), index),
        "sandbox_base": memory.get("sandbox_base"),
    }


def build_operand_cases(
    row: Dict[str, Any],
    target_profile: str,
    *,
    cpuid_profile: str = "base",
    split: str = "smoke",
    cases_per_iform: int = 1,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    if split not in TRACE_SPLITS:
        raise ValueError(f"unknown trace split {split!r}; choose one of: {', '.join(TRACE_SPLITS)}")
    profile_id = resolve_profile(target_profile)
    profile = profile_policy(profile_id)
    policy = build_operand_policy(row, profile_id)
    rng = random.Random(f"{policy['policy_id']}:{split}:{seed}:{cpuid_profile}")
    count = max(0, int(cases_per_iform))
    cases: List[Dict[str, Any]] = []
    offset = SPLIT_OFFSETS[split]
    for case_index in range(count):
        variant = offset + case_index * 3 + rng.randrange(0, 3)
        shape = _pick(list(policy.get("operand_shapes") or ["implicit"]), variant)
        operand_values: Dict[str, Any] = {}
        if "register" in shape:
            operand_values.update(_register_operands(policy, variant))
        if "memory" in shape:
            operand_values["memory"] = _memory_operands(policy, variant)
            if "dst" not in operand_values:
                operand_values["dst"] = operand_values["memory"]
            elif "src" not in operand_values:
                operand_values["src"] = operand_values["memory"]
        if "immediate" in shape:
            immediates = list(policy.get("immediates") or [0])
            operand_values["imm"] = immediates[variant % len(immediates)]
        if split == "exception":
            fault_axes = list(policy.get("fault_axes") or ["forced_invalid_encoding"])
            operand_values["fault_axis"] = fault_axes[case_index % len(fault_axes)]
        if split == "negative-control":
            operand_values["negative_control"] = "opcode_byte_flip"
        case = {
            "schema_version": "pan-x86-operand-case-v1",
            "target_profile": profile_id,
            "cpuid_profile": cpuid_profile,
            "iform": row.get("iform"),
            "operand_policy_id": policy["policy_id"],
            "split": split,
            "case_index": case_index,
            "case_kind": shape,
            "operand_values": operand_values,
            "prefix_variant": {
                "lock": bool(policy.get("prefixes", {}).get("lock") and case_index % 2 == 1),
                "rep": bool(policy.get("prefixes", {}).get("rep") and case_index % 3 == 1),
                "operand_size_override": bool(policy.get("prefixes", {}).get("operand_size_override") and split != "smoke"),
                "address_size_override": bool(policy.get("prefixes", {}).get("address_size_override") and split in {"validation", "long"}),
            },
            "mode_bits": {
                "operand_bits": profile["operand_bits"],
                "address_bits": profile["address_bits"],
                "long_mode": profile["long_mode"],
                "ring": profile["ring"],
            },
        }
        case["operand_case_id"] = stable_case_id(case)
        cases.append(case)
    return cases
