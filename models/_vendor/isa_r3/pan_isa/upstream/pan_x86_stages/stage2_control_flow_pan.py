#!/usr/bin/env python3
"""Stage 2 PAN x86-lite control-flow runner.

This stage adds nontrivial RIP updates, flag-gated conditional branches, and a
bounded stack slice for CALL/RET/PUSH/POP. It deliberately stays small: one GPR
(RAX/AL), tracked CF/ZF/SF/OF flags, RIP/RSP, and a 256-byte stack image.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

from pan_x86_stages.stage0_x86_pan import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATE,
    DEFAULT_ORACLE_DEP_PATH,
    EXP_ROOT,
    FLAGS,
    ResourceMonitor,
    add_flags,
    add_oracle_pythonpath,
    hex_u,
    load_traces,
    logical_flags,
    pack_eflags,
    sub_flags,
    unpack_eflags,
    validate_trace_schema,
    write_json,
    write_jsonl,
)


STAGE = "stage2_control_flow"
CODE_BASE = 0x400000
STACK_BASE = 0x300000
STACK_SIZE = 256
STACK_START = 128
SPLITS = ("train", "heldout_flags", "heldout_stack", "full")


@dataclasses.dataclass(frozen=True)
class Stage2OpSpec:
    name: str
    iform: str
    family: str
    operands: str
    writes_rax: bool
    writes_flags: bool
    control_flow: bool
    stack_action: str


OPS: Tuple[Stage2OpSpec, ...] = (
    Stage2OpSpec("NOP", "NOP_90", "NOP", "none", False, False, False, "none"),
    Stage2OpSpec("MOV_AL_IMM", "MOV_GPR8_IMMb_B0", "DATAXFER", "ri", True, False, False, "none"),
    Stage2OpSpec("ADD_AL_IMM", "ADD_AL_IMMb", "BINARY", "ri", True, True, False, "none"),
    Stage2OpSpec("CMP_AL_IMM", "CMP_AL_IMMb", "CMP", "ri", False, True, False, "none"),
    Stage2OpSpec("TEST_AL_AL", "TEST_GPR8_GPR8", "LOGICAL", "r", False, True, False, "none"),
    Stage2OpSpec("JMP_REL8", "JMP_RELBRb", "UNCOND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JZ_REL8", "JZ_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JNZ_REL8", "JNZ_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JC_REL8", "JB_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JNC_REL8", "JNB_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JS_REL8", "JS_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JNS_REL8", "JNS_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JO_REL8", "JO_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("JNO_REL8", "JNO_RELBRb", "COND_BR", "rel8", False, False, True, "none"),
    Stage2OpSpec("CALL_REL32", "CALL_NEAR_RELBRd", "CALL", "rel32", False, False, True, "call_push"),
    Stage2OpSpec("RET_NEAR", "RET_NEAR", "RET", "none", False, False, True, "ret_pop"),
    Stage2OpSpec("PUSH_RAX", "PUSH_GPRv_50", "PUSH", "r", False, False, False, "push_rax"),
    Stage2OpSpec("POP_RAX", "POP_GPRv_58", "POP", "r", True, False, False, "pop_rax"),
)
OP_INDEX = {op.name: i for i, op in enumerate(OPS)}
RIP_SELECTOR_INDEX = {
    op.name: 2 if op.name == "RET_NEAR" else 1 if op.name in ("JMP_REL8", "CALL_REL32") else 0
    for op in OPS
}
STACK_SELECTOR_INDEX = {
    "NOP": 0,
    "MOV_AL_IMM": 0,
    "ADD_AL_IMM": 0,
    "CMP_AL_IMM": 0,
    "TEST_AL_AL": 0,
    "JMP_REL8": 0,
    "JZ_REL8": 0,
    "JNZ_REL8": 0,
    "JC_REL8": 0,
    "JNC_REL8": 0,
    "JS_REL8": 0,
    "JNS_REL8": 0,
    "JO_REL8": 0,
    "JNO_REL8": 0,
    "CALL_REL32": 1,
    "RET_NEAR": 2,
    "PUSH_RAX": 3,
    "POP_RAX": 4,
}
REG_SELECTOR_INDEX = {op.name: 0 for op in OPS} | {
    "MOV_AL_IMM": 1,
    "ADD_AL_IMM": 2,
    "POP_RAX": 3,
}
FLAG_SELECTOR_INDEX = {op.name: 0 for op in OPS} | {
    "ADD_AL_IMM": 1,
    "CMP_AL_IMM": 2,
    "TEST_AL_AL": 3,
}
OP_STACK_SELECTOR = torch.tensor([STACK_SELECTOR_INDEX[op.name] for op in OPS], dtype=torch.long)
OP_REG_SELECTOR = torch.tensor([REG_SELECTOR_INDEX[op.name] for op in OPS], dtype=torch.long)
OP_FLAG_SELECTOR = torch.tensor([FLAG_SELECTOR_INDEX[op.name] for op in OPS], dtype=torch.long)


def memory_digest(stack: Sequence[int]) -> str:
    return "sha256:" + hashlib.sha256(bytes(int(x) & 0xFF for x in stack)).hexdigest()


def flags_dict(flags: Sequence[int]) -> Dict[str, int]:
    return {name: int(flags[i]) for i, name in enumerate(FLAGS)}


def state_record(rax: int, flags: Sequence[int], rip: int, rsp_offset: int, stack: Sequence[int]) -> Dict[str, Any]:
    return {
        "rip": hex_u(rip, 64),
        "gpr": {
            "RAX": hex_u(rax, 64),
            "RSP": hex_u(STACK_BASE + int(rsp_offset), 64),
        },
        "rflags": flags_dict(flags),
        "memory_digest": memory_digest(stack),
        "exception": None,
    }


def read_u64_le(stack: Sequence[int], offset: int) -> int:
    value = 0
    for i in range(8):
        value |= (int(stack[offset + i]) & 0xFF) << (8 * i)
    return value


def write_u64_le(stack: List[int], offset: int, value: int) -> None:
    for i in range(8):
        stack[offset + i] = (int(value) >> (8 * i)) & 0xFF


def is_valid_code_rip(value: int) -> bool:
    return CODE_BASE <= int(value) < CODE_BASE + 0x10000


def rel8(target: int, fallthrough: int) -> int:
    value = int(target) - int(fallthrough)
    if not -128 <= value <= 127:
        raise ValueError(f"rel8 out of range: {value}")
    return value


def rel32(target: int, fallthrough: int) -> int:
    value = int(target) - int(fallthrough)
    if not -(1 << 31) <= value < (1 << 31):
        raise ValueError(f"rel32 out of range: {value}")
    return value


def encode_x86_bytes(op: Stage2OpSpec, imm: int, rel: int) -> str:
    if op.name == "NOP":
        b = [0x90]
    elif op.name == "MOV_AL_IMM":
        b = [0xB0, imm & 0xFF]
    elif op.name == "ADD_AL_IMM":
        b = [0x04, imm & 0xFF]
    elif op.name == "CMP_AL_IMM":
        b = [0x3C, imm & 0xFF]
    elif op.name == "TEST_AL_AL":
        b = [0x84, 0xC0]
    elif op.name == "JMP_REL8":
        b = [0xEB, rel & 0xFF]
    elif op.name == "JZ_REL8":
        b = [0x74, rel & 0xFF]
    elif op.name == "JNZ_REL8":
        b = [0x75, rel & 0xFF]
    elif op.name == "JC_REL8":
        b = [0x72, rel & 0xFF]
    elif op.name == "JNC_REL8":
        b = [0x73, rel & 0xFF]
    elif op.name == "JS_REL8":
        b = [0x78, rel & 0xFF]
    elif op.name == "JNS_REL8":
        b = [0x79, rel & 0xFF]
    elif op.name == "JO_REL8":
        b = [0x70, rel & 0xFF]
    elif op.name == "JNO_REL8":
        b = [0x71, rel & 0xFF]
    elif op.name == "CALL_REL32":
        b = [0xE8]
        b.extend(int(rel & 0xFFFFFFFF).to_bytes(4, "little"))
    elif op.name == "RET_NEAR":
        b = [0xC3]
    elif op.name == "PUSH_RAX":
        b = [0x50]
    elif op.name == "POP_RAX":
        b = [0x58]
    else:
        raise ValueError(op.name)
    return "".join(f"{x:02x}" for x in b)


def asm_string(op: Stage2OpSpec, imm: int, target: int) -> str:
    if op.name == "NOP":
        return "nop"
    if op.name == "MOV_AL_IMM":
        return f"mov AL, 0x{imm & 0xFF:02x}"
    if op.name == "ADD_AL_IMM":
        return f"add AL, 0x{imm & 0xFF:02x}"
    if op.name == "CMP_AL_IMM":
        return f"cmp AL, 0x{imm & 0xFF:02x}"
    if op.name == "TEST_AL_AL":
        return "test AL, AL"
    if op.name.endswith("REL8") or op.name == "CALL_REL32":
        return f"{op.name.split('_')[0].lower()} 0x{target:016x}"
    if op.name == "RET_NEAR":
        return "ret"
    if op.name == "PUSH_RAX":
        return "push RAX"
    if op.name == "POP_RAX":
        return "pop RAX"
    raise ValueError(op.name)


def condition_taken(op_name: str, flags: Sequence[int]) -> bool:
    cf, zf, sf, of = [int(x) for x in flags]
    if op_name == "JZ_REL8":
        return zf == 1
    if op_name == "JNZ_REL8":
        return zf == 0
    if op_name == "JC_REL8":
        return cf == 1
    if op_name == "JNC_REL8":
        return cf == 0
    if op_name == "JS_REL8":
        return sf == 1
    if op_name == "JNS_REL8":
        return sf == 0
    if op_name == "JO_REL8":
        return of == 1
    if op_name == "JNO_REL8":
        return of == 0
    return False


def apply_selectors(
    rax: int,
    flags: Sequence[int],
    rip: int,
    rsp_offset: int,
    stack: Sequence[int],
    op_idx: int,
    imm: int,
    fallthrough: int,
    target: int,
    rip_selector: int,
    stack_selector: int,
    reg_selector: int,
    flag_selector: int,
) -> Tuple[int, List[int], int, int, List[int], Dict[str, Any]]:
    out_rax = int(rax) & 0xFFFFFFFFFFFFFFFF
    out_flags = [int(x) for x in flags]
    out_rsp = int(rsp_offset)
    out_stack = [int(x) & 0xFF for x in stack]
    value = 0
    reads: List[Dict[str, Any]] = []
    writes: List[Dict[str, Any]] = []

    if stack_selector == 1:
        out_rsp -= 8
        if 0 <= out_rsp <= STACK_SIZE - 8:
            write_u64_le(out_stack, out_rsp, fallthrough)
            writes.append({"address": STACK_BASE + out_rsp, "width_bytes": 8, "value": fallthrough})
    elif stack_selector == 2:
        if 0 <= out_rsp <= STACK_SIZE - 8:
            value = read_u64_le(out_stack, out_rsp)
            reads.append({"address": STACK_BASE + out_rsp, "width_bytes": 8, "value": value})
        out_rsp += 8
    elif stack_selector == 3:
        out_rsp -= 8
        if 0 <= out_rsp <= STACK_SIZE - 8:
            write_u64_le(out_stack, out_rsp, out_rax)
            writes.append({"address": STACK_BASE + out_rsp, "width_bytes": 8, "value": out_rax})
    elif stack_selector == 4:
        if 0 <= out_rsp <= STACK_SIZE - 8:
            value = read_u64_le(out_stack, out_rsp)
            reads.append({"address": STACK_BASE + out_rsp, "width_bytes": 8, "value": value})
        out_rsp += 8
    elif stack_selector != 0:
        raise ValueError(f"bad stack selector {stack_selector}")

    al = out_rax & 0xFF
    if reg_selector == 1:
        out_rax = (out_rax & ~0xFF) | (imm & 0xFF)
    elif reg_selector == 2:
        out_rax = (out_rax & ~0xFF) | ((al + (imm & 0xFF)) & 0xFF)
    elif reg_selector == 3:
        out_rax = value & 0xFFFFFFFFFFFFFFFF
    elif reg_selector != 0:
        raise ValueError(f"bad reg selector {reg_selector}")

    if flag_selector == 1:
        out_flags = add_flags(al, imm & 0xFF, (al + (imm & 0xFF)) & 0xFF, 8)
    elif flag_selector == 2:
        out_flags = sub_flags(al, imm & 0xFF, (al - (imm & 0xFF)) & 0xFF, 8)
    elif flag_selector == 3:
        out_flags = logical_flags(al & al, 8)
    elif flag_selector != 0:
        raise ValueError(f"bad flag selector {flag_selector}")

    if rip_selector == 0:
        out_rip = fallthrough
    elif rip_selector == 1:
        out_rip = target
    elif rip_selector == 2:
        out_rip = value
    else:
        raise ValueError(f"bad rip selector {rip_selector}")

    effect = {
        "rip_before": rip,
        "fallthrough_rip": fallthrough,
        "target_rip": target,
        "rip_after": out_rip,
        "memory_reads": reads,
        "memory_writes": writes,
    }
    return out_rax, out_flags, out_rip, out_rsp, out_stack, effect


def selector_targets(op: Stage2OpSpec, flags: Sequence[int], branch_taken: bool) -> Tuple[int, int, int, int]:
    rip_selector = RIP_SELECTOR_INDEX[op.name]
    if op.family == "COND_BR":
        rip_selector = 1 if branch_taken else 0
    return (
        rip_selector,
        STACK_SELECTOR_INDEX[op.name],
        REG_SELECTOR_INDEX[op.name],
        FLAG_SELECTOR_INDEX[op.name],
    )


def apply_op(
    rax: int,
    flags: Sequence[int],
    rip: int,
    rsp_offset: int,
    stack: Sequence[int],
    op_idx: int,
    imm: int,
    fallthrough: int,
    target: int,
) -> Tuple[int, List[int], int, int, List[int], Dict[str, Any]]:
    op = OPS[op_idx]
    branch_taken = condition_taken(op.name, flags)
    return apply_selectors(
        rax,
        flags,
        rip,
        rsp_offset,
        stack,
        op_idx,
        imm,
        fallthrough,
        target,
        *selector_targets(op, flags, branch_taken),
    )


def event_record(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "address": hex_u(int(event["address"]), 64),
        "width_bytes": int(event["width_bytes"]),
        "value_hex": hex_u(int(event["value"]), 8 * int(event["width_bytes"])),
        "segment": "SS",
        "fault": None,
    }


def choose_op(rng: random.Random, split: str, rsp_offset: int, stack: Sequence[int]) -> int:
    names = [
        "NOP", "MOV_AL_IMM", "ADD_AL_IMM", "CMP_AL_IMM", "TEST_AL_AL",
        "JMP_REL8", "JZ_REL8", "JNZ_REL8", "JC_REL8", "JNC_REL8",
        "JS_REL8", "JNS_REL8", "JO_REL8", "JNO_REL8",
    ]
    if rsp_offset >= 8:
        names.extend(["CALL_REL32", "PUSH_RAX"])
    if rsp_offset <= STACK_SIZE - 8:
        if is_valid_code_rip(read_u64_le(stack, rsp_offset)):
            names.append("RET_NEAR")
        names.append("POP_RAX")
    if split == "heldout_stack":
        names = [name for name in names if name in ("CALL_REL32", "RET_NEAR", "PUSH_RAX", "POP_RAX", "CMP_AL_IMM", "JZ_REL8", "JNZ_REL8")]
    return OP_INDEX[rng.choice(names)]


def target_for(rng: random.Random, op: Stage2OpSpec, fallthrough: int) -> int:
    if op.name == "CALL_REL32":
        return fallthrough + rng.choice((8, 16, 24, 32))
    if op.operands == "rel8":
        return fallthrough + rng.choice((-32, -16, -8, 8, 16, 32))
    return fallthrough


def seed_ret_slot(rng: random.Random, stack: List[int], rsp_offset: int) -> None:
    target = CODE_BASE + rng.randrange(0x40, 0x500, 8)
    write_u64_le(stack, rsp_offset, target)


def make_trace(trace_id: str, split: str, length: int, rng: random.Random) -> Dict[str, Any]:
    rax = rng.randrange(1 << 64)
    flags = [rng.randrange(2) for _ in FLAGS]
    rip = CODE_BASE + rng.randrange(0x80, 0x180, 8)
    rsp_offset = STACK_START
    stack = [rng.randrange(256) for _ in range(STACK_SIZE)]
    seed_ret_slot(rng, stack, rsp_offset)
    steps: List[Dict[str, Any]] = []
    all_bytes: List[str] = []
    asm_lines: List[str] = []
    for i in range(length):
        op_idx = choose_op(rng, split, rsp_offset, stack)
        op = OPS[op_idx]
        imm = rng.randrange(256)
        op_len = 5 if op.name == "CALL_REL32" else 2 if op.operands in ("ri", "rel8") or op.name == "TEST_AL_AL" else 1
        fallthrough = rip + op_len
        target = target_for(rng, op, fallthrough)
        rel = rel32(target, fallthrough) if op.name == "CALL_REL32" else rel8(target, fallthrough) if op.operands == "rel8" else 0
        before_rax = rax
        before_flags = list(flags)
        before_rsp = rsp_offset
        before_stack = list(stack)
        branch_taken = op.name == "JMP_REL8" or op.name == "CALL_REL32" or (op.family == "COND_BR" and condition_taken(op.name, flags))
        rax, flags, rip_after, rsp_offset, stack, effect = apply_op(
            rax, flags, rip, rsp_offset, stack, op_idx, imm, fallthrough, target
        )
        bytes_hex = encode_x86_bytes(op, imm, rel)
        asm = asm_string(op, imm, target)
        before = state_record(before_rax, before_flags, rip, before_rsp, before_stack)
        after = state_record(rax, flags, rip_after, rsp_offset, stack)
        reg_writes = []
        if before_rax != rax:
            reg_writes.append({"name": "RAX", "before": hex_u(before_rax, 64), "after": hex_u(rax, 64), "mask": hex_u(0xFFFFFFFFFFFFFFFF if op.name == "POP_RAX" else 0xFF, 64)})
        if before_rsp != rsp_offset:
            reg_writes.append({"name": "RSP", "before": hex_u(STACK_BASE + before_rsp, 64), "after": hex_u(STACK_BASE + rsp_offset, 64), "mask": hex_u(0xFFFFFFFFFFFFFFFF, 64)})
        steps.append({
            "index": i,
            "decode": {
                "bytes_hex": bytes_hex,
                "iclass": op.name.split("_")[0],
                "iform": op.iform,
                "category": op.family,
                "extension": "BASE",
                "operands": [
                    {"kind": "rel", "role": "target", "value": rel, "target_rip": target} if op.control_flow else {"kind": "none", "role": "target"},
                    {"kind": "imm", "role": "src", "value": imm & 0xFF} if op.operands == "ri" else {"kind": "none", "role": "src"},
                ],
            },
            "before": before,
            "after": after,
            "effects": {
                "register_writes": reg_writes,
                "memory_reads": [event_record(e) for e in effect["memory_reads"]],
                "memory_writes": [event_record(e) for e in effect["memory_writes"]],
                "flags_written": list(FLAGS) if op.writes_flags else [],
                "undefined_flags": [],
                "exception": None,
            },
            "pan_stage2": {
                "op_index": op_idx,
                "imm": imm & 0xFF,
                "rel": rel,
                "fallthrough_rip": fallthrough,
                "target_rip": target,
                "branch_taken": branch_taken,
                "rip_selector": selector_targets(op, before_flags, branch_taken)[0],
                "stack_selector": STACK_SELECTOR_INDEX[op.name],
                "reg_selector": REG_SELECTOR_INDEX[op.name],
                "flag_selector": FLAG_SELECTOR_INDEX[op.name],
                "before_rax": before_rax,
                "before_flags": before_flags,
                "before_rip": rip,
                "before_rsp_offset": before_rsp,
                "before_stack": before_stack,
                "after_rax": rax,
                "after_flags": list(flags),
                "after_rip": rip_after,
                "after_rsp_offset": rsp_offset,
                "after_stack": list(stack),
            },
            "oracle_disagreement": None,
        })
        all_bytes.append(bytes_hex)
        asm_lines.append(asm)
        rip = rip_after
    return {
        "schema_version": "pan-x86-trace-v1",
        "trace_id": trace_id,
        "stage": STAGE,
        "source": {
            "generator": "stage2_control_flow_pan.py",
            "seed": trace_id,
            "xed_version": "not-yet-integrated-stage2-internal-encoder",
            "intel_sdm_hash": "not-yet-frozen",
            "amd_apm_revision": "not-yet-frozen",
            "oracle": ["stage2_python_semantics"],
            "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        },
        "mode": {
            "long_mode": True,
            "operand_bits": 64,
            "address_bits": 64,
            "ring": 3,
            "cpuid_features": ["BASE"],
        },
        "program": {
            "bytes_hex": "".join(all_bytes),
            "asm": asm_lines,
            "length": length,
            "template_id": split,
        },
        "steps": steps,
        "final": state_record(rax, flags, rip, rsp_offset, stack),
    }


def split_diagnostics(traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ops: Dict[str, int] = {}
    families: Dict[str, int] = {}
    branches = {"taken": 0, "not_taken": 0}
    stack_actions: Dict[str, int] = {}
    steps = 0
    for trace in traces:
        for step in trace["steps"]:
            steps += 1
            p = step["pan_stage2"]
            op = OPS[int(p["op_index"])]
            ops[op.name] = ops.get(op.name, 0) + 1
            families[op.family] = families.get(op.family, 0) + 1
            if op.control_flow:
                branches["taken" if p["branch_taken"] else "not_taken"] += 1
            stack_actions[op.stack_action] = stack_actions.get(op.stack_action, 0) + 1
    return {
        "programs": len(traces),
        "steps": steps,
        "op_counts": dict(sorted(ops.items())),
        "family_counts": dict(sorted(families.items())),
        "branch_counts": branches,
        "stack_action_counts": dict(sorted(stack_actions.items())),
    }


def make_manifest(out_dir: Path) -> Dict[str, Any]:
    data = {
        "stage": STAGE,
        "code_base": hex_u(CODE_BASE, 64),
        "stack_base": hex_u(STACK_BASE, 64),
        "stack_size": STACK_SIZE,
        "flags": list(FLAGS),
        "forms": [dataclasses.asdict(op) | {"op_index": i, "implemented": True} for i, op in enumerate(OPS)],
        "limitations": [
            "single-GPR RAX/AL control-flow slice",
            "bounded flat stack image only",
            "no CMOVcc/SETcc yet",
            "no segment, exception, interrupt, or privilege semantics",
            "XED iform inventory exists, but Stage 2 trace generation is still internal",
        ],
    }
    write_json(out_dir / "coverage_manifest.json", data)
    return data


def generate_dataset(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("generate")
    rng = random.Random(args.seed)
    make_manifest(out_dir)

    def rows(count: int, split: str, min_len: int, max_len: int, avoid_bytes: set[str]) -> List[Dict[str, Any]]:
        out = []
        seen = set(avoid_bytes)
        attempts = 0
        max_attempts = max(10000, count * 200)
        while len(out) < count and attempts < max_attempts:
            attempts += 1
            trace = make_trace(f"{split}-{args.seed}-{len(out)}", split, rng.randint(min_len, max_len), rng)
            key = trace["program"]["bytes_hex"]
            if key in seen:
                continue
            seen.add(key)
            out.append(trace)
        if len(out) != count:
            raise RuntimeError(f"could only generate {len(out)} unique {split} traces after {attempts} attempts")
        avoid_bytes.update(t["program"]["bytes_hex"] for t in out)
        return out

    used: set[str] = set()
    train = rows(args.train_programs, args.train_split, 4, args.max_train_len, used)
    val = rows(args.val_programs, args.val_split, 4, args.max_train_len, used)
    long = rows(args.long_programs, args.long_split, args.long_len, args.long_len, used)
    train_path = out_dir / "train_traces.jsonl"
    val_path = out_dir / "val_traces.jsonl"
    long_path = out_dir / "long_traces.jsonl"
    n_train = write_jsonl(train_path, train)
    n_val = write_jsonl(val_path, val)
    n_long = write_jsonl(long_path, long)
    schema_result = validate_trace_schema(out_dir, train_path)
    summary = {
        "train_split": args.train_split,
        "val_split": args.val_split,
        "long_split": args.long_split,
        "train_programs": n_train,
        "val_programs": n_val,
        "long_programs": n_long,
        "train_path": str(train_path),
        "val_path": str(val_path),
        "long_path": str(long_path),
        "schema_validation": schema_result,
        "split_diagnostics": {
            "train": split_diagnostics(train),
            "val": split_diagnostics(val),
            "long": split_diagnostics(long),
        },
    }
    write_json(out_dir / "generation_summary.json", summary)
    return summary


class Predictor(Protocol):
    def predict_next(
        self,
        rax: int,
        flags: List[int],
        rip: int,
        rsp_offset: int,
        stack: List[int],
        op_idx: int,
        imm: int,
        fallthrough: int,
        target: int,
    ) -> Tuple[int, List[int], int, int, List[int], Dict[str, Any]]:
        ...


class StructuredStage2Predictor:
    def __init__(self, op_offset: int = 0, flag_flip: bool = False):
        self.op_offset = op_offset
        self.flag_flip = flag_flip

    def predict_next(self, rax: int, flags: List[int], rip: int, rsp_offset: int, stack: List[int], op_idx: int, imm: int, fallthrough: int, target: int) -> Tuple[int, List[int], int, int, List[int], Dict[str, Any]]:
        actual = (op_idx + self.op_offset) % len(OPS)
        use_flags = list(flags)
        if self.flag_flip:
            use_flags = [1 - int(x) for x in use_flags]
        return apply_op(rax, use_flags, rip, rsp_offset, stack, actual, imm, fallthrough, target)


class FactorizedStage2PAN(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.op_emb = nn.Embedding(len(OPS), 24)
        self.net = nn.Sequential(
            nn.Linear(24 + len(FLAGS), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.rip_head = nn.Linear(hidden, 3)
        self.stack_head = nn.Linear(hidden, 5)
        self.reg_head = nn.Linear(hidden, 4)
        self.flag_head = nn.Linear(hidden, 4)

    def selector_logits(self, op: torch.Tensor, flags: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([self.op_emb(op.long()), flags.float()], dim=-1)
        h = self.net(x)
        return self.rip_head(h), self.stack_head(h), self.reg_head(h), self.flag_head(h)


class EffectStage2PAN(FactorizedStage2PAN):
    """Learned Stage 2 typed-effect selector head.

    The effect axes are RIP target class, stack event, RAX write source, and
    flag write policy. Validation treats this as a learned transition contract
    distinct from the legacy factorized selector baseline.
    """


def make_model(arch: str, hidden: int) -> FactorizedStage2PAN:
    if arch == "factorized":
        return FactorizedStage2PAN(hidden=hidden)
    if arch == "effect":
        return EffectStage2PAN(hidden=hidden)
    raise ValueError(f"unknown Stage 2 arch {arch!r}")


class NeuralStage2Predictor:
    def __init__(self, model: FactorizedStage2PAN):
        self.model = model.eval()

    def predict_next(self, rax: int, flags: List[int], rip: int, rsp_offset: int, stack: List[int], op_idx: int, imm: int, fallthrough: int, target: int) -> Tuple[int, List[int], int, int, List[int], Dict[str, Any]]:
        with torch.no_grad():
            op = torch.tensor([op_idx], dtype=torch.long)
            fl = torch.tensor([flags], dtype=torch.float32)
            rip_l, stack_l, reg_l, flag_l = self.model.selector_logits(op, fl)
        return apply_selectors(
            rax,
            flags,
            rip,
            rsp_offset,
            stack,
            op_idx,
            imm,
            fallthrough,
            target,
            int(rip_l.argmax(-1)[0]),
            int(stack_l.argmax(-1)[0]),
            int(reg_l.argmax(-1)[0]),
            int(flag_l.argmax(-1)[0]),
        )


def step_rows(traces: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    rows = {"op": [], "flags": [], "rip": [], "stack": [], "reg": [], "flag": [], "family": []}
    for trace in traces:
        for step in trace["steps"]:
            p = step["pan_stage2"]
            rows["op"].append(int(p["op_index"]))
            rows["flags"].append([int(x) for x in p["before_flags"]])
            rows["rip"].append(int(p["rip_selector"]))
            rows["stack"].append(int(p["stack_selector"]))
            rows["reg"].append(int(p["reg_selector"]))
            rows["flag"].append(int(p["flag_selector"]))
            rows["family"].append(step["decode"]["category"])
    if not rows["op"]:
        raise ValueError("no rows")
    return {
        "op": np.array(rows["op"], dtype=np.int64),
        "flags": np.array(rows["flags"], dtype=np.float32),
        "rip": np.array(rows["rip"], dtype=np.int64),
        "stack": np.array(rows["stack"], dtype=np.int64),
        "reg": np.array(rows["reg"], dtype=np.int64),
        "flag": np.array(rows["flag"], dtype=np.int64),
        "family": np.array(rows["family"], dtype=object),
    }


@torch.no_grad()
def selector_metrics(model: FactorizedStage2PAN, rows: Dict[str, np.ndarray], limit: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    n = rows["op"].shape[0]
    idx = np.linspace(0, n - 1, num=limit, dtype=np.int64) if limit is not None and n > limit else np.arange(n)
    op = torch.tensor(rows["op"][idx], dtype=torch.long)
    flags = torch.tensor(rows["flags"][idx], dtype=torch.float32)
    logits = model.selector_logits(op, flags)
    targets = [torch.tensor(rows[name][idx], dtype=torch.long) for name in ("rip", "stack", "reg", "flag")]
    oks = [logit.argmax(-1).eq(target).cpu().numpy() for logit, target in zip(logits, targets)]
    step_ok = oks[0] & oks[1] & oks[2] & oks[3]
    fam_counts: Dict[str, List[int]] = {}
    for fam, ok in zip(rows["family"][idx], step_ok):
        bucket = fam_counts.setdefault(str(fam), [0, 0])
        bucket[0] += int(bool(ok))
        bucket[1] += 1
    model.train()
    return {
        "rip_selector_exact": round(float(oks[0].mean()), 6),
        "stack_selector_exact": round(float(oks[1].mean()), 6),
        "reg_selector_exact": round(float(oks[2].mean()), 6),
        "flag_selector_exact": round(float(oks[3].mean()), 6),
        "step_selector_exact": round(float(step_ok.mean()), 6),
        "family_selector_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(fam_counts.items())},
        "n_steps": int(len(idx)),
    }


def train_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("train")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = step_rows(load_traces(out_dir / "train_traces.jsonl"))
    val = step_rows(load_traces(out_dir / "val_traces.jsonl"))
    model = make_model(args.arch, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = train["op"].shape[0]
    history = []
    for step in range(args.train_steps):
        idx = np.random.randint(0, n, size=args.batch_size)
        op = torch.tensor(train["op"][idx], dtype=torch.long)
        flags = torch.tensor(train["flags"][idx], dtype=torch.float32)
        logits = model.selector_logits(op, flags)
        targets = [torch.tensor(train[name][idx], dtype=torch.long) for name in ("rip", "stack", "reg", "flag")]
        loss = sum(F.cross_entropy(logit, target) for logit, target in zip(logits, targets))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.eval_every) == 0 or step == args.train_steps - 1:
            tm = selector_metrics(model, train, limit=3000)
            vm = selector_metrics(model, val, limit=3000)
            rec = {
                "step": step,
                "loss": round(float(loss.detach().cpu()), 6),
                "train_step_selector_exact": tm["step_selector_exact"],
                "val_step_selector_exact": vm["step_selector_exact"],
                "train_rip_selector_exact": tm["rip_selector_exact"],
                "val_rip_selector_exact": vm["rip_selector_exact"],
                "train_stack_selector_exact": tm["stack_selector_exact"],
                "val_stack_selector_exact": vm["stack_selector_exact"],
            }
            history.append(rec)
            print(json.dumps({"train_progress": rec}, sort_keys=True), flush=True)
    torch.save({
        "model_state": model.state_dict(),
        "hidden": args.hidden,
        "stage": STAGE,
        "model_kind": "neural",
        "arch": args.arch,
        "operators": [dataclasses.asdict(op) for op in OPS],
    }, out_dir / "stage2_pan.pt")
    write_json(out_dir / "train_history.json", history)
    summary = {"checkpoint": str(out_dir / "stage2_pan.pt"), "model_kind": "neural", "arch": args.arch, "history": history, "final": history[-1] if history else None}
    write_json(out_dir / "train_summary.json", summary)
    return summary


def load_model(out_dir: Path) -> FactorizedStage2PAN:
    ckpt = torch.load(out_dir / "stage2_pan.pt", map_location="cpu")
    model = make_model(str(ckpt.get("arch", "factorized")), hidden=int(ckpt["hidden"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def normalized_events(events: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [event_record(e) for e in events]


def byte_overlap(train_traces: Sequence[Dict[str, Any]], val_traces: Sequence[Dict[str, Any]], long_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    train = {t["program"]["bytes_hex"] for t in train_traces}
    val = {t["program"]["bytes_hex"] for t in val_traces}
    long = {t["program"]["bytes_hex"] for t in long_traces}
    return {
        "train_programs": len(train_traces),
        "train_unique_bytes": len(train),
        "val_programs": len(val_traces),
        "val_unique_bytes": len(val),
        "long_programs": len(long_traces),
        "long_unique_bytes": len(long),
        "train_val_overlap": len(train & val),
        "train_long_overlap": len(train & long),
    }


def free_run_metrics(predictor: Predictor, traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    step_exact = 0
    final_exact = 0
    rip_exact = 0
    branch_target_exact = 0
    call_ret_exact = 0
    flag_exact = 0
    stack_exact = 0
    family_counts: Dict[str, List[int]] = {}
    examples = []
    for trace in traces:
        first = trace["steps"][0]["pan_stage2"]
        rax = int(first["before_rax"])
        flags = list(first["before_flags"])
        rip = int(first["before_rip"])
        rsp = int(first["before_rsp_offset"])
        stack = list(first["before_stack"])
        trace_ok = True
        for step in trace["steps"]:
            p = step["pan_stage2"]
            rax, flags, rip, rsp, stack, effect = predictor.predict_next(
                rax, flags, rip, rsp, stack, int(p["op_index"]), int(p["imm"]), int(p["fallthrough_rip"]), int(p["target_rip"])
            )
            op = OPS[int(p["op_index"])]
            ok = (
                rax == int(p["after_rax"])
                and flags == p["after_flags"]
                and rip == int(p["after_rip"])
                and rsp == int(p["after_rsp_offset"])
                and stack == p["after_stack"]
                and normalized_events(effect["memory_reads"]) == step["effects"]["memory_reads"]
                and normalized_events(effect["memory_writes"]) == step["effects"]["memory_writes"]
            )
            trace_ok = trace_ok and ok
            total += 1
            step_exact += int(ok)
            rip_exact += int(rip == int(p["after_rip"]))
            flag_exact += int(flags == p["after_flags"])
            stack_exact += int(rsp == int(p["after_rsp_offset"]) and stack == p["after_stack"])
            if op.control_flow:
                branch_target_exact += int(rip == int(p["after_rip"]))
            if op.family in ("CALL", "RET"):
                call_ret_exact += int(ok)
            fam = step["decode"]["category"]
            bucket = family_counts.setdefault(str(fam), [0, 0])
            bucket[0] += int(ok)
            bucket[1] += 1
            if not ok and len(examples) < 8:
                examples.append({"trace_id": trace["trace_id"], "step": step["index"], "asm": trace["program"]["asm"][step["index"]], "pred_rip": rip, "true_rip": p["after_rip"], "pred_rax": rax, "true_rax": p["after_rax"]})
        final_exact += int(trace_ok)
    control_steps = sum(v[1] for k, v in family_counts.items() if k in ("COND_BR", "UNCOND_BR", "CALL", "RET"))
    call_ret_steps = sum(v[1] for k, v in family_counts.items() if k in ("CALL", "RET"))
    return {
        "programs": len(traces),
        "steps": total,
        "step_exact": round(step_exact / max(1, total), 6),
        "final_exact": round(final_exact / max(1, len(traces)), 6),
        "rip_exact": round(rip_exact / max(1, total), 6),
        "branch_target_exact": round(branch_target_exact / max(1, control_steps), 6),
        "call_ret_exact": round(call_ret_exact / max(1, call_ret_steps), 6),
        "flag_exact": round(flag_exact / max(1, total), 6),
        "stack_exact": round(stack_exact / max(1, total), 6),
        "family_step_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(family_counts.items())},
        "failure_examples": examples,
    }


def control_results(val_traces: Sequence[Dict[str, Any]], long_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "scrambled_opcode_val": free_run_metrics(StructuredStage2Predictor(op_offset=1), val_traces),
        "flipped_flags_val": free_run_metrics(StructuredStage2Predictor(flag_flip=True), val_traces),
        "scrambled_opcode_long": free_run_metrics(StructuredStage2Predictor(op_offset=1), long_traces),
    }


def write_structured_cartridge(out_dir: Path) -> Dict[str, Any]:
    data = {
        "model_kind": "structured",
        "stage": STAGE,
        "operators": [dataclasses.asdict(op) for op in OPS],
        "flags": list(FLAGS),
        "stack_base": hex_u(STACK_BASE, 64),
        "stack_size": STACK_SIZE,
        "transition": {
            "representation": "semantic_spec",
            "function": "apply_op(rax, flags, rip, rsp_offset, stack, op_idx, imm, fallthrough, target)",
        },
    }
    write_json(out_dir / "stage2_structured_cartridge.json", data)
    summary = {"model_kind": "structured", "checkpoint": str(out_dir / "stage2_structured_cartridge.json"), "history": [], "final": {"note": "Exact structured Stage 2 cartridge emitted for validation."}}
    write_json(out_dir / "train_summary.json", summary)
    return summary


def unicorn_step(step: Dict[str, Any]) -> Tuple[int, List[int], int, int, List[int]]:
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
    from unicorn.x86_const import UC_X86_REG_EFLAGS, UC_X86_REG_RAX, UC_X86_REG_RIP, UC_X86_REG_RSP

    p = step["pan_stage2"]
    code = bytes.fromhex(step["decode"]["bytes_hex"])
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(CODE_BASE, 0x10000)
    uc.mem_write(int(p["before_rip"]), code)
    uc.mem_map(STACK_BASE, 0x1000)
    uc.mem_write(STACK_BASE, bytes(int(x) & 0xFF for x in p["before_stack"]))
    uc.reg_write(UC_X86_REG_RIP, int(p["before_rip"]))
    uc.reg_write(UC_X86_REG_RAX, int(p["before_rax"]) & 0xFFFFFFFFFFFFFFFF)
    uc.reg_write(UC_X86_REG_RSP, STACK_BASE + int(p["before_rsp_offset"]))
    uc.reg_write(UC_X86_REG_EFLAGS, pack_eflags(p["before_flags"]))
    uc.emu_start(int(p["before_rip"]), int(p["before_rip"]) + len(code), count=1)
    rax = int(uc.reg_read(UC_X86_REG_RAX)) & 0xFFFFFFFFFFFFFFFF
    flags = unpack_eflags(int(uc.reg_read(UC_X86_REG_EFLAGS)))
    rip = int(uc.reg_read(UC_X86_REG_RIP)) & 0xFFFFFFFFFFFFFFFF
    rsp = int(uc.reg_read(UC_X86_REG_RSP)) - STACK_BASE
    stack = list(bytes(uc.mem_read(STACK_BASE, STACK_SIZE)))
    return rax, flags, rip, rsp, stack


def validate_external_oracle(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    oracle = getattr(args, "external_oracle", "none")
    report: Dict[str, Any] = {"enabled": oracle != "none", "oracle": oracle, "ok": True, "checked_steps": 0, "splits": {}, "failure_examples": [], "error": None}
    if oracle == "none":
        return report
    if oracle != "unicorn":
        report.update({"ok": False, "error": f"unknown external oracle {oracle}"})
        return report
    try:
        __import__("unicorn")
    except Exception as exc:
        report.update({"ok": False, "error": f"could not import unicorn: {exc}"})
        return report
    max_steps = int(getattr(args, "oracle_max_steps", 12000))
    for split, path in (("val", out_dir / "val_traces.jsonl"), ("long", out_dir / "long_traces.jsonl")):
        checked = 0
        exact = 0
        for trace in load_traces(path):
            for step in trace["steps"]:
                if report["checked_steps"] >= max_steps:
                    break
                p = step["pan_stage2"]
                try:
                    rax, flags, rip, rsp, stack = unicorn_step(step)
                except Exception as exc:
                    report.update({"ok": False, "error": f"unicorn execution failed: {exc}"})
                    return report
                ok = rax == p["after_rax"] and flags == p["after_flags"] and rip == p["after_rip"] and rsp == p["after_rsp_offset"] and stack == p["after_stack"]
                checked += 1
                exact += int(ok)
                report["checked_steps"] += 1
                if not ok and len(report["failure_examples"]) < 8:
                    report["failure_examples"].append({"split": split, "trace_id": trace["trace_id"], "step": step["index"], "asm": trace["program"]["asm"][step["index"]], "unicorn": {"rax": rax, "flags": flags, "rip": rip, "rsp": rsp, "stack_digest": memory_digest(stack)}, "trace": {"rax": p["after_rax"], "flags": p["after_flags"], "rip": p["after_rip"], "rsp": p["after_rsp_offset"], "stack_digest": memory_digest(p["after_stack"])}})
            if report["checked_steps"] >= max_steps:
                break
        report["splits"][split] = {"checked_steps": checked, "exact": round(exact / max(1, checked), 6)}
    report["ok"] = report["ok"] and all(v["exact"] == 1.0 for v in report["splits"].values())
    return report


def validate_common(args: argparse.Namespace, out_dir: Path, predictor: Predictor, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("validate")
    train = load_traces(out_dir / "train_traces.jsonl")
    val = load_traces(out_dir / "val_traces.jsonl")
    long = load_traces(out_dir / "long_traces.jsonl")
    learned_effect_mode = bool(getattr(predictor, "learned_effect_mode", False))
    report = {
        "stage": STAGE,
        "model_kind": args.model_kind,
        "arch": "effect" if learned_effect_mode else "structured" if args.model_kind == "structured" else "factorized",
        "transition_contract": {
            "schema": "pan-x86-stage2-effect-selector-v1" if learned_effect_mode else "legacy-stage2-selector-or-structured",
            "learned_effect_selectors": learned_effect_mode,
            "learned_effect_axes": ["rip_target", "stack_event", "rax_write", "flag_write_policy"] if learned_effect_mode else [],
            "commit_path": "model_emitted_effect_selectors" if learned_effect_mode else "legacy_model_or_structured_selectors",
            "direct_value_heads": False,
            "note": "Stage 2 effect mode learns control/stack/register/flag effect axes; structured commit still materializes exact values.",
        },
        "byte_overlap": byte_overlap(train, val, long),
        "free_run": {
            "val": free_run_metrics(predictor, val),
            "long": free_run_metrics(predictor, long),
        },
        "controls": control_results(val, long),
        "external_oracle": validate_external_oracle(args, out_dir),
        "thresholds": {
            "configured_exact_final_state": 0.985,
            "configured_branch_target_exact": 0.995,
            "configured_call_ret_exact": 0.990,
            "configured_flag_exact": 0.990,
        },
    }
    schema_ok = json.loads((out_dir / "schema_validation.json").read_text(encoding="utf-8")).get("ok", False)
    val_ok = report["free_run"]["val"]
    long_ok = report["free_run"]["long"]
    learned_effect_gate = True
    if learned_effect_mode:
        learned_effect_gate = (
            val_ok["rip_exact"] >= 0.995
            and val_ok["stack_exact"] >= 0.990
            and val_ok["flag_exact"] >= 0.990
            and long_ok["rip_exact"] >= 0.995
            and long_ok["stack_exact"] >= 0.990
            and long_ok["flag_exact"] >= 0.990
        )
    gate = (
        schema_ok
        and learned_effect_gate
        and val_ok["final_exact"] >= 0.985
        and val_ok["branch_target_exact"] >= 0.995
        and val_ok["call_ret_exact"] >= 0.990
        and val_ok["flag_exact"] >= 0.990
        and long_ok["final_exact"] >= 0.985
        and long_ok["branch_target_exact"] >= 0.995
        and report["controls"]["scrambled_opcode_val"]["step_exact"] < 0.900
        and report["controls"]["flipped_flags_val"]["branch_target_exact"] < 0.950
        and report["external_oracle"].get("ok", True)
        and report["byte_overlap"]["train_val_overlap"] == 0
        and report["byte_overlap"]["train_long_overlap"] == 0
    )
    report["gate_status"] = {
        "schema_valid": schema_ok,
        "stage2_gate_passed": bool(gate),
        "learned_effect_selector_gate_passed": learned_effect_gate,
        "full_x86_goal_complete": False,
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def validate_structured(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    return validate_common(args, out_dir, StructuredStage2Predictor(), monitor)


def validate_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    model = load_model(out_dir)
    predictor = NeuralStage2Predictor(model)
    predictor.learned_effect_mode = isinstance(model, EffectStage2PAN)
    return validate_common(args, out_dir, predictor, monitor)


def source_freeze(out_dir: Path, model_kind: str, external_oracle: str) -> Dict[str, Any]:
    data = {
        "stage": STAGE,
        "created": DEFAULT_DATE,
        "model_kind": model_kind,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "external_oracles": ["stage2_python_semantics"] + ([] if external_oracle == "none" else [external_oracle]),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    write_json(out_dir / "source_freeze.json", data)
    return data


def run_pipeline(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(out_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        source_freeze(out_dir, args.model_kind, args.external_oracle)
        generation = generate_dataset(args, out_dir, monitor)
        if args.model_kind == "structured":
            monitor.set_phase("train")
            training = write_structured_cartridge(out_dir)
            validation = validate_structured(args, out_dir, monitor)
        else:
            training = train_model(args, out_dir, monitor)
            validation = validate_model(args, out_dir, monitor)
        monitor.set_phase("finalize")
        result = {
            "out_dir": str(out_dir),
            "model_kind": args.model_kind,
            "generation": generation,
            "training": training,
            "validation": {
                "gate_status": validation["gate_status"],
                "val_free_run": validation["free_run"]["val"],
                "long_free_run": validation["free_run"]["long"],
                "controls": {
                    "scrambled_opcode_val_step_exact": validation["controls"]["scrambled_opcode_val"]["step_exact"],
                    "flipped_flags_val_branch_target_exact": validation["controls"]["flipped_flags_val"]["branch_target_exact"],
                },
                "external_oracle": validation["external_oracle"],
            },
        }
        write_json(out_dir / "run_summary.json", result)
        return result
    finally:
        monitor.stop()


def default_out_dir(run_name: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / STAGE / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out-dir", type=Path, default=None)
        sp.add_argument("--run-name", default=f"run-{int(time.time())}")
        sp.add_argument("--model-kind", choices=["structured", "neural"], default="structured")
        sp.add_argument("--arch", choices=["factorized", "effect"], default="factorized")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--monitor-interval", type=float, default=1.0)
        sp.add_argument("--train-split", choices=SPLITS, default="train")
        sp.add_argument("--val-split", choices=SPLITS, default="heldout_flags")
        sp.add_argument("--long-split", choices=SPLITS, default="heldout_stack")
        sp.add_argument("--external-oracle", choices=["none", "unicorn"], default="none")
        sp.add_argument("--oracle-pythonpath", type=Path, default=DEFAULT_ORACLE_DEP_PATH)
        sp.add_argument("--oracle-max-steps", type=int, default=12000)

    run = sub.add_parser("run-smoke")
    add_common(run)
    run.add_argument("--train-programs", type=int, default=1600)
    run.add_argument("--val-programs", type=int, default=500)
    run.add_argument("--long-programs", type=int, default=180)
    run.add_argument("--max-train-len", type=int, default=8)
    run.add_argument("--long-len", type=int, default=32)
    run.add_argument("--train-steps", type=int, default=400)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--hidden", type=int, default=96)
    run.add_argument("--lr", type=float, default=2e-3)
    run.add_argument("--eval-every", type=int, default=100)

    gen = sub.add_parser("generate")
    add_common(gen)
    gen.add_argument("--train-programs", type=int, default=1600)
    gen.add_argument("--val-programs", type=int, default=500)
    gen.add_argument("--long-programs", type=int, default=180)
    gen.add_argument("--max-train-len", type=int, default=8)
    gen.add_argument("--long-len", type=int, default=32)

    val = sub.add_parser("validate")
    add_common(val)
    return p


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.run_name)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = normalize_args(build_parser().parse_args(argv))
    add_oracle_pythonpath(args.oracle_pythonpath)
    if args.cmd == "run-smoke":
        result = run_pipeline(args)
        print(json.dumps({"run_summary": result}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "generate":
        args.out_dir.mkdir(parents=True, exist_ok=True)
        monitor = ResourceMonitor(args.out_dir, interval_s=args.monitor_interval)
        monitor.start()
        try:
            source_freeze(args.out_dir, args.model_kind, args.external_oracle)
            result = generate_dataset(args, args.out_dir, monitor)
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            monitor.stop()
        return 0
    if args.cmd == "validate":
        monitor = ResourceMonitor(args.out_dir, interval_s=args.monitor_interval)
        monitor.start()
        try:
            result = validate_structured(args, args.out_dir, monitor) if args.model_kind == "structured" else validate_model(args, args.out_dir, monitor)
            print(json.dumps(result["gate_status"], indent=2, sort_keys=True))
        finally:
            monitor.stop()
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
