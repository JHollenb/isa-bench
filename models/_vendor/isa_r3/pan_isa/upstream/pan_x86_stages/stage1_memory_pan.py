#!/usr/bin/env python3
"""Stage 1 PAN x86-lite memory/addressing runner.

This validates the next cartridge boundary after Stage 0: effective addresses,
typed memory events, load/store effects, LEA, and XCHG over a small tracked
memory image. The structured cartridge remains the oracle baseline; neural
`--arch effect` runs learn typed value/memory effect selectors and validation
commits only the model-emitted effect selectors.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

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
    ResourceMonitor,
    add_oracle_pythonpath,
    hex_u,
    load_traces,
    validate_trace_schema,
    write_json,
    write_jsonl,
)


STAGE = "stage1_address_memory"
MEM_BASE = 0x200000
MEM_SIZE = 256
DATA_REGS = ("RAX", "RCX", "RDX", "RBX")
DATA_BYTES = ("AL", "CL", "DL", "BL")
DATA_WORDS = ("AX", "CX", "DX", "BX")
DATA_DWORDS = ("EAX", "ECX", "EDX", "EBX")
ADDR_REGS = ("RSI", "RDI", "RBP")
DATA_REG_CODES = {"RAX": 0, "RCX": 1, "RDX": 2, "RBX": 3}
ADDR_RM_CODES = {"RSI": 6, "RDI": 7, "RBP": 5}
INDEX_REGS = ("R8", "R9")
INDEX_RM_CODES = {"R8": 0, "R9": 1}
STAGE1_SPLITS = ("train", "heldout_disp", "heldout_base", "full")


@dataclasses.dataclass(frozen=True)
class Stage1OpSpec:
    name: str
    iform: str
    family: str
    operands: str
    writes_dst: bool
    reads_memory: bool
    writes_memory: bool
    width_bytes: int


OPS: Tuple[Stage1OpSpec, ...] = (
    Stage1OpSpec("MOV_LOAD_RM", "MOV_GPR8_MEMb", "LOAD", "r_m", True, True, False, 1),
    Stage1OpSpec("MOV_STORE_MR", "MOV_MEMb_GPR8", "STORE", "m_r", False, False, True, 1),
    Stage1OpSpec("MOV_STORE_MI", "MOV_MEMb_IMMb", "STORE", "m_i", False, False, True, 1),
    Stage1OpSpec("LEA_RA", "LEA_GPR64_AGEN", "LEA", "r_a", True, False, False, 8),
    Stage1OpSpec("XCHG_RM", "XCHG_GPR8_MEMb", "XCHG", "r_m", True, True, True, 1),
    Stage1OpSpec("MOV_LOAD_RM16", "MOV_GPR16_MEMw", "LOAD", "r_m", True, True, False, 2),
    Stage1OpSpec("MOV_LOAD_RM32", "MOV_GPR32_MEMd", "LOAD", "r_m", True, True, False, 4),
    Stage1OpSpec("MOV_LOAD_RM64", "MOV_GPR64_MEMq", "LOAD", "r_m", True, True, False, 8),
    Stage1OpSpec("MOV_STORE_MR16", "MOV_MEMw_GPR16", "STORE", "m_r", False, False, True, 2),
    Stage1OpSpec("MOV_STORE_MR32", "MOV_MEMd_GPR32", "STORE", "m_r", False, False, True, 4),
    Stage1OpSpec("MOV_STORE_MR64", "MOV_MEMq_GPR64", "STORE", "m_r", False, False, True, 8),
    Stage1OpSpec("MOV_STORE_MI16", "MOV_MEMw_IMMw", "STORE", "m_i", False, False, True, 2),
    Stage1OpSpec("MOV_STORE_MI32", "MOV_MEMd_IMMz", "STORE", "m_i", False, False, True, 4),
    Stage1OpSpec("XCHG_RM16", "XCHG_GPR16_MEMw", "XCHG", "r_m", True, True, True, 2),
    Stage1OpSpec("XCHG_RM32", "XCHG_GPR32_MEMd", "XCHG", "r_m", True, True, True, 4),
    Stage1OpSpec("XCHG_RM64", "XCHG_GPR64_MEMq", "XCHG", "r_m", True, True, True, 8),
)
VALUE_SELECTOR_INDEX = {
    op.name: (2 if op.family == "LEA" else 1 if op.reads_memory else 0)
    for op in OPS
}
MEMORY_SELECTOR_INDEX = {
    op.name: (
        3 if op.family == "XCHG"
        else 2 if op.operands == "m_i"
        else 1 if op.operands == "m_r"
        else 0
    )
    for op in OPS
}
OP_VALUE_SELECTOR = torch.tensor([VALUE_SELECTOR_INDEX[op.name] for op in OPS], dtype=torch.long)
OP_MEMORY_SELECTOR = torch.tensor([MEMORY_SELECTOR_INDEX[op.name] for op in OPS], dtype=torch.long)


def memory_digest(memory: Sequence[int]) -> str:
    data = bytes(int(x) & 0xFF for x in memory)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def byte_hex(value: int) -> str:
    return "0x" + format(int(value) & 0xFF, "02x")


def width_hex(value: int, width_bytes: int) -> str:
    bits = 8 * int(width_bytes)
    mask = (1 << bits) - 1
    return "0x" + format(int(value) & mask, f"0{int(width_bytes) * 2}x")


def full_address(offset: int) -> int:
    return MEM_BASE + int(offset)


def reg_name_for_width(index: int, width_bytes: int) -> str:
    if width_bytes == 1:
        return DATA_BYTES[index]
    if width_bytes == 2:
        return DATA_WORDS[index]
    if width_bytes == 4:
        return DATA_DWORDS[index]
    if width_bytes == 8:
        return DATA_REGS[index]
    raise ValueError(f"unsupported register width: {width_bytes}")


def ptr_name_for_width(width_bytes: int) -> str:
    return {
        1: "byte ptr",
        2: "word ptr",
        4: "dword ptr",
        8: "qword ptr",
    }[int(width_bytes)]


def width_mask(width_bytes: int) -> int:
    return (1 << (8 * int(width_bytes))) - 1


def read_reg_width(value: int, width_bytes: int) -> int:
    return int(value) & width_mask(width_bytes)


def write_reg_width(old: int, value: int, width_bytes: int) -> int:
    width_bytes = int(width_bytes)
    value &= width_mask(width_bytes)
    if width_bytes == 8:
        return value
    if width_bytes == 4:
        return value
    mask = width_mask(width_bytes)
    return (int(old) & ~mask) | value


def register_write_mask(width_bytes: int) -> int:
    if int(width_bytes) in (4, 8):
        return 0xFFFFFFFFFFFFFFFF
    return width_mask(width_bytes)


def read_mem_le(memory: Sequence[int], offset: int, width_bytes: int) -> int:
    offset = int(offset)
    width_bytes = int(width_bytes)
    value = 0
    for i in range(width_bytes):
        value |= (int(memory[offset + i]) & 0xFF) << (8 * i)
    return value


def write_mem_le(memory: List[int], offset: int, width_bytes: int, value: int) -> None:
    offset = int(offset)
    width_bytes = int(width_bytes)
    for i in range(width_bytes):
        memory[offset + i] = (int(value) >> (8 * i)) & 0xFF


def state_record(
    data_regs: Sequence[int],
    addr_offsets: Sequence[int],
    memory: Sequence[int],
    rip: int,
    index_offsets: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    gpr = {name: hex_u(int(data_regs[i]), 64) for i, name in enumerate(DATA_REGS)}
    for i, name in enumerate(ADDR_REGS):
        gpr[name] = hex_u(full_address(int(addr_offsets[i])), 64)
    if index_offsets is not None:
        for i, name in enumerate(INDEX_REGS):
            gpr[name] = hex_u(int(index_offsets[i]), 64)
    return {
        "rip": hex_u(rip, 64),
        "gpr": gpr,
        "rflags": {},
        "memory_digest": memory_digest(memory),
        "exception": None,
    }


def modrm_disp8(reg: int, rm: int) -> int:
    return 0x40 | ((reg & 7) << 3) | (rm & 7)


def has_index(index: int) -> bool:
    return int(index) >= 0


def scale_bits(scale: int) -> int:
    mapping = {1: 0, 2: 1, 4: 2, 8: 3}
    if int(scale) not in mapping:
        raise ValueError(f"unsupported index scale: {scale}")
    return mapping[int(scale)]


def rex_prefix(width: int, index: int) -> List[int]:
    value = 0x40
    if int(width) == 8:
        value |= 0x08
    if has_index(index):
        value |= 0x02
    return [] if value == 0x40 else [value]


def opcode_prefix(width: int, index: int) -> List[int]:
    out: List[int] = []
    if int(width) == 2:
        out.append(0x66)
    out.extend(rex_prefix(width, index))
    return out


def address_bytes(reg: int, base: int, index: int, scale: int, disp: int) -> List[int]:
    base_code = ADDR_RM_CODES[ADDR_REGS[base]]
    if not has_index(index):
        return [modrm_disp8(reg, base_code), disp & 0xFF]
    index_code = INDEX_RM_CODES[INDEX_REGS[index]]
    sib = (scale_bits(scale) << 6) | ((index_code & 7) << 3) | (base_code & 7)
    return [modrm_disp8(reg, 4), sib, disp & 0xFF]


def encode_x86_bytes(op: Stage1OpSpec, dst: int, src: int, base: int, index: int, scale: int, disp: int, imm: int) -> str:
    width = int(op.width_bytes)
    reg = DATA_REG_CODES[DATA_REGS[dst]]
    src_reg = DATA_REG_CODES[DATA_REGS[src]]
    if op.name.startswith("MOV_LOAD"):
        if width == 1:
            b = opcode_prefix(width, index) + [0x8A] + address_bytes(reg, base, index, scale, disp)
        elif width == 2:
            b = opcode_prefix(width, index) + [0x8B] + address_bytes(reg, base, index, scale, disp)
        elif width == 4:
            b = opcode_prefix(width, index) + [0x8B] + address_bytes(reg, base, index, scale, disp)
        elif width == 8:
            b = opcode_prefix(width, index) + [0x8B] + address_bytes(reg, base, index, scale, disp)
        else:
            raise ValueError(op.name)
    elif op.name.startswith("MOV_STORE_MR"):
        if width == 1:
            b = opcode_prefix(width, index) + [0x88] + address_bytes(src_reg, base, index, scale, disp)
        elif width == 2:
            b = opcode_prefix(width, index) + [0x89] + address_bytes(src_reg, base, index, scale, disp)
        elif width == 4:
            b = opcode_prefix(width, index) + [0x89] + address_bytes(src_reg, base, index, scale, disp)
        elif width == 8:
            b = opcode_prefix(width, index) + [0x89] + address_bytes(src_reg, base, index, scale, disp)
        else:
            raise ValueError(op.name)
    elif op.name.startswith("MOV_STORE_MI"):
        if width == 1:
            b = opcode_prefix(width, index) + [0xC6] + address_bytes(0, base, index, scale, disp) + [imm & 0xFF]
        elif width == 2:
            b = opcode_prefix(width, index) + [0xC7] + address_bytes(0, base, index, scale, disp)
            b.extend(int(imm & 0xFFFF).to_bytes(2, "little"))
        elif width == 4:
            b = opcode_prefix(width, index) + [0xC7] + address_bytes(0, base, index, scale, disp)
            b.extend(int(imm & 0xFFFFFFFF).to_bytes(4, "little"))
        else:
            raise ValueError(op.name)
    elif op.name == "LEA_RA":
        b = opcode_prefix(8, index) + [0x8D] + address_bytes(DATA_REG_CODES[DATA_REGS[dst]], base, index, scale, disp)
    elif op.name.startswith("XCHG_RM"):
        if width == 1:
            b = opcode_prefix(width, index) + [0x86] + address_bytes(reg, base, index, scale, disp)
        elif width == 2:
            b = opcode_prefix(width, index) + [0x87] + address_bytes(reg, base, index, scale, disp)
        elif width == 4:
            b = opcode_prefix(width, index) + [0x87] + address_bytes(reg, base, index, scale, disp)
        elif width == 8:
            b = opcode_prefix(width, index) + [0x87] + address_bytes(reg, base, index, scale, disp)
        else:
            raise ValueError(op.name)
    else:
        raise ValueError(op.name)
    return "".join(f"{x:02x}" for x in b)


def address_expr(base: int, index: int, scale: int, disp: int) -> str:
    parts = [ADDR_REGS[base]]
    if has_index(index):
        term = INDEX_REGS[index]
        if int(scale) != 1:
            term = f"{term}*{int(scale)}"
        parts.append(term)
    parts.append(f"0x{disp:02x}")
    return " + ".join(parts)


def asm_string(op: Stage1OpSpec, dst: int, src: int, base: int, index: int, scale: int, disp: int, imm: int) -> str:
    width = int(op.width_bytes)
    mem = f"{ptr_name_for_width(width)} [{address_expr(base, index, scale, disp)}]"
    if op.name.startswith("MOV_LOAD"):
        return f"mov {reg_name_for_width(dst, width)}, {mem}"
    if op.name.startswith("MOV_STORE_MR"):
        return f"mov {mem}, {reg_name_for_width(src, width)}"
    if op.name.startswith("MOV_STORE_MI"):
        return f"mov {mem}, {width_hex(imm, width)}"
    if op.name == "LEA_RA":
        return f"lea {DATA_REGS[dst]}, [{address_expr(base, index, scale, disp)}]"
    if op.name.startswith("XCHG_RM"):
        return f"xchg {reg_name_for_width(dst, width)}, {mem}"
    raise ValueError(op.name)


def apply_op(
    data_regs: Sequence[int],
    addr_offsets: Sequence[int],
    index_offsets: Sequence[int],
    memory: Sequence[int],
    op_idx: int,
    dst: int,
    src: int,
    base: int,
    index: int,
    scale: int,
    disp: int,
    imm: int,
    alias_offset: int = 0,
    op_offset: int = 0,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    actual_op = OPS[(op_idx + op_offset) % len(OPS)]
    width = int(actual_op.width_bytes)
    out_data = [int(x) & 0xFFFFFFFFFFFFFFFF for x in data_regs]
    out_addr = [int(x) for x in addr_offsets]
    out_mem = [int(x) & 0xFF for x in memory]
    index_term = int(index_offsets[index]) * int(scale) if has_index(index) else 0
    ea_offset = int(out_addr[base]) + index_term + int(disp) + int(alias_offset)
    if ea_offset < 0 or ea_offset + width > MEM_SIZE:
        raise ValueError(f"effective address offset out of range: {ea_offset}")
    before_value = read_mem_le(out_mem, ea_offset, width)
    reads: List[Dict[str, Any]] = []
    writes: List[Dict[str, Any]] = []

    if actual_op.name.startswith("MOV_LOAD"):
        reads.append({"address": full_address(ea_offset), "width_bytes": width, "value": before_value})
        out_data[dst] = write_reg_width(out_data[dst], before_value, width)
    elif actual_op.name.startswith("MOV_STORE_MR"):
        value = read_reg_width(out_data[src], width)
        write_mem_le(out_mem, ea_offset, width, value)
        writes.append({"address": full_address(ea_offset), "width_bytes": width, "value": value})
    elif actual_op.name.startswith("MOV_STORE_MI"):
        value = int(imm) & width_mask(width)
        write_mem_le(out_mem, ea_offset, width, value)
        writes.append({"address": full_address(ea_offset), "width_bytes": width, "value": value})
    elif actual_op.name == "LEA_RA":
        out_data[dst] = full_address(ea_offset)
    elif actual_op.name.startswith("XCHG_RM"):
        reg_value = read_reg_width(out_data[dst], width)
        reads.append({"address": full_address(ea_offset), "width_bytes": width, "value": before_value})
        out_data[dst] = write_reg_width(out_data[dst], before_value, width)
        write_mem_le(out_mem, ea_offset, width, reg_value)
        writes.append({"address": full_address(ea_offset), "width_bytes": width, "value": reg_value})
    else:
        raise ValueError(actual_op.name)
    effect = {
        "op_name": actual_op.name,
        "effective_offset": ea_offset,
        "effective_address": full_address(ea_offset),
        "memory_reads": reads,
        "memory_writes": writes,
    }
    return out_data, out_addr, out_mem, effect


def apply_selected(
    data_regs: Sequence[int],
    addr_offsets: Sequence[int],
    index_offsets: Sequence[int],
    memory: Sequence[int],
    op_idx: int,
    dst: int,
    src: int,
    base: int,
    index: int,
    scale: int,
    disp: int,
    imm: int,
    value_selector: int,
    memory_selector: int,
) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
    out_data = [int(x) & 0xFFFFFFFFFFFFFFFF for x in data_regs]
    out_addr = [int(x) for x in addr_offsets]
    out_mem = [int(x) & 0xFF for x in memory]
    op = OPS[op_idx]
    width = int(op.width_bytes)
    index_term = int(index_offsets[index]) * int(scale) if has_index(index) else 0
    ea_offset = int(out_addr[base]) + index_term + int(disp)
    if ea_offset < 0 or ea_offset + width > MEM_SIZE:
        raise ValueError(f"effective address offset out of range: {ea_offset}")
    before_value = read_mem_le(out_mem, ea_offset, width)
    reads: List[Dict[str, Any]] = []
    writes: List[Dict[str, Any]] = []

    if value_selector == 1:
        reads.append({"address": full_address(ea_offset), "width_bytes": width, "value": before_value})
        out_data[dst] = write_reg_width(out_data[dst], before_value, width)
    elif value_selector == 2:
        out_data[dst] = full_address(ea_offset)
    elif value_selector != 0:
        raise ValueError(f"unknown value selector {value_selector}")

    if memory_selector == 1:
        value = read_reg_width(out_data[src], width)
        write_mem_le(out_mem, ea_offset, width, value)
        writes.append({"address": full_address(ea_offset), "width_bytes": width, "value": value})
    elif memory_selector == 2:
        value = int(imm) & width_mask(width)
        write_mem_le(out_mem, ea_offset, width, value)
        writes.append({"address": full_address(ea_offset), "width_bytes": width, "value": value})
    elif memory_selector == 3:
        value = read_reg_width(data_regs[dst], width)
        write_mem_le(out_mem, ea_offset, width, value)
        writes.append({"address": full_address(ea_offset), "width_bytes": width, "value": value})
    elif memory_selector != 0:
        raise ValueError(f"unknown memory selector {memory_selector}")

    effect = {
        "op_name": OPS[op_idx].name,
        "effective_offset": ea_offset,
        "effective_address": full_address(ea_offset),
        "memory_reads": reads,
        "memory_writes": writes,
    }
    return out_data, out_addr, out_mem, effect


def event_record(event: Dict[str, Any]) -> Dict[str, Any]:
    width = int(event["width_bytes"])
    return {
        "address": hex_u(int(event["address"]), 64),
        "width_bytes": width,
        "value_hex": width_hex(int(event["value"]), width),
        "segment": "flat",
        "fault": None,
    }


def choose_instruction(rng: random.Random, split: str) -> Tuple[int, int, int, int, int, int, int, int]:
    if split not in STAGE1_SPLITS:
        raise ValueError(f"unknown split {split!r}")
    op_idx = rng.randrange(len(OPS))
    dst = rng.randrange(len(DATA_REGS))
    src = rng.randrange(len(DATA_REGS))
    if split == "heldout_base":
        base = ADDR_REGS.index("RBP")
    else:
        base = rng.randrange(2)
    use_index = split in ("heldout_base", "full") or rng.random() < 0.5
    if use_index:
        index = rng.randrange(len(INDEX_REGS))
        scale = rng.choice((1, 2) if split in ("train", "heldout_disp") else (1, 2, 4, 8))
    else:
        index = -1
        scale = 1
    if split == "train":
        disp = rng.randrange(0, 32, 2)
    elif split == "heldout_disp":
        disp = rng.randrange(1, 32, 2)
    else:
        disp = rng.randrange(0, 32)
    op = OPS[op_idx]
    if op.operands == "m_i":
        imm = rng.randrange(1 << (8 * op.width_bytes))
    else:
        imm = rng.randrange(256)
    return op_idx, dst, src, base, index, scale, disp, imm


def make_trace(trace_id: str, split: str, length: int, rng: random.Random) -> Dict[str, Any]:
    data_regs = [rng.randrange(1 << 64) for _ in DATA_REGS]
    addr_offsets = [rng.randrange(0, 96) for _ in ADDR_REGS]
    index_offsets = [rng.randrange(0, 12) for _ in INDEX_REGS]
    memory = [rng.randrange(256) for _ in range(MEM_SIZE)]
    steps = []
    all_bytes = []
    asm_lines = []
    rip = 0x400000
    for i in range(length):
        op_idx, dst, src, base, index, scale, disp, imm = choose_instruction(rng, split)
        op = OPS[op_idx]
        before_data = list(data_regs)
        before_addr = list(addr_offsets)
        before_index = list(index_offsets)
        before_memory = list(memory)
        data_regs, addr_offsets, memory, effect = apply_op(
            data_regs, addr_offsets, index_offsets, memory, op_idx, dst, src, base, index, scale, disp, imm
        )
        bytes_hex = encode_x86_bytes(op, dst, src, base, index, scale, disp, imm)
        asm = asm_string(op, dst, src, base, index, scale, disp, imm)
        all_bytes.append(bytes_hex)
        asm_lines.append(asm)
        before = state_record(before_data, before_addr, before_memory, rip, before_index)
        rip_after = rip + len(bytes.fromhex(bytes_hex))
        after = state_record(data_regs, addr_offsets, memory, rip_after, index_offsets)

        reg_writes = []
        for ridx, name in enumerate(DATA_REGS):
            if before_data[ridx] != data_regs[ridx]:
                reg_writes.append({
                    "name": name,
                    "before": hex_u(before_data[ridx], 64),
                    "after": hex_u(data_regs[ridx], 64),
                    "mask": hex_u(
                        0xFFFFFFFFFFFFFFFF if op.name == "LEA_RA" else register_write_mask(op.width_bytes),
                        64,
                    ),
                })
        dst_name = DATA_REGS[dst] if op.name == "LEA_RA" else reg_name_for_width(dst, op.width_bytes)
        src_operand = {"kind": "none", "role": "src"}
        if op.operands == "m_r":
            src_operand = {"kind": "reg", "role": "src", "name": reg_name_for_width(src, op.width_bytes)}
        elif op.operands == "m_i":
            src_operand = {"kind": "imm", "role": "src", "value": imm & width_mask(op.width_bytes)}
        steps.append({
            "index": i,
            "decode": {
                "bytes_hex": bytes_hex,
                "iclass": op.name.split("_")[0],
                "iform": op.iform,
                "category": op.family,
                "extension": "BASE",
                "operands": [
                    {"kind": "reg", "role": "dst", "name": dst_name},
                    {
                        "kind": "mem",
                        "role": "mem",
                        "base": ADDR_REGS[base],
                        "index": INDEX_REGS[index] if has_index(index) else None,
                        "scale": scale,
                        "disp": disp,
                        "width_bytes": op.width_bytes,
                    },
                    src_operand,
                ],
            },
            "before": before,
            "after": after,
            "effects": {
                "register_writes": reg_writes,
                "memory_reads": [event_record(e) for e in effect["memory_reads"]],
                "memory_writes": [event_record(e) for e in effect["memory_writes"]],
                "flags_written": [],
                "undefined_flags": [],
                "exception": None,
            },
            "pan_stage1": {
                "op_index": op_idx,
                "dst": dst,
                "src": src,
                "base": base,
                "index": index,
                "scale": scale,
                "disp": disp,
                "imm": imm & width_mask(op.width_bytes),
                "effective_address": effect["effective_address"],
                "before_data_regs": before_data,
                "before_addr_offsets": before_addr,
                "before_index_offsets": before_index,
                "before_memory": before_memory,
                "after_data_regs": data_regs,
                "after_addr_offsets": addr_offsets,
                "after_index_offsets": list(index_offsets),
                "after_memory": memory,
            },
            "oracle_disagreement": None,
        })
        rip = rip_after
    return {
        "schema_version": "pan-x86-trace-v1",
        "trace_id": trace_id,
        "stage": STAGE,
        "source": {
            "generator": "stage1_memory_pan.py",
            "seed": trace_id,
            "xed_version": "not-yet-integrated-stage1-internal-encoder",
            "intel_sdm_hash": "not-yet-frozen",
            "amd_apm_revision": "not-yet-frozen",
            "oracle": ["stage1_python_semantics"],
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
        "final": state_record(data_regs, addr_offsets, memory, rip),
    }


def split_diagnostics(traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    families: Dict[str, int] = {}
    ops: Dict[str, int] = {}
    bases: Dict[str, int] = {}
    widths: Dict[str, int] = {}
    indexes: Dict[str, int] = {}
    scales: Dict[str, int] = {}
    templates: Dict[str, int] = {}
    disps = set()
    lengths = []
    memory_reads = 0
    memory_writes = 0
    for trace in traces:
        lengths.append(int(trace["program"]["length"]))
        for step in trace["steps"]:
            p = step["pan_stage1"]
            op = OPS[int(p["op_index"])]
            families[op.family] = families.get(op.family, 0) + 1
            ops[op.name] = ops.get(op.name, 0) + 1
            bases[ADDR_REGS[int(p["base"])]] = bases.get(ADDR_REGS[int(p["base"])], 0) + 1
            index = int(p.get("index", -1))
            index_name = INDEX_REGS[index] if has_index(index) else "none"
            indexes[index_name] = indexes.get(index_name, 0) + 1
            scale = int(p.get("scale", 1))
            scales[str(scale)] = scales.get(str(scale), 0) + 1
            template = "base_index_scale_disp" if has_index(index) else "base_disp"
            templates[template] = templates.get(template, 0) + 1
            widths[str(op.width_bytes)] = widths.get(str(op.width_bytes), 0) + 1
            disps.add(int(p["disp"]))
            memory_reads += len(step["effects"]["memory_reads"])
            memory_writes += len(step["effects"]["memory_writes"])
    return {
        "programs": len(traces),
        "steps": sum(lengths),
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "op_counts": dict(sorted(ops.items())),
        "family_counts": dict(sorted(families.items())),
        "base_counts": dict(sorted(bases.items())),
        "index_counts": dict(sorted(indexes.items())),
        "scale_counts": dict(sorted(scales.items(), key=lambda kv: int(kv[0]))),
        "address_template_counts": dict(sorted(templates.items())),
        "width_counts": dict(sorted(widths.items(), key=lambda kv: int(kv[0]))),
        "disp_values": sorted(disps),
        "memory_reads": memory_reads,
        "memory_writes": memory_writes,
    }


def make_manifest(out_dir: Path) -> Dict[str, Any]:
    forms = []
    for i, op in enumerate(OPS):
        forms.append(dataclasses.asdict(op) | {
            "op_index": i,
            "implemented": True,
            "encoding_policy": "real x86-64 encodings over 8/16/32/64-bit GPR memory forms and base/index/scale/disp addressing",
        })
    manifest = {
        "stage": STAGE,
        "memory_base": hex_u(MEM_BASE, 64),
        "memory_size": MEM_SIZE,
        "data_registers": list(DATA_REGS),
        "data_byte_aliases": list(DATA_BYTES),
        "data_word_aliases": list(DATA_WORDS),
        "data_dword_aliases": list(DATA_DWORDS),
        "address_registers": list(ADDR_REGS),
        "index_registers": list(INDEX_REGS),
        "forms": forms,
        "limitations": [
            "integer load/store/XCHG memory widths only; no vector or floating memory forms",
            "flat mapped memory only",
            "no branches, exceptions, SIMD/x87, prefixes, or mode variation",
            "XED iform inventory exists, but Stage 1 trace generation is still internal",
        ],
    }
    write_json(out_dir / "coverage_manifest.json", manifest)
    return manifest


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
            length = rng.randint(min_len, max_len)
            trace = make_trace(f"{split}-{args.seed}-{len(out)}", split, length, rng)
            bytes_hex = trace["program"]["bytes_hex"]
            if bytes_hex in seen:
                continue
            seen.add(bytes_hex)
            out.append(trace)
        if len(out) != count:
            raise RuntimeError(f"could only generate {len(out)} unique {split} traces after {attempts} attempts")
        avoid_bytes.update(t["program"]["bytes_hex"] for t in out)
        return out

    used_bytes: set[str] = set()
    train_traces = rows(args.train_programs, args.train_split, 1, args.max_train_len, used_bytes)
    val_traces = rows(args.val_programs, args.val_split, 1, args.max_train_len, used_bytes)
    long_traces = rows(args.long_programs, args.long_split, args.long_len, args.long_len, used_bytes)
    train_path = out_dir / "train_traces.jsonl"
    val_path = out_dir / "val_traces.jsonl"
    long_path = out_dir / "long_traces.jsonl"
    n_train = write_jsonl(train_path, train_traces)
    n_val = write_jsonl(val_path, val_traces)
    n_long = write_jsonl(long_path, long_traces)
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
            "train": split_diagnostics(train_traces),
            "val": split_diagnostics(val_traces),
            "long": split_diagnostics(long_traces),
        },
    }
    write_json(out_dir / "generation_summary.json", summary)
    return summary


class Predictor(Protocol):
    def predict_next(
        self,
        data_regs: List[int],
        addr_offsets: List[int],
        index_offsets: List[int],
        memory: List[int],
        op_idx: int,
        dst: int,
        src: int,
        base: int,
        index: int,
        scale: int,
        disp: int,
        imm: int,
    ) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
        ...


class StructuredStage1Predictor:
    def __init__(self, alias_offset: int = 0, op_offset: int = 0):
        self.alias_offset = alias_offset
        self.op_offset = op_offset

    def predict_next(
        self,
        data_regs: List[int],
        addr_offsets: List[int],
        index_offsets: List[int],
        memory: List[int],
        op_idx: int,
        dst: int,
        src: int,
        base: int,
        index: int,
        scale: int,
        disp: int,
        imm: int,
    ) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
        return apply_op(
            data_regs,
            addr_offsets,
            index_offsets,
            memory,
            op_idx,
            dst,
            src,
            base,
            index,
            scale,
            disp,
            imm,
            alias_offset=self.alias_offset,
            op_offset=self.op_offset,
        )


class FactorizedStage1PAN(nn.Module):
    """Learn selectors over phase-native Stage 1 transition candidates."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.op_emb = nn.Embedding(len(OPS), 16)
        self.value_net = nn.Sequential(
            nn.Linear(16, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 3),
        )
        self.memory_net = nn.Sequential(
            nn.Linear(16, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 4),
        )

    def selector_logits(self, op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.op_emb(op.long())
        return self.value_net(x), self.memory_net(x)


class EffectStage1PAN(FactorizedStage1PAN):
    """Learned Stage 1 typed-effect selector head.

    Stage 1 effects are decomposed into two architectural axes: register value
    source and memory event kind. The old neural path learned those axes but
    reported them as generic selectors; this class makes the effect contract
    explicit and gives validation a separate promotion gate.
    """


def make_model(arch: str, hidden: int) -> FactorizedStage1PAN:
    if arch == "factorized":
        return FactorizedStage1PAN(hidden=hidden)
    if arch == "effect":
        return EffectStage1PAN(hidden=hidden)
    raise ValueError(f"unknown Stage 1 arch {arch!r}")


class NeuralStage1Predictor:
    def __init__(self, model: FactorizedStage1PAN):
        self.model = model
        with torch.no_grad():
            ops = torch.arange(len(OPS), dtype=torch.long)
            value_logits, memory_logits = self.model.selector_logits(ops)
        self.value_selectors = value_logits.argmax(-1).cpu().tolist()
        self.memory_selectors = memory_logits.argmax(-1).cpu().tolist()

    def predict_next(
        self,
        data_regs: List[int],
        addr_offsets: List[int],
        index_offsets: List[int],
        memory: List[int],
        op_idx: int,
        dst: int,
        src: int,
        base: int,
        index: int,
        scale: int,
        disp: int,
        imm: int,
    ) -> Tuple[List[int], List[int], List[int], Dict[str, Any]]:
        return apply_selected(
            data_regs,
            addr_offsets,
            index_offsets,
            memory,
            op_idx,
            dst,
            src,
            base,
            index,
            scale,
            disp,
            imm,
            self.value_selectors[op_idx],
            self.memory_selectors[op_idx],
        )


def step_rows(traces: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    ops = []
    families = []
    for trace in traces:
        for step in trace["steps"]:
            p = step["pan_stage1"]
            ops.append(int(p["op_index"]))
            families.append(step["decode"]["category"])
    if not ops:
        raise ValueError("no step rows")
    return {
        "op": np.array(ops, dtype=np.int64),
        "family": np.array(families, dtype=object),
    }


@torch.no_grad()
def selector_metrics(model: FactorizedStage1PAN, rows: Dict[str, np.ndarray], limit: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    n = rows["op"].shape[0]
    if limit is not None and n > limit:
        idx = np.linspace(0, n - 1, num=limit, dtype=np.int64)
    else:
        idx = np.arange(n)
    ops = torch.tensor(rows["op"][idx], dtype=torch.long)
    value_logits, memory_logits = model.selector_logits(ops)
    value_pred = value_logits.argmax(-1)
    memory_pred = memory_logits.argmax(-1)
    value_target = OP_VALUE_SELECTOR[ops]
    memory_target = OP_MEMORY_SELECTOR[ops]
    value_ok = value_pred.eq(value_target).cpu().numpy()
    memory_ok = memory_pred.eq(memory_target).cpu().numpy()
    step_ok = value_ok & memory_ok
    family_counts: Dict[str, List[int]] = {}
    for fam, ok in zip(rows["family"][idx], step_ok):
        bucket = family_counts.setdefault(str(fam), [0, 0])
        bucket[0] += int(bool(ok))
        bucket[1] += 1
    model.train()
    return {
        "value_selector_exact": round(float(value_ok.mean()), 6),
        "memory_selector_exact": round(float(memory_ok.mean()), 6),
        "step_selector_exact": round(float(step_ok.mean()), 6),
        "family_selector_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(family_counts.items())},
        "n_steps": int(len(idx)),
    }


def train_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("train")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_rows = step_rows(load_traces(out_dir / "train_traces.jsonl"))
    val_rows = step_rows(load_traces(out_dir / "val_traces.jsonl"))
    model = make_model(args.arch, hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = train_rows["op"].shape[0]
    history = []
    for step in range(args.train_steps):
        idx = np.random.randint(0, n, size=args.batch_size)
        ops = torch.tensor(train_rows["op"][idx], dtype=torch.long)
        value_logits, memory_logits = model.selector_logits(ops)
        value_target = OP_VALUE_SELECTOR[ops]
        memory_target = OP_MEMORY_SELECTOR[ops]
        loss = F.cross_entropy(value_logits, value_target) + F.cross_entropy(memory_logits, memory_target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.eval_every) == 0 or step == args.train_steps - 1:
            train_metrics = selector_metrics(model, train_rows, limit=2000)
            val_metrics = selector_metrics(model, val_rows, limit=2000)
            rec = {
                "step": step,
                "loss": round(float(loss.detach().cpu()), 6),
                "train_step_selector_exact": train_metrics["step_selector_exact"],
                "val_step_selector_exact": val_metrics["step_selector_exact"],
                "train_value_selector_exact": train_metrics["value_selector_exact"],
                "val_value_selector_exact": val_metrics["value_selector_exact"],
                "train_memory_selector_exact": train_metrics["memory_selector_exact"],
                "val_memory_selector_exact": val_metrics["memory_selector_exact"],
            }
            history.append(rec)
            print(json.dumps({"train_progress": rec}, sort_keys=True), flush=True)
    ckpt = {
        "model_state": model.state_dict(),
        "stage": STAGE,
        "model_kind": "neural",
        "arch": args.arch,
        "hidden": args.hidden,
        "operators": [dataclasses.asdict(op) for op in OPS],
        "data_registers": list(DATA_REGS),
        "address_registers": list(ADDR_REGS),
        "value_selector_index": VALUE_SELECTOR_INDEX,
        "memory_selector_index": MEMORY_SELECTOR_INDEX,
    }
    torch.save(ckpt, out_dir / "stage1_pan.pt")
    write_json(out_dir / "train_history.json", history)
    summary = {
        "checkpoint": str(out_dir / "stage1_pan.pt"),
        "model_kind": "neural",
        "arch": args.arch,
        "history": history,
        "final": history[-1] if history else None,
    }
    write_json(out_dir / "train_summary.json", summary)
    return summary


def load_model(out_dir: Path) -> FactorizedStage1PAN:
    ckpt = torch.load(out_dir / "stage1_pan.pt", map_location="cpu")
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
    total_steps = 0
    step_exact = 0
    final_exact = 0
    register_exact = 0
    memory_digest_exact = 0
    effective_address_exact = 0
    memory_event_exact = 0
    untouched_memory_preservation = 0
    family_counts: Dict[str, List[int]] = {}
    examples = []
    for trace in traces:
        first = trace["steps"][0]["pan_stage1"]
        data_regs = list(first["before_data_regs"])
        addr_offsets = list(first["before_addr_offsets"])
        index_offsets = list(first.get("before_index_offsets", [0 for _ in INDEX_REGS]))
        memory = list(first["before_memory"])
        trace_ok = True
        for step in trace["steps"]:
            p = step["pan_stage1"]
            before_memory = list(memory)
            data_regs, addr_offsets, memory, effect = predictor.predict_next(
                data_regs,
                addr_offsets,
                index_offsets,
                memory,
                int(p["op_index"]),
                int(p["dst"]),
                int(p["src"]),
                int(p["base"]),
                int(p.get("index", -1)),
                int(p.get("scale", 1)),
                int(p["disp"]),
                int(p["imm"]),
            )
            reg_ok = (
                data_regs == p["after_data_regs"]
                and addr_offsets == p["after_addr_offsets"]
                and index_offsets == p.get("after_index_offsets", index_offsets)
            )
            mem_ok = memory == p["after_memory"]
            ea_ok = int(effect["effective_address"]) == int(p["effective_address"])
            events_ok = (
                normalized_events(effect["memory_reads"]) == step["effects"]["memory_reads"]
                and normalized_events(effect["memory_writes"]) == step["effects"]["memory_writes"]
            )
            written = {
                int(e["address"]) - MEM_BASE + delta
                for e in effect["memory_writes"]
                for delta in range(int(e["width_bytes"]))
                if 0 <= int(e["address"]) - MEM_BASE + delta < MEM_SIZE
            }
            untouched_ok = all(
                memory[i] == before_memory[i]
                for i in range(MEM_SIZE)
                if i not in written
            )
            ok = reg_ok and mem_ok and ea_ok and events_ok
            trace_ok = trace_ok and ok
            register_exact += int(reg_ok)
            memory_digest_exact += int(mem_ok)
            effective_address_exact += int(ea_ok)
            memory_event_exact += int(events_ok)
            untouched_memory_preservation += int(untouched_ok)
            step_exact += int(ok)
            total_steps += 1
            fam = step["decode"]["category"]
            bucket = family_counts.setdefault(str(fam), [0, 0])
            bucket[0] += int(ok)
            bucket[1] += 1
            if not ok and len(examples) < 8:
                examples.append({
                    "trace_id": trace["trace_id"],
                    "step": step["index"],
                    "asm": trace["program"]["asm"][step["index"]],
                    "pred_data_regs": data_regs,
                    "true_data_regs": p["after_data_regs"],
                    "pred_memory_digest": memory_digest(memory),
                    "true_memory_digest": memory_digest(p["after_memory"]),
                    "pred_effective_address": effect["effective_address"],
                    "true_effective_address": p["effective_address"],
                })
        final_exact += int(trace_ok)
    return {
        "programs": len(traces),
        "steps": total_steps,
        "step_exact": round(step_exact / max(1, total_steps), 6),
        "final_exact": round(final_exact / max(1, len(traces)), 6),
        "register_exact": round(register_exact / max(1, total_steps), 6),
        "memory_digest_exact": round(memory_digest_exact / max(1, total_steps), 6),
        "effective_address_exact": round(effective_address_exact / max(1, total_steps), 6),
        "memory_event_exact": round(memory_event_exact / max(1, total_steps), 6),
        "untouched_memory_preservation": round(untouched_memory_preservation / max(1, total_steps), 6),
        "family_step_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(family_counts.items())},
        "failure_examples": examples,
    }


def control_results(val_traces: Sequence[Dict[str, Any]], long_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "wrong_memory_alias_val": free_run_metrics(StructuredStage1Predictor(alias_offset=1), val_traces),
        "scrambled_opcode_val": free_run_metrics(StructuredStage1Predictor(op_offset=1), val_traces),
        "wrong_memory_alias_long": free_run_metrics(StructuredStage1Predictor(alias_offset=1), long_traces),
    }


def write_structured_cartridge(out_dir: Path) -> Dict[str, Any]:
    data = {
        "model_kind": "structured",
        "stage": STAGE,
        "memory_base": hex_u(MEM_BASE, 64),
        "memory_size": MEM_SIZE,
        "data_registers": list(DATA_REGS),
        "address_registers": list(ADDR_REGS),
        "index_registers": list(INDEX_REGS),
        "operators": [dataclasses.asdict(op) for op in OPS],
        "transition": {
            "representation": "semantic_spec",
            "function": "apply_op(data_regs, addr_offsets, index_offsets, memory, op_idx, dst, src, base, index, scale, disp, imm)",
            "effects": ["effective_address", "register_write", "memory_read", "memory_write"],
        },
        "note": "Exact structured Stage 1 cartridge generated from x86-lite memory/addressing semantic maps.",
    }
    write_json(out_dir / "stage1_structured_cartridge.json", data)
    summary = {
        "model_kind": "structured",
        "checkpoint": str(out_dir / "stage1_structured_cartridge.json"),
        "history": [],
        "final": {"note": "No learned weights: exact Stage 1 structured cartridge emitted for validation."},
    }
    write_json(out_dir / "train_summary.json", summary)
    return summary


def unicorn_step(step: Dict[str, Any]) -> Tuple[List[int], List[int]]:
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
    from unicorn.x86_const import (
        UC_X86_REG_RAX,
        UC_X86_REG_RBX,
        UC_X86_REG_RCX,
        UC_X86_REG_RDI,
        UC_X86_REG_RDX,
        UC_X86_REG_RBP,
        UC_X86_REG_RIP,
        UC_X86_REG_RSI,
        UC_X86_REG_R8,
        UC_X86_REG_R9,
    )

    p = step["pan_stage1"]
    code = bytes.fromhex(step["decode"]["bytes_hex"])
    code_base = 0x100000
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(code_base, 0x1000)
    uc.mem_write(code_base, code)
    uc.mem_map(MEM_BASE, 0x1000)
    uc.mem_write(MEM_BASE, bytes(int(x) & 0xFF for x in p["before_memory"]))
    reg_map = [UC_X86_REG_RAX, UC_X86_REG_RCX, UC_X86_REG_RDX, UC_X86_REG_RBX]
    addr_map = [UC_X86_REG_RSI, UC_X86_REG_RDI, UC_X86_REG_RBP]
    index_map = [UC_X86_REG_R8, UC_X86_REG_R9]
    for reg, value in zip(reg_map, p["before_data_regs"]):
        uc.reg_write(reg, int(value) & 0xFFFFFFFFFFFFFFFF)
    for reg, offset in zip(addr_map, p["before_addr_offsets"]):
        uc.reg_write(reg, full_address(int(offset)))
    for reg, offset in zip(index_map, p.get("before_index_offsets", [0 for _ in INDEX_REGS])):
        uc.reg_write(reg, int(offset) & 0xFFFFFFFFFFFFFFFF)
    uc.reg_write(UC_X86_REG_RIP, code_base)
    uc.emu_start(code_base, code_base + len(code), count=1)
    out_regs = [int(uc.reg_read(reg)) & 0xFFFFFFFFFFFFFFFF for reg in reg_map]
    out_mem = list(bytes(uc.mem_read(MEM_BASE, MEM_SIZE)))
    return out_regs, out_mem


def validate_external_oracle(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    oracle = getattr(args, "external_oracle", "none")
    report: Dict[str, Any] = {
        "enabled": oracle != "none",
        "oracle": oracle,
        "ok": True,
        "checked_steps": 0,
        "splits": {},
        "failure_examples": [],
        "error": None,
    }
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
    max_steps = int(getattr(args, "oracle_max_steps", 10000))
    for split, path in (("val", out_dir / "val_traces.jsonl"), ("long", out_dir / "long_traces.jsonl")):
        checked = 0
        exact = 0
        for trace in load_traces(path):
            for step in trace["steps"]:
                if report["checked_steps"] >= max_steps:
                    break
                p = step["pan_stage1"]
                try:
                    regs, memory = unicorn_step(step)
                except Exception as exc:
                    report["ok"] = False
                    report["error"] = f"unicorn execution failed: {exc}"
                    return report
                ok = regs == p["after_data_regs"] and memory == p["after_memory"]
                checked += 1
                exact += int(ok)
                report["checked_steps"] += 1
                if not ok and len(report["failure_examples"]) < 8:
                    report["failure_examples"].append({
                        "split": split,
                        "trace_id": trace["trace_id"],
                        "step": step["index"],
                        "asm": trace["program"]["asm"][step["index"]],
                        "bytes_hex": step["decode"]["bytes_hex"],
                        "unicorn_regs": regs,
                        "trace_regs": p["after_data_regs"],
                        "unicorn_memory_digest": memory_digest(memory),
                        "trace_memory_digest": memory_digest(p["after_memory"]),
                    })
            if report["checked_steps"] >= max_steps:
                break
        report["splits"][split] = {
            "checked_steps": checked,
            "exact": round(exact / max(1, checked), 6),
        }
    report["ok"] = len(report["failure_examples"]) == 0 and report["checked_steps"] > 0
    return report


def validate_structured(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("validate")
    predictor = StructuredStage1Predictor()
    train_traces = load_traces(out_dir / "train_traces.jsonl")
    val_traces = load_traces(out_dir / "val_traces.jsonl")
    long_traces = load_traces(out_dir / "long_traces.jsonl")
    report = {
        "stage": STAGE,
        "model_kind": "structured",
        "thresholds": {
            "configured_stage1_exact_final_state": 0.990,
            "configured_stage1_memory_event_exact": 0.990,
            "configured_stage1_effective_address_exact": 0.995,
            "configured_stage1_untouched_memory_preservation": 0.999,
        },
        "free_run": {
            "train": free_run_metrics(predictor, train_traces),
            "val": free_run_metrics(predictor, val_traces),
            "long": free_run_metrics(predictor, long_traces),
        },
        "byte_overlap": byte_overlap(train_traces, val_traces, long_traces),
        "controls": control_results(val_traces, long_traces),
        "external_oracle": validate_external_oracle(args, out_dir),
    }
    schema_valid = json.loads((out_dir / "schema_validation.json").read_text(encoding="utf-8")).get("ok", False)
    val = report["free_run"]["val"]
    long = report["free_run"]["long"]
    report["gate_status"] = {
        "schema_valid": schema_valid,
        "stage1_gate_passed": (
            schema_valid
            and report["byte_overlap"]["train_val_overlap"] == 0
            and report["byte_overlap"]["train_long_overlap"] == 0
            and val["final_exact"] >= 0.990
            and val["step_exact"] >= 0.990
            and long["final_exact"] >= 0.990
            and long["step_exact"] >= 0.990
            and val["memory_event_exact"] >= 0.990
            and val["effective_address_exact"] >= 0.995
            and val["untouched_memory_preservation"] >= 0.999
            and report["controls"]["wrong_memory_alias_val"]["step_exact"] < 0.900
            and report["controls"]["scrambled_opcode_val"]["step_exact"] < 0.900
            and (not report["external_oracle"]["enabled"] or report["external_oracle"]["ok"])
        ),
        "full_x86_goal_complete": False,
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def validate_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("validate")
    model = load_model(out_dir)
    predictor = NeuralStage1Predictor(model)
    learned_effect_mode = isinstance(model, EffectStage1PAN)
    train_traces = load_traces(out_dir / "train_traces.jsonl")
    val_traces = load_traces(out_dir / "val_traces.jsonl")
    long_traces = load_traces(out_dir / "long_traces.jsonl")
    train_rows = step_rows(train_traces)
    val_rows = step_rows(val_traces)
    long_rows = step_rows(long_traces)
    report = {
        "stage": STAGE,
        "model_kind": "neural",
        "arch": "effect" if learned_effect_mode else "factorized",
        "transition_contract": {
            "schema": "pan-x86-stage1-effect-selector-v1" if learned_effect_mode else "legacy-stage1-factorized-selector",
            "learned_effect_selectors": learned_effect_mode,
            "learned_effect_axes": ["register_value_source", "memory_event_kind"] if learned_effect_mode else [],
            "commit_path": "model_emitted_effect_selectors" if learned_effect_mode else "legacy_model_selectors",
            "direct_value_heads": False,
            "note": "Stage 1 effect mode learns typed effect axes; final byte values still use the structured commit cartridge.",
        },
        "thresholds": {
            "configured_stage1_exact_final_state": 0.990,
            "configured_stage1_memory_event_exact": 0.990,
            "configured_stage1_effective_address_exact": 0.995,
            "configured_stage1_untouched_memory_preservation": 0.999,
        },
        "selector_metrics": {
            "train": selector_metrics(model, train_rows),
            "val": selector_metrics(model, val_rows),
            "long": selector_metrics(model, long_rows),
        },
        "free_run": {
            "train": free_run_metrics(predictor, train_traces),
            "val": free_run_metrics(predictor, val_traces),
            "long": free_run_metrics(predictor, long_traces),
        },
        "byte_overlap": byte_overlap(train_traces, val_traces, long_traces),
        "controls": control_results(val_traces, long_traces),
        "external_oracle": validate_external_oracle(args, out_dir),
    }
    schema_valid = json.loads((out_dir / "schema_validation.json").read_text(encoding="utf-8")).get("ok", False)
    val = report["free_run"]["val"]
    long = report["free_run"]["long"]
    learned_effect_gate = True
    if learned_effect_mode:
        learned_effect_gate = (
            report["selector_metrics"]["val"]["value_selector_exact"] >= 0.990
            and report["selector_metrics"]["val"]["memory_selector_exact"] >= 0.990
            and report["selector_metrics"]["long"]["value_selector_exact"] >= 0.990
            and report["selector_metrics"]["long"]["memory_selector_exact"] >= 0.990
        )
    report["gate_status"] = {
        "schema_valid": schema_valid,
        "stage1_gate_passed": (
            schema_valid
            and report["byte_overlap"]["train_val_overlap"] == 0
            and report["byte_overlap"]["train_long_overlap"] == 0
            and report["selector_metrics"]["val"]["step_selector_exact"] >= 0.990
            and learned_effect_gate
            and val["final_exact"] >= 0.990
            and val["step_exact"] >= 0.990
            and long["final_exact"] >= 0.990
            and long["step_exact"] >= 0.990
            and val["memory_event_exact"] >= 0.990
            and val["effective_address_exact"] >= 0.995
            and val["untouched_memory_preservation"] >= 0.999
            and report["controls"]["wrong_memory_alias_val"]["step_exact"] < 0.900
            and report["controls"]["scrambled_opcode_val"]["step_exact"] < 0.900
            and (not report["external_oracle"]["enabled"] or report["external_oracle"]["ok"])
        ),
        "learned_effect_selector_gate_passed": learned_effect_gate,
        "full_x86_goal_complete": False,
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def source_freeze(out_dir: Path, model_kind: str, external_oracle: str) -> Dict[str, Any]:
    data = {
        "stage": STAGE,
        "model_kind": model_kind,
        "created": DEFAULT_DATE,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "external_xed": "not integrated in this Stage 1 implementation",
        "external_oracles": ["stage1_python_semantics"] + ([] if external_oracle == "none" else [external_oracle]),
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
            if monitor:
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
                    "wrong_memory_alias_val_step_exact": validation["controls"]["wrong_memory_alias_val"]["step_exact"],
                    "scrambled_opcode_val_step_exact": validation["controls"]["scrambled_opcode_val"]["step_exact"],
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
        sp.add_argument("--train-split", choices=STAGE1_SPLITS, default="train")
        sp.add_argument("--val-split", choices=STAGE1_SPLITS, default="heldout_disp")
        sp.add_argument("--long-split", choices=STAGE1_SPLITS, default="heldout_base")
        sp.add_argument("--external-oracle", choices=["none", "unicorn"], default="none")
        sp.add_argument("--oracle-pythonpath", type=Path, default=DEFAULT_ORACLE_DEP_PATH)
        sp.add_argument("--oracle-max-steps", type=int, default=10000)

    run = sub.add_parser("run-smoke", help="generate and validate a Stage 1 structured memory run")
    add_common(run)
    run.add_argument("--train-programs", type=int, default=1200)
    run.add_argument("--val-programs", type=int, default=400)
    run.add_argument("--long-programs", type=int, default=160)
    run.add_argument("--max-train-len", type=int, default=5)
    run.add_argument("--long-len", type=int, default=32)
    run.add_argument("--train-steps", type=int, default=300)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--hidden", type=int, default=64)
    run.add_argument("--lr", type=float, default=2e-3)
    run.add_argument("--eval-every", type=int, default=75)

    gen = sub.add_parser("generate", help="generate Stage 1 traces only")
    add_common(gen)
    gen.add_argument("--train-programs", type=int, default=1200)
    gen.add_argument("--val-programs", type=int, default=400)
    gen.add_argument("--long-programs", type=int, default=160)
    gen.add_argument("--max-train-len", type=int, default=5)
    gen.add_argument("--long-len", type=int, default=32)

    val = sub.add_parser("validate", help="validate an existing Stage 1 run directory")
    add_common(val)
    return p


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.out_dir is None:
        args.out_dir = default_out_dir(args.run_name)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = normalize_args(parser.parse_args(argv))
    add_oracle_pythonpath(args.oracle_pythonpath)
    if args.cmd == "run-smoke":
        result = run_pipeline(args)
        print(json.dumps({"run_summary": result}, indent=2, sort_keys=True))
        return 0
    if args.cmd == "generate":
        monitor = ResourceMonitor(args.out_dir, args.monitor_interval)
        monitor.start()
        try:
            source_freeze(args.out_dir, args.model_kind, args.external_oracle)
            result = generate_dataset(args, args.out_dir, monitor)
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            monitor.stop()
        return 0
    if args.cmd == "validate":
        monitor = ResourceMonitor(args.out_dir, args.monitor_interval)
        monitor.start()
        try:
            if args.model_kind == "structured":
                result = validate_structured(args, args.out_dir, monitor)
            else:
                result = validate_model(args, args.out_dir, monitor)
            print(json.dumps(result["gate_status"], indent=2, sort_keys=True))
        finally:
            monitor.stop()
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
