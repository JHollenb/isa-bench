#!/usr/bin/env python3
"""Integrated PAN x86 model runner.

This is the first cumulative model surface for the staged x86 work. Stages 0-4
remain useful narrow validators; this runner composes their structured semantic
cartridges into one full-state execution model so Stage 3 scalar work no longer
stands apart from Stage 2 control-flow/stack work.

It is still bounded to the instructions implemented by the existing stages. It
does not claim full x86 ISA coverage.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pan_x86_stages import stage0_x86_pan as s0
from pan_x86_stages import stage1_memory_pan as s1
from pan_x86_stages import stage2_control_flow_pan as s2
from pan_x86_stages import stage3_complex_scalar_pan as s3
from pan_x86_stages import stage4_simd_x87_pan as s4
from pan_x86_stages.stage0_x86_pan import FLAGS, DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, hex_u, write_json


STAGE = "stage5_full_model"
SUBSTAGE = "stage5_cumulative_stages_0_4_v1"
DEFAULT_OUT_DIR = DEFAULT_ARTIFACT_ROOT / "2026-06-17" / "pan-x86-isa" / "full_stage_model"
FULL_CODE_BASE = 0x700000
MASK64 = (1 << 64) - 1
LOW_BYTE_REGS = ("RAX", "RCX", "RDX", "RBX")
S1_OP_INDEX = {op.name: i for i, op in enumerate(s1.OPS)}


def u64(value: int) -> int:
    return int(value) & MASK64


def memory_digest(data: Sequence[int]) -> str:
    return "sha256:" + hashlib.sha256(bytes(int(x) & 0xFF for x in data)).hexdigest()


def xmem_read_u64(memory: Sequence[int], offset: int) -> int:
    return int.from_bytes(bytes(int(x) & 0xFF for x in memory[offset:offset + 8]), "little")


def copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def event_hex(event: Dict[str, Any]) -> Dict[str, Any]:
    width = int(event["width_bytes"])
    return {
        "address": hex_u(int(event["address"]), 64),
        "width_bytes": width,
        "value_hex": hex_u(int(event["value"]), 8 * width),
        "segment": event.get("segment", "flat"),
        "fault": event.get("fault"),
    }


@dataclass
class FullState:
    gpr: Dict[str, int] = field(default_factory=dict)
    flags: List[int] = field(default_factory=list)
    flag_defined: List[bool] = field(default_factory=list)
    xmm: List[int] = field(default_factory=list)
    memory: Dict[str, List[int]] = field(default_factory=dict)

    @classmethod
    def initial(cls) -> "FullState":
        stage1_memory = [0] * s1.MEM_SIZE
        stage2_stack = [0] * s2.STACK_SIZE
        stage3_memory = [0] * s3.MEM_SIZE
        stage4_memory = [0] * s4.MEM_SIZE

        stage3_memory[0:8] = list((0x10).to_bytes(8, "little"))
        stage4_memory[0:16] = list((0x0102030405060708090A0B0C0D0E0F10).to_bytes(16, "little"))

        return cls(
            gpr={
                "RAX": 0x0,
                "RBX": 0x5,
                "RCX": 0x3,
                "RDX": 0x2,
                "RSI": s1.MEM_BASE + 16,
                "RDI": s1.MEM_BASE + 64,
                "RBP": s1.MEM_BASE + 96,
                "R8": 8,
                "R9": 16,
                "RSP": s2.STACK_BASE + s2.STACK_START,
                "RIP": FULL_CODE_BASE,
            },
            flags=[0, 0, 0, 0],
            flag_defined=[True, True, True, True],
            xmm=[
                0x00000000000000010000000000000002,
                0x00000000000000030000000000000004,
                0,
                0,
                *([0] * 12),
            ],
            memory={
                "stage1_data": stage1_memory,
                "stage2_stack": stage2_stack,
                "stage3_data": stage3_memory,
                "stage4_data": stage4_memory,
            },
        )

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "FullState":
        return cls(
            gpr={name: int(value) for name, value in data["gpr_raw"].items()},
            flags=[int(x) for x in data["flags_raw"]],
            flag_defined=[bool(x) for x in data["flag_defined"]],
            xmm=[int(x) for x in data["xmm_raw"]],
            memory={name: [int(x) & 0xFF for x in values] for name, values in data["memory_raw"].items()},
        )

    def clone(self) -> "FullState":
        return FullState.from_json(self.raw_json())

    def raw_json(self) -> Dict[str, Any]:
        return {
            "gpr_raw": {name: int(value) for name, value in sorted(self.gpr.items())},
            "flags_raw": [int(x) for x in self.flags],
            "flag_defined": [bool(x) for x in self.flag_defined],
            "xmm_raw": [int(x) for x in self.xmm],
            "memory_raw": {name: [int(x) & 0xFF for x in values] for name, values in sorted(self.memory.items())},
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "rip": hex_u(self.gpr["RIP"], 64),
            "gpr": {name: hex_u(value, 64) for name, value in sorted(self.gpr.items())},
            "rflags": {name: int(self.flags[i]) for i, name in enumerate(FLAGS)},
            "flag_defined": {name: bool(self.flag_defined[i]) for i, name in enumerate(FLAGS)},
            "xmm": {s4.XMM_REGS[i]: hex_u(value, 128) for i, value in enumerate(self.xmm)},
            "memory_digest": {name: memory_digest(values) for name, values in sorted(self.memory.items())},
            "exception": None,
        }


class FullStageModel:
    def __init__(self, state: Optional[FullState] = None):
        self.state = state or FullState.initial()

    def _record_step(
        self,
        index: int,
        stage: str,
        operation: Dict[str, Any],
        before: Dict[str, Any],
        after: Dict[str, Any],
        decode: Dict[str, Any],
        effects: Dict[str, Any],
        stage_state: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "index": index,
            "stage": stage,
            "operation": operation,
            "decode": decode,
            "before": before,
            "after": after,
            "effects": effects,
            "stage_state": stage_state,
            "oracle_disagreement": None,
        }

    def apply(self, index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
        stage = str(operation["stage"])
        if stage == "stage0":
            return self.apply_stage0(index, operation)
        if stage == "stage1":
            return self.apply_stage1(index, operation)
        if stage == "stage2":
            return self.apply_stage2(index, operation)
        if stage == "stage3":
            return self.apply_stage3(index, operation)
        if stage == "stage4":
            return self.apply_stage4(index, operation)
        raise ValueError(f"unknown full-stage operation {stage!r}")

    def apply_stage0(self, index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
        before = self.state.snapshot()
        op = s0.OPS[s0.OP_INDEX[str(operation["op_name"])]]
        dst = int(operation.get("dst", 0))
        src = int(operation.get("src", 0))
        imm = int(operation.get("imm", 0))
        word_bits = int(operation.get("word_bits", 8))
        regs = [self.state.gpr[name] & 0xFF for name in LOW_BYTE_REGS]
        after_regs, after_flags = s0.apply_op(regs, self.state.flags, s0.OP_INDEX[op.name], dst, src, imm, word_bits)
        for i, name in enumerate(LOW_BYTE_REGS):
            self.state.gpr[name] = (self.state.gpr[name] & ~0xFF) | (int(after_regs[i]) & 0xFF)
        self.state.flags = [int(x) for x in after_flags]
        bytes_hex = s0.encode_x86_bytes(op, dst, src, imm, word_bits)
        self.state.gpr["RIP"] = u64(self.state.gpr["RIP"] + len(bytes.fromhex(bytes_hex)))
        after = self.state.snapshot()
        reg_writes = [
            {
                "name": s0.PARENT_REGS[s0.REGS[i]],
                "before": before["gpr"][s0.PARENT_REGS[s0.REGS[i]]],
                "after": after["gpr"][s0.PARENT_REGS[s0.REGS[i]]],
                "mask": hex_u(0xFF, 64),
            }
            for i in range(len(s0.REGS))
            if int(regs[i]) != int(after_regs[i])
        ]
        return self._record_step(
            index,
            "stage0_low_byte_alu",
            copy_jsonable(operation),
            before,
            after,
            {
                "bytes_hex": bytes_hex,
                "asm": s0.asm_string(op, dst, src, imm),
                "iform": op.iform,
                "category": op.family,
                "operands": op.operands,
            },
            {
                "register_writes": reg_writes,
                "memory_reads": [],
                "memory_writes": [],
                "flags_written": list(FLAGS) if op.writes_flags else [],
                "undefined_flags": [],
                "exception": None,
            },
            {"before_low_regs": regs, "after_low_regs": after_regs, "after_flags": after_flags},
        )

    def apply_stage1(self, index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
        before = self.state.snapshot()
        op = s1.OPS[S1_OP_INDEX[str(operation["op_name"])]]
        dst = int(operation.get("dst", 0))
        src = int(operation.get("src", 0))
        base = int(operation.get("base", 0))
        index_reg = int(operation.get("index", 0))
        scale = int(operation.get("scale", 1))
        disp = int(operation.get("disp", 0))
        imm = int(operation.get("imm", 0))
        data_regs = [self.state.gpr[name] for name in s1.DATA_REGS]
        addr_offsets = [self.state.gpr[name] - s1.MEM_BASE for name in s1.ADDR_REGS]
        index_offsets = [self.state.gpr[name] for name in s1.INDEX_REGS]
        mem_before = self.state.memory["stage1_data"]
        out_data, out_addr, out_mem, effect = s1.apply_op(
            data_regs,
            addr_offsets,
            index_offsets,
            mem_before,
            S1_OP_INDEX[op.name],
            dst,
            src,
            base,
            index_reg,
            scale,
            disp,
            imm,
        )
        for i, name in enumerate(s1.DATA_REGS):
            self.state.gpr[name] = u64(out_data[i])
        for i, name in enumerate(s1.ADDR_REGS):
            self.state.gpr[name] = u64(s1.MEM_BASE + int(out_addr[i]))
        self.state.memory["stage1_data"] = out_mem
        bytes_hex = s1.encode_x86_bytes(op, dst, src, base, index_reg, scale, disp, imm)
        self.state.gpr["RIP"] = u64(self.state.gpr["RIP"] + len(bytes.fromhex(bytes_hex)))
        after = self.state.snapshot()
        return self._record_step(
            index,
            "stage1_address_memory",
            copy_jsonable(operation),
            before,
            after,
            {
                "bytes_hex": bytes_hex,
                "asm": s1.asm_string(op, dst, src, base, index_reg, scale, disp, imm),
                "iform": op.iform,
                "category": op.family,
                "operands": op.operands,
                "effective_address": hex_u(int(effect["effective_address"]), 64),
            },
            {
                "register_writes": [
                    {"name": name, "before": before["gpr"][name], "after": after["gpr"][name], "mask": hex_u(MASK64, 64)}
                    for name in s1.DATA_REGS
                    if before["gpr"][name] != after["gpr"][name]
                ],
                "memory_reads": [s1.event_record(e) for e in effect["memory_reads"]],
                "memory_writes": [s1.event_record(e) for e in effect["memory_writes"]],
                "flags_written": [],
                "undefined_flags": [],
                "exception": None,
            },
            {
                "before_data_regs": data_regs,
                "after_data_regs": out_data,
                "effective_offset": int(effect["effective_offset"]),
                "effective_address": int(effect["effective_address"]),
            },
        )

    def apply_stage2(self, index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
        before = self.state.snapshot()
        op = s2.OPS[s2.OP_INDEX[str(operation["op_name"])]]
        imm = int(operation.get("imm", 0))
        op_len = 5 if op.name == "CALL_REL32" else 2 if op.operands in ("ri", "rel8") or op.name == "TEST_AL_AL" else 1
        rip = int(self.state.gpr["RIP"])
        fallthrough = rip + op_len
        target = int(operation.get("target", fallthrough))
        rel = s2.rel32(target, fallthrough) if op.name == "CALL_REL32" else s2.rel8(target, fallthrough) if op.operands == "rel8" else 0
        rsp_offset = int(self.state.gpr["RSP"]) - s2.STACK_BASE
        stack_before = self.state.memory["stage2_stack"]
        branch_taken = op.name == "JMP_REL8" or op.name == "CALL_REL32" or (op.family == "COND_BR" and s2.condition_taken(op.name, self.state.flags))
        out_rax, out_flags, out_rip, out_rsp, out_stack, effect = s2.apply_op(
            self.state.gpr["RAX"],
            self.state.flags,
            rip,
            rsp_offset,
            stack_before,
            s2.OP_INDEX[op.name],
            imm,
            fallthrough,
            target,
        )
        self.state.gpr["RAX"] = u64(out_rax)
        self.state.flags = [int(x) for x in out_flags]
        self.state.gpr["RIP"] = u64(out_rip)
        self.state.gpr["RSP"] = u64(s2.STACK_BASE + int(out_rsp))
        self.state.memory["stage2_stack"] = out_stack
        after = self.state.snapshot()
        reg_writes = []
        if before["gpr"]["RAX"] != after["gpr"]["RAX"]:
            reg_writes.append({"name": "RAX", "before": before["gpr"]["RAX"], "after": after["gpr"]["RAX"], "mask": hex_u(MASK64 if op.name == "POP_RAX" else 0xFF, 64)})
        if before["gpr"]["RSP"] != after["gpr"]["RSP"]:
            reg_writes.append({"name": "RSP", "before": before["gpr"]["RSP"], "after": after["gpr"]["RSP"], "mask": hex_u(MASK64, 64)})
        return self._record_step(
            index,
            "stage2_control_flow",
            copy_jsonable(operation),
            before,
            after,
            {
                "bytes_hex": s2.encode_x86_bytes(op, imm, rel),
                "asm": s2.asm_string(op, imm, target),
                "iform": op.iform,
                "category": op.family,
                "operands": op.operands,
                "fallthrough_rip": hex_u(fallthrough, 64),
                "target_rip": hex_u(target, 64),
                "branch_taken": bool(branch_taken),
            },
            {
                "register_writes": reg_writes,
                "memory_reads": [s2.event_record(e) for e in effect["memory_reads"]],
                "memory_writes": [s2.event_record(e) for e in effect["memory_writes"]],
                "flags_written": list(FLAGS) if op.writes_flags else [],
                "undefined_flags": [],
                "exception": None,
            },
            {
                "before_rsp_offset": rsp_offset,
                "after_rsp_offset": int(out_rsp),
                "before_stack_digest": memory_digest(stack_before),
                "after_stack_digest": memory_digest(out_stack),
                "branch_taken": bool(branch_taken),
            },
        )

    def apply_stage3(self, index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
        before = self.state.snapshot()
        op = s3.OPS[s3.OP_INDEX[str(operation["op_name"])]]
        dst = int(operation.get("dst", 0))
        src = int(operation.get("src", 0))
        imm = int(operation.get("imm", 0))
        mem_offset = operation.get("mem_offset")
        regs = [self.state.gpr[name] for name in s3.REGS]
        memory_before = self.state.memory["stage3_data"]
        mem_value = None
        mem_addr = 0
        if op.operands in s3.MEMORY_OPERANDS:
            mem_offset = 0 if mem_offset is None else int(mem_offset)
            mem_addr = s3.MEM_BASE + int(mem_offset)
            mem_value = s3.read_memory_u64(memory_before, int(mem_offset))
        missing_reads = [name for name in s3.flag_reads(op) if not self.state.flag_defined[s3.FLAG_INDEX[name]]]
        out_regs, out_flags, flag_mask = s3.apply_op(regs, self.state.flags, s3.OP_INDEX[op.name], dst, src, imm, mem_value)
        out_memory = s3.apply_memory_effect(memory_before, s3.OP_INDEX[op.name], None if mem_offset is None else int(mem_offset), mem_value, imm, regs, src, self.state.flags)
        if out_memory is None:
            out_memory = list(memory_before)
        undefined_flags = s3.undefined_flag_names(op, flag_mask)
        self.state.flag_defined = s3.flags_defined_after(self.state.flag_defined, flag_mask, undefined_flags)
        for i, name in enumerate(s3.REGS):
            self.state.gpr[name] = u64(out_regs[i])
        self.state.flags = [int(x) for x in out_flags]
        self.state.memory["stage3_data"] = out_memory
        bytes_hex = s3.encode_x86_bytes(op, dst, src, imm, mem_addr)
        self.state.gpr["RIP"] = u64(self.state.gpr["RIP"] + len(bytes.fromhex(bytes_hex)))
        after = self.state.snapshot()
        reads = []
        writes = []
        if mem_value is not None and mem_offset is not None:
            width = s3.memory_operand_width(op)
            reads.append({
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "value_hex": hex_u(mem_value & ((1 << (8 * width)) - 1), 8 * width),
                "segment": "flat",
                "fault": None,
            })
        if op.name in s3.MEMORY_WRITE_OPS and mem_value is not None and mem_offset is not None:
            writes.append({
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "value_hex": hex_u(s3.read_memory_u64(out_memory, int(mem_offset)), 64),
                "segment": "flat",
                "fault": None,
            })
        return self._record_step(
            index,
            "stage3_complex_scalar",
            copy_jsonable(operation),
            before,
            after,
            {
                "bytes_hex": bytes_hex,
                "asm": s3.asm_string(op, dst, src, imm, mem_addr),
                "iform": op.iform,
                "category": op.family,
                "extension": s3.extension_for_op(op),
                "operands": s3.decode_operands(op, dst, src, imm, mem_addr),
            },
            {
                "register_writes": [
                    {"name": name, "before": before["gpr"][name], "after": after["gpr"][name], "mask": hex_u(MASK64, 64)}
                    for name in s3.REGS
                    if before["gpr"][name] != after["gpr"][name]
                ],
                "memory_reads": reads,
                "memory_writes": writes,
                "flags_written": [FLAGS[i] for i, bit in enumerate(flag_mask) if bit],
                "undefined_flags": undefined_flags,
                "flag_reads": s3.flag_reads(op),
                "undefined_flag_reads": missing_reads,
                "exception": op.expected_exception,
            },
            {
                "before_regs": regs,
                "after_regs": out_regs,
                "before_flags": before["rflags"],
                "after_flags": after["rflags"],
                "flag_mask": flag_mask,
                "before_flag_defined": before["flag_defined"],
                "after_flag_defined": after["flag_defined"],
                "mem_offset": mem_offset,
            },
        )

    def apply_stage4(self, index: int, operation: Dict[str, Any]) -> Dict[str, Any]:
        before = self.state.snapshot()
        op = s4.OPS[s4.OP_INDEX[str(operation["op_name"])]]
        dst = int(operation.get("dst", 0))
        src = int(operation.get("src", 0))
        imm = int(operation.get("imm", 0))
        mem_offset = operation.get("mem_offset")
        mem_addr = 0
        src_value = None
        memory_before = self.state.memory["stage4_data"]
        if op.operands in s4.MEMORY_OPERANDS:
            mem_offset = 0 if mem_offset is None else int(mem_offset)
            mem_addr = s4.MEM_BASE + int(mem_offset)
            if op.operands in s4.MEMORY_SOURCE_OPERANDS:
                src_value = s4.read_memory_operand(memory_before, int(mem_offset), op)
        before_xmm = list(self.state.xmm)
        out_xmm = s4.apply_op(before_xmm, s4.OP_INDEX[op.name], dst, src, imm, src_value)
        out_memory = list(memory_before)
        if op.operands in s4.MEMORY_DEST_OPERANDS and mem_offset is not None:
            out_memory = s4.write_memory_operand(memory_before, int(mem_offset), op, before_xmm[src])
        self.state.xmm = [int(x) for x in out_xmm]
        self.state.memory["stage4_data"] = out_memory
        bytes_hex = s4.encode_x86_bytes(op, dst, src, imm, mem_addr)
        self.state.gpr["RIP"] = u64(self.state.gpr["RIP"] + len(bytes.fromhex(bytes_hex)))
        after = self.state.snapshot()
        reads = []
        writes = []
        if op.operands in s4.MEMORY_SOURCE_OPERANDS and mem_offset is not None and src_value is not None:
            width = s4.memory_operand_width_bytes(op)
            reads.append({"address": hex_u(mem_addr, 64), "width_bytes": width, "value_hex": hex_u(src_value, width * 8), "segment": "flat", "fault": None})
        if op.operands in s4.MEMORY_DEST_OPERANDS and mem_offset is not None:
            width = s4.memory_operand_width_bytes(op)
            writes.append({
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "value_hex": hex_u(s4.memory_store_value(op, before_xmm[src]), width * 8),
                "segment": "flat",
                "fault": None,
            })
        return self._record_step(
            index,
            "stage4_simd_x87",
            copy_jsonable(operation),
            before,
            after,
            {
                "bytes_hex": bytes_hex,
                "asm": s4.asm_string(op, dst, src, imm, mem_addr),
                "iform": op.iform,
                "category": op.family,
                "operands": op.operands,
                "memory_alignment": op.memory_alignment,
            },
            {
                "register_writes": [
                    {"name": s4.XMM_REGS[i], "before": before["xmm"][s4.XMM_REGS[i]], "after": after["xmm"][s4.XMM_REGS[i]], "mask": hex_u((1 << 128) - 1, 128)}
                    for i in range(len(s4.XMM_REGS))
                    if before["xmm"][s4.XMM_REGS[i]] != after["xmm"][s4.XMM_REGS[i]]
                ],
                "memory_reads": reads,
                "memory_writes": writes,
                "flags_written": [],
                "undefined_flags": [],
                "exception": op.expected_exception,
            },
            {"before_xmm": before_xmm, "after_xmm": out_xmm, "mem_offset": mem_offset},
        )


def demo_operations() -> List[Dict[str, Any]]:
    return [
        {"stage": "stage0", "op_name": "MOV_RI", "dst": 0, "src": 0, "imm": 0x12, "word_bits": 8},
        {"stage": "stage0", "op_name": "ADD_RI", "dst": 0, "src": 0, "imm": 0x30, "word_bits": 8},
        {"stage": "stage1", "op_name": "MOV_STORE_MR64", "dst": 0, "src": 0, "base": 0, "index": 0, "scale": 1, "disp": 0, "imm": 0},
        {"stage": "stage1", "op_name": "MOV_LOAD_RM64", "dst": 1, "src": 0, "base": 0, "index": 0, "scale": 1, "disp": 0, "imm": 0},
        {"stage": "stage1", "op_name": "LEA_RA", "dst": 2, "src": 0, "base": 0, "index": 0, "scale": 2, "disp": 4, "imm": 0},
        {"stage": "stage2", "op_name": "CMP_AL_IMM", "imm": 0x42},
        {"stage": "stage2", "op_name": "JZ_REL8", "target": FULL_CODE_BASE + 0x80},
        {"stage": "stage2", "op_name": "CALL_REL32", "target": FULL_CODE_BASE + 0xA0},
        {"stage": "stage2", "op_name": "PUSH_RAX"},
        {"stage": "stage2", "op_name": "MOV_AL_IMM", "imm": 0x7E},
        {"stage": "stage2", "op_name": "POP_RAX"},
        {"stage": "stage2", "op_name": "RET_NEAR"},
        {"stage": "stage3", "op_name": "ADD_R64_R64", "dst": 0, "src": 1, "imm": 0},
        {"stage": "stage3", "op_name": "BTS_M64_IMM8", "dst": 0, "src": 0, "imm": 5, "mem_offset": 0},
        {"stage": "stage4", "op_name": "MOVDQA_XMM_XMM", "dst": 2, "src": 0, "imm": 0},
        {"stage": "stage4", "op_name": "PADDQ_XMM_XMM", "dst": 2, "src": 1, "imm": 0},
        {"stage": "stage4", "op_name": "MOVDQU_M128_XMM", "dst": 0, "src": 2, "imm": 0, "mem_offset": 0},
        {"stage": "stage4", "op_name": "MOVDQU_XMM_M128", "dst": 3, "src": 0, "imm": 0, "mem_offset": 0},
    ]


def build_trace() -> Dict[str, Any]:
    model = FullStageModel(FullState.initial())
    initial_state = model.state.raw_json()
    steps = []
    for index, operation in enumerate(demo_operations()):
        steps.append(model.apply(index, operation))
    return {
        "schema_version": "pan-x86-full-stage-trace-v1",
        "stage": STAGE,
        "substage": SUBSTAGE,
        "trace_id": "full-stage-cumulative-demo-v1",
        "source": {
            "generator": "stage5_full_model_pan.py",
            "stages": ["stage0_low_byte_alu", "stage1_address_memory", "stage2_control_flow", "stage3_complex_scalar", "stage4_simd_x87"],
            "oracle": ["composed_structured_stage_semantics"],
        },
        "mode": {"long_mode": True, "operand_bits": 64, "address_bits": 64, "ring": 0, "cpuid_features": list(s3.CPUID_FEATURES)},
        "initial_state": initial_state,
        "program": {
            "asm": [step["decode"]["asm"] for step in steps],
            "bytes_hex_by_step": [step["decode"]["bytes_hex"] for step in steps],
            "length": len(steps),
            "template_id": "cumulative_stages_0_4_component_program",
        },
        "steps": steps,
        "final": steps[-1]["after"],
    }


def compare_step(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a["after"] == b["after"] and a["effects"] == b["effects"] and a["decode"] == b["decode"]


def replay_trace(trace: Dict[str, Any], mutate: Optional[str] = None) -> List[Dict[str, Any]]:
    model = FullStageModel(FullState.from_json(trace["initial_state"]))
    replayed = []
    for index, step in enumerate(trace["steps"]):
        op = copy.deepcopy(step["operation"])
        if mutate == "wrong_stage3_opcode" and step["stage"] == "stage3_complex_scalar":
            op["op_name"] = "SUB_R64_R64"
            mutate = None
        elif mutate == "wrong_stack_action" and step["decode"]["category"] == "CALL":
            op["op_name"] = "NOP"
            mutate = None
        elif mutate == "wrong_xmm_source" and step["stage"] == "stage4_simd_x87" and op.get("op_name") == "PADDQ_XMM_XMM":
            op["src"] = 0
            mutate = None
        replayed.append(model.apply(index, op))
    return replayed


def validate_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
    replayed = replay_trace(trace)
    exact = [compare_step(a, b) for a, b in zip(trace["steps"], replayed)]
    by_stage: Dict[str, Dict[str, int]] = {}
    component_touch = {
        "gpr": False,
        "rflags": False,
        "rip": False,
        "rsp": False,
        "stack_memory": False,
        "data_memory": False,
        "stage3_memory": False,
        "xmm": False,
        "stage4_memory": False,
        "branch": False,
        "call_ret": False,
    }
    stage3_undefined_reads = 0
    failure_examples = []
    for expected, actual, ok in zip(trace["steps"], replayed, exact):
        stage = expected["stage"]
        bucket = by_stage.setdefault(stage, {"steps": 0, "exact": 0})
        bucket["steps"] += 1
        bucket["exact"] += int(ok)
        before = expected["before"]
        after = expected["after"]
        effects = expected["effects"]
        if before["gpr"] != after["gpr"]:
            component_touch["gpr"] = True
        if before["rflags"] != after["rflags"]:
            component_touch["rflags"] = True
        if before["rip"] != after["rip"]:
            component_touch["rip"] = True
        if before["gpr"].get("RSP") != after["gpr"].get("RSP"):
            component_touch["rsp"] = True
        if expected["decode"]["category"] in {"COND_BR", "UNCOND_BR"}:
            component_touch["branch"] = True
        if expected["decode"]["category"] in {"CALL", "RET"}:
            component_touch["call_ret"] = True
        for event in effects["memory_reads"] + effects["memory_writes"]:
            address = int(str(event["address"]), 16)
            if s2.STACK_BASE <= address < s2.STACK_BASE + s2.STACK_SIZE:
                component_touch["stack_memory"] = True
            if s1.MEM_BASE <= address < s1.MEM_BASE + s1.MEM_SIZE:
                component_touch["data_memory"] = True
            if s3.MEM_BASE <= address < s3.MEM_BASE + s3.MEM_SIZE:
                component_touch["stage3_memory"] = True
            if s4.MEM_BASE <= address < s4.MEM_BASE + s4.MEM_SIZE:
                component_touch["stage4_memory"] = True
        if stage == "stage4_simd_x87" and effects["register_writes"]:
            component_touch["xmm"] = True
        if stage == "stage3_complex_scalar":
            stage3_undefined_reads += len(effects.get("undefined_flag_reads", []))
        if not ok and len(failure_examples) < 8:
            failure_examples.append({
                "index": expected["index"],
                "stage": stage,
                "asm": expected["decode"]["asm"],
                "expected_after": expected["after"],
                "actual_after": actual["after"],
            })
    required_stages = {"stage0_low_byte_alu", "stage1_address_memory", "stage2_control_flow", "stage3_complex_scalar", "stage4_simd_x87"}
    negative = {
        "wrong_stage3_opcode_exact": sum(compare_step(a, b) for a, b in zip(trace["steps"], replay_trace(trace, "wrong_stage3_opcode"))) / max(1, len(trace["steps"])),
        "wrong_stack_action_exact": sum(compare_step(a, b) for a, b in zip(trace["steps"], replay_trace(trace, "wrong_stack_action"))) / max(1, len(trace["steps"])),
        "wrong_xmm_source_exact": sum(compare_step(a, b) for a, b in zip(trace["steps"], replay_trace(trace, "wrong_xmm_source"))) / max(1, len(trace["steps"])),
    }
    gate = (
        all(exact)
        and set(by_stage) == required_stages
        and all(component_touch.values())
        and stage3_undefined_reads == 0
        and all(value < 1.0 for value in negative.values())
    )
    return {
        "schema_version": "pan-x86-full-stage-validation-v1",
        "stage": STAGE,
        "substage": SUBSTAGE,
        "steps": len(trace["steps"]),
        "step_exact": round(sum(exact) / max(1, len(exact)), 6),
        "by_stage": {stage: {"steps": v["steps"], "exact": round(v["exact"] / max(1, v["steps"]), 6)} for stage, v in sorted(by_stage.items())},
        "component_touch": component_touch,
        "stage3_continuous_undefined_flag_reads": stage3_undefined_reads,
        "negative_controls": {key: round(value, 6) for key, value in negative.items()},
        "failure_examples": failure_examples,
        "gate_status": {
            "full_stage_model_gate_passed": bool(gate),
            "all_stage_slices_present": set(by_stage) == required_stages,
            "all_required_components_touched": all(component_touch.values()),
            "full_x86_goal_complete": False,
        },
    }


def write_inspection(trace: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    rows = []
    for step in trace["steps"]:
        rows.append({
            "index": step["index"],
            "stage": step["stage"],
            "asm": step["decode"]["asm"],
            "rip_before": step["before"]["rip"],
            "rip_after": step["after"]["rip"],
            "rax_before": step["before"]["gpr"]["RAX"],
            "rax_after": step["after"]["gpr"]["RAX"],
            "rsp_before": step["before"]["gpr"]["RSP"],
            "rsp_after": step["after"]["gpr"]["RSP"],
            "memory_reads": step["effects"]["memory_reads"],
            "memory_writes": step["effects"]["memory_writes"],
            "register_writes": step["effects"]["register_writes"],
        })
    script = [
        "set pagination off",
        "set disassembly-flavor intel",
        "echo PAN full-stage trace is offline JSON, not a live inferior.\\n",
        f"echo Trace: {out_dir / 'full_stage_trace.json'}\\n",
        f"echo Inspection: {out_dir / 'full_stage_inspection.json'}\\n",
        "quit",
    ]
    write_json(out_dir / "full_stage_inspection.json", rows)
    (out_dir / "full_stage_gdb_notes.gdb").write_text("\n".join(script) + "\n", encoding="utf-8")
    return {"inspection_path": str(out_dir / "full_stage_inspection.json"), "gdb_notes": str(out_dir / "full_stage_gdb_notes.gdb")}


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = build_trace()
    validation = validate_trace(trace)
    inspection = write_inspection(trace, out_dir)
    write_json(out_dir / "full_stage_trace.json", trace)
    write_json(out_dir / "full_stage_validation_report.json", validation)
    summary = {
        "stage": STAGE,
        "substage": SUBSTAGE,
        "out_dir": str(out_dir),
        "trace_path": str(out_dir / "full_stage_trace.json"),
        "validation_path": str(out_dir / "full_stage_validation_report.json"),
        "inspection": inspection,
        "gate_status": validation["gate_status"],
        "validation": {
            "step_exact": validation["step_exact"],
            "by_stage": validation["by_stage"],
            "component_touch": validation["component_touch"],
            "stage3_continuous_undefined_flag_reads": validation["stage3_continuous_undefined_flag_reads"],
            "negative_controls": validation["negative_controls"],
        },
    }
    write_json(out_dir / "full_stage_summary.json", summary)
    return summary


def cmd_run_smoke(args: argparse.Namespace) -> int:
    summary = run_smoke(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["gate_status"]["full_stage_model_gate_passed"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    report = validate_trace(trace)
    out = args.out or (args.trace.parent / "full_stage_validation_report.json")
    write_json(out, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["gate_status"]["full_stage_model_gate_passed"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run-smoke")
    run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    run.set_defaults(func=cmd_run_smoke)

    validate = sub.add_parser("validate")
    validate.add_argument("--trace", type=Path, default=DEFAULT_OUT_DIR / "full_stage_trace.json")
    validate.add_argument("--out", type=Path, default=None)
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
