#!/usr/bin/env python3
"""Stage 0 PAN x86-lite runner.

This is the first runnable implementation for the pan-x86-isa experiment. It is
deliberately narrow: low-byte ALU semantics for AL/CL/DL/BL, represented as a
small modulo word so smoke runs complete quickly. The script writes oracle
traces, trains a phase-feature PAN transition model with per-step supervision,
validates teacher-forced and free-run execution, and samples CPU/memory/disk
usage before and during the run.

The full x86 goal remains larger than this stage. This runner is the executable
Stage 0 foothold and the artifact format testbed.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import jsonschema
except Exception:  # pragma: no cover - optional validation nicety
    jsonschema = None


EXP_ROOT = Path(os.environ.get("PAN_X86_EXP_ROOT", Path(__file__).resolve().parents[1])).expanduser().resolve()
ROOT = EXP_ROOT.parent
DEFAULT_ARTIFACT_ROOT = Path(os.environ.get("PAN_X86_ARTIFACT_BASE", ROOT / "data" / "artifacts")).expanduser().resolve()
DEFAULT_DATE = "2026-06-16"
DEFAULT_ORACLE_DEP_PATH = DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / "python-deps" / "unicorn"

REGS = ("AL", "CL", "DL", "BL")
PARENT_REGS = {"AL": "RAX", "CL": "RCX", "DL": "RDX", "BL": "RBX"}
FLAGS = ("CF", "ZF", "SF", "OF")
SPLITS = ("train", "heldout_imm", "heldout_reg_tuple", "full")
FLAG_BITS = {"CF": 0, "ZF": 6, "SF": 7, "OF": 11}


@dataclasses.dataclass(frozen=True)
class OpSpec:
    name: str
    iform: str
    family: str
    operands: str  # rr, ri, r, none
    writes_dst: bool
    writes_flags: bool


OPS: Tuple[OpSpec, ...] = (
    OpSpec("MOV_RI", "MOV_GPR8_IMMb", "MOV", "ri", True, False),
    OpSpec("MOV_RR", "MOV_GPR8_GPR8", "MOV", "rr", True, False),
    OpSpec("ADD_RR", "ADD_GPR8_GPR8", "ADD", "rr", True, True),
    OpSpec("ADD_RI", "ADD_GPR8_IMMb", "ADD", "ri", True, True),
    OpSpec("SUB_RR", "SUB_GPR8_GPR8", "SUB", "rr", True, True),
    OpSpec("SUB_RI", "SUB_GPR8_IMMb", "SUB", "ri", True, True),
    OpSpec("ADC_RR", "ADC_GPR8_GPR8", "ADC", "rr", True, True),
    OpSpec("SBB_RR", "SBB_GPR8_GPR8", "SBB", "rr", True, True),
    OpSpec("INC_R", "INC_GPR8", "INC", "r", True, True),
    OpSpec("DEC_R", "DEC_GPR8", "DEC", "r", True, True),
    OpSpec("NEG_R", "NEG_GPR8", "NEG", "r", True, True),
    OpSpec("AND_RR", "AND_GPR8_GPR8", "AND", "rr", True, True),
    OpSpec("AND_RI", "AND_GPR8_IMMb", "AND", "ri", True, True),
    OpSpec("OR_RR", "OR_GPR8_GPR8", "OR", "rr", True, True),
    OpSpec("OR_RI", "OR_GPR8_IMMb", "OR", "ri", True, True),
    OpSpec("XOR_RR", "XOR_GPR8_GPR8", "XOR", "rr", True, True),
    OpSpec("XOR_RI", "XOR_GPR8_IMMb", "XOR", "ri", True, True),
    OpSpec("NOT_R", "NOT_GPR8", "NOT", "r", True, False),
    OpSpec("CMP_RR", "CMP_GPR8_GPR8", "CMP", "rr", False, True),
    OpSpec("CMP_RI", "CMP_GPR8_IMMb", "CMP", "ri", False, True),
    OpSpec("TEST_RR", "TEST_GPR8_GPR8", "TEST", "rr", False, True),
    OpSpec("SHL1_R", "SHL_GPR8_1", "SHIFT", "r", True, True),
    OpSpec("SHR1_R", "SHR_GPR8_1", "SHIFT", "r", True, True),
    OpSpec("ROL1_R", "ROL_GPR8_1", "ROTATE", "r", True, True),
    OpSpec("ROR1_R", "ROR_GPR8_1", "ROTATE", "r", True, True),
)

OP_INDEX = {op.name: i for i, op in enumerate(OPS)}
REG_INDEX = {reg: i for i, reg in enumerate(REGS)}
OP_WRITES_DST = torch.tensor([int(op.writes_dst) for op in OPS], dtype=torch.bool)
OP_WRITES_FLAGS = torch.tensor([int(op.writes_flags) for op in OPS], dtype=torch.bool)
OP_USES_IMM = torch.tensor([int(op.operands == "ri") for op in OPS], dtype=torch.bool)
VALUE_CANDIDATE_INDEX = {
    "MOV_RI": 1,
    "MOV_RR": 1,
    "ADD_RR": 2,
    "ADD_RI": 2,
    "SUB_RR": 4,
    "SUB_RI": 4,
    "ADC_RR": 3,
    "SBB_RR": 5,
    "INC_R": 6,
    "DEC_R": 7,
    "NEG_R": 8,
    "AND_RR": 9,
    "AND_RI": 9,
    "OR_RR": 10,
    "OR_RI": 10,
    "XOR_RR": 11,
    "XOR_RI": 11,
    "NOT_R": 12,
    "CMP_RR": 4,
    "CMP_RI": 4,
    "TEST_RR": 9,
    "SHL1_R": 13,
    "SHR1_R": 14,
    "ROL1_R": 15,
    "ROR1_R": 16,
}
FLAG_CANDIDATE_INDEX = {
    "MOV_RI": 0,
    "MOV_RR": 0,
    "ADD_RR": 1,
    "ADD_RI": 1,
    "SUB_RR": 3,
    "SUB_RI": 3,
    "ADC_RR": 2,
    "SBB_RR": 4,
    "INC_R": 5,
    "DEC_R": 6,
    "NEG_R": 7,
    "AND_RR": 8,
    "AND_RI": 8,
    "OR_RR": 9,
    "OR_RI": 9,
    "XOR_RR": 10,
    "XOR_RI": 10,
    "NOT_R": 0,
    "CMP_RR": 3,
    "CMP_RI": 3,
    "TEST_RR": 8,
    "SHL1_R": 11,
    "SHR1_R": 12,
    "ROL1_R": 13,
    "ROR1_R": 14,
}
OP_VALUE_CANDIDATE = torch.tensor([VALUE_CANDIDATE_INDEX[op.name] for op in OPS], dtype=torch.long)
OP_FLAG_CANDIDATE = torch.tensor([FLAG_CANDIDATE_INDEX[op.name] for op in OPS], dtype=torch.long)


def now_s() -> float:
    return time.time()


def artifact_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


class ResourceMonitor:
    def __init__(self, out_dir: Path, interval_s: float = 1.0):
        self.out_dir = out_dir
        self.interval_s = interval_s
        self.path = out_dir / "resource_monitor.jsonl"
        self.summary_path = out_dir / "resource_summary.json"
        self.proc = psutil.Process(os.getpid())
        self.proc.cpu_percent(None)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.samples: List[Dict[str, Any]] = []
        self.phase = "init"
        self.lock = threading.Lock()

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.sample("before", "monitor_start")
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self.lock:
            self.phase = phase
        self.sample(phase, "phase_start")

    def stop(self) -> None:
        self.sample(self.phase, "monitor_stop")
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_s * 2))
        self.write_summary()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            with self.lock:
                phase = self.phase
            self.sample(phase, "periodic")

    def sample(self, phase: str, event: str) -> None:
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage(self.out_dir if self.out_dir.exists() else self.out_dir.parent)
        try:
            io = self.proc.io_counters()._asdict()
        except Exception:
            io = {}
        rec = {
            "ts": now_s(),
            "phase": phase,
            "event": event,
            "pid": os.getpid(),
            "process_cpu_percent": self.proc.cpu_percent(None),
            "process_rss_bytes": self.proc.memory_info().rss,
            "process_vms_bytes": self.proc.memory_info().vms,
            "system_cpu_percent": psutil.cpu_percent(None),
            "system_mem_total_bytes": mem.total,
            "system_mem_available_bytes": mem.available,
            "system_mem_percent": mem.percent,
            "disk_total_bytes": disk.total,
            "disk_used_bytes": disk.used,
            "disk_free_bytes": disk.free,
            "artifact_size_bytes": artifact_size_bytes(self.out_dir),
            "process_io": io,
        }
        self.samples.append(rec)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def write_summary(self) -> None:
        if not self.samples:
            return
        summary = {
            "sample_count": len(self.samples),
            "started_ts": self.samples[0]["ts"],
            "finished_ts": self.samples[-1]["ts"],
            "duration_s": round(self.samples[-1]["ts"] - self.samples[0]["ts"], 4),
            "max_process_cpu_percent": max(s["process_cpu_percent"] for s in self.samples),
            "max_process_rss_bytes": max(s["process_rss_bytes"] for s in self.samples),
            "max_system_mem_percent": max(s["system_mem_percent"] for s in self.samples),
            "min_disk_free_bytes": min(s["disk_free_bytes"] for s in self.samples),
            "max_artifact_size_bytes": max(s["artifact_size_bytes"] for s in self.samples),
            "phases": sorted({s["phase"] for s in self.samples}),
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hex_u(value: int, bits: int) -> str:
    width = max(2, (bits + 3) // 4)
    return "0x" + format(value & ((1 << bits) - 1), f"0{width}x")


def flag_dict(flags: Sequence[int]) -> Dict[str, int]:
    return {name: int(flags[i]) for i, name in enumerate(FLAGS)}


def state_record(regs: Sequence[int], flags: Sequence[int], word_bits: int, rip: int = 0) -> Dict[str, Any]:
    return {
        "rip": hex_u(rip, 64),
        "gpr": {PARENT_REGS[reg]: hex_u(int(regs[i]), 64) for i, reg in enumerate(REGS)},
        "rflags": flag_dict(flags),
        "memory_digest": "sha256:empty",
        "exception": None,
    }


def parity8(x: int) -> int:
    return 1 if bin(x & 0xFF).count("1") % 2 == 0 else 0


def add_flags(a: int, b: int, result: int, bits: int, carry_in: int = 0) -> List[int]:
    mod = 1 << bits
    mask = mod - 1
    full = a + b + carry_in
    sign = 1 << (bits - 1)
    cf = int(full >= mod)
    zf = int((result & mask) == 0)
    sf = int(bool(result & sign))
    of = int(((~(a ^ b) & (a ^ result)) & sign) != 0)
    return [cf, zf, sf, of]


def sub_flags(a: int, b: int, result: int, bits: int, borrow_in: int = 0) -> List[int]:
    mask = (1 << bits) - 1
    sign = 1 << (bits - 1)
    rhs = b + borrow_in
    cf = int(a < rhs)
    zf = int((result & mask) == 0)
    sf = int(bool(result & sign))
    of = int((((a ^ rhs) & (a ^ result)) & sign) != 0)
    return [cf, zf, sf, of]


def logical_flags(result: int, bits: int) -> List[int]:
    sign = 1 << (bits - 1)
    return [0, int(result == 0), int(bool(result & sign)), 0]


def encode_x86_bytes(op: OpSpec, dst: int, src: int, imm: int, word_bits: int) -> str:
    """Return real x86 bytes for the low-byte subset when possible.

    The values may be narrower than 8 bits for smoke runs, but encodings still
    describe the corresponding low-byte x86 instruction form.
    """
    imm8 = imm & 0xFF
    modrm_rr = lambda reg, rm: 0xC0 | ((reg & 7) << 3) | (rm & 7)
    modrm_grp = lambda group, rm: 0xC0 | ((group & 7) << 3) | (rm & 7)
    b: List[int]
    if op.name == "MOV_RI":
        b = [0xB0 + dst, imm8]
    elif op.name == "MOV_RR":
        b = [0x88, modrm_rr(src, dst)]
    elif op.name == "ADD_RR":
        b = [0x00, modrm_rr(src, dst)]
    elif op.name == "ADD_RI":
        b = [0x80, modrm_grp(0, dst), imm8]
    elif op.name == "OR_RR":
        b = [0x08, modrm_rr(src, dst)]
    elif op.name == "OR_RI":
        b = [0x80, modrm_grp(1, dst), imm8]
    elif op.name == "ADC_RR":
        b = [0x10, modrm_rr(src, dst)]
    elif op.name == "SBB_RR":
        b = [0x18, modrm_rr(src, dst)]
    elif op.name == "AND_RR":
        b = [0x20, modrm_rr(src, dst)]
    elif op.name == "AND_RI":
        b = [0x80, modrm_grp(4, dst), imm8]
    elif op.name == "SUB_RR":
        b = [0x28, modrm_rr(src, dst)]
    elif op.name == "SUB_RI":
        b = [0x80, modrm_grp(5, dst), imm8]
    elif op.name == "XOR_RR":
        b = [0x30, modrm_rr(src, dst)]
    elif op.name == "XOR_RI":
        b = [0x80, modrm_grp(6, dst), imm8]
    elif op.name == "CMP_RR":
        b = [0x38, modrm_rr(src, dst)]
    elif op.name == "CMP_RI":
        b = [0x80, modrm_grp(7, dst), imm8]
    elif op.name == "TEST_RR":
        b = [0x84, modrm_rr(src, dst)]
    elif op.name == "INC_R":
        b = [0xFE, modrm_grp(0, dst)]
    elif op.name == "DEC_R":
        b = [0xFE, modrm_grp(1, dst)]
    elif op.name == "NOT_R":
        b = [0xF6, modrm_grp(2, dst)]
    elif op.name == "NEG_R":
        b = [0xF6, modrm_grp(3, dst)]
    elif op.name == "ROL1_R":
        b = [0xD0, modrm_grp(0, dst)]
    elif op.name == "ROR1_R":
        b = [0xD0, modrm_grp(1, dst)]
    elif op.name == "SHL1_R":
        b = [0xD0, modrm_grp(4, dst)]
    elif op.name == "SHR1_R":
        b = [0xD0, modrm_grp(5, dst)]
    else:
        raise ValueError(f"no encoder for {op.name}")
    return "".join(f"{x:02x}" for x in b)


def asm_string(op: OpSpec, dst: int, src: int, imm: int) -> str:
    d = REGS[dst]
    s = REGS[src]
    op0 = op.name.split("_")[0].lower()
    if op.operands == "rr":
        return f"{op0} {d}, {s}"
    if op.operands == "ri":
        return f"{op0} {d}, 0x{imm & 0xFF:02x}"
    if op.operands == "r":
        return f"{op0} {d}"
    return op0


def apply_op(
    regs: Sequence[int],
    flags: Sequence[int],
    op_idx: int,
    dst: int,
    src: int,
    imm: int,
    word_bits: int,
) -> Tuple[List[int], List[int]]:
    op = OPS[op_idx]
    mask = (1 << word_bits) - 1
    sign = 1 << (word_bits - 1)
    out_regs = [int(x) & mask for x in regs]
    out_flags = [int(x) for x in flags]
    a = out_regs[dst]
    b = out_regs[src] if op.operands == "rr" else (imm & mask)
    cf_in = out_flags[0]

    if op.name == "MOV_RI":
        out_regs[dst] = b
    elif op.name == "MOV_RR":
        out_regs[dst] = b
    elif op.name in ("ADD_RR", "ADD_RI"):
        res = (a + b) & mask
        out_regs[dst] = res
        out_flags = add_flags(a, b, res, word_bits)
    elif op.name == "ADC_RR":
        res = (a + b + cf_in) & mask
        out_regs[dst] = res
        out_flags = add_flags(a, b, res, word_bits, cf_in)
    elif op.name in ("SUB_RR", "SUB_RI", "CMP_RR", "CMP_RI"):
        res = (a - b) & mask
        if op.writes_dst:
            out_regs[dst] = res
        out_flags = sub_flags(a, b, res, word_bits)
    elif op.name == "SBB_RR":
        res = (a - b - cf_in) & mask
        out_regs[dst] = res
        out_flags = sub_flags(a, b, res, word_bits, cf_in)
    elif op.name == "INC_R":
        res = (a + 1) & mask
        old_cf = out_flags[0]
        out_regs[dst] = res
        tmp = add_flags(a, 1, res, word_bits)
        out_flags = [old_cf, tmp[1], tmp[2], tmp[3]]
    elif op.name == "DEC_R":
        res = (a - 1) & mask
        old_cf = out_flags[0]
        out_regs[dst] = res
        tmp = sub_flags(a, 1, res, word_bits)
        out_flags = [old_cf, tmp[1], tmp[2], tmp[3]]
    elif op.name == "NEG_R":
        res = (-a) & mask
        out_regs[dst] = res
        out_flags = sub_flags(0, a, res, word_bits)
        out_flags[0] = int(a != 0)
    elif op.name in ("AND_RR", "AND_RI", "TEST_RR"):
        res = a & b
        if op.writes_dst:
            out_regs[dst] = res
        out_flags = logical_flags(res, word_bits)
    elif op.name in ("OR_RR", "OR_RI"):
        res = a | b
        out_regs[dst] = res
        out_flags = logical_flags(res, word_bits)
    elif op.name in ("XOR_RR", "XOR_RI"):
        res = a ^ b
        out_regs[dst] = res
        out_flags = logical_flags(res, word_bits)
    elif op.name == "NOT_R":
        out_regs[dst] = (~a) & mask
    elif op.name == "SHL1_R":
        res = (a << 1) & mask
        out_regs[dst] = res
        cf = int(bool(a & sign))
        of = int(((res & sign) != 0) ^ bool(cf))
        out_flags = [cf, int(res == 0), int(bool(res & sign)), of]
    elif op.name == "SHR1_R":
        res = (a >> 1) & mask
        out_regs[dst] = res
        cf = a & 1
        of = int(bool(a & sign))
        out_flags = [cf, int(res == 0), int(bool(res & sign)), of]
    elif op.name == "ROL1_R":
        cf = int(bool(a & sign))
        res = ((a << 1) | cf) & mask
        out_regs[dst] = res
        of = int(bool((res & sign) >> (word_bits - 1)) ^ bool(cf))
        out_flags = [cf, out_flags[1], out_flags[2], of]
    elif op.name == "ROR1_R":
        cf = a & 1
        res = ((a >> 1) | (cf << (word_bits - 1))) & mask
        out_regs[dst] = res
        msb = int(bool(res & sign))
        next_msb = int(bool(res & (sign >> 1))) if word_bits > 1 else 0
        out_flags = [cf, out_flags[1], out_flags[2], msb ^ next_msb]
    else:
        raise ValueError(op.name)
    return out_regs, out_flags


def choose_instruction(rng: random.Random, split: str, word_bits: int) -> Tuple[int, int, int, int]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    op_idx = rng.randrange(len(OPS))
    op = OPS[op_idx]
    dst = rng.randrange(len(REGS))
    src = rng.randrange(len(REGS))
    max_value = 1 << word_bits
    if split == "train":
        imm_pool = list(range(0, max_value, 2)) or [0]
    elif split == "heldout_imm":
        imm_pool = list(range(1, max_value, 2)) or [0]
    elif split == "full":
        imm_pool = list(range(max_value))
    else:
        imm_pool = list(range(max_value))
    imm = rng.choice(imm_pool)
    if split == "heldout_reg_tuple":
        dst = (src + 1) % len(REGS)
    return op_idx, dst, src, imm


def make_trace(trace_id: str, split: str, length: int, rng: random.Random, word_bits: int) -> Dict[str, Any]:
    max_value = 1 << word_bits
    regs = [rng.randrange(max_value) for _ in REGS]
    flags = [rng.randrange(2) for _ in FLAGS]
    steps = []
    all_bytes = []
    asm_lines = []
    rip = 0x400000
    for i in range(length):
        op_idx, dst, src, imm = choose_instruction(rng, split, word_bits)
        op = OPS[op_idx]
        before_regs, before_flags = list(regs), list(flags)
        regs, flags = apply_op(regs, flags, op_idx, dst, src, imm, word_bits)
        bytes_hex = encode_x86_bytes(op, dst, src, imm, word_bits)
        asm = asm_string(op, dst, src, imm)
        all_bytes.append(bytes_hex)
        asm_lines.append(asm)
        before = state_record(before_regs, before_flags, word_bits, rip)
        rip_after = rip + len(bytes.fromhex(bytes_hex))
        after = state_record(regs, flags, word_bits, rip_after)
        reg_writes = []
        for ridx, reg in enumerate(REGS):
            if before_regs[ridx] != regs[ridx]:
                reg_writes.append({
                    "name": PARENT_REGS[reg],
                    "before": hex_u(before_regs[ridx], 64),
                    "after": hex_u(regs[ridx], 64),
                    "mask": hex_u((1 << word_bits) - 1, 64),
                })
        steps.append({
            "index": i,
            "decode": {
                "bytes_hex": bytes_hex,
                "iclass": op.name.split("_")[0],
                "iform": op.iform,
                "category": op.family,
                "extension": "BASE",
                "operands": [
                    {"kind": "reg", "role": "dst", "name": REGS[dst]},
                    {"kind": "reg", "role": "src", "name": REGS[src]} if op.operands == "rr"
                    else {"kind": "imm", "role": "src", "value": imm & ((1 << word_bits) - 1)}
                    if op.operands == "ri"
                    else {"kind": "none", "role": "src"},
                ],
            },
            "before": before,
            "after": after,
            "effects": {
                "register_writes": reg_writes,
                "memory_reads": [],
                "memory_writes": [],
                "flags_written": list(FLAGS) if op.writes_flags else [],
                "undefined_flags": [],
                "exception": None,
            },
            "pan_stage0": {
                "op_index": op_idx,
                "dst": dst,
                "src": src,
                "imm": imm & ((1 << word_bits) - 1),
                "before_regs": before_regs,
                "before_flags": before_flags,
                "after_regs": regs,
                "after_flags": flags,
            },
            "oracle_disagreement": None,
        })
        rip = rip_after
    return {
        "schema_version": "pan-x86-trace-v1",
        "trace_id": trace_id,
        "stage": "stage0_x86_lite_alu",
        "source": {
            "generator": "stage0_x86_pan.py",
            "seed": trace_id,
            "xed_version": "not-yet-integrated-stage0-internal-encoder",
            "intel_sdm_hash": "not-yet-frozen",
            "amd_apm_revision": "not-yet-frozen",
            "oracle": ["stage0_python_semantics"],
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
        "final": state_record(regs, flags, word_bits, rip),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


def load_traces(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def make_manifest(out_dir: Path, word_bits: int) -> Dict[str, Any]:
    forms = []
    for i, op in enumerate(OPS):
        forms.append(dataclasses.asdict(op) | {
            "op_index": i,
            "implemented": True,
            "encoding_policy": "real low-byte x86 encoding for AL/CL/DL/BL forms",
        })
    manifest = {
        "stage": "stage0_x86_lite_alu",
        "word_bits": word_bits,
        "registers": list(REGS),
        "parent_registers": PARENT_REGS,
        "flags": list(FLAGS),
        "forms": forms,
        "limitations": [
            "low-byte ALU subset only",
            "value domain may be narrower than 8 bits for smoke runs",
            "Intel XED and external execution oracles are not integrated in this Stage 0 runner",
            "no memory, branches, SIMD/x87, exceptions, prefixes, or mode variation",
        ],
    }
    write_json(out_dir / "coverage_manifest.json", manifest)
    return manifest


def validate_trace_schema(out_dir: Path, trace_path: Path) -> Dict[str, Any]:
    schema_path = EXP_ROOT / "schemas" / "trace_record.schema.json"
    result = {
        "schema_path": str(schema_path),
        "trace_path": str(trace_path),
        "jsonschema_available": jsonschema is not None,
        "checked_records": 0,
        "ok": False,
        "error": None,
    }
    if jsonschema is None:
        result["ok"] = True
        return result
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    for row in load_traces(trace_path)[:25]:
        errors = sorted(validator.iter_errors(row), key=lambda e: list(e.path))
        if errors:
            result["error"] = f"{list(errors[0].path)}: {errors[0].message}"
            write_json(out_dir / "schema_validation.json", result)
            return result
        result["checked_records"] += 1
    result["ok"] = True
    write_json(out_dir / "schema_validation.json", result)
    return result


def add_oracle_pythonpath(path: Optional[Path]) -> None:
    if path is not None and path.exists():
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def pack_eflags(flags: Sequence[int]) -> int:
    eflags = 0x2
    for i, name in enumerate(FLAGS):
        if int(flags[i]):
            eflags |= 1 << FLAG_BITS[name]
    return eflags


def unpack_eflags(eflags: int) -> List[int]:
    return [int(bool(eflags & (1 << FLAG_BITS[name]))) for name in FLAGS]


def unicorn_step(bytes_hex: str, regs: Sequence[int], flags: Sequence[int]) -> Tuple[List[int], List[int]]:
    from unicorn import Uc, UC_ARCH_X86, UC_MODE_64
    from unicorn.x86_const import (
        UC_X86_REG_EFLAGS,
        UC_X86_REG_RAX,
        UC_X86_REG_RBX,
        UC_X86_REG_RCX,
        UC_X86_REG_RDX,
        UC_X86_REG_RIP,
    )

    code = bytes.fromhex(bytes_hex)
    base = 0x100000
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(base, 0x1000)
    uc.mem_write(base, code)
    uc.reg_write(UC_X86_REG_RIP, base)
    uc.reg_write(UC_X86_REG_RAX, int(regs[0]) & 0xFF)
    uc.reg_write(UC_X86_REG_RCX, int(regs[1]) & 0xFF)
    uc.reg_write(UC_X86_REG_RDX, int(regs[2]) & 0xFF)
    uc.reg_write(UC_X86_REG_RBX, int(regs[3]) & 0xFF)
    uc.reg_write(UC_X86_REG_EFLAGS, pack_eflags(flags))
    uc.emu_start(base, base + len(code), count=1)
    out_regs = [
        int(uc.reg_read(UC_X86_REG_RAX)) & 0xFF,
        int(uc.reg_read(UC_X86_REG_RCX)) & 0xFF,
        int(uc.reg_read(UC_X86_REG_RDX)) & 0xFF,
        int(uc.reg_read(UC_X86_REG_RBX)) & 0xFF,
    ]
    out_flags = unpack_eflags(int(uc.reg_read(UC_X86_REG_EFLAGS)))
    return out_regs, out_flags


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
    if args.word_bits != 8:
        report.update({"ok": False, "error": "unicorn oracle requires word_bits=8"})
        return report
    try:
        __import__("unicorn")
    except Exception as exc:
        report.update({"ok": False, "error": f"could not import unicorn: {exc}"})
        return report

    max_steps = int(getattr(args, "oracle_max_steps", 10000))
    for split, path in (("val", out_dir / "val_traces.jsonl"), ("long", out_dir / "long_traces.jsonl")):
        split_checked = 0
        split_ok = 0
        for trace in load_traces(path):
            for step in trace["steps"]:
                if report["checked_steps"] >= max_steps:
                    break
                p = step["pan_stage0"]
                try:
                    regs, flags = unicorn_step(step["decode"]["bytes_hex"], p["before_regs"], p["before_flags"])
                except Exception as exc:
                    report["ok"] = False
                    report["error"] = f"unicorn execution failed: {exc}"
                    return report
                ok = regs == p["after_regs"] and flags == p["after_flags"]
                split_checked += 1
                report["checked_steps"] += 1
                split_ok += int(ok)
                if not ok and len(report["failure_examples"]) < 8:
                    report["failure_examples"].append({
                        "split": split,
                        "trace_id": trace["trace_id"],
                        "step": step["index"],
                        "asm": trace["program"]["asm"][step["index"]],
                        "bytes_hex": step["decode"]["bytes_hex"],
                        "unicorn_regs": regs,
                        "trace_regs": p["after_regs"],
                        "unicorn_flags": flags,
                        "trace_flags": p["after_flags"],
                    })
            if report["checked_steps"] >= max_steps:
                break
        report["splits"][split] = {
            "checked_steps": split_checked,
            "exact": round(split_ok / max(1, split_checked), 6),
        }
    report["ok"] = len(report["failure_examples"]) == 0 and report["checked_steps"] > 0
    return report


def split_diagnostics(traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    programs = len(traces)
    steps = 0
    imms = set()
    ops: Dict[str, int] = {}
    families: Dict[str, int] = {}
    lengths: List[int] = []
    for trace in traces:
        lengths.append(int(trace["program"]["length"]))
        for step in trace["steps"]:
            steps += 1
            p = step["pan_stage0"]
            op_name = OPS[int(p["op_index"])].name
            if OPS[int(p["op_index"])].operands == "ri":
                imms.add(int(p["imm"]))
            family = step["decode"]["category"]
            ops[op_name] = ops.get(op_name, 0) + 1
            families[family] = families.get(family, 0) + 1
    return {
        "programs": programs,
        "steps": steps,
        "length_min": min(lengths) if lengths else 0,
        "length_max": max(lengths) if lengths else 0,
        "unique_immediates": len(imms),
        "immediate_values": sorted(imms),
        "op_counts": dict(sorted(ops.items())),
        "family_counts": dict(sorted(families.items())),
    }


def generate_dataset(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor] = None) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("generate")
    rng = random.Random(args.seed)
    make_manifest(out_dir, args.word_bits)

    def rows(count: int, split: str, min_len: int, max_len: int, avoid_bytes: set[str]) -> List[Dict[str, Any]]:
        out = []
        seen = set(avoid_bytes)
        attempts = 0
        max_attempts = max(10000, count * 200)
        while len(out) < count and attempts < max_attempts:
            attempts += 1
            length = rng.randint(min_len, max_len)
            trace = make_trace(f"{split}-{args.seed}-{len(out)}", split, length, rng, args.word_bits)
            bytes_hex = trace["program"]["bytes_hex"]
            if bytes_hex in seen:
                continue
            seen.add(bytes_hex)
            out.append(trace)
        if len(out) != count:
            raise RuntimeError(f"could only generate {len(out)} unique {split} traces after {attempts} attempts")
        avoid_bytes.update(t["program"]["bytes_hex"] for t in out)
        return out

    train_path = out_dir / "train_traces.jsonl"
    val_path = out_dir / "val_traces.jsonl"
    long_path = out_dir / "long_traces.jsonl"
    used_bytes: set[str] = set()
    train_traces = rows(args.train_programs, args.train_split, 1, args.max_train_len, used_bytes)
    val_traces = rows(args.val_programs, args.val_split, 1, args.max_train_len, used_bytes)
    long_traces = rows(args.long_programs, args.long_split, args.long_len, args.long_len, used_bytes)
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


def step_rows(traces: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    rows = []
    for trace in traces:
        for step in trace["steps"]:
            p = step["pan_stage0"]
            rows.append((
                p["before_regs"],
                p["before_flags"],
                p["op_index"],
                p["dst"],
                p["src"],
                p["imm"],
                p["after_regs"],
                p["after_flags"],
                step["decode"]["category"],
            ))
    if not rows:
        raise ValueError("no step rows")
    before_regs = np.array([r[0] for r in rows], dtype=np.int64)
    before_flags = np.array([r[1] for r in rows], dtype=np.int64)
    op = np.array([r[2] for r in rows], dtype=np.int64)
    dst = np.array([r[3] for r in rows], dtype=np.int64)
    src = np.array([r[4] for r in rows], dtype=np.int64)
    imm = np.array([r[5] for r in rows], dtype=np.int64)
    after_regs = np.array([r[6] for r in rows], dtype=np.int64)
    after_flags = np.array([r[7] for r in rows], dtype=np.int64)
    family = np.array([r[8] for r in rows], dtype=object)
    return {
        "before_regs": before_regs,
        "before_flags": before_flags,
        "op": op,
        "dst": dst,
        "src": src,
        "imm": imm,
        "after_regs": after_regs,
        "after_flags": after_flags,
        "family": family,
    }


class Stage0PAN(nn.Module):
    def __init__(self, word_bits: int, hidden: int = 192, freqs: int = 8):
        super().__init__()
        self.word_bits = word_bits
        self.m = 1 << word_bits
        self.freqs = min(freqs, max(1, self.m // 2))
        self.op_emb = nn.Embedding(len(OPS), 24)
        self.reg_emb = nn.Embedding(len(REGS), 8)
        table = []
        for v in range(self.m):
            feats = []
            for k in range(1, self.freqs + 1):
                ang = 2 * math.pi * k * v / self.m
                feats.extend([math.cos(ang), math.sin(ang)])
            table.append(feats)
        self.register_buffer("phase_table", torch.tensor(table, dtype=torch.float32))
        phase_dim = self.freqs * 2
        in_dim = (
            len(REGS) * phase_dim
            + len(FLAGS)
            + phase_dim * 5
            + 24
            + 8 * 2
        )
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(REGS) * self.m + len(FLAGS) * 2),
        )

    def phase(self, values: torch.Tensor) -> torch.Tensor:
        return self.phase_table[values.long().clamp(0, self.m - 1)]

    def forward(
        self,
        regs: torch.Tensor,
        flags: torch.Tensor,
        op: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        imm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        regs = regs.long()
        flags = flags.float()
        dst = dst.long()
        src = src.long()
        imm = imm.long() % self.m
        batch = regs.shape[0]
        all_reg_phase = self.phase(regs.reshape(-1)).reshape(batch, len(REGS), -1).reshape(batch, -1)
        dst_val = regs[torch.arange(batch, device=regs.device), dst]
        src_val = regs[torch.arange(batch, device=regs.device), src]
        dst_phase = self.phase(dst_val)
        src_phase = self.phase(src_val)
        imm_phase = self.phase(imm)
        add_mix = self.phase((dst_val + src_val) % self.m)
        sub_mix = self.phase((dst_val - src_val) % self.m)
        x = torch.cat([
            all_reg_phase,
            flags,
            dst_phase,
            src_phase,
            imm_phase,
            add_mix,
            sub_mix,
            self.op_emb(op.long()),
            self.reg_emb(dst),
            self.reg_emb(src),
        ], dim=1)
        y = self.net(x)
        reg_logits = y[:, : len(REGS) * self.m].reshape(batch, len(REGS), self.m)
        flag_logits = y[:, len(REGS) * self.m :].reshape(batch, len(FLAGS), 2)
        return reg_logits, flag_logits


class FactorizedStage0PAN(nn.Module):
    """Shared Stage 0 transition head.

    The dense model predicts every register slot independently. This head
    predicts only the touched low-byte value plus flags from a shared operand
    basis, then writes the predicted value back through the selected destination
    register. It is still trained by supervision, but it gives the learner the
    same factored structure the structured cartridge uses.
    """

    def __init__(self, word_bits: int, hidden: int = 192, freqs: int = 8):
        super().__init__()
        self.word_bits = word_bits
        self.m = 1 << word_bits
        self.mask = self.m - 1
        self.sign = 1 << (word_bits - 1)
        self.freqs = min(freqs, max(1, self.m // 2))
        self.op_emb = nn.Embedding(len(OPS), 32)
        self.reg_emb = nn.Embedding(len(REGS), 8)
        table = []
        for v in range(self.m):
            feats = []
            for k in range(1, self.freqs + 1):
                ang = 2 * math.pi * k * v / self.m
                feats.extend([math.cos(ang), math.sin(ang)])
            table.append(feats)
        self.register_buffer("phase_table", torch.tensor(table, dtype=torch.float32))
        candidate_count = 17
        flag_candidate_count = 15
        self.value_net = nn.Sequential(
            nn.Linear(32, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, candidate_count),
        )
        self.flag_net = nn.Sequential(
            nn.Linear(32, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, flag_candidate_count),
        )

    def phase(self, values: torch.Tensor) -> torch.Tensor:
        return self.phase_table[values.long().clamp(0, self.m - 1)]

    def selector_logits(self, op: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        op_features = self.op_emb(op.long())
        return self.value_net(op_features), self.flag_net(op_features)

    def _add_flags(self, a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, carry_in: torch.Tensor) -> torch.Tensor:
        full = a + b + carry_in
        cf = full.ge(self.m)
        zf = result.eq(0)
        sf = result.bitwise_and(self.sign).ne(0)
        of = (torch.bitwise_and(torch.bitwise_not(torch.bitwise_xor(a, b)), torch.bitwise_xor(a, result))).bitwise_and(self.sign).ne(0)
        return torch.stack([cf, zf, sf, of], dim=1).float()

    def _sub_flags(self, a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, borrow_in: torch.Tensor) -> torch.Tensor:
        rhs = b + borrow_in
        cf = a.lt(rhs)
        zf = result.eq(0)
        sf = result.bitwise_and(self.sign).ne(0)
        of = torch.bitwise_and(torch.bitwise_and(torch.bitwise_xor(a, rhs), torch.bitwise_xor(a, result)), self.sign).ne(0)
        return torch.stack([cf, zf, sf, of], dim=1).float()

    def _logical_flags(self, result: torch.Tensor) -> torch.Tensor:
        cf = torch.zeros_like(result, dtype=torch.bool)
        zf = result.eq(0)
        sf = result.bitwise_and(self.sign).ne(0)
        of = torch.zeros_like(result, dtype=torch.bool)
        return torch.stack([cf, zf, sf, of], dim=1).float()

    def forward(
        self,
        regs: torch.Tensor,
        flags: torch.Tensor,
        op: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        imm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        regs = regs.long()
        flags_long = flags.long()
        flags_f = flags.float()
        op = op.long()
        dst = dst.long()
        src = src.long()
        imm = imm.long() % self.m
        batch = regs.shape[0]
        device = regs.device
        row = torch.arange(batch, device=device)
        dst_val = regs[row, dst]
        src_val = regs[row, src]
        uses_imm = OP_USES_IMM.to(device)[op]
        arg_val = torch.where(uses_imm, imm, src_val)
        cf = flags_long[:, 0]

        add_res = (dst_val + arg_val) & self.mask
        addc_res = (dst_val + arg_val + cf) & self.mask
        sub_res = (dst_val - arg_val) & self.mask
        subb_res = (dst_val - arg_val - cf) & self.mask
        inc_res = (dst_val + 1) & self.mask
        dec_res = (dst_val - 1) & self.mask
        neg_res = (-dst_val) & self.mask
        and_res = dst_val & arg_val
        or_res = dst_val | arg_val
        xor_res = dst_val ^ arg_val
        not_res = torch.bitwise_not(dst_val) & self.mask
        shl_res = (dst_val << 1) & self.mask
        shr_res = (dst_val >> 1) & self.mask
        rol_cf = dst_val.bitwise_and(self.sign).ne(0).long()
        rol_res = ((dst_val << 1) | rol_cf) & self.mask
        ror_cf = dst_val & 1
        ror_res = ((dst_val >> 1) | (ror_cf << (self.word_bits - 1))) & self.mask

        candidates = torch.stack([
            dst_val,
            arg_val,
            add_res,
            addc_res,
            sub_res,
            subb_res,
            inc_res,
            dec_res,
            neg_res,
            and_res,
            or_res,
            xor_res,
            not_res,
            shl_res,
            shr_res,
            rol_res,
            ror_res,
        ], dim=1)
        old_cf = flags_f[:, 0:1]
        add_f = self._add_flags(dst_val, arg_val, add_res, torch.zeros_like(cf))
        addc_f = self._add_flags(dst_val, arg_val, addc_res, cf)
        sub_f = self._sub_flags(dst_val, arg_val, sub_res, torch.zeros_like(cf))
        subb_f = self._sub_flags(dst_val, arg_val, subb_res, cf)
        inc_f = torch.cat([old_cf, self._add_flags(dst_val, torch.ones_like(dst_val), inc_res, torch.zeros_like(cf))[:, 1:]], dim=1)
        dec_f = torch.cat([old_cf, self._sub_flags(dst_val, torch.ones_like(dst_val), dec_res, torch.zeros_like(cf))[:, 1:]], dim=1)
        neg_f = self._sub_flags(torch.zeros_like(dst_val), dst_val, neg_res, torch.zeros_like(cf))
        neg_f = torch.cat([dst_val.ne(0).float().reshape(batch, 1), neg_f[:, 1:]], dim=1)
        logical_and_f = self._logical_flags(and_res)
        logical_or_f = self._logical_flags(or_res)
        logical_xor_f = self._logical_flags(xor_res)
        shl_cf = dst_val.bitwise_and(self.sign).ne(0).float().reshape(batch, 1)
        shl_f = torch.cat([
            shl_cf,
            shl_res.eq(0).float().reshape(batch, 1),
            shl_res.bitwise_and(self.sign).ne(0).float().reshape(batch, 1),
            (shl_res.bitwise_and(self.sign).ne(0) ^ shl_cf.bool().reshape(batch)).float().reshape(batch, 1),
        ], dim=1)
        shr_cf = dst_val.bitwise_and(1).float().reshape(batch, 1)
        shr_f = torch.cat([
            shr_cf,
            shr_res.eq(0).float().reshape(batch, 1),
            shr_res.bitwise_and(self.sign).ne(0).float().reshape(batch, 1),
            dst_val.bitwise_and(self.sign).ne(0).float().reshape(batch, 1),
        ], dim=1)
        rol_msb = rol_res.bitwise_and(self.sign).ne(0)
        rol_f = torch.cat([
            rol_cf.float().reshape(batch, 1),
            flags_f[:, 1:2],
            flags_f[:, 2:3],
            (rol_msb ^ rol_cf.bool()).float().reshape(batch, 1),
        ], dim=1)
        ror_msb = ror_res.bitwise_and(self.sign).ne(0)
        ror_next_msb = ror_res.bitwise_and(self.sign >> 1).ne(0) if self.word_bits > 1 else torch.zeros_like(ror_msb)
        ror_f = torch.cat([
            ror_cf.float().reshape(batch, 1),
            flags_f[:, 1:2],
            flags_f[:, 2:3],
            (ror_msb ^ ror_next_msb).float().reshape(batch, 1),
        ], dim=1)
        flag_candidates = torch.cat([
            flags_f,
            add_f,
            addc_f,
            sub_f,
            subb_f,
            inc_f,
            dec_f,
            neg_f,
            logical_and_f,
            logical_or_f,
            logical_xor_f,
            shl_f,
            shr_f,
            rol_f,
            ror_f,
        ], dim=1)

        op_features = self.op_emb(op)
        candidate_scores = self.value_net(op_features)

        dst_logits = torch.full((batch, self.m), -20.0, device=device, dtype=candidate_scores.dtype)
        for i in range(candidates.shape[1]):
            target = candidates[:, i].long().clamp(0, self.m - 1)
            old = dst_logits[row, target]
            dst_logits[row, target] = torch.maximum(old, candidate_scores[:, i])

        flag_candidate_values = flag_candidates.reshape(batch, -1, len(FLAGS)).long().clamp(0, 1)
        flag_scores = self.flag_net(op_features)
        flag_logits = torch.full((batch, len(FLAGS), 2), -20.0, device=device, dtype=flag_scores.dtype)
        for i in range(flag_candidate_values.shape[1]):
            for fidx in range(len(FLAGS)):
                target = flag_candidate_values[:, i, fidx]
                old = flag_logits[row, fidx, target]
                flag_logits[row, fidx, target] = torch.maximum(old, flag_scores[:, i])
        reg_logits = torch.zeros(batch, len(REGS), self.m, device=device, dtype=dst_logits.dtype)
        reg_logits[row, dst, :] = dst_logits
        return reg_logits, flag_logits


class EffectStage0PAN(nn.Module):
    """Learned typed-effect Stage 0 transition head.

    Unlike the dense and factorized baselines, this model does not get the
    architectural write masks at commit time. It learns register/flag write
    masks and the corresponding values, then the predictor commits exactly the
    typed effects the model emitted.
    """

    def __init__(self, word_bits: int, hidden: int = 192, freqs: int = 8):
        super().__init__()
        self.word_bits = word_bits
        self.m = 1 << word_bits
        self.freqs = min(freqs, max(1, self.m // 2))
        self.op_emb = nn.Embedding(len(OPS), 24)
        self.reg_emb = nn.Embedding(len(REGS), 8)
        table = []
        for v in range(self.m):
            feats = []
            for k in range(1, self.freqs + 1):
                ang = 2 * math.pi * k * v / self.m
                feats.extend([math.cos(ang), math.sin(ang)])
            table.append(feats)
        self.register_buffer("phase_table", torch.tensor(table, dtype=torch.float32))
        phase_dim = self.freqs * 2
        in_dim = (
            len(REGS) * phase_dim
            + len(FLAGS)
            + phase_dim * 5
            + 24
            + 8 * 2
        )
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.reg_write_head = nn.Linear(hidden, len(REGS) * 2)
        self.reg_value_head = nn.Linear(hidden, len(REGS) * self.m)
        self.flag_write_head = nn.Linear(hidden, len(FLAGS) * 2)
        self.flag_value_head = nn.Linear(hidden, len(FLAGS) * 2)

    def phase(self, values: torch.Tensor) -> torch.Tensor:
        return self.phase_table[values.long().clamp(0, self.m - 1)]

    def features(
        self,
        regs: torch.Tensor,
        flags: torch.Tensor,
        op: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        imm: torch.Tensor,
    ) -> torch.Tensor:
        regs = regs.long()
        flags = flags.float()
        dst = dst.long()
        src = src.long()
        imm = imm.long() % self.m
        batch = regs.shape[0]
        all_reg_phase = self.phase(regs.reshape(-1)).reshape(batch, len(REGS), -1).reshape(batch, -1)
        row = torch.arange(batch, device=regs.device)
        dst_val = regs[row, dst]
        src_val = regs[row, src]
        dst_phase = self.phase(dst_val)
        src_phase = self.phase(src_val)
        imm_phase = self.phase(imm)
        add_mix = self.phase((dst_val + src_val) % self.m)
        sub_mix = self.phase((dst_val - src_val) % self.m)
        return torch.cat([
            all_reg_phase,
            flags,
            dst_phase,
            src_phase,
            imm_phase,
            add_mix,
            sub_mix,
            self.op_emb(op.long()),
            self.reg_emb(dst),
            self.reg_emb(src),
        ], dim=1)

    def forward(
        self,
        regs: torch.Tensor,
        flags: torch.Tensor,
        op: torch.Tensor,
        dst: torch.Tensor,
        src: torch.Tensor,
        imm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = regs.shape[0]
        h = self.body(self.features(regs, flags, op, dst, src, imm))
        reg_write_logits = self.reg_write_head(h).reshape(batch, len(REGS), 2)
        reg_value_logits = self.reg_value_head(h).reshape(batch, len(REGS), self.m)
        flag_write_logits = self.flag_write_head(h).reshape(batch, len(FLAGS), 2)
        flag_value_logits = self.flag_value_head(h).reshape(batch, len(FLAGS), 2)
        return reg_write_logits, reg_value_logits, flag_write_logits, flag_value_logits


def make_model(arch: str, word_bits: int, hidden: int, freqs: int) -> nn.Module:
    if arch == "dense":
        return Stage0PAN(word_bits, hidden=hidden, freqs=freqs)
    if arch == "factorized":
        return FactorizedStage0PAN(word_bits, hidden=hidden, freqs=freqs)
    if arch == "effect":
        return EffectStage0PAN(word_bits, hidden=hidden, freqs=freqs)
    raise ValueError(f"unknown arch {arch!r}")


class Predictor(Protocol):
    def predict_next(self, regs: List[int], flags: List[int], op_idx: int, dst: int, src: int, imm: int) -> Tuple[List[int], List[int]]:
        ...


class NeuralPredictor:
    def __init__(self, model: nn.Module):
        self.model = model

    @torch.no_grad()
    def predict_next(self, regs: List[int], flags: List[int], op_idx: int, dst: int, src: int, imm: int) -> Tuple[List[int], List[int]]:
        device = next(self.model.parameters()).device
        outputs = self.model(
            torch.tensor([regs], device=device),
            torch.tensor([flags], device=device),
            torch.tensor([op_idx], device=device),
            torch.tensor([dst], device=device),
            torch.tensor([src], device=device),
            torch.tensor([imm], device=device),
        )
        next_regs = list(regs)
        next_flags = list(flags)
        if isinstance(self.model, EffectStage0PAN):
            reg_write_logits, reg_value_logits, flag_write_logits, flag_value_logits = outputs
            reg_write = reg_write_logits.argmax(-1)[0].cpu().tolist()
            reg_values = reg_value_logits.argmax(-1)[0].cpu().tolist()
            flag_write = flag_write_logits.argmax(-1)[0].cpu().tolist()
            flag_values = flag_value_logits.argmax(-1)[0].cpu().tolist()
            for ridx, do_write in enumerate(reg_write):
                if int(do_write):
                    next_regs[ridx] = int(reg_values[ridx])
            for fidx, do_write in enumerate(flag_write):
                if int(do_write):
                    next_flags[fidx] = int(flag_values[fidx])
        else:
            reg_logits, flag_logits = outputs
            raw_regs = reg_logits.argmax(-1)[0].cpu().tolist()
            raw_flags = flag_logits.argmax(-1)[0].cpu().tolist()
            if OPS[op_idx].writes_dst:
                next_regs[dst] = raw_regs[dst]
            if OPS[op_idx].writes_flags:
                next_flags = raw_flags
        return next_regs, next_flags


class StructuredPredictor:
    """Exact Stage 0 structured PAN cartridge.

    The cartridge uses the same phase/ISA decomposition as the neural runner,
    but its transition operators are the explicit x86-lite semantic maps. It is
    a validation baseline: if this fails, the trace/oracle pipeline is wrong.
    If it passes while the neural model fails, the learning/generalization
    problem is isolated to the learner.
    """

    def __init__(self, word_bits: int, op_offset: int = 0, scramble_flags: bool = False):
        self.word_bits = word_bits
        self.op_offset = op_offset
        self.scramble_flags = scramble_flags

    def predict_next(self, regs: List[int], flags: List[int], op_idx: int, dst: int, src: int, imm: int) -> Tuple[List[int], List[int]]:
        actual_op = (op_idx + self.op_offset) % len(OPS)
        next_regs, next_flags = apply_op(regs, flags, actual_op, dst, src, imm, self.word_bits)
        if self.scramble_flags:
            next_flags = [1 - int(x) for x in next_flags]
        return next_regs, next_flags


def batch_tensors(rows: Dict[str, np.ndarray], idx: np.ndarray, device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: torch.tensor(v[idx], device=device)
        for k, v in rows.items()
        if k != "family"
    }


def effect_targets(batch: Dict[str, torch.Tensor], device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    op = batch["op"].long()
    dst = batch["dst"].long()
    rows = torch.arange(op.shape[0], device=device)
    reg_write = torch.zeros((op.shape[0], len(REGS)), dtype=torch.long, device=device)
    writes_dst = OP_WRITES_DST.to(device)[op]
    if writes_dst.any():
        reg_write[rows[writes_dst], dst[writes_dst]] = 1
    flag_write = OP_WRITES_FLAGS.to(device)[op].long().reshape(-1, 1).repeat(1, len(FLAGS))
    return reg_write, flag_write


def effect_commit_batch(
    rows: Dict[str, np.ndarray],
    sub: np.ndarray,
    outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    reg_write_logits, reg_value_logits, flag_write_logits, flag_value_logits = outputs
    pred_reg_write = reg_write_logits.argmax(-1).cpu().numpy()
    pred_reg_values = reg_value_logits.argmax(-1).cpu().numpy()
    pred_flag_write = flag_write_logits.argmax(-1).cpu().numpy()
    pred_flag_values = flag_value_logits.argmax(-1).cpu().numpy()
    pred_regs = rows["before_regs"][sub].copy()
    pred_flags = rows["before_flags"][sub].copy()
    for j in range(len(sub)):
        for ridx in range(len(REGS)):
            if int(pred_reg_write[j, ridx]):
                pred_regs[j, ridx] = int(pred_reg_values[j, ridx])
        for fidx in range(len(FLAGS)):
            if int(pred_flag_write[j, fidx]):
                pred_flags[j, fidx] = int(pred_flag_values[j, fidx])

    target_batch = {
        "op": torch.tensor(rows["op"][sub]),
        "dst": torch.tensor(rows["dst"][sub]),
    }
    target_reg_write, target_flag_write = effect_targets(target_batch, torch.device("cpu"))
    reg_write_mask_ok = (pred_reg_write == target_reg_write.numpy()).all(axis=1)
    flag_write_mask_ok = (pred_flag_write == target_flag_write.numpy()).all(axis=1)
    mask_metrics = {
        "reg_write_mask_exact": float(reg_write_mask_ok.mean()),
        "flag_write_mask_exact": float(flag_write_mask_ok.mean()),
        "reg_write_mask_ok": int(reg_write_mask_ok.sum()),
        "flag_write_mask_ok": int(flag_write_mask_ok.sum()),
        "mask_total": int(len(sub)),
    }
    return pred_regs, pred_flags, mask_metrics


def train_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor] = None) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("train")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_rows = step_rows(load_traces(out_dir / "train_traces.jsonl"))
    val_rows = step_rows(load_traces(out_dir / "val_traces.jsonl"))
    device = torch.device("cpu")
    model = make_model(args.arch, args.word_bits, hidden=args.hidden, freqs=args.freqs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    n = train_rows["op"].shape[0]
    history = []
    for step in range(args.train_steps):
        idx = np.random.randint(0, n, size=args.batch_size)
        b = batch_tensors(train_rows, idx, device)
        row = torch.arange(b["op"].shape[0], device=device)
        writes_dst = OP_WRITES_DST.to(device)[b["op"].long()]
        writes_flags = OP_WRITES_FLAGS.to(device)[b["op"].long()]
        selector_loss = torch.zeros((), device=device)
        mask_loss = torch.zeros((), device=device)
        if isinstance(model, EffectStage0PAN):
            reg_write_logits, reg_value_logits, flag_write_logits, flag_value_logits = model(
                b["before_regs"], b["before_flags"], b["op"], b["dst"], b["src"], b["imm"]
            )
            target_reg_write, target_flag_write = effect_targets(b, device)
            reg_mask_loss = F.cross_entropy(reg_write_logits.reshape(-1, 2), target_reg_write.reshape(-1))
            flag_mask_loss = F.cross_entropy(flag_write_logits.reshape(-1, 2), target_flag_write.reshape(-1))
            if writes_dst.any():
                reg_loss = F.cross_entropy(
                    reg_value_logits[row[writes_dst], b["dst"][writes_dst].long(), :],
                    b["after_regs"][row[writes_dst], b["dst"][writes_dst].long()].long(),
                )
            else:
                reg_loss = torch.zeros((), device=device)
            if writes_flags.any():
                flag_loss = sum(
                    F.cross_entropy(flag_value_logits[writes_flags, i, :], b["after_flags"][writes_flags, i].long())
                    for i in range(len(FLAGS))
                ) / len(FLAGS)
            else:
                flag_loss = torch.zeros((), device=device)
            mask_loss = reg_mask_loss + flag_mask_loss
            loss = reg_loss + 0.5 * flag_loss + args.effect_mask_loss_weight * mask_loss
        else:
            reg_logits, flag_logits = model(b["before_regs"], b["before_flags"], b["op"], b["dst"], b["src"], b["imm"])
            if writes_dst.any():
                reg_loss = F.cross_entropy(
                    reg_logits[row[writes_dst], b["dst"][writes_dst].long(), :],
                    b["after_regs"][row[writes_dst], b["dst"][writes_dst].long()].long(),
                )
            else:
                reg_loss = torch.zeros((), device=device)
            if writes_flags.any():
                flag_loss = sum(
                    F.cross_entropy(flag_logits[writes_flags, i, :], b["after_flags"][writes_flags, i].long())
                    for i in range(len(FLAGS))
                ) / len(FLAGS)
            else:
                flag_loss = torch.zeros((), device=device)
            if isinstance(model, FactorizedStage0PAN) and args.selector_loss_weight > 0:
                value_scores, flag_scores = model.selector_logits(b["op"])
                if writes_dst.any():
                    selector_loss = selector_loss + F.cross_entropy(
                        value_scores[writes_dst],
                        OP_VALUE_CANDIDATE.to(device)[b["op"].long()[writes_dst]],
                    )
                if writes_flags.any():
                    selector_loss = selector_loss + F.cross_entropy(
                        flag_scores[writes_flags],
                        OP_FLAG_CANDIDATE.to(device)[b["op"].long()[writes_flags]],
                    )
            loss = reg_loss + 0.5 * flag_loss + args.selector_loss_weight * selector_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, args.eval_every) == 0 or step == args.train_steps - 1:
            metrics = teacher_forced_metrics(model, train_rows, args.word_bits, limit=2000)
            val_metrics = teacher_forced_metrics(model, val_rows, args.word_bits, limit=2000)
            rec = {
                "step": step,
                "loss": round(float(loss.detach().cpu()), 6),
                "selector_loss": round(float(selector_loss.detach().cpu()), 6),
                "effect_mask_loss": round(float(mask_loss.detach().cpu()), 6),
                "train_step_exact": metrics["step_exact"],
                "val_step_exact": val_metrics["step_exact"],
                "train_reg_exact": metrics["reg_exact"],
                "val_reg_exact": val_metrics["reg_exact"],
                "train_flag_exact": metrics["flag_exact"],
                "val_flag_exact": val_metrics["flag_exact"],
                "train_reg_write_mask_exact": metrics.get("reg_write_mask_exact"),
                "val_reg_write_mask_exact": val_metrics.get("reg_write_mask_exact"),
                "train_flag_write_mask_exact": metrics.get("flag_write_mask_exact"),
                "val_flag_write_mask_exact": val_metrics.get("flag_write_mask_exact"),
            }
            history.append(rec)
            print(json.dumps({"train_progress": rec}, sort_keys=True), flush=True)
    ckpt = {
        "model_state": model.state_dict(),
        "arch": args.arch,
        "word_bits": args.word_bits,
        "hidden": args.hidden,
        "freqs": args.freqs,
        "ops": [dataclasses.asdict(op) for op in OPS],
        "regs": list(REGS),
        "flags": list(FLAGS),
    }
    torch.save(ckpt, out_dir / "stage0_pan.pt")
    write_json(out_dir / "train_history.json", history)
    summary = {
        "checkpoint": str(out_dir / "stage0_pan.pt"),
        "model_kind": "neural",
        "arch": args.arch,
        "selector_loss_weight": args.selector_loss_weight,
        "effect_mask_loss_weight": args.effect_mask_loss_weight,
        "history": history,
        "final": history[-1] if history else None,
    }
    write_json(out_dir / "train_summary.json", summary)
    return summary


@torch.no_grad()
def teacher_forced_metrics(model: nn.Module, rows: Dict[str, np.ndarray], word_bits: int, limit: Optional[int] = None) -> Dict[str, Any]:
    model.eval()
    n = rows["op"].shape[0]
    if limit is not None and n > limit:
        idx = np.linspace(0, n - 1, num=limit, dtype=np.int64)
    else:
        idx = np.arange(n)
    device = next(model.parameters()).device
    reg_ok_total = 0
    flag_ok_total = 0
    step_ok_total = 0
    reg_write_mask_ok_total = 0
    flag_write_mask_ok_total = 0
    mask_metric_total = 0
    total = 0
    family_counts: Dict[str, List[int]] = {}
    batch = 1024
    for start in range(0, len(idx), batch):
        sub = idx[start : start + batch]
        b = batch_tensors(rows, sub, device)
        outputs = model(b["before_regs"], b["before_flags"], b["op"], b["dst"], b["src"], b["imm"])
        if isinstance(model, EffectStage0PAN):
            pred_regs, pred_flags, mask_metrics = effect_commit_batch(rows, sub, outputs)
            reg_write_mask_ok_total += int(mask_metrics["reg_write_mask_ok"])
            flag_write_mask_ok_total += int(mask_metrics["flag_write_mask_ok"])
            mask_metric_total += int(mask_metrics["mask_total"])
        else:
            reg_logits, flag_logits = outputs
            pred_regs_raw = reg_logits.argmax(-1).cpu().numpy()
            pred_flags_raw = flag_logits.argmax(-1).cpu().numpy()
            pred_regs = rows["before_regs"][sub].copy()
            pred_flags = rows["before_flags"][sub].copy()
            for j, row_idx in enumerate(sub):
                op_idx = int(rows["op"][row_idx])
                dst = int(rows["dst"][row_idx])
                if OPS[op_idx].writes_dst:
                    pred_regs[j, dst] = pred_regs_raw[j, dst]
                if OPS[op_idx].writes_flags:
                    pred_flags[j, :] = pred_flags_raw[j, :]
        target_regs = rows["after_regs"][sub]
        target_flags = rows["after_flags"][sub]
        reg_row_ok = (pred_regs == target_regs).all(axis=1)
        flag_row_ok = (pred_flags == target_flags).all(axis=1)
        step_ok = reg_row_ok & flag_row_ok
        reg_ok_total += int(reg_row_ok.sum())
        flag_ok_total += int(flag_row_ok.sum())
        step_ok_total += int(step_ok.sum())
        total += len(sub)
        for fam, ok in zip(rows["family"][sub], step_ok):
            bucket = family_counts.setdefault(str(fam), [0, 0])
            bucket[0] += int(bool(ok))
            bucket[1] += 1
    model.train()
    result = {
        "reg_exact": round(reg_ok_total / max(1, total), 6),
        "flag_exact": round(flag_ok_total / max(1, total), 6),
        "step_exact": round(step_ok_total / max(1, total), 6),
        "family_step_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(family_counts.items())},
        "n_steps": total,
    }
    if isinstance(model, EffectStage0PAN):
        result.update({
            "reg_write_mask_exact": round(reg_write_mask_ok_total / max(1, mask_metric_total), 6),
            "flag_write_mask_exact": round(flag_write_mask_ok_total / max(1, mask_metric_total), 6),
        })
    return result


def load_model(out_dir: Path) -> nn.Module:
    ckpt = torch.load(out_dir / "stage0_pan.pt", map_location="cpu")
    model = make_model(str(ckpt.get("arch", "dense")), int(ckpt["word_bits"]), int(ckpt["hidden"]), int(ckpt["freqs"]))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def free_run_metrics(predictor: Predictor, traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_steps = 0
    step_exact = 0
    final_exact = 0
    family_counts: Dict[str, List[int]] = {}
    examples = []
    for trace in traces:
        first = trace["steps"][0]["pan_stage0"]
        regs = list(first["before_regs"])
        flags = list(first["before_flags"])
        trace_ok = True
        for step in trace["steps"]:
            p = step["pan_stage0"]
            regs, flags = predictor.predict_next(regs, flags, p["op_index"], p["dst"], p["src"], p["imm"])
            ok = regs == p["after_regs"] and flags == p["after_flags"]
            trace_ok = trace_ok and ok
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
                    "pred_regs": regs,
                    "true_regs": p["after_regs"],
                    "pred_flags": flags,
                    "true_flags": p["after_flags"],
                })
        final_exact += int(trace_ok)
    return {
        "programs": len(traces),
        "steps": total_steps,
        "step_exact": round(step_exact / max(1, total_steps), 6),
        "final_exact": round(final_exact / max(1, len(traces)), 6),
        "family_step_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(family_counts.items())},
        "failure_examples": examples,
    }


def operator_structure(predictor: Predictor, word_bits: int, probe_bits: Optional[int] = None) -> Dict[str, Any]:
    """Probe one-register effects for each op as a learned transition map."""
    effective_bits = min(word_bits, probe_bits if probe_bits is not None else word_bits)
    mask = (1 << effective_bits) - 1
    rows = {}
    for op_idx, op in enumerate(OPS):
        table = np.zeros((mask + 1, mask + 1), dtype=np.float64)
        for a in range(mask + 1):
            for b in range(mask + 1):
                regs = [0, 0, 0, 0]
                flags = [0, 0, 0, 0]
                regs[0] = a
                regs[1] = b
                pred_regs, _ = predictor.predict_next(regs, flags, op_idx, 0, 1, b)
                table[a, int(pred_regs[0]) & mask] += 1.0
        row_sums = table.sum(axis=1, keepdims=True)
        probs = table / np.maximum(row_sums, 1.0)
        perm_peak = float((probs.max(axis=1) ** 2).sum() / max(1e-12, (probs ** 2).sum()))
        rank = int(np.linalg.matrix_rank(probs, tol=1e-6))
        fft = np.fft.fft(probs, axis=1)
        diag_energy = float(np.sum(np.abs(np.diag(fft)) ** 2) / max(1e-12, np.sum(np.abs(fft) ** 2)))
        rows[op.name] = {
            "family": op.family,
            "probe_bits": effective_bits,
            "perm_peak": round(perm_peak, 6),
            "rank": rank,
            "dft_diag_energy": round(diag_energy, 6),
        }
    return rows


def structured_teacher_forced_metrics(predictor: Predictor, rows: Dict[str, np.ndarray], word_bits: int, limit: Optional[int] = None) -> Dict[str, Any]:
    n = rows["op"].shape[0]
    if limit is not None and n > limit:
        idx = np.linspace(0, n - 1, num=limit, dtype=np.int64)
    else:
        idx = np.arange(n)
    reg_ok_total = 0
    flag_ok_total = 0
    step_ok_total = 0
    family_counts: Dict[str, List[int]] = {}
    for row_idx in idx:
        regs = rows["before_regs"][row_idx].tolist()
        flags = rows["before_flags"][row_idx].tolist()
        pred_regs, pred_flags = predictor.predict_next(
            regs,
            flags,
            int(rows["op"][row_idx]),
            int(rows["dst"][row_idx]),
            int(rows["src"][row_idx]),
            int(rows["imm"][row_idx]),
        )
        target_regs = rows["after_regs"][row_idx].tolist()
        target_flags = rows["after_flags"][row_idx].tolist()
        reg_ok = pred_regs == target_regs
        flag_ok = pred_flags == target_flags
        step_ok = reg_ok and flag_ok
        reg_ok_total += int(reg_ok)
        flag_ok_total += int(flag_ok)
        step_ok_total += int(step_ok)
        bucket = family_counts.setdefault(str(rows["family"][row_idx]), [0, 0])
        bucket[0] += int(step_ok)
        bucket[1] += 1
    total = len(idx)
    return {
        "reg_exact": round(reg_ok_total / max(1, total), 6),
        "flag_exact": round(flag_ok_total / max(1, total), 6),
        "step_exact": round(step_ok_total / max(1, total), 6),
        "family_step_exact": {k: round(v[0] / max(1, v[1]), 6) for k, v in sorted(family_counts.items())},
        "n_steps": total,
    }


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


def control_results(word_bits: int, val_traces: Sequence[Dict[str, Any]], long_traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    scrambled_op = StructuredPredictor(word_bits, op_offset=1)
    scrambled_flags = StructuredPredictor(word_bits, scramble_flags=True)
    return {
        "scrambled_opcode_val": free_run_metrics(scrambled_op, val_traces),
        "scrambled_flags_val": free_run_metrics(scrambled_flags, val_traces),
        "scrambled_opcode_long": free_run_metrics(scrambled_op, long_traces),
    }


def validate_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor] = None) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("validate")
    model = load_model(out_dir)
    predictor = NeuralPredictor(model)
    train_traces = load_traces(out_dir / "train_traces.jsonl")
    train_rows = step_rows(train_traces)
    val_traces = load_traces(out_dir / "val_traces.jsonl")
    long_traces = load_traces(out_dir / "long_traces.jsonl")
    val_rows = step_rows(val_traces)
    long_rows = step_rows(long_traces)
    learned_effect_mode = isinstance(model, EffectStage0PAN)
    report = {
        "stage": "stage0_x86_lite_alu",
        "model_kind": "neural",
        "arch": getattr(args, "arch", "dense"),
        "transition_contract": {
            "schema": "pan-x86-stage0-transition-effect-v1" if learned_effect_mode else "legacy-stage0-value-logits",
            "learned_write_masks": learned_effect_mode,
            "learned_flag_definedness": False,
            "flag_definedness_policy": "stage0_flags_are_treated_as_defined_values_only",
            "runtime_proposal_schema": "pan-isa-x86-runtime-spf TransitionProposal v1 compatible metadata pending shared adapter",
            "commit_path": "model_emitted_typed_effects" if learned_effect_mode else "known_op_metadata_masks",
            "deterministic_semantic_candidates_used_for_commit": False,
        },
        "word_bits": args.word_bits,
        "thresholds": {
            "configured_stage0_exact_final_state": 0.995,
            "configured_stage0_exact_step_state": 0.990,
            "configured_stage0_effect_mask_exact": 0.990,
            "note": "Smoke runs are allowed to fall short; failures are evidence for next training pass.",
        },
        "teacher_forced": {
            "train": teacher_forced_metrics(model, train_rows, args.word_bits),
            "val": teacher_forced_metrics(model, val_rows, args.word_bits),
            "long": teacher_forced_metrics(model, long_rows, args.word_bits),
        },
        "free_run": {
            "val": free_run_metrics(predictor, val_traces),
            "long": free_run_metrics(predictor, long_traces),
        },
        "operator_structure": operator_structure(predictor, args.word_bits, args.operator_probe_bits),
        "byte_overlap": byte_overlap(train_traces, val_traces, long_traces),
        "controls": control_results(args.word_bits, val_traces, long_traces),
        "external_oracle": validate_external_oracle(args, out_dir),
    }
    effect_mask_gate = True
    if learned_effect_mode:
        effect_mask_gate = (
            report["teacher_forced"]["val"].get("reg_write_mask_exact", 0.0) >= 0.990
            and report["teacher_forced"]["val"].get("flag_write_mask_exact", 0.0) >= 0.990
            and report["teacher_forced"]["long"].get("reg_write_mask_exact", 0.0) >= 0.990
            and report["teacher_forced"]["long"].get("flag_write_mask_exact", 0.0) >= 0.990
        )
    report["gate_status"] = {
        "schema_valid": json.loads((out_dir / "schema_validation.json").read_text(encoding="utf-8")).get("ok", False),
        "stage0_gate_passed": (
            report["byte_overlap"]["train_val_overlap"] == 0
            and report["byte_overlap"]["train_long_overlap"] == 0
            and report["free_run"]["val"]["final_exact"] >= 0.995
            and report["free_run"]["val"]["step_exact"] >= 0.990
            and report["free_run"]["long"]["final_exact"] >= 0.995
            and report["free_run"]["long"]["step_exact"] >= 0.990
            and report["teacher_forced"]["val"]["step_exact"] >= 0.990
            and effect_mask_gate
            and report["controls"]["scrambled_opcode_val"]["step_exact"] < 0.900
            and report["controls"]["scrambled_flags_val"]["step_exact"] < 0.900
            and (not report["external_oracle"]["enabled"] or report["external_oracle"]["ok"])
        ),
        "learned_effect_mask_gate_passed": effect_mask_gate,
        "full_x86_goal_complete": False,
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def write_structured_cartridge(out_dir: Path, word_bits: int, emit_tables: bool = False) -> None:
    data: Dict[str, Any] = {
        "model_kind": "structured",
        "word_bits": word_bits,
        "registers": list(REGS),
        "flags": list(FLAGS),
        "operators": [dataclasses.asdict(op) for op in OPS],
        "transition": {
            "representation": "semantic_spec",
            "function": "apply_op(regs, flags, op_idx, dst, src, imm, word_bits)",
            "mask": "(1 << word_bits) - 1",
            "carry_flags": ["CF"],
            "derived_flags": ["CF", "ZF", "SF", "OF"],
        },
        "table_encoding": {
            "emitted": emit_tables,
            "entries_if_emitted": len(OPS) * (1 << word_bits) * (1 << word_bits),
            "note": "Explicit tables are omitted by default because word_bits=8 already exceeds 650k JSON rows.",
        },
        "note": "Exact structured Stage 0 cartridge generated from x86-lite semantic maps.",
    }
    if emit_tables:
        predictor = StructuredPredictor(word_bits)
        tables = {}
        for op_idx, op in enumerate(OPS):
            entries = []
            for dst_val in range(1 << word_bits):
                for src_val in range(1 << word_bits):
                    regs = [0, 0, 0, 0]
                    flags = [0, 0, 0, 0]
                    regs[0] = dst_val
                    regs[1] = src_val
                    pred_regs, pred_flags = predictor.predict_next(regs, flags, op_idx, 0, 1, src_val)
                    entries.append({
                        "dst": dst_val,
                        "src_or_imm": src_val,
                        "out": pred_regs[0],
                        "flags": pred_flags,
                    })
            tables[op.name] = {
                "family": op.family,
                "iform": op.iform,
                "entries": entries,
            }
        data["tables"] = tables
    write_json(out_dir / "stage0_structured_cartridge.json", data)


def train_structured_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor] = None) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("train")
    emit_tables = bool(getattr(args, "emit_structured_tables", False))
    write_structured_cartridge(out_dir, args.word_bits, emit_tables=emit_tables)
    summary = {
        "model_kind": "structured",
        "checkpoint": str(out_dir / "stage0_structured_cartridge.json"),
        "emit_structured_tables": emit_tables,
        "history": [],
        "final": {
            "note": "No learned weights: exact Stage 0 x86-lite semantic cartridge emitted for validation.",
        },
    }
    write_json(out_dir / "train_summary.json", summary)
    return summary


def validate_structured(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor] = None) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("validate")
    predictor = StructuredPredictor(args.word_bits)
    train_traces = load_traces(out_dir / "train_traces.jsonl")
    val_traces = load_traces(out_dir / "val_traces.jsonl")
    long_traces = load_traces(out_dir / "long_traces.jsonl")
    train_rows = step_rows(train_traces)
    val_rows = step_rows(val_traces)
    long_rows = step_rows(long_traces)
    report = {
        "stage": "stage0_x86_lite_alu",
        "model_kind": "structured",
        "word_bits": args.word_bits,
        "thresholds": {
            "configured_stage0_exact_final_state": 0.995,
            "configured_stage0_exact_step_state": 0.990,
        },
        "teacher_forced": {
            "train": structured_teacher_forced_metrics(predictor, train_rows, args.word_bits),
            "val": structured_teacher_forced_metrics(predictor, val_rows, args.word_bits),
            "long": structured_teacher_forced_metrics(predictor, long_rows, args.word_bits),
        },
        "free_run": {
            "val": free_run_metrics(predictor, val_traces),
            "long": free_run_metrics(predictor, long_traces),
        },
        "operator_structure": operator_structure(predictor, args.word_bits, args.operator_probe_bits),
        "byte_overlap": byte_overlap(train_traces, val_traces, long_traces),
        "controls": control_results(args.word_bits, val_traces, long_traces),
        "external_oracle": validate_external_oracle(args, out_dir),
    }
    report["gate_status"] = {
        "schema_valid": json.loads((out_dir / "schema_validation.json").read_text(encoding="utf-8")).get("ok", False),
        "stage0_gate_passed": (
            report["byte_overlap"]["train_val_overlap"] == 0
            and report["byte_overlap"]["train_long_overlap"] == 0
            and report["free_run"]["val"]["final_exact"] >= 0.995
            and report["free_run"]["val"]["step_exact"] >= 0.990
            and report["free_run"]["long"]["final_exact"] >= 0.995
            and report["free_run"]["long"]["step_exact"] >= 0.990
            and report["teacher_forced"]["val"]["step_exact"] >= 0.990
            and report["controls"]["scrambled_opcode_val"]["step_exact"] < 0.900
            and report["controls"]["scrambled_flags_val"]["step_exact"] < 0.900
            and (not report["external_oracle"]["enabled"] or report["external_oracle"]["ok"])
        ),
        "full_x86_goal_complete": False,
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def source_freeze(out_dir: Path, model_kind: str = "neural", arch: str = "dense", external_oracle: str = "none") -> Dict[str, Any]:
    data = {
        "stage": "stage0_x86_lite_alu",
        "model_kind": model_kind,
        "arch": arch,
        "created": DEFAULT_DATE,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "psutil": psutil.__version__,
        "external_xed": "not integrated in this Stage 0 implementation",
        "external_oracles": ["stage0_python_semantics"] + ([] if external_oracle == "none" else [external_oracle]),
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
        source_freeze(out_dir, args.model_kind, args.arch, args.external_oracle)
        generation = generate_dataset(args, out_dir, monitor)
        if args.model_kind == "structured":
            training = train_structured_model(args, out_dir, monitor)
            validation = validate_structured(args, out_dir, monitor)
        else:
            training = train_model(args, out_dir, monitor)
            validation = validate_model(args, out_dir, monitor)
        monitor.set_phase("finalize")
        result = {
            "out_dir": str(out_dir),
            "model_kind": args.model_kind,
            "arch": args.arch,
            "generation": generation,
            "training": training,
            "validation": {
                "gate_status": validation["gate_status"],
                "val_free_run": validation["free_run"]["val"],
                "long_free_run": validation["free_run"]["long"],
                "val_teacher_forced": validation["teacher_forced"]["val"],
                "controls": {
                    "scrambled_opcode_val_step_exact": validation["controls"]["scrambled_opcode_val"]["step_exact"],
                    "scrambled_flags_val_step_exact": validation["controls"]["scrambled_flags_val"]["step_exact"],
                },
            },
        }
        write_json(out_dir / "run_summary.json", result)
        return result
    finally:
        monitor.stop()


def default_out_dir(run_name: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / "stage0_x86_lite_alu" / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--out-dir", type=Path, default=None)
        sp.add_argument("--run-name", default=f"run-{int(time.time())}")
        sp.add_argument("--word-bits", type=int, default=4, choices=[3, 4, 5, 6, 7, 8])
        sp.add_argument("--model-kind", choices=["neural", "structured"], default="neural")
        sp.add_argument("--arch", choices=["dense", "factorized", "effect"], default="dense")
        sp.add_argument("--emit-structured-tables", action="store_true")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--monitor-interval", type=float, default=1.0)
        sp.add_argument("--train-split", choices=SPLITS, default="train")
        sp.add_argument("--val-split", choices=SPLITS, default="heldout_imm")
        sp.add_argument("--long-split", choices=SPLITS, default="heldout_reg_tuple")
        sp.add_argument("--external-oracle", choices=["none", "unicorn"], default="none")
        sp.add_argument("--oracle-pythonpath", type=Path, default=DEFAULT_ORACLE_DEP_PATH)
        sp.add_argument("--oracle-max-steps", type=int, default=10000)
        sp.add_argument("--operator-probe-bits", type=int, default=None)

    run = sub.add_parser("run-smoke", help="generate, train, validate, and monitor a Stage 0 run")
    add_common(run)
    run.add_argument("--train-programs", type=int, default=3000)
    run.add_argument("--val-programs", type=int, default=500)
    run.add_argument("--long-programs", type=int, default=200)
    run.add_argument("--max-train-len", type=int, default=4)
    run.add_argument("--long-len", type=int, default=16)
    run.add_argument("--train-steps", type=int, default=600)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--hidden", type=int, default=192)
    run.add_argument("--freqs", type=int, default=8)
    run.add_argument("--lr", type=float, default=2e-3)
    run.add_argument("--eval-every", type=int, default=150)
    run.add_argument("--selector-loss-weight", type=float, default=0.05)
    run.add_argument("--effect-mask-loss-weight", type=float, default=1.0)

    gen = sub.add_parser("generate", help="generate traces only")
    add_common(gen)
    gen.add_argument("--train-programs", type=int, default=3000)
    gen.add_argument("--val-programs", type=int, default=500)
    gen.add_argument("--long-programs", type=int, default=200)
    gen.add_argument("--max-train-len", type=int, default=4)
    gen.add_argument("--long-len", type=int, default=16)

    train = sub.add_parser("train", help="train against an existing run directory")
    add_common(train)
    train.add_argument("--train-steps", type=int, default=600)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--hidden", type=int, default=192)
    train.add_argument("--freqs", type=int, default=8)
    train.add_argument("--lr", type=float, default=2e-3)
    train.add_argument("--eval-every", type=int, default=150)
    train.add_argument("--selector-loss-weight", type=float, default=0.05)
    train.add_argument("--effect-mask-loss-weight", type=float, default=1.0)

    val = sub.add_parser("validate", help="validate an existing run directory")
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
            source_freeze(args.out_dir, args.model_kind, args.arch, args.external_oracle)
            result = generate_dataset(args, args.out_dir, monitor)
            print(json.dumps(result, indent=2, sort_keys=True))
        finally:
            monitor.stop()
        return 0
    if args.cmd == "train":
        monitor = ResourceMonitor(args.out_dir, args.monitor_interval)
        monitor.start()
        try:
            if args.model_kind == "structured":
                result = train_structured_model(args, args.out_dir, monitor)
            else:
                result = train_model(args, args.out_dir, monitor)
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
