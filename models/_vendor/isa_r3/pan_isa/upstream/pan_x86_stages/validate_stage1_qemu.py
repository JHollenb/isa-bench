#!/usr/bin/env python3
"""Validate Stage 1 traces with a QEMU long-mode boot-image oracle."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, ResourceMonitor, load_traces, write_json
from pan_x86_stages.stage1_memory_pan import INDEX_REGS, MEM_BASE, MEM_SIZE, OPS
from pan_x86_stages.validate_stage0_qemu import (
    FLOPPY_IMAGE_BYTES,
    MAX_FLOPPY_STAGE_SECTORS,
    generate_boot_sector,
    run_qemu_until_result,
    text_tail,
)


DEFAULT_RUN_DIR = (
    DEFAULT_ARTIFACT_ROOT
    / DEFAULT_DATE
    / "pan-x86-isa"
    / "stage1_address_memory"
    / "stage1-memory-indexed-neural-unicorn"
)
ORIGIN = 0x8000
SERIAL_DATA = 0x03F8
SERIAL_LINE_STATUS = SERIAL_DATA + 5
DEBUG_EXIT_PORT = 0x00F4
BUFFER_ADDR = 0x100000
STACK_ADDR = 0x108000
PML4_ADDR = 0x110000
PDPT_ADDR = 0x111000
PD_ADDR = 0x112000
MAX_STAGE_SECTORS = MAX_FLOPPY_STAGE_SECTORS


class Code:
    def __init__(self, origin: int):
        self.origin = origin
        self.buf = bytearray()
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[str, int, str]] = []

    @property
    def offset(self) -> int:
        return len(self.buf)

    def addr(self, label: str) -> int:
        return self.origin + self.labels[label]

    def label(self, name: str) -> None:
        if name in self.labels:
            raise ValueError(f"duplicate label {name}")
        self.labels[name] = self.offset

    def emit(self, *values: int) -> None:
        self.buf.extend(v & 0xFF for v in values)

    def emit_bytes(self, values: bytes) -> None:
        self.buf.extend(values)

    def u16(self, value: int) -> None:
        self.emit(value, value >> 8)

    def u32(self, value: int) -> None:
        self.emit(value, value >> 8, value >> 16, value >> 24)

    def u64(self, value: int) -> None:
        self.u32(value & 0xFFFFFFFF)
        self.u32((value >> 32) & 0xFFFFFFFF)

    def fix(self, kind: str, label: str) -> None:
        self.fixups.append((kind, self.offset, label))
        if kind in ("abs16", "rel16"):
            self.u16(0)
        elif kind in ("abs32", "rel32"):
            self.u32(0)
        elif kind == "abs64":
            self.u64(0)
        else:
            raise ValueError(kind)

    def jmp32(self, label: str) -> None:
        self.emit(0xE9)
        self.fix("rel32", label)

    def call32(self, label: str) -> None:
        self.emit(0xE8)
        self.fix("rel32", label)

    def jne32(self, label: str) -> None:
        self.emit(0x0F, 0x85)
        self.fix("rel32", label)

    def je32(self, label: str) -> None:
        self.emit(0x0F, 0x84)
        self.fix("rel32", label)

    def resolve(self) -> bytes:
        for kind, pos, label in self.fixups:
            if label not in self.labels:
                raise ValueError(f"unknown label {label}")
            target = self.labels[label]
            abs_target = self.origin + target
            if kind == "abs16":
                if not 0 <= abs_target <= 0xFFFF:
                    raise ValueError(f"abs16 out of range: {label}=0x{abs_target:x}")
                self.buf[pos] = abs_target & 0xFF
                self.buf[pos + 1] = (abs_target >> 8) & 0xFF
            elif kind == "abs32":
                self.buf[pos:pos + 4] = int(abs_target).to_bytes(4, "little")
            elif kind == "abs64":
                self.buf[pos:pos + 8] = int(abs_target).to_bytes(8, "little")
            elif kind == "rel16":
                rel = target - (pos + 2)
                if not -32768 <= rel <= 32767:
                    raise ValueError(f"rel16 out of range for {label}: {rel}")
                self.buf[pos:pos + 2] = int(rel & 0xFFFF).to_bytes(2, "little")
            elif kind == "rel32":
                rel = target - (pos + 4)
                if not -(1 << 31) <= rel < (1 << 31):
                    raise ValueError(f"rel32 out of range for {label}: {rel}")
                self.buf[pos:pos + 4] = int(rel & 0xFFFFFFFF).to_bytes(4, "little")
            else:
                raise ValueError(kind)
        if self.origin + len(self.buf) >= BUFFER_ADDR:
            raise ValueError(f"second stage overlaps scratch buffer: end=0x{self.origin + len(self.buf):x}")
        return bytes(self.buf)


def collect_steps(
    run_dir: Path,
    splits: Sequence[str],
    max_steps: int,
    per_split_limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for split in splits:
        checked = 0
        stop_split = False
        for trace in load_traces(run_dir / f"{split}_traces.jsonl"):
            for step in trace["steps"]:
                if len(selected) >= max_steps:
                    counts[split] = checked
                    return selected, counts
                if per_split_limit is not None and checked >= per_split_limit:
                    stop_split = True
                    break
                selected.append(
                    {
                        "split": split,
                        "trace_id": trace["trace_id"],
                        "step_index": step["index"],
                        "asm": trace["program"]["asm"][step["index"]],
                        "bytes_hex": step["decode"]["bytes_hex"],
                        "step": step,
                    }
                )
                checked += 1
            if stop_split:
                break
        counts[split] = checked
    return selected, counts


def emit_serial_init16(code: Code) -> None:
    for port, value in (
        (SERIAL_DATA + 1, 0x00),
        (SERIAL_DATA + 3, 0x80),
        (SERIAL_DATA + 0, 0x03),
        (SERIAL_DATA + 1, 0x00),
        (SERIAL_DATA + 3, 0x03),
        (SERIAL_DATA + 2, 0xC7),
        (SERIAL_DATA + 4, 0x0B),
    ):
        code.emit(0xBA)
        code.u16(port)
        code.emit(0xB0, value, 0xEE)


def emit_mov_dword_abs32(code: Code, addr: int, value: int) -> None:
    code.emit(0xC7, 0x05)
    code.u32(addr)
    code.u32(value)


def emit_enter_long_mode(code: Code) -> None:
    code.emit(0xE9)
    code.fix("rel16", "after_boot_tables")
    code.label("gdt")
    code.emit_bytes(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    code.emit_bytes(b"\xff\xff\x00\x00\x00\x9a\xcf\x00")
    code.emit_bytes(b"\xff\xff\x00\x00\x00\x92\xcf\x00")
    code.emit_bytes(b"\xff\xff\x00\x00\x00\x9a\xaf\x00")
    code.label("gdtr")
    code.u16((4 * 8) - 1)
    code.fix("abs32", "gdt")
    code.label("after_boot_tables")
    code.emit(0xFA)
    emit_serial_init16(code)
    code.emit(0xE4, 0x92, 0x0C, 0x02, 0xE6, 0x92)
    code.emit(0x0F, 0x01, 0x16)
    code.fix("abs16", "gdtr")
    code.emit(0x0F, 0x20, 0xC0, 0x66, 0x83, 0xC8, 0x01, 0x0F, 0x22, 0xC0)
    code.emit(0x66, 0xEA)
    code.fix("abs32", "protected_entry")
    code.u16(0x08)

    code.label("protected_entry")
    code.emit(0x66, 0xB8)
    code.u16(0x10)
    code.emit(0x8E, 0xD8, 0x8E, 0xC0, 0x8E, 0xD0)
    code.emit(0xBF)
    code.u32(PML4_ADDR)
    code.emit(0x31, 0xC0, 0xB9)
    code.u32((4096 * 3) // 4)
    code.emit(0xF3, 0xAB)
    emit_mov_dword_abs32(code, PML4_ADDR, PDPT_ADDR | 0x03)
    emit_mov_dword_abs32(code, PML4_ADDR + 4, 0)
    emit_mov_dword_abs32(code, PDPT_ADDR, PD_ADDR | 0x03)
    emit_mov_dword_abs32(code, PDPT_ADDR + 4, 0)
    code.emit(0xBF)
    code.u32(PD_ADDR)
    code.emit(0xB8)
    code.u32(0x83)
    code.emit(0xB9)
    code.u32(512)
    code.label("pd_loop")
    code.emit(0x89, 0x07, 0xC7, 0x47, 0x04)
    code.u32(0)
    code.emit(0x05)
    code.u32(0x200000)
    code.emit(0x83, 0xC7, 0x08, 0xE2)
    loop_rel = code.labels["pd_loop"] - (code.offset + 1)
    code.emit(loop_rel & 0xFF)

    code.emit(0x0F, 0x20, 0xE0, 0x83, 0xC8, 0x20, 0x0F, 0x22, 0xE0)
    code.emit(0xB8)
    code.u32(PML4_ADDR)
    code.emit(0x0F, 0x22, 0xD8)
    code.emit(0xB9)
    code.u32(0xC0000080)
    code.emit(0x0F, 0x32, 0x0D)
    code.u32(0x100)
    code.emit(0x0F, 0x30)
    code.emit(0x0F, 0x20, 0xC0, 0x0D)
    code.u32(0x80000001)
    code.emit(0x0F, 0x22, 0xC0)
    code.emit(0xEA)
    code.fix("abs32", "long_entry")
    code.u16(0x18)

    code.label("long_entry")
    code.emit(0x66, 0xB8)
    code.u16(0x10)
    code.emit(0x8E, 0xD8, 0x8E, 0xC0, 0x8E, 0xD0)
    code.emit(0x48, 0xBC)
    code.u64(STACK_ADDR)
    code.emit(0xFC)


def emit_puts64(code: Code) -> None:
    code.label("putc64")
    code.emit(0x50, 0x52, 0x66, 0xBA)
    code.u16(SERIAL_LINE_STATUS)
    code.label("putc64_wait")
    code.emit(0xEC, 0xA8, 0x20)
    code.emit(0x74)
    rel = code.labels["putc64_wait"] - (code.offset + 1)
    code.emit(rel & 0xFF)
    code.emit(0x66, 0xBA)
    code.u16(SERIAL_DATA)
    code.emit(0x8A, 0x44, 0x24, 0x08, 0xEE, 0x5A, 0x58, 0xC3)

    code.label("puts64")
    code.label("puts64_loop")
    code.emit(0xAC, 0x84, 0xC0)
    code.emit(0x74)
    done_patch = code.offset
    code.emit(0)
    code.call32("putc64")
    code.jmp32("puts64_loop")
    code.label("puts64_done")
    code.buf[done_patch] = (code.labels["puts64_done"] - (done_patch + 1)) & 0xFF
    code.emit(0xC3)


def emit_memcheck64(code: Code) -> None:
    code.label("memcheck64")
    code.emit(0x45, 0x31, 0xC0)
    code.label("memcheck_loop")
    code.emit(0x41, 0x39, 0xC8)
    code.je32("memcheck_ok")
    code.emit(0x41, 0x39, 0xD0)
    code.je32("memcheck_next")
    code.emit(0x42, 0x8A, 0x04, 0x06)
    code.emit(0x42, 0x38, 0x04, 0x07)
    code.jne32("memcheck_bad")
    code.label("memcheck_next")
    code.emit(0x41, 0xFF, 0xC0)
    code.jmp32("memcheck_loop")
    code.label("memcheck_ok")
    code.emit(0x31, 0xC0, 0xC3)
    code.label("memcheck_bad")
    code.emit(0xB8)
    code.u32(1)
    code.emit(0xC3)


def emit_mov_reg_imm64(code: Code, reg: str, value: int) -> None:
    opcodes = {
        "rax": (0x48, 0xB8),
        "rcx": (0x48, 0xB9),
        "rdx": (0x48, 0xBA),
        "rbx": (0x48, 0xBB),
        "rbp": (0x48, 0xBD),
        "rsi": (0x48, 0xBE),
        "rdi": (0x48, 0xBF),
        "r8": (0x49, 0xB8),
        "r9": (0x49, 0xB9),
    }
    code.emit(*opcodes[reg])
    code.u64(value & 0xFFFFFFFFFFFFFFFF)


def emit_compare_reg_imm64(code: Code, reg_index: int, value: int, fail_label: str) -> None:
    emit_mov_reg_imm64(code, "r8", value)
    code.emit(0x4C, 0x39, 0xC0 + reg_index)
    code.jne32(fail_label)


def expected_compare_regs(p: Dict[str, Any]) -> List[int]:
    expected = [int(x) & 0xFFFFFFFFFFFFFFFF for x in p["after_data_regs"]]
    op = OPS[int(p["op_index"])]
    if op.name == "LEA_RA":
        dst = int(p["dst"])
        expected[dst] = BUFFER_ADDR + (expected[dst] - MEM_BASE)
    return expected


def emit_case64(code: Code, index: int, item: Dict[str, Any]) -> None:
    p = item["step"]["pan_stage1"]
    fail = f"fail_{index}"
    next_case = f"case_{index + 1}"
    before_mem_label = f"before_mem_{index}"
    after_mem_label = f"after_mem_{index}"

    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, before_mem_label))
    emit_mov_reg_imm64(code, "rdi", BUFFER_ADDR)
    code.emit(0xB9)
    code.u32(MEM_SIZE)
    code.emit(0xFC, 0xF3, 0xA4)

    before_regs = [int(x) & 0xFFFFFFFFFFFFFFFF for x in p["before_data_regs"]]
    for reg, value in zip(("rax", "rcx", "rdx", "rbx"), before_regs):
        emit_mov_reg_imm64(code, reg, value)
    addr_offsets = [int(x) for x in p["before_addr_offsets"]]
    emit_mov_reg_imm64(code, "rsi", BUFFER_ADDR + addr_offsets[0])
    emit_mov_reg_imm64(code, "rdi", BUFFER_ADDR + addr_offsets[1])
    emit_mov_reg_imm64(code, "rbp", BUFFER_ADDR + addr_offsets[2])
    index_offsets = [int(x) for x in p.get("before_index_offsets", [0 for _ in INDEX_REGS])]
    emit_mov_reg_imm64(code, "r8", index_offsets[0])
    emit_mov_reg_imm64(code, "r9", index_offsets[1])

    code.emit_bytes(bytes.fromhex(item["bytes_hex"]))

    for reg_index, expected in enumerate(expected_compare_regs(p)):
        emit_compare_reg_imm64(code, reg_index, expected, fail)

    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, after_mem_label))
    emit_mov_reg_imm64(code, "rdi", BUFFER_ADDR)
    code.emit(0xB9)
    code.u32(MEM_SIZE)
    code.emit(0xBA)
    code.u32(0xFFFFFFFF)
    code.call32("memcheck64")
    code.emit(0x85, 0xC0)
    code.jne32(fail)
    code.jmp32(next_case)

    code.label(fail)
    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, f"fail_msg_{index}"))
    code.call32("puts64")
    code.jmp32("final_fail")
    code.label(next_case)


def emit_progress64(code: Code, label: str) -> None:
    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, label))
    code.call32("puts64")


def emit_exit_hang64(code: Code, label: str) -> None:
    code.emit(0xB0, 0x00, 0x66, 0xBA)
    code.u16(DEBUG_EXIT_PORT)
    code.emit(0xEE)
    code.label(label)
    code.emit(0xF4)
    code.jmp32(label)


def generate_second_stage(selected: Sequence[Dict[str, Any]], progress_interval: int = 0) -> bytes:
    code = Code(ORIGIN)
    emit_enter_long_mode(code)
    if progress_interval > 0:
        emit_progress64(code, "progress_begin")
    for index, item in enumerate(selected):
        if progress_interval > 0 and index > 0 and index % progress_interval == 0:
            emit_progress64(code, f"progress_{index}")
        emit_case64(code, index, item)

    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, "success_msg"))
    code.call32("puts64")
    emit_exit_hang64(code, "success_hang64")

    code.label("final_fail")
    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, "failure_result_msg"))
    code.call32("puts64")
    emit_exit_hang64(code, "failure_hang64")

    emit_puts64(code)
    emit_memcheck64(code)

    code.label("success_msg")
    code.emit_bytes(f"RESULT {len(selected)} {len(selected)} 0\n".encode("ascii") + b"\x00")
    code.label("failure_result_msg")
    code.emit_bytes(f"RESULT {len(selected)} 0 1\n".encode("ascii") + b"\x00")
    if progress_interval > 0:
        code.label("progress_begin")
        code.emit_bytes(f"BEGIN {len(selected)}\n".encode("ascii") + b"\x00")
        for index in range(progress_interval, len(selected), progress_interval):
            code.label(f"progress_{index}")
            code.emit_bytes(f"PROGRESS {index}\n".encode("ascii") + b"\x00")
    for index in range(len(selected)):
        code.label(f"fail_msg_{index}")
        code.emit_bytes(f"FAIL {index}\n".encode("ascii") + b"\x00")
    for index, item in enumerate(selected):
        code.label(f"before_mem_{index}")
        before_mem = bytes(int(x) & 0xFF for x in item["step"]["pan_stage1"]["before_memory"])
        if len(before_mem) != MEM_SIZE:
            raise ValueError(f"unexpected memory size for case {index}: {len(before_mem)}")
        code.emit_bytes(before_mem)
        code.label(f"after_mem_{index}")
        after_mem = bytes(int(x) & 0xFF for x in item["step"]["pan_stage1"]["after_memory"])
        if len(after_mem) != MEM_SIZE:
            raise ValueError(f"unexpected memory size for case {index}: {len(after_mem)}")
        code.emit_bytes(after_mem)
    return code.resolve()


def build_image(
    selected: Sequence[Dict[str, Any]],
    image_path: Path,
    allow_multitrack: bool,
    progress_interval: int = 0,
) -> Dict[str, Any]:
    second_stage = generate_second_stage(selected, progress_interval=progress_interval)
    sectors = math.ceil(len(second_stage) / 512)
    if sectors > MAX_STAGE_SECTORS:
        raise ValueError(f"generated second stage needs {sectors} sectors; limit is {MAX_STAGE_SECTORS}")
    boot = generate_boot_sector(sectors, load_addr=ORIGIN)
    padded_stage = second_stage + b"\x00" * (sectors * 512 - len(second_stage))
    image = boot + padded_stage
    if len(image) > FLOPPY_IMAGE_BYTES:
        raise ValueError(f"boot image exceeds 1.44MB floppy size: {len(image)} bytes")
    image = image + b"\x00" * (FLOPPY_IMAGE_BYTES - len(image))
    image_path.write_bytes(image)
    return {
        "image_path": str(image_path),
        "image_bytes": len(image),
        "used_image_bytes": 512 + len(padded_stage),
        "floppy_image_bytes": FLOPPY_IMAGE_BYTES,
        "loader": "bios_chs_multitrack_floppy",
        "second_stage_bytes": len(second_stage),
        "second_stage_sectors": sectors,
        "max_stage_sectors": MAX_STAGE_SECTORS,
        "allow_multitrack": allow_multitrack,
        "progress_interval": progress_interval,
    }


def parse_qemu_output(stdout: str, selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    checked = 0
    exact = 0
    failures = 0
    failure_examples = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "FAIL" and len(parts) == 2:
            idx = int(parts[1])
            if 0 <= idx < len(selected):
                item = selected[idx]
                failure_examples.append(
                    {
                        "index": idx,
                        "split": item["split"],
                        "trace_id": item["trace_id"],
                        "step": item["step_index"],
                        "asm": item["asm"],
                        "bytes_hex": item["bytes_hex"],
                    }
                )
        elif parts[0] == "RESULT" and len(parts) == 4:
            checked = int(parts[1])
            exact = int(parts[2])
            failures = int(parts[3])
    return {
        "checked_steps": checked,
        "exact_steps": exact,
        "exact": round(exact / max(1, checked), 6),
        "failure_count": failures,
        "failure_examples": failure_examples,
    }


def run_qemu_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = args.report_dir or (args.run_dir / "qemu_stage1_oracle")
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(report_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("collect")
        selected, source_counts = collect_steps(args.run_dir, args.splits, args.max_steps, args.per_split_limit)
        if not selected:
            raise ValueError("no Stage 1 steps selected")

        qemu = shutil.which(args.qemu_binary)
        availability = {
            "qemu_binary": args.qemu_binary,
            "qemu_path": qemu,
            "qemu_available": qemu is not None,
        }
        if qemu is None:
            report = {
                "enabled": True,
                "oracle": "qemu_system_x86_64_long_mode",
                "ok": False,
                "error": f"{args.qemu_binary} not found",
                "availability": availability,
            }
            write_json(report_dir / "qemu_stage1_oracle_report.json", report)
            write_json(args.run_dir / "qemu_stage1_oracle_report.json", report)
            return report

        monitor.set_phase("build_image")
        image_info = build_image(
            selected,
            report_dir / "stage1_qemu_oracle.img",
            args.allow_multitrack,
            args.progress_interval,
        )
        serial_path = report_dir / "qemu_serial.log"
        serial_path.write_text("", encoding="utf-8")

        cmd = [
            qemu,
            "-display",
            "none",
            "-monitor",
            "none",
            "-serial",
            f"file:{serial_path}",
            "-boot",
            "a",
            "-no-reboot",
            "-no-shutdown",
            "-device",
            "isa-debug-exit,iobase=0xf4,iosize=0x01",
            "-drive",
            f"file={image_info['image_path']},format=raw,if=floppy",
        ]
        monitor.set_phase("execute")
        run = run_qemu_until_result(cmd, serial_path, args.timeout_s)
        parsed = parse_qemu_output(run["serial"], selected)
        ok = (
            parsed["checked_steps"] == len(selected)
            and parsed["exact_steps"] == len(selected)
            and parsed["failure_count"] == 0
            and run["stopped_after_result"]
        )
        split_counts = {
            split: {"checked_steps": sum(1 for item in selected if item["split"] == split)}
            for split in args.splits
        }
        report = {
            "enabled": True,
            "oracle": "qemu_system_x86_64_long_mode",
            "ok": ok,
            "stage": "stage1_address_memory",
            "run_dir": str(args.run_dir),
            "report_dir": str(report_dir),
            "checked_steps": parsed["checked_steps"],
            "exact_steps": parsed["exact_steps"],
            "exact": parsed["exact"],
            "failure_count": parsed["failure_count"],
            "failure_examples": parsed["failure_examples"],
            "splits": split_counts,
            "requested_splits": list(args.splits),
            "source_trace_counts": source_counts,
            "max_steps": args.max_steps,
            "per_split_limit": args.per_split_limit,
            "elapsed_s": run["elapsed_s"],
            "availability": availability,
            "image": image_info,
            "cmd": cmd,
            "returncode": run["returncode"],
            "timed_out": run["timed_out"],
            "stopped_after_result": run["stopped_after_result"],
            "serial_path": str(serial_path),
            "serial_tail": run["serial"][-4000:],
            "stdout_tail": run["stdout"][-4000:],
            "stderr_tail": run["stderr"][-4000:],
            "normalization": (
                "Stage 1 executes the x86-64 instruction bytes in QEMU long mode. "
                "LEA register results are normalized from MEM_BASE to the boot-image scratch buffer."
            ),
        }
        write_json(report_dir / "qemu_stage1_oracle_report.json", report)
        write_json(args.run_dir / "qemu_stage1_oracle_report.json", report)
        return report
    finally:
        monitor.set_phase("finalize")
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "long"], default=["val", "long"])
    p.add_argument("--max-steps", type=int, default=8)
    p.add_argument("--per-split-limit", type=int, default=4)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--qemu-binary", default="qemu-system-x86_64")
    p.add_argument("--allow-multitrack", action="store_true")
    p.add_argument("--progress-interval", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_qemu_oracle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
