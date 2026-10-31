#!/usr/bin/env python3
"""Validate Stage 3 traces with QEMU or Bochs long-mode boot-image oracles."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, FLAGS, ResourceMonitor, load_traces, write_json
from pan_x86_stages.stage3_complex_scalar_pan import CODE_BASE, MEM_BASE, MEM_SIZE, OPS, REGS
from pan_x86_stages.validate_stage0_qemu import FLOPPY_IMAGE_BYTES, MAX_FLOPPY_STAGE_SECTORS, generate_boot_sector, run_qemu_until_result
from pan_x86_stages.validate_stage1_qemu import Code, DEBUG_EXIT_PORT, ORIGIN, emit_enter_long_mode, emit_memcheck64, emit_mov_reg_imm64, emit_puts64
from pan_x86_stages.validate_stage2_bochs import bochs_command, run_bochs_until_result
from pan_x86_stages.validate_stage2_qemu import (
    CODE_CLEAR_BYTES,
    HARNESS_STACK_ADDR,
    emit_copy_label_to_addr,
    emit_exit_hang64,
    emit_jmp_r11,
    emit_mov_r11_imm64,
    emit_mov_rsp_imm64,
    emit_pushfq_from_word,
    emit_memset64,
)


DEFAULT_RUN_DIR = (
    DEFAULT_ARTIFACT_ROOT
    / DEFAULT_DATE
    / "pan-x86-isa"
    / "stage3_complex_scalar"
    / "stage3-complex-scalar-base-alu-v21-neural"
)
RESULT_BASE = 0x140000
RESULT_ADDRS = {
    "RAX": RESULT_BASE,
    "RBX": RESULT_BASE + 0x08,
    "RCX": RESULT_BASE + 0x10,
    "RDX": RESULT_BASE + 0x18,
    "RIP": RESULT_BASE + 0x20,
    "FLAGS": RESULT_BASE + 0x28,
}
RESULT_EXCEPTION_VECTOR = RESULT_BASE + 0x30
RESULT_EXCEPTION_ERROR = RESULT_BASE + 0x38
EXCEPTION_CONTINUATION_ADDR = RESULT_BASE + 0x40
IDTR_ADDR = 0x12FF00
IDT_ADDR = 0x130000
DE_VECTOR = 0
FLAG_BITS = {"CF": 0, "ZF": 6, "SF": 7, "OF": 11}
TRAMPOLINE_BYTES = 82
MAX_STAGE_SECTORS = MAX_FLOPPY_STAGE_SECTORS


def flags_word(flags: Sequence[int]) -> int:
    word = 0x0002
    for value, name in zip(flags, FLAGS):
        if int(value):
            word |= 1 << FLAG_BITS[name]
    return word & 0xFFFFFFFF


def flag_mask_word(mask: Sequence[int]) -> int:
    word = 0
    for value, name in zip(mask, FLAGS):
        if int(value):
            word |= 1 << FLAG_BITS[name]
    return word


def step_expected_exception(step: Dict[str, Any]) -> Optional[str]:
    return step["pan_stage3"].get("exception") or step["effects"].get("exception")


def step_trampoline_rip(step: Dict[str, Any]) -> int:
    p = step["pan_stage3"]
    if step_expected_exception(step):
        return int(p.get("fallthrough_rip", int(p["before_rip"]) + len(bytes.fromhex(step["decode"]["bytes_hex"]))))
    return int(p["after_rip"])


def is_safe_step(step: Dict[str, Any]) -> bool:
    p = step["pan_stage3"]
    before_rip = int(p["before_rip"])
    insn_len = len(bytes.fromhex(step["decode"]["bytes_hex"]))
    insn_start = before_rip
    insn_end = before_rip + insn_len
    tramp_start = step_trampoline_rip(step)
    tramp_end = tramp_start + TRAMPOLINE_BYTES
    code_start = CODE_BASE
    code_end = CODE_BASE + CODE_CLEAR_BYTES
    if not (code_start <= insn_start and insn_end <= code_end):
        return False
    if not (code_start <= tramp_start and tramp_end <= code_end):
        return False
    return tramp_end <= insn_start or insn_end <= tramp_start


def collect_steps(
    run_dir: Path,
    splits: Sequence[str],
    max_steps: int,
    per_split_limit: Optional[int],
    priority_category: Optional[str] = None,
    priority_min: int = 0,
    priority_operands: Optional[Sequence[str]] = None,
    priority_iclasses: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int], int]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {split: 0 for split in splits}
    skipped: Dict[str, int] = {split: 0 for split in splits}
    seen: set[Tuple[str, str, int]] = set()
    skipped_seen: set[Tuple[str, str, int]] = set()
    priority_selected = 0
    priority_categories = {
        category.strip()
        for category in str(priority_category or "").split(",")
        if category.strip()
    }
    priority_operand_set = set(priority_operands or [])
    priority_iclass_set = set(priority_iclasses or [])

    def step_key(split: str, trace: Dict[str, Any], step: Dict[str, Any]) -> Tuple[str, str, int]:
        return split, str(trace["trace_id"]), int(step["index"])

    def append_step(split: str, trace: Dict[str, Any], step: Dict[str, Any]) -> bool:
        key = step_key(split, trace, step)
        if key in seen:
            return False
        if not is_safe_step(step):
            if key not in skipped_seen:
                skipped[split] += 1
                skipped_seen.add(key)
            return False
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
        seen.add(key)
        counts[split] += 1
        return True

    def priority_matches(step: Dict[str, Any]) -> bool:
        decode = step["decode"]
        if priority_categories and decode.get("category") not in priority_categories:
            return False
        if priority_operand_set:
            op = OPS[int(step["pan_stage3"]["op_index"])]
            if op.operands not in priority_operand_set:
                return False
        if priority_iclass_set and decode.get("iclass") not in priority_iclass_set:
            return False
        return True

    if (priority_categories or priority_operand_set or priority_iclass_set) and priority_min > 0:
        for split in splits:
            if priority_selected >= priority_min or len(selected) >= max_steps:
                break
            for trace in load_traces(run_dir / f"{split}_traces.jsonl"):
                if priority_selected >= priority_min or len(selected) >= max_steps:
                    break
                for step in trace["steps"]:
                    if priority_selected >= priority_min or len(selected) >= max_steps:
                        break
                    if per_split_limit is not None and counts[split] >= per_split_limit:
                        break
                    if not priority_matches(step):
                        continue
                    if append_step(split, trace, step):
                        priority_selected += 1

    for split in splits:
        stop_split = False
        for trace in load_traces(run_dir / f"{split}_traces.jsonl"):
            for step in trace["steps"]:
                if len(selected) >= max_steps:
                    return selected, counts, skipped, priority_selected
                if per_split_limit is not None and counts[split] >= per_split_limit:
                    stop_split = True
                    break
                append_step(split, trace, step)
            if stop_split:
                break
    return selected, counts, skipped, priority_selected


def count_memory_operand_iclasses(items: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        step = item["step"]
        if "mem_value" not in step["pan_stage3"]:
            continue
        iclass = str(step["decode"]["iclass"])
        counts[iclass] = counts.get(iclass, 0) + 1
    return dict(sorted(counts.items()))


def emit_store_reg_abs32(code: Code, reg_code: int, addr: int) -> None:
    code.emit(0x48, 0x89, 0x04 | ((reg_code & 7) << 3), 0x25)
    code.u32(addr)


def emit_store_rax_abs32(code: Code, addr: int) -> None:
    emit_store_reg_abs32(code, 0, addr)


def emit_load_abs32_rax(code: Code, addr: int) -> None:
    code.emit(0x48, 0x8B, 0x04, 0x25)
    code.u32(addr)


def emit_store_eax_abs32(code: Code, addr: int) -> None:
    code.emit(0x89, 0x04, 0x25)
    code.u32(addr)


def emit_store_ax_abs32(code: Code, addr: int) -> None:
    code.emit(0x66, 0x89, 0x04, 0x25)
    code.u32(addr)


def emit_store_word_imm_abs32(code: Code, addr: int, value: int) -> None:
    code.emit(0x66, 0xC7, 0x04, 0x25)
    code.u32(addr)
    code.u16(value)


def emit_store_dword_imm_abs32(code: Code, addr: int, value: int) -> None:
    code.emit(0xC7, 0x04, 0x25)
    code.u32(addr)
    code.u32(value)


def emit_mov_rax_label64(code: Code, label: str) -> None:
    code.emit(0x48, 0xB8)
    code.fix("abs64", label)


def emit_store_label_addr_abs32(code: Code, addr: int, label: str) -> None:
    emit_mov_rax_label64(code, label)
    emit_store_rax_abs32(code, addr)


def emit_compare_rax_imm64(code: Code, value: int, fail_label: str) -> None:
    emit_mov_reg_imm64(code, "r8", value)
    code.emit(0x4C, 0x39, 0xC0)
    code.jne32(fail_label)


def emit_compare_result64(code: Code, addr: int, expected: int, fail_label: str) -> None:
    emit_load_abs32_rax(code, addr)
    emit_compare_rax_imm64(code, expected & 0xFFFFFFFFFFFFFFFF, fail_label)


def emit_compare_flags(code: Code, expected_flags: Sequence[int], mask: Sequence[int], fail_label: str) -> None:
    mask_word = flag_mask_word(mask)
    if mask_word == 0:
        return
    emit_load_abs32_rax(code, RESULT_ADDRS["FLAGS"])
    code.emit(0x48, 0x25)
    code.u32(mask_word)
    emit_compare_rax_imm64(code, flags_word(expected_flags) & mask_word, fail_label)


def emit_setup_idt64(code: Code) -> None:
    emit_memset64(code, IDT_ADDR, 0, 4096)
    gate_addr = IDT_ADDR + DE_VECTOR * 16
    emit_mov_rax_label64(code, "de_handler")
    emit_store_ax_abs32(code, gate_addr)
    emit_store_dword_imm_abs32(code, gate_addr + 2, 0x8E000018)
    code.emit(0x48, 0xC1, 0xE8, 0x10)
    emit_store_ax_abs32(code, gate_addr + 6)
    code.emit(0x48, 0xC1, 0xE8, 0x10)
    emit_store_eax_abs32(code, gate_addr + 8)
    emit_store_dword_imm_abs32(code, gate_addr + 12, 0)
    emit_store_word_imm_abs32(code, IDTR_ADDR, 4095)
    emit_mov_reg_imm64(code, "rax", IDT_ADDR)
    code.emit(0x48, 0xA3)
    code.u64(IDTR_ADDR + 2)
    code.emit(0x0F, 0x01, 0x1C, 0x25)
    code.u32(IDTR_ADDR)


def emit_de_handler64(code: Code) -> None:
    code.label("de_handler")
    emit_store_reg_abs32(code, 0, RESULT_ADDRS["RAX"])
    emit_store_reg_abs32(code, 3, RESULT_ADDRS["RBX"])
    emit_store_reg_abs32(code, 1, RESULT_ADDRS["RCX"])
    emit_store_reg_abs32(code, 2, RESULT_ADDRS["RDX"])
    code.emit(0x48, 0x8B, 0x04, 0x24)
    emit_store_rax_abs32(code, RESULT_ADDRS["RIP"])
    code.emit(0x48, 0x8B, 0x44, 0x24, 0x10)
    emit_store_rax_abs32(code, RESULT_ADDRS["FLAGS"])
    emit_mov_reg_imm64(code, "rax", DE_VECTOR)
    emit_store_rax_abs32(code, RESULT_EXCEPTION_VECTOR)
    emit_mov_reg_imm64(code, "rax", 0)
    emit_store_rax_abs32(code, RESULT_EXCEPTION_ERROR)
    code.emit(0x48, 0x83, 0xC4, 0x18)
    emit_load_abs32_rax(code, EXCEPTION_CONTINUATION_ADDR)
    code.emit(0xFF, 0xE0)


def emit_trampoline_blob(code: Code, label: str, continuation_label: str, actual_rip: int) -> None:
    code.label(label)
    start = code.offset
    emit_store_reg_abs32(code, 0, RESULT_ADDRS["RAX"])
    emit_store_reg_abs32(code, 3, RESULT_ADDRS["RBX"])
    emit_store_reg_abs32(code, 1, RESULT_ADDRS["RCX"])
    emit_store_reg_abs32(code, 2, RESULT_ADDRS["RDX"])
    emit_mov_reg_imm64(code, "rax", actual_rip)
    emit_store_reg_abs32(code, 0, RESULT_ADDRS["RIP"])
    emit_mov_rsp_imm64(code, HARNESS_STACK_ADDR)
    code.emit(0x9C, 0x58)
    emit_store_reg_abs32(code, 0, RESULT_ADDRS["FLAGS"])
    code.emit(0x48, 0xB8)
    code.fix("abs64", continuation_label)
    code.emit(0xFF, 0xE0)
    size = code.offset - start
    if size != TRAMPOLINE_BYTES:
        raise ValueError(f"unexpected trampoline size: {size}")


def emit_compare_memory_result(code: Code, base_addr: int, expected_label: str, fail_label: str) -> None:
    emit_mov_reg_imm64(code, "rsi", base_addr)
    code.emit(0x48, 0xBF)
    code.fix("abs64", expected_label)
    emit_mov_reg_imm64(code, "rcx", MEM_SIZE)
    code.call32("memcheck64")
    code.emit(0x85, 0xC0)
    code.jne32(fail_label)


def emit_case64(code: Code, index: int, item: Dict[str, Any]) -> None:
    p = item["step"]["pan_stage3"]
    has_memory = "before_memory" in p
    reasons = ("RAX", "RBX", "RCX", "RDX", "RIP", "FLAGS", "EXCEPTION", "ERROR", *(("MEMORY",) if has_memory else ()))
    fail_labels = {name: f"fail_{index}_{name.lower()}" for name in reasons}
    next_case = f"case_{index + 1}"
    insn_label = f"instruction_{index}"
    trampoline_label = f"trampoline_{index}"
    expected_exception = step_expected_exception(item["step"])
    expected_vector = DE_VECTOR if expected_exception == "#DE" else -1
    expected_error = 0 if expected_exception == "#DE" else -1
    trampoline_rip = step_trampoline_rip(item["step"])

    emit_memset64(code, CODE_BASE, 0xF4, CODE_CLEAR_BYTES)
    emit_copy_label_to_addr(code, insn_label, int(p["before_rip"]), len(bytes.fromhex(item["bytes_hex"])))
    emit_copy_label_to_addr(code, trampoline_label, trampoline_rip, TRAMPOLINE_BYTES)
    if has_memory:
        if len(p["before_memory"]) != MEM_SIZE:
            raise ValueError(f"case {index} has {len(p['before_memory'])} memory bytes; expected {MEM_SIZE}")
        if len(p["after_memory"]) != MEM_SIZE:
            raise ValueError(f"case {index} has {len(p['after_memory'])} after-memory bytes; expected {MEM_SIZE}")
        emit_copy_label_to_addr(code, f"before_memory_{index}", MEM_BASE, MEM_SIZE)

    before_regs = [int(x) for x in p["before_regs"]]
    for reg_name, value in zip(("rax", "rbx", "rcx", "rdx"), before_regs):
        emit_mov_reg_imm64(code, reg_name, value)
    emit_mov_reg_imm64(code, "rax", 0xFFFFFFFFFFFFFFFF)
    emit_store_rax_abs32(code, RESULT_EXCEPTION_VECTOR)
    emit_store_rax_abs32(code, RESULT_EXCEPTION_ERROR)
    emit_store_label_addr_abs32(code, EXCEPTION_CONTINUATION_ADDR, f"case_{index}_after")
    emit_mov_reg_imm64(code, "rax", before_regs[0])
    emit_mov_rsp_imm64(code, HARNESS_STACK_ADDR)
    emit_pushfq_from_word(code, flags_word(p["before_flags"]))
    emit_mov_r11_imm64(code, int(p["before_rip"]))
    emit_jmp_r11(code)

    code.label(f"case_{index}_after")
    after_regs = [int(x) for x in p["after_regs"]]
    for reg_name, value in zip(("RAX", "RBX", "RCX", "RDX"), after_regs):
        emit_compare_result64(code, RESULT_ADDRS[reg_name], value, fail_labels[reg_name])
    emit_compare_result64(code, RESULT_ADDRS["RIP"], int(p["after_rip"]), fail_labels["RIP"])
    emit_compare_flags(code, p["after_flags"], p["flag_mask"], fail_labels["FLAGS"])
    emit_compare_result64(code, RESULT_EXCEPTION_VECTOR, expected_vector, fail_labels["EXCEPTION"])
    emit_compare_result64(code, RESULT_EXCEPTION_ERROR, expected_error, fail_labels["ERROR"])
    if has_memory:
        emit_compare_memory_result(code, MEM_BASE, f"after_memory_{index}", fail_labels["MEMORY"])
    code.jmp32(next_case)

    for reason, fail_label in fail_labels.items():
        code.label(fail_label)
        emit_mov_reg_imm64(code, "rsi", 0)
        code.fixups.append(("abs64", code.offset - 8, f"fail_msg_{index}_{reason}"))
        code.call32("puts64")
        code.jmp32("final_fail")
    code.label(next_case)


def emit_progress64(code: Code, label: str) -> None:
    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, label))
    code.call32("puts64")


def generate_second_stage(selected: Sequence[Dict[str, Any]], progress_interval: int = 0) -> bytes:
    code = Code(ORIGIN)
    emit_enter_long_mode(code)
    emit_setup_idt64(code)
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
    emit_de_handler64(code)

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
    for index, item in enumerate(selected):
        for reason in ("RAX", "RBX", "RCX", "RDX", "RIP", "FLAGS", "EXCEPTION", "ERROR", *(("MEMORY",) if "before_memory" in item["step"]["pan_stage3"] else ())):
            code.label(f"fail_msg_{index}_{reason}")
            code.emit_bytes(f"FAIL {index} {reason}\n".encode("ascii") + b"\x00")
        code.label(f"instruction_{index}")
        code.emit_bytes(bytes.fromhex(item["bytes_hex"]))
        code.label(f"trampoline_{index}")
        emit_trampoline_blob(code, f"trampoline_blob_{index}", f"case_{index}_after", int(item["step"]["pan_stage3"]["after_rip"]))
        p = item["step"]["pan_stage3"]
        if "before_memory" in p:
            code.label(f"before_memory_{index}")
            code.emit_bytes(bytes(int(x) & 0xFF for x in p["before_memory"]))
            code.label(f"after_memory_{index}")
            code.emit_bytes(bytes(int(x) & 0xFF for x in p["after_memory"]))
    return code.resolve()


def build_image(selected: Sequence[Dict[str, Any]], image_path: Path, progress_interval: int = 0) -> Dict[str, Any]:
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
        "code_clear_bytes": CODE_CLEAR_BYTES,
        "trampoline_bytes": TRAMPOLINE_BYTES,
        "progress_interval": progress_interval,
    }


def parse_serial_output(stdout: str, selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    checked = 0
    exact = 0
    failures = 0
    failure_examples = []
    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "FAIL" and len(parts) in (2, 3):
            idx = int(parts[1])
            if 0 <= idx < len(selected):
                item = selected[idx]
                failure_examples.append(
                    {
                        "index": idx,
                        "global_index": item.get("global_index", idx),
                        "split": item["split"],
                        "trace_id": item["trace_id"],
                        "step": item["step_index"],
                        "asm": item["asm"],
                        "bytes_hex": item["bytes_hex"],
                        "reason": parts[2] if len(parts) == 3 else None,
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


def run_engine(args: argparse.Namespace, image_info: Dict[str, Any], serial_path: Path, log_path: Path) -> Dict[str, Any]:
    if args.engine == "qemu":
        qemu = shutil.which(args.qemu_binary)
        if qemu is None:
            raise FileNotFoundError(f"{args.qemu_binary} not found")
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
            "-cpu",
            args.qemu_cpu,
            "-device",
            "isa-debug-exit,iobase=0xf4,iosize=0x01",
            "-drive",
            f"file={image_info['image_path']},format=raw,if=floppy",
        ]
        return run_qemu_until_result(cmd, serial_path, args.timeout_s)
    bochs = shutil.which(args.bochs_binary)
    if bochs is None:
        raise FileNotFoundError(f"{args.bochs_binary} not found")
    cmd = bochs_command(
        bochs,
        Path(image_info["image_path"]),
        serial_path,
        log_path,
        args.bochs_cpu_model,
        args.bochs_ips,
        args.bochs_memory_mib,
    )
    return run_bochs_until_result(cmd, serial_path, args.timeout_s)


def run_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    report_name = f"{args.engine}_stage3_oracle"
    report_dir = args.report_dir or (args.run_dir / report_name)
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(report_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("collect")
        selected, source_counts, skipped_counts, priority_selected = collect_steps(
            args.run_dir,
            args.splits,
            args.max_steps,
            args.per_split_limit,
            args.priority_category,
            args.priority_min,
            args.priority_operands,
            args.priority_iclasses,
        )
        if not selected:
            raise ValueError("no Stage 3 steps selected")

        chunks = []
        checked_steps = 0
        exact_steps = 0
        failure_count = 0
        failure_examples: List[Dict[str, Any]] = []
        all_stopped_after_result = True
        any_timed_out = False
        last_run: Dict[str, Any] = {}
        for chunk_index, start in enumerate(range(0, len(selected), args.chunk_size)):
            chunk_items = [dict(item, global_index=start + local_index) for local_index, item in enumerate(selected[start:start + args.chunk_size])]
            monitor.set_phase(f"build_image_{chunk_index}")
            image_info = build_image(chunk_items, report_dir / f"stage3_{args.engine}_oracle_{chunk_index:03d}.img", args.progress_interval)
            serial_path = report_dir / f"{args.engine}_serial_{chunk_index:03d}.log"
            log_path = report_dir / f"{args.engine}_{chunk_index:03d}.log"
            serial_path.write_text("", encoding="utf-8")
            monitor.set_phase(f"execute_{chunk_index}")
            run = run_engine(args, image_info, serial_path, log_path)
            parsed = parse_serial_output(run["serial"], chunk_items)
            chunk_ok = (
                parsed["checked_steps"] == len(chunk_items)
                and parsed["exact_steps"] == len(chunk_items)
                and parsed["failure_count"] == 0
                and run["stopped_after_result"]
            )
            checked_steps += parsed["checked_steps"]
            exact_steps += parsed["exact_steps"]
            failure_count += parsed["failure_count"]
            failure_examples.extend(parsed["failure_examples"])
            all_stopped_after_result = all_stopped_after_result and run["stopped_after_result"]
            any_timed_out = any_timed_out or run["timed_out"]
            last_run = run
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_index": start,
                    "steps": len(chunk_items),
                    "ok": chunk_ok,
                    "checked_steps": parsed["checked_steps"],
                    "exact_steps": parsed["exact_steps"],
                    "exact": parsed["exact"],
                    "failure_count": parsed["failure_count"],
                    "failure_examples": parsed["failure_examples"],
                    "elapsed_s": run["elapsed_s"],
                    "returncode": run["returncode"],
                    "timed_out": run["timed_out"],
                    "stopped_after_result": run["stopped_after_result"],
                    "serial_path": str(serial_path),
                    "log_path": str(log_path),
                    "serial_tail": run["serial"][-2000:],
                    "stderr_tail": run["stderr"][-2000:],
                    "image": image_info,
                }
            )
            if not chunk_ok:
                break

        op_counts: Dict[str, int] = {}
        family_counts: Dict[str, int] = {}
        extension_counts: Dict[str, int] = {}
        for item in selected:
            step = item["step"]
            op = str(step["decode"]["iclass"])
            family = str(step["decode"]["category"])
            extension = str(step["decode"].get("extension", "BASE"))
            op_counts[op] = op_counts.get(op, 0) + 1
            family_counts[family] = family_counts.get(family, 0) + 1
            extension_counts[extension] = extension_counts.get(extension, 0) + 1

        ok = checked_steps == len(selected) and exact_steps == len(selected) and failure_count == 0 and all_stopped_after_result
        report = {
            "enabled": True,
            "oracle": f"{args.engine}_x86_64_long_mode",
            "ok": ok,
            "stage": "stage3_complex_scalar",
            "run_dir": str(args.run_dir),
            "report_dir": str(report_dir),
            "checked_steps": checked_steps,
            "exact_steps": exact_steps,
            "exact": round(exact_steps / max(1, checked_steps), 6),
            "failure_count": failure_count,
            "failure_examples": failure_examples,
            "exception_steps": sum(1 for item in selected if step_expected_exception(item["step"])),
            "memory_operand_steps": sum(1 for item in selected if "mem_value" in item["step"]["pan_stage3"]),
            "memory_operand_iclasses": count_memory_operand_iclasses(selected),
            "memory_source_steps": sum(1 for item in selected if "mem_value" in item["step"]["pan_stage3"]),
            "memory_source_iclasses": count_memory_operand_iclasses(selected),
            "op_counts": dict(sorted(op_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "extension_counts": dict(sorted(extension_counts.items())),
            "splits": {
                split: {
                    "checked_steps": sum(1 for item in selected if item["split"] == split),
                    "exception_steps": sum(1 for item in selected if item["split"] == split and step_expected_exception(item["step"])),
                    "memory_operand_steps": sum(1 for item in selected if item["split"] == split and "mem_value" in item["step"]["pan_stage3"]),
                    "memory_operand_iclasses": count_memory_operand_iclasses([item for item in selected if item["split"] == split]),
                    "memory_source_steps": sum(1 for item in selected if item["split"] == split and "mem_value" in item["step"]["pan_stage3"]),
                    "memory_source_iclasses": count_memory_operand_iclasses([item for item in selected if item["split"] == split]),
                }
                for split in args.splits
            },
            "requested_splits": list(args.splits),
            "source_trace_counts": source_counts,
            "skipped_unsafe_steps": skipped_counts,
            "priority_category": args.priority_category,
            "priority_operands": list(args.priority_operands or []),
            "priority_iclasses": list(args.priority_iclasses or []),
            "priority_min": args.priority_min,
            "priority_selected_steps": priority_selected,
            "max_steps": args.max_steps,
            "per_split_limit": args.per_split_limit,
            "chunk_size": args.chunk_size,
            "elapsed_s": round(sum(float(chunk["elapsed_s"]) for chunk in chunks), 4),
            "chunks": chunks,
            "image": {
                "chunk_count": len(chunks),
                "max_used_image_bytes": max((chunk["image"]["used_image_bytes"] for chunk in chunks), default=0),
                "max_second_stage_bytes": max((chunk["image"]["second_stage_bytes"] for chunk in chunks), default=0),
                "max_second_stage_sectors": max((chunk["image"]["second_stage_sectors"] for chunk in chunks), default=0),
                "floppy_image_bytes": FLOPPY_IMAGE_BYTES,
            },
            "cmd": last_run.get("cmd", []),
            "returncode": last_run.get("returncode"),
            "timed_out": any_timed_out,
            "stopped_after_result": all_stopped_after_result,
            "serial_tail": str(last_run.get("serial", ""))[-4000:],
            "stdout_tail": str(last_run.get("stdout", ""))[-4000:],
            "stderr_tail": str(last_run.get("stderr", ""))[-4000:],
            "normalization": (
                "Stage 3 executes each complex-scalar instruction at its trace before_rip in x86-64 long mode. "
                "The v21 CPUID-gated scalar and memory-operand sample runs QEMU with an explicit CPU model and Bochs with its configured CPU model so MOVSX/MOVZX r8/r16/m8/m16, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR r64/r64 and r64/m64, TEST r64/r64 and m64/r64, BSF/BSR/POPCNT/LZCNT/TZCNT, selected BMI1/BMI2, BT/BTS/BTR/BTC m64 imm8/r64, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 encodings take their intended architectural paths. "
                "A generated trampoline at fall-through captures non-faulting RAX/RBX/RCX/RDX/RIP/RFLAGS; "
                "a vector-0 IDT handler captures DIV/IDIV #DE fault state at the saved RIP. "
                f"Memory-operand cases copy the trace memory image to {hex(MEM_BASE)} before execution and compare the expected after-memory image. "
                "Only deterministic flag bits declared in pan_stage3.flag_mask are compared."
            ),
        }
        write_json(report_dir / f"{args.engine}_stage3_oracle_report.json", report)
        write_json(args.run_dir / f"{args.engine}_stage3_oracle_report.json", report)
        return report
    finally:
        monitor.set_phase("finalize")
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", choices=["qemu", "bochs"], default="qemu")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "long"], default=["val", "long"])
    p.add_argument("--max-steps", type=int, default=768)
    p.add_argument("--per-split-limit", type=int, default=384)
    p.add_argument("--priority-category", default=None)
    p.add_argument("--priority-operands", nargs="*", default=None)
    p.add_argument("--priority-iclasses", nargs="*", default=None)
    p.add_argument("--priority-min", type=int, default=0)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=60.0)
    p.add_argument("--progress-interval", type=int, default=96)
    p.add_argument("--chunk-size", type=int, default=384)
    p.add_argument("--qemu-binary", default="qemu-system-x86_64")
    p.add_argument("--qemu-cpu", default="max")
    p.add_argument("--bochs-binary", default="bochs")
    p.add_argument("--bochs-cpu-model", default="corei7_haswell_4770")
    p.add_argument("--bochs-ips", type=int, default=50_000_000)
    p.add_argument("--bochs-memory-mib", type=int, default=64)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_oracle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
