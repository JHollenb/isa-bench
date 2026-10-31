#!/usr/bin/env python3
"""Validate Stage 2 traces with a QEMU long-mode boot-image oracle."""
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, FLAGS, ResourceMonitor, load_traces, write_json
from pan_x86_stages.stage2_control_flow_pan import CODE_BASE, STACK_BASE, STACK_SIZE
from pan_x86_stages.validate_stage0_qemu import FLOPPY_IMAGE_BYTES, MAX_FLOPPY_STAGE_SECTORS, generate_boot_sector, run_qemu_until_result
from pan_x86_stages.validate_stage1_qemu import Code, DEBUG_EXIT_PORT, ORIGIN, emit_enter_long_mode, emit_memcheck64, emit_mov_reg_imm64, emit_puts64


DEFAULT_RUN_DIR = (
    DEFAULT_ARTIFACT_ROOT
    / DEFAULT_DATE
    / "pan-x86-isa"
    / "stage2_control_flow"
    / "stage2-control-neural-unicorn-v3"
)
CODE_CLEAR_BYTES = 0x10000
HARNESS_STACK_ADDR = 0x128000
RESULT_RAX = 0x130000
RESULT_RSP = 0x130008
RESULT_RIP = 0x130010
RESULT_FLAGS = 0x130018
FLAG_BITS = {"CF": 0, "ZF": 6, "SF": 7, "OF": 11}
TRACKED_FLAG_MASK = sum(1 << FLAG_BITS[name] for name in FLAGS)
TRAMPOLINE_BYTES = 77
MAX_STAGE_SECTORS = MAX_FLOPPY_STAGE_SECTORS


def flags_word(flags: Sequence[int]) -> int:
    word = 0x0002
    for value, name in zip(flags, FLAGS):
        if int(value):
            word |= 1 << FLAG_BITS[name]
    return word & 0xFFFFFFFF


def collect_steps(
    run_dir: Path,
    splits: Sequence[str],
    max_steps: int,
    per_split_limit: Optional[int],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    skipped: Dict[str, int] = {}
    for split in splits:
        checked = 0
        skipped_split = 0
        stop_split = False
        for trace in load_traces(run_dir / f"{split}_traces.jsonl"):
            for step in trace["steps"]:
                if len(selected) >= max_steps:
                    counts[split] = checked
                    skipped[split] = skipped_split
                    return selected, counts, skipped
                if per_split_limit is not None and checked >= per_split_limit:
                    stop_split = True
                    break
                if not is_safe_step(step):
                    skipped_split += 1
                    continue
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
        skipped[split] = skipped_split
    return selected, counts, skipped


def is_safe_step(step: Dict[str, Any]) -> bool:
    p = step["pan_stage2"]
    before_rip = int(p["before_rip"])
    after_rip = int(p["after_rip"])
    insn_len = len(bytes.fromhex(step["decode"]["bytes_hex"]))
    insn_start = before_rip
    insn_end = before_rip + insn_len
    tramp_start = after_rip
    tramp_end = after_rip + TRAMPOLINE_BYTES
    code_start = CODE_BASE
    code_end = CODE_BASE + CODE_CLEAR_BYTES
    if not (code_start <= insn_start and insn_end <= code_end):
        return False
    if not (code_start <= tramp_start and tramp_end <= code_end):
        return False
    return tramp_end <= insn_start or insn_end <= tramp_start


def emit_mov_rsp_imm64(code: Code, value: int) -> None:
    code.emit(0x48, 0xBC)
    code.u64(value & 0xFFFFFFFFFFFFFFFF)


def emit_mov_r11_imm64(code: Code, value: int) -> None:
    code.emit(0x49, 0xBB)
    code.u64(value & 0xFFFFFFFFFFFFFFFF)


def emit_jmp_r11(code: Code) -> None:
    code.emit(0x41, 0xFF, 0xE3)


def emit_pushfq_from_word(code: Code, value: int) -> None:
    code.emit(0x68)
    code.u32(value & 0xFFFFFFFF)
    code.emit(0x9D)


def emit_copy_label_to_addr(code: Code, label: str, addr: int, size: int) -> None:
    emit_mov_reg_imm64(code, "rsi", 0)
    code.fixups.append(("abs64", code.offset - 8, label))
    emit_mov_reg_imm64(code, "rdi", addr)
    code.emit(0xB9)
    code.u32(size)
    code.emit(0xFC, 0xF3, 0xA4)


def emit_memset64(code: Code, addr: int, value: int, size: int) -> None:
    emit_mov_reg_imm64(code, "rdi", addr)
    code.emit(0xB0, value & 0xFF)
    code.emit(0xB9)
    code.u32(size)
    code.emit(0xFC, 0xF3, 0xAA)


def emit_load_moffs64_rax(code: Code, addr: int) -> None:
    code.emit(0x48, 0xA1)
    code.u64(addr)


def emit_store_rax_moffs64(code: Code, addr: int) -> None:
    code.emit(0x48, 0xA3)
    code.u64(addr)


def emit_compare_rax_imm64(code: Code, value: int, fail_label: str) -> None:
    emit_mov_reg_imm64(code, "r8", value)
    code.emit(0x4C, 0x39, 0xC0)
    code.jne32(fail_label)


def emit_compare_result64(code: Code, addr: int, expected: int, fail_label: str) -> None:
    emit_load_moffs64_rax(code, addr)
    emit_compare_rax_imm64(code, expected & 0xFFFFFFFFFFFFFFFF, fail_label)


def emit_compare_flags(code: Code, expected_flags: Sequence[int], fail_label: str) -> None:
    emit_load_moffs64_rax(code, RESULT_FLAGS)
    code.emit(0x48, 0x25)
    code.u32(TRACKED_FLAG_MASK)
    emit_compare_rax_imm64(code, flags_word(expected_flags) & TRACKED_FLAG_MASK, fail_label)


def emit_stack_compare(code: Code, after_stack_label: str, fail_label: str) -> None:
    emit_mov_reg_imm64(code, "rsi", STACK_BASE)
    emit_mov_reg_imm64(code, "rdi", 0)
    code.fixups.append(("abs64", code.offset - 8, after_stack_label))
    code.emit(0xB9)
    code.u32(STACK_SIZE)
    code.emit(0xBA)
    code.u32(0xFFFFFFFF)
    code.call32("memcheck64")
    code.emit(0x85, 0xC0)
    code.jne32(fail_label)


def emit_case64(code: Code, index: int, item: Dict[str, Any]) -> None:
    p = item["step"]["pan_stage2"]
    fail_labels = {
        "RAX": f"fail_{index}_rax",
        "RSP": f"fail_{index}_rsp",
        "RIP": f"fail_{index}_rip",
        "FLAGS": f"fail_{index}_flags",
        "STACK": f"fail_{index}_stack",
    }
    cont = f"case_{index}_after"
    next_case = f"case_{index + 1}"
    before_stack_label = f"before_stack_{index}"
    after_stack_label = f"after_stack_{index}"
    insn_label = f"instruction_{index}"
    trampoline_label = f"trampoline_{index}"

    emit_memset64(code, CODE_BASE, 0xF4, CODE_CLEAR_BYTES)
    emit_copy_label_to_addr(code, before_stack_label, STACK_BASE, STACK_SIZE)
    emit_copy_label_to_addr(code, insn_label, int(p["before_rip"]), len(bytes.fromhex(item["bytes_hex"])))
    emit_copy_label_to_addr(code, trampoline_label, int(p["after_rip"]), TRAMPOLINE_BYTES)

    emit_mov_reg_imm64(code, "rax", int(p["before_rax"]))
    emit_mov_rsp_imm64(code, HARNESS_STACK_ADDR)
    emit_pushfq_from_word(code, flags_word(p["before_flags"]))
    emit_mov_rsp_imm64(code, STACK_BASE + int(p["before_rsp_offset"]))
    emit_mov_r11_imm64(code, int(p["before_rip"]))
    emit_jmp_r11(code)

    code.label(cont)
    emit_compare_result64(code, RESULT_RAX, int(p["after_rax"]), fail_labels["RAX"])
    emit_compare_result64(code, RESULT_RSP, STACK_BASE + int(p["after_rsp_offset"]), fail_labels["RSP"])
    emit_compare_result64(code, RESULT_RIP, int(p["after_rip"]), fail_labels["RIP"])
    emit_compare_flags(code, p["after_flags"], fail_labels["FLAGS"])
    emit_stack_compare(code, after_stack_label, fail_labels["STACK"])
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


def emit_exit_hang64(code: Code, label: str) -> None:
    code.emit(0xB0, 0x00, 0x66, 0xBA)
    code.u16(DEBUG_EXIT_PORT)
    code.emit(0xEE)
    code.label(label)
    code.emit(0xF4)
    code.jmp32(label)


def emit_trampoline_blob(code: Code, label: str, continuation_label: str, actual_rip: int) -> None:
    code.label(label)
    start = code.offset
    emit_store_rax_moffs64(code, RESULT_RAX)
    code.emit(0x48, 0x89, 0xE0)
    emit_store_rax_moffs64(code, RESULT_RSP)
    emit_mov_reg_imm64(code, "rax", actual_rip)
    emit_store_rax_moffs64(code, RESULT_RIP)
    emit_mov_rsp_imm64(code, HARNESS_STACK_ADDR)
    code.emit(0x9C, 0x58)
    emit_store_rax_moffs64(code, RESULT_FLAGS)
    code.emit(0x48, 0xB8)
    code.fix("abs64", continuation_label)
    code.emit(0xFF, 0xE0)
    size = code.offset - start
    if size != TRAMPOLINE_BYTES:
        raise ValueError(f"unexpected trampoline size: {size}")


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
    for index, item in enumerate(selected):
        p = item["step"]["pan_stage2"]
        for reason in ("RAX", "RSP", "RIP", "FLAGS", "STACK"):
            code.label(f"fail_msg_{index}_{reason}")
            code.emit_bytes(f"FAIL {index} {reason}\n".encode("ascii") + b"\x00")
        code.label(f"instruction_{index}")
        code.emit_bytes(bytes.fromhex(item["bytes_hex"]))
        code.label(f"trampoline_{index}")
        emit_trampoline_blob(code, f"trampoline_blob_{index}", f"case_{index}_after", int(p["after_rip"]))
        code.label(f"before_stack_{index}")
        before_stack = bytes(int(x) & 0xFF for x in p["before_stack"])
        if len(before_stack) != STACK_SIZE:
            raise ValueError(f"unexpected before stack size for case {index}: {len(before_stack)}")
        code.emit_bytes(before_stack)
        code.label(f"after_stack_{index}")
        after_stack = bytes(int(x) & 0xFF for x in p["after_stack"])
        if len(after_stack) != STACK_SIZE:
            raise ValueError(f"unexpected after stack size for case {index}: {len(after_stack)}")
        code.emit_bytes(after_stack)
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


def parse_qemu_output(stdout: str, selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
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


def run_qemu_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = args.report_dir or (args.run_dir / "qemu_stage2_oracle")
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(report_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("collect")
        selected, source_counts, skipped_counts = collect_steps(args.run_dir, args.splits, args.max_steps, args.per_split_limit)
        if not selected:
            raise ValueError("no Stage 2 steps selected")

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
            write_json(report_dir / "qemu_stage2_oracle_report.json", report)
            write_json(args.run_dir / "qemu_stage2_oracle_report.json", report)
            return report

        chunks = []
        checked_steps = 0
        exact_steps = 0
        failure_count = 0
        failure_examples: List[Dict[str, Any]] = []
        all_stopped_after_result = True
        any_timed_out = False
        last_cmd: List[str] = []
        last_returncode: Optional[int] = None
        last_stdout = ""
        last_stderr = ""
        last_serial = ""
        for chunk_index, start in enumerate(range(0, len(selected), args.chunk_size)):
            chunk_items = [dict(item, global_index=start + local_index) for local_index, item in enumerate(selected[start:start + args.chunk_size])]
            monitor.set_phase(f"build_image_{chunk_index}")
            image_info = build_image(chunk_items, report_dir / f"stage2_qemu_oracle_{chunk_index:03d}.img", args.progress_interval)
            serial_path = report_dir / f"qemu_serial_{chunk_index:03d}.log"
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
            monitor.set_phase(f"execute_{chunk_index}")
            run = run_qemu_until_result(cmd, serial_path, args.timeout_s)
            parsed = parse_qemu_output(run["serial"], chunk_items)
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
            last_cmd = cmd
            last_returncode = run["returncode"]
            last_stdout = run["stdout"]
            last_stderr = run["stderr"]
            last_serial = run["serial"]
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
                    "serial_tail": run["serial"][-2000:],
                    "stderr_tail": run["stderr"][-2000:],
                    "image": image_info,
                }
            )
            if not chunk_ok:
                break

        ok = (
            checked_steps == len(selected)
            and exact_steps == len(selected)
            and failure_count == 0
            and all_stopped_after_result
        )
        split_counts = {
            split: {"checked_steps": sum(1 for item in selected if item["split"] == split)}
            for split in args.splits
        }
        report = {
            "enabled": True,
            "oracle": "qemu_system_x86_64_long_mode",
            "ok": ok,
            "stage": "stage2_control_flow",
            "run_dir": str(args.run_dir),
            "report_dir": str(report_dir),
            "checked_steps": checked_steps,
            "exact_steps": exact_steps,
            "exact": round(exact_steps / max(1, checked_steps), 6),
            "failure_count": failure_count,
            "failure_examples": failure_examples,
            "splits": split_counts,
            "requested_splits": list(args.splits),
            "source_trace_counts": source_counts,
            "skipped_unsafe_steps": skipped_counts,
            "max_steps": args.max_steps,
            "per_split_limit": args.per_split_limit,
            "chunk_size": args.chunk_size,
            "elapsed_s": round(sum(float(chunk["elapsed_s"]) for chunk in chunks), 4),
            "availability": availability,
            "chunks": chunks,
            "image": {
                "chunk_count": len(chunks),
                "max_used_image_bytes": max((chunk["image"]["used_image_bytes"] for chunk in chunks), default=0),
                "max_second_stage_bytes": max((chunk["image"]["second_stage_bytes"] for chunk in chunks), default=0),
                "max_second_stage_sectors": max((chunk["image"]["second_stage_sectors"] for chunk in chunks), default=0),
                "floppy_image_bytes": FLOPPY_IMAGE_BYTES,
            },
            "cmd": last_cmd,
            "returncode": last_returncode,
            "timed_out": any_timed_out,
            "stopped_after_result": all_stopped_after_result,
            "serial_tail": last_serial[-4000:],
            "stdout_tail": last_stdout[-4000:],
            "stderr_tail": last_stderr[-4000:],
            "normalization": (
                "Stage 2 executes each x86-64 instruction at its trace before_rip in QEMU long mode. "
                "A generated trampoline at the expected after_rip captures RAX/RSP/RIP/RFLAGS before "
                "returning to the harness; the tracked 256-byte stack image is compared byte-for-byte."
            ),
        }
        write_json(report_dir / "qemu_stage2_oracle_report.json", report)
        write_json(args.run_dir / "qemu_stage2_oracle_report.json", report)
        return report
    finally:
        monitor.set_phase("finalize")
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "long"], default=["val", "long"])
    p.add_argument("--max-steps", type=int, default=768)
    p.add_argument("--per-split-limit", type=int, default=384)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=48.0)
    p.add_argument("--qemu-binary", default="qemu-system-x86_64")
    p.add_argument("--progress-interval", type=int, default=96)
    p.add_argument("--chunk-size", type=int, default=384)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_qemu_oracle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
