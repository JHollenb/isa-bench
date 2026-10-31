#!/usr/bin/env python3
"""Validate Stage 0 traces with a QEMU real-mode boot-image oracle."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, FLAGS, ResourceMonitor, load_traces, write_json


DEFAULT_RUN_DIR = (
    DEFAULT_ARTIFACT_ROOT
    / DEFAULT_DATE
    / "pan-x86-isa"
    / "stage0_x86_lite_alu"
    / "neural-vector-heldout-w8-unicorn-selector"
)
ORIGIN = 0x0000
LOAD_SEGMENT = 0x1000
STACK_TOP = 0xFFFE
SERIAL_DATA = 0x03F8
SERIAL_LINE_STATUS = SERIAL_DATA + 5
DEBUG_EXIT_PORT = 0x00F4
FLOPPY_IMAGE_BYTES = 1_474_560
MAX_FLOPPY_STAGE_SECTORS = (FLOPPY_IMAGE_BYTES // 512) - 1
FLAG_BITS = {"CF": 0, "ZF": 6, "SF": 7, "OF": 11}
TRACKED_FLAG_MASK = sum(1 << FLAG_BITS[name] for name in FLAGS)


def text_tail(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class Code16:
    def __init__(self, origin: int):
        self.origin = origin
        self.buf = bytearray()
        self.labels: Dict[str, int] = {}
        self.fixups: List[Tuple[str, int, str]] = []

    @property
    def offset(self) -> int:
        return len(self.buf)

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

    def mov_al_imm(self, value: int) -> None:
        self.emit(0xB0, value)

    def mov_dx_imm(self, value: int) -> None:
        self.emit(0xBA)
        self.u16(value)

    def mov_si_label(self, label: str) -> None:
        self.emit(0xBE)
        pos = self.offset
        self.u16(0)
        self.fixups.append(("abs16", pos, label))

    def call(self, label: str) -> None:
        self.emit(0xE8)
        pos = self.offset
        self.u16(0)
        self.fixups.append(("rel16", pos, label))

    def call_abs(self, label: str) -> None:
        self.emit(0xBB)
        pos = self.offset
        self.u16(0)
        self.fixups.append(("abs16", pos, label))
        self.emit(0xFF, 0xD3)

    def jmp(self, label: str) -> None:
        self.emit(0xE9)
        pos = self.offset
        self.u16(0)
        self.fixups.append(("rel16", pos, label))

    def jmp_abs(self, label: str) -> None:
        self.emit(0xBB)
        pos = self.offset
        self.u16(0)
        self.fixups.append(("abs16", pos, label))
        self.emit(0xFF, 0xE3)

    def jne_short(self, label: str) -> None:
        self.emit(0x75)
        pos = self.offset
        self.emit(0)
        self.fixups.append(("rel8", pos, label))

    def jz_short(self, label: str) -> None:
        self.emit(0x74)
        pos = self.offset
        self.emit(0)
        self.fixups.append(("rel8", pos, label))

    def string(self, text: str) -> None:
        self.emit_bytes(text.encode("ascii") + b"\x00")

    def resolve(self) -> bytes:
        for kind, pos, label in self.fixups:
            if label not in self.labels:
                raise ValueError(f"unknown label {label}")
            target = self.labels[label]
            if kind == "abs16":
                value = self.origin + target
                if not 0 <= value <= 0xFFFF:
                    raise ValueError(f"absolute label out of 16-bit range: {label}=0x{value:x}")
                self.buf[pos] = value & 0xFF
                self.buf[pos + 1] = (value >> 8) & 0xFF
            elif kind == "rel16":
                rel = target - (pos + 2)
                if not -32768 <= rel <= 32767:
                    raise ValueError(f"rel16 out of range for {label}: {rel}")
                self.buf[pos] = rel & 0xFF
                self.buf[pos + 1] = (rel >> 8) & 0xFF
            elif kind == "rel8":
                rel = target - (pos + 1)
                if not -128 <= rel <= 127:
                    raise ValueError(f"rel8 out of range for {label}: {rel}")
                self.buf[pos] = rel & 0xFF
            else:
                raise ValueError(kind)
        if self.origin + len(self.buf) > 0x10000:
            raise ValueError(f"second stage crosses 16-bit segment boundary: {len(self.buf)} bytes")
        return bytes(self.buf)


def flags_word(flags: Sequence[int]) -> int:
    word = 0x0002
    for value, name in zip(flags, FLAGS):
        if int(value):
            word |= 1 << FLAG_BITS[name]
    return word & TRACKED_FLAG_MASK


def collect_steps(
    run_dir: Path,
    splits: Sequence[str],
    max_steps: int,
    per_split_limit: Optional[int] = None,
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


def emit_serial_init(code: Code16) -> None:
    for port, value in (
        (SERIAL_DATA + 1, 0x00),
        (SERIAL_DATA + 3, 0x80),
        (SERIAL_DATA + 0, 0x03),
        (SERIAL_DATA + 1, 0x00),
        (SERIAL_DATA + 3, 0x03),
        (SERIAL_DATA + 2, 0xC7),
        (SERIAL_DATA + 4, 0x0B),
    ):
        code.mov_dx_imm(port)
        code.mov_al_imm(value)
        code.emit(0xEE)


def emit_exit(code: Code16, status: int) -> None:
    code.mov_al_imm(status)
    code.mov_dx_imm(DEBUG_EXIT_PORT)
    code.emit(0xEE)
    code.label(f"hang_{status}_{code.offset}")
    code.emit(0xF4)
    code.jmp(f"hang_{status}_{code.offset - 1}")


def emit_puts_routines(code: Code16) -> None:
    code.label("putc")
    code.emit(0x50)
    code.mov_dx_imm(SERIAL_LINE_STATUS)
    code.label("putc_wait")
    code.emit(0xEC)
    code.emit(0xA8, 0x20)
    code.jz_short("putc_wait")
    code.mov_dx_imm(SERIAL_DATA)
    code.emit(0x58)
    code.emit(0xEE)
    code.emit(0xC3)

    code.label("puts")
    code.label("puts_loop")
    code.emit(0xAC)
    code.emit(0x84, 0xC0)
    code.jz_short("puts_done")
    code.call("putc")
    code.jmp("puts_loop")
    code.label("puts_done")
    code.emit(0xC3)


def emit_case(code: Code16, index: int, item: Dict[str, Any]) -> None:
    p = item["step"]["pan_stage0"]
    before_regs = [int(x) & 0xFF for x in p["before_regs"]]
    after_regs = [int(x) & 0xFF for x in p["after_regs"]]
    before_flags = flags_word(p["before_flags"])
    after_flags = flags_word(p["after_flags"])
    fail = f"fail_{index}"
    next_case = f"case_{index + 1}"

    code.emit(0x68)
    code.u16(before_flags)
    code.emit(0x9D)
    for opcode, value in zip((0xB0, 0xB1, 0xB2, 0xB3), before_regs):
        code.emit(opcode, value)
    code.emit_bytes(bytes.fromhex(item["bytes_hex"]))
    code.emit(0x9C, 0x5E)

    for reg_opcode, expected in ((None, after_regs[0]), (0xF9, after_regs[1]), (0xFA, after_regs[2]), (0xFB, after_regs[3])):
        if reg_opcode is None:
            code.emit(0x3C, expected)
        else:
            code.emit(0x80, reg_opcode, expected)
        code.jne_short(fail)

    code.emit(0x89, 0xF0)
    code.emit(0x25)
    code.u16(TRACKED_FLAG_MASK)
    code.emit(0x3D)
    code.u16(after_flags)
    code.jne_short(fail)
    code.jmp(next_case)

    code.label(fail)
    code.mov_si_label(f"fail_msg_{index}")
    code.call_abs("puts")
    code.jmp_abs("final_fail")
    code.label(next_case)


def emit_progress(code: Code16, label: str) -> None:
    code.mov_si_label(label)
    code.call_abs("puts")


def emit_segment_prologue(code: Code16) -> None:
    code.emit(0x0E, 0x1F)
    code.emit(0x0E, 0x17)
    code.emit(0xBC)
    code.u16(STACK_TOP)


def generate_second_stage(selected: Sequence[Dict[str, Any]], progress_interval: int = 0) -> bytes:
    code = Code16(ORIGIN)
    emit_segment_prologue(code)
    emit_serial_init(code)
    if progress_interval > 0:
        emit_progress(code, "progress_begin")
    for index, item in enumerate(selected):
        if progress_interval > 0 and index > 0 and index % progress_interval == 0:
            emit_progress(code, f"progress_{index}")
        emit_case(code, index, item)

    code.mov_si_label("success_msg")
    code.call_abs("puts")
    emit_exit(code, 0)

    code.label("final_fail")
    code.mov_si_label("failure_result_msg")
    code.call_abs("puts")
    emit_exit(code, 1)

    emit_puts_routines(code)

    code.label("success_msg")
    code.string(f"RESULT {len(selected)} {len(selected)} 0\n")
    code.label("failure_result_msg")
    code.string(f"RESULT {len(selected)} 0 1\n")
    if progress_interval > 0:
        code.label("progress_begin")
        code.string(f"BEGIN {len(selected)}\n")
        for index in range(progress_interval, len(selected), progress_interval):
            code.label(f"progress_{index}")
            code.string(f"PROGRESS {index}\n")
    for index in range(len(selected)):
        code.label(f"fail_msg_{index}")
        code.string(f"FAIL {index}\n")
    return code.resolve()


def generate_boot_sector(second_stage_sectors: int, load_addr: int = ORIGIN, load_segment: int = 0) -> bytes:
    if not 1 <= second_stage_sectors <= MAX_FLOPPY_STAGE_SECTORS:
        raise ValueError(f"unsupported sector count: {second_stage_sectors}")
    if not 0 <= load_addr <= 0xFFFF:
        raise ValueError(f"unsupported real-mode load address: 0x{load_addr:x}")
    if not 0 <= load_segment <= 0xFFFF:
        raise ValueError(f"unsupported real-mode load segment: 0x{load_segment:x}")
    stack_addr = load_addr if load_addr >= 0x0600 else 0x7C00
    code = bytearray()
    e = code.extend
    e(b"\xFA\x31\xC0\x8E\xD8\x8E\xD0\xBC")
    code.extend((stack_addr & 0xFF, (stack_addr >> 8) & 0xFF))
    e(b"\xB8")
    code.extend((load_segment & 0xFF, (load_segment >> 8) & 0xFF))
    e(b"\x8E\xC0")
    e(b"\x88\x16\x00\x7C")
    boot_drive_patch = len(code) - 2
    e(b"\xBE")
    code.extend((second_stage_sectors & 0xFF, (second_stage_sectors >> 8) & 0xFF))
    e(b"\xBB")
    code.extend((load_addr & 0xFF, (load_addr >> 8) & 0xFF))
    e(b"\xB1\x02\xB6\x00\xB5\x00")
    loop = len(code)
    e(b"\x83\xFE\x00")
    e(b"\x74\x00")
    done_jz_patch = len(code) - 1
    e(b"\xB4\x02\xB0\x01\x8A\x16\x00\x7C")
    drive_load_patch = len(code) - 2
    e(b"\xCD\x13")
    e(b"\x72\x00")
    error_jc_patch = len(code) - 1
    e(b"\x81\xC3\x00\x02")
    e(b"\x73\x00")
    no_es_bump_patch = len(code) - 1
    e(b"\x8C\xC0\x05\x00\x10\x8E\xC0")
    no_es_bump = len(code)
    e(b"\xFE\xC1\x4E\x80\xF9\x13")
    e(b"\x75\x00")
    no_track_patch = len(code) - 1
    e(b"\xB1\x01\xFE\xC6\x80\xFE\x02")
    e(b"\x75\x00")
    no_cyl_patch = len(code) - 1
    e(b"\xB6\x00\xFE\xC5")
    no_cyl = len(code)
    no_track = len(code)
    rel = loop - (len(code) + 2)
    e(b"\xEB" + bytes([rel & 0xFF]))
    done = len(code)
    e(b"\xEA")
    code.extend((load_addr & 0xFF, (load_addr >> 8) & 0xFF))
    code.extend((load_segment & 0xFF, (load_segment >> 8) & 0xFF))
    error = len(code)
    e(b"\xB0\x45\xBA\xF8\x03\xEE\xB0\x02\xBA\xF4\x00\xEE\xF4")

    boot_drive = 0x7C00 + len(code)
    code.append(0)
    for patch in (boot_drive_patch, drive_load_patch):
        code[patch] = boot_drive & 0xFF
        code[patch + 1] = (boot_drive >> 8) & 0xFF

    code[done_jz_patch] = (done - (done_jz_patch + 1)) & 0xFF
    code[error_jc_patch] = (error - (error_jc_patch + 1)) & 0xFF
    code[no_es_bump_patch] = (no_es_bump - (no_es_bump_patch + 1)) & 0xFF
    code[no_track_patch] = (no_track - (no_track_patch + 1)) & 0xFF
    code[no_cyl_patch] = (no_cyl - (no_cyl_patch + 1)) & 0xFF
    if len(code) > 510:
        raise ValueError(f"boot sector too large: {len(code)} bytes")
    code.extend(b"\x00" * (510 - len(code)))
    code.extend(b"\x55\xAA")
    return bytes(code)


def build_image(selected: Sequence[Dict[str, Any]], image_path: Path, progress_interval: int = 0) -> Dict[str, Any]:
    second_stage = generate_second_stage(selected, progress_interval=progress_interval)
    sectors = math.ceil(len(second_stage) / 512)
    boot = generate_boot_sector(sectors, load_addr=ORIGIN, load_segment=LOAD_SEGMENT)
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
        "load_segment": LOAD_SEGMENT,
        "load_offset": ORIGIN,
        "second_stage_bytes": len(second_stage),
        "second_stage_sectors": sectors,
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


def run_qemu_until_result(cmd: Sequence[str], serial_path: Path, timeout_s: float) -> Dict[str, Any]:
    started = time.time()
    proc = subprocess.Popen(list(cmd), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout = ""
    stderr = ""
    serial_text = ""
    stopped_after_result = False
    timed_out = False
    try:
        while True:
            if serial_path.exists():
                serial_text = serial_path.read_text(encoding="utf-8", errors="replace")
                if "RESULT " in serial_text:
                    stopped_after_result = True
                    proc.terminate()
                    break
            if proc.poll() is not None:
                break
            if time.time() - started > timeout_s:
                timed_out = True
                proc.kill()
                break
            time.sleep(0.05)
        try:
            out, err = proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, err = proc.communicate(timeout=2.0)
        stdout = text_tail(out)
        stderr = text_tail(err)
    finally:
        if proc.poll() is None:
            proc.kill()
    if serial_path.exists():
        serial_text = serial_path.read_text(encoding="utf-8", errors="replace")
    return {
        "elapsed_s": round(time.time() - started, 4),
        "returncode": proc.returncode,
        "timed_out": timed_out,
        "stopped_after_result": stopped_after_result,
        "stdout": stdout,
        "stderr": stderr,
        "serial": serial_text,
    }


def run_qemu_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = args.report_dir or (args.run_dir / "qemu_stage0_oracle")
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(report_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("collect")
        selected, source_counts = collect_steps(args.run_dir, args.splits, args.max_steps, args.per_split_limit)
        if not selected:
            raise ValueError("no Stage 0 steps selected")

        qemu = shutil.which(args.qemu_binary)
        availability = {
            "qemu_binary": args.qemu_binary,
            "qemu_path": qemu,
            "qemu_available": qemu is not None,
        }
        if qemu is None:
            report = {
                "enabled": True,
                "oracle": "qemu_system_i386_real_mode",
                "ok": False,
                "error": f"{args.qemu_binary} not found",
                "availability": availability,
            }
            write_json(report_dir / "qemu_stage0_oracle_report.json", report)
            write_json(args.run_dir / "qemu_stage0_oracle_report.json", report)
            return report

        monitor.set_phase("build_image")
        image_info = build_image(selected, report_dir / "stage0_qemu_oracle.img", args.progress_interval)
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
            "oracle": "qemu_system_i386_real_mode",
            "ok": ok,
            "stage": "stage0_x86_lite_alu",
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
            "normalization": "Stage 0 executes low-byte AL/CL/DL/BL instruction bytes in QEMU real mode; only CF/ZF/SF/OF and low bytes are compared.",
        }
        write_json(report_dir / "qemu_stage0_oracle_report.json", report)
        write_json(args.run_dir / "qemu_stage0_oracle_report.json", report)
        return report
    finally:
        monitor.set_phase("finalize")
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "long"], default=["val", "long"])
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--per-split-limit", type=int, default=None)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=30.0)
    p.add_argument("--qemu-binary", default="qemu-system-i386")
    p.add_argument("--progress-interval", type=int, default=0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_qemu_oracle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
