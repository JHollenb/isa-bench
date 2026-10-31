#!/usr/bin/env python3
"""Validate Stage 1 traces with an x86_64 Rosetta execution oracle.

The Stage 1 runners already validate against Unicorn. This script adds a second
execution path on Apple Silicon: generate a small x86_64 Mach-O oracle program,
run it through Rosetta, and compare register/memory effects against Stage 1
trace records. LEA results are normalized from the runtime mmap base back into
the trace memory base coordinate system.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pan_x86_stages.stage0_x86_pan import ResourceMonitor, load_traces, write_json
from pan_x86_stages.stage1_memory_pan import DATA_BYTES, DATA_REGS, MEM_BASE, MEM_SIZE, OPS, memory_digest


DATA_REG_ASM = ("rax", "rcx", "rdx", "rbx")
DATA_BYTE_ASM = ("al", "cl", "dl", "bl")
BASE_ASM = ("rsi", "rdi", "rbp")


def c_array_u8(values: Sequence[int]) -> str:
    return "{" + ",".join(str(int(v) & 0xFF) for v in values) + "}"


def c_array_u64(values: Sequence[int]) -> str:
    return "{" + ",".join(str(int(v) & 0xFFFFFFFFFFFFFFFF) + "ULL" for v in values) + "}"


def instruction_asm(step: Dict[str, Any]) -> str:
    p = step["pan_stage1"]
    op = OPS[int(p["op_index"])]
    dst = int(p["dst"])
    src = int(p["src"])
    base = int(p["base"])
    disp = int(p["disp"])
    mem = f"byte ptr [{BASE_ASM[base]} + {disp}]"
    if op.name == "MOV_LOAD_RM":
        return f"  mov {DATA_BYTE_ASM[dst]}, {mem}"
    if op.name == "MOV_STORE_MR":
        return f"  mov {mem}, {DATA_BYTE_ASM[src]}"
    if op.name == "MOV_STORE_MI":
        return f"  mov {mem}, {int(p['imm']) & 0xFF}"
    if op.name == "LEA_RA":
        return f"  lea {DATA_REG_ASM[dst]}, [{BASE_ASM[base]} + {disp}]"
    if op.name == "XCHG_RM":
        return f"  xchg {DATA_BYTE_ASM[dst]}, {mem}"
    raise ValueError(op.name)


def collect_steps(run_dir: Path, splits: Sequence[str], max_steps: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    selected: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for split in splits:
        path = run_dir / f"{split}_traces.jsonl"
        checked = 0
        for trace in load_traces(path):
            for step in trace["steps"]:
                if len(selected) >= max_steps:
                    counts[split] = checked
                    return selected, counts
                selected.append({
                    "split": split,
                    "trace_id": trace["trace_id"],
                    "step_index": step["index"],
                    "asm": trace["program"]["asm"][step["index"]],
                    "step": step,
                })
                checked += 1
        counts[split] = checked
    return selected, counts


def generate_c_source(selected: Sequence[Dict[str, Any]]) -> str:
    cases = []
    for item in selected:
        p = item["step"]["pan_stage1"]
        cases.append(
            "{"
            f"{int(p['op_index'])}ULL,"
            f"{int(p['dst'])}ULL,"
            f"{c_array_u64(p['before_data_regs'])},"
            f"{c_array_u64(p['before_addr_offsets'])},"
            f"{c_array_u8(p['before_memory'])},"
            f"{c_array_u64(p['after_data_regs'])},"
            f"{c_array_u8(p['after_memory'])}"
            "}"
        )
    func_decls = "\n".join(f"void case_{i}(struct Case*, struct Result*, uint64_t);" for i in range(len(selected)))
    func_table = ",".join(f"case_{i}" for i in range(len(selected)))
    return f"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define NCASES {len(selected)}
#define MEM_BASE {MEM_BASE}ULL
#define MEM_SIZE {MEM_SIZE}

struct Case {{
  uint64_t op_idx;
  uint64_t dst;
  uint64_t data[4];
  uint64_t addr[3];
  uint8_t mem[MEM_SIZE];
  uint64_t expected_data[4];
  uint8_t expected_mem[MEM_SIZE];
}};

struct Result {{ uint64_t regs[4]; }};
typedef void (*case_fn)(struct Case*, struct Result*, uint64_t);

{func_decls}

static case_fn funcs[NCASES] = {{{func_table}}};
static struct Case cases[NCASES] = {{
{",".join(cases)}
}};

int main(void) {{
  uint8_t* mapped = (uint8_t*)mmap(0, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
  if (mapped == (void*)-1) {{
    perror("mmap");
    return 2;
  }}
  int exact = 0;
  int failures = 0;
  for (int i = 0; i < NCASES; i++) {{
    struct Case c = cases[i];
    struct Result r;
    memset(&r, 0, sizeof(r));
    memcpy(mapped, c.mem, MEM_SIZE);
    funcs[i](&c, &r, (uint64_t)mapped);
    int reg_ok = 1;
    for (int j = 0; j < 4; j++) {{
      uint64_t got = r.regs[j];
      if (c.op_idx == 3 && c.dst == (uint64_t)j) {{
        got = MEM_BASE + (got - (uint64_t)mapped);
      }}
      if (got != c.expected_data[j]) {{
        reg_ok = 0;
      }}
    }}
    int mem_ok = memcmp(mapped, c.expected_mem, MEM_SIZE) == 0;
    if (reg_ok && mem_ok) {{
      exact++;
    }} else {{
      if (failures < 8) {{
        printf("FAIL %d %d %d\\n", i, reg_ok, mem_ok);
      }}
      failures++;
    }}
  }}
  printf("RESULT %d %d %d\\n", NCASES, exact, failures);
  return failures == 0 ? 0 : 1;
}}
"""


def generate_asm_source(selected: Sequence[Dict[str, Any]]) -> str:
    funcs = []
    for i, item in enumerate(selected):
        instr = instruction_asm(item["step"])
        funcs.append(f"""
.globl _case_{i}
_case_{i}:
  push rbp
  push rbx
  push r12
  mov r12, rdx
  mov r11, rdi
  mov r10, rsi
  mov rax, qword ptr [r11 + 16]
  mov rcx, qword ptr [r11 + 24]
  mov rdx, qword ptr [r11 + 32]
  mov rbx, qword ptr [r11 + 40]
  mov r9, qword ptr [r11 + 48]
  lea rsi, [r12 + r9]
  mov r9, qword ptr [r11 + 56]
  lea rdi, [r12 + r9]
  mov r9, qword ptr [r11 + 64]
  lea rbp, [r12 + r9]
{instr}
  mov qword ptr [r10 + 0], rax
  mov qword ptr [r10 + 8], rcx
  mov qword ptr [r10 + 16], rdx
  mov qword ptr [r10 + 24], rbx
  pop r12
  pop rbx
  pop rbp
  ret
""")
    return ".intel_syntax noprefix\n.text\n" + "\n".join(funcs)


def parse_output(stdout: str, selected: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    failures = []
    checked = 0
    exact = 0
    failure_count = 0
    for line in stdout.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if parts[0] == "FAIL" and len(parts) == 4:
            idx = int(parts[1])
            if 0 <= idx < len(selected):
                item = selected[idx]
                failures.append({
                    "index": idx,
                    "split": item["split"],
                    "trace_id": item["trace_id"],
                    "step": item["step_index"],
                    "asm": item["asm"],
                    "reg_ok": bool(int(parts[2])),
                    "memory_ok": bool(int(parts[3])),
                })
        elif parts[0] == "RESULT" and len(parts) == 4:
            checked = int(parts[1])
            exact = int(parts[2])
            failure_count = int(parts[3])
    return {
        "checked_steps": checked,
        "exact_steps": exact,
        "exact": round(exact / max(1, checked), 6),
        "failure_count": failure_count,
        "failure_examples": failures,
    }


def run_rosetta_oracle(args: argparse.Namespace) -> Dict[str, Any]:
    report_dir = args.report_dir or (args.run_dir / "rosetta_oracle")
    report_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(report_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("collect")
        selected, split_counts = collect_steps(args.run_dir, args.splits, args.max_steps)
        build_dir = report_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        c_path = build_dir / "rosetta_oracle.c"
        s_path = build_dir / "rosetta_oracle.s"
        exe_path = build_dir / "rosetta_oracle"
        c_path.write_text(generate_c_source(selected), encoding="utf-8")
        s_path.write_text(generate_asm_source(selected), encoding="utf-8")

        clang = shutil.which("clang")
        available = {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "clang": clang,
            "arch": shutil.which("arch"),
            "can_run_arch_x86_64": False,
        }
        if clang is None:
            report = {
                "enabled": True,
                "oracle": "rosetta_x86_64",
                "ok": False,
                "error": "clang not found",
                "availability": available,
            }
            write_json(report_dir / "rosetta_oracle_report.json", report)
            write_json(args.run_dir / "rosetta_oracle_report.json", report)
            return report
        arch_probe = subprocess.run(["/usr/bin/arch", "-x86_64", "/usr/bin/true"], text=True, capture_output=True)
        available["can_run_arch_x86_64"] = arch_probe.returncode == 0
        if not available["can_run_arch_x86_64"]:
            report = {
                "enabled": True,
                "oracle": "rosetta_x86_64",
                "ok": False,
                "error": "arch -x86_64 failed; Rosetta unavailable",
                "availability": available,
                "stderr": arch_probe.stderr,
            }
            write_json(report_dir / "rosetta_oracle_report.json", report)
            write_json(args.run_dir / "rosetta_oracle_report.json", report)
            return report

        monitor.set_phase("compile")
        compile_cmd = [clang, "-arch", "x86_64", "-O2", "-Wno-everything", str(c_path), str(s_path), "-o", str(exe_path)]
        compile_run = subprocess.run(compile_cmd, text=True, capture_output=True)
        if compile_run.returncode != 0:
            report = {
                "enabled": True,
                "oracle": "rosetta_x86_64",
                "ok": False,
                "error": "oracle compile failed",
                "availability": available,
                "compile_cmd": compile_cmd,
                "compile_stdout": compile_run.stdout[-4000:],
                "compile_stderr": compile_run.stderr[-4000:],
            }
            write_json(report_dir / "rosetta_oracle_report.json", report)
            write_json(args.run_dir / "rosetta_oracle_report.json", report)
            return report

        monitor.set_phase("execute")
        started = time.time()
        run = subprocess.run([str(exe_path)], text=True, capture_output=True, timeout=args.timeout_s)
        elapsed = round(time.time() - started, 4)
        parsed = parse_output(run.stdout, selected)
        split_exact = {
            split: {
                "checked_steps": sum(1 for item in selected if item["split"] == split),
            }
            for split in args.splits
        }
        report = {
            "enabled": True,
            "oracle": "rosetta_x86_64",
            "ok": run.returncode == 0 and parsed["failure_count"] == 0 and parsed["checked_steps"] == len(selected),
            "stage": "stage1_address_memory",
            "run_dir": str(args.run_dir),
            "report_dir": str(report_dir),
            "checked_steps": parsed["checked_steps"],
            "exact_steps": parsed["exact_steps"],
            "exact": parsed["exact"],
            "failure_count": parsed["failure_count"],
            "failure_examples": parsed["failure_examples"],
            "splits": split_exact,
            "requested_splits": list(args.splits),
            "source_trace_counts": split_counts,
            "max_steps": args.max_steps,
            "elapsed_s": elapsed,
            "availability": available,
            "compile_cmd": compile_cmd,
            "stdout_tail": run.stdout[-4000:],
            "stderr_tail": run.stderr[-4000:],
            "normalization": "LEA result registers are normalized from the Rosetta mmap base back to MEM_BASE before comparison.",
        }
        write_json(report_dir / "rosetta_oracle_report.json", report)
        write_json(args.run_dir / "rosetta_oracle_report.json", report)
        return report
    finally:
        monitor.set_phase("finalize")
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--report-dir", type=Path, default=None)
    p.add_argument("--splits", nargs="+", choices=["train", "val", "long"], default=["val", "long"])
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--timeout-s", type=float, default=60.0)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_rosetta_oracle(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
