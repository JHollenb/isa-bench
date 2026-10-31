#!/usr/bin/env python3
"""Emit a staged coverage ledger for the implemented PAN x86 slices.

This is not a replacement for Intel XED iform enumeration. It records the
implemented Stage 0, Stage 1, Stage 2, Stage 3, and Stage 4 forms, their validation artifacts, and the
remaining coverage gap to full x86.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, OPS as STAGE0_OPS, ResourceMonitor, write_json
from pan_x86_stages.stage1_memory_pan import OPS as STAGE1_OPS
from pan_x86_stages.stage2_control_flow_pan import OPS as STAGE2_OPS
from pan_x86_stages.stage3_complex_scalar_pan import OPS as STAGE3_OPS
from pan_x86_stages.stage4_simd_x87_pan import OPS as STAGE4_OPS


PAN_ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa"


def artifact_root_from_out_dir(out_dir: Path) -> Path:
    if out_dir.parent.name == "coverage_ledger":
        return out_dir.parent.parent
    return PAN_ARTIFACT_ROOT


def stage_run_dirs(artifact_root: Path) -> Dict[str, Path]:
    return {
        "stage0_x86_lite_alu": artifact_root / "stage0_x86_lite_alu" / "neural-vector-heldout-w8-unicorn-selector",
        "stage1_address_memory": artifact_root / "stage1_address_memory" / "stage1-memory-indexed-neural-unicorn",
        "stage2_control_flow": artifact_root / "stage2_control_flow" / "stage2-control-neural-unicorn-v3",
        "stage3_complex_scalar": artifact_root / "stage3_complex_scalar" / "stage3-complex-scalar-base-alu-v21-neural",
        "stage4_simd_x87": artifact_root / "stage4_simd_x87" / "stage4-sse-aligned-exception-v31-neural",
    }


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stage_metrics(run_dir: Path, stage_id: str) -> Dict[str, Any]:
    validation = load_json(run_dir / "validation_report.json")
    rosetta = load_json(run_dir / "rosetta_oracle_report.json")
    qemu_reports = []
    if stage_id == "stage4_simd_x87":
        qemu_reports.append(run_dir / "qemu_stage4_non_exception_priority_oracle" / "qemu_stage4_oracle_report.json")
    qemu_reports.extend(sorted(run_dir.glob("qemu_stage*_oracle/qemu_stage*_oracle_report.json")))
    qemu_reports.extend(sorted(run_dir.glob("qemu_stage*_oracle_report.json")))
    qemu_reports = [path for path in qemu_reports if path.exists()]
    qemu = load_json(qemu_reports[0]) if qemu_reports else {}
    bochs_reports = []
    if stage_id == "stage4_simd_x87":
        bochs_reports.append(run_dir / "bochs_stage4_aligned_exception_priority_oracle" / "bochs_stage4_oracle_report.json")
    bochs_reports.extend(sorted(run_dir.glob("bochs_stage*_oracle/bochs_stage*_oracle_report.json")))
    bochs_reports.extend(sorted(run_dir.glob("bochs_stage*_oracle_report.json")))
    bochs_reports = [path for path in bochs_reports if path.exists()]
    bochs = load_json(bochs_reports[0]) if bochs_reports else {}
    resource = load_json(run_dir / "resource_summary.json")
    return {
        "validation_gate": validation.get("gate_status", {}),
        "byte_overlap": validation.get("byte_overlap", {}),
        "external_oracle": validation.get("external_oracle", {}),
        "rosetta_oracle": {
            "present": bool(rosetta),
            "ok": rosetta.get("ok"),
            "checked_steps": rosetta.get("checked_steps"),
            "exact": rosetta.get("exact"),
        },
        "qemu_oracle": {
            "present": bool(qemu),
            "oracle": qemu.get("oracle"),
            "ok": qemu.get("ok"),
            "checked_steps": qemu.get("checked_steps"),
            "exception_steps": qemu.get("exception_steps"),
            "exact": qemu.get("exact"),
            "report": str(qemu_reports[0]) if qemu_reports else None,
        },
        "bochs_oracle": {
            "present": bool(bochs),
            "oracle": bochs.get("oracle"),
            "ok": bochs.get("ok"),
            "checked_steps": bochs.get("checked_steps"),
            "exception_steps": bochs.get("exception_steps"),
            "exact": bochs.get("exact"),
            "report": str(bochs_reports[0]) if bochs_reports else None,
        },
        "resource_summary": resource,
    }


def make_records() -> Sequence[Dict[str, Any]]:
    records = []
    for op in STAGE0_OPS:
        records.append({
            "stage": "stage0_x86_lite_alu",
            "iform": op.iform,
            "name": op.name,
            "family": op.family,
            "operands": op.operands,
            "status": "learned_validated",
            "scope": "low-byte ALU over AL/CL/DL/BL",
            "xed_source": "internal_real_encoding_not_xed_enumerated",
            "limitations": ["no memory", "no control flow", "no prefixes or mode variation"],
        })
    for op in STAGE1_OPS:
        records.append({
            "stage": "stage1_address_memory",
            "iform": op.iform,
            "name": op.name,
            "family": op.family,
            "operands": op.operands,
            "status": "learned_validated",
            "scope": "8/16/32/64-bit scalar memory/addressing over RAX/RCX/RDX/RBX, RSI/RDI/RBP bases, and R8/R9 index-scale offsets",
            "xed_source": "internal_real_encoding_not_xed_enumerated",
            "limitations": ["scalar integer memory only", "flat mapped memory", "no page faults", "no segment behavior"],
        })
    for op in STAGE2_OPS:
        records.append({
            "stage": "stage2_control_flow",
            "iform": op.iform,
            "name": op.name,
            "family": op.family,
            "operands": op.operands,
            "status": "learned_validated",
            "scope": "RAX/AL plus CF/ZF/SF/OF, RIP/RSP, conditional branches, near CALL/RET, and PUSH/POP over a bounded stack slice",
            "xed_source": "internal_real_encoding_not_xed_enumerated",
            "limitations": ["single-GPR control-flow slice", "bounded flat stack only", "no SETcc/CMOVcc", "no exceptions, interrupts, or privilege behavior"],
        })
    for op in STAGE3_OPS:
        records.append({
            "stage": "stage3_complex_scalar",
            "iform": op.iform,
            "name": op.name,
            "family": op.family,
            "operands": op.operands,
            "status": "learned_validated",
            "scope": (
                "RAX/RBX/RCX/RDX 64-bit register forms for BSWAP, MOVSX/MOVZX r8/r16/m8/m16, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR r64/r64 and r64/m64, TEST r64/r64 and m64/r64, bit scans, "
                "bit test/modify, rotate-through-carry, multiply, safe no-fault divide, DIV/IDIV #DE, "
                "POPCNT/LZCNT/TZCNT register forms, read-only MOVSX/MOVZX m8/m16 and BSF/BSR/POPCNT/LZCNT/TZCNT m64 source-memory forms, selected BMI1 ANDN/BEXTR/BLSI/BLSMSK/BLSR register and m64 source-memory forms, "
                "selected BMI2 BZHI/SHLX/SHRX/SARX/RORX/MULX/PDEP/PEXT register and m64 source-memory forms, BT/BTS/BTR/BTC m64 imm8/r64 bit-test/modify forms, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 forms"
            ),
            "xed_source": "internal_real_encoding_not_xed_enumerated",
            "limitations": [
                "divide-error coverage is limited to DIV/IDIV #DE cases",
                "register operands over RAX/RBX/RCX/RDX only",
                "memory operands are limited to read-only MOVSX/MOVZX m8/m16, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR m64 source forms, TEST m64 r64, BSF/BSR/POPCNT/LZCNT/TZCNT, selected BMI1/BMI2 m64 source forms, BT/BTS/BTR/BTC m64 imm8/r64 bit-test/modify forms, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms over a flat 256-byte image",
                "undefined flags are masked rather than claimed exact",
                "CPUID-gated scalar coverage is limited to POPCNT/LZCNT/TZCNT plus selected BMI1/BMI2 register and m64 source forms",
                "BMI1 coverage is limited to TZCNT plus selected ANDN/BEXTR/BLSI/BLSMSK/BLSR register and m64 source forms",
                "BMI2 coverage is limited to selected BZHI/SHLX/SHRX/SARX/RORX/MULX/PDEP/PEXT register and m64 source forms",
            ],
        })
    for op in STAGE4_OPS:
        records.append({
            "stage": "stage4_simd_x87",
            "iform": op.iform,
            "name": op.name,
            "family": op.family,
            "operands": op.operands,
            "status": "learned_validated",
            "scope": "SSE/SSE2 integer and FP XMM register-register, aligned source-memory, MOVDQA/MOVDQU/MOVAPD/MOVUPD/MOVNTDQ/MOVNTPD m128 memory-destination, MOVSD/MOVLPD/MOVHPD/MOVQ m64 memory-destination, MOVD m32 memory-destination, reciprocal MOVD/MOVQ/MOVLPD/MOVHPD/MOVAPD/MOVUPD memory-source transfer loads, unaligned MOVDQU source/destination, aligned m128 unaligned-memory #GP(0) cases, scalar/packed single and double add/sub/mul/div/sqrt/min/max register and memory-source forms over safe finite plus non-trapping zero/infinity operands, CMPSS/CMPPS/CMPSD/CMPPD register and memory-source compare forms over safe finite, zero, infinity, quiet-NaN, and signed-zero operands, immediate-shift, and variable-count shift arithmetic, logical, compare, pack, unpack, min/max, average, multiply-sum, and shift forms over XMM0-XMM15 with exact 128-bit state, memory, and exception comparison",
            "xed_source": "internal_real_encoding_not_xed_enumerated",
            "limitations": [
                "XMM0-XMM15 register operands",
                "register-register, aligned absolute m128 source-memory, reciprocal MOVD/MOVQ/MOVLPD/MOVHPD/MOVAPD/MOVUPD memory-source transfer loads, MOVDQA/MOVDQU/MOVAPD/MOVUPD/MOVNTDQ/MOVNTPD m128 memory-destination, MOVSD/MOVLPD/MOVHPD/MOVQ m64 memory-destination, MOVD m32 memory-destination, unaligned MOVDQU source/destination, aligned m128 unaligned-memory #GP(0), scalar/packed single and double add/sub/mul/div/sqrt/min/max register and memory-source forms over safe finite plus non-trapping zero/infinity operands, CMPSS/CMPPS/CMPSD/CMPPD register and memory-source forms over safe finite, zero, infinity, quiet-NaN, and signed-zero operands, selected imm8, and XMM-count shift forms only",
                "memory sources and destinations are limited to the current aligned/unaligned m128, m64, and m32 transfer/FP/integer subset; exception modeling is limited to aligned m128 unaligned-memory #GP(0)",
                "no AVX/VEX/EVEX encodings, masks, YMM/ZMM, MMX, or x87",
                "no subnormal, overflow, underflow, divide-by-zero status-flag, invalid-operation status-flag, signaling-NaN behavior, quiet-NaN arithmetic/min-max payload behavior, signed-zero min-max edge claims beyond oracle-agreeing non-trapping cases, or broader MXCSR exception behavior",
            ],
        })
    return records


def build_ledger(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    stage_runs = stage_run_dirs(artifact_root_from_out_dir(out_dir))
    records = list(make_records())
    by_stage: Dict[str, int] = {}
    by_status: Dict[str, int] = {}
    for rec in records:
        by_stage[rec["stage"]] = by_stage.get(rec["stage"], 0) + 1
        by_status[rec["status"]] = by_status.get(rec["status"], 0) + 1
    ledger = {
        "created": DEFAULT_DATE,
        "run_name": args.run_name,
        "ledger_kind": "implemented_stage_subset",
        "full_x86_coverage_complete": False,
        "intel_xed_iform_enumeration_integrated": False,
        "summary": {
            "implemented_forms": len(records),
            "by_stage": by_stage,
            "by_status": by_status,
            "next_required_ledger": "Intel XED iform-level coverage over the frozen x86-64 user-mode target",
        },
        "stage_evidence": {
            "stage0_x86_lite_alu": {
                "run_dir": str(stage_runs["stage0_x86_lite_alu"]),
                **stage_metrics(stage_runs["stage0_x86_lite_alu"], "stage0_x86_lite_alu"),
            },
            "stage1_address_memory": {
                "run_dir": str(stage_runs["stage1_address_memory"]),
                **stage_metrics(stage_runs["stage1_address_memory"], "stage1_address_memory"),
            },
            "stage2_control_flow": {
                "run_dir": str(stage_runs["stage2_control_flow"]),
                **stage_metrics(stage_runs["stage2_control_flow"], "stage2_control_flow"),
            },
            "stage3_complex_scalar": {
                "run_dir": str(stage_runs["stage3_complex_scalar"]),
                **stage_metrics(stage_runs["stage3_complex_scalar"], "stage3_complex_scalar"),
            },
            "stage4_simd_x87": {
                "run_dir": str(stage_runs["stage4_simd_x87"]),
                **stage_metrics(stage_runs["stage4_simd_x87"], "stage4_simd_x87"),
            },
        },
        "records": records,
        "unimplemented_stage_groups": [
            "partial-register aliases beyond selected low/word/dword/qword GPR forms",
            "addressing forms beyond base/index/scale/disp over the selected registers",
            "control-flow forms beyond the selected Jcc/JMP/CALL/RET slice",
            "stack semantics beyond bounded PUSH/POP/CALL/RET over RAX/RSP",
            "complex scalar exception cases beyond DIV/IDIV #DE and memory operands beyond the selected read-only MOVSX/MOVZX m8/m16, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR m64 source, TEST m64 r64, BSF/BSR/POPCNT/LZCNT/TZCNT/BMI1/BMI2 m64 sources plus BT/BTS/BTR/BTC m64 imm8/r64, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms",
            "CPUID-gated scalar forms beyond POPCNT/LZCNT/TZCNT and selected BMI1/BMI2 register and m64 source forms, including memory-destination BMI forms, aliases, broader BMI forms, and APX",
            "SIMD/x87 beyond the expanded SSE/SSE2 XMM0-XMM15 register, aligned source-memory, reciprocal MOVD/MOVQ/MOVLPD/MOVHPD/MOVAPD/MOVUPD transfer loads, current m128/m64/m32 destination stores, MOVDQU unaligned load/store, aligned m128 #GP alignment-fault cases, scalar/packed single and double add/sub/mul/div/sqrt/min/max register and memory-source forms over safe finite plus non-trapping zero/infinity operands, and CMPSS/CMPPS/CMPSD/CMPPD register/memory-source compare forms over safe finite, zero, infinity, quiet-NaN, and signed-zero operands",
            "exceptions/system/privileged",
            "compatibility and legacy modes",
        ],
    }
    write_json(out_dir / "coverage_ledger.json", ledger)
    return ledger


def default_out_dir(run_name: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / "coverage_ledger" / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-name", default=f"coverage-{int(time.time())}")
    p.add_argument("--monitor-interval", type=float, default=0.5)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or default_out_dir(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(out_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("enumerate")
        ledger = build_ledger(args, out_dir)
        monitor.set_phase("finalize")
        print(json.dumps({
            "out_dir": str(out_dir),
            "summary": ledger["summary"],
            "full_x86_coverage_complete": ledger["full_x86_coverage_complete"],
            "intel_xed_iform_enumeration_integrated": ledger["intel_xed_iform_enumeration_integrated"],
        }, indent=2, sort_keys=True))
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
