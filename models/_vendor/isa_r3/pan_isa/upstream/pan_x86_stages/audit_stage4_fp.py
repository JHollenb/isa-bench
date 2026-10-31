#!/usr/bin/env python3
"""Adversarial trace audit for the Stage 4 SSE/SSE2 FP slice."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, load_traces, write_json
from pan_x86_stages.stage4_simd_x87_pan import (
    HIGH64_MASK,
    HIGH96_MASK,
    MASK32,
    MASK64,
    MASK128,
    MEM_SIZE,
    MEMORY_DEST_OPERANDS,
    MEMORY_OPERANDS,
    MEMORY_SOURCE_OPERANDS,
    OPS,
    XMM_REGS,
    apply_op,
    fp_operands_safe,
    is_fp_compare_name,
    is_compare_fp_bits,
    is_inf_fp_bits,
    is_quiet_nan_f32_bits,
    is_quiet_nan_f64_bits,
    is_quiet_nan_fp_bits,
    is_safe_fp_bits,
    is_zero_fp_bits,
    is_zero_f32_bits,
    is_zero_f64_bits,
    memory_digest,
    memory_operand_width_bytes,
    packed_fp_result_bits,
    register_write_mask,
    scalar_fp_result_bits,
    write_memory_operand,
)


DEFAULT_RUN_DIR = (
    DEFAULT_ARTIFACT_ROOT
    / DEFAULT_DATE
    / "pan-x86-isa"
    / "stage4_simd_x87"
    / "stage4-sse-aligned-exception-v31-neural"
)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def hex128(value: int) -> str:
    return f"0x{int(value) & MASK128:032x}"


def ints(values: Sequence[Any]) -> List[int]:
    return [int(value) for value in values]


def safe_fp_register_bits(value: int) -> bool:
    f64_lanes = [(value >> shift) & MASK64 for shift in (0, 64)]
    f32_lanes = [(value >> shift) & MASK32 for shift in (0, 32, 64, 96)]
    return (
        all(is_safe_fp_bits(lane, 64) or is_compare_fp_bits(lane, 64) or is_inf_fp_bits(lane, 64) for lane in f64_lanes)
        or all(is_safe_fp_bits(lane, 32) or is_compare_fp_bits(lane, 32) or is_inf_fp_bits(lane, 32) for lane in f32_lanes)
    )


def inc(mapping: Dict[str, int], key: str, amount: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + amount


def compare_operand_specials(op_name: str, before_dst: int, source_value: int, lane_bits: int) -> Dict[str, int]:
    lane_count = 128 // lane_bits if op_name.startswith(("CMPPD_", "CMPPS_")) else 1
    mask = (1 << lane_bits) - 1
    nan_lanes = 0
    zero_lanes = 0
    for lane in range(lane_count):
        shift = lane * lane_bits
        for value in ((before_dst >> shift) & mask, (source_value >> shift) & mask):
            if (is_quiet_nan_f32_bits(value) if lane_bits == 32 else is_quiet_nan_f64_bits(value)):
                nan_lanes += 1
            if (is_zero_f32_bits(value) if lane_bits == 32 else is_zero_f64_bits(value)):
                zero_lanes += 1
    return {
        "nan_lanes": nan_lanes,
        "zero_lanes": zero_lanes,
        "nan_step": int(nan_lanes > 0),
        "zero_step": int(zero_lanes > 0),
    }


def fp_operand_specials(op_name: str, family: str, before_dst: int, source_value: int, lane_bits: int) -> Dict[str, int]:
    lane_count = 128 // lane_bits if family == "SSE_FP_PACKED" else 1
    source_only = op_name.split("_", 1)[0] in {"SQRTSD", "SQRTPD", "SQRTSS", "SQRTPS"}
    mask = (1 << lane_bits) - 1
    zero_lanes = 0
    infinity_lanes = 0
    quiet_nan_lanes = 0
    for lane in range(lane_count):
        shift = lane * lane_bits
        values = [(source_value >> shift) & mask]
        if not source_only:
            values.append((before_dst >> shift) & mask)
        for value in values:
            zero_lanes += int(is_zero_fp_bits(value, lane_bits))
            infinity_lanes += int(is_inf_fp_bits(value, lane_bits))
            quiet_nan_lanes += int(is_quiet_nan_fp_bits(value, lane_bits))
    return {
        "zero_lanes": zero_lanes,
        "infinity_lanes": infinity_lanes,
        "quiet_nan_lanes": quiet_nan_lanes,
        "zero_step": int(zero_lanes > 0),
        "infinity_step": int(infinity_lanes > 0),
        "quiet_nan_step": int(quiet_nan_lanes > 0),
    }


def add_failure(report: Dict[str, Any], kind: str, split: str, trace_id: str, step_index: int, detail: Dict[str, Any]) -> None:
    totals = report["totals"]
    inc(totals, f"{kind}_failures")
    inc(report["splits"][split], f"{kind}_failures")
    if len(report["failure_examples"]) < report["max_failure_examples"]:
        report["failure_examples"].append(
            {
                "kind": kind,
                "split": split,
                "trace_id": trace_id,
                "step_index": step_index,
                **detail,
            }
        )


def expected_after_xmm(step: Dict[str, Any]) -> List[int]:
    p = step["pan_stage4"]
    op = OPS[int(p["op_index"])]
    src_value: Optional[int] = None
    if op.operands in MEMORY_SOURCE_OPERANDS:
        src_value = int(p["mem_value"])
    return apply_op(
        ints(p["before_xmm"]),
        int(p["op_index"]),
        int(p["dst"]),
        int(p["src"]),
        int(p["imm"]),
        src_value,
    )


def expected_after_memory(step: Dict[str, Any]) -> Optional[List[int]]:
    p = step["pan_stage4"]
    op = OPS[int(p["op_index"])]
    if "before_memory" not in p:
        return None
    before_memory = [int(x) & 0xFF for x in p["before_memory"]]
    if op.operands in MEMORY_DEST_OPERANDS and op.expected_exception is None:
        return write_memory_operand(before_memory, int(p["mem_offset"]), op, int(p["before_xmm"][int(p["src"])]))
    return before_memory


def audit_register_writes(
    report: Dict[str, Any],
    split: str,
    trace_id: str,
    step: Dict[str, Any],
    before_xmm: Sequence[int],
    after_xmm: Sequence[int],
) -> None:
    p = step["pan_stage4"]
    op = OPS[int(p["op_index"])]
    expected_mask = register_write_mask(op)
    write_by_name = {write["name"]: write for write in step["effects"].get("register_writes", [])}
    for index, reg in enumerate(XMM_REGS):
        before = int(before_xmm[index])
        after = int(after_xmm[index])
        changed_bits = (before ^ after) & MASK128
        write = write_by_name.get(reg)
        if changed_bits and write is None:
            add_failure(
                report,
                "mask",
                split,
                trace_id,
                int(step["index"]),
                {"op": op.name, "reg": reg, "reason": "missing_register_write"},
            )
        if write is not None:
            mask = int(str(write["mask"]), 16)
            if mask != expected_mask:
                add_failure(
                    report,
                    "mask",
                    split,
                    trace_id,
                    int(step["index"]),
                    {
                        "op": op.name,
                        "reg": reg,
                        "reason": "wrong_write_mask",
                        "expected_mask": hex128(expected_mask),
                        "actual_mask": str(write["mask"]),
                    },
                )
            if changed_bits & ~mask:
                add_failure(
                    report,
                    "mask",
                    split,
                    trace_id,
                    int(step["index"]),
                    {
                        "op": op.name,
                        "reg": reg,
                        "reason": "changed_bits_outside_mask",
                        "changed_bits": hex128(changed_bits),
                        "mask": str(write["mask"]),
                    },
                )


def audit_fp_step(
    report: Dict[str, Any],
    split: str,
    trace_id: str,
    step: Dict[str, Any],
    before_xmm: Sequence[int],
    after_xmm: Sequence[int],
) -> None:
    p = step["pan_stage4"]
    op = OPS[int(p["op_index"])]
    if not op.family.startswith("SSE_FP_"):
        return
    dst = int(p["dst"])
    src = int(p["src"])
    source_value = int(p["mem_value"]) if op.operands in MEMORY_SOURCE_OPERANDS else int(before_xmm[src])
    specials = fp_operand_specials(op.name, op.family, int(before_xmm[dst]), source_value, op.lane_bits)
    inc(report["totals"], "fp_zero_operand_steps", specials["zero_step"])
    inc(report["splits"][split], "fp_zero_operand_steps", specials["zero_step"])
    inc(report["totals"], "fp_zero_operand_lanes", specials["zero_lanes"])
    inc(report["splits"][split], "fp_zero_operand_lanes", specials["zero_lanes"])
    inc(report["totals"], "fp_infinity_operand_steps", specials["infinity_step"])
    inc(report["splits"][split], "fp_infinity_operand_steps", specials["infinity_step"])
    inc(report["totals"], "fp_infinity_operand_lanes", specials["infinity_lanes"])
    inc(report["splits"][split], "fp_infinity_operand_lanes", specials["infinity_lanes"])
    inc(report["totals"], "fp_quiet_nan_operand_steps", specials["quiet_nan_step"])
    inc(report["splits"][split], "fp_quiet_nan_operand_steps", specials["quiet_nan_step"])
    inc(report["totals"], "fp_quiet_nan_operand_lanes", specials["quiet_nan_lanes"])
    inc(report["splits"][split], "fp_quiet_nan_operand_lanes", specials["quiet_nan_lanes"])
    compare_specials = (
        compare_operand_specials(op.name, int(before_xmm[dst]), source_value, op.lane_bits)
        if is_fp_compare_name(op.name)
        else None
    )
    if compare_specials is not None:
        inc(report["totals"], "fp_compare_steps")
        inc(report["splits"][split], "fp_compare_steps")
        inc(report["totals"], "fp_compare_nan_lanes", compare_specials["nan_lanes"])
        inc(report["splits"][split], "fp_compare_nan_lanes", compare_specials["nan_lanes"])
        inc(report["totals"], "fp_compare_zero_lanes", compare_specials["zero_lanes"])
        inc(report["splits"][split], "fp_compare_zero_lanes", compare_specials["zero_lanes"])
        inc(report["totals"], "fp_compare_nan_operand_steps", compare_specials["nan_step"])
        inc(report["splits"][split], "fp_compare_nan_operand_steps", compare_specials["nan_step"])
        inc(report["totals"], "fp_compare_zero_operand_steps", compare_specials["zero_step"])
        inc(report["splits"][split], "fp_compare_zero_operand_steps", compare_specials["zero_step"])
    if op.family == "SSE_FP_SCALAR":
        inc(report["totals"], "fp_scalar_steps")
        inc(report["splits"][split], "fp_scalar_steps")
        inc(report["totals"]["fp_scalar_iclasses"], step["decode"]["iclass"])
        if op.operands in MEMORY_SOURCE_OPERANDS:
            inc(report["totals"], "fp_memory_source_steps")
            inc(report["splits"][split], "fp_memory_source_steps")
            inc(report["totals"]["fp_memory_source_iclasses"], step["decode"]["iclass"])
        if not fp_operands_safe(op, before_xmm, dst, src, source_value if op.operands in MEMORY_SOURCE_OPERANDS else None):
            add_failure(report, "fp_semantic", split, trace_id, int(step["index"]), {"op": op.name, "reason": "unsafe_operands"})
            return
        lane_mask = (1 << op.lane_bits) - 1
        high_mask = MASK128 ^ lane_mask
        expected_low = scalar_fp_result_bits(
            op.name,
            int(before_xmm[dst]) & lane_mask,
            source_value & lane_mask,
            int(p["imm"]),
            op.lane_bits,
        )
        if (int(after_xmm[dst]) & lane_mask) != expected_low:
            add_failure(
                report,
                "fp_semantic",
                split,
                trace_id,
                int(step["index"]),
                {
                    "op": op.name,
                    "reason": f"scalar_low{op.lane_bits}_mismatch",
                    "expected_low": f"0x{expected_low:0{op.lane_bits // 4}x}",
                    "actual_low": f"0x{int(after_xmm[dst]) & lane_mask:0{op.lane_bits // 4}x}",
                },
            )
        if (int(after_xmm[dst]) & high_mask) != (int(before_xmm[dst]) & high_mask):
            add_failure(
                report,
                "fp_mask",
                split,
                trace_id,
                int(step["index"]),
                {"op": op.name, "reason": f"scalar_high{128 - op.lane_bits}_not_preserved"},
            )
    elif op.family == "SSE_FP_PACKED":
        inc(report["totals"], "fp_packed_steps")
        inc(report["splits"][split], "fp_packed_steps")
        inc(report["totals"]["fp_packed_iclasses"], step["decode"]["iclass"])
        if op.operands in MEMORY_SOURCE_OPERANDS:
            inc(report["totals"], "fp_memory_source_steps")
            inc(report["splits"][split], "fp_memory_source_steps")
            inc(report["totals"]["fp_memory_source_iclasses"], step["decode"]["iclass"])
        if not fp_operands_safe(op, before_xmm, dst, src, source_value if op.operands in MEMORY_SOURCE_OPERANDS else None):
            add_failure(report, "fp_semantic", split, trace_id, int(step["index"]), {"op": op.name, "reason": "unsafe_operands"})
            return
        expected = packed_fp_result_bits(op.name, int(before_xmm[dst]), source_value, int(p["imm"]), op.lane_bits)
        if int(after_xmm[dst]) != expected:
            add_failure(
                report,
                "fp_semantic",
                split,
                trace_id,
                int(step["index"]),
                {"op": op.name, "reason": "packed_result_mismatch", "expected": hex128(expected), "actual": hex128(after_xmm[dst])},
            )
    else:
        return
    for index, reg in enumerate(XMM_REGS):
        if index != dst and int(after_xmm[index]) != int(before_xmm[index]):
            add_failure(
                report,
                "fp_mask",
                split,
                trace_id,
                int(step["index"]),
                {"op": op.name, "reason": "non_destination_xmm_changed", "reg": reg},
            )


def audit_step(report: Dict[str, Any], split: str, trace: Dict[str, Any], step: Dict[str, Any]) -> None:
    p = step["pan_stage4"]
    op = OPS[int(p["op_index"])]
    before_xmm = ints(p["before_xmm"])
    after_xmm = ints(p["after_xmm"])
    expected_xmm = expected_after_xmm(step)
    if expected_xmm != after_xmm:
        add_failure(
            report,
            "semantic",
            split,
            trace["trace_id"],
            int(step["index"]),
            {"op": op.name, "reason": "xmm_after_mismatch"},
        )
    expected_memory = expected_after_memory(step)
    if expected_memory is not None:
        actual_memory = [int(x) & 0xFF for x in p["after_memory"]]
        if actual_memory != expected_memory:
            add_failure(report, "semantic", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "memory_after_mismatch"})
        if memory_digest([int(x) & 0xFF for x in p["before_memory"]]) != step["before"]["memory_digest"]:
            add_failure(report, "metadata", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "before_memory_digest_mismatch"})
        if memory_digest(actual_memory) != step["after"]["memory_digest"]:
            add_failure(report, "metadata", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "after_memory_digest_mismatch"})
    if step["decode"].get("iform") != op.iform or step["decode"].get("category") != op.family:
        add_failure(report, "metadata", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "decode_metadata_mismatch"})
    if op.expected_exception is not None:
        inc(report["totals"], "exception_steps")
        inc(report["splits"][split], "exception_steps")
        inc(report["totals"]["exception_iclasses"], step["decode"]["iclass"])
        if p.get("exception") != op.expected_exception:
            add_failure(report, "metadata", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "expected_exception_missing"})
    if op.operands in MEMORY_OPERANDS and "mem_offset" in p:
        offset = int(p["mem_offset"])
        width = memory_operand_width_bytes(op)
        if offset < 0 or offset + width > MEM_SIZE:
            add_failure(report, "metadata", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "memory_offset_out_of_bounds"})
        if op.expected_exception is not None and width == 16 and offset % 16 == 0:
            add_failure(report, "metadata", split, trace["trace_id"], int(step["index"]), {"op": op.name, "reason": "exception_offset_is_aligned"})
    if op.operands in MEMORY_DEST_OPERANDS:
        inc(report["totals"], "memory_destination_steps")
        inc(report["splits"][split], "memory_destination_steps")
        inc(report["totals"]["memory_destination_iclasses"], step["decode"]["iclass"])
    if op.operands in MEMORY_SOURCE_OPERANDS:
        inc(report["totals"], "memory_source_steps")
        inc(report["splits"][split], "memory_source_steps")
        inc(report["totals"]["memory_source_iclasses"], step["decode"]["iclass"])
    audit_register_writes(report, split, trace["trace_id"], step, before_xmm, after_xmm)
    audit_fp_step(report, split, trace["trace_id"], step, before_xmm, after_xmm)


def audit_oracle_report(
    report: Dict[str, Any],
    engine: str,
    path: Path,
    priority_min: int,
    expected_steps: int,
    expected_exception_steps: Optional[int] = None,
) -> None:
    oracle = load_json(path)
    summary = {
        "path": str(path),
        "present": bool(oracle),
        "ok": oracle.get("ok"),
        "checked_steps": oracle.get("checked_steps"),
        "exact": oracle.get("exact"),
        "priority_category": oracle.get("priority_category"),
        "priority_operands": oracle.get("priority_operands"),
        "priority_iclasses": oracle.get("priority_iclasses"),
        "priority_selected_steps": oracle.get("priority_selected_steps"),
        "selected_iclasses": oracle.get("selected_iclasses", {}),
        "selected_iforms": oracle.get("selected_iforms", {}),
        "fp_scalar_steps": oracle.get("fp_scalar_steps"),
        "fp_scalar_iclasses": oracle.get("fp_scalar_iclasses", {}),
        "fp_packed_steps": oracle.get("fp_packed_steps"),
        "fp_packed_iclasses": oracle.get("fp_packed_iclasses", {}),
        "fp_memory_source_steps": oracle.get("fp_memory_source_steps"),
        "fp_memory_source_iclasses": oracle.get("fp_memory_source_iclasses", {}),
        "fp_zero_operand_steps": oracle.get("fp_zero_operand_steps"),
        "fp_infinity_operand_steps": oracle.get("fp_infinity_operand_steps"),
        "fp_quiet_nan_operand_steps": oracle.get("fp_quiet_nan_operand_steps"),
        "memory_source_steps": oracle.get("memory_source_steps"),
        "memory_source_iclasses": oracle.get("memory_source_iclasses", {}),
        "memory_destination_steps": oracle.get("memory_destination_steps"),
        "memory_destination_iclasses": oracle.get("memory_destination_iclasses", {}),
        "exception_steps": oracle.get("exception_steps"),
        "exception_iclasses": oracle.get("exception_iclasses", {}),
    }
    requested_iclasses = [str(x) for x in oracle.get("priority_iclasses", [])]
    combined_iclasses: Dict[str, int] = {}
    for key in (
        "selected_iclasses",
        "fp_scalar_iclasses",
        "fp_packed_iclasses",
        "memory_source_iclasses",
        "memory_destination_iclasses",
        "exception_iclasses",
    ):
        for iclass, count in oracle.get(key, {}).items():
            combined_iclasses[str(iclass)] = combined_iclasses.get(str(iclass), 0) + int(count)
    iclass_condition = all(combined_iclasses.get(iclass, 0) > 0 for iclass in requested_iclasses)
    if not requested_iclasses:
        priority_categories = set(str(x).strip() for x in str(oracle.get("priority_category") or "").split(",") if str(x).strip())
        priority_operands = set(str(x) for x in oracle.get("priority_operands", []))
        if "EXCEPTION" in priority_categories:
            iclass_condition = int(oracle.get("exception_steps", 0)) >= priority_min
        elif priority_operands.intersection(MEMORY_SOURCE_OPERANDS):
            iclass_condition = int(oracle.get("memory_source_steps", 0)) >= priority_min
        elif priority_operands.intersection(MEMORY_DEST_OPERANDS):
            iclass_condition = int(oracle.get("memory_destination_steps", 0)) >= priority_min
        else:
            iclass_condition = True
    report["oracles"][engine] = summary
    conditions = [
        bool(oracle),
        oracle.get("ok") is True,
        int(oracle.get("checked_steps", 0)) == expected_steps,
        float(oracle.get("exact", 0.0)) == 1.0,
        int(oracle.get("priority_selected_steps", 0)) >= priority_min,
        iclass_condition,
    ]
    if expected_exception_steps is not None:
        conditions.append(int(oracle.get("exception_steps", 0)) == int(expected_exception_steps))
    if not all(conditions):
        inc(report["totals"], "oracle_failures")
        if len(report["failure_examples"]) < report["max_failure_examples"]:
            report["failure_examples"].append({"kind": "oracle", "engine": engine, "summary": summary})


def audit_run(args: argparse.Namespace) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "splits_audited": list(args.splits),
        "max_failure_examples": args.max_failure_examples,
        "totals": {
            "traces": 0,
            "steps": 0,
            "fp_scalar_steps": 0,
            "fp_packed_steps": 0,
            "fp_compare_steps": 0,
            "fp_compare_nan_operand_steps": 0,
            "fp_compare_zero_operand_steps": 0,
            "fp_compare_nan_lanes": 0,
            "fp_compare_zero_lanes": 0,
            "fp_zero_operand_steps": 0,
            "fp_infinity_operand_steps": 0,
            "fp_quiet_nan_operand_steps": 0,
            "fp_zero_operand_lanes": 0,
            "fp_infinity_operand_lanes": 0,
            "fp_quiet_nan_operand_lanes": 0,
            "fp_scalar_iclasses": {},
            "fp_packed_iclasses": {},
            "fp_memory_source_steps": 0,
            "fp_memory_source_iclasses": {},
            "exception_steps": 0,
            "exception_iclasses": {},
            "memory_source_steps": 0,
            "memory_source_iclasses": {},
            "memory_destination_steps": 0,
            "memory_destination_iclasses": {},
            "initial_xmm_registers": 0,
            "initial_non_safe_fp_registers": 0,
            "semantic_failures": 0,
            "continuity_failures": 0,
            "metadata_failures": 0,
            "mask_failures": 0,
            "fp_semantic_failures": 0,
            "fp_mask_failures": 0,
            "coverage_failures": 0,
            "oracle_failures": 0,
        },
        "splits": {},
        "oracles": {},
        "failure_examples": [],
    }
    for split in args.splits:
        split_report = {
            "traces": 0,
            "steps": 0,
            "fp_scalar_steps": 0,
            "fp_packed_steps": 0,
            "fp_compare_steps": 0,
            "fp_compare_nan_operand_steps": 0,
            "fp_compare_zero_operand_steps": 0,
            "fp_compare_nan_lanes": 0,
            "fp_compare_zero_lanes": 0,
            "fp_zero_operand_steps": 0,
            "fp_infinity_operand_steps": 0,
            "fp_quiet_nan_operand_steps": 0,
            "fp_zero_operand_lanes": 0,
            "fp_infinity_operand_lanes": 0,
            "fp_quiet_nan_operand_lanes": 0,
            "fp_memory_source_steps": 0,
            "exception_steps": 0,
            "memory_source_steps": 0,
            "memory_destination_steps": 0,
            "initial_xmm_registers": 0,
            "initial_non_safe_fp_registers": 0,
            "semantic_failures": 0,
            "continuity_failures": 0,
            "metadata_failures": 0,
            "mask_failures": 0,
            "fp_semantic_failures": 0,
            "fp_mask_failures": 0,
        }
        report["splits"][split] = split_report
        for trace in load_traces(args.run_dir / f"{split}_traces.jsonl"):
            split_report["traces"] += 1
            report["totals"]["traces"] += 1
            previous_after_xmm: Optional[List[int]] = None
            previous_after_digest: Optional[str] = None
            for step_index, step in enumerate(trace["steps"]):
                split_report["steps"] += 1
                report["totals"]["steps"] += 1
                p = step["pan_stage4"]
                before_xmm = ints(p["before_xmm"])
                if step_index == 0:
                    nonsafe = sum(1 for value in before_xmm if not safe_fp_register_bits(value))
                    split_report["initial_xmm_registers"] += len(before_xmm)
                    split_report["initial_non_safe_fp_registers"] += nonsafe
                    report["totals"]["initial_xmm_registers"] += len(before_xmm)
                    report["totals"]["initial_non_safe_fp_registers"] += nonsafe
                if previous_after_xmm is not None and previous_after_xmm != before_xmm:
                    add_failure(report, "continuity", split, trace["trace_id"], int(step["index"]), {"reason": "xmm_state_not_carried"})
                if previous_after_digest is not None and previous_after_digest != step["before"]["memory_digest"]:
                    add_failure(report, "continuity", split, trace["trace_id"], int(step["index"]), {"reason": "memory_digest_not_carried"})
                audit_step(report, split, trace, step)
                previous_after_xmm = ints(p["after_xmm"])
                previous_after_digest = step["after"]["memory_digest"]
    if args.check_oracles:
        audit_oracle_report(
            report,
            "qemu",
            args.run_dir / "qemu_stage4_non_exception_priority_oracle" / "qemu_stage4_oracle_report.json",
            args.qemu_priority_min,
            args.qemu_oracle_checked_steps,
            expected_exception_steps=0,
        )
        audit_oracle_report(
            report,
            "bochs",
            args.run_dir / "bochs_stage4_aligned_exception_priority_oracle" / "bochs_stage4_oracle_report.json",
            args.priority_min,
            args.oracle_checked_steps,
        )
    coverage_requirements = [
        ("fp_compare_nan_operand_steps", args.min_compare_nan_steps),
        ("fp_compare_zero_operand_steps", args.min_compare_zero_steps),
        ("fp_zero_operand_steps", args.min_fp_zero_steps),
        ("fp_infinity_operand_steps", args.min_fp_infinity_steps),
        ("exception_steps", args.min_exception_steps),
    ]
    for key, minimum in coverage_requirements:
        if int(report["totals"].get(key, 0)) < int(minimum):
            inc(report["totals"], "coverage_failures")
            if len(report["failure_examples"]) < report["max_failure_examples"]:
                report["failure_examples"].append(
                    {
                        "kind": "coverage",
                        "counter": key,
                        "actual": int(report["totals"].get(key, 0)),
                        "minimum": int(minimum),
                    }
                )
    failure_keys = [key for key in report["totals"] if key.endswith("_failures")]
    report["ok"] = all(int(report["totals"].get(key, 0)) == 0 for key in failure_keys)
    report["initial_non_safe_fp_register_fraction"] = round(
        report["totals"]["initial_non_safe_fp_registers"] / max(1, report["totals"]["initial_xmm_registers"]),
        6,
    )
    write_json(args.out, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "long"])
    parser.add_argument("--priority-min", type=int, default=512)
    parser.add_argument("--oracle-checked-steps", type=int, default=1024)
    parser.add_argument("--qemu-priority-min", type=int, default=512)
    parser.add_argument("--qemu-oracle-checked-steps", type=int, default=512)
    parser.add_argument("--check-oracles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-compare-nan-steps", type=int, default=0)
    parser.add_argument("--min-compare-zero-steps", type=int, default=0)
    parser.add_argument("--min-fp-zero-steps", type=int, default=0)
    parser.add_argument("--min-fp-infinity-steps", type=int, default=0)
    parser.add_argument("--min-exception-steps", type=int, default=0)
    parser.add_argument("--max-failure-examples", type=int, default=20)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.out is None:
        args.out = args.run_dir / "stage4_aligned_exception_v31_adversarial_audit.json"
    report = audit_run(args)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "ok": report["ok"],
                "totals": report["totals"],
                "initial_non_safe_fp_register_fraction": report["initial_non_safe_fp_register_fraction"],
                "oracles": report["oracles"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
