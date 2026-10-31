#!/usr/bin/env python3
"""Stage 3 PAN x86 complex-scalar runner.

This stage adds scalar operations whose semantics stress non-linear update
cartridges: byte extension, 64-bit base ALU/logical forms, bit scans, bit
test/modify, rotates through carry, multiply, safe no-fault divide,
divide-error exceptions, selected CPUID-gated bit-count/BMI1/BMI2 forms,
read-only memory-source byte/word extension/base-ALU/bit-scan/bit-count/BMI1/BMI2
slices, immediate/register-indexed memory-destination bit-test/modify slices,
and one-count plus selected variable-count memory-destination shift/rotate slices.
It is still a scoped substage, not full x86.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pan_x86_stages.stage0_x86_pan import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_DATE,
    EXP_ROOT,
    FLAGS,
    ResourceMonitor,
    hex_u,
    load_traces,
    validate_trace_schema,
    write_json,
    write_jsonl,
)


STAGE = "stage3_complex_scalar"
SUBSTAGE = "stage3_complex_scalar_base_alu_v21"
REGS = ("RAX", "RBX", "RCX", "RDX")
REG_INDEX = {name: i for i, name in enumerate(REGS)}
REG_X86_CODE = {"RAX": 0, "RCX": 1, "RDX": 2, "RBX": 3}
LOW8 = {"RAX": "al", "RBX": "bl", "RCX": "cl", "RDX": "dl"}
LOW16 = {"RAX": "ax", "RBX": "bx", "RCX": "cx", "RDX": "dx"}
CODE_BASE = 0x500000
MEM_BASE = 0x160000
MEM_SIZE = 256
MASK64 = (1 << 64) - 1
FLAG_MASKS = {
    "none": [0, 0, 0, 0],
    "cf": [1, 0, 0, 0],
    "zf": [0, 1, 0, 0],
    "cf_zf": [1, 1, 0, 0],
    "cf_of": [1, 0, 0, 1],
    "cf_zf_sf_of": [1, 1, 1, 1],
    "dynamic_shift_rotate": [0, 0, 0, 0],
}
CPUID_FEATURES = ("POPCNT", "LZCNT", "BMI1", "BMI2")
FLAG_INDEX = {name: i for i, name in enumerate(FLAGS)}


@dataclasses.dataclass(frozen=True)
class Stage3OpSpec:
    name: str
    iform: str
    family: str
    operands: str
    flag_policy: str
    expected_exception: Optional[str] = None


OPS: Tuple[Stage3OpSpec, ...] = (
    Stage3OpSpec("BSWAP_R64", "BSWAP_GPRv", "BSWAP", "r", "none"),
    Stage3OpSpec("MOVZX_R64_R8", "MOVZX_GPRv_GPR8", "DATAXFER", "rr8", "none"),
    Stage3OpSpec("MOVSX_R64_R8", "MOVSX_GPRv_GPR8", "DATAXFER", "rr8", "none"),
    Stage3OpSpec("MOVZX_R64_R16", "MOVZX_GPRv_GPR16", "DATAXFER", "rr16", "none"),
    Stage3OpSpec("MOVSX_R64_R16", "MOVSX_GPRv_GPR16", "DATAXFER", "rr16", "none"),
    Stage3OpSpec("MOVZX_R64_M8", "MOVZX_GPRv_MEMb", "DATAXFER", "r,m8", "none"),
    Stage3OpSpec("MOVSX_R64_M8", "MOVSX_GPRv_MEMb", "DATAXFER", "r,m8", "none"),
    Stage3OpSpec("MOVZX_R64_M16", "MOVZX_GPRv_MEMw", "DATAXFER", "r,m16", "none"),
    Stage3OpSpec("MOVSX_R64_M16", "MOVSX_GPRv_MEMw", "DATAXFER", "r,m16", "none"),
    Stage3OpSpec("ADD_R64_R64", "ADD_GPRv_GPRv_03", "BINARY", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("ADC_R64_R64", "ADC_GPRv_GPRv_13", "BINARY", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("SUB_R64_R64", "SUB_GPRv_GPRv_2B", "BINARY", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("SBB_R64_R64", "SBB_GPRv_GPRv_1B", "BINARY", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("CMP_R64_R64", "CMP_GPRv_GPRv_3B", "BINARY", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("AND_R64_R64", "AND_GPRv_GPRv_23", "LOGICAL", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("OR_R64_R64", "OR_GPRv_GPRv_0B", "LOGICAL", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("XOR_R64_R64", "XOR_GPRv_GPRv_33", "LOGICAL", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("TEST_R64_R64", "TEST_GPRv_GPRv", "LOGICAL", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("ADD_R64_M64", "ADD_GPRv_MEMv", "BINARY", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("ADC_R64_M64", "ADC_GPRv_MEMv", "BINARY", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("SUB_R64_M64", "SUB_GPRv_MEMv", "BINARY", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("SBB_R64_M64", "SBB_GPRv_MEMv", "BINARY", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("CMP_R64_M64", "CMP_GPRv_MEMv", "BINARY", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("AND_R64_M64", "AND_GPRv_MEMv", "LOGICAL", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("OR_R64_M64", "OR_GPRv_MEMv", "LOGICAL", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("XOR_R64_M64", "XOR_GPRv_MEMv", "LOGICAL", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("TEST_M64_R64", "TEST_MEMv_GPRv", "LOGICAL", "m64,r64_ro", "cf_zf_sf_of"),
    Stage3OpSpec("BSF_R64_R64", "BSF_GPRv_GPRv", "BITSCAN", "rr", "zf"),
    Stage3OpSpec("BSR_R64_R64", "BSR_GPRv_GPRv", "BITSCAN", "rr", "zf"),
    Stage3OpSpec("BSF_R64_M64", "BSF_GPRv_MEMv", "BITSCAN", "r,m64", "zf"),
    Stage3OpSpec("BSR_R64_M64", "BSR_GPRv_MEMv", "BITSCAN", "r,m64", "zf"),
    Stage3OpSpec("BT_R64_IMM8", "BT_GPRv_IMMb", "BITTEST", "ri", "cf"),
    Stage3OpSpec("BTS_R64_IMM8", "BTS_GPRv_IMMb", "BITTEST", "ri", "cf"),
    Stage3OpSpec("BTR_R64_IMM8", "BTR_GPRv_IMMb", "BITTEST", "ri", "cf"),
    Stage3OpSpec("BTC_R64_IMM8", "BTC_GPRv_IMMb", "BITTEST", "ri", "cf"),
    Stage3OpSpec("BT_M64_IMM8", "BT_MEMv_IMMb", "BITTEST", "m64,imm8", "cf"),
    Stage3OpSpec("BTS_M64_IMM8", "BTS_MEMv_IMMb", "BITTEST", "m64,imm8", "cf"),
    Stage3OpSpec("BTR_M64_IMM8", "BTR_MEMv_IMMb", "BITTEST", "m64,imm8", "cf"),
    Stage3OpSpec("BTC_M64_IMM8", "BTC_MEMv_IMMb", "BITTEST", "m64,imm8", "cf"),
    Stage3OpSpec("BT_M64_R64", "BT_MEMv_GPRv", "BITTEST", "m64,r64", "cf"),
    Stage3OpSpec("BTS_M64_R64", "BTS_MEMv_GPRv", "BITTEST", "m64,r64", "cf"),
    Stage3OpSpec("BTR_M64_R64", "BTR_MEMv_GPRv", "BITTEST", "m64,r64", "cf"),
    Stage3OpSpec("BTC_M64_R64", "BTC_MEMv_GPRv", "BITTEST", "m64,r64", "cf"),
    Stage3OpSpec("RCL_R64_1", "RCL_GPRv_ONE", "ROTATE", "r", "cf_of"),
    Stage3OpSpec("RCR_R64_1", "RCR_GPRv_ONE", "ROTATE", "r", "cf_of"),
    Stage3OpSpec("RCL_M64_1", "RCL_MEMv_ONE", "ROTATE", "m64", "cf_of"),
    Stage3OpSpec("RCR_M64_1", "RCR_MEMv_ONE", "ROTATE", "m64", "cf_of"),
    Stage3OpSpec("RCL_R64_CL", "RCL_GPRv_CL", "ROTATE", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("RCR_R64_CL", "RCR_GPRv_CL", "ROTATE", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("RCL_M64_CL", "RCL_MEMv_CL", "ROTATE", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("RCR_M64_CL", "RCR_MEMv_CL", "ROTATE", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("RCL_R64_IMM8", "RCL_GPRv_IMMb", "ROTATE", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("RCR_R64_IMM8", "RCR_GPRv_IMMb", "ROTATE", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("RCL_M64_IMM8", "RCL_MEMv_IMMb", "ROTATE", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("RCR_M64_IMM8", "RCR_MEMv_IMMb", "ROTATE", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("ROL_R64_1", "ROL_GPRv_ONE", "ROTATE", "r", "cf_of"),
    Stage3OpSpec("ROR_R64_1", "ROR_GPRv_ONE", "ROTATE", "r", "cf_of"),
    Stage3OpSpec("ROL_M64_1", "ROL_MEMv_ONE", "ROTATE", "m64", "cf_of"),
    Stage3OpSpec("ROR_M64_1", "ROR_MEMv_ONE", "ROTATE", "m64", "cf_of"),
    Stage3OpSpec("SHL_R64_1", "SHL_GPRv_ONE_D1r4", "SHIFT", "r", "cf_zf_sf_of"),
    Stage3OpSpec("SHR_R64_1", "SHR_GPRv_ONE", "SHIFT", "r", "cf_zf_sf_of"),
    Stage3OpSpec("SAR_R64_1", "SAR_GPRv_ONE", "SHIFT", "r", "cf_zf_sf_of"),
    Stage3OpSpec("SHL_M64_1", "SHL_MEMv_ONE_D1r4", "SHIFT", "m64", "cf_zf_sf_of"),
    Stage3OpSpec("SHR_M64_1", "SHR_MEMv_ONE", "SHIFT", "m64", "cf_zf_sf_of"),
    Stage3OpSpec("SAR_M64_1", "SAR_MEMv_ONE", "SHIFT", "m64", "cf_zf_sf_of"),
    Stage3OpSpec("ROL_R64_CL", "ROL_GPRv_CL", "ROTATE", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("ROR_R64_CL", "ROR_GPRv_CL", "ROTATE", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("ROL_M64_CL", "ROL_MEMv_CL", "ROTATE", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("ROR_M64_CL", "ROR_MEMv_CL", "ROTATE", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("SHL_R64_CL", "SHL_GPRv_CL_D3r4", "SHIFT", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("SHR_R64_CL", "SHR_GPRv_CL", "SHIFT", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("SAR_R64_CL", "SAR_GPRv_CL", "SHIFT", "r,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("SHL_M64_CL", "SHL_MEMv_CL_D3r4", "SHIFT", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("SHR_M64_CL", "SHR_MEMv_CL", "SHIFT", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("SAR_M64_CL", "SAR_MEMv_CL", "SHIFT", "m64,cl", "dynamic_shift_rotate"),
    Stage3OpSpec("ROL_R64_IMM8", "ROL_GPRv_IMMb", "ROTATE", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("ROR_R64_IMM8", "ROR_GPRv_IMMb", "ROTATE", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("ROL_M64_IMM8", "ROL_MEMv_IMMb", "ROTATE", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("ROR_M64_IMM8", "ROR_MEMv_IMMb", "ROTATE", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("SHL_R64_IMM8", "SHL_GPRv_IMMb_C1r4", "SHIFT", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("SHR_R64_IMM8", "SHR_GPRv_IMMb", "SHIFT", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("SAR_R64_IMM8", "SAR_GPRv_IMMb", "SHIFT", "r,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("SHL_M64_IMM8", "SHL_MEMv_IMMb_C1r4", "SHIFT", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("SHR_M64_IMM8", "SHR_MEMv_IMMb", "SHIFT", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("SAR_M64_IMM8", "SAR_MEMv_IMMb", "SHIFT", "m64,imm8", "dynamic_shift_rotate"),
    Stage3OpSpec("IMUL_R64_R64", "IMUL_GPRv_GPRv", "MUL", "rr", "cf_of"),
    Stage3OpSpec("IMUL_R64_R64_IMM8", "IMUL_GPRv_GPRv_IMMb", "MUL", "rri", "cf_of"),
    Stage3OpSpec("MUL_R64", "MUL_GPRv", "MUL", "implicit_rdx_rax_r", "cf_of"),
    Stage3OpSpec("IMUL_ONE_R64", "IMUL_GPRv", "MUL", "implicit_rdx_rax_r", "cf_of"),
    Stage3OpSpec("DIV_R64", "DIV_GPRv", "DIV", "implicit_rdx_rax_r", "none"),
    Stage3OpSpec("IDIV_R64", "IDIV_GPRv", "DIV", "implicit_rdx_rax_r", "none"),
    Stage3OpSpec("DIV_R64_DE", "DIV_GPRv", "DIV_EXCEPTION", "implicit_rdx_rax_r", "none", "#DE"),
    Stage3OpSpec("IDIV_R64_DE", "IDIV_GPRv", "DIV_EXCEPTION", "implicit_rdx_rax_r", "none", "#DE"),
    Stage3OpSpec("POPCNT_R64_R64", "POPCNT_GPRv_GPRv", "BITCOUNT", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("LZCNT_R64_R64", "LZCNT_GPRv_GPRv", "BITCOUNT", "rr", "cf_zf"),
    Stage3OpSpec("TZCNT_R64_R64", "TZCNT_GPRv_GPRv", "BITCOUNT", "rr", "cf_zf"),
    Stage3OpSpec("POPCNT_R64_M64", "POPCNT_GPRv_MEMv", "BITCOUNT", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("LZCNT_R64_M64", "LZCNT_GPRv_MEMv", "BITCOUNT", "r,m64", "cf_zf"),
    Stage3OpSpec("TZCNT_R64_M64", "TZCNT_GPRv_MEMv", "BITCOUNT", "r,m64", "cf_zf"),
    Stage3OpSpec("ANDN_R64_R64_R64", "ANDN_GPR64q_GPR64q_GPR64q", "BMI1", "rrr", "cf_zf_sf_of"),
    Stage3OpSpec("ANDN_R64_R64_M64", "ANDN_GPR64q_GPR64q_MEMq", "BMI1", "rr,m64", "cf_zf_sf_of"),
    Stage3OpSpec("BEXTR_R64_R64_R64", "BEXTR_GPR64q_GPR64q_GPR64q", "BMI1", "rrr_control", "cf_zf_sf_of"),
    Stage3OpSpec("BEXTR_R64_M64_R64", "BEXTR_GPR64q_MEMq_GPR64q", "BMI1", "r,m64,r_control", "cf_zf_sf_of"),
    Stage3OpSpec("BLSI_R64_R64", "BLSI_GPR64q_GPR64q", "BMI1", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("BLSI_R64_M64", "BLSI_GPR64q_MEMq", "BMI1", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("BLSMSK_R64_R64", "BLSMSK_GPR64q_GPR64q", "BMI1", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("BLSMSK_R64_M64", "BLSMSK_GPR64q_MEMq", "BMI1", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("BLSR_R64_R64", "BLSR_GPR64q_GPR64q", "BMI1", "rr", "cf_zf_sf_of"),
    Stage3OpSpec("BLSR_R64_M64", "BLSR_GPR64q_MEMq", "BMI1", "r,m64", "cf_zf_sf_of"),
    Stage3OpSpec("SHLX_R64_R64_R64", "SHLX_GPR64q_GPR64q_GPR64q", "BMI2", "rrr_count", "none"),
    Stage3OpSpec("SHLX_R64_M64_R64", "SHLX_GPR64q_MEMq_GPR64q", "BMI2", "r,m64,r_count", "none"),
    Stage3OpSpec("SHRX_R64_R64_R64", "SHRX_GPR64q_GPR64q_GPR64q", "BMI2", "rrr_count", "none"),
    Stage3OpSpec("SHRX_R64_M64_R64", "SHRX_GPR64q_MEMq_GPR64q", "BMI2", "r,m64,r_count", "none"),
    Stage3OpSpec("SARX_R64_R64_R64", "SARX_GPR64q_GPR64q_GPR64q", "BMI2", "rrr_count", "none"),
    Stage3OpSpec("SARX_R64_M64_R64", "SARX_GPR64q_MEMq_GPR64q", "BMI2", "r,m64,r_count", "none"),
    Stage3OpSpec("RORX_R64_R64_IMM8", "RORX_GPR64q_GPR64q_IMMb", "BMI2", "rri", "none"),
    Stage3OpSpec("RORX_R64_M64_IMM8", "RORX_GPR64q_MEMq_IMMb", "BMI2", "r,m64,imm8", "none"),
    Stage3OpSpec("BZHI_R64_R64_R64", "BZHI_GPR64q_GPR64q_GPR64q", "BMI2", "rrr_index", "cf_zf"),
    Stage3OpSpec("BZHI_R64_M64_R64", "BZHI_GPR64q_MEMq_GPR64q", "BMI2", "r,m64,r_index", "cf_zf"),
    Stage3OpSpec("MULX_R64_R64_R64", "MULX_GPR64q_GPR64q_GPR64q", "BMI2", "mulx_high_low_src", "none"),
    Stage3OpSpec("MULX_R64_R64_M64", "MULX_GPR64q_GPR64q_MEMq", "BMI2", "mulx_high_low_m64", "none"),
    Stage3OpSpec("PDEP_R64_R64_R64", "PDEP_GPR64q_GPR64q_GPR64q", "BMI2", "rrr_mask", "none"),
    Stage3OpSpec("PDEP_R64_R64_M64", "PDEP_GPR64q_GPR64q_MEMq", "BMI2", "rr,m64", "none"),
    Stage3OpSpec("PEXT_R64_R64_R64", "PEXT_GPR64q_GPR64q_GPR64q", "BMI2", "rrr_mask", "none"),
    Stage3OpSpec("PEXT_R64_R64_M64", "PEXT_GPR64q_GPR64q_MEMq", "BMI2", "rr,m64", "none"),
)
OP_INDEX = {op.name: i for i, op in enumerate(OPS)}
BASE_ALU_OPS = {
    "ADD_R64_R64",
    "ADC_R64_R64",
    "SUB_R64_R64",
    "SBB_R64_R64",
    "CMP_R64_R64",
    "AND_R64_R64",
    "OR_R64_R64",
    "XOR_R64_R64",
    "TEST_R64_R64",
    "ADD_R64_M64",
    "ADC_R64_M64",
    "SUB_R64_M64",
    "SBB_R64_M64",
    "CMP_R64_M64",
    "AND_R64_M64",
    "OR_R64_M64",
    "XOR_R64_M64",
    "TEST_M64_R64",
}
BASE_ALU_BINARY_OPS = {
    "ADD_R64_R64",
    "ADC_R64_R64",
    "SUB_R64_R64",
    "SBB_R64_R64",
    "CMP_R64_R64",
    "ADD_R64_M64",
    "ADC_R64_M64",
    "SUB_R64_M64",
    "SBB_R64_M64",
    "CMP_R64_M64",
}
BASE_ALU_LOGICAL_OPS = BASE_ALU_OPS - BASE_ALU_BINARY_OPS
BASE_ALU_NO_WRITE_OPS = {"CMP_R64_R64", "CMP_R64_M64", "TEST_R64_R64", "TEST_M64_R64"}
MEMORY_SOURCE_OPERANDS = {
    "r,m8",
    "r,m16",
    "r,m64",
    "rr,m64",
    "r,m64,r_control",
    "r,m64,r_count",
    "r,m64,imm8",
    "r,m64,r_index",
    "mulx_high_low_m64",
}
MEMORY_READ_ONLY_OPERANDS = {
    "m64,r64_ro",
}
MEMORY_DESTINATION_OPERANDS = {
    "m64",
    "m64,imm8",
    "m64,r64",
    "m64,cl",
}
MEMORY_OPERANDS = MEMORY_SOURCE_OPERANDS | MEMORY_DESTINATION_OPERANDS | MEMORY_READ_ONLY_OPERANDS
MEMORY_BITTEST_OPS = {
    "BT_M64_IMM8",
    "BTS_M64_IMM8",
    "BTR_M64_IMM8",
    "BTC_M64_IMM8",
    "BT_M64_R64",
    "BTS_M64_R64",
    "BTR_M64_R64",
    "BTC_M64_R64",
}
MEMORY_BITTEST_WRITE_OPS = {
    "BTS_M64_IMM8",
    "BTR_M64_IMM8",
    "BTC_M64_IMM8",
    "BTS_M64_R64",
    "BTR_M64_R64",
    "BTC_M64_R64",
}
MEMORY_ROTATE_OPS = {
    "RCL_M64_1",
    "RCR_M64_1",
    "RCL_M64_CL",
    "RCR_M64_CL",
    "RCL_M64_IMM8",
    "RCR_M64_IMM8",
    "ROL_M64_1",
    "ROR_M64_1",
    "ROL_M64_CL",
    "ROR_M64_CL",
    "ROL_M64_IMM8",
    "ROR_M64_IMM8",
}
MEMORY_SHIFT_OPS = {
    "SHL_M64_1",
    "SHR_M64_1",
    "SAR_M64_1",
    "SHL_M64_CL",
    "SHR_M64_CL",
    "SAR_M64_CL",
    "SHL_M64_IMM8",
    "SHR_M64_IMM8",
    "SAR_M64_IMM8",
}
MEMORY_WRITE_OPS = MEMORY_BITTEST_WRITE_OPS | MEMORY_ROTATE_OPS | MEMORY_SHIFT_OPS

SHIFT_ROTATE_OPS = {
    "RCL_R64_1",
    "RCR_R64_1",
    "RCL_M64_1",
    "RCR_M64_1",
    "RCL_R64_CL",
    "RCR_R64_CL",
    "RCL_M64_CL",
    "RCR_M64_CL",
    "RCL_R64_IMM8",
    "RCR_R64_IMM8",
    "RCL_M64_IMM8",
    "RCR_M64_IMM8",
    "ROL_R64_1",
    "ROR_R64_1",
    "ROL_M64_1",
    "ROR_M64_1",
    "SHL_R64_1",
    "SHR_R64_1",
    "SAR_R64_1",
    "SHL_M64_1",
    "SHR_M64_1",
    "SAR_M64_1",
    "ROL_R64_CL",
    "ROR_R64_CL",
    "ROL_M64_CL",
    "ROR_M64_CL",
    "SHL_R64_CL",
    "SHR_R64_CL",
    "SAR_R64_CL",
    "SHL_M64_CL",
    "SHR_M64_CL",
    "SAR_M64_CL",
    "ROL_R64_IMM8",
    "ROR_R64_IMM8",
    "ROL_M64_IMM8",
    "ROR_M64_IMM8",
    "SHL_R64_IMM8",
    "SHR_R64_IMM8",
    "SAR_R64_IMM8",
    "SHL_M64_IMM8",
    "SHR_M64_IMM8",
    "SAR_M64_IMM8",
}


def u64(value: int) -> int:
    return int(value) & MASK64


def s64(value: int) -> int:
    value &= MASK64
    return value - (1 << 64) if value & (1 << 63) else value


def sign_extend(value: int, bits: int) -> int:
    mask = (1 << bits) - 1
    value &= mask
    sign = 1 << (bits - 1)
    return (value ^ sign) - sign


def add_flags64(a: int, b: int, result: int, carry_in: int = 0) -> List[int]:
    full = (a & MASK64) + (b & MASK64) + int(carry_in)
    result &= MASK64
    sign = 1 << 63
    cf = int(full > MASK64)
    zf = int(result == 0)
    sf = int(bool(result & sign))
    of = int(((~(a ^ b) & (a ^ result)) & sign) != 0)
    return [cf, zf, sf, of]


def sub_flags64(a: int, b: int, result: int, borrow_in: int = 0) -> List[int]:
    rhs = (b & MASK64) + int(borrow_in)
    result &= MASK64
    sign = 1 << 63
    cf = int((a & MASK64) < rhs)
    zf = int(result == 0)
    sf = int(bool(result & sign))
    of = int((((a ^ rhs) & (a ^ result)) & sign) != 0)
    return [cf, zf, sf, of]


def logical_flags64(result: int) -> List[int]:
    result &= MASK64
    return [0, int(result == 0), int(bool(result & (1 << 63))), 0]


def trailing_zero_count(value: int, width: int = 64) -> int:
    value &= (1 << width) - 1
    if value == 0:
        return width
    return (value & -value).bit_length() - 1


def leading_zero_count(value: int, width: int = 64) -> int:
    value &= (1 << width) - 1
    if value == 0:
        return width
    return width - value.bit_length()


def rotate_right(value: int, count: int, width: int = 64) -> int:
    count %= width
    value &= (1 << width) - 1
    if count == 0:
        return value
    return ((value >> count) | (value << (width - count))) & ((1 << width) - 1)


def parallel_deposit(src: int, mask: int) -> int:
    result = 0
    src_bit = 0
    bit = 1
    mask &= MASK64
    while bit <= MASK64:
        if mask & bit:
            if src & (1 << src_bit):
                result |= bit
            src_bit += 1
        bit <<= 1
    return result & MASK64


def parallel_extract(src: int, mask: int) -> int:
    result = 0
    dst_bit = 0
    bit = 1
    mask &= MASK64
    src &= MASK64
    while bit <= MASK64:
        if mask & bit:
            if src & bit:
                result |= 1 << dst_bit
            dst_bit += 1
        bit <<= 1
    return result & MASK64


def shift_rotate_count(op_name: str, regs: Sequence[int], imm: int) -> int:
    if op_name.endswith("_1"):
        return 1
    if op_name.endswith("_CL"):
        return int(regs[REG_INDEX["RCX"]]) & 0xFF
    if op_name.endswith("_IMM8"):
        return int(imm) & 0xFF
    raise ValueError(op_name)


def shift_rotate_base(op_name: str) -> str:
    return op_name.split("_", 1)[0]


def shift_rotate_flag_mask(base: str, count: int) -> List[int]:
    if count == 0:
        return [0, 0, 0, 0]
    if base in {"RCL", "RCR", "ROL", "ROR"}:
        return [1, 0, 0, int(count == 1)]
    if base in {"SHL", "SHR", "SAR"}:
        return [1, 1, 1, int(count == 1)]
    raise ValueError(base)


def apply_shift_rotate(op_name: str, target: int, flags: Sequence[int], count: int) -> Tuple[int, List[int], List[int]]:
    value = u64(target)
    out_flags = [int(x) for x in flags]
    base = shift_rotate_base(op_name)
    if base in {"RCL", "RCR"}:
        effective_count = count & 0x3F
    else:
        effective_count = count & 0x3F
    flag_mask = shift_rotate_flag_mask(base, effective_count)
    if effective_count == 0:
        return value, out_flags, flag_mask
    if base == "RCL":
        old_cf = out_flags[0]
        ring = (old_cf << 64) | value
        rotated = ((ring << effective_count) | (ring >> (65 - effective_count))) & ((1 << 65) - 1)
        result = rotated & MASK64
        new_cf = (rotated >> 64) & 1
        out_flags[0] = new_cf
        if effective_count == 1:
            out_flags[3] = ((result >> 63) & 1) ^ new_cf
    elif base == "RCR":
        old_cf = out_flags[0]
        ring = (old_cf << 64) | value
        rotated = ((ring >> effective_count) | (ring << (65 - effective_count))) & ((1 << 65) - 1)
        result = rotated & MASK64
        new_cf = (rotated >> 64) & 1
        out_flags[0] = new_cf
        if effective_count == 1:
            out_flags[3] = ((result >> 63) & 1) ^ ((result >> 62) & 1)
    elif base == "ROL":
        result = ((value << effective_count) | (value >> (64 - effective_count))) & MASK64
        new_cf = result & 1
        out_flags[0] = new_cf
        if effective_count == 1:
            out_flags[3] = ((result >> 63) & 1) ^ new_cf
    elif base == "ROR":
        result = ((value >> effective_count) | (value << (64 - effective_count))) & MASK64
        new_cf = (result >> 63) & 1
        out_flags[0] = new_cf
        if effective_count == 1:
            out_flags[3] = ((result >> 63) & 1) ^ ((result >> 62) & 1)
    elif base == "SHL":
        new_cf = (value >> (64 - effective_count)) & 1
        result = (value << effective_count) & MASK64
        out_flags[0] = new_cf
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        if effective_count == 1:
            out_flags[3] = ((result >> 63) & 1) ^ new_cf
    elif base == "SHR":
        new_cf = (value >> (effective_count - 1)) & 1
        result = value >> effective_count
        out_flags[0] = new_cf
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        if effective_count == 1:
            out_flags[3] = (value >> 63) & 1
    elif base == "SAR":
        new_cf = (value >> (effective_count - 1)) & 1
        result = u64(s64(value) >> effective_count)
        out_flags[0] = new_cf
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        if effective_count == 1:
            out_flags[3] = 0
    else:
        raise ValueError(op_name)
    return u64(result), out_flags, flag_mask


def bit_test_index(op: Stage3OpSpec, regs: Sequence[int], src: int, imm: int) -> int:
    if op.operands == "m64,r64":
        return int(regs[src]) & 63
    return int(imm) & 63


def extension_for_op(op: Stage3OpSpec) -> str:
    if op.name in {"POPCNT_R64_R64", "POPCNT_R64_M64"}:
        return "POPCNT"
    if op.name in {"LZCNT_R64_R64", "LZCNT_R64_M64"}:
        return "LZCNT"
    if op.name in {
        "TZCNT_R64_R64",
        "TZCNT_R64_M64",
        "ANDN_R64_R64_R64",
        "ANDN_R64_R64_M64",
        "BEXTR_R64_R64_R64",
        "BEXTR_R64_M64_R64",
        "BLSI_R64_R64",
        "BLSI_R64_M64",
        "BLSMSK_R64_R64",
        "BLSMSK_R64_M64",
        "BLSR_R64_R64",
        "BLSR_R64_M64",
    }:
        return "BMI1"
    if op.name in {
        "SHLX_R64_R64_R64",
        "SHLX_R64_M64_R64",
        "SHRX_R64_R64_R64",
        "SHRX_R64_M64_R64",
        "SARX_R64_R64_R64",
        "SARX_R64_M64_R64",
        "RORX_R64_R64_IMM8",
        "RORX_R64_M64_IMM8",
        "BZHI_R64_R64_R64",
        "BZHI_R64_M64_R64",
        "MULX_R64_R64_R64",
        "MULX_R64_R64_M64",
        "PDEP_R64_R64_R64",
        "PDEP_R64_R64_M64",
        "PEXT_R64_R64_R64",
        "PEXT_R64_R64_M64",
    }:
        return "BMI2"
    return "BASE"


def trunc_div(numer: int, denom: int) -> int:
    q = abs(numer) // abs(denom)
    return -q if (numer < 0) ^ (denom < 0) else q


def flags_dict(flags: Sequence[int]) -> Dict[str, int]:
    return {name: int(flags[i]) for i, name in enumerate(FLAGS)}


def undefined_flag_names(op: Stage3OpSpec, flag_mask: Sequence[int]) -> List[str]:
    if op.flag_policy == "none":
        return []
    if op.flag_policy == "dynamic_shift_rotate":
        if not any(int(x) for x in flag_mask):
            return []
        base = shift_rotate_base(op.name)
        if base in {"ROL", "ROR", "RCL", "RCR", "SHL", "SHR", "SAR"} and not int(flag_mask[3]):
            return ["OF"]
        return []
    return [FLAGS[i] for i, bit in enumerate(flag_mask) if not bit]


def flag_reads(op: Stage3OpSpec) -> List[str]:
    if op.name.startswith(("ADC_", "SBB_", "RCL_", "RCR_")):
        return ["CF"]
    return []


def flags_defined_after(before_defined: Sequence[bool], flag_mask: Sequence[int], undefined_flags: Sequence[str]) -> List[bool]:
    out = [bool(x) for x in before_defined]
    for index, bit in enumerate(flag_mask):
        if int(bit):
            out[index] = True
    for name in undefined_flags:
        if name in FLAG_INDEX:
            out[FLAG_INDEX[name]] = False
    return out


def flag_reads_are_defined(op: Stage3OpSpec, flag_defined: Sequence[bool]) -> bool:
    return all(bool(flag_defined[FLAG_INDEX[name]]) for name in flag_reads(op))


def memory_digest(memory: Optional[Sequence[int]] = None) -> str:
    data = b"" if memory is None else bytes(int(x) & 0xFF for x in memory)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def state_record(
    regs: Sequence[int],
    flags: Sequence[int],
    rip: int,
    memory: Optional[Sequence[int]] = None,
    exception: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "rip": hex_u(rip, 64),
        "gpr": {reg: hex_u(int(regs[i]), 64) for i, reg in enumerate(REGS)},
        "rflags": flags_dict(flags),
        "memory_digest": memory_digest(memory),
        "exception": exception,
    }


def modrm(reg: int, rm: int) -> int:
    return 0xC0 | ((reg & 7) << 3) | (rm & 7)


def modrm_abs(reg: int) -> int:
    return 0x04 | ((reg & 7) << 3)


def vex3_w1_vvvv(reg: int, pp: int = 0) -> int:
    return 0x80 | (((~reg) & 0xF) << 3) | (pp & 3)


def x86_reg(index: int) -> int:
    return REG_X86_CODE[REGS[index]]


def encode_abs_m64_opcode(prefix: Sequence[int], opcode: int, reg: int, mem_addr: int) -> List[int]:
    b = list(prefix) + [0x0F, opcode, modrm_abs(x86_reg(reg)), 0x25]
    b.extend(int(mem_addr).to_bytes(4, "little"))
    return b


def encode_abs_m64_onebyte_opcode(prefix: Sequence[int], opcode: int, reg: int, mem_addr: int) -> List[int]:
    b = list(prefix) + [opcode, modrm_abs(x86_reg(reg)), 0x25]
    b.extend(int(mem_addr).to_bytes(4, "little"))
    return b


def encode_abs_m64_group_opcode(prefix: Sequence[int], opcode: int, group: int, mem_addr: int, imm8: Optional[int] = None) -> List[int]:
    b = list(prefix) + [0x0F, opcode, modrm_abs(group), 0x25]
    b.extend(int(mem_addr).to_bytes(4, "little"))
    if imm8 is not None:
        b.append(imm8 & 0xFF)
    return b


def encode_abs_m64_onebyte_group_opcode(prefix: Sequence[int], opcode: int, group: int, mem_addr: int, imm8: Optional[int] = None) -> List[int]:
    b = list(prefix) + [opcode, modrm_abs(group), 0x25]
    b.extend(int(mem_addr).to_bytes(4, "little"))
    if imm8 is not None:
        b.append(imm8 & 0xFF)
    return b


def encode_abs_m64_vex3(map_select: int, vex3: int, opcode: int, reg: int, mem_addr: int, imm8: Optional[int] = None) -> List[int]:
    b = [0xC4, map_select, vex3, opcode, modrm_abs(reg), 0x25]
    b.extend(int(mem_addr).to_bytes(4, "little"))
    if imm8 is not None:
        b.append(imm8 & 0xFF)
    return b


def encode_x86_bytes(op: Stage3OpSpec, dst: int, src: int, imm: int, mem_addr: int = 0) -> str:
    if op.name == "BSWAP_R64":
        b = [0x48, 0x0F, 0xC8 + x86_reg(dst)]
    elif op.name == "MOVZX_R64_R8":
        b = [0x48, 0x0F, 0xB6, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "MOVSX_R64_R8":
        b = [0x48, 0x0F, 0xBE, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "MOVZX_R64_R16":
        b = [0x48, 0x0F, 0xB7, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "MOVSX_R64_R16":
        b = [0x48, 0x0F, 0xBF, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "MOVZX_R64_M8":
        b = encode_abs_m64_opcode([0x48], 0xB6, dst, mem_addr)
    elif op.name == "MOVSX_R64_M8":
        b = encode_abs_m64_opcode([0x48], 0xBE, dst, mem_addr)
    elif op.name == "MOVZX_R64_M16":
        b = encode_abs_m64_opcode([0x48], 0xB7, dst, mem_addr)
    elif op.name == "MOVSX_R64_M16":
        b = encode_abs_m64_opcode([0x48], 0xBF, dst, mem_addr)
    elif op.name in (
        "ADD_R64_R64",
        "ADC_R64_R64",
        "SUB_R64_R64",
        "SBB_R64_R64",
        "CMP_R64_R64",
        "AND_R64_R64",
        "OR_R64_R64",
        "XOR_R64_R64",
    ):
        opcode = {
            "ADD_R64_R64": 0x03,
            "ADC_R64_R64": 0x13,
            "SUB_R64_R64": 0x2B,
            "SBB_R64_R64": 0x1B,
            "CMP_R64_R64": 0x3B,
            "AND_R64_R64": 0x23,
            "OR_R64_R64": 0x0B,
            "XOR_R64_R64": 0x33,
        }[op.name]
        b = [0x48, opcode, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "TEST_R64_R64":
        b = [0x48, 0x85, modrm(x86_reg(src), x86_reg(dst))]
    elif op.name in (
        "ADD_R64_M64",
        "ADC_R64_M64",
        "SUB_R64_M64",
        "SBB_R64_M64",
        "CMP_R64_M64",
        "AND_R64_M64",
        "OR_R64_M64",
        "XOR_R64_M64",
    ):
        opcode = {
            "ADD_R64_M64": 0x03,
            "ADC_R64_M64": 0x13,
            "SUB_R64_M64": 0x2B,
            "SBB_R64_M64": 0x1B,
            "CMP_R64_M64": 0x3B,
            "AND_R64_M64": 0x23,
            "OR_R64_M64": 0x0B,
            "XOR_R64_M64": 0x33,
        }[op.name]
        b = encode_abs_m64_onebyte_opcode([0x48], opcode, dst, mem_addr)
    elif op.name == "TEST_M64_R64":
        b = encode_abs_m64_onebyte_opcode([0x48], 0x85, src, mem_addr)
    elif op.name == "BSF_R64_R64":
        b = [0x48, 0x0F, 0xBC, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "BSR_R64_R64":
        b = [0x48, 0x0F, 0xBD, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "BSF_R64_M64":
        b = encode_abs_m64_opcode([0x48], 0xBC, dst, mem_addr)
    elif op.name == "BSR_R64_M64":
        b = encode_abs_m64_opcode([0x48], 0xBD, dst, mem_addr)
    elif op.name in ("BT_R64_IMM8", "BTS_R64_IMM8", "BTR_R64_IMM8", "BTC_R64_IMM8"):
        group = {"BT_R64_IMM8": 4, "BTS_R64_IMM8": 5, "BTR_R64_IMM8": 6, "BTC_R64_IMM8": 7}[op.name]
        b = [0x48, 0x0F, 0xBA, modrm(group, x86_reg(dst)), imm & 0xFF]
    elif op.name in ("BT_M64_IMM8", "BTS_M64_IMM8", "BTR_M64_IMM8", "BTC_M64_IMM8"):
        group = {"BT_M64_IMM8": 4, "BTS_M64_IMM8": 5, "BTR_M64_IMM8": 6, "BTC_M64_IMM8": 7}[op.name]
        b = encode_abs_m64_group_opcode([0x48], 0xBA, group, mem_addr, imm & 0xFF)
    elif op.name in ("BT_M64_R64", "BTS_M64_R64", "BTR_M64_R64", "BTC_M64_R64"):
        opcode = {"BT_M64_R64": 0xA3, "BTS_M64_R64": 0xAB, "BTR_M64_R64": 0xB3, "BTC_M64_R64": 0xBB}[op.name]
        b = encode_abs_m64_opcode([0x48], opcode, src, mem_addr)
    elif op.name == "RCL_R64_1":
        b = [0x48, 0xD1, modrm(2, x86_reg(dst))]
    elif op.name == "RCR_R64_1":
        b = [0x48, 0xD1, modrm(3, x86_reg(dst))]
    elif op.name == "RCL_M64_1":
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xD1, 2, mem_addr)
    elif op.name == "RCR_M64_1":
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xD1, 3, mem_addr)
    elif op.name in ("RCL_R64_CL", "RCR_R64_CL"):
        group = {"RCL_R64_CL": 2, "RCR_R64_CL": 3}[op.name]
        b = [0x48, 0xD3, modrm(group, x86_reg(dst))]
    elif op.name in ("RCL_M64_CL", "RCR_M64_CL"):
        group = {"RCL_M64_CL": 2, "RCR_M64_CL": 3}[op.name]
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xD3, group, mem_addr)
    elif op.name in ("RCL_R64_IMM8", "RCR_R64_IMM8"):
        group = {"RCL_R64_IMM8": 2, "RCR_R64_IMM8": 3}[op.name]
        b = [0x48, 0xC1, modrm(group, x86_reg(dst)), imm & 0xFF]
    elif op.name in ("RCL_M64_IMM8", "RCR_M64_IMM8"):
        group = {"RCL_M64_IMM8": 2, "RCR_M64_IMM8": 3}[op.name]
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xC1, group, mem_addr, imm & 0xFF)
    elif op.name in ("ROL_R64_1", "ROR_R64_1", "SHL_R64_1", "SHR_R64_1", "SAR_R64_1"):
        group = {"ROL_R64_1": 0, "ROR_R64_1": 1, "SHL_R64_1": 4, "SHR_R64_1": 5, "SAR_R64_1": 7}[op.name]
        b = [0x48, 0xD1, modrm(group, x86_reg(dst))]
    elif op.name in ("ROL_M64_1", "ROR_M64_1", "SHL_M64_1", "SHR_M64_1", "SAR_M64_1"):
        group = {"ROL_M64_1": 0, "ROR_M64_1": 1, "SHL_M64_1": 4, "SHR_M64_1": 5, "SAR_M64_1": 7}[op.name]
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xD1, group, mem_addr)
    elif op.name in ("ROL_R64_CL", "ROR_R64_CL", "SHL_R64_CL", "SHR_R64_CL", "SAR_R64_CL"):
        group = {"ROL_R64_CL": 0, "ROR_R64_CL": 1, "SHL_R64_CL": 4, "SHR_R64_CL": 5, "SAR_R64_CL": 7}[op.name]
        b = [0x48, 0xD3, modrm(group, x86_reg(dst))]
    elif op.name in ("ROL_M64_CL", "ROR_M64_CL", "SHL_M64_CL", "SHR_M64_CL", "SAR_M64_CL"):
        group = {"ROL_M64_CL": 0, "ROR_M64_CL": 1, "SHL_M64_CL": 4, "SHR_M64_CL": 5, "SAR_M64_CL": 7}[op.name]
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xD3, group, mem_addr)
    elif op.name in ("ROL_R64_IMM8", "ROR_R64_IMM8", "SHL_R64_IMM8", "SHR_R64_IMM8", "SAR_R64_IMM8"):
        group = {"ROL_R64_IMM8": 0, "ROR_R64_IMM8": 1, "SHL_R64_IMM8": 4, "SHR_R64_IMM8": 5, "SAR_R64_IMM8": 7}[op.name]
        b = [0x48, 0xC1, modrm(group, x86_reg(dst)), imm & 0xFF]
    elif op.name in ("ROL_M64_IMM8", "ROR_M64_IMM8", "SHL_M64_IMM8", "SHR_M64_IMM8", "SAR_M64_IMM8"):
        group = {"ROL_M64_IMM8": 0, "ROR_M64_IMM8": 1, "SHL_M64_IMM8": 4, "SHR_M64_IMM8": 5, "SAR_M64_IMM8": 7}[op.name]
        b = encode_abs_m64_onebyte_group_opcode([0x48], 0xC1, group, mem_addr, imm & 0xFF)
    elif op.name == "IMUL_R64_R64":
        b = [0x48, 0x0F, 0xAF, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "IMUL_R64_R64_IMM8":
        b = [0x48, 0x6B, modrm(x86_reg(dst), x86_reg(src)), imm & 0xFF]
    elif op.name in ("MUL_R64", "IMUL_ONE_R64", "DIV_R64", "IDIV_R64", "DIV_R64_DE", "IDIV_R64_DE"):
        group = {"MUL_R64": 4, "IMUL_ONE_R64": 5, "DIV_R64": 6, "IDIV_R64": 7, "DIV_R64_DE": 6, "IDIV_R64_DE": 7}[op.name]
        b = [0x48, 0xF7, modrm(group, x86_reg(src))]
    elif op.name == "POPCNT_R64_R64":
        b = [0xF3, 0x48, 0x0F, 0xB8, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "LZCNT_R64_R64":
        b = [0xF3, 0x48, 0x0F, 0xBD, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "TZCNT_R64_R64":
        b = [0xF3, 0x48, 0x0F, 0xBC, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "POPCNT_R64_M64":
        b = encode_abs_m64_opcode([0xF3, 0x48], 0xB8, dst, mem_addr)
    elif op.name == "LZCNT_R64_M64":
        b = encode_abs_m64_opcode([0xF3, 0x48], 0xBD, dst, mem_addr)
    elif op.name == "TZCNT_R64_M64":
        b = encode_abs_m64_opcode([0xF3, 0x48], 0xBC, dst, mem_addr)
    elif op.name == "ANDN_R64_R64_R64":
        src2 = imm & 3
        b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(src)), 0xF2, modrm(x86_reg(dst), x86_reg(src2))]
    elif op.name == "ANDN_R64_R64_M64":
        b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(src)), 0xF2, x86_reg(dst), mem_addr)
    elif op.name == "BEXTR_R64_R64_R64":
        control = imm & 3
        b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(control)), 0xF7, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "BEXTR_R64_M64_R64":
        control = imm & 3
        b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(control)), 0xF7, x86_reg(dst), mem_addr)
    elif op.name in ("BLSI_R64_R64", "BLSMSK_R64_R64", "BLSR_R64_R64", "BLSI_R64_M64", "BLSMSK_R64_M64", "BLSR_R64_M64"):
        group = {
            "BLSR_R64_R64": 1,
            "BLSR_R64_M64": 1,
            "BLSMSK_R64_R64": 2,
            "BLSMSK_R64_M64": 2,
            "BLSI_R64_R64": 3,
            "BLSI_R64_M64": 3,
        }[op.name]
        if op.name.endswith("_M64"):
            b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(dst)), 0xF3, group, mem_addr)
        else:
            b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(dst)), 0xF3, modrm(group, x86_reg(src))]
    elif op.name in ("SHLX_R64_R64_R64", "SHRX_R64_R64_R64", "SARX_R64_R64_R64"):
        pp = {"SHLX_R64_R64_R64": 1, "SARX_R64_R64_R64": 2, "SHRX_R64_R64_R64": 3}[op.name]
        count = imm & 3
        b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(count), pp), 0xF7, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name in ("SHLX_R64_M64_R64", "SHRX_R64_M64_R64", "SARX_R64_M64_R64"):
        pp = {"SHLX_R64_M64_R64": 1, "SARX_R64_M64_R64": 2, "SHRX_R64_M64_R64": 3}[op.name]
        count = imm & 3
        b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(count), pp), 0xF7, x86_reg(dst), mem_addr)
    elif op.name == "RORX_R64_R64_IMM8":
        b = [0xC4, 0xE3, vex3_w1_vvvv(0, 3), 0xF0, modrm(x86_reg(dst), x86_reg(src)), imm & 0xFF]
    elif op.name == "RORX_R64_M64_IMM8":
        b = encode_abs_m64_vex3(0xE3, vex3_w1_vvvv(0, 3), 0xF0, x86_reg(dst), mem_addr, imm & 0xFF)
    elif op.name == "BZHI_R64_R64_R64":
        index = imm & 3
        b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(index), 0), 0xF5, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "BZHI_R64_M64_R64":
        index = imm & 3
        b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(index), 0), 0xF5, x86_reg(dst), mem_addr)
    elif op.name == "MULX_R64_R64_R64":
        low = imm & 3
        b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(low), 3), 0xF6, modrm(x86_reg(dst), x86_reg(src))]
    elif op.name == "MULX_R64_R64_M64":
        low = imm & 3
        b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(low), 3), 0xF6, x86_reg(dst), mem_addr)
    elif op.name in ("PDEP_R64_R64_R64", "PEXT_R64_R64_R64"):
        pp = 3 if op.name == "PDEP_R64_R64_R64" else 2
        mask = imm & 3
        b = [0xC4, 0xE2, vex3_w1_vvvv(x86_reg(src), pp), 0xF5, modrm(x86_reg(dst), x86_reg(mask))]
    elif op.name in ("PDEP_R64_R64_M64", "PEXT_R64_R64_M64"):
        pp = 3 if op.name == "PDEP_R64_R64_M64" else 2
        b = encode_abs_m64_vex3(0xE2, vex3_w1_vvvv(x86_reg(src), pp), 0xF5, x86_reg(dst), mem_addr)
    else:
        raise ValueError(op.name)
    return "".join(f"{x:02x}" for x in b)


def asm_string(op: Stage3OpSpec, dst: int, src: int, imm: int, mem_addr: int = 0) -> str:
    d = REGS[dst].lower()
    s = REGS[src].lower()
    if op.name == "BSWAP_R64":
        return f"bswap {d}"
    if op.name == "MOVZX_R64_R8":
        return f"movzx {d}, {LOW8[REGS[src]]}"
    if op.name == "MOVSX_R64_R8":
        return f"movsx {d}, {LOW8[REGS[src]]}"
    if op.name == "MOVZX_R64_R16":
        return f"movzx {d}, {LOW16[REGS[src]]}"
    if op.name == "MOVSX_R64_R16":
        return f"movsx {d}, {LOW16[REGS[src]]}"
    if op.name == "MOVZX_R64_M8":
        return f"movzx {d}, byte ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "MOVSX_R64_M8":
        return f"movsx {d}, byte ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "MOVZX_R64_M16":
        return f"movzx {d}, word ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "MOVSX_R64_M16":
        return f"movsx {d}, word ptr [{hex_u(mem_addr, 64)}]"
    if op.name in BASE_ALU_OPS:
        mnemonic = op.name.split("_")[0].lower()
        if op.name == "TEST_M64_R64":
            return f"test qword ptr [{hex_u(mem_addr, 64)}], {s}"
        if op.name.endswith("_M64"):
            return f"{mnemonic} {d}, qword ptr [{hex_u(mem_addr, 64)}]"
        return f"{mnemonic} {d}, {s}"
    if op.name == "BSF_R64_R64":
        return f"bsf {d}, {s}"
    if op.name == "BSR_R64_R64":
        return f"bsr {d}, {s}"
    if op.name == "BSF_R64_M64":
        return f"bsf {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "BSR_R64_M64":
        return f"bsr {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name in ("BT_R64_IMM8", "BTS_R64_IMM8", "BTR_R64_IMM8", "BTC_R64_IMM8"):
        return f"{op.name.split('_')[0].lower()} {d}, {imm & 63}"
    if op.name in ("BT_M64_IMM8", "BTS_M64_IMM8", "BTR_M64_IMM8", "BTC_M64_IMM8"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], {imm & 63}"
    if op.name in ("BT_M64_R64", "BTS_M64_R64", "BTR_M64_R64", "BTC_M64_R64"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], {s}"
    if op.name == "RCL_R64_1":
        return f"rcl {d}, 1"
    if op.name == "RCR_R64_1":
        return f"rcr {d}, 1"
    if op.name == "RCL_M64_1":
        return f"rcl qword ptr [{hex_u(mem_addr, 64)}], 1"
    if op.name == "RCR_M64_1":
        return f"rcr qword ptr [{hex_u(mem_addr, 64)}], 1"
    if op.name in ("RCL_R64_CL", "RCR_R64_CL"):
        return f"{op.name.split('_')[0].lower()} {d}, cl"
    if op.name in ("RCL_M64_CL", "RCR_M64_CL"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], cl"
    if op.name in ("RCL_R64_IMM8", "RCR_R64_IMM8"):
        return f"{op.name.split('_')[0].lower()} {d}, {imm & 0xFF}"
    if op.name in ("RCL_M64_IMM8", "RCR_M64_IMM8"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], {imm & 0xFF}"
    if op.name in ("ROL_R64_1", "ROR_R64_1", "SHL_R64_1", "SHR_R64_1", "SAR_R64_1"):
        return f"{op.name.split('_')[0].lower()} {d}, 1"
    if op.name in ("ROL_M64_1", "ROR_M64_1", "SHL_M64_1", "SHR_M64_1", "SAR_M64_1"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], 1"
    if op.name in ("ROL_R64_CL", "ROR_R64_CL", "SHL_R64_CL", "SHR_R64_CL", "SAR_R64_CL"):
        return f"{op.name.split('_')[0].lower()} {d}, cl"
    if op.name in ("ROL_M64_CL", "ROR_M64_CL", "SHL_M64_CL", "SHR_M64_CL", "SAR_M64_CL"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], cl"
    if op.name in ("ROL_R64_IMM8", "ROR_R64_IMM8", "SHL_R64_IMM8", "SHR_R64_IMM8", "SAR_R64_IMM8"):
        return f"{op.name.split('_')[0].lower()} {d}, {imm & 0xFF}"
    if op.name in ("ROL_M64_IMM8", "ROR_M64_IMM8", "SHL_M64_IMM8", "SHR_M64_IMM8", "SAR_M64_IMM8"):
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], {imm & 0xFF}"
    if op.name == "IMUL_R64_R64":
        return f"imul {d}, {s}"
    if op.name == "IMUL_R64_R64_IMM8":
        return f"imul {d}, {s}, {sign_extend(imm, 8)}"
    if op.name == "MUL_R64":
        return f"mul {s}"
    if op.name == "IMUL_ONE_R64":
        return f"imul {s}"
    if op.name in ("DIV_R64", "DIV_R64_DE"):
        return f"div {s}"
    if op.name in ("IDIV_R64", "IDIV_R64_DE"):
        return f"idiv {s}"
    if op.name == "POPCNT_R64_R64":
        return f"popcnt {d}, {s}"
    if op.name == "LZCNT_R64_R64":
        return f"lzcnt {d}, {s}"
    if op.name == "TZCNT_R64_R64":
        return f"tzcnt {d}, {s}"
    if op.name == "POPCNT_R64_M64":
        return f"popcnt {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "LZCNT_R64_M64":
        return f"lzcnt {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "TZCNT_R64_M64":
        return f"tzcnt {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "ANDN_R64_R64_R64":
        return f"andn {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "ANDN_R64_R64_M64":
        return f"andn {d}, {s}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "BEXTR_R64_R64_R64":
        return f"bextr {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "BEXTR_R64_M64_R64":
        return f"bextr {d}, qword ptr [{hex_u(mem_addr, 64)}], {REGS[imm & 3].lower()}"
    if op.name == "BLSI_R64_R64":
        return f"blsi {d}, {s}"
    if op.name == "BLSI_R64_M64":
        return f"blsi {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "BLSMSK_R64_R64":
        return f"blsmsk {d}, {s}"
    if op.name == "BLSMSK_R64_M64":
        return f"blsmsk {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "BLSR_R64_R64":
        return f"blsr {d}, {s}"
    if op.name == "BLSR_R64_M64":
        return f"blsr {d}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "SHLX_R64_R64_R64":
        return f"shlx {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "SHLX_R64_M64_R64":
        return f"shlx {d}, qword ptr [{hex_u(mem_addr, 64)}], {REGS[imm & 3].lower()}"
    if op.name == "SHRX_R64_R64_R64":
        return f"shrx {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "SHRX_R64_M64_R64":
        return f"shrx {d}, qword ptr [{hex_u(mem_addr, 64)}], {REGS[imm & 3].lower()}"
    if op.name == "SARX_R64_R64_R64":
        return f"sarx {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "SARX_R64_M64_R64":
        return f"sarx {d}, qword ptr [{hex_u(mem_addr, 64)}], {REGS[imm & 3].lower()}"
    if op.name == "RORX_R64_R64_IMM8":
        return f"rorx {d}, {s}, {imm & 63}"
    if op.name == "RORX_R64_M64_IMM8":
        return f"rorx {d}, qword ptr [{hex_u(mem_addr, 64)}], {imm & 63}"
    if op.name == "BZHI_R64_R64_R64":
        return f"bzhi {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "BZHI_R64_M64_R64":
        return f"bzhi {d}, qword ptr [{hex_u(mem_addr, 64)}], {REGS[imm & 3].lower()}"
    if op.name == "MULX_R64_R64_R64":
        return f"mulx {d}, {REGS[imm & 3].lower()}, {s}"
    if op.name == "MULX_R64_R64_M64":
        return f"mulx {d}, {REGS[imm & 3].lower()}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "PDEP_R64_R64_R64":
        return f"pdep {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "PDEP_R64_R64_M64":
        return f"pdep {d}, {s}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.name == "PEXT_R64_R64_R64":
        return f"pext {d}, {s}, {REGS[imm & 3].lower()}"
    if op.name == "PEXT_R64_R64_M64":
        return f"pext {d}, {s}, qword ptr [{hex_u(mem_addr, 64)}]"
    raise ValueError(op.name)


def decode_operands(op: Stage3OpSpec, dst: int, src: int, imm: int, mem_addr: int = 0) -> List[Dict[str, Any]]:
    if op.name in BASE_ALU_OPS:
        if op.name == "TEST_M64_R64":
            return [
                {
                    "name": "lhs",
                    "kind": "mem",
                    "address": hex_u(mem_addr, 64),
                    "width_bytes": 8,
                    "alignment": 8,
                },
                {"name": "rhs", "reg": REGS[src]},
            ]
        if op.name.endswith("_M64"):
            return [
                {"name": "dst", "reg": REGS[dst]},
                {
                    "name": "src",
                    "kind": "mem",
                    "address": hex_u(mem_addr, 64),
                    "width_bytes": 8,
                    "alignment": 8,
                },
            ]
        lhs_name = "lhs" if op.name in BASE_ALU_NO_WRITE_OPS else "dst"
        rhs_name = "rhs" if op.name in BASE_ALU_NO_WRITE_OPS else "src"
        return [
            {"name": lhs_name, "reg": REGS[dst]},
            {"name": rhs_name, "reg": REGS[src]},
        ]
    if op.operands in {"rr8", "rr16"}:
        width = 1 if op.operands == "rr8" else 2
        low_names = LOW8 if width == 1 else LOW16
        return [
            {"name": "dst", "reg": REGS[dst]},
            {
                "name": "src",
                "reg": REGS[src],
                "low_name": low_names[REGS[src]],
                "width_bytes": width,
            },
        ]
    if op.name in ("BT_M64_IMM8", "BTS_M64_IMM8", "BTR_M64_IMM8", "BTC_M64_IMM8"):
        return [
            {
                "name": "mem",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "imm", "value": imm & 63},
        ]
    if op.name in ("BT_M64_R64", "BTS_M64_R64", "BTR_M64_R64", "BTC_M64_R64"):
        return [
            {
                "name": "mem",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "index", "reg": REGS[src]},
        ]
    if op.name in (
        "ROL_R64_1",
        "ROR_R64_1",
        "SHL_R64_1",
        "SHR_R64_1",
        "SAR_R64_1",
        "RCL_R64_1",
        "RCR_R64_1",
    ):
        return [
            {"name": "dst", "reg": REGS[dst]},
            {"name": "imm", "value": 1},
        ]
    if op.name in ("RCL_R64_CL", "RCR_R64_CL", "ROL_R64_CL", "ROR_R64_CL", "SHL_R64_CL", "SHR_R64_CL", "SAR_R64_CL"):
        return [
            {"name": "dst", "reg": REGS[dst]},
            {"name": "count", "reg": "RCX", "low8": "CL"},
        ]
    if op.name in ("RCL_R64_IMM8", "RCR_R64_IMM8", "ROL_R64_IMM8", "ROR_R64_IMM8", "SHL_R64_IMM8", "SHR_R64_IMM8", "SAR_R64_IMM8"):
        return [
            {"name": "dst", "reg": REGS[dst]},
            {"name": "imm", "value": imm & 0xFF},
        ]
    if op.name in MEMORY_ROTATE_OPS | MEMORY_SHIFT_OPS:
        count_operand: Dict[str, Any]
        if op.name.endswith("_CL"):
            count_operand = {"name": "count", "reg": "RCX", "low8": "CL"}
        elif op.name.endswith("_IMM8"):
            count_operand = {"name": "imm", "value": imm & 0xFF}
        else:
            count_operand = {"name": "imm", "value": 1}
        return [
            {
                "name": "mem",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            count_operand,
        ]
    if op.name == "ANDN_R64_R64_M64":
        return [
            {"name": "dst", "reg": REGS[dst]},
            {"name": "src1", "reg": REGS[src]},
            {
                "name": "src2",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
        ]
    if op.name == "BEXTR_R64_M64_R64":
        return [
            {"name": "dst", "reg": REGS[dst]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "control", "reg": REGS[imm & 3]},
        ]
    if op.name in ("SHLX_R64_M64_R64", "SHRX_R64_M64_R64", "SARX_R64_M64_R64"):
        return [
            {"name": "dst", "reg": REGS[dst]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "count", "reg": REGS[imm & 3]},
        ]
    if op.name == "RORX_R64_M64_IMM8":
        return [
            {"name": "dst", "reg": REGS[dst]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "imm", "value": imm & 63},
        ]
    if op.name == "BZHI_R64_M64_R64":
        return [
            {"name": "dst", "reg": REGS[dst]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "index", "reg": REGS[imm & 3]},
        ]
    if op.name == "MULX_R64_R64_M64":
        return [
            {"name": "high", "reg": REGS[dst]},
            {"name": "low", "reg": REGS[imm & 3]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
            {"name": "implicit_multiplicand", "reg": "RDX"},
        ]
    if op.name in ("PDEP_R64_R64_M64", "PEXT_R64_R64_M64"):
        return [
            {"name": "dst", "reg": REGS[dst]},
            {"name": "src", "reg": REGS[src]},
            {
                "name": "mask",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "alignment": 8,
            },
        ]
    if op.operands in MEMORY_SOURCE_OPERANDS:
        width = memory_operand_width(op)
        return [
            {"name": "dst", "reg": REGS[dst]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "alignment": width,
            },
        ]
    if op.name == "MULX_R64_R64_R64":
        return [
            {"name": "high", "reg": REGS[dst]},
            {"name": "low", "reg": REGS[imm & 3]},
            {"name": "src", "reg": REGS[src]},
            {"name": "implicit_multiplicand", "reg": "RDX"},
        ]
    operands = [
        {"name": "dst", "reg": REGS[dst]},
        {"name": "src", "reg": REGS[src]},
    ]
    if op.name == "ANDN_R64_R64_R64":
        operands[1]["name"] = "src1"
        operands.append({"name": "src2", "reg": REGS[imm & 3]})
    elif op.name == "BEXTR_R64_R64_R64":
        operands.append({"name": "control", "reg": REGS[imm & 3]})
    elif op.name in ("SHLX_R64_R64_R64", "SHRX_R64_R64_R64", "SARX_R64_R64_R64"):
        operands.append({"name": "count", "reg": REGS[imm & 3]})
    elif op.name == "BZHI_R64_R64_R64":
        operands.append({"name": "index", "reg": REGS[imm & 3]})
    elif op.name in ("PDEP_R64_R64_R64", "PEXT_R64_R64_R64"):
        operands.append({"name": "mask", "reg": REGS[imm & 3]})
    else:
        operands.append({"name": "imm", "value": imm})
    return operands


def apply_op(
    regs: Sequence[int],
    flags: Sequence[int],
    op_idx: int,
    dst: int,
    src: int,
    imm: int,
    mem_value: Optional[int] = None,
) -> Tuple[List[int], List[int], List[int]]:
    op = OPS[op_idx]
    out = [u64(x) for x in regs]
    out_flags = [int(x) for x in flags]
    if op.expected_exception is not None:
        return out, out_flags, [0, 0, 0, 0]
    a = out[dst]
    b = u64(mem_value) if op.operands in MEMORY_SOURCE_OPERANDS and mem_value is not None else out[src]
    flag_mask = list(FLAG_MASKS[op.flag_policy])

    if op.name == "BSWAP_R64":
        out[dst] = int.from_bytes(a.to_bytes(8, "little"), "big")
    elif op.name in ("MOVZX_R64_R8", "MOVZX_R64_M8"):
        out[dst] = b & 0xFF
    elif op.name in ("MOVSX_R64_R8", "MOVSX_R64_M8"):
        out[dst] = u64(sign_extend(b, 8))
    elif op.name in ("MOVZX_R64_R16", "MOVZX_R64_M16"):
        out[dst] = b & 0xFFFF
    elif op.name in ("MOVSX_R64_R16", "MOVSX_R64_M16"):
        out[dst] = u64(sign_extend(b, 16))
    elif op.name in BASE_ALU_OPS:
        lhs = u64(mem_value) if op.name == "TEST_M64_R64" and mem_value is not None else a
        rhs = out[src] if op.name == "TEST_M64_R64" else b
        if op.name.startswith("ADD_"):
            result = u64(lhs + rhs)
            out_flags = add_flags64(lhs, rhs, result)
        elif op.name.startswith("ADC_"):
            carry = int(out_flags[0])
            result = u64(lhs + rhs + carry)
            out_flags = add_flags64(lhs, rhs, result, carry)
        elif op.name.startswith("SUB_") or op.name.startswith("CMP_"):
            result = u64(lhs - rhs)
            out_flags = sub_flags64(lhs, rhs, result)
        elif op.name.startswith("SBB_"):
            borrow = int(out_flags[0])
            result = u64(lhs - rhs - borrow)
            out_flags = sub_flags64(lhs, rhs, result, borrow)
        elif op.name.startswith("AND_") or op.name.startswith("TEST_"):
            result = u64(lhs & rhs)
            out_flags = logical_flags64(result)
        elif op.name.startswith("OR_"):
            result = u64(lhs | rhs)
            out_flags = logical_flags64(result)
        elif op.name.startswith("XOR_"):
            result = u64(lhs ^ rhs)
            out_flags = logical_flags64(result)
        else:
            raise ValueError(op.name)
        if op.name not in BASE_ALU_NO_WRITE_OPS:
            out[dst] = result
    elif op.name in ("BSF_R64_R64", "BSF_R64_M64"):
        if b == 0:
            out_flags[1] = 1
        else:
            out[dst] = (b & -b).bit_length() - 1
            out_flags[1] = 0
    elif op.name in ("BSR_R64_R64", "BSR_R64_M64"):
        if b == 0:
            out_flags[1] = 1
        else:
            out[dst] = b.bit_length() - 1
            out_flags[1] = 0
    elif op.name in (
        "BT_R64_IMM8",
        "BTS_R64_IMM8",
        "BTR_R64_IMM8",
        "BTC_R64_IMM8",
        *MEMORY_BITTEST_OPS,
    ):
        bit = bit_test_index(op, out, src, imm)
        bit_mask = 1 << bit
        target = u64(mem_value) if op.operands in MEMORY_DESTINATION_OPERANDS and mem_value is not None else a
        out_flags[0] = int(bool(target & bit_mask))
        if op.name == "BTS_R64_IMM8":
            out[dst] = a | bit_mask
        elif op.name == "BTR_R64_IMM8":
            out[dst] = a & ~bit_mask
        elif op.name == "BTC_R64_IMM8":
            out[dst] = a ^ bit_mask
    elif op.name in SHIFT_ROTATE_OPS:
        target = u64(mem_value) if op.operands.startswith("m64") and mem_value is not None else a
        count = shift_rotate_count(op.name, out, imm)
        result, out_flags, flag_mask = apply_shift_rotate(op.name, target, out_flags, count)
        if not op.operands.startswith("m64"):
            out[dst] = result
    elif op.name == "IMUL_R64_R64":
        product = s64(a) * s64(b)
        low = u64(product)
        out[dst] = low
        overflow = int(product != s64(low))
        out_flags[0] = overflow
        out_flags[3] = overflow
    elif op.name == "IMUL_R64_R64_IMM8":
        product = s64(b) * sign_extend(imm, 8)
        low = u64(product)
        out[dst] = low
        overflow = int(product != s64(low))
        out_flags[0] = overflow
        out_flags[3] = overflow
    elif op.name == "MUL_R64":
        product = out[0] * b
        out[0] = product & MASK64
        out[3] = (product >> 64) & MASK64
        overflow = int(out[3] != 0)
        out_flags[0] = overflow
        out_flags[3] = overflow
    elif op.name == "IMUL_ONE_R64":
        product = s64(out[0]) * s64(b)
        out[0] = u64(product)
        out[3] = (product >> 64) & MASK64
        overflow = int(product != s64(out[0]))
        out_flags[0] = overflow
        out_flags[3] = overflow
    elif op.name == "DIV_R64":
        dividend = (out[3] << 64) | out[0]
        if b == 0:
            raise ZeroDivisionError("generated DIV divisor was zero")
        out[0] = dividend // b
        out[3] = dividend % b
    elif op.name == "IDIV_R64":
        dividend = (out[3] << 64) | out[0]
        if dividend & (1 << 127):
            dividend -= 1 << 128
        divisor = s64(b)
        if divisor == 0:
            raise ZeroDivisionError("generated IDIV divisor was zero")
        q = trunc_div(dividend, divisor)
        r = dividend - q * divisor
        out[0] = u64(q)
        out[3] = u64(r)
    elif op.name in ("POPCNT_R64_R64", "POPCNT_R64_M64"):
        result = int(b.bit_count())
        out[dst] = result
        out_flags[0] = 0
        out_flags[1] = int(b == 0)
        out_flags[2] = 0
        out_flags[3] = 0
    elif op.name in ("LZCNT_R64_R64", "LZCNT_R64_M64"):
        result = leading_zero_count(b)
        out[dst] = result
        out_flags[0] = int(b == 0)
        out_flags[1] = int(result == 0)
    elif op.name in ("TZCNT_R64_R64", "TZCNT_R64_M64"):
        result = trailing_zero_count(b)
        out[dst] = result
        out_flags[0] = int(b == 0)
        out_flags[1] = int(result == 0)
    elif op.name in ("ANDN_R64_R64_R64", "ANDN_R64_R64_M64"):
        src1 = out[src]
        src2 = u64(mem_value) if op.name == "ANDN_R64_R64_M64" and mem_value is not None else out[imm & 3]
        result = u64((~src1) & src2)
        out[dst] = result
        out_flags[0] = 0
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        out_flags[3] = 0
    elif op.name in ("BEXTR_R64_R64_R64", "BEXTR_R64_M64_R64"):
        control = out[imm & 3]
        start = control & 0xFF
        length = (control >> 8) & 0xFF
        if start >= 64 or length == 0:
            result = 0
        else:
            mask_len = min(length, 64 - start)
            result = (b >> start) & ((1 << mask_len) - 1)
        out[dst] = result
        out_flags[0] = 0
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        out_flags[3] = 0
    elif op.name in ("BLSI_R64_R64", "BLSI_R64_M64"):
        result = u64(b & -b)
        out[dst] = result
        out_flags[0] = int(b != 0)
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        out_flags[3] = 0
    elif op.name in ("BLSMSK_R64_R64", "BLSMSK_R64_M64"):
        result = u64(b ^ u64(b - 1))
        out[dst] = result
        out_flags[0] = int(b == 0)
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        out_flags[3] = 0
    elif op.name in ("BLSR_R64_R64", "BLSR_R64_M64"):
        result = u64(b & u64(b - 1))
        out[dst] = result
        out_flags[0] = int(b == 0)
        out_flags[1] = int(result == 0)
        out_flags[2] = (result >> 63) & 1
        out_flags[3] = 0
    elif op.name in ("SHLX_R64_R64_R64", "SHLX_R64_M64_R64"):
        out[dst] = u64(b << (out[imm & 3] & 63))
    elif op.name in ("SHRX_R64_R64_R64", "SHRX_R64_M64_R64"):
        out[dst] = b >> (out[imm & 3] & 63)
    elif op.name in ("SARX_R64_R64_R64", "SARX_R64_M64_R64"):
        out[dst] = u64(s64(b) >> (out[imm & 3] & 63))
    elif op.name in ("RORX_R64_R64_IMM8", "RORX_R64_M64_IMM8"):
        out[dst] = rotate_right(b, imm & 63)
    elif op.name in ("BZHI_R64_R64_R64", "BZHI_R64_M64_R64"):
        index = out[imm & 3] & 0xFF
        if index >= 64:
            result = b
            out_flags[0] = 1
        else:
            result = 0 if index == 0 else b & ((1 << index) - 1)
            out_flags[0] = 0
        out[dst] = result
        out_flags[1] = int(result == 0)
    elif op.name in ("MULX_R64_R64_R64", "MULX_R64_R64_M64"):
        low = imm & 3
        product = out[REG_INDEX["RDX"]] * b
        out[dst] = (product >> 64) & MASK64
        out[low] = product & MASK64
    elif op.name in ("PDEP_R64_R64_R64", "PDEP_R64_R64_M64"):
        src_value = out[src]
        mask_value = u64(mem_value) if op.name == "PDEP_R64_R64_M64" and mem_value is not None else out[imm & 3]
        out[dst] = parallel_deposit(src_value, mask_value)
    elif op.name in ("PEXT_R64_R64_R64", "PEXT_R64_R64_M64"):
        src_value = out[src]
        mask_value = u64(mem_value) if op.name == "PEXT_R64_R64_M64" and mem_value is not None else out[imm & 3]
        out[dst] = parallel_extract(src_value, mask_value)
    else:
        raise ValueError(op.name)
    return out, out_flags, flag_mask


def safe_unsigned_div_source(regs: Sequence[int], src: int) -> bool:
    divisor = u64(regs[src])
    if divisor == 0:
        return False
    dividend = (u64(regs[REG_INDEX["RDX"]]) << 64) | u64(regs[REG_INDEX["RAX"]])
    return dividend // divisor <= MASK64


def safe_signed_div_source(regs: Sequence[int], src: int) -> bool:
    divisor = s64(regs[src])
    if divisor == 0:
        return False
    dividend = (u64(regs[REG_INDEX["RDX"]]) << 64) | u64(regs[REG_INDEX["RAX"]])
    if dividend & (1 << 127):
        dividend -= 1 << 128
    q = trunc_div(dividend, divisor)
    return -(1 << 63) <= q < (1 << 63)


def normalize_inputs(regs: List[int], flags: List[int], op_idx: int, dst: int, src: int, imm: int, rng: random.Random) -> Optional[Tuple[List[int], List[int], int, int, int]]:
    op = OPS[op_idx]
    regs = [u64(x) for x in regs]
    flags = [int(x) for x in flags]
    if op.name in ("BSF_R64_R64", "BSR_R64_R64") and regs[src] == 0:
        candidates = [i for i, value in enumerate(regs) if value != 0]
        if not candidates:
            return None
        src = rng.choice(candidates)
    if op.name == "DIV_R64_DE":
        candidates = [i for i in range(len(REGS)) if not safe_unsigned_div_source(regs, i)]
        if not candidates:
            return None
        if src not in candidates:
            src = rng.choice(candidates)
    elif op.name == "IDIV_R64_DE":
        candidates = [i for i in range(len(REGS)) if not safe_signed_div_source(regs, i)]
        if not candidates:
            return None
        if src not in candidates:
            src = rng.choice(candidates)
    elif op.name in ("DIV_R64", "IDIV_R64"):
        pred = safe_unsigned_div_source if op.name == "DIV_R64" else safe_signed_div_source
        candidates = [i for i in range(len(REGS)) if pred(regs, i)]
        if not candidates:
            return None
        if src not in candidates:
            src = rng.choice(candidates)
    if op.name in ("IMUL_R64_R64_IMM8",):
        imm = sign_extend(imm, 8) & 0xFF
    if op.name in (
        "BT_R64_IMM8",
        "BTS_R64_IMM8",
        "BTR_R64_IMM8",
        "BTC_R64_IMM8",
        "BT_M64_IMM8",
        "BTS_M64_IMM8",
        "BTR_M64_IMM8",
        "BTC_M64_IMM8",
    ):
        imm &= 63
    if op.name in ("BT_M64_R64", "BTS_M64_R64", "BTR_M64_R64", "BTC_M64_R64"):
        candidates = [i for i, value in enumerate(regs) if u64(value) < 64]
        if not candidates:
            return None
        if src not in candidates:
            src = rng.choice(candidates)
    if op.name in (
        "ANDN_R64_R64_R64",
        "ANDN_R64_R64_M64",
        "BEXTR_R64_R64_R64",
        "BEXTR_R64_M64_R64",
        "SHLX_R64_R64_R64",
        "SHLX_R64_M64_R64",
        "SHRX_R64_R64_R64",
        "SHRX_R64_M64_R64",
        "SARX_R64_R64_R64",
        "SARX_R64_M64_R64",
        "BZHI_R64_R64_R64",
        "BZHI_R64_M64_R64",
        "PDEP_R64_R64_R64",
        "PDEP_R64_R64_M64",
        "PEXT_R64_R64_R64",
        "PEXT_R64_R64_M64",
    ):
        imm &= 3
    if op.name in ("MULX_R64_R64_R64", "MULX_R64_R64_M64"):
        candidates = [i for i in range(len(REGS)) if i != dst]
        low = imm & 3
        imm = low if low != dst else rng.choice(candidates)
    if op.name in ("RORX_R64_R64_IMM8", "RORX_R64_M64_IMM8"):
        imm &= 63
    return regs, flags, dst, src, imm


def random_regs(rng: random.Random, split: str) -> List[int]:
    boundaries = [0, 1, 2, 0x7F, 0x80, 0xFF, 0xFFFF, 0x7FFFFFFF, 0x80000000, MASK64]
    if split == "heldout_boundary":
        return [rng.choice(boundaries) ^ rng.getrandbits(rng.choice([0, 8, 16])) for _ in REGS]
    regs = [rng.getrandbits(64) for _ in REGS]
    regs[rng.randrange(len(regs))] = rng.randrange(64)
    return regs


def random_flags(rng: random.Random) -> List[int]:
    return [rng.randrange(2) for _ in FLAGS]


def random_memory(rng: random.Random, split: str) -> List[int]:
    memory = [rng.randrange(256) for _ in range(MEM_SIZE)]
    if split == "heldout_boundary":
        boundary_values = [
            0,
            1,
            MASK64,
            0x8000000000000000,
            0x7FFFFFFFFFFFFFFF,
            0x00FF00FF00FF00FF,
            0xFF00FF00FF00FF00,
        ]
        for offset in range(0, MEM_SIZE, 8):
            value = rng.choice(boundary_values) ^ rng.getrandbits(rng.choice([0, 8, 16]))
            memory[offset:offset + 8] = list(u64(value).to_bytes(8, "little"))
    return memory


def read_memory_u64(memory: Sequence[int], offset: int) -> int:
    return int.from_bytes(bytes(int(x) & 0xFF for x in memory[offset:offset + 8]), "little")


def memory_operand_width(op: Stage3OpSpec) -> int:
    if op.operands == "r,m8":
        return 1
    if op.operands == "r,m16":
        return 2
    return 8


def write_memory_u64(memory: Sequence[int], offset: int, value: int) -> List[int]:
    out = [int(x) & 0xFF for x in memory]
    out[offset:offset + 8] = list(u64(value).to_bytes(8, "little"))
    return out


def apply_memory_effect(
    memory: Optional[Sequence[int]],
    op_idx: int,
    mem_offset: Optional[int],
    mem_value: Optional[int],
    imm: int,
    regs: Optional[Sequence[int]] = None,
    src: Optional[int] = None,
    flags: Optional[Sequence[int]] = None,
) -> Optional[List[int]]:
    if memory is None:
        return None
    out = [int(x) & 0xFF for x in memory]
    op = OPS[op_idx]
    if op.operands not in MEMORY_DESTINATION_OPERANDS or mem_offset is None or mem_value is None:
        return out
    value = u64(mem_value)
    if op.name in MEMORY_ROTATE_OPS | MEMORY_SHIFT_OPS:
        count = shift_rotate_count(op.name, regs or [0 for _ in REGS], imm)
        result, _, _ = apply_shift_rotate(op.name, value, flags or [0 for _ in FLAGS], count)
        return write_memory_u64(out, mem_offset, result)
    bit = bit_test_index(op, regs or [0 for _ in REGS], int(src or 0), imm)
    bit_mask = 1 << bit
    if op.name in ("BTS_M64_IMM8", "BTS_M64_R64"):
        value |= bit_mask
    elif op.name in ("BTR_M64_IMM8", "BTR_M64_R64"):
        value &= ~bit_mask
    elif op.name in ("BTC_M64_IMM8", "BTC_M64_R64"):
        value ^= bit_mask
    return write_memory_u64(out, mem_offset, value)


def make_step(
    index: int,
    split: str,
    regs: List[int],
    flags: List[int],
    memory: List[int],
    rip: int,
    rng: random.Random,
    flag_defined: Optional[Sequence[bool]] = None,
) -> Tuple[Dict[str, Any], List[int], List[int], List[int], int, str]:
    flag_defined = [True for _ in FLAGS] if flag_defined is None else [bool(x) for x in flag_defined]
    selected = None
    for _ in range(200):
        op_idx = rng.randrange(len(OPS))
        dst = rng.randrange(len(REGS))
        src = rng.randrange(len(REGS))
        imm = rng.randrange(256)
        selected = normalize_inputs(regs, flags, op_idx, dst, src, imm, rng)
        if selected is not None and flag_reads_are_defined(OPS[op_idx], flag_defined):
            break
        selected = None
    if selected is None:
        op_idx = OP_INDEX["BSWAP_R64"]
        selected = normalize_inputs(regs, flags, op_idx, rng.randrange(len(REGS)), 0, 0, rng)
    assert selected is not None
    op = OPS[op_idx]
    regs, flags, dst, src, imm = selected
    before_regs = list(regs)
    before_flags = list(flags)
    before_memory = list(memory)
    mem_offset = -1
    mem_addr = 0
    mem_value: Optional[int] = None
    if op.operands in MEMORY_OPERANDS:
        mem_offset = rng.randrange(0, MEM_SIZE - 7, 8)
        mem_addr = MEM_BASE + mem_offset
        mem_value = read_memory_u64(before_memory, mem_offset)
    bytes_hex = encode_x86_bytes(op, dst, src, imm, mem_addr)
    after_regs, after_flags, flag_mask = apply_op(regs, flags, op_idx, dst, src, imm, mem_value)
    after_memory = apply_memory_effect(
        before_memory,
        op_idx,
        mem_offset if mem_offset >= 0 else None,
        mem_value,
        imm,
        before_regs,
        src,
        before_flags,
    )
    assert after_memory is not None
    fallthrough_rip = rip + len(bytes.fromhex(bytes_hex))
    exception = op.expected_exception
    after_rip = rip if exception else fallthrough_rip
    exception_vector = 0 if exception == "#DE" else -1
    exception_error_code = 0 if exception == "#DE" else -1
    reg_writes = [
        {"name": reg, "before": hex_u(before_regs[i], 64), "after": hex_u(after_regs[i], 64), "mask": hex_u(MASK64, 64)}
        for i, reg in enumerate(REGS)
        if before_regs[i] != after_regs[i]
    ]
    flags_written = [FLAGS[i] for i, bit in enumerate(flag_mask) if bit]
    undefined_flags = undefined_flag_names(op, flag_mask)
    after_flag_defined = flags_defined_after(flag_defined, flag_mask, undefined_flags)
    if mem_value is not None:
        read_width = memory_operand_width(op)
        read_mask = (1 << (read_width * 8)) - 1
        memory_reads = [
            {
                "address": hex_u(mem_addr, 64),
                "width_bytes": read_width,
                "value_hex": hex_u((mem_value or 0) & read_mask, read_width * 8),
                "segment": "flat",
                "fault": None,
            }
        ]
    else:
        memory_reads = []
    memory_writes = []
    if op.name in MEMORY_WRITE_OPS and mem_value is not None:
        memory_writes.append(
            {
                "address": hex_u(mem_addr, 64),
                "width_bytes": 8,
                "value_hex": hex_u(read_memory_u64(after_memory, mem_offset), 64),
                "segment": "flat",
                "fault": None,
            }
        )
    step = {
        "index": index,
        "decode": {
            "bytes_hex": bytes_hex,
            "iclass": op.name.split("_")[0],
            "iform": op.iform,
            "category": op.family,
            "extension": extension_for_op(op),
            "operands": decode_operands(op, dst, src, imm, mem_addr),
        },
        "before": state_record(before_regs, before_flags, rip, before_memory),
        "after": state_record(after_regs, after_flags, after_rip, after_memory, exception=exception),
        "effects": {
            "register_writes": reg_writes,
            "memory_reads": memory_reads,
            "memory_writes": memory_writes,
            "flags_written": flags_written,
            "undefined_flags": undefined_flags,
            "exception": exception,
        },
        "oracle_disagreement": None,
        "pan_stage3": {
            "op_index": op_idx,
            "dst": dst,
            "src": src,
            "imm": imm,
            "selector": op_idx,
            "before_regs": before_regs,
            "before_flags": before_flags,
            "before_flag_defined": list(flag_defined),
            "after_regs": after_regs,
            "after_flags": after_flags,
            "after_flag_defined": after_flag_defined,
            "flag_mask": flag_mask,
            "flag_reads": flag_reads(op),
            "before_rip": rip,
            "after_rip": after_rip,
            "fallthrough_rip": fallthrough_rip,
            "exception": exception,
            "exception_vector": exception_vector,
            "exception_error_code": exception_error_code,
        },
    }
    if mem_value is not None:
        step["pan_stage3"].update(
            {
                "mem_offset": mem_offset,
                "mem_addr": mem_addr,
                "mem_value": mem_value,
                "before_memory": before_memory,
                "after_memory": after_memory,
            }
        )
    return step, after_regs, after_flags, after_memory, fallthrough_rip + rng.randrange(1, 8), asm_string(op, dst, src, imm, mem_addr)


def make_trace(trace_id: str, split: str, length: int, rng: random.Random) -> Dict[str, Any]:
    regs = random_regs(rng, split)
    flags = random_flags(rng)
    flag_defined = [True for _ in FLAGS]
    memory = random_memory(rng, split)
    rip = CODE_BASE + rng.randrange(0, 0x8000)
    steps = []
    asm = []
    bytes_parts = []
    for idx in range(length):
        step, regs, flags, memory, rip, asm_text = make_step(idx, split, regs, flags, memory, rip, rng, flag_defined)
        flag_defined = [bool(x) for x in step["pan_stage3"]["after_flag_defined"]]
        steps.append(step)
        asm.append(asm_text)
        bytes_parts.append(step["decode"]["bytes_hex"])
    return {
        "schema_version": "pan-x86-trace-v1",
        "trace_id": trace_id,
        "stage": STAGE,
        "source": {
            "generator": "stage3_complex_scalar_pan.py",
            "seed": trace_id,
            "xed_version": "intel-xed-idata-af3c1a1",
            "oracle": ["internal_semantics"],
        },
        "mode": {
            "long_mode": True,
            "operand_bits": 64,
            "address_bits": 64,
            "ring": 0,
            "cpuid_features": list(CPUID_FEATURES),
        },
        "program": {
            "bytes_hex": "".join(bytes_parts),
            "asm": asm,
            "length": length,
            "template_id": split,
        },
        "steps": steps,
        "final": steps[-1]["after"],
    }


def split_diagnostics(traces: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    op_counts: Dict[str, int] = {}
    family_counts: Dict[str, int] = {}
    steps = 0
    for trace in traces:
        for step in trace["steps"]:
            op = OPS[step["pan_stage3"]["op_index"]]
            op_counts[op.name] = op_counts.get(op.name, 0) + 1
            family_counts[op.family] = family_counts.get(op.family, 0) + 1
            steps += 1
    return {
        "programs": len(traces),
        "steps": steps,
        "op_counts": dict(sorted(op_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
    }


def make_manifest(out_dir: Path) -> Dict[str, Any]:
    data = {
        "stage": STAGE,
        "substage": SUBSTAGE,
        "code_base": hex_u(CODE_BASE, 64),
        "registers": list(REGS),
        "flags": list(FLAGS),
        "cpuid_features": list(CPUID_FEATURES),
        "memory_base": hex_u(MEM_BASE, 64),
        "memory_size": MEM_SIZE,
        "forms": [dataclasses.asdict(op) | {"op_index": i, "implemented": True} for i, op in enumerate(OPS)],
        "limitations": [
            "divide-error coverage is limited to DIV/IDIV divisor-zero #DE cases",
            "register operands over RAX/RBX/RCX/RDX only",
            "memory operands are limited to read-only MOVSX/MOVZX m8/m16, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR m64 source forms, TEST m64 r64, BSF/BSR/POPCNT/LZCNT/TZCNT, selected BMI1/BMI2 m64 source forms, BT/BTS/BTR/BTC m64 imm8 plus m64 r64 bit-test/modify forms, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms over a flat 256-byte memory image",
            "undefined flags are masked instead of claimed exact",
            "CPUID-gated coverage is limited to POPCNT/LZCNT/TZCNT plus selected BMI1/BMI2 register and m64 source forms",
            "BMI1 coverage is limited to TZCNT plus selected ANDN/BEXTR/BLSI/BLSMSK/BLSR register and m64 source forms",
            "BMI2 coverage is limited to selected BZHI/SHLX/SHRX/SARX/RORX/MULX/PDEP/PEXT register and m64 source forms",
        ],
    }
    write_json(out_dir / "coverage_manifest.json", data)
    return data


def generate_dataset(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("generate")
    rng = random.Random(args.seed)
    make_manifest(out_dir)

    def rows(count: int, split: str, min_len: int, max_len: int, used: set[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        attempts = 0
        while len(out) < count and attempts < max(10000, count * 200):
            attempts += 1
            trace = make_trace(f"{split}-{args.seed}-{len(out)}", split, rng.randint(min_len, max_len), rng)
            key = trace["program"]["bytes_hex"]
            if key in used:
                continue
            used.add(key)
            out.append(trace)
        if len(out) != count:
            raise RuntimeError(f"could only generate {len(out)} unique {split} traces")
        return out

    used: set[str] = set()
    train = rows(args.train_programs, "train", 2, args.max_train_len, used)
    val = rows(args.val_programs, "heldout_boundary", 2, args.max_train_len, used)
    long = rows(args.long_programs, "long", args.long_len, args.long_len, used)
    n_train = write_jsonl(out_dir / "train_traces.jsonl", train)
    n_val = write_jsonl(out_dir / "val_traces.jsonl", val)
    n_long = write_jsonl(out_dir / "long_traces.jsonl", long)
    schema_result = validate_trace_schema(out_dir, out_dir / "train_traces.jsonl")
    summary = {
        "train_programs": n_train,
        "val_programs": n_val,
        "long_programs": n_long,
        "train_path": str(out_dir / "train_traces.jsonl"),
        "val_path": str(out_dir / "val_traces.jsonl"),
        "long_path": str(out_dir / "long_traces.jsonl"),
        "schema_validation": schema_result,
        "split_diagnostics": {
            "train": split_diagnostics(train),
            "val": split_diagnostics(val),
            "long": split_diagnostics(long),
        },
    }
    write_json(out_dir / "generation_summary.json", summary)
    return summary


def load_step_rows(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for trace in load_traces(path):
        for step in trace["steps"]:
            p = step["pan_stage3"]
            rows.append(
                {
                    "op": int(p["op_index"]),
                    "selector": int(p["selector"]),
                    "flags": [int(x) for x in p["before_flags"]],
                }
            )
    return rows


class Predictor(Protocol):
    def predict_next(
        self,
        regs: List[int],
        flags: List[int],
        memory: Optional[List[int]],
        step: Dict[str, Any],
    ) -> Tuple[List[int], List[int], Optional[List[int]], Dict[str, Any]]:
        ...


def transition_with_selector(
    regs: List[int],
    flags: List[int],
    memory: Optional[List[int]],
    step: Dict[str, Any],
    selector: int,
    flip_cf: bool = False,
) -> Tuple[List[int], List[int], Optional[List[int]], Dict[str, Any]]:
    p = step["pan_stage3"]
    use_flags = list(flags)
    if flip_cf:
        use_flags[0] = 1 - use_flags[0]
    selected_op = OPS[selector]
    use_memory = [int(x) & 0xFF for x in memory] if memory is not None else None
    if use_memory is None and "before_memory" in p:
        use_memory = [int(x) & 0xFF for x in p["before_memory"]]
    mem_offset = int(p["mem_offset"]) if "mem_offset" in p else None
    mem_value = None
    if selected_op.operands in MEMORY_OPERANDS and use_memory is not None and mem_offset is not None:
        mem_value = read_memory_u64(use_memory, mem_offset)
    out_regs, out_flags, flag_mask = apply_op(
        regs,
        use_flags,
        selector,
        int(p["dst"]),
        int(p["src"]),
        int(p["imm"]),
        mem_value,
    )
    out_memory = apply_memory_effect(use_memory, selector, mem_offset, mem_value, int(p["imm"]), regs, int(p["src"]), use_flags)
    return out_regs, out_flags, out_memory, {
        "selector": selector,
        "flag_mask": flag_mask,
        "exception": selected_op.expected_exception,
    }


class StructuredStage3Predictor:
    def __init__(self, op_offset: int = 0, flip_cf: bool = False):
        self.op_offset = op_offset
        self.flip_cf = flip_cf

    def predict_next(
        self,
        regs: List[int],
        flags: List[int],
        memory: Optional[List[int]],
        step: Dict[str, Any],
    ) -> Tuple[List[int], List[int], Optional[List[int]], Dict[str, Any]]:
        p = step["pan_stage3"]
        selector = (int(p["op_index"]) + self.op_offset) % len(OPS)
        try:
            return transition_with_selector(regs, flags, memory, step, selector, flip_cf=self.flip_cf)
        except Exception as exc:
            return list(regs), list(flags), memory, {"selector": selector, "flag_mask": [0, 0, 0, 0], "error": f"{type(exc).__name__}: {exc}"}


class FactorizedStage3PAN(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.op_emb = nn.Embedding(len(OPS), 32)
        self.net = nn.Sequential(
            nn.Linear(32 + len(FLAGS), hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(OPS)),
        )

    def forward(self, op: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.op_emb(op.long()), flags.float()], dim=-1)
        return self.net(x)


class EffectStage3PAN(FactorizedStage3PAN):
    """Learned Stage 3 effect-family selector head.

    Stage 3 covers wide 64-bit scalar and exception-producing operations. This
    first learned-effect mode emits the operation/effect family used for commit
    and validation gates it separately from the legacy factorized selector path.
    """


def make_model(arch: str, hidden: int) -> FactorizedStage3PAN:
    if arch == "factorized":
        return FactorizedStage3PAN(hidden=hidden)
    if arch == "effect":
        return EffectStage3PAN(hidden=hidden)
    raise ValueError(f"unknown Stage 3 arch {arch!r}")


class NeuralStage3Predictor:
    def __init__(self, model: FactorizedStage3PAN):
        self.model = model.eval()
        self.learned_effect_mode = isinstance(model, EffectStage3PAN)

    def predict_next(
        self,
        regs: List[int],
        flags: List[int],
        memory: Optional[List[int]],
        step: Dict[str, Any],
    ) -> Tuple[List[int], List[int], Optional[List[int]], Dict[str, Any]]:
        p = step["pan_stage3"]
        with torch.no_grad():
            op = torch.tensor([int(p["op_index"])], dtype=torch.long)
            fl = torch.tensor([flags], dtype=torch.float32)
            selector = int(self.model(op, fl).argmax(-1)[0])
        try:
            return transition_with_selector(regs, flags, memory, step, selector)
        except Exception as exc:
            return list(regs), list(flags), memory, {"selector": selector, "flag_mask": [0, 0, 0, 0], "error": f"{type(exc).__name__}: {exc}"}


def selector_exact(model: FactorizedStage3PAN, rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    with torch.no_grad():
        op = torch.tensor([r["op"] for r in rows], dtype=torch.long)
        flags = torch.tensor([r["flags"] for r in rows], dtype=torch.float32)
        target = torch.tensor([r["selector"] for r in rows], dtype=torch.long)
        pred = model(op, flags).argmax(-1)
    return float((pred == target).float().mean().item())


def train_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("train")
    train_rows = load_step_rows(out_dir / "train_traces.jsonl")
    val_rows = load_step_rows(out_dir / "val_traces.jsonl")
    model = make_model(args.arch, args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    rng = random.Random(args.seed + 17)
    for step in range(args.train_steps):
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        op = torch.tensor([r["op"] for r in batch], dtype=torch.long)
        flags = torch.tensor([r["flags"] for r in batch], dtype=torch.float32)
        target = torch.tensor([r["selector"] for r in batch], dtype=torch.long)
        loss = F.cross_entropy(model(op, flags), target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % args.eval_every == 0 or step == args.train_steps - 1:
            rec = {
                "step": step,
                "loss": float(loss.item()),
                "train_selector_exact": selector_exact(model, train_rows[: min(5000, len(train_rows))]),
                "val_selector_exact": selector_exact(model, val_rows),
            }
            history.append(rec)
            if rec["val_selector_exact"] >= 1.0 and step >= args.min_train_steps:
                break
    torch.save({
        "model_state": model.state_dict(),
        "arch": args.arch,
        "hidden": args.hidden,
        "ops": [dataclasses.asdict(op) for op in OPS],
    }, out_dir / "stage3_pan.pt")
    write_json(out_dir / "train_history.json", history)
    summary = {
        "model_kind": "neural_effect_selector" if args.arch == "effect" else "neural_factorized_selector",
        "arch": args.arch,
        "train_steps_requested": args.train_steps,
        "train_steps_completed": history[-1]["step"] if history else 0,
        "final": history[-1] if history else {},
        "checkpoint": str(out_dir / "stage3_pan.pt"),
    }
    write_json(out_dir / "train_summary.json", summary)
    return summary


def compare_flags(actual: Sequence[int], expected: Sequence[int], mask: Sequence[int]) -> bool:
    return all(int(m) == 0 or int(actual[i]) == int(expected[i]) for i, m in enumerate(mask))


def validate_split(path: Path, predictor: Predictor, split: str) -> Dict[str, Any]:
    programs = 0
    steps = 0
    exact_steps = 0
    final_exact = 0
    reg_exact = 0
    flag_exact = 0
    memory_exact = 0
    memory_checked = 0
    selector_exact_count = 0
    exception_exact_count = 0
    exception_checked = 0
    flag_compared = 0
    undefined_flag_read_count = 0
    undefined_flag_read_examples: List[Dict[str, Any]] = []
    by_family: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []
    for trace in load_traces(path):
        programs += 1
        regs = [int(x) for x in trace["steps"][0]["pan_stage3"]["before_regs"]]
        flags = [int(x) for x in trace["steps"][0]["pan_stage3"]["before_flags"]]
        flag_defined = [True for _ in FLAGS]
        memory: Optional[List[int]] = None
        trace_ok = True
        for step in trace["steps"]:
            p = step["pan_stage3"]
            op = OPS[int(p["op_index"])]
            reads = flag_reads(op)
            missing_reads = [name for name in reads if not flag_defined[FLAG_INDEX[name]]]
            if missing_reads:
                undefined_flag_read_count += 1
                if len(undefined_flag_read_examples) < 10:
                    undefined_flag_read_examples.append(
                        {
                            "trace_id": trace["trace_id"],
                            "step": int(step["index"]),
                            "op": op.name,
                            "reads": reads,
                            "undefined_reads": missing_reads,
                        }
                    )
            if memory is None and "before_memory" in p:
                memory = [int(x) & 0xFF for x in p["before_memory"]]
            pred_regs, pred_flags, pred_memory, aux = predictor.predict_next(regs, flags, memory, step)
            exp_regs = [int(x) for x in p["after_regs"]]
            exp_flags = [int(x) for x in p["after_flags"]]
            exp_memory = [int(x) & 0xFF for x in p["after_memory"]] if "after_memory" in p else memory
            before_memory_ok = True
            if memory is not None and "before_memory" in p:
                before_memory_ok = memory == [int(x) & 0xFF for x in p["before_memory"]]
            mask = [int(x) for x in p["flag_mask"]]
            regs_ok = pred_regs == exp_regs
            flags_ok = compare_flags(pred_flags, exp_flags, mask)
            memory_ok = before_memory_ok and (pred_memory == exp_memory)
            selector_ok = int(aux["selector"]) == int(p["selector"])
            expected_exception = p.get("exception")
            exception_ok = aux.get("exception") == expected_exception
            if expected_exception is not None or aux.get("exception") is not None:
                exception_checked += 1
                exception_exact_count += int(exception_ok)
            step_ok = regs_ok and flags_ok and memory_ok and selector_ok and exception_ok
            steps += 1
            exact_steps += int(step_ok)
            reg_exact += int(regs_ok)
            if "after_memory" in p:
                memory_checked += 1
                memory_exact += int(memory_ok)
            selector_exact_count += int(selector_ok)
            if any(mask):
                flag_compared += 1
                flag_exact += int(flags_ok)
            fam = by_family.setdefault(op.family, {"steps": 0, "exact": 0})
            fam["steps"] += 1
            fam["exact"] += int(step_ok)
            if not step_ok and len(failures) < 10:
                failures.append({
                    "trace_id": trace["trace_id"],
                    "step": int(step["index"]),
                    "op": op.name,
                    "selector": int(aux["selector"]),
                    "expected_selector": int(p["selector"]),
                    "regs_ok": regs_ok,
                    "flags_ok": flags_ok,
                    "memory_ok": memory_ok,
                    "exception_ok": exception_ok,
                    "expected_exception": expected_exception,
                    "actual_exception": aux.get("exception"),
                })
            trace_ok = trace_ok and step_ok
            regs = pred_regs
            flags = pred_flags
            memory = pred_memory
            flag_defined = flags_defined_after(
                flag_defined,
                [int(x) for x in p["flag_mask"]],
                step["effects"].get("undefined_flags", []),
            )
        final_p = trace["steps"][-1]["pan_stage3"]
        final_memory_ok = True
        if memory is not None:
            final_memory_ok = memory_digest(memory) == trace["final"]["memory_digest"]
        final_exact += int(trace_ok and regs == [int(x) for x in final_p["after_regs"]] and final_memory_ok)
    worst_family_exact = min((v["exact"] / max(1, v["steps"]) for v in by_family.values()), default=0.0)
    return {
        "split": split,
        "programs": programs,
        "steps": steps,
        "step_exact": round(exact_steps / max(1, steps), 6),
        "final_exact": round(final_exact / max(1, programs), 6),
        "reg_exact": round(reg_exact / max(1, steps), 6),
        "flag_exact": round(flag_exact / max(1, flag_compared), 6),
        "memory_exact": round(memory_exact / max(1, memory_checked), 6) if memory_checked else 1.0,
        "memory_checked_steps": memory_checked,
        "selector_exact": round(selector_exact_count / max(1, steps), 6),
        "exception_exact": round(exception_exact_count / max(1, exception_checked), 6),
        "exception_checked_steps": exception_checked,
        "continuous_undefined_flag_reads": undefined_flag_read_count,
        "continuous_flag_safety_exact": round(1.0 - (undefined_flag_read_count / max(1, steps)), 6),
        "worst_family_exact": round(worst_family_exact, 6),
        "by_family": {k: {"steps": v["steps"], "exact": round(v["exact"] / max(1, v["steps"]), 6)} for k, v in sorted(by_family.items())},
        "failure_examples": failures,
        "undefined_flag_read_examples": undefined_flag_read_examples,
    }


def byte_overlap(out_dir: Path) -> Dict[str, Any]:
    sets = {}
    for split in ("train", "val", "long"):
        sets[split] = {trace["program"]["bytes_hex"] for trace in load_traces(out_dir / f"{split}_traces.jsonl")}
    return {
        "train_programs": len(sets["train"]),
        "val_programs": len(sets["val"]),
        "long_programs": len(sets["long"]),
        "train_val_overlap": len(sets["train"] & sets["val"]),
        "train_long_overlap": len(sets["train"] & sets["long"]),
    }


def validate_common(args: argparse.Namespace, out_dir: Path, predictor: Predictor, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("validate")
    schema_ok = bool(validate_trace_schema(out_dir, out_dir / "train_traces.jsonl").get("ok"))
    val = validate_split(out_dir / "val_traces.jsonl", predictor, "val")
    long = validate_split(out_dir / "long_traces.jsonl", predictor, "long")
    scrambled = validate_split(out_dir / "val_traces.jsonl", StructuredStage3Predictor(op_offset=1), "scrambled_opcode_val")
    flipped = validate_split(out_dir / "val_traces.jsonl", StructuredStage3Predictor(flip_cf=True), "flipped_cf_val")
    overlap = byte_overlap(out_dir)
    learned_effect_mode = bool(getattr(predictor, "learned_effect_mode", False))
    learned_effect_gate = True
    if learned_effect_mode:
        learned_effect_gate = (
            val["selector_exact"] >= args.gate_selector_exact
            and long["selector_exact"] >= args.gate_selector_exact
            and val["reg_exact"] >= args.gate_step_exact
            and long["reg_exact"] >= args.gate_step_exact
            and val["memory_exact"] >= args.gate_memory_exact
            and long["memory_exact"] >= args.gate_memory_exact
            and val["exception_exact"] >= args.gate_exception_exact
            and long["exception_exact"] >= args.gate_exception_exact
        )
    gate = (
        schema_ok
        and learned_effect_gate
        and val["step_exact"] >= args.gate_step_exact
        and long["step_exact"] >= args.gate_step_exact
        and val["selector_exact"] >= args.gate_selector_exact
        and long["selector_exact"] >= args.gate_selector_exact
        and val["exception_exact"] >= args.gate_exception_exact
        and long["exception_exact"] >= args.gate_exception_exact
        and val["memory_exact"] >= args.gate_memory_exact
        and long["memory_exact"] >= args.gate_memory_exact
        and val["worst_family_exact"] >= args.gate_worst_family_exact
        and overlap["train_val_overlap"] == 0
        and overlap["train_long_overlap"] == 0
        and val["continuous_undefined_flag_reads"] == 0
        and long["continuous_undefined_flag_reads"] == 0
    )
    report = {
        "schema_valid": schema_ok,
        "transition_contract": {
            "schema": "pan-x86-stage3-effect-selector-v1" if learned_effect_mode else "legacy-stage3-factorized-selector",
            "learned_effect_selectors": learned_effect_mode,
            "learned_effect_axes": ["scalar_family_effect", "flag_mask", "memory_effect", "exception_class"] if learned_effect_mode else [],
            "commit_path": "model_emitted_effect_selector" if learned_effect_mode else "legacy_model_selector",
            "direct_value_heads": False,
            "note": "Stage 3 effect mode learns the scalar effect family used for typed commit; direct 64-bit value heads are still future work.",
        },
        "val": val,
        "long": long,
        "negative_controls": {
            "scrambled_opcode_val": scrambled,
            "flipped_cf_val": flipped,
        },
        "byte_overlap": overlap,
        "gate_status": {
            "schema_valid": schema_ok,
            "stage3_gate_passed": bool(gate),
            "learned_effect_selector_gate_passed": learned_effect_gate,
            "full_x86_goal_complete": False,
        },
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def validate_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    ckpt = torch.load(out_dir / "stage3_pan.pt", map_location="cpu")
    model = make_model(str(ckpt.get("arch", getattr(args, "arch", "factorized"))), int(ckpt.get("hidden", args.hidden)))
    model.load_state_dict(ckpt["model_state"])
    return validate_common(args, out_dir, NeuralStage3Predictor(model), monitor)


def run_smoke(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir or (DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / STAGE / args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(out_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        generation = generate_dataset(args, out_dir, monitor)
        train = train_model(args, out_dir, monitor)
        validation = validate_model(args, out_dir, monitor)
        monitor.set_phase("finalize")
        summary = {
            "stage": STAGE,
            "substage": SUBSTAGE,
            "run_name": args.run_name,
            "out_dir": str(out_dir),
            "generation": generation,
            "train_summary": train,
            "gate_status": validation["gate_status"],
            "validation": {
                "val_step_exact": validation["val"]["step_exact"],
                "long_step_exact": validation["long"]["step_exact"],
                "val_exception_exact": validation["val"]["exception_exact"],
                "long_exception_exact": validation["long"]["exception_exact"],
                "val_memory_exact": validation["val"]["memory_exact"],
                "long_memory_exact": validation["long"]["memory_exact"],
                "val_exception_checked_steps": validation["val"]["exception_checked_steps"],
                "long_exception_checked_steps": validation["long"]["exception_checked_steps"],
                "val_memory_checked_steps": validation["val"]["memory_checked_steps"],
                "long_memory_checked_steps": validation["long"]["memory_checked_steps"],
                "val_continuous_undefined_flag_reads": validation["val"]["continuous_undefined_flag_reads"],
                "long_continuous_undefined_flag_reads": validation["long"]["continuous_undefined_flag_reads"],
                "val_worst_family_exact": validation["val"]["worst_family_exact"],
                "long_worst_family_exact": validation["long"]["worst_family_exact"],
                "scrambled_opcode_val_step_exact": validation["negative_controls"]["scrambled_opcode_val"]["step_exact"],
                "flipped_cf_val_step_exact": validation["negative_controls"]["flipped_cf_val"]["step_exact"],
            },
            "full_x86_goal_complete": False,
        }
        write_json(out_dir / "run_summary.json", summary)
        return summary
    finally:
        monitor.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run-smoke")
    run.add_argument("--run-name", default="stage3-complex-scalar-base-alu-v21-neural")
    run.add_argument("--out-dir", type=Path, default=None)
    run.add_argument("--seed", type=int, default=20260649)
    run.add_argument("--train-programs", type=int, default=1200)
    run.add_argument("--val-programs", type=int, default=360)
    run.add_argument("--long-programs", type=int, default=120)
    run.add_argument("--max-train-len", type=int, default=6)
    run.add_argument("--long-len", type=int, default=24)
    run.add_argument("--train-steps", type=int, default=300)
    run.add_argument("--min-train-steps", type=int, default=60)
    run.add_argument("--batch-size", type=int, default=256)
    run.add_argument("--hidden", type=int, default=96)
    run.add_argument("--arch", choices=["factorized", "effect"], default="factorized")
    run.add_argument("--lr", type=float, default=0.003)
    run.add_argument("--eval-every", type=int, default=30)
    run.add_argument("--monitor-interval", type=float, default=0.5)
    run.add_argument("--gate-step-exact", type=float, default=0.995)
    run.add_argument("--gate-selector-exact", type=float, default=0.999)
    run.add_argument("--gate-exception-exact", type=float, default=0.995)
    run.add_argument("--gate-memory-exact", type=float, default=0.995)
    run.add_argument("--gate-worst-family-exact", type=float, default=0.970)

    val = sub.add_parser("validate")
    val.add_argument("--out-dir", type=Path, required=True)
    val.add_argument("--hidden", type=int, default=96)
    val.add_argument("--arch", choices=["factorized", "effect"], default="factorized")
    val.add_argument("--monitor-interval", type=float, default=0.5)
    val.add_argument("--gate-step-exact", type=float, default=0.995)
    val.add_argument("--gate-selector-exact", type=float, default=0.999)
    val.add_argument("--gate-exception-exact", type=float, default=0.995)
    val.add_argument("--gate-memory-exact", type=float, default=0.995)
    val.add_argument("--gate-worst-family-exact", type=float, default=0.970)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run-smoke":
        result = run_smoke(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_status"].get("stage3_gate_passed") else 1
    if args.cmd == "validate":
        monitor = ResourceMonitor(args.out_dir, interval_s=args.monitor_interval)
        monitor.start()
        try:
            report = validate_model(args, args.out_dir, monitor)
            print(json.dumps(report["gate_status"], indent=2, sort_keys=True))
            return 0 if report["gate_status"].get("stage3_gate_passed") else 1
        finally:
            monitor.stop()
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
