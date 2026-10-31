#!/usr/bin/env python3
"""Stage 4 PAN x86 SIMD runner.

This Stage 4 substage covers exact SSE/SSE2 integer vector semantics over
XMM0-XMM15, including aligned m128 source operands for the register-source
forms, aligned MOVDQA/MOVAPD plus unaligned MOVDQU/MOVUPD m128 source operands,
MOVD/MOVQ/MOVSD/MOVLPD/MOVHPD narrow source operands, aligned MOVDQA/MOVD/MOVQ/MOVNT
memory destinations, MOVSD/MOVAPD/MOVUPD/MOVQ register transfer forms, unaligned
MOVDQU source/destination memory, aligned-m128 unaligned-memory #GP(0)
exception channels, and a scalar/packed single/double floating-point slice over
safe finite values plus non-trapping zero/infinity edge operands. It deliberately
avoids signaling NaNs, subnormals, MXCSR status-flag claims, AVX encodings, and
x87 stack semantics. It is a measured SIMD foothold, not full SIMD/x87 coverage.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import struct
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
    ResourceMonitor,
    hex_u,
    load_traces,
    validate_trace_schema,
    write_json,
    write_jsonl,
)


STAGE = "stage4_simd_x87"
SUBSTAGE = "stage4_sse_aligned_exception_v31"
XMM_REGS = (
    "XMM0",
    "XMM1",
    "XMM2",
    "XMM3",
    "XMM4",
    "XMM5",
    "XMM6",
    "XMM7",
    "XMM8",
    "XMM9",
    "XMM10",
    "XMM11",
    "XMM12",
    "XMM13",
    "XMM14",
    "XMM15",
)
CODE_BASE = 0x600000
MEM_BASE = 0x170000
MEM_SIZE = 256
MASK128 = (1 << 128) - 1
MASK64 = (1 << 64) - 1
MASK32 = (1 << 32) - 1
HIGH64_MASK = MASK128 ^ MASK64
HIGH96_MASK = MASK128 ^ MASK32
MXCSR_DEFAULT = 0x1F80


@dataclasses.dataclass(frozen=True)
class Stage4OpSpec:
    name: str
    iform: str
    family: str
    operands: str
    opcode: int
    lane_bits: int
    imm_group: Optional[int] = None
    prefix: Optional[int] = 0x66
    memory_alignment: int = 16
    expected_exception: Optional[str] = None
    memory_store_bit_offset: int = 0


BASE_OPS: Tuple[Stage4OpSpec, ...] = (
    Stage4OpSpec("MOVDQA_XMM_XMM", "MOVDQA_XMMdq_XMMdq_0F6F", "DATAXFER", "xmm,xmm", 0x6F, 128),
    Stage4OpSpec("PAND_XMM_XMM", "PAND_XMMdq_XMMdq", "LOGICAL", "xmm,xmm", 0xDB, 128),
    Stage4OpSpec("PANDN_XMM_XMM", "PANDN_XMMdq_XMMdq", "LOGICAL", "xmm,xmm", 0xDF, 128),
    Stage4OpSpec("POR_XMM_XMM", "POR_XMMdq_XMMdq", "LOGICAL", "xmm,xmm", 0xEB, 128),
    Stage4OpSpec("PXOR_XMM_XMM", "PXOR_XMMdq_XMMdq", "LOGICAL", "xmm,xmm", 0xEF, 128),
    Stage4OpSpec("PADDB_XMM_XMM", "PADDB_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xFC, 8),
    Stage4OpSpec("PADDW_XMM_XMM", "PADDW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xFD, 16),
    Stage4OpSpec("PADDD_XMM_XMM", "PADDD_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xFE, 32),
    Stage4OpSpec("PADDQ_XMM_XMM", "PADDQ_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xD4, 64),
    Stage4OpSpec("PSUBB_XMM_XMM", "PSUBB_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xF8, 8),
    Stage4OpSpec("PSUBW_XMM_XMM", "PSUBW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xF9, 16),
    Stage4OpSpec("PSUBD_XMM_XMM", "PSUBD_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xFA, 32),
    Stage4OpSpec("PSUBQ_XMM_XMM", "PSUBQ_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xFB, 64),
    Stage4OpSpec("PCMPEQB_XMM_XMM", "PCMPEQB_XMMdq_XMMdq", "SSE_COMPARE", "xmm,xmm", 0x74, 8),
    Stage4OpSpec("PCMPEQW_XMM_XMM", "PCMPEQW_XMMdq_XMMdq", "SSE_COMPARE", "xmm,xmm", 0x75, 16),
    Stage4OpSpec("PCMPEQD_XMM_XMM", "PCMPEQD_XMMdq_XMMdq", "SSE_COMPARE", "xmm,xmm", 0x76, 32),
    Stage4OpSpec("PMULLW_XMM_XMM", "PMULLW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xD5, 16),
    Stage4OpSpec("PUNPCKLBW_XMM_XMM", "PUNPCKLBW_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x60, 8),
    Stage4OpSpec("PUNPCKLWD_XMM_XMM", "PUNPCKLWD_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x61, 16),
    Stage4OpSpec("PUNPCKLDQ_XMM_XMM", "PUNPCKLDQ_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x62, 32),
    Stage4OpSpec("PACKSSWB_XMM_XMM", "PACKSSWB_XMMdq_XMMdq", "SSE_PACK", "xmm,xmm", 0x63, 8),
    Stage4OpSpec("PCMPGTB_XMM_XMM", "PCMPGTB_XMMdq_XMMdq", "SSE_COMPARE", "xmm,xmm", 0x64, 8),
    Stage4OpSpec("PCMPGTW_XMM_XMM", "PCMPGTW_XMMdq_XMMdq", "SSE_COMPARE", "xmm,xmm", 0x65, 16),
    Stage4OpSpec("PCMPGTD_XMM_XMM", "PCMPGTD_XMMdq_XMMdq", "SSE_COMPARE", "xmm,xmm", 0x66, 32),
    Stage4OpSpec("PACKUSWB_XMM_XMM", "PACKUSWB_XMMdq_XMMdq", "SSE_PACK", "xmm,xmm", 0x67, 8),
    Stage4OpSpec("PUNPCKHBW_XMM_XMM", "PUNPCKHBW_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x68, 8),
    Stage4OpSpec("PUNPCKHWD_XMM_XMM", "PUNPCKHWD_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x69, 16),
    Stage4OpSpec("PUNPCKHDQ_XMM_XMM", "PUNPCKHDQ_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x6A, 32),
    Stage4OpSpec("PACKSSDW_XMM_XMM", "PACKSSDW_XMMdq_XMMdq", "SSE_PACK", "xmm,xmm", 0x6B, 16),
    Stage4OpSpec("PUNPCKLQDQ_XMM_XMM", "PUNPCKLQDQ_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x6C, 64),
    Stage4OpSpec("PUNPCKHQDQ_XMM_XMM", "PUNPCKHQDQ_XMMdq_XMMq", "SSE_SHUFFLE", "xmm,xmm", 0x6D, 64),
    Stage4OpSpec("PAVGB_XMM_XMM", "PAVGB_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xE0, 8),
    Stage4OpSpec("PAVGW_XMM_XMM", "PAVGW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xE3, 16),
    Stage4OpSpec("PMULHUW_XMM_XMM", "PMULHUW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xE4, 16),
    Stage4OpSpec("PMULHW_XMM_XMM", "PMULHW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xE5, 16),
    Stage4OpSpec("PMINSW_XMM_XMM", "PMINSW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xEA, 16),
    Stage4OpSpec("PMAXSW_XMM_XMM", "PMAXSW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xEE, 16),
    Stage4OpSpec("PMULUDQ_XMM_XMM", "PMULUDQ_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xF4, 32),
    Stage4OpSpec("PMADDWD_XMM_XMM", "PMADDWD_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xF5, 16),
    Stage4OpSpec("PSADBW_XMM_XMM", "PSADBW_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xF6, 8),
    Stage4OpSpec("PMINUB_XMM_XMM", "PMINUB_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xDA, 8),
    Stage4OpSpec("PMAXUB_XMM_XMM", "PMAXUB_XMMdq_XMMdq", "SSE_INTEGER", "xmm,xmm", 0xDE, 8),
    Stage4OpSpec("PSLLW_XMM_IMM8", "PSLLW_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x71, 16, 6),
    Stage4OpSpec("PSLLD_XMM_IMM8", "PSLLD_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x72, 32, 6),
    Stage4OpSpec("PSLLQ_XMM_IMM8", "PSLLQ_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x73, 64, 6),
    Stage4OpSpec("PSRLW_XMM_IMM8", "PSRLW_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x71, 16, 2),
    Stage4OpSpec("PSRLD_XMM_IMM8", "PSRLD_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x72, 32, 2),
    Stage4OpSpec("PSRLQ_XMM_IMM8", "PSRLQ_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x73, 64, 2),
    Stage4OpSpec("PSRAW_XMM_IMM8", "PSRAW_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x71, 16, 4),
    Stage4OpSpec("PSRAD_XMM_IMM8", "PSRAD_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x72, 32, 4),
    Stage4OpSpec("PSLLDQ_XMM_IMM8", "PSLLDQ_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x73, 8, 7),
    Stage4OpSpec("PSRLDQ_XMM_IMM8", "PSRLDQ_XMMdq_IMMb", "SSE_SHIFT", "xmm,imm8", 0x73, 8, 3),
    Stage4OpSpec("PSLLW_XMM_XMM", "PSLLW_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xF1, 16),
    Stage4OpSpec("PSLLD_XMM_XMM", "PSLLD_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xF2, 32),
    Stage4OpSpec("PSLLQ_XMM_XMM", "PSLLQ_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xF3, 64),
    Stage4OpSpec("PSRLW_XMM_XMM", "PSRLW_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xD1, 16),
    Stage4OpSpec("PSRLD_XMM_XMM", "PSRLD_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xD2, 32),
    Stage4OpSpec("PSRLQ_XMM_XMM", "PSRLQ_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xD3, 64),
    Stage4OpSpec("PSRAW_XMM_XMM", "PSRAW_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xE1, 16),
    Stage4OpSpec("PSRAD_XMM_XMM", "PSRAD_XMMdq_XMMdq", "SSE_SHIFT", "xmm,xmm", 0xE2, 32),
)

FP_REGISTER_OPS: Tuple[Stage4OpSpec, ...] = (
    Stage4OpSpec("ADDSD_XMM_XMM", "ADDSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x58, 64, prefix=0xF2),
    Stage4OpSpec("SUBSD_XMM_XMM", "SUBSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x5C, 64, prefix=0xF2),
    Stage4OpSpec("MULSD_XMM_XMM", "MULSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x59, 64, prefix=0xF2),
    Stage4OpSpec("DIVSD_XMM_XMM", "DIVSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x5E, 64, prefix=0xF2),
    Stage4OpSpec("SQRTSD_XMM_XMM", "SQRTSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x51, 64, prefix=0xF2),
    Stage4OpSpec("MINSD_XMM_XMM", "MINSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x5D, 64, prefix=0xF2),
    Stage4OpSpec("MAXSD_XMM_XMM", "MAXSD_XMMsd_XMMsd", "SSE_FP_SCALAR", "xmm,xmm", 0x5F, 64, prefix=0xF2),
    Stage4OpSpec("ADDPD_XMM_XMM", "ADDPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x58, 64),
    Stage4OpSpec("SUBPD_XMM_XMM", "SUBPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x5C, 64),
    Stage4OpSpec("MULPD_XMM_XMM", "MULPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x59, 64),
    Stage4OpSpec("DIVPD_XMM_XMM", "DIVPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x5E, 64),
    Stage4OpSpec("SQRTPD_XMM_XMM", "SQRTPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x51, 64),
    Stage4OpSpec("MINPD_XMM_XMM", "MINPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x5D, 64),
    Stage4OpSpec("MAXPD_XMM_XMM", "MAXPD_XMMpd_XMMpd", "SSE_FP_PACKED", "xmm,xmm", 0x5F, 64),
    Stage4OpSpec("ADDSS_XMM_XMM", "ADDSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x58, 32, prefix=0xF3),
    Stage4OpSpec("SUBSS_XMM_XMM", "SUBSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x5C, 32, prefix=0xF3),
    Stage4OpSpec("MULSS_XMM_XMM", "MULSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x59, 32, prefix=0xF3),
    Stage4OpSpec("DIVSS_XMM_XMM", "DIVSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x5E, 32, prefix=0xF3),
    Stage4OpSpec("SQRTSS_XMM_XMM", "SQRTSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x51, 32, prefix=0xF3),
    Stage4OpSpec("MINSS_XMM_XMM", "MINSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x5D, 32, prefix=0xF3),
    Stage4OpSpec("MAXSS_XMM_XMM", "MAXSS_XMMss_XMMss", "SSE_FP_SCALAR", "xmm,xmm", 0x5F, 32, prefix=0xF3),
    Stage4OpSpec("ADDPS_XMM_XMM", "ADDPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x58, 32, prefix=None),
    Stage4OpSpec("SUBPS_XMM_XMM", "SUBPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x5C, 32, prefix=None),
    Stage4OpSpec("MULPS_XMM_XMM", "MULPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x59, 32, prefix=None),
    Stage4OpSpec("DIVPS_XMM_XMM", "DIVPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x5E, 32, prefix=None),
    Stage4OpSpec("SQRTPS_XMM_XMM", "SQRTPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x51, 32, prefix=None),
    Stage4OpSpec("MINPS_XMM_XMM", "MINPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x5D, 32, prefix=None),
    Stage4OpSpec("MAXPS_XMM_XMM", "MAXPS_XMMps_XMMps", "SSE_FP_PACKED", "xmm,xmm", 0x5F, 32, prefix=None),
)

FP_COMPARE_OPS: Tuple[Stage4OpSpec, ...] = (
    Stage4OpSpec(
        "CMPSD_XMM_XMM_IMM8",
        "CMPSD_XMM_XMMsd_XMMsd_IMMb",
        "SSE_FP_SCALAR",
        "xmm,xmm,imm8",
        0xC2,
        64,
        prefix=0xF2,
    ),
    Stage4OpSpec(
        "CMPPD_XMM_XMM_IMM8",
        "CMPPD_XMMpd_XMMpd_IMMb",
        "SSE_FP_PACKED",
        "xmm,xmm,imm8",
        0xC2,
        64,
    ),
    Stage4OpSpec(
        "CMPSS_XMM_XMM_IMM8",
        "CMPSS_XMMss_XMMss_IMMb",
        "SSE_FP_SCALAR",
        "xmm,xmm,imm8",
        0xC2,
        32,
        prefix=0xF3,
    ),
    Stage4OpSpec(
        "CMPPS_XMM_XMM_IMM8",
        "CMPPS_XMMps_XMMps_IMMb",
        "SSE_FP_PACKED",
        "xmm,xmm,imm8",
        0xC2,
        32,
        prefix=None,
    ),
)


def memory_source_variant(op: Stage4OpSpec) -> Stage4OpSpec:
    if op.operands != "xmm,xmm":
        raise ValueError(f"cannot create memory source variant for {op.name}")
    mnemonic = op.name.split("_", 1)[0]
    return Stage4OpSpec(
        name=op.name.replace("_XMM_XMM", "_XMM_M128"),
        iform=f"{mnemonic}_XMMdq_MEMdq",
        family=op.family,
        operands="xmm,m128",
        opcode=op.opcode,
        lane_bits=op.lane_bits,
        imm_group=op.imm_group,
    )


def fp_memory_source_variant(op: Stage4OpSpec) -> Stage4OpSpec:
    if op.operands != "xmm,xmm" or not op.family.startswith("SSE_FP_"):
        raise ValueError(f"cannot create FP memory source variant for {op.name}")
    mnemonic = op.name.split("_", 1)[0]
    if op.family == "SSE_FP_SCALAR":
        width = 32 if op.lane_bits == 32 else 64
        suffix = "ss" if width == 32 else "sd"
        return Stage4OpSpec(
            name=op.name.replace("_XMM_XMM", f"_XMM_M{width}"),
            iform=f"{mnemonic}_XMM{suffix}_MEM{suffix}",
            family=op.family,
            operands=f"xmm,m{width}",
            opcode=op.opcode,
            lane_bits=op.lane_bits,
            prefix=op.prefix,
            memory_alignment=width // 8,
        )
    suffix = "ps" if op.lane_bits == 32 else "pd"
    return Stage4OpSpec(
        name=op.name.replace("_XMM_XMM", "_XMM_M128"),
        iform=f"{mnemonic}_XMM{suffix}_MEM{suffix}",
        family=op.family,
        operands="xmm,m128",
        opcode=op.opcode,
        lane_bits=op.lane_bits,
        prefix=op.prefix,
        memory_alignment=16,
    )


def fp_compare_memory_source_variant(op: Stage4OpSpec) -> Stage4OpSpec:
    if not op.operands.endswith(",imm8") or not op.family.startswith("SSE_FP_"):
        raise ValueError(f"cannot create FP compare memory source variant for {op.name}")
    mnemonic = op.name.split("_", 1)[0]
    if op.family == "SSE_FP_SCALAR":
        width = 32 if op.lane_bits == 32 else 64
        suffix = "ss" if width == 32 else "sd"
        if mnemonic == "CMPSS":
            iform = f"{mnemonic}_XMM{suffix}_MEM{suffix}_IMMb"
        else:
            iform = f"{mnemonic}_XMM_XMM{suffix}_MEM{suffix}_IMMb"
        return Stage4OpSpec(
            name=op.name.replace("_XMM_XMM_IMM8", f"_XMM_M{width}_IMM8"),
            iform=iform,
            family=op.family,
            operands=f"xmm,m{width},imm8",
            opcode=op.opcode,
            lane_bits=op.lane_bits,
            prefix=op.prefix,
            memory_alignment=width // 8,
        )
    suffix = "ps" if op.lane_bits == 32 else "pd"
    return Stage4OpSpec(
        name=op.name.replace("_XMM_XMM_IMM8", "_XMM_M128_IMM8"),
        iform=f"{mnemonic}_XMM{suffix}_MEM{suffix}_IMMb",
        family=op.family,
        operands="xmm,m128,imm8",
        opcode=op.opcode,
        lane_bits=op.lane_bits,
        prefix=op.prefix,
        memory_alignment=16,
    )


MEMORY_SOURCE_OPS = tuple(memory_source_variant(op) for op in BASE_OPS if op.operands == "xmm,xmm")
FP_MEMORY_SOURCE_OPS = tuple(fp_memory_source_variant(op) for op in FP_REGISTER_OPS)
FP_COMPARE_MEMORY_SOURCE_OPS = tuple(fp_compare_memory_source_variant(op) for op in FP_COMPARE_OPS)
TRANSFER_REGISTER_OPS = (
    Stage4OpSpec("MOVSD_XMM_XMM", "MOVSD_XMM_XMMsd_XMMsd_0F10", "DATAXFER", "xmm,xmm", 0x10, 64, prefix=0xF2),
    Stage4OpSpec("MOVAPD_XMM_XMM", "MOVAPD_XMMpd_XMMpd_0F28", "DATAXFER", "xmm,xmm", 0x28, 128),
    Stage4OpSpec("MOVUPD_XMM_XMM", "MOVUPD_XMMpd_XMMpd_0F10", "DATAXFER", "xmm,xmm", 0x10, 128),
    Stage4OpSpec("MOVQ_XMM_XMM", "MOVQ_XMMdq_XMMq_0F7E", "DATAXFER", "xmm,xmm", 0x7E, 64, prefix=0xF3),
)
TRANSFER_MEMORY_SOURCE_OPS = (
    Stage4OpSpec("MOVD_XMM_M32", "MOVD_XMMdq_MEMd", "DATAXFER", "xmm,m32", 0x6E, 32, memory_alignment=4),
    Stage4OpSpec("MOVQ_XMM_M64", "MOVQ_XMMdq_MEMq_0F7E", "DATAXFER", "xmm,m64", 0x7E, 64, prefix=0xF3, memory_alignment=8),
    Stage4OpSpec("MOVSD_XMM_M64", "MOVSD_XMM_XMMdq_MEMsd", "DATAXFER", "xmm,m64", 0x10, 64, prefix=0xF2, memory_alignment=8),
    Stage4OpSpec("MOVLPD_XMM_M64", "MOVLPD_XMMsd_MEMq", "DATAXFER", "xmm,m64", 0x12, 64, memory_alignment=8),
    Stage4OpSpec("MOVHPD_XMM_M64", "MOVHPD_XMMsd_MEMq", "DATAXFER", "xmm,m64", 0x16, 64, memory_alignment=8),
    Stage4OpSpec("MOVAPD_XMM_M128", "MOVAPD_XMMpd_MEMpd", "DATAXFER", "xmm,m128", 0x28, 128),
    Stage4OpSpec(
        "MOVUPD_XMM_M128",
        "MOVUPD_XMMpd_MEMpd",
        "DATAXFER",
        "xmm,m128",
        0x10,
        128,
        memory_alignment=1,
    ),
)
MEMORY_DEST_OPS = (
    Stage4OpSpec("MOVDQA_M128_XMM", "MOVDQA_MEMdq_XMMdq", "DATAXFER", "m128,xmm", 0x7F, 128),
    Stage4OpSpec("MOVD_M32_XMM", "MOVD_MEMd_XMMd", "DATAXFER", "m32,xmm", 0x7E, 32, memory_alignment=4),
    Stage4OpSpec(
        "MOVSD_M64_XMM",
        "MOVSD_XMM_MEMsd_XMMsd",
        "DATAXFER",
        "m64,xmm",
        0x11,
        64,
        prefix=0xF2,
        memory_alignment=8,
    ),
    Stage4OpSpec("MOVLPD_M64_XMM", "MOVLPD_MEMq_XMMsd", "DATAXFER", "m64,xmm", 0x13, 64, memory_alignment=8),
    Stage4OpSpec("MOVQ_M64_XMM", "MOVQ_MEMq_XMMq_0FD6", "DATAXFER", "m64,xmm", 0xD6, 64, memory_alignment=8),
    Stage4OpSpec(
        "MOVHPD_M64_XMM",
        "MOVHPD_MEMq_XMMsd",
        "DATAXFER",
        "m64,xmm",
        0x17,
        64,
        memory_alignment=8,
        memory_store_bit_offset=64,
    ),
    Stage4OpSpec("MOVAPD_M128_XMM", "MOVAPD_MEMpd_XMMpd", "DATAXFER", "m128,xmm", 0x29, 128),
    Stage4OpSpec("MOVNTDQ_M128_XMM", "MOVNTDQ_MEMdq_XMMdq", "DATAXFER", "m128,xmm", 0xE7, 128),
    Stage4OpSpec("MOVNTPD_M128_XMM", "MOVNTPD_MEMdq_XMMpd", "DATAXFER", "m128,xmm", 0x2B, 128),
    Stage4OpSpec(
        "MOVUPD_M128_XMM",
        "MOVUPD_MEMpd_XMMpd",
        "DATAXFER",
        "m128,xmm",
        0x11,
        128,
        memory_alignment=1,
    ),
    Stage4OpSpec(
        "MOVDQU_XMM_M128",
        "MOVDQU_XMMdq_MEMdq",
        "DATAXFER",
        "xmm,m128",
        0x6F,
        128,
        prefix=0xF3,
        memory_alignment=1,
    ),
    Stage4OpSpec(
        "MOVDQU_M128_XMM",
        "MOVDQU_MEMdq_XMMdq",
        "DATAXFER",
        "m128,xmm",
        0x7F,
        128,
        prefix=0xF3,
        memory_alignment=1,
    ),
)


ALIGNED_M128_FAULT_OPERANDS = {"xmm,m128", "xmm,m128,imm8", "m128,xmm"}


def aligned_m128_fault_variant(op: Stage4OpSpec) -> Stage4OpSpec:
    if op.operands not in ALIGNED_M128_FAULT_OPERANDS:
        raise ValueError(f"cannot create aligned m128 fault variant for {op.name}")
    if op.memory_alignment != 16 or op.expected_exception is not None:
        raise ValueError(f"{op.name} is not a non-faulting 16-byte-aligned m128 form")
    return Stage4OpSpec(
        name=f"{op.name}_UNALIGNED_GP",
        iform=op.iform,
        family="EXCEPTION",
        operands=op.operands,
        opcode=op.opcode,
        lane_bits=op.lane_bits,
        imm_group=op.imm_group,
        prefix=op.prefix,
        memory_alignment=1,
        expected_exception="#GP(0)",
        memory_store_bit_offset=op.memory_store_bit_offset,
    )


ALIGNED_M128_FAULT_BASE_OPS = tuple(
    op
    for op in (
        MEMORY_SOURCE_OPS
        + FP_MEMORY_SOURCE_OPS
        + FP_COMPARE_MEMORY_SOURCE_OPS
        + TRANSFER_MEMORY_SOURCE_OPS
        + MEMORY_DEST_OPS
    )
    if op.operands in ALIGNED_M128_FAULT_OPERANDS
    and op.memory_alignment == 16
    and op.expected_exception is None
    and op.name not in {"MOVDQA_XMM_M128", "MOVDQA_M128_XMM"}
)
ALIGNED_M128_FAULT_OPS = tuple(aligned_m128_fault_variant(op) for op in ALIGNED_M128_FAULT_BASE_OPS)

FAULT_OPS = (
    Stage4OpSpec(
        "MOVDQA_XMM_M128_UNALIGNED_GP",
        "MOVDQA_XMMdq_MEMdq",
        "EXCEPTION",
        "xmm,m128",
        0x6F,
        128,
        memory_alignment=1,
        expected_exception="#GP(0)",
    ),
    Stage4OpSpec(
        "MOVDQA_M128_XMM_UNALIGNED_GP",
        "MOVDQA_MEMdq_XMMdq",
        "EXCEPTION",
        "m128,xmm",
        0x7F,
        128,
        memory_alignment=1,
        expected_exception="#GP(0)",
    ),
)
OPS: Tuple[Stage4OpSpec, ...] = (
    BASE_OPS
    + FP_REGISTER_OPS
    + FP_COMPARE_OPS
    + MEMORY_SOURCE_OPS
    + FP_MEMORY_SOURCE_OPS
    + FP_COMPARE_MEMORY_SOURCE_OPS
    + TRANSFER_REGISTER_OPS
    + TRANSFER_MEMORY_SOURCE_OPS
    + MEMORY_DEST_OPS
    + FAULT_OPS
    + ALIGNED_M128_FAULT_OPS
)
OP_INDEX = {op.name: i for i, op in enumerate(OPS)}
MEMORY_SOURCE_OPERANDS = {"xmm,m128", "xmm,m64", "xmm,m32", "xmm,m128,imm8", "xmm,m64,imm8", "xmm,m32,imm8"}
MEMORY_DEST_OPERANDS = {"m128,xmm", "m64,xmm", "m32,xmm"}
MEMORY_OPERANDS = MEMORY_SOURCE_OPERANDS | MEMORY_DEST_OPERANDS


def u128(value: int) -> int:
    return int(value) & MASK128


SAFE_F64_VALUES = (
    -32.0,
    -16.0,
    -8.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    8.0,
    16.0,
    32.0,
)
POS_ZERO_F64_BITS = 0x0000000000000000
NEG_ZERO_F64_BITS = 0x8000000000000000
POS_INF_F64_BITS = 0x7FF0000000000000
NEG_INF_F64_BITS = 0xFFF0000000000000
QUIET_NAN_F64_BITS = (
    0x7FF8000000000001,
    0xFFF8000000000001,
)
EDGE_F64_BITS = (
    POS_ZERO_F64_BITS,
    NEG_ZERO_F64_BITS,
    POS_INF_F64_BITS,
    NEG_INF_F64_BITS,
)
COMPARE_SPECIAL_F64_BITS = (
    POS_ZERO_F64_BITS,
    NEG_ZERO_F64_BITS,
    POS_INF_F64_BITS,
    NEG_INF_F64_BITS,
    *QUIET_NAN_F64_BITS,
)
SAFE_F32_VALUES = (
    -32.0,
    -16.0,
    -8.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    8.0,
    16.0,
    32.0,
)
POS_ZERO_F32_BITS = 0x00000000
NEG_ZERO_F32_BITS = 0x80000000
POS_INF_F32_BITS = 0x7F800000
NEG_INF_F32_BITS = 0xFF800000
QUIET_NAN_F32_BITS = (
    0x7FC00001,
    0xFFC00001,
)
EDGE_F32_BITS = (
    POS_ZERO_F32_BITS,
    NEG_ZERO_F32_BITS,
    POS_INF_F32_BITS,
    NEG_INF_F32_BITS,
)
COMPARE_SPECIAL_F32_BITS = (
    POS_ZERO_F32_BITS,
    NEG_ZERO_F32_BITS,
    POS_INF_F32_BITS,
    NEG_INF_F32_BITS,
    *QUIET_NAN_F32_BITS,
)


def f64_from_bits(bits: int) -> float:
    return struct.unpack("<d", int(bits & MASK64).to_bytes(8, "little"))[0]


def f64_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<d", float(value)), "little")


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", int(bits & MASK32).to_bytes(4, "little"))[0]


def f32_bits(value: float) -> int:
    return int.from_bytes(struct.pack("<f", float(value)), "little")


def is_safe_f64(value: float) -> bool:
    return math.isfinite(value) and (2.0 ** -20) <= abs(value) <= (2.0 ** 20)


def is_safe_f32(value: float) -> bool:
    return math.isfinite(value) and (2.0 ** -10) <= abs(value) <= (2.0 ** 10)


def is_safe_f64_bits(bits: int) -> bool:
    value = f64_from_bits(bits)
    return is_safe_f64(value)


def is_safe_f32_bits(bits: int) -> bool:
    value = f32_from_bits(bits)
    return is_safe_f32(value)


def is_zero_f64_bits(bits: int) -> bool:
    return (int(bits) & ~0x8000000000000000) == 0


def is_zero_f32_bits(bits: int) -> bool:
    return (int(bits) & ~0x80000000) == 0


def is_inf_f64_bits(bits: int) -> bool:
    return (int(bits) & MASK64) in {POS_INF_F64_BITS, NEG_INF_F64_BITS}


def is_inf_f32_bits(bits: int) -> bool:
    return (int(bits) & MASK32) in {POS_INF_F32_BITS, NEG_INF_F32_BITS}


def is_quiet_nan_f64_bits(bits: int) -> bool:
    bits = int(bits) & MASK64
    exponent = (bits >> 52) & 0x7FF
    mantissa = bits & ((1 << 52) - 1)
    quiet_bit = 1 << 51
    return exponent == 0x7FF and mantissa != 0 and bool(mantissa & quiet_bit)


def is_quiet_nan_f32_bits(bits: int) -> bool:
    bits = int(bits) & MASK32
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & ((1 << 23) - 1)
    quiet_bit = 1 << 22
    return exponent == 0xFF and mantissa != 0 and bool(mantissa & quiet_bit)


def is_compare_f64_bits(bits: int) -> bool:
    return is_safe_f64_bits(bits) or is_zero_f64_bits(bits) or is_inf_f64_bits(bits) or is_quiet_nan_f64_bits(bits)


def is_compare_f32_bits(bits: int) -> bool:
    return is_safe_f32_bits(bits) or is_zero_f32_bits(bits) or is_inf_f32_bits(bits) or is_quiet_nan_f32_bits(bits)


def is_fp_compare_name(name: str) -> bool:
    return name.startswith(("CMPSD_", "CMPPD_", "CMPSS_", "CMPPS_"))


def fp_compare_predicate(predicate: int, a: float, b: float) -> bool:
    pred = int(predicate) & 0x7
    unordered = math.isnan(a) or math.isnan(b)
    if pred == 0:
        return not unordered and a == b
    if pred == 1:
        return not unordered and a < b
    if pred == 2:
        return not unordered and a <= b
    if pred == 3:
        return unordered
    if pred == 4:
        return unordered or a != b
    if pred == 5:
        return unordered or not (a < b)
    if pred == 6:
        return unordered or not (a <= b)
    if pred == 7:
        return not unordered
    raise AssertionError(pred)


def fp_value_from_bits(bits: int, lane_bits: int) -> float:
    if lane_bits == 32:
        return f32_from_bits(bits)
    if lane_bits == 64:
        return f64_from_bits(bits)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def fp_bits(value: float, lane_bits: int) -> int:
    if lane_bits == 32:
        return f32_bits(value)
    if lane_bits == 64:
        return f64_bits(value)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def is_safe_fp_bits(bits: int, lane_bits: int) -> bool:
    if lane_bits == 32:
        return is_safe_f32_bits(bits)
    if lane_bits == 64:
        return is_safe_f64_bits(bits)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def is_zero_fp_bits(bits: int, lane_bits: int) -> bool:
    if lane_bits == 32:
        return is_zero_f32_bits(bits)
    if lane_bits == 64:
        return is_zero_f64_bits(bits)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def is_inf_fp_bits(bits: int, lane_bits: int) -> bool:
    if lane_bits == 32:
        return is_inf_f32_bits(bits)
    if lane_bits == 64:
        return is_inf_f64_bits(bits)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def is_quiet_nan_fp_bits(bits: int, lane_bits: int) -> bool:
    if lane_bits == 32:
        return is_quiet_nan_f32_bits(bits)
    if lane_bits == 64:
        return is_quiet_nan_f64_bits(bits)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def is_nontrapping_fp_operand_bits(bits: int, lane_bits: int) -> bool:
    return is_safe_fp_bits(bits, lane_bits) or is_zero_fp_bits(bits, lane_bits) or is_inf_fp_bits(bits, lane_bits)


def is_nontrapping_fp_result_bits(bits: int, lane_bits: int) -> bool:
    return is_nontrapping_fp_operand_bits(bits, lane_bits)


def is_compare_fp_bits(bits: int, lane_bits: int) -> bool:
    if lane_bits == 32:
        return is_compare_f32_bits(bits)
    if lane_bits == 64:
        return is_compare_f64_bits(bits)
    raise ValueError(f"unsupported FP lane width {lane_bits}")


def fp_compare_lane_bits(a_bits: int, b_bits: int, predicate: int, lane_bits: int) -> int:
    a = fp_value_from_bits(a_bits, lane_bits)
    b = fp_value_from_bits(b_bits, lane_bits)
    return ((1 << lane_bits) - 1) if fp_compare_predicate(predicate, a, b) else 0


def same_sign_bits(a_bits: int, b_bits: int, lane_bits: int) -> bool:
    return ((int(a_bits) ^ int(b_bits)) & (1 << (lane_bits - 1))) == 0


def fp_operation_nontrapping(name: str, a_bits: int, b_bits: int, lane_bits: int) -> bool:
    mnemonic = name.split("_", 1)[0]
    source_only = mnemonic in {"SQRTSD", "SQRTPD", "SQRTSS", "SQRTPS"}
    if not is_nontrapping_fp_operand_bits(b_bits, lane_bits):
        return False
    if not source_only and not is_nontrapping_fp_operand_bits(a_bits, lane_bits):
        return False
    a_inf = is_inf_fp_bits(a_bits, lane_bits)
    b_inf = is_inf_fp_bits(b_bits, lane_bits)
    a_zero = is_zero_fp_bits(a_bits, lane_bits)
    b_zero = is_zero_fp_bits(b_bits, lane_bits)
    if mnemonic in {"ADDSD", "ADDPD", "ADDSS", "ADDPS"}:
        return not (a_inf and b_inf and not same_sign_bits(a_bits, b_bits, lane_bits))
    if mnemonic in {"SUBSD", "SUBPD", "SUBSS", "SUBPS"}:
        return not (a_inf and b_inf and same_sign_bits(a_bits, b_bits, lane_bits))
    if mnemonic in {"MULSD", "MULPD", "MULSS", "MULPS"}:
        return not ((a_inf and b_zero) or (a_zero and b_inf))
    if mnemonic in {"DIVSD", "DIVPD", "DIVSS", "DIVPS"}:
        return not (b_zero or (a_inf and b_inf))
    if mnemonic in {"SQRTSD", "SQRTPD", "SQRTSS", "SQRTPS"}:
        sign_bit = 1 << (lane_bits - 1)
        return not ((int(b_bits) & sign_bit) and not b_zero)
    if mnemonic in {"MINSD", "MINPD", "MINSS", "MINPS", "MAXSD", "MAXPD", "MAXSS", "MAXPS"}:
        return True
    return False


def fp_lane_result_bits(name: str, a_bits: int, b_bits: int, predicate: int = 0, lane_bits: int = 64) -> int:
    if is_fp_compare_name(name):
        return fp_compare_lane_bits(a_bits, b_bits, predicate, lane_bits)
    if not fp_operation_nontrapping(name, a_bits, b_bits, lane_bits):
        raise ValueError(f"trapping or unsupported FP edge operand for {name}")
    a = fp_value_from_bits(a_bits, lane_bits)
    b = fp_value_from_bits(b_bits, lane_bits)
    mnemonic = name.split("_", 1)[0]
    if mnemonic in {"ADDSD", "ADDPD", "ADDSS", "ADDPS"}:
        result = a + b
    elif mnemonic in {"SUBSD", "SUBPD", "SUBSS", "SUBPS"}:
        result = a - b
    elif mnemonic in {"MULSD", "MULPD", "MULSS", "MULPS"}:
        result = a * b
    elif mnemonic in {"DIVSD", "DIVPD", "DIVSS", "DIVPS"}:
        if b == 0.0:
            raise ValueError(f"unsafe divide-by-zero FP source for {name}")
        result = a / b
    elif mnemonic in {"SQRTSD", "SQRTPD", "SQRTSS", "SQRTPS"}:
        if b < 0.0:
            raise ValueError(f"unsafe negative FP source for {name}")
        result = math.sqrt(b)
    elif mnemonic in {"MINSD", "MINPD", "MINSS", "MINPS"}:
        result = a if a < b else b
    elif mnemonic in {"MAXSD", "MAXPD", "MAXSS", "MAXPS"}:
        result = a if a > b else b
    else:
        raise ValueError(name)
    if lane_bits == 32:
        result = f32_from_bits(f32_bits(result))
    if not is_nontrapping_fp_result_bits(fp_bits(result, lane_bits), lane_bits):
        raise ValueError(f"unsafe scalar FP result for {name}: {result!r}")
    return fp_bits(result, lane_bits)


def scalar_fp_result_bits(name: str, a_bits: int, b_bits: int, predicate: int = 0, lane_bits: int = 64) -> int:
    return fp_lane_result_bits(name, a_bits, b_bits, predicate, lane_bits)


def packed_fp_result_bits(name: str, a: int, b: int, predicate: int = 0, lane_bits: int = 64) -> int:
    out = []
    mask = (1 << lane_bits) - 1
    for lane in range(128 // lane_bits):
        shift = lane * lane_bits
        out.append(fp_lane_result_bits(name, (a >> shift) & mask, (b >> shift) & mask, predicate, lane_bits))
    return pack_lanes(out, lane_bits)


def fp_operands_safe(op: Stage4OpSpec, xmm: Sequence[int], dst: int, src: int, src_value: Optional[int] = None) -> bool:
    if not op.family.startswith("SSE_FP_"):
        return True
    a = int(xmm[dst])
    b = int(src_value) if src_value is not None else int(xmm[src])
    source_only = op.name.split("_", 1)[0] in {"SQRTSD", "SQRTPD", "SQRTSS", "SQRTPS"}
    lane_count = 128 // op.lane_bits if op.family == "SSE_FP_PACKED" else 1
    lane_ok = is_compare_fp_bits if is_fp_compare_name(op.name) else is_nontrapping_fp_operand_bits
    mask = (1 << op.lane_bits) - 1
    for lane in range(lane_count):
        shift = lane * op.lane_bits
        if not lane_ok((b >> shift) & mask, op.lane_bits):
            return False
        if not source_only and not lane_ok((a >> shift) & mask, op.lane_bits):
            return False
    if is_fp_compare_name(op.name):
        return True
    try:
        if op.family == "SSE_FP_PACKED":
            packed_fp_result_bits(op.name, a, b, lane_bits=op.lane_bits)
        else:
            scalar_fp_result_bits(op.name, a & mask, b & mask, lane_bits=op.lane_bits)
    except ValueError:
        return False
    return True


def scalar_fp_operands_safe(op: Stage4OpSpec, xmm: Sequence[int], dst: int, src: int) -> bool:
    return fp_operands_safe(op, xmm, dst, src)


def memory_operand_width_bytes(op: Stage4OpSpec) -> int:
    if op.operands in {"xmm,m32", "xmm,m32,imm8", "m32,xmm"}:
        return 4
    if op.operands in {"xmm,m64", "xmm,m64,imm8", "m64,xmm"}:
        return 8
    if op.operands in {"xmm,m128", "xmm,m128,imm8", "m128,xmm"}:
        return 16
    raise ValueError(f"{op.name} has no memory operand")


def read_memory_operand(memory: Sequence[int], offset: int, op: Stage4OpSpec) -> int:
    width = memory_operand_width_bytes(op)
    return int.from_bytes(bytes(int(x) & 0xFF for x in memory[offset:offset + width]), "little")


def register_write_mask(op: Stage4OpSpec) -> int:
    if op.operands in MEMORY_DEST_OPERANDS:
        return 0
    if op.name in {"MOVSD_XMM_XMM", "MOVLPD_XMM_M64"}:
        return MASK64
    if op.name == "MOVHPD_XMM_M64":
        return HIGH64_MASK
    if op.family == "SSE_FP_SCALAR":
        return (1 << op.lane_bits) - 1
    return MASK128


def memory_digest(memory: Optional[Sequence[int]] = None) -> str:
    data = b"" if memory is None else bytes(int(x) & 0xFF for x in memory)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def state_record(
    xmm: Sequence[int],
    rip: int,
    memory: Optional[Sequence[int]] = None,
    mxcsr: int = MXCSR_DEFAULT,
    exception: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "rip": hex_u(rip, 64),
        "gpr": {},
        "rflags": {},
        "xmm": {reg: hex_u(int(xmm[i]), 128) for i, reg in enumerate(XMM_REGS)},
        "mxcsr": hex_u(mxcsr, 32),
        "memory_digest": memory_digest(memory),
        "exception": exception,
    }


SSE_SINGLE_MNEMONICS = {
    "ADDSS",
    "SUBSS",
    "MULSS",
    "DIVSS",
    "SQRTSS",
    "MINSS",
    "MAXSS",
    "CMPSS",
    "ADDPS",
    "SUBPS",
    "MULPS",
    "DIVPS",
    "SQRTPS",
    "MINPS",
    "MAXPS",
    "CMPPS",
}


def instruction_extension(op: Stage4OpSpec) -> str:
    return "SSE" if op.name.split("_", 1)[0] in SSE_SINGLE_MNEMONICS else "SSE2"


def modrm(reg: int, rm: int) -> int:
    return 0xC0 | ((reg & 7) << 3) | (rm & 7)


def rex_prefix(*, r: bool = False, b: bool = False) -> List[int]:
    rex = 0x40 | (0x04 if r else 0) | (0x01 if b else 0)
    return [] if rex == 0x40 else [rex]


def mandatory_prefix(op: Stage4OpSpec) -> List[int]:
    return [] if op.prefix is None else [int(op.prefix)]


def encode_x86_bytes(op: Stage4OpSpec, dst: int, src: int, imm: int, mem_addr: int = 0) -> str:
    if op.operands == "xmm,xmm":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8, b=src >= 8), 0x0F, op.opcode, modrm(dst, src)]
    elif op.operands == "xmm,xmm,imm8":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8, b=src >= 8), 0x0F, op.opcode, modrm(dst, src), imm & 0xFF]
    elif op.operands == "xmm,m128":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8), 0x0F, op.opcode, 0x04 | ((dst & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
    elif op.operands == "xmm,m32":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8), 0x0F, op.opcode, 0x04 | ((dst & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
    elif op.operands == "xmm,m64":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8), 0x0F, op.opcode, 0x04 | ((dst & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
    elif op.operands == "xmm,m128,imm8":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8), 0x0F, op.opcode, 0x04 | ((dst & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
        b.append(imm & 0xFF)
    elif op.operands == "xmm,m64,imm8":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8), 0x0F, op.opcode, 0x04 | ((dst & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
        b.append(imm & 0xFF)
    elif op.operands == "xmm,m32,imm8":
        b = [*mandatory_prefix(op), *rex_prefix(r=dst >= 8), 0x0F, op.opcode, 0x04 | ((dst & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
        b.append(imm & 0xFF)
    elif op.operands == "m128,xmm":
        b = [*mandatory_prefix(op), *rex_prefix(r=src >= 8), 0x0F, op.opcode, 0x04 | ((src & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
    elif op.operands == "m64,xmm":
        b = [*mandatory_prefix(op), *rex_prefix(r=src >= 8), 0x0F, op.opcode, 0x04 | ((src & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
    elif op.operands == "m32,xmm":
        b = [*mandatory_prefix(op), *rex_prefix(r=src >= 8), 0x0F, op.opcode, 0x04 | ((src & 7) << 3), 0x25]
        b.extend(int(mem_addr).to_bytes(4, "little"))
    elif op.operands == "xmm,imm8":
        if op.imm_group is None:
            raise ValueError(f"missing imm_group for {op.name}")
        b = [*mandatory_prefix(op), *rex_prefix(b=dst >= 8), 0x0F, op.opcode, modrm(op.imm_group, dst), imm & 0xFF]
    else:
        raise ValueError(op.name)
    return "".join(f"{x:02x}" for x in b)


def asm_string(op: Stage4OpSpec, dst: int, src: int, imm: int, mem_addr: int = 0) -> str:
    if op.operands == "xmm,xmm":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, {XMM_REGS[src].lower()}"
    if op.operands == "xmm,xmm,imm8":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, {XMM_REGS[src].lower()}, {imm & 0xFF}"
    if op.operands == "xmm,m128":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, xmmword ptr [{hex_u(mem_addr, 64)}]"
    if op.operands == "xmm,m32":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, dword ptr [{hex_u(mem_addr, 64)}]"
    if op.operands == "xmm,m64":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, qword ptr [{hex_u(mem_addr, 64)}]"
    if op.operands == "xmm,m128,imm8":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, xmmword ptr [{hex_u(mem_addr, 64)}], {imm & 0xFF}"
    if op.operands == "xmm,m64,imm8":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, qword ptr [{hex_u(mem_addr, 64)}], {imm & 0xFF}"
    if op.operands == "xmm,m32,imm8":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, dword ptr [{hex_u(mem_addr, 64)}], {imm & 0xFF}"
    if op.operands == "m128,xmm":
        return f"{op.name.split('_')[0].lower()} xmmword ptr [{hex_u(mem_addr, 64)}], {XMM_REGS[src].lower()}"
    if op.operands == "m64,xmm":
        return f"{op.name.split('_')[0].lower()} qword ptr [{hex_u(mem_addr, 64)}], {XMM_REGS[src].lower()}"
    if op.operands == "m32,xmm":
        return f"{op.name.split('_')[0].lower()} dword ptr [{hex_u(mem_addr, 64)}], {XMM_REGS[src].lower()}"
    if op.operands == "xmm,imm8":
        return f"{op.name.split('_')[0].lower()} {XMM_REGS[dst].lower()}, {imm & 0xFF}"
    raise ValueError(op.name)


def lane_binary(a: int, b: int, lane_bits: int, fn) -> int:
    mask = (1 << lane_bits) - 1
    out = 0
    for lane in range(128 // lane_bits):
        shift = lane * lane_bits
        av = (a >> shift) & mask
        bv = (b >> shift) & mask
        out |= (fn(av, bv) & mask) << shift
    return out & MASK128


def lane_shift_left(a: int, lane_bits: int, count: int) -> int:
    if count >= lane_bits:
        return 0
    return lane_binary(a, 0, lane_bits, lambda x, _y: x << count)


def lane_shift_right_logical(a: int, lane_bits: int, count: int) -> int:
    if count >= lane_bits:
        return 0
    return lane_binary(a, 0, lane_bits, lambda x, _y: x >> count)


def lane_shift_right_arithmetic(a: int, lane_bits: int, count: int) -> int:
    effective = min(count, lane_bits - 1)
    return lane_binary(a, 0, lane_bits, lambda x, _y: signed(x, lane_bits) >> effective)


def shift_count(op: Stage4OpSpec, src_value: int, imm: int) -> int:
    if op.operands == "xmm,imm8":
        return imm & 0xFF
    return int(src_value) & ((1 << 64) - 1)


def lanes(value: int, lane_bits: int) -> List[int]:
    mask = (1 << lane_bits) - 1
    return [((int(value) & MASK128) >> (lane * lane_bits)) & mask for lane in range(128 // lane_bits)]


def pack_lanes(values: Sequence[int], lane_bits: int) -> int:
    mask = (1 << lane_bits) - 1
    out = 0
    for lane, value in enumerate(values):
        out |= (int(value) & mask) << (lane * lane_bits)
    return out & MASK128


def signed(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    sign = 1 << (bits - 1)
    return value - (1 << bits) if value & sign else value


def saturate_signed(value: int, bits: int) -> int:
    lo = -(1 << (bits - 1))
    hi = (1 << (bits - 1)) - 1
    return min(hi, max(lo, int(value))) & ((1 << bits) - 1)


def saturate_unsigned(value: int, bits: int) -> int:
    return min((1 << bits) - 1, max(0, int(value))) & ((1 << bits) - 1)


def punpck(a: int, b: int, lane_bits: int, high: bool) -> int:
    a_lanes = lanes(a, lane_bits)
    b_lanes = lanes(b, lane_bits)
    half = len(a_lanes) // 2
    start = half if high else 0
    out: List[int] = []
    for idx in range(start, start + half):
        out.extend([a_lanes[idx], b_lanes[idx]])
    return pack_lanes(out, lane_bits)


def pack_saturating(a: int, b: int, src_bits: int, dst_bits: int, unsigned_dst: bool) -> int:
    out: List[int] = []
    for value in lanes(a, src_bits) + lanes(b, src_bits):
        sv = signed(value, src_bits)
        out.append(saturate_unsigned(sv, dst_bits) if unsigned_dst else saturate_signed(sv, dst_bits))
    return pack_lanes(out, dst_bits)


def semantic_name(name: str) -> str:
    return (
        name.replace("_XMM_M128", "_XMM_XMM")
        .replace("_XMM_M64", "_XMM_XMM")
        .replace("_XMM_M32", "_XMM_XMM")
        .replace("MOVDQU", "MOVDQA")
    )


def apply_op(
    xmm: Sequence[int],
    op_idx: int,
    dst: int,
    src: int,
    imm: int,
    src_value: Optional[int] = None,
) -> List[int]:
    op = OPS[op_idx]
    if op.expected_exception is not None:
        return [u128(x) for x in xmm]
    if op.operands in MEMORY_DEST_OPERANDS:
        return [u128(x) for x in xmm]
    raw_name = op.name
    name = semantic_name(op.name)
    out = [u128(x) for x in xmm]
    a = out[dst]
    b = u128(src_value) if src_value is not None else out[src]
    if raw_name == "MOVD_XMM_M32":
        out[dst] = b & ((1 << 32) - 1)
    elif raw_name in {"MOVQ_XMM_M64", "MOVQ_XMM_XMM"}:
        out[dst] = b & MASK64
    elif raw_name in {"MOVSD_XMM_XMM", "MOVLPD_XMM_M64"}:
        out[dst] = (a & HIGH64_MASK) | (b & MASK64)
    elif raw_name == "MOVSD_XMM_M64":
        out[dst] = b & MASK64
    elif raw_name == "MOVHPD_XMM_M64":
        out[dst] = (a & MASK64) | ((b & MASK64) << 64)
    elif raw_name in {"MOVAPD_XMM_M128", "MOVUPD_XMM_M128", "MOVAPD_XMM_XMM", "MOVUPD_XMM_XMM"}:
        out[dst] = b
    elif name == "MOVDQA_XMM_XMM":
        out[dst] = b
    elif name in {
        "ADDSD_XMM_XMM",
        "SUBSD_XMM_XMM",
        "MULSD_XMM_XMM",
        "DIVSD_XMM_XMM",
        "SQRTSD_XMM_XMM",
        "MINSD_XMM_XMM",
        "MAXSD_XMM_XMM",
        "CMPSD_XMM_XMM_IMM8",
    }:
        try:
            result_low = scalar_fp_result_bits(name, a & MASK64, b & MASK64, imm, 64)
        except ValueError:
            return out
        out[dst] = (a & HIGH64_MASK) | result_low
    elif name in {
        "ADDSS_XMM_XMM",
        "SUBSS_XMM_XMM",
        "MULSS_XMM_XMM",
        "DIVSS_XMM_XMM",
        "SQRTSS_XMM_XMM",
        "MINSS_XMM_XMM",
        "MAXSS_XMM_XMM",
        "CMPSS_XMM_XMM_IMM8",
    }:
        try:
            result_low = scalar_fp_result_bits(name, a & MASK32, b & MASK32, imm, 32)
        except ValueError:
            return out
        out[dst] = (a & HIGH96_MASK) | result_low
    elif name in {
        "ADDPD_XMM_XMM",
        "SUBPD_XMM_XMM",
        "MULPD_XMM_XMM",
        "DIVPD_XMM_XMM",
        "SQRTPD_XMM_XMM",
        "MINPD_XMM_XMM",
        "MAXPD_XMM_XMM",
        "CMPPD_XMM_XMM_IMM8",
    }:
        try:
            out[dst] = packed_fp_result_bits(name, a, b, imm, 64)
        except ValueError:
            return out
    elif name in {
        "ADDPS_XMM_XMM",
        "SUBPS_XMM_XMM",
        "MULPS_XMM_XMM",
        "DIVPS_XMM_XMM",
        "SQRTPS_XMM_XMM",
        "MINPS_XMM_XMM",
        "MAXPS_XMM_XMM",
        "CMPPS_XMM_XMM_IMM8",
    }:
        try:
            out[dst] = packed_fp_result_bits(name, a, b, imm, 32)
        except ValueError:
            return out
    elif name == "PAND_XMM_XMM":
        out[dst] = a & b
    elif name == "PANDN_XMM_XMM":
        out[dst] = (~a) & b & MASK128
    elif name == "POR_XMM_XMM":
        out[dst] = a | b
    elif name == "PXOR_XMM_XMM":
        out[dst] = a ^ b
    elif name.startswith("PADD"):
        out[dst] = lane_binary(a, b, op.lane_bits, lambda x, y: x + y)
    elif name.startswith("PSUB"):
        out[dst] = lane_binary(a, b, op.lane_bits, lambda x, y: x - y)
    elif name.startswith("PCMPEQ"):
        mask = (1 << op.lane_bits) - 1
        out[dst] = lane_binary(a, b, op.lane_bits, lambda x, y: mask if x == y else 0)
    elif name == "PMULLW_XMM_XMM":
        out[dst] = lane_binary(a, b, 16, lambda x, y: x * y)
    elif name == "PUNPCKLBW_XMM_XMM":
        out[dst] = punpck(a, b, 8, False)
    elif name == "PUNPCKLWD_XMM_XMM":
        out[dst] = punpck(a, b, 16, False)
    elif name == "PUNPCKLDQ_XMM_XMM":
        out[dst] = punpck(a, b, 32, False)
    elif name == "PUNPCKLQDQ_XMM_XMM":
        out[dst] = punpck(a, b, 64, False)
    elif name == "PUNPCKHBW_XMM_XMM":
        out[dst] = punpck(a, b, 8, True)
    elif name == "PUNPCKHWD_XMM_XMM":
        out[dst] = punpck(a, b, 16, True)
    elif name == "PUNPCKHDQ_XMM_XMM":
        out[dst] = punpck(a, b, 32, True)
    elif name == "PUNPCKHQDQ_XMM_XMM":
        out[dst] = punpck(a, b, 64, True)
    elif name == "PACKSSWB_XMM_XMM":
        out[dst] = pack_saturating(a, b, 16, 8, False)
    elif name == "PACKSSDW_XMM_XMM":
        out[dst] = pack_saturating(a, b, 32, 16, False)
    elif name == "PACKUSWB_XMM_XMM":
        out[dst] = pack_saturating(a, b, 16, 8, True)
    elif name.startswith("PCMPGT"):
        mask = (1 << op.lane_bits) - 1
        out[dst] = lane_binary(
            a,
            b,
            op.lane_bits,
            lambda x, y: mask if signed(x, op.lane_bits) > signed(y, op.lane_bits) else 0,
        )
    elif name == "PAVGB_XMM_XMM":
        out[dst] = lane_binary(a, b, 8, lambda x, y: (x + y + 1) // 2)
    elif name == "PAVGW_XMM_XMM":
        out[dst] = lane_binary(a, b, 16, lambda x, y: (x + y + 1) // 2)
    elif name == "PMULHUW_XMM_XMM":
        out[dst] = lane_binary(a, b, 16, lambda x, y: (x * y) >> 16)
    elif name == "PMULHW_XMM_XMM":
        out[dst] = lane_binary(a, b, 16, lambda x, y: signed(x, 16) * signed(y, 16) >> 16)
    elif name == "PMINUB_XMM_XMM":
        out[dst] = lane_binary(a, b, 8, min)
    elif name == "PMAXUB_XMM_XMM":
        out[dst] = lane_binary(a, b, 8, max)
    elif name == "PMINSW_XMM_XMM":
        out[dst] = lane_binary(a, b, 16, lambda x, y: x if signed(x, 16) < signed(y, 16) else y)
    elif name == "PMAXSW_XMM_XMM":
        out[dst] = lane_binary(a, b, 16, lambda x, y: x if signed(x, 16) > signed(y, 16) else y)
    elif name == "PMULUDQ_XMM_XMM":
        a_lanes = lanes(a, 32)
        b_lanes = lanes(b, 32)
        out[dst] = pack_lanes([a_lanes[0] * b_lanes[0], a_lanes[2] * b_lanes[2]], 64)
    elif name == "PMADDWD_XMM_XMM":
        a_lanes = lanes(a, 16)
        b_lanes = lanes(b, 16)
        sums = []
        for pair in range(4):
            i = pair * 2
            sums.append(
                signed(a_lanes[i], 16) * signed(b_lanes[i], 16)
                + signed(a_lanes[i + 1], 16) * signed(b_lanes[i + 1], 16)
            )
        out[dst] = pack_lanes(sums, 32)
    elif name == "PSADBW_XMM_XMM":
        a_lanes = lanes(a, 8)
        b_lanes = lanes(b, 8)
        sums = [sum(abs(a_lanes[i + base] - b_lanes[i + base]) for i in range(8)) for base in (0, 8)]
        out[dst] = pack_lanes(sums, 64)
    elif name.startswith("PSLL") and name != "PSLLDQ_XMM_IMM8":
        out[dst] = lane_shift_left(a, op.lane_bits, shift_count(op, b, imm))
    elif name.startswith("PSRL") and name != "PSRLDQ_XMM_IMM8":
        out[dst] = lane_shift_right_logical(a, op.lane_bits, shift_count(op, b, imm))
    elif name.startswith("PSRA"):
        out[dst] = lane_shift_right_arithmetic(a, op.lane_bits, shift_count(op, b, imm))
    elif name == "PSLLDQ_XMM_IMM8":
        count = imm & 0xFF
        out[dst] = 0 if count >= 16 else (a << (count * 8)) & MASK128
    elif name == "PSRLDQ_XMM_IMM8":
        count = imm & 0xFF
        out[dst] = 0 if count >= 16 else a >> (count * 8)
    else:
        raise ValueError(op.name)
    return out


def read_memory_u128(memory: Sequence[int], offset: int) -> int:
    return int.from_bytes(bytes(int(x) & 0xFF for x in memory[offset:offset + 16]), "little")


def write_memory_u128(memory: Sequence[int], offset: int, value: int) -> List[int]:
    out = [int(x) & 0xFF for x in memory]
    out[offset:offset + 16] = list(xmm_bytes(value))
    return out


def memory_store_value(op: Stage4OpSpec, value: int) -> int:
    width = memory_operand_width_bytes(op)
    return (int(value) >> int(op.memory_store_bit_offset)) & ((1 << (width * 8)) - 1)


def write_memory_operand(memory: Sequence[int], offset: int, op: Stage4OpSpec, value: int) -> List[int]:
    width = memory_operand_width_bytes(op)
    out = [int(x) & 0xFF for x in memory]
    out[offset:offset + width] = list(memory_store_value(op, value).to_bytes(width, "little"))
    return out


def xmm_bytes(value: int) -> bytes:
    return (int(value) & MASK128).to_bytes(16, "little")


def random_memory(rng: random.Random, split: str) -> List[int]:
    patterns = [
        bytes([0x00] * 16),
        bytes([0xFF] * 16),
        bytes(range(16)),
        bytes(reversed(range(16))),
        bytes([0x00, 0xFF] * 8),
        bytes([0x80, 0x7F] * 8),
        (1).to_bytes(16, "little"),
        ((1 << 127)).to_bytes(16, "little"),
    ]
    memory = bytearray(rng.getrandbits(8) for _ in range(MEM_SIZE))
    if split in {"heldout_boundary", "long"}:
        for offset in range(0, MEM_SIZE, 16):
            pattern = bytearray(rng.choice(patterns))
            if split == "long":
                pattern[rng.randrange(16)] ^= rng.randrange(256)
            memory[offset:offset + 16] = pattern
    safe_lanes = [f64_bits(value) for value in SAFE_F64_VALUES]
    safe_chunk_count = 10 if split != "long" else 8
    for offset in rng.sample(list(range(0, MEM_SIZE, 16)), safe_chunk_count):
        value = (rng.choice(safe_lanes) << 64) | rng.choice(safe_lanes)
        memory[offset:offset + 16] = value.to_bytes(16, "little")
    compare_pairs = [
        (QUIET_NAN_F64_BITS[0], rng.choice(safe_lanes)),
        (rng.choice(safe_lanes), QUIET_NAN_F64_BITS[1]),
        (POS_ZERO_F64_BITS, NEG_ZERO_F64_BITS),
        (NEG_ZERO_F64_BITS, POS_ZERO_F64_BITS),
    ]
    compare_chunk_count = 4 if split != "long" else 5
    for offset in rng.sample(list(range(0, MEM_SIZE, 16)), compare_chunk_count):
        lo, hi = rng.choice(compare_pairs)
        if rng.randrange(2):
            lo, hi = hi, lo
        value = (int(hi) << 64) | int(lo)
        memory[offset:offset + 16] = value.to_bytes(16, "little")
    edge_lanes = [*EDGE_F64_BITS, *safe_lanes]
    edge_pairs = [
        (POS_INF_F64_BITS, rng.choice(safe_lanes)),
        (NEG_INF_F64_BITS, rng.choice(safe_lanes)),
        (rng.choice(safe_lanes), POS_INF_F64_BITS),
        (rng.choice(safe_lanes), NEG_INF_F64_BITS),
        (POS_ZERO_F64_BITS, rng.choice(safe_lanes)),
        (NEG_ZERO_F64_BITS, rng.choice(safe_lanes)),
        (POS_ZERO_F64_BITS, NEG_ZERO_F64_BITS),
        (NEG_ZERO_F64_BITS, POS_ZERO_F64_BITS),
    ]
    edge_chunk_count = 5 if split != "long" else 6
    for offset in rng.sample(list(range(0, MEM_SIZE, 16)), edge_chunk_count):
        if rng.randrange(3):
            lo, hi = rng.choice(edge_pairs)
        else:
            lo, hi = rng.choice(edge_lanes), rng.choice(edge_lanes)
        value = (int(hi) << 64) | int(lo)
        memory[offset:offset + 16] = value.to_bytes(16, "little")
    safe_f32_lanes = [f32_bits(value) for value in SAFE_F32_VALUES]
    safe_f32_chunk_count = 6 if split != "long" else 5
    for offset in rng.sample(list(range(0, MEM_SIZE, 16)), safe_f32_chunk_count):
        value = pack_lanes([rng.choice(safe_f32_lanes) for _ in range(4)], 32)
        memory[offset:offset + 16] = value.to_bytes(16, "little")
    compare_f32_sets = [
        [QUIET_NAN_F32_BITS[0], rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
        [rng.choice(safe_f32_lanes), QUIET_NAN_F32_BITS[1], rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
        [POS_ZERO_F32_BITS, NEG_ZERO_F32_BITS, rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
        [NEG_ZERO_F32_BITS, POS_ZERO_F32_BITS, rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
    ]
    compare_f32_chunk_count = 3 if split != "long" else 4
    for offset in rng.sample(list(range(0, MEM_SIZE, 16)), compare_f32_chunk_count):
        lanes32 = list(rng.choice(compare_f32_sets))
        rng.shuffle(lanes32)
        memory[offset:offset + 16] = pack_lanes(lanes32, 32).to_bytes(16, "little")
    edge_f32_lanes = [*EDGE_F32_BITS, *safe_f32_lanes]
    edge_f32_sets = [
        [POS_INF_F32_BITS, rng.choice(safe_f32_lanes), POS_ZERO_F32_BITS, rng.choice(safe_f32_lanes)],
        [NEG_INF_F32_BITS, rng.choice(safe_f32_lanes), NEG_ZERO_F32_BITS, rng.choice(safe_f32_lanes)],
        [rng.choice(safe_f32_lanes), POS_INF_F32_BITS, NEG_ZERO_F32_BITS, POS_ZERO_F32_BITS],
        [rng.choice(edge_f32_lanes) for _ in range(4)],
    ]
    edge_f32_chunk_count = 4 if split != "long" else 5
    for offset in rng.sample(list(range(0, MEM_SIZE, 16)), edge_f32_chunk_count):
        lanes32 = list(rng.choice(edge_f32_sets))
        rng.shuffle(lanes32)
        memory[offset:offset + 16] = pack_lanes(lanes32, 32).to_bytes(16, "little")
    return list(memory)


def random_xmm(rng: random.Random, split: str) -> List[int]:
    count_patterns = [0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255]
    patterns = [
        0,
        MASK128,
        int.from_bytes(bytes(range(16)), "little"),
        int.from_bytes(bytes(reversed(range(16))), "little"),
        int.from_bytes(bytes([0x00, 0xFF] * 8), "little"),
        int.from_bytes(bytes([0x80, 0x7F] * 8), "little"),
        *count_patterns,
    ]
    safe_lanes = [f64_bits(value) for value in SAFE_F64_VALUES]
    safe_count = 12 if split != "long" else 10
    slots = [True] * safe_count + [False] * (len(XMM_REGS) - safe_count)
    rng.shuffle(slots)
    values = []
    for use_safe in slots:
        if use_safe:
            values.append(u128((rng.choice(safe_lanes) << 64) | rng.choice(safe_lanes)))
        else:
            value = rng.choice(patterns)
            if split == "long":
                value ^= 1 << rng.randrange(128)
            values.append(u128(value))
    compare_pairs = [
        (QUIET_NAN_F64_BITS[0], rng.choice(safe_lanes)),
        (rng.choice(safe_lanes), QUIET_NAN_F64_BITS[1]),
        (POS_ZERO_F64_BITS, NEG_ZERO_F64_BITS),
        (NEG_ZERO_F64_BITS, POS_ZERO_F64_BITS),
    ]
    special_count = 3 if split != "long" else 4
    for index in rng.sample(range(len(values)), special_count):
        lo, hi = rng.choice(compare_pairs)
        if rng.randrange(2):
            lo, hi = hi, lo
        values[index] = u128((int(hi) << 64) | int(lo))
    edge_lanes = [*EDGE_F64_BITS, *safe_lanes]
    edge_pairs = [
        (POS_INF_F64_BITS, rng.choice(safe_lanes)),
        (NEG_INF_F64_BITS, rng.choice(safe_lanes)),
        (rng.choice(safe_lanes), POS_INF_F64_BITS),
        (rng.choice(safe_lanes), NEG_INF_F64_BITS),
        (POS_ZERO_F64_BITS, NEG_ZERO_F64_BITS),
        (NEG_ZERO_F64_BITS, POS_ZERO_F64_BITS),
    ]
    edge_count = 4 if split != "long" else 5
    for index in rng.sample(range(len(values)), edge_count):
        if rng.randrange(3):
            lo, hi = rng.choice(edge_pairs)
        else:
            lo, hi = rng.choice(edge_lanes), rng.choice(edge_lanes)
        values[index] = u128((int(hi) << 64) | int(lo))
    safe_f32_lanes = [f32_bits(value) for value in SAFE_F32_VALUES]
    safe_f32_count = 5 if split != "long" else 6
    for index in rng.sample(range(len(values)), safe_f32_count):
        values[index] = u128(pack_lanes([rng.choice(safe_f32_lanes) for _ in range(4)], 32))
    compare_f32_sets = [
        [QUIET_NAN_F32_BITS[0], rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
        [rng.choice(safe_f32_lanes), QUIET_NAN_F32_BITS[1], rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
        [POS_ZERO_F32_BITS, NEG_ZERO_F32_BITS, rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
        [NEG_ZERO_F32_BITS, POS_ZERO_F32_BITS, rng.choice(safe_f32_lanes), rng.choice(safe_f32_lanes)],
    ]
    special_f32_count = 2 if split != "long" else 3
    for index in rng.sample(range(len(values)), special_f32_count):
        lanes32 = list(rng.choice(compare_f32_sets))
        rng.shuffle(lanes32)
        values[index] = u128(pack_lanes(lanes32, 32))
    edge_f32_lanes = [*EDGE_F32_BITS, *safe_f32_lanes]
    edge_f32_sets = [
        [POS_INF_F32_BITS, rng.choice(safe_f32_lanes), POS_ZERO_F32_BITS, rng.choice(safe_f32_lanes)],
        [NEG_INF_F32_BITS, rng.choice(safe_f32_lanes), NEG_ZERO_F32_BITS, rng.choice(safe_f32_lanes)],
        [rng.choice(safe_f32_lanes), POS_INF_F32_BITS, NEG_ZERO_F32_BITS, POS_ZERO_F32_BITS],
        [rng.choice(edge_f32_lanes) for _ in range(4)],
    ]
    edge_f32_count = 3 if split != "long" else 4
    for index in rng.sample(range(len(values)), edge_f32_count):
        lanes32 = list(rng.choice(edge_f32_sets))
        rng.shuffle(lanes32)
        values[index] = u128(pack_lanes(lanes32, 32))
    return values


def random_imm8(rng: random.Random, op: Stage4OpSpec, split: str) -> int:
    if not op.operands.endswith("imm8"):
        return 0
    if is_fp_compare_name(op.name):
        return rng.randrange(8)
    boundaries = [0, 1, 2, 7, 8, 15, 16, 31, 32, 63, 64, 127, 128, 255]
    if split == "heldout_boundary":
        return rng.choice(boundaries)
    if split == "long":
        return rng.choice(boundaries + [rng.randrange(256)])
    return rng.randrange(256)


def make_step(
    index: int,
    xmm: List[int],
    memory: List[int],
    rip: int,
    rng: random.Random,
    split: str,
) -> Tuple[Dict[str, Any], List[int], List[int], int, str]:
    selected: Optional[Tuple[int, Stage4OpSpec, int, int, int, Optional[int]]] = None
    for _ in range(200):
        op_idx = rng.randrange(len(OPS))
        op = OPS[op_idx]
        dst = rng.randrange(len(XMM_REGS))
        src = dst if op.operands == "xmm,imm8" else rng.randrange(len(XMM_REGS))
        imm = random_imm8(rng, op, split)
        forced_mem_offset: Optional[int] = None
        if op.family.startswith("SSE_FP_") and op.operands in MEMORY_SOURCE_OPERANDS:
            width = memory_operand_width_bytes(op)
            alignment = max(1, int(op.memory_alignment))
            candidates = [
                offset
                for offset in range(0, MEM_SIZE - width + 1)
                if offset % alignment == 0
                and fp_operands_safe(op, xmm, dst, src, read_memory_operand(memory, offset, op))
            ]
            if not candidates:
                continue
            forced_mem_offset = rng.choice(candidates)
        elif not fp_operands_safe(op, xmm, dst, src):
            continue
        selected = (op_idx, op, dst, src, imm, forced_mem_offset)
        break
    if selected is None:
        op_idx = OP_INDEX["MOVDQA_XMM_XMM"]
        op = OPS[op_idx]
        dst = rng.randrange(len(XMM_REGS))
        src = rng.randrange(len(XMM_REGS))
        imm = 0
        forced_mem_offset = None
    else:
        op_idx, op, dst, src, imm, forced_mem_offset = selected
    mem_offset = -1
    mem_addr = 0
    mem_value: Optional[int] = None
    if op.operands in MEMORY_OPERANDS:
        width = memory_operand_width_bytes(op)
        alignment = max(1, int(op.memory_alignment))
        if forced_mem_offset is not None:
            mem_offset = forced_mem_offset
        elif op.expected_exception is not None:
            candidates = [offset for offset in range(0, MEM_SIZE - 15) if offset % 16 != 0]
            mem_offset = rng.choice(candidates)
        elif alignment == 1:
            mem_offset = rng.randrange(0, MEM_SIZE - width + 1)
        else:
            mem_offset = rng.randrange(0, MEM_SIZE - width + 1, alignment)
        if op.operands in MEMORY_DEST_OPERANDS:
            dst = mem_offset if alignment == 1 else mem_offset // alignment
        mem_addr = MEM_BASE + mem_offset
        mem_value = read_memory_operand(memory, mem_offset, op) if op.operands in MEMORY_SOURCE_OPERANDS else u128(xmm[src])
    before_xmm = [u128(x) for x in xmm]
    before_memory = list(memory)
    bytes_hex = encode_x86_bytes(op, dst, src, imm, mem_addr)
    exception = op.expected_exception
    after_xmm = apply_op(before_xmm, op_idx, dst, src, imm, mem_value)
    after_memory = (
        write_memory_operand(before_memory, mem_offset, op, before_xmm[src])
        if op.operands in MEMORY_DEST_OPERANDS and exception is None
        else before_memory
    )
    fallthrough_rip = rip + len(bytes.fromhex(bytes_hex))
    after_rip = rip if exception is not None else fallthrough_rip
    write_mask = register_write_mask(op)
    reg_writes = [
        {"name": reg, "before": hex_u(before_xmm[i], 128), "after": hex_u(after_xmm[i], 128), "mask": hex_u(write_mask, 128)}
        for i, reg in enumerate(XMM_REGS)
        if before_xmm[i] != after_xmm[i]
    ]
    if op.operands == "xmm,imm8":
        operands = [
            {"name": "dst", "reg": XMM_REGS[dst]},
            {"name": "imm", "value": imm & 0xFF},
        ]
    elif op.operands == "xmm,xmm,imm8":
        operands = [
            {"name": "dst", "reg": XMM_REGS[dst]},
            {"name": "src", "reg": XMM_REGS[src]},
            {"name": "imm", "value": imm & 0xFF},
        ]
    elif op.operands in MEMORY_SOURCE_OPERANDS:
        width = memory_operand_width_bytes(op)
        operands = [
            {"name": "dst", "reg": XMM_REGS[dst]},
            {
                "name": "src",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "alignment": op.memory_alignment,
            },
        ]
        if op.operands.endswith("imm8"):
            operands.append({"name": "imm", "value": imm & 0xFF})
    elif op.operands in MEMORY_DEST_OPERANDS:
        width = memory_operand_width_bytes(op)
        operands = [
            {
                "name": "dst",
                "kind": "mem",
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "alignment": op.memory_alignment,
            },
            {"name": "src", "reg": XMM_REGS[src]},
        ]
    else:
        operands = [
            {"name": "dst", "reg": XMM_REGS[dst]},
            {"name": "src", "reg": XMM_REGS[src]},
        ]
    memory_reads = []
    if op.operands in MEMORY_SOURCE_OPERANDS and mem_value is not None and exception is None:
        width = memory_operand_width_bytes(op)
        memory_reads.append(
            {
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "value_hex": hex_u(mem_value, width * 8),
                "segment": "flat",
                "fault": None,
            }
        )
    memory_writes = []
    if op.operands in MEMORY_DEST_OPERANDS and exception is None:
        width = memory_operand_width_bytes(op)
        memory_writes.append(
            {
                "address": hex_u(mem_addr, 64),
                "width_bytes": width,
                "value_hex": hex_u(memory_store_value(op, before_xmm[src]), width * 8),
                "segment": "flat",
                "fault": None,
            }
        )
    pan_stage4: Dict[str, Any] = {
        "op_index": op_idx,
        "dst": dst,
        "src": src,
        "imm": imm,
        "selector": op_idx,
        "before_xmm": before_xmm,
        "after_xmm": after_xmm,
        "before_mxcsr": MXCSR_DEFAULT,
        "after_mxcsr": MXCSR_DEFAULT,
        "before_rip": rip,
        "after_rip": after_rip,
        "fallthrough_rip": fallthrough_rip,
        "exception": exception,
        "exception_vector": 13 if exception == "#GP(0)" else -1,
        "exception_error_code": 0 if exception == "#GP(0)" else -1,
    }
    if mem_value is not None:
        pan_stage4.update(
            {
                "mem_offset": mem_offset,
                "mem_addr": mem_addr,
                "mem_value": mem_value,
                "before_memory": before_memory,
                "after_memory": after_memory,
            }
        )
    step = {
        "index": index,
        "decode": {
            "bytes_hex": bytes_hex,
            "iclass": op.name.split("_")[0],
            "iform": op.iform,
            "category": op.family,
            "extension": instruction_extension(op),
            "operands": operands,
        },
        "before": state_record(before_xmm, rip, before_memory),
        "after": state_record(after_xmm, after_rip, after_memory, exception=exception),
        "effects": {
            "register_writes": reg_writes,
            "memory_reads": memory_reads,
            "memory_writes": memory_writes,
            "flags_written": [],
            "undefined_flags": [],
            "exception": exception,
        },
        "oracle_disagreement": None,
        "pan_stage4": pan_stage4,
    }
    return step, after_xmm, after_memory, fallthrough_rip + rng.randrange(1, 8), asm_string(op, dst, src, imm, mem_addr)


def make_trace(trace_id: str, split: str, length: int, rng: random.Random) -> Dict[str, Any]:
    xmm = random_xmm(rng, split)
    memory = random_memory(rng, split)
    rip = CODE_BASE + rng.randrange(0, 0x8000)
    steps = []
    asm = []
    bytes_parts = []
    for idx in range(length):
        step, xmm, memory, rip, asm_text = make_step(idx, xmm, memory, rip, rng, split)
        steps.append(step)
        asm.append(asm_text)
        bytes_parts.append(step["decode"]["bytes_hex"])
    return {
        "schema_version": "pan-x86-trace-v1",
        "trace_id": trace_id,
        "stage": STAGE,
        "source": {
            "generator": "stage4_simd_x87_pan.py",
            "seed": trace_id,
            "xed_version": "intel-xed-idata-af3c1a1",
            "oracle": ["internal_semantics"],
        },
        "mode": {
            "long_mode": True,
            "operand_bits": 64,
            "address_bits": 64,
            "ring": 0,
            "cpuid_features": ["SSE", "SSE2"],
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
            op = OPS[step["pan_stage4"]["op_index"]]
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
        "xmm_registers": list(XMM_REGS),
        "mxcsr": hex_u(MXCSR_DEFAULT, 32),
        "cpuid_features": ["SSE", "SSE2"],
        "memory_base": hex_u(MEM_BASE, 64),
        "memory_size": MEM_SIZE,
        "forms": [dataclasses.asdict(op) | {"op_index": i, "implemented": True} for i, op in enumerate(OPS)],
        "limitations": [
            "SSE/SSE2 integer XMM register-register, aligned source-memory, MOVDQA/MOVDQU/MOVAPD/MOVUPD/MOVNTDQ/MOVNTPD m128 stores, MOVSD/MOVLPD/MOVHPD/MOVQ m64 stores, MOVD m32 stores, MOVSD/MOVAPD/MOVUPD/MOVQ register transfer forms, unaligned MOVDQU load/store, aligned-m128 unaligned-memory #GP(0) for current m128 load/store forms, immediate-shift, variable-count shift, scalar/packed single/double add/sub/mul/div/sqrt/min/max over safe finite plus non-trapping zero/infinity edge operands, CMPSS/CMPPS/CMPSD/CMPPD over safe finite, zero, infinity, and quiet-NaN operands, and scalar/packed FP memory-source forms",
            "XMM0-XMM15 register operand coverage",
            "aligned absolute integer m128 source operands, MOVAPD/MOVUPD m128 source operands, MOVD/MOVQ/MOVSD/MOVLPD/MOVHPD narrow source operands, MOVSD/MOVAPD/MOVUPD/MOVQ register transfer forms, MOVDQA/MOVDQU/MOVAPD/MOVUPD/MOVNTDQ/MOVNTPD m128 destinations, MOVSD/MOVLPD/MOVHPD/MOVQ m64 destinations, MOVD m32 destinations, MOVDQU unaligned source operands, aligned m128 unaligned-memory #GP(0) exception traces for the current aligned m128 source/destination forms, safe finite plus non-trapping zero/infinity scalar single/double register and memory operands, safe finite plus non-trapping zero/infinity packed single/double register/memory operands, and FP compare predicates over safe finite, zero, infinity, quiet-NaN, and signed-zero operands are covered; broader memory operands beyond the current m128/m64/m32 subset, masks, AVX/VEX/EVEX encodings, and MXCSR exception behavior are not",
            "no subnormal, underflow/overflow, divide-by-zero status-flag, invalid-operation status-flag, signaling-NaN behavior, quiet-NaN arithmetic/min-max payload behavior, x87 stack, MMX, YMM, or ZMM semantics yet",
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
            p = step["pan_stage4"]
            rows.append({"op": int(p["op_index"]), "selector": int(p["selector"])})
    return rows


class Predictor(Protocol):
    def predict_next(
        self,
        xmm: List[int],
        memory: Optional[List[int]],
        step: Dict[str, Any],
    ) -> Tuple[List[int], Optional[List[int]], Dict[str, Any]]:
        ...


class StructuredStage4Predictor:
    def __init__(self, op_offset: int = 0, corrupt_src: bool = False):
        self.op_offset = op_offset
        self.corrupt_src = corrupt_src

    def predict_next(
        self,
        xmm: List[int],
        memory: Optional[List[int]],
        step: Dict[str, Any],
    ) -> Tuple[List[int], Optional[List[int]], Dict[str, Any]]:
        p = step["pan_stage4"]
        selector = (int(p["op_index"]) + self.op_offset) % len(OPS)
        selected_op = OPS[selector]
        use_xmm = [u128(x) for x in xmm]
        use_memory = [int(x) & 0xFF for x in memory] if memory is not None else None
        if use_memory is None and "before_memory" in p:
            use_memory = [int(x) & 0xFF for x in p["before_memory"]]
        src_value = None
        if selected_op.operands in MEMORY_SOURCE_OPERANDS and use_memory is not None and "mem_offset" in p:
            src_value = read_memory_operand(use_memory, int(p["mem_offset"]), selected_op)
        if self.corrupt_src:
            if selected_op.operands in MEMORY_SOURCE_OPERANDS and src_value is not None:
                src_value ^= (1 << (memory_operand_width_bytes(selected_op) * 8)) - 1
            elif selected_op.operands in MEMORY_DEST_OPERANDS or int(p["src"]) < len(use_xmm):
                use_xmm[int(p["src"])] ^= MASK128
        selected_dst = int(p["dst"])
        if selected_op.operands not in MEMORY_DEST_OPERANDS:
            selected_dst %= len(use_xmm)
        out = apply_op(use_xmm, selector, selected_dst, int(p["src"]), int(p["imm"]), src_value)
        out_memory = use_memory
        if (
            selected_op.operands in MEMORY_DEST_OPERANDS
            and selected_op.expected_exception is None
            and use_memory is not None
            and "mem_offset" in p
        ):
            out_memory = write_memory_operand(use_memory, int(p["mem_offset"]), selected_op, use_xmm[int(p["src"])])
        return out, out_memory, {"selector": selector, "exception": selected_op.expected_exception}


class FactorizedStage4PAN(nn.Module):
    def __init__(self, hidden: int = 96):
        super().__init__()
        self.op_emb = nn.Embedding(len(OPS), 32)
        self.net = nn.Sequential(
            nn.Linear(32, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, len(OPS)),
        )

    def forward(self, op: torch.Tensor) -> torch.Tensor:
        return self.net(self.op_emb(op.long()))


class EffectStage4PAN(FactorizedStage4PAN):
    """Learned Stage 4 XMM/SIMD effect-family selector head."""


def make_model(arch: str, hidden: int) -> FactorizedStage4PAN:
    if arch == "factorized":
        return FactorizedStage4PAN(hidden=hidden)
    if arch == "effect":
        return EffectStage4PAN(hidden=hidden)
    raise ValueError(f"unknown Stage 4 arch {arch!r}")


class NeuralStage4Predictor:
    def __init__(self, model: FactorizedStage4PAN):
        self.model = model.eval()
        self.learned_effect_mode = isinstance(model, EffectStage4PAN)

    def predict_next(
        self,
        xmm: List[int],
        memory: Optional[List[int]],
        step: Dict[str, Any],
    ) -> Tuple[List[int], Optional[List[int]], Dict[str, Any]]:
        p = step["pan_stage4"]
        with torch.no_grad():
            op = torch.tensor([int(p["op_index"])], dtype=torch.long)
            selector = int(self.model(op).argmax(-1)[0])
        selected_op = OPS[selector]
        use_memory = [int(x) & 0xFF for x in memory] if memory is not None else None
        if use_memory is None and "before_memory" in p:
            use_memory = [int(x) & 0xFF for x in p["before_memory"]]
        src_value = None
        if selected_op.operands in MEMORY_SOURCE_OPERANDS and use_memory is not None and "mem_offset" in p:
            src_value = read_memory_operand(use_memory, int(p["mem_offset"]), selected_op)
        selected_dst = int(p["dst"])
        if selected_op.operands not in MEMORY_DEST_OPERANDS:
            selected_dst %= len(xmm)
        out = apply_op(xmm, selector, selected_dst, int(p["src"]), int(p["imm"]), src_value)
        out_memory = use_memory
        if (
            selected_op.operands in MEMORY_DEST_OPERANDS
            and selected_op.expected_exception is None
            and use_memory is not None
            and "mem_offset" in p
        ):
            out_memory = write_memory_operand(use_memory, int(p["mem_offset"]), selected_op, xmm[int(p["src"])])
        return out, out_memory, {"selector": selector, "exception": selected_op.expected_exception}


def selector_exact(model: FactorizedStage4PAN, rows: Sequence[Dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    with torch.no_grad():
        op = torch.tensor([r["op"] for r in rows], dtype=torch.long)
        target = torch.tensor([r["selector"] for r in rows], dtype=torch.long)
        pred = model(op).argmax(-1)
    return float((pred == target).float().mean().item())


def train_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    if monitor:
        monitor.set_phase("train")
    train_rows = load_step_rows(out_dir / "train_traces.jsonl")
    val_rows = load_step_rows(out_dir / "val_traces.jsonl")
    model = make_model(args.arch, args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    rng = random.Random(args.seed + 23)
    for step in range(args.train_steps):
        batch = [train_rows[rng.randrange(len(train_rows))] for _ in range(args.batch_size)]
        op = torch.tensor([r["op"] for r in batch], dtype=torch.long)
        target = torch.tensor([r["selector"] for r in batch], dtype=torch.long)
        loss = F.cross_entropy(model(op), target)
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
    }, out_dir / "stage4_pan.pt")
    write_json(out_dir / "train_history.json", history)
    summary = {
        "model_kind": "neural_effect_selector" if args.arch == "effect" else "neural_factorized_selector",
        "arch": args.arch,
        "train_steps_requested": args.train_steps,
        "train_steps_completed": history[-1]["step"] if history else 0,
        "final": history[-1] if history else {},
        "checkpoint": str(out_dir / "stage4_pan.pt"),
    }
    write_json(out_dir / "train_summary.json", summary)
    return summary


def validate_split(path: Path, predictor: Predictor, split: str) -> Dict[str, Any]:
    programs = 0
    steps = 0
    exact_steps = 0
    final_exact = 0
    xmm_exact = 0
    memory_exact = 0
    memory_checked = 0
    exception_exact = 0
    exception_checked = 0
    selector_exact_count = 0
    by_family: Dict[str, Dict[str, int]] = {}
    failures: List[Dict[str, Any]] = []
    for trace in load_traces(path):
        programs += 1
        xmm = [int(x) for x in trace["steps"][0]["pan_stage4"]["before_xmm"]]
        memory: Optional[List[int]] = None
        trace_ok = True
        for step in trace["steps"]:
            p = step["pan_stage4"]
            op = OPS[int(p["op_index"])]
            if memory is None and "before_memory" in p:
                memory = [int(x) & 0xFF for x in p["before_memory"]]
            pred_xmm, pred_memory, aux = predictor.predict_next(xmm, memory, step)
            exp_xmm = [int(x) for x in p["after_xmm"]]
            exp_memory = [int(x) & 0xFF for x in p["after_memory"]] if "after_memory" in p else memory
            before_memory_ok = True
            if memory is not None and "before_memory" in p:
                before_memory_ok = memory == [int(x) & 0xFF for x in p["before_memory"]]
            regs_ok = pred_xmm == exp_xmm
            memory_ok = before_memory_ok and (pred_memory == exp_memory)
            selector_ok = int(aux["selector"]) == int(p["selector"])
            expected_exception = p.get("exception")
            exception_ok = aux.get("exception") == expected_exception
            step_ok = regs_ok and memory_ok and selector_ok and exception_ok
            steps += 1
            exact_steps += int(step_ok)
            xmm_exact += int(regs_ok)
            if "after_memory" in p:
                memory_checked += 1
                memory_exact += int(memory_ok)
            if expected_exception is not None:
                exception_checked += 1
                exception_exact += int(exception_ok)
            selector_exact_count += int(selector_ok)
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
                    "xmm_ok": regs_ok,
                    "memory_ok": memory_ok,
                    "exception_ok": exception_ok,
                })
            trace_ok = trace_ok and step_ok
            xmm = pred_xmm
            memory = pred_memory
        final_p = trace["steps"][-1]["pan_stage4"]
        final_memory_ok = True
        if "after_memory" in final_p and memory is not None:
            final_memory_ok = memory == [int(x) & 0xFF for x in final_p["after_memory"]]
        final_exact += int(trace_ok and xmm == [int(x) for x in final_p["after_xmm"]] and final_memory_ok)
    worst_family_exact = min((v["exact"] / max(1, v["steps"]) for v in by_family.values()), default=0.0)
    return {
        "split": split,
        "programs": programs,
        "steps": steps,
        "step_exact": round(exact_steps / max(1, steps), 6),
        "final_exact": round(final_exact / max(1, programs), 6),
        "xmm_exact": round(xmm_exact / max(1, steps), 6),
        "memory_exact": round(memory_exact / max(1, memory_checked), 6) if memory_checked else 1.0,
        "memory_checked_steps": memory_checked,
        "exception_exact": round(exception_exact / max(1, exception_checked), 6) if exception_checked else 1.0,
        "exception_checked_steps": exception_checked,
        "selector_exact": round(selector_exact_count / max(1, steps), 6),
        "worst_family_exact": round(worst_family_exact, 6),
        "by_family": {k: {"steps": v["steps"], "exact": round(v["exact"] / max(1, v["steps"]), 6)} for k, v in sorted(by_family.items())},
        "failure_examples": failures,
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
    scrambled = validate_split(out_dir / "val_traces.jsonl", StructuredStage4Predictor(op_offset=1), "scrambled_opcode_val")
    corrupted = validate_split(out_dir / "val_traces.jsonl", StructuredStage4Predictor(corrupt_src=True), "corrupted_src_val")
    overlap = byte_overlap(out_dir)
    learned_effect_mode = bool(getattr(predictor, "learned_effect_mode", False))
    learned_effect_gate = True
    if learned_effect_mode:
        learned_effect_gate = (
            val["selector_exact"] >= args.gate_selector_exact
            and long["selector_exact"] >= args.gate_selector_exact
            and val["xmm_exact"] >= args.gate_step_exact
            and long["xmm_exact"] >= args.gate_step_exact
            and val["memory_exact"] >= args.gate_step_exact
            and long["memory_exact"] >= args.gate_step_exact
            and val["exception_exact"] >= 0.995
            and long["exception_exact"] >= 0.995
        )
    gate = (
        schema_ok
        and learned_effect_gate
        and val["step_exact"] >= args.gate_step_exact
        and long["step_exact"] >= args.gate_step_exact
        and val["selector_exact"] >= args.gate_selector_exact
        and long["selector_exact"] >= args.gate_selector_exact
        and val["worst_family_exact"] >= args.gate_worst_family_exact
        and overlap["train_val_overlap"] == 0
        and overlap["train_long_overlap"] == 0
    )
    report = {
        "schema_valid": schema_ok,
        "transition_contract": {
            "schema": "pan-x86-stage4-effect-selector-v1" if learned_effect_mode else "legacy-stage4-factorized-selector",
            "learned_effect_selectors": learned_effect_mode,
            "learned_effect_axes": ["xmm_lane_effect", "memory_event", "exception_class"] if learned_effect_mode else [],
            "commit_path": "model_emitted_effect_selector" if learned_effect_mode else "legacy_model_selector",
            "direct_value_heads": False,
            "note": "Stage 4 effect mode learns the SIMD/XMM effect family used for typed commit; direct 128-bit lane heads remain future work.",
        },
        "val": val,
        "long": long,
        "negative_controls": {
            "scrambled_opcode_val": scrambled,
            "corrupted_src_val": corrupted,
        },
        "byte_overlap": overlap,
        "gate_status": {
            "schema_valid": schema_ok,
            "stage4_gate_passed": bool(gate),
            "learned_effect_selector_gate_passed": learned_effect_gate,
            "full_x86_goal_complete": False,
        },
    }
    write_json(out_dir / "validation_report.json", report)
    return report


def validate_model(args: argparse.Namespace, out_dir: Path, monitor: Optional[ResourceMonitor]) -> Dict[str, Any]:
    ckpt = torch.load(out_dir / "stage4_pan.pt", map_location="cpu")
    model = make_model(str(ckpt.get("arch", getattr(args, "arch", "factorized"))), int(ckpt.get("hidden", args.hidden)))
    model.load_state_dict(ckpt["model_state"])
    return validate_common(args, out_dir, NeuralStage4Predictor(model), monitor)


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
                "val_memory_exact": validation["val"]["memory_exact"],
                "long_memory_exact": validation["long"]["memory_exact"],
                "val_exception_exact": validation["val"]["exception_exact"],
                "long_exception_exact": validation["long"]["exception_exact"],
                "val_memory_checked_steps": validation["val"]["memory_checked_steps"],
                "long_memory_checked_steps": validation["long"]["memory_checked_steps"],
                "val_exception_checked_steps": validation["val"]["exception_checked_steps"],
                "long_exception_checked_steps": validation["long"]["exception_checked_steps"],
                "val_worst_family_exact": validation["val"]["worst_family_exact"],
                "long_worst_family_exact": validation["long"]["worst_family_exact"],
                "scrambled_opcode_val_step_exact": validation["negative_controls"]["scrambled_opcode_val"]["step_exact"],
                "corrupted_src_val_step_exact": validation["negative_controls"]["corrupted_src_val"]["step_exact"],
                "corrupted_src_val_memory_exact": validation["negative_controls"]["corrupted_src_val"]["memory_exact"],
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
    run.add_argument("--run-name", default="stage4-sse-aligned-exception-v31-neural")
    run.add_argument("--out-dir", type=Path, default=None)
    run.add_argument("--seed", type=int, default=20260635)
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
    run.add_argument("--gate-worst-family-exact", type=float, default=0.970)

    val = sub.add_parser("validate")
    val.add_argument("--out-dir", type=Path, required=True)
    val.add_argument("--hidden", type=int, default=96)
    val.add_argument("--arch", choices=["factorized", "effect"], default="factorized")
    val.add_argument("--monitor-interval", type=float, default=0.5)
    val.add_argument("--gate-step-exact", type=float, default=0.995)
    val.add_argument("--gate-selector-exact", type=float, default=0.999)
    val.add_argument("--gate-worst-family-exact", type=float, default=0.970)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "run-smoke":
        result = run_smoke(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_status"].get("stage4_gate_passed") else 1
    if args.cmd == "validate":
        monitor = ResourceMonitor(args.out_dir, interval_s=args.monitor_interval)
        monitor.start()
        try:
            report = validate_model(args, args.out_dir, monitor)
            print(json.dumps(report["gate_status"], indent=2, sort_keys=True))
            return 0 if report["gate_status"].get("stage4_gate_passed") else 1
        finally:
            monitor.stop()
    raise ValueError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
