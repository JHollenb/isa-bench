#!/usr/bin/env python3
"""Emit a XED-derived iform ledger and map the current PAN subset onto it."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, OPS as STAGE0_OPS, ResourceMonitor, write_json
from pan_x86_stages.stage1_memory_pan import OPS as STAGE1_OPS
from pan_x86_stages.stage2_control_flow_pan import OPS as STAGE2_OPS
from pan_x86_stages.stage3_complex_scalar_pan import OPS as STAGE3_OPS
from pan_x86_stages.stage4_simd_x87_pan import OPS as STAGE4_OPS


PAN_ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa"


def artifact_root_from_out_dir(out_dir: Path) -> Path:
    if out_dir.parent.name == "xed_iform_ledger":
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

# These mappings are intentionally exact to the byte encoders in stage0/1.
# They are not prefix matches, because XED suffixes often distinguish opcode
# direction, opcode-extension groups, APX forms, and undocumented aliases.
STAGE0_XED_IFORMS: Dict[str, Tuple[str, ...]] = {
    "MOV_RI": ("MOV_GPR8_IMMb_B0",),
    "MOV_RR": ("MOV_GPR8_GPR8_88",),
    "ADD_RR": ("ADD_GPR8_GPR8_00",),
    "ADD_RI": ("ADD_GPR8_IMMb_80r0",),
    "SUB_RR": ("SUB_GPR8_GPR8_28",),
    "SUB_RI": ("SUB_GPR8_IMMb_80r5",),
    "ADC_RR": ("ADC_GPR8_GPR8_10",),
    "SBB_RR": ("SBB_GPR8_GPR8_18",),
    "INC_R": ("INC_GPR8",),
    "DEC_R": ("DEC_GPR8",),
    "NEG_R": ("NEG_GPR8",),
    "AND_RR": ("AND_GPR8_GPR8_20",),
    "AND_RI": ("AND_GPR8_IMMb_80r4",),
    "OR_RR": ("OR_GPR8_GPR8_08",),
    "OR_RI": ("OR_GPR8_IMMb_80r1",),
    "XOR_RR": ("XOR_GPR8_GPR8_30",),
    "XOR_RI": ("XOR_GPR8_IMMb_80r6",),
    "NOT_R": ("NOT_GPR8",),
    "CMP_RR": ("CMP_GPR8_GPR8_38",),
    "CMP_RI": ("CMP_GPR8_IMMb_80r7",),
    "TEST_RR": ("TEST_GPR8_GPR8",),
    "SHL1_R": ("SHL_GPR8_ONE_D0r4",),
    "SHR1_R": ("SHR_GPR8_ONE",),
    "ROL1_R": ("ROL_GPR8_ONE",),
    "ROR1_R": ("ROR_GPR8_ONE",),
}
STAGE1_XED_IFORMS: Dict[str, Tuple[str, ...]] = {
    "MOV_LOAD_RM": ("MOV_GPR8_MEMb",),
    "MOV_LOAD_RM16": ("MOV_GPRv_MEMv",),
    "MOV_LOAD_RM32": ("MOV_GPRv_MEMv",),
    "MOV_LOAD_RM64": ("MOV_GPRv_MEMv",),
    "MOV_STORE_MR": ("MOV_MEMb_GPR8",),
    "MOV_STORE_MR16": ("MOV_MEMv_GPRv",),
    "MOV_STORE_MR32": ("MOV_MEMv_GPRv",),
    "MOV_STORE_MR64": ("MOV_MEMv_GPRv",),
    "MOV_STORE_MI": ("MOV_MEMb_IMMb",),
    "MOV_STORE_MI16": ("MOV_MEMv_IMMz",),
    "MOV_STORE_MI32": ("MOV_MEMv_IMMz",),
    "LEA_RA": ("LEA_GPRv_AGEN",),
    "XCHG_RM": ("XCHG_MEMb_GPR8",),
    "XCHG_RM16": ("XCHG_MEMv_GPRv",),
    "XCHG_RM32": ("XCHG_MEMv_GPRv",),
    "XCHG_RM64": ("XCHG_MEMv_GPRv",),
}
STAGE2_XED_IFORMS: Dict[str, Tuple[str, ...]] = {
    "NOP": ("NOP_90",),
    "MOV_AL_IMM": ("MOV_GPR8_IMMb_B0",),
    "ADD_AL_IMM": ("ADD_AL_IMMb",),
    "CMP_AL_IMM": ("CMP_AL_IMMb",),
    "TEST_AL_AL": ("TEST_GPR8_GPR8",),
    "JMP_REL8": ("JMP_RELBRb",),
    "JZ_REL8": ("JZ_RELBRb",),
    "JNZ_REL8": ("JNZ_RELBRb",),
    "JC_REL8": ("JB_RELBRb",),
    "JNC_REL8": ("JNB_RELBRb",),
    "JS_REL8": ("JS_RELBRb",),
    "JNS_REL8": ("JNS_RELBRb",),
    "JO_REL8": ("JO_RELBRb",),
    "JNO_REL8": ("JNO_RELBRb",),
    "CALL_REL32": ("CALL_NEAR_RELBRd",),
    "RET_NEAR": ("RET_NEAR",),
    "PUSH_RAX": ("PUSH_GPRv_50",),
    "POP_RAX": ("POP_GPRv_58",),
}
STAGE3_XED_IFORMS: Dict[str, Tuple[str, ...]] = {
    "BSWAP_R64": ("BSWAP_GPRv",),
    "MOVZX_R64_R8": ("MOVZX_GPRv_GPR8",),
    "MOVSX_R64_R8": ("MOVSX_GPRv_GPR8",),
    "MOVZX_R64_R16": ("MOVZX_GPRv_GPR16",),
    "MOVSX_R64_R16": ("MOVSX_GPRv_GPR16",),
    "MOVZX_R64_M8": ("MOVZX_GPRv_MEMb",),
    "MOVSX_R64_M8": ("MOVSX_GPRv_MEMb",),
    "MOVZX_R64_M16": ("MOVZX_GPRv_MEMw",),
    "MOVSX_R64_M16": ("MOVSX_GPRv_MEMw",),
    "ADD_R64_R64": ("ADD_GPRv_GPRv_03",),
    "ADC_R64_R64": ("ADC_GPRv_GPRv_13",),
    "SUB_R64_R64": ("SUB_GPRv_GPRv_2B",),
    "SBB_R64_R64": ("SBB_GPRv_GPRv_1B",),
    "CMP_R64_R64": ("CMP_GPRv_GPRv_3B",),
    "AND_R64_R64": ("AND_GPRv_GPRv_23",),
    "OR_R64_R64": ("OR_GPRv_GPRv_0B",),
    "XOR_R64_R64": ("XOR_GPRv_GPRv_33",),
    "TEST_R64_R64": ("TEST_GPRv_GPRv",),
    "ADD_R64_M64": ("ADD_GPRv_MEMv",),
    "ADC_R64_M64": ("ADC_GPRv_MEMv",),
    "SUB_R64_M64": ("SUB_GPRv_MEMv",),
    "SBB_R64_M64": ("SBB_GPRv_MEMv",),
    "CMP_R64_M64": ("CMP_GPRv_MEMv",),
    "AND_R64_M64": ("AND_GPRv_MEMv",),
    "OR_R64_M64": ("OR_GPRv_MEMv",),
    "XOR_R64_M64": ("XOR_GPRv_MEMv",),
    "TEST_M64_R64": ("TEST_MEMv_GPRv",),
    "BSF_R64_R64": ("BSF_GPRv_GPRv",),
    "BSR_R64_R64": ("BSR_GPRv_GPRv",),
    "BSF_R64_M64": ("BSF_GPRv_MEMv",),
    "BSR_R64_M64": ("BSR_GPRv_MEMv",),
    "BT_R64_IMM8": ("BT_GPRv_IMMb",),
    "BTS_R64_IMM8": ("BTS_GPRv_IMMb",),
    "BTR_R64_IMM8": ("BTR_GPRv_IMMb",),
    "BTC_R64_IMM8": ("BTC_GPRv_IMMb",),
    "BT_M64_IMM8": ("BT_MEMv_IMMb",),
    "BTS_M64_IMM8": ("BTS_MEMv_IMMb",),
    "BTR_M64_IMM8": ("BTR_MEMv_IMMb",),
    "BTC_M64_IMM8": ("BTC_MEMv_IMMb",),
    "BT_M64_R64": ("BT_MEMv_GPRv",),
    "BTS_M64_R64": ("BTS_MEMv_GPRv",),
    "BTR_M64_R64": ("BTR_MEMv_GPRv",),
    "BTC_M64_R64": ("BTC_MEMv_GPRv",),
    "RCL_R64_1": ("RCL_GPRv_ONE",),
    "RCR_R64_1": ("RCR_GPRv_ONE",),
    "RCL_M64_1": ("RCL_MEMv_ONE",),
    "RCR_M64_1": ("RCR_MEMv_ONE",),
    "RCL_R64_CL": ("RCL_GPRv_CL",),
    "RCR_R64_CL": ("RCR_GPRv_CL",),
    "RCL_M64_CL": ("RCL_MEMv_CL",),
    "RCR_M64_CL": ("RCR_MEMv_CL",),
    "RCL_R64_IMM8": ("RCL_GPRv_IMMb",),
    "RCR_R64_IMM8": ("RCR_GPRv_IMMb",),
    "RCL_M64_IMM8": ("RCL_MEMv_IMMb",),
    "RCR_M64_IMM8": ("RCR_MEMv_IMMb",),
    "ROL_R64_1": ("ROL_GPRv_ONE",),
    "ROR_R64_1": ("ROR_GPRv_ONE",),
    "ROL_M64_1": ("ROL_MEMv_ONE",),
    "ROR_M64_1": ("ROR_MEMv_ONE",),
    "SHL_R64_1": ("SHL_GPRv_ONE_D1r4",),
    "SHR_R64_1": ("SHR_GPRv_ONE",),
    "SAR_R64_1": ("SAR_GPRv_ONE",),
    "SHL_M64_1": ("SHL_MEMv_ONE_D1r4",),
    "SHR_M64_1": ("SHR_MEMv_ONE",),
    "SAR_M64_1": ("SAR_MEMv_ONE",),
    "ROL_R64_CL": ("ROL_GPRv_CL",),
    "ROR_R64_CL": ("ROR_GPRv_CL",),
    "ROL_M64_CL": ("ROL_MEMv_CL",),
    "ROR_M64_CL": ("ROR_MEMv_CL",),
    "SHL_R64_CL": ("SHL_GPRv_CL_D3r4",),
    "SHR_R64_CL": ("SHR_GPRv_CL",),
    "SAR_R64_CL": ("SAR_GPRv_CL",),
    "SHL_M64_CL": ("SHL_MEMv_CL_D3r4",),
    "SHR_M64_CL": ("SHR_MEMv_CL",),
    "SAR_M64_CL": ("SAR_MEMv_CL",),
    "ROL_R64_IMM8": ("ROL_GPRv_IMMb",),
    "ROR_R64_IMM8": ("ROR_GPRv_IMMb",),
    "ROL_M64_IMM8": ("ROL_MEMv_IMMb",),
    "ROR_M64_IMM8": ("ROR_MEMv_IMMb",),
    "SHL_R64_IMM8": ("SHL_GPRv_IMMb_C1r4",),
    "SHR_R64_IMM8": ("SHR_GPRv_IMMb",),
    "SAR_R64_IMM8": ("SAR_GPRv_IMMb",),
    "SHL_M64_IMM8": ("SHL_MEMv_IMMb_C1r4",),
    "SHR_M64_IMM8": ("SHR_MEMv_IMMb",),
    "SAR_M64_IMM8": ("SAR_MEMv_IMMb",),
    "IMUL_R64_R64": ("IMUL_GPRv_GPRv",),
    "IMUL_R64_R64_IMM8": ("IMUL_GPRv_GPRv_IMMb",),
    "MUL_R64": ("MUL_GPRv",),
    "IMUL_ONE_R64": ("IMUL_GPRv",),
    "DIV_R64": ("DIV_GPRv",),
    "IDIV_R64": ("IDIV_GPRv",),
    "DIV_R64_DE": ("DIV_GPRv",),
    "IDIV_R64_DE": ("IDIV_GPRv",),
    "POPCNT_R64_R64": ("POPCNT_GPRv_GPRv",),
    "LZCNT_R64_R64": ("LZCNT_GPRv_GPRv",),
    "TZCNT_R64_R64": ("TZCNT_GPRv_GPRv",),
    "POPCNT_R64_M64": ("POPCNT_GPRv_MEMv",),
    "LZCNT_R64_M64": ("LZCNT_GPRv_MEMv",),
    "TZCNT_R64_M64": ("TZCNT_GPRv_MEMv",),
    "ANDN_R64_R64_R64": ("ANDN_GPR64q_GPR64q_GPR64q",),
    "ANDN_R64_R64_M64": ("ANDN_GPR64q_GPR64q_MEMq",),
    "BEXTR_R64_R64_R64": ("BEXTR_GPR64q_GPR64q_GPR64q",),
    "BEXTR_R64_M64_R64": ("BEXTR_GPR64q_MEMq_GPR64q",),
    "BLSI_R64_R64": ("BLSI_GPR64q_GPR64q",),
    "BLSI_R64_M64": ("BLSI_GPR64q_MEMq",),
    "BLSMSK_R64_R64": ("BLSMSK_GPR64q_GPR64q",),
    "BLSMSK_R64_M64": ("BLSMSK_GPR64q_MEMq",),
    "BLSR_R64_R64": ("BLSR_GPR64q_GPR64q",),
    "BLSR_R64_M64": ("BLSR_GPR64q_MEMq",),
    "SHLX_R64_R64_R64": ("SHLX_GPR64q_GPR64q_GPR64q",),
    "SHLX_R64_M64_R64": ("SHLX_GPR64q_MEMq_GPR64q",),
    "SHRX_R64_R64_R64": ("SHRX_GPR64q_GPR64q_GPR64q",),
    "SHRX_R64_M64_R64": ("SHRX_GPR64q_MEMq_GPR64q",),
    "SARX_R64_R64_R64": ("SARX_GPR64q_GPR64q_GPR64q",),
    "SARX_R64_M64_R64": ("SARX_GPR64q_MEMq_GPR64q",),
    "RORX_R64_R64_IMM8": ("RORX_GPR64q_GPR64q_IMMb",),
    "RORX_R64_M64_IMM8": ("RORX_GPR64q_MEMq_IMMb",),
    "BZHI_R64_R64_R64": ("BZHI_GPR64q_GPR64q_GPR64q",),
    "BZHI_R64_M64_R64": ("BZHI_GPR64q_MEMq_GPR64q",),
    "MULX_R64_R64_R64": ("MULX_GPR64q_GPR64q_GPR64q",),
    "MULX_R64_R64_M64": ("MULX_GPR64q_GPR64q_MEMq",),
    "PDEP_R64_R64_R64": ("PDEP_GPR64q_GPR64q_GPR64q",),
    "PDEP_R64_R64_M64": ("PDEP_GPR64q_GPR64q_MEMq",),
    "PEXT_R64_R64_R64": ("PEXT_GPR64q_GPR64q_GPR64q",),
    "PEXT_R64_R64_M64": ("PEXT_GPR64q_GPR64q_MEMq",),
}
STAGE4_XED_IFORMS: Dict[str, Tuple[str, ...]] = {
    "MOVDQA_XMM_XMM": ("MOVDQA_XMMdq_XMMdq_0F6F",),
    "PAND_XMM_XMM": ("PAND_XMMdq_XMMdq",),
    "PANDN_XMM_XMM": ("PANDN_XMMdq_XMMdq",),
    "POR_XMM_XMM": ("POR_XMMdq_XMMdq",),
    "PXOR_XMM_XMM": ("PXOR_XMMdq_XMMdq",),
    "PADDB_XMM_XMM": ("PADDB_XMMdq_XMMdq",),
    "PADDW_XMM_XMM": ("PADDW_XMMdq_XMMdq",),
    "PADDD_XMM_XMM": ("PADDD_XMMdq_XMMdq",),
    "PADDQ_XMM_XMM": ("PADDQ_XMMdq_XMMdq",),
    "PSUBB_XMM_XMM": ("PSUBB_XMMdq_XMMdq",),
    "PSUBW_XMM_XMM": ("PSUBW_XMMdq_XMMdq",),
    "PSUBD_XMM_XMM": ("PSUBD_XMMdq_XMMdq",),
    "PSUBQ_XMM_XMM": ("PSUBQ_XMMdq_XMMdq",),
    "PCMPEQB_XMM_XMM": ("PCMPEQB_XMMdq_XMMdq",),
    "PCMPEQW_XMM_XMM": ("PCMPEQW_XMMdq_XMMdq",),
    "PCMPEQD_XMM_XMM": ("PCMPEQD_XMMdq_XMMdq",),
    "PMULLW_XMM_XMM": ("PMULLW_XMMdq_XMMdq",),
    "PUNPCKLBW_XMM_XMM": ("PUNPCKLBW_XMMdq_XMMq",),
    "PUNPCKLWD_XMM_XMM": ("PUNPCKLWD_XMMdq_XMMq",),
    "PUNPCKLDQ_XMM_XMM": ("PUNPCKLDQ_XMMdq_XMMq",),
    "PACKSSWB_XMM_XMM": ("PACKSSWB_XMMdq_XMMdq",),
    "PCMPGTB_XMM_XMM": ("PCMPGTB_XMMdq_XMMdq",),
    "PCMPGTW_XMM_XMM": ("PCMPGTW_XMMdq_XMMdq",),
    "PCMPGTD_XMM_XMM": ("PCMPGTD_XMMdq_XMMdq",),
    "PACKUSWB_XMM_XMM": ("PACKUSWB_XMMdq_XMMdq",),
    "PUNPCKHBW_XMM_XMM": ("PUNPCKHBW_XMMdq_XMMq",),
    "PUNPCKHWD_XMM_XMM": ("PUNPCKHWD_XMMdq_XMMq",),
    "PUNPCKHDQ_XMM_XMM": ("PUNPCKHDQ_XMMdq_XMMq",),
    "PACKSSDW_XMM_XMM": ("PACKSSDW_XMMdq_XMMdq",),
    "PUNPCKLQDQ_XMM_XMM": ("PUNPCKLQDQ_XMMdq_XMMq",),
    "PUNPCKHQDQ_XMM_XMM": ("PUNPCKHQDQ_XMMdq_XMMq",),
    "PAVGB_XMM_XMM": ("PAVGB_XMMdq_XMMdq",),
    "PAVGW_XMM_XMM": ("PAVGW_XMMdq_XMMdq",),
    "PMULHUW_XMM_XMM": ("PMULHUW_XMMdq_XMMdq",),
    "PMULHW_XMM_XMM": ("PMULHW_XMMdq_XMMdq",),
    "PMINSW_XMM_XMM": ("PMINSW_XMMdq_XMMdq",),
    "PMAXSW_XMM_XMM": ("PMAXSW_XMMdq_XMMdq",),
    "PMULUDQ_XMM_XMM": ("PMULUDQ_XMMdq_XMMdq",),
    "PMADDWD_XMM_XMM": ("PMADDWD_XMMdq_XMMdq",),
    "PSADBW_XMM_XMM": ("PSADBW_XMMdq_XMMdq",),
    "PMINUB_XMM_XMM": ("PMINUB_XMMdq_XMMdq",),
    "PMAXUB_XMM_XMM": ("PMAXUB_XMMdq_XMMdq",),
    "PSLLW_XMM_IMM8": ("PSLLW_XMMdq_IMMb",),
    "PSLLD_XMM_IMM8": ("PSLLD_XMMdq_IMMb",),
    "PSLLQ_XMM_IMM8": ("PSLLQ_XMMdq_IMMb",),
    "PSRLW_XMM_IMM8": ("PSRLW_XMMdq_IMMb",),
    "PSRLD_XMM_IMM8": ("PSRLD_XMMdq_IMMb",),
    "PSRLQ_XMM_IMM8": ("PSRLQ_XMMdq_IMMb",),
    "PSRAW_XMM_IMM8": ("PSRAW_XMMdq_IMMb",),
    "PSRAD_XMM_IMM8": ("PSRAD_XMMdq_IMMb",),
    "PSLLDQ_XMM_IMM8": ("PSLLDQ_XMMdq_IMMb",),
    "PSRLDQ_XMM_IMM8": ("PSRLDQ_XMMdq_IMMb",),
    "PSLLW_XMM_XMM": ("PSLLW_XMMdq_XMMdq",),
    "PSLLD_XMM_XMM": ("PSLLD_XMMdq_XMMdq",),
    "PSLLQ_XMM_XMM": ("PSLLQ_XMMdq_XMMdq",),
    "PSRLW_XMM_XMM": ("PSRLW_XMMdq_XMMdq",),
    "PSRLD_XMM_XMM": ("PSRLD_XMMdq_XMMdq",),
    "PSRLQ_XMM_XMM": ("PSRLQ_XMMdq_XMMdq",),
    "PSRAW_XMM_XMM": ("PSRAW_XMMdq_XMMdq",),
    "PSRAD_XMM_XMM": ("PSRAD_XMMdq_XMMdq",),
}
for _stage4_op in STAGE4_OPS:
    STAGE4_XED_IFORMS.setdefault(_stage4_op.name, (_stage4_op.iform,))


@dataclasses.dataclass(frozen=True)
class XedRow:
    iclass: str
    extension: str
    category: str
    iform: str
    isa_set: str
    attributes: str

    @property
    def attribute_list(self) -> List[str]:
        return [] if self.attributes == "-" else self.attributes.split(":")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_idata(path: Path) -> List[XedRow]:
    rows: List[XedRow] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 6:
            raise ValueError(f"{path}:{line_no}: expected 6 columns, got {len(parts)}")
        rows.append(XedRow(*parts))
    return rows


def stage_evidence(artifact_root: Path) -> Dict[str, Dict[str, Any]]:
    stage_runs = stage_run_dirs(artifact_root)
    return {
        "stage0_x86_lite_alu": {
            "run_dir": str(stage_runs["stage0_x86_lite_alu"]),
            "validation_report": str(stage_runs["stage0_x86_lite_alu"] / "validation_report.json"),
            "resource_summary": str(stage_runs["stage0_x86_lite_alu"] / "resource_summary.json"),
            "scope": "low-byte ALU over AL/CL/DL/BL, no memory or control flow",
            "oracle": "Unicorn",
        },
        "stage1_address_memory": {
            "run_dir": str(stage_runs["stage1_address_memory"]),
            "validation_report": str(stage_runs["stage1_address_memory"] / "validation_report.json"),
            "qemu_oracle_report": str(stage_runs["stage1_address_memory"] / "qemu_stage1_oracle" / "qemu_stage1_oracle_report.json"),
            "resource_summary": str(stage_runs["stage1_address_memory"] / "resource_summary.json"),
            "scope": "8/16/32/64-bit scalar memory/addressing over RAX/RCX/RDX/RBX, RSI/RDI/RBP bases, and R8/R9 index-scale offsets",
            "oracle": "Unicorn plus QEMU sample",
        },
        "stage2_control_flow": {
            "run_dir": str(stage_runs["stage2_control_flow"]),
            "validation_report": str(stage_runs["stage2_control_flow"] / "validation_report.json"),
            "qemu_oracle_report": str(stage_runs["stage2_control_flow"] / "qemu_stage2_oracle" / "qemu_stage2_oracle_report.json"),
            "bochs_oracle_report": str(stage_runs["stage2_control_flow"] / "bochs_stage2_oracle" / "bochs_stage2_oracle_report.json"),
            "resource_summary": str(stage_runs["stage2_control_flow"] / "resource_summary.json"),
            "scope": "RAX/AL plus CF/ZF/SF/OF, RIP/RSP, Jcc/JMP/CALL/RET/PUSH/POP over a bounded stack",
            "oracle": "Unicorn plus QEMU and Bochs samples",
        },
        "stage3_complex_scalar": {
            "run_dir": str(stage_runs["stage3_complex_scalar"]),
            "validation_report": str(stage_runs["stage3_complex_scalar"] / "validation_report.json"),
            "qemu_oracle_report": str(stage_runs["stage3_complex_scalar"] / "qemu_stage3_base_alu_priority_oracle" / "qemu_stage3_oracle_report.json"),
            "bochs_oracle_report": str(stage_runs["stage3_complex_scalar"] / "bochs_stage3_base_alu_priority_oracle" / "bochs_stage3_oracle_report.json"),
            "resource_summary": str(stage_runs["stage3_complex_scalar"] / "resource_summary.json"),
            "scope": "RAX/RBX/RCX/RDX 64-bit complex-scalar register subset with safe no-fault divide, DIV/IDIV #DE, MOVSX/MOVZX r8/r16/m8/m16 extension forms, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR r64/r64 and r64/m64 forms, TEST r64/r64 and m64/r64 forms, BSF/BSR/POPCNT/LZCNT/TZCNT register forms, read-only BSF/BSR/POPCNT/LZCNT/TZCNT m64 source-memory forms over a flat 256-byte memory image, selected BMI1 ANDN/BEXTR/BLSI/BLSMSK/BLSR forms, selected BMI2 BZHI/SHLX/SHRX/SARX/RORX/MULX/PDEP/PEXT forms, BT/BTS/BTR/BTC m64 imm8/r64 bit-test/modify forms, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 forms",
            "oracle": "internal semantics plus QEMU and Bochs samples",
        },
        "stage4_simd_x87": {
            "run_dir": str(stage_runs["stage4_simd_x87"]),
            "validation_report": str(stage_runs["stage4_simd_x87"] / "validation_report.json"),
            "qemu_oracle_report": str(stage_runs["stage4_simd_x87"] / "qemu_stage4_non_exception_priority_oracle" / "qemu_stage4_oracle_report.json"),
            "bochs_oracle_report": str(stage_runs["stage4_simd_x87"] / "bochs_stage4_aligned_exception_priority_oracle" / "bochs_stage4_oracle_report.json"),
            "resource_summary": str(stage_runs["stage4_simd_x87"] / "resource_summary.json"),
            "scope": "SSE/SSE2 XMM0-XMM15 register-register, aligned source-memory, reciprocal MOVD/MOVQ/MOVSD/MOVLPD/MOVHPD/MOVAPD/MOVUPD memory-source transfer loads, MOVSD/MOVAPD/MOVUPD/MOVQ register transfer forms, MOVDQA/MOVDQU/MOVAPD/MOVUPD/MOVNTDQ/MOVNTPD m128 stores, MOVSD/MOVLPD/MOVHPD/MOVQ m64 stores, MOVD m32 stores, aligned m128 unaligned-memory #GP(0) cases, scalar/packed single and double add/sub/mul/div/sqrt/min/max register and memory-source forms over safe finite plus non-trapping zero/infinity operands, and CMPSS/CMPPS/CMPSD/CMPPD register/memory-source compare forms over safe finite, zero, infinity, quiet-NaN, and signed-zero operands with exact XMM/RIP/memory/exception comparison",
            "oracle": "internal semantics plus QEMU and Bochs samples",
        },
    }


def implemented_subset_records() -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for op in STAGE0_OPS:
        records.append(
            {
                "stage": "stage0_x86_lite_alu",
                "internal_name": op.name,
                "internal_iform": op.iform,
                "family": op.family,
                "operands": op.operands,
                "xed_iforms": list(STAGE0_XED_IFORMS[op.name]),
                "coverage_status": "operand_subset_learned_validated",
                "limitations": [
                    "AL/CL/DL/BL low-byte operands only",
                    "no AH/SPL/BPL/SIL/DIL/R8B-R15B alias coverage",
                    "no mode, prefix, exception, or memory variation",
                ],
            }
        )
    for op in STAGE1_OPS:
        records.append(
            {
                "stage": "stage1_address_memory",
                "internal_name": op.name,
                "internal_iform": op.iform,
                "family": op.family,
                "operands": op.operands,
                "xed_iforms": list(STAGE1_XED_IFORMS[op.name]),
                "coverage_status": "operand_subset_learned_validated",
                "limitations": [
                    "RAX/RCX/RDX/RBX data operands only",
                    "RSI/RDI/RBP bases plus R8/R9 index-scale addressing only",
                    "flat 256-byte memory image; no faults, segments, pages, or full alias set",
                ],
            }
        )
    for op in STAGE2_OPS:
        records.append(
            {
                "stage": "stage2_control_flow",
                "internal_name": op.name,
                "internal_iform": op.iform,
                "family": op.family,
                "operands": op.operands,
                "xed_iforms": list(STAGE2_XED_IFORMS[op.name]),
                "coverage_status": "operand_subset_learned_validated",
                "limitations": [
                    "RAX/AL and RSP only; no broad GPR/control-register alias coverage",
                    "CF/ZF/SF/OF tracked for selected flag-gated branches only",
                    "bounded flat stack image; no page faults, exceptions, interrupts, or privilege transitions",
                    "no SETcc, CMOVcc, far calls, indirect branches, or RIP-relative memory forms",
                ],
            }
        )
    for op in STAGE3_OPS:
        records.append(
            {
                "stage": "stage3_complex_scalar",
                "internal_name": op.name,
                "internal_iform": op.iform,
                "family": op.family,
                "operands": op.operands,
                "xed_iforms": list(STAGE3_XED_IFORMS[op.name]),
                "coverage_status": "operand_subset_learned_validated",
                "limitations": [
                    "RAX/RBX/RCX/RDX register operands plus selected read-only m8/m64 source-memory, BT/BTS/BTR/BTC m64 imm8/r64 memory-destination, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR r64/m64 forms only",
                    "divide-error coverage is limited to DIV/IDIV #DE cases",
                    "undefined flags are masked rather than claimed exact",
                    "CPUID-gated scalar coverage is limited to POPCNT/LZCNT/TZCNT plus selected BMI1/BMI2 register and m64 source forms",
                    "memory operands are limited to selected read-only MOVSX/MOVZX m8/m16, ADD/ADC/SUB/SBB/CMP/AND/OR/XOR m64 source forms, TEST m64 r64, selected BMI1/BMI2 m64 source forms, BT/BTS/BTR/BTC m64 imm8/r64, one-count RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms, and CL/imm8 RCL/RCR/ROL/ROR/SHL/SHR/SAR m64 forms over a flat 256-byte image",
                    "no broader memory operands, broader BMI forms, APX variants, or mode variation",
                ],
            }
        )
    for op in STAGE4_OPS:
        records.append(
            {
                "stage": "stage4_simd_x87",
                "internal_name": op.name,
                "internal_iform": op.iform,
                "family": op.family,
                "operands": op.operands,
                "xed_iforms": list(STAGE4_XED_IFORMS[op.name]),
                "coverage_status": "operand_subset_learned_validated",
                "limitations": [
                    "XMM0-XMM15 register operands",
                    "register-register, aligned absolute m128 source-memory, reciprocal MOVD/MOVQ/MOVLPD/MOVHPD/MOVAPD/MOVUPD memory-source transfer loads, MOVDQA/MOVDQU/MOVAPD/MOVUPD/MOVNTDQ/MOVNTPD m128 memory-destination, MOVSD/MOVLPD/MOVHPD/MOVQ m64 memory-destination, MOVD m32 memory-destination, unaligned MOVDQU source/destination, aligned m128 unaligned-memory #GP(0), scalar/packed single and double add/sub/mul/div/sqrt/min/max register and memory-source forms over safe finite plus non-trapping zero/infinity operands, CMPSS/CMPPS/CMPSD/CMPPD register and memory-source forms over safe finite, zero, infinity, quiet-NaN, and signed-zero operands, selected imm8, and XMM-count shift forms only",
                    "no memory sources or destinations beyond the current transfer/FP/integer subset, and no exception modeling beyond aligned m128 unaligned-memory #GP(0)",
                    "no AVX/VEX/EVEX forms, masks, YMM/ZMM, MMX, or x87 stack behavior",
                    "no subnormal, overflow, underflow, divide-by-zero status-flag, invalid-operation status-flag, signaling-NaN behavior, quiet-NaN arithmetic/min-max payload behavior, signed-zero min-max edge claims beyond oracle-agreeing non-trapping cases, or broader MXCSR exception behavior",
                ],
            }
        )
    return records


def histogram(rows: Sequence[XedRow], field: str) -> Dict[str, int]:
    counter = Counter(getattr(row, field) for row in rows)
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def attribute_histogram(rows: Sequence[XedRow]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.attribute_list)
    return dict(sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])))


def xed_presence(xed_install: Path) -> Dict[str, Any]:
    include_dir = xed_install / "include" / "xed"
    lib_dir = xed_install / "lib"
    misc_dir = xed_install / "misc"
    required = {
        "idata": misc_dir / "idata.txt",
        "iform_enum": include_dir / "xed-iform-enum.h",
        "iclass_enum": include_dir / "xed-iclass-enum.h",
        "libxed": lib_dir / "libxed.a",
    }
    return {
        "install_dir": str(xed_install),
        "required": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in required.items()
        },
        "kit_ok": all(path.exists() for path in required.values()),
    }


def build_ledger(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    artifact_root = artifact_root_from_out_dir(out_dir)
    default_xed_install = artifact_root / "source_deps" / "intel-xed-main-2026-06-16" / "install"
    xed_install = args.xed_install or default_xed_install
    xed_build_report = args.xed_build_report or (xed_install.parent / "xed_build_report.json")
    idata_path = xed_install / "misc" / "idata.txt"
    if not idata_path.exists():
        raise FileNotFoundError(f"missing XED idata file: {idata_path}")

    rows = parse_idata(idata_path)
    by_iform = {row.iform: row for row in rows}
    subset_records = implemented_subset_records()
    subset_iforms = {
        xed_iform
        for record in subset_records
        for xed_iform in record["xed_iforms"]
    }
    missing_mapped_iforms = sorted(subset_iforms.difference(by_iform))
    if missing_mapped_iforms:
        raise ValueError(f"mapped XED iforms missing from idata: {missing_mapped_iforms}")

    build_report = load_json(xed_build_report)
    records: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        status = "operand_subset_learned_validated" if row.iform in subset_iforms else "unimplemented"
        rec: Dict[str, Any] = {
            "index": index,
            "iclass": row.iclass,
            "extension": row.extension,
            "category": row.category,
            "iform": row.iform,
            "isa_set": row.isa_set,
            "attributes": row.attribute_list,
            "coverage_status": status,
        }
        if status == "operand_subset_learned_validated":
            matched = [r for r in subset_records if row.iform in r["xed_iforms"]]
            rec["subset_evidence"] = matched
        records.append(rec)

    status_counts = Counter(rec["coverage_status"] for rec in records)
    validated_categories = Counter(
        rec["category"] for rec in records if rec["coverage_status"] == "operand_subset_learned_validated"
    )
    unimplemented_categories = Counter(
        rec["category"] for rec in records if rec["coverage_status"] == "unimplemented"
    )
    ledger = {
        "created": DEFAULT_DATE,
        "run_name": args.run_name,
        "ledger_kind": "intel_xed_idata_iform_inventory",
        "full_x86_goal_complete": False,
        "full_x86_coverage_complete": False,
        "coverage_status_semantics": {
            "operand_subset_learned_validated": (
                "The stage runner learned and validated a constrained operand/state subset "
                "for this XED iform. This is not full iform semantic coverage."
            ),
            "unimplemented": "No PAN trace/training/validation coverage exists yet.",
        },
        "xed_source": {
            "install_dir": str(xed_install),
            "idata_path": str(idata_path),
            "idata_sha256": sha256_file(idata_path),
            "build_report": str(xed_build_report),
            "xed_url": build_report.get("xed_url"),
            "mbuild_url": build_report.get("mbuild_url"),
            "branch": build_report.get("branch"),
            "xed_commit": build_report.get("xed_commit"),
            "mbuild_commit": build_report.get("mbuild_commit"),
            "kit_ok": xed_presence(xed_install)["kit_ok"],
            "cli_ok": bool(build_report.get("xed_binary_probe", {}).get("ok")),
        },
        "summary": {
            "xed_idata_rows": len(rows),
            "unique_xed_iforms": len({row.iform for row in rows}),
            "unique_iclasses": len({row.iclass for row in rows}),
            "status_counts": dict(sorted(status_counts.items())),
            "operand_subset_validated_xed_iforms": status_counts.get("operand_subset_learned_validated", 0),
            "fully_validated_xed_iforms": 0,
            "implemented_internal_stage_forms": len(subset_records),
            "validated_subset_categories": dict(sorted(validated_categories.items())),
            "largest_unimplemented_categories": dict(unimplemented_categories.most_common(20)),
        },
        "histograms": {
            "category": histogram(rows, "category"),
            "extension": histogram(rows, "extension"),
            "isa_set": histogram(rows, "isa_set"),
            "attributes": attribute_histogram(rows),
        },
        "stage_evidence": stage_evidence(artifact_root),
        "implemented_subset_map": subset_records,
        "records": records,
        "next_required_work": [
            "Expand each operand-subset match to full operand/register/addressing coverage or mark residual gaps.",
            "Add XED-driven trace generation for selected unimplemented scalar iform families.",
            "Add portable QEMU/TCG and Bochs oracles for new generated forms.",
            "Freeze Intel SDM and AMD APM references by hash before promoting beyond measured local evidence.",
        ],
    }
    write_json(out_dir / "xed_iform_ledger.json", ledger)
    write_json(
        out_dir / "summary.json",
        {
            "run_name": args.run_name,
            "out_dir": str(out_dir),
            "summary": ledger["summary"],
            "xed_source": ledger["xed_source"],
            "full_x86_coverage_complete": ledger["full_x86_coverage_complete"],
        },
    )
    return ledger


def default_out_dir(run_name: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / "xed_iform_ledger" / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-name", default=f"xed-iform-ledger-{int(time.time())}")
    p.add_argument("--xed-install", type=Path, default=None)
    p.add_argument("--xed-build-report", type=Path, default=None)
    p.add_argument("--monitor-interval", type=float, default=0.5)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or default_out_dir(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(out_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("enumerate_xed")
        ledger = build_ledger(args, out_dir)
        monitor.set_phase("finalize")
        print(
            json.dumps(
                {
                    "out_dir": str(out_dir),
                    "summary": ledger["summary"],
                    "xed_source": ledger["xed_source"],
                    "full_x86_coverage_complete": ledger["full_x86_coverage_complete"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
