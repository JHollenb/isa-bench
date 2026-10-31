"""SPF primary instruction-transition model for the standalone runtime."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from .compat import ensure_scripts_importable
from .decode import DecodedInstruction
from .spf_backend import SPFKernelBundle, build_runtime_kernels
from .transitions import MemoryWrite, TransitionProposal

ensure_scripts_importable()

from pan_x86_isa.runtime import (  # noqa: E402
    GPR64_BY_CODE,
    MASK64,
    SIGNAL_SEGV,
    SIGNAL_TRAP,
    StopReason,
    UnsupportedInstruction,
    _decode_modrm_address,
    _read_le,
    _read_reg,
    _sign_extend,
)


FLAG_EFFECT_MASK = (1 << 0) | (1 << 6) | (1 << 7) | (1 << 11)
MASK80 = (1 << 80) - 1
MASK128 = (1 << 128) - 1
X87_ONE = 0x3FFF8000000000000000
X87_ZERO = 0


def _reg_name(code: int) -> str:
    return GPR64_BY_CODE[code & 0xF]


def _xmm_name(code: int) -> str:
    return f"XMM{code & 0xF}"


def _read_xmm(state: object, code: int) -> int:
    return int(state.xmm[code & 0xF]) & MASK128


def _xmm_bytes(value: int) -> bytes:
    return (int(value) & MASK128).to_bytes(16, "little")


def _lane_binary(value_a: int, value_b: int, lane_bits: int, op: object) -> int:
    lane_mask = (1 << lane_bits) - 1
    out = 0
    for bit in range(0, 128, lane_bits):
        a = (int(value_a) >> bit) & lane_mask
        b = (int(value_b) >> bit) & lane_mask
        out |= (int(op(a, b)) & lane_mask) << bit
    return out & MASK128


def _write_mask(width: int) -> int:
    if width == 8:
        return 0xFF
    if width == 32:
        # In long mode, 32-bit GPR writes architecturally zero the high half.
        return MASK64
    if width == 64:
        return MASK64
    raise ValueError(f"unsupported register width {width}")


def _final_reg_value(current: int, value: int, width: int) -> int:
    if width == 32:
        return int(value) & 0xFFFFFFFF
    if width == 64:
        return int(value) & MASK64
    if width == 8:
        return (int(current) & ~0xFF) | (int(value) & 0xFF)
    raise ValueError(f"unsupported register width {width}")


def _flagged(rflags: int, value: int, width: int) -> int:
    mask = (1 << width) - 1
    value &= mask
    if value == 0:
        rflags |= 1 << 6
    else:
        rflags &= ~(1 << 6)
    if value & (1 << (width - 1)):
        rflags |= 1 << 7
    else:
        rflags &= ~(1 << 7)
    return rflags & MASK64


def _memory_read_event(address: int, size: int, *, purpose: str = "read") -> dict[str, Any]:
    return {"address": hex(int(address) & MASK64), "size": int(size), "access": "read", "purpose": purpose}


def _x87_state_record(state: object) -> dict[str, Any]:
    return {
        "st": [None if value is None else int(value) & MASK80 for value in state.x87],
        "top": int(state.x87_top) & 0x7,
        "tag": int(state.x87_tag) & 0xFFFF,
        "status": int(state.x87_status) & 0xFFFF,
        "control": int(state.x87_control) & 0xFFFF,
    }


def _x87_set_tag(tag_word: int, index: int, tag: int) -> int:
    shift = (index & 0x7) * 2
    return (int(tag_word) & ~(0x3 << shift)) | ((int(tag) & 0x3) << shift)


def _x87_status_with_top(status: int, top: int) -> int:
    return (int(status) & ~(0x7 << 11)) | ((int(top) & 0x7) << 11)


def _x87_push_state(state: object, value: int, *, zero: bool = False) -> dict[str, Any] | StopReason:
    out = _x87_state_record(state)
    new_top = (out["top"] - 1) & 0x7
    if out["st"][new_top] is not None:
        return StopReason("fault", signal=SIGNAL_SEGV, detail="x87 stack overflow", address=state.gpr["RIP"])
    out["st"][new_top] = int(value) & MASK80
    out["top"] = new_top
    out["tag"] = _x87_set_tag(out["tag"], new_top, 1 if zero else 0)
    out["status"] = _x87_status_with_top(out["status"], new_top)
    return out


def _x87_pop_state(state: object) -> dict[str, Any] | StopReason:
    out = _x87_state_record(state)
    top = int(out["top"]) & 0x7
    if out["st"][top] is None:
        return StopReason("fault", signal=SIGNAL_SEGV, detail="x87 stack underflow", address=state.gpr["RIP"])
    out["st"][top] = None
    out["tag"] = _x87_set_tag(out["tag"], top, 3)
    out["top"] = (top + 1) & 0x7
    out["status"] = _x87_status_with_top(out["status"], out["top"])
    return out


def _x87_fstp_state(state: object, logical_index: int) -> dict[str, Any] | StopReason:
    out = _x87_state_record(state)
    top = int(out["top"]) & 0x7
    value = out["st"][top]
    if value is None:
        return StopReason("fault", signal=SIGNAL_SEGV, detail="x87 stack underflow", address=state.gpr["RIP"])
    dst = (top + int(logical_index)) & 0x7
    out["st"][dst] = value
    out["tag"] = _x87_set_tag(out["tag"], dst, 0)
    out["st"][top] = None
    out["tag"] = _x87_set_tag(out["tag"], top, 3)
    out["top"] = (top + 1) & 0x7
    out["status"] = _x87_status_with_top(out["status"], out["top"])
    return out


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _decoded_identity(instruction: DecodedInstruction) -> str:
    prefixes = instruction.prefixes
    opcode = instruction.opcode
    raw = instruction.raw
    cursor = instruction.cursor_after_opcode
    if opcode == 0x0F:
        second = raw[cursor]
        prefix = prefixes[-1] if prefixes else None
        if second == 0x05 and prefix is None:
            return "SYSCALL"
        if prefix == 0x66 and second == 0x6F:
            return "MOVDQA_XMM_XMM_M128"
        if prefix == 0xF3 and second == 0x6F:
            return "MOVDQU_XMM_XMM_M128"
        if prefix == 0x66 and second == 0x7F:
            return "MOVDQA_XMM_M128_XMM"
        if prefix == 0xF3 and second == 0x7F:
            return "MOVDQU_XMM_M128_XMM"
        if prefix == 0x66 and second == 0xEF:
            return "PXOR_XMM_XMM_M128"
        if prefix == 0x66 and second == 0xD4:
            return "PADDQ_XMM_XMM_M128"
        return f"OP_0F{second:02X}"
    if opcode == 0xD9:
        second = raw[cursor]
        if second == 0xD0:
            return "FNOP"
        if second == 0xE8:
            return "FLD1"
        if second == 0xEE:
            return "FLDZ"
        return f"X87_D9_{second:02X}"
    if opcode == 0xDD:
        second = raw[cursor]
        if 0xD8 <= second <= 0xDF:
            return "FSTP_ST"
        return f"X87_DD_{second:02X}"
    if opcode == 0x90:
        return "NOP"
    if opcode == 0xCC:
        return "INT3"
    if opcode == 0x0F and instruction.raw[instruction.cursor_after_opcode] == 0x05:
        return "SYSCALL"
    if opcode == 0x55 and instruction.rex == 0:
        return "PUSH_RBP"
    if opcode == 0x5D and instruction.rex == 0:
        return "POP_RBP"
    if opcode == 0xC9:
        return "LEAVE"
    if opcode == 0xC3:
        return "RET"
    if opcode == 0xE8:
        return "CALL_REL32"
    if opcode == 0xE9:
        return "JMP_REL32"
    if opcode == 0xEB:
        return "JMP_REL8"
    if 0xB8 <= opcode <= 0xBF:
        return "MOV_IMM_REG"
    if opcode == 0x89:
        return "MOV_RM_REG"
    if opcode == 0x8B:
        return "MOV_REG_RM"
    if opcode == 0x8D:
        return "LEA"
    if opcode == 0x31:
        return "XOR_RM_REG"
    if opcode == 0x83:
        return "ADD_SUB_IMM_RM"
    if opcode == 0xC7:
        return "MOV_IMM_RM"
    return f"OP_{opcode:02X}"


def _promoted_forms_from_ledger(ledger: dict[str, Any], profile: str) -> set[str]:
    promoted: set[str] = set()
    for key, item in (ledger.get("promotions") or {}).items():
        if "::" in str(key):
            iform, row_profile = str(key).split("::", 1)
            if row_profile == profile:
                promoted.add(iform)
        elif isinstance(item, dict):
            row_profile = item.get("profile") or item.get("target_profile")
            if row_profile in (None, profile):
                promoted.add(str(key))
        else:
            promoted.add(str(key))
    return promoted


def _runtime_stage_for_name(name: str) -> str:
    if name in {"MOV_RM_REG", "MOV_REG_RM", "MOV_IMM_RM", "LEA"}:
        return "stage1_address_memory"
    if name in {"NOP", "INT3", "SYSCALL", "PUSH_RBP", "POP_RBP", "LEAVE", "RET", "CALL_REL32", "JMP_REL32", "JMP_REL8"}:
        return "stage2_control_flow"
    if name in {"XOR_RM_REG", "ADD_IMM_RM", "SUB_IMM_RM"}:
        return "stage3_complex_scalar"
    if name.startswith(("SSE", "XMM", "MOVDQA", "MOVDQU", "ADDPS", "ADDPD", "PXOR", "PADDQ", "F")) or name.startswith("X87_"):
        return "stage4_simd_x87"
    return "stage0_register_scalar"


def _runtime_effect_axes(
    name: str,
    *,
    next_rip: int | None = None,
    register_writes: dict[str, int] | None = None,
    xmm_writes: dict[str, int] | None = None,
    memory_reads: tuple[dict[str, Any], ...] = (),
    memory_writes: tuple[MemoryWrite, ...] = (),
    rflags: int | None = None,
    stack_event: str = "none",
    scalar_effect: str | None = None,
    vector_effect: str | None = None,
    x87_effect: str | None = None,
    exception: str | None = None,
) -> dict[str, Any]:
    writes = register_writes or {}
    xmm = xmm_writes or {}
    stage = _runtime_stage_for_name(name)
    if stage == "stage1_address_memory":
        if memory_writes:
            event_kind = "store"
        elif memory_reads:
            event_kind = "load"
        else:
            event_kind = "none"
        value_source = "effective_address" if name == "LEA" else "memory_read" if memory_reads else "register_or_immediate"
        return {
            "stage": stage,
            "register_value_source": value_source,
            "memory_event_kind": event_kind,
            "memory_read_count": len(memory_reads),
            "memory_write_count": len(memory_writes),
            "written_registers": sorted(writes),
        }
    if stage == "stage2_control_flow":
        if name == "RET":
            rip_target = "stack_pop"
        elif name.startswith(("JMP", "CALL")):
            rip_target = "relative_target"
        elif name in {"INT3", "SYSCALL"}:
            rip_target = "trap_or_syscall"
        else:
            rip_target = "fallthrough"
        return {
            "stage": stage,
            "rip_target": rip_target,
            "stack_event": stack_event,
            "next_rip": None if next_rip is None else hex(int(next_rip) & MASK64),
            "written_registers": sorted(writes),
        }
    if stage == "stage3_complex_scalar":
        return {
            "stage": stage,
            "scalar_family_effect": scalar_effect or name,
            "flag_write_policy": "writes_zf_sf_cf_of_subset" if rflags is not None else "preserve",
            "memory_event_kind": "store" if memory_writes else "read" if memory_reads else "none",
            "written_registers": sorted(writes),
        }
    if stage == "stage4_simd_x87":
        return {
            "stage": stage,
            "xmm_lane_effect": vector_effect or name,
            "memory_event_kind": "store" if memory_writes else "read" if memory_reads else "none",
            "memory_read_count": len(memory_reads),
            "memory_write_count": len(memory_writes),
            "written_xmm_registers": sorted(xmm),
            "x87_stack_effect": x87_effect,
            "exception_class": exception,
        }
    return {
        "stage": stage,
        "register_value_source": "immediate_or_scalar",
        "written_registers": sorted(writes),
    }


@dataclass
class BootstrapSPFTransitionModel:
    """SPF-backed transition model for the first standalone runtime milestone.

    This is the bootstrap cartridge: it uses Bob SPFBlockMatrix/SPFTransportKernel
    for the model envelope and strict proof/audit metadata, while the per-op heads
    are deterministic transition heads for the initial static-ELF instruction
    subset. Trained SPF heads can replace these methods without changing the
    runtime commit contract.
    """

    kernels: SPFKernelBundle | None = None
    model_id: str = "pan-isa-x86-spf-bootstrap-v1"
    profile: str = "long64-ring3"

    def __post_init__(self) -> None:
        if self.kernels is None:
            self.kernels = build_runtime_kernels()

    @property
    def manifest(self) -> dict[str, Any]:
        assert self.kernels is not None
        return {
            "model_id": self.model_id,
            "model_kind": "spf_primary_bootstrap_transition_model",
            "runtime_mode": "bootstrap",
            "target_profile": self.profile,
            "trainable_weights_loaded": False,
            "primary_semantics_path": "SPFBlockMatrix/SPFTransportKernel transition proposal",
            "commit_contract": "typed exact architectural state transition",
            "spf": self.kernels.manifest,
        }

    def _proposal(
        self,
        instruction: DecodedInstruction,
        *,
        name: str,
        next_rip: int | None = None,
        kind: str = "normal",
        register_writes: dict[str, int] | None = None,
        register_write_masks: dict[str, int] | None = None,
        xmm_writes: dict[str, int] | None = None,
        xmm_write_masks: dict[str, int] | None = None,
        memory_reads: tuple[dict[str, Any], ...] = (),
        memory_writes: tuple[MemoryWrite, ...] = (),
        rflags: int | None = None,
        flag_write_mask: int | None = None,
        mxcsr: int | None = None,
        x87_state: dict[str, Any] | None = None,
        stack_event: str = "none",
        scalar_effect: str | None = None,
        vector_effect: str | None = None,
        x87_effect: str | None = None,
        exception: str | None = None,
        stop_reason: StopReason | None = None,
    ) -> TransitionProposal:
        writes = dict(register_writes or {})
        xmm_writes = dict(xmm_writes or {})
        axes = _runtime_effect_axes(
            name,
            next_rip=next_rip,
            register_writes=writes,
            xmm_writes=xmm_writes,
            memory_reads=memory_reads,
            memory_writes=memory_writes,
            rflags=rflags,
            stack_event=stack_event,
            scalar_effect=scalar_effect,
            vector_effect=vector_effect,
            x87_effect=x87_effect,
            exception=exception,
        )
        return TransitionProposal(
            kind=kind,
            rip=instruction.rip,
            instruction_hex=instruction.opcode_hex,
            instruction_name=name,
            stage=_runtime_stage_for_name(name),
            effect_axes=axes,
            profile=self.profile,
            iclass=name,
            iform=name,
            next_rip=next_rip,
            register_writes=writes,
            register_write_masks=dict(register_write_masks or {}),
            xmm_writes=xmm_writes,
            xmm_write_masks=dict(xmm_write_masks or {}),
            memory_reads=tuple(memory_reads),
            memory_writes=tuple(memory_writes),
            rflags=rflags,
            flag_write_mask=flag_write_mask,
            flag_defined_mask=flag_write_mask,
            mxcsr=mxcsr,
            x87_state=x87_state,
            exception=None if exception is None else {"class": exception},
            stop_reason=stop_reason,
            spf_manifest=self.manifest,
            family_provenance=_runtime_stage_for_name(name),
            confidence=1.0,
        )

    def _stage4_simd_proposal(
        self,
        runtime: object,
        instruction: DecodedInstruction,
    ) -> TransitionProposal | None:
        state = runtime.state
        memory = runtime.memory
        raw = instruction.raw
        rex = instruction.rex
        rip = instruction.rip
        cursor = instruction.cursor_after_opcode
        second = raw[cursor]
        prefix = instruction.prefixes[-1] if instruction.prefixes else None
        if prefix not in (0x66, 0xF3) or second not in (0x6F, 0x7F, 0xEF, 0xD4):
            return None
        if second in (0xEF, 0xD4) and prefix != 0x66:
            return None

        modrm = raw[cursor + 1]
        mod, reg, rm_or_address, end = _decode_modrm_address(state, memory, rip + cursor + 2, modrm, rex)
        is_unaligned_move = prefix == 0xF3
        move_kind = "MOVDQU" if is_unaligned_move else "MOVDQA"
        aligned_required = prefix == 0x66 and second in (0x6F, 0x7F)

        if aligned_required and mod != 3 and (rm_or_address & 0xF):
            name = f"{move_kind}_ALIGNMENT_FAULT"
            return self._proposal(
                instruction,
                name=name,
                kind="exception",
                exception="general_protection",
                vector_effect="alignment_fault",
                stop_reason=StopReason(
                    "fault",
                    signal=SIGNAL_SEGV,
                    detail=f"{move_kind} unaligned memory operand #GP(0)",
                    address=rm_or_address,
                ),
            )

        if second == 0x6F:
            dst = _xmm_name(reg)
            memory_reads: tuple[dict[str, Any], ...] = ()
            if mod == 3:
                value = _read_xmm(state, rm_or_address)
                name = f"{move_kind}_XMM_XMM"
                vector_effect = "xmm_register_move"
            else:
                value = _read_le(memory.read(rm_or_address, 16))
                memory_reads = (_memory_read_event(rm_or_address, 16, purpose="xmm_load"),)
                name = f"{move_kind}_XMM_M128"
                vector_effect = "xmm_load"
            return self._proposal(
                instruction,
                name=name,
                next_rip=end,
                xmm_writes={dst: value},
                xmm_write_masks={dst: MASK128},
                memory_reads=memory_reads,
                vector_effect=vector_effect,
            )

        if second == 0x7F:
            src_value = _read_xmm(state, reg)
            if mod == 3:
                dst = _xmm_name(rm_or_address)
                return self._proposal(
                    instruction,
                    name=f"{move_kind}_XMM_XMM",
                    next_rip=end,
                    xmm_writes={dst: src_value},
                    xmm_write_masks={dst: MASK128},
                    vector_effect="xmm_register_move",
                )
            return self._proposal(
                instruction,
                name=f"{move_kind}_M128_XMM",
                next_rip=end,
                memory_writes=(MemoryWrite(rm_or_address, _xmm_bytes(src_value)),),
                vector_effect="xmm_store",
            )

        dst = _xmm_name(reg)
        a = _read_xmm(state, reg)
        if mod == 3:
            b = _read_xmm(state, rm_or_address)
            memory_reads = ()
        else:
            b = _read_le(memory.read(rm_or_address, 16))
            memory_reads = (_memory_read_event(rm_or_address, 16, purpose="xmm_operand_load"),)
        if second == 0xEF:
            value = (a ^ b) & MASK128
            name = "PXOR_XMM_XMM" if mod == 3 else "PXOR_XMM_M128"
            vector_effect = "packed_xor"
        else:
            value = _lane_binary(a, b, 64, lambda x, y: x + y)
            name = "PADDQ_XMM_XMM" if mod == 3 else "PADDQ_XMM_M128"
            vector_effect = "packed_add_qword"
        return self._proposal(
            instruction,
            name=name,
            next_rip=end,
            xmm_writes={dst: value},
            xmm_write_masks={dst: MASK128},
            memory_reads=memory_reads,
            vector_effect=vector_effect,
        )

    def _stage4_x87_proposal(
        self,
        runtime: object,
        instruction: DecodedInstruction,
    ) -> TransitionProposal | None:
        state = runtime.state
        raw = instruction.raw
        rip = instruction.rip
        cursor = instruction.cursor_after_opcode
        opcode = instruction.opcode
        second = raw[cursor]
        next_rip = rip + cursor + 1
        if instruction.prefixes or instruction.rex:
            return None

        if opcode == 0xD9 and second == 0xD0:
            return self._proposal(instruction, name="FNOP", next_rip=next_rip, x87_effect="nop")
        if opcode == 0xD9 and second in (0xE8, 0xEE):
            is_zero = second == 0xEE
            x87_state = _x87_push_state(state, X87_ZERO if is_zero else X87_ONE, zero=is_zero)
            name = "FLDZ" if is_zero else "FLD1"
            if isinstance(x87_state, StopReason):
                return self._proposal(
                    instruction,
                    name=name,
                    kind="exception",
                    exception="x87_stack_overflow",
                    x87_effect="push_fault",
                    stop_reason=x87_state,
                )
            return self._proposal(
                instruction,
                name=name,
                next_rip=next_rip,
                x87_state=x87_state,
                x87_effect="push_constant",
            )
        if opcode == 0xDD and 0xD8 <= second <= 0xDF:
            logical_index = second - 0xD8
            x87_state = _x87_fstp_state(state, logical_index)
            if isinstance(x87_state, StopReason):
                return self._proposal(
                    instruction,
                    name="FSTP_ST",
                    kind="exception",
                    exception="x87_stack_underflow",
                    x87_effect="pop_fault",
                    stop_reason=x87_state,
                )
            return self._proposal(
                instruction,
                name="FSTP_ST",
                next_rip=next_rip,
                x87_state=x87_state,
                x87_effect="store_and_pop",
            )
        return None

    def propose(self, runtime: object, instruction: DecodedInstruction) -> TransitionProposal:
        state = runtime.state
        memory = runtime.memory
        rip = instruction.rip
        raw = instruction.raw
        rex = instruction.rex
        opcode = instruction.opcode
        cursor = instruction.cursor_after_opcode
        width = instruction.width

        if opcode == 0x90:
            return self._proposal(instruction, name="NOP", next_rip=rip + cursor)
        if opcode == 0xCC:
            return self._proposal(
                instruction,
                name="INT3",
                next_rip=rip + cursor,
                kind="breakpoint",
                exception="breakpoint",
                stop_reason=StopReason("breakpoint", signal=SIGNAL_TRAP, detail="int3", address=rip),
            )
        if opcode == 0x0F and raw[cursor] == 0x05:
            return self._proposal(instruction, name="SYSCALL", next_rip=rip + cursor + 1, kind="syscall")
        if opcode == 0x0F:
            simd = self._stage4_simd_proposal(runtime, instruction)
            if simd is not None:
                return simd
        if opcode in (0xD9, 0xDD):
            x87 = self._stage4_x87_proposal(runtime, instruction)
            if x87 is not None:
                return x87
        if opcode == 0x55 and rex == 0:
            rsp = (state.gpr["RSP"] - 8) & MASK64
            return self._proposal(
                instruction,
                name="PUSH_RBP",
                next_rip=rip + cursor,
                register_writes={"RSP": rsp},
                register_write_masks={"RSP": MASK64},
                memory_writes=(MemoryWrite(rsp, int(state.gpr["RBP"] & MASK64).to_bytes(8, "little")),),
                stack_event="push",
            )
        if opcode == 0x5D and rex == 0:
            rsp = state.gpr["RSP"]
            value = memory.read_int(rsp, 8)
            return self._proposal(
                instruction,
                name="POP_RBP",
                next_rip=rip + cursor,
                register_writes={"RBP": value & MASK64, "RSP": (rsp + 8) & MASK64},
                register_write_masks={"RBP": MASK64, "RSP": MASK64},
                memory_reads=(_memory_read_event(rsp, 8, purpose="stack_pop"),),
                stack_event="pop",
            )
        if opcode == 0xC9:
            rsp = state.gpr["RBP"]
            value = memory.read_int(rsp, 8)
            return self._proposal(
                instruction,
                name="LEAVE",
                next_rip=rip + cursor,
                register_writes={"RBP": value & MASK64, "RSP": (rsp + 8) & MASK64},
                register_write_masks={"RBP": MASK64, "RSP": MASK64},
                memory_reads=(_memory_read_event(rsp, 8, purpose="leave_pop"),),
                stack_event="leave",
            )
        if opcode == 0xC3:
            rsp = state.gpr["RSP"]
            value = memory.read_int(rsp, 8)
            return self._proposal(
                instruction,
                name="RET",
                next_rip=value & MASK64,
                register_writes={"RSP": (rsp + 8) & MASK64},
                register_write_masks={"RSP": MASK64},
                memory_reads=(_memory_read_event(rsp, 8, purpose="return_target"),),
                stack_event="ret_pop",
            )
        if opcode == 0xE8:
            rel = _sign_extend(_read_le(raw[cursor:cursor + 4]), 32)
            next_rip = rip + cursor + 4
            rsp = (state.gpr["RSP"] - 8) & MASK64
            return self._proposal(
                instruction,
                name="CALL_REL32",
                next_rip=(next_rip + rel) & MASK64,
                register_writes={"RSP": rsp},
                register_write_masks={"RSP": MASK64},
                memory_writes=(MemoryWrite(rsp, int(next_rip & MASK64).to_bytes(8, "little")),),
                stack_event="call_push",
            )
        if opcode == 0xE9:
            rel = _sign_extend(_read_le(raw[cursor:cursor + 4]), 32)
            next_rip = rip + cursor + 4
            return self._proposal(instruction, name="JMP_REL32", next_rip=(next_rip + rel) & MASK64)
        if opcode == 0xEB:
            rel = _sign_extend(raw[cursor], 8)
            next_rip = rip + cursor + 1
            return self._proposal(instruction, name="JMP_REL8", next_rip=(next_rip + rel) & MASK64)
        if 0xB8 <= opcode <= 0xBF:
            reg = (opcode - 0xB8) | ((rex & 1) << 3)
            if width == 64 and rex & 0x08:
                imm = _read_le(raw[cursor:cursor + 8])
                cursor += 8
            else:
                imm = _read_le(raw[cursor:cursor + 4])
                cursor += 4
                width = 32
            name = _reg_name(reg)
            value = _final_reg_value(state.gpr[name], imm, width)
            return self._proposal(
                instruction,
                name="MOV_IMM_REG",
                next_rip=rip + cursor,
                register_writes={name: value},
                register_write_masks={name: _write_mask(width)},
            )
        if opcode in (0x89, 0x8B, 0x8D, 0x31):
            modrm = raw[cursor]
            mod, reg, rm_or_address, end = _decode_modrm_address(state, memory, rip + cursor + 1, modrm, rex)
            reg_width = width
            writes: dict[str, int] = {}
            write_masks: dict[str, int] = {}
            memory_reads: tuple[dict[str, Any], ...] = ()
            memory_writes: tuple[MemoryWrite, ...] = ()
            rflags = None
            flag_write_mask = None
            scalar_effect = None
            if opcode == 0x8D:
                dst = _reg_name(reg)
                writes[dst] = _final_reg_value(state.gpr[dst], rm_or_address, 64)
                write_masks[dst] = MASK64
                name = "LEA"
            elif opcode == 0x89:
                value = _read_reg(state, reg, reg_width)
                if mod == 3:
                    dst = _reg_name(rm_or_address)
                    writes[dst] = _final_reg_value(state.gpr[dst], value, reg_width)
                    write_masks[dst] = _write_mask(reg_width)
                else:
                    memory_writes = (MemoryWrite(rm_or_address, value.to_bytes(reg_width // 8, "little")),)
                name = "MOV_RM_REG"
            elif opcode == 0x8B:
                if mod == 3:
                    value = _read_reg(state, rm_or_address, reg_width)
                else:
                    value = memory.read_int(rm_or_address, reg_width // 8)
                    memory_reads = (_memory_read_event(rm_or_address, reg_width // 8, purpose="operand_load"),)
                dst = _reg_name(reg)
                writes[dst] = _final_reg_value(state.gpr[dst], value, reg_width)
                write_masks[dst] = _write_mask(reg_width)
                name = "MOV_REG_RM"
            else:
                src = _read_reg(state, reg, 32)
                dst_value = _read_reg(state, rm_or_address, 32) if mod == 3 else memory.read_int(rm_or_address, 4)
                value = (dst_value ^ src) & 0xFFFFFFFF
                if mod == 3:
                    dst = _reg_name(rm_or_address)
                    writes[dst] = _final_reg_value(state.gpr[dst], value, 32)
                    write_masks[dst] = _write_mask(32)
                else:
                    memory_reads = (_memory_read_event(rm_or_address, 4, purpose="read_modify_write"),)
                    memory_writes = (MemoryWrite(rm_or_address, value.to_bytes(4, "little")),)
                rflags = _flagged(state.rflags & ~((1 << 0) | (1 << 11)), value, 32)
                flag_write_mask = FLAG_EFFECT_MASK
                scalar_effect = "xor"
                name = "XOR_RM_REG"
            return self._proposal(
                instruction,
                name=name,
                next_rip=end,
                register_writes=writes,
                register_write_masks=write_masks,
                memory_reads=memory_reads,
                memory_writes=memory_writes,
                rflags=rflags,
                flag_write_mask=flag_write_mask,
                scalar_effect=scalar_effect,
            )
        if opcode in (0x83, 0xC7):
            modrm = raw[cursor]
            op = (modrm >> 3) & 0x7
            mod, _reg, rm_or_address, end = _decode_modrm_address(state, memory, rip + cursor + 1, modrm, rex)
            if opcode == 0x83 and op in (0, 5):
                imm = _sign_extend(memory.read(end, 1, access="execute", record=False)[0], 8)
                end += 1
                old = _read_reg(state, rm_or_address, width) if mod == 3 else memory.read_int(rm_or_address, width // 8)
                value = (old + imm) & MASK64 if op == 0 else (old - imm) & MASK64
                writes: dict[str, int] = {}
                write_masks: dict[str, int] = {}
                memory_reads: tuple[dict[str, Any], ...] = ()
                memory_writes: tuple[MemoryWrite, ...] = ()
                if mod == 3:
                    dst = _reg_name(rm_or_address)
                    writes[dst] = _final_reg_value(state.gpr[dst], value, width)
                    write_masks[dst] = _write_mask(width)
                else:
                    memory_reads = (_memory_read_event(rm_or_address, width // 8, purpose="read_modify_write"),)
                    memory_writes = (MemoryWrite(rm_or_address, value.to_bytes(width // 8, "little")),)
                return self._proposal(
                    instruction,
                    name="ADD_IMM_RM" if op == 0 else "SUB_IMM_RM",
                    next_rip=end,
                    register_writes=writes,
                    register_write_masks=write_masks,
                    memory_reads=memory_reads,
                    memory_writes=memory_writes,
                    rflags=_flagged(state.rflags, value, width),
                    flag_write_mask=FLAG_EFFECT_MASK,
                    scalar_effect="add" if op == 0 else "sub",
                )
            if opcode == 0xC7 and op == 0:
                imm = _read_le(memory.read(end, 4, access="execute", record=False))
                end += 4
                writes = {}
                write_masks = {}
                memory_writes = ()
                if mod == 3:
                    dst = _reg_name(rm_or_address)
                    writes[dst] = _final_reg_value(state.gpr[dst], imm, width)
                    write_masks[dst] = _write_mask(width)
                else:
                    memory_writes = (MemoryWrite(rm_or_address, imm.to_bytes(width // 8, "little")),)
                return self._proposal(
                    instruction,
                    name="MOV_IMM_RM",
                    next_rip=end,
                    register_writes=writes,
                    register_write_masks=write_masks,
                    memory_writes=memory_writes,
                )
        raise UnsupportedInstruction(rip, raw[: min(len(raw), cursor + 6)].hex())


@dataclass
class LearnedManifestTransitionModel:
    """Manifest-backed learned runtime adapter.

    This loader is intentionally fail-closed. It lets the runtime load and audit
    learned artifacts now, but refuses execution until a future adapter provides
    real typed transition heads for decoded runtime instructions.
    """

    manifest_data: dict[str, Any]
    profile: str = "long64-ring3"
    promotion_ledger: dict[str, Any] | None = None

    @classmethod
    def from_path(
        cls,
        path: Path,
        *,
        profile: str = "long64-ring3",
        promotion_ledger: dict[str, Any] | None = None,
    ) -> "LearnedManifestTransitionModel":
        path = Path(path)
        artifact_hash = _file_sha256(path)
        payload: dict[str, Any]
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".pt":
            try:
                import torch
                checkpoint = torch.load(path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - exact torch error text varies
                raise ValueError(f"could not inspect learned checkpoint {path}: {exc}") from exc
            payload = {
                "model_id": path.stem,
                "model_kind": "torch_checkpoint_transition_model",
                "checkpoint_arch": checkpoint.get("arch") if isinstance(checkpoint, dict) else None,
                "checkpoint_word_bits": checkpoint.get("word_bits") if isinstance(checkpoint, dict) else None,
            }
        else:
            raise ValueError(f"unsupported learned model artifact type: {path.suffix or '<none>'}")
        manifest = dict(payload.get("model_manifest") or payload)
        manifest.update(
            {
                "model_id": manifest.get("model_id") or path.stem,
                "model_kind": manifest.get("model_kind") or "learned_manifest_transition_model",
                "runtime_mode": "learned",
                "target_profile": manifest.get("target_profile") or profile,
                "trainable_weights_loaded": bool(manifest.get("trainable_weights_loaded", True)),
                "runtime_execution_head_loaded": bool(manifest.get("runtime_execution_head_loaded", False)),
                "artifact_path": str(path),
                "model_artifact_hash": artifact_hash,
                "commit_contract": "typed exact architectural state transition",
                "primary_semantics_path": "learned typed transition proposal",
            }
        )
        if promotion_ledger:
            promoted = sorted(_promoted_forms_from_ledger(promotion_ledger, profile))
            manifest["promotion_ledger_loaded"] = True
            manifest["promotion_profile"] = profile
            manifest["promoted_forms"] = promoted
        else:
            manifest.setdefault("promoted_forms", [])
        return cls(manifest_data=manifest, profile=profile, promotion_ledger=promotion_ledger)

    @property
    def manifest(self) -> dict[str, Any]:
        return dict(self.manifest_data)

    def propose(self, runtime: object, instruction: DecodedInstruction) -> TransitionProposal:
        del runtime
        name = _decoded_identity(instruction)
        promoted = set(str(item) for item in self.manifest_data.get("promoted_forms") or [])
        execution_head_loaded = bool(self.manifest_data.get("runtime_execution_head_loaded"))
        if promoted and name not in promoted:
            detail = f"learned runtime model has no promotion for decoded form {name}"
        elif not execution_head_loaded:
            detail = "learned runtime model loaded, but no runtime execution head adapter is available"
        else:
            detail = f"learned runtime execution head for {name} is not implemented"
        return TransitionProposal(
            kind="unsupported",
            rip=instruction.rip,
            instruction_hex=instruction.opcode_hex,
            instruction_name=name,
            stage=_runtime_stage_for_name(name),
            effect_axes=_runtime_effect_axes(name),
            profile=self.profile,
            iclass=name,
            iform=name,
            stop_reason=StopReason("unsupported_runtime_feature", detail=detail, address=instruction.rip),
            spf_manifest=self.manifest,
            model_artifact_hash=self.manifest_data.get("model_artifact_hash"),
            family_provenance="learned_manifest_runtime_adapter",
            confidence=1.0,
        )


def load_transition_model(
    model_path: str | None,
    *,
    learned_only: bool = False,
    profile: str = "long64-ring3",
    promotion_ledger_path: str | None = None,
) -> Any:
    promotion_ledger = None
    if promotion_ledger_path:
        promotion_ledger = json.loads(Path(promotion_ledger_path).read_text(encoding="utf-8"))
    if model_path:
        from .neural_runtime import maybe_load_neural_transition_model

        neural = maybe_load_neural_transition_model(
            Path(model_path),
            profile=profile,
            promotion_ledger=promotion_ledger,
        )
        if neural is not None:
            return neural
        return LearnedManifestTransitionModel.from_path(
            Path(model_path),
            profile=profile,
            promotion_ledger=promotion_ledger,
        )
    if learned_only:
        raise ValueError("learned runtime mode requires --model")
    return BootstrapSPFTransitionModel(profile=profile)
