"""PS/PL backend routing for the PAN x86 SPF runtime.

The programmable-logic path is intentionally authority-free: PL receives a
typed packet, returns typed deltas, and the PS runtime validates and commits the
result through the existing TransitionProposal contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import struct
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Optional

from .decode import DecodedInstruction, decode_current
from .transitions import MASK128, MemoryWrite, TransitionProposal, validate_transition_proposal

from pan_x86_isa.memory import MemoryFault  # noqa: E402
from pan_x86_isa.runtime import MASK64, StopReason, UnsupportedInstruction  # noqa: E402


PL_PACKET_MAGIC = "PANPLV1"
PL_BINARY_MAGIC = b"PANPLB1\0"
PL_BINARY_OUTPUT_MAGIC = b"PANPLO1\0"
PL_PACKET_VERSION = 1
PL_BACKEND_MODES = ("auto", "ps", "pl-sim", "pl-xrt")
PL_SIMD_BLOCK_IFORM = "PLTransitionPacketV1"
PL_SIMD_BLOCK_NAME = "PL_SIMD_BLOCK"
PL_BINARY_HEADER_FORMAT = "<8sHHIQQIIIIII"
PL_BINARY_OP_FORMAT = "<QQHBBBBBHH16s"
PL_BINARY_MEMORY_WINDOW_FORMAT = "<QII16s"
PL_BINARY_OUTPUT_HEADER_FORMAT = "<8sHHIQIIII"
PL_BINARY_OUTPUT_WRITE_FORMAT = "<QI16s"
PL_BINARY_XMM_COUNT = 16
PL_BINARY_M128_SIZE = 16
PL_MEMORY_ACCESS_READ = 1
PL_MEMORY_ACCESS_WRITE = 2
PL_OUTPUT_STATUS_OK = 0
PL_OUTPUT_STATUS_FAULT = 1
PL_OUTPUT_STATUS_UNSUPPORTED = 2
PL_OP_IDS = {
    "MOVDQU_XMM_M128": 1,
    "MOVDQU_M128_XMM": 2,
    "PXOR_XMM_XMM": 3,
    "PXOR_XMM_M128": 4,
    "PADDQ_XMM_XMM": 5,
    "PADDQ_XMM_M128": 6,
}
PL_OP_NAMES_BY_ID = {value: key for key, value in PL_OP_IDS.items()}
SUPPORTED_PL_SIMD_FORMS = {
    "MOVDQU_XMM_M128",
    "MOVDQU_M128_XMM",
    "PXOR_XMM_XMM",
    "PXOR_XMM_M128",
    "PADDQ_XMM_XMM",
    "PADDQ_XMM_M128",
}
PL_BINARY_HEADER_SIZE = struct.calcsize(PL_BINARY_HEADER_FORMAT)
PL_BINARY_OP_SIZE = struct.calcsize(PL_BINARY_OP_FORMAT)
PL_BINARY_MEMORY_WINDOW_SIZE = struct.calcsize(PL_BINARY_MEMORY_WINDOW_FORMAT)
PL_BINARY_OUTPUT_HEADER_SIZE = struct.calcsize(PL_BINARY_OUTPUT_HEADER_FORMAT)
PL_BINARY_OUTPUT_WRITE_SIZE = struct.calcsize(PL_BINARY_OUTPUT_WRITE_FORMAT)


def _parse_int(value: Any) -> int:
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return int(value, 16) if value.lower().startswith("0x") else int(value, 10)
    raise TypeError(f"cannot parse integer field from {value!r}")


@dataclass(frozen=True)
class PLOpV1:
    rip: int
    next_rip: int
    instruction_hex: str
    instruction_name: str
    iform: str
    opcode_id: int = 0
    dst_xmm: int = 0xFF
    src_xmm: int = 0xFF
    read_window_index: int = 0xFFFF
    write_window_index: int = 0xFFFF

    def to_json(self) -> dict[str, Any]:
        return {
            "rip": hex(int(self.rip) & MASK64),
            "next_rip": hex(int(self.next_rip) & MASK64),
            "instruction_hex": self.instruction_hex,
            "instruction_name": self.instruction_name,
            "iform": self.iform,
            "opcode_id": int(self.opcode_id),
            "dst_xmm": int(self.dst_xmm),
            "src_xmm": int(self.src_xmm),
            "read_window_index": int(self.read_window_index),
            "write_window_index": int(self.write_window_index),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PLOpV1":
        return cls(
            rip=_parse_int(payload["rip"]),
            next_rip=_parse_int(payload["next_rip"]),
            instruction_hex=str(payload["instruction_hex"]),
            instruction_name=str(payload["instruction_name"]),
            iform=str(payload.get("iform") or payload["instruction_name"]),
            opcode_id=int(payload.get("opcode_id", 0)),
            dst_xmm=int(payload.get("dst_xmm", 0xFF)),
            src_xmm=int(payload.get("src_xmm", 0xFF)),
            read_window_index=int(payload.get("read_window_index", 0xFFFF)),
            write_window_index=int(payload.get("write_window_index", 0xFFFF)),
        )

    def to_binary(self) -> bytes:
        raw = bytes.fromhex(self.instruction_hex)
        if len(raw) > 16:
            raise ValueError("PL binary op raw instruction field is limited to 16 bytes")
        return struct.pack(
            PL_BINARY_OP_FORMAT,
            int(self.rip) & MASK64,
            int(self.next_rip) & MASK64,
            int(self.opcode_id) & 0xFFFF,
            len(raw) & 0xFF,
            int(self.dst_xmm) & 0xFF,
            int(self.src_xmm) & 0xFF,
            0,
            0,
            int(self.read_window_index) & 0xFFFF,
            int(self.write_window_index) & 0xFFFF,
            raw + b"\x00" * (16 - len(raw)),
        )

    @classmethod
    def from_binary(cls, data: bytes, offset: int) -> "PLOpV1":
        (
            rip,
            next_rip,
            opcode_id,
            raw_len,
            dst_xmm,
            src_xmm,
            _flags,
            _reserved,
            read_window_index,
            write_window_index,
            raw16,
        ) = struct.unpack_from(PL_BINARY_OP_FORMAT, data, offset)
        instruction_name = PL_OP_NAMES_BY_ID.get(opcode_id, f"PL_OP_{opcode_id}")
        return cls(
            rip=rip,
            next_rip=next_rip,
            instruction_hex=raw16[:raw_len].hex(),
            instruction_name=instruction_name,
            iform=instruction_name,
            opcode_id=opcode_id,
            dst_xmm=dst_xmm,
            src_xmm=src_xmm,
            read_window_index=read_window_index,
            write_window_index=write_window_index,
        )


@dataclass(frozen=True)
class PLMemoryWindowV1:
    address: int
    size: int
    access: str
    data: bytes = b""

    def to_json(self) -> dict[str, Any]:
        return {
            "address": hex(int(self.address) & MASK64),
            "size": int(self.size),
            "access": self.access,
            "data_hex": self.data.hex(),
        }

    def to_binary(self) -> bytes:
        if len(self.data) > PL_BINARY_M128_SIZE:
            raise ValueError("PL binary memory window data is limited to 16 bytes in v1")
        access_flags = 0
        if "r" in self.access or self.access == "read":
            access_flags |= PL_MEMORY_ACCESS_READ
        if "w" in self.access or self.access == "write":
            access_flags |= PL_MEMORY_ACCESS_WRITE
        return struct.pack(
            PL_BINARY_MEMORY_WINDOW_FORMAT,
            int(self.address) & MASK64,
            int(self.size),
            access_flags,
            self.data + b"\x00" * (PL_BINARY_M128_SIZE - len(self.data)),
        )

    @classmethod
    def from_binary(cls, data: bytes, offset: int) -> "PLMemoryWindowV1":
        address, size, access_flags, raw = struct.unpack_from(PL_BINARY_MEMORY_WINDOW_FORMAT, data, offset)
        access = "rw" if access_flags == (PL_MEMORY_ACCESS_READ | PL_MEMORY_ACCESS_WRITE) else "write" if access_flags & PL_MEMORY_ACCESS_WRITE else "read"
        return cls(address=address, size=size, access=access, data=raw[:size])

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PLMemoryWindowV1":
        return cls(
            address=_parse_int(payload["address"]),
            size=int(payload["size"]),
            access=str(payload["access"]),
            data=bytes.fromhex(str(payload.get("data_hex", ""))),
        )


@dataclass(frozen=True)
class PLMemoryWriteV1:
    address: int
    data: bytes

    def to_json(self) -> dict[str, Any]:
        return {
            "address": hex(int(self.address) & MASK64),
            "size": len(self.data),
            "data_hex": self.data.hex(),
        }

    def to_binary(self) -> bytes:
        if len(self.data) > PL_BINARY_M128_SIZE:
            raise ValueError("PL binary memory writes are limited to 16 bytes in v1")
        return struct.pack(
            PL_BINARY_OUTPUT_WRITE_FORMAT,
            int(self.address) & MASK64,
            len(self.data),
            self.data + b"\x00" * (PL_BINARY_M128_SIZE - len(self.data)),
        )

    @classmethod
    def from_binary(cls, data: bytes, offset: int) -> "PLMemoryWriteV1":
        address, size, raw = struct.unpack_from(PL_BINARY_OUTPUT_WRITE_FORMAT, data, offset)
        return cls(address=address, data=raw[:size])

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PLMemoryWriteV1":
        return cls(address=_parse_int(payload["address"]), data=bytes.fromhex(str(payload["data_hex"])))


@dataclass(frozen=True)
class PLTransitionOutputV1:
    status: str
    final_rip: int
    op_count: int
    xmm_writes: dict[str, int] = field(default_factory=dict)
    xmm_write_masks: dict[str, int] = field(default_factory=dict)
    memory_writes: tuple[PLMemoryWriteV1, ...] = ()
    fault_code: Optional[str] = None
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "final_rip": hex(int(self.final_rip) & MASK64),
            "op_count": int(self.op_count),
            "xmm_writes": {
                name: f"0x{int(value) & MASK128:032x}" for name, value in sorted(self.xmm_writes.items())
            },
            "xmm_write_masks": {
                name: f"0x{int(value) & MASK128:032x}" for name, value in sorted(self.xmm_write_masks.items())
            },
            "memory_writes": [write.to_json() for write in self.memory_writes],
            "fault_code": self.fault_code,
            "detail": self.detail,
        }

    def encode_binary(self) -> bytes:
        status_code = {
            "ok": PL_OUTPUT_STATUS_OK,
            "fault": PL_OUTPUT_STATUS_FAULT,
            "unsupported": PL_OUTPUT_STATUS_UNSUPPORTED,
        }.get(self.status, PL_OUTPUT_STATUS_UNSUPPORTED)
        xmm_values = bytearray()
        xmm_write_mask = 0
        for index in range(PL_BINARY_XMM_COUNT):
            name = f"XMM{index}"
            value = int(self.xmm_writes.get(name, 0)) & MASK128
            if name in self.xmm_writes:
                xmm_write_mask |= 1 << index
            xmm_values.extend(value.to_bytes(16, "little"))
        writes = b"".join(write.to_binary() for write in self.memory_writes)
        write_offset = PL_BINARY_OUTPUT_HEADER_SIZE + len(xmm_values)
        total_size = write_offset + len(writes)
        return (
            struct.pack(
                PL_BINARY_OUTPUT_HEADER_FORMAT,
                PL_BINARY_OUTPUT_MAGIC,
                PL_PACKET_VERSION,
                status_code,
                int(self.op_count),
                int(self.final_rip) & MASK64,
                xmm_write_mask,
                len(self.memory_writes),
                write_offset,
                total_size,
            )
            + bytes(xmm_values)
            + writes
        )

    @classmethod
    def decode_binary(cls, data: bytes) -> "PLTransitionOutputV1":
        if len(data) < PL_BINARY_OUTPUT_HEADER_SIZE:
            raise ValueError("PL binary output is shorter than the header")
        magic, version, status_code, op_count, final_rip, xmm_write_mask, write_count, write_offset, total_size = struct.unpack_from(
            PL_BINARY_OUTPUT_HEADER_FORMAT,
            data,
            0,
        )
        if magic != PL_BINARY_OUTPUT_MAGIC or version != PL_PACKET_VERSION:
            raise ValueError(f"unsupported PL binary output header: magic={magic!r} version={version}")
        if len(data) < total_size:
            raise ValueError("PL binary output is truncated")
        xmm_offset = PL_BINARY_OUTPUT_HEADER_SIZE
        xmm_writes: dict[str, int] = {}
        xmm_masks: dict[str, int] = {}
        for index in range(PL_BINARY_XMM_COUNT):
            if xmm_write_mask & (1 << index):
                start = xmm_offset + index * 16
                value = int.from_bytes(data[start : start + 16], "little")
                name = f"XMM{index}"
                xmm_writes[name] = value & MASK128
                xmm_masks[name] = MASK128
        writes = tuple(
            PLMemoryWriteV1.from_binary(data, write_offset + index * PL_BINARY_OUTPUT_WRITE_SIZE)
            for index in range(write_count)
        )
        status = "ok" if status_code == PL_OUTPUT_STATUS_OK else "fault" if status_code == PL_OUTPUT_STATUS_FAULT else "unsupported"
        return cls(
            status=status,
            final_rip=final_rip,
            op_count=op_count,
            xmm_writes=xmm_writes,
            xmm_write_masks=xmm_masks,
            memory_writes=writes,
            fault_code=None if status == "ok" else str(status_code),
        )

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PLTransitionOutputV1":
        return cls(
            status=str(payload["status"]),
            final_rip=_parse_int(payload["final_rip"]),
            op_count=int(payload["op_count"]),
            xmm_writes={str(name): _parse_int(value) for name, value in dict(payload.get("xmm_writes") or {}).items()},
            xmm_write_masks={
                str(name): _parse_int(value) for name, value in dict(payload.get("xmm_write_masks") or {}).items()
            },
            memory_writes=tuple(PLMemoryWriteV1.from_json(item) for item in payload.get("memory_writes") or ()),
            fault_code=None if payload.get("fault_code") is None else str(payload["fault_code"]),
            detail=str(payload.get("detail") or ""),
        )


@dataclass(frozen=True)
class PLTransitionPacketV1:
    initial_rip: int
    expected_final_rip: int
    ops: tuple[PLOpV1, ...]
    input_xmm: dict[str, int]
    memory_windows: tuple[PLMemoryWindowV1, ...] = ()
    flags: int = 0
    output: Optional[PLTransitionOutputV1] = None
    magic: str = PL_PACKET_MAGIC
    version: int = PL_PACKET_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "magic": self.magic,
            "version": int(self.version),
            "flags": int(self.flags),
            "instruction_count": len(self.ops),
            "initial_rip": hex(int(self.initial_rip) & MASK64),
            "expected_final_rip": hex(int(self.expected_final_rip) & MASK64),
            "ops": [op.to_json() for op in self.ops],
            "input_xmm": {
                name: f"0x{int(value) & MASK128:032x}" for name, value in sorted(self.input_xmm.items())
            },
            "memory_windows": [window.to_json() for window in self.memory_windows],
            "output": None if self.output is None else self.output.to_json(),
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "PLTransitionPacketV1":
        magic = str(payload.get("magic") or "")
        version = int(payload.get("version", 0))
        if magic != PL_PACKET_MAGIC or version != PL_PACKET_VERSION:
            raise ValueError(f"unsupported PL packet header: magic={magic!r} version={version}")
        ops = tuple(PLOpV1.from_json(item) for item in payload.get("ops") or ())
        instruction_count = int(payload.get("instruction_count", len(ops)))
        if instruction_count != len(ops):
            raise ValueError("PL packet instruction_count does not match ops length")
        return cls(
            initial_rip=_parse_int(payload["initial_rip"]),
            expected_final_rip=_parse_int(payload["expected_final_rip"]),
            ops=ops,
            input_xmm={str(name): _parse_int(value) for name, value in dict(payload.get("input_xmm") or {}).items()},
            memory_windows=tuple(PLMemoryWindowV1.from_json(item) for item in payload.get("memory_windows") or ()),
            flags=int(payload.get("flags", 0)),
            output=None if payload.get("output") is None else PLTransitionOutputV1.from_json(payload["output"]),
        )

    def encode(self) -> bytes:
        return json.dumps(self.to_json(), sort_keys=True, separators=(",", ":")).encode("utf-8")

    @classmethod
    def decode(cls, data: bytes) -> "PLTransitionPacketV1":
        return cls.from_json(json.loads(data.decode("utf-8")))

    def encode_binary(self) -> bytes:
        ops = b"".join(op.to_binary() for op in self.ops)
        xmm_values = bytearray()
        for index in range(PL_BINARY_XMM_COUNT):
            value = int(self.input_xmm.get(f"XMM{index}", 0)) & MASK128
            xmm_values.extend(value.to_bytes(16, "little"))
        windows = b"".join(window.to_binary() for window in self.memory_windows)
        op_offset = PL_BINARY_HEADER_SIZE
        xmm_offset = op_offset + len(ops)
        window_offset = xmm_offset + len(xmm_values)
        total_size = window_offset + len(windows)
        return (
            struct.pack(
                PL_BINARY_HEADER_FORMAT,
                PL_BINARY_MAGIC,
                int(self.version),
                int(self.flags),
                len(self.ops),
                int(self.initial_rip) & MASK64,
                int(self.expected_final_rip) & MASK64,
                PL_BINARY_XMM_COUNT,
                len(self.memory_windows),
                op_offset,
                xmm_offset,
                window_offset,
                total_size,
            )
            + ops
            + bytes(xmm_values)
            + windows
        )

    @classmethod
    def decode_binary(cls, data: bytes) -> "PLTransitionPacketV1":
        if len(data) < PL_BINARY_HEADER_SIZE:
            raise ValueError("PL binary packet is shorter than the header")
        (
            magic,
            version,
            flags,
            instruction_count,
            initial_rip,
            expected_final_rip,
            xmm_count,
            window_count,
            op_offset,
            xmm_offset,
            window_offset,
            total_size,
        ) = struct.unpack_from(PL_BINARY_HEADER_FORMAT, data, 0)
        if magic != PL_BINARY_MAGIC or version != PL_PACKET_VERSION:
            raise ValueError(f"unsupported PL binary packet header: magic={magic!r} version={version}")
        if xmm_count != PL_BINARY_XMM_COUNT:
            raise ValueError(f"unsupported PL binary XMM count: {xmm_count}")
        if len(data) < total_size:
            raise ValueError("PL binary packet is truncated")
        ops = tuple(
            PLOpV1.from_binary(data, op_offset + index * PL_BINARY_OP_SIZE)
            for index in range(instruction_count)
        )
        input_xmm = {
            f"XMM{index}": int.from_bytes(data[xmm_offset + index * 16 : xmm_offset + (index + 1) * 16], "little")
            for index in range(PL_BINARY_XMM_COUNT)
        }
        windows = tuple(
            PLMemoryWindowV1.from_binary(data, window_offset + index * PL_BINARY_MEMORY_WINDOW_SIZE)
            for index in range(window_count)
        )
        return cls(
            initial_rip=initial_rip,
            expected_final_rip=expected_final_rip,
            ops=ops,
            input_xmm=input_xmm,
            memory_windows=windows,
            flags=flags,
        )


@dataclass(frozen=True)
class RouteDecision:
    rip: int
    requested_backend: str
    selected_backend: str
    reason: str
    eligible: bool
    instruction_count: int = 1
    instruction_names: tuple[str, ...] = ()
    packet_magic: Optional[str] = None
    transfer_to_pl_bytes: int = 0
    transfer_from_pl_bytes: int = 0
    elapsed_seconds: float = 0.0
    template_key: Optional[str] = None
    template_hit: bool = False
    template_seen_count: int = 0
    repeat_group_size: int = 0
    repeat_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "rip": hex(int(self.rip) & MASK64),
            "requested_backend": self.requested_backend,
            "selected_backend": self.selected_backend,
            "reason": self.reason,
            "eligible": bool(self.eligible),
            "instruction_count": int(self.instruction_count),
            "instruction_names": list(self.instruction_names),
            "packet_magic": self.packet_magic,
            "transfer_to_pl_bytes": int(self.transfer_to_pl_bytes),
            "transfer_from_pl_bytes": int(self.transfer_from_pl_bytes),
            "elapsed_seconds": float(self.elapsed_seconds),
            "template_key": self.template_key,
            "template_hit": bool(self.template_hit),
            "template_seen_count": int(self.template_seen_count),
            "repeat_group_size": int(self.repeat_group_size),
            "repeat_count": int(self.repeat_count),
        }


@dataclass(frozen=True)
class RoutedProposal:
    proposal: TransitionProposal
    decision: RouteDecision
    instruction_count: int = 1


@dataclass(frozen=True)
class BlockCandidate:
    initial_rip: int
    expected_final_rip: int
    proposals: tuple[TransitionProposal, ...]
    packet: PLTransitionPacketV1
    output: PLTransitionOutputV1
    template_key: str = ""
    template_hit: bool = False
    template_seen_count: int = 1
    repeat_group_size: int = 0
    repeat_count: int = 0

    @property
    def instruction_count(self) -> int:
        return len(self.proposals)

    @property
    def instruction_names(self) -> tuple[str, ...]:
        return tuple(proposal.instruction_name for proposal in self.proposals)


def _proposal_confidence(proposals: Iterable[TransitionProposal]) -> float:
    values = [float(proposal.confidence) for proposal in proposals]
    return min(values) if values else 1.0


def _template_key(proposals: Iterable[TransitionProposal]) -> str:
    parts = []
    for proposal in proposals:
        try:
            op = _pl_op_from_proposal(proposal)
            parts.append(
                f"{op.opcode_id}:{op.dst_xmm}:{op.src_xmm}:"
                f"{int(op.read_window_index != 0xFFFF)}:{int(op.write_window_index != 0xFFFF)}"
            )
        except Exception:
            parts.append(proposal.instruction_name)
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"simd-v1:{digest}"


def _repeat_shape(names: tuple[str, ...]) -> tuple[int, int]:
    if not names:
        return 0, 0
    for group_size in range(1, (len(names) // 2) + 1):
        if len(names) % group_size:
            continue
        group = names[:group_size]
        repeats = len(names) // group_size
        if repeats > 1 and all(names[index : index + group_size] == group for index in range(0, len(names), group_size)):
            return group_size, repeats
    return len(names), 1


def _manifest_with_backend(runtime: object, backend_name: str, packet: PLTransitionPacketV1) -> dict[str, Any]:
    manifest = dict(getattr(getattr(runtime, "model", None), "manifest", {}) or {})
    manifest["ps_pl_router"] = {
        "backend": backend_name,
        "packet_magic": packet.magic,
        "packet_version": packet.version,
        "instruction_count": len(packet.ops),
        "commit_authority": "ps_validated_transition_proposal",
    }
    return manifest


def _apply_transition_to_shadow(shadow: object, proposal: TransitionProposal) -> None:
    if proposal.kind != "normal" or proposal.stop_reason is not None or proposal.next_rip is None:
        raise ValueError(f"cannot shadow-apply non-normal proposal {proposal.kind}:{proposal.instruction_name}")
    for write in proposal.memory_writes:
        shadow.memory.write(write.address, write.data, record=False)
    for name, value in proposal.register_writes.items():
        shadow.state.gpr[name] = int(value) & MASK64
    for name, value in proposal.xmm_writes.items():
        shadow.state.xmm[int(name[3:])] = int(value) & MASK128
    if proposal.rflags is not None:
        shadow.state.rflags = int(proposal.rflags) & MASK64
    if proposal.mxcsr is not None:
        shadow.state.mxcsr = int(proposal.mxcsr) & 0xFFFFFFFF
    if proposal.x87_state is not None:
        stack = proposal.x87_state["st"]
        shadow.state.x87 = [None if value is None else int(value) & ((1 << 80) - 1) for value in stack]
        shadow.state.x87_top = int(proposal.x87_state["top"]) & 0x7
        shadow.state.x87_tag = int(proposal.x87_state["tag"]) & 0xFFFF
        shadow.state.x87_status = int(proposal.x87_state["status"]) & 0xFFFF
        shadow.state.x87_control = int(proposal.x87_state["control"]) & 0xFFFF
    shadow.state.gpr["RIP"] = int(proposal.next_rip) & MASK64


def _has_repeated_group(names: tuple[str, ...], min_groups: int) -> bool:
    if min_groups <= 1:
        return bool(names)
    if len(names) < min_groups:
        return False
    for group_size in range(1, (len(names) // min_groups) + 1):
        suffix = names[-group_size:]
        if all(names[-group_size * (index + 1) : -group_size * index if index else None] == suffix for index in range(min_groups)):
            return True
    return False


def _memory_read_address(read: dict[str, Any]) -> tuple[int, int]:
    address = _parse_int(read["address"])
    return address, int(read["size"])


def _read_window(runtime: object, address: int, size: int, access: str) -> PLMemoryWindowV1:
    try:
        data = runtime.memory.read(address, size, record=False)
    except Exception:
        data = b""
    return PLMemoryWindowV1(address=address, size=size, access=access, data=data)


def _xmm_index_from_name(name: str) -> int:
    if not name.startswith("XMM"):
        raise ValueError(f"unsupported XMM register name {name!r}")
    index = int(name[3:])
    if not 0 <= index < PL_BINARY_XMM_COUNT:
        raise ValueError(f"XMM register index out of range: {name!r}")
    return index


def _first_xmm_write_index(proposal: TransitionProposal) -> int:
    if not proposal.xmm_writes:
        return 0xFF
    return _xmm_index_from_name(next(iter(sorted(proposal.xmm_writes))))


def _simd_modrm_fields(proposal: TransitionProposal) -> tuple[int, int, int]:
    raw = bytes.fromhex(proposal.instruction_hex)
    cursor = 0
    while cursor < len(raw) and raw[cursor] in (0x66, 0xF2, 0xF3):
        cursor += 1
    rex = 0
    if cursor < len(raw) and 0x40 <= raw[cursor] <= 0x4F:
        rex = raw[cursor]
        cursor += 1
    if cursor + 2 >= len(raw) or raw[cursor] != 0x0F:
        raise ValueError(f"cannot decode SIMD ModRM fields from {proposal.instruction_hex}")
    modrm = raw[cursor + 2]
    mod = (modrm >> 6) & 0x3
    reg = ((modrm >> 3) & 0x7) | (((rex >> 2) & 1) << 3)
    rm = (modrm & 0x7) | ((rex & 1) << 3)
    return mod, reg, rm


def _pl_op_from_proposal(
    proposal: TransitionProposal,
    *,
    read_window_index: int = 0xFFFF,
    write_window_index: int = 0xFFFF,
) -> PLOpV1:
    opcode_id = PL_OP_IDS[proposal.instruction_name]
    _mod, reg, rm = _simd_modrm_fields(proposal)
    dst_xmm = 0xFF
    src_xmm = 0xFF
    if proposal.instruction_name == "MOVDQU_XMM_M128":
        dst_xmm = _first_xmm_write_index(proposal)
    elif proposal.instruction_name == "MOVDQU_M128_XMM":
        src_xmm = reg
    elif proposal.instruction_name in {"PXOR_XMM_XMM", "PADDQ_XMM_XMM"}:
        dst_xmm = _first_xmm_write_index(proposal)
        src_xmm = rm
    elif proposal.instruction_name in {"PXOR_XMM_M128", "PADDQ_XMM_M128"}:
        dst_xmm = _first_xmm_write_index(proposal)
    return PLOpV1(
        rip=proposal.rip,
        next_rip=int(proposal.next_rip or 0),
        instruction_hex=proposal.instruction_hex,
        instruction_name=proposal.instruction_name,
        iform=proposal.iform or proposal.instruction_name,
        opcode_id=opcode_id,
        dst_xmm=dst_xmm,
        src_xmm=src_xmm,
        read_window_index=read_window_index,
        write_window_index=write_window_index,
    )


def _candidate_packet(runtime: object, proposals: tuple[TransitionProposal, ...], shadow: object) -> PLTransitionPacketV1:
    input_xmm = {f"XMM{i}": int(value) & MASK128 for i, value in enumerate(runtime.state.xmm)}
    windows: dict[tuple[int, int, str], PLMemoryWindowV1] = {}
    window_indices: dict[tuple[int, int, str], int] = {}
    op_windows: list[tuple[int, int]] = []
    for proposal in proposals:
        read_index = 0xFFFF
        write_index = 0xFFFF
        for read in proposal.memory_reads:
            address, size = _memory_read_address(read)
            key = (address, size, "read")
            if key not in windows:
                window_indices[key] = len(windows)
                windows[key] = _read_window(runtime, address, size, "read")
            read_index = window_indices[key]
        for write in proposal.memory_writes:
            key = (int(write.address), len(write.data), "write")
            if key not in windows:
                window_indices[key] = len(windows)
                windows[key] = _read_window(runtime, int(write.address), len(write.data), "write")
            write_index = window_indices[key]
        op_windows.append((read_index, write_index))
    ops = tuple(
        _pl_op_from_proposal(proposal, read_window_index=read_index, write_window_index=write_index)
        for proposal, (read_index, write_index) in zip(proposals, op_windows)
    )

    xmm_writes: dict[str, int] = {}
    xmm_masks: dict[str, int] = {}
    for index, initial in enumerate(runtime.state.xmm):
        final = int(shadow.state.xmm[index]) & MASK128
        if final != (int(initial) & MASK128):
            name = f"XMM{index}"
            xmm_writes[name] = final
            xmm_masks[name] = MASK128
    memory_writes = tuple(
        PLMemoryWriteV1(address=int(write.address), data=bytes(write.data))
        for proposal in proposals
        for write in proposal.memory_writes
    )
    output = PLTransitionOutputV1(
        status="ok",
        final_rip=int(shadow.state.gpr["RIP"]) & MASK64,
        op_count=len(proposals),
        xmm_writes=xmm_writes,
        xmm_write_masks=xmm_masks,
        memory_writes=memory_writes,
    )
    return PLTransitionPacketV1(
        initial_rip=int(proposals[0].rip) & MASK64,
        expected_final_rip=int(shadow.state.gpr["RIP"]) & MASK64,
        ops=ops,
        input_xmm=input_xmm,
        memory_windows=tuple(windows.values()),
        output=output,
    )


def _block_proposal_from_output(
    runtime: object,
    candidate: BlockCandidate,
    output: PLTransitionOutputV1,
    *,
    backend_name: str,
) -> TransitionProposal:
    detail = output.detail or output.fault_code or f"PL backend {backend_name} returned status {output.status}"
    if output.status != "ok":
        return TransitionProposal(
            kind="unsupported",
            rip=candidate.initial_rip,
            instruction_hex="".join(op.instruction_hex for op in candidate.packet.ops),
            instruction_name=PL_SIMD_BLOCK_NAME,
            stage="stage4_simd_x87",
            effect_axes={"stage": "stage4_simd_x87", "backend": backend_name, "failure": output.status},
            profile=getattr(runtime, "profile", "long64-ring3"),
            iclass=PL_SIMD_BLOCK_NAME,
            iform=PL_SIMD_BLOCK_IFORM,
            stop_reason=StopReason("unsupported_runtime_feature", detail=detail, address=candidate.initial_rip),
            spf_manifest=_manifest_with_backend(runtime, backend_name, candidate.packet),
            family_provenance=backend_name,
            confidence=_proposal_confidence(candidate.proposals),
        )
    memory_writes = tuple(MemoryWrite(write.address, write.data) for write in output.memory_writes)
    memory_reads = tuple(read for proposal in candidate.proposals for read in proposal.memory_reads)
    axes = {
        "stage": "stage4_simd_x87",
        "xmm_lane_effect": "pl_simd_block",
        "memory_event_kind": "mixed" if memory_reads and memory_writes else "store" if memory_writes else "read" if memory_reads else "none",
        "memory_read_count": len(memory_reads),
        "memory_write_count": len(memory_writes),
        "written_xmm_registers": sorted(output.xmm_writes),
        "x87_stack_effect": None,
        "exception_class": None,
        "backend": backend_name,
        "packet_magic": candidate.packet.magic,
        "packet_version": candidate.packet.version,
        "instruction_count": candidate.instruction_count,
        "instruction_names": list(candidate.instruction_names),
        "template_key": candidate.template_key,
        "template_hit": candidate.template_hit,
        "template_seen_count": candidate.template_seen_count,
        "repeat_group_size": candidate.repeat_group_size,
        "repeat_count": candidate.repeat_count,
    }
    proposal = TransitionProposal(
        kind="normal",
        rip=candidate.initial_rip,
        instruction_hex="".join(op.instruction_hex for op in candidate.packet.ops),
        instruction_name=PL_SIMD_BLOCK_NAME,
        stage="stage4_simd_x87",
        effect_axes=axes,
        profile=getattr(runtime, "profile", "long64-ring3"),
        iclass=PL_SIMD_BLOCK_NAME,
        iform=PL_SIMD_BLOCK_IFORM,
        operands=tuple(op.to_json() for op in candidate.packet.ops),
        widths={"packet_ops": candidate.instruction_count},
        next_rip=output.final_rip,
        xmm_writes=dict(output.xmm_writes),
        xmm_write_masks=dict(output.xmm_write_masks),
        memory_reads=memory_reads,
        memory_writes=memory_writes,
        spf_manifest=_manifest_with_backend(runtime, backend_name, candidate.packet),
        model_artifact_hash=getattr(getattr(runtime, "model", None), "manifest", {}).get("model_artifact_hash"),
        family_provenance=backend_name,
        confidence=_proposal_confidence(candidate.proposals),
    )
    if proposal.next_rip != candidate.expected_final_rip:
        return replace(
            proposal,
            kind="unsupported",
            next_rip=None,
            stop_reason=StopReason(
                "unsupported_runtime_feature",
                detail=(
                    f"PL final RIP mismatch: got 0x{int(proposal.next_rip or 0):x}, "
                    f"expected 0x{candidate.expected_final_rip:x}"
                ),
                address=candidate.initial_rip,
            ),
        )
    validation = validate_transition_proposal(
        proposal,
        expected_profile=getattr(runtime, "profile", "long64-ring3"),
        exact_confidence=float(proposal.spf_manifest.get("runtime_confidence_threshold", proposal.spf_manifest.get("min_confidence", 1.0))),
        require_learned=bool(proposal.spf_manifest.get("runtime_mode") == "learned"),
    )
    if validation is not None:
        return replace(proposal, kind="unsupported", next_rip=None, stop_reason=validation)
    return proposal


class PSBackend:
    name = "ps"

    def propose(self, runtime: object, instruction: Optional[DecodedInstruction] = None, *, reason: str = "ps_selected") -> RoutedProposal:
        instruction = instruction or decode_current(runtime)
        proposal = runtime.model.propose(runtime, instruction)
        decision = RouteDecision(
            rip=instruction.rip,
            requested_backend=getattr(runtime, "backend_mode", "ps"),
            selected_backend=self.name,
            reason=reason,
            eligible=False,
            instruction_count=1,
            instruction_names=(proposal.instruction_name,),
        )
        return RoutedProposal(proposal=proposal, decision=decision, instruction_count=1)


class PLSimBackend:
    name = "pl-sim"

    def propose(self, runtime: object, candidate: BlockCandidate) -> RoutedProposal:
        started = time.time()
        packet_bytes = candidate.packet.encode_binary()
        output = candidate.output
        output_bytes = output.encode_binary()
        proposal = _block_proposal_from_output(runtime, candidate, output, backend_name=self.name)
        elapsed = time.time() - started
        decision = RouteDecision(
            rip=candidate.initial_rip,
            requested_backend=getattr(runtime, "backend_mode", self.name),
            selected_backend=self.name,
            reason="eligible_simd_block",
            eligible=True,
            instruction_count=candidate.instruction_count,
            instruction_names=candidate.instruction_names,
            packet_magic=candidate.packet.magic,
            transfer_to_pl_bytes=len(packet_bytes),
            transfer_from_pl_bytes=len(output_bytes),
            elapsed_seconds=elapsed,
            template_key=candidate.template_key,
            template_hit=candidate.template_hit,
            template_seen_count=candidate.template_seen_count,
            repeat_group_size=candidate.repeat_group_size,
            repeat_count=candidate.repeat_count,
        )
        return RoutedProposal(proposal=proposal, decision=decision, instruction_count=candidate.instruction_count)


def xrt_backend_report(
    xclbin_path: Optional[str] = None,
    *,
    kernel_name: str = "pan_simd_v1",
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False,
        "backend": "pl-xrt",
        "kernel_name": kernel_name,
        "xclbin_path": xclbin_path,
        "xclbin_exists": bool(xclbin_path and Path(xclbin_path).exists()),
        "pyxrt_imported": False,
        "xrt_module": None,
        "device_visible": False,
        "device_error": None,
        "kernel_loaded": False,
        "kernel_error": None,
        "reason": None,
        "kernel_abi": "pan_simd_v1(input_bo, output_bo, input_size, output_size) binary PLTransitionPacketV1",
    }
    import_error: Optional[str] = None
    try:
        try:
            import pyxrt as xrt  # type: ignore

            report["xrt_module"] = "pyxrt"
        except ImportError:
            import xrt  # type: ignore

            report["xrt_module"] = "xrt"
        report["pyxrt_imported"] = True
    except Exception as exc:  # pragma: no cover - platform dependent
        import_error = f"pyxrt import failed: {exc}"
    if report["pyxrt_imported"]:
        try:
            device = xrt.device(0)
            report["device_visible"] = True
            del device
        except Exception as exc:  # pragma: no cover - platform dependent
            report["device_error"] = str(exc)
    if not xclbin_path:
        report["reason"] = "pl_xclbin_missing"
        return report
    if not report["xclbin_exists"]:
        report["reason"] = "pl_xclbin_not_found"
        return report
    if import_error is not None:
        report["reason"] = import_error
        return report
    if not report["device_visible"]:
        report["reason"] = f"xrt_device_unavailable: {report['device_error']}"
        return report
    try:
        device = xrt.device(0)
        uuid = device.load_xclbin(str(xclbin_path))
        xrt.kernel(device, uuid, kernel_name)
        report["kernel_loaded"] = True
    except Exception as exc:  # pragma: no cover - platform dependent
        report["kernel_error"] = str(exc)
        report["reason"] = f"xrt_kernel_unavailable: {exc}"
        return report
    report["ok"] = True
    report["reason"] = "available"
    return report


def _xrt_cacheable_bo_flag(xrt: Any) -> Any:
    bo_type = getattr(xrt, "bo")
    if hasattr(bo_type, "cacheable"):
        return bo_type.cacheable
    flags = getattr(bo_type, "flags", None)
    if flags is not None and hasattr(flags, "cacheable"):
        return flags.cacheable
    if flags is not None and hasattr(flags, "normal"):
        return flags.normal
    return 0


def _xrt_sync_bo(bo: Any, direction: Any, size: int) -> None:
    try:
        bo.sync(direction, int(size), 0)
    except TypeError:  # pragma: no cover - binding-version dependent
        bo.sync(direction)


def _xrt_write_mapped_bo(bo: Any, payload: bytes) -> None:
    view = memoryview(bo.map()).cast("B")
    view[: len(payload)] = payload


def _xrt_read_mapped_bo(bo: Any, size: int) -> bytes:
    return bytes(memoryview(bo.map()).cast("B")[: int(size)])


class XRTPLBackend:
    name = "pl-xrt"

    def __init__(self, xclbin_path: Optional[str], *, kernel_name: str = "pan_simd_v1"):
        self.xclbin_path = xclbin_path
        self.kernel_name = kernel_name
        self.report = xrt_backend_report(xclbin_path, kernel_name=kernel_name)

    def available(self) -> bool:
        return bool(self.report.get("ok"))

    def unavailable_reason(self) -> str:
        return str(self.report.get("reason") or "pl_unavailable")

    def _unsupported(self, runtime: object, candidate: BlockCandidate, detail: str) -> RoutedProposal:
        proposal = TransitionProposal(
            kind="unsupported",
            rip=candidate.initial_rip,
            instruction_hex="".join(op.instruction_hex for op in candidate.packet.ops),
            instruction_name=PL_SIMD_BLOCK_NAME,
            stage="stage4_simd_x87",
            effect_axes={"stage": "stage4_simd_x87", "backend": self.name, "failure": detail},
            profile=getattr(runtime, "profile", "long64-ring3"),
            iclass=PL_SIMD_BLOCK_NAME,
            iform=PL_SIMD_BLOCK_IFORM,
            stop_reason=StopReason("unsupported_runtime_feature", detail=detail, address=candidate.initial_rip),
            spf_manifest=_manifest_with_backend(runtime, self.name, candidate.packet),
            family_provenance=self.name,
            confidence=_proposal_confidence(candidate.proposals),
        )
        decision = RouteDecision(
            rip=candidate.initial_rip,
            requested_backend=getattr(runtime, "backend_mode", self.name),
            selected_backend=self.name,
            reason=detail,
            eligible=True,
            instruction_count=candidate.instruction_count,
            instruction_names=candidate.instruction_names,
            packet_magic=candidate.packet.magic,
            template_key=candidate.template_key,
            template_hit=candidate.template_hit,
            template_seen_count=candidate.template_seen_count,
            repeat_group_size=candidate.repeat_group_size,
            repeat_count=candidate.repeat_count,
        )
        return RoutedProposal(proposal=proposal, decision=decision, instruction_count=candidate.instruction_count)

    def propose(self, runtime: object, candidate: BlockCandidate) -> RoutedProposal:
        if not self.available():
            return self._unsupported(runtime, candidate, self.unavailable_reason())
        started = time.time()
        packet_bytes = replace(candidate.packet, output=None).encode_binary()
        out_size = max(4096, len(packet_bytes) * 2)
        try:
            try:
                import pyxrt as xrt  # type: ignore
            except ImportError:
                import xrt  # type: ignore

            device = xrt.device(0)
            uuid = device.load_xclbin(str(self.xclbin_path))
            kernel = xrt.kernel(device, uuid, self.kernel_name)
            bo_flag = _xrt_cacheable_bo_flag(xrt)
            in_bo = xrt.bo(device, len(packet_bytes), bo_flag, kernel.group_id(0))
            out_bo = xrt.bo(device, out_size, bo_flag, kernel.group_id(1))
            _xrt_write_mapped_bo(in_bo, packet_bytes)
            _xrt_sync_bo(in_bo, xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, len(packet_bytes))
            run = kernel(in_bo, out_bo, len(packet_bytes), out_size)
            run.wait()
            _xrt_sync_bo(out_bo, xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, out_size)
            raw = _xrt_read_mapped_bo(out_bo, out_size)
            output = PLTransitionOutputV1.decode_binary(raw)
        except Exception as exc:  # pragma: no cover - depends on deployed xclbin ABI
            return self._unsupported(runtime, candidate, f"xrt_kernel_invocation_failed: {exc}")
        proposal = _block_proposal_from_output(runtime, candidate, output, backend_name=self.name)
        output_bytes = output.encode_binary()
        elapsed = time.time() - started
        decision = RouteDecision(
            rip=candidate.initial_rip,
            requested_backend=getattr(runtime, "backend_mode", self.name),
            selected_backend=self.name,
            reason="eligible_simd_block",
            eligible=True,
            instruction_count=candidate.instruction_count,
            instruction_names=candidate.instruction_names,
            packet_magic=candidate.packet.magic,
            transfer_to_pl_bytes=len(packet_bytes),
            transfer_from_pl_bytes=len(output_bytes),
            elapsed_seconds=elapsed,
            template_key=candidate.template_key,
            template_hit=candidate.template_hit,
            template_seen_count=candidate.template_seen_count,
            repeat_group_size=candidate.repeat_group_size,
            repeat_count=candidate.repeat_count,
        )
        return RoutedProposal(proposal=proposal, decision=decision, instruction_count=candidate.instruction_count)


class RuntimeRouter:
    """Form SIMD transition blocks and select PS, PL simulator, or XRT PL."""

    def __init__(
        self,
        *,
        backend: str = "auto",
        pl_xclbin: Optional[str] = None,
        pl_kernel_name: str = "pan_simd_v1",
        min_instructions: int = 8,
        min_repeated_groups: int = 2,
        max_block_instructions: int = 128,
    ):
        if backend not in PL_BACKEND_MODES:
            raise ValueError(f"unsupported backend {backend!r}; expected one of {PL_BACKEND_MODES}")
        self.backend = backend
        self.pl_xclbin = pl_xclbin
        self.pl_kernel_name = pl_kernel_name
        self.min_instructions = int(min_instructions)
        self.min_repeated_groups = int(min_repeated_groups)
        self.max_block_instructions = int(max_block_instructions)
        self.ps = PSBackend()
        self.pl_sim = PLSimBackend()
        self.pl_xrt = XRTPLBackend(pl_xclbin, kernel_name=pl_kernel_name)
        self.template_cache: dict[str, int] = {}
        self.stats: dict[str, Any] = {
            "mode": backend,
            "pl_xclbin": pl_xclbin,
            "pl_kernel_name": pl_kernel_name,
            "blocks": {"ps": 0, "pl-sim": 0, "pl-xrt": 0},
            "instructions": {"ps": 0, "pl-sim": 0, "pl-xrt": 0},
            "fallback_reasons": {},
            "pl_unavailable": 0,
            "transfer_to_pl_bytes": 0,
            "transfer_from_pl_bytes": 0,
            "backend_elapsed_seconds": {"pl-sim": 0.0, "pl-xrt": 0.0},
            "templates": {"entries": 0, "hits": 0, "misses": 0, "repeat_blocks": 0},
            "xrt": dict(self.pl_xrt.report),
        }

    def backend_report(self) -> dict[str, Any]:
        return dict(self.pl_xrt.report)

    def _record(self, routed: RoutedProposal) -> RoutedProposal:
        backend = routed.decision.selected_backend
        self.stats["blocks"][backend] = int(self.stats["blocks"].get(backend, 0)) + 1
        self.stats["instructions"][backend] = int(self.stats["instructions"].get(backend, 0)) + routed.instruction_count
        if backend == "ps" and routed.decision.reason not in {"ps_selected", "backend_ps"}:
            reasons = self.stats["fallback_reasons"]
            reasons[routed.decision.reason] = int(reasons.get(routed.decision.reason, 0)) + 1
        if "unavailable" in routed.decision.reason or routed.decision.reason.startswith("pl_xclbin"):
            self.stats["pl_unavailable"] = int(self.stats.get("pl_unavailable", 0)) + 1
        self.stats["transfer_to_pl_bytes"] = int(self.stats.get("transfer_to_pl_bytes", 0)) + routed.decision.transfer_to_pl_bytes
        self.stats["transfer_from_pl_bytes"] = int(self.stats.get("transfer_from_pl_bytes", 0)) + routed.decision.transfer_from_pl_bytes
        if backend in self.stats["backend_elapsed_seconds"]:
            self.stats["backend_elapsed_seconds"][backend] = (
                float(self.stats["backend_elapsed_seconds"].get(backend, 0.0)) + routed.decision.elapsed_seconds
            )
        return routed

    def _annotate_candidate(self, candidate: BlockCandidate) -> BlockCandidate:
        key = _template_key(candidate.proposals)
        seen = int(self.template_cache.get(key, 0))
        self.template_cache[key] = seen + 1
        names = candidate.instruction_names
        repeat_group_size, repeat_count = _repeat_shape(names)
        templates = self.stats["templates"]
        if seen:
            templates["hits"] = int(templates.get("hits", 0)) + 1
        else:
            templates["misses"] = int(templates.get("misses", 0)) + 1
        templates["entries"] = len(self.template_cache)
        if repeat_count > 1 or seen:
            templates["repeat_blocks"] = int(templates.get("repeat_blocks", 0)) + 1
        return replace(
            candidate,
            template_key=key,
            template_hit=bool(seen),
            template_seen_count=seen + 1,
            repeat_group_size=repeat_group_size,
            repeat_count=repeat_count,
        )

    def route(self, runtime: object) -> RoutedProposal:
        runtime.backend_mode = self.backend
        instruction = decode_current(runtime)
        if self.backend == "ps":
            return self._record(self.ps.propose(runtime, instruction, reason="backend_ps"))
        if self.backend == "auto" and not self.pl_xrt.available():
            return self._record(self.ps.propose(runtime, instruction, reason=self.pl_xrt.unavailable_reason()))
        candidate, reason = self._form_candidate(runtime)
        if candidate is None:
            return self._record(self.ps.propose(runtime, instruction, reason=reason))
        if self.backend == "pl-sim":
            return self._record(self.pl_sim.propose(runtime, candidate))
        if self.backend == "pl-xrt":
            if self.pl_xrt.available():
                return self._record(self.pl_xrt.propose(runtime, candidate))
            return self._record(self.pl_xrt._unsupported(runtime, candidate, self.pl_xrt.unavailable_reason()))
        if self.backend == "auto":
            return self._record(self.pl_xrt.propose(runtime, candidate))
        return self._record(self.ps.propose(runtime, instruction, reason="unknown_backend"))

    def _form_candidate(self, runtime: object) -> tuple[Optional[BlockCandidate], str]:
        shadow = SimpleNamespace(state=copy.deepcopy(runtime.state), memory=runtime.memory.clone())
        proposals: list[TransitionProposal] = []
        while len(proposals) < self.max_block_instructions:
            try:
                instruction = decode_current(shadow)
                proposal = runtime.model.propose(shadow, instruction)
            except MemoryFault:
                return (None, "memory_fault_risk") if not proposals else self._candidate_or_threshold(runtime, shadow, proposals)
            except UnsupportedInstruction:
                return (None, "unsupported_form") if not proposals else self._candidate_or_threshold(runtime, shadow, proposals)
            if not self._is_pl_eligible_proposal(proposal):
                if not proposals:
                    return None, self._ineligible_reason(proposal)
                return self._candidate_or_threshold(runtime, shadow, proposals)
            proposals.append(proposal)
            _apply_transition_to_shadow(shadow, proposal)
        return self._candidate_or_threshold(runtime, shadow, proposals)

    def _candidate_or_threshold(
        self,
        runtime: object,
        shadow: object,
        proposals: list[TransitionProposal],
    ) -> tuple[Optional[BlockCandidate], str]:
        names = tuple(proposal.instruction_name for proposal in proposals)
        if len(proposals) >= self.min_instructions or _has_repeated_group(names, self.min_repeated_groups):
            packet = _candidate_packet(runtime, tuple(proposals), shadow)
            assert packet.output is not None
            candidate = BlockCandidate(
                initial_rip=int(proposals[0].rip) & MASK64,
                expected_final_rip=int(shadow.state.gpr["RIP"]) & MASK64,
                proposals=tuple(proposals),
                packet=packet,
                output=packet.output,
            )
            return (
                self._annotate_candidate(candidate),
                "eligible_simd_block",
            )
        return None, "below_pl_threshold"

    @staticmethod
    def _is_pl_eligible_proposal(proposal: TransitionProposal) -> bool:
        if proposal.kind != "normal" or proposal.stop_reason is not None or proposal.next_rip is None:
            return False
        if proposal.stage != "stage4_simd_x87":
            return False
        if proposal.instruction_name not in SUPPORTED_PL_SIMD_FORMS:
            return False
        if proposal.register_writes or proposal.rflags is not None or proposal.mxcsr is not None or proposal.x87_state is not None:
            return False
        return True

    @staticmethod
    def _ineligible_reason(proposal: TransitionProposal) -> str:
        if proposal.kind in {"syscall", "breakpoint"}:
            return f"{proposal.kind}_on_ps"
        if proposal.stage == "stage2_control_flow":
            return "control_or_syscall_on_ps"
        if proposal.stage == "stage4_simd_x87" and proposal.instruction_name.startswith(("F", "X87_")):
            return "x87_on_ps"
        if proposal.stage == "stage4_simd_x87":
            return "unsupported_stage4_form"
        return "non_simd_on_ps"


def backend_availability_report(
    xclbin_path: Optional[str] = None,
    *,
    kernel_name: str = "pan_simd_v1",
) -> dict[str, Any]:
    return {
        "modes": list(PL_BACKEND_MODES),
        "pl_sim_available": True,
        "xrt": xrt_backend_report(xclbin_path, kernel_name=kernel_name),
        "supported_pl_simd_forms": sorted(SUPPORTED_PL_SIMD_FORMS),
        "packet_magic": PL_PACKET_MAGIC,
        "packet_version": PL_PACKET_VERSION,
        "binary_packet_magic": PL_BINARY_MAGIC.rstrip(b"\x00").decode("ascii"),
        "binary_output_magic": PL_BINARY_OUTPUT_MAGIC.rstrip(b"\x00").decode("ascii"),
        "binary_header_size": PL_BINARY_HEADER_SIZE,
        "binary_op_size": PL_BINARY_OP_SIZE,
        "binary_memory_window_size": PL_BINARY_MEMORY_WINDOW_SIZE,
        "binary_output_header_size": PL_BINARY_OUTPUT_HEADER_SIZE,
        "binary_output_write_size": PL_BINARY_OUTPUT_WRITE_SIZE,
    }
