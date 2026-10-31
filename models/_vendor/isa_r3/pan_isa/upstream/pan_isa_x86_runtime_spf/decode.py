"""Deterministic byte-stream decode front end for the SPF executor."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodedInstruction:
    rip: int
    raw: bytes
    prefixes: tuple[int, ...]
    rex: int
    opcode: int
    cursor_after_opcode: int
    width: int

    @property
    def opcode_hex(self) -> str:
        end = min(len(self.raw), max(self.cursor_after_opcode + 6, 1))
        return self.raw[:end].hex()


def decode_current(runtime: object) -> DecodedInstruction:
    state = getattr(runtime, "state")
    memory = getattr(runtime, "memory")
    rip = int(state.gpr["RIP"])
    raw = memory.read(rip, 15, access="execute", record=False)
    cursor = 0
    prefixes = []
    while raw[cursor] in (0x66, 0xF2, 0xF3):
        prefixes.append(raw[cursor])
        cursor += 1
    rex = 0
    if 0x40 <= raw[cursor] <= 0x4F:
        rex = raw[cursor]
        cursor += 1
    opcode = raw[cursor]
    cursor += 1
    width = 64 if rex & 0x08 else 32
    return DecodedInstruction(
        rip=rip,
        raw=raw,
        prefixes=tuple(prefixes),
        rex=rex,
        opcode=opcode,
        cursor_after_opcode=cursor,
        width=width,
    )
