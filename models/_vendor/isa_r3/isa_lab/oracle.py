"""Independent emulator oracle and a labeled Python mechanics reference."""
from __future__ import annotations

import importlib.metadata

from .contracts import State
from .profile import instruction_key


class PythonReference:
    authority = "python-mechanics-reference; no independent emulator certification"

    def step(self, instruction: bytes, before: State) -> State:
        instruction_key(instruction)
        # Independently expressed sign arithmetic, deliberately separate from candidates.
        op = instruction.hex() if instruction[0] in (0x90, 0xFE) else instruction[:1].hex()
        a, b = before.al, instruction[1] if len(instruction) == 2 else 0
        values = before.to_dict()
        values["rip"] = (before.rip + len(instruction)) % 2**64
        if op == "90":
            return State(**values)
        if op == "b0":
            values["al"] = b
            return State(**values)
        signed = lambda x: x if x < 128 else x - 256
        if op in ("04", "fec0"):
            rhs = 1 if op == "fec0" else b
            mathematical = a + rhs
            signed_result = signed(a) + signed(rhs)
            values["cf"] = before.cf if op == "fec0" else int(mathematical >= 256)
            values["of"] = int(not -128 <= signed_result <= 127)
        elif op in ("2c", "3c", "fec8"):
            rhs = 1 if op == "fec8" else b
            mathematical = a - rhs
            signed_result = signed(a) - signed(rhs)
            values["cf"] = before.cf if op == "fec8" else int(mathematical < 0)
            values["of"] = int(not -128 <= signed_result <= 127)
        else:
            mathematical = {"34": a ^ b, "24": a & b, "0c": a | b, "a8": a & b}[op]
            values["cf"] = values["of"] = 0
        value = mathematical % 256
        values["zf"], values["sf"] = int(value == 0), int(value >= 128)
        if op not in ("3c", "a8"):
            values["al"] = value
        return State(**values)


class UnicornOracle:
    def __init__(self):
        import unicorn
        self.unicorn = unicorn
        self.authority = f"unicorn-{importlib.metadata.version('unicorn')}; x86-64 single-step"

    def step(self, instruction: bytes, before: State) -> State:
        instruction_key(instruction)
        from unicorn import x86_const as reg
        uc = self.unicorn.Uc(self.unicorn.UC_ARCH_X86, self.unicorn.UC_MODE_64)
        page = before.rip & ~4095
        uc.mem_map(page, 8192)
        uc.mem_write(before.rip, instruction)
        uc.reg_write(reg.UC_X86_REG_RAX, before.al)
        uc.reg_write(reg.UC_X86_REG_RBX, before.bl)
        flags = 2 | before.cf | (before.zf << 6) | (before.sf << 7) | (before.of << 11)
        uc.reg_write(reg.UC_X86_REG_EFLAGS, flags)
        uc.emu_start(before.rip, before.rip + len(instruction), count=1)
        flags = uc.reg_read(reg.UC_X86_REG_EFLAGS)
        return State(al=uc.reg_read(reg.UC_X86_REG_RAX) & 255,
                     bl=uc.reg_read(reg.UC_X86_REG_RBX) & 255,
                     rip=uc.reg_read(reg.UC_X86_REG_RIP),
                     cf=flags & 1, zf=(flags >> 6) & 1, sf=(flags >> 7) & 1, of=(flags >> 11) & 1)
