"""PAN x86 process execution runtime.

This is a deliberately small, fail-closed runtime foundation. It executes the
subset needed by the first remote-debuggable proof programs and records explicit
stop reasons for unsupported instructions, memory faults, breakpoints, exits,
and syscalls.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .memory import MemoryFault, SparseMemory, align_up
from .process import ProcessImage, load_elf64


MASK64 = (1 << 64) - 1
SIGNAL_TRAP = 5
SIGNAL_ILL = 4
SIGNAL_SEGV = 11
ENOSYS = 38

SUPPORTED_SYSCALLS = {
    0: "read",
    1: "write",
    9: "mmap",
    10: "mprotect",
    11: "munmap",
    12: "brk",
    39: "getpid",
    60: "exit",
    158: "arch_prctl",
    231: "exit_group",
}

THREAD_SYSCALLS = {
    56: "clone",
    57: "fork",
    58: "vfork",
    202: "futex",
    334: "rseq",
    435: "clone3",
}
SIGNAL_SYSCALLS = {
    62: "kill",
    13: "rt_sigaction",
    14: "rt_sigprocmask",
    15: "rt_sigreturn",
    127: "rt_sigpending",
    128: "rt_sigtimedwait",
    129: "rt_sigqueueinfo",
    130: "rt_sigsuspend",
    131: "sigaltstack",
    200: "tkill",
    234: "tgkill",
}
KNOWN_UNSUPPORTED_SYSCALLS = {
    2: ("filesystem", "open"),
    3: ("file descriptor", "close"),
    4: ("filesystem", "stat"),
    5: ("filesystem", "fstat"),
    6: ("filesystem", "lstat"),
    16: ("device/ioctl", "ioctl"),
    17: ("file descriptor", "pread64"),
    18: ("file descriptor", "pwrite64"),
    21: ("filesystem", "access"),
    22: ("file descriptor", "pipe"),
    59: ("process image", "execve"),
    61: ("process", "wait4"),
    63: ("system information", "uname"),
    72: ("file descriptor", "fcntl"),
    79: ("filesystem", "getcwd"),
    89: ("filesystem", "readlink"),
    97: ("resource limit", "getrlimit"),
    102: ("identity", "getuid"),
    104: ("identity", "getgid"),
    107: ("identity", "geteuid"),
    108: ("identity", "getegid"),
    110: ("process", "getppid"),
    186: ("threading", "gettid"),
    217: ("filesystem", "getdents64"),
    257: ("filesystem", "openat"),
    262: ("filesystem", "newfstatat"),
    273: ("threading", "set_robust_list"),
    302: ("resource limit", "prlimit64"),
    318: ("randomness", "getrandom"),
}


def runtime_scope() -> Dict[str, object]:
    return {
        "target_profile": "long64-ring3",
        "linux_user_abi": "single-thread static ELF subset",
        "dynamic_linking": False,
        "threading": False,
        "signals": False,
        "supported_syscalls": [SUPPORTED_SYSCALLS[number] for number in sorted(SUPPORTED_SYSCALLS)],
        "unsupported_cases": [
            "ET_DYN/PIE ELF loading",
            "PT_INTERP dynamic linker loading",
            "threading and process-clone syscalls",
            "signal delivery and signal-mask syscalls",
            "filesystem and descriptor syscalls outside stdout/stderr writes",
            "process image replacement and wait syscalls",
            "all remaining unimplemented Linux x86-64 syscalls",
            "unsupported x86 instructions",
        ],
        "unsupported_instructions_fail_closed": True,
        "unsupported_syscalls_fail_closed": True,
    }

GPR64_BY_CODE = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15")
GDB_REG_ORDER = ("RAX", "RBX", "RCX", "RDX", "RSI", "RDI", "RBP", "RSP", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15", "RIP")


class UnsupportedInstruction(Exception):
    def __init__(self, rip: int, opcode_hex: str, reason: str = "unsupported instruction"):
        super().__init__(f"{reason} at 0x{rip:x}: {opcode_hex}")
        self.rip = int(rip)
        self.opcode_hex = opcode_hex
        self.reason = reason


@dataclass
class StopReason:
    kind: str
    signal: Optional[int] = None
    exit_code: Optional[int] = None
    detail: str = ""
    address: Optional[int] = None

    def to_json(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "signal": self.signal,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "address": None if self.address is None else hex(self.address),
        }


@dataclass
class CPUState:
    gpr: Dict[str, int] = field(default_factory=dict)
    rflags: int = 0x202
    segments: Dict[str, int] = field(default_factory=lambda: {"CS": 0x33, "SS": 0x2B, "DS": 0, "ES": 0, "FS": 0, "GS": 0})
    control: Dict[str, int] = field(default_factory=dict)
    msr: Dict[str, int] = field(default_factory=dict)
    xmm: List[int] = field(default_factory=lambda: [0] * 16)
    mxcsr: int = 0x1F80
    x87: List[Optional[int]] = field(default_factory=lambda: [None] * 8)
    x87_top: int = 0
    x87_tag: int = 0xFFFF
    x87_status: int = 0
    x87_control: int = 0x037F

    @classmethod
    def initial(cls, entry: int, rsp: int) -> "CPUState":
        regs = {name: 0 for name in GPR64_BY_CODE}
        regs["RIP"] = int(entry) & MASK64
        regs["RSP"] = int(rsp) & MASK64
        return cls(gpr=regs)

    def snapshot(self, memory: SparseMemory) -> Dict[str, object]:
        return {
            "rip": hex(self.gpr["RIP"]),
            "gpr": {name: hex(self.gpr.get(name, 0) & MASK64) for name in sorted([*GPR64_BY_CODE, "RIP"])},
            "rflags": hex(self.rflags & MASK64),
            "segments": {name: hex(value) for name, value in sorted(self.segments.items())},
            "control": {name: hex(value) for name, value in sorted(self.control.items())},
            "msr": {name: hex(value) for name, value in sorted(self.msr.items())},
            "xmm": {f"XMM{i}": f"0x{value:032x}" for i, value in enumerate(self.xmm)},
            "mxcsr": hex(self.mxcsr),
            "x87": {
                "top": self.x87_top,
                "tag": hex(self.x87_tag & 0xFFFF),
                "status": hex(self.x87_status & 0xFFFF),
                "control": hex(self.x87_control & 0xFFFF),
                "st": {
                    f"ST{i}": None if value is None else f"0x{int(value) & ((1 << 80) - 1):020x}"
                    for i, value in enumerate(self.x87)
                },
            },
            "memory_digest": memory.digest(memory.dirty_pages),
        }


def _sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & (sign - 1)) - (value & sign)


def _read_le(data: bytes) -> int:
    return int.from_bytes(data, "little", signed=False)


def _reg_name(code: int, width: int) -> str:
    del width
    return GPR64_BY_CODE[code & 0xF]


def _write_reg(state: CPUState, code: int, value: int, width: int) -> None:
    name = _reg_name(code, width)
    if width == 32:
        state.gpr[name] = int(value) & 0xFFFFFFFF
    elif width == 64:
        state.gpr[name] = int(value) & MASK64
    elif width == 8:
        state.gpr[name] = (state.gpr[name] & ~0xFF) | (int(value) & 0xFF)
    else:
        raise ValueError(f"unsupported register width {width}")


def _read_reg(state: CPUState, code: int, width: int) -> int:
    value = state.gpr[_reg_name(code, width)] & MASK64
    if width == 32:
        return value & 0xFFFFFFFF
    if width == 8:
        return value & 0xFF
    return value


def _decode_modrm_address(state: CPUState, memory: SparseMemory, rip_after_modrm: int, modrm: int, rex: int) -> Tuple[int, int, int, int]:
    mod = (modrm >> 6) & 0x3
    reg = ((modrm >> 3) & 0x7) | (((rex >> 2) & 1) << 3)
    rm = (modrm & 0x7) | ((rex & 1) << 3)
    cursor = rip_after_modrm
    if mod == 3:
        return mod, reg, rm, cursor

    base_value = 0
    if (rm & 0x7) == 4:
        sib = memory.read(cursor, 1, access="execute", record=False)[0]
        cursor += 1
        scale = 1 << ((sib >> 6) & 0x3)
        index_code = ((sib >> 3) & 0x7) | (((rex >> 1) & 1) << 3)
        base_code = (sib & 0x7) | ((rex & 1) << 3)
        if (sib & 0x7) == 5 and mod == 0:
            base_value = 0
            disp = _sign_extend(_read_le(memory.read(cursor, 4, access="execute", record=False)), 32)
            cursor += 4
        else:
            base_value = _read_reg(state, base_code, 64)
            disp = 0
        index_value = 0 if (index_code & 0x7) == 4 else _read_reg(state, index_code, 64) * scale
        address = (base_value + index_value + disp) & MASK64
    elif (rm & 0x7) == 5 and mod == 0:
        disp = _sign_extend(_read_le(memory.read(cursor, 4, access="execute", record=False)), 32)
        cursor += 4
        address = (cursor + disp) & MASK64
    else:
        address = _read_reg(state, rm, 64)

    if mod == 1:
        disp = _sign_extend(memory.read(cursor, 1, access="execute", record=False)[0], 8)
        cursor += 1
        address = (address + disp) & MASK64
    elif mod == 2:
        disp = _sign_extend(_read_le(memory.read(cursor, 4, access="execute", record=False)), 32)
        cursor += 4
        address = (address + disp) & MASK64
    return mod, reg, address, cursor


class PanRuntime:
    def __init__(self, image: ProcessImage):
        self.image = image
        self.memory = image.memory
        self.state = CPUState.initial(image.entry, image.initial_rsp)
        self.stdout = bytearray()
        self.stderr = bytearray()
        self.stdin = bytearray()
        self.stop_reason: Optional[StopReason] = None
        self.breakpoints: set[int] = set()
        self.steps = 0
        self.syscalls: List[Dict[str, object]] = []
        self.start_unix = time.time()
        self.next_mmap = 0x00007FFFF0000000

    @classmethod
    def from_elf(cls, path: Path, argv: Optional[List[str]] = None, env: Optional[Dict[str, str]] = None) -> "PanRuntime":
        return cls(load_elf64(Path(path), argv=argv, env=env))

    def _fetch(self, size: int, offset: int = 0) -> bytes:
        return self.memory.read(self.state.gpr["RIP"] + offset, size, access="execute", record=False)

    def push_u64(self, value: int) -> None:
        rsp = (self.state.gpr["RSP"] - 8) & MASK64
        self.memory.write_int(rsp, value & MASK64, 8)
        self.state.gpr["RSP"] = rsp

    def pop_u64(self) -> int:
        rsp = self.state.gpr["RSP"]
        value = self.memory.read_int(rsp, 8)
        self.state.gpr["RSP"] = (rsp + 8) & MASK64
        return value

    def _set_zero_sign_flags(self, value: int, width: int) -> None:
        mask = (1 << width) - 1
        value &= mask
        if value == 0:
            self.state.rflags |= 1 << 6
        else:
            self.state.rflags &= ~(1 << 6)
        if value & (1 << (width - 1)):
            self.state.rflags |= 1 << 7
        else:
            self.state.rflags &= ~(1 << 7)

    def _syscall(self) -> StopReason:
        number = self.state.gpr["RAX"] & MASK64
        args = [self.state.gpr[name] & MASK64 for name in ("RDI", "RSI", "RDX", "R10", "R8", "R9")]
        self.syscalls.append({"number": number, "args": [hex(arg) for arg in args], "rip": hex(self.state.gpr["RIP"])})
        if number in (60, 231):
            code = args[0] & 0xFF
            self.stop_reason = StopReason("exit", exit_code=code, detail="exit syscall")
            return self.stop_reason
        if number == 1:
            fd, buf, count = args[0], args[1], args[2]
            data = self.memory.read(buf, count)
            if fd == 1:
                self.stdout.extend(data)
                self.state.gpr["RAX"] = count
            elif fd == 2:
                self.stderr.extend(data)
                self.state.gpr["RAX"] = count
            else:
                self.state.gpr["RAX"] = (-9) & MASK64
            return StopReason("running")
        if number == 0:
            self.state.gpr["RAX"] = 0
            return StopReason("running")
        if number == 12:
            requested = args[0]
            if requested == 0:
                self.state.gpr["RAX"] = self.image.brk
            else:
                new_brk = align_up(requested)
                if new_brk > self.image.brk:
                    self.memory.map_region(self.image.brk, new_brk - self.image.brk, "rw", "[brk]")
                self.image.brk = new_brk
                self.state.gpr["RAX"] = requested
            return StopReason("running")
        if number == 9:
            length = align_up(args[1])
            addr = args[0] or self.next_mmap
            self.next_mmap = align_up(addr + length + 0x10000)
            self.memory.map_region(addr, length, "rwx", "[mmap]")
            self.state.gpr["RAX"] = addr
            return StopReason("running")
        if number == 10:
            self.state.gpr["RAX"] = 0
            return StopReason("running")
        if number == 11:
            self.memory.unmap_region(args[0], args[1])
            self.state.gpr["RAX"] = 0
            return StopReason("running")
        if number == 39:
            self.state.gpr["RAX"] = 4242
            return StopReason("running")
        if number == 158:
            self.state.gpr["RAX"] = 0
            return StopReason("running")
        if number in THREAD_SYSCALLS:
            return self._unsupported_syscall(f"threading syscall {THREAD_SYSCALLS[number]} is not implemented")
        if number in SIGNAL_SYSCALLS:
            return self._unsupported_syscall(f"signal syscall {SIGNAL_SYSCALLS[number]} is not implemented")
        if number in KNOWN_UNSUPPORTED_SYSCALLS:
            category, name = KNOWN_UNSUPPORTED_SYSCALLS[number]
            return self._unsupported_syscall(f"{category} syscall {name} is not implemented")
        return self._unsupported_syscall(f"syscall {number} is not implemented")

    def _unsupported_syscall(self, detail: str) -> StopReason:
        self.state.gpr["RAX"] = (-ENOSYS) & MASK64
        return StopReason("unsupported_runtime_feature", signal=SIGNAL_ILL, detail=detail)

    def step(self, *, ignore_breakpoint: bool = False) -> StopReason:
        if self.stop_reason and self.stop_reason.kind == "exit":
            return self.stop_reason
        rip = self.state.gpr["RIP"]
        if not ignore_breakpoint and rip in self.breakpoints:
            self.stop_reason = StopReason("breakpoint", signal=SIGNAL_TRAP, detail="software breakpoint", address=rip)
            return self.stop_reason
        try:
            reason = self._step_instruction()
            self.steps += 1
            if reason.kind != "running":
                self.stop_reason = reason
                return reason
            return reason
        except MemoryFault as exc:
            self.stop_reason = StopReason("fault", signal=SIGNAL_SEGV, detail=exc.reason, address=exc.address)
            return self.stop_reason
        except UnsupportedInstruction as exc:
            self.stop_reason = StopReason("unsupported_instruction", signal=SIGNAL_ILL, detail=exc.reason, address=exc.rip)
            return self.stop_reason

    def _step_instruction(self) -> StopReason:
        rip = self.state.gpr["RIP"]
        raw = self._fetch(15)
        cursor = 0
        rex = 0
        if 0x40 <= raw[cursor] <= 0x4F:
            rex = raw[cursor]
            cursor += 1
        opcode = raw[cursor]
        cursor += 1
        width = 64 if rex & 0x08 else 32

        if opcode == 0x90:
            self.state.gpr["RIP"] = rip + cursor
            return StopReason("running")
        if opcode == 0xCC:
            self.state.gpr["RIP"] = rip + cursor
            return StopReason("breakpoint", signal=SIGNAL_TRAP, detail="int3", address=rip)
        if opcode == 0x0F and raw[cursor] == 0x05:
            self.state.gpr["RIP"] = rip + cursor + 1
            return self._syscall()
        if opcode == 0x55 and rex == 0:
            self.push_u64(self.state.gpr["RBP"])
            self.state.gpr["RIP"] = rip + cursor
            return StopReason("running")
        if opcode == 0x5D and rex == 0:
            self.state.gpr["RBP"] = self.pop_u64()
            self.state.gpr["RIP"] = rip + cursor
            return StopReason("running")
        if opcode == 0xC9:
            self.state.gpr["RSP"] = self.state.gpr["RBP"]
            self.state.gpr["RBP"] = self.pop_u64()
            self.state.gpr["RIP"] = rip + cursor
            return StopReason("running")
        if opcode == 0xC3:
            self.state.gpr["RIP"] = self.pop_u64()
            return StopReason("running")
        if opcode == 0xE8:
            rel = _sign_extend(_read_le(raw[cursor:cursor + 4]), 32)
            next_rip = rip + cursor + 4
            self.push_u64(next_rip)
            self.state.gpr["RIP"] = (next_rip + rel) & MASK64
            return StopReason("running")
        if opcode in (0xE9,):
            rel = _sign_extend(_read_le(raw[cursor:cursor + 4]), 32)
            next_rip = rip + cursor + 4
            self.state.gpr["RIP"] = (next_rip + rel) & MASK64
            return StopReason("running")
        if opcode == 0xEB:
            rel = _sign_extend(raw[cursor], 8)
            next_rip = rip + cursor + 1
            self.state.gpr["RIP"] = (next_rip + rel) & MASK64
            return StopReason("running")
        if 0xB8 <= opcode <= 0xBF:
            reg = (opcode - 0xB8) | ((rex & 1) << 3)
            if width == 64 and rex & 0x08:
                imm = _read_le(raw[cursor:cursor + 8])
                cursor += 8
            else:
                imm = _read_le(raw[cursor:cursor + 4])
                cursor += 4
                width = 32
            _write_reg(self.state, reg, imm, width)
            self.state.gpr["RIP"] = rip + cursor
            return StopReason("running")
        if opcode in (0x89, 0x8B, 0x8D, 0x31):
            modrm = raw[cursor]
            mod, reg, rm_or_address, end = _decode_modrm_address(self.state, self.memory, rip + cursor + 1, modrm, rex)
            reg_width = width
            if opcode == 0x8D:
                _write_reg(self.state, reg, rm_or_address, 64)
            elif opcode == 0x89:
                value = _read_reg(self.state, reg, reg_width)
                if mod == 3:
                    _write_reg(self.state, rm_or_address, value, reg_width)
                else:
                    self.memory.write(rm_or_address, value.to_bytes(reg_width // 8, "little"))
            elif opcode == 0x8B:
                if mod == 3:
                    value = _read_reg(self.state, rm_or_address, reg_width)
                else:
                    value = self.memory.read_int(rm_or_address, reg_width // 8)
                _write_reg(self.state, reg, value, reg_width)
            elif opcode == 0x31:
                src = _read_reg(self.state, reg, 32)
                dst = _read_reg(self.state, rm_or_address, 32) if mod == 3 else self.memory.read_int(rm_or_address, 4)
                value = (dst ^ src) & 0xFFFFFFFF
                if mod == 3:
                    _write_reg(self.state, rm_or_address, value, 32)
                else:
                    self.memory.write_int(rm_or_address, value, 4)
                self.state.rflags &= ~((1 << 0) | (1 << 11))
                self._set_zero_sign_flags(value, 32)
            self.state.gpr["RIP"] = end
            return StopReason("running")
        if opcode in (0x83, 0xC7):
            modrm = raw[cursor]
            op = (modrm >> 3) & 0x7
            mod, _reg, rm_or_address, end = _decode_modrm_address(self.state, self.memory, rip + cursor + 1, modrm, rex)
            if opcode == 0x83 and op in (0, 5):
                imm = _sign_extend(self.memory.read(end, 1, access="execute", record=False)[0], 8)
                end += 1
                old = _read_reg(self.state, rm_or_address, width) if mod == 3 else self.memory.read_int(rm_or_address, width // 8)
                value = (old + imm) & MASK64 if op == 0 else (old - imm) & MASK64
                if mod == 3:
                    _write_reg(self.state, rm_or_address, value, width)
                else:
                    self.memory.write(rm_or_address, value.to_bytes(width // 8, "little"))
                self._set_zero_sign_flags(value, width)
                self.state.gpr["RIP"] = end
                return StopReason("running")
            if opcode == 0xC7 and op == 0:
                imm = _read_le(self.memory.read(end, 4, access="execute", record=False))
                end += 4
                if mod == 3:
                    _write_reg(self.state, rm_or_address, imm, width)
                else:
                    self.memory.write(rm_or_address, imm.to_bytes(width // 8, "little"))
                self.state.gpr["RIP"] = end
                return StopReason("running")
        raise UnsupportedInstruction(rip, raw[: min(len(raw), cursor + 6)].hex())

    def run(self, max_steps: int = 100000) -> StopReason:
        reason = StopReason("running")
        for _ in range(max_steps):
            reason = self.step()
            if reason.kind != "running":
                return reason
        self.stop_reason = StopReason("timeout", detail=f"max_steps={max_steps}")
        return self.stop_reason

    def summary(self) -> Dict[str, object]:
        elapsed = time.time() - self.start_unix
        return {
            "ok": bool(self.stop_reason and self.stop_reason.kind == "exit"),
            "program": self.image.program,
            "steps": self.steps,
            "elapsed_seconds": elapsed,
            "steps_per_second": self.steps / elapsed if elapsed else None,
            "stop_reason": None if self.stop_reason is None else self.stop_reason.to_json(),
            "exit_code": None if self.stop_reason is None else self.stop_reason.exit_code,
            "stdout_hex": bytes(self.stdout).hex(),
            "stdout_text": bytes(self.stdout).decode("utf-8", errors="replace"),
            "stderr_hex": bytes(self.stderr).hex(),
            "syscalls": self.syscalls,
            "state": self.state.snapshot(self.memory),
            "mapped_ranges": self.memory.mapped_ranges(),
            "runtime_scope": runtime_scope(),
        }
