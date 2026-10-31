"""ELF process loading and Linux ABI setup for the PAN x86 runtime."""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .memory import SparseMemory, align_down, align_up


PT_LOAD = 1
PT_INTERP = 3
PF_X = 1
PF_W = 2
PF_R = 4
ET_EXEC = 2
ET_DYN = 3

AT_NULL = 0
AT_PHDR = 3
AT_PHENT = 4
AT_PHNUM = 5
AT_PAGESZ = 6
AT_BASE = 7
AT_ENTRY = 9
AT_UID = 11
AT_EUID = 12
AT_GID = 13
AT_EGID = 14
AT_RANDOM = 25
AT_EXECFN = 31

STACK_TOP = 0x00007FFFFFFFE000
STACK_SIZE = 8 * 1024 * 1024
DEFAULT_BRK_BASE = 0x0000000060000000


@dataclass
class LoadedSegment:
    vaddr: int
    memsz: int
    filesz: int
    offset: int
    permissions: str


@dataclass
class ProcessImage:
    memory: SparseMemory
    entry: int
    initial_rsp: int
    brk: int
    program: str
    argv: List[str]
    env: Dict[str, str]
    segments: List[LoadedSegment] = field(default_factory=list)
    auxv: Dict[int, int] = field(default_factory=dict)


def _segment_permissions(flags: int) -> str:
    permissions = ""
    if flags & PF_R:
        permissions += "r"
    if flags & PF_W:
        permissions += "w"
    if flags & PF_X:
        permissions += "x"
    return permissions or "r"


def _write_u64(mem: SparseMemory, address: int, value: int) -> None:
    mem.write(address, int(value).to_bytes(8, "little"), record=False)


def _push_u64(mem: SparseMemory, rsp: int, value: int) -> int:
    rsp -= 8
    _write_u64(mem, rsp, value)
    return rsp


def _push_bytes(mem: SparseMemory, rsp: int, data: bytes) -> tuple[int, int]:
    rsp -= len(data)
    mem.write(rsp, data, record=False)
    return rsp, rsp


def _parse_elf64(data: bytes) -> Dict[str, int]:
    if len(data) < 64 or data[:4] != b"\x7fELF":
        raise ValueError("not an ELF file")
    if data[4] != 2 or data[5] != 1:
        raise ValueError("only little-endian ELF64 is supported")
    header = struct.unpack_from("<16sHHIQQQIHHHHHH", data, 0)
    return {
        "type": header[1],
        "machine": header[2],
        "entry": header[4],
        "phoff": header[5],
        "phentsize": header[9],
        "phnum": header[10],
    }


def _iter_program_headers(data: bytes, elf: Dict[str, int]) -> Iterable[Dict[str, int]]:
    phoff = elf["phoff"]
    phentsize = elf["phentsize"]
    for index in range(elf["phnum"]):
        off = phoff + index * phentsize
        fields = struct.unpack_from("<IIQQQQQQ", data, off)
        yield {
            "type": fields[0],
            "flags": fields[1],
            "offset": fields[2],
            "vaddr": fields[3],
            "filesz": fields[5],
            "memsz": fields[6],
            "align": fields[7],
        }


def _write_initial_stack(
    memory: SparseMemory,
    *,
    program: str,
    argv: List[str],
    env: Dict[str, str],
    entry: int,
    phdr: int,
    phent: int,
    phnum: int,
    uid: Optional[int] = None,
    euid: Optional[int] = None,
    gid: Optional[int] = None,
    egid: Optional[int] = None,
) -> tuple[int, Dict[int, int]]:
    memory.map_region(STACK_TOP - STACK_SIZE, STACK_SIZE, "rw", "[stack]")
    rsp = STACK_TOP
    string_ptrs: Dict[str, int] = {}
    strings = [program, *argv[1:], *[f"{key}={value}" for key, value in sorted(env.items())]]
    for text in reversed(strings):
        rsp, ptr = _push_bytes(memory, rsp, text.encode("utf-8") + b"\x00")
        string_ptrs[text] = ptr

    rsp, random_ptr = _push_bytes(memory, rsp, bytes(range(16)))
    rsp &= ~0xF
    auxv = {
        AT_PHDR: phdr,
        AT_PHENT: phent,
        AT_PHNUM: phnum,
        AT_PAGESZ: 4096,
        AT_BASE: 0,
        AT_ENTRY: entry,
        AT_UID: int(uid) if uid is not None else (os.getuid() if hasattr(os, "getuid") else 0),
        AT_EUID: int(euid) if euid is not None else (os.geteuid() if hasattr(os, "geteuid") else 0),
        AT_GID: int(gid) if gid is not None else (os.getgid() if hasattr(os, "getgid") else 0),
        AT_EGID: int(egid) if egid is not None else (os.getegid() if hasattr(os, "getegid") else 0),
        AT_RANDOM: random_ptr,
        AT_EXECFN: string_ptrs[program],
    }

    rsp = _push_u64(memory, rsp, AT_NULL)
    rsp = _push_u64(memory, rsp, 0)
    for key, value in reversed(list(auxv.items())):
        rsp = _push_u64(memory, rsp, value)
        rsp = _push_u64(memory, rsp, key)

    rsp = _push_u64(memory, rsp, 0)
    env_strings = [f"{key}={value}" for key, value in sorted(env.items())]
    for text in reversed(env_strings):
        rsp = _push_u64(memory, rsp, string_ptrs[text])

    rsp = _push_u64(memory, rsp, 0)
    for text in reversed([program, *argv[1:]]):
        rsp = _push_u64(memory, rsp, string_ptrs[text])
    rsp = _push_u64(memory, rsp, len(argv))
    return rsp, auxv


def load_elf64(
    path: Path,
    argv: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    *,
    uid: Optional[int] = None,
    euid: Optional[int] = None,
    gid: Optional[int] = None,
    egid: Optional[int] = None,
) -> ProcessImage:
    path = Path(path)
    data = path.read_bytes()
    elf = _parse_elf64(data)
    if elf["machine"] != 0x3E:
        raise ValueError("only x86-64 ELF files are supported")
    if elf["type"] == ET_DYN:
        raise ValueError("dynamic/PIE ELF loading is not implemented in PAN runtime")
    if elf["type"] != ET_EXEC:
        raise ValueError(f"unsupported ELF type {elf['type']}; only static ET_EXEC is implemented")
    headers = list(_iter_program_headers(data, elf))
    if any(ph["type"] == PT_INTERP for ph in headers):
        raise ValueError("dynamic linker PT_INTERP ELF loading is not implemented in PAN runtime")

    memory = SparseMemory()
    segments: List[LoadedSegment] = []
    max_end = 0
    first_load_vaddr = 0
    for ph in headers:
        if ph["type"] != PT_LOAD:
            continue
        vaddr = int(ph["vaddr"])
        memsz = int(ph["memsz"])
        filesz = int(ph["filesz"])
        offset = int(ph["offset"])
        permissions = _segment_permissions(int(ph["flags"]))
        map_start = align_down(vaddr)
        map_end = align_up(vaddr + memsz)
        if not first_load_vaddr:
            first_load_vaddr = vaddr
        memory.map_region(map_start, map_end - map_start, "rw" if "w" in permissions else permissions + "w", path.name)
        memory.load_bytes(vaddr, data[offset:offset + filesz])
        if "w" not in permissions:
            memory.protect_region(map_start, map_end - map_start, permissions)
        segments.append(LoadedSegment(vaddr, memsz, filesz, offset, permissions))
        max_end = max(max_end, map_end)

    if not segments:
        raise ValueError("ELF contains no loadable segments")

    brk = align_up(max(max_end, DEFAULT_BRK_BASE))
    memory.map_region(brk, 4096, "rw", "[brk]")
    argv = list(argv or [str(path)])
    if not argv:
        argv = [str(path)]
    env = dict(env or {})
    phdr = first_load_vaddr + int(elf["phoff"])
    rsp, auxv = _write_initial_stack(
        memory,
        program=argv[0],
        argv=argv,
        env=env,
        entry=int(elf["entry"]),
        phdr=phdr,
        phent=int(elf["phentsize"]),
        phnum=int(elf["phnum"]),
        uid=uid,
        euid=euid,
        gid=gid,
        egid=egid,
    )
    return ProcessImage(
        memory=memory,
        entry=int(elf["entry"]),
        initial_rsp=rsp,
        brk=brk,
        program=str(path),
        argv=argv,
        env=env,
        segments=segments,
        auxv=auxv,
    )
