"""M4 data side: a broad x86-64 integer form catalog, a random encoder, and Unicorn transitions with paged memory.

This module is the ORACLE side. It knows x86 (it encodes instructions and picks register values so that memory
operands land on canonical addresses); the learner (`m4_learn`) never imports it. A training row is

  before:  window (15 bytes at RIP), 16 GPRs, RIP, six status flags, and a page store
  after:   16 GPRs, RIP, six status flags, the changed memory bytes, and a trap bit (SYSCALL reached)

The page store is a deterministic function of (row seed, address): every byte not overridden is
`hash_byte(seed, addr)` ("random" mode) or a sparse 0/1 word pattern ("small" mode, so equal values and zero
results occur), plus explicit overrides (the code window at RIP, canonical return addresses / jump targets for
RET and indirect jumps). The learner gets the same function (`PageStore`), so it can read memory anywhere, exactly
as a model-virtual-memory page table would serve it.

Data restrictions (reported): no legacy high-byte registers (AH..BH), no 16-bit operand-size ALU forms (only the
66-prefixed NOPs), no shift by a zero count, no string instructions, division operands chosen so DIV/IDIV do not
fault. Labels are Unicorn 2.1.4 (QEMU TCG) single steps on a fresh code mapping per row.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

MASK64 = (1 << 64) - 1
WINDOW = 15
GPRS = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15")
FLAGS = ("CF", "PF", "AF", "ZF", "SF", "OF")
FLAG_BITS = (0, 2, 4, 6, 7, 11)
EFLAGS_FIXED = 0x202  # reserved bit 1 and IF
CANON = (1 << 47) - 1

# ------------------------------------------------------------------ page store (shared with the learner)

_M1, _M2 = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB)


def _mix(x):
    x = np.asarray(x, dtype=np.uint64)
    with np.errstate(over="ignore"):
        x = x + np.uint64(0x9E3779B97F4A7C15)
        x = (x ^ (x >> np.uint64(30))) * _M1
        x = (x ^ (x >> np.uint64(27))) * _M2
        return x ^ (x >> np.uint64(31))


def hash_bytes(seed, mode, addr):
    """Vectorized background memory: seed (N,), mode (N,) 0 random / 1 small, addr (N, K) uint64 → (N, K) uint8."""
    seed = np.asarray(seed, dtype=np.uint64)[:, None]
    addr = np.asarray(addr, dtype=np.uint64)
    with np.errstate(over="ignore"):
        w = _mix(seed * np.uint64(0x100000001B3) ^ (addr >> np.uint64(3)))
    byte = ((w >> ((addr & np.uint64(7)) * np.uint64(8))) & np.uint64(0xFF)).astype(np.uint8)
    small = ((w & np.uint64(1)).astype(np.uint8)) * ((addr & np.uint64(7)) == 0)
    return np.where(np.asarray(mode)[:, None] == 1, small, byte).astype(np.uint8)


class PageStore:
    """Byte-addressed memory of N rows: background hash plus per-row overrides, held as flat sorted arrays
    (row r owns addr[ptr[r]:ptr[r+1]]) so forked workers share them without copy-on-write growth."""

    def __init__(self, seed, mode, overrides=None, csr=None):
        self.seed = np.asarray(seed, dtype=np.uint64)
        self.mode = np.asarray(mode, dtype=np.uint8)
        if csr is None:
            ptr = np.zeros(len(overrides) + 1, np.int64)
            ks, vs = [], []
            for r, ov in enumerate(overrides):
                k = sorted(ov)
                ks.append(np.array(k, np.uint64))
                vs.append(np.array([ov[a] for a in k], np.uint8))
                ptr[r + 1] = ptr[r] + len(k)
            csr = (ptr, np.concatenate(ks) if ks else np.zeros(0, np.uint64), np.concatenate(vs) if vs else np.zeros(0, np.uint8))
        self.ptr, self.addr, self.val = csr

    @staticmethod
    def concat(a, b):
        return PageStore(np.concatenate([a.seed, b.seed]), np.concatenate([a.mode, b.mode]),
                         csr=(np.concatenate([a.ptr, b.ptr[1:] + a.ptr[-1]]), np.concatenate([a.addr, b.addr]), np.concatenate([a.val, b.val])))

    def _keys(self, i):
        a, b = self.ptr[i], self.ptr[i + 1]
        return self.addr[a:b], self.val[a:b]

    def take(self, rows):
        rows = np.asarray(rows, np.int64)
        lo, hi = self.ptr[rows], self.ptr[rows + 1]
        n = hi - lo
        ptr = np.zeros(len(rows) + 1, np.int64)
        np.cumsum(n, out=ptr[1:])
        idx = np.repeat(lo - ptr[:-1], n) + np.arange(ptr[-1])
        return PageStore(self.seed[rows], self.mode[rows], csr=(ptr, self.addr[idx], self.val[idx]))

    def read(self, addr, nbytes):
        """addr (N,) or (N, C) uint64 -> little-endian values (same shape) of nbytes bytes."""
        addr = np.asarray(addr, dtype=np.uint64)
        flat = addr.reshape(len(self.seed), -1)
        C = flat.shape[1]
        offs = np.arange(nbytes, dtype=np.uint64)
        with np.errstate(over="ignore"):
            a = (flat[:, :, None] + offs[None, None, :]).reshape(len(self.seed), -1)
        b = hash_bytes(self.seed, self.mode, a)
        for i in range(len(self.seed)):
            k, v = self._keys(i)
            if len(k):
                pos = np.searchsorted(k, a[i])
                pos = np.minimum(pos, len(k) - 1)
                hit = k[pos] == a[i]
                if hit.any():
                    b[i, hit] = v[pos[hit]]
        b = b.reshape(len(self.seed), C, nbytes).astype(np.uint64)
        val = (b << (np.uint64(8) * offs)[None, None, :]).sum(-1, dtype=np.uint64)
        return val.reshape(addr.shape)


# ------------------------------------------------------------------ form catalog

ALU = {"add": 0, "or": 1, "adc": 2, "sbb": 3, "and": 4, "sub": 5, "xor": 6, "cmp": 7}
CC = ("o", "no", "b", "ae", "e", "ne", "be", "a", "s", "ns", "p", "np", "l", "ge", "le", "g")


@dataclass(frozen=True)
class Form:
    name: str
    opcode: bytes
    width: int  # operand width (REX.W when 64; 8 = byte form)
    modrm: str  # "rm_r" (reg = register operand), "digit" (reg = fixed), "plus_r" (register in opcode), "none"
    digit: int = 0
    rm: str = "any"  # "r", "m" or "any": rm operand kind for modrm forms
    imm: int = 0  # immediate bytes
    rel: int = 0  # relative displacement bytes (branches)
    prefix: bytes = b""
    rex: str = "auto"  # "auto" (REX.W from width, REX.R/X/B as needed), "none", "plain40"
    reg_width: int = 0  # width of the ModRM.reg register when it differs from width (movzx/movsx)
    rows: int = 8  # rows per encoding key
    special: str = ""


def _catalog():
    F = []
    for op, n in ALU.items():
        base = n * 8
        for w in (8, 32, 64):
            o8 = w == 8
            for rmk in ("r", "m"):
                F.append(Form(f"{op}_rm{w}{rmk}_r", bytes([base + (0 if o8 else 1)]), w, "rm_r", rm=rmk))
                F.append(Form(f"{op}_r_rm{w}{rmk}", bytes([base + (2 if o8 else 3)]), w, "rm_r", rm=rmk))
                if o8:
                    F.append(Form(f"{op}_rm8{rmk}_i8", b"\x80", 8, "digit", n, rm=rmk, imm=1, rows=12))
                else:
                    F.append(Form(f"{op}_rm{w}{rmk}_i32", b"\x81", w, "digit", n, rm=rmk, imm=4, rows=12))
                    F.append(Form(f"{op}_rm{w}{rmk}_i8", b"\x83", w, "digit", n, rm=rmk, imm=1, rows=12))
            F.append(Form(f"{op}_acc{w}_i", bytes([base + (4 if o8 else 5)]), w, "none", imm=1 if o8 else 4, rows=24))
    for w in (8, 32, 64):
        o8 = w == 8
        for rmk in ("r", "m"):
            F.append(Form(f"test_rm{w}{rmk}_r", b"\x84" if o8 else b"\x85", w, "rm_r", rm=rmk))
            F.append(Form(f"test_rm{w}{rmk}_i", b"\xf6" if o8 else b"\xf7", w, "digit", 0, rm=rmk, imm=1 if o8 else 4, rows=12))
            F.append(Form(f"mov_rm{w}{rmk}_r", b"\x88" if o8 else b"\x89", w, "rm_r", rm=rmk))
            F.append(Form(f"mov_r_rm{w}{rmk}", b"\x8a" if o8 else b"\x8b", w, "rm_r", rm=rmk))
            F.append(Form(f"mov_rm{w}{rmk}_i", b"\xc6" if o8 else b"\xc7", w, "digit", 0, rm=rmk, imm=1 if o8 else 4, rows=12))
            for d, nm in ((0, "inc"), (1, "dec"), (2, "not"), (3, "neg")):
                F.append(Form(f"{nm}_rm{w}{rmk}", b"\xfe" if o8 and d < 2 else (b"\xff" if d < 2 else (b"\xf6" if o8 else b"\xf7")),
                              w, "digit", d, rm=rmk))
            for d, nm in ((4, "shl"), (5, "shr"), (7, "sar"), (0, "rol"), (1, "ror")):
                F.append(Form(f"{nm}_rm{w}{rmk}_i", b"\xc0" if o8 else b"\xc1", w, "digit", d, rm=rmk, imm=1, rows=12, special="shift"))
                F.append(Form(f"{nm}_rm{w}{rmk}_1", b"\xd0" if o8 else b"\xd1", w, "digit", d, rm=rmk))
                F.append(Form(f"{nm}_rm{w}{rmk}_cl", b"\xd2" if o8 else b"\xd3", w, "digit", d, rm=rmk, rows=12, special="shift_cl"))
            F.append(Form(f"set_rm8{rmk}", b"", 8, "digit", 0, rm=rmk, rows=48, special="setcc"))
        F.append(Form(f"test_acc{w}_i", b"\xa8" if o8 else b"\xa9", w, "none", imm=1 if o8 else 4, rows=24))
        F.append(Form(f"mov_r{w}_i", bytes([0xB0 if o8 else 0xB8]), w, "plus_r", imm=1 if o8 else (8 if w == 64 else 4), rows=12))
    for w in (32, 64):
        for rmk in ("r", "m"):
            F.append(Form(f"movzx_r{w}_rm8{rmk}", b"\x0f\xb6", w, "rm_r", rm=rmk, special="src8"))
            F.append(Form(f"movzx_r{w}_rm16{rmk}", b"\x0f\xb7", w, "rm_r", rm=rmk, special="src16"))
            F.append(Form(f"movsx_r{w}_rm8{rmk}", b"\x0f\xbe", w, "rm_r", rm=rmk, special="src8"))
            F.append(Form(f"movsx_r{w}_rm16{rmk}", b"\x0f\xbf", w, "rm_r", rm=rmk, special="src16"))
            F.append(Form(f"imul_r_rm{w}{rmk}", b"\x0f\xaf", w, "rm_r", rm=rmk))
            F.append(Form(f"imul_r_rm{w}{rmk}_i8", b"\x6b", w, "rm_r", rm=rmk, imm=1, rows=12))
            F.append(Form(f"imul_r_rm{w}{rmk}_i32", b"\x69", w, "rm_r", rm=rmk, imm=4, rows=12))
            for d, nm in ((4, "mul"), (5, "imul1"), (6, "div"), (7, "idiv")):
                F.append(Form(f"{nm}_rm{w}{rmk}", b"\xf7", w, "digit", d, rm=rmk, rows=12, special=nm))
            F.append(Form(f"cmov_r_rm{w}{rmk}", b"", w, "rm_r", rm=rmk, rows=48, special="cmov"))
        F.append(Form(f"lea_r{w}", b"\x8d", w, "rm_r", rm="m"))
    F.append(Form("movsxd_r64_rm32r", b"\x63", 64, "rm_r", rm="r", special="src32"))
    F.append(Form("movsxd_r64_rm32m", b"\x63", 64, "rm_r", rm="m", special="src32"))
    F.append(Form("cdqe", b"\x98", 64, "none"))
    F.append(Form("cdq", b"\x99", 32, "none"))
    F.append(Form("cqo", b"\x99", 64, "none"))
    F.append(Form("push_r", b"\x50", 64, "plus_r", rex="noW"))
    F.append(Form("pop_r", b"\x58", 64, "plus_r", rex="noW"))
    F.append(Form("push_i8", b"\x6a", 64, "none", imm=1, rex="none", rows=12))
    F.append(Form("push_i32", b"\x68", 64, "none", imm=4, rex="none", rows=12))
    F.append(Form("call_rel32", b"\xe8", 64, "none", rel=4, rex="none", rows=12))
    F.append(Form("call_rm", b"\xff", 64, "digit", 2, rm="any", rex="noW", special="target"))
    F.append(Form("jmp_rel32", b"\xe9", 64, "none", rel=4, rex="none", rows=12))
    F.append(Form("jmp_rel8", b"\xeb", 64, "none", rel=1, rex="none", rows=12))
    F.append(Form("jmp_rm", b"\xff", 64, "digit", 4, rm="any", rex="noW", special="target"))
    F.append(Form("ret", b"\xc3", 64, "none", rex="none", rows=12, special="ret"))
    F.append(Form("leave", b"\xc9", 64, "none", rex="none", rows=12, special="leave"))
    F.append(Form("jcc_rel8", b"", 64, "none", rel=1, rex="none", rows=48, special="jcc8"))
    F.append(Form("jcc_rel32", b"", 64, "none", rel=4, rex="none", rows=48, special="jcc32"))
    F.append(Form("nop", b"\x90", 64, "none", rex="none", rows=6))
    F.append(Form("nop_66", b"\x66\x90", 64, "none", rex="none", rows=6))
    F.append(Form("nop_rm", b"\x0f\x1f", 32, "digit", 0, rm="m", rex="none", rows=6))
    F.append(Form("nop_rm_66", b"\x0f\x1f", 32, "digit", 0, rm="m", rex="none", prefix=b"\x66", rows=6))
    F.append(Form("nop_rm_66cs", b"\x0f\x1f", 32, "digit", 0, rm="m", rex="none", prefix=b"\x66\x2e", rows=6))
    F.append(Form("syscall", b"\x0f\x05", 64, "none", rex="none", rows=6, special="syscall"))
    return {f.name: f for f in F}


FORMS = _catalog()
FORM_NAMES = tuple(sorted(FORMS))


# ------------------------------------------------------------------ encoding


@dataclass
class Ins:
    form: str
    code: bytes  # full instruction bytes
    key: bytes  # bytes before disp/imm/rel (the oracle's grouping key; the learner never sees it)
    regs: dict  # role -> register index (dest/src/base/index/rm/reg)
    mem: dict | None  # {"mod", "base", "index", "scale", "disp_bytes", "riprel"}
    cc: int | None = None


def _rex(w, r, x, b, force=False):
    v = 0x40 | (8 if w else 0) | (4 if r & 8 else 0) | (2 if x & 8 else 0) | (1 if b & 8 else 0)
    return bytes([v]) if (v != 0x40 or force) else b""


def _reg8_ok(code, rex_present):
    return rex_present or code < 4


ADDR_CLASSES = ("base", "base_d8", "base_d32", "rip", "sib_d0", "sib_d8", "sib_d32", "sibni_d0", "sibni_d8", "sibni_d32",
                "sib_nobase", "sib_nobase_noindex")


def _addr_choice(rng, addr):
    """(mod, rm3, scale, index, base_low3) for a forced addressing class; index/base 3-bit parts only."""
    if addr in ("base", "base_d8", "base_d32"):
        mod = {"base": 0, "base_d8": 1, "base_d32": 2}[addr]
        rm3 = rng.choice([0, 1, 2, 3, 6, 7] + ([5] if mod else []))
        return mod, rm3, None, None, None
    if addr == "rip":
        return 0, 5, None, None, None
    scale = rng.randrange(4)
    if addr.startswith("sibni"):
        mod = {"sibni_d0": 0, "sibni_d8": 1, "sibni_d32": 2}[addr]
        return mod, 4, scale, 4, rng.choice([0, 1, 2, 3, 4, 6, 7] + ([5] if mod else []))
    if addr.startswith("sib_d"):
        mod = {"sib_d0": 0, "sib_d8": 1, "sib_d32": 2}[addr]
        return mod, 4, scale, rng.choice([0, 1, 2, 3, 5, 6, 7]), rng.choice([0, 1, 2, 3, 4, 6, 7] + ([5] if mod else []))
    if addr == "sib_nobase":
        return 0, 4, scale, rng.choice([0, 1, 2, 3, 5, 6, 7]), 5
    if addr == "sib_nobase_noindex":
        return 0, 4, scale, 4, 5
    raise ValueError(addr)


def sample_key(rng, form: Form, held=None, force=None):
    """Random encoding key (everything but disp/imm/rel bytes). Returns (key bytes, regs, mem layout, cc).
    force (optional) pins an encoding-shape cell: {"cc": int, "rex": "none"|"present", "addr": "reg"|ADDR_CLASSES[i]}."""
    f = form
    force = force or {}
    norex = force.get("rex") == "none"
    for _ in range(400):
        regs, mem, cc = {}, None, None
        opcode = f.opcode
        hi = lambda: rng.randrange(8) if norex else rng.randrange(16)
        if f.special in ("jcc8", "jcc32", "setcc", "cmov"):
            cc = force["cc"] if "cc" in force else rng.randrange(16)
            opcode = {"jcc8": bytes([0x70 + cc]), "jcc32": bytes([0x0F, 0x80 + cc]), "setcc": bytes([0x0F, 0x90 + cc]),
                      "cmov": bytes([0x0F, 0x40 + cc])}[f.special]
        w64 = f.width == 64 and f.rex not in ("noW", "none")
        if f.modrm == "none":
            rex = _rex(w64, 0, 0, 0) if f.rex == "auto" else b""
            key = f.prefix + rex + opcode
        elif f.modrm == "plus_r":
            r = hi()
            regs["reg"] = r
            rex = _rex(w64, 0, 0, r) if f.rex in ("auto", "noW") else b""
            if f.width == 8 and not _reg8_ok(r, bool(rex)):
                rex = _rex(False, 0, 0, r, force=True) if rng.random() < 0.5 and not norex else None
                if rex is None:
                    continue
            key = f.prefix + rex + bytes([opcode[0] + (r & 7)])
        else:
            reg = f.digit if f.modrm == "digit" else hi()
            addr = force.get("addr")
            rmk = f.rm if f.rm != "any" else ("r" if addr in ("reg", "reg_alias") else "m" if addr else rng.choice("rm"))
            if rmk == "r":
                mod, rmv = 3, (reg if addr == "reg_alias" else hi())
                regs["rm"] = rmv
                sib, x, b = b"", 0, rmv
                disp_bytes = 0
            else:
                b_ext = 0 if norex else rng.randrange(2) * 8
                x_ext = 0 if norex else rng.randrange(2) * 8
                if addr:
                    mod, rm3, scale, index3, base3 = _addr_choice(rng, addr)
                else:
                    mod, rm3 = rng.randrange(3), rng.randrange(8)
                    scale, index3, base3 = rng.randrange(4), None, None
                x = 0
                sib = b""
                disp_bytes = {0: 0, 1: 1, 2: 4}[mod]
                riprel = False
                if rm3 == 4:
                    index = (index3 | x_ext) if index3 is not None else rng.randrange(16) if not norex else rng.randrange(8)
                    if index3 == 4:
                        index = 4  # no index (index 12 = R12 is a real index)
                    base = ((base3 if base3 is not None else rng.randrange(8)) | b_ext)
                    sib = bytes([(scale << 6) | ((index & 7) << 3) | (base & 7)])
                    x = index
                    no_base = mod == 0 and (base & 7) == 5
                    if no_base:
                        disp_bytes = 4
                    mem = {"base": None if no_base else base, "index": None if index == 4 else index, "scale": scale}
                    b = base
                elif rm3 == 5 and mod == 0:
                    riprel = True
                    disp_bytes = 4
                    mem = {"base": "rip", "index": None, "scale": 0}
                    b = b_ext
                else:
                    mem = {"base": rm3 | b_ext, "index": None, "scale": 0}
                    b = rm3 | b_ext
                mem.update(mod=mod, disp_bytes=disp_bytes, riprel=riprel)
                rmv = rm3
            if f.modrm == "rm_r":
                regs["reg"] = reg
            rex_needed = (reg & 8) or (x & 8) or (b & 8)
            if f.rex == "none":
                if rex_needed:
                    continue
                rex = b""
            else:
                rex = _rex(w64, reg if f.modrm == "rm_r" else 0, x, b)
                if f.width == 8 or f.special == "setcc":
                    need8 = []
                    if f.modrm == "rm_r":
                        need8.append(reg)
                    if rmk == "r":
                        need8.append(regs["rm"])
                    if f.special in ("src8",) and rmk == "r":
                        need8 = [regs["rm"]]
                    if any(c >= 4 and c < 8 for c in need8) and not rex:
                        rex = _rex(False, reg if f.modrm == "rm_r" else 0, x, b, force=True)
                elif f.special == "src8" and rmk == "r" and (regs["rm"] & 7) >= 4 and regs["rm"] < 8 and not rex:
                    rex = _rex(False, reg, x, b, force=True)
            modrm = bytes([(mod << 6) | ((reg & 7) << 3) | ((rmv if rmk == "r" else rmv) & 7)])
            key = f.prefix + rex + opcode + modrm + sib
            if rmk == "m":
                regs.update({"base": mem["base"], "index": mem["index"]})
        if "rex" in force and (force["rex"] == "none") != (not _has_rex(key, f)):
            continue
        if held is not None and held(f, regs):
            continue
        return key, regs, mem, cc
    raise RuntimeError(f"could not sample {f.name} {force}")


def _has_rex(key, f):
    k = key[len(f.prefix):]
    return bool(k) and 0x40 <= k[0] <= 0x4F


def shape_cells(f: Form):
    """Every encoding-shape cell of a form: (cc, rex, addr)."""
    ccs = list(range(16)) if f.special in ("jcc8", "jcc32", "setcc", "cmov") else [None]
    w64 = f.width == 64 and f.rex not in ("noW", "none")
    if f.rex == "none":
        rexs = ["none"]
    elif w64:
        rexs = ["present"]
    elif f.modrm == "none":
        rexs = ["none"]
    else:
        rexs = ["none", "present"]
    if f.modrm in ("none", "plus_r"):
        addrs = [None]
    else:
        # M5c: "reg_alias" = register form with the reg field naming the same register as rm (xor r,r / cmp r,r shapes)
        alias = ["reg_alias"] if f.modrm == "rm_r" and _dist() == "m5c" else []
        addrs = ((["reg"] + alias) if f.rm in ("r", "any") else []) \
            + (list(ADDR_CLASSES) if f.rm in ("m", "any") else [])
    return [(c, r, a) for c in ccs for r in rexs for a in addrs]


def shape_of(f: Form, key, mem, cc):
    """The shape cell an encoding key falls in (oracle side)."""
    rex = "present" if _has_rex(key, f) else "none"
    if f.modrm in ("none", "plus_r"):
        return (cc, rex, None)
    if mem is None:
        if f.modrm == "rm_r":
            at = len(f.prefix) + (1 if rex == "present" else 0) + len(f.opcode or b"x") + (1 if f.special in ("setcc", "cmov") else 0)
            if at < len(key):
                rx = key[len(f.prefix)] if rex == "present" else 0
                mm = key[at]
                if ((mm >> 3) & 7 | (rx & 4) << 1) == (mm & 7 | (rx & 1) << 3):
                    return (cc, rex, "reg_alias")
        return (cc, rex, "reg")
    if mem["base"] == "rip":
        return (cc, rex, "rip")
    mod = mem["mod"]
    sib = len(key) > len(f.prefix) + (1 if rex == "present" else 0) + len(f.opcode or b"x") + 1 + (1 if f.special in ("setcc", "cmov") else 0)
    if not sib:
        return (cc, rex, ("base", "base_d8", "base_d32")[mod])
    if mem["base"] is None:
        return (cc, rex, "sib_nobase_noindex" if mem["index"] is None else "sib_nobase")
    return (cc, rex, ("sibni_d0", "sibni_d8", "sibni_d32")[mod] if mem["index"] is None else ("sib_d0", "sib_d8", "sib_d32")[mod])


# ------------------------------------------------------------------ Unicorn


class Oracle:
    def __init__(self):
        import unicorn
        from unicorn import x86_const as X
        self.U, self.X = unicorn, X
        self.reg_ids = [getattr(X, f"UC_X86_REG_{n}") for n in GPRS]

    def step(self, regs, rip, flags, window, seed, mode, overrides):
        """Single-step one instruction. Returns (after_regs, after_rip, after_flags, changed {addr: byte}, trap, error)."""
        U, X = self.U, self.X
        uc = U.Uc(U.UC_ARCH_X86, U.UC_MODE_64)
        mapped = set()
        ov = dict(overrides)
        for i, b in enumerate(window):
            ov[(rip + i) & MASK64] = b

        def page_bytes(page):
            addr = np.arange(page, page + 4096, dtype=np.uint64)[None, :]
            data = hash_bytes(np.array([seed]), np.array([mode]), addr)[0]
            for a, v in ov.items():
                if page <= a < page + 4096:
                    data[a - page] = v
            return data.tobytes()

        def ensure(addr, size):
            for page in range(addr & ~0xFFF, ((addr + max(size, 1) - 1) & ~0xFFF) + 1, 0x1000):
                if page not in mapped:
                    uc.mem_map(page, 0x1000)
                    uc.mem_write(page, page_bytes(page))
                    mapped.add(page)

        def on_unmapped(uc_, access, addr, size, value, data):
            if addr > CANON:
                return False
            ensure(addr, size)
            return True

        writes = []
        trap = []

        def on_write(uc_, access, addr, size, value, data):
            writes.append((addr, size))

        def on_syscall(uc_, data):
            trap.append(1)
            uc_.emu_stop()

        ensure(rip, WINDOW + 16)
        uc.hook_add(U.UC_HOOK_MEM_UNMAPPED, on_unmapped)
        uc.hook_add(U.UC_HOOK_MEM_WRITE, on_write)
        uc.hook_add(U.UC_HOOK_INSN, on_syscall, None, 1, 0, X.UC_X86_INS_SYSCALL)
        for rid, v in zip(self.reg_ids, regs):
            uc.reg_write(rid, v)
        uc.reg_write(X.UC_X86_REG_EFLAGS, EFLAGS_FIXED | sum(f << b for f, b in zip(flags, FLAG_BITS)))
        before = {}
        try:
            uc.emu_start(rip, (rip + 0x10000) & MASK64, count=1)
        except U.UcError as e:
            return None, None, None, None, 0, str(e)
        changed = {}
        for addr, size in writes:
            for a in range(addr, addr + size):
                old = ov.get(a)
                if old is None:
                    old = int(hash_bytes(np.array([seed]), np.array([mode]), np.array([[a]], dtype=np.uint64))[0, 0])
                new = uc.mem_read(a, 1)[0]
                if new != old:
                    changed[a] = new
        ef = uc.reg_read(X.UC_X86_REG_EFLAGS)
        after_rip = uc.reg_read(X.UC_X86_REG_RIP)
        if trap:
            after_rip = (rip + 2) & MASK64  # the OS boundary owns SYSCALL; the step stops at the instruction
        return ([uc.reg_read(r) & MASK64 for r in self.reg_ids], after_rip & MASK64,
                [(ef >> b) & 1 for b in FLAG_BITS], changed, int(bool(trap)), None)


# ------------------------------------------------------------------ rows


def _canon(rng):
    return rng.randrange(0x10000, CANON - 0x100000) & ~0x7 | rng.randrange(8) * (rng.random() < 0.3)


def random_value(rng, special):
    if special:
        if rng.random() < 0.4:  # near a carry boundary of a byte / word / dword / qword part
            b = rng.choice((8, 16, 32, 64))
            v = ((1 << b) - rng.randrange(1, 17)) if rng.random() < 0.5 else ((1 << b) + rng.randrange(16))
            return (rng.getrandbits(64) & ~((1 << b) - 1) | (v & ((1 << b) - 1))) & MASK64 if b < 64 else v & MASK64
        return rng.choice([0, 1, 2, rng.randrange(256), MASK64, 1 << 63, (1 << 63) - 1, 0xFFFFFFFF, 0x80000000, 0x7FFFFFFF,
                           rng.randrange(1 << 32)])
    return rng.getrandbits(64)


EDGE_P = 0.1  # M5c boundary-state rows (fixed, not tuned)


def _dist():
    """Row distribution: "m5c" (default: boundary-state rows at EDGE_P, half the shift counts from the full byte range)
    or "legacy" (the M4/M5a/M5b generator, bit for bit). Read from ISA_ROW_DIST so forkserver workers see it."""
    import os
    return os.environ.get("ISA_ROW_DIST", "m5c")


def _set_dist(dist):
    """Generator workers take the caller's distribution with their job (a forkserver child keeps the server's env)."""
    import os
    os.environ["ISA_ROW_DIST"] = dist
EDGE_KINDS = ("equal", "zero", "ones", "sign", "negate")


def _edge_slots(f, regs, mem):
    """Operand slots a boundary state may set: data registers (never RSP or an address register), the immediate, the
    memory operand."""
    prot = {4}
    if mem:
        prot |= {mem.get("base"), mem.get("index")}
    out = []
    for role in ("reg", "rm"):
        r = regs.get(role)
        if isinstance(r, int) and r not in prot and not (role == "rm" and mem):
            out.append(("reg", r))
    if f.imm:
        out.append(("imm", None))
    if mem:
        out.append(("mem", None))
    return list(dict.fromkeys(out))  # an aliased reg/rm register is one slot


def _edge_pick(rng, f, regs, mem):
    """A boundary-state plan for one row, or None: {"kind", "src", "tk", "t", "v"}. Generic over forms: equal operands,
    an operand 0 / all-ones / at the sign boundary, or one operand the negation of another (their sum is 0)."""
    if f.width not in (8, 16, 32, 64) or f.special in ("div", "idiv", "ret", "target", "leave") or rng.random() >= EDGE_P:
        return None
    slots = _edge_slots(f, regs, mem)
    if not slots:
        return None
    kind = rng.choice(EDGE_KINDS)
    w = f.width
    if kind in ("equal", "negate"):
        tg = [x for x in slots if x[0] in ("reg", "mem")]
        if len(slots) < 2 or not tg:
            return None
        t = rng.choice(tg)
        # the source is a register or the immediate: a memory source's bytes are drawn after the plan is applied
        srcs = [x for x in slots if x != t and x[0] != "mem"]
        if not srcs:
            return None
        return {"kind": kind, "src": rng.choice(srcs), "tk": t[0], "t": t[1], "v": None}
    t = rng.choice(slots)
    v = 0 if kind == "zero" else (1 << w) - 1 if kind == "ones" else rng.choice((1 << (w - 1), (1 << (w - 1)) - 1))
    return {"kind": kind, "src": None, "tk": t[0], "t": t[1], "v": v}


def _sext_int(v, bits):
    return v - (1 << bits) if v >> (bits - 1) & 1 else v


def _edge_value(e, f, st, imm, imm_b):
    """The value the plan's target takes (None when the source is the memory operand)."""
    if e["v"] is not None:
        return e["v"]
    m = (1 << f.width) - 1
    src = e["src"]
    v = (_sext_int(imm, 8 * imm_b) & m) if src[0] == "imm" else (st[src[1]] & m) if src[0] == "reg" else None
    if v is None:
        return None
    return v if e["kind"] == "equal" else (-v) & m


def _edge_regs_imm(e, f, st, imm, imm_b, literal_ok):
    m = (1 << f.width) - 1
    st = list(st)
    if e["tk"] == "reg":
        v = _edge_value(e, f, st, imm, imm_b)
        if v is not None:
            st[e["t"]] = (st[e["t"]] & ~m) | v
    elif e["tk"] == "imm":
        nv = e["v"] & ((1 << (8 * imm_b)) - 1)
        if literal_ok is None or literal_ok(nv.to_bytes(imm_b, "little")):
            imm = nv
    return st, imm


def _edge_mem(e, f, st, imm, imm_b, ea):
    v = _edge_value(e, f, st, imm, imm_b)
    if v is None:
        return {}
    return {(ea + i) & MASK64: b for i, b in enumerate(v.to_bytes(f.width // 8, "little"))}


def make_rows(rng, oracle, form: Form, key, regs, mem, cc, nrows, literal_ok=None, rows_out=None):
    """nrows transitions of one encoding key with random disp/imm/rel and random states."""
    f = form
    out = rows_out if rows_out is not None else []
    tries = made = 0
    while made < nrows and tries < nrows * 6:
        tries += 1
        special = rng.random() < 0.3
        st = [random_value(rng, special and rng.random() < 0.6) for _ in range(16)]
        if special and rng.random() < 0.5:  # copies make equal operands (zero results, equal compares)
            a, b = rng.randrange(16), rng.randrange(16)
            st[a] = st[b]
        flags = [rng.randrange(2) for _ in FLAGS]
        rip = _canon(rng) & ~0xF | rng.randrange(16)
        seed = rng.getrandbits(62)
        mode = 1 if special and rng.random() < 0.5 else 0
        st[4] = _canon(rng)  # RSP canonical (stack ops); sometimes next to a 16-/32-bit carry boundary
        if rng.random() < 0.4:
            b = rng.choice((8, 16, 32))
            st[4] = (st[4] & ~((1 << b) - 1)) | (((1 << b) - 1 - rng.randrange(16)) if rng.random() < 0.5 else rng.randrange(16))
        if f.special in ("leave",):
            st[5] = _canon(rng)
        disp_b = mem["disp_bytes"] if mem else 0
        disp = rng.getrandbits(8 * disp_b) if disp_b else 0
        if disp_b and literal_ok is not None:
            while not literal_ok(disp.to_bytes(disp_b, "little")):
                disp = rng.getrandbits(8 * disp_b)
        imm_b = f.imm + f.rel
        imm = rng.getrandbits(8 * imm_b) if imm_b else 0
        m5c = _dist() == "m5c"
        # M5c: half the shift counts come from the full byte range (the count is masked by the CPU; 0 keeps the flags)
        if f.special == "shift":
            imm = rng.getrandbits(8) if m5c and rng.random() < 0.5 else rng.randrange(1, 64 if f.width == 64 else 32)
        if f.special == "shift_cl":
            st[1] = (st[1] & ~0xFF) | (rng.getrandbits(8) if m5c and rng.random() < 0.5 else rng.randrange(1, 64 if f.width == 64 else 32))
        edge = _edge_pick(rng, f, regs, mem) if m5c else None
        if edge is not None:
            st, imm = _edge_regs_imm(edge, f, st, imm, imm_b, literal_ok)
        if imm_b and literal_ok is not None and f.special != "shift":
            while not literal_ok(imm.to_bytes(imm_b, "little")):
                imm = rng.getrandbits(8 * imm_b)
        overrides = {}
        if mem:  # place the effective address on a random canonical address
            target = _canon(rng)
            sdisp = disp - (1 << (8 * disp_b)) if disp_b and disp >> (8 * disp_b - 1) else disp
            base, index, scale = mem["base"], mem["index"], mem["scale"]
            if index is not None:
                st[index] = rng.randrange(1 << 20) if rng.random() < 0.8 else rng.randrange(256)
            if base == "rip":
                pass
            elif base is None:
                pass
            elif index is not None and index == base:
                st[base] = (target - sdisp) // (1 + (1 << scale)) & MASK64
            else:
                idx = (st[index] << scale) if index is not None else 0
                st[base] = (target - sdisp - idx) & MASK64
                if base == 4 and f.special in ("ret",):
                    pass
        code = key + (disp.to_bytes(disp_b, "little") if disp_b else b"") + (imm.to_bytes(imm_b, "little") if imm_b else b"")
        ea = None
        if mem:
            sdisp = disp - (1 << (8 * disp_b)) if disp_b and disp >> (8 * disp_b - 1) else disp
            if mem["base"] == "rip":
                ea = rip + len(code) + sdisp
            else:
                ea = (st[mem["base"]] if mem["base"] is not None else 0) + ((st[mem["index"]] << mem["scale"]) if mem["index"] is not None else 0) + sdisp
            ea &= MASK64
        if edge is not None and mem and edge["tk"] == "mem":
            overrides.update(_edge_mem(edge, f, st, imm, imm_b, ea))
        if f.special in ("div", "idiv"):
            src = regs.get("rm")
            w = f.width
            m = (1 << w) - 1
            dv = rng.randrange(1, m) if rng.random() < 0.7 else rng.randrange(1, 1 << 16)
            if src is not None:
                st[src] = (st[src] & ~m) | dv if w == 32 else dv
                dv = st[src] & m
            else:
                overrides.update({(ea + i) & MASK64: b for i, b in enumerate(dv.to_bytes(w // 8, "little"))})
            if f.special == "div":
                hi = rng.randrange(dv) if dv else 0
                lo = rng.getrandbits(w)
            else:
                q = rng.randrange(-(1 << (w - 2)), 1 << (w - 2))
                sd = dv - (1 << w) if dv >> (w - 1) else dv
                if sd == 0:
                    continue
                r = rng.randrange(abs(sd)) * (1 if rng.random() < 0.5 else -1)
                if q * sd < 0 and r > 0 or q * sd > 0 and r < 0:
                    r = -r
                n = q * sd + r
                hi, lo = (n >> w) & m, n & m
            if src in (0, 2) or (mem and (mem["base"] in (0, 2) or mem["index"] in (0, 2))):
                continue
            st[0] = (st[0] & ~m) | lo if w == 32 else lo
            st[2] = (st[2] & ~m) | hi if w == 32 else hi
        if f.special == "ret":
            overrides.update({(st[4] + i) & MASK64: b for i, b in enumerate(_canon(rng).to_bytes(8, "little"))})
        if f.special == "target":
            t = _canon(rng)
            if "rm" in regs:
                st[regs["rm"]] = t
        if f.special == "target" and mem:  # memory-held target: override the 8 bytes at the effective address
            overrides.update({(ea + i) & MASK64: b for i, b in enumerate(_canon(rng).to_bytes(8, "little"))})
        window = code + bytes(rng.randrange(256) for _ in range(WINDOW - len(code)))
        res = oracle.step(st, rip, flags, window, seed, mode, overrides)
        if res[5] is not None:
            continue
        a_regs, a_rip, a_flags, changed, trap, _ = res
        made += 1
        out.append({"form": f.name, "key": key, "window": window, "regs": st, "rip": rip, "flags": flags, "seed": seed, "mode": mode,
                    "overrides": overrides, "edge": edge["kind"] if edge else "", "a_regs": a_regs, "a_rip": a_rip, "a_flags": a_flags, "changed": changed, "trap": trap,
                    "length": len(code), "regs_roles": regs, "cc": cc})
    return out


def pack(rows):
    u = lambda v: np.array(v, dtype=np.uint64)
    n = len(rows)
    ch_addr = np.zeros((n, 16), dtype=np.uint64)
    ch_val = np.zeros((n, 16), dtype=np.uint8)
    ch_n = np.zeros(n, dtype=np.int16)
    for i, r in enumerate(rows):
        items = sorted(r["changed"].items())[:16]
        ch_n[i] = len(r["changed"])
        for j, (a, v) in enumerate(items):
            ch_addr[i, j], ch_val[i, j] = a, v
    return {"window": np.frombuffer(b"".join(r["window"] for r in rows), np.uint8).reshape(n, WINDOW).copy(),
            "regs": u([r["regs"] for r in rows]), "rip": u([r["rip"] for r in rows]), "flags": np.array([r["flags"] for r in rows], np.uint8),
            "seed": u([r["seed"] for r in rows]), "mode": np.array([r["mode"] for r in rows], np.uint8),
            "a_regs": u([r["a_regs"] for r in rows]), "a_rip": u([r["a_rip"] for r in rows]), "a_flags": np.array([r["a_flags"] for r in rows], np.uint8),
            "ch_n": ch_n, "ch_addr": ch_addr, "ch_val": ch_val, "trap": np.array([r["trap"] for r in rows], np.uint8),
            # oracle-only columns (never given to the learner; used for per-form reporting and the grid)
            "o_form": np.array([r["form"] for r in rows]), "o_length": np.array([r["length"] for r in rows], np.int16),
            "o_key": np.array([r["key"].hex() for r in rows]),
            "o_edge": np.array([r.get("edge", "") for r in rows]),  # M5c boundary-state kind ("" = ordinary row)
            "overrides": [r["overrides"] for r in rows]}


# ------------------------------------------------------------------ splits


HELD_BYTES = tuple(sorted(random.Random(20261015).sample([b for b in range(256) if b not in (0, 1, 8, 16, 255)], 20)))


def make_grid(seed=20261015):
    """Held-out (form, role, register) cells, about half on R8-R15, plus the HELD_BYTES literal panel."""
    rng = random.Random(seed)
    roles = []
    picks = [("mov_r_rm64m", "reg"), ("mov_rm64m_r", "reg"), ("add_rm64r_i8", "rm"), ("add_rm64r_r", "reg"), ("add_rm64r_r", "rm"),
             ("lea_r64", "reg"), ("lea_r64", "base"), ("mov_r_rm64m", "base"), ("mov_r_rm64m", "index"), ("cmp_rm64r_r", "rm"),
             ("cmp_rm64m_i8", "base"), ("movzx_r32_rm8m", "reg"), ("push_r", "reg"), ("pop_r", "reg"), ("sub_rm64r_i8", "rm"),
             ("test_rm64r_r", "reg"), ("shr_rm64r_i", "rm"), ("imul_r_rm64r", "reg"), ("mov_rm8m_r", "reg"), ("xor_rm32r_r", "rm")]
    used = {}
    for i, (form, role) in enumerate(picks):
        hi = i % 2 == 0
        options = [r for r in (range(8, 16) if hi else range(8)) if r != 4 and (form, role, r) not in used]
        if role in ("base",):
            options = [r for r in options if r not in (4, 5, 12, 13)]
        r = rng.choice(options)
        used[(form, role, r)] = 1
        roles.append([form, role, r])
    return {"seed": seed, "cells": roles, "held_bytes": list(HELD_BYTES)}


def hits_cell(grid, form, regs):
    for f, role, r in grid["cells"]:
        if f == form.name and regs.get(role) == r:
            return True
    return False


def literal_ok_train(b):
    return not any(x in HELD_BYTES for x in b)


def _gen_ctx():
    """Pool context for the row generators: their jobs carry only seeds and key specs, so the workers come from a
    small forkserver process instead of forking a large learner heap (mrun sums tree RSS, which counts a forked
    child's inherited pages again). ISA_GEN_FORK=1 restores fork."""
    import multiprocessing as mp
    import os
    return mp.get_context("fork" if os.environ.get("ISA_GEN_FORK") == "1" else "forkserver")


def generate_stratified(keys_per_cell, seed, grid, forms=FORM_NAMES, workers=4, rows_per_key=8):
    """Blind coverage-uniform training rows: keys_per_cell keys in every encoding-shape cell of every form
    (condition code x REX presence x addressing class), random registers/values within the cell. Held-out cells and
    literal bytes stay excluded as in the train split."""
    import multiprocessing as mp
    jobs = [(forms[i::workers], keys_per_cell, seed * 1000 + i, grid, rows_per_key, _dist()) for i in range(workers)]
    if workers > 1:
        with _gen_ctx().Pool(workers) as pool:
            parts = pool.map(_gen_strat_chunk, jobs)
    else:
        parts = [_gen_strat_chunk(j) for j in jobs]
    rows = [r for p in parts for r in p]
    random.Random(seed).shuffle(rows)
    return pack(rows)


def _gen_strat_chunk(args):
    forms, kpc, seed, grid, rpk, dist = args
    _set_dist(dist)
    rng, oracle = random.Random(seed), Oracle()
    held = (lambda ff, rr: hits_cell(grid, ff, rr)) if grid else None
    rows = []
    for name in forms:
        f = FORMS[name]
        for cc, rex, addr in shape_cells(f):
            force = {"rex": rex}
            if cc is not None:
                force["cc"] = cc
            if addr:
                force["addr"] = addr
            for _ in range(kpc):
                try:
                    key, regs, mem, c2 = sample_key(rng, f, held, force)
                except RuntimeError:
                    continue
                make_rows(rng, oracle, f, key, regs, mem, c2, max(rpk, f.rows // 6), literal_ok=literal_ok_train, rows_out=rows)
    return rows


def cell_coverage(windows, forms_col=None):
    """Distinct encoding-shape cells hit by a set of instruction windows (oracle-side parse), out of all cells."""
    hit = set()
    for w in windows:
        got = identify(bytes(w))
        if got is None:
            continue
        name, key, regs, mem, cc = got
        hit.add((name,) + shape_of(FORMS[name], key, mem, cc))
    total = sum(len(shape_cells(FORMS[n])) for n in FORM_NAMES)
    return {"cells_hit": len(hit), "cells_total": total}


def generate(n_keys_per_form, seed, split="train", grid=None, forms=FORM_NAMES, workers=4, max_keys=None):
    import multiprocessing as mp
    jobs = [(forms[i::workers], n_keys_per_form, seed * 1000 + i, split, grid, max_keys, _dist()) for i in range(workers)]
    if workers > 1:
        with _gen_ctx().Pool(workers) as pool:
            parts = pool.map(_gen_chunk, jobs)
    else:
        parts = [_gen_chunk(j) for j in jobs]
    rows = [r for p in parts for r in p]
    random.Random(seed).shuffle(rows)
    return pack(rows)


def _gen_chunk(args):
    forms, n_keys, seed, split, grid, max_keys, dist = args
    _set_dist(dist)
    rng, oracle = random.Random(seed), Oracle()
    rows = []
    for name in forms:
        f = FORMS[name]
        if split == "train":
            held = (lambda ff, rr: hits_cell(grid, ff, rr)) if grid else None
            lit = literal_ok_train
        elif split == "iid":
            held = (lambda ff, rr: hits_cell(grid, ff, rr)) if grid else None
            lit = literal_ok_train
        elif split == "cells":
            if not any(c[0] == name for c in grid["cells"]):
                continue
            held = lambda ff, rr: not hits_cell(grid, ff, rr)
            lit = literal_ok_train
        elif split == "literal":
            if f.imm + f.rel == 0 and f.rm != "m":
                continue
            held = (lambda ff, rr: hits_cell(grid, ff, rr)) if grid else None
            lit = lambda b: any(x in HELD_BYTES for x in b)
        nk = n_keys if max_keys is None else min(n_keys, max_keys)
        for _ in range(nk):
            key, regs, mem, cc = sample_key(rng, f, held)
            if split == "literal" and f.imm + f.rel == 0 and not (mem and mem["disp_bytes"]):
                continue
            make_rows(rng, oracle, f, key, regs, mem, cc, f.rows if split == "train" else max(1, f.rows // 4),
                      literal_ok=lit, rows_out=rows)
    return rows


# ------------------------------------------------------------------ dev-program-guided form selection (oracle side)


def identify(window: bytes):
    """Oracle-side parse of an instruction's encoding key against the catalog: (form name, key, regs, mem, cc) or None.
    Used only to choose WHICH form/key to sample fresh random-state Unicorn transitions for; never for labels."""
    w = bytes(window)
    for name in FORM_NAMES:
        got = _parse(FORMS[name], w)
        if got is not None:
            return (name,) + got
    return None


def _parse(f: Form, w: bytes):
    pos = 0
    if f.prefix:
        if not w.startswith(f.prefix):
            return None
        pos = len(f.prefix)
    rex = 0
    if 0x40 <= w[pos] <= 0x4F:
        if f.rex == "none":
            return None
        rex = w[pos]
        pos += 1
    W, R, X, B = (rex >> 3) & 1, (rex >> 2) & 1, (rex >> 1) & 1, rex & 1
    w64 = f.width == 64 and f.rex not in ("noW", "none")
    if W != int(w64):
        return None
    cc = None
    if f.special in ("jcc8", "jcc32", "setcc", "cmov"):
        base = {"jcc8": b"\x70", "jcc32": b"\x0f\x80", "setcc": b"\x0f\x90", "cmov": b"\x0f\x40"}[f.special]
        op = w[pos:pos + len(base)]
        if op[:-1] != base[:-1] or (op[-1] & 0xF0) != base[-1]:
            return None
        cc = op[-1] & 0xF
        pos += len(base)
    elif f.modrm == "plus_r":
        if (w[pos] & 0xF8) != f.opcode[0]:
            return None
        reg = (w[pos] & 7) | (B << 3)
        if f.width == 8 and not _reg8_ok(reg, bool(rex)):
            return None
        return w[:pos + 1], {"reg": reg}, None, None
    else:
        if w[pos:pos + len(f.opcode)] != f.opcode:
            return None
        pos += len(f.opcode)
    if f.modrm == "none":
        if rex and f.rex == "noW" and (R or X or B):
            return None
        return w[:pos], {}, None, cc
    modrm = w[pos]
    pos += 1
    mod, reg3, rm3 = modrm >> 6, (modrm >> 3) & 7, modrm & 7
    if f.modrm == "digit" and reg3 != f.digit:
        return None
    if f.rm == "r" and mod != 3 or f.rm == "m" and mod == 3:
        return None
    regs, mem = {}, None
    reg = reg3 | (R << 3)
    if f.modrm == "rm_r":
        regs["reg"] = reg
    if mod == 3:
        regs["rm"] = rm3 | (B << 3)
    else:
        disp_bytes = {0: 0, 1: 1, 2: 4}[mod]
        riprel = False
        if rm3 == 4:
            sib = w[pos]
            pos += 1
            scale, index, base = sib >> 6, ((sib >> 3) & 7) | (X << 3), (sib & 7) | (B << 3)
            no_base = mod == 0 and (base & 7) == 5
            if no_base:
                disp_bytes = 4
            mem = {"base": None if no_base else base, "index": None if index == 4 else index, "scale": scale}
        elif rm3 == 5 and mod == 0:
            riprel, disp_bytes = True, 4
            mem = {"base": "rip", "index": None, "scale": 0}
        else:
            mem = {"base": rm3 | (B << 3), "index": None, "scale": 0}
        mem.update(mod=mod, disp_bytes=disp_bytes, riprel=riprel)
        regs.update({"base": mem["base"], "index": mem["index"]})
    if f.width == 8 or f.special == "setcc" or f.special == "src8":
        need8 = []
        if f.modrm == "rm_r" and f.special != "src8":
            need8.append(reg)
        if mod == 3:
            need8.append(regs["rm"])
        if any(not _reg8_ok(c, bool(rex)) for c in need8):
            return None
    return w[:pos], regs, mem, cc


def generate_targeted(targets, seed, grid, workers=4, nearby_keys=6):
    """Fresh random-state Unicorn rows for chosen (form, key) targets plus a few random nearby keys of each form.
    targets: [(form name, key, regs, mem, cc, n_rows)]. Held-out cells / literal bytes stay excluded as in train."""
    import multiprocessing as mp
    jobs = [(targets[i::workers], seed * 1000 + i, grid, nearby_keys, _dist()) for i in range(workers)]
    if workers > 1 and len(targets) > 1:
        with _gen_ctx().Pool(workers) as pool:
            parts = pool.map(_gen_targeted_chunk, jobs)
    else:
        parts = [_gen_targeted_chunk(j) for j in jobs]
    rows = [r for p in parts for r in p]
    random.Random(seed).shuffle(rows)
    return pack(rows) if rows else None


def _gen_targeted_chunk(args):
    targets, seed, grid, nearby, dist = args
    _set_dist(dist)
    rng, oracle = random.Random(seed), Oracle()
    held = (lambda ff, rr: hits_cell(grid, ff, rr)) if grid else None
    rows = []
    for name, key, regs, mem, cc, n in targets:
        f = FORMS[name]
        if held is None or not held(f, regs):
            make_rows(rng, oracle, f, key, regs, mem, cc, n, literal_ok=literal_ok_train, rows_out=rows)
        for _ in range(nearby):
            k2, r2, m2, c2 = sample_key(rng, f, held)
            make_rows(rng, oracle, f, k2, r2, m2, c2, f.rows, literal_ok=literal_ok_train, rows_out=rows)
    return rows


def sibling_targets(name, key, regs, mem, cc, grid, n_keys, n_rows, rng, tries=40000):
    """Training-legal keys of the same form and addressing shape as a held-out-cell key, differing only in the held
    role's register (the held cell itself is never sampled). Lets a program that executes a held cell test whether the
    machine generalizes to it from its neighbours."""
    f = FORMS[name]
    held = lambda ff, rr: hits_cell(grid, ff, rr)
    roles = [role for ff, role, r in grid["cells"] if ff == name and regs.get(role) == r]
    shape = None if mem is None else (mem.get("mod"), mem.get("scale"), mem.get("disp_bytes"), mem.get("riprel"),
                                      mem.get("base") is None, mem.get("index") is None, mem.get("base") == mem.get("index"))
    out, seen = [], {key}
    for _ in range(tries):
        if len(out) >= n_keys:
            break
        k2, r2, m2, c2 = sample_key(rng, f, held)
        if k2 in seen or c2 != cc:
            continue
        s2 = None if m2 is None else (m2.get("mod"), m2.get("scale"), m2.get("disp_bytes"), m2.get("riprel"),
                                      m2.get("base") is None, m2.get("index") is None, m2.get("base") == m2.get("index"))
        if s2 != shape:
            continue
        if any(r2.get(k) != regs.get(k) for k in ("reg", "rm") if k not in roles):
            continue
        if any(role in ("base", "index") for role in roles) and any(r2.get(k) != regs.get(k) for k in ("base", "index") if k not in roles
                                                                     and regs.get("base") != regs.get("index")):
            continue
        seen.add(k2)
        out.append((name, k2, r2, m2, c2, n_rows))
    return out
