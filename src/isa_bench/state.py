"""Reference traces, memory images and per-step comparison.

The reference for every program is the native ptrace trace written by
``programs/tools/trace_full.c``: before every instruction, RIP, the sixteen GPRs
and EFLAGS as 18 little-endian u64, plus the initial memory image of the process.
Every model starts from ``steps[0]`` and that image, and its state after
instruction ``i`` is compared with ``steps[i + 1]``.
"""
from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from pathlib import Path

MASK64 = (1 << 64) - 1
GPRS = ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI",
        "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15")
FLAG_BITS = {"CF": 0, "PF": 2, "AF": 4, "ZF": 6, "SF": 7, "OF": 11}
TF = 0x100
R11 = GPRS.index("R11")


ALL_FLAGS = frozenset(FLAG_BITS)
# Intel SDM flag effects per mnemonic stem: (flags the instruction defines, flags it leaves undefined).
# Undefined flags are not compared until a later instruction defines them again; a mismatch on them is
# hardware-specific, not an execution error (Unicorn vs the 4080 host differ on IMUL's SF/ZF/AF/PF).
_LOGIC = (frozenset({"CF", "PF", "ZF", "SF", "OF"}), frozenset({"AF"}))
_SHIFT = (frozenset({"CF", "PF", "ZF", "SF"}), frozenset({"AF", "OF"}))
FLAG_EFFECTS = {
    **{m: (ALL_FLAGS, frozenset()) for m in ("add", "adc", "sub", "sbb", "cmp", "neg", "xadd", "cmpxchg", "popcnt",
                                             "popf", "comiss", "comisd", "ucomiss", "ucomisd", "cmps", "scas")},
    **{m: (frozenset({"PF", "AF", "ZF", "SF", "OF"}), frozenset()) for m in ("inc", "dec")},
    **{m: _LOGIC for m in ("and", "or", "xor", "test")},
    **{m: (frozenset({"CF", "OF"}), frozenset({"PF", "AF", "ZF", "SF"})) for m in ("imul", "mul")},
    **{m: (frozenset(), ALL_FLAGS) for m in ("div", "idiv")},
    **{m: _SHIFT for m in ("shl", "sal", "shr", "sar", "shld", "shrd")},
    **{m: (frozenset({"CF"}), frozenset({"OF"})) for m in ("rol", "ror", "rcl", "rcr")},
    **{m: (frozenset({"ZF"}), ALL_FLAGS - {"ZF"}) for m in ("bsf", "bsr")},
    **{m: (frozenset({"CF"}), frozenset({"PF", "AF", "SF", "OF"})) for m in ("bt", "bts", "btr", "btc")},
    **{m: (frozenset({"CF", "ZF"}), frozenset({"PF", "AF", "SF", "OF"})) for m in ("lzcnt", "tzcnt")},
    **{m: (frozenset({"CF"}), frozenset()) for m in ("clc", "stc", "cmc")},
    "sahf": (frozenset({"CF", "PF", "AF", "ZF", "SF"}), frozenset()),
    "andn": (frozenset({"CF", "ZF", "SF", "OF"}), frozenset({"AF", "PF"})),
}


def flag_effects(mnemonic: str | None):
    if not mnemonic:
        return None
    m = mnemonic.split()[-1] if mnemonic.startswith(("rep", "lock")) else mnemonic
    if m in FLAG_EFFECTS:
        return FLAG_EFFECTS[m]
    if m[-1:] in "bwlq" and m[:-1] in FLAG_EFFECTS:
        return FLAG_EFFECTS[m[:-1]]
    return None


def parse_disasm(text: str) -> dict:
    """objdump -d --no-show-raw-insn -> {address: mnemonic} (prefixes kept, e.g. "rep stos")."""
    out = {}
    for line in text.splitlines():
        if line[:1] != " " or ":\t" not in line:
            continue
        addr, _, insn = line.partition(":\t")
        words = insn.split()
        if not words:
            continue
        mnem = words[0]
        if mnem in ("rep", "repz", "repnz", "repe", "repne", "lock", "notrack", "bnd") and len(words) > 1:
            mnem = f"{mnem} {words[1]}"
        try:
            out[int(addr.strip(), 16)] = mnem
        except ValueError:
            continue
    return out


def flags_of(eflags: int) -> dict:
    return {name: (int(eflags) >> bit) & 1 for name, bit in FLAG_BITS.items()}


def eflags_of(flags: dict, rest: int = 0x202) -> int:
    value = int(rest) & ~sum(1 << b for b in FLAG_BITS.values())
    for name, bit in FLAG_BITS.items():
        value |= (int(flags.get(name, 0)) & 1) << bit
    return value


@dataclass
class ArchState:
    """Architectural state as the harness compares it. Missing registers are ``None``."""

    rip: int
    regs: list            # 16 entries in GPRS order; None where the model has no such register
    flags: dict           # subset of FLAG_BITS names -> bit

    def to_json(self) -> dict:
        return {"rip": hex(self.rip),
                "regs": {GPRS[i]: (None if v is None else hex(v)) for i, v in enumerate(self.regs)},
                "flags": dict(self.flags)}


def state_from_row(row) -> ArchState:
    row = [int(v) for v in row]
    return ArchState(rip=row[0], regs=row[1:17], flags=flags_of(row[17]))


class Trace:
    """Native reference trace of one program (``trace.npz`` with ``steps`` [N, 18] uint64)."""

    def __init__(self, program_dir: Path):
        import numpy as np

        self.dir = Path(program_dir)
        self.steps = np.load(self.dir / "trace.npz")["steps"]
        self.native = json.loads((self.dir / "native.json").read_text())
        dis = self.dir / "disasm.txt"
        self.mnemonics = parse_disasm(dis.read_text()) if dis.exists() else {}

    def mnemonic(self, i: int) -> str | None:
        return self.mnemonics.get(int(self.steps[i][0])) if 0 <= i < len(self) else None

    def __len__(self) -> int:
        return int(self.steps.shape[0])

    def state(self, i: int) -> ArchState:
        return state_from_row(self.steps[i])

    @property
    def initial_eflags(self) -> int:
        return int(self.steps[0][17]) & ~TF

    @property
    def expected_stdout(self) -> bytes:
        return bytes.fromhex(self.native["stdout_hex"])

    @property
    def expected_exit(self) -> int:
        return int(self.native["exit_code"])


def load_image(path: Path):
    """Records ``u64 start, u64 length, u8 perms[4], bytes[length]`` -> (pages, perms) at 4 KiB granularity."""
    pages, perms = {}, {}
    data = Path(path).read_bytes()
    off = 0
    while off < len(data):
        start, length = struct.unpack_from("<QQ", data, off)
        perm = data[off + 16:off + 20].decode()
        body = data[off + 20:off + 20 + length]
        off += 20 + length
        for p in range(0, length, 4096):
            pages[start + p] = bytearray(body[p:p + 4096])
            perms[start + p] = perm
    return pages, perms


def image_regions(path: Path):
    """Records as (start, length, perms, bytes) for runtimes that map whole regions."""
    out = []
    data = Path(path).read_bytes()
    off = 0
    while off < len(data):
        start, length = struct.unpack_from("<QQ", data, off)
        perm = data[off + 16:off + 20].decode()
        out.append((start, length, perm, data[off + 20:off + 20 + length]))
        off += 20 + length
    return out


class PageMemory:
    """Flat little-endian page store from the native image; unmapped access records a fault."""

    def __init__(self, path: Path):
        self.pages, self.perms = load_image(path)
        self.fault = None

    def _page(self, addr: int):
        page = self.pages.get(addr & ~0xFFF)
        if page is None and self.fault is None:
            self.fault = {"address": hex(addr)}
        return page

    def read(self, addr: int, n: int) -> bytes:
        out = bytearray()
        for k in range(n):
            a = (addr + k) & MASK64
            page = self._page(a)
            if page is None:
                return bytes(out)
            out.append(page[a & 0xFFF])
        return bytes(out)

    def write(self, addr: int, data: bytes) -> bool:
        for k, b in enumerate(data):
            a = (addr + k) & MASK64
            page = self._page(a)
            if page is None:
                return False
            page[a & 0xFFF] = b
        return True

    def fetch(self, rip: int, n: int = 15) -> bytes:
        out = bytearray()
        for k in range(n):
            a = (rip + k) & MASK64
            page = self.pages.get(a & ~0xFFF)
            if page is None:
                break
            out.append(page[a & 0xFFF])
        return bytes(out)


@dataclass
class Comparator:
    """Step-by-step comparison of a model's committed states with the native trace.

    ``gprs`` is how many leading GPRs the model represents (8 or 16) and ``flags``
    which flags it represents. R11's TF bit is masked: the ptrace tracer runs the
    program with TF set and SYSCALL copies RFLAGS into R11.
    """

    trace: Trace
    gprs: int = 16
    flags: tuple = ("CF", "ZF", "SF", "OF")
    exact_prefix: int = 0
    exact_total: int = 0
    compared: int = 0
    first_divergence: dict | None = None
    masked_flag_diffs: int = 0          # mismatches on architecturally undefined flags (not divergences)
    undefined: set = field(default_factory=set)
    _prefix_open: bool = True
    divergence_inputs: dict | None = field(default=None, repr=False)

    def expected(self, i: int) -> ArchState | None:
        return self.trace.state(i) if 0 <= i < len(self.trace) else None

    def diff(self, got: ArchState, want: ArchState) -> dict:
        out = {}
        if got.rip != want.rip:
            out["RIP"] = [hex(got.rip), hex(want.rip)]
        for r in range(self.gprs):
            g, w = got.regs[r], want.regs[r]
            if r == R11 and g is not None:
                g, w = g & ~TF, w & ~TF
            if g != w:
                out[GPRS[r]] = [None if g is None else hex(g), hex(w)]
        for name in self.flags:
            if got.flags.get(name) != want.flags.get(name):
                if name in self.undefined:
                    self.masked_flag_diffs += 1
                    continue
                out[name] = [got.flags.get(name), want.flags.get(name)]
        return out

    def observe(self, i: int, got: ArchState, *, window: bytes = b"", before: ArchState | None = None,
                extra: dict | None = None) -> bool:
        """Compare the state after executing instruction ``i``; returns True when exact."""
        want = self.expected(i + 1)
        if want is None:
            return False
        effects = flag_effects(self.trace.mnemonic(i))
        if effects is not None:  # native instruction i defines / undefines these flags
            self.undefined = (self.undefined - effects[0]) | effects[1]
        self.compared += 1
        delta = self.diff(got, want)
        if not delta:
            self.exact_total += 1
            if self._prefix_open:
                self.exact_prefix += 1
            return True
        self._prefix_open = False
        if self.first_divergence is None:
            self.first_divergence = {"step": i, "rip": hex(self.trace.state(i).rip), "window": window.hex(),
                                     "diff": delta, **(extra or {})}
            self.divergence_inputs = {"step": i, "window": window.hex(),
                                      "before": (before or self.trace.state(i)).to_json()}
        return False

    def summary(self) -> dict:
        return {"exact_prefix_steps": self.exact_prefix, "exact_steps": self.exact_total,
                "compared_steps": self.compared, "native_steps": len(self.trace),
                "exact_prefix_fraction": self.exact_prefix / max(1, len(self.trace) - 1),
                "first_divergence": self.first_divergence, "divergence_inputs": self.divergence_inputs,
                "masked_undefined_flag_diffs": self.masked_flag_diffs,
                "compared_gprs": self.gprs, "compared_flags": list(self.flags)}
