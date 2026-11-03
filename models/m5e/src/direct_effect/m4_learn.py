"""M4 structured learner: FieldLM microcode records over an SPF datapath, induced from Unicorn transitions only.

The learner never imports the oracle (`m4_isa`) except for the shared page-store function. It sees, per training
row, the before-state (15 window bytes at RIP, 16 GPRs, RIP, six status flags, a readable page store) and the
after-state (16 GPRs, RIP, six flags, the changed memory bytes, a trap bit). It learns:

  datapath  per-bit automata (lane A induction), executed on SPF `monomial_scan` and, bit-identically (checked),
            through byte-chunk compositions of the same tables; the one table used for every address/RIP sum
  decode    records keyed by instruction bytes (a byte trie, smallest key first), trailing runs (displacement /
            immediate) at learned positions, operand fields (contiguous bit runs, optionally extended by one bit
            of another byte), one global code -> register map and one code -> shift map
  effects   each changed location = a small expression from the supplied menu (`MENU`), found by enumeration
            with observational equivalence and verified on every row; flags = learned taps over a signal
            vocabulary; conditions = learned predicates over flags

Nothing here is an x86 table: there is no ModRM/SIB decoder, no condition-code table and no flag formula.
Everything x86-specific is induced and then checked by the scoring code against Unicorn and native ptrace.
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter, defaultdict

import numpy as np

from . import spf_datapath as D
from .m4_isa import FLAGS, MASK64, PageStore, hash_bytes

U64 = np.uint64
NF = len(FLAGS)
CF_I = FLAGS.index("CF")
WIDTHS = (64, 32, 16, 8)
MASKS = {8: 0xFF, 16: 0xFFFF, 32: 0xFFFFFFFF, 64: MASK64}

MENU = [
    ("record", "an instruction is a byte string; a record is keyed by its leading bytes (some bits may be operand fields or don't-care); the shortest key that fits wins (byte trie, MDL)"),
    ("run", "an immediate/displacement is a run of 1, 2, 4 or 8 bytes directly after the key; at most two runs (displacement then immediate); a run is read zero- or sign-extended"),
    ("field", "an operand field is a contiguous run of 2-3 bits inside one key byte, optionally extended by one bit of another key byte (top bit of the code)"),
    ("code-map", "one global code -> register map shared by every register field; one global code -> shift map for scale fields"),
    ("state", "machine state = 16 64-bit registers, RIP, six status flags; register parts are the low 8/16/32/64 bits (no high-byte registers)"),
    ("write", "a register write has a learned width (8/16/32/64) and merge rule (keep the rest, or zero it); a memory write has a learned address and byte count"),
    ("address", "an address is a sum of terms: base register, index register shifted left by a learned 0-3, one run, the fall-through address, or a learned constant; every sum uses one learned automaton (the adder)"),
    ("fall-through", "fall-through = RIP + learned length, computed by the adder; RIP update = select(condition, target, fall-through) where target may be fall-through + run, a register or a loaded value"),
    ("condition", "a condition is a learned predicate (truth table) over at most three status flags; it can also be written as a 0/1 value or select between two values"),
    ("alu", "a binary operation is a per-bit automaton (<= 2 states, start state chosen by incoming CF) read on the {0, pi} grid (lane A), with operands from this menu"),
    ("constant", "an operand may be a learned constant (any value consistent across rows)"),
    ("extend", "a value may be zero- or sign-extended from 8/16/32 bits, or be the sign fill of a register part"),
    ("shift", "a bit gather: result bit i = operand bit i -/+ count, filled with 0, the top bit, or wrapped; count from a run, the low byte of a register, or 1; ADDED (M5c): the count may be ANDed with a learned constant mask (31 or 63, chosen by the rows; raw stays a candidate). Reason: M5a diag rows with CL >= width were wrong (raw count shifted everything out)"),
    ("mul-div", "multiply (low or high half, signed or unsigned) and divide (quotient or remainder of a two-register dividend) as micro-loops of the learned adder/subtractor"),
    ("flags", "each status flag is a learned truth table over at most three signals of the operation (result top/bit-4/low-byte-parity/nonzero, operand top/bit-4, scan state at the top bit and at the end, shifted-out bit, count==1, high-half nonzero / not sign fill, the old flag); REVISED (M5c): at most FOUR signals, and a shift's signals include count==0 (masked; the CPU then keeps the flags). Version spaces (M5): every consistent alternative table, every consistent flag-source reading (imul signedness; flag-only operations across widths/operands, at most 16, else abstain), and a table's unobserved entries make the machine abstain where they matter"),
    ("field-coverage", "ADDED (M4; revised M5): a record's field takes a code only with evidence -- the code was seen at the record's instruction length in rows matching the record, or its low bits were seen working (and never selecting another length) in a sibling record with a field at the same bit position and the same fixed bits in that byte, pooled across key lengths, byte positions and REX/non-REX siblings; any other code leaves the record closed (abstain). Reason: M4 fib-O2 step 26 (rm=100 claimed as a register) and M4 held-out bsearch-O2 step 9 (the same, in a non-REX record); REVISED (M5c): a code is evidence only where the record's own template is EXACT on rows of that code (same length is not enough: SIB index 100 = no index keeps the length), and a row of the record's length that the template gets wrong is evidence against its one unseen code"),
    ("memory", "memory is an external page store (supplied); the learner emits load addresses and store address/value; fetch of the instruction window at RIP is supplied"),
    ("os", "SYSCALL is a learned record of kind trap; the OS boundary (write/exit, RAX/RCX/R11 effects) is supplied"),
    ("register-condition", "ADDED (M5e): when no flag predicate explains a condition, it may read register tests (register == 0, register == 1) of ONE register (at most both tests, a 4-entry table; version space over registers as for flags). Reason: glibc census job-454b57b4777a, rep stos at step 48 of every static glibc program: a string-op step does nothing at a zero count and falls through after the last iteration"),
    ("current-rip", "ADDED (M5e): the RIP update may select the instruction's own address (the step repeats it) against the fall-through. Reason: as register-condition (REP string steps)"),
    ("select-step", "ADDED (M5e): a conditional write may select between the old value and the old value plus a learned constant (the adder), as it may between the old value and another operand. Reason: as register-condition (pointer and count updates of a REP step)"),
    ("flag-only-shift", "ADDED (M5e): a flags-only effect may read the signals of a right shift (count from a register's low byte or a run, raw or masked 31/63; one signal per flag) as well as of a table op. Reason: glibc census, bt at step 1245 (CF = the tested bit = the shifted value's bottom bit)"),
]

# ------------------------------------------------------------------ tables


class LUT:
    """Byte-chunk composition of a lane A Table (exact: the same transition/emission maps, 8 bits per lookup)."""

    def __init__(self, table: D.Table):
        self.t = table
        n = table.n
        assert n <= 2
        out = np.zeros((n, 256, 256), np.uint8)
        nxt = np.zeros((n, 256, 256), np.uint8)
        bef = np.zeros((n, 256, 256), np.uint8)
        a = np.arange(256)[:, None].repeat(256, 1)
        b = np.arange(256)[None, :].repeat(256, 0)
        trans = np.array(table.trans)
        emit = np.array(table.emit)
        order = range(7, -1, -1) if table.reverse else range(8)
        for q0 in range(n):
            q = np.full((256, 256), q0)
            for j in order:
                s = 2 * ((a >> j) & 1) + ((b >> j) & 1)
                bef[q0] |= (q.astype(np.uint8) << j)
                out[q0] |= (emit[s, q].astype(np.uint8) << j)
                q = trans[s, q]
            nxt[q0] = q
        self.out, self.nxt, self.bef = out, nxt, bef
        self.start = np.array(table.start)

    def run(self, a, b, cf, w, signals=False):
        a = np.asarray(a, U64)
        b = np.asarray(b, U64)
        shape = np.broadcast(a, b).shape
        a = np.broadcast_to(a, shape)
        b = np.broadcast_to(b, shape)
        q = np.broadcast_to(self.start[np.asarray(cf, np.int64)], shape).astype(np.int64)
        res = np.zeros(shape, U64)
        nb = w // 8
        befs = [None] * nb
        chunks = range(nb - 1, -1, -1) if self.t.reverse else range(nb)
        for c in chunks:
            sh = U64(8 * c)
            ab = ((a >> sh) & U64(0xFF)).astype(np.int64)
            bb = ((b >> sh) & U64(0xFF)).astype(np.int64)
            res |= self.out[q, ab, bb].astype(U64) << sh
            befs[c] = self.bef[q, ab, bb]
            q = self.nxt[q, ab, bb].astype(np.int64)
        if not signals:
            return res
        top = w - 1
        sig = {"r_top": ((res >> U64(top)) & U64(1)).astype(np.int8), "a_top": ((a >> U64(top)) & U64(1)).astype(np.int8),
               "b_top": ((b >> U64(top)) & U64(1)).astype(np.int8), "q_top": ((befs[top // 8] >> (top % 8)) & 1).astype(np.int8),
               "q_final": q.astype(np.int8), "nz": (res != 0).astype(np.int8), "par8": _parity8(res),
               "a4": ((a >> U64(4)) & U64(1)).astype(np.int8), "b4": ((b >> U64(4)) & U64(1)).astype(np.int8),
               "r4": ((res >> U64(4)) & U64(1)).astype(np.int8)}
        return res, sig


def _parity8(x):
    v = (np.asarray(x, U64) & U64(0xFF)).astype(np.int64)
    p = np.zeros_like(v)
    for j in range(8):
        p ^= (v >> j) & 1
    return p.astype(np.int8)


def invert(lut: LUT, a, v, cf, w):
    """B such that table(a, B) == v, bit by bit (None where no or two b bits fit). Returns (b, ok)."""
    t = lut.t
    a = np.asarray(a, U64)
    v = np.asarray(v, U64)
    q = lut.start[np.asarray(cf, np.int64)].astype(np.int64) * np.ones(a.shape, np.int64)
    b = np.zeros(a.shape, U64)
    ok = np.ones(a.shape, bool)
    trans = np.array(t.trans)
    emit = np.array(t.emit)
    order = range(w - 1, -1, -1) if t.reverse else range(w)
    for i in order:
        ai = ((a >> U64(i)) & U64(1)).astype(np.int64)
        vi = ((v >> U64(i)) & U64(1)).astype(np.int64)
        e0 = emit[2 * ai, q]
        e1 = emit[2 * ai + 1, q]
        m0, m1 = e0 == vi, e1 == vi
        ok &= m0 ^ m1
        bi = m1.astype(np.int64)
        b |= bi.astype(U64) << U64(i)
        q = trans[2 * ai + bi, q]
    return b, ok


# ------------------------------------------------------------------ helpers


def sext(x, bits):
    x = np.asarray(x, U64) & U64((1 << bits) - 1)
    if bits >= 64:
        return x
    sign = U64(1 << (bits - 1))
    return (x ^ sign) - sign if True else x


def mask(x, w):
    return np.asarray(x, U64) & U64(MASKS[w])


def _sext_np(x, bits):
    x = np.asarray(x, U64) & U64((1 << bits) - 1)
    if bits >= 64:
        return x
    with np.errstate(over="ignore"):
        return (x ^ U64(1 << (bits - 1))) - U64(1 << (bits - 1))


def shift_gather(a, c, kind, w):
    """Bit gather on w bits: kind = (dir, fill) with dir 'l'/'r', fill 'z'/'t'(top)/'w'(wrap). Returns (result, spill)."""
    a = np.asarray(a, U64) & U64(MASKS[w])
    c = np.asarray(c, np.int64)
    out = np.zeros(np.broadcast(a, c).shape, U64)
    spill = np.zeros(out.shape, np.int8)
    d, fill = kind
    top = ((a >> U64(w - 1)) & U64(1)).astype(np.int64)
    for i in range(w):
        src = i - c if d == "l" else i + c
        if fill == "w":
            src = src % w
            bit = (a >> src.astype(U64)) & U64(1)
        else:
            inside = (src >= 0) & (src < w)
            bit = np.where(inside, (a >> np.clip(src, 0, w - 1).astype(U64)) & U64(1),
                           U64(0) if fill == "z" else top.astype(U64) * (src >= w) if d == "r" else U64(0))
        out |= bit.astype(U64) << U64(i)
    # last bit shifted out
    if d == "l":
        pos = (w - c) % w if fill == "w" else w - c
    else:
        pos = (c - 1) % w if fill == "w" else c - 1
    inside = (pos >= 0) & (pos < w)
    spill = np.where(inside, (a >> np.clip(pos, 0, w - 1).astype(U64)) & U64(1), 0).astype(np.int8)
    if fill == "w":
        spill = (((out >> U64(0)) & U64(1)) if d == "l" else ((out >> U64(w - 1)) & U64(1))).astype(np.int8)
    return out & U64(MASKS[w]), spill


# ------------------------------------------------------------------ the machine (library + records)


class Machine:
    def __init__(self):
        self.tables = []  # D.Table
        self.luts = []
        self.adder = None
        self.records = []  # dicts
        self.code_map = {}
        self.shift_map = {}
        self.stats = Counter()

    def add_table(self, t):
        for i, x in enumerate(self.tables):
            if x.key() == t.key():
                return i
        self.tables.append(t)
        self.luts.append(LUT(t))
        return len(self.tables) - 1


# ------------------------------------------------------------------ evaluation context


class Ctx:
    """Rows of one group under one layout (key length k, run sizes). All values are uint64 arrays over rows."""

    def __init__(self, m: Machine, data, rows, k, runs):
        self.m = m
        self.rows = np.asarray(rows)
        g = {key: data[key][self.rows] for key in ("window", "regs", "rip", "flags", "a_regs", "a_rip", "a_flags", "ch_n", "ch_addr", "ch_val", "trap")}
        self.g = g
        self.n = len(self.rows)
        self.k, self.runs = k, tuple(r for r in runs)
        self.L = k + sum(runs)
        self.store = data["store"].take(self.rows)
        win = g["window"].astype(U64)
        self.run_vals = []
        pos = k
        for size in runs:
            if size == 0:
                self.run_vals.append(None)
                continue
            v = np.zeros(self.n, U64)
            for j in range(size):
                v |= win[:, pos + j] << U64(8 * j)
            self.run_vals.append((v, size))
            pos += size
        self.cf = g["flags"][:, CF_I].astype(np.int64)
        self._next = None
        self._addr = None
        self._loads = {}

    @property
    def next(self):
        if self._next is None:
            self._next = self.m.luts[self.m.adder].run(self.g["rip"], U64(self.L), 0, 64)
        return self._next

    def add(self, a, b):
        return self.m.luts[self.m.adder].run(a, b, 0, 64)

    @property
    def pin(self):
        """Predicate inputs (flags + M5e register tests), cached per context."""
        if getattr(self, "_pin", None) is None:
            self._pin = pred_columns(self.g["flags"], self.g["regs"])
        return self._pin


def ev(m, e, x, old=None, want_sig=False):
    """Evaluate expression e over the context (vectorized). old = the target's old value (for ['old'])."""
    op = e[0]
    if op == "reg":
        _, r, pw, ext = e
        v = x.g["regs"][:, r] & U64(MASKS[pw])
        return _sext_np(v, pw) if ext == "s" else v
    if op == "run":
        v, size = x.run_vals[e[1]]
        return _sext_np(v, 8 * size) if e[2] == "s" else v
    if op == "cmask":  # M5c: an operand ANDed with a learned constant mask (shift counts: 31 / 63)
        return ev(m, e[1], x, old) & U64(e[2])
    if op == "const":
        return np.full(x.n, e[1] & MASK64, U64)
    if op == "fill":
        _, r, pw = e
        top = (x.g["regs"][:, r] >> U64(pw - 1)) & U64(1)
        return np.where(top == 1, U64(MASK64), U64(0))
    if op == "next":
        return x.next
    if op == "old":
        return old
    if op == "addr":
        return eval_addr(m, e[1], x)
    if op == "load":
        a = eval_addr(m, e[1], x)
        v = x.store.read(a, e[2])
        return _sext_np(v, 8 * e[2]) if e[3] == "s" else v
    if op == "table":
        _, t, A, B, w = e
        a = mask(ev(m, A, x, old), w)
        b = mask(ev(m, B, x, old), w)
        return m.luts[t].run(a, b, x.cf, w, signals=want_sig)
    if op == "shift":
        _, kind, A, C, w = e
        a = ev(m, A, x, old)
        c = (ev(m, C, x, old) & U64(0xFF)).astype(np.int64)
        res, spill = shift_gather(a, c, tuple(kind), w)
        if want_sig:
            return res, shift_signals(a, res, spill, c, w)
        return res
    if op == "mul":
        _, A, B, w, signed, part = e
        a = mask(ev(m, A, x, old), w)
        b = mask(ev(m, B, x, old), w)
        lo, hi = mul_wide(a, b, w, signed)
        res = lo if part == "lo" else hi
        if want_sig:
            return res, mul_signals(lo, hi, w)
        return res
    if op == "div":
        _, HI, LO, B, w, signed, part = e
        hi = mask(ev(m, HI, x, old), w)
        lo = mask(ev(m, LO, x, old), w)
        b = mask(ev(m, B, x, old), w)
        q, r = divmod_wide(hi, lo, b, w, signed)
        res = q if part == "q" else r
        if want_sig:
            return res, {}
        return res
    if op == "cur":  # M5e: the instruction's own address
        return x.g["rip"].astype(U64)
    if op == "pred":
        return eval_pred(e, x.pin).astype(U64)
    if op == "select":
        p = eval_pred(e[1], x.pin)
        return np.where(p == 1, ev(m, e[2], x, old), ev(m, e[3], x, old))
    raise ValueError(op)


def eval_addr(m, a, x):
    _, base, index, scale, disp = a
    terms = []
    if base == "next":
        terms.append(x.next)
    elif base is not None:
        terms.append(x.g["regs"][:, base])
    if index is not None:
        terms.append((x.g["regs"][:, index] << U64(scale)))
    if disp is not None:
        terms.append(ev(m, disp, x))
    if not terms:
        return np.zeros(x.n, U64)
    s = terms[0]
    for t in terms[1:]:
        s = x.add(s, t)
    return s


def pred_columns(flags, regs):
    """M5e predicate inputs: the six flags, then for every register r the tests (r == 0) at NF + 2r and (r == 1) at
    NF + 2r + 1. Indices < NF are the flags exactly as before, so every flag predicate reads the same columns."""
    flags = np.asarray(flags, np.int64)
    if regs is None:
        return flags
    regs = np.asarray(regs, U64)
    ext = np.stack([(regs[:, r] == U64(c)).astype(np.int64) for r in range(16) for c in (0, 1)], 1)
    return np.concatenate([flags, ext], 1)


def eval_pred(p, flags):
    fs, table = p[1], p[2]
    idx = np.zeros(len(flags), np.int64)
    for f in fs:
        idx = idx * 2 + flags[:, f].astype(np.int64)
    tab = np.array(table, np.int64)
    return tab[idx]


def mul_wide(a, b, w, signed):
    """(lo, hi) of the 2w-bit product, vectorized. This is the micro-loop of the learned adder (sum of shifted partial
    products) evaluated with machine integers; it is used only after `choose_adder` has checked that the learned
    adder equals integer addition (`Machine.adder_ok`)."""
    a = np.asarray(a, U64) & U64(MASKS[w])
    b = np.asarray(b, U64) & U64(MASKS[w])
    a, b = np.broadcast_arrays(a, b)
    with np.errstate(over="ignore"):
        if w <= 32:
            p = a * b
            lo, hi = p & U64(MASKS[w]), (p >> U64(w)) & U64(MASKS[w])
        else:
            M32 = U64(0xFFFFFFFF)
            a0, a1, b0, b1 = a & M32, a >> U64(32), b & M32, b >> U64(32)
            p00, p01, p10, p11 = a0 * b0, a0 * b1, a1 * b0, a1 * b1
            mid = (p00 >> U64(32)) + (p01 & M32) + (p10 & M32)
            lo = (p00 & M32) | (mid << U64(32))
            hi = p11 + (p01 >> U64(32)) + (p10 >> U64(32)) + (mid >> U64(32))
        if signed:
            sa = (a >> U64(w - 1)) & U64(1)
            sb = (b >> U64(w - 1)) & U64(1)
            hi = (hi - sa * b - sb * a) & U64(MASKS[w])
    return lo & U64(MASKS[w]), hi & U64(MASKS[w])


def divmod_wide(hi, lo, b, w, signed):
    q = np.zeros(len(b), U64)
    r = np.zeros(len(b), U64)
    mw = (1 << w) - 1
    for i, (h, l, d) in enumerate(zip(hi.tolist(), lo.tolist(), b.tolist())):
        n = (h << w) | l
        if signed:
            n = n - (1 << 2 * w) if n >> (2 * w - 1) else n
            d = d - (1 << w) if d >> (w - 1) else d
        if d == 0:
            q[i], r[i] = 0, 0
            continue
        qq = abs(n) // abs(d)
        if (n < 0) != (d < 0):
            qq = -qq
        rr = n - qq * d
        q[i], r[i] = qq & mw, rr & mw
    return q, r


def shift_signals(a, res, spill, c, w):
    top = w - 1
    return {"r_top": ((res >> U64(top)) & U64(1)).astype(np.int8), "a_top": ((a >> U64(top)) & U64(1)).astype(np.int8),
            "a_top2": ((a >> U64(top - 1)) & U64(1)).astype(np.int8), "nz": (res != 0).astype(np.int8), "par8": _parity8(res),
            "spill": spill.astype(np.int8), "c1": (c == 1).astype(np.int8), "c0": (c == 0).astype(np.int8), "r_bot": (res & U64(1)).astype(np.int8),
            "r_top2": ((res >> U64(top - 1)) & U64(1)).astype(np.int8)}


def mul_signals(lo, hi, w):
    top = w - 1
    fill = np.where(((lo >> U64(top)) & U64(1)) == 1, U64(MASKS[w]), U64(0))
    return {"r_top": ((lo >> U64(top)) & U64(1)).astype(np.int8), "nz": (lo != 0).astype(np.int8), "par8": _parity8(lo),
            "hi_nz": (hi != 0).astype(np.int8), "hi_nfill": (hi != fill).astype(np.int8)}


# ------------------------------------------------------------------ taps and predicates


def tap_candidates(sig, old, target, max_size=4, extra=0):
    """All minimal-size truth tables over <= max_size signals (plus the old flag) consistent with every row, then (extra
    > 0, M5c) also every consistent table up to `extra` signals larger. Returns (minimal size, [tap]) where
    tap = [names, table, unobserved entries], minimal ones first."""
    names = sorted(sig) + ["old"]
    cols = {k: np.asarray(v, np.int64) for k, v in sig.items()}
    cols["old"] = np.asarray(old, np.int64)
    target = np.asarray(target, np.int64)
    found, smin = [], None
    if np.all(target == target[0]):
        found, smin = [[[], [int(target[0])], []]], 0
    for size in range(1, max_size + 1):
        if smin is not None and size > smin + extra:
            break
        for combo in itertools.combinations(names, size):
            idx = np.zeros(len(target), np.int64)
            for c in combo:
                idx = idx * 2 + (cols[c] & 1)
            tab = np.full(1 << size, -1, np.int64)
            tab[idx] = target  # last write wins; check conflicts
            if np.array_equal(tab[idx], target):
                found.append([list(combo), [int(v) if v >= 0 else 0 for v in tab], [int(e) for e in np.flatnonzero(tab < 0)]])
        if found and smin is None:
            smin = size
            if not extra:
                break
    return (smin, found) if found else (None, [])


def _tap_index(names, sig, old):
    cols = dict(sig)
    cols["old"] = old
    idx = np.zeros(len(old), np.int64)
    for c in names:
        idx = idx * 2 + (np.asarray(cols[c], np.int64) & 1)
    return idx


def _tap_unobserved(tap, sig, old):
    """Table entries of a tap that none of the rows reached."""
    seen = set(np.unique(_tap_index(tap[0], sig, np.asarray(old, np.int64))).tolist()) if tap[0] else {0}
    return [e for e in range(1 << len(tap[0])) if e not in seen]


# more consistent tables than this: the flag abstains on every row (no pick). M5c had 48; the proof gate found 80
# cmp/test records whose OF/AF version space overflowed it (vacuous: proved but abstaining everywhere), so M5c' keeps
# up to 1024 (each alternative is a <= 4-signal table; the scalar runner caches the check per signal vector)
TAP_ALT_CAP = 1024
# M5e: a flag-source alternative that no table of its op kind's shared rule explains is dropped ("drop"); "fit" keeps
# the per-record fit (M5c''/M5d behaviour, the negative control)
SRC_ALT_NONE_CONSISTENT = "drop"
# M5e: an 8-bit register field without a fixed REX byte keeps codes 4-7 closed unless its own rows show them working
# (without REX those codes are AH/CH/DH/BH, which the menu cannot express; with REX they are SPL/BPL/SIL/DIL)
BYTE_REG_GUARD = True


def _alt_abstain(alt, sig, old, v):
    out = apply_tap(alt, sig, old) != v
    if len(alt) > 2 and alt[2] and alt[0]:
        out |= np.isin(_tap_index(alt[0], sig, old), alt[2])
    return out


def tap_abstain(fl, f, sig, old, v):
    """Rows where flag f is not determined by the evidence: its tap's entry was never observed, or a consistent
    alternative tap predicts differently (M5)."""
    out = np.zeros(len(old), bool)
    tap = fl["taps"][f]
    un = (fl.get("tap_unobs") or {}).get(f)
    if un and tap[0]:
        out |= np.isin(_tap_index(tap[0], sig, old), un)
    alts = (fl.get("tap_alts") or {}).get(f, [])
    if alts == "overflow":
        return np.ones(len(old), bool)
    for alt in alts:
        out |= _alt_abstain(alt, sig, old, v)
    return out


def apply_tap(tap, sig, old):
    names, table = tap[0], tap[1]
    if not names:
        return np.full(len(old), table[0], np.int64)
    cols = dict(sig)
    cols["old"] = old
    idx = np.zeros(len(old), np.int64)
    for c in names:
        idx = idx * 2 + (np.asarray(cols[c], np.int64) & 1)
    return np.array(table, np.int64)[idx]


def induce_pred(flags, y, care=None, max_size=3):
    """Smallest flag subset whose truth table explains y (rows with care=False are free). Returns pred or None.

    M5 version space: every other flag subset of size <= max_size that also explains y is kept as an alternative
    ([combo, table, unobserved entries]); the machine abstains on a row where any consistent alternative is
    unobserved or predicts differently (a predicate under-determined by its rows never guesses)."""
    flags = np.asarray(flags, np.int64)
    y = np.asarray(y, np.int64)
    care = np.ones(len(y), bool) if care is None else care
    if care.sum() == 0:
        return None
    yc, fc = y[care], flags[care]
    found = []  # (size, unobserved, combo, table)
    if np.all(yc == yc[0]):
        found.append((0, 0, [], [int(yc[0])], []))
    for size in range(1, max_size + 1):
        for combo in itertools.combinations(range(NF), size):
            idx = np.zeros(len(yc), np.int64)
            for f in combo:
                idx = idx * 2 + fc[:, f]
            tab = np.full(1 << size, -1, np.int64)
            tab[idx] = yc
            if np.array_equal(tab[idx], yc):
                found.append((size, int((tab < 0).sum()), list(combo), [int(v) if v >= 0 else 0 for v in tab],
                              [int(i) for i in np.flatnonzero(tab < 0)]))
    if not found:
        return None
    found.sort(key=lambda c: (c[0], c[1]))
    best = found[0]
    return ["pred", best[2], best[3], best[4], [[c[2], c[3], c[4]] for c in found[1:]]]


REG_PRED_ALT_CAP = 8


def induce_pred_regs(cols, y, care=None):
    """M5e register condition: the smallest set of tests of ONE register ((r == 0), (r == 1) or both) whose truth table
    explains y on the care rows; every other register's consistent reading is kept as a version-space alternative
    (more than REG_PRED_ALT_CAP readings: None, the condition is not determined). cols = pred_columns(flags, regs)."""
    cols = np.asarray(cols, np.int64)
    y = np.asarray(y, np.int64)
    care = np.ones(len(y), bool) if care is None else care
    if care.sum() == 0 or cols.shape[1] <= NF:
        return None
    yc, fc = y[care], cols[care]
    found = []
    for r in range(16):
        for combo in ([NF + 2 * r], [NF + 2 * r + 1], [NF + 2 * r, NF + 2 * r + 1]):
            idx = np.zeros(len(yc), np.int64)
            for f in combo:
                idx = idx * 2 + fc[:, f]
            if len(np.unique(idx)) < 2:
                continue  # a test that never varies on the rows explains nothing
            tab = np.full(1 << len(combo), -1, np.int64)
            tab[idx] = yc
            if np.array_equal(tab[idx], yc):
                found.append((len(combo), int((tab < 0).sum()), combo, [int(v) if v >= 0 else 0 for v in tab],
                              [int(i) for i in np.flatnonzero(tab < 0)]))
    if not found or len(found) > REG_PRED_ALT_CAP:
        return None
    found.sort(key=lambda c: (c[0], c[1]))
    best = found[0]
    return ["pred", best[2], best[3], best[4], [[c[2], c[3], c[4]] for c in found[1:]]]


def _pred_index(combo, flags):
    idx = np.zeros(len(flags), np.int64)
    for f in combo:
        idx = idx * 2 + flags[:, f].astype(np.int64)
    return idx


# ------------------------------------------------------------------ synthesis


class Fail(Exception):
    pass


ALL_NONE = (None,)


class Synth:
    """Enumerative search for one group's effect. Every accepted expression holds on every row."""

    def __init__(self, m: Machine, x: Ctx, allow_flag_search=False, allow_induce=False):
        self.m, self.x = m, x
        self.allow_flag_search = allow_flag_search
        self.allow_induce = allow_induce
        self._leaf_cache = {}
        self._addr = None
        self._load_cache = {}
        self.exclude = set()

    def _ok(self, e):
        return json.dumps(e) not in self.exclude

    # ---- leaves
    def leaves(self, w):
        if w in self._leaf_cache:
            return self._leaf_cache[w]
        x = self.x
        out = []
        for r in range(16):
            out.append((["reg", r, w, "z"], mask(x.g["regs"][:, r], w)))
            for pw in (8, 16, 32):
                if pw < w:
                    for ext in ("z", "s"):
                        e = ["reg", r, pw, ext]
                        out.append((e, mask(ev(self.m, e, x), w)))
        for j, rv in enumerate(x.run_vals):
            if rv is None:
                continue
            for ext in ("z", "s"):
                e = ["run", j, ext]
                out.append((e, mask(ev(self.m, e, x), w)))
        for c in (0, 1, MASK64):
            out.append((["const", c & MASKS[w]], np.full(x.n, c & MASKS[w], U64)))
        for r in range(16):
            for pw in (32, 64, 16, 8):
                if pw <= w or pw == 64:
                    e = ["fill", r, pw]
                    out.append((e, mask(ev(self.m, e, x), w)))
        if self.m.adder is not None:
            out.append((["next"], mask(x.next, w)))
        self._leaf_cache[w] = out
        return out

    def addresses(self):
        """All address expressions with their values: [(expr, values)], computed with the learned adder."""
        if self._addr is not None:
            return self._addr
        x, m = self.x, self.m
        regs = x.g["regs"]
        lut = m.luts[m.adder]
        bases = [None] + list(range(16)) + (["next"] if m.adder is not None else [])
        idxs = [(None, 0)] + [(r, s) for r in range(16) for s in range(4)]
        disps = [None] + [["run", j, "s"] for j, rv in enumerate(x.run_vals) if rv is not None]
        zero = np.zeros(x.n, U64)
        bv = np.stack([x.next if b == "next" else regs[:, b] if b is not None else zero for b in bases], 0)
        iv = np.stack([(regs[:, r] << U64(s)) if r is not None else zero for r, s in idxs], 0)
        part = lut.run(bv[:, None, :], iv[None, :, :], 0, 64)  # (B, I, n): adding zero is the identity for the adder
        dv = np.stack([ev(m, d, x) if d is not None else zero for d in disps], 0)
        tot = lut.run(part[:, :, None, :], dv[None, None, :, :], 0, 64)  # (B, I, D, n)
        out = []
        for bi, b in enumerate(bases):
            for ii, (r, s) in enumerate(idxs):
                if b is None and r is not None and s == 0:
                    continue  # a lone unscaled register is a base (fewest parameters)
                for di, d in enumerate(disps):
                    if b is None and r is None and d is None:
                        continue
                    out.append((["sum", b, r, s if r is not None else 0, d], tot[bi, ii, di]))
        self._addr = out
        return out

    def const_addresses(self, targets):
        """base + learned constant, the constant found by inverting the adder against target address vectors."""
        x, m = self.x, self.m
        lut = m.luts[m.adder]
        out = []
        for b in range(16):
            bv = x.g["regs"][:, b]
            for tv in targets:
                c, ok = invert(lut, bv, tv, np.zeros(x.n, np.int64), 64)
                if ok.all():
                    vals, cnt = np.unique(c, return_counts=True)
                    cval = int(vals[cnt.argmax()])
                    if cnt.max() >= max(2, x.n // 2) and cval != 0:
                        e = ["sum", b, None, 0, ["const", cval]]
                        out.append((e, lut.run(bv, U64(cval), 0, 64)))
        return out

    def loads(self, nb):
        if nb in self._load_cache:
            return self._load_cache[nb]
        addrs = self.addresses()
        A = np.stack([v for _, v in addrs], 1)  # (n, C)
        vals = self.x.store.read(A, nb)  # (n, C)
        out = [(e, vals[:, i]) for i, (e, _) in enumerate(addrs)]
        self._load_cache[nb] = out
        return out

    def value_index(self, w, with_loads=True, with_addr=True, extra=()):
        """dict bytes(values masked to w) -> expr, simplest first."""
        idx = {}
        def put(e, v):
            kb = mask(v, w).tobytes()
            if kb not in idx:
                idx[kb] = e
        for e, v in self.leaves(w):
            put(e, v)
        for e, v in extra:
            put(e, v)
        if with_loads:
            for nb in (1, 2, 4, 8):
                if 8 * nb > w:
                    continue
                for e, v in self.loads(nb):
                    put(["load", e, nb, "z"], v)
                    if 8 * nb < w:
                        put(["load", e, nb, "s"], _sext_np(v, 8 * nb))
        if with_addr:
            for e, v in self.addresses():
                put(["addr", e], v)
        return idx

    # ---- value search for one target
    def find_value(self, V, O, w, allow_ops=True):
        """Expression e with e (masked to w) == V on every row; O = target old value (w bits) or None.
        When the flags change, operations that produce flags are tried before plain address sums."""
        x, m = self.x, self.m
        V = mask(V, w)
        key = V.tobytes()
        flags_change = not np.array_equal(x.g["flags"], x.g["a_flags"])
        # 1. leaves, loads, addresses
        idx = self.value_index(w, with_loads=False, with_addr=False)
        if key in idx:
            return idx[key]
        if np.isin(V, [0, 1]).all() and not (V == V[0]).all():
            p = induce_pred(x.g["flags"], V.astype(np.int64))
            if p is not None:
                return p
        idx = self.value_index(w, with_addr=not flags_change)
        if key in idx:
            return idx[key]
        if not allow_ops:
            return None
        # 2. table ops: A in {old, const 0, registers}, B by inversion
        As = []
        if O is not None:
            As.append((["old"], mask(O, w)))
        As.append((["const", 0], np.zeros(x.n, U64)))
        As += [(["reg", r, w, "z"], mask(x.g["regs"][:, r], w)) for r in range(16)]
        for t, lut in enumerate(m.luts):
            for Ae, Av in As:
                b, ok = invert(lut, Av, V, x.cf, w)
                if not ok.all():
                    continue
                kb = b.tobytes()
                if kb in idx:
                    if self._ok(["table", t, Ae, idx[kb], w]):
                        return ["table", t, Ae, idx[kb], w]
                if (b == b[0]).all() and Ae[0] != "const":
                    if self._ok(["table", t, Ae, ["const", int(b[0])], w]):
                        return ["table", t, Ae, ["const", int(b[0])], w]
        # non-invertible tables: one-state tables by bitwise constraint (each bit of B is forced, free, or impossible),
        # two-state ones by brute force over non-load operands
        cands = list(idx.items())
        Bv = np.stack([np.frombuffer(kb, U64) for kb, _ in cands], 0) if cands else None
        nonload = np.array([e[0] != "load" for _, e in cands]) if cands else None
        for t, lut in enumerate(m.luts):
            tab = m.tables[t]
            for Ae, Av in As:
                if Bv is None:
                    break
                if tab.n == 1:
                    em = np.array(tab.emit)[:, 0]
                    det, val, bad = np.zeros(x.n, U64), np.zeros(x.n, U64), np.zeros(x.n, bool)
                    for i in range(w):
                        ai = ((Av >> U64(i)) & U64(1)).astype(np.int64)
                        vi = ((V >> U64(i)) & U64(1)).astype(np.int64)
                        f0, f1 = em[2 * ai] == vi, em[2 * ai + 1] == vi
                        bad |= ~(f0 | f1)
                        forced = f0 ^ f1
                        det |= forced.astype(U64) << U64(i)
                        val |= (forced & f1).astype(U64) << U64(i)
                    if bad.any():
                        continue
                    hit = np.flatnonzero((((Bv ^ val[None, :]) & det[None, :]) == 0).all(1))
                else:
                    sel = np.flatnonzero(nonload)
                    res = lut.run(Av[None, :], Bv[sel], x.cf[None, :], w)
                    hit = sel[np.flatnonzero((res == V[None, :]).all(1))]
                for h in hit[:8]:
                    if np.array_equal(lut.run(Av, Bv[h], x.cf, w), V):
                        if self._ok(["table", t, Ae, cands[h][1], w]):
                            return ["table", t, Ae, cands[h][1], w]
        # 3. shifts
        counts = [(["const", 1], np.ones(x.n, np.int64))]
        raw = []
        for j, rv in enumerate(x.run_vals):
            if rv is not None and rv[1] == 1:
                raw.append((["run", j, "z"], (rv[0] & U64(0xFF)).astype(np.int64)))
        for r in range(16):
            raw.append((["reg", r, 8, "z"], (x.g["regs"][:, r] & U64(0xFF)).astype(np.int64)))
        # M5c: a count may be masked by a learned constant (31 or 63); the rows choose masked or raw
        for ce, cv in raw:
            counts += [(["cmask", ce, 31], cv & 31), (["cmask", ce, 63], cv & 63), (ce, cv)]
        srcs = ([(["old"], O)] if O is not None else []) + [(["reg", r, w, "z"], x.g["regs"][:, r]) for r in range(16)]
        for kind in (("l", "z"), ("r", "z"), ("r", "t"), ("l", "w"), ("r", "w")):
            for Ae, Av in srcs[:1] if O is not None else srcs:
                for Ce, Cv in counts:
                    res, _ = shift_gather(Av, Cv, kind, w)
                    if np.array_equal(res, V):
                        if self._ok(["shift", list(kind), Ae, Ce, w]):
                            return ["shift", list(kind), Ae, Ce, w]
        # 4. multiply (low half)
        if m.adder_ok and Bv is not None:
            nl = np.flatnonzero(nonload)
            loadA = [(["load", e, w // 8, "z"], v) for e, v in self.loads(w // 8)]
            # A = a load (memory source), B = a non-load operand
            if loadA:
                LA = np.stack([mask(v, w) for _, v in loadA], 0)
                for bi in nl:
                    lo, _ = mul_wide(LA, Bv[bi][None, :], w, False)
                    hit = np.flatnonzero((lo == V[None, :]).all(1))
                    for h in hit[:2]:
                        for signed in (False, True):
                            if self._ok(["mul", loadA[h][0], cands[bi][1], w, signed, "lo"]):
                                return ["mul", loadA[h][0], cands[bi][1], w, signed, "lo"]
            for Ae, Av in As[:1] + As[2:]:
                lo, _ = mul_wide(mask(Av, w)[None, :], Bv, w, False)
                hit = np.flatnonzero((lo == V[None, :]).all(1))
                for h in hit[:4]:
                    for signed in (False, True):  # same low half; the flags tell signed from unsigned
                        if self._ok(["mul", Ae, cands[h][1], w, signed, "lo"]):
                            return ["mul", Ae, cands[h][1], w, signed, "lo"]
        if flags_change:
            idx2 = self.value_index(w, with_addr=True)
            if key in idx2:
                return idx2[key]
        # 5. select (conditional move): V = pred ? B : old
        if O is not None:
            Om = mask(O, w)
            for kb, Be in cands:
                Bvv = np.frombuffer(kb, U64)
                isB, isO = Bvv == V, Om == V
                if not (isB | isO).all() or isO.all() or isB.all():
                    continue
                care = isB ^ isO
                p = induce_pred(x.g["flags"], isB.astype(np.int64), care)
                if p is not None:
                    return ["select", p, Be, ["old"]]
            # M5e: register conditions, and (select-step) the old value plus a learned constant as the other operand
            step = []
            ch = Om != V
            if m.adder is not None and ch.any() and not ch.all():
                b, ok = invert(m.luts[m.adder], Om[ch], V[ch], np.zeros(int(ch.sum()), np.int64), w)
                if ok.all() and (b == b[0]).all():
                    c = int(b[0])
                    step.append((mask(m.luts[m.adder].run(Om, U64(c), 0, w), w).tobytes(),
                                 ["table", m.adder, ["old"], ["const", c], w]))
            for kb, Be in step + cands:
                Bvv = np.frombuffer(kb, U64)
                isB, isO = Bvv == V, Om == V
                if not (isB | isO).all() or isO.all() or isB.all():
                    continue
                p = induce_pred_regs(x.pin, isB.astype(np.int64), isB ^ isO)
                if p is not None:
                    return ["select", p, Be, ["old"]]
        return None

    # ---- the whole group
    def fit(self):
        x, m = self.x, self.m
        g = x.g
        eff = {"k": x.k, "runs": list(x.runs), "L": x.L, "writes": [], "flags": None, "trap": 0}
        # RIP
        if g["trap"].any():
            if not g["trap"].all():
                raise Fail("mixed trap")
            eff["trap"] = 1
            if not np.array_equal(g["a_rip"], x.next):
                raise Fail("trap rip")
            eff["rip"] = ["next"]
        else:
            eff["rip"] = self.fit_rip()
        # memory
        changed_rows = g["ch_n"] > 0
        if changed_rows.any():
            if changed_rows.mean() < 0.5:
                raise Fail("rare memory change")
            eff["writes"].append(self.fit_store())
        # registers
        diff = g["a_regs"] != g["regs"]
        C = [r for r in range(16) if diff[:, r].any()]
        if eff["trap"]:
            if C or changed_rows.any():
                raise Fail("trap with effects")
        for r in C:
            if diff[:, r].mean() < 0.5 and x.n >= 6:
                raise Fail(f"rare change r{r}")
        pending = []
        for r in C:
            e = self.fit_reg(r)
            if e is None:
                pending.append(r)
            else:
                eff["writes"].append(e)
        if pending:
            if len(pending) == 1:
                others = [w[0][1] for w in eff["writes"] if w[0][0] == "reg"]
                for o in others:
                    try:
                        joint = self.fit_joint([o, pending[0]])
                    except Fail:
                        continue
                    eff["writes"] = [w for w in eff["writes"] if not (w[0][0] == "reg" and w[0][1] == o)] + joint
                    pending = []
                    break
            if pending:
                eff["writes"] += self.fit_joint(pending)
        # flags; if the chosen operation cannot explain them, exclude it and search for the next explanation
        try:
            eff["flags"] = self.fit_flags(eff)
        except Fail as e:
            if getattr(self, "_retries", 0) >= 4:
                raise
            ops = [json.dumps(o) for _, ex in eff["writes"] for o in _ops(ex)]
            if not ops:
                raise
            self._retries = getattr(self, "_retries", 0) + 1
            self.exclude.update(ops)
            return self.fit()
        return eff

    def fit_rip(self):
        x, m, g = self.x, self.m, self.x.g
        if np.array_equal(g["a_rip"], x.next):
            return ["next"]
        for j, rv in enumerate(x.run_vals):
            if rv is None:
                continue
            tgt = x.add(x.next, _sext_np(rv[0], 8 * rv[1]))
            e_t = ["table", m.adder, ["next"], ["run", j, "s"], 64]
            if np.array_equal(g["a_rip"], tgt):
                if j != len(x.runs) - 1 or x.runs[j] == 0:
                    pass
                return e_t
            taken = g["a_rip"] == tgt
            fall = g["a_rip"] == x.next
            if (taken | fall).all() and taken.any() and fall.any():
                p = induce_pred(g["flags"], taken.astype(np.int64), taken ^ fall)
                if p is not None:
                    return ["select", p, e_t, ["next"]]
        # M5e (current-rip): the step repeats the instruction or falls through, under a flag or register condition
        stay, fall = g["a_rip"] == g["rip"], g["a_rip"] == x.next
        if (stay | fall).all() and stay.any() and fall.any() and not any(x.runs):
            p = induce_pred(g["flags"], stay.astype(np.int64), stay ^ fall)
            if p is None:
                p = induce_pred_regs(x.pin, stay.astype(np.int64), stay ^ fall)
            if p is not None:
                return ["select", p, ["cur"], ["next"]]
        if x.runs != (0, 0) and any(x.runs):
            raise Fail("rip with unused runs")
        idx = self.value_index(64, with_addr=False)
        e = idx.get(g["a_rip"].tobytes())
        if e is not None and e[0] in ("reg", "load"):
            return e
        raise Fail("rip")

    def fit_store(self):
        x, g = self.x, self.x.g
        n = x.n
        ch_n, ch_addr = g["ch_n"].astype(np.int64), g["ch_addr"]
        valid = np.arange(16)[None, :] < np.minimum(ch_n, 16)[:, None]
        lo = np.where(valid, ch_addr, U64(MASK64)).min(1)
        rows_c = ch_n > 0
        addrs = self.addresses() + self.const_addresses([np.where(rows_c, lo, lo[rows_c][0] if rows_c.any() else 0)])
        A = np.stack([v for _, v in addrs], 0)  # (C, n)
        for nb in (1, 2, 4, 8):
            with np.errstate(over="ignore"):
                off = ch_addr[None, :, :] - A[:, :, None]  # (C, n, 16)
            inside = (off < U64(nb)) | ~valid[None, :, :]
            okc = np.flatnonzero(inside.all((1, 2)))
            for ci in okc[:64]:
                ae, av = addrs[ci]
                old = x.store.read(av, nb)
                new = old.copy()
                for i in range(n):
                    for j in range(min(int(ch_n[i]), 16)):
                        o = int(ch_addr[i, j] - av[i])
                        new[i] = (int(new[i]) & ~(0xFF << (8 * o))) | (int(g["ch_val"][i, j]) << (8 * o))
                e = self.find_value(new, old, 8 * nb)
                if e is not None:
                    return [["mem", ae, nb], e]
        raise Fail("store")

    def fit_reg(self, r):
        x, g = self.x, self.x.g
        V, O = g["a_regs"][:, r], g["regs"][:, r]
        # narrowest write first (MDL: the smallest claim about which bits change)
        for w, merge in ((8, "keep"), (16, "keep"), (32, "zero"), (64, "repl")):
            if merge == "zero" and not ((V >> U64(w)) == 0).all():
                continue
            if merge == "keep" and not ((V >> U64(w)) == (O >> U64(w))).all():
                continue
            e = self.find_value(V, O, w)
            if e is not None:
                if merge == "keep":
                    # the rows cannot tell a narrow write from a full one when no carry crossed the narrow width
                    # (e.g. RSP -/+ 8 away from a byte boundary); random operands of a real narrow op always cross,
                    # so a full-width explanation that also fits is preferred
                    e64 = self.find_value(V, O, 64)
                    if e64 is not None:
                        return [["reg", r, 64, "repl"], e64]
                return [["reg", r, w, merge], e]
        return None

    def fit_joint(self, regs_left):
        """Two-register results: wide multiply (lo, hi) and divide (quotient, remainder)."""
        x, m, g = self.x, self.m, self.x.g
        if len(regs_left) != 2 or not m.adder_ok:
            raise Fail(f"unexplained registers {regs_left}")
        for w, merge in ((64, "repl"), (32, "zero")):
            V = {r: g["a_regs"][:, r] for r in regs_left}
            if merge == "zero" and not all(((V[r] >> U64(32)) == 0).all() for r in regs_left):
                continue
            idx = self.value_index(w, with_addr=False)
            cands = list(idx.items())
            regs = [(["reg", r, w, "z"], mask(g["regs"][:, r], w)) for r in range(16)]
            for t1, t2 in (regs_left, regs_left[::-1]):
                for signed in (False, True):
                    Bv = np.stack([np.frombuffer(kb, U64) for kb, _ in cands], 0)
                    for Ae, Av in regs:
                        lo, hi = mul_wide(Av[None, :], Bv, w, signed)
                        hit = np.flatnonzero((lo == mask(V[t1], w)[None, :]).all(1) & (hi == mask(V[t2], w)[None, :]).all(1))
                        if len(hit):
                            Be = cands[hit[0]][1]
                            return [[["reg", t1, w, merge], ["mul", Ae, Be, w, signed, "lo"]],
                                    [["reg", t2, w, merge], ["mul", Ae, Be, w, signed, "hi"]]]
                    for (He, Hv), (Le, Lv) in itertools.permutations(regs, 2):
                        if He[1] not in regs_left and Le[1] not in regs_left:
                            continue  # search order: a dividend register is overwritten by a result
                        for kb, Be in cands:
                            Bv = np.frombuffer(kb, U64)
                            if (Bv == 0).any():
                                continue
                            q, rr = divmod_wide(Hv, Lv, Bv, w, signed)
                            if np.array_equal(q, mask(V[t1], w)) and np.array_equal(rr, mask(V[t2], w)):
                                return [[["reg", t1, w, merge], ["div", He, Le, Be, w, signed, "q"]],
                                        [["reg", t2, w, merge], ["div", He, Le, Be, w, signed, "r"]]]
        raise Fail(f"unexplained registers {regs_left}")

    def fit_flags(self, eff):
        x, g, m = self.x, self.x.g, self.m
        before, after = g["flags"].astype(np.int64), g["a_flags"].astype(np.int64)
        if np.array_equal(before, after):
            return {"src": None, "taps": {f: [["old"], [0, 1], 0] for f in FLAGS}}
        srcs = []
        for tgt, e in eff["writes"]:
            for sub in _ops(e):
                srcs.append((tgt, sub))
        for tgt, sub in srcs:
            old = None
            if tgt[0] == "reg":
                old = mask(g["regs"][:, tgt[1]], tgt[2])
            elif tgt[0] == "mem":
                old = x.store.read(eval_addr(m, tgt[1], x), tgt[2])
            out = ev(m, sub, x, old, want_sig=True)
            if not isinstance(out, tuple):
                continue
            taps = self._taps(out[1], before, after)
            if taps is not None:
                return {"src": sub, "src_old": tgt, "taps": {f: c[0] for f, c in taps.items()}, "tap_cands": taps}
        if not self.allow_flag_search:
            raise Fail("flags (deferred)")
        return self.flag_search(before, after)

    def _taps(self, sig, before, after):
        taps = {}
        for k, f in enumerate(FLAGS):
            size, cands = tap_candidates(sig, before[:, k], after[:, k])
            if size is None:
                return None
            taps[f] = cands
        return taps

    def flag_search(self, before, after):
        """Flags-only effects (compare/test): an operation whose result is not written. Candidate (A, B) pairs with at
        most one load (A from registers/loads, B from registers/runs/constants, or A a register and B a load) are
        filtered row by row with the operation's shared taps (pivot rows), then the survivors are verified."""
        x, m = self.x, self.m
        known = getattr(m, "known_taps", {})
        found = []
        for w in (64, 32, 8, 16):
            leaves = [(e, mask(v, w)) for e, v in self.leaves(w) if e[0] in ("reg", "run", "const") and (e[0] != "reg" or e[2] == w)]
            loads = [(["load", e, w // 8, "z"], v) for e, v in self.loads(w // 8)]
            regs = [(e, v) for e, v in leaves if e[0] == "reg"]
            combos = [(regs + loads, leaves), (regs, loads)]
            for t, lut in enumerate(m.luts):
                kt = known.get(t)
                if not kt:
                    continue
                for As, Bs in combos:
                    if not As or not Bs:
                        continue
                    Av = np.stack([v for _, v in As], 0)
                    Bv = np.stack([v for _, v in Bs], 0)
                    ai, bi = np.meshgrid(np.arange(len(As)), np.arange(len(Bs)), indexing="ij")
                    ai, bi = ai.ravel(), bi.ravel()
                    for row in range(x.n):
                        if not len(ai):
                            break
                        res, sig = lut.run(Av[ai, row], Bv[bi, row], x.cf[row], w, signals=True)
                        good = np.ones(len(ai), bool)
                        for k, f in enumerate(FLAGS):
                            good &= apply_tap(kt[f], sig, np.full(len(ai), before[row, k])) == after[row, k]
                        ai, bi = ai[good], bi[good]
                    for a_i, b_i in zip(ai, bi):
                        sub = ["table", t, As[a_i][0], Bs[b_i][0], w]
                        out = ev(m, sub, x, None, want_sig=True)
                        taps = self._taps(out[1], before, after)
                        if taps is not None:
                            found.append({"src": sub, "src_old": None, "taps": {f: c[0] for f, c in taps.items()}, "tap_cands": taps})
                            if len(found) > FLAG_SEARCH_CAP:
                                raise Fail("flags ambiguous (cap)")  # too many readings: abstain rather than pick
        if found:
            # M5c version space: every (operation, operands, width) reading that explains the rows is kept; the first is
            # the effect, the others are flag-source alternatives (share_taps gives them taps; the machine abstains where
            # they disagree)
            eff = found[0]
            eff["src_cands"] = [c["src"] for c in found[1:]]
            return eff
        found = self.flag_search_shift(before, after)  # M5e (flag-only-shift)
        if found:
            eff = found[0]
            eff["src_cands"] = [c["src"] for c in found[1:]]
            return eff
        raise Fail("flags")

    def flag_search_shift(self, before, after):
        """M5e flags-only shift readings (bit test): a right shift of a register by a register's low byte or a run (raw
        or masked 31/63) whose signals explain every flag with taps; every consistent reading is kept (version space)."""
        x = self.x
        counts = []
        for j, rv in enumerate(x.run_vals):
            if rv is not None and rv[1] == 1:
                counts.append((["run", j, "z"], (rv[0] & U64(0xFF)).astype(np.int64)))
        for r in range(16):
            counts.append((["reg", r, 8, "z"], (x.g["regs"][:, r] & U64(0xFF)).astype(np.int64)))
        counts = [c for ce, cv in counts for c in ((["cmask", ce, 31], cv & 31), (["cmask", ce, 63], cv & 63), (ce, cv))]
        found = []
        for w in (64, 32):
            for r in range(16):
                Ae, Av = ["reg", r, w, "z"], mask(x.g["regs"][:, r], w)
                for Ce, Cv in counts:
                    res, spill = shift_gather(Av, Cv, ("r", "z"), w)
                    taps = self._taps(shift_signals(Av, res, spill, Cv, w), before, after)
                    # single-signal taps only (each flag = one shift signal, its old value or a constant): a multi-signal
                    # tap over an unrelated register fits random rows often enough to overflow the cap (bt fixture,
                    # job-166fb9584415: 16-27 readings per key at 32 rows)
                    if taps is None or any(len(c[0][0]) > 1 for c in taps.values()):
                        continue
                    found.append({"src": ["shift", ["r", "z"], Ae, Ce, w], "src_old": None,
                                  "taps": {f: c[0] for f, c in taps.items()}, "tap_cands": taps})
                    if len(found) > FLAG_SEARCH_CAP:
                        raise Fail("flags ambiguous (cap)")
        return found


FLAG_SEARCH_CAP = 16


def _ops(e):
    """Operation nodes inside an expression, outermost first."""
    if not isinstance(e, list) or not e:
        return []
    out = []
    if e[0] in ("table", "shift", "mul", "div"):
        out.append(e)
    for sub in e[1:]:
        if isinstance(sub, list):
            out += _ops(sub)
    return out


# ------------------------------------------------------------------ bootstrap: datapath tables and the adder


def bootstrap_tables(m: Machine, data, groups, budget=400, max_rows=24, patience=60):
    """Induce per-bit tables on single-register-change groups (A = old value, B = another register or a run)."""
    tried = since_new = 0
    for (k, key), rows in groups:
        if tried >= budget or since_new >= patience:
            break
        g_regs, a_regs = data["regs"][rows], data["a_regs"][rows]
        diff = g_regs != a_regs
        C = [r for r in range(16) if diff[:, r].mean() > 0.8]
        if len(C) != 1 or (data["ch_n"][rows] > 0).any() or len(rows) < 6:
            continue
        r = C[0]
        Lobs = (data["a_rip"][rows] - data["rip"][rows]).astype(np.int64)
        if not (Lobs == Lobs[0]).all() or not 1 <= Lobs[0] <= 15:
            continue
        tried += 1
        L = int(Lobs[0])
        win = data["window"][rows].astype(U64)
        cf = data["flags"][rows, CF_I]
        V, O = a_regs[:, r], g_regs[:, r]
        Bs = [g_regs[:, s] for s in range(16) if s != r]
        for size in (1, 4):
            if L - size >= 1:
                v = np.zeros(len(rows), U64)
                for j in range(size):
                    v |= win[:, L - size + j] << U64(8 * j)
                Bs.append(_sext_np(v, 8 * size))
        for w in (64, 32, 8):
            if w < 64 and not ((V >> U64(w)) == (0 if w == 32 else O >> U64(w))).all():
                continue
            hit = None
            for B in Bs:
                # library first
                for t, lut in enumerate(m.luts):
                    if np.array_equal(lut.run(mask(O, w), mask(B, w), cf, w), mask(V, w)):
                        hit = t
                        break
                if hit is not None:
                    break
                tab, info = D.induce(mask(O, w)[:max_rows], mask(B, w)[:max_rows], mask(V, w)[:max_rows], w, cf[:max_rows], max_states=2)
                if tab is not None and info.get("unidentified_entries", 1) == 0:
                    lut = LUT(tab)
                    if np.array_equal(lut.run(mask(O, w), mask(B, w), cf, w), mask(V, w)):
                        m.add_table(tab)
                        hit = len(m.tables) - 1
                        break
            if hit is not None:
                m.stats[f"bootstrap_t{hit}"] += 1
                break
        n_before = getattr(m, "_nt", 0)
        since_new = 0 if len(m.tables) > n_before else since_new + 1
        m._nt = len(m.tables)
    return m


def choose_adder(m: Machine, data, sample=4000):
    """The adder = the library table that maps (RIP, observed length) to the next RIP on the most rows."""
    rip, a_rip = data["rip"][:sample], data["a_rip"][:sample]
    L = (a_rip - rip).astype(np.int64)
    ok = (L >= 1) & (L <= 15)
    cf = data["flags"][:sample, CF_I][ok].astype(np.int64)  # the row's own incoming CF (a carry-in table is not the adder)
    best, score = None, -1
    for t, lut in enumerate(m.luts):
        s = int((lut.run(rip[ok], L[ok].astype(U64), cf, 64) == a_rip[ok]).sum())
        if s > score:
            best, score = t, s
    m.adder = best
    # the adder is also checked against integer addition on random words: if equal, multiply/divide micro-loops
    # (repeated learned add / subtract) can be evaluated with integer arithmetic during search and execution
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2**63, 4096, dtype=np.uint64) * U64(2) + rng.integers(0, 2, 4096, dtype=np.uint64)
    b = rng.integers(0, 2**63, 4096, dtype=np.uint64) * U64(2) + rng.integers(0, 2, 4096, dtype=np.uint64)
    with np.errstate(over="ignore"):
        m.adder_ok = bool(np.array_equal(m.luts[best].run(a, b, 0, 64), a + b))
    return best, score, int(ok.sum())


# ------------------------------------------------------------------ applying an effect


def apply_effect(m, eff, x):
    """Predicted after-state over the context rows: (regs, rip, flags, writes [(addr, nb, value)], trap, abstain)."""
    g = x.g
    n = x.n
    regs = g["regs"].copy()
    abstain = np.zeros(n, bool)
    rip_e = eff["rip"]
    if rip_e[0] == "select":
        abstain |= _pred_unobserved(rip_e[1], x.pin)
    rip = ev(m, rip_e, x)
    writes = []
    olds = {}
    for tgt, e in eff["writes"]:
        abstain |= _expr_abstain(e, x.pin)
        if tgt[0] == "reg":
            _, r, w, merge = tgt
            old = mask(g["regs"][:, r], w)
            v = mask(ev(m, e, x, old), w)
            if merge == "repl" or merge == "zero":
                regs[:, r] = v
            else:
                regs[:, r] = (g["regs"][:, r] & ~U64(MASKS[w])) | v
        else:
            _, a_e, nb = tgt
            addr = eval_addr(m, a_e, x)
            old = x.store.read(addr, nb)
            v = mask(ev(m, e, x, old), 8 * nb)
            writes.append((addr, nb, v))
    fl = eff["flags"]
    flags = g["flags"].astype(np.int64).copy()
    if fl and fl.get("src") is not None:
        tgt = fl.get("src_old")
        old = None
        if tgt is not None and tgt[0] == "reg":
            old = mask(g["regs"][:, tgt[1]], tgt[2])
        elif tgt is not None and tgt[0] == "mem":
            old = x.store.read(eval_addr(m, tgt[1], x), tgt[2])
        _, sig = ev(m, fl["src"], x, old, want_sig=True)
        for k, f in enumerate(FLAGS):
            o = g["flags"][:, k].astype(np.int64)
            flags[:, k] = apply_tap(fl["taps"][f], sig, o)
            abstain |= tap_abstain(fl, f, sig, o, flags[:, k])
        if fl.get("src_alts"):
            abstain |= src_alt_abstain(m, fl, x, old, flags)
    elif fl:
        for k, f in enumerate(FLAGS):
            flags[:, k] = apply_tap(fl["taps"][f], {}, g["flags"][:, k].astype(np.int64))
    return regs, rip, flags, writes, eff.get("trap", 0), abstain


def _pred_unobserved(p, flags):
    """Rows where the predicate is not determined by its evidence: its own table entry was never observed, or (M5) a
    consistent alternative predicate is unobserved there or disagrees."""
    out = np.zeros(len(flags), bool)
    if len(p) >= 4 and p[3]:
        out |= np.isin(_pred_index(p[1], flags), p[3])
    if len(p) >= 5 and p[4]:
        v = eval_pred(p, flags)
        for combo, table, unobs in p[4]:
            idx = _pred_index(combo, flags)
            out |= np.array(table, np.int64)[idx] != v
            if unobs:
                out |= np.isin(idx, unobs)
    return out


def _expr_abstain(e, flags):
    out = np.zeros(len(flags), bool)
    if isinstance(e, list) and e:
        if e[0] == "pred":
            out |= _pred_unobserved(e, flags)
        for s in e[1:]:
            if isinstance(s, list):
                out |= _expr_abstain(s, flags)
    return out


def check_rows(m, eff, x):
    """Per-row exactness of an effect against the observed after-state (registers, RIP, flags, memory, trap)."""
    regs, rip, flags, writes, trap, abstain = apply_effect(m, eff, x)
    g = x.g
    ok = (regs == g["a_regs"]).all(1) & (rip == g["a_rip"]) & (flags == g["a_flags"]).all(1) & (g["trap"] == trap)
    pred_changed = [dict() for _ in range(x.n)]
    for addr, nb, v in writes:
        old = x.store.read(addr, nb)
        for i in range(x.n):
            a0, vv, oo = int(addr[i]), int(v[i]), int(old[i])
            for j in range(nb):
                nbyte, obyte = (vv >> (8 * j)) & 0xFF, (oo >> (8 * j)) & 0xFF
                if nbyte != obyte:
                    pred_changed[i][(a0 + j) & MASK64] = nbyte
    for i in range(x.n):
        obs = {int(g["ch_addr"][i, j]): int(g["ch_val"][i, j]) for j in range(min(int(g["ch_n"][i]), 16))}
        if obs != pred_changed[i]:
            ok[i] = False
    return ok & ~abstain, abstain


def refit_taps(m, eff, x):
    """Re-induce the flag taps of a fitted effect on every row of the group (the structure came from a sample)."""
    fl = eff.get("flags") or {}
    if fl.get("src") is None:
        return
    tgt = fl.get("src_old")
    old = None
    if tgt is not None and tgt[0] == "reg":
        old = mask(x.g["regs"][:, tgt[1]], tgt[2])
    elif tgt is not None and tgt[0] == "mem":
        old = x.store.read(eval_addr(m, tgt[1], x), tgt[2])
    _, sig = ev(m, fl["src"], x, old, want_sig=True)
    before, after = x.g["flags"].astype(np.int64), x.g["a_flags"].astype(np.int64)
    for k, f in enumerate(FLAGS):
        size, cands = tap_candidates(sig, before[:, k], after[:, k])
        if size is None:
            return
        fl["taps"][f] = cands[0]
        fl["tap_cands"][f] = cands


def runs_used(eff):
    s = json.dumps([eff["rip"], eff["writes"], (eff["flags"] or {}).get("src")])
    return all(f'["run", {j},' in s for j, size in enumerate(eff["runs"]) if size)


# ------------------------------------------------------------------ trie fitting

LAYOUTS = [(), (1,), (2,), (4,), (8,), (1, 1), (1, 2), (1, 4), (4, 1), (4, 2), (4, 4)]


def fit_group(m, data, rows, k, allow_flag_search, stats, sample_rows=12):
    g_rip, g_arip, trap = data["rip"][rows], data["a_rip"][rows], data["trap"][rows]
    with np.errstate(over="ignore"):
        Lobs = (g_arip - g_rip).astype(np.int64)
    fall = (Lobs >= 1) & (Lobs <= 15)
    for runs in LAYOUTS:
        L = k + sum(runs)
        if L > 15:
            continue
        if trap.any():
            if runs or L != 2 and False:
                continue
        nfall = int((Lobs == L).sum())
        if nfall == 0 and not trap.any():
            if not runs and (fall.any()):
                continue  # rows fall through elsewhere: wrong length
        elif nfall < len(rows) and not runs:
            pass
        if nfall == 0 and fall.mean() > 0.5 and not trap.any():
            continue
        eff = None
        for size in (sample_rows, 64):
            sample = rows[:size]
            xs = Ctx(m, data, sample, k, runs)
            s = Synth(m, xs, allow_flag_search=allow_flag_search)
            try:
                eff = s.fit()
            except Fail as e:
                stats[f"fail:{str(e)[:24]}"] += 1
                eff = None
                break
            # a run must be used, unless the record falls through and its length is thereby observed (don't-care bytes)
            if not runs_used(eff):
                # an unused run is don't-care bytes only when the record falls through (its length is observed) and
                # those bytes actually vary across the group; otherwise they belong to the key (longer k)
                unused_vary = all(len({bytes(r[k + sum(runs[:j]):k + sum(runs[:j + 1])]) for r in data["window"][rows]}) > 1
                                  for j in range(len(runs)) if f'["run", {j},' not in json.dumps([eff["rip"], eff["writes"], (eff["flags"] or {}).get("src")]))
                if eff["rip"] != ["next"] or not unused_vary:
                    stats["fail:unused run"] += 1
                    eff = None
                    break
            # every byte of a run varies across the group; a constant byte belongs to the key (a longer k), not to
            # an immediate that happens to be constant here
            if len(rows) >= 4 and runs:
                W = data["window"][rows]
                off = k
                const_byte = False
                for ln in runs:
                    const_byte |= any((W[:, off + j] == W[0, off + j]).all() for j in range(ln))
                    off += ln
                if const_byte:
                    stats["fail:constant run byte"] += 1
                    eff = None
                    break
            x = Ctx(m, data, rows, k, runs) if len(rows) > len(sample) else xs
            if len(rows) > len(sample):
                refit_taps(m, eff, x)
            ok, _ = check_rows(m, eff, x)
            if ok.all():
                break
            stats["fail:verify"] += 1
            eff = None
            if len(rows) <= size:
                break
        if eff is None:
            continue
        return eff, x
    return None, None


_POOL_STATE = {}


def _fit_job(args):
    kb, rr, k = args
    st = _POOL_STATE
    stats = Counter()
    try:
        eff, _ = fit_group(st["m"], st["data"], rr, k, st["allow"], stats)
    except Exception as e:  # noqa: BLE001 - a crash in one group must not lose the level
        stats[f"crash:{type(e).__name__}:{str(e)[:40]}"] += 1
        eff = None
    return kb, rr, eff, stats


def fit_trie(m, data, rows=None, min_rows=6, max_k=7, allow_flag_search=False, log=print, time_budget=None, workers=1):
    """Byte trie, shortest key first: every group of pending rows sharing their first k bytes is fitted (any layout of
    up to two runs after the key); fitted rows leave the trie. Groups at one level are independent (parallel)."""
    import time
    t0 = time.time()
    N = len(data["rip"])
    pending = np.ones(N, bool) if rows is None else np.isin(np.arange(N), rows)
    records = []
    stats = Counter()
    win = data["window"]
    pool = None
    if workers > 1:
        import multiprocessing as mp
        import gc
        _POOL_STATE.update(m=m, data=data, allow=allow_flag_search)
        gc.collect()
        gc.freeze()  # keep the collector from writing to (and so copying) the parent's objects in the children
        pool = mp.get_context("fork").Pool(workers, maxtasksperchild=400)
    try:
        for k in range(1, max_k + 1):
            idx = np.flatnonzero(pending)
            if not len(idx):
                break
            groups = defaultdict(list)
            for i, kb in zip(idx, (bytes(r[:k]) for r in win[idx])):
                groups[kb].append(i)
            jobs = [(kb, np.array(rr), k) for kb, rr in sorted(groups.items(), key=lambda kv: -len(kv[1])) if len(rr) >= min_rows]
            if pool is not None:
                _POOL_STATE.update(m=m, data=data, allow=allow_flag_search)
                results = pool.map(_fit_job, jobs, chunksize=max(1, len(jobs) // (workers * 8)))
            else:
                _POOL_STATE.update(m=m, data=data, allow=allow_flag_search)
                results = [_fit_job(j) for j in jobs]
            done = 0
            for kb, rr, eff, st in results:
                stats += st
                if eff is None:
                    continue
                records.append({"key": kb, "mask": b"\xff" * k, "effect": eff, "rows": rr, "fields": {}})
                pending[rr] = False
                done += 1
            log(f"[m4] k={k}: groups {len(groups)} tried {len(jobs)} fitted {done} pending rows {int(pending.sum())} t={time.time() - t0:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return records, pending, stats


# ------------------------------------------------------------------ machine-wide flag-tap sharing (MDL)


def op_key(src):
    if src is None:
        return None
    if src[0] == "table":
        return ("table", src[1])
    if src[0] == "shift":
        return ("shift", tuple(src[1]))
    if src[0] == "mul":
        return ("mul", bool(src[4]))
    return (src[0],)


def record_signals(m, data, rec):
    eff = rec["effect"]
    fl = eff.get("flags") or {}
    if fl.get("src") is None:
        return None
    x = Ctx(m, data, rec["rows"], eff["k"], eff["runs"])
    tgt = fl.get("src_old")
    old = None
    if tgt is not None and tgt[0] == "reg":
        old = mask(x.g["regs"][:, tgt[1]], tgt[2])
    elif tgt is not None and tgt[0] == "mem":
        old = x.store.read(eval_addr(m, tgt[1], x), tgt[2])
    _, sig = ev(m, fl["src"], x, old, want_sig=True)
    return sig, x.g["flags"].astype(np.int64), x.g["a_flags"].astype(np.int64)


def share_taps(m, data, records):
    """Per operation kind and flag, every record takes the consistent candidate tap with the largest total support
    (training rows of all records of that kind it explains). A constant that only holds because a small record
    never saw the case loses to the shared rule."""
    by_op = defaultdict(list)
    sigs = {}
    for i, rec in enumerate(records):
        src = (rec["effect"].get("flags") or {}).get("src")
        if src is None:
            continue
        s = record_signals(m, data, rec)
        if s is None:
            continue
        sigs[i] = s
        by_op[op_key(src)].append(i)
    changed = 0
    m.known_taps = {}
    pooled_tables, op_pools = {}, {}
    for opk, idxs in by_op.items():
        for k, f in enumerate(FLAGS):
            cands = {}
            for i in idxs:
                for t in (records[i]["effect"]["flags"].get("tap_cands") or {}).get(f, []) + [records[i]["effect"]["flags"]["taps"][f]]:
                    cands[json.dumps(t[:2])] = t
            ok = {}
            for c, t in cands.items():
                ok[c] = [i for i in idxs if np.array_equal(apply_tap(t, sigs[i][0], sigs[i][1][:, k]), sigs[i][2][:, k])]
            support = {c: sum(len(records[i]["rows"]) for i in ok[c]) for c in cands}
            groups_t = defaultdict(list)
            for i in idxs:
                good = [c for c in cands if i in set(ok[c])]
                if not good:
                    continue
                best = max(good, key=lambda c: (support[c], -len(cands[c][0]), c))
                newtap = cands[best][:2] + [0]
                if records[i]["effect"]["flags"]["taps"][f][:2] != newtap[:2]:
                    changed += 1
                records[i]["effect"]["flags"]["taps"][f] = newtap
                groups_t[best].append(i)
            # M5c version space of the SHARED rule: the records that took the same table pool their rows; every table up
            # to one signal beyond the minimal size that explains all of them stays an alternative (an imul whose rows
            # all had CF = 1 keeps 'hi_nfill' next to the constant; proof gate rec 303), and the chosen table's entries
            # none of them reached stay unobserved
            for best, members in groups_t.items():
                keys = set.intersection(*(set(sigs[i][0]) for i in members))
                psig = {kk: np.concatenate([np.asarray(sigs[i][0][kk]) for i in members]) for kk in keys}
                pold = np.concatenate([sigs[i][1][:, k] for i in members])
                pnew = np.concatenate([sigs[i][2][:, k] for i in members])
                tap = cands[best][:2] + [0]
                if not set(tap[0]) - {"old"} <= keys:  # members disagree on the signal set: no pooled evidence, abstain
                    for i in members:
                        records[i]["effect"]["flags"].setdefault("tap_alts", {})[f] = "overflow"
                    continue
                _, pool = tap_candidates(psig, pold, pnew, extra=1)
                for i in members:
                    pooled_tables[(i, f)] = pool  # the shared rule's version space (src_alts are checked against it)
                for t in pool:
                    op_pools.setdefault((opk, f), {}).setdefault(json.dumps(t[:2]), t)
                alts = [t for t in pool if t[:2] != tap[:2]]
                ev_alts = alts if len(alts) <= TAP_ALT_CAP else "overflow"
                unobs = _tap_unobserved(tap, psig, pold)
                for i in members:
                    fl_i = records[i]["effect"]["flags"]
                    fl_i.setdefault("tap_alts", {})[f] = ev_alts
                    fl_i.setdefault("tap_unobs", {})[f] = unobs
            if opk[0] == "table" and cands:
                top = max(cands, key=lambda c: support[c])
                m.known_taps.setdefault(opk[1], {})[f] = cands[top][:2] + [0]
    for t in list(m.known_taps):
        if len(m.known_taps[t]) != NF:
            del m.known_taps[t]
    # M5 flag-source alternatives: the low half of a product is the same signed or unsigned, so a 'mul lo' record's
    # rows may not decide the signedness its flags come from (M4 recs 330/1045/1809 took unsigned for imul; Z3/SLEIGH
    # refuted CF/OF). The other signedness stays an alternative, with every tap that explains the rows under it;
    # the machine abstains where any of them disagrees with the chosen flags.
    fallbacks = Counter()  # src_alt flags that kept the per-record fit (logged by the M5d rebuild)
    for i in sigs:
        fl = records[i]["effect"]["flags"]
        src = fl["src"]
        fl.pop("src_alts", None)
        alts = []
        if src[0] == "mul" and src[5] == "lo":
            alt = list(src)
            alt[4] = not src[4]
            alts.append(alt)
        alts += fl.get("src_cands") or []  # M5c: flag_search readings that also explain the rows
        # M5c': a reading equal to the chosen source adds nothing (its tables are the source's own version space,
        # already in tap_alts); proof gate recs 9660/9733 abstained everywhere on such duplicates
        runs = records[i]["effect"].get("runs") or []
        seen_src = {json.dumps(_canon_src(src, runs))}
        alts = [a for a in alts if not (json.dumps(_canon_src(a, runs)) in seen_src or seen_src.add(json.dumps(_canon_src(a, runs))))]
        out = []
        for alt in alts:
            r2 = {"effect": dict(records[i]["effect"], flags=dict(fl, src=alt)), "rows": records[i]["rows"]}
            sig2, before, after = record_signals(m, data, r2)
            cands = {}
            for k, f in enumerate(FLAGS):
                if op_key(alt) == op_key(src):
                    pool = pooled_tables.get((i, f))
                else:  # another op kind (imul vs mul, xor vs and): that kind's shared rules, pooled over its groups
                    pool = list(op_pools[(op_key(alt), f)].values()) if (op_key(alt), f) in op_pools else None
                # M5c'' (proof gate recs 9991/9575/...): the flag function is the op kind's shared rule, fitted on the
                # pooled member rows; an alternative source keeps only those pooled tables that also explain this
                # record's rows under it. Fitting on the record's own rows alone kept constant-ZF tables (its rows never
                # had a zero result), so the machine abstained exactly on the branch exit. An op kind with no pooled
                # records, or rows no pooled table explains, keeps the per-record fit.
                cc = alt_pooled_candidates(pool, sig2, before[:, k], after[:, k]) if pool is not None else []
                if not cc:
                    same = "same op" if op_key(alt) == op_key(src) else "other op"
                    if pool is not None and SRC_ALT_NONE_CONSISTENT == "drop":
                        # M5e (NOTES-m5 M5e #2): no table of the op kind's shared rule explains this record's rows under
                        # the alternative, so the alternative is refuted under the sharing assumption (e.g. `xor r,r` as
                        # the reading of `test r,r`: the shared xor rule would set ZF on every row). The per-record fit
                        # kept constant flag tables that its rows never refuted, and the machine abstained exactly where
                        # the record's rows were thin (zero results).
                        fallbacks[(same, f, "none consistent, dropped")] += 1
                        break
                    fallbacks[(same, f, "no pooled tables" if pool is None else "none consistent")] += 1
                    size, cc = tap_candidates(sig2, before[:, k], after[:, k], extra=1)
                    if size is None:
                        break
                cands[f] = cc[:TAP_ALT_CAP] if len(cc) <= TAP_ALT_CAP else "overflow"
            else:
                out.append([alt, cands])
        if out:
            fl["src_alts"] = out
    share_taps.last_fallbacks = {" | ".join(k): v for k, v in sorted(fallbacks.items())}
    return changed


def alt_pooled_candidates(pool, sig, old, target):
    """Tables of the shared (pooled) version space that are consistent with a record's rows under an alternative flag
    source; each keeps the pooled table's unobserved entries."""
    out = []
    for t in pool:
        if not set(t[0]) - {"old"} <= set(sig):
            continue
        if np.array_equal(apply_tap(t, sig, np.asarray(old, np.int64)), np.asarray(target, np.int64)):
            out.append(t)
    return out


def const_zf_alt_records(records):
    """Records whose chosen ZF is a real function but some flag-source alternative keeps a constant ZF (the machine
    abstains exactly on a zero result there; proof gate recs 9991/9575/...)."""
    out = []
    for i, r in enumerate(records):
        fl = r["effect"].get("flags") or {}
        if not fl.get("src_alts") or fl["taps"]["ZF"][0] in ([], ["a4"]):
            continue
        if any(isinstance(c.get("ZF"), list) and any(t[0] in ([], ["a4"]) for t in c["ZF"]) for _, c in fl["src_alts"]):
            out.append(i)
    return out


def _canon_src(e, runs):
    """A flag source with operand extension flags that cannot matter normalized away (M5c''; proof gate rec 9560:
    cmp r/m32, imm32 kept a reading that differed only in the 4-byte run's 's' vs 'z' at width 32): a table op masks
    both operands to its width, so a run, register or load at least that wide reads the same either way."""
    if not (isinstance(e, list) and e and e[0] == "table"):
        return e
    w = e[4]

    def norm(a):
        if not isinstance(a, list) or not a:
            return a
        if a[0] == "run" and len(a) == 3 and a[1] < len(runs) and 8 * runs[a[1]] >= w:
            return ["run", a[1], "z"]
        if a[0] == "reg" and len(a) == 4 and isinstance(a[2], int) and a[2] >= w:
            return ["reg", a[1], a[2], "z"]
        if a[0] == "load" and len(a) == 4 and isinstance(a[2], int) and 8 * a[2] >= w:
            return ["load", a[1], a[2], "z"]
        return a
    return [e[0], e[1], norm(e[2]), norm(e[3]), e[4]]


def src_alt_abstain(m, fl, x, old, flags):
    """Rows where an alternative flag source (M5) predicts some flag differently from the chosen one."""
    out = np.zeros(x.n, bool)
    for alt, cands in fl.get("src_alts") or []:
        _, sig = ev(m, alt, x, old, want_sig=True)
        for k, f in enumerate(FLAGS):
            o = x.g["flags"][:, k].astype(np.int64)
            if cands[f] == "overflow":
                out |= True
                continue
            for t in cands[f]:
                out |= _alt_abstain(t, sig, o, flags[:, k])
    return out


# ------------------------------------------------------------------ fields (MDL merge of records)

REG_POS = {"reg": (1,), "fill": (1,)}


def _abstract(eff):
    """Effect with register numbers and scales replaced by slot markers. Returns (template json, slots [(kind, value)])."""
    eff = json.loads(json.dumps({k: v for k, v in eff.items() if k not in ("tap_cands",)}))
    if eff.get("flags"):
        eff["flags"] = {k: v for k, v in eff["flags"].items() if k not in ("tap_cands", "src_cands")}
    def strip(e):
        if isinstance(e, list) and e:
            if e[0] in ("reg", "fill") and len(e) >= 3 and isinstance(e[1], int):
                return [e[0], "R"] + [strip(s) for s in e[2:]]
            if e[0] == "sum":
                return ["sum", "R" if isinstance(e[1], int) else e[1], "R" if isinstance(e[2], int) else e[2], "S" if e[2] is not None else 0, strip(e[4])]
            return [strip(s) for s in e]
        return e
    eff["writes"] = sorted(eff["writes"], key=lambda w: json.dumps(strip(w)))
    slots = []
    def walk(e):
        if isinstance(e, list) and e:
            if e[0] in ("reg", "fill") and len(e) >= 3 and isinstance(e[1], int):
                return [e[0], _slot(slots, "reg", e[1])] + [walk(s) for s in e[2:]]
            if e[0] == "sum":
                b = _slot(slots, "reg", e[1]) if isinstance(e[1], int) else e[1]
                i = _slot(slots, "reg", e[2]) if isinstance(e[2], int) else e[2]
                sc = _slot(slots, "scale", e[3]) if e[2] is not None else 0
                return ["sum", b, i, sc, walk(e[4])]
            return [walk(s) for s in e]
        if isinstance(e, dict):
            return {kk: walk(vv) for kk, vv in sorted(e.items())}
        return e
    tmpl = {"rip": walk(eff["rip"]), "writes": walk(eff["writes"]), "flags": walk(eff["flags"]), "trap": eff["trap"],
            "k": eff["k"], "runs": eff["runs"], "L": eff["L"]}
    return tmpl, slots


def _slot(slots, kind, value):
    for i, (k, v) in enumerate(slots):
        if k == kind and v == value:
            return {"$": i}
    slots.append((kind, value))
    return {"$": len(slots) - 1}


def bind(tmpl, values):
    def walk(e):
        if isinstance(e, dict) and "$" in e:
            return values[e["$"]]
        if isinstance(e, list):
            return [walk(s) for s in e]
        if isinstance(e, dict):
            return {k: walk(v) for k, v in e.items()}
        return e
    return walk(tmpl)


def _field_cands(k, kind):
    out = []
    lens = (3,) if kind == "reg" else (2,)
    for b in range(k):
        for ln in lens:
            for lo in range(0, 9 - ln):
                out.append((b, lo, ln, None))
                if kind == "reg":
                    for b2 in range(k):
                        if b2 != b:
                            for e in range(8):
                                out.append((b, lo, ln, (b2, e)))
    return out


def field_code(keys, f):
    """keys (R, k) uint8 → codes (R,)"""
    b, lo, ln, ext = f
    c = (keys[:, b].astype(np.int64) >> lo) & ((1 << ln) - 1)
    if ext is not None:
        c |= ((keys[:, ext[0]].astype(np.int64) >> ext[1]) & 1) << ln
    return c


def _functional(codes, vals):
    seen = {}
    for c, v in zip(codes.tolist(), vals):
        if seen.setdefault(c, v) != v:
            return None
    return seen


def learn_maps(clusters):
    """Global code -> register and code -> shift maps by votes of the best functional field per (cluster, slot)."""
    votes = {"reg": defaultdict(Counter), "scale": defaultdict(Counter)}
    ext_votes = defaultdict(Counter)
    for recs, tmpl, k in clusters:
        keys = np.array([list(r["key"]) for r in recs], np.uint8)
        slots = recs[0]["slots"]
        for j, (kind, _) in enumerate(slots):
            vals = [r["slots"][j][1] for r in recs]
            if len(set(vals)) < 3:
                continue
            best, maps_at_best, f_at_best = None, [], []
            for f in _field_cands(k, kind):
                codes = field_code(keys, f)
                mp = _functional(codes, vals)
                if mp is None:
                    continue
                score = (len(mp), -(f[2] + (f[3] is not None)))
                if best is None or score > best[0]:
                    best, maps_at_best, f_at_best = (score, mp, f), [mp], [f]
                elif score == best[0]:
                    maps_at_best.append(mp)
                    f_at_best.append(f)
            # vote only when the best field is unambiguous (few records make coincidental fields common)
            if best is not None and best[0][0] >= 3 and all(mp == best[1] for mp in maps_at_best):
                # M5: a vote weighs the cluster's member count (a 3-record cluster's coincidental field must not outvote
                # a 49-record one)
                for c, v in best[1].items():
                    votes[kind][c][v] += len(recs)
                if len({f[3] for f in f_at_best}) == 1 and best[2][3] is not None:
                    ext_votes[(kind, best[2][1], best[2][2])][best[2][3]] += len(recs)
    # M5: a contested code stays unmapped (records that need it abstain): the winner must hold at least twice the
    # vote weight of the runner-up
    def win(cnt):
        top = cnt.most_common(2)
        return top[0][0] if len(top) == 1 or top[0][1] >= 2 * top[1][1] else None
    out = {kind: {c: win(cnt) for c, cnt in vv.items() if win(cnt) is not None} for kind, vv in votes.items()}
    out["ext"] = {key: win(cnt) for key, cnt in ext_votes.items() if win(cnt) is not None}
    return out


def explain_cluster(recs, k, maps, dontcare=False):
    """Fields for every non-constant slot under the global maps. Returns (fields, mask bytes) or None."""
    keys = np.array([list(r["key"]) for r in recs], np.uint8)
    slots = recs[0]["slots"]
    fields = {}
    used = np.zeros((k, 8), bool)
    for j, (kind, _) in enumerate(slots):
        vals = [r["slots"][j][1] for r in recs]
        if len(set(vals)) == 1:
            continue
        mp = maps.get(kind, {})
        best = None
        for f in _field_cands(k, kind):
            codes = field_code(keys, f)
            if all(mp.get(c) == v for c, v in zip(codes.tolist(), vals)):
                # ties between extension bits the members cannot separate go to the globally voted one
                pref = f[3] is not None and maps.get("ext", {}).get((kind, f[1], f[2])) == tuple(f[3])
                score = (len(set(codes.tolist())), -(f[2] + (f[3] is not None)), pref)
                if best is None or score > best[0]:
                    best = (score, f)
        if best is None:
            return None
        f = best[1]
        fields[j] = f
        b, lo, ln, ext = f
        used[b, lo:lo + ln] = True
        if ext is not None:
            col = (keys[:, ext[0]] >> ext[1]) & 1
            voted = maps.get("ext", {}).get((kind, lo, ln)) == tuple(ext)
            # an extension bit the members never varied generalizes only if it is the globally voted one;
            # otherwise it stays a fixed bit of the record
            if col.min() != col.max() or voted:
                used[ext[0], ext[1]] = True
    varying = np.zeros((k, 8), bool)
    for b in range(k):
        for bit in range(8):
            col = (keys[:, b] >> bit) & 1
            varying[b, bit] = col.min() != col.max()
    free = varying & ~used
    if free.any() and not dontcare:
        return None
    fixed = ~(used | free)
    mask_bytes = bytes(int(sum(1 << bit for bit in range(8) if fixed[b, bit])) for b in range(k))
    return fields, mask_bytes


TAP_EVIDENCE = ("tap_alts", "tap_unobs", "src_alts")


def _norm_preds(t):
    """Template with every predicate reduced to (flags, table) and the taps' evidence dropped: the evidence parts do
    not decide the cluster."""
    if isinstance(t, list):
        if t and t[0] == "pred":
            return ["pred", t[1], t[2]]
        return [_norm_preds(e) for e in t]
    if isinstance(t, dict):
        return {k: _norm_preds(v) for k, v in t.items() if k not in TAP_EVIDENCE}
    return t


def _union_taps(lists):
    """Alternative tables of a merged record: every combo any member kept; entries observed by any member fix the value
    (members that disagree on an observed entry refute the combo); an entry stays unobserved only if no member saw it.
    A union abstains at least as often as each member (never less)."""
    if any(x == "overflow" for x in lists):
        return "overflow"
    by = {}
    for lst in lists:
        for t in lst:
            names = tuple(t[0])
            n = len(t[1])
            un = set(t[2]) if len(t) > 2 else set()
            cur = by.setdefault(names, {"tab": [None] * n, "dead": False})
            for e in range(n):
                if e in un:
                    continue
                if cur["tab"][e] is not None and cur["tab"][e] != t[1][e]:
                    cur["dead"] = True
                cur["tab"][e] = t[1][e]
    out = []
    for names, c in sorted(by.items()):
        if c["dead"]:
            continue
        out.append([list(names), [0 if v is None else v for v in c["tab"]], [e for e, v in enumerate(c["tab"]) if v is None]])
    return out[:TAP_ALT_CAP] if len(out) <= TAP_ALT_CAP else "overflow"


def _merge_src_alts(ts):
    """An alternative flag source survives the union only if every member kept it (a member without it refuted it);
    its taps are the union of the members' (abstain more, never less)."""
    by = [{json.dumps(a[0]): a[1] for a in (t.get("src_alts") or [])} for t in ts]
    out = []
    for key in by[0]:
        if not all(key in b for b in by):
            continue
        out.append([json.loads(key), {f: _union_taps([b[key][f] for b in by]) for f in by[0][key]}])
    return out


def _merge_tap_evidence(ts):
    """Union of the members' rows: alternatives are unioned by signal set; an entry of the chosen table is unobserved
    only if no member observed it."""
    alts, unobs = {}, {}
    for f in ts[0]["taps"]:
        alts[f] = _union_taps([(t.get("tap_alts") or {}).get(f, []) for t in ts])
        us = [set((t.get("tap_unobs") or {}).get(f, [])) for t in ts]
        unobs[f] = sorted(set.intersection(*us))
    return alts, unobs


def merge_preds(ps):
    """One predicate for the union of the members' rows: a candidate survives if every member's version space holds it
    with tables that agree on the entries both observed; unobserved entries are those no member observed."""
    def space(p):
        sp = {tuple(p[1]): (p[2], set(p[3]) if len(p) > 3 else set())}
        for combo, table, unobs in (p[4] if len(p) > 4 else []):
            sp[tuple(combo)] = (table, set(unobs))
        return sp
    spaces = [space(p) for p in ps]
    out = {}
    for combo in spaces[0]:
        if not all(combo in sp for sp in spaces):
            continue
        n = 1 << len(combo)
        tab, seen, ok = [0] * n, [False] * n, True
        for sp in spaces:
            table, unobs = sp[combo]
            for e in range(n):
                if e in unobs:
                    continue
                if seen[e] and tab[e] != table[e]:
                    ok = False
                tab[e], seen[e] = table[e], True
        if ok:
            out[combo] = (tab, [e for e in range(n) if not seen[e]])
    main = tuple(ps[0][1])
    tab, unobs = out.pop(main)  # the main predicate is shared by every member (same cluster signature)
    return ["pred", list(main), tab, unobs, [[list(c), t, u] for c, (t, u) in sorted(out.items())]]


def _merge_tmpl(ts):
    t0 = ts[0]
    if isinstance(t0, list):
        if t0 and t0[0] == "pred":
            return merge_preds(ts)
        return [_merge_tmpl([t[i] for t in ts]) for i in range(len(t0))]
    if isinstance(t0, dict):
        out = {k: _merge_tmpl([t[k] for t in ts]) for k in t0 if k not in TAP_EVIDENCE}
        if "taps" in t0 and any(k in t for t in ts for k in TAP_EVIDENCE):
            out["tap_alts"], out["tap_unobs"] = _merge_tap_evidence(ts)
            sa = _merge_src_alts(ts)
            if sa:
                out["src_alts"] = sa
        return out
    return t0


def merge_records(records, maps=None, min_cluster=2, log=print, dontcare=False):
    """Cluster literal records by (key length, abstract effect) and replace each cluster by one masked record."""
    for r in records:
        tmpl, slots = _abstract(r["effect"])
        r["tmpl"], r["slots"] = tmpl, slots
        r["sig"] = json.dumps([_norm_preds(tmpl), [k for k, _ in slots]], sort_keys=True)
    clusters = defaultdict(list)
    for r in records:
        clusters[(r["effect"]["k"], r["sig"])].append(r)
    cl = [(v, v[0]["tmpl"], k) for (k, _), v in clusters.items() if len(v) >= min_cluster]
    if maps is None:
        maps = learn_maps(cl)
    out, merged, kept = [], 0, 0
    for (k, sig), recs in clusters.items():
        groups = [recs]
        result = []
        depth = 0
        while groups and depth < 4:
            nxt = []
            for grp in groups:
                if len(grp) < min_cluster:
                    result += [("lit", r) for r in grp]
                    continue
                ex = explain_cluster(grp, k, maps, dontcare=dontcare)
                if ex is not None:
                    result.append(("merged", grp, ex))
                    continue
                keys = np.array([list(r["key"]) for r in grp], np.uint8)
                nvals = [len(set(keys[:, b].tolist())) for b in range(k)]
                cand = [b for b in range(k) if nvals[b] > 1]
                if not cand:
                    result += [("lit", r) for r in grp]
                    continue
                b = min(cand, key=lambda b: (nvals[b], b))
                for v in sorted(set(keys[:, b].tolist())):
                    nxt.append([r for r in grp if r["key"][b] == v])
            groups = nxt
            depth += 1
        for grp in groups:
            result += [("lit", r) for r in grp]
        for item in result:
            if item[0] == "lit":
                r = item[1]
                out.append({"key": r["key"], "mask": b"\xff" * k, "fields": {}, "tmpl": r["tmpl"], "consts": [v for _, v in r["slots"]],
                            "slot_kinds": [kk for kk, _ in r["slots"]], "rows": r["rows"], "members": 1})
                kept += 1
            else:
                _, grp, (fields, mb) = item
                consts = [grp[0]["slots"][j][1] if j not in fields else None for j in range(len(grp[0]["slots"]))]
                out.append({"key": grp[0]["key"], "mask": mb, "fields": fields, "tmpl": _merge_tmpl([r["tmpl"] for r in grp]), "consts": consts,
                            "slot_kinds": [kk for kk, _ in grp[0]["slots"]], "rows": np.concatenate([r["rows"] for r in grp]),
                            "members": len(grp)})
                merged += 1
    log(f"[m4] merge: {len(records)} literal records -> {len(out)} ({merged} masked, {kept} literal); maps {len(maps.get('reg', {}))} reg codes, {len(maps.get('scale', {}))} scale codes")
    return out, maps


def _exact_by_code(m, maps, r, data, rows, codes, k, cap=64):
    """Rows (of the record's length) checked under the record's own template bound with each row's field codes:
    (exact, refuted) boolean arrays. A row the template abstains on, or whose code has no map entry, is neither."""
    exact = np.zeros(len(rows), bool)
    refuted = np.zeros(len(rows), bool)
    js = sorted(r["fields"])
    tup = np.stack([codes[j] for j in js], 1) if js else np.zeros((len(rows), 0), np.int64)
    groups = defaultdict(list)
    for t, c in enumerate(map(tuple, tup.tolist())):
        if len(groups[c]) < cap:
            groups[c].append(t)
    for c, ts in groups.items():
        vals = []
        for j, kind in enumerate(r["slot_kinds"]):
            if r["consts"][j] is not None:
                vals.append(r["consts"][j])
                continue
            v = maps.get(kind, {}).get(c[js.index(j)])
            if v is None:
                vals = None
                break
            vals.append(v)
        if vals is None:
            continue
        eff = bind(r["tmpl"], vals)
        ts = np.array(ts)
        x = Ctx(m, data, rows[ts], k, eff["runs"])
        try:
            ok, ab = check_rows(m, eff, x)
        except Exception:  # noqa: BLE001 - an effect that cannot run on these rows is no evidence either way
            continue
        exact[ts] = ok
        refuted[ts] = ~ok & ~ab
    return exact, refuted


def _implied_ext(r, f, bit):
    """The extension-bit value a field without one carries in its record's key: the fixed REX bit at `bit` when the key
    has a REX byte (40-4f) before the field byte, 0 when it has none, None when the REX bit is not fixed."""
    for p in range(min(f[0], 2)):
        if (r["mask"][p] & 0xF0) == 0xF0 and (r["key"][p] & 0xF0) == 0x40:
            return (r["key"][p] >> bit) & 1 if (r["mask"][p] >> bit) & 1 else None
    return 0


def _pool_allow(records, local, ctx, full_code=True):
    """Pooled field evidence -> each record's allowed codes. full_code (M5c''; proof gate recs 1603/7825/5617): sibling
    evidence counts by the FULL code, extension bit included -- SIB index 100 is r12 with REX.X = 1 and "no index"
    with X = 0, so low bits seen good under one extension value must not open them under the other. A field without an
    extension bit borrows under its key's implied value (the fixed REX bit, 0 without REX). Blame (another length) stays
    pooled on the low bits, so it closes every extension value (full-code blame let rm = 12 / SIB through for cmov:
    strat wrong 0.51 -> 0.67%, job-a28a1263e580). full_code=False is the M5c/M5c' low-bit pooling (negative control)."""
    bits = defaultdict(Counter)
    for ri, per in local.items():
        r = records[ri]
        for j in per:
            f = r["fields"][j]
            if f[3] is not None:
                bits[ctx(r, f)][f[3][1]] += 1

    def full(r, f, c, codes):
        lm = (1 << f[2]) - 1
        if not full_code:
            return {x & lm for x in codes}
        if f[3] is not None:
            return set(codes)
        b = bits[c].most_common(1)[0][0] if bits[c] else None
        has_rex = any((r["mask"][p] & 0xF0) == 0xF0 and (r["key"][p] & 0xF0) == 0x40 for p in range(min(f[0], 2)))
        e = _implied_ext(r, f, b) if b is not None else (None if has_rex else 0)
        if e is None:
            return None
        return {(x & lm) | (e << f[2]) for x in codes}
    good_p, bad_p = defaultdict(set), defaultdict(set)
    for ri, per in local.items():
        r = records[ri]
        for j, (bad, good, _) in per.items():
            f = r["fields"][j]
            c = ctx(r, f)
            g = full(r, f, c, good)
            if g is not None:
                good_p[c] |= g
            bad_p[c] |= {x & ((1 << f[2]) - 1) for x in bad}  # blame stays on the low bits: it closes every extension value
    n_closed = n_rec = 0
    for ri, per in local.items():
        r = records[ri]
        allow = {}
        for j, (bad, good, bad_ref) in per.items():
            f = r["fields"][j]
            c = ctx(r, f)
            ncode = 1 << (f[2] + (f[3] is not None))
            ok = set()
            for x in range(ncode):
                if x in good:
                    ok.add(x)
                    continue
                if x in bad or x in bad_ref:
                    continue
                px = full(r, f, c, [x])
                if px and next(iter(px)) in good_p[c] and (x & ((1 << f[2]) - 1)) not in bad_p[c]:
                    ok.add(x)
            allow[j] = frozenset(ok)
            n_closed += ncode - len(ok)
        r["allow"] = allow
        n_rec += 1
    return good_p, bad_p, n_closed, n_rec


def field_coverage(data, records, log=print, m=None, maps=None, pool_refutation=True, trace=None, sib_pool=False, refuted_good=True,
                   full_code=True, diag=None):
    """Negative evidence for field generalization. For a masked record that falls through, look at every training row
    whose key matches the record's fixed bits: a field code that occurs only in rows of a different length than the
    record (and never in rows of the record's length), as the one such code in its row, is excluded -- that code
    selects a different key shape there.
    Every other code, seen or unseen, stays open (the field generalizes).
    M5c': SIB fields are their own context (sib_pool=True = M5c, which pooled SIB scale 11 with ModRM mod 11), and a
    code with a refuted row whose other fields sit at the record's own codes is not good evidence (refuted_good=False
    = M5c). Refutation blame stays pooled like other-length blame (pool_refutation=True): keeping it local (tried in
    M5c', job-0119589fb4f9) raised strat-panel wrong from 0.81% to 1.46%. trace: list to receive (record, field, code, row, kind)."""
    with np.errstate(over="ignore"):
        Lr = (data["a_rip"] - data["rip"]).astype(np.int64)
    W = data["window"]
    ok_rows = np.flatnonzero((Lr >= 1) & (Lr <= 15) & (data["trap"] == 0))
    buckets = defaultdict(list)
    for i, (a, b) in zip(ok_rows.tolist(), W[ok_rows, :2].tolist()):
        buckets[(a, b)].append(i)
    buckets = {kk: np.array(v) for kk, v in buckets.items()}
    local = {}  # record index -> {field j: (excluded codes, codes seen at the record's length)}
    for ri, r in enumerate(records):
        r.pop("coverage", None)
        r.pop("exclude", None)
        r.pop("allow", None)
        if not r["fields"]:
            continue
        k, mk, ky = len(r["key"]), r["mask"], r["key"]
        if r["tmpl"]["rip"] != ["next"]:
            # length unobservable (control transfer): evidence = the record's own rows
            own = {j: set(field_code(W[r["rows"], :k], f).tolist()) for j, f in r["fields"].items()}
            local[ri] = {j: (set(), own[j], set()) for j in own}
            continue
        b1s = range(256) if k < 2 else [b for b in range(256) if (b & mk[1]) == (ky[1] & mk[1])]
        b0s = [b for b in range(256) if (b & mk[0]) == (ky[0] & mk[0])]
        cand = [buckets[(a, b)] for a in b0s for b in b1s if (a, b) in buckets]
        if not cand:
            own = {j: set(field_code(W[r["rows"], :k], f).tolist()) for j, f in r["fields"].items()}
            local[ri] = {j: (set(), own[j], set()) for j in own}
            continue
        rows = np.concatenate(cand)
        sel = np.ones(len(rows), bool)
        for b in range(2, k):
            sel &= (W[rows, b] & mk[b]) == (ky[b] & mk[b])
        rows = rows[sel]
        same = Lr[rows] == r["tmpl"]["L"]
        keys = W[rows, :k]
        codes = {j: field_code(keys, f) for j, f in r["fields"].items()}
        if m is not None and maps is not None:
            # M5c: a code is good evidence only where the record's own template is EXACT on its rows; a row of the
            # record's length that the template gets wrong is other-meaning evidence (e.g. SIB index 100 = no index,
            # which keeps the length but drops the register)
            si = np.flatnonzero(same)
            ex, rf = _exact_by_code(m, maps, r, data, rows[si], {j: c[si] for j, c in codes.items()}, k)
            exact_rows = np.zeros(len(rows), bool)
            exact_rows[si] = ex
            ref_rows = np.zeros(len(rows), bool)
            ref_rows[si] = rf
            own = {j: set(field_code(W[r["rows"], :k], f).tolist()) for j, f in r["fields"].items()}
            good = {j: set(c[exact_rows].tolist()) | own[j] for j, c in codes.items()}
            if refuted_good:
                # M5c' (proof gate rec 9024: SIB index 100 = no index read as RSP; 63 refuted vs 10 accidentally
                # exact rows): refuted evidence beats exact evidence for a code.
                # M5c'' (gate recs 7825/5617: 10% of index-100 rows exact by accident, the rest refuted with several
                # unseen codes per row): any refuted row takes its codes out of the good evidence (the record's own
                # codes stay); a code it wrongly drops only abstains
                for t in np.flatnonzero(ref_rows):
                    for j, c in codes.items():
                        if int(c[t]) not in own[j]:
                            good[j].discard(int(c[t]))
            other = ~same | ref_rows
        else:
            good = {j: set(c[same].tolist()) for j, c in codes.items()}
            other = ~same
        bad = {j: set() for j in codes}  # other-length blame: about key shape, pooled over sibling contexts below
        bad_ref = {j: set() for j in codes}  # refutation blame: about this record's meaning, kept local (M5c')
        for t in np.flatnonzero(other):
            # a row of another length blames a field only when that field is the one unseen code in the row
            culprits = [j for j, c in codes.items() if int(c[t]) not in good[j]]
            if len(culprits) == 1:
                ref = bool(same[t])
                (bad if not ref or pool_refutation else bad_ref)[culprits[0]].add(int(codes[culprits[0]][t]))
                if trace is not None:
                    trace.append((ri, culprits[0], int(codes[culprits[0]][t]), int(rows[t]), "ref" if ref else "len"))
        local[ri] = {j: (bad[j], good[j], bad_ref[j]) for j in codes}
    # Pool the evidence over every record that has a field at the same bit position with the same fixed bits in the
    # field's byte -- across key lengths, byte positions and REX/non-REX siblings -- on the field's low bits (an
    # extension bit only widens the code). A code is OPEN for a record if it was seen at the record's length in the
    # record's own rows, or (low bits) in any sibling context; it is CLOSED if its low bits ever selected another key
    # length in a sibling context and the record never saw it work. Unseen without evidence = closed (abstain).
    def ctx(r, f):
        # M5c': a SIB byte (after a fixed ModRM with rm = 100, mod != 11) is its own context: SIB scale 11 has the same
        # fixed bits as ModRM mod 11, but base 101 there means "no base, disp32" (another length)
        b = f[0]
        sib = (not sib_pool and b >= 1 and (r["mask"][b - 1] & 0xC7) == 0xC7 and (r["key"][b - 1] & 7) == 4
               and (r["key"][b - 1] & 0xC0) != 0xC0)
        return (f[1], f[2], r["mask"][b], r["key"][b] & r["mask"][b], sib)
    good_p, bad_p, n_closed, n_rec = _pool_allow(records, local, ctx, full_code)
    n_byte = byte_reg_guard(records, W) if BYTE_REG_GUARD else 0
    n_byte += sib_index_guard(records) if BYTE_REG_GUARD else 0
    if diag is not None:
        diag.update(good_p=good_p, bad_p=bad_p, local=local, ctx=ctx)

    log(f"[m4] field coverage: {n_rec} records with field evidence; {n_closed} field codes closed (no evidence or other-length evidence)"
        f"; byte-register and SIB-index guards closed {n_byte} more")
    return records


def byte_reg_slots(tmpl):
    """Slots a template uses as an 8-bit register part (read, sign fill or write)."""
    out = set()

    def walk(e):
        if isinstance(e, list):
            if len(e) >= 3 and e[0] in ("reg", "fill") and isinstance(e[1], dict) and "$" in e[1] and e[2] == 8:
                out.add(e[1]["$"])
            for s in e:
                walk(s)
        elif isinstance(e, dict):
            for v in e.values():
                walk(v)
    walk(tmpl)
    return out


LEGACY_PREFIXES = frozenset((0x66, 0x67, 0xF2, 0xF3, 0xF0, 0x2E, 0x36, 0x3E, 0x26, 0x64, 0x65))


def rex_state(r, f):
    """'rex' if the key fixes a REX byte (40-4f) before the field's byte, 'none' if the first fixed byte after any fixed
    legacy prefixes is not one (that byte is the opcode, and REX can only come right before it), None if a byte before
    it does not fix its high nibble (REX presence varies across the record's keys)."""
    for p in range(f[0]):
        if (r["mask"][p] & 0xF0) != 0xF0:
            return None
        if (r["key"][p] & 0xF0) == 0x40:
            return "rex"
        if r["mask"][p] == 0xFF and r["key"][p] in LEGACY_PREFIXES:
            continue
        return "none"
    return "none"


def byte_reg_guard(records, W):
    """M5e (NOTES-m5 M5e #6; proof gate recs 337/409/360): an 8-bit register field in a record without a fixed REX byte
    keeps codes 4-7 closed unless the record's own (exactly fitted) rows used them. Without REX those codes name
    AH/CH/DH/BH; sibling evidence from REX records (SPL..DIL) must not open them. Returns the number of codes closed."""
    n = 0
    for r in records:
        al = r.get("allow")
        if not r["fields"] or al is None:
            continue
        slots = byte_reg_slots(r["tmpl"])
        for j, f in r["fields"].items():
            if j not in slots or j not in al or f[3] is not None or rex_state(r, f) == "rex":
                continue
            own = set(field_code(W[r["rows"], :len(r["key"])], f).tolist())
            drop = {c for c in al[j] if 4 <= (c & 7) <= 7 and c not in own}
            if drop:
                al[j] = frozenset(al[j] - drop)
                n += len(drop)
    return n


def sib_index_slots(tmpl):
    """Template slots read as an address index (["sum", base, {"$": j}, scale, disp])."""
    out = set()

    def walk(e):
        if isinstance(e, list):
            if e and e[0] == "sum" and isinstance(e[2], dict) and "$" in e[2]:
                out.add(e[2]["$"])
            for s in e:
                walk(s)
        elif isinstance(e, dict):
            for v in e.values():
                walk(v)
    walk(tmpl)
    return out


def sib_index_guard(records):
    """M5e (NOTES-m5 M5e item 1; proof gate records 1603/5617/7825): a SIB index field never takes code 100 without
    REX.X: that encoding means "no index", so no row of the record can have shown it as a register, and pooled
    evidence that opened it binds RSP as an index. Returns the number of codes closed."""
    n = 0
    for r in records:
        al = r.get("allow")
        if not r["fields"] or al is None:
            continue
        idx = sib_index_slots(r["tmpl"])
        for j, f in r["fields"].items():
            if j in idx and j in al and f[1] == 3 and f[2] == 3 and 4 in al[j]:
                al[j] = frozenset(al[j] - {4})
                n += 1
    return n


def sib_run_evidence(m, data, records, maps, log=print):
    """M5c' (proof gate rec 105, `88 04` runs [1, 4]): a literal record whose key ends on a ModRM byte with rm = 100 and
    mod != 11 reads the next byte (the SIB) as part of a data run. That is admitted only with evidence that the byte is
    data: the training rows with the record's key take at least two values in each of the byte's 3-bit groups (bits
    0-2 and 3-5) and the record is exact on every one of them (RIP included). Otherwise the record is dropped
    (abstain). The rule cannot tell an opcode byte from a ModRM (`0f 84` = jcc rel32 also ends in rm = 100); such
    records pass when their rows vary."""
    W = data["window"]
    first = defaultdict(list)
    for i, b in enumerate(W[:, 0].tolist()):
        first[b].append(i)
    first = {b: np.array(v) for b, v in first.items()}
    drop = []
    for ri, r in enumerate(records):
        k = len(r["key"])
        if r["fields"] or k < 2 or bytes(r["mask"]) != b"\xff" * k or not any(r["tmpl"].get("runs") or ()):
            continue
        b = r["key"][k - 1]
        if (b & 7) != 4 or (b >> 6) == 3:
            continue
        rows = first.get(r["key"][0], np.zeros(0, np.int64))
        rows = rows[np.all(W[rows, :k] == np.frombuffer(bytes(r["key"]), np.uint8), 1)]
        vals = W[rows, k].astype(np.int64)
        varied = len(set((vals & 7).tolist())) >= 2 and len(set(((vals >> 3) & 7).tolist())) >= 2
        ok = varied
        if ok:  # exact on every row of the key (RIP included, so a SIB-dependent length refutes it)
            good, _ = score(m, Decoder([r], maps), data, rows)
            ok = bool(good.all())
        if not ok:
            drop.append(ri)
    log(f"[m4] SIB-run evidence: {len(drop)} literal records reading a SIB byte as data dropped")
    return [r for i, r in enumerate(records) if i not in set(drop)]


def resolve_collisions(m, data, records, maps, log=print):
    """Records with the same (mask, fields, masked key) but different templates are alternative explanations that
    were each exact on their own rows (e.g. a write width or a signedness that the group's rows could not separate).
    Keep the template that is exact on the most rows of the union (ties: more member keys)."""
    by = defaultdict(list)
    for i, r in enumerate(records):
        by[(r["mask"], json.dumps({str(j): f for j, f in sorted(r["fields"].items())}), bytes(a & b for a, b in zip(r["key"], r["mask"])))].append(i)
    drop, n_groups, n_full = set(), 0, 0
    for ii in by.values():
        if len(ii) < 2:
            continue
        n_groups += 1
        rows = np.unique(np.concatenate([records[i]["rows"] for i in ii]))
        best = None
        for i in ii:
            ok, _ = score(m, Decoder([records[i]], maps), data, rows)
            sc = (int(ok.sum()), records[i]["members"])
            if best is None or sc > best[0]:
                best = (sc, i)
        win = best[1]
        n_full += best[0][0] == len(rows)
        records[win] = dict(records[win], rows=rows, members=sum(records[i]["members"] for i in ii))
        drop |= set(ii) - {win}
    log(f"[m4] collisions: {n_groups} groups, winner exact on every union row in {n_full}; dropped {len(drop)} records")
    return [r for i, r in enumerate(records) if i not in drop]


# ------------------------------------------------------------------ lookup and scoring


class Decoder:
    def __init__(self, records, maps):
        self.records = records
        self.maps = maps
        self.index = defaultdict(list)
        for i, r in enumerate(records):
            kb, mb = r["key"], r["mask"]
            b0s = [b for b in range(256) if (b & mb[0]) == (kb[0] & mb[0])]
            if len(kb) == 1:
                for b0 in b0s:
                    self.index[(b0,)].append(i)
                continue
            b1s = [b for b in range(256) if (b & mb[1]) == (kb[1] & mb[1])]
            for b0 in b0s:
                for b1 in b1s:
                    self.index[(b0, b1)].append(i)
        self.spec = [sum(bin(x).count("1") for x in r["mask"]) for r in records]
        self.cache = {}
        self.maxk = max((len(r["key"]) for r in records), default=2)
        self.maxk = max(self.maxk, 2)

    def _match(self, wb):
        """Most specific record whose fixed bits match and none of whose field codes is excluded."""
        best = None
        for i in self.index.get((wb[0],), []) + self.index.get((wb[0], wb[1]), []):
            r = self.records[i]
            k = len(r["key"])
            if not all((wb[j] & r["mask"][j]) == (r["key"][j] & r["mask"][j]) for j in range(k)):
                continue
            al = r.get("allow")
            if r["fields"]:
                if al is None:
                    continue  # a field record without evidence (e.g. not falling through) stays closed
                keyarr = np.frombuffer(wb[:k], np.uint8)[None, :]
                if any(int(field_code(keyarr, r["fields"][j])[0]) not in al.get(j, ()) for j in r["fields"]):
                    continue
            score = (self.spec[i], len(r["rows"]))
            if best is None or score > best[0]:
                best = (score, i)
        return None if best is None else best[1]

    def lookup(self, window):
        """(record index, concrete effect) for the most specific matching record, or (None, None)."""
        wb = bytes(window[:self.maxk])
        hit = self.cache.get(wb)
        if hit is not None:
            return hit
        best = self._match(wb)
        if best is None:
            self.cache[wb] = (None, None)
            return None, None
        i = best
        r = self.records[i]
        k = len(r["key"])
        vals = []
        keyarr = np.frombuffer(wb[:k], np.uint8)[None, :]
        for j, kind in enumerate(r["slot_kinds"]):
            if r["consts"][j] is not None:
                vals.append(r["consts"][j])
                continue
            code = int(field_code(keyarr, r["fields"][j])[0])
            v = self.maps.get(kind, {}).get(code)
            if v is None:
                vals = None
                break
            vals.append(v)
        out = (i, None if vals is None else bind(r["tmpl"], vals))
        self.cache[wb] = out
        return out


def score(m, dec, data, rows=None):
    """Exactness of the learned machine on transition rows. Returns per-row ok, abstain arrays."""
    N = len(data["rip"])
    rows = np.arange(N) if rows is None else np.asarray(rows)
    ok = np.zeros(N, bool)
    abst = np.zeros(N, bool)
    groups = defaultdict(list)
    for i in rows:
        ri, eff = dec.lookup(data["window"][i])
        if eff is None:
            abst[i] = True
            continue
        groups[json.dumps(eff, sort_keys=True)].append(i)
    for ej, rr in groups.items():
        eff = json.loads(ej)
        x = Ctx(m, data, np.array(rr), eff["k"], eff["runs"])
        try:
            o, a = check_rows(m, eff, x)
        except Exception:
            o, a = np.zeros(len(rr), bool), np.ones(len(rr), bool)
        ok[rr] = o
        abst[rr] = a
    return ok[rows], abst[rows]


def learner_store(data):
    """The learner's view of before-memory: the row's page store with the fetched window at RIP."""
    ov = []
    for o, w, rip in zip(data["overrides"], data["window"], data["rip"].tolist()):
        o = dict(o)
        for i, b in enumerate(w.tolist()):
            o[(rip + i) & MASK64] = b
        ov.append(o)
    return PageStore(data["seed"], data["mode"], ov)


def learn(data, log=print, max_k=7, workers=1, salvage=None, partial=None):
    """salvage(stage, state): called after each fitting pass so a killed lease keeps its work; partial: a state from
    an earlier salvage of the SAME rows, whose finished passes are skipped."""
    import time
    t0 = time.time()
    if partial is not None and partial.get("stage") == "pass2":
        m, recs1, records, pending = partial["m"], partial["recs1"], partial["records"], partial["pending"]
        st1, st2 = Counter(partial["st1"]), Counter(partial["st2"])
        adder = partial["adder"]
        data["store"] = learner_store(data)
        log(f"[m4] resumed after pass 2 from salvage: {len(records)} literal records")
        return _learn_finish(m, data, records, pending, st1, st2, adder, log, t0)
    m = Machine()
    data["store"] = learner_store(data)
    N = len(data["rip"])
    with np.errstate(over="ignore"):
        L = (data["a_rip"] - data["rip"]).astype(np.int64)
    groups = defaultdict(list)
    for i in range(N):
        if 1 <= L[i] <= 15:
            groups[(int(L[i]), bytes(data["window"][i][:L[i] - 1]))].append(i)
    gl = [(k, np.array(v)) for k, v in sorted(groups.items(), key=lambda kv: -len(kv[1]))]
    bootstrap_tables(m, data, gl)
    adder = choose_adder(m, data)
    log(f"[m4] bootstrap: {len(m.tables)} tables, adder t{adder[0]} ({adder[1]}/{adder[2]} rows), adder == integer add: {m.adder_ok}, t={time.time() - t0:.0f}s")
    if partial is not None and partial.get("stage") == "pass1":
        m, recs1, pending, st1 = partial["m"], partial["recs1"], partial["pending"], Counter(partial["st1"])
        log(f"[m4] resumed after pass 1 from salvage: {len(recs1)} records")
    else:
        recs1, pending, st1 = fit_trie(m, data, allow_flag_search=False, log=log, max_k=max_k, workers=workers)
        if salvage is not None:
            salvage("pass1", {"stage": "pass1", "m": m, "recs1": recs1, "pending": pending, "st1": dict(st1), "adder": adder})
    log(f"[m4] pass 1: {len(recs1)} records; fail stats {dict(st1.most_common(12))}")
    if "o_form" in data:  # oracle column, used for this log line only
        log(f"[m4] pass 1 pending by form (report only): {Counter(data['o_form'][pending].tolist()).most_common(25)}")
    share_taps(m, data, recs1)
    log(f"[m4] known taps for tables {sorted(getattr(m, 'known_taps', {}))}, t={time.time() - t0:.0f}s")
    recs2, pending, st2 = fit_trie(m, data, rows=np.flatnonzero(pending), allow_flag_search=True, log=log, max_k=max_k, workers=workers)
    records = recs1 + recs2
    if salvage is not None:
        salvage("pass2", {"stage": "pass2", "m": m, "recs1": recs1, "records": records, "pending": pending, "st1": dict(st1),
                          "st2": dict(st2), "adder": adder})
    return _learn_finish(m, data, records, pending, st1, st2, adder, log, t0)


def _learn_finish(m, data, records, pending, st1, st2, adder, log, t0):
    import time
    changed = share_taps(m, data, records)
    log(f"[m4] fitted {len(records)} literal records; taps changed by sharing {changed}; pending rows {int(pending.sum())}; t={time.time() - t0:.0f}s")
    merged, maps = merge_records(records, log=log)
    merged = resolve_collisions(m, data, merged, maps, log=log)
    merged = sib_run_evidence(m, data, merged, maps, log=log)
    merged = field_coverage(data, merged, log=log, m=m, maps=maps)
    dec = FabricDecoder(merged, maps)
    m.literal_records = records
    m.records = merged
    m.code_map = maps
    info = {"tables": [t.describe() for t in m.tables], "adder": adder[0], "adder_ok": m.adder_ok, "literal_records": len(records),
            "records": len(merged), "pending_rows": int(pending.sum()), "fail_stats": dict((st1 + st2).most_common(40)),
            "maps": {k: {str(c): v for c, v in mp.items()} for k, mp in maps.items()}, "fit_s": round(time.time() - t0, 1)}
    return m, dec, info, pending


class FabricDecoder(Decoder):
    """Decoder whose records live in a FieldLM FieldFabric view ("microcode-m4") and are served by a FabricProgram
    read keyed by (shape, masked key). The byte index only proposes which shape matches; the record body (effect
    template, constants, field specs) comes from the fabric read."""

    def __init__(self, records, maps):
        super().__init__(records, maps)
        from isa_lab.dependencies import field_modules
        api, chart = field_modules()
        self.api = api
        self.fabric = api.FieldFabric()
        view = chart.RoleChartField.from_state({"charts": [], "classes": {}, "merge_log": [], "role_class_merges": [],
                                                "records": [], "active_axes": None})
        self.fabric.register_view("microcode-m4", view)
        self.fabric.manage_view("microcode-m4")
        rows = []
        self.shape_of = []
        for i, r in enumerate(records):
            shape = json.dumps([r["mask"].hex(), {str(j): list(f[:3]) + [list(f[3]) if f[3] else None] for j, f in r["fields"].items()}])
            key = bytes(a & b for a, b in zip(r["key"], r["mask"])).hex()
            body = json.dumps({"tmpl": r["tmpl"], "consts": r["consts"], "slot_kinds": r["slot_kinds"]}, sort_keys=True)
            rows.append(chart.Record("isa", "x86-m4", 0, (shape, key, body), f"m4-{i}", i))
            self.shape_of.append((shape, key))
        self.fabric.mutate_records("microcode-m4", add=rows)
        self.fabric._refresh_view_indices()
        self.program = api.FabricProgram("read-m4", (api.ReadStep("microcode-m4", 0, {0: "$shape", 1: "$key"}, {2: "body"}),), ("body",))
        self.reads = 0
        self._bodies = {}

    def body(self, i):
        if i not in self._bodies:
            shape, key = self.shape_of[i]
            reader = self.fabric.__class__.__new__(self.fabric.__class__)
            reader.__dict__.update(self.fabric.__dict__)
            reader.frames, reader.abstractions = {}, {}
            r = reader.run(self.program, {"shape": shape, "key": key}, "isa", "x86-m4")
            self.reads += 1
            self._bodies[i] = json.loads(r["value"]) if r["status"] == "resolved" else None
        return self._bodies[i]

    def lookup(self, window):
        wb = bytes(window[:self.maxk])
        hit = self.cache.get(wb)
        if hit is not None:
            return hit
        best = self._match(wb)
        if best is None:
            self.cache[wb] = (None, None)
            return None, None
        i = best
        r = self.records[i]
        b = self.body(i)
        if b is None:
            self.cache[wb] = (i, None)
            return i, None
        k = len(r["key"])
        keyarr = np.frombuffer(wb[:k], np.uint8)[None, :]
        vals = []
        for j, kind in enumerate(b["slot_kinds"]):
            if b["consts"][j] is not None:
                vals.append(b["consts"][j])
                continue
            code = int(field_code(keyarr, r["fields"][j])[0])
            v = self.maps.get(kind, {}).get(code)
            if v is None:
                vals = None
                break
            vals.append(v)
        out = (i, None if vals is None else bind(b["tmpl"], vals))
        self.cache[wb] = out
        return out


def _rebuild_fabric_decoder(records, maps):
    return FabricDecoder(records, maps)


FabricDecoder.__reduce__ = lambda self: (_rebuild_fabric_decoder, (self.records, self.maps))


# ---- M5e shared effective-address unit ----
# M5e shared effective-address unit (NOTES-m5 M5e #7): records factored into an operation part and an address part,
# each pooled across the other, and composed where no whole record matches.
# 
# M5d keeps the address arithmetic inside every record, per operand shape: `mov [rsp+0x20], rcx` needs a record (or a
# merged family) for exactly that opcode and that addressing shape. Most address records are literal (one key each), and
# a field code is opened by pooled evidence whose context ignores the opcode, so `81 /1` (or [mem], imm32) closes the
# register code 1 of `48 89 /r` (m5e-diag job-274c7327f9ed, rec 4869). This module factors every literal record whose
# effect uses exactly one memory address into
#   an OPERATION pseudo-record  key = bytes up to and including the addressing byte, with the address bits cleared,
#                               effect = the record's effect with the address replaced by a placeholder and the
#                               displacement run removed;
#   an ADDRESS pseudo-record    key = (REX address bits or 0, addressing byte with the operation bits cleared, SIB byte
#                               if any), effect = the address sum (base, index << scale, displacement run),
# and runs the M4 merge (clusters by template, fields under the global code maps) over each family separately. A window
# with no whole-record match is decoded as the most specific operation record on its leading bytes plus the most
# specific address record on its addressing bytes; two readings of the same bytes abstain.
# 
# Supplied (reported with the results): the bit partition between the parts: in the REX byte (found after legacy
# prefixes) W/R belong to the operation, X/B to the address; in the addressing byte bits 3-5 belong to the operation,
# mod and rm to the address; an address may continue into one more byte. Learned/evidenced: which byte of each literal
# record is its addressing byte (the one whose bits reproduce the record's own address through the learned code maps),
# the operation and address templates, their fields, and the open codes: an operation field takes every mapped code
# except codes that an operation pseudo-record of another template carries under the same fixed bits (that code selects
# another operation); an address field takes only the codes its own members carried, across all opcodes (address codes
# have specials: rm 100 = SIB, mod 00 rm 101 = RIP; a code no member showed is not evidenced).

EA = ["ea"]  # the address placeholder inside an operation template
REX_OP, REX_ADDR = 0xFC, 0x03  # W R | X B
MODRM_OP, MODRM_ADDR = 0x38, 0xC7  # reg | mod rm


def rex_pos(key):
    p = 0
    while p < len(key) and key[p] in LEGACY_PREFIXES:
        p += 1
    return p if p < len(key) and (key[p] & 0xF0) == 0x40 else None


def opcode_pos(key):
    p = 0
    while p < len(key) and key[p] in LEGACY_PREFIXES:
        p += 1
    return p + 1 if p < len(key) and (key[p] & 0xF0) == 0x40 else p


def _sums(e, out):
    if isinstance(e, list):
        if e and e[0] == "sum":
            out.append(e)
        for s in e:
            _sums(s, out)
    elif isinstance(e, dict):
        for v in e.values():
            _sums(v, out)
    return out


def _replace_sum(e, sub):
    if isinstance(e, list):
        if e and e[0] == "sum":
            return sub
        return [_replace_sum(s, sub) for s in e]
    if isinstance(e, dict):
        return {k: _replace_sum(v, sub) for k, v in e.items()}
    return e


def _fill_ea(e, s):
    if isinstance(e, list):
        if e == EA:
            return s
        return [_fill_ea(x, s) for x in e]
    if isinstance(e, dict):
        return {k: _fill_ea(v, s) for k, v in e.items()}
    return e


def _renumber_runs(e, mp):
    if isinstance(e, list):
        if len(e) == 3 and e[0] == "run" and isinstance(e[1], int):
            return ["run", mp[e[1]], e[2]]
        return [_renumber_runs(s, mp) for s in e]
    if isinstance(e, dict):
        return {k: _renumber_runs(v, mp) for k, v in e.items()}
    return e


def ea_op_key(key, p):
    """Operation part of a key whose addressing byte is at p."""
    kb = bytearray(key[:p + 1])
    rp = rex_pos(key)
    if rp is not None and rp < p:
        kb[rp] &= REX_OP
    kb[p] &= MODRM_OP
    return bytes(kb)


def ea_addr_key(key, p, sib):
    rp = rex_pos(key)
    rex = key[rp] & REX_ADDR if rp is not None and rp < p else 0
    return bytes([rex, key[p] & MODRM_ADDR]) + bytes(key[p + 1:p + 1 + sib])


def ea_locate(maps, key, s):
    """(p, sib, address) such that the literal key's last byte(s) reproduce the address s through the learned code maps,
    or None. p = position of the addressing byte; the key ends with it (no SIB) or with the byte after it (SIB). A key
    whose last two bytes have the SIB shape (rm 100, mod != 11) is located as SIB or not at all; under mod 00, SIB base
    101 means no base, and a lone scale-1 index the learner wrote as a base is rewritten as that index (same value)."""
    reg, sc = maps.get("reg", {}), maps.get("scale", {})
    rp = rex_pos(key)
    rex = key[rp] if rp is not None else 0
    B, X = rex & 1, (rex >> 1) & 1
    first = opcode_pos(key)
    if first < len(key) and key[first] == 0x0F:  # supplied: 0F escapes a two-byte opcode (0F 38 / 0F 3A: three)
        first += 2 if first + 1 < len(key) and key[first + 1] in (0x38, 0x3A) else 1
    _, b, i, scale, d = s
    k = len(key)
    if k - 2 > first and (key[k - 2] & 7) == 4 and (key[k - 2] >> 6) != 3:
        sb, mod = key[k - 1], key[k - 2] >> 6
        base = None if mod == 0 and (sb & 7) == 5 else reg.get((sb & 7) | (B << 3))
        idx = None if ((sb >> 3) & 7) == 4 and not X else reg.get(((sb >> 3) & 7) | (X << 3))
        isc = sc.get(sb >> 6) if idx is not None else 0
        if (b, i, scale if i is not None else 0) == (base, idx, isc) and (base is not None or b is None):
            return k - 2, 1, s
        if base is None and i is None and idx is not None and b == idx and isc == 0:
            return k - 2, 1, ["sum", None, idx, 0, d]
        return None
    if i is None and b is not None and k - 1 > first:
        if b == "next" or reg.get((key[k - 1] & 7) | (B << 3)) == b:
            return k - 1, 0, s
    return None


def ea_factor(literals, maps):
    """Operation and address pseudo-records from the literal records that use exactly one memory address. Each
    pseudo-record's rows is its own index (members of a merged record are then its pseudo-records)."""
    ops, addrs, skipped = [], [], defaultdict(int)
    for r in literals:
        eff = r["effect"]
        sums = {json.dumps(x) for x in _sums([eff["rip"], eff["writes"], eff["flags"]], [])}
        if len(sums) != 1:
            if sums:
                skipped["several addresses"] += 1
            continue
        s = json.loads(next(iter(sums)))
        key = bytes(r["key"])
        loc = ea_locate(maps, key, s)
        if loc is None:
            skipped["unlocated"] += 1
            continue
        p, sib, s = loc
        d = s[4]
        disp_j = d[1] if isinstance(d, list) and len(d) == 3 and d[0] == "run" else None
        if d is not None and disp_j != 0:
            skipped["displacement not the first run"] += 1  # a constant (absolute) one has no run to carry
            continue
        runs = list(eff["runs"])
        akey = ea_addr_key(key, p, sib)
        a_sum = ["sum", s[1], s[2], s[3], ["run", 0, d[2]] if disp_j is not None else None]
        addrs.append({"key": akey, "rows": np.array([len(addrs)]),
                      "effect": {"rip": ["next"], "writes": [[["mem", a_sum, 1], ["const", 0]]], "flags": {}, "trap": 0,
                                 "k": len(akey), "runs": runs[:1] if disp_j is not None else [], "L": None}})
        keep = [j for j in range(len(runs)) if j != disp_j]
        mp = {j: n for n, j in enumerate(keep)}
        try:
            oeff = _renumber_runs(_replace_sum({kk: eff[kk] for kk in ("rip", "writes", "flags")}, EA), mp)
        except KeyError:
            skipped["displacement read outside the address"] += 1
            addrs.pop()
            continue
        ops.append({"key": ea_op_key(key, p), "rows": np.array([len(ops)]),
                    "effect": dict(oeff, trap=eff.get("trap", 0), k=p + 1, runs=[runs[j] for j in keep], L=None)})
    return ops, addrs, dict(skipped)


def _key_matrix(pseudo):
    n = max((len(r["key"]) for r in pseudo), default=1)
    K = np.zeros((len(pseudo), n), np.uint8)
    lens = np.zeros(len(pseudo), np.int64)
    for i, r in enumerate(pseudo):
        kb = bytes(r["key"])
        K[i, :len(kb)] = np.frombuffer(kb, np.uint8)
        lens[i] = len(kb)
    return K, lens


def ea_allow_codes(records, pseudo, maps, own_only):
    """Open codes per field (see the module docstring). records are merged from pseudo (rows = pseudo indices)."""
    K, lens = _key_matrix(pseudo)
    for r in records:
        if not r["fields"]:
            continue
        k = len(r["key"])
        members = np.asarray(r["rows"]).ravel()
        al = {}
        for j, f in r["fields"].items():
            kind = r["slot_kinds"][j]
            own = set(field_code(K[members, :k], f).tolist())
            if own_only:
                al[j] = frozenset(own)
                continue
            need = max(f[0], f[3][0] if f[3] else 0) + 1
            sel = lens >= need
            for b in range(min(k, K.shape[1])):
                m = r["mask"][b]
                sel &= ((K[:, b] & m) == (r["key"][b] & m)) | (lens <= b)
            sel[members] = False
            other = set(field_code(K[sel], f).tolist()) if sel.any() else set()
            ncode = 1 << (f[2] + (f[3] is not None))
            codes = {c for c in range(ncode) if c in maps.get(kind, {})}
            al[j] = frozenset((codes - other) | own)
        r["allow"] = al


class EAUnit:
    """Merged operation and address families plus their open codes."""

    def __init__(self, literals, maps, log=print):
        self.maps = maps
        ops, addrs, skipped = ea_factor(literals, maps)
        self.ops, _ = merge_records(ops, maps=maps, log=log)
        self.addrs, _ = merge_records(addrs, maps=maps, log=log)
        ea_allow_codes(self.ops, ops, maps, own_only=False)
        ea_allow_codes(self.addrs, addrs, maps, own_only=True)
        n_byte = byte_reg_guard(self.ops, _key_matrix(ops)[0])
        # contradictory evidence: one canonical key whose pseudo-records carry different templates (e.g. `or cl, [m]`
        # fitted 8-bit on some shapes and 64-bit on others, each exact on its own rows) composes nothing
        self.contested = []
        for fam in (ops, addrs):
            sigs = defaultdict(set)
            for r in fam:
                sigs[bytes(r["key"])].add(r["sig"])
            self.contested.append({kb for kb, v in sigs.items() if len(v) > 1})
        self.stats = {"op_pseudo": len(ops), "addr_pseudo": len(addrs), "skipped": skipped, "op_records": len(self.ops),
                      "addr_records": len(self.addrs), "byte_guard_closed": n_byte,
                      "contested_op_keys": len(self.contested[0]), "contested_addr_keys": len(self.contested[1])}
        log(f"[ea] factored {len(ops)} literal address records (skipped {skipped}): {len(self.ops)} operation records, "
            f"{len(self.addrs)} address records; byte-register guard closed {n_byte} operation codes")
        self._index()

    def _index(self):
        self.op_by, self.addr_by = defaultdict(list), defaultdict(list)
        for i, r in enumerate(self.ops):
            self.op_by[len(r["key"])].append(i)
        for i, r in enumerate(self.addrs):
            self.addr_by[len(r["key"])].append(i)
        self.op_spec = [sum(bin(x).count("1") for x in r["mask"]) for r in self.ops]
        self.addr_spec = [sum(bin(x).count("1") for x in r["mask"]) for r in self.addrs]

    def _best(self, records, idxs, spec, kb):
        best = None
        for i in idxs:
            r = records[i]
            k = len(r["key"])
            if not all((kb[j] & r["mask"][j]) == (r["key"][j] & r["mask"][j]) for j in range(k)):
                continue
            if r["fields"]:
                al = r.get("allow")
                if al is None:
                    continue
                keyarr = np.frombuffer(kb[:k], np.uint8)[None, :]
                if any(int(field_code(keyarr, f)[0]) not in al.get(j, ()) for j, f in r["fields"].items()):
                    continue
            sig = json.dumps(_norm_preds(r["tmpl"]), sort_keys=True)
            if best is None or spec[i] > best[0]:
                best = (spec[i], {sig}, i)
            elif spec[i] == best[0]:
                best[1].add(sig)
        if best is None or len(best[1]) > 1:  # equally specific records that disagree: not evidenced which
            return None
        return best[2]

    def _bind(self, r, kb):
        keyarr = np.frombuffer(kb[:len(r["key"])], np.uint8)[None, :]
        vals = []
        for j, kind in enumerate(r["slot_kinds"]):
            if r["consts"][j] is not None:
                vals.append(r["consts"][j])
                continue
            v = self.maps.get(kind, {}).get(int(field_code(keyarr, r["fields"][j])[0]))
            if v is None:
                return None
            vals.append(v)
        return bind(r["tmpl"], vals)

    def compose(self, window):
        """(["ea", op record, address record], effect) or (None, None)."""
        wb = bytes(window)
        first = opcode_pos(wb)
        if wb[first] == 0x0F:
            first += 2 if wb[first + 1] in (0x38, 0x3A) else 1
        found = []
        for p in range(first + 1, min(first + 4, len(wb) - 1)):
            okb = ea_op_key(wb, p)
            if okb in self.contested[0]:
                continue
            oi = self._best(self.ops, self.op_by.get(p + 1, []), self.op_spec, okb)
            if oi is None:
                continue
            for sib in (0, 1):
                akb = ea_addr_key(wb, p, sib)
                if akb in self.contested[1]:
                    continue
                ai = self._best(self.addrs, self.addr_by.get(len(akb), []), self.addr_spec, akb)
                if ai is not None:
                    found.append((p, sib, oi, ai, okb, akb))
        if len(found) != 1:  # none, or two readings of the same bytes: not evidenced which one
            return None, None
        p, sib, oi, ai, okb, akb = found[0]
        op, ad = self._bind(self.ops[oi], okb), self._bind(self.addrs[ai], akb)
        if op is None or ad is None:
            return None, None
        a_sum = ad["writes"][0][0][1]
        d_runs = list(ad["runs"])
        mp = {j: j + len(d_runs) for j in range(len(op["runs"]))}
        body = _fill_ea(_renumber_runs({kk: op[kk] for kk in ("rip", "writes", "flags")}, mp), a_sum)
        k = p + 1 + sib
        runs = d_runs + list(op["runs"])
        return ["ea", oi, ai], dict(body, trap=op.get("trap", 0), k=k, runs=runs, L=k + sum(runs))


class EADecoder:
    """A base decoder plus the EA unit: the base record wins; composition only where the base has no record."""

    def __init__(self, base, unit):
        self.base, self.unit = base, unit
        self.ea_cache = {}

    def __getattr__(self, name):
        if name in ("base", "unit", "ea_cache") or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def lookup(self, window):
        ri, eff = self.base.lookup(window)
        if ri is not None:
            return ri, eff
        wb = bytes(window[:15])
        hit = self.ea_cache.get(wb)
        if hit is None:
            hit = self.ea_cache[wb] = self.unit.compose(wb)
        return hit
