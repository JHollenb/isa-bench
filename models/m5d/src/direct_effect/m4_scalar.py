"""Scalar (one-state, Python-int) interpreter of M4 effects for program runs: the same learned tables (byte-chunk
compositions of the lane A automata), maps and taps as the vectorized `m4_learn.ev`; agreement between the two paths
is checked on transition rows by the lease (`check_agreement`)."""
from __future__ import annotations

import numpy as np

from .m4_isa import FLAG_BITS, MASK64

MASKS = {8: 0xFF, 16: 0xFFFF, 32: 0xFFFFFFFF, 64: MASK64}


def sx(v, bits):
    v &= (1 << bits) - 1
    return v - (1 << bits) if v >> (bits - 1) else v


class ScalarTables:
    def __init__(self, m):
        self.t = []
        for tab, lut in zip(m.tables, m.luts):
            self.t.append((lut.out.tobytes(), lut.nxt.tobytes(), lut.bef.tobytes(), [int(v) for v in lut.start], tab.reverse))

    def run(self, t, a, b, cf, w, sig=False):
        out, nxt, bef, start, rev = self.t[t]
        q = start[cf]
        res = 0
        nb = w // 8
        befs = [0] * nb
        for c in (range(nb - 1, -1, -1) if rev else range(nb)):
            i = (q << 16) | (((a >> (8 * c)) & 0xFF) << 8) | ((b >> (8 * c)) & 0xFF)
            res |= out[i] << (8 * c)
            befs[c] = bef[i]
            q = nxt[i]
        if not sig:
            return res
        top = w - 1
        return res, {"r_top": (res >> top) & 1, "a_top": (a >> top) & 1, "b_top": (b >> top) & 1,
                     "q_top": (befs[top // 8] >> (top % 8)) & 1, "q_final": q, "nz": int(res != 0), "par8": bin(res & 0xFF).count("1") & 1,
                     "a4": (a >> 4) & 1, "b4": (b >> 4) & 1, "r4": (res >> 4) & 1}


def shift_s(a, c, kind, w):
    d, fill = kind
    a &= MASKS[w]
    top = (a >> (w - 1)) & 1
    out = 0
    for i in range(w):
        src = i - c if d == "l" else i + c
        if fill == "w":
            bit = (a >> (src % w)) & 1
        elif 0 <= src < w:
            bit = (a >> src) & 1
        else:
            bit = top if (fill == "t" and d == "r" and src >= w) else 0
        out |= bit << i
    if fill == "w":
        spill = (out & 1) if d == "l" else (out >> (w - 1)) & 1
    else:
        pos = w - c if d == "l" else c - 1
        spill = (a >> pos) & 1 if 0 <= pos < w else 0
    return out & MASKS[w], spill


def mul_s(a, b, w, signed):
    a &= MASKS[w]
    b &= MASKS[w]
    if signed:
        p = sx(a, w) * sx(b, w)
    else:
        p = a * b
    return p & MASKS[w], (p >> w) & MASKS[w]


def div_s(hi, lo, b, w, signed):
    n = (hi << w) | lo
    d = b
    if signed:
        n = n - (1 << 2 * w) if n >> (2 * w - 1) else n
        d = sx(d, w)
    if d == 0:
        return 0, 0
    q = abs(n) // abs(d)
    if (n < 0) != (d < 0):
        q = -q
    r = n - q * d
    return q & MASKS[w], r & MASKS[w]


def _pidx(combo, flags):
    idx = 0
    for f in combo:
        idx = idx * 2 + flags[f]
    return idx


def pred_s(p, flags):
    """(value, abstain): abstain where the entry is unobserved or (M5) a consistent alternative is unobserved or differs."""
    idx = _pidx(p[1], flags)
    v = p[2][idx]
    ab = len(p) > 3 and idx in p[3]
    if not ab and len(p) > 4:
        for combo, table, unobs in p[4]:
            j = _pidx(combo, flags)
            if table[j] != v or j in unobs:
                ab = True
                break
    return v, ab


class Scalar:
    def __init__(self, m):
        self.m = m
        self._alt_cache = {}
        self._alt_keep = {}
        self.T = ScalarTables(m)
        self.adder = m.adder

    def add(self, a, b):
        return self.T.run(self.adder, a & MASK64, b & MASK64, 0, 64)

    def addr(self, a, st):
        _, base, index, scale, disp = a
        terms = []
        if base == "next":
            terms.append(st["next"])
        elif base is not None:
            terms.append(st["regs"][base])
        if index is not None:
            terms.append((st["regs"][index] << scale) & MASK64)
        if disp is not None:
            terms.append(self.ev(disp, st, None))
        if not terms:
            return 0
        s = terms[0]
        for t in terms[1:]:
            s = self.add(s, t)
        return s

    def ev(self, e, st, old, sig=False):
        op = e[0]
        if op == "reg":
            v = st["regs"][e[1]] & MASKS[e[2]]
            return sx(v, e[2]) & MASK64 if e[3] == "s" else v
        if op == "run":
            v, size = st["runs"][e[1]]
            return sx(v, 8 * size) & MASK64 if e[2] == "s" else v
        if op == "cmask":  # M5c count mask (mirror of m4_learn.ev)
            return self.ev(e[1], st, old) & e[2]
        if op == "const":
            return e[1] & MASK64
        if op == "fill":
            return MASK64 if (st["regs"][e[1]] >> (e[2] - 1)) & 1 else 0
        if op == "next":
            return st["next"]
        if op == "old":
            return old
        if op == "addr":
            return self.addr(e[1], st)
        if op == "load":
            v = st["mem"].read_int(self.addr(e[1], st), e[2])
            return sx(v, 8 * e[2]) & MASK64 if e[3] == "s" else v
        if op == "table":
            _, t, A, B, w = e
            a = self.ev(A, st, old) & MASKS[w]
            b = self.ev(B, st, old) & MASKS[w]
            return self.T.run(t, a, b, st["flags"][0], w, sig)
        if op == "shift":
            _, kind, A, C, w = e
            a = self.ev(A, st, old) & MASKS[w]
            c = self.ev(C, st, old) & 0xFF
            res, spill = shift_s(a, c, kind, w)
            if not sig:
                return res
            top = w - 1
            return res, {"r_top": (res >> top) & 1, "a_top": (a >> top) & 1, "a_top2": (a >> (top - 1)) & 1, "nz": int(res != 0),
                         "par8": bin(res & 0xFF).count("1") & 1, "spill": spill, "c1": int(c == 1), "c0": int(c == 0), "r_bot": res & 1,
                         "r_top2": (res >> (top - 1)) & 1}
        if op == "mul":
            _, A, B, w, signed, part = e
            lo, hi = mul_s(self.ev(A, st, old), self.ev(B, st, old), w, signed)
            res = lo if part == "lo" else hi
            if not sig:
                return res
            top = w - 1
            fill = MASKS[w] if (lo >> top) & 1 else 0
            return res, {"r_top": (lo >> top) & 1, "nz": int(lo != 0), "par8": bin(lo & 0xFF).count("1") & 1,
                         "hi_nz": int(hi != 0), "hi_nfill": int(hi != fill)}
        if op == "div":
            _, HI, LO, B, w, signed, part = e
            q, r = div_s(self.ev(HI, st, old) & MASKS[w], self.ev(LO, st, old) & MASKS[w], self.ev(B, st, old) & MASKS[w], w, signed)
            res = q if part == "q" else r
            return (res, {}) if sig else res
        if op == "pred":
            v, unobs = pred_s(e, st["flags"])
            st["abstain"] |= unobs
            return v
        if op == "select":
            v, unobs = pred_s(e[1], st["flags"])
            st["abstain"] |= unobs
            return self.ev(e[2], st, old) if v else self.ev(e[3], st, old)
        raise ValueError(op)

    def apply(self, eff, state, window, mem):
        """state {'rip','regs','flags'} → (new state, writes [(addr, nbytes, value)], trap, abstain)."""
        k, runs = eff["k"], eff["runs"]
        rv = []
        pos = k
        for size in runs:
            v = 0
            for j in range(size):
                v |= window[pos + j] << (8 * j)
            rv.append((v, size))
            pos += size
        L = k + sum(runs)
        st = {"regs": state["regs"], "flags": state["flags"], "runs": rv, "mem": mem, "abstain": False,
              "next": self.add(state["rip"], L)}
        regs = list(state["regs"])
        rip = self.ev(eff["rip"], st, None)
        writes = []
        for tgt, e in eff["writes"]:
            if tgt[0] == "reg":
                _, r, w, merge = tgt
                v = self.ev(e, st, state["regs"][r] & MASKS[w]) & MASKS[w]
                regs[r] = v if merge in ("repl", "zero") else (state["regs"][r] & ~MASKS[w] & MASK64) | v
            else:
                _, a_e, nb = tgt
                addr = self.addr(a_e, st)
                old = mem.read_int(addr, nb)
                writes.append((addr, nb, self.ev(e, st, old) & MASKS[8 * nb]))
        flags = list(state["flags"])
        fl = eff.get("flags") or {}
        if fl.get("src") is not None:
            tgt = fl.get("src_old")
            old = None
            if tgt is not None and tgt[0] == "reg":
                old = state["regs"][tgt[1]] & MASKS[tgt[2]]
            elif tgt is not None and tgt[0] == "mem":
                old = mem.read_int(self.addr(tgt[1], st), tgt[2])
            _, sig = self.ev(fl["src"], st, old, sig=True)
        else:
            sig = {}
        def tidx(names, i):
            idx = 0
            for nm in names:
                idx = idx * 2 + ((state["flags"][i] if nm == "old" else sig[nm]) & 1)
            return idx
        for i, f in enumerate(("CF", "PF", "AF", "ZF", "SF", "OF")):
            tap = fl["taps"][f]
            idx = tidx(tap[0], i)
            flags[i] = tap[1][idx]
            # M5 version space (mirror of m4_learn.tap_abstain)
            if tap[0] and idx in ((fl.get("tap_unobs") or {}).get(f) or ()):
                st["abstain"] = True
            alts = (fl.get("tap_alts") or {}).get(f, [])
            if alts == "overflow":
                st["abstain"] = True
                continue
            if len(alts) > 16:  # M5c': large version spaces are checked once per (alternatives, signal vector, flag)
                self._alt_keep.setdefault(id(alts), alts)  # keeps the list alive, so its id is never reused
                ck = (id(alts), i, flags[i], state["flags"][i], tuple(sorted(sig.items())))
                hit = self._alt_cache.get(ck)
                if hit is None:
                    hit = any(alt[1][tidx(alt[0], i)] != flags[i] or (len(alt) > 2 and alt[0] and tidx(alt[0], i) in alt[2]) for alt in alts)
                    self._alt_cache[ck] = hit
                if hit:
                    st["abstain"] = True
                continue
            for alt in alts:
                j = tidx(alt[0], i)
                if alt[1][j] != flags[i] or (len(alt) > 2 and alt[0] and j in alt[2]):
                    st["abstain"] = True
        for alt_src, cands in fl.get("src_alts") or []:  # M5 flag-source alternatives (mirror of m4_learn)
            _, sig = self.ev(alt_src, st, old, sig=True)
            for i, f in enumerate(("CF", "PF", "AF", "ZF", "SF", "OF")):
                if cands[f] == "overflow":
                    st["abstain"] = True
                    continue
                for t in cands[f]:
                    j = tidx(t[0], i)
                    if t[1][j] != flags[i] or (len(t) > 2 and t[0] and j in t[2]):
                        st["abstain"] = True
        return {"rip": rip, "regs": regs, "flags": flags}, writes, eff.get("trap", 0), st["abstain"]
