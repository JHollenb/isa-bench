"""M4 program execution: the learned machine runs a freestanding ELF from its native initial memory image, step by
step, and is compared with the native ptrace trace (RIP, 16 GPRs, six status flags) at every step.

Supplied around the learned machine (reported): the initial memory image (every readable mapping of the native
process at its first instruction, from native_asm/trace_full.c), the page store (reads/writes of the bytes the
machine addresses), the fetch of 15 window bytes at the RIP the machine emits, and the Linux OS boundary at a learned
trap record (write(1, buf, n) and exit; RAX = result, RCX = return RIP, R11 = RFLAGS as the kernel leaves them).
The machine is never corrected from the native trace: the run continues on its own state after a divergence, and stops
on abstention, an unmapped access, or exit.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from . import m4_learn as L
from .m4_isa import FLAG_BITS, FLAGS, GPRS, MASK64

U64 = np.uint64
STATUS_MASK = sum(1 << b for b in FLAG_BITS)


def load_image(path):
    pages = {}
    perms = {}
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


class ProgMem:
    """Page store of one running program (n = 1 rows), API-compatible with m4_isa.PageStore.read."""

    def __init__(self, pages):
        self.pages = pages
        self.fault = None

    def byte(self, a):
        pg = self.pages.get(a & ~0xFFF)
        if pg is None:
            if self.fault is None:
                self.fault = a
            return 0
        return pg[a & 0xFFF]

    def read(self, addr, nbytes):
        addr = np.asarray(addr, U64)
        flat = addr.reshape(-1).tolist()
        out = []
        for a in flat:
            v = 0
            for j in range(nbytes):
                v |= self.byte((a + j) & MASK64) << (8 * j)
            out.append(v)
        return np.array(out, U64).reshape(addr.shape)

    def write(self, a, nbytes, value):
        for j in range(nbytes):
            aa = (a + j) & MASK64
            pg = self.pages.get(aa & ~0xFFF)
            if pg is None:
                if self.fault is None:
                    self.fault = aa
                continue
            pg[aa & 0xFFF] = (value >> (8 * j)) & 0xFF

    def take(self, rows):
        return self

    def read_int(self, a, nbytes):
        v = 0
        for j in range(nbytes):
            v |= self.byte((a + j) & MASK64) << (8 * j)
        return v


class StepCtx(L.Ctx):
    """One-row context built from the running state (no observed after-state)."""

    def __init__(self, m, state, window, mem, k, runs):
        self.m = m
        self.rows = np.zeros(1, np.int64)
        self.g = {"window": np.frombuffer(window, np.uint8)[None, :], "regs": np.array([state["regs"]], U64),
                  "rip": np.array([state["rip"]], U64), "flags": np.array([state["flags"]], np.uint8)}
        self.n = 1
        self.k, self.runs = k, tuple(runs)
        self.L = k + sum(runs)
        self.store = mem
        win = self.g["window"].astype(U64)
        self.run_vals = []
        pos = k
        for size in runs:
            if size == 0:
                self.run_vals.append(None)
                continue
            v = np.zeros(1, U64)
            for j in range(size):
                v |= win[:, pos + j] << U64(8 * j)
            self.run_vals.append((v, size))
            pos += size
        self.cf = self.g["flags"][:, L.CF_I].astype(np.int64)
        self._next = None
        self._addr = None
        self._loads = {}


def eflags_to_flags(e):
    return [(int(e) >> b) & 1 for b in FLAG_BITS]


def run_program(m, dec, mem_path, trace, max_steps=None, log=None, oracle=None, scalar=None):
    """Execute the learned machine from the native initial state; compare with the native trace step by step."""
    pages, _ = load_image(mem_path)
    mem = ProgMem(pages)
    steps = np.asarray(trace)
    first = steps[0]
    state = {"rip": int(first[0]), "regs": [int(v) for v in first[1:17]], "flags": eflags_to_flags(first[17])}
    rflags_rest = int(first[17]) & ~STATUS_MASK  # IF and reserved bits, which the machine does not model
    n_native = len(steps)
    max_steps = max_steps or n_native + 16
    stdout = bytearray()
    first_div, exact, status, exit_code = None, 0, "running", None
    exact_regs, first_div_regs, exact_or, first_div_or = 0, None, 0, None
    exact_regs_tf = 0
    n_or = len(oracle) if oracle is not None else 0
    abstentions = 0
    div_detail = div_regs_detail = div_or_detail = None
    flag_only_div = 0
    i = 0
    for i in range(max_steps):
        window = bytes(mem.byte((state["rip"] + j) & MASK64) for j in range(15))
        ri, eff = dec.lookup(window)
        if eff is None:
            status, abstentions = "abstain", abstentions + 1
            break
        if scalar is not None:
            new, writes, trap, abst = scalar.apply(eff, state, window, mem)
            if abst:
                status, abstentions = "abstain_predicate", abstentions + 1
                break
            for addr, nb, v in writes:
                mem.write(addr, nb, v)
        else:
            x = StepCtx(m, state, window, mem, eff["k"], eff["runs"])
            regs, rip, flags, writes, trap, abst = L.apply_effect(m, eff, x)
            if abst[0]:
                status, abstentions = "abstain_predicate", abstentions + 1
                break
            for addr, nb, v in writes:
                mem.write(int(addr[0]), nb, int(v[0]))
            new = {"rip": int(rip[0]), "regs": [int(v) for v in regs[0]], "flags": [int(v) for v in flags[0]]}
        if trap:
            new = {"rip": new["rip"], "regs": list(state["regs"]), "flags": list(state["flags"])}
            st, code = os_boundary(new, mem, stdout, rflags_rest)
            if st != "running":
                status, exit_code = st, code
                i += 1
                break
        if mem.fault is not None:
            status = "unmapped_access"
            break
        state = new
        ref = steps[i + 1] if i + 1 < n_native else None
        oref = oracle[i + 1] if i + 1 < n_or else None
        if oref is not None:
            if [int(v) for v in oref[1:17]] == state["regs"] and int(oref[0]) == state["rip"] and eflags_to_flags(oref[17]) == state["flags"]:
                exact_or += 1
            elif first_div_or is None:
                first_div_or = i
                div_or_detail = {"step": i, "rip": hex(int(oracle[i][0])), "window": window.hex(), "record": ri}
        if ref is not None:
            if [int(v) for v in ref[1:17]] == state["regs"] and int(ref[0]) == state["rip"]:
                exact_regs += 1
            # the ptrace single-step tracer runs the program with TF set; SYSCALL saves RFLAGS (with TF) into R11
            if [int(v) for v in ref[1:12]] + [int(ref[12]) & ~0x100] + [int(v) for v in ref[13:17]] == \
                    state["regs"][:11] + [state["regs"][11] & ~0x100] + state["regs"][12:] and int(ref[0]) == state["rip"]:
                exact_regs_tf += 1
            elif first_div_regs is None:
                first_div_regs = i
                div_regs_detail = {"step": i, "rip": hex(int(steps[i][0])), "window": window.hex(), "record": ri,
                                   "effect": json.dumps(eff)[:600],
                                   "regs_diff": {GPRS[r]: [hex(state["regs"][r]), hex(int(ref[1 + r]))] for r in range(16) if state["regs"][r] != int(ref[1 + r])},
                                   "rip_diff": [hex(state["rip"]), hex(int(ref[0]))] if state["rip"] != int(ref[0]) else None}
            ok_regs = [int(v) for v in ref[1:17]] == state["regs"] and int(ref[0]) == state["rip"]
            ok_flags = eflags_to_flags(ref[17]) == state["flags"]
            if ok_regs and ok_flags:
                exact += 1
            else:
                if ok_regs:
                    flag_only_div += 1
                if first_div is None:
                    first_div = i
                    div_detail = {"step": i, "rip": hex(int(steps[i][0])), "window": window.hex(), "record": ri,
                                  "effect": json.dumps(eff)[:600],
                                  "regs_diff": {GPRS[r]: [hex(state["regs"][r]), hex(int(ref[1 + r]))] for r in range(16) if state["regs"][r] != int(ref[1 + r])},
                                  "rip_diff": [hex(state["rip"]), hex(int(ref[0]))] if state["rip"] != int(ref[0]) else None,
                                  "flags_diff": {FLAGS[k]: [state["flags"][k], eflags_to_flags(ref[17])[k]] for k in range(6) if state["flags"][k] != eflags_to_flags(ref[17])[k]}}
    return {"status": status, "exit_code": exit_code, "steps_run": i + (0 if status == "running" else 0), "native_steps": n_native,
            "exact_steps": exact, "first_divergence": first_div, "flag_only_divergent_steps": flag_only_div,
            "exact_steps_regs_rip": exact_regs, "first_divergence_regs_rip": first_div_regs,
            "exact_steps_regs_rip_r11_tf_masked": exact_regs_tf,
            "exact_steps_vs_oracle": exact_or, "first_divergence_vs_oracle": first_div_or, "oracle_steps": n_or,
            "abstentions": abstentions, "stdout": stdout.decode(errors="replace"), "divergence": div_detail,
            "divergence_regs_rip": div_regs_detail, "divergence_vs_oracle": div_or_detail,
            "abstain_at": {"step": i, "rip": hex(state["rip"]), "window": window.hex()} if status.startswith("abstain") else None}


def os_boundary(state, mem, stdout, rflags_rest):
    """Linux write/exit at a trap record. Kernel effects: RAX = result, RCX = return RIP, R11 = RFLAGS."""
    rax, rdi, rsi, rdx = state["regs"][0], state["regs"][7], state["regs"][6], state["regs"][2]
    rflags = rflags_rest | sum(f << b for f, b in zip(state["flags"], FLAG_BITS))
    if rax in (60, 231):
        return "exit", rdi & 0xFF
    if rax == 1 and rdi in (1, 2) and rdx < 1 << 20:
        data = mem.read(np.array([(rsi + j) & MASK64 for j in range(rdx)], U64), 1).astype(np.uint8).tobytes()
        if rdi == 1:
            stdout.extend(data)
        state["regs"][0] = rdx
        state["regs"][1] = state["rip"]
        state["regs"][11] = rflags
        return "running", None
    return "unsupported_syscall", None


def oracle_program(mem_path, trace, max_steps=None, record=False):
    """The label source on the same program: Unicorn single steps from the same image, compared with native."""
    import unicorn as U
    from unicorn import x86_const as X
    pages, perms = load_image(mem_path)
    uc = U.Uc(U.UC_ARCH_X86, U.UC_MODE_64)
    for p, data in pages.items():
        uc.mem_map(p, 0x1000)
        uc.mem_write(p, bytes(data))
    ids = [getattr(X, f"UC_X86_REG_{n}") for n in GPRS]
    steps = np.asarray(trace)
    first = steps[0]
    for rid, v in zip(ids, first[1:17]):
        uc.reg_write(rid, int(v))
    uc.reg_write(X.UC_X86_REG_EFLAGS, int(first[17]) & ~0x100)
    rip = int(first[0])
    trap = []
    uc.hook_add(U.UC_HOOK_INSN, lambda u, d: (trap.append(1), u.emu_stop()), None, 1, 0, X.UC_X86_INS_SYSCALL)
    stdout = bytearray()
    exact, first_div, flag_only = 0, None, 0
    detail, flag_kinds = None, {}
    traj = [[int(v) for v in first]]
    n = len(steps)
    status, code = "running", None
    max_steps = max_steps or n + 16
    i = 0
    for i in range(max_steps):
        trap.clear()
        rip0 = rip
        try:
            uc.emu_start(rip, (rip + 0x10000) & MASK64, count=1)
        except U.UcError as e:
            status = f"error:{e}"
            break
        rip = uc.reg_read(X.UC_X86_REG_RIP)
        if trap:
            rip = rip0 + 2
            rax, rdi, rsi, rdx = (uc.reg_read(getattr(X, f"UC_X86_REG_{r}")) for r in ("RAX", "RDI", "RSI", "RDX"))
            if rax in (60, 231):
                status, code = "exit", rdi & 0xFF
                i += 1
                break
            if rax != 1:
                status = "unsupported_syscall"
                break
            if rdi == 1:
                stdout.extend(bytes(uc.mem_read(rsi, rdx)))
            uc.reg_write(X.UC_X86_REG_RAX, rdx)
            uc.reg_write(X.UC_X86_REG_RCX, rip)
            uc.reg_write(X.UC_X86_REG_R11, uc.reg_read(X.UC_X86_REG_EFLAGS))
            uc.reg_write(X.UC_X86_REG_RIP, rip)
        regs = [uc.reg_read(r) for r in ids]
        ef = uc.reg_read(X.UC_X86_REG_EFLAGS)
        fl = eflags_to_flags(ef)
        if record:
            traj.append([rip] + regs + [ef])
        ref = steps[i + 1] if i + 1 < n else None
        if ref is not None:
            okr = [int(v) for v in ref[1:17]] == regs and int(ref[0]) == rip
            okf = eflags_to_flags(ref[17]) == fl
            if okr and okf:
                exact += 1
            else:
                flag_only += okr
                if first_div is None:
                    first_div = i
                    detail = {"step": i, "rip": hex(int(steps[i][0])), "code": bytes(uc.mem_read(int(steps[i][0]), 15)).hex(),
                              "flags": [fl, eflags_to_flags(ref[17])], "regs_ok": okr}
                if okr and not okf:
                    key = (bytes(uc.mem_read(int(steps[i][0]), 4)).hex()[:4],
                           tuple(FLAGS[k] for k in range(6) if fl[k] != eflags_to_flags(ref[17])[k]))
                    flag_kinds[key] = flag_kinds.get(key, 0) + 1
    return {"status": status, "exit_code": code, "exact_steps": exact, "native_steps": n, "first_divergence": first_div,
            "flag_only_divergent_steps": flag_only, "stdout": stdout.decode(errors="replace"), "first_detail": detail,
            "flag_divergence_kinds": {f"{k[0]}:{'/'.join(k[1])}": v for k, v in sorted(flag_kinds.items(), key=lambda kv: -kv[1])[:12]},
            **({"trajectory": np.array(traj, dtype=np.uint64)} if record else {})}


# ------------------------------------------------------------------ the lease


def milestone(form):
    """Reporting class of an oracle form name (never seen by the learner)."""
    if form.startswith(("jcc", "jmp", "call", "ret")):
        return "M4 branches"
    if form.startswith(("push", "pop", "leave", "lea")) or form.endswith(("m", "m_r", "m_i", "m_i8", "m_i32", "m_1", "m_cl")) or "rm8m" in form \
            or "rm16m" in form or "rm32m" in form or "rm64m" in form or form.startswith("nop_rm"):
        return "M3 memory/stack"
    return "M2 registers/ALU"


def verify_spf(m, rows=4096, seed=7):
    """Every learned table: byte-chunk executor vs SPF monomial_scan (spf_datapath.SPFExecutor), bit for bit."""
    from . import spf_datapath as D
    rng = np.random.default_rng(seed)
    out = []
    for t, (tab, lut) in enumerate(zip(m.tables, m.luts)):
        ex = D.SPFExecutor(tab)
        res = {}
        for w in (8, 32, 64):
            a = rng.integers(0, 2**63, rows, dtype=np.uint64) * U64(2) + rng.integers(0, 2, rows, dtype=np.uint64)
            b = rng.integers(0, 2**63, rows, dtype=np.uint64) * U64(2) + rng.integers(0, 2, rows, dtype=np.uint64)
            a &= U64(L.MASKS[w])
            b &= U64(L.MASKS[w])
            cf = rng.integers(0, 2, rows)
            bits, before, final = ex.run(D.bits(a, w), D.bits(b, w), cf)
            spf = D.from_bits(bits)
            lres = lut.run(a, b, cf, w)
            res[str(w)] = bool(np.array_equal(spf, lres))
        out.append({"table": t, "states": tab.n, "bit_identical": res, "spf_counters": dict(ex.counters)})
    return out


def score_split(m, dec, data, name, log):
    ok, ab = L.score(m, dec, data)
    forms = data["o_form"]
    per = {}
    for f in sorted(set(forms.tolist())):
        sel = forms == f
        per[f] = {"n": int(sel.sum()), "exact": float(ok[sel].mean()), "abstain": float(ab[sel].mean())}
    by_ms = {}
    for f, v in per.items():
        ms = milestone(f)
        d = by_ms.setdefault(ms, {"n": 0, "exact": 0, "abstain": 0})
        d["n"] += v["n"]
        d["exact"] += v["exact"] * v["n"]
        d["abstain"] += v["abstain"] * v["n"]
    for d in by_ms.values():
        d["exact"] = d["exact"] / max(d["n"], 1)
        d["abstain"] = d["abstain"] / max(d["n"], 1)
    res = {"n": int(len(ok)), "exact": float(ok.mean()), "abstain": float(ab.mean()), "wrong": float((~ok & ~ab).mean()),
           "by_milestone": by_ms, "per_form": per}
    log(f"[m4] {name}: n={res['n']} exact={res['exact']:.4f} abstain={res['abstain']:.4f} wrong={res['wrong']:.4f}")
    return res


class RowMem:
    def __init__(self, store, i):
        self.s = store.take([i])

    def read_int(self, a, nb):
        return int(self.s.read(np.array([a], U64), nb)[0])


def check_agreement(m, dec, data, n=3000, seed=5):
    """Scalar interpreter (program runs) vs vectorized evaluator (panels) on the same rows: predicted state equal?"""
    from .m4_scalar import Scalar
    sc = Scalar(m)
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(data["rip"]), min(n, len(data["rip"])), replace=False)
    agree = total = 0
    bad = []
    for i in rows.tolist():
        _, eff = dec.lookup(data["window"][i])
        if eff is None:
            continue
        x = L.Ctx(m, data, np.array([i]), eff["k"], eff["runs"])
        regs, rip, flags, writes, trap, abst = L.apply_effect(m, eff, x)
        state = {"rip": int(data["rip"][i]), "regs": [int(v) for v in data["regs"][i]], "flags": [int(v) for v in data["flags"][i]]}
        new, sw, strap, sab = sc.apply(eff, state, bytes(data["window"][i]), RowMem(data["store"], i))
        same = (new["rip"] == int(rip[0]) and new["regs"] == [int(v) for v in regs[0]] and new["flags"] == [int(v) for v in flags[0]]
                and [(a, nb, v) for a, nb, v in sw] == [(int(a[0]), nb, int(v[0])) for a, nb, v in writes] and bool(sab) == bool(abst[0]))
        total += 1
        agree += same
        if not same and len(bad) < 5:
            bad.append({"row": i, "effect": json.dumps(eff)[:300]})
    return {"rows": total, "agree": agree, "examples": bad}


_PROG = {}


def _program_job(pdir):
    import time
    from .m4_scalar import Scalar
    pdir = Path(pdir)
    m, dec = _PROG["m"], _PROG["dec"]
    tr = np.load(pdir / "trace.npz")["steps"]
    orc = oracle_program(pdir / "mem.bin", tr, record=True)
    traj = orc.pop("trajectory")
    t = time.time()
    r = run_program(m, dec, pdir / "mem.bin", tr, oracle=traj, scalar=Scalar(m))
    r["wall_s"] = round(time.time() - t, 1)
    native = json.loads((pdir / "native.json").read_text()) if (pdir / "native.json").exists() else {}
    return pdir.name, {"learned": r, "unicorn": orc, "native": native}


def worker():
    import os
    import pickle
    import time
    from . import m4_isa as I
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=False)
    logf = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    grid = I.make_grid()
    W = cfg["workers"]
    train = I.generate(cfg["keys_per_form"], cfg["seed"], "train", grid, workers=W)
    log(f"[m4] train rows {len(train['rip'])} t={time.time() - t0:.0f}s")
    m, dec, info, pending = L.learn(train, log=log, workers=W)
    info["train_rows"] = int(len(train["rip"]))
    info["pending_by_form"] = dict(sorted(((str(k), int(v)) for k, v in zip(*np.unique(train["o_form"][pending], return_counts=True))),
                                          key=lambda kv: -kv[1])[:60])
    ok_tr, ab_tr = L.score(m, dec, train)
    info["train_exact"] = float(ok_tr.mean())
    info["train_abstain"] = float(ab_tr.mean())
    log(f"[m4] train exact {info['train_exact']:.4f} abstain {info['train_abstain']:.4f} t={time.time() - t0:.0f}s")
    info["spf"] = verify_spf(m)
    res = {"config": cfg, "menu": L.MENU, "learn": info, "grid": grid}
    for split, seed in (("iid", cfg["seed"] * 10 + 1), ("cells", cfg["seed"] * 10 + 2), ("literal", cfg["seed"] * 10 + 3)):
        d = I.generate(cfg["test_keys_per_form"], seed, split, grid, workers=W)
        d["store"] = L.learner_store(d)
        res[split] = score_split(m, dec, d, split, log)
        if split == "iid":
            res["scalar_vs_vector"] = check_agreement(m, dec, d)
            log(f"[m4] scalar vs vector agreement {res['scalar_vs_vector']['agree']}/{res['scalar_vs_vector']['rows']}")
    with open(out / "machine.pkl", "wb") as f:
        pickle.dump({"tables": m.tables, "adder": m.adder, "adder_ok": m.adder_ok, "records": m.records, "maps": m.code_map,
                     "known_taps": getattr(m, "known_taps", {})}, f)
    with open(out / "machine_full.pkl", "wb") as f:
        pickle.dump({"m": m, "dec": dec}, f)
    # salvage point: panels + learned machine are published before the program phase
    res["programs"] = None
    res["phase"] = "panels"
    res["wall_s_panels"] = round(time.time() - t0, 1)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    from isa_lab.full_custody import publish
    publish([str(out / f) for f in ("m4.json", "machine.pkl", "machine_full.pkl", "log.txt")], "m4-panels")
    res["programs"] = run_programs(m, dec, Path(cfg["native_dir"]), W, log, out=out)
    res["phase"] = "done"
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    from isa_lab.full_custody import pack
    pack(out)


def run_programs(m, dec, base, W, log, out=None):
    progs = {}
    _PROG.update(m=m, dec=dec)
    import multiprocessing as mp
    pdirs = sorted(p for p in base.glob("*") if p.is_dir())
    import gc
    gc.collect()
    gc.freeze()  # forked program workers must not copy the machine by touching it in their own GC passes
    with mp.get_context("fork").Pool(min(W, len(pdirs))) as pool:
        for name, entry in pool.imap_unordered(_program_job, [str(p) for p in pdirs]):
            progs[name] = entry
            r, native = entry["learned"], entry["native"]
            log(f"[m4] program {name}: {r['status']} exit={r['exit_code']} exact {r['exact_steps']}/{r['native_steps']} "
                f"regs+rip {r['exact_steps_regs_rip']} vs-unicorn {r['exact_steps_vs_oracle']}/{r['oracle_steps']} first-div {r['first_divergence']} "
                f"stdout ok {r['stdout'] == native.get('native_stdout')} wall {r['wall_s']}s")
            if out is not None:  # each finished program is published as it lands
                (out / "m4_programs.json").write_text(json.dumps(progs, indent=1, sort_keys=True, default=str))
                from isa_lab.full_custody import publish
                publish([str(out / "m4_programs.json"), str(out / "log.txt")], f"m4-program-{name}")
    return progs


def worker_programs():
    """Run a frozen machine (machine_full.pkl from a learn lease) on staged native programs; no learning here."""
    import pickle
    import time
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=False)
    logf = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    with open(cfg["machine"], "rb") as f:
        mm = pickle.load(f)
    res = {"config": cfg, "programs": run_programs(mm["m"], mm["dec"], Path(cfg["native_dir"]), cfg["workers"], log, out=out)}
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    from isa_lab.full_custody import pack
    pack(out)
