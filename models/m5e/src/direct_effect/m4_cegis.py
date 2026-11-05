"""M4 counterexample-driven closed loop (CEGIS / active learning) for the structured learner.

The base machine is learned from random Unicorn transitions (m4_learn.learn). The dev programs then only CHOOSE which
encodings to study ("dev-program-guided form selection"):

  1. static: every distinct instruction in the dev programs' code (RIPs from the native trace, bytes from the initial
     image) is parsed oracle-side into (form, key) and validated on fresh random-state Unicorn rows for that key;
  2. dynamic: the programs are executed by the learned machine; an abstention, or the first step where the machine
     leaves the Unicorn trajectory, names the instruction at that RIP;
  3. drain: forms with unfitted training rows.

For every chosen key, fresh random-state Unicorn rows (the exact key plus a few random keys of the same form) are added
to the training data and fitted incrementally (the trie over the new rows only; tables, adder and maps are kept); the
literal records are then re-merged. Labels never come from the programs' trace states. Held-out cells and literal
bytes stay excluded exactly as in the train split.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import m4_isa as I
from . import m4_learn as L
from . import m4_run as R

_P = {}


def append_rows(data, new):
    """Append packed rows to the training data in place; returns the new row indices."""
    n0 = len(data["rip"])
    store = L.learner_store(new)
    for k, v in new.items():
        if k in ("overrides", "store"):
            continue
        data[k] = np.concatenate([data[k], v])
    data["store"] = I.PageStore.concat(data["store"], store)
    return np.arange(n0, n0 + len(new["rip"]))


def program_keys(base):
    """Distinct instructions of every dev program: {(form, key hex): info}, plus per-program counts."""
    keys, per = {}, {}
    for pdir in sorted(p for p in Path(base).glob("*") if p.is_dir()):
        tr = np.load(pdir / "trace.npz")["steps"]
        pages, _ = R.load_image(pdir / "mem.bin")
        mem = R.ProgMem(pages)
        n = 0
        for rip in np.unique(tr[:, 0]).tolist():
            w = bytes(mem.byte((rip + j) & I.MASK64) for j in range(I.WINDOW))
            got = I.identify(w)
            n += 1
            if got is None:
                keys.setdefault(("?", w[:4].hex()), {"programs": set(), "window": w.hex(), "ident": None})["programs"].add(pdir.name)
                continue
            name, key, regs, mem_l, cc = got
            e = keys.setdefault((name, key.hex()), {"programs": set(), "window": w.hex(), "ident": got})
            e["programs"].add(pdir.name)
        per[pdir.name] = n
    return keys, per


def _run_job(name):
    from .m4_scalar import Scalar
    m, dec, base = _P["m"], _P["dec"], _P["base"]
    pdir = Path(base) / name
    tr = np.load(pdir / "trace.npz")["steps"]
    t = time.time()
    r = R.run_program(m, dec, pdir / "mem.bin", tr, oracle=_P["traj"][name], scalar=Scalar(m),
                      cpu=R.cpu_model(pdir / "mem.bin", tr))  # M5e: supplied CPUID/XGETBV (glibc programs)
    r["wall_s"] = round(time.time() - t, 1)
    return name, r


def _oracle_job(pdir):
    tr = np.load(Path(pdir) / "trace.npz")["steps"]
    orc = R.oracle_program(Path(pdir) / "mem.bin", tr, record=True, cpu=R.cpu_model(Path(pdir) / "mem.bin", tr))
    return Path(pdir).name, orc


def run_all(m, dec, base, names, W):
    import multiprocessing as mp
    _P.update(m=m, dec=dec, base=str(base))
    with mp.get_context("fork").Pool(min(W, len(names))) as pool:
        return dict(pool.map(_run_job, names))


def rebuild(m, data, log):
    merged, maps = L.merge_records(m.literal_records, log=log)
    merged = L.resolve_collisions(m, data, merged, maps, log=log)
    merged = L.sib_run_evidence(m, data, merged, maps, log=log)
    merged = L.field_coverage(data, merged, log=log, m=m, maps=maps)
    m.records, m.code_map = merged, maps
    return L.FabricDecoder(merged, maps)


def refute(m, data, idx, n_old, log, rounds=3):
    """Older literal records that are wrong on new training rows the decoder routes to them are dropped (evidence,
    not oracle names): a stale record learned from rows that could not separate two explanations loses to new rows."""
    dropped = 0
    dec = rebuild(m, data, log)
    for _ in range(rounds):
        ok, ab = L.score(m, dec, data, idx)
        wrong = idx[~ok & ~ab]
        if not len(wrong):
            break
        kill = set()
        for i in wrong.tolist():
            w = bytes(data["window"][i])
            for j, r in enumerate(m.literal_records[:n_old]):
                if j in kill or not w.startswith(r["key"]):
                    continue
                eff = r["effect"]
                x = L.Ctx(m, data, np.array([i]), eff["k"], eff["runs"])
                try:
                    good = bool(L.check_rows(m, eff, x)[0][0])
                except Exception:  # noqa: BLE001 - a record that cannot even evaluate the row is refuted
                    good = False
                if not good:
                    kill.add(j)
        if not kill:
            break
        m.literal_records = [r for j, r in enumerate(m.literal_records) if j not in kill]
        n_old -= sum(1 for j in kill if j < n_old)
        dropped += len(kill)
        dec = rebuild(m, data, log)
    return dec, dropped


def validate(m, dec, idents, seed, grid, W, rows_per_key):
    """Fresh random-state rows for each (form, key); returns {key: (n, exact, abstain)} and the rows."""
    targets = [(name, key, regs, mem, cc, rows_per_key) for name, key, regs, mem, cc in idents]
    v = I.generate_targeted(targets, seed, grid, workers=W, nearby_keys=0)
    out = {}
    if v is None:
        return out
    v["store"] = L.learner_store(v)
    ok, ab = L.score(m, dec, v)
    for k in np.unique(v["o_key"]):
        sel = v["o_key"] == k
        out[str(k)] = (int(sel.sum()), int(ok[sel].sum()), int(ab[sel].sum()))
    return out


def worker():
    import pickle
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=False)
    logf = open(out / "log.txt", "w")
    iters_f = open(out / "iterations.jsonl", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack, publish
    t0 = time.time()
    W = cfg["workers"]
    grid = I.make_grid()
    budget = cfg["budget_s"]
    base = Path(cfg["native_dir"])
    names = sorted(p.name for p in base.glob("*") if p.is_dir())

    # ---- base machine from random transitions
    train = I.generate(cfg["keys_per_form"], cfg["seed"], "train", grid, workers=W)
    log(f"[cegis] train rows {len(train['rip'])} t={time.time() - t0:.0f}s")
    m, dec, info, pending = L.learn(train, log=log, workers=W)
    train.pop("overrides", None)  # the learner's page store holds them (flat arrays); the dict copies are not needed
    info["base_train_rows"] = int(len(train["rip"]))
    res = {"config": cfg, "menu": L.MENU, "learn": info, "grid": grid, "iterations": []}

    # ---- dev-program-guided selection: what the programs execute (code bytes only)
    pkeys, per_prog = program_keys(base)
    unident = [k for k, e in pkeys.items() if e["ident"] is None]
    log(f"[cegis] dev programs: {sum(per_prog.values())} distinct instructions, {len(pkeys)} distinct keys, "
        f"{len({k[0] for k in pkeys})} forms, unidentified {len(unident)}")
    held_keys = {k for k, e in pkeys.items() if e["ident"] is not None and I.hits_cell(grid, I.FORMS[k[0]], e["ident"][2])}
    if held_keys:
        log(f"[cegis] {len(held_keys)} program keys touch held-out cells: never trained on (they stay a cells-type test)")

    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(W, len(names))) as pool:
        orc = dict(pool.map(_oracle_job, [str(base / n) for n in names]))
    traj = {n: o.pop("trajectory") for n, o in orc.items()}
    _P["traj"] = traj
    log(f"[cegis] Unicorn trajectories for {len(traj)} programs t={time.time() - t0:.0f}s")

    attempts = Counter()
    stuck = {}
    fixed_log = {}
    it = 0
    prog_res = {}
    while True:
        it += 1
        ti = time.time()
        rec = {"iter": it, "t": round(ti - t0, 1)}
        # static validation of every program key (fresh random states)
        idents = [e["ident"] for k, e in pkeys.items() if e["ident"] is not None and k not in held_keys]
        val = validate(m, dec, idents, cfg["seed"] * 100 + it, grid, W, cfg["validate_rows"])
        bad_static = {k: v for k, v in val.items() if v[1] < v[0]}
        rec["static"] = {"keys": len(val), "exact_keys": len(val) - len(bad_static),
                         "abstain_keys": sum(1 for v in bad_static.values() if v[2] == v[0] - v[1]),
                         "wrong_keys": sum(1 for v in bad_static.values() if v[2] < v[0] - v[1])}
        # dynamic: the programs
        prog_res = run_all(m, dec, base, names, W)
        dyn = {}
        done_progs = 0
        for n, r in prog_res.items():
            nat = json.loads((base / n / "native.json").read_text())
            ok = r["status"] == "exit" and r["exit_code"] == nat.get("native_exit") and r["stdout"] == nat.get("native_stdout")
            done_progs += ok
            site = None
            if r["status"].startswith("abstain") and r.get("abstain_at"):
                site, why = r["abstain_at"]["window"], "abstain"
            elif r.get("divergence_vs_oracle"):
                site, why = r["divergence_vs_oracle"]["window"], "left Unicorn trajectory"
            if site is not None:
                got = I.identify(bytes.fromhex(site))
                if got is not None:
                    dyn.setdefault(got[1].hex(), (got, why, n))
        rec["programs"] = {n: {"status": r["status"], "exit": r["exit_code"], "steps": r["steps_run"],
                               "exact_regs_rip": r["exact_steps_regs_rip"], "vs_unicorn": r["exact_steps_vs_oracle"],
                               "first_div_vs_unicorn": r["first_divergence_vs_oracle"]} for n, r in prog_res.items()}
        rec["programs_done"] = done_progs
        # drain (first iteration): forms with unfitted training rows
        drain = {}
        if it == 1 and pending.any():
            seen_f = Counter()
            for i in np.flatnonzero(pending):
                f = str(train["o_form"][i])
                if seen_f[f] >= cfg["drain_keys_per_form"]:
                    continue
                got = I.identify(bytes(train["window"][i]))
                if got is not None and got[1].hex() not in drain:
                    drain[got[1].hex()] = (got, "unfitted train rows", None)
                    seen_f[f] += 1
        # targets
        cand = {}
        for k, v in bad_static.items():
            e = next(e for kk, e in pkeys.items() if kk[1] == k)
            cand[k] = (e["ident"], "static " + ("abstain" if v[2] == v[0] - v[1] else "wrong") + f" {v[1]}/{v[0]}", sorted(e["programs"]))
        held_cand = {}
        for k, (got, why, n) in dyn.items():
            if (got[0], k) in held_keys:
                held_cand[k] = (got, why, n)
                continue
            cand.setdefault(k, (got, f"dynamic {why}", [n]))
        for k, (got, why, n) in drain.items():
            cand.setdefault(k, (got, f"drain {why}", []))
        for k in list(cand):
            if k in stuck:
                cand.pop(k)
        for k in list(held_cand):
            if k in stuck:
                held_cand.pop(k)
        rec["candidates"] = len(cand) + len(held_cand)
        rec["stuck"] = len(stuck)
        for k in list(fixed_log):
            if fixed_log[k] is None and k not in cand and k in val:
                fixed_log[k] = it
        all_done = done_progs == len(names)
        left = budget - (time.time() - t0)
        if all_done or not (cand or held_cand) or left < cfg["final_reserve_s"] or it > cfg["max_iter"]:
            rec["stop"] = "all programs exit with native stdout/exit" if all_done else ("no candidates" if not (cand or held_cand) else "budget")
            res["iterations"].append(rec)
            iters_f.write(json.dumps(rec, default=str) + "\n")
            iters_f.flush()
            log(f"[cegis] iter {it}: stop ({rec['stop']}); programs done {done_progs}/{len(names)}")
            break
        targets, tlog = [], []
        for k, (got, why, progs) in cand.items():
            attempts[k] += 1
            a = attempts[k]
            if a > cfg["max_attempts"]:
                stuck[k] = why
                continue
            n_rows = cfg["target_rows"] * (4 if a > 1 else 1)
            name, key, regs, mem, cc = got
            targets.append((name, key, regs, mem, cc, n_rows))
            fixed_log.setdefault(k, None)
            tlog.append({"form": name, "key": k, "why": why, "attempt": a, "rows": n_rows, "programs": progs})
        import random
        srng = random.Random(cfg["seed"] * 7919 + it)
        for k, (got, why, progs) in held_cand.items():
            attempts[k] += 1
            if attempts[k] > cfg["max_attempts"]:
                stuck[k] = "held-out cell: " + why
                continue
            sib = I.sibling_targets(*got, grid, cfg.get("sibling_keys", 6) * attempts[k], cfg["target_rows"], srng)
            targets += sib
            tlog.append({"form": got[0], "key": k, "why": f"held-out cell, {why}: trained {len(sib)} sibling keys (held role varied; "
                         "the cell itself never sampled)", "attempt": attempts[k], "rows": cfg["target_rows"] * len(sib), "programs": [progs],
                         "siblings": [t[1].hex() for t in sib]})
        tseed = cfg["seed"] * 1000 + cfg.get("iter_offset", 0) + it
        targets_f.write(json.dumps({"iter": it, "seed": tseed, "workers": W, "nearby_keys": cfg["nearby_keys"],
                                    "targets": [(a, b.hex(), c, d, e, n) for a, b, c, d, e, n in targets]}, default=str) + "\n")
        targets_f.flush()
        new = I.generate_targeted(targets, tseed, grid, workers=W, nearby_keys=cfg["nearby_keys"]) if targets else None
        n_new = 0 if new is None else len(new["rip"])
        if new is not None:
            idx = append_rows(train, new)
            recs, pend, st = L.fit_trie(m, train, rows=idx, allow_flag_search=True, log=lambda s: None, workers=W)
            n_old = len(m.literal_records)
            m.literal_records += recs
            if cfg.get("share_taps", True):
                L.share_taps(m, train, m.literal_records)
            dec, dropped = refute(m, train, idx, n_old, log)
            rec["fit"] = {"new_rows": n_new, "new_records": len(recs), "pending_new_rows": int(pend[idx].sum()),
                          "refuted_old_records": dropped, "fail_stats": dict(st.most_common(8))}
        rec["targets"] = tlog
        rec["wall_s"] = round(time.time() - ti, 1)
        res["iterations"].append(rec)
        iters_f.write(json.dumps(rec, default=str) + "\n")
        iters_f.flush()
        log(f"[cegis] iter {it}: static exact keys {rec['static']['exact_keys']}/{rec['static']['keys']}, programs done "
            f"{done_progs}/{len(names)}, candidates {len(cand)} (stuck {len(stuck)}), new rows {n_new}, "
            f"records {len(m.records)}, wall {rec['wall_s']}s t={time.time() - t0:.0f}s; "
            + " ".join(f"{n}:{r['status'][:7]}@{r['steps_run']}" for n, r in sorted(prog_res.items())))
        publish([str(out / "iterations.jsonl"), str(out / "log.txt")], f"m4-cegis-iter-{it}")

    # ---- freeze, then the panels and the final program runs
    res["stuck"] = stuck
    res["fixed_at_iteration"] = fixed_log
    res["train_rows_final"] = int(len(train["rip"]))
    with open(out / "machine_full.pkl", "wb") as f:
        pickle.dump({"m": m, "dec": dec}, f)
    publish([str(out / "machine_full.pkl"), str(out / "iterations.jsonl"), str(out / "log.txt")], "m4-cegis-machine")
    res["spf"] = R.verify_spf(m)
    for split, seed in (("iid", cfg["seed"] * 10 + 1), ("cells", cfg["seed"] * 10 + 2), ("literal", cfg["seed"] * 10 + 3)):
        d = I.generate(cfg["test_keys_per_form"], seed, split, grid, workers=W)
        d["store"] = L.learner_store(d)
        res[split] = R.score_split(m, dec, d, split, log)
        if split == "iid":
            res["scalar_vs_vector"] = R.check_agreement(m, dec, d)
    progs = {}
    for n in names:
        nat = json.loads((base / n / "native.json").read_text())
        progs[n] = {"learned": prog_res[n], "unicorn": orc[n], "native": nat}
    res["programs"] = progs
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    publish([str(out / "m4.json")], "m4-cegis-final")
    pack(out)


# ------------------------------------------------------------------ M5d: dev-guided CEGIS on top of the blind machine

# proof gate on M5d v2 (job-463af3aab544, cert-job-6c3f2e66d6b1): records whose src_alts kept a constant ZF; the
# machine abstained exactly when the result was zero (13 of the 28 unguarded stops)
GATE_ZF_WINDOWS = {9991: "4885c075e1488b15", 9995: "4885f6742a660f1f", 9620: "85d275d783c330c6", 9575: "84d2741680fa0175",
                   10464: "81fb2c0100000f85", 9569: "84c90f8499000000", 9994: "4885ed0f84600100", 9658: "39c1745c4863c848"}


def custody(m, dec, data, n=20000):
    """The resumed machine on a sample of its own records' regenerated rows: exact, abstain and wrong separately; the
    rows must be the ones it was fitted on (exact > 0.97) and it must not be wrong on them (< 1e-3)."""
    rr = np.unique(np.concatenate([r["rows"] for r in m.records]))
    assert rr.max() < len(data["rip"]), "machine references rows beyond the regenerated data"
    sample = np.random.default_rng(0).choice(rr, size=min(n, len(rr)), replace=False)
    ok, ab = L.score(m, dec, data, rows=sample)
    out = {"n": int(len(sample)), "exact": float(ok.mean()), "abstain": float(ab.mean()), "wrong": float((~ok & ~ab).mean())}
    assert out["exact"] > 0.97, "regenerated rows do not match the machine's records"
    assert out["wrong"] < 1e-3, "regenerated rows: the machine is wrong on its own record rows"
    return out


def rebuild_pass(cfg, it0):
    """M5d rebuild lease: one program pass on the rebuilt machine (iteration it0 + 1), never a refit."""
    if cfg.get("rebuild"):
        cfg["max_iter"] = it0


def programs_done(prog_res, base):
    n = 0
    for name, r in prog_res.items():
        nat = json.loads((Path(base) / name / "native.json").read_text())
        n += r["status"] == "exit" and r["exit_code"] == nat.get("native_exit") and r["stdout"] == nat.get("native_stdout")
    return n


def panel_data(cfg, grid, W):
    out = {}
    for split, seed in (("iid", cfg["seed"] * 10 + 1), ("cells", cfg["seed"] * 10 + 2), ("literal", cfg["seed"] * 10 + 3)):
        d = I.generate(cfg["test_keys_per_form"], seed, split, grid, workers=W)
        d["store"] = L.learner_store(d)
        out[split] = d
    return out


def sweep_data(pkeys, seed, grid, W, n_gate=4000, n_other=1000, n_shift=2000, gate=None):
    """Fresh random-state Unicorn rows (M5c distribution, boundary-state rows included) for the zero-result and the
    masked-zero shift-count cases, which iid probes almost never reach. The programs choose keys only: the gate's
    constant-ZF records, every program test/cmp key, every program shift-by-CL key."""
    targets, kind = [], {}
    for rid, w in (GATE_ZF_WINDOWS if gate is None else gate).items():
        got = I.identify(bytes.fromhex(w) + bytes(I.WINDOW - len(w) // 2))
        if got is not None:
            name, key, regs, mem, cc = got
            targets.append((name, key, regs, mem, cc, n_gate))
            kind[key.hex()] = ("zf", f"gate rec {rid}")
    for (name, kh), e in sorted(pkeys.items()):
        if e["ident"] is None or kh in kind:
            continue
        if name.startswith(("test_", "cmp_")):
            targets.append(tuple(e["ident"]) + (n_other,))
            kind[kh] = ("zf", "program test/cmp")
        elif name.endswith("_cl"):
            targets.append(tuple(e["ident"]) + (n_shift,))
            kind[kh] = ("shift0", "program shift by CL")
    d = I.generate_targeted(targets, seed, grid, workers=W, nearby_keys=0)
    d["store"] = L.learner_store(d)
    keys = np.array([str(k) for k in d["o_key"]])
    forms = np.asarray(d["o_form"])
    sel = np.zeros(len(keys), bool)
    for i, k in enumerate(keys.tolist()):
        if kind[k][0] == "zf":
            sel[i] = d["a_flags"][i, I.FLAGS.index("ZF")] == 1  # the true result is zero (Unicorn label)
        else:
            m = 63 if "64" in str(forms[i]) else 31
            sel[i] = (int(d["regs"][i, 1]) & m) == 0  # the masked count is zero (before-state)
    return {"data": d, "sel": sel, "keys": keys, "kind": kind, "forms": forms}


def sweep_score(m, dec, sw, log, tag):
    d, sel, keys = sw["data"], sw["sel"], sw["keys"]
    ok, ab = L.score(m, dec, d)
    out = {}
    for k in sorted(sw["kind"]):
        s = keys == k
        for part, mask in (("case", s & sel), ("rest", s & ~sel)):
            n = int(mask.sum())
            if n == 0:
                continue
            ri, _ = dec.lookup(d["window"][np.flatnonzero(s)[0]])
            out.setdefault(k, {"kind": sw["kind"][k], "form": str(sw["forms"][np.flatnonzero(s)[0]]), "record": ri})[part] = {
                "n": n, "exact": int(ok[mask].sum()), "abstain": int(ab[mask].sum()), "wrong": int((~ok[mask] & ~ab[mask]).sum())}
    for kk in ("zf", "shift0"):
        tot = Counter()
        for v in out.values():
            if v["kind"][0] == kk and "case" in v:
                tot.update(v["case"])
        log(f"[sweep] {tag} {kk}: case rows {tot['n']} exact {tot['exact']} abstain {tot['abstain']} wrong {tot['wrong']}")
    for k, v in out.items():
        if v["kind"][1].startswith("gate") and "case" in v:
            log(f"[sweep] {tag} {v['kind'][1]} {k} {v['form']} record {v['record']}: {v['case']}")
    return out


def worker_devcegis():
    """M5d: the M4 dev-program-guided loop (static validation of every program key, dynamic first stop, refit, rerun)
    on top of a frozen blind machine (M5c'). The blind base rows are regenerated bit for bit (same seeds, keys and
    worker count) and the frozen records must be exact on them. Programs only choose which keys are sampled; every
    label is a fresh random-state Unicorn row (the M5c row distribution, boundary-state rows included). All dev
    programs (dev 22 + the spent fresh 24) are dev here."""
    import gc
    import os
    import pickle
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=True)
    logf = open(out / "log.txt", "w")
    iters_f = open(out / "iterations.jsonl", "w")
    targets_f = open(out / "targets.jsonl", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack, publish
    t0 = time.time()
    W, PW = cfg["workers"], cfg["prog_workers"]
    grid = I.make_grid()
    budget = cfg["budget_s"]
    base = Path(cfg["native_dir"])
    names = sorted(p.name for p in base.glob("*") if p.is_dir())
    os.environ["ISA_ROW_DIST"] = "m5c"
    gen_w = cfg["base_workers"]
    # ---- the blind machine's rows, regenerated (custody: its records must be exact on them)
    rand = I.generate(cfg["keys_per_form"], cfg["seed"], "train", grid, workers=gen_w)
    strat = I.generate_stratified(cfg["keys_per_cell"], cfg["seed"], grid, workers=gen_w)
    train = concat_data(rand, strat)
    del rand, strat
    for tf in cfg["replay_targets"]:
        for line in open(tf):
            t = json.loads(line)
            tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
            new = I.generate_targeted(tg, t["seed"], grid, workers=gen_w, nearby_keys=t["nearby_keys"])
            if new is not None:
                train = concat_data(train, new)
    train["store"] = L.learner_store(train)
    train.pop("overrides", None)
    # resume (a killed M5d lease): this lease's earlier iterations' rows, replayed in order, then its latest machine
    it0 = 0
    for tf in cfg.get("resume_targets", []):
        for line in open(tf):
            t = json.loads(line)
            tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
            if tg:
                append_rows(train, I.generate_targeted(tg, t["seed"], grid, workers=t["workers"], nearby_keys=t["nearby_keys"]))
            targets_f.write(line)
            it0 = max(it0, int(t["iter"]))
    if cfg.get("resume_latest"):
        cfg["resume_machine"], cfg["resume_machine_sha256"] = cfg["resume_latest"], cfg["resume_latest_sha256"]
    import hashlib
    h = hashlib.sha256()
    with open(cfg["resume_machine"], "rb") as f:  # a Beast-staged pickle (too big for the payload), checked by hash
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == cfg["resume_machine_sha256"], "staged machine pickle does not match its recorded sha256"
    with open(cfg["resume_machine"], "rb") as f:
        mm = pickle.load(f)
    m, dec = mm["m"], mm["dec"]
    cust = custody(m, dec, train)
    log(f"[devcegis] resumed {cfg['resume_machine']}: {len(m.records)} records, rows {len(train['rip'])}, record rows exact "
        f"{cust['exact']:.4f} abstain {cust['abstain']:.4f} wrong {cust['wrong']:.5f} on {cust['n']} t={time.time() - t0:.0f}s")
    res = {"config": cfg, "menu": L.MENU, "resume_record_rows_exact": cust["exact"], "resume_record_rows": cust, "iterations": []}
    rebuild_pass(cfg, it0)

    pkeys, per_prog = program_keys(base)
    log(f"[devcegis] {len(names)} dev programs: {sum(per_prog.values())} distinct instructions, {len(pkeys)} distinct keys, "
        f"{len({k[0] for k in pkeys})} forms, unidentified {sum(1 for e in pkeys.values() if e['ident'] is None)}")
    import multiprocessing as mp
    with mp.get_context("fork").Pool(min(PW, len(names))) as pool:
        orc = dict(pool.map(_oracle_job, [str(base / n) for n in names]))
    _P["traj"] = {n: o.pop("trajectory") for n, o in orc.items()}
    log(f"[devcegis] Unicorn trajectories for {len(names)} programs t={time.time() - t0:.0f}s")

    sweep_rows, panel_rows = None, None
    if cfg.get("rebuild"):
        # control on the identical regenerated rows: the loaded machine's programs, sweeps and panels before the rebuild
        sweep_rows = sweep_data(pkeys, cfg["seed"] * 1000 + 900, grid, W)
        res["sweep_loaded"] = sweep_score(m, dec, sweep_rows, log, "loaded")
        res["programs_loaded"] = run_all(m, dec, base, names, PW)
        log(f"[devcegis] loaded machine: programs done {programs_done(res['programs_loaded'], base)}/{len(names)} t={time.time() - t0:.0f}s")
        panel_rows = panel_data(cfg, grid, W)
        for split, d in panel_rows.items():
            res["loaded/m5c/" + split] = R.score_split(m, dec, d, "loaded/m5c/" + split, log)
    if not cfg.get("resume_latest"):
        # the blind machine's merge/coverage redone with this runtime's rules (M5c'': pooled refutation blame)
        zf0 = L.const_zf_alt_records(m.literal_records)
        L.share_taps(m, train, m.literal_records)  # src_alts dedupe under this runtime (M5c'': extension-flag canon)
        zf1 = L.const_zf_alt_records(m.literal_records)
        res["const_zf_alt_records"] = {"before": zf0, "after": zf1}
        res["src_alt_fallbacks"] = getattr(L.share_taps, "last_fallbacks", {})
        log(f"[devcegis] src_alt flags on the per-record fit: {res['src_alt_fallbacks']}")
        log(f"[devcegis] constant-ZF src_alts on records with a real ZF: {len(zf0)} before share_taps, {len(zf1)} after "
            f"(before: {zf0[:12]})")
        dec = rebuild(m, train, log)
        log(f"[devcegis] rebuilt taps + merge + field coverage under this runtime: {len(m.records)} records t={time.time() - t0:.0f}s")
    if sweep_rows is not None:
        res["sweep_rebuilt"] = sweep_score(m, dec, sweep_rows, log, "rebuilt")

    attempts, stuck, fixed_log = Counter(), {}, {}
    it = it0
    prog_res = {}
    strat_hist = []
    while True:
        it += 1
        ti = time.time()
        rec = {"iter": it, "t": round(ti - t0, 1)}
        gc.collect()
        gc.freeze()
        # wrong guard (opus48): fresh blind probes every iteration (seeds disjoint from the panels and M5a/M5b probes);
        # their WRONG keys are candidates first; the loop stops on a wrong rise and freezes the machine before it
        pr = I.generate(cfg["probe_keys"], cfg["seed"] * 1000 + 550 + it, "iid", grid, workers=W)
        ps = I.generate_stratified(1, cfg["seed"] * 1000 + 750 + it, grid, workers=W, rows_per_key=2)
        fails_r, st_r = probe_failures(m, dec, pr)
        fails_s, st_s = probe_failures(m, dec, ps)
        del pr, ps
        rec["probe_iid"], rec["probe_strat"] = st_r, st_s
        probe_wrong = {k: v for k, v in list(fails_r.items()) + list(fails_s.items()) if v[1] == "wrong"}
        strat_hist.append(st_s["wrong"])
        log(f"[devcegis] iter {it}: probe iid wrong {st_r['wrong']:.4f} (exact {st_r['exact']:.4f}), strat wrong {st_s['wrong']:.4f} "
            f"(exact {st_s['exact']:.4f}); probe wrong keys {len(probe_wrong)}")
        guard = None
        if st_r["wrong"] > cfg["guard_iid_wrong"]:
            guard = f"wrong guard: probe iid wrong {st_r['wrong']:.4f} > {cfg['guard_iid_wrong']}"
        elif len(strat_hist) >= 4 and strat_hist[-1] > strat_hist[-2] > strat_hist[-3] > strat_hist[-4]:
            guard = f"wrong guard: probe strat wrong rose 3 iterations running {strat_hist[-4:]}"
        if guard is not None and (out / "machine_prev.pkl").exists():
            with open(out / "machine_prev.pkl", "rb") as f:  # the machine before the refit that raised wrong
                mm = pickle.load(f)
            m, dec = mm["m"], mm["dec"]
            rec["stop"] = guard + f"; frozen machine = iteration {mm['iter']} (before that refit)"
            res["iterations"].append(rec)
            iters_f.write(json.dumps(rec, default=str) + "\n")
            iters_f.flush()
            log(f"[devcegis] iter {it}: stop ({rec['stop']})")
            prog_res = run_all(m, dec, base, names, PW)
            break
        idents = [e["ident"] for k, e in pkeys.items() if e["ident"] is not None]
        val = validate(m, dec, idents, cfg["seed"] * 100 + 50 + it, grid, W, cfg["validate_rows"])
        bad_static = {k: v for k, v in val.items() if v[1] < v[0]}
        rec["static"] = {"keys": len(val), "exact_keys": len(val) - len(bad_static),
                         "abstain_keys": sum(1 for v in bad_static.values() if v[2] == v[0] - v[1]),
                         "wrong_keys": sum(1 for v in bad_static.values() if v[2] < v[0] - v[1])}
        prog_res = run_all(m, dec, base, names, PW)
        dyn, done_progs = {}, 0
        for n, r in prog_res.items():
            nat = json.loads((base / n / "native.json").read_text())
            ok = r["status"] == "exit" and r["exit_code"] == nat.get("native_exit") and r["stdout"] == nat.get("native_stdout")
            done_progs += ok
            site = None
            if r["status"].startswith("abstain") and r.get("abstain_at"):
                site, why = r["abstain_at"]["window"], "abstain"
            elif r.get("divergence_vs_oracle"):
                site, why = r["divergence_vs_oracle"]["window"], "left Unicorn trajectory"
            if site is not None:
                got = I.identify(bytes.fromhex(site))
                if got is not None:
                    dyn.setdefault(got[1].hex(), (got, why, n))
        rec["programs"] = {n: {"status": r["status"], "exit": r["exit_code"], "steps": r["steps_run"],
                               "exact_regs_rip": r["exact_steps_regs_rip"], "vs_unicorn": r["exact_steps_vs_oracle"],
                               "first_div_vs_unicorn": r["first_divergence_vs_oracle"],
                               "stop_window": (r.get("abstain_at") or {}).get("window")} for n, r in prog_res.items()}
        rec["programs_done"] = done_progs
        cand = {}
        for k, (got, why) in probe_wrong.items():  # wrong first
            cand[k] = (got, "blind probe wrong", [])
        for k, v in bad_static.items():
            e = next(e for kk, e in pkeys.items() if kk[1] == k)
            cand.setdefault(k, (e["ident"], "static " + ("abstain" if v[2] == v[0] - v[1] else "wrong") + f" {v[1]}/{v[0]}", sorted(e["programs"])))
        for k, (got, why, n) in dyn.items():
            cand.setdefault(k, (got, f"dynamic {why}", [n]))
        for k in list(cand):
            if k in stuck:
                cand.pop(k)
        rec["candidates"], rec["stuck"] = len(cand), len(stuck)
        for k in list(fixed_log):
            if fixed_log[k] is None and k not in cand and k in val:
                fixed_log[k] = it
        all_done = done_progs == len(names)
        left = budget - (time.time() - t0)
        if all_done or not cand or left < cfg["final_reserve_s"] or it > cfg["max_iter"]:
            rec["stop"] = ("rebuild: one program pass, no refit" if cfg.get("rebuild") else
                           "all programs exit with native stdout/exit" if all_done else ("no candidates" if not cand else "budget"))
            res["iterations"].append(rec)
            iters_f.write(json.dumps(rec, default=str) + "\n")
            iters_f.flush()
            log(f"[devcegis] iter {it}: stop ({rec['stop']}); programs done {done_progs}/{len(names)}")
            break
        targets, tlog = [], []
        for k, (got, why, progs) in cand.items():
            attempts[k] += 1
            a = attempts[k]
            if a > cfg["max_attempts"]:
                stuck[k] = why
                continue
            n_rows = cfg["target_rows"] * (4 if a > 1 else 1)
            name, key, regs, mem, cc = got
            targets.append((name, key, regs, mem, cc, n_rows))
            fixed_log.setdefault(k, None)
            tlog.append({"form": name, "key": k, "why": why, "attempt": a, "rows": n_rows, "programs": progs})
        tseed = cfg["seed"] * 1000 + 300 + it
        targets_f.write(json.dumps({"iter": it, "seed": tseed, "workers": W, "nearby_keys": cfg["nearby_keys"],
                                    "targets": [(a, b.hex(), c, d, e, n) for a, b, c, d, e, n in targets]}, default=str) + "\n")
        targets_f.flush()
        new = I.generate_targeted(targets, tseed, grid, workers=W, nearby_keys=cfg["nearby_keys"]) if targets else None
        n_new = 0 if new is None else len(new["rip"])
        with open(out / "machine_prev.pkl", "wb") as f:  # the wrong guard's fallback
            pickle.dump({"m": m, "dec": dec, "iter": it - 1}, f)
        if new is not None:
            idx = append_rows(train, new)
            recs, pend, st = L.fit_trie(m, train, rows=idx, allow_flag_search=True, log=lambda s: None, workers=cfg["fit_workers"])
            n_old = len(m.literal_records)
            m.literal_records += recs
            L.share_taps(m, train, m.literal_records)
            dec, dropped = refute(m, train, idx, n_old, log)
            rec["fit"] = {"new_rows": n_new, "new_records": len(recs), "pending_new_rows": int(pend[idx].sum()),
                          "refuted_old_records": dropped, "fail_stats": dict(st.most_common(8))}
        rec["targets"] = tlog
        rec["wall_s"] = round(time.time() - ti, 1)
        res["iterations"].append(rec)
        iters_f.write(json.dumps(rec, default=str) + "\n")
        iters_f.flush()
        blob = pickle.dumps({"m": m, "dec": dec, "iter": it})
        # salvage: the loop state after every refit; every iteration's machine is kept (the refit is not monotone on
        # programs: M5d iterations 9/11/12 reached 23/46 and later ones fewer)
        for name in ("machine_latest.pkl", f"machine_iter{it}.pkl"):
            (out / name).write_bytes(blob)
        del blob
        log(f"[devcegis] iter {it}: static exact keys {rec['static']['exact_keys']}/{rec['static']['keys']} (wrong "
            f"{rec['static']['wrong_keys']}), programs done {done_progs}/{len(names)}, candidates {len(cand)} (stuck {len(stuck)}), "
            f"new rows {n_new}, records {len(m.records)}, wall {rec['wall_s']}s t={time.time() - t0:.0f}s; "
            + " ".join(f"{n}:{r['status'][:7]}@{r['steps_run']}" for n, r in sorted(prog_res.items())))
        publish([str(out / "iterations.jsonl"), str(out / "log.txt"), str(out / "machine_latest.pkl"), str(out / "targets.jsonl")],
                f"m5d-iter-{it}")

    # ---- freeze, then quick panels (M5c row distribution) and the final dev program results
    res["stuck"], res["fixed_at_iteration"] = stuck, fixed_log
    res["train_rows_final"] = int(len(train["rip"]))
    with open(out / "machine_full.pkl", "wb") as f:
        pickle.dump({"m": m, "dec": dec}, f)
    publish([str(out / "machine_full.pkl"), str(out / "iterations.jsonl"), str(out / "log.txt"), str(out / "targets.jsonl")], "m5d-machine")
    del train
    gc.collect()
    for split, d in (panel_rows or panel_data(cfg, grid, W)).items():
        res["m5c/" + split] = R.score_split(m, dec, d, "m5c/" + split, log)
    del panel_rows
    progs = {}
    for n in names:
        nat = json.loads((base / n / "native.json").read_text())
        progs[n] = {"learned": prog_res[n], "unicorn": orc[n], "native": nat}
        if "programs_loaded" in res:
            progs[n]["loaded"] = res["programs_loaded"][n]
    res.pop("programs_loaded", None)
    res["programs"] = progs
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    publish([str(out / "m4.json")], "m5d-final")
    pack(out)


# ------------------------------------------------------------------ M5: blind coverage (no program guidance)


def concat_data(a, b):
    return {k: (np.concatenate([a[k], b[k]]) if isinstance(a[k], np.ndarray) else a[k] + b[k]) for k in a if k != "store"}


def probe_failures(m, dec, probe):
    """(key hex -> (ident, why)) for every probe row the machine abstains on or gets wrong."""
    probe["store"] = L.learner_store(probe)
    ok, ab = L.score(m, dec, probe)
    out = {}
    for i in np.flatnonzero(~ok):
        w = bytes(probe["window"][i])
        got = I.identify(w)
        if got is None:
            continue
        k = got[1].hex()
        if k not in out:
            out[k] = (got, "wrong" if not ab[i] else "abstain")
        elif not ab[i]:
            out[k] = (got, "wrong")
    return out, {"rows": int(len(ok)), "exact": float(ok.mean()), "abstain": float(ab.mean()), "wrong": float((~ok & ~ab).mean())}


def worker_blind():
    """Blind coverage loop: stratified + random base data, then drain fresh iid/stratified probe failures and unfitted
    training keys; programs are only run at the end (dev checks), never used to choose what to learn."""
    import pickle
    import random
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=True)
    logf = open(out / "log.txt", "w")
    iters_f = open(out / "iterations.jsonl", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack, publish
    t0 = time.time()
    W = cfg["workers"]
    grid = I.make_grid()
    budget = cfg["budget_s"]

    import gc
    import os
    # row distribution: a resumed lease must regenerate its base rows bit for bit (legacy); a fresh one uses M5c's
    os.environ["ISA_ROW_DIST"] = cfg.get("row_dist", "legacy" if cfg.get("resume_machine") else "m5c")
    SF = tuple(cfg.get("strat_forms") or I.FORM_NAMES)  # smoke runs restrict the stratified/probe forms
    # base rows are a pure function of (seed, keys, worker count): a resumed lease regenerates them bit for bit
    gen_w = cfg.get("base_workers", W)
    rand = I.generate(cfg["keys_per_form"], cfg["seed"], "train", grid, workers=gen_w)
    cov_rand = I.cell_coverage(rand["window"])
    strat = I.generate_stratified(cfg["keys_per_cell"], cfg["seed"], grid, forms=SF, workers=gen_w)
    train = concat_data(rand, strat)
    cov_all = I.cell_coverage(train["window"])
    log(f"[blind] train rows {len(train['rip'])} (random {len(rand['rip'])}, stratified {len(strat['rip'])}); shape cells hit: "
        f"random {cov_rand['cells_hit']}/{cov_rand['cells_total']}, with stratified {cov_all['cells_hit']}/{cov_all['cells_total']} t={time.time() - t0:.0f}s")
    del rand, strat
    # M5c refit: earlier leases' blind target lists are replayed as KEYS with fresh rows from this generator (no
    # selection here; the keys were chosen by those leases' probe failures)
    for tf in cfg.get("replay_targets", []):
        for line in open(tf):
            t = json.loads(line)
            tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
            new = I.generate_targeted(tg, t["seed"], grid, workers=W, nearby_keys=t["nearby_keys"])
            if new is not None:
                train = concat_data(train, new)
        log(f"[blind] replayed {tf}: train rows {len(train['rip'])} t={time.time() - t0:.0f}s")
    targets_f = open(out / "targets.jsonl", "w")  # every iteration's targets + seed: the rows can be replayed exactly
    if cfg.get("resume_machine"):
        # M5b: continue from a frozen machine whose records index these base rows (plus replayed target rows)
        with open(cfg["resume_machine"], "rb") as f:
            mm = pickle.load(f)
        m, dec = mm["m"], mm["dec"]
        train["store"] = L.learner_store(train)
        train.pop("overrides", None)
        for tf in cfg.get("resume_targets", []):
            for line in open(tf):
                t = json.loads(line)
                tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
                append_rows(train, I.generate_targeted(tg, t["seed"], grid, workers=t["workers"], nearby_keys=t["nearby_keys"]))
                targets_f.write(line)
        # custody: the records' own rows must still be exact (the regenerated rows are the ones they were fitted on)
        rr = np.unique(np.concatenate([r["rows"] for r in m.records]))
        assert rr.max() < len(train["rip"]), "machine references rows beyond the regenerated data"
        sample = np.random.default_rng(0).choice(rr, size=min(20000, len(rr)), replace=False)
        ok_s, _ = L.score(m, dec, train, rows=sample)
        info = {"resumed_from": cfg["resume_machine"], "resume_rows": int(len(train["rip"])), "resume_record_rows_exact": float(ok_s.mean())}
        log(f"[blind] resumed {cfg['resume_machine']}: {len(m.records)} records, rows {len(train['rip'])}, "
            f"record rows exact {ok_s.mean():.4f} on {len(sample)} t={time.time() - t0:.0f}s")
        assert ok_s.mean() > 0.97, "regenerated rows do not match the machine's records"
        pending = np.zeros(len(train["rip"]), bool)
        res = {"config": cfg, "menu": L.MENU, "learn": info, "grid": grid, "coverage": {"random_only": cov_rand, "with_stratified": cov_all},
               "iterations": []}
    else:
        def salvage(stage, state):  # a lease killed mid-learn keeps its finished passes (resume_partial)
            with open(out / f"learn_{stage}.pkl", "wb") as f:
                pickle.dump(state, f)
            publish([str(out / f"learn_{stage}.pkl"), str(out / "log.txt")], f"m5-learn-{stage}")
            log(f"[blind] salvaged learn {stage} t={time.time() - t0:.0f}s")

        partial = None
        if cfg.get("resume_partial"):
            with open(cfg["resume_partial"], "rb") as f:
                partial = pickle.load(f)
        m, dec, info, pending = L.learn(train, log=log, workers=cfg.get("fit_workers", W), salvage=salvage, partial=partial)  # fork pools: tree RSS
        if partial is not None:
            # custody: the salvaged records' rows must be exact on the regenerated rows (same seeds and worker count)
            rr = np.unique(np.concatenate([r["rows"] for r in m.records]))
            assert rr.max() < len(train["rip"]), "salvaged records reference rows beyond the regenerated data"
            sample = np.random.default_rng(0).choice(rr, size=min(20000, len(rr)), replace=False)
            ok_s, _ = L.score(m, dec, train, rows=sample)
            info["resume_partial_record_rows_exact"] = float(ok_s.mean())
            log(f"[blind] resumed learn from {cfg['resume_partial']}: record rows exact {ok_s.mean():.4f} on {len(sample)}")
            assert ok_s.mean() > 0.97, "regenerated rows do not match the salvaged learn state"
        train.pop("overrides", None)
        info["base_train_rows"] = int(len(train["rip"]))
        res = {"config": cfg, "menu": L.MENU, "learn": info, "grid": grid, "coverage": {"random_only": cov_rand, "with_stratified": cov_all},
               "iterations": []}
        with open(out / "machine_base.pkl", "wb") as f:
            pickle.dump({"m": m, "dec": dec}, f)
        # base machine on the same iid panel the frozen machine is scored on (what the closed loop adds)
        d = I.generate(cfg["test_keys_per_form"], cfg["seed"] * 10 + 1, "iid", grid, workers=W)
        d["store"] = L.learner_store(d)
        res["base_iid"] = R.score_split(m, dec, d, "base_iid", log)
        del d
        publish([str(out / "machine_base.pkl"), str(out / "log.txt")], "m5-base")

    attempts, stuck = Counter(), {}
    it = 0
    while True:
        it += 1
        ti = time.time()
        rec = {"iter": it, "t": round(ti - t0, 1)}
        gc.collect()
        gc.freeze()  # forked workers must not copy the training heap by touching it in their own GC passes
        # fresh probes (seeds disjoint from the test panels): random iid keys and one key per shape cell
        off = cfg.get("iter_offset", 0)  # a resumed lease draws fresh probes (iteration numbers restart at 1)
        pr = I.generate(cfg["probe_keys"], cfg["seed"] * 1000 + 500 + off + it, "iid", grid, forms=SF, workers=W)
        ps = I.generate_stratified(1, cfg["seed"] * 1000 + 700 + off + it, grid, forms=SF, workers=W, rows_per_key=2)
        fails_r, st_r = probe_failures(m, dec, pr)
        fails_s, st_s = probe_failures(m, dec, ps)
        rec["probe_iid"], rec["probe_strat"] = st_r, st_s
        cand = {}
        for k, v in list(fails_r.items()) + list(fails_s.items()):
            if k not in stuck and (k not in cand or v[1] == "wrong"):
                cand[k] = v
        if it == 1 and pending.any():
            seen = Counter()
            for i in np.flatnonzero(pending):
                got = I.identify(bytes(train["window"][i]))
                if got is None or got[1].hex() in cand or seen[got[0]] >= cfg["drain_keys_per_form"]:
                    continue
                cand[got[1].hex()] = (got, "unfitted train rows")
                seen[got[0]] += 1
        rec["candidates"] = len(cand)
        rec["stuck"] = len(stuck)
        left = budget - (time.time() - t0)
        log(f"[blind] iter {it}: probe iid exact {st_r['exact']:.4f} abstain {st_r['abstain']:.4f} wrong {st_r['wrong']:.4f}; "
            f"probe strat exact {st_s['exact']:.4f} abstain {st_s['abstain']:.4f} wrong {st_s['wrong']:.4f}; candidates {len(cand)} stuck {len(stuck)} t={time.time() - t0:.0f}s")
        full = len(train["rip"]) > cfg.get("max_train_rows", 1 << 40)
        if not cand or left < cfg["final_reserve_s"] or it > cfg["max_iter"] or full:
            rec["stop"] = "no candidates" if not cand else "row cap" if full else "budget"
            res["iterations"].append(rec)
            iters_f.write(json.dumps(rec, default=str) + "\n")
            iters_f.flush()
            break
        # wrong rows first, then round-robin over forms in a seeded random order (no opcode-order bias under the cap)
        rng = random.Random(cfg["seed"] * 1000 + it)
        order = sorted(cand, key=lambda k: (cand[k][1] != "wrong", rng.random()))
        seen_f = Counter()
        ranked = []
        for k in order:
            ranked.append((cand[k][1] != "wrong", seen_f[cand[k][0][0]], len(ranked), k))
            seen_f[cand[k][0][0]] += 1
        keys = [k for *_, k in sorted(ranked)][:cfg["max_targets"]]
        targets, why = [], Counter()
        room = cfg.get("max_train_rows", 1 << 40) - len(train["rip"])  # the row cap holds before appending
        for k in keys:
            got, w = cand[k]
            n = cfg["target_rows"] * (2 if attempts[k] >= 1 else 1)
            cost = n + cfg["nearby_keys"] * I.FORMS[got[0]].rows  # the target's rows plus its nearby keys' rows
            if cost > room:
                break
            attempts[k] += 1
            if attempts[k] > cfg["max_attempts"]:
                stuck[k] = w
                continue
            name, key, regs, mem, cc = got
            targets.append((name, key, regs, mem, cc, n))
            room -= cost
            why[w] += 1
        rec["targets"] = {"n": len(targets), "why": dict(why), "forms": dict(Counter(t[0] for t in targets).most_common(15))}
        if not targets:  # every candidate is stuck or the row cap leaves no room
            rec["stop"] = "no targets (stuck or row cap)"
            res["iterations"].append(rec)
            iters_f.write(json.dumps(rec, default=str) + "\n")
            iters_f.flush()
            break
        tseed = cfg["seed"] * 1000 + cfg.get("iter_offset", 0) + it
        targets_f.write(json.dumps({"iter": it, "seed": tseed, "workers": W, "nearby_keys": cfg["nearby_keys"],
                                    "targets": [(a, b.hex(), c, d, e, n) for a, b, c, d, e, n in targets]}, default=str) + "\n")
        targets_f.flush()
        new = I.generate_targeted(targets, tseed, grid, workers=W, nearby_keys=cfg["nearby_keys"])
        if new is not None:
            idx = append_rows(train, new)
            # fit workers fork the learner heap: fewer of them keep the tree RSS under the grant
            recs, pend, st = L.fit_trie(m, train, rows=idx, allow_flag_search=True, log=lambda s: None,
                                        workers=cfg.get("fit_workers", W))
            n_old = len(m.literal_records)
            m.literal_records += recs
            L.share_taps(m, train, m.literal_records)
            dec, dropped = refute(m, train, idx, n_old, log)
            rec["fit"] = {"new_rows": int(len(new["rip"])), "new_records": len(recs), "pending_new_rows": int(pend[idx].sum()),
                          "refuted_old_records": dropped, "fail_stats": dict(st.most_common(8))}
        rec["wall_s"] = round(time.time() - ti, 1)
        res["iterations"].append(rec)
        iters_f.write(json.dumps(rec, default=str) + "\n")
        iters_f.flush()
        log(f"[blind] iter {it}: targets {len(targets)} {dict(why)}, fit {rec.get('fit', {}).get('new_records')} records, "
            f"records {len(m.records)}, wall {rec['wall_s']}s")
        with open(out / "machine_latest.pkl", "wb") as f:  # salvage: the loop state after every refit
            pickle.dump({"m": m, "dec": dec, "iter": it}, f)
        publish([str(out / "iterations.jsonl"), str(out / "log.txt"), str(out / "machine_latest.pkl"), str(out / "targets.jsonl")],
                f"m5-iter-{it}")

    # ---- freeze
    res["stuck"] = stuck
    res["train_rows_final"] = int(len(train["rip"]))
    with open(out / "machine_full.pkl", "wb") as f:
        pickle.dump({"m": m, "dec": dec}, f)
    publish([str(out / "machine_full.pkl"), str(out / "iterations.jsonl"), str(out / "log.txt"), str(out / "targets.jsonl")], "m5-machine")
    del train, pending
    gc.collect()
    if not cfg.get("panels", True):  # panels and programs run in a separate eval-only lease
        (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
        publish([str(out / "m4.json")], "m5-final")
        pack(out)
        return
    res["spf"] = R.verify_spf(m)
    for split, seed in (("iid", cfg["seed"] * 10 + 1), ("cells", cfg["seed"] * 10 + 2), ("literal", cfg["seed"] * 10 + 3)):
        d = I.generate(cfg["test_keys_per_form"], seed, split, grid, workers=W)
        d["store"] = L.learner_store(d)
        res[split] = R.score_split(m, dec, d, split, log)
        if split == "iid":
            res["scalar_vs_vector"] = R.check_agreement(m, dec, d)
    dst = I.generate_stratified(1, cfg["seed"] * 10 + 4, grid, workers=W, rows_per_key=2)
    dst["store"] = L.learner_store(dst)
    res["strat_panel"] = R.score_split(m, dec, dst, "strat_panel", log)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    publish([str(out / "m4.json")], "m5-panels")
    # dev checks (programs never chose anything above)
    progs = {}
    for setname in cfg.get("program_sets", []):
        base = Path(setname)
        progs.update({f"{base.name}/{k}": v for k, v in R.run_programs(m, dec, base, cfg.get("prog_workers", W), log, out=out).items()})
    res["programs"] = progs
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    publish([str(out / "m4.json")], "m5-final")
    pack(out)


def worker_eval():
    """Eval-only lease on frozen machines (no learning, no training data in memory): per machine, SPF verify, the
    transition panels at the M4 seeds under each row distribution, the stratified panel, then every staged program set
    vs native ptrace. One <label>.json per machine (m4.json when there is a single machine)."""
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack
    machines = cfg.get("machines") or [["m4", cfg["machine"]]]
    for label, path in machines:
        _eval_machine(cfg, label, path, out, log)
    pack(out)


def _eval_machine(cfg, label, path, out, log):
    import pickle
    from isa_lab.full_custody import publish
    t0 = time.time()
    W = cfg["workers"]
    grid = I.make_grid()
    want = (cfg.get("machine_host_sha256") or {}).get(label)
    if want:  # a Beast-staged pickle read in place (too big for the payload), checked by hash
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        assert h.hexdigest() == want, f"staged machine {label} does not match its sha256"
    with open(path, "rb") as f:
        mm = pickle.load(f)
    m, dec = mm["m"], mm["dec"]
    log(f"[eval] machine {label}: {path}")
    res = {"config": cfg, "label": label, "machine": path}
    js = out / f"{label}.json"
    res["spf"] = R.verify_spf(m)
    import os
    # panels under each row distribution: "legacy" = the M4/M5a/M5b generator bit for bit (comparable to v2), "m5c" =
    # with boundary-state rows (scored with and without them) and the reg == rm cell
    for dist in [d for d in cfg.get("row_dists", ["legacy"]) if d]:  # empty: programs only (a resumed eval)
        os.environ["ISA_ROW_DIST"] = dist
        pre = "" if dist == "legacy" else dist + "/"
        for split, seed in (("iid", cfg["seed"] * 10 + 1), ("cells", cfg["seed"] * 10 + 2), ("literal", cfg["seed"] * 10 + 3)):
            d = I.generate(cfg["test_keys_per_form"], seed, split, grid, workers=W)
            d["store"] = L.learner_store(d)
            res[pre + split] = R.score_split(m, dec, d, pre + split, log)
            if "o_edge" in d and (d["o_edge"] != "").any():
                keep = np.flatnonzero(d["o_edge"] == "")
                sub = {k: (v[keep] if isinstance(v, np.ndarray) else [v[i] for i in keep]) for k, v in d.items() if k != "store"}
                sub["store"] = L.learner_store(sub)
                res[pre + split + "_no_edge"] = R.score_split(m, dec, sub, pre + split + "_no_edge", log)
                del sub
            if split == "iid" and not pre:
                res["scalar_vs_vector"] = R.check_agreement(m, dec, d)
                log(f"[eval] scalar vs vector agreement {res['scalar_vs_vector']['agree']}/{res['scalar_vs_vector']['rows']}")
            del d
        d = I.generate_stratified(1, cfg["seed"] * 10 + 4, grid, workers=W, rows_per_key=2)
        d["store"] = L.learner_store(d)
        res[pre + "strat_panel"] = R.score_split(m, dec, d, pre + "strat_panel", log)
        res[pre + "strat_panel"]["cells"] = I.cell_coverage(d["window"])
        del d
    js.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    publish([str(js), str(out / "log.txt")], f"m5-eval-{label}-panels")
    progs = {}
    for setname in cfg["program_sets"]:
        base = Path(setname)
        log(f"[eval] program set {base.name}")
        progs.update({f"{base.name}/{k}": v for k, v in R.run_programs(m, dec, base, cfg.get("prog_workers", W), log, out=out).items()})
        res["programs"] = progs
        js.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
        publish([str(js), str(out / "log.txt")], f"m5-eval-{label}-{base.name}")
    res["wall_s"] = round(time.time() - t0, 1)
    js.write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
