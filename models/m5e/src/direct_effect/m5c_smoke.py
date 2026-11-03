"""M5c pre-launch smoke C1 (opus48 review): the proof gate's rec-303 situation on the REAL fitted path.

`imul r32, r/m32, imm32` is fitted (learn -> share_taps -> src_alts -> FabricDecoder -> Scalar.apply) from rows that
ALL have CF = OF = 1 (every other row of the form is dropped, so the version space only ever sees signed overflow).
The query eax = -1 x 0x9e3779b1 has CF = OF = 0 (Unicorn). M5c must abstain there or predict CF = OF = 0; the negative
control strips the alternative evidence (tap_alts / tap_unobs / src_alts) from the same record and must guess CF = 1.

Heavy (a from-scratch learn): run on Beast (`scripts/m4.py launch-c1`), not on the laptop."""
import copy
import json
import os
import time
from pathlib import Path

import numpy as np

QUERY = bytes.fromhex("69c0b179379e") + bytes(9)  # imul eax, eax, 0x9e3779b1
FORM = "imul_r_rm32r_i32"


def _cat(*ds):
    return {k: (np.concatenate([d[k] for d in ds]) if isinstance(ds[0][k], np.ndarray) else sum((d[k] for d in ds), []))
            for k in ds[0]}


def _take(d, keep):
    return {k: (v[keep] if isinstance(v, np.ndarray) else [x for x, kk in zip(v, keep) if kk]) for k, v in d.items()}


def build(workers=4, log=print):
    os.environ.setdefault("ISA_ROW_DIST", "m5c")
    from direct_effect import m4_isa as I
    from direct_effect import m4_learn as M
    grid = I.make_grid()
    a = I.generate(2, 719, "train", grid, workers=workers)
    keys = []
    for rm in range(8):
        for reg in range(8):
            if rm == 4 or reg == 4:
                continue
            got = I.identify(bytes([0x69, 0xC0 | reg << 3 | rm, 1, 2, 3, 4]) + bytes(9))
            if got is not None and got[0] == FORM:
                keys.append(got + (64,))
    c = I.generate_targeted(keys, 13, grid, workers=workers, nearby_keys=0)
    d = _cat(a, c)
    fl = d["a_flags"]
    is_form = d["o_form"] == FORM
    keep = ~is_form | ((fl[:, 0] == 1) & (fl[:, 5] == 1))
    # the query key itself (eax, eax) is trained too, but only on overflowing rows
    d = _take(d, keep)
    n_form = int((d["o_form"] == FORM).sum())
    log(f"[c1] rows {len(d['rip'])}, {FORM} rows kept (all CF=OF=1) {n_form}, dropped {int((is_form & ~keep).sum())}")
    t0 = time.time()
    m, dec, info, pending = M.learn(d, log=log, workers=workers)
    log(f"[c1] learn {time.time() - t0:.0f}s, {len(m.records)} records")
    return m, dec, {"rows": int(len(d["rip"])), "form_rows_all_cf1": n_form, "records": len(m.records)}


def check(m, dec):
    from direct_effect import m4_learn as M
    from direct_effect.m4_scalar import Scalar
    ri, eff = dec.lookup(np.frombuffer(QUERY, np.uint8))
    out = {"record": ri, "decoded": eff is not None}
    if eff is None:
        out.update(abstain=True, why="no record (abstains)")
        return out
    sc = Scalar(m)
    regs = [0] * 16
    regs[0] = 0xFFFFFFFF
    st = {"rip": 0x401000, "regs": regs, "flags": [0] * 6}
    new, _, _, ab = sc.apply(eff, st, QUERY, None)
    out.update(src=eff["flags"].get("src"), abstain=bool(ab), CF=None if ab else new["flags"][0], OF=None if ab else new["flags"][5],
               evidence={k: (eff["flags"].get(k) is not None) for k in M.TAP_EVIDENCE})
    out["pass"] = bool(ab) or (new["flags"][0] == 0 and new["flags"][5] == 0)
    # negative control: the same record without the alternative evidence (M5a behaviour)
    e0 = copy.deepcopy(eff)
    for k in M.TAP_EVIDENCE:
        e0["flags"].pop(k, None)
    new0, _, _, ab0 = sc.apply(e0, st, QUERY, None)
    out["control"] = {"abstain": bool(ab0), "CF": None if ab0 else new0["flags"][0], "OF": None if ab0 else new0["flags"][5]}
    out["control_guesses_cf1"] = (not ab0) and new0["flags"][0] == 1
    return out


def main():
    out = Path("result")
    out.mkdir(exist_ok=True)
    cfg = json.loads(Path("study.json").read_text())
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    m, dec, info = build(cfg.get("workers", 4), log)
    res = {"config": cfg, "build": info, "check": check(m, dec)}
    log(f"[c1] {json.dumps(res['check'], default=str)}")
    (out / "c1.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "log.txt").write_text("\n".join(lines) + "\n")
    from isa_lab.full_custody import pack
    pack(out)


def coverage_diag_main():
    """M5c' diagnosis on Beast: regenerate the M5c training rows (same seeds, keys, worker count, m5c distribution),
    rerun field_coverage on the frozen M5c records with refutation blame pooled (= M5c, custody check against the
    pickle's allow sets) and kept local (M5c'), and list the blame events that closed low codes 0 and 5 for the
    reg-reg xor r/m64, r64 record (key 4c31d4) and its mod siblings."""
    import pickle
    from collections import Counter
    from direct_effect import m4_isa as I
    from direct_effect import m4_learn as L
    from direct_effect.m4_cegis import concat_data
    out = Path("result")
    out.mkdir(exist_ok=True)
    cfg = json.loads(Path("study.json").read_text())
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    os.environ["ISA_ROW_DIST"] = "m5c"
    W = cfg["gen_workers"]
    grid = I.make_grid()
    rand = I.generate(cfg["keys_per_form"], cfg["seed"], "train", grid, workers=W)
    strat = I.generate_stratified(cfg["keys_per_cell"], cfg["seed"], grid, workers=W)
    train = concat_data(rand, strat)
    del rand, strat
    for tf in cfg["replay_targets"]:
        for line in open(tf):
            t = json.loads(line)
            tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
            new = I.generate_targeted(tg, t["seed"], grid, workers=W, nearby_keys=t["nearby_keys"])
            if new is not None:
                train = concat_data(train, new)
    log(f"[covdiag] rows {len(train['rip'])} (M5c log: 301316)")
    train["store"] = L.learner_store(train)
    with open(cfg["machine"], "rb") as f:
        mm = pickle.load(f)
    m = mm["m"]
    ref_allow = [r.get("allow") for r in m.records]
    res = {"rows": int(len(train["rip"]))}
    for mode in ("pooled", "default"):
        recs = [dict(r) for r in m.records]
        tr = []
        kw = {"pool_refutation": True, "sib_pool": True, "refuted_good": False, "full_code": False} if mode == "pooled" else {}
        L.field_coverage(train, recs, log=log, m=m, maps=m.code_map, trace=tr, **kw)
        same = sum(1 for a, r in zip(ref_allow, recs) if a == r.get("allow"))
        log(f"[covdiag] {mode}: allow sets equal to the frozen M5c pickle on {same}/{len(recs)} records; blame events {Counter(k for *_, k in tr)}")
        res[mode] = {"allow_equal_to_m5c": same, "events": dict(Counter(k for *_, k in tr))}
        idx = [i for i, r in enumerate(recs) if bytes(r["key"]).hex() in ("4c31d4", "493143", "4c31ae", "493111")]
        res[mode]["xor_records"] = {bytes(recs[i]["key"]).hex(): {str(j): sorted(set(range(16)) - set(a)) for j, a in recs[i]["allow"].items()} for i in idx}
        log(f"[covdiag] {mode}: xor r/m64,r64 closed codes {res[mode]['xor_records']}")
        dec = L.FabricDecoder(recs, m.code_map)
        probe = {h: dec.lookup(np.frombuffer(bytes.fromhex(h) + bytes(15 - len(h) // 2), np.uint8))[1] is not None
                 for h in ("4831ed", "4831c0", "4831c8", "31c0", "31ed", "4889e5", "4883e4f0", "c704e101020304", "8b45fc")}
        res[mode]["decodes"] = probe
        log(f"[covdiag] {mode}: decodes {probe}")
        res.setdefault("gate_by_mode", {})
        gate_diag(train, m, recs, log, res)
        res["gate_by_mode"][mode] = res.pop("gate")
        if mode == "pooled":
            # the blame events whose low code (0 or 5) lands in the same pooled context as rec 4c31d4's rm field
            r0 = recs[idx[0]]
            f0 = r0["fields"][0]
            ctx0 = (f0[1], f0[2], r0["mask"][f0[0]], r0["key"][f0[0]] & r0["mask"][f0[0]])  # sib_pool context
            ev = []
            for ri, j, code, row, kind in tr:
                r = recs[ri]
                f = r["fields"][j]
                if (f[1], f[2], r["mask"][f[0]], r["key"][f[0]] & r["mask"][f[0]]) == ctx0 and (code & 7) in (0, 5):
                    w = bytes(train["window"][row][:6]).hex()
                    ev.append({"record": ri, "record_key": bytes(r["key"]).hex(), "field": j, "code": code, "kind": kind,
                               "row_form": str(train["o_form"][row]), "row_edge": str(train["o_edge"][row]) if "o_edge" in train else "",
                               "window": w})
            res["blame_ctx_4c31d4_rm"] = ev
            log(f"[covdiag] {len(ev)} blame events on codes 0/5 in rec 4c31d4's rm context; by (kind, record form, row form): "
                f"{Counter((e['kind'], e['record_key'][:4], e['row_form']) for e in ev).most_common(20)}")
    (out / "covdiag.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "log.txt").write_text("\n".join(lines) + "\n")
    from isa_lab.full_custody import pack
    pack(out)


def _rows_matching(train, r):
    """Training rows whose first len(key) bytes match the record's fixed bits."""
    k = len(r["key"])
    W = train["window"][:, :k]
    ok = np.ones(len(W), bool)
    for j in range(k):
        ok &= (W[:, j] & r["mask"][j]) == (r["key"][j] & r["mask"][j])
    return np.flatnonzero(ok)


def gate_diag(train, m, recs, log, res):
    """M5c' gate findings on the frozen records (after field_coverage with the given flags): rec 9024's SIB index code 4,
    the 8b45fc lookup, literal records whose run covers a SIB byte, and flag tap alternative counts."""
    from collections import Counter
    from direct_effect import m4_learn as L
    with np.errstate(over="ignore"):
        Lr = (train["a_rip"] - train["rip"]).astype(np.int64)
    out = {}
    # (1) SIB index 100 (= no index under REX.X = 0) in gate recs 9024 / 1603 / 7825 / 5617: the rows behind code 4
    for gate_rec, (kh, mh) in {"9024": ("4a0fbf4447", "f8ffffc700"), "1603": ("c6446c", "ffff00"), "7825": ("4a8ba43d", "f8ffc700"),
                               "5617": ("498b24b8", "feffc700")}.items():
        hit = [(i, r) for i, r in enumerate(recs) if bytes(r["key"]).hex() == kh and bytes(r["mask"]).hex() == mh]
        if not hit:
            out[gate_rec] = "no record"
            log(f"[gatediag] rec {gate_rec}: no record")
            continue
        i, r = hit[0]
        k = len(r["key"])
        fi = [j for j, f in r["fields"].items() if f[0] == k - 1 and f[1] == 3]
        if not fi:
            out[gate_rec] = {"fields": {str(j): list(f) for j, f in r["fields"].items()}}
            continue
        f = r["fields"][fi[0]]
        rows = _rows_matching(train, r)
        full = dict(r, allow={j: frozenset(range(16)) for j in r["fields"]})
        ok, _ = L.score(m, L.Decoder([full], m.code_map), train, rows)
        codes = L.field_code(train["window"][rows, :k], f)
        own = set(np.asarray(r["rows"]).tolist())
        sel = np.flatnonzero(codes == 4)
        lst = [{"own": int(rows[t]) in own, "exact": bool(ok[t]), "len_same": int(Lr[rows[t]]) == r["tmpl"]["L"],
                "edge": str(train["o_edge"][rows[t]]) if "o_edge" in train else "", "window": bytes(train["window"][rows[t]][:k + 1]).hex()}
               for t in sel]
        maps_idx = {kind: m.code_map.get(kind, {}).get(4) for kind in r["slot_kinds"]} if "slot_kinds" in r else None
        out[gate_rec] = {"index": i, "field": list(f), "allow": sorted(r["allow"][fi[0]]), "rows_matching": int(len(rows)),
                         "own_rows": len(own), "code4_rows": len(lst), "code4_summary": {str(kk): v for kk, v in Counter((e["own"], e["exact"], e["edge"]) for e in lst).items()},
                         "code4_examples": lst[:6], "slot_maps_code4": maps_idx}
        log(f"[gatediag] rec {gate_rec}: field {f} allow {sorted(r['allow'][fi[0]])}; rows {len(rows)} own {len(own)}; full code 4 rows "
            f"{len(lst)} by (own, exact, edge) {out[gate_rec]['code4_summary']}; slot maps for code 4 {maps_idx}; e.g. {lst[:3]}")
    # (2) why does 8b45fc (mov eax, [rbp-4]) not decode
    wb = bytes.fromhex("8b45fc") + bytes(12)
    cand = []
    for i, r in enumerate(recs):
        k = len(r["key"])
        if all((wb[j] & r["mask"][j]) == (r["key"][j] & r["mask"][j]) for j in range(k)):
            al = r.get("allow")
            codes = {j: int(L.field_code(np.frombuffer(wb[:k], np.uint8)[None, :], f)[0]) for j, f in r["fields"].items()}
            cand.append({"index": i, "key": bytes(r["key"]).hex(), "mask": bytes(r["mask"]).hex(), "codes": codes,
                         "closed": {str(j): (al is None or c not in al.get(j, ())) for j, c in codes.items()}})
    out["8b45fc"] = cand
    log(f"[gatediag] 8b45fc candidate records: {cand[:8]}")
    # (3) literal records whose run starts right after a ModRM with rm = 100, mod != 11 (a SIB byte read as data)
    sib_runs = []
    for i, r in enumerate(recs):
        k = len(r["key"])
        if r["fields"] or k < 2 or r["mask"] != b"\xff" * k:
            continue
        b = r["key"][k - 1]
        if (b & 7) == 4 and (b >> 6) != 3:
            rows = _rows_matching(train, r)
            vals = train["window"][rows, k]
            sib_runs.append({"index": i, "key": bytes(r["key"]).hex(), "runs": r["tmpl"].get("runs") if isinstance(r["tmpl"], dict) else None,
                             "L": r["tmpl"]["L"], "own_rows": len(r["rows"]), "rows_matching": int(len(rows)),
                             "distinct_sib": int(len(set(vals.tolist()))), "lens": dict(Counter(Lr[rows].tolist()))})
    out["literal_sib_runs"] = sib_runs
    log(f"[gatediag] literal records over a SIB byte: {len(sib_runs)}; e.g. {sib_runs[:6]}")
    # (4) flag tap alternatives: list sizes and overflow counts by flag
    over = Counter()
    sizes = Counter()
    for r in recs:
        fl = (r["tmpl"].get("flags") or {}) if isinstance(r["tmpl"], dict) else {}
        for f, a in (fl.get("tap_alts") or {}).items():
            if a == "overflow":
                over[f] += 1
            else:
                sizes[min(len(a), 64) // 8 * 8] += 1
    out["tap_alts_overflow"] = dict(over)
    out["tap_alts_sizes"] = dict(sizes)
    log(f"[gatediag] tap_alts overflow by flag {dict(over)}; list sizes (bucket of 8) {dict(sorted(sizes.items()))}")
    res["gate"] = out


def _wrong_split(m, dec, d, rows):
    """Wrong rows split into flag-only (registers, RIP, memory and trap right) and the rest; plus wrong forms/records."""
    import json as _j
    from collections import Counter, defaultdict
    from direct_effect import m4_learn as L
    groups = defaultdict(list)
    for i in rows.tolist():
        ri, eff = dec.lookup(d["window"][i])
        groups[(ri, _j.dumps(eff, sort_keys=True))].append(i)
    flag_only, other, flags_by, forms, recs = 0, 0, Counter(), Counter(), Counter()
    for (ri, ej), rr in groups.items():
        eff = _j.loads(ej)
        x = L.Ctx(m, d, np.array(rr), eff["k"], eff["runs"])
        regs, rip, flags, writes, trap, ab = L.apply_effect(m, eff, x)
        g = x.g
        nonflag = (regs == g["a_regs"]).all(1) & (rip == g["a_rip"]) & (g["trap"] == trap)
        for t, i in enumerate(rr):
            if nonflag[t] and not (flags[t] == g["a_flags"][t]).all():
                flag_only += 1
                for k, f in enumerate(L.FLAGS):
                    if flags[t][k] != g["a_flags"][t][k]:
                        flags_by[f] += 1
            else:
                other += 1
            forms[str(d["o_form"][i])] += 1
            recs[ri] += 1
    return {"flag_only": flag_only, "other": other, "flags": dict(flags_by), "forms": forms.most_common(15), "records": recs.most_common(15)}


def stratdiag_main():
    """opus48 smoke before M5d: why did M5c' strat-panel wrong rise (0.94% M5c -> 1.46%)? On the frozen M5c' records:
    A as frozen; B tap/src alternative lists > 48 set to "overflow" (= M5c's cap); C field coverage with M5c's pooled
    rules (pool_refutation + sib_pool); D pool_refutation only; E sib_pool only. Each scored on the strat (seed 7194) and
    iid (7191) panels under the M5c row distribution, wrong rows split flag-only vs the rest."""
    import copy
    import hashlib
    import pickle
    from direct_effect import m4_isa as I
    from direct_effect import m4_learn as L
    from direct_effect.m4_cegis import concat_data
    out = Path("result")
    out.mkdir(exist_ok=True)
    cfg = json.loads(Path("study.json").read_text())
    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)
    os.environ["ISA_ROW_DIST"] = "m5c"
    W = cfg["gen_workers"]
    grid = I.make_grid()
    h = hashlib.sha256(open(cfg["machine"], "rb").read()).hexdigest()
    assert h == cfg["machine_sha256"], h
    with open(cfg["machine"], "rb") as f:
        mm = pickle.load(f)
    m0 = mm["m"]
    if not cfg.get("frozen_only"):
        rand = I.generate(cfg["keys_per_form"], cfg["seed"], "train", grid, workers=W)
        strat = I.generate_stratified(cfg["keys_per_cell"], cfg["seed"], grid, workers=W)
        train = concat_data(rand, strat)
        del rand, strat
        for tf in cfg["replay_targets"]:
            for line in open(tf):
                t = json.loads(line)
                tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
                new = I.generate_targeted(tg, t["seed"], grid, workers=W, nearby_keys=t["nearby_keys"])
                if new is not None:
                    train = concat_data(train, new)
        train["store"] = L.learner_store(train)
    panels = {}
    panels["strat"] = I.generate_stratified(1, cfg["seed"] * 10 + 4, grid, workers=W, rows_per_key=2)
    panels["iid"] = I.generate(cfg["test_keys_per_form"], cfg["seed"] * 10 + 1, "iid", grid, workers=W)
    for p in panels.values():
        p["store"] = L.learner_store(p)

    def cap48(recs):
        recs = copy.deepcopy(recs)
        for r in recs:
            fl = r["tmpl"].get("flags") or {}
            for f, a in list((fl.get("tap_alts") or {}).items()):
                if a != "overflow" and len(a) > 48:
                    fl["tap_alts"][f] = "overflow"
            for alt in fl.get("src_alts") or []:
                for f, c in list(alt[1].items()):
                    if c != "overflow" and len(c) > 48:
                        alt[1][f] = "overflow"
        return recs

    variants = {"A_m5cp": lambda: [dict(r) for r in m0.records]}
    for name, kw in () if cfg.get("frozen_only") else (("H_default", {}), ("F_low_bit_pooling", {"full_code": False}),
                     ("C_m5c", {"pool_refutation": True, "sib_pool": True, "refuted_good": False, "full_code": False})):
        def mk(kw=kw):
            recs = [dict(r) for r in m0.records]
            return L.field_coverage(train, recs, log=log, m=m0, maps=m0.code_map, **kw)
        variants[name] = mk
    res = {}
    for name, mkr in variants.items():
        recs = mkr()
        dec = L.FabricDecoder(recs, m0.code_map)
        res[name] = {}
        probe = {hx: dec.lookup(np.frombuffer(bytes.fromhex(hx) + bytes(15 - len(hx) // 2), np.uint8))[1] is not None
                 for hx in ("4831ed", "4831c0", "4831c8", "31ed", "8b45fc", "480fbf44240e", "4a0fbf44240e", "88040c",
                           "c644246f0a", "488b0424", "4a8b0424", "498b2424")}
        ix = {}
        for gate_rec, (kh, mh) in {"9024": ("4a0fbf4447", "f8ffffc700"), "1603": ("c6446c", "ffff00"), "7825": ("4a8ba43d", "f8ffc700"),
                                   "5617": ("498b24b8", "feffc700")}.items():
            rr = [r for r in recs if bytes(r["key"]).hex() == kh and bytes(r["mask"]).hex() == mh]
            if rr and rr[0].get("allow"):
                js = [j for j, f in rr[0]["fields"].items() if f[0] == len(rr[0]["key"]) - 1 and f[1] == 3]
                ix[gate_rec] = sorted(rr[0]["allow"][js[0]]) if js else "no index field"
            else:
                ix[gate_rec] = "no record" if not rr else "no allow"
        res[name]["decodes"], res[name]["sib_index_allow"] = probe, ix
        log(f"[stratdiag] {name}: decodes {probe}; SIB index allow by gate record {ix}")
        for pn, p in panels.items():
            ok, ab = L.score(m0, dec, p)
            wrong = np.flatnonzero(~ok & ~ab)
            sp = _wrong_split(m0, dec, p, wrong)
            res[name][pn] = {"n": int(len(ok)), "exact": float(ok.mean()), "abstain": float(ab.mean()), "wrong": float(len(wrong) / len(ok)), **sp}
            log(f"[stratdiag] {name} {pn}: exact {ok.mean():.4f} abstain {ab.mean():.4f} wrong {len(wrong) / len(ok):.4f} "
                f"(flag-only {sp['flag_only']}, other {sp['other']}, flags {sp['flags']}); forms {sp['forms'][:8]}")
        (out / "stratdiag.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "log.txt").write_text("\n".join(lines) + "\n")
    from isa_lab.full_custody import pack
    pack(out)


def abstain_reasons(m, eff, x):
    """Per row, which channel abstains: each flag's unobserved tap entry, its tap alternatives, each source
    alternative's flags; 'other' = every abstain the flags do not explain (registers, RIP, predicates, memory)."""
    from direct_effect import m4_learn as L
    regs, rip, flags, writes, trap, abst = L.apply_effect(m, eff, x)
    g, n = x.g, x.n
    ch = {}
    fl = eff.get("flags") or {}
    flag_any = np.zeros(n, bool)
    if fl.get("src") is not None:
        tgt = fl.get("src_old")
        old = None
        if tgt is not None and tgt[0] == "reg":
            old = L.mask(g["regs"][:, tgt[1]], tgt[2])
        elif tgt is not None and tgt[0] == "mem":
            old = x.store.read(L.eval_addr(m, tgt[1], x), tgt[2])
        _, sig = L.ev(m, fl["src"], x, old, want_sig=True)
        for k, f in enumerate(L.FLAGS):
            o = g["flags"][:, k].astype(np.int64)
            tap = fl["taps"][f]
            un = (fl.get("tap_unobs") or {}).get(f)
            if un and tap[0]:
                ch[f"tap_unobs:{f}"] = np.isin(L._tap_index(tap[0], sig, o), un)
            alts = (fl.get("tap_alts") or {}).get(f, [])
            if alts == "overflow":
                ch[f"tap_alts_overflow:{f}"] = np.ones(n, bool)
            else:
                a = np.zeros(n, bool)
                for t in alts:
                    a |= L._alt_abstain(t, sig, o, flags[:, k])
                ch[f"tap_alts:{f}"] = a
        for j, (alt, cands) in enumerate(fl.get("src_alts") or []):
            _, sig2 = L.ev(m, alt, x, old, want_sig=True)
            for k, f in enumerate(L.FLAGS):
                o = g["flags"][:, k].astype(np.int64)
                if cands[f] == "overflow":
                    ch[f"src_alt{j}_overflow:{f}"] = np.ones(n, bool)
                    continue
                a = np.zeros(n, bool)
                for t in cands[f]:
                    a |= L._alt_abstain(t, sig2, o, flags[:, k])
                ch[f"src_alt{j}:{f}"] = a
        for v in ch.values():
            flag_any |= v
    ch["other"] = abst & ~flag_any
    return {k: int(v.sum()) for k, v in ch.items() if v.any()}, abst


def abstaindiag_main():
    """Why do the proof-gate constant-ZF records (M5d v2) still abstain on a zero result after the src_alts fix
    (rebuild job-faaab4ec7bd4)? On a frozen machine: fresh random-state Unicorn rows (M5c distribution) for the gate
    keys; on the true-zero-result rows, the abstain channel per row; plus the record's ZF evidence."""
    import hashlib
    import pickle
    from direct_effect import m4_isa as I
    from direct_effect import m4_learn as L
    from direct_effect.m4_cegis import GATE_ZF_WINDOWS
    out = Path("result")
    out.mkdir(exist_ok=True)
    cfg = json.loads(Path("study.json").read_text())
    os.environ["ISA_ROW_DIST"] = "m5c"
    h = hashlib.sha256(open(cfg["machine"], "rb").read()).hexdigest()
    assert h == cfg["machine_sha256"], h
    with open(cfg["machine"], "rb") as f:
        mm = pickle.load(f)
    m, dec = mm["m"], mm["dec"]
    grid = I.make_grid()
    targets, rid_of = [], {}
    for rid, w in GATE_ZF_WINDOWS.items():
        got = I.identify(bytes.fromhex(w) + bytes(I.WINDOW - len(w) // 2))
        name, key, regs, mem, cc = got
        targets.append((name, key, regs, mem, cc, 4000))
        rid_of[key.hex()] = rid
    d = I.generate_targeted(targets, cfg["seed"] * 1000 + 910, grid, workers=cfg["gen_workers"], nearby_keys=0)
    d["store"] = L.learner_store(d)
    keys = np.array([str(k) for k in d["o_key"]])
    zf = d["a_flags"][:, I.FLAGS.index("ZF")] == 1
    res = {"config": cfg, "keys": {}}
    for kh, rid in rid_of.items():
        s = keys == kh
        entry = {"gate_record": rid}
        for part, sel in (("zero", s & zf), ("nonzero", s & ~zf)):
            rows = np.flatnonzero(sel)
            if len(rows) == 0:
                continue
            ri, eff = dec.lookup(d["window"][rows[0]])
            x = L.Ctx(m, d, rows, eff["k"], eff["runs"])
            reasons, abst = abstain_reasons(m, eff, x)
            ok, ab = L.check_rows(m, eff, x)
            entry[part] = {"n": int(len(rows)), "exact": int(ok.sum()), "abstain": int(ab.sum()), "reasons": reasons}
            fl = eff.get("flags") or {}
            entry["record"] = ri
            entry["ZF"] = {"src": fl.get("src"), "tap": fl.get("taps", {}).get("ZF"), "tap_unobs": (fl.get("tap_unobs") or {}).get("ZF"),
                           "tap_alts": ((fl.get("tap_alts") or {}).get("ZF") or [])[:8] if (fl.get("tap_alts") or {}).get("ZF") != "overflow" else "overflow",
                           "src_alts": [[a, (c.get("ZF") or [])[:4] if c.get("ZF") != "overflow" else "overflow"] for a, c in (fl.get("src_alts") or [])][:6],
                           "n_record_rows": int(len(m.records[ri]["rows"])) if ri is not None and ri < len(m.records) else None}
        res["keys"][kh] = entry
        print(f"[abstaindiag] rec {rid} {kh} record {entry.get('record')}: zero {entry.get('zero')} | nonzero "
              f"{ {k: v for k, v in (entry.get('nonzero') or {}).items() if k != 'reasons'} } | ZF {json.dumps(entry.get('ZF'))[:600]}", flush=True)
    (out / "abstaindiag.json").write_text(json.dumps(res, indent=1, default=str))
