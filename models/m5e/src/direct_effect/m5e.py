"""M5e: the M5d-final v3 machine rebuilt under the M5e rules, diagnosed, then improved by the dev-guided loop.

M5e rules (m4_learn): flag-source alternatives that no table of their op kind's shared rule explains are dropped
(SRC_ALT_NONE_CONSISTENT = "drop"), and 8-bit register fields without a fixed REX byte keep codes 4-7 closed
(BYTE_REG_GUARD). Everything else is the M5d runtime.

worker_diag (one lease): regenerate v3's training rows bit for bit (same seeds, keys and worker counts as the M5d
leases), custody-check v3 on them, then
  1. field coverage diagnosis: for each stop window, the records whose fixed bits match and, for every closed field
     code, the pooled good/blame evidence of its context and the blame events that closed it;
  2. fit diagnosis: fresh rows for each stop window's key (held-out cells are reported, never sampled) fitted alone;
  3. the M5e rebuild (share_taps + merge + coverage under the M5e rules), zero-result sweeps before/after with the
     abstain channel of every remaining zero-result abstention, and every dev program (dev46 + isa-bench c_apps) on
     the rebuilt machine, in this process (no fork pool: mrun counts a forked child's inherited pages again).
The rebuilt machine is saved (result/machine_rebuilt.pkl)."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import m4_isa as I
from . import m4_learn as L
from . import m4_run as R
from .m4_cegis import GATE_ZF_WINDOWS, append_rows, concat_data, rebuild, sweep_data, sweep_score


def regen_train(cfg, grid, log):
    """The v3 training rows, regenerated exactly as worker_devcegis does on resume."""
    os.environ["ISA_ROW_DIST"] = "m5c"
    gen_w = cfg["base_workers"]
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
    for tf in cfg.get("resume_targets", []):
        for line in open(tf):
            t = json.loads(line)
            tg = [(a, bytes.fromhex(b), c, d, e, n) for a, b, c, d, e, n in t["targets"]]
            if tg:
                append_rows(train, I.generate_targeted(tg, t["seed"], grid, workers=t["workers"], nearby_keys=t["nearby_keys"]))
    log(f"[m5e] training rows regenerated: {len(train['rip'])}")
    return train


def custody(m, dec, data, n=20000):
    """m4_cegis.custody without the asserts (the rebuilt machine is reported, not gated, here)."""
    rr = np.unique(np.concatenate([r["rows"] for r in m.records]))
    sample = np.random.default_rng(0).choice(rr, size=min(n, len(rr)), replace=False)
    ok, ab = L.score(m, dec, data, rows=sample)
    return {"n": int(len(sample)), "exact": float(ok.mean()), "abstain": float(ab.mean()), "wrong": float((~ok & ~ab).mean())}


def load_machine(path, sha256):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    assert h.hexdigest() == sha256, f"{path}: sha256 {h.hexdigest()} != {sha256}"
    with open(path, "rb") as f:
        mm = pickle.load(f)
    return mm["m"], mm["dec"]


def matching_records(records, wb):
    out = []
    for i, r in enumerate(records):
        k = len(r["key"])
        if k <= len(wb) and all((wb[j] & r["mask"][j]) == (r["key"][j] & r["mask"][j]) for j in range(k)):
            out.append(i)
    return out


def coverage_diag(train, records, maps, m, windows, log):
    """Closed field codes on each window's matching records, with the pooled evidence of their context."""
    trace, diag = [], {}
    recs = [dict(r) for r in records]
    L.field_coverage(train, recs, log=log, m=m, maps=maps, trace=trace, diag=diag)
    same = sum(1 for a, r in zip(records, recs) if a.get("allow") == r.get("allow"))
    W = train["window"]
    ctx, good_p, bad_p = diag["ctx"], diag["good_p"], diag["bad_p"]
    by_ctx = {}
    for ri, j, code, row, kind in trace:
        r = recs[ri]
        by_ctx.setdefault(ctx(r, r["fields"][j]), []).append((ri, j, code, row, kind))
    out = {"allow_equal_to_frozen": same, "records": len(recs), "windows": {}}
    for name, hx in windows.items():
        wb = bytes.fromhex(hx)
        rows = []
        for i in matching_records(recs, wb):
            r = recs[i]
            keyarr = np.frombuffer(wb[:len(r["key"])], np.uint8)[None, :]
            fields = []
            for j, f in sorted(r["fields"].items()):
                c = int(L.field_code(keyarr, f)[0])
                al = (r.get("allow") or {}).get(j)
                ok = al is not None and c in al
                info = {"slot": j, "kind": r["slot_kinds"][j], "field": list(f[:3]) + [list(f[3]) if f[3] else None], "code": c, "open": ok}
                if not ok:
                    cx = ctx(r, f)
                    lm = (1 << f[2]) - 1
                    info["ctx"] = str(cx)
                    info["pooled_good"] = sorted(good_p.get(cx, ()))
                    info["pooled_bad_low"] = sorted(bad_p.get(cx, ()))
                    info["local"] = [sorted(s) for s in diag["local"].get(i, {}).get(j, (set(), set(), set()))]
                    ev = [e for e in by_ctx.get(cx, []) if (e[2] & lm) == (c & lm)]
                    info["blame_events"] = len(ev)
                    info["blame_by"] = Counter((bytes(recs[e[0]]["key"]).hex(), e[4], str(train["o_form"][e[3]]) if "o_form" in train else "")
                                               for e in ev).most_common(12)
                    info["blame_rows"] = [bytes(W[e[3]][:6]).hex() for e in ev[:8]]
                fields.append(info)
            rows.append({"record": i, "key": bytes(r["key"]).hex(), "mask": bytes(r["mask"]).hex(), "L": r["tmpl"]["L"],
                         "members": r["members"], "fields": fields})
        out["windows"][name] = {"window": hx, "candidates": rows}
        log(f"[m5e] coverage {name} {hx[:16]}: " + "; ".join(
            f"rec {c['record']} {c['key']}/{c['mask']} " + ",".join(f"{x['kind']}{x['code']}:{'ok' if x['open'] else 'CLOSED(' + str(x.get('blame_events')) + ')'}" for x in c["fields"])
            for c in rows))
    return out


def fit_diag(m, windows, grid, seed, log, n_rows=24, nearby=2):
    """Fresh rows for each window's key fitted alone (the machine's tables and adder): which key length and layout fit,
    or why nothing does. A window whose (form, roles) hits a held-out cell is reported and not sampled."""
    out = {}
    for name, hx in windows.items():
        got = I.identify(bytes.fromhex(hx))
        if got is None:
            out[name] = {"ident": None}
            log(f"[m5e] fit {name}: not in the catalog")
            continue
        fname, key, regs, mem, cc = got
        held = I.hits_cell(grid, I.FORMS[fname], regs)
        rec = {"form": fname, "key": key.hex(), "regs": regs, "held_cell": held}
        d = I.generate_targeted([(fname, key, regs, mem, cc, n_rows)], seed, grid, workers=1, nearby_keys=nearby)
        if d is None:
            out[name] = rec
            continue
        d["store"] = L.learner_store(d)
        exact = np.array([bytes(w[:len(key)]) == key for w in d["window"]])
        rec["rows"], rec["exact_key_rows"] = int(len(d["rip"])), int(exact.sum())
        recs, pending, st = L.fit_trie(m, d, allow_flag_search=True, log=lambda s: None, workers=1)
        rec["fitted"] = [{"key": bytes(r["key"]).hex(), "runs": r["effect"]["runs"], "L": r["effect"]["L"], "rows": int(len(r["rows"])),
                          "writes": json.dumps(r["effect"]["writes"])[:240]} for r in recs]
        rec["pending_exact_key_rows"] = int((pending & exact).sum())
        rec["fail_stats"] = dict(st.most_common(10))
        out[name] = rec
        log(f"[m5e] fit {name}: {fname} key {key.hex()} held {held}; rows {rec['rows']} (exact key {rec['exact_key_rows']}), "
            f"fitted {[(x['key'], x['runs']) for x in rec['fitted']]}, pending exact {rec['pending_exact_key_rows']}, fails {rec['fail_stats']}")
    return out


def zero_channels(m, dec, sw):
    """Abstain channel counts on the zero-result rows of each swept key."""
    from .m5c_smoke import abstain_reasons
    d, sel, keys = sw["data"], sw["sel"], sw["keys"]
    out = {}
    for k in sorted(sw["kind"]):
        rows = np.flatnonzero((keys == k) & sel)
        if not len(rows):
            continue
        ri, eff = dec.lookup(d["window"][rows[0]])
        if eff is None:
            out[k] = {"record": None}
            continue
        x = L.Ctx(m, d, rows, eff["k"], eff["runs"])
        try:
            ch, ab = abstain_reasons(m, eff, x)
        except Exception as e:  # noqa: BLE001
            ch, ab = {"error": repr(e)[:120]}, np.zeros(len(rows), bool)
        out[k] = {"record": ri, "n": int(len(rows)), "abstain": int(np.sum(ab)), "channels": ch}
    return out


def run_programs_inproc(m, dec, base, names, log):
    from .m4_scalar import Scalar
    sc = Scalar(m)
    res = {}
    for n in names:
        p = Path(base) / n
        tr = np.load(p / "trace.npz")["steps"]
        t = time.time()
        r = R.run_program(m, dec, p / "mem.bin", tr, scalar=sc)
        nat = json.loads((p / "native.json").read_text())
        r["done"] = bool(r["status"] == "exit" and r["exit_code"] == nat.get("native_exit") and r["stdout"] == nat.get("native_stdout"))
        r["wall_s"] = round(time.time() - t, 1)
        res[n] = r
        log(f"[m5e] program {n}: {r['status']} steps {r['steps_run']}/{r['native_steps']} done {r['done']} "
            f"exact {r.get('exact_steps')} wall {r['wall_s']}s")
    return res


def worker_diag():
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack, publish
    t0 = time.time()
    grid = I.make_grid()
    res = {"config": cfg}
    train = regen_train(cfg, grid, log)
    m, dec = load_machine(cfg["machine"], cfg["machine_sha256"])
    res["custody"] = custody(m, dec, train)
    log(f"[m5e] v3 custody {res['custody']} t={time.time() - t0:.0f}s")
    windows = cfg["windows"]
    res["coverage"] = coverage_diag(train, m.records, m.code_map, m, windows, log)
    log(f"[m5e] coverage rerun: allow sets equal to the frozen machine on {res['coverage']['allow_equal_to_frozen']}/"
        f"{res['coverage']['records']} records (M5e byte-register guard on) t={time.time() - t0:.0f}s")
    res["fit"] = fit_diag(m, windows, grid, cfg["seed"] * 1000 + 990, log)
    (out / "m5e_diag.json").write_text(json.dumps(res, indent=1, default=str))
    publish([str(out / "m5e_diag.json"), str(out / "log.txt")], "m5e-diag-part1")

    gate = dict(GATE_ZF_WINDOWS)
    gate.update({f"stop:{n}": w for n, w in cfg.get("zero_windows", {}).items()})
    sw = sweep_data({}, cfg["seed"] * 1000 + 900, grid, cfg["workers"], gate=gate)
    res["sweep_v3"] = sweep_score(m, dec, sw, log, "v3")
    res["zero_channels_v3"] = zero_channels(m, dec, sw)
    zf0 = L.const_zf_alt_records(m.literal_records)
    L.share_taps(m, train, m.literal_records)
    res["src_alt_fallbacks"] = getattr(L.share_taps, "last_fallbacks", {})
    res["const_zf_alt_records"] = {"before": len(zf0), "after": len(L.const_zf_alt_records(m.literal_records))}
    log(f"[m5e] src_alt flags (M5e drop rule): {res['src_alt_fallbacks']}; constant-ZF src_alt records {res['const_zf_alt_records']}")
    dec = rebuild(m, train, log)
    log(f"[m5e] rebuilt: {len(m.records)} records t={time.time() - t0:.0f}s")
    res["custody_rebuilt"] = custody(m, dec, train)
    res["sweep_rebuilt"] = sweep_score(m, dec, sw, log, "rebuilt")
    res["zero_channels_rebuilt"] = zero_channels(m, dec, sw)
    for k, v in res["zero_channels_rebuilt"].items():
        if v.get("abstain"):
            log(f"[m5e] rebuilt zero-result abstain {k}: {v}")
    with open(out / "machine_rebuilt.pkl", "wb") as f:
        pickle.dump({"m": m, "dec": dec}, f)
    del sw, train
    gc.collect()
    publish([str(out / "machine_rebuilt.pkl")], "m5e-rebuilt-machine")
    base = Path(cfg["native_dir"])
    names = sorted(p.name for p in base.glob("*") if p.is_dir())
    res["programs_rebuilt"] = run_programs_inproc(m, dec, base, names, log)
    res["programs_done_rebuilt"] = sum(r["done"] for r in res["programs_rebuilt"].values())
    log(f"[m5e] rebuilt machine: programs done {res['programs_done_rebuilt']}/{len(names)} t={time.time() - t0:.0f}s")
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m5e_diag.json").write_text(json.dumps(res, indent=1, default=str))
    pack(out)


def _bind_record(r, maps, wb):
    k = len(r["key"])
    keyarr = np.frombuffer(bytes(wb[:k]), np.uint8)[None, :]
    vals = []
    for j, kind in enumerate(r["slot_kinds"]):
        if r["consts"][j] is not None:
            vals.append(r["consts"][j])
            continue
        v = maps.get(kind, {}).get(int(L.field_code(keyarr, r["fields"][j])[0]))
        if v is None:
            return None
        vals.append(v)
    return L.bind(r["tmpl"], vals)


def _consistent(t, sig, old, truth):
    return bool(np.array_equal(L.apply_tap(t, sig, old), truth))


def _seen(t, sig, old):
    return set(np.unique(L._tap_index(t[0], sig, old)).tolist()) if t[0] else {0}


def refine_flags(m, base, data, log):
    """M5e #3 (NOTES-m5): flag evidence from fresh Unicorn rows of the record a stop window routes to. Rows on which
    the record's registers and RIP are exact refute every tap alternative and every flag-source alternative table
    that predicts a flag differently, and mark the table entries they reach as observed; an alternative with no table
    left for some flag is dropped. A record whose chosen taps or registers disagree with a row is left alone (it is
    reported: the record itself is refuted there). Edits go to the record templates (shared by every member)."""
    stats = Counter()
    by_key = {}
    for i, w in enumerate(data["window"]):
        by_key.setdefault(bytes(w[:15]), []).append(i)
    groups = defaultdict(list)
    for wb, rows in by_key.items():
        ri = base._match(wb)
        if ri is not None:
            groups[(ri, wb[:len(base.records[ri]["key"])])] += rows
    for (ri, kb), rows in sorted(groups.items(), key=lambda kv: kv[0][0]):
        r = base.records[ri]
        eff = _bind_record(r, base.maps, kb)
        fl = (eff or {}).get("flags") or {}
        if eff is None or fl.get("src") is None:
            stats["no flag source"] += 1
            continue
        x = L.Ctx(m, data, np.array(rows), eff["k"], eff["runs"])
        try:
            regs, rip, flags, writes, trap, abst = L.apply_effect(m, eff, x)
        except Exception:  # noqa: BLE001
            stats["effect fails"] += 1
            continue
        g = x.g
        good = (regs == g["a_regs"]).all(1) & (rip == g["a_rip"])
        if not good.all():
            stats["registers refuted"] += 1
            log(f"[refine] rec {ri} {kb.hex()}: registers/RIP wrong on {int((~good).sum())}/{len(rows)} rows (left alone)")
            continue
        tgt = fl.get("src_old")
        old = None
        if tgt is not None and tgt[0] == "reg":
            old = L.mask(g["regs"][:, tgt[1]], tgt[2])
        elif tgt is not None and tgt[0] == "mem":
            old = x.store.read(L.eval_addr(m, tgt[1], x), tgt[2])
        _, sig = L.ev(m, fl["src"], x, old, want_sig=True)
        before, truth = g["flags"].astype(np.int64), g["a_flags"].astype(np.int64)
        T = r["tmpl"]["flags"]
        refuted = [f for k, f in enumerate(L.FLAGS) if not _consistent(fl["taps"][f], sig, before[:, k], truth[:, k])]
        swap = {}
        for f in refuted:
            # the chosen table is refuted; a tap alternative the record's own rows kept and these rows do not refute
            # becomes the choice (the others stay alternatives); none left = the record stays as it is (abstains)
            k = L.FLAGS.index(f)
            alts = (fl.get("tap_alts") or {}).get(f)
            ok = [t for t in alts if _consistent(t, sig, before[:, k], truth[:, k])] if isinstance(alts, list) else []
            if not ok:
                break
            swap[f] = ok
        if refuted and len(swap) < len(refuted):
            # the chosen flag source is refuted: a source alternative (e.g. the other signedness of a 'mul lo') whose
            # tables explain these rows for every flag becomes the source
            found = None
            for j, (alt, cands) in enumerate(fl.get("src_alts") or []):
                try:
                    _, sig2 = L.ev(m, alt, x, old, want_sig=True)
                except Exception:  # noqa: BLE001
                    continue
                ok = {}
                for k, f in enumerate(L.FLAGS):
                    if cands[f] == "overflow":
                        break
                    c = [t for t in cands[f] if _consistent(t, sig2, before[:, k], truth[:, k])]
                    if not c:
                        break
                    ok[f] = c
                else:
                    found = (j, sig2, ok)
                    break
            if found is None:
                stats["chosen taps refuted, no alternative"] += 1
                log(f"[refine] rec {ri} {kb.hex()}: chosen taps {refuted} wrong on these rows, no consistent alternative (left alone)")
                continue
            j, sig, ok = found
            T["src"] = T["src_alts"][j][0]
            fl["src"] = fl["src_alts"][j][0]
            T["src_alts"] = [a for jj, a in enumerate(T["src_alts"]) if jj != j]
            fl["src_alts"] = [a for jj, a in enumerate(fl["src_alts"]) if jj != j]
            swap = ok
            stats["flag source replaced by an alternative"] += 1
            log(f"[refine] rec {ri} {kb.hex()}: flag source refuted ({refuted}); source alternative {j} explains every flag")
        for f, ok in swap.items():
            t = ok[0]
            T["taps"][f] = [t[0], t[1], 0]
            T.setdefault("tap_unobs", {})[f] = list(t[2]) if len(t) > 2 and t[2] else []
            T.setdefault("tap_alts", {})[f] = ok[1:]
            fl["taps"][f] = T["taps"][f]
            fl.setdefault("tap_unobs", {})[f] = T["tap_unobs"][f]
            fl.setdefault("tap_alts", {})[f] = T["tap_alts"][f]
            stats["chosen taps replaced by an alternative"] += 1
            log(f"[refine] rec {ri} {kb.hex()}: {f} tap {fl['taps'][f][:2]} chosen from {len(ok)} consistent alternatives")
        for k, f in enumerate(L.FLAGS):
            o, tv = before[:, k], truth[:, k]
            un = (T.get("tap_unobs") or {}).get(f)
            if un and fl["taps"][f][0]:
                seen = _seen(fl["taps"][f], sig, o)
                T["tap_unobs"][f] = [e for e in un if e not in seen]
                stats["tap entries observed"] += len(un) - len(T["tap_unobs"][f])
            alts = (fl.get("tap_alts") or {}).get(f)
            if isinstance(alts, list) and alts:
                keep = []
                for t in alts:
                    if not _consistent(t, sig, o, tv):
                        stats["tap alternatives refuted"] += 1
                        continue
                    if len(t) > 2 and t[2] and t[0]:
                        t = [t[0], t[1], [e for e in t[2] if e not in _seen(t, sig, o)]] + list(t[3:])
                    keep.append(t)
                T["tap_alts"][f] = keep
        src_alts = fl.get("src_alts") or []
        if src_alts:
            out = []
            for j, (alt, cands) in enumerate(src_alts):
                try:
                    _, sig2 = L.ev(m, alt, x, old, want_sig=True)
                except Exception:  # noqa: BLE001
                    out.append(T["src_alts"][j])
                    continue
                new, dead = {}, False
                for k, f in enumerate(L.FLAGS):
                    if cands[f] == "overflow":
                        new[f] = "overflow"
                        continue
                    kept = []
                    for t in cands[f]:
                        if _consistent(t, sig2, before[:, k], truth[:, k]):
                            if len(t) > 2 and t[2] and t[0]:
                                t = [t[0], t[1], [e for e in t[2] if e not in _seen(t, sig2, before[:, k])]] + list(t[3:])
                            kept.append(t)
                    stats["src_alt tables refuted"] += len(cands[f]) - len(kept)
                    if not kept:
                        dead = True
                        break
                    new[f] = kept
                if dead:
                    stats["src_alts dropped"] += 1
                    continue
                out.append([T["src_alts"][j][0], new])
            T["src_alts"] = out
        stats["records refined"] += 1
    log(f"[refine] {dict(stats)}")
    return dict(stats)


def fit_stops(m, windows, grid, seed, log, n_rows=24, nearby=2, n_check=64):
    """M4/M5d dev-guided fitting without a rebuild: fresh Unicorn rows for the key of every no-record stop window in the
    catalog (held-out cells are never sampled), fitted by the trie alone; a fitted record is kept only if it is exact
    (never wrong) on a second, independent batch of rows for the same key. Kept records are literal decoder records."""
    out, report = [], {}
    for name, hx in sorted(windows.items()):
        got = I.identify(bytes.fromhex(hx))
        if got is None:
            report[name] = "not in the catalog"
            continue
        fname, key, regs, mem, cc = got
        if I.hits_cell(grid, I.FORMS[fname], regs):
            report[name] = f"{fname} {key.hex()}: held-out cell (not sampled)"
            continue
        d = I.generate_targeted([(fname, key, regs, mem, cc, n_rows)], seed, grid, workers=1, nearby_keys=nearby)
        if d is None:
            report[name] = f"{fname} {key.hex()}: no rows"
            continue
        d["store"] = L.learner_store(d)
        recs, pending, st = L.fit_trie(m, d, allow_flag_search=True, log=lambda s: None, workers=1)
        chk = I.generate_targeted([(fname, key, regs, mem, cc, n_check)], seed + 7, grid, workers=1, nearby_keys=0)
        chk["store"] = L.learner_store(chk)
        kept = []
        for r in recs:
            kb = bytes(r["key"])
            rows = np.flatnonzero([bytes(w[:len(kb)]) == kb for w in chk["window"]])
            if not len(rows):
                continue
            eff = r["effect"]
            try:
                ok, ab = L.check_rows(m, eff, L.Ctx(m, chk, rows, eff["k"], eff["runs"]))
            except Exception:  # noqa: BLE001
                continue
            if (~ok & ~ab).any():
                log(f"[fit] {name}: record {kb.hex()} wrong on {int((~ok & ~ab).sum())}/{len(rows)} check rows (dropped)")
                continue
            tmpl, slots = L._abstract(eff)
            kept.append({"key": kb, "mask": b"\xff" * len(kb), "fields": {}, "tmpl": tmpl, "consts": [v for _, v in slots],
                         "slot_kinds": [k for k, _ in slots], "rows": np.zeros(0, np.int64), "members": 1})
            m.literal_records.append(r)
        out += kept
        report[name] = f"{fname} {key.hex()}: fitted {[bytes(r['key']).hex() for r in recs]}, kept {[r['key'].hex() for r in kept]}"
        log(f"[fit] {name}: {report[name]}; fails {dict(st.most_common(4))}")
    return out, report


def program_summary(res):
    done = sum(r["done"] for r in res.values())
    wrong = sorted(n for n, r in res.items() if r.get("first_divergence_regs_rip") is not None)
    stops = {n: (r["status"], (r.get("abstain_at") or {}).get("window")) for n, r in res.items() if not r["done"]}
    return done, wrong, stops


def worker_m5e():
    """M5e lease: v3 + the M5e guards (byte register, SIB index none) + the shared EA unit + flag refinement rounds
    on fresh rows of the dev programs' predicate stops. Every machine is saved."""
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack, publish

    EADecoder, EAUnit = L.EADecoder, L.EAUnit
    t0 = time.time()
    grid = I.make_grid()
    res = {"config": cfg}
    train = regen_train(cfg, grid, log)
    m, dec = load_machine(cfg["machine"], cfg["machine_sha256"])
    res["custody_v3"] = custody(m, dec, train)
    log(f"[m5e] v3 custody {res['custody_v3']} t={time.time() - t0:.0f}s")
    n_byte = L.byte_reg_guard(m.records, train["window"])
    n_sib = L.sib_index_guard(m.records)
    base = L.FabricDecoder(m.records, m.code_map)
    res["guards"] = {"byte_register_closed": n_byte, "sib_index_closed": n_sib}
    res["custody_guarded"] = custody(m, base, train)
    log(f"[m5e] guards {res['guards']}; custody {res['custody_guarded']}")
    unit = EAUnit(m.literal_records, m.code_map, log=log)
    res["ea"] = unit.stats

    class Forced:
        maps = m.code_map

        def lookup(self, w):
            return unit.compose(bytes(w[:15]))

    # EA check 1: composition alone on the training rows of the literal address records it was factored from
    rows = np.unique(np.concatenate([r["rows"] for r in m.literal_records if '"sum"' in json.dumps(r["effect"])]))
    rows = np.random.default_rng(1).choice(rows, size=min(20000, len(rows)), replace=False)
    ok, ab = L.score(m, Forced(), train, rows=rows)
    res["ea_train"] = {"n": int(len(rows)), "exact": int(ok.sum()), "abstain": int(ab.sum()), "wrong": int((~ok & ~ab).sum())}
    log(f"[m5e] EA composition alone on training rows of address records: {res['ea_train']}")
    del train
    gc.collect()
    # EA check 2: the M4 held-out-cell panel (never trained on): base vs base+EA
    os.environ["ISA_ROW_DIST"] = "m5c"
    cells = I.generate(cfg["test_keys_per_form"], cfg["seed"] * 10 + 2, "cells", grid, workers=cfg["base_workers"])
    cells["store"] = L.learner_store(cells)
    for tag, d in (("base", base), ("base+ea", EADecoder(base, unit))):
        ok, ab = L.score(m, d, cells)
        res[f"cells_{tag}"] = {"n": int(len(ok)), "exact": int(ok.sum()), "abstain": int(ab.sum()), "wrong": int((~ok & ~ab).sum())}
        log(f"[m5e] held-out cells panel {tag}: {res[f'cells_{tag}']}")
    del cells
    gc.collect()
    progs = Path(cfg["native_dir"])
    names = sorted(p.name for p in progs.glob("*") if p.is_dir())
    res["rounds"] = []
    for rnd in range(cfg.get("refine_rounds", 4) + 1):
        dec = EADecoder(base, unit)
        pr = run_programs_inproc(m, dec, progs, names, log)
        done, wrong, stops = program_summary(pr)
        rec = {"round": rnd, "done": done, "of": len(names), "wrong_programs": wrong, "stops": stops,
               "composed_windows": sum(1 for v in dec.ea_cache.values() if v[0] is not None),
               "programs": {n: {k: pr[n].get(k) for k in ("status", "steps_run", "native_steps", "done", "first_divergence_regs_rip",
                                                          "abstain_at", "divergence_regs_rip")} for n in names}}
        log(f"[m5e] round {rnd}: programs done {done}/{len(names)}, programs with a wrong step {wrong}, EA-composed windows "
            f"{rec['composed_windows']} t={time.time() - t0:.0f}s")
        with open(out / f"machine_r{rnd}.pkl", "wb") as f:
            pickle.dump({"m": m, "dec": dec}, f)
        res["rounds"].append(rec)
        (out / "m5e.json").write_text(json.dumps(res, indent=1, default=str))
        gate = {f"r{rnd}:{n}": w[:16] for n, (st, w) in sorted(stops.items()) if st == "abstain_predicate" and w}
        nore = {n: w for n, (st, w) in sorted(stops.items()) if st == "abstain" and w}
        if rnd == cfg.get("refine_rounds", 4) or not (gate or nore):
            break
        if gate:
            sw = sweep_data({}, cfg["seed"] * 1000 + 910 + rnd, grid, cfg["workers"], gate=gate, n_gate=cfg.get("refine_rows", 2000))
            rec["sweep_before"] = sweep_score(m, dec, sw, log, f"r{rnd} before")
            rec["refine"] = refine_flags(m, base, sw["data"], log)
            del sw
        if nore:
            new, rec["fit"] = fit_stops(m, nore, grid, cfg["seed"] * 1000 + 950 + rnd, log, n_rows=cfg.get("fit_rows", 24))
            m.records = m.records + new
            rec["fitted_records"] = len(new)
        base = L.FabricDecoder(m.records, m.code_map)
        gc.collect()
    best = max(res["rounds"], key=lambda r: (not r["wrong_programs"], r["done"], -r["round"]))
    res["best_round"] = best["round"]
    log(f"[m5e] best round {best['round']}: done {best['done']}/{best['of']}, wrong programs {best['wrong_programs']} "
        f"t={time.time() - t0:.0f}s")
    res["wall_s"] = round(time.time() - t0, 1)
    (out / "m5e.json").write_text(json.dumps(res, indent=1, default=str))
    publish([str(out / f"machine_r{r['round']}.pkl") for r in res["rounds"]] + [str(out / "m5e.json")], "m5e-machines")
    pack(out)


def worker_eadiag():
    """Where EA composition alone is wrong on training rows: window, composed effect, the whole record's effect."""
    cfg = json.loads(Path("study.json").read_text())
    out = Path("result")
    out.mkdir(exist_ok=True)
    logf = open(out / "log.txt", "w")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    from isa_lab.full_custody import pack
    grid = I.make_grid()
    train = regen_train(cfg, grid, log)
    m, dec = load_machine(cfg["machine"], cfg["machine_sha256"])
    unit = L.EAUnit(m.literal_records, m.code_map, log=log)
    lits = [r for r in m.literal_records if '"sum"' in json.dumps(r["effect"])]
    rows = np.unique(np.concatenate([r["rows"] for r in lits]))
    rows = np.random.default_rng(1).choice(rows, size=min(cfg.get("ea_rows", 60000), len(rows)), replace=False)
    by_eff = defaultdict(list)
    for i in rows.tolist():
        ri, eff = unit.compose(train["window"][i])
        if eff is not None:
            by_eff[json.dumps(eff, sort_keys=True)].append(i)
    bad = []
    for ej, rr in by_eff.items():
        eff = json.loads(ej)
        x = L.Ctx(m, train, np.array(rr), eff["k"], eff["runs"])
        try:
            ok, ab = L.check_rows(m, eff, x)
        except Exception as e:  # noqa: BLE001
            bad.append({"error": repr(e)[:200], "window": bytes(train["window"][rr[0]]).hex(), "n": len(rr)})
            continue
        wr = np.flatnonzero(~ok & ~ab)
        if len(wr):
            w = bytes(train["window"][rr[wr[0]]])
            bri, beff = dec.lookup(train["window"][rr[wr[0]]])
            ri, _ = unit.compose(w)
            bad.append({"window": w.hex(), "n": len(rr), "wrong": int(len(wr)), "ea": ri, "composed": eff,
                        "base_record": bri, "base": beff,
                        "ops_key": bytes(unit.ops[ri[1]]["key"]).hex(), "ops_mask": bytes(unit.ops[ri[1]]["mask"]).hex(),
                        "addr_key": bytes(unit.addrs[ri[2]]["key"]).hex(), "addr_mask": bytes(unit.addrs[ri[2]]["mask"]).hex()})
    bad.sort(key=lambda b: -b.get("wrong", 0))
    log(f"[eadiag] composed groups {len(by_eff)}, wrong groups {len(bad)}, wrong rows {sum(b.get('wrong', 0) for b in bad)}")
    for b in bad[:60]:
        log(json.dumps(b, default=str)[:1500])
    (out / "eadiag.json").write_text(json.dumps(bad, indent=1, default=str))
    pack(out)
