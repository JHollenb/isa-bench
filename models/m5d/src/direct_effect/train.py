"""Data, Saturn-controlled training, shadow-checked evaluation, rollouts and Rosetta probes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from . import profile as P
from .model import FLAG_ORDER, Executor, batch_tensors, decode_outputs, from_bits, loss, targets, to_bits

# ---------------------------------------------------------------- datasets


def generate(n, seed, **kw):
    rng, oracle = P.make_rng(seed), P.Unicorn()
    return [P.example(rng, oracle, **kw) for _ in range(n)]


def held_pairs(n, seed):
    rng, oracle = P.make_rng(seed), P.Unicorn()
    out = []
    pairs = sorted(P.HELDOUT_PAIRS)
    while len(out) < n:
        form, reg = pairs[len(out) % len(pairs)]
        ins = P.encode(form, reg, P.random_imm(rng, P.FORMS[form][2], "train"), rng.randrange(8))
        out.append(P.example(rng, oracle, ins=ins))
    return out


def tensors(examples, device):
    win, regs, flags = batch_tensors(examples, device)
    return {"win": win, "regs": regs, "flags": flags, "tgt": targets(examples, device)}


def slice_t(data, idx):
    return {"win": data["win"][idx], "regs": data["regs"][idx], "flags": data["flags"][idx],
            "tgt": {k: v[idx] for k, v in data["tgt"].items()}}


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def isolated(model, data, examples, chunk=256):
    model.eval()
    rows = []
    n = data["win"].shape[0]
    for start in range(0, n, chunk):
        idx = torch.arange(start, min(n, start + chunk), device=data["win"].device)
        d = slice_t(data, idx)
        hard = decode_outputs(model(d["win"], d["regs"], d["flags"]))
        t = d["tgt"]
        width_bits = {0: 8, 1: 32, 2: 64}
        for j in range(len(idx)):
            e = examples[start + j]
            reg_ok = bool((hard["regs"][j] == t["regs"][j].long()).all())
            dest_ok = bool((hard["regs"][j, e["dest"]] == t["regs"][j, e["dest"]].long()).all())
            flags_ok = [bool(hard["flags"][j, k] == t["flags"][j, k]) for k in range(4)]
            w = e["width"]
            imm_ok = bool((hard["imm"][j, :w] == t["imm"][j, :w].long()).all()) if P.FORMS[e["form"]][2] else True
            rows.append({"form": e["form"], "state": reg_ok and all(flags_ok) and int(hard["length"][j]) == e["length"]
                         and int(hard["syscall"][j]) == e["syscall"], "regs": reg_ok, "dest_value": dest_ok,
                         "flags": all(flags_ok), **{f: flags_ok[k] for k, f in enumerate(FLAG_ORDER)},
                         "length": int(hard["length"][j]) == e["length"], "dest_decode": int(hard["dest"][j]) == e["dest"],
                         "width_decode": width_bits[int(hard["width"][j])] == w, "imm_decode": imm_ok,
                         "syscall": int(hard["syscall"][j]) == e["syscall"]})
    return rows


def summarize(rows, keys=("state", "regs", "dest_value", "flags", "CF", "ZF", "SF", "OF", "length", "dest_decode",
                          "width_decode", "imm_decode")):
    out = {"n": len(rows), **{k: sum(r[k] for r in rows) / max(len(rows), 1) for k in keys}}
    forms = {}
    for r in rows:
        f = forms.setdefault(r["form"], [0, 0, 0])
        f[0] += r["state"]
        f[1] += r["flags"]
        f[2] += 1
    out["per_form_state"] = {k: round(v[0] / v[2], 4) for k, v in sorted(forms.items())}
    out["per_form_flags"] = {k: round(v[1] / v[2], 4) for k, v in sorted(forms.items())}
    return out


@torch.no_grad()
def step_model(model, code, state, device):
    """One autonomous step: the model sees fetched bytes at its own RIP and its own previous state."""
    off = state["rip"] - P.CODE_BASE
    window = (code[off:off + P.WINDOW] + bytes(P.WINDOW)).ljust(P.WINDOW, b"\0")[:P.WINDOW]
    e = {"window": window.hex(), "before": state}
    win, regs, flags = batch_tensors([e], device)
    hard = decode_outputs(model(win, regs, flags))
    regs_out = [from_bits(hard["regs"][0, r].tolist()) for r in range(8)]
    nxt = {"regs": regs_out, "flags": {f: int(hard["flags"][0, k]) for k, f in enumerate(FLAG_ORDER)},
           "rip": state["rip"] + int(hard["length"][0])}
    return nxt, int(hard["syscall"][0])


def reference_rollout(code, state, max_steps):
    oracle, out = P.Unicorn(), [state]
    for _ in range(max_steps):
        off = state["rip"] - P.CODE_BASE
        if off >= len(code):
            break
        if code[off:off + 2] == b"\x0f\x05":
            out.append({"syscall": True})
            break
        after = oracle.step(dict(state, rip=P.CODE_BASE), code[off:off + P.WINDOW])
        after["rip"] = state["rip"] + (after["rip"] - P.CODE_BASE)
        state = after
        out.append(state)
    return out


def rollout(model, code, initial, max_steps, device, reference=None):
    """Feed only predicted state. Returns first divergence vs reference and the model trajectory."""
    state, traj, first = initial, [initial], None
    syscall = False
    for i in range(max_steps):
        state, sc = step_model(model, code, state, device)
        if sc:
            syscall = True
            break
        traj.append(state)
        if reference is not None and first is None:
            ref = reference[i + 1] if i + 1 < len(reference) else None
            if ref is None or "syscall" in ref or ref != state:
                first = i
    return {"steps": len(traj) - 1, "syscall": syscall, "first_divergence": first, "final": state, "trajectory": traj}


def random_program(rng, n, residue="train", allow_heldout_pairs=False):
    forms = tuple(f for f in P.FORMS if f != "syscall")
    return b"".join(P.random_instruction(rng, residue=residue, allow_heldout_pairs=allow_heldout_pairs, forms=forms).code
                    for _ in range(n))


def program_panel(model, device, seed, lengths=(16, 128, 1024), counts=(16, 8, 4), residue="train"):
    rng = P.make_rng(seed)
    out = {}
    for length, count in zip(lengths, counts, strict=True):
        rows = []
        for _ in range(count):
            code = random_program(rng, length, residue)
            init = P.random_state(rng)
            ref = reference_rollout(code, init, length)
            r = rollout(model, code, init, length, device, ref)
            first = r["first_divergence"] if r["first_divergence"] is not None or r["steps"] == length else r["steps"]
            rows.append({"first_divergence": first, "exact_all": first is None})
        out[str(length)] = {"programs": count, "exact": sum(x["exact_all"] for x in rows),
                            "first_divergence": [x["first_divergence"] for x in rows]}
    return out


def counter_panel(model, device, initial_states, lengths=(16, 128, 1024), native=None):
    out = {}
    for n in lengths:
        code = P.counter_code(n)
        init = initial_states.get(str(n), {"regs": [0] * 8, "flags": {f: 0 for f in FLAG_ORDER}, "rip": P.CODE_BASE})
        ref = reference_rollout(code, init, n)
        r = rollout(model, code, init, n, device, ref)
        exit_code = r["final"]["regs"][7] & 0xFF
        row = {"steps": r["steps"], "syscall": r["syscall"], "first_divergence_vs_unicorn": r["first_divergence"],
               "exit_code": exit_code, "expected_exit": (n - 3) & 0xFF, "rax_at_syscall": r["final"]["regs"][0],
               "complete": r["syscall"] and exit_code == (n - 3) & 0xFF and r["final"]["regs"][0] == 60}
        if native and str(n) in native:
            hw = native[str(n)]
            agree, first = 0, None
            for i, h in enumerate(hw[: len(r["trajectory"])]):
                m = r["trajectory"][i]
                ok = m["rip"] == h["rip"] and m["regs"] == h["regs"] and m["flags"] == h["flags"]
                agree += ok
                if not ok and first is None:
                    first = i
            row.update({"native_steps_compared": min(len(hw), len(r["trajectory"])), "native_agree": agree,
                        "native_first_disagreement": first})
        out[str(n)] = row
    return out


def pan_shadow_panel(examples):
    """Shadow-only: how often the frozen PAN typed adapter agrees with Unicorn on the same inputs."""
    out = {}
    for e in examples:
        if e["form"] in P.AL8 or e["form"] in ("nop", "syscall"):
            continue
        b = out.setdefault(e["form"], {"n": 0, "regs": 0, "flags": 0, "CF": 0, "OF": 0, "unsupported": 0})
        b["n"] += 1
        try:
            sh = P.pan_shadow(e["before"], bytes.fromhex(e["window"])[:e["length"]])
        except Exception:  # noqa: BLE001
            b["unsupported"] += 1
            continue
        b["regs"] += sh["regs"] == e["after"]["regs"]
        b["flags"] += sh["flags"] == e["after"]["flags"]
        b["CF"] += sh["flags"]["CF"] == e["after"]["flags"]["CF"]
        b["OF"] += sh["flags"]["OF"] == e["after"]["flags"]["OF"]
    return out


# ---------------------------------------------------------------- Rosetta probes


def logistic_probe(X_train, y_train, X_test, y_test, steps=300):
    """Standardized logistic-regression probe (torch L-BFGS); returns accuracy and balanced accuracy."""
    Xtr, Xte = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    ytr = torch.tensor(y_train, dtype=torch.float32)
    w = torch.zeros(Xtr.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss_ = torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-4 * w.square().sum()
        loss_.backward()
        return loss_
    opt.step(closure)
    with torch.no_grad():
        pred = ((Xte @ w + b) > 0).long().numpy()
    y = np.asarray(y_test)
    acc = float((pred == y).mean())
    pos, neg = y == 1, y == 0
    bal = float(((pred[pos] == 1).mean() if pos.any() else 1.0) / 2 + ((pred[neg] == 0).mean() if neg.any() else 1.0) / 2)
    return acc, bal


def carry_targets(e):
    """Carry (ADD) / borrow (SUB) into each bit position below the operation width."""
    width, kind, _ = P.FORMS[e["form"]]
    a = e["before"]["regs"][e["dest"]]
    if kind in ("inc", "dec"):
        b = 1
        kind = "add" if kind == "inc" else "sub"
    else:
        b = e["imm"]
    if kind not in ("add", "sub", "cmp"):
        return None
    sub = kind in ("sub", "cmp")
    c, out = 0, []
    for i in range(width):
        out.append(c)
        ai, bi = (a >> i) & 1, (b >> i) & 1
        c = ((1 - ai) & bi) | ((1 - (ai ^ bi)) & c) if sub else (ai & bi) | (c & (ai ^ bi))
    return out


def or_prefix_targets(e):
    width = P.FORMS[e["form"]][0]
    r = e["after"]["regs"][e["dest"]]
    acc, out = 0, []
    for i in range(width):
        acc |= (r >> i) & 1
        out.append(acc)
    return out


@torch.no_grad()
def probes(model, examples, device):
    """Linear probes per layer for carry-in and running-OR of the result at each bit position."""
    arith = [e for e in examples if carry_targets(e) is not None]
    data = tensors(arith, device)
    cap = {}
    model.eval()
    for start in range(0, len(arith), 256):
        part = {}
        idx = torch.arange(start, min(len(arith), start + 256), device=device)
        d = slice_t(data, idx)
        model(d["win"], d["regs"], d["flags"], capture=part)
        for k, v in part.items():
            cap.setdefault(k, []).append(v.cpu())
    cap = {k: torch.cat(v) for k, v in cap.items()}
    layers = ["input"] + [k for k in cap if k.startswith("layer")]
    results = {}
    for target_name, fn in (("carry_in", carry_targets), ("result_or_prefix", or_prefix_targets)):
        X_rows, y, ex_index = [], [], []
        for j, e in enumerate(arith):
            t = fn(e)
            for i, v in enumerate(t):
                X_rows.append((j, i))
                y.append(v)
                ex_index.append(j)
        y = np.array(y)
        split = np.array([j % 2 for j in ex_index]) == 0
        per_layer = {}
        for layer in layers:
            h = cap[layer].cpu().numpy()
            X = np.stack([h[j, i] for j, i in X_rows])
            if len(set(y[split])) < 2:
                per_layer[layer] = None
                continue
            acc, bal = logistic_probe(X[split], y[split], X[~split], y[~split])
            per_layer[layer] = {"acc": acc, "balanced": bal}
        results[target_name] = {"per_layer": per_layer, "majority": float(max(y[~split].mean(), 1 - y[~split].mean())),
                                "n_positions": int(len(y))}
    return results


# ---------------------------------------------------------------- training with Saturn


def train(arch, *, seed, out, device, train_n, steps_per_interval, intervals, batch, lr, emit=print):
    from isa_lab.dependencies import saturn_controller
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    discovery = generate(train_n, seed * 10 + 1)
    feedback = generate(2048, seed * 10 + 2)
    gen_s = time.perf_counter() - t0
    d = tensors(discovery, device)
    f = tensors(feedback, device)
    model = Executor(arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    saturn = saturn_controller()
    cursor = {"updates": 0}
    attempt = {"interval": -1, "step": 0}  # batch draws differ per interval attempt; rollback restores the model only
    observations = []
    probe_idx = torch.arange(0, min(512, train_n), device=device)
    policy = saturn.PromotionPolicy("lexicographic", (
        saturn.PromotionObjective("validation", "discovery_loss", "minimize"),
        saturn.PromotionObjective("autonomous_rollout", "rollout", "maximize"),
        saturn.PromotionObjective("trace_endpoints", "state", "maximize")))

    def restore_cursor(value):
        cursor.clear()
        cursor.update(value)

    controller = saturn.DeterministicCheckpointController(
        controller_id=f"isa-direct-effect-{arch}-{seed}", model=model, optimizer=optimizer, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore_cursor, max_interval_steps=steps_per_interval,
        receipt_store=saturn.ImmutableDirectoryReceiptStore(out / "decisions"))
    fb_rng_seed = seed * 10 + 3

    def evaluator():
        model.eval()
        with torch.no_grad():
            sub = slice_t(d, probe_idx)
            dl = float(loss(model(sub["win"], sub["regs"], sub["flags"]), sub["tgt"], sub["regs"])[0])
        rows = isolated(model, f, feedback)
        s = summarize(rows)
        progs = program_panel(model, device, fb_rng_seed, lengths=(32,), counts=(8,))
        roll = progs["32"]["exact"] / progs["32"]["programs"]
        observations.append({"updates": cursor["updates"], "discovery_loss": dl, "feedback_state": s["state"],
                             "feedback_flags": s["flags"], "feedback_regs": s["regs"], "rollout32": roll,
                             "wall_s": time.perf_counter() - t0})
        emit("ISA_DE_FUTURE:" + json.dumps({"arch": arch, **observations[-1]}))
        model.train()
        return saturn.EvaluationScores(validation={"discovery_loss": dl}, autonomous_rollout={"rollout": roll},
            trace_endpoints={"state": s["state"]},
            evidence={"feedback_split": "feedback (Unicorn labels)", "optimizer_split": "discovery", "labels_used": True})

    def update(_):
        attempt["step"] += 1
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + attempt["interval"] * 100_000 + attempt["step"])
        idx = torch.randint(0, train_n, (batch,), generator=g).to(device)
        sub = slice_t(d, idx)
        optimizer.zero_grad(set_to_none=True)
        total, _ = loss(model(sub["win"], sub["regs"], sub["flags"]), sub["tgt"], sub["regs"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cursor["updates"] += 1
        return {"loss": float(total.detach())}

    model.train()
    controller.establish_baseline(evaluator)
    for interval in range(intervals):
        attempt["interval"] = interval
        controller.run_interval(interval_id=f"interval-{interval:03d}", steps=steps_per_interval, train_step=update,
                                evaluator=evaluator)
    model.eval()
    torch.save(model.state_dict(), out / "model.pt")
    return model, {"trajectory": observations, "generation_s": gen_s, "train_wall_s": time.perf_counter() - t0,
                   "parameters": sum(p.numel() for p in model.parameters()), "accepted_updates": cursor["updates"],
                   "rollback_receipts": [r.to_dict() for r in controller.ledger.receipts]}


# ---------------------------------------------------------------- C3b: latent register state


def generate_sequences(n, length, seed, residue="train"):
    rng, oracle = P.make_rng(seed), P.Unicorn()
    forms = tuple(f for f in P.FORMS if f != "syscall")
    seqs = []
    for _ in range(n):
        state = P.random_state(rng)
        ins = [P.random_instruction(rng, residue=residue, forms=forms) for _ in range(length)]
        if rng.random() < 0.2:
            ins[-1] = P.encode("syscall")
        code = b"".join(i.code for i in ins)
        steps, cur = [], state
        for i in ins:
            after = P.label(oracle, cur, i)
            off = cur["rip"] - P.CODE_BASE
            win = (code[off:off + P.WINDOW]).ljust(P.WINDOW, b"\0")
            steps.append({"form": i.form, "window": win.hex(), "length": len(i.code), "dest": i.dest, "src": i.src,
                          "width": i.width, "imm": i.imm, "syscall": int(i.form == "syscall"), "before": cur, "after": after})
            cur = after
        seqs.append(steps)
    return seqs


def seq_tensors(seqs, device):
    T = len(seqs[0])
    per_step = [tensors([s[t] for s in seqs], device) for t in range(T)]
    return per_step


def latent_loss(model, steps):
    from torch.nn import functional as F
    first = steps[0]
    h = model.encode(first["regs"], first["flags"])
    rec = model.readout(h)
    total = F.binary_cross_entropy_with_logits(rec["regs"], first["regs"]) + \
        F.binary_cross_entropy_with_logits(rec["flags"], first["flags"])
    for st in steps:
        out, h = model(st["win"], h)
        t = st["tgt"]
        w = 1 + 20 * (t["regs"] != st["regs"]).float()
        total = total + F.binary_cross_entropy_with_logits(out["regs"], t["regs"], weight=w) \
            + F.binary_cross_entropy_with_logits(out["flags"], t["flags"]) \
            + F.cross_entropy(out["length"], t["length"]) + F.binary_cross_entropy_with_logits(out["syscall"], t["syscall"]) \
            + F.cross_entropy(out["dest"], t["dest"]) + F.cross_entropy(out["width"], t["width"]) \
            + F.binary_cross_entropy_with_logits(out["imm"], t["imm"])
    return total / len(steps)


@torch.no_grad()
def latent_rollout(model, code, initial, max_steps, device, reference=None, capture_codes=False):
    win, regs, flags = batch_tensors([{"window": "00" * P.WINDOW, "before": initial}], device)
    h = model.encode(regs, flags)
    state, traj, first, syscall, codes = initial, [initial], None, False, []
    for i in range(max_steps):
        off = state["rip"] - P.CODE_BASE
        window = (code[off:off + P.WINDOW]).ljust(P.WINDOW, b"\0")[:P.WINDOW]
        w, _, _ = batch_tensors([{"window": window.hex(), "before": initial}], device)
        out, h = model(w, h)
        if capture_codes:
            codes.append(h[0].cpu())
        hard = decode_outputs({**out, "src": out["dest"]})
        if int(hard["syscall"][0]):
            syscall = True
            break
        state = {"regs": [from_bits(hard["regs"][0, r].tolist()) for r in range(8)],
                 "flags": {f: int(hard["flags"][0, k]) for k, f in enumerate(FLAG_ORDER)},
                 "rip": state["rip"] + int(hard["length"][0])}
        traj.append(state)
        if reference is not None and first is None:
            ref = reference[i + 1] if i + 1 < len(reference) else None
            if ref is None or "syscall" in ref or ref != state:
                first = i
    return {"steps": len(traj) - 1, "syscall": syscall, "first_divergence": first, "final": state, "trajectory": traj,
            "codes": codes}


def latent_program_panel(model, device, seed, lengths, counts, residue="train"):
    rng = P.make_rng(seed)
    out = {}
    for length, count in zip(lengths, counts, strict=True):
        firsts = []
        for _ in range(count):
            code = random_program(rng, length, residue)
            init = P.random_state(rng)
            ref = reference_rollout(code, init, length)
            r = latent_rollout(model, code, init, length, device, ref)
            first = r["first_divergence"] if r["steps"] == length else (r["first_divergence"] if r["first_divergence"] is not None else r["steps"])
            firsts.append(first)
        out[str(length)] = {"programs": count, "exact": sum(f is None for f in firsts), "first_divergence": firsts}
    return out


def latent_counter_panel(model, device, initial_states, lengths, native=None):
    out = {}
    for n in lengths:
        code = P.counter_code(n)
        init = initial_states.get(str(n), {"regs": [0] * 8, "flags": {f: 0 for f in FLAG_ORDER}, "rip": P.CODE_BASE})
        ref = reference_rollout(code, init, n)
        r = latent_rollout(model, code, init, n, device, ref)
        exit_code = r["final"]["regs"][7] & 0xFF
        row = {"steps": r["steps"], "syscall": r["syscall"], "first_divergence_vs_unicorn": r["first_divergence"],
               "exit_code": exit_code, "complete": r["syscall"] and exit_code == (n - 3) & 0xFF and r["final"]["regs"][0] == 60}
        if native and str(n) in native:
            hw = native[str(n)]
            agree = sum(r["trajectory"][i] == h for i, h in enumerate(hw[: len(r["trajectory"])]))
            row.update({"native_steps_compared": min(len(hw), len(r["trajectory"])), "native_agree": agree})
        out[str(n)] = row
    return out


@torch.no_grad()
def latent_probes(model, device, seed, n=64, length=24):
    """Rosetta probes on the learned latent code: register bits and carry, by linear probes."""
    rng = P.make_rng(seed)
    X, Y = [], []
    for _ in range(n):
        code = random_program(rng, length)
        init = P.random_state(rng)
        ref = reference_rollout(code, init, length)
        r = latent_rollout(model, code, init, length, device, None, capture_codes=True)
        for t, c in enumerate(r["codes"]):
            if t + 1 >= len(ref) or "syscall" in ref[t + 1]:
                break
            truth = ref[t + 1]
            for i in range(64):
                X.append(c[i].numpy())
                Y.append([(truth["regs"][k] >> i) & 1 for k in range(8)])
    X, Y = np.array(X), np.array(Y)
    half = len(X) // 2
    out = {}
    for k, name in enumerate(P.GPRS):
        y = Y[:, k]
        if len(set(y[:half])) < 2:
            continue
        out[name] = logistic_probe(X[:half], y[:half], X[half:], y[half:])[1]
    return {"register_bit_from_code_balanced": out, "positions": int(len(X))}


def train_latent(arch, *, seed, out, device, n_seq, seq_len, steps_per_interval, intervals, batch, lr, emit=print):
    from isa_lab.dependencies import saturn_controller

    from .model import LatentExecutor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    seqs = generate_sequences(n_seq, seq_len, seed * 10 + 11)
    data = seq_tensors(seqs, device)
    gen_s = time.perf_counter() - t0
    model = LatentExecutor(arch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    saturn = saturn_controller()
    cursor = {"updates": 0}
    attempt = {"interval": -1, "step": 0}
    observations = []
    policy = saturn.PromotionPolicy("lexicographic", (
        saturn.PromotionObjective("validation", "discovery_loss", "minimize"),
        saturn.PromotionObjective("autonomous_rollout", "rollout", "maximize"),
        saturn.PromotionObjective("trace_endpoints", "counter", "maximize")))

    def restore_cursor(value):
        cursor.clear()
        cursor.update(value)

    controller = saturn.DeterministicCheckpointController(
        controller_id=f"isa-direct-effect-latent-{arch}-{seed}", model=model, optimizer=optimizer, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore_cursor, max_interval_steps=steps_per_interval,
        receipt_store=saturn.ImmutableDirectoryReceiptStore(out / "decisions"))
    sub_idx = torch.arange(0, min(256, n_seq), device=device)

    def evaluator():
        model.eval()
        with torch.no_grad():
            dl = float(latent_loss(model, [slice_t(s, sub_idx) for s in data]))
        progs = latent_program_panel(model, device, seed * 10 + 12, (32,), (8,))
        roll = progs["32"]["exact"] / 8
        counter = latent_counter_panel(model, device, {}, (16,))["16"]["complete"]
        observations.append({"updates": cursor["updates"], "discovery_loss": dl, "rollout32": roll, "counter16": counter,
                             "wall_s": time.perf_counter() - t0})
        emit("ISA_DE_FUTURE:" + json.dumps({"arch": arch + "_latent", **observations[-1]}))
        model.train()
        return saturn.EvaluationScores(validation={"discovery_loss": dl}, autonomous_rollout={"rollout": roll},
            trace_endpoints={"counter": float(counter)},
            evidence={"feedback_split": "feedback programs (Unicorn)", "optimizer_split": "discovery sequences", "labels_used": True})

    def update(_):
        attempt["step"] += 1
        g = torch.Generator(device="cpu").manual_seed(seed * 1_000_033 + attempt["interval"] * 100_000 + attempt["step"])
        idx = torch.randint(0, n_seq, (batch,), generator=g).to(device)
        optimizer.zero_grad(set_to_none=True)
        total = latent_loss(model, [slice_t(s, idx) for s in data])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        cursor["updates"] += 1
        return {"loss": float(total.detach())}

    model.train()
    controller.establish_baseline(evaluator)
    for interval in range(intervals):
        attempt["interval"] = interval
        controller.run_interval(interval_id=f"interval-{interval:03d}", steps=steps_per_interval, train_step=update,
                                evaluator=evaluator)
    model.eval()
    torch.save(model.state_dict(), out / "model.pt")
    return model, {"trajectory": observations, "generation_s": gen_s, "train_wall_s": time.perf_counter() - t0,
                   "parameters": sum(p.numel() for p in model.parameters()), "accepted_updates": cursor["updates"],
                   "rollback_receipts": [r.to_dict() for r in controller.ledger.receipts]}
