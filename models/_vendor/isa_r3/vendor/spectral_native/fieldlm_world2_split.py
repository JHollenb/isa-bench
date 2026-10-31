"""Evaluation side of track C2: paraphrase split, probes and induction quality.

This module reads the world's gold structure (templates, lexicon, trees).  It
is used only to build splits and to score what ``fieldlm_world2`` induced; the
learner never imports it.
"""

from __future__ import annotations

import dataclasses
import math
import re
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .fieldlm_world import (
    ATTR_R,
    ATTR_S,
    FAMILY,
    NVAL,
    REGION,
    RULE_SOURCE,
    SPECIES,
    TEMPLATES,
    VALUE,
    ZONE,
    Corpus,
    World,
    render,
    split_sentences,
)

# Templates removed from training entirely in the paraphrase split, and the
# kept template of the same group that replaces each occurrence.
HELD_OUT = {("isa", 2): ("isa", 0), ("partof", 1): ("partof", 0), ("fact", 1): ("fact", 0),
            ("habitat", 1): ("habitat", 0), ("qa2", 0): ("qa1", 0)}
SLOT_TYPES = {"X": None, "Y": None, "A": (ATTR_S, ATTR_R), "V": (VALUE,), "R": (REGION, ZONE)}


def gold_match(sent: list[str], lex: dict[str, str]) -> tuple[str, int, dict[str, str]] | None:
    """(template group, index, slot fillers) of one sentence, by exact template match."""

    for group, tpls in TEMPLATES.items():
        for i, tpl in enumerate(tpls):
            words = tpl.split()
            if len(words) != len(sent):
                continue
            slots: dict[str, str] = {}
            ok = True
            for w, s in zip(words, sent):
                if w.startswith("{"):
                    name = w[1:-1]
                    allowed = SLOT_TYPES.get(name)
                    if allowed is not None and lex.get(s) not in allowed:
                        ok = False
                        break
                    if allowed is None and lex.get(s) not in (SPECIES, FAMILY, REGION, ZONE):
                        ok = False
                        break
                    slots[name] = s
                elif w != s:
                    ok = False
                    break
            if ok:
                return group, i, slots
    return None


def gold_relation(group: str, slots: dict[str, str], lex: dict[str, str]) -> str:
    if group in ("isa", "partof"):
        return group
    if group in ("habitat", "story"):
        return "habitat"
    if group == "qah":
        return f"hop:{slots['A']}"
    return f"attr:{slots['A']}"


def paraphrase_units(world: World, corpus: Corpus) -> tuple[list[list[str]], dict[str, int]]:
    """Training units with every held-out template re-rendered in its replacement."""

    lex = world.lexicon()
    counts: Counter = Counter()
    out = []
    for unit in corpus.train_units:
        new: list[str] = []
        for sent in split_sentences(unit):
            if sent[-1] != ".":
                new += sent
                continue
            m = gold_match(sent, lex)
            if m is not None and (m[0], m[1]) in HELD_OUT:
                group, idx = HELD_OUT[(m[0], m[1])]
                counts[f"{m[0]}[{m[1]}]"] += 1
                new += render(TEMPLATES[group][idx], **m[2])
            else:
                new += sent
        out.append(new)
    # the question prefix of a unit is followed by its answer inside the same sentence
    return out, dict(counts)


def paraphrase_corpus(world: World, corpus: Corpus) -> tuple[Corpus, dict[str, int]]:
    units, counts = paraphrase_units(world, corpus)
    return dataclasses.replace(corpus, train_units=units), counts


def _cut(tpl: str, stop: str, **slots: str) -> tuple[list[str], str]:
    toks = render(tpl, **slots)
    words = tpl.split()
    i = words.index("{" + stop + "}")
    return toks[:i], toks[i]


def paraphrase_probes(world: World, corpus: Corpus, story_ks=(0, 16, 256)) -> dict[str, list[tuple[list[str], str, list[str] | None]]]:
    """Probes phrased only in held-out templates: (prompt, answer, context)."""

    T = world.value_token
    sp, rg = world.species, world.regions
    probes: dict[str, list] = defaultdict(list)
    qa2 = TEMPLATES["qa2"][0]
    for key in ("single", "rule"):
        for q in corpus.test[key]:
            n, a = q.meta["node"], q.meta["attr"]
            tok = world.sp_tok[n]
            p, ans = _cut(qa2, "V", X=tok, A=a, V=q.answer)
            probes[f"qa2_{key}"].append((p, ans, None))
            if q.meta.get("inferable"):
                probes[f"qa2_{key}_inferable"].append((p, ans, None))
            p, ans = _cut(TEMPLATES["fact"][1], "V", X=tok, A=a, V=q.answer)
            probes[f"has_{key}"].append((p, ans, None))
            if q.meta.get("inferable"):
                probes[f"has_{key}_inferable"].append((p, ans, None))
    for n, a in sorted(corpus.stated_species_cells):
        v = T(a, world.truth_species(n, a))
        p, ans = _cut(qa2, "V", X=world.sp_tok[n], A=a, V=v)
        probes["qa2_recall"].append((p, ans, None))
        p, ans = _cut(TEMPLATES["fact"][1], "V", X=world.sp_tok[n], A=a, V=v)
        probes["has_recall"].append((p, ans, None))
    for n in sp.order()[1:]:
        p, ans = _cut(TEMPLATES["isa"][2], "Y", X=world.sp_tok[n], Y=world.sp_tok[int(sp.parent[n])])
        probes["isa_one_of"].append((p, ans, None))
    for n in rg.order()[1:]:
        p, ans = _cut(TEMPLATES["partof"][1], "Y", X=world.rg_tok[n], Y=world.rg_tok[int(rg.parent[n])])
        probes["partof_lies_within"].append((p, ans, None))
    for n in sorted(corpus.stated_habitat):
        p, ans = _cut(TEMPLATES["habitat"][1], "R", X=world.sp_tok[n], R=world.rg_tok[world.habitat[n]])
        probes["home_recall"].append((p, ans, None))
    # reading: the story's target statement in the held-out habitat template
    story_tpl = TEMPLATES["story"][0].split()
    for k, items in corpus.story_test.items():
        if k not in story_ks:
            continue
        for doc, q in items:
            first = doc[: len(story_tpl)]
            x, r = first[4], first[7]
            new_doc = render(TEMPLATES["habitat"][1], X=x, R=r) + doc[len(story_tpl):]
            probes[f"story_home_{k}"].append((q.prompt, q.answer, new_doc))
    return dict(probes)


# ---------------------------------------------------------------------------
# Induction quality against gold
# ---------------------------------------------------------------------------


def _pair_scores(pred: dict[str, str], gold: dict[str, str]) -> dict[str, float]:
    """Pairwise precision/recall of a predicted partition against a gold partition."""

    keys = [k for k in pred if k in gold]
    by_pred: dict[str, Counter] = defaultdict(Counter)
    for k in keys:
        by_pred[pred[k]][gold[k]] += 1
    c2 = lambda n: n * (n - 1) // 2
    tp = sum(c2(n) for cnt in by_pred.values() for n in cnt.values())
    pp = sum(c2(sum(cnt.values())) for cnt in by_pred.values())
    gp = sum(c2(n) for n in Counter(gold[k] for k in keys).values())
    return {"precision": tp / pp if pp else 1.0, "recall": tp / gp if gp else 1.0, "items": len(keys)}


def gold_class(tok: str, lex: dict[str, str]) -> str:
    t = lex.get(tok)
    if t in (SPECIES, FAMILY):
        return "entity:species"
    if t in (REGION, ZONE):
        return "entity:region"
    if t == VALUE:
        return "value:" + re.sub(r"\d+$", "", tok)
    return "token:" + tok


def induction_quality(lm, world: World) -> dict[str, Any]:
    lex = world.lexicon()
    p = lm.p
    out: dict[str, Any] = {}
    pred_cls = {w: (c if c in p.open else "single:" + w) for w, c in p.cls.items()}
    gold_cls = {w: gold_class(w, lex) for w in p.cls}
    out["classes"] = _pair_scores(pred_cls, gold_cls)
    out["classes"]["open_classes"] = len(p.open)
    out["classes"]["function_tokens"] = sum(1 for c in pred_cls.values() if c.startswith("single:"))
    # templates -> relations
    pred_rel, gold_rel = {}, {}
    unmatched = 0
    for t in p.templates.values():
        if not t.slots:
            continue
        m = gold_match(list(t.example), lex)
        if m is None:
            unmatched += 1
            continue
        r = p.rel_of_template.get(t.tid)
        pred_rel[str(t.tid)] = str(r.rid) if r is not None else "none"
        gold_rel[str(t.tid)] = gold_relation(m[0], m[2], lex)
    out["relations"] = _pair_scores(pred_rel, gold_rel)
    out["relations"]["templates"] = len(p.templates)
    out["relations"]["unmatched_templates"] = unmatched
    out["relations"]["induced"] = len(p.relations)
    out["relations"]["gold"] = len(set(gold_rel.values()))
    # relation kinds and explanations
    rel_gold: dict[int, str] = {}
    for tid, g in gold_rel.items():
        rel_gold.setdefault(int(pred_rel[tid]), g) if pred_rel[tid] != "none" else None
    kinds = []
    for r in p.relations:
        g = rel_gold.get(r.rid)
        if g is None:
            continue
        want = "hierarchy" if g in ("isa", "partof") else "link" if g == "habitat" else "attribute"
        kinds.append(r.kind == want)
    out["kind_accuracy"] = float(np.mean(kinds)) if kinds else None
    expl = {}
    correct = []
    for r in lm.attrs:
        g = rel_gold.get(r.rid)
        e = lm.explain[r.rid]
        src = rel_gold.get(e.get("source")) if e["type"] == "rule" else None
        expl[g or f"rel{r.rid}"] = {"type": e["type"], "form": e.get("form"), "source": src, "bits": float(e["bits"])}
        if g is None:
            continue
        name = g.split(":", 1)[1]
        if g.startswith("hop:"):
            correct.append(e["type"] == "rule" and src == f"attr:{name}")
        elif name in RULE_SOURCE:
            correct.append(e["type"] == "rule" and src == f"attr:{RULE_SOURCE[name]}")
        else:
            correct.append(e["type"] == "tree")
    out["explanations"] = expl
    out["explanation_accuracy"] = float(np.mean(correct)) if correct else None
    # hierarchies: induced parent of every node vs the world's parent
    for name, tree, tok in (("species", world.species, world.sp_tok), ("regions", world.regions, world.rg_tok)):
        cls = p.slot_class(tok[0])
        h = lm.hier.get(cls)
        if h is None:
            out[f"hierarchy_{name}"] = {"edges": 0, "accuracy": 0.0}
            continue
        inv = {i: t for t, i in h.index.items()}
        ok = []
        for n in range(1, tree.size):
            node = h.index.get(tok[n])
            up = None if node is None or h.tree.parent[node] < 0 else inv[int(h.tree.parent[node])]
            ok.append(up == tok[int(tree.parent[n])])
        out[f"hierarchy_{name}"] = {"edges": len(ok), "accuracy": float(np.mean(ok))}
    return out


def value_order(members: list[str]) -> list[str] | None:
    """Ablation A2 (supplied): value token ``{attr}{v}`` has id v, for all NVAL values."""

    attrs = {re.sub(r"\d+$", "", w) for w in members}
    if len(attrs) != 1 or not all(re.search(r"\d+$", w) for w in members):
        return None
    attr = attrs.pop()
    return [f"{attr}{v}" for v in range(NVAL)]
