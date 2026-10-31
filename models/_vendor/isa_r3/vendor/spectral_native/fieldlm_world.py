"""A synthetic world with a known ontology, rendered as text, and a field LM whose
answer heads join against the 2-adic knowledge store.

World: a species taxonomy (balanced 4-ary tree), a region hierarchy, a habitat
relation (cutting across the taxonomy), species attributes generated down the
taxonomy, region attributes generated down the region tree, and two rule
attributes that follow a region attribute of the habitat.

Text: every fact is rendered with one of several paraphrase templates:
is-a sentences, part-of sentences, species/family facts, region facts,
habitat sentences, question-answer pairs (single hop and habitat hop), and
short "stories" that restate a habitat in context.

Field LM (``FieldLM``):
- Lexical types of tokens are supplied (species, family, region, zone,
  species attribute, region attribute, value, function word).  A template is
  the token sequence with slots replaced by their types.
- Declarative templates write facts into stores; the relation is determined by
  the slot types (supplied).  Question templates choose their read operation
  (fact lookup, habitat hop, region lookup) by agreement with the answers in
  the training text (learned).
- The species store is a family program on the 2-adic codes of the taxonomy
  built from the is-a sentences: constrained small parsimony with stated family
  facts as constraints, most specific family wins, looked up by dyadic phase
  characters.  Region facts likewise on the region tree.  Species attributes
  choose, by description length, between inheritance and a rule on a region
  attribute of the habitat.  Hidden habitats are abduced from stated rule facts.
- Context sentences (stories) write an overlay that takes precedence
  (recency), so a restated habitat is used however far back it appears.
- Next-token probabilities: a Bayesian mixture over templates consistent with
  the prefix; function words from the template, slot tokens from a unigram
  prior of their type, and the answer slot from the store lookup with
  calibrated smoothing.
- A family-level edit changes one override entry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .knowledge import UNKNOWN, Program, Tree, execute_by_characters, leaves_under

SP_ATTRS = ["diet", "color", "size", "pattern", "call", "activity", "nest", "span"]
RULE_ATTRS = ["coat", "skin"]  # coat follows climate, skin follows terrain
RULE_SOURCE = {"coat": "climate", "skin": "terrain"}
REG_ATTRS = ["climate", "terrain"]
ALL_SPECIES_ATTRS = SP_ATTRS + RULE_ATTRS
NVAL = 8

TEMPLATES = {
    "isa": ["the {X} is a kind of {Y} .", "{X} belongs to the {Y} family .", "{X} is one of the {Y} ."],
    "partof": ["{X} is part of {Y} .", "{X} lies within {Y} ."],
    "fact": ["the {A} of {X} is {V} .", "{X} has {A} {V} ."],
    "family_fact": ["members of {X} have {A} {V} ."],
    "habitat": ["{X} lives in {R} .", "the home of {X} is {R} ."],
    "qa1": ["q : what is the {A} of {X} ? a : {V} ."],
    "qa2": ["q : {X} has which {A} ? a : {V} ."],
    "qah": ["q : what {A} does {X} live in ? a : {V} ."],
    "story": ["in this story , {X} lives in {R} ."],
}

SPECIES, FAMILY, REGION, ZONE, ATTR_S, ATTR_R, VALUE, FUNC = (
    "SPECIES", "FAMILY", "REGION", "ZONE", "ATTR_S", "ATTR_R", "VALUE", "FUNC")


# ---------------------------------------------------------------------------
# World generation
# ---------------------------------------------------------------------------


def balanced(branching: int, depth: int) -> Tree:
    from .knowledge import balanced_tree

    return balanced_tree(branching, depth)


def gen_node_values(tree: Tree, values: int, mutation: float, exception: float, rng: np.random.Generator):
    node_val = np.zeros(tree.size, dtype=int)
    node_val[0] = rng.integers(values)
    overrides = {0}
    leaf_set = set(tree.leaves.tolist())
    for n in tree.order()[1:]:
        up = node_val[tree.parent[n]]
        rate = exception if n in leaf_set else mutation
        if rng.random() < rate:
            new = int(rng.integers(values - 1))
            node_val[n] = new + (new >= up)
            overrides.add(int(n))
        else:
            node_val[n] = up
    return node_val, overrides


def relabel_from_overrides(tree: Tree, overrides: dict[int, int]) -> np.ndarray:
    labels = np.zeros(tree.size, dtype=int)
    for n in tree.order():
        labels[n] = overrides[n] if n in overrides else labels[tree.parent[n]]
    return labels


@dataclass
class World:
    seed: int
    species: Tree
    regions: Tree
    sp_tok: dict[int, str]
    rg_tok: dict[int, str]
    tok_node: dict[str, tuple[str, int]]
    sp_over: dict[str, dict[int, int]]  # attr -> generator overrides node -> value
    rg_over: dict[str, dict[int, int]]
    habitat: dict[int, int]  # species leaf node -> region leaf node
    rule_g: dict[str, np.ndarray]
    rule_flip: dict[str, dict[int, int]]  # attr -> species leaf -> exception value

    def value_token(self, attr: str, v: int) -> str:
        return f"{attr}{v}"

    def sp_labels(self, attr: str) -> np.ndarray:
        return relabel_from_overrides(self.species, self.sp_over[attr])

    def rg_labels(self, attr: str) -> np.ndarray:
        return relabel_from_overrides(self.regions, self.rg_over[attr])

    def truth_species(self, node: int, attr: str, habitat: dict[int, int] | None = None) -> int:
        if attr in SP_ATTRS:
            return int(self.sp_labels(attr)[node])
        hab = (habitat or self.habitat)[node]
        if node in self.rule_flip[attr] and habitat is None:
            return self.rule_flip[attr][node]
        return int(self.rule_g[attr][self.rg_labels(RULE_SOURCE[attr])[hab]])

    def truth_region(self, node: int, attr: str) -> int:
        return int(self.rg_labels(attr)[node])

    def lexicon(self) -> dict[str, str]:
        lex: dict[str, str] = {}
        leaf_sp = set(self.species.leaves.tolist())
        leaf_rg = set(self.regions.leaves.tolist())
        for n, t in self.sp_tok.items():
            lex[t] = SPECIES if n in leaf_sp else FAMILY
        for n, t in self.rg_tok.items():
            lex[t] = REGION if n in leaf_rg else ZONE
        for a in ALL_SPECIES_ATTRS:
            lex[a] = ATTR_S
            for v in range(NVAL):
                lex[self.value_token(a, v)] = VALUE
        for a in REG_ATTRS:
            lex[a] = ATTR_R
            for v in range(NVAL):
                lex[self.value_token(a, v)] = VALUE
        for group in TEMPLATES.values():
            for tpl in group:
                for w in tpl.split():
                    if not w.startswith("{"):
                        lex.setdefault(w, FUNC)
        return lex


def make_world(seed: int, rule_kind: str = "table") -> World:
    """``rule_kind``: "table" (random permutation, identified only where observed)
    or "affine" (y = a*x + k mod NVAL with a odd; identified from few pairs)."""

    rng = np.random.default_rng(seed)
    species = balanced(4, 4)
    regions = balanced(4, 3)
    sp_tok: dict[int, str] = {}
    leaf_index = {int(n): i for i, n in enumerate(species.leaves)}
    fam = 0
    for n in species.order():
        if n == 0:
            sp_tok[n] = "animal"
        elif n in leaf_index:
            sp_tok[n] = f"sp{leaf_index[n]}"
        else:
            sp_tok[n] = f"fam{fam}"
            fam += 1
    rg_tok: dict[int, str] = {}
    rleaf = {int(n): i for i, n in enumerate(regions.leaves)}
    zone = 0
    for n in regions.order():
        if n == 0:
            rg_tok[n] = "earth"
        elif n in rleaf:
            rg_tok[n] = f"reg{rleaf[n]}"
        else:
            rg_tok[n] = f"zone{zone}"
            zone += 1
    tok_node = {t: ("species", n) for n, t in sp_tok.items()}
    tok_node.update({t: ("region", n) for n, t in rg_tok.items()})
    sp_over, rg_over = {}, {}
    for a in SP_ATTRS:
        vals, over = gen_node_values(species, NVAL, 0.15, 0.05, rng)
        sp_over[a] = {n: int(vals[n]) for n in over}
    for a in REG_ATTRS:
        vals, over = gen_node_values(regions, NVAL, 0.5, 0.05, rng)
        rg_over[a] = {n: int(vals[n]) for n in over}
    habitat = {int(n): int(regions.leaves[rng.integers(regions.leaves.shape[0])]) for n in species.leaves}
    if rule_kind == "affine":
        rule_g = {}
        for a in RULE_ATTRS:
            mult = int(rng.choice([1, 3, 5, 7]))
            shift = int(rng.integers(NVAL))
            rule_g[a] = np.array([(mult * x + shift) % NVAL for x in range(NVAL)])
    else:
        rule_g = {a: rng.permutation(NVAL) for a in RULE_ATTRS}
    rule_flip: dict[str, dict[int, int]] = {}
    for a in RULE_ATTRS:
        rule_flip[a] = {}
        for n in species.leaves:
            if rng.random() < 0.03:
                rule_flip[a][int(n)] = int(rng.integers(NVAL))
    return World(seed, species, regions, sp_tok, rg_tok, tok_node, sp_over, rg_over, habitat, rule_g, rule_flip)


# ---------------------------------------------------------------------------
# Text rendering and splits
# ---------------------------------------------------------------------------


def render(tpl: str, **slots: str) -> list[str]:
    out = []
    for w in tpl.split():
        if w.startswith("{"):
            out.append(slots[w[1:-1]])
        else:
            out.append(w)
    return out


@dataclass
class QA:
    prompt: list[str]  # tokens up to and including "a :"
    answer: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Corpus:
    world: World
    train_units: list[list[str]]  # each unit: one sentence or one story document
    val: dict[str, list[QA]]
    test: dict[str, list[QA]]
    heldout_sentences: list[list[str]]
    heldout_answer_index: list[int]
    story_test: dict[int, list[tuple[list[str], QA]]]
    stated_species_cells: set[tuple[int, str]]
    stated_family_facts: set[tuple[int, str]]
    stated_habitat: set[int]


def qa_tokens(kind: str, entity: str, attr: str, value: str) -> tuple[list[str], str]:
    tpl = TEMPLATES[kind][0]
    tokens = render(tpl, X=entity, A=attr, V=value)
    cut = tokens.index("a") + 2  # up to and including "a :"
    return tokens[:cut], tokens[cut]


def build_corpus(world: World, seed: int, stories_train: int = 1500, story_ks=(0, 4, 16, 64, 256), stories_test: int = 100) -> Corpus:
    rng = np.random.default_rng(10_000 + seed)
    sp, rg = world.species, world.regions
    T = world.value_token
    units: list[list[str]] = []
    pick = lambda group: TEMPLATES[group][int(rng.integers(len(TEMPLATES[group])))]
    # taxonomy and region hierarchy: every edge stated once
    for n in sp.order()[1:]:
        units.append(render(pick("isa"), X=world.sp_tok[n], Y=world.sp_tok[int(sp.parent[n])]))
    for n in rg.order()[1:]:
        units.append(render(pick("partof"), X=world.rg_tok[n], Y=world.rg_tok[int(rg.parent[n])]))
    leaves = [int(n) for n in sp.leaves]
    internal = [int(n) for n in sp.order() if n not in set(leaves)]
    sp_labels = {a: world.sp_labels(a) for a in SP_ATTRS}
    # family-level facts: generator override families, stated with prob 0.6
    stated_family: set[tuple[int, str]] = set()
    for a in SP_ATTRS:
        for n, v in world.sp_over[a].items():
            if n in internal and rng.random() < 0.6:
                stated_family.add((n, a))
                tpl = TEMPLATES["fact"] + TEMPLATES["family_fact"]
                units.append(render(tpl[int(rng.integers(len(tpl)))], X=world.sp_tok[n], A=a, V=T(a, v)))
                if rng.random() < 0.5:
                    p, ans = qa_tokens("qa1", world.sp_tok[n], a, T(a, v))
                    units.append(p + [ans, "."])
    # species cells: 50% stated, 10% validation, 40% test
    stated: set[tuple[int, str]] = set()
    val_single, test_single, val_rule, test_rule = [], [], [], []
    heldout_sentences, heldout_answer = [], []
    habitat_stated = {n for n in leaves if rng.random() < 0.85}
    for n in leaves:
        for a in ALL_SPECIES_ATTRS:
            truth = world.truth_species(n, a)
            u = rng.random()
            tok = world.sp_tok[n]
            if u < 0.5:
                stated.add((n, a))
                units.append(render(pick("fact"), X=tok, A=a, V=T(a, truth)))
                if rng.random() < 0.5:
                    kind = "qa1" if rng.random() < 0.7 else "qa2"
                    p, ans = qa_tokens(kind, tok, a, T(a, truth))
                    units.append(p + [ans, "."])
                continue
            if a in SP_ATTRS:
                parent_val = int(sp_labels[a][int(sp.parent[n])])
                inferable = parent_val == truth
            else:
                inferable = n not in world.rule_flip[a] and n in habitat_stated
            kind = "qa1" if rng.random() < 0.7 else "qa2"
            p, ans = qa_tokens(kind, tok, a, T(a, truth))
            item = QA(p, ans, {"node": n, "attr": a, "inferable": bool(inferable), "template": kind})
            target = (val_single if a in SP_ATTRS else val_rule) if u < 0.6 else (test_single if a in SP_ATTRS else test_rule)
            target.append(item)
            if u >= 0.6:
                sent = render(pick("fact"), X=tok, A=a, V=T(a, truth))
                heldout_sentences.append(sent)
                heldout_answer.append(len(sent) - 2)
    # habitats
    for n in leaves:
        if n in habitat_stated:
            units.append(render(pick("habitat"), X=world.sp_tok[n], R=world.rg_tok[world.habitat[n]]))
    # region facts: leaf cells 50%, zone overrides 60%
    rleaves = set(int(n) for n in rg.leaves)
    for a in REG_ATTRS:
        labels = world.rg_labels(a)
        for n in rg.order():
            n = int(n)
            if n in rleaves and rng.random() < 0.5:
                units.append(render(pick("fact"), X=world.rg_tok[n], A=a, V=T(a, int(labels[n]))))
            elif n not in rleaves and n in world.rg_over[a] and rng.random() < 0.6:
                units.append(render(pick("fact"), X=world.rg_tok[n], A=a, V=T(a, int(labels[n]))))
                if rng.random() < 0.5:
                    p, ans = qa_tokens("qa1", world.rg_tok[n], a, T(a, int(labels[n])))
                    units.append(p + [ans, "."])
    # habitat-hop questions: training for 40% of species with stated habitat
    val_hop, test_hop, test_hop_hidden = [], [], []
    for n in leaves:
        for a in REG_ATTRS:
            truth = world.truth_region(world.habitat[n], a)
            p, ans = qa_tokens("qah", world.sp_tok[n], a, T(a, truth))
            if n not in habitat_stated:
                test_hop_hidden.append(QA(p, ans, {"node": n, "attr": a}))
                continue
            u = rng.random()
            if u < 0.4:
                units.append(p + [ans, "."])
            elif u < 0.5:
                val_hop.append(QA(p, ans, {"node": n, "attr": a}))
            else:
                test_hop.append(QA(p, ans, {"node": n, "attr": a}))
    # stories: a habitat restated in context, distractors, then a hop question
    world_sentences = list(units)

    def story(k: int) -> tuple[list[str], QA]:
        n = leaves[int(rng.integers(len(leaves)))]
        region = int(rg.leaves[int(rng.integers(rg.leaves.shape[0]))])
        a = REG_ATTRS[int(rng.integers(len(REG_ATTRS)))]
        doc = render(TEMPLATES["story"][0], X=world.sp_tok[n], R=world.rg_tok[region])
        for _ in range(k):
            if rng.random() < 0.5:
                other = leaves[int(rng.integers(len(leaves)))]
                if other == n:
                    continue
                r2 = int(rg.leaves[int(rng.integers(rg.leaves.shape[0]))])
                doc += render(TEMPLATES["story"][0], X=world.sp_tok[other], R=world.rg_tok[r2])
            else:
                doc += world_sentences[int(rng.integers(len(world_sentences)))]
        truth = world.truth_region(region, a)
        p, ans = qa_tokens("qah", world.sp_tok[n], a, T(a, truth))
        return doc, QA(p, ans, {"node": n, "attr": a, "region": region, "k": k})

    for _ in range(stories_train):
        k = int(rng.integers(0, 5))
        doc, q = story(k)
        units.append(doc + q.prompt + [q.answer, "."])
    story_test = {k: [story(k) for _ in range(stories_test)] for k in story_ks}
    val = {"single": val_single, "rule": val_rule, "hop": val_hop}
    test = {"single": test_single, "rule": test_rule, "hop": test_hop, "hop_hidden_habitat": test_hop_hidden}
    return Corpus(world, units, val, test, heldout_sentences, heldout_answer, story_test, stated, stated_family, habitat_stated)


def split_sentences(unit: list[str]) -> list[list[str]]:
    out, cur = [], []
    for w in unit:
        cur.append(w)
        if w == ".":
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# ---------------------------------------------------------------------------
# Field LM
# ---------------------------------------------------------------------------


def compress_constrained(tree: Tree, leaf_obs: np.ndarray, constraints: dict[int, int], values: int) -> Program:
    """Minimal override program reproducing observed leaves and stated families."""

    big = 10**9
    cost = np.zeros((tree.size, values), dtype=np.int64)
    leaf_of = {int(node): i for i, node in enumerate(tree.leaves)}
    order = tree.order()
    for node in reversed(order):
        if node in leaf_of:
            obs = leaf_obs[leaf_of[node]]
            c = np.zeros(values, dtype=np.int64)
            if obs != UNKNOWN:
                c[:] = big
                c[obs] = 0
        else:
            c = np.zeros(values, dtype=np.int64)
            for child in tree.children[node]:
                cc = cost[child]
                c += np.minimum(cc, cc.min() + 1)
        if node in constraints:
            forced = np.full(values, big, dtype=np.int64)
            forced[constraints[node]] = c[constraints[node]]
            c = forced
        cost[node] = c
    labels = np.zeros(tree.size, dtype=int)
    overrides: dict[int, int] = {0: int(cost[0].argmin())}
    labels[0] = overrides[0]
    for node in order[1:]:
        up = labels[tree.parent[node]]
        cc = cost[node]
        best = int(cc.argmin())
        if cc[up] <= cc[best] + 1 and not (node in constraints and constraints[node] != up):
            labels[node] = up
        else:
            labels[node] = best
            overrides[node] = best
    return Program([overrides], labels[None, :])


def tree_from_parents(parent: dict[str, str], leaf_order: list[str] | None = None) -> tuple[Tree, dict[str, int]]:
    """Build a Tree (with 2-adic codes) from child -> parent token pairs."""

    children: dict[str, list[str]] = {}
    nodes = set(parent) | set(parent.values())
    for c, p in parent.items():
        children.setdefault(p, []).append(c)
    roots = [n for n in nodes if n not in parent]
    if len(roots) != 1:
        raise ValueError(f"taxonomy from text has {len(roots)} roots")
    order = [roots[0]]
    for n in order:
        order.extend(sorted(children.get(n, []), key=_natural))
    index = {t: i for i, t in enumerate(order)}
    par = np.full(len(order), -1)
    kids: list[list[int]] = [[] for _ in order]
    depth = np.zeros(len(order), dtype=int)
    for t in order[1:]:
        par[index[t]] = index[parent[t]]
        kids[index[parent[t]]].append(index[t])
        depth[index[t]] = depth[index[parent[t]]] + 1
    leaves_tok = [t for t in order if not children.get(t)]
    if leaf_order is not None:
        leaves_tok = [t for t in leaf_order if t in index and not children.get(t)]
    max_kids = max(len(k) for k in kids)
    bits = max(1, math.ceil(math.log2(max(2, max_kids))))
    code = np.zeros(len(order), dtype=np.int64)
    for t in order[1:]:
        i, p = index[t], index[parent[t]]
        code[i] = code[p] + kids[p].index(i) * (1 << (bits * (depth[p])))
    tree = Tree(par, kids, depth, np.array([index[t] for t in leaves_tok]), code, bits)
    return tree, index


def _natural(tok: str) -> tuple[str, int]:
    head = tok.rstrip("0123456789")
    tail = tok[len(head):]
    return (head, int(tail) if tail else -1)


class FieldLM:
    """Template front end + 2-adic family stores + learned question operations."""

    def __init__(self, lexicon: dict[str, str], smoothing: float = 0.03) -> None:
        self.lex = lexicon
        self.eps = smoothing
        self.vocab_by_type: dict[str, list[str]] = {}
        for tok, typ in lexicon.items():
            self.vocab_by_type.setdefault(typ, []).append(tok)
        self.overlay_habitat: dict[str, str] = {}
        self.abstained = 0

    # --- parsing ---------------------------------------------------------
    def signature(self, sent: list[str]) -> tuple[str, ...]:
        return tuple(w if self.lex.get(w, FUNC) == FUNC else self.lex[w] for w in sent)

    def slots(self, sent: list[str]) -> list[tuple[int, str, str]]:
        return [(i, w, self.lex[w]) for i, w in enumerate(sent) if self.lex.get(w, FUNC) != FUNC]

    def write_relation(self, sig: tuple[str, ...]) -> str | None:
        """Declarative relation from slot types (supplied); None for questions."""

        if sig[:2] == ("q", ":"):
            return None
        types = [t for t in sig if t in (SPECIES, FAMILY, REGION, ZONE, ATTR_S, ATTR_R, VALUE)]
        ent = [t for t in types if t in (SPECIES, FAMILY)]
        reg = [t for t in types if t in (REGION, ZONE)]
        if len(ent) == 2 and not reg and ATTR_S not in types:
            return "isa"
        if len(reg) == 2 and not ent:
            return "partof"
        if len(ent) == 1 and ATTR_S in types and VALUE in types:
            return "fact_s"
        if len(reg) == 1 and ATTR_R in types and VALUE in types and not ent:
            return "fact_r"
        if len(ent) == 1 and len(reg) == 1 and ent[0] == SPECIES and reg[0] == REGION:
            return "habitat"
        return None

    def parse_write(self, sent: list[str]) -> tuple[str, dict[str, str]] | None:
        rel = self.write_relation(self.signature(sent))
        if rel is None:
            return None
        s = self.slots(sent)
        get = lambda types: [w for _i, w, t in s if t in types]
        if rel == "isa":
            a, b = get((SPECIES, FAMILY))
            return rel, {"child": a, "parent": b}
        if rel == "partof":
            a, b = get((REGION, ZONE))
            return rel, {"child": a, "parent": b}
        if rel == "fact_s":
            return rel, {"entity": get((SPECIES, FAMILY))[0], "attr": get((ATTR_S,))[0], "value": get((VALUE,))[0]}
        if rel == "fact_r":
            return rel, {"entity": get((REGION, ZONE))[0], "attr": get((ATTR_R,))[0], "value": get((VALUE,))[0]}
        if rel == "habitat":
            return rel, {"species": get((SPECIES,))[0], "region": get((REGION,))[0]}
        return None

    # --- learning from text ------------------------------------------------
    def fit(self, units: list[list[str]]) -> "FieldLM":
        self.sig_count: dict[tuple[str, ...], int] = {}
        self.slot_unigram: dict[str, dict[str, int]] = {}
        sp_parent, rg_parent = {}, {}
        self.facts_s: dict[tuple[str, str], str] = {}
        self.facts_r: dict[tuple[str, str], str] = {}
        self.habitat: dict[str, str] = {}
        questions: list[tuple[list[str], list[str]]] = []
        for unit in units:
            sentences = split_sentences(unit)
            document = len(sentences) > 1  # multi-sentence documents write only to their own context
            for sent in sentences:
                sig = self.signature(sent)
                self.sig_count[sig] = self.sig_count.get(sig, 0) + 1
                for _i, w, t in self.slots(sent):
                    self.slot_unigram.setdefault(t, {})
                    self.slot_unigram[t][w] = self.slot_unigram[t].get(w, 0) + 1
                if sig[:2] == ("q", ":"):
                    questions.append((sent, unit if document else []))
                    continue
                if document:
                    continue
                parsed = self.parse_write(sent)
                if parsed is None:
                    continue
                rel, f = parsed
                if rel == "isa":
                    sp_parent[f["child"]] = f["parent"]
                elif rel == "partof":
                    rg_parent[f["child"]] = f["parent"]
                elif rel == "fact_s":
                    self.facts_s[(f["entity"], f["attr"])] = f["value"]
                elif rel == "fact_r":
                    self.facts_r[(f["entity"], f["attr"])] = f["value"]
                elif rel == "habitat":
                    self.habitat[f["species"]] = f["region"]
        species_leaves = sorted([t for t, typ in self.lex.items() if typ == SPECIES], key=_natural)
        region_leaves = sorted([t for t, typ in self.lex.items() if typ == REGION], key=_natural)
        self.sp_tree, self.sp_index = tree_from_parents(sp_parent, species_leaves)
        self.rg_tree, self.rg_index = tree_from_parents(rg_parent, region_leaves)
        self.sp_leaf_tokens = [t for t in species_leaves if t in self.sp_index]
        self.rg_leaf_tokens = [t for t in region_leaves if t in self.rg_index]
        self.value_index = {a: [f"{a}{v}" for v in range(NVAL)] for a in ALL_SPECIES_ATTRS + REG_ATTRS}
        self._build_region_programs()
        self._build_species_programs()
        self._abduce_hidden_habitats()
        self._learn_question_ops(questions)
        return self

    def _vid(self, attr: str, tok: str) -> int:
        return int(tok[len(attr):])

    def _build_region_programs(self) -> None:
        self.rg_prog: dict[str, Program] = {}
        for a in REG_ATTRS:
            obs = np.full(len(self.rg_leaf_tokens), UNKNOWN)
            for i, t in enumerate(self.rg_leaf_tokens):
                if (t, a) in self.facts_r:
                    obs[i] = self._vid(a, self.facts_r[(t, a)])
            cons = {self.rg_index[t]: self._vid(a, v) for (t, aa), v in self.facts_r.items()
                    if aa == a and self.lex[t] == ZONE}
            self.rg_prog[a] = compress_constrained(self.rg_tree, obs, cons, NVAL)

    def region_value(self, region_tok: str, attr: str) -> int:
        return int(self.rg_prog[attr].labels[0, self.rg_index[region_tok]])

    def _build_species_programs(self) -> None:
        """Per species attribute: inheritance program or rule on a habitat region attribute (MDL)."""

        self.sp_prog: dict[str, Program] = {}
        self.explain: dict[str, dict[str, Any]] = {}
        n = len(self.sp_leaf_tokens)
        exc_bits = math.log2(max(2, n)) + math.log2(NVAL)
        for a in ALL_SPECIES_ATTRS:
            obs = np.full(n, UNKNOWN)
            for i, t in enumerate(self.sp_leaf_tokens):
                if (t, a) in self.facts_s:
                    obs[i] = self._vid(a, self.facts_s[(t, a)])
            cons = {self.sp_index[t]: self._vid(a, v) for (t, aa), v in self.facts_s.items()
                    if aa == a and self.lex[t] == FAMILY}
            prog = compress_constrained(self.sp_tree, obs, cons, NVAL)
            self.sp_prog[a] = prog
            tree_bits = len(prog.overrides[0]) * (math.log2(self.sp_tree.size) + math.log2(NVAL))
            best = {"type": "tree", "bits": tree_bits}
            for j in REG_ATTRS:
                g = np.zeros(NVAL, dtype=int)
                support = np.zeros(NVAL, dtype=int)
                pairs = []
                for i, t in enumerate(self.sp_leaf_tokens):
                    if obs[i] != UNKNOWN and t in self.habitat:
                        pairs.append((self.region_value(self.habitat[t], j), obs[i]))
                if not pairs:
                    continue
                for v in range(NVAL):
                    hits = [y for x, y in pairs if x == v]
                    support[v] = len(hits)
                    g[v] = np.bincount(hits, minlength=NVAL).argmax() if hits else 0
                errors = sum(g[x] != y for x, y in pairs)
                # species with no known habitat fall back to an inheritance program
                # fit on those cells alone
                fallback_obs = np.full(n, UNKNOWN)
                for i, t in enumerate(self.sp_leaf_tokens):
                    if obs[i] != UNKNOWN and t not in self.habitat:
                        fallback_obs[i] = obs[i]
                fallback = compress_constrained(self.sp_tree, fallback_obs, {}, NVAL)
                fallback_bits = len(fallback.overrides[0]) * (math.log2(self.sp_tree.size) + math.log2(NVAL))
                bits = NVAL * math.log2(NVAL) + 1 + errors * exc_bits + fallback_bits
                if bits < best["bits"]:
                    best = {"type": "rule", "source": j, "g": g, "support": support, "bits": bits, "form": "table"}
                # affine rules y = m*x + k (mod NVAL), m odd: 5 bits, identified from few pairs
                for mult in (1, 3, 5, 7):
                    for shift in range(NVAL):
                        ga = np.array([(mult * x + shift) % NVAL for x in range(NVAL)])
                        err = sum(ga[x] != y for x, y in pairs)
                        abits = 5 + 1 + err * exc_bits + fallback_bits
                        if abits < best["bits"]:
                            best = {"type": "rule", "source": j, "g": ga, "support": np.full(NVAL, 1), "bits": abits,
                                    "form": f"affine:{mult}x+{shift}"}
            self.explain[a] = best
        # stated facts that the chosen explanation does not reproduce are exceptions
        self.exceptions: dict[tuple[str, str], int] = {}
        for (t, a), v in self.facts_s.items():
            if self.lex[t] != SPECIES:
                continue
            e = self.explain[a]
            if e["type"] == "rule" and t in self.habitat:
                if int(e["g"][self.region_value(self.habitat[t], e["source"])]) != self._vid(a, v):
                    self.exceptions[(t, a)] = self._vid(a, v)
            elif e["type"] == "rule":
                self.exceptions[(t, a)] = self._vid(a, v)

    def _abduce_hidden_habitats(self) -> None:
        self.abduced: dict[str, str] = {}
        rule_attrs = [a for a, e in self.explain.items() if e["type"] == "rule"]
        for t in self.sp_leaf_tokens:
            if t in self.habitat:
                continue
            scores = np.zeros(len(self.rg_leaf_tokens), dtype=int)
            for a in rule_attrs:
                if (t, a) not in self.facts_s:
                    continue
                e = self.explain[a]
                y = self._vid(a, self.facts_s[(t, a)])
                pred = np.array([e["g"][self.region_value(r, e["source"])] for r in self.rg_leaf_tokens])
                scores += pred == y
            self.abduced[t] = self.rg_leaf_tokens[int(scores.argmax())] if scores.max() > 0 else None

    def habitat_of(self, species_tok: str) -> str | None:
        if species_tok in self.overlay_habitat:
            return self.overlay_habitat[species_tok]
        return self.habitat.get(species_tok) or self.abduced.get(species_tok)

    # --- lookups ---------------------------------------------------------------
    def species_value(self, entity: str, attr: str) -> int | None:
        if entity not in self.sp_index:
            return None
        if (entity, attr) in self.exceptions:
            return self.exceptions[(entity, attr)]
        e = self.explain[attr]
        if e["type"] == "rule" and self.lex[entity] == SPECIES:
            hab = self.habitat_of(entity)
            if hab is not None:
                cause = self.region_value(hab, e["source"])
                if e["support"][cause] == 0:
                    self.abstained += 1
                    return None  # the consequence of an unseen cause is not identified
                return int(e["g"][cause])
        # family program lookup by dyadic phase characters (most specific family)
        node = self.sp_index[entity]
        prog = self.sp_prog[attr]
        tree = self.sp_tree
        value = execute_by_characters(tree, prog, 0, np.array([tree.code[node]]))[0]
        if value == UNKNOWN or tree.depth[node] < tree.depth.max():
            # families: the program label at the family node itself
            value = int(prog.labels[0, node])
        return int(value)

    def op_predict(self, op: str, slots: list[tuple[int, str, str]]) -> str | None:
        ent = [w for _i, w, t in slots if t in (SPECIES, FAMILY)]
        reg = [w for _i, w, t in slots if t in (REGION, ZONE)]
        attr = [w for _i, w, t in slots if t in (ATTR_S, ATTR_R)]
        if op == "fact_s" and ent and attr and attr[0] in ALL_SPECIES_ATTRS:
            v = self.species_value(ent[0], attr[0])
            return None if v is None else f"{attr[0]}{v}"
        if op == "fact_r" and reg and attr and attr[0] in REG_ATTRS:
            return f"{attr[0]}{self.region_value(reg[0], attr[0])}"
        if op == "hop" and ent and attr and attr[0] in REG_ATTRS:
            hab = self.habitat_of(ent[0])
            if hab is None:
                return None
            return f"{attr[0]}{self.region_value(hab, attr[0])}"
        if op == "isa" and ent:
            node = self.sp_index.get(ent[0])
            if node is None or self.sp_tree.parent[node] < 0:
                return None
            inv = {i: t for t, i in self.sp_index.items()}
            return inv[int(self.sp_tree.parent[node])]
        if op == "partof" and reg:
            node = self.rg_index.get(reg[0])
            if node is None or self.rg_tree.parent[node] < 0:
                return None
            inv = {i: t for t, i in self.rg_index.items()}
            return inv[int(self.rg_tree.parent[node])]
        if op == "habitat" and ent:
            return self.habitat_of(ent[0])
        return None

    def _learn_question_ops(self, questions: list[list[str]]) -> None:
        """Pick, per question template, the read operation that best predicts its answers."""

        ops = ("fact_s", "fact_r", "hop")
        tally: dict[tuple[str, ...], dict[str, int]] = {}
        for sent, unit in questions:
            self.read_context([unit] if unit else [])
            sig = self.signature(sent)
            s = self.slots(sent)
            answer = s[-1][1]
            tally.setdefault(sig, {op: 0 for op in ops})
            for op in ops:
                if self.op_predict(op, s[:-1]) == answer:
                    tally[sig][op] += 1
        self.overlay_habitat = {}
        self.question_op = {sig: max(t, key=t.get) for sig, t in tally.items() if max(t.values()) > 0}
        self.question_tally = {" ".join(k): v for k, v in tally.items()}

    # --- answering, context, next-token probabilities ------------------------
    def read_context(self, units: list[list[str]]) -> None:
        self.overlay_habitat = {}
        for unit in units:
            for sent in split_sentences(unit):
                parsed = self.parse_write(sent)
                if parsed and parsed[0] == "habitat":
                    self.overlay_habitat[parsed[1]["species"]] = parsed[1]["region"]

    def answer(self, prompt: list[str], context: list[str] | None = None) -> str | None:
        self.read_context([context] if context else [])
        sents = split_sentences(prompt)
        q = sents[-1]
        s = self.slots(q)
        op = self._question_op_for_prefix(q)
        out = self.op_predict(op, s) if op else None
        self.overlay_habitat = {}
        return out

    def _question_op_for_prefix(self, prefix: list[str]) -> str | None:
        psig = self.signature(prefix)
        for sig, op in self.question_op.items():
            if sig[: len(psig)] == psig:
                return op
        return None

    def answer_op_for(self, sig: tuple[str, ...]) -> str | None:
        rel = self.write_relation(sig)
        if rel in ("fact_s", "fact_r", "habitat", "isa", "partof"):
            return rel
        return self.question_op.get(sig)

    def sentence_logprobs(self, sent: list[str], context: list[str] | None = None) -> list[float]:
        """log2 p(token_i | prefix) for every token of one sentence."""

        self.read_context([context] if context else [])
        total = sum(self.sig_count.values())
        sigs = [(sig, c / total) for sig, c in self.sig_count.items()]
        out = []
        weights = {sig: math.log(p) for sig, p in sigs}
        for i, tok in enumerate(sent):
            dist: dict[str, float] = {}
            norm = 0.0
            mx = max(weights.values()) if weights else 0.0
            for sig, lw in weights.items():
                if i >= len(sig):
                    continue
                w = math.exp(lw - mx)
                norm += w
                for t, p in self._slot_distribution(sig, i, sent[:i]).items():
                    dist[t] = dist.get(t, 0.0) + w * p
            p_tok = dist.get(tok, 0.0) / norm if norm else 0.0
            p_tok = max(p_tok, 1e-6)  # floor for tokens outside every consistent template
            out.append(math.log2(p_tok))
            new = {}
            for sig, lw in weights.items():
                if i >= len(sig):
                    continue
                p = self._slot_distribution(sig, i, sent[:i]).get(tok, 0.0)
                if p > 0:
                    new[sig] = lw + math.log(p)
            weights = new or weights
        self.overlay_habitat = {}
        return out

    def _slot_distribution(self, sig: tuple[str, ...], i: int, prefix: list[str]) -> dict[str, float]:
        want = sig[i]
        if want not in (SPECIES, FAMILY, REGION, ZONE, ATTR_S, ATTR_R, VALUE):
            return {want: 1.0}
        is_answer = i == max(j for j, t in enumerate(sig) if t in (SPECIES, FAMILY, REGION, ZONE, ATTR_S, ATTR_R, VALUE))
        if is_answer:
            op = self.answer_op_for(sig)
            if op:
                pred = self.op_predict(op, self.slots(prefix))
                if pred is not None:
                    cands = self._candidates(want, prefix)
                    others = [c for c in cands if c != pred]
                    dist = {pred: 1.0 - self.eps}
                    for c in others:
                        dist[c] = self.eps / max(1, len(others))
                    return dist
        counts = self.slot_unigram.get(want, {})
        cands = self._candidates(want, prefix)
        tot = sum(counts.get(c, 0) + 1 for c in cands)
        return {c: (counts.get(c, 0) + 1) / tot for c in cands}

    def _candidates(self, typ: str, prefix: list[str]) -> list[str]:
        if typ == VALUE:
            attrs = [w for w in prefix if self.lex.get(w) in (ATTR_S, ATTR_R)]
            if attrs:
                return self.value_index[attrs[-1]]
        return self.vocab_by_type.get(typ, [])

    # --- editing ----------------------------------------------------------------
    def edit_family(self, entity: str, attr: str, value_tok: str) -> None:
        v = self._vid(attr, value_tok)
        if attr in REG_ATTRS:
            prog, tree, node = self.rg_prog[attr], self.rg_tree, self.rg_index[entity]
        else:
            prog, tree, node = self.sp_prog[attr], self.sp_tree, self.sp_index[entity]
        prog.overrides[0][node] = v
        labels = prog.labels[0]
        for n in tree.order():
            labels[n] = prog.overrides[0][n] if n in prog.overrides[0] else labels[tree.parent[n]]

    # --- size -------------------------------------------------------------------
    def bits(self) -> dict[str, float]:
        sp_nodes, rg_nodes = self.sp_tree.size, self.rg_tree.size
        taxonomy = (sp_nodes - 1) * math.log2(sp_nodes) + (rg_nodes - 1) * math.log2(rg_nodes)
        per = math.log2(sp_nodes) + math.log2(NVAL)
        programs = 0.0
        for a in ALL_SPECIES_ATTRS:
            e = self.explain[a]
            if e["type"] == "tree":
                programs += len(self.sp_prog[a].overrides[0]) * per
            else:
                programs += e["bits"]
        programs += sum(len(p.overrides[0]) for p in self.rg_prog.values()) * (math.log2(rg_nodes) + math.log2(NVAL))
        habitat = len(self.habitat) * math.log2(max(2, len(self.rg_leaf_tokens)))
        templates = len(self.sig_count) * 8
        return {"taxonomy": taxonomy, "programs": programs, "habitat": habitat, "templates": templates,
                "total": taxonomy + programs + habitat + templates}


def complete(lm: FieldLM, prefix: list[str], context: list[str] | None = None) -> str | None:
    """Predict the answer slot that follows a declarative or question prefix."""

    lm.read_context([context] if context else [])
    psig = lm.signature(prefix)
    slot_types = (SPECIES, FAMILY, REGION, ZONE, ATTR_S, ATTR_R, VALUE)
    best = None
    for sig, count in sorted(lm.sig_count.items(), key=lambda kv: -kv[1]):
        if len(sig) <= len(prefix) or sig[: len(prefix)] != psig:
            continue
        if sig[len(prefix)] not in slot_types:
            continue
        last = max(j for j, t in enumerate(sig) if t in slot_types)
        if last != len(prefix):
            continue
        op = lm.answer_op_for(sig)
        if op:
            best = lm.op_predict(op, lm.slots(prefix))
            if best is not None:
                break
    lm.overlay_habitat = {}
    return best


# ---------------------------------------------------------------------------
# Edits and probes (shared by the field LM and the baselines)
# ---------------------------------------------------------------------------


@dataclass
class Edit:
    kind: str  # "species" (family-level species fact) or "region" (zone-level region fact)
    entity: str
    attr: str
    new: str
    old: str
    edit_units: list[list[str]]  # the edit stated as text (for fine-tuning and text stores)
    probes: dict[str, list[tuple[list[str], str]]]  # probe group -> (prompt, expected token)
    locality: list[list[str]]  # prompts whose prediction should not change


def make_edits(world: World, corpus: Corpus, seed: int, n_species: int = 10, n_region: int = 10) -> list[Edit]:
    rng = np.random.default_rng(20_000 + seed)
    T = world.value_token
    sp, rg = world.species, world.regions
    leaves = [int(n) for n in sp.leaves]
    edits: list[Edit] = []
    fam_nodes = [int(n) for n in sp.order() if 1 <= sp.depth[n] <= 3]
    for _ in range(n_species):
        node = fam_nodes[int(rng.integers(len(fam_nodes)))]
        attr = SP_ATTRS[int(rng.integers(len(SP_ATTRS)))]
        before = world.sp_labels(attr)
        old = int(before[node])
        new = (old + 1 + int(rng.integers(NVAL - 1))) % NVAL
        over = dict(world.sp_over[attr])
        over[node] = new
        after = relabel_from_overrides(sp, over)
        ent = world.sp_tok[node]
        members = [leaves[i] for i in sorted(leaves_under(sp, node))]
        probes: dict[str, list] = {
            "efficacy": [qa_tokens("qa1", ent, attr, T(attr, new))],
            "paraphrase": [(["the", attr, "of", ent, "is"], T(attr, new)), (["members", "of", ent, "have", attr], T(attr, new))],
            "members_stated": [], "members_unstated": [], "members_kept": [],
        }
        for m in members:
            p = qa_tokens("qa1", world.sp_tok[m], attr, T(attr, int(after[m])))
            if after[m] != before[m]:
                probes["members_stated" if (m, attr) in corpus.stated_species_cells else "members_unstated"].append(p)
            else:
                probes["members_kept"].append(p)
        outside = [l for l in leaves if l not in set(members)]
        locality = [qa_tokens("qa1", world.sp_tok[outside[int(rng.integers(len(outside)))]], attr, "?")[0] for _ in range(20)]
        others = [a for a in SP_ATTRS if a != attr]
        locality += [qa_tokens("qa1", world.sp_tok[members[int(rng.integers(len(members)))]], others[int(rng.integers(len(others)))], "?")[0] for _ in range(20)]
        units = [render(t, X=ent, A=attr, V=T(attr, new)) for t in TEMPLATES["fact"] + TEMPLATES["family_fact"]]
        units.append(qa_tokens("qa1", ent, attr, T(attr, new))[0] + [T(attr, new), "."])
        edits.append(Edit("species", ent, attr, T(attr, new), T(attr, old), units, probes, locality))
    zones = [int(n) for n in rg.order() if 1 <= rg.depth[n] <= 2]
    stated_hab = sorted(corpus.stated_habitat)
    for _ in range(n_region):
        node = zones[int(rng.integers(len(zones)))]
        attr = REG_ATTRS[int(rng.integers(len(REG_ATTRS)))]
        before = world.rg_labels(attr)
        old = int(before[node])
        new = (old + 1 + int(rng.integers(NVAL - 1))) % NVAL
        over = dict(world.rg_over[attr])
        over[node] = new
        after = relabel_from_overrides(rg, over)
        ent = world.rg_tok[node]
        rule_attr = [a for a in RULE_ATTRS if RULE_SOURCE[a] == attr][0]
        probes = {
            "efficacy": [qa_tokens("qa1", ent, attr, T(attr, new))],
            "paraphrase": [(["the", attr, "of", ent, "is"], T(attr, new)), ([ent, "has", attr], T(attr, new))],
            "hop": [], "rule_stated": [], "rule_unstated": [],
        }
        affected_species = []
        for s in stated_hab:
            r = world.habitat[s]
            if after[r] != before[r]:
                affected_species.append(s)
                probes["hop"].append(qa_tokens("qah", world.sp_tok[s], attr, T(attr, int(after[r]))))
                if s not in world.rule_flip[rule_attr]:
                    val = int(world.rule_g[rule_attr][after[r]])
                    key = "rule_stated" if (s, rule_attr) in corpus.stated_species_cells else "rule_unstated"
                    probes[key].append(qa_tokens("qa1", world.sp_tok[s], rule_attr, T(rule_attr, val)))
        unaffected = [s for s in stated_hab if s not in set(affected_species)]
        locality = [qa_tokens("qah", world.sp_tok[unaffected[int(rng.integers(len(unaffected)))]], attr, "?")[0] for _ in range(20)]
        other_zones = [z for z in zones if z != node]
        locality += [qa_tokens("qa1", world.rg_tok[other_zones[int(rng.integers(len(other_zones)))]], REG_ATTRS[int(rng.integers(2))], "?")[0] for _ in range(20)]
        units = [render(t, X=ent, A=attr, V=T(attr, new)) for t in TEMPLATES["fact"]]
        units.append(qa_tokens("qa1", ent, attr, T(attr, new))[0] + [T(attr, new), "."])
        edits.append(Edit("region", ent, attr, T(attr, new), T(attr, old), units, probes, locality))
    return edits


def field_edit_copy(lm: FieldLM) -> FieldLM:
    import copy

    return copy.deepcopy(lm)
