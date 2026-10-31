"""Axes, a relational web and a reasoning axis for compositions never seen in training.

Three mechanisms, one description-length learner over a training composition table:

- axes: relations become displacements on commuting integer coordinates; composing
  relations adds them, so unseen compositions follow from the coordinates alone;
- web: length-2 rewrite edges between atom letters carry the non-commuting residue
  (kinship: down.up = 1 while up.down = sibling);
- reasoning axis: a step coordinate carried by the working frontier, never by fact
  keys, lets one relation-variable read loop run chains of any length.

The mixed form marks some atom letters as axes (they commute with every letter) and
reduces the rest with web rules. Pure abelian and pure monoid are its two corners, so
description length decides how much reasoning moves into axes and how much stays web.
Reuses ``clutrr_field_v2.Algebra`` unchanged; relation names are opaque identifiers;
inverse pairs are read from reversed world facts. The hypothesis class (at most two
atoms, length-2 rules with length <= 1 right sides) is supplied and disclosed.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, product
from typing import Any, Callable, Iterable, Mapping, Sequence

from .clutrr_field_v2 import Algebra, _log2, table_bits
from .role_chart_field import Record, RoleChartField

Element = tuple
Table = dict[tuple[str, str], Counter]


# ---------------------------------------------------------------------------------------------
# ground-truth domains
# ---------------------------------------------------------------------------------------------


def _kin_compose(a: Element, b: Element) -> Element:
    """Bicyclic monoid on (up, down): a child's parent is the person, a parent's child a sibling."""
    (u1, d1), (u2, d2) = a, b
    return (u1 + max(0, u2 - d1), d2 + max(0, d1 - u2))


def _add(a: Element, b: Element) -> Element:
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _product_compose(a: Element, b: Element) -> Element:
    return (*_kin_compose(a[:2], b[:2]), a[2] + b[2])


@dataclass(frozen=True)
class Domain:
    name: str
    elements: tuple[Element, ...]
    compose: Callable[[Element, Element], Element]
    identity: Element
    withheld: Element  # the relation removed from training and taught by few shots
    note: str

    def fold(self, chain: Iterable[Element]) -> Element:
        out = self.identity
        for element in chain:
            out = self.compose(out, element)
        return out


def domains() -> dict[str, Domain]:
    kin = tuple((u, d) for u in range(3) for d in range(3))
    grid = tuple((x, y) for x in (-1, 0, 1) for y in (-1, 0, 1))
    prod = tuple((u, d, y) for u in range(2) for d in range(2) for y in (-1, 0, 1))
    return {
        "kinship": Domain("kinship", kin, _kin_compose, (0, 0), (2, 1),
                          "non-commuting: generations up/down with down.up = 1; withheld = parent's sibling"),
        "spatial": Domain("spatial", grid, _add, (0, 0), (1, 1),
                          "commuting: grid displacements; withheld = north-east"),
        "product": Domain("product", prod, _product_compose, (0, 0, 0), (1, 1, 1),
                          "kinship (up/down <= 1) x a north/south line; withheld = sibling one step north"),
    }


def opaque_names(domain: Domain, seed: int) -> dict[Element, str]:
    """Relation identifiers carry no structure: a seeded permutation of r00..rNN."""
    order = list(domain.elements)
    random.Random(f"{domain.name}:{seed}:names").shuffle(order)
    return {element: f"r{i:02d}" for i, element in enumerate(order)}


# ---------------------------------------------------------------------------------------------
# entity worlds (facts are what a reader would store; inverses are read from them)
# ---------------------------------------------------------------------------------------------


@dataclass
class World:
    domain: str
    entities: list[str]
    facts: dict[str, dict[str, set[str]]]  # relation name -> subject -> objects ("object is subject's r")

    def walk(self, start: str, chain: Sequence[str]) -> list[set[str]]:
        frontier, trail = {start}, []
        for relation in chain:
            frontier = set().union(*(self.facts[relation].get(x, set()) for x in frontier)) if frontier else set()
            trail.append(frontier)
        return trail

    def size(self, relations: Iterable[str] | None = None) -> int:
        keep = self.facts if relations is None else {r: self.facts[r] for r in relations}
        return sum(len(objs) for rows in keep.values() for objs in rows.values())


def _tree(roots: int, branching: int, generations: int) -> tuple[list[str], dict[str, str | None]]:
    people, parent, level = [], {}, []
    for r in range(roots):
        name = f"p{len(people)}"
        people.append(name)
        parent[name] = None
        level.append(name)
    for _ in range(generations - 1):
        nxt = []
        for p in level:
            for _ in range(branching):
                name = f"p{len(people)}"
                people.append(name)
                parent[name] = p
                nxt.append(name)
        level = nxt
    return people, parent


def _kin_targets(person: str, u: int, d: int, parent: Mapping[str, str | None],
                 children: Mapping[str, list[str]]) -> set[str]:
    node: str | None = person
    for _ in range(u):
        node = parent[node] if node is not None else None
    if node is None:
        return set()
    frontier = {node}
    for _ in range(d):
        frontier = {c for x in frontier for c in children.get(x, [])}
    return frontier


def build_world(domain: Domain, names: Mapping[Element, str]) -> World:
    facts: dict[str, dict[str, set[str]]] = {names[e]: defaultdict(set) for e in domain.elements}
    if domain.name in ("kinship", "product"):
        people, parent = _tree(*((2, 2, 5) if domain.name == "kinship" else (1, 2, 3)))
        children: dict[str, list[str]] = defaultdict(list)
        for child, par in parent.items():
            if par is not None:
                children[par].append(child)
    if domain.name == "kinship":
        entities = people
        for e in domain.elements:
            for x in people:
                facts[names[e]][x] |= _kin_targets(x, e[0], e[1], parent, children)
    elif domain.name == "spatial":
        cells = [(x, y) for x in range(5) for y in range(5)]
        entities = [f"c{x}_{y}" for x, y in cells]
        for e in domain.elements:
            for x, y in cells:
                tx, ty = x + e[0], y + e[1]
                if 0 <= tx < 5 and 0 <= ty < 5:
                    facts[names[e]][f"c{x}_{y}"].add(f"c{tx}_{ty}")
    else:
        line = range(4)
        entities = [f"{p}@{k}" for p in people for k in line]
        for e in domain.elements:
            for p in people:
                kin = _kin_targets(p, e[0], e[1], parent, children)
                for k in line:
                    if 0 <= k + e[2] < 4:
                        facts[names[e]][f"{p}@{k}"] |= {f"{q}@{k + e[2]}" for q in kin}
    return World(domain.name, entities, {r: {x: set(v) for x, v in rows.items() if v} for r, rows in facts.items()})


def inverse_from_facts(world: World, exclude: Iterable[str] = ()) -> dict[str, str]:
    """inv(r) = the relation whose facts are exactly r's facts reversed (Jaccard 1.0 only).

    Excluded relations (the one taught by few shots) are unavailable on both sides, so no
    inverse link can hand the learner a withheld relation's word.
    """
    excluded = set(exclude)
    pairs = {r: {(x, y) for x, objs in rows.items() for y in objs}
             for r, rows in world.facts.items() if r not in excluded}
    inv: dict[str, str] = {}
    for r, facts in sorted(pairs.items()):
        reverse = {(y, x) for x, y in facts}
        exact = sorted(q for q, other in pairs.items() if other == reverse)
        if exact:
            inv[r] = exact[0]
    return inv


# ---------------------------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------------------------


@dataclass
class Split:
    domain: str
    names: dict[Element, str]
    element_of: dict[str, Element]
    train: Table
    shots: list[tuple[str, str, str]]
    heldout_pairs: list[tuple[str, str, str]]
    withheld_pairs: list[tuple[str, str, str]]
    chains: list[dict[str, Any]]
    withheld: str


def make_split(domain: Domain, seed: int, *, holdout: float = 0.3, shots: int = 2,
               chains_per_length: int = 40, lengths: Sequence[int] = tuple(range(3, 11))) -> Split:
    names = opaque_names(domain, seed)
    element_of = {n: e for e, n in names.items()}
    rng = random.Random(f"{domain.name}:{seed}:split")
    star = names[domain.withheld]
    pairs = []
    for a, b in product(domain.elements, repeat=2):
        c = domain.compose(a, b)
        if c in names:
            pairs.append((names[a], names[b], names[c]))
    involving = [p for p in pairs if star in p]
    others = [p for p in pairs if star not in p]
    rng.shuffle(others)
    n_hold = round(holdout * len(others))
    heldout, train_pairs = sorted(others[:n_hold]), sorted(others[n_hold:])
    shot_pool = sorted(p for p in involving if p[0] == star and p[1] != star and p[2] != star)
    rng.shuffle(shot_pool)
    shot_rows = sorted(shot_pool[:shots])
    withheld_pairs = sorted(p for p in involving if p not in shot_rows)
    train: Table = {(a, b): Counter({c: 1}) for a, b, c in train_pairs + shot_rows}
    known = [n for n in element_of if n != star]
    chains = []
    for k in lengths:
        seen: set[tuple[str, ...]] = set()
        tries = 0
        while sum(1 for c in chains if c["length"] == k) < chains_per_length and tries < 200_000:
            tries += 1
            chain = tuple(rng.choice(known) for _ in range(k))
            if chain in seen:
                continue
            target = domain.fold(element_of[n] for n in chain)
            if target in names and names[target] != star:
                seen.add(chain)
                chains.append({"length": k, "chain": list(chain), "answer": names[target]})
    return Split(domain.name, names, element_of, train, shot_rows, heldout, withheld_pairs, chains, star)


# ---------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------


def _cyk(chain: Sequence[str], table: Mapping[tuple[str, str], set[str]]) -> set[str]:
    """Every bracketing a pair table can derive; associativity makes all correct ones agree."""
    n = len(chain)
    cell: dict[tuple[int, int], set[str]] = {(i, i + 1): {chain[i]} for i in range(n)}
    for width in range(2, n + 1):
        for i in range(n - width + 1):
            j, out = i + width, set()
            for m in range(i + 1, j):
                for x in cell[(i, m)]:
                    for y in cell[(m, j)]:
                        out |= table.get((x, y), set())
            cell[(i, j)] = out
    return cell[(0, n)]


class TableArm:
    form = "table"

    def __init__(self, train: Table) -> None:
        self.pairs = {k: set(v) for k, v in train.items()}

    def compose(self, chain: Sequence[str]) -> dict[str, Any]:
        if len(chain) == 1:
            return {"status": "resolved", "value": chain[0]}
        values = _cyk(chain, self.pairs)
        if len(values) == 1:
            return {"status": "resolved", "value": next(iter(values))}
        return {"status": "ambiguous" if values else "missing", "value": None, "candidates": sorted(values)}

    def state(self) -> dict[str, Any]:
        return {"form": self.form, "pairs": {f"{a}|{b}": sorted(v) for (a, b), v in sorted(self.pairs.items())}}


class WebArm(TableArm):
    """Web only: edges between relation names completed by associativity and a learned identity."""

    form = "web"

    def __init__(self, train: Table, max_rounds: int = 16) -> None:
        super().__init__(train)
        names = sorted({x for pair in self.pairs for x in pair} | {v for vs in self.pairs.values() for v in vs})
        left: Counter = Counter()
        bad: set[str] = set()
        for (a, b), vs in self.pairs.items():
            for side, e, other in ((0, a, b), (1, b, a)):
                if vs == {other}:
                    left[e] += 1
                else:
                    bad.add(e)
        identity = [e for e, n in left.items() if n >= 3 and e not in bad]
        self.identity = identity[0] if len(identity) == 1 else None
        if self.identity is not None:
            for a in names:
                self.pairs.setdefault((self.identity, a), set()).add(a)
                self.pairs.setdefault((a, self.identity), set()).add(a)
        self.rounds = 0
        for _ in range(max_rounds):
            added = self._associate(names)
            self.rounds += 1
            if not added:
                break

    def _associate(self, names: Sequence[str]) -> int:
        added = 0
        by_left: dict[str, list[tuple[str, set[str]]]] = defaultdict(list)
        for (b, c), ys in list(self.pairs.items()):
            by_left[b].append((c, ys))
        for (a, b), xs in list(self.pairs.items()):
            for c, ys in by_left.get(b, []):
                for x in list(xs):
                    for y in list(ys):
                        zs = self.pairs.get((x, c))
                        ws = self.pairs.get((a, y))
                        if zs and not (ws and zs <= ws):
                            self.pairs.setdefault((a, y), set()).update(zs)
                            added += 1
                        if ws and not (zs and ws <= zs):
                            self.pairs.setdefault((x, c), set()).update(ws)
                            added += 1
        return added

    def state(self) -> dict[str, Any]:
        return {**super().state(), "identity": self.identity, "rounds": self.rounds}


class MixedAlgebra(Algebra):
    """Commuting atom letters are axes (exponent vectors); the rest reduce by web rules."""

    def __init__(self, atoms: list, inv: dict, rules: dict | None = None, commute: Iterable[str] = ()) -> None:
        super().__init__("mixed", atoms, inv, rules)
        self.commute = frozenset(commute)

    def nf(self, w: tuple[str, ...]) -> tuple[str, ...]:
        exp: Counter = Counter()
        rest: list[str] = []
        for x in w:
            up = x.upper()
            if up not in self.commute:
                rest.append(x)
            elif self.inv_letter[x] == x:
                exp[up] = (exp[up] + 1) % 2
            else:
                exp[up] += 1 if x.isupper() else -1
        changed = True
        while changed:
            changed = False
            for i in range(len(rest) - 1):
                r = self.rules.get((rest[i], rest[i + 1]))
                if r is not None:
                    rest[i:i + 2] = list(r)
                    changed = True
                    break
        tail: list[str] = []
        for up in sorted(exp):
            e = exp[up]
            tail.extend([up] * e if e > 0 else [up.lower()] * (-e))
        return tuple(rest) + tuple(tail)

    def bits(self, table: Table, n_classes: int) -> dict[str, float]:
        out = super().bits(table, n_classes)
        out["model"] += len(self.atoms)  # one axis-or-web flag per atom
        out["total"] = out["model"] + out["data"]
        return out


def _make(form: str, atoms: list, inv: dict, rules: dict, commute: frozenset[str]) -> Algebra:
    if form == "mixed":
        return MixedAlgebra(atoms, inv, rules, commute)
    return Algebra(form, atoms, inv, rules)


def _greedy(alg: Algebra, table: Table, n_classes: int, max_rules: int) -> tuple[Algebra, float]:
    commute = getattr(alg, "commute", frozenset())
    letters = [x for x in alg.letters if x.upper() not in commute]
    rhs_options: list[tuple[str, ...]] = [()] + [(z,) for z in letters]
    cur = alg.bits(table, n_classes)["total"]
    for _ in range(max_rules):
        best: tuple[float, Algebra | None] = (cur, None)
        for x in letters:
            for y in letters:
                if (x, y) in alg.rules:
                    continue
                for rhs in rhs_options:
                    trial = _make(alg.form, alg.atoms, alg.inv_class, {**alg.rules, (x, y): rhs}, commute)
                    trial.derive(table)
                    t = trial.bits(table, n_classes)["total"]
                    if t < best[0] - 1e-9:
                        best = (t, trial)
        if best[1] is None:
            break
        cur, alg = best
    return alg, cur


def atom_candidates(table: Table, inv: Mapping[str, str]) -> list[str]:
    """One representative per inverse pair among relations in the training table."""
    seen = sorted({x for pair in table for x in pair} | {c for row in table.values() for c in row})
    out: list[str] = []
    for c in seen:
        if c in out or inv.get(c) in out:
            continue
        out.append(c)
    return out


def learn_forms(table: Table, inv: Mapping[str, str], *, max_atoms: int = 2, max_rules: int = 8,
                forms: Sequence[str] = ("abelian", "monoid", "mixed")) -> dict[str, dict[str, Any]]:
    """Best algebra per form by description length on the training table only."""
    classes = sorted({x for pair in table for x in pair} | {c for row in table.values() for c in row})
    n_classes = max(2, len(classes))
    cands = atom_candidates(table, inv)
    out: dict[str, dict[str, Any]] = {"table": {"bits": table_bits(table, n_classes), "algebra": None, "starts": 0}}
    for form in forms:
        best: tuple[float, Algebra | None] = (float("inf"), None)
        starts = 0
        for k in range(1, max_atoms + 1):
            for atoms in combinations(cands, k):
                probe = Algebra("abelian", list(atoms), dict(inv))
                uppers = [probe.letter[a] for a in atoms]
                subsets = ([frozenset()] if form != "mixed" else
                           [frozenset(s) for r in range(len(uppers) + 1) for s in combinations(uppers, r)])
                for commute in subsets:
                    alg = _make(form, list(atoms), dict(inv), {}, commute)
                    alg.derive(table)
                    starts += 1
                    if form == "abelian":
                        cur = alg.bits(table, n_classes)["total"]
                    else:
                        alg, cur = _greedy(alg, table, n_classes, max_rules)
                    if cur < best[0] - 1e-9:
                        best = (cur, alg)
        alg = best[1]
        out[form] = {"bits": alg.bits(table, n_classes) if alg else None, "algebra": alg, "starts": starts}
    return out


def mdl_choice(learned: Mapping[str, Mapping[str, Any]]) -> str:
    return min(learned, key=lambda f: (learned[f]["bits"]["total"], f))


class AlgebraArm:
    def __init__(self, alg: Algebra) -> None:
        self.alg = alg
        self.form = alg.form

    def word(self, relation: str) -> tuple[str, ...] | None:
        return self.alg.words.get(relation)

    def reduce(self, chain: Sequence[str]) -> tuple[str, ...] | None:
        w: tuple[str, ...] = ()
        for r in chain:
            if r not in self.alg.words:
                return None
            w = self.alg.nf(w + self.alg.words[r])
        return w

    def compose(self, chain: Sequence[str]) -> dict[str, Any]:
        w = self.reduce(chain)
        if w is None:
            return {"status": "missing", "value": None}
        value = self.alg.word2class.get(w)
        return {"status": "resolved" if value is not None else "unnamed", "value": value, "word": "".join(w)}

    def letter_relations(self) -> dict[str, str]:
        out = {}
        for atom in self.alg.atoms:
            up = self.alg.letter[atom]
            out[up] = atom
            lo = self.alg.inv_letter[up]
            if lo != up:
                out[lo] = self.alg.inv_class[atom]
        return out

    def state(self) -> dict[str, Any]:
        return algebra_state(self.alg)


def abduction_candidates(alg: Algebra, relation: str, rows: Iterable[tuple[str, str, str]],
                         max_len: int = 3) -> list[str]:
    """Every distinct normal-form word (length <= max_len) consistent with the rows naming ``relation``.

    More than one means the evidence underdetermines the new relation: the model can
    see that it does not yet know what the relation is, before any target is graded.
    """
    obs = [(a, b, c) for a, b, c in rows if relation in (a, b) and relation != c]
    words = [()]
    for _ in range(max_len):
        words = words + [w + (x,) for w in words for x in alg.letters if len(w) < max_len]
    found: set[tuple[str, ...]] = set()
    for w in {alg.nf(w) for w in words}:
        ok = bool(obs)
        for a, b, c in obs:
            left = w if a == relation else alg.words.get(a)
            right = w if b == relation else alg.words.get(b)
            if left is None or right is None or c not in alg.words or alg.nf(left + right) != alg.words[c]:
                ok = False
                break
        if ok:
            found.add(w)
    return sorted("".join(w) or "1" for w in found)


def algebra_state(alg: Algebra) -> dict[str, Any]:
    return {
        "form": alg.form,
        "atoms": list(alg.atoms),
        "inv": dict(sorted(alg.inv_class.items())),
        "rules": {f"{x}{y}": "".join(r) or "1" for (x, y), r in sorted(alg.rules.items())},
        "commute": sorted(getattr(alg, "commute", ())),
        "words": {c: "".join(w) or "1" for c, w in sorted(alg.words.items())},
        "word2class": {"".join(w) or "1": c for w, c in sorted(alg.word2class.items())},
    }


def algebra_from_state(state: Mapping[str, Any]) -> Algebra:
    def word(s: str) -> tuple[str, ...]:
        return () if s == "1" else tuple(s)
    rules = {(k[0], k[1]): word(v) for k, v in state["rules"].items()}
    alg = _make(state["form"], list(state["atoms"]), dict(state["inv"]), rules, frozenset(state["commute"]))
    alg.words = {c: word(w) for c, w in state["words"].items()}
    alg.word2class = {word(w): c for w, c in state["word2class"].items()}
    return alg


# ---------------------------------------------------------------------------------------------
# reasoning axis: relation-variable reads around the native RoleChartField consumer
# ---------------------------------------------------------------------------------------------


def _view(records: list[Record]) -> RoleChartField:
    view = RoleChartField.from_state({"charts": [], "classes": {}, "merge_log": [], "role_class_merges": [],
                                      "records": [], "active_axes": None})
    view.records = records
    return view


@dataclass
class ReasoningStore:
    """World facts as binary records; a query tape as (query, step, relation) records."""

    relation_ids: dict[str, int]
    world: RoleChartField
    tape: RoleChartField = field(default_factory=lambda: _view([]))
    step_in_fact_key: bool = False

    @classmethod
    def from_world(cls, world: World, relations: Iterable[str], *, step_in_fact_key: bool = False) -> ReasoningStore:
        ids = {r: i for i, r in enumerate(sorted(world.facts))}
        records = []
        for r in sorted(relations):
            for x, objs in sorted(world.facts[r].items()):
                for y in sorted(objs):
                    roles = (x, y, "0") if step_in_fact_key else (x, y)
                    records.append(Record("world", world.domain, ids[r], roles, f"{r}:{x}:{y}", len(records)))
        return cls(ids, _view(records), step_in_fact_key=step_in_fact_key)

    def write_tape(self, query: str, chain: Sequence[str]) -> None:
        for t, r in enumerate(chain):
            self.tape.records.append(Record("tape", query, 0, (query, str(t), r), f"{query}:{t}", t))
        self.tape._indices.clear()

    def run(self, query: str, start: str, *, budget: int = 64) -> dict[str, Any]:
        """Loop: read relation at step t from the tape, read it from the world, advance t."""
        frontier, trace, t = {start}, [], 0
        while t < budget:
            cell = self.tape.lookup(0, {0: query, 1: str(t)}, namespace="tape", scope=query)
            if not cell:
                break
            relation = cell[0].roles[2]  # the relation is a bound variable, not a program constant
            rid = self.relation_ids.get(relation)
            nxt: set[str] = set()
            for x in sorted(frontier):
                binding = {0: x, 2: str(t)} if self.step_in_fact_key else {0: x}
                if rid is not None:
                    nxt |= {rec.roles[1] for rec in self.world.lookup(rid, binding, namespace="world",
                                                                    scope=self.world_scope)}
            trace.append({"step": t, "relation": relation, "frontier": len(nxt)})
            frontier, t = nxt, t + 1
        return {"answer": sorted(frontier), "reads": t, "trace": trace}

    @property
    def world_scope(self) -> str:
        return self.world.records[0].scope if self.world.records else ""


def sample_queries(world: World, domain: Domain, names: Mapping[Element, str], seed: int, *,
                   per_length: int = 20, lengths: Sequence[int] = tuple(range(1, 11)),
                   exclude: Iterable[str] = ()) -> list[dict[str, Any]]:
    rng = random.Random(f"{domain.name}:{seed}:queries")
    relations = sorted(set(names.values()) - set(exclude))
    out: list[dict[str, Any]] = []
    for k in lengths:
        got, tries = 0, 0
        while got < per_length and tries < 100_000:
            tries += 1
            start = rng.choice(world.entities)
            chain = [rng.choice(relations) for _ in range(k)]
            trail = world.walk(start, chain)
            if all(trail):
                out.append({"id": f"q{len(out)}", "length": k, "start": start, "chain": chain,
                            "answer": sorted(trail[-1])})
                got += 1
    return out


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


__all__ = [
    "AlgebraArm", "Domain", "abduction_candidates", "MixedAlgebra", "ReasoningStore", "Split", "TableArm", "WebArm", "World",
    "algebra_from_state", "algebra_state", "build_world", "digest", "domains", "inverse_from_facts",
    "learn_forms", "make_split", "mdl_choice", "opaque_names", "sample_queries",
]
