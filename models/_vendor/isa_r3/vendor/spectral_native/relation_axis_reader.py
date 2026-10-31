"""A relation-axis reader: relation phrases are a record coordinate, not a relation partition.

FieldLM2 splits one surface form into several relation IDs by the lexical class of its
relation word, so fresh identities match every split and are dropped as ambiguous. Here
the relation phrase occupies its own role axis of one fact family:

- templates are induced by substitution (positions whose token varies are slots);
- mention identities are supplied (as in the role/grammar experiments), all other
  varying positions are phrase slots;
- paraphrase surfaces are aligned to a reference surface by shared fillers inside the
  same source document;
- one orientation bit (which entity role is the input of ``R``) is chosen from answered
  source tapes;
- equation surfaces (one phrase equals a composition of two) choose their slot reading
  from source equations that agree with the learned algebra;
- tape surfaces ("begin at X", "then the R") put the step coordinate in sentence order.

Records are stored in a ``RoleChartField`` view and read with its native ``lookup``.
The relation algebra comes from ``axis_web_reasoning`` and is learned from compositions
the reader itself observes in source documents.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import permutations
from typing import Any, Iterable, Mapping, Sequence

from .axis_web_reasoning import Algebra, Domain, World, _view
from .role_chart_field import Record, RoleChartField

Unit = list[str]
MIN_FILLERS = 3  # supplied: a position is a slot when at least this many distinct tokens fill it

FACT_SURFACES = (
    lambda x, r, y: f"{y} is the {r} of {x} .",
    lambda x, r, y: f"the {r} of {x} is {y} .",
    lambda x, r, y: f"{x} has {y} as {r} .",
)


def equation_sentence(x: str, lhs: str, first: str, second: str) -> str:
    """"the lhs of x is the second of the first of x": lhs = first then second."""
    return f"the {lhs} of {x} is the {second} of the {first} of {x} ."


def tape_sentences(start: str, chain: Sequence[str]) -> list[str]:
    return [f"begin at {start} ."] + [f"then the {r} ." for r in chain]


def answer_sentences(answers: Iterable[str]) -> list[str]:
    return [f"we reach {y} ." for y in sorted(answers)]


# ---------------------------------------------------------------------------------------------
# corpus
# ---------------------------------------------------------------------------------------------


@dataclass
class Document:
    doc: str
    units: list[Unit]
    entities: set[str]
    kind: str
    meta: dict[str, Any] = field(default_factory=dict)


def rename_world(world: World, prefix: str) -> tuple[World, dict[str, str]]:
    rename = {e: f"{prefix}{i}" for i, e in enumerate(world.entities)}
    facts = {r: {rename[x]: {rename[y] for y in objs} for x, objs in rows.items()} for r, rows in world.facts.items()}
    return World(world.domain, [rename[e] for e in world.entities], facts), rename


def fact_document(doc: str, world: World, relations: Iterable[str], rng: random.Random, *,
                  all_surfaces: bool) -> Document:
    units: list[Unit] = []
    for r in sorted(relations):
        for x, objs in sorted(world.facts[r].items()):
            for y in sorted(objs):
                surfaces = FACT_SURFACES if all_surfaces else (rng.choice(FACT_SURFACES),)
                units.extend(s(x, r, y).split() for s in surfaces)
    rng.shuffle(units)
    return Document(doc, units, set(world.entities), "facts")


# ---------------------------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    length: int
    entity_slots: tuple[int, ...]
    phrase_slots: tuple[int, ...]
    skeleton: tuple[tuple[int, str], ...]

    def matches(self, tokens: Sequence[str], entities: set[str]) -> bool:
        if len(tokens) != self.length:
            return False
        if any(tokens[i] != w for i, w in self.skeleton):
            return False
        return all(tokens[i] in entities or tokens[i] == "?" for i in self.entity_slots) and \
            all(tokens[i] not in entities for i in self.phrase_slots)

    def key(self) -> str:
        words = dict(self.skeleton)
        return " ".join(words.get(i, "<E>" if i in self.entity_slots else "<R>") for i in range(self.length))


class RelationAxisReader:
    """Induced surfaces + learned alignment; facts land on one family with a phrase axis."""

    def __init__(self, source: Sequence[Document]) -> None:
        self.surfaces = self._induce_surfaces(source)
        self.fact_surfaces = [s for s in self.surfaces if len(s.phrase_slots) == 1 and len(s.entity_slots) == 2]
        self.equation_surfaces = [s for s in self.surfaces if len(s.phrase_slots) == 3]
        self.step_surfaces = [s for s in self.surfaces if len(s.phrase_slots) == 1 and not s.entity_slots]
        self.begin_surfaces, self.answer_surfaces = self._tape_roles(source)
        self.alignment = self._align(source)
        self.orientation = 0
        self.equation_reading: dict[str, Any] | None = None
        self.view: RoleChartField = _view([])
        self.log: dict[str, Any] = {"surfaces": [s.key() for s in self.surfaces],
                                    "alignment": {k.key(): v for k, v in self.alignment.items()}}

    # -- induction ------------------------------------------------------------------------------
    @staticmethod
    def _induce_surfaces(source: Sequence[Document]) -> list[Surface]:
        groups: dict[tuple, list[Unit]] = defaultdict(list)
        for d in source:
            for u in d.units:
                ents = tuple(i for i, w in enumerate(u) if w in d.entities)
                groups[(len(u), ents)].append(u)
        surfaces = []
        for (length, ents), units in sorted(groups.items()):
            # a slot varies over >= MIN_FILLERS tokens; closed function words vary less and split surfaces
            varying = tuple(i for i in range(length)
                            if i not in ents and len({u[i] for u in units}) >= MIN_FILLERS)
            by_skeleton: dict[tuple, int] = Counter(
                tuple((i, u[i]) for i in range(length) if i not in ents and i not in varying) for u in units)
            for skeleton, n in by_skeleton.items():
                if n >= 3:
                    surfaces.append(Surface(length, ents, varying, skeleton))
        return surfaces

    def _tape_roles(self, source: Sequence[Document]) -> tuple[list[Surface], list[Surface]]:
        """Single-entity surfaces that open answered source tapes vs those that close them."""
        singles = [s for s in self.surfaces if not s.phrase_slots and len(s.entity_slots) == 1]
        first, last = Counter(), Counter()
        for d in source:
            if d.kind != "tape" or not d.units:
                continue
            for s in singles:
                if s.matches(d.units[0], d.entities):
                    first[s] += 1
                if s.matches(d.units[-1], d.entities):
                    last[s] += 1
        begin = [s for s in singles if first[s] > last[s]]
        answer = [s for s in singles if last[s] > first[s]]
        return begin, answer

    def _align(self, source: Sequence[Document]) -> dict[Surface, tuple[int, int]]:
        """Entity-slot order of each fact surface relative to the most frequent one."""
        counts = Counter()
        seen: dict[Surface, list[tuple[str, str, str, str]]] = defaultdict(list)
        for d in source:
            for u in d.units:
                for s in self.fact_surfaces:
                    if s.matches(u, d.entities):
                        a, b = (u[i] for i in s.entity_slots)
                        seen[s].append((d.doc, u[s.phrase_slots[0]], a, b))
                        counts[s] += 1
                        break
        if not counts:
            return {}
        ref = max(counts, key=lambda s: (counts[s], s.key()))
        index = defaultdict(set)
        for doc, r, a, b in seen[ref]:
            index[(doc, r)].add((a, b))
        out = {ref: (0, 1)}
        for s in self.fact_surfaces:
            if s == ref:
                continue
            votes = Counter()
            for doc, r, a, b in seen[s]:
                if a == b:
                    continue
                pool = index.get((doc, r), set())
                votes[(0, 1)] += (a, b) in pool
                votes[(1, 0)] += (b, a) in pool
            out[s] = max(votes, key=lambda k: (votes[k], k)) if votes else (0, 1)
        self.reference = ref
        return out

    # -- reading ----------------------------------------------------------------------------------
    def read_fact(self, tokens: Sequence[str], entities: set[str]) -> tuple[str, str, str] | None:
        for s in self.fact_surfaces:
            if s.matches(tokens, entities):
                ents = [tokens[i] for i in s.entity_slots]
                a, b = (ents[i] for i in self.alignment[s])
                return tokens[s.phrase_slots[0]], a, b
        return None

    def ingest(self, doc: Document) -> dict[str, int]:
        added = 0
        for pos, u in enumerate(doc.units):
            fact = self.read_fact(u, doc.entities)
            if fact is not None:
                self.view.records.append(Record("fact", doc.doc, 0, fact, " ".join(u), pos))
                added += 1
        self.view._indices.clear()
        return {"units": len(doc.units), "facts": added}

    def relation(self, phrase: str, x: str, scope: str) -> set[str]:
        """R(x): targets of phrase from x under the learned orientation, via native lookup."""
        src, dst = (1, 2) if self.orientation == 0 else (2, 1)
        return {rec.roles[dst] for rec in self.view.lookup(0, {0: phrase, src: x}, namespace="fact", scope=scope)}

    def phrases(self, scope: str | None = None) -> set[str]:
        return {rec.roles[0] for rec in self.view.records if scope is None or rec.scope == scope}

    def read_tape(self, doc: Document) -> tuple[str | None, list[str], set[str]]:
        start, chain, answers = None, [], set()
        for u in doc.units:  # sentence order is the step coordinate
            for s in self.begin_surfaces:
                if s.matches(u, doc.entities):
                    start = u[s.entity_slots[0]]
            for s in self.step_surfaces:
                if s.matches(u, doc.entities):
                    chain.append(u[s.phrase_slots[0]])
            for s in self.answer_surfaces:
                if s.matches(u, doc.entities):
                    answers.add(u[s.entity_slots[0]])
        return start, chain, answers

    def read_equations(self, doc: Document) -> list[tuple[str, str, str]]:
        """(lhs, first, second) under the learned reading."""
        out = []
        if self.equation_reading is None:
            return out
        for u in doc.units:
            for s in self.equation_surfaces:
                if s.matches(u, doc.entities):
                    slots = [u[i] for i in s.phrase_slots]
                    lhs, first, second = (slots[i] for i in self.equation_reading["order"])
                    out.append((lhs, first, second))
        return out

    # -- learning from source feedback ----------------------------------------------------------
    def walk(self, start: str, chain: Sequence[str], scope: str) -> set[str]:
        frontier = {start}
        for r in chain:
            frontier = set().union(*(self.relation(r, x, scope) for x in frontier)) if frontier else set()
        return frontier

    def learn_orientation(self, tapes: Sequence[tuple[Document, str]]) -> dict[str, int]:
        """Choose the one-bit input/output convention from answered source tapes."""
        score = {}
        for bit in (0, 1):
            self.orientation = bit
            ok = 0
            for doc, scope in tapes:
                start, chain, answers = self.read_tape(doc)
                ok += start is not None and self.walk(start, chain, scope) == answers
            score[bit] = ok
        self.orientation = max(score, key=lambda b: (score[b], -b))
        self.log["orientation_scores"] = score
        return score

    def learn_equation_reading(self, source: Sequence[Document], alg: Algebra) -> dict[str, Any]:
        """Which phrase slot is the defined relation and which order composes the other two."""
        rows = []
        for doc in source:
            for u in doc.units:
                for s in self.equation_surfaces:
                    if s.matches(u, doc.entities):
                        rows.append([u[i] for i in s.phrase_slots])
        scores = {}
        for order in permutations(range(3)):
            ok = 0
            for slots in rows:
                lhs, first, second = (slots[i] for i in order)
                if all(p in alg.words for p in (lhs, first, second)):
                    ok += alg.nf(alg.words[first] + alg.words[second]) == alg.words[lhs]
            scores[order] = ok
        best = max(scores, key=lambda o: (scores[o], [-i for i in o]))
        self.equation_reading = {"order": list(best), "scores": {"".join(map(str, k)): v for k, v in scores.items()},
                                 "source_equations": len(rows)}
        return self.equation_reading


def observed_compositions(reader: RelationAxisReader, scope: str, entities: Sequence[str],
                          phrases: Sequence[str]) -> dict[tuple[str, str], Counter]:
    """(r1, r2) -> r3 whenever r2(r1(x)) equals r3(x) as a non-empty set, over the reader's records."""
    cache = {(r, x): reader.relation(r, x, scope) for r in phrases for x in entities}
    table: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for x in entities:
        for r1 in phrases:
            ys = cache[(r1, x)]
            if not ys:
                continue
            for r2 in phrases:
                z = set().union(*(cache.get((r2, y)) or reader.relation(r2, y, scope) for y in ys))
                if not z:
                    continue
                for r3 in phrases:
                    if cache[(r3, x)] == z:
                        table[(r1, r2)][r3] += 1
    return dict(table)


def reader_world(reader: RelationAxisReader, scope: str, entities: Sequence[str], domain: str) -> World:
    facts: dict[str, dict[str, set[str]]] = defaultdict(dict)
    for r in sorted(reader.phrases(scope)):
        for x in entities:
            ys = reader.relation(r, x, scope)
            if ys:
                facts[r][x] = ys
    return World(domain, list(entities), dict(facts))


def extensional_candidates(alg: Algebra, reader: RelationAxisReader, relation: str, anchors: Sequence[str],
                           letters: Mapping[str, str], *, example_scope: str, fact_scope: str,
                           max_len: int = 3) -> list[str]:
    """Words whose walk over atom facts reproduces the stated facts of ``relation`` at each anchor."""
    words = [()]
    for _ in range(max_len):
        words = words + [w + (x,) for w in words for x in alg.letters if len(w) < max_len]
    targets = {a: reader.relation(relation, a, example_scope) for a in anchors}
    found = set()
    for w in {alg.nf(w) for w in words}:
        if all(reader.walk(a, [letters[x] for x in w], fact_scope) == targets[a] for a in anchors):
            found.add(w)
    return sorted("".join(w) or "1" for w in found)


__all__ = [
    "FACT_SURFACES", "Document", "RelationAxisReader", "Surface", "answer_sentences", "equation_sentence",
    "extensional_candidates", "fact_document", "observed_compositions", "reader_world", "rename_world",
    "tape_sentences",
]
