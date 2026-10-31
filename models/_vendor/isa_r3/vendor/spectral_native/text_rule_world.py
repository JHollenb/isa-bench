"""Text-stated rule worlds read by FieldLM2 and executed by the frozen rule-reading Field.

A world has ``width`` rooms, each with a left path (edge A), a back door
(edge B) and a sign (one of four local opcode symbols).  The world binds the
four signs to the global behaviors A / B / Flip / Stop with its own
permutation and states that binding in sentences.  Everything is written as
whitespace-token sentences; FieldLM2 is fitted on the world's text alone and
must induce the classes, templates and relations itself.

A declared compiler turns FieldLM2's answers into the rule Field's records:
room tokens -> node ids, the four target sign tokens -> symbol ids, and the
four behavior words -> global behavior codes.  It never repairs a model read
with generator truth: an answer outside the declared inventory becomes a
sentinel (edge -1, symbol ``NONE_SYMBOL``, behavior ``NONE_BEHAVIOR``).

Edges are random *functions* rather than permutations.  FieldLM2's relation
merge only checks functionality when a relation has a many-to-one key; two
bijective relations with the same argument classes (left path and back door
over rooms) would merge on range overlap alone.  The generator therefore
rejects worlds whose left-path or back-door map is injective (declared).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace

import torch

from .fieldlm_world2 import FieldLM2
from .recurrent_program import oracle_run
from .recurrent_rule import BEHAVIORS, NONE_BEHAVIOR, NONE_SYMBOL, SYMBOLS, RuleBatch

ROOM_PREFIX = "room"
SIGNS = tuple(f"glyph{i}" for i in range(SYMBOLS))
LEGEND_SIGNS = tuple(f"glyph{i}" for i in range(SYMBOLS, SYMBOLS + 8))
LEGEND_BEHAVIORS = (0, 1, 2, 3, 0, 1, 2, 3)  # two practice signs per behavior
BEHAVIOR_WORDS = ("walk-path", "walk-door", "flip", "halt")  # A, B, Flip, Stop

TEMPLATES = {
    "edge_a": ("the left path of {X} ends at {Y} .", "from {X} the left path runs to {Y} ."),
    "edge_b": ("{X} has a back door that opens on {Y} .", "through the back door of {X} lies {Y} ."),
    "sign": ("a stone in {X} bears {S} .", "the mark on {X} is {S} ."),
    "definition": ("the sign {S} means {V} .", "{S} is read as {V} ."),
}
WITHHELD_DEFINITION = "whoever meets {S} must {V} ."
PROMPTS = {
    "edge_a": "the left path of {X} ends at",
    "edge_b": "{X} has a back door that opens on",
    "sign": "a stone in {X} bears",
    "definition": "the sign {S} means",
    "definition_withheld_query": "in this land {S} means",
}
CONDITIONS = ("familiar", "withheld_query", "withheld_statement", "withheld_statement_support", "no_definitions")


def room(i: int) -> str:
    return f"{ROOM_PREFIX}{i}"


def fill(template: str, **slots: str) -> list[str]:
    return template.format(**slots).split()


@dataclass(frozen=True)
class TextWorld:
    edges: torch.Tensor  # (2, width) A/B successor
    symbols: torch.Tensor  # (width,) local sign id
    defs: torch.Tensor  # (4,) global behavior bound to each sign
    start: int
    bit: int
    moves: int

    @property
    def width(self) -> int:
        return int(self.edges.shape[1])


def sample_worlds(width: int, moves: int, count: int, seed: int, perms: tuple[tuple[int, ...], ...]) -> list[TextWorld]:
    """Fresh worlds with the requested stop depth; random non-injective A/B maps."""
    generator = torch.Generator().manual_seed(seed)
    perm_table = torch.tensor(perms)
    worlds: list[TextWorld] = []
    attempts = 0
    while len(worlds) < count and attempts < max(20_000, count * 20_000):
        attempts += 1
        edges = torch.randint(width, (2, width), generator=generator)
        behaviors = torch.randint(3, (width,), generator=generator)
        stop_nodes = torch.randperm(width, generator=generator)[: max(2, width // 4)]
        behaviors[stop_nodes] = 3
        start = int(torch.randint(width, (1,), generator=generator))
        bit = int(torch.randint(2, (1,), generator=generator))
        defs = perm_table[int(torch.randint(len(perms), (1,), generator=generator))].clone()
        if any(len(set(row.tolist())) == width for row in edges):
            continue  # declared: bijective path maps are indistinguishable to FieldLM2's merge rule
        inverse = torch.empty_like(defs)
        inverse[defs] = torch.arange(BEHAVIORS)
        symbols = inverse[behaviors]
        _node, _bit, halted, _nodes, _bits, actions = oracle_run(edges, defs[symbols], start, bit, max_steps=moves + 2)
        if halted and len(actions) - 1 == moves:
            worlds.append(TextWorld(edges, symbols, defs, start, bit, moves))
    if len(worlds) != count:
        raise RuntimeError(f"sampled {len(worlds)}/{count} worlds at width={width}, moves={moves}")
    return worlds


def interpret_world(world: TextWorld, *, max_steps: int = 16) -> tuple[int, int, bool, list[int], list[int], list[int]]:
    return oracle_run(world.edges, world.defs[world.symbols], world.start, world.bit, max_steps=max_steps)


def outcome(world: TextWorld) -> tuple[int, int, bool]:
    node, bit, halted, *_ = interpret_world(world)
    return node, bit, halted


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def world_sentences(world: TextWorld, condition: str, *, order_seed: int) -> list[list[str]]:
    """The world's text under one condition, one sentence per unit, shuffled."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    units: list[list[str]] = []
    for x in range(world.width):
        for kind, row in (("edge_a", 0), ("edge_b", 1)):
            for template in TEMPLATES[kind]:
                units.append(fill(template, X=room(x), Y=room(int(world.edges[row, x]))))
        for template in TEMPLATES["sign"]:
            units.append(fill(template, X=room(x), S=SIGNS[int(world.symbols[x])]))
    if condition != "no_definitions":
        for sign, behavior in zip(LEGEND_SIGNS, LEGEND_BEHAVIORS):
            for template in TEMPLATES["definition"]:
                units.append(fill(template, S=sign, V=BEHAVIOR_WORDS[behavior]))
        for s in range(SYMBOLS):
            word = BEHAVIOR_WORDS[int(world.defs[s])]
            if condition in ("withheld_statement", "withheld_statement_support"):
                units.append(fill(WITHHELD_DEFINITION, S=SIGNS[s], V=word))
            else:
                for template in TEMPLATES["definition"]:
                    units.append(fill(template, S=SIGNS[s], V=word))
        if condition == "withheld_statement_support":
            # Eight non-target examples: the practice legend restated in the withheld wording.
            for sign, behavior in zip(LEGEND_SIGNS, LEGEND_BEHAVIORS):
                units.append(fill(WITHHELD_DEFINITION, S=sign, V=BEHAVIOR_WORDS[behavior]))
    random.Random(order_seed).shuffle(units)
    return units


# ---------------------------------------------------------------------------
# Compiler: FieldLM2 answers -> rule-Field records (no truth repair)
# ---------------------------------------------------------------------------


def compile_world(model: FieldLM2, world: TextWorld, *, definition_prompt: str = "definition") -> dict:
    """Read every record the executor may need from FieldLM2 and build a RuleBatch."""
    width = world.width
    room_ids = {room(i): i for i in range(width)}
    sign_ids = {s: i for i, s in enumerate(SIGNS)}
    behavior_ids = {w: i for i, w in enumerate(BEHAVIOR_WORDS)}
    edges = torch.full((2, width), -1, dtype=torch.long)
    symbols = torch.full((width,), NONE_SYMBOL, dtype=torch.long)
    defs = torch.full((SYMBOLS,), NONE_BEHAVIOR, dtype=torch.long)
    answers: dict[str, list] = {"edge_a": [], "edge_b": [], "sign": [], "definition": []}
    for x in range(width):
        for kind, row in (("edge_a", 0), ("edge_b", 1)):
            token = model.answer(fill(PROMPTS[kind], X=room(x)))
            answers[kind].append(token)
            edges[row, x] = room_ids.get(token, -1)
        token = model.answer(fill(PROMPTS["sign"], X=room(x)))
        answers["sign"].append(token)
        symbols[x] = sign_ids.get(token, NONE_SYMBOL)
    for s in range(SYMBOLS):
        token = model.answer(fill(PROMPTS[definition_prompt], S=SIGNS[s]))
        answers["definition"].append(token)
        defs[s] = behavior_ids.get(token, NONE_BEHAVIOR)
    return {"edges": edges, "symbols": symbols, "defs": defs, "answers": answers}


def read_accuracy(compiled: dict, world: TextWorld) -> dict[str, list[int]]:
    return {
        "edge_a": [int((compiled["edges"][0] == world.edges[0]).sum()), world.width],
        "edge_b": [int((compiled["edges"][1] == world.edges[1]).sum()), world.width],
        "sign": [int((compiled["symbols"] == world.symbols).sum()), world.width],
        "definition": [int((compiled["defs"] == world.defs).sum()), SYMBOLS],
    }


def as_batch(worlds: list[TextWorld], compiled: list[dict] | None = None) -> RuleBatch:
    """RuleBatch whose records are compiled reads (or truth when ``compiled`` is None).

    Target and trace fields always hold generator truth for scoring; the
    executor never reads them.
    """
    truth = [interpret_world(w, max_steps=w.moves + 2) for w in worlds]
    moves = {w.moves for w in worlds}
    if len(moves) != 1:
        raise ValueError("a batch must share one stop depth")
    source = compiled or [{"edges": w.edges, "symbols": w.symbols, "defs": w.defs} for w in worlds]
    return RuleBatch(
        edges=torch.stack([c["edges"] for c in source]),
        symbols=torch.stack([c["symbols"] for c in source]),
        defs=torch.stack([c["defs"] for c in source]),
        starts=torch.tensor([w.start for w in worlds]),
        initial_bits=torch.tensor([w.bit for w in worlds]),
        target_nodes=torch.tensor([t[0] for t in truth]), target_bits=torch.tensor([t[1] for t in truth]),
        truth_nodes=torch.tensor([t[3] for t in truth]), truth_bits=torch.tensor([t[4] for t in truth]),
        truth_actions=torch.tensor([t[5] for t in truth]),
    )


def interpret_compiled(compiled: dict, start: int, bit: int, *, max_steps: int = 16):
    """Independent interpreter of a compiled graph; None if the path touches a sentinel."""
    edges, symbols, defs = compiled["edges"], compiled["symbols"], compiled["defs"]
    node, nodes, bits, actions = int(start), [int(start)], [int(bit)], []
    for _ in range(max_steps):
        symbol = int(symbols[node])
        if symbol == NONE_SYMBOL:
            return None
        behavior = int(defs[symbol])
        if behavior == NONE_BEHAVIOR:
            return None
        if behavior == 3:
            actions.append(2)
            return node, bit, True, nodes, bits, actions
        if behavior == 2:
            bit = 1 - bit
            action = bit
        else:
            action = behavior
        target = int(edges[action, node])
        if target < 0:
            return None
        actions.append(action)
        node = target
        nodes.append(node)
        bits.append(bit)
    return node, bit, False, nodes, bits, actions


def edited(world: TextWorld, **changes) -> TextWorld:
    return replace(world, **changes)


class TextRuleReader(FieldLM2):
    """FieldLM2 whose unseen-template reader also routes to plain (non-hierarchy) relations.

    FieldLM2's ``complete_unseen`` only considers hierarchy, link and attribute
    relations; every relation in these worlds is a plain functional relation
    ("other"), so an unseen question form could never be answered.  This
    extension applies the *same* anchor-specificity score to plain relations
    as well.  Nothing else changes.
    """

    def complete_unseen(self, q: list[str]) -> str | None:
        import math
        from collections import Counter, defaultdict

        slot_toks = [w for w in q if self.p.slot_class(w)]
        if len(slot_toks) != 1:
            return None
        key_cls = self.p.slot_class(slot_toks[0])
        anchors = [w for w in q if not self.p.slot_class(w) and w in self.p.cls]
        rel_tokens: dict[int, Counter] = defaultdict(Counter)
        for t in self.p.templates.values():
            r = self.p.rel_of_template.get(t.tid)
            if r is None:
                continue
            for x in t.skeleton:
                if not x.startswith("<"):
                    rel_tokens[r.rid][x] += 1
        df = Counter()
        for toks in rel_tokens.values():
            for x in toks:
                df[x] += 1
        nrel = max(1, len(rel_tokens))
        best = None
        for r in self.p.relations:
            if len(r.roles) != 2 or r.kind not in ("attribute", "link", "hierarchy", "other") or r.roles[r.key] != key_cls:
                continue
            # A fragmented template cluster may not contain this entity.
            # Route among relations that can actually supply the requested
            # fact, rather than choosing an empty exact-template sibling.
            answer = self.lookup(r, slot_toks[0])
            if answer is None:
                continue
            toks = rel_tokens.get(r.rid, Counter())
            score = sum(math.log(nrel / df[a]) for a in anchors if a in toks and df[a])
            templates = [self.p.by_id[tid] for tid in r.members]
            required = set.intersection(*[{x for x in t.skeleton if not x.startswith("<")} for t in templates]) if templates else set()
            score -= sum(math.log(nrel / df[x]) for x in sorted(required) if x not in q and df[x])
            if best is None or score > best[0]:
                best = (score, answer)
        if best is None or best[0] <= 0:
            return None
        return best[1]


WITHHELD_PROMPTS = {
    "edge_a": "leaving {X} by the left path you reach",
    "edge_b": "behind the back door of {X} you find",
    "sign": "the symbol carved in {X} is",
    "definition": "in this land {S} means",
}


def compile_world_withheld(model: FieldLM2, world: TextWorld) -> dict:
    """Same compiler, but every record is read through an unseen question wording."""
    width = world.width
    room_ids = {room(i): i for i in range(width)}
    sign_ids = {s: i for i, s in enumerate(SIGNS)}
    behavior_ids = {w: i for i, w in enumerate(BEHAVIOR_WORDS)}
    edges = torch.full((2, width), -1, dtype=torch.long)
    symbols = torch.full((width,), NONE_SYMBOL, dtype=torch.long)
    defs = torch.full((SYMBOLS,), NONE_BEHAVIOR, dtype=torch.long)
    answers: dict[str, list] = {"edge_a": [], "edge_b": [], "sign": [], "definition": []}
    for x in range(width):
        for kind, row in (("edge_a", 0), ("edge_b", 1)):
            token = model.answer(fill(WITHHELD_PROMPTS[kind], X=room(x)))
            answers[kind].append(token)
            edges[row, x] = room_ids.get(token, -1)
        token = model.answer(fill(WITHHELD_PROMPTS["sign"], X=room(x)))
        answers["sign"].append(token)
        symbols[x] = sign_ids.get(token, NONE_SYMBOL)
    for s in range(SYMBOLS):
        token = model.answer(fill(WITHHELD_PROMPTS["definition"], S=SIGNS[s]))
        answers["definition"].append(token)
        defs[s] = behavior_ids.get(token, NONE_BEHAVIOR)
    return {"edges": edges, "symbols": symbols, "defs": defs, "answers": answers}
