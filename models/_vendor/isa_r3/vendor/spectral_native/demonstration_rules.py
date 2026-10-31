"""Rules inferred from observed transitions, with no target definition sentences.

The four-behavior hypothesis alphabet and trial sentence schema are supplied.
Trial inference receives read records and observed outcomes, never world truth.
Unidentified behaviors become sentinels; no permutation completion is used.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .recurrent_rule import NONE_BEHAVIOR, SYMBOLS
from .text_rule_world import TextRuleReader, TextWorld, compile_world, interpret_compiled, room, world_sentences

BITS = ("dark", "light")


@dataclass(frozen=True)
class Trial:
    start: int
    bit: int
    target: int
    target_bit: int
    halted: bool


def demonstration_sentences(world: TextWorld, *, order_seed: int):
    facts = world_sentences(world, "no_definitions", order_seed=order_seed)
    trials = []
    for symbol in range(SYMBOLS):
        rooms = [i for i in range(world.width) if int(world.symbols[i]) == symbol]
        if not rooms:
            continue
        # Distinct successor observations identify A vs B when possible.
        start = next((i for i in rooms if world.edges[0, i] != world.edges[1, i]), rooms[0])
        for bit in (0, 1):
            c = {"edges": world.edges, "symbols": world.symbols, "defs": world.defs}
            target, target_bit, halted, *_ = interpret_compiled(c, start, bit, max_steps=1)
            status = "stopped" if halted else "continued"
            trials.append(
                (
                    f"trial from {room(start)} holding {BITS[bit]} reached {room(target)} "
                    f"holding {BITS[target_bit]} and {status} ."
                ).split()
            )
    return facts, trials


def read_trials(sentences, width: int):
    inventory = {room(i): i for i in range(width)}
    read = []
    for sentence in sentences:
        if (
            len(sentence) != 12
            or sentence[0:2] != ["trial", "from"]
            or sentence[3] != "holding"
            or sentence[5] != "reached"
            or sentence[7] != "holding"
            or sentence[9] != "and"
            or sentence[-1] != "."
        ):
            continue
        try:
            if sentence[10] not in ("stopped", "continued"):
                continue
            read.append(
                Trial(
                    inventory[sentence[2]],
                    BITS.index(sentence[4]),
                    inventory[sentence[6]],
                    BITS.index(sentence[8]),
                    sentence[10] == "stopped",
                )
            )
        except (KeyError, ValueError, IndexError):
            continue
    return read


def infer_definitions(compiled: dict, trials: list[Trial]):
    candidates, definitions = [], torch.full((SYMBOLS,), NONE_BEHAVIOR, dtype=torch.long)
    for symbol in range(SYMBOLS):
        applicable = [trial for trial in trials if int(compiled["symbols"][trial.start]) == symbol]
        compatible = []
        for behavior in range(4):
            defs = torch.full((SYMBOLS,), NONE_BEHAVIOR, dtype=torch.long)
            defs[symbol] = behavior
            records = {**compiled, "defs": defs}
            matches = True
            for trial in applicable:
                predicted = interpret_compiled(records, trial.start, trial.bit, max_steps=1)
                if predicted is None or predicted[:3] != (trial.target, trial.target_bit, trial.halted):
                    matches = False
                    break
            if matches:
                compatible.append(behavior)
        if len(compatible) == 1 and applicable:
            definitions[symbol] = compatible[0]
        candidates.append({"symbol": symbol, "n_trials": len(applicable), "behaviors": compatible})
    return definitions, candidates


def compile_demonstrations(facts, trial_sentences, inventory_world):
    reader = TextRuleReader(facts)
    compiled = compile_world(reader, inventory_world)
    trials = read_trials(trial_sentences, inventory_world.width)
    definitions, candidates = infer_definitions(compiled, trials)
    compiled["defs"] = definitions
    return compiled, {"trials_read": len(trials), "candidate_behaviors": candidates}
