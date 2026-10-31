"""Supplied legal-form parsing and shared exact candidate semantics.

The parser exposes instruction boundaries/operands, not a semantic model label.
Every candidate-selection model uses this same declared library. Direct-effect
models must emit their own values through the common commit contract instead.
"""
from __future__ import annotations

from dataclasses import replace

from .contracts import ACTIONS, PROFILE, Proposal, State

IMMEDIATE_OPCODES = (0xB0, 0x04, 0x2C, 0x34, 0x24, 0x0C, 0x3C, 0xA8)
FIXED_FORMS = (bytes([0x90]), bytes([0xFE, 0xC0]), bytes([0xFE, 0xC8]))


def instruction_key(instruction: bytes) -> bytes:
    if type(instruction) is not bytes:
        raise ValueError("instruction must be bytes")
    if len(instruction) == 2 and instruction[0] in IMMEDIATE_OPCODES:
        return instruction[:1]
    if instruction in FIXED_FORMS:
        return instruction
    raise ValueError(f"unsupported bytes for {PROFILE}: {instruction.hex()}")


def candidate(action: str, instruction: bytes, state: State) -> State:
    instruction_key(instruction)
    if action not in ACTIONS:
        raise ValueError("unknown candidate")
    a = state.al
    b = instruction[1] if instruction[0] in IMMEDIATE_OPCODES else 0
    after = replace(state, rip=(state.rip + len(instruction)) % 2**64)
    if action == "nop":
        return after
    if action == "mov":
        return replace(after, al=b)
    if action in ("add", "inc"):
        b = 1 if action == "inc" else b
        value = (a + b) & 255
        cf = state.cf if action == "inc" else int(a + b > 255)
        of = int(bool((~(a ^ b) & (a ^ value)) & 128))
    elif action in ("sub", "cmp", "dec"):
        b = 1 if action == "dec" else b
        value = (a - b) & 255
        cf = state.cf if action == "dec" else int(a < b)
        of = int(bool(((a ^ b) & (a ^ value)) & 128))
    else:
        value = {"xor": a ^ b, "and": a & b, "or": a | b, "test": a & b}[action]
        cf = of = 0
    return replace(after, al=a if action in ("cmp", "test") else value,
                   cf=cf, zf=int(value == 0), sf=int(bool(value & 128)), of=of)


def proposal(action: str, instruction: bytes, before: State, evidence: dict) -> Proposal:
    return Proposal(candidate(action, instruction, before), action, evidence)


def infer_routes(discovery: list[dict]) -> dict[str, str]:
    """Learn label assignments from discovery transitions, not opcode names.

    Intersect all candidates consistent with observations of each legal byte key.
    Unidentified or inconsistent keys remain explicit errors in this starter fit.
    """
    possible = {}
    for row in discovery:
        instruction = bytes.fromhex(row["instruction"])
        key = instruction_key(instruction).hex()
        before, after = State.from_dict(row["before"]), State.from_dict(row["after"])
        matches = {action for action in ACTIONS if candidate(action, instruction, before) == after}
        possible[key] = possible.get(key, set(ACTIONS)) & matches
    routes = {}
    for key, choices in possible.items():
        if len(choices) != 1:
            raise ValueError(f"discovery does not identify one route for {key}: {sorted(choices)}")
        routes[key] = next(iter(choices))
    return routes
