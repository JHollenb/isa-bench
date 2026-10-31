"""Frozen serving benchmark: no oracle scoring in the timed consumer."""
from __future__ import annotations

from .contracts import State, validate_commit
from .data import digest
from .measurement import Usage


def serve(model, panel):
    outputs, attempted = [], 0
    for row in panel["observations"]:
        before = State.from_dict(row["before"])
        instruction = bytes.fromhex(row["instruction"])
        after = validate_commit(before, model.propose(instruction, before), instruction)
        outputs.append(after.to_dict())
        attempted += 1
    for trace in panel["traces"]:
        state = State.from_dict(trace["initial"])
        for row in trace["steps"]:
            instruction = bytes.fromhex(row["instruction"])
            state = validate_commit(state, model.propose(instruction, state), instruction)
            outputs.append(state.to_dict())
            attempted += 1
    return outputs, attempted


def measure(model, panel, *, warmups=1, repeats=2):
    if warmups < 1 or repeats < 1:
        raise ValueError("warmups and repetitions must be positive")
    expected = None
    for _ in range(warmups):
        outputs, _ = serve(model, panel)
        expected = digest(outputs)
    rows = []
    for index in range(repeats):
        with Usage() as usage:
            outputs, count = serve(model, panel)
        actual = digest(outputs)
        if actual != expected:
            raise RuntimeError("frozen serving changed between warmup and timed replay")
        rows.append({"repeat": index, "instructions": count, "outputs_sha256": actual,
                     "us_per_instruction": usage.result["wall_s"] * 1e6 / count,
                     "instructions_per_s": count / usage.result["wall_s"], **usage.result})
    return rows
