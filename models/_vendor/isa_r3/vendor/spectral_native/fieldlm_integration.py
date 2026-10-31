"""Compose existing text, demonstration and joint Field modules at a rule ABI.

No new phase arithmetic, executor or learning algorithm is implemented here.
Record compilation receives observed text/trials and a room inventory only.
Rule meanings are canonical text-road codes; latent execution-road assignments
are applied afterward. Contradictions remain sentinels and read failures on
the executed path withhold answer authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import torch

from .demonstration_rules import infer_definitions, read_trials
from .joint_field_reader import ASSIGNMENTS, candidate_records_from_text
from .recurrent_rule import NONE_BEHAVIOR

BEHAVIOR_ABI = "field-canonical-text-roads-v1"


@dataclass(frozen=True)
class FieldMemory:
    records: dict
    road_bank: torch.Tensor
    valid_bank: bool
    evidence: tuple[dict, ...]
    signatures: tuple
    input_sha256: str
    behavior_abi: str = BEHAVIOR_ABI


def compile_memory(text, trials, *, width: int) -> FieldMemory:
    """Intersect statement and trial evidence before latent road assignment."""
    records, bank, diagnostic = candidate_records_from_text(text, width=width)
    parsed_trials = read_trials(trials, width)
    _, candidates = infer_definitions(records, parsed_trials)
    meanings = torch.full_like(records["defs"], NONE_BEHAVIOR)
    evidence = []
    for candidate in candidates:
        symbol = candidate["symbol"]
        stated = int(records["defs"][symbol])
        has_statement = stated != NONE_BEHAVIOR
        observed = set(candidate["behaviors"]) if candidate["n_trials"] else set(range(4))
        compatible = observed & ({stated} if has_statement else set(range(4)))
        identified = len(compatible) == 1 and (has_statement or candidate["n_trials"] > 0)
        if identified:
            meanings[symbol] = next(iter(compatible))
        status = "identified" if identified else "contradictory" if not compatible else "ambiguous"
        evidence.append(
            {
                "symbol": symbol,
                "statement": stated if has_statement else None,
                "trials": candidate["n_trials"],
                "compatible": sorted(compatible),
                "status": status,
            }
        )
    records = {**records, "defs": meanings}
    input_digest = hashlib.sha256(
        json.dumps({"text": text, "trials": trials, "width": width}, sort_keys=True).encode()
    ).hexdigest()
    return FieldMemory(
        records, bank, diagnostic["valid"], tuple(evidence), tuple(diagnostic["signatures"]), input_digest
    )


def local_rule_meanings(memory: FieldMemory, trials, *, assignment: tuple[int, int]) -> torch.Tensor:
    """Deliberately incompatible control: infer in the executor's local names."""
    records = {**memory.records, "edges": memory.road_bank[list(assignment)]}
    definitions, _ = infer_definitions(records, read_trials(trials, memory.road_bank.shape[-1]))
    return definitions


def authoritative_rollout(program, batch, *, budget: int = 16):
    """Inspect native reads, without repairing any transition or payload."""
    out = program.rollout(batch, budget=budget)
    steps = torch.arange(budget)[None, :]
    active = (out["stop_step"][:, None] < 0) | (steps <= out["stop_step"][:, None])
    unknown = ((out["rows"] == NONE_BEHAVIOR) & active).any(-1)
    authority = ~unknown & ~out["invalid_read"] & out["halted"]
    return out, authority, unknown


def selected_assignment(reader_logits):
    return ASSIGNMENTS[int(torch.as_tensor(reader_logits).argmax())]
