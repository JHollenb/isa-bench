from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

PROFILE = "x86-al8-v1"
TRACK = "candidate-selection-v1"
FLAGS = ("cf", "zf", "sf", "of")
ACTIONS = ("nop", "mov", "add", "sub", "xor", "and", "or", "cmp", "test", "inc", "dec")


@dataclass(frozen=True)
class State:
    al: int = 0
    bl: int = 0
    rip: int = 0x1000
    cf: int = 0
    zf: int = 0
    sf: int = 0
    of: int = 0

    def __post_init__(self):
        for name in ("al", "bl"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 255:
                raise ValueError(f"{name} must be an unsigned byte")
        if type(self.rip) is not int or not 0 <= self.rip < 2**64:
            raise ValueError("rip must be an unsigned 64-bit integer")
        for name in FLAGS:
            if type(getattr(self, name)) is not int or getattr(self, name) not in (0, 1):
                raise ValueError(f"{name} must be an integer bit")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("state fields do not match x86-al8-v1")
        return cls(**value)


@dataclass(frozen=True)
class Proposal:
    after: State
    selected_action: str
    evidence: dict


class Abstained(RuntimeError):
    """An explicit missing/ambiguous model route; retained per row by evaluation."""


class Model(Protocol):
    name: str

    def propose(self, instruction: bytes, before: State) -> Proposal: ...

    def export(self) -> dict: ...


def validate_commit(before: State, proposal: Proposal, instruction: bytes) -> State:
    if not isinstance(proposal, Proposal) or not isinstance(proposal.after, State):
        raise TypeError("invalid typed transition proposal")
    if proposal.selected_action not in ACTIONS:
        raise ValueError("unsupported action proposal")
    if proposal.after.bl != before.bl:
        raise ValueError("proposal writes preserved BL")
    if proposal.after.rip != (before.rip + len(instruction)) % 2**64:
        raise ValueError("proposal violates the linear profile's RIP contract")
    # This verifies mechanics/preserved state, never computes a gold answer.
    return proposal.after
