from __future__ import annotations

import hashlib
import json
import random

from .contracts import PROFILE, State
from .oracle import PythonReference, UnicornOracle
from .profile import FIXED_FORMS, IMMEDIATE_OPCODES, instruction_key

SPLITS = ("discovery", "feedback", "heldout", "terminal")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def generate(seed=17, examples_per_form=32, traces_per_split=4, oracle="unicorn"):
    if examples_per_form < 8 or traces_per_split < 1:
        raise ValueError("use at least eight discovery probes per form and one trace per split")
    backend = UnicornOracle() if oracle == "unicorn" else PythonReference() if oracle == "python" else None
    if backend is None:
        raise ValueError("oracle must be unicorn or python")
    forms = [bytes([opcode]) for opcode in IMMEDIATE_OPCODES] + list(FIXED_FORMS)
    splits = {}
    for partition, name in enumerate(SPLITS):
        rng = random.Random(seed * 1000 + partition)
        observations = []
        for form in forms:
            for index in range(examples_per_form):
                al = 4 * rng.randrange(64) + partition
                immediate = 4 * rng.randrange(64) + partition
                instruction = form + bytes([immediate]) if len(form) == 1 and form[0] != 0x90 else form
                before = State(al=al, bl=rng.randrange(256), cf=rng.randrange(2), zf=rng.randrange(2),
                               sf=rng.randrange(2), of=rng.randrange(2))
                observations.append({"id": f"{name}-{form.hex()}-{index}", "instruction": instruction.hex(),
                                     "before": before.to_dict(), "after": backend.step(instruction, before).to_dict()})
        traces = []
        for index in range(traces_per_split):
            initial = State(al=4 * rng.randrange(64) + partition, bl=rng.randrange(256), cf=rng.randrange(2))
            state = initial
            steps = []
            for _ in range((8, 16, 32, 64)[partition]):
                form = rng.choice(forms)
                instruction = (form + bytes([4 * rng.randrange(64) + partition])
                               if len(form) == 1 and form[0] != 0x90 else form)
                state = backend.step(instruction, state)
                steps.append({"instruction": instruction.hex(), "after": state.to_dict()})
            traces.append({"id": f"{name}-trace-{index}", "initial": initial.to_dict(), "steps": steps})
        splits[name] = {"observations": observations, "traces": traces}
    suite = {"schema": "isa-suite-v1", "profile": PROFILE, "seed": seed,
             "oracle": backend.authority,
             "split_policy": "imm8 and isolated AL residues modulo four; trace initial AL residue and length",
             "boundary": "fixed-byte forms overlap across splits; continuous intermediate AL is model/oracle-derived",
             "splits": splits}
    suite["sha256"] = digest(suite)
    validate_suite(suite)
    return suite


def validate_suite(suite):
    if suite.get("schema") != "isa-suite-v1" or suite.get("profile") != PROFILE:
        raise ValueError("unknown suite schema/profile")
    if suite.get("sha256") != digest({k: v for k, v in suite.items() if k != "sha256"}):
        raise ValueError("suite hash mismatch")
    if set(suite["splits"]) != set(SPLITS):
        raise ValueError("four quarantined splits are required")
    seen = set()
    trace_identities = set()
    for partition, name in enumerate(SPLITS):
        panel = suite["splits"][name]
        if not panel["observations"] or not panel["traces"]:
            raise ValueError("empty evidence panel")
        for row in panel["observations"]:
            instruction = bytes.fromhex(row["instruction"])
            key = instruction_key(instruction)
            before = State.from_dict(row["before"])
            State.from_dict(row["after"])
            if before.al % 4 != partition or (len(key) == 1 and len(instruction) == 2
                                              and instruction[1] % 4 != partition):
                raise ValueError("split residue violation")
            identity = digest({"instruction": row["instruction"], "before": row["before"]})
            if (identity, partition) in seen:
                # Repeated independent observations are allowed, not additional unique specimens.
                continue
            if any((identity, other) in seen for other in range(4) if other != partition):
                raise ValueError("cross-split observation leakage")
            seen.add((identity, partition))
        for trace in panel["traces"]:
            if not trace["steps"] or State.from_dict(trace["initial"]).al % 4 != partition:
                raise ValueError("invalid trace initial split")
            trace_identity = digest({"initial": trace["initial"],
                                     "instructions": [row["instruction"] for row in trace["steps"]]})
            if trace_identity in trace_identities:
                raise ValueError("duplicate trace evidence")
            trace_identities.add(trace_identity)
            for row in trace["steps"]:
                instruction = bytes.fromhex(row["instruction"])
                key = instruction_key(instruction)
                if len(key) == 1 and len(instruction) == 2 and instruction[1] % 4 != partition:
                    raise ValueError("trace immediate split residue violation")
                State.from_dict(row["after"])
    return suite
