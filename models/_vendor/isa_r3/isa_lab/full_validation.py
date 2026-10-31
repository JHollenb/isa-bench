"""Bounded executable ELF and uninterrupted-continuation probes on frozen models."""
from __future__ import annotations

import gzip
import importlib
import json
import struct
import tempfile
from pathlib import Path

from isa_lab.data import digest
from isa_lab.full_scope import serving
from isa_lab.measurement import Usage
from pan_isa.native import mount, stages
from pan_isa.runtime_adapter import SharedRuntimeModel


def elf_bytes(code):
    """Minimal static ELF64, independently consumed by the unchanged PAN loader."""
    ident = b"\x7fELF" + bytes((2, 1, 1, 0)) + bytes(8)
    header = struct.pack("<16sHHIQQQIHHHHHH", ident, 2, 62, 1, 0x400000, 64, 0, 0, 64, 56, 1, 0, 0, 0)
    phdr = struct.pack("<IIQQQQQQ", 1, 5, 0x1000, 0x400000, 0x400000, len(code), len(code), 0x1000)
    return header + phdr + bytes(0x1000 - len(header) - len(phdr)) + code


def static_elf(model, seed):
    mount()
    runtime_api = importlib.import_module("pan_isa_x86_runtime_spf.runtime")
    value = int(seed & 255)
    code = bytes.fromhex("b83c000000bf") + value.to_bytes(4, "little") + bytes.fromhex("0f05")
    with tempfile.TemporaryDirectory(prefix="isa-elf-") as directory:
        path = Path(directory) / "exit.elf"
        path.write_bytes(elf_bytes(code))
        runtime = runtime_api.SPFPrimaryPanRuntime.from_elf(path, model=SharedRuntimeModel(model))
        with Usage() as usage:
            stop = runtime.run(max_steps=8)
        return {"correct": stop.kind == "exit" and stop.exit_code == value, "stop": stop.to_json(),
                "expected_exit_code": value, "steps": runtime.steps, "usage": usage.result,
                "elf_sha256": digest(elf_bytes(code).hex()), "scope": "actual static ELF loader plus learned runtime and syscalls"}


def uninterrupted(model, consumer, seed, instructions=8192, out=None):
    stages()
    benchmark = importlib.import_module("pan_x86_stages.benchmark_stage3_uninterrupted")
    program = benchmark.make_uninterrupted_trace(instructions, seed, "heldout_boundary", 0x100000, True)
    initial = program["initial_stage3"]
    state = {"regs": list(initial["regs"]), "flags": list(initial["flags"]), "memory": list(initial["memory"])}
    correct, first_divergence, fields = 0, None, {}
    # Input-only stream and expected states are split before execution.
    stream = [serving(3, step) for step in program["steps"]]
    refs = [step["pan_stage3"] for step in program["steps"]]
    expected_states = []
    gold_memory = initial["memory"]
    for reference in refs:
        gold_memory = reference.get("after_memory", gold_memory)
        expected_states.append({"regs": reference["after_regs"], "flags": reference["after_flags"],
                                "memory": gold_memory, "exception": reference["exception"]})
    outputs = []
    with Usage() as usage:
        for instruction in stream:
            state, _choice = consumer.step(model, 3, instruction, state)
            outputs.append(state)
    if out is not None:
        out = Path(out)
        with gzip.open(out / "uninterrupted-program.json.gz", "wt") as writer:
            json.dump(program, writer, sort_keys=True)
        with gzip.open(out / "uninterrupted-predictions.jsonl.gz", "wt") as writer:
            for index, actual in enumerate(outputs):
                writer.write(json.dumps({"index": index, "actual": actual}, sort_keys=True) + "\n")
    for index, (actual, expected) in enumerate(zip(outputs, expected_states, strict=True)):
        ok = actual == expected
        correct += ok
        for key, value in expected.items():
            counts = fields.setdefault(key, [0, 0])
            counts[0] += actual[key] == value
            counts[1] += 1
        if not ok and first_divergence is None:
            first_divergence = {"step": index, "input": stream[index], "actual": actual, "reference": expected}
    # Native builder must not have inserted architectural state between steps.
    discontinuities = []
    previous_regs, previous_flags, previous_memory = initial["regs"], initial["flags"], initial["memory"]
    for index, reference in enumerate(refs):
        if reference["before_regs"] != previous_regs or reference["before_flags"] != previous_flags or reference.get("before_memory", previous_memory) != previous_memory:
            discontinuities.append(index)
        previous_regs, previous_flags = reference["after_regs"], reference["after_flags"]
        previous_memory = reference.get("after_memory", previous_memory)
    return {"steps": instructions, "correct": correct, "first_divergence": first_divergence,
            "fields": fields, "usage": usage.result, "program_sha256": digest(program),
            "input_discontinuities": discontinuities, "consumer": "pure predicted-state Stage 3 continuation; no input patches",
            "instrument_status": "valid" if not discontinuities else "reference-input-discontinuity",
            "terminal_status": "not-certified", "oracle": "native semantic reference; no new QEMU/Bochs claim"}
