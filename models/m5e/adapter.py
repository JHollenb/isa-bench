"""M5e structured learned machine (isa feat/m5e @ 108c04f): M5d-final v3 + byte-register/SIB-index guards, the shared
effective-address unit (operation x address composition where no whole record matches), flag refinement on fresh rows
of dev stops, fitted no-record stops.

The machine is a byte-keyed trie of learned records (fields, immediate runs and an
effect in a small language: per-bit automata, one learned adder, flag-tap tables,
branch predicates). It abstains wherever its version space disagrees. Supplied:
fetch, the paged memory store, the native initial image and Linux write/exit.
"""
from __future__ import annotations

import json
import pickle

from isa_bench.adapters import Base, finish
from isa_bench.state import ArchState

M4_FLAGS = ("CF", "PF", "AF", "ZF", "SF", "OF")


class Adapter(Base):
    family = "m5e"

    def load(self):
        self.add_paths(self.code_dir() / "src")
        from direct_effect import m4_run
        from direct_effect.m4_scalar import Scalar

        with open(self.artifact / "machine_full.pkl", "rb") as f:
            machine = pickle.load(f)
        self.m, self.dec = machine["m"], machine["dec"]
        self.scalar = Scalar(self.m)
        self.R = m4_run
        records = getattr(self.dec, "records", None)
        self.metadata = {**self.base_metadata(), "parameter_count": 0,
                         "learned_records": len(records) if records is not None else None,
                         "implementation": "M5e structured machine (records + SPF automata + EA unit), runtime 108c04f"}

    def run(self, program_dir, trace, max_steps):
        R = self.R
        pages, _ = R.load_image(program_dir / "mem.bin")
        mem = R.ProgMem(pages)
        first = trace.steps[0]
        state = {"rip": int(first[0]), "regs": [int(v) for v in first[1:17]], "flags": R.eflags_to_flags(first[17])}
        rflags_rest = int(first[17]) & ~R.STATUS_MASK
        cmp, stdout, stop = self.comparator(trace), bytearray(), None
        self.records_used = []
        i = 0
        for i in range(max_steps):
            if self.time_up(i):
                stop = {"kind": "wall_limit"}
                break
            window = bytes(mem.byte((state["rip"] + j) & R.MASK64) for j in range(15))
            ri, eff = self.dec.lookup(window)
            if eff is None:
                stop = {"kind": "abstain", "reason": "no_record", "rip": hex(state["rip"]), "window": window.hex()}
                break
            new, writes, trap, abst = self.scalar.apply(eff, state, window, mem)
            if abst:
                stop = {"kind": "abstain", "reason": "predicate", "rip": hex(state["rip"]), "window": window.hex(),
                        "record": ri}
                break
            for addr, nb, v in writes:
                mem.write(addr, nb, v)
            if trap:
                new = {"rip": new["rip"], "regs": list(state["regs"]), "flags": list(state["flags"])}
                st, code = R.os_boundary(new, mem, stdout, rflags_rest)
                if st == "exit":
                    stop = {"kind": "exit", "exit_code": code}
                    break
                if st != "running":
                    stop = {"kind": st}
                    break
            if mem.fault is not None:
                stop = {"kind": "unmapped_access", "detail": str(mem.fault)}
                break
            before = ArchState(state["rip"], list(state["regs"]), dict(zip(M4_FLAGS, state["flags"])))
            state = new
            got = ArchState(state["rip"], list(state["regs"]), dict(zip(M4_FLAGS, state["flags"])))
            cmp.observe(i, got, window=window, before=before,
                        extra={"record": ri, "effect": json.dumps(eff, default=str)[:800]})
        else:
            stop = {"kind": "step_limit"}
        return finish({"stop": stop, "stdout_hex": bytes(stdout).hex(), "steps_run": i + 1,
                       "exit_code": stop.get("exit_code")}, cmp, trace)

    def debug_bundle(self, outcome):
        div = outcome.get("first_divergence") or (outcome.get("stop") if outcome.get("status") == "abstain" else None)
        return None if not div else {"kind": "m5e", **div}
