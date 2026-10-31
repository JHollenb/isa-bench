"""Selector track (2026-10-14 six-model cohort) and the supplied PAN reference runtime.

A selector picks one of 33 runtime instruction families; the frozen PAN consumer
parses operands and computes registers, flags and memory. ``selector: null`` runs
the consumer with its bootstrap transition model and no learned selector, which
separates consumer coverage from learned selection.

The runtime is started from the native initial state (GPRs, RFLAGS and the full
memory image from ptrace) so every step can be compared with the native trace.
"""
from __future__ import annotations

import json

from isa_bench.adapters import Base, finish
from isa_bench.state import GPRS, TF, ArchState, flags_of, image_regions


class Adapter(Base):
    family = "selector"

    def load(self):
        self.add_paths(self.vendor_dir())
        from pan_isa.native import mount

        mount()
        import torch

        torch.set_num_threads(int(self.cfg.get("threads", 1)))
        from pan_isa_x86_runtime_spf.runtime import SPFPrimaryPanRuntime

        self.Runtime = SPFPrimaryPanRuntime
        self.runtime_model, meta = None, {"parameter_count": 0, "implementation": "supplied PAN consumer (no selector)"}
        if self.cfg.get("selector"):
            from isa_lab.full_training import restore
            from pan_isa.runtime_adapter import SharedRuntimeModel

            selector = restore(json.loads((self.artifact / "model.json").read_text()))
            self.runtime_model = SharedRuntimeModel(selector)
            payload = selector.export()
            meta = {k: payload[k] for k in ("parameter_count", "learned_record_count", "packed_plane_bytes") if k in payload}
            meta["implementation"] = selector.implementation
            meta["tensor_state_bytes"] = sum(t.numel() * t.element_size() for t in selector.state_dict().values()
                                             if isinstance(t, torch.Tensor))
        self.metadata = {**self.base_metadata(), **meta}

    def _native_memory(self, program_dir):
        from pan_x86_isa.memory import SparseMemory

        mem = SparseMemory()
        # image records can share pages (unaligned section ends): union permissions per page, then map runs
        pages: dict[int, list] = {}
        for start, length, perm, data in image_regions(program_dir / "mem.bin"):
            perms = set(c for c in perm[:3] if c != "-") or {"r"}
            addr, end = start, start + length
            while addr < end:
                base = addr & ~0xFFF
                n = min(end, base + 4096) - addr
                page = pages.setdefault(base, [bytearray(4096), set()])
                page[0][addr - base:addr - base + n] = data[addr - start:addr - start + n]
                page[1] |= perms
                addr += n
        runs = []
        for base in sorted(pages):
            perms = "".join(c for c in "rwx" if c in pages[base][1])
            if runs and runs[-1][0] + runs[-1][1] == base and runs[-1][2] == perms:
                runs[-1][1] += 4096
            else:
                runs.append([base, 4096, perms])
        for base, size, perms in runs:
            mem.map_region(base, size, perms if "w" in perms else perms + "w", name="native")
            mem.load_bytes(base, b"".join(bytes(pages[b][0]) for b in range(base, base + size, 4096)))
            if "w" not in perms:
                mem.protect_region(base, size, perms)
        return mem, runs

    def _native_runtime(self, program_dir, trace):
        """Runtime over the native image (not PAN's ELF loader, which faults on adjacent R/RW segments)."""
        from pan_x86_isa.process import ProcessImage

        mem, runs = self._native_memory(program_dir)
        first = trace.state(0)
        rsp = first.regs[GPRS.index("RSP")]
        below_stack = [b + n for b, n, _ in runs if b + n <= rsp - (1 << 20)]
        image = ProcessImage(memory=mem, entry=first.rip, initial_rsp=rsp,
                             brk=max(below_stack) if below_stack else 0, program=str(program_dir / "prog.elf"),
                             argv=[str(program_dir / "prog.elf")], env={})
        runtime = self.Runtime(image, model=self.runtime_model, backend="ps", trace_proposals=False)
        runtime.state.gpr.update({name: value for name, value in zip(GPRS, first.regs)})
        runtime.state.gpr["RIP"] = first.rip
        runtime.state.rflags = trace.initial_eflags
        return runtime

    def run(self, program_dir, trace, max_steps):
        runtime = self._native_runtime(program_dir, trace)
        cmp, stop = self.comparator(trace), None
        i = 0
        for i in range(max_steps):
            if self.time_up(i):
                stop = {"kind": "wall_limit"}
                break
            rip = runtime.state.gpr["RIP"]
            try:
                window = bytes(runtime.memory.read(rip, 15, access="execute", record=False))
            except Exception:
                window = b""
            try:
                reason = runtime.step()
            except (KeyError, LookupError, ValueError) as error:  # FieldLM/SPF selectors raise on an unseen route key
                if "route" not in str(error):
                    raise
                stop = {"kind": "refused_missing_route", "detail": str(error)[:300], "rip": hex(rip)}
                break
            if reason.kind == "exit":
                stop = {"kind": "exit", "exit_code": reason.exit_code}
                break
            if reason.kind != "running":
                stop = {"kind": reason.kind, "detail": reason.detail,
                        "address": None if reason.address is None else hex(reason.address)}
                break
            state = runtime.state
            got = ArchState(state.gpr["RIP"], [state.gpr.get(n, 0) for n in GPRS], flags_of(state.rflags & ~TF))
            cmp.observe(i, got, window=window)
        else:
            stop = {"kind": "step_limit"}
        return finish({"stop": stop, "stdout_hex": bytes(runtime.stdout).hex(), "steps_run": i + 1,
                       "exit_code": stop.get("exit_code")}, cmp, trace)
