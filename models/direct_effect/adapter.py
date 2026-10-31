"""Direct-effect executors (NN / Mamba / Hydra over the 64-bit axis).

The network sees a 15-byte fetch window, 8 GPRs and CF/ZF/SF/OF and emits every
next-state bit, the instruction length, one optional 8-byte store and a SYSCALL
bit. Supplied by the harness: fetch at the model's RIP, the native initial memory
image, committing the predicted store, and Linux write/exit. There is no memory
read input, so loads beyond the window cannot be represented by this profile.
"""
from __future__ import annotations

import json

from isa_bench.adapters import Base, finish, linux_boundary
from isa_bench.state import MASK64, ArchState

FLAGS = ("CF", "ZF", "SF", "OF")


class Adapter(Base):
    family = "direct_effect"

    def load(self):
        self.add_paths(self.code_dir() / "src", self.vendor_dir())
        import torch

        torch.set_num_threads(int(self.cfg.get("threads", 1)))
        from direct_effect import model as model_module

        result = json.loads((self.artifact / "result.json").read_text())
        config = result.get("config", {})
        arch = self.cfg["arch"]
        geometry = dict(self.cfg.get("architecture") or {})
        options = {k: config[k] for k in ("aligned", "relptr", "regptr", "fieldptr", "maskkey", "aux_detach") if k in config}
        kwargs = {"width": geometry.get("width", 48), "layers": geometry.get("layers", 3),
                  "state": geometry.get("state", 4), "store": geometry.get("store", True), **options}
        self.model = model_module.Executor(arch, **kwargs)
        state = torch.load(self.artifact / "model.pt", map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        self.torch = torch
        self.metadata = {**self.base_metadata(), "architecture": arch, "executor_kwargs": kwargs,
                         "parameter_count": sum(p.numel() for p in self.model.parameters()),
                         "tensor_state_bytes": sum(t.numel() * t.element_size() for t in self.model.state_dict().values()),
                         "train_config": config}

    # one forward: raw predicted effects, no correction
    def step(self, window: bytes, regs: list, flags: dict, rip: int) -> dict:
        torch = self.torch
        window = window.ljust(15, b"\0")[:15]
        win = torch.tensor([[(v >> i) & 1 for v in window for i in range(8)]], dtype=torch.float32)
        reg = torch.tensor([[[(v >> i) & 1 for i in range(64)] for v in regs]], dtype=torch.float32)
        flg = torch.tensor([[flags[n] for n in FLAGS]], dtype=torch.float32)
        with torch.inference_mode():
            out = self.model(win, reg, flg)

        def word(t):
            return sum(int(b) << i for i, b in enumerate((t > 0).long().tolist()))

        length = int(out["length"][0].argmax()) + 1
        return {"regs": [word(out["regs"][0][r]) for r in range(8)],
                "flags": dict(zip(FLAGS, (out["flags"][0] > 0).long().tolist())),
                "length": length, "rip": (rip + length) & MASK64,
                "store": bool(out["store"][0] > 0), "store_addr": word(out["store_addr"][0]),
                "store_value": word(out["store_value"][0]), "syscall": bool(out["syscall"][0] > 0)}

    def run(self, program_dir, trace, max_steps):
        from isa_bench.state import PageMemory

        mem = PageMemory(program_dir / "mem.bin")
        first = trace.state(0)
        regs, flags, rip = list(first.regs[:8]), {n: first.flags[n] for n in FLAGS}, first.rip
        cmp, stdout, stop = self.comparator(trace), bytearray(), None
        self.last_inputs = None
        i = 0
        for i in range(max_steps):
            if self.time_up(i):
                stop = {"kind": "wall_limit"}
                break
            window = mem.fetch(rip)
            if not window:
                stop = {"kind": "fetch_fault", "rip": hex(rip)}
                break
            before = ArchState(rip, regs + [None] * 8, dict(flags))
            nxt = self.step(window, regs, flags, rip)
            if nxt["store"] and not mem.write(nxt["store_addr"], nxt["store_value"].to_bytes(8, "little")):
                stop = {"kind": "store_fault", "address": hex(nxt["store_addr"])}
                break
            if nxt["syscall"]:
                rip_after = (rip + 2) & MASK64
                stop, after = linux_boundary(regs + [None] * 8, rip_after, mem, stdout)
                if stop is not None:
                    break
                regs, rip = after[:8], rip_after
            else:
                regs, flags, rip = nxt["regs"], nxt["flags"], nxt["rip"]
            cmp.observe(i, ArchState(rip, regs + [None] * 8, dict(flags)), window=window, before=before,
                        extra={"predicted_length": nxt["length"], "store": nxt["store"], "syscall": nxt["syscall"]})
        else:
            stop = {"kind": "step_limit"}
        return finish({"stop": stop, "stdout_hex": bytes(stdout).hex(), "steps_run": i + 1,
                       "exit_code": stop.get("exit_code")}, cmp, trace)

    def debug_bundle(self, outcome):
        div = outcome.get("divergence_inputs")
        if not div:
            return None
        return {"kind": "direct_effect", "arch": self.cfg["arch"], "artifact": self.cfg["artifact"],
                "artifact_sha256": self.metadata.get("artifact_sha256"), **div}
