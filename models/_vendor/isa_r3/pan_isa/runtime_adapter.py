"""Use every trained selector inside the original ELF/runtime proposal consumer."""
from __future__ import annotations

from dataclasses import replace

from pan_isa.native import mount, runtime_module


def runtime_fixture(row):
    module = runtime_module()
    options = row["options"]
    runtime = module._fake_runtime(bytes.fromhex(row["code"]), seed=row["seed"],
                                  unaligned=options.get("unaligned", False), x87_ready=options.get("x87_ready", False))
    runtime.state.gpr.update(options.get("gpr", {}))
    return runtime


class SharedRuntimeModel:
    def __init__(self, selector, min_confidence=0.5):
        mount()
        from pan_isa_x86_runtime_spf.model import BootstrapSPFTransitionModel
        self.selector = selector
        self.adapter = BootstrapSPFTransitionModel()
        self.min_confidence = min_confidence

    @property
    def manifest(self):
        return {"runtime_mode": "learned", "runtime_execution_head_loaded": True,
                "trainable_weights_loaded": True, "profile": "long64-ring3",
                "min_confidence": self.min_confidence,
                "model": self.selector.name, "implementation": self.selector.implementation,
                "primary_semantics_path": "learned selector plus original PAN typed SPF adapter"}

    def propose(self, runtime, instruction):
        module = runtime_module()
        from pan_x86_isa.runtime import UnsupportedInstruction
        raw, numeric = module.encode_runtime_features(runtime, instruction)
        name, confidence = self.selector.runtime_predict(raw, numeric)
        if confidence < self.min_confidence:
            raise UnsupportedInstruction(instruction.rip, instruction.opcode_hex,
                                         f"selector confidence {confidence:.6f} below common threshold")
        proposal = self.adapter.propose(runtime, instruction)
        if proposal.instruction_name != name:
            raise UnsupportedInstruction(instruction.rip, instruction.opcode_hex,
                                         f"learned selector {name}; typed adapter {proposal.instruction_name}")
        return replace(proposal, confidence=confidence, family_provenance=self.selector.implementation,
                       spf_manifest=self.manifest)
