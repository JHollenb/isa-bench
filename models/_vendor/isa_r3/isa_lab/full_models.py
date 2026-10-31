"""Shared selector interface for the entire native PAN-ISA instruction scope."""
from __future__ import annotations

import json

import torch
from torch import nn

from pan_isa.native import runtime_module

MODEL_NAMES = ("nn", "pan_isa", "fieldlm", "spf_transformer")
EXTENDED_MODEL_NAMES = (*MODEL_NAMES, "mamba", "hydra")
HEAD_SIZES = ((17, 15), (3, 4), (3, 5, 4, 4), (125,), (268,))


def runtime_key(raw, numeric):
    """Supplied operand-invariance chart, without an instruction-name lookup."""
    names = runtime_module().NUMERIC_FEATURE_NAMES
    n = dict(zip(names, numeric, strict=True))
    opcode = round(n["opcode_norm"] * 255)
    if 0xB8 <= opcode <= 0xBF:
        opcode = 0xB8
    second = round(n["second_norm"] * 255) if opcode in (0x0F, 0xD9, 0xDD) else 0
    reg = round(n["modrm_reg"] * 7) if opcode in (0x83, 0xC7) else 0
    memory = int(n["modrm_mod"] < 1) if opcode in (0x89, 0x8B, 0x31, 0x0F) else 0
    aligned = int(n["memory_address_aligned16"]) if opcode == 0x0F and n["prefix_66"] and second in (0x6F, 0x7F) else 0
    # Register number is not a family variable for FSTP ST(i).
    if opcode == 0xDD:
        second &= 0xF8
    return tuple(int(x) for x in (5, opcode, second, n["prefix_66"], n["prefix_f2"],
                                 n["prefix_f3"], memory, reg, aligned))


def stage_key(stage, op, flags):
    return (stage, op, *(int(x) for x in flags)) if stage == 2 else (stage, op)


class TrainableSelectors(nn.Module):
    def logits(self, stage, ops, flags):
        head = self.heads[str(stage)]
        if stage in (0, 1):
            return head.selector_logits(ops)
        if stage == 2:
            return head.selector_logits(ops, flags)
        return (head(ops, flags) if stage == 3 else head(ops),)

    def selectors(self, stage, op, flags):
        with torch.no_grad():
            out = self.logits(stage, torch.tensor([op]), torch.tensor([flags], dtype=torch.float32))
            return tuple(int(x.argmax(-1)[0]) for x in out)

    def runtime_predict(self, raw, numeric):
        with torch.no_grad():
            logits = self.heads["runtime"](torch.tensor([raw]), torch.tensor([numeric]))[0]
            probs = logits.softmax(-1)
            i = int(probs.argmax())
            return runtime_module().RUNTIME_NEURAL_LABELS[i], float(probs[i])

    def loss(self, rows):
        terms = []
        for stage in range(5):
            batch = rows[str(stage)]
            ops = torch.tensor([r["op"] for r in batch])
            flags = torch.tensor([r["flags"] for r in batch], dtype=torch.float32)
            target = torch.tensor([r["target"] for r in batch])
            out = self.logits(stage, ops, flags)
            terms.extend(nn.functional.cross_entropy(logits, target[:, i]) for i, logits in enumerate(out))
        batch = rows["runtime"]
        out = self.heads["runtime"](torch.tensor([r["raw"] for r in batch]),
                                    torch.tensor([r["numeric"] for r in batch]))
        terms.append(nn.functional.cross_entropy(out, torch.tensor([r["target"][0] for r in batch])))
        return sum(terms)

    def update(self, rows, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = self.loss(rows)
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def export(self):
        return {"weights": {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()},
                "parameter_count": sum(p.numel() for p in self.parameters()), "implementation": self.implementation}

    def restore_export(self, payload):
        state = self.state_dict()
        self.load_state_dict({k: torch.tensor(v, dtype=state[k].dtype) for k, v in payload["weights"].items()})


class StructuralSelectors(nn.Module):
    def __init__(self, seed=17):
        super().__init__()
        self.routes = {}
        self.prepare()

    def update(self, rows, optimizer=None):
        routes = {}
        for stage, batch in rows.items():
            for r in batch:
                key = runtime_key(r["raw"], r["numeric"]) if stage == "runtime" else stage_key(int(stage), r["op"], r["flags"])
                encoded = json.dumps(key, separators=(",", ":"))
                if encoded in routes and routes[encoded] != r["target"]:
                    raise ValueError(f"chart aliases incompatible discovery effects: {key}")
                routes[encoded] = r["target"]
        self.routes = routes
        self.prepare()
        return 0.0

    def selectors(self, stage, op, flags):
        return tuple(self.read(stage_key(stage, op, flags)))

    def runtime_predict(self, raw, numeric):
        label = self.read(runtime_key(raw, numeric))[0]
        return runtime_module().RUNTIME_NEURAL_LABELS[label], 1.0

    def get_extra_state(self):
        return dict(self.routes)

    def set_extra_state(self, state):
        self.routes = dict(state)
        self.prepare()

    def export(self):
        return {"routes": self.routes, "parameter_count": 0, "learned_record_count": len(self.routes),
                "implementation": self.implementation}

    def restore_export(self, payload):
        self.set_extra_state(payload["routes"])


def create(name, seed):
    if name == "mamba":
        from mamba_isa.full_model import MambaISASelectors
        return MambaISASelectors(seed)
    if name == "hydra":
        from hydra_isa.full_model import HydraISASelectors
        return HydraISASelectors(seed)
    if name == "pan_isa":
        from pan_isa.full_model import PANISASelectors
        return PANISASelectors(seed)
    if name == "nn":
        from nn.full_model import FullNNSelectors
        return FullNNSelectors(seed)
    if name == "fieldlm":
        from fieldlm.full_model import FullFieldSelectors
        return FullFieldSelectors(seed)
    if name == "spf_transformer":
        from spf_transformer.full_model import FullSPFSelectors
        return FullSPFSelectors(seed)
    raise ValueError(name)
