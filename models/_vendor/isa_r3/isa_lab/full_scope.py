"""Source-derived coverage, typed inputs and shared native semantic consumers.

Serving never receives reference after-state or effect labels. Stage 3/4 input
normalization and trampoline changes are explicit exogenous plan interventions.
"""
from __future__ import annotations

import copy
import dataclasses
import random

import torch
from torch import nn

from isa_lab.data import digest
from pan_isa.native import runtime_module, stages

PROFILE = "pan-native-stages0-4-runtime-v1"


def inventory():
    s = stages()
    return {"profile": PROFILE, "stages": {str(i): [dataclasses.asdict(op) for op in m.OPS]
                                         for i, m in enumerate(s)},
            "runtime": list(runtime_module().RUNTIME_NEURAL_LABELS),
            "stage_forms": sum(len(m.OPS) for m in s)}


class ForcedOpRandom(random.Random):
    """Generate through unchanged upstream builders with a declared opcode intervention."""
    def __init__(self, seed):
        super().__init__(seed)
        self.op, self.count, self.name = 0, 0, None

    def randrange(self, start, stop=None, step=1):
        if stop is None and start == self.count:
            return self.op
        return super().randrange(start, stop, step)

    def choice(self, seq):
        if self.name is not None and self.name in seq:
            return self.name
        return super().choice(seq)


def make_trace(stage, seed, split, length=1, force=None):
    module = stages()[stage]
    rng = random.Random(seed) if force is None else ForcedOpRandom(seed)
    if force is not None:
        rng.op, rng.count = force, len(module.OPS)
        rng.name = module.OPS[force].name if stage == 2 else None
    args = (f"{split}-{stage}-{seed}", split, length, rng)
    return module.make_trace(*args, 8) if stage == 0 else module.make_trace(*args)


def targets(stage, step):
    module = stages()[stage]
    p = step[f"pan_stage{stage}"]
    op = int(p["op_index"])
    if stage == 0:
        return [int(module.OP_VALUE_CANDIDATE[op]), int(module.OP_FLAG_CANDIDATE[op])]
    if stage == 1:
        return [int(module.OP_VALUE_SELECTOR[op]), int(module.OP_MEMORY_SELECTOR[op])]
    if stage == 2:
        return [int(p[k]) for k in ("rip_selector", "stack_selector", "reg_selector", "flag_selector")]
    return [op]


def before(stage, step):
    p = step[f"pan_stage{stage}"]
    if stage == 0:
        return {"regs": list(p["before_regs"]), "flags": list(p["before_flags"])}
    if stage == 1:
        return {"regs": list(p["before_data_regs"]), "addr": list(p["before_addr_offsets"]),
                "indices": list(p["before_index_offsets"]), "memory": list(p["before_memory"])}
    if stage == 2:
        return {"rax": int(p["before_rax"]), "flags": list(p["before_flags"]), "rip": int(p["before_rip"]),
                "rsp": int(p["before_rsp_offset"]), "memory": list(p["before_stack"])}
    if stage == 3:
        return {"regs": list(p["before_regs"]), "flags": list(p["before_flags"]),
                "memory": list(p.get("before_memory", []))}
    return {"xmm": list(p["before_xmm"]), "memory": list(p.get("before_memory", []))}


def expected(stage, step, input_state):
    p = step[f"pan_stage{stage}"]
    if stage == 0:
        return {"regs": list(p["after_regs"]), "flags": list(p["after_flags"])}
    if stage == 1:
        return {"regs": list(p["after_data_regs"]), "addr": list(p["after_addr_offsets"]),
                "indices": list(p["after_index_offsets"]), "memory": list(p["after_memory"])}
    if stage == 2:
        return {"rax": int(p["after_rax"]), "flags": list(p["after_flags"]), "rip": int(p["after_rip"]),
                "rsp": int(p["after_rsp_offset"]), "memory": list(p["after_stack"])}
    if stage == 3:
        return {"regs": list(p["after_regs"]), "flags": list(p["after_flags"]),
                "memory": list(p.get("after_memory", input_state["memory"])), "exception": p["exception"]}
    return {"xmm": list(p["after_xmm"]), "memory": list(p.get("after_memory", input_state["memory"])),
            "exception": p["exception"]}


def serving(stage, step):
    p = step[f"pan_stage{stage}"]
    names = ("op_index", "dst", "src", "imm", "base", "index", "scale", "disp", "rel", "mem_offset")
    return {k: p[k] for k in names if k in p} | {"length": len(bytes.fromhex(step["decode"]["bytes_hex"]))}


def trace_packet(stage, trace):
    """Compile supplied input interventions separately from quarantined references."""
    packets, previous = [], None
    # The image is unchanged until the first memory operation. Its before-image
    # supplies the initial memory when non-memory prefix steps omit that image.
    memory = next((list(x[f"pan_stage{stage}"]["before_memory"]) for x in trace["steps"]
                   if "before_memory" in x[f"pan_stage{stage}"]), []) if stage in (3, 4) else []
    for step in trace["steps"]:
        initial = before(stage, step)
        if stage in (3, 4):
            if initial["memory"]:
                memory = initial["memory"]
            else:
                initial["memory"] = list(memory)
        gold = expected(stage, step, initial)
        patches = {}
        if previous is not None and stage in (3, 4):
            # Native generators intentionally normalize registers between instructions.
            for field, value in initial.items():
                prior = previous[field]
                if isinstance(value, list):
                    delta = {str(i): x for i, x in enumerate(value) if i >= len(prior) or x != prior[i]}
                    if delta:
                        patches[field] = delta
                elif value != prior:
                    patches[field] = value
        packets.append({"input": serving(stage, step), "before": initial, "reference": gold,
                        "patches": patches, "target": targets(stage, step),
                        "bytes": step["decode"]["bytes_hex"], "family": step["decode"]["category"]})
        previous, memory = gold, gold.get("memory", memory)
    return {"id": trace["trace_id"], "stage": stage, "steps": packets,
            "continuation": "predicted-state-with-explicit-native-input-interventions" if stage in (3, 4)
                            else "predicted-state"}


def stage_rows(traces):
    rows = []
    for trace in traces:
        for step in trace["steps"]:
            state = step["before"]
            rows.append({"op": step["input"]["op_index"], "flags": state.get("flags", [0] * 4),
                         "target": step["target"]})
    return rows


def generate(seed, examples=3, trace_length=32, include_terminal=False):
    """Every native form in each split, including all 16 branch flag contexts."""
    suite = {"profile": PROFILE, "seed": seed, "inventory": inventory(), "splits": {}}
    split_names = (("discovery", "train"), ("feedback", "full"), ("heldout", "full"), ("terminal", "full"))
    if not include_terminal:
        split_names = split_names[:3]
    for split_index, (name, native_split) in enumerate(split_names):
        panel = {"stages": {}, "runtime": []}
        for stage, module in enumerate(stages()):
            traces = []
            for op in range(len(module.OPS)):
                for rep in range(examples):
                    for attempt in range(100):
                        specimen_seed = seed * 1000000 + split_index * 100000 + stage * 10000 + op * 101 + rep * 30000000 + attempt
                        trace = make_trace(stage, specimen_seed, native_split, force=op)
                        if trace["steps"][0][f"pan_stage{stage}"]["op_index"] == op:
                            break
                    else:
                        raise ValueError(f"native generator cannot cover stage {stage} op {op}")
                    if stage == 2:
                        # The native branch targets are recomputed for each supplied before-flag vector.
                        for bits in range(16):
                            clone = copy.deepcopy(trace)
                            p = clone["steps"][0]["pan_stage2"]
                            fl = [(bits >> i) & 1 for i in range(4)]
                            p["before_flags"] = fl
                            out = module.apply_op(p["before_rax"], fl, p["before_rip"], p["before_rsp_offset"],
                                                  p["before_stack"], op, p["imm"], p["fallthrough_rip"], p["target_rip"])
                            p.update(dict(zip(("after_rax", "after_flags", "after_rip", "after_rsp_offset", "after_stack"), out[:5], strict=True)))
                            p.update(dict(zip(("rip_selector", "stack_selector", "reg_selector", "flag_selector"),
                                              module.selector_targets(module.OPS[op], fl, module.condition_taken(module.OPS[op].name, fl)), strict=True)))
                            clone["trace_id"] += f"-flags-{bits}"
                            traces.append(trace_packet(stage, clone))
                    else:
                        traces.append(trace_packet(stage, trace))
            # Additional native continuations exercise memory/state reuse and error propagation.
            traces.append(trace_packet(stage, make_trace(stage, seed + split_index * 1000 + stage * 100,
                                                       native_split, length=trace_length)))
            panel["stages"][str(stage)] = traces
        runtime = runtime_module()
        from isa_lab.full_evaluate import execute_runtime
        from pan_isa.runtime_adapter import runtime_fixture
        specs_per_seed = len(list(runtime._training_code_specs(1)))
        for index, (code, options) in enumerate(runtime._training_code_specs(examples * 8 * 5)):
            if (index // specs_per_seed) % 5 != split_index:
                continue
            fixture = {"code": code.hex(), "seed": seed + split_index * 10000 + index, "options": options}
            instance = runtime_fixture(fixture)
            instruction = runtime.decode_current(instance)
            raw, numeric = runtime.encode_runtime_features(instance, instruction)
            label = instance.model.propose(instance, instruction).instruction_name
            panel["runtime"].append({**fixture, "raw": raw, "numeric": numeric,
                                      "target": [runtime.RUNTIME_NEURAL_LABELS.index(label)],
                                      "reference": execute_runtime(None, fixture)})
        suite["splits"][name] = panel
        from isa_lab.cumulative import fixture as cumulative_fixture
        panel["cumulative"] = cumulative_fixture(seed + split_index * 1000)
        # The instruction learner selects SYSCALL; Linux service behavior is shared native code.
        from isa_lab.runtime_cases import cases
        if name == "discovery":
            panel["runtime_discovery_rows"] = []
        for fixture in cases(seed + split_index * 10000):
            panel["runtime"].append({**fixture, "target": [2], "reference": execute_runtime(None, fixture)})
            if name == "discovery":
                # Supervise actual instruction boundaries and carried runtime context
                # in discovery programs, including small/zero literals and x87 depth.
                teacher = runtime_fixture(fixture)
                for _ in range(fixture.get("max_steps", 1)):
                    instruction = runtime.decode_current(teacher)
                    raw, numeric = runtime.encode_runtime_features(teacher, instruction)
                    label = teacher.model.propose(teacher, instruction).instruction_name
                    panel["runtime_discovery_rows"].append({"raw": raw, "numeric": numeric,
                        "target": [runtime.RUNTIME_NEURAL_LABELS.index(label)], "source": fixture["case"]})
                    if teacher.step().kind != "running":
                        break
    suite["sha256"] = digest(suite)
    return suite


class FixedLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.value = None

    def forward(self, _):
        return self.value


class Consumer:
    def __init__(self):
        self.s = stages()
        self.alu = self.s[0].FactorizedStage0PAN(8)
        self.alu.op_emb = nn.Identity()
        self.alu.value_net, self.alu.flag_net = FixedLogits(), FixedLogits()

    def step(self, model, stage, instruction, state):
        """Only serving inputs and the model's current predicted state enter this boundary."""
        op, flags = instruction["op_index"], state.get("flags", [0] * 4)
        choice = model.selectors(stage, op, flags)
        m = self.s[stage]
        if stage == 0:
            self.alu.value_net.value = nn.functional.one_hot(torch.tensor([choice[0]]), 17).float()
            self.alu.flag_net.value = nn.functional.one_hot(torch.tensor([choice[1]]), 15).float()
            with torch.no_grad():
                reg, fl = self.alu(torch.tensor([state["regs"]]), torch.tensor([flags]), torch.tensor([op]),
                                   torch.tensor([instruction["dst"]]), torch.tensor([instruction["src"]]),
                                   torch.tensor([instruction["imm"]]))
            out = {"regs": list(state["regs"]), "flags": list(flags)}
            if m.OPS[op].writes_dst:
                out["regs"][instruction["dst"]] = int(reg.argmax(-1)[0, instruction["dst"]])
            if m.OPS[op].writes_flags:
                out["flags"] = fl.argmax(-1)[0].tolist()
            self.alu.value_net.value = self.alu.flag_net.value = None
        elif stage == 1:
            regs, addr, mem, _ = m.apply_selected(state["regs"], state["addr"], state["indices"], state["memory"],
                  op, *(instruction[k] for k in ("dst", "src", "base", "index", "scale", "disp", "imm")), *choice)
            out = {"regs": regs, "addr": addr, "indices": list(state["indices"]), "memory": mem}
        elif stage == 2:
            fallthrough = state["rip"] + instruction["length"]
            rax, fl, rip, rsp, mem, _ = m.apply_selectors(state["rax"], flags, state["rip"], state["rsp"],
                            state["memory"], op, instruction["imm"], fallthrough,
                            fallthrough + instruction["rel"], *choice)
            out = {"rax": rax, "flags": fl, "rip": rip, "rsp": rsp, "memory": mem}
        elif stage == 3:
            regs, fl, mem, effect = m.transition_with_selector(state["regs"], flags, state["memory"],
                                                   {"pan_stage3": instruction}, choice[0])
            out = {"regs": regs, "flags": fl, "memory": mem, "exception": effect["exception"]}
        else:
            predictor = m.StructuredStage4Predictor(op_offset=choice[0] - op)
            xmm, mem, effect = predictor.predict_next(state["xmm"], state["memory"], {"pan_stage4": instruction})
            out = {"xmm": xmm, "memory": mem, "exception": effect["exception"]}
        return out, choice


def apply_patches(state, patches):
    result = copy.deepcopy(state)
    for field, patch in patches.items():
        if isinstance(patch, dict):
            for index, value in patch.items():
                result[field][int(index)] = value
        else:
            result[field] = patch
    result.pop("exception", None)
    return result


def inputs_only(panel):
    return {"stages": {key: [{"id": t["id"], "stage": t["stage"], "initial": t["steps"][0]["before"],
                             "steps": [{"input": x["input"], "patches": x["patches"]} for x in t["steps"]]}
                            for t in traces] for key, traces in panel["stages"].items()},
            "runtime": [{k: row[k] for k in ("code", "seed", "options", "max_steps") if k in row} for row in panel["runtime"]],
            "cumulative": {k: v for k, v in panel.get("cumulative", {}).items() if k != "reference"}}
