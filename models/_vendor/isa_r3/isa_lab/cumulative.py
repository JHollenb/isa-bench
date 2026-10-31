"""Shared full-state composition using learned selections at every stage boundary."""
from __future__ import annotations

import importlib

from pan_isa.native import stages


def module():
    stages()
    return importlib.import_module("pan_x86_stages.stage5_full_model_pan")


def fixture(seed):
    m = module()
    initial = m.FullState.initial()
    initial.gpr["RBX"] += seed & 31
    initial.gpr["RCX"] += seed & 15
    initial.xmm[0] += seed
    initial.xmm[1] += seed * 3
    reference = m.FullStageModel(initial.clone())
    operations = m.demo_operations()
    results = [reference.apply(i, op)["after"] for i, op in enumerate(operations)]
    return {"initial": initial.raw_json(), "operations": operations, "reference": results}


def serve(model, item, consumer):
    """Only the initial full state and operation program enter this executor."""
    m = module()
    s = consumer.s
    state = m.FullState.from_json(item["initial"])
    results = []
    for operation in item["operations"]:
        stage = int(operation["stage"][-1])
        op = next(i for i, spec in enumerate(s[stage].OPS) if spec.name == operation["op_name"])
        instruction = {k: int(operation.get(k, 0)) for k in ("dst", "src", "imm")}
        instruction["op_index"] = op
        if stage == 0:
            incoming = {"regs": [state.gpr[n] & 255 for n in m.LOW_BYTE_REGS], "flags": list(state.flags)}
            result, choice = consumer.step(model, stage, instruction, incoming)
            for i, name in enumerate(m.LOW_BYTE_REGS):
                state.gpr[name] = (state.gpr[name] & ~255) | result["regs"][i]
            state.flags = result["flags"]
            code = s[0].encode_x86_bytes(s[0].OPS[op], instruction["dst"], instruction["src"], instruction["imm"], 8)
        elif stage == 1:
            instruction.update({k: int(operation.get(k, d)) for k, d in (("base", 0), ("index", 0), ("scale", 1), ("disp", 0))})
            incoming = {"regs": [state.gpr[n] for n in s[1].DATA_REGS],
                        "addr": [state.gpr[n] - s[1].MEM_BASE for n in s[1].ADDR_REGS],
                        "indices": [state.gpr[n] for n in s[1].INDEX_REGS], "memory": state.memory["stage1_data"]}
            result, choice = consumer.step(model, stage, instruction, incoming)
            for i, name in enumerate(s[1].DATA_REGS):
                state.gpr[name] = result["regs"][i] & m.MASK64
            for i, name in enumerate(s[1].ADDR_REGS):
                state.gpr[name] = (s[1].MEM_BASE + result["addr"][i]) & m.MASK64
            state.memory["stage1_data"] = result["memory"]
            code = s[1].encode_x86_bytes(s[1].OPS[op], *(instruction[k] for k in ("dst", "src", "base", "index", "scale", "disp", "imm")))
        elif stage == 2:
            spec = s[2].OPS[op]
            length = 5 if spec.name == "CALL_REL32" else 2 if spec.operands in ("ri", "rel8") or spec.name == "TEST_AL_AL" else 1
            fallthrough = state.gpr["RIP"] + length
            instruction.update({"length": length, "rel": int(operation.get("target", fallthrough)) - fallthrough})
            incoming = {"rax": state.gpr["RAX"], "flags": state.flags, "rip": state.gpr["RIP"],
                        "rsp": state.gpr["RSP"] - s[2].STACK_BASE, "memory": state.memory["stage2_stack"]}
            result, choice = consumer.step(model, stage, instruction, incoming)
            state.gpr.update({"RAX": result["rax"] & m.MASK64, "RSP": s[2].STACK_BASE + result["rsp"], "RIP": result["rip"]})
            state.flags, state.memory["stage2_stack"] = result["flags"], result["memory"]
            results.append(state.snapshot())
            continue
        elif stage == 3:
            instruction["mem_offset"] = int(operation.get("mem_offset", 0))
            incoming = {"regs": [state.gpr[n] for n in s[3].REGS], "flags": state.flags,
                        "memory": state.memory["stage3_data"]}
            result, choice = consumer.step(model, stage, instruction, incoming)
            _, _, _, effect = s[3].transition_with_selector(incoming["regs"], incoming["flags"], incoming["memory"],
                                                          {"pan_stage3": instruction}, choice[0])
            selected_op = s[3].OPS[choice[0]]
            mask = effect["flag_mask"]
            state.flag_defined = s[3].flags_defined_after(state.flag_defined, mask, s[3].undefined_flag_names(selected_op, mask))
            for i, name in enumerate(s[3].REGS):
                state.gpr[name] = result["regs"][i] & m.MASK64
            state.flags, state.memory["stage3_data"] = result["flags"], result["memory"]
            addr = s[3].MEM_BASE + instruction["mem_offset"] if s[3].OPS[op].operands in s[3].MEMORY_OPERANDS else 0
            code = s[3].encode_x86_bytes(s[3].OPS[op], instruction["dst"], instruction["src"], instruction["imm"], addr)
        else:
            instruction["mem_offset"] = int(operation.get("mem_offset", 0))
            incoming = {"xmm": state.xmm, "memory": state.memory["stage4_data"]}
            result, choice = consumer.step(model, stage, instruction, incoming)
            state.xmm, state.memory["stage4_data"] = result["xmm"], result["memory"]
            addr = s[4].MEM_BASE + instruction["mem_offset"] if s[4].OPS[op].operands in s[4].MEMORY_OPERANDS else 0
            code = s[4].encode_x86_bytes(s[4].OPS[op], instruction["dst"], instruction["src"], instruction["imm"], addr)
        state.gpr["RIP"] = (state.gpr["RIP"] + len(bytes.fromhex(code))) & m.MASK64
        results.append(state.snapshot())
    return results
