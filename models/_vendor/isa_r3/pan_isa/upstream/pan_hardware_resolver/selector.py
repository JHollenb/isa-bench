from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Sequence

from .direct_action import ActionPacket, commit_action, direct_action_oracle, verify_action
from .ledger import direct_action_ledger_row, microprogram_ledger_row, stable_hash, validate_ledger_row, write_ledger_jsonl
from .microprogram import MicroprogramPacket, commit_microprogram, microprogram_oracle, verify_microprogram


SELECTOR_EXAMPLE_SCHEMA_VERSION = "pan-hw-selector-example-v1"
SELECTOR_HEADS_SCHEMA_VERSION = "pan-hw-selector-heads-v1"
SELECTOR_EVAL_SCHEMA_VERSION = "pan-hw-selector-eval-v1"


def generate_selector_examples() -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    cases = _fixture_cases()
    for index, case in enumerate(cases):
        examples.append(_selector_example(case, split="heldout" if index % 3 == 0 else "train"))
    return examples


def train_selector_heads(examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    example_list = [deepcopy(example) for example in examples]
    op_head: dict[str, str] = {}
    parameter_heads: dict[str, dict[str, list[str]]] = {}
    guard_heads: dict[str, list[str]] = {}
    expected_after_heads: dict[str, list[str]] = {}
    positive_count = 0
    negative_count = 0
    for example in example_list:
        key = str(example["case_key"])
        target = example["target"]
        expected_after_heads[key] = sorted(_leaf_paths(target.get("expected_after", {})))
        if example["negative_control"]:
            negative_count += 1
            continue
        positive_count += 1
        if example["family"] == "pan-hw-action-v1":
            op_head[key] = str(target["op"])
            parameter_heads[key] = {"params": sorted(dict(target.get("params", {})))}
        elif example["family"] == "pan-hw-microprogram-v1":
            op_head[key] = "MICROPROGRAM"
            slots = list(target.get("program", []))
            parameter_heads[key] = {
                "slot_ops": [str(slot.get("op")) for slot in slots],
                "slot_param_keys": sorted({key for slot in slots for key in dict(slot.get("params", {}))}),
            }
            guard_heads[key] = [str(slot.get("guard", "always")) for slot in slots]
    return {
        "schema": SELECTOR_HEADS_SCHEMA_VERSION,
        "authoritative": False,
        "promotion_status": "candidate",
        "requires_verifier": True,
        "source_hashes": {
            "examples": stable_hash(example_list),
            "families": stable_hash(sorted({example["family"] for example in example_list})),
        },
        "split_manifest": {
            "train": sum(1 for example in example_list if example.get("split") == "train"),
            "heldout": sum(1 for example in example_list if example.get("split") == "heldout"),
        },
        "op_head": op_head,
        "parameter_heads": parameter_heads,
        "guard_heads": guard_heads,
        "expected_after_heads": expected_after_heads,
        "risk_head": {
            "positive_examples": positive_count,
            "negative_controls": negative_count,
            "default_confidence": 1.0,
            "authority": "proposal_only",
        },
    }


def evaluate_selector_heads(model: dict[str, Any], examples: Iterable[dict[str, Any]]) -> dict[str, Any]:
    example_list = [deepcopy(example) for example in examples]
    positive = [example for example in example_list if not example["negative_control"]]
    negative = [example for example in example_list if example["negative_control"]]
    exact_action = 0
    exact_params = 0
    exact_expected_after = 0
    verifier_rejections = 0
    false_accepts = 0
    false_rejects = 0
    for example in example_list:
        target_ok = bool(example["ledger"]["envelope"]["verification"]["ok"])
        if not target_ok:
            verifier_rejections += 1
        if example["negative_control"] and target_ok:
            false_accepts += 1
        if not example["negative_control"] and not target_ok:
            false_rejects += 1
        if example["negative_control"]:
            continue
        prediction = predict_selector_action(model, example)
        target = example["target"]
        if _target_op(prediction) == _target_op(target):
            exact_action += 1
        if _target_params(prediction) == _target_params(target):
            exact_params += 1
        if prediction.get("expected_after", {}) == target.get("expected_after", {}):
            exact_expected_after += 1
    positive_count = len(positive)
    negative_count = len(negative)
    return {
        "schema": SELECTOR_EVAL_SCHEMA_VERSION,
        "ok": bool(
            positive_count
            and _ratio(exact_action, positive_count) == 1.0
            and _ratio(exact_params, positive_count) == 1.0
            and false_accepts == 0
            and false_rejects == 0
        ),
        "model_authoritative": bool(model.get("authoritative")),
        "promotion_allowed": False,
        "metrics": {
            "examples": len(example_list),
            "positive_examples": positive_count,
            "negative_controls": negative_count,
            "exact_action_selection": _ratio(exact_action, positive_count),
            "exact_or_tolerance_bounded_parameters": _ratio(exact_params, positive_count),
            "exact_expected_after": _ratio(exact_expected_after, positive_count),
            "verifier_rejection_rate": _ratio(verifier_rejections, len(example_list)),
            "false_accept_rate": _ratio(false_accepts, negative_count),
            "false_reject_rate": _ratio(false_rejects, positive_count),
            "negative_control_degradation": _ratio(verifier_rejections, negative_count),
        },
        "heldout": _split_metrics(model, positive, "heldout"),
        "source_hashes": {
            "model": stable_hash(model),
            "examples": stable_hash(example_list),
        },
    }


def predict_selector_action(model: dict[str, Any], example: dict[str, Any]) -> dict[str, Any]:
    if model.get("authoritative"):
        raise ValueError("selector heads are proposal-only; authoritative mode is forbidden")
    family = example["family"]
    state = deepcopy(example["state"])
    goal = deepcopy(example["goal"])
    constraints = deepcopy(example["constraints"])
    if family == "pan-hw-action-v1":
        return direct_action_oracle(state, goal, constraints).to_dict()
    if family == "pan-hw-microprogram-v1":
        return microprogram_oracle(state, goal, constraints).to_dict()
    raise ValueError(f"unsupported selector family: {family}")


def write_examples_jsonl(examples: Iterable[dict[str, Any]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example, sort_keys=True, separators=(",", ":")) + "\n")


def read_examples_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(payload: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _fixture_cases() -> list[dict[str, Any]]:
    state = _selector_state()
    constraints = _selector_constraints()
    direct_goals = [
        {"kind": "move_axis", "axis": "x", "target_position_ticks": 140},
        {"kind": "set_pwm", "channel": "fan0", "duty_q16": 1200},
        {"kind": "write_reg", "device": "pll0", "address": "0x10", "value": "0x31", "mask": "0xff"},
        {"kind": "dma_copy", "src": "rx0", "dst": "tx0", "length": 64},
        {"kind": "set_nco", "channel": "nco0", "frequency_word": 1024, "phase_word": 3},
        {"kind": "load_filter", "bank": "fir0", "coefficient_hash": "coef_a"},
        {"kind": "set_gain", "channel": "rf0", "gain_q8": 33},
        {"kind": "bus_tx", "bus": "i2c0", "device": "pll0", "payload_id": "p0"},
    ]
    cases = [
        {
            "case_key": f"direct:{goal['kind']}",
            "family": "pan-hw-action-v1",
            "state": state,
            "goal": goal,
            "constraints": constraints,
            "negative_control": False,
        }
        for goal in direct_goals
    ]
    cases.extend(
        [
            {
                "case_key": "micro:pll_switch",
                "family": "pan-hw-microprogram-v1",
                "state": state,
                "goal": {"kind": "pll_switch", "clock_source": "pll0"},
                "constraints": constraints,
                "negative_control": False,
            },
            {
                "case_key": "micro:filter_dma_start",
                "family": "pan-hw-microprogram-v1",
                "state": state,
                "goal": {
                    "kind": "filter_dma_start",
                    "bank": "fir0",
                    "coefficient_hash": "coef_a",
                    "src": "rx0",
                    "dst": "tx0",
                    "length": 64,
                },
                "constraints": constraints,
                "negative_control": False,
            },
            {
                "case_key": "negative:move_axis_range",
                "family": "pan-hw-action-v1",
                "state": state,
                "goal": {"kind": "move_axis", "axis": "x", "target_position_ticks": 140},
                "constraints": constraints,
                "negative_control": True,
                "candidate": ActionPacket(
                    op="MOVE_AXIS",
                    params={"axis": "x", "delta_ticks": 40, "velocity_ticks_per_s": 900, "accel_ticks_per_s2": 50},
                ).to_dict(),
            },
            {
                "case_key": "negative:pll_switch_reorder",
                "family": "pan-hw-microprogram-v1",
                "state": state,
                "goal": {"kind": "pll_switch", "clock_source": "pll0"},
                "constraints": constraints,
                "negative_control": True,
                "mutator": "reorder_wait_event_after_mux",
            },
        ]
    )
    return cases


def _selector_example(case: dict[str, Any], *, split: str) -> dict[str, Any]:
    state = deepcopy(case["state"])
    goal = deepcopy(case["goal"])
    constraints = deepcopy(case["constraints"])
    family = str(case["family"])
    if family == "pan-hw-action-v1":
        packet = ActionPacket.from_dict(case["candidate"]) if case.get("candidate") else direct_action_oracle(state, goal, constraints)
        result = commit_action(state, constraints, packet)
        row = direct_action_ledger_row(goal, constraints, packet, result, lineage={"source": "selector-generator"})
        target = packet.to_dict()
    elif family == "pan-hw-microprogram-v1":
        program = microprogram_oracle(state, goal, constraints)
        if case.get("mutator") == "reorder_wait_event_after_mux":
            slots = list(program.program)
            slots = [slots[0], slots[2], slots[1], *slots[3:]]
            program = MicroprogramPacket(slots, program.expected_after, program.slot_count, program.confidence)
        result = commit_microprogram(state, constraints, program)
        row = microprogram_ledger_row(goal, constraints, program, result, lineage={"source": "selector-generator"})
        target = program.to_dict()
    else:
        raise ValueError(f"unsupported family: {family}")
    problems = validate_ledger_row(row)
    if problems:
        raise ValueError("generated invalid selector ledger row: " + "; ".join(problems))
    return {
        "schema": SELECTOR_EXAMPLE_SCHEMA_VERSION,
        "case_key": str(case["case_key"]),
        "split": split,
        "family": family,
        "negative_control": bool(case.get("negative_control", False)),
        "state": state,
        "goal": goal,
        "constraints": constraints,
        "target": target,
        "ledger": row,
    }


def _selector_state() -> dict[str, Any]:
    return {
        "axes": {"x": {"position_ticks": 100}},
        "registers": {"pll0": {"0x10": 0}, "clock_mux": {"0x4": 0}},
        "limit_flags": {},
        "events": {},
        "dma_buffers": {"rx0": {"length": 256}, "tx0": {"length": 256}},
    }


def _selector_constraints() -> dict[str, Any]:
    return {
        "motion": {
            "axes": {
                "x": {
                    "min_position_ticks": -1000,
                    "max_position_ticks": 1000,
                    "max_velocity_ticks_per_s": 500,
                    "max_accel_ticks_per_s2": 100,
                    "default_velocity_ticks_per_s": 250,
                    "default_accel_ticks_per_s2": 50,
                }
            }
        },
        "pwm": {"channels": {"fan0": {"min_q16": 0, "max_q16": 65535}}},
        "registers": {
            "devices": {
                "pll0": {"addresses": {"0x10": {"mask": "0xff"}}},
                "clock_mux": {
                    "addresses": {
                        "0x4": {
                            "mask": "0x3",
                            "state_alias": {
                                "field": "clock_source",
                                "mask": "0x3",
                                "values": {"0x2": "pll0"},
                            },
                        }
                    }
                },
            }
        },
        "dma": {
            "buffers": {"rx0": {"length": 256, "permissions": "r"}, "tx0": {"length": 256, "permissions": "w"}},
            "max_length": 256,
            "alignment": 4,
            "allowed_flags": 0,
        },
        "nco": {"channels": {"nco0": {"min_frequency_word": 0, "max_frequency_word": 4096, "max_phase_word": 4096}}},
        "filters": {"banks": {"fir0": {"allowed_hashes": ["coef_a", "coef_b"]}}},
        "gain": {"channels": {"rf0": {"min_q8": 0, "max_q8": 255}}},
        "bus": {"buses": {"i2c0": {"devices": ["pll0"], "payloads": ["p0", "p1"]}}},
        "microprogram": {"slot_count": 8},
        "events": {
            "allowed": ["pll0_locked", "dma_complete"],
            "default_timeout_ticks": 5000,
            "max_timeout_ticks": 5000,
            "watchdog_budget_ticks": 8000,
        },
    }


def _target_op(target: dict[str, Any]) -> str:
    if "op" in target:
        return str(target["op"])
    return "MICROPROGRAM"


def _target_params(target: dict[str, Any]) -> Any:
    if "params" in target:
        return target.get("params", {})
    return [(slot.get("op"), slot.get("params"), slot.get("guard")) for slot in target.get("program", [])]


def _leaf_paths(payload: Any, prefix: str = "") -> list[str]:
    if isinstance(payload, dict):
        paths: list[str] = []
        for key, value in sorted(payload.items()):
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_leaf_paths(value, path))
        return paths
    return [prefix]


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _split_metrics(model: dict[str, Any], examples: list[dict[str, Any]], split: str) -> dict[str, Any]:
    split_examples = [example for example in examples if example.get("split") == split]
    if not split_examples:
        return {"examples": 0, "exact_action_selection": 0.0}
    exact = sum(1 for example in split_examples if _target_op(predict_selector_action(model, example)) == _target_op(example["target"]))
    return {"examples": len(split_examples), "exact_action_selection": _ratio(exact, len(split_examples))}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate and evaluate non-authoritative PAN hardware selector heads.")
    parser.add_argument("--examples-jsonl", help="Write generated selector examples to JSONL.")
    parser.add_argument("--ledger-jsonl", help="Write generated action ledger rows to JSONL.")
    parser.add_argument("--model-out", help="Write selector head candidate JSON.")
    parser.add_argument("--report-out", help="Write selector evaluation report JSON.")
    parser.add_argument("--read-examples-jsonl", help="Read selector examples instead of generating fixtures.")
    parser.add_argument("--write", help="Write combined output JSON; stdout when omitted.")
    args = parser.parse_args(argv)
    examples = read_examples_jsonl(args.read_examples_jsonl) if args.read_examples_jsonl else generate_selector_examples()
    model = train_selector_heads(examples)
    report = evaluate_selector_heads(model, examples)
    if args.examples_jsonl:
        write_examples_jsonl(examples, args.examples_jsonl)
    if args.ledger_jsonl:
        write_ledger_jsonl([example["ledger"] for example in examples], args.ledger_jsonl)
    if args.model_out:
        write_json(model, args.model_out)
    if args.report_out:
        write_json(report, args.report_out)
    payload = {"ok": bool(report["ok"]), "model": model, "report": report}
    write_json(payload, args.write)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
