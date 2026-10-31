"""Common native consumers, per-field outcomes and autonomous continuation."""
from __future__ import annotations

import copy
import dataclasses

from isa_lab.data import digest
from isa_lab.full_scope import apply_patches
from pan_isa.runtime_adapter import SharedRuntimeModel, runtime_fixture


def runtime_snapshot(runtime):
    return {"state": dataclasses.asdict(runtime.state),
            "pages": {str(base): digest(bytes(value).hex()) for base, value in sorted(runtime.memory._pages.items())},
            "regions": [dataclasses.asdict(r) for r in runtime.memory.regions],
            "stdout": bytes(runtime.stdout).hex(), "stderr": bytes(runtime.stderr).hex(),
            "stop": None if runtime.stop_reason is None else runtime.stop_reason.to_json()}


def execute_runtime(selector, row):
    runtime = runtime_fixture(row)
    if selector is not None:
        runtime.model = SharedRuntimeModel(selector)
    count = 0
    for _ in range(row.get("max_steps", 1)):
        count += 1
        reason = runtime.step()
        if reason.kind != "running":
            break
    return {**runtime_snapshot(runtime), "executed_steps": count}


def serve(model, inputs, consumer):
    """Timed serving API accepts an input-only manifest, never gold or labels."""
    result = {"stages": {}, "runtime": []}
    for key, traces in inputs["stages"].items():
        states = []
        for trace in traces:
            state = copy.deepcopy(trace["initial"])
            outputs = []
            for index, step in enumerate(trace["steps"]):
                if index:
                    state = apply_patches(state, step["patches"])
                try:
                    state, choice = consumer.step(model, int(key), step["input"], state)
                    outputs.append({"state": state, "choice": choice})
                except (ValueError, IndexError, ZeroDivisionError, OverflowError) as exc:
                    outputs.append({"error": type(exc).__name__, "detail": str(exc)})
                    break
            states.append(outputs)
        result["stages"][key] = states
    for row in inputs["runtime"]:
        try:
            result["runtime"].append(execute_runtime(model, row))
        except Exception as exc:  # noqa: BLE001 - retain each failed consumer row
            result["runtime"].append({"error": type(exc).__name__, "detail": str(exc)})
    if inputs.get("cumulative"):
        from isa_lab.cumulative import serve as cumulative_serve
        try:
            result["cumulative"] = cumulative_serve(model, inputs["cumulative"], consumer)
        except Exception as exc:  # noqa: BLE001 - retain each failed cumulative program
            result["cumulative"] = {"error": type(exc).__name__, "detail": str(exc)}
    return result


def evaluate(model, panel, consumer, *, capture=True):
    result = {"stages": {}, "runtime": {}, "terminal_status": "not-certified"}
    for key, traces in panel["stages"].items():
        stage = int(key)
        iso_ok = rollout_ok = final_ok = total = errors = selector_ok = 0
        fields, families, examples = {}, {}, []
        for trace in traces:
            state = copy.deepcopy(trace["steps"][0]["before"])
            last_ok = False
            for index, step in enumerate(trace["steps"]):
                total += 1
                bucket = families.setdefault(step["family"], [0, 0])
                bucket[1] += 1
                try:
                    isolated, choice = consumer.step(model, stage, step["input"], step["before"])
                    selector_ok += tuple(choice) == tuple(step["target"])
                    iso_ok += isolated == step["reference"]
                    if index:
                        state = apply_patches(state, step["patches"])
                    state, _ = consumer.step(model, stage, step["input"], state)
                    last_ok = state == step["reference"]
                    rollout_ok += last_ok
                    bucket[0] += last_ok
                    for field, gold in step["reference"].items():
                        counts = fields.setdefault(field, [0, 0])
                        counts[0] += state.get(field) == gold
                        counts[1] += 1
                except (ValueError, IndexError, ZeroDivisionError, OverflowError) as exc:
                    errors += 1
                    last_ok = False
                    if len(examples) < 8:
                        examples.append({"trace": trace["id"], "step": index, "error": type(exc).__name__, "detail": str(exc)})
                if not last_ok and len(examples) < 8:
                    examples.append({"trace": trace["id"], "step": index, "bytes": step["bytes"],
                                     "predicted": copy.deepcopy(state), "reference": step["reference"]})
            final_ok += last_ok
        result["stages"][key] = {"steps": total, "isolated_correct": iso_ok, "rollout_correct": rollout_ok,
              "selector_correct": selector_ok, "traces": len(traces), "final_correct": final_ok, "invalid": errors,
              "fields": fields, "families": families, "first_divergences": examples,
              "continuation": traces[-1]["continuation"]}
    correct = 0
    per_label, failures = {}, []
    from pan_isa.native import runtime_module
    labels = runtime_module().RUNTIME_NEURAL_LABELS
    for row in panel["runtime"]:
        label = row.get("case", labels[row["target"][0]])
        try:
            actual = execute_runtime(model, row)
            expected = row["reference"]
            ok = actual == expected
        except Exception as exc:  # noqa: BLE001 - retain each failed consumer row
            actual, ok = {"error": type(exc).__name__, "detail": str(exc)}, False
        correct += ok
        counts = per_label.setdefault(label, [0, 0])
        counts[0] += ok
        counts[1] += 1
        if not ok and len(failures) < 8:
            failures.append({"code": row["code"], "label": label, "predicted": actual, "reference": row["reference"]})
    result["runtime"] = {"correct": correct, "total": len(panel["runtime"]), "families": per_label,
                         "first_divergences": failures, "consumer": "original runtime.step including typed validation and commit"}
    if panel.get("cumulative"):
        from isa_lab.cumulative import serve as cumulative_serve
        item = panel["cumulative"]
        try:
            actual = cumulative_serve(model, {k: v for k, v in item.items() if k != "reference"}, consumer)
            correct = sum(x == y for x, y in zip(actual, item["reference"], strict=True))
            result["cumulative"] = {"correct": correct, "steps": len(actual), "predicted": actual,
                                     "reference": item["reference"], "continuation": "fully predicted full architectural state"}
        except Exception as exc:  # noqa: BLE001 - retain each failed cumulative program
            result["cumulative"] = {"correct": 0, "steps": len(item["reference"]), "error": type(exc).__name__, "detail": str(exc)}
    if capture:
        from isa_lab.full_scope import inputs_only
        predictions = serve(model, inputs_only(panel), consumer)
        result["serving_digest"] = digest(predictions)
    return result


def scores(measured):
    rows = measured["stages"].values()
    steps = sum(x["steps"] for x in rows)
    return {"isolated": sum(x["isolated_correct"] for x in rows) / steps,
            "rollout": sum(x["rollout_correct"] for x in rows) / steps,
            "runtime": measured["runtime"]["correct"] / measured["runtime"]["total"]}
