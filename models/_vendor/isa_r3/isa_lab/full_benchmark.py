"""Warm component costs, separating instruction execution from fixture setup."""
from __future__ import annotations

from isa_lab.data import digest
from isa_lab.full_evaluate import runtime_snapshot, serve
from isa_lab.measurement import Usage
from pan_isa.runtime_adapter import SharedRuntimeModel, runtime_fixture


def components(model, inputs, consumer, expected):
    result = {}
    for key, traces in inputs["stages"].items():
        subset = {"stages": {key: traces}, "runtime": []}
        with Usage() as cost:
            predicted = serve(model, subset, consumer)
        if digest(predicted["stages"][key]) != digest(expected["stages"][key]):
            raise ValueError("stage component output parity drift")
        count = sum(len(t["steps"]) for t in traces)
        result[key] = {**cost.result, "instructions": count, "us_per_instruction": cost.result["wall_s"] * 1e6 / count,
                       "scope": "scalar learned selection, native semantics, state construction, continuation patches"}
    runtimes = [runtime_fixture(row) for row in inputs["runtime"]]
    adapters = [SharedRuntimeModel(model) for _ in runtimes]
    for runtime, adapter in zip(runtimes, adapters, strict=True):
        runtime.model = adapter
    count = 0
    counts = []
    errors = {}
    with Usage() as cost:
        for runtime, row in zip(runtimes, inputs["runtime"], strict=True):
            executed = 0
            for _ in range(row.get("max_steps", 1)):
                count += 1
                executed += 1
                try:
                    reason = runtime.step()
                except Exception as exc:  # noqa: BLE001 - retain the failed row and remaining components
                    errors[len(counts)] = {"error": type(exc).__name__, "detail": str(exc)}
                    break
                if reason.kind != "running":
                    break
            counts.append(executed)
    actual = [errors.get(i, {**runtime_snapshot(runtime), "executed_steps": n})
              for i, (runtime, n) in enumerate(zip(runtimes, counts, strict=True))]
    if digest(actual) != digest(expected["runtime"]):
        raise ValueError("prepared native runtime output parity drift")
    attempted = sum(r.get("max_steps", 1) for r in inputs["runtime"])
    result["runtime"] = {**cost.result, "instructions": count, "attempted_instructions": attempted,
                         "failed_rows": errors,
                         "us_per_instruction": cost.result["wall_s"] * 1e6 / count,
                         "scope": "runtime.step learned selector, native SPF proposal validation/commit; setup and snapshots excluded"}
    return result
