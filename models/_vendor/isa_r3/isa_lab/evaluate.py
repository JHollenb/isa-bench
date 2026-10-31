"""One consumer and metrics implementation for every architecture."""
from __future__ import annotations

import time

from .contracts import State, validate_commit


def evaluate(model, panel):
    rows, duration = [], 0
    step_correct = trace_correct = trace_steps = trace_step_correct = 0
    fields = {name: {"correct": 0, "total": 0} for name in State.__dataclass_fields__}

    def predict(instruction, before):
        nonlocal duration
        start = time.perf_counter_ns()
        try:
            proposal = model.propose(instruction, before)
            after = validate_commit(before, proposal, instruction)
            return after, proposal.evidence, None
        except Exception as exc:  # noqa: BLE001 - optional model failures must remain per row
            # Optional model failures stay local; all other rows/traces are still evaluated.
            return None, {}, {"type": type(exc).__name__, "message": str(exc)}
        finally:
            duration += time.perf_counter_ns() - start

    for row in panel["observations"]:
        after, evidence, error = predict(bytes.fromhex(row["instruction"]), State.from_dict(row["before"]))
        expected = State.from_dict(row["after"])
        exact = after == expected
        step_correct += exact
        for name, counts in fields.items():
            counts["total"] += 1
            counts["correct"] += int(after is not None and getattr(after, name) == getattr(expected, name))
        rows.append({"lane": "isolated", "id": row["id"], "instruction": row["instruction"],
                     "predicted": after.to_dict() if after else None, "expected": row["after"],
                     "exact": exact, "error": error, "evidence": evidence})
    for trace in panel["traces"]:
        state = State.from_dict(trace["initial"])
        first_divergence = None
        stopped = False
        for index, row in enumerate(trace["steps"]):
            # Only predicted state feeds the next instruction. Oracle state is scoring-only.
            after, evidence, error = (predict(bytes.fromhex(row["instruction"]), state)
                                      if not stopped else (None, {}, {"type": "StoppedTrace"}))
            exact = after is not None and after.to_dict() == row["after"]
            trace_steps += 1
            trace_step_correct += exact
            if not exact and first_divergence is None:
                first_divergence = index
            if after is None:
                stopped = True
            else:
                state = after
            rows.append({"lane": "rollout", "id": trace["id"], "step": index,
                         "instruction": row["instruction"], "predicted": after.to_dict() if after else None,
                         "expected": row["after"], "exact": exact, "error": error, "evidence": evidence})
        final_exact = not stopped and state.to_dict() == trace["steps"][-1]["after"]
        trace_correct += final_exact
        rows.append({"lane": "trace-summary", "id": trace["id"], "final_exact": final_exact,
                     "first_divergence": first_divergence})
    return {"isolated": {"correct": step_correct, "total": len(panel["observations"])},
            "rollout": {"correct": trace_step_correct, "total": trace_steps},
            "trace_endpoints": {"correct": trace_correct, "total": len(panel["traces"])},
            "fields": fields, "inference_ns": duration, "rows": rows}


def ratio(counts):
    return counts["correct"] / counts["total"] if counts["total"] else 0.0
