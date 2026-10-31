"""One resident six-head model, bounded Saturn futures and quarantined feedback."""
from __future__ import annotations

import time
from pathlib import Path

import torch

from isa_lab.data import digest
from isa_lab.dependencies import saturn_controller
from isa_lab.full_evaluate import evaluate, scores
from isa_lab.full_models import TrainableSelectors, create
from isa_lab.full_scope import PROFILE, Consumer, stage_rows
from isa_lab.training import write_json


def model_artifact(model, seed, suite_hash):
    result = {"schema": "isa-full-model-v1", "profile": PROFILE, "name": model.name,
              "seed": seed, "suite_sha256": suite_hash, "payload": model.export()}
    result["sha256"] = digest(result)
    return result


def restore(value):
    expected = value["sha256"]
    if digest({k: v for k, v in value.items() if k != "sha256"}) != expected:
        raise ValueError("model artifact identity mismatch")
    model = create(value["name"], value["seed"])
    model.restore_export(value["payload"])
    return model


def fit(name, discovery, feedback, *, seed, suite_hash, out, intervals=4, steps=64, emit=print):
    out = Path(out)
    out.mkdir(exist_ok=False, parents=True)
    model = create(name, seed)
    consumer = Consumer()
    rows = {k: stage_rows(v) for k, v in discovery["stages"].items()}
    rows["runtime"] = [{k: r[k] for k in ("raw", "numeric", "target")} for r in discovery["runtime"] if "raw" in r]
    rows["runtime"].extend({k: r[k] for k in ("raw", "numeric", "target")}
                           for r in discovery.get("runtime_discovery_rows", []))
    neural = isinstance(model, TrainableSelectors)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001) if neural else None
    saturn = saturn_controller()
    cursor = {"updates": 0}
    physical = 0
    observations = []
    # Consumer trends remain observations. Discovery loss chooses the next reversible
    # branch; a temporary consumer regression does not erase the trajectory.
    objectives = ((saturn.PromotionObjective("validation", "discovery_loss", "minimize"),
                   saturn.PromotionObjective("autonomous_rollout", "rollout", "maximize"),
                   saturn.PromotionObjective("trace_endpoints", "runtime", "maximize"))) if neural else (
        saturn.PromotionObjective("validation", "isolated", "maximize"),
        saturn.PromotionObjective("autonomous_rollout", "rollout", "maximize"),
        saturn.PromotionObjective("trace_endpoints", "runtime", "maximize"))
    policy = saturn.PromotionPolicy("lexicographic" if neural else "pareto", objectives)

    def restore_cursor(value):
        cursor.clear()
        cursor.update(value)

    controller = saturn.DeterministicCheckpointController(
        controller_id=f"isa-full-{name}-{seed}", model=model, optimizer=optimizer, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore_cursor, max_interval_steps=steps,
        receipt_store=saturn.ImmutableDirectoryReceiptStore(out / "decisions"))

    def evaluator():
        measured = evaluate(model, feedback, consumer, capture=False)
        observed = scores(measured)
        with torch.no_grad():
            loss = float(model.loss(rows)) if neural else 0.0
        tag = f"future-{len(observations):03d}"
        write_json(out / f"{tag}-feedback.json", measured)
        write_json(out / f"{tag}-model.json", model_artifact(model, seed, suite_hash))
        from isa_lab.full_custody import publish
        publish((out / f"{tag}-feedback.json", out / f"{tag}-model.json"), f"future-{name}-{seed}-{tag}")
        observations.append({**observed, "discovery_loss": loss, "physical_updates": physical})
        emit({"event": "future", "name": name, "seed": seed, "id": tag, **observations[-1]})
        return saturn.EvaluationScores(validation={"discovery_loss": loss, "isolated": observed["isolated"]},
            autonomous_rollout={"rollout": observed["rollout"]}, trace_endpoints={"runtime": observed["runtime"]},
            evidence={"feedback_split": "feedback", "optimizer_split": "discovery", "labels_used": True, "future": tag})

    update_wall = update_cpu = 0.0

    def update(_):
        nonlocal physical, update_wall, update_cpu
        t, c = time.perf_counter(), time.process_time()
        loss = model.update(rows, optimizer)
        update_wall += time.perf_counter() - t
        update_cpu += time.process_time() - c
        cursor["updates"] += 1
        physical += 1
        return {"loss": loss}

    controller.establish_baseline(evaluator)
    for interval in range(intervals if neural else 1):
        controller.run_interval(interval_id=f"interval-{interval:03d}", steps=steps if neural else 1,
                                train_step=update, evaluator=evaluator)
    artifact = model_artifact(model, seed, suite_hash)
    write_json(out / "model.json", artifact)
    summary = {"name": name, "seed": seed, "suite_sha256": suite_hash, "physical_updates": physical,
               "accepted_updates": cursor["updates"], "examples_per_update": {k: len(v) for k, v in rows.items()},
               "update_wall_s": update_wall, "update_cpu_s": update_cpu, "trajectory": observations,
               "rollback_receipts": [r.to_dict() for r in controller.ledger.receipts],
               "policy": policy.to_dict(), "mechanics_status": "valid", "terminal_status": "not-certified"}
    write_json(out / "training.json", summary)
    return model, consumer, summary
