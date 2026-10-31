"""Bounded Saturn training transactions with discovery-only fitting.

Feedback labels select continuation checkpoints. Sealed heldout/terminal panels
are absent from this API. Rejected artifacts and consumer measurements survive.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from .contracts import PROFILE, TRACK
from .data import digest
from .dependencies import REPO, fingerprints, saturn_controller
from .evaluate import evaluate, ratio
from .measurement import elapsed, stamp
from .models import create
from .profile import infer_routes


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def contract_fingerprint():
    return digest({name: (REPO / "isa_lab" / name).read_text()
                   for name in ("contracts.py", "profile.py", "evaluate.py", "data.py")})


def artifact(model, seed, suite_sha256):
    payload = {"schema": "isa-model-v1", "profile": PROFILE, "track": TRACK, "model": model.name,
               "seed": seed, "suite_sha256": suite_sha256, "contract_sha256": contract_fingerprint(),
               "payload": model.export()}
    payload["sha256"] = digest(payload)
    return payload


def fit(name, discovery, feedback, *, suite_sha256, seed, out, intervals=4, steps=32, learning_rate=0.003,
        on_event=None):
    """CPU starter lane. The caller provides only the two authorized panels."""
    if intervals < 1 or steps < 1 or not 0 < learning_rate <= 1:
        raise ValueError("invalid bounded training budget")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    saturn = saturn_controller()
    timing = {}
    start = stamp()
    model = create(name, seed, "cpu")
    timing["model_prepare"] = elapsed(start)
    start = stamp()
    routes = infer_routes(discovery["observations"])
    timing["label_prepare"] = elapsed(start)
    timing["updates"] = {"wall_s": 0.0, "cpu_s": 0.0}
    timing["feedback"] = {"wall_s": 0.0, "cpu_s": 0.0}
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if name == "nn" else None
    cursor = {"accepted_updates": 0}
    evaluations = 0
    physical_updates = 0
    candidate_artifacts = []

    def restore_cursor(value):
        cursor.clear()
        cursor.update(value)

    objectives = (
        saturn.PromotionObjective("validation", "isolated_exact", "maximize"),
        saturn.PromotionObjective("autonomous_rollout", "step_exact", "maximize"),
        saturn.PromotionObjective("trace_endpoints", "final_exact", "maximize"),
    )
    if name == "nn":
        objectives += (saturn.PromotionObjective("validation", "discovery_ce_loss", "minimize"),)
    policy = saturn.PromotionPolicy("pareto", objectives)
    controller = saturn.DeterministicCheckpointController(
        controller_id=f"isa-{name}-{seed}", model=model, optimizer=optimizer, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore_cursor, max_interval_steps=steps,
        receipt_store=saturn.ImmutableDirectoryReceiptStore(out / "decisions"))

    def evaluator():
        nonlocal evaluations
        start = stamp()
        measured = evaluate(model, feedback)
        discovery_loss = model.discovery_loss(discovery["observations"], routes) if name == "nn" else None
        cost = elapsed(start)
        for key in cost:
            timing["feedback"][key] += cost[key]
        tag = f"future-{evaluations:03d}"
        write_json(out / f"{tag}-feedback.json", measured)
        if discovery_loss is not None:
            write_json(out / f"{tag}-discovery-progress.json", {"split": "discovery", "ce_loss": discovery_loss})
        future = artifact(model, seed, suite_sha256)
        write_json(out / f"{tag}-model.json", future)
        candidate_artifacts.append({"id": tag, "sha256": future["sha256"]})
        if on_event:
            on_event({"event": "feedback", "model": name, "seed": seed, "step": evaluations,
                      "isolated_exact": ratio(measured["isolated"]),
                      "rollout_exact": ratio(measured["rollout"]),
                      "final_exact": ratio(measured["trace_endpoints"]), "discovery_ce_loss": discovery_loss, "cost": cost})
        evaluations += 1
        validation = {"isolated_exact": ratio(measured["isolated"])}
        if discovery_loss is not None:
            validation["discovery_ce_loss"] = discovery_loss
        return saturn.EvaluationScores(validation=validation,
                    autonomous_rollout={"step_exact": ratio(measured["rollout"])},
                    trace_endpoints={"final_exact": ratio(measured["trace_endpoints"])},
                    evidence={"consumer_split": "feedback", "progress_split": "discovery" if name == "nn" else None,
                              "labels_used_for_continuation": True, "artifact": tag})

    def update(_):
        nonlocal physical_updates
        start = stamp()
        result = model.update(discovery["observations"], routes, optimizer)
        cost = elapsed(start)
        for key in cost:
            timing["updates"][key] += cost[key]
        cursor["accepted_updates"] += 1
        physical_updates += 1
        if on_event:
            on_event({"event": "update", "model": name, "seed": seed,
                      "step": physical_updates, "cost": cost, **result})
        return result

    started = time.perf_counter_ns()
    controller.establish_baseline(evaluator)
    for interval in range(intervals if name == "nn" else 1):
        decision = controller.run_interval(interval_id=f"interval-{interval:03d}", steps=steps if name == "nn" else 1,
                                           train_step=update, evaluator=evaluator)
        if on_event:
            on_event({"event": "decision", "model": name, "seed": seed, "step": interval,
                      **decision.to_dict()["decision"]})
    controller_elapsed_ns = time.perf_counter_ns() - started
    final = artifact(model, seed, suite_sha256)
    write_json(out / "model.json", final)
    summary = {"schema": "isa-training-v1", "model": name, "seed": seed, "suite_sha256": suite_sha256,
               "contract_sha256": contract_fingerprint(), "backend": "cpu", "training_ns": controller_elapsed_ns,
               "physical_updates": physical_updates, "accepted_updates": cursor["accepted_updates"],
               "discovery_examples": len(discovery["observations"]), "feedback_evaluations": evaluations,
               "candidate_artifacts": candidate_artifacts, "policy": policy.to_dict(),
               "timing": timing,
               "rollback_receipts": [r.to_dict() for r in controller.ledger.receipts],
               "mechanics_status": "valid", "instrument_status": "consumer-observations-recorded",
               "trend_status": "observation-only", "terminal_status": "not-certified",
               "environment": {"python": platform.python_version(), "torch": torch.__version__,
                               "machine": platform.machine(), "platform": platform.platform()}}
    write_json(out / "training.json", summary)
    write_json(out / "source-manifest.json", fingerprints())
    return model, summary
