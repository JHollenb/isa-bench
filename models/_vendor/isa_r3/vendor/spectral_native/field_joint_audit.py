"""Post-training trace and causal audit of the accepted joint Field states.

Gold records and traces are used only here, after checkpoints are immutable.
Road-label normalization is determined from complete edge records, never by
choosing the action permutation that maximizes a trace score.
"""

from __future__ import annotations

import json

import torch
from scripts.run_recurrent_rule import load_state

from spectral_native.joint_field_reader import ASSIGNMENTS, JointFieldReader, prepare_cell
from spectral_native.recurrent_rule import rule_batch
from spectral_native.recurrent_rule_split import PERM_HELDOUT


def trace_diagnostics(model, cells, truths):
    total = {
        "n": 0,
        "terminal": 0,
        "node_trace": 0,
        "bit_trace": 0,
        "canonical_trace": 0,
        "road_labels_identified": 0,
        "normalized_trace": 0,
    }
    rows = []
    for cell, truth in zip(cells, truths, strict=True):
        batch = model.hard_batch(cell)
        out = model.program.rollout(batch, budget=16)
        moves = batch.moves
        stopped = out["stop_step"] == moves
        nodes = (out["nodes"][:, : moves + 1] == truth.truth_nodes).all(-1)
        bits = (out["bits"][:, : moves + 1] == truth.truth_bits).all(-1)
        raw_actions = out["actions"][:, : moves + 1]
        actions = (raw_actions == truth.truth_actions).all(-1)
        edge_matches = (batch.edges[:, :, None, :] == truth.edges[:, None, :, :]).all(-1)
        identified = (edge_matches.sum(-1) == 1).all(-1)
        labels = edge_matches.long().argmax(-1)
        normalized = torch.where(raw_actions < 2, labels.gather(1, raw_actions.clamp(0, 1)), raw_actions)
        normalized_trace = nodes & bits & stopped & identified & (normalized == truth.truth_actions).all(-1)
        terminal = (out["node"] == truth.target_nodes) & (out["bit"] == truth.target_bits) & stopped
        metrics = {
            "terminal": terminal,
            "node_trace": nodes & stopped,
            "bit_trace": bits & stopped,
            "canonical_trace": nodes & bits & stopped & actions,
            "road_labels_identified": identified,
            "normalized_trace": normalized_trace,
        }
        total["n"] += len(batch.starts)
        for name, values in metrics.items():
            total[name] += int(values.sum())
        for i in range(len(batch.starts)):
            rows.append(
                {
                    "moves": moves,
                    "i": i,
                    "latent_road_labels": labels[i].tolist() if identified[i] else None,
                    **{name: bool(values[i]) for name, values in metrics.items()},
                }
            )
    return {"summary": total, "rows": rows}


def joint_audit(report_path, *, seed, cases):
    from saturn.ar.training_checkpoint_controller import TrainingStateBoundary

    report = json.loads(report_path.read_text())
    source = report["result"]
    if source["seed"] != seed or source["cases"] != cases:
        raise ValueError("audit geometry must match immutable source report")
    depths = (0, 1, 2, 3, 4, 6, 8, 10)
    cells = [prepare_cell(16, m, cases, seed * 1_000_000 + 700_000 + m, PERM_HELDOUT) for m in depths]
    truths = [rule_batch(16, m, cases, seed * 1_000_000 + 700_000 + m, PERM_HELDOUT) for m in depths]
    arms = {}
    for label, record in source["arms"].items():
        model = JointFieldReader(load_state(record["accepted_state"]))
        with torch.no_grad():
            model.reader_logits.copy_(torch.tensor(record["reader_logits"]))
        diagnostic = trace_diagnostics(model, cells, truths)
        diagnostic["original_audit_equal"] = model.score(cells, audit=True) == record["audit"]
        arms[label] = diagnostic
    record = source["arms"]["joint"]
    model = JointFieldReader(load_state(record["accepted_state"]))
    with torch.no_grad():
        model.reader_logits.copy_(torch.tensor(record["reader_logits"]))
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.08)
    boundary = TrainingStateBoundary(
        model=model, optimizer=optimizer, cursor_capture=lambda: {}, cursor_restore=lambda _: None
    )
    parent = boundary.capture()
    interventions = {}
    reverse = [ASSIGNMENTS.index(tuple(reversed(a))) for a in ASSIGNMENTS]
    for label, reader_swap, policy_swap in (
        ("reader_only_swap", True, False),
        ("policy_only_swap", False, True),
        ("both_swap", True, True),
    ):
        boundary.restore(parent)
        with torch.no_grad():
            if reader_swap:
                model.reader_logits.copy_(model.reader_logits[reverse].clone())
            if policy_swap:
                model.program.transition_logits.copy_(model.program.transition_logits[:, :, [1, 0, 2], :].clone())
        interventions[label] = {
            "audit": trace_diagnostics(model, cells, truths)["summary"],
            "restored_exact": boundary.restore(parent) == parent.fingerprint,
        }
    return {
        "source_request_sha256": report["request_sha256"],
        "arms": arms,
        "interventions": interventions,
        "boundary": (
            "post-training diagnostic; no optimization or checkpoint selection; original action-trace score retained"
        ),
        "summary": {label: row["summary"] for label, row in arms.items()},
    }
