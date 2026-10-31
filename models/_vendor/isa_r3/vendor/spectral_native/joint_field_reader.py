"""Finite text-role routing and policy learning from one terminal answer signal.

Class/template induction, typed inventories and learned Field routes are supplied.
Three text-derived room relations include an irrelevant tunnel. Six candidate
A/B role assignments remain latent; reader logits and the transition table
can receive gradients through the same marginal terminal likelihood.
"""

from __future__ import annotations

import itertools
import json
import tempfile
from dataclasses import fields, replace

import torch
from torch import nn

from .recurrent_rule import RuleBatch, rule_batch
from .text_rule_world import TextRuleReader, TextWorld, as_batch, compile_world, fill, room, world_sentences

ASSIGNMENTS = tuple(itertools.permutations(range(3), 2))


def candidate_records(world: TextWorld, *, seed: int):
    generator = torch.Generator().manual_seed(seed)
    tunnel = torch.randint(world.width, (world.width,), generator=generator)
    text = world_sentences(world, "familiar", order_seed=seed)
    for i, target in enumerate(tunnel.tolist()):
        for template in ("the hidden tunnel of {X} ends at {Y} .", "from {X} the hidden tunnel runs to {Y} ."):
            text.append(fill(template, X=room(i), Y=room(target)))
    return candidate_records_from_text(text, width=world.width)


def candidate_records_from_text(text, *, width: int):
    """Compile supplied text using only its declared room inventory.

    This boundary accepts no generator edges, symbols, meanings or targets.
    It is shared by sentence-definition and demonstration integrations.
    """
    from types import SimpleNamespace

    reader = TextRuleReader(text)
    compiled = compile_world(reader, SimpleNamespace(width=width))
    inventory = {room(i): i for i in range(width)}
    relations = []
    for relation in reader.p.relations:
        if len(relation.roles) != 2:
            continue
        facts = reader.facts(relation)
        if not facts or not all(x in inventory and y in inventory for x, y in facts.items()):
            continue
        signature = tuple(
            sorted(
                tuple("<slot>" if x.startswith("<") else x for x in reader.p.by_id[tid].skeleton)
                for tid in relation.members
            )
        )
        values = torch.full((width,), -1, dtype=torch.long)
        for x, y in facts.items():
            values[inventory[x]] = inventory[y]
        relations.append((signature, values))
    relations.sort(key=lambda row: row[0])
    valid = len(relations) == 3 and all(bool((values >= 0).all()) for _, values in relations)
    bank = (
        torch.stack([values for _, values in relations])
        if valid
        else torch.full((3, width), -1, dtype=torch.long)
    )
    return compiled, bank, {"valid": valid, "signatures": [signature for signature, _ in relations]}


def prepare_cell(width, moves, count, seed, permutations):
    original = rule_batch(width, moves, count, seed, permutations)
    worlds = [
        TextWorld(
            original.edges[i],
            original.symbols[i],
            original.defs[i],
            int(original.starts[i]),
            int(original.initial_bits[i]),
            moves,
        )
        for i in range(count)
    ]
    compiled, banks, diagnostics = [], [], []
    for i, world in enumerate(worlds):
        c, bank, diagnostic = candidate_records(world, seed=seed + i + 999)
        compiled.append(c)
        banks.append(bank)
        diagnostics.append(diagnostic)
    return {
        "batch": as_batch(worlds, compiled),
        "banks": torch.stack(banks),
        "valid": torch.tensor([d["valid"] for d in diagnostics]),
        "diagnostics": diagnostics,
    }


def subset(cell, indices):
    return {
        "batch": RuleBatch(**{f.name: getattr(cell["batch"], f.name)[indices] for f in fields(RuleBatch)}),
        "banks": cell["banks"][indices],
        "valid": cell["valid"][indices],
    }


def assigned_batch(cell, assignment):
    return replace(cell["batch"], edges=cell["banks"][:, list(assignment)])


class JointFieldReader(nn.Module):
    def __init__(self, program):
        super().__init__()
        self.program = program
        self.reader_logits = nn.Parameter(torch.tensor([3.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    def loss(self, cell):
        batch = cell["batch"]
        count = len(batch.starts)
        choices = [assigned_batch(cell, a) for a in ASSIGNMENTS]
        merged = RuleBatch(**{f.name: torch.cat([getattr(b, f.name) for b in choices]) for f in fields(RuleBatch)})
        # Sanitization is confined to the soft surrogate. Invalid parses have
        # exactly zero answer mass and retain sentinels in hard execution.
        merged = replace(merged, edges=merged.edges.clamp_min(0))
        terminal = self.program.terminal_probabilities(merged, budget=8)
        index = torch.arange(len(merged.starts))
        probability = terminal[index, merged.target_nodes, merged.target_bits].reshape(len(ASSIGNMENTS), count)
        marginal = (self.reader_logits.softmax(0)[:, None] * probability).sum(0) * cell["valid"]
        return -marginal.clamp_min(1e-12).log().mean()

    def hard_batch(self, cell):
        return assigned_batch(cell, ASSIGNMENTS[int(self.reader_logits.argmax())])

    def score(self, cells, *, audit=False):
        total = {
            "correct": 0,
            "endpoint_correct": 0,
            "trace_correct": 0 if audit else None,
            "n": 0,
            "instrument_valid": 0,
            "cells": [],
        }
        for cell in cells:
            batch = self.hard_batch(cell)
            out = self.program.rollout(batch, budget=16)
            moves = batch.moves
            correct = (
                (out["node"] == batch.target_nodes) & (out["bit"] == batch.target_bits) & (out["stop_step"] == moves)
            )
            endpoint = (out["node"] == batch.target_nodes) & (out["bit"] == batch.target_bits)
            trace = None
            if audit:
                trace = (
                    (out["nodes"][:, : moves + 1] == batch.truth_nodes).all(-1)
                    & (out["bits"][:, : moves + 1] == batch.truth_bits).all(-1)
                    & (out["actions"][:, : moves + 1] == batch.truth_actions).all(-1)
                    & (out["stop_step"] == moves)
                )
            total["correct"] += int(correct.sum())
            total["endpoint_correct"] += int(endpoint.sum())
            if audit:
                total["trace_correct"] += int(trace.sum())
            total["n"] += len(batch.starts)
            total["instrument_valid"] += int(cell["valid"].sum())
            total["cells"].append(
                {
                    "width": batch.edges.shape[-1],
                    "moves": moves,
                    "correct": correct.int().tolist(),
                    "trace": trace.int().tolist() if audit else None,
                    "valid": cell["valid"].int().tolist(),
                }
            )
        total["accuracy"] = total["correct"] / total["n"]
        return total


def joint_panel(
    policy_path, seed, *, cases=64, futures=30, interval_steps=10, train_count=32, discovery_count=16, receipt_root=None
):
    from pathlib import Path

    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
    )
    from scripts.run_recurrent_rule import _state, load_state

    from spectral_native.recurrent_rule_split import PERM_HELDOUT, PERM_TRAIN

    torch.manual_seed(seed)
    parent_state = json.loads(policy_path.read_text())["arms"]["rule"]["state"]
    model = JointFieldReader(load_state(parent_state))
    parent = {k: v.clone() for k, v in model.state_dict().items()}
    random_table = torch.randn(model.program.transition_logits.shape) * 0.02
    receipt_root = receipt_root or Path(tempfile.mkdtemp(prefix="field-joint-reader-"))
    train = [
        prepare_cell(w, m, train_count, seed * 1_000_000 + 600_000 + w * 100 + m, PERM_TRAIN)
        for w in (8, 12)
        for m in (0, 1, 2, 3)
    ]
    discovery = [
        prepare_cell(w, m, discovery_count, seed * 1_000_000 + 650_000 + w * 100 + m, PERM_TRAIN)
        for w in (8, 12)
        for m in (0, 1, 2, 3)
    ]
    arms = {}
    for arm in ("fixed_reader_and_policy", "reader_only", "policy_only", "joint"):
        model.load_state_dict(parent)
        for head in model.program.heads:
            head.requires_grad_(False)
        model.reader_logits.requires_grad_(arm in ("reader_only", "joint"))
        model.program.transition_logits.requires_grad_(arm in ("policy_only", "joint"))
        if arm in ("policy_only", "joint"):
            with torch.no_grad():
                model.program.transition_logits.copy_(random_table)
        history, updates = [], {"accepted": 0, "physical": 0}

        def evaluate():
            with torch.no_grad():
                loss = sum(float(model.loss(c)) for c in discovery) / len(discovery)
                scores = model.score(discovery)
            return EvaluationScores(
                validation={"answer_nll": loss},
                autonomous_rollout={"terminal": scores["accuracy"]},
                trace_endpoints={"endpoint": scores["endpoint_correct"] / scores["n"]},
                evidence={
                    "reader_probabilities": model.reader_logits.softmax(0).tolist(),
                    "reader_assignment": list(ASSIGNMENTS[int(model.reader_logits.argmax())]),
                    "policy_diagnostic_only": model.program.policy_metrics(),
                    "execution_state": _state(model.program, seed),
                    "instrument_valid": scores["instrument_valid"],
                },
            )

        baseline = evaluate().to_dict()
        if arm != "fixed_reader_and_policy":
            optimizer = torch.optim.Adamax([p for p in model.parameters() if p.requires_grad], lr=0.08)

            def restore_cursor(state, updates=updates):
                updates["accepted"] = state["accepted"]

            controller = DeterministicCheckpointController(
                controller_id=f"joint-text-field-{arm}-{seed}",
                model=model,
                optimizer=optimizer,
                cursor_capture=lambda updates=updates: {"accepted": updates["accepted"]},
                cursor_restore=restore_cursor,
                max_interval_steps=interval_steps,
                policy=PromotionPolicy(
                    kind="lexicographic",
                    objectives=(
                        PromotionObjective("validation", "answer_nll", "minimize", tolerance=1e-6),
                        PromotionObjective("autonomous_rollout", "terminal", "maximize"),
                        PromotionObjective("trace_endpoints", "endpoint", "maximize"),
                    ),
                ),
                receipt_store=ImmutableDirectoryReceiptStore(receipt_root / arm),
            )
            controller.establish_baseline(evaluate)

            def train_step(_, updates=updates, optimizer=optimizer):
                cell = train[updates["accepted"] % len(train)]
                generator = torch.Generator().manual_seed(seed * 10000 + updates["physical"])
                indices = torch.randperm(len(cell["batch"].starts), generator=generator)[:16]
                loss = model.loss(subset(cell, indices))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                reader_grad = float(model.reader_logits.grad.norm()) if model.reader_logits.grad is not None else 0.0
                policy_grad = (
                    float(model.program.transition_logits.grad.norm())
                    if model.program.transition_logits.grad is not None
                    else 0.0
                )
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                updates["accepted"] += 1
                updates["physical"] += 1
                return {
                    "loss": float(loss.detach()),
                    "reader_gradient_norm": reader_grad,
                    "policy_gradient_norm": policy_grad,
                }

            for interval in range(futures):
                receipt = controller.run_interval(
                    interval_id=f"future-{interval}", steps=interval_steps, train_step=train_step, evaluator=evaluate
                ).to_dict()
                history.append(receipt)
                scores = receipt["candidate"]["evaluation"]
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "future": interval,
                            "decision": receipt["decision"],
                            "nll": scores["validation"],
                            "terminal": scores["autonomous_rollout"],
                            "reader": scores["evidence"]["reader_assignment"],
                        }
                    ),
                    flush=True,
                )
                if evaluate().to_dict()["autonomous_rollout"]["terminal"] == 1.0:
                    break
        arms[arm] = {
            "baseline": baseline,
            "history": history,
            "physical_updates": updates["physical"],
            "accepted_updates": updates["accepted"],
            "discovery": evaluate().to_dict(),
            "accepted_state": _state(model.program, seed),
            "reader_logits": model.reader_logits.detach().tolist(),
        }
    # The audit is constructed only after every arm's final state is fixed.
    audit = [
        prepare_cell(16, m, cases, seed * 1_000_000 + 700_000 + m, PERM_HELDOUT) for m in (0, 1, 2, 3, 4, 6, 8, 10)
    ]
    for record in arms.values():
        model.program.load_state_dict(load_state(record["accepted_state"]).state_dict())
        with torch.no_grad():
            model.reader_logits.copy_(torch.tensor(record["reader_logits"]))
        record["audit"] = model.score(audit, audit=True)
    return {
        "arms": arms,
        "seed": seed,
        "cases": cases,
        "boundary": (
            "answer feedback jointly trains six role-assignment logits and the finite transition table; "
            "class induction and learned routes frozen; repeated generated text and typed compiler inventory supplied"
        ),
        "cache": {
            "training_worlds": 8 * train_count,
            "discovery_worlds": 8 * discovery_count,
            "audit_worlds": 8 * cases,
            "physical_candidate_worlds_per_training_batch": 96,
            "candidate_assignments": [list(a) for a in ASSIGNMENTS],
        },
        "summary": {
            k: {
                "audit": [v["audit"]["correct"], v["audit"]["n"]],
                "trace": v["audit"]["trace_correct"],
                "instrument_valid": v["audit"]["instrument_valid"],
                "physical_updates": v["physical_updates"],
                "reader_assignment": list(ASSIGNMENTS[max(range(6), key=lambda i: v["reader_logits"][i])]),
            }
            for k, v in arms.items()
        },
    }
