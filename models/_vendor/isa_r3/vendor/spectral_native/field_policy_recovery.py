"""Same-parent policy futures with frozen learned routes and answer-only selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from scripts.run_recurrent_rule import _state, causal_panel, data, load_state, metrics, set_split

from spectral_native.recurrent_rule import rule_batch
from spectral_native.recurrent_rule_split import PERM_TRAIN


def recovery_panel(
    report_path: Path,
    audit_seed: int,
    *,
    cases: int = 64,
    futures: int = 12,
    interval_override: int | None = None,
    receipt_root: Path | None = None,
):
    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
        TrainingStateBoundary,
    )

    source = json.loads(report_path.read_text())["arms"]["rule"]
    set_split("parity")
    torch.manual_seed(731)
    model = load_state(source["state"])
    for head in model.heads:
        head.requires_grad_(False)
    optimizer = torch.optim.Adamax([model.transition_logits], lr=0.08)
    cursor = {"updates": 0}

    def restore_cursor(state):
        cursor.clear()
        cursor.update(state)

    boundary = TrainingStateBoundary(
        model=model, optimizer=optimizer, cursor_capture=lambda: dict(cursor), cursor_restore=restore_cursor
    )
    parent = boundary.capture()
    receipt_root = receipt_root or Path(tempfile.mkdtemp(prefix="field-policy-recovery-"))
    discovery = [data(2, "discovery", w, m, 32) for w in (8, 12) for m in (0, 1, 2, 3)]
    audit = [data(audit_seed, "heldout_unseen", 16, m, cases) for m in (0, 1, 2, 3, 4, 6, 8, 10)]
    baseline_audit = metrics(model, audit)
    config = [
        ("ten_step", 10, "none", "hard"),
        ("forty_step", 40, "none", "hard"),
        ("table_restart", 10, "all", "hard"),
        ("uncertain_entry_restart", 40, "entropy", "hard"),
        ("loss_first_forty", 40, "none", "loss"),
    ]
    branches = {}
    for label, interval_steps, reset, priority in config:
        interval_steps = interval_override or interval_steps
        restored = boundary.restore(parent)
        if reset != "none":
            generator = torch.Generator().manual_seed(991)
            fresh = torch.randn(model.transition_logits.shape, generator=generator) * 0.02
            with torch.no_grad():
                if reset == "all":
                    model.transition_logits.copy_(fresh)
                else:
                    probabilities = model.transition_probabilities[:4].reshape(8, 6)
                    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
                    for index in entropy.topk(2).indices.tolist():
                        row, bit = divmod(index, 2)
                        model.transition_logits[row, bit].copy_(fresh[row, bit])
        hard = PromotionObjective("autonomous_rollout", "terminal", "maximize")
        ce = PromotionObjective("validation", "ce", "minimize", tolerance=1e-6)
        endpoint = PromotionObjective("trace_endpoints", "node_bit", "maximize")
        objectives = (hard, ce, endpoint) if priority == "hard" else (ce, hard, endpoint)
        controller = DeterministicCheckpointController(
            controller_id=f"field-policy-recovery-{label}",
            model=model,
            optimizer=optimizer,
            cursor_capture=lambda: dict(cursor),
            cursor_restore=restore_cursor,
            policy=PromotionPolicy(kind="lexicographic", objectives=objectives),
            max_interval_steps=interval_steps,
            receipt_store=ImmutableDirectoryReceiptStore(receipt_root / label),
        )

        def evaluate():
            prior = model.training
            model.eval()
            try:
                with torch.no_grad():
                    ce_value = sum(float(model.loss(b, budget=8)) for b in discovery) / len(discovery)
                    scores = metrics(model, discovery)
                return EvaluationScores(
                    validation={"ce": ce_value},
                    autonomous_rollout={"terminal": scores["terminal_accuracy"]},
                    trace_endpoints={"node_bit": scores["node_bit_accuracy"]},
                    evidence={
                        "execution_state": _state(model, 2),
                        "policy_diagnostic_only": model.policy_metrics(),
                        "routes_diagnostic_only": model.route_accuracy(16),
                    },
                )
            finally:
                model.train(prior)

        baseline = controller.establish_baseline(evaluate).to_dict()
        history, attempts = [], {"physical": 0}

        def train_step(_, attempts=attempts):
            update = cursor["updates"]
            width, moves = (8, 12)[update % 2], (update // 2) % 4
            batch = rule_batch(width, moves, 32, audit_seed * 1_000_000 + 900_000 + attempts["physical"], PERM_TRAIN)
            loss = model.loss(batch, budget=8)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cursor["updates"] += 1
            attempts["physical"] += 1
            return {"loss": float(loss.detach()), "physical_update": attempts["physical"]}

        for interval in range(futures):
            receipt = controller.run_interval(
                interval_id=f"future-{interval}", steps=interval_steps, train_step=train_step, evaluator=evaluate
            ).to_dict()
            history.append(receipt)
            evaluation = receipt["candidate"]["evaluation"]
            print(
                json.dumps(
                    {
                        "branch": label,
                        "future": interval,
                        "physical_updates": attempts["physical"],
                        "decision": receipt["decision"],
                        "scores": evaluation["autonomous_rollout"],
                        "loss": evaluation["validation"],
                    }
                ),
                flush=True,
            )
            if evaluate().to_dict()["autonomous_rollout"]["terminal"] == 1.0:
                break
        state = _state(model, 2)
        measured = metrics(model, audit)
        replay = metrics(load_state(state), audit)
        branch = {
            "parent_restored_exact": restored == parent.fingerprint,
            "interval_steps": interval_steps,
            "reset": reset,
            "priority": priority,
            "baseline": baseline,
            "history": history,
            "physical_updates": attempts["physical"],
            "accepted_updates": cursor["updates"],
            "accepted_state": state,
            "discovery": evaluate().to_dict(),
            "audit": measured,
            "state_rehydration_equal": measured == replay,
            "policy_diagnostic_only": model.policy_metrics(),
            "causal": causal_panel(model, audit_seed, count=16),
        }
        branches[label] = branch
        branch["parent_restored_after_exact"] = boundary.restore(parent) == parent.fingerprint
    return {
        "optimizer_contract": "fresh Adamax; historical optimizer and RNG were not retained in source report",
        "parent_state": source["state"],
        "parent_fingerprint": parent.fingerprint,
        "baseline_audit": baseline_audit,
        "branches": branches,
        "summary": {
            k: {
                "audit": [v["audit"]["correct"], v["audit"]["n"]],
                "trace": v["audit"]["trace_correct"],
                "physical_updates": v["physical_updates"],
                "accepted_updates": v["accepted_updates"],
                "parent_restored_exact": v["parent_restored_after_exact"],
            }
            for k, v in branches.items()
        },
    }
