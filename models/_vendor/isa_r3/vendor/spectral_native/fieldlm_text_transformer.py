"""Matched byte-level Transformer baseline for the real-text field LM comparison.

Decoder-only, pre-LayerNorm, rotary positions (RoPE), GELU MLP, embedding tied
to the output.  Trained on random windows of the training split under Saturn's
deterministic checkpoint controller: every interval is a transacted future that
is promoted only if validation bits per byte improve, otherwise rolled back;
training stops after two consecutive rejections (early stopping).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf


@dataclass(frozen=True)
class GPTConfig:
    d_model: int
    layers: int
    heads: int
    context: int = 256


SIZES = {
    "small": GPTConfig(128, 4, 4),
    "large": GPTConfig(256, 6, 8),
}


def _rope(x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    freq = 1.0 / (10000 ** (torch.arange(half, device=x.device, dtype=torch.float32) / half))
    angle = positions[:, None].float() * freq[None, :]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    a, b = x[..., :half], x[..., half:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(), nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.cfg.heads
        q, k, v = self.qkv(self.ln1(x)).view(b, t, 3, h, d // h).unbind(2)
        pos = torch.arange(t, device=x.device)
        q = _rope(q.transpose(1, 2), pos)
        k = _rope(k.transpose(1, 2), pos)
        v = v.transpose(1, 2)
        att = nnf.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(att.transpose(1, 2).reshape(b, t, d))
        return x + self.mlp(self.ln2(x))


class ByteGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(256, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])
        self.ln = nn.LayerNorm(cfg.d_model)
        # GPT-2 style initialization; with a tied output, N(0, 1) embeddings
        # would start with logits of standard deviation sqrt(d_model).
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x)
        return self.ln(x) @ self.embed.weight.T


def parameter_count(cfg: GPTConfig) -> int:
    return sum(p.numel() for p in ByteGPT(cfg).parameters())


def _windows(data: np.ndarray, context: int, batch: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, data.shape[0] - context - 1, size=batch)
    block = np.stack([data[s : s + context + 1] for s in starts]).astype(np.int64)
    return block[:, :-1], block[:, 1:]


@torch.no_grad()
def segment_bits(model: ByteGPT, data: np.ndarray, seg_len: int, device: torch.device, batch_tokens: int = 1 << 16) -> np.ndarray:
    """Bits for every position with local index >= 1 of each length-``seg_len`` segment.

    Matches the field model's position set (compute_charts(x, seg_len)).
    """

    was_training = model.training
    model.eval()
    n = data.shape[0]
    full = n // seg_len
    out = []
    rows = max(1, batch_tokens // seg_len)
    segs = data[: full * seg_len].reshape(full, seg_len).astype(np.int64)
    for i in range(0, full, rows):
        x = torch.tensor(segs[i : i + rows], device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x[:, :-1]).float()
        lp = nnf.log_softmax(logits, dim=-1).gather(-1, x[:, 1:, None])[..., 0]
        out.append((-lp / math.log(2)).reshape(-1).cpu().numpy())
    tail = data[full * seg_len :]
    if tail.shape[0] >= 2:
        x = torch.tensor(tail[None].astype(np.int64), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x[:, :-1]).float()
        lp = nnf.log_softmax(logits, dim=-1).gather(-1, x[:, 1:, None])[..., 0]
        out.append((-lp / math.log(2)).reshape(-1).cpu().numpy())
    model.train(was_training)
    return np.concatenate(out)


@torch.no_grad()
def sliding_bits(model: ByteGPT, data: np.ndarray, window: int, device: torch.device, rows: int = 256) -> np.ndarray:
    """Every position after the first ``window // 2`` scored with a sliding window
    of ``window`` bytes (stride window/2): the model never sees a context longer
    than it was trained on."""

    was_training = model.training
    model.eval()
    stride = window // 2
    starts = np.arange(0, data.shape[0] - window, stride)
    out = []
    for i in range(0, starts.size, rows):
        block = np.stack([data[s : s + window] for s in starts[i : i + rows]]).astype(np.int64)
        x = torch.tensor(block, device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x[:, :-1]).float()
        lp = nnf.log_softmax(logits, dim=-1).gather(-1, x[:, 1:, None])[..., 0]
        out.append((-lp[:, stride - 1 :] / math.log(2)).reshape(-1).cpu().numpy())
    model.train(was_training)
    return np.concatenate(out)


def train_transformer(
    cfg: GPTConfig,
    train: np.ndarray,
    valid: np.ndarray,
    lr: float,
    seed: int,
    device: torch.device,
    receipts: Path,
    batch: int = 64,
    interval_steps: int = 1000,
    max_intervals: int = 12,
    warmup: int = 200,
) -> tuple[ByteGPT, dict[str, Any]]:
    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
    )

    torch.manual_seed(seed)
    model = ByteGPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1, betas=(0.9, 0.95))
    cursor = {"step": 0}
    # The learning-rate backoff is outer-loop policy, not transacted training
    # state: keeping it out of the cursor keeps the live state equal to the
    # accepted checkpoint after a rollback (Saturn refuses drift).
    lr_factors = (1.0, 0.5, 0.25, 0.125)
    backoff = 0
    valid_probe = valid[: 256 * 256]
    last_loss = {"bits": float("nan")}

    def evaluator() -> EvaluationScores:
        bits = segment_bits(model, valid_probe, cfg.context, device)
        return EvaluationScores(
            validation={"valid_bits": float(bits.mean())},
            autonomous_rollout={"valid_frac_under_1bit": float((bits < 1.0).mean())},
            trace_endpoints={"train_bits": float(last_loss["bits"]) if math.isfinite(last_loss["bits"]) else 99.0},
            evidence={"step": cursor["step"]},
        )

    policy = PromotionPolicy(
        kind="lexicographic",
        objectives=(
            PromotionObjective("validation", "valid_bits", "minimize", tolerance=1e-4),
            PromotionObjective("autonomous_rollout", "valid_frac_under_1bit", "maximize"),
            PromotionObjective("trace_endpoints", "train_bits", "minimize"),
        ),
    )

    def restore(state: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(state)

    controller = DeterministicCheckpointController(
        controller_id=f"fieldlm-text-gpt-{cfg.d_model}x{cfg.layers}-lr{lr}-seed{seed}",
        model=model,
        optimizer=optimizer,
        policy=PromotionPolicy(kind=policy.kind, objectives=policy.objectives),
        cursor_capture=lambda: dict(cursor),
        cursor_restore=restore,
        max_interval_steps=interval_steps,
        receipt_store=ImmutableDirectoryReceiptStore(receipts),
    )
    model.train()
    baseline = controller.establish_baseline(evaluator).to_dict()["candidate"]["evaluation"]
    rows = []
    rejections = 0
    started = time.time()
    for interval in range(max_intervals):
        factor = lr_factors[min(backoff, len(lr_factors) - 1)]

        def step_fn(_i: int, factor: float = factor) -> dict[str, Any]:
            step = cursor["step"]
            warm = min(1.0, (step + 1) / warmup)
            for group in optimizer.param_groups:
                group["lr"] = lr * factor * warm
            x, y = _windows(train, cfg.context, batch, seed * 1_000_003 + step)
            xb = torch.tensor(x, device=device)
            yb = torch.tensor(y, device=device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(xb).float()
            loss = nnf.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            cursor["step"] = step + 1
            last_loss["bits"] = float(loss.detach()) / math.log(2)
            return {"train_bits": last_loss["bits"]}

        receipt = controller.run_interval(
            interval_id=f"interval-{interval + 1}", steps=interval_steps, train_step=step_fn, evaluator=evaluator
        ).to_dict()
        promoted = bool(receipt["decision"]["promoted"])
        rows.append({
            "interval": interval + 1,
            "promoted": promoted,
            "lr_factor": factor,
            "valid_bits": receipt["candidate"]["evaluation"]["validation"]["valid_bits"],
            "rollback_verified_exact": bool(receipt["rollback"]["verified_exact"]),
        })
        if promoted:
            rejections = 0
        else:
            rejections += 1
            backoff += 1
            if rejections >= 2:
                break
    model.eval()
    info = {
        "config": cfg.__dict__,
        "lr": lr,
        "parameters": sum(p.numel() for p in model.parameters()),
        "baseline_valid_bits": baseline["validation"]["valid_bits"],
        "intervals": rows,
        "accepted_steps": cursor["step"],
        "seconds": time.time() - started,
        "all_rollbacks_exact": all(r["rollback_verified_exact"] for r in rows),
    }
    return model, info


def finetune_edit(
    model: ByteGPT,
    windows: list[np.ndarray],
    replay: np.ndarray | None,
    steps: int,
    lr: float,
    device: torch.device,
    seed: int,
    batch: int = 32,
) -> dict[str, Any]:
    """Knowledge edit for the Transformer: fine-tune on edited windows (optionally with replay)."""

    rng = np.random.default_rng(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    started = time.time()
    context = model.cfg.context
    for _ in range(steps):
        rows = []
        for _ in range(batch):
            if replay is not None and rng.random() < 0.5:
                s = int(rng.integers(0, replay.shape[0] - context - 1))
                rows.append(replay[s : s + context + 1])
            else:
                rows.append(windows[int(rng.integers(len(windows)))])
        block = torch.tensor(np.stack(rows).astype(np.int64), device=device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(block[:, :-1]).float()
        loss = nnf.cross_entropy(logits.reshape(-1, 256), block[:, 1:].reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return {"steps": steps, "lr": lr, "seconds": time.time() - started, "replay": replay is not None}
