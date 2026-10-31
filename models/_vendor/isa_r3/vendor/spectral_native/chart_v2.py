"""Chart field v2: fair baselines and controls for the learned-chart field LM.

Preregistered in ``docs/chart-field-v2-findings.md``.  Adds, next to ``chart_field`` and
``chart_dynamics``:

- ``SPFPhaseChart``: the SPF-native trunk as a chart encoder under ``train_dyn``.  The chart is an
  integer (ternary) torus endomorphism of the trunk's phase channels plus a phase offset,
  ``theta_chart = ternary(A) theta_trunk + b (mod 1)``, so it is well defined on the torus and exact
  on the ring.
- ``ContinuousBottleneck`` + ``train_bottleneck``: the Cartesian control.  Same encoder, an N-dim
  *linear* bottleneck (no wrap, no phases, no cells) and an MLP readout, trained on the same schedule.
- ``nested_pq_words``: product-quantized k-means on the frozen hidden state, one subspace per axis,
  with nested codes (a 3-bit coarse code whose two children form the 4-bit code), written as 16-bit
  words so the product-of-experts compile runs on it unchanged: a like-for-like quantizer control.
- Edit helpers: recount chain and product-of-experts tables from edited training text while keeping
  every fitted weight fixed (the field's edit semantics), scored per position.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from . import chart_field as C
from .model import T_Q, ste, ternary, wrap

# --------------------------------------------------------------------------- #
# SPF-native trunk as a chart encoder
# --------------------------------------------------------------------------- #


class SPFPhaseChart(nn.Module):
    """SPF trunk phases -> chart by an integer torus endomorphism (duck-types ``TiedChart``)."""

    def __init__(self, encoder: C.SPFEncoder, n: int, seed: int = 0) -> None:
        super().__init__()
        self.encoder, self.n, self.tie, self.m = encoder, n, "none", n
        for name, p in encoder.net.named_parameters():  # the trunk's own LM head is unused here: not counted, not trained
            if name.startswith("out_"):
                p.requires_grad_(False)
        self.register_buffer("omega", torch.eye(n))
        self.proj = nn.Linear(encoder.width, n)  # weight: latent of the ternary matrix A; bias: phase offset
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(n, encoder.width, generator=g))
            self.proj.bias.copy_(torch.rand(n, generator=g))

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        # phases stay float32 (bf16 is far coarser than the 16-bit ring)
        with torch.autocast(ids.device.type, enabled=False):
            _, th = self.encoder(ids)
        return th.float()

    def coords(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):
            return h @ ternary(self.proj.weight).T + self.proj.bias

    def from_coords(self, u: torch.Tensor) -> torch.Tensor:
        return ste(wrap(u @ self.omega), T_Q)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.from_coords(self.coords(self.features(ids)))


# --------------------------------------------------------------------------- #
# Cartesian control: continuous bottleneck
# --------------------------------------------------------------------------- #


class ContinuousBottleneck(nn.Module):
    """Encoder -> u in R^N (linear, no wrap) -> MLP readout.  No phases, no cells."""

    def __init__(self, encoder: nn.Module, n: int, hidden: int) -> None:
        super().__init__()
        self.encoder, self.n = encoder, n
        self.proj = nn.Linear(encoder.width, n)
        self.head = nn.Sequential(nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, 256))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        with C._amp(ids.device):
            h = self.encoder(ids)
        with torch.autocast(ids.device.type, enabled=False):
            return self.head(self.proj(h.float()))

    def readout_parameters(self) -> int:
        return sum(p.numel() for p in self.proj.parameters()) + sum(p.numel() for p in self.head.parameters())


class PlainLM(nn.Module):
    """A whole byte LM (ids -> logits) under ``train_bottleneck``: the SPF trunk's plain-LM twin, i.e. the
    same ``SpectralNative`` network trained with its own phase-kernel LM head instead of a chart."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.encoder, self.n = net, 0

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.autocast(ids.device.type, enabled=False):
            return self.encoder(ids).float()

    def readout_parameters(self) -> int:
        return sum(p.numel() for n, p in self.encoder.named_parameters() if n.startswith("out_"))


@torch.no_grad()
def bottleneck_target_probs(model: ContinuousBottleneck | PlainLM, segs: np.ndarray, device: torch.device, rows: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows], device=device)
        out.append(torch.softmax(model(x[:, :-1]), -1).gather(-1, x[:, 1:, None])[..., 0].cpu().numpy())
    return np.concatenate(out).reshape(-1).astype(np.float64)


def train_bottleneck(model: ContinuousBottleneck | PlainLM, train: np.ndarray, valid: np.ndarray, device: torch.device, receipts,
                     *, lr: float, batch: int, interval_steps: int, max_intervals: int, seed: int, accum: int = 1,
                     eval_rows: int = 64, log: Callable[[dict[str, Any], int], None] | None = None) -> dict[str, Any]:
    """Same schedule as ``chart_dynamics.train_dyn``: Saturn intervals, promotion on validation bits,
    learning rate halved per rejection (outside the cursor), stop after 3 consecutive rejections."""

    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
    )

    torch.manual_seed(seed)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adamax(params, lr=lr)
    cursor = {"step": 0}
    val_segs = C.segments(valid[: C.SEG * 256])
    last = {"bits": 99.0}

    def valid_bits() -> float:
        p = bottleneck_target_probs(model, val_segs, device, rows=eval_rows)
        model.train()
        return float(-np.log2(np.maximum(p, 1e-300)).mean())

    def evaluator() -> EvaluationScores:
        return EvaluationScores(validation={"valid_bits": valid_bits()}, autonomous_rollout={"neg_train_bits": -last["bits"]},
                                trace_endpoints={"train_bits": last["bits"]}, evidence={"step": cursor["step"]})

    policy = PromotionPolicy(kind="lexicographic", objectives=(
        PromotionObjective("validation", "valid_bits", "minimize", tolerance=1e-4),
        PromotionObjective("autonomous_rollout", "neg_train_bits", "maximize"),
        PromotionObjective("trace_endpoints", "train_bits", "minimize"),
    ))

    def restore(state: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(state)

    controller = DeterministicCheckpointController(
        controller_id=f"chart-bottleneck-N{model.n}-seed{seed}", model=model, optimizer=opt, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore, max_interval_steps=interval_steps,
        receipt_store=ImmutableDirectoryReceiptStore(receipts))
    model.train()
    controller.establish_baseline(evaluator)
    base_lr = lr
    rows, backoff, rejections, started = [], 0, 0, time.time()
    for interval in range(max_intervals):
        factor = 0.5 ** backoff

        def step_fn(_i: int, factor: float = factor) -> dict[str, Any]:
            step = cursor["step"]
            for g in opt.param_groups:
                g["lr"] = base_lr * factor * min(1.0, (step + 1) / 200)
            x_all = torch.tensor(C.windows(train, batch, seed * 1_000_003 + step), device=device)
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for x in x_all.chunk(accum):
                loss = nnf.cross_entropy(model(x[:, :-1]).reshape(-1, 256), x[:, 1:].reshape(-1)) / accum
                loss.backward()
                total = total + loss.detach()
            opt.step()
            cursor["step"] = step + 1
            last["bits"] = float(total) / math.log(2)
            if log is not None and step % 100 == 0:
                log({"train/bits": last["bits"]}, step)
            return {"train_bits": last["bits"]}

        receipt = controller.run_interval(interval_id=f"interval-{interval + 1}", steps=interval_steps, train_step=step_fn,
                                          evaluator=evaluator).to_dict()
        ev = receipt["candidate"]["evaluation"]
        row = {"interval": interval + 1, "promoted": bool(receipt["decision"]["promoted"]), "lr_factor": factor,
               "valid_bits": ev["validation"]["valid_bits"], "rollback_verified_exact": bool(receipt["rollback"]["verified_exact"])}
        rows.append(row)
        if log is not None:
            log({"valid/bits": row["valid_bits"], "valid/promoted": float(row["promoted"])}, cursor["step"])
        if row["promoted"]:
            rejections = 0
        else:
            rejections += 1
            backoff += 1
        if rejections >= 3:
            break
    model.eval()
    return {"intervals": rows, "accepted_steps": cursor["step"], "seconds": time.time() - started,
            "all_rollbacks_exact": all(r["rollback_verified_exact"] for r in rows),
            "parameters": {"encoder": sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
                           "readout": model.readout_parameters()}}


# --------------------------------------------------------------------------- #
# Like-for-like quantizer control: nested product quantization
# --------------------------------------------------------------------------- #


def nested_pq_words(hid: dict[str, np.ndarray], n: int, device: torch.device, seed: int = 0,
                    sample: int = 400_000) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """One subspace per axis; per subspace a 3-bit k-means and a 2-means split of each coarse cell.

    Word = code4 << 12 with code4 = 2 * code3 + child, so the top 3 bits are the coarse code and the
    top 4 bits the fine code: nested digits, exactly what ``poe_compile`` and ``cells`` read.
    """

    width = hid["train"].shape[1]
    if width % n:
        raise ValueError("hidden width must be divisible by the number of axes")
    d = width // n
    words = {s: np.zeros((h.shape[0], n), dtype=np.uint16) for s, h in hid.items()}
    info = {"subspace_dim": d, "coarse_k": 8, "fine_k": 16}
    started = time.time()
    for m in range(n):
        sl = slice(m * d, (m + 1) * d)
        c3 = C.kmeans(hid["train"][:, sl], 8, device, seed=seed * 1000 + m, sample=sample)
        a3 = {s: C.assign(h[:, sl], c3, device) for s, h in hid.items()}
        children = []
        for j in range(8):
            members = hid["train"][:, sl][a3["train"] == j]
            if members.shape[0] >= 2:
                children.append(C.kmeans(members, 2, device, seed=seed * 1000 + m * 8 + j + 1, sample=sample))
            else:  # empty coarse cell: both children equal the parent
                children.append(c3[j : j + 1].repeat(2, 1))
        for s, h in hid.items():
            child = np.zeros(h.shape[0], dtype=np.int64)
            for j in range(8):
                idx = np.where(a3[s] == j)[0]
                if idx.size:
                    child[idx] = C.assign(h[:, sl][idx], children[j], device)  # slice first: 8 dims, not 256
            code4 = 2 * a3[s].astype(np.int64) + child
            words[s][:, m] = (code4 << 12).astype(np.uint16)
    info["seconds"] = time.time() - started
    return words, info


# --------------------------------------------------------------------------- #
# Edits: recount tables, keep fitted weights
# --------------------------------------------------------------------------- #

CHAIN_ORDERS = (1, 2, 3, 4, 6)


def chain_tables(segs: np.ndarray) -> dict[str, Any]:
    y = segs[:, 1:].reshape(-1)
    ctx = {k: C.byte_context(segs, k) for k in CHAIN_ORDERS}
    o0 = (np.bincount(y, minlength=256) + 1.0) / (y.shape[0] + 256)
    return {"o0": o0, "tabs": {k: C.CountTable(ctx[k], y) for k in CHAIN_ORDERS}}


def chain_probs(tables: dict[str, Any], segs: np.ndarray) -> dict[str, np.ndarray]:
    y = segs[:, 1:].reshape(-1)
    p = tables["o0"][y]
    out: dict[str, np.ndarray] = {}
    for k in CHAIN_ORDERS:
        p, tot = tables["tabs"][k].prob(C.byte_context(segs, k), y, p)
        if k == 2:
            out["order2"] = p
        if k == 3:
            out["bucket"] = np.digitize(tot, [1, 10, 100])
    out["chain"] = p
    return out


def poe_tables(words: np.ndarray, y: np.ndarray, n: int, smooth: float = 8.0) -> dict[str, Any]:
    groups = C.poe_groups(n)
    uni = (np.bincount(y, minlength=256) + 1.0) / (y.shape[0] + 256)
    logp, counts, bits = [], [], 0
    for axes, k in groups:
        ncell = 1 << (k * len(axes))
        cnt = np.bincount(C.dense_cell(words, axes, k) * 256 + y, minlength=ncell * 256).reshape(ncell, 256).astype(np.float64)
        bits += 24 * int((cnt > 0).sum())
        counts.append(cnt.astype(np.int32))
        logp.append(np.log((cnt + smooth * uni[None]) / (cnt.sum(1, keepdims=True) + smooth)).astype(np.float32))
    return {"groups": groups, "logp": logp, "counts": counts, "bits": bits}


def poe_fit(tabs: dict[str, Any], words: np.ndarray, y: np.ndarray, device: torch.device, steps: int = 400,
            batch: int = 32768, seed: int = 0) -> dict[str, np.ndarray]:
    logp = [torch.tensor(t, device=device) for t in tabs["logp"]]
    idx_all = [torch.tensor(C.dense_cell(words, a, k), device=device) for a, k in tabs["groups"]]
    raw_w = torch.zeros(len(logp), device=device, requires_grad=True)
    bias = torch.zeros(256, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_w, bias], lr=0.05)
    yt = torch.tensor(y, device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, yt.shape[0], (batch,), generator=gen).to(device)
        w = nnf.softplus(raw_w - 2.0)
        z = bias[None].expand(i.shape[0], 256).clone()
        for j in range(len(logp)):
            z = z + w[j] * logp[j][idx_all[j][i]]
        loss = nnf.cross_entropy(z, yt[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"w": nnf.softplus(raw_w - 2.0).detach().cpu().numpy(), "bias": bias.detach().cpu().numpy()}


@torch.no_grad()
def poe_probs(tabs: dict[str, Any], weights: dict[str, np.ndarray], words: np.ndarray, y: np.ndarray,
              device: torch.device, batch: int = 65536) -> np.ndarray:
    logp = [torch.tensor(t, device=device) for t in tabs["logp"]]
    idx_all = [torch.tensor(C.dense_cell(words, a, k), device=device) for a, k in tabs["groups"]]
    w = torch.tensor(weights["w"], device=device)
    bias = torch.tensor(weights["bias"], device=device)
    yt = torch.tensor(y, device=device)
    out = []
    for s in range(0, yt.shape[0], batch):
        i = torch.arange(s, min(s + batch, yt.shape[0]), device=device)
        z = bias[None].expand(i.shape[0], 256).clone()
        for j in range(len(logp)):
            z = z + w[j] * logp[j][idx_all[j][i]]
        out.append(torch.softmax(z, -1).gather(-1, yt[i, None])[:, 0].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def fit_mixer(pv: np.ndarray, bv: np.ndarray, iters: int = 150) -> list[np.ndarray]:
    """Per-bucket arithmetic-mixture weights by EM (same procedure as ``chart_field.fit_mixture``)."""

    weights = []
    for b in range(4):
        mv = bv == b
        w = np.full(pv.shape[1], 1.0 / pv.shape[1])
        if mv.any():
            for _ in range(iters):
                r = pv[mv] * w
                r /= r.sum(1, keepdims=True)
                w = r.mean(0) + 1e-6
                w /= w.sum()
        weights.append(w)
    return weights


def mixer_bits(pt: np.ndarray, bt: np.ndarray, weights: list[np.ndarray]) -> np.ndarray:
    """Per-position bits of a fixed per-bucket mixture."""

    out = np.zeros(pt.shape[0])
    for b, w in enumerate(weights):
        m = bt == b
        if m.any():
            out[m] = -np.log2(np.maximum(pt[m] @ w, 1e-300))
    return out


def peak_rss_mb() -> float:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024.0 if sys.platform != "darwin" else peak / 2**20
