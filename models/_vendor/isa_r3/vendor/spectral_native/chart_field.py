"""Learned charts for the field language model: an N-dimensional torus chart, trained in field semantics.

Track D showed the exact field loses on real text because its charts are fixed byte quotients:
``loss >= H(Y | r(X))`` for any predictor built on a chart ``r``, whatever geometry sits on top.
This module learns ``r``.

Chart.  An encoder maps the causal context to Lie-algebra coordinates ``u in R^N``; the exponential
map to the torus is ``theta = u mod 1``, stored as SPF-32 phase words (16-bit ring, straight-through).
Every axis is one SPF phase channel.  The encoder is either

- ``frozen-gpt``: the track-D byte Transformer, frozen (chart head only is trained);
- ``gpt``: a small byte Transformer whose only output is the chart;
- ``spf``: the SPF-native trunk (match attention, tropical MLP, binding residual = Lie-algebra
  addition on (L, theta)); the chart is an integer torus endomorphism of its phase channels,
  ``theta_chart = A theta mod 1`` with ternary ``A`` (so it stays exact on the ring).

Readout (training surrogate) mirrors an SPF v3 block, ``BaseModel x ResidualCodec``:

- **base** (smooth generator): logits from low-frequency characters ``cos/sin(2 pi f theta_d)`` of
  every axis.  Nearby charts share statistics: this is the graded-similarity part.
- **residual** (exact partition): one table per axis pair over the joint cells of that pair at
  ``k`` bits per axis (a coset join on the integer chart ``floor(2^k theta)``).  During training the
  cell assignment is soft (circular Gaussian over bin centres, annealed toward hard); at compile the
  tables are *replaced by counts* on the training text, so knowledge lives in exact, recountable,
  editable tables while the chart only decides where things go.

``compile_field`` builds the exact model: Witten-Bell count tables keyed on joins of chart cells
(pairs, quads, all axes at coarse radius) and conjunctions with the last bytes, a byte n-gram chain,
and an arithmetic mixer fit on validation.  ``kmeans_cells`` is the standard-algorithm control.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .fieldlm_text_transformer import Block, ByteGPT, GPTConfig
from .model import Config as SPFConfig
from .model import SpectralNative, T_Q, circ, smoothmax, ste, ternary, wrap

SEG = 256
FREQS = (1.0, 2.0, 4.0, 8.0)
U64 = np.uint64


# --------------------------------------------------------------------------- #
# Encoders
# --------------------------------------------------------------------------- #


class FrozenGPT(nn.Module):
    """Track-D Transformer, frozen; returns the final normalized hidden state."""

    def __init__(self, gpt: ByteGPT) -> None:
        super().__init__()
        self.gpt = gpt
        for p in self.gpt.parameters():
            p.requires_grad_(False)
        self.width = gpt.cfg.d_model

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            x = self.gpt.embed(ids)
            for block in self.gpt.blocks:
                x = block(x)
            return self.gpt.ln(x).float()


class GPTEncoder(nn.Module):
    """Small byte Transformer with no language-model head: its only output is features for the chart."""

    def __init__(self, d: int, layers: int, heads: int) -> None:
        super().__init__()
        cfg = GPTConfig(d, layers, heads, SEG)
        self.embed = nn.Embedding(256, d)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.width = d
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for b in self.blocks:
            x = b(x)
        return self.ln(x)


class SPFEncoder(nn.Module):
    """SPF-native trunk (phase-first).  Returns (L, theta) channels; no Cartesian superposition."""

    def __init__(self, channels: int, layers: int, heads: int = 4) -> None:
        super().__init__()
        self.net = SpectralNative(SPFConfig(vocab=256, channels=channels, heads=heads, layers=layers,
                                            hidden=channels, max_len=SEG))
        self.width = channels

    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        net = self.net
        t = ids.shape[1]
        pos = torch.arange(t, device=ids.device, dtype=torch.float64)
        th = wrap(ste(net.emb_T, T_Q)[ids] + (pos[:, None] * net.omega[None, :]).remainder(1.0).float()[None])
        L = ste(net.emb_L, 2.0**8)[ids]
        for attn, mlp in zip(net.attn, net.mlp):
            L = L - smoothmax(L, -1, net.cfg.tau)[..., None]
            dL, dT = attn(L, th)
            L, th = net.combine(L, th, dL, dT)
            L = L - smoothmax(L, -1, net.cfg.tau)[..., None]
            dL, dT = mlp(L, th)
            L, th = net.combine(L, th, dL, dT)
            L, th = ste(L, 2.0**8), ste(th, T_Q)
        return L, th


class Chart(nn.Module):
    """Encoder -> N torus coordinates (SPF phase words)."""

    def __init__(self, encoder: nn.Module, n_axes: int, kind: str) -> None:
        super().__init__()
        self.encoder, self.n, self.kind = encoder, n_axes, kind
        if kind == "spf":
            # integer torus endomorphism of the trunk's phase channels (exact on the ring)
            self.A = nn.Parameter(torch.randn(n_axes, encoder.width) * 0.1)
        else:
            self.proj = nn.Linear(encoder.width, n_axes)
            nn.init.normal_(self.proj.weight, std=0.5 / math.sqrt(encoder.width))
            nn.init.uniform_(self.proj.bias, 0.0, 1.0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # Phases must stay float32: bf16 keeps 8 mantissa bits, far coarser than the 16-bit ring.
        if self.kind == "spf":
            with torch.autocast(ids.device.type, enabled=False):
                _, th = self.encoder(ids)
                return ste(wrap(th @ ternary(self.A).T), T_Q)
        with _amp(ids.device):
            h = self.encoder(ids)
        with torch.autocast(ids.device.type, enabled=False):
            return ste(wrap(self.proj(h.float())), T_Q)


# --------------------------------------------------------------------------- #
# Readout surrogate: base generator + pair-cell residual tables
# --------------------------------------------------------------------------- #


class ChartReadout(nn.Module):
    def __init__(self, n_axes: int, k_bits: int = 4) -> None:
        super().__init__()
        assert n_axes % 2 == 0
        self.n, self.k, self.K = n_axes, k_bits, 1 << k_bits
        self.base = nn.Linear(n_axes * 2 * len(FREQS), 256)
        self.register_buffer("freqs", torch.tensor(FREQS))
        self.register_buffer("centres", (torch.arange(self.K) + 0.5) / self.K)
        self.tables = nn.Parameter(torch.zeros(n_axes // 2, self.K * self.K, 256))

    def features(self, th: torch.Tensor) -> torch.Tensor:
        a = 2 * math.pi * th[..., None] * self.freqs
        return torch.cat([a.cos(), a.sin()], -1).flatten(-2)

    def soft_bins(self, th: torch.Tensor, beta: float) -> torch.Tensor:
        d = circ(th[..., None] - self.centres) * self.K  # distance in bin units
        return torch.softmax(-beta * d * d, -1)  # ..., N, K

    def hard_bins(self, th: torch.Tensor) -> torch.Tensor:
        return nnf.one_hot(torch.floor(th * self.K).long().clamp_(0, self.K - 1), self.K).float()

    def forward(self, th: torch.Tensor, beta: float | None, use_base: bool = True, use_tables: bool = True):
        logits = self.base(self.features(th)) if use_base else th.new_zeros(*th.shape[:-1], 256)
        bins = self.hard_bins(th) if beta is None else self.soft_bins(th, beta)
        if use_tables:
            a, b = bins[..., 0::2, :], bins[..., 1::2, :]  # ..., H, K
            joint = (a[..., :, None] * b[..., None, :]).flatten(-2)  # ..., H, K*K
            logits = logits + torch.einsum("...hc,hcv->...v", joint, self.tables)
        return logits, bins


def utilization_penalty(bins: torch.Tensor) -> torch.Tensor:
    """log K - H(batch-marginal bin occupancy), averaged over axes (0 = every bin used equally)."""

    marg = bins.reshape(-1, bins.shape[-2], bins.shape[-1]).mean(0)
    h = -(marg * marg.clamp_min(1e-9).log()).sum(-1)
    return (math.log(bins.shape[-1]) - h).mean()


@dataclass
class ChartConfig:
    kind: str  # frozen-gpt | gpt | spf
    n_axes: int = 32
    k_bits: int = 4
    d: int = 64
    layers: int = 2
    heads: int = 4
    lr: float = 2e-3
    batch: int = 32
    interval_steps: int = 1000
    max_intervals: int = 6
    beta0: float = 0.5
    beta1: float = 16.0
    util_weight: float = 0.05
    seed: int = 0


def build_chart(cfg: ChartConfig, gpt: ByteGPT | None = None) -> tuple[Chart, ChartReadout]:
    if cfg.kind == "frozen-gpt":
        enc: nn.Module = FrozenGPT(gpt)
    elif cfg.kind == "gpt":
        enc = GPTEncoder(cfg.d, cfg.layers, cfg.heads)
    elif cfg.kind == "spf":
        enc = SPFEncoder(cfg.d, cfg.layers, cfg.heads)
    else:
        raise ValueError(cfg.kind)
    return Chart(enc, cfg.n_axes, "spf" if cfg.kind == "spf" else "linear"), ChartReadout(cfg.n_axes, cfg.k_bits)


def trainable_parameters(chart: Chart, readout: ChartReadout) -> dict[str, int]:
    enc = sum(p.numel() for p in chart.encoder.parameters() if p.requires_grad)
    head = sum(p.numel() for n, p in chart.named_parameters() if not n.startswith("encoder."))
    return {"encoder": enc, "chart_head": head, "base": sum(p.numel() for p in readout.base.parameters()),
            "surrogate_tables": readout.tables.numel()}


# --------------------------------------------------------------------------- #
# Data helpers
# --------------------------------------------------------------------------- #


def segments(data: np.ndarray) -> np.ndarray:
    full = data.shape[0] // SEG
    return data[: full * SEG].reshape(full, SEG).astype(np.int64)


def windows(train: np.ndarray, batch: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, train.shape[0] - SEG - 1, size=batch)
    return np.stack([train[s : s + SEG] for s in starts]).astype(np.int64)


def _amp(device: torch.device):
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")


@torch.no_grad()
def surrogate_bits(chart: Chart, readout: ChartReadout, segs: np.ndarray, device: torch.device, rows: int = 64,
                   **kw: Any) -> float:
    chart.eval()
    tot, n = 0.0, 0
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows], device=device)
        th = chart(x[:, :-1])
        logits, _ = readout(th, None, **kw)
        tot += float(nnf.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1), reduction="sum"))
        n += x[:, 1:].numel()
    chart.train()
    return tot / n / math.log(2)


def train_chart(cfg: ChartConfig, train: np.ndarray, valid: np.ndarray, device: torch.device, receipts,
                gpt: ByteGPT | None = None, log: Callable[[dict[str, Any], int], None] | None = None):
    """Train chart + surrogate under Saturn's deterministic checkpoint controller."""

    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
    )

    torch.manual_seed(cfg.seed)
    chart, readout = build_chart(cfg, gpt)
    chart.to(device)
    readout.to(device)
    model = nn.ModuleDict({"chart": chart, "readout": readout})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))
    cursor = {"step": 0}
    total = cfg.interval_steps * cfg.max_intervals
    probe = segments(valid[: SEG * 256])
    last = {"bits": 99.0}

    def beta_at(step: int) -> float:
        f = min(1.0, step / max(1, int(0.8 * total)))
        return cfg.beta0 * (cfg.beta1 / cfg.beta0) ** f

    def evaluator() -> EvaluationScores:
        hard = surrogate_bits(chart, readout, probe, device)
        return EvaluationScores(
            validation={"valid_bits_hard": hard},
            autonomous_rollout={"neg_base_bits": -surrogate_bits(chart, readout, probe, device, use_tables=False)},
            trace_endpoints={"train_bits": last["bits"]},
            evidence={"step": cursor["step"]},
        )

    policy = PromotionPolicy(kind="lexicographic", objectives=(
        PromotionObjective("validation", "valid_bits_hard", "minimize", tolerance=1e-4),
        PromotionObjective("autonomous_rollout", "neg_base_bits", "minimize"),
        PromotionObjective("trace_endpoints", "train_bits", "minimize"),
    ))

    def restore(state: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(state)

    controller = DeterministicCheckpointController(
        controller_id=f"chart-field-{cfg.kind}-d{cfg.d}x{cfg.layers}-N{cfg.n_axes}-seed{cfg.seed}",
        model=model, optimizer=opt, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore,
        max_interval_steps=cfg.interval_steps, receipt_store=ImmutableDirectoryReceiptStore(receipts),
    )
    model.train()
    baseline = controller.establish_baseline(evaluator).to_dict()["candidate"]["evaluation"]
    rows, rejections, backoff, started = [], 0, 0, time.time()
    for interval in range(cfg.max_intervals):
        factor = (1.0, 0.5, 0.25, 0.125)[min(backoff, 3)]

        def step_fn(_i: int, factor: float = factor) -> dict[str, Any]:
            step = cursor["step"]
            for g in opt.param_groups:
                g["lr"] = cfg.lr * factor * min(1.0, (step + 1) / 200)
            x = torch.tensor(windows(train, cfg.batch, cfg.seed * 1_000_003 + step), device=device)
            beta = beta_at(step)
            th = chart(x[:, :-1])
            logits, bins = readout(th, beta)
            ce = nnf.cross_entropy(logits.reshape(-1, 256), x[:, 1:].reshape(-1))
            util = utilization_penalty(bins)
            loss = ce + cfg.util_weight * util
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            cursor["step"] = step + 1
            last["bits"] = float(ce.detach()) / math.log(2)
            if log is not None and step % 50 == 0:
                log({"train/bits_soft": last["bits"], "train/util_penalty": float(util.detach()), "train/beta": beta}, step)
            return {"train_bits": last["bits"]}

        receipt = controller.run_interval(interval_id=f"interval-{interval + 1}", steps=cfg.interval_steps,
                                          train_step=step_fn, evaluator=evaluator).to_dict()
        promoted = bool(receipt["decision"]["promoted"])
        ev = receipt["candidate"]["evaluation"]
        row = {"interval": interval + 1, "promoted": promoted, "lr_factor": factor,
               "valid_bits_hard": ev["validation"]["valid_bits_hard"],
               "valid_base_bits": -ev["autonomous_rollout"]["neg_base_bits"],
               "rollback_verified_exact": bool(receipt["rollback"]["verified_exact"])}
        rows.append(row)
        if log is not None:
            log({"valid/bits_hard": row["valid_bits_hard"], "valid/base_bits": row["valid_base_bits"],
                 "valid/promoted": float(promoted)}, cursor["step"])
        if promoted:
            rejections = 0
        else:
            rejections += 1
            backoff += 1
            if rejections >= 2:
                break
    model.eval()
    info = {"config": asdict(cfg), "parameters": trainable_parameters(chart, readout),
            "baseline_valid_bits_hard": baseline["validation"]["valid_bits_hard"], "intervals": rows,
            "accepted_steps": cursor["step"], "seconds": time.time() - started,
            "all_rollbacks_exact": all(r["rollback_verified_exact"] for r in rows)}
    return chart, readout, info


# --------------------------------------------------------------------------- #
# Compile: exact count tables on chart joins
# --------------------------------------------------------------------------- #


@torch.no_grad()
def chart_words(chart: Chart, segs: np.ndarray, device: torch.device, rows: int = 64) -> np.ndarray:
    """uint16 phase words, shape (segments, SEG-1, N): the chart for predicting byte p from bytes < p."""

    out = []
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows, :-1], device=device)
        th = chart(x)
        out.append(torch.floor(th * T_Q).long().clamp_(0, int(T_Q) - 1).to(torch.int32).cpu().numpy().astype(np.uint16))
    return np.concatenate(out)


@torch.no_grad()
def model_target_probs(chart: Chart, readout: ChartReadout, segs: np.ndarray, device: torch.device, rows: int = 64,
                       **kw: Any) -> np.ndarray:
    out = []
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows], device=device)
        th = chart(x[:, :-1])
        logits, _ = readout(th, None, **kw)
        out.append(torch.softmax(logits, -1).gather(-1, x[:, 1:, None])[..., 0].cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def mix(*parts: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = np.zeros(parts[0].shape, dtype=U64)
        for i, p in enumerate(parts):
            z = (z ^ p.astype(U64)) * U64(0x9E3779B97F4A7C15) + U64(i + 1)
            z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
            z = z ^ (z >> U64(31))
    return z


class CountTable:
    """Witten-Bell counts key -> next byte; 24 bits per entry, 56 per key (track-D accounting)."""

    def __init__(self, keys: np.ndarray, y: np.ndarray) -> None:
        kv = mix(keys, y.astype(U64))
        self.kv, self.kvc = np.unique(kv, return_counts=True)
        self.uk, inv, self.tot = np.unique(keys, return_inverse=True, return_counts=True)
        pair = np.unique(inv.astype(np.int64) * 256 + y.astype(np.int64))
        self.dist = np.bincount(pair // 256, minlength=len(self.uk))

    def bits(self) -> int:
        return 24 * len(self.kv) + 56 * len(self.uk)

    def prob(self, keys: np.ndarray, y: np.ndarray, lower: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        i = np.minimum(np.searchsorted(self.uk, keys), len(self.uk) - 1)
        hit = self.uk[i] == keys
        tot = np.where(hit, self.tot[i], 0)
        dist = np.where(hit, self.dist[i], 0)
        kv = mix(keys, y.astype(U64))
        j = np.minimum(np.searchsorted(self.kv, kv), len(self.kv) - 1)
        c = np.where(self.kv[j] == kv, self.kvc[j], 0)
        den = tot + dist
        return np.where(den > 0, (c + dist * lower) / np.maximum(den, 1), lower), tot


def byte_context(segs: np.ndarray, k: int) -> np.ndarray:
    """Hash of the last k bytes before each target (sentinel 300 before the segment start)."""

    s, t = segs.shape
    h = np.full((s, t - 1), k, dtype=U64)
    for j in range(1, k + 1):
        b = np.full((s, t - 1), 300, np.int64)
        b[:, j - 1 :] = segs[:, : t - j]
        h = mix(h, b)
    return h.reshape(-1)


def cells(words: np.ndarray, axes: tuple[int, ...], k: int) -> np.ndarray:
    """Exact join key on the integer chart floor(2^k theta) over the given axes."""

    w = words.reshape(-1, words.shape[-1])
    key = np.full(w.shape[0], hash((axes, k)) & 0xFFFFFFFF, dtype=U64)
    code, used = np.zeros(w.shape[0], dtype=U64), 0
    for a in axes:
        if used + k > 64:  # fold a full 64-bit chunk into the hash; never shift bits out
            key, code, used = mix(key, code), np.zeros(w.shape[0], dtype=U64), 0
        code = (code << U64(k)) | (w[:, a] >> (16 - k)).astype(U64)
        used += k
    return mix(key, code)


def chart_heads(n_axes: int) -> dict[str, tuple[tuple[int, ...], int, int]]:
    """name -> (axes, bits per axis, conjoined last bytes).  Radii nest: coarse joins over many axes,
    fine joins over few; conjunctions tie a learned cell to the local byte context."""

    heads: dict[str, tuple[tuple[int, ...], int, int]] = {}
    for p in range(n_axes // 2):
        heads[f"pair{p}"] = ((2 * p, 2 * p + 1), 4, 0)
        heads[f"pair{p}+b1"] = ((2 * p, 2 * p + 1), 3, 1)
    for q in range(n_axes // 4):
        heads[f"quad{q}"] = (tuple(range(4 * q, 4 * q + 4)), 3, 0)
        heads[f"quad{q}+b2"] = (tuple(range(4 * q, 4 * q + 4)), 2, 2)
    every = tuple(range(n_axes))
    heads["all1"] = (every, 1, 0)
    heads["all2"] = (every, 2, 0)
    heads["all1+b1"] = (every, 1, 1)
    heads["all1+b3"] = (every, 1, 3)
    return heads


def fit_mixture(pv: np.ndarray, bv: np.ndarray, pt: np.ndarray, bt: np.ndarray, iters: int = 150) -> tuple[float, list]:
    """Arithmetic mixture, one weight vector per evidence bucket (EM on valid), scored on test (bits/byte)."""

    bits, weights = 0.0, []
    for b in range(int(max(bv.max(), bt.max())) + 1):
        mv, mt = bv == b, bt == b
        w = np.full(pv.shape[1], 1.0 / pv.shape[1])
        if mv.any():
            for _ in range(iters):
                r = pv[mv] * w
                r /= r.sum(1, keepdims=True)
                w = r.mean(0) + 1e-6
                w /= w.sum()
        weights.append(w.tolist())
        if mt.any():
            bits += float(-np.log2(np.maximum(pt[mt] @ w, 1e-300)).sum())
    return bits / pt.shape[0], weights


def compile_field(train_segs: np.ndarray, valid_segs: np.ndarray, test_segs: np.ndarray,
                  words: dict[str, np.ndarray] | None, extra: dict[str, dict[str, np.ndarray]],
                  n_axes: int, cell_keys: dict[str, dict[str, np.ndarray]] | None = None,
                  combos: list[list[str]] | None = None) -> dict[str, Any]:
    """Exact field on learned charts.

    ``words``: split -> chart words (or None for chain-only / k-means arms).
    ``extra``: named per-position target probabilities to mix in (split -> array), e.g. the learned base.
    ``cell_keys``: precomputed head keys (split -> {name: keys}), used for the k-means control.
    """

    t0 = time.time()
    y = {s: segs[:, 1:].reshape(-1) for s, segs in (("train", train_segs), ("valid", valid_segs), ("test", test_segs))}
    segs = {"train": train_segs, "valid": valid_segs, "test": test_segs}
    ctx = {s: {k: byte_context(segs[s], k) for k in (1, 2, 3, 4, 6)} for s in segs}
    o0 = (np.bincount(y["train"], minlength=256) + 1.0) / (y["train"].shape[0] + 256)
    chain_tabs = {k: CountTable(ctx["train"][k], y["train"]) for k in (1, 2, 3, 4, 6)}
    storage = {"chain": sum(t.bits() for t in chain_tabs.values())}
    heads: dict[str, dict[str, np.ndarray]] = {s: {} for s in ("valid", "test")}
    order2, bucket = {}, {}
    for s in ("valid", "test"):
        p = o0[y[s]]
        for k in (1, 2, 3, 4, 6):
            p, tot = chain_tabs[k].prob(ctx[s][k], y[s], p)
            if k == 2:
                order2[s] = p
            if k == 3:
                bucket[s] = np.digitize(tot, [1, 10, 100])
        heads[s]["chain"] = p
    specs: dict[str, tuple] = {}
    if words is not None:
        specs = chart_heads(n_axes)
    names = list(specs) + (list(cell_keys["train"]) if cell_keys else [])
    for name in names:
        def keys(split: str, name: str = name) -> np.ndarray:
            if cell_keys and name in cell_keys[split]:
                return cell_keys[split][name]
            axes, k, nb = specs[name]
            key = cells(words[split], axes, k)
            return mix(key, ctx[split][nb]) if nb else key
        tab = CountTable(keys("train"), y["train"])
        storage[name] = tab.bits()
        for s in ("valid", "test"):
            heads[s][name] = tab.prob(keys(s), y[s], order2[s])[0]
    for name, probs in extra.items():
        for s in ("valid", "test"):
            heads[s][name] = probs[s]

    def arm(members: list[str]) -> dict[str, Any]:
        members = [m for m in members if m in heads["test"]]
        pv = np.stack([heads["valid"][m] for m in members], 1)
        pt = np.stack([heads["test"][m] for m in members], 1)
        bits, w = fit_mixture(pv, bucket["valid"], pt, bucket["test"])
        return {"bits_per_byte": bits, "members": members, "storage_bits": sum(storage.get(m, 0) for m in members)}

    table_heads = [n for n in names]
    arms = {"chain": arm(["chain"])}
    if table_heads:
        arms["chain+cells"] = arm(["chain"] + table_heads)
    for name in extra:
        arms[f"chain+{name}"] = arm(["chain", name])
        if table_heads:
            arms[f"chain+cells+{name}"] = arm(["chain"] + table_heads + [name])
    if len(extra) > 1:
        arms["chain+cells+" + "+".join(extra)] = arm(["chain"] + table_heads + list(extra))
    for combo in combos or []:  # explicit member sets (v2: e.g. chain+poe+transformer)
        arms["+".join(combo)] = arm(list(combo))
    # per-head single contribution over chain (which joins carry information)
    single = {}
    for name in table_heads:
        single[name] = arms["chain"]["bits_per_byte"] - arm(["chain", name])["bits_per_byte"]
    return {"arms": arms, "storage_bits": storage, "single_head_gain": single,
            "positions": {s: int(y[s].shape[0]) for s in y}, "seconds": time.time() - t0}


def axis_utilization(words: np.ndarray, k: int = 4) -> dict[str, Any]:
    """Per-axis occupancy entropy (bits, max k) and mean absolute correlation between axes."""

    w = words.reshape(-1, words.shape[-1])
    sub = w[:: max(1, w.shape[0] // 200_000)]
    ent = []
    for a in range(w.shape[1]):
        c = np.bincount(sub[:, a] >> (16 - k), minlength=1 << k) / sub.shape[0]
        ent.append(float(-(c[c > 0] * np.log2(c[c > 0])).sum()))
    ang = 2 * np.pi * sub.astype(np.float64) / 65536.0
    z = np.concatenate([np.cos(ang), np.sin(ang)], 1)
    cor = np.corrcoef(z.T)
    n = w.shape[1]
    off = np.abs(cor[:n, :n][~np.eye(n, dtype=bool)])
    return {"entropy_bits": ent, "mean_entropy_bits": float(np.mean(ent)), "max_bits": k,
            "mean_abs_cos_corr": float(off.mean())}


# --------------------------------------------------------------------------- #
# Standard-algorithm control: k-means on the frozen hidden state
# --------------------------------------------------------------------------- #


@torch.no_grad()
def hidden_states(enc: FrozenGPT, segs: np.ndarray, device: torch.device, rows: int = 64) -> np.ndarray:
    # preallocated: a list + np.concatenate held the train split twice (v2-pq-s0 job-57a198664e32 was
    # killed at 11.9 GB RSS 29 s in; the states themselves are 6.1 GB for all splits)
    t = segs.shape[1] - 1
    out = np.empty((segs.shape[0] * t, enc.width), dtype=np.float16)
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows, :-1], device=device)
        with _amp(device):
            h = enc(x).half().cpu().numpy()
        out[i * t : i * t + h.shape[0] * t] = h.reshape(-1, enc.width)
    return out


@torch.no_grad()
def kmeans(x: np.ndarray, k: int, device: torch.device, iters: int = 25, sample: int = 400_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs = torch.tensor(x[rng.choice(x.shape[0], min(sample, x.shape[0]), replace=False)], device=device).float()
    c = xs[torch.randperm(xs.shape[0], generator=torch.Generator().manual_seed(seed))[:k].to(device)].clone()
    for _ in range(iters):
        a = torch.cat([_nearest(xs[i : i + 32768], c) for i in range(0, xs.shape[0], 32768)])
        s = torch.zeros_like(c).index_add_(0, a, xs)
        n = torch.bincount(a, minlength=k).float()
        c = torch.where(n[:, None] > 0, s / n.clamp_min(1)[:, None], c)
    return c


def _nearest(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """argmin_j |x - c_j|^2 via one matmul (chunked by the caller: rows x K floats)."""

    return ((c * c).sum(1)[None] - 2.0 * x @ c.T).argmin(1)


@torch.no_grad()
def assign(x: np.ndarray, c: torch.Tensor, device: torch.device, rows: int = 32768) -> np.ndarray:
    out = []
    for i in range(0, x.shape[0], rows):
        out.append(_nearest(torch.tensor(x[i : i + rows], device=device).float(), c).cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Post hoc: product-of-experts compile (log-domain composition across axis groups)
# --------------------------------------------------------------------------- #


def dense_cell(words: np.ndarray, axes: tuple[int, ...], k: int) -> np.ndarray:
    w = words.reshape(-1, words.shape[-1])
    code = np.zeros(w.shape[0], dtype=np.int64)
    for a in axes:
        code = (code << k) | (w[:, a].astype(np.int64) >> (16 - k))
    return code


def poe_groups(n_axes: int) -> list[tuple[tuple[int, ...], int]]:
    """Dense groups: every axis pair at 4 bits (256 cells) and every quad at 3 bits (4,096 cells)."""

    return ([((2 * p, 2 * p + 1), 4) for p in range(n_axes // 2)]
            + [(tuple(range(4 * q, 4 * q + 4)), 3) for q in range(n_axes // 4)])


def poe_compile(words: dict[str, np.ndarray], y: dict[str, np.ndarray], n_axes: int, device: torch.device,
                smooth: float = 8.0, steps: int = 400, batch: int = 32768, seed: int = 0) -> dict[str, Any]:
    """Exact count tables per axis group, composed as a product of experts.

    ``log p(y | chart) = sum_j w_j log phat_j(y | cell_j) + b_y - log Z`` with ``phat_j`` add-``smooth``
    count estimates toward the unigram (recountable, editable).  Only the exponents ``w_j >= 0`` and
    biases ``b_y`` (``H + 256`` numbers) are fit, on the validation split.  In SPF terms the product is a
    sum of log-magnitudes, the cheap operation; only the final normalization needs the add path.
    """

    groups = poe_groups(n_axes)
    uni = (np.bincount(y["train"], minlength=256) + 1.0) / (y["train"].shape[0] + 256)
    logp, cellidx, bits = [], {s: [] for s in ("valid", "test")}, 0
    for axes, k in groups:
        ncell = 1 << (k * len(axes))
        c_tr = dense_cell(words["train"], axes, k)
        cnt = np.bincount(c_tr * 256 + y["train"], minlength=ncell * 256).reshape(ncell, 256).astype(np.float64)
        bits += 24 * int((cnt > 0).sum())
        p = (cnt + smooth * uni[None]) / (cnt.sum(1, keepdims=True) + smooth)
        logp.append(torch.tensor(np.log(p), dtype=torch.float32, device=device))
        for s in cellidx:
            cellidx[s].append(torch.tensor(dense_cell(words[s], axes, k), device=device))
    H = len(groups)
    raw_w = torch.zeros(H, device=device, requires_grad=True)
    bias = torch.zeros(256, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw_w, bias], lr=0.05)
    yv = torch.tensor(y["valid"], device=device)
    yt = torch.tensor(y["test"], device=device)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def logits(split: str, idx: torch.Tensor) -> torch.Tensor:
        w = nnf.softplus(raw_w - 2.0)
        out = bias[None].expand(idx.shape[0], 256).clone()
        for j in range(H):
            out = out + w[j] * logp[j][cellidx[split][j][idx]]
        return out

    for _ in range(steps):
        idx = torch.randint(0, yv.shape[0], (batch,), generator=gen).to(device)
        loss = nnf.cross_entropy(logits("valid", idx), yv[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out = {}
    with torch.no_grad():
        for s, ys in (("valid", yv), ("test", yt)):
            probs = []
            for i in range(0, ys.shape[0], batch):
                idx = torch.arange(i, min(i + batch, ys.shape[0]), device=device)
                probs.append(torch.softmax(logits(s, idx), -1).gather(-1, ys[idx, None])[:, 0].cpu().numpy())
            out[s] = np.concatenate(probs).astype(np.float64)
    w = nnf.softplus(raw_w - 2.0).detach().cpu().numpy()
    return {"probs": out, "exponents": w.tolist(), "groups": [list(a) + [k] for a, k in groups],
            "table_bits": bits, "fit_params": H + 256}
