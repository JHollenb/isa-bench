"""How a chart learns in field semantics: gauge, phase churn, tying across dimensions, native optimizer.

Instruments and arms for the learned-chart field (``chart_field``).  Three hypotheses are tested:

1. **Gauge.**  The surrogate logits are ``B(theta) + T(cells(theta))``.  Any per-axis function both
   parts can express moves between them at zero loss, so ``B`` alone is not identified.  Along those
   flat directions the gradient is zero-mean noise, and Adam divides by ``sqrt(v) ~ |noise|``, which
   turns noise into full-size steps: a random walk along the gauge orbit (base collapse).  The SPF v3
   block already fixes the gauge by construction: the residual is stored *relative to* the base.
   ``gauge="residual"`` trains ``CE(B) + CE(stopgrad(B) + T)``.
2. **Phase churn.**  A Cartesian optimizer on the chart generator ``W`` (``u = W h``) takes
   sign-like steps whose induced phase motion ``|dtheta| ~ lr * sum_i |h_i|`` can cross cell edges every
   step.  ``optimizer="spf"``: AdaMax (running max, no second-moment sum), a phase trust region that
   bounds each axis's induced motion per step (in turns), phases wrapped after every step, and
   L-valued tables updated additively in L (a multiplicative update of the magnitude).
3. **Tying across dimensions** (the prototype's phase law ``Theta(r)_c = sum_d r_d Omega_{d,c}``):
   - ``tie="none"``: N independent phases.
   - ``tie="ladder"``: M = N / J Lie-algebra coordinates, each read at J frequencies
     ``2^{j-J+1}`` (unwrapped layout): coarse channels give unambiguous gradient, fine channels
     precision, and the cells of one coordinate are its nested (2-adic) digits.
   - ``tie="mixed"``: M = N / 2 coordinates, channels are mixed characters ``sum_m Omega_{m,c} u_m``
     with a fixed sparse integer ``Omega`` (exact on the ring; reaches non-product cosets).
   - ``base="match"``: the native base, ``b_y - sum_d a_{y,d} circ(theta_d - phi_{y,d})`` with
     ``a = 2^L``: one prototype phase per (symbol, axis), harmonics tied by the circular kernel
     (circular distance is a fixed positive Fourier series), against the free Fourier-linear base.
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

from . import chart_field as C
from .model import T_Q, circ, ste, wrap

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #


def ladder_omega(m: int, j: int) -> torch.Tensor:
    """(M, M*J): channel c = jj*M + mm reads coordinate mm at frequency 2^(jj-J+1)."""

    om = torch.zeros(m, m * j)
    for jj in range(j):
        for mm in range(m):
            om[mm, jj * m + mm] = 2.0 ** (jj - j + 1)
    return om


def mixed_omega(m: int, n: int, seed: int = 0) -> torch.Tensor:
    """(M, N) sparse integer characters: channel c = +-1 u_a +- {1,2} u_b (two coordinates each)."""

    g = torch.Generator().manual_seed(seed)
    om = torch.zeros(m, n)
    for c in range(n):
        a = c % m
        b = int(torch.randint(0, m - 1, (1,), generator=g))
        b = b + (b >= a)
        om[a, c] = 1.0
        om[b, c] = float([-2, -1, 1, 2][int(torch.randint(0, 4, (1,), generator=g))])
    return om


class TiedChart(nn.Module):
    """Encoder features -> Lie-algebra coordinates u in R^M -> channels theta = u Omega mod 1 (N channels)."""

    def __init__(self, encoder: nn.Module, n: int, tie: str, ladder: int = 4, seed: int = 0) -> None:
        super().__init__()
        self.encoder, self.n, self.tie = encoder, n, tie
        if tie == "none":
            m, om, span = n, torch.eye(n), 1.0
        elif tie == "ladder":
            m, span = n // ladder, 2.0 ** (ladder - 1)
            om = ladder_omega(m, ladder)
        elif tie == "mixed":
            m, span = n // 2, 1.0
            om = mixed_omega(m, n, seed)
        else:
            raise ValueError(tie)
        self.m = m
        self.register_buffer("omega", om)
        self.proj = nn.Linear(encoder.width, m)
        # coordinates start spread over the coarsest channel's full turn (no wrap at init)
        nn.init.normal_(self.proj.weight, std=0.5 * span / math.sqrt(encoder.width))
        nn.init.uniform_(self.proj.bias, 0.0, span)

    def features(self, ids: torch.Tensor) -> torch.Tensor:
        with C._amp(ids.device):
            h = self.encoder(ids)
        return h.float()

    def coords(self, h: torch.Tensor) -> torch.Tensor:
        with torch.autocast(h.device.type, enabled=False):
            return self.proj(h)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.from_coords(self.coords(self.features(ids)))

    def from_coords(self, u: torch.Tensor) -> torch.Tensor:
        return ste(wrap(u @ self.omega), T_Q)


# --------------------------------------------------------------------------- #
# Readout with explicit base / residual parts
# --------------------------------------------------------------------------- #


class NativeReadout(C.ChartReadout):
    def __init__(self, n: int, k_bits: int = 4, base: str = "fourier") -> None:
        super().__init__(n, k_bits)
        self.base_kind = base
        if base == "match":
            del self.base
            self.phi = nn.Parameter(torch.rand(256, n))  # prototype phase per (symbol, axis), turns
            self.la = nn.Parameter(torch.full((256, n), 1.0))  # log2 kernel weight per (symbol, axis)
            self.bias = nn.Parameter(torch.zeros(256))

    def base_logits(self, th: torch.Tensor) -> torch.Tensor:
        if self.base_kind == "fourier":
            return self.base(self.features(th))
        # b_y - sum_d 2^{L_{y,d}} circ(theta_d - phi_{y,d}); chunked over symbols to bound memory
        w = torch.exp2(self.la)
        out = []
        for v0 in range(0, 256, 64):
            d = circ(th[..., None, :] - ste(wrap(self.phi[v0 : v0 + 64]), T_Q))  # ..., 64, N
            out.append(-(d * w[v0 : v0 + 64]).sum(-1))
        return torch.cat(out, -1) + self.bias

    def table_logits(self, bins: torch.Tensor) -> torch.Tensor:
        a, b = bins[..., 0::2, :], bins[..., 1::2, :]
        joint = (a[..., :, None] * b[..., None, :]).flatten(-2)
        return torch.einsum("...hc,hcv->...v", joint, self.tables)

    def parts(self, th: torch.Tensor, beta: float | None):
        bins = self.hard_bins(th) if beta is None else self.soft_bins(th, beta)
        return self.base_logits(th), self.table_logits(bins), bins

    def forward(self, th: torch.Tensor, beta: float | None, use_base: bool = True, use_tables: bool = True):
        base, tab, bins = self.parts(th, beta)
        logits = (base if use_base else torch.zeros_like(base)) + (tab if use_tables else 0.0)
        return logits, bins

    def base_parameters(self) -> list[nn.Parameter]:
        return list(self.base.parameters()) if self.base_kind == "fourier" else [self.phi, self.la, self.bias]


# --------------------------------------------------------------------------- #
# SPF-native optimizer
# --------------------------------------------------------------------------- #


class SPFAdaMax(torch.optim.Optimizer):
    """AdaMax with SPF parameter kinds.

    ``kind`` per group:
    - ``"L"``: log-magnitude-valued (tables, log weights): additive step in L (multiplicative in magnitude).
    - ``"phase"``: turns; step, then wrap into [0, 1).
    - ``"generator"``: rows map features to chart coordinates.  Each row's step is scaled so its induced
      phase motion ``|dW_d| . E|h| + |db_d|`` stays below ``trust`` turns (``feature_abs`` is set per step).
    - ``"real"``: plain AdaMax (Cartesian parts such as a Fourier-linear base or an encoder).
    """

    def __init__(self, groups: list[dict[str, Any]], lr: float = 2e-3, betas=(0.9, 0.999), eps: float = 1e-8) -> None:
        # ``differentiable`` is declared up front: torch's ``__setstate__`` (run by ``load_state_dict``)
        # inserts it otherwise, and Saturn's bit-exact rollback then sees changed optimizer defaults.
        super().__init__(groups, dict(lr=lr, betas=betas, eps=eps, kind="real", trust=None, differentiable=False))
        self.feature_abs: torch.Tensor | None = None
        self.last_motion = 0.0

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102
        for group in self.param_groups:
            b1, b2 = group["betas"]
            steps: list[tuple[torch.Tensor, torch.Tensor]] = []
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["u"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                torch.maximum(st["u"].mul_(b2), p.grad.abs().add_(group["eps"]), out=st["u"])
                steps.append((p, group["lr"] / (1 - b1 ** st["t"]) * st["m"] / st["u"]))
            if group["kind"] == "generator" and group["trust"] and self.feature_abs is not None:
                w_step = [s for p, s in steps if p.dim() == 2]
                b_step = [s for p, s in steps if p.dim() == 1]
                motion = (w_step[0].abs() @ self.feature_abs) if w_step else 0.0
                if b_step:
                    motion = motion + b_step[0].abs()
                scale = torch.clamp(group["trust"] / (motion + 1e-12), max=1.0)
                self.last_motion = float((motion * scale).max())
                steps = [(p, s * (scale[:, None] if s.dim() == 2 else scale)) for p, s in steps]
            for p, s in steps:
                p.sub_(s)
                if group["kind"] == "phase":
                    p.remainder_(1.0)


# --------------------------------------------------------------------------- #
# Instruments
# --------------------------------------------------------------------------- #


def plugin_mi(cells: np.ndarray, y: np.ndarray, k: int) -> float:
    """Plug-in I(Y; cell) in bits minus the Miller-Madow bias correction."""

    joint = np.bincount(cells * 256 + y, minlength=k * 256).reshape(k, 256).astype(np.float64)
    n = joint.sum()
    pc, py = joint.sum(1, keepdims=True) / n, joint.sum(0, keepdims=True) / n
    pj = joint / n
    nz = pj > 0
    mi = float((pj[nz] * np.log2(pj[nz] / (pc @ py)[nz])).sum())
    bias = ((nz.sum() - (pc > 0).sum() - (py > 0).sum() + 1) / (2 * n * math.log(2)))
    return mi - float(bias)


@torch.no_grad()
def probe(chart: TiedChart, readout: NativeReadout, x: torch.Tensor, prev: dict[str, Any], rows: int = 64) -> dict[str, float]:
    """Churn, gauge, utilization and information on a fixed probe batch."""

    th = torch.cat([chart(x[i : i + rows, :-1]) for i in range(0, x.shape[0], rows)])
    y = x[:, 1:].reshape(-1)
    base, tab, _ = readout.parts(th, None)
    K = readout.K
    cells = torch.floor(th * K).long().clamp_(0, K - 1)
    out: dict[str, float] = {}
    if "theta" in prev:
        d = circ(th - prev["theta"])
        out["churn/mean_dtheta_turns"] = float(d.mean())
        out["churn/p99_dtheta_turns"] = float(d.flatten().quantile(0.99))
        out["churn/cell_change_frac"] = float((cells != prev["cells"]).float().mean())
    prev["theta"], prev["cells"] = th, cells
    bl = base - base.mean(-1, keepdim=True)
    tl = tab - tab.mean(-1, keepdim=True)
    out["gauge/base_logit_std"] = float(bl.std(-1).mean())
    out["gauge/table_logit_std"] = float(tl.std(-1).mean())
    out["gauge/base_table_corr"] = float(((bl * tl).sum(-1) / (bl.norm(dim=-1) * tl.norm(dim=-1) + 1e-9)).mean())
    out["bits/base"] = float(nnf.cross_entropy(base.reshape(-1, 256), y)) / math.log(2)
    out["bits/hard"] = float(nnf.cross_entropy((base + tab).reshape(-1, 256), y)) / math.log(2)
    cn = cells.reshape(-1, cells.shape[-1]).cpu().numpy()
    yn = y.cpu().numpy()
    mis = [plugin_mi(cn[:, a], yn, K) for a in range(cn.shape[1])]
    occ = [np.bincount(cn[:, a], minlength=K) / cn.shape[0] for a in range(cn.shape[1])]
    ent = [float(-(o[o > 0] * np.log2(o[o > 0])).sum()) for o in occ]
    out["axis/mean_mi_bits"] = float(np.mean(mis))
    out["axis/min_mi_bits"] = float(np.min(mis))
    out["axis/dead_axes"] = float(sum(e < 1.0 for e in ent))
    out["axis/mean_entropy_bits"] = float(np.mean(ent))
    # redundancy: mean |corr| of cos/sin embeddings across axes
    ang = 2 * math.pi * th.reshape(-1, th.shape[-1]).double()
    z = torch.cat([ang.cos(), ang.sin()], 1)
    cor = torch.corrcoef(z.T)[: th.shape[-1], : th.shape[-1]].abs()
    out["axis/mean_abs_corr"] = float((cor.sum() - cor.diagonal().sum()) / (cor.numel() - cor.shape[0]))
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class DynConfig:
    arm: str
    kind: str = "frozen-gpt"
    n_axes: int = 32
    tie: str = "none"  # none | ladder | mixed
    base: str = "fourier"  # fourier | match
    gauge: str = "joint"  # joint | residual
    optimizer: str = "adam"  # adam | spf
    trust: float = 1.0 / 64  # turns per step (a quarter of a 16-bin cell)
    lr: float = 2e-3
    lr_phase: float = 2e-3
    batch: int = 32
    interval_steps: int = 1000
    max_intervals: int = 4
    beta0: float = 0.5
    beta1: float = 16.0
    util_weight: float = 0.05
    probe_every: int = 100
    d: int = 64
    layers: int = 2
    heads: int = 4
    seed: int = 0
    accum: int = 1  # micro-batches per optimizer step (same batch, same steps; bounds activation memory)
    eval_rows: int = 64  # segments per forward in evaluation and probes


def build(cfg: DynConfig, gpt=None) -> tuple[TiedChart, NativeReadout]:
    if cfg.kind == "frozen-gpt":
        enc: nn.Module = C.FrozenGPT(gpt)
    elif cfg.kind == "gpt":
        enc = C.GPTEncoder(cfg.d, cfg.layers, cfg.heads)
    elif cfg.kind == "spf":
        # SPF-native trunk: the chart is an integer torus endomorphism of its phase channels
        from .chart_v2 import SPFPhaseChart

        if cfg.tie != "none":
            raise ValueError("the SPF trunk chart supports tie='none' only")
        return SPFPhaseChart(C.SPFEncoder(cfg.d, cfg.layers, cfg.heads), cfg.n_axes, seed=cfg.seed), NativeReadout(cfg.n_axes, 4, cfg.base)
    else:
        raise ValueError(cfg.kind)
    return TiedChart(enc, cfg.n_axes, cfg.tie, seed=cfg.seed), NativeReadout(cfg.n_axes, 4, cfg.base)


def make_optimizer(cfg: DynConfig, chart: TiedChart, readout: NativeReadout):
    enc = [p for p in chart.encoder.parameters() if p.requires_grad]
    base = readout.base_parameters()
    if cfg.optimizer == "adam":
        params = [p for p in list(chart.parameters()) + list(readout.parameters()) if p.requires_grad]
        return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01, betas=(0.9, 0.95))
    groups = [{"params": [chart.proj.weight, chart.proj.bias], "kind": "generator", "trust": cfg.trust, "lr": cfg.lr_phase},
              {"params": [readout.tables], "kind": "L"}]
    if cfg.base == "match":
        groups += [{"params": [readout.phi], "kind": "phase", "lr": cfg.lr_phase},
                   {"params": [readout.la, readout.bias], "kind": "L"}]
    else:
        groups += [{"params": base, "kind": "real"}]
    if enc:
        groups += [{"params": enc, "kind": "real"}]
    return SPFAdaMax(groups, lr=cfg.lr)


def train_dyn(cfg: DynConfig, train: np.ndarray, valid: np.ndarray, device: torch.device, receipts, gpt=None,
              log: Callable[[dict[str, Any], int], None] | None = None):
    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
    )

    torch.manual_seed(cfg.seed)
    chart, readout = build(cfg, gpt)
    chart.to(device)
    readout.to(device)
    model = nn.ModuleDict({"chart": chart, "readout": readout})
    opt = make_optimizer(cfg, chart, readout)
    params = [p for g in opt.param_groups for p in g["params"]]
    cursor = {"step": 0}
    total = cfg.interval_steps * cfg.max_intervals
    val_segs = C.segments(valid[: C.SEG * 256])
    probe_x = torch.tensor(C.segments(valid)[-64:], device=device)  # disjoint from the eval slice on the real split
    prev: dict[str, Any] = {}
    last = {"bits": 99.0}
    trace: list[dict[str, Any]] = []

    def beta_at(step: int) -> float:
        f = min(1.0, step / max(1, int(0.8 * total)))
        return cfg.beta0 * (cfg.beta1 / cfg.beta0) ** f

    def evaluator() -> EvaluationScores:
        hard = C.surrogate_bits(chart, readout, val_segs, device, rows=cfg.eval_rows)
        base = C.surrogate_bits(chart, readout, val_segs, device, rows=cfg.eval_rows, use_tables=False)
        return EvaluationScores(validation={"valid_bits_hard": hard}, autonomous_rollout={"valid_base_bits": base},
                                trace_endpoints={"train_bits": last["bits"]}, evidence={"step": cursor["step"]})

    # the base is part of the promoted object: a checkpoint that improves hard bits by wrecking the base is
    # not better (lexicographic, hard bits first with tolerance, then base bits)
    policy = PromotionPolicy(kind="lexicographic", objectives=(
        PromotionObjective("validation", "valid_bits_hard", "minimize", tolerance=1e-4),
        PromotionObjective("autonomous_rollout", "valid_base_bits", "minimize"),
        PromotionObjective("trace_endpoints", "train_bits", "minimize"),
    ))

    def restore(state: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(state)

    controller = DeterministicCheckpointController(
        controller_id=f"chart-dyn-{cfg.arm}-seed{cfg.seed}", model=model, optimizer=opt, policy=policy,
        cursor_capture=lambda: dict(cursor), cursor_restore=restore, max_interval_steps=cfg.interval_steps,
        receipt_store=ImmutableDirectoryReceiptStore(receipts))
    model.train()
    controller.establish_baseline(evaluator)
    rows, started = [], time.time()
    # Learning-rate backoff is outer-loop policy, kept outside the transacted cursor.  Without it a
    # rejected interval is replayed bit-identically (same cursor -> same batches -> same future).
    base_lrs = [g["lr"] for g in opt.param_groups]
    backoff, rejections = 0, 0
    for interval in range(cfg.max_intervals):
        factor = 0.5 ** backoff

        def step_fn(_i: int, factor: float = factor) -> dict[str, Any]:
            step = cursor["step"]
            warm = min(1.0, (step + 1) / 200)
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * warm * factor
            x_all = torch.tensor(C.windows(train, cfg.batch, cfg.seed * 1_000_003 + step), device=device)
            beta = beta_at(step)
            opt.zero_grad(set_to_none=True)
            feature_abs, ce_total, ce_base, util = 0.0, 0.0, 0.0, 0.0
            for x in x_all.chunk(cfg.accum):  # accum = 1: the whole batch, as before
                h = chart.features(x[:, :-1])
                feature_abs = feature_abs + h.detach().abs().reshape(-1, h.shape[-1]).mean(0) / cfg.accum
                th = chart.from_coords(chart.coords(h))
                base, tab, bins = readout.parts(th, beta)
                yb = x[:, 1:].reshape(-1)
                ce_t = nnf.cross_entropy((base + tab).reshape(-1, 256), yb)
                if cfg.gauge == "residual":
                    ce_b = nnf.cross_entropy(base.reshape(-1, 256), yb)
                    loss = ce_b + nnf.cross_entropy((base.detach() + tab).reshape(-1, 256), yb)
                else:
                    ce_b = ce_t.detach() * 0
                    loss = ce_t
                u = C.utilization_penalty(bins)
                ((loss + cfg.util_weight * u) / cfg.accum).backward()
                ce_total, ce_base, util = ce_total + ce_t.detach() / cfg.accum, ce_base + ce_b.detach() / cfg.accum, util + u.detach() / cfg.accum
            if isinstance(opt, SPFAdaMax):
                opt.feature_abs = feature_abs
            gn = {}
            if step % cfg.probe_every == 0:
                for name, ps in (("chart", [chart.proj.weight, chart.proj.bias]), ("base", readout.base_parameters()),
                                 ("tables", [readout.tables])):
                    gn[f"grad/{name}"] = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in ps if p.grad is not None)))
            if not isinstance(opt, SPFAdaMax):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            cursor["step"] = step + 1
            last["bits"] = float(ce_total.detach()) / math.log(2)
            if step % cfg.probe_every == 0:
                chart.eval()
                row = {"step": step, "train/bits_total": last["bits"], "train/bits_base": float(ce_base) / math.log(2),
                       "train/beta": beta, "train/util": float(util), **gn, **probe(chart, readout, probe_x, prev, cfg.eval_rows)}
                if isinstance(opt, SPFAdaMax):
                    row["churn/trust_motion_turns"] = opt.last_motion
                chart.train()
                trace.append(row)
                if log is not None:
                    log({k: v for k, v in row.items() if k != "step"}, step)
            return {"train_bits": last["bits"]}

        receipt = controller.run_interval(interval_id=f"interval-{interval + 1}", steps=cfg.interval_steps,
                                          train_step=step_fn, evaluator=evaluator).to_dict()
        ev = receipt["candidate"]["evaluation"]
        row = {"interval": interval + 1, "promoted": bool(receipt["decision"]["promoted"]),
               "valid_bits_hard": ev["validation"]["valid_bits_hard"],
               "valid_base_bits": ev["autonomous_rollout"]["valid_base_bits"],
               "rollback_verified_exact": bool(receipt["rollback"]["verified_exact"])}
        row["lr_factor"] = factor
        rows.append(row)
        if row["promoted"]:
            rejections = 0
        else:
            rejections += 1
            backoff += 1
        if log is not None:
            log({"valid/bits_hard": row["valid_bits_hard"], "valid/base_bits": row["valid_base_bits"],
                 "valid/promoted": float(row["promoted"]), "valid/lr_factor": factor}, cursor["step"])
        if rejections >= 3:
            break
    model.eval()
    info = {"config": asdict(cfg), "intervals": rows, "accepted_steps": cursor["step"], "seconds": time.time() - started,
            "all_rollbacks_exact": all(r["rollback_verified_exact"] for r in rows),
            "parameters": {"chart_head": sum(p.numel() for p in chart.proj.parameters()),
                           "base": sum(p.numel() for p in readout.base_parameters()),
                           "encoder": sum(p.numel() for p in chart.encoder.parameters() if p.requires_grad)}}
    return chart, readout, info, trace
