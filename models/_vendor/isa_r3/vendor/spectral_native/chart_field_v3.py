"""Chart field v3: unwrapped charts, 2-adic backoff counts, count-aware geometric mixing.

Preregistered in ``docs/chart-field-v3-findings.md``.  Tests four diagnoses of the v2 failures
(``docs/chart-field-v2-findings.md``):

1. **Aliasing (Q3).**  A v2 chart axis is ``theta = (W h + b) mod 1``.  Hidden states whose projections
   differ by an integer get the same phase: the wrap identifies distinct contexts.  ``UnwrappedChart``
   makes the coordinate map injective: ``layout="sigmoid"`` (``v = sigmoid(u)``) or
   ``layout="longperiod"`` (``v = u / P + 1/2 mod 1`` with a period ``P`` far above the coordinate's
   span, the V4 unwrapping condition).  ``LineReadout`` reads ``v`` as a line: cells ``floor(K v)`` with
   non-circular soft assignment, and the base kernel on the half-turn phase ``v / 2`` (no neighbour
   across the ends).  ``coordinate_stats`` measures how much of each axis the wrap identifies.
2. **Estimation (F1).**  v2 product-of-experts tables are add-8 counts at one depth.  ``group_tables``
   with ``mode="backoff"`` builds Witten-Bell counts at every 2-adic prefix depth of each axis group and
   interpolates each depth with its parent (exact, count-based, recountable).
3. **Composition (F1, Q5).**  v2 mixes the byte chain and the tables with a fixed per-bucket arithmetic
   mixture.  ``fit_geometric`` fits a log-linear (geometric) mixture over full next-byte distributions
   whose weight set is selected by the chain's evidence state (highest seen order x deterministic),
   optionally followed by an SSE-style calibration.  ``ChainCSR`` stores the chain with explicit
   (context, byte) counts so full distributions exist; its target probabilities equal
   ``chart_field.CountTable``'s.
"""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from . import chart_dynamics as D
from . import chart_field as C
from .model import T_Q, ste, wrap

ORDERS = (1, 2, 3, 4, 6)
N_CHAIN_BUCKETS = 2 * (len(ORDERS) + 1)
LONG_PERIOD = 4.0
_ORIG_BUILD = D.build

# --------------------------------------------------------------------------- #
# Unwrapped charts
# --------------------------------------------------------------------------- #


class UnwrappedChart(D.TiedChart):
    """Encoder -> u in R^N -> v in [0, 1) with v monotone in u (no mod-1 identification)."""

    def __init__(self, encoder: nn.Module, n: int, layout: str, seed: int = 0) -> None:
        super().__init__(encoder, n, "none", seed=seed)
        if layout not in ("sigmoid", "longperiod"):
            raise ValueError(layout)
        self.layout = layout
        with torch.no_grad():  # centred: u spans about +-1.5 (sigmoid) or +-3/8 of a turn (period 4)
            nn.init.normal_(self.proj.weight, std=1.5 / math.sqrt(encoder.width))
            nn.init.zeros_(self.proj.bias)

    def turns(self, u: torch.Tensor) -> torch.Tensor:
        """Pre-wrap coordinate in turns; the chart value is ``turns mod 1`` (never wraps for sigmoid)."""

        return torch.sigmoid(u) if self.layout == "sigmoid" else u / LONG_PERIOD + 0.5

    def from_coords(self, u: torch.Tensor) -> torch.Tensor:
        return ste(wrap(self.turns(u @ self.omega)), T_Q)


class LineReadout(D.NativeReadout):
    """Readout for unwrapped charts: ``v`` is a line coordinate.

    Cells are ``floor(K v)`` (inherited ``hard_bins``) with a non-circular soft assignment.  The base
    reads the half-turn phase ``v / 2``, on which circular distance equals line distance.
    """

    def soft_bins(self, th: torch.Tensor, beta: float) -> torch.Tensor:
        d = (th[..., None] - self.centres) * self.K
        return torch.softmax(-beta * d * d, -1)

    def base_logits(self, th: torch.Tensor) -> torch.Tensor:
        return super().base_logits(th * 0.5)


def build_v3(cfg: D.DynConfig, gpt=None, layout: str = "sigmoid"):
    if layout == "wrapped":
        return _ORIG_BUILD(cfg, gpt)
    if cfg.kind == "frozen-gpt":
        enc: nn.Module = C.FrozenGPT(gpt)
    elif cfg.kind == "gpt":
        enc = C.GPTEncoder(cfg.d, cfg.layers, cfg.heads)
    else:
        raise ValueError(f"unwrapped layouts support frozen-gpt and gpt encoders, not {cfg.kind}")
    return UnwrappedChart(enc, cfg.n_axes, layout, seed=cfg.seed), LineReadout(cfg.n_axes, 4, cfg.base)


def save_chart_state(path, chart: nn.Module, readout: nn.Module, cfg: D.DynConfig, layout: str, **extra: Any) -> None:
    """Atomic write of a compile-ready chart state (the frozen Transformer's weights are not stored)."""

    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({"chart": {k: v for k, v in chart.state_dict().items() if not k.startswith("encoder.gpt.")},
                "readout": readout.state_dict(), "config": dict(cfg.__dict__), "layout": layout, **extra}, tmp)
    tmp.replace(path)


def train_v3(cfg: D.DynConfig, layout: str, train: np.ndarray, valid: np.ndarray, device: torch.device, receipts,
             gpt=None, log: Callable[[dict[str, Any], int], None] | None = None, checkpoint=None):
    """``chart_dynamics.train_dyn`` (Saturn intervals, exact rollback) with the v3 chart builder.

    ``checkpoint``: path rewritten after every interval with the controller's accepted state (a rejected interval
    is rolled back before the interval log fires), so a lease that times out still leaves a compilable chart.
    """

    built: dict[str, nn.Module] = {}

    def builder(c, g=None):
        chart, readout = build_v3(c, g, layout)
        built["chart"], built["readout"] = chart, readout
        return chart, readout

    def log_and_checkpoint(metrics: dict[str, Any], step: int) -> None:
        if log is not None:
            log(metrics, step)
        if checkpoint is not None and "valid/bits_hard" in metrics and built:
            save_chart_state(checkpoint, built["chart"], built["readout"], cfg, layout, accepted_step=int(step),
                             valid_bits_hard=float(metrics["valid/bits_hard"]))

    D.build = builder
    try:
        return D.train_dyn(cfg, train, valid, device, receipts, gpt=gpt, log=log_and_checkpoint)
    finally:
        D.build = _ORIG_BUILD


@torch.no_grad()
def coordinate_stats(chart: nn.Module, segs: np.ndarray, device: torch.device, rows: int = 64, k: int = 4) -> dict[str, Any]:
    """How much of each axis the wrap identifies, on the given segments.

    ``t`` = pre-wrap coordinate in turns (wrapped chart: ``u Omega``; unwrapped: ``turns(u Omega)``).
    ``outside_frac``: positions with ``t`` outside [0, 1).  ``alias_frac``: positions whose k-bit cell
    also holds a different winding ``floor(t)`` and that are not in the cell's majority winding (the
    share of contexts the cell merges with others that are a whole turn away).
    """

    chart.eval()
    ts = []
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows, :-1], device=device)
        u = chart.coords(chart.features(x)) @ chart.omega
        t = chart.turns(u) if isinstance(chart, UnwrappedChart) else u
        ts.append(t.reshape(-1, t.shape[-1]).double().cpu().numpy())
    t = np.concatenate(ts)
    K = 1 << k
    winding = np.floor(t).astype(np.int64)
    cell = np.clip(np.floor((t - np.floor(t)) * K).astype(np.int64), 0, K - 1)
    alias, spans = [], []
    for a in range(t.shape[1]):
        w = winding[:, a] - winding[:, a].min()
        joint = np.bincount(cell[:, a] * (w.max() + 1) + w, minlength=K * (w.max() + 1)).reshape(K, -1)
        alias.append(1.0 - joint.max(1).sum() / t.shape[0])
        spans.append(float(np.quantile(t[:, a], 0.99) - np.quantile(t[:, a], 0.01)))
    outside = ((t < 0) | (t >= 1)).mean(0)
    return {"positions": int(t.shape[0]), "outside_frac_mean": float(outside.mean()), "outside_frac_max": float(outside.max()),
            "alias_frac_mean": float(np.mean(alias)), "alias_frac_max": float(np.max(alias)),
            "span_turns_p1_p99_mean": float(np.mean(spans)), "span_turns_p1_p99_max": float(np.max(spans)),
            "axes_over_one_turn": int(sum(s > 1.0 for s in spans)), "alias_frac_per_axis": [float(x) for x in alias],
            "span_turns_per_axis": spans}


# --------------------------------------------------------------------------- #
# Byte chain with full distributions
# --------------------------------------------------------------------------- #


class ChainCSR:
    """Witten-Bell byte chain (orders 1, 2, 3, 4, 6) with explicit (context, byte) counts.

    Same estimator as ``chart_field.CountTable``: ``p_k = (c_k(y) + d_k p_{k-1}) / (n_k + d_k)`` for a seen
    context (``d_k`` distinct next bytes, ``n_k`` total), ``p_{k-1}`` otherwise, from an add-one unigram.
    """

    def __init__(self, segs: np.ndarray) -> None:
        y = segs[:, 1:].reshape(-1)
        self.o0 = (np.bincount(y, minlength=256) + 1.0) / (y.shape[0] + 256)
        self.tabs: dict[int, dict[str, np.ndarray]] = {}
        for k in ORDERS:
            keys = C.byte_context(segs, k)
            order = np.lexsort((y, keys))
            ks, ys = keys[order], y[order]
            new = np.ones(ks.shape[0], dtype=bool)
            new[1:] = (ks[1:] != ks[:-1]) | (ys[1:] != ys[:-1])
            idx = np.flatnonzero(new)
            pk, py = ks[idx], ys[idx].astype(np.uint8)
            pc = np.diff(np.append(idx, ks.shape[0])).astype(np.int64)
            newk = np.ones(pk.shape[0], dtype=bool)
            newk[1:] = pk[1:] != pk[:-1]
            start = np.flatnonzero(newk)
            end = np.append(start[1:], pk.shape[0])
            self.tabs[k] = {"uk": pk[start], "start": start, "end": end, "tot": np.add.reduceat(pc, start),
                            "dist": end - start, "y": py, "c": pc}
            del keys, order, ks, ys

    def entries(self) -> int:
        return sum(int(t["y"].shape[0]) for t in self.tabs.values())

    def keys(self) -> int:
        return sum(int(t["uk"].shape[0]) for t in self.tabs.values())

    def bits(self) -> int:
        """Track-D accounting: 24 bits per (context, byte) entry, 56 per context."""

        return 24 * self.entries() + 56 * self.keys()

    def full(self, segs: np.ndarray, chunk: int = 32768, dtype=np.float16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(log p over 256 bytes per position [dtype], evidence bucket per position, exact target p [float64])."""

        y = segs[:, 1:].reshape(-1)
        n = y.shape[0]
        ctx = {k: C.byte_context(segs, k) for k in ORDERS}
        out = np.empty((n, 256), dtype=dtype)
        bucket = np.zeros(n, dtype=np.int64)
        target = np.empty(n, dtype=np.float64)
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            p = np.broadcast_to(self.o0, (e - s, 256)).copy()
            hi = np.zeros(e - s, dtype=np.int64)
            det = np.zeros(e - s, dtype=bool)
            for oi, k in enumerate(ORDERS):
                t = self.tabs[k]
                kk = ctx[k][s:e]
                i = np.minimum(np.searchsorted(t["uk"], kk), t["uk"].shape[0] - 1)
                rows = np.flatnonzero(t["uk"][i] == kk)
                if rows.size == 0:
                    continue
                ii = i[rows]
                st, ln = t["start"][ii], t["dist"][ii]
                rep = np.repeat(np.arange(rows.size), ln)
                pos = np.repeat(st, ln) + (np.arange(int(ln.sum())) - np.repeat(np.cumsum(ln) - ln, ln))
                cnt = np.zeros((rows.size, 256))
                cnt[rep, t["y"][pos]] = t["c"][pos]
                tot = t["tot"][ii][:, None].astype(np.float64)
                d = ln[:, None].astype(np.float64)
                p[rows] = (cnt + d * p[rows]) / (tot + d)
                hi[rows] = oi + 1
                det[rows] = ln == 1
            out[s:e] = np.log(p).astype(dtype)
            target[s:e] = p[np.arange(e - s), y[s:e]]
            bucket[s:e] = 2 * hi + det
        return out, bucket, target


# --------------------------------------------------------------------------- #
# Count tables on chart cells: flat (v2) and 2-adic backoff
# --------------------------------------------------------------------------- #


def groups_v3(n_axes: int, pair_k: int = 4, quad_k: int = 3) -> list[tuple[tuple[int, ...], int]]:
    return ([((2 * p, 2 * p + 1), pair_k) for p in range(n_axes // 2)]
            + [(tuple(range(4 * q, 4 * q + 4)), quad_k) for q in range(n_axes // 4)])


def parent_cells(n_axes_in_group: int, k: int) -> np.ndarray:
    """Dense cell code at depth k (k bits per axis, ``chart_field.dense_cell`` order) -> code at depth k - 1."""

    codes = np.arange(1 << (k * n_axes_in_group), dtype=np.int64)
    parent = np.zeros_like(codes)
    for a in range(n_axes_in_group):
        shift = (n_axes_in_group - 1 - a) * k
        digit = (codes >> shift) & ((1 << k) - 1)
        parent |= (digit >> 1) << ((n_axes_in_group - 1 - a) * (k - 1))
    return parent


def group_tables(words: np.ndarray, y: np.ndarray, groups: list[tuple[tuple[int, ...], int]], mode: str,
                 smooth: float = 8.0) -> dict[str, Any]:
    """Per group: log p(y | cell) over all cells at the group's depth.

    ``flat``: add-``smooth`` counts toward the unigram at the group's depth (v2's ``poe_tables``).
    ``backoff``: Witten-Bell counts at every depth 1..k, each interpolated with its 2-adic parent cell
    (depth 0 = the add-one unigram).  Both are pure functions of counts (recountable).
    """

    w = words.reshape(-1, words.shape[-1])
    uni = (np.bincount(y, minlength=256) + 1.0) / (y.shape[0] + 256)
    logp, bits = [], 0
    for axes, k in groups:
        if mode == "flat":
            cnt = np.bincount(C.dense_cell(w, axes, k) * 256 + y, minlength=(1 << (k * len(axes))) * 256)
            cnt = cnt.reshape(-1, 256).astype(np.float64)
            bits += 24 * int((cnt > 0).sum())
            p = (cnt + smooth * uni[None]) / (cnt.sum(1, keepdims=True) + smooth)
        elif mode == "backoff":
            p = uni[None].copy()
            for j in range(1, k + 1):
                cnt = np.bincount(C.dense_cell(w, axes, j) * 256 + y, minlength=(1 << (j * len(axes))) * 256)
                cnt = cnt.reshape(-1, 256).astype(np.float64)
                bits += 24 * int((cnt > 0).sum())
                lower = p[parent_cells(len(axes), j)]
                tot = cnt.sum(1, keepdims=True)
                d = (cnt > 0).sum(1, keepdims=True).astype(np.float64)
                p = np.where(tot > 0, (cnt + d * lower) / np.maximum(tot + d, 1.0), lower)
                del cnt, lower
        else:
            raise ValueError(mode)
        logp.append(np.log(p).astype(np.float32))
    return {"groups": [(tuple(a), k) for a, k in groups], "logp": logp, "bits": bits, "mode": mode}


def cell_index(words: np.ndarray, groups: list[tuple[tuple[int, ...], int]]) -> list[np.ndarray]:
    w = words.reshape(-1, words.shape[-1])
    return [C.dense_cell(w, axes, k) for axes, k in groups]


# --------------------------------------------------------------------------- #
# Geometric (log-linear) mixing with evidence-selected weight sets
# --------------------------------------------------------------------------- #


class Experts:
    """Next-byte experts for one split: table experts (log-p tables + cell indices) and dense experts
    (per-position log-p over 256 bytes, float16 on CPU)."""

    def __init__(self, tables: dict[str, Any] | None, cells: list[np.ndarray] | None, dense: dict[str, np.ndarray],
                 device: torch.device, table_tensors: list[torch.Tensor] | None = None) -> None:
        self.dense = dense
        self.names = ([f"g{j}" for j in range(len(tables["logp"]))] if tables else []) + list(dense)
        self.tabs = table_tensors if table_tensors is not None else (
            [torch.tensor(t, device=device) for t in tables["logp"]] if tables else [])
        self.cells = [torch.tensor(c, device=device) for c in cells] if cells else []
        self.device = device

    def __len__(self) -> int:
        return len(self.tabs) + len(self.dense)

    def logits(self, idx: np.ndarray, w: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """``bias + sum_j w_j log p_j`` for positions ``idx``; ``w`` is (B, J), ``bias`` (B, 256)."""

        it = torch.tensor(idx, device=self.device)
        z = bias.clone()
        for j, tab in enumerate(self.tabs):
            z = z + w[:, j : j + 1] * tab[self.cells[j][it].long()].float()
        t0 = len(self.tabs)
        for k, name in enumerate(self.dense):
            z = z + w[:, t0 + k : t0 + k + 1] * torch.tensor(self.dense[name][idx], device=self.device).float()
        return z


def _conf_bin(z: torch.Tensor) -> torch.Tensor:
    top = torch.softmax(z, -1).max(-1).values
    return torch.bucketize(top, torch.tensor([0.3, 0.6, 0.9], device=z.device))


def fit_geometric(ev: Experts, y_valid: np.ndarray, b_valid: np.ndarray, n_buckets: int, device: torch.device,
                  steps: int = 600, batch: int = 16384, seed: int = 0, sse: bool = False, sse_steps: int = 300) -> dict[str, Any]:
    """Fit ``p(y) ∝ exp(bias_b[y] + sum_j w_{b,j} log p_j(y))`` on validation, ``b`` = evidence bucket.

    ``w >= 0`` (softplus).  With ``sse``, a second stage fits a temperature per (bucket, confidence bin)
    of the mixed logits, mixer frozen (calibration of the final distribution).
    """

    g = torch.Generator().manual_seed(seed)
    J = len(ev)
    raw = torch.full((n_buckets, J), math.log(math.expm1(1.0 / max(1, J) ** 0.5)), device=device, requires_grad=True)
    bias = torch.zeros(n_buckets, 256, device=device, requires_grad=True)
    opt = torch.optim.Adam([raw, bias], lr=0.05)
    yv = torch.tensor(y_valid, device=device)
    bv = torch.tensor(b_valid, device=device)
    n = y_valid.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (batch,), generator=g).numpy()
        b = bv[idx]
        z = ev.logits(idx, nnf.softplus(raw)[b], bias[b])
        loss = nnf.cross_entropy(z, yv[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    out: dict[str, Any] = {"weights": nnf.softplus(raw).detach().cpu().numpy(), "bias": bias.detach(),
                           "raw": raw.detach(), "names": ev.names, "n_buckets": n_buckets, "tau": None}
    if sse:
        tau_raw = torch.zeros(n_buckets, 4, device=device, requires_grad=True)
        opt2 = torch.optim.Adam([tau_raw], lr=0.05)
        w_all = nnf.softplus(out["raw"])
        for _ in range(sse_steps):
            idx = torch.randint(0, n, (batch,), generator=g).numpy()
            b = bv[idx]
            with torch.no_grad():
                z = ev.logits(idx, w_all[b], out["bias"][b])
                cb = _conf_bin(z)
            loss = nnf.cross_entropy(z * torch.exp(tau_raw[b, cb])[:, None], yv[idx])
            opt2.zero_grad()
            loss.backward()
            opt2.step()
        out["tau"] = torch.exp(tau_raw.detach())
    return out


@torch.no_grad()
def geometric_bits(ev: Experts, fit: dict[str, Any], y: np.ndarray, b: np.ndarray, device: torch.device,
                   chunk: int = 32768) -> np.ndarray:
    """Per-position bits of the fitted geometric mixture."""

    w_all = nnf.softplus(fit["raw"])
    bt = torch.tensor(b, device=device)
    yt = torch.tensor(y, device=device)
    out = np.empty(y.shape[0], dtype=np.float64)
    for s in range(0, y.shape[0], chunk):
        idx = np.arange(s, min(s + chunk, y.shape[0]))
        bb = bt[idx]
        z = ev.logits(idx, w_all[bb], fit["bias"][bb])
        if fit["tau"] is not None:
            z = z * fit["tau"][bb, _conf_bin(z)][:, None]
        lp = torch.log_softmax(z, -1).gather(-1, yt[idx, None])[:, 0]
        out[idx] = (-lp / math.log(2)).double().cpu().numpy()
    return out


@torch.no_grad()
def dense_logp(model_logits: Callable[[torch.Tensor], torch.Tensor], segs: np.ndarray, device: torch.device, rows: int = 64,
               dtype=np.float16) -> np.ndarray:
    """Full next-byte log-probabilities of a network (ids -> logits) per position, float16 on CPU."""

    t = segs.shape[1] - 1
    out = np.empty((segs.shape[0] * t, 256), dtype=dtype)
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows, :-1], device=device)
        lp = torch.log_softmax(model_logits(x).float(), -1).reshape(-1, 256)
        out[i * t : i * t + lp.shape[0]] = lp.cpu().numpy().astype(dtype)
    return out


@torch.no_grad()
def target_probs(model_logits: Callable[[torch.Tensor], torch.Tensor], segs: np.ndarray, device: torch.device, rows: int = 64) -> np.ndarray:
    """Exact (float32 softmax) probability of each position's true next byte."""

    out = []
    for i in range(0, segs.shape[0], rows):
        x = torch.tensor(segs[i : i + rows], device=device)
        out.append(torch.softmax(model_logits(x[:, :-1]).float(), -1).gather(-1, x[:, 1:, None])[..., 0].cpu().numpy())
    return np.concatenate(out).reshape(-1).astype(np.float64)


def peak_rss_mb() -> float:
    import resource
    import sys

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1024.0 if sys.platform != "darwin" else peak / 2**20


def timed(label: str, fn: Callable[[], Any], log: dict[str, float]) -> Any:
    t0 = time.time()
    out = fn()
    log[label] = time.time() - t0
    print(f"[v3] {label}: {log[label]:.0f} s, rss {peak_rss_mb():.0f} MB", flush=True)
    return out
