"""Multi-view 2-adic axes for real text, learned without a Transformer teacher (L4).

A *view* is a discrete function of the causal context (the previous word, the
last 4 bytes, the innermost open bracket, ...).  An *axis* is a 2-adic
hierarchy over a view's values: binary splits are address digits, so every
value gets a code and every code prefix is a ball.  The prediction model is
additive in log space over balls of every axis:

    log p(y | x) = b[y] + B[x_{t-1}, y] + sum_a sum_{d <= depth} theta_{a,d}[ball_{a,d}(x), y] - log Z(x)

Because each ``Q_a`` is a function of the context ``X``, the chain rule gives

    I(Y; X) = sum_a I(Y; Q_a | Q_<a) + I(Y; X | Q_1..N),

so each axis can only reduce the information the model misses by its
conditional mutual information with the target.  Axes are learned greedily:
each split of a node maximizes the saturated-correction gain relative to the
current model,

    G(S) = sum_y O_S[y] ln(O_S[y] / E_S[y]),      E_S[y] = sum_{x in S} p_model(y | x),

which is n_S times the KL divergence between observed and predicted next-byte
distributions in the node.  G of the leaf partition, measured on held-out
positions of the same distribution, is the plug-in estimate of the conditional
information the axis adds beyond the current model.  Splitting uses the
Poisson/KL Bregman form

    G(S) = max_lambda sum_{u in S} [<O_u, lambda> - <E_u, exp(lambda) - 1>],

which decomposes over values, so a Lloyd iteration (assign each value to the
side whose log-ratio vector scores it best, then update the log-ratios)
increases G monotonically.  A split is kept only if its gain exceeds its
degrees of freedom (an AIC gate: expected held-out gain > 0).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .fieldlm_text import ALNUM, U64, SubstringHasher, _mix, segment_starts

LN2 = math.log(2.0)
VIEWS = ("L2", "L4", "L8", "P", "W1", "W2", "S", "T")
_VIEW_TAG = {v: U64(1000 + i) for i, v in enumerate(VIEWS)}
OPENERS = frozenset(b"([{<")
CLOSERS = frozenset(b")]}>")
ALPHA = np.zeros(256, dtype=bool)
for _c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
    ALPHA[_c] = True


# ---------------------------------------------------------------------------
# Views (exact, causal, reset at segment starts)
# ---------------------------------------------------------------------------


@dataclass
class Tokens:
    tid: np.ndarray  # token id per byte, -1 if not a token byte
    starts: np.ndarray
    ends: np.ndarray
    hashes: np.ndarray  # uint64 per token


def tokens(x: np.ndarray, seg: np.ndarray, hasher: SubstringHasher) -> Tokens:
    n = x.shape[0]
    idx = np.arange(n, dtype=np.int64)
    alnum = ALNUM[x]
    prev_alnum = np.concatenate([[False], alnum[:-1]]) & (idx > seg)
    start = alnum & ~prev_alnum
    tid = np.where(alnum, np.cumsum(start) - 1, -1)
    starts = np.where(start)[0]
    cont_next = np.concatenate([alnum[1:] & ~start[1:], [False]])
    ends = np.where(alnum & ~cont_next)[0] + 1
    assert starts.shape == ends.shape
    hashes = hasher(starts, ends) if starts.size else np.zeros(0, dtype=U64)
    return Tokens(tid, starts, ends, hashes)


def topical_set(x: np.ndarray, seg_len: int = 256, lo: int = 5, hi: int = 5000, min_len: int = 4) -> np.ndarray:
    """Sorted token hashes that can serve as topic anchors: mid-frequency words (train only)."""

    x = x.astype(np.int64)
    seg = segment_starts(x.shape[0], seg_len)
    tk = tokens(x, seg, SubstringHasher(x))
    keep = (tk.ends - tk.starts >= min_len) & ALPHA[x[tk.starts]]
    uniq, counts = np.unique(tk.hashes[keep], return_counts=True)
    return uniq[(counts >= lo) & (counts <= hi)]


def _innermost_opener(x: np.ndarray, seg_len: int) -> np.ndarray:
    """Position of the innermost unclosed bracket before each byte (segment reset), -1 if none."""

    data = x.astype(np.uint8).tobytes()
    out = np.empty(len(data), dtype=np.int64)
    stack: list[int] = []
    for t, c in enumerate(data):
        if t % seg_len == 0:
            stack.clear()
        out[t] = stack[-1] if stack else -1
        if c in OPENERS:
            if len(stack) < 64:
                stack.append(t)
        elif c in CLOSERS and stack:
            stack.pop()
    return out


@dataclass
class ViewSet:
    positions: np.ndarray  # scored target positions (segment-local index >= 1)
    targets: np.ndarray  # int64
    prev: np.ndarray  # previous byte, int64
    keys: dict[str, np.ndarray]  # view -> uint64 key per scored position (never 0)


def compute_views(x: np.ndarray, topical: np.ndarray, seg_len: int = 256) -> ViewSet:
    x = x.astype(np.int64)
    n = x.shape[0]
    idx = np.arange(n, dtype=np.int64)
    seg = segment_starts(n, seg_len)
    local = idx - seg
    pos = idx[local >= 1]
    hasher = SubstringHasher(x)
    tk = tokens(x, seg, hasher)
    t = pos
    j = t - 1
    s = seg[t]
    inprog = ALNUM[x[j]]
    kcur = np.where(inprog, tk.tid[j], -1)
    ff = np.maximum.accumulate(tk.tid)
    last = ff[j]
    k1 = np.where(inprog, kcur - 1, last)
    nt = max(tk.starts.size, 1)
    starts_c = lambda k: tk.starts[np.clip(k, 0, nt - 1)] if tk.starts.size else np.zeros_like(k)  # noqa: E731
    valid1 = (k1 >= 0) & (starts_c(k1) >= s)
    k2 = k1 - 1
    valid2 = valid1 & (k2 >= 0) & (starts_c(k2) >= s)
    th = tk.hashes if tk.hashes.size else np.zeros(1, dtype=U64)
    tok_hash = lambda k: th[np.clip(k, 0, nt - 1)]  # noqa: E731

    keys: dict[str, np.ndarray] = {}
    for k, name in ((2, "L2"), (4, "L4"), (8, "L8")):
        a = np.maximum(s, t - k)
        keys[name] = _mix(hasher(a, t), np.full(t.shape, _VIEW_TAG[name], dtype=U64))
    prefix = hasher(np.where(inprog, starts_c(kcur), t), t)
    keys["P"] = np.where(inprog, _mix(prefix, np.full(t.shape, _VIEW_TAG["P"], dtype=U64)), _VIEW_TAG["P"])
    keys["W1"] = np.where(valid1, _mix(tok_hash(k1), np.full(t.shape, _VIEW_TAG["W1"], dtype=U64)), _VIEW_TAG["W1"])
    keys["W2"] = np.where(valid2, _mix(tok_hash(k2), np.full(t.shape, _VIEW_TAG["W2"], dtype=U64)), _VIEW_TAG["W2"])

    # S: innermost unclosed bracket, the word right before it, the word right after it
    opener = _innermost_opener(x, seg_len)[t]
    has_o = opener >= 0
    o = np.clip(opener, 0, n - 1)
    before = np.where(has_o & (o - 1 >= s) & ALNUM[x[np.clip(o - 1, 0, n - 1)]], tk.tid[np.clip(o - 1, 0, n - 1)], -1)
    after_id = np.where(has_o & (o + 1 < t) & ALNUM[x[np.clip(o + 1, 0, n - 1)]], tk.tid[np.clip(o + 1, 0, n - 1)], -1)
    after_done = (after_id >= 0) & (tk.ends[np.clip(after_id, 0, nt - 1)] <= t) if tk.ends.size else np.zeros_like(has_o)
    after_key = np.where(after_done, tok_hash(after_id), np.where(after_id >= 0, U64(3), U64(0)))
    before_key = np.where(before >= 0, tok_hash(before), U64(0))
    s_key = _mix(x[o].astype(U64), before_key, after_key, np.full(t.shape, _VIEW_TAG["S"], dtype=U64))
    keys["S"] = np.where(has_o, s_key, _VIEW_TAG["S"])

    # T: most recent topical (mid-frequency) word before W1, inside the segment
    if tk.hashes.size:
        is_top = np.zeros(tk.hashes.size, dtype=bool)
        if topical.size:
            loc = np.clip(np.searchsorted(topical, tk.hashes), 0, topical.size - 1)
            is_top = topical[loc] == tk.hashes
        last_top = np.maximum.accumulate(np.where(is_top, np.arange(tk.hashes.size), -1))
        cand = np.where(valid1 & (k1 >= 1), last_top[np.clip(k1 - 1, 0, nt - 1)], -1)
        ok = (cand >= 0) & (starts_c(cand) >= s)
        keys["T"] = np.where(ok, _mix(tok_hash(cand), np.full(t.shape, _VIEW_TAG["T"], dtype=U64)), _VIEW_TAG["T"])
    else:
        keys["T"] = np.full(t.shape, _VIEW_TAG["T"], dtype=U64)
    for name in keys:
        keys[name] = np.where(keys[name] == 0, U64(1), keys[name])
    return ViewSet(positions=pos, targets=x[pos], prev=x[j], keys=keys)


@dataclass
class Vocab:
    keys: np.ndarray  # sorted uint64
    counts: np.ndarray

    @staticmethod
    def build(keys: np.ndarray, min_count: int, max_size: int) -> "Vocab":
        uniq, counts = np.unique(keys, return_counts=True)
        keep = counts >= min_count
        uniq, counts = uniq[keep], counts[keep]
        if uniq.size > max_size:
            top = np.sort(np.argsort(-counts, kind="stable")[:max_size])
            uniq, counts = uniq[top], counts[top]
        return Vocab(uniq, counts)

    def ids(self, keys: np.ndarray) -> np.ndarray:
        if self.keys.size == 0:
            return np.full(keys.shape, -1, dtype=np.int64)
        loc = np.clip(np.searchsorted(self.keys, keys), 0, self.keys.size - 1)
        return np.where(self.keys[loc] == keys, loc, -1).astype(np.int64)


# ---------------------------------------------------------------------------
# Information accounting
# ---------------------------------------------------------------------------


def g_stat(O: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """sum_y O ln(O/E) along the last axis (0 ln 0 = 0), in nats."""

    O = O.double()
    E = E.double().clamp_min(1e-300)
    return torch.where(O > 0, O * torch.log(O.clamp_min(1e-300) / E), torch.zeros_like(O)).sum(-1)


def plugin_mi_bits(q: np.ndarray, y: np.ndarray, v: int = 256) -> float:
    """Plug-in I(Y; Q) in bits from paired samples (used by the tests)."""

    _, qi = np.unique(q, return_inverse=True)
    joint = np.zeros((qi.max() + 1, v))
    np.add.at(joint, (qi, y), 1.0)
    joint /= joint.sum()
    pq = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    nz = joint > 0
    return float((joint[nz] * np.log2(joint[nz] / (pq @ py)[nz])).sum())


# ---------------------------------------------------------------------------
# Axis learning: divisive Bregman (Poisson-KL) splitting with an AIC gate
# ---------------------------------------------------------------------------


@dataclass
class Axis:
    name: str
    vocab: Vocab
    value_node: np.ndarray  # node id per vocab value (0 = root: no evidence)
    parent: np.ndarray  # parent node id per node (-1 for root)
    depth: np.ndarray
    split_gain: np.ndarray  # training-fold G gain of the split that created each node's children (per parent)
    notes: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return int(self.parent.size)

    def leaves(self) -> np.ndarray:
        has_child = np.zeros(self.n_nodes, dtype=bool)
        has_child[self.parent[self.parent >= 0]] = True
        return np.where(~has_child)[0]

    def node_of_ids(self, ids: np.ndarray) -> np.ndarray:
        return np.where(ids >= 0, self.value_node[np.clip(ids, 0, max(self.value_node.size - 1, 0))], 0).astype(np.int32)

    def truncated(self, max_depth: int, min_count: int) -> "Axis":
        """Collapse nodes deeper than ``max_depth`` into their ancestor; drop rare values (to root)."""

        keep = self.depth <= max_depth
        remap = np.full(self.n_nodes, -1, dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        anc = np.arange(self.n_nodes)
        for _ in range(int(self.depth.max(initial=0)) + 1):
            deep = ~keep[anc]
            anc = np.where(deep, self.parent[anc], anc)
        new_node = remap[anc]
        value_node = new_node[self.value_node]
        value_node = np.where(self.vocab.counts >= min_count, value_node, 0)
        parent = np.where(self.parent[keep] >= 0, remap[np.clip(self.parent[keep], 0, None)], -1)
        return Axis(self.name, self.vocab, value_node, parent, self.depth[keep], self.split_gain[keep],
                    {**self.notes, "truncated": [max_depth, min_count], "kept_nodes": np.where(keep)[0]})

    def storage_bits(self, param_bits: int = 16, key_bits: int = 56) -> int:
        keys = int((self.value_node > 0).sum())
        return self.n_nodes * 256 * param_bits + keys * key_bits

    def ancestors(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = [], []
        for v in range(self.n_nodes):
            u = v
            while u >= 0:
                rows.append(v)
                cols.append(u)
                u = int(self.parent[u])
        return np.asarray(rows), np.asarray(cols)


def _split(Os: torch.Tensor, Es: torch.Tensor, alpha: float, lloyd_iters: int) -> torch.Tensor:
    """Poisson-KL Bregman 2-means over a node's values (rows); returns the side mask."""

    ratio = (Os.sum(0) + alpha) / (Es.sum(0) + alpha)
    Z = (Os - ratio * Es) / torch.sqrt(ratio * Es + 1.0)
    v = Z.abs().sum(0)
    v = v / v.norm().clamp_min(1e-12)
    for _ in range(8):  # power iteration for the top right singular vector of the standardized residual
        v = Z.T @ (Z @ v)
        v = v / v.norm().clamp_min(1e-12)
    side = (Z @ v) > 0
    if bool(side.all()) or not bool(side.any()):
        side = torch.arange(Os.shape[0], device=Os.device) % 2 == 0
    for _ in range(lloyd_iters):
        lam0 = torch.log((Os[~side].sum(0) + alpha) / (Es[~side].sum(0) + alpha))
        lam1 = torch.log((Os[side].sum(0) + alpha) / (Es[side].sum(0) + alpha))
        score0 = Os @ lam0 - Es @ (torch.exp(lam0) - 1.0)
        score1 = Os @ lam1 - Es @ (torch.exp(lam1) - 1.0)
        new = score1 > score0
        if torch.equal(new, side) or bool(new.all()) or not bool(new.any()):
            break
        side = new
    return side


def learn_axis(
    name: str,
    vocab: Vocab,
    O: torch.Tensor,
    E: torch.Tensor,
    max_depth: int = 12,
    min_node: float = 1000.0,
    lloyd_iters: int = 12,
    alpha: float = 0.5,
    aic: float = 1.0,
    cpu_below: int = 512,
) -> Axis:
    """Top-down 2-adic hierarchy over a view's values from (observed, expected) next-byte counts.

    ``O``, ``E``: [U, 256] on the fitting fold.  A split is accepted when its G gain exceeds
    ``aic`` x its degrees of freedom (distinct observed next bytes in the node minus one).
    Nodes with fewer than ``cpu_below`` values are processed on the CPU (kernel launches
    dominate small nodes on a GPU).
    """

    U = O.shape[0]
    n_all = O.sum(1)
    value_node = np.zeros(U, dtype=np.int64)
    parent = [-1]
    depth = [0]
    split_gain = [0.0]
    root = torch.where(n_all > 0)[0]

    def local(members: torch.Tensor, Os: torch.Tensor, Es: torch.Tensor):
        if members.numel() < cpu_below and Os.device.type != "cpu":
            return members.cpu(), Os.cpu(), Es.cpu()
        return members, Os, Es

    frontier = [(0, *local(root, O[root].double(), E[root].double()))]
    accepted = rejected = 0
    while frontier:
        node, members, Os, Es = frontier.pop()
        n_u = Os.sum(1)
        if depth[node] >= max_depth or members.numel() < 2 or float(n_u.sum()) < 2 * min_node:
            continue
        side = _split(Os, Es, alpha, lloyd_iters)
        n0, n1 = float(n_u[~side].sum()), float(n_u[side].sum())
        if min(n0, n1) < min_node:
            rejected += 1
            continue
        O0, O1, E0, E1 = Os[~side].sum(0), Os[side].sum(0), Es[~side].sum(0), Es[side].sum(0)
        gain = float(g_stat(O0, E0) + g_stat(O1, E1) - g_stat(O0 + O1, E0 + E1))
        df = float(((O0 + O1) > 0).sum()) - 1.0
        if gain <= aic * df:
            rejected += 1
            continue
        accepted += 1
        for c in (False, True):
            child = len(parent)
            parent.append(node)
            depth.append(depth[node] + 1)
            split_gain.append(gain / 2.0)
            m = side == c
            sub = members[m]
            value_node[sub.cpu().numpy()] = child
            frontier.append((child, *local(sub, Os[m], Es[m])))
    return Axis(name, vocab, value_node, np.asarray(parent, dtype=np.int64), np.asarray(depth, dtype=np.int64),
                np.asarray(split_gain), {"accepted_splits": accepted, "rejected_splits": rejected})


def leaf_accounting(axis: Axis, O_fold: torch.Tensor, E_fold: torch.Tensor) -> dict[str, float]:
    """G of the axis' deepest-node partition on held-out fold stats (per value -> per node)."""

    node = torch.as_tensor(axis.node_of_ids(np.arange(O_fold.shape[0])), device=O_fold.device)
    On = torch.zeros(axis.n_nodes, 256, dtype=torch.float64, device=O_fold.device).index_add_(0, node, O_fold.double())
    En = torch.zeros(axis.n_nodes, 256, dtype=torch.float64, device=O_fold.device).index_add_(0, node, E_fold.double())
    g = float(g_stat(On, En).sum())
    occupied = On.sum(1) > 0
    df = float(((On > 0).sum(1)[occupied] - 1).sum())
    return {"G_nats": g, "df": df, "n": float(On.sum()), "occupied_nodes": int(occupied.sum())}


# ---------------------------------------------------------------------------
# Additive log-linear model over axis balls
# ---------------------------------------------------------------------------


class AxisModel(nn.Module):
    def __init__(self, axes: list[Axis], device: torch.device) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(256, device=device))
        self.base = nn.Parameter(torch.zeros(256, 256, device=device))
        self.thetas = nn.ParameterList([nn.Parameter(torch.zeros(a.n_nodes, 256, device=device)) for a in axes])
        self._anc = []
        for a in axes:
            r, c = a.ancestors()
            idx = torch.as_tensor(np.stack([r, c]), device=device)
            self._anc.append(torch.sparse_coo_tensor(idx, torch.ones(r.size, device=device), (a.n_nodes, a.n_nodes),
                                                     check_invariants=False).coalesce())

    def tables(self) -> list[torch.Tensor]:
        return [torch.sparse.mm(A, th) for A, th in zip(self._anc, self.thetas)]

    def logits(self, prev: torch.Tensor, nodes: list[torch.Tensor], tables: list[torch.Tensor]) -> torch.Tensor:
        z = self.bias + self.base[prev]
        for T, nd in zip(tables, nodes):
            z = z + T[nd]
        return z

    def prior(self) -> torch.Tensor:
        total = (self.base ** 2).sum()
        for th in self.thetas:
            total = total + (th ** 2).sum()
        return total

    def storage_bits(self, axes: list[Axis], param_bits: int = 16) -> int:
        return (256 + 256 * 256) * param_bits + sum(a.storage_bits(param_bits) for a in axes)


@dataclass
class Batch:
    """Device tensors for a set of scored positions."""

    prev: torch.Tensor
    y: torch.Tensor
    nodes: list[torch.Tensor]

    def take(self, idx: torch.Tensor) -> "Batch":
        return Batch(self.prev[idx], self.y[idx], [n[idx] for n in self.nodes])

    @property
    def size(self) -> int:
        return int(self.y.numel())


def make_batch(vs: ViewSet, axes: list[Axis], ids: dict[str, np.ndarray], device: torch.device) -> Batch:
    nodes = [torch.as_tensor(a.node_of_ids(ids[a.name]), device=device) for a in axes]
    return Batch(torch.as_tensor(vs.prev.astype(np.int32), device=device),
                 torch.as_tensor(vs.targets, device=device), nodes)


def axis_nodes(axis: Axis, ids: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(axis.node_of_ids(ids), device=device)


def extend(batch: Batch, nodes: torch.Tensor) -> Batch:
    return Batch(batch.prev, batch.y, batch.nodes + [nodes])


# ---------------------------------------------------------------------------
# Closed-form initialisation (one Poisson / IRLS step)
# ---------------------------------------------------------------------------


def bigram_init(prev: np.ndarray, y: np.ndarray, alpha: float = 0.5) -> torch.Tensor:
    """log p(y | prev) with add-alpha smoothing: the base-only model's MAP optimum up to the prior."""

    C = np.bincount(prev.astype(np.int64) * 256 + y.astype(np.int64), minlength=65536).reshape(256, 256).astype(np.float64)
    P = (C + alpha) / (C.sum(1, keepdims=True) + 256 * alpha)
    return torch.as_tensor(np.log(P), dtype=torch.float32)


def axis_table_init(axis: Axis, O: torch.Tensor, E: torch.Tensor, alpha: float = 0.5) -> torch.Tensor:
    """theta for a new axis from per-value observed/expected counts under the current model.

    c[n] = log((O_sub(n) + alpha) / (E_sub(n) + alpha)) over the subtree of n; theta[n] = c[n] - c[parent]
    (theta[root] = c[root]), so a value at node n gets the path sum T[n] = c[n]: the saturated
    multiplicative correction of its node, i.e. one Newton step of Poisson regression.
    """

    device = O.device
    node = torch.as_tensor(axis.node_of_ids(np.arange(O.shape[0])).astype(np.int64), device=device)
    On = torch.zeros(axis.n_nodes, 256, dtype=torch.float64, device=device).index_add_(0, node, O.double())
    En = torch.zeros(axis.n_nodes, 256, dtype=torch.float64, device=device).index_add_(0, node, E.double())
    order = np.argsort(-axis.depth, kind="stable")  # children before parents
    parent = axis.parent
    for n in order:
        if parent[n] >= 0:
            On[parent[n]] += On[n]
            En[parent[n]] += En[n]
    c = torch.log((On + alpha) / (En + alpha))
    theta = c.clone()
    has_parent = torch.as_tensor(parent >= 0, device=device)
    par = torch.as_tensor(np.clip(parent, 0, None), device=device)
    theta[has_parent] = c[has_parent] - c[par[has_parent]]
    return theta.float()


def truncated_table_init(full: Axis, trunc: Axis, full_theta: torch.Tensor) -> torch.Tensor:
    """theta of a truncated axis = the full model's theta on the kept nodes (same path sums)."""

    kept = torch.as_tensor(trunc.notes["kept_nodes"], device=full_theta.device)
    return full_theta[kept].clone()


@torch.no_grad()
def position_bits(model: AxisModel, batch: Batch, chunk: int = 262_144) -> torch.Tensor:
    tables = model.tables()
    out = torch.empty(batch.size, dtype=torch.float64, device=batch.y.device)
    for s in range(0, batch.size, chunk):
        b = batch.take(torch.arange(s, min(s + chunk, batch.size), device=batch.y.device))
        z = model.logits(b.prev, b.nodes, tables).float()
        out[s : s + b.size] = (torch.logsumexp(z, -1) - z.gather(1, b.y[:, None])[:, 0]).double() / LN2
    return out


@torch.no_grad()
def residual_counts(model: AxisModel, batch: Batch, value_ids: torch.Tensor, U: int, chunk: int = 262_144) -> tuple[torch.Tensor, torch.Tensor]:
    """Observed and model-expected next-byte counts per view value over ``batch``."""

    device = batch.y.device
    O = torch.zeros(U, 256, dtype=torch.float32, device=device)
    E = torch.zeros(U, 256, dtype=torch.float32, device=device)
    tables = model.tables()
    for s in range(0, batch.size, chunk):
        sl = torch.arange(s, min(s + chunk, batch.size), device=device)
        b = batch.take(sl)
        vid = value_ids[sl]
        keep = vid >= 0
        if not bool(keep.any()):
            continue
        p = torch.softmax(model.logits(b.prev, b.nodes, tables).float(), -1)
        E.index_add_(0, vid[keep], p[keep])
        flat = torch.bincount(vid[keep] * 256 + b.y[keep], minlength=U * 256)
        O += flat.view(U, 256).float()
    return O, E


@torch.no_grad()
def expected_counts_multi(model: AxisModel, batch: Batch, value_ids: list[torch.Tensor], sizes: list[int],
                          chunk: int = 262_144) -> list[torch.Tensor]:
    """Model-expected next-byte counts per view value for several views in ONE model pass.

    Returns float32 [U_c, 256] tensors on ``batch``'s device.  Accumulation uses a sparse
    one-hot [U, n] @ p [n, 256] product (a row-wise reduction) instead of atomics.
    """

    device = batch.y.device
    E = [torch.zeros(U, 256, dtype=torch.float32, device=device) for U in sizes]
    tables = model.tables()
    for s in range(0, batch.size, chunk):
        sl = torch.arange(s, min(s + chunk, batch.size), device=device)
        b = batch.take(sl)
        p = torch.softmax(model.logits(b.prev, b.nodes, tables).float(), -1)
        for c, vid_all in enumerate(value_ids):
            vid = vid_all[sl]
            keep = torch.nonzero(vid >= 0).flatten()
            if keep.numel() == 0:
                continue
            idx = torch.stack([vid[keep], torch.arange(keep.numel(), device=device)])
            onehot = torch.sparse_coo_tensor(idx, torch.ones(keep.numel(), device=device),
                                             (sizes[c], keep.numel()), check_invariants=False)
            E[c] += torch.sparse.mm(onehot, p[keep])
    return E


def observed_counts(value_ids: np.ndarray, y: np.ndarray, U: int) -> np.ndarray:
    keep = value_ids >= 0
    return np.bincount(value_ids[keep] * 256 + y[keep], minlength=U * 256).reshape(U, 256).astype(np.float32)


def fit_model(
    axes: list[Axis],
    train: Batch,
    valid: Batch,
    device: torch.device,
    receipts: Path,
    controller_id: str,
    sigma: float = 2.0,
    lr: float = 0.05,
    batch_size: int = 1_000_000,
    interval_steps: int = 50,
    max_intervals: int = 8,
    seed: int = 0,
    init: AxisModel | None = None,
    base_init: torch.Tensor | None = None,
    table_init: dict[int, torch.Tensor] | None = None,
    only_new: bool = False,
) -> tuple[AxisModel, dict[str, Any]]:
    """Convex MAP fit (Gaussian prior sigma) by minibatch Adam under Saturn's checkpoint controller.

    - ``init`` warm-starts from a model over a prefix of ``axes``; ``base_init`` sets the bigram
      table; ``table_init`` sets axis tables by index (closed-form Poisson steps).  The objective is
      convex, so initialisation changes the speed of convergence, not the optimum.
    - ``only_new`` trains only the axis tables beyond ``init``'s prefix, with everything else frozen:
      the gain of the new axis given the previous model, which is what the accounting predicts.
      (A fresh Adam state moves every converged parameter by ``lr`` per step on noise, so a joint
      refit from a warm start must use a small ``lr``; Saturn rolls back any interval that hurts.)
    """

    from saturn.ar.training_checkpoint_controller import (
        DeterministicCheckpointController,
        EvaluationScores,
        ImmutableDirectoryReceiptStore,
        PromotionObjective,
        PromotionPolicy,
    )

    torch.manual_seed(seed)
    model = AxisModel(axes, device)
    n_old = 0
    with torch.no_grad():
        if init is not None:
            model.bias.copy_(init.bias)
            model.base.copy_(init.base)
            for dst, src in zip(model.thetas, init.thetas):
                if dst.shape == src.shape:
                    dst.copy_(src)
            n_old = len(init.thetas)
        if base_init is not None:
            model.base.copy_(base_init.to(device))
        for i, t in (table_init or {}).items():
            model.thetas[i].copy_(t.to(device))
    if only_new:
        trainable = [model.thetas[i] for i in range(n_old, len(axes))]
    else:
        trainable = list(model.parameters())
    optimizer = torch.optim.Adam(trainable, lr=lr)
    cursor = {"step": 0}
    lr_factors = (1.0, 0.5, 0.25, 0.125)
    backoff = 0
    last = {"bits": float("nan")}
    n_fit = train.size
    prior_scale = 1.0 / (2.0 * sigma * sigma * n_fit)

    def evaluator() -> EvaluationScores:
        bits = position_bits(model, valid)
        return EvaluationScores(
            validation={"valid_bits": float(bits.mean())},
            autonomous_rollout={"valid_frac_under_1bit": float((bits < 1.0).double().mean())},
            trace_endpoints={"train_bits": float(last["bits"]) if math.isfinite(last["bits"]) else 99.0},
            evidence={"step": cursor["step"]},
        )

    policy = PromotionPolicy(
        kind="lexicographic",
        objectives=(
            PromotionObjective("validation", "valid_bits", "minimize", tolerance=1e-5),
            PromotionObjective("autonomous_rollout", "valid_frac_under_1bit", "maximize"),
            PromotionObjective("trace_endpoints", "train_bits", "minimize"),
        ),
    )

    def restore(state: dict[str, int]) -> None:
        cursor.clear()
        cursor.update(state)

    receipts.mkdir(parents=True, exist_ok=True)
    controller = DeterministicCheckpointController(
        controller_id=controller_id,
        model=model,
        optimizer=optimizer,
        policy=policy,
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
            for group in optimizer.param_groups:
                group["lr"] = lr * factor
            g = torch.Generator(device="cpu").manual_seed(seed * 1_000_003 + step)
            take = min(batch_size, n_fit)
            idx = torch.randint(0, n_fit, (take,), generator=g).to(device)
            b = train.take(idx)
            tables = model.tables()
            z = model.logits(b.prev, b.nodes, tables)
            loss = nn.functional.cross_entropy(z, b.y) + prior_scale * model.prior()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            cursor["step"] = step + 1
            last["bits"] = float(loss.detach()) / LN2
            return {"train_bits": last["bits"]}

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
        "baseline_valid_bits": baseline["validation"]["valid_bits"],
        "intervals": rows,
        "accepted_steps": cursor["step"],
        "seconds": time.time() - started,
        "all_rollbacks_exact": all(r["rollback_verified_exact"] for r in rows),
        "n_fit": n_fit,
    }
    return model, info


def segment_bootstrap_se(diff: np.ndarray, seg_positions: int = 255, reps: int = 400, seed: int = 0) -> float:
    """Standard error of a mean per-position difference, resampling whole 256-byte segments."""

    n = diff.size - diff.size % seg_positions
    if n == 0:
        return float("nan")
    seg_means = diff[:n].reshape(-1, seg_positions).mean(1)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, seg_means.size, size=(reps, seg_means.size))
    return float(seg_means[draws].mean(1).std(ddof=1))
