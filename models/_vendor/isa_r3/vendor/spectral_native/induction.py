"""Two-layer spectral induction head: organism, model, executors, baselines.

Organism.  A sequence is a run of episodes.  Each episode is a segment of
``L`` distinct tokens followed by the same segment again.  In the second copy
the next token is exactly what an induction head predicts: the token that
followed the most recent earlier occurrence of the current token.  Only those
positions are supervised; every other next token is random.

Model.  Every token carries a position chart ``p`` and a content chart ``x``.

- Layer 1 is one route over the charts ``(p, x)`` with a learned displacement
  ``delta1`` and one gate per chart.  It reads the selected token's content
  and writes it as a new chart ``y`` ("prev-content" when the route is the
  previous position).
- Layer 2 joins a query chart against a key chart.  The dictionary holds
  four pairs, each with a learned gate: content/content, content/prev-content,
  prev-content/prev-content and position/position with a learned
  displacement ``delta2``.  It emits the selected key's content.  A fixed
  integer recency rule breaks ties toward the most recent key.

The induction program is ``delta1 = 1`` with only the position gate on in
layer 1, and only the content/prev-content gate on in layer 2.

Training uses an explicit surrogate: layer 1 is a softmax over keys, so the
prev-content chart of key ``s`` is a distribution ``P1[s]`` and each layer-2
pair distance is its expectation.  Next-token logits pool layer-2 scores over
keys that carry each vocabulary symbol.  Evaluation uses hard routes three
ways: the float argmax, exact join semantics with dictionaries, and an
integer phase-ring scan with an exact zero-distance test.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .model import L_Q, circ, ste, wrap

VOCAB = 256
NULL = VOCAB  # content of a missing predecessor
RING = 1 << 16
POSITION_PERIODS = (16.0, 256.0, 4096.0, 65536.0)
PAIRS = ("content/content", "content/prev", "prev/prev", "position/position")
LAYER1_CHARTS = ("position", "content")


@dataclass(frozen=True)
class Sequence:
    tokens: torch.Tensor  # (T,) int64
    mask: torch.Tensor  # (T,) bool: next token is determined by induction
    target: torch.Tensor  # (T,) int64: x_{t+1} (valid where mask)


def make_sequence(length: int, seg_min: int, seg_max: int, generator: torch.Generator, vocab: int = VOCAB) -> Sequence:
    """Episodes of a distinct-token segment and its repeat, truncated to ``length``."""

    tokens: list[int] = []
    supervised: list[int] = []
    while len(tokens) < length:
        seg = int(torch.randint(seg_min, seg_max + 1, (1,), generator=generator))
        segment = torch.randperm(vocab, generator=generator)[:seg].tolist()
        start = len(tokens)
        tokens.extend(segment)
        tokens.extend(segment)
        # Second copy: position start+seg+i predicts segment[i+1].
        supervised.extend(start + seg + i for i in range(seg - 1))
    tokens = tokens[:length]
    x = torch.tensor(tokens, dtype=torch.int64)
    mask = torch.zeros(length, dtype=torch.bool)
    for position in supervised:
        if position + 1 < length:
            mask[position] = True
    target = torch.zeros(length, dtype=torch.int64)
    target[:-1] = x[1:]
    return Sequence(x, mask, target)


def make_split(count: int, length: int, seg_min: int, seg_max: int, seed: int) -> list[Sequence]:
    generator = torch.Generator().manual_seed(seed)
    return [make_sequence(length, seg_min, seg_max, generator) for _ in range(count)]


def _position_distance(query_pos: torch.Tensor, key_pos: torch.Tensor, omega: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """Summed circular distance between phase(p_q - delta) and phase(p_k)."""

    q = wrap(query_pos[:, None].to(omega.dtype) * omega[None] - ste(delta, 1.0) * omega[None])
    k = wrap(key_pos[:, None].to(omega.dtype) * omega[None])
    return circ(q[:, None, :] - k[None, :, :]).sum(-1)


class SpectralInduction(nn.Module):
    """Two hard layers of gated spectral joins with about eight parameters."""

    def __init__(
        self,
        variant: str = "joint",
        beta1: float = 32.0,
        beta2: float = 32.0,
        rho1: float = 2.0**-16,
        rho2: float = 2.0**-14,
        gate_mode: str = "relu",
        gate_init: float = 1.0,
    ) -> None:
        super().__init__()
        if variant not in {"joint", "onelayer", "frozenl1"}:
            raise ValueError(f"unknown variant {variant!r}")
        if gate_mode not in {"relu", "softplus"}:
            raise ValueError(f"unknown gate mode {gate_mode!r}")
        self.variant = variant
        self.gate_mode = gate_mode
        # Parameter value whose gate weight equals gate_init.
        raw = gate_init if gate_mode == "relu" else math.log(math.expm1(gate_init))
        self.beta1, self.beta2, self.rho1, self.rho2 = beta1, beta2, rho1, rho2
        self.register_buffer("omega", 1.0 / torch.tensor(POSITION_PERIODS, dtype=torch.float32))
        if variant == "frozenl1":
            # Layer 1 fixed to read the token itself: its prev-content chart is
            # just the content chart, so no induction program exists.
            self.register_buffer("delta1", torch.zeros(1))
            self.register_buffer("gate1", torch.tensor([1.0, 0.0]))
        else:
            self.delta1 = nn.Parameter(torch.randn(1) * 0.1)
            self.gate1 = nn.Parameter(torch.full((2,), raw))
        self.delta2 = nn.Parameter(torch.randn(1) * 0.1)
        self.gate2 = nn.Parameter(torch.full((4,), raw))
        pair_mask = torch.ones(4)
        if variant == "onelayer":
            pair_mask[1] = 0.0
            pair_mask[2] = 0.0
        self.register_buffer("pair_mask", pair_mask)

    # -- weights -----------------------------------------------------------
    def _gate(self, raw: torch.Tensor) -> torch.Tensor:
        if self.gate_mode == "softplus":
            # Softplus gates cannot die at zero, so credit keeps reaching
            # layer 1 while layer-2 pairs are still wrong.  Join programs are
            # recovered afterwards by compile_program.
            return nnf.softplus(raw)
        return nnf.relu(ste(raw, L_Q))

    def w1(self) -> torch.Tensor:
        if self.variant == "frozenl1":
            return self.gate1
        return self._gate(self.gate1)

    def w2(self) -> torch.Tensor:
        return self._gate(self.gate2) * self.pair_mask

    # -- layer 1 -----------------------------------------------------------
    def layer1_scores(self, x: torch.Tensor, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Scores (rows, T) of every key j <= k for query rows k."""

        length = x.shape[0]
        pos = torch.arange(length)
        rows = pos if rows is None else rows
        w = self.w1()
        dist = w[0] * _position_distance(rows, pos, self.omega, self.delta1)
        dist = dist + w[1] * (x[rows][:, None] != x[None, :]).to(dist.dtype)
        gap = (rows[:, None] - pos[None, :]).to(dist.dtype)
        scores = -dist - self.rho1 * gap
        return scores.masked_fill(pos[None, :] > rows[:, None], float("-inf"))

    @torch.no_grad()
    def hard_prev(self, x: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        length = x.shape[0]
        out = []
        for start in range(0, length, chunk):
            rows = torch.arange(start, min(length, start + chunk))
            out.append(self.layer1_scores(x, rows).argmax(-1))
        return x[torch.cat(out)]

    # -- layer 2 surrogate -------------------------------------------------
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Next-token logits (T, V) from the differentiable two-layer surrogate."""

        length = x.shape[0]
        pos = torch.arange(length)
        a1 = torch.softmax(self.beta1 * self.layer1_scores(x), -1)
        onehot = nnf.one_hot(x, VOCAB + 1).to(a1.dtype)
        p1 = a1 @ onehot  # (T, V+1): distribution of each token's layer-1 read
        w = self.w2()
        d_cc = (x[:, None] != x[None, :]).to(a1.dtype)
        d_cp = 1.0 - p1[:, x].T  # [t, s] = 1 - P1[s, x_t]
        d_pp = 1.0 - p1 @ p1.T
        d_pos = _position_distance(pos, pos, self.omega, self.delta2)
        dist = w[0] * d_cc + w[1] * d_cp + w[2] * d_pp + w[3] * d_pos
        gap = (pos[:, None] - pos[None, :]).to(dist.dtype)
        scores = -dist - self.rho2 * gap
        # Pool keys by the symbol they carry.  A symbol absent from the causal
        # context gets a finite floor instead of log(0): an all -inf
        # log-sum-exp has a NaN gradient.
        valid = (pos[None, :] <= pos[:, None])[:, :, None] & onehot[None, :, :VOCAB].bool()
        pooled = torch.where(valid, self.beta2 * scores[:, :, None], torch.full_like(valid, -1e4, dtype=scores.dtype))
        return torch.logsumexp(pooled, dim=1)

    # -- hard float route --------------------------------------------------
    @torch.no_grad()
    def hard_predict(self, x: torch.Tensor, chunk: int = 256) -> torch.Tensor:
        length = x.shape[0]
        pos = torch.arange(length)
        prev = self.hard_prev(x, chunk)
        w = self.w2()
        out = []
        for start in range(0, length, chunk):
            rows = torch.arange(start, min(length, start + chunk))
            dist = w[0] * (x[rows][:, None] != x[None, :]).float()
            dist = dist + w[1] * (x[rows][:, None] != prev[None, :]).float()
            dist = dist + w[2] * (prev[rows][:, None] != prev[None, :]).float()
            dist = dist + w[3] * _position_distance(rows, pos, self.omega, self.delta2)
            gap = (rows[:, None] - pos[None, :]).float()
            scores = (-dist - self.rho2 * gap).masked_fill(pos[None, :] > rows[:, None], float("-inf"))
            out.append(x[scores.argmax(-1)])
        return torch.cat(out)

    # -- program readout ---------------------------------------------------
    @torch.no_grad()
    def program(self) -> dict:
        return {
            "delta1": int(torch.round(self.delta1).item()),
            "gates1": {name: float(v) for name, v in zip(LAYER1_CHARTS, self.w1())},
            "delta2": int(torch.round(self.delta2).item()),
            "gates2": {name: float(v) for name, v in zip(PAIRS, self.w2())},
        }


@torch.no_grad()
def program_from_model(model: SpectralInduction, prune: float = 0.0, delta1_offset: int = 0) -> dict:
    """Snap a model to a join program: gates below prune*max become wildcards."""

    w1, w2 = model.w1(), model.w2()
    return {
        "delta1": int(torch.round(model.delta1).item()) + delta1_offset,
        "active1": [bool(v) for v in ((w1 > 0) & (w1 >= prune * w1.max()))],
        "delta2": int(torch.round(model.delta2).item()),
        "active2": [bool(v) for v in ((w2 > 0) & (w2 >= prune * w2.max()))],
    }


@torch.no_grad()
def hash_predict(model: SpectralInduction, x: torch.Tensor, prune: float = 0.0) -> torch.Tensor:
    return hash_execute(program_from_model(model, prune), x)


def hash_execute(program: dict, x: torch.Tensor) -> torch.Tensor:
    """Exact join semantics: dictionaries keyed on active charts, last write wins.

    A miss predicts -1 (always wrong).
    """

    active1 = torch.tensor(program["active1"])
    active2 = torch.tensor(program["active2"])
    d1, d2 = program["delta1"], program["delta2"]
    xs = x.tolist()
    length = len(xs)

    def key1(position: int, content: int, shift: int) -> tuple:
        return tuple(v for v, on in zip((position - shift, content), active1.tolist()) if on)

    table: dict[tuple, int] = {}
    prev = []
    for k in range(length):
        table[key1(k, xs[k], 0)] = k
        j = table.get(key1(k, xs[k], d1))
        prev.append(xs[j] if j is not None else NULL)

    def key2(s: int) -> tuple:
        return tuple(v for v, on in zip((xs[s], prev[s], prev[s], s), active2.tolist()) if on)

    def query2(t: int) -> tuple:
        return tuple(v for v, on in zip((xs[t], xs[t], prev[t], t - d2), active2.tolist()) if on)

    table2: dict[tuple, int] = {}
    out = []
    for t in range(length):
        table2[key2(t)] = t
        s = table2.get(query2(t))
        out.append(xs[s] if s is not None else -1)
    return torch.tensor(out)


def _content_codes(seed: int = 1729) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(RING, (VOCAB + 1, 2), generator=generator, dtype=torch.int64)
    if len({tuple(row) for row in codes.tolist()}) != VOCAB + 1:
        raise RuntimeError("content phase codes collide")
    return codes


CONTENT_CODES = _content_codes()
POSITION_CODES = torch.tensor([int(RING / period) for period in POSITION_PERIODS], dtype=torch.int64)


def _ring_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    diff = torch.remainder(a[:, None, :] - b[None, :, :], RING)
    return torch.minimum(diff, RING - diff).sum(-1)


@torch.no_grad()
def integer_predict(model: SpectralInduction, x: torch.Tensor, prune: float = 0.0, chunk: int = 256) -> torch.Tensor:
    return integer_execute(program_from_model(model, prune), x, chunk)


@torch.no_grad()
def integer_execute(program: dict, x: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Join semantics evaluated on the 2**16 phase ring: exact zero distance on
    every active chart pair, most recent key among exact matches."""

    active1, active2 = program["active1"], program["active2"]
    d1, d2 = program["delta1"], program["delta2"]
    length = x.shape[0]
    pos = torch.arange(length)
    pos_codes = torch.remainder(pos[:, None] * POSITION_CODES[None], RING)
    content = CONTENT_CODES[x]

    def select(match_rows: list[torch.Tensor], rows: torch.Tensor) -> torch.Tensor:
        exact = torch.ones(rows.shape[0], length, dtype=torch.bool)
        for m in match_rows:
            exact &= m
        exact &= pos[None, :] <= rows[:, None]
        best = torch.where(exact, pos[None, :], torch.full_like(exact, -1, dtype=torch.int64)).max(-1).values
        return best

    prev_idx = []
    for start in range(0, length, chunk):
        rows = pos[start : start + chunk]
        matches = []
        if active1[0]:
            q = torch.remainder(pos_codes[rows] - d1 * POSITION_CODES[None], RING)
            matches.append(_ring_distance(q, pos_codes) == 0)
        if active1[1]:
            matches.append(_ring_distance(content[rows], content) == 0)
        prev_idx.append(select(matches, rows))
    prev_idx = torch.cat(prev_idx)
    prev = torch.where(prev_idx >= 0, x[prev_idx.clamp_min(0)], torch.full_like(x, NULL))
    prev_codes = CONTENT_CODES[prev]
    out = []
    for start in range(0, length, chunk):
        rows = pos[start : start + chunk]
        matches = []
        if active2[0]:
            matches.append(_ring_distance(content[rows], content) == 0)
        if active2[1]:
            matches.append(_ring_distance(content[rows], prev_codes) == 0)
        if active2[2]:
            matches.append(_ring_distance(prev_codes[rows], prev_codes) == 0)
        if active2[3]:
            q = torch.remainder(pos_codes[rows] - d2 * POSITION_CODES[None], RING)
            matches.append(_ring_distance(q, pos_codes) == 0)
        best = select(matches, rows)
        out.append(torch.where(best >= 0, x[best.clamp_min(0)], torch.full_like(best, -1)))
    return torch.cat(out)


COMPILE_THRESHOLDS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9)


def supervised_accuracy(predict, sequences: list[Sequence]) -> float:
    correct = total = 0
    for sequence in sequences:
        prediction = predict(sequence.tokens)
        correct += int((prediction[sequence.mask] == sequence.target[sequence.mask]).sum())
        total += int(sequence.mask.sum())
    return correct / total


@torch.no_grad()
def compile_program(model: SpectralInduction, discovery: list[Sequence]) -> dict:
    """Choose a join program from training data only.

    Candidates: every prune threshold, and the rounded layer-1 displacement
    and its two integer neighbours (straight-through training can stop just
    across a rounding boundary).  Keep the candidate with the best discovery
    hash accuracy; break ties toward fewer active charts, then toward the
    rounded displacement.
    """

    best = None
    for offset in (0, -1, 1):
        for tau in COMPILE_THRESHOLDS:
            program = program_from_model(model, tau, offset)
            accuracy = supervised_accuracy(lambda x, program=program: hash_execute(program, x), discovery)
            size = sum(program["active1"]) + sum(program["active2"])
            key = (accuracy, -size, -abs(offset))
            if best is None or key > best[0]:
                best = (key, {**program, "tau": tau, "delta1_offset": offset, "discovery_hash_accuracy": accuracy})
    return best[1]


# -- attention-only Transformer baseline --------------------------------------


def _rotate(x: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    first, second = x[..., :half], x[..., half:]
    cos, sin = torch.cos(angle), torch.sin(angle)
    return torch.cat((first * cos - second * sin, first * sin + second * cos), -1)


class AttnOnlyBlock(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        batch, length, width = h.shape
        q, k, v = self.qkv(self.norm(h)).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        angle = torch.arange(length, device=h.device, dtype=torch.float32)[:, None] * self.inv_freq[None]
        q, k = _rotate(q, angle), _rotate(k, angle)
        attended = nnf.scaled_dot_product_attention(q, k, v, is_causal=True)
        return h + self.proj(attended.transpose(1, 2).reshape(batch, length, width))


class AttnOnlyTransformer(nn.Module):
    """Attention-only Transformer with 1D rotary positions (the standard setting
    in which induction heads are known to form)."""

    def __init__(self, d_model: int, layers: int, heads: int = 4) -> None:
        super().__init__()
        self.tok = nn.Embedding(VOCAB, d_model)
        self.blocks = nn.ModuleList(AttnOnlyBlock(d_model, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, VOCAB)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.tok(ids)
        for block in self.blocks:
            h = block(h)
        return self.out(self.norm(h))
