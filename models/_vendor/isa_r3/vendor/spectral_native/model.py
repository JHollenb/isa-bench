"""A spectral-native sequence model: every forward operation is exact on the SPF lattice.

State: each token carries C channels, each a log2 magnitude ``L`` and a phase ``Θ``
(turns in [0, 1)). Training runs in float with every parameter and intermediate
snapped to the SPF-32 grid (L: 8 fraction bits, Θ: 16-bit ring) through
straight-through estimators; ``integer_forward`` runs the trained model in pure
integers and must reproduce the float forward's predictions.

Components (each has a Cartesian ablation behind a flag):

* embedding    — per-token phase code (Θ) and log magnitude (L); no linear projection
* position     — Θ += p·ω, an SPF phase law (integer on the ring; relative position exact)
* attention    — MATCH: score = −Σ_c w_c · circular_distance(Θq, Θk) (integers), hard top-1
                 selection per head, value = the selected token's state bound with a head
                 mask (a copy, no sum).                     ablation: soft weighted sum
* residual     — BINDING: L += ΔL, Θ += ΔΘ (a product in value space).  ablation: value add
* mlp (L)      — TROPICAL rational: max_i(W1 + L) → max_j(W2 + h) − max_j(W3 + h).
                                                               ablation: linear + ReLU
* mlp (Θ)      — ternary phase map Θ' = A·Θ mod 1, A ∈ {−1,0,1} (products of powers)
* norm         — MAX-NORM: L −= max_c L
* readout      — MATCH against vocabulary phase codes: logits = −Σ_c α_c · d(Θ, Θ_v) + b_v

The only float in the integer forward is none: distances, weights, maxima and sums of
integers are integer arithmetic. (Summing integer scores is exact; what SPF cannot do
exactly is superpose complex values, which this model never does in its native form.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

L_Q = 2.0**8      # SPF-32 log2-magnitude grid
T_Q = 2.0**16     # SPF-32 phase ring


def ste(x: torch.Tensor, q: float) -> torch.Tensor:
    return x + (torch.round(x * q) / q - x).detach()


def wrap(x: torch.Tensor) -> torch.Tensor:
    return torch.remainder(x, 1.0)


def circ(x: torch.Tensor) -> torch.Tensor:
    """Circular distance in turns, in [0, 0.5]."""
    return torch.abs(torch.remainder(x + 0.5, 1.0) - 0.5)


def ternary(w: torch.Tensor) -> torch.Tensor:
    """{-1,0,1} with straight-through gradient (threshold 0.5·mean|w|)."""
    t = 0.5 * w.abs().mean().detach()
    hard = (w > t).to(w.dtype) - (w < -t).to(w.dtype)
    return w + (hard - w).detach()


def smoothmax(x: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
    """Forward: exact max. Backward: log-sum-exp with temperature tau."""
    hard = x.max(dim=dim).values
    soft = tau * torch.logsumexp(x / tau, dim=dim)
    return soft + (hard - soft).detach()


@dataclass
class Config:
    vocab: int = 256
    channels: int = 128
    heads: int = 4
    layers: int = 4
    hidden: int = 128
    max_len: int = 512
    attn: str = "select"      # select | soft
    mlp: str = "tropical"     # tropical | relu
    residual: str = "bind"    # bind | add
    tau: float = 0.1          # backward temperature for max / top-1


class MatchAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c, h = cfg.channels, cfg.heads
        assert c % h == 0
        self.cfg, self.h, self.ch = cfg, h, c // h
        self.q_mask = nn.Parameter(torch.rand(c))          # binding masks (turns)
        self.k_mask = nn.Parameter(torch.rand(c))
        self.v_mask = nn.Parameter(torch.rand(c))
        self.v_gain = nn.Parameter(torch.zeros(c))         # L added to the read value
        self.w = nn.Parameter(torch.ones(h, c // h))       # per-head channel weights on the distance
        self.temp = nn.Parameter(torch.tensor(0.0))       # softplus ≈ 0.69

    def scores(self, L: torch.Tensor, th: torch.Tensor) -> torch.Tensor:
        b, t, c = th.shape
        q = ste(wrap(th + ste(self.q_mask, T_Q)), T_Q).view(b, t, self.h, self.ch)
        k = ste(wrap(th + ste(self.k_mask, T_Q)), T_Q).view(b, t, self.h, self.ch)
        d = circ(q[:, :, None] - k[:, None, :])                       # b,t,s,h,ch
        # the query's own magnitudes choose which channels count: weight = max(0, Lq + w)
        w = F.relu(L.view(b, t, self.h, self.ch) + ste(self.w, L_Q)[None, None])     # b,t,h,ch
        return -(d * w[:, :, None]).sum(-1).permute(0, 3, 1, 2)                     # b,h,t,s (grid)

    def forward(self, L: torch.Tensor, th: torch.Tensor):
        b, t, c = L.shape
        causal = torch.ones(t, t, dtype=torch.bool, device=L.device).tril()
        raw = self.scores(L, th)
        sc = raw.masked_fill(~causal, float("-inf"))                                     # selection (integer scores)
        soft = torch.softmax((raw * F.softplus(self.temp) / self.cfg.tau).masked_fill(~causal, float("-inf")), dim=-1)
        vL = (L + ste(self.v_gain, L_Q)).view(b, t, self.h, self.ch)
        vT = ste(wrap(th + ste(self.v_mask, T_Q)), T_Q).view(b, t, self.h, self.ch)
        if self.cfg.attn == "soft":  # Cartesian ablation: superpose the values
            z = torch.exp2(vL) * torch.exp(2j * math.pi * vT)
            zo = torch.einsum("bhts,bshc->bthc", soft.to(z.dtype), z)
            oL = torch.log2(zo.abs().clamp_min(1e-30))
            oT = wrap(torch.angle(zo) / (2 * math.pi))
            return oL.reshape(b, t, c), oT.reshape(b, t, c)
        idx = sc.argmax(-1)                                                  # b,h,t  (ties -> lowest index)
        gi = idx.permute(0, 2, 1)[..., None].expand(b, t, self.h, self.ch)   # b,t,h,ch
        hL = torch.gather(vL, 1, gi)
        hT = torch.gather(vT, 1, gi)
        # value read is an exact copy of the selected token; gradient flows through the soft weights
        sL = torch.einsum("bhts,bshc->bthc", soft, vL)
        rel = torch.remainder(vT[:, None] - hT[:, :, None] + 0.5, 1.0) - 0.5    # b,t,s,h,ch, unwrapped at the pick
        sT = torch.einsum("bhts,btshc->bthc", soft, rel)
        oL = hL + (sL - sL.detach())
        oT = hT + (sT - sT.detach())
        return oL.reshape(b, t, c), wrap(oT.reshape(b, t, c))


class TropicalMLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c, d = cfg.channels, cfg.hidden
        self.cfg = cfg
        self.W1 = nn.Parameter(torch.randn(d, 2 * c) * 0.5)
        self.anchor = nn.Parameter(torch.rand(c))           # phase anchors (turns)
        self.W2 = nn.Parameter(torch.randn(c, d) * 0.5)
        self.W3 = nn.Parameter(torch.randn(c, d) * 0.5)
        self.lin1 = nn.Linear(2 * c, d)
        self.lin2 = nn.Linear(d, c)
        self.A = nn.Parameter(torch.randn(c, c) * 0.1)      # phase map, ternarized
        self.phase_bias = nn.Parameter(torch.rand(c))

    def forward(self, L: torch.Tensor, th: torch.Tensor):
        tau = self.cfg.tau
        # inputs: magnitudes and (negated, scaled) phase distances to learned anchors, all on the grid
        x = torch.cat([L, -8.0 * circ(th - ste(self.anchor, T_Q))], -1)
        if self.cfg.mlp == "tropical":
            h = smoothmax(ste(self.W1, L_Q)[None, None] + x[:, :, None, :], -1, tau)       # b,t,d
            dL = smoothmax(ste(self.W2, L_Q)[None, None] + h[:, :, None, :], -1, tau) \
                - smoothmax(ste(self.W3, L_Q)[None, None] + h[:, :, None, :], -1, tau)
        else:
            dL = self.lin2(F.relu(self.lin1(x)))
        A = ternary(self.A)
        dT = wrap(th @ A.T + ste(self.phase_bias, T_Q))
        return ste(dL, L_Q), ste(dT, T_Q)


class SpectralNative(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        c = cfg.channels
        self.emb_T = nn.Parameter(torch.rand(cfg.vocab, c))
        self.emb_L = nn.Parameter(torch.zeros(cfg.vocab, c))
        omega = torch.tensor([2.0 ** -(1 + 14 * i / c) for i in range(c)], dtype=torch.float64)
        self.register_buffer("omega", torch.round(omega * T_Q) / T_Q)       # fixed SPF phase law, on the ring
        self.attn = nn.ModuleList([MatchAttention(cfg) for _ in range(cfg.layers)])
        self.mlp = nn.ModuleList([TropicalMLP(cfg) for _ in range(cfg.layers)])
        self.out_T = nn.Parameter(torch.rand(cfg.vocab, c))
        self.out_w = nn.Parameter(torch.ones(c))          # gate offset: relu(L + out_w)
        self.out_b = nn.Parameter(torch.zeros(cfg.vocab))
        self.out_scale = nn.Parameter(torch.tensor(-2.0))   # softplus ≈ 0.13: logits start near uniform

    def combine(self, L, th, dL, dT):
        if self.cfg.residual == "bind":
            return L + dL, wrap(th + dT)
        z = torch.exp2(L) * torch.exp(2j * math.pi * th) + torch.exp2(dL) * torch.exp(2j * math.pi * dT)
        return torch.log2(z.abs().clamp_min(1e-30)), wrap(torch.angle(z) / (2 * math.pi))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device, dtype=torch.float64)
        th = wrap(ste(self.emb_T, T_Q)[ids] + (pos[:, None] * self.omega[None, :]).remainder(1.0).float()[None])
        L = ste(self.emb_L, L_Q)[ids]
        for attn, mlp in zip(self.attn, self.mlp):
            L = L - smoothmax(L, -1, self.cfg.tau)[..., None]            # max-norm
            dL, dT = attn(L, th)
            L, th = self.combine(L, th, dL, dT)
            L = L - smoothmax(L, -1, self.cfg.tau)[..., None]
            dL, dT = mlp(L, th)
            L, th = self.combine(L, th, dL, dT)
            L, th = ste(L, L_Q), ste(th, T_Q)
        d = circ(th[:, :, None, :] - ste(self.out_T, T_Q)[None, None])      # b,t,V,c
        w = F.relu(L + ste(self.out_w, L_Q))                                 # b,t,c: magnitude gates each channel
        logits = -(d * w[:, :, None, :]).sum(-1) + ste(self.out_b, L_Q)
        return logits * F.softplus(self.out_scale)


class TinyTransformer(nn.Module):
    """Cartesian baseline of comparable width."""

    def __init__(self, vocab: int, d: int, heads: int, layers: int, max_len: int):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d, vocab)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        t = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(t, device=ids.device))[None]
        mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=ids.device)
        return self.out(self.enc(x, mask=mask, is_causal=True))


def phase_params(model: nn.Module) -> list[nn.Parameter]:
    names = ("emb_T", "out_T", "q_mask", "k_mask", "v_mask", "phase_bias", "anchor")
    return [p for n, p in model.named_parameters() if n.split(".")[-1] in names]


@torch.no_grad()
def wrap_phase_params(model: nn.Module) -> None:
    for p in phase_params(model):
        p.remainder_(1.0)
