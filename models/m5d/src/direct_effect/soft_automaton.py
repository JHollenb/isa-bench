"""Lane A gradient comparison: the same per-bit automaton as `spf_datapath`, trained by gradient in a float surrogate.

Three arms, all with the scan direction supplied (LSB→MSB, which favours them; induction had to find it):
  phase  phase-native parameterization: soft transition maps (softmax over next state), an emission angle φ per
         (symbol, state) read through the fixed Cartesian projection bit = (1 − cos φ) / 2, soft start state.
  ste    the same parameters, hard forward (argmax maps, φ snapped to {0, π}), straight-through backward.
  cart   CAS-style: a complex state vector carried by soft monomials M = e^{iφ}·softmax(P), read by a learned linear
         readout of (re, im) per symbol, as in saturn/experiments/2026-10-09-cas-learned-comparison/payload/arms.py.
Every arm is exported to an integer `spf_datapath.Table` and run on SPF `monomial_scan` for the exactness check.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from . import spf_datapath as D


class Soft(torch.nn.Module):
    def __init__(self, arm, n=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.arm, self.n = arm, n
        self.trans = torch.nn.Parameter(0.5 * torch.randn(4, n, n, generator=g))  # logits over next state
        self.phi = torch.nn.Parameter(0.5 * torch.randn(4, n, generator=g))  # emission angle (phase / ste)
        self.start = torch.nn.Parameter(0.5 * torch.randn(2, n, generator=g))
        if arm == "cart":
            self.read = torch.nn.Parameter(0.5 * torch.randn(4, 2 * n, generator=g))
            self.read_b = torch.nn.Parameter(torch.zeros(4))

    def forward(self, sym, cf):
        """sym (B, w) long in [0, 4), cf (B,) long → bit logits or probabilities (B, w)."""
        B, w = sym.shape
        hard = self.arm == "ste"
        P = self.trans.softmax(-1)
        if hard:
            P = F.one_hot(P.argmax(-1), self.n).float() + P - P.detach()
        s0 = self.start.softmax(-1)[cf]
        if hard:
            s0 = F.one_hot(s0.argmax(-1), self.n).float() + s0 - s0.detach()
        if self.arm == "cart":
            xr, xi = s0, torch.zeros_like(s0)
            outs = []
            for i in range(w):
                s = sym[:, i]
                rd = self.read[s]  # (B, 2n)
                outs.append((rd[:, :self.n] * xr + rd[:, self.n:] * xi).sum(-1) + self.read_b[s])
                m = P[s]  # (B, n, n): next j from state q weight m[q, j]
                c, sn = torch.cos(self.phi[s]), torch.sin(self.phi[s])  # phase on the transition taken from q
                ar, ai = xr * c - xi * sn, xr * sn + xi * c
                xr, xi = torch.einsum("bq,bqj->bj", ar, m), torch.einsum("bq,bqj->bj", ai, m)
            return torch.stack(outs, 1), "logits"
        bitp = (1 - torch.cos(self.phi)) / 2  # (4, n)
        if hard:
            snapped = (torch.round(self.phi / math.pi).remainder(2)).float()
            bitp = snapped + bitp - bitp.detach()
        p = s0
        outs = []
        for i in range(w):
            s = sym[:, i]
            outs.append((p * bitp[s]).sum(-1))
            p = torch.einsum("bq,bqj->bj", p, P[s])
        if hard:  # forward values are exactly 0/1, so a clamped log loss would have zero gradient: use squared error
            return torch.stack(outs, 1), "hard_probs"
        return torch.stack(outs, 1).clamp(1e-6, 1 - 1e-6), "probs"

    def export(self):
        """Integer Table: argmax maps; emission from the snapped angle (phase/ste) or the readout at a phase-0 basis state (cart)."""
        with torch.no_grad():
            trans = self.trans.argmax(-1).tolist()
            start = self.start.argmax(-1).tolist()
            if self.arm == "cart":
                rd = self.read
                emit = [[int((rd[s, q] + self.read_b[s]) > 0) for q in range(self.n)] for s in range(4)]
            else:
                emit = (torch.round(self.phi / math.pi).remainder(2)).long().tolist()
        return D._canonical(self.n, trans, emit, start, False)

    def grid_distance(self):
        with torch.no_grad():
            return float((self.phi / math.pi - torch.round(self.phi / math.pi)).abs().mean() * math.pi)


def train_arm(arm, a, b, r, cf, width, seed, steps=1500, lr=0.05, batch=64):
    torch.manual_seed(seed)
    m = Soft(arm, seed=seed)
    sym = torch.from_numpy(2 * D.bits(a, width).astype(np.int64) + D.bits(b, width).astype(np.int64))
    y = torch.from_numpy(D.bits(r, width).astype(np.float32))
    c = torch.from_numpy(np.asarray(cf, dtype=np.int64))
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed + 1)
    for _ in range(steps):
        idx = torch.randint(0, len(sym), (min(batch, len(sym)),), generator=g)
        out, kind = m(sym[idx], c[idx])
        if kind == "logits":
            loss = F.binary_cross_entropy_with_logits(out, y[idx])
        elif kind == "hard_probs":
            loss = F.mse_loss(out, y[idx])
        else:
            loss = F.binary_cross_entropy(out, y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m, loss.item()


@torch.no_grad()
def soft_predict(m, a, b, cf, width, snap=False):
    sym = torch.from_numpy(2 * D.bits(a, width).astype(np.int64) + D.bits(b, width).astype(np.int64))
    c = torch.from_numpy(np.asarray(cf, dtype=np.int64))
    if snap:
        saved = (m.trans.clone(), m.phi.clone(), m.start.clone())
        m.trans.copy_(F.one_hot(m.trans.argmax(-1), m.n).float() * 50 - 25)
        m.start.copy_(F.one_hot(m.start.argmax(-1), m.n).float() * 50 - 25)
        m.phi.copy_(torch.round(m.phi / math.pi) * math.pi)
    out, kind = m(sym, c)
    if snap:
        m.trans.copy_(saved[0])
        m.phi.copy_(saved[1])
        m.start.copy_(saved[2])
    bit = (out > 0) if kind == "logits" else (out > 0.5)
    return D.from_bits(bit.numpy().astype(np.uint8))
