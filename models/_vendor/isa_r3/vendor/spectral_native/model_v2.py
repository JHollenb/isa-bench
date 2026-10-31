"""Spectral-native model v2: tokens are register files of separate slots (no superposition).

v1 failed to learn retrieval because position and content shared the same channels (a head
could shift phases but not separate them) and the binding residual wrote each head's output
onto those same channels (superposition by product). v2 keeps every piece of information in
its own slot, as the Registrar and Black Mamba do:

* slot 0            position law  Θ = p·ω (SPF phase law), L = 0
* slot 1            content       Θ = token phase code, L = token log magnitude
* slots 2..         one slot per attention head and one per MLP, written once by overwrite

Attention head: two comparison terms, each comparing a selected slot of the query token
(bound with a learned phase mask) against a selected slot of the key token, by integer
circular distance with learned channel weights. Hard top-1 selection of the key token,
then an exact copy of a selected slot of that token into the head's own slot.
Slot selectors are hard one-hots with straight-through gradients.

MLP: tropical rational map on all slots' L plus phase distances to anchors, and a ternary
phase map over all slots' phases, written into the MLP's own slot.

Readout: two match terms, each a selected slot (bound with a mask) against the tied
vocabulary phase codes (the same codes the embedding uses), plus a per-token bias. An exact
copy of the right token's content slot therefore matches its own code exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import L_Q, T_Q, circ, smoothmax, ste, ternary, wrap


@dataclass
class ConfigV2:
    vocab: int = 256
    slot_ch: int = 32
    heads: int = 4
    layers: int = 4
    hidden: int = 64
    max_len: int = 512
    attn: str = "select"      # select | soft
    mlp: str = "tropical"     # tropical | relu
    tau: float = 0.1
    sel_tau: float = 1.0

    @property
    def slots(self) -> int:
        return 2 + self.layers * (self.heads + 1)


def pick(x: torch.Tensor, logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Select one slot of x (…, S, C) by a hard one-hot over S with a straight-through softmax."""
    soft = torch.softmax(logits / tau, -1)
    hard = F.one_hot(logits.argmax(-1), logits.shape[-1]).to(soft.dtype)
    p = soft + (hard - soft).detach()
    return torch.einsum("s,...sc->...c", p, x)


def pick_phase(th: torch.Tensor, logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Phase-safe slot selection: forward is the exact selected phase; gradient via unwrapped soft mix."""
    soft = torch.softmax(logits / tau, -1)
    k = int(logits.argmax())
    hard = th[..., k, :]
    rel = torch.remainder(th - hard[..., None, :] + 0.5, 1.0) - 0.5
    s = torch.einsum("s,...sc->...c", soft, rel)
    return wrap(hard + (s - s.detach()))


class SlotHead(nn.Module):
    def __init__(self, cfg: ConfigV2, out_slot: int):
        super().__init__()
        S, C = cfg.slots, cfg.slot_ch
        self.cfg, self.out_slot = cfg, out_slot
        self.sel = nn.ParameterDict({k: nn.Parameter(torch.randn(S) * 0.1) for k in ("qa", "ka", "qb", "kb", "v")})
        with torch.no_grad():  # a sensible start: term a compares positions, term b compares content, read content
            self.sel["qa"][0] += 1.0; self.sel["ka"][0] += 1.0
            self.sel["qb"][1] += 1.0; self.sel["kb"][1] += 1.0
            self.sel["v"][1] += 1.0
        self.mask_a = nn.Parameter(torch.rand(C) * 0.01)
        self.mask_b = nn.Parameter(torch.rand(C) * 0.01)
        self.w_a = nn.Parameter(torch.ones(C))
        self.w_b = nn.Parameter(torch.ones(C))
        self.temp = nn.Parameter(torch.tensor(0.0))

    def forward(self, L: torch.Tensor, th: torch.Tensor):
        b, t, S, C = th.shape
        tau = self.cfg.sel_tau
        qa = wrap(pick_phase(th, self.sel["qa"], tau) + ste(self.mask_a, T_Q))
        ka = pick_phase(th, self.sel["ka"], tau)
        qb = wrap(pick_phase(th, self.sel["qb"], tau) + ste(self.mask_b, T_Q))
        kb = pick_phase(th, self.sel["kb"], tau)
        # the query slot's own magnitudes gate which channels count: weight = max(0, L + w)
        wa = F.relu(pick(L, self.sel["qa"], tau) + ste(self.w_a, L_Q))[:, :, None]      # b,t,1,C
        wb = F.relu(pick(L, self.sel["qb"], tau) + ste(self.w_b, L_Q))[:, :, None]
        raw = -(circ(qa[:, :, None] - ka[:, None]) * wa).sum(-1) - (circ(qb[:, :, None] - kb[:, None]) * wb).sum(-1)  # b,t,s
        causal = torch.ones(t, t, dtype=torch.bool, device=th.device).tril()
        sc = raw.masked_fill(~causal, float("-inf"))
        soft = torch.softmax((raw * F.softplus(self.temp) / self.cfg.tau).masked_fill(~causal, float("-inf")), -1)
        vT = pick_phase(th, self.sel["v"], tau)            # b,t,C
        vL = pick(L, self.sel["v"], tau)
        if self.cfg.attn == "soft":  # Cartesian ablation: superpose the selected values
            z = torch.exp2(vL) * torch.exp(2j * math.pi * vT)
            zo = torch.einsum("bts,bsc->btc", soft.to(z.dtype), z)
            return torch.log2(zo.abs().clamp_min(1e-30)), wrap(torch.angle(zo) / (2 * math.pi))
        idx = sc.argmax(-1)                                                         # b,t
        hT = torch.gather(vT, 1, idx[..., None].expand(b, t, C))
        hL = torch.gather(vL, 1, idx[..., None].expand(b, t, C))
        rel = torch.remainder(vT[:, None] - hT[:, :, None] + 0.5, 1.0) - 0.5         # b,t,s,C
        sT = torch.einsum("bts,btsc->btc", soft, rel)
        sL = torch.einsum("bts,bsc->btc", soft, vL)
        return hL + (sL - sL.detach()), wrap(hT + (sT - sT.detach()))


class SlotMLP(nn.Module):
    """Reads the position slot, the content slot and this layer's head slots; writes its own slot."""

    def __init__(self, cfg: ConfigV2, out_slot: int, in_slots: list[int]):
        super().__init__()
        S, C, d = len(in_slots), cfg.slot_ch, cfg.hidden
        self.cfg, self.out_slot = cfg, out_slot
        self.register_buffer("in_slots", torch.tensor(in_slots), persistent=False)
        self.anchor = nn.Parameter(torch.rand(S, C))
        self.W1 = nn.Parameter(torch.randn(d, 2 * S * C) * 0.5)
        self.W2 = nn.Parameter(torch.randn(C, d) * 0.5)
        self.W3 = nn.Parameter(torch.randn(C, d) * 0.5)
        self.lin1 = nn.Linear(2 * S * C, d)
        self.lin2 = nn.Linear(d, C)
        self.A = nn.Parameter(torch.randn(C, S * C) * 0.05)
        self.phase_bias = nn.Parameter(torch.rand(C))

    def forward(self, L: torch.Tensor, th: torch.Tensor):
        L, th = L[:, :, self.in_slots], th[:, :, self.in_slots]
        b, t, S, C = th.shape
        tau = self.cfg.tau
        x = torch.cat([L.reshape(b, t, S * C), -8.0 * circ(th - ste(self.anchor, T_Q)).reshape(b, t, S * C)], -1)
        if self.cfg.mlp == "tropical":
            h = smoothmax(ste(self.W1, L_Q)[None, None] + x[:, :, None, :], -1, tau)
            oL = smoothmax(ste(self.W2, L_Q)[None, None] + h[:, :, None, :], -1, tau) \
                - smoothmax(ste(self.W3, L_Q)[None, None] + h[:, :, None, :], -1, tau)
        else:
            oL = self.lin2(F.relu(self.lin1(x)))
        oT = wrap(th.reshape(b, t, S * C) @ ternary(self.A).T + ste(self.phase_bias, T_Q))
        return ste(oL, L_Q), ste(oT, T_Q)


class SpectralNativeV2(nn.Module):
    def __init__(self, cfg: ConfigV2):
        super().__init__()
        self.cfg = cfg
        S, C = cfg.slots, cfg.slot_ch
        self.emb_T = nn.Parameter(torch.rand(cfg.vocab, C))
        self.emb_L = nn.Parameter(torch.zeros(cfg.vocab, C))
        omega = torch.tensor([2.0 ** -(1 + 9 * i / C) for i in range(C)], dtype=torch.float64)
        self.register_buffer("omega", torch.round(omega * T_Q) / T_Q)
        self.heads = nn.ModuleList()
        self.mlps = nn.ModuleList()
        slot = 2
        for _ in range(cfg.layers):
            layer = nn.ModuleList()
            first = slot
            for _ in range(cfg.heads):
                layer.append(SlotHead(cfg, slot)); slot += 1
            self.heads.append(layer)
            self.mlps.append(SlotMLP(cfg, slot, [0, 1] + list(range(first, slot)))); slot += 1
        self.ro_sel = nn.ParameterDict({k: nn.Parameter(torch.randn(S) * 0.1) for k in ("a", "b")})
        with torch.no_grad():
            self.ro_sel["a"][2 + (cfg.layers - 1) * (cfg.heads + 1)] += 1.0   # start reading the last layer's first head
            self.ro_sel["b"][S - 1] += 1.0                                    # and the last MLP slot
        self.ro_mask_a = nn.Parameter(torch.zeros(C))
        self.ro_mask_b = nn.Parameter(torch.zeros(C))
        self.ro_w_a = nn.Parameter(torch.ones(C))
        self.ro_w_b = nn.Parameter(torch.ones(C) * 0.1)
        self.ro_bias = nn.Parameter(torch.zeros(cfg.vocab))
        self.ro_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        S, C = self.cfg.slots, self.cfg.slot_ch
        dev = ids.device
        th = torch.zeros(b, t, S, C, device=dev)
        L = torch.zeros(b, t, S, C, device=dev)
        pos = torch.arange(t, device=dev, dtype=torch.float64)
        slot_pos = (pos[:, None] * self.omega[None]).remainder(1.0).float()[None].expand(b, t, C)
        emb_T = ste(self.emb_T, T_Q)
        slot_T = [slot_pos, emb_T[ids]] + [torch.zeros(b, t, C, device=dev)] * (S - 2)
        slot_L = [torch.zeros(b, t, C, device=dev), ste(self.emb_L, L_Q)[ids]] + [torch.zeros(b, t, C, device=dev)] * (S - 2)
        for layer, mlp in zip(self.heads, self.mlps):
            th, L = torch.stack(slot_T, 2), torch.stack(slot_L, 2)
            for head in layer:
                oL, oT = head(L, th)
                slot_T[head.out_slot], slot_L[head.out_slot] = oT, oL
            th, L = torch.stack(slot_T, 2), torch.stack(slot_L, 2)
            oL, oT = mlp(L, th)
            slot_T[mlp.out_slot], slot_L[mlp.out_slot] = oT, oL
        th, L = torch.stack(slot_T, 2), torch.stack(slot_L, 2)
        tau = self.cfg.sel_tau
        ra = wrap(pick_phase(th, self.ro_sel["a"], tau) + ste(self.ro_mask_a, T_Q))
        rb = wrap(pick_phase(th, self.ro_sel["b"], tau) + ste(self.ro_mask_b, T_Q))
        da = circ(ra[:, :, None, :] - emb_T[None, None])            # tied vocabulary codes
        db = circ(rb[:, :, None, :] - emb_T[None, None])
        wa = F.relu(pick(L, self.ro_sel["a"], tau) + ste(self.ro_w_a, L_Q))[:, :, None]   # magnitude-gated match
        wb = F.relu(pick(L, self.ro_sel["b"], tau) + ste(self.ro_w_b, L_Q))[:, :, None]
        logits = -(da * wa).sum(-1) - (db * wb).sum(-1) + ste(self.ro_bias, L_Q)
        return logits * F.softplus(self.ro_scale)


def phase_params_v2(model: nn.Module) -> list[nn.Parameter]:
    names = ("emb_T", "mask_a", "mask_b", "anchor", "phase_bias", "ro_mask_a", "ro_mask_b")
    return [p for n, p in model.named_parameters() if n.split(".")[-1] in names]


@torch.no_grad()
def wrap_phase_params_v2(model: nn.Module) -> None:
    for p in phase_params_v2(model):
        p.remainder_(1.0)
