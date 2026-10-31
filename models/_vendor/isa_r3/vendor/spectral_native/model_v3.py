"""Spectral-native model v3: v2's register file, with the audit fixes.

v2 (measured, job-d6940951b50f) never learned recall or copy. A probe at init showed why:
every head selected its own token (100%), the backward softmax was saturated (scaled logit
std ≈ 18, max prob 1.0), and the gradient reaching every "where to look" parameter was
~1e-28. The forward was exact and superposition-free; the model simply could not move.

The math audit found that, and four places where v2 dropped half of an SPF quantity:

1. Trainability. Hard top-1 selection is scale-free in the forward (argmax ignores the
   temperature), so the backward temperature is a free choice. v3 scores in channel-mean
   units and initializes the backward scale so a random query's logits have std ≈ 1.
2. Self-collapse. Distance matching has a degenerate fixed point: a token matches itself
   at distance 0. The register file already holds the token's own content, so a head never
   needs to copy itself: v3 attends strictly to j < t (position 0 keeps itself).
3. The position law was used as 32 free phases. A relative offset δ is one integer on the
   law, Θ_q = p·ω − δ·ω (exact, T1). v3 parameterizes the position query by an integer δ
   per head (straight-through), initialized to diverse offsets 1..heads. The ``free``
   ablation keeps v2's per-channel masks (same init, same function at init).
4. The law's magnitude half was dropped. ℓ(τ) = e0 + τ·v has a log-magnitude part (the
   decay of ALiBi / xPos / Mamba). v3 adds a per-head recency slope λ ≥ 0: score −= λ·(t−j).
5. The key's magnitude was dropped from the score. Re(q̄k) = |q||k|cos Δ; in the log domain
   the gate is L_q + L_k, an integer add. v2 gated by L_q only.
6. The MLP read phase through |Θ − anchor|, which drops which side of the anchor Θ is on.
   v3 feeds the signed offset s and −s (a tropical map is monotone, so it needs both signs).
7. The MLP's phase output bound ~140 input phases per channel (a dense ternary map, the phase
   analog of a Linear layer), a hash nothing downstream can match. v3's MLP query binds at most
   two selected slots: q = Θ_i + c·Θ_j + mask, c ∈ {−1, 0, 1} (role ⊗ filler).
8. The two-layer tropical MLP collapses (max-plus products are associative), and what remains is
   the same class of map as a nearest-key lookup. v3's MLP is a lookup over a learned table:
   match → select → copy, the attention primitive over parameters instead of context
   (SlotMLPV3 docstring). It reads any slot written before it through the query selectors.
9. The readout matched only phase codes; v3 ties the magnitude half too (gate L + emb_L[v]).

No change adds a sum of complex values: every slot is still written once, by an exact copy,
a tropical selection or a binding (phase add).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import L_Q, T_Q, circ, smoothmax, ste, wrap
from .model_v2 import pick, pick_phase


@dataclass
class ConfigV3:
    vocab: int = 256
    slot_ch: int = 32
    heads: int = 4
    layers: int = 4
    hidden: int = 64
    max_len: int = 512
    attn: str = "select"      # select | soft
    mlp: str = "lookup"       # lookup | tropical | relu
    mlp_keys: int = 256       # lookup table entries per MLP
    pos: str = "law"          # law | free
    tau: float = 0.1          # tropical smoothmax backward temperature
    sel_tau: float = 1.0      # slot-selector backward temperature
    bwd_scale: float = 16.0   # initial backward softmax scale on channel-mean scores
    rev: int = 1              # 2 = v3b: fixes from the second (Opus 4.8) audit, see pick_v()
    lookup_scale: float = 64.0  # v3b: backward scale for the K-entry table (sharper than attention)
    selector_grad: str = "double"  # double (v3b) | single (third-audit ablation)
    recency: str = "float"         # float (v3b) | quantized (fully integer hard route)

    @property
    def slots(self) -> int:
        return 2 + self.layers * (self.heads + 1)


def pick_v(
    x: torch.Tensor,
    logits: torch.Tensor,
    tau: float,
    rev: int,
    selector_grad: str = "double",
) -> torch.Tensor:
    """Slot selection. Forward: the exact selected slot. rev 1 (v2's ``pick``): value gradient only to the
    selected slot, so the L of a slot nobody selects at init gets exactly zero gradient (measured).
    rev 2: value gradient through the soft mix, like ``pick_phase``; the forward is unchanged."""
    if rev < 2:
        return pick(x, logits, tau)
    soft = torch.softmax(logits / tau, -1)
    k = int(logits.argmax())
    hard = x[..., k, :]
    s = torch.einsum("s,...sc->...c", soft, x)
    result = hard + (s - s.detach())
    if selector_grad == "single":
        return result
    if selector_grad != "double":
        raise ValueError(f"unknown selector_grad mode: {selector_grad}")
    return result + torch.einsum("s,...sc->...c", soft - soft.detach(), x)


def signed(x: torch.Tensor) -> torch.Tensor:
    """Signed ring offset in turns, in [-0.5, 0.5)."""
    return torch.remainder(x + 0.5, 1.0) - 0.5


class SlotHeadV3(nn.Module):
    def __init__(self, cfg: ConfigV3, out_slot: int, offset: int, omega: torch.Tensor):
        super().__init__()
        S, C = cfg.slots, cfg.slot_ch
        self.cfg, self.out_slot = cfg, out_slot
        self.sel = nn.ParameterDict({k: nn.Parameter(torch.randn(S) * 0.1) for k in ("qb", "kb", "v")})
        with torch.no_grad():  # start: content term compares content, read content
            self.sel["qb"][1] += 1.0
            self.sel["kb"][1] += 1.0
            self.sel["v"][1] += 1.0
        self.delta = nn.Parameter(torch.tensor(float(offset)))            # law: integer relative offset
        self.mask_a = nn.Parameter(torch.remainder(-offset * omega.float(), 1.0))  # free: same shift, 32 phases
        self.mask_b = nn.Parameter(torch.rand(C) * 0.01)
        self.w_a = nn.Parameter(torch.ones(C))
        self.w_b = nn.Parameter(torch.ones(C))
        self.slope = nn.Parameter(torch.tensor(-4.0))                     # λ = softplus(slope) ≈ 0.018
        self.temp = nn.Parameter(torch.tensor(math.log(cfg.bwd_scale)))

    def recency_slope(self) -> torch.Tensor:
        """Nonnegative recency coefficient used by the hard route."""

        slope = F.softplus(self.slope)
        if self.cfg.recency == "quantized":
            return ste(slope, L_Q)
        if self.cfg.recency != "float":
            raise ValueError(f"unknown recency mode: {self.cfg.recency}")
        return slope

    def forward(self, L, th, pos_T, omega):
        b, t, _slots, C = th.shape
        tau = self.cfg.sel_tau
        if self.cfg.pos == "law":
            qa = wrap(pos_T - ste(self.delta, 1.0) * omega.float())       # exact: (p − δ)·ω on the ring
        else:
            qa = wrap(pos_T + ste(self.mask_a, T_Q))
        da = circ(qa[:, None] - pos_T[None, :])                            # t,s,C (same for every batch row)
        pos_term = (da * F.relu(ste(self.w_a, L_Q))).mean(-1)              # t,s
        qb = wrap(pick_phase(th, self.sel["qb"], tau) + ste(self.mask_b, T_Q))
        kb = pick_phase(th, self.sel["kb"], tau)
        gate = F.relu(
            pick_v(L, self.sel["qb"], tau, self.cfg.rev, self.cfg.selector_grad)[:, :, None]
            + pick_v(L, self.sel["kb"], tau, self.cfg.rev, self.cfg.selector_grad)[:, None]
            + ste(self.w_b, L_Q)
        )
        content_term = (circ(qb[:, :, None] - kb[:, None]) * gate).mean(-1)  # b,t,s
        ti = torch.arange(t, device=th.device)
        age = (ti[:, None] - ti[None, :]).to(th.dtype)                     # integer t − j
        raw = -pos_term[None] - content_term - self.recency_slope() * age / C
        allowed = ti[None, :] < ti[:, None]
        allowed[0, 0] = True                                               # position 0 has only itself
        sc = raw.masked_fill(~allowed, float("-inf"))
        soft = torch.softmax((raw * self.temp.exp()).masked_fill(~allowed, float("-inf")), -1)  # mask after scaling
        vT = pick_phase(th, self.sel["v"], tau)
        vL = pick_v(L, self.sel["v"], tau, self.cfg.rev, self.cfg.selector_grad)
        if self.cfg.attn == "soft":  # Cartesian ablation: superpose the selected values
            z = torch.exp2(vL) * torch.exp(2j * math.pi * vT)
            zo = torch.einsum("bts,bsc->btc", soft.to(z.dtype), z)
            return torch.log2(zo.abs().clamp_min(1e-30)), wrap(torch.angle(zo) / (2 * math.pi))
        idx = sc.argmax(-1)
        hT = torch.gather(vT, 1, idx[..., None].expand(b, t, C))
        hL = torch.gather(vL, 1, idx[..., None].expand(b, t, C))
        rel = signed(vT[:, None] - hT[:, :, None])
        sT = torch.einsum("bts,btsc->btc", soft, rel)
        sL = torch.einsum("bts,bsc->btc", soft, vL)
        return hL + (sL - sL.detach()), wrap(hT + (sT - sT.detach()))


class SlotMLPV3(nn.Module):
    """One slot out. Default: a LOOKUP — the same match → select → copy primitive as attention, over a
    learned table instead of the context.

    Why (the algebra, see docs/findings.md): a two-layer max-plus network with no other operation
    between its layers collapses, because max-plus matrix products are associative:
    max_k(W2_ck + max_i(W1_ki + x_i)) = max_i(x_i + (W2 ⊗ W1)_ci). The hidden layer is only a
    tropical-rank bottleneck, and materialising it cost b·t·d·N floats. What is left, max − max over
    inputs, is piecewise affine on the torus; nearest-key selection under the ring L1 metric with a
    carried offset is the same class of map. So the native MLP is a table lookup:

        query   q  = Θ_i + c·Θ_j + mask        (binding of at most two slots, c ∈ {−1,0,1}; exact)
        score_k    = −mean_c relu(L_q + Kw_kc) · circ(q_c − Kt_kc)
        k*         = argmax_k score_k           (exact integer compare)
        out Θ      = Vt_k* + ρ·(q − Kt_k*)      (ρ ∈ {0,1}: constant or affine piece; exact adds)
        out L      = VL_k*

    Cost b·t·K·C instead of b·t·d·N. Ablations: ``tropical`` (the collapsed one-layer max − max
    over every slot's L and signed phase offsets) and ``relu`` (linear + ReLU on the same inputs).
    """

    def __init__(self, cfg: ConfigV3, out_slot: int, n_in: int):
        super().__init__()
        C, K = cfg.slot_ch, cfg.mlp_keys
        self.cfg, self.out_slot, self.n_in = cfg, out_slot, n_in
        self.bsel = nn.ParameterDict({k: nn.Parameter(torch.randn(n_in) * 0.1) for k in ("i", "j")})
        with torch.no_grad():
            self.bsel["i"][1] += 1.0
            self.bsel["j"][0] += 1.0  # content ⊗ position
        # logits over c ∈ {−1, 0, +1}; v3b starts with the binding on (c = +1) so its selector gets gradient
        self.coef = nn.Parameter(torch.tensor([0.0, 0.0, 1.0] if cfg.rev >= 2 else [0.0, 1.0, 0.0]))
        self.mask_o = nn.Parameter(torch.rand(C) * 0.01)
        if cfg.mlp == "lookup":
            self.key_T = nn.Parameter(torch.rand(K, C))
            self.key_w = nn.Parameter(torch.ones(K, C))
            self.val_T = nn.Parameter(torch.rand(K, C))
            self.val_L = nn.Parameter(torch.randn(K, C) * 0.1 if cfg.rev >= 2 else torch.zeros(K, C))
            # logits over ρ ∈ {0, 1}
            self.carry = nn.Parameter(torch.tensor([0.0, 1.0] if cfg.rev >= 2 else [1.0, 0.0]))
            self.temp = nn.Parameter(torch.tensor(math.log(cfg.lookup_scale if cfg.rev >= 2 else cfg.bwd_scale)))
        else:
            N = 3 * n_in * C
            self.anchor = nn.Parameter(torch.rand(n_in, C))
            if cfg.mlp == "tropical":
                self.A = nn.Parameter(torch.randn(C, N) * 0.5)
                self.B = nn.Parameter(torch.randn(C, N) * 0.5)
            else:
                self.lin1 = nn.Linear(N, cfg.hidden)
                self.lin2 = nn.Linear(cfg.hidden, C)

    @staticmethod
    def _choice(logits: torch.Tensor, values: list[float]) -> torch.Tensor:
        soft = torch.softmax(logits, -1)
        hard = F.one_hot(logits.argmax(), len(values)).to(soft.dtype)
        return ((soft + (hard - soft).detach()) * torch.tensor(values, device=logits.device)).sum()

    def forward(self, L, th):
        L, th = L[:, :, :self.n_in], th[:, :, :self.n_in]
        b, t, S, C = th.shape
        tau = self.cfg.sel_tau
        c = self._choice(self.coef, [-1.0, 0.0, 1.0])
        q = wrap(pick_phase(th, self.bsel["i"], tau) + c * pick_phase(th, self.bsel["j"], tau) + ste(self.mask_o, T_Q))
        if self.cfg.mlp == "lookup":
            kT = ste(self.key_T, T_Q)
            Lq = pick_v(L, self.bsel["i"], tau, self.cfg.rev, self.cfg.selector_grad)
            gate = F.relu(Lq[:, :, None] + ste(self.key_w, L_Q)[None, None])            # b,t,K,C
            raw = -(circ(q[:, :, None] - kT[None, None]) * gate).mean(-1)              # b,t,K
            soft = torch.softmax(raw * self.temp.exp(), -1)
            idx = raw.argmax(-1)                                                       # b,t
            vT, vL = ste(self.val_T, T_Q), ste(self.val_L, L_Q)
            hT, hL, hK = vT[idx], vL[idx], kT[idx]                                     # b,t,C
            sT = torch.einsum("btk,btkc->btc", soft, signed(vT[None, None] - hT[:, :, None]))
            sL = torch.einsum("btk,kc->btc", soft, vL)
            rho = self._choice(self.carry, [0.0, 1.0])
            oT = wrap(hT + (sT - sT.detach()) + rho * signed(q - hK))
            return hL + (sL - sL.detach()), ste(oT, T_Q)
        d = th - ste(self.anchor, T_Q)
        if self.cfg.rev >= 2:   # two quadrature ring distances keep the side without a cut
            s1, s2 = -8.0 * circ(d).reshape(b, t, S * C), -8.0 * circ(d - 0.25).reshape(b, t, S * C)
        else:
            s1 = 8.0 * signed(d).reshape(b, t, S * C)
            s2 = -s1
        x = torch.cat([L.reshape(b, t, S * C), s1, s2], -1)
        if self.cfg.mlp == "tropical":   # collapsed: max_i(A_ci + x_i) − max_i(B_ci + x_i)
            oL = smoothmax(ste(self.A, L_Q)[None, None] + x[:, :, None, :], -1, self.cfg.tau) \
                - smoothmax(ste(self.B, L_Q)[None, None] + x[:, :, None, :], -1, self.cfg.tau)
        else:
            oL = self.lin2(F.relu(self.lin1(x)))
        return ste(oL, L_Q), q


class SpectralNativeV3(nn.Module):
    def __init__(self, cfg: ConfigV3):
        super().__init__()
        self.cfg = cfg
        S, C = cfg.slots, cfg.slot_ch
        self.emb_T = nn.Parameter(torch.rand(cfg.vocab, C))
        self.emb_L = nn.Parameter(torch.randn(cfg.vocab, C) * 0.1 if cfg.rev >= 2 else torch.zeros(cfg.vocab, C))
        omega = torch.tensor([2.0 ** -(1 + 9 * i / C) for i in range(C)], dtype=torch.float64)
        self.register_buffer("omega", torch.round(omega * T_Q) / T_Q)
        self.heads = nn.ModuleList()
        self.mlps = nn.ModuleList()
        slot = 2
        for _ in range(cfg.layers):
            layer = nn.ModuleList()
            for h in range(cfg.heads):
                layer.append(SlotHeadV3(cfg, slot, h + 1, self.omega))
                slot += 1
            self.heads.append(layer)
            self.mlps.append(SlotMLPV3(cfg, slot, slot))
            slot += 1
        self.ro_sel = nn.ParameterDict({k: nn.Parameter(torch.randn(S) * 0.1) for k in ("a", "b")})
        with torch.no_grad():
            self.ro_sel["a"][2 + (cfg.layers - 1) * (cfg.heads + 1)] += 1.0
            self.ro_sel["b"][S - 1] += 1.0
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
        pos = torch.arange(t, device=dev, dtype=torch.float64)
        pos_T = (pos[:, None] * self.omega[None]).remainder(1.0).float()   # t,C  exact on the 2^-16 grid
        emb_T, emb_L = ste(self.emb_T, T_Q), ste(self.emb_L, L_Q)
        zero = torch.zeros(b, t, C, device=dev)
        slot_T = [pos_T[None].expand(b, t, C), emb_T[ids]] + [zero] * (S - 2)
        slot_L = [zero, emb_L[ids]] + [zero] * (S - 2)
        for layer, mlp in zip(self.heads, self.mlps, strict=True):
            th, L = torch.stack(slot_T, 2), torch.stack(slot_L, 2)
            for head in layer:
                oL, oT = head(L, th, pos_T, self.omega)
                slot_T[head.out_slot], slot_L[head.out_slot] = oT, oL
            th, L = torch.stack(slot_T, 2), torch.stack(slot_L, 2)
            oL, oT = mlp(L, th)
            slot_T[mlp.out_slot], slot_L[mlp.out_slot] = oT, oL
        th, L = torch.stack(slot_T, 2), torch.stack(slot_L, 2)
        tau = self.cfg.sel_tau
        logits = ste(self.ro_bias, L_Q)[None, None]
        for k, mask, w in (("a", self.ro_mask_a, self.ro_w_a), ("b", self.ro_mask_b, self.ro_w_b)):
            r = wrap(pick_phase(th, self.ro_sel[k], tau) + ste(mask, T_Q))
            d = circ(r[:, :, None, :] - emb_T[None, None])                           # tied phase codes
            gate = F.relu(
                pick_v(L, self.ro_sel[k], tau, self.cfg.rev, self.cfg.selector_grad)[:, :, None]
                + emb_L[None, None]
                + ste(w, L_Q)
            )  # tied magnitudes
            logits = logits - (d * gate).sum(-1)
        return logits * F.softplus(self.ro_scale)


def phase_params_v3(model: nn.Module) -> list[nn.Parameter]:
    names = ("emb_T", "mask_a", "mask_b", "mask_o", "anchor", "key_T", "val_T", "ro_mask_a", "ro_mask_b")
    return [p for n, p in model.named_parameters() if n.split(".")[-1] in names]


@torch.no_grad()
def wrap_phase_params_v3(model: nn.Module) -> None:
    for p in phase_params_v3(model):
        p.remainder_(1.0)
