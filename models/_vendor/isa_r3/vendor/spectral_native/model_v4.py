"""Multidimensional spectral-field register machine.

V3 has many phase channels but only one base coordinate: every positional
address is on the rank-one orbit ``t * omega``.  V4 changes the base space to
an integer lattice ``r in Z**D`` and uses the exact affine phase law

    Theta(r, c) = sum_d r[d] * Omega[d, c]  (mod 1 turn).

The differentiable implementation below is a grid-snapped training surrogate.
``integer_v4`` is the authoritative hard inference path: it carries phase and
log-magnitude codes as integers and performs no complex decode.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .model import L_Q, T_Q, circ, ste, wrap
from .model_v2 import pick_phase
from .model_v3 import ConfigV3, SlotMLPV3, pick_v


@dataclass
class ConfigV4(ConfigV3):
    """V3 register configuration plus the rank of the field base space."""

    coord_dims: int = 2
    axis_layout: str = "orthogonal"

    def __post_init__(self) -> None:
        if self.coord_dims < 1:
            raise ValueError("coord_dims must be positive")
        if self.axis_layout != "orthogonal":
            raise ValueError("v4 currently implements only the orthogonal axis layout")
        if self.pos != "law":
            raise ValueError("v4 requires a vector phase law")


def orthogonal_axis_frequencies(coord_dims: int, channels: int) -> torch.Tensor:
    """Return independent grid-snapped phase laws, one channel band per axis.

    For ``coord_dims == 1`` this is bit-identical to v3's frequency vector.
    With more axes, channel ``c`` belongs to ``c % coord_dims`` and each axis
    starts at a quarter turn rather than a half turn.  That removes the
    two-cell alias that would otherwise make opposite one-step displacements
    locally indistinguishable.  The zero entries make axis independence
    inspectable and turn a joint address comparison into a sum of separable
    per-axis comparisons.
    """

    if coord_dims < 1 or channels < coord_dims:
        raise ValueError("channels must be at least coord_dims, and both must be positive")
    omega = torch.zeros(coord_dims, channels, dtype=torch.float64)
    exponent_start = 1.0 if coord_dims == 1 else 2.0
    exponent_span = 9.0 if coord_dims == 1 else 8.0
    for axis in range(coord_dims):
        axis_channels = list(range(axis, channels, coord_dims))
        count = len(axis_channels)
        for local_index, channel in enumerate(axis_channels):
            exponent = exponent_start + exponent_span * local_index / count
            omega[axis, channel] = 2.0**-exponent
    return torch.round(omega * T_Q) / T_Q


def field_phase_turns(coords: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """Evaluate the multidimensional phase law in turns.

    ``coords`` is integer ``(batch, token, dimension)`` and ``omega`` is the
    grid-snapped ``(dimension, channel)`` law.  The float tensor exists for the
    training surrogate only; hard inference evaluates the same products in
    ring codes in :mod:`integer_v4`.
    """

    if coords.ndim != 3:
        raise ValueError("coords must have shape (batch, token, dimension)")
    if omega.ndim != 2 or coords.shape[-1] != omega.shape[0]:
        raise ValueError("coordinate rank does not match the phase law")
    if coords.is_floating_point() and not torch.equal(coords, torch.round(coords)):
        raise ValueError("field coordinates must be integer-valued")
    phase = torch.einsum("btd,dc->btc", coords.to(torch.float64), omega.to(torch.float64))
    return torch.remainder(phase, 1.0).to(torch.float32)


def field_position_scores(
    position_phase: torch.Tensor,
    omega: torch.Tensor,
    delta: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Score every source cell against a vector-displaced query cell."""

    return field_cross_position_scores(
        position_phase,
        position_phase,
        omega,
        delta,
        weights,
    )


def field_cross_position_scores(
    query_position_phase: torch.Tensor,
    key_position_phase: torch.Tensor,
    omega: torch.Tensor,
    delta: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Score stored keys against queries that may name the next address.

    A causal language state at token ``t`` predicts token ``t + 1``.  Its
    query coordinate can therefore be the address of the next token while the
    stored key coordinate remains the address of the observed token.  Keeping
    those two clocks explicit avoids flattening the field back into sequence
    order merely to account for the one-token language-model shift.
    """

    if query_position_phase.ndim != 3 or key_position_phase.ndim != 3:
        raise ValueError("query and key position phases must be rank three")
    if query_position_phase.shape != key_position_phase.shape:
        raise ValueError("query and key position phases must have the same shape")
    if delta.ndim != 1 or delta.shape[0] != omega.shape[0]:
        raise ValueError("delta must have one component per coordinate axis")
    shift = torch.einsum("d,dc->c", ste(delta, 1.0), omega.to(delta.dtype))
    query = wrap(query_position_phase - shift)
    distance = circ(query[:, :, None, :] - key_position_phase[:, None, :, :])
    return -(distance * nnf.relu(ste(weights, L_Q))).mean(-1)


class SlotHeadV4(nn.Module):
    """A v3 hard-copy head whose positional address is a vector displacement."""

    def __init__(
        self,
        cfg: ConfigV4,
        out_slot: int,
        offset_axis: int,
        offset_steps: int,
        omega: torch.Tensor,
    ) -> None:
        super().__init__()
        slots, channels = cfg.slots, cfg.slot_ch
        self.cfg, self.out_slot = cfg, out_slot
        self.sel = nn.ParameterDict({key: nn.Parameter(torch.randn(slots) * 0.1) for key in ("qb", "kb", "v")})
        with torch.no_grad():
            self.sel["qb"][1] += 1.0
            self.sel["kb"][1] += 1.0
            self.sel["v"][1] += 1.0
        initial_delta = torch.zeros(cfg.coord_dims)
        initial_delta[offset_axis] = float(offset_steps)
        self.delta = nn.Parameter(initial_delta)
        # Kept for checkpoint-shape continuity with v3 and as a future free-law
        # ablation. The vector-law hard path does not consume it.
        self.mask_a = nn.Parameter(torch.remainder(-initial_delta @ omega.float(), 1.0))
        self.mask_b = nn.Parameter(torch.rand(channels) * 0.01)
        self.w_a = nn.Parameter(torch.ones(channels))
        self.w_b = nn.Parameter(torch.ones(channels))
        self.slope = nn.Parameter(torch.tensor(-4.0))
        self.temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(cfg.bwd_scale)))))

    def recency_slope(self) -> torch.Tensor:
        slope = nnf.softplus(self.slope)
        if self.cfg.recency == "quantized":
            return ste(slope, L_Q)
        if self.cfg.recency != "float":
            raise ValueError(f"unknown recency mode: {self.cfg.recency}")
        return slope

    def forward(
        self,
        logmag: torch.Tensor,
        phase: torch.Tensor,
        query_position_phase: torch.Tensor,
        key_position_phase: torch.Tensor,
        omega: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _slots, channels = phase.shape
        tau = self.cfg.sel_tau
        position_term = -field_cross_position_scores(
            query_position_phase,
            key_position_phase,
            omega,
            self.delta,
            self.w_a,
        )
        query_phase = wrap(pick_phase(phase, self.sel["qb"], tau) + ste(self.mask_b, T_Q))
        key_phase = pick_phase(phase, self.sel["kb"], tau)
        gate = nnf.relu(
            pick_v(
                logmag,
                self.sel["qb"],
                tau,
                self.cfg.rev,
                self.cfg.selector_grad,
            )[:, :, None]
            + pick_v(
                logmag,
                self.sel["kb"],
                tau,
                self.cfg.rev,
                self.cfg.selector_grad,
            )[:, None]
            + ste(self.w_b, L_Q)
        )
        content_term = (circ(query_phase[:, :, None] - key_phase[:, None]) * gate).mean(-1)
        token_index = torch.arange(tokens, device=phase.device)
        age = (token_index[:, None] - token_index[None, :]).to(phase.dtype)
        raw = -position_term - content_term - self.recency_slope() * age / channels
        allowed = token_index[None, :] < token_index[:, None]
        allowed[0, 0] = True
        scores = raw.masked_fill(~allowed, float("-inf"))
        soft = torch.softmax(
            (raw * self.temp.exp()).masked_fill(~allowed, float("-inf")),
            -1,
        )
        value_phase = pick_phase(phase, self.sel["v"], tau)
        value_logmag = pick_v(
            logmag,
            self.sel["v"],
            tau,
            self.cfg.rev,
            self.cfg.selector_grad,
        )
        route = scores.argmax(-1)
        gather_index = route[..., None].expand(batch, tokens, channels)
        hard_phase = torch.gather(value_phase, 1, gather_index)
        hard_logmag = torch.gather(value_logmag, 1, gather_index)
        relative_phase = (
            torch.remainder(
                value_phase[:, None] - hard_phase[:, :, None] + 0.5,
                1.0,
            )
            - 0.5
        )
        soft_phase = torch.einsum("bts,btsc->btc", soft, relative_phase)
        soft_logmag = torch.einsum("bts,bsc->btc", soft, value_logmag)
        return (
            hard_logmag + (soft_logmag - soft_logmag.detach()),
            wrap(hard_phase + (soft_phase - soft_phase.detach())),
        )


class SpectralFieldV4(nn.Module):
    """Append-only spectral register program over a D-dimensional base lattice."""

    def __init__(self, cfg: ConfigV4):
        super().__init__()
        cfg.__post_init__()
        self.cfg = cfg
        slots, channels = cfg.slots, cfg.slot_ch
        self.emb_T = nn.Parameter(torch.rand(cfg.vocab, channels))
        self.emb_L = nn.Parameter(torch.randn(cfg.vocab, channels) * 0.1)
        omega = orthogonal_axis_frequencies(cfg.coord_dims, channels)
        self.register_buffer("omega", omega)
        self.heads = nn.ModuleList()
        self.mlps = nn.ModuleList()
        slot = 2
        for _layer_index in range(cfg.layers):
            layer = nn.ModuleList()
            for head_index in range(cfg.heads):
                axis = head_index % cfg.coord_dims
                steps = head_index // cfg.coord_dims + 1
                layer.append(SlotHeadV4(cfg, slot, axis, steps, self.omega))
                slot += 1
            self.heads.append(layer)
            self.mlps.append(SlotMLPV3(cfg, slot, slot))
            slot += 1
        self.ro_sel = nn.ParameterDict({key: nn.Parameter(torch.randn(slots) * 0.1) for key in ("a", "b")})
        with torch.no_grad():
            self.ro_sel["a"][2 + (cfg.layers - 1) * (cfg.heads + 1)] += 1.0
            self.ro_sel["b"][slots - 1] += 1.0
        self.ro_mask_a = nn.Parameter(torch.zeros(channels))
        self.ro_mask_b = nn.Parameter(torch.zeros(channels))
        self.ro_w_a = nn.Parameter(torch.ones(channels))
        self.ro_w_b = nn.Parameter(torch.ones(channels) * 0.1)
        self.ro_bias = nn.Parameter(torch.zeros(cfg.vocab))
        self.ro_scale = nn.Parameter(torch.tensor(0.0))

    def normalize_coords(self, ids: torch.Tensor, coords: torch.Tensor | None) -> torch.Tensor:
        batch, tokens = ids.shape
        if coords is None:
            if self.cfg.coord_dims != 1:
                raise ValueError("multidimensional v4 forward requires explicit coordinates")
            coords = torch.arange(tokens, device=ids.device)[None, :, None].expand(batch, -1, -1)
        if coords.shape != (batch, tokens, self.cfg.coord_dims):
            raise ValueError(
                f"coords must have shape {(batch, tokens, self.cfg.coord_dims)}, got {tuple(coords.shape)}"
            )
        return coords.to(device=ids.device, dtype=torch.int64)

    def position_phase(self, ids: torch.Tensor, coords: torch.Tensor | None) -> torch.Tensor:
        return field_phase_turns(self.normalize_coords(ids, coords), self.omega)

    def forward(
        self,
        ids: torch.Tensor,
        coords: torch.Tensor | None = None,
        query_coords: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, tokens = ids.shape
        slots, channels = self.cfg.slots, self.cfg.slot_ch
        position_phase = self.position_phase(ids, coords)
        query_position_phase = position_phase if query_coords is None else self.position_phase(ids, query_coords)
        embedding_phase, embedding_logmag = ste(self.emb_T, T_Q), ste(self.emb_L, L_Q)
        zero = torch.zeros(batch, tokens, channels, device=ids.device)
        slot_phase = [position_phase, embedding_phase[ids]] + [zero] * (slots - 2)
        slot_logmag = [zero, embedding_logmag[ids]] + [zero] * (slots - 2)
        for layer, mlp in zip(self.heads, self.mlps, strict=True):
            phase = torch.stack(slot_phase, 2)
            logmag = torch.stack(slot_logmag, 2)
            for head in layer:
                output_logmag, output_phase = head(
                    logmag,
                    phase,
                    query_position_phase,
                    position_phase,
                    self.omega,
                )
                slot_phase[head.out_slot] = output_phase
                slot_logmag[head.out_slot] = output_logmag
            phase = torch.stack(slot_phase, 2)
            logmag = torch.stack(slot_logmag, 2)
            output_logmag, output_phase = mlp(logmag, phase)
            slot_phase[mlp.out_slot] = output_phase
            slot_logmag[mlp.out_slot] = output_logmag
        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_logmag, 2)
        tau = self.cfg.sel_tau
        logits = ste(self.ro_bias, L_Q)[None, None]
        for key, mask, weight in (
            ("a", self.ro_mask_a, self.ro_w_a),
            ("b", self.ro_mask_b, self.ro_w_b),
        ):
            read_phase = wrap(pick_phase(phase, self.ro_sel[key], tau) + ste(mask, T_Q))
            distance = circ(read_phase[:, :, None, :] - embedding_phase[None, None])
            gate = nnf.relu(
                pick_v(
                    logmag,
                    self.ro_sel[key],
                    tau,
                    self.cfg.rev,
                    self.cfg.selector_grad,
                )[:, :, None]
                + embedding_logmag[None, None]
                + ste(weight, L_Q)
            )
            logits = logits - (distance * gate).sum(-1)
        return logits * nnf.softplus(self.ro_scale)


def phase_params_v4(model: nn.Module) -> list[nn.Parameter]:
    names = (
        "emb_T",
        "mask_a",
        "mask_b",
        "mask_o",
        "anchor",
        "key_T",
        "val_T",
        "ro_mask_a",
        "ro_mask_b",
    )
    return [parameter for name, parameter in model.named_parameters() if name.split(".")[-1] in names]


@torch.no_grad()
def wrap_phase_params_v4(model: nn.Module) -> None:
    for parameter in phase_params_v4(model):
        parameter.remainder_(1.0)
