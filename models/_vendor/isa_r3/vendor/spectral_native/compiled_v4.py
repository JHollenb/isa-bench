"""Compiled hard forward for the multidimensional spectral-field model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as nnf

from .compiled_v3 import HardForwardTrace, LookupProgram, _margin
from .model import L_Q, T_Q, circ, ste, wrap
from .model_v3 import signed
from .model_v4 import SpectralFieldV4, field_cross_position_scores


@dataclass(frozen=True)
class FieldHeadProgram:
    out_slot: int
    query_slot: int
    key_slot: int
    value_slot: int
    relative_offset: tuple[int, ...]


@dataclass(frozen=True)
class FieldLayerProgram:
    heads: tuple[FieldHeadProgram, ...]
    lookup: LookupProgram


@dataclass(frozen=True)
class CompiledV4Program:
    coord_dims: int
    layers: tuple[FieldLayerProgram, ...]
    readout_slots: tuple[int, int]


def _hard_index(logits: torch.Tensor) -> int:
    return int(logits.argmax())


def compile_field_program(model: SpectralFieldV4) -> CompiledV4Program:
    """Freeze selectors and vector offsets into a literal field program."""

    cfg = model.cfg
    if cfg.rev < 2 or cfg.attn != "select" or cfg.pos != "law" or cfg.mlp != "lookup":
        raise ValueError("compiled v4 requires rev>=2, select attention, a vector law, and lookup MLP")
    layers: list[FieldLayerProgram] = []
    for heads, lookup in zip(model.heads, model.mlps, strict=True):
        head_programs = tuple(
            FieldHeadProgram(
                out_slot=int(head.out_slot),
                query_slot=_hard_index(head.sel["qb"]),
                key_slot=_hard_index(head.sel["kb"]),
                value_slot=_hard_index(head.sel["v"]),
                relative_offset=tuple(
                    int(value) for value in torch.round(head.delta.detach()).to(torch.int64).tolist()
                ),
            )
            for head in heads
        )
        lookup_program = LookupProgram(
            out_slot=int(lookup.out_slot),
            left_slot=_hard_index(lookup.bsel["i"]),
            right_slot=_hard_index(lookup.bsel["j"]),
            coefficient=(-1, 0, 1)[_hard_index(lookup.coef)],
            carry=(0, 1)[_hard_index(lookup.carry)],
        )
        layers.append(FieldLayerProgram(head_programs, lookup_program))
    return CompiledV4Program(
        coord_dims=cfg.coord_dims,
        layers=tuple(layers),
        readout_slots=(_hard_index(model.ro_sel["a"]), _hard_index(model.ro_sel["b"])),
    )


@torch.no_grad()
def compiled_field_forward(
    model: SpectralFieldV4,
    ids: torch.Tensor,
    coords: torch.Tensor | None,
    *,
    query_coords: torch.Tensor | None = None,
    program: CompiledV4Program | None = None,
) -> tuple[torch.Tensor, HardForwardTrace]:
    """Execute the route-frozen field program without backward surrogates."""

    program = compile_field_program(model) if program is None else program
    if program.coord_dims != model.cfg.coord_dims:
        raise ValueError("compiled program coordinate rank does not match model")
    if len(program.layers) != len(model.heads):
        raise ValueError("compiled program layer count does not match model")
    batch, tokens = ids.shape
    slots, channels = model.cfg.slots, model.cfg.slot_ch
    position_phase = model.position_phase(ids, coords)
    query_position_phase = position_phase if query_coords is None else model.position_phase(ids, query_coords)
    embedding_phase, embedding_logmag = ste(model.emb_T, T_Q), ste(model.emb_L, L_Q)
    zero = torch.zeros(batch, tokens, channels, device=ids.device)
    slot_phase = [position_phase, embedding_phase[ids]] + [zero] * (slots - 2)
    slot_logmag = [zero, embedding_logmag[ids]] + [zero] * (slots - 2)
    all_head_routes: list[tuple[torch.Tensor, ...]] = []
    all_head_margins: list[tuple[torch.Tensor, ...]] = []
    all_lookup_indices: list[torch.Tensor] = []
    all_lookup_margins: list[torch.Tensor] = []

    for head_modules, lookup_module, layer_program in zip(
        model.heads,
        model.mlps,
        program.layers,
        strict=True,
    ):
        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_logmag, 2)
        layer_routes: list[torch.Tensor] = []
        layer_margins: list[torch.Tensor] = []
        for head, head_program in zip(head_modules, layer_program.heads, strict=True):
            delta = torch.tensor(
                head_program.relative_offset,
                device=ids.device,
                dtype=model.omega.dtype,
            )
            position_scores = field_cross_position_scores(
                query_position_phase,
                position_phase,
                model.omega,
                delta,
                head.w_a,
            )
            query_phase = wrap(phase[:, :, head_program.query_slot] + ste(head.mask_b, T_Q))
            key_phase = phase[:, :, head_program.key_slot]
            query_logmag = logmag[:, :, head_program.query_slot]
            key_logmag = logmag[:, :, head_program.key_slot]
            gate = nnf.relu(query_logmag[:, :, None] + key_logmag[:, None] + ste(head.w_b, L_Q))
            content_term = (circ(query_phase[:, :, None] - key_phase[:, None]) * gate).mean(-1)
            token_index = torch.arange(tokens, device=ids.device)
            age = (token_index[:, None] - token_index[None, :]).to(query_phase.dtype)
            raw = position_scores - content_term - head.recency_slope() * age / channels
            allowed = token_index[None, :] < token_index[:, None]
            allowed[0, 0] = True
            scores = raw.masked_fill(~allowed, float("-inf"))
            route = scores.argmax(-1)
            gather_index = route[..., None].expand(batch, tokens, channels)
            slot_phase[head_program.out_slot] = wrap(
                torch.gather(phase[:, :, head_program.value_slot], 1, gather_index)
            )
            slot_logmag[head_program.out_slot] = torch.gather(
                logmag[:, :, head_program.value_slot],
                1,
                gather_index,
            )
            layer_routes.append(route.detach().cpu())
            layer_margins.append(_margin(scores).detach().cpu())
        all_head_routes.append(tuple(layer_routes))
        all_head_margins.append(tuple(layer_margins))

        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_logmag, 2)
        lookup_program = layer_program.lookup
        query_phase = wrap(
            phase[:, :, lookup_program.left_slot]
            + lookup_program.coefficient * phase[:, :, lookup_program.right_slot]
            + ste(lookup_module.mask_o, T_Q)
        )
        key_phase = ste(lookup_module.key_T, T_Q)
        query_logmag = logmag[:, :, lookup_program.left_slot]
        gate = nnf.relu(query_logmag[:, :, None] + ste(lookup_module.key_w, L_Q)[None, None])
        lookup_scores = -(circ(query_phase[:, :, None] - key_phase[None, None]) * gate).mean(-1)
        lookup_index = lookup_scores.argmax(-1)
        value_phase = ste(lookup_module.val_T, T_Q)
        value_logmag = ste(lookup_module.val_L, L_Q)
        hard_phase = value_phase[lookup_index]
        hard_key = key_phase[lookup_index]
        output_phase = wrap(hard_phase + lookup_program.carry * signed(query_phase - hard_key))
        slot_phase[lookup_program.out_slot] = ste(output_phase, T_Q)
        slot_logmag[lookup_program.out_slot] = value_logmag[lookup_index]
        all_lookup_indices.append(lookup_index.detach().cpu())
        all_lookup_margins.append(_margin(lookup_scores).detach().cpu())

    phase = torch.stack(slot_phase, 2)
    logmag = torch.stack(slot_logmag, 2)
    logits = ste(model.ro_bias, L_Q)[None, None]
    for slot, mask, weight in zip(
        program.readout_slots,
        (model.ro_mask_a, model.ro_mask_b),
        (model.ro_w_a, model.ro_w_b),
        strict=True,
    ):
        query_phase = wrap(phase[:, :, slot] + ste(mask, T_Q))
        distance = circ(query_phase[:, :, None] - embedding_phase[None, None])
        gate = nnf.relu(logmag[:, :, slot, None] + embedding_logmag[None, None] + ste(weight, L_Q))
        logits = logits - (distance * gate).sum(-1)
    logits = logits * nnf.softplus(model.ro_scale)
    trace = HardForwardTrace(
        head_routes=tuple(all_head_routes),
        head_margins=tuple(all_head_margins),
        lookup_indices=tuple(all_lookup_indices),
        lookup_margins=tuple(all_lookup_margins),
        slot_log_magnitudes=logmag.detach().cpu(),
        slot_phases=phase.detach().cpu(),
        readout_margin=_margin(logits).detach().cpu(),
    )
    return logits, trace
