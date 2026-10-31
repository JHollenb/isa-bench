"""Exact integer hard forward for the multidimensional spectral-field model."""

from __future__ import annotations

import torch
import torch.nn.functional as nnf

from .compiled_v4 import CompiledV4Program, compile_field_program
from .integer_v3 import IntegerV3Trace
from .model import L_Q, T_Q
from .model_v4 import SpectralFieldV4


def _phase_codes(value: torch.Tensor, ring: int) -> torch.Tensor:
    return torch.remainder(torch.round(value.detach() * ring).to(torch.int64), ring)


def _log_codes(value: torch.Tensor, quantum: int) -> torch.Tensor:
    return torch.round(value.detach() * quantum).to(torch.int64)


def field_phase_codes(
    coords: torch.Tensor,
    omega_codes: torch.Tensor,
    ring: int,
) -> torch.Tensor:
    """Evaluate ``coords @ omega`` directly in integer ring codes."""

    if coords.ndim != 3 or omega_codes.ndim != 2:
        raise ValueError("coords and omega_codes must be rank 3 and rank 2")
    if coords.shape[-1] != omega_codes.shape[0]:
        raise ValueError("coordinate rank does not match omega")
    return torch.remainder(
        torch.einsum("btd,dc->btc", coords.to(torch.int64), omega_codes.to(torch.int64)),
        ring,
    )


def _circular_distance(left: torch.Tensor, right: torch.Tensor, ring: int) -> torch.Tensor:
    difference = torch.remainder(left - right, ring)
    return torch.minimum(difference, ring - difference)


def _masked_route(scores: torch.Tensor) -> torch.Tensor:
    tokens = scores.shape[-1]
    token_index = torch.arange(tokens, device=scores.device)
    allowed = token_index[None, :] < token_index[:, None]
    allowed[0, 0] = True
    return scores.masked_fill(~allowed, torch.iinfo(scores.dtype).min).argmax(-1)


@torch.no_grad()
def integer_field_forward(
    model: SpectralFieldV4,
    ids: torch.Tensor,
    coords: torch.Tensor | None,
    *,
    query_coords: torch.Tensor | None = None,
    program: CompiledV4Program | None = None,
) -> IntegerV3Trace:
    """Run v4 entirely on phase/log integer codes and score numerators."""

    program = compile_field_program(model) if program is None else program
    normalized_coords = model.normalize_coords(ids, coords)
    normalized_query_coords = normalized_coords if query_coords is None else model.normalize_coords(ids, query_coords)
    ring, quantum = int(T_Q), int(L_Q)
    batch, tokens = ids.shape
    slots, channels = model.cfg.slots, model.cfg.slot_ch
    omega_codes = _phase_codes(model.omega, ring)
    position_codes = field_phase_codes(normalized_coords, omega_codes, ring)
    query_position_codes = field_phase_codes(
        normalized_query_coords,
        omega_codes,
        ring,
    )
    embedding_phase = _phase_codes(model.emb_T, ring)
    embedding_logmag = _log_codes(model.emb_L, quantum)
    zero = torch.zeros(
        batch,
        tokens,
        channels,
        dtype=torch.int64,
        device=ids.device,
    )
    slot_phase = [position_codes, embedding_phase[ids]] + [zero] * (slots - 2)
    slot_logmag = [zero, embedding_logmag[ids]] + [zero] * (slots - 2)
    all_head_routes: list[tuple[torch.Tensor, ...]] = []
    all_lookup_indices: list[torch.Tensor] = []

    for head_modules, lookup_module, layer_program in zip(
        model.heads,
        model.mlps,
        program.layers,
        strict=True,
    ):
        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_logmag, 2)
        layer_routes: list[torch.Tensor] = []
        for head, head_program in zip(head_modules, layer_program.heads, strict=True):
            delta = torch.tensor(
                head_program.relative_offset,
                dtype=torch.int64,
                device=ids.device,
            )
            shift_codes = torch.einsum("d,dc->c", delta, omega_codes)
            query_position = torch.remainder(
                query_position_codes - shift_codes,
                ring,
            )
            position_distance = _circular_distance(
                query_position[:, :, None],
                position_codes[:, None],
                ring,
            )
            position_gate = torch.relu(_log_codes(head.w_a, quantum))
            position_numerator = (position_distance * position_gate).sum(-1, dtype=torch.int64)

            query_phase = torch.remainder(
                phase[:, :, head_program.query_slot] + _phase_codes(head.mask_b, ring),
                ring,
            )
            key_phase = phase[:, :, head_program.key_slot]
            gate = torch.relu(
                logmag[:, :, head_program.query_slot, None]
                + logmag[:, None, :, head_program.key_slot]
                + _log_codes(head.w_b, quantum)
            )
            content_distance = _circular_distance(
                query_phase[:, :, None],
                key_phase[:, None],
                ring,
            )
            content_numerator = (content_distance * gate).sum(-1, dtype=torch.int64)
            token_index = torch.arange(tokens, device=ids.device, dtype=torch.int64)
            age = token_index[:, None] - token_index[None, :]
            recency_code = torch.round(nnf.softplus(head.slope.detach()) * quantum).to(torch.int64)
            recency_numerator = recency_code * age * ring
            scores = -(position_numerator + content_numerator + recency_numerator[None])
            route = _masked_route(scores)
            gather_index = route[..., None].expand(batch, tokens, channels)
            slot_phase[head_program.out_slot] = torch.gather(
                phase[:, :, head_program.value_slot],
                1,
                gather_index,
            )
            slot_logmag[head_program.out_slot] = torch.gather(
                logmag[:, :, head_program.value_slot],
                1,
                gather_index,
            )
            layer_routes.append(route.cpu())
        all_head_routes.append(tuple(layer_routes))

        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_logmag, 2)
        lookup_program = layer_program.lookup
        query_phase = torch.remainder(
            phase[:, :, lookup_program.left_slot]
            + lookup_program.coefficient * phase[:, :, lookup_program.right_slot]
            + _phase_codes(lookup_module.mask_o, ring),
            ring,
        )
        key_phase = _phase_codes(lookup_module.key_T, ring)
        gate = torch.relu(logmag[:, :, lookup_program.left_slot, None] + _log_codes(lookup_module.key_w, quantum))
        distance = _circular_distance(
            query_phase[:, :, None],
            key_phase[None, None],
            ring,
        )
        lookup_numerator = (distance * gate).sum(-1, dtype=torch.int64)
        lookup_index = lookup_numerator.argmin(-1)
        value_phase = _phase_codes(lookup_module.val_T, ring)
        value_logmag = _log_codes(lookup_module.val_L, quantum)
        if lookup_program.carry:
            output_phase = torch.remainder(
                value_phase[lookup_index] + query_phase - key_phase[lookup_index],
                ring,
            )
        else:
            output_phase = value_phase[lookup_index]
        slot_phase[lookup_program.out_slot] = output_phase
        slot_logmag[lookup_program.out_slot] = value_logmag[lookup_index]
        all_lookup_indices.append(lookup_index.cpu())

    phase = torch.stack(slot_phase, 2)
    logmag = torch.stack(slot_logmag, 2)
    readout_score = _log_codes(model.ro_bias, quantum)[None, None] * ring
    readout_score = readout_score.expand(batch, tokens, -1).clone()
    for slot, mask, weight in zip(
        program.readout_slots,
        (model.ro_mask_a, model.ro_mask_b),
        (model.ro_w_a, model.ro_w_b),
        strict=True,
    ):
        query_phase = torch.remainder(
            phase[:, :, slot] + _phase_codes(mask, ring),
            ring,
        )
        distance = _circular_distance(
            query_phase[:, :, None],
            embedding_phase[None, None],
            ring,
        )
        gate = torch.relu(logmag[:, :, slot, None] + embedding_logmag[None, None] + _log_codes(weight, quantum))
        readout_score -= (distance * gate).sum(-1, dtype=torch.int64)

    return IntegerV3Trace(
        head_routes=tuple(all_head_routes),
        lookup_indices=tuple(all_lookup_indices),
        slot_log_codes=torch.stack(slot_logmag, 2).cpu(),
        slot_phase_codes=torch.stack(slot_phase, 2).cpu(),
        readout_score_numerators=readout_score.cpu(),
        predictions=readout_score.argmax(-1).cpu(),
        quantized_recency=True,
    )
