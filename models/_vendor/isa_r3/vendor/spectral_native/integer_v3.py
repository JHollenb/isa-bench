"""Integer lowering for the hard v3b spectral register machine.

The phase ring has ``R = 2**16`` ticks and log magnitudes have ``Q = 2**8``
ticks per bit.  Once checkpoint-global slot selectors are compiled, every v3b
hard-forward operation is integer except the learned recency coefficient in an
attention head.  This module exposes both useful interpretations:

``quantize_recency=False``
    Keep that one coefficient continuous while computing every circular and
    magnitude term from exact integer numerators.  This is a parity oracle for
    the current model.

``quantize_recency=True``
    Snap the nonnegative coefficient to the log-magnitude grid.  Attention,
    lookup, phase writes, magnitude copies, and readout then use integer
    arithmetic end to end.  The returned readout scores are exact integer
    numerators; the model's final positive scalar does not affect argmax.

This is an inference/reference implementation.  It intentionally fails closed
through :func:`compile_hard_program` rather than pretending to lower soft or
Cartesian ablation arms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as nnf

from .compiled_v3 import CompiledV3Program, compile_hard_program
from .model import L_Q, T_Q
from .model_v3 import SpectralNativeV3


@dataclass(frozen=True)
class IntegerV3Trace:
    """Routes, terminal state, and exact readout numerators."""

    head_routes: tuple[tuple[torch.Tensor, ...], ...]
    lookup_indices: tuple[torch.Tensor, ...]
    slot_log_codes: torch.Tensor
    slot_phase_codes: torch.Tensor
    readout_score_numerators: torch.Tensor
    predictions: torch.Tensor
    quantized_recency: bool


def _phase_codes(value: torch.Tensor, ring: int) -> torch.Tensor:
    return torch.remainder(torch.round(value.detach() * ring).to(torch.int64), ring)


def _log_codes(value: torch.Tensor, quantum: int) -> torch.Tensor:
    return torch.round(value.detach() * quantum).to(torch.int64)


def _circular_distance_codes(left: torch.Tensor, right: torch.Tensor, ring: int) -> torch.Tensor:
    difference = torch.remainder(left - right, ring)
    return torch.minimum(difference, ring - difference)


def _masked_route(scores: torch.Tensor) -> torch.Tensor:
    """Apply v3's strict-past mask and return deterministic first argmax."""

    tokens = scores.shape[-1]
    token_index = torch.arange(tokens, device=scores.device)
    allowed = token_index[None, :] < token_index[:, None]
    allowed[0, 0] = True
    if scores.is_floating_point():
        return scores.masked_fill(~allowed, float("-inf")).argmax(-1)
    floor = torch.iinfo(scores.dtype).min
    return scores.masked_fill(~allowed, floor).argmax(-1)


@torch.no_grad()
def integer_hard_forward(
    model: SpectralNativeV3,
    ids: torch.Tensor,
    *,
    program: CompiledV3Program | None = None,
    quantize_recency: bool = True,
) -> IntegerV3Trace:
    """Lower a hard v3b forward to ring/log codes and exact score numerators."""

    program = compile_hard_program(model) if program is None else program
    if ids.ndim != 2:
        raise ValueError("ids must have shape (batch, tokens)")
    if len(program.layers) != len(model.heads):
        raise ValueError("compiled program layer count does not match model")

    ring, quantum = int(T_Q), int(L_Q)
    batch, tokens = ids.shape
    slots, channels = model.cfg.slots, model.cfg.slot_ch
    device = ids.device
    position = torch.arange(tokens, device=device, dtype=torch.int64)
    omega_codes = _phase_codes(model.omega, ring)
    position_codes = torch.remainder(position[:, None] * omega_codes[None], ring)
    embedding_phase = _phase_codes(model.emb_T, ring)
    embedding_log = _log_codes(model.emb_L, quantum)
    zero = torch.zeros(batch, tokens, channels, dtype=torch.int64, device=device)
    slot_phase = [position_codes[None].expand(batch, tokens, channels), embedding_phase[ids]] + [
        zero
    ] * (slots - 2)
    slot_log = [zero, embedding_log[ids]] + [zero] * (slots - 2)
    all_head_routes: list[tuple[torch.Tensor, ...]] = []
    all_lookup_indices: list[torch.Tensor] = []

    for head_modules, lookup_module, layer_program in zip(
        model.heads,
        model.mlps,
        program.layers,
        strict=True,
    ):
        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_log, 2)
        layer_routes: list[torch.Tensor] = []

        for head, head_program in zip(head_modules, layer_program.heads, strict=True):
            query_position = torch.remainder(
                position_codes - head_program.relative_offset * omega_codes,
                ring,
            )
            position_distance = _circular_distance_codes(
                query_position[:, None],
                position_codes[None, :],
                ring,
            )
            position_gate = torch.relu(_log_codes(head.w_a, quantum))
            position_numerator = (position_distance * position_gate).sum(-1, dtype=torch.int64)

            query_phase = torch.remainder(
                phase[:, :, head_program.query_slot] + _phase_codes(head.mask_b, ring),
                ring,
            )
            key_phase = phase[:, :, head_program.key_slot]
            query_log = logmag[:, :, head_program.query_slot]
            key_log = logmag[:, :, head_program.key_slot]
            gate = torch.relu(
                query_log[:, :, None]
                + key_log[:, None]
                + _log_codes(head.w_b, quantum)
            )
            content_distance = _circular_distance_codes(
                query_phase[:, :, None],
                key_phase[:, None],
                ring,
            )
            content_numerator = (content_distance * gate).sum(-1, dtype=torch.int64)
            base_numerator = position_numerator[None] + content_numerator

            token_index = torch.arange(tokens, device=device, dtype=torch.int64)
            age = token_index[:, None] - token_index[None, :]
            if quantize_recency:
                recency_code = torch.round(nnf.softplus(head.slope.detach()) * quantum).to(torch.int64)
                recency_numerator = recency_code * age * ring
                scores = -(base_numerator + recency_numerator[None])
            else:
                scores = -base_numerator.to(torch.float64) / (quantum * ring * channels)
                scores = scores - head.recency_slope().detach().double() * age / channels
            route = _masked_route(scores)
            gather_index = route[..., None].expand(batch, tokens, channels)
            slot_phase[head_program.out_slot] = torch.gather(
                phase[:, :, head_program.value_slot],
                1,
                gather_index,
            )
            slot_log[head_program.out_slot] = torch.gather(
                logmag[:, :, head_program.value_slot],
                1,
                gather_index,
            )
            layer_routes.append(route.cpu())
        all_head_routes.append(tuple(layer_routes))

        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_log, 2)
        lookup_program = layer_program.lookup
        query_phase = torch.remainder(
            phase[:, :, lookup_program.left_slot]
            + lookup_program.coefficient * phase[:, :, lookup_program.right_slot]
            + _phase_codes(lookup_module.mask_o, ring),
            ring,
        )
        key_phase = _phase_codes(lookup_module.key_T, ring)
        query_log = logmag[:, :, lookup_program.left_slot]
        gate = torch.relu(query_log[:, :, None] + _log_codes(lookup_module.key_w, quantum))
        distance = _circular_distance_codes(query_phase[:, :, None], key_phase[None, None], ring)
        lookup_numerator = (distance * gate).sum(-1, dtype=torch.int64)
        lookup_index = lookup_numerator.argmin(-1)
        value_phase = _phase_codes(lookup_module.val_T, ring)
        value_log = _log_codes(lookup_module.val_L, quantum)
        if lookup_program.carry:
            output_phase = torch.remainder(
                value_phase[lookup_index] + query_phase - key_phase[lookup_index],
                ring,
            )
        else:
            output_phase = value_phase[lookup_index]
        slot_phase[lookup_program.out_slot] = output_phase
        slot_log[lookup_program.out_slot] = value_log[lookup_index]
        all_lookup_indices.append(lookup_index.cpu())

    phase = torch.stack(slot_phase, 2)
    logmag = torch.stack(slot_log, 2)
    readout_score = _log_codes(model.ro_bias, quantum)[None, None] * ring
    readout_score = readout_score.expand(batch, tokens, -1).clone()
    for slot, mask, weight in zip(
        program.readout_slots,
        (model.ro_mask_a, model.ro_mask_b),
        (model.ro_w_a, model.ro_w_b),
        strict=True,
    ):
        query_phase = torch.remainder(phase[:, :, slot] + _phase_codes(mask, ring), ring)
        distance = _circular_distance_codes(query_phase[:, :, None], embedding_phase[None, None], ring)
        gate = torch.relu(
            logmag[:, :, slot, None]
            + embedding_log[None, None]
            + _log_codes(weight, quantum)
        )
        readout_score -= (distance * gate).sum(-1, dtype=torch.int64)

    return IntegerV3Trace(
        head_routes=tuple(all_head_routes),
        lookup_indices=tuple(all_lookup_indices),
        slot_log_codes=torch.stack(slot_log, 2).cpu(),
        slot_phase_codes=torch.stack(slot_phase, 2).cpu(),
        readout_score_numerators=readout_score.cpu(),
        predictions=readout_score.argmax(-1).cpu(),
        quantized_recency=quantize_recency,
    )
