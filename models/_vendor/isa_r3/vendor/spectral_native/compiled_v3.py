"""Hard-forward compiler and route-conditioned phase proof for spectral v3.

The default v3b model is an append-only register machine, not the residual
binding architecture in ``model.py``.  Every head and lookup writes a fresh
slot while earlier slots remain available.  This module removes backward-only
soft paths, freezes checkpoint-global selector decisions, records the dynamic
routes, and compiles one realized phase trace into an affine program over the
phase ring.

The lowering is deliberately inference-only and fail-closed to the native v3b
``select/law/lookup`` arm.  It reuses the model's quantization, wrap, distance,
and signed-offset primitives rather than defining a second arithmetic system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as nnf

from .model import L_Q, T_Q, circ, ste, wrap
from .model_v3 import SpectralNativeV3, signed


@dataclass(frozen=True)
class HeadProgram:
    out_slot: int
    query_slot: int
    key_slot: int
    value_slot: int
    relative_offset: int


@dataclass(frozen=True)
class LookupProgram:
    out_slot: int
    left_slot: int
    right_slot: int
    coefficient: int
    carry: int


@dataclass(frozen=True)
class LayerProgram:
    heads: tuple[HeadProgram, ...]
    lookup: LookupProgram


@dataclass(frozen=True)
class CompiledV3Program:
    layers: tuple[LayerProgram, ...]
    readout_slots: tuple[int, int]


@dataclass
class HardForwardTrace:
    head_routes: tuple[tuple[torch.Tensor, ...], ...]
    head_margins: tuple[tuple[torch.Tensor, ...], ...]
    lookup_indices: tuple[torch.Tensor, ...]
    lookup_margins: tuple[torch.Tensor, ...]
    slot_log_magnitudes: torch.Tensor
    slot_phases: torch.Tensor
    readout_margin: torch.Tensor


@dataclass(frozen=True)
class PhaseAffineTrace:
    """One batch row's affine slot program over initial token-content phases."""

    coefficients: np.ndarray  # (slot, token, channel, source_token), modulo T_Q
    constants: np.ndarray  # (slot, token, channel), modulo T_Q
    reconstructed_codes: np.ndarray
    native_codes: np.ndarray

    @property
    def exact(self) -> bool:
        return bool(np.array_equal(self.reconstructed_codes, self.native_codes))


def _hard_index(logits: torch.Tensor) -> int:
    return int(logits.argmax())


def compile_hard_program(model: SpectralNativeV3) -> CompiledV3Program:
    """Freeze the checkpoint-global hard wiring of a default v3b model."""

    cfg = model.cfg
    if cfg.rev < 2 or cfg.attn != "select" or cfg.pos != "law" or cfg.mlp != "lookup":
        raise ValueError("compiled v3 lowering requires rev>=2, select attention, law position, and lookup MLP")
    layers: list[LayerProgram] = []
    for heads, lookup in zip(model.heads, model.mlps, strict=True):
        head_programs = tuple(
            HeadProgram(
                out_slot=int(head.out_slot),
                query_slot=_hard_index(head.sel["qb"]),
                key_slot=_hard_index(head.sel["kb"]),
                value_slot=_hard_index(head.sel["v"]),
                relative_offset=int(torch.round(head.delta.detach()).item()),
            )
            for head in heads
        )
        coefficient = (-1, 0, 1)[_hard_index(lookup.coef)]
        carry = (0, 1)[_hard_index(lookup.carry)]
        lookup_program = LookupProgram(
            out_slot=int(lookup.out_slot),
            left_slot=_hard_index(lookup.bsel["i"]),
            right_slot=_hard_index(lookup.bsel["j"]),
            coefficient=coefficient,
            carry=carry,
        )
        layers.append(LayerProgram(head_programs, lookup_program))
    return CompiledV3Program(
        layers=tuple(layers),
        readout_slots=(_hard_index(model.ro_sel["a"]), _hard_index(model.ro_sel["b"])),
    )


def _margin(scores: torch.Tensor) -> torch.Tensor:
    if scores.shape[-1] < 2:
        return torch.full(scores.shape[:-1], float("inf"), device=scores.device, dtype=scores.dtype)
    top = torch.topk(scores, 2, dim=-1).values
    return top[..., 0] - top[..., 1]


@torch.no_grad()
def compiled_hard_forward(
    model: SpectralNativeV3,
    ids: torch.Tensor,
    *,
    program: CompiledV3Program | None = None,
) -> tuple[torch.Tensor, HardForwardTrace]:
    """Execute the hard v3b forward without any backward-only soft mixtures."""

    program = compile_hard_program(model) if program is None else program
    if len(program.layers) != len(model.heads):
        raise ValueError("compiled program layer count does not match model")
    batch, tokens = ids.shape
    slots, channels = model.cfg.slots, model.cfg.slot_ch
    device = ids.device
    position = torch.arange(tokens, device=device, dtype=torch.float64)
    pos_phase = (position[:, None] * model.omega[None]).remainder(1.0).float()
    emb_phase, emb_log = ste(model.emb_T, T_Q), ste(model.emb_L, L_Q)
    zero = torch.zeros(batch, tokens, channels, device=device)
    slot_phase = [pos_phase[None].expand(batch, tokens, channels), emb_phase[ids]] + [zero] * (slots - 2)
    slot_log = [zero, emb_log[ids]] + [zero] * (slots - 2)

    all_head_routes: list[tuple[torch.Tensor, ...]] = []
    all_head_margins: list[tuple[torch.Tensor, ...]] = []
    all_lookup_indices: list[torch.Tensor] = []
    all_lookup_margins: list[torch.Tensor] = []

    for layer_modules, lookup_module, layer_program in zip(
        model.heads,
        model.mlps,
        program.layers,
        strict=True,
    ):
        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_log, 2)
        layer_routes: list[torch.Tensor] = []
        layer_margins: list[torch.Tensor] = []
        for head, head_program in zip(layer_modules, layer_program.heads, strict=True):
            query_pos = wrap(pos_phase - head_program.relative_offset * model.omega.float())
            pos_distance = circ(query_pos[:, None] - pos_phase[None, :])
            pos_term = (pos_distance * nnf.relu(ste(head.w_a, L_Q))).mean(-1)

            query_phase = wrap(phase[:, :, head_program.query_slot] + ste(head.mask_b, T_Q))
            key_phase = phase[:, :, head_program.key_slot]
            query_log = logmag[:, :, head_program.query_slot]
            key_log = logmag[:, :, head_program.key_slot]
            gate = nnf.relu(query_log[:, :, None] + key_log[:, None] + ste(head.w_b, L_Q))
            content_term = (circ(query_phase[:, :, None] - key_phase[:, None]) * gate).mean(-1)

            token_index = torch.arange(tokens, device=device)
            age = (token_index[:, None] - token_index[None, :]).to(query_phase.dtype)
            raw = -pos_term[None] - content_term - head.recency_slope() * age / channels
            allowed = token_index[None, :] < token_index[:, None]
            allowed[0, 0] = True
            scores = raw.masked_fill(~allowed, float("-inf"))
            route = scores.argmax(-1)
            value_phase = phase[:, :, head_program.value_slot]
            value_log = logmag[:, :, head_program.value_slot]
            gather_index = route[..., None].expand(batch, tokens, channels)
            out_phase = wrap(torch.gather(value_phase, 1, gather_index))
            out_log = torch.gather(value_log, 1, gather_index)
            slot_phase[head_program.out_slot] = out_phase
            slot_log[head_program.out_slot] = out_log
            layer_routes.append(route.detach().cpu())
            layer_margins.append(_margin(scores).detach().cpu())
        all_head_routes.append(tuple(layer_routes))
        all_head_margins.append(tuple(layer_margins))

        phase = torch.stack(slot_phase, 2)
        logmag = torch.stack(slot_log, 2)
        lookup_program = layer_program.lookup
        query_phase = wrap(
            phase[:, :, lookup_program.left_slot]
            + lookup_program.coefficient * phase[:, :, lookup_program.right_slot]
            + ste(lookup_module.mask_o, T_Q)
        )
        key_phase = ste(lookup_module.key_T, T_Q)
        query_log = logmag[:, :, lookup_program.left_slot]
        gate = nnf.relu(query_log[:, :, None] + ste(lookup_module.key_w, L_Q)[None, None])
        lookup_scores = -(circ(query_phase[:, :, None] - key_phase[None, None]) * gate).mean(-1)
        lookup_index = lookup_scores.argmax(-1)
        value_phase = ste(lookup_module.val_T, T_Q)
        value_log = ste(lookup_module.val_L, L_Q)
        hard_phase = value_phase[lookup_index]
        hard_log = value_log[lookup_index]
        hard_key = key_phase[lookup_index]
        out_phase = wrap(hard_phase + lookup_program.carry * signed(query_phase - hard_key))
        slot_phase[lookup_program.out_slot] = ste(out_phase, T_Q)
        slot_log[lookup_program.out_slot] = hard_log
        all_lookup_indices.append(lookup_index.detach().cpu())
        all_lookup_margins.append(_margin(lookup_scores).detach().cpu())

    phase = torch.stack(slot_phase, 2)
    logmag = torch.stack(slot_log, 2)
    logits = ste(model.ro_bias, L_Q)[None, None]
    for slot, mask, weight in zip(
        program.readout_slots,
        (model.ro_mask_a, model.ro_mask_b),
        (model.ro_w_a, model.ro_w_b),
        strict=True,
    ):
        query_phase = wrap(phase[:, :, slot] + ste(mask, T_Q))
        distance = circ(query_phase[:, :, None] - emb_phase[None, None])
        gate = nnf.relu(logmag[:, :, slot, None] + emb_log[None, None] + ste(weight, L_Q))
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


def _phase_codes(value: torch.Tensor) -> np.ndarray:
    ring = int(T_Q)
    return np.remainder(
        np.rint(value.detach().cpu().double().numpy() * ring).astype(np.int64),
        ring,
    ).astype(np.int64)


def compile_phase_affine_trace(
    model: SpectralNativeV3,
    ids: torch.Tensor,
    program: CompiledV3Program,
    trace: HardForwardTrace,
    *,
    batch_index: int = 0,
) -> PhaseAffineTrace:
    """Compile one realized route trace into an affine phase-register DAG.

    Coefficients refer only to the input token-content phases in slot 1.  The
    position law and learned table phases live in the constant term.  Channels
    never mix after routing, so each channel has its own ``N``-variable affine
    expression.
    """

    if ids.ndim != 2 or not 0 <= batch_index < ids.shape[0]:
        raise ValueError("ids must be (batch, tokens) and batch_index must be valid")
    tokens = ids.shape[1]
    slots, channels = model.cfg.slots, model.cfg.slot_ch
    ring = int(T_Q)
    coefficients = np.zeros((slots, tokens, channels, tokens), dtype=np.int64)
    constants = np.zeros((slots, tokens, channels), dtype=np.int64)
    native_codes = _phase_codes(trace.slot_phases[batch_index])
    constants[0] = native_codes[:, 0]
    for token in range(tokens):
        coefficients[1, token, :, token] = 1
    base_content = native_codes[:, 1]

    for layer_index, (layer_program, lookup_module) in enumerate(zip(program.layers, model.mlps, strict=True)):
        before_coefficients = coefficients.copy()
        before_constants = constants.copy()
        for head_index, head_program in enumerate(layer_program.heads):
            route = trace.head_routes[layer_index][head_index][batch_index].numpy()
            for token, source in enumerate(route.tolist()):
                coefficients[head_program.out_slot, token] = before_coefficients[
                    head_program.value_slot, source
                ]
                constants[head_program.out_slot, token] = before_constants[head_program.value_slot, source]

        lookup_program = layer_program.lookup
        lookup_index = trace.lookup_indices[layer_index][batch_index].numpy()
        value_codes = _phase_codes(ste(lookup_module.val_T, T_Q))
        key_codes = _phase_codes(ste(lookup_module.key_T, T_Q))
        mask_codes = _phase_codes(ste(lookup_module.mask_o, T_Q))
        for token, key_index in enumerate(lookup_index.tolist()):
            if lookup_program.carry == 0:
                coefficients[lookup_program.out_slot, token] = 0
                constants[lookup_program.out_slot, token] = value_codes[key_index]
                continue
            coefficients[lookup_program.out_slot, token] = (
                coefficients[lookup_program.left_slot, token]
                + lookup_program.coefficient * coefficients[lookup_program.right_slot, token]
            ) % ring
            constants[lookup_program.out_slot, token] = (
                constants[lookup_program.left_slot, token]
                + lookup_program.coefficient * constants[lookup_program.right_slot, token]
                + mask_codes
                + value_codes[key_index]
                - key_codes[key_index]
            ) % ring

    reconstructed = (
        constants + np.einsum("stcn,nc->stc", coefficients, base_content, dtype=np.int64)
    ) % ring
    return PhaseAffineTrace(
        coefficients=coefficients,
        constants=constants,
        reconstructed_codes=reconstructed,
        native_codes=np.transpose(native_codes, (1, 0, 2)),
    )
