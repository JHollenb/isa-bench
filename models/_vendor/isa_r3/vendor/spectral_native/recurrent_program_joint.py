"""Joint terminal-answer training of spectral routes and a recurrent policy."""

from __future__ import annotations

import torch
import torch.nn.functional as nnf

from .recurrent_program import ProgramBatch, RecurrentFieldProgram


class JointRecurrentFieldProgram(RecurrentFieldProgram):
    """Differentiate through soft V4 reads, then execute hard integer routes."""

    def __init__(self) -> None:
        super().__init__(((0.0, 1.0), (0.0, 0.0)))
        self.router.requires_grad_(True)

    def terminal_probabilities(self, batch: ProgramBatch, *, budget: int = 16) -> torch.Tensor:
        count, _, width = batch.edges.shape
        route_prob = torch.softmax(48.0 * self.router.route_scores(width), dim=-1)
        edge_values = nnf.one_hot(batch.edges.reshape(count, 2 * width), width).to(route_prob.dtype)
        opcode_values = torch.cat((nnf.one_hot(batch.opcodes, 4).to(route_prob.dtype),
                                   torch.zeros(count, width, 4, dtype=route_prob.dtype)), dim=1)
        edge_a = torch.einsum("nk,bkv->bnv", route_prob[0], edge_values)
        edge_b = torch.einsum("nk,bkv->bnv", route_prob[1], edge_values)
        opcodes = torch.einsum("nk,bko->bno", route_prob[0], opcode_values)
        policy = torch.einsum("bno,omaj->bnmaj", opcodes, self.transition_probabilities)
        active = nnf.one_hot(batch.starts, width).to(policy.dtype)[:, :, None] * nnf.one_hot(
            batch.initial_bits, 2
        ).to(policy.dtype)[:, None, :]
        halted = torch.zeros_like(active)
        for _ in range(budget):
            mass_a = (active[:, :, :, None] * policy[:, :, :, 0, :]).sum(dim=2)
            mass_b = (active[:, :, :, None] * policy[:, :, :, 1, :]).sum(dim=2)
            new_active = torch.einsum("bnj,bnv->bvj", mass_a, edge_a)
            new_active = new_active + torch.einsum("bnj,bnv->bvj", mass_b, edge_b)
            halted = halted + (active[:, :, :, None] * policy[:, :, :, 2, :]).sum(dim=2)
            active = new_active
        return halted
