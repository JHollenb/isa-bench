"""Terminal-answer learning of a finite program on frozen learned Field routes.

The recurrent executor exposes generic A/B/Stop actions and a one-bit state.
Its opcode-to-action-and-next-bit policy is learned. Address selection uses the
integer V4 routes learned in the earlier recurrent Field experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .recurrent_field import RecurrentFieldGraph


@dataclass(frozen=True)
class ProgramBatch:
    edges: torch.Tensor  # (batch, A/B, node)
    opcodes: torch.Tensor  # (batch, node): A=0, B=1, Flip=2, Stop=3
    starts: torch.Tensor  # (batch,)
    initial_bits: torch.Tensor  # (batch,)
    target_nodes: torch.Tensor  # (batch,)
    target_bits: torch.Tensor  # (batch,)
    truth_nodes: torch.Tensor  # (batch, moves+1), includes start and Stop node
    truth_bits: torch.Tensor  # (batch, moves+1)
    truth_actions: torch.Tensor  # (batch, moves+1), final action=Stop

    @property
    def moves(self) -> int:
        return self.truth_actions.shape[1] - 1


def oracle_step(edges: torch.Tensor, opcodes: torch.Tensor, node: int, bit: int) -> tuple[int, int, int]:
    """Independent generator interpreter: return (next node, next bit, action)."""
    opcode = int(opcodes[node])
    if opcode == 0:
        return int(edges[0, node]), bit, 0
    if opcode == 1:
        return int(edges[1, node]), bit, 1
    if opcode == 2:
        new_bit = 1 - bit
        action = new_bit
        return int(edges[action, node]), new_bit, action
    if opcode == 3:
        return node, bit, 2
    raise ValueError(f"invalid opcode {opcode}")


def oracle_run(
    edges: torch.Tensor, opcodes: torch.Tensor, start: int, bit: int, *, max_steps: int = 16
) -> tuple[int, int, bool, list[int], list[int], list[int]]:
    node = int(start)
    bit = int(bit)
    nodes = [node]
    bits = [bit]
    actions = []
    for _ in range(max_steps):
        node, bit, action = oracle_step(edges, opcodes, node, bit)
        actions.append(action)
        if action == 2:
            return node, bit, True, nodes, bits, actions
        nodes.append(node)
        bits.append(bit)
    return node, bit, False, nodes, bits, actions


def program_batch(width: int, moves: int, count: int, seed: int) -> ProgramBatch:
    """Reject-sample fresh random worlds with the requested stop depth."""
    if width < 8 or moves < 0 or count < 1:
        raise ValueError("width >= 8, moves >= 0, and count >= 1 are required")
    generator = torch.Generator().manual_seed(seed)
    rows = []
    attempts = 0
    max_attempts = max(20_000, count * 20_000)
    while len(rows) < count and attempts < max_attempts:
        attempts += 1
        edges = torch.stack((torch.randperm(width, generator=generator), torch.randperm(width, generator=generator)))
        opcodes = torch.randint(3, (width,), generator=generator)
        stop_nodes = torch.randperm(width, generator=generator)[: max(2, width // 4)]
        opcodes[stop_nodes] = 3
        start = int(torch.randint(width, (1,), generator=generator))
        bit = int(torch.randint(2, (1,), generator=generator))
        final_node, final_bit, halted, nodes, bits, actions = oracle_run(
            edges, opcodes, start, bit, max_steps=moves + 2
        )
        if halted and len(actions) - 1 == moves:
            rows.append((edges, opcodes, start, bit, final_node, final_bit, nodes, bits, actions))
    if len(rows) != count:
        raise RuntimeError(f"sampled {len(rows)}/{count} worlds at width={width}, moves={moves} in {attempts} attempts")
    return ProgramBatch(
        edges=torch.stack([row[0] for row in rows]),
        opcodes=torch.stack([row[1] for row in rows]),
        starts=torch.tensor([row[2] for row in rows]),
        initial_bits=torch.tensor([row[3] for row in rows]),
        target_nodes=torch.tensor([row[4] for row in rows]),
        target_bits=torch.tensor([row[5] for row in rows]),
        truth_nodes=torch.tensor([row[6] for row in rows]),
        truth_bits=torch.tensor([row[7] for row in rows]),
        truth_actions=torch.tensor([row[8] for row in rows]),
    )


class RecurrentFieldProgram(nn.Module):
    """Frozen two-axis Field reader plus a learned finite-state transition."""

    def __init__(
        self, route_deltas: tuple[tuple[float, float], tuple[float, float]],
        *, address_mode: str = "field2d",
    ) -> None:
        super().__init__()
        if address_mode not in {"field2d", "flat_width_aware"}:
            raise ValueError("address_mode must be field2d or flat_width_aware")
        self.address_mode = address_mode
        self.router = RecurrentFieldGraph()
        with torch.no_grad():
            for head, values in zip(self.router.heads, route_deltas, strict=True):
                head.delta.copy_(torch.tensor(values))
        self.router.requires_grad_(False)
        # opcode x incoming bit x action(A/B/Stop) x outgoing bit
        self.transition_logits = nn.Parameter(torch.randn(4, 2, 3, 2) * 0.02)

    @property
    def transition_probabilities(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits.reshape(4, 2, 6), dim=-1).reshape(4, 2, 3, 2)

    @torch.no_grad()
    def route_accuracy(self, width: int) -> dict[str, float]:
        expected = torch.stack((torch.arange(width), torch.arange(width) + width))
        floating = self._routes(width, integer=False)
        integer = self._routes(width, integer=True)
        return {"float": float((floating == expected).float().mean()),
                "integer": float((integer == expected).float().mean()),
                "agreement": float((floating == integer).float().mean())}

    def _routes(self, width: int, *, integer: bool) -> torch.Tensor:
        if self.address_mode == "flat_width_aware":
            # Query at node+2*width, scalar offsets 2*width and width.
            query = torch.arange(width) + 2 * width
            return torch.stack((query - 2 * width, query - width))
        return self.router.route_indices(width, integer=integer)

    def addressed_records(self, batch: ProgramBatch, *, integer: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Field-addressed A/B successors and opcode for every possible node."""
        count, _, width = batch.edges.shape
        routes = self._routes(width, integer=integer)
        if routes.shape != (2, width) or bool(((routes < 0) | (routes >= 2 * width)).any()):
            raise ValueError("frozen Field route selected a non-record address")
        edge_values = batch.edges.reshape(count, 2 * width)
        opcode_values = torch.cat((batch.opcodes, torch.zeros_like(batch.opcodes)), dim=1)
        return edge_values[:, routes[0]], edge_values[:, routes[1]], opcode_values[:, routes[0]]

    def terminal_probabilities(self, batch: ProgramBatch, *, budget: int = 16) -> torch.Tensor:
        """Soft execution used only for terminal-answer gradient training."""
        count, _, width = batch.edges.shape
        edge_a, edge_b, opcodes = self.addressed_records(batch)
        policy = self.transition_probabilities[opcodes]  # batch,node,in-bit,action,out-bit
        active = nnf.one_hot(batch.starts, width).to(policy.dtype)[:, :, None] * nnf.one_hot(
            batch.initial_bits, 2
        ).to(policy.dtype)[:, None, :]
        halted = torch.zeros_like(active)
        for _ in range(budget):
            new_active = torch.zeros_like(active)
            for action, target in ((0, edge_a), (1, edge_b)):
                mass = (active[:, :, :, None] * policy[:, :, :, action, :]).sum(dim=2)
                new_active.scatter_add_(1, target[:, :, None].expand(-1, -1, 2), mass)
            stopped = (active[:, :, :, None] * policy[:, :, :, 2, :]).sum(dim=2)
            halted = halted + stopped
            active = new_active
        return halted

    def loss(self, batch: ProgramBatch, *, budget: int = 16) -> torch.Tensor:
        terminal = self.terminal_probabilities(batch, budget=budget)
        rows = torch.arange(len(batch.starts))
        target = terminal[rows, batch.target_nodes, batch.target_bits]
        return -torch.log(target.clamp_min(1e-12)).mean()

    @torch.no_grad()
    def rollout(
        self, batch: ProgramBatch, *, budget: int = 16,
        patch_before_step: int | None = None,
        patch_nodes: torch.Tensor | None = None,
        patch_bits: torch.Tensor | None = None,
        write_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Hard native consumer; every next read uses the model's own state."""
        count, _, width = batch.edges.shape
        edge_a, edge_b, opcodes = self.addressed_records(batch)
        choices = self.transition_logits.reshape(4, 2, 6).argmax(-1)
        current = batch.starts.clone()
        bit = batch.initial_bits.clone()
        halted = torch.zeros(count, dtype=torch.bool)
        stop_step = torch.full((count,), -1, dtype=torch.long)
        nodes_trace = [current.clone()]
        bits_trace = [bit.clone()]
        actions_trace = []
        opcode_trace = []
        source_trace = []
        rows = torch.arange(count)
        routes = self._routes(width, integer=True)
        for step in range(budget):
            if patch_before_step is not None and step == patch_before_step:
                if patch_nodes is None or patch_bits is None or patch_nodes.shape != current.shape or patch_bits.shape != bit.shape:
                    raise ValueError("patch_nodes and patch_bits must match batch shape")
                current = torch.where(halted, current, patch_nodes)
                bit = torch.where(halted, bit, patch_bits)
                nodes_trace[-1] = current.clone()
                bits_trace[-1] = bit.clone()
            opcode = opcodes[rows, current]
            decision = choices[opcode, bit]
            action = decision // 2
            new_bit = decision % 2
            action = torch.where(halted, torch.full_like(action, 2), action)
            new_bit = torch.where(halted, bit, new_bit)
            selected_source = torch.where(action == 1, routes[1, current], routes[0, current])
            next_a = edge_a[rows, current]
            next_b = edge_b[rows, current]
            next_node = torch.where(action == 0, next_a, torch.where(action == 1, next_b, current))
            if not write_enabled:
                next_node = current
                new_bit = bit
            just_stopped = (action == 2) & ~halted
            stop_step = torch.where(just_stopped, torch.full_like(stop_step, step), stop_step)
            halted = halted | just_stopped
            current = next_node
            bit = new_bit
            actions_trace.append(action.clone())
            opcode_trace.append(opcode.clone())
            source_trace.append(selected_source.clone())
            nodes_trace.append(current.clone())
            bits_trace.append(bit.clone())
        return {"node": current, "bit": bit, "halted": halted, "stop_step": stop_step,
                "nodes": torch.stack(nodes_trace, dim=1), "bits": torch.stack(bits_trace, dim=1),
                "actions": torch.stack(actions_trace, dim=1),
                "opcodes": torch.stack(opcode_trace, dim=1),
                "sources": torch.stack(source_trace, dim=1)}
