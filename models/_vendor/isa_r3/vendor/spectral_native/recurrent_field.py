"""Controlled recurrent pointer reasoning on the existing V4 spectral router.

The graph and pointer-update operator are supplied. The two relation routes are
learned from final answers, and hard execution repeatedly consumes its own
predicted node. Phase matching delegates to SpectralFieldPointerLM, including
its integer SPF path; this module only supplies a recurrent consumer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .field_language import SpectralFieldPointerLM


@dataclass(frozen=True)
class GraphBatch:
    edges: torch.Tensor  # (batch, relation=2, node)
    starts: torch.Tensor  # (batch,)
    instructions: torch.Tensor  # (batch, steps)
    targets: torch.Tensor  # (batch,)
    truth_traces: torch.Tensor  # (batch, steps + 1)


def graph_batch(width: int, depth: int, count: int, seed: int) -> GraphBatch:
    if width < 4 or depth < 1 or count < 1:
        raise ValueError("width >= 4, depth >= 1 and count >= 1 are required")
    generator = torch.Generator().manual_seed(seed)
    edges = torch.stack(
        [torch.stack([torch.randperm(width, generator=generator) for _ in range(2)]) for _ in range(count)]
    )
    starts = torch.randint(width, (count,), generator=generator)
    instructions = torch.randint(2, (count, depth), generator=generator)
    current = starts
    trace = [current]
    rows = torch.arange(count)
    for step in range(depth):
        current = edges[rows, instructions[:, step], current]
        trace.append(current)
    return GraphBatch(edges, starts, instructions, current, torch.stack(trace, dim=1))


def record_coordinates(width: int, *, flat: bool = False) -> torch.Tensor:
    """Rows 0/1 store A/B; row 2 is the query register."""
    node = torch.arange(width)
    records = torch.stack(
        [torch.stack((node, torch.full_like(node, row)), dim=1) for row in range(2)]
    ).reshape(2 * width, 2)
    if flat:
        return (records[:, 0] + width * records[:, 1])[:, None]
    return records


class RecurrentFieldGraph(nn.Module):
    """Two learned V4 reads, one shared hard pointer update per reasoning step."""

    def __init__(
        self,
        *,
        flat: bool = False,
        channels: int = 18,
        temperature: float = 48.0,
        collapse_role: bool = False,
        scramble_row: bool = False,
    ) -> None:
        super().__init__()
        self.flat = flat
        self.temperature = temperature
        self.collapse_role = collapse_role
        self.scramble_row = scramble_row
        dims = 1 if flat else 2
        self.heads = nn.ModuleList(
            [SpectralFieldPointerLM(coord_dims=dims, channels=channels, vocab=64) for _ in range(2)]
        )
        with torch.no_grad():
            if flat:
                self.heads[0].delta.fill_(2.0)
                self.heads[1].delta.fill_(1.0)
            else:
                self.heads[0].delta.copy_(torch.tensor([0.0, 1.0]))
                self.heads[1].delta.copy_(torch.tensor([0.0, 0.0]))

    def _coordinate_pair(self, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        records = record_coordinates(width, flat=self.flat)
        if self.scramble_row:
            if self.flat:
                records[width:, 0] = torch.remainder(records[width:, 0] + 1, width) + width
            else:
                records[width:, 0] = torch.remainder(records[width:, 0] + 1, width)
        if self.collapse_role:
            if self.flat:
                raise ValueError("role collapse is only defined for 2D addresses")
            records[:, 1] = 0
        queries = []
        keys = []
        for node in range(width):
            query = torch.tensor([node + 2 * width] if self.flat else [node, 2])
            if self.collapse_role:
                query[1] = 0
            # The final key is only a shape-compatible sentinel. It is excluded
            # from the train surrogate and hard route selection below.
            sentinel = torch.tensor([5 * width + node] if self.flat else [width + 20 + node, 3])
            queries.append(torch.cat((records, query[None]), dim=0))
            keys.append(torch.cat((records, sentinel[None]), dim=0))
        return torch.stack(keys), torch.stack(queries)

    def route_scores(self, width: int) -> torch.Tensor:
        """(relation, current-node, record-index) scores from V4 phase matching."""
        key, query = self._coordinate_pair(width)
        return torch.stack([head.route_scores(key, query)[:, -1, : 2 * width] for head in self.heads])

    def transitions(self, edges: torch.Tensor) -> torch.Tensor:
        """Soft relation transition matrices for final-answer-only training."""
        batch, relations, width = edges.shape
        if relations != 2:
            raise ValueError("two relations are required")
        route_prob = torch.softmax(self.temperature * self.route_scores(width), dim=-1)
        values = nnf.one_hot(edges.reshape(batch, 2 * width), width).to(route_prob.dtype)
        return torch.einsum("rnk,bkv->brnv", route_prob, values)

    def answer_probabilities(self, batch: GraphBatch) -> torch.Tensor:
        count, _, width = batch.edges.shape
        trans = self.transitions(batch.edges)
        state = nnf.one_hot(batch.starts, width).to(trans.dtype)
        rows = torch.arange(count)
        for step in range(batch.instructions.shape[1]):
            selected = trans[rows, batch.instructions[:, step]]
            state = torch.bmm(state[:, None, :], selected)[:, 0]
        return state

    def loss(self, batch: GraphBatch) -> torch.Tensor:
        probability = self.answer_probabilities(batch)
        target = probability.gather(1, batch.targets[:, None]).squeeze(1)
        return -torch.log(target.clamp_min(1e-12)).mean()

    @torch.no_grad()
    def route_indices(self, width: int, *, integer: bool) -> torch.Tensor:
        if not integer:
            return self.route_scores(width).argmax(-1)
        key, query = self._coordinate_pair(width)
        routes = torch.stack([head.integer_routes(key, query)[:, -1] for head in self.heads])
        # The sentinel is not an eligible knowledge record. Report it as an
        # invalid route instead of silently wrapping it into an edge.
        return routes

    @torch.no_grad()
    def rollout(
        self,
        batch: GraphBatch,
        *,
        integer: bool = True,
        max_steps: int | None = None,
        patch_step: int | None = None,
        patch_nodes: torch.Tensor | None = None,
        write_enabled: bool = True,
        swap_relations: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return final nodes, every intermediate node, and every selected record."""
        count, _, width = batch.edges.shape
        routes = self.route_indices(width, integer=integer)
        current = batch.starts.clone()
        trace = [current]
        source_trace = []
        edge_values = batch.edges.reshape(count, 2 * width)
        rows = torch.arange(count)
        steps = batch.instructions.shape[1] if max_steps is None else min(max_steps, batch.instructions.shape[1])
        for step in range(steps):
            relation = batch.instructions[:, step]
            if swap_relations:
                relation = 1 - relation
            selected = routes[relation, current]
            source_trace.append(selected)
            safe = selected.clamp(0, 2 * width - 1)
            next_node = edge_values[rows, safe]
            if not write_enabled:
                next_node = current
            if patch_step is not None and step == patch_step:
                if patch_nodes is None or patch_nodes.shape != current.shape:
                    raise ValueError("patch_nodes must match batch starts")
                next_node = patch_nodes
            current = next_node
            trace.append(current)
        return current, torch.stack(trace, dim=1), torch.stack(source_trace, dim=1)
