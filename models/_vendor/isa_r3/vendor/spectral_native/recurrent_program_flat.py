"""Width-conditioned flat spectral comparator for the recurrent Field program.

The flat source index is node + row * width. Each learned route has two
scalars: a width coefficient and an intercept. Its route law therefore has
the same four trainable scalars as the two two-axis Field displacements.
"""

from __future__ import annotations

import torch
from torch import nn

from .field_language import _integer_routes
from .model import T_Q
from .model_v4 import field_cross_position_scores, field_phase_turns, orthogonal_axis_frequencies
from .recurrent_program_joint import JointRecurrentFieldProgram


class WidthConditionedFlatRouter(nn.Module):
    def __init__(self, *, channels: int = 18) -> None:
        super().__init__()
        self.coefficients = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
        self.register_buffer("omega", orthogonal_axis_frequencies(1, channels))
        self.register_buffer("weights", torch.ones(channels))

    def _coordinates(self, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        node = torch.arange(width)
        records = torch.arange(2 * width)
        queries = []
        keys = []
        for n in node:
            query = n + 2 * width
            queries.append(torch.cat((records, query.reshape(1)))[:, None])
            keys.append(torch.cat((records, (5 * width + n).reshape(1)))[:, None])
        return torch.stack(keys), torch.stack(queries)

    def _deltas(self, width: int) -> torch.Tensor:
        return (self.coefficients[:, 0] * width + self.coefficients[:, 1])[:, None]

    def _omega(self, width: int) -> torch.Tensor:
        return self.omega

    def route_scores(self, width: int) -> torch.Tensor:
        key, query = self._coordinates(width)
        omega = self._omega(width)
        key_phase = field_phase_turns(key, omega)
        query_phase = field_phase_turns(query, omega)
        return torch.stack([
            field_cross_position_scores(query_phase, key_phase, omega,
                                        self._deltas(width)[head], self.weights)[:, -1, : 2 * width]
            for head in range(2)
        ])

    @torch.no_grad()
    def route_indices(self, width: int, *, integer: bool) -> torch.Tensor:
        if not integer:
            return self.route_scores(width).argmax(-1)
        key, query = self._coordinates(width)
        return _integer_routes(key_coords=key, query_coords=query,
                               omega=self._omega(width), delta=self._deltas(width))[:, :, -1].permute(1, 0)


class WidthScaledFlatRouter(WidthConditionedFlatRouter):
    """One flat coordinate with a width-dependent, grid-snapped phase law."""

    def _omega(self, width: int) -> torch.Tensor:
        return torch.round(self.omega / width * T_Q) / T_Q


class WidthScaledHalfStepRouter(WidthScaledFlatRouter):
    """Same phase law with a declared half-row-short route initialization."""

    def __init__(self, *, channels: int = 18) -> None:
        super().__init__(channels=channels)
        with torch.no_grad():
            self.coefficients.copy_(torch.tensor([[1.5, 0.0], [0.5, 0.0]]))


class FlatWidthConditionedProgram(JointRecurrentFieldProgram):
    """The same transition and soft recurrence with a learned flat route law."""

    def __init__(self) -> None:
        super().__init__()
        self.router = WidthConditionedFlatRouter()


class FlatWidthScaledProgram(JointRecurrentFieldProgram):
    """Same parameter budget and native program with width-scaled phases."""

    def __init__(self) -> None:
        super().__init__()
        self.router = WidthScaledFlatRouter()


class FlatWidthScaledHalfStepProgram(JointRecurrentFieldProgram):
    """Width-scaled flat phase program initialized half a row short."""

    def __init__(self) -> None:
        super().__init__()
        self.router = WidthScaledHalfStepRouter()
