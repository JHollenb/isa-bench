"""Minimal autoregressive language consumers for a spectral address field.

The model is intentionally a pointer language model: a hard spectral route
selects one earlier token and emits that token as the next-token prediction.
It isolates whether multidimensional address geometry can support autonomous
language continuation before the full register/lookup stack is trained.

The deployed path is selection only.  The vocabulary log-sum-exp below is an
explicit differentiable training surrogate; integer hard inference uses phase
codes and never decodes a complex value.

``SpectralFieldAddLM`` extends the pointer with two learned vector routes and
an exact phase-binding operator.  It retrieves two prior token codes, adds
their phases on the SPF ring, and emits the newly synthesized payload symbol.
Its training surrogate marginalizes over source pairs; its deployed path is
two hard integer routes followed by integer phase addition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .integer_v4 import field_phase_codes
from .model import T_Q
from .model_v4 import (
    field_cross_position_scores,
    field_phase_turns,
    orthogonal_axis_frequencies,
)


def _causal_mask(scores: torch.Tensor) -> torch.Tensor:
    """Allow every observed token, including the current token."""

    tokens = scores.shape[-1]
    index = torch.arange(tokens, device=scores.device)
    allowed = index[None, :] <= index[:, None]
    return scores.masked_fill(~allowed, float("-inf"))


def _integer_routes(
    *,
    key_coords: torch.Tensor,
    query_coords: torch.Tensor,
    omega: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    """Route one or more vector queries using only integer phase codes."""

    ring = int(T_Q)
    omega_codes = torch.round(omega * ring).to(torch.int64)
    key_codes = field_phase_codes(key_coords, omega_codes, ring)
    query_codes = field_phase_codes(query_coords, omega_codes, ring)
    rounded_delta = torch.round(delta).to(torch.int64)
    if rounded_delta.ndim == 1:
        rounded_delta = rounded_delta[None]
    shift_codes = torch.einsum("hd,dc->hc", rounded_delta, omega_codes)
    shifted_query = torch.remainder(
        query_codes[:, None] - shift_codes[None, :, None],
        ring,
    )
    difference = torch.remainder(
        shifted_query[:, :, :, None] - key_codes[:, None, None],
        ring,
    )
    distance = torch.minimum(difference, ring - difference)
    scores = -distance.sum(-1, dtype=torch.int64)
    tokens = scores.shape[-1]
    index = torch.arange(tokens, device=scores.device)
    allowed = index[None, :] <= index[:, None]
    return scores.masked_fill(
        ~allowed,
        torch.iinfo(scores.dtype).min,
    ).argmax(-1)


class SpectralFieldPointerLM(nn.Module):
    """One learned vector route closed through a next-token copy consumer."""

    def __init__(
        self,
        *,
        coord_dims: int,
        channels: int = 18,
        vocab: int = 64,
        backward_scale: float = 32.0,
    ) -> None:
        super().__init__()
        if vocab < 2:
            raise ValueError("vocab must be at least two")
        self.coord_dims = int(coord_dims)
        self.channels = int(channels)
        self.vocab = int(vocab)
        self.backward_scale = float(backward_scale)
        self.delta = nn.Parameter(torch.randn(coord_dims) * 0.1)
        self.register_buffer(
            "omega",
            orthogonal_axis_frequencies(coord_dims, channels),
        )
        self.register_buffer("weights", torch.ones(channels))

    def route_scores(
        self,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        if key_coords.shape != query_coords.shape:
            raise ValueError("key and query coordinates must have the same shape")
        key_phase = field_phase_turns(key_coords, self.omega)
        query_phase = field_phase_turns(query_coords, self.omega)
        return _causal_mask(
            field_cross_position_scores(
                query_phase,
                key_phase,
                self.omega,
                self.delta,
                self.weights,
            )
        )

    def forward(
        self,
        ids: torch.Tensor,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Return differentiable next-token logits over tokens present in context."""

        if ids.ndim != 2 or ids.shape != key_coords.shape[:2]:
            raise ValueError("ids and coordinate prefix shapes do not match")
        if ids.min() < 0 or ids.max() >= self.vocab:
            raise ValueError("token id is outside the configured vocabulary")
        scores = self.route_scores(key_coords, query_coords) * self.backward_scale
        token_membership = nnf.one_hot(ids, self.vocab).to(torch.bool)
        per_token_source_scores = scores[:, :, :, None].masked_fill(
            ~token_membership[:, None, :, :],
            float("-inf"),
        )
        return torch.logsumexp(per_token_source_scores, dim=2)

    @torch.no_grad()
    def hard_routes(
        self,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        return self.route_scores(key_coords, query_coords).argmax(-1)

    @torch.no_grad()
    def hard_predictions(
        self,
        ids: torch.Tensor,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        route = self.hard_routes(key_coords, query_coords)
        return torch.gather(ids, 1, route)

    @torch.no_grad()
    def integer_routes(
        self,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate the learned route entirely in integer phase codes."""

        return _integer_routes(
            key_coords=key_coords,
            query_coords=query_coords,
            omega=self.omega,
            delta=self.delta,
        )[:, 0]


class SpectralFieldAddLM(nn.Module):
    """Two learned vector routes closed through exact token-phase binding."""

    route_count = 2

    def __init__(
        self,
        *,
        coord_dims: int,
        channels: int = 18,
        vocab: int = 64,
        payload_vocab: int = 16,
        backward_scale: float = 32.0,
    ) -> None:
        super().__init__()
        if payload_vocab < 2 or int(T_Q) % payload_vocab:
            raise ValueError("payload_vocab must divide the SPF phase ring")
        if vocab <= payload_vocab:
            raise ValueError("vocab must leave room for non-payload control tokens")
        self.coord_dims = int(coord_dims)
        self.channels = int(channels)
        self.vocab = int(vocab)
        self.payload_vocab = int(payload_vocab)
        self.backward_scale = float(backward_scale)
        self.delta = nn.Parameter(torch.randn(self.route_count, coord_dims) * 0.1)
        self.register_buffer(
            "omega",
            orthogonal_axis_frequencies(coord_dims, channels),
        )
        self.register_buffer("weights", torch.ones(self.route_count, channels))

    def route_scores(
        self,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``(batch, route, query, key)`` circular-match scores."""

        if key_coords.shape != query_coords.shape:
            raise ValueError("key and query coordinates must have the same shape")
        key_phase = field_phase_turns(key_coords, self.omega)
        query_phase = field_phase_turns(query_coords, self.omega)
        scores = torch.stack(
            [
                field_cross_position_scores(
                    query_phase,
                    key_phase,
                    self.omega,
                    self.delta[route],
                    self.weights[route],
                )
                for route in range(self.route_count)
            ],
            dim=1,
        )
        return _causal_mask(scores)

    def forward(
        self,
        ids: torch.Tensor,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Return differentiable logits induced by all routed source pairs."""

        if ids.ndim != 2 or ids.shape != key_coords.shape[:2]:
            raise ValueError("ids and coordinate prefix shapes do not match")
        if ids.min() < 0 or ids.max() >= self.vocab:
            raise ValueError("token id is outside the configured vocabulary")
        scores = self.route_scores(key_coords, query_coords) * self.backward_scale
        pair_scores = scores[:, 0, :, :, None] + scores[:, 1, :, None, :]
        surrogate_floor = torch.full_like(pair_scores, -1.0e4)
        pair_scores = torch.where(
            torch.isfinite(pair_scores),
            pair_scores,
            surrogate_floor,
        )
        payload_ids = torch.remainder(ids, self.payload_vocab)
        pair_values = torch.remainder(
            payload_ids[:, :, None] + payload_ids[:, None, :],
            self.payload_vocab,
        )
        membership = nnf.one_hot(pair_values, self.payload_vocab).to(torch.bool)
        per_output_pair_scores = pair_scores[:, :, :, :, None].masked_fill(
            ~membership[:, None],
            -1.0e4,
        )
        return torch.logsumexp(
            per_output_pair_scores.flatten(2, 3),
            dim=2,
        )

    @torch.no_grad()
    def hard_routes(
        self,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        return self.route_scores(key_coords, query_coords).argmax(-1)

    def _bind_selected(self, selected: torch.Tensor) -> torch.Tensor:
        """Bind token phases exactly and decode the resulting payload symbol."""

        ring = int(T_Q)
        phase_step = ring // self.payload_vocab
        phase_codes = torch.remainder(selected, self.payload_vocab) * phase_step
        bound_codes = torch.remainder(phase_codes.sum(dim=1), ring)
        return torch.div(bound_codes, phase_step, rounding_mode="floor").to(torch.long)

    @torch.no_grad()
    def predictions_from_routes(self, ids: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        batch, route_count, queries = routes.shape
        if route_count != self.route_count or ids.shape[0] != batch:
            raise ValueError("route tensor does not match the binary binding model")
        expanded_ids = ids[:, None, None, :].expand(
            batch,
            route_count,
            queries,
            ids.shape[1],
        )
        selected = torch.gather(expanded_ids, 3, routes[:, :, :, None]).squeeze(-1)
        return self._bind_selected(selected)

    @torch.no_grad()
    def hard_predictions(
        self,
        ids: torch.Tensor,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictions_from_routes(
            ids,
            self.hard_routes(key_coords, query_coords),
        )

    @torch.no_grad()
    def integer_routes(
        self,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        return _integer_routes(
            key_coords=key_coords,
            query_coords=query_coords,
            omega=self.omega,
            delta=self.delta,
        )

    @torch.no_grad()
    def integer_predictions(
        self,
        ids: torch.Tensor,
        key_coords: torch.Tensor,
        query_coords: torch.Tensor,
    ) -> torch.Tensor:
        return self.predictions_from_routes(
            ids,
            self.integer_routes(key_coords, query_coords),
        )
