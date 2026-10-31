"""GRU reader for the per-world rule task, on the Transformer's exact inputs.

The record tokens, room-identity codes, width token, state queries, heads,
soft executor, gold loss and hard rollout are those of ``RuleTransformer``;
only the reader differs.  For every executor state (room, bit) a GRU scans the
world's record sequence (edge A records, edge B records, opcode records,
definition records, width token) with the state query concatenated to every
input, and its final hidden state feeds the Stop / room-pointer / next-bit
heads.  The executor state therefore stays (room, bit), so the exact
state-marginal soft executor and terminal loss are shared with the Field and
the Transformer.  A controller that carries a hidden vector across executor
steps would need a different training surrogate and is not this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .recurrent_rule import BEHAVIORS, SYMBOLS, RuleBatch
from .recurrent_rule_transformer import (
    ROLE_A, ROLE_B, ROLE_DEF, ROLE_OP, ROLE_QUERY, ROLE_WIDTH, RuleTransformer,
)


class RuleGRUReader(nn.Module):
    def __init__(self, d: int, hidden: int, *, layers: int = 1, id_pool: int = 32, width_scale: float = 16.0) -> None:
        super().__init__()
        self.d = d
        self.hidden = hidden
        self.id_pool = id_pool
        self.width_scale = width_scale
        self.config = {"kind": "gru", "d": d, "hidden": hidden, "layers": layers, "id_pool": id_pool,
                       "width_scale": width_scale}
        self.id_emb = nn.Embedding(id_pool, d)
        self.sym_emb = nn.Embedding(SYMBOLS, d)
        self.beh_emb = nn.Embedding(BEHAVIORS, d)
        self.role_emb = nn.Embedding(6, d)
        self.bit_emb = nn.Embedding(2, d)
        self.subject = nn.Linear(d, d, bias=False)
        self.payload = nn.Linear(d, d, bias=False)
        self.width = nn.Linear(1, d)
        self.gru = nn.GRU(2 * d, hidden, num_layers=layers, batch_first=True)
        self.out_norm = nn.LayerNorm(hidden)
        self.pointer = nn.Linear(hidden, d)
        self.bit_head = nn.Linear(hidden, 2)
        self.stop_head = nn.Linear(hidden, 1)

    parameter_count = RuleTransformer.parameter_count
    assign_ids = RuleTransformer.assign_ids
    terminal_probabilities = RuleTransformer.terminal_probabilities
    loss = RuleTransformer.loss
    gold_loss = RuleTransformer.gold_loss
    rollout = RuleTransformer.rollout

    def transition_logits(self, batch: RuleBatch, *, ids: torch.Tensor | None = None,
                          width_override: int | None = None) -> dict[str, torch.Tensor]:
        count, _, width = batch.edges.shape
        if ids is None:
            ids = self.assign_ids(count, width)
        room = self.id_emb(ids)
        rows = torch.arange(count)[:, None]
        subj_room = self.subject(room)
        role = self.role_emb.weight
        edge_a = role[ROLE_A] + subj_room + self.payload(room[rows, batch.edges[:, 0]])
        edge_b = role[ROLE_B] + subj_room + self.payload(room[rows, batch.edges[:, 1]])
        opcode = role[ROLE_OP] + subj_room + self.payload(self.sym_emb(batch.symbols))
        definition = (role[ROLE_DEF] + self.subject(self.sym_emb.weight)[None]
                      + self.payload(self.beh_emb(batch.defs)))
        shown_width = float(width if width_override is None else width_override)
        width_token = role[ROLE_WIDTH] + self.width(torch.full((count, 1, 1), shown_width / self.width_scale))
        records = torch.cat((edge_a, edge_b, opcode, definition, width_token), 1)  # (b, T, d)
        query = ((role[ROLE_QUERY] + subj_room)[:, :, None, :] + self.bit_emb.weight[None, None]).reshape(
            count, 2 * width, self.d)
        steps = records.shape[1]
        inputs = torch.cat((records[:, None].expand(-1, 2 * width, -1, -1),
                            query[:, :, None].expand(-1, -1, steps, -1)), -1).reshape(count * 2 * width, steps, 2 * self.d)
        _, final = self.gru(inputs)
        hidden = self.out_norm(final[-1]).reshape(count, 2 * width, self.hidden)
        pointer = torch.einsum("bqd,bmd->bqm", self.pointer(hidden), room) / self.d ** 0.5
        return {"stop": self.stop_head(hidden).reshape(count, width, 2),
                "node": pointer.reshape(count, width, 2, width),
                "bit": self.bit_head(hidden).reshape(count, width, 2, 2)}
