"""Matched learned-attention controller for the per-world rule-reading task.

The world is given as an unordered set of record tokens: edge A / edge B /
opcode records per room and one definition record per symbol, plus a width
token.  Rooms carry identity codes drawn per world from a pool of learned
embeddings (randomised labels), so addressing is by content, never by
absolute room index, and width transfer is possible in principle.

For every executor state (room, bit) a small Transformer (record encoder +
cross-attention state decoder) outputs Stop, a pointer over rooms for the next
room, and the next bit.  Training reuses the Field's soft executor: the
transition table for all states is propagated for a fixed budget and the loss
reads only the halted mass at the terminal (room, bit).  Scoring uses the hard
native rollout, like the Field.  ``gold_loss`` is a teacher-forced trace loss
for the diagnostic expressivity ceiling only.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .recurrent_rule import BEHAVIORS, SYMBOLS, RuleBatch

ROLE_A, ROLE_B, ROLE_OP, ROLE_DEF, ROLE_QUERY, ROLE_WIDTH = range(6)


class _CrossLayer(nn.Module):
    def __init__(self, d: int, heads: int, ff: int) -> None:
        super().__init__()
        self.norm_q = nn.LayerNorm(d)
        self.norm_m = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.norm_f = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        mem = self.norm_m(memory)
        query = query + self.attn(self.norm_q(query), mem, mem, need_weights=False)[0]
        return query + self.ff(self.norm_f(query))


class RuleTransformer(nn.Module):
    def __init__(self, d: int, *, enc_layers: int, dec_layers: int, heads: int, ff_mult: int = 2,
                 id_pool: int = 32, width_scale: float = 16.0) -> None:
        super().__init__()
        self.d = d
        self.id_pool = id_pool
        self.width_scale = width_scale
        self.config = {"d": d, "enc_layers": enc_layers, "dec_layers": dec_layers, "heads": heads,
                       "ff_mult": ff_mult, "id_pool": id_pool, "width_scale": width_scale}
        self.id_emb = nn.Embedding(id_pool, d)
        self.sym_emb = nn.Embedding(SYMBOLS, d)
        self.beh_emb = nn.Embedding(BEHAVIORS, d)
        self.role_emb = nn.Embedding(6, d)
        self.bit_emb = nn.Embedding(2, d)
        self.subject = nn.Linear(d, d, bias=False)
        self.payload = nn.Linear(d, d, bias=False)
        self.width = nn.Linear(1, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff_mult * d, dropout=0.0, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, enc_layers, enable_nested_tensor=False) if enc_layers else None
        self.decoder = nn.ModuleList([_CrossLayer(d, heads, ff_mult * d) for _ in range(dec_layers)])
        self.out_norm = nn.LayerNorm(d)
        self.pointer = nn.Linear(d, d)
        self.bit_head = nn.Linear(d, 2)
        self.stop_head = nn.Linear(d, 1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def assign_ids(self, count: int, width: int, generator: torch.Generator | None = None) -> torch.Tensor:
        if generator is None:
            generator = torch.Generator().manual_seed(0)
        return torch.stack([torch.randperm(self.id_pool, generator=generator)[:width] for _ in range(count)])

    def transition_logits(self, batch: RuleBatch, *, ids: torch.Tensor | None = None,
                          width_override: int | None = None) -> dict[str, torch.Tensor]:
        """Stop / next-room / next-bit logits for every (room, bit) state of every world."""
        count, _, width = batch.edges.shape
        if ids is None:
            ids = self.assign_ids(count, width)
        room = self.id_emb(ids)  # (b, W, d)
        rows = torch.arange(count)[:, None]
        succ_a = room[rows, batch.edges[:, 0]]
        succ_b = room[rows, batch.edges[:, 1]]
        subj_room = self.subject(room)
        role = self.role_emb.weight
        edge_a = role[ROLE_A] + subj_room + self.payload(succ_a)
        edge_b = role[ROLE_B] + subj_room + self.payload(succ_b)
        opcode = role[ROLE_OP] + subj_room + self.payload(self.sym_emb(batch.symbols))
        definition = (role[ROLE_DEF] + self.subject(self.sym_emb.weight)[None]
                      + self.payload(self.beh_emb(batch.defs)))
        shown_width = float(width if width_override is None else width_override)
        width_token = (role[ROLE_WIDTH] + self.width(torch.full((count, 1, 1), shown_width / self.width_scale)))
        memory = torch.cat((edge_a, edge_b, opcode, definition, width_token), 1)
        if self.encoder is not None:
            memory = self.encoder(memory)
        query = (role[ROLE_QUERY] + subj_room)[:, :, None, :] + self.bit_emb.weight[None, None]  # (b, W, 2, d)
        query = query.reshape(count, 2 * width, self.d)
        for layer in self.decoder:
            query = layer(query, memory)
        hidden = self.out_norm(query)
        pointer = torch.einsum("bqd,bmd->bqm", self.pointer(hidden), room) / self.d ** 0.5
        return {"stop": self.stop_head(hidden).reshape(count, width, 2),
                "node": pointer.reshape(count, width, 2, width),
                "bit": self.bit_head(hidden).reshape(count, width, 2, 2)}

    def terminal_probabilities(self, batch: RuleBatch, *, budget: int = 8, ids: torch.Tensor | None = None) -> torch.Tensor:
        count, _, width = batch.edges.shape
        logits = self.transition_logits(batch, ids=ids)
        p_stop = torch.sigmoid(logits["stop"])
        p_node = torch.softmax(logits["node"], -1)
        p_bit = torch.softmax(logits["bit"], -1)
        active = nnf.one_hot(batch.starts, width).float()[:, :, None] * nnf.one_hot(batch.initial_bits, 2).float()[:, None, :]
        halted = torch.zeros_like(active)
        for _ in range(budget):
            halted = halted + active * p_stop
            move = active * (1 - p_stop)
            active = torch.einsum("bnj,bnjm,bnjk->bmk", move, p_node, p_bit)
        return halted

    def loss(self, batch: RuleBatch, *, budget: int = 8, ids: torch.Tensor | None = None) -> torch.Tensor:
        terminal = self.terminal_probabilities(batch, budget=budget, ids=ids)
        rows = torch.arange(len(batch.starts))
        return -torch.log(terminal[rows, batch.target_nodes, batch.target_bits].clamp_min(1e-12)).mean()

    def gold_loss(self, batch: RuleBatch, *, ids: torch.Tensor | None = None) -> torch.Tensor:
        """Teacher-forced per-step trace loss (diagnostic ceiling arm only)."""
        logits = self.transition_logits(batch, ids=ids)
        count = len(batch.starts)
        moves = batch.moves
        rows = torch.arange(count)
        total = torch.zeros(())
        for step in range(moves + 1):
            node = batch.truth_nodes[:, step]
            bit = batch.truth_bits[:, step]
            stop_logit = logits["stop"][rows, node, bit]
            stop_target = torch.full((count,), float(step == moves))
            total = total + nnf.binary_cross_entropy_with_logits(stop_logit, stop_target)
            if step < moves:
                total = total + nnf.cross_entropy(logits["node"][rows, node, bit], batch.truth_nodes[:, step + 1])
                total = total + nnf.cross_entropy(logits["bit"][rows, node, bit], batch.truth_bits[:, step + 1])
        return total / (moves + 1)

    @torch.no_grad()
    def rollout(self, batch: RuleBatch, *, budget: int = 16, width_override: int | None = None,
                write_enabled: bool = True) -> dict[str, torch.Tensor]:
        count, _, width = batch.edges.shape
        logits = self.transition_logits(batch, width_override=width_override)
        stop_table = logits["stop"] > 0
        node_table = logits["node"].argmax(-1)
        bit_table = logits["bit"].argmax(-1)
        rows = torch.arange(count)
        current = batch.starts.clone()
        bit = batch.initial_bits.clone()
        halted = torch.zeros(count, dtype=torch.bool)
        stop_step = torch.full((count,), -1, dtype=torch.long)
        nodes, bits, actions = [current.clone()], [bit.clone()], []
        for step in range(budget):
            stop = stop_table[rows, current, bit] & ~halted
            nxt = node_table[rows, current, bit]
            nbit = bit_table[rows, current, bit]
            action = torch.where(nxt == batch.edges[rows, 0, current], 0, torch.where(nxt == batch.edges[rows, 1, current], 1, 3))
            action = torch.where(stop | halted, 2, action)
            move = ~(stop | halted)
            if not write_enabled:
                move = torch.zeros_like(move)
            stop_step = torch.where(stop, torch.full_like(stop_step, step), stop_step)
            halted = halted | stop
            current = torch.where(move, nxt, current)
            bit = torch.where(move, nbit, bit)
            nodes.append(current.clone())
            bits.append(bit.clone())
            actions.append(action)
        return {"node": current, "bit": bit, "halted": halted, "stop_step": stop_step,
                "invalid_read": torch.zeros(count, dtype=torch.bool),
                "nodes": torch.stack(nodes, 1), "bits": torch.stack(bits, 1), "actions": torch.stack(actions, 1)}
