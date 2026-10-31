"""Per-world opcode meanings read in context through a learned Field route.

Each world binds four local opcode *symbols* to the global behaviors
A / B / Flip / Stop with its own permutation, and states that binding as four
definition records on sheet row 3.  Node records keep their two successor
edges on rows 0 and 1; the row-0 record also carries the node's opcode symbol,
never its behavior.  To act, the model must read a node's symbol through its
A route, read that symbol's definition through a third learned V4 route, and
only then apply its behavior-indexed transition.  A symbol-indexed baseline
and an oracle-definition ceiling share the same executor.

Supplied: record roles and rows, the A/B/Stop action vocabulary, the one-bit
register, the max-step executor and halt absorption, and the global behavior
code alphabet.  Learned from terminal answers: the three route displacements
and the behavior (or symbol) transition table.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as nnf

from .field_language import SpectralFieldPointerLM
from .recurrent_program import oracle_run

SYMBOLS = 4
BEHAVIORS = 4  # A=0, B=1, Flip=2, Stop=3
NONE_SYMBOL = SYMBOLS  # value read from a record that carries no symbol
NONE_BEHAVIOR = BEHAVIORS  # value read from a record that carries no definition
DEF_ROW = 3
QUERY_ROW = 2
PERMUTATIONS = tuple(itertools.permutations(range(BEHAVIORS)))
# Held-out bindings: symbol 0 means Stop.  That symbol/behavior pair never
# occurs in a training world, so a symbol-indexed table cannot have learned it.
HELDOUT_PERMS = tuple(p for p in PERMUTATIONS if p[0] == 3)
TRAIN_PERMS = tuple(p for p in PERMUTATIONS if p[0] != 3)
ORACLE_POLICY = [[0, 1], [2, 3], [3, 0], [4, 5]]  # behavior x bit -> action*2+out_bit
MODES = ("rule", "symbol", "oracle")


@dataclass(frozen=True)
class RuleBatch:
    edges: torch.Tensor  # (batch, A/B, node)
    symbols: torch.Tensor  # (batch, node) local opcode symbol
    defs: torch.Tensor  # (batch, symbol) behavior bound to each symbol
    starts: torch.Tensor
    initial_bits: torch.Tensor
    target_nodes: torch.Tensor
    target_bits: torch.Tensor
    truth_nodes: torch.Tensor
    truth_bits: torch.Tensor
    truth_actions: torch.Tensor

    @property
    def moves(self) -> int:
        return self.truth_actions.shape[1] - 1

    @property
    def behaviors(self) -> torch.Tensor:
        return self.defs.gather(1, self.symbols)


def select(batch: RuleBatch, index: int) -> RuleBatch:
    return RuleBatch(**{f.name: getattr(batch, f.name)[index : index + 1] for f in fields(RuleBatch)})


def interpret(edges: torch.Tensor, symbols: torch.Tensor, defs: torch.Tensor, start: int, bit: int,
              *, max_steps: int = 16) -> tuple[int, int, bool, list[int], list[int], list[int]]:
    """Independent interpreter for one world: resolve symbols through defs, then run."""
    return oracle_run(edges, defs[symbols], start, bit, max_steps=max_steps)


def rule_batch(width: int, moves: int, count: int, seed: int, perms: tuple[tuple[int, ...], ...]) -> RuleBatch:
    """Fresh random worlds with the requested stop depth and a per-world binding."""
    if width < 8 or moves < 0 or count < 1 or not perms:
        raise ValueError("width >= 8, moves >= 0, count >= 1 and a permutation set are required")
    generator = torch.Generator().manual_seed(seed)
    perm_table = torch.tensor(perms)
    rows = []
    attempts = 0
    while len(rows) < count and attempts < max(20_000, count * 20_000):
        attempts += 1
        edges = torch.stack((torch.randperm(width, generator=generator), torch.randperm(width, generator=generator)))
        behaviors = torch.randint(3, (width,), generator=generator)
        stop_nodes = torch.randperm(width, generator=generator)[: max(2, width // 4)]
        behaviors[stop_nodes] = 3
        start = int(torch.randint(width, (1,), generator=generator))
        bit = int(torch.randint(2, (1,), generator=generator))
        defs = perm_table[int(torch.randint(len(perms), (1,), generator=generator))]
        inverse = torch.empty_like(defs)
        inverse[defs] = torch.arange(BEHAVIORS)
        symbols = inverse[behaviors]
        final_node, final_bit, halted, nodes, bits, actions = interpret(
            edges, symbols, defs, start, bit, max_steps=moves + 2)
        if halted and len(actions) - 1 == moves:
            rows.append((edges, symbols, defs.clone(), start, bit, final_node, final_bit, nodes, bits, actions))
    if len(rows) != count:
        raise RuntimeError(f"sampled {len(rows)}/{count} worlds at width={width}, moves={moves}")
    return RuleBatch(
        edges=torch.stack([r[0] for r in rows]), symbols=torch.stack([r[1] for r in rows]),
        defs=torch.stack([r[2] for r in rows]), starts=torch.tensor([r[3] for r in rows]),
        initial_bits=torch.tensor([r[4] for r in rows]), target_nodes=torch.tensor([r[5] for r in rows]),
        target_bits=torch.tensor([r[6] for r in rows]), truth_nodes=torch.tensor([r[7] for r in rows]),
        truth_bits=torch.tensor([r[8] for r in rows]), truth_actions=torch.tensor([r[9] for r in rows]),
    )


def record_coordinates(width: int) -> torch.Tensor:
    """A records (n,0), B records (n,1), then definition records (s,3)."""
    node = torch.arange(width)
    symbol = torch.arange(SYMBOLS)
    return torch.cat((torch.stack((node, torch.zeros_like(node)), 1),
                      torch.stack((node, torch.ones_like(node)), 1),
                      torch.stack((symbol, torch.full_like(symbol, DEF_ROW)), 1)))


class RecurrentRuleProgram(nn.Module):
    """Field routes + recurrent executor whose opcode meanings may be read in context."""

    def __init__(self, mode: str = "rule", *, channels: int = 18, temperature: float = 48.0,
                 rule_init: tuple[float, float] = (0.0, 0.0)) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.temperature = temperature
        heads = 3 if mode == "rule" else 2
        self.heads = nn.ModuleList(
            [SpectralFieldPointerLM(coord_dims=2, channels=channels, vocab=64) for _ in range(heads)])
        with torch.no_grad():
            # A and B start one row short as in the joint program; the rule head
            # starts at zero displacement, equidistant from row 1 and row 3.
            self.heads[0].delta.copy_(torch.tensor([0.0, 1.0]))
            self.heads[1].delta.copy_(torch.tensor([0.0, 0.0]))
            if mode == "rule":
                self.heads[2].delta.copy_(torch.tensor(rule_init))
        rows = SYMBOLS + 1 if mode == "symbol" else BEHAVIORS + 1
        # (behavior or symbol, incl. none) x incoming bit x action(A/B/Stop) x outgoing bit
        self.transition_logits = nn.Parameter(torch.randn(rows, 2, 3, 2) * 0.02)

    # ----- addressing -------------------------------------------------
    def _pairs(self, width: int, query_columns: int) -> tuple[torch.Tensor, torch.Tensor]:
        records = record_coordinates(width)
        keys, queries = [], []
        for column in range(query_columns):
            query = torch.tensor([column, QUERY_ROW])
            sentinel = torch.tensor([width + 20 + column, 7])
            queries.append(torch.cat((records, query[None])))
            keys.append(torch.cat((records, sentinel[None])))
        return torch.stack(keys), torch.stack(queries)

    def _scores(self, head: int, width: int) -> torch.Tensor:
        columns = SYMBOLS if head == 2 else width
        keys, queries = self._pairs(width, columns)
        return self.heads[head].route_scores(keys, queries)[:, -1, : 2 * width + SYMBOLS]

    @torch.no_grad()
    def routes(self, width: int, *, integer: bool = True) -> list[torch.Tensor]:
        """Hard route index per head; index 2W+K is the sentinel (invalid)."""
        out = []
        for head in range(len(self.heads)):
            if integer:
                columns = SYMBOLS if head == 2 else width
                keys, queries = self._pairs(width, columns)
                out.append(self.heads[head].integer_routes(keys, queries)[:, -1])
            else:
                out.append(self._scores(head, width).argmax(-1))
        return out

    @torch.no_grad()
    def route_accuracy(self, width: int) -> dict[str, float]:
        expected = [torch.arange(width), torch.arange(width) + width,
                    torch.arange(SYMBOLS) + 2 * width][: len(self.heads)]
        floating = self.routes(width, integer=False)
        integer = self.routes(width, integer=True)
        names = ["A", "B", "rule"]
        result = {}
        for name, exp, flo, intr in zip(names, expected, floating, integer):
            result[name] = {"float": float((flo == exp).float().mean()),
                            "integer": float((intr == exp).float().mean()),
                            "agreement": float((flo == intr).float().mean())}
        return result

    @property
    def transition_probabilities(self) -> torch.Tensor:
        rows = self.transition_logits.shape[0]
        return torch.softmax(self.transition_logits.reshape(rows, 2, 6), -1).reshape(rows, 2, 3, 2)

    # ----- soft surrogate ----------------------------------------------
    def terminal_probabilities(self, batch: RuleBatch, *, budget: int = 8) -> torch.Tensor:
        count, _, width = batch.edges.shape
        dtype = torch.float32
        node_prob = [torch.softmax(self.temperature * self._scores(h, width), -1) for h in (0, 1)]
        pad = torch.zeros(count, SYMBOLS, width, dtype=dtype)
        edge_values = torch.cat((nnf.one_hot(batch.edges.reshape(count, 2 * width), width).to(dtype), pad), 1)
        none_sym = nnf.one_hot(torch.full((count, width + SYMBOLS), NONE_SYMBOL), SYMBOLS + 1).to(dtype)
        sym_values = torch.cat((nnf.one_hot(batch.symbols, SYMBOLS + 1).to(dtype), none_sym), 1)
        edge_a = torch.einsum("nk,bkv->bnv", node_prob[0], edge_values)
        edge_b = torch.einsum("nk,bkv->bnv", node_prob[1], edge_values)
        symbol = torch.einsum("nk,bks->bns", node_prob[0], sym_values)  # (b, node, K+1)
        if self.mode == "symbol":
            row_mix = symbol
        else:
            if self.mode == "rule":
                rule_prob = torch.softmax(self.temperature * self._scores(2, width), -1)
                none_def = nnf.one_hot(torch.full((count, 2 * width), NONE_BEHAVIOR), BEHAVIORS + 1).to(dtype)
                def_values = torch.cat((none_def, nnf.one_hot(batch.defs, BEHAVIORS + 1).to(dtype)), 1)
                meaning = torch.einsum("sk,bkq->bsq", rule_prob, def_values)  # (b, K, 5)
            else:
                meaning = nnf.one_hot(batch.defs, BEHAVIORS + 1).to(dtype)
            none_row = nnf.one_hot(torch.full((count, 1), NONE_BEHAVIOR), BEHAVIORS + 1).to(dtype)
            meaning = torch.cat((meaning, none_row), 1)  # (b, K+1, 5)
            row_mix = torch.einsum("bns,bsq->bnq", symbol, meaning)
        policy = torch.einsum("bnq,qmaj->bnmaj", row_mix, self.transition_probabilities)
        active = nnf.one_hot(batch.starts, width).to(dtype)[:, :, None] * nnf.one_hot(
            batch.initial_bits, 2).to(dtype)[:, None, :]
        halted = torch.zeros_like(active)
        for _ in range(budget):
            mass_a = (active[:, :, :, None] * policy[:, :, :, 0, :]).sum(2)
            mass_b = (active[:, :, :, None] * policy[:, :, :, 1, :]).sum(2)
            new_active = torch.einsum("bnj,bnv->bvj", mass_a, edge_a) + torch.einsum("bnj,bnv->bvj", mass_b, edge_b)
            halted = halted + (active[:, :, :, None] * policy[:, :, :, 2, :]).sum(2)
            active = new_active
        return halted

    def loss(self, batch: RuleBatch, *, budget: int = 8) -> torch.Tensor:
        terminal = self.terminal_probabilities(batch, budget=budget)
        rows = torch.arange(len(batch.starts))
        return -torch.log(terminal[rows, batch.target_nodes, batch.target_bits].clamp_min(1e-12)).mean()

    # ----- hard native consumer ------------------------------------------
    @torch.no_grad()
    def row_index(self, batch: RuleBatch, *, rule_shift: int = 0, rule_read: bool = True) -> dict[str, torch.Tensor]:
        """Per-node transition-table row from hard integer reads (plus the reads themselves)."""
        count, _, width = batch.edges.shape
        routes = self.routes(width, integer=True)
        invalid = torch.full((count, 1), -1)
        record_edge = torch.cat((batch.edges.reshape(count, 2 * width), torch.full((count, SYMBOLS), -1), invalid), 1)
        record_sym = torch.cat((batch.symbols, torch.full((count, width + SYMBOLS + 1), NONE_SYMBOL)), 1)
        record_def = torch.cat((torch.full((count, 2 * width), NONE_BEHAVIOR), batch.defs,
                                torch.full((count, 1), NONE_BEHAVIOR)), 1)
        symbol = record_sym[:, routes[0]]
        out = {"edge_a": record_edge[:, routes[0]], "edge_b": record_edge[:, routes[1]], "symbol": symbol}
        if self.mode == "symbol":
            out["row"] = symbol
            return out
        if self.mode == "rule":
            rule_routes = routes[2]
            if rule_shift:
                rule_routes = rule_routes[(torch.arange(SYMBOLS) + rule_shift) % SYMBOLS]
            meaning = record_def[:, rule_routes]
            if not rule_read:
                meaning = torch.full_like(meaning, NONE_BEHAVIOR)
        else:
            meaning = batch.defs.clone()
        meaning = torch.cat((meaning, torch.full((count, 1), NONE_BEHAVIOR)), 1)
        out["meaning"] = meaning[:, :SYMBOLS]
        out["row"] = meaning.gather(1, symbol)
        return out

    @torch.no_grad()
    def rollout(self, batch: RuleBatch, *, budget: int = 16, rule_shift: int = 0, rule_read: bool = True,
                write_enabled: bool = True) -> dict[str, torch.Tensor]:
        count = len(batch.starts)
        reads = self.row_index(batch, rule_shift=rule_shift, rule_read=rule_read)
        choices = self.transition_logits.reshape(self.transition_logits.shape[0], 2, 6).argmax(-1)
        current = batch.starts.clone()
        bit = batch.initial_bits.clone()
        halted = torch.zeros(count, dtype=torch.bool)
        invalid = torch.zeros(count, dtype=torch.bool)
        stop_step = torch.full((count,), -1, dtype=torch.long)
        rows = torch.arange(count)
        nodes, bits, actions, table_rows = [current.clone()], [bit.clone()], [], []
        for step in range(budget):
            row = reads["row"][rows, current]
            decision = choices[row, bit]
            action = torch.where(halted, torch.full_like(decision, 2), decision // 2)
            new_bit = torch.where(halted, bit, decision % 2)
            target = torch.where(action == 0, reads["edge_a"][rows, current],
                                 torch.where(action == 1, reads["edge_b"][rows, current], current))
            bad = target < 0
            invalid = invalid | (bad & ~halted)
            target = torch.where(bad, current, target)
            if not write_enabled:
                target, new_bit = current, bit
            just = (action == 2) & ~halted
            stop_step = torch.where(just, torch.full_like(stop_step, step), stop_step)
            halted = halted | just
            current, bit = target, new_bit
            actions.append(action.clone())
            table_rows.append(row.clone())
            nodes.append(current.clone())
            bits.append(bit.clone())
        return {"node": current, "bit": bit, "halted": halted, "stop_step": stop_step, "invalid_read": invalid,
                "nodes": torch.stack(nodes, 1), "bits": torch.stack(bits, 1),
                "actions": torch.stack(actions, 1), "rows": torch.stack(table_rows, 1)}

    @torch.no_grad()
    def policy_metrics(self) -> dict:
        predicted = self.transition_logits.reshape(self.transition_logits.shape[0], 2, 6).argmax(-1).tolist()
        result = {"predicted": predicted}
        if self.mode != "symbol":
            result["correct"] = sum(predicted[b][x] == ORACLE_POLICY[b][x] for b in range(4) for x in range(2))
            result["total"] = 8
        return result

    def deltas(self) -> list[list[float]]:
        return [head.delta.detach().tolist() for head in self.heads]
