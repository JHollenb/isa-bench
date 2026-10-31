"""Exact candidate index for select heads: phase-sector cells with integer bounds.

A select head scores ``s(t, j) = sim(q_t, k_j) + lam * (P_q[t] - P_k[j])`` and
keeps the top-k by the composite ``s * cap + pref(j)`` (``pref`` is the tie
rank). Scoring every causal pair costs O(T_q T_k C). This index proposes
candidates instead and never changes the answer:

* Keys go into cells keyed by the top ``sector_bits`` of the phase code on a
  few index channels (``circular_l1``) or a hash of the whole word
  (``exact_match``). A cell on channel c covers one contiguous phase arc.
* For a query, the circular distance from ``Theta_q[c]`` to any code in an arc
  is at least the distance to the nearer arc endpoint (zero if the arc holds
  the query). That gives an exact integer upper bound on ``sim`` for every key
  in the cell; channels that are not indexed contribute their best case (0 for
  ``circular_l1``, ``w_c`` for ``exact_match``).
* The clock term is bounded by the largest ``-lam * P_k`` among the cell's
  eligible keys. Keys are stored in position order, so the eligible keys of a
  causal query are a prefix of each cell and a running maximum gives the bound.
* Cells are visited in decreasing bound order. Once k candidates are held, a
  cell (or a key) whose bound composite cannot exceed the k-th best composite is
  skipped. Composites are unique per key, so the skipped keys cannot enter the
  top-k: the result equals the all-pairs scan bit for bit.

Everything is integer arithmetic on phase codes; no word is decoded or summed.
Keys must be appended in nondecreasing position order (the batch executor sorts
them first). A small tail of recent keys is scored exhaustively until it is
merged into the cell layout, so streaming appends stay cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .core import SPFError, circular_diff_u16, unpack_L, unpack_theta

_PHASE_BITS = 16
_PHASE_MOD = 1 << _PHASE_BITS
_HALF_TURN = 1 << (_PHASE_BITS - 1)
_MAX_CELL_BITS = 16
_WORD_HASH = np.uint64(0x9E3779B97F4A7C15)
_COMPOSITE_BUDGET = 1 << 62
_MAX_ROUND = 64
_ID_BLOCKS = 64


@dataclass
class SelectIndexCounters:
    """Declared op-model counts for the indexed path (not latency measurements).

    ``pairs_scored`` counts full query-key score evaluations (every channel);
    ``bound_evaluations`` counts cell or key bounds, each O(index channels).
    """

    pairs_scored: int = 0
    buckets_probed: int = 0
    bound_evaluations: int = 0
    integer_ops: int = 0
    comparisons: int = 0


def _int_array(value: npt.ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise SPFError(f"{name} must be an integer array")
    return arr.astype(np.int64)


def _arc_distance(theta: int, start: np.ndarray, width: int) -> np.ndarray:
    """Least circular distance from ``theta`` to the arcs ``[start, start + width)``."""
    first = np.abs(circular_diff_u16(theta, start)).astype(np.int64)
    last = np.abs(circular_diff_u16(theta, start + (width - 1))).astype(np.int64)
    inside = ((theta - start) & (_PHASE_MOD - 1)) < width
    return np.where(inside, 0, np.minimum(first, last))


class SelectHeadIndex:
    """Append-only exact candidate index for one select-head key stream.

    ``capacity`` is the largest number of keys (ids are 0..capacity-1 in append
    order) and scales the tie composite exactly as ``T_k`` does in the
    all-pairs executor. ``clocked=True`` requires a clock reading per key and
    per query; the score then carries ``forget_scale * (P_q - P_k)``.
    """

    def __init__(
        self,
        channel_weights: npt.ArrayLike,
        *,
        capacity: int,
        score: str = "circular_l1",
        clocked: bool = False,
        forget_scale: int = 1,
        tie: str = "latest",
        index_channels: npt.ArrayLike | None = None,
        sector_bits: int = 4,
        tail_limit: int = 256,
    ) -> None:
        weights = _int_array(channel_weights, "channel_weights")
        if weights.ndim != 1 or np.any(weights < 0):
            raise SPFError("channel_weights must be one nonnegative integer per channel")
        if score not in ("circular_l1", "exact_match"):
            raise SPFError("score must be 'circular_l1' or 'exact_match'")
        if tie not in ("latest", "earliest"):
            raise SPFError("tie must be 'latest' or 'earliest'")
        if isinstance(forget_scale, (bool, np.bool_)) or not isinstance(forget_scale, (int, np.integer)):
            raise SPFError("forget_scale must be an integer score-per-clock-code scale")
        if not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise SPFError("capacity must be a positive integer")
        self.n_channels = int(weights.shape[0])
        self.score = score
        self.tie = tie
        self.capacity = int(capacity)
        self.clocked = bool(clocked)
        self.lam = int(forget_scale) if clocked else 0
        self.tail_limit = int(tail_limit)
        self._used = np.flatnonzero(weights)
        self._w = weights[self._used]
        used = [int(c) for c in self._used]
        if index_channels is None:
            ranked = sorted(used, key=lambda c: (-int(weights[c]), c))
            chosen = ranked[: min(2, len(ranked))]
        else:
            chosen = [int(c) for c in np.atleast_1d(_int_array(index_channels, "index_channels"))]
            if len(set(chosen)) != len(chosen) or any(c not in used for c in chosen):
                raise SPFError("index_channels must be distinct channels with nonzero weight")
        bits = int(sector_bits)
        if bits < 1 or bits > _PHASE_BITS or len(chosen) * bits > _MAX_CELL_BITS:
            raise SPFError(f"sector_bits must be in [1, 16] with index channels x bits <= {_MAX_CELL_BITS}")
        self.index_channels = tuple(chosen)
        self.sector_bits = bits
        self._slot = np.array([used.index(c) for c in chosen], dtype=np.int64)
        free = np.ones(len(used), dtype=np.bool_)
        free[self._slot] = False
        # Best case of the channels the cells do not constrain.
        self._free_best = 0 if score == "circular_l1" else int(self._w[free].sum())
        self._next_id = 0
        self._last_pos: int | None = None
        self._g_abs = 0
        self._tail: list[tuple[np.ndarray, ...]] = []
        self._tail_n = 0
        self._frozen = self._empty_frozen()
        self.counters = SelectIndexCounters()

    # ------------------------------------------------------------------ layout

    def _empty_frozen(self) -> dict[str, np.ndarray]:
        n_used = len(self._used)
        data_dtype = np.uint16 if self.score == "circular_l1" else np.uint32
        z = np.zeros(0, dtype=np.int64)
        return {
            "ids": z,
            "pos": z,
            "g": z,
            "data": np.zeros((0, n_used), dtype=data_dtype),
            "cell": z,
            "csr_ids": z,
            "csr_g": z,
            "csr_gmax": z,
            "csr_data": np.zeros((0, n_used), dtype=data_dtype),
            "seg_sectors": np.zeros((0, len(self._slot)), dtype=np.int64),
            "seg_start": z,
            "seg_mono": np.zeros(0, dtype=np.bool_),
            "seg_key": z,
            "seg_upto": np.zeros((0, _ID_BLOCKS + 1), dtype=np.int64),
            "id_block": np.int64(1),
        }

    def _sectors(self, data: np.ndarray) -> np.ndarray:
        """Per-index-channel sector of each row (``data`` holds the used channels)."""
        cols = data[:, self._slot] if data.ndim == 2 else data[self._slot]
        if self.score == "circular_l1":
            return cols.astype(np.int64) >> (_PHASE_BITS - self.sector_bits)
        hashed = cols.astype(np.uint64) * _WORD_HASH
        return (hashed >> np.uint64(64 - self.sector_bits)).astype(np.int64)

    def _cell(self, data: np.ndarray) -> np.ndarray:
        sectors = self._sectors(data)
        cell = np.zeros(sectors.shape[0], dtype=np.int64)
        for i in range(sectors.shape[1]):
            cell |= sectors[:, i] << (self.sector_bits * i)
        return cell

    def _key_data(self, words: np.ndarray) -> np.ndarray:
        cols = words[..., self._used]
        if self.score == "circular_l1":
            if cols.size and np.any(unpack_L(cols) != 0):
                raise SPFError("circular_l1 channels must be finite nonzero unit-circle words (L=0)")
            return unpack_theta(cols).astype(np.uint16)
        return cols.astype(np.uint32)

    def append(
        self,
        key_words: npt.ArrayLike,
        *,
        positions: npt.ArrayLike | None = None,
        clock: npt.ArrayLike | None = None,
        valid: npt.ArrayLike | None = None,
    ) -> np.ndarray:
        """Append keys (n, C) in nondecreasing position order; returns their ids."""
        words = np.asarray(key_words)
        if words.dtype != np.uint32 or words.ndim != 2 or words.shape[1] != self.n_channels:
            raise SPFError("key_words must be packed SPF32 uint32 words shaped (n, C)")
        n = words.shape[0]
        if self._next_id + n > self.capacity:
            raise SPFError("append would exceed the index capacity")
        ids = np.arange(self._next_id, self._next_id + n, dtype=np.int64)
        pos = ids.copy() if positions is None else _int_array(positions, "positions")
        if pos.shape != (n,):
            raise SPFError("positions must give one integer per appended key")
        if n and (np.any(np.diff(pos) < 0) or (self._last_pos is not None and pos[0] < self._last_pos)):
            raise SPFError("keys must be appended in nondecreasing position order")
        if self.clocked:
            if clock is None:
                raise SPFError("a clocked index needs one clock reading per key")
            g = -self.lam * _int_array(clock, "clock")
            if g.shape != (n,):
                raise SPFError("clock must give one integer per appended key")
        else:
            g = np.zeros(n, dtype=np.int64)
        live = np.ones(n, dtype=np.bool_) if valid is None else np.asarray(valid)
        if live.dtype != np.bool_ or live.shape != (n,):
            raise SPFError("valid must be a boolean mask with one entry per key")
        data = self._key_data(words)
        self._next_id += n
        if n:
            self._last_pos = int(pos[-1])
        keep = np.flatnonzero(live)
        if keep.size:
            self._g_abs = max(self._g_abs, int(np.abs(g[keep]).max()))
            self._tail.append((ids[keep], pos[keep], g[keep], data[keep]))
            self._tail_n += keep.size
            if self._tail_n > self.tail_limit:
                self.freeze()
        return ids

    def freeze(self) -> None:
        """Merge the exhaustively scored tail into the cell layout."""
        if not self._tail:
            return
        old = self._frozen
        ids = np.concatenate([old["ids"], *(t[0] for t in self._tail)])
        pos = np.concatenate([old["pos"], *(t[1] for t in self._tail)])
        g = np.concatenate([old["g"], *(t[2] for t in self._tail)])
        data = np.concatenate([old["data"], *(t[3] for t in self._tail)])
        cell = np.concatenate([old["cell"], *(self._cell(t[3]) for t in self._tail)])
        self._tail, self._tail_n = [], 0
        order = np.argsort(cell, kind="stable")  # by cell, then id (append order)
        csr_cell = cell[order]
        cells, seg_start = np.unique(csr_cell, return_index=True)
        seg_end = np.append(seg_start[1:], csr_cell.size)
        csr_g = g[order]
        gmax = np.empty_like(csr_g)
        mono = np.empty(cells.size, dtype=np.bool_)
        for s, (lo, hi) in enumerate(zip(seg_start, seg_end, strict=True)):
            gmax[lo:hi] = np.maximum.accumulate(csr_g[lo:hi])
            mono[s] = bool(np.all(gmax[lo:hi] == csr_g[lo:hi]))
        mask = (1 << self.sector_bits) - 1
        sectors = np.zeros((cells.size, len(self._slot)), dtype=np.int64)
        for i in range(len(self._slot)):
            sectors[:, i] = (cells >> (self.sector_bits * i)) & mask
        seg_rank = np.repeat(np.arange(cells.size, dtype=np.int64), seg_end - seg_start)
        seg_key = seg_rank * self.capacity + ids[order]
        # seg_upto[s, b]: end of segment s's keys with id < b * block (a coarse causal prefix).
        block = max(1, -(-(int(ids.max()) + 1) // _ID_BLOCKS))
        edges = np.minimum(np.arange(_ID_BLOCKS + 1, dtype=np.int64) * block, self.capacity)  # stay in-segment
        seg_first = np.arange(cells.size, dtype=np.int64) * self.capacity
        self._frozen = {
            "ids": ids,
            "pos": pos,
            "g": g,
            "data": data,
            "cell": cell,
            "csr_ids": ids[order],
            "csr_g": csr_g,
            "csr_gmax": gmax,
            "csr_data": data[order],
            "seg_sectors": sectors,
            "seg_start": seg_start.astype(np.int64),
            "seg_mono": mono,
            "seg_key": seg_key,
            "seg_upto": np.searchsorted(seg_key, seg_first[:, None] + edges[None, :], side="left"),
            "id_block": np.int64(block),
        }

    # ------------------------------------------------------------------- query

    def _pref(self, ids: np.ndarray) -> np.ndarray:
        return ids if self.tie == "latest" else (self.capacity - 1) - ids

    def _sim(self, q: np.ndarray, data: np.ndarray) -> np.ndarray:
        if self.score == "circular_l1":
            distance = np.abs(circular_diff_u16(q[None, :], data)).astype(np.int64)
            return -(distance @ self._w)
        return (data == q[None, :]).astype(np.int64) @ self._w

    def _sector_tables(self, q: np.ndarray) -> np.ndarray:
        """(index channels, 2^bits) best similarity contribution per sector."""
        n_sect = 1 << self.sector_bits
        sectors = np.arange(n_sect, dtype=np.int64)
        table = np.empty((len(self._slot), n_sect), dtype=np.int64)
        for i, slot in enumerate(self._slot):
            w = int(self._w[slot])
            if self.score == "circular_l1":
                width = 1 << (_PHASE_BITS - self.sector_bits)
                table[i] = -w * _arc_distance(int(q[slot]), sectors * width, width)
            else:
                table[i] = np.where(sectors == self._sectors(q[None, :])[0, i], w, 0)
        return table

    def query(
        self,
        query_words: npt.ArrayLike,
        *,
        position: int | None = None,
        clock: int | None = None,
        k: int = 1,
        causal: bool = True,
        strict: bool = True,
        min_score: int | None = None,
        max_probes: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Top-k ``(ids, scores)`` in decreasing composite order (length <= k).

        Exact unless ``max_probes`` caps the number of cells visited; a capped
        query is an approximate multi-probe search whose recall must be measured.
        """
        qw = np.asarray(query_words)
        if qw.dtype != np.uint32 or qw.shape != (self.n_channels,):
            raise SPFError("query_words must be one packed SPF32 word per channel")
        if causal and position is None:
            raise SPFError("a causal query needs its position")
        if self.clocked and clock is None:
            raise SPFError("a clocked index needs the query clock reading")
        q = self._key_data(qw[None, :])[0]
        lam_pq = self.lam * int(clock) if self.clocked else 0
        cap = self.capacity
        bound = int(self._w.sum()) * (_HALF_TURN if self.score == "circular_l1" else 1) + abs(lam_pq) + self._g_abs
        if min_score is not None:
            bound = max(bound, abs(int(min_score)))
        if (2 * bound + 2) * cap >= _COMPOSITE_BUDGET:
            raise SPFError("score range too wide for exact int64 top-k selection")
        side = "left" if strict else "right"
        n_used = len(self._used)
        c = self.counters
        best_comp = np.zeros(0, dtype=np.int64)
        best_ids = np.zeros(0, dtype=np.int64)
        best_s = np.zeros(0, dtype=np.int64)

        def merge(ids: np.ndarray, s: np.ndarray) -> int | None:
            nonlocal best_comp, best_ids, best_s
            if min_score is not None:
                keep = s >= int(min_score)
                ids, s = ids[keep], s[keep]
            comp = s * cap + self._pref(ids)
            best_comp = np.concatenate((best_comp, comp))
            best_ids = np.concatenate((best_ids, ids))
            best_s = np.concatenate((best_s, s))
            c.comparisons += int(comp.size)
            if best_comp.size > k:
                top = np.argpartition(best_comp, best_comp.size - k)[best_comp.size - k :]
                best_comp, best_ids, best_s = best_comp[top], best_ids[top], best_s[top]
            return int(best_comp.min()) if best_comp.size == k else None

        def score(ids: np.ndarray, g: np.ndarray, data: np.ndarray) -> np.ndarray:
            c.pairs_scored += int(ids.size)
            c.integer_ops += int(ids.size) * (4 * n_used + (3 if self.clocked else 0))
            return self._sim(q, data) + lam_pq + g

        tau: int | None = None
        # Tail: recent keys, scored exhaustively (they usually carry the best clock term).
        for ids, pos, g, data in self._tail:
            take = np.ones(ids.size, dtype=np.bool_) if not causal else pos < position if strict else pos <= position
            if causal:
                c.comparisons += int(ids.size)
            if np.any(take):
                tau = merge(ids[take], score(ids[take], g[take], data[take]))

        fz = self._frozen
        n_seg = fz["seg_start"].size
        if n_seg:
            if causal:
                e = int(np.searchsorted(fz["pos"], position, side=side))
                cutoff = int(fz["ids"][e]) if e < fz["ids"].size else cap
            else:
                cutoff = cap
            # Superset of each cell's eligible prefix: keys with id below the next block edge.
            # Bounds over a superset stay valid; probed cells cut the exact prefix below.
            column = min(-(-cutoff // int(fz["id_block"])), _ID_BLOCKS)
            ends = fz["seg_upto"][:, column]
            starts = fz["seg_start"]
            live = ends > starts
            table = self._sector_tables(q)
            ub_sim = np.full(n_seg, self._free_best, dtype=np.int64)
            for i in range(table.shape[0]):
                ub_sim += table[i, fz["seg_sectors"][:, i]]
            last = np.maximum(ends - 1, 0)
            ub_s = ub_sim + lam_pq + fz["csr_gmax"][last]
            if min_score is not None:
                live &= ub_s >= int(min_score)
            pref_max = self._pref(fz["csr_ids"][last] if self.tie == "latest" else fz["csr_ids"][starts])
            ub_comp = ub_s * cap + pref_max
            c.bound_evaluations += n_seg
            c.integer_ops += n_seg * (table.shape[0] + 3)
            budget = [max_probes]

            def probe(take: np.ndarray) -> None:
                nonlocal tau
                if tau is not None:
                    take = take[ub_comp[take] > tau]
                if budget[0] is not None:
                    take = take[: budget[0]]
                    budget[0] -= take.size
                pieces = []
                for seg in take:
                    lo, hi = int(starts[seg]), int(ends[seg])
                    if cutoff < cap:
                        hi = lo + int(np.searchsorted(fz["csr_ids"][lo:hi], cutoff, side="left"))
                    # Keep keys whose bound composite can still beat tau (or reach min_score).
                    floor = None
                    if tau is not None:
                        floor = (tau - (cap - 1)) // cap - int(ub_sim[seg]) - lam_pq
                    if min_score is not None:
                        need = int(min_score) - int(ub_sim[seg]) - lam_pq - 1
                        floor = need if floor is None else max(floor, need)
                    if floor is not None and self.clocked and fz["seg_mono"][seg]:
                        lo += int(np.searchsorted(fz["csr_g"][lo:hi], floor, side="right"))
                        c.bound_evaluations += 1
                        pieces.append(np.arange(lo, hi))
                    elif floor is not None and self.clocked:
                        span = np.arange(lo, hi)
                        c.bound_evaluations += span.size
                        pieces.append(span[fz["csr_g"][lo:hi] > floor])
                    else:
                        pieces.append(np.arange(lo, hi))
                c.buckets_probed += int(take.size)
                rows = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int64)
                if rows.size:
                    ids = fz["csr_ids"][rows]
                    tau = merge(ids, score(ids, fz["csr_g"][rows], fz["csr_data"][rows]))

            order = np.flatnonzero(live)
            if order.size > 1 and tau is None:
                # Probe the best cell first; its keys usually prune most of the rest before sorting.
                head = int(np.argmax(ub_comp[order]))
                probe(order[head : head + 1])
                order = np.delete(order, head)
            if tau is not None:
                order = order[ub_comp[order] > tau]
            order = order[np.argsort(-ub_comp[order], kind="stable")]
            at, width = 0, 1
            while at < order.size and (budget[0] is None or budget[0] > 0):
                if tau is not None and ub_comp[order[at]] <= tau:
                    break
                probe(order[at : at + width])
                at += width
                width = min(2 * width, _MAX_ROUND)
        ranked = np.argsort(-best_comp, kind="stable")
        return best_ids[ranked], best_s[ranked]


__all__ = ["SelectHeadIndex", "SelectIndexCounters"]
