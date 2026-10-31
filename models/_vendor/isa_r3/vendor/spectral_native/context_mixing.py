"""Exact contexts with geometric mixing: the compression-community recipe on field joins.

Every context model is an exact integer join (a hashed key per position) into a
count table of next bytes, as in ``fieldlm_text``.  What changes is how the
tables' evidence is combined:

- **Prior.** A dense Witten-Bell chain over orders 0-2, ``q(y)``.
- **Model distributions.** Each context model ``m`` with counts ``n_m(y)``,
  total ``N_m`` and ``u_m`` distinct followers gives the Witten-Bell
  distribution ``p_m(y) = (n_m(y) + u_m q(y)) / (N_m + u_m)`` (``q`` if unseen).
- **Geometric mixing** (Mattern, DCC 2012; PAQ's logistic mixing generalized to
  a 256-letter alphabet): ``p(y) ∝ q(y)^α Π_m p_m(y)^{w_m}``.  Because every
  ``p_m`` is a smoothed version of the same ``q``, the log score is exactly

      s(y) = α' log q(y) + Σ_m w_m δ_m(y),   δ_m(y) = log(1 + n_m(y) / (u_m q(y)))

  (constant terms cancel in the normalization), so only each context's stored
  entries enter, and the objective is convex in (α', w).
- **Weight sets** are selected by small contexts (match tier x order evidence;
  previous-byte class x word start x line start), and each model's weight also
  depends on its own count bucket (confidence), all additive, so still convex.
- **APM / SSE.** A secondary estimation stage: ``s'(y) = s̃(y) + g[c, bucket(s̃(y))]``
  with ``s̃ = log p`` and linear interpolation between buckets, then
  renormalized; convex in ``g`` with the mixer fixed.
- **Arithmetic mixing** (the ablation): ``p(y*) = Σ_m π_m p_m(y*)`` with softmax
  weights from the same selectors.

Storage follows track D: 24 bits per table entry, 56 per key, 32 per mixer or
APM number.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from .fieldlm_text import U64, SubstringHasher, _byte_class, _mix, compute_charts, segment_starts

ENTRY_BITS = 24
KEY_BITS = 56
PARAM_BITS = 32
BOS = 256

ORDERS = (3, 4, 5, 6, 8)
MATCH_TIERS = (32, 20, 12, 8, 6, 5, 4)  # longest first
GROUPS: dict[str, tuple[str, ...]] = {
    "orders": tuple(f"o{k}" for k in ORDERS),
    "word": ("w1", "w2", "w3"),
    "sparse": ("sk2", "sk23", "sk34", "sk13", "sk456"),
    "indirect": ("ind1", "ind2"),
    "column": ("struct_col", "struct_depth", "sheet2d", "sheet2d_b"),
    "match": ("match", "match_c1"),
}
ALL_MODELS: tuple[str, ...] = tuple(m for g in GROUPS.values() for m in g)
ARMS: dict[str, dict] = {
    "gm_full": {"mix": "geometric", "models": ALL_MODELS, "apm": True},
    "gm_no_indirect_sparse": {
        "mix": "geometric",
        "models": tuple(m for m in ALL_MODELS if m not in GROUPS["sparse"] + GROUPS["indirect"]),
        "apm": True,
    },
    "gm_orders": {"mix": "geometric", "models": GROUPS["orders"], "apm": True},
    "am_full": {"mix": "arithmetic", "models": ALL_MODELS, "apm": False},
}
GM_ARMS = tuple(k for k, v in ARMS.items() if v["mix"] == "geometric")

COUNT_EDGES = (1, 2, 4, 8, 16, 64, 256)  # count buckets 0..7 (0 = unseen)
N_BUCKETS = len(COUNT_EDGES) + 1
N_S1 = 4 * (len(ORDERS) + 1)  # match-tier bucket x number of order models with evidence
N_S2 = 5 * 2 * 2  # previous-byte class x at word start x at line start
APM_KNOTS = 33
APM_LO = -24.0  # nats


# ---------------------------------------------------------------------------
# Contexts
# ---------------------------------------------------------------------------

ALPHA = np.zeros(256, dtype=bool)
for _c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
    ALPHA[_c] = True


def _const(value: int, shape) -> np.ndarray:
    return np.full(shape, value, dtype=U64)


def _prev_same(key_all: np.ndarray) -> np.ndarray:
    """Most recent earlier index with the same nonzero key, else -1."""

    n = key_all.shape[0]
    order = np.lexsort((np.arange(n), key_all))
    sorted_keys = key_all[order]
    prev = np.full(n, -1, dtype=np.int64)
    same = (sorted_keys[1:] == sorted_keys[:-1]) & (sorted_keys[1:] != 0)
    prev[order[1:][same]] = order[:-1][same]
    return prev


@dataclass
class Contexts:
    positions: np.ndarray
    targets: np.ndarray
    c1: np.ndarray  # previous byte (always present: segment-local index >= 1)
    c2: np.ndarray  # byte before that, BOS at the segment start
    keys: dict[str, np.ndarray]  # model -> uint64 key per position (0 = no context)
    match_pred: np.ndarray  # -1 if none
    match_tier: np.ndarray  # 0 none, else index into MATCH_TIERS (1 = longest)
    s2: np.ndarray  # selector 2 (static part)
    categories: dict[str, np.ndarray] = field(default_factory=dict)

    def take(self, idx: np.ndarray) -> "Contexts":
        return Contexts(
            self.positions[idx], self.targets[idx], self.c1[idx], self.c2[idx],
            {k: v[idx] for k, v in self.keys.items()}, self.match_pred[idx], self.match_tier[idx], self.s2[idx],
            {k: v[idx] for k, v in self.categories.items()},
        )


def compute_contexts(x: np.ndarray, seg_len: int | None, models: tuple[str, ...] = ALL_MODELS) -> Contexts:
    """Exact causal context keys for predicting x[t] from x[seg_start(t):t].

    ``seg_len=None`` uses the whole history (the online protocol).
    """

    x = x.astype(np.int64)
    n = x.shape[0]
    idx = np.arange(n, dtype=np.int64)
    seg = segment_starts(n, seg_len)
    local = idx - seg
    pos = idx[local >= 1]
    y = x[pos]
    hasher = SubstringHasher(x)
    shape = pos.shape

    def byte_at(offset: int) -> np.ndarray:
        return np.where(local[pos] >= offset, x[np.clip(pos - offset, 0, n - 1)], BOS)

    c1 = byte_at(1)
    c2 = byte_at(2)
    keys: dict[str, np.ndarray] = {}
    want = set(models)

    # content orders; near a segment start the available prefix is its own (BOS) context
    for k in ORDERS:
        if f"o{k}" in want:
            a = np.maximum(pos - k, seg[pos])
            keys[f"o{k}"] = _mix(hasher(a, pos), _const(k, shape)) | U64(1)

    # sparse / skip contexts
    sparse = {
        "sk2": (2,), "sk23": (2, 3), "sk34": (3, 4), "sk13": (1, 3), "sk456": (4, 5, 6),
    }
    for i, (name, offs) in enumerate(sparse.items()):
        if name in want:
            parts = [_const(100 + i, shape)] + [byte_at(o).astype(U64) for o in offs]
            keys[name] = _mix(*parts) | U64(1)

    # word contexts on case-folded letters (PAQ's text model)
    if want & set(GROUPS["word"]):
        alpha = ALPHA[np.clip(x, 0, 255)]
        x_lc = np.where(alpha, x | 32, x)
        lc = SubstringHasher(x_lc)
        nonalpha_idx = np.where(~alpha, idx, -1)
        last_nonalpha = np.maximum.accumulate(nonalpha_idx)
        last_nonalpha_before = np.concatenate([[-1], last_nonalpha[:-1]])
        alpha_idx = np.where(alpha, idx, -1)
        last_alpha = np.maximum.accumulate(alpha_idx)
        last_alpha_before = np.concatenate([[-1], last_alpha[:-1]])
        word_start = np.maximum(np.maximum(last_nonalpha_before + 1, seg), idx - 24)

        def previous_word(start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            s_c = np.clip(start, 0, n - 1)
            end = np.where(start > 0, last_alpha_before[s_c] + 1, 0)
            has = end > seg
            e_c = np.clip(end - 1, 0, n - 1)
            begin = np.maximum(np.maximum(last_nonalpha_before[e_c] + 1, seg), end - 24)
            return begin, end, has

        pws, pwe, has1 = previous_word(word_start)
        ppws, ppwe, has2 = previous_word(np.where(has1, pws, 0))
        has2 = has2 & has1
        prefix = lc(word_start[pos], pos)
        w1h = np.where(has1[pos], lc(pws[pos], pwe[pos]), U64(7))
        w2h = np.where(has2[pos], lc(ppws[pos], ppwe[pos]), U64(7))
        keys["w1"] = _mix(_const(401, shape), prefix) | U64(1)
        keys["w2"] = _mix(_const(402, shape), prefix, w1h) | U64(1)
        keys["w3"] = _mix(_const(403, shape), prefix, w1h, w2h) | U64(1)
        word_start_at = word_start[pos] == pos
    else:
        word_start_at = np.zeros(shape, dtype=bool)

    # structural and 2D-sheet contexts (track D's charts)
    charts = compute_charts(x, seg_len, joins=False)
    for name in GROUPS["column"]:
        if name in want:
            keys[name] = charts.keys[name]
    line_start = charts.column == 0

    # indirect contexts: the bytes that followed the last two occurrences of the
    # order-1 / order-2 context (in-context joins, PAQ's indirect model)
    for k in (1, 2):
        name = f"ind{k}"
        if name not in want:
            continue
        valid_all = local >= k
        h_all = hasher(np.clip(idx - k, 0, n), idx)
        # the join key carries the segment so joins stay inside it; the stored key does not
        join_all = np.where(valid_all, _mix(h_all, seg.astype(U64)) | U64(1), U64(0))
        prev = _prev_same(join_all)
        p1 = prev[pos]
        t1 = np.where(p1 >= 0, x[np.clip(p1, 0, n - 1)], BOS)
        p2 = np.where(p1 >= 0, prev[np.clip(p1, 0, n - 1)], -1)
        t2 = np.where(p2 >= 0, x[np.clip(p2, 0, n - 1)], BOS)
        keys[name] = np.where(
            valid_all[pos],
            _mix(_const(500 + k, shape), h_all[pos], t1.astype(U64), t2.astype(U64)) | U64(1),
            U64(0),
        )

    # match join: most recent earlier occurrence of the longest suffix
    match_pred = np.full(shape, -1, dtype=np.int64)
    match_tier = np.zeros(shape, dtype=np.int64)
    for tier, k in enumerate(MATCH_TIERS, start=1):
        valid_all = local >= k
        h_all = np.where(valid_all, hasher(np.clip(idx - k, 0, n), idx), U64(0))
        key_all = np.where(valid_all, _mix(h_all, seg.astype(U64)) | U64(1), U64(0))
        cand = _prev_same(key_all)[pos]
        take = (cand >= 0) & (match_tier == 0)
        match_pred = np.where(take, x[np.clip(cand, 0, n - 1)], match_pred)
        match_tier = np.where(take, tier, match_tier)
    has_match = match_tier > 0
    if "match" in want:
        keys["match"] = np.where(
            has_match, _mix(_const(601, shape), match_tier.astype(U64), match_pred.astype(U64)) | U64(1), U64(0))
    if "match_c1" in want:
        keys["match_c1"] = np.where(
            has_match,
            _mix(_const(602, shape), match_tier.astype(U64), match_pred.astype(U64), c1.astype(U64)) | U64(1),
            U64(0),
        )

    cls1 = _byte_class(np.clip(c1, 0, 255))
    cls1 = np.where(c1 == BOS, 4, cls1)
    s2 = (cls1 * 2 + word_start_at.astype(np.int64)) * 2 + line_start.astype(np.int64)
    ctx = Contexts(pos, y, c1, c2, keys, match_pred, match_tier, s2)
    ctx.categories["byte_class"] = charts.categories["byte_class"]
    ctx.categories["copyable"] = match_pred == y
    return ctx


def match_bucket(tier: np.ndarray) -> np.ndarray:
    """0 none, 1 suffix >= 12 bytes, 2 suffix 6-8, 3 suffix 4-5."""

    length = np.where(tier > 0, np.array((0,) + MATCH_TIERS)[np.clip(tier, 0, len(MATCH_TIERS))], 0)
    return np.where(tier == 0, 0, np.where(length >= 12, 1, np.where(length >= 6, 2, 3)))


# ---------------------------------------------------------------------------
# Count tables
# ---------------------------------------------------------------------------


@dataclass
class Table:
    combined: np.ndarray  # sorted uint64 (key << 8 | byte)
    counts: np.ndarray  # int64
    keys: np.ndarray  # sorted unique 56-bit keys
    total: np.ndarray
    distinct: np.ndarray
    start: np.ndarray  # row start index into combined

    @staticmethod
    def build(keys: np.ndarray, targets: np.ndarray) -> "Table":
        valid = keys != 0
        combined = (keys[valid] << U64(8)) | targets[valid].astype(U64)
        uniq, counts = np.unique(combined, return_counts=True)
        return Table.from_entries(uniq, counts.astype(np.int64))

    @staticmethod
    def from_entries(combined: np.ndarray, counts: np.ndarray) -> "Table":
        key_of = combined >> U64(8)
        if key_of.size:
            starts = np.concatenate([[0], np.where(key_of[1:] != key_of[:-1])[0] + 1]).astype(np.int64)
            keys = key_of[starts]
            total = np.add.reduceat(counts, starts)
            distinct = np.diff(np.concatenate([starts, [key_of.size]]))
        else:
            starts = np.zeros(0, dtype=np.int64)
            keys = np.zeros(0, dtype=U64)
            total = np.zeros(0, dtype=np.int64)
            distinct = np.zeros(0, dtype=np.int64)
        return Table(combined, counts, keys, total, distinct, starts)

    @staticmethod
    def build_valued(keys: np.ndarray, targets: np.ndarray, logq: np.ndarray) -> "Table":
        """As ``build``, plus each entry's leave-one-out value (see ``loo_values``)."""

        valid = keys != 0
        combined = (keys[valid] << U64(8)) | targets[valid].astype(U64)
        uniq, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
        sum_logq = np.bincount(inverse, weights=logq[valid].astype(np.float64), minlength=uniq.size)
        table = Table.from_entries(uniq, counts.astype(np.int64))
        table.value = loo_values(table, sum_logq)
        return table

    def pruned(self, min_count: int) -> "Table":
        if min_count <= 1:
            return self
        keep = self.counts >= min_count
        return Table.from_entries(self.combined[keep], self.counts[keep])

    def pruned_by_value(self, threshold: float) -> "Table":
        keep = self.value >= threshold
        return Table.from_entries(self.combined[keep], self.counts[keep])

    def storage_bits_at_value(self, threshold: float) -> int:
        if not hasattr(self, "_sorted_value"):
            self._sorted_value = np.sort(self.value)
            row_max = np.maximum.reduceat(self.value, self.start) if self.start.size else self.value
            self._sorted_row_value = np.sort(row_max)
        entries = self._sorted_value.size - np.searchsorted(self._sorted_value, threshold)
        keys = self._sorted_row_value.size - np.searchsorted(self._sorted_row_value, threshold)
        return int(entries) * ENTRY_BITS + int(keys) * KEY_BITS

    def storage_bits(self) -> int:
        return int(self.counts.size) * ENTRY_BITS + int(self.keys.size) * KEY_BITS

    def storage_bits_at(self, min_count: int) -> int:
        """Storage after pruning at ``min_count``, without building the pruned table."""

        if min_count <= 1:
            return self.storage_bits()
        if not hasattr(self, "_row_max"):
            self._row_max = np.maximum.reduceat(self.counts, self.start) if self.start.size else self.counts
            self._sorted_counts = np.sort(self.counts)
            self._sorted_row_max = np.sort(self._row_max)
        entries = self._sorted_counts.size - np.searchsorted(self._sorted_counts, min_count)
        keys = self._sorted_row_max.size - np.searchsorted(self._sorted_row_max, min_count)
        return int(entries) * ENTRY_BITS + int(keys) * KEY_BITS

    def _knapsack_prep(self) -> None:
        if hasattr(self, "_ks_order"):
            return
        row = np.repeat(np.arange(self.keys.size), self.distinct)
        order = np.lexsort((-self.value, row))  # rows in key order, entries by value descending
        self._ks_order = order
        self._ks_value = self.value[order]
        self._ks_rank = np.arange(order.size) - np.repeat(self.start, self.distinct)
        self._ks_cost = ENTRY_BITS + KEY_BITS * (self._ks_rank == 0)

    def knapsack_keep(self, price: float) -> tuple[np.ndarray, int]:
        """Per row, keep the value-ordered prefix maximizing sum(value - price * bits), where the
        first entry also pays the 56-bit key; drop the row if no prefix is positive.
        Returns (mask over entries in table order, storage bits)."""

        self._knapsack_prep()
        if self.counts.size == 0:
            return np.zeros(0, dtype=bool), 0
        gain = self._ks_value - price * self._ks_cost
        csum = np.cumsum(gain)
        offset = np.repeat(np.concatenate([[0.0], csum[self.start[1:] - 1]]), self.distinct)
        within = csum - offset
        best = np.maximum.reduceat(within, self.start)
        best_e = np.repeat(best, self.distinct)
        first_best = np.minimum.reduceat(np.where(within == best_e, self._ks_rank, np.iinfo(np.int64).max), self.start)
        keep_sorted = (self._ks_rank <= np.repeat(first_best, self.distinct)) & (best_e > 0)
        mask = np.zeros(self.counts.size, dtype=bool)
        mask[self._ks_order[keep_sorted]] = True
        bits = int(keep_sorted.sum()) * ENTRY_BITS + int((best > 0).sum()) * KEY_BITS
        return mask, bits

    def pruned_by_price(self, price: float) -> "Table":
        mask, _ = self.knapsack_keep(price)
        return Table.from_entries(self.combined[mask], self.counts[mask])

    def rows(self, keys: np.ndarray) -> np.ndarray:
        kq = (keys << U64(8)) >> U64(8)
        if self.keys.size == 0:
            return np.full(keys.shape, -1, dtype=np.int64)
        j = np.searchsorted(self.keys, kq)
        j_c = np.clip(j, 0, self.keys.size - 1)
        hit = (self.keys[j_c] == kq) & (keys != 0)
        return np.where(hit, j_c, -1)


def loo_values(table: Table, sum_logq: np.ndarray) -> np.ndarray:
    """Leave-one-out log-likelihood gain (nats) of each entry over the prior on its own occurrences.

    For an entry with count n in a row with total N and u distinct followers, each held-out
    occurrence is predicted by the row without it: Witten-Bell (n-1 + u'q)/(N-1+u'), where
    u' = u - 1 if the entry was a singleton.  ``q`` is the geometric mean of the prior over
    the entry's occurrences.  A singleton row predicts its only occurrence by the prior alone,
    so it is worth nothing: this is the deleted (Good-Turing-like) estimate of what an
    entry adds on unseen text, and the pruning order used by MDL-value pruning.
    """

    n = table.counts.astype(np.float64)
    row = np.repeat(np.arange(table.keys.size), table.distinct)
    big_n = table.total[row].astype(np.float64)
    u = table.distinct[row].astype(np.float64)
    mean_logq = sum_logq / n
    q = np.exp(mean_logq)
    u_held = np.where(n >= 2, u, u - 1.0)
    denom = big_n - 1.0 + u_held
    p = np.where(denom > 0, (n - 1.0 + u_held * q) / np.maximum(denom, 1e-12), q)
    return n * (np.log(np.maximum(p, 1e-300)) - mean_logq)


@dataclass
class Prior:
    """Dense Witten-Bell chain over orders 0, 1, 2 (stored as sparse counts)."""

    uni: np.ndarray  # (256,)
    c1: np.ndarray  # (257, 256) counts after previous byte (257 = BOS)
    c2: np.ndarray  # (257*257, 256) counts after (byte-2, byte-1)

    @staticmethod
    def build(ctx: Contexts) -> "Prior":
        uni = np.bincount(ctx.targets, minlength=256).astype(np.int64)
        c1 = np.zeros((257, 256), dtype=np.int32)
        np.add.at(c1, (ctx.c1, ctx.targets), 1)
        c2 = np.zeros((257 * 257, 256), dtype=np.int32)
        np.add.at(c2, (ctx.c2 * 257 + ctx.c1, ctx.targets), 1)
        return Prior(uni, c1, c2)

    def storage_bits(self) -> int:
        bits = 256 * 32
        for table in (self.c1, self.c2):
            bits += int(np.count_nonzero(table)) * ENTRY_BITS + int(np.count_nonzero(table.sum(1))) * KEY_BITS
        return bits

    def dense(self) -> np.ndarray:
        """(257*257, 256) float32 order-2 Witten-Bell distributions."""

        p0 = (self.uni + 1.0) / (self.uni.sum() + 256.0)
        n1 = self.c1.sum(1, keepdims=True).astype(np.float64)
        u1 = (self.c1 > 0).sum(1, keepdims=True).astype(np.float64)
        p1 = np.where(n1 + u1 > 0, (self.c1 + u1 * p0) / np.maximum(n1 + u1, 1), p0)
        out = np.empty((257 * 257, 256), dtype=np.float32)
        step = 257 * 16
        for a in range(0, 257 * 257, step):
            block = self.c2[a : a + step].astype(np.float64)
            lower = p1[np.arange(a, min(a + step, 257 * 257)) % 257]
            n2 = block.sum(1, keepdims=True)
            u2 = (block > 0).sum(1, keepdims=True)
            out[a : a + step] = np.where(n2 + u2 > 0, (block + u2 * lower) / np.maximum(n2 + u2, 1), lower)
        return out


@dataclass
class Store:
    prior: Prior
    tables: dict[str, Table]
    min_count: int = 1

    def storage_bits(self, models: tuple[str, ...], min_count: int = 1) -> int:
        return self.prior.storage_bits() + sum(self.tables[m].storage_bits_at(min_count) for m in models)

    def pruned(self, min_count: int, models: tuple[str, ...]) -> "Store":
        """Prune the given models' tables; the order-0..2 prior is never pruned (as in track D)."""

        return Store(self.prior, {m: self.tables[m].pruned(min_count) for m in models}, min_count)

    def storage_bits_at_value(self, models: tuple[str, ...], threshold: float) -> int:
        return self.prior.storage_bits() + sum(self.tables[m].storage_bits_at_value(threshold) for m in models)

    def pruned_by_value(self, threshold: float, models: tuple[str, ...]) -> "Store":
        return Store(self.prior, {m: self.tables[m].pruned_by_value(threshold) for m in models}, 0)


def build_store(ctx: Contexts, models: tuple[str, ...] = ALL_MODELS, mask: np.ndarray | None = None,
                valued: bool = False) -> Store:
    sub = ctx if mask is None else ctx.take(np.nonzero(mask)[0])
    prior = Prior.build(sub)
    if not valued:
        return Store(prior, {m: Table.build(sub.keys[m], sub.targets) for m in models})
    logq = np.log(np.maximum(prior.dense()[sub.c2 * 257 + sub.c1, sub.targets], 1e-30))
    return Store(prior, {m: Table.build_valued(sub.keys[m], sub.targets, logq) for m in models})


def choose_price(store: Store, arm: str, budget_bits: float, steps: int = 36) -> float:
    """Smallest price (nats per bit) whose per-row knapsack pruning fits the budget (log bisection)."""

    models = ARMS[arm]["models"]
    fixed = param_count(arm) * PARAM_BITS + store.prior.storage_bits()

    def bits(price: float) -> int:
        return fixed + sum(store.tables[m].knapsack_keep(price)[1] for m in models)

    lo, hi = -12.0, 12.0  # log price
    if bits(float(np.exp(lo))) <= budget_bits:
        return float(np.exp(lo))
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if bits(float(np.exp(mid))) <= budget_bits:
            hi = mid
        else:
            lo = mid
    return float(np.exp(hi))


def choose_value_threshold(store: Store, arm: str, budget_bits: float) -> float:
    """Smallest leave-one-out value threshold whose pruned tables fit the budget (bisection)."""

    models = ARMS[arm]["models"]
    fixed = param_count(arm) * PARAM_BITS
    if store.storage_bits_at_value(models, -np.inf) + fixed <= budget_bits:
        return -np.inf
    values = np.concatenate([store.tables[m].value for m in models])
    lo, hi = float(values.min()), float(values.max()) + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if store.storage_bits_at_value(models, mid) + fixed <= budget_bits:
            hi = mid
        else:
            lo = mid
    return hi


def param_count(arm: str) -> int:
    m = len(ARMS[arm]["models"])
    n = (N_S1 + N_S2) * (m + 1) + m * N_BUCKETS
    if ARMS[arm]["apm"]:
        n += (257 + N_S1) * APM_KNOTS
    return n


def choose_min_count(store: Store, arm: str, budget_bits: float | None) -> int:
    if budget_bits is None:
        return 1
    models = ARMS[arm]["models"]
    fixed = param_count(arm) * PARAM_BITS
    for candidate in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 1024, 2048, 4096):
        if store.storage_bits(models, candidate) + fixed <= budget_bits:
            return candidate
    return 1 << 20


# ---------------------------------------------------------------------------
# Per-chunk evidence
# ---------------------------------------------------------------------------


def _ragged(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    total = int(lens.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = starts - (np.cumsum(lens) - lens)
    return np.arange(total, dtype=np.int64) + np.repeat(offsets, lens)


@dataclass
class Chunk:
    logq: np.ndarray  # (B, 256) float32
    ent_pos: np.ndarray  # int64
    ent_m: np.ndarray  # int64
    ent_y: np.ndarray  # int64
    delta: np.ndarray  # float32
    bucket: np.ndarray  # (B, M) int64
    n_total: np.ndarray  # (B, M)
    y: np.ndarray
    s1: np.ndarray
    s2: np.ndarray
    c1: np.ndarray
    p_models_target: np.ndarray  # (B, M) p_m(y*) (Witten-Bell toward q)
    q_target: np.ndarray


def make_chunk(store: Store, dense_prior: np.ndarray, ctx: Contexts, models: tuple[str, ...]) -> Chunk:
    b = ctx.targets.shape[0]
    q = dense_prior[ctx.c2 * 257 + ctx.c1]  # (B, 256)
    m_count = len(models)
    n_total = np.zeros((b, m_count), dtype=np.int64)
    u_total = np.zeros((b, m_count), dtype=np.int64)
    pos_parts, m_parts, y_parts, n_parts = [], [], [], []
    order_evidence = np.zeros(b, dtype=np.int64)
    for mi, name in enumerate(models):
        table = store.tables[name]
        r = table.rows(ctx.keys[name])
        hit = r >= 0
        rr = r[hit]
        lens = table.distinct[rr]
        n_total[hit, mi] = table.total[rr]
        u_total[hit, mi] = lens
        e = _ragged(table.start[rr], lens)
        pos_parts.append(np.repeat(np.nonzero(hit)[0], lens))
        m_parts.append(np.full(e.size, mi, dtype=np.int64))
        y_parts.append((table.combined[e] & U64(255)).astype(np.int64))
        n_parts.append(table.counts[e])
    for name in GROUPS["orders"]:
        table = store.tables.get(name)
        if table is not None:
            order_evidence += (table.rows(ctx.keys[name]) >= 0).astype(np.int64)
    ent_pos = np.concatenate(pos_parts) if pos_parts else np.zeros(0, dtype=np.int64)
    ent_m = np.concatenate(m_parts) if m_parts else np.zeros(0, dtype=np.int64)
    ent_y = np.concatenate(y_parts) if y_parts else np.zeros(0, dtype=np.int64)
    ent_n = np.concatenate(n_parts).astype(np.float64) if n_parts else np.zeros(0)
    qv = q[ent_pos, ent_y].astype(np.float64)
    u_e = u_total[ent_pos, ent_m].astype(np.float64)
    delta = np.log1p(ent_n / (u_e * qv)).astype(np.float32)
    # p_m(y*) for the arithmetic ablation
    at_target = ent_y == ctx.targets[ent_pos]
    n_target = np.zeros((b, m_count))
    np.add.at(n_target, (ent_pos[at_target], ent_m[at_target]), ent_n[at_target])
    q_target = q[np.arange(b), ctx.targets].astype(np.float64)
    denom = (n_total + u_total).astype(np.float64)
    p_models = np.where(denom > 0, (n_target + u_total * q_target[:, None]) / np.maximum(denom, 1), q_target[:, None])
    s1 = match_bucket(ctx.match_tier) * (len(ORDERS) + 1) + np.minimum(order_evidence, len(ORDERS))
    return Chunk(
        logq=np.log(np.maximum(q, 1e-30)).astype(np.float32), ent_pos=ent_pos, ent_m=ent_m, ent_y=ent_y, delta=delta,
        bucket=np.digitize(n_total, COUNT_EDGES).astype(np.int64), n_total=n_total, y=ctx.targets.astype(np.int64),
        s1=s1, s2=ctx.s2.astype(np.int64), c1=ctx.c1.astype(np.int64), p_models_target=p_models, q_target=q_target,
    )


def chunks_for(store: Store, dense_prior: np.ndarray, ctx: Contexts, models: tuple[str, ...], size: int = 20_000):
    for a in range(0, ctx.targets.shape[0], size):
        yield make_chunk(store, dense_prior, ctx.take(np.arange(a, min(a + size, ctx.targets.shape[0]))), models)


# ---------------------------------------------------------------------------
# Mixers (torch, deterministic full-batch L-BFGS)
# ---------------------------------------------------------------------------


def _torch():
    import torch

    torch.set_num_threads(max(1, torch.get_num_threads()))
    return torch


class GeometricMixer:
    """s(y) = α log q(y) + Σ_m w_m δ_m(y), weights from selectors and count buckets."""

    def __init__(self, n_models: int, l2: float = 1e-4) -> None:
        torch = _torch()
        self.m = n_models
        self.l2 = l2
        self.a1 = torch.zeros(N_S1, n_models + 1, dtype=torch.float32, requires_grad=True)
        self.a2 = torch.zeros(N_S2, n_models + 1, dtype=torch.float32, requires_grad=True)
        self.cb = torch.zeros(n_models, N_BUCKETS, dtype=torch.float32, requires_grad=True)
        self.apm_c1 = torch.zeros(257, APM_KNOTS, dtype=torch.float32, requires_grad=True)
        self.apm_s1 = torch.zeros(N_S1, APM_KNOTS, dtype=torch.float32, requires_grad=True)
        self.use_apm = False

    def params(self) -> list:
        return [self.a1, self.a2, self.cb]

    def apm_params(self) -> list:
        return [self.apm_c1, self.apm_s1]

    def scores(self, ch: Chunk, tensors: dict):
        torch = _torch()
        b = ch.y.shape[0]
        s1, s2 = tensors["s1"], tensors["s2"]
        alpha = 1.0 + self.a1[s1, 0] + self.a2[s2, 0]
        weights = self.a1[s1, 1:] + self.a2[s2, 1:] + self.cb[torch.arange(self.m)[None, :], tensors["bucket"]]
        s = alpha[:, None] * tensors["logq"]
        contrib = weights.reshape(-1)[tensors["wi"]] * tensors["delta"]
        s = s.reshape(-1).index_add(0, tensors["flat"], contrib).reshape(b, 256)
        return s

    def apm(self, s, tensors: dict):
        torch = _torch()
        logp = s - torch.logsumexp(s, dim=1, keepdim=True)
        t = (logp.clamp(min=APM_LO, max=0.0) - APM_LO) / (-APM_LO) * (APM_KNOTS - 1)
        k = t.floor().clamp(max=APM_KNOTS - 2)
        frac = t - k
        k = k.long()
        c1 = tensors["c1"][:, None].expand_as(k)
        s1 = tensors["s1"][:, None].expand_as(k)
        g = (1 - frac) * (self.apm_c1[c1, k] + self.apm_s1[s1, k]) + frac * (self.apm_c1[c1, k + 1] + self.apm_s1[s1, k + 1])
        return logp + g

    def bits(self, ch: Chunk, tensors: dict, apm: bool | None = None):
        torch = _torch()
        s = self.scores(ch, tensors)
        if self.use_apm if apm is None else apm:
            s = self.apm(s, tensors)
        logz = torch.logsumexp(s, dim=1)
        target = s.gather(1, tensors["y"][:, None])[:, 0]
        return (logz - target) / math.log(2.0)

    def storage_params(self) -> int:
        n = sum(int(p.numel()) for p in self.params())
        if self.use_apm:
            n += sum(int(p.numel()) for p in self.apm_params())
        return n


class ArithmeticMixer:
    """p(y*) = Σ_m π_m p_m(y*) over the prior and every model; π from the same selectors."""

    def __init__(self, n_models: int, l2: float = 1e-4) -> None:
        torch = _torch()
        self.m = n_models
        self.l2 = l2
        self.a1 = torch.zeros(N_S1, n_models + 1, dtype=torch.float32, requires_grad=True)
        self.a2 = torch.zeros(N_S2, n_models + 1, dtype=torch.float32, requires_grad=True)
        self.cb = torch.zeros(n_models, N_BUCKETS, dtype=torch.float32, requires_grad=True)
        self.use_apm = False

    def params(self) -> list:
        return [self.a1, self.a2, self.cb]

    def bits(self, ch: Chunk, tensors: dict, apm: bool | None = None):
        torch = _torch()
        s1, s2 = tensors["s1"], tensors["s2"]
        logits = self.a1[s1] + self.a2[s2]
        logits = torch.cat([logits[:, :1], logits[:, 1:] + self.cb[torch.arange(self.m)[None, :], tensors["bucket"]]], dim=1)
        pi = torch.softmax(logits, dim=1)
        probs = torch.cat([tensors["q_target"][:, None], tensors["p_models"]], dim=1)
        return -torch.log2((pi * probs).sum(1).clamp_min(1e-30))

    def storage_params(self) -> int:
        return sum(int(p.numel()) for p in self.params())


def to_tensors(ch: Chunk) -> dict:
    """Compact per-entry indexes (int32: index into the (B, 256) scores and the (B, M) weights)."""

    torch = _torch()
    m = ch.bucket.shape[1]
    return {
        "logq": torch.from_numpy(ch.logq),
        "wi": torch.from_numpy((ch.ent_pos * m + ch.ent_m).astype(np.int32)),
        "delta": torch.from_numpy(ch.delta),
        "flat": torch.from_numpy((ch.ent_pos * 256 + ch.ent_y).astype(np.int32)),
        "bucket": torch.from_numpy(ch.bucket),
        "y": torch.from_numpy(ch.y),
        "s1": torch.from_numpy(ch.s1),
        "s2": torch.from_numpy(ch.s2),
        "c1": torch.from_numpy(ch.c1),
        "p_models": torch.from_numpy(ch.p_models_target.astype(np.float32)),
        "q_target": torch.from_numpy(ch.q_target.astype(np.float32)),
    }


def _fit(params: list, loss_fn, n_positions: int, l2: float, max_iter: int) -> list[float]:
    torch = _torch()
    optimizer = torch.optim.LBFGS(params, lr=1.0, max_iter=max_iter, history_size=20, line_search_fn="strong_wolfe",
                                  tolerance_grad=1e-7, tolerance_change=1e-9)
    trace: list[float] = []

    def closure():
        optimizer.zero_grad()
        total = loss_fn() / n_positions + l2 * sum((p * p).sum() for p in params)
        total.backward()
        trace.append(float(total.detach()))
        return total

    optimizer.step(closure)
    return trace


def fit_mixer(mixer, chunks: list[Chunk], max_iter: int = 60, apm: bool = False) -> dict:
    """Full-batch L-BFGS on the mixer weights, then (optionally) on the APM with the mixer fixed."""

    tensors = [to_tensors(ch) for ch in chunks]
    for ch in chunks:  # keep only what the losses read; the tensors hold the entry data
        ch.ent_pos = ch.ent_m = ch.ent_y = ch.delta = None
    n = sum(ch.y.shape[0] for ch in chunks)
    started = time.time()

    def loss(use_apm: bool):
        return lambda: sum(mixer.bits(ch, t, apm=use_apm).sum() for ch, t in zip(chunks, tensors))

    trace = _fit(mixer.params(), loss(False), n, mixer.l2, max_iter)
    info = {"mixer_trace": trace[-5:], "mixer_evals": len(trace), "mixer_seconds": time.time() - started}
    if apm:
        for p in mixer.params():
            p.requires_grad_(False)
        started = time.time()
        mixer.use_apm = True
        trace = _fit(mixer.apm_params(), loss(True), n, mixer.l2, max_iter)
        info.update({"apm_trace": trace[-5:], "apm_evals": len(trace), "apm_seconds": time.time() - started})
    return info


def score(mixer, store: Store, dense_prior: np.ndarray, ctx: Contexts, models: tuple[str, ...], apm: bool | None = None) -> np.ndarray:
    return score_both(mixer, store, dense_prior, ctx, models, apm=apm)[0]


def score_both(mixer, store: Store, dense_prior: np.ndarray, ctx: Contexts, models: tuple[str, ...],
               apm: bool | None = None) -> tuple[np.ndarray, np.ndarray | None]:
    """Bits per position (with the mixer's APM setting, or ``apm``), plus bits without APM
    from the same chunk evidence when the mixer has an APM stage."""

    torch = _torch()
    out, out_raw = [], []
    raw = bool(getattr(mixer, "use_apm", False)) and apm is None
    with torch.no_grad():
        for ch in chunks_for(store, dense_prior, ctx, models):
            t = to_tensors(ch)
            out.append(mixer.bits(ch, t, apm=apm).numpy())
            if raw:
                out_raw.append(mixer.bits(ch, t, apm=False).numpy())
    cat = lambda parts: np.concatenate(parts) if parts else np.zeros(0)  # noqa: E731
    return cat(out), (cat(out_raw) if raw else None)


# ---------------------------------------------------------------------------
# Two-fold static protocol (track D's): counts on train-A, mixer on train-B,
# final counts on all of train
# ---------------------------------------------------------------------------


@dataclass
class Fitted:
    arm: str
    mixer: object
    store: Store
    dense: np.ndarray
    min_count: int
    notes: dict

    def storage_bits(self) -> int:
        return self.store.storage_bits(ARMS[self.arm]["models"]) + self.mixer.storage_params() * PARAM_BITS


def fit_arm(arm: str, full: Store, part_a: Store, ctx_b: Contexts, budget_bits: float | None,
            mixer_positions: int = 250_000, seed: int = 0, max_iter: int = 60, pruning: str = "count") -> Fitted:
    spec = ARMS[arm]
    models = spec["models"]
    value_threshold = None
    if pruning == "knapsack" and budget_bits is not None:
        value_threshold = choose_price(full, arm, budget_bits)
        min_count = 0
        store = Store(full.prior, {m: full.tables[m].pruned_by_price(value_threshold) for m in models}, 0)
        store_a = Store(part_a.prior, {m: part_a.tables[m].pruned_by_price(value_threshold) for m in models}, 0)
    elif pruning == "value" and budget_bits is not None:
        value_threshold = choose_value_threshold(full, arm, budget_bits)
        min_count = 0
        store = full.pruned_by_value(value_threshold, models)
        store_a = part_a.pruned_by_value(value_threshold, models)
    else:
        min_count = choose_min_count(full, arm, budget_bits)
        store = full.pruned(min_count, models)
        store_a = part_a.pruned(min_count, models)
    rng = np.random.default_rng(seed)
    n_b = ctx_b.targets.shape[0]
    keep = np.sort(rng.choice(n_b, size=min(mixer_positions, n_b), replace=False))
    dense_a = store_a.prior.dense()
    chunks = list(chunks_for(store_a, dense_a, ctx_b.take(keep), models))
    entries = sum(c.ent_pos.size for c in chunks)
    mixer = GeometricMixer(len(models)) if spec["mix"] == "geometric" else ArithmeticMixer(len(models))
    info = fit_mixer(mixer, chunks, max_iter=max_iter, apm=spec["apm"])
    info.update({"min_count": min_count, "pruning": pruning, "value_threshold": value_threshold,
                 "mixer_positions": int(keep.size),
                 "entries_per_position": float(entries / max(keep.size, 1))})
    return Fitted(arm, mixer, store, store.prior.dense(), min_count, info)
