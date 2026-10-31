"""Field language model on real bytes: exact charts, knowledge tables, joins, mixer.

The model predicts byte ``x[t]`` from the strictly earlier bytes of its context
segment.  Everything that addresses information is an exact integer join:

- **Charts** are exact causal counters: line, column, indentation, bracket
  depth, word boundaries, and the byte directly above in the previous line (a
  2D sheet translation by (0, 1) on (column, line)).
- **Content charts** are hashed substrings: order-k contexts, the current word
  prefix and the previous word.  Hashes are canonical polynomial hashes mod
  2**64, so equal substrings get equal keys wherever they occur.
- **Knowledge tables** map a head's key to counts of the next byte, built on
  training text (counted, not learned).  Content orders form a Witten-Bell
  interpolated chain; every other stored head is smoothed toward chain order 2.
- **In-context joins**: the match head is the induction-head join (most recent
  earlier occurrence of the current k-gram in the segment, longest k wins) and
  predicts the byte that followed it; the column-above head copies the byte
  above.  Their reliabilities are small counted tables.
- **Mixer**: an arithmetic mixture of head distributions with softmax weights
  selected by a small discrete state (match tier, order-4 evidence, line
  start).  This is the only gradient-learned part.

Each head is a normalized distribution over 256 bytes and the mixture is
convex, so bits per byte are proper.  Evaluation needs only the probability of
the target (and of a copy head's predicted byte), so it is fully vectorized.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

U64 = np.uint64
PRIME = U64(0x100000001B3)  # odd, so invertible mod 2**64
ALNUM = np.zeros(256, dtype=bool)
for _c in b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_":
    ALNUM[_c] = True
OPEN = np.zeros(256, dtype=np.int64)
for _c in b"([{":
    OPEN[_c] = 1
CLOSE = np.zeros(256, dtype=np.int64)
for _c in b")]}":
    CLOSE[_c] = 1

CHAIN_ORDERS = (1, 2, 3, 4, 6)
MATCH_ORDERS = (20, 12, 8, 5)  # longest first
STORED_HEADS = ("word1", "word2", "struct_col", "struct_depth", "sheet2d", "sheet2d_b")
HEADS = tuple(f"order{k}" for k in CHAIN_ORDERS) + STORED_HEADS + ("match", "colabove")
ENTRY_BITS = 24  # next byte (8) + count (16)
KEY_BITS = 56  # key hash (32) + total (16) + distinct (8)
MIXER_WEIGHT_BITS = 32


def _mix(*parts: np.ndarray) -> np.ndarray:
    """splitmix64-style combination of uint64 arrays (wrapping arithmetic)."""

    with np.errstate(over="ignore"):
        z = np.zeros_like(parts[0], dtype=U64)
        for i, p in enumerate(parts):
            z = (z ^ p.astype(U64)) * U64(0x9E3779B97F4A7C15) + U64(i + 1)
            z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
            z = z ^ (z >> U64(31))
    return z


def _inverse_mod_2_64(a: int) -> int:
    x = 1
    for _ in range(7):
        x = (x * (2 - a * x)) % (1 << 64)
    return x


class SubstringHasher:
    """Canonical substring hashes: h(a, b) = sum_{j<b-a} (x[a+j]+1) P^j mod 2**64."""

    def __init__(self, x: np.ndarray) -> None:
        n = x.shape[0]
        with np.errstate(over="ignore"):
            powers = np.cumprod(np.concatenate([[U64(1)], np.full(n, PRIME, dtype=U64)]), dtype=U64)
            inv = U64(_inverse_mod_2_64(int(PRIME)))
            inv_powers = np.cumprod(np.concatenate([[U64(1)], np.full(n, inv, dtype=U64)]), dtype=U64)
            weighted = (x.astype(U64) + U64(1)) * powers[:n]
            self.prefix = np.concatenate([[U64(0)], np.cumsum(weighted, dtype=U64)])
        self.inv_powers = inv_powers

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        with np.errstate(over="ignore"):
            raw = (self.prefix[b] - self.prefix[a]) * self.inv_powers[a]
        return _mix(raw, (b - a).astype(U64))


@dataclass
class Charts:
    """Per-target-position features for predicting x[t] from x[seg_start[t]:t]."""

    positions: np.ndarray  # target positions t (seg-relative index >= 1)
    targets: np.ndarray  # x[t]
    keys: dict[str, np.ndarray]  # head -> uint64 key per position (0 = invalid)
    above_byte: np.ndarray  # int, -1 if none
    same_indent: np.ndarray  # bool
    column: np.ndarray
    match_pred: np.ndarray  # int, -1 if none
    match_tier: np.ndarray  # 0 none, 1..len(MATCH_ORDERS) (1 = longest)
    categories: dict[str, np.ndarray] = field(default_factory=dict)


def segment_starts(n: int, length: int | None) -> np.ndarray:
    idx = np.arange(n)
    if length is None:
        return np.zeros(n, dtype=np.int64)
    return (idx // length) * length


def compute_charts(x: np.ndarray, seg_len: int | None, joins: bool = True) -> Charts:
    """Exact causal charts with resets at segment starts (vectorized).

    ``joins=False`` skips the in-context match join (not needed to build tables).
    """

    # int64 bytes: numpy >= 2.5 refuses to mix uint8 arrays with sentinels like -1
    x = x.astype(np.int64)
    n = x.shape[0]
    idx = np.arange(n, dtype=np.int64)
    seg = segment_starts(n, seg_len)
    local = idx - seg
    pos = idx[local >= 1]
    tgt = x[pos]
    hasher = SubstringHasher(x)

    # line starts (segment start counts as a line start)
    nl_idx = np.where(x == 10, idx, -1)
    last_nl = np.maximum.accumulate(nl_idx)
    last_nl_before = np.concatenate([[-1], last_nl[:-1]])
    line_start = np.maximum(last_nl_before + 1, seg)
    column = idx - line_start
    # indentation so far on the current line
    ws = (x == 32) | (x == 9)
    nonws_idx = np.where(~ws, idx, n)
    next_nonws = np.minimum.accumulate(nonws_idx[::-1])[::-1]
    next_nonws = np.concatenate([next_nonws, [n]])
    indent = np.minimum(next_nonws[line_start], idx) - line_start
    # previous line: start, length, indentation
    prev_nl = line_start - 1  # the newline ending the previous line, if in segment
    has_prev = prev_nl >= seg
    prev_nl_c = np.clip(prev_nl, 0, n - 1)
    prev_start = np.where(has_prev, line_start[prev_nl_c], 0)
    prev_len = np.where(has_prev, prev_nl - prev_start, -1)
    above = np.where(has_prev & (column < prev_len), prev_start + column, -1)
    above_byte = np.where(above >= 0, x[np.clip(above, 0, n - 1)], -1)
    above_next = np.where((above >= 0) & (column + 1 < prev_len), x[np.clip(above + 1, 0, n - 1)], -1)
    prev_indent = np.where(has_prev, np.minimum(next_nonws[prev_start], prev_nl) - prev_start, -1)
    same_indent = has_prev & (prev_indent == indent)
    # bracket depth, reset per segment
    delta = OPEN[x] - CLOSE[x]
    depth_after = np.cumsum(delta)
    depth_at = np.concatenate([[0], depth_after[:-1]])
    base = np.where(seg > 0, depth_after[np.clip(seg - 1, 0, n - 1)], 0)
    depth = np.clip(depth_at - base, -4, 60)
    # word prefix and previous word
    alnum = ALNUM[x]
    nonalnum_idx = np.where(~alnum, idx, -1)
    last_nonalnum = np.maximum.accumulate(nonalnum_idx)
    last_nonalnum_before = np.concatenate([[-1], last_nonalnum[:-1]])
    word_start = np.maximum(last_nonalnum_before + 1, seg)
    word_start = np.maximum(word_start, idx - 32)
    alnum_idx = np.where(alnum, idx, -1)
    last_alnum = np.maximum.accumulate(alnum_idx)
    ws_c = np.clip(word_start, 0, n - 1)
    prev_word_end = np.where(word_start > 0, np.concatenate([[-1], last_alnum[:-1]])[ws_c] + 1, 0)
    has_prev_word = prev_word_end > seg
    pwe_c = np.clip(prev_word_end - 1, 0, n - 1)
    prev_word_start = np.maximum(np.maximum(last_nonalnum_before[pwe_c] + 1, seg), prev_word_end - 32)

    def at(a: np.ndarray) -> np.ndarray:
        return a[pos]

    x_prev1 = x[np.clip(pos - 1, 0, n - 1)].astype(U64)
    x_prev2 = np.where(local[pos] >= 2, x[np.clip(pos - 2, 0, n - 1)], 256).astype(U64)
    keys: dict[str, np.ndarray] = {}
    for k in CHAIN_ORDERS:
        valid = local[pos] >= k
        h = hasher(np.clip(pos - k, 0, n), pos)
        keys[f"order{k}"] = np.where(valid, _mix(h, np.full(pos.shape, k, dtype=U64)) | U64(1), U64(0))
    word = hasher(at(word_start), pos)
    prev_word = np.where(at(has_prev_word), hasher(at(prev_word_start), at(prev_word_end)), U64(7))
    keys["word1"] = _mix(word, U64(11) + np.zeros_like(word)) | U64(1)
    keys["word2"] = _mix(word, prev_word) | U64(1)
    keys["struct_col"] = _mix(np.minimum(at(column), 64).astype(U64), np.minimum(at(indent), 32).astype(U64), x_prev1) | U64(1)
    keys["struct_depth"] = _mix((at(depth) + 8).astype(U64), x_prev2, x_prev1) | U64(1)
    keys["sheet2d"] = _mix((at(above_byte) + 1).astype(U64), x_prev1) | U64(1)
    keys["sheet2d_b"] = _mix((at(above_byte) + 1).astype(U64), (at(above_next) + 1).astype(U64), x_prev1) | U64(1)

    # induction/match join: most recent earlier occurrence (same segment) of the
    # k-gram ending at t, longest k first; predict the byte that followed it.
    match_pred = np.full(pos.shape, -1, dtype=np.int64)
    match_tier = np.zeros(pos.shape, dtype=np.int64)
    all_t = idx
    for tier, k in enumerate(MATCH_ORDERS if joins else (), start=1):
        valid_all = (local >= k)
        h_all = np.where(valid_all, hasher(np.clip(all_t - k, 0, n), all_t), U64(0))
        # key includes segment start so joins stay inside the segment
        key_all = np.where(valid_all, _mix(h_all, seg.astype(U64)), U64(0))
        order = np.lexsort((all_t, key_all))
        sorted_keys = key_all[order]
        prev_same = np.full(n, -1, dtype=np.int64)
        same = (sorted_keys[1:] == sorted_keys[:-1]) & (sorted_keys[1:] != 0)
        prev_same[order[1:][same]] = order[:-1][same]
        cand = prev_same[pos]
        take = (cand >= 0) & (match_tier == 0)
        match_pred = np.where(take, x[np.clip(cand, 0, n - 1)].astype(np.int64), match_pred)
        match_tier = np.where(take, tier, match_tier)

    charts = Charts(
        positions=pos,
        targets=tgt,
        keys=keys,
        above_byte=at(above_byte).astype(np.int64),
        same_indent=at(same_indent),
        column=at(column),
        match_pred=match_pred,
        match_tier=match_tier,
    )
    # categories for gap decomposition
    cls = np.full(pos.shape, 4)  # 0 space/newline, 1 punct/markup, 2 word-initial alnum, 3 word-internal alnum, 4 other
    t_bytes = x[pos]
    is_space = (t_bytes == 32) | (t_bytes == 10) | (t_bytes == 9) | (t_bytes == 13)
    is_alnum = ALNUM[t_bytes]
    initial = is_alnum & (at(word_start) == pos)
    cls = np.where(is_space, 0, cls)
    cls = np.where(~is_space & ~is_alnum & (t_bytes < 128), 1, cls)
    cls = np.where(initial, 2, cls)
    cls = np.where(is_alnum & ~initial, 3, cls)
    charts.categories["byte_class"] = cls
    charts.categories["copyable"] = (match_pred == tgt)
    charts.categories["match_tier"] = match_tier
    return charts


# ---------------------------------------------------------------------------
# Knowledge tables
# ---------------------------------------------------------------------------


@dataclass
class CountTable:
    combined: np.ndarray  # sorted uint64 (key << 8 | byte)
    counts: np.ndarray  # int64
    key_sorted: np.ndarray  # sorted unique keys (uint64, key << 8 >> 8 form)
    key_total: np.ndarray
    key_distinct: np.ndarray

    @staticmethod
    def build(keys: np.ndarray, targets: np.ndarray, min_count: int = 1) -> "CountTable":
        valid = keys != 0
        combined = (keys[valid] << U64(8)) | targets[valid].astype(U64)
        uniq, counts = np.unique(combined, return_counts=True)
        if min_count > 1:
            keep = counts >= min_count
            uniq, counts = uniq[keep], counts[keep]
        return CountTable.from_entries(uniq, counts.astype(np.int64))

    @staticmethod
    def from_entries(combined: np.ndarray, counts: np.ndarray) -> "CountTable":
        key_of = combined >> U64(8)
        if key_of.size:
            starts = np.concatenate([[0], np.where(key_of[1:] != key_of[:-1])[0] + 1])
            key_sorted = key_of[starts]
            key_total = np.add.reduceat(counts, starts)
            key_distinct = np.diff(np.concatenate([starts, [key_of.size]]))
        else:
            key_sorted = np.zeros(0, dtype=U64)
            key_total = np.zeros(0, dtype=np.int64)
            key_distinct = np.zeros(0, dtype=np.int64)
        return CountTable(combined, counts, key_sorted, key_total, key_distinct)

    def pruned(self, min_count: int) -> "CountTable":
        keep = self.counts >= min_count
        return CountTable.from_entries(self.combined[keep], self.counts[keep])

    def entries(self) -> int:
        return int(self.counts.size)

    def storage_bits(self) -> int:
        return self.entries() * ENTRY_BITS + int(self.key_sorted.size) * KEY_BITS

    def lookup(self, keys: np.ndarray, byte: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(count of byte after key, key total, key distinct); zeros if unseen."""

        q = (keys << U64(8)) | byte.astype(U64)
        i = np.searchsorted(self.combined, q)
        i_c = np.clip(i, 0, max(self.combined.size - 1, 0))
        hit = (self.combined.size > 0) & (self.combined[i_c] == q) & (keys != 0)
        c = np.where(hit, self.counts[i_c] if self.counts.size else 0, 0)
        kq = (keys << U64(8)) >> U64(8)
        j = np.searchsorted(self.key_sorted, kq)
        j_c = np.clip(j, 0, max(self.key_sorted.size - 1, 0))
        khit = (self.key_sorted.size > 0) & (self.key_sorted[j_c] == kq) & (keys != 0)
        n = np.where(khit, self.key_total[j_c] if self.key_total.size else 0, 0)
        u = np.where(khit, self.key_distinct[j_c] if self.key_distinct.size else 0, 0)
        return c, n, u


def witten_bell(c: np.ndarray, n: np.ndarray, u: np.ndarray, lower: np.ndarray) -> np.ndarray:
    denom = n + u
    return np.where(denom > 0, (c + u * lower) / np.maximum(denom, 1), lower)


@dataclass
class Tables:
    unigram: np.ndarray  # counts per byte (256)
    stored: dict[str, CountTable]

    def storage_bits(self) -> int:
        return 256 * 32 + sum(t.storage_bits() for t in self.stored.values())

    def entries(self) -> int:
        return 256 + sum(t.entries() for t in self.stored.values())


def build_tables(x: np.ndarray) -> Tables:
    charts = compute_charts(x, None, joins=False)
    unigram = np.bincount(x, minlength=256).astype(np.int64)
    stored = {}
    for head in [f"order{k}" for k in CHAIN_ORDERS] + list(STORED_HEADS):
        stored[head] = CountTable.build(charts.keys[head], charts.targets)
    return Tables(unigram, stored)


def prune_tables(tables: Tables, min_count: int, protected: tuple[str, ...] = ("order1", "order2")) -> Tables:
    return Tables(
        tables.unigram,
        {h: (t if h in protected or min_count <= 1 else t.pruned(min_count)) for h, t in tables.stored.items()},
    )


def chain_probability(tables: Tables, charts: Charts, byte: np.ndarray) -> dict[int, np.ndarray]:
    """Witten-Bell interpolated chain probabilities of ``byte`` at each order."""

    p = (tables.unigram[byte] + 1.0) / (tables.unigram.sum() + 256.0)
    out = {0: p}
    for k in CHAIN_ORDERS:
        c, n, u = tables.stored[f"order{k}"].lookup(charts.keys[f"order{k}"], byte)
        p = witten_bell(c, n, u, p)
        out[k] = p
    return out


def head_probabilities(tables: Tables, charts: Charts, hits: "HitTables") -> np.ndarray:
    """(positions, heads) probability of the target under every head."""

    y = charts.targets
    chain = chain_probability(tables, charts, y)
    cols = [chain[k] for k in CHAIN_ORDERS]
    base2 = chain[2]
    for head in STORED_HEADS:
        c, n, u = tables.stored[head].lookup(charts.keys[head], y)
        cols.append(witten_bell(c, n, u, base2))
    top = chain[CHAIN_ORDERS[-1]]
    # copy heads: mass p_hit on the predicted byte, rest spread by the chain
    for pred, p_hit in (
        (charts.match_pred, hits.match_p(charts)),
        (charts.above_byte, hits.above_p(charts)),
    ):
        has = pred >= 0
        pred_c = np.clip(pred, 0, 255)
        p_pred = chain_probability(tables, charts, pred_c)[CHAIN_ORDERS[-1]]
        rest = np.where(y == pred_c, p_hit, (1.0 - p_hit) * top / np.maximum(1.0 - p_pred, 1e-12))
        cols.append(np.where(has, rest, top))
    return np.clip(np.stack(cols, axis=1), 1e-12, 1.0)


def _byte_class(b: np.ndarray) -> np.ndarray:
    b = np.clip(b, 0, 255)
    alpha = ((b >= 65) & (b <= 90)) | ((b >= 97) & (b <= 122))
    digit = (b >= 48) & (b <= 57)
    space = (b == 32) | (b == 10) | (b == 9)
    return np.where(alpha, 0, np.where(digit, 1, np.where(space, 2, np.where(b < 128, 3, 4))))


@dataclass
class HitTables:
    """Counted reliabilities of the two copy heads (Beta(1,1) prior)."""

    match_hits: np.ndarray = field(default_factory=lambda: np.ones((len(MATCH_ORDERS) + 1, 5)))
    match_total: np.ndarray = field(default_factory=lambda: np.full((len(MATCH_ORDERS) + 1, 5), 2.0))
    above_hits: np.ndarray = field(default_factory=lambda: np.ones((2, 5, 4)))
    above_total: np.ndarray = field(default_factory=lambda: np.full((2, 5, 4), 2.0))

    @staticmethod
    def _col_bucket(col: np.ndarray) -> np.ndarray:
        return np.digitize(col, [1, 4, 16])

    def fit(self, charts: Charts) -> "HitTables":
        has = charts.match_pred >= 0
        cls = _byte_class(charts.match_pred)
        np.add.at(self.match_total, (charts.match_tier[has], cls[has]), 1.0)
        np.add.at(self.match_hits, (charts.match_tier[has], cls[has]), (charts.match_pred == charts.targets)[has].astype(float))
        has = charts.above_byte >= 0
        idx = (charts.same_indent[has].astype(int), _byte_class(charts.above_byte)[has], self._col_bucket(charts.column)[has])
        np.add.at(self.above_total, idx, 1.0)
        np.add.at(self.above_hits, idx, (charts.above_byte == charts.targets)[has].astype(float))
        return self

    def match_p(self, charts: Charts) -> np.ndarray:
        cls = _byte_class(charts.match_pred)
        return (self.match_hits / self.match_total)[charts.match_tier, cls]

    def above_p(self, charts: Charts) -> np.ndarray:
        idx = (charts.same_indent.astype(int), _byte_class(charts.above_byte), self._col_bucket(charts.column))
        return (self.above_hits / self.above_total)[idx]

    def storage_bits(self) -> int:
        return int(self.match_hits.size + self.above_hits.size) * 2 * 32


def selector_state(tables: Tables, charts: Charts) -> np.ndarray:
    """Mixer state: match tier (5) x order-4 evidence (4) x line start (2) = 40."""

    _, n4, _ = tables.stored["order4"].lookup(charts.keys["order4"], charts.targets)
    evidence = np.digitize(n4, [1, 4, 32])
    return (charts.match_tier * 4 + evidence) * 2 + (charts.column == 0).astype(int)


N_STATES = (len(MATCH_ORDERS) + 1) * 4 * 2


def fit_mixer(probs: np.ndarray, states: np.ndarray, steps: int = 400, lr: float = 0.05, seed: int = 0):
    """Maximize sum log sum_h softmax(W[s])_h p_h (arithmetic mixture)."""

    import torch

    torch.manual_seed(seed)
    p = torch.tensor(probs, dtype=torch.float64)
    s = torch.tensor(states)
    weights = torch.zeros(N_STATES, probs.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([weights], lr=lr)
    for _ in range(steps):
        mix = (torch.softmax(weights[s], dim=-1) * p).sum(-1)
        loss = -torch.log2(mix).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return weights.detach().numpy(), float(loss.detach())


def mixture_bits(probs: np.ndarray, states: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w = weights[states]
    w = np.exp(w - w.max(-1, keepdims=True))
    w /= w.sum(-1, keepdims=True)
    return -np.log2((w * probs).sum(-1))


@dataclass
class FieldModel:
    tables: Tables
    hits: HitTables
    weights: np.ndarray
    min_count: int
    notes: dict = field(default_factory=dict)

    def storage_bits(self) -> int:
        return self.tables.storage_bits() + self.hits.storage_bits() + self.weights.size * MIXER_WEIGHT_BITS

    def bits(self, x: np.ndarray, seg_len: int) -> tuple[np.ndarray, Charts]:
        charts = compute_charts(x, seg_len)
        probs = head_probabilities(self.tables, charts, self.hits)
        states = selector_state(self.tables, charts)
        return mixture_bits(probs, states, self.weights), charts

    def ngram_bits(self, x: np.ndarray, seg_len: int) -> np.ndarray:
        """The Witten-Bell n-gram chain alone (the count-based baseline)."""

        charts = compute_charts(x, seg_len)
        return -np.log2(chain_probability(self.tables, charts, charts.targets)[CHAIN_ORDERS[-1]])


def train_field_model(
    train: np.ndarray,
    budget_bits: float | None,
    mixer_fraction: float = 0.1,
    mixer_positions: int = 400_000,
    seg_len: int = 256,
    seed: int = 0,
    cache: dict | None = None,
) -> FieldModel:
    """Two-fold protocol: counts on train-A, hit tables and mixer on train-B, final counts on all of train.

    With a storage budget, a global minimum count prunes every stored table
    except orders 1-2; the same threshold is applied to the train-A tables used
    to fit the mixer.
    """

    started = time.time()
    split = int(train.shape[0] * (1.0 - mixer_fraction))
    part_a, part_b = train[:split], train[split:]
    # unpruned tables and train-B charts can be shared across budgets
    cache = {} if cache is None else cache
    if "full" not in cache:
        cache["full"] = build_tables(train)
        cache["a"] = build_tables(part_a)
        cache["charts_b"] = compute_charts(part_b, seg_len)
    full = cache["full"]
    min_count = 1
    if budget_bits is not None:
        fixed = HitTables().storage_bits() + N_STATES * len(HEADS) * MIXER_WEIGHT_BITS
        for candidate in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024):
            if prune_tables(full, candidate).storage_bits() + fixed <= budget_bits:
                min_count = candidate
                break
        else:
            min_count = 1024
    full = prune_tables(full, min_count)
    tables_a = prune_tables(cache["a"], min_count)
    charts_b = cache["charts_b"]
    rng = np.random.default_rng(seed)
    if charts_b.targets.size > mixer_positions:
        keep = np.sort(rng.choice(charts_b.targets.size, size=mixer_positions, replace=False))
    else:
        keep = np.arange(charts_b.targets.size)
    hits = HitTables().fit(charts_b)
    probs = head_probabilities(tables_a, charts_b, hits)[keep]
    states = selector_state(tables_a, charts_b)[keep]
    weights, mixer_loss = fit_mixer(probs, states, seed=seed)
    model = FieldModel(full, hits, weights, min_count)
    model.notes = {
        "min_count": min_count,
        "mixer_train_bits": mixer_loss,
        "train_a_bytes": int(split),
        "train_b_bytes": int(part_b.shape[0]),
        "seconds": time.time() - started,
    }
    return model


def rebuild_with_edit(model: FieldModel, edited_train: np.ndarray, edited_tables: Tables | None = None) -> tuple[FieldModel, dict]:
    """Knowledge edit = recount the tables from the edited source; mixer and hit tables unchanged."""

    started = time.time()
    tables = prune_tables(edited_tables if edited_tables is not None else build_tables(edited_train), model.min_count)
    changed = 0
    total = 0
    for head, old in model.tables.stored.items():
        new = tables.stored[head]
        common, i_old, i_new = np.intersect1d(old.combined, new.combined, return_indices=True)
        same = int((old.counts[i_old] == new.counts[i_new]).sum())
        changed += (old.entries() - same) + (new.entries() - same)
        total += old.entries()
    edited = FieldModel(tables, model.hits, model.weights, model.min_count, dict(model.notes))
    return edited, {"seconds": time.time() - started, "changed_entries": int(changed), "total_entries": int(total)}
