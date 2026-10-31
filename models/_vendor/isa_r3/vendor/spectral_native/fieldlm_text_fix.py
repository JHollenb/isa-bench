"""Post-hoc fix for the real-text field LM: quotient-chart heads.

An exact join generalizes only to *identical* keys.  A quotient chart first
projects the context through a many-to-one map (a quotient of the content
chart) and then joins exactly, so every context in the same coset shares one
set of counts.  This is the algebra's own form of similarity-based smoothing:

- ``fold4``: the last 4 bytes with letter case folded;
- ``class6``: the last 6 bytes mapped to byte classes (upper, lower, digit,
  space, newline, punctuation kept exact), plus the exact previous byte;
- ``skip23`` / ``skip24``: sparse contexts that drop the previous byte (a
  wildcard axis): (x[t-2], x[t-3]) and (x[t-2], x[t-4]);
- ``fold_word``: the case-folded current word prefix.

Every head is Witten-Bell smoothed toward chain order 2, stored under the same
storage accounting, pruned by the same budget rule and mixed by the same
selector-gated mixer as the v1 heads.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from spectral_native import fieldlm_text as F

FIX_HEADS = ("fold4", "class6", "skip23", "skip24", "fold_word")
ALL_HEADS = F.HEADS + FIX_HEADS

FOLD = np.arange(256, dtype=np.uint64)
FOLD[65:91] += 32
CLASS = np.arange(256, dtype=np.uint64)
CLASS[65:91] = 300
CLASS[97:123] = 301
CLASS[48:58] = 302
CLASS[32] = 303
CLASS[9] = 303


def fix_keys(x: np.ndarray, seg_len: int | None) -> dict[str, np.ndarray]:
    """Keys for the quotient heads at the same target positions as compute_charts."""

    n = x.shape[0]
    idx = np.arange(n, dtype=np.int64)
    seg = F.segment_starts(n, seg_len)
    local = idx - seg
    pos = idx[local >= 1]
    loc = local[pos]

    def back(i: int, table: np.ndarray | None = None) -> np.ndarray:
        raw = x[np.clip(pos - i, 0, n - 1)].astype(np.uint64)
        mapped = table[raw] if table is not None else raw
        return np.where(loc >= i, mapped, np.uint64(512))

    keys = {
        "fold4": F._mix(back(1, FOLD), back(2, FOLD), back(3, FOLD), back(4, FOLD), np.full(pos.shape, 41, np.uint64)),
        "class6": F._mix(back(1), back(2, CLASS), back(3, CLASS), back(4, CLASS), back(5, CLASS), back(6, CLASS)),
        "skip23": F._mix(back(2), back(3), np.full(pos.shape, 23, np.uint64)),
        "skip24": F._mix(back(2), back(4), np.full(pos.shape, 24, np.uint64)),
    }
    # case-folded word prefix: rehash the folded bytes of [word_start, t)
    folded = FOLD[x].astype(np.uint8)
    hasher = F.SubstringHasher(folded)
    alnum = F.ALNUM[x]
    nonalnum_idx = np.where(~alnum, idx, -1)
    last_nonalnum_before = np.concatenate([[-1], np.maximum.accumulate(nonalnum_idx)[:-1]])
    word_start = np.maximum(np.maximum(last_nonalnum_before + 1, seg), idx - 32)
    keys["fold_word"] = F._mix(hasher(word_start[pos], pos), np.full(pos.shape, 99, np.uint64))
    return {k: v | np.uint64(1) for k, v in keys.items()}


def build_fix_tables(x: np.ndarray) -> dict[str, F.CountTable]:
    keys = fix_keys(x, None)
    n = x.shape[0]
    pos = np.arange(1, n)
    y = x[pos].astype(np.int64)
    return {h: F.CountTable.build(keys[h], y) for h in FIX_HEADS}


@dataclass
class FixModel:
    tables: F.Tables
    fix_tables: dict[str, F.CountTable]
    hits: F.HitTables
    weights: np.ndarray
    min_count: int
    notes: dict = field(default_factory=dict)

    def storage_bits(self) -> int:
        return (
            self.tables.storage_bits()
            + sum(t.storage_bits() for t in self.fix_tables.values())
            + self.hits.storage_bits()
            + self.weights.size * F.MIXER_WEIGHT_BITS
        )

    def probs(self, x: np.ndarray, seg_len: int) -> tuple[np.ndarray, np.ndarray, F.Charts]:
        charts = F.compute_charts(x, seg_len)
        return (*_probs(self.tables, self.fix_tables, self.hits, charts, fix_keys(x, seg_len)), charts)

    def bits(self, x: np.ndarray, seg_len: int) -> tuple[np.ndarray, F.Charts]:
        probs, states, charts = self.probs(x, seg_len)
        return F.mixture_bits(probs, states, self.weights), charts


def _probs(tables, fix_tables, hits, charts, keys) -> tuple[np.ndarray, np.ndarray]:
    base = F.head_probabilities(tables, charts, hits)
    order2 = F.chain_probability(tables, charts, charts.targets)[2]
    extra = []
    for head in FIX_HEADS:
        c, n, u = fix_tables[head].lookup(keys[head], charts.targets)
        extra.append(F.witten_bell(c, n, u, order2))
    probs = np.concatenate([base, np.clip(np.stack(extra, axis=1), 1e-12, 1.0)], axis=1)
    return probs, F.selector_state(tables, charts)


def _prune_fix(fix_tables: dict[str, F.CountTable], min_count: int) -> dict[str, F.CountTable]:
    return {h: (t if min_count <= 1 else t.pruned(min_count)) for h, t in fix_tables.items()}


def train_fix_model(
    train: np.ndarray,
    budget_bits: float | None,
    cache: dict,
    mixer_fraction: float = 0.1,
    mixer_positions: int = 400_000,
    seg_len: int = 256,
    seed: int = 0,
) -> FixModel:
    """Same two-fold protocol and budget rule as v1, with the quotient heads added."""

    started = time.time()
    split = int(train.shape[0] * (1.0 - mixer_fraction))
    part_a, part_b = train[:split], train[split:]
    if "full" not in cache:
        cache["full"] = F.build_tables(train)
        cache["a"] = F.build_tables(part_a)
        cache["charts_b"] = F.compute_charts(part_b, seg_len)
    if "fix_full" not in cache:
        cache["fix_full"] = build_fix_tables(train)
        cache["fix_a"] = build_fix_tables(part_a)
        cache["fix_keys_b"] = fix_keys(part_b, seg_len)
    min_count = 1
    if budget_bits is not None:
        fixed = F.HitTables().storage_bits() + F.N_STATES * len(ALL_HEADS) * F.MIXER_WEIGHT_BITS
        for candidate in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024):
            total = F.prune_tables(cache["full"], candidate).storage_bits() + sum(
                t.storage_bits() for t in _prune_fix(cache["fix_full"], candidate).values()
            )
            if total + fixed <= budget_bits:
                min_count = candidate
                break
        else:
            min_count = 1024
    tables = F.prune_tables(cache["full"], min_count)
    fix_tables = _prune_fix(cache["fix_full"], min_count)
    tables_a = F.prune_tables(cache["a"], min_count)
    fix_a = _prune_fix(cache["fix_a"], min_count)
    charts_b = cache["charts_b"]
    hits = F.HitTables().fit(charts_b)
    rng = np.random.default_rng(seed)
    keep = np.arange(charts_b.targets.size)
    if keep.size > mixer_positions:
        keep = np.sort(rng.choice(keep.size, size=mixer_positions, replace=False))
    probs, states = _probs(tables_a, fix_a, hits, charts_b, cache["fix_keys_b"])
    weights, loss = F.fit_mixer(probs[keep], states[keep], seed=seed)
    return FixModel(tables, fix_tables, hits, weights, min_count,
                    {"min_count": min_count, "mixer_train_bits": loss, "seconds": time.time() - started})
