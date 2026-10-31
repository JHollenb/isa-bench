"""L3b: family programs over the Hearst matrix, smoothed-evidence closure, vector-induced trees.

Fixes for the L3 failures (docs/text-taxonomy-v2-findings.md):

- **Family program over the Hearst matrix.** Rows are hyponym terms, columns are "is-a H"
  attributes, entries are pattern counts. A 2-adic tree over the rows is grown by level-synchronous
  spherical bisection of a random sketch of each row's PPMI hypernym profile (one bit per level, root
  bit lowest, so a ball at depth d is a congruence class of the code mod 2^d). Each ball stores its
  hypernym counts; minimum description length keeps a ball's table only when its likelihood gain over
  the nearest kept ancestor pays for its entries (the family's override). An unobserved pair (x, H) is
  scored by Dirichlet backoff down x's chain of kept balls: sharing across rows, the discrete
  counterpart of a low-rank factorization.
- **Closure over smoothed evidence.** Direct score plus one hop through each term's top-K hypernyms
  under the same scorer: max(s(x, G), max_H min(s(x, H), s(H, G))).
- **Multi-parent closure on raw counts.** Widest-path strength over each term's top-k count parents,
  depth limited.
- **Vector-induced entity trees.** Average-linkage trees over distributional text vectors, optionally
  with Hearst support sets as a soft similarity term whose weight is chosen by description length.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# 2-adic tree over rows by level-synchronous spherical bisection
# --------------------------------------------------------------------------- #


def l2_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


def sparse_l2_rows(mat: Any) -> Any:
    from scipy import sparse

    m = mat.tocsr().astype(np.float64)
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    inv = np.where(norms > 0, 1.0 / np.maximum(norms, 1e-300), 0.0)
    return sparse.diags(inv) @ m


def gaussian_sketch(mat: Any, dims: int, seed: int = 0) -> np.ndarray:
    """Johnson-Lindenstrauss sketch of sparse rows: mat @ G / sqrt(dims), float32."""

    rng = np.random.default_rng(seed)
    g = rng.standard_normal((mat.shape[1], dims), dtype=np.float32) / np.float32(math.sqrt(dims))
    return np.asarray(mat.astype(np.float32) @ g, dtype=np.float32)


def _group_sum(ids: np.ndarray, x: np.ndarray, groups: int) -> np.ndarray:
    from scipy import sparse

    s = sparse.csr_matrix((np.ones(ids.size, dtype=np.float32), (ids, np.arange(ids.size))), shape=(groups, ids.size))
    return np.asarray(s @ x, dtype=np.float32)


@dataclass
class BisectionTree:
    """Level ids and 2-adic codes of a level-synchronous binary partition of rows."""

    levels: list[np.ndarray]  # levels[d][i] = compact ball id of row i at depth d (levels[0] all 0)
    codes: np.ndarray  # uint64; bit d = side taken at depth d (root bit lowest)

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def ball_sizes(self, d: int) -> np.ndarray:
        return np.bincount(self.levels[d])

    def same_ball(self, i: int, j: int, d: int) -> bool:
        return bool(self.levels[d][i] == self.levels[d][j])

    def congruent(self, i: int, j: int, d: int) -> bool:
        mask = np.uint64((1 << d) - 1) if d < 64 else np.uint64(0xFFFFFFFFFFFFFFFF)
        return bool((self.codes[i] & mask) == (self.codes[j] & mask))


def bisection_tree(x: np.ndarray, depth: int, iters: int = 4, seed: int = 0) -> BisectionTree:
    """Grow a binary tree over the rows of ``x`` (L2-normalized, float32), all balls split per level.

    Each ball is split by spherical 2-means started from a random-member direction; a ball of one row
    never splits. Level d's ball ids are the compacted values of code mod 2^d.
    """

    assert depth <= 63
    rows = x.shape[0]
    rng = np.random.default_rng(seed)
    bid = np.zeros(rows, dtype=np.int64)
    levels = [bid.copy()]
    codes = np.zeros(rows, dtype=np.uint64)
    for d in range(depth):
        nb = int(bid.max()) + 1
        sizes = np.bincount(bid, minlength=nb)
        if sizes.max() < 2:
            break
        mean = _group_sum(bid, x, nb) / np.maximum(sizes, 1)[:, None]
        key = rng.random(rows)
        order = np.lexsort((key, bid))
        first = np.ones(rows, dtype=bool)
        first[1:] = bid[order][1:] != bid[order][:-1]
        pivot = np.empty(nb, dtype=np.int64)
        pivot[bid[order][first]] = order[first]
        direction = x[pivot] - mean
        side = np.einsum("ij,ij->i", x - mean[bid], direction[bid]) > 0
        for _ in range(iters):
            g = bid * 2 + side
            cent = l2_rows(_group_sum(g, x, 2 * nb))
            s0 = np.einsum("ij,ij->i", x, cent[bid * 2])
            s1 = np.einsum("ij,ij->i", x, cent[bid * 2 + 1])
            side = s1 > s0
        side &= sizes[bid] >= 2
        codes |= side.astype(np.uint64) << np.uint64(d)
        bid = np.unique(bid * 2 + side, return_inverse=True)[1].astype(np.int64)
        levels.append(bid.copy())
    return BisectionTree(levels=levels, codes=codes)


# --------------------------------------------------------------------------- #
# Family program over the Hearst matrix: ball tables, MDL overrides, Dirichlet backoff
# --------------------------------------------------------------------------- #


@dataclass
class FamilyBackoff:
    """Per-ball hypernym count tables along a BisectionTree, with MDL-kept overrides.

    ``counts`` is the (tree rows x hypernym columns) count matrix. ``keep[d][b]`` says whether ball b
    at depth d stores its own table (an override) or inherits its nearest kept ancestor's.
    """

    counts: Any
    tree: BisectionTree
    entry_bits: float
    tables: list[Any] = field(default_factory=list)
    totals: list[np.ndarray] = field(default_factory=list)
    keep: list[np.ndarray] = field(default_factory=list)
    gain_bits: list[np.ndarray] = field(default_factory=list)
    rowtot: np.ndarray = field(default_factory=lambda: np.zeros(0))
    df_tables: list[Any] = field(default_factory=list)  # rows of each ball carrying hypernym H
    sizes: list[np.ndarray] = field(default_factory=list)

    @classmethod
    def build(cls, counts: Any, tree: BisectionTree, entry_bits: float) -> "FamilyBackoff":
        from scipy import sparse

        counts = counts.tocsr().astype(np.float64)
        fb = cls(counts=counts, tree=tree, entry_bits=entry_bits)
        rows = counts.shape[0]
        binary = counts.copy()
        binary.data = np.ones_like(binary.data)
        for d, bid in enumerate(tree.levels):
            nb = int(bid.max()) + 1
            s = sparse.csr_matrix((np.ones(rows), (bid, np.arange(rows))), shape=(nb, rows))
            t = (s @ counts).tocsr()
            t.sum_duplicates()
            fb.tables.append(t)
            fb.totals.append(np.asarray(t.sum(axis=1)).ravel())
            fb.df_tables.append((s @ binary).tocsr())
            fb.sizes.append(np.bincount(bid, minlength=nb).astype(np.float64))
        fb.keep.append(np.ones(1, dtype=bool))
        fb.gain_bits.append(np.zeros(1))
        # nearest kept ancestor (level, id) per row; root always kept
        ref_level = np.zeros(rows, dtype=np.int64)
        ref_id = np.zeros(rows, dtype=np.int64)
        for d in range(1, len(tree.levels)):
            bid = tree.levels[d]
            t = fb.tables[d].tocoo()
            nb = t.shape[0]
            first_row = np.full(nb, -1, dtype=np.int64)
            first_row[bid[::-1]] = np.arange(rows)[::-1]
            b_level, b_id = ref_level[first_row], ref_id[first_row]
            n_ref = np.zeros(t.nnz)
            e_level, e_id = b_level[t.row], b_id[t.row]
            for lv in np.unique(e_level):
                m = e_level == lv
                n_ref[m] = np.asarray(fb.tables[lv][e_id[m], t.col[m]]).ravel()
            big_n = fb.totals[d][t.row]
            big_ref = _gather_totals(fb.totals, e_level, e_id)
            with np.errstate(divide="ignore", invalid="ignore"):
                contrib = t.data * (np.log2(t.data / big_n) - np.log2(n_ref / big_ref))
            contrib = np.where(np.isfinite(contrib), contrib, 0.0)
            gain = np.bincount(t.row, weights=contrib, minlength=nb)
            nnz_b = np.bincount(t.row, minlength=nb)
            # a one-row ball is the row itself (its own counts enter at the leaf step): never an override
            keep = (gain > nnz_b * entry_bits) & (np.bincount(bid, minlength=nb) >= 2)
            fb.keep.append(keep)
            fb.gain_bits.append(gain)
            upd = keep[bid]
            ref_level = np.where(upd, d, ref_level)
            ref_id = np.where(upd, bid, ref_id)
        fb.rowtot = np.asarray(counts.sum(axis=1)).ravel()
        return fb

    def program_bits(self, code_bits: float) -> dict[str, float]:
        kept_entries = 0
        for d in range(1, len(self.tables)):
            nnz_b = np.bincount(self.tables[d].tocoo().row, minlength=self.tables[d].shape[0])
            kept_entries += int(nnz_b[self.keep[d]].sum())
        leaf_entries = int(self.counts.nnz)
        return {"kept_ball_entries": kept_entries, "root_entries": int(self.tables[0].nnz),
                "leaf_entries": leaf_entries, "kept_balls": int(sum(k.sum() for k in self.keep[1:])),
                "balls": int(sum(k.size for k in self.keep[1:])),
                "table_bits": float((kept_entries + self.tables[0].nnz) * self.entry_bits),
                "code_bits": float(code_bits)}

    def score(self, rows: np.ndarray, cols: np.ndarray, alpha: float, form: str = "pmi", use_keep: bool = True) -> np.ndarray:
        """Backoff score of (row, col) pairs; row < 0 or col < 0 means unknown.

        ``rate`` treats each "is-a H" as a binary attribute (K1 semantics): a stored pair is recalled
        (score 1 + log(1 + count), above every inferred pair) and an unstored pair gets the smoothed
        rate of H among the rows of x's kept families. ``pmi`` / ``prob`` treat x's hypernym tokens as a
        multinomial smoothed down the same chain.
        """

        if form == "rate":
            return self._rate(np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64), alpha, use_keep)

        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        out = np.zeros(rows.size)
        known_col = cols >= 0
        root = self.tables[0]
        big_root = self.totals[0][0]
        q0 = np.zeros(rows.size)
        if known_col.any():
            q0[known_col] = np.asarray(root[np.zeros(int(known_col.sum()), dtype=np.int64), cols[known_col]]).ravel() / big_root
        q = q0.copy()
        v = (rows >= 0) & known_col
        if v.any():
            r, c = rows[v], cols[v]
            qv = q[v]
            for d in range(1, len(self.tables)):
                b = self.tree.levels[d][r]
                n = np.asarray(self.tables[d][b, c]).ravel()
                big = self.totals[d][b]
                new = (n + alpha * qv) / (big + alpha)
                qv = np.where(self.keep[d][b], new, qv) if use_keep else new
            n = np.asarray(self.counts[r, c]).ravel()
            qv = (n + alpha * qv) / (self.rowtot[r] + alpha)
            q[v] = qv
        if form == "prob":
            out = q
        else:
            with np.errstate(divide="ignore"):
                out = np.log(np.maximum(q, 1e-300)) - np.log(np.maximum(q0, 1e-300))
            out[~known_col] = 0.0
            out[(rows < 0) & known_col] = 0.0
        return out


def _rate_impl(self, rows: np.ndarray, cols: np.ndarray, alpha: float, use_keep: bool) -> np.ndarray:
    out = np.zeros(rows.size)
    known_col = cols >= 0
    r0 = np.zeros(rows.size)
    if known_col.any():
        r0[known_col] = np.asarray(self.df_tables[0][np.zeros(int(known_col.sum()), dtype=np.int64),
                                                     cols[known_col]]).ravel() / self.sizes[0][0]
    out[:] = r0
    v = (rows >= 0) & known_col
    if v.any():
        r, c = rows[v], cols[v]
        rv = r0[v]
        for d in range(1, len(self.df_tables)):
            b = self.tree.levels[d][r]
            df = np.asarray(self.df_tables[d][b, c]).ravel()
            new = (df + alpha * rv) / (self.sizes[d][b] + alpha)
            rv = np.where(self.keep[d][b], new, rv) if use_keep else new
        n = np.asarray(self.counts[r, c]).ravel()
        out[v] = np.where(n > 0, 1.0 + np.log1p(n), rv)
    return out


FamilyBackoff._rate = _rate_impl


def _gather_totals(totals: list[np.ndarray], levels: np.ndarray, ids: np.ndarray) -> np.ndarray:
    out = np.zeros(levels.size)
    for lv in np.unique(levels):
        m = levels == lv
        out[m] = totals[lv][ids[m]]
    return out


# --------------------------------------------------------------------------- #
# Closures
# --------------------------------------------------------------------------- #


def one_hop_closure(direct: np.ndarray, parent_scores: np.ndarray, parent_to_target: np.ndarray) -> np.ndarray:
    """max(direct, max_k min(s(x, H_k), s(H_k, G))) with (pairs x K) arrays; K may be 0."""

    if parent_scores.size == 0:
        return direct
    hop = np.minimum(parent_scores, parent_to_target).max(axis=1)
    return np.maximum(direct, hop)


def top_parents_by(pairs: dict[tuple[str, str], float], k: int) -> dict[str, list[tuple[str, float]]]:
    by: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (x, y), c in pairs.items():
        by[x].append((c, y))
    out = {}
    for x, lst in by.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        out[x] = [(y, c) for c, y in lst[:k]]
    return out


def widest_paths(adj: dict[str, list[tuple[str, float]]], source: str, max_depth: int) -> dict[str, float]:
    """Max over paths of at most ``max_depth`` edges of the minimum edge weight (source excluded)."""

    best: dict[str, float] = {}
    frontier = {source: math.inf}
    for _ in range(max_depth):
        nxt: dict[str, float] = {}
        for u, w_u in frontier.items():
            for v, w in adj.get(u, ()):
                if v == source:
                    continue
                cand = min(w_u, w)
                if cand > best.get(v, 0.0):
                    best[v] = cand
                if cand > nxt.get(v, 0.0):
                    nxt[v] = cand
        frontier = nxt
        if not frontier:
            break
    return best


# --------------------------------------------------------------------------- #
# Vector-induced entity trees with Hearst support as a soft similarity term
# --------------------------------------------------------------------------- #


def support_sets(ents: list[str], up: dict[str, list[tuple[str, float]]]) -> list[set[str]]:
    """Direct and two-hop pattern hypernyms of each entity (L3's context-family support)."""

    out = []
    for e in ents:
        s: set[str] = set()
        for z, _c in up.get(e, ()):
            s.add(z)
            for y, _c2 in up.get(z, ()):
                if y != e:
                    s.add(y)
        out.append(s)
    return out


def jaccard_matrix(sets: list[set[str]]) -> np.ndarray:
    n = len(sets)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = sets[i], sets[j]
            u = len(a | b)
            out[i, j] = out[j, i] = (len(a & b) / u) if u else 0.0
    return out


def similarity_linkage(sim: np.ndarray):
    """Average-linkage over distance 1 - sim (sim in [-1, 1] scaled to [0, 1])."""

    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    d = np.clip(1.0 - sim, 0.0, None)
    np.fill_diagonal(d, 0.0)
    d = (d + d.T) / 2.0
    return linkage(squareform(d, checks=False), method="average")


def vector_tree_similarity(vectors: np.ndarray, jac: np.ndarray | None, beta: float) -> np.ndarray:
    v = l2_rows(np.asarray(vectors, dtype=np.float64)).astype(np.float64)
    cos = v @ v.T
    if jac is None or beta == 0.0:
        return cos
    return (cos + beta * jac) / (1.0 + beta)
