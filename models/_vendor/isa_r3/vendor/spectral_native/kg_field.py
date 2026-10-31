"""Spectral-field programs for knowledge-graph completion.

A field program answers (e, q, ?) with exact joins over a knowledge store:

- the store is the training graph, one boolean adjacency per relation in the
  2R view (inverse relations included);
- entities carry family charts: WN18RR entities are coded by the hypernym /
  instance-hypernym forest (a family is an ancestor ball, the 2-adic ball of
  the code), FB15k-237 entities by a hierarchy induced from their relation
  signatures (average linkage cut at several levels);
- rules are joins with a confidence ``support / (body + pc)``:
  L1  q(x, y) <- q'(x, y)                    (q' any other 2R relation; symmetry is q' = q^-1)
  L2  q(x, y) <- q1(x, z), q2(z, y)          (paths, z != x, y != x)
  C   q(x, c) <- b(x)                        (b: an edge to a specific entity, having any
                                              q'-edge, or membership in a family)
- a candidate's score aggregates the confidences of the rules predicting it:
  maximum with a noisy-or tie-break, or noisy-or.

Everything a rule does is an exact lookup; confidences are the only numbers.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .kg_data import KG, adjacency, filter_sets, filtered_rank, metrics, to_queries


@dataclass
class FieldConfig:
    pc: float = 5.0
    min_support: int = 2
    min_conf: float = 0.001
    aggregation: str = "max"  # "max" or "noisyor"
    families: str = "auto"  # "auto", "none"
    path_pairs: str = "exact"  # "exact" or "sampled"
    path_samples: int = 400  # training triples per head relation for body discovery
    max_product_nnz: int = 30_000_000
    max_path_rules: int = 200  # per head relation, by confidence
    max_const_rules: int = 200_000  # per head relation, by confidence
    const_min_count: int = 2
    signature_levels: tuple[int, ...] = (16, 64, 256, 1024)
    seed: int = 0
    product_cache_nnz: int = 400_000_000  # beyond this, products are computed and not kept
    max_path_len: int = 2  # 3 enables sampled length-3 path rules (post hoc extension)
    path3_samples: int = 300
    path3_fanout: int = 48
    max_path3_rules: int = 150
    path3_candidates: int = 150  # most frequent sampled bodies per head that get exact counts
    prototype_weight: float = 0.0  # post hoc: weight of the within-coset prototype order
    prototype_alpha: float = 0.5
    reflexive: bool = False  # post hoc: score x as its own answer with the relation's self-loop rate


@dataclass
class Program:
    config: FieldConfig
    n: int
    R: int
    A: list[sp.csr_matrix]
    X: sp.csr_matrix  # entity x attribute
    attr_kind: np.ndarray  # 0 edge-constant, 1 has-relation, 2 family
    attr_rel: np.ndarray
    attr_obj: np.ndarray
    family_bits: float
    l1: dict[int, list[tuple[int, float]]] = field(default_factory=dict)
    l2: dict[int, list[tuple[tuple[int, int], float]]] = field(default_factory=dict)
    const: dict[int, sp.csr_matrix] = field(default_factory=dict)  # q -> (attributes x n) conf
    raw_l1: dict[int, list[tuple[int, int, int]]] = field(default_factory=dict)  # (k, support, body)
    raw_l2: dict[int, list[tuple[tuple[int, int], int, int]]] = field(default_factory=dict)
    raw_l3: dict[int, list[tuple[tuple[int, int, int], int, int]]] = field(default_factory=dict)
    l3: dict[int, list[tuple[tuple[int, int, int], float]]] = field(default_factory=dict)
    raw_const: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = field(default_factory=dict)
    prior: dict[int, np.ndarray] = field(default_factory=dict)
    products: dict[tuple[int, int], sp.csr_matrix] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


def _binary(m: sp.spmatrix) -> sp.csr_matrix:
    m = sp.csr_matrix(m)
    m.setdiag(0)
    m.eliminate_zeros()
    m.data[:] = 1.0
    return m


# ---------------------------------------------------------------------------
# Family charts
# ---------------------------------------------------------------------------


def hypernym_families(kg: KG, train: np.ndarray) -> tuple[list[list[int]], float]:
    """Ancestor sets from the hypernym / instance-hypernym forest (families = balls)."""

    names = [i for i, r in enumerate(kg.relations) if "hypernym" in r]
    parents: dict[int, list[int]] = {}
    for h, r, t in train:
        if r in names:
            parents.setdefault(int(h), []).append(int(t))
    families: list[list[int]] = []
    for x in range(kg.n):
        seen, frontier = set(), list(parents.get(x, []))
        depth = 0
        while frontier and depth < 32:
            nxt = []
            for p in frontier:
                if p not in seen:
                    seen.add(p)
                    nxt.extend(parents.get(p, []))
            frontier, depth = nxt, depth + 1
        families.append(sorted(seen))
    return families, 0.0  # derived from stored hypernym facts: no extra bits


def signature_families(A: list[sp.csr_matrix], n: int, levels: tuple[int, ...]) -> tuple[list[list[int]], float]:
    """Families from a hierarchy induced on relation signatures (average linkage)."""

    from scipy.cluster.hierarchy import fcluster, linkage

    signature = np.stack([np.asarray(m.sum(1)).ravel() > 0 for m in A], axis=1)
    unique, inverse = np.unique(signature, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    families: list[list[int]] = [[] for _ in range(n)]
    if unique.shape[0] < 2:
        return families, 0.0
    tree = linkage(unique.astype(bool), method="average", metric="jaccard")
    offset = 0
    bits = 0.0
    for k in levels:
        k_eff = min(k, unique.shape[0])
        labels = fcluster(tree, t=k_eff, criterion="maxclust") - 1
        for x in range(n):
            families[x].append(offset + int(labels[inverse[x]]))
        offset += int(labels.max()) + 1
        bits += n * math.log2(max(2, k_eff))
    return families, bits


# ---------------------------------------------------------------------------
# Attributes and rule learning
# ---------------------------------------------------------------------------


def build_attributes(A: list[sp.csr_matrix], n: int, families: list[list[int]]):
    rows, cols, kinds, rels, objs = [], [], [], [], []
    col = 0
    # edge-constant attributes (q, d)
    for q, m in enumerate(A):
        coo = m.tocoo()
        uniq, inv = np.unique(coo.col, return_inverse=True)
        rows.append(coo.row)
        cols.append(col + inv)
        kinds.append(np.zeros(uniq.shape[0], dtype=np.int8))
        rels.append(np.full(uniq.shape[0], q))
        objs.append(uniq)
        col += uniq.shape[0]
    # has-relation attributes (q, *)
    for q, m in enumerate(A):
        who = np.where(np.diff(m.indptr) > 0)[0]
        rows.append(who)
        cols.append(np.full(who.shape[0], col))
        kinds.append(np.array([1], dtype=np.int8))
        rels.append(np.array([q]))
        objs.append(np.array([-1]))
        col += 1
    # family attributes
    fam_ids = sorted({f for fs in families for f in fs})
    fam_col = {f: col + i for i, f in enumerate(fam_ids)}
    fr, fc = [], []
    for x, fs in enumerate(families):
        for f in fs:
            fr.append(x)
            fc.append(fam_col[f])
    if fr:
        rows.append(np.array(fr))
        cols.append(np.array(fc))
        kinds.append(np.full(len(fam_ids), 2, dtype=np.int8))
        rels.append(np.full(len(fam_ids), -1))
        objs.append(np.array(fam_ids))
        col += len(fam_ids)
    r = np.concatenate(rows)
    c = np.concatenate(cols)
    X = sp.csr_matrix((np.ones(r.shape[0], dtype=np.float32), (r, c)), shape=(n, col))
    X.sum_duplicates()
    X.data[:] = 1.0
    return X, np.concatenate(kinds), np.concatenate(rels), np.concatenate(objs)


def _product(program: Program, q1: int, q2: int) -> sp.csr_matrix | None:
    key = (q1, q2)
    if key in program.products:
        return program.products[key]
    a, b = program.A[q1], program.A[q2]
    # estimate output size through the shared middle entities
    estimate = float(np.asarray(a.sum(0)).ravel() @ np.asarray(b.sum(1)).ravel())
    if estimate > program.config.max_product_nnz:
        return None
    m = _binary(a @ b)
    cached = getattr(program, "_cache_nnz", 0)
    if cached + m.nnz <= program.config.product_cache_nnz:
        program.products[key] = m
        program._cache_nnz = cached + m.nnz
    return m


def _edge_lists(program: Program) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All labelled edges in the 2R view as CSR-style arrays: indptr, relation, neighbour."""

    rows, rels, nbrs = [], [], []
    for k, m in enumerate(program.A):
        coo = m.tocoo()
        rows.append(coo.row)
        rels.append(np.full(coo.nnz, k))
        nbrs.append(coo.col)
    rows_a, rels_a, nbrs_a = np.concatenate(rows), np.concatenate(rels), np.concatenate(nbrs)
    order = np.argsort(rows_a, kind="stable")
    indptr = np.zeros(program.n + 1, dtype=np.int64)
    np.add.at(indptr, rows_a + 1, 1)
    return np.cumsum(indptr), rels_a[order], nbrs_a[order]


def _sampled_bodies(program: Program, q: int, rng: np.random.Generator, edges) -> set[tuple[int, int]]:
    """Path bodies (q1, q2) seen on sampled training edges of q (object identity: z != x, y)."""

    indptr, rel, nbr = edges
    two_r = 2 * program.R
    m = program.A[q].tocoo()
    if m.nnz == 0:
        return set()
    pick = rng.choice(m.nnz, size=min(program.config.path_samples, m.nnz), replace=False)
    bodies: set[tuple[int, int]] = set()
    for i in pick:
        x, y = int(m.row[i]), int(m.col[i])
        rx, zx = rel[indptr[x]:indptr[x + 1]], nbr[indptr[x]:indptr[x + 1]]
        ry, zy = rel[indptr[y]:indptr[y + 1]], nbr[indptr[y]:indptr[y + 1]]
        # an edge y -k-> z is the edge z -inv(k)-> y
        into = {}
        for k, z in zip(((ry + program.R) % two_r).tolist(), zy.tolist()):
            if z != x and z != y:
                into.setdefault(z, []).append(k)
        for k1, z in zip(rx.tolist(), zx.tolist()):
            for k2 in into.get(z, ()):
                bodies.add((k1, k2))
    return bodies


def _sampled_bodies3(program: Program, q: int, rng: np.random.Generator, edges) -> set[tuple[int, int, int]]:
    """Length-3 bodies (k1, k2, k3) seen on sampled training edges of q.

    Object identity: the path x -k1-> z1 -k2-> z2 -k3-> y visits four distinct
    entities.  Fan-out per step is capped by random subsampling.
    """

    indptr, rel, nbr = edges
    two_r = 2 * program.R
    cfg = program.config
    m = program.A[q].tocoo()
    if m.nnz == 0:
        return set()
    pick = rng.choice(m.nnz, size=min(cfg.path3_samples, m.nnz), replace=False)
    from collections import Counter

    bodies: Counter = Counter()

    def out(e: int):
        lo, hi = indptr[e], indptr[e + 1]
        if hi - lo > cfg.path3_fanout:
            sel = lo + rng.choice(hi - lo, size=cfg.path3_fanout, replace=False)
            return rel[sel].tolist(), nbr[sel].tolist()
        return rel[lo:hi].tolist(), nbr[lo:hi].tolist()

    for i in pick:
        x, y = int(m.row[i]), int(m.col[i])
        into: dict[int, list[int]] = {}
        ry, zy = rel[indptr[y]:indptr[y + 1]], nbr[indptr[y]:indptr[y + 1]]
        for k, z in zip(((ry + program.R) % two_r).tolist(), zy.tolist()):
            into.setdefault(z, []).append(k)
        r1, z1s = out(x)
        for k1, z1 in zip(r1, z1s):
            if z1 in (x, y):
                continue
            r2, z2s = out(z1)
            for k2, z2 in zip(r2, z2s):
                if z2 in (x, y, z1):
                    continue
                for k3 in into.get(z2, ()):
                    bodies[(k1, k2, k3)] += 1
    return {body for body, _count in bodies.most_common(cfg.path3_candidates)}


def _path3_matrix(program: Program, body: tuple[int, int, int]) -> sp.csr_matrix | None:
    """offdiag(A_k1 A_k2) A_k3 with the result's diagonal removed (x != z2, y != x)."""

    a, b, c = body
    inner = _product(program, a, b)
    if inner is None:
        return None
    estimate = float(np.asarray(inner.sum(0)).ravel() @ np.asarray(program.A[c].sum(1)).ravel())
    if estimate > program.config.max_product_nnz:
        return None
    return _binary(inner @ program.A[c])


def learn(kg: KG, train: np.ndarray, config: FieldConfig) -> Program:
    started = time.time()
    n, R = kg.n, kg.R
    A = adjacency(train, n, R)
    if config.families == "none":
        families, fam_bits = [[] for _ in range(n)], 0.0
    elif any("hypernym" in r for r in kg.relations):
        families, fam_bits = hypernym_families(kg, train)
    else:
        families, fam_bits = signature_families(A, n, config.signature_levels)
    X, kinds, rels, objs = build_attributes(A, n, families)
    program = Program(config, n, R, A, X, kinds, rels, objs, fam_bits)
    program.timings["charts"] = time.time() - started
    rng = np.random.default_rng(config.seed)
    body_count = np.asarray(X.sum(0)).ravel()
    edges = _edge_lists(program) if config.path_pairs == "sampled" else None
    for q in range(2 * R):
        target = A[q]
        if target.nnz == 0:
            continue
        # L1
        l1 = []
        for k in range(2 * R):
            if k == q:
                continue
            support = target.multiply(A[k]).nnz
            if support >= config.min_support:
                l1.append((k, int(support), int(A[k].nnz)))
        program.raw_l1[q] = l1
        # L2
        if config.path_pairs == "exact":
            pairs = [(a, b) for a in range(2 * R) for b in range(2 * R)]
        else:
            pairs = sorted(_sampled_bodies(program, q, rng, edges))
        l2 = []
        for a, b in pairs:
            m = _product(program, a, b)
            if m is None or m.nnz == 0:
                continue
            support = target.multiply(m).nnz
            if support >= config.min_support:
                l2.append(((a, b), int(support), int(m.nnz)))
        # keep the strongest bodies under a generous correction; finalize() re-ranks per pc
        l2.sort(key=lambda item: -item[1] / (item[2] + 5.0))
        program.raw_l2[q] = l2[: 2 * config.max_path_rules]
        if config.max_path_len >= 3:
            if edges is None:
                edges = _edge_lists(program)
            l3 = []
            for body in sorted(_sampled_bodies3(program, q, rng, edges)):
                m3 = _path3_matrix(program, body)
                if m3 is None or m3.nnz == 0:
                    continue
                support = target.multiply(m3).nnz
                if support >= config.min_support:
                    l3.append((body, int(support), int(m3.nnz)))
            l3.sort(key=lambda item: -item[1] / (item[2] + 5.0))
            program.raw_l3[q] = l3[: 2 * config.max_path3_rules]
        # constant rules q(x, c) <- b(x)
        counts = np.asarray(target.sum(0)).ravel()
        constants = np.where(counts >= config.const_min_count)[0]
        program.prior[q] = counts.astype(np.float32)
        if constants.size:
            support = (X.T @ target[:, constants]).tocoo()  # attributes x constants
            keep = support.data >= config.min_support
            b_idx, c_loc, s = support.row[keep], support.col[keep], support.data[keep]
            c_idx = constants[c_loc]
            # drop the identity attribute (q, c) predicting q(x, c)
            ident = (kinds[b_idx] == 0) & (rels[b_idx] == q) & (objs[b_idx] == c_idx)
            b_idx, c_idx, s = b_idx[~ident], c_idx[~ident], s[~ident]
            body = body_count[b_idx]
            loose = s / (body + 1.0) >= 0.01
            program.raw_const[q] = (b_idx[loose], c_idx[loose], s[loose].astype(np.int64), body[loose].astype(np.int64))
    finalize(program, config.pc)
    # drop cached products not used by any stored path rule; if any used product is
    # missing from the cache, evaluate every path join lazily (exact, query rows only)
    used = {body for rules in program.raw_l2.values() for body, _s, _b in rules}
    program.products = {k: v for k, v in program.products.items() if k in used}
    program.lazy_products = any(body not in program.products for body in used)
    program.timings["learn"] = time.time() - started
    return program


def finalize(program: Program, pc: float) -> None:
    """Turn raw (support, body) statistics into confidences for one correction pc."""

    config = program.config
    config.pc = pc
    program.l1 = {}
    program.l2 = {}
    program.const = {}
    for q, rules in program.raw_l1.items():
        conf = [(k, s / (b + pc)) for k, s, b in rules]
        program.l1[q] = sorted([r for r in conf if r[1] >= config.min_conf], key=lambda item: -item[1])
    for q, rules in program.raw_l2.items():
        conf = [(body, s / (b + pc)) for body, s, b in rules]
        conf = [r for r in conf if r[1] >= config.min_conf]
        program.l2[q] = sorted(conf, key=lambda item: -item[1])[: config.max_path_rules]
    for q, rules in program.raw_l3.items():
        conf = [(body, s / (b + pc)) for body, s, b in rules]
        conf = [r for r in conf if r[1] >= config.min_conf]
        program.l3[q] = sorted(conf, key=lambda item: -item[1])[: config.max_path3_rules]
    for q, (b_idx, c_idx, s, body) in program.raw_const.items():
        conf = s / (body + pc)
        good = conf >= max(config.min_conf, 0.01)
        bi, ci, cf = b_idx[good], c_idx[good], conf[good]
        if cf.size > config.max_const_rules:
            top = np.argpartition(-cf, config.max_const_rules)[: config.max_const_rules]
            bi, ci, cf = bi[top], ci[top], cf[top]
        program.const[q] = sp.csr_matrix((cf.astype(np.float32), (bi, ci)), shape=(program.X.shape[1], program.n))


# ---------------------------------------------------------------------------
# Scoring and evaluation
# ---------------------------------------------------------------------------


def score_batch(program: Program, q: int, xs: np.ndarray) -> np.ndarray:
    """Scores (len(xs) x n) for queries (x, q, ?)."""

    n = program.n
    Q = xs.shape[0]
    s_max = np.zeros((Q, n), dtype=np.float32)
    s_log = np.zeros((Q, n), dtype=np.float32)

    def apply(m: sp.csr_matrix, conf: float) -> None:
        sub = m[xs].tocoo()
        if sub.nnz:
            np.maximum.at(s_max, (sub.row, sub.col), conf)
            s_log[sub.row, sub.col] += math.log1p(-min(conf, 0.999999))

    for k, conf in program.l1.get(q, []):
        apply(program.A[k], conf)
    for (a, b), conf in program.l2.get(q, []):
        if getattr(program, "lazy_products", False):
            # the same join computed for the query rows only: (A_a[xs] @ A_b) minus x itself
            sub = (program.A[a][xs] @ program.A[b]).tocoo()
            keep = sub.col != xs[sub.row]
            if keep.any():
                rows, cols = sub.row[keep], sub.col[keep]
                np.maximum.at(s_max, (rows, cols), conf)
                s_log[rows, cols] += math.log1p(-min(conf, 0.999999))
        else:
            apply(program.products[(a, b)], conf)
    for (a, b, c), conf in program.l3.get(q, []):
        inner = (program.A[a][xs] @ program.A[b]).tocoo()
        keep = inner.col != xs[inner.row]  # z2 != x
        inner = sp.csr_matrix((inner.data[keep], (inner.row[keep], inner.col[keep])), shape=(xs.shape[0], n))
        inner.data[:] = 1.0
        sub = (inner @ program.A[c]).tocoo()
        keep = sub.col != xs[sub.row]  # y != x
        if keep.any():
            rows, cols = sub.row[keep], sub.col[keep]
            np.maximum.at(s_max, (rows, cols), conf)
            s_log[rows, cols] += math.log1p(-min(conf, 0.999999))
    table = program.const.get(q)
    if table is not None and table.nnz:
        xq = program.X[xs].tocoo()
        rows = table[xq.col]  # one row per (query, attribute) pair
        rc = rows.tocoo()
        if rc.nnz:
            qi = xq.row[rc.row]
            np.maximum.at(s_max, (qi, rc.col), rc.data)
            np.add.at(s_log, (qi, rc.col), np.log1p(-np.minimum(rc.data, 0.999999)))
    noisy = 1.0 - np.exp(s_log.astype(np.float64))
    proto = prototype_scores(program, q, xs) if program.config.prototype_weight > 0 else 0.0
    prior = program.prior.get(q)
    prior_term = 0.0 if prior is None else 1e-9 * prior[None, :]
    if program.config.aggregation == "max":
        score = s_max.astype(np.float64) + 1e-3 * noisy + prior_term
    else:
        score = noisy + prior_term
    score = score + program.config.prototype_weight * proto
    if program.config.reflexive:
        # reflexive rule q(x, x): confidence = share of q's training heads that are self-loops
        if not hasattr(program, "_self_rate"):
            program._self_rate = {}
        if q not in program._self_rate:
            m = program.A[q]
            heads = max(1, int((np.diff(m.indptr) > 0).sum()))
            program._self_rate[q] = float(m.diagonal().sum()) / (heads + program.config.pc)
        score[np.arange(Q), xs] = program._self_rate[q]
    else:
        score[np.arange(Q), xs] = -np.inf  # an entity is not its own answer
    return score


def prototype_scores(program: Program, q: int, xs: np.ndarray) -> np.ndarray:
    """Within-coset order: weighted overlap of each candidate with e's answer prototype.

    For query entity e with known training answers A_e = {y : q(e, y)}, the
    weight of attribute b is its log-odds enrichment among A_e relative to all
    entities, kept only where positive.  A candidate's score is the normalized
    weighted count of its attributes, in [0, 1].  This is a metric program: an
    exact weighted overlap over attribute charts, the soft counterpart of a
    coset join.  Queries whose entity has no known answers score 0 everywhere.
    """

    alpha = program.config.prototype_alpha
    n = program.n
    answers = program.A[q][xs]  # Q x n
    sizes = np.asarray(answers.sum(1)).ravel()
    counts = (answers @ program.X).tocoo()  # Q x attributes
    if counts.nnz == 0:
        return np.zeros((xs.shape[0], n))
    if not hasattr(program, "_attr_freq"):
        program._attr_freq = np.asarray(program.X.sum(0)).ravel()
    freq = program._attr_freq[counts.col]
    inside = (counts.data + alpha) / (sizes[counts.row] + 2 * alpha)
    outside = (freq + alpha) / (n + 2 * alpha)
    weight = np.log(inside / outside)
    keep = weight > 0
    w = sp.csr_matrix((weight[keep], (counts.row[keep], counts.col[keep])), shape=(xs.shape[0], program.X.shape[1]))
    total = np.asarray(w.sum(1)).ravel()
    raw = (w @ program.X.T).toarray()
    return raw / np.maximum(total, 1e-12)[:, None]


def evaluate(program: Program, kg: KG, split: np.ndarray, known: dict, batch: int = 256,
             per_relation: bool = True) -> dict:
    queries = to_queries(split, kg.R)
    out_ranks = np.zeros((queries.shape[0], 2), dtype=np.int64)
    covered = np.zeros(queries.shape[0], dtype=bool)  # some rule fired on the answer
    for q in np.unique(queries[:, 1]):
        idx = np.where(queries[:, 1] == q)[0]
        for start in range(0, idx.shape[0], batch):
            sel = idx[start:start + batch]
            scores = score_batch(program, int(q), queries[sel, 0])
            for row, i in enumerate(sel):
                e, qq, a = queries[i]
                out_ranks[i] = filtered_rank(scores[row], int(a), known[(int(e), int(qq))])
                covered[i] = scores[row, int(a)] > 1e-4  # rule confidences are >= 1e-3; the popularity term stays < 1e-5
    result = {"all": metrics(out_ranks)}
    result["all"]["answer_covered_by_a_rule"] = float(covered.mean())
    if per_relation:
        per = {}
        for q in np.unique(queries[:, 1]):
            mask = queries[:, 1] == q
            name = kg.relations[q % kg.R] + ("" if q < kg.R else " (head)")
            per[name] = metrics(out_ranks[mask])
            per[name]["answer_covered_by_a_rule"] = float(covered[mask].mean())
            if covered[mask].any():
                per[name]["MRR_when_covered"] = metrics(out_ranks[mask & covered])["MRR"]
        result["per_relation"] = per
    result["ranks"] = out_ranks
    return result


def program_bits(program: Program, kg: KG) -> dict[str, float]:
    n, R = program.n, program.R
    lg_n, lg_q = math.log2(n), math.log2(2 * R)
    lg_a = math.log2(program.X.shape[1])
    store = kg.train.shape[0] * (2 * lg_n + math.log2(R))
    l1 = sum(len(v) for v in program.l1.values()) * (2 * lg_q + 16)
    l2 = sum(len(v) for v in program.l2.values()) * (3 * lg_q + 16)
    l2 += sum(len(v) for v in program.l3.values()) * (4 * lg_q + 16)
    const = sum(m.nnz for m in program.const.values()) * (lg_q + lg_a + lg_n + 16)
    return {"store": store, "rules_l1": l1, "rules_l2": l2, "rules_const": const,
            "families": program.family_bits, "total": store + l1 + l2 + const + program.family_bits,
            "rule_counts": {"l1": sum(len(v) for v in program.l1.values()),
                            "l2": sum(len(v) for v in program.l2.values()),
                            "l3": sum(len(v) for v in program.l3.values()),
                            "const": int(sum(m.nnz for m in program.const.values()))}}


__all__ = ["FieldConfig", "Program", "learn", "score_batch", "evaluate", "program_bits", "filter_sets"]
