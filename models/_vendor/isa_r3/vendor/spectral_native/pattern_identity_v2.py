"""Relation identity of real-text surface patterns, v2 (experiment L2b).

L2 (``pattern_identity``) failed all four preregistered bars; its diagnosis
(docs/pattern-identity-findings.md) is turned into the following mechanisms,
all exact counting and parsimony except the explicitly graded mixers:

- **Containment against the relation class.** Agreement on a shared subject x is
  measured as "some object that q states for x is in the relation's training
  fact set KB_r(x)" instead of agreement with one other pattern (a set-valued
  relation with m objects per subject agrees by equality only about 1/m of the
  time).  ``FactIndex`` stores the training facts entity-major (subject -> labels,
  mode object) and pair-major ((subject, object) -> labels).  Each relation's
  cardinality (functional vs set-valued) is chosen by minimum description length.
- **Per-relation rates.** eps_r and pi_r (and pi_{g,r} per gold relation g) are
  Beta-shrunk counts (``shrink``), giving pair-specific likelihood ratios and a
  pair-specific overlap requirement n ~ ln(1/delta) / KL(Bern(1-eps) || Bern(pi)).
- **Argument-type axes.** Entities get types from their training roles (argmax
  role, or spherical k-means over role profiles) and, where T-REx aligns them,
  from Wikidata P31 facts.  A pattern's type cell is the (left type, right type)
  pair of its occurrences.  ``TypeProgram`` chooses per cell, by minimum
  description length, a star (type alone decides), a product (a word taxonomy
  inside the cell decides) or flat storage (words across the whole labelled set).
- **Evidence mixing.** Route distributions and fact log-likelihood ratios are
  combined geometrically with count-bucketed weights fitted on dev
  (``fit_mixer``), instead of L2's hard routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spectral_native.knowledge import compress
from spectral_native.pattern_identity import (
    LCA,
    Corpus,
    FieldTaxonomy,
    cosine_linkage_tree,
    occurrences_of,
)

# ---------------------------------------------------------------------------
# CSR helpers
# ---------------------------------------------------------------------------


def csr_gather(ptr: np.ndarray, data: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate CSR rows ``rows``: (values, index into ``rows`` of each value)."""

    rows = np.asarray(rows, dtype=np.int64)
    lengths = ptr[rows + 1] - ptr[rows]
    total = int(lengths.sum())
    if total == 0:
        return data[:0], np.zeros(0, dtype=np.int64)
    starts = np.repeat(ptr[rows], lengths)
    offsets = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return data[starts + offsets], np.repeat(np.arange(len(rows), dtype=np.int64), lengths)


def _ptr_from_sorted(keys: np.ndarray, size: int) -> np.ndarray:
    ptr = np.zeros(size + 1, dtype=np.int64)
    np.add.at(ptr, keys + 1, 1)
    return np.cumsum(ptr)


# ---------------------------------------------------------------------------
# Fact index: training facts of every relation, for containment tests
# ---------------------------------------------------------------------------


def gamma_bits(k: np.ndarray) -> np.ndarray:
    """Elias gamma code length of positive integers."""

    return 2 * np.floor(np.log2(np.maximum(k, 1))) + 1


@dataclass
class FactIndex:
    n_ent: int
    n_lab: int
    ent_ptr: np.ndarray  # (E+1,) CSR over subjects
    ent_label: np.ndarray  # label of each (subject, label) group
    ent_mode: np.ndarray  # most frequent object of the group (ties: smallest id)
    ent_nobj: np.ndarray  # distinct objects of the group
    pair_code: np.ndarray  # sorted unique subject * E + object
    pair_ptr: np.ndarray
    pair_label: np.ndarray
    cardinality: dict[int, dict[str, float]] = field(default_factory=dict)

    def functional(self, label: int) -> bool:
        info = self.cardinality.get(int(label))
        return True if info is None else bool(info["functional"])


def build_fact_index(corpus: Corpus, split: int = 0) -> FactIndex:
    """Facts (subject=left, label, object=right) from the occurrence labels of one split."""

    n_ent, n_lab = len(corpus.entities), len(corpus.labels)
    occ = np.where(corpus.split[corpus.occ_pattern] == split)[0]
    labs = corpus.occ_labels[occ]
    rep = np.repeat(occ, labs.shape[1])
    lab = labs.ravel()
    keep = lab >= 0
    rep, lab = rep[keep], lab[keep].astype(np.int64)
    x = corpus.occ_left[rep].astype(np.int64)
    y = corpus.occ_right[rep].astype(np.int64)
    key = (x * n_lab + lab) * n_ent + y
    uk, cnt = np.unique(key, return_counts=True)
    y_u = uk % n_ent
    xl = uk // n_ent
    lab_u = xl % n_lab
    x_u = xl // n_lab
    # (subject, label) groups: uk is sorted by (x, label, y)
    new = np.r_[True, xl[1:] != xl[:-1]]
    grp_start = np.flatnonzero(new)
    grp_nobj = np.diff(np.r_[grp_start, len(xl)])
    order = np.lexsort((y_u, -cnt, xl))  # by group, then most frequent, then smallest object
    first = order[np.r_[True, xl[order][1:] != xl[order][:-1]]]
    g_x, g_lab, g_mode = x_u[first], lab_u[first], y_u[first]
    ent_ptr = _ptr_from_sorted(g_x, n_ent)
    # pair-major
    pk = x_u * n_ent + y_u
    order2 = np.lexsort((lab_u, pk))
    pk_s, lab_s = pk[order2], lab_u[order2]
    pair_code, pstart = np.unique(pk_s, return_index=True)
    pair_ptr = np.r_[pstart, len(pk_s)].astype(np.int64)
    index = FactIndex(n_ent, n_lab, ent_ptr, g_lab.astype(np.int64), g_mode.astype(np.int64),
                      grp_nobj.astype(np.int64), pair_code, pair_ptr, lab_s.astype(np.int64))
    index.cardinality = relation_cardinality(g_lab, grp_nobj, lab_u, y_u)
    return index


def relation_cardinality(g_lab: np.ndarray, g_nobj: np.ndarray, lab_u: np.ndarray,
                         y_u: np.ndarray) -> dict[int, dict[str, float]]:
    """Functional vs set-valued per label by minimum description length.

    Functional code: one object per subject (log2 O bits), every extra object an
    exception naming subject and object.  Set code: an Elias-gamma object count
    per subject plus the objects.  The cheaper code decides.
    """

    out: dict[int, dict[str, float]] = {}
    objects = {}
    order = np.argsort(lab_u, kind="stable")
    ls, ys = lab_u[order], y_u[order]
    starts = np.flatnonzero(np.r_[True, ls[1:] != ls[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(ls)]):
        objects[int(ls[s])] = len(np.unique(ys[s:e]))
    order = np.argsort(g_lab, kind="stable")
    gl, gk = g_lab[order], g_nobj[order]
    starts = np.flatnonzero(np.r_[True, gl[1:] != gl[:-1]])
    for s, e in zip(starts, np.r_[starts[1:], len(gl)]):
        lab = int(gl[s])
        k = gk[s:e].astype(float)
        n_subj = float(len(k))
        log_o = math.log2(max(2, objects.get(lab, 2)))
        f_bits = n_subj * log_o + float((k - 1).sum()) * (math.log2(max(2.0, n_subj)) + log_o)
        s_bits = float(gamma_bits(k).sum() + k.sum() * log_o)
        out[lab] = {"subjects": n_subj, "objects": float(objects.get(lab, 0)), "mean_objects": float(k.mean()),
                    "functional_bits": f_bits, "set_bits": s_bits, "functional": bool(f_bits <= s_bits)}
    return out


@dataclass
class PatternFacts:
    """Per candidate label: shared subjects n, containment agreements, mode-equality agreements."""

    labels: np.ndarray
    n: np.ndarray
    a_class: np.ndarray
    a_eq: np.ndarray


def pattern_facts(index: FactIndex, corpus: Corpus, p: int, max_subjects: int = 2000) -> PatternFacts:
    occ = occurrences_of(corpus, int(p))
    e = index.n_ent
    xs = corpus.occ_left[occ].astype(np.int64)
    ys = corpus.occ_right[occ].astype(np.int64)
    subj = np.unique(xs)
    if len(subj) > max_subjects:  # deterministic subsample, declared in the preregistration
        rng = np.random.default_rng(int(p))
        subj = np.sort(rng.choice(subj, max_subjects, replace=False))
        keep = np.isin(xs, subj)
        xs, ys = xs[keep], ys[keep]
    pairs = np.unique(xs * e + ys)
    labs, row = csr_gather(index.ent_ptr, index.ent_label, subj)
    if len(labs) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return PatternFacts(empty, empty, empty, empty)
    modes, _ = csr_gather(index.ent_ptr, index.ent_mode, subj)
    labels_u, inv, n = np.unique(labs, return_inverse=True, return_counts=True)
    mode_code = subj[row] * e + modes
    pos = np.searchsorted(pairs, mode_code)
    hit = (pos < len(pairs)) & (pairs[np.minimum(pos, len(pairs) - 1)] == mode_code)
    a_eq = np.bincount(inv[hit], minlength=len(labels_u))
    pos = np.searchsorted(index.pair_code, pairs)
    found = (pos < len(index.pair_code)) & (index.pair_code[np.minimum(pos, len(index.pair_code) - 1)] == pairs)
    plabs, prow = csr_gather(index.pair_ptr, index.pair_label, pos[found])
    if len(plabs):
        px = (pairs[found] // e)[prow]
        u = np.unique(px * index.n_lab + plabs)
        ulab = u % index.n_lab
        li = np.searchsorted(labels_u, ulab)
        ok = (li < len(labels_u)) & (labels_u[np.minimum(li, len(labels_u) - 1)] == ulab)
        a_class = np.bincount(li[ok], minlength=len(labels_u))
    else:
        a_class = np.zeros(len(labels_u), dtype=np.int64)
    return PatternFacts(labels_u.astype(np.int64), n.astype(np.int64), a_class.astype(np.int64),
                        a_eq.astype(np.int64))


# ---------------------------------------------------------------------------
# Rates: Beta shrinkage, pair-specific LLR and predictions
# ---------------------------------------------------------------------------


def beta_prior(k: np.ndarray, n: np.ndarray, min_n: float = 20.0) -> tuple[float, float]:
    """Method-of-moments Beta(alpha, beta) over groups with at least ``min_n`` trials."""

    sel = n >= min_n
    if sel.sum() < 3:
        mean = float((k.sum() + 0.5) / (n.sum() + 1.0))
        return mean * 2.0, (1 - mean) * 2.0
    r = k[sel] / n[sel]
    mean, var = float(r.mean()), float(r.var())
    mean = min(max(mean, 1e-3), 1 - 1e-3)
    strength = mean * (1 - mean) / max(var, 1e-6) - 1.0
    strength = min(max(strength, 2.0), 1e4)
    return mean * strength, (1 - mean) * strength


def shrink(k: np.ndarray, n: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return (k + alpha) / (n + alpha + beta)


def clip_rate(x):
    return np.clip(x, 1e-4, 1 - 1e-4)


def merge_threshold_v(eps: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Merge iff a > c n: the LLR > 0 cut, vectorized."""

    agree = np.log((1 - eps) / pi)
    disagree = np.log(eps / (1 - pi))
    return -disagree / (agree - disagree)


def llr_v(a: np.ndarray, n: np.ndarray, eps: np.ndarray, pi: np.ndarray) -> np.ndarray:
    return a * np.log((1 - eps) / pi) + (n - a) * np.log(eps / (1 - pi))


def kl_bern_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * np.log(a / b) + (1 - a) * np.log((1 - a) / (1 - b))


def predicted_correct(n: np.ndarray, eps: np.ndarray, pi: np.ndarray, same: np.ndarray) -> np.ndarray:
    """Exact binomial probability that the LLR>0 rule decides each unit correctly."""

    from scipy.stats import binom

    c = merge_threshold_v(eps, pi)
    cut = np.floor(c * n + 1e-12)
    p_same = binom.sf(cut, n, 1 - eps)
    p_dist = binom.cdf(cut, n, pi)
    return np.where(same, p_same, p_dist)


def predicted_balanced(n: np.ndarray, eps: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Gold-free predicted balanced accuracy of the rule at overlap n."""

    return 0.5 * (predicted_correct(n, eps, pi, np.ones_like(n, dtype=bool))
                  + predicted_correct(n, eps, pi, np.zeros_like(n, dtype=bool)))


def n99_exact(eps: float, pi: float, n_max: int = 512) -> int | None:
    n = np.arange(1, n_max + 1)
    bal = predicted_balanced(n.astype(float), np.full(n_max, eps), np.full(n_max, pi))
    hit = np.flatnonzero(bal >= 0.99)
    return int(n[hit[0]]) if len(hit) else None


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------


def entity_roles(corpus: Corpus, split: int = 0):
    """(entities x 2*labels) counts: role 2l = left mention of label l, 2l+1 = right mention."""

    from scipy.sparse import coo_matrix

    n_ent, n_lab = len(corpus.entities), len(corpus.labels)
    occ = np.where(corpus.split[corpus.occ_pattern] == split)[0]
    labs = corpus.occ_labels[occ]
    rep = np.repeat(occ, labs.shape[1])
    lab = labs.ravel()
    keep = lab >= 0
    rep, lab = rep[keep], lab[keep].astype(np.int64)
    rows = np.r_[corpus.occ_left[rep], corpus.occ_right[rep]].astype(np.int64)
    cols = np.r_[2 * lab, 2 * lab + 1]
    m = coo_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n_ent, 2 * n_lab)).tocsr()
    m.sum_duplicates()
    return m


def type_top(roles) -> np.ndarray:
    nnz = np.diff(roles.indptr) > 0
    out = np.full(roles.shape[0], -1, dtype=np.int64)
    if nnz.any():
        out[nnz] = np.asarray(roles[nnz].argmax(axis=1)).ravel()
    return compact_types(out)


def compact_types(t: np.ndarray) -> np.ndarray:
    """Map type ids to 0..K-1 (by frequency, most frequent first); -1 stays -1."""

    out = np.full(len(t), -1, dtype=np.int64)
    known = t >= 0
    if known.any():
        vals, inv, counts = np.unique(t[known], return_inverse=True, return_counts=True)
        rank = np.empty(len(vals), dtype=np.int64)
        rank[np.argsort(-counts, kind="stable")] = np.arange(len(vals))
        out[known] = rank[inv]
    return out


def type_kmeans(roles, k: int, iters: int = 15, seed: int = 0) -> np.ndarray:
    """Spherical k-means over TF-IDF role profiles; entities without roles get -1."""

    from scipy.sparse import csr_matrix, diags

    nnz = np.diff(roles.indptr) > 0
    rows = np.flatnonzero(nnz)
    x = roles[rows].astype(np.float32)
    df = np.bincount(x.indices, minlength=x.shape[1]).astype(np.float32)
    idf = np.log((1 + x.shape[0]) / (1 + df)) + 1.0
    x = x @ diags(idf)
    x.data = np.log1p(x.data)
    norms = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    norms[norms == 0] = 1.0
    x = diags(1.0 / norms) @ x
    x = csr_matrix(x)
    rng = np.random.default_rng(seed)
    k = min(k, x.shape[0])
    centres = x[rng.choice(x.shape[0], k, replace=False)].toarray().astype(np.float32)
    assign = np.zeros(x.shape[0], dtype=np.int64)
    for _ in range(iters):
        sims = np.asarray(x @ centres.T)
        assign = sims.argmax(1)
        ind = csr_matrix((np.ones(len(assign), dtype=np.float32), (assign, np.arange(len(assign)))),
                         shape=(k, x.shape[0]))
        centres = np.asarray((ind @ x).todense(), dtype=np.float32)
        cn = np.linalg.norm(centres, axis=1)
        empty = cn == 0
        if empty.any():
            centres[empty] = x[rng.choice(x.shape[0], int(empty.sum()), replace=False)].toarray()
            cn[empty] = np.linalg.norm(centres[empty], axis=1)
        centres /= np.maximum(cn, 1e-12)[:, None]
    out = np.full(roles.shape[0], -1, dtype=np.int64)
    out[rows] = assign
    return compact_types(out)


def type_p31(corpus: Corpus, split: int = 0, top: int = 500) -> tuple[np.ndarray, dict[str, Any]]:
    """Most frequent Wikidata P31 (instance of) object per entity from aligned training facts."""

    lab_index = {name: i for i, name in enumerate(corpus.labels)}
    out = np.full(len(corpus.entities), -1, dtype=np.int64)
    info: dict[str, Any] = {"p31_labels": [n for n in ("P31:f", "P31:r") if n in lab_index]}
    if not info["p31_labels"]:
        info["entities_typed"] = 0
        return out, info
    occ = np.where(corpus.split[corpus.occ_pattern] == split)[0]
    subj_all, obj_all = [], []
    for name in info["p31_labels"]:
        lid = lab_index[name]
        hit = occ[(corpus.occ_labels[occ] == lid).any(1)]
        left, right = corpus.occ_left[hit].astype(np.int64), corpus.occ_right[hit].astype(np.int64)
        subj, obj = (left, right) if name.endswith(":f") else (right, left)
        subj_all.append(subj)
        obj_all.append(obj)
    subj, obj = np.concatenate(subj_all), np.concatenate(obj_all)
    if len(subj) == 0:
        info["entities_typed"] = 0
        return out, info
    vals, counts = np.unique(obj, return_counts=True)
    keep_types = vals[np.argsort(-counts, kind="stable")[:top]]
    obj_c = np.where(np.isin(obj, keep_types), obj, -2)
    key = subj * (len(corpus.entities) + 2) + (obj_c + 2)
    uk, cnt = np.unique(key, return_counts=True)
    s_u, o_u = uk // (len(corpus.entities) + 2), uk % (len(corpus.entities) + 2) - 2
    order = np.lexsort((o_u, -cnt, s_u))
    first = order[np.r_[True, s_u[order][1:] != s_u[order][:-1]]]
    out[s_u[first]] = o_u[first] + 2  # shift so "other" (-2) becomes 0
    info["entities_typed"] = int((out >= 0).sum())
    return compact_types(out), info


def pattern_cells(corpus: Corpus, ids: np.ndarray, etype: np.ndarray) -> np.ndarray:
    """Most frequent (left type, right type) of each pattern's occurrences, as (n, 2); -1 unknown."""

    out = np.full((len(ids), 2), -1, dtype=np.int64)
    k = int(etype.max()) + 2 if len(etype) else 2
    for r, p in enumerate(ids):
        occ = occurrences_of(corpus, int(p))
        lt = etype[corpus.occ_left[occ]] + 1
        rt = etype[corpus.occ_right[occ]] + 1
        code = lt * k + rt
        vals, counts = np.unique(code, return_counts=True)
        best = vals[np.argmax(counts)]
        out[r] = (best // k - 1, best % k - 1)
    return out


def _type_codes(corpus: Corpus, p: int, etype: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    occ = occurrences_of(corpus, int(p))
    lt = etype[corpus.occ_left[occ]] + 1
    rt = etype[corpus.occ_right[occ]] + 1
    return lt, rt, lt * k + rt


@dataclass
class TypeVocab:
    """Type feature columns seen in the labelled set: left types, right types, type pairs."""

    k: int
    blocks: list[np.ndarray]  # sorted codes per block
    offsets: np.ndarray

    @property
    def size(self) -> int:
        return int(self.offsets[-1])


def type_vocab(corpus: Corpus, ids: np.ndarray, etype: np.ndarray) -> TypeVocab:
    k = int(etype.max()) + 2 if len(etype) else 2
    seen: list[set[int]] = [set(), set(), set()]
    for p in ids:
        for b, codes in enumerate(_type_codes(corpus, int(p), etype, k)):
            seen[b].update(np.unique(codes).tolist())
    blocks = [np.array(sorted(x), dtype=np.int64) for x in seen]
    offsets = np.cumsum([0, *[len(b) for b in blocks]])
    return TypeVocab(k, blocks, offsets)


def type_matrix(corpus: Corpus, ids: np.ndarray, etype: np.ndarray, vocab: TypeVocab):
    """Sparse type features per pattern over ``vocab``: left type, right type and their pair, weighted by
    occurrence share; each block L2-normalised and scaled by 1/sqrt(3).  Unseen codes are dropped."""

    from scipy.sparse import csr_matrix

    rows, cols, vals = [], [], []
    for r, p in enumerate(ids):
        codes = _type_codes(corpus, int(p), etype, vocab.k)
        tot = float(len(codes[0]))
        for b, c in enumerate(codes):
            v, cnt = np.unique(c, return_counts=True)
            pos = np.searchsorted(vocab.blocks[b], v)
            pos_c = np.minimum(pos, max(0, len(vocab.blocks[b]) - 1))
            ok = (pos < len(vocab.blocks[b])) & (vocab.blocks[b][pos_c] == v) if len(vocab.blocks[b]) else \
                np.zeros(len(v), dtype=bool)
            if not ok.any():
                continue
            w = cnt[ok] / tot
            w = w / max(np.sqrt((w * w).sum()), 1e-12) / math.sqrt(3.0)
            rows.extend([r] * int(ok.sum()))
            cols.extend((vocab.offsets[b] + pos[ok]).tolist())
            vals.extend(w.tolist())
    return csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(ids), vocab.size))


# ---------------------------------------------------------------------------
# Type program: MDL choice per type cell among star, product and flat
# ---------------------------------------------------------------------------


@dataclass
class CellModel:
    kind: str  # star | product | flat
    members: np.ndarray  # indices into the labelled set
    majority: int
    counts: np.ndarray  # label counts over members (values,)
    bits: dict[str, float]
    tax: FieldTaxonomy | None = None


@dataclass
class TypeProgram:
    values: int
    cells: dict[tuple[int, int], CellModel]
    left: dict[int, tuple[int, int]]  # left type -> (majority, members)
    right: dict[int, tuple[int, int]]
    bits: float
    flat_bits: float
    kinds: dict[str, int]


def _star_bits(counts: np.ndarray, values: int) -> float:
    n = int(counts.sum())
    exceptions = n - int(counts.max())
    return math.log2(values) + exceptions * (math.log2(max(2, n)) + math.log2(values))


def fit_type_program(cells: np.ndarray, labels: np.ndarray, x_lab, values: int, min_product: int = 3,
                     max_product: int = 12_000) -> TypeProgram:
    """``cells`` (n, 2) type cell per labelled pattern; ``labels`` compact relation ids.  Cells with more
    than ``max_product`` members choose between star and flat only (tree workspace bound)."""

    log_v = math.log2(values)
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (lt, rt) in enumerate(cells):
        groups.setdefault((int(lt), int(rt)), []).append(i)
    models: dict[tuple[int, int], CellModel] = {}
    kinds = {"star": 0, "product": 0, "flat": 0}
    total = 0.0
    for key, members in groups.items():
        m = np.asarray(members, dtype=np.int64)
        y = labels[m]
        counts = np.bincount(y, minlength=values)
        bits = {"star": _star_bits(counts, values), "flat": len(m) * log_v}
        tax = None
        if min_product <= len(m) <= max_product and counts.max() < len(m):
            tree = cosine_linkage_tree(x_lab[m])
            program = compress(tree, y.reshape(-1, 1), values)
            bits["product"] = program.entries() * (math.log2(tree.size) + log_v)
            tax = FieldTaxonomy(tree, LCA(tree), program.labels[0], program.entries())
        kind = min(bits, key=lambda k_: (bits[k_], k_))
        kinds[kind] += 1
        total += bits[kind]
        models[key] = CellModel(kind, m, int(counts.argmax()), counts, bits, tax if kind == "product" else None)
    left: dict[int, tuple[int, int]] = {}
    right: dict[int, tuple[int, int]] = {}
    for side, table in ((0, left), (1, right)):
        order = np.argsort(cells[:, side], kind="stable")
        s = cells[order, side]
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        for a, b in zip(starts, np.r_[starts[1:], len(s)]):
            y = labels[order[a:b]]
            table[int(s[a])] = (int(np.bincount(y, minlength=values).argmax()), int(b - a))
    return TypeProgram(values, models, left, right, total, len(labels) * log_v, kinds)


def cell_neighbours(prog: TypeProgram, q_cells: np.ndarray, q_x, x_lab, m_max: int,
                    batch: int = 2000) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Top-``m_max`` word neighbours of each query inside its product cell: cell -> (rows, idx, sim)."""

    out: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    by_cell: dict[tuple[int, int], list[int]] = {}
    for r, (lt, rt) in enumerate(q_cells):
        by_cell.setdefault((int(lt), int(rt)), []).append(r)
    for key, rows in by_cell.items():
        model = prog.cells.get(key)
        if model is None or model.kind != "product":
            continue
        rows_a = np.asarray(rows, dtype=np.int64)
        base = x_lab[model.members].T.tocsc()
        kk = min(m_max, len(model.members))
        idx = np.zeros((len(rows_a), kk), dtype=np.int64)
        sim = np.zeros((len(rows_a), kk), dtype=np.float32)
        for s in range(0, len(rows_a), batch):
            block = np.asarray((q_x[rows_a[s:s + batch]] @ base).todense())
            order = np.argsort(-block, axis=1, kind="stable")[:, :kk]
            idx[s:s + batch] = order
            sim[s:s + batch] = np.take_along_axis(block, order, 1)
        out[key] = (rows_a, idx, sim)
    return out


def predict_type_program(prog: TypeProgram, q_cells: np.ndarray, flat_pred: np.ndarray,
                         neigh: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]],
                         tau: float, m: int, backoff_min: int) -> tuple[np.ndarray, dict[str, int]]:
    """Hard predictions (compact ids).  ``flat_pred``: word-only prediction per query, used for flat
    cells and for unseen cells without a usable one-sided backoff."""

    out = flat_pred.copy()
    route = {"star": 0, "product": 0, "flat": 0, "left": 0, "right": 0, "words": 0}
    by_cell: dict[tuple[int, int], list[int]] = {}
    for r, (lt, rt) in enumerate(q_cells):
        by_cell.setdefault((int(lt), int(rt)), []).append(r)
    for key, rows in by_cell.items():
        rows_a = np.asarray(rows, dtype=np.int64)
        model = prog.cells.get(key)
        if model is not None:
            if model.kind == "star":
                out[rows_a] = model.majority
                route["star"] += len(rows)
            elif model.kind == "product":
                n_rows, idx, sim = neigh[key]
                kk = min(m, idx.shape[1])
                out[n_rows] = model.tax.predict(idx[:, :kk], sim[:, :kk], tau, kk, model.majority)
                route["product"] += len(rows)
            else:
                route["flat"] += len(rows)
            continue
        lt, rt = key
        left, right = prog.left.get(lt), prog.right.get(rt)
        if left is not None and left[1] >= backoff_min and (right is None or left[1] >= right[1]):
            out[rows_a] = left[0]
            route["left"] += len(rows)
        elif right is not None and right[1] >= backoff_min:
            out[rows_a] = right[0]
            route["right"] += len(rows)
        else:
            route["words"] += len(rows)
    return out, route


@dataclass
class LatticeNeighbours:
    """Per query: the most specific family level with members, and top word neighbours inside it."""

    level: np.ndarray  # 0 cell, 1 left type, 2 right type, 3 all labelled
    family_counts: np.ndarray  # (n, V) label counts of the chosen family
    idx: np.ndarray  # (n, k) labelled indices (-1 padded)
    sim: np.ndarray  # (n, k)


def _family_members(cells: np.ndarray) -> tuple[dict, dict, dict]:
    cell: dict[tuple[int, int], list[int]] = {}
    left: dict[int, list[int]] = {}
    right: dict[int, list[int]] = {}
    for i, (lt, rt) in enumerate(cells):
        cell.setdefault((int(lt), int(rt)), []).append(i)
        left.setdefault(int(lt), []).append(i)
        right.setdefault(int(rt), []).append(i)
    return ({k: np.asarray(v, dtype=np.int64) for k, v in cell.items()},
            {k: np.asarray(v, dtype=np.int64) for k, v in left.items()},
            {k: np.asarray(v, dtype=np.int64) for k, v in right.items()})


def lattice_neighbours(lab_cells: np.ndarray, labels: np.ndarray, values: int, x_lab, q_cells: np.ndarray, q_x,
                       global_idx: np.ndarray, global_sim: np.ndarray, k: int, min_members: int,
                       batch: int = 2000) -> LatticeNeighbours:
    """Place each query in the most specific family of the product lattice cell < {left type, right type} < all
    that has at least ``min_members`` labelled members (the larger one-sided family when both qualify), then
    rank that family's members by word similarity.  Level 3 reuses the global word neighbours."""

    cell_m, left_m, right_m = _family_members(lab_cells)
    n = len(q_cells)
    level = np.full(n, 3, dtype=np.int64)
    fam_key: list[Any] = [None] * n
    for r, (lt, rt) in enumerate(q_cells):
        key = (int(lt), int(rt))
        if key in cell_m and len(cell_m[key]) >= min_members:
            level[r], fam_key[r] = 0, key
            continue
        lm, rm = left_m.get(int(lt)), right_m.get(int(rt))
        ln = len(lm) if lm is not None else 0
        rn = len(rm) if rm is not None else 0
        if max(ln, rn) >= min_members:
            level[r], fam_key[r] = (1, int(lt)) if ln >= rn else (2, int(rt))
    idx = np.full((n, k), -1, dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    counts = np.zeros((n, values), dtype=np.float32)
    all_counts = np.bincount(labels, minlength=values).astype(np.float32)
    groups: dict[tuple[int, Any], list[int]] = {}
    for r in range(n):
        groups.setdefault((int(level[r]), fam_key[r]), []).append(r)
    for (lev, key), rows in groups.items():
        rows_a = np.asarray(rows, dtype=np.int64)
        if lev == 3:
            kk = min(k, global_idx.shape[1])
            idx[rows_a, :kk] = global_idx[rows_a, :kk]
            sim[rows_a, :kk] = global_sim[rows_a, :kk]
            counts[rows_a] = all_counts
            continue
        members = (cell_m if lev == 0 else (left_m if lev == 1 else right_m))[key]
        counts[rows_a] = np.bincount(labels[members], minlength=values)
        base = x_lab[members].T.tocsc()
        kk = min(k, len(members))
        for s in range(0, len(rows_a), batch):
            block = np.asarray((q_x[rows_a[s:s + batch]] @ base).todense())
            if block.shape[1] > kk:
                part = np.argpartition(-block, kk - 1, axis=1)[:, :kk]
            else:
                part = np.tile(np.arange(block.shape[1]), (block.shape[0], 1))
            vals = np.take_along_axis(block, part, 1)
            order = np.argsort(-vals, axis=1, kind="stable")
            idx[rows_a[s:s + batch], :kk] = members[np.take_along_axis(part, order, 1)]
            sim[rows_a[s:s + batch], :kk] = np.take_along_axis(vals, order, 1)
    return LatticeNeighbours(level, counts, idx, sim)


def lattice_distribution(nb: LatticeNeighbours, labels: np.ndarray, values: int, k: int, beta: float) -> np.ndarray:
    """Similarity-weighted vote of the top-k family members plus ``beta`` x the family's label frequencies.
    With no positive-similarity member the family prior alone decides (its majority)."""

    n = nb.idx.shape[0]
    kk = min(k, nb.idx.shape[1])
    ids, s = nb.idx[:, :kk], np.clip(nb.sim[:, :kk], 0.0, None)
    s = np.where(ids >= 0, s, 0.0)
    votes = np.zeros((n, values), dtype=np.float32)
    rows = np.repeat(np.arange(n), kk)
    np.add.at(votes, (rows, labels[np.maximum(ids, 0)].ravel()), s.ravel())
    tot = votes.sum(1, keepdims=True)
    votes = np.where(tot > 0, votes / np.maximum(tot, 1e-12), 0.0)
    prior = nb.family_counts / np.maximum(nb.family_counts.sum(1, keepdims=True), 1e-12)
    out = votes + beta * prior
    none = tot[:, 0] <= 0
    out[none] = prior[none]
    return out


# ---------------------------------------------------------------------------
# Graded sources and the geometric mixer
# ---------------------------------------------------------------------------

COUNT_EDGES = (1, 3, 10, 100)  # bucket 0: none, 1: 1-2, 2: 3-9, 3: 10-99, 4: 100+
SIM_EDGES = (1e-9, 0.3, 0.6)  # bucket 0: none, 1: <0.3, 2: <0.6, 3: >=0.6


def count_bucket(n: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(COUNT_EDGES), n, side="right")


def sim_bucket(s: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.asarray(SIM_EDGES), s, side="right")


def smoothed_log(counts: np.ndarray, prior: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """log((counts + alpha * prior) / (n + alpha)) row-wise; counts (n, V)."""

    n = counts.sum(1, keepdims=True)
    return np.log((counts + alpha * prior[None]) / (n + alpha)).astype(np.float32)


def knn_distribution(idx: np.ndarray, sim: np.ndarray, labels: np.ndarray, values: int, k: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], values), dtype=np.float32)
    s = np.clip(sim[:, :k], 0.0, None)
    rows = np.repeat(np.arange(idx.shape[0]), min(k, idx.shape[1]))
    np.add.at(out, (rows, labels[idx[:, :k]].ravel()), s.ravel())
    return out


@dataclass
class Sources:
    """log-probabilities (n, V) per source and the evidence bucket of each row."""

    logp: dict[str, np.ndarray]
    bucket: dict[str, np.ndarray]


def type_sources(prog: TypeProgram, q_cells: np.ndarray, prior: np.ndarray, alpha: float = 1.0) -> Sources:
    v = prog.values
    n = len(q_cells)
    cell_c = np.zeros((n, v), dtype=np.float32)
    left_c = np.zeros((n, v), dtype=np.float32)
    right_c = np.zeros((n, v), dtype=np.float32)
    left_counts: dict[int, np.ndarray] = {}
    right_counts: dict[int, np.ndarray] = {}
    for (lt, rt), model in prog.cells.items():
        left_counts[lt] = left_counts.get(lt, 0) + model.counts
        right_counts[rt] = right_counts.get(rt, 0) + model.counts
    for r, (lt, rt) in enumerate(q_cells):
        model = prog.cells.get((int(lt), int(rt)))
        if model is not None:
            cell_c[r] = model.counts
        if int(lt) in left_counts:
            left_c[r] = left_counts[int(lt)]
        if int(rt) in right_counts:
            right_c[r] = right_counts[int(rt)]
    logp = {"cell": smoothed_log(cell_c, prior, alpha), "left": smoothed_log(left_c, prior, alpha),
            "right": smoothed_log(right_c, prior, alpha)}
    bucket = {"cell": count_bucket(cell_c.sum(1)), "left": count_bucket(left_c.sum(1)),
              "right": count_bucket(right_c.sum(1))}
    return Sources(logp, bucket)


def fit_mixer(sources: Sources, target: np.ndarray, extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
              steps: int = 300, seed: int = 0) -> dict[str, np.ndarray]:
    """Geometric mixing weights w[source][bucket] >= 0 minimising the cross-entropy of ``target``.

    logits = sum_k w[k][bucket_k(row)] * M_k(row, :).  The objective is convex in w (a log-linear
    model with fixed features), so L-BFGS-B with bounds w >= 0 finds the optimum; ``extra`` adds
    further score matrices (e.g. fact LLRs by evidence bucket) with their own weights.
    """

    from scipy.optimize import minimize

    del seed
    mats = dict(sources.logp)
    buckets = dict(sources.bucket)
    for k, (m, b) in (extra or {}).items():
        mats[k], buckets[k] = m, b
    names = list(mats)
    sizes = [int(buckets[k].max()) + 1 if len(buckets[k]) else 1 for k in names]
    offsets = np.cumsum([0, *sizes])
    n = len(target)
    rows = np.arange(n)

    def unpack(w: np.ndarray) -> dict[str, np.ndarray]:
        return {k: w[offsets[i]:offsets[i + 1]] for i, k in enumerate(names)}

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        ws = unpack(w)
        logits = np.zeros_like(mats[names[0]], dtype=np.float64)
        for k in names:
            logits += ws[k][buckets[k]][:, None] * mats[k]
        logits -= logits.max(1, keepdims=True)
        lse = np.log(np.exp(logits).sum(1))
        loss = float((lse - logits[rows, target]).mean())
        g = np.exp(logits - lse[:, None])
        g[rows, target] -= 1.0
        grad = np.zeros_like(w)
        for i, k in enumerate(names):
            per_row = (g * mats[k]).sum(1)
            grad[offsets[i]:offsets[i + 1]] = np.bincount(buckets[k], weights=per_row, minlength=sizes[i]) / n
        return loss, grad

    w0 = np.ones(int(offsets[-1])) * 0.5
    res = minimize(objective, w0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(w0),
                   options={"maxiter": steps})
    return {k: np.asarray(v, dtype=np.float64) for k, v in unpack(res.x).items()}


def mix_scores(sources: Sources, weights: dict[str, np.ndarray],
               extra: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> np.ndarray:
    total = None
    for k, m in sources.logp.items():  # a bucket never seen when fitting takes the nearest fitted one
        term = weights[k][np.minimum(sources.bucket[k], len(weights[k]) - 1)][:, None] * m
        total = term if total is None else total + term
    for k, (m, b) in (extra or {}).items():
        total = total + weights[k][np.minimum(b, len(weights[k]) - 1)][:, None] * m
    return total


def log_softmax_rows(scores: np.ndarray) -> np.ndarray:
    m = scores.max(1, keepdims=True)
    z = scores - m
    return (z - np.log(np.exp(z).sum(1, keepdims=True))).astype(np.float32)
