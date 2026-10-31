"""KG v2: frozen replication, additive rule evidence, RotatE charts, bootstrap.

See ``docs/kg-v2-findings.md`` for the preregistration.  Everything here reuses
the track-B field program (``kg_field``); this module adds

- ``frozen_config``: the post hoc WN18RR method, frozen;
- ``grid_ranks``: filtered ranks for several prototype weights in one pass;
- ``feature_batch``: 15 exact features per (query, candidate) from rule firings;
- ``fit_additive`` / ``additive_score``: a linear score over those features,
  fitted by softmax cross-entropy on validation queries;
- ``RotatE``: the official RotatE scoring function over a trained checkpoint;
- ``bootstrap_ci`` / ``paired_ci``: percentile bootstrap over test queries.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .kg_data import KG, filtered_rank, to_queries
from .kg_field import FieldConfig, Program, prototype_scores, score_batch

LAMBDAS = (0.0, 0.001, 0.01, 0.1, 1.0)
FEATURES = (
    "l1_max", "l1_sum", "l1_cnt",
    "l2_max", "l2_sum", "l2_cnt",
    "l3_max", "l3_sum", "l3_cnt",
    "c_max", "c_sum", "c_cnt",
    "noisy", "prior", "proto",
)
_CLASS = {"l1": 0, "l2": 1, "l3": 2, "c": 3}


def frozen_config(seed: int, path_pairs: str = "exact", **overrides) -> FieldConfig:
    """The post hoc WN18RR program of track B (kg-field-wn18rr-proto v3), frozen."""

    config = FieldConfig(families="none", pc=1.0, aggregation="max", max_path_len=3, path3_samples=300,
                         path3_fanout=48, max_const_rules=200_000, path_pairs=path_pairs, seed=seed)
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def rr(higher_tied: np.ndarray) -> np.ndarray:
    ht = np.asarray(higher_tied, dtype=np.float64)
    return 1.0 / (1.0 + ht[:, 0] + ht[:, 1] / 2.0)


# ---------------------------------------------------------------------------
# Ranks for several prototype weights in one pass
# ---------------------------------------------------------------------------


def grid_ranks(program: Program, split: np.ndarray, R: int, known: dict, lambdas=LAMBDAS,
               batch: int = 256) -> dict[float, np.ndarray]:
    """Filtered (higher, tied) per query of a split (tail then head queries), per prototype weight."""

    return grid_ranks_queries(program, to_queries(split, R), known, lambdas, batch)


def grid_ranks_queries(program: Program, queries: np.ndarray, known: dict, lambdas=LAMBDAS,
                       batch: int = 256) -> dict[float, np.ndarray]:
    """Track B's scorer (maximum aggregation) plus lambda * prototype, for given (e, q, a) rows."""

    out = {lam: np.zeros((queries.shape[0], 2), dtype=np.int64) for lam in lambdas}
    saved = program.config.prototype_weight
    program.config.prototype_weight = 0.0
    try:
        for q in np.unique(queries[:, 1]):
            idx = np.where(queries[:, 1] == q)[0]
            for start in range(0, idx.shape[0], batch):
                sel = idx[start:start + batch]
                xs = queries[sel, 0]
                base = score_batch(program, int(q), xs)
                proto = prototype_scores(program, int(q), xs) if any(lam > 0 for lam in lambdas) else 0.0
                for lam in lambdas:
                    scores = base + lam * proto if lam > 0 else base
                    for row, i in enumerate(sel):
                        e, qq, a = queries[i]
                        out[lam][i] = filtered_rank(scores[row], int(a), known[(int(e), int(qq))])
    finally:
        program.config.prototype_weight = saved
    return out


# ---------------------------------------------------------------------------
# Additive evidence
# ---------------------------------------------------------------------------


def feature_batch(program: Program, q: int, xs: np.ndarray) -> np.ndarray:
    """(15, Q, n) exact features of every candidate for queries (x, q, ?).

    Rule application mirrors ``kg_field.score_batch`` (lazy path joins on the
    query rows, the same object-identity conditions); instead of a maximum it
    records per rule class the maximum, the sum and log(1 + count) of the
    confidences firing on each candidate, plus the noisy-or over all rules,
    the candidate's log training count as an answer of q, and the prototype
    overlap score.
    """

    n, Q = program.n, xs.shape[0]
    F = np.zeros((len(FEATURES), Q, n), dtype=np.float32)
    s_log = np.zeros((Q, n), dtype=np.float64)

    def add(cls: str, rows: np.ndarray, cols: np.ndarray, conf) -> None:
        if rows.size == 0:
            return
        base = 3 * _CLASS[cls]
        np.maximum.at(F[base], (rows, cols), conf)
        np.add.at(F[base + 1], (rows, cols), conf)
        np.add.at(F[base + 2], (rows, cols), 1.0)
        np.add.at(s_log, (rows, cols), np.log1p(-np.minimum(conf, 0.999999)))

    for k, conf in program.l1.get(q, []):
        sub = program.A[k][xs].tocoo()
        add("l1", sub.row, sub.col, conf)
    for (a, b), conf in program.l2.get(q, []):
        sub = (program.A[a][xs] @ program.A[b]).tocoo()
        keep = sub.col != xs[sub.row]
        add("l2", sub.row[keep], sub.col[keep], conf)
    for (a, b, c), conf in program.l3.get(q, []):
        inner = (program.A[a][xs] @ program.A[b]).tocoo()
        keep = inner.col != xs[inner.row]
        inner = sp.csr_matrix((np.ones(int(keep.sum()), dtype=np.float32), (inner.row[keep], inner.col[keep])),
                              shape=(Q, n))
        sub = (inner @ program.A[c]).tocoo()
        keep = sub.col != xs[sub.row]
        add("l3", sub.row[keep], sub.col[keep], conf)
    table = program.const.get(q)
    if table is not None and table.nnz:
        xq = program.X[xs].tocoo()
        rc = table[xq.col].tocoo()
        if rc.nnz:
            add("c", xq.row[rc.row], rc.col, rc.data.astype(np.float32))
    for cls in _CLASS:
        cnt = 3 * _CLASS[cls] + 2
        F[cnt] = np.log1p(F[cnt])
    F[12] = (1.0 - np.exp(s_log)).astype(np.float32)
    prior = program.prior.get(q)
    if prior is not None:
        F[13] = np.log1p(prior)[None, :]
    F[14] = prototype_scores(program, q, xs).astype(np.float32)
    return F


def max_score(F: np.ndarray, lam: float) -> np.ndarray:
    """Track B's maximum-aggregation score reconstructed from the features."""

    s_max = np.maximum(np.maximum(F[0], F[3]), np.maximum(F[6], F[9])).astype(np.float64)
    prior_count = np.expm1(F[13].astype(np.float64))
    return s_max + 1e-3 * F[12] + 1e-9 * prior_count + lam * F[14]


@dataclass
class Additive:
    weights: np.ndarray  # on raw features (standardization folded in)
    mu: float
    names: tuple[str, ...]

    def score(self, F: np.ndarray, extra: np.ndarray | None = None) -> np.ndarray:
        feats = F if extra is None else np.concatenate([F, extra[None]], axis=0)
        return np.tensordot(self.weights, feats.astype(np.float64), axes=(0, 0))


def _self_and_known_mask(scores: np.ndarray, xs: np.ndarray, answers: np.ndarray, known_rows: list[np.ndarray]) -> np.ndarray:
    """Boolean (Q, n): candidates that count in the filtered ranking (answer included)."""

    allowed = np.ones(scores.shape, dtype=bool)
    for row, (x, a, kn) in enumerate(zip(xs, answers, known_rows)):
        allowed[row, kn] = False
        allowed[row, a] = True
        allowed[row, x] = False
    return allowed


def candidate_sets(F: np.ndarray, xs: np.ndarray, answers: np.ndarray, known_rows: list[np.ndarray], lam: float,
                   top: int, rand: int, rng: np.random.Generator,
                   extra: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(Q, 1 + top + rand, d) features with column 0 the answer, and the valid column count per query."""

    base = max_score(F, lam)
    allowed = _self_and_known_mask(base, xs, answers, known_rows)
    feats = F if extra is None else np.concatenate([F, extra[None]], axis=0)
    out = np.zeros((xs.shape[0], 1 + top + rand, feats.shape[0]), dtype=np.float32)
    counts = np.zeros(xs.shape[0], dtype=np.int64)
    for row, a in enumerate(answers):
        ok = allowed[row].copy()
        ok[a] = False
        pool = np.where(ok)[0]
        order = pool[np.argsort(-base[row, pool], kind="stable")][:top]
        rest = np.setdiff1d(pool, order, assume_unique=False)
        rnd = rng.choice(rest, size=min(rand, rest.size), replace=False) if rest.size else np.array([], dtype=np.int64)
        cols = np.concatenate([[a], order, rnd]).astype(np.int64)
        counts[row] = cols.size
        out[row, : cols.size] = feats[:, row, cols].T
    return out, counts


def fit_additive(sets: np.ndarray, valid_counts: np.ndarray, mu: float, names: tuple[str, ...]) -> Additive:
    """Softmax cross-entropy over candidate sets (answer at column 0), L2 on standardized weights."""

    from scipy.optimize import minimize

    Q, C, d = sets.shape
    mask = np.arange(C)[None, :] < valid_counts[:, None]
    flat = sets[mask]
    mean = flat.mean(0)
    std = flat.std(0)
    std[std < 1e-8] = 1.0
    Z = ((sets - mean) / std).astype(np.float64)
    Z[~mask] = 0.0
    neg_inf_mask = np.where(mask, 0.0, -np.inf)

    def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
        logits = Z @ w + neg_inf_mask
        m = logits.max(1, keepdims=True)
        p = np.exp(logits - m)
        z = p.sum(1, keepdims=True)
        p /= z
        loss = float((np.log(z[:, 0]) + m[:, 0] - logits[:, 0]).mean()) + mu * float(w @ w)
        grad = np.einsum("qc,qcd->d", p, Z) / Q - Z[:, 0].mean(0) + 2 * mu * w
        return loss, grad

    w0 = np.zeros(d)
    w0[0] = 1.0
    res = minimize(objective, w0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return Additive(weights=res.x / std, mu=mu, names=names)


# ---------------------------------------------------------------------------
# RotatE (official scoring function)
# ---------------------------------------------------------------------------


class RotatE:
    """score(h, r, t) = gamma - sum_d |h_d * r_d - t_d| with unit-modulus relations."""

    def __init__(self, directory: str | Path, threads: int = 8) -> None:
        import torch

        torch.set_num_threads(threads)
        directory = Path(directory)
        cfg = json.loads((directory / "config.json").read_text())
        self.gamma = float(cfg["gamma"])
        hidden = int(cfg["hidden_dim"])
        embedding_range = (self.gamma + 2.0) / hidden  # epsilon = 2.0 in the official model
        entity = np.load(directory / "entity_embedding.npy").astype(np.float32)
        relation = np.load(directory / "relation_embedding.npy").astype(np.float32)
        self.re = torch.from_numpy(entity[:, :hidden].copy())
        self.im = torch.from_numpy(entity[:, hidden:].copy())
        phase = relation / (embedding_range / math.pi)
        self.cr = torch.from_numpy(np.cos(phase))
        self.sr = torch.from_numpy(np.sin(phase))
        self.R = relation.shape[0]
        self.params = int(entity.size + relation.size)

    def _anchor(self, q: int, xs: np.ndarray):
        import torch

        r = q % self.R
        idx = torch.from_numpy(np.asarray(xs, dtype=np.int64))
        x_re, x_im = self.re[idx], self.im[idx]
        cr, sr = self.cr[r], self.sr[r]
        if q < self.R:  # (x, r, ?): tail-batch, compare h o r with every t
            return x_re * cr - x_im * sr, x_re * sr + x_im * cr
        # (?, r, x): head-batch, compare t o conj(r) with every h
        return cr * x_re + sr * x_im, cr * x_im - sr * x_re

    def scores(self, q: int, xs: np.ndarray, chunk: int = 1024, qchunk: int = 64) -> np.ndarray:
        import torch

        n = self.re.shape[0]
        out = np.empty((len(xs), n), dtype=np.float32)
        with torch.no_grad():
            for qs in range(0, len(xs), qchunk):
                a_re, a_im = self._anchor(q, xs[qs:qs + qchunk])
                for start in range(0, n, chunk):
                    d_re = a_re[:, None, :] - self.re[None, start:start + chunk]
                    d_im = a_im[:, None, :] - self.im[None, start:start + chunk]
                    out[qs:qs + qchunk, start:start + chunk] = (self.gamma - torch.sqrt(d_re * d_re + d_im * d_im).sum(-1)).numpy()
        return out

    def ranks(self, split: np.ndarray, R: int, known: dict, batch: int = 256) -> np.ndarray:
        queries = to_queries(split, R)
        out = np.zeros((queries.shape[0], 2), dtype=np.int64)
        for q in np.unique(queries[:, 1]):
            idx = np.where(queries[:, 1] == q)[0]
            for start in range(0, idx.shape[0], batch):
                sel = idx[start:start + batch]
                scores = self.scores(int(q), queries[sel, 0])
                scores[np.arange(sel.size), queries[sel, 0]] = -np.inf  # as for the field program
                for row, i in enumerate(sel):
                    e, qq, a = queries[i]
                    out[i] = filtered_rank(scores[row], int(a), known[(int(e), int(qq))])
        return out


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(values: np.ndarray, resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = values.shape[0]
    means = np.empty(resamples)
    for b in range(resamples):
        means[b] = values[rng.integers(0, n, n)].mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_ci(a: np.ndarray, b: np.ndarray, resamples: int = 10_000, seed: int = 0) -> tuple[float, float, float]:
    return bootstrap_ci(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64), resamples, seed)


__all__ = ["LAMBDAS", "FEATURES", "frozen_config", "rr", "grid_ranks", "grid_ranks_queries", "feature_batch", "max_score", "Additive",
           "candidate_sets", "fit_additive", "RotatE", "bootstrap_ci", "paired_ci", "KG"]
