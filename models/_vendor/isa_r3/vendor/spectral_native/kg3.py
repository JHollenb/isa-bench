"""KG v3: MDL-pruned constant rules, additive evidence everywhere, gated hybrid.

See ``docs/kg-v3-findings.md`` for the preregistration.  Builds on ``kg2``:

- ``const_gains`` / ``mdl_prune``: keep a constant rule q(x, c) <- b(x) only if the
  bits it saves on the training graph pay its description length (the v2 ruler,
  ``program_bits``).  ``IND`` scores every rule alone against the relation's base
  rate; ``NR`` (non-redundant) adds rules greedily per (q, c) under a noisy-or with
  leak and keeps a rule only if its *marginal* saving pays.  Under the noisy-or the
  marginal saving of a rule can only fall as rules are added (negatives always cost
  log(1 - conf); positives gain log(P_new / P_old), decreasing in P_old), so NR is
  run only over IND survivors.
- ``pruned_program``: a shallow copy of a learned program with pruned tables.
- ``query_features``: per-query gate inputs.
- ``fit_gated`` / ``Gated``: linear evidence whose weights depend on the query's
  relation (shrunk to a shared vector) and on query features; convex, softmax
  cross-entropy over the v2 candidate sets.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .kg_field import Program

GATE_FEATURES = ("bias", "log_fired", "log_known", "log_rel_fanout", "top_rule", "rotate_margin")


# ---------------------------------------------------------------------------
# MDL pruning of constant rules
# ---------------------------------------------------------------------------


def const_rule_bits(program: Program) -> float:
    """Description length of one constant rule under the v2 ruler (kg_field.program_bits)."""

    lg_n, lg_q = math.log2(program.n), math.log2(2 * program.R)
    return lg_q + math.log2(program.X.shape[1]) + lg_n + 16


def base_rates(program: Program, q: int) -> np.ndarray:
    """p0(c): share of entities that are a training answer c of q (leak of the noisy-or)."""

    prior = program.prior.get(q)
    if prior is None:
        return np.full(program.n, 0.5 / program.n)
    return (prior.astype(np.float64) + 0.5) / (program.n + 1.0)


def const_gains(program: Program, q: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(b, c, conf, standalone gain in bits) for every stored constant rule of q.

    Standalone gain: the log-likelihood ratio, over the rule's body population, of the
    noisy-or P = 1 - (1 - p0(c)) (1 - conf) against the base rate p0(c), on the training
    graph.  Positives = support s (exact, from ``raw_const``), negatives = body size - s.
    """

    table = program.const.get(q)
    if table is None or table.nnz == 0:
        e = np.zeros(0)
        return e.astype(np.int64), e.astype(np.int64), e, e
    coo = table.tocoo()
    b, c, conf = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float64)
    s, body = _support(program, q, b, c)
    p0 = base_rates(program, q)[c]
    conf = np.clip(conf, 1e-9, 1 - 1e-9)
    p_new = 1.0 - (1.0 - p0) * (1.0 - conf)
    gain = s * np.log2(p_new / p0) + (body - s) * np.log2((1.0 - p_new) / (1.0 - p0))
    return b, c, conf, gain


def _support(program: Program, q: int, b: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact (support, body size) of stored rules (b, c), looked up in ``raw_const``."""

    rb, rc, rs, rbody = program.raw_const[q]
    keys = rb.astype(np.int64) * program.n + rc.astype(np.int64)
    order = np.argsort(keys, kind="stable")
    sk = keys[order]
    want = b * program.n + c
    pos = np.searchsorted(sk, want)
    if np.any(pos >= sk.size) or np.any(sk[np.minimum(pos, sk.size - 1)] != want):
        raise KeyError("stored constant rule missing from raw_const")
    idx = order[pos]
    return rs[idx].astype(np.float64), rbody[idx].astype(np.float64)


def nonredundant_keep(program: Program, q: int, b: np.ndarray, c: np.ndarray, conf: np.ndarray,
                      gain: np.ndarray, threshold: float) -> np.ndarray:
    """Greedy marginal-gain selection per (q, c) under a noisy-or with leak p0(c).

    Rules are visited by standalone gain, highest first; a rule is kept if its marginal
    saving on its body population, given the rules already kept for the same (q, c),
    is at least ``threshold`` bits.  Only rules whose standalone gain passes can pass
    (the marginal saving never exceeds the standalone one), so only those are visited.
    """

    keep = np.zeros(b.shape[0], dtype=bool)
    cand = np.where(gain >= threshold)[0]
    if cand.size == 0:
        return keep
    Xc = program.X.tocsc()
    Aq = program.A[q].tocsc()
    p0_all = base_rates(program, q)
    order = cand[np.lexsort((-gain[cand], c[cand]))]  # grouped by c, best first within a group
    bounds = np.flatnonzero(np.diff(c[order])) + 1
    for group in np.split(order, bounds):
        cc = int(c[group[0]])
        if group.size == 1:  # a lone rule's marginal saving is its standalone gain
            keep[group[0]] = True
            continue
        pops = [Xc.indices[Xc.indptr[b[i]]:Xc.indptr[b[i] + 1]] for i in group]
        union = np.unique(np.concatenate(pops))
        positives = Aq.indices[Aq.indptr[cc]:Aq.indptr[cc + 1]]
        is_pos_u = np.isin(union, positives)
        one_minus = np.full(union.size, 1.0 - float(p0_all[cc]))
        for i, pop in zip(group, pops):
            loc = np.searchsorted(union, pop)
            om = one_minus[loc]
            new_om = om * (1.0 - conf[i])
            pos = is_pos_u[loc]
            marginal = float(np.log2((1.0 - new_om[pos]) / (1.0 - om[pos])).sum()
                             + np.log2(new_om[~pos] / om[~pos]).sum())
            if marginal >= threshold:
                keep[i] = True
                one_minus[loc] = new_om
    return keep


@dataclass
class PruneReport:
    kept: int
    total: int
    per_relation: dict[int, tuple[int, int]]


def mdl_prune(program: Program, mode: str, kappa: float = 1.0) -> tuple[dict[int, sp.csr_matrix], PruneReport]:
    """Pruned constant tables. mode: 'ind' (standalone) or 'nr' (non-redundant greedy)."""

    threshold = kappa * const_rule_bits(program)
    tables: dict[int, sp.csr_matrix] = {}
    per: dict[int, tuple[int, int]] = {}
    kept_total = total = 0
    for q, table in program.const.items():
        b, c, conf, gain = const_gains(program, q)
        if mode == "ind":
            keep = gain >= threshold
        elif mode == "nr":
            keep = nonredundant_keep(program, q, b, c, conf, gain, threshold)
        else:
            raise ValueError(mode)
        tables[q] = sp.csr_matrix((conf[keep].astype(np.float32), (b[keep], c[keep])), shape=table.shape)
        per[q] = (int(keep.sum()), int(b.size))
        kept_total += int(keep.sum())
        total += int(b.size)
    return tables, PruneReport(kept_total, total, per)


def pruned_program(program: Program, tables: dict[int, sp.csr_matrix]) -> Program:
    """A shallow copy sharing graph and path rules, with its own constant tables."""

    out = copy.copy(program)
    out.const = dict(tables)
    for attr in ("_attr_freq", "_self_rate"):
        if hasattr(program, attr):
            setattr(out, attr, getattr(program, attr))
    return out


# ---------------------------------------------------------------------------
# Gate inputs
# ---------------------------------------------------------------------------


def relation_fanout(program: Program, q: int) -> float:
    """Mean number of training answers per head that has any, for query relation q."""

    m = program.A[q]
    heads = int((np.diff(m.indptr) > 0).sum())
    return float(m.nnz) / max(1, heads)


def query_features(program: Program, q: int, xs: np.ndarray, F: np.ndarray, rot: np.ndarray | None) -> np.ndarray:
    """(Q, len(GATE_FEATURES)) gate inputs for queries (x, q, ?).

    fired: candidates with any rule evidence; known: x's training answers of q;
    rel_fanout: q's mean answers per head; top_rule: best single rule confidence on any
    candidate; rotate_margin: RotatE top-1 minus top-10 score (0 without RotatE).
    """

    Q = xs.shape[0]
    fired = (F[12] > 0).sum(1).astype(np.float64)
    fired[:] -= (F[12][np.arange(Q), xs] > 0)
    known = np.diff(program.A[q].indptr)[xs].astype(np.float64)
    top_rule = np.maximum(np.maximum(F[0], F[3]), np.maximum(F[6], F[9])).astype(np.float64)
    top_rule[np.arange(Q), xs] = 0.0
    if rot is not None:
        r = rot.astype(np.float64).copy()
        r[np.arange(Q), xs] = -np.inf
        part = -np.partition(-r, 9, axis=1)[:, :10]
        part.sort(axis=1)
        margin = part[:, -1] - part[:, 0]
    else:
        margin = np.zeros(Q)
    return np.stack([np.ones(Q), np.log1p(fired), np.log1p(known),
                     np.full(Q, math.log1p(relation_fanout(program, q))), top_rule.max(1), margin], axis=1)


# ---------------------------------------------------------------------------
# Gated linear evidence
# ---------------------------------------------------------------------------


@dataclass
class Gated:
    """score = sum_k F_k * (w_k + delta[q, k] + sum_j U[k, j] phi_j), raw-feature scale."""

    w: np.ndarray  # (d,)
    delta: np.ndarray  # (2R, d)
    U: np.ndarray  # (d, g)
    phi_mean: np.ndarray
    phi_std: np.ndarray
    names: tuple[str, ...]
    mu: float
    mu_q: float

    def weights(self, q: int, phi: np.ndarray) -> np.ndarray:
        z = (phi - self.phi_mean) / self.phi_std
        z[:, 0] = 1.0
        return self.w[None, :] + self.delta[q][None, :] + z @ self.U.T  # (Q, d)

    def score(self, q: int, F: np.ndarray, extra: np.ndarray | None, phi: np.ndarray) -> np.ndarray:
        feats = F if extra is None else np.concatenate([F, extra[None]], axis=0)
        W = self.weights(q, phi)  # (Q, d)
        return np.einsum("dqn,qd->qn", feats.astype(np.float64), W)


def fit_gated(sets: np.ndarray, counts: np.ndarray, qidx: np.ndarray, phi: np.ndarray, n_q: int,
              mu: float, mu_q: float, mu_u: float, names: tuple[str, ...], gate: bool = True,
              per_relation: bool = True, maxiter: int = 500) -> Gated:
    """Softmax cross-entropy over candidate sets (answer in column 0).

    Parameters on standardized features: shared w (L2 mu), per-relation delta (L2 mu_q),
    gate matrix U (d x g, L2 mu_u; column 0 of phi is the bias and carries no gate).
    Folded back to raw-feature scale.  float32 storage, float64 accumulation.
    """

    from scipy.optimize import minimize

    Q, C, d = sets.shape
    mask = np.arange(C)[None, :] < counts[:, None]
    flat = sets[mask]
    mean = flat.mean(0).astype(np.float64)
    std = flat.std(0).astype(np.float64)
    std[std < 1e-8] = 1.0
    del flat
    Z = ((sets - mean.astype(np.float32)) / std.astype(np.float32)).astype(np.float32)
    Z[~mask] = 0.0
    pad = np.where(mask, 0.0, -np.inf)
    del mask
    g = phi.shape[1]
    phi_mean = phi.mean(0)
    phi_std = phi.std(0)
    phi_std[phi_std < 1e-8] = 1.0
    P = (phi - phi_mean) / phi_std
    P[:, 0] = 0.0  # bias handled by w / delta
    P = P[:, 1:]
    g1 = P.shape[1]
    n_w, n_d, n_u = d, (n_q * d if per_relation else 0), (d * g1 if gate else 0)

    def unpack(theta: np.ndarray):
        w = theta[:n_w]
        delta = theta[n_w:n_w + n_d].reshape(n_q, d) if per_relation else np.zeros((n_q, d))
        U = theta[n_w + n_d:].reshape(d, g1) if gate else np.zeros((d, g1))
        return w, delta, U

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        w, delta, U = unpack(theta)
        W = w[None, :] + delta[qidx] + P @ U.T  # (Q, d) in standardized space
        logits = np.matmul(Z, W.astype(np.float32)[:, :, None])[:, :, 0].astype(np.float64) + pad
        m = logits.max(1, keepdims=True)
        p = np.exp(logits - m)
        z = p.sum(1, keepdims=True)
        p /= z
        loss = float((np.log(z[:, 0]) + m[:, 0] - logits[:, 0]).mean())
        p[:, 0] -= 1.0
        G = np.matmul(p.astype(np.float32)[:, None, :], Z)[:, 0, :].astype(np.float64) / Q  # dL/dW per query
        gw = G.sum(0) + 2 * mu * w
        loss += mu * float(w @ w)
        grads = [gw]
        if per_relation:
            gd = np.zeros((n_q, d))
            np.add.at(gd, qidx, G)
            gd += 2 * mu_q * delta
            loss += mu_q * float((delta * delta).sum())
            grads.append(gd.ravel())
        if gate:
            gu = G.T @ P + 2 * mu_u * U
            loss += mu_u * float((U * U).sum())
            grads.append(gu.ravel())
        return loss, np.concatenate(grads)

    theta0 = np.zeros(n_w + n_d + n_u)
    theta0[0] = 1.0
    res = minimize(objective, theta0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    w, delta, U = unpack(res.x)
    # fold standardization of candidate features into the weights (raw-feature scale);
    # the constant shift from means is identical across a query's candidates and drops out
    U_full = np.zeros((d, g))
    U_full[:, 1:] = U
    return Gated(w=w / std, delta=delta / std[None, :], U=U_full / std[:, None], phi_mean=phi_mean,
                 phi_std=phi_std, names=names, mu=mu, mu_q=mu_q)


__all__ = ["GATE_FEATURES", "const_rule_bits", "base_rates", "const_gains", "nonredundant_keep", "mdl_prune",
           "PruneReport", "pruned_program", "relation_fanout", "query_features", "Gated", "fit_gated"]
