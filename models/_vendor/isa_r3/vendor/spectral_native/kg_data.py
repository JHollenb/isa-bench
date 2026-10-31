"""Knowledge-graph benchmark data and the standard filtered ranking protocol.

Triples are integer arrays ``(h, r, t)``.  Queries are unified in a 2R
relation view: a tail query (h, r, ?) is ``(h, r, t)`` and a head query
(?, r, t) is ``(t, r + R, h)``, so both directions are "given entity e and
relation q, rank the answer".  The filtered rank of the answer is the expected
rank under uniformly random tie-breaking,

    rank = 1 + #(filtered candidates scored higher) + #(filtered others tied) / 2,

which is the unbiased choice for scores that tie often (rule systems).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp


@dataclass
class KG:
    name: str
    entities: list[str]
    relations: list[str]
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray

    @property
    def n(self) -> int:
        return len(self.entities)

    @property
    def R(self) -> int:  # noqa: N802
        return len(self.relations)


def _dict(path: Path) -> list[str]:
    rows = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines() if line.strip()]
    return [name for _i, name in sorted(((int(i), name) for i, name in rows))]


def load_kg(directory: str | Path) -> KG:
    directory = Path(directory)
    entities = _dict(directory / "entities.dict")
    relations = _dict(directory / "relations.dict")
    e_id = {e: i for i, e in enumerate(entities)}
    r_id = {r: i for i, r in enumerate(relations)}

    def triples(name: str) -> np.ndarray:
        out = []
        for line in (directory / name).read_text().splitlines():
            if line.strip():
                h, r, t = line.split("\t")
                out.append((e_id[h], r_id[r], e_id[t]))
        return np.array(out, dtype=np.int64)

    return KG(directory.name, entities, relations, triples("train.txt"), triples("valid.txt"), triples("test.txt"))


def to_queries(triples: np.ndarray, R: int) -> np.ndarray:  # noqa: N803
    """(e, q, answer) rows: tail queries then head queries."""

    tail = triples[:, [0, 1, 2]]
    head = np.stack([triples[:, 2], triples[:, 1] + R, triples[:, 0]], axis=1)
    return np.concatenate([tail, head], axis=0)


def adjacency(triples: np.ndarray, n: int, R: int) -> list[sp.csr_matrix]:  # noqa: N803
    """Boolean adjacency per relation in the 2R view (index r + R is the inverse)."""

    mats = []
    for q in range(2 * R):
        r = q % R
        sel = triples[triples[:, 1] == r]
        rows, cols = (sel[:, 0], sel[:, 2]) if q < R else (sel[:, 2], sel[:, 0])
        m = sp.csr_matrix((np.ones(rows.shape[0], dtype=np.float32), (rows, cols)), shape=(n, n))
        m.sum_duplicates()
        m.data[:] = 1.0
        mats.append(m)
    return mats


def filter_sets(kg: KG) -> dict[tuple[int, int], np.ndarray]:
    """All known answers for (entity, 2R relation) over train, valid and test."""

    known: dict[tuple[int, int], set[int]] = {}
    for split in (kg.train, kg.valid, kg.test):
        for e, q, a in to_queries(split, kg.R):
            known.setdefault((int(e), int(q)), set()).add(int(a))
    return {key: np.fromiter(v, dtype=np.int64) for key, v in known.items()}


def filtered_rank(scores: np.ndarray, answer: int, known: np.ndarray) -> tuple[int, int]:
    """(#filtered candidates scored higher, #filtered others tied with the answer)."""

    target = scores[answer]
    mask = np.ones(scores.shape[0], dtype=bool)
    mask[known] = False
    mask[answer] = False
    others = scores[mask]
    return int((others > target).sum()), int((others == target).sum())


def metrics(higher_tied: np.ndarray) -> dict[str, float]:
    """Metrics from (higher, tied) pairs.

    Primary (preregistered): rank = 1 + higher + tied/2, then MRR = mean(1/rank)
    and Hits@k = mean(rank <= k).  Secondary: exact expectations under uniformly
    random tie-breaking, E[1/rank] and P(rank <= k).
    """

    from scipy.special import digamma

    ht = np.asarray(higher_tied, dtype=np.float64)
    higher, tied = ht[:, 0], ht[:, 1]
    rank = 1.0 + higher + tied / 2.0
    out = {
        "MRR": float((1.0 / rank).mean()),
        "H@1": float((rank <= 1.0).mean()),
        "H@3": float((rank <= 3.0).mean()),
        "H@10": float((rank <= 10.0).mean()),
        "MR": float(rank.mean()),
        "count": int(rank.shape[0]),
    }
    # E[1/rank] with rank uniform on higher+1 .. higher+tied+1
    expected_rr = (digamma(higher + tied + 2.0) - digamma(higher + 1.0)) / (tied + 1.0)
    out["MRR_random_ties"] = float(expected_rr.mean())
    for k in (1, 3, 10):
        out[f"H@{k}_random_ties"] = float(np.clip((k - higher) / (tied + 1.0), 0.0, 1.0).mean())
    return out
