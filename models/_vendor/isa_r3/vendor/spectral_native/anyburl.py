"""AnyBURL rule-learner baseline: adapters onto the owner's KG protocol.

This module lets the AnyBURL symbolic rule learner (Meilicke et al.) be scored
with *exactly* the owner's filtered ranking protocol in ``kg_data`` (expected
rank ``1 + higher + tied/2``, both the query-entity-excluded and the official
candidate sets), so the numbers are directly comparable with the spectral-field
KG results and the reproduced RotatE.

AnyBURL itself is a Java program; it is run as a subprocess by
``scripts/run_anyburl.py`` on Beast.  Everything here is pure Python/NumPy and
operates on AnyBURL's text artefacts:

* its rules file (``rules-100`` / ``rules-1000``) -> ``parse_rules`` + bit
  accounting (``rule_bits`` / ``store_bits``);
* its prediction file (``preds``) -> ``parse_predictions`` -> a per-query dense
  score vector over all N entities, reconstructed so the owner's own filter and
  tie handling apply (``score_queries``).

Entities and relations are remapped to the tokens ``e<id>`` / ``r<id>`` with the
owner's own integer ids (from ``entities.dict`` / ``relations.dict``) before
AnyBURL sees them, so AnyBURL's entity space is bit-identical to the scorer's.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kg_data import filtered_rank, metrics

ENT_RE = re.compile(r"^e(\d+)$")
REL_RE = re.compile(r"^r(\d+)$")


# ---------------------------------------------------------------------------
# Remapping the owner's integer triples to AnyBURL-safe tokens and back
# ---------------------------------------------------------------------------


def write_anyburl_triples(triples: np.ndarray, path: str | Path) -> None:
    """Write (h, r, t) integer rows as ``e<h>\\tr<r>\\te<t>`` lines.

    The tokens are the owner's own ids, so no separate id map is needed on the
    way back: ``e123`` is entity 123 and ``r4`` is relation 4.
    """

    path = Path(path)
    lines = [f"e{int(h)}\tr{int(r)}\te{int(t)}" for h, r, t in triples]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _ent(token: str) -> int | None:
    m = ENT_RE.match(token)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Predictions  (preds file)  ->  per-query dense score vectors over N entities
# ---------------------------------------------------------------------------


def parse_predictions(path: str | Path, n: int, R: int) -> dict[tuple[int, int], dict[int, float]]:  # noqa: N803
    """AnyBURL ``preds`` -> ``{(entity, 2R-relation): {candidate: score}}`` (sparse).

    AnyBURL writes three lines per evaluated test triple::

        e<h>  r<r>  e<t>
        Heads: e<c1>  <s1>  e<c2>  <s2>  ...
        Tails: e<c1>  <s1>  e<c2>  <s2>  ...

    ``Tails`` ranks candidate tails of (h, r, ?)  -> owner query ``(h, r)``.
    ``Heads`` ranks candidate heads of (?, r, t)  -> owner query ``(t, r + R)``
    (the 2R view: a head query is ``(t, r + R, h)``).

    AnyBURL filters *other* known answers out of each list but keeps the triple's
    own target, so different test targets of the same query give different lists.
    We therefore merge every block of a query by ``max``: non-answer candidates
    have the same score in every block (a no-op under max), and each block
    contributes its own target's score.  The owner's ``filtered_rank`` then does
    all filtering and tie handling itself.  Storage is sparse (only scored
    candidates); ``dense_vector`` reconstructs one query's vector over N at a
    time, so memory stays O(N) rather than O(queries x N).
    """

    _ = n  # n is used only by dense_vector; kept for signature symmetry
    preds: dict[tuple[int, int], dict[int, float]] = {}

    def apply_line(parts: list[str], key: tuple[int, int]) -> None:
        d = preds.setdefault(key, {})
        i = 0
        while i + 1 < len(parts):  # alternating  entity  score ...
            e = _ent(parts[i])
            if e is None:
                i += 1
                continue
            try:
                s = float(parts[i + 1])
            except ValueError:
                i += 1
                continue
            if s > d.get(e, -np.inf):
                d[e] = s
            i += 2

    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        head = lines[i].split()
        i += 1
        if len(head) < 3:
            continue
        h, rm, t = _ent(head[0]), REL_RE.match(head[1]), _ent(head[2])
        if h is None or rm is None or t is None:
            continue
        r = int(rm.group(1))
        while i < len(lines) and (lines[i].startswith("Heads:") or lines[i].startswith("Tails:")):
            parts = lines[i].split()
            if parts[0] == "Tails:":
                apply_line(parts[1:], (h, r))          # (h, r, ?) tail query
            else:
                apply_line(parts[1:], (t, r + R))      # (?, r, t) head query
            i += 1
    return preds


def dense_vector(preds: dict, key: tuple[int, int], n: int) -> np.ndarray:
    """Reconstruct the dense score vector over N entities for one query (0 default)."""

    v = np.zeros(n, dtype=np.float64)
    d = preds.get(key)
    if d:
        idx = np.fromiter(d.keys(), dtype=np.int64, count=len(d))
        val = np.fromiter(d.values(), dtype=np.float64, count=len(d))
        v[idx] = val
    return v


def score_queries(preds: dict, queries: np.ndarray, known: dict, n: int, protocol: str) -> np.ndarray:
    """(higher, tied) per (e, q, a) query under the owner's filtered protocol.

    ``protocol='excluded'`` sets the query entity's own score to -inf (the
    owner's track-B / preregistered candidate set); ``protocol='official'``
    leaves it (RotatE's candidate set: the query entity stays a candidate unless
    it is a known answer).  Reuses ``kg_data.filtered_rank`` verbatim.
    """

    out = np.zeros((queries.shape[0], 2), dtype=np.int64)
    for i, (e, q, a) in enumerate(queries):
        e, q, a = int(e), int(q), int(a)
        scores = dense_vector(preds, (e, q), n)
        if protocol == "excluded":
            scores[e] = -np.inf
        out[i] = filtered_rank(scores, a, known[(e, q)])
    return out


def rr(higher_tied: np.ndarray) -> np.ndarray:
    ht = np.asarray(higher_tied, dtype=np.float64)
    return 1.0 / (1.0 + ht[:, 0] + ht[:, 1] / 2.0)


def eval_predictions(preds: dict, queries: np.ndarray, known: dict, n: int,
                     protocol: str) -> dict:
    ht = score_queries(preds, queries, known, n, protocol)
    out = metrics(ht)
    return out, ht


# ---------------------------------------------------------------------------
# Rules  (rules-<t> file)  ->  rule objects and bit accounting
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    predicted: int
    correct: int
    confidence: float
    head_rel: str
    head_const: bool          # head has a constant argument (AC2-style)
    body_rels: tuple[str, ...]
    body_consts: int          # number of constant (entity) arguments in the body
    raw: str

    @property
    def length(self) -> int:
        return len(self.body_rels)

    @property
    def kind(self) -> str:
        if self.body_consts == 0 and not self.head_const:
            return "cyclic"           # path rule, all variables
        return "acyclic"              # rule with at least one constant


_ATOM = re.compile(r"\s*([A-Za-z_][\w]*)\(([^,]+),\s*([^)]+)\)\s*")


def _atom_tokens(atom: str) -> tuple[str, list[str]] | None:
    m = _ATOM.match(atom.strip())
    if not m:
        return None
    return m.group(1), [m.group(2).strip(), m.group(3).strip()]


def parse_rules(path: str | Path) -> list[Rule]:
    """Parse an AnyBURL rules file (``predicted\\tcorrect\\tconf\\trule``).

    Rule syntax: ``rhead(X,Y) <= b1(X,A), b2(A,Y)``.  A constant argument is an
    ``e<id>`` token (entities were remapped to ``e<id>``); anything else is a
    logical variable.  Relations are ``r<id>`` tokens.
    """

    rules: list[Rule] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 4:
            # some AnyBURL builds use spaces between the three counts and the rule
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            fields = parts
        try:
            predicted, correct, confidence = int(fields[0]), int(fields[1]), float(fields[2])
        except ValueError:
            continue
        rule_str = fields[3].strip()
        if "<=" not in rule_str:
            continue
        head_str, body_str = rule_str.split("<=", 1)
        head = _atom_tokens(head_str)
        if head is None:
            continue
        head_rel, head_args = head
        head_const = any(_ent(a) is not None for a in head_args)
        body_rels: list[str] = []
        body_consts = 0
        for atom in body_str.split("),"):
            atom = atom if atom.strip().endswith(")") else atom + ")"
            tok = _atom_tokens(atom)
            if tok is None:
                continue
            rel, args = tok
            body_rels.append(rel)
            body_consts += sum(1 for a in args if _ent(a) is not None)
        rules.append(Rule(predicted, correct, confidence, head_rel, head_const,
                           tuple(body_rels), body_consts, rule_str))
    return rules


def store_bits(n_train: int, n: int, R: int) -> float:  # noqa: N803
    """The owner's exact store ruler (kg_field.program_bits): train triples as
    (entity, relation, entity) ids."""

    return n_train * (2 * math.log2(n) + math.log2(R))


def rule_bits(rules: list[Rule], n: int, R: int) -> dict:  # noqa: N803
    """A field-comparable encoding of a rule set.

    Modelled on the owner's ``program_bits`` ruler (each relation slot costs
    ``log2(2R)``, each constant entity ``log2(n)``, plus a 16-bit confidence),
    applied atom-by-atom to AnyBURL's rules.  It is **not** AnyBURL's own rule
    language nor the field's fixed templates, so it is reported as a comparable
    encoding, alongside the raw rule count and the gzip-compressed file size.
    """

    lg_q, lg_n = math.log2(2 * R), math.log2(n)
    total = 0.0
    by_kind = {"cyclic": [0, 0.0], "acyclic": [0, 0.0]}
    for rule in rules:
        slots = 1 + rule.length                       # head + body relation slots
        consts = rule.body_consts + (1 if rule.head_const else 0)
        bits = slots * lg_q + consts * lg_n + 16.0
        total += bits
        by_kind[rule.kind][0] += 1
        by_kind[rule.kind][1] += bits
    return {
        "encoding": "field-comparable: relation slot=log2(2R), constant=log2(n), confidence=16 bits",
        "n_rules": len(rules),
        "bits": total,
        "cyclic_rules": by_kind["cyclic"][0], "cyclic_bits": by_kind["cyclic"][1],
        "acyclic_rules": by_kind["acyclic"][0], "acyclic_bits": by_kind["acyclic"][1],
    }


__all__ = ["write_anyburl_triples", "parse_predictions", "dense_vector", "score_queries", "rr",
           "eval_predictions", "Rule", "parse_rules", "store_bits", "rule_bits"]
