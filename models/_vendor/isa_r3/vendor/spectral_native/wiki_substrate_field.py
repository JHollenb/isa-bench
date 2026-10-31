"""Single-substrate Wikipedia field (WS1): readings stay provisional until a consumer decides.

One address space holds pair observations (pattern coordinates), source Wikidata facts,
entity-type coordinates and learned two-step relation rules. A reading of an occurrence is
a graded distribution over directional labels combined from several projected views:

- ``pattern``: the exact between-mention pattern;
- ``words``: the pattern's tokens (naive-Bayes style product);
- ``cell``: the (left type, right type) cell, types from source facts (L2b "top" role);
- ``pair``: source facts already linking the two entities (memory, L2b pair lookup);
- ``rules``: two-step paths in source facts that learned rules say imply the label.

Views combine geometrically with non-negative weights fitted on feedback. The staged arm
commits each occurrence to its argmax before a consumer runs; substrate arms hand the
consumer the graded readings, so later evidence (the second hop) can select among them.
Counting and joins only; no new SPF arithmetic.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

Occ = list  # [article, sid, left, right, pattern, labels]
VIEWS = ("pattern", "words", "cell", "cell31", "pooled", "pair", "rules", "learned")
RULE_FANOUT = 300  # first-hop neighbours scanned per entity when proposing rule-derived labels


def flip(label: str) -> str:
    p, d = label.split(":")
    return f"{p}:{'r' if d == 'f' else 'f'}"


def fact_of(left: str, right: str, label: str) -> tuple[str, str, str]:
    p, d = label.split(":")
    return (left, p, right) if d == "f" else (right, p, left)


@dataclass
class Substrate:
    labels: list[str]
    prior: Counter
    pattern: dict[str, Counter]
    words: dict[str, Counter]
    word_totals: Counter
    cell: dict[tuple[str, str], Counter]
    types: dict[str, str]
    out: dict[str, dict[str, set[str]]]          # subject -> predicate -> objects (source facts)
    inn: dict[str, dict[str, set[str]]]          # object -> predicate -> subjects
    rules: dict[str, list[tuple[tuple[str, str, str, str], float]]] = field(default_factory=dict)
    types31: dict[str, str] = field(default_factory=dict)
    cell31: dict[tuple[str, str], Counter] = field(default_factory=dict)
    by_body: dict[tuple[str, str, str, str], list[tuple[str, float]]] = field(default_factory=dict)

    # -- construction -----------------------------------------------------------------------------
    @classmethod
    def build(cls, occurrences: Iterable[Occ], facts: Iterable[tuple[str, str, str]], *, min_label: int = 20,
              max_degree: int = 200, min_rule_support: int = 5, min_rule_conf: float = 0.2) -> Substrate:
        occ = [o for o in occurrences if o[5]]
        prior = Counter(l for o in occ for l in o[5])
        labels = sorted(l for l, n in prior.items() if n >= min_label)
        keep = set(labels)
        pattern: dict[str, Counter] = defaultdict(Counter)
        words: dict[str, Counter] = defaultdict(Counter)
        word_totals: Counter = Counter()
        out: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        inn: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for s, p, o in facts:
            out[s][p].add(o)
            inn[o][p].add(s)
        roles: dict[str, Counter] = defaultdict(Counter)
        for s, preds in out.items():
            for p, objs in preds.items():
                roles[s][f"{p}>"] += len(objs)
                for o in objs:
                    roles[o][f">{p}"] += 1
        types = {e: c.most_common(1)[0][0] for e, c in roles.items()}
        classes = Counter(o for s in out for o in out[s].get("P31", ()))
        types31 = {e: (max(sorted(out[e]["P31"]), key=lambda c: classes[c]) if out.get(e, {}).get("P31")
                       else f"role:{t}") for e, t in types.items()}
        cell: dict[tuple[str, str], Counter] = defaultdict(Counter)
        cell31: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for o in occ:
            ls = [l for l in o[5] if l in keep]
            for l in ls:
                pattern[o[4]][l] += 1
                for w in set(o[4].split()):
                    words[w][l] += 1
                    word_totals[l] += 1
                cell[(types.get(o[2], "?"), types.get(o[3], "?"))][l] += 1
                cell31[(types31.get(o[2], "?"), types31.get(o[3], "?"))][l] += 1
        sub = cls(labels, Counter({l: prior[l] for l in labels}), dict(pattern), dict(words), word_totals,
                  dict(cell), types, out, inn, types31=types31, cell31=dict(cell31))
        sub.rules = sub._learn_rules(max_degree, min_rule_support, min_rule_conf)
        by_body: dict[tuple, list] = defaultdict(list)
        for head, lst in sub.rules.items():
            for body, conf in lst:
                by_body[body].append((head, conf))
        sub.by_body = dict(by_body)
        return sub

    def _learn_rules(self, max_degree: int, min_support: int, min_conf: float
                     ) -> dict[str, list[tuple[tuple[str, str, str, str], float]]]:
        """Head (x P z) <- x -P1-> y -P2-> z with directions; confidence = head hits / body paths."""
        body: Counter = Counter()
        hits: Counter = Counter()
        for y in set(self.out) | set(self.inn):
            left = [(p, "r", x) for p, xs in self.inn.get(y, {}).items() for x in xs]    # x -p-> y
            left += [(p, "f", x) for p, xs in self.out.get(y, {}).items() for x in xs]   # y -p-> x, read backwards
            right = [(p, "f", z) for p, zs in self.out.get(y, {}).items() for z in zs]   # y -p-> z
            right += [(p, "r", z) for p, zs in self.inn.get(y, {}).items() for z in zs]  # z -p-> y
            if len(left) * len(right) > max_degree * max_degree:
                continue
            for p1, d1, x in left:
                for p2, d2, z in right:
                    if x == z:
                        continue
                    key = (p1, d1, p2, d2)
                    body[key] += 1
                    for p in self.out.get(x, {}):
                        if z in self.out[x][p]:
                            hits[(key, p)] += 1
        rules: dict[str, list] = defaultdict(list)
        for (key, p), n in hits.items():
            conf = n / body[key]
            if n >= min_support and conf >= min_conf:
                rules[p].append((key, conf))
        return {p: sorted(v, key=lambda t: -t[1])[:20] for p, v in rules.items()}

    # -- graded views -----------------------------------------------------------------------------
    def _logp(self, counts: Counter | None, label: str, alpha: float = 0.5) -> float:
        n = sum(counts.values()) if counts else 0
        k = counts.get(label, 0) if counts else 0
        base = self.prior[label] / max(1, sum(self.prior.values()))
        return math.log((k + alpha * len(self.labels) * base) / (n + alpha * len(self.labels)))

    def pair_labels(self, left: str, right: str) -> set[str]:
        out = set()
        for p, objs in self.out.get(left, {}).items():
            if right in objs:
                out.add(f"{p}:f")
        for p, objs in self.out.get(right, {}).items():
            if left in objs:
                out.add(f"{p}:r")
        return out

    def rule_heads(self, left: str, right: str) -> dict[str, float]:
        """Labels implied by learned rules along two-step paths in source facts (best confidence)."""
        found: dict[str, float] = {}
        for x, z, d in ((left, right, "f"), (right, left, "r")):
            firsts = []
            for p1, ys in sorted(self.out.get(x, {}).items()):
                firsts.extend((p1, "r", y) for y in ys)   # x -p1-> y
            for p1, ys in sorted(self.inn.get(x, {}).items()):
                firsts.extend((p1, "f", y) for y in ys)   # y -p1-> x
            for p1, d1, y in firsts[:RULE_FANOUT]:
                if y == z:
                    continue
                bodies = [(p1, d1, p2, "f") for p2, zs in self.out.get(y, {}).items() if z in zs]
                bodies += [(p1, d1, p2, "r") for p2, ss in self.inn.get(y, {}).items() if z in ss]
                for body in bodies:
                    for head, conf in self.by_body.get(body, []):
                        label = f"{head}:{d}"
                        if label in self.prior and conf > found.get(label, 0.0):
                            found[label] = conf
        return found

    def candidates(self, o: Occ, k: int = 25, heads: Mapping[str, float] | None = None,
                   learned: Mapping[str, float] | None = None) -> list[str]:
        pool = Counter()
        for l, prob in (learned or {}).items():
            if l in self.prior:
                pool[l] += int(10**4 * prob) + 1
        for c in (self.pattern.get(o[4]), self.cell.get((self.types.get(o[2], "?"), self.types.get(o[3], "?"))),
                  self.cell31.get((self.types31.get(o[2], "?"), self.types31.get(o[3], "?")))):
            if c:
                pool.update(dict(c.most_common(k)))
        for l in self.pair_labels(o[2], o[3]):
            if l in self.prior:
                pool[l] += 10**6
        for l, conf in (heads or {}).items():
            pool[l] += int(1000 * conf)
        if not pool:
            pool.update(dict(self.prior.most_common(k)))
        return [l for l, _ in pool.most_common(k)]

    def text_score(self, o: Occ, label: str) -> float:
        if label not in self.prior:  # a flipped direction the source never labels carries no evidence
            return 0.0
        pat = self.pattern.get(o[4])
        toks = set(o[4].split())
        base = self._logp(None, label)
        wsum = sum(self._logp(self.words.get(w), label) - base for w in toks if w in self.words)
        return (self._logp(pat, label) - base if pat else 0.0) + wsum / max(1, len(toks))

    def view_scores(self, o: Occ, labels: Sequence[str], siblings: Sequence[Occ] = (),
                    heads: Mapping[str, float] | None = None,
                    learned: Mapping[str, float] | None = None) -> dict[str, dict[str, float]]:
        cell = self.cell.get((self.types.get(o[2], "?"), self.types.get(o[3], "?")))
        cell31 = self.cell31.get((self.types31.get(o[2], "?"), self.types31.get(o[3], "?")))
        pat = self.pattern.get(o[4])
        toks = set(o[4].split())
        pair = self.pair_labels(o[2], o[3])
        heads = self.rule_heads(o[2], o[3]) if heads is None else heads
        out = {}
        for l in labels:
            base = self._logp(None, l)
            wsum = sum(self._logp(self.words.get(w), l) - base for w in toks if w in self.words)
            pooled = [self.text_score(s, l if s[2] == o[2] else flip(l)) for s in siblings]
            out[l] = {
                "prior": base,
                "pattern": self._logp(pat, l) - base if pat else 0.0,
                "words": wsum / max(1, len(toks)),
                "cell": self._logp(cell, l) - base if cell else 0.0,
                "cell31": self._logp(cell31, l) - base if cell31 else 0.0,
                "pooled": sum(pooled) / len(pooled) if pooled else 0.0,
                "pair": 1.0 if l in pair else 0.0,
                "rules": heads.get(l, 0.0),
                # a graded learned reader is one more view; its log-probability relative to the prior
                "learned": (math.log(max(learned.get(l, 0.0), 1e-6)) - base) if learned else 0.0,
            }
        return out


def reading(sub: Substrate, o: Occ, weights: Mapping[str, float], siblings: Sequence[Occ] = (),
            k: int = 25, learned: Mapping[str, float] | None = None) -> list[tuple[str, float]]:
    """Graded reading: normalized probabilities over candidate labels."""
    heads = sub.rule_heads(o[2], o[3]) if weights.get("rules") else {}
    learned = learned if weights.get("learned") else None
    labels = sub.candidates(o, k, heads, learned)
    if not labels:
        return []
    vs = sub.view_scores(o, labels, siblings if weights.get("pooled") else (), heads, learned)
    score = {l: vs[l]["prior"] + sum(weights.get(v, 0.0) * vs[l][v] for v in VIEWS) for l in labels}
    top = max(score.values())
    z = sum(math.exp(s - top) for s in score.values())
    return sorted(((l, math.exp(s - top) / z) for l, s in score.items()), key=lambda t: -t[1])


def fit_weights(sub: Substrate, rows: Sequence[Occ], views: Sequence[str],
                siblings_of=lambda o: (), grid: Sequence[float] = (0, 0.5, 1, 2, 4),
                rounds: int = 2, learned_of=None) -> dict[str, float]:
    """Coordinate ascent on feedback top-1 accuracy (ties -> smaller weight)."""
    cache = []
    for o in rows:
        if not o[5]:
            continue
        heads = sub.rule_heads(o[2], o[3]) if "rules" in views else {}
        learned = learned_of(o) if (learned_of is not None and "learned" in views) else None
        labels = sub.candidates(o, 25, heads, learned)
        if not labels:
            continue
        sibs = siblings_of(o) if "pooled" in views else ()
        cache.append((o, labels, sub.view_scores(o, labels, sibs, heads, learned)))

    def acc(w: Mapping[str, float]) -> int:
        ok = 0
        for o, labels, vs in cache:
            best = max(labels, key=lambda l: vs[l]["prior"] + sum(w.get(v, 0.0) * vs[l][v] for v in VIEWS))
            ok += best in o[5]
        return ok

    w = {v: 1.0 for v in views}
    for _ in range(rounds):
        for v in views:
            best = max(grid, key=lambda g: (acc({**w, v: g}), -g))
            w[v] = best
    return w


def two_hop(facts: Mapping[tuple[str, str], Mapping[str, float]], start: str, p1: str, p2: str
            ) -> list[tuple[str, float]]:
    """Score answers z by max over y of p(start p1 y) * p(y p2 z); facts[(s, p)] = {o: prob}."""
    scores: dict[str, float] = {}
    for y, a in facts.get((start, p1), {}).items():
        for z, b in facts.get((y, p2), {}).items():
            if z != start:
                scores[z] = max(scores.get(z, 0.0), a * b)
    return sorted(scores.items(), key=lambda t: -t[1])


__all__ = ["Substrate", "VIEWS", "fact_of", "fit_weights", "flip", "reading", "two_hop"]
