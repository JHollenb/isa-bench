"""CLUTRR field model v2 (L1b): a learned relation algebra, joint decoding, soft composition, path voting
and rare-word placement on top of the L1 reader (``clutrr_field``). Preregistered in
docs/clutrr-field-v2-findings.md.

Components (each switchable, so every arm isolates one change):

  algebra  structural-form selection by MDL. Relation classes map to words over a few atom classes and
           their inverses. Candidate forms: L1's composition table; an abelian group (Z^a x Z2^b); a
           monoid presented by length-2 rewrite rules (x y -> z | 1), learned greedily by MDL. The bicyclic
           monoid <up, down | down.up = 1> is one instance. The form CLASS is a supplied structural prior;
           which form, atoms, rules and class words are chosen from training data only.
  joint    per-story joint decoding: every anchor (relation word occurrence) states one fact, chosen
           jointly across anchors by beam search over its top candidates, with gender-consistency,
           connectivity and query-connectivity terms. Redundant restatements are allowed (they close
           cycles instead of donating a spurious edge). Possessive pronouns also propose the clause subject
           as owner (cue "pronS"); its reliability is learned like every other frame.
  soft     the answer is the argmax of its marginal over the top joint readings (sum over readings
           weighted by exp(score / tau)), not commit-then-compose.
  vote     within a reading, every simple path between the query pair votes, weighted by the product of
           its edge probabilities (a repetition code), instead of the single BFS path.
  rare     unknown or rare relation words are placed by Bayesian inheritance over two axes: morphology
           (edit distance, shared prefix) and distribution (context vectors from training); orphan classes
           merge into a core class when the prior and their composition rows agree (SPRT).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from .clutrr_field import (FWD, POSSESSIVE, PRONOUN_GENDER, REV, SUBJECT_PRONOUNS, Candidate, FieldProgram,
                           Prediction, Reader, Reading, Story, _llr, story_path)

# ---------------------------------------------------------------------------------------------
# relation algebra
# ---------------------------------------------------------------------------------------------

LETTERS = "ABCDEF"


def _log2(x: float) -> float:
    return math.log2(max(2.0, x))


class Algebra:
    """Words over atom letters; composition = concatenation reduced to a normal form."""

    def __init__(self, form: str, atoms: list[tuple], inv: dict[tuple, tuple], rules: dict | None = None) -> None:
        self.form = form                       # "abelian" | "monoid"
        self.atoms = list(atoms)
        self.inv_class = inv
        self.rules: dict[tuple[str, str], tuple[str, ...]] = dict(rules or {})
        self.letter: dict[tuple, str] = {}
        self.inv_letter: dict[str, str] = {}
        for i, a in enumerate(self.atoms):
            up = LETTERS[i]
            self.letter[a] = up
            b = inv.get(a, a)
            if b == a:
                self.inv_letter[up] = up
            else:
                lo = up.lower()
                self.letter[b] = lo
                self.inv_letter[up], self.inv_letter[lo] = lo, up
        self.letters = sorted(self.inv_letter)
        self.words: dict[tuple, tuple[str, ...]] = {}
        self.word2class: dict[tuple[str, ...], tuple] = {}

    # -- normal forms ---------------------------------------------------------------------------
    def inv_word(self, w: tuple[str, ...]) -> tuple[str, ...]:
        return self.nf(tuple(self.inv_letter[x] for x in reversed(w)))

    def nf(self, w: tuple[str, ...]) -> tuple[str, ...]:
        if self.form == "abelian":
            exp: Counter = Counter()
            for x in w:
                lo = x.lower()
                if self.inv_letter[x] == x:          # involution: Z2
                    exp[x] = (exp[x] + 1) % 2
                elif x.isupper():
                    exp[x] += 1
                else:
                    exp[x.upper()] -= 1
            out: list[str] = []
            for x in sorted(exp):
                e = exp[x]
                if self.inv_letter.get(x) == x:
                    out.extend([x] * e)
                else:
                    out.extend([x] * e if e > 0 else [x.lower()] * (-e))
            return tuple(out)
        seq = list(w)
        changed = True
        while changed:
            changed = False
            for i in range(len(seq) - 1):
                r = self.rules.get((seq[i], seq[i + 1]))
                if r is not None:
                    seq[i:i + 2] = list(r)
                    changed = True
                    break
        return tuple(seq)

    # -- words by propagation through observed compositions --------------------------------------
    def derive(self, table: dict[tuple, Counter], iters: int = 8) -> None:
        fixed = set(self.letter)
        words = {c: (l,) for c, l in self.letter.items()}
        for _ in range(iters):
            votes: dict[tuple, Counter] = defaultdict(Counter)
            for (a, b), row in table.items():
                if a in words and b in words:
                    o, n = row.most_common(1)[0]
                    votes[o][self.nf(words[a] + words[b])] += n
            changed = False
            for o, v in sorted(votes.items(), key=lambda kv: -sum(kv[1].values())):
                if o in fixed:
                    continue
                w = v.most_common(1)[0][0]
                if words.get(o) != w:
                    words[o] = w
                    changed = True
                io = self.inv_class.get(o)
                if io is not None and io not in fixed and io not in votes:
                    iw = self.inv_word(w)
                    if words.get(io) != iw:
                        words[io] = iw
                        changed = True
            if not changed:
                break
        # abduction for classes seen only as composition inputs (rare synonyms such as "mom"): the short
        # word that makes their observed compositions come out right
        inputs: dict[tuple, list[tuple[int, tuple, tuple, float]]] = defaultdict(list)
        for (a, b), row in table.items():
            o, n = row.most_common(1)[0]
            if o not in words:
                continue
            if a not in words and b in words:
                inputs[a].append((0, b, o, n))
            elif b not in words and a in words:
                inputs[b].append((1, a, o, n))
        cand_words: list[tuple[str, ...]] = [()]
        for _ in range(3):
            cand_words = cand_words + [w + (x,) for w in cand_words for x in self.letters if len(w) < 3]
        cand_words = sorted(set(cand_words), key=lambda w: (len(w), w))
        for c, obs in sorted(inputs.items(), key=lambda kv: str(kv[0])):
            total = sum(n for *_, n in obs)
            best = (0.0, None)
            for w in cand_words:
                agree = 0.0
                for side, other, o, n in obs:
                    comp = w + words[other] if side == 0 else words[other] + w
                    if self.nf(comp) == words[o]:
                        agree += n
                if agree > best[0]:
                    best = (agree, w)
            if best[1] is not None and best[0] >= 0.6 * total:
                words[c] = self.nf(best[1])
                io = self.inv_class.get(c)
                if io is not None and io not in words:
                    words[io] = self.inv_word(words[c])
        self.words = words
        mass: Counter = Counter()
        for (a, b), row in table.items():
            mass[a] += sum(row.values())
            mass[b] += sum(row.values())
            for o, n in row.items():
                mass[o] += n
        w2c: dict[tuple, tuple] = {}
        for c, w in sorted(words.items(), key=lambda kv: (-mass[kv[0]], str(kv[0]))):
            w2c.setdefault(w, c)
        self.word2class = w2c

    def compose_classes(self, classes: list[tuple]) -> tuple | None:
        w: tuple[str, ...] = ()
        for c in classes:
            if c not in self.words:
                return None
            w = self.nf(w + self.words[c])
        return self.word2class.get(w)

    def is_identity_between(self, classes: list[tuple]) -> bool:
        """A walk between two distinct people whose relation reduces to the empty word (self)."""
        w: tuple[str, ...] = ()
        for c in classes:
            if c not in self.words:
                return False
            w = self.nf(w + self.words[c])
        return len(classes) > 0 and w == ()

    # -- description length ----------------------------------------------------------------------
    def bits(self, table: dict[tuple, Counter], n_classes: int) -> dict[str, float]:
        L = len(self.letters)
        lg = _log2(n_classes)
        entry_bits = 3.0 * lg
        data = 0.0
        exceptions = 0
        matched = 0
        for (a, b), row in table.items():
            total = sum(row.values())
            o, cnt = row.most_common(1)[0]
            pred = None
            if a in self.words and b in self.words:
                pred = self.word2class.get(self.nf(self.words[a] + self.words[b]))
            if pred == o:
                matched += 1
                data += (total - cnt) * lg
            else:
                exceptions += 1
                data += entry_bits + (total - cnt) * lg
        model = len(self.atoms) * lg + math.log2(3) * len(self.atoms)
        model += len(self.rules) * (2 * _log2(L) + _log2(L + 1))
        model += sum((len(w) + 1) * _log2(L + 1) for c, w in self.words.items() if c not in self.letter)
        return {"model": model, "data": data, "total": model + data, "exceptions": exceptions,
                "matched": matched, "entries": len(table)}

    def export(self, name: dict[tuple, str]) -> dict[str, Any]:
        return {"form": self.form, "atoms": [name.get(a, str(a)) for a in self.atoms],
                "rules": {f"{x}{y}": "".join(r) or "1" for (x, y), r in sorted(self.rules.items())},
                "words": {name.get(c, str(c)): "".join(w) or "1" for c, w in sorted(self.words.items(),
                                                                                  key=lambda kv: str(kv[0]))}}


def table_bits(table: dict[tuple, Counter], n_classes: int) -> dict[str, float]:
    lg = _log2(n_classes)
    data = sum(3.0 * lg + (sum(r.values()) - r.most_common(1)[0][1]) * lg for r in table.values())
    return {"model": 0.0, "data": data, "total": data, "exceptions": len(table), "matched": 0,
            "entries": len(table)}


def learn_algebra(program: FieldProgram, top_atoms: int = 5, max_atoms: int = 3,
                  max_rules: int = 16, rule_starts: int = 0) -> tuple[Algebra | None, dict[str, Any]]:
    """MDL choice among {table, abelian, monoid} on the learned composition table (training data only)."""
    comp = program.composer
    cls = comp.classes
    table = comp.table
    members: dict[tuple, list[tuple]] = defaultdict(list)
    for node in list(cls.parent):
        members[cls.find(node)].append(node)
    inv: dict[tuple, tuple] = {}
    for c, ms in members.items():
        inv[c] = cls.find((ms[0][0], 1 - ms[0][1]))
    freq: Counter = Counter()
    for (a, b), row in table.items():
        n = sum(row.values())
        freq[a] += n
        freq[b] += n
    n_classes = max(2, len(cls.classes()))
    # atom candidates: frequent classes, one per inverse pair
    cands: list[tuple] = []
    for c, _ in freq.most_common():
        if c in cands or inv.get(c) in cands:
            continue
        cands.append(c)
        if len(cands) >= top_atoms:
            break
    report: dict[str, Any] = {"table": table_bits(table, n_classes), "candidates": {}}
    best: tuple[float, Algebra | None, str] = (report["table"]["total"], None, "table")
    for form in ("abelian", "monoid"):
        best_form: tuple[float, Algebra | None] = (math.inf, None)
        scored: list[tuple[float, Algebra]] = []
        for k in range(1, max_atoms + 1):
            for atoms in combinations(cands, k):
                alg = Algebra(form, list(atoms), inv)
                alg.derive(table)
                scored.append((alg.bits(table, n_classes)["total"], alg))
        scored.sort(key=lambda t: t[0])
        # rule search from every atom set: before any rule exists, the right atoms can look worst
        # (words grow without cancellation), so initial bits cannot rank the starts
        for cur, alg in scored[: (rule_starts or len(scored)) if form == "monoid" else len(scored)]:
            if form == "monoid":
                alg, cur = _greedy_rules(alg, table, n_classes, cur, max_rules)
            if cur < best_form[0]:
                best_form = (cur, alg)
        alg = best_form[1]
        if alg is not None:
            report["candidates"][form] = alg.bits(table, n_classes)
            if best_form[0] < best[0]:
                best = (best_form[0], alg, form)
    report["chosen"] = best[2]
    return best[1], report


def _greedy_rules(alg: Algebra, table: dict[tuple, Counter], n_classes: int, cur: float,
                  max_rules: int) -> tuple[Algebra, float]:
    letters = alg.letters
    rhs_options: list[tuple[str, ...]] = [()] + [(z,) for z in letters]
    for _ in range(max_rules):
        best = (cur, None)
        for x in letters:
            for y in letters:
                if (x, y) in alg.rules:
                    continue
                for rhs in rhs_options:
                    trial = Algebra(alg.form, alg.atoms, alg.inv_class, {**alg.rules, (x, y): rhs})
                    trial.derive(table)
                    t = trial.bits(table, n_classes)["total"]
                    if t < best[0] - 1e-9:
                        best = (t, trial)
        if best[1] is None:
            break
        cur, alg = best[0], best[1]
    return alg, cur


# ---------------------------------------------------------------------------------------------
# rare words: morphology and distribution axes
# ---------------------------------------------------------------------------------------------


def edit_distance(a: str, b: str, cap: int = 3) -> int:
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def morph_logodds(word: str, known: str) -> float:
    """Evidence that ``word`` is a variant of ``known`` from spelling alone (generic, not kinship)."""
    if word == known:
        return 10.0
    ed = edit_distance(word, known)
    if ed == 1 and min(len(word), len(known)) >= 5:
        return 4.0
    if ed == 2 and min(len(word), len(known)) >= 8:
        return 2.5
    pre = 0
    for x, y in zip(word, known):
        if x != y:
            break
        pre += 1
    if pre >= 5:
        return 1.0 + 0.25 * (pre - 5)
    return -2.0


def context_vectors(stories: list[Story], window: int = 2) -> dict[str, Counter]:
    vec: dict[str, Counter] = defaultdict(Counter)
    for s in stories:
        toks = ["<N>" if k == "name" else v for k, v in s.tokens]
        for i, t in enumerate(toks):
            if t == "<N>":
                continue
            for d in range(-window, window + 1):
                if d and 0 <= i + d < len(toks):
                    vec[t][(d, toks[i + d])] += 1
    return vec


def cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    dot = sum(v * b.get(k, 0) for k, v in a.items())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---------------------------------------------------------------------------------------------
# joint reader
# ---------------------------------------------------------------------------------------------


@dataclass
class JointReading(Reading):
    # (joint score, spanning forest over the chosen facts, all chosen facts incl. restatements)
    alternatives: list[tuple[float, list[Candidate], list[Candidate]]] = field(default_factory=list)


class JointReader(Reader):
    """Every anchor states one fact; the facts of a story are chosen jointly by beam search."""

    OWNER_CUE = {**Reader.OWNER_CUE, "pronS": 1.0}

    def __init__(self, lexicon, genders, top_k: int = 3, beam: int = 32, keep: int = 8,
                 skip_penalty: float = 2.0, gender_penalty: float = 2.0, component_penalty: float = 1.5,
                 query_bonus: float = 3.0) -> None:
        super().__init__(lexicon, genders)
        self.top_k, self.beam, self.keep = top_k, beam, keep
        self.skip_penalty, self.gender_penalty = skip_penalty, gender_penalty
        self.component_penalty, self.query_bonus = component_penalty, query_bonus
        self.query: tuple[str, str] | None = None
        self.subject_owner = True

    def prior(self, frame: tuple[str, ...]) -> float:
        return self.OWNER_CUE.get(frame[0], 0.0) + self.RELATIVE_CUE.get(frame[1], 0.0)

    def candidates(self, story: Story, resolved: dict[int, str]) -> list[Candidate]:
        out = super().candidates(story, resolved)
        if not self.subject_owner:
            return out
        # clause-subject binding: a possessive pronoun may refer to its clause's subject (the first
        # mention of the sentence), when that differs from the recency referent
        toks = story.tokens
        first_mention: dict[int, str] = {}
        for i, (kind, value) in enumerate(toks):
            sent = story.sent_of[i]
            if sent in first_mention:
                continue
            if kind == "name":
                first_mention[sent] = value
            elif kind == "word" and value in SUBJECT_PRONOUNS and i in resolved:
                first_mention[sent] = resolved[i]
        # a second-ranked referent that is also the clause subject gets its own frame ("pronS"), so its
        # reliability is learned separately from plain second-ranked referents
        subj_of = {i: first_mention.get(story.sent_of[i]) for i in range(len(toks))}
        relabelled: list[Candidate] = []
        for c in out:
            if c.frame[0] == "pron2":
                subj = subj_of.get(c.anchor_pos)
                owner = c.relative if c.frame[-1] == "flip" else c.owner
                if subj is not None and owner == subj:
                    c = Candidate(c.owner, c.relative, c.word, ("pronS",) + c.frame[1:], c.anchor_pos, c.locality)
            relabelled.append(c)
        out = relabelled
        extra: list[Candidate] = []
        seen = {(c.owner, c.relative, c.anchor_pos) for c in out}
        for c in out:
            if c.frame[0] != "pron" or c.frame[-1] == "flip":
                continue
            # the possessive right before the anchor (possibly with modifiers) decides the gender ball
            q = c.anchor_pos - 1
            poss = None
            while q >= 0 and story.sent_of[q] == story.sent_of[c.anchor_pos] and c.anchor_pos - q <= 4:
                if toks[q][0] == "word" and toks[q][1] in POSSESSIVE:
                    poss = toks[q][1]
                    break
                q -= 1
            if poss is None:
                continue
            subj = first_mention.get(story.sent_of[c.anchor_pos])
            g = PRONOUN_GENDER[poss]
            if subj is None or subj == c.owner or subj == c.relative:
                continue
            if self._genders_cache.get(subj) not in (g, None):
                continue
            for owner, rel, frame in ((subj, c.relative, ("pronS",) + c.frame[1:]),
                                      (c.relative, subj, ("pronS",) + c.frame[1:] + ("flip",))):
                key = (owner, rel, c.anchor_pos)
                if key in seen:
                    continue
                seen.add(key)
                extra.append(Candidate(owner, rel, c.word, frame, c.anchor_pos, c.locality + 1))
        return out + extra

    def _gender_term(self, c: Candidate, genders: dict[str, str]) -> float:
        sym = self.lexicon.lookup(c.word)
        g = self.lexicon.gender.get(sym) if sym else None
        e = genders.get(c.relative)
        if g and e and g != e:
            return -self.gender_penalty
        return 0.0

    def _forest(self, story: Story, genders: dict[str, str], resolved: dict[int, str]) -> Reading:
        self._genders_cache = genders
        cands = self.candidates(story, resolved)
        entities = sorted({v for kind, v in story.tokens if kind == "name"})
        ents = set(entities)
        by_anchor: dict[int, list[Candidate]] = defaultdict(list)
        for c in cands:
            if c.owner in ents and c.relative in ents and c.owner != c.relative:
                by_anchor[c.anchor_pos].append(c)
        options: list[list[tuple[float, list[Candidate]]]] = []
        for pos in sorted(by_anchor):
            cs = sorted(by_anchor[pos], key=lambda c: (-self.score(c), c.owner, c.relative, c.frame))
            opts: list[tuple[float, list[Candidate]]] = []
            seen: set[tuple[str, str]] = set()
            for c in cs:
                if (c.owner, c.relative) in seen:
                    continue
                seen.add((c.owner, c.relative))
                flip = c.frame[-1] == "flip"
                mates = []
                mate_keys = set()
                for d in cs:
                    if d is c or d.owner != c.owner or d.frame[1] != "list" or (d.frame[-1] == "flip") != flip:
                        continue
                    if d.relative in mate_keys or d.relative == c.relative:
                        continue
                    mate_keys.add(d.relative)
                    mates.append(d)
                s = self.score(c) + self._gender_term(c, genders) + sum(
                    max(0.0, self.score(d)) + self._gender_term(d, genders) for d in mates)
                opts.append((s, [c] + mates))
                if len(opts) >= self.top_k:
                    break
            opts.append((-self.skip_penalty, []))
            options.append(opts)
        # beam over anchors; state = (score, choices, union-find parents)
        beam: list[tuple[float, list[Candidate], dict[str, str]]] = [(0.0, [], {e: e for e in entities})]

        def find(par: dict[str, str], x: str) -> str:
            while par[x] != x:
                x = par[x]
            return x

        for opts in options:
            nxt = []
            for score, chosen, par in beam:
                for s, edges in opts:
                    p2 = dict(par)
                    for e in edges:
                        ra, rb = find(p2, e.owner), find(p2, e.relative)
                        if ra != rb:
                            p2[ra] = rb
                    nxt.append((score + s, chosen + edges, p2))
            nxt.sort(key=lambda t: -t[0])
            beam = nxt[: self.beam]
        finals = []
        for score, chosen, par in beam:
            comps = len({find(par, e) for e in entities}) if entities else 1
            j = score - self.component_penalty * (comps - 1)
            if self.query and self.query[0] in par and self.query[1] in par \
                    and find(par, self.query[0]) == find(par, self.query[1]):
                j += self.query_bonus
            finals.append((j, chosen))
        finals.sort(key=lambda t: -t[0])
        pool = [c for cs in by_anchor.values() for c in cs]
        alts = [(j, self._span(chosen, entities, pool), chosen) for j, chosen in finals[: self.keep]]
        best = alts[0][1] if alts else []
        return JointReading(genders, cands, best, entities, alternatives=alts)

    def _span(self, chosen: list[Candidate], entities: list[str], pool: list[Candidate]) -> list[Candidate]:
        """Path graph: maximum spanning forest over the chosen facts first (restatements consumed their
        anchor but add no edge that could shortcut a path), then L1's rule over the remaining candidates
        only to connect components the chosen facts left apart."""
        parent = {e: e for e in entities}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        out = []
        order = lambda c: (-self.score(c), c.anchor_pos, c.owner, c.relative)  # noqa: E731
        chosen_ids = {id(c) for c in chosen}
        for group in (sorted(chosen, key=order), sorted((c for c in pool if id(c) not in chosen_ids), key=order)):
            for c in group:
                ra, rb = find(c.owner), find(c.relative)
                if ra != rb:
                    parent[ra] = rb
                    out.append(c)
        return out

    def read(self, story: Story) -> Reading:
        self.query = story.query
        return super().read(story)


# ---------------------------------------------------------------------------------------------
# the v2 program
# ---------------------------------------------------------------------------------------------


def _union(forest: list[Candidate], facts: list[Candidate]) -> list[Candidate]:
    """Voting graph: the chosen facts (restatements included) plus the forest's connecting edges."""
    ids = {id(c) for c in facts}
    return list(facts) + [c for c in forest if id(c) not in ids]


def simple_paths(edges: list[Candidate], a: str, c: str, max_len: int = 14,
                 max_paths: int = 32) -> list[list[tuple[Candidate, int]]]:
    adj: dict[str, list[tuple[str, Candidate, int]]] = defaultdict(list)
    for e in edges:
        adj[e.owner].append((e.relative, e, FWD))
        adj[e.relative].append((e.owner, e, REV))
    if a not in adj or c not in adj:
        return []
    out: list[list[tuple[Candidate, int]]] = []
    frontier: list[tuple[str, list[tuple[Candidate, int]], frozenset]] = [(a, [], frozenset([a]))]
    while frontier and len(out) < max_paths:
        nxt = []
        for node, path, seen in frontier:
            for y, e, d in adj[node]:
                if y in seen:
                    continue
                p2 = path + [(e, d)]
                if y == c:
                    out.append(p2)
                    if len(out) >= max_paths:
                        break
                elif len(p2) < max_len:
                    nxt.append((y, p2, seen | {y}))
            if len(out) >= max_paths:
                break
        frontier = nxt[:4096]
    return out


class FieldProgramV2(FieldProgram):
    """L1's program, trained exactly as L1, with post-fit and inference-time components.

    Joint decoding inside hard-EM training hurt on development data (it chose low-score facts to
    connect the query before frames were learned), so every v2 component is applied after an L1 fit:
    the learned algebra and rare-word placement are learned from the training stories after the fit,
    and joint decoding, soft composition and path voting act at inference only.
    """

    def __init__(self, algebra: bool = False, joint: bool = False, soft: bool = False, vote: bool = False,
                 rare: bool = False, tau: float = 1.0, **joint_kw: Any) -> None:
        super().__init__()
        self._init_v2(algebra, joint, soft, vote, rare, tau, joint_kw)

    def _init_v2(self, algebra: bool, joint: bool, soft: bool, vote: bool, rare: bool, tau: float,
                 joint_kw: dict[str, Any]) -> None:
        self.use_algebra, self.use_joint, self.use_soft, self.use_vote, self.use_rare = \
            algebra, joint, soft, vote, rare
        self.tau = tau
        self.joint_kw = dict(joint_kw)
        self.algebra: Algebra | None = None
        self.algebra_report: dict[str, Any] = {}
        self.rare_report: dict[str, Any] = {}
        self._ctx: dict[str, Counter] = {}
        self._core: list[str] = []
        self._rare_cache: dict[str, str | None] = {}
        self._orig_lookup = self.lexicon.lookup
        self.stats: Counter = Counter()

    @classmethod
    def upgrade(cls, base: FieldProgram, stories: list[Story], **flags: Any) -> "FieldProgramV2":
        """A copy of an L1-trained program with this arm's components applied (the L1 fit is shared)."""
        import copy
        prog = copy.deepcopy(base)
        prog.__class__ = cls
        joint_kw = {k: flags.pop(k) for k in list(flags) if k not in
                    ("algebra", "joint", "soft", "vote", "rare", "tau")}
        prog._init_v2(flags.get("algebra", False), flags.get("joint", False), flags.get("soft", False),
                      flags.get("vote", False), flags.get("rare", False), flags.get("tau", 1.0), joint_kw)
        prog.post_fit(stories)
        return prog

    # -- learning -------------------------------------------------------------------------------
    def fit(self, stories: list[Story], arm: str = "primary", rounds: int = 4) -> dict[str, Any]:
        info = FieldProgram.fit(self, stories, arm=arm, rounds=rounds)
        info.update(self.post_fit(stories))
        return info

    def post_fit(self, stories: list[Story]) -> dict[str, Any]:
        if self.use_rare:
            self.rare_report = self._fit_rare(stories)
            self.reader.invalidate()
        if self.use_algebra:
            self.algebra, self.algebra_report = learn_algebra(self)
        learned: dict[str, float] = {}
        if self.use_joint:
            jr = JointReader(self.lexicon, self.genders, **self.joint_kw)
            jr.frame_score = dict(self.reader.frame_score)
            jr.flip_penalty = self.reader.flip_penalty
            self.reader = jr
            learned = self._learn_new_frames(stories)
        self.reader.invalidate()
        return {"algebra": self.algebra_report, "rare": self.rare_report, "new_frames": learned}

    def _learn_new_frames(self, stories: list[Story]) -> dict[str, float]:
        """One hard-EM frame step for frames L1 never had ("pronS"): log-odds of lying on the path of a
        correctly answered training story. Other frames keep their L1 scores."""
        good: Counter = Counter()
        bad: Counter = Counter()
        for s in stories:
            reading = self.reader.read(s)
            path = story_path(reading, *s.query)
            if not path:
                continue
            ok = self.predict(s).label == s.target
            for c, _ in path:
                if c.frame[0] == "pronS":
                    (good if ok else bad)[c.frame] += 1
        for f in set(good) | set(bad):
            self.reader.frame_score[f] = math.log((good[f] + 1.0) / (bad[f] + 1.0))
        self.reader.invalidate()
        return {"|".join(f): round(self.reader.frame_score[f], 3) for f in sorted(set(good) | set(bad))}

    def _fit_rare(self, stories: list[Story]) -> dict[str, Any]:
        """Place rare relation words by morphology + distribution, then merge orphan classes whose
        composition rows agree with the placement (SPRT with the placement as prior)."""
        self._ctx = context_vectors(stories)
        comp = self.composer
        cls = comp.classes
        # core words: lexicon words whose class takes part in at least 20 composition observations
        mass: Counter = Counter()
        for (a, b), row in comp.table.items():
            n = sum(row.values())
            mass[a] += n
            mass[b] += n
        core = [w for w in self.lexicon.symbol
                if mass[cls.find((self.lexicon.symbol[w], FWD))] + mass[cls.find((self.lexicon.symbol[w], REV))] >= 20]
        orphans = [w for w in self.lexicon.symbol if w not in core and w not in self.labels]
        placed: dict[str, dict[str, Any]] = {}
        obs, _ = self._observations(stories)
        for w in orphans:
            best = self._place(w, core)
            if best is None:
                continue
            target, prior = best
            a = cls.find((self.lexicon.symbol[w], FWD))
            b = cls.find((self.lexicon.symbol[target], FWD))
            if a == b:
                continue
            # evidence from composition rows (both directions)
            left, right = comp.rows()
            agree = disagree = 0
            for rows in (left, right):
                ra, rb = rows.get(a, {}), rows.get(b, {})
                for ctx in set(ra) & set(rb):
                    if ra[ctx] == rb[ctx]:
                        agree += 1
                    else:
                        disagree += 1
            llr = prior + _llr(agree, disagree)
            placed[w] = {"target": target, "prior": round(prior, 3), "agree": agree, "disagree": disagree,
                         "llr": round(llr, 3)}
            if llr >= math.log(20.0):
                cls.union((self.lexicon.symbol[w], FWD), (self.lexicon.symbol[target], FWD))
                if w not in self.lexicon.gender and self.lexicon.symbol[target] in self.lexicon.gender:
                    pass   # the word keeps its own learned gender (mom: f even though its class is parent)
                placed[w]["merged"] = True
        comp.rebuild(obs)
        comp.prune()
        self._core = core
        return {"core": sorted(core), "orphans": sorted(orphans), "placed": placed}

    def _place(self, word: str, core: list[str]) -> tuple[str, float] | None:
        best: tuple[float, str] | None = None
        cv = self._ctx.get(word, Counter())
        for k in core:
            m = morph_logodds(word, k)
            d = cosine(cv, self._ctx.get(k, Counter()))
            # distribution axis: cosine above 0.5 is evidence, below 0.2 against
            dl = 6.0 * (d - 0.35)
            s = m + dl
            if best is None or s > best[0]:
                best = (s, k)
        if best is None or best[0] < 1.5:
            return None
        return best[1], best[0]

    def rare_lookup(self, word: str) -> str | None:
        sym = self._orig_lookup(word)
        if sym is not None or not self.use_rare:
            return sym
        cache = self.__dict__.setdefault("_rare_cache", {})
        if word in cache:
            return cache[word]
        core = getattr(self, "_core", [])
        target = None
        if word in self._ctx:
            # seen in training (e.g. dropped for lack of composition entries): both axes
            placed = self._place(word, core)
            target = placed[0] if placed else None
        else:
            # never seen: spelling alone, and only strong evidence (a typo of a core word)
            best: tuple[float, str] | None = None
            for k in core:
                m = morph_logodds(word, k)
                if m >= 2.5 and (best is None or m > best[0]):
                    best = (m, k)
            target = best[1] if best else None
        out = self.lexicon.symbol[target] if target is not None else None
        cache[word] = out
        if out is not None:
            self.stats["rare_inferred"] += 1
        return out

    # -- inference ------------------------------------------------------------------------------
    def _classes(self, path: list[tuple[Candidate, int]]) -> list[tuple]:
        return [self.composer.classes.find(n) for n in self.seq(path)]

    def _compose(self, path: list[tuple[Candidate, int]]) -> tuple[dict[tuple, float], str]:
        return compose_hybrid(self, self.seq(path))

    def predict(self, story: Story) -> Prediction:
        if self.use_rare:
            self.lexicon.lookup = self.rare_lookup           # type: ignore[method-assign]
        reading = self.reader.read(story)
        a, c = story.query
        gC = reading.genders.get(c)
        alts = getattr(reading, "alternatives", None) if self.use_soft else None
        if alts:
            readings = [(j, _union(forest, all_facts) if self.use_vote else forest) for j, forest, all_facts in alts]
        elif self.use_vote and getattr(reading, "alternatives", None):
            j, forest, all_facts = reading.alternatives[0]
            readings = [(j, _union(forest, all_facts))]
        else:
            readings = [(0.0, reading.edges)]
        top = max(j for j, _ in readings)
        votes: Counter = Counter()
        any_path = False
        undefined = 0
        best_syms: list[str] = []
        best_len = None
        for j, edges in readings:
            wr = math.exp((j - top) / self.tau)
            if self.use_vote:
                paths = simple_paths(edges, a, c)
            else:
                if edges is reading.edges and not self.use_joint:
                    p = story_path(reading, a, c)
                else:
                    p = story_path(Reading(reading.genders, [], edges, reading.entities), a, c)
                paths = [p] if p else []
            if not paths:
                continue
            any_path = True
            weights = []
            for p in paths:
                lw = sum(math.log(1.0 / (1.0 + math.exp(-self.reader.score(e)))) for e, _ in p)
                weights.append(lw)
            m = max(weights)
            norm = sum(math.exp(w - m) for w in weights)
            for p, lw in zip(paths, weights):
                dist, how = self._compose(p)
                if not dist:
                    undefined += 1
                    continue
                cls = max(dist.items(), key=lambda kv: kv[1])[0]
                label = self.composer.label_for(cls, gC)
                if label is None:
                    continue
                votes[label] += wr * math.exp(lw - m) / norm
                if not best_syms:
                    best_syms = [f"{s}{'' if d == FWD else '^-1'}" for s, d in self.seq(p)]
                    best_len = len(p)
        if self.use_rare:
            self.lexicon.lookup = self._orig_lookup          # type: ignore[method-assign]
        if not any_path:
            return Prediction(None, None, "no-path")
        if not votes:
            return Prediction(None, best_len, "undefined-composition" if undefined else "no-realisation", best_syms)
        label = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return Prediction(label, best_len, "ok", best_syms)

    def export(self) -> dict[str, Any]:
        out = super().export()
        if self.algebra is not None:
            cls = self.composer.classes
            members: dict[tuple, list[str]] = defaultdict(list)
            for node in list(cls.parent):
                members[cls.find(node)].append(f"{node[0]}{'' if node[1] == FWD else '^-1'}")
            name = {c: min(v, key=lambda s: (len(s), s)) for c, v in members.items()}
            out["algebra"] = self.algebra.export(name)
        out["algebra_report"] = self.algebra_report
        out["rare_report"] = self.rare_report
        return out

    def bits(self) -> dict[str, float]:
        parts = super().bits()
        if self.algebra is not None:
            n_cls = len(self.composer.classes.classes())
            ab = self.algebra.bits(self.composer.table, n_cls)
            # the algebra replaces the table except for its exceptions (counted in ab["data"])
            parts["composition"] = ab["total"] + len(self.composer.realise) * (math.log2(max(2, n_cls)) + 1.0
                                                                               + math.log2(24))
            parts["total"] = sum(v for k, v in parts.items() if k != "total")
        return parts


ARMS: dict[str, dict[str, Any]] = {
    "L1": {},
    "A": {"algebra": True},
    "R": {"rare": True},
    "J": {"joint": True},
    "JS": {"joint": True, "soft": True},
    "JV": {"joint": True, "vote": True},
    "FULL": {"algebra": True, "joint": True, "soft": True, "vote": True, "rare": True},
    "FULL-noA": {"joint": True, "soft": True, "vote": True, "rare": True},
}


def make_program(arm: str, **overrides: Any) -> FieldProgram:
    cfg = {**ARMS[arm], **overrides}
    if not cfg:
        return FieldProgram()
    return FieldProgramV2(**cfg)


def arm_program(arm: str, base: FieldProgram, stories: list[Story], **overrides: Any) -> FieldProgram:
    """The arm's program from a shared L1 fit (L1 itself is the base, unchanged)."""
    cfg = {**ARMS[arm], **overrides}
    if not cfg:
        return base
    return FieldProgramV2.upgrade(base, stories, **cfg)


ALGEBRA_FALLBACK_LOGP = math.log(0.5)


def compose_hybrid(program: FieldProgram, seq: list[tuple]) -> tuple[dict[tuple, float], str]:
    """The MDL model is "algebra + stored exceptions": a binary step uses the learned table entry when
    one exists (matched entries agree with the algebra; exceptions override it) and the algebra only for
    pairs the table never saw. CYK over bracketings as in L1; without an algebra this is L1's composition."""
    comp = program.composer
    alg = getattr(program, "algebra", None)
    dist, _ = comp.compose_rewrite(seq)
    if dist or alg is None:
        # L1's composition wherever it is defined: the algebra only fills what the table leaves undefined
        return dist, "table" if dist else "undefined"
    k = len(seq)
    if k == 0:
        return {}, "undefined"
    span: dict[tuple[int, int], dict[tuple, float]] = {}
    used_alg = False
    for i, node in enumerate(seq):
        span[(i, i + 1)] = {comp.cls(node): 0.0}
    for length in range(2, k + 1):
        for i in range(0, k - length + 1):
            j = i + length
            cell: dict[tuple, float] = {}
            for m in range(i + 1, j):
                for lc, lp in span[(i, m)].items():
                    for rc, rp in span[(m, j)].items():
                        row = comp.table.get((lc, rc))
                        if row and sum(row.values()) >= comp.min_count:
                            out, cnt = row.most_common(1)[0]
                            p = lp + rp + math.log((cnt + 0.5) / (sum(row.values()) + 1.0))
                        else:
                            out = alg.compose_classes([lc, rc])
                            if out is None:
                                continue
                            p = lp + rp + ALGEBRA_FALLBACK_LOGP
                            used_alg = True
                        if p > cell.get(out, -1e18):
                            cell[out] = p
            span[(i, j)] = dict(sorted(cell.items(), key=lambda kv: -kv[1])[:6])
    dist = span[(0, k)]
    return dist, ("algebra" if used_alg else "table") if dist else "undefined"


def composition_only_v2(program: FieldProgram, stories: list[Story]) -> dict[str, Any]:
    """Post hoc D5 with the arm's composition (algebra when present, else table) on gold sequences."""
    by_k: dict[int, list[int]] = defaultdict(list)
    for s in stories:
        if not s.gold_edges:
            continue
        seq = [program.composer.classes.node(program.lexicon.lookup(rel) or rel, FWD) for _, rel, _ in s.gold_edges]
        label = None
        dist, _ = compose_hybrid(program, seq)
        if dist:
            cls = max(dist.items(), key=lambda kv: kv[1])[0]
            label = program.composer.label_for(cls, s.gold_genders.get(s.query[1]))
        by_k[s.k].append(int(label == s.target))
    return {str(k): {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in sorted(by_k.items())}
