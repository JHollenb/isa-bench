"""CLUTRR field model v4 (L1d): a conditional referent model for pronouns, trained on the answer signal only.
Preregistered in docs/clutrr-referent-ranker-proposal.md.

L1c measured that the reader proposes 99.4% of the gold pairs of long stories but chooses 86%, and that the
losses are mostly pronoun referents: a possessive bound by recency to a nearer name (often the possessor inside
an appositive) instead of its clause subject. L1c's salience arm added a per-role score to each candidate fact;
that is not a comparison between the competing referents of one pronoun, and it lost on dev.

This module replaces the recency order of each pronoun's coset join (the names not known to be of the other
gender, most recent first) with a conditional logit over those same names:

    P(e | pronoun i) = exp(w . phi(e, i)) / sum_{e' in coset(i)} exp(w . phi(e', i))

phi is generic and text-only: recency rank, token distance, same sentence, the role of the name's latest
mention (sentence subject, possessor, possessor inside an appositive, the relative named by a relation phrase
of the program's learned lexicon: "his brother [Scott]"), whether the name is the subject of the
pronoun's sentence or of the previous one, prior mention count and whether its gender is known; each feature
also interacts with the pronoun's kind (possessive, subject, object). The initial weights w0 reproduce recency
exactly (only the rank-0 feature is on), so a reader at w0 is L1c's reader.

Learning uses answers only. For each training story and each pronoun whose coset holds two or more names,
every name is forced in turn as the referent (other pronouns keep the current model's choice) and the story is
re-answered by the unchanged field program. Pronouns where some but not all names give the gold answer are
training instances; the names that give it are the positive set G. w maximises
    sum_i log sum_{e in G_i} P(e | i)  -  lambda/2 |w - w0|^2
(latent positives; L-BFGS from w0), and the instances are re-collected under the new model (two passes).
The reader then orders every coset join by w: the top name is the pronoun's referent (frames "pron"/"poss"),
the second is L1's alternative ("pron2"/"pronS"), and the recall reader's options follow the same order.
Everything else (lexicon, frames, algebra, joint decoding, soft composition) is the L1c program unchanged.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from scipy.optimize import minimize

from .clutrr_field import PRONOUN_GENDER, Candidate, FieldProgram, Reader, Story
from .clutrr_field_v2 import learn_algebra
from .clutrr_field_v3 import ARMS as V3_ARMS, V3_KEYS, FieldProgramV3, RecallReader, candidate_recall

# ---------------------------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------------------------

BASE_FEATURES = ("rank0", "rank1", "same_sent", "dist", "role_subj", "role_possr", "role_appos", "role_relobj",
                 "subj_here", "subj_prev", "mentions", "gender_known")
KINDS = ("poss", "subj", "obj")
FEATURES = tuple(f"all:{f}" for f in BASE_FEATURES) + tuple(f"{k}:{f}" for k in KINDS for f in BASE_FEATURES)
NB = len(BASE_FEATURES)
NF = len(FEATURES)
W0 = np.zeros(NF)
W0[FEATURES.index("all:rank0")] = 1.0          # recency: the most recent name wins, ties keep recency order

# a word after "her" that makes it objective rather than possessive (generic function words)
NOT_A_NOUN = {"and", "to", "with", "for", "at", "in", "on", "that", "who", "because", "about", "but", "so", "as",
              "from", "out", "up", "a", "the", "an", "is", "was", "by", "of", "into", "over", "around", "again",
              "back", "after", "before", "when", "while", "if", "since", "too", "very", "how", "what", "where",
              "then", "down", "off", "home", "some", "all", "every", "whenever", "once", "yesterday", "today"}


def pronoun_kind(story: Story, i: int) -> str:
    v = story.tokens[i][1]
    if v in ("he", "she"):
        return "subj"
    if v == "his":
        return "poss"
    if v == "her":
        nxt = story.tokens[i + 1] if i + 1 < len(story.tokens) else ("punct", ".")
        return "poss" if nxt[0] == "word" and nxt[1] not in NOT_A_NOUN and nxt[1] not in PRONOUN_GENDER else "obj"
    return "obj"


def _layout(story: Story) -> dict[str, Any]:
    """Per-story token facts the features read (cached on the story object)."""
    lay = getattr(story, "_ref_layout", None)
    if lay is not None:
        return lay
    toks = story.tokens
    first_name: dict[int, tuple[int, str]] = {}
    mentions: dict[str, list[int]] = defaultdict(list)
    for j, (k, v) in enumerate(toks):
        if k == "name":
            first_name.setdefault(story.sent_of[j], (j, v))
            mentions[v].append(j)
    lay = {"first_name": first_name, "mentions": dict(mentions)}
    story._ref_layout = lay                    # type: ignore[attr-defined]
    return lay


def _introduced_by_anchor(story: Story, j: int, is_anchor) -> bool:
    """The mention at j is the relative of a relation phrase: an anchor word (the program's learned lexicon)
    at most three tokens before it in its sentence, with no other name in between ("her brother , [Scott]")."""
    toks = story.tokens
    for p in range(j - 1, max(-1, j - 4), -1):
        if story.sent_of[p] != story.sent_of[j] or toks[p][0] == "name":
            return False
        if toks[p][0] == "word" and is_anchor(toks[p][1]):
            return True
    return False


def base_features(story: Story, i: int, e: str, rank: int, genders: dict[str, str], g: str | None,
                  is_anchor=None) -> list[float]:
    toks = story.tokens
    lay = _layout(story)
    sent = story.sent_of[i]
    before = [j for j in lay["mentions"].get(e, []) if j < i]
    j = before[-1] if before else i
    sj = story.sent_of[j]
    fn = lay["first_name"]
    possr = j + 1 < len(toks) and toks[j + 1] == ("word", "'s")
    appos = possr and j >= 2 and toks[j - 1] == ("punct", ",") and toks[j - 2][0] == "name"
    here = fn.get(sent)
    prev = fn.get(sent - 1)
    relobj = is_anchor is not None and j < i and _introduced_by_anchor(story, j, is_anchor)
    return [float(rank == 0), float(rank == 1), float(sj == sent), min(i - j, 40) / 10.0,
            float(fn.get(sj, (-1, None))[0] == j), float(possr), float(appos), float(relobj),
            float(here is not None and here[0] < i and here[1] == e), float(prev is not None and prev[1] == e),
            math.log1p(len(before)), float(g is not None and genders.get(e) == g)]


def referent_features(story: Story, i: int, cands: list[str], genders: dict[str, str], is_anchor=None) -> np.ndarray:
    """Rows of phi(e, i) for the coset's names in recency order (shared block + the pronoun kind's block)."""
    g = PRONOUN_GENDER.get(story.tokens[i][1])
    kind = pronoun_kind(story, i)
    k = KINDS.index(kind)
    X = np.zeros((len(cands), NF))
    for r, e in enumerate(cands):
        b = base_features(story, i, e, r, genders, g, is_anchor)
        X[r, :NB] = b
        X[r, NB * (1 + k): NB * (2 + k)] = b
    return X


# ---------------------------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------------------------


class RankedReader(RecallReader):
    """L1c's reader whose pronoun coset joins are ordered by the conditional referent model (w0 = recency)."""

    def __init__(self, lexicon, genders, weights: np.ndarray | None = None, **kw: Any) -> None:
        super().__init__(lexicon, genders, **kw)
        self.weights = np.array(W0 if weights is None else weights, dtype=float)
        self.force: dict[int, str] = {}

    def _order(self, story: Story, pos: int, ranked: list[str], genders: dict[str, str]) -> list[str]:
        f = self.force.get(pos)
        if f is not None and f in ranked:
            return [f] + [e for e in ranked if e != f]
        if len(ranked) < 2:
            return ranked
        s = referent_features(story, pos, ranked, genders, self.is_anchor) @ self.weights
        order = sorted(range(len(ranked)), key=lambda r: (-s[r], r))
        return [ranked[r] for r in order]

    def is_anchor(self, word: str) -> bool:
        return self.lexicon.lookup(word) is not None

    def _ranked(self, story: Story, pos: int, g: str, genders: dict[str, str]) -> list[str]:
        return self._order(story, pos, Reader._ranked(self, story, pos, g, genders), genders)

    def _ranked_wide(self, story: Story, pos: int, g: str, genders: dict[str, str]) -> list[str]:
        return self._order(story, pos, RecallReader._ranked_wide(self, story, pos, g, genders), genders)

    def recency_coset(self, story: Story, pos: int, genders: dict[str, str]) -> list[str]:
        """The unordered-by-model coset (L1's recency join, one sentence back), for training and oracles."""
        return Reader._ranked(self, story, pos, PRONOUN_GENDER[story.tokens[pos][1]], genders)


def _forced_label(prog: FieldProgram, story: Story, force: dict[int, str]) -> str | None:
    reader = prog.reader
    reader.force = force
    reader._cache.pop(story.sid, None)
    try:
        return prog.predict(story).label
    finally:
        reader.force = {}
        reader._cache.pop(story.sid, None)


def _pronouns(story: Story) -> list[int]:
    return [i for i, (k, v) in enumerate(story.tokens) if k == "word" and v in PRONOUN_GENDER]


# ---------------------------------------------------------------------------------------------
# learning (answer signal only)
# ---------------------------------------------------------------------------------------------


def collect_instances(prog: FieldProgram, stories: list[Story]) -> tuple[list[tuple[np.ndarray, np.ndarray]], Counter]:
    """Counterfactual referents: (phi rows, positive mask) for every answer-decisive pronoun."""
    reader: RankedReader = prog.reader               # type: ignore[assignment]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    stats: Counter = Counter()
    for s in stories:
        reading = reader.read(s)
        genders = reading.genders
        for i in _pronouns(s):
            cands = reader.recency_coset(s, i, genders)
            if len(cands) < 2:
                continue
            stats["ambiguous_pronouns"] += 1
            oks = np.array([_forced_label(prog, s, {i: e}) == s.target for e in cands], dtype=bool)
            stats["forced_reads"] += len(cands)
            if oks.any() and not oks.all():
                out.append((referent_features(s, i, cands, genders, reader.is_anchor), oks))
                stats["decisive"] += 1
                stats[f"decisive_{pronoun_kind(s, i)}"] += 1
                stats["decisive_recency_wrong"] += int(not oks[0])
            elif oks.all():
                stats["all_correct"] += 1
            else:
                stats["none_correct"] += 1
    return out, stats


def fit_weights(instances: list[tuple[np.ndarray, np.ndarray]], lam: float, w0: np.ndarray = W0) -> tuple[np.ndarray, dict[str, Any]]:
    """Latent-positive conditional logit, L2 toward w0, L-BFGS from w0."""
    if not instances:
        return w0.copy(), {"instances": 0}
    X = np.concatenate([x for x, _ in instances])
    pos = np.concatenate([m for _, m in instances]).astype(float)
    sizes = np.array([len(m) for _, m in instances])
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    group = np.repeat(np.arange(len(instances)), sizes)

    def f(w: np.ndarray) -> tuple[float, np.ndarray]:
        s = X @ w
        mx = np.maximum.reduceat(s, starts)
        ex = np.exp(s - mx[group])
        z = np.add.reduceat(ex, starts)
        zp = np.add.reduceat(ex * pos, starts)
        nll = -np.sum(np.log(zp) - np.log(z))
        p = ex / z[group]
        q = ex * pos / zp[group]
        grad = X.T @ (p - q)
        d = w - w0
        return nll + 0.5 * lam * float(d @ d), grad + lam * d

    res = minimize(f, w0.copy(), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    w = res.x
    s = X @ w
    top_ok = [bool(m[int(np.argmax(s[a:a + n]))]) for a, n, (_, m) in zip(starts, sizes, instances)]
    top0 = [bool(m[0]) for _, m in instances]
    return w, {"instances": len(instances), "nll": float(res.fun), "converged": bool(res.success),
               "iterations": int(res.nit), "train_top1_in_G": sum(top_ok) / len(top_ok),
               "recency_top1_in_G": sum(top0) / len(top0)}


def learn_referents(prog: FieldProgram, stories: list[Story], lam: float, passes: int) -> dict[str, Any]:
    reader: RankedReader = prog.reader               # type: ignore[assignment]
    history = []
    for _ in range(passes):
        inst, stats = collect_instances(prog, stories)
        w, info = fit_weights(inst, lam)
        reader.weights = w
        reader.invalidate()
        history.append({"stats": dict(stats), "fit": info})
    return {"lambda": lam, "passes": passes, "history": history,
            "weights": {name: round(float(v), 4) for name, v in zip(FEATURES, reader.weights) if abs(v) > 1e-9}}


# ---------------------------------------------------------------------------------------------
# the v4 program
# ---------------------------------------------------------------------------------------------

V4_KEYS = ("referent", "ref_lambda", "ref_passes")


class FieldProgramV4(FieldProgramV3):
    """L1c's program with the conditional referent model (only the pronoun coset order changes)."""

    def _init_v4(self, referent: bool = False, ref_lambda: float = 1.0, ref_passes: int = 2) -> None:
        self.use_referent, self.ref_lambda, self.ref_passes = referent, ref_lambda, ref_passes
        self.referent_report: dict[str, Any] = {}

    @classmethod
    def upgrade(cls, base: FieldProgram, stories: list[Story], **flags: Any) -> "FieldProgramV4":
        v4 = {k: flags.pop(k) for k in V4_KEYS if k in flags}
        prog = copy.deepcopy(base)
        prog.__class__ = cls
        v3 = {k: flags.pop(k) for k in V3_KEYS if k in flags}
        joint_kw = {k: flags.pop(k) for k in list(flags) if k not in ("algebra", "joint", "soft", "vote", "rare", "tau")}
        prog._init_v2(flags.get("algebra", False), flags.get("joint", False), flags.get("soft", False), False,
                      flags.get("rare", False), flags.get("tau", 1.0), joint_kw)
        prog._init_v3(**v3)
        prog._init_v4(**v4)
        prog.post_fit(stories)
        return prog

    def post_fit(self, stories: list[Story]) -> dict[str, Any]:
        """FieldProgramV3.post_fit with the ranked reader (identical to L1c at w0), then the referent model."""
        self._base_frames = set(self.reader.frame_score)
        if self.use_rare:
            self.rare_report = self._fit_rare(stories)
            self.reader.invalidate()
        if self.use_algebra:
            self.algebra, self.algebra_report = learn_algebra(self)
        learned: dict[str, float] = {}
        if self.use_joint:
            jr = RankedReader(self.lexicon, self.genders, recall=self.use_recall, pron_rank=self.pron_rank,
                              pron_window=self.pron_window, salience=self.use_salience, **self.joint_kw)
            jr.frame_score = dict(self.reader.frame_score)
            jr.flip_penalty = self.reader.flip_penalty
            self.reader = jr
            learned = self._learn_new_frames(stories)
            if self.use_salience:
                self.salience_report = self._learn_roles(stories)
            if self.use_referent:
                self.referent_report = learn_referents(self, stories, self.ref_lambda, self.ref_passes)
        self.reader.invalidate()
        return {"algebra": self.algebra_report, "rare": self.rare_report, "new_frames": learned,
                "referent": self.referent_report}

    def bits(self) -> dict[str, float]:
        parts = super().bits()
        if self.use_referent:
            nz = int(np.sum(np.abs(np.asarray(self.reader.weights) - W0) > 1e-9))
            parts["referent_model"] = nz * (16.0 + math.log2(NF))     # 16-bit weights plus their index
            parts["total"] = sum(v for k, v in parts.items() if k != "total")
        return parts

    def export(self) -> dict[str, Any]:
        out = super().export()
        if self.use_referent:
            out["referent_report"] = self.referent_report
        return out


ARMS: dict[str, dict[str, Any]] = {
    "L1": {},
    "JSA": dict(V3_ARMS["JSA"]),                     # L1c baseline (identity: must reproduce L1c's JSA)
    "FULL": dict(V3_ARMS["FULL"]),                   # L1c primary (identity: must reproduce L1c's FULL)
    "RK": {**V3_ARMS["JSA"], "referent": True},     # primary: JSA + the referent model
    "RKF": {**V3_ARMS["FULL"], "referent": True},   # reported: FULL + the referent model
}


def arm_program(arm: str, base: FieldProgram, stories: list[Story], **overrides: Any) -> FieldProgram:
    cfg = {**ARMS[arm], **overrides}
    if arm == "L1":
        return base
    if arm in ("JSA", "FULL"):
        cfg = {k: v for k, v in cfg.items() if k not in V4_KEYS}
        return FieldProgramV3.upgrade(base, stories, **cfg)
    return FieldProgramV4.upgrade(base, stories, **cfg)


# ---------------------------------------------------------------------------------------------
# post hoc: oracle ceiling (dev only) and failure classes
# ---------------------------------------------------------------------------------------------


def oracle_single(prog: FieldProgramV4, stories: list[Story]) -> dict[str, Any]:
    """Dev only (uses answers at inference): accuracy when any ONE pronoun's referent may be replaced by
    another name of its coset. A lower bound on what referent choice alone can buy."""
    reader: RankedReader = prog.reader               # type: ignore[assignment]
    by_k: dict[int, Counter] = defaultdict(Counter)
    for s in stories:
        c = by_k[s.k]
        c["n"] += 1
        if prog.predict(s).label == s.target:
            c["correct"] += 1
            c["oracle"] += 1
            continue
        genders = reader.read(s).genders
        done = False
        for i in _pronouns(s):
            cands = reader.recency_coset(s, i, genders)
            if len(cands) < 2:
                continue
            for e in cands:
                if _forced_label(prog, s, {i: e}) == s.target:
                    done = True
                    break
            if done:
                break
        c["oracle"] += int(done)
    tot = Counter()
    for c in by_k.values():
        tot.update(c)
    return {"by_k": {str(k): {"accuracy": v["correct"] / v["n"], "oracle": v["oracle"] / v["n"], "n": v["n"]}
                     for k, v in sorted(by_k.items())},
            "accuracy": tot["correct"] / max(1, tot["n"]), "oracle": tot["oracle"] / max(1, tot["n"])}


NEUTRAL = {"son": "child", "daughter": "child", "father": "parent", "mother": "parent", "grandson": "grandchild",
           "granddaughter": "grandchild", "grandfather": "grandparent", "grandmother": "grandparent",
           "nephew": "nibling", "niece": "nibling", "uncle": "pibling", "aunt": "pibling"}


def text_contradicts_label(prog: FieldProgram, s: Story) -> bool:
    """L1's D4 flag: a chosen fact reads the same asymmetric relation type as a gold edge, reversed."""
    gold = {(x, y): NEUTRAL.get(w) for x, w, y in s.gold_edges}
    for e in prog.reader.read(s).edges:
        t = NEUTRAL.get(prog.lexicon.lookup(e.word) or e.word)
        if t and gold.get((e.relative, e.owner)) == t:
            return True
    return False


def failure_classes(prog: FieldProgram, stories: list[Story], ks: tuple[str, ...], flag_progs: list[FieldProgram]) -> dict[str, Any]:
    """Post hoc (gold edges used here only): each failed story at the given k, first matching class:
    label-contradiction (L1's D4 flag under any of flag_progs' readings) > gold-read (every gold pair chosen:
    composition, direction or realisation) > referent (a missed gold pair's anchor chose a pronoun-supplied
    fact) > candidate-miss (a gold pair absent from candidates) > other-selection."""
    counts: Counter = Counter()
    n = fails = 0
    for s in stories:
        if str(s.k) not in ks or not s.gold_edges:
            continue
        n += 1
        if prog.predict(s).label == s.target:
            continue
        fails += 1
        if any(text_contradicts_label(p, s) for p in flag_progs):
            counts["label-contradiction"] += 1
            continue
        reading = prog.reader.read(s)
        chosen = {frozenset((c.owner, c.relative)) for c in reading.edges}
        missed = [frozenset((x, y)) for x, _, y in s.gold_edges if frozenset((x, y)) not in chosen]
        if not missed:
            counts["gold-read"] += 1
            continue
        by_anchor: dict[int, list[Candidate]] = defaultdict(list)
        for c in reading.edges:
            by_anchor[c.anchor_pos].append(c)
        cls = "other-selection"
        cand_miss = False
        for pair in missed:
            hits = [c for c in reading.candidates if frozenset((c.owner, c.relative)) == pair]
            if not hits:
                cand_miss = True
                continue
            if any(RecallReader._pron_endpoint(d) is not None or d.frame[0] in ("pron", "pron2", "pronS")
                   for h in hits for d in by_anchor.get(h.anchor_pos, [])):
                cls = "referent"
                break
        if cls != "referent" and cand_miss:
            cls = "candidate-miss"
        counts[cls] += 1
    return {"n": n, "failures": fails, "classes": dict(counts),
            "share_of_failures": {k: v / max(1, fails) for k, v in counts.items()}}


__all__ = ["ARMS", "FEATURES", "FieldProgramV4", "RankedReader", "W0", "arm_program", "candidate_recall",
           "collect_instances", "failure_classes", "fit_weights", "learn_referents", "oracle_single",
           "pronoun_kind", "referent_features"]
