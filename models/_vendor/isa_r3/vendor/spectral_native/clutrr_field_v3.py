"""CLUTRR field model v3 (L1c): recall-oriented candidates, cycle consistency instead of path voting and
evidence-gated rare-word merges, on top of L1 (``clutrr_field``) and L1b's learned algebra plus joint and
soft composition (``clutrr_field_v2``). Preregistered in docs/clutrr-field-v3-findings.md.

  recall  every relation-word occurrence (anchor) proposes every ordered pair of mentions in its sentence:
          names, and each pronoun's top-ranked recency-coset referents within a two-sentence window.
          The next sentence's first name joins when the anchor's sentence holds fewer than two people.
          Possessive chains ("X's W1's W2 Y") propose one composite fact whose relation is W1 then W2.
          New frames are coarse (anchor side x mention kind); their reliabilities come from the same
          hard-EM step L1b used for its new frame. Joint decoding prunes.
  cycle   paths are not independent codewords. In a correct reading every cycle composes consistently:
          a chosen fact outside the reading's spanning forest must equal the composition of the forest
          path between its people, and the forest path between distinct query people must not reduce to
          the identity. Each violation costs lambda in the joint score, which is a sum of per-anchor
          log-odds against the skip option (a well-read extra fact raises it; no length bias).
  merge   a rare word joins a core class only with composition evidence that separates the class from its
          inverse: at least one agreeing context where class and inverse compose differently, and no
          disagreeing context. The spelling/context prior alone never merges.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from itertools import permutations
from typing import Any

from .clutrr_field import (FWD, POSSESSIVE, PRONOUN_GENDER, REV, Candidate, FieldProgram, Prediction,
                           Reading, Story, _llr, story_path)
from .clutrr_field_v2 import (ARMS as V2_ARMS, FieldProgramV2, JointReader, compose_hybrid, context_vectors,
                              learn_algebra)

COMPOSITE = "+"


# ---------------------------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------------------------


class RecallReader(JointReader):
    """L1b's joint reader with a recall-oriented candidate generator (switchable)."""

    def __init__(self, lexicon, genders, recall: bool = True, pron_rank: int = 3, pron_window: int = 2,
                 salience: bool = False, **joint_kw: Any) -> None:
        super().__init__(lexicon, genders, **joint_kw)
        self.recall, self.pron_rank, self.pron_window = recall, pron_rank, pron_window
        self.salience = salience
        self.role_delta: dict[str, float] = {}

    # -- salience (logged post-dev change, see findings) -----------------------------------------
    PRON_CUES = {"pron", "pron2", "pronS"}

    @staticmethod
    def _pron_endpoint(c: Candidate) -> str | None:
        """The entity a pronoun supplied to this fact, if any."""
        head, rel = c.frame[0], c.frame[1]
        if head in RecallReader.PRON_CUES:
            return c.relative if c.frame[-1] == "flip" else c.owner
        if head.startswith("w") and head[-1] in "pq":
            return c.owner
        if rel.startswith("w") and rel[-1] in "pq":
            return c.relative
        return None

    @staticmethod
    def _role(story: Story, entity: str, anchor: int) -> str:
        """Syntactic role of the referent's most recent mention before the anchor (generic salience)."""
        toks = story.tokens
        for i in range(anchor - 1, -1, -1):
            if toks[i] == ("name", entity):
                sent = story.sent_of[i]
                first = next(j for j in range(len(toks)) if story.sent_of[j] == sent and toks[j][0] == "name")
                role = "subj" if first == i else ("possr" if i + 1 < len(toks) and toks[i + 1][1] == "'s" else "obj")
                return ("prev-" if sent < story.sent_of[anchor] else "") + role
        return "none"

    def _tag(self, story: Story, cands: list[Candidate]) -> list[Candidate]:
        out = []
        for c in cands:
            e = self._pron_endpoint(c)
            if e is None:
                out.append(c)
                continue
            role = "@" + self._role(story, e, c.anchor_pos)
            out.append(Candidate(c.owner, c.relative, c.word, c.frame[:2] + (role,) + c.frame[2:], c.anchor_pos,
                                 c.locality))
        return out

    def score(self, c: Candidate) -> float:
        roles = [x for x in c.frame if x.startswith("@")]
        if not roles:
            return super().score(c)
        base = tuple(x for x in c.frame if not x.startswith("@"))
        learned = self.frame_score.get(c.frame, self.frame_score.get(base, 0.0))
        flip = -self.flip_penalty if base[-1] == "flip" else 0.0
        return self.prior(base) + learned + flip - 0.02 * c.locality + self.role_delta.get(roles[0], 0.0)

    def _ranked_wide(self, story: Story, pos: int, g: str, genders: dict[str, str]) -> list[str]:
        """Recency coset join over a wider window: earlier names not known to be the other gender."""
        sent = story.sent_of[pos]
        ranked: list[str] = []
        for i in range(pos - 1, -1, -1):
            if story.sent_of[i] < sent - self.pron_window:
                break
            kind, value = story.tokens[i]
            if kind == "name" and genders.get(value) in (g, None) and value not in ranked:
                ranked.append(value)
        return ranked

    def candidates(self, story: Story, resolved: dict[int, str]) -> list[Candidate]:
        out = self._recall_candidates(story, resolved)
        return self._tag(story, out) if self.salience else out

    def _recall_candidates(self, story: Story, resolved: dict[int, str]) -> list[Candidate]:
        out = super().candidates(story, resolved)
        if not self.recall:
            return out
        toks = story.tokens
        n = len(toks)
        genders = self._genders_cache
        options: dict[int, list[tuple[str, str]]] = {}
        for i, (kind, value) in enumerate(toks):
            if kind == "name":
                options[i] = [(value, "n")]
            elif kind == "word" and value in PRONOUN_GENDER:
                ranked = self._ranked_wide(story, i, PRONOUN_GENDER[value], genders)
                if i in resolved:
                    ranked = [resolved[i]] + [e for e in ranked if e != resolved[i]]
                opts = [(e, "p" if r == 0 else "q") for r, e in enumerate(ranked[: self.pron_rank])]
                if opts:
                    options[i] = opts
        by_sent: dict[int, list[int]] = defaultdict(list)
        for i in sorted(options):
            by_sent[story.sent_of[i]].append(i)
        first_name: dict[int, tuple[int, str]] = {}
        for i, (kind, value) in enumerate(toks):
            if kind == "name":
                first_name.setdefault(story.sent_of[i], (i, value))
        anchors = [(i, v) for i, (k, v) in enumerate(toks) if k == "word" and self.lexicon.lookup(v) is not None]
        seen = {(c.owner, c.relative, c.anchor_pos) for c in out}
        extra: list[Candidate] = []
        for a, word in anchors:
            sent = story.sent_of[a]
            ments = [(p, e, k) for p in by_sent.get(sent, []) if p != a for e, k in options[p]]
            if len({e for _, e, _ in ments}) < 2 and sent + 1 in first_name:
                p, e = first_name[sent + 1]
                ments.append((p, e, "d"))
            for (p1, e1, k1), (p2, e2, k2) in permutations(ments, 2):
                if e1 == e2 or p1 == p2 or (e1, e2, a) in seen:
                    continue
                seen.add((e1, e2, a))
                frame = (f"w{'B' if p1 < a else 'A'}{k1}", f"w{'B' if p2 < a else 'A'}{k2}")
                extra.append(Candidate(e1, e2, word, frame, a, abs(p1 - a) + abs(p2 - a)))
        # possessive chains: X 's W1 's W2 (fillers<=3) Y   |   his/her W1 's W2 (fillers<=3) Y
        for a2, w2 in anchors:
            if a2 < 2 or toks[a2 - 1][1] != "'s" or toks[a2 - 2][0] != "word":
                continue
            a1, w1 = a2 - 2, toks[a2 - 2][1]
            if self.lexicon.lookup(w1) is None:
                continue
            owner = None
            if a1 >= 2 and toks[a1 - 1][1] == "'s" and toks[a1 - 2][0] == "name":
                owner = toks[a1 - 2][1]
            elif a1 >= 1 and toks[a1 - 1][1] in POSSESSIVE and (a1 - 1) in resolved:
                owner = resolved[a1 - 1]
            if owner is None:
                continue
            rel, p, fill = None, a2 + 1, 0
            while p < n and story.sent_of[p] == story.sent_of[a2] and fill <= 3:
                if toks[p][0] == "name":
                    rel = toks[p][1]
                    break
                fill += 1
                p += 1
            if rel is None or rel == owner:
                continue
            extra.append(Candidate(owner, rel, f"{w1}{COMPOSITE}{w2}", ("nest", "after"), a2, (a2 - a1) + (p - a2)))
        return out + extra


# ---------------------------------------------------------------------------------------------
# the v3 program
# ---------------------------------------------------------------------------------------------


def gated_merge(agree: int, disagree: int, directional: int, llr: float) -> bool:
    """L1c merge rule: composition evidence that separates class from inverse, none against, LLR >= ln 20."""
    return directional >= 1 and disagree == 0 and agree >= 1 and llr >= math.log(20.0)


class FieldProgramV3(FieldProgramV2):
    """L1's fit, L1b's algebra and joint + soft composition, plus L1c's recall, cycle and merge fixes."""

    def __init__(self, **flags: Any) -> None:
        v3 = {k: flags.pop(k) for k in V3_KEYS if k in flags}
        super().__init__(**flags)
        self._init_v3(**v3)

    def _init_v3(self, recall: bool = False, cycle: bool = False, mergefix: bool = False, salience: bool = False,
                 cycle_lambda: float = 1.0, pron_rank: int = 3, pron_window: int = 2) -> None:
        self.use_recall, self.use_cycle, self.use_mergefix = recall, cycle, mergefix
        self.use_salience = salience
        self.salience_report: dict[str, Any] = {}
        self.cycle_lambda, self.pron_rank, self.pron_window = cycle_lambda, pron_rank, pron_window
        self._base_frames: set = set()

    @classmethod
    def upgrade(cls, base: FieldProgram, stories: list[Story], **flags: Any) -> "FieldProgramV3":
        prog = copy.deepcopy(base)
        prog.__class__ = cls
        v3 = {k: flags.pop(k) for k in V3_KEYS if k in flags}
        joint_kw = {k: flags.pop(k) for k in list(flags) if k not in ("algebra", "joint", "soft", "vote", "rare", "tau")}
        prog._init_v2(flags.get("algebra", False), flags.get("joint", False), flags.get("soft", False), False,
                      flags.get("rare", False), flags.get("tau", 1.0), joint_kw)
        prog._init_v3(**v3)
        prog.post_fit(stories)
        return prog

    # -- learning -------------------------------------------------------------------------------
    def post_fit(self, stories: list[Story]) -> dict[str, Any]:
        self._base_frames = set(self.reader.frame_score)
        if self.use_rare:
            self.rare_report = self._fit_rare(stories)
            self.reader.invalidate()
        if self.use_algebra:
            self.algebra, self.algebra_report = learn_algebra(self)
        learned: dict[str, float] = {}
        if self.use_joint:
            jr = RecallReader(self.lexicon, self.genders, recall=self.use_recall, pron_rank=self.pron_rank,
                              pron_window=self.pron_window, salience=self.use_salience, **self.joint_kw)
            jr.frame_score = dict(self.reader.frame_score)
            jr.flip_penalty = self.reader.flip_penalty
            self.reader = jr
            learned = self._learn_new_frames(stories)
            if self.use_salience:
                self.salience_report = self._learn_roles(stories)
        self.reader.invalidate()
        return {"algebra": self.algebra_report, "rare": self.rare_report, "new_frames": learned,
                "salience": self.salience_report}

    def _learn_roles(self, stories: list[Story], passes: int = 2) -> dict[str, Any]:
        """Counterfactual hard EM for referent roles (training stories and their answers only).

        For each pronoun-supplied fact on the query path, swap in each candidate at the same anchor that
        keeps the fact's word and other person and changes only the pronoun's referent. When the swap
        flips the answer between right and wrong, the role that gives the right answer gains a count and
        the other loses one. role_delta = log((good + 1) / (bad + 1)) per role."""
        jr = self.reader
        history = []
        for _ in range(passes):
            good: Counter = Counter()
            bad: Counter = Counter()
            for s in stories:
                reading = jr.read(s)
                path = story_path(reading, *s.query)
                if not path:
                    continue
                base_ok = self._answer(reading, reading.edges, s) == s.target
                by_anchor: dict[int, list[Candidate]] = defaultdict(list)
                for c in reading.candidates:
                    by_anchor[c.anchor_pos].append(c)
                for f, _ in path:
                    rf = next((x for x in f.frame if x.startswith("@")), None)
                    ef = RecallReader._pron_endpoint(f)
                    if rf is None or ef is None:
                        continue
                    other = f.relative if ef == f.owner else f.owner
                    seen = set()
                    for g in by_anchor[f.anchor_pos]:
                        rg = next((x for x in g.frame if x.startswith("@")), None)
                        eg = RecallReader._pron_endpoint(g)
                        if rg is None or eg is None or eg == ef or g.word != f.word or rg == rf:
                            continue
                        if other not in (g.owner, g.relative) or (eg, rg) in seen:
                            continue
                        # same orientation: the other person keeps its side of the fact
                        if (other == f.relative) != (other == g.relative):
                            continue
                        seen.add((eg, rg))
                        edges = [g if e is f else e for e in reading.edges]
                        ok = self._answer(reading, edges, s) == s.target
                        if ok == base_ok:
                            continue
                        (good if ok else bad)[rg] += 1
                        (bad if ok else good)[rf] += 1
            keys = set(good) | set(bad)
            jr.role_delta = {k: math.log((good[k] + 1.0) / (bad[k] + 1.0)) for k in keys}
            jr.invalidate()
            history.append({"good": dict(good), "bad": dict(bad)})
        return {"role_delta": {k: round(v, 3) for k, v in sorted(jr.role_delta.items())}, "passes": history}

    def _answer(self, reading: Reading, edges: list[Candidate], story: Story) -> str | None:
        a, c = story.query
        p = story_path(Reading(reading.genders, [], edges, reading.entities), a, c)
        if not p:
            return None
        dist, _ = compose_hybrid(self, self.seq(p))
        if not dist:
            return None
        return self.composer.label_for(max(dist.items(), key=lambda kv: kv[1])[0], reading.genders.get(c))

    @staticmethod
    def _new_frame(frame: tuple[str, ...], recall: bool) -> bool:
        """L1b's new frame (pronS) and, with recall, the recall generator's frames (w*, nest)."""
        head = frame[0]
        return head == "pronS" or (recall and (head == "nest" or head.startswith("w")))

    def _learn_new_frames(self, stories: list[Story]) -> dict[str, float]:
        """L1b's one hard-EM frame step for frames L1 never had (two passes when recall adds frames).
        Without recall this is exactly L1b's step, so JSA here equals L1b's algebra + joint + soft."""
        passes = 2 if self.use_recall else 1
        learned: set = set()
        for _ in range(passes):
            good: Counter = Counter()
            bad: Counter = Counter()
            for s in stories:
                reading = self.reader.read(s)
                path = story_path(reading, *s.query)
                if not path:
                    continue
                ok = self.predict(s).label == s.target
                for c, _ in path:
                    if self._new_frame(c.frame, self.use_recall):
                        (good if ok else bad)[c.frame] += 1
            for f in set(good) | set(bad):
                self.reader.frame_score[f] = math.log((good[f] + 1.0) / (bad[f] + 1.0))
                learned.add(f)
            self.reader.invalidate()
        return {"|".join(f): round(self.reader.frame_score[f], 3) for f in sorted(learned)}

    def _fit_rare(self, stories: list[Story]) -> dict[str, Any]:
        if not self.use_mergefix:
            return super()._fit_rare(stories)
        self._ctx = context_vectors(stories)
        comp = self.composer
        cls = comp.classes
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
            b_inv = cls.find((self.lexicon.symbol[target], REV))
            if a == b:
                continue
            left, right = comp.rows()
            agree = disagree = directional = 0
            for rows in (left, right):
                ra, rb, ri = rows.get(a, {}), rows.get(b, {}), rows.get(b_inv, {})
                for ctx in set(ra) & set(rb):
                    if ra[ctx] == rb[ctx]:
                        agree += 1
                        # the context separates the class from its inverse
                        if b_inv == b or ri.get(ctx) != rb[ctx]:
                            directional += 1
                    else:
                        disagree += 1
            llr = prior + _llr(agree, disagree)
            merge = gated_merge(agree, disagree, directional, llr)
            placed[w] = {"target": target, "prior": round(prior, 3), "agree": agree, "disagree": disagree,
                         "directional": directional, "llr": round(llr, 3), "merged": merge}
            if merge:
                cls.union((self.lexicon.symbol[w], FWD), (self.lexicon.symbol[target], FWD))
        comp.rebuild(obs)
        comp.prune()
        self._core = core
        return {"core": sorted(core), "orphans": sorted(orphans), "placed": placed}

    # -- inference ------------------------------------------------------------------------------
    def seq(self, path: list[tuple[Candidate, int]]) -> list[tuple[str, int]]:
        """Composite facts (possessive chains) expand into their two relation steps."""
        out: list[tuple[str, int]] = []
        for c, d in path:
            if COMPOSITE in c.word:
                parts = [self.composer.classes.node(self.lexicon.lookup(w) or w, FWD) for w in c.word.split(COMPOSITE)]
                if d == REV:
                    parts = [self.composer.classes.node(sym, REV) for sym, _ in reversed(parts)]
                out.extend(parts)
            else:
                out.append(self.composer.classes.node(self.lexicon.lookup(c.word) or c.word, d))
        return out

    def _fact_class(self, c: Candidate) -> tuple | None:
        dist, _ = compose_hybrid(self, self.seq([(c, FWD)]))
        return max(dist.items(), key=lambda kv: kv[1])[0] if dist else None

    def inconsistency(self, forest: list[Candidate], facts: list[Candidate], reading: Reading) -> int:
        """Violations of cycle consistency in one reading (see module docstring)."""
        ids = {id(c) for c in forest}
        rd = Reading(reading.genders, [], forest, reading.entities)
        bad = 0
        for f in facts:
            if id(f) in ids or f.owner == f.relative:
                continue
            path = story_path(rd, f.owner, f.relative)
            if not path:
                continue
            dist, _ = compose_hybrid(self, self.seq(path))
            fc = self._fact_class(f)
            if dist and fc is not None and max(dist.items(), key=lambda kv: kv[1])[0] != fc:
                bad += 1
        a, c = self.reader.query if isinstance(self.reader, JointReader) and self.reader.query else (None, None)
        if self.algebra is not None and a is not None and a != c:
            path = story_path(rd, a, c)
            if path:
                classes = [self.composer.classes.find(n) for n in self.seq(path)]
                if self.algebra.is_identity_between(classes):
                    bad += 1
        return bad

    def predict(self, story: Story) -> Prediction:
        if self.use_rare:
            self.lexicon.lookup = self.rare_lookup           # type: ignore[method-assign]
        try:
            return self._predict(story)
        finally:
            if self.use_rare:
                self.lexicon.lookup = self._orig_lookup      # type: ignore[method-assign]

    def _predict(self, story: Story) -> Prediction:
        reading = self.reader.read(story)
        a, c = story.query
        gC = reading.genders.get(c)
        alts = getattr(reading, "alternatives", None)
        if alts:
            readings = alts if self.use_soft else alts[:1]
        else:
            readings = [(0.0, reading.edges, reading.edges)]
        scored = []
        for j, forest, facts in readings:
            if self.use_cycle:
                viol = self.inconsistency(forest, facts, reading)
                self.stats["violations"] += viol
                j = j - self.cycle_lambda * viol
            scored.append((j, forest))
        if not self.use_soft:
            scored = [max(scored, key=lambda t: t[0])]
        top = max(j for j, _ in scored)
        votes: Counter = Counter()
        any_path = False
        undefined = 0
        best_syms: list[str] = []
        best_len = None
        for j, forest in sorted(scored, key=lambda t: -t[0]):
            p = story_path(Reading(reading.genders, [], forest, reading.entities), a, c)
            if not p:
                continue
            any_path = True
            seq = self.seq(p)
            dist, _ = compose_hybrid(self, seq)
            if not dist:
                undefined += 1
                continue
            cls = max(dist.items(), key=lambda kv: kv[1])[0]
            label = self.composer.label_for(cls, gC)
            if label is None:
                continue
            votes[label] += math.exp((j - top) / self.tau)
            if not best_syms:
                best_syms = [f"{s}{'' if d == FWD else '^-1'}" for s, d in seq]
                best_len = len(p)
        if not any_path:
            return Prediction(None, None, "no-path")
        if not votes:
            return Prediction(None, best_len, "undefined-composition" if undefined else "no-realisation", best_syms)
        label = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
        return Prediction(label, best_len, "ok", best_syms)


ARMS: dict[str, dict[str, Any]] = {
    "L1": {},
    "JS": {"joint": True, "soft": True},                         # L1b arm, identity check (v2 code, L1b constants)
    "JSA": {"algebra": True, "joint": True, "soft": True},
    "RC": {"algebra": True, "joint": True, "soft": True, "recall": True},
    "CY": {"algebra": True, "joint": True, "soft": True, "cycle": True},
    "MF": {"algebra": True, "joint": True, "soft": True, "rare": True, "mergefix": True},
    # logged post-dev addition (dev taxonomy: pronoun salience); reported arm only, not part of FULL (it lowered
    # accuracy on the development slice; see findings)
    "SAL": {"algebra": True, "joint": True, "soft": True, "salience": True},
    "FULL": {"algebra": True, "joint": True, "soft": True, "recall": True, "cycle": True, "rare": True, "mergefix": True},
}
V3_KEYS = ("recall", "cycle", "mergefix", "salience", "cycle_lambda", "pron_rank", "pron_window")


def arm_program(arm: str, base: FieldProgram, stories: list[Story], **overrides: Any) -> FieldProgram:
    """The arm's program from a shared L1 fit. L1 is the base; JS is L1b's v2 code (identity check)."""
    cfg = {**ARMS[arm], **overrides}
    if arm == "L1":
        return base
    if arm == "JS":
        return FieldProgramV2.upgrade(base, stories, **{k: v for k, v in cfg.items() if k not in V3_KEYS})
    return FieldProgramV3.upgrade(base, stories, **cfg)


# ---------------------------------------------------------------------------------------------
# diagnoses (post hoc; gold edges are used here only)
# ---------------------------------------------------------------------------------------------


def _gold_class(program: FieldProgram, rel: str) -> tuple:
    return program.composer.classes.find(program.composer.classes.node(program.lexicon.lookup(rel) or rel, FWD))


def _cand_class(program: FieldProgram, c: Candidate, d: int) -> tuple | None:
    if isinstance(program, FieldProgramV3):
        seq = program.seq([(c, d)])
        if len(seq) == 1:
            return program.composer.classes.find(seq[0])
        dist, _ = compose_hybrid(program, seq)
        return max(dist.items(), key=lambda kv: kv[1])[0] if dist else None
    return program.composer.classes.find(program.composer.classes.node(program.lexicon.lookup(c.word) or c.word, d))


def candidate_recall(program: FieldProgram, stories: list[Story]) -> dict[str, Any]:
    """Per k: gold edges whose pair (and fact: pair + relation class in the stated direction) is among the
    reader's candidates, among the per-anchor options joint decoding considers, and in the chosen reading."""
    per_k: dict[int, Counter] = defaultdict(Counter)
    top_k = getattr(program.reader, "top_k", None)
    for s in stories:
        if not s.gold_edges:
            continue
        reading = program.reader.read(s)
        cands = reading.candidates
        opts = cands
        if top_k is not None:
            by_anchor: dict[int, list[Candidate]] = defaultdict(list)
            for c in cands:
                by_anchor[c.anchor_pos].append(c)
            opts = []
            for pos, cs in by_anchor.items():
                seen: set = set()
                for c in sorted(cs, key=lambda c: (-program.reader.score(c), c.owner, c.relative, c.frame)):
                    if (c.owner, c.relative) in seen:
                        continue
                    seen.add((c.owner, c.relative))
                    opts.append(c)
                    if len(seen) >= top_k:
                        break
        chosen = reading.edges
        for x, rel, y in s.gold_edges:
            g = _gold_class(program, rel)
            per_k[s.k]["gold"] += 1
            for name, pool in (("cand", cands), ("opt", opts), ("chosen", chosen)):
                pair = any({c.owner, c.relative} == {x, y} for c in pool)
                fact = any((c.owner == x and c.relative == y and _cand_class(program, c, FWD) == g) or
                           (c.owner == y and c.relative == x and _cand_class(program, c, REV) == g) for c in pool)
                per_k[s.k][f"{name}_pair"] += int(pair)
                per_k[s.k][f"{name}_fact"] += int(fact)
    out = {}
    for k, v in sorted(per_k.items()):
        g = max(1, v["gold"])
        out[str(k)] = {"gold": v["gold"], **{key: round(v[key] / g, 4) for key in
                                            ("cand_pair", "cand_fact", "opt_pair", "opt_fact", "chosen_pair", "chosen_fact")}}
    return out


def pooled_recall(rec: dict[str, Any], ks: tuple[str, ...], key: str = "cand_pair") -> float | None:
    num = sum(rec[k][key] * rec[k]["gold"] for k in ks if k in rec)
    den = sum(rec[k]["gold"] for k in ks if k in rec)
    return num / den if den else None


FUNCTION_WORDS = {"the", "a", "an", "and", "to", "of", "with", "for", "his", "her", "he", "she", "is", "was", "had",
                  "has", "in", "on", "at", "went", "who", "that", "their", "they", "him", "them", "be", "been", "it"}


def missed_taxonomy(program: FieldProgram, stories: list[Story], examples: int = 4) -> dict[str, Any]:
    """Post hoc, dev only: why a gold edge pair is absent from the reader's candidates."""
    counts: Counter = Counter()
    ex: dict[str, list[str]] = defaultdict(list)
    unknown_words: Counter = Counter()
    total = 0
    for s in stories:
        if not s.gold_edges:
            continue
        reading = program.reader.read(s)
        pairs = {frozenset((c.owner, c.relative)) for c in reading.candidates}
        _, resolved = program.reader.estimate_genders(s)
        sents_of: dict[str, set[int]] = defaultdict(set)
        for i, (kind, value) in enumerate(s.tokens):
            if kind == "name":
                sents_of[value].add(s.sent_of[i])
        pron_sents: dict[str, set[int]] = defaultdict(set)
        for i, e in resolved.items():
            pron_sents[e].add(s.sent_of[i])
        by_sent: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for i, tok in enumerate(s.tokens):
            by_sent[s.sent_of[i]].append(tok)
        for x, rel, y in s.gold_edges:
            total += 1
            if frozenset((x, y)) in pairs:
                continue
            sx = sents_of[x] | pron_sents[x]
            sy = sents_of[y] | pron_sents[y]
            co = sx & sy
            if not co:
                # a pronoun of the other person's gold gender sits in a sentence with one of them
                gg = s.gold_genders
                pron_miss = False
                for i, (kind, value) in enumerate(s.tokens):
                    if kind == "word" and value in PRONOUN_GENDER:
                        sent = s.sent_of[i]
                        for here, other in ((x, y), (y, x)):
                            if sent in sents_of[here] and gg.get(other) == PRONOUN_GENDER[value] \
                                    and resolved.get(i) != other:
                                pron_miss = True
                cls = "pronoun-misresolved" if pron_miss else "cross-sentence"
            else:
                toks = [t for sent in sorted(co) for t in by_sent[sent]]
                known = [v for k, v in toks if k == "word" and program.lexicon.lookup(v) is not None]
                if not known:
                    cls = "unknown-word"
                    for k, v in toks:
                        if k == "word" and v not in FUNCTION_WORDS and v not in PRONOUN_GENDER and len(v) > 2:
                            unknown_words[v] += 1
                else:
                    words = [v for _, v in toks]
                    nested = any(words[i] == "'s" and i >= 1 and program.lexicon.lookup(words[i - 1]) is not None
                                 and i + 1 < len(words) and program.lexicon.lookup(words[i + 1]) is not None
                                 for i in range(len(words)))
                    appos = any(toks[i][0] == "name" and i + 2 < len(toks) and toks[i + 1][1] == ","
                                and toks[i + 2][0] == "name" for i in range(len(toks)))
                    cls = "nested-possessive" if nested else ("appositive" if appos else "frame-gap")
            counts[cls] += 1
            if len(ex[cls]) < examples:
                ex[cls].append(f"{s.text} || gold: {y} is {x}'s {rel}")
    missed = sum(counts.values())
    return {"gold_edges": total, "missed_pairs": missed, "missed_rate": round(missed / max(1, total), 4),
            "classes": dict(counts.most_common()), "unknown_words_top": unknown_words.most_common(15),
            "examples": dict(ex)}


def reading_stats(program: FieldProgram) -> dict[str, Any]:
    return dict(getattr(program, "stats", {}))


__all__ = ["ARMS", "FieldProgramV3", "RecallReader", "arm_program", "candidate_recall", "missed_taxonomy",
           "pooled_recall", "V2_ARMS"]
