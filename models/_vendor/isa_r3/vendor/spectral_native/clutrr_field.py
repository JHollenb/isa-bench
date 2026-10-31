"""Field model for CLUTRR (Sinha et al. 2019): read kinship stories as text, answer by exact composition.

The learned program is discrete and its description length is accounted in bits:

  gender axis   entity -> {m, f}: per-story pronoun and kin-word evidence (a recency coset join
                resolves pronouns), a global name table, and suffix-trie inheritance for unseen names
  reader        anchor words with frames: which mention owns the relation and which is the relative;
                candidate facts are chosen as a maximum spanning forest over the mentioned entities
  lexicon       word -> directed relation node; nodes are merged by a sequential probability ratio
                test on their composition rows (synonyms, gender twins and inverses are all merges)
  composition   T[(left class, right class)] -> class, learned by counting; CYK over the story path
  realisation   (class, gender of the answer entity) -> label

Two learning arms share the machinery. The primary arm sees only (story, query, target). The
secondary arm may also use CLUTRR's gold edge labels to align words and frames and to observe
composition on gold symbol sequences. Test input is text only in both arms.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

TOKEN_RE = re.compile(r"\[([^\]]+)\]|['’]s\b|[A-Za-z]+(?:-[A-Za-z]+)*(?:['’](?![sS]\b)[A-Za-z]+)?|\d+|[^\sA-Za-z\d]")
PRONOUN_GENDER = {"he": "m", "him": "m", "his": "m", "himself": "m",
                  "she": "f", "her": "f", "hers": "f", "herself": "f"}
POSSESSIVE = {"his", "her"}
SUBJECT_PRONOUNS = {"he", "she"}
SENTENCE_END = {".", "!", "?"}
HAS_WORDS = {"has", "had", "have", "having"}
IS_WORDS = {"is", "was", "are", "were", "'s", "’s", "being", "been", "be"}
FWD, REV = 0, 1


# ---------------------------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------------------------


@dataclass
class Story:
    sid: str
    text: str
    tokens: list[tuple[str, str]]          # (kind, value): kind in {"name", "word", "punct"}
    sent_of: list[int]                      # sentence index per token
    query: tuple[str, str]
    target: str
    k: int
    gold_edges: list[tuple[str, str, str]] = field(default_factory=list)   # (x, rel, y): y is x's rel
    gold_genders: dict[str, str] = field(default_factory=dict)             # diagnosis only
    task: str = ""


def tokenize(text: str) -> tuple[list[tuple[str, str]], list[int]]:
    tokens: list[tuple[str, str]] = []
    sent_of: list[int] = []
    sent = 0
    for m in TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            tokens.append(("name", m.group(1).strip()))
        else:
            tok = m.group(0)
            if tok in ("'s", "’s"):
                tokens.append(("word", "'s"))
            elif tok[0].isalnum():
                tokens.append(("word", tok.lower().replace("’", "'")))
            else:
                tokens.append(("punct", tok))
        sent_of.append(sent)
        if tokens[-1][0] == "punct" and tokens[-1][1] in SENTENCE_END:
            sent += 1
    return tokens, sent_of


def _gold(row: dict[str, str]) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    genders: dict[str, str] = {}
    order: list[str] = []
    for item in (row.get("genders") or "").split(","):
        if ":" in item:
            name, g = item.rsplit(":", 1)
            order.append(name.strip())
            genders[name.strip()] = "m" if g.strip().startswith("m") else "f"
    edges: list[tuple[str, str, str]] = []
    try:
        pairs = ast.literal_eval(row.get("story_edges") or "[]")
        types = ast.literal_eval(row.get("edge_types") or "[]")
        for (a, b), rel in zip(pairs, types):
            edges.append((order[a], rel, order[b]))
    except (ValueError, SyntaxError, IndexError):
        edges = []
    return edges, genders


def load_split(path: Path) -> list[Story]:
    stories: list[Story] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            tokens, sent_of = tokenize(row["story"])
            query = tuple(ast.literal_eval(row["query"]))
            task = row.get("task_name") or ""
            k = int(task.split(".")[-1]) if task.startswith("task_") and "." in task else 0
            edges, genders = _gold(row)
            stories.append(Story(row["id"], row["story"], tokens, sent_of, (query[0], query[1]),
                                 row["target_text"], k, edges, genders, task))
    return stories


# ---------------------------------------------------------------------------------------------
# gender axis
# ---------------------------------------------------------------------------------------------


@dataclass
class GenderModel:
    """Global name -> gender table, with suffix-trie inheritance for names never seen."""

    names: dict[str, Counter] = field(default_factory=dict)
    suffix: dict[str, Counter] = field(default_factory=dict)
    min_support: int = 4
    min_purity: float = 0.75

    def fit(self, evidence: Iterable[tuple[str, str]]) -> None:
        self.names = defaultdict(Counter)
        for name, g in evidence:
            self.names[name][g] += 1
        self.names = dict(self.names)
        self.suffix = defaultdict(Counter)
        for name, c in self.names.items():
            g = c.most_common(1)[0][0]
            low = name.lower()
            for n in range(1, 4):
                if len(low) > n:
                    self.suffix[low[-n:]][g] += 1
        self.suffix = dict(self.suffix)

    def prior(self, name: str) -> dict[str, float]:
        """Log-odds evidence (m minus f) from the name table, or from its longest reliable suffix family."""
        c = self.names.get(name)
        if c and sum(c.values()) > 0:
            m, f = c.get("m", 0) + 0.5, c.get("f", 0) + 0.5
            return {"m": math.log(m / (m + f)), "f": math.log(f / (m + f)), "source": 2.0}
        low = name.lower()
        for n in (3, 2, 1):
            s = self.suffix.get(low[-n:]) if len(low) > n else None
            if s and sum(s.values()) >= self.min_support:
                g, top = s.most_common(1)[0]
                if top / sum(s.values()) >= self.min_purity:
                    m, f = s.get("m", 0) + 1.0, s.get("f", 0) + 1.0
                    return {"m": math.log(m / (m + f)), "f": math.log(f / (m + f)), "source": 1.0}
        return {"m": math.log(0.5), "f": math.log(0.5), "source": 0.0}

    def bits(self) -> float:
        # each stored name: its characters (5 bits each) plus a gender bit; suffix families are derived
        return sum(5.0 * len(n) + 1.0 for n in self.names)


# ---------------------------------------------------------------------------------------------
# reader
# ---------------------------------------------------------------------------------------------


@dataclass
class Candidate:
    owner: str
    relative: str
    word: str
    frame: tuple[str, ...]
    anchor_pos: int
    locality: float


@dataclass
class Reading:
    genders: dict[str, str]
    candidates: list[Candidate]
    edges: list[Candidate]          # chosen maximum spanning forest
    entities: list[str]


class Lexicon:
    """Anchor words -> symbols. Unknown words in anchor position inherit from the longest known prefix."""

    def __init__(self) -> None:
        self.symbol: dict[str, str] = {}          # word -> symbol (the word itself, or its inherited parent)
        self.gender: dict[str, str] = {}          # symbol -> gender of the relative it names (learned)
        self.inherited: dict[str, str] = {}

    def add(self, word: str) -> None:
        self.symbol.setdefault(word, word)

    def remove(self, word: str) -> None:
        self.symbol.pop(word, None)

    def lookup(self, word: str) -> str | None:
        if word in self.symbol:
            return self.symbol[word]
        if word in self.inherited:
            return self.inherited[word]
        # morphological inheritance: a known word plus an inflection suffix (plural 's'/'es')
        best = None
        for known in self.symbol:
            if len(known) >= 3 and word.startswith(known) and word[len(known):] in ("s", "es"):
                if best is None or len(known) > len(best):
                    best = known
        if best is not None:
            self.inherited[word] = self.symbol[best]
            return self.symbol[best]
        return None

    def bits(self, n_classes: int) -> float:
        return sum(5.0 * len(w) + math.log2(max(2, n_classes)) + 1.0 for w in self.symbol)


FILLER_AFTER = {",", "named", "called", "name", "namely", "is", ":", "-", "who", "was", "the", "a", "an",
                "by", "of", "whose"}


class Reader:
    """Candidate relation facts from anchor frames, pronoun joins and a spanning-forest decision."""

    def __init__(self, lexicon: Lexicon, genders: GenderModel) -> None:
        self.lexicon = lexicon
        self.gender_model = genders
        self.frame_score: dict[tuple[str, ...], float] = {}
        self.flip_penalty = 1.0
        self.version = 0
        self._cache: dict[str, tuple[int, "Reading"]] = {}
        self._genders_cache: dict[str, str] = {}

    def invalidate(self) -> None:
        self.version += 1
        self._cache.clear()

    # -- genders and pronouns -------------------------------------------------------------------
    def _ranked(self, story: Story, pos: int, g: str, genders: dict[str, str]) -> list[str]:
        """Recency coset join: earlier name mentions in the pronoun's gender ball, most recent first."""
        sent = story.sent_of[pos]
        ranked: list[str] = []
        for i in range(pos - 1, -1, -1):
            if story.sent_of[i] < sent - 1:
                break
            kind, value = story.tokens[i]
            if kind != "name":
                continue
            # the coset is "not known to be the other gender", taken in recency order
            if genders.get(value) in (g, None) and value not in ranked:
                ranked.append(value)
        return ranked

    def _resolve(self, story: Story, pos: int, g: str, genders: dict[str, str]) -> str | None:
        ranked = self._ranked(story, pos, g, genders)
        return ranked[0] if ranked else None

    def estimate_genders(self, story: Story, rounds: int = 3,
                         seed: dict[str, str] | None = None) -> tuple[dict[str, str], dict[int, str]]:
        entities = [v for kind, v in story.tokens if kind == "name"]
        base: dict[str, dict[str, float]] = {}
        for e in set(entities):
            p = self.gender_model.prior(e)
            w = 2.0 if p["source"] >= 2 else (1.0 if p["source"] >= 1 else 0.0)
            base[e] = {"m": w * (p["m"] - math.log(0.5)), "f": w * (p["f"] - math.log(0.5))}
        for e, g in (seed or {}).items():
            if e in base:
                base[e][g] += 1.5
        # kin-word evidence: a name right after an anchor word is that word's relative
        for i, (kind, value) in enumerate(story.tokens):
            if kind != "word":
                continue
            sym = self.lexicon.lookup(value)
            if sym is None or sym not in self.lexicon.gender:
                continue
            for j in range(i + 1, min(i + 4, len(story.tokens))):
                kj, vj = story.tokens[j]
                if kj == "name":
                    base[vj][self.lexicon.gender[sym]] += 1.5
                    break
                if kj == "word" and vj not in FILLER_AFTER:
                    break
        genders = {e: ("m" if s["m"] > s["f"] else "f") if s["m"] != s["f"] else None
                   for e, s in base.items()}
        resolved: dict[int, str] = {}
        for _ in range(rounds):
            score = {e: dict(s) for e, s in base.items()}
            resolved = {}
            for i, (kind, value) in enumerate(story.tokens):
                if kind == "word" and value in PRONOUN_GENDER:
                    g = PRONOUN_GENDER[value]
                    e = self._resolve(story, i, g, genders)
                    if e is not None:
                        resolved[i] = e
                        score[e][g] += 1.0
            genders = {e: ("m" if s["m"] > s["f"] else "f") if s["m"] != s["f"] else None
                       for e, s in score.items()}
        return {e: g for e, g in genders.items() if g is not None}, resolved

    # -- candidates -----------------------------------------------------------------------------
    def candidates(self, story: Story, resolved: dict[int, str]) -> list[Candidate]:
        toks = story.tokens
        n = len(toks)
        mention_at: dict[int, tuple[str, str]] = {}   # pos -> (entity, kind)
        for i, (kind, value) in enumerate(toks):
            if kind == "name":
                mention_at[i] = (value, "name")
            elif i in resolved:
                mention_at[i] = (resolved[i], "poss" if value in POSSESSIVE else "pron")
        alt: dict[int, str] = {}
        for i, (kind, value) in enumerate(toks):
            if kind == "word" and value in POSSESSIVE and i in resolved:
                ranked = self._ranked(story, i, PRONOUN_GENDER[value], self._genders_cache)
                if len(ranked) > 1:
                    alt[i] = ranked[1]
        anchors = []
        for i, (kind, value) in enumerate(toks):
            if kind == "word":
                sym = self.lexicon.lookup(value)
                if sym is not None:
                    anchors.append((i, value, sym))
        anchor_pos = {a for a, _, _ in anchors}
        out: list[Candidate] = []
        last_owner_by_sent: dict[int, tuple[str, int]] = {}
        for a, word, sym in anchors:
            sent = story.sent_of[a]
            owners: list[tuple[str, str, int]] = []
            # possessive name: X 's (mod){0,3} W
            for p in range(a - 1, max(-1, a - 6), -1):
                if story.sent_of[p] != sent or p in anchor_pos:
                    break
                if p in mention_at and mention_at[p][1] == "name" and p + 1 < n and toks[p + 1][1] == "'s":
                    owners.append((mention_at[p][0], "poss", p))
                    break
                if p in mention_at and mention_at[p][1] == "poss":
                    owners.append((mention_at[p][0], "pron", p))
                    if p in alt:
                        owners.append((alt[p], "pron2", p))
                    break
                if p in mention_at:
                    break
            # of-owner: W (tokens with 'of') X
            for p in range(a + 1, min(n, a + 7)):
                if story.sent_of[p] != sent or p in anchor_pos:
                    break
                if p in mention_at:
                    between = {toks[q][1] for q in range(a + 1, p)}
                    if "of" in between or "to" in between:
                        owners.append((mention_at[p][0], "of", p))
                    break
            # has-owner: X has (a|two|...) W
            for p in range(a - 1, max(-1, a - 8), -1):
                if story.sent_of[p] != sent:
                    break
                if p in mention_at and mention_at[p][1] in ("name", "pron"):
                    between = {toks[q][1] for q in range(p + 1, a)}
                    if between & HAS_WORDS:
                        owners.append((mention_at[p][0], "has", p))
                    break
            # coordination: previous anchor's owner carries over ("X's son Y and husband Z")
            if sent in last_owner_by_sent and not owners:
                o, op = last_owner_by_sent[sent]
                owners.append((o, "coord", op))
            # nearest earlier mention in the sentence (weak)
            for p in range(a - 1, -1, -1):
                if story.sent_of[p] != sent:
                    break
                if p in mention_at:
                    if all(o[0] != mention_at[p][0] for o in owners):
                        owners.append((mention_at[p][0], "near", p))
                    break
            relatives: list[tuple[str, str, int]] = []
            # after: W (filler){0,3} Y  and plural lists "W, Y and Z"
            fillers = 0
            p = a + 1
            while p < n and story.sent_of[p] == sent and fillers <= 3:
                if p in mention_at and mention_at[p][1] == "name":
                    relatives.append((mention_at[p][0], "after", p))
                    q = p + 1
                    while q < n and story.sent_of[q] == sent:
                        if toks[q][1] in (",", "and") or (q in mention_at and mention_at[q][1] == "name"):
                            if q in mention_at and mention_at[q][1] == "name":
                                relatives.append((mention_at[q][0], "list", q))
                            q += 1
                            continue
                        break
                    break
                if p in anchor_pos or p in mention_at:
                    break
                fillers += 1
                p += 1
            # subject-is: Y is (X's|the ...) W
            for p in range(a - 1, max(-1, a - 10), -1):
                if story.sent_of[p] != sent:
                    break
                if p in mention_at and mention_at[p][1] in ("name", "pron") and p + 1 < n \
                        and toks[p + 1][1] in IS_WORDS and toks[p + 1][1] != "'s":
                    relatives.append((mention_at[p][0], "is", p))
                    break
            # deferred: no relative distinct from the owners -> first name of the next sentence
            owner_names = {o[0] for o in owners}
            if not [r for r in relatives if r[0] not in owner_names]:
                for p in range(a + 1, n):
                    if story.sent_of[p] > sent + 1:
                        break
                    if p in mention_at and mention_at[p][1] == "name" and story.sent_of[p] == sent + 1:
                        relatives.append((mention_at[p][0], "next", p))
                        break
            # nearest later / earlier names (weak): up to two each side, so an appositive
            # ("Y, X's W") proposes Y past the owner X
            for step in (1, -1):
                p = a + step
                found = 0
                while 0 <= p < n and story.sent_of[p] == sent and found < 2:
                    if p in mention_at and mention_at[p][1] == "name":
                        found += 1
                        if all(r[0] != mention_at[p][0] for r in relatives):
                            relatives.append((mention_at[p][0], "nafter" if step > 0 else "nbefore", p))
                    p += step
            for o, ocue, op in owners:
                for r, rcue, rp in relatives:
                    if o == r:
                        continue
                    loc = abs(op - a) + abs(rp - a)
                    out.append(Candidate(o, r, word, (ocue, rcue), a, loc))
                    out.append(Candidate(r, o, word, (ocue, rcue, "flip"), a, loc))
            primary = [o for o in owners if o[1] in ("poss", "pron", "of", "has")]
            if primary:
                last_owner_by_sent[sent] = (primary[0][0], primary[0][2])
        return out

    # Generic syntactic prior (not kinship-specific): an explicit marker (possessive, 'of', 'has',
    # a copula before the relative) outranks bare adjacency. Learned log-odds are added on top.
    OWNER_CUE = {"poss": 2.0, "pron": 2.0, "of": 2.0, "has": 1.5, "coord": 1.0, "pron2": 0.5, "near": 0.0}
    RELATIVE_CUE = {"after": 1.0, "list": 1.0, "is": 1.0, "next": 0.5, "nafter": 0.0, "nbefore": 0.0}

    def prior(self, frame: tuple[str, ...]) -> float:
        return self.OWNER_CUE.get(frame[0], 0.0) + self.RELATIVE_CUE.get(frame[1], 0.0)

    def score(self, c: Candidate) -> float:
        base = self.prior(c.frame) + self.frame_score.get(c.frame, 0.0)
        flip = -self.flip_penalty if c.frame[-1] == "flip" else 0.0
        return base + flip - 0.02 * c.locality

    def read(self, story: Story) -> Reading:
        hit = self._cache.get(story.sid)
        if hit is not None and hit[0] == self.version:
            return hit[1]
        reading = self._read(story)
        self._cache[story.sid] = (self.version, reading)
        return reading

    def _read(self, story: Story) -> Reading:
        genders, resolved = self.estimate_genders(story)
        reading = self._forest(story, genders, resolved)
        # fixed point: each chosen fact's kin word genders its relative; re-read once with that evidence
        seed = {c.relative: self.lexicon.gender[self.lexicon.lookup(c.word)]
                for c in reading.edges
                if self.lexicon.lookup(c.word) in self.lexicon.gender}
        if any(genders.get(e) != g for e, g in seed.items()):
            genders, resolved = self.estimate_genders(story, seed=seed)
            reading = self._forest(story, genders, resolved)
        return reading

    def _forest(self, story: Story, genders: dict[str, str], resolved: dict[int, str]) -> Reading:
        self._genders_cache = genders
        cands = self.candidates(story, resolved)
        entities = sorted({v for kind, v in story.tokens if kind == "name"})
        # maximum spanning forest (Kruskal): every mentioned person connected by the fewest facts
        parent = {e: e for e in entities}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        chosen: list[Candidate] = []
        for c in sorted(cands, key=lambda c: (-self.score(c), c.anchor_pos, c.owner, c.relative)):
            if c.owner not in parent or c.relative not in parent:
                continue
            ra, rb = find(c.owner), find(c.relative)
            if ra != rb:
                parent[ra] = rb
                chosen.append(c)
        return Reading(genders, cands, chosen, entities)


def story_path(reading: Reading, a: str, c: str) -> list[tuple[Candidate, int]] | None:
    """Edges on the forest path from a to c, each with its traversal direction (FWD owner->relative)."""
    adj: dict[str, list[tuple[str, Candidate, int]]] = defaultdict(list)
    for e in reading.edges:
        adj[e.owner].append((e.relative, e, FWD))
        adj[e.relative].append((e.owner, e, REV))
    if a not in adj or c not in adj:
        return None
    prev: dict[str, tuple[str, Candidate, int] | None] = {a: None}
    frontier = [a]
    while frontier:
        nxt = []
        for x in frontier:
            for y, e, d in adj[x]:
                if y not in prev:
                    prev[y] = (x, e, d)
                    nxt.append(y)
        frontier = nxt
    if c not in prev:
        return None
    out = []
    cur = c
    while prev[cur] is not None:
        x, e, d = prev[cur]
        out.append((e, d))
        cur = x
    return out[::-1]


# ---------------------------------------------------------------------------------------------
# relation classes and composition
# ---------------------------------------------------------------------------------------------


class Classes:
    """Union-find over directed relation nodes (symbol, direction)."""

    def __init__(self) -> None:
        self.parent: dict[tuple[str, int], tuple[str, int]] = {}

    def node(self, sym: str, d: int) -> tuple[str, int]:
        key = (sym, d)
        self.parent.setdefault(key, key)
        return key

    def find(self, x: tuple[str, int]) -> tuple[str, int]:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, a: tuple[str, int], b: tuple[str, int]) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        # deterministic representative: the lexicographically smaller node
        if rb < ra:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True

    def union(self, a: tuple[str, int], b: tuple[str, int]) -> None:
        """Merge two relations and, by the inverse law (r = s implies r^-1 = s^-1), their inverses."""
        pending = [(a, b)]
        while pending:
            x, y = pending.pop()
            if self._union(x, y):
                # every member of the new class forces its inverse into the inverse of any other member
                members = [n for n in list(self.parent) if self.find(n) == self.find(x)]
                inv = [(m[0], 1 - m[1]) for m in members]
                for i in range(1, len(inv)):
                    self.parent.setdefault(inv[0], inv[0])
                    self.parent.setdefault(inv[i], inv[i])
                    if self.find(inv[0]) != self.find(inv[i]):
                        pending.append((inv[0], inv[i]))

    def classes(self) -> set[tuple[str, int]]:
        return {self.find(x) for x in list(self.parent)}


@dataclass
class Obs:
    left: tuple[str, int] | tuple            # node, or ("#class", class) for an inferred intermediate
    right: tuple[str, int] | tuple
    label: str
    gender: str | None
    weight: float = 1.0


def _llr(agree: int, disagree: int, eps: float = 0.1, pi: float = 0.25) -> float:
    return agree * math.log((1 - eps) / pi) + disagree * math.log(eps / (1 - pi))


class Composer:
    """Composition table over relation classes, learned by counting; realisation by answer gender."""

    def __init__(self) -> None:
        self.classes = Classes()
        self.table: dict[tuple, Counter] = {}
        self.realise: dict[tuple, Counter] = {}
        self.label_twin: dict[str, str] = {}
        self.min_count = 1

    # labels are fwd nodes of their own symbol (lexical identity with the story words)
    def label_node(self, label: str) -> tuple[str, int]:
        return self.classes.node(label, FWD)

    def cls(self, node) -> tuple:
        if isinstance(node, tuple) and node and node[0] == "#class":
            return self.classes.find(node[1])
        return self.classes.find(node)

    def rebuild(self, obs: list[Obs]) -> None:
        table: dict[tuple, Counter] = defaultdict(Counter)
        realise: dict[tuple, Counter] = defaultdict(Counter)
        for o in obs:
            out = self.classes.find(self.label_node(o.label))
            table[(self.cls(o.left), self.cls(o.right))][out] += o.weight
            if o.gender is not None:
                realise[(out, o.gender)][o.label] += o.weight
        self.table = dict(table)
        self.realise = dict(realise)

    def predict_pair(self, left, right) -> Counter | None:
        return self.table.get((self.cls(left), self.cls(right)))

    def fold(self, seq: list[tuple], left: bool = True) -> dict[tuple, float]:
        """Post hoc variant: a single left (or right) fold instead of CYK over all bracketings."""
        classes = [self.cls(n) for n in seq]
        if not left:
            classes = classes[::-1]
        acc, lp = classes[0], 0.0
        for c in classes[1:]:
            row = self.table.get((acc, c) if left else (c, acc))
            if not row:
                return {}
            out, cnt = row.most_common(1)[0]
            lp += math.log((cnt + 0.5) / (sum(row.values()) + 1.0))
            acc = out
        return {acc: lp}

    def compose(self, seq: list[tuple]) -> dict[tuple, float]:
        """CYK over the path: every bracketing whose binary steps are all defined; log-probabilities."""
        k = len(seq)
        if k == 0:
            return {}
        span: dict[tuple[int, int], dict[tuple, float]] = {}
        for i, node in enumerate(seq):
            span[(i, i + 1)] = {self.cls(node): 0.0}
        for length in range(2, k + 1):
            for i in range(0, k - length + 1):
                j = i + length
                cell: dict[tuple, float] = {}
                for m in range(i + 1, j):
                    for lc, lp in span[(i, m)].items():
                        for rc, rp in span[(m, j)].items():
                            row = self.table.get((lc, rc))
                            if not row:
                                continue
                            total = sum(row.values())
                            if total < self.min_count:
                                continue
                            out, cnt = row.most_common(1)[0]
                            p = lp + rp + math.log((cnt + 0.5) / (total + 1.0))
                            if p > cell.get(out, -1e18):
                                cell[out] = p
                # keep the best few per span (the answer set is small)
                span[(i, j)] = dict(sorted(cell.items(), key=lambda kv: -kv[1])[:6])
        return span[(0, k)]

    def preimages(self, max_per_class: int = 2) -> dict[tuple, list[tuple[tuple, tuple]]]:
        """Composite classes and the table pairs that define them (c = a . b), best supported first."""
        pre: dict[tuple, list[tuple[float, tuple, tuple]]] = defaultdict(list)
        for (a, b), row in self.table.items():
            out, cnt = row.most_common(1)[0]
            if out not in (a, b):
                pre[out].append((cnt, a, b))
        return {c: [(a, b) for _, a, b in sorted(v, key=lambda t: (-t[0], t[1], t[2]))[:max_per_class]]
                for c, v in pre.items()}

    mode: str = "cyk"   # post hoc variants: "left", "right"

    def _compose_mode(self, seq: list[tuple]) -> dict[tuple, float]:
        if self.mode == "cyk":
            return self.compose(seq)
        d = self.fold(seq, left=self.mode == "left")
        return d or self.compose(seq)

    def compose_rewrite(self, seq: list[tuple], max_expand: int = 2) -> tuple[dict[tuple, float], int]:
        """CYK; if undefined, rewrite up to max_expand composite symbols into their defining pairs."""
        dist = self._compose_mode(seq)
        if dist:
            return dist, 0
        pre = self.preimages()
        classes = [self.cls(n) for n in seq]
        slots = [i for i, c in enumerate(classes) if c in pre]
        frontier = [((), 0.0)]
        for depth in range(1, max_expand + 1):
            best: dict[tuple, float] = {}
            new_frontier = []
            for chosen, _ in frontier:
                start = chosen[-1][0] + 1 if chosen else 0
                for i in slots:
                    if i < start:
                        continue
                    for pair in pre[classes[i]]:
                        new = chosen + ((i, pair),)
                        new_frontier.append((new, 0.0))
                        expanded: list = []
                        rep = dict(new)
                        for j, c in enumerate(classes):
                            expanded.extend(("#class", x) for x in rep[j]) if j in rep else expanded.append(("#class", c))
                        d = self._compose_mode(expanded)
                        for out, lp in d.items():
                            lp -= 1.0 * depth          # each rewrite is a hypothesis: small cost
                            if lp > best.get(out, -1e18):
                                best[out] = lp
            if best:
                return best, depth
            frontier = new_frontier
        return {}, max_expand

    def label_for(self, cls: tuple, gender: str | None) -> str | None:
        for g in ([gender] if gender else []) + ["m", "f"]:
            row = self.realise.get((cls, g))
            if row:
                return row.most_common(1)[0][0]
        # a class whose members include a label never realised: use the representative symbol if it is a label
        return None

    # -- merges ---------------------------------------------------------------------------------
    def merge_label_twins(self, obs: list[Obs], labels: Iterable[str], min_keys: int = 3) -> list[tuple[str, str]]:
        """Labels that the same key produces for male vs female answers are one relation, split by gender."""
        per_key: dict[tuple, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
        label_gender: dict[str, Counter] = defaultdict(Counter)
        for o in obs:
            if o.gender is None:
                continue
            per_key[(self.cls(o.left), self.cls(o.right))][o.label][o.gender] += o.weight
            label_gender[o.label][o.gender] += o.weight
        lg = {l: c.most_common(1)[0][0] for l, c in label_gender.items() if c}
        pair_count: Counter = Counter()
        for key, by_label in per_key.items():
            males = [l for l, c in by_label.items() if lg.get(l) == "m" and c.get("m", 0) >= 1]
            females = [l for l, c in by_label.items() if lg.get(l) == "f" and c.get("f", 0) >= 1]
            for lm in males:
                for lf in females:
                    pair_count[(lm, lf)] += 1
        merged: list[tuple[str, str]] = []
        used: set[str] = set()
        for (lm, lf), cnt in pair_count.most_common():
            if cnt < min_keys or lm in used or lf in used:
                continue
            if self.classes.find(self.classes.node(lm, FWD)) == self.classes.find(self.classes.node(lf, FWD)):
                used.update((lm, lf))
                continue
            # a twin pair must dominate: each label's best partner is the other
            best_m = max((c for (a, b), c in pair_count.items() if a == lm), default=0)
            best_f = max((c for (a, b), c in pair_count.items() if b == lf), default=0)
            if cnt < best_m or cnt < best_f:
                continue
            used.update((lm, lf))
            merged.append((lm, lf))
            for d in (FWD, REV):
                self.classes.union(self.classes.node(lm, d), self.classes.node(lf, d))
            self.label_twin[lm], self.label_twin[lf] = lf, lm
        return merged

    def rows(self) -> tuple[dict[tuple, dict[tuple, tuple]], dict[tuple, dict[tuple, tuple]]]:
        left: dict[tuple, dict[tuple, tuple]] = defaultdict(dict)
        right: dict[tuple, dict[tuple, tuple]] = defaultdict(dict)
        for (a, b), row in self.table.items():
            out = row.most_common(1)[0][0]
            left[a][b] = out
            right[b][a] = out
        return left, right

    def merge_by_rows(self, obs: list[Obs], threshold: float = math.log(100.0), max_merges: int = 200,
                      protect: set[tuple] | None = None) -> list[tuple[tuple, tuple, float]]:
        """Greedy SPRT merges: classes whose composition rows agree on shared contexts are one relation."""
        merges = []
        for _ in range(max_merges):
            self.rebuild(obs)
            left, right = self.rows()
            # inverted index: classes that share a context are the only candidate pairs
            stats: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])
            for rows in (left, right):
                by_ctx: dict[tuple, list[tuple[tuple, tuple]]] = defaultdict(list)
                for x, row in rows.items():
                    for ctx, out in row.items():
                        by_ctx[ctx].append((x, out))
                for lst in by_ctx.values():
                    for i in range(len(lst)):
                        x, ox = lst[i]
                        for y, oy in lst[i + 1:]:
                            key = (x, y) if x < y else (y, x)
                            stats[key][0 if ox == oy else 1] += 1
            best = None
            # multiple comparisons: naming one merge among the candidate pairs costs ln(pairs) nats
            bar = threshold + math.log(max(1, len(stats)))
            for (x, y), (agree, disagree) in stats.items():
                llr = _llr(agree, disagree)
                if llr >= bar and (best is None or llr > best[2] or (llr == best[2] and (x, y) < best[:2])):
                    best = (x, y, llr)
            if best is None:
                break
            self.classes.union(best[0], best[1])
            merges.append(best)
        self.rebuild(obs)
        return merges

    def merge_gender_twins(self, obs: list[Obs], class_gender: dict[tuple, str], threshold: float = math.log(100.0),
                           max_merges: int = 20) -> list[tuple[tuple, tuple, float]]:
        """Joint hypothesis: x (male relative) and y (female relative) are one relation, and their outputs
        differ only by gender through a consistent bijection (son-in-law <-> daughter-in-law). One SPRT over
        shared contexts: agreement = equal output or a consistent pairing; disagreement = a broken pairing."""
        merges = []
        for _ in range(max_merges):
            self.rebuild(obs)
            left, right = self.rows()
            # pure-gender input classes: every gendered member word names a relative of one gender
            member_genders: dict[tuple, set[str]] = defaultdict(set)
            for c, g in class_gender.items():
                member_genders[self.classes.find(c)].add(g)
            males = [c for c, gs in member_genders.items() if gs == {"m"}]
            females = [c for c, gs in member_genders.items() if gs == {"f"}]
            # pure-gender output classes, from the answers they were realised for
            out_gender: dict[tuple, str] = {}
            weight: dict[tuple, Counter] = defaultdict(Counter)
            for (c, g), row in self.realise.items():
                weight[c][g] += sum(row.values())
            for c, w in weight.items():
                g, n = w.most_common(1)[0]
                if n >= 0.9 * sum(w.values()):
                    out_gender[c] = g
            best = None
            n_pairs = max(1, len(males) * len(females))
            for x in males:
                for y in females:
                    if x == y:
                        continue
                    pairing: dict[tuple, tuple] = {}
                    back: dict[tuple, tuple] = {}
                    agree = disagree = 0
                    for rows in (left, right):
                        rx, ry = rows.get(x, {}), rows.get(y, {})
                        for ctx in set(rx) & set(ry):
                            ox, oy = rx[ctx], ry[ctx]
                            if ox == oy:
                                agree += 1
                            elif out_gender.get(ox) == "m" and out_gender.get(oy) == "f" \
                                    and pairing.get(ox, oy) == oy and back.get(oy, ox) == ox:
                                pairing[ox], back[oy] = oy, ox
                                agree += 1
                            else:
                                disagree += 1
                    llr = _llr(agree, disagree)
                    if llr >= threshold + math.log(n_pairs) and (best is None or llr > best[2]):
                        best = (x, y, llr, dict(pairing))
            if best is None:
                break
            x, y, llr, pairing = best
            self.classes.union(x, y)
            for ox, oy in pairing.items():
                self.classes.union(ox, oy)
            merges.append((x, y, llr))
        self.rebuild(obs)
        return merges

    def prune(self, min_count: float = 2.0) -> None:
        """MDL: keep an entry only when its observations pay for it."""
        n_cls = max(2, len(self.classes.classes()))
        entry_bits = 3.0 * math.log2(n_cls)
        kept = {}
        for key, row in self.table.items():
            total = sum(row.values())
            out, cnt = row.most_common(1)[0]
            # bits saved over a uniform guess among classes, minus exceptions stored
            saved = cnt * math.log2(n_cls) - (total - cnt) * math.log2(n_cls)
            if total >= min_count and saved > entry_bits:
                kept[key] = row
        self.table = kept

    def bits(self) -> float:
        n_cls = max(2, len(self.classes.classes()))
        t = len(self.table) * 3.0 * math.log2(n_cls)
        r = len(self.realise) * (math.log2(n_cls) + 1.0 + math.log2(24))
        return t + r


# ---------------------------------------------------------------------------------------------
# the field program: learning and inference
# ---------------------------------------------------------------------------------------------


@dataclass
class Prediction:
    label: str | None
    path_len: int | None
    reason: str
    symbols: list[str] = field(default_factory=list)


class FieldProgram:
    def __init__(self) -> None:
        self.lexicon = Lexicon()
        self.genders = GenderModel()
        self.reader = Reader(self.lexicon, self.genders)
        self.composer = Composer()
        self.labels: list[str] = []
        self.log: list[dict[str, Any]] = []

    # -- inference ------------------------------------------------------------------------------
    def seq(self, path: list[tuple[Candidate, int]]) -> list[tuple[str, int]]:
        return [self.composer.classes.node(self.lexicon.lookup(c.word) or c.word, d) for c, d in path]

    def predict(self, story: Story) -> Prediction:
        reading = self.reader.read(story)
        a, c = story.query
        path = story_path(reading, a, c)
        if not path:
            return Prediction(None, None, "no-path")
        seq = self.seq(path)
        dist, rewrites = self.composer.compose_rewrite(seq)
        syms = [f"{s}{'' if d == FWD else '^-1'}" for s, d in seq]
        if not dist:
            return Prediction(None, len(path), "undefined-composition", syms)
        if rewrites:
            syms.append(f"rewrites={rewrites}")
        cls = max(dist.items(), key=lambda kv: kv[1])[0]
        label = self.composer.label_for(cls, reading.genders.get(c))
        if label is None:
            return Prediction(None, len(path), "no-realisation", syms)
        return Prediction(label, len(path), "ok", syms)

    # -- learning -------------------------------------------------------------------------------
    def _gender_evidence(self, stories: list[Story]) -> list[tuple[str, str]]:
        ev = []
        for s in stories:
            g, _ = self.reader.estimate_genders(s)
            ev.extend(g.items())
        return ev

    def _refit_genders(self, stories: list[Story]) -> None:
        self.genders.fit(self._gender_evidence(stories))
        self.reader.invalidate()

    def _word_genders(self, stories: list[Story]) -> None:
        counts: dict[str, Counter] = defaultdict(Counter)
        for s in stories:
            g, resolved = self.reader.estimate_genders(s)
            for c in self.reader.candidates(s, resolved):
                if c.frame[-1] != "flip" and c.frame[1] in ("after", "list"):
                    sym = self.lexicon.lookup(c.word)
                    if sym and c.relative in g:
                        counts[sym][g[c.relative]] += 1
        self.lexicon.gender = {}
        for sym, cnt in counts.items():
            total = sum(cnt.values())
            top, n = cnt.most_common(1)[0]
            if total >= 5 and n / total >= 0.85:
                self.lexicon.gender[sym] = top
        self.reader.invalidate()

    def _observations(self, stories: list[Story], gold_sequences: bool = False) -> tuple[list[Obs], dict[str, Any]]:
        """Hard-EM E-step: decode each training story and emit composition observations."""
        obs: list[Obs] = []
        frame_good: Counter = Counter()
        frame_bad: Counter = Counter()
        stats = Counter()
        for s in stories:
            reading = self.reader.read(s)
            a, c = s.query
            gC = reading.genders.get(c)
            seqs = []
            path = story_path(reading, a, c)
            if path:
                seqs.append(self.seq(path))
            else:
                stats["no_path"] += 1
            if gold_sequences and s.gold_edges and self.composer.label_twin:
                # gold relation words are gendered; once twins exist they share gender-free classes
                seqs.append([self.composer.classes.node(rel, FWD) for _, rel, _ in s.gold_edges])
                stats["gold_seq"] += 1
            for seq in seqs:
                if len(seq) == 2:
                    obs.append(Obs(seq[0], seq[1], s.target, gC))
                elif len(seq) == 3:
                    left = self.composer.predict_pair(seq[0], seq[1])
                    right = self.composer.predict_pair(seq[1], seq[2])
                    opts = []
                    if left:
                        opts.append(Obs(("#class", left.most_common(1)[0][0]), seq[2], s.target, gC))
                    if right:
                        opts.append(Obs(seq[0], ("#class", right.most_common(1)[0][0]), s.target, gC))
                    for o in opts:
                        o.weight = 1.0 / len(opts)
                    obs.extend(opts)
                stats[f"len{len(seq)}"] += 1
        return obs, dict(stats)

    def _frame_update(self, stories: list[Story], gold: bool = False) -> dict[str, Any]:
        """Frame reliabilities: frames on paths of correctly answered stories (or gold-matching edges)."""
        good: Counter = Counter()
        bad: Counter = Counter()
        for s in stories:
            reading = self.reader.read(s)
            if gold and s.gold_edges:
                gold_pairs = {(x, y) for x, _, y in s.gold_edges} | {(y, x) for x, _, y in s.gold_edges}
                for c in reading.candidates:
                    (good if (c.owner, c.relative) in gold_pairs else bad)[c.frame] += 1
                continue
            path = story_path(reading, *s.query)
            if not path:
                continue
            p = self.predict(s)
            for c, _ in path:
                (good if p.label == s.target else bad)[c.frame] += 1
        frames = set(good) | set(bad)
        self.reader.frame_score = {f: math.log((good[f] + 1.0) / (bad[f] + 1.0)) for f in frames}
        self.reader.invalidate()
        return {"frames": len(frames)}

    def discover_anchors(self, stories: list[Story], min_count: int = 3) -> list[str]:
        """Words in anchor position (owner cue before, a name right after) become candidate anchors."""
        counts: Counter = Counter()
        total: Counter = Counter()
        for s in stories:
            toks = s.tokens
            for i, (kind, value) in enumerate(toks):
                if kind == "word":
                    total[value] += 1
                if kind != "word" or value in PRONOUN_GENDER or value in FILLER_AFTER or len(value) < 3:
                    continue
                before = toks[i - 1][1] if i >= 1 else ""
                before2 = toks[i - 2][1] if i >= 2 else ""
                owner_cue = before in ("'s", "his", "her", "their") or before2 in ("'s", "his", "her", "their")
                # the head noun right before the name: only ',', 'named', 'called' may intervene
                j = i + 1
                while j < len(toks) and toks[j][1] in (",", "named", "called"):
                    j += 1
                after_name = j < len(toks) and toks[j][0] == "name"
                if owner_cue and after_name:
                    counts[value] += 1
        # position specificity: a relation word sits in anchor position in a large share of its uses;
        # function words ('and', 'with') occur everywhere
        new = [w for w, c in counts.items()
               if c >= min_count and c / total[w] >= 0.2 and self.lexicon.lookup(w) is None]
        for w in new:
            self.lexicon.add(w)
        self.reader.invalidate()
        return new

    def fit(self, stories: list[Story], arm: str = "primary", rounds: int = 4) -> dict[str, Any]:
        gold = arm == "secondary"
        self.labels = sorted({s.target for s in stories})
        for lab in self.labels:
            self.lexicon.add(lab)
        if gold:
            for s in stories:
                for _, rel, _ in s.gold_edges:
                    self.lexicon.add(rel)
        # gender axis: pronoun evidence first, then kin words, then the global name table
        self._refit_genders(stories)
        self._word_genders(stories)
        self._refit_genders(stories)
        discovered = self.discover_anchors(stories)
        self._word_genders(stories)
        history = []
        obs: list[Obs] = []
        for r in range(rounds):
            if gold and r == 0:
                self._frame_update(stories, gold=True)
            twins: list[tuple[str, str]] = []
            merges: list = []
            for _ in range(3):
                # k=3 stories use the current pair entries for their latent intermediate
                obs, stats = self._observations(stories, gold_sequences=gold)
                self.composer.rebuild(obs)
                # twins and row merges feed each other: alternate to a fixed point
                new_twins = self.composer.merge_label_twins(obs, self.labels)
                if gold:
                    before = len(self.composer.classes.classes())
                    self._alignment_obs(stories)
                    if len(self.composer.classes.classes()) != before:
                        new_twins = new_twins + [("aligned", str(before - len(self.composer.classes.classes())))]
                new_merges = self.composer.merge_by_rows(obs)
                class_gender = {self.composer.classes.node(sym, FWD): g for sym, g in self.lexicon.gender.items()}
                new_merges += self.composer.merge_gender_twins(obs, class_gender)
                twins += new_twins
                merges += new_merges
                if not new_twins and not new_merges:
                    break
            self.composer.prune()
            if not gold:
                self._frame_update(stories)
            else:
                self._frame_update(stories, gold=True)
            self._word_genders(stories)
            self._refit_genders(stories)
            acc = sum(self.predict(s).label == s.target for s in stories) / max(1, len(stories))
            history.append({"round": r, "train_acc": acc, "twins": twins, "merges": len(merges),
                            "entries": len(self.composer.table), "classes": len(self.composer.classes.classes()),
                            **{f"obs_{k}": v for k, v in stats.items()}})
        # drop anchors that never enter a kept composition entry (MDL: they explain nothing)
        used = {c for key in self.composer.table for c in key} | {o for row in self.composer.table.values() for o in row}
        dropped = []
        for w in list(self.lexicon.symbol):
            sym = self.lexicon.symbol[w]
            if w in self.labels:
                continue
            nodes = {self.composer.classes.find((sym, FWD)), self.composer.classes.find((sym, REV))}
            if not nodes & used:
                self.lexicon.remove(w)
                dropped.append(w)
        self.lexicon.inherited.clear()
        self.reader.invalidate()
        self.log = history
        return {"arm": arm, "rounds": history, "discovered": discovered, "dropped": dropped}

    def _alignment_obs(self, stories: list[Story]) -> list[Obs]:
        """Secondary arm: gold edges align story words to gold relation symbols (as forced merges)."""
        votes: dict[tuple[str, int], Counter] = defaultdict(Counter)
        for s in stories:
            if not s.gold_edges:
                continue
            reading = self.reader.read(s)
            gold = {(x, y): rel for x, rel, y in s.gold_edges}
            for c in reading.edges:
                sym = self.lexicon.lookup(c.word) or c.word
                if (c.owner, c.relative) in gold:
                    votes[(sym, FWD)][gold[(c.owner, c.relative)]] += 1
                elif (c.relative, c.owner) in gold:
                    votes[(sym, REV)][gold[(c.relative, c.owner)]] += 1
        cls = self.composer.classes
        for node, cnt in votes.items():
            by_class: Counter = Counter()
            rep: dict[tuple, str] = {}
            for rel, n in cnt.items():
                k = cls.find(cls.node(rel, FWD))      # twins (father, mother) vote together
                by_class[k] += n
                rep.setdefault(k, rel)
            k, n = by_class.most_common(1)[0]
            if n < 10 or n / sum(by_class.values()) < 0.6 or cls.find(node) == k:
                continue
            # never create a symmetry the gold relation does not have (closure would spread it)
            inv_node = cls.find(cls.node(node[0], 1 - node[1]))
            gold_inv = cls.find(cls.node(rep[k], REV))
            if inv_node == k and gold_inv != k:
                continue
            # a story word read in this direction means that gold relation class
            cls.union(node, cls.node(rep[k], FWD))
        return []

    def bits(self) -> dict[str, float]:
        n_cls = len(self.composer.classes.classes())
        frames = len(self.reader.frame_score) * 8.0
        parts = {"composition": self.composer.bits(), "lexicon": self.lexicon.bits(n_cls),
                 "frames": frames, "names": self.genders.bits(),
                 "word_genders": len(self.lexicon.gender) * (math.log2(max(2, n_cls)) + 1.0)}
        parts["total"] = sum(parts.values())
        return parts

    def export(self) -> dict[str, Any]:
        cls = self.composer.classes
        members: dict[tuple, list[str]] = defaultdict(list)
        for node in list(cls.parent):
            members[cls.find(node)].append(f"{node[0]}{'' if node[1] == FWD else '^-1'}")
        name = {c: min(v, key=lambda s: (len(s), s)) for c, v in members.items()}
        table = {f"{name[a]} . {name[b]}": {name[o]: n for o, n in row.items()}
                 for (a, b), row in sorted(self.composer.table.items(), key=lambda kv: str(kv[0]))}
        return {"classes": {name[c]: sorted(v) for c, v in members.items()},
                "table": table,
                "lexicon": dict(sorted(self.lexicon.symbol.items())),
                "word_genders": dict(sorted(self.lexicon.gender.items())),
                "label_twins": self.composer.label_twin,
                "frames": {"|".join(k): round(v, 3) for k, v in sorted(self.reader.frame_score.items())}}


# ---------------------------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------------------------


def evaluate(program: FieldProgram, stories: list[Story]) -> dict[str, Any]:
    by_k: dict[int, list[int]] = defaultdict(list)
    reasons: Counter = Counter()
    per_story = []
    for s in stories:
        p = program.predict(s)
        ok = int(p.label == s.target)
        by_k[s.k].append(ok)
        reasons["correct" if ok else ("wrong-label" if p.reason == "ok" else p.reason)] += 1
        per_story.append({"id": s.sid, "k": s.k, "target": s.target, "pred": p.label, "reason": p.reason,
                          "path_len": p.path_len, "symbols": p.symbols})
    acc = {str(k): sum(v) / len(v) for k, v in sorted(by_k.items())}
    n = {str(k): len(v) for k, v in sorted(by_k.items())}
    total = sum(sum(v) for v in by_k.values()) / max(1, sum(len(v) for v in by_k.values()))
    return {"accuracy_by_k": acc, "n_by_k": n, "accuracy": total, "reasons": dict(reasons), "per_story": per_story}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def edge_diagnosis(program: FieldProgram, stories: list[Story]) -> dict[str, Any]:
    """Post hoc: how often each chosen fact matches a gold edge pair (reading accuracy), by k."""
    per_k: dict[int, Counter] = defaultdict(Counter)
    for s in stories:
        if not s.gold_edges:
            continue
        reading = program.reader.read(s)
        gold_pairs = {frozenset((x, y)) for x, _, y in s.gold_edges}
        chosen = {frozenset((c.owner, c.relative)) for c in reading.edges}
        per_k[s.k]["gold_edges"] += len(gold_pairs)
        per_k[s.k]["recovered_pairs"] += len(gold_pairs & chosen)
        per_k[s.k]["stories"] += 1
        per_k[s.k]["all_pairs_recovered"] += int(gold_pairs <= chosen)
        gg = s.gold_genders
        per_k[s.k]["gender_checked"] += len(gg)
        per_k[s.k]["gender_correct"] += sum(reading.genders.get(e) == g for e, g in gg.items())
    return {str(k): dict(v) for k, v in sorted(per_k.items())}


def composition_only(program: FieldProgram, stories: list[Story]) -> dict[str, Any]:
    """Post hoc D5: the learned composition table on gold edge sequences, realised with gold genders."""
    by_k: dict[int, list[int]] = defaultdict(list)
    for s in stories:
        if not s.gold_edges:
            continue
        seq = [program.composer.classes.node(program.lexicon.lookup(rel) or rel, FWD) for _, rel, _ in s.gold_edges]
        dist, _ = program.composer.compose_rewrite(seq)
        label = None
        if dist:
            cls = max(dist.items(), key=lambda kv: kv[1])[0]
            label = program.composer.label_for(cls, s.gold_genders.get(s.query[1]))
        by_k[s.k].append(int(label == s.target))
    return {str(k): {"accuracy": sum(v) / len(v), "n": len(v)} for k, v in sorted(by_k.items())}
