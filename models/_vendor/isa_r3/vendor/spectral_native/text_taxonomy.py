"""Taxonomy from real text: Hearst patterns -> 2-adic tree -> exact inheritance.

Pipeline (preregistered in docs/text-taxonomy-findings.md):

1. Clean Wikipedia wikitext (enwik9) into sentences.
2. Extract is-a pairs (hyponym, hypernym) with Hearst-style lexico-syntactic
   patterns (Roller, Kiela and Nickel 2018, Table 1 family).  Noun phrases are
   found without a POS tagger: a phrase head is the longest suffix of a word
   run that is a noun lemma in the WordNet noun lexicon (only the lemma list
   is used, never WordNet relations).
3. Aggregate with Roller's post-processing: keep pairs seen with at least two
   distinct patterns, drop the rarer direction of each pair.
4. Induce a forest: a greedy maximum-weight arborescence over the pattern
   pairs (edges in decreasing PPMI; an edge is kept if its hyponym has no
   parent yet and it closes no cycle), so each term has one parent.
5. Give the forest 2-adic prefix codes.  ``y`` is an ancestor of ``x`` iff
   ``code(x) = code(y) (mod 2**offset(depth(y)))``: a family is a 2-adic ball.
   The is-a score of an unobserved pair is the max-min closure of PPMI along
   the tree path (an ultrametric completion), the exact-structure analogue of
   Roller's low-rank smoothing.
6. Terms without pattern evidence are placed by distributional evidence: the
   deepest node that contains at least ``m`` of their ``k`` nearest
   distributional neighbours (a concept hull, as in knowledge.concept_hull).

Everything here is exact counting and combinatorics; the only floating point is
PPMI, cosine similarity and SVD for the distributional vectors and baselines.
"""

from __future__ import annotations

import html
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# Wikitext cleaning
# ---------------------------------------------------------------------------

_PAGE_TEXT = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)
_COMMENT = re.compile(r"<!--.*?-->", re.S)
_REF_SELF = re.compile(r"<ref[^>]*/>", re.S | re.I)
_REF = re.compile(r"<ref[^>]*>.*?</ref>", re.S | re.I)
_BLOCK_TAGS = re.compile(r"<(math|gallery|timeline|pre|code|source|score|imagemap)[^>]*>.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
_TABLE = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.S)
_NS_LINK = re.compile(r"\[\[[^\[\]|]*:[^\[\]]*\]\]")
_LINK = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]*)\]\]")
_EXT_LINK_TEXT = re.compile(r"\[(?:https?|ftp)://\S+\s([^\]]*)\]")
_EXT_LINK = re.compile(r"\[(?:https?|ftp)://\S+\]")
_QUOTES = re.compile(r"'{2,}")
_PAREN = re.compile(r"\([^()]*\)")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_TOKEN = re.compile(r"[a-z]+(?:[-'][a-z]+)*|[,;:.]")


def _until_stable(pattern: re.Pattern, repl: str, text: str, limit: int = 8) -> str:
    for _ in range(limit):
        new = pattern.sub(repl, text)
        if new == text:
            return new
        text = new
    return text


def clean_wikitext(raw: str) -> str:
    """Plain prose from one article's wikitext (XML-escaped as in enwik)."""

    text = html.unescape(raw)
    text = _COMMENT.sub(" ", text)
    text = _REF_SELF.sub(" ", text)
    text = _REF.sub(" ", text)
    text = _BLOCK_TAGS.sub(" ", text)
    text = _TAG.sub(" ", text)
    text = _until_stable(_TEMPLATE, " ", text)
    text = _until_stable(_TABLE, " ", text)
    for _ in range(4):
        new = _LINK.sub(r"\1", _NS_LINK.sub(" ", text))
        if new == text:
            break
        text = new
    text = _EXT_LINK_TEXT.sub(r"\1", text)
    text = _EXT_LINK.sub(" ", text)
    text = _QUOTES.sub("", text)
    return text


def sentences(text: str) -> list[str]:
    """Sentences of cleaned prose; headings, list markers and table rows dropped."""

    out = []
    for line in text.split("\n"):
        line = line.strip()
        if not line or line[0] in "=|!{}" or line.startswith("__"):
            continue
        line = line.lstrip("*#:; ").strip()
        line = _until_stable(_PAREN, " ", line)
        if len(line) < 12:
            continue
        out.extend(s for s in _SENT_SPLIT.split(line) if s)
    return out


def tokenize(sentence: str) -> list[str]:
    return _TOKEN.findall(sentence.lower())


def iter_article_texts(blob: str, limit: int | None = None) -> Iterator[tuple[str, str]]:
    """(title, raw wikitext) for main-namespace, non-redirect pages in ``blob``."""

    start = 0
    count = 0
    while True:
        a = blob.find("<page>", start)
        if a < 0 or (limit is not None and a >= limit):
            return
        b = blob.find("</page>", a)
        if b < 0:
            return
        page = blob[a:b]
        start = b + 7
        title_m = _TITLE.search(page)
        text_m = _PAGE_TEXT.search(page)
        if not title_m or not text_m:
            continue
        title = title_m.group(1)
        raw = text_m.group(1)
        if ":" in title or raw.lstrip()[:9].upper() == "#REDIRECT":
            continue
        count += 1
        yield title, raw


# ---------------------------------------------------------------------------
# Noun lexicon (WordNet lemma list only) and an NLTK-compatible noun lemmatizer
# ---------------------------------------------------------------------------

_NOUN_RULES = (("s", ""), ("ses", "s"), ("ves", "f"), ("xes", "x"), ("zes", "z"),
               ("ches", "ch"), ("shes", "sh"), ("men", "man"), ("ies", "y"))


@dataclass
class NounLexicon:
    """Noun lemma inventory and WordNet's morphy for nouns.

    ``lemma`` reproduces nltk's ``WordNetLemmatizer().lemmatize(word)`` (noun
    POS): morphy candidates that exist as noun lemmas, shortest first, else the
    word itself.  Only the lemma list and ``noun.exc`` are used.
    """

    lemmas: set[str]
    exceptions: dict[str, list[str]]
    _cache: dict[str, str] = field(default_factory=dict)

    def morphy(self, form: str) -> list[str]:
        forms: list[str] = []
        if form in self.exceptions:
            forms = [form] + self.exceptions[form]
        else:
            forms = [form] + [form[: -len(old)] + new for old, new in _NOUN_RULES if form.endswith(old)]
        seen, out = set(), []
        for f in forms:
            if f in self.lemmas and f not in seen:
                seen.add(f)
                out.append(f)
        return out

    def lemma(self, word: str) -> str:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        forms = self.morphy(word)
        out = min(forms, key=len) if forms else word
        self._cache[word] = out
        return out

    def term(self, words: list[str]) -> str | None:
        """Noun lemma for a word sequence, last word lemmatized, or None."""

        if not words:
            return None
        if len(words) == 1:
            forms = self.morphy(words[0])
            return min(forms, key=len) if forms else None
        head_forms = self.morphy(words[-1]) or [words[-1]]
        for head in sorted(head_forms, key=len):
            cand = "_".join(words[:-1] + [head])
            if cand in self.lemmas:
                return cand
        cand = "_".join(words)
        return cand if cand in self.lemmas else None


def load_noun_lexicon(wordnet_dir: Path) -> NounLexicon:
    lemmas: set[str] = set()
    with open(wordnet_dir / "index.noun", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("  "):
                continue
            lemmas.add(line.split(" ", 1)[0].lower())
    exceptions: dict[str, list[str]] = {}
    with open(wordnet_dir / "noun.exc", encoding="latin-1") as handle:
        for line in handle:
            fields = line.split()
            if len(fields) >= 2:
                exceptions[fields[0]] = fields[1:]
    return NounLexicon(lemmas, exceptions)


# ---------------------------------------------------------------------------
# Hearst patterns
# ---------------------------------------------------------------------------

DETERMINERS = frozenset("a an the some many several most all any various certain numerous few these those this that "
                        "such other another each every its their his her our your my".split())
STOP = frozenset((
    "a an the and or but nor of in on at to for with by from as into onto upon about over under between among "
    "through during without within than via per is are was were be been being has have had can could may might "
    "will would should shall must do does did not no only also often usually especially particularly mainly "
    "mostly very more most less least such like unlike including which that who whom whose where when while "
    "what how why it they he she we you i there here this these those its their his her our your my "
    "native endemic found known called named related similar common commonly generally typically closely "
    "characterized used living located based considered being having").split())
PATTERNS = ("such_as", "such_y_as", "including", "especially", "and_other", "copula", "kind_of",
            "superlative", "like_other", "which_is")
KIND_OF = frozenset("kind type sort form species specie breed variety example class genus subspecies family group "
                    "race strain".split())
COPULA_SKIP = frozenset("member part given piece name one unit number lot series set".split())
LIST_JOIN = frozenset({",", "and", "or"})
CUE = re.compile(r"\b(?:such|including|especially|other|is|was|like|unlike)\b", re.I)


def _word(tok: str) -> bool:
    return tok[0].isalpha()


LIGHT_NOUNS = frozenset("specie species kind type sort variety breed form group family genus subspecies member "
                        "class strain race".split())


def _np_terms(run: list[str], lex: NounLexicon, light_heads: bool = False) -> list[str]:
    """Head noun term of a word run, plus its single-word head if different.

    The head is the longest noun term ending at the run's last word; if the
    last word is not a noun (a verb after the phrase, with no POS tagger), the
    rightmost end position that yields a noun term is used instead.  With
    ``light_heads`` (post hoc, D3) a light noun ("bird species", "dog breeds")
    passes the head to its modifier, and a bare light noun yields nothing.
    """

    out: list[str] = []
    for end in range(len(run), 0, -1):
        for n in range(min(3, end), 0, -1):
            t = lex.term(run[end - n:end])
            if t is not None:
                out.append(t)
                break
        if out:
            run = run[:end]
            break
    if light_heads and out and out[0] in LIGHT_NOUNS:
        return _np_terms(run[:-1], lex, light_heads) if len(run) > 1 else []
    if out and "_" in out[0]:
        head = lex.term(run[-1:])
        if head is not None and head != out[0]:
            out.append(head)
    return out


def _run_left(tokens: list[str], end: int, max_len: int = 4) -> list[str]:
    """Maximal content-word run ending at ``end`` (inclusive)."""

    run: list[str] = []
    i = end
    while i >= 0 and len(run) < max_len and _word(tokens[i]) and tokens[i] not in STOP:
        run.append(tokens[i])
        i -= 1
    return run[::-1]


def _run_right(tokens: list[str], start: int, max_len: int = 5, allow_commas: bool = False) -> tuple[list[str], int]:
    """Content-word run from ``start`` (after determiners); returns (run, next index)."""

    i = start
    while i < len(tokens) and tokens[i] in DETERMINERS:
        i += 1
    run: list[str] = []
    while i < len(tokens) and len(run) < max_len:
        tok = tokens[i]
        if allow_commas and tok == "," and i + 1 < len(tokens) and _word(tokens[i + 1]) and tokens[i + 1] not in STOP:
            i += 1
            continue
        if not _word(tok) or tok in STOP:
            break
        run.append(tok)
        i += 1
    return run, i


def _list_right(tokens: list[str], start: int, max_items: int = 10) -> list[list[str]]:
    items: list[list[str]] = []
    i = start
    while i < len(tokens) and len(items) < max_items:
        run, i = _run_right(tokens, i)
        if not run:
            break
        items.append(run)
        if i < len(tokens) and tokens[i] in LIST_JOIN:
            i += 1
            if i < len(tokens) and tokens[i] in ("and", "or"):
                i += 1
            continue
        break
    return items


def _list_left(tokens: list[str], end: int, max_items: int = 10) -> list[list[str]]:
    """Items X1, X2 and ... ending at ``end`` (inclusive), right to left."""

    items: list[list[str]] = []
    i = end
    while i >= 0 and len(items) < max_items:
        run = _run_left(tokens, i)
        if not run:
            break
        items.append(run)
        i -= len(run)
        if i >= 0 and tokens[i] == ",":
            i -= 1
            continue
        break
    return items


def _left_hyper(tokens: list[str], cue: int) -> list[str]:
    end = cue - 1
    if end >= 0 and tokens[end] == ",":
        end -= 1
    return _run_left(tokens, end) if end >= 0 else []


def extract_sentence(tokens: list[str], lex: NounLexicon, light_heads: bool = False) -> list[tuple[str, str, str]]:
    """All (hyponym, hypernym, pattern) hits in one tokenized sentence."""

    hits: list[tuple[str, str, str]] = []

    def emit(hypo_runs: list[list[str]], hyper_run: list[str], pattern: str) -> None:
        hypers = _np_terms(hyper_run, lex, light_heads)
        if not hypers:
            return
        for run in hypo_runs:
            for x in _np_terms(run, lex, light_heads):
                for y in hypers:
                    if x != y:
                        hits.append((x, y, pattern))

    n = len(tokens)
    for i, tok in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < n else ""
        if tok == "such" and nxt == "as":
            hyper = _left_hyper(tokens, i)
            if hyper and hyper[-1] not in ("features", "properties", "feature", "property"):
                emit(_list_right(tokens, i + 2), hyper, "such_as")
        elif tok == "such" and nxt and nxt != "as":
            run, j = _run_right(tokens, i + 1, max_len=4)
            if run and j < n and tokens[j] == "as":
                emit(_list_right(tokens, j + 1), run, "such_y_as")
        elif tok == "including":
            hyper = _left_hyper(tokens, i)
            if hyper:
                emit(_list_right(tokens, i + 1), hyper, "including")
        elif tok == "especially":
            hyper = _left_hyper(tokens, i)
            if hyper:
                emit(_list_right(tokens, i + 1), hyper, "especially")
        elif tok in ("and", "or") and nxt == "other" or (tok in ("and", "or") and nxt in ("any", "some")
                                                        and i + 2 < n and tokens[i + 2] == "other"):
            skip = 2 if nxt == "other" else 3
            hyper, _ = _run_right(tokens, i + skip, max_len=4)
            if hyper:
                emit(_list_left(tokens, i - 1), hyper, "and_other")
        elif tok in ("is", "was") and nxt in ("a", "an", "the"):
            subject_end = i - 1
            pattern_base = "copula"
            if subject_end >= 0 and tokens[subject_end] == "which":
                subject_end -= 1
                if subject_end >= 0 and tokens[subject_end] == ",":
                    subject_end -= 1
                pattern_base = "which_is"
            hypo = _run_left(tokens, subject_end) if subject_end >= 0 else []
            if not hypo:
                continue
            if nxt == "the":
                if i + 2 < n and (tokens[i + 2] == "most" or (tokens[i + 2].endswith("est") and len(tokens[i + 2]) > 4)):
                    start = i + 4 if tokens[i + 2] == "most" else i + 3
                    run, _ = _run_right(tokens, start, max_len=4)
                    if run:
                        emit([hypo], run, "superlative")
                continue
            run, j = _run_right(tokens, i + 2, max_len=5, allow_commas=True)
            if not run:
                continue
            head = run[-1]
            head_lemma = lex.lemma(head)
            kind_word = head in KIND_OF or head_lemma in KIND_OF
            if kind_word and j < n and tokens[j] == "of":
                of_run, _ = _run_right(tokens, j + 1, max_len=4)
                if of_run:
                    emit([hypo], of_run, "kind_of")
                continue
            if kind_word or head_lemma in COPULA_SKIP:
                # a bare "species" / "kind" / "member" says nothing about the category
                continue
            emit([hypo], run, pattern_base)
        elif tok in ("like", "unlike") and nxt in ("most", "all", "any", "other"):
            hyper, j = _run_right(tokens, i + 2, max_len=4)
            if hyper and j < n and tokens[j] == ",":
                hypo, _ = _run_right(tokens, j + 1, max_len=4)
                if hypo:
                    emit([hypo], hyper, "like_other")
    return hits


def extract_text(text: str, lex: NounLexicon, light_heads: bool = False) -> Iterator[tuple[str, str, str]]:
    for sent in sentences(text):
        if not CUE.search(sent):
            continue
        yield from extract_sentence(tokenize(sent), lex, light_heads)


# ---------------------------------------------------------------------------
# Aggregation (Roller et al. 2018 post-processing) and pattern-based scores
# ---------------------------------------------------------------------------


def roller_filter(hits: dict[tuple[str, str, str], int], min_patterns: int = 2) -> dict[tuple[str, str], int]:
    """Pairs seen with >= ``min_patterns`` distinct patterns; rarer direction dropped."""

    count: Counter = Counter()
    patterns: dict[tuple[str, str], set[str]] = defaultdict(set)
    for (x, y, p), c in hits.items():
        count[(x, y)] += c
        patterns[(x, y)].add(p)
    kept = {pair: c for pair, c in count.items() if len(patterns[pair]) >= min_patterns}
    out = {}
    for (x, y), c in kept.items():
        if kept.get((y, x), 0) > c:
            continue
        out[(x, y)] = c
    return out


@dataclass
class PatternMatrix:
    """Hearst count matrix over one vocabulary (row = hyponym, col = hypernym).

    Index 0 is ``<OOV>`` with an empty row and column, as in hypernymysuite.
    """

    vocab: dict[str, int]
    words: list[str]
    counts: Any  # scipy.sparse.csr_matrix

    @classmethod
    def from_pairs(cls, pairs: dict[tuple[str, str], float]) -> "PatternMatrix":
        from scipy import sparse

        vocab = {"<OOV>": 0}
        words = ["<OOV>"]
        rows, cols, vals = [], [], []
        for (x, y), c in pairs.items():
            for w in (x, y):
                if w not in vocab:
                    vocab[w] = len(words)
                    words.append(w)
            rows.append(vocab[x])
            cols.append(vocab[y])
            vals.append(float(c))
        n = len(words)
        mat = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        mat.sum_duplicates()
        return cls(vocab, words, mat)

    def ppmi(self) -> Any:
        """PPMI exactly as hypernymysuite's PPMIModel (row/col sums of counts)."""

        mat = self.counts.tocoo()
        p_w = np.asarray(self.counts.sum(axis=1)).ravel()
        p_c = np.asarray(self.counts.sum(axis=0)).ravel()
        total = p_w.sum()
        pmi = np.log(total) + np.log(mat.data) - np.log(p_w[mat.row]) - np.log(p_c[mat.col])
        from scipy import sparse

        out = sparse.csr_matrix((np.clip(pmi, 0.0, None), (mat.row, mat.col)), shape=mat.shape)
        out.eliminate_zeros()
        return out

    def lookup(self, matrix: Any, hypos: Iterable[str], hypers: Iterable[str]) -> np.ndarray:
        rows = np.array([self.vocab.get(x, 0) for x in hypos], dtype=np.int64)
        cols = np.array([self.vocab.get(y, 0) for y in hypers], dtype=np.int64)
        if rows.size == 0:
            return np.zeros(0)
        return np.asarray(matrix[rows, cols]).ravel().astype(np.float64)


def svd_factors(matrix: Any, rank: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """U diag(S), V of a truncated SVD (hypernymysuite _SvdMixIn, randomized solver)."""

    from sklearn.utils.extmath import randomized_svd

    rank = int(min(rank, min(matrix.shape) - 1))
    u, s, vt = randomized_svd(matrix, n_components=rank, n_iter=5, random_state=seed)
    return u * s[None, :], vt.T


# ---------------------------------------------------------------------------
# 2-adic taxonomy
# ---------------------------------------------------------------------------


@dataclass
class TextTaxonomy:
    """Forest under a super-root (node 0), with 2-adic prefix codes.

    ``parent[i]`` is the parent node, ``weight[i]`` the PPMI of the edge to it
    (0 for the super-root's children), ``placed[i]`` marks distributional
    placements.  Codes: the children of a node at depth d-1 get digits
    0..c-1 in bits [offset[d-1], offset[d]); ``code`` is exact (Python int).
    """

    terms: list[str]
    index: dict[str, int]
    parent: list[int]
    weight: list[float]
    placed: list[bool]
    depth: list[int] = field(default_factory=list)
    code: list[int] = field(default_factory=list)
    offset: list[int] = field(default_factory=list)

    def finalize(self) -> "TextTaxonomy":
        n = len(self.terms)
        children: list[list[int]] = [[] for _ in range(n)]
        for i in range(1, n):
            children[self.parent[i]].append(i)
        depth = [0] * n
        order = [0]
        for node in order:
            for c in children[node]:
                depth[c] = depth[node] + 1
                order.append(c)
        if len(order) != n:
            raise ValueError("taxonomy is not a tree under the super-root")
        max_depth = max(depth)
        width = [0] * (max_depth + 1)
        for node in range(n):
            if children[node]:
                d = depth[node] + 1
                width[d] = max(width[d], max(1, math.ceil(math.log2(len(children[node]) + 1))))
        offset = [0] * (max_depth + 1)
        for d in range(1, max_depth + 1):
            offset[d] = offset[d - 1] + width[d]
        code = [0] * n
        for node in order:
            for digit, c in enumerate(children[node], start=1):
                code[c] = code[node] + (digit << offset[depth[node]])
        self.depth, self.code, self.offset = depth, code, offset
        self._children = children
        return self

    @property
    def children(self) -> list[list[int]]:
        return self._children

    def is_ancestor(self, up: int, node: int) -> bool:
        """Exact 2-adic ball membership: code(node) = code(up) mod 2**offset(depth(up))."""

        if up == node:
            return False
        d = self.depth[up]
        if d >= self.depth[node]:
            return False
        return (self.code[node] - self.code[up]) % (1 << self.offset[d]) == 0

    def ancestors(self, node: int) -> list[int]:
        out = []
        node = self.parent[node]
        while node > 0:
            out.append(node)
            node = self.parent[node]
        return out

    def closure(self, node: int, up: int) -> float:
        """Max-min path strength from ``node`` up to ancestor ``up`` (0 if not an ancestor)."""

        if not self.is_ancestor(up, node):
            return 0.0
        strength = math.inf
        while node != up:
            strength = min(strength, self.weight[node])
            node = self.parent[node]
        return float(strength)

    def leaves_under(self, node: int) -> list[int]:
        out, stack = [], [node]
        while stack:
            cur = stack.pop()
            kids = self._children[cur]
            if not kids:
                out.append(cur)
            stack.extend(kids)
        return out


def generality(pairs: dict[tuple[str, str], float]) -> dict[str, float]:
    """Hypernym-side extraction mass p+(t) (column sums), the generality order."""

    g: Counter = Counter()
    for (_x, y), c in pairs.items():
        g[y] += c
    return dict(g)


def induce_forest(pairs: dict[tuple[str, str], float], weights: dict[tuple[str, str], float]) -> TextTaxonomy:
    """One parent per term, greedy maximum-weight arborescence over the pattern pairs.

    Edges (hyponym -> hypernym) are taken in decreasing (weight, count) order;
    an edge is kept if its hyponym has no parent yet and the hypernym's current
    ancestor chain does not contain the hyponym (no cycle).  Pair direction
    already encodes generality (Roller's filter keeps the dominant direction).
    """

    terms: set[str] = set()
    edges = []
    for (x, y), c in pairs.items():
        terms.add(x)
        terms.add(y)
        edges.append((-weights.get((x, y), 0.0), -c, x, y))
    edges.sort()
    up: dict[str, str] = {}
    w_of: dict[str, float] = {}
    for negw, _negc, x, y in edges:
        if x in up:
            continue
        cur: str | None = y
        loop = False
        while cur is not None:
            if cur == x:
                loop = True
                break
            cur = up.get(cur)
        if loop:
            continue
        up[x] = y
        w_of[x] = -negw
    # breadth-first names so that parents precede children
    kids: dict[str, list[str]] = {}
    for x, y in up.items():
        kids.setdefault(y, []).append(x)
    roots = sorted(t for t in terms if t not in up)
    names = ["<root>"]
    queue = list(roots)
    for t in queue:
        names.append(t)
        queue.extend(sorted(kids.get(t, [])))
    index = {t: i for i, t in enumerate(names)}
    parent = [-1] + [0] * (len(names) - 1)
    weight = [0.0] * len(names)
    for x, y in up.items():
        parent[index[x]] = index[y]
        weight[index[x]] = float(w_of[x])
    return TextTaxonomy(names, index, parent, weight, [False] * len(names)).finalize()


def with_placements(tax: TextTaxonomy, placements: dict[str, tuple[int, float]]) -> TextTaxonomy:
    """New taxonomy with extra leaves ``term -> (parent node, edge weight)``."""

    names = list(tax.terms)
    index = dict(tax.index)
    parent = list(tax.parent)
    weight = list(tax.weight)
    placed = list(tax.placed)
    for term, (up, w) in sorted(placements.items()):
        if term in index:
            node = index[term]
            if parent[node] == 0 and not tax.children[node]:
                parent[node], weight[node], placed[node] = up, w, True
            continue
        index[term] = len(names)
        names.append(term)
        parent.append(up)
        weight.append(w)
        placed.append(True)
    return TextTaxonomy(names, index, parent, weight, placed).finalize()


def hull_node(tax: TextTaxonomy, nodes: list[int], need: int) -> int:
    """Deepest node whose subtree (itself included) holds at least ``need`` of ``nodes``."""

    hits: Counter = Counter()
    for n in nodes:
        cur = n
        while cur > 0:
            hits[cur] += 1
            cur = tax.parent[cur]
    best, best_depth = 0, -1
    for node, c in hits.items():
        if c >= need and tax.depth[node] > best_depth:
            best, best_depth = node, tax.depth[node]
    return best


def closure_scores(tax: TextTaxonomy, direct: dict[tuple[str, str], float], hypos: list[str],
                   hypers: list[str]) -> np.ndarray:
    """max(direct PPMI, tree closure) for each pair; 0 when either term is unknown."""

    out = np.zeros(len(hypos))
    for k, (x, y) in enumerate(zip(hypos, hypers)):
        s = direct.get((x, y), 0.0)
        i, j = tax.index.get(x), tax.index.get(y)
        if i is not None and j is not None:
            s = max(s, tax.closure(i, j))
        out[k] = s
    return out


# ---------------------------------------------------------------------------
# Post hoc diagnoses: several parents per term (D1) and context families (D2)
# ---------------------------------------------------------------------------


def top_parents(pairs: dict[tuple[str, str], float], k: int | None) -> dict[str, list[tuple[str, float]]]:
    """Each term's ``k`` strongest hypernyms by count (all when ``k`` is None)."""

    by: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (x, y), c in pairs.items():
        by[x].append((c, y))
    out = {}
    for x, lst in by.items():
        lst.sort(key=lambda t: (-t[0], t[1]))
        out[x] = [(y, c) for c, y in (lst if k is None else lst[:k])]
    return out


def bottleneck_from(adj: dict[str, list[tuple[str, float]]], source: str) -> dict[str, float]:
    """Widest-path strength from ``source`` to every reachable hypernym (max over paths of min edge)."""

    import heapq

    best: dict[str, float] = {}
    heap = [(-math.inf, source)]
    seen: set[str] = set()
    while heap:
        negw, u = heapq.heappop(heap)
        if u in seen:
            continue
        seen.add(u)
        if u != source:
            best[u] = -negw
        for v, w in adj.get(u, ()):
            if v in seen or v == source:
                continue
            cand = min(-negw, w)
            if cand > best.get(v, 0.0):
                best[v] = cand
                heapq.heappush(heap, (-cand, v))
    return best


def context_families(ents: list[str], pairs: dict[tuple[str, str], float]) -> list[tuple[float, str, frozenset[int]]]:
    """Candidate families over an entity set, scored by in-set support times in-set share.

    s(e, y) = max(c(e, y), max_z min(c(e, z), c(z, y))): direct or two-hop pattern support.
    A hypernym y proposes the family F_y = {e : s(e, y) > 0} if it holds at least two
    entities and not all of them.  Score = support(F_y) * support(F_y) / total(y), where
    total(y) is all of y's hyponym support in the corpus: a family that is specific to this
    set of entities (its context) beats a generic one ("character", "product").
    """

    up = top_parents(pairs, None)
    total: Counter = Counter()
    for (_x, y), c in pairs.items():
        total[y] += c
    support: dict[str, dict[int, float]] = defaultdict(dict)
    for i, e in enumerate(ents):
        strength: dict[str, float] = {}
        for z, c1 in up.get(e, ()):
            strength[z] = max(strength.get(z, 0.0), c1)
            for y, c2 in up.get(z, ()):
                if y != e:
                    strength[y] = max(strength.get(y, 0.0), min(c1, c2))
        for y, v in strength.items():
            support[y][i] = v
    out = []
    n = len(ents)
    for y, members in support.items():
        if 2 <= len(members) < n:
            sup = sum(members.values())
            out.append((sup * sup / max(total[y], sup), y, frozenset(members)))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


def laminar_tree(n: int, families: list[tuple[float, str, frozenset[int]]]) -> tuple[Any, list[tuple[str, list[int]]]]:
    """Greedy laminar family (nested or disjoint sets, best score first) as a knowledge.Tree."""

    from .knowledge import Tree

    chosen: list[tuple[str, frozenset[int]]] = []
    for _score, name, members in families:
        ok = True
        for _n, other in chosen:
            if members == other or not (members <= other or other <= members or not (members & other)):
                ok = False
                break
        if ok:
            chosen.append((name, members))
    chosen.sort(key=lambda t: -len(t[1]))
    parent = [-1]
    names = ["<root>"]
    node_of: list[int] = []
    for name, members in chosen:
        up = 0
        for j, (_n2, other) in enumerate(chosen[: len(node_of)]):
            if members < other:
                up = j + 1  # families are sorted by size, so the last container is the smallest
        parent.append(up)
        names.append(name)
        node_of.append(len(parent) - 1)
    leaves = []
    for e in range(n):
        up = 0
        for j, (_n2, members) in enumerate(chosen):
            if e in members:
                up = j + 1
        parent.append(up)
        leaves.append(len(parent) - 1)
    parent_arr = np.array(parent)
    children: list[list[int]] = [[] for _ in parent]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    depth = np.zeros(len(parent), dtype=int)
    order = [0]
    for node in order:
        for c in children[node]:
            depth[c] = depth[node] + 1
            order.append(c)
    return Tree(parent_arr, children, depth, np.array(leaves)), [(nm, sorted(m)) for nm, m in chosen]


# ---------------------------------------------------------------------------
# Distributional vectors (PPMI + SVD over a symmetric word window)
# ---------------------------------------------------------------------------


def ppmi_svd(cooc: Any, dims: int = 300, seed: int = 0) -> np.ndarray:
    """Row-normalized SVD embedding of the PPMI-transformed co-occurrence matrix."""

    from scipy import sparse
    from sklearn.utils.extmath import randomized_svd

    mat = cooc.tocoo().astype(np.float64)
    row = np.asarray(cooc.sum(axis=1)).ravel()
    col = np.asarray(cooc.sum(axis=0)).ravel()
    total = mat.data.sum()
    col_s = np.power(col, 0.75)  # context distribution smoothing (Levy et al. 2015)
    col_p = col_s / col_s.sum()
    pmi = np.log(mat.data / total) - np.log(row[mat.row] / total) - np.log(col_p[mat.col])
    keep = pmi > 0
    ppmi = sparse.csr_matrix((pmi[keep], (mat.row[keep], mat.col[keep])), shape=cooc.shape)
    dims = int(min(dims, min(ppmi.shape) - 1))
    u, s, _ = randomized_svd(ppmi, n_components=dims, n_iter=5, random_state=seed)
    vec = u * np.sqrt(s)[None, :]
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    return vec / np.maximum(norm, 1e-12)


def nearest(vectors: np.ndarray, query: int, pool: np.ndarray, k: int) -> np.ndarray:
    sims = vectors[pool] @ vectors[query]
    order = np.argsort(-sims, kind="stable")[:k]
    return pool[order]


# ---------------------------------------------------------------------------
# WordNet checks (evaluation only)
# ---------------------------------------------------------------------------


def wordnet_ancestor_lemmas(nouns: dict[str, dict[str, Any]], index: dict[str, list[str]], lemma: str,
                            cache: dict[str, set[str]]) -> set[str]:
    """Lemmas of all transitive hypernyms (incl. instance hypernyms) of any noun sense."""

    hit = cache.get(lemma)
    if hit is not None:
        return hit
    out: set[str] = set()
    for off in index.get(lemma, []):
        stack = list(nouns[off]["hypernyms"])
        seen: set[str] = set()
        while stack:
            s = stack.pop()
            if s in seen:
                continue
            seen.add(s)
            out.update(l.lower() for l in nouns[s]["lemmas"])
            stack.extend(nouns[s]["hypernyms"])
    cache[lemma] = out
    return out


def lca_depths(parent: np.ndarray, depth: np.ndarray, leaves: np.ndarray) -> np.ndarray:
    """Depth of the lowest common ancestor for every pair of leaves."""

    chains = []
    for leaf in leaves:
        chain = []
        node = int(leaf)
        while node != -1:
            chain.append(node)
            node = int(parent[node])
        chains.append(set(chain))
    n = len(leaves)
    out = np.zeros((n, n), dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            common = chains[i] & chains[j]
            out[i, j] = out[j, i] = max(int(depth[c]) for c in common)
    return out


def triplet_agreement(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Agreement on which pair of each leaf triple is closest (resolved triples of the reference)."""

    n = reference.shape[0]
    resolved = agree = cand_resolved = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                r = (reference[i, j], reference[i, k], reference[j, k])
                c = (candidate[i, j], candidate[i, k], candidate[j, k])
                rt = _closest(r)
                if rt is None:
                    continue
                resolved += 1
                ct = _closest(c)
                if ct is not None:
                    cand_resolved += 1
                    agree += int(ct == rt)
    return {"resolved_reference_triples": resolved,
            "agreement": agree / resolved if resolved else float("nan"),
            "candidate_resolves": cand_resolved / resolved if resolved else float("nan")}


def _closest(depths: tuple[int, int, int]) -> int | None:
    top = max(depths)
    if sorted(depths)[1] == top:
        return None
    return depths.index(top)
