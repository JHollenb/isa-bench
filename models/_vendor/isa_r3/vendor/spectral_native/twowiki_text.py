"""WS2 text front end: English question -> graded start entities and graded relation chains.

Both readers hand the field distributions, not decisions (the WS1 lesson applied at the
language boundary):

- ``NameIndex`` / ``Linker``: every question n-gram that is a T-REx surface string or article
  title proposes entities; a small learned scorer (count, title, capitalization, popularity,
  field support, neighbouring words) turns all proposals into one distribution over entities;
- ``PathReader``: the question with the linked span masked, read as a distribution over
  two-step Wikidata property chains (softmax on hashed words and bigrams).

Scores are fitted on 2Wiki train questions only.
"""

from __future__ import annotations

import gzip
import hashlib
import math
import random
import re
import string
import unicodedata
from collections import Counter, defaultdict
from typing import Callable, Mapping, Sequence

MAX_NGRAM = 16
PER_SPAN = 8
MAX_CANDIDATES = 256
STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "is", "was", "who", "what", "where", "when", "which", "did",
    "does", "do", "s", "and", "or", "for", "by", "with", "from", "that", "this", "film", "song", "album", "born",
    "die", "died", "father", "mother", "director", "place", "birth", "death", "country", "spouse", "wife",
    "husband", "child", "award", "work", "performer", "composer", "grandfather", "grandmother", "law",
}


def _h(x: str) -> int:
    return int(hashlib.sha256(x.encode()).hexdigest()[:8], 16)


def _strip(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def norm(s: str) -> str:
    """Same normalization as scripts/index_twowiki.py."""
    return " ".join(re.findall(r"[0-9a-z]+", _strip(s).casefold()))


def tokens(text: str) -> tuple[list[str], list[bool]]:
    """Normalized tokens and whether each was capitalized in the original text."""
    low = re.findall(r"[0-9a-z]+", _strip(text).casefold())
    orig = re.findall(r"[0-9A-Za-z]+", _strip(text))
    caps = [t[:1].isupper() for t in orig] if len(orig) == len(low) else [False] * len(low)
    return low, caps


class NameIndex:
    def __init__(self, names: Mapping[str, Counter], titles: Mapping[str, str], mentions: Mapping[str, int],
                 support: Callable[[str], int]) -> None:
        self.names, self.titles, self.mentions, self.support = names, titles, mentions, support
        self.title_norm = {}
        for q, t in titles.items():
            self.title_norm[q] = {norm(t), norm(re.sub(r"\s*\([^)]*\)\s*$", "", t))}

    @classmethod
    def load(cls, index_dir, support: Callable[[str], int], keep_titles: set[str] | None = None) -> "NameIndex":
        names: dict[str, Counter] = defaultdict(Counter)
        with gzip.open(f"{index_dir}/names.tsv.gz", "rt") as f:
            for line in f:
                s, q, n = line.rstrip("\n").split("\t")
                names[s][q] += int(n)
        wanted = {q for c in names.values() for q in c} | (keep_titles or set())
        titles, mentions = {}, {}
        with gzip.open(f"{index_dir}/titles.tsv.gz", "rt") as f:
            for line in f:
                q, t = line.rstrip("\n").split("\t", 1)
                if q in wanted:
                    titles[q] = t
        with gzip.open(f"{index_dir}/mentions.tsv.gz", "rt") as f:
            for line in f:
                q, n = line.rstrip("\n").split("\t")
                if q in wanted:
                    mentions[q] = int(n)
        return cls(dict(names), titles, mentions, support)

    def candidates(self, question: str) -> list[dict]:
        toks, caps = tokens(question)
        spans = []
        for i in range(len(toks)):
            for j in range(i + 1, min(len(toks), i + MAX_NGRAM) + 1):
                g = " ".join(toks[i:j])
                if g in self.names and not all(t in STOP for t in toks[i:j]) and len(g) >= 2:
                    spans.append((i, j, g))
        covered = defaultdict(int)
        for i, j, _ in spans:
            for i2, j2, _ in spans:
                if i2 <= i and j <= j2 and (j2 - i2) > (j - i):
                    covered[(i, j)] += 1
        out = []
        for i, j, g in spans:
            c = self.names[g]
            total = sum(c.values())
            for q, n in c.most_common(PER_SPAN):
                out.append({"span": (i, j), "text": g, "entity": q, "dense": [
                    math.log(n / total), math.log1p(n), math.log1p(total), (j - i) / 4.0, (j - i) / max(1, len(toks)),
                    float(g in self.title_norm.get(q, ())), math.log1p(self.mentions.get(q, 0)) / 5.0,
                    math.log1p(self.support(q)) / 5.0, sum(caps[i:j]) / (j - i), float(caps[i]),
                    float(covered[(i, j)] == 0), sum(t in STOP for t in toks[i:j]) / (j - i)],
                    "sparse": [f"prev:{toks[i - 1] if i else '^'}", f"next:{toks[j] if j < len(toks) else '$'}",
                               f"prev2:{' '.join(toks[max(0, i - 2):i])}", f"len:{min(j - i, 6)}"]})
        out.sort(key=lambda c: -c["dense"][1] - 3 * c["dense"][5] - c["dense"][3])
        return out[:MAX_CANDIDATES]


class Linker:
    """Softmax over all (span, entity) proposals of a question; entity probability = summed mass."""

    DIM = 1 << 16
    DENSE = 12

    def __init__(self, index: NameIndex, train: Sequence[Mapping], seed: int, epochs: int = 4) -> None:
        import torch

        self.torch, self.index = torch, index
        torch.manual_seed(seed)
        self.lin = torch.nn.Linear(self.DENSE, 1)
        self.emb = torch.nn.EmbeddingBag(self.DIM, 1, mode="sum")
        torch.nn.init.zeros_(self.emb.weight)
        data = []
        self.coverage = Counter()
        for q in train:
            cands = index.candidates(q["question"])
            gold = [k for k, c in enumerate(cands) if c["entity"] == q["start"]]
            self.coverage["train"] += 1
            self.coverage["gold_in_candidates"] += bool(gold)
            if gold:
                data.append((cands, gold))
        opt = torch.optim.Adam([*self.lin.parameters(), *self.emb.parameters()], lr=0.05)
        rng = random.Random(seed)
        for _ in range(epochs):
            rng.shuffle(data)
            for i in range(0, len(data), 256):
                loss = 0.0
                for cands, gold in data[i:i + 256]:
                    logits = self._logits(cands)
                    loss = loss - (torch.logsumexp(logits[gold], 0) - torch.logsumexp(logits, 0))
                opt.zero_grad()
                (loss / len(data[i:i + 256])).backward()
                opt.step()

    def _logits(self, cands: Sequence[Mapping]):
        torch = self.torch
        dense = torch.tensor([c["dense"] for c in cands], dtype=torch.float32)
        flat = torch.tensor([_h(s) % self.DIM for c in cands for s in c["sparse"]], dtype=torch.long)
        offsets = torch.tensor([0] + [len(c["sparse"]) for c in cands[:-1]], dtype=torch.long).cumsum(0)
        return self.lin(dense)[:, 0] + self.emb(flat, offsets)[:, 0]

    def link(self, question: str, k: int = 5) -> list[dict]:
        """Top-k entities: {entity, p, span} with the entity's most probable span."""
        cands = self.index.candidates(question)
        if not cands:
            return []
        with self.torch.no_grad():
            probs = self.torch.softmax(self._logits(cands), 0).tolist()
        mass: dict[str, float] = defaultdict(float)
        best: dict[str, tuple[float, tuple[int, int]]] = {}
        for c, p in zip(cands, probs):
            mass[c["entity"]] += p
            if p > best.get(c["entity"], (-1.0, None))[0]:
                best[c["entity"]] = (p, c["span"])
        top = sorted(mass.items(), key=lambda t: -t[1])[:k]
        return [{"entity": e, "p": p, "span": best[e][1]} for e, p in top]


def masked(question: str, span: tuple[int, int] | None) -> list[str]:
    toks, _ = tokens(question)
    if span is None:
        return toks
    i, j = span
    return toks[:i] + ["<e>"] + toks[j:]


class PathReader:
    """Distribution over (P1, P2) chains seen in train from the masked question's words."""

    DIM = 1 << 18

    def __init__(self, train: Sequence[tuple[list[str], tuple[str, str]]], seed: int, epochs: int = 5) -> None:
        import torch

        self.torch = torch
        self.chains = sorted({c for _, c in train})
        self.index = {c: i for i, c in enumerate(self.chains)}
        torch.manual_seed(seed)
        self.emb = torch.nn.EmbeddingBag(self.DIM, len(self.chains), mode="mean")
        torch.nn.init.zeros_(self.emb.weight)
        self.bias = torch.nn.Parameter(torch.zeros(len(self.chains)))
        data = [(self._feats(t), self.index[c]) for t, c in train]
        opt = torch.optim.Adam([*self.emb.parameters(), self.bias], lr=0.05)
        rng = random.Random(seed)
        for _ in range(epochs):
            rng.shuffle(data)
            for i in range(0, len(data), 512):
                batch = data[i:i + 512]
                logits = self._logits([f for f, _ in batch])
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([y for _, y in batch]))
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _feats(self, toks: Sequence[str]) -> list[int]:
        names = [f"w:{t}" for t in toks] + [f"b:{a}_{b}" for a, b in zip(toks, toks[1:])]
        names += [f"t:{a}_{b}_{c}" for a, b, c in zip(toks, toks[1:], toks[2:])]
        return [_h(n) % self.DIM for n in names] or [0]

    def _logits(self, feats: Sequence[list[int]]):
        torch = self.torch
        flat = torch.tensor([i for f in feats for i in f], dtype=torch.long)
        offsets = torch.tensor([0] + [len(f) for f in feats[:-1]], dtype=torch.long).cumsum(0)
        return self.emb(flat, offsets) + self.bias

    def read(self, toks: Sequence[str], k: int = 5) -> list[tuple[tuple[str, str], float]]:
        with self.torch.no_grad():
            probs = self.torch.softmax(self._logits([self._feats(toks)])[0], -1)
        top = self.torch.topk(probs, min(k, len(self.chains)))
        return [(self.chains[i], float(p)) for p, i in zip(top.values.tolist(), top.indices.tolist())]


# -- 2Wiki official answer normalization (2wikimultihop_evaluate_v1.1.py / SQuAD) ------------------

def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def em_f1(pred: str, golds: Sequence[str]) -> tuple[float, float]:
    return (max(float(normalize_answer(pred) == normalize_answer(g)) for g in golds),
            max(f1(pred, g) for g in golds))
