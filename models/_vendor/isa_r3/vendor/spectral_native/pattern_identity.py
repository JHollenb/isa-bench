"""Relation identity of real-text surface patterns (experiment L2).

A surface pattern is the token span between two aligned entity mentions in one
sentence of T-REx (Wikipedia abstracts aligned to Wikidata), lower-cased, digits
mapped to 0, other entity mentions inside the span replaced by ``<e>``.  An
occurrence is (pattern, left entity, right entity, gold labels); a gold label is a
Wikidata predicate with its direction (``P19:f`` when the subject is the left
mention, ``P19:r`` otherwise).  Patterns never see their gold labels: identity is
inferred from facts (shared subjects whose objects agree) or from words.

Two regimes, as derived in the "What is a field" note:

- Common patterns: the same relation stated through two patterns agrees on shared
  subjects with probability 1-eps; distinct relations agree by chance with
  probability pi.  A likelihood-ratio merge decision needs about
  ``ln(1/delta) / KL(Bern(1-eps) || Bern(pi))`` shared subjects.
- Rare patterns share too few subjects to be identified by facts.  They are
  placed in a pattern taxonomy by their words (the pattern's words are its facts)
  and inherit the relation of the most specific family that contains them.

Everything in the field path is exact counting, trees and parsimony
(``spectral_native.knowledge``).  Gradient and embedding baselines live here too
so every method sees the same features and splits.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from spectral_native.knowledge import UNKNOWN, Tree, compress, tree_from_linkage

TOKEN = re.compile(r"\w+|[^\w\s]")
DIGIT = re.compile(r"\d")
MAX_TOKENS = 10
ENT = "<e>"
SPLITS = ("train", "dev", "test")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def tokens(text: str) -> list[str]:
    return TOKEN.findall(DIGIT.sub("0", text.lower()))


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def between_pattern(text: str, start: int, end: int, mentions: list[tuple[int, int]]) -> str | None:
    """Tokens of ``text[start:end]`` with inner entity mentions replaced by ``<e>``."""

    out: list[str] = []
    pos = start
    for m0, m1 in mentions:
        if m0 < pos or m1 > end:
            continue
        out.extend(tokens(text[pos:m0]))
        out.append(ENT)
        pos = m1
    out.extend(tokens(text[pos:end]))
    if len(out) > MAX_TOKENS:
        return None
    return " ".join(out)


def doc_occurrences(doc: dict[str, Any]) -> list[tuple[str, str, str, tuple[str, ...]]]:
    """Deduplicated (pattern, left qid, right qid, labels) for one T-REx document."""

    text = doc["text"]
    sents = doc.get("sentences_boundaries") or []
    mentions = sorted(
        {tuple(e["boundaries"]) for e in doc.get("entities") or [] if e.get("boundaries")}
    )
    seen: dict[tuple, list] = {}
    for triple in doc.get("triples") or []:
        subj, obj = triple["subject"], triple["object"]
        sb, ob = subj.get("boundaries"), obj.get("boundaries")
        sid = triple.get("sentence_id")
        if sb is None or ob is None or sid is None or not (0 <= sid < len(sents)):
            continue
        lo, hi = sents[sid]
        if not (lo <= sb[0] and sb[1] <= hi and lo <= ob[0] and ob[1] <= hi):
            continue
        sq, oq = _qid(subj["uri"]), _qid(obj["uri"])
        if sq == oq:
            continue
        forward = sb[0] < ob[0]
        left, right = (sb, ob) if forward else (ob, sb)
        if left[1] > right[0]:
            continue
        label = f"{_qid(triple['predicate']['uri'])}:{'f' if forward else 'r'}"
        key = (sid, tuple(left), tuple(right))
        if key in seen:
            seen[key][3].add(label)
            continue
        inner = [m for m in mentions if left[1] <= m[0] and m[1] <= right[0]]
        pattern = between_pattern(text, left[1], right[0], inner)
        if pattern is None:
            continue
        lq, rq = (sq, oq) if forward else (oq, sq)
        seen[key] = [pattern, lq, rq, {label}]
    return [(p, lq, rq, tuple(sorted(ls))) for p, lq, rq, ls in seen.values()]


def extract_member(args: tuple[str, str]) -> list[tuple[str, str, str, tuple[str, ...]]]:
    zip_path, name = args
    with zipfile.ZipFile(zip_path) as archive:
        docs = json.loads(archive.read(name))
    out: list[tuple[str, str, str, tuple[str, ...]]] = []
    for doc in docs:
        out.extend(doc_occurrences(doc))
    return out


def split_of(pattern: str) -> str:
    bucket = int(hashlib.sha1(pattern.encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "test")


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------


@dataclass
class Corpus:
    patterns: list[str]
    labels: list[str]
    entities: list[str]
    occ_pattern: np.ndarray  # (N,) pattern id
    occ_left: np.ndarray  # (N,) entity id
    occ_right: np.ndarray  # (N,) entity id
    occ_labels: np.ndarray  # (N, 3) label ids, -1 padded

    # derived per pattern
    count: np.ndarray | None = None
    gold: np.ndarray | None = None  # primary plurality label
    gold_sets: list[frozenset[int]] | None = None
    purity: np.ndarray | None = None
    split: np.ndarray | None = None  # 0 train, 1 dev, 2 test
    occ_start: np.ndarray | None = None  # CSR over occurrences sorted by pattern
    occ_order: np.ndarray | None = None


class CorpusBuilder:
    """Streaming id assignment: strings are interned once, occurrences kept as int32 arrays."""

    def __init__(self) -> None:
        from array import array

        self.pat_id: dict[str, int] = {}
        self.lab_id: dict[str, int] = {}
        self.ent_id: dict[str, int] = {}
        self.ps, self.ls, self.rs = array("i"), array("i"), array("i")
        self.labs = array("i")

    def add(self, occurrences: Iterable[tuple[str, str, str, tuple[str, ...]]]) -> None:
        for pattern, lq, rq, labels in occurrences:
            p = self.pat_id.setdefault(pattern, len(self.pat_id))
            left = self.ent_id.setdefault(lq, len(self.ent_id))
            right = self.ent_id.setdefault(rq, len(self.ent_id))
            self.ps.append(p)
            self.ls.append(left)
            self.rs.append(right)
            row = [self.lab_id.setdefault(x, len(self.lab_id)) for x in labels[:3]]
            self.labs.extend(row + [-1] * (3 - len(row)))

    def build(self) -> Corpus:
        corpus = Corpus(
            patterns=list(self.pat_id), labels=list(self.lab_id), entities=list(self.ent_id),
            occ_pattern=np.frombuffer(self.ps, dtype=np.int32).copy(),
            occ_left=np.frombuffer(self.ls, dtype=np.int32).copy(),
            occ_right=np.frombuffer(self.rs, dtype=np.int32).copy(),
            occ_labels=np.frombuffer(self.labs, dtype=np.int32).copy().reshape(-1, 3),
        )
        derive(corpus)
        return corpus


def build_corpus(occurrences: Iterable[tuple[str, str, str, tuple[str, ...]]]) -> Corpus:
    builder = CorpusBuilder()
    builder.add(occurrences)
    return builder.build()


def derive(corpus: Corpus) -> None:
    n_pat = len(corpus.patterns)
    order = np.argsort(corpus.occ_pattern, kind="stable")
    counts = np.bincount(corpus.occ_pattern, minlength=n_pat)
    start = np.zeros(n_pat + 1, dtype=np.int64)
    np.cumsum(counts, out=start[1:])
    gold = np.zeros(n_pat, dtype=np.int32)
    purity = np.zeros(n_pat)
    gold_sets: list[frozenset[int]] = []
    labs = corpus.occ_labels[order]
    for p in range(n_pat):
        block = labs[start[p]:start[p + 1]]
        c = Counter(int(x) for x in block.ravel() if x >= 0)
        top = max(c.values())
        best = sorted(k for k, v in c.items() if v == top)
        gold[p] = best[0]
        gold_sets.append(frozenset(best))
        purity[p] = top / max(1, counts[p])
    corpus.count = counts
    corpus.gold = gold
    corpus.gold_sets = gold_sets
    corpus.purity = purity
    corpus.split = np.array([SPLITS.index(split_of(p)) for p in corpus.patterns], dtype=np.int8)
    corpus.occ_start = start
    corpus.occ_order = order


def save_corpus(corpus: Corpus, path_npz: str, path_json: str) -> None:
    np.savez_compressed(path_npz, occ_pattern=corpus.occ_pattern, occ_left=corpus.occ_left,
                        occ_right=corpus.occ_right, occ_labels=corpus.occ_labels)
    import gzip

    with gzip.open(path_json, "wt") as handle:
        json.dump({"patterns": corpus.patterns, "labels": corpus.labels, "entities": corpus.entities}, handle)


def load_corpus(path_npz: str, path_json: str) -> Corpus:
    import gzip

    arrays = np.load(path_npz)
    with gzip.open(path_json, "rt") as handle:
        names = json.load(handle)
    corpus = Corpus(names["patterns"], names["labels"], names["entities"], arrays["occ_pattern"],
                    arrays["occ_left"], arrays["occ_right"], arrays["occ_labels"])
    derive(corpus)
    return corpus


def occurrences_of(corpus: Corpus, p: int) -> np.ndarray:
    return corpus.occ_order[corpus.occ_start[p]:corpus.occ_start[p + 1]]


def key_table(corpus: Corpus, p: int) -> dict[int, set[int]]:
    """Left entity -> set of right entities stated through pattern ``p``."""

    table: dict[int, set[int]] = {}
    for i in occurrences_of(corpus, p):
        table.setdefault(int(corpus.occ_left[i]), set()).add(int(corpus.occ_right[i]))
    return table


# ---------------------------------------------------------------------------
# Test 1: merge identifiability
# ---------------------------------------------------------------------------


def llr_weights(eps: float, pi: float) -> tuple[float, float]:
    """Per-shared-subject log-likelihood ratio of same vs distinct relation."""

    return math.log((1 - eps) / pi), math.log(eps / (1 - pi))


def merge_threshold(eps: float, pi: float) -> float:
    """Merge iff agreements a > c * n (equal priors, LLR > 0)."""

    agree, disagree = llr_weights(eps, pi)
    return -disagree / (agree - disagree)


def kl_bern(a: float, b: float) -> float:
    return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))


def predicted_accuracy(n: int, eps: float, pi: float) -> tuple[float, float]:
    """Exact binomial accuracy of the LLR>0 rule at n shared subjects: (same, distinct)."""

    from scipy.stats import binom

    c = merge_threshold(eps, pi)
    cut = math.floor(c * n + 1e-12)  # merge iff a > c n  <=>  a >= cut+1
    same = float(binom.sf(cut, n, 1 - eps))
    distinct = float(binom.cdf(cut, n, pi))
    return same, distinct


def first_n_reaching(curve: dict[int, float], target: float) -> int | None:
    for n in sorted(curve):
        if curve[n] >= target:
            return n
    return None


def sample_pairs(corpus: Corpus, patterns: np.ndarray, rng: np.random.Generator,
                 per_key: int, max_pairs: int) -> list[tuple[int, int]]:
    """Pattern pairs sharing at least one left entity, sampled through shared keys."""

    by_key: dict[int, list[int]] = {}
    for p in patterns:
        for key in {int(corpus.occ_left[i]) for i in occurrences_of(corpus, int(p))}:
            by_key.setdefault(key, []).append(int(p))
    keys = [k for k, v in by_key.items() if len(v) > 1]
    rng.shuffle(keys)
    pairs: set[tuple[int, int]] = set()
    for key in keys:
        members = by_key[key]
        m = len(members)
        total = m * (m - 1) // 2
        if total <= per_key:
            chosen = [(members[i], members[j]) for i in range(m) for j in range(i + 1, m)]
        else:
            chosen = []
            for _ in range(per_key):
                i, j = rng.choice(m, 2, replace=False)
                chosen.append((members[i], members[j]))
        for a, b in chosen:
            pairs.add((a, b) if a < b else (b, a))
        if len(pairs) >= max_pairs:
            break
    return sorted(pairs)


def pair_stats(corpus: Corpus, pairs: list[tuple[int, int]]) -> np.ndarray:
    """(n shared subjects, agreements, same gold) per pair."""

    cache: dict[int, dict[int, set[int]]] = {}

    def table(p: int) -> dict[int, set[int]]:
        if p not in cache:
            cache[p] = key_table(corpus, p)
        return cache[p]

    out = np.zeros((len(pairs), 3), dtype=np.int64)
    for row, (p, q) in enumerate(pairs):
        tp, tq = table(p), table(q)
        if len(tp) > len(tq):
            tp, tq = tq, tp
        n = a = 0
        for key, objs in tp.items():
            other = tq.get(key)
            if other is None:
                continue
            n += 1
            a += bool(objs & other)
        out[row] = (n, a, bool(corpus.gold_sets[p] & corpus.gold_sets[q]))
    return out


N_BINS = ((1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 7), (8, 11), (12, 19), (20, 31), (32, 63), (64, 10**9))


def accuracy_curve(stats: np.ndarray, c: float | np.ndarray, min_count: int = 30) -> dict[str, Any]:
    """Empirical accuracy of 'merge iff a > c n' in overlap bins."""

    n, a, same = stats[:, 0], stats[:, 1], stats[:, 2].astype(bool)
    cc = np.broadcast_to(np.asarray(c, dtype=float), n.shape)
    merge = a > cc * n
    rows = []
    for lo, hi in N_BINS:
        sel = (n >= lo) & (n <= hi)
        s_sel, d_sel = sel & same, sel & ~same
        row = {"lo": lo, "hi": hi, "same_pairs": int(s_sel.sum()), "distinct_pairs": int(d_sel.sum())}
        if s_sel.sum() >= min_count and d_sel.sum() >= min_count:
            acc_s = float(merge[s_sel].mean())
            acc_d = float((~merge[d_sel]).mean())
            row.update(same_acc=acc_s, distinct_acc=acc_d, balanced=(acc_s + acc_d) / 2)
        rows.append(row)
    return {"bins": rows}


def calibrate(stats: np.ndarray) -> dict[str, float]:
    """Jeffreys-smoothed (+1/2) rates so an empty or unanimous class never gives 0 or 1."""

    n, a, same = stats[:, 0], stats[:, 1], stats[:, 2].astype(bool)
    eps = (n[same].sum() - a[same].sum() + 0.5) / (n[same].sum() + 1.0)
    pi = (a[~same].sum() + 0.5) / (n[~same].sum() + 1.0)
    return {"eps": float(eps), "pi": float(pi), "same_subjects": int(n[same].sum()),
            "distinct_subjects": int(n[~same].sum()), "same_pairs": int(same.sum()),
            "distinct_pairs": int((~same).sum())}


# ---------------------------------------------------------------------------
# Features shared by every tail method
# ---------------------------------------------------------------------------


def features(pattern: str) -> list[str]:
    toks = pattern.split() if pattern else []
    feats = {f"len={min(len(toks), 8)}"}
    if not toks:
        feats.add("EMPTY")
    else:
        feats.update(f"u={t}" for t in toks)
        feats.update(f"b={x}_{y}" for x, y in zip(toks, toks[1:]))
        feats.add(f"first={toks[0]}")
        feats.add(f"last={toks[-1]}")
    return sorted(feats)


@dataclass
class FeatureSpace:
    index: dict[str, int]
    idf: np.ndarray

    def matrix(self, patterns: list[str]):
        from scipy.sparse import csr_matrix

        rows, cols, vals = [], [], []
        for r, pattern in enumerate(patterns):
            for f in features(pattern):
                j = self.index.get(f)
                if j is not None:
                    rows.append(r)
                    cols.append(j)
                    vals.append(self.idf[j])
        m = csr_matrix((vals, (rows, cols)), shape=(len(patterns), len(self.index)), dtype=np.float32)
        norms = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
        norms[norms == 0] = 1.0
        from scipy.sparse import diags

        return diags(1.0 / norms) @ m

    def known_features(self, pattern: str) -> int:
        return sum(1 for f in features(pattern) if f in self.index and not f.startswith("len="))


def feature_space(patterns: list[str]) -> FeatureSpace:
    df: Counter[str] = Counter()
    for pattern in patterns:
        df.update(features(pattern))
    index = {f: i for i, f in enumerate(sorted(df))}
    n = len(patterns)
    idf = np.array([math.log((1 + n) / (1 + df[f])) + 1.0 for f in index], dtype=np.float32)
    return FeatureSpace(index, idf)


def top_similar(query, base, k: int, batch: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Top-k cosine neighbours (indices, sims), rows sorted by decreasing sim."""

    idx = np.zeros((query.shape[0], k), dtype=np.int64)
    sim = np.zeros((query.shape[0], k), dtype=np.float32)
    base_t = base.T.tocsc()
    for s in range(0, query.shape[0], batch):
        block = (query[s:s + batch] @ base_t).toarray()
        part = np.argpartition(-block, min(k, block.shape[1] - 1), axis=1)[:, :k]
        vals = np.take_along_axis(block, part, 1)
        order = np.argsort(-vals, axis=1, kind="stable")
        idx[s:s + batch] = np.take_along_axis(part, order, 1)
        sim[s:s + batch] = np.take_along_axis(vals, order, 1)
    return idx, sim


def knn_predict(idx: np.ndarray, sim: np.ndarray, labels: np.ndarray, k: int, fallback: int) -> np.ndarray:
    out = np.full(idx.shape[0], fallback, dtype=np.int64)
    for r in range(idx.shape[0]):
        score: dict[int, float] = {}
        for j, s in zip(idx[r, :k], sim[r, :k]):
            if s <= 0:
                continue
            lab = int(labels[j])
            score[lab] = score.get(lab, 0.0) + float(s)
        if score:
            out[r] = max(score.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return out


# ---------------------------------------------------------------------------
# Field path: pattern taxonomy, relation program, placement, inheritance
# ---------------------------------------------------------------------------


def cosine_linkage_tree(x, block: int = 2048) -> Tree:
    """Average-linkage tree over the rows of an L2-normalised sparse matrix."""

    from scipy.cluster.hierarchy import linkage

    n = x.shape[0]
    condensed = np.empty(n * (n - 1) // 2, dtype=np.float64)
    xt = x.T.tocsc()
    for s in range(0, n, block):
        g = (x[s:s + block] @ xt).toarray()
        for r in range(g.shape[0]):
            i = s + r
            if i == n - 1:
                break
            # condensed index of (i, i+1..n-1)
            off = n * i - i * (i + 1) // 2
            condensed[off:off + n - i - 1] = 1.0 - np.clip(g[r, i + 1:], 0.0, 1.0)
    return tree_from_linkage(linkage(condensed, method="average"), n)


def overlap_linkage_tree(sets: list[set[int]], words, block: int = 2048, weight: float = 0.5) -> Tree:
    """Average-linkage tree mixing fact overlap and word similarity.

    Fact similarity = |F_p & F_q| / min(|F_p|, |F_q|) over (left, right) pairs;
    distance = 1 - (weight * fact + (1 - weight) * word cosine).
    """

    from scipy.cluster.hierarchy import linkage
    from scipy.sparse import csr_matrix

    n = len(sets)
    vocab: dict[int, int] = {}
    rows, cols = [], []
    for r, s in enumerate(sets):
        for f in s:
            rows.append(r)
            cols.append(vocab.setdefault(f, len(vocab)))
    inc = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, max(1, len(vocab))))
    sizes = np.array([max(1, len(s)) for s in sets], dtype=np.float64)
    inc_t = inc.T.tocsc()
    words_t = words.T.tocsc()
    condensed = np.empty(n * (n - 1) // 2, dtype=np.float64)
    for s in range(0, n, block):
        shared = (inc[s:s + block] @ inc_t).toarray()
        wsim = np.clip((words[s:s + block] @ words_t).toarray(), 0.0, 1.0)
        for r in range(shared.shape[0]):
            i = s + r
            if i == n - 1:
                break
            j = np.arange(i + 1, n)
            fact = shared[r, i + 1:] / np.minimum(sizes[i], sizes[j])
            off = n * i - i * (i + 1) // 2
            condensed[off:off + n - i - 1] = 1.0 - (weight * fact + (1 - weight) * wsim[r, i + 1:])
    return tree_from_linkage(linkage(condensed, method="average"), n)


class LCA:
    """O(1) lowest common ancestor by Euler tour + sparse-table range minimum."""

    def __init__(self, tree: Tree) -> None:
        euler, depth_seq = [], []
        first = np.zeros(tree.size, dtype=np.int64)
        stack = [(0, 0)]
        while stack:
            node, child_i = stack.pop()
            if child_i == 0:
                first[node] = len(euler)
            euler.append(node)
            depth_seq.append(int(tree.depth[node]))
            kids = tree.children[node]
            if child_i < len(kids):
                stack.append((node, child_i + 1))
                stack.append((kids[child_i], 0))
        self.euler = np.asarray(euler)
        self.first = first
        d = np.asarray(depth_seq)
        levels = [np.arange(len(d))]
        span = 1
        while 2 * span <= len(d):
            prev = levels[-1]
            left, right = prev[:-span], prev[span:]
            levels.append(np.where(d[left] <= d[right], left, right))
            span *= 2
        self.table = levels
        self.d = d

    def of_set(self, nodes: np.ndarray) -> int:
        pos = self.first[nodes]
        lo, hi = int(pos.min()), int(pos.max())
        k = int(math.floor(math.log2(hi - lo + 1)))
        a, b = self.table[k][lo], self.table[k][hi - (1 << k) + 1]
        best = a if self.d[a] <= self.d[b] else b
        return int(self.euler[best])


@dataclass
class FieldTaxonomy:
    tree: Tree
    lca: LCA
    labels_at: np.ndarray  # (nodes,) inherited relation at every node
    overrides: int

    def predict(self, idx: np.ndarray, sim: np.ndarray, tau: float, m: int, fallback: int) -> np.ndarray:
        """Place each query at the smallest family containing its most consistent leaves."""

        out = np.full(idx.shape[0], fallback, dtype=np.int64)
        leaves = self.tree.leaves
        for r in range(idx.shape[0]):
            top = float(sim[r, 0])
            if top <= 0:
                out[r] = int(self.labels_at[0])
                continue
            keep = idx[r, :m][sim[r, :m] >= top - tau]
            node = self.lca.of_set(leaves[keep])
            out[r] = int(self.labels_at[node])
        return out


def field_taxonomy(tree: Tree, labels: np.ndarray, values: int) -> FieldTaxonomy:
    observed = labels.reshape(-1, 1).astype(int)
    program = compress(tree, observed, values)
    return FieldTaxonomy(tree, LCA(tree), program.labels[0], program.entries())


# ---------------------------------------------------------------------------
# Baselines: softmax regression on the same features, bge-small kNN
# ---------------------------------------------------------------------------


def _torch_sparse(x):
    import torch

    x = x.tocoo()
    indices = torch.tensor(np.vstack([x.row, x.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(indices, torch.tensor(x.data, dtype=torch.float32), x.shape).coalesce()


def softmax_regression(x_train, y_train: np.ndarray, classes: int, weight_decay: float,
                       steps: int = 200, seed: int = 0):
    """Full-batch multinomial logistic regression on the sparse IDF features."""

    import torch

    torch.manual_seed(seed)
    xs = _torch_sparse(x_train)
    y = torch.tensor(y_train, dtype=torch.long)
    w = torch.zeros(x_train.shape[1], classes, requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(torch.sparse.mm(xs, w) + b, y)
        loss.backward()
        opt.step()
    return w.detach(), b.detach()


def softmax_predict(model, x) -> np.ndarray:
    import torch

    w, b = model
    with torch.no_grad():
        return (torch.sparse.mm(_torch_sparse(x), w) + b).argmax(1).numpy()


def bge_embed(texts: list[str], model_dir: str, batch: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval()
    out = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tok(texts[s:s + batch], padding=True, truncation=True, max_length=32, return_tensors="pt")
            cls = model(**enc).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(cls, dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 384), dtype=np.float32)


def pattern_sentence(pattern: str) -> str:
    body = pattern.replace(ENT, "something") if pattern else ""
    return f"X {body} Y".replace("  ", " ")


def dense_top(query: np.ndarray, base: np.ndarray, k: int, batch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    idx = np.zeros((query.shape[0], k), dtype=np.int64)
    sim = np.zeros((query.shape[0], k), dtype=np.float32)
    for s in range(0, query.shape[0], batch):
        block = query[s:s + batch] @ base.T
        part = np.argpartition(-block, k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(block, part, 1)
        order = np.argsort(-vals, axis=1, kind="stable")
        idx[s:s + batch] = np.take_along_axis(part, order, 1)
        sim[s:s + batch] = np.take_along_axis(vals, order, 1)
    return idx, sim


# ---------------------------------------------------------------------------
# Test 3: facts first, words second
# ---------------------------------------------------------------------------


def relation_fact_table(corpus: Corpus, patterns: np.ndarray, labels: np.ndarray) -> dict[int, dict[int, set[int]]]:
    """Relation -> left entity -> right entities, from labelled patterns' occurrences."""

    table: dict[int, dict[int, set[int]]] = {}
    for p, lab in zip(patterns, labels):
        rel = table.setdefault(int(lab), {})
        for i in occurrences_of(corpus, int(p)):
            rel.setdefault(int(corpus.occ_left[i]), set()).add(int(corpus.occ_right[i]))
    return table


def fact_decision(corpus: Corpus, p: int, facts_by_key: dict[int, list[tuple[int, set[int]]]],
                  eps: float, pi: float) -> tuple[int, float, int]:
    """Best relation by shared-subject LLR: (relation, llr, shared subjects); (-1, 0, 0) if none."""

    agree_w, disagree_w = llr_weights(eps, pi)
    n: Counter[int] = Counter()
    a: Counter[int] = Counter()
    for key, objs in key_table(corpus, p).items():
        for rel, rel_objs in facts_by_key.get(key, ()):
            n[rel] += 1
            a[rel] += bool(objs & rel_objs)
    if not n:
        return -1, 0.0, 0
    best = max(n, key=lambda r: (a[r] * agree_w + (n[r] - a[r]) * disagree_w, a[r], -r))
    return best, a[best] * agree_w + (n[best] - a[best]) * disagree_w, n[best]


def facts_by_key(table: dict[int, dict[int, set[int]]]) -> dict[int, list[tuple[int, set[int]]]]:
    out: dict[int, list[tuple[int, set[int]]]] = {}
    for rel, keys in table.items():
        for key, objs in keys.items():
            out.setdefault(key, []).append((rel, objs))
    return out


def macro_f1(pred: np.ndarray, gold: np.ndarray) -> float:
    classes = np.unique(gold)
    f1s = []
    for c in classes:
        tp = float(((pred == c) & (gold == c)).sum())
        fp = float(((pred == c) & (gold != c)).sum())
        fn = float(((pred != c) & (gold == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def correct(pred: np.ndarray, gold_sets: list[frozenset[int]]) -> np.ndarray:
    return np.array([int(p) in g for p, g in zip(pred, gold_sets)], dtype=bool)
