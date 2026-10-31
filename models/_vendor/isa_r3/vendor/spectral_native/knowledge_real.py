"""Real knowledge for the spectral field: WordNet, real attribute data, multi-tree induction.

Builds supplied taxonomies from WordNet 3.0 noun hypernyms, joins them to real
attribute data (AwA2 animal attributes, UCI Zoo) and to WordNet gloss features,
and provides:

- taxonomy trees for any entity set (first-hypernym paths, unary chains
  collapsed, internal entities given their own leaf);
- multi-tree induction: several cross-cutting hierarchies induced from facts,
  attributes assigned to the tree that describes them most cheaply, alternated
  to a fixed point, plus conjunctive product families over tree cuts;
- strong baselines: tuned k-nearest neighbours, a tuned categorical matrix
  factorization with weight decay and early stopping, and a Poincare-ball
  prototype model.

Family programs themselves come from ``spectral_native.knowledge``.
"""

from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .knowledge import UNKNOWN, Tree, compress, predict, tree_from_linkage

# ---------------------------------------------------------------------------
# WordNet parsing
# ---------------------------------------------------------------------------

ANIMAL = "00015388"
MAMMAL = "01861778"
FISH = "02512053"
ORGANISM = "00004475"
ENTITY = "00001740"


def parse_wordnet_nouns(directory: Path) -> dict[str, dict[str, Any]]:
    """offset -> {lemmas, lexfile, hypernyms, gloss} from ``data.noun``."""

    out: dict[str, dict[str, Any]] = {}
    with open(directory / "data.noun", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("  "):
                continue
            head, _, gloss = line.partition(" | ")
            fields = head.split()
            offset, lexfile = fields[0], int(fields[1])
            count = int(fields[3], 16)
            lemmas = [fields[4 + 2 * i] for i in range(count)]
            cursor = 4 + 2 * count
            pointers = int(fields[cursor])
            cursor += 1
            hypernyms = []
            for _ in range(pointers):
                symbol, target, pos = fields[cursor], fields[cursor + 1], fields[cursor + 2]
                cursor += 4
                if symbol in ("@", "@i") and pos == "n":
                    hypernyms.append(target)
            out[offset] = {"lemmas": lemmas, "lexfile": lexfile, "hypernyms": hypernyms, "gloss": gloss.strip()}
    return out


def parse_index_noun(directory: Path) -> dict[str, list[str]]:
    """lemma -> synset offsets in sense order."""

    out: dict[str, list[str]] = {}
    with open(directory / "index.noun", encoding="latin-1") as handle:
        for line in handle:
            if line.startswith("  "):
                continue
            fields = line.split()
            lemma, synsets = fields[0], int(fields[2])
            out[lemma] = fields[-synsets:]
    return out


def ancestors_closure(nouns: dict[str, dict[str, Any]], offset: str) -> set[str]:
    seen, stack = set(), [offset]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(nouns[node]["hypernyms"])
    return seen


def parent_within(nouns: dict[str, dict[str, Any]], offset: str, root: str, closure_cache: dict) -> str | None:
    """First hypernym whose ancestor closure contains ``root`` (a tree path to root)."""

    for hyper in nouns[offset]["hypernyms"]:
        if hyper not in closure_cache:
            closure_cache[hyper] = root in ancestors_closure(nouns, hyper)
        if closure_cache[hyper]:
            return hyper
    return None


def map_name(nouns, index, name: str, root: str) -> str | None:
    """First noun sense of ``name`` that lies under ``root``."""

    key = name.lower().replace("+", "_").replace(" ", "_")
    for offset in index.get(key, []):
        if root in ancestors_closure(nouns, offset):
            return offset
    return None


# ---------------------------------------------------------------------------
# Taxonomy trees for entity sets
# ---------------------------------------------------------------------------


def taxonomy_tree(parent_of: dict[str, str | None], entities: list[str]) -> tuple[Tree, dict[str, int]]:
    """Tree over the union of the entities' root paths.

    Unary internal chains are collapsed.  An entity that is also an ancestor of
    another entity gets its own leaf child, so every entity is a leaf.
    Returns the tree and a map from kept synset to node id.
    """

    nodes: set[str] = set()
    for e in entities:
        node: str | None = e
        while node is not None:
            nodes.add(node)
            node = parent_of[node]
    children: dict[str, list[str]] = {n: [] for n in nodes}
    roots = []
    for n in nodes:
        p = parent_of[n]
        if p is None:
            roots.append(n)
        else:
            children[p].append(n)
    if len(roots) != 1:
        raise ValueError(f"expected one root, found {len(roots)}")
    entity_set = set(entities)
    # leaf keys: ("leaf", entity) for every entity; synset keys for internal
    kids: dict[Any, list[Any]] = {}

    def build(n: str) -> list[Any]:
        out = []
        for c in sorted(children[n]):
            out.append(c)
        if n in entity_set and children[n]:
            out.append(("leaf", n))
        return out

    for n in nodes:
        kids[n] = build(n)
    for e in entities:
        if children[e]:
            kids[("leaf", e)] = []
    # collapse unary chains (keep root and leaves)
    def resolve(n: Any) -> Any:
        while not (isinstance(n, tuple)) and len(kids[n]) == 1 and n not in entity_set:
            n = kids[n][0]
        return n

    root = roots[0]
    order: list[Any] = [root]
    parent_idx = [-1]
    id_of: dict[Any, int] = {root: 0}
    queue = [root]
    for n in queue:
        for c in kids[n]:
            c = resolve(c)
            id_of[c] = len(order)
            order.append(c)
            parent_idx.append(id_of[n])
            queue.append(c)
    parent = np.array(parent_idx)
    child_lists: list[list[int]] = [[] for _ in order]
    for i, p in enumerate(parent_idx):
        if p >= 0:
            child_lists[p].append(i)
    depth = np.zeros(len(order), dtype=int)
    for i in range(1, len(order)):
        depth[i] = depth[parent[i]] + 1
    leaf_key = lambda e: ("leaf", e) if children[e] else e  # noqa: E731
    leaves = np.array([id_of[leaf_key(e)] for e in entities])
    if any(child_lists[leaf] for leaf in leaves):
        raise ValueError("entity leaf has children")
    synset_node = {k: v for k, v in id_of.items() if isinstance(k, str)}
    return Tree(parent, child_lists, depth, leaves), synset_node


def two_level_tree(groups: np.ndarray) -> Tree:
    """Root -> one node per group -> entities (e.g. the Zoo class labels)."""

    labels = sorted(set(groups.tolist()))
    parent = [-1] + [0] * len(labels)
    group_node = {g: i + 1 for i, g in enumerate(labels)}
    leaves = []
    for g in groups.tolist():
        parent.append(group_node[g])
        leaves.append(len(parent) - 1)
    parent_arr = np.array(parent)
    children: list[list[int]] = [[] for _ in parent]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    depth = np.array([0] + [1] * len(labels) + [2] * len(groups))
    return Tree(parent_arr, children, depth, np.array(leaves))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class RealDataset:
    name: str
    entities: list[str]
    attributes: list[str]
    facts: np.ndarray  # (entities, attributes) small integers
    values: np.ndarray  # per attribute number of values
    trees: dict[str, Tree]  # supplied taxonomies
    notes: dict[str, Any]


def load_prepared(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt") as handle:
        return json.load(handle)


def _tree_from_prepared(blob: dict[str, Any]) -> Tree:
    parent = np.array(blob["parent"])
    children: list[list[int]] = [[] for _ in parent]
    for i, p in enumerate(parent.tolist()):
        if p >= 0:
            children[p].append(i)
    return Tree(parent, children, np.array(blob["depth"]), np.array(blob["leaves"]))


def dataset_from_prepared(blob: dict[str, Any]) -> RealDataset:
    trees = {name: _tree_from_prepared(t) for name, t in blob["trees"].items()}
    facts = np.array(blob["facts"], dtype=int)
    return RealDataset(blob["name"], blob["entities"], blob["attributes"], facts,
                       np.array(blob["values"], dtype=int), trees, blob.get("notes", {}))


def tree_to_blob(tree: Tree) -> dict[str, Any]:
    return {"parent": tree.parent.tolist(), "depth": tree.depth.tolist(), "leaves": tree.leaves.tolist()}


# ---------------------------------------------------------------------------
# Description lengths and single-tree programs
# ---------------------------------------------------------------------------


def attribute_program_bits(tree: Tree, observed: np.ndarray, values: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-attribute program bits and the tree's inheritance predictions."""

    program = compress(tree, observed, values)
    per_entry = math.log2(tree.size) + math.log2(values)
    bits = np.array([len(t) * per_entry for t in program.overrides])
    return bits, predict(tree, program)


def enumerative_program_bits(tree: Tree, observed: np.ndarray, values: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-attribute program bits under an enumerative code for the override set.

    Uniform coding spends log2 N + log2 V bits per override.  Sending which m of
    the N nodes carry overrides costs log2 C(N, m) + log2 N (for m), plus
    log2 V for the root value and log2(V - 1) per other override (a change
    must differ from the inherited value).  This is within a few bits of the
    information content of the override set.
    """

    from scipy.special import gammaln

    program = compress(tree, observed, values)
    n = tree.size
    bits = []
    for table in program.overrides:
        m = len(table)
        log_choose = (gammaln(n + 1) - gammaln(m + 1) - gammaln(n - m + 1)) / math.log(2)
        bits.append(log_choose + math.log2(n) + math.log2(values) + (m - 1) * math.log2(max(values - 1, 1)))
    return np.array(bits), predict(tree, program)


def leave_one_attribute_out(observed: np.ndarray, values: int, method: str = "average") -> np.ndarray:
    """Predict each attribute from a tree induced from all other attributes.

    Removes the selection bias of predicting an attribute with a tree built
    from its own training cells, while keeping almost all of the evidence.
    """

    attrs = observed.shape[1]
    out = np.zeros_like(observed)
    for a in range(attrs):
        others = np.array([b for b in range(attrs) if b != a])
        tree, _ = induce_tree_with(observed[:, others], method)
        out[:, a] = attribute_program_bits(tree, observed[:, [a]], values)[1][:, 0]
    return out


def flat_bits(observed: np.ndarray, values: int) -> np.ndarray:
    return (observed != UNKNOWN).sum(0) * math.log2(values)


def tree_shape_bits(tree: Tree, entities: int) -> float:
    """Bits to transmit an induced tree: shape (~2 bits/node) plus leaf labels."""

    return 2.0 * tree.size + entities * math.log2(max(2, entities))


# ---------------------------------------------------------------------------
# Multi-tree induction
# ---------------------------------------------------------------------------


def _linkage(observed: np.ndarray, method: str = "average"):
    from scipy.cluster.hierarchy import linkage

    from .knowledge import hamming_distances

    # float32 rounding can leave identical rows at -1e-7; distances are >= 0
    condensed = np.clip(hamming_distances(observed).astype(np.float64), 0.0, None)
    if method == "ward":
        # ward needs Euclidean geometry; use it on the mismatch metric as is
        return linkage(condensed, method="ward")
    return linkage(condensed, method=method)


def induce_tree_with(observed: np.ndarray, method: str = "average"):
    link = _linkage(observed, method)
    return tree_from_linkage(link, observed.shape[0]), link


def attribute_similarity_groups(observed: np.ndarray, groups: int, rng: np.random.Generator) -> np.ndarray:
    """Initial attribute groups by co-variation (average linkage on 1 - |agreement correlation|)."""

    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    attrs = observed.shape[1]
    dist = np.zeros((attrs, attrs))
    for a in range(attrs):
        for b in range(a + 1, attrs):
            both = (observed[:, a] != UNKNOWN) & (observed[:, b] != UNKNOWN)
            if both.sum() < 3:
                dist[a, b] = dist[b, a] = 1.0
                continue
            x, y = observed[both, a], observed[both, b]
            # normalized mutual information on co-observed cells
            joint = {}
            for u, v in zip(x.tolist(), y.tolist()):
                joint[(u, v)] = joint.get((u, v), 0) + 1
            n = both.sum()
            px = np.bincount(x) / n
            py = np.bincount(y) / n
            mi = sum(c / n * math.log((c / n) / (px[u] * py[v])) for (u, v), c in joint.items())
            hx = -sum(p * math.log(p) for p in px if p > 0)
            hy = -sum(p * math.log(p) for p in py if p > 0)
            nmi = mi / max(1e-9, math.sqrt(hx * hy)) if hx > 0 and hy > 0 else 0.0
            dist[a, b] = dist[b, a] = min(1.0, max(0.0, 1.0 - nmi))
    if groups <= 1:
        return np.zeros(attrs, dtype=int)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=groups, criterion="maxclust")
    labels = labels - 1
    # guarantee every group has at least two attributes
    for g in range(groups):
        if (labels == g).sum() < 2:
            labels[rng.choice(attrs, size=2, replace=False)] = g
    return labels


def coagreement_similarity(observed: np.ndarray, rng: np.random.Generator, max_pairs: int = 400_000,
                           neighbours: int = 10) -> np.ndarray:
    """Attribute x attribute correlation of pairwise-agreement indicators.

    For attribute a and entity pair (i, j), g_a(i, j) = [x_ia == x_ja].  Two
    attributes generated on the same hierarchy agree on the same pairs (close
    relatives in that hierarchy), so their agreement vectors correlate.
    Attributes generated on independent hierarchies have uncorrelated
    agreement vectors, even when each is strongly structured.  Value-level
    mutual information misses this: two attributes that mutate at different
    nodes of the same tree share little information about each other's values.

    Pairs: all pairs when affordable, otherwise random pairs plus each entity's
    nearest neighbours by overall mismatch (where the tree signal lives).
    """

    entities, attrs = observed.shape
    total = entities * (entities - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(entities, k=1)
    else:
        i = rng.integers(entities, size=max_pairs)
        j = rng.integers(entities, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
        from .knowledge import hamming_distances
        from scipy.spatial.distance import squareform

        if entities <= 6000:
            dist = squareform(np.clip(hamming_distances(observed), 0, None))
            np.fill_diagonal(dist, np.inf)
            near = np.argsort(dist, axis=1)[:, :neighbours]
            ni = np.repeat(np.arange(entities), neighbours)
            i = np.concatenate([i, ni])
            j = np.concatenate([j, near.reshape(-1)])
    xi, xj = observed[i], observed[j]
    both = (xi != UNKNOWN) & (xj != UNKNOWN)
    agree = ((xi == xj) & both).astype(np.float32)
    # fill unobserved pairs with the attribute's mean agreement
    mean = agree.sum(0) / np.maximum(both.sum(0), 1)
    agree = np.where(both, agree, mean[None, :])
    agree -= agree.mean(0, keepdims=True)
    norm = np.sqrt((agree ** 2).sum(0)) + 1e-9
    return (agree.T @ agree) / np.outer(norm, norm)


def coagreement_groups(observed: np.ndarray, groups: int, rng: np.random.Generator) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    attrs = observed.shape[1]
    if groups <= 1:
        return np.zeros(attrs, dtype=int)
    corr = coagreement_similarity(observed, rng)
    dist = np.clip(1.0 - corr, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=groups, criterion="maxclust") - 1
    for g in range(groups):
        if (labels == g).sum() < 2:
            labels[rng.choice(attrs, size=2, replace=False)] = g
    return labels


@dataclass
class MultiTreeResult:
    trees: list[Tree]
    links: list[Any]
    assignment: np.ndarray  # per attribute: tree index, or -1 for flat
    bits: dict[str, float]
    predictions: np.ndarray
    iterations: int
    history: list[list[int]]


def multi_tree_induction(
    observed: np.ndarray,
    values: int,
    groups: int,
    rng: np.random.Generator,
    iterations: int = 6,
    min_attributes: int = 2,
    init: str = "nmi",
    initial_labels: np.ndarray | None = None,
) -> MultiTreeResult:
    """Alternate: build one tree per attribute group; move each attribute to its cheapest tree.

    An attribute's cost under a tree is its minimal override program in bits.
    Attributes cheaper as a flat column stay in their group for tree building
    but are stored flat.  Stops at a fixed point or after ``iterations``.
    """

    entities, attrs = observed.shape
    if initial_labels is not None:
        labels = np.asarray(initial_labels).copy()
    elif init == "coagreement":
        labels = coagreement_groups(observed, groups, rng)
    else:
        labels = attribute_similarity_groups(observed, groups, rng)
    history = [labels.tolist()]
    trees: list[Tree] = []
    links: list[Any] = []
    for it in range(iterations):
        trees, links = [], []
        for g in range(groups):
            cols = np.where(labels == g)[0]
            if cols.size < min_attributes:
                cols = np.arange(attrs)
            tree, link = induce_tree_with(observed[:, cols])
            trees.append(tree)
            links.append(link)
        costs = np.stack([attribute_program_bits(t, observed, values)[0] for t in trees], axis=0)
        new = costs.argmin(0)
        if np.array_equal(new, labels):
            break
        labels = new
        history.append(labels.tolist())
    costs, preds = [], []
    for t in trees:
        c, p = attribute_program_bits(t, observed, values)
        costs.append(c)
        preds.append(p)
    costs_arr = np.stack(costs, axis=0)
    flat = flat_bits(observed, values)
    best_tree = costs_arr.argmin(0)
    best_cost = costs_arr.min(0)
    use_flat = flat < best_cost
    assignment = np.where(use_flat, -1, best_tree)
    predictions = np.zeros_like(observed)
    for a in range(attrs):
        predictions[:, a] = preds[best_tree[a]][:, a]
    shape = sum(tree_shape_bits(t, entities) for t in trees)
    bits = {
        "programs": float(np.minimum(best_cost, flat).sum() + attrs * math.log2(groups + 1)),
        "tree_shapes": float(shape),
    }
    bits["total"] = bits["programs"] + bits["tree_shapes"]
    return MultiTreeResult(trees, links, assignment, bits, predictions, it + 1, history)


def conjunctive_explanations(
    observed: np.ndarray,
    values: int,
    links: list[Any],
    cuts: tuple[int, ...] = (2, 4, 8, 16),
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Per attribute: best product-family table over cuts of two induced trees.

    Returns bits, predictions and the chosen (tree pair, cuts) per attribute.
    """

    from scipy.cluster.hierarchy import fcluster

    entities, attrs = observed.shape
    best_bits = np.full(attrs, np.inf)
    best_pred = np.zeros_like(observed)
    choice: list[dict[str, Any]] = [{} for _ in range(attrs)]
    parts = [{c: fcluster(l, t=c, criterion="maxclust") for c in cuts} for l in links]
    exc_bits = math.log2(entities) + math.log2(values)
    for i in range(len(links)):
        for j in range(i + 1, len(links)):
            for c1 in cuts:
                for c2 in cuts:
                    key = parts[i][c1] * (c2 + 1) + parts[j][c2]
                    for a in range(attrs):
                        known = observed[:, a] != UNKNOWN
                        table: dict[int, np.ndarray] = {}
                        for k, v in zip(key[known].tolist(), observed[known, a].tolist()):
                            table.setdefault(k, np.zeros(values, dtype=int))[v] += 1
                        mode = {k: int(cnt.argmax()) for k, cnt in table.items()}
                        pred = np.array([mode.get(k, 0) for k in key.tolist()])
                        errors = int((pred[known] != observed[known, a]).sum())
                        bits = len(table) * (math.log2(c1) + math.log2(c2) + math.log2(values)) + errors * exc_bits + 8
                        if bits < best_bits[a]:
                            best_bits[a] = bits
                            best_pred[:, a] = pred
                            choice[a] = {"trees": [i, j], "cuts": [c1, c2]}
    return best_bits, best_pred, choice


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------


def knn_scores(observed: np.ndarray) -> np.ndarray:
    known = (observed != UNKNOWN).astype(np.float32)
    agree = np.zeros((observed.shape[0], observed.shape[0]), dtype=np.float32)
    for v in range(int(observed.max()) + 1):
        hot = (observed == v).astype(np.float32)
        agree += hot @ hot.T
    overlap = known @ known.T
    sim = agree / np.maximum(overlap, 1.0)
    np.fill_diagonal(sim, -1.0)
    return sim


def knn_predict_cells(observed: np.ndarray, sim: np.ndarray, values: int, k: int, power: float) -> np.ndarray:
    """Predict every cell by a similarity-weighted vote of the k most similar rows observing it."""

    entities, attrs = observed.shape
    order = np.argsort(-sim, axis=1)
    out = np.zeros_like(observed)
    for i in range(entities):
        ranked = order[i]
        for a in range(attrs):
            col = observed[ranked, a]
            mask = col != UNKNOWN
            idx = ranked[mask][:k]
            if idx.size == 0:
                out[i, a] = 0
                continue
            w = np.maximum(sim[i, idx], 0.0) ** power + 1e-9
            out[i, a] = int(np.bincount(observed[idx, a], weights=w, minlength=values).argmax())
    return out


def tuned_knn(observed: np.ndarray, values: int, rng: np.random.Generator,
              ks=(1, 3, 5, 10, 20, 40), powers=(0.0, 1.0, 4.0)) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose k and weighting on an inner validation split of the observed cells."""

    known = np.argwhere(observed != UNKNOWN)
    val = known[rng.random(known.shape[0]) < 0.1]
    inner = observed.copy()
    inner[val[:, 0], val[:, 1]] = UNKNOWN
    sim = knn_scores(inner)
    best = (-1.0, None)
    for k in ks:
        for p in powers:
            pred = knn_predict_cells(inner, sim, values, k, p)
            acc = float((pred[val[:, 0], val[:, 1]] == observed[val[:, 0], val[:, 1]]).mean())
            if acc > best[0]:
                best = (acc, (k, p))
    k, p = best[1]
    pred = knn_predict_cells(observed, knn_scores(observed), values, k, p)
    return pred, {"k": k, "power": p, "inner_validation_accuracy": best[0]}


def _factor_model(entities: int, attrs: int, values: int, dim: int, kind: str):
    import torch
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.entity = nn.Parameter(torch.randn(entities, dim) * 0.1)
            self.proto = nn.Parameter(torch.randn(attrs, values, dim) * 0.1)
            self.bias = nn.Parameter(torch.zeros(attrs, values))
            self.scale = nn.Parameter(torch.tensor(1.0))

        def forward(self, e, a):
            if kind == "euclidean":
                return torch.einsum("nd,nvd->nv", self.entity[e], self.proto[a]) + self.bias[a]
            # Poincare ball (curvature -1): points via the exponential map at 0
            def expmap(u):
                norm = u.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                return torch.tanh(norm) * u / norm * 0.999

            x = expmap(self.entity[e])[:, None, :]
            p = expmap(self.proto[a])
            sq = ((x - p) ** 2).sum(-1)
            denom = (1 - (x ** 2).sum(-1)) * (1 - (p ** 2).sum(-1))
            dist = torch.acosh(1 + 2 * sq / denom.clamp_min(1e-6))
            return -self.scale.abs() * dist + self.bias[a]

    return Model()


def train_factor(observed: np.ndarray, values_per_attr: np.ndarray, kind: str, dim: int, wd: float,
                 seed: int, max_steps: int = 3000, patience: int = 200) -> tuple[Any, float, int]:
    """Categorical factorization with weight decay and early stopping on an inner split."""

    import torch
    import torch.nn.functional as nnf

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    entities, attrs = observed.shape
    values = int(values_per_attr.max())
    known = np.argwhere(observed != UNKNOWN)
    is_val = rng.random(known.shape[0]) < 0.1
    tr, va = known[~is_val], known[is_val]
    mask = torch.full((attrs, values), float("-inf"))
    for a, v in enumerate(values_per_attr.tolist()):
        mask[a, :v] = 0.0
    model = _factor_model(entities, attrs, values, dim, kind)
    opt = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=wd)
    e_tr, a_tr = torch.tensor(tr[:, 0]), torch.tensor(tr[:, 1])
    y_tr = torch.tensor(observed[tr[:, 0], tr[:, 1]])
    e_va, a_va = torch.tensor(va[:, 0]), torch.tensor(va[:, 1])
    y_va = torch.tensor(observed[va[:, 0], va[:, 1]])
    best, best_state, best_step, since = -1.0, None, 0, 0
    for step in range(max_steps):
        logits = model(e_tr, a_tr) + mask[a_tr]
        loss = nnf.cross_entropy(logits, y_tr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 10 == 0:
            with torch.no_grad():
                acc = float(((model(e_va, a_va) + mask[a_va]).argmax(-1) == y_va).float().mean())
            if acc > best:
                best, best_step, since = acc, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since += 10
                if since >= patience:
                    break
    model.load_state_dict(best_state)
    model.mask = mask
    return model, best, best_step


def tuned_factor(observed: np.ndarray, values_per_attr: np.ndarray, kind: str, seed: int,
                 dims=(2, 4, 8, 16), wds=(1e-4, 1e-3, 1e-2, 1e-1)) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    best = (-1.0, None, None)
    for dim in dims:
        for wd in wds:
            model, acc, step = train_factor(observed, values_per_attr, kind, dim, wd, seed)
            if acc > best[0]:
                best = (acc, model, {"dim": dim, "weight_decay": wd, "best_step": step})
    model = best[1]
    entities, attrs = observed.shape
    e = torch.arange(entities).repeat_interleave(attrs)
    a = torch.arange(attrs).repeat(entities)
    with torch.no_grad():
        pred = (model(e, a) + model.mask[a]).argmax(-1).view(entities, attrs).numpy()
    params = sum(p.numel() for p in model.parameters())
    return pred, {"inner_validation_accuracy": best[0], "parameters": params, **best[2]}


# ---------------------------------------------------------------------------
# Synthetic cross-cutting hierarchies (for multi-tree discovery)
# ---------------------------------------------------------------------------


def attach_entities(tree: Tree, group_of_entity: np.ndarray) -> Tree:
    """New tree whose leaves are entities hung under the leaves of ``tree``.

    ``group_of_entity[i]`` is the index (into ``tree.leaves``) of entity i's leaf.
    """

    parent = tree.parent.tolist()
    depth = tree.depth.tolist()
    leaves = []
    for g in group_of_entity.tolist():
        host = int(tree.leaves[g])
        parent.append(host)
        depth.append(depth[host] + 1)
        leaves.append(len(parent) - 1)
    children: list[list[int]] = [[] for _ in parent]
    for i, p in enumerate(parent):
        if p >= 0:
            children[p].append(i)
    return Tree(np.array(parent), children, np.array(depth), np.array(leaves))


def cross_cutting_kb(rng: np.random.Generator, exception: float, values: int = 8) -> dict[str, Any]:
    """Entities sit in two independent hierarchies; attributes follow one, both, or neither.

    T1: 4-ary depth 5 (1024 entities, one per leaf).  T2: 4-ary depth 3 (64
    groups), entities assigned uniformly at random, so T2 cuts across T1.
    Attributes: 20 inherited on T1, 20 inherited on T2, 12 conjunctive tables
    over (T1 family at depth k1, T2 family at depth k2), 8 random.
    """

    from .knowledge import balanced_tree, tree_generated_values

    t1 = balanced_tree(4, 5)
    t2 = balanced_tree(4, 3)
    n = t1.leaves.shape[0]
    group = rng.integers(t2.leaves.shape[0], size=n)
    t2_entities = attach_entities(t2, group)
    cols, kinds, meta = [], [], []

    def exceptions(col: np.ndarray) -> np.ndarray:
        flip = rng.random(n) < exception
        noise = rng.integers(values, size=n)
        return np.where(flip, noise, col)

    for _ in range(20):
        vals, _used = tree_generated_values(t1, values, 0.15, exception, rng)
        cols.append(vals)
        kinds.append("t1")
        meta.append({})
    for _ in range(20):
        region_vals, _used = tree_generated_values(t2, values, 0.15, 0.0, rng)
        cols.append(exceptions(region_vals[group]))
        kinds.append("t2")
        meta.append({})
    code1 = t1.code[t1.leaves]
    code2 = t2.code[t2.leaves][group]
    for _ in range(12):
        k1, k2 = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        a1 = code1 % (1 << (t1.bits * k1))
        a2 = code2 % (1 << (t2.bits * k2))
        table: dict[tuple[int, int], int] = {}
        col = np.zeros(n, dtype=int)
        for i in range(n):
            key = (int(a1[i]), int(a2[i]))
            if key not in table:
                table[key] = int(rng.integers(values))
            col[i] = table[key]
        cols.append(exceptions(col))
        kinds.append("conj")
        meta.append({"levels": [k1, k2]})
    for _ in range(8):
        cols.append(rng.integers(values, size=n))
        kinds.append("random")
        meta.append({})
    return {
        "facts": np.stack(cols, axis=1),
        "kinds": np.array(kinds, dtype=object),
        "meta": meta,
        "t1": t1,
        "t2_entities": t2_entities,
        "partitions": {
            "t1": {k: code1 % (1 << (t1.bits * k)) for k in (1, 2)},
            "t2": {k: code2 % (1 << (t2.bits * k)) for k in (1, 2)},
        },
        "values": values,
    }


def conjunctive_from_partitions(
    observed: np.ndarray, values: int, first: dict[int, np.ndarray], second: dict[int, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Best product-family table over supplied partitions (oracle conjunctions)."""

    entities, attrs = observed.shape
    exc_bits = math.log2(entities) + math.log2(values)
    best_bits = np.full(attrs, np.inf)
    best_pred = np.zeros_like(observed)
    for k1, p1 in first.items():
        for k2, p2 in second.items():
            key = p1.astype(np.int64) * (int(p2.max()) + 1) + p2
            c1, c2 = len(set(p1.tolist())), len(set(p2.tolist()))
            for a in range(attrs):
                known = observed[:, a] != UNKNOWN
                table: dict[int, np.ndarray] = {}
                for k, v in zip(key[known].tolist(), observed[known, a].tolist()):
                    table.setdefault(k, np.zeros(values, dtype=int))[v] += 1
                mode = {k: int(c.argmax()) for k, c in table.items()}
                pred = np.array([mode.get(k, 0) for k in key.tolist()])
                errors = int((pred[known] != observed[known, a]).sum())
                bits = len(table) * (math.log2(c1) + math.log2(c2) + math.log2(values)) + errors * exc_bits + 8
                if bits < best_bits[a]:
                    best_bits[a] = bits
                    best_pred[:, a] = pred
    return best_bits, best_pred


def sequential_multi_tree(
    observed: np.ndarray,
    values: int,
    rng: np.random.Generator,
    max_trees: int = 4,
    refine: int = 3,
    code: str = "uniform",
    seeding: str = "flat",
) -> MultiTreeResult:
    """Explain away: each new tree is induced from the attributes no earlier tree explains.

    1. Build a tree from the currently unexplained attributes (initially all).
    2. An attribute is explained if its cheapest tree program beats a flat column.
    3. Keep the new tree only if total description length (programs, attribute
       tree choices, tree shapes) decreases; otherwise stop.
    4. Refine: rebuild each tree from the attributes that chose it.

    Structureless attributes never earn a tree, because a tree built from them
    does not reduce description length; a value-similarity clustering, by
    contrast, groups them together precisely because they resemble nothing.
    """

    entities, attrs = observed.shape
    flat = flat_bits(observed, values)
    trees: list[Tree] = []
    links: list[Any] = []
    bits_of = enumerative_program_bits if code == "enumerative" else attribute_program_bits

    def total(ts: list[Tree]) -> tuple[float, np.ndarray]:
        if not ts:
            return float(flat.sum()), np.full(attrs, np.inf)
        costs = np.stack([bits_of(t, observed, values)[0] for t in ts])
        best = np.minimum(costs.min(0), flat)
        dl = float(best.sum() + attrs * math.log2(len(ts) + 1) + sum(tree_shape_bits(t, entities) for t in ts))
        return dl, costs.min(0)

    current, best_cost = total(trees)
    unexplained = np.arange(attrs)
    history: list[list[int]] = []
    while len(trees) < max_trees and unexplained.size >= 2:
        tree, link = induce_tree_with(observed[:, unexplained])
        dl, cost = total(trees + [tree])
        if dl >= current:
            break
        trees.append(tree)
        links.append(link)
        current, best_cost = dl, cost
        if seeding == "ratio":
            # seed the next tree with the attributes explained worst relative
            # to their flat cost; a better code makes partial explanations
            # "cheaper than flat", so a flat threshold hides them
            ratio = np.minimum(best_cost, flat) / flat
            unexplained = np.where(ratio >= np.median(ratio))[0]
        else:
            unexplained = np.where(best_cost >= flat)[0]
        history.append(unexplained.tolist())
    for _ in range(refine if len(trees) > 1 else 0):
        costs = np.stack([bits_of(t, observed, values)[0] for t in trees])
        owner = costs.argmin(0)
        rebuilt, rebuilt_links = [], []
        for k in range(len(trees)):
            cols = np.where((owner == k) & (costs.min(0) < flat))[0]
            if cols.size < 2:
                rebuilt.append(trees[k])
                rebuilt_links.append(links[k])
                continue
            t, l = induce_tree_with(observed[:, cols])
            rebuilt.append(t)
            rebuilt_links.append(l)
        dl, _ = total(rebuilt)
        if dl >= current:
            break
        trees, links, current = rebuilt, rebuilt_links, dl
    if not trees:
        tree, link = induce_tree_with(observed)
        trees, links = [tree], [link]
    costs, preds = [], []
    for t in trees:
        c, p = bits_of(t, observed, values)
        costs.append(c)
        preds.append(p)
    costs_arr = np.stack(costs)
    best_tree = costs_arr.argmin(0)
    use_flat = flat < costs_arr.min(0)
    predictions = np.zeros_like(observed)
    for a in range(attrs):
        predictions[:, a] = preds[best_tree[a]][:, a]
    shape = sum(tree_shape_bits(t, entities) for t in trees)
    programs = float(np.minimum(costs_arr.min(0), flat).sum() + attrs * math.log2(len(trees) + 1))
    return MultiTreeResult(trees, links, np.where(use_flat, -1, best_tree),
                           {"programs": programs, "tree_shapes": float(shape), "total": programs + shape},
                           predictions, len(trees), history)
