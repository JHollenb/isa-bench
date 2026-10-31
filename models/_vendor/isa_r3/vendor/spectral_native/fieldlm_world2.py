"""Field LM on the track-C world with the parsing learned from raw text.

Track C (``fieldlm_world.FieldLM``) was handed the lexical type of every token,
the relation expressed by each declarative template, which relations are
hierarchies, and the value ids (read from token names).  This module removes
all of that.  The only inputs are the training units as token strings.

Induction pipeline (every stage is exact, combinatorial and data-driven):

1. Terminator: the token that ends every training unit.
2. Token classes by substitution.  (a) Frame substitution: tokens that fill
   the same hole in an otherwise identical sentence are merged.  (b) Skeleton
   minimal pairs: after replacing class members by their class, sentence
   skeletons that differ in exactly one position merge the two symbols at
   that position.  Iterated to a fixed point.  Classes with more than one
   member are open (slot) classes; singletons are function tokens.
3. Templates: distinct skeletons.  Slots are positions holding open classes.
4. Relations: templates with the same multiset of slot classes are merged
   when their slot fillers overlap and the union stays functional (conflicts
   from multi-sentence documents are ignored: documents are context).
5. Structure: a functional, acyclic relation between two slots of one class
   is a hierarchy (tree with 2-adic codes).  A functional relation from a
   hierarchy class to another hierarchy class is a link.  A functional
   relation from a hierarchy class to a non-hierarchy class is an attribute.
6. Explanations (minimum description length): each attribute is either a
   family program on its hierarchy (constrained parsimony, most specific
   family wins) or a lookup-table rule on an attribute of a linked hierarchy
   (this includes plain compositions, which is how the habitat-hop questions
   are explained without an operation library).
7. Answering: the next slot of a prefix is read from the relation of the
   matching template (or, for an unseen template, the relation chosen by
   anchor-token specificity and argument classes).  Multi-sentence documents
   write their facts into a context overlay (recency wins).

Supplied (declared): tokenization into whitespace tokens; documents
(multi-sentence units) write only to their own context; the rule hypothesis
class is lookup tables (value tokens are opaque, so no algebra on values can
be identified; the affine class needs a supplied value order, see ablations).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .fieldlm_world import compress_constrained
from .knowledge import UNKNOWN, Program, Tree, execute_by_characters

HOLE = "\x00HOLE"


# ---------------------------------------------------------------------------
# 1-2. Terminator and token classes
# ---------------------------------------------------------------------------


def induce_terminator(units: list[list[str]]) -> str:
    last = Counter(u[-1] for u in units if u)
    tok, n = last.most_common(1)[0]
    if n < 0.99 * len(units):
        raise ValueError("no token ends almost every unit")
    return tok


def split_by(unit: list[str], term: str) -> list[list[str]]:
    out, cur = [], []
    for w in unit:
        cur.append(w)
        if w == term:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if ra > rb:
            ra, rb = rb, ra
        self.parent[rb] = ra
        return True


def induce_classes(sentences: list[list[str]], max_rounds: int = 10) -> dict[str, str]:
    """Token -> class representative, by frame substitution and skeleton minimal pairs."""

    uf = UnionFind()
    for s in sentences:
        for w in s:
            uf.find(w)
    # (a) frame substitution on raw sentences
    frames: dict[tuple, list[str]] = defaultdict(list)
    for s in sorted(set(tuple(x) for x in sentences)):
        for i, w in enumerate(s):
            frames[s[:i] + (HOLE,) + s[i + 1:]].append(w)
    for fillers in frames.values():
        for w in fillers[1:]:
            uf.union(fillers[0], w)
    # (b) skeleton minimal pairs, iterated with (a) on abstracted sentences
    for _ in range(max_rounds):
        members: dict[str, int] = Counter(uf.find(w) for w in list(uf.parent))
        def sym(w: str) -> str:
            r = uf.find(w)
            return "<" + r + ">" if members[r] > 1 else w
        skeletons = {tuple(sym(w) for w in s) for s in sentences}
        changed = False
        buckets: dict[tuple, set[str]] = defaultdict(set)
        for sk in sorted(skeletons):
            for i, x in enumerate(sk):
                buckets[(len(sk), i) + sk[:i] + sk[i + 1:]].add(x)
        for _bucket, symbols in sorted(buckets.items()):
            if len(symbols) < 2:
                continue
            toks = [x[1:-1] if x.startswith("<") and x.endswith(">") else x for x in sorted(symbols)]
            for t in toks[1:]:
                changed |= uf.union(toks[0], t)
        if not changed:
            break
    return {w: uf.find(w) for w in sorted(uf.parent)}


# ---------------------------------------------------------------------------
# 3-4. Templates and relations
# ---------------------------------------------------------------------------


@dataclass
class Template:
    tid: int
    skeleton: tuple[str, ...]  # function tokens and "<cls>" slot symbols
    slots: list[int]  # positions of slots
    slot_classes: list[str]
    tuples_global: list[tuple[str, ...]] = field(default_factory=list)
    tuples_doc: list[tuple[str, ...]] = field(default_factory=list)
    count: int = 0
    example: tuple[str, ...] = ()

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(sorted(self.slot_classes))


@dataclass
class Relation:
    rid: int
    roles: list[str]  # class of each role
    members: dict[int, list[int]]  # template id -> role index of each of its slots
    tuples: set[tuple[str, ...]] = field(default_factory=set)  # global statements, in role order
    doc_tuples: set[tuple[str, ...]] = field(default_factory=set)  # statements inside documents
    kind: str = "other"  # hierarchy | link | attribute | other
    key: int = 0  # functional key role (binary relations)

    def fillers(self) -> list[set[str]]:
        out = [set() for _ in self.roles]
        for t in self.tuples | self.doc_tuples:
            for k, w in enumerate(t):
                out[k].add(w)
        return out


def _functional(tuples: set[tuple[str, ...]], key: int) -> bool:
    seen: dict[str, str] = {}
    for t in tuples:
        k, v = t[key], t[1 - key]
        if seen.setdefault(k, v) != v:
            return False
    return True


def relation_key(tuples: set[tuple[str, ...]]) -> int | None:
    """Functional role with more distinct fillers (the many-to-one direction)."""

    if not tuples or len(next(iter(tuples))) != 2:
        return None
    f = [_functional(tuples, k) for k in (0, 1)]
    n = [len({t[k] for t in tuples}) for k in (0, 1)]
    if f[0] and f[1]:
        return None if n[0] == n[1] else (0 if n[0] > n[1] else 1)
    return 0 if f[0] else (1 if f[1] else None)


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


class Parser:
    """Induced token classes, templates and relations."""

    def __init__(self, units: list[list[str]], docs_are_context: bool = True,
                 role_class_closure: bool = False) -> None:
        self.term = induce_terminator(units)
        self.units = units
        all_sents = [s for u in units for s in split_by(u, self.term)]
        self.cls = induce_classes(all_sents)
        self.role_class_merges: list[list[str]] = []
        if role_class_closure:
            # A lexical subclass (e.g. a parity partition of referents) need
            # not be a distinct argument type. Classes occupying exactly the
            # same abstract grammatical positions share a role-level parent.
            # This optional coarsening is inferred from text, without labels.
            counts = Counter(self.cls.values())
            signatures: dict[str, set[tuple]] = defaultdict(set)
            for sentence in all_sents:
                skeleton = tuple("<>" if counts[self.cls[word]] > 1 else word for word in sentence)
                for position, word in enumerate(sentence):
                    cls = self.cls[word]
                    if counts[cls] > 1:
                        signatures[cls].add((skeleton, position))
            families: dict[tuple, list[str]] = defaultdict(list)
            for cls, signature in signatures.items():
                families[tuple(sorted(signature))].append(cls)
            mapping = {}
            for family in families.values():
                if len(family) > 1:
                    self.role_class_merges.append(sorted(family))
                mapping.update((cls, min(family)) for cls in family)
            self.cls = {word: mapping.get(cls, cls) for word, cls in self.cls.items()}
        size = Counter(self.cls.values())
        self.open = {c for c, n in size.items() if n > 1}
        self.members: dict[str, list[str]] = defaultdict(list)
        for w, c in self.cls.items():
            if c in self.open:
                self.members[c].append(w)
        self.templates: dict[tuple[str, ...], Template] = {}
        for u in units:
            sents = split_by(u, self.term)
            doc = docs_are_context and len(sents) > 1
            for s in sents:
                sk, slots, fillers, classes = self.abstract(s)
                t = self.templates.get(sk)
                if t is None:
                    t = Template(len(self.templates), sk, slots, classes, example=tuple(s))
                    self.templates[sk] = t
                t.count += 1
                (t.tuples_doc if doc else t.tuples_global).append(tuple(fillers))
        self.by_id = {t.tid: t for t in self.templates.values()}
        self._cluster_relations()

    def slot_class(self, w: str) -> str | None:
        c = self.cls.get(w)
        return c if c in self.open else None

    def abstract(self, s: list[str]) -> tuple[tuple[str, ...], list[int], list[str], list[str]]:
        sk, slots, fillers, classes = [], [], [], []
        for i, w in enumerate(s):
            c = self.slot_class(w)
            if c is None:
                sk.append(w)
            else:
                sk.append("<" + c + ">")
                slots.append(i)
                fillers.append(w)
                classes.append(c)
        return tuple(sk), slots, fillers, classes

    def _cluster_relations(self) -> None:
        """Merge templates into relations, agglomeratively, to a fixed point.

        Two clusters with the same slot classes are merged under the slot
        alignment with the most evidence, where evidence is a shared stated
        tuple (one fact in both phrasings) or range agreement (filler overlap
        coefficient >= 0.5 in some role).  A merge must preserve functionality:
        the union stays functional in the key role of each side that has one
        (key = the functional role with more distinct fillers, i.e. the
        many-to-one direction).  Statements inside documents count for filler
        overlap but not for functionality (documents are context).
        """

        rels: list[Relation] = []
        self.merge_log: list[dict[str, Any]] = []
        for t in sorted(self.templates.values(), key=lambda t: (-t.count, t.tid)):
            if t.slots:
                r = Relation(len(rels), list(t.slot_classes), {t.tid: list(range(len(t.slots)))})
                r.tuples, r.doc_tuples = set(t.tuples_global), set(t.tuples_doc)
                rels.append(r)
        while True:
            best = None
            for i, a in enumerate(rels):
                ka = relation_key(a.tuples)
                fa = a.fillers()
                for b in rels[i + 1:]:
                    if sorted(a.roles) != sorted(b.roles):
                        continue
                    kb = relation_key(b.tuples)
                    fb = b.fillers()
                    for perm in self._alignments_roles(b.roles, a.roles):
                        mapped = {tuple(tp[perm.index(k)] for k in range(len(perm))) for tp in b.tuples}
                        kb_m = None if kb is None else perm[kb]
                        if ka is not None and kb_m is not None and ka != kb_m:
                            continue
                        key = ka if ka is not None else kb_m
                        if key is not None and not _functional(a.tuples | mapped, key):
                            continue
                        # Bijective sides have no preferred key, but still have
                        # functional directions. Shared entities are not enough
                        # to merge two different functions (e.g. two road maps).
                        if key is None and len(a.roles) == 2 and a.tuples and mapped:
                            common_keys = [k for k in (0, 1)
                                           if _functional(a.tuples, k) and _functional(mapped, k)]
                            if common_keys and not any(_functional(a.tuples | mapped, k) for k in common_keys):
                                continue
                        shared = len(a.tuples & mapped)
                        over = max(_overlap(fa[k], fb[perm.index(k)]) for k in range(len(perm)))
                        if shared == 0 and over < 0.5:
                            continue
                        score = (shared, over)
                        if best is None or score > best[0]:
                            best = (score, a, b, perm)
            if best is None:
                break
            (shared, over), a, b, perm = best
            self.merge_log.append({"into": sorted(a.members), "merged": sorted(b.members), "shared_tuples": shared,
                                   "range_overlap": over})
            for tid, align in b.members.items():
                a.members[tid] = [perm[k] for k in align]
            a.tuples |= {tuple(tp[perm.index(k)] for k in range(len(perm))) for tp in b.tuples}
            a.doc_tuples |= {tuple(tp[perm.index(k)] for k in range(len(perm))) for tp in b.doc_tuples}
            rels.remove(b)
        for i, r in enumerate(rels):
            r.rid = i
            if len(r.roles) == 2:
                k = relation_key(r.tuples or r.doc_tuples)
                r.key = 0 if k is None else k
        self.relations = rels
        self.rel_of_template = {tid: r for r in rels for tid in r.members}

    @staticmethod
    def _alignments_roles(src: list[str], dst: list[str]) -> list[list[int]]:
        """perm[k] = role in dst of src role k, respecting classes."""

        import itertools

        # Enumerating all roles first costs N! even when every role has its own
        # class. Generate only class-compatible assignments. The remaining
        # ambiguity is real (e.g. two person roles), rather than manufactured
        # by unrelated axes. Preserve the old lexicographic tie order.
        if len(src) != len(dst) or Counter(src) != Counter(dst):
            return []
        groups = []
        for cls in sorted(set(src)):
            left = [i for i, value in enumerate(src) if value == cls]
            right = [i for i, value in enumerate(dst) if value == cls]
            groups.append((left, list(itertools.permutations(right))))
        result = []
        for choices in itertools.product(*(perms for _, perms in groups)):
            assignment = [0] * len(src)
            for (left, _), right in zip(groups, choices, strict=True):
                for i, j in zip(left, right, strict=True):
                    assignment[i] = j
            result.append(assignment)
        return sorted(result)

    # -- unseen tokens ------------------------------------------------------------
    def extend_classes(self, s: list[str]) -> list[str]:
        """Assign unseen tokens to the slot class they fill in a known template.

        A token never seen in training is placed in class c when the sentence
        matches a known template everywhere else and that template has a slot
        of class c at the token's position (frame substitution at test time).
        Returns the tokens that were added.
        """

        unknown = [i for i, w in enumerate(s) if w not in self.cls]
        if not unknown:
            return []
        added = []
        for t in self.templates.values():
            if len(t.skeleton) != len(s):
                continue
            ok = True
            for i, x in enumerate(t.skeleton):
                if i in unknown:
                    ok = x.startswith("<")
                elif x.startswith("<"):
                    ok = self.slot_class(s[i]) == x[1:-1]
                else:
                    ok = x == s[i]
                if not ok:
                    break
            if ok:
                for i in unknown:
                    c = t.skeleton[i][1:-1]
                    self.cls[s[i]] = c
                    self.members[c].append(s[i])
                    added.append(s[i])
                return added
        return added

    # -- matching ---------------------------------------------------------------
    def match_prefix(self, prefix: list[str]) -> list[tuple[Template, list[str]]]:
        """Templates whose skeleton extends the prefix; returns (template, prefix fillers)."""

        out = []
        n = len(prefix)
        for t in self.templates.values():
            if len(t.skeleton) <= n:
                continue
            fillers = []
            ok = True
            for i in range(n):
                x = t.skeleton[i]
                if x.startswith("<") and x.endswith(">"):
                    if self.slot_class(prefix[i]) != x[1:-1]:
                        ok = False
                        break
                    fillers.append(prefix[i])
                elif x != prefix[i]:
                    ok = False
                    break
            if ok:
                out.append((t, fillers))
        return out


# ---------------------------------------------------------------------------
# 5-7. Structure, explanations and the knowledge store
# ---------------------------------------------------------------------------


def tree_from_parent_map(parent: dict[str, str], first_seen: dict[str, int]) -> tuple[Tree, dict[str, int]]:
    """Tree with 2-adic codes; children ordered by first appearance (no string structure)."""

    children: dict[str, list[str]] = defaultdict(list)
    nodes = set(parent) | set(parent.values())
    for c, p in parent.items():
        children[p].append(c)
    roots = [n for n in nodes if n not in parent]
    if len(roots) != 1:
        raise ValueError(f"hierarchy has {len(roots)} roots")
    order = [roots[0]]
    for n in order:
        order.extend(sorted(children.get(n, []), key=lambda t: (first_seen.get(t, 0), t)))
    index = {t: i for i, t in enumerate(order)}
    par = np.full(len(order), -1)
    kids: list[list[int]] = [[] for _ in order]
    depth = np.zeros(len(order), dtype=int)
    for t in order[1:]:
        par[index[t]] = index[parent[t]]
        kids[index[parent[t]]].append(index[t])
        depth[index[t]] = depth[index[parent[t]]] + 1
    leaves = [index[t] for t in order if not children.get(t)]
    max_kids = max(len(k) for k in kids)
    bits = max(1, math.ceil(math.log2(max(2, max_kids))))
    code = np.zeros(len(order), dtype=np.int64)
    for t in order[1:]:
        i, p = index[t], index[parent[t]]
        code[i] = code[p] + kids[p].index(i) * (1 << (bits * int(depth[p])))
    return Tree(par, kids, depth, np.array(leaves), code, bits), index


@dataclass
class Hierarchy:
    rel: Relation
    cls: str
    tree: Tree
    index: dict[str, int]
    leaf_tokens: list[str]


class FieldLM2:
    """Field LM whose parsing (classes, templates, relations, hierarchies) is induced."""

    def __init__(self, units: list[list[str]], smoothing: float = 0.03, value_order=None,
                 docs_are_context: bool = True, push_compositions: bool = False) -> None:
        self.eps = smoothing
        self.docs_are_context = docs_are_context
        # ablation A2: callable(class members) -> full ordered value list, enables affine rules
        self.value_order = value_order
        # post hoc F1: statements of a copy composition are observations of its source at the link target.
        # True: push when the first-pass explanation is copy.  "joint" (F1b): push when copy with the
        # pushed observations has a shorter description than the first-pass explanation.
        self.push_compositions = push_compositions
        self.pushed: dict[int, dict[str, str]] = {}
        self.p = Parser(units, docs_are_context)
        self.first_seen: dict[str, int] = {}
        for u in units:
            for w in u:
                self.first_seen.setdefault(w, len(self.first_seen))
        self.overlay: dict[int, dict[str, str]] = {}
        self.abstained = 0
        self._build()
        self._class_counts: dict[str, Counter] = defaultdict(Counter)
        for t in self.p.templates.values():
            for tp in t.tuples_global + t.tuples_doc:
                for w, c in zip(tp, t.slot_classes):
                    self._class_counts[c][w] += 1

    # -- structure ---------------------------------------------------------------
    def _build(self) -> None:
        p = self.p
        rels = p.relations
        self.hier: dict[str, Hierarchy] = {}
        for r in rels:
            if len(r.roles) == 2 and r.roles[0] == r.roles[1] and r.tuples:
                for key in (0, 1):
                    if not _functional(r.tuples, key):
                        continue
                    parent = {t[key]: t[1 - key] for t in sorted(r.tuples)}
                    try:
                        tree, index = tree_from_parent_map(parent, self.first_seen)
                    except ValueError:
                        continue
                    if r.roles[0] in self.hier and len(self.hier[r.roles[0]].index) >= len(index):
                        continue
                    r.kind, r.key = "hierarchy", key
                    leaves = [t for t, i in sorted(index.items(), key=lambda kv: kv[1]) if i in set(tree.leaves.tolist())]
                    leaf_ids = np.array([index[t] for t in leaves])
                    tree.leaves = leaf_ids
                    self.hier[r.roles[0]] = Hierarchy(r, r.roles[0], tree, index, leaves)
                    break
        self.attrs: list[Relation] = []
        self.links: list[Relation] = []
        for r in rels:
            if r.kind == "hierarchy" or len(r.roles) != 2:
                continue
            for key in (0, 1):
                kc, vc = r.roles[key], r.roles[1 - key]
                if kc not in self.hier:
                    continue
                tuples = r.tuples or r.doc_tuples
                if not tuples or not _functional(tuples, key):
                    continue
                r.key = key
                r.kind = "link" if vc in self.hier else "attribute"
                break
            if r.kind == "link":
                self.links.append(r)
            elif r.kind == "attribute":
                self.attrs.append(r)
        # value ids: order of first appearance (value tokens are opaque)
        self.value_ids: dict[str, list[str]] = {}
        for r in self.attrs:
            vc = r.roles[1 - r.key]
            if vc not in self.value_ids:
                order = self.value_order(self.p.members[vc]) if self.value_order else None
                self.value_ids[vc] = order or sorted(self.p.members[vc], key=lambda t: self.first_seen.get(t, 0))
        self._build_programs()

    def vid(self, vc: str, tok: str) -> int:
        return self.value_ids[vc].index(tok)

    def facts(self, r: Relation) -> dict[str, str]:
        groups: dict[str, set[str]] = defaultdict(set)
        for t in sorted(r.tuples):
            groups[t[r.key]].add(t[1 - r.key])
        # Conflicting global facts are ambiguous, not arbitrary hash-order
        # overwrites. Documents retain their explicitly ordered context path.
        return {key: next(iter(values)) for key, values in groups.items() if len(values) == 1}

    def _build_programs(self) -> None:
        self._fit_programs()
        if not self.push_compositions:
            return
        pushed: dict[int, dict[str, str]] = defaultdict(dict)
        for r in self.attrs:
            e = self.explain[r.rid]
            if self.push_compositions == "joint":
                cand = self._copy_push_candidate(r)
                if cand is None or cand[0] >= e["bits"]:
                    continue
                _bits, src, obs = cand
            else:
                if e["type"] != "rule" or e.get("form") != "copy":
                    continue
                src = self._rel(e["source"])
                direct = self.facts(src)
                obs = {}
                for ent, val in self.facts(r).items():
                    target = self.link_facts[e["link"]].get(ent)
                    if target is not None and target not in direct:
                        obs.setdefault(target, val)
            for target, val in obs.items():
                pushed[src.rid].setdefault(target, val)
        self.pushed = dict(pushed)
        if self.pushed:
            self._fit_programs()

    def _program_bits(self, r: Relation, prog: Program) -> float:
        h = self.hier[r.roles[r.key]]
        return len(prog.overrides[0]) * (math.log2(h.tree.size) + math.log2(len(self.value_ids[r.roles[1 - r.key]])))

    def _copy_push_candidate(self, r: Relation):
        """Description length of r as a copy of a linked attribute whose program absorbs r's statements.

        Returns (bits, source relation, pushed observations) for the best source, or None.
        bits = increase of the source program + copy flag + exceptions (statements that
        conflict with a direct statement of the source or with each other).
        """

        h = self.hier[r.roles[r.key]]
        vc = r.roles[1 - r.key]
        nval = len(self.value_ids[vc])
        exc_bits = math.log2(max(2, len(h.leaf_tokens))) + math.log2(nval)
        best = None
        for l in self.links:
            if l.roles[l.key] != h.cls or l.roles[1 - l.key] not in self.hier:
                continue
            for src in self.attrs:
                if src.rid == r.rid or src.roles[src.key] != l.roles[1 - l.key] or src.roles[1 - src.key] != vc:
                    continue
                direct = self.facts(src)
                obs: dict[str, str] = {}
                errors = 0
                for ent, val in self.facts(r).items():
                    target = self.link_facts[l.rid].get(ent)
                    if target is None:
                        continue
                    if target in direct:
                        errors += direct[target] != val
                    elif obs.setdefault(target, val) != val:
                        errors += 1
                if not obs:
                    continue
                saved = self.pushed.get(src.rid)
                self.pushed[src.rid] = {**(saved or {}), **obs}
                refit = self._tree_program(src)
                if saved is None:
                    del self.pushed[src.rid]
                else:
                    self.pushed[src.rid] = saved
                bits = self._program_bits(src, refit) - self._program_bits(src, self.prog[src.rid]) + 2 + errors * exc_bits
                if best is None or bits < best[0]:
                    best = (bits, src, obs)
        return best

    def observed(self, r: Relation) -> dict[str, str]:
        """Stated facts of an attribute, plus observations pushed through compositions (direct wins)."""

        return {**self.pushed.get(r.rid, {}), **self.facts(r)}

    def _fit_programs(self) -> None:
        self.prog: dict[int, Program] = {}
        self.explain: dict[int, dict[str, Any]] = {}
        # attributes: tree programs first (needed as rule sources)
        for r in self.attrs:
            self.prog[r.rid] = self._tree_program(r)
        link_facts = {l.rid: self.facts(l) for l in self.links}
        self.link_facts = link_facts
        for r in self.attrs:
            h = self.hier[r.roles[r.key]]
            vc = r.roles[1 - r.key]
            nval = len(self.value_ids[vc])
            leaf_obs = self._leaf_obs(r)
            tree_bits = len(self.prog[r.rid].overrides[0]) * (math.log2(h.tree.size) + math.log2(nval))
            best: dict[str, Any] = {"type": "tree", "bits": tree_bits}
            exc_bits = math.log2(max(2, len(h.leaf_tokens))) + math.log2(nval)
            for l in self.links:
                if l.roles[l.key] != h.cls:
                    continue
                target = self.hier.get(l.roles[1 - l.key])
                if target is None:
                    continue
                for src in self.attrs:
                    if src.rid == r.rid or src.roles[src.key] != target.cls:
                        continue
                    svc = src.roles[1 - src.key]
                    snval = len(self.value_ids[svc])
                    pairs = []
                    for i, tok in enumerate(h.leaf_tokens):
                        if leaf_obs[i] != UNKNOWN and tok in link_facts[l.rid]:
                            pairs.append((self.attr_value(src, link_facts[l.rid][tok]), leaf_obs[i]))
                    if len(pairs) < 2:
                        continue
                    g = np.zeros(snval, dtype=int)
                    support = np.zeros(snval, dtype=int)
                    for v in range(snval):
                        hits = [y for x, y in pairs if x == v]
                        support[v] = len(hits)
                        g[v] = np.bincount(hits, minlength=nval).argmax() if hits else 0
                    errors = sum(g[x] != y for x, y in pairs)
                    fallback_obs = np.array([leaf_obs[i] if h.leaf_tokens[i] not in link_facts[l.rid] else UNKNOWN
                                             for i in range(len(h.leaf_tokens))])
                    fb = compress_constrained(h.tree, fallback_obs, {}, nval)
                    fb_bits = len(fb.overrides[0]) * (math.log2(h.tree.size) + math.log2(nval))
                    bits = snval * math.log2(nval) + 1 + errors * exc_bits + fb_bits
                    if bits < best["bits"]:
                        best = {"type": "rule", "link": l.rid, "source": src.rid, "g": g, "support": support,
                                "bits": bits, "form": "table"}
                    if svc == vc:
                        # copy: the attribute is the source attribute read through the link
                        err = sum(x != y for x, y in pairs)
                        cbits = 1 + 1 + err * exc_bits + fb_bits
                        if cbits < best["bits"]:
                            best = {"type": "rule", "link": l.rid, "source": src.rid, "g": np.arange(snval),
                                    "support": np.full(snval, 1), "bits": cbits, "form": "copy"}
                    if self.value_order and snval == nval:
                        for mult in (1, 3, 5, 7):
                            for shift in range(nval):
                                ga = np.array([(mult * x + shift) % nval for x in range(snval)])
                                err = sum(ga[x] != y for x, y in pairs)
                                abits = 5 + 1 + err * exc_bits + fb_bits
                                if abits < best["bits"]:
                                    best = {"type": "rule", "link": l.rid, "source": src.rid, "g": ga,
                                            "support": np.full(snval, 1), "bits": abits, "form": f"affine:{mult}x+{shift}"}
            self.explain[r.rid] = best
        # exceptions: stated leaf facts the rule does not reproduce
        self.exceptions: dict[tuple[int, str], int] = {}
        for r in self.attrs:
            e = self.explain[r.rid]
            if e["type"] != "rule":
                continue
            h = self.hier[r.roles[r.key]]
            vc = r.roles[1 - r.key]
            leaves = set(h.leaf_tokens)
            src = self._rel(e["source"])
            for ent, val in self.facts(r).items():
                if ent not in leaves:
                    continue
                hab = self.link_facts[e["link"]].get(ent)
                if hab is None or int(e["g"][self.attr_value(src, hab)]) != self.vid(vc, val):
                    self.exceptions[(r.rid, ent)] = self.vid(vc, val)
        self._abduce()

    def _rel(self, rid: int) -> Relation:
        return self.p.relations[rid]

    def _leaf_obs(self, r: Relation) -> np.ndarray:
        h = self.hier[r.roles[r.key]]
        vc = r.roles[1 - r.key]
        f = self.observed(r)
        return np.array([self.vid(vc, f[t]) if t in f else UNKNOWN for t in h.leaf_tokens])

    def _tree_program(self, r: Relation) -> Program:
        h = self.hier[r.roles[r.key]]
        vc = r.roles[1 - r.key]
        leaves = set(h.leaf_tokens)
        cons = {h.index[t]: self.vid(vc, v) for t, v in self.facts(r).items() if t not in leaves and t in h.index}
        return compress_constrained(h.tree, self._leaf_obs(r), cons, len(self.value_ids[vc]))

    def _abduce(self) -> None:
        """Hidden link targets chosen to explain the stated rule facts (effects -> cause)."""

        self.abduced: dict[int, dict[str, str]] = {}
        for l in self.links:
            h = self.hier[l.roles[l.key]]
            target = self.hier[l.roles[1 - l.key]]
            rules = [(r, self.explain[r.rid]) for r in self.attrs
                     if self.explain[r.rid]["type"] == "rule" and self.explain[r.rid]["link"] == l.rid]
            out: dict[str, str] = {}
            for ent in h.leaf_tokens:
                if ent in self.link_facts[l.rid]:
                    continue
                scores = np.zeros(len(target.leaf_tokens), dtype=int)
                for r, e in rules:
                    f = self.facts(r)
                    if ent not in f:
                        continue
                    vc = r.roles[1 - r.key]
                    y = self.vid(vc, f[ent])
                    src = self._rel(e["source"])
                    pred = np.array([e["g"][self.attr_value(src, t)] for t in target.leaf_tokens])
                    scores += pred == y
                out[ent] = target.leaf_tokens[int(scores.argmax())] if scores.size and scores.max() > 0 else None
            self.abduced[l.rid] = out

    # -- lookups -----------------------------------------------------------------
    def attr_value(self, r: Relation, ent: str) -> int:
        """Value id of attribute ``r`` at entity ``ent`` (family program; no rules)."""

        h = self.hier[r.roles[r.key]]
        node = h.index[ent]
        return int(self.prog[r.rid].labels[0, node])

    def link_target(self, l: Relation, ent: str) -> str | None:
        ov = self.overlay.get(l.rid, {})
        if ent in ov:
            return ov[ent]
        return self.link_facts[l.rid].get(ent) or self.abduced.get(l.rid, {}).get(ent)

    def lookup(self, r: Relation, ent: str) -> str | None:
        """Answer token for relation ``r`` at key ``ent``."""

        if r.kind == "hierarchy":
            h = self.hier[r.roles[0]]
            if ent not in h.index:
                return None
            node = h.index[ent]
            up = int(h.tree.parent[node])
            if up < 0:
                return None
            inv = getattr(h, "_inv", None) or {i: t for t, i in h.index.items()}
            h._inv = inv
            return inv[up]
        if r.kind == "link":
            return self.link_target(r, ent)
        if r.kind != "attribute":
            f = self.facts(r)
            return f.get(ent)
        vc = r.roles[1 - r.key]
        h = self.hier[r.roles[r.key]]
        if ent not in h.index:
            return None
        if (r.rid, ent) in self.exceptions:
            return self.value_ids[vc][self.exceptions[(r.rid, ent)]]
        e = self.explain[r.rid]
        if e["type"] == "rule" and ent in set(h.leaf_tokens):
            target = self.link_target(self._rel(e["link"]), ent)
            if target is not None:
                cause = self.attr_value(self._rel(e["source"]), target)
                if e.get("form") == "copy":
                    return self.value_ids[vc][cause]
                if cause >= len(e["support"]) or e["support"][cause] == 0:
                    self.abstained += 1
                    return None  # the consequence of an unseen cause is not identified
                return self.value_ids[vc][int(e["g"][cause])]
        node = h.index[ent]
        prog = self.prog[r.rid]
        value = execute_by_characters(h.tree, prog, 0, np.array([h.tree.code[node]]))[0]
        if value == UNKNOWN or h.tree.depth[node] < h.tree.depth.max():
            value = int(prog.labels[0, node])
        return self.value_ids[vc][int(value)]

    # -- reading context and answering -------------------------------------------
    def read_context(self, units: list[list[str]]) -> None:
        self.overlay = {}
        for unit in units:
            for s in split_by(unit, self.p.term):
                sk, _slots, fillers, _cls = self.p.abstract(s)
                t = self.p.templates.get(sk)
                if t is None:
                    r, align = self.identify_full(s)
                else:
                    r = self.p.rel_of_template.get(t.tid)
                    align = r.members[t.tid] if r else None
                if r is None or r.kind != "link" or align is None:
                    continue
                roles = [None] * len(r.roles)
                for i, k in enumerate(align):
                    roles[k] = fillers[i]
                self.overlay.setdefault(r.rid, {})[roles[r.key]] = roles[1 - r.key]

    def identify_full(self, s: list[str]) -> tuple[Relation | None, list[int] | None]:
        """Relation of an unseen complete sentence by its argument classes (unique signature)."""

        _sk, _slots, fillers, classes = self.p.abstract(s)
        cands = [r for r in self.p.relations if sorted(r.roles) == sorted(classes) and r.kind in ("link", "attribute", "hierarchy")]
        if len(cands) != 1:
            return None, None
        r = cands[0]
        align = []
        used = set()
        for c in classes:
            k = next(i for i, rc in enumerate(r.roles) if rc == c and i not in used)
            used.add(k)
            align.append(k)
        return r, align

    def complete(self, prefix: list[str], context: list[str] | None = None) -> str | None:
        """Predict the next slot token after ``prefix`` (the relation's answer role)."""

        self.read_context([context] if context else [])
        try:
            sent_start = max((i + 1 for i, w in enumerate(prefix[:-1]) if w == self.p.term), default=0)
            q = prefix[sent_start:]
            for t, fillers in sorted(self.p.match_prefix(q), key=lambda tf: -tf[0].count):
                n = len(q)
                if not (t.skeleton[n].startswith("<")) or n != t.slots[-1]:
                    continue
                r = self.p.rel_of_template.get(t.tid)
                if r is None or len(r.roles) != 2:
                    continue
                align = r.members[t.tid]
                ans_role = align[-1]
                if ans_role == r.key:
                    continue
                key_filler = fillers[-1] if fillers else None
                if key_filler is None:
                    continue
                out = self.lookup(r, key_filler)
                if out is not None:
                    return out
            return self.complete_unseen(q)
        finally:
            self.overlay = {}

    def complete_unseen(self, q: list[str]) -> str | None:
        """Unseen template: pick the relation by anchor-token specificity and argument classes."""

        slot_toks = [w for w in q if self.p.slot_class(w)]
        if len(slot_toks) != 1:
            return None
        key_cls = self.p.slot_class(slot_toks[0])
        anchors = [w for w in q if not self.p.slot_class(w) and w in self.p.cls]
        rel_tokens: dict[int, Counter] = defaultdict(Counter)
        for t in self.p.templates.values():
            r = self.p.rel_of_template.get(t.tid)
            if r is None:
                continue
            for x in t.skeleton:
                if not x.startswith("<"):
                    rel_tokens[r.rid][x] += 1
        df = Counter()
        for toks in rel_tokens.values():
            for x in toks:
                df[x] += 1
        nrel = max(1, len(rel_tokens))
        best = None
        for r in self.p.relations:
            if len(r.roles) != 2 or r.kind not in ("attribute", "link", "hierarchy") or r.roles[r.key] != key_cls:
                continue
            toks = rel_tokens.get(r.rid, Counter())
            score = sum(math.log(nrel / df[a]) for a in anchors if a in toks and df[a])
            # tokens every template of this relation carries but the prefix lacks
            templates = [self.p.by_id[tid] for tid in r.members]
            required = set.intersection(*[{x for x in t.skeleton if not x.startswith("<")} for t in templates]) if templates else set()
            score -= sum(math.log(nrel / df[x]) for x in sorted(required) if x not in q and df[x])
            if best is None or score > best[0]:
                best = (score, r)
        if best is None or best[0] <= 0:
            return None
        return self.lookup(best[1], slot_toks[0])

    def answer(self, prompt: list[str], context: list[str] | None = None) -> str | None:
        return self.complete(prompt, context)

    # -- next-token probabilities ---------------------------------------------------
    def sentence_logprobs(self, sent: list[str]) -> list[float]:
        """log2 p(token_i | prefix): mixture over induced templates consistent with the prefix."""

        total = sum(t.count for t in self.p.templates.values())
        weights = {t.tid: math.log(t.count / total) for t in self.p.templates.values()}
        out = []
        for i, tok in enumerate(sent):
            dist: dict[str, float] = {}
            norm = 0.0
            mx = max(weights.values()) if weights else 0.0
            for tid, lw in weights.items():
                t = self.p.by_id[tid]
                if i >= len(t.skeleton):
                    continue
                w = math.exp(lw - mx)
                norm += w
                for x, pr in self._slot_dist(t, i, sent[:i]).items():
                    dist[x] = dist.get(x, 0.0) + w * pr
            p_tok = dist.get(tok, 0.0) / norm if norm else 0.0
            out.append(math.log2(max(p_tok, 1e-6)))
            new = {}
            for tid, lw in weights.items():
                t = self.p.by_id[tid]
                if i >= len(t.skeleton):
                    continue
                pr = self._slot_dist(t, i, sent[:i]).get(tok, 0.0)
                if pr > 0:
                    new[tid] = lw + math.log(pr)
            weights = new or weights
        return out

    def _slot_dist(self, t: Template, i: int, prefix: list[str]) -> dict[str, float]:
        x = t.skeleton[i]
        if not x.startswith("<"):
            return {x: 1.0}
        c = x[1:-1]
        cands = self.p.members[c]
        if i == t.slots[-1]:
            r = self.p.rel_of_template.get(t.tid)
            if r is not None and len(r.roles) == 2:
                align = r.members[t.tid]
                if align[-1] != r.key:
                    key_pos = t.slots[align.index(r.key)]
                    key_tok = prefix[key_pos] if key_pos < len(prefix) else None
                    pred = self.lookup(r, key_tok) if key_tok is not None else None
                    if pred is not None and pred in cands:
                        others = [w for w in cands if w != pred]
                        d = {pred: 1.0 - self.eps}
                        for w in others:
                            d[w] = self.eps / max(1, len(others))
                        return d
        counts = self._class_counts.get(c, Counter())
        tot = sum(counts.get(w, 0) + 1 for w in cands)
        return {w: (counts.get(w, 0) + 1) / tot for w in cands}

    # -- editing -----------------------------------------------------------------
    def attribute_for(self, entity: str, value_tok: str) -> Relation | None:
        vc = self.p.slot_class(value_tok)
        ec = self.p.slot_class(entity)
        for r in self.attrs:
            if r.roles[1 - r.key] == vc and r.roles[r.key] == ec:
                return r
        return None

    def edit_text(self, sent: list[str]) -> bool:
        """Apply an edit stated as one sentence: the family program of its relation changes."""

        for w in self.p.extend_classes(sent):
            c = self.p.cls[w]
            if c in self.value_ids and w not in self.value_ids[c]:
                self.value_ids[c].append(w)
        sk, _slots, fillers, _cls = self.p.abstract(sent)
        t = self.p.templates.get(sk)
        if t is None:
            r, align = self.identify_full(sent)
        else:
            r = self.p.rel_of_template.get(t.tid)
            align = r.members[t.tid] if r else None
        if r is None or r.kind != "attribute" or align is None:
            return False
        roles = [None] * len(r.roles)
        for i, k in enumerate(align):
            roles[k] = fillers[i]
        self._edit_program(r, roles[r.key], roles[1 - r.key])
        return True

    def edit_family(self, entity: str, _attr_token: str, value_tok: str) -> None:
        r = self.attribute_for(entity, value_tok)
        if r is not None:
            self._edit_program(r, entity, value_tok)

    def _edit_program(self, r: Relation, entity: str, value_tok: str) -> None:
        h = self.hier[r.roles[r.key]]
        vc = r.roles[1 - r.key]
        prog = self.prog[r.rid]
        node = h.index[entity]
        prog.overrides[0][node] = self.vid(vc, value_tok)
        labels = prog.labels[0]
        for n in h.tree.order():
            labels[n] = prog.overrides[0][n] if n in prog.overrides[0] else labels[h.tree.parent[n]]

    # -- size ------------------------------------------------------------------------
    def bits(self) -> dict[str, float]:
        tax = sum((h.tree.size - 1) * math.log2(h.tree.size) for h in self.hier.values())
        programs = 0.0
        for r in self.attrs:
            h = self.hier[r.roles[r.key]]
            nval = len(self.value_ids[r.roles[1 - r.key]])
            e = self.explain[r.rid]
            programs += (len(self.prog[r.rid].overrides[0]) * (math.log2(h.tree.size) + math.log2(nval))
                         if e["type"] == "tree" else e["bits"])
        links = sum(len(self.link_facts[l.rid]) * math.log2(max(2, len(self.hier[l.roles[1 - l.key]].leaf_tokens)))
                    for l in self.links)
        lexicon = len(self.p.cls) * math.log2(max(2, len(set(self.p.cls.values()))))
        templates = sum(len(t.skeleton) for t in self.p.templates.values()) * math.log2(max(2, len(self.p.cls)))
        return {"taxonomy": tax, "programs": programs, "links": links, "lexicon": lexicon, "templates": templates,
                "total": tax + programs + links + lexicon + templates}


def field2_edit_copy(lm: FieldLM2) -> FieldLM2:
    import copy

    return copy.deepcopy(lm)
