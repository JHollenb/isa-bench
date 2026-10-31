"""Endpoint-local text records and count-based readers; no SPF arithmetic."""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from spectral_native.ordinary_language_structure import Row, features
from spectral_native.pattern_identity import Corpus, _qid, between_pattern, tokens
from spectral_native.pattern_identity_v2 import entity_roles, type_kmeans


@dataclass(frozen=True)
class SentenceRow(Row):
    sentence: str = ""
    marked: tuple[str, ...] = ()
    source: str = ""
    bounds: tuple[int, ...] = ()
    partition: int = 0
    aliases: tuple[int, int] = (0, 0)
    anchor_marked: tuple[str, ...] = ()


def argument(uri, surface):
    alias = _qid(uri)
    if re.fullmatch(r"Q[0-9]+", alias):
        return alias, "entity"
    kind = "date" if alias == "XMLSchema#dateTime" else "other-literal"
    return f"literal:{alias}:{' '.join(surface.lower().split())}", kind


def mark_sentence(text, lo, hi, left, right, mentions):
    spans = [(left[0], left[1], "<L>"), (right[0], right[1], "<R>")]
    for a, b in sorted(set(mentions)):
        if lo <= a < b <= hi and not any(a < y and b > x for x, y, _ in spans):
            spans.append((a, b, "<e>"))
    out, pos = [], lo
    for a, b, marker in sorted(spans):
        if not (pos <= a < b <= hi):
            raise ValueError("overlapping or outside argument spans")
        out.extend(tokens(text[pos:a]))
        out.append(marker)
        pos = b
    out.extend(tokens(text[pos:hi]))
    if out.count("<L>") != 1 or out.count("<R>") != 1:
        raise ValueError("argument marker identity")
    return tuple(out)


def doc_rows(doc, source, entity_ids, label_ids, diagnostics):
    text = doc["text"]
    sentences = doc.get("sentences_boundaries") or []
    mentions = sorted({tuple(e["boundaries"]) for e in doc.get("entities") or [] if e.get("boundaries")})
    grouped = {}
    for triple in doc.get("triples") or []:
        sb, ob = triple["subject"].get("boundaries"), triple["object"].get("boundaries")
        sid = triple.get("sentence_id")
        if sb is None or ob is None or sid is None or not 0 <= sid < len(sentences):
            continue
        lo, hi = sentences[sid]
        if not (lo <= sb[0] < sb[1] <= hi and lo <= ob[0] < ob[1] <= hi):
            continue
        forward = sb[0] < ob[0]
        ls, rs = (triple["subject"], triple["object"]) if forward else (triple["object"], triple["subject"])
        left, right = tuple(ls["boundaries"]), tuple(rs["boundaries"])
        if left[1] > right[0]:
            continue
        aliases = (_qid(ls["uri"]), _qid(rs["uri"]))
        if aliases[0] == aliases[1] or any(alias not in entity_ids for alias in aliases):
            diagnostics["unknown_or_equal_alias"] += 1
            continue
        pattern = between_pattern(text, left[1], right[0], mentions)
        if pattern is None:
            diagnostics["between_over_ten_tokens"] += 1
            continue
        label = f"{_qid(triple['predicate']['uri'])}:{'f' if forward else 'r'}"
        if label not in label_ids:
            diagnostics["unknown_label"] += 1
            continue
        lvalue, lkind = argument(ls["uri"], text[left[0]:left[1]])
        rvalue, rkind = argument(rs["uri"], text[right[0]:right[1]])
        key = (sid, left, right)
        value = (lvalue, rvalue, aliases)
        if key in grouped:
            if grouped[key][0] != value:
                diagnostics["conflicting_anchor_identity"] += 1
                grouped[key][2] = True
            grouped[key][1].add(label_ids[label])
            continue
        marked = mark_sentence(text, lo, hi, left, right, mentions)
        row = SentenceRow(index=0, pattern=pattern, left=lvalue, right=rvalue, labels=(), kinds=(lkind, rkind),
                          sentence=text[lo:hi], marked=marked, source=f"{source}:{sid}",
                          bounds=(lo, hi, *left, *right), aliases=tuple(entity_ids[x] for x in aliases),
                          anchor_marked=mark_sentence(text, lo, hi, left, right, []))
        grouped[key] = [value, {label_ids[label]}, False, row]
    return [item[3].__class__(**{**item[3].__dict__, "labels": tuple(sorted(item[1]))})
            for item in grouped.values() if not item[2]]


def learn_roles(rows, seed=101):
    """Reuse the existing sparse role-profile implementation on fit facts only."""
    names = sorted({value for row in rows for value in (row.left, row.right)})
    entities = {value: i for i, value in enumerate(names)}
    unique = {(row.left, row.right, row.labels) for row in rows}
    facts = sorted(unique)
    n_labels = max((label for _, _, labels in facts for label in labels), default=-1) + 1
    left = np.array([entities[l] for l, _, _ in facts], dtype=np.int32)
    right = np.array([entities[r] for _, r, _ in facts], dtype=np.int32)
    labels = np.full((len(facts), max((len(x[2]) for x in facts), default=1)), -1, dtype=np.int32)
    for i, (_, _, values) in enumerate(facts):
        labels[i, :len(values)] = values
    corpus = Corpus(["fit"], [str(i) for i in range(n_labels)], names,
                    np.zeros(len(facts), dtype=np.int32), left, right, labels)
    corpus.split = np.zeros(1, dtype=np.int8)
    matrix = entity_roles(corpus)
    clusters = type_kmeans(matrix, 8, seed=seed) if len(facts) else np.full(len(names), -1)
    mapping = {name: int(clusters[i]) for i, name in enumerate(names)}
    return mapping, {"fit_distinct_facts": len(facts), "fit_entities": len(names),
                     "clusters": len(set(mapping.values()) - {-1}), "profile_shape": list(matrix.shape)}


def local_keys(row, mode, roles):
    if mode == "between":
        return features(row.pattern, "typed", row.kinds)
    role_pair = tuple(str(roles.get(value, -1)) for value in (row.left, row.right))
    if mode == "roles_only":
        return {("role_pair", *row.kinds, *role_pair)}
    words = row.anchor_marked if mode == "anchor_only" else row.marked
    l, r = words.index("<L>"), words.index("<R>")
    regions = (("between", list(words[l + 1:r])),
               ("left_before", list(words[max(0, l - 4):l])),
               ("right_after", list(words[r + 1:r + 5])))
    keys = set()
    for region, seq in regions:
        for width in range(1, 5):
            for i in range(len(seq) - width + 1):
                gram = seq[i:i + width]
                if all(token in ("<e>", "<L>", "<R>") for token in gram):
                    continue
                gap = len(seq) - i - width if region in ("between", "left_before") else i
                bucket = str(min(gap, 4))
                key = ("located", *row.kinds, region, bucket, *gram)
                keys.add(key)
                if mode == "roles_context":
                    keys.add(("roles", *role_pair, *key))
    return keys


def key_distance(key):
    if key[0] == "roles":
        key = key[3:]
    return int(key[4]) if key and key[0] == "located" else 0


def choose_candidate(curve):
    active = [point for point in curve if point["covered"]]
    preferred = [point for point in active if point["precision"] >= 0.95]
    pool = preferred or active
    if not pool:
        return {"threshold": 0.0, "distance_weight": 0.0}
    if preferred:
        return min(pool, key=lambda p: (-p["coverage"], p["threshold"], p["distance_weight"]))
    return min(pool, key=lambda p: (-p["precision"], -p["coverage"], p["distance_weight"], p["threshold"]))
