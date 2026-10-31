"""Small supervised lexical-rule organism; ordinary counts, no SPF arithmetic."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import torch


def pair_partition(left, right):
    """Unordered-pair split; mention order cannot leak the reverse pair."""
    a = np.minimum(left, right).astype(np.uint64)
    b = np.maximum(left, right).astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (a + np.uint64(1)) * np.uint64(0x9E3779B185EBCA87)
        x ^= (b + np.uint64(1)) * np.uint64(0xC2B2AE3D27D4EB4F)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xFF51AFD7ED558CCD)
        x ^= x >> np.uint64(33)
    bucket = x % np.uint64(100)
    return np.where(bucket < 60, 0, np.where(bucket < 80, 1, 2)).astype(np.int8)


def wilson(correct, total):
    if not total:
        return [0.0, 1.0]
    p = correct / total
    z = 1.96
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [max(0.0, center - half), min(1.0, center + half)]


def features(pattern, kind, kinds=("entity", "entity")):
    if kind == "typed":
        return {("types", *kinds, *key) for key in features(pattern, "ordered")}
    words = pattern.split()
    if kind == "exact":
        return {("full", *words)}
    seq = ["^", *words, "$"] if kind == "ordered" else words
    widths = range(1, 5) if kind == "ordered" else (1,)
    return {
        ("gram", *seq[i : i + width])
        for width in widths
        for i in range(len(seq) - width + 1)
        if any(token not in ("^", "$", "<e>") for token in seq[i : i + width])
    }


@dataclass(frozen=True)
class Row:
    index: int
    pattern: str
    left: int
    right: int
    labels: tuple[int, ...]
    kinds: tuple[str, str] = ("entity", "entity")

    @property
    def pair(self):
        return tuple(sorted((self.left, self.right)))


class RuleBank(torch.nn.Module):
    """Keys fixed at induction; typed predictions and threshold are restorable."""

    def __init__(self, kind, rules):
        super().__init__()
        self.kind = kind
        self.keys = tuple(tuple(r["key"]) for r in rules)
        self.metadata = rules
        self.lookup = {key: i for i, key in enumerate(self.keys)}
        self.register_buffer("labels", torch.tensor([r["label"] for r in rules], dtype=torch.long))
        self.register_buffer("confidence", torch.tensor([r["lower"] for r in rules], dtype=torch.float64))
        # A structural control parameter, never updated by gradients in this experiment.
        self.threshold = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64), requires_grad=False)

    def predict(self, pattern, kinds=("entity", "entity")):
        return self.predict_features(features(pattern, self.kind, kinds))

    def predict_features(self, keys):
        ids = [self.lookup[key] for key in keys if key in self.lookup]
        ids = [i for i in ids if float(self.confidence[i]) >= float(self.threshold)]
        if not ids:
            return -1, -1
        chosen = min(ids, key=lambda i: (-float(self.confidence[i]), -self.metadata[i]["support"], self.keys[i]))
        return int(self.labels[chosen]), chosen


def induce(rows, kind, max_rules=4096):
    return induce_feature_rows(rows, lambda row: features(row.pattern, kind, row.kinds), kind, max_rules)


def induce_feature_rows(rows, feature_fn, kind, max_rules=4096):
    evidence = defaultdict(dict)
    for row in rows:
        for key in feature_fn(row):
            evidence[key].setdefault(row.pair, set()).update(row.labels)
    rules = []
    for key, pairs in evidence.items():
        support = len(pairs)
        if support < 10:
            continue
        counts = Counter(label for labels in pairs.values() for label in labels)
        label = min(counts, key=lambda x: (-counts[x], x))
        matches = counts[label]
        rules.append({"key": list(key), "label": label, "support": support,
                      "matches": matches, "precision": matches / support,
                      "lower": wilson(matches, support)[0],
                      "label_counts": {str(k): v for k, v in sorted(counts.items())}})
    rules.sort(key=lambda r: (-r["support"], -r["lower"], r["key"]))
    return RuleBank(kind, rules[:max_rules])


def summary(rows, predictions):
    covered = sum(pred >= 0 for pred in predictions)
    matches = sum(pred in row.labels for row, pred in zip(rows, predictions, strict=True))
    return {"rows": len(rows), "covered": covered, "correct": matches,
            "coverage": covered / max(1, len(rows)), "precision": matches / covered if covered else None,
            "precision_wilson95": wilson(matches, covered),
            "answer_accuracy": matches / max(1, len(rows))}


def choose_threshold(curve):
    active = [row for row in curve if row["covered"]]
    reliable = [row for row in active if row["precision"] >= 0.95]
    if reliable:
        return min(reliable, key=lambda row: (-row["coverage"], row["threshold"]))["threshold"]
    if active:
        return min(active, key=lambda row: (-row["precision"], -row["coverage"], row["threshold"]))["threshold"]
    return 0.0


def record(label, left, right, label_names):
    if label < 0:
        return None
    relation, direction = label_names[label].rsplit(":", 1)
    if direction not in ("f", "r"):
        raise ValueError(direction)
    subject, obj = (left, right) if direction == "f" else (right, left)
    return {"relation": relation, "subject": subject, "object": obj}


def consume(compiled, row, label_names):
    """Query only a compiled record; never repairs it from the aligned label."""
    for label in row.labels:
        query = record(label, row.left, row.right, label_names)
        if (compiled and compiled["relation"] == query["relation"]
                and compiled["subject"] == query["subject"] and compiled["object"] == query["object"]):
            return True
    return False
