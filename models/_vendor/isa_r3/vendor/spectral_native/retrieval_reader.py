"""Experimental lexical/sparse-semantic example reader for FieldLM.

Ordinary postings and calibrated votes, not SPF arithmetic. Semantic coordinates
propose relation/direction labels; they never become entity or role coordinates.
Only source examples carry labels. Serving rows must be label-free. The caller
owns encoder identity, split quarantine, and validation of downstream behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def reader_text(row):
    """Mention identities are excluded, while relative mention positions remain."""
    return " ".join({"<L>": "Alice", "<R>": "Bob", "<e>": "someone"}.get(w, w) for w in row["marked"])


def lexical_terms(text):
    return re.findall(r"\w+", text.lower())


def polarity(row):
    # Supplied diagnostic heuristic, not learned negation/attachment semantics.
    return int(any(w.lower() in {"not", "never", "no", "n't"} for w in row["marked"]))


@dataclass(frozen=True)
class ReaderPolicy:
    alpha: float = 1.0
    neighbors: int = 5
    confidence: float = 0.0
    margin: float = 0.0
    compatibility: bool = False
    semantic_only: bool = False

    def __post_init__(self):
        if not math.isfinite(self.alpha) or self.alpha < 0 or self.neighbors < 1:
            raise ValueError("invalid retrieval policy")
        if not 0 <= self.confidence <= 1 or not 0 <= self.margin <= 1:
            raise ValueError("confidence/margin must be in [0,1]")


class SparseExampleIndex:
    """BM25 and sparse dot products in disjoint posting namespaces.

    Scores retain separate lexical and semantic evidence. Query-local RMS
    balances the two score channels; no Evoke fitted constants are imported.
    Every source example stays attached to its labels and source identity.
    """

    def __init__(self, source: Sequence[Mapping], vectors: Mapping[str, Mapping[int, float]]):
        self.documents = []
        grouped = {}
        for row in source:
            if not row.get("labels"):
                continue
            key = (reader_text(row), tuple(row["kinds"]), polarity(row))
            grouped.setdefault(key, {"labels": Counter(), "sources": []})
            grouped[key]["labels"].update(row["labels"])
            grouped[key]["sources"].append(row["pair_id"])
        self.lexical = defaultdict(list)
        self.semantic = defaultdict(list)
        self.lengths = []
        for (text, kinds, negated), evidence in sorted(grouped.items()):
            ordinal = len(self.documents)
            self.documents.append(
                {
                    "text": text,
                    "kinds": list(kinds),
                    "polarity": negated,
                    "labels": dict(sorted(evidence["labels"].items())),
                    "sources": sorted(evidence["sources"]),
                }
            )
            counts = Counter(lexical_terms(text))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.lexical[term].append((ordinal, count))
            for term, impact in vectors.get(text, {}).items():
                if not math.isfinite(impact) or impact < 0:
                    raise ValueError("semantic impacts must be finite/nonnegative")
                if impact:
                    self.semantic[int(term)].append((ordinal, float(impact)))
        self.average_length = sum(self.lengths) / max(1, len(self.lengths))

    def state(self):
        return {
            "schema": "fieldlm-sparse-example-index-v1",
            "documents": self.documents,
            "lexical": {k: v for k, v in sorted(self.lexical.items())},
            "semantic": {str(k): v for k, v in sorted(self.semantic.items())},
            "lengths": self.lengths,
            "average_length": self.average_length,
        }

    @classmethod
    def from_state(cls, state):
        if state["schema"] != "fieldlm-sparse-example-index-v1":
            raise ValueError("unsupported sparse example index")
        obj = cls.__new__(cls)
        obj.documents = state["documents"]
        obj.lexical = {k: [tuple(v) for v in rows] for k, rows in state["lexical"].items()}
        obj.semantic = {int(k): [tuple(v) for v in rows] for k, rows in state["semantic"].items()}
        obj.lengths = state["lengths"]
        obj.average_length = state["average_length"]
        return obj

    def score(self, row, vector, policy):
        if row.get("labels"):
            raise ValueError("serving labels are forbidden")
        count = len(self.documents)
        lexical, semantic = [0.0] * count, [0.0] * count
        touched = {"lexical": 0, "semantic": 0}
        for term, qtf in Counter(lexical_terms(reader_text(row))).items():
            postings = self.lexical.get(term, ())
            idf = math.log1p((count - len(postings) + 0.5) / (len(postings) + 0.5))
            for document, tf in postings:
                length = self.lengths[document] / max(self.average_length, 1.0)
                lexical[document] += qtf * idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * length))
                touched["lexical"] += 1
        for term, impact in vector.items():
            if not math.isfinite(impact) or impact < 0:
                raise ValueError("invalid query impact")
            for document, weight in self.semantic.get(int(term), ()):
                semantic[document] += impact * weight
                touched["semantic"] += 1
        lexical_rms = math.sqrt(sum(x * x for x in lexical) / max(count, 1))
        semantic_rms = math.sqrt(sum(x * x for x in semantic) / max(count, 1))
        scale = (lexical_rms or 1.0) / (semantic_rms or 1.0)
        ranked = []
        for i, document in enumerate(self.documents):
            if policy.compatibility and (
                list(row["kinds"]) != document["kinds"] or polarity(row) != document["polarity"]
            ):
                continue
            score = (0.0 if policy.semantic_only else lexical[i]) + policy.alpha * scale * semantic[i]
            if score > 0:
                ranked.append(
                    {
                        "document": i,
                        "score": score,
                        "lexical": lexical[i],
                        "semantic": semantic[i],
                        "semantic_scale": scale,
                    }
                )
        ranked.sort(key=lambda item: (-item["score"], item["document"]))
        return ranked, touched

    def predict(self, row, vector, policy=None):
        policy = ReaderPolicy() if policy is None else policy
        ranked, touched = self.score(row, vector, policy)
        neighbors = ranked[: policy.neighbors]
        votes = defaultdict(float)
        for hit in neighbors:
            evidence = self.documents[hit["document"]]["labels"]
            total = sum(evidence.values())
            for label, support in evidence.items():
                votes[label] += hit["score"] * support / total
        total = sum(votes.values())
        candidates = [
            {"label": label, "score": value, "share": value / total}
            for label, value in sorted(votes.items(), key=lambda item: (-item[1], item[0]))
        ]
        best = candidates[0]["share"] if candidates else 0.0
        margin = best - (candidates[1]["share"] if len(candidates) > 1 else 0.0)
        selected = (
            candidates[0]["label"] if candidates and best >= policy.confidence and margin >= policy.margin else None
        )
        return {
            "selected": selected,
            "candidates": candidates,
            "confidence": best,
            "margin": margin,
            "neighbors": neighbors,
            "postings_touched": touched,
            "policy": asdict(policy),
            "status": "resolved" if selected else "abstained",
        }
