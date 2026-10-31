"""Factor semantic relation proposals from source-learned FieldLM direction."""

from collections import defaultdict

from .retrieval_reader import ReaderPolicy, SparseExampleIndex
from .wiki_axis_field import WikiAxisConfig, WikiAxisField, WikiPair


class FactorizedExampleReader:
    def __init__(self, index, direction_reader):
        self.index = index
        self.direction_reader = direction_reader

    @classmethod
    def fit(cls, source, vectors):
        direction_rows = [
            {**row, "labels": sorted({"P0:" + label.rsplit(":", 1)[1] for label in row["labels"]})} for row in source
        ]
        reader = WikiAxisField.fit(
            [WikiPair.from_dict(row) for row in direction_rows],
            config=WikiAxisConfig(
                views=("located",),
                active_axes={"located": ("word_sequence", "side", "distance", "left_kind", "right_kind")},
                min_support=2,
                min_article_support=1,
                ambiguity_policy="unique-best",
                max_candidates=128,
            ),
        )
        return cls(SparseExampleIndex(source, vectors), reader)

    def predict(self, row, vector, *, confidence=0.0, compatibility=True, direction_mode="selected"):
        if direction_mode not in {"selected", "first"}:
            raise ValueError("unknown direction policy")
        semantic = self.index.predict(row, vector, ReaderPolicy(semantic_only=True, compatibility=compatibility))
        relation_votes = defaultdict(float)
        for candidate in semantic["candidates"]:
            relation_votes[candidate["label"].rsplit(":", 1)[0]] += candidate["score"]
        total = sum(relation_votes.values())
        relations = [
            {"relation": relation, "share": vote / total}
            for relation, vote in sorted(relation_votes.items(), key=lambda item: (-item[1], item[0]))
        ]
        direction = self.direction_reader.predict(WikiPair.from_dict(row), detail=False)
        picked = direction.get("trace", {}).get("decision", {}).get("selected")
        if picked is None and direction_mode == "first" and direction["candidates"]:
            picked = direction["candidates"][0]
        chosen_direction = picked["direction"] if picked else None
        selected = (
            f"{relations[0]['relation']}:{chosen_direction}"
            if relations and relations[0]["share"] >= confidence and chosen_direction
            else None
        )
        directions = sorted({c["direction"] for c in direction["candidates"]})
        candidates = [
            {"label": f"{relation['relation']}:{d}", "share": relation["share"]}
            for relation in relations
            for d in directions
        ]
        return {
            "selected": selected,
            "candidates": candidates,
            "status": "resolved" if selected else "abstained",
            "relation_votes": relations,
            "direction": {"selected": chosen_direction, "native": direction},
            "retrieval": semantic,
            "policy": {
                "confidence": confidence,
                "compatibility": compatibility,
                "direction_mode": direction_mode,
            },
        }

    def state(self):
        return {
            "schema": "fieldlm-factorized-example-reader-v1",
            "index": self.index.state(),
            "direction": self.direction_reader.state(),
        }

    @classmethod
    def from_state(cls, state):
        if state["schema"] != "fieldlm-factorized-example-reader-v1":
            raise ValueError("unsupported factorized reader")
        return cls(SparseExampleIndex.from_state(state["index"]), WikiAxisField.from_state(state["direction"]))
