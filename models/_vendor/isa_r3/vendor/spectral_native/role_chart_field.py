"""Open-arity language charts compiled to projected address joins.

Uses FieldLM2's substitution classes and relation/role induction unchanged.
Unlike its hierarchy-specific knowledge store, this consumer preserves every
induced relation and every role. No gradient optimizer or semantic inventory.
Hash joins are the exact compiled address semantics, not a new SPF primitive.
Scopes, namespaces, source identity and the completion/iteration operations
are supplied. Surface grammar and role alignments are induced from raw units.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .fieldlm_world2 import Parser


@dataclass(frozen=True)
class Record:
    namespace: str
    scope: str
    relation: int
    roles: tuple[str, ...]
    source: str
    position: int


class RoleChartField:
    def __init__(self, units: list[list[str]], *, role_class_closure: bool = True) -> None:
        parser = Parser(units, docs_are_context=False, role_class_closure=role_class_closure)
        self.charts = [
            {"skeleton": list(template.skeleton), "slots": list(template.slots),
             "relation": relation.rid, "alignment": list(relation.members[template.tid]),
             "classes": list(template.slot_classes)}
            for template in parser.templates.values()
            if (relation := parser.rel_of_template.get(template.tid)) is not None
        ]
        self.classes = dict(parser.cls)
        self.merge_log = parser.merge_log
        self.role_class_merges = parser.role_class_merges
        self.records: list[Record] = []
        self.active_axes: set[int] | None = None
        self._indices: dict[tuple, dict[tuple, list[Record]]] = {}

    @property
    def grammar_sha256(self) -> str:
        return hashlib.sha256(json.dumps(self.charts, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_state(cls, state: dict) -> RoleChartField:
        obj = cls.__new__(cls)
        obj.charts = state["charts"]
        obj.classes = state["classes"]
        obj.merge_log = state["merge_log"]
        obj.role_class_merges = state["role_class_merges"]
        obj.records = [Record(**{**row, "roles": tuple(row["roles"])}) for row in state["records"]]
        obj.active_axes = None if state["active_axes"] is None else set(state["active_axes"])
        obj._indices = {}
        return obj

    def state(self) -> dict:
        return {"charts": self.charts, "classes": self.classes, "merge_log": self.merge_log,
                "role_class_merges": self.role_class_merges,
                "records": [record.__dict__ for record in self.records],
                "active_axes": None if self.active_axes is None else sorted(self.active_axes)}

    def rechart(self, units: list[list[str]]) -> RoleChartField:
        """Grow the source grammar while transporting existing addressed facts.

        Shared surface charts establish the relation/role coordinate change.
        Refuse an ambiguous migration before mutating either live memory.
        Source/namespace/scope stay attached to each transported observation.
        """
        grown = RoleChartField(units)
        def surface(chart):
            slots = set(chart["slots"])
            return tuple("<>" if i in slots else word for i, word in enumerate(chart["skeleton"]))
        targets: dict[tuple, list[dict]] = {}
        for chart in grown.charts:
            targets.setdefault(surface(chart), []).append(chart)
        mappings: dict[int, set[tuple]] = {}
        for old in self.charts:
            for new in targets.get(surface(old), []):
                if len(old["slots"]) != len(new["slots"]):
                    continue
                permutation = [0] * len(old["slots"])
                for old_axis, new_axis in zip(old["alignment"], new["alignment"], strict=True):
                    permutation[old_axis] = new_axis
                mappings.setdefault(old["relation"], set()).add((new["relation"], tuple(permutation)))
        for record in self.records:
            choices = mappings.get(record.relation, set())
            if len(choices) != 1:
                raise ValueError("existing record has no unique chart migration")
            relation, permutation = next(iter(choices))
            roles = [""] * len(record.roles)
            for old_axis, new_axis in enumerate(permutation):
                roles[new_axis] = record.roles[old_axis]
            grown.records.append(Record(record.namespace, record.scope, relation, tuple(roles),
                                        record.source, record.position))
        # Experimental gate overrides are not a learned grammar and may use
        # old role IDs. Recharting restores normal projected-query semantics.
        return grown

    def matches(self, tokens: list[str], *, holes: bool = False) -> list[tuple[dict, tuple[str, ...]]]:
        matches = []
        for chart in self.charts:
            if len(tokens) != len(chart["skeleton"]):
                continue
            slots = set(chart["slots"])
            if any(word != expected for i, (word, expected) in enumerate(
                    zip(tokens, chart["skeleton"], strict=True)) if i not in slots):
                continue
            fillers = [tokens[i] for i in chart["slots"]]
            if not holes and "?" in fillers:
                continue
            if any(word != "?" and word in self.classes and self.classes[word] != cls
                   for word, cls in zip(fillers, chart["classes"], strict=True)):
                continue
            roles = [""] * len(fillers)
            for word, role in zip(fillers, chart["alignment"], strict=True):
                roles[role] = word
            matches.append((chart, tuple(roles)))
        return matches

    def ingest(self, sentences: list[list[str]], *, namespace: str, scope: str) -> dict:
        recognized = ambiguous = 0
        for position, sentence in enumerate(sentences):
            interpretations = {(chart["relation"], roles) for chart, roles in self.matches(sentence)}
            if len(interpretations) == 1:
                relation, roles = next(iter(interpretations))
                source = hashlib.sha256(" ".join(sentence).encode()).hexdigest()
                self.records.append(Record(namespace, scope, relation, roles, source, position))
                recognized += 1
            elif interpretations:
                ambiguous += 1
        self._indices.clear()
        return {"sentences": len(sentences), "recognized": recognized, "ambiguous": ambiguous}

    def lookup(self, relation: int, bindings: dict[int, str], *, namespace: str, scope: str) -> list[Record]:
        # Roles not in the query are wildcards; source and recency are evidence
        # axes, never accidental hard keys. Project before constructing a hash.
        axes = tuple(sorted(bindings))
        if self.active_axes is not None:
            axes = tuple(axis for axis in axes if axis in self.active_axes)
        index_key = (relation, axes)
        if index_key not in self._indices:
            index: dict[tuple, list[Record]] = {}
            for record in self.records:
                # Open-arity views may share a relation coordinate. A tuple
                # lacking a requested role cannot satisfy that projection.
                if record.relation == relation and all(axis < len(record.roles) for axis in axes):
                    key = (record.namespace, record.scope, *(record.roles[axis] for axis in axes))
                    index.setdefault(key, []).append(record)
            self._indices[index_key] = index
        key = (namespace, scope, *(bindings[axis] for axis in axes))
        return self._indices[index_key].get(key, [])

    def complete(self, tokens: list[str], *, namespace: str, scope: str) -> dict:
        values, sources = set(), set()
        for chart, roles in self.matches(tokens, holes=True):
            missing = [i for i, value in enumerate(roles) if value == "?"]
            if len(missing) != 1:
                continue
            bindings = {i: value for i, value in enumerate(roles) if value != "?"}
            for record in self.lookup(chart["relation"], bindings, namespace=namespace, scope=scope):
                values.add(record.roles[missing[0]])
                sources.add(record.source)
        status = "resolved" if len(values) == 1 else "ambiguous" if values else "missing"
        return {"status": status, "value": next(iter(values)) if len(values) == 1 else None,
                "candidates": sorted(values), "sources": sorted(sources)}

    def chain(self, start: str, query_templates: list[list[str]], *, namespace: str, scope: str) -> dict:
        state, trace = start, []
        for query in query_templates:
            tokens = [state if word == "$state" else word for word in query]
            result = self.complete(tokens, namespace=namespace, scope=scope)
            trace.append({"query": tokens, "read": result})
            if result["status"] != "resolved":
                return {"status": result["status"], "value": None, "trace": trace}
            state = result["value"]
        return {"status": "resolved", "value": state, "trace": trace}
