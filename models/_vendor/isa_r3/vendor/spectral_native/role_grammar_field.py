"""Source-induced coordination and argument sharing over FieldLM role charts.

The supplied hypothesis class is concatenation of clauses, optionally eliding
one leading argument. Connector strings and the donor binding are learned from
source text and repeated source facts. Mention spans/identities are supplied;
there is no entity detector, neural reader, or new SPF arithmetic here.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict

from .role_chart_field import Record, RoleChartField


def token_kind(word: str) -> str:
    """A supplied lexical view; admitted kinds are inferred from source slots."""
    return "lexical" if any(char.isalnum() for char in word) else "punctuation"


def collapse_mentions(tokens: list[str], mentions: list[dict]) -> list[str]:
    """Replace supplied disjoint [start,end) mention spans by opaque identities."""
    output, cursor = [], 0
    for mention in sorted(mentions, key=lambda item: item["start"]):
        start, end = mention["start"], mention["end"]
        if not cursor <= start < end <= len(tokens):
            raise ValueError("overlapping or invalid mention span")
        output.extend(tokens[cursor:start])
        output.append(mention["identity"])
        cursor = end
    return output + tokens[cursor:]


class RoleGrammarField:
    def __init__(self, base_units: list[list[str]], compounds: list[list[str]]) -> None:
        self.field = RoleChartField(base_units)
        self.term = base_units[0][-1]
        self.connectors: list[list[str]] = []
        self.elisions: list[dict] = []
        self.evidence: dict = {}
        self.kind_guard = True
        self.elision_enabled = True
        self.closed_tokens = sorted({word for chart in self.field.charts
                                     for position, word in enumerate(chart["skeleton"])
                                     if position not in chart["slots"]})
        self.kinds = sorted({token_kind(word) for chart in self.field.charts
                             for word, cls in self.field.classes.items() if cls in chart["classes"]})
        self._induce(base_units, compounds)

    def state(self) -> dict:
        return {"field": self.field.state(), "term": self.term, "connectors": self.connectors,
                "elisions": self.elisions, "evidence": self.evidence, "kinds": self.kinds,
                "kind_guard": self.kind_guard, "elision_enabled": self.elision_enabled,
                "closed_tokens": self.closed_tokens}

    @classmethod
    def from_state(cls, state: dict) -> RoleGrammarField:
        obj = cls.__new__(cls)
        obj.field = RoleChartField.from_state(state["field"])
        for name in ("term", "connectors", "elisions", "evidence", "kinds", "kind_guard", "elision_enabled"):
            setattr(obj, name, state[name])
        # Older artifacts did not distinguish connector words from arguments.
        # Keep their measured behavior rather than silently repairing replay.
        obj.closed_tokens = state.get("closed_tokens", [])
        return obj

    @property
    def grammar_sha256(self) -> str:
        state = self.state()
        state["field"] = {key: value for key, value in state["field"].items()
                          if key not in ("records", "active_axes")}
        return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()

    def _matches(self, body: list[str], *, holes=False):
        return [(chart, roles) for chart, roles in self.field.matches([*body, self.term], holes=holes)
                if not self.kind_guard or all(word == "?" or (word not in self.closed_tokens and
                                               token_kind(word) in self.kinds) for word in roles)]

    @staticmethod
    def _signature(chart: dict) -> list[str]:
        slots = set(chart["slots"])
        return ["<>" if position in slots else word for position, word in enumerate(chart["skeleton"])]

    def _induce(self, base_units, compounds):
        facts = {(chart["relation"], roles) for unit in base_units for chart, roles in self._matches(unit[:-1])}
        connectors = Counter()
        widths = sorted({len(chart["skeleton"]) - 1 for chart in self.field.charts})
        # Exact known clauses establish where an intervening connector lives.
        for unit in compounds:
            body = unit[:-1]
            for left in widths:
                if not self._matches(body[:left]):
                    continue
                for right in widths:
                    if left + right < len(body) and self._matches(body[-right:]):
                        connectors[tuple(body[left:-right])] += 1
        self.connectors = [list(value) for value in sorted(connectors)]
        self.closed_tokens = sorted(set(self.closed_tokens) | {word for tokens in self.connectors for word in tokens})
        candidates: dict[tuple, Counter] = defaultdict(Counter)
        exposures = Counter()
        for unit in compounds:
            body = unit[:-1]
            for left in widths:
                for donor, donor_roles in self._matches(body[:left]):
                    for connector in self.connectors:
                        if body[left:left + len(connector)] != connector:
                            continue
                        tail = body[left + len(connector):]
                        for target in self.field.charts:
                            # The bounded first hypothesis is a missing leading slot.
                            if not target["slots"] or target["slots"][0] != 0:
                                continue
                            if len(tail) != len(target["skeleton"]) - 2:
                                continue
                            signature = tuple(self._signature(target))
                            if any(word != expected for position, (word, expected) in enumerate(
                                    zip(tail, target["skeleton"][1:-1], strict=True), start=1)
                                   if position not in target["slots"]):
                                continue
                            exposures[signature] += 1
                            for ordinal, donor_axis in enumerate(donor["alignment"]):
                                reconstructed = [donor_roles[donor_axis], *tail]
                                for chart, roles in self._matches(reconstructed):
                                    if (self._signature(chart) == list(signature) and
                                            (chart["relation"], roles) in facts):
                                        candidates[signature][ordinal] += 1
        # Continuation choice uses source witnesses only; retain every rival's
        # signed support. Ties remain competing grammar hypotheses at runtime.
        for signature, counts in sorted(candidates.items()):
            best = max(counts.values())
            self.elisions.extend({"signature": list(signature), "donor_slot": ordinal,
                                  "support": count, "exposures": exposures[signature]}
                                 for ordinal, count in sorted(counts.items()) if count == best)
        self.evidence = {"connector_support": [{"tokens": list(key), "support": count}
                                               for key, count in sorted(connectors.items())],
                         "elision_candidates": [{"signature": list(signature),
                                                  "exposures": exposures[signature],
                                                  "support_by_donor_slot": dict(counts)}
                                                 for signature, counts in sorted(candidates.items())],
                         "source_facts": len(facts), "source_compounds": len(compounds),
                         "selection": "maximum repeated-source-fact support; preserve ties and rival counts"}

    def parse(self, tokens: list[str]) -> dict:
        if not tokens or tokens[-1] != self.term:
            return {"status": "missing", "interpretations": [], "trace": []}
        body, complete = tokens[:-1], {}
        charts = self.field.charts

        def visit(position, previous, records, trace):
            if position == len(body):
                complete.setdefault(tuple(records), trace)
                return
            for chart in charts:
                width = len(chart["skeleton"]) - 1
                choices = []
                for matched, roles in self._matches(body[position:position + width]):
                    if matched is chart:
                        choices.append((width, roles, None))
                if previous and self.elision_enabled and chart["slots"][0] == 0:
                    donor, donor_roles = previous
                    for rule in self.elisions:
                        if rule["signature"] != self._signature(chart) or rule["donor_slot"] >= len(donor["alignment"]):
                            continue
                        donor_axis = donor["alignment"][rule["donor_slot"]]
                        reconstructed = [donor_roles[donor_axis], *body[position:position + width - 1]]
                        for matched, roles in self._matches(reconstructed):
                            if matched is chart:
                                choices.append((width - 1, roles, {"donor_axis": donor_axis,
                                                                  "target_axis": chart["alignment"][0],
                                                                  "value": donor_roles[donor_axis]}))
                for consumed, roles, inherited in choices:
                    end = position + consumed
                    read = (chart["relation"], roles)
                    next_trace = [*trace, {"span": [position, end], "relation": chart["relation"],
                                          "roles": list(roles), "inherited_binding": inherited}]
                    if end == len(body):
                        visit(end, (chart, roles), [*records, read], next_trace)
                    else:
                        for connector in self.connectors:
                            if body[end:end + len(connector)] == connector:
                                visit(end + len(connector), (chart, roles), [*records, read], next_trace)

        visit(0, None, [], [])
        status = "resolved" if len(complete) == 1 else "ambiguous" if complete else "missing"
        return {"status": status,
                "interpretations": [[{"relation": relation, "roles": list(roles)} for relation, roles in key]
                                    for key in complete],
                "trace": next(iter(complete.values())) if len(complete) == 1 else [],
                "alternative_traces": list(complete.values()) if len(complete) > 1 else []}

    def ingest(self, sentences: list[list[str]], *, namespace: str, scope: str) -> dict:
        rows, written = [], 0
        for position, sentence in enumerate(sentences):
            parsed = self.parse(sentence)
            rows.append(parsed)
            if parsed["status"] == "resolved":
                source = hashlib.sha256(json.dumps(sentence).encode()).hexdigest()
                for clause, row in enumerate(parsed["interpretations"][0]):
                    self.field.records.append(Record(namespace, scope, row["relation"], tuple(row["roles"]),
                                                     source, position * 1000000 + clause))
                    written += 1
        self.field._indices.clear()
        return {"sentences": len(sentences), "resolved": sum(row["status"] == "resolved" for row in rows),
                "records_written": written, "rows": rows}

    def complete(self, tokens, *, namespace, scope):
        return self.field.complete(tokens, namespace=namespace, scope=scope)

    def chain(self, start, query_templates, *, namespace, scope):
        return self.field.chain(start, query_templates, namespace=namespace, scope=scope)
