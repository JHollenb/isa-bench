"""Frozen-reader fusion over canonical records, without score averaging."""
from __future__ import annotations

from spectral_native.ordinary_language_structure import record
from spectral_native.unified_fieldlm import Entity, Evidence, UnifiedFieldLM


def resolve_proposals(proposals):
    groups = {}
    for proposal in proposals:
        row = proposal["record"]
        if row is not None:
            key = (row["subject"], row["relation"], row["object"])
            groups.setdefault(key, []).append(proposal)
    status = "missing" if not groups else "ambiguous" if len(groups) > 1 else "resolved"
    resolved = next(iter(groups.values()))[0]["record"] if status == "resolved" else None
    return {"status": status, "record": resolved, "proposals": proposals}


class IntegratedFieldLM(UnifiedFieldLM):
    """The unified native runtime plus a deliberately supplied fusion policy.

    Accept one available interpretation or agreement. Preserve conflicting
    proposals and abstain. Agreement does not make correlated readers exact.
    Spans, tokenization, namespaces and input capability remain supplied.
    """

    def read_sentence(self, row, *, readers=("language", "attachment", "surface"), scope, namespace):
        proposals = []
        for name in readers:
            component = self.component(name)
            if component.kind == "language":
                result = self.invoke(name, row.pattern, row.left, row.right, kinds=row.kinds)
                pred, rule, compiled = result["label"], result["rule"], result["record"]
            elif component.kind == "attachment":
                pred, rule = self.invoke(name, row)
                compiled = record(pred, row.left, row.right, component.state["label_names"])
            else:
                raise ValueError("only directional record readers can vote")
            bank = component.state["model"]
            proposals.append({"reader": name, "record": compiled, "rule": rule,
                              "label": component.state["label_names"][pred] if pred >= 0 else None,
                              "source": self.manifest["components"][name]["sha256"],
                              "empirical_lower_bound": float(bank.confidence[rule]) if rule >= 0 else None})
        result = resolve_proposals(proposals)
        if result["record"]:
            value = result["record"]
            for proposal in proposals:
                if proposal["record"] == value:
                    self.memory.write(Evidence(Entity(namespace, value["subject"]), value["relation"],
                        Entity(namespace, value["object"]), scope, proposal["source"], "inferred"))
        return result
