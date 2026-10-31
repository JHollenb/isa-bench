"""Joint text axes and bounded interpretation futures in the shared fabric."""

from __future__ import annotations

from .wiki_fieldlm import WikiFieldLM


class WikiJointFieldLM(WikiFieldLM):
    def wiki_component(self, capability="wiki_joint"):
        component = self.component(capability)
        if component.kind not in {"wiki", "wiki_joint"}:
            raise ValueError("a captured Wikipedia field capability is required")
        return component

    def read_wiki_pair(self, pair, *, capability="wiki_joint"):
        return super().read_wiki_pair(pair, capability=capability)

    def observe_wiki_pairs(self, pairs, *, namespace, scope, view="wiki-facts", capability="wiki_joint"):
        return super().observe_wiki_pairs(pairs, namespace=namespace, scope=scope, view=view, capability=capability)

    def observe_wiki_document(self, document, *, namespace, scope, capability="wiki_joint"):
        return super().observe_wiki_document(document, namespace=namespace, scope=scope, capability=capability)

    def reason_wiki(self, start, relations, *, namespace, scope, budget=2, capability="wiki_joint", competing=False):
        """Consume bounded alternatives; both names share one frozen policy.

        The caller supplies the relation path. Agreement among surviving
        answer branches does not verify the text interpretations.
        """
        if len(relations) != 2:
            raise ValueError("this captured task requires exactly two relation IDs")
        return self.invoke(
            capability,
            "two-hop",
            {"start": start, "r0": relations[0], "r1": relations[1]},
            namespace=namespace,
            scope=scope,
            budget=budget,
            operation="solve_competing" if competing else "solve",
        )
