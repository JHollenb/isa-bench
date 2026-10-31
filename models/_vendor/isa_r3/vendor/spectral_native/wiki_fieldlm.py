"""Native Wikipedia views feeding the consolidated learned binding fabric."""

from __future__ import annotations

from .fabric_fieldlm import FabricFieldLM


class WikiFieldLM(FabricFieldLM):
    def wiki_component(self, capability="wiki"):
        component = self.component(capability)
        if component.kind != "wiki":
            raise ValueError("a captured Wikipedia field capability is required")
        return component

    def read_wiki_pair(self, pair, *, capability="wiki"):
        """Read a label-free supplied mention pair through source-induced axes."""
        return self.invoke(capability, pair)

    def observe_wiki_pairs(self, pairs, *, namespace, scope, view="wiki-facts", capability="wiki"):
        """Retain inputs and write only reader-resolved inferred fact records."""
        return self.invoke(capability, view=view, pairs=pairs, namespace=namespace, scope=scope, operation="populate")

    def reason_wiki(self, start, relations, *, namespace, scope, budget=2, capability="wiki"):
        """Run a captured two-read program; relation path is supplied by caller."""
        if len(relations) != 2:
            raise ValueError("this captured task requires exactly two relation IDs")
        return self.invoke(capability, "two-hop", {"start": start, "r0": relations[0], "r1": relations[1]},
                           namespace=namespace, scope=scope, budget=budget, operation="solve")

    def observe_wiki_document(self, document, *, namespace, scope, capability="wiki"):
        """Enumerate supplied mentions without using alignment labels/triples.

        Entity recognition/linking and sentence segmentation remain caller-supplied.
        This deliberately shares the pre-label candidate builder with the pilot.
        """
        from scripts.extract_wiki_axis_corpus import article_pairs

        clean = {key: value for key, value in document.items() if key != "triples"}
        rows = article_pairs(clean, str(clean.get("docid") or clean.get("uri") or scope))
        for row in rows:
            row.pop("labels", None)
        return self.observe_wiki_pairs(rows, namespace=namespace, scope=scope, capability=capability)
