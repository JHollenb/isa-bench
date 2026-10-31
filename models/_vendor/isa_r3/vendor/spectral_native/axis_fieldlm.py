"""Structural FieldLM entry point using the native open-arity role engine.

No new phase or join implementation. Legacy consumers remain explicit. Relation
IDs belong to a grammar, so a numeric ID is never shared between grammars.
"""

from __future__ import annotations

from spectral_native.fieldlm_fusion import IntegratedFieldLM
from spectral_native.role_chart_field import Record


class AxisFieldLM(IntegratedFieldLM):
    @property
    def profiles(self):
        return self.manifest.get("profiles", {})

    def role_model(self, capability="roles"):
        component = self.component(capability)
        if component.kind == "grammar":
            return component.state["model"].field
        if component.kind != "roles":
            raise ValueError("an open-arity role capability is required")
        return component.state["model"]

    def parse_tokens(self, tokens, *, mentions=None, capability="grammar"):
        """Parse complete clauses; mention spans/identities remain supplied."""
        if mentions is not None:
            from spectral_native.role_grammar_field import collapse_mentions

            tokens = collapse_mentions(tokens, mentions)
        return self.invoke(capability, tokens, operation="parse")

    def observe_tokens(self, sentences, *, namespace, scope, capability="roles"):
        return self.invoke(capability, sentences, operation="ingest", namespace=namespace, scope=scope)

    def complete_tokens(self, tokens, *, namespace, scope, capability="roles"):
        return self.invoke(capability, tokens, namespace=namespace, scope=scope)

    def chain_tokens(self, start, templates, *, namespace, scope, capability="roles"):
        return self.invoke(capability, start, templates, operation="chain", namespace=namespace, scope=scope)

    def project_roles(self, relation, bindings, *, namespace, scope, capability="roles"):
        """Unbound argument roles remain wildcards in the native consumer."""
        return self.invoke(capability, relation, bindings, operation="lookup", namespace=namespace, scope=scope)

    def relation_address(self, relation, *, capability="roles"):
        field = self.role_model(capability)
        charts = [c for c in field.charts if c["relation"] == relation]
        if not charts:
            raise KeyError(relation)
        identity = self.component(capability).state["model"].grammar_sha256
        return {
            "capability": capability,
            "grammar_sha256": identity,
            "relation": relation,
            "arity": len(charts[0]["alignment"]),
        }

    def import_binary_evidence(self, evidence, *, capability="roles"):
        """Explicit legacy triple -> learned property chart adapter.

        The adapter supplies the surface form. The native charts determine its
        role alignment. Source and epistemic status survive; ranks/scores must
        first be deliberately converted to Evidence by the caller.
        Cross-namespace endpoints are refused, rather than aliasing their names.
        """
        if evidence.subject.namespace != evidence.object.namespace:
            raise ValueError("binary import requires endpoints in one namespace")
        component = self.component(capability)
        field = self.role_model(capability)
        tokens = ["the", evidence.relation, "of", evidence.subject.name, "is", evidence.object.name, "."]
        candidates = {(c["relation"], roles) for c, roles in field.matches(tokens)}
        if len(candidates) != 1:
            return {"status": "ambiguous" if candidates else "missing", "imported": False}
        relation, roles = next(iter(candidates))
        record = Record(
            evidence.subject.namespace, evidence.scope, relation, roles, evidence.source, len(field.records)
        )
        field.records.append(record)
        field._indices.clear()
        component.state.setdefault("evidence_metadata", []).append(
            {**record.__dict__, "status": evidence.status, "adapter": "supplied-property-surface"}
        )
        self.memory.write(evidence)
        return {
            "status": "resolved",
            "imported": True,
            "record": record,
            "address": self.relation_address(relation, capability=capability),
        }
