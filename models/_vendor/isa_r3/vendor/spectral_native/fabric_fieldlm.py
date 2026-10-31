"""Consolidated entry point for learned programs over shared Field bindings.

Native role views share their existing record objects and projected consumer.
Attaching a view supplies access; source feedback must identify useful programs
and coordinate alignments. There is no inferred universal predicate inventory.
"""

from __future__ import annotations

from spectral_native.axis_fieldlm import AxisFieldLM


class FabricFieldLM(AxisFieldLM):
    def fabric_model(self, capability="fabric"):
        component = self.component(capability)
        if component.kind != "fabric":
            raise ValueError("a learned field fabric capability is required")
        return component.state["model"]

    def reason(self, task, inputs, *, namespace, scope, budget=16, context=None, capability="fabric"):
        """Select a source-supported program and carry its predicted bindings."""
        return self.invoke(capability, task, inputs, namespace=namespace, scope=scope, budget=budget, context=context)

    def run_program(self, program, inputs, *, namespace, scope, budget=16, capability="fabric"):
        return self.invoke(
            capability, program, inputs, namespace=namespace, scope=scope, budget=budget, operation="run"
        )

    def write_observation(self, observation, *, view, namespace, scope, source=None, capability="fabric"):
        """Retain a partial observation without authorizing a world fact."""
        return self.invoke(
            capability,
            view,
            observation,
            namespace=namespace,
            scope=scope,
            source=source,
            provisional=True,
            operation="observe_raw",
        )

    def abstraction(self, address, *, capability="fabric"):
        return self.invoke(capability, address, operation="materialized")

    def attach_role_view(self, name, *, role_capability="grammar", capability="fabric"):
        """Share an existing arbitrary-arity view, preserving its local IDs.

        This operation provides the native data boundary. It does not supply a
        semantic route from an unrelated task to the newly attached view.
        """
        bank = self.fabric_model(capability)
        bank.fabric.register_view(name, self.role_model(role_capability))
        return {
            "view": name,
            "capability": role_capability,
            "grammar": self.component(role_capability).state["model"].grammar_sha256,
        }
