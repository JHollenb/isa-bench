"""Default research FieldLM: semantic relation proposals, native direction and joins.

Sparse vectors come from the pinned external encoder declared in the bundle.
Mentions, kinds, scopes and query relations are supplied. This runtime does not
claim to learn mention extraction, negation attachment or the join program.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .field_fabric import FabricProgram, FieldFabric, ReadStep
from .gate_runtime import GateRuntime
from .retrieval_attachment import FactorizedExampleReader
from .role_chart_field import Record, RoleChartField
from .unified_fieldlm import NativeComponent, SharedMemory, file_sha
from .wiki_joint_fieldlm import WikiJointFieldLM

DEFAULT_POLICY = {"confidence": 0.55, "compatibility": True, "direction_mode": "selected"}


class SparseFieldModel:
    """A source-specific reader bank and its explicit mutable native closure."""

    def __init__(self, reader, policy=None, fabric=None):
        self.reader = reader
        self.policy = dict(DEFAULT_POLICY if policy is None else policy)
        if set(self.policy) != set(DEFAULT_POLICY):
            raise ValueError("incomplete reader policy")
        if not 0 <= self.policy["confidence"] <= 1 or self.policy["direction_mode"] not in {"selected", "first"}:
            raise ValueError("invalid reader policy")
        self.fabric = fabric if fabric is not None else FieldFabric()
        if "facts" not in self.fabric.views:
            self.fabric.register_view(
                "facts",
                RoleChartField.from_state(
                    {
                        "charts": [],
                        "classes": {},
                        "merge_log": [],
                        "role_class_merges": [],
                        "records": [],
                        "active_axes": None,
                    }
                ),
            )
        self.fabric.manage_view("facts")
        self.gates = GateRuntime(self.fabric)

    @classmethod
    def fit(cls, source, document_vectors, *, policy=None):
        return cls(FactorizedExampleReader.fit(source, document_vectors), policy)

    def predict(self, row, vector):
        if any(row.get(key) for key in ("labels", "gold", "expected")):
            raise ValueError("serving rows must be label-free")
        if any(int(atom) < 0 or not math.isfinite(float(value)) or float(value) < 0 for atom, value in vector.items()):
            raise ValueError("sparse impacts must be finite and non-negative")
        return self.reader.predict(row, vector, **self.policy)

    def populate(self, rows, vectors, *, namespace, scope=None):
        """Infer first, then admit selected records; candidates remain observations.

        A supplied scope overrides article scopes. Without it, each row retains
        its own article_id. Batch shape/serving errors leave memory unchanged.
        """
        rows, vectors = list(rows), list(vectors)
        predictions = [self.predict(row, vector) for row, vector in zip(rows, vectors, strict=True)]
        observations, records = [], []
        field = self.fabric.view("facts")
        for row, prediction in zip(rows, predictions, strict=True):
            local_scope = str(scope if scope is not None else row["article_id"])
            source = str(row["pair_id"])
            selected = prediction["selected"]
            if selected:
                relation, direction = selected.rsplit(":", 1)
                a, b = (row["left"], row["right"]) if direction == "f" else (row["right"], row["left"])
                records.append(
                    Record(namespace, local_scope, 0, (relation, a, b), source, len(field.records) + len(records))
                )
            observations.append((local_scope, source, copy.deepcopy(row), prediction))
        for local_scope, source, row, prediction in observations:
            self.fabric.observe_raw(
                "facts",
                {"row": row, "prediction": prediction},
                namespace,
                local_scope,
                source=source,
                provisional=False,
                status="reader-inference",
            )
        self.fabric.mutate_records("facts", add=records)
        return {
            "predictions": predictions,
            "records": [self.fabric._record_state(r) for r in records],
            "evidence_status": "inferred",
        }

    def solve(self, start, relations, *, namespace, scope, budget=None):
        """Supplied exact directional join over reader-selected records."""
        relations = tuple(relations)
        if not relations:
            raise ValueError("at least one relation is required")
        steps = tuple(
            ReadStep("facts", 0, {0: relation, 1: f"$node{i}"}, {2: f"node{i + 1}"})
            for i, relation in enumerate(relations)
        )
        program = FabricProgram("base-follow", steps, (f"node{len(relations)}",))
        return self.fabric.run(program, {"node0": start}, namespace, scope, budget=budget)

    def state(self):
        return json.loads(
            json.dumps(
                {
                    "schema": "fieldlm-sparse-base-state-v1",
                    "reader": self.reader.state(),
                    "policy": dict(self.policy),
                    "fabric": self.fabric.state(),
                    "gates": {key: value for key, value in self.gates.state().items() if key != "fabric"},
                }
            )
        )

    @classmethod
    def from_state(cls, state):
        if state["schema"] != "fieldlm-sparse-base-state-v1":
            raise ValueError("unsupported sparse base state")
        obj = cls(
            FactorizedExampleReader.from_state(state["reader"]),
            state["policy"],
            FieldFabric.from_state(state["fabric"]),
        )
        if "gates" in state:
            obj.gates = GateRuntime.from_state({**state["gates"], "fabric": obj.fabric.state()}, fabric=obj.fabric)
        return obj


class FieldLM(WikiJointFieldLM):
    """Load the selected research base by default, or an explicit family bundle.

    The new default uses JSON captures. Existing unified family bundles retain
    their explicit capability APIs and are accepted through ``root``.
    """

    def __init__(self, root=None):
        if root is None:
            models = Path(__file__).resolve().parents[1] / "models"
            selection = json.loads((models / "base.json").read_text())
            root = models / selection["bundle"]
            if file_sha(root / "manifest.json") != selection["manifest_sha256"]:
                raise ValueError("base selection integrity mismatch")
        super().__init__(root)

    def component(self, name):
        if name in self.components:
            return self.components[name]
        entry = self.manifest["components"][name]
        if entry.get("format") != "json":
            return super().component(name)
        path = self.root / entry["path"]
        if path.parent.resolve() != (self.root / "states").resolve() or file_sha(path) != entry["sha256"]:
            raise ValueError("checkpoint integrity mismatch")
        state = json.loads(path.read_text())
        self.components[name] = NativeComponent(
            "sparse_reader", {"model": SparseFieldModel.from_state(state)}, entry["provenance"]
        )
        return self.components[name]

    def read_pair(self, row, vector, *, capability="sparse_reader"):
        return self.invoke(capability, row, vector, operation="predict")

    def observe_pairs(self, rows, vectors, *, namespace, scope=None, capability="sparse_reader"):
        return self.invoke(capability, rows, vectors, namespace=namespace, scope=scope, operation="populate")

    def follow(self, start, relations, *, namespace, scope, budget=None, capability="sparse_reader"):
        return self.invoke(
            capability, start, relations, namespace=namespace, scope=scope, budget=budget, operation="solve"
        )

    @classmethod
    def from_source(cls, source, document_vectors, *, policy=None):
        """Fit a new domain bank without silently reusing the controlled fixture."""
        obj = cls.__new__(cls)
        model = SparseFieldModel.fit(source, document_vectors, policy=policy)
        obj.root = None
        obj.manifest = {
            "schema": "unified-fieldlm-v1",
            "profiles": {"research_base": ["sparse_reader"]},
            "components": {"sparse_reader": {"kind": "sparse_reader"}},
        }
        obj.components = {
            "sparse_reader": NativeComponent(
                "sparse_reader", {"model": model}, {"capture": "caller-supplied-source-bank"}
            )
        }
        obj.memory = SharedMemory()
        return obj

    def export_state(self, *, capability="sparse_reader"):
        return copy.deepcopy(self.component(capability).state["model"].state())

    def gate_runtime(self, *, capability="sparse_reader"):
        """The base's owned graph, typed cartridges and reversible working memory."""
        component = self.component(capability)
        if component.kind != "sparse_reader":
            raise ValueError("gate runtime requires a sparse FieldLM component")
        return component.state["model"].gates

    def install_rule(self, name, cartridge, *, capability="sparse_reader"):
        return self.gate_runtime(capability=capability).install_rule(name, cartridge)

    def apply_rule(self, name, inputs, *, namespace, scope, budget=None, capability="sparse_reader"):
        return self.gate_runtime(capability=capability).execute_rule(
            name, inputs, namespace=namespace, scope=scope, budget=budget
        )

    def install_hypotheses(self, name, laws, *, namespace, scope, capability="sparse_reader"):
        return self.gate_runtime(capability=capability).install_bank(name, laws, namespace=namespace, scope=scope)

    def predict_quantity(self, name, quantities, *, namespace, scope, capability="sparse_reader"):
        return self.gate_runtime(capability=capability).predict_bank(name, quantities, namespace=namespace, scope=scope)

    def suggest_observation(self, name, menu, *, namespace, scope, capability="sparse_reader"):
        return self.gate_runtime(capability=capability).suggest_observation(
            name, menu, namespace=namespace, scope=scope
        )

    def record_observation(self, ticket, quantity, *, source, capability="sparse_reader"):
        return self.gate_runtime(capability=capability).record_observation(ticket, quantity, source=source)


__all__ = ["DEFAULT_POLICY", "FieldLM", "SparseFieldModel"]
