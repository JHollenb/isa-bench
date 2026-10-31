"""A portable, lazy federation of the native FieldLM engines.

The capability name is an explicit API request, not a learned task router. Native
engines keep their representations and consumers. Only typed records cross the
shared-memory boundary; likelihoods and ranking scores never become facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import pickle
from pathlib import Path


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True, order=True)
class Entity:
    namespace: str
    name: str


@dataclass(frozen=True)
class Evidence:
    subject: Entity
    relation: str
    object: Entity
    scope: str
    source: str
    status: str = "inferred"


@dataclass
class SharedMemory:
    """Scope-local, directional records with explicit cardinality.

    More than one value is a conflict only for a declared functional relation.
    Cross-source agreement does not certify that a reader's inference is true.
    """
    functional: set[str] = field(default_factory=set)
    records: list[Evidence] = field(default_factory=list)

    def write(self, record):
        if record.status not in {"observed", "inferred", "supplied"}:
            raise ValueError("unknown evidence status")
        if record not in self.records:
            self.records.append(record)

    def read(self, subject, relation, *, scope):
        rows = [r for r in self.records if (r.subject, r.relation, r.scope) == (subject, relation, scope)]
        values = sorted({r.object for r in rows})
        status = "missing" if not values else "ambiguous" if relation in self.functional and len(values) > 1 else "resolved"
        return {"status": status, "values": values if status == "resolved" else [], "candidates": values, "evidence": rows}

    def follow(self, subject, relations, *, scope):
        """Supplied exact join over records; this does not learn composition."""
        frontier, evidence = {subject}, []
        for relation in relations:
            following = set()
            for node in sorted(frontier):
                result = self.read(node, relation, scope=scope)
                evidence.extend(result["evidence"])
                if result["status"] != "resolved":
                    return {"status": result["status"], "values": [], "evidence": evidence}
                following.update(result["values"])
            frontier = following
        return {"status": "resolved", "values": sorted(frontier), "evidence": evidence}


@dataclass
class NativeComponent:
    kind: str
    state: dict
    provenance: dict

    def invoke(self, *args, **kwargs):
        if self.kind == "sparse_reader":
            operation = kwargs.pop("operation", "predict")
            if operation not in {"predict", "populate", "solve"}:
                raise ValueError("unsupported sparse FieldLM operation")
            return getattr(self.state["model"], operation)(*args, **kwargs)
        if self.kind == "wiki_joint":
            operation = kwargs.pop("operation", "predict")
            if operation not in {"predict", "populate", "solve", "solve_competing"}:
                raise ValueError("unsupported joint Wikipedia field operation")
            return getattr(self.state["model"], operation)(*args, **kwargs)
        if self.kind == "wiki":
            operation = kwargs.pop("operation", "predict")
            if operation == "predict":
                return self.state["reader"].predict(*args, **kwargs)
            if operation == "populate":
                return self.state["reader"].populate(self.state["model"].fabric, *args, **kwargs)
            if operation == "solve":
                return self.state["model"].solve(*args, **kwargs)
            raise ValueError("unsupported Wikipedia field operation")
        if self.kind == "fabric":
            operation = kwargs.pop("operation", "solve")
            bank = self.state["model"]
            if operation == "solve":
                return bank.solve(*args, **kwargs)
            if operation not in {"run", "observe_raw", "materialized"}:
                raise ValueError("unsupported field fabric operation")
            return getattr(bank.fabric, operation)(*args, **kwargs)
        if self.kind == "grammar":
            operation = kwargs.pop("operation", "complete")
            if operation == "lookup":
                return self.state["model"].field.lookup(*args, **kwargs)
            if operation not in {"parse", "complete", "chain", "ingest"}:
                raise ValueError("unsupported structural grammar operation")
            return getattr(self.state["model"], operation)(*args, **kwargs)
        if self.kind == "roles":
            operation = kwargs.pop("operation", "complete")
            if operation not in {"complete", "chain", "lookup", "matches", "ingest"}:
                raise ValueError("unsupported role-chart operation")
            return getattr(self.state["model"], operation)(*args, **kwargs)
        if self.kind in {"world", "world2"}:
            return self.state["model"].answer(*args, **kwargs)
        if self.kind == "kinship":
            # The native model consumes text/query, never diagnosis-only gold.
            return self.state["model"].predict(*args, **kwargs)
        if self.kind == "language":
            from .ordinary_language_structure import record
            pattern, left, right = args
            pred, rule = self.state["model"].predict(pattern, kwargs.get("kinds", ("entity", "entity")))
            return {"record": record(pred, left, right, self.state["label_names"]), "label": pred, "rule": rule}
        if self.kind == "attachment":
            from scripts.run_language_attachment import prepare, predict_ranked
            row, = args
            weight = self.state["distance_weight"]
            _, ranked = prepare(self.state["model"], [row], self.state["mode"], self.state["roles"], [weight])
            return predict_ranked(self.state["model"], ranked[weight])[0]
        if self.kind == "rule":
            from .fieldlm_integration import authoritative_rollout
            return authoritative_rollout(self.state["model"], *args, **kwargs)
        if self.kind == "joint":
            from .fieldlm_integration import authoritative_rollout
            cell, = args
            model = self.state["model"]
            return authoritative_rollout(model.program, model.hard_batch(cell), **kwargs)
        if self.kind == "kg":
            from .kg_field import score_batch
            from .kg2 import feature_batch
            q, xs = args
            program = self.state["model"]
            if "readout" in self.state:
                extra = self.state["rotate"].scores(q, xs) if "rotate" in self.state else None
                return self.state["readout"].score(feature_batch(program, q, xs), extra)
            return score_batch(program, q, xs)
        if self.kind == "bytes":
            return self.state["model"].bits(*args, **kwargs)
        if self.kind == "chart":
            import torch
            from .chart_field import model_target_probs
            return model_target_probs(self.state["chart"], self.state["readout"], *args,
                                      device=torch.device("cpu"), **kwargs)
        raise KeyError(self.kind)


class UnifiedFieldLM:
    """One runtime and bundle, with engines loaded only when requested."""
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["schema"] != "unified-fieldlm-v1":
            raise ValueError("unsupported bundle")
        self.components = {}
        self.memory = SharedMemory()

    @property
    def capabilities(self):
        return tuple(self.manifest["components"])

    def component(self, name):
        if name not in self.components:
            entry = self.manifest["components"][name]
            path = self.root / entry["path"]
            if path.parent.resolve() != (self.root / "states").resolve() or file_sha(path) != entry["sha256"]:
                raise ValueError("checkpoint integrity mismatch")
            # Only load trusted, source-bound internal captures. Pickle is not a
            # format for accepting untrusted third-party model files.
            with path.open("rb") as stream:
                self.components[name] = pickle.load(stream)
            component = self.components[name]
            if "model_ref" in component.state:
                component.state["model"] = self.component(component.state["model_ref"]).state["model"]
        return self.components[name]

    def invoke(self, capability, *args, **kwargs):
        return self.component(capability).invoke(*args, **kwargs)

    def release(self, capability):
        component = self.components.pop(capability, None)
        if component and (ref := component.state.get("model_ref")):
            if not any(c.state.get("model_ref") == ref for c in self.components.values()):
                self.release(ref)

    def extract(self, pattern, left, right, *, scope, namespace, kinds=("entity", "entity")):
        result = self.invoke("language", pattern, left, right, kinds=kinds)
        if result["record"]:
            row = result["record"]
            self.memory.write(Evidence(Entity(namespace, row["subject"]), row["relation"],
                                       Entity(namespace, row["object"]), scope,
                                       self.manifest["components"]["language"]["sha256"], "inferred"))
        return result


def save_component(root, name, component):
    root = Path(root)
    path = root / "states" / f"{name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        pickle.dump(component, stream, protocol=5)
    return {"path": str(path.relative_to(root)), "sha256": file_sha(path), "bytes": path.stat().st_size,
            "kind": component.kind, "provenance": component.provenance}
