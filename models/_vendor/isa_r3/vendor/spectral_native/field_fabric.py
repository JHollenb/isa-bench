"""Bounded composition over existing RoleChartField projected joins.

This is an experimental execution fabric, not a new phase/SPF primitive.  A
program carries named bindings through view-local relation coordinates.  Every
read delegates to :meth:`RoleChartField.lookup`; complete matching tuples are
kept as distinct branches until the requested return bindings are compared.

The serialized state is the fabric's explicit mutable closure: registered
views, programs, runtime frontiers, retained observations, and materialized
results.  Materializations bind the full program structure and exact
namespace/scope.  Context fingerprints make direct record edits visible even
when they bypass ``RoleChartField.ingest`` and its index invalidation.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .role_chart_field import Record, RoleChartField


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _json_value(value: Any) -> Any:
    """Copy a value into the deliberately small JSON state boundary."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    raise TypeError(f"fabric state accepts JSON values, got {type(value).__name__}")


@dataclass(frozen=True)
class RelationRef:
    """A relation coordinate qualified by its view-local address space."""

    view: str
    relation: int

    def state(self) -> dict[str, Any]:
        return {"view": self.view, "relation": self.relation}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> RelationRef:
        return cls(str(state["view"]), int(state["relation"]))


@dataclass(frozen=True)
class ReadStep:
    """One projected read and the variables written from its matching tuple."""

    view: str
    relation: int | RelationRef
    bindings: Mapping[int, str]
    outputs: Mapping[int, str]

    def __post_init__(self) -> None:
        if not self.view:
            raise ValueError("read view must be non-empty")
        if isinstance(self.relation, RelationRef) and self.relation.view != self.view:
            raise ValueError("relation coordinates are local to their declared view")
        bindings = {int(axis): str(value) for axis, value in sorted(self.bindings.items())}
        outputs = {int(axis): str(value) for axis, value in sorted(self.outputs.items())}
        if any(axis < 0 for axis in (*bindings, *outputs)):
            raise ValueError("role axes must be non-negative")
        if any(not variable or variable.startswith("$") for variable in outputs.values()):
            raise ValueError("output names are bare variable names")
        object.__setattr__(self, "bindings", bindings)
        object.__setattr__(self, "outputs", outputs)

    @property
    def relation_id(self) -> int:
        return self.relation.relation if isinstance(self.relation, RelationRef) else int(self.relation)

    def state(self) -> dict[str, Any]:
        relation = (
            self.relation.state()
            if isinstance(self.relation, RelationRef)
            else {"view": self.view, "relation": int(self.relation)}
        )
        return {
            "view": self.view,
            "relation": relation,
            "bindings": [{"axis": axis, "value": value} for axis, value in self.bindings.items()],
            "outputs": [{"axis": axis, "variable": value} for axis, value in self.outputs.items()],
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> ReadStep:
        relation = RelationRef.from_state(state["relation"])
        return cls(
            str(state["view"]),
            relation,
            {int(item["axis"]): str(item["value"]) for item in state["bindings"]},
            {int(item["axis"]): str(item["variable"]) for item in state["outputs"]},
        )


@dataclass(frozen=True)
class FabricProgram:
    """An addressable sequence of reads with named returned bindings."""

    name: str
    steps: tuple[ReadStep, ...]
    returns: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("program name must be non-empty")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "returns", tuple(str(value) for value in self.returns))
        if not self.returns or any(not value or value.startswith("$") for value in self.returns):
            raise ValueError("program returns are non-empty bare variable names")
        if len(set(self.returns)) != len(self.returns):
            raise ValueError("program returns must be unique")

    @property
    def fingerprint(self) -> str:
        return _digest(self.state())

    def state(self) -> dict[str, Any]:
        return {"name": self.name, "steps": [step.state() for step in self.steps], "returns": list(self.returns)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FabricProgram:
        return cls(
            str(state["name"]),
            tuple(ReadStep.from_state(step) for step in state["steps"]),
            tuple(str(value) for value in state["returns"]),
        )


@dataclass
class BindingFrame:
    """Branch-local recurrent working memory for one program frontier."""

    bindings: dict[str, str]
    sources: tuple[str, ...] = ()
    dependencies: tuple[dict[str, Any], ...] = ()
    trace: tuple[dict[str, Any], ...] = ()
    cursor: int = 0
    status: str = "running"

    def state(self) -> dict[str, Any]:
        return {
            "bindings": dict(self.bindings),
            "sources": list(self.sources),
            "dependencies": _json_value(self.dependencies),
            "trace": _json_value(self.trace),
            "cursor": self.cursor,
            "status": self.status,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> BindingFrame:
        return cls(
            {str(key): str(value) for key, value in state["bindings"].items()},
            tuple(str(value) for value in state.get("sources", [])),
            tuple(copy.deepcopy(state.get("dependencies", []))),
            tuple(copy.deepcopy(state.get("trace", []))),
            int(state.get("cursor", 0)),
            str(state.get("status", "running")),
        )


class _OwnedView(RoleChartField):
    """A copied view whose record collection can only be replaced by the fabric."""

    @property
    def records(self):
        return self._owned_records

    @records.setter
    def records(self, value):
        raise TypeError("managed records are read-only; use FieldFabric.mutate_records")

    @classmethod
    def copy_of(cls, view):
        obj = cls.__new__(cls)
        state = copy.deepcopy(view.state())
        obj.charts = state["charts"]
        obj.classes = state["classes"]
        obj.merge_log = state["merge_log"]
        obj.role_class_merges = state["role_class_merges"]
        obj.active_axes = None if state["active_axes"] is None else set(state["active_axes"])
        obj._owned_records = tuple(Record(**{**row, "roles": tuple(row["roles"])}) for row in state["records"])
        obj._indices = {}
        return obj


class FieldFabric:
    """Registry, bounded executor, working memory, and materialized-view cache."""

    STATE_VERSION = 1

    def __init__(self) -> None:
        self.views: dict[str, RoleChartField] = {}
        self.programs: dict[str, FabricProgram] = {}
        self.frames: dict[str, list[BindingFrame]] = {}
        self.abstractions: dict[str, dict[str, Any]] = {}
        self.raw_observations: list[dict[str, Any]] = []
        self.provisional_observations: list[dict[str, Any]] = []
        self._view_fingerprints: dict[str, str] = {}
        self._managed_views: set[str] = set()

    def register_view(self, name: str, view: RoleChartField) -> None:
        if not name:
            raise ValueError("view name must be non-empty")
        if not isinstance(view, RoleChartField):
            raise TypeError("field fabric views must be RoleChartField instances")
        self.views[name] = view
        self._managed_views.discard(name)
        view._indices.clear()
        self._view_fingerprints[name] = self._view_fingerprint(view)

    def view(self, name: str) -> RoleChartField:
        try:
            return self.views[name]
        except KeyError as exc:
            raise KeyError(f"unknown field view {name!r}") from exc

    def invalidate_view(self, name: str) -> None:
        view = self.view(name)
        view._indices.clear()
        self._view_fingerprints[name] = self._validation_token(name)

    def manage_view(self, name: str) -> None:
        """Take a private copy and opt into query-dependency validation.

        The exposed record tuple is read-only. Existing caller-owned aliases do
        not mutate this copy. Grammar and active-axis changes are still detected.
        Ordinary registered views retain full-view validation for direct writers.
        """
        if name not in self._managed_views:
            self.views[name] = _OwnedView.copy_of(self.view(name))
            self._managed_views.add(name)
            self.invalidate_view(name)

    def mutate_records(self, name: str, *, add=(), remove=()) -> dict[str, int]:
        """Atomically replace records and invalidate lookup indexes.

        Removals use complete Record identity, including provenance and position.
        Every requested removal must exist; invalid batches change nothing.
        """
        view = self.view(name)
        additions, removals = list(add), list(remove)
        for record in (*additions, *removals):
            if (
                not isinstance(record, Record)
                or type(record.relation) is not int
                or record.relation < 0
                or type(record.position) is not int
                or record.position < 0
                or not isinstance(record.roles, tuple)
                or any(not isinstance(value, str) for value in record.roles)
                or any(not isinstance(value, str) for value in (record.namespace, record.scope, record.source))
            ):
                raise ValueError("invalid native record")
        records = list(view.records)
        for record in removals:
            try:
                records.remove(record)
            except ValueError as exc:
                raise ValueError("removed record is absent") from exc
        records.extend(additions)
        if name in self._managed_views:
            view._owned_records = tuple(records)
        else:
            view.records = records
        self.invalidate_view(name)
        return {"added": len(additions), "removed": len(removals)}

    def register_program(self, program: FabricProgram) -> None:
        if not isinstance(program, FabricProgram):
            raise TypeError("program must be a FabricProgram")
        self._validate_program(program)
        self.programs[program.name] = program

    def observe_raw(
        self,
        view: str,
        observation: Any,
        namespace: str,
        scope: str,
        source: str | None = None,
        provisional: bool = True,
        status: str | None = None,
    ) -> dict[str, Any]:
        """Retain an observation without parsing or asserting it as a field fact."""
        self.view(view)
        payload = _json_value(observation)
        status = status or ("provisional" if provisional else "raw")
        if status in {"asserted", "fact", "resolved"}:
            raise ValueError("raw observations cannot be recorded as asserted facts")
        source = source or _digest(
            {
                "view": view,
                "namespace": namespace,
                "scope": scope,
                "observation": payload,
                "ordinal": len(self.raw_observations),
            }
        )
        record = {
            "view": view,
            "namespace": namespace,
            "scope": scope,
            "source": source,
            "status": status,
            "observation": payload,
        }
        self.raw_observations.append(copy.deepcopy(record))
        if provisional:
            self.provisional_observations.append(copy.deepcopy(record))
        return copy.deepcopy(record)

    def run(
        self,
        program: str | FabricProgram,
        inputs: Mapping[str, str],
        namespace: str,
        scope: str,
        budget: int | None = None,
    ) -> dict[str, Any]:
        """Run a stored or supplied program with an explicit logical-read budget."""
        selected = self._program(program)
        self._validate_program(selected)
        initial = {str(key): str(value) for key, value in inputs.items()}
        if budget is None:
            budget = len(selected.steps)
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer or None")

        changed = self._refresh_view_indices()
        address = self._address(selected, initial, namespace, scope, budget)
        cached = self.abstractions.get(address)
        predicate_mode = all(step.view in self._managed_views for step in selected.steps)
        mode = "predicates" if predicate_mode else "full-view"
        fingerprints = (
            self._predicate_fingerprints(cached.get("predicates", []))
            if predicate_mode and cached is not None
            else self._dependency_fingerprints(selected, namespace, scope)
            if not predicate_mode
            else []
        )
        if (
            cached is not None
            and cached.get("dependency_mode", "full-view") == mode
            and cached["fingerprints"] == fingerprints
        ):
            result = copy.deepcopy(cached["result"])
            result["cached"] = True
            return result

        frames = [BindingFrame(dict(initial))]
        run_trace: list[dict[str, Any]] = []
        limit = min(budget, len(selected.steps))
        for cursor, step in enumerate(selected.steps[:limit]):
            next_frames: list[BindingFrame] = []
            branches: list[dict[str, Any]] = []
            field_view = self.views[step.view]
            for frame in frames:
                query, missing = self._query_bindings(step, frame)
                records = (
                    [] if missing else field_view.lookup(step.relation_id, query, namespace=namespace, scope=scope)
                )
                branch = {
                    "input_bindings": dict(frame.bindings),
                    "projected_bindings": {str(axis): value for axis, value in query.items()},
                    "missing_variables": missing,
                    "matches": [self._record_state(record) for record in records],
                }
                branches.append(branch)
                for record in records:
                    candidate = self._extend_frame(frame, step, record, cursor, namespace, scope)
                    if candidate is not None:
                        next_frames.append(candidate)
            run_trace.append(
                {
                    "cursor": cursor,
                    "read": {"view": step.view, "relation": step.relation_id, "namespace": namespace, "scope": scope},
                    "branches": branches,
                }
            )
            frames = next_frames
            if not frames:
                break

        exhausted = bool(frames) and limit < len(selected.steps)
        if exhausted:
            for frame in frames:
                frame.status = "budget-exhausted"
            result = self._frontier_result(selected, frames, run_trace, limit)
        else:
            result = self._terminal_result(selected, frames, run_trace)
        predicates = self._trace_predicates(run_trace) if predicate_mode else []
        if predicate_mode:
            fingerprints = self._predicate_fingerprints(predicates)
        sources = sorted({source for frame in frames for source in frame.sources})
        result.update(
            {
                "address": address,
                "cached": False,
                "program_fingerprint": selected.fingerprint,
                "fingerprints": copy.deepcopy(fingerprints),
                "sources": result.get("sources", sources),
            }
        )
        if cached is not None or changed.intersection(step.view for step in selected.steps):
            result["recomputed"] = True

        self.frames[address] = copy.deepcopy(frames)
        self.abstractions[address] = {
            "address": address,
            "program": selected.state(),
            "inputs": dict(initial),
            "namespace": namespace,
            "scope": scope,
            "budget": budget,
            "sources": sources,
            "fingerprints": fingerprints,
            "dependency_mode": mode,
            "predicates": predicates,
            "result": copy.deepcopy(result),
        }
        return copy.deepcopy(result)

    def materialized(self, address: str) -> dict[str, Any] | None:
        """Load an abstraction, recomputing it if a dependency has changed."""
        materialized = self.abstractions.get(address)
        if materialized is None:
            return None
        program = FabricProgram.from_state(materialized["program"])
        return self.run(
            program, materialized["inputs"], materialized["namespace"], materialized["scope"], materialized["budget"]
        )

    def state(self) -> dict[str, Any]:
        state = {
            "version": self.STATE_VERSION,
            "views": {name: view.state() for name, view in sorted(self.views.items())},
            "programs": {name: program.state() for name, program in sorted(self.programs.items())},
            "frames": {address: [frame.state() for frame in frames] for address, frames in sorted(self.frames.items())},
            "abstractions": copy.deepcopy(self.abstractions),
            "raw_observations": copy.deepcopy(self.raw_observations),
            "provisional_observations": copy.deepcopy(self.provisional_observations),
        }
        if self._managed_views:
            state["managed_views"] = sorted(self._managed_views)
        json.dumps(state, sort_keys=True)
        return state

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> FieldFabric:
        if int(state.get("version", -1)) != cls.STATE_VERSION:
            raise ValueError("unsupported field fabric state version")
        fabric = cls()
        for name, view_state in state.get("views", {}).items():
            fabric.register_view(str(name), RoleChartField.from_state(copy.deepcopy(view_state)))
        for name in state.get("managed_views", []):
            fabric.manage_view(str(name))
        for name, program_state in state.get("programs", {}).items():
            program = FabricProgram.from_state(program_state)
            if program.name != name:
                raise ValueError("program registry key disagrees with serialized program name")
            fabric.register_program(program)
        fabric.frames = {
            str(address): [BindingFrame.from_state(frame) for frame in frames]
            for address, frames in state.get("frames", {}).items()
        }
        fabric.abstractions = copy.deepcopy(state.get("abstractions", {}))
        fabric.raw_observations = copy.deepcopy(state.get("raw_observations", []))
        fabric.provisional_observations = copy.deepcopy(state.get("provisional_observations", []))
        return fabric

    def _program(self, program: str | FabricProgram) -> FabricProgram:
        if isinstance(program, FabricProgram):
            return program
        try:
            return self.programs[program]
        except KeyError as exc:
            raise KeyError(f"unknown fabric program {program!r}") from exc

    def _validate_program(self, program: FabricProgram) -> None:
        for step in program.steps:
            if step.view not in self.views:
                raise KeyError(f"program {program.name!r} refers to unknown view {step.view!r}")

    @staticmethod
    def _query_bindings(step: ReadStep, frame: BindingFrame) -> tuple[dict[int, str], list[str]]:
        query: dict[int, str] = {}
        missing: list[str] = []
        for axis, expression in step.bindings.items():
            if expression.startswith("$"):
                variable = expression[1:]
                if variable not in frame.bindings:
                    missing.append(variable)
                    continue
                query[axis] = frame.bindings[variable]
            else:
                query[axis] = expression
        return query, missing

    @staticmethod
    def _record_state(record: Record) -> dict[str, Any]:
        return {
            "namespace": record.namespace,
            "scope": record.scope,
            "relation": record.relation,
            "roles": list(record.roles),
            "source": record.source,
            "position": record.position,
        }

    def _extend_frame(
        self,
        frame: BindingFrame,
        step: ReadStep,
        record: Record,
        cursor: int,
        namespace: str,
        scope: str,
    ) -> BindingFrame | None:
        bindings = dict(frame.bindings)
        for axis, variable in step.outputs.items():
            if axis >= len(record.roles):
                raise ValueError(f"view {step.view!r} relation {step.relation_id} has no role axis {axis}")
            value = record.roles[axis]
            if variable in bindings and bindings[variable] != value:
                return None
            bindings[variable] = value
        dependency = {
            "view": step.view,
            "relation": step.relation_id,
            "namespace": namespace,
            "scope": scope,
            "source": record.source,
            "roles": list(record.roles),
        }
        event = {
            "cursor": cursor,
            "read": RelationRef(step.view, step.relation_id).state(),
            "record": self._record_state(record),
            "bindings": dict(bindings),
        }
        sources = tuple(dict.fromkeys((*frame.sources, record.source)))
        return BindingFrame(
            bindings, sources, (*frame.dependencies, dependency), (*frame.trace, event), cursor + 1, "running"
        )

    @staticmethod
    def _candidate(frame: BindingFrame, names: tuple[str, ...] | None = None) -> dict[str, Any]:
        bindings = (
            frame.bindings
            if names is None
            else {name: frame.bindings[name] for name in names if name in frame.bindings}
        )
        return {
            "bindings": dict(bindings),
            "sources": list(frame.sources),
            "dependencies": _json_value(frame.dependencies),
            "trace": _json_value(frame.trace),
            "cursor": frame.cursor,
            "status": frame.status,
        }

    def _frontier_result(
        self,
        program: FabricProgram,
        frames: list[BindingFrame],
        trace: list[dict[str, Any]],
        cursor: int,
    ) -> dict[str, Any]:
        candidates = [self._candidate(frame) for frame in frames]
        result: dict[str, Any] = {
            "status": "budget-exhausted",
            "value": None,
            "candidates": candidates,
            "trace": trace,
            "cursor": cursor,
            "remaining": len(program.steps) - cursor,
        }
        if len(candidates) == 1:
            result["bindings"] = candidates[0]["bindings"]
        return result

    def _terminal_result(
        self,
        program: FabricProgram,
        frames: list[BindingFrame],
        trace: list[dict[str, Any]],
    ) -> dict[str, Any]:
        complete = [frame for frame in frames if all(name in frame.bindings for name in program.returns)]
        if not complete:
            return {"status": "missing", "value": None, "candidates": [], "trace": trace}

        grouped: dict[tuple[str, ...], list[BindingFrame]] = {}
        for frame in complete:
            grouped.setdefault(tuple(frame.bindings[name] for name in program.returns), []).append(frame)
        candidates = []
        for values, alternatives in sorted(grouped.items()):
            binding = dict(zip(program.returns, values, strict=True))
            candidates.append(
                {
                    "bindings": binding,
                    "sources": sorted({source for frame in alternatives for source in frame.sources}),
                    "alternatives": [self._candidate(frame, program.returns) for frame in alternatives],
                }
            )
        if len(candidates) != 1:
            return {"status": "ambiguous", "value": None, "candidates": candidates, "trace": trace}

        resolved = candidates[0]
        result: dict[str, Any] = {
            "status": "resolved",
            "bindings": resolved["bindings"],
            "sources": resolved["sources"],
            "dependencies": resolved["alternatives"][0]["dependencies"],
            "trace": trace,
        }
        result["value"] = resolved["bindings"][program.returns[0]] if len(program.returns) == 1 else None
        return result

    @staticmethod
    def _view_fingerprint(view: RoleChartField, namespace: str | None = None, scope: str | None = None) -> str:
        state = view.state()
        if namespace is not None and scope is not None:
            state = {
                **state,
                "records": [row for row in state["records"] if row["namespace"] == namespace and row["scope"] == scope],
            }
        return _digest(state)

    def _refresh_view_indices(self) -> set[str]:
        changed = set()
        for name, view in self.views.items():
            current = self._validation_token(name)
            if self._view_fingerprints.get(name) != current:
                view._indices.clear()
                self._view_fingerprints[name] = current
                changed.add(name)
        return changed

    def _validation_token(self, name: str) -> str:
        view = self.views[name]
        if name not in self._managed_views:
            return self._view_fingerprint(view)
        return _digest(
            {
                "charts": view.charts,
                "classes": view.classes,
                "merge_log": view.merge_log,
                "role_class_merges": view.role_class_merges,
                "active_axes": None if view.active_axes is None else sorted(view.active_axes),
            }
        )

    @staticmethod
    def _trace_predicates(trace):
        predicates = {}
        for read in trace:
            for branch in read["branches"]:
                if not branch["missing_variables"]:
                    item = {**read["read"], "bindings": branch["projected_bindings"]}
                    predicates[_digest(item)] = item
        return [predicates[key] for key in sorted(predicates)]

    def _predicate_fingerprints(self, predicates):
        out = []
        for item in predicates:
            records = self.view(item["view"]).lookup(
                item["relation"],
                {int(axis): value for axis, value in item["bindings"].items()},
                namespace=item["namespace"],
                scope=item["scope"],
            )
            out.append(
                {
                    "predicate": copy.deepcopy(item),
                    "fingerprint": _digest(
                        {
                            "metadata": self._validation_token(item["view"]),
                            "records": [self._record_state(record) for record in records],
                        }
                    ),
                }
            )
        return out

    def _dependency_fingerprints(self, program: FabricProgram, namespace: str, scope: str) -> list[dict[str, str]]:
        return [
            {
                "view": name,
                "namespace": namespace,
                "scope": scope,
                "fingerprint": self._view_fingerprint(self.views[name], namespace, scope),
            }
            for name in sorted({step.view for step in program.steps})
        ]

    @staticmethod
    def _address(
        program: FabricProgram,
        inputs: Mapping[str, str],
        namespace: str,
        scope: str,
        budget: int,
    ) -> str:
        identity = {
            "program": program.state(),
            "inputs": dict(sorted(inputs.items())),
            "namespace": namespace,
            "scope": scope,
            "budget": budget,
        }
        return f"field://program/{program.name}/{_digest(identity)}"


__all__ = ["BindingFrame", "FabricProgram", "FieldFabric", "ReadStep", "RelationRef"]
