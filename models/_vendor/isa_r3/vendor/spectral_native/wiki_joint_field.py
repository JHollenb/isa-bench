"""Joint Wikipedia reading and bounded execution over the native Field fabric.

``WikiAxisField`` supplies source-induced relation interpretations.  This
wrapper keeps every eligible interpretation as a provisional native fact and
lets an already learned ``LearnedFabricBank`` consume those facts.  It never
uses target labels, treats source identifiers only as provenance, and returns
an answer only when every surviving complete answer branch agrees.  Dead-end
interpretations remain visible in the trace but do not vote as answers.

The branch preflight delegates projected reads and frame extension to the
existing ``FieldFabric`` implementation.  It is an execution guard around the
frozen program, not another join or SPF/phase primitive.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .field_fabric import BindingFrame, FieldFabric
from .field_fabric_learning import LearnedFabricBank
from .role_chart_field import Record, RoleChartField
from .wiki_axis_field import FACT_RELATION, FACT_SCHEMA, WikiAxisField, WikiPair

STATE_VERSION = 1
STATE_SCHEMA = "wiki-joint-field-v1"
_JOINT_AXES = frozenset({"word_sequence", "clause_scope", "polarity"})
_SERVING_LABEL_KEYS = frozenset({"answer", "answers", "expected", "gold", "label", "labels", "target"})


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _json_copy(value: Any) -> Any:
    result = copy.deepcopy(value)
    json.dumps(result, sort_keys=True)
    return result


def _blank_field() -> RoleChartField:
    return RoleChartField.from_state(
        {
            "charts": [],
            "classes": {},
            "merge_log": [],
            "role_class_merges": [],
            "records": [],
            "active_axes": None,
        }
    )


def _coerce_pair(value: WikiPair | Mapping[str, Any]) -> WikiPair:
    return value if isinstance(value, WikiPair) else WikiPair.from_dict(value)


def _scope_key(namespace: str, scope: str) -> str:
    return json.dumps([str(namespace), str(scope)], separators=(",", ":"), ensure_ascii=False)


class WikiJointField:
    """A joint reader plus a frozen, branch-bounded learned consumer.

    ``branch_cap`` counts successful native record-to-frame extensions across
    all selected programs.  The preflight follows deterministic program,
    frame, and record insertion order.  An extension beyond the cap causes an
    abstention before the learned bank executes.
    """

    def __init__(
        self,
        reader: WikiAxisField,
        bank: LearnedFabricBank,
        *,
        fact_view: str = "wiki-facts",
        branch_cap: int = 256,
        max_hops: int = 2,
        require_joint_projection: bool = True,
    ) -> None:
        if not isinstance(reader, WikiAxisField):
            raise TypeError("reader must be a WikiAxisField")
        if not isinstance(bank, LearnedFabricBank):
            raise TypeError("bank must be a LearnedFabricBank")
        if not fact_view:
            raise ValueError("fact_view must be non-empty")
        if isinstance(branch_cap, bool) or not isinstance(branch_cap, int) or branch_cap < 1:
            raise ValueError("branch_cap must be a positive integer")
        if isinstance(max_hops, bool) or not isinstance(max_hops, int) or max_hops < 1:
            raise ValueError("max_hops must be a positive integer")

        projection_violations = {
            view: sorted(_JOINT_AXES - set(reader.config.active_axes[view]))
            for view in reader.config.views
            if not _JOINT_AXES.issubset(reader.config.active_axes[view])
        }
        if require_joint_projection and projection_violations:
            raise ValueError(
                "joint Wikipedia views must project word_sequence, clause_scope, and polarity together: "
                f"{projection_violations}"
            )

        self.reader = reader
        self.bank = bank
        self.fabric = bank.fabric
        self.fact_view = str(fact_view)
        self.branch_cap = branch_cap
        self.max_hops = max_hops
        self.require_joint_projection = bool(require_joint_projection)
        self.projection_violations = projection_violations
        self.provenance: dict[str, dict[str, Any]] = {}
        self.population_history: list[dict[str, Any]] = []
        self.scope_decisions: dict[str, dict[str, Any]] = {}

        if self.fact_view not in self.fabric.views:
            self.fabric.register_view(self.fact_view, _blank_field())
        self._validate_fact_view()

    @property
    def model_sha256(self) -> str:
        return _digest(
            {
                "schema": STATE_SCHEMA,
                "reader": self.reader.model_sha256,
                "bank": self.bank.state(),
                "config": self._config_state(),
            }
        )

    def _config_state(self) -> dict[str, Any]:
        return {
            "fact_view": self.fact_view,
            "branch_cap": self.branch_cap,
            "branch_unit": "successful-native-frame-extensions",
            "max_hops": self.max_hops,
            "require_joint_projection": self.require_joint_projection,
            "ambiguity_policy": self.reader.config.ambiguity_policy,
        }

    def _validate_fact_view(self) -> None:
        field = self.fabric.view(self.fact_view)
        if any(
            int(chart["relation"]) == FACT_RELATION
            and len(chart.get("alignment", chart.get("classes", ()))) != len(FACT_SCHEMA["axes"])
            for chart in field.charts
        ):
            raise ValueError("existing fact view relation 0 has a conflicting chart arity")
        if any(record.relation != FACT_RELATION or len(record.roles) != 3 for record in field.records):
            raise ValueError("existing fact view does not implement (predicate, subject, object)")

    def predict(self, pair: WikiPair | Mapping[str, Any], *, detail: bool = True) -> dict[str, Any]:
        """Read one label-free pair and expose the exact joint projections."""
        result = self.reader.predict(pair, detail=detail)
        result["joint"] = {
            "required_axes": sorted(_JOINT_AXES),
            "active_axes": {
                view: list(self.reader.config.active_axes[view]) for view in self.reader.config.views
            },
            "same_projected_read": not self.projection_violations,
            "projection_violations": copy.deepcopy(self.projection_violations),
            "ambiguity_policy": self.reader.config.ambiguity_policy,
            "consumer_interpretations": "all eligible candidates",
            "claim_boundary": "candidate retrieval is not accuracy and no candidate is gold-selected",
        }
        return result

    def populate(
        self,
        pairs: Iterable[WikiPair | Mapping[str, Any]] = (),
        *,
        namespace: str,
        scope: str,
        detail: bool = True,
        view: str | None = None,
    ) -> dict[str, Any]:
        """Write eligible interpretations as provisional native fact records.

        The relation and direction come only from the frozen reader.  Labels on
        serving rows are rejected by ``WikiAxisField.predict``.  Every native
        record has a separate provisional observation carrying its source IDs;
        source IDs never participate in a projected fact lookup.
        """
        if view is not None and str(view) != self.fact_view:
            raise ValueError(f"this joint model writes only its captured fact view {self.fact_view!r}")
        namespace, scope = str(namespace), str(scope)
        field = self.fabric.view(self.fact_view)
        existing = {
            (record.namespace, record.scope, record.relation, record.roles, record.source)
            for record in field.records
        }
        results: list[dict[str, Any]] = []
        written = duplicates = eligible_total = truncated_pairs = 0

        for pair_value in pairs:
            pair = _coerce_pair(pair_value)
            if pair.labels:
                raise ValueError("populate accepts label-free WikiPair rows")
            raw_source = f"wiki-target:{_digest([pair.article_id, pair.pair_id])[:24]}"
            self.fabric.observe_raw(
                self.fact_view,
                pair.state(),
                namespace,
                scope,
                source=raw_source,
                provisional=True,
                status="reader-input",
            )
            prediction = self.predict(pair, detail=detail)
            candidates = [candidate for candidate in prediction["candidates"] if candidate.get("eligible", False)]
            eligible_total += len(candidates)
            reader_truncated = bool(prediction.get("trace", {}).get("decision", {}).get("truncated"))
            truncated_pairs += reader_truncated
            candidate_rows = []
            for ordinal, candidate in enumerate(candidates):
                roles = (str(candidate["relation"]), str(candidate["subject"]), str(candidate["object"]))
                source = f"wiki-joint:{_digest([self.reader.model_sha256, raw_source, ordinal, roles])[:24]}"
                source_state = {
                    "source": source,
                    "origin": "reader-candidate",
                    "raw_source": raw_source,
                    "reader_model_sha256": self.reader.model_sha256,
                    "reader_sources": sorted(str(value) for value in candidate.get("sources", ())),
                    "article_id": pair.article_id,
                    "pair_id": pair.pair_id,
                    "sentence_id": pair.sentence_id,
                    "direction": str(candidate["direction"]),
                }
                previous = self.provenance.get(source)
                if previous is not None and previous != source_state:
                    raise ValueError("provisional source identity collision")
                self.provenance[source] = source_state
                self.fabric.observe_raw(
                    self.fact_view,
                    {
                        "roles": list(roles),
                        "direction": str(candidate["direction"]),
                        "reader_sources": source_state["reader_sources"],
                        "reader_status": prediction["status"],
                    },
                    namespace,
                    scope,
                    source=source,
                    provisional=True,
                    status="reader-candidate",
                )
                identity = (namespace, scope, FACT_RELATION, roles, source)
                if identity in existing:
                    duplicates += 1
                else:
                    field.records.append(Record(namespace, scope, FACT_RELATION, roles, source, len(field.records)))
                    existing.add(identity)
                    written += 1
                candidate_rows.append({"source": source, "roles": list(roles), "direction": candidate["direction"]})
            results.append(
                {
                    "pair_id": pair.pair_id,
                    "article_id": pair.article_id,
                    **prediction,
                    "eligible_interpretations": candidate_rows,
                    "reader_decision": copy.deepcopy(prediction.get("trace", {}).get("decision", {})),
                }
            )

        if written:
            self.fabric.invalidate_view(self.fact_view)

        key = _scope_key(namespace, scope)
        previous_scope = self.scope_decisions.get(key, {})
        scope_state = {
            "namespace": namespace,
            "scope": scope,
            "population_calls": int(previous_scope.get("population_calls", 0)) + 1,
            "pairs": int(previous_scope.get("pairs", 0)) + len(results),
            "provisional_records_written": int(previous_scope.get("provisional_records_written", 0)) + written,
            "reader_candidate_truncations": int(previous_scope.get("reader_candidate_truncations", 0))
            + truncated_pairs,
            "complete": not (int(previous_scope.get("reader_candidate_truncations", 0)) + truncated_pairs),
        }
        self.scope_decisions[key] = scope_state
        report = {
            "status": "populated" if scope_state["complete"] else "incomplete",
            "pairs": len(results),
            "eligible_interpretations": eligible_total,
            "provisional_records_written": written,
            "duplicates": duplicates,
            "namespace": namespace,
            "scope": scope,
            "view": self.fact_view,
            "fact_schema": {"relation": FACT_RELATION, "roles": list(FACT_SCHEMA["axes"])},
            "ambiguity_policy": self.reader.config.ambiguity_policy,
            "interpretation_policy": "preserve-all-eligible",
            "reader_candidate_truncations": truncated_pairs,
            "results": results,
        }
        self.population_history.append(copy.deepcopy(report))
        return report

    @staticmethod
    def _reject_serving_labels(values: Mapping[str, Any] | None) -> None:
        if values is None:
            return
        found = sorted(_SERVING_LABEL_KEYS.intersection(str(key).lower() for key in values))
        if found:
            raise ValueError(f"serving inputs cannot contain labels or expected answers: {found}")

    def _preflight(
        self,
        programs: tuple[Any, ...],
        visible: Mapping[str, Any],
        *,
        namespace: str,
        scope: str,
        budget: int,
    ) -> dict[str, Any]:
        """Count the exact existing fabric extensions before bounded execution."""
        self.fabric._refresh_view_indices()
        initial = {str(key): str(value) for key, value in visible.items()}
        work = 0
        sources: set[str] = set()
        rows: list[dict[str, Any]] = []
        for program in programs:
            frames = [BindingFrame(dict(initial))]
            steps = program.steps[: min(budget, len(program.steps))]
            step_rows = []
            for cursor, step in enumerate(steps):
                next_frames: list[BindingFrame] = []
                branches = []
                field = self.fabric.view(step.view)
                for frame in frames:
                    query, missing = self.fabric._query_bindings(step, frame)
                    records = [] if missing else field.lookup(
                        step.relation_id, query, namespace=namespace, scope=scope
                    )
                    accepted_sources = []
                    for record in records:
                        candidate = self.fabric._extend_frame(frame, step, record, cursor, namespace, scope)
                        if candidate is None:
                            continue
                        work += 1
                        if work > self.branch_cap:
                            sources.add(record.source)
                            return {
                                "status": "branch-cap-exceeded",
                                "branch_cap": self.branch_cap,
                                "branch_unit": "successful-native-frame-extensions",
                                "observed_extensions": work,
                                "executed": False,
                                "programs": rows,
                                "sources": sorted(sources),
                                "overflow": {
                                    "program": program.name,
                                    "cursor": cursor,
                                    "source": record.source,
                                    "roles": list(record.roles),
                                    "projected_bindings": {
                                        str(axis): value for axis, value in query.items()
                                    },
                                },
                            }
                        next_frames.append(candidate)
                        sources.add(record.source)
                        accepted_sources.append(record.source)
                    branches.append(
                        {
                            "bindings": dict(frame.bindings),
                            "projected_bindings": {str(axis): value for axis, value in query.items()},
                            "missing_variables": missing,
                            "match_count": len(records),
                            "accepted_sources": accepted_sources,
                        }
                    )
                step_rows.append(
                    {
                        "cursor": cursor,
                        "view": step.view,
                        "relation": step.relation_id,
                        "input_branches": len(frames),
                        "output_branches": len(next_frames),
                        "branches": branches,
                    }
                )
                frames = next_frames
                if not frames:
                    break
            rows.append(
                {
                    "program": program.name,
                    "program_fingerprint": program.fingerprint,
                    "steps": step_rows,
                    "frontier_branches": len(frames),
                    "budget_exhausted": bool(frames) and len(steps) < len(program.steps),
                }
            )
        return {
            "status": "within-cap",
            "branch_cap": self.branch_cap,
            "branch_unit": "successful-native-frame-extensions",
            "observed_extensions": work,
            "executed": True,
            "execution_role": "dry-preflight-replayed-by-native-consumer",
            "programs": rows,
            "sources": sorted(sources),
        }

    def _source_provenance(self, sources: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {
            source: copy.deepcopy(
                self.provenance.get(source, {"source": source, "origin": "pre-existing-fabric-record"})
            )
            for source in sorted(set(sources))
        }

    def _abstention(
        self,
        reason: str,
        *,
        task: str,
        candidates: Iterable[Any] = (),
        native_status: str = "not-executed",
        native: Mapping[str, Any] | None = None,
        preflight: Mapping[str, Any] | None = None,
        scope_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        branch_state = {} if preflight is None else copy.deepcopy(preflight)
        return {
            "status": "abstained",
            "abstained": True,
            "reason": reason,
            "value": None,
            "candidates": _json_copy(list(candidates)),
            "task": task,
            "native_status": native_status,
            "native": None if native is None else copy.deepcopy(native),
            "decision": {
                "policy": "unanimous-complete-answer-branches",
                "agreement_domain": "surviving complete answer branches; dead ends do not vote",
                "outcome": "abstain",
                "reason": reason,
                "branch_cap": self.branch_cap,
                "branch_unit": "successful-native-frame-extensions",
                "ambiguity_policy": self.reader.config.ambiguity_policy,
            },
            "branch_preflight": branch_state,
            "source_provenance": self._source_provenance(branch_state.get("sources", ())),
            "scope_population": None if scope_state is None else copy.deepcopy(scope_state),
        }

    def solve(
        self,
        task: str,
        inputs: Mapping[str, Any],
        *,
        namespace: str,
        scope: str,
        budget: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return only answers shared by every surviving complete branch.

        The existing fabric prunes missing and contradictory frames while it
        executes the frozen program.  Those dead ends remain in the native
        trace; unanimity is deliberately limited to branches that reach a
        returned answer.
        """
        self._reject_serving_labels(inputs)
        self._reject_serving_labels(context)
        overlap = set(inputs) & set(context or {})
        if overlap:
            raise ValueError(f"input/context collision: {sorted(overlap)}")
        if budget is None:
            budget = self.max_hops
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
            raise ValueError("budget must be a non-negative integer or None")
        if budget > self.max_hops:
            raise ValueError(f"budget exceeds the frozen {self.max_hops}-hop consumer")

        task, namespace, scope = str(task), str(namespace), str(scope)
        scope_state = self.scope_decisions.get(_scope_key(namespace, scope))
        if scope_state is not None and not scope_state.get("complete", True):
            return self._abstention(
                "incomplete-reader-candidates",
                task=task,
                scope_state=scope_state,
            )

        programs = self.bank.programs_for(task)
        visible = {**inputs, **(context or {})}
        preflight = self._preflight(
            programs, visible, namespace=namespace, scope=scope, budget=budget
        )
        if preflight["status"] != "within-cap":
            return self._abstention(
                "branch-cap-exceeded",
                task=task,
                preflight=preflight,
                scope_state=scope_state,
            )

        native = self.bank.solve(
            task,
            inputs,
            namespace=namespace,
            scope=scope,
            budget=budget,
            context=context,
        )
        candidates = list(native.get("candidates", ()))
        native_status = str(native.get("status", "missing"))
        if native_status == "resolved" and native.get("value") is not None and len(candidates) == 1:
            return {
                "status": "resolved",
                "abstained": False,
                "reason": None,
                "value": copy.deepcopy(native["value"]),
                "candidates": _json_copy(candidates),
                "task": task,
                "native_status": native_status,
                "native": copy.deepcopy(native),
                "decision": {
                    "policy": "unanimous-complete-answer-branches",
                    "agreement_domain": "surviving complete answer branches; dead ends do not vote",
                    "outcome": "return",
                    "reason": "one-distinct-answer",
                    "answer_branches": len(candidates),
                    "branch_cap": self.branch_cap,
                    "branch_unit": "successful-native-frame-extensions",
                    "ambiguity_policy": self.reader.config.ambiguity_policy,
                },
                "branch_preflight": preflight,
                "source_provenance": self._source_provenance(preflight["sources"]),
                "scope_population": None if scope_state is None else copy.deepcopy(scope_state),
            }

        reasons = {
            "ambiguous": "disagreeing-answer-branches",
            "missing": "missing-answer-branches",
            "unknown_task": "unknown-task",
            "unlearned_task": "unlearned-task",
            "budget-exhausted": "budget-exhausted",
        }
        return self._abstention(
            reasons.get(native_status, f"native-{native_status}"),
            task=task,
            candidates=candidates,
            native_status=native_status,
            native=native,
            preflight=preflight,
            scope_state=scope_state,
        )

    # The dispatcher exposes this explicit name for callers that want to make
    # the retained-interpretation policy visible at the call site.  There is
    # one execution policy: both entry points preserve and adjudicate the same
    # native branches.
    solve_competing = solve

    def state(self) -> dict[str, Any]:
        state = {
            "version": STATE_VERSION,
            "schema": STATE_SCHEMA,
            "kind": "wiki_joint",
            "config": self._config_state(),
            "reader": self.reader.state(),
            "fabric": self.fabric.state(),
            "bank": self.bank.state(),
            "provenance": copy.deepcopy(self.provenance),
            "population_history": copy.deepcopy(self.population_history),
            "scope_decisions": copy.deepcopy(self.scope_decisions),
            "projection_violations": copy.deepcopy(self.projection_violations),
            "model_sha256": self.model_sha256,
        }
        return _json_copy(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> WikiJointField:
        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported WikiJointField state version")
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported WikiJointField schema")
        config = state.get("config", {})
        reader_state = state.get("reader", state.get("reader_state"))
        fabric_state = state.get("fabric", state.get("fabric_state"))
        bank_state = state.get("bank", state.get("bank_state"))
        if reader_state is None or fabric_state is None or bank_state is None:
            raise ValueError("WikiJointField state requires reader, fabric, and bank")
        reader = WikiAxisField.from_state(reader_state)
        fabric = FieldFabric.from_state(fabric_state)
        bank = LearnedFabricBank.from_state(bank_state, fabric)
        obj = cls(
            reader,
            bank,
            fact_view=str(config.get("fact_view", "wiki-facts")),
            branch_cap=int(config.get("branch_cap", 256)),
            max_hops=int(config.get("max_hops", 2)),
            require_joint_projection=bool(config.get("require_joint_projection", True)),
        )
        obj.provenance = copy.deepcopy(state.get("provenance", {}))
        obj.population_history = copy.deepcopy(state.get("population_history", []))
        obj.scope_decisions = copy.deepcopy(state.get("scope_decisions", {}))
        expected = state.get("model_sha256")
        if expected is not None and expected != obj.model_sha256:
            raise ValueError("WikiJointField model fingerprint mismatch")
        return obj


__all__ = ["WikiJointField"]
