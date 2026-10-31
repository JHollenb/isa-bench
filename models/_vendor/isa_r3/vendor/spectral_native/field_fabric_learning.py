"""Bounded source-supervised structural learning over :mod:`field_fabric`.

The learner searches a finite, supplied vocabulary of projected reads.  It can
discover which relation/role coordinates connect views, but relation meaning,
the operator set, task labels, and source endpoint feedback are supplied.  It
does not claim unrestricted language induction.  Fresh target answers never
enter ``fit`` or the serialized serving artifact.

Read outputs are local program-state writes (``OperatorVocabulary`` calls the
operation ``bind``); this module never mutates a ``RoleChartField`` while
searching.  Every candidate is executed by ``FieldFabric.run``, whose reads in
turn use the native ``RoleChartField.lookup`` consumer.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from .field_fabric import FabricProgram, FieldFabric, ReadStep, RelationRef


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@dataclass(frozen=True)
class SourceExample:
    """One explicitly supervised source interaction.

    ``context`` is model-visible program input, not hidden selection metadata.
    Callers must supply it again at serving time when a learned program uses it.
    The split guard makes accidental target/audit selection fail closed.
    """

    example_id: str
    task: str
    inputs: Mapping[str, Any]
    expected: Any
    context: Mapping[str, Any] = field(default_factory=dict)
    namespace: str = "source"
    scope: str = "train"
    split: str = "source"

    def merged_inputs(self) -> dict[str, Any]:
        overlap = set(self.inputs) & set(self.context)
        if overlap:
            raise ValueError(f"input/context collision: {sorted(overlap)}")
        return {**self.inputs, **self.context}


@dataclass(frozen=True)
class AxisLiteral:
    """A supplied literal coordinate; literals are never mined from answers."""

    view: str
    relation: int
    axis: int
    value: Any


@dataclass(frozen=True)
class OperatorVocabulary:
    """Finite hypothesis geometry for structural search.

    ``read`` is a projected relation read, ``bind`` writes read outputs into
    local bindings, and ``return`` exposes a binding.  No persistent write is
    admitted by this first fabric contract.
    """

    relations: tuple[RelationRef, ...] | None = None
    literals: tuple[AxisLiteral, ...] = ()
    operators: tuple[str, ...] = ("read", "bind", "return")
    max_steps: int = 3
    max_bindings_per_read: int = 2
    max_outputs_per_read: int = 1
    beam_width: int = 256
    max_candidate_work: int = 4096
    max_candidates: int = 256

    def __post_init__(self) -> None:
        if set(self.operators) != {"read", "bind", "return"}:
            raise ValueError("this learner requires exactly the read/bind/return operator vocabulary")
        for name in ("max_steps", "max_bindings_per_read", "max_outputs_per_read",
                     "beam_width", "max_candidate_work", "max_candidates"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_steps > 3:
            raise ValueError("bounded fabric search supports path lengths 1..3")


@dataclass(frozen=True)
class CandidateFeatures:
    support_rate: float
    compatible_rate: float
    coverage_rate: float
    simplicity: float
    future_score: float

    def state(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class TinyRouteScorer:
    """A five-feature linear route scorer, intentionally not a language model.

    All features also enter the reported structural score and are stored in the
    serving artifact.  Endpoint labels are not features at inference time.
    """

    weights: Mapping[str, float] = field(default_factory=dict)
    bias: float = 0.0

    AXES = ("support_rate", "compatible_rate", "coverage_rate", "simplicity", "future_score")

    def score(self, features: CandidateFeatures) -> float:
        return float(self.bias + sum(float(self.weights.get(axis, 0.0)) * getattr(features, axis)
                                     for axis in self.AXES))

    def state(self) -> dict[str, Any]:
        return {"kind": "linear", "weights": {axis: float(self.weights.get(axis, 0.0)) for axis in self.AXES},
                "bias": float(self.bias)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> TinyRouteScorer:
        return cls(weights=dict(state.get("weights", {})), bias=float(state.get("bias", 0.0)))


@dataclass(frozen=True)
class TinyNeuralRouteScorer:
    """Fixed width-four tanh comparator over the same transparent features.

    This is a graded route tie-breaker, not a dense language model.  Its full
    parameter state is JSON-visible and small enough for same-parent Saturn
    proposals to copy and replay exactly.
    """

    input_weights: tuple[tuple[float, ...], ...] = (
        (1.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0, 1.0),
    )
    hidden_bias: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    output_weights: tuple[float, ...] = (1.0, 0.5, 0.25, 0.25)
    output_bias: float = 0.0

    AXES = TinyRouteScorer.AXES
    WIDTH = 4

    def __post_init__(self) -> None:
        if len(self.input_weights) != self.WIDTH or any(len(row) != len(self.AXES)
                                                        for row in self.input_weights):
            raise ValueError("neural route input weights must have shape 4x5")
        if len(self.hidden_bias) != self.WIDTH or len(self.output_weights) != self.WIDTH:
            raise ValueError("neural route hidden/output vectors must have width 4")
        values = (*itertools.chain.from_iterable(self.input_weights), *self.hidden_bias,
                  *self.output_weights, self.output_bias)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("neural route parameters must be finite")

    def score(self, features: CandidateFeatures) -> float:
        inputs = tuple(getattr(features, axis) for axis in self.AXES)
        hidden = tuple(math.tanh(sum(weight * value for weight, value in zip(row, inputs, strict=True)) + bias)
                       for row, bias in zip(self.input_weights, self.hidden_bias, strict=True))
        return float(sum(weight * value for weight, value in zip(self.output_weights, hidden, strict=True))
                     + self.output_bias)

    def state(self) -> dict[str, Any]:
        return {"kind": "neural-width4-tanh", "input_weights": [list(row) for row in self.input_weights],
                "hidden_bias": list(self.hidden_bias), "output_weights": list(self.output_weights),
                "output_bias": float(self.output_bias), "feature_axes": list(self.AXES)}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> TinyNeuralRouteScorer:
        return cls(tuple(tuple(float(value) for value in row) for row in state["input_weights"]),
                   tuple(float(value) for value in state["hidden_bias"]),
                   tuple(float(value) for value in state["output_weights"]), float(state["output_bias"]))


RouteScorer = TinyRouteScorer | TinyNeuralRouteScorer


@dataclass(frozen=True)
class RouteUpdate:
    """One reversible, caller-evaluated scorer future."""

    axis: str
    dose: float
    scorer: TinyRouteScorer

    def state(self) -> dict[str, Any]:
        return {"axis": self.axis, "dose": self.dose, "scorer": self.scorer.state()}


def propose_route_updates(
    scorer: TinyRouteScorer,
    *,
    axes: Sequence[str] = TinyRouteScorer.AXES,
    doses: Sequence[float] = (-0.1, 0.1),
    max_updates: int = 10,
) -> list[RouteUpdate]:
    """Return a bounded panel for a Saturn boundary/controller to replay.

    The function proposes states only.  Promotion, rollback, and immutable
    receipts remain the responsibility of the Saturn controller used by the
    experiment worker; there is no epoch loop here.
    """

    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    updates = []
    for axis, dose in itertools.product(axes, doses):
        if axis not in TinyRouteScorer.AXES or not math.isfinite(float(dose)):
            raise ValueError("invalid route update")
        weights = dict(scorer.weights)
        weights[axis] = float(weights.get(axis, 0.0)) + float(dose)
        updates.append(RouteUpdate(axis, float(dose), TinyRouteScorer(weights, scorer.bias)))
        if len(updates) == max_updates:
            break
    return updates


@dataclass(frozen=True)
class NeuralWeightUpdate:
    """One immutable neural comparator future for bounded replay."""

    parameter: str
    index: tuple[int, ...]
    dose: float
    scorer: TinyNeuralRouteScorer

    def state(self) -> dict[str, Any]:
        return {"parameter": self.parameter, "index": list(self.index), "dose": self.dose,
                "scorer": self.scorer.state()}


def propose_neural_route_updates(
    scorer: TinyNeuralRouteScorer,
    *,
    coordinates: Sequence[tuple[str, tuple[int, ...]]] | None = None,
    doses: Sequence[float] = (-0.05, 0.05),
    max_updates: int = 8,
) -> list[NeuralWeightUpdate]:
    """Propose finite reversible parameter changes without fitting an epoch."""

    if max_updates < 1:
        raise ValueError("max_updates must be positive")
    coordinates = coordinates or tuple(("output_weights", (index,)) for index in range(scorer.WIDTH))
    updates = []
    for (parameter, index), dose in itertools.product(coordinates, doses):
        if not math.isfinite(float(dose)) or parameter not in {
            "input_weights", "hidden_bias", "output_weights", "output_bias"
        }:
            raise ValueError("invalid neural route update")
        inputs = [list(row) for row in scorer.input_weights]
        hidden = list(scorer.hidden_bias)
        outputs = list(scorer.output_weights)
        output_bias = scorer.output_bias
        if any(position < 0 for position in index):
            raise ValueError("neural route update index out of range")
        try:
            if parameter == "input_weights" and len(index) == 2:
                inputs[index[0]][index[1]] += float(dose)
            elif parameter == "hidden_bias" and len(index) == 1:
                hidden[index[0]] += float(dose)
            elif parameter == "output_weights" and len(index) == 1:
                outputs[index[0]] += float(dose)
            elif parameter == "output_bias" and not index:
                output_bias += float(dose)
            else:
                raise ValueError("parameter/index shape mismatch")
        except IndexError as exc:
            raise ValueError("neural route update index out of range") from exc
        future = TinyNeuralRouteScorer(tuple(tuple(row) for row in inputs), tuple(hidden),
                                       tuple(outputs), output_bias)
        updates.append(NeuralWeightUpdate(parameter, tuple(index), float(dose), future))
        if len(updates) == max_updates:
            break
    return updates


def _route_scorer_from_state(state: Mapping[str, Any]) -> RouteScorer:
    if state.get("kind") == "neural-width4-tanh":
        return TinyNeuralRouteScorer.from_state(state)
    return TinyRouteScorer.from_state(state)


def structural_score(features: CandidateFeatures) -> float:
    """Transparent comparator using exactly the route scorer's information."""

    return (4.0 * features.support_rate + features.compatible_rate + features.coverage_rate
            + features.simplicity + features.future_score)


def _program_state(program: FabricProgram) -> dict[str, Any]:
    return {
        "name": program.name,
        "steps": [{"view": step.view,
                   "relation": int(step.relation.relation if isinstance(step.relation, RelationRef)
                                   else step.relation),
                   "bindings": {str(axis): value for axis, value in step.bindings.items()},
                   "outputs": {str(axis): value for axis, value in step.outputs.items()}}
                  for step in program.steps],
        "returns": list(program.returns),
    }


def _program_from_state(state: Mapping[str, Any]) -> FabricProgram:
    return FabricProgram(
        str(state["name"]),
        tuple(ReadStep(str(row["view"]), int(row["relation"]),
                       {int(axis): value for axis, value in row["bindings"].items()},
                       {int(axis): value for axis, value in row["outputs"].items()})
              for row in state["steps"]),
        tuple(state["returns"]),
    )


def alpha_normalize(program: FabricProgram) -> dict[str, Any]:
    """Canonicalize input and local variable names while preserving structure."""

    names: dict[str, str] = {}
    next_input = next_local = 0

    def rename(name: str, *, output: bool = False) -> str:
        nonlocal next_input, next_local
        bare = name[1:] if name.startswith("$") else name
        if bare not in names:
            if output:
                names[bare] = f"v{next_local}"
                next_local += 1
            else:
                names[bare] = f"i{next_input}"
                next_input += 1
        return ("$" if name.startswith("$") else "") + names[bare]

    steps = []
    for step in program.steps:
        bindings = []
        for axis, value in sorted(step.bindings.items()):
            bindings.append([int(axis), rename(value) if isinstance(value, str) and value.startswith("$") else value])
        outputs = [[int(axis), rename(value, output=True)] for axis, value in sorted(step.outputs.items())]
        relation = int(step.relation.relation if isinstance(step.relation, RelationRef) else step.relation)
        steps.append({"view": step.view, "relation": relation, "bindings": bindings, "outputs": outputs})
    return {"steps": steps, "returns": [rename(value) for value in program.returns]}


def _concrete_signature(program: FabricProgram) -> str:
    """A name-free key that keeps task input coordinates distinct."""

    state = _program_state(program)
    state.pop("name")
    return _json_key(state)


def _description_length(program: FabricProgram) -> int:
    relations = {(step.view, int(step.relation.relation if isinstance(step.relation, RelationRef)
                                 else step.relation)) for step in program.steps}
    literals = sum(not (isinstance(value, str) and value.startswith("$"))
                   for step in program.steps for value in step.bindings.values())
    return (8 * len(program.steps) + 2 * sum(len(step.bindings) + len(step.outputs) for step in program.steps)
            + 2 * len(relations) + 3 * literals + len(program.returns))


def _result_values(result: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = []
    status = result.get("status")
    if status == "resolved":
        if "value" in result:
            values.append(result["value"])
        elif "bindings" in result:
            binding = result["bindings"]
            value = next(iter(binding.values())) if isinstance(binding, Mapping) and len(binding) == 1 else binding
            values.append(value)
    # Terminal ambiguity candidates contain only requested returns.  A
    # budget-exhausted frontier instead contains every partial/input binding;
    # those are not endpoint alternatives and must never look resolved.
    if status == "ambiguous":
        for candidate in result.get("candidates", []) or []:
            if isinstance(candidate, Mapping) and "value" in candidate:
                values.append(candidate["value"])
            elif isinstance(candidate, Mapping) and "bindings" in candidate:
                binding = candidate["bindings"]
                values.append(next(iter(binding.values())) if len(binding) == 1 else binding)
            else:
                values.append(candidate)
    unique = {_json_key(value): value for value in values}
    return [unique[key] for key in sorted(unique)]


def evaluate_candidate(
    fabric: FieldFabric,
    program: FabricProgram,
    examples: Sequence[SourceExample],
    *,
    budget: int | None = None,
) -> dict[str, Any]:
    """Evaluate one program on source feedback without retaining answer values."""

    exact = compatible = coverage = 0
    status_counts: dict[str, int] = {}
    observations = []
    for example in examples:
        result = fabric.run(program, example.merged_inputs(), namespace=example.namespace,
                            scope=example.scope, budget=budget)
        values = _result_values(result)
        status = str(result.get("status", "invalid"))
        status_counts[status] = status_counts.get(status, 0) + 1
        coverage += bool(values)
        is_compatible = any(_json_key(value) == _json_key(example.expected) for value in values)
        compatible += is_compatible
        is_exact = status == "resolved" and len(values) == 1 and is_compatible
        exact += is_exact
        observations.append({"example_id": example.example_id, "status": status,
                             "covered": bool(values), "compatible": is_compatible, "exact": is_exact})
    return {"exact_support": exact, "compatible_support": compatible, "coverage": coverage,
            "source_total": len(examples), "status_counts": status_counts,
            "observations": observations}


@dataclass
class LearnedCandidate:
    task: str
    program: FabricProgram
    exact_support: int
    compatible_support: int
    coverage: int
    source_total: int
    description_length: int
    future_score: float = 0.0
    future_metrics: dict[str, float] = field(default_factory=dict)
    route_score: float = 0.0
    selected: bool = False

    @property
    def features(self) -> CandidateFeatures:
        total = max(1, self.source_total)
        return CandidateFeatures(self.exact_support / total, self.compatible_support / total,
                                 self.coverage / total, 1.0 / (1.0 + self.description_length),
                                 self.future_score)

    @property
    def selection_key(self) -> tuple[float, ...]:
        return (float(self.exact_support), float(self.compatible_support), float(self.coverage),
                float(self.future_score), -float(self.description_length), float(self.route_score))

    def state(self) -> dict[str, Any]:
        return {"task": self.task, "program": _program_state(self.program),
                "alpha_normalized": alpha_normalize(self.program),
                "exact_support": self.exact_support, "compatible_support": self.compatible_support,
                "coverage": self.coverage, "source_total": self.source_total,
                "description_length": self.description_length, "future_score": self.future_score,
                "future_metrics": dict(self.future_metrics), "features": self.features.state(),
                "structural_score": structural_score(self.features), "route_score": self.route_score,
                "selected": self.selected}

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LearnedCandidate:
        return cls(task=str(state["task"]), program=_program_from_state(state["program"]),
                   exact_support=int(state["exact_support"]),
                   compatible_support=int(state["compatible_support"]), coverage=int(state["coverage"]),
                   source_total=int(state["source_total"]), description_length=int(state["description_length"]),
                   future_score=float(state.get("future_score", 0.0)),
                   future_metrics={key: float(value) for key, value in state.get("future_metrics", {}).items()},
                   route_score=float(state.get("route_score", 0.0)), selected=bool(state.get("selected", False)))


def rank_candidates(candidates: Iterable[LearnedCandidate]) -> list[LearnedCandidate]:
    """Rank without hiding any score component or counter-trend."""

    return sorted(candidates, key=lambda candidate: (
        tuple(-value for value in candidate.selection_key), _json_key(alpha_normalize(candidate.program))))


@dataclass(frozen=True)
class _Draft:
    steps: tuple[ReadStep, ...]
    available: tuple[str, ...]
    recent: tuple[str, ...]


def _relation_signatures(fabric: FieldFabric, vocabulary: OperatorVocabulary) -> list[tuple[RelationRef, int]]:
    allowed = None if vocabulary.relations is None else {(ref.view, ref.relation) for ref in vocabulary.relations}
    signatures: dict[tuple[str, int], int] = {}
    for view_name, view in sorted(fabric.views.items()):
        for chart in getattr(view, "charts", ()):  # structural schema only; never record values
            relation = int(chart["relation"])
            key = (view_name, relation)
            if allowed is None or key in allowed:
                signatures[key] = max(signatures.get(key, 0), len(chart.get("alignment", chart.get("classes", ()))))
        for record in getattr(view, "records", ()):
            key = (view_name, int(record.relation))
            if allowed is None or key in allowed:
                signatures[key] = max(signatures.get(key, 0), len(record.roles))
    if allowed is not None:
        missing = allowed - set(signatures)
        if missing:
            raise ValueError(f"vocabulary relations absent from fabric signatures: {sorted(missing)}")
    return [(RelationRef(view, relation), arity) for (view, relation), arity in sorted(signatures.items()) if arity > 1]


def _literal_variants(
    ref: RelationRef,
    base: dict[int, Any],
    vocabulary: OperatorVocabulary,
) -> list[dict[int, Any]]:
    choices: dict[int, list[Any]] = {}
    for literal in vocabulary.literals:
        if (literal.view, literal.relation) == (ref.view, ref.relation) and literal.axis not in base:
            choices.setdefault(literal.axis, []).append(literal.value)
    variants = [dict(base)]
    capacity = vocabulary.max_bindings_per_read - len(base)
    axes = sorted(choices)
    for count in range(1, min(capacity, len(axes)) + 1):
        for selected_axes in itertools.combinations(axes, count):
            for values in itertools.product(*(choices[axis] for axis in selected_axes)):
                variants.append({**base, **dict(zip(selected_axes, values, strict=True))})
    return variants


def _expand(
    draft: _Draft,
    signatures: Sequence[tuple[RelationRef, int]],
    vocabulary: OperatorVocabulary,
) -> Iterable[_Draft]:
    step_index = len(draft.steps)
    for ref, arity in signatures:
        max_bound = min(vocabulary.max_bindings_per_read, arity - 1, len(draft.available))
        for count in range(1, max_bound + 1):
            for variables in itertools.permutations(draft.available, count):
                if draft.recent and not set(variables) & set(draft.recent):
                    continue  # programs are connected; later reads consume a recent write
                for axes in itertools.permutations(range(arity), count):
                    base = {axis: f"${variable}" for axis, variable in zip(axes, variables, strict=True)}
                    for bindings in _literal_variants(ref, base, vocabulary):
                        free_axes = [axis for axis in range(arity) if axis not in bindings]
                        max_outputs = min(vocabulary.max_outputs_per_read, len(free_axes))
                        for output_count in range(1, max_outputs + 1):
                            for output_axes in itertools.combinations(free_axes, output_count):
                                outputs = {axis: f"_v{step_index}_{ordinal}"
                                           for ordinal, axis in enumerate(output_axes)}
                                step = ReadStep(ref.view, ref.relation, bindings, outputs)
                                names = tuple(outputs.values())
                                yield _Draft((*draft.steps, step), (*draft.available, *names), names)


def _name(task: str, signature: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", task).strip("-") or "task"
    return f"learned.{slug}.{hashlib.sha256(signature.encode()).hexdigest()[:12]}"


class LearnedFabricBank:
    """Frozen task-to-program choices plus every retained rejected future."""

    def __init__(self, fabric: FieldFabric, candidates: Sequence[LearnedCandidate], *,
                 search: Mapping[str, Any], route_scorer: RouteScorer | None = None) -> None:
        self.fabric = fabric
        self.candidates = list(candidates)
        self.search = dict(search)
        self.route_scorer = route_scorer
        self._by_task: dict[str, list[LearnedCandidate]] = {}
        for candidate in self.candidates:
            self._by_task.setdefault(candidate.task, []).append(candidate)
            if candidate.selected:
                self.fabric.register_program(candidate.program)
        self.report = self._report()

    def programs_for(self, task: str) -> tuple[FabricProgram, ...]:
        return tuple(candidate.program for candidate in self._by_task.get(task, ()) if candidate.selected)

    def solve(self, task: str, inputs: Mapping[str, Any], *, namespace: str, scope: str,
              budget: int | None = None, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        programs = self.programs_for(task)
        if not programs:
            return {"status": "unknown_task" if task not in self._by_task else "unlearned_task",
                    "value": None, "candidates": [], "task": task, "trace": []}
        overlap = set(inputs) & set(context or {})
        if overlap:
            raise ValueError(f"input/context collision: {sorted(overlap)}")
        visible = {**inputs, **(context or {})}
        rows, values, statuses = [], {}, []
        for program in programs:
            result = self.fabric.run(program, visible, namespace=namespace, scope=scope, budget=budget)
            rows.append({"program": program.name, "result": result})
            statuses.append(result.get("status"))
            for value in _result_values(result):
                values[_json_key(value)] = value
        ordered = [values[key] for key in sorted(values)]
        if len(ordered) == 1:
            status = "resolved"
        elif ordered:
            status = "ambiguous"
        elif "budget-exhausted" in statuses:
            status = "budget-exhausted"
        else:
            status = "missing"
        return {"status": status, "value": ordered[0] if len(ordered) == 1 else None,
                "candidates": ordered, "task": task, "programs": [program.name for program in programs],
                "trace": rows}

    predict = solve

    def _report(self) -> dict[str, Any]:
        selected = [candidate for candidate in self.candidates if candidate.selected]
        families: dict[str, dict[str, Any]] = {}
        for candidate in selected:
            signature = _json_key(alpha_normalize(candidate.program))
            family = families.setdefault(signature, {"signature": json.loads(signature), "tasks": set(),
                                                       "programs": []})
            family["tasks"].add(candidate.task)
            family["programs"].append(candidate.program.name)
        reusable = []
        for signature, family in sorted(families.items()):
            if len(family["tasks"]) >= 2:
                reusable.append({"family_id": f"family-{hashlib.sha256(signature.encode()).hexdigest()[:12]}",
                                 "signature": family["signature"], "tasks": sorted(family["tasks"]),
                                 "programs": sorted(family["programs"]),
                                 "justification": "same alpha-normalized executable structure across tasks"})
        alignments = []
        for candidate in selected:
            for left, right in itertools.pairwise(candidate.program.steps):
                shared = sorted(set(left.outputs.values()) & {value[1:] for value in right.bindings.values()
                                                               if isinstance(value, str) and value.startswith("$")})
                for variable in shared:
                    alignments.append({"task": candidate.task, "producer_view": left.view,
                                       "producer_relation": int(left.relation),
                                       "producer_axis": next(axis for axis, name in left.outputs.items()
                                                             if name == variable),
                                       "consumer_view": right.view, "consumer_relation": int(right.relation),
                                       "consumer_axis": next(axis for axis, value in right.bindings.items()
                                                             if value == f"${variable}"),
                                       "source_support": candidate.exact_support})
        return {"claim_boundary": "finite supplied relations/operators and explicit source endpoint supervision; "
                                   "no unrestricted language or unseen-task claim",
                "selection_split": "source", "serving_expected_answers": False,
                "selection": "source exact/compatible support, native future score, description length; retain ties",
                "search": dict(self.search), "tasks": {task: {
                    "status": "learned" if self.programs_for(task) else "unlearned",
                    "selected_programs": [program.name for program in self.programs_for(task)],
                    "candidate_count": len(rows)} for task, rows in sorted(self._by_task.items())},
                "candidates": [candidate.state() for candidate in self.candidates],
                "families": reusable, "learned_alignments": alignments,
                "route_scorer": None if self.route_scorer is None else self.route_scorer.state()}

    def state(self) -> dict[str, Any]:
        # Deliberately excludes SourceExample objects, expected values, outputs,
        # and per-example match labels. Aggregate source support is retained.
        state = {"version": 1, "candidates": [candidate.state() for candidate in self.candidates],
                 "search": dict(self.search),
                 "route_scorer": None if self.route_scorer is None else self.route_scorer.state()}
        return json.loads(json.dumps(state, sort_keys=True))

    @classmethod
    def from_state(cls, state: Mapping[str, Any], fabric: FieldFabric) -> LearnedFabricBank:
        if int(state.get("version", 0)) != 1:
            raise ValueError("unsupported learned fabric state")
        scorer_state = state.get("route_scorer")
        return cls(fabric, [LearnedCandidate.from_state(row) for row in state["candidates"]],
                   search=state.get("search", {}),
                   route_scorer=None if scorer_state is None else _route_scorer_from_state(scorer_state))


class BoundedFabricLearner:
    """Finite relation-path search with source-only continuation selection."""

    def __init__(self, fabric: FieldFabric, vocabulary: OperatorVocabulary | None = None) -> None:
        self.fabric = fabric
        self.vocabulary = vocabulary or OperatorVocabulary()
        self.last_evidence: dict[str, Any] = {}

    def fit(
        self,
        source_examples: Sequence[SourceExample],
        *,
        future_scorer: Callable[[FabricProgram, Sequence[SourceExample]], float | Mapping[str, float]] | None = None,
        route_scorer: RouteScorer | None = None,
    ) -> LearnedFabricBank:
        if not source_examples:
            raise ValueError("source_examples must not be empty")
        if any(example.split != "source" for example in source_examples):
            raise ValueError("fit accepts source examples only; target/audit labels cannot select programs")
        # Candidate execution materializations contain source outputs.  Search
        # on an exact fabric clone so those caches and frontiers cannot become
        # serving-time state in the accepted bank.
        search_fabric = FieldFabric.from_state(self.fabric.state())
        grouped: dict[str, list[SourceExample]] = {}
        seen_ids = set()
        for example in source_examples:
            if example.example_id in seen_ids:
                raise ValueError(f"duplicate source example id: {example.example_id}")
            seen_ids.add(example.example_id)
            grouped.setdefault(example.task, []).append(example)

        signatures = _relation_signatures(self.fabric, self.vocabulary)
        if not signatures:
            raise ValueError("fabric/vocabulary has no arbitrary-role relation signatures")
        retained: list[LearnedCandidate] = []
        evidence_tasks: dict[str, Any] = {}
        total_work = 0
        for task, examples in sorted(grouped.items()):
            input_names = set(examples[0].merged_inputs())
            for example in examples[1:]:
                input_names &= set(example.merged_inputs())
            if not input_names:
                raise ValueError(f"task {task!r} has no shared source input/context coordinates")
            drafts = [_Draft((), tuple(sorted(input_names)), ())]
            rows_by_signature: dict[str, tuple[LearnedCandidate, dict[str, Any]]] = {}
            task_work = 0
            stopped = False
            for _depth in range(1, self.vocabulary.max_steps + 1):
                next_rows: list[tuple[_Draft, LearnedCandidate, dict[str, Any]]] = []
                for draft in drafts:
                    for expanded in _expand(draft, signatures, self.vocabulary):
                        signature = _concrete_signature(FabricProgram("candidate", expanded.steps,
                                                                       expanded.recent))
                        if signature in rows_by_signature:
                            continue
                        program = FabricProgram(_name(task, signature), expanded.steps, expanded.recent)
                        evaluation = evaluate_candidate(search_fabric, program, examples,
                                                        budget=len(expanded.steps))
                        candidate = LearnedCandidate(task, program, evaluation["exact_support"],
                                                     evaluation["compatible_support"], evaluation["coverage"],
                                                     evaluation["source_total"], _description_length(program))
                        rows_by_signature[signature] = (candidate, evaluation)
                        next_rows.append((expanded, candidate, evaluation))
                        task_work += 1
                        total_work += 1
                        if task_work >= self.vocabulary.max_candidate_work:
                            stopped = True
                            break
                    if stopped:
                        break
                ranked_partial = sorted(next_rows, key=lambda row: (
                    -row[1].exact_support, -row[1].compatible_support, -row[1].coverage,
                    row[1].description_length, _json_key(alpha_normalize(row[1].program))))
                drafts = [row[0] for row in ranked_partial[:self.vocabulary.beam_width]]
                if stopped or not drafts:
                    break

            structural = rank_candidates(candidate for candidate, _ in rows_by_signature.values()
                                         if candidate.exact_support > 0)
            if not structural:
                structural = rank_candidates(candidate for candidate, _ in rows_by_signature.values())[:1]
            # Score a bounded panel. Include all ties at the retention boundary.
            panel = structural[:self.vocabulary.max_candidates]
            if panel:
                boundary_key = panel[-1].selection_key
                panel.extend(candidate for candidate in structural[self.vocabulary.max_candidates:]
                             if candidate.selection_key == boundary_key)
            scored = []
            for candidate in panel:
                if future_scorer is not None:
                    future = future_scorer(candidate.program, tuple(examples))
                    if isinstance(future, Mapping):
                        metrics = {key: float(value) for key, value in future.items()}
                        score = float(metrics.get("score", next(iter(metrics.values()), 0.0)))
                    else:
                        score, metrics = float(future), {"score": float(future)}
                    if not math.isfinite(score) or any(not math.isfinite(value) for value in metrics.values()):
                        raise ValueError("future scorer returned a non-finite value")
                    candidate.future_score = score
                    candidate.future_metrics = metrics
                if route_scorer is not None:
                    candidate.route_score = route_scorer.score(candidate.features)
                scored.append(candidate)
            ranked = rank_candidates(scored)
            if ranked and ranked[0].exact_support > 0:
                best = ranked[0].selection_key
                ranked = [replace(candidate, selected=candidate.selection_key == best) for candidate in ranked]
            retained.extend(ranked)
            evidence_tasks[task] = {"source_examples": len(examples), "candidate_work": task_work,
                                    "work_cap_reached": stopped, "retained_futures": len(ranked),
                                    "all_rejected_futures_retained": True}

        search = {"operators": list(self.vocabulary.operators), "max_steps": self.vocabulary.max_steps,
                  "beam_width": self.vocabulary.beam_width,
                  "max_candidate_work_per_task": self.vocabulary.max_candidate_work,
                  "max_candidates_per_task": self.vocabulary.max_candidates,
                  "candidate_work": total_work, "relation_signatures": len(signatures),
                  "literal_coordinates_supplied": len(self.vocabulary.literals),
                  "source_only": True, "target_selection_calls": 0, "search_isolated_copy": True}
        self.last_evidence = {"tasks": evidence_tasks, "search": search,
                              "note": "per-source expected values and outputs intentionally excluded"}
        return LearnedFabricBank(self.fabric, retained, search=search, route_scorer=route_scorer)


def learn_programs(
    fabric: FieldFabric,
    source_examples: Sequence[SourceExample],
    *,
    vocabulary: OperatorVocabulary | None = None,
    future_scorer: Callable[[FabricProgram, Sequence[SourceExample]], float | Mapping[str, float]] | None = None,
    route_scorer: RouteScorer | None = None,
) -> LearnedFabricBank:
    """Convenience entry point for a single bounded structural fit."""

    return BoundedFabricLearner(fabric, vocabulary).fit(
        source_examples, future_scorer=future_scorer, route_scorer=route_scorer)


__all__ = ["AxisLiteral", "BoundedFabricLearner", "CandidateFeatures", "LearnedCandidate",
           "LearnedFabricBank", "NeuralWeightUpdate", "OperatorVocabulary", "RouteUpdate",
           "SourceExample", "TinyNeuralRouteScorer", "TinyRouteScorer", "alpha_normalize",
           "evaluate_candidate", "learn_programs", "propose_neural_route_updates",
           "propose_route_updates", "rank_candidates", "structural_score"]
