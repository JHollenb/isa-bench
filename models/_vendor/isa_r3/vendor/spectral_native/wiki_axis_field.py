"""Wikipedia relation reading as projected joins over native role charts.

The reader stores each source text feature as an arbitrary-arity
``RoleChartField`` record.  Inference extracts the same supplied text views and
queries only the configured coordinate subset; every omitted coordinate is a
real chart wildcard.  Entity handles are carried to the final fact consumer,
but never enter a feature address.

Labels are weak source alignment evidence, not negative supervision.  Missing
labels therefore add no field record.  Competing source-supported labels remain
separate candidates, and the configured ambiguity policy is reported in the
trace before any native fact is written.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import Any

from .language_attachment import local_keys
from .ordinary_language_structure import features, wilson
from .pattern_identity import tokens
from .role_chart_field import Record, RoleChartField

STATE_VERSION = 1
SOURCE_NAMESPACE = "wiki-axis-source"
SOURCE_SCOPE = "induction"
FEATURE_RELATION = 0
FACT_RELATION = 0
FEATURE_AXES = (
    "word_sequence",
    "side",
    "distance",
    "left_kind",
    "right_kind",
    "clause_scope",
    "polarity",
)
OUTPUT_AXES = ("relation", "direction")
ROLE_AXES = (*FEATURE_AXES, *OUTPUT_AXES)
AXIS_INDEX = {name: index for index, name in enumerate(ROLE_AXES)}
VIEW_SCHEMAS = {
    name: {"relation": FEATURE_RELATION, "axes": ROLE_AXES}
    for name in ("between", "located", "scope")
}
FACT_SCHEMA = {"relation": FACT_RELATION, "axes": ("predicate", "subject", "object")}

_MARKERS = {"<L>", "<R>", "<e>"}
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = set(_OPEN.values())
_CLAUSE_BREAK = {",", ";", ":", ".", "!", "?"}
_NEGATION = {"not", "never", "no", "n't"}
_VALID_POLICIES = {"preserve", "unique-best", "margin"}


def _json_copy(value: Any) -> Any:
    result = copy.deepcopy(value)
    json.dumps(result, sort_keys=True)
    return result


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


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


def _word_sequence(words: Iterable[str]) -> str:
    return json.dumps(list(words), ensure_ascii=False, separators=(",", ":"))


def _distance_bucket(distance: int) -> str:
    return str(min(max(int(distance), 0), 4))


def _source_id(pair: WikiPair) -> str:
    return f"wiki:{_digest([pair.article_id, pair.pair_id])[:24]}"


def _label_parts(label: str) -> tuple[str, str]:
    try:
        relation, direction = label.rsplit(":", 1)
    except ValueError as exc:
        raise ValueError(f"invalid source label {label!r}") from exc
    if not relation.startswith("P") or not relation[1:].isdigit() or direction not in {"f", "r"}:
        raise ValueError(f"invalid source label {label!r}")
    return relation, direction


@dataclass(frozen=True)
class WikiPair:
    """One supplied mention pair; labels are optional source-only evidence."""

    pair_id: str
    article_id: str
    sentence_id: str
    left: str
    right: str
    sentence: str
    marked: tuple[str, ...]
    kinds: tuple[str, str]
    labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("pair_id", "article_id", "sentence_id", "left", "right", "sentence"):
            object.__setattr__(self, name, str(getattr(self, name)))
        object.__setattr__(self, "marked", tuple(str(value) for value in self.marked))
        object.__setattr__(self, "kinds", tuple(str(value) for value in self.kinds))
        object.__setattr__(self, "labels", tuple(sorted({str(value) for value in self.labels})))
        if len(self.kinds) != 2:
            raise ValueError("kinds must contain the surface-left and surface-right kinds")
        for label in self.labels:
            _label_parts(label)

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> WikiPair:
        """Accept the public names plus explicit ``*_handle`` aliases."""
        left = row["left"] if "left" in row else row["left_handle"]
        right = row["right"] if "right" in row else row["right_handle"]
        marked = row["marked"] if "marked" in row else row["marked_tokens"]
        return cls(
            pair_id=str(row["pair_id"]),
            article_id=str(row["article_id"]),
            sentence_id=str(row["sentence_id"]),
            left=str(left),
            right=str(right),
            sentence=str(row["sentence"]),
            marked=tuple(str(value) for value in marked),
            kinds=tuple(str(value) for value in row["kinds"]),
            labels=tuple(str(value) for value in (row.get("labels") or ())),
        )

    def serving(self) -> WikiPair:
        return replace(self, labels=())

    def state(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "article_id": self.article_id,
            "sentence_id": self.sentence_id,
            "left": self.left,
            "right": self.right,
            "sentence": self.sentence,
            "marked": list(self.marked),
            "kinds": list(self.kinds),
            "labels": list(self.labels),
        }


def _default_axes() -> dict[str, tuple[str, ...]]:
    return {
        "between": ("word_sequence", "left_kind", "right_kind", "clause_scope", "polarity"),
        "located": FEATURE_AXES,
        "scope": FEATURE_AXES,
    }


@dataclass(frozen=True)
class WikiAxisConfig:
    """A bounded, serializable continuation policy for structural queries."""

    views: tuple[str, ...] = ("between", "located", "scope")
    active_axes: Mapping[str, tuple[str, ...]] | None = field(default=None)
    min_support: int = 2
    min_article_support: int = 1
    ambiguity_policy: str = "preserve"
    score_margin: float = 0.0
    max_candidates: int = 16
    max_features_per_view: int = 256

    def __post_init__(self) -> None:
        views = tuple(str(value) for value in self.views)
        if not views or len(set(views)) != len(views) or any(value not in VIEW_SCHEMAS for value in views):
            raise ValueError(f"views must be a unique subset of {tuple(VIEW_SCHEMAS)}")
        defaults = _default_axes()
        supplied = {} if self.active_axes is None else dict(self.active_axes)
        unknown_views = set(supplied) - set(VIEW_SCHEMAS)
        if unknown_views:
            raise ValueError(f"unknown active-axis views: {sorted(unknown_views)}")
        axes: dict[str, tuple[str, ...]] = {}
        for view in views:
            selected = tuple(str(value) for value in supplied.get(view, defaults[view]))
            if not selected or len(set(selected)) != len(selected):
                raise ValueError(f"active axes for {view!r} must be non-empty and unique")
            unknown = set(selected) - set(FEATURE_AXES)
            if unknown:
                raise ValueError(f"unknown axes for {view!r}: {sorted(unknown)}")
            axes[view] = selected
        if (isinstance(self.min_support, bool) or not isinstance(self.min_support, int)
                or not 1 <= self.min_support <= 100_000):
            raise ValueError("min_support must be in [1, 100000]")
        if (isinstance(self.min_article_support, bool) or not isinstance(self.min_article_support, int)
                or not 1 <= self.min_article_support <= 100_000):
            raise ValueError("min_article_support must be in [1, 100000]")
        if self.ambiguity_policy not in _VALID_POLICIES:
            raise ValueError(f"ambiguity_policy must be one of {sorted(_VALID_POLICIES)}")
        if not 0.0 <= float(self.score_margin) <= 1.0:
            raise ValueError("score_margin must be in [0, 1]")
        if (isinstance(self.max_candidates, bool) or not isinstance(self.max_candidates, int)
                or not 1 <= self.max_candidates <= 1024):
            raise ValueError("max_candidates must be in [1, 1024]")
        if (isinstance(self.max_features_per_view, bool) or not isinstance(self.max_features_per_view, int)
                or not 1 <= self.max_features_per_view <= 4096):
            raise ValueError("max_features_per_view must be in [1, 4096]")
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "active_axes", axes)
        object.__setattr__(self, "score_margin", float(self.score_margin))

    def state(self) -> dict[str, Any]:
        return {
            "views": list(self.views),
            "active_axes": {view: list(self.active_axes[view]) for view in self.views},
            "min_support": self.min_support,
            "min_article_support": self.min_article_support,
            "ambiguity_policy": self.ambiguity_policy,
            "score_margin": self.score_margin,
            "max_candidates": self.max_candidates,
            "max_features_per_view": self.max_features_per_view,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> WikiAxisConfig:
        active_axes = state.get("active_axes") or {}
        return cls(
            views=tuple(state.get("views", ("between", "located", "scope"))),
            active_axes={str(view): tuple(str(axis) for axis in axes)
                         for view, axes in active_axes.items()},
            min_support=int(state.get("min_support", 2)),
            min_article_support=int(state.get("min_article_support", 1)),
            ambiguity_policy=str(state.get("ambiguity_policy", "preserve")),
            score_margin=float(state.get("score_margin", 0.0)),
            max_candidates=int(state.get("max_candidates", 16)),
            max_features_per_view=int(state.get("max_features_per_view", 256)),
        )


@dataclass(frozen=True, order=True)
class _Feature:
    view: str
    coordinates: tuple[str, ...]

    def mapping(self) -> dict[str, str]:
        return dict(zip(FEATURE_AXES, self.coordinates, strict=True))


def _normalize_marked(marked: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in marked:
        marker = next((item for item in _MARKERS if value.lower() == item.lower()), None)
        if marker is not None:
            normalized.append(marker)
        else:
            normalized.extend(tokens(value))
    return tuple(normalized)


def _negative_positions(marked: tuple[str, ...]) -> set[int]:
    found = {index for index, value in enumerate(marked) if value in _NEGATION or value.endswith("n't")}
    for index in range(len(marked) - 2):
        if marked[index].endswith("n") and marked[index + 1:index + 3] == ("'", "t"):
            found.add(index)
    return found


def _scope_coordinates(
    marked: tuple[str, ...], left: int, right: int
) -> tuple[str, str, list[int], list[tuple[int, str]]]:
    clause, clause_at = 0, []
    stack: list[int] = []
    stack_at: list[tuple[int, ...]] = []
    group = 0
    structural: list[tuple[int, str]] = []
    for index, value in enumerate(marked):
        if value in _CLOSE and stack:
            stack.pop()
        clause_at.append(clause)
        stack_at.append(tuple(stack))
        if value in _OPEN:
            group += 1
            stack.append(group)
            structural.append((index, value))
        elif value in _CLOSE:
            structural.append((index, value))
        elif value in _CLAUSE_BREAK:
            structural.append((index, value))
            clause += 1

    clause_gap = abs(clause_at[right] - clause_at[left])
    clause_part = "same-clause" if not clause_gap else f"cross-clause:{_distance_bucket(clause_gap)}"
    left_stack, right_stack = stack_at[left], stack_at[right]
    common = 0
    for lvalue, rvalue in zip(left_stack, right_stack, strict=False):
        if lvalue != rvalue:
            break
        common += 1
    if left_stack == right_stack:
        paren_part = "same-root" if not left_stack else "same-parenthetical"
    elif common:
        paren_part = "nested-parenthetical"
    else:
        paren_part = "split-parenthetical"
    negative = _negative_positions(marked)
    if any(left < position < right for position in negative):
        polarity = "negative-between"
    elif any(clause_at[position] in {clause_at[left], clause_at[right]} for position in negative):
        polarity = "negative-clause"
    else:
        polarity = "no-negative-marker"
    return f"{clause_part}|{paren_part}", polarity, clause_at, structural


def _extract_features(pair: WikiPair, config: WikiAxisConfig) -> dict[str, list[_Feature]]:
    marked = _normalize_marked(pair.marked)
    if marked.count("<L>") != 1 or marked.count("<R>") != 1:
        raise ValueError("marked tokens require exactly one <L> and one <R>")
    left, right = marked.index("<L>"), marked.index("<R>")
    if left >= right:
        raise ValueError("<L> must precede <R> in surface order")
    between = marked[left + 1:right]
    clause_scope, polarity, clause_at, structural = _scope_coordinates(marked, left, right)
    common = {
        "left_kind": pair.kinds[0],
        "right_kind": pair.kinds[1],
        "clause_scope": clause_scope,
        "polarity": polarity,
    }
    result: dict[str, list[_Feature]] = {view: [] for view in config.views}

    if "between" in result:
        # Reuse the established exact-between feature definition.  Keeping the
        # sequence in one role makes it independently projectable from kinds,
        # scope, polarity, side, and distance.
        exact = sorted(features(" ".join(between), "exact"))
        for key in exact:
            coords = {
                "word_sequence": _word_sequence(key[1:]),
                "side": "between",
                "distance": _distance_bucket(len(between)),
                **common,
            }
            result["between"].append(_Feature("between", tuple(coords[axis] for axis in FEATURE_AXES)))

    if "located" in result:
        row = SimpleNamespace(
            pattern=" ".join(between),
            left="<left-handle-withheld>",
            right="<right-handle-withheld>",
            labels=(),
            kinds=pair.kinds,
            marked=marked,
            anchor_marked=tuple(value for value in marked if value != "<e>"),
        )
        for key in sorted(local_keys(row, "anchor_only", {})):
            # local_keys ABI: located, left kind, right kind, side, gap, words...
            coords = {
                "word_sequence": _word_sequence(key[5:]),
                "side": key[3],
                "distance": key[4],
                **common,
            }
            result["located"].append(_Feature("located", tuple(coords[axis] for axis in FEATURE_AXES)))

    if "scope" in result:
        marker_words = [value for position, value in structural if left < position < right]
        summary = {
            "word_sequence": _word_sequence(marker_words or ["<none>"]),
            "side": "pair",
            "distance": _distance_bucket(abs(clause_at[right] - clause_at[left])),
            **common,
        }
        result["scope"].append(_Feature("scope", tuple(summary[axis] for axis in FEATURE_AXES)))
        for position, value in structural:
            if position < left:
                side, distance = "left_before", left - position - 1
            elif position > right:
                side, distance = "right_after", position - right - 1
            elif left < position < right:
                side, distance = "between", min(position - left - 1, right - position - 1)
            else:
                continue
            coords = {
                "word_sequence": _word_sequence([value]),
                "side": side,
                "distance": _distance_bucket(distance),
                **common,
            }
            result["scope"].append(_Feature("scope", tuple(coords[axis] for axis in FEATURE_AXES)))

    for view in result:
        result[view] = sorted(set(result[view]))[:config.max_features_per_view]
    return result


def _coerce_pair(pair: WikiPair | Mapping[str, Any]) -> WikiPair:
    return pair if isinstance(pair, WikiPair) else WikiPair.from_dict(pair)


class WikiAxisField:
    """Source-induced multi-view reader backed only by projected chart reads."""

    def __init__(self) -> None:
        raise TypeError("use WikiAxisField.fit or WikiAxisField.from_state")

    @classmethod
    def fit(
        cls,
        source_pairs: Iterable[WikiPair | Mapping[str, Any]],
        config: WikiAxisConfig | Mapping[str, Any] | None = None,
    ) -> WikiAxisField:
        selected = (WikiAxisConfig() if config is None else config
                    if isinstance(config, WikiAxisConfig) else WikiAxisConfig.from_state(config))
        obj = cls.__new__(cls)
        obj.config = selected
        obj.fields = {view: _blank_field() for view in selected.views}
        obj.raw_pairs: list[dict[str, Any]] = []
        obj.provenance: dict[str, dict[str, Any]] = {}
        obj.fit_trace: list[dict[str, Any]] = []
        seen: dict[str, set[tuple[Any, ...]]] = {view: set() for view in selected.views}

        for pair_value in source_pairs:
            pair = _coerce_pair(pair_value)
            # Custody precedes feature interpretation; even malformed or
            # unlabeled source rows remain visible in the replayable state.
            obj.raw_pairs.append(pair.state())
            source = _source_id(pair)
            provenance = {
                "source": source,
                "article_id": pair.article_id,
                "pair_id": pair.pair_id,
                "sentence_id": pair.sentence_id,
            }
            previous = obj.provenance.get(source)
            if previous is not None and previous != provenance:
                raise ValueError("source identity collision")
            obj.provenance[source] = provenance
            try:
                extracted = _extract_features(pair, selected)
            except ValueError as exc:
                obj.fit_trace.append({**provenance, "status": "instrument-broken", "reason": str(exc),
                                      "labels": list(pair.labels), "feature_counts": {}})
                continue
            counts = {view: len(rows) for view, rows in extracted.items()}
            obj.fit_trace.append({**provenance, "status": "observed", "labels": list(pair.labels),
                                  "feature_counts": counts})
            # No labels means unknown, never a negative or a synthetic class.
            for label in pair.labels:
                relation, direction = _label_parts(label)
                for view, rows in extracted.items():
                    field_view = obj.fields[view]
                    for feature in rows:
                        roles = (*feature.coordinates, relation, direction)
                        identity = (source, roles)
                        if identity in seen[view]:
                            continue
                        seen[view].add(identity)
                        field_view.records.append(
                            Record(SOURCE_NAMESPACE, SOURCE_SCOPE, FEATURE_RELATION, roles,
                                   source, len(field_view.records))
                        )
        obj._model_sha256 = obj._compute_model_sha256()
        return obj

    @property
    def model_sha256(self) -> str:
        return self._model_sha256

    def _compute_model_sha256(self) -> str:
        material = {
            "version": STATE_VERSION,
            "schema": {view: VIEW_SCHEMAS[view] for view in self.config.views},
            "config": self.config.state(),
            "fields": {view: self.fields[view].state() for view in self.config.views},
        }
        return _digest(material)

    def state(self) -> dict[str, Any]:
        state = {
            "version": STATE_VERSION,
            "schema": {
                "name": "wiki-axis-field-v1",
                "views": {view: {"relation": FEATURE_RELATION, "axes": list(ROLE_AXES)}
                          for view in self.config.views},
                "fact": {"relation": FACT_RELATION, "axes": list(FACT_SCHEMA["axes"])},
            },
            "config": self.config.state(),
            "fields": {view: self.fields[view].state() for view in self.config.views},
            "raw_pairs": copy.deepcopy(self.raw_pairs),
            "provenance": copy.deepcopy(self.provenance),
            "fit_trace": copy.deepcopy(self.fit_trace),
        }
        state["model_sha256"] = self.model_sha256
        return _json_copy(state)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> WikiAxisField:
        if int(state.get("version", -1)) != STATE_VERSION:
            raise ValueError("unsupported WikiAxisField state version")
        schema = state.get("schema", {})
        if schema.get("name") != "wiki-axis-field-v1":
            raise ValueError("unsupported WikiAxisField schema")
        obj = cls.__new__(cls)
        obj.config = WikiAxisConfig.from_state(state["config"])
        obj.fields = {}
        for view in obj.config.views:
            declared = schema.get("views", {}).get(view)
            if declared != {"relation": FEATURE_RELATION, "axes": list(ROLE_AXES)}:
                raise ValueError(f"schema mismatch for view {view!r}")
            obj.fields[view] = RoleChartField.from_state(copy.deepcopy(state["fields"][view]))
            if any(record.relation != FEATURE_RELATION or len(record.roles) != len(ROLE_AXES)
                   for record in obj.fields[view].records):
                raise ValueError(f"invalid record shape for view {view!r}")
        obj.raw_pairs = copy.deepcopy(state.get("raw_pairs", []))
        obj.provenance = copy.deepcopy(state.get("provenance", {}))
        obj.fit_trace = copy.deepcopy(state.get("fit_trace", []))
        obj._model_sha256 = obj._compute_model_sha256()
        expected = state.get("model_sha256")
        if expected is not None and expected != obj._model_sha256:
            raise ValueError("WikiAxisField model fingerprint mismatch")
        return obj

    def predict(self, pair_value: WikiPair | Mapping[str, Any], *, detail: bool = True) -> dict[str, Any]:
        pair = _coerce_pair(pair_value)
        if pair.labels:
            raise ValueError("prediction accepts a label-free WikiPair; call serving() first")
        try:
            extracted = _extract_features(pair, self.config)
        except ValueError as exc:
            return {
                "status": "instrument-broken",
                "candidates": [],
                "trace": {"reason": str(exc), "query": pair.state(), "views": [],
                          "decision": {"policy": self.config.ambiguity_policy}},
                "feature_counts": {view: 0 for view in self.config.views},
            }

        accumulators: dict[tuple[str, str], dict[str, Any]] = {}
        trace_views: list[dict[str, Any]] = []
        matched_sources: set[str] = set()
        total_matches = 0
        for view in self.config.views:
            active_axes = tuple(self.config.active_axes[view])
            wildcard_axes = tuple(axis for axis in FEATURE_AXES if axis not in active_axes)
            feature_trace = []
            field_view = self.fields[view]
            for feature in extracted[view]:
                coordinates = feature.mapping()
                bindings = {AXIS_INDEX[axis]: coordinates[axis] for axis in active_axes}
                records = field_view.lookup(
                    FEATURE_RELATION, bindings, namespace=SOURCE_NAMESPACE, scope=SOURCE_SCOPE
                )
                total_matches += len(records)
                by_label: dict[tuple[str, str], set[str]] = defaultdict(set)
                all_sources: set[str] = set()
                for record in records:
                    label = (record.roles[AXIS_INDEX["relation"]], record.roles[AXIS_INDEX["direction"]])
                    by_label[label].add(record.source)
                    all_sources.add(record.source)
                matched_sources.update(all_sources)
                label_evidence = []
                for label, sources in sorted(by_label.items()):
                    articles = {self.provenance[source]["article_id"] for source in sources}
                    lower, upper = wilson(len(sources), len(all_sources))
                    eligible = (len(sources) >= self.config.min_support
                                and len(articles) >= self.config.min_article_support)
                    evidence = {
                        "view": view,
                        "feature": coordinates,
                        "pair_support": len(sources),
                        "article_support": len(articles),
                        "competing_pair_support": len(all_sources),
                        "wilson95": [lower, upper],
                        "eligible": eligible,
                        "sources": sorted(sources),
                    }
                    label_evidence.append({"relation": label[0], "direction": label[1], **evidence})
                    accumulator = accumulators.setdefault(
                        label,
                        {"sources": set(), "articles": set(), "evidence": [], "view_scores": defaultdict(list)},
                    )
                    accumulator["sources"].update(sources)
                    accumulator["articles"].update(articles)
                    if detail:
                        accumulator["evidence"].append(evidence)
                    accumulator["view_scores"][view].append((lower, eligible))
                if detail:
                    feature_trace.append(
                    {
                        "feature": coordinates,
                        "projected_bindings": {axis: coordinates[axis] for axis in active_axes},
                        "match_count": len(records),
                        "label_evidence": label_evidence,
                    }
                )
            trace_views.append(
                {
                    "view": view,
                    "relation": FEATURE_RELATION,
                    "active_axes": list(active_axes),
                    "wildcard_axes": list(wildcard_axes),
                    "lookup_count": len(extracted[view]),
                    "features": feature_trace,
                }
            )

        candidates = []
        for (relation, direction), accumulator in accumulators.items():
            view_scores = {
                view: max((score for score, eligible in values if eligible), default=0.0)
                for view, values in accumulator["view_scores"].items()
            }
            eligible_views = sorted(view for view, score in view_scores.items() if score > 0.0)
            score = sum(view_scores.values()) / len(self.config.views)
            eligible = bool(eligible_views)
            subject, obj = (pair.left, pair.right) if direction == "f" else (pair.right, pair.left)
            candidates.append(
                {
                    "relation": relation,
                    "direction": direction,
                    "subject": subject,
                    "object": obj,
                    "score": score,
                    "sources": sorted(accumulator["sources"]),
                    "support": len(accumulator["sources"]),
                    "article_support": len(accumulator["articles"]),
                    "matched_views": sorted(accumulator["view_scores"]),
                    "eligible_views": eligible_views,
                    "eligible": eligible,
                    "evidence": sorted(accumulator["evidence"], key=lambda item: (
                        item["view"], tuple(item["feature"].values()), -item["pair_support"]
                    )),
                }
            )
        candidates.sort(key=lambda item: (
            not item["eligible"], -item["score"], -item["support"], item["relation"], item["direction"]
        ))
        eligible = [item for item in candidates if item["eligible"]]
        selected: dict[str, Any] | None = None
        if self.config.ambiguity_policy == "preserve":
            if len(eligible) == 1:
                selected = eligible[0]
        elif eligible:
            runner_score = eligible[1]["score"] if len(eligible) > 1 else -1.0
            required = self.config.score_margin if self.config.ambiguity_policy == "margin" else 0.0
            if eligible[0]["score"] > runner_score + required:
                selected = eligible[0]

        candidate_total = len(candidates)
        truncated = candidate_total > self.config.max_candidates
        candidates = candidates[:self.config.max_candidates]
        if not accumulators:
            status = "missing"
        elif not eligible:
            status = "insufficient-evidence"
        elif selected is not None:
            status = "resolved"
        else:
            status = "ambiguous"
        decision = {
            "policy": self.config.ambiguity_policy,
            "score_margin": self.config.score_margin,
            "eligible_candidates": len(eligible),
            "total_candidates": candidate_total,
            "selected": None if selected is None else {
                "relation": selected["relation"], "direction": selected["direction"]
            },
            "candidate_limit": self.config.max_candidates,
            "truncated": truncated,
        }
        trace = {
            "query": pair.state(),
            "model_sha256": self.model_sha256,
            "views": trace_views,
            "lookup_geometry": {
                "logical_projected_reads": sum(len(rows) for rows in extracted.values()),
                "matching_source_records": total_matches,
                "view_indices": len(self.config.views),
            },
            "source_provenance": ({source: copy.deepcopy(self.provenance[source])
                                   for source in sorted(matched_sources)} if detail else {}),
            "decision": decision,
        }
        return {
            "status": status,
            "candidates": candidates,
            "trace": trace,
            "feature_counts": {view: len(extracted[view]) for view in self.config.views},
        }

    def infer(self, pair: WikiPair | Mapping[str, Any]) -> dict[str, Any]:
        return self.predict(pair)

    def populate(
        self,
        fabric: Any,
        view: str = "wiki-facts",
        pairs: Iterable[WikiPair | Mapping[str, Any]] = (),
        namespace: str = "wiki",
        scope: str = "default",
        *,
        detail: bool = True,
    ) -> dict[str, Any]:
        """Retain raw rows and write only explicitly resolved native fact tuples."""
        if view not in fabric.views:
            fabric.register_view(view, _blank_field())
        fact_field = fabric.view(view)
        if any(int(chart["relation"]) == FACT_RELATION
               and len(chart.get("alignment", chart.get("classes", ()))) != 3
               for chart in fact_field.charts):
            raise ValueError("existing fact view relation 0 has a conflicting chart arity")
        if any(record.relation != FACT_RELATION or len(record.roles) != 3 for record in fact_field.records):
            raise ValueError("existing fact view does not implement (predicate, subject, object)")
        existing = {
            (record.namespace, record.scope, record.relation, record.roles, record.source)
            for record in fact_field.records
        }
        results, written = [], 0
        for pair_value in pairs:
            pair = _coerce_pair(pair_value)
            if pair.labels:
                raise ValueError("populate accepts label-free WikiPair rows")
            raw_source = f"wiki-target:{_digest([pair.article_id, pair.pair_id])[:24]}"
            fabric.observe_raw(
                view, pair.state(), namespace, scope, source=raw_source,
                provisional=True, status="reader-input",
            )
            result = self.predict(pair, detail=detail)
            results.append({"pair_id": pair.pair_id, "article_id": pair.article_id, **result})
            if result["status"] != "resolved":
                continue
            selected = result["trace"]["decision"]["selected"]
            candidate = next(item for item in result["candidates"]
                             if item["relation"] == selected["relation"]
                             and item["direction"] == selected["direction"])
            roles = (candidate["relation"], candidate["subject"], candidate["object"])
            source = f"wiki-inference:{_digest([self.model_sha256, raw_source, roles, candidate['sources']])[:24]}"
            identity = (namespace, scope, FACT_RELATION, roles, source)
            if identity in existing:
                continue
            fact_field.records.append(
                Record(namespace, scope, FACT_RELATION, roles, source, len(fact_field.records))
            )
            existing.add(identity)
            written += 1
        fabric.invalidate_view(view)
        return {
            "pairs": len(results),
            "written": written,
            "status_counts": dict(sorted(
                {status: sum(row["status"] == status for row in results)
                 for status in {row["status"] for row in results}}.items()
            )),
            "view": view,
            "namespace": namespace,
            "scope": scope,
            "fact_schema": {"relation": FACT_RELATION, "roles": list(FACT_SCHEMA["axes"])},
            "results": results,
        }
