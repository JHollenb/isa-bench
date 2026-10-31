"""Deterministic checkpoint promotion and rollback for bounded training.

The controller owns the mutable training boundary, not the training objective.
It snapshots model parameters/buffers, optimizer state, gradients, module modes,
process RNG state, named ``torch.Generator`` objects, and a caller-owned cursor.
Validation, autonomous rollout, and trace-endpoint scores are evaluated under an
isolated RNG/cursor guard.  A candidate is retained only when an explicit
lexicographic or Pareto policy promotes it; every other outcome restores the
previous accepted state and verifies its full fingerprint.

Decision receipts contain no tensors.  They are canonical, hash chained, and
immutable.  ``ImmutableDirectoryReceiptStore`` can publish them with an atomic
no-overwrite link; callers with a remote store can provide the same ``put``
interface (for example, an adapter around a Saturn session-step receipt).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import re
import tempfile
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeVar

import torch

try:
    import numpy as np
except ImportError:  # pragma: no cover - Saturn's shared environment includes NumPy.
    np = None  # type: ignore[assignment]

TRAINING_CHECKPOINT_SCHEMA = "saturn-ar-training-state-checkpoint-v1"
EVALUATION_SCORES_SCHEMA = "saturn-ar-training-evaluation-scores-v1"
PROMOTION_POLICY_SCHEMA = "saturn-ar-training-promotion-policy-v1"
DECISION_RECEIPT_SCHEMA = "saturn-ar-training-checkpoint-decision-receipt-v1"

ScoreSource = Literal["validation", "autonomous_rollout", "trace_endpoints"]
Direction = Literal["maximize", "minimize"]
PolicyKind = Literal["lexicographic", "pareto"]
_SerializationResult = TypeVar("_SerializationResult")

_SOURCES: tuple[ScoreSource, ...] = (
    "validation",
    "autonomous_rollout",
    "trace_endpoints",
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class TrainingCheckpointError(RuntimeError):
    """Base error for deterministic training checkpoint control."""


class TrainingStateDriftError(TrainingCheckpointError):
    """Raised when the live state no longer matches the accepted checkpoint."""


class EvaluationMutationError(TrainingCheckpointError):
    """Raised when an evaluator mutates model or optimizer state."""


class TrainingIntervalError(TrainingCheckpointError):
    """Raised after a failed interval has been rolled back and receipted."""

    def __init__(self, message: str, *, receipt: DecisionReceipt) -> None:
        super().__init__(message)
        self.receipt = receipt


class DecisionReceiptError(TrainingCheckpointError):
    """Raised when an immutable decision receipt cannot be verified or stored."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_name(value: str, *, label: str) -> str:
    if not isinstance(value, str) or _NAME_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be 1-128 ASCII letters, digits, '.', '_' or '-'")
    return value


def _json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} mapping keys must be non-empty strings")
            result[key] = _json_value(item, label=f"{label}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_json_value(item, label=f"{label}[{index}]") for index, item in enumerate(value)]
    raise ValueError(f"{label} is not JSON-like: {type(value).__name__}")


def _metric_map(value: Mapping[str, float], *, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must contain at least one score")
    result: dict[str, float] = {}
    for name, score in value.items():
        resolved_name = _validate_name(name, label=f"{label} metric")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{label}.{resolved_name} must be numeric")
        resolved_score = float(score)
        if not math.isfinite(resolved_score):
            raise ValueError(f"{label}.{resolved_name} must be finite")
        result[resolved_name] = resolved_score
    return dict(sorted(result.items()))


class EvaluationScores:
    """Immutable scores from the three required checkpoint observables."""

    __slots__ = ("_body", "_fingerprint")

    def __init__(
        self,
        *,
        validation: Mapping[str, float],
        autonomous_rollout: Mapping[str, float],
        trace_endpoints: Mapping[str, float],
        evidence: Mapping[str, Any] | None = None,
    ) -> None:
        body = {
            "schema": EVALUATION_SCORES_SCHEMA,
            "validation": _metric_map(validation, label="validation"),
            "autonomous_rollout": _metric_map(autonomous_rollout, label="autonomous_rollout"),
            "trace_endpoints": _metric_map(trace_endpoints, label="trace_endpoints"),
            "evidence": _json_value(evidence or {}, label="evidence"),
        }
        self._body = _canonical_json_bytes(body)
        self._fingerprint = hashlib.sha256(self._body).hexdigest()

    @classmethod
    def from_value(cls, value: EvaluationScores | Mapping[str, Any]) -> EvaluationScores:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("checkpoint evaluator must return EvaluationScores or a mapping")
        return cls(
            validation=value.get("validation", {}),
            autonomous_rollout=value.get("autonomous_rollout", {}),
            trace_endpoints=value.get("trace_endpoints", {}),
            evidence=value.get("evidence", {}),
        )

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def metric(self, source: ScoreSource, name: str) -> float:
        payload = json.loads(self._body)
        try:
            return float(payload[source][name])
        except KeyError as exc:
            raise ValueError(f"evaluation is missing policy score {source}.{name}") from exc

    def to_dict(self) -> dict[str, Any]:
        return {**json.loads(self._body), "fingerprint": self._fingerprint}


TrainStep = Callable[[int], Mapping[str, Any] | None]
Evaluator = Callable[[], EvaluationScores | Mapping[str, Any]]
CursorCapture = Callable[[], Any]
CursorRestore = Callable[[Any], None]


@dataclass(frozen=True, slots=True)
class PromotionObjective:
    source: ScoreSource
    metric: str
    direction: Direction
    tolerance: float = 0.0

    def __post_init__(self) -> None:
        if self.source not in _SOURCES:
            raise ValueError(f"unsupported score source {self.source!r}")
        _validate_name(self.metric, label="objective metric")
        if self.direction not in {"maximize", "minimize"}:
            raise ValueError(f"unsupported objective direction {self.direction!r}")
        if (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(float(self.tolerance))
            or float(self.tolerance) < 0.0
        ):
            raise ValueError("objective tolerance must be finite and non-negative")
        object.__setattr__(self, "tolerance", float(self.tolerance))

    @property
    def key(self) -> tuple[str, str]:
        return self.source, self.metric

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "metric": self.metric,
            "direction": self.direction,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class PromotionPolicy:
    kind: PolicyKind
    objectives: tuple[PromotionObjective, ...]
    require_all_score_sources: bool = field(default=True, init=False)
    schema: str = field(default=PROMOTION_POLICY_SCHEMA, init=False)

    def __post_init__(self) -> None:
        if self.kind not in {"lexicographic", "pareto"}:
            raise ValueError(f"unsupported promotion policy {self.kind!r}")
        objectives = tuple(self.objectives)
        if not objectives:
            raise ValueError("promotion policy requires at least one objective")
        if len({objective.key for objective in objectives}) != len(objectives):
            raise ValueError("promotion objectives must be unique")
        if {objective.source for objective in objectives} != set(_SOURCES):
            raise ValueError(
                "promotion policy must score validation, autonomous rollout, and trace endpoints"
            )
        object.__setattr__(self, "objectives", objectives)

    @property
    def fingerprint(self) -> str:
        return _digest_json(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "kind": self.kind,
            "objectives": [objective.to_dict() for objective in self.objectives],
            "require_all_score_sources": self.require_all_score_sources,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    def compare(self, baseline: EvaluationScores, candidate: EvaluationScores) -> dict[str, Any]:
        comparisons = []
        states: list[str] = []
        for objective in self.objectives:
            baseline_score = baseline.metric(objective.source, objective.metric)
            candidate_score = candidate.metric(objective.source, objective.metric)
            oriented_delta = (
                candidate_score - baseline_score
                if objective.direction == "maximize"
                else baseline_score - candidate_score
            )
            if oriented_delta > objective.tolerance:
                state = "better"
            elif oriented_delta < -objective.tolerance:
                state = "worse"
            else:
                state = "equivalent"
            states.append(state)
            comparisons.append(
                {
                    **objective.to_dict(),
                    "baseline": baseline_score,
                    "candidate": candidate_score,
                    "oriented_delta": oriented_delta,
                    "comparison": state,
                }
            )
        decisive_objective: int | None = None
        if self.kind == "lexicographic":
            decisive_objective = next(
                (index for index, state in enumerate(states) if state != "equivalent"),
                None,
            )
            if decisive_objective is None:
                promoted = False
                relation = "equivalent"
            else:
                promoted = states[decisive_objective] == "better"
                relation = "lexicographic-improvement" if promoted else "lexicographic-regression"
        else:
            has_better = "better" in states
            has_worse = "worse" in states
            promoted = has_better and not has_worse
            if promoted:
                relation = "pareto-dominates-baseline"
            elif has_worse and has_better:
                relation = "pareto-tradeoff"
            elif has_worse:
                relation = "pareto-dominated-by-baseline"
            else:
                relation = "equivalent"
        return {
            "policy_fingerprint": self.fingerprint,
            "kind": self.kind,
            "promoted": promoted,
            "relation": relation,
            "decisive_objective_index": decisive_objective,
            "comparisons": comparisons,
            "scalarized_score_used": False,
        }


def _tensor_bytes(value: torch.Tensor) -> bytes:
    if value.layout is not torch.strided:
        raise TrainingCheckpointError(
            f"checkpoint state requires strided tensors, found {value.layout}"
        )
    if value.device.type == "meta":
        raise TrainingCheckpointError("checkpoint state cannot contain meta tensors")
    contiguous = value.detach().cpu().contiguous()
    if contiguous.numel() == 0:
        return b""
    return contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()


def _tensor_view_key(value: torch.Tensor) -> tuple[Any, ...]:
    if value.layout is not torch.strided or value.device.type == "meta":
        _tensor_bytes(value)
    storage = value.untyped_storage()
    return (
        str(value.device),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        str(value.dtype),
        int(value.storage_offset()),
        tuple(int(item) for item in value.shape),
        tuple(int(item) for item in value.stride()),
    )


def _state_manifest(value: Any, *, _tensor_memo: dict[tuple[Any, ...], Any] | None = None) -> Any:
    if _tensor_memo is None:
        _tensor_memo = {}
    if isinstance(value, torch.Tensor):
        key = _tensor_view_key(value)
        cached = _tensor_memo.get(key)
        if cached is not None:
            return cached
        raw = _tensor_bytes(value)
        manifest = {
            "kind": "torch.Tensor",
            "dtype": str(value.dtype),
            "device": str(value.device),
            "shape": [int(item) for item in value.shape],
            "stride": [int(item) for item in value.stride()],
            "storage_offset": int(value.storage_offset()),
            "numel": int(value.numel()),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        _tensor_memo[key] = manifest
        return manifest
    if np is not None and isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "numpy.ndarray",
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        entries = [
            (
                _state_manifest(key, _tensor_memo=_tensor_memo),
                _state_manifest(item, _tensor_memo=_tensor_memo),
            )
            for key, item in value.items()
        ]
        entries.sort(key=lambda row: _canonical_json_bytes(row[0]))
        return {"kind": "mapping", "entries": entries}
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_state_manifest(item, _tensor_memo=_tensor_memo) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_state_manifest(item, _tensor_memo=_tensor_memo) for item in value],
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if value is None:
        return {"kind": "none"}
    if isinstance(value, bool):
        return {"kind": "bool", "value": value}
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, float):
        return {"kind": "float", "hex": value.hex()}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    if isinstance(value, torch.dtype):
        return {"kind": "torch.dtype", "value": str(value)}
    if isinstance(value, torch.device):
        return {"kind": "torch.device", "value": str(value)}
    raise TrainingCheckpointError(f"checkpoint state contains unsupported {type(value).__name__}")


def _state_digest(value: Any) -> str:
    return _digest_json(_state_manifest(value))


def _training_state_fingerprints(value: Mapping[str, Any]) -> tuple[str, str]:
    """Hash the complete state and its model component in one tensor pass."""

    manifest = _state_manifest(value)
    if not isinstance(manifest, Mapping) or manifest.get("kind") != "mapping":
        raise TrainingCheckpointError("training state manifest is not a mapping")
    model_key = {"kind": "str", "value": "model_state"}
    model_manifest = None
    for row in manifest.get("entries", ()):
        if isinstance(row, (tuple, list)) and len(row) == 2 and row[0] == model_key:
            model_manifest = row[1]
            break
    if model_manifest is None:
        raise TrainingCheckpointError("training state manifest has no model state")
    return _digest_json(manifest), _digest_json(model_manifest)


def _clone_state(
    value: Any, *, _tensor_memo: dict[tuple[Any, ...], torch.Tensor] | None = None
) -> Any:
    if _tensor_memo is None:
        _tensor_memo = {}
    if isinstance(value, torch.Tensor):
        if value.layout is not torch.strided or value.device.type == "meta":
            _tensor_bytes(value)
        key = _tensor_view_key(value)
        if key not in _tensor_memo:
            _tensor_memo[key] = value.detach().cpu().clone(memory_format=torch.preserve_format)
        return _tensor_memo[key]
    if np is not None and isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {
            copy.deepcopy(key): _clone_state(item, _tensor_memo=_tensor_memo)
            for key, item in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_clone_state(item, _tensor_memo=_tensor_memo) for item in value)
    if isinstance(value, list):
        return [_clone_state(item, _tensor_memo=_tensor_memo) for item in value]
    if value is None or isinstance(
        value, (str, bool, int, float, bytes, torch.dtype, torch.device)
    ):
        return value
    return copy.deepcopy(value)


def _tensor_inventory(value: Any, *, _seen: set[tuple[Any, ...]] | None = None) -> dict[str, int]:
    if _seen is None:
        _seen = set()
    tensors = 0
    elements = 0
    bytes_ = 0
    if isinstance(value, torch.Tensor):
        key = _tensor_view_key(value)
        if key in _seen:
            return {"tensors": 0, "elements": 0, "bytes": 0}
        _seen.add(key)
        return {
            "tensors": 1,
            "elements": int(value.numel()),
            "bytes": int(value.numel() * value.element_size()),
        }
    if isinstance(value, Mapping):
        children = tuple(value.values())
    elif isinstance(value, (tuple, list)):
        children = tuple(value)
    else:
        children = ()
    for child in children:
        inventory = _tensor_inventory(child, _seen=_seen)
        tensors += inventory["tensors"]
        elements += inventory["elements"]
        bytes_ += inventory["bytes"]
    return {"tensors": tensors, "elements": elements, "bytes": bytes_}


def _capture_runtime_flags() -> dict[str, Any]:
    warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", None)
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": bool(warn_only()) if warn_only else False,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    }


def _restore_runtime_flags(value: Mapping[str, Any]) -> None:
    torch.use_deterministic_algorithms(
        bool(value["deterministic_algorithms"]),
        warn_only=bool(value["deterministic_algorithms_warn_only"]),
    )
    torch.set_float32_matmul_precision(str(value["float32_matmul_precision"]))
    torch.backends.cudnn.benchmark = bool(value["cudnn_benchmark"])
    torch.backends.cudnn.deterministic = bool(value["cudnn_deterministic"])
    torch.backends.cudnn.allow_tf32 = bool(value["cudnn_allow_tf32"])
    torch.backends.cuda.matmul.allow_tf32 = bool(value["cuda_matmul_allow_tf32"])


def _capture_rng(generators: Mapping[str, torch.Generator]) -> dict[str, Any]:
    numpy_state = np.random.get_state() if np is not None else None
    # Freeze the CUDA RNG inventory at the first checkpoint whenever CUDA is
    # available.  Some CPU optimizers probe CUDA lazily (AdamW does this in
    # recent PyTorch releases), so conditioning this capture on
    # ``is_initialized()`` lets a baseline contain zero CUDA generators while
    # the candidate contains one.  CUDA cannot be de-initialized during an
    # exact rollback; eagerly materializing and capturing every visible device
    # makes the inventory stable before the baseline fingerprint is sealed.
    cuda_states = tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    return {
        "python": random.getstate(),
        "numpy": numpy_state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
        "named_torch_generators": {
            name: generator.get_state() for name, generator in generators.items()
        },
        "runtime_flags": _capture_runtime_flags(),
    }


def _restore_rng(value: Mapping[str, Any], generators: Mapping[str, torch.Generator]) -> None:
    random.setstate(value["python"])
    if np is not None and value["numpy"] is not None:
        np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch_cpu"])
    cuda_states = tuple(value["torch_cuda"])
    if cuda_states:
        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            raise TrainingCheckpointError("cannot restore captured CUDA RNG state")
        if len(cuda_states) != torch.cuda.device_count():
            raise TrainingCheckpointError("CUDA RNG device inventory changed")
        torch.cuda.set_rng_state_all(list(cuda_states))
    named = value["named_torch_generators"]
    if set(named) != set(generators):
        raise TrainingCheckpointError("named torch.Generator inventory changed")
    for name, generator in generators.items():
        generator.set_state(named[name])
    _restore_runtime_flags(value["runtime_flags"])


def _module_modes(model: torch.nn.Module) -> dict[str, bool]:
    return {name: bool(module.training) for name, module in model.named_modules()}


def _restore_module_modes(model: torch.nn.Module, modes: Mapping[str, bool]) -> None:
    modules = dict(model.named_modules())
    if set(modules) != set(modes):
        raise TrainingCheckpointError("model module inventory changed")
    for name, training in modes.items():
        modules[name].training = bool(training)


def _parameter_flags(model: torch.nn.Module) -> dict[str, bool]:
    return {name: bool(parameter.requires_grad) for name, parameter in model.named_parameters()}


def _restore_parameter_flags(model: torch.nn.Module, flags: Mapping[str, bool]) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(flags):
        raise TrainingCheckpointError("model parameter inventory changed")
    for name, requires_grad in flags.items():
        parameters[name].requires_grad_(bool(requires_grad))


def _parameter_gradients(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else _clone_state(parameter.grad)
        for name, parameter in model.named_parameters()
    }


def _restore_parameter_gradients(
    model: torch.nn.Module, gradients: Mapping[str, torch.Tensor | None]
) -> None:
    parameters = dict(model.named_parameters())
    if set(parameters) != set(gradients):
        raise TrainingCheckpointError("model gradient inventory changed")
    for name, saved in gradients.items():
        parameter = parameters[name]
        if saved is None:
            parameter.grad = None
        else:
            parameter.grad = saved.to(device=parameter.device, dtype=parameter.dtype).clone()


def _nonpersistent_buffers(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        for local_name in module._non_persistent_buffers_set:
            value = module._buffers.get(local_name)
            if value is None:
                continue
            name = f"{module_name}.{local_name}" if module_name else local_name
            result[name] = value
    return result


def _optimizer_parameter_topology(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[tuple[str, ...], ...]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    observed: set[int] = set()
    groups = []
    for group_index, group in enumerate(optimizer.param_groups):
        names = []
        for parameter in group.get("params", ()):  # type: ignore[union-attr]
            identity = id(parameter)
            if identity not in names_by_id:
                raise TrainingCheckpointError(
                    f"optimizer group {group_index} owns a parameter outside the model"
                )
            if identity in observed:
                raise TrainingCheckpointError(
                    "optimizer parameter appears in more than one parameter group"
                )
            observed.add(identity)
            names.append(names_by_id[identity])
        if not names:
            raise TrainingCheckpointError(f"optimizer group {group_index} has no model parameters")
        groups.append(tuple(names))
    if not groups:
        raise TrainingCheckpointError("optimizer has no parameter groups")
    return tuple(groups)


def _restore_optimizer(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    state: Mapping[str, Any],
    defaults: Mapping[str, Any],
    topology: tuple[tuple[str, ...], ...],
) -> None:
    parameters = dict(model.named_parameters())
    saved_groups = state.get("param_groups")
    if not isinstance(saved_groups, list) or len(saved_groups) != len(topology):
        raise TrainingCheckpointError("saved optimizer parameter groups changed")
    restored_groups = []
    for saved_group, names in zip(saved_groups, topology, strict=True):
        if not isinstance(saved_group, Mapping) or set(names) - set(parameters):
            raise TrainingCheckpointError("saved optimizer topology cannot be restored")
        restored_groups.append(
            {
                **{
                    key: _clone_state(value)
                    for key, value in saved_group.items()
                    if key != "params"
                },
                "params": [parameters[name] for name in names],
            }
        )
    optimizer.defaults.clear()
    optimizer.defaults.update(_clone_state(defaults))
    optimizer.param_groups = restored_groups
    optimizer.state = defaultdict(dict)
    optimizer.load_state_dict(_clone_state(state))
    if _optimizer_parameter_topology(model, optimizer) != topology:
        raise TrainingCheckpointError("restored optimizer topology changed")


def _restore_nonpersistent_buffers(
    model: torch.nn.Module, buffers: Mapping[str, torch.Tensor]
) -> None:
    current = _nonpersistent_buffers(model)
    if set(current) != set(buffers):
        raise TrainingCheckpointError("model non-persistent buffer inventory changed")
    with torch.no_grad():
        for name, saved in buffers.items():
            target = current[name]
            target.copy_(saved.to(device=target.device, dtype=target.dtype))


@dataclass(frozen=True, slots=True)
class TrainingStateSnapshot:
    """Opaque exact-restoration snapshot plus a tensor-free public summary."""

    fingerprint: str
    summary: Mapping[str, Any]
    _state_manifest: Mapping[str, Any] = field(repr=False)
    _model_state_fingerprint: str = field(repr=False)
    _model_state: Mapping[str, Any] = field(repr=False)
    _optimizer_state: Mapping[str, Any] = field(repr=False)
    _optimizer_defaults: Mapping[str, Any] = field(repr=False)
    _optimizer_topology: tuple[tuple[str, ...], ...] = field(repr=False)
    _gradients: Mapping[str, torch.Tensor | None] = field(repr=False)
    _nonpersistent_buffers: Mapping[str, torch.Tensor] = field(repr=False)
    _module_modes: Mapping[str, bool] = field(repr=False)
    _parameter_flags: Mapping[str, bool] = field(repr=False)
    _rng: Mapping[str, Any] = field(repr=False)
    _cursor: Any = field(repr=False)
    schema: str = field(default=TRAINING_CHECKPOINT_SCHEMA, init=False)

    @property
    def model_state_fingerprint(self) -> str:
        """Fingerprint of the deployment-visible subset captured with this state."""

        return self._model_state_fingerprint

    def _state_parts(self) -> dict[str, Any]:
        return {
            "model_state": self._model_state,
            "optimizer_state": self._optimizer_state,
            "optimizer_defaults": self._optimizer_defaults,
            "optimizer_parameter_topology": self._optimizer_topology,
            "gradients": self._gradients,
            "nonpersistent_buffers": self._nonpersistent_buffers,
            "module_modes": self._module_modes,
            "parameter_requires_grad": self._parameter_flags,
            "rng": self._rng,
            "cursor": self._cursor,
        }

    @staticmethod
    def _model_manifest(manifest: Mapping[str, Any]) -> Any:
        if manifest.get("kind") != "mapping":
            raise TrainingCheckpointError("training snapshot manifest is not a mapping")
        model_key = {"kind": "str", "value": "model_state"}
        for row in manifest.get("entries", ()):
            if isinstance(row, (tuple, list)) and len(row) == 2 and row[0] == model_key:
                return row[1]
        raise TrainingCheckpointError("training snapshot manifest has no model state")

    @classmethod
    def _manifest_matches(cls, expected: Any, observed: Any) -> bool:
        """Compare serialized state content while allowing device relocation."""

        if isinstance(expected, Mapping) and isinstance(observed, Mapping):
            if set(expected) != set(observed):
                return False
            for key in expected:
                if (
                    key == "device"
                    and expected.get("kind") == "torch.Tensor"
                    and observed.get("kind") == "torch.Tensor"
                ):
                    continue
                if not cls._manifest_matches(expected[key], observed[key]):
                    return False
            return True
        if isinstance(expected, (tuple, list)) and isinstance(observed, (tuple, list)):
            return len(expected) == len(observed) and all(
                cls._manifest_matches(left, right)
                for left, right in zip(expected, observed, strict=True)
            )
        return expected == observed

    def verify(self) -> None:
        """Verify snapshot custody before a caller mutates its live runtime."""

        if self.schema != TRAINING_CHECKPOINT_SCHEMA:
            raise TrainingCheckpointError("training snapshot schema changed")
        try:
            fingerprint = _digest_json(self._state_manifest)
            model_fingerprint = _digest_json(self._model_manifest(self._state_manifest))
            observed_manifest = _state_manifest(self._state_parts())
        except Exception as exc:
            raise TrainingCheckpointError("training snapshot manifest is invalid") from exc
        if (
            fingerprint != self.fingerprint
            or model_fingerprint != self.model_state_fingerprint
            or not self._manifest_matches(self._state_manifest, observed_manifest)
        ):
            raise TrainingCheckpointError("training snapshot fingerprint changed")

    def to_state_dict(self) -> dict[str, Any]:
        """Return a detached, tensor-bearing payload for safe external packing.

        The payload contains no executable objects or pickle bytes.  Callers
        should encode tensor leaves with a tensor-safe format such as
        safetensors and encode the remaining tagged tree as canonical JSON.
        """

        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint,
            "summary": _clone_state(dict(self.summary)),
            "state_manifest": _clone_state(self._state_manifest),
            "model_state_fingerprint": self.model_state_fingerprint,
            "model_state": _clone_state(self._model_state),
            "optimizer_state": _clone_state(self._optimizer_state),
            "optimizer_defaults": _clone_state(self._optimizer_defaults),
            "optimizer_topology": _clone_state(self._optimizer_topology),
            "gradients": _clone_state(self._gradients),
            "nonpersistent_buffers": _clone_state(self._nonpersistent_buffers),
            "module_modes": _clone_state(self._module_modes),
            "parameter_flags": _clone_state(self._parameter_flags),
            "rng": _clone_state(self._rng),
            "cursor": _clone_state(self._cursor),
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> TrainingStateSnapshot:
        """Rehydrate a safe payload and reject any content or schema drift."""

        if not isinstance(payload, Mapping):
            raise TrainingCheckpointError("training snapshot payload must be a mapping")
        expected_keys = {
            "schema",
            "fingerprint",
            "summary",
            "state_manifest",
            "model_state_fingerprint",
            "model_state",
            "optimizer_state",
            "optimizer_defaults",
            "optimizer_topology",
            "gradients",
            "nonpersistent_buffers",
            "module_modes",
            "parameter_flags",
            "rng",
            "cursor",
        }
        if set(payload) != expected_keys:
            raise TrainingCheckpointError("training snapshot payload fields changed")
        if payload.get("schema") != TRAINING_CHECKPOINT_SCHEMA:
            raise TrainingCheckpointError("training snapshot schema changed")
        summary = payload.get("summary")
        state_manifest = payload.get("state_manifest")
        model_state = payload.get("model_state")
        optimizer_state = payload.get("optimizer_state")
        optimizer_defaults = payload.get("optimizer_defaults")
        gradients = payload.get("gradients")
        nonpersistent = payload.get("nonpersistent_buffers")
        modes = payload.get("module_modes")
        flags = payload.get("parameter_flags")
        rng = payload.get("rng")
        if not isinstance(summary, Mapping) or not isinstance(state_manifest, Mapping):
            raise TrainingCheckpointError("training snapshot summary or manifest is invalid")
        for value, label in (
            (model_state, "model state"),
            (optimizer_state, "optimizer state"),
            (optimizer_defaults, "optimizer defaults"),
            (gradients, "gradients"),
            (nonpersistent, "non-persistent buffers"),
            (modes, "module modes"),
            (flags, "parameter flags"),
            (rng, "RNG state"),
        ):
            if not isinstance(value, Mapping):
                raise TrainingCheckpointError(f"training snapshot {label} must be a mapping")
        topology = payload.get("optimizer_topology")
        if not isinstance(topology, (tuple, list)):
            raise TrainingCheckpointError("training snapshot optimizer topology is invalid")
        normalized_topology: tuple[tuple[str, ...], ...] = tuple(
            tuple(_validate_name(str(name), label="optimizer parameter name") for name in group)
            if isinstance(group, (tuple, list))
            else (_validate_name(str(group), label="optimizer parameter name"),)
            for group in topology
        )
        if any(not group for group in normalized_topology):
            raise TrainingCheckpointError("training snapshot optimizer topology is empty")
        if not normalized_topology and (
            optimizer_state != {"state": {}, "param_groups": []} or optimizer_defaults != {}
        ):
            raise TrainingCheckpointError("optimizer-free snapshot contains optimizer state")
        flattened = tuple(name for group in normalized_topology for name in group)
        if len(set(flattened)) != len(flattened):
            raise TrainingCheckpointError(
                "training snapshot optimizer topology contains duplicate parameters"
            )
        try:
            normalized_summary = _json_value(summary, label="training snapshot summary")
            normalized_manifest = _json_value(
                state_manifest, label="training snapshot state manifest"
            )
        except Exception as exc:
            raise TrainingCheckpointError("training snapshot JSON fields are invalid") from exc
        snapshot = cls(
            fingerprint=str(payload.get("fingerprint")),
            summary=MappingProxyType(normalized_summary),
            _state_manifest=normalized_manifest,
            _model_state_fingerprint=str(payload.get("model_state_fingerprint")),
            _model_state=_clone_state(model_state),
            _optimizer_state=_clone_state(optimizer_state),
            _optimizer_defaults=_clone_state(optimizer_defaults),
            _optimizer_topology=normalized_topology,
            _gradients=_clone_state(gradients),
            _nonpersistent_buffers=_clone_state(nonpersistent),
            _module_modes=_clone_state(modes),
            _parameter_flags=_clone_state(flags),
            _rng=_clone_state(rng),
            _cursor=_clone_state(payload.get("cursor")),
        )
        snapshot.verify()
        return snapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "fingerprint": self.fingerprint,
            **dict(self.summary),
        }


@dataclass(frozen=True, slots=True)
class AcceptedStateSerializationView:
    """Ephemeral read-only borrow of one verified accepted CPU snapshot.

    The controller owns every object reachable through ``runtime_state``.  A
    serializer may inspect those objects synchronously but must never mutate or
    retain them.  Keeping the borrow callback-scoped lets resume custody avoid a
    second optimizer/RNG/cursor snapshot without making rollback state public.
    """

    accepted_state_fingerprint: str
    model_state_fingerprint: str
    runtime_state: Mapping[str, Any] = field(repr=False)


class TrainingStateBoundary:
    """Policy-free exact state boundary shared by training and structural runtimes.

    The boundary owns the mutable model/optimizer pair and the caller-owned
    cursor.  It captures enough runtime state to restore the pair exactly,
    while leaving promotion policy and durable serialization to callers.  A
    ``optimizer=None`` supports parameter-free structural programs without
    introducing dummy parameters or a gradient optimizer. A model may rebuild
    its registered modules from ``get_extra_state`` during
    ``load_state_dict``; the normal PyTorch state-dict contract therefore
    remains the topology extension point.
    """

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        cursor_capture: CursorCapture,
        cursor_restore: CursorRestore,
        torch_generators: Mapping[str, torch.Generator] | None = None,
    ) -> None:
        if not callable(cursor_capture) or not callable(cursor_restore):
            raise ValueError("cursor capture and restore callbacks are required")
        generators = dict(torch_generators or {})
        for name, generator in generators.items():
            _validate_name(name, label="torch generator name")
            if not isinstance(generator, torch.Generator):
                raise ValueError(f"named RNG {name!r} is not a torch.Generator")
        self.model = model
        self.optimizer = optimizer
        self.cursor_capture = cursor_capture
        self.cursor_restore = cursor_restore
        self.torch_generators = MappingProxyType(generators)
        if self.optimizer is not None:
            _optimizer_parameter_topology(self.model, self.optimizer)

    def _state_parts(self) -> dict[str, Any]:
        return {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict() if self.optimizer is not None
            else {"state": {}, "param_groups": []},
            "optimizer_defaults": self.optimizer.defaults if self.optimizer is not None else {},
            "optimizer_parameter_topology": _optimizer_parameter_topology(
                self.model, self.optimizer
            ) if self.optimizer is not None else (),
            "gradients": {
                name: parameter.grad for name, parameter in self.model.named_parameters()
            },
            "nonpersistent_buffers": _nonpersistent_buffers(self.model),
            "module_modes": _module_modes(self.model),
            "parameter_requires_grad": _parameter_flags(self.model),
            "rng": _capture_rng(self.torch_generators),
            "cursor": self.cursor_capture(),
        }

    def _current_state_fingerprints(self) -> tuple[str, str]:
        return _training_state_fingerprints(self._state_parts())

    def current_state_fingerprint(self) -> str:
        return self._current_state_fingerprints()[0]

    def _capture_snapshot(
        self, *, expected_fingerprint: str | None = None
    ) -> TrainingStateSnapshot:
        live_fingerprint = expected_fingerprint or self.current_state_fingerprint()
        live_parts = self._state_parts()
        live_manifest = _state_manifest(live_parts)
        model_state = _clone_state(live_parts["model_state"])
        optimizer_state = _clone_state(live_parts["optimizer_state"])
        optimizer_defaults = _clone_state(live_parts["optimizer_defaults"])
        optimizer_topology = live_parts["optimizer_parameter_topology"]
        gradients = _parameter_gradients(self.model)
        nonpersistent = {
            name: _clone_state(value) for name, value in _nonpersistent_buffers(self.model).items()
        }
        modes = _module_modes(self.model)
        flags = _parameter_flags(self.model)
        rng = _clone_state(_capture_rng(self.torch_generators))
        cursor = _clone_state(self.cursor_capture())
        inventory = {
            "model": _tensor_inventory(model_state),
            "optimizer": _tensor_inventory(optimizer_state),
            "gradients": _tensor_inventory(gradients),
            "nonpersistent_buffers": _tensor_inventory(nonpersistent),
        }
        summary = {
            "model_state_entries": len(model_state),
            "optimizer_parameter_groups": len(optimizer_state.get("param_groups", [])),
            "module_count": len(modes),
            "parameter_count": len(flags),
            "tensor_inventory": inventory,
            "snapshot_tensor_bytes": sum(row["bytes"] for row in inventory.values()),
            "rng_fingerprint": _state_digest(rng),
            "cursor_fingerprint": _state_digest(cursor),
        }
        post_fingerprint, model_state_fingerprint = self._current_state_fingerprints()
        if post_fingerprint != live_fingerprint:
            raise TrainingCheckpointError("state changed while its checkpoint was captured")
        return TrainingStateSnapshot(
            fingerprint=live_fingerprint,
            summary=MappingProxyType(summary),
            _state_manifest=live_manifest,
            _model_state_fingerprint=model_state_fingerprint,
            _model_state=model_state,
            _optimizer_state=optimizer_state,
            _optimizer_defaults=optimizer_defaults,
            _optimizer_topology=optimizer_topology,
            _gradients=gradients,
            _nonpersistent_buffers=nonpersistent,
            _module_modes=modes,
            _parameter_flags=flags,
            _rng=rng,
            _cursor=cursor,
        )

    def _verify_snapshot(self, snapshot: TrainingStateSnapshot) -> None:
        if not isinstance(snapshot, TrainingStateSnapshot):
            raise TypeError("restore expects a TrainingStateSnapshot")
        snapshot.verify()

    def restore(self, snapshot: TrainingStateSnapshot) -> str:
        """Restore a captured state after verifying its immutable manifest.

        Snapshot verification happens before the first live mutation.  This
        primitive is otherwise non-atomic if a later model, optimizer, cursor,
        or RNG restore step raises; callers that require atomic restore must
        capture a rollback state and wrap this method, as
        :class:`StructuralGrowthSession` does.
        """

        self._verify_snapshot(snapshot)
        return self._restore_snapshot(snapshot)

    def capture(self, *, expected_fingerprint: str | None = None) -> TrainingStateSnapshot:
        """Capture the complete mutable boundary as an opaque snapshot."""

        return self._capture_snapshot(expected_fingerprint=expected_fingerprint)

    def _restore_snapshot(self, snapshot: TrainingStateSnapshot) -> str:
        if (self.optimizer is None) != (not snapshot._optimizer_topology):
            raise TrainingCheckpointError("optimizer presence changed across the state boundary")
        self.model.load_state_dict(_clone_state(snapshot._model_state), strict=True)
        if self.optimizer is not None:
            _restore_optimizer(
                self.model,
                self.optimizer,
                state=snapshot._optimizer_state,
                defaults=snapshot._optimizer_defaults,
                topology=snapshot._optimizer_topology,
            )
        _restore_parameter_flags(self.model, snapshot._parameter_flags)
        _restore_parameter_gradients(self.model, snapshot._gradients)
        _restore_nonpersistent_buffers(self.model, snapshot._nonpersistent_buffers)
        _restore_module_modes(self.model, snapshot._module_modes)
        self.cursor_restore(_clone_state(snapshot._cursor))
        _restore_rng(snapshot._rng, self.torch_generators)
        restored = self.current_state_fingerprint()
        if restored != snapshot.fingerprint:
            expected_parts = {
                "model_state": snapshot._model_state,
                "optimizer_state": snapshot._optimizer_state,
                "optimizer_defaults": snapshot._optimizer_defaults,
                "optimizer_parameter_topology": snapshot._optimizer_topology,
                "gradients": snapshot._gradients,
                "nonpersistent_buffers": snapshot._nonpersistent_buffers,
                "module_modes": snapshot._module_modes,
                "parameter_requires_grad": snapshot._parameter_flags,
                "rng": snapshot._rng,
                "cursor": snapshot._cursor,
            }
            observed_parts = self._state_parts()
            differing = sorted(
                name
                for name in expected_parts
                if _state_digest(expected_parts[name]) != _state_digest(observed_parts[name])
            )
            detail = ", ".join(differing) if differing else "cross-component tensor alias topology"
            raise TrainingCheckpointError(
                "restored training state is not bit-exact with its checkpoint; "
                f"differing components: {detail}"
            )
        return restored


class DecisionReceipt:
    """Canonical immutable receipt; callers receive fresh mappings on read."""

    __slots__ = ("_canonical", "_fingerprint")

    def __init__(self, body: Mapping[str, Any]) -> None:
        normalized = _json_value(body, label="decision receipt")
        if normalized.get("schema") != DECISION_RECEIPT_SCHEMA:
            raise DecisionReceiptError("decision receipt schema changed")
        canonical_body = _canonical_json_bytes(normalized)
        fingerprint = hashlib.sha256(canonical_body).hexdigest()
        self._fingerprint = fingerprint
        self._canonical = _canonical_json_bytes({**normalized, "fingerprint": fingerprint})

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def canonical_bytes(self) -> bytes:
        return self._canonical

    def to_dict(self) -> dict[str, Any]:
        payload = json.loads(self._canonical)
        fingerprint = payload.pop("fingerprint")
        if fingerprint != _digest_json(payload):
            raise DecisionReceiptError("decision receipt fingerprint changed")
        return {**payload, "fingerprint": fingerprint}


class DecisionReceiptStore(Protocol):
    def put(self, receipt: DecisionReceipt) -> Any:
        """Persist one no-overwrite decision receipt."""


class ImmutableDirectoryReceiptStore:
    """Atomic, content-addressed, no-overwrite receipt publication."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise DecisionReceiptError("receipt root must be one real directory")

    def put(self, receipt: DecisionReceipt) -> Path:
        payload = receipt.to_dict()
        controller_id = _validate_name(payload["controller_id"], label="receipt controller_id")
        index = int(payload["decision_index"])
        destination = self.root / (f"{controller_id}-{index:06d}-{receipt.fingerprint}.json")
        encoded = receipt.canonical_bytes
        descriptor, temporary_name = tempfile.mkstemp(prefix=".saturn-decision-", dir=self.root)
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                if destination.read_bytes() != encoded:
                    raise DecisionReceiptError(
                        "immutable decision receipt path already contains different bytes"
                    ) from exc
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)
        if destination.read_bytes() != encoded:
            raise DecisionReceiptError("published decision receipt cannot be reopened")
        return destination


class DecisionLedger:
    """Append-only, fingerprint-chained in-process receipt custody."""

    def __init__(self, controller_id: str) -> None:
        self.controller_id = _validate_name(controller_id, label="controller_id")
        self._receipts: tuple[DecisionReceipt, ...] = ()

    @property
    def receipts(self) -> tuple[DecisionReceipt, ...]:
        return self._receipts

    @property
    def head_fingerprint(self) -> str | None:
        return self._receipts[-1].fingerprint if self._receipts else None

    def append(self, receipt: DecisionReceipt) -> None:
        payload = receipt.to_dict()
        if payload.get("controller_id") != self.controller_id:
            raise DecisionReceiptError("decision receipt belongs to another controller")
        if payload.get("decision_index") != len(self._receipts):
            raise DecisionReceiptError("decision receipt index is not append-only")
        if payload.get("previous_receipt_fingerprint") != self.head_fingerprint:
            raise DecisionReceiptError("decision receipt chain parent changed")
        self._receipts = (*self._receipts, receipt)

    def verify(self) -> bool:
        previous: str | None = None
        for index, receipt in enumerate(self._receipts):
            payload = receipt.to_dict()
            if (
                payload.get("controller_id") != self.controller_id
                or payload.get("decision_index") != index
                or payload.get("previous_receipt_fingerprint") != previous
            ):
                return False
            previous = receipt.fingerprint
        return True


class DeterministicCheckpointController(TrainingStateBoundary):
    """Own one accepted training state and transact bounded update intervals."""

    def __init__(
        self,
        *,
        controller_id: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        policy: PromotionPolicy,
        cursor_capture: CursorCapture,
        cursor_restore: CursorRestore,
        max_interval_steps: int,
        torch_generators: Mapping[str, torch.Generator] | None = None,
        receipt_store: DecisionReceiptStore | None = None,
    ) -> None:
        if isinstance(max_interval_steps, bool) or int(max_interval_steps) < 1:
            raise ValueError("max_interval_steps must be positive")
        self.controller_id = _validate_name(controller_id, label="controller_id")
        super().__init__(
            model=model,
            optimizer=optimizer,
            cursor_capture=cursor_capture,
            cursor_restore=cursor_restore,
            torch_generators=torch_generators,
        )
        self.policy = policy
        self.max_interval_steps = int(max_interval_steps)
        self.receipt_store = receipt_store
        self.ledger = DecisionLedger(self.controller_id)
        self._accepted_snapshot: TrainingStateSnapshot | None = None
        self._accepted_scores: EvaluationScores | None = None
        self._interval_ids: set[str] = set()

    @property
    def initialized(self) -> bool:
        return self._accepted_snapshot is not None

    @property
    def accepted_state_fingerprint(self) -> str | None:
        return self._accepted_snapshot.fingerprint if self._accepted_snapshot is not None else None

    @property
    def accepted_scores(self) -> EvaluationScores | None:
        return self._accepted_scores

    def with_verified_accepted_state(
        self,
        serializer: Callable[[AcceptedStateSerializationView], _SerializationResult],
    ) -> _SerializationResult:
        """Serialize from the accepted CPU snapshot after one live-state check.

        Controller operations are synchronous.  The callback receives an
        ephemeral borrow and must not retain or mutate it.  Serialization reads
        only the accepted snapshot after the live equality check, so mutable
        model/optimizer tensors are neither re-hashed nor copied a second time.
        """

        if not callable(serializer):
            raise ValueError("accepted-state serializer callback is required")
        snapshot = self._accepted_snapshot
        if snapshot is None:
            raise TrainingCheckpointError("checkpoint controller has no accepted state")
        if self.current_state_fingerprint() != snapshot.fingerprint:
            raise TrainingStateDriftError(
                "live training state does not match the accepted checkpoint"
            )
        runtime_state = MappingProxyType(
            {
                "optimizer_state": snapshot._optimizer_state,
                "optimizer_defaults": snapshot._optimizer_defaults,
                "optimizer_parameter_topology": snapshot._optimizer_topology,
                "gradients": snapshot._gradients,
                "nonpersistent_buffers": snapshot._nonpersistent_buffers,
                "module_modes": snapshot._module_modes,
                "parameter_requires_grad": snapshot._parameter_flags,
                "rng": snapshot._rng,
                "accepted_training_cursor": snapshot._cursor,
            }
        )
        return serializer(
            AcceptedStateSerializationView(
                accepted_state_fingerprint=snapshot.fingerprint,
                model_state_fingerprint=snapshot.model_state_fingerprint,
                runtime_state=runtime_state,
            )
        )

    def _evaluate_isolated(
        self,
        evaluator: Evaluator,
        *,
        expected_fingerprint: str | None = None,
    ) -> EvaluationScores:
        before = expected_fingerprint or self.current_state_fingerprint()
        rng = _clone_state(_capture_rng(self.torch_generators))
        cursor = _clone_state(self.cursor_capture())
        modes = _module_modes(self.model)
        try:
            scores = EvaluationScores.from_value(evaluator())
        finally:
            _restore_module_modes(self.model, modes)
            self.cursor_restore(_clone_state(cursor))
            _restore_rng(rng, self.torch_generators)
        after = self.current_state_fingerprint()
        if after != before:
            raise EvaluationMutationError(
                "checkpoint evaluator mutated model, optimizer, gradients, or topology"
            )
        return scores

    def _record(self, body: Mapping[str, Any]) -> DecisionReceipt:
        receipt = DecisionReceipt(
            {
                "schema": DECISION_RECEIPT_SCHEMA,
                "controller_id": self.controller_id,
                "decision_index": len(self.ledger.receipts),
                "previous_receipt_fingerprint": self.ledger.head_fingerprint,
                **dict(body),
            }
        )
        if self.receipt_store is not None:
            self.receipt_store.put(receipt)
        self.ledger.append(receipt)
        return receipt

    def establish_baseline(self, evaluator: Evaluator) -> DecisionReceipt:
        if self.initialized:
            raise TrainingCheckpointError("checkpoint controller already has a baseline")
        snapshot = self._capture_snapshot()
        try:
            scores = self._evaluate_isolated(evaluator, expected_fingerprint=snapshot.fingerprint)
        except Exception:
            self._restore_snapshot(snapshot)
            raise
        receipt = self._record(
            {
                "kind": "baseline",
                "interval_id": "baseline",
                "bounded_steps": 0,
                "policy": self.policy.to_dict(),
                "baseline": None,
                "candidate": {
                    "state": snapshot.to_dict(),
                    "evaluation": scores.to_dict(),
                },
                "comparison": None,
                "decision": {
                    "promoted": True,
                    "action": "establish-baseline",
                    "reason": "initial-state",
                },
                "rollback": {
                    "required": False,
                    "verified_exact": True,
                },
                "post_state_fingerprint": snapshot.fingerprint,
                "train_metadata": {},
            }
        )
        self._accepted_snapshot = snapshot
        self._accepted_scores = scores
        return receipt

    def _error_receipt(
        self,
        *,
        interval_id: str,
        steps: int,
        baseline: TrainingStateSnapshot,
        baseline_scores: EvaluationScores,
        error: Exception,
        candidate_fingerprint: str | None,
        train_metadata: Mapping[str, Any],
    ) -> DecisionReceipt:
        restored = self._restore_snapshot(baseline)
        return self._record(
            {
                "kind": "interval-error",
                "interval_id": interval_id,
                "bounded_steps": steps,
                "policy": self.policy.to_dict(),
                "baseline": {
                    "state": baseline.to_dict(),
                    "evaluation": baseline_scores.to_dict(),
                },
                "candidate": {
                    "state_fingerprint_before_restore": candidate_fingerprint,
                    "evaluation": None,
                },
                "comparison": None,
                "decision": {
                    "promoted": False,
                    "action": "restore-prior-after-error",
                    "reason": "interval-error",
                    "error_type": type(error).__qualname__,
                    "error_message": str(error),
                },
                "rollback": {
                    "required": True,
                    "verified_exact": restored == baseline.fingerprint,
                },
                "post_state_fingerprint": restored,
                "train_metadata": dict(train_metadata),
            }
        )

    def run_interval(
        self,
        *,
        interval_id: str,
        steps: int,
        train_step: TrainStep,
        evaluator: Evaluator,
    ) -> DecisionReceipt:
        if not self.initialized:
            raise TrainingCheckpointError("establish a baseline before training")
        resolved_interval = _validate_name(interval_id, label="interval_id")
        if resolved_interval == "baseline" or resolved_interval in self._interval_ids:
            raise ValueError("interval_id must be unique and cannot be 'baseline'")
        if isinstance(steps, bool) or not 1 <= int(steps) <= self.max_interval_steps:
            raise ValueError(f"steps must be between 1 and {self.max_interval_steps}")
        if not callable(train_step) or not callable(evaluator):
            raise ValueError("train_step and evaluator callbacks are required")
        baseline = self._accepted_snapshot
        baseline_scores = self._accepted_scores
        assert baseline is not None and baseline_scores is not None
        observed = self.current_state_fingerprint()
        if observed != baseline.fingerprint:
            self._restore_snapshot(baseline)
            raise TrainingStateDriftError(
                "live training state drifted from the accepted checkpoint; prior restored"
            )
        self._interval_ids.add(resolved_interval)
        train_metadata: dict[str, Any] = {}
        candidate_fingerprint: str | None = None
        try:
            step_receipts = []
            for step_index in range(int(steps)):
                raw_metadata = train_step(step_index)
                if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
                    raise ValueError("train_step callback metadata must be a mapping or None")
                step_receipts.append(
                    _json_value(raw_metadata or {}, label=f"train step {step_index}")
                )
            train_metadata = {
                "step_callback_invocations": int(steps),
                "step_receipts": step_receipts,
            }
            candidate_fingerprint = self.current_state_fingerprint()
            candidate_scores = self._evaluate_isolated(
                evaluator, expected_fingerprint=candidate_fingerprint
            )
            comparison = self.policy.compare(baseline_scores, candidate_scores)
        except Exception as exc:
            try:
                if candidate_fingerprint is None:
                    candidate_fingerprint = self.current_state_fingerprint()
            except Exception:
                candidate_fingerprint = None
            receipt = self._error_receipt(
                interval_id=resolved_interval,
                steps=int(steps),
                baseline=baseline,
                baseline_scores=baseline_scores,
                error=exc,
                candidate_fingerprint=candidate_fingerprint,
                train_metadata=train_metadata,
            )
            raise TrainingIntervalError(
                f"training interval {resolved_interval!r} failed and was restored",
                receipt=receipt,
            ) from exc
        promoted = bool(comparison["promoted"])
        candidate_snapshot: TrainingStateSnapshot | None = None
        if promoted:
            try:
                candidate_snapshot = self._capture_snapshot(
                    expected_fingerprint=candidate_fingerprint
                )
                if candidate_snapshot.fingerprint != candidate_fingerprint:
                    raise TrainingCheckpointError(
                        "candidate changed between evaluation and checkpoint capture"
                    )
            except Exception as exc:
                receipt = self._error_receipt(
                    interval_id=resolved_interval,
                    steps=int(steps),
                    baseline=baseline,
                    baseline_scores=baseline_scores,
                    error=exc,
                    candidate_fingerprint=candidate_fingerprint,
                    train_metadata=train_metadata,
                )
                raise TrainingIntervalError(
                    f"training interval {resolved_interval!r} failed and was restored",
                    receipt=receipt,
                ) from exc
            post_fingerprint = candidate_snapshot.fingerprint
        else:
            post_fingerprint = self._restore_snapshot(baseline)
        receipt_body = {
            "kind": "interval-decision",
            "interval_id": resolved_interval,
            "bounded_steps": int(steps),
            "policy": self.policy.to_dict(),
            "baseline": {
                "state": baseline.to_dict(),
                "evaluation": baseline_scores.to_dict(),
            },
            "candidate": {
                "state_fingerprint": candidate_fingerprint,
                "checkpoint": (
                    candidate_snapshot.to_dict() if candidate_snapshot is not None else None
                ),
                "evaluation": candidate_scores.to_dict(),
            },
            "comparison": comparison,
            "decision": {
                "promoted": promoted,
                "action": "promote-candidate" if promoted else "restore-prior",
                "reason": comparison["relation"],
            },
            "rollback": {
                "required": not promoted,
                "verified_exact": (
                    post_fingerprint
                    == (candidate_fingerprint if promoted else baseline.fingerprint)
                ),
            },
            "post_state_fingerprint": post_fingerprint,
            "train_metadata": train_metadata,
        }
        try:
            receipt = self._record(receipt_body)
        except Exception:
            if promoted:
                self._restore_snapshot(baseline)
            raise
        if promoted:
            assert candidate_snapshot is not None
            self._accepted_snapshot = candidate_snapshot
            self._accepted_scores = candidate_scores
        return receipt


__all__ = [
    "AcceptedStateSerializationView",
    "DECISION_RECEIPT_SCHEMA",
    "EVALUATION_SCORES_SCHEMA",
    "PROMOTION_POLICY_SCHEMA",
    "TRAINING_CHECKPOINT_SCHEMA",
    "DecisionLedger",
    "DecisionReceipt",
    "DecisionReceiptError",
    "DeterministicCheckpointController",
    "EvaluationMutationError",
    "EvaluationScores",
    "ImmutableDirectoryReceiptStore",
    "PromotionObjective",
    "PromotionPolicy",
    "TrainingStateBoundary",
    "TrainingCheckpointError",
    "TrainingIntervalError",
    "TrainingStateDriftError",
    "TrainingStateSnapshot",
]
