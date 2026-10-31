"""PAN neural-network graph runtime.

This module implements the first PAN-LLM substrate: a typed model graph,
runtime-visible tensor values, region formation, PS oracle execution, and
fabric-style placement traces. Real PL kernels for matmul/attention are not
hidden here; explicit fabric-session execution fails closed unless a region can
lower to an existing fabric graph backend.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .fabric_executor import FabricGraphExecutor, FabricRuntimeSession, _PSOpEngine
from .fabric_foundry import FabricFoundryCatalog, FabricRegion, generate_candidates, scheduler_hint_for_graph
from .fabric_graph import FabricBufferSpec, FabricGraph, FabricGraphOp, FabricGraphPlanner, FabricPolicyHead
from .fabric_runtime import DEFAULT_FABRIC_WORKDIR, FabricRegistry, active_xmutil_app


PAN_NN_GRAPH_SCHEMA = "pan_nn_graph_v1"
PAN_NN_EXECUTION_SCHEMA = "pan_nn_execution_v1"
PAN_NN_VALUE_SCHEMA = "pan_nn_tensor_value_v1"
PAN_NN_TRACE_SCHEMA = "pan_nn_trace_v1"
PAN_NN_TRAINING_SCHEMA = "pan_nn_training_record_v1"
DEFAULT_NN_VALIDATION_RTOL = 1.0e-4
DEFAULT_NN_VALIDATION_ATOL = 1.0e-5

NN_RESIDENCIES = frozenset({"ps", "daemon", "xrt_bo", "dma_buf", "catalog_core", "unknown"})
NN_DTYPE_ALIASES = {
    "float": "float32",
    "fp32": "float32",
    "fp16": "float16",
    "half": "float16",
    "single": "float32",
    "double": "float64",
    "int": "int32",
    "uint": "uint32",
    "spf32": "uint32",
    "spf32_word": "uint32",
}
SUPPORTED_NN_DTYPES = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "bool",
    }
)
PS_SUPPORTED_OPS = frozenset(
    {
        "matmul",
        "einsum",
        "gemm",
        "conv2d",
        "depthwise_conv2d",
        "batchnorm",
        "identity",
        "dropout",
        "sum",
        "mean",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "pow",
        "mod",
        "min",
        "max",
        "clip",
        "floor",
        "ceil",
        "exp",
        "log",
        "reciprocal",
        "sqrt",
        "reduce_mean",
        "reduce_sum",
        "reduce_min",
        "reduce_max",
        "reduce_prod",
        "concat",
        "relu",
        "gelu",
        "bias_gelu",
        "quick_gelu",
        "silu",
        "sigmoid",
        "tanh",
        "erf",
        "sin",
        "cos",
        "layernorm",
        "rmsnorm",
        "rotary_embedding",
        "softmax",
        "log_softmax",
        "argmax",
        "argmin",
        "topk",
        "top_p_filter",
        "kv_cache_append",
        "quantize_linear",
        "dequantize_linear",
        "qlinear_matmul",
        "matmul_integer",
        "dynamic_quantize_linear",
        "cast",
        "cast_like",
        "size",
        "shape",
        "range",
        "constant_of_shape",
        "cumsum",
        "pad",
        "flatten",
        "unsqueeze",
        "squeeze",
        "expand",
        "tile",
        "trilu",
        "split",
        "gather",
        "gather_elements",
        "gather_nd",
        "scatter_elements",
        "scatter_nd",
        "slice",
        "where",
        "equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "not",
        "and",
        "or",
        "reshape",
        "transpose",
        "attention_qkv",
        "attention_score",
        "attention_value",
        "spf32.mul",
        "spf32.add",
        "spf32.div",
        "spf32.abs2",
        "spf32.fft64",
        "mlp.gemm_k16",
    }
)
FLOAT_ONLY_NN_OPS = frozenset(
    {
        "relu",
        "gelu",
        "bias_gelu",
        "quick_gelu",
        "silu",
        "sigmoid",
        "tanh",
        "erf",
        "sin",
        "cos",
        "floor",
        "ceil",
        "layernorm",
        "rmsnorm",
        "rotary_embedding",
        "exp",
        "log",
        "reciprocal",
        "sqrt",
        "reduce_mean",
        "softmax",
        "log_softmax",
    }
)
FABRIC_SIM_ELIGIBLE_OPS = frozenset(
    {
        "matmul",
        "einsum",
        "gemm",
        "conv2d",
        "depthwise_conv2d",
        "batchnorm",
        "identity",
        "dropout",
        "sum",
        "mean",
        "add",
        "sub",
        "mul",
        "div",
        "neg",
        "abs",
        "pow",
        "mod",
        "min",
        "max",
        "clip",
        "floor",
        "ceil",
        "exp",
        "log",
        "reciprocal",
        "sqrt",
        "reduce_mean",
        "reduce_sum",
        "reduce_min",
        "reduce_max",
        "reduce_prod",
        "relu",
        "gelu",
        "bias_gelu",
        "quick_gelu",
        "silu",
        "sigmoid",
        "tanh",
        "erf",
        "sin",
        "cos",
        "layernorm",
        "rmsnorm",
        "rotary_embedding",
        "softmax",
        "log_softmax",
        "argmax",
        "argmin",
        "topk",
        "top_p_filter",
        "kv_cache_append",
        "quantize_linear",
        "dequantize_linear",
        "qlinear_matmul",
        "matmul_integer",
        "dynamic_quantize_linear",
        "where",
        "scatter_elements",
        "scatter_nd",
        "cumsum",
        "trilu",
        "pad",
        "equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "not",
        "and",
        "or",
        "attention_qkv",
        "attention_score",
        "attention_value",
        "spf32.mul",
        "spf32.add",
        "spf32.div",
        "spf32.abs2",
        "spf32.fft64",
    }
)
FABRIC_SESSION_ELIGIBLE_OPS = frozenset({"spf32.mul", "spf32.add", "spf32.div", "spf32.abs2", "spf32.fft64", "mlp.gemm_k16"})
SPF32_BINARY_NN_OPS = frozenset({"spf32.mul", "spf32.add", "spf32.div"})
SPF32_UNARY_NN_OPS = frozenset({"spf32.abs2", "spf32.fft64"})
MLP_FABRIC_OPS = frozenset({"mlp.gemm_k16"})
FABRIC_BINARY_NN_OPS = SPF32_BINARY_NN_OPS | MLP_FABRIC_OPS
QUANTIZED_LINEAR_OPS = frozenset({"matmul", "gemm", "attention_qkv", "attention_score", "attention_value"})
BROADCAST_OUTPUT_NN_OPS = frozenset(
    {
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "mod",
        "sum",
        "mean",
        "min",
        "max",
        "bias_gelu",
        "equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "and",
        "or",
        "where",
    }
)
BOOL_OUTPUT_NN_OPS = frozenset({"equal", "less", "less_equal", "greater", "greater_equal", "not", "and", "or"})
VERIFIER_OUTPUT_INFER_OPS = frozenset(
    {
        *BROADCAST_OUTPUT_NN_OPS,
        "attention_score",
        "attention_value",
        "batchnorm",
        "concat",
        "conv2d",
        "depthwise_conv2d",
        "expand",
        "flatten",
        "gather",
        "reshape",
        "slice",
        "split",
        "squeeze",
        "not",
        "transpose",
        "unsqueeze",
    }
)
LLM_FOUNDRY_SEED_OPS = frozenset(
    {
        "rmsnorm",
        "attention_qkv",
        "rotary_embedding",
        "attention_score",
        "attention_value",
        "kv_cache_append",
        "top_p_filter",
        "topk",
    }
)
SIDE_EFFECT_ATTRS = frozenset(
    {
        "has_side_effects",
        "inplace",
        "mutates_state",
        "side_effect",
        "side_effects",
        "stateful",
        "writes_global",
    }
)
STATEFUL_NN_OPS = frozenset(
    {
        "bernoulli",
        "multinomial",
        "randomnormal",
        "randomnormal_like",
        "randomuniform",
        "randomuniform_like",
    }
)
_NN_ACTIVE_APP_CACHE: dict[str, Optional[str]] = {}


def _resolve_nn_active_app(active_app: str) -> str:
    if active_app != "auto":
        return str(active_app)
    cached = _NN_ACTIVE_APP_CACHE.get(active_app)
    if cached is None and active_app not in _NN_ACTIVE_APP_CACHE:
        cached = active_xmutil_app()
        _NN_ACTIVE_APP_CACHE[active_app] = cached
    return str(cached or "unknown")


@dataclass(frozen=True)
class NNTensorSpec:
    name: str
    dtype: str
    shape: tuple[int | str, ...]
    initializer: bool = False
    data: Optional[Any] = None
    quantization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, name: str, payload: Any, *, initializer: bool = False) -> "NNTensorSpec":
        if isinstance(payload, list):
            return cls(name=name, dtype="float32", shape=(len(payload),), initializer=initializer, data=payload)
        if not isinstance(payload, dict):
            raise ValueError(f"tensor {name!r} must be an object or data list")
        shape = tuple(payload.get("shape", ()))
        return cls(
            name=str(payload.get("name", name)),
            dtype=str(payload.get("dtype", "float32")),
            shape=shape,
            initializer=bool(payload.get("initializer", initializer)),
            data=payload.get("data"),
            quantization=dict(payload.get("quantization", {})),
        )

    def to_json(self, *, include_data: bool = False) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "shape_constraints": _shape_constraints_for_shape(self.shape),
            "initializer": self.initializer,
            "quantization": dict(self.quantization),
        }
        if include_data and self.data is not None:
            payload["data"] = self.data
        return payload


@dataclass(frozen=True)
class NNNode:
    node_id: str
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any], index: int) -> "NNNode":
        op = str(payload.get("op", payload.get("op_type", ""))).lower()
        if not op:
            raise ValueError(f"node {index} is missing op")
        return cls(
            node_id=str(payload.get("id", payload.get("name", f"node{index}"))),
            op=op,
            inputs=tuple(str(name) for name in payload.get("inputs", [])),
            outputs=tuple(str(name) for name in payload.get("outputs", [])),
            attrs=dict(payload.get("attrs", payload.get("attributes", {}))),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.node_id,
            "op": self.op,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
        }


@dataclass(frozen=True)
class NNRegion:
    region_id: str
    nodes: tuple[NNNode, ...]
    reason: str

    @property
    def ops(self) -> tuple[str, ...]:
        return tuple(node.op for node in self.nodes)

    @property
    def outputs(self) -> tuple[str, ...]:
        if not self.nodes:
            return ()
        return self.nodes[-1].outputs

    def signature(self) -> str:
        return "+".join(self.ops)

    def to_json(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "ops": list(self.ops),
            "signature": self.signature(),
            "reason": self.reason,
            "nodes": [node.to_json() for node in self.nodes],
        }


@dataclass(frozen=True)
class NNGraph:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    tensors: dict[str, NNTensorSpec]
    nodes: tuple[NNNode, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, payload: dict[str, Any], *, source_name: str = "model") -> "NNGraph":
        if payload.get("schema") not in {None, PAN_NN_GRAPH_SCHEMA}:
            raise ValueError(f"unsupported NN graph schema {payload.get('schema')!r}")
        tensors: dict[str, NNTensorSpec] = {}
        for item in payload.get("inputs", []):
            spec = NNTensorSpec.from_json(str(item.get("name")), item)
            tensors[spec.name] = spec
        for name, item in dict(payload.get("initializers", {})).items():
            spec = NNTensorSpec.from_json(str(name), item, initializer=True)
            tensors[spec.name] = spec
        for item in payload.get("tensors", []):
            spec = NNTensorSpec.from_json(str(item.get("name")), item)
            tensors[spec.name] = spec
        nodes = tuple(NNNode.from_json(node, index) for index, node in enumerate(payload.get("nodes", [])))
        if not nodes:
            raise ValueError("NN graph must contain at least one node")
        inputs = tuple(str(name) for name in payload.get("graph_inputs", payload.get("input_names", [])))
        if not inputs:
            inputs = tuple(spec.name for spec in tensors.values() if not spec.initializer and spec.name not in _produced_names(nodes))
        outputs = tuple(str(name) for name in payload.get("outputs", []))
        if not outputs:
            outputs = tuple(nodes[-1].outputs)
        return cls(
            name=str(payload.get("name", source_name)),
            inputs=inputs,
            outputs=outputs,
            tensors=tensors,
            nodes=nodes,
            metadata=dict(payload.get("metadata", {})),
        )

    def regions(self) -> tuple[NNRegion, ...]:
        return form_nn_regions(self)

    def to_json(self, *, include_initializer_data: bool = False) -> dict[str, Any]:
        return {
            "schema": PAN_NN_GRAPH_SCHEMA,
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "tensors": {
                name: spec.to_json(include_data=include_initializer_data)
                for name, spec in sorted(self.tensors.items())
            },
            "nodes": [node.to_json() for node in self.nodes],
            "metadata": dict(self.metadata),
        }


@dataclass
class NNTensorValue:
    value_id: str
    name: str
    dtype: str
    shape: tuple[int, ...]
    residency: str
    materialized: bool
    producer: Optional[str] = None
    validation: Optional[dict[str, Any]] = None
    materialization_reason: Optional[str] = None
    daemon_handle: Optional[dict[str, Any]] = None
    ps_digest: Optional[dict[str, Any]] = None
    refcount: int = 1
    array: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def elements(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return int(total)

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema": PAN_NN_VALUE_SCHEMA,
            "value_id": self.value_id,
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "elements": self.elements,
            "residency": self.residency,
            "materialized": bool(self.materialized),
            "producer": self.producer,
            "validation": self.validation,
            "materialization_reason": self.materialization_reason,
            "daemon_handle": self.daemon_handle,
            "refcount": int(self.refcount),
            "metadata": dict(self.metadata),
        }
        digest = self.ps_digest
        if digest is None and self.array is not None and self.materialized:
            digest = tensor_digest(self.array)
        if digest is not None:
            payload["ps_digest"] = digest
        return payload


class NNTensorValueStore:
    def __init__(self, *, session_id: Optional[str] = None):
        self.session_id = session_id or f"nn-session-{time.time_ns()}"
        self._values: dict[str, NNTensorValue] = {}
        self._latest_by_name: dict[str, str] = {}
        self._counter = 0
        self.materialization_count = 0
        self.release_count = 0

    def add_array(
        self,
        name: str,
        array: Any,
        *,
        dtype: Optional[str] = None,
        residency: str = "ps",
        materialized: bool = True,
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
        materialization_reason: Optional[str] = None,
        daemon_handle: Optional[dict[str, Any]] = None,
        ps_digest: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> NNTensorValue:
        np = _numpy()
        arr = np.asarray(array, dtype=_numpy_dtype(dtype or str(np.asarray(array).dtype)))
        if residency not in NN_RESIDENCIES:
            raise ValueError(f"unsupported NN tensor residency {residency!r}")
        _validate_nn_value_residency(
            name,
            residency=residency,
            daemon_handle=daemon_handle,
            shape=arr.shape,
            materialized=bool(
                materialized
                or (metadata or {}).get("retained")
                or (metadata or {}).get("selected_action") == "run_daemon_resident_now"
            ),
        )
        if materialization_reason:
            self.materialization_count += 1
        value = NNTensorValue(
            value_id=self._next_id(name),
            name=str(name),
            dtype=dtype or str(arr.dtype),
            shape=tuple(int(dim) for dim in arr.shape),
            residency=residency,
            materialized=bool(materialized),
            producer=producer,
            validation=validation,
            materialization_reason=materialization_reason,
            daemon_handle=None if daemon_handle is None else dict(daemon_handle),
            ps_digest=ps_digest,
            array=arr,
            metadata=dict(metadata or {}),
        )
        self._values[value.value_id] = value
        self._latest_by_name[value.name] = value.value_id
        return value

    def add_daemon(
        self,
        name: str,
        handle: dict[str, Any],
        *,
        dtype: str = "uint32",
        shape: Optional[Sequence[int]] = None,
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> NNTensorValue:
        _validate_daemon_handle_descriptor(str(name), handle, shape=shape)
        elements = int(handle.get("elements") or 0)
        if shape is None:
            shape = (elements,)
        value = NNTensorValue(
            value_id=self._next_id(name),
            name=str(name),
            dtype=str(dtype or handle.get("dtype") or "uint32"),
            shape=tuple(int(dim) for dim in shape),
            residency="daemon",
            materialized=False,
            producer=producer,
            validation=validation,
            daemon_handle=dict(handle),
            array=None,
            metadata=dict(metadata or {}),
        )
        self._values[value.value_id] = value
        self._latest_by_name[value.name] = value.value_id
        return value

    def add_digest_value(
        self,
        name: str,
        digest: dict[str, Any],
        *,
        dtype: str,
        shape: Sequence[int],
        residency: str = "ps",
        materialized: bool = True,
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
        materialization_reason: Optional[str] = None,
        daemon_handle: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> NNTensorValue:
        if residency not in NN_RESIDENCIES:
            raise ValueError(f"unsupported NN tensor residency {residency!r}")
        _validate_nn_value_residency(
            name,
            residency=residency,
            daemon_handle=daemon_handle,
            shape=shape,
            materialized=bool(materialized),
        )
        if materialization_reason:
            self.materialization_count += 1
        value = NNTensorValue(
            value_id=self._next_id(name),
            name=str(name),
            dtype=str(dtype),
            shape=tuple(int(dim) for dim in shape),
            residency=residency,
            materialized=bool(materialized),
            producer=producer,
            validation=validation,
            materialization_reason=materialization_reason,
            daemon_handle=None if daemon_handle is None else dict(daemon_handle),
            ps_digest=dict(digest),
            array=None,
            metadata=dict(metadata or {}),
        )
        self._values[value.value_id] = value
        self._latest_by_name[value.name] = value.value_id
        return value

    def get_latest(self, name: str) -> NNTensorValue:
        value_id = self._latest_by_name[str(name)]
        return self._values[value_id]

    def release(self, value_id: str) -> None:
        if value_id in self._values:
            value = self._values[value_id]
            self.release_count += 1
            del self._values[value_id]
            if self._latest_by_name.get(value.name) == value_id:
                del self._latest_by_name[value.name]

    def values(self) -> tuple[NNTensorValue, ...]:
        return tuple(self._values.values())

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "pan_nn_value_store_v1",
            "session_id": self.session_id,
            "value_count": len(self._values),
            "materialization_count": self.materialization_count,
            "release_count": self.release_count,
            "latest_by_name": dict(sorted(self._latest_by_name.items())),
            "values": {
                value_id: value.to_json()
                for value_id, value in sorted(self._values.items())
            },
        }

    def _next_id(self, name: str) -> str:
        self._counter += 1
        return f"{self.session_id}:{name}:{self._counter}"


def _validate_nn_value_residency(
    name: str,
    *,
    residency: str,
    daemon_handle: Optional[dict[str, Any]],
    shape: Optional[Sequence[int]],
    materialized: bool = False,
) -> None:
    if residency not in NN_RESIDENCIES:
        raise ValueError(f"unsupported NN tensor residency {residency!r}")
    if residency == "daemon":
        if daemon_handle is None:
            if materialized:
                return
            raise ValueError(f"NN tensor {name!r} has daemon residency without daemon handle")
        _validate_daemon_handle_descriptor(name, daemon_handle, shape=shape)
        return
    if daemon_handle is not None:
        raise ValueError(f"NN tensor {name!r} has non-daemon residency with daemon handle")


def _validate_daemon_handle_descriptor(
    name: str,
    handle: dict[str, Any],
    *,
    shape: Optional[Sequence[int]],
) -> None:
    if not isinstance(handle, dict) or handle.get("schema") != "pan_daemon_bo_v1":
        raise ValueError(f"NN tensor {name!r} daemon handle must use pan_daemon_bo_v1 schema")
    try:
        handle_id = int(handle.get("handle"))
        elements = int(handle.get("elements"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"NN tensor {name!r} daemon handle has invalid handle/elements") from exc
    if handle_id <= 0:
        raise ValueError(f"NN tensor {name!r} daemon handle id must be positive")
    if elements <= 0:
        raise ValueError(f"NN tensor {name!r} daemon handle elements must be positive")
    if shape is not None:
        expected = _shape_element_count(tuple(int(dim) for dim in shape))
        if expected != elements:
            raise ValueError(f"NN tensor {name!r} daemon handle elements mismatch: expected {expected}, got {elements}")
    nbytes = handle.get("nbytes")
    if nbytes is not None:
        try:
            parsed_nbytes = int(nbytes)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"NN tensor {name!r} daemon handle nbytes is invalid") from exc
        if parsed_nbytes < elements * 4:
            raise ValueError(f"NN tensor {name!r} daemon handle nbytes too small")


@dataclass(frozen=True)
class NNExecutionResult:
    ok: bool
    graph: NNGraph
    backend: str
    outputs: dict[str, dict[str, Any]]
    trace: tuple[dict[str, Any], ...]
    training_records: tuple[dict[str, Any], ...]
    value_store: dict[str, Any]
    elapsed_seconds: float
    validation: Optional[dict[str, Any]] = None
    error: Optional[str] = None

    def summary(self) -> dict[str, Any]:
        fabric_summaries = [
            dict((row.get("fabric_execution") or {}).get("summary") or {})
            for row in self.trace
        ]
        fabric_daemon_rows = [
            row for row in self.trace if row.get("selected_action") == "run_daemon_resident_now"
        ]
        fabric_daemon_elapsed = sum(
            float(summary.get("timed_daemon_elapsed_seconds") or 0.0)
            for summary in fabric_summaries
        )
        fabric_wall_elapsed = sum(float(row.get("elapsed_seconds") or 0.0) for row in fabric_daemon_rows)
        return {
            "region_count": len(self.trace),
            "run_ps_count": sum(1 for row in self.trace if row.get("selected_action") == "run_ps_now"),
            "run_daemon_resident_count": sum(
                1 for row in self.trace if row.get("selected_action") == "run_daemon_resident_now"
            ),
            "run_catalog_core_count": sum(
                1 for row in self.trace if row.get("selected_action") == "run_catalog_core_now"
            ),
            "compile_better_count": sum(
                1 for row in self.trace if row.get("selected_action") == "compile_better_for_next_time"
            ),
            "foundry_queued_count": sum(
                1 for row in self.trace if (row.get("foundry") or {}).get("queued") is not None
            ),
            "reject_fail_closed_count": sum(
                1 for row in self.trace if row.get("selected_action") == "reject_fail_closed"
            ),
            "timed_transfer_bytes": sum(int(row.get("timed_transfer_bytes") or 0) for row in self.trace),
            "setup_transfer_bytes": sum(int(row.get("setup_transfer_bytes") or 0) for row in self.trace),
            "forced_materialization_bytes": sum(
                int(row.get("forced_materialization_bytes") or 0) for row in self.trace
            ),
            "fabric_timed_elapsed_seconds": sum(
                float(summary.get("timed_elapsed_seconds") or 0.0) for summary in fabric_summaries
            ),
            "fabric_daemon_elapsed_seconds": fabric_daemon_elapsed,
            "fabric_planning_elapsed_seconds": sum(
                float(summary.get("planning_elapsed_seconds") or 0.0) for summary in fabric_summaries
            ),
            "fabric_plan_cache_hit_count": sum(
                int(summary.get("plan_cache_hit_count") or 0) for summary in fabric_summaries
            ),
            "fabric_plan_cache_miss_count": sum(
                int(summary.get("plan_cache_miss_count") or 0) for summary in fabric_summaries
            ),
            "fabric_control_overhead_seconds": max(0.0, fabric_wall_elapsed - fabric_daemon_elapsed),
        }

    def to_json(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload = {
            "schema": PAN_NN_EXECUTION_SCHEMA,
            "ok": bool(self.ok),
            "backend": self.backend,
            "graph": self.graph.to_json(),
            "outputs": self.outputs,
            "summary": self.summary(),
            "value_store": self.value_store,
            "validation": self.validation,
            "elapsed_seconds": self.elapsed_seconds,
            "error": self.error,
        }
        if include_trace:
            payload["trace"] = list(self.trace)
            payload["training_records"] = list(self.training_records)
        return payload


@dataclass(frozen=True)
class _NNFabricPlanCacheEntry:
    fabric_graph: FabricGraph
    plan: Any
    output_shapes: dict[str, tuple[int, ...]]


class NNRuntimeSession:
    """Long-lived PAN NN execution session with retained initializer values."""

    def __init__(
        self,
        graph: NNGraph,
        *,
        resident_weights: bool = False,
        resident_inputs: bool = False,
        seed: int = 1,
        session_id: Optional[str] = None,
    ):
        self.graph = graph
        self.seed = int(seed)
        self.resident_weights = bool(resident_weights)
        self.resident_inputs = bool(resident_inputs)
        self.value_store = NNTensorValueStore(session_id=session_id)
        self.retained_values = _prepare_initializer_values(
            graph,
            self.value_store,
            resident_weights=self.resident_weights,
        )
        self.retained_input_values: dict[str, NNTensorValue] = {}
        self.retained_feedback_values: dict[str, NNTensorValue] = {}
        self._retained_input_signature: Optional[str] = None
        self._fabric_sessions: dict[str, FabricRuntimeSession] = {}
        self._fabric_plan_cache: dict[str, "_NNFabricPlanCacheEntry"] = {}
        self.run_count = 0

    def __enter__(self) -> "NNRuntimeSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def close(self) -> None:
        released: set[str] = set()
        for value in [
            *self.value_store.values(),
            *self.retained_values.values(),
            *self.retained_input_values.values(),
            *self.retained_feedback_values.values(),
        ]:
            self._release_fabric_runtime_for_value(value, released=released)
        for session in list(self._fabric_sessions.values()):
            session.__exit__(None, None, None)
        self._fabric_sessions.clear()
        self._fabric_plan_cache.clear()

    @classmethod
    def from_model(
        cls,
        model_path: str | Path,
        *,
        resident_weights: bool = False,
        resident_inputs: bool = False,
        seed: int = 1,
        session_id: Optional[str] = None,
    ) -> "NNRuntimeSession":
        return cls(
            load_nn_model(model_path),
            resident_weights=resident_weights,
            resident_inputs=resident_inputs,
            seed=seed,
            session_id=session_id,
        )

    def run(
        self,
        *,
        backend: str = "auto",
        inputs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = None,
        no_output_digest: bool = False,
        validate_output: bool = False,
        validation_rtol: Optional[float] = None,
        validation_atol: Optional[float] = None,
        validation_sample_count: Optional[int] = None,
        materialize_reason: Optional[str] = None,
        foundry_auto_queue: bool = False,
        foundry_catalog: Optional[str] = None,
        foundry_action_ledger_path: Optional[str] = None,
        foundry_target: str = "kv260",
        fabric_daemon: str = "/tmp/pan-fabricd.sock",
        workdir: Optional[str] = None,
        xclbin_dirs: Optional[Sequence[str]] = None,
        active_app: str = "auto",
        fabric_backend: str = "pl",
        daemon_buffer_storage: str = "inline",
        policy: Optional[FabricPolicyHead] = None,
        policy_materialization: bool = False,
        feedback_map: Optional[dict[str, str]] = None,
    ) -> NNExecutionResult:
        run_seed = self.seed if seed is None else int(seed)
        self._release_replaced_boundary_values()
        values = dict(self.retained_values)
        values.update(
            self._input_values_for_run(
                inputs=dict(inputs or {}),
                seed=run_seed,
                feedback_map=dict(feedback_map or {}),
            )
        )
        self.run_count += 1
        result = _execute_nn_graph(
            self.graph,
            values,
            self.value_store,
            backend=backend,
            inputs=dict(inputs or {}),
            seed=run_seed,
            no_output_digest=no_output_digest,
            validate_output=validate_output,
            validation_rtol=validation_rtol,
            validation_atol=validation_atol,
            validation_sample_count=validation_sample_count,
            materialize_reason=materialize_reason,
            foundry_auto_queue=foundry_auto_queue,
            foundry_catalog=foundry_catalog,
            foundry_action_ledger_path=foundry_action_ledger_path,
            foundry_target=foundry_target,
            fabric_daemon=fabric_daemon,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            fabric_backend=fabric_backend,
            daemon_buffer_storage=daemon_buffer_storage,
            policy=policy,
            policy_materialization=policy_materialization,
            fabric_session_cache=self._fabric_sessions,
            fabric_plan_cache=self._fabric_plan_cache,
            fabric_value_release=self._release_fabric_runtime_for_value,
        )
        if result.ok and feedback_map:
            self._update_feedback_values(dict(feedback_map))
        return result

    def _release_replaced_boundary_values(self) -> None:
        retained_ids = {
            value.value_id
            for value in [
                *self.retained_values.values(),
                *self.retained_input_values.values(),
                *self.retained_feedback_values.values(),
            ]
        }
        latest = dict(self.value_store.to_json()["latest_by_name"])
        for name in [*self.graph.inputs, *self.graph.outputs]:
            value_id = latest.get(name)
            if value_id is not None and value_id not in retained_ids:
                self._release_fabric_runtime_for_value(self.value_store.get_latest(name))
                self.value_store.release(value_id)

    def _input_values_for_run(
        self,
        *,
        inputs: dict[str, Any],
        seed: int,
        feedback_map: dict[str, str],
    ) -> dict[str, NNTensorValue]:
        feedback_values = {
            name: value
            for name, value in self.retained_feedback_values.items()
            if name in feedback_map and name in self.graph.inputs
        }
        prepared_names = tuple(name for name in self.graph.inputs if name not in feedback_values)
        prepared = _prepared_input_arrays(self.graph, inputs=inputs, seed=seed, input_names=prepared_names)
        if not self.resident_inputs:
            values = _add_prepared_input_values(self.value_store, prepared, retained=False)
            values.update(feedback_values)
            return values
        signature = _prepared_input_signature(prepared)
        if self._retained_input_signature == signature and self.retained_input_values:
            values = dict(self.retained_input_values)
            values.update(feedback_values)
            return values
        self._release_retained_input_values()
        self.retained_input_values = _add_prepared_input_values(self.value_store, prepared, retained=True)
        self._retained_input_signature = signature
        values = dict(self.retained_input_values)
        values.update(feedback_values)
        return values

    def _update_feedback_values(self, feedback_map: dict[str, str]) -> None:
        for input_name, output_name in sorted(feedback_map.items()):
            if input_name not in self.graph.inputs:
                raise ValueError(f"feedback input {input_name!r} is not a graph input")
            try:
                value = self.value_store.get_latest(output_name)
            except KeyError as exc:
                raise ValueError(f"feedback output {output_name!r} was not produced") from exc
            prior = self.retained_feedback_values.get(input_name)
            value.metadata["retained_feedback"] = True
            value.metadata["feedback_input_name"] = str(input_name)
            value.metadata["feedback_output_name"] = str(output_name)
            self.retained_feedback_values[str(input_name)] = value
            if prior is not None and prior.value_id != value.value_id:
                self._release_fabric_runtime_for_value(prior)
                self.value_store.release(prior.value_id)

    def _release_retained_input_values(self) -> None:
        for value in list(self.retained_input_values.values()):
            self._release_fabric_runtime_for_value(value)
            self.value_store.release(value.value_id)
        self.retained_input_values.clear()
        self._retained_input_signature = None

    def _release_fabric_runtime_for_value(self, value: NNTensorValue, *, released: Optional[set[str]] = None) -> None:
        metadata = value.metadata.get("fabric_runtime")
        if not isinstance(metadata, dict) or not metadata.get("owned"):
            return
        fabric_value_id = metadata.get("value_id")
        session_id = metadata.get("session_id")
        if not fabric_value_id or not session_id:
            return
        release_key = f"{session_id}:{fabric_value_id}"
        if released is not None:
            if release_key in released:
                return
            released.add(release_key)
        for session in self._fabric_sessions.values():
            if session.value_store.session_id != session_id:
                continue
            try:
                fabric_value = session.value_store.get(str(fabric_value_id))
            except KeyError:
                return
            session.release_value(fabric_value)
            return


def load_nn_model(path: str | Path) -> NNGraph:
    source = Path(path)
    data = source.read_bytes()
    stripped = data.lstrip()
    if stripped.startswith(b"{"):
        return NNGraph.from_json(json.loads(data.decode("utf-8")), source_name=source.name)
    if source.suffix.lower() == ".onnx":
        return _load_onnx_model(source)
    raise ValueError(f"unsupported NN model format for {source}")


def run_nn_model(
    model_path: str | Path,
    *,
    backend: str = "auto",
    inputs: Optional[dict[str, Any]] = None,
    seed: int = 1,
    resident_weights: bool = False,
    no_output_digest: bool = False,
    validate_output: bool = False,
    validation_rtol: Optional[float] = None,
    validation_atol: Optional[float] = None,
    validation_sample_count: Optional[int] = None,
    materialize_reason: Optional[str] = None,
    foundry_auto_queue: bool = False,
    foundry_catalog: Optional[str] = None,
    foundry_action_ledger_path: Optional[str] = None,
    foundry_target: str = "kv260",
    fabric_daemon: str = "/tmp/pan-fabricd.sock",
    workdir: Optional[str] = None,
    xclbin_dirs: Optional[Sequence[str]] = None,
    active_app: str = "auto",
    fabric_backend: str = "pl",
    daemon_buffer_storage: str = "inline",
    policy: Optional[FabricPolicyHead] = None,
    policy_materialization: bool = False,
) -> NNExecutionResult:
    if backend not in {"auto", "ps", "sim", "fabric-session"}:
        raise ValueError(f"unsupported NN backend {backend!r}")
    graph = load_nn_model(model_path)
    value_store = NNTensorValueStore()
    try:
        values = _prepare_initial_values(
            graph,
            value_store,
            inputs=dict(inputs or {}),
            seed=int(seed),
            resident_weights=bool(resident_weights),
        )
    except Exception as exc:
        return NNExecutionResult(
            ok=False,
            graph=graph,
            backend=backend,
            outputs={},
            trace=(),
            training_records=(),
            value_store=value_store.to_json(),
            elapsed_seconds=0.0,
            validation=None,
            error=str(exc),
        )
    fabric_session_cache: dict[str, FabricRuntimeSession] | None = {} if backend == "fabric-session" else None
    fabric_plan_cache: dict[str, _NNFabricPlanCacheEntry] | None = {} if backend == "fabric-session" else None
    try:
        return _execute_nn_graph(
            graph,
            values,
            value_store,
            backend=backend,
            inputs=dict(inputs or {}),
            seed=int(seed),
            no_output_digest=no_output_digest,
            validate_output=validate_output,
            validation_rtol=validation_rtol,
            validation_atol=validation_atol,
            validation_sample_count=validation_sample_count,
            materialize_reason=materialize_reason,
            foundry_auto_queue=foundry_auto_queue,
            foundry_catalog=foundry_catalog,
            foundry_action_ledger_path=foundry_action_ledger_path,
            foundry_target=foundry_target,
            fabric_daemon=fabric_daemon,
            workdir=workdir,
            xclbin_dirs=xclbin_dirs,
            active_app=active_app,
            fabric_backend=fabric_backend,
            daemon_buffer_storage=daemon_buffer_storage,
            policy=policy,
            policy_materialization=policy_materialization,
            fabric_session_cache=fabric_session_cache,
            fabric_plan_cache=fabric_plan_cache,
        )
    finally:
        if fabric_session_cache is not None:
            for session in list(fabric_session_cache.values()):
                session.__exit__(None, None, None)
            fabric_session_cache.clear()


def _execute_nn_graph(
    graph: NNGraph,
    values: dict[str, NNTensorValue],
    value_store: NNTensorValueStore,
    *,
    backend: str,
    inputs: dict[str, Any],
    seed: int,
    no_output_digest: bool,
    validate_output: bool,
    validation_rtol: Optional[float],
    validation_atol: Optional[float],
    validation_sample_count: Optional[int],
    materialize_reason: Optional[str],
    foundry_auto_queue: bool,
    foundry_catalog: Optional[str],
    foundry_action_ledger_path: Optional[str],
    foundry_target: str,
    fabric_daemon: str,
    workdir: Optional[str],
    xclbin_dirs: Optional[Sequence[str]],
    active_app: str,
    fabric_backend: str,
    daemon_buffer_storage: str,
    policy: Optional[FabricPolicyHead],
    policy_materialization: bool,
    fabric_session_cache: Optional[dict[str, FabricRuntimeSession]] = None,
    fabric_plan_cache: Optional[dict[str, _NNFabricPlanCacheEntry]] = None,
    fabric_value_release: Optional[Callable[[NNTensorValue], None]] = None,
) -> NNExecutionResult:
    if backend not in {"auto", "ps", "sim", "fabric-session"}:
        raise ValueError(f"unsupported NN backend {backend!r}")
    started = time.perf_counter()
    trace: list[dict[str, Any]] = []
    training: list[dict[str, Any]] = []
    error = None
    outputs: dict[str, dict[str, Any]] = {}
    validation = None
    observed_active_app = _resolve_nn_active_app(active_app)
    spf32_ps_engine = _new_spf32_ps_engine(workdir) if _graph_has_spf32_ops(graph) else None

    try:
        remaining_consumers = _tensor_consumer_counts(graph)
        for region_index, region in enumerate(graph.regions()):
            action = _select_region_action(
                region,
                backend=backend,
                values=values,
                graph=graph,
                foundry_auto_queue=foundry_auto_queue,
                foundry_catalog=foundry_catalog,
                foundry_target=foundry_target,
                policy=policy,
            )
            if action["selected_action"] == "reject_fail_closed":
                raise RuntimeError(action["reason"])
            before = {name: value.residency for name, value in values.items()}
            region_started = time.perf_counter()
            fabric_result = None
            region_selected_action = action["selected_action"]
            execute_fabric_session = (
                (
                    region_selected_action == "run_daemon_resident_now"
                    or (
                        region_selected_action == "run_catalog_core_now"
                        and bool(fabric_daemon)
                        and Path(fabric_daemon).exists()
                    )
                )
                and _region_can_lower_to_fabric_session(region)
                and backend in {"auto", "fabric-session"}
            )
            if execute_fabric_session:
                visible_region_outputs = [output for output in region.outputs if output in graph.outputs]
                if visible_region_outputs:
                    materialize_region_outputs, region_materialization_reason, _ = _resolve_materialization_policy(
                        action,
                        visible_outputs=visible_region_outputs,
                        default_include_output_digest=bool(not no_output_digest),
                        validate_output=validate_output,
                        materialize_reason=materialize_reason,
                        policy_materialization=policy_materialization,
                    )
                else:
                    materialize_region_outputs = False
                    region_materialization_reason = None
                fabric_result = _execute_region_fabric_session(
                    graph,
                    values,
                    value_store,
                    region,
                    fabric_daemon=fabric_daemon,
                    workdir=workdir,
                    xclbin_dirs=xclbin_dirs,
                    active_app=active_app,
                    fabric_backend=fabric_backend,
                    daemon_buffer_storage=daemon_buffer_storage,
                    materialize_outputs=materialize_region_outputs,
                    materialization_reason=region_materialization_reason,
                    fabric_session_cache=fabric_session_cache,
                    fabric_plan_cache=fabric_plan_cache,
                    resolved_active_app=observed_active_app,
                    selected_action=region_selected_action,
                )
            else:
                _execute_region_ps(graph, values, value_store, region, action=action, spf32_engine=spf32_ps_engine)
            elapsed = time.perf_counter() - region_started
            tensor_metadata = _nn_region_tensor_metadata(region, values)
            output_names = list(region.outputs)
            foundry_result = None
            if foundry_catalog and foundry_auto_queue and _action_requests_foundry(action):
                foundry_result = _queue_foundry_for_nn_region(
                    graph,
                    region,
                    values,
                    action,
                    catalog_root=foundry_catalog,
                    action_ledger_path=foundry_action_ledger_path,
                    target=foundry_target,
                    elapsed_seconds=elapsed,
                    fabric_result=fabric_result,
                )
            released_values = _release_dead_tensors(
                graph,
                region,
                values,
                value_store,
                remaining_consumers,
                release_callback=fabric_value_release,
            )
            after = {name: value.residency for name, value in values.items()}
            row = _trace_row(
                graph,
                region,
                region_index=region_index,
                action=action,
                elapsed_seconds=elapsed,
                residency_before=before,
                residency_after=after,
                output_names=output_names,
                fabric_result=fabric_result,
                foundry_result=foundry_result,
                tensor_metadata=tensor_metadata,
                released_values=released_values,
                runtime_context={
                    "target": foundry_target,
                    "active_app": observed_active_app,
                    "active_app_requested": active_app,
                    "fabric_backend": fabric_backend,
                    "daemon_buffer_storage": daemon_buffer_storage,
                },
            )
            trace.append(row)
        if validate_output:
            validation_policy = _resolve_validation_policy(
                graph,
                validation_rtol=validation_rtol,
                validation_atol=validation_atol,
                validation_sample_count=validation_sample_count,
            )
            expected = execute_nn_ps_oracle(
                graph,
                inputs=_validation_inputs_for_graph(graph, values, fallback_inputs=inputs),
                seed=int(seed),
                workdir=workdir,
            )
            validation = _validate_outputs(
                values,
                graph.outputs,
                expected,
                rtol=float(validation_policy["rtol"]),
                atol=float(validation_policy["atol"]),
                sample_count=validation_policy.get("sample_count"),
                seed=int(seed),
            )
            validation["policy"] = validation_policy
            validation_summary = {
                "ok": bool(validation.get("ok", False)),
                "oracle": validation.get("oracle"),
                "mode": validation.get("mode"),
                "rtol": float(validation_policy["rtol"]),
                "atol": float(validation_policy["atol"]),
                "sample_count": validation.get("sample_count"),
                "policy": validation_policy,
            }
            for row in trace:
                row["validation"] = validation_summary
            for output_name, result in validation.get("outputs", {}).items():
                value = values.get(output_name)
                if value is not None:
                    value.validation = result
        output_materialized, output_materialization_reason, _ = _resolve_trace_output_materialization(
            trace,
            graph_outputs=graph.outputs,
            default_include_output_digest=not no_output_digest,
            validate_output=validate_output,
            materialize_reason=materialize_reason,
            policy_materialization=policy_materialization,
        )
        _annotate_materialization_actions(
            trace,
            graph_outputs=graph.outputs,
            include_output_digest=output_materialized,
            materialization_reason=output_materialization_reason,
            policy_materialization=policy_materialization,
        )
        for output_name in graph.outputs:
            value = values[output_name]
            outputs[output_name] = _output_summary(
                value,
                include_digest=output_materialized,
                materialization_reason=output_materialization_reason,
            )
        training = [_training_row(row) for row in trace]
    except Exception as exc:
        error = str(exc)
    if not training and trace:
        training = [_training_row(row) for row in trace]
    elapsed = time.perf_counter() - started
    return NNExecutionResult(
        ok=error is None and (validation is None or bool(validation.get("ok", False))),
        graph=graph,
        backend=backend,
        outputs=outputs,
        trace=tuple(trace),
        training_records=tuple(training),
        value_store=value_store.to_json(),
        elapsed_seconds=elapsed,
        validation=validation,
        error=error,
    )


def _validation_inputs_for_graph(
    graph: NNGraph,
    values: dict[str, NNTensorValue],
    *,
    fallback_inputs: dict[str, Any],
) -> dict[str, Any]:
    validation_inputs: dict[str, Any] = {}
    for name in graph.inputs:
        value = values.get(name)
        if value is not None and value.array is not None:
            validation_inputs[name] = value.array
        elif name in fallback_inputs:
            validation_inputs[name] = fallback_inputs[name]
    return validation_inputs


def train_nn_policy_from_traces(
    paths: Sequence[str | Path],
    *,
    l2: float = 1.0e-6,
) -> FabricPolicyHead:
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("schema") != PAN_NN_TRAINING_SCHEMA:
                    continue
                if not isinstance(row.get("features"), dict):
                    row["features"] = _nn_training_features_from_row(row)
                if not isinstance(row.get("labels"), dict):
                    row["labels"] = _nn_training_labels_from_row(row)
                rows.append(row)
    if not rows:
        raise ValueError("no PAN NN training records found")
    np = _numpy()
    feature_names = sorted({name for row in rows for name in row["features"]})
    x = np.ones((len(rows), len(feature_names) + 1), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, name in enumerate(feature_names):
            x[row_index, feature_index + 1] = float(row["features"].get(name, 0.0))
        y[row_index] = float(row.get("elapsed_seconds") or 0.0)
    reg = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
    return FabricPolicyHead(
        bias=float(coef[0]),
        weights={name: float(coef[index + 1]) for index, name in enumerate(feature_names)},
        policy_kind="linear_neural_nn_fabric_head_v1_trained_from_trace",
        policy_role="neural_candidate",
        promotion_status="candidate",
        can_override_rules=False,
        min_confidence=0.85,
        heads=_nn_policy_heads(rows),
        heldout_metrics={
            "status": "unmeasured",
            "heads_trained": [
                "placement_action",
                "preferred_residency",
                "materialization_needed",
                "compile_priority",
                "latency_estimate",
                "transfer_estimate",
                "energy_estimate",
                "resource_estimate",
                "reconfiguration_estimate",
                "transfer_avoidance_estimate",
                "validation_risk",
            ],
            "required_for_promotion": [
                "latency_beats_rule_scheduler",
                "transfer_bytes_not_worse",
                "energy_not_worse",
                "validation_failures_zero",
                "output_residency_not_worse",
                "materialization_regressions_zero",
                "compile_priority_precision_checked",
            ],
        },
    )


def execute_nn_ps_oracle(
    graph: NNGraph,
    *,
    inputs: Optional[dict[str, Any]] = None,
    seed: int = 1,
    workdir: Optional[str] = None,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    rng = _numpy().random.default_rng(int(seed))
    spf32_engine = _new_spf32_ps_engine(workdir) if _graph_has_spf32_ops(graph) else None
    for name, spec in graph.tensors.items():
        if spec.initializer:
            values[name] = _spec_array(spec)
    for name in graph.inputs:
        if inputs and name in inputs:
            spec = graph.tensors.get(name)
            dtype = spec.dtype if spec is not None else "float32"
            values[name] = _array_from_payload(inputs[name], dtype=dtype)
        elif name in values:
            continue
        else:
            spec = graph.tensors.get(name)
            values[name] = _generated_input(spec, rng)
    for node in graph.nodes:
        _execute_node(values, node, tensor_specs=graph.tensors, spf32_engine=spf32_engine)
    return {name: values[name] for name in graph.outputs}


def _graph_has_spf32_ops(graph: NNGraph) -> bool:
    return any(node.op in SPF32_BINARY_NN_OPS.union(SPF32_UNARY_NN_OPS) for node in graph.nodes)


def _new_spf32_ps_engine(workdir: Optional[str]) -> _PSOpEngine:
    return _PSOpEngine(workdir=workdir or DEFAULT_FABRIC_WORKDIR, libspf32_path=None)


def form_nn_regions(graph: NNGraph) -> tuple[NNRegion, ...]:
    regions: list[NNRegion] = []
    consumer_counts = _tensor_consumer_counts(graph)
    graph_outputs = set(graph.outputs)
    index = 0
    while index < len(graph.nodes):
        node = graph.nodes[index]
        if node.op in FABRIC_SESSION_ELIGIBLE_OPS:
            nodes = [node]
            current_output = node.outputs[0] if len(node.outputs) == 1 else None
            cursor = index + 1
            while cursor < len(graph.nodes):
                candidate = graph.nodes[cursor]
                if (
                    not current_output
                    or current_output in graph_outputs
                    or consumer_counts.get(current_output, 0) != 1
                    or candidate.op not in FABRIC_SESSION_ELIGIBLE_OPS
                    or current_output not in candidate.inputs
                    or len(candidate.outputs) != 1
                ):
                    break
                nodes.append(candidate)
                current_output = candidate.outputs[0]
                cursor += 1
            reason = "fused_spf32_fabric_region" if len(nodes) > 1 else "single_op_region"
            regions.append(NNRegion(f"region{len(regions)}", tuple(nodes), reason))
            index = cursor
            continue
        raw_attention = _raw_attention_region_nodes(graph.nodes, index)
        if raw_attention:
            regions.append(NNRegion(f"region{len(regions)}", raw_attention, "fused_attention_region"))
            index += len(raw_attention)
            continue
        gated_mlp = _gated_mlp_region_nodes(graph, index, consumer_counts, graph_outputs)
        if gated_mlp:
            regions.append(NNRegion(f"region{len(regions)}", gated_mlp, "fused_gated_mlp_region"))
            index += len(gated_mlp)
            continue
        decode_output = _decode_output_region_nodes(graph, index, consumer_counts, graph_outputs)
        if decode_output:
            regions.append(NNRegion(f"region{len(regions)}", decode_output, "fused_decode_output_region"))
            index += len(decode_output)
            continue
        decode_sampling = _decode_sampling_region_nodes(graph, index, consumer_counts, graph_outputs)
        if decode_sampling:
            regions.append(NNRegion(f"region{len(regions)}", decode_sampling, "fused_decode_sampling_region"))
            index += len(decode_sampling)
            continue
        if node.op in {"matmul", "gemm", "qlinear_matmul", "matmul_integer", "conv2d", "depthwise_conv2d"}:
            first_op = node.op
            nodes = [node]
            current_output = node.outputs[0] if node.outputs else None
            cursor = index + 1
            while cursor < len(graph.nodes):
                candidate = graph.nodes[cursor]
                if not current_output or current_output not in candidate.inputs:
                    break
                decomposed_gelu = _parse_decomposed_gelu_activation(
                    graph,
                    cursor,
                    current_output,
                    consumer_counts,
                    graph_outputs,
                )
                if decomposed_gelu is not None:
                    cursor, current_output, activation_nodes = decomposed_gelu
                    nodes.extend(activation_nodes)
                    continue
                if candidate.op not in {"add", "batchnorm", "relu", "gelu", "bias_gelu", "quick_gelu", "silu", "sigmoid", "tanh", "erf", "layernorm", "rmsnorm"}:
                    break
                nodes.append(candidate)
                current_output = candidate.outputs[0] if candidate.outputs else current_output
                cursor += 1
            if (
                first_op in {"matmul", "gemm"}
                and cursor < len(graph.nodes)
                and _region_has_activation(nodes)
                and _can_append_mlp_projection(graph.nodes[cursor], current_output)
            ):
                nodes.append(graph.nodes[cursor])
                current_output = graph.nodes[cursor].outputs[0] if graph.nodes[cursor].outputs else current_output
                cursor += 1
                if current_output:
                    parsed_bias = _parse_optional_region_add(graph, cursor, current_output, consumer_counts, graph_outputs)
                    if parsed_bias is not None:
                        next_cursor, next_output, bias_nodes = parsed_bias
                        if bias_nodes:
                            nodes.extend(bias_nodes)
                            current_output = next_output
                            cursor = next_cursor
                reason = "fused_mlp_region"
            else:
                reason = (
                    "fused_conv_region"
                    if first_op in {"conv2d", "depthwise_conv2d"} and len(nodes) > 1
                    else "fused_linear_region" if len(nodes) > 1 else "single_compute_region"
                )
            regions.append(NNRegion(f"region{len(regions)}", tuple(nodes), reason))
            index = cursor
            continue
        if node.op == "attention_score":
            nodes = [node]
            current_output = node.outputs[0] if node.outputs else None
            cursor = index + 1
            while cursor < len(graph.nodes):
                candidate = graph.nodes[cursor]
                if not current_output or current_output not in candidate.inputs:
                    break
                if candidate.op not in {"softmax", "attention_value"}:
                    break
                nodes.append(candidate)
                current_output = candidate.outputs[0] if candidate.outputs else current_output
                cursor += 1
                if candidate.op == "attention_value":
                    break
            reason = "fused_attention_region" if len(nodes) > 1 else "single_attention_region"
            regions.append(NNRegion(f"region{len(regions)}", tuple(nodes), reason))
            index = cursor
            continue
        if node.op == "add" and index + 1 < len(graph.nodes):
            next_node = graph.nodes[index + 1]
            if (
                next_node.op in {"layernorm", "rmsnorm"}
                and node.outputs
                and node.outputs[0] in next_node.inputs
                and _residual_layernorm_specs_match(graph, node, next_node)
            ):
                reason = "fused_residual_layernorm" if next_node.op == "layernorm" else "fused_residual_rmsnorm"
                regions.append(NNRegion(f"region{len(regions)}", (node, next_node), reason))
                index += 2
                continue
        reason = "unsupported_region" if node.op not in PS_SUPPORTED_OPS else "single_op_region"
        regions.append(NNRegion(f"region{len(regions)}", (node,), reason))
        index += 1
    return tuple(regions)


def _raw_attention_region_nodes(nodes: Sequence[NNNode], index: int) -> tuple[NNNode, ...]:
    if index >= len(nodes):
        return ()
    score = nodes[index]
    if score.op != "matmul" or not score.outputs:
        return ()
    region = [score]
    current_output = score.outputs[0]
    cursor = index + 1
    if cursor < len(nodes):
        scale = nodes[cursor]
        if scale.op in {"mul", "div"} and current_output in scale.inputs and scale.outputs:
            region.append(scale)
            current_output = scale.outputs[0]
            cursor += 1
    if cursor < len(nodes):
        mask = nodes[cursor]
        if mask.op in {"add", "where"} and current_output in mask.inputs and mask.outputs:
            region.append(mask)
            current_output = mask.outputs[0]
            cursor += 1
    if cursor >= len(nodes):
        return ()
    softmax = nodes[cursor]
    if softmax.op != "softmax" or current_output not in softmax.inputs or not softmax.outputs:
        return ()
    region.append(softmax)
    current_output = softmax.outputs[0]
    cursor += 1
    if cursor >= len(nodes):
        return ()
    value = nodes[cursor]
    if value.op not in {"matmul", "attention_value"} or current_output not in value.inputs or not value.outputs:
        return ()
    region.append(value)
    return tuple(region)


def _decode_output_region_nodes(
    graph: NNGraph,
    index: int,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    prefix = _decode_projection_prefix_nodes(graph, index, consumer_counts, graph_outputs)
    if prefix is None:
        return ()
    cursor, projection_output, projection_nodes = prefix
    if cursor >= len(graph.nodes):
        return ()
    selection = graph.nodes[cursor]
    if (
        selection.op != "argmax"
        or projection_output not in selection.inputs
        or len(selection.outputs) != 1
        or not _region_tensor_is_private(projection_output, consumer_counts, graph_outputs)
    ):
        return ()
    return (*projection_nodes, selection)


def _decode_sampling_region_nodes(
    graph: NNGraph,
    index: int,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    if index >= len(graph.nodes):
        return ()
    first = graph.nodes[index]
    if first.op in {"matmul", "gemm"}:
        prefix = _decode_projection_prefix_nodes(graph, index, consumer_counts, graph_outputs)
        if prefix is None:
            return ()
        cursor, projection_output, projection_nodes = prefix
        if cursor + 1 >= len(graph.nodes):
            return ()
        filter_node = graph.nodes[cursor]
        selection = graph.nodes[cursor + 1]
        if filter_node.op != "top_p_filter" or projection_output not in filter_node.inputs:
            return ()
        sampling = _decode_sampling_filter_topk_nodes(filter_node, selection, consumer_counts, graph_outputs)
        return (*projection_nodes, *sampling) if sampling else ()
    if index + 1 >= len(graph.nodes):
        return ()
    return _decode_sampling_filter_topk_nodes(first, graph.nodes[index + 1], consumer_counts, graph_outputs)


def _decode_sampling_filter_topk_nodes(
    filter_node: NNNode,
    selection: NNNode,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    if filter_node.op != "top_p_filter" or not filter_node.outputs:
        return ()
    filtered_logits = filter_node.outputs[0]
    if any(output in graph_outputs or consumer_counts.get(output, 0) for output in filter_node.outputs[1:]):
        return ()
    if (
        selection.op != "topk"
        or filtered_logits not in selection.inputs
        or len(selection.outputs) < 2
        or not _region_tensor_is_private(filtered_logits, consumer_counts, graph_outputs)
    ):
        return ()
    return (filter_node, selection)


def _decode_projection_prefix_nodes(
    graph: NNGraph,
    index: int,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, tuple[NNNode, ...]]]:
    if index >= len(graph.nodes):
        return None
    projection = graph.nodes[index]
    if projection.op not in {"matmul", "gemm"} or len(projection.outputs) != 1:
        return None
    projection_output = projection.outputs[0]
    parsed_bias = _parse_optional_region_add(graph, index + 1, projection_output, consumer_counts, graph_outputs)
    if parsed_bias is None:
        return None
    cursor, current_output, bias_nodes = parsed_bias
    if not bias_nodes and not _region_tensor_is_private(projection_output, consumer_counts, graph_outputs):
        return None
    return cursor, current_output, (projection, *bias_nodes)


def _gated_mlp_region_nodes(
    graph: NNGraph,
    index: int,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    if index >= len(graph.nodes):
        return ()
    gate_projection = graph.nodes[index]
    if gate_projection.op not in {"matmul", "gemm"} or len(gate_projection.outputs) != 1:
        return ()
    gate_source = _linear_data_input(graph, gate_projection)
    if not gate_source:
        return ()
    gate_output = gate_projection.outputs[0]
    gate_cursor = index + 1
    parsed = _parse_optional_region_add(graph, gate_cursor, gate_output, consumer_counts, graph_outputs)
    if parsed is None:
        return ()
    cursor, gate_value, gate_bias_nodes = parsed
    prefix = [gate_projection, *gate_bias_nodes]
    for parser in (_parse_gated_mlp_activation_first, _parse_gated_mlp_projection_first):
        matched = parser(graph, cursor, gate_source, gate_value, prefix, consumer_counts, graph_outputs)
        if matched:
            return matched
    return ()


def _parse_gated_mlp_activation_first(
    graph: NNGraph,
    cursor: int,
    gate_source: str,
    gate_value: str,
    prefix: Sequence[NNNode],
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    parsed_activation = _parse_gated_mlp_activation(graph.nodes, cursor, gate_value, consumer_counts, graph_outputs)
    if parsed_activation is None:
        return ()
    cursor, gate_activation_value, activation_nodes = parsed_activation
    parsed_up = _parse_shared_mlp_projection(graph, cursor, gate_source, consumer_counts, graph_outputs)
    if parsed_up is None:
        return ()
    cursor, up_value, up_nodes = parsed_up
    return _finish_gated_mlp_region(
        graph,
        cursor,
        [*prefix, *activation_nodes, *up_nodes],
        gate_activation_value,
        up_value,
        consumer_counts,
        graph_outputs,
    )


def _parse_gated_mlp_projection_first(
    graph: NNGraph,
    cursor: int,
    gate_source: str,
    gate_value: str,
    prefix: Sequence[NNNode],
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    parsed_up = _parse_shared_mlp_projection(graph, cursor, gate_source, consumer_counts, graph_outputs)
    if parsed_up is None:
        return ()
    cursor, up_value, up_nodes = parsed_up
    parsed_activation = _parse_gated_mlp_activation(graph.nodes, cursor, gate_value, consumer_counts, graph_outputs)
    if parsed_activation is None:
        return ()
    cursor, gate_activation_value, activation_nodes = parsed_activation
    return _finish_gated_mlp_region(
        graph,
        cursor,
        [*prefix, *up_nodes, *activation_nodes],
        gate_activation_value,
        up_value,
        consumer_counts,
        graph_outputs,
    )


def _parse_gated_mlp_activation(
    nodes: Sequence[NNNode],
    cursor: int,
    gate_value: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, tuple[NNNode, ...]]]:
    if cursor >= len(nodes):
        return None
    activation = nodes[cursor]
    decomposed = _parse_decomposed_silu_activation(nodes, cursor, gate_value, consumer_counts, graph_outputs)
    if decomposed is not None:
        return decomposed
    if (
        activation.op not in {"relu", "gelu", "bias_gelu", "quick_gelu", "silu", "sigmoid", "tanh"}
        or gate_value not in activation.inputs
        or len(activation.outputs) != 1
        or not _region_tensor_is_private(gate_value, consumer_counts, graph_outputs)
    ):
        return None
    return cursor + 1, activation.outputs[0], (activation,)


def _parse_decomposed_gelu_activation(
    graph: NNGraph,
    cursor: int,
    source: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, tuple[NNNode, ...]]]:
    exact = _parse_exact_erf_gelu_activation(graph, cursor, source, consumer_counts, graph_outputs)
    if exact is not None:
        return exact
    return _parse_tanh_approx_gelu_activation(graph, cursor, source, consumer_counts, graph_outputs)


def _parse_exact_erf_gelu_activation(
    graph: NNGraph,
    cursor: int,
    source: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, tuple[NNNode, ...]]]:
    nodes = graph.nodes
    if not source or source in graph_outputs or cursor + 4 >= len(nodes):
        return None
    scale = nodes[cursor]
    erf = nodes[cursor + 1]
    bias = nodes[cursor + 2]
    gate_mul = nodes[cursor + 3]
    half_mul = nodes[cursor + 4]
    if (
        scale.op not in {"div", "mul"}
        or source not in scale.inputs
        or len(scale.outputs) != 1
        or consumer_counts.get(source, 0) != 2
    ):
        return None
    scale_constant = _single_other_input(scale, source)
    if scale.op == "div":
        if not _initializer_scalar_close(graph, scale_constant, math.sqrt(2.0)):
            return None
    elif not _initializer_scalar_close(graph, scale_constant, 1.0 / math.sqrt(2.0)):
        return None
    scaled = scale.outputs[0]
    if not _region_tensor_is_private(scaled, consumer_counts, graph_outputs):
        return None
    if erf.op != "erf" or scaled not in erf.inputs or len(erf.outputs) != 1:
        return None
    erfed = erf.outputs[0]
    if not _region_tensor_is_private(erfed, consumer_counts, graph_outputs):
        return None
    if bias.op != "add" or erfed not in bias.inputs or len(bias.outputs) != 1:
        return None
    if not _initializer_scalar_close(graph, _single_other_input(bias, erfed), 1.0):
        return None
    gate = bias.outputs[0]
    if not _region_tensor_is_private(gate, consumer_counts, graph_outputs):
        return None
    if (
        gate_mul.op != "mul"
        or source not in gate_mul.inputs
        or gate not in gate_mul.inputs
        or len(gate_mul.outputs) != 1
    ):
        return None
    gated = gate_mul.outputs[0]
    if not _region_tensor_is_private(gated, consumer_counts, graph_outputs):
        return None
    if half_mul.op != "mul" or gated not in half_mul.inputs or len(half_mul.outputs) != 1:
        return None
    if not _initializer_scalar_close(graph, _single_other_input(half_mul, gated), 0.5):
        return None
    return cursor + 5, half_mul.outputs[0], (scale, erf, bias, gate_mul, half_mul)


def _parse_tanh_approx_gelu_activation(
    graph: NNGraph,
    cursor: int,
    source: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, tuple[NNNode, ...]]]:
    nodes = graph.nodes
    if not source or source in graph_outputs or cursor + 7 >= len(nodes):
        return None
    power = nodes[cursor]
    cubic_mul = nodes[cursor + 1]
    inner_add = nodes[cursor + 2]
    scale_mul = nodes[cursor + 3]
    tanh = nodes[cursor + 4]
    bias = nodes[cursor + 5]
    gate_mul = nodes[cursor + 6]
    half_mul = nodes[cursor + 7]
    if (
        power.op != "pow"
        or source not in power.inputs
        or len(power.outputs) != 1
        or consumer_counts.get(source, 0) != 3
        or not _initializer_scalar_close(graph, _single_other_input(power, source), 3.0)
    ):
        return None
    cubed = power.outputs[0]
    if not _region_tensor_is_private(cubed, consumer_counts, graph_outputs):
        return None
    if (
        cubic_mul.op != "mul"
        or cubed not in cubic_mul.inputs
        or len(cubic_mul.outputs) != 1
        or not _initializer_scalar_close(graph, _single_other_input(cubic_mul, cubed), 0.044715)
    ):
        return None
    cubic_term = cubic_mul.outputs[0]
    if not _region_tensor_is_private(cubic_term, consumer_counts, graph_outputs):
        return None
    if inner_add.op != "add" or source not in inner_add.inputs or cubic_term not in inner_add.inputs or len(inner_add.outputs) != 1:
        return None
    inner = inner_add.outputs[0]
    if not _region_tensor_is_private(inner, consumer_counts, graph_outputs):
        return None
    if (
        scale_mul.op != "mul"
        or inner not in scale_mul.inputs
        or len(scale_mul.outputs) != 1
        or not _initializer_scalar_close(graph, _single_other_input(scale_mul, inner), math.sqrt(2.0 / math.pi))
    ):
        return None
    scaled = scale_mul.outputs[0]
    if not _region_tensor_is_private(scaled, consumer_counts, graph_outputs):
        return None
    if tanh.op != "tanh" or scaled not in tanh.inputs or len(tanh.outputs) != 1:
        return None
    tanhed = tanh.outputs[0]
    if not _region_tensor_is_private(tanhed, consumer_counts, graph_outputs):
        return None
    if (
        bias.op != "add"
        or tanhed not in bias.inputs
        or len(bias.outputs) != 1
        or not _initializer_scalar_close(graph, _single_other_input(bias, tanhed), 1.0)
    ):
        return None
    gate = bias.outputs[0]
    if not _region_tensor_is_private(gate, consumer_counts, graph_outputs):
        return None
    if (
        gate_mul.op != "mul"
        or source not in gate_mul.inputs
        or gate not in gate_mul.inputs
        or len(gate_mul.outputs) != 1
    ):
        return None
    gated = gate_mul.outputs[0]
    if not _region_tensor_is_private(gated, consumer_counts, graph_outputs):
        return None
    if half_mul.op != "mul" or gated not in half_mul.inputs or len(half_mul.outputs) != 1:
        return None
    if not _initializer_scalar_close(graph, _single_other_input(half_mul, gated), 0.5):
        return None
    return cursor + 8, half_mul.outputs[0], (power, cubic_mul, inner_add, scale_mul, tanh, bias, gate_mul, half_mul)


def _single_other_input(node: NNNode, source: str) -> Optional[str]:
    others = [name for name in node.inputs if name != source]
    if len(others) != 1:
        return None
    return others[0]


def _initializer_scalar_close(graph: NNGraph, name: Optional[str], expected: float) -> bool:
    if not name:
        return False
    spec = graph.tensors.get(name)
    if spec is None or not spec.initializer or spec.data is None:
        return False
    try:
        np = _numpy()
        arr = np.asarray(spec.data)
        if arr.size != 1:
            return False
        actual = float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return False
    return math.isclose(actual, float(expected), rel_tol=1.0e-5, abs_tol=1.0e-7)


def _parse_decomposed_silu_activation(
    nodes: Sequence[NNNode],
    cursor: int,
    gate_value: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, tuple[NNNode, ...]]]:
    if cursor + 1 >= len(nodes):
        return None
    sigmoid = nodes[cursor]
    multiply = nodes[cursor + 1]
    if (
        sigmoid.op != "sigmoid"
        or gate_value not in sigmoid.inputs
        or len(sigmoid.outputs) != 1
        or gate_value in graph_outputs
        or consumer_counts.get(gate_value, 0) != 2
    ):
        return None
    sigmoid_output = sigmoid.outputs[0]
    if (
        multiply.op != "mul"
        or len(multiply.outputs) != 1
        or sigmoid_output not in multiply.inputs
        or gate_value not in multiply.inputs
        or not _region_tensor_is_private(sigmoid_output, consumer_counts, graph_outputs)
    ):
        return None
    return cursor + 2, multiply.outputs[0], (sigmoid, multiply)


def _parse_shared_mlp_projection(
    graph: NNGraph,
    cursor: int,
    gate_source: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, list[NNNode]]]:
    if cursor >= len(graph.nodes):
        return None
    projection = graph.nodes[cursor]
    if projection.op not in {"matmul", "gemm"} or len(projection.outputs) != 1:
        return None
    if gate_source not in _linear_data_inputs(graph, projection):
        return None
    output = projection.outputs[0]
    parsed = _parse_optional_region_add(graph, cursor + 1, output, consumer_counts, graph_outputs)
    if parsed is None:
        return None
    next_cursor, projected_value, bias_nodes = parsed
    return next_cursor, projected_value, [projection, *bias_nodes]


def _finish_gated_mlp_region(
    graph: NNGraph,
    cursor: int,
    nodes: Sequence[NNNode],
    gate_activation_value: str,
    up_value: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> tuple[NNNode, ...]:
    if cursor >= len(graph.nodes):
        return ()
    multiply = graph.nodes[cursor]
    if (
        multiply.op != "mul"
        or len(multiply.outputs) != 1
        or gate_activation_value not in multiply.inputs
        or up_value not in multiply.inputs
        or gate_activation_value == up_value
        or not _region_tensor_is_private(gate_activation_value, consumer_counts, graph_outputs)
        or not _region_tensor_is_private(up_value, consumer_counts, graph_outputs)
    ):
        return ()
    mul_output = multiply.outputs[0]
    next_cursor = cursor + 1
    if next_cursor >= len(graph.nodes):
        return ()
    down_projection = graph.nodes[next_cursor]
    if (
        down_projection.op not in {"matmul", "gemm"}
        or len(down_projection.outputs) != 1
        or mul_output not in _linear_data_inputs(graph, down_projection)
        or not _region_tensor_is_private(mul_output, consumer_counts, graph_outputs)
    ):
        return ()
    region_nodes = [*nodes, multiply, down_projection]
    down_output = down_projection.outputs[0]
    parsed_bias = _parse_optional_region_add(graph, next_cursor + 1, down_output, consumer_counts, graph_outputs)
    if parsed_bias is not None:
        _, _, bias_nodes = parsed_bias
        region_nodes.extend(bias_nodes)
    return tuple(region_nodes)


def _parse_optional_region_add(
    graph: NNGraph,
    cursor: int,
    current_output: str,
    consumer_counts: dict[str, int],
    graph_outputs: set[str],
) -> Optional[tuple[int, str, list[NNNode]]]:
    nodes = graph.nodes
    if cursor >= len(nodes):
        return cursor, current_output, []
    candidate = nodes[cursor]
    if candidate.op != "add" or current_output not in candidate.inputs:
        return cursor, current_output, []
    if len(candidate.outputs) != 1 or not _region_tensor_is_private(current_output, consumer_counts, graph_outputs):
        return None
    other_inputs = [name for name in candidate.inputs if name != current_output]
    if not other_inputs:
        return cursor, current_output, []
    for name in other_inputs:
        spec = graph.tensors.get(name)
        if spec is None or not spec.initializer:
            return cursor, current_output, []
    return cursor + 1, candidate.outputs[0], [candidate]


def _region_tensor_is_private(name: str, consumer_counts: dict[str, int], graph_outputs: set[str]) -> bool:
    return bool(name and name not in graph_outputs and consumer_counts.get(name, 0) == 1)


def _linear_data_inputs(graph: NNGraph, node: NNNode) -> tuple[str, ...]:
    non_initializer_inputs = []
    for name in node.inputs:
        spec = graph.tensors.get(name)
        if spec is None or not spec.initializer:
            non_initializer_inputs.append(name)
    return tuple(non_initializer_inputs or node.inputs)


def _linear_data_input(graph: NNGraph, node: NNNode) -> Optional[str]:
    inputs = _linear_data_inputs(graph, node)
    return inputs[0] if inputs else None


def _region_has_activation(nodes: Sequence[NNNode]) -> bool:
    return any(node.op in {"relu", "gelu", "bias_gelu", "quick_gelu", "silu", "sigmoid", "tanh", "erf"} for node in nodes)


def _mlp_feature_counts(ops: Sequence[str], *, reason: str = "") -> dict[str, float]:
    reason_lower = str(reason).lower()
    mlp_region = "mlp" in reason_lower
    gated_mlp_region = "gated_mlp" in reason_lower
    return {
        "mlp_region_count": float(mlp_region),
        "gated_mlp_region_count": float(gated_mlp_region),
        "gated_mlp_op_count": float(len(ops) if gated_mlp_region else 0),
    }


def _attention_feature_counts(ops: Sequence[str], *, reason: str = "") -> dict[str, float]:
    attention_ops = sum(1 for op in ops if op in {"attention_qkv", "attention_score", "attention_value"})
    attention_region = "attention" in str(reason).lower() or attention_ops > 0
    mask_ops = sum(
        1
        for op in ops
        if op in {"where", "equal", "less", "less_equal", "greater", "greater_equal", "not", "and", "or"}
    )
    return {
        "attention_region_count": float(attention_region),
        "attention_op_count": float(attention_ops if attention_ops else (len(ops) if attention_region else 0)),
        "mask_op_count": float(mask_ops),
    }


def _decode_feature_counts(ops: Sequence[str]) -> dict[str, float]:
    return {
        "decode_selection_op_count": float(sum(1 for op in ops if op in {"argmax", "topk", "top_p_filter"})),
        "kv_cache_op_count": float(sum(1 for op in ops if op == "kv_cache_append")),
    }


def _can_append_mlp_projection(node: NNNode, current_output: Optional[str]) -> bool:
    return bool(
        current_output
        and node.op in {"matmul", "gemm"}
        and current_output in node.inputs
        and node.outputs
    )


def _residual_layernorm_specs_match(graph: NNGraph, add_node: NNNode, layernorm_node: NNNode) -> bool:
    specs = [graph.tensors.get(name) for name in (*add_node.inputs, *layernorm_node.inputs)]
    known = [spec for spec in specs if spec is not None]
    if not known:
        return True
    dtypes = {spec.dtype for spec in known}
    if len(dtypes) > 1:
        return False
    quantization = {
        json.dumps(spec.quantization, sort_keys=True, separators=(",", ":"))
        for spec in known
        if spec.quantization
    }
    if len(quantization) > 1:
        return False
    output_name = add_node.outputs[0] if add_node.outputs else None
    output_spec = graph.tensors.get(output_name) if output_name else None
    base_spec = output_spec or graph.tensors.get(layernorm_node.inputs[0])
    if base_spec is None:
        return True
    base_shape = _static_spec_shape(base_spec)
    if base_shape is None:
        return False
    for name in (*add_node.inputs, *layernorm_node.inputs[1:]):
        spec = graph.tensors.get(name)
        if spec is None:
            continue
        shape = _static_spec_shape(spec)
        if shape is None:
            return False
        if _broadcast_shape_result(base_shape, shape) is None:
            return False
    return True


def _static_spec_shape(spec: NNTensorSpec) -> Optional[tuple[int, ...]]:
    shape = []
    for dim in spec.shape:
        if isinstance(dim, str):
            return None
        shape.append(int(dim))
    return tuple(shape)


def _broadcast_shape_result(left: Sequence[int], right: Sequence[int]) -> Optional[tuple[int, ...]]:
    try:
        np = _numpy()
        return tuple(int(dim) for dim in np.broadcast_shapes(tuple(left), tuple(right)))
    except ValueError:
        return None


def _shape_constraints_for_shape(shape: Sequence[int | str]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = []
    for axis, dim in enumerate(shape):
        if isinstance(dim, str) and not dim.isdigit():
            if dim in {"?", "dynamic"}:
                constraints.append(
                    {
                        "axis": int(axis),
                        "kind": "dynamic",
                        "constraint": "runtime_bound",
                    }
                )
            else:
                constraints.append(
                    {
                        "axis": int(axis),
                        "kind": "symbolic",
                        "symbol": dim,
                        "constraint": "runtime_bound",
                    }
                )
            continue
        constraints.append(
            {
                "axis": int(axis),
                "kind": "static",
                "value": int(dim),
                "constraint": "exact",
            }
        )
    return constraints


def _shape_metadata_for_array(spec: Optional[NNTensorSpec], array: Any, *, role: str = "value") -> dict[str, Any]:
    if spec is None:
        return {}
    np = _numpy()
    actual_shape = tuple(int(dim) for dim in np.asarray(array).shape)
    bindings = _shape_bindings_for_spec(spec, actual_shape, role=role)
    return {
        "spec_shape": list(spec.shape),
        "shape_constraints": _shape_constraints_for_shape(spec.shape),
        "shape_bindings": bindings,
    }


def _shape_bindings_for_spec(spec: NNTensorSpec, actual_shape: Sequence[int], *, role: str = "value") -> dict[str, int]:
    if spec.shape and len(spec.shape) != len(actual_shape):
        raise ValueError(
            f"shape_mismatch:{role}_rank:{spec.name}:expected{len(spec.shape)}:got{len(actual_shape)}"
        )
    bindings: dict[str, int] = {}
    for axis, (declared, actual) in enumerate(zip(spec.shape, actual_shape)):
        actual_dim = int(actual)
        if isinstance(declared, str) and not declared.isdigit():
            if declared in {"?", "dynamic"}:
                continue
            prior = bindings.get(declared)
            if prior is not None and prior != actual_dim:
                raise ValueError(
                    f"shape_mismatch:{role}_symbolic_dim:{spec.name}:{declared}:"
                    f"axis{axis}:expected{prior}:got{actual_dim}"
                )
            bindings[declared] = actual_dim
            continue
        expected = int(declared)
        if expected != actual_dim:
            raise ValueError(
                f"shape_mismatch:{role}_dim:{spec.name}:axis{axis}:expected{expected}:got{actual_dim}"
            )
    return bindings


def tensor_digest(array: Any) -> dict[str, Any]:
    np = _numpy()
    arr = np.ascontiguousarray(array)
    flat = arr.reshape(-1)
    first = flat[:8].tolist()
    return {
        "dtype": str(arr.dtype),
        "shape": [int(dim) for dim in arr.shape],
        "elements": int(flat.size),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
        "first_values": [float(value) if arr.dtype.kind == "f" else int(value) for value in first],
    }


def _prepare_initial_values(
    graph: NNGraph,
    value_store: NNTensorValueStore,
    *,
    inputs: dict[str, Any],
    seed: int,
    resident_weights: bool,
) -> dict[str, NNTensorValue]:
    values: dict[str, NNTensorValue] = {}
    values.update(_prepare_initializer_values(graph, value_store, resident_weights=resident_weights))
    values.update(_prepare_input_values(graph, value_store, inputs=inputs, seed=seed))
    return values


def _prepare_initializer_values(
    graph: NNGraph,
    value_store: NNTensorValueStore,
    *,
    resident_weights: bool,
) -> dict[str, NNTensorValue]:
    values: dict[str, NNTensorValue] = {}
    for name, spec in graph.tensors.items():
        if not spec.initializer:
            continue
        arr = _spec_array(spec)
        values[name] = value_store.add_array(
            name,
            arr,
            dtype=spec.dtype,
            residency="daemon" if resident_weights else "ps",
            materialized=not resident_weights,
            producer="initializer",
            metadata={
                "weight": True,
                "retained": bool(resident_weights),
                "quantization": dict(spec.quantization),
                **_shape_metadata_for_array(spec, arr, role="initializer"),
            },
        )
    return values


def _prepare_input_values(
    graph: NNGraph,
    value_store: NNTensorValueStore,
    *,
    inputs: dict[str, Any],
    seed: int,
) -> dict[str, NNTensorValue]:
    return _add_prepared_input_values(
        value_store,
        _prepared_input_arrays(graph, inputs=inputs, seed=seed),
        retained=False,
    )


def _prepared_input_arrays(
    graph: NNGraph,
    *,
    inputs: dict[str, Any],
    seed: int,
    input_names: Optional[Sequence[str]] = None,
) -> list[tuple[str, Any, str, dict[str, Any]]]:
    np = _numpy()
    rng = np.random.default_rng(int(seed))
    prepared: list[tuple[str, Any, str, dict[str, Any]]] = []
    for name in tuple(graph.inputs if input_names is None else input_names):
        spec = graph.tensors.get(name)
        if name in inputs:
            arr = _array_from_payload(inputs[name], dtype=spec.dtype if spec is not None else "float32")
        else:
            arr = _generated_input(spec, rng)
        metadata = {
            "quantization": dict(spec.quantization) if spec is not None else {},
            **_shape_metadata_for_array(spec, arr, role="input"),
        }
        prepared.append((name, arr, str(arr.dtype), metadata))
    return prepared


def _add_prepared_input_values(
    value_store: NNTensorValueStore,
    prepared: Sequence[tuple[str, Any, str, dict[str, Any]]],
    *,
    retained: bool,
) -> dict[str, NNTensorValue]:
    values: dict[str, NNTensorValue] = {}
    for name, arr, dtype, metadata in prepared:
        item_metadata = dict(metadata)
        if retained:
            item_metadata["retained_input"] = True
        values[name] = value_store.add_array(
            name,
            arr,
            dtype=dtype,
            residency="ps",
            producer="input",
            metadata=item_metadata,
        )
    return values


def _prepared_input_signature(prepared: Sequence[tuple[str, Any, str, dict[str, Any]]]) -> str:
    payload = []
    for name, arr, dtype, _metadata in prepared:
        digest = tensor_digest(arr)
        payload.append(
            {
                "name": str(name),
                "dtype": str(dtype),
                "shape": digest["shape"],
                "sha256": digest["sha256"],
            }
        )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _tensor_consumer_counts(graph: NNGraph) -> dict[str, int]:
    counts: dict[str, int] = {}
    for node in graph.nodes:
        for name in node.inputs:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _release_dead_tensors(
    graph: NNGraph,
    region: NNRegion,
    values: dict[str, NNTensorValue],
    value_store: NNTensorValueStore,
    remaining_consumers: dict[str, int],
    *,
    release_callback: Optional[Callable[[NNTensorValue], None]] = None,
) -> list[dict[str, Any]]:
    for node in region.nodes:
        for name in node.inputs:
            if name in remaining_consumers:
                remaining_consumers[name] = max(0, int(remaining_consumers[name]) - 1)
    graph_outputs = set(graph.outputs)
    candidate_names = {
        name
        for node in region.nodes
        for name in (*node.inputs, *node.outputs)
        if name in values and name not in graph_outputs
    }
    released: list[dict[str, Any]] = []
    for name in sorted(candidate_names):
        value = values.get(name)
        if value is None:
            continue
        if remaining_consumers.get(name, 0) > 0:
            continue
        if value.metadata.get("retained_feedback"):
            continue
        if value.producer in {None, "input", "initializer"}:
            continue
        released.append(
            {
                "name": value.name,
                "value_id": value.value_id,
                "residency": value.residency,
                "producer": value.producer,
                "reason": "last_consumer_committed",
            }
        )
        if release_callback is not None:
            release_callback(value)
        del values[name]
        value_store.release(value.value_id)
    return released


def _verify_region_constraints(
    region: NNRegion,
    values: dict[str, NNTensorValue],
    graph: NNGraph,
) -> dict[str, Any]:
    local_values = dict(values)
    produced: set[str] = set()
    checks: list[str] = []
    for node in region.nodes:
        missing = [
            name
            for name in node.inputs
            if name not in local_values and name not in produced
        ]
        if missing:
            return _constraint_failure("missing_input_value", node=node, inputs=missing, checks=checks)
        if _node_has_side_effects(node):
            return _constraint_failure("side_effecting_node", node=node, checks=checks)
        node_values = [local_values[name] for name in node.inputs if name in local_values]
        dtype_reason = _verify_node_dtype_constraints(node, node_values)
        if dtype_reason is not None:
            return _constraint_failure(dtype_reason, node=node, checks=checks)
        quant_reason = _verify_node_quantization_constraints(node, node_values, graph)
        if quant_reason is not None:
            return _constraint_failure(quant_reason, node=node, checks=checks)
        shape_reason = _verify_node_shape_constraints(node, node_values)
        if shape_reason is not None:
            return _constraint_failure(shape_reason, node=node, checks=checks)
        checks.append(f"{node.node_id}:ok")
        produced.update(node.outputs)
        local_values.update(_inferred_verifier_outputs(node, node_values))
    return {
        "schema": "pan_nn_constraint_verification_v1",
        "ok": True,
        "reason": "verified",
        "checks": checks,
    }


def _node_has_side_effects(node: NNNode) -> bool:
    if node.op == "dropout" and _truthy_metadata_value(node.attrs.get("training_mode", False)):
        return True
    if node.op in STATEFUL_NN_OPS:
        return True
    return any(_truthy_metadata_value(node.attrs.get(name)) for name in SIDE_EFFECT_ATTRS)


def _truthy_metadata_value(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _inferred_verifier_outputs(node: NNNode, node_values: Sequence[NNTensorValue]) -> dict[str, NNTensorValue]:
    if not node.outputs or not node_values:
        return {}
    if len(node_values) != len(node.inputs):
        return {}
    if node.op in SPF32_BINARY_NN_OPS.union(SPF32_UNARY_NN_OPS):
        inferred_outputs = {node.outputs[0]: node_values[0].shape}
    elif node.op in VERIFIER_OUTPUT_INFER_OPS:
        inferred_outputs = _foundry_infer_node_outputs(node, node_values, graph=None)
    else:
        return {}
    outputs: dict[str, NNTensorValue] = {}
    for output_name in node.outputs:
        if output_name not in inferred_outputs:
            continue
        outputs[output_name] = NNTensorValue(
            value_id=f"verifier:{node.node_id}:{output_name}",
            name=output_name,
            dtype=_foundry_output_dtype(node, node_values, output_name),
            shape=tuple(int(dim) for dim in inferred_outputs[output_name]),
            residency="unknown",
            materialized=False,
            producer=node.node_id,
            metadata=dict(node_values[0].metadata),
        )
    return outputs


def _verify_node_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if node.op == "cast":
        target = _cast_target_dtype(node.attrs)
        if target is None:
            return "unsupported_cast_dtype"
        try:
            _numpy_dtype(target)
        except (TypeError, ValueError):
            return "unsupported_cast_dtype"
        return None
    if node.op == "cast_like":
        if len(node_values) != 2:
            return "dtype_mismatch:cast_like_input_count"
        try:
            _numpy_dtype(node_values[1].dtype)
        except (TypeError, ValueError):
            return "unsupported_cast_like_dtype"
        return None
    if node.op == "einsum":
        return _verify_einsum_dtype_constraints(node, node_values)
    if node.op == "dropout":
        return _verify_dropout_dtype_constraints(node, node_values)
    if node.op == "qlinear_matmul":
        return _verify_qlinear_matmul_dtype_constraints(node, node_values)
    if node.op == "matmul_integer":
        return _verify_matmul_integer_dtype_constraints(node, node_values)
    if node.op == "dynamic_quantize_linear":
        return _verify_dynamic_quantize_linear_dtype_constraints(node, node_values)
    if node.op in {"quantize_linear", "dequantize_linear"}:
        return _verify_qlinear_dtype_constraints(node, node_values)
    if node.op == "batchnorm":
        return _verify_batchnorm_dtype_constraints(node, node_values)
    if node.op == "bias_gelu":
        if len(node.inputs) != 2:
            return "dtype_mismatch:bias_gelu_input_count"
        if len(node_values) < 2:
            return None
        try:
            kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
            dtypes = {str(value.dtype) for value in node_values}
        except (TypeError, ValueError):
            return "dtype_mismatch:bias_gelu_float"
        if kinds != {"f"}:
            return "dtype_mismatch:bias_gelu_float"
        if len(dtypes) > 1:
            return "dtype_mismatch:bias_gelu"
        return None
    if node.op in FLOAT_ONLY_NN_OPS and node_values:
        try:
            input_kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
        except (TypeError, ValueError):
            return f"dtype_mismatch:{node.op}_float"
        if input_kinds != {"f"}:
            return f"dtype_mismatch:{node.op}_float"
        return None
    if node.op in {"neg", "abs", "reduce_sum", "reduce_min", "reduce_max", "reduce_prod"} and node_values:
        try:
            input_kind = _numpy_dtype(node_values[0].dtype).kind
        except (TypeError, ValueError):
            return f"dtype_mismatch:{node.op}_numeric"
        if input_kind not in {"i", "u", "f"}:
            return f"dtype_mismatch:{node.op}_numeric"
        return None
    if node.op in {"min", "max"} and node_values:
        return _verify_variadic_numeric_dtype_constraints(node, node_values)
    if node.op == "sum":
        return _verify_variadic_numeric_dtype_constraints(node, node_values)
    if node.op == "mean":
        if not node_values:
            return "dtype_mismatch:mean_input_count"
        try:
            kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
            dtypes = {str(value.dtype) for value in node_values}
        except (TypeError, ValueError):
            return "dtype_mismatch:mean_float"
        if kinds != {"f"}:
            return "dtype_mismatch:mean_float"
        if len(dtypes) > 1:
            return "dtype_mismatch:mean"
        return None
    if node.op == "mod" and len(node_values) >= 2:
        try:
            kinds = {_numpy_dtype(value.dtype).kind for value in node_values[:2]}
            dtypes = {str(value.dtype) for value in node_values[:2]}
        except (TypeError, ValueError):
            return "dtype_mismatch:mod_numeric"
        if not kinds.issubset({"i", "u", "f"}):
            return "dtype_mismatch:mod_numeric"
        if len(dtypes) > 1:
            return "dtype_mismatch:mod"
        return None
    if node.op == "clip":
        return _verify_clip_dtype_constraints(node, node_values)
    if node.op in SPF32_BINARY_NN_OPS.union(SPF32_UNARY_NN_OPS):
        if any(str(value.dtype) != "uint32" for value in node_values):
            return "dtype_mismatch:spf32_word"
        return None
    if node.op == "concat" and len(node_values) >= 2:
        dtypes = {str(value.dtype) for value in node_values}
        if len(dtypes) > 1:
            return "dtype_mismatch:concat"
        return None
    if node.op == "gather" and len(node_values) >= 2:
        try:
            index_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:gather_indices"
        if index_kind not in {"i", "u"}:
            return "dtype_mismatch:gather_indices"
        return None
    if node.op == "gather_elements" and len(node_values) >= 2:
        try:
            index_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:gather_elements_indices"
        if index_kind not in {"i", "u"}:
            return "dtype_mismatch:gather_elements_indices"
        return None
    if node.op == "gather_nd" and len(node_values) >= 2:
        try:
            index_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:gather_nd_indices"
        if index_kind not in {"i", "u"}:
            return "dtype_mismatch:gather_nd_indices"
        return None
    if node.op == "scatter_elements" and len(node_values) >= 3:
        try:
            index_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:scatter_elements_indices"
        if index_kind not in {"i", "u"}:
            return "dtype_mismatch:scatter_elements_indices"
        if str(node_values[0].dtype) != str(node_values[2].dtype):
            return "dtype_mismatch:scatter_elements_updates"
        return None
    if node.op == "scatter_nd" and len(node_values) >= 3:
        try:
            index_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:scatter_nd_indices"
        if index_kind not in {"i", "u"}:
            return "dtype_mismatch:scatter_nd_indices"
        if str(node_values[0].dtype) != str(node_values[2].dtype):
            return "dtype_mismatch:scatter_nd_updates"
        return None
    if node.op == "slice":
        for value in node_values[1:]:
            try:
                index_kind = _numpy_dtype(value.dtype).kind
            except (TypeError, ValueError):
                return "dtype_mismatch:slice_indices"
            if index_kind not in {"i", "u"}:
                return "dtype_mismatch:slice_indices"
        return None
    if node.op in {"unsqueeze", "squeeze"} and len(node_values) >= 2:
        try:
            axes_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return f"dtype_mismatch:{node.op}_axes"
        if axes_kind not in {"i", "u"}:
            return f"dtype_mismatch:{node.op}_axes"
        return None
    if node.op == "split" and len(node_values) >= 2:
        try:
            split_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:split_sizes"
        if split_kind not in {"i", "u"}:
            return "dtype_mismatch:split_sizes"
        return None
    if node.op == "expand" and len(node_values) >= 2:
        try:
            shape_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:expand_shape"
        if shape_kind not in {"i", "u"}:
            return "dtype_mismatch:expand_shape"
        return None
    if node.op == "tile":
        return _verify_tile_dtype_constraints(node, node_values)
    if node.op == "range":
        return _verify_range_dtype_constraints(node, node_values)
    if node.op == "constant_of_shape":
        return _verify_constant_of_shape_dtype_constraints(node, node_values)
    if node.op == "cumsum":
        return _verify_cumsum_dtype_constraints(node, node_values)
    if node.op == "pad":
        return _verify_pad_dtype_constraints(node, node_values)
    if node.op == "trilu":
        return _verify_trilu_dtype_constraints(node, node_values)
    if node.op == "where" and len(node_values) >= 3:
        try:
            condition_kind = _numpy_dtype(node_values[0].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:where_condition"
        if condition_kind != "b":
            return "dtype_mismatch:where_condition"
        branch_dtypes = {str(value.dtype) for value in node_values[1:3]}
        if len(branch_dtypes) > 1:
            return "dtype_mismatch:where_branches"
        return None
    if node.op in {"not", "and", "or"} and node_values:
        try:
            kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
        except (TypeError, ValueError):
            return f"dtype_mismatch:{node.op}_bool"
        if kinds != {"b"}:
            return f"dtype_mismatch:{node.op}_bool"
        return None
    if node.op in {"equal", "less", "less_equal", "greater", "greater_equal"} and len(node_values) >= 2:
        dtypes = {str(value.dtype) for value in node_values[:2]}
        if len(dtypes) > 1:
            return f"dtype_mismatch:{node.op}"
        return None
    if node.op in {"argmax", "argmin"} and node_values:
        try:
            input_kind = _numpy_dtype(node_values[0].dtype).kind
        except (TypeError, ValueError):
            return f"dtype_mismatch:{node.op}_input"
        if input_kind == "b":
            return f"dtype_mismatch:{node.op}_input"
        return None
    if node.op == "topk" and node_values:
        try:
            input_kind = _numpy_dtype(node_values[0].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:topk_input"
        if input_kind == "b":
            return "dtype_mismatch:topk_input"
        if len(node_values) >= 2:
            try:
                k_kind = _numpy_dtype(node_values[1].dtype).kind
            except (TypeError, ValueError):
                return "dtype_mismatch:topk_k"
            if k_kind not in {"i", "u"}:
                return "dtype_mismatch:topk_k"
        return None
    if node.op == "top_p_filter" and node_values:
        try:
            input_kind = _numpy_dtype(node_values[0].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:top_p_filter_input"
        if input_kind != "f":
            return "dtype_mismatch:top_p_filter_input"
        if len(node_values) >= 2:
            try:
                p_kind = _numpy_dtype(node_values[1].dtype).kind
            except (TypeError, ValueError):
                return "dtype_mismatch:top_p_filter_p"
            if p_kind != "f":
                return "dtype_mismatch:top_p_filter_p"
        return None
    if node.op == "kv_cache_append" and len(node_values) >= 2:
        dtypes = {str(value.dtype) for value in node_values[:2]}
        if len(dtypes) > 1:
            return "dtype_mismatch:kv_cache_append"
        return None
    if node.op not in {
        "matmul",
        "einsum",
        "gemm",
        "conv2d",
        "depthwise_conv2d",
        "add",
        "sub",
        "mul",
        "div",
        "pow",
        "quantize_linear",
        "dequantize_linear",
        "qlinear_matmul",
        "matmul_integer",
        "dynamic_quantize_linear",
        "spf32.mul",
        "spf32.add",
        "spf32.div",
        "spf32.abs2",
        "spf32.fft64",
        "attention_qkv",
        "attention_score",
        "attention_value",
    }:
        return None
    if node.op in QUANTIZED_LINEAR_OPS and _node_has_quantization(node_values):
        quantized_dtypes = {str(value.dtype) for value in node_values[:2]}
        if not quantized_dtypes.issubset({"int8", "uint8", "int16", "uint16", "int32", "uint32"}):
            return "dtype_mismatch:quantized_linear_inputs"
        for value in node_values[2:]:
            if str(value.dtype) not in {"float16", "float32", "float64", "int32"}:
                return "dtype_mismatch:quantized_linear_bias"
        return None
    dtypes = {str(value.dtype) for value in node_values}
    if len(dtypes) > 1:
        return "dtype_mismatch"
    return None


def _verify_qlinear_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 2:
        return f"dtype_mismatch:{node.op}_requires_scale"
    try:
        input_kind = _numpy_dtype(node_values[0].dtype).kind
        scale_kind = _numpy_dtype(node_values[1].dtype).kind
    except (TypeError, ValueError):
        return f"dtype_mismatch:{node.op}"
    if scale_kind != "f":
        return f"dtype_mismatch:{node.op}_scale"
    if node.op == "dequantize_linear":
        if input_kind not in {"i", "u"}:
            return "dtype_mismatch:dequantize_linear_input"
    else:
        if input_kind != "f":
            return "dtype_mismatch:quantize_linear_input"
    if len(node_values) >= 3:
        try:
            zero_kind = _numpy_dtype(node_values[2].dtype).kind
        except (TypeError, ValueError):
            return f"dtype_mismatch:{node.op}_zero_point"
        if zero_kind not in {"i", "u"}:
            return f"dtype_mismatch:{node.op}_zero_point"
    return None


def _verify_einsum_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if not node_values:
        return "dtype_mismatch:einsum_input_count"
    try:
        kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
        dtypes = {str(value.dtype) for value in node_values}
    except (TypeError, ValueError):
        return "dtype_mismatch:einsum"
    if kinds != {"f"}:
        return "dtype_mismatch:einsum_float"
    if len(dtypes) > 1:
        return "dtype_mismatch:einsum"
    return None


def _verify_batchnorm_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 5:
        return "dtype_mismatch:batchnorm_input_count"
    try:
        kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
        dtypes = {str(value.dtype) for value in node_values}
    except (TypeError, ValueError):
        return "dtype_mismatch:batchnorm"
    if kinds != {"f"}:
        return "dtype_mismatch:batchnorm_float"
    if len(dtypes) > 1:
        return "dtype_mismatch:batchnorm"
    return None


def _verify_range_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 3:
        return "dtype_mismatch:range_input_count"
    try:
        kinds = {_numpy_dtype(value.dtype).kind for value in node_values}
        dtypes = {str(value.dtype) for value in node_values}
    except (TypeError, ValueError):
        return "dtype_mismatch:range"
    if not kinds.issubset({"i", "u", "f"}):
        return "dtype_mismatch:range_numeric"
    if len(dtypes) > 1:
        return "dtype_mismatch:range"
    return None


def _verify_constant_of_shape_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 1:
        return "dtype_mismatch:constant_of_shape_input_count"
    try:
        input_kind = _numpy_dtype(node_values[0].dtype).kind
    except (TypeError, ValueError):
        return "dtype_mismatch:constant_of_shape_shape"
    if input_kind not in {"i", "u"}:
        return "dtype_mismatch:constant_of_shape_shape"
    value = _constant_of_shape_value_array(node.attrs)
    if value is None or int(value.size) != 1:
        return "dtype_mismatch:constant_of_shape_value"
    return None


def _verify_tile_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 2:
        return "dtype_mismatch:tile_input_count"
    try:
        repeats_kind = _numpy_dtype(node_values[1].dtype).kind
    except (TypeError, ValueError):
        return "dtype_mismatch:tile_repeats"
    if repeats_kind not in {"i", "u"}:
        return "dtype_mismatch:tile_repeats"
    return None


def _verify_cumsum_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if not node_values:
        return "dtype_mismatch:cumsum_input_count"
    try:
        input_kind = _numpy_dtype(node_values[0].dtype).kind
    except (TypeError, ValueError):
        return "dtype_mismatch:cumsum_input"
    if input_kind not in {"i", "u", "f"}:
        return "dtype_mismatch:cumsum_input"
    if len(node_values) >= 2:
        try:
            axis_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:cumsum_axis"
        if axis_kind not in {"i", "u"}:
            return "dtype_mismatch:cumsum_axis"
    return None


def _verify_trilu_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if not node_values:
        return "dtype_mismatch:trilu_input_count"
    if len(node_values) >= 2:
        try:
            k_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:trilu_k"
        if k_kind not in {"i", "u"}:
            return "dtype_mismatch:trilu_k"
    return None


def _verify_pad_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 1 or len(node_values) > 3:
        return "dtype_mismatch:pad_input_count"
    if len(node_values) >= 2:
        try:
            pads_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:pad_pads"
        if pads_kind not in {"i", "u"}:
            return "dtype_mismatch:pad_pads"
    if len(node_values) >= 3:
        if str(node_values[2].dtype) != str(node_values[0].dtype):
            return "dtype_mismatch:pad_value"
        if _shape_element_count(node_values[2].shape) != 1:
            return "dtype_mismatch:pad_value_scalar"
    return None


def _verify_variadic_numeric_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 1:
        return f"dtype_mismatch:{node.op}_input_count"
    dtypes = {str(value.dtype) for value in node_values}
    if len(dtypes) > 1:
        return f"dtype_mismatch:{node.op}"
    try:
        input_kind = _numpy_dtype(node_values[0].dtype).kind
    except (TypeError, ValueError):
        return f"dtype_mismatch:{node.op}_numeric"
    if input_kind not in {"i", "u", "f"}:
        return f"dtype_mismatch:{node.op}_numeric"
    return None


def _verify_clip_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if not node_values:
        return "dtype_mismatch:clip_input_count"
    try:
        input_kind = _numpy_dtype(node_values[0].dtype).kind
    except (TypeError, ValueError):
        return "dtype_mismatch:clip_input"
    if input_kind not in {"i", "u", "f"}:
        return "dtype_mismatch:clip_input"
    input_dtype = str(node_values[0].dtype)
    for bound in node_values[1:3]:
        try:
            bound_kind = _numpy_dtype(bound.dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:clip_bound"
        if bound_kind not in {"i", "u", "f"}:
            return "dtype_mismatch:clip_bound"
        if _shape_element_count(bound.shape) != 1:
            return "dtype_mismatch:clip_bound_scalar"
        if input_dtype != str(bound.dtype):
            return "dtype_mismatch:clip_bound"
    return None


def _verify_dropout_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 1 or len(node_values) > 3:
        return "dtype_mismatch:dropout_input_count"
    if len(node_values) >= 2:
        try:
            ratio_kind = _numpy_dtype(node_values[1].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:dropout_ratio"
        if ratio_kind != "f":
            return "dtype_mismatch:dropout_ratio"
    if len(node_values) >= 3:
        try:
            training_kind = _numpy_dtype(node_values[2].dtype).kind
        except (TypeError, ValueError):
            return "dtype_mismatch:dropout_training_mode"
        if training_kind != "b":
            return "dtype_mismatch:dropout_training_mode"
    return None


def _verify_qlinear_matmul_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 8:
        return "dtype_mismatch:qlinear_matmul_input_count"
    try:
        a_kind = _numpy_dtype(node_values[0].dtype).kind
        a_scale_kind = _numpy_dtype(node_values[1].dtype).kind
        a_zero_kind = _numpy_dtype(node_values[2].dtype).kind
        b_kind = _numpy_dtype(node_values[3].dtype).kind
        b_scale_kind = _numpy_dtype(node_values[4].dtype).kind
        b_zero_kind = _numpy_dtype(node_values[5].dtype).kind
        y_scale_kind = _numpy_dtype(node_values[6].dtype).kind
        y_zero_kind = _numpy_dtype(node_values[7].dtype).kind
    except (TypeError, ValueError):
        return "dtype_mismatch:qlinear_matmul"
    if a_kind not in {"i", "u"} or b_kind not in {"i", "u"}:
        return "dtype_mismatch:qlinear_matmul_inputs"
    if a_scale_kind != "f" or b_scale_kind != "f" or y_scale_kind != "f":
        return "dtype_mismatch:qlinear_matmul_scale"
    if a_zero_kind not in {"i", "u"} or b_zero_kind not in {"i", "u"} or y_zero_kind not in {"i", "u"}:
        return "dtype_mismatch:qlinear_matmul_zero_point"
    return None


def _verify_matmul_integer_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 2:
        return "dtype_mismatch:matmul_integer_input_count"
    try:
        a_kind = _numpy_dtype(node_values[0].dtype).kind
        b_kind = _numpy_dtype(node_values[1].dtype).kind
    except (TypeError, ValueError):
        return "dtype_mismatch:matmul_integer"
    if a_kind not in {"i", "u"} or b_kind not in {"i", "u"}:
        return "dtype_mismatch:matmul_integer_inputs"
    if len(node_values) >= 3:
        if str(node_values[2].dtype) != str(node_values[0].dtype):
            return "dtype_mismatch:matmul_integer_a_zero_point"
    if len(node_values) >= 4:
        if str(node_values[3].dtype) != str(node_values[1].dtype):
            return "dtype_mismatch:matmul_integer_b_zero_point"
    return None


def _verify_dynamic_quantize_linear_dtype_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 1:
        return "dtype_mismatch:dynamic_quantize_linear_input_count"
    if str(node_values[0].dtype) != "float32":
        return "dtype_mismatch:dynamic_quantize_linear_input"
    return None


def _node_has_quantization(node_values: Sequence[NNTensorValue]) -> bool:
    return any(bool(value.metadata.get("quantization")) for value in node_values)


def _verify_node_quantization_constraints(
    node: NNNode,
    node_values: Sequence[NNTensorValue],
    graph: NNGraph,
) -> Optional[str]:
    if node.op == "qlinear_matmul":
        return None
    if node.op == "matmul_integer":
        return None
    if node.op == "dynamic_quantize_linear":
        return None
    quantizations = [_quantization_for_value(value, graph=graph) for value in node_values]
    if node.op in QUANTIZED_LINEAR_OPS:
        for quantization in quantizations:
            if quantization and not _is_affine_quantization(quantization):
                return "unsupported_quantization:affine_required"
        return None
    signatures = {
        json.dumps(quantization, sort_keys=True, separators=(",", ":"))
        for quantization in quantizations
        if quantization
    }
    if len(signatures) > 1:
        return "quantization_mismatch"
    return None


def _quantization_for_value(value: NNTensorValue, *, graph: NNGraph) -> dict[str, Any]:
    quantization = value.metadata.get("quantization")
    if not quantization:
        spec = graph.tensors.get(value.name)
        quantization = None if spec is None else spec.quantization
    return dict(quantization or {})


def _is_affine_quantization(quantization: dict[str, Any]) -> bool:
    try:
        scale = float(quantization.get("scale"))
    except (TypeError, ValueError):
        return False
    if not math.isfinite(scale) or scale <= 0:
        return False
    zero_point = quantization.get("zero_point", 0)
    try:
        int(zero_point)
    except (TypeError, ValueError):
        return False
    scheme = str(quantization.get("scheme", "affine"))
    return scheme in {"affine", "asymmetric", "symmetric", "per_tensor_affine"}


def _verify_node_shape_constraints(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if node.op in {"add", "sub", "mul", "div", "pow", "mod", "sum", "mean", "min", "max", "bias_gelu"} and len(node_values) >= 2:
        return _broadcast_shape_failure(node, node_values)
    if node.op == "clip" and len(node_values) >= 1:
        return _clip_shape_failure(node, node_values)
    if node.op == "dropout" and len(node_values) >= 1:
        return _dropout_shape_failure(node, node_values)
    if node.op in {"equal", "less", "less_equal", "greater", "greater_equal", "and", "or"} and len(node_values) >= 2:
        return _broadcast_shape_failure(node, node_values[:2])
    if node.op == "where" and len(node_values) >= 3:
        return _broadcast_shape_failure(node, node_values[:3])
    if node.op in {"reduce_mean", "reduce_sum", "reduce_min", "reduce_max", "reduce_prod"} and len(node_values) >= 1:
        return _reduction_shape_failure(node, node_values)
    if node.op == "rmsnorm" and len(node_values) >= 1:
        return _rmsnorm_shape_failure(node, node_values)
    if node.op == "rotary_embedding" and len(node_values) >= 1:
        return _rotary_embedding_shape_failure(node, node_values)
    if node.op == "batchnorm" and len(node_values) >= 1:
        return _batchnorm_shape_failure(node, node_values)
    if node.op in {"softmax", "log_softmax"} and len(node_values) >= 1:
        rank = len(node_values[0].shape)
        if rank == 0 or _normalized_axis_value(node.attrs.get("axis", -1), rank) is None:
            return f"shape_mismatch:{node.op}_axis"
    if node.op in {"argmax", "argmin"} and len(node_values) >= 1:
        rank = len(node_values[0].shape)
        if rank == 0 or _normalized_axis_value(node.attrs.get("axis", 0), rank) is None:
            return f"shape_mismatch:{node.op}_axis"
    if node.op == "topk" and len(node_values) >= 1:
        return _topk_shape_failure(node, node_values)
    if node.op == "top_p_filter" and len(node_values) >= 1:
        return _top_p_filter_shape_failure(node, node_values)
    if node.op == "kv_cache_append" and len(node_values) >= 2:
        return _kv_cache_append_shape_failure(node, node_values)
    if node.op == "concat" and len(node_values) >= 1:
        return _concat_shape_failure(node, node_values)
    if node.op == "einsum":
        return _einsum_shape_failure(node, node_values)
    if node.op in {"matmul", "gemm"} and len(node_values) >= 2:
        left = _linear_operand_shape(node_values[0].shape, node.attrs.get("transA", 0) if node.op == "gemm" else 0)
        right = _linear_operand_shape(node_values[1].shape, node.attrs.get("transB", 0) if node.op == "gemm" else 0)
        if _matmul_output_shape(left, right) is None:
            return "shape_mismatch:matmul"
    if node.op == "qlinear_matmul" and len(node_values) >= 8:
        return _qlinear_matmul_shape_failure(node, node_values)
    if node.op == "matmul_integer" and len(node_values) >= 2:
        return _matmul_integer_shape_failure(node, node_values)
    if node.op == "dynamic_quantize_linear":
        if len(node.outputs) != 3:
            return "shape_mismatch:dynamic_quantize_linear_outputs"
    if node.op in SPF32_BINARY_NN_OPS and len(node_values) >= 2:
        if node_values[0].shape != node_values[1].shape:
            return "shape_mismatch:spf32_word"
    if node.op == "spf32.fft64" and len(node_values) >= 1:
        if node_values[0].elements != 64:
            return "shape_mismatch:spf32_fft64_elements"
    if node.op == "attention_qkv" and len(node_values) >= 2:
        left = node_values[0].shape
        right = node_values[1].shape
        if len(left) < 1 or len(right) < 2 or int(left[-1]) != int(right[-2]):
            return "shape_mismatch:attention_qkv"
        if int(right[-1]) % 3 != 0:
            return "shape_mismatch:attention_qkv_projection_not_divisible_by_three"
    if node.op == "attention_score" and len(node_values) >= 2:
        left = node_values[0].shape
        right = node_values[1].shape
        if not left or not right or int(left[-1]) != int(right[-1]):
            return "shape_mismatch:attention_score"
    if node.op == "attention_value" and len(node_values) >= 2:
        left = node_values[0].shape
        right = node_values[1].shape
        if not left or len(right) < 2 or int(left[-1]) != int(right[-2]):
            return "shape_mismatch:attention_value"
    if node.op == "layernorm" and len(node_values) >= 1:
        return _layernorm_shape_failure(node, node_values)
    if node.op in {"quantize_linear", "dequantize_linear"} and len(node_values) >= 2:
        return _qlinear_shape_failure(node, node_values)
    if node.op == "reshape":
        shape = _reshape_target_shape(node, node_values)
        if shape is None:
            return "dynamic_shape_transform_requires_ps"
        shape_reason = _verify_reshape_target_shape(node_values[0].shape, shape)
        if shape_reason is not None:
            return shape_reason
    if node.op == "shape":
        return _shape_op_shape_failure(node, node_values)
    if node.op == "size" and len(node_values) != 1:
        return "shape_mismatch:size_input_count"
    if node.op == "range":
        return _range_shape_failure(node, node_values)
    if node.op == "constant_of_shape":
        return _constant_of_shape_shape_failure(node, node_values)
    if node.op == "cumsum" and len(node_values) >= 1:
        return _cumsum_shape_failure(node, node_values)
    if node.op == "pad" and len(node_values) >= 1:
        return _pad_shape_failure(node, node_values)
    if node.op == "transpose" and len(node_values) >= 1:
        return _transpose_shape_failure(node, node_values)
    if node.op == "flatten" and len(node_values) >= 1:
        if _flatten_axis(node.attrs, len(node_values[0].shape)) is None:
            return "shape_mismatch:flatten_axis"
    if node.op == "unsqueeze" and len(node_values) >= 1:
        return _unsqueeze_shape_failure(node, node_values)
    if node.op == "squeeze" and len(node_values) >= 1:
        return _squeeze_shape_failure(node, node_values)
    if node.op == "expand" and len(node_values) >= 1:
        return _expand_shape_failure(node, node_values)
    if node.op == "tile" and len(node_values) >= 1:
        return _tile_shape_failure(node, node_values)
    if node.op == "trilu" and len(node_values) >= 1:
        return _trilu_shape_failure(node, node_values)
    if node.op == "split" and len(node_values) >= 1:
        return _split_shape_failure(node, node_values)
    if node.op == "gather" and len(node_values) >= 2:
        return _gather_shape_failure(node, node_values)
    if node.op == "gather_elements" and len(node_values) >= 2:
        return _gather_elements_shape_failure(node, node_values)
    if node.op == "gather_nd" and len(node_values) >= 2:
        return _gather_nd_shape_failure(node, node_values)
    if node.op == "scatter_elements" and len(node_values) >= 1:
        return _scatter_elements_shape_failure(node, node_values)
    if node.op == "scatter_nd" and len(node_values) >= 1:
        return _scatter_nd_shape_failure(node, node_values)
    if node.op == "slice" and len(node_values) >= 1:
        return _slice_shape_failure(node, node_values)
    if node.op in {"conv2d", "depthwise_conv2d"} and len(node_values) >= 2:
        return _conv_shape_failure(node, node_values)
    return None


def _broadcast_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    try:
        np = _numpy()
        np.broadcast_shapes(*(value.shape for value in node_values))
    except ValueError:
        return f"shape_mismatch:{node.op}"
    return None


def _broadcast_output_shape(node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    try:
        np = _numpy()
        return tuple(int(dim) for dim in np.broadcast_shapes(*(value.shape for value in node_values)))
    except ValueError:
        return None


def _gather_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    data_shape = tuple(int(dim) for dim in node_values[0].shape)
    rank = len(data_shape)
    axis = _normalized_axis_value(node.attrs.get("axis", 0), rank)
    if rank == 0 or axis is None:
        return "shape_mismatch:gather_axis"
    indices = node_values[1].array
    if indices is None:
        return None
    np = _numpy()
    arr = np.asarray(indices, dtype=np.int64)
    axis_dim = int(data_shape[axis])
    if int(arr.size) == 0:
        return None
    if bool(np.any(arr < -axis_dim)) or bool(np.any(arr >= axis_dim)):
        return "shape_mismatch:gather_indices"
    return None


def _gather_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if _gather_shape_failure(node, node_values) is not None:
        return None
    data_shape = tuple(int(dim) for dim in node_values[0].shape)
    indices_shape = tuple(int(dim) for dim in node_values[1].shape)
    axis = _normalized_axis_value(node.attrs.get("axis", 0), len(data_shape))
    if axis is None:
        return None
    return data_shape[:axis] + indices_shape + data_shape[axis + 1 :]


def _gather_elements_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    data_shape = tuple(int(dim) for dim in node_values[0].shape)
    indices_shape = tuple(int(dim) for dim in node_values[1].shape)
    rank = len(data_shape)
    if rank == 0 or len(indices_shape) != rank:
        return "shape_mismatch:gather_elements_rank"
    axis = _normalized_axis_value(node.attrs.get("axis", 0), rank)
    if axis is None:
        return "shape_mismatch:gather_elements_axis"
    for dim_index, (data_dim, index_dim) in enumerate(zip(data_shape, indices_shape)):
        if dim_index != axis and int(index_dim) > int(data_dim):
            return "shape_mismatch:gather_elements_indices"
    indices = node_values[1].array
    if indices is None:
        return None
    np = _numpy()
    arr = np.asarray(indices, dtype=np.int64)
    if tuple(int(dim) for dim in arr.shape) != indices_shape:
        return "shape_mismatch:gather_elements_indices"
    axis_dim = int(data_shape[axis])
    if int(arr.size) == 0:
        return None
    if bool(np.any(arr < -axis_dim)) or bool(np.any(arr >= axis_dim)):
        return "shape_mismatch:gather_elements_indices"
    return None


def _gather_nd_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        return "shape_mismatch:gather_nd_batch_dims"
    data_shape = tuple(int(dim) for dim in node_values[0].shape)
    data_rank = len(data_shape)
    indices_shape = tuple(int(dim) for dim in node_values[1].shape)
    if data_rank == 0 or not indices_shape:
        return "shape_mismatch:gather_nd_rank"
    index_depth = int(indices_shape[-1])
    if index_depth <= 0 or index_depth > data_rank:
        return "shape_mismatch:gather_nd_index_depth"
    indices = node_values[1].array
    if indices is not None:
        np = _numpy()
        arr = np.asarray(indices, dtype=np.int64)
        if tuple(int(dim) for dim in arr.shape) != indices_shape:
            return "shape_mismatch:gather_nd_indices"
        if int(arr.size) == 0:
            return None
        for dim_index in range(index_depth):
            axis_dim = int(data_shape[dim_index])
            component = arr[..., dim_index]
            if bool(np.any(component < -axis_dim)) or bool(np.any(component >= axis_dim)):
                return "shape_mismatch:gather_nd_indices"
    return None


def _gather_nd_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if _gather_nd_shape_failure(node, node_values) is not None:
        return None
    index_depth = int(node_values[1].shape[-1])
    return tuple(int(dim) for dim in node_values[1].shape[:-1]) + tuple(int(dim) for dim in node_values[0].shape[index_depth:])


def _scatter_elements_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 3:
        return "shape_mismatch:scatter_elements_input_count"
    reduction = node.attrs.get("reduction", "none")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8", errors="replace")
    if str(reduction).lower() not in {"", "none"}:
        return "unsupported_scatter_elements_reduction"
    data_shape = tuple(int(dim) for dim in node_values[0].shape)
    indices_shape = tuple(int(dim) for dim in node_values[1].shape)
    updates_shape = tuple(int(dim) for dim in node_values[2].shape)
    rank = len(data_shape)
    if rank == 0 or len(indices_shape) != rank or len(updates_shape) != rank:
        return "shape_mismatch:scatter_elements_rank"
    if updates_shape != indices_shape:
        return "shape_mismatch:scatter_elements_updates"
    axis = _normalized_axis_value(node.attrs.get("axis", 0), rank)
    if axis is None:
        return "shape_mismatch:scatter_elements_axis"
    for dim_index, (data_dim, index_dim) in enumerate(zip(data_shape, indices_shape)):
        if dim_index != axis and int(index_dim) > int(data_dim):
            return "shape_mismatch:scatter_elements_indices"
    if node_values[1].array is None:
        return None
    np = _numpy()
    indices = np.asarray(node_values[1].array, dtype=np.int64)
    if tuple(int(dim) for dim in indices.shape) != indices_shape:
        return "shape_mismatch:scatter_elements_indices"
    seen_targets: set[tuple[int, ...]] = set()
    axis_dim = int(data_shape[axis])
    for index_coord in np.ndindex(indices.shape):
        target = list(index_coord)
        for dim_index, target_index in enumerate(target):
            if dim_index != axis and int(target_index) >= int(data_shape[dim_index]):
                return "shape_mismatch:scatter_elements_indices"
        axis_index = int(indices[index_coord])
        if axis_index < 0:
            axis_index += axis_dim
        if axis_index < 0 or axis_index >= axis_dim:
            return "shape_mismatch:scatter_elements_indices"
        target[axis] = axis_index
        target_tuple = tuple(int(dim) for dim in target)
        if target_tuple in seen_targets:
            return "shape_mismatch:scatter_elements_duplicate_indices"
        seen_targets.add(target_tuple)
    return None


def _scatter_elements_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if _scatter_elements_shape_failure(node, node_values) is not None:
        return None
    return tuple(int(dim) for dim in node_values[0].shape)


def _scatter_nd_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 3:
        return "shape_mismatch:scatter_nd_input_count"
    reduction = node.attrs.get("reduction", "none")
    if isinstance(reduction, bytes):
        reduction = reduction.decode("utf-8", errors="replace")
    if str(reduction).lower() not in {"", "none"}:
        return "unsupported_scatter_nd_reduction"
    data_shape = tuple(int(dim) for dim in node_values[0].shape)
    indices_shape = tuple(int(dim) for dim in node_values[1].shape)
    updates_shape = tuple(int(dim) for dim in node_values[2].shape)
    data_rank = len(data_shape)
    if data_rank == 0 or not indices_shape:
        return "shape_mismatch:scatter_nd_rank"
    index_depth = int(indices_shape[-1])
    if index_depth <= 0 or index_depth > data_rank:
        return "shape_mismatch:scatter_nd_index_depth"
    expected_updates_shape = tuple(indices_shape[:-1]) + tuple(data_shape[index_depth:])
    if updates_shape != expected_updates_shape:
        return "shape_mismatch:scatter_nd_updates"
    if node_values[1].array is None:
        return None
    np = _numpy()
    indices = np.asarray(node_values[1].array, dtype=np.int64)
    if tuple(int(dim) for dim in indices.shape) != indices_shape:
        return "shape_mismatch:scatter_nd_indices"
    seen_targets: set[tuple[int, ...]] = set()
    for index_coord in np.ndindex(indices_shape[:-1]):
        raw_index = np.asarray(indices[index_coord], dtype=np.int64).reshape(-1)
        target = []
        for dim_index, raw_value in enumerate(raw_index.tolist()):
            axis_index = int(raw_value)
            dim_size = int(data_shape[dim_index])
            if axis_index < 0:
                axis_index += dim_size
            if axis_index < 0 or axis_index >= dim_size:
                return "shape_mismatch:scatter_nd_indices"
            target.append(axis_index)
        target_tuple = tuple(target)
        if target_tuple in seen_targets:
            return "shape_mismatch:scatter_nd_duplicate_indices"
        seen_targets.add(target_tuple)
    return None


def _scatter_nd_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if _scatter_nd_shape_failure(node, node_values) is not None:
        return None
    return tuple(int(dim) for dim in node_values[0].shape)


def _clip_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) > 3:
        return "shape_mismatch:clip_input_count"
    for bound in node_values[1:3]:
        if _shape_element_count(bound.shape) != 1:
            return "shape_mismatch:clip_bound_scalar"
    return None


def _dropout_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 1 or len(node_values) > 3:
        return "shape_mismatch:dropout_input_count"
    if len(node.outputs) not in {1, 2}:
        return "shape_mismatch:dropout_outputs"
    ratio = _dropout_ratio_from_sources(node.attrs, [value.array for value in node_values[1:2]])
    if ratio is None or not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
        return "shape_mismatch:dropout_ratio"
    training = _dropout_training_mode_from_sources(node.attrs, [value.array for value in node_values[2:3]])
    if training is None or training:
        return "shape_mismatch:dropout_training_mode"
    return None


def _topk_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    rank = len(node_values[0].shape)
    axis = _normalized_axis_value(node.attrs.get("axis", -1), rank)
    if rank == 0 or axis is None:
        return "shape_mismatch:topk_axis"
    k = _topk_k_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]])
    if k is None:
        return "shape_mismatch:topk_k"
    if k <= 0 or k > int(node_values[0].shape[axis]):
        return "shape_mismatch:topk_k"
    return None


def _argmax_output_shape(node: NNNode, input_shape: Sequence[int]) -> Optional[tuple[int, ...]]:
    rank = len(input_shape)
    axis = _normalized_axis_value(node.attrs.get("axis", 0), rank)
    if rank == 0 or axis is None:
        return None
    keepdims = _truthy_int(node.attrs.get("keepdims", 1))
    if keepdims:
        output_shape = list(int(dim) for dim in input_shape)
        output_shape[axis] = 1
        return tuple(output_shape)
    return tuple(int(dim) for index, dim in enumerate(input_shape) if index != axis)


def _top_p_filter_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    rank = len(node_values[0].shape)
    axis = _normalized_axis_value(node.attrs.get("axis", -1), rank)
    if rank == 0 or axis is None:
        return "shape_mismatch:top_p_filter_axis"
    p = _top_p_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]])
    if p is None or not math.isfinite(p) or p <= 0.0 or p > 1.0:
        return "shape_mismatch:top_p_filter_p"
    min_tokens = _safe_int(node.attrs.get("min_tokens_to_keep", 1), default=1)
    if min_tokens <= 0 or min_tokens > int(node_values[0].shape[axis]):
        return "shape_mismatch:top_p_filter_min_tokens"
    return None


def _kv_cache_append_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    cache_shape = tuple(int(dim) for dim in node_values[0].shape)
    update_shape = tuple(int(dim) for dim in node_values[1].shape)
    if len(cache_shape) != len(update_shape):
        return "shape_mismatch:kv_cache_append_rank"
    axis = _kv_cache_axis(node.attrs, len(cache_shape))
    if axis is None:
        return "shape_mismatch:kv_cache_append_axis"
    for index, (cache_dim, update_dim) in enumerate(zip(cache_shape, update_shape)):
        if index == axis:
            continue
        if int(cache_dim) != int(update_dim):
            return "shape_mismatch:kv_cache_append_dim"
    return None


def _topk_k_from_attrs_or_value(attrs: dict[str, Any], inputs: Sequence[Any]) -> Optional[int]:
    source = attrs.get("k", attrs.get("K"))
    if source is None and inputs:
        source = inputs[0]
    if source is None:
        return None
    try:
        np = _numpy()
        arr = np.asarray(source)
        if arr.size != 1:
            return None
        return int(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _top_p_from_attrs_or_value(attrs: dict[str, Any], inputs: Sequence[Any]) -> Optional[float]:
    source = attrs.get("p", attrs.get("top_p", attrs.get("threshold")))
    if source is None and inputs:
        source = inputs[0]
    if source is None:
        return None
    try:
        np = _numpy()
        arr = np.asarray(source)
        if arr.size != 1:
            return None
        return float(arr.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def _kv_cache_axis(attrs: dict[str, Any], rank: int) -> Optional[int]:
    default_axis = -2 if rank >= 2 else 0
    return _normalized_axis_value(attrs.get("axis", default_axis), rank)


def _qlinear_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    scale_shape = tuple(int(dim) for dim in node_values[1].shape)
    if _shape_element_count(scale_shape) == 1:
        expected_shape = scale_shape
    else:
        axis = _normalized_axis_value(node.attrs.get("axis", 1), len(input_shape))
        if axis is None:
            return f"shape_mismatch:{node.op}_axis"
        if len(scale_shape) != 1 or int(scale_shape[0]) != int(input_shape[axis]):
            return f"shape_mismatch:{node.op}_scale"
        expected_shape = scale_shape
    if len(node_values) >= 3:
        zero_shape = tuple(int(dim) for dim in node_values[2].shape)
        if _shape_element_count(zero_shape) != 1 and zero_shape != expected_shape:
            return f"shape_mismatch:{node.op}_zero_point"
    return None


def _qlinear_matmul_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    left = tuple(int(dim) for dim in node_values[0].shape)
    right = tuple(int(dim) for dim in node_values[3].shape)
    if _matmul_output_shape(left, right) is None:
        return "shape_mismatch:qlinear_matmul"
    for index, suffix in ((1, "a_scale"), (2, "a_zero_point"), (4, "b_scale"), (5, "b_zero_point"), (6, "y_scale"), (7, "y_zero_point")):
        if _shape_element_count(node_values[index].shape) != 1:
            return f"shape_mismatch:qlinear_matmul_{suffix}"
    return None


def _matmul_integer_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    left = tuple(int(dim) for dim in node_values[0].shape)
    right = tuple(int(dim) for dim in node_values[1].shape)
    if _matmul_output_shape(left, right) is None:
        return "shape_mismatch:matmul_integer"
    if len(node_values) >= 3 and not _matmul_integer_zero_point_shape_ok(node_values[2].shape, left, side="a"):
        return "shape_mismatch:matmul_integer_a_zero_point"
    if len(node_values) >= 4 and not _matmul_integer_zero_point_shape_ok(node_values[3].shape, right, side="b"):
        return "shape_mismatch:matmul_integer_b_zero_point"
    return None


def _matmul_integer_zero_point_shape_ok(
    zero_shape: Sequence[int],
    target_shape: Sequence[int],
    *,
    side: str,
) -> bool:
    zero = tuple(int(dim) for dim in zero_shape)
    target = tuple(int(dim) for dim in target_shape)
    if _shape_element_count(zero) == 1:
        return True
    if len(target) < 2:
        return False
    if side == "a" and len(zero) == 1:
        return int(zero[0]) == int(target[-2])
    if side == "b" and len(zero) == 1:
        return int(zero[0]) == int(target[-1])
    try:
        np = _numpy()
        np.broadcast_shapes(zero, target)
    except ValueError:
        return False
    return True


def _concat_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    rank = len(node_values[0].shape)
    axis = _concat_axis(node.attrs, rank)
    if axis is None:
        return "shape_mismatch:concat_axis"
    for value in node_values[1:]:
        if len(value.shape) != rank:
            return "shape_mismatch:concat_rank"
        for dim_index, (left, right) in enumerate(zip(node_values[0].shape, value.shape)):
            if dim_index == axis:
                continue
            if int(left) != int(right):
                return "shape_mismatch:concat_dim"
    return None


def _concat_axis(attrs: dict[str, Any], rank: int) -> Optional[int]:
    try:
        axis = int(attrs.get("axis", 0))
    except (TypeError, ValueError):
        return None
    if axis < 0:
        axis += rank
    if rank == 0 or axis < 0 or axis >= rank:
        return None
    return axis


def _concat_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values:
        return None
    if _concat_shape_failure(node, node_values) is not None:
        return None
    axis = _concat_axis(node.attrs, len(node_values[0].shape))
    if axis is None:
        return None
    output_shape = list(node_values[0].shape)
    output_shape[axis] = sum(int(value.shape[axis]) for value in node_values)
    return tuple(int(dim) for dim in output_shape)


def _reduction_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    rank = len(node_values[0].shape)
    axes = _reduction_axes_from_sources(node.attrs, [value.array for value in node_values[1:]], rank=rank)
    if axes is None:
        return "dynamic_shape_transform_requires_ps"
    seen_axes: set[int] = set()
    for axis in axes:
        if axis < 0 or axis >= rank:
            return f"shape_mismatch:{node.op}_axis"
        if axis in seen_axes:
            return f"shape_mismatch:{node.op}_duplicate_axis"
        seen_axes.add(axis)
    return None


def _reduction_output_shape(node: NNNode, input_shape: Sequence[int], axes: Sequence[int]) -> tuple[int, ...]:
    rank = len(input_shape)
    keepdims = bool(int(node.attrs.get("keepdims", 1)))
    axes_set = set(int(axis) for axis in axes)
    if keepdims:
        return tuple(1 if axis in axes_set else int(dim) for axis, dim in enumerate(input_shape))
    return tuple(int(dim) for axis, dim in enumerate(input_shape) if axis not in axes_set)


def _rmsnorm_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    rank = len(input_shape)
    if rank == 0:
        return "shape_mismatch:rmsnorm_axis"
    axis = _normalized_axis_value(node.attrs.get("axis", -1), rank)
    if axis is None:
        return "shape_mismatch:rmsnorm_axis"
    np = _numpy()
    for index, value in enumerate(node_values[1:3], start=1):
        try:
            broadcast_shape = np.broadcast_shapes(input_shape, tuple(int(dim) for dim in value.shape))
        except ValueError:
            return "shape_mismatch:rmsnorm_scale" if index == 1 else "shape_mismatch:rmsnorm_bias"
        if tuple(broadcast_shape) != input_shape:
            return "shape_mismatch:rmsnorm_scale" if index == 1 else "shape_mismatch:rmsnorm_bias"
    return None


def _layernorm_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    rank = len(input_shape)
    if rank == 0:
        return "shape_mismatch:layernorm_axis"
    axis = _normalized_axis_value(node.attrs.get("axis", -1), rank)
    if axis is None:
        return "shape_mismatch:layernorm_axis"
    normalized_shape = input_shape[axis:]
    if len(node_values) >= 2 and tuple(int(dim) for dim in node_values[1].shape) != normalized_shape:
        return "shape_mismatch:layernorm_scale"
    if len(node_values) >= 3 and tuple(int(dim) for dim in node_values[2].shape) != normalized_shape:
        return "shape_mismatch:layernorm_bias"
    return None


def _rotary_embedding_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    if not input_shape or input_shape[-1] % 2 != 0:
        return "shape_mismatch:rotary_embedding_dim"
    if len(node_values) < 3:
        return None
    for index, value in enumerate(node_values[1:3], start=1):
        table_shape = tuple(int(dim) for dim in value.shape)
        if not _rotary_table_shape_compatible(input_shape, table_shape):
            return "shape_mismatch:rotary_embedding_cos" if index == 1 else "shape_mismatch:rotary_embedding_sin"
    return None


def _rotary_table_shape_compatible(input_shape: Sequence[int], table_shape: Sequence[int]) -> bool:
    if _broadcast_shape_result(input_shape, table_shape) == tuple(input_shape):
        return True
    half_shape = (*tuple(input_shape[:-1]), int(input_shape[-1]) // 2)
    return _broadcast_shape_result(half_shape, table_shape) == half_shape


def _batchnorm_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 5:
        return "shape_mismatch:batchnorm_input_count"
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    if len(input_shape) < 2:
        return "shape_mismatch:batchnorm_rank"
    channel_count = int(input_shape[1])
    for index, value in enumerate(node_values[1:5], start=1):
        if tuple(int(dim) for dim in value.shape) != (channel_count,):
            return f"shape_mismatch:batchnorm_param_{index}"
    if len(node.outputs) != 1:
        return "shape_mismatch:batchnorm_outputs"
    if _truthy_int(node.attrs.get("training_mode", 0)) or _truthy_int(node.attrs.get("training", 0)):
        return "unsupported_batchnorm_training"
    return None


def _unsqueeze_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_rank = len(node_values[0].shape)
    axes = _axes_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]])
    if axes is None:
        return "dynamic_shape_transform_requires_ps"
    output_rank = input_rank + len(axes)
    seen_axes: set[int] = set()
    for axis in axes:
        normalized_axis = axis + output_rank if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= output_rank:
            return "shape_mismatch:unsqueeze_axis"
        if normalized_axis in seen_axes:
            return "shape_mismatch:unsqueeze_duplicate_axis"
        seen_axes.add(normalized_axis)
    return None


def _unsqueeze_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values or _unsqueeze_shape_failure(node, node_values) is not None:
        return None
    input_shape = list(int(dim) for dim in node_values[0].shape)
    axes = _axes_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]])
    if axes is None:
        return None
    output_rank = len(input_shape) + len(axes)
    output_shape = list(input_shape)
    for axis in sorted(axis + output_rank if axis < 0 else axis for axis in axes):
        output_shape.insert(axis, 1)
    return tuple(output_shape)


def _squeeze_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    rank = len(input_shape)
    axes = _axes_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]])
    if axes is None:
        axes = [axis for axis, dim in enumerate(input_shape) if dim == 1]
    seen_axes: set[int] = set()
    for axis in axes:
        normalized_axis = axis + rank if axis < 0 else axis
        if normalized_axis < 0 or normalized_axis >= rank:
            return "shape_mismatch:squeeze_axis"
        if normalized_axis in seen_axes:
            return "shape_mismatch:squeeze_duplicate_axis"
        if input_shape[normalized_axis] != 1:
            return "shape_mismatch:squeeze_dim"
        seen_axes.add(normalized_axis)
    return None


def _squeeze_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values or _squeeze_shape_failure(node, node_values) is not None:
        return None
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    rank = len(input_shape)
    axes = _axes_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]])
    if axes is None:
        return tuple(dim for dim in input_shape if dim != 1)
    normalized_axes = {axis + rank if axis < 0 else axis for axis in axes}
    return tuple(dim for index, dim in enumerate(input_shape) if index not in normalized_axes)


def _axes_from_attrs_or_value(attrs: dict[str, Any], sources: Sequence[Any]) -> Optional[list[int]]:
    axes_source = attrs.get("axes", sources[0] if sources else None)
    if axes_source is None:
        return None
    return _int_list_from_source(axes_source)


def _axis_from_attrs_or_value(attrs: dict[str, Any], sources: Sequence[Any], *, rank: int) -> Optional[int]:
    axis_source = attrs.get("axis", sources[0] if sources else None)
    if axis_source is None:
        return None
    try:
        axis_values = _numpy().asarray(axis_source).reshape(-1).tolist()
    except (TypeError, ValueError):
        return None
    if len(axis_values) != 1:
        return None
    try:
        axis = int(axis_values[0])
    except (TypeError, ValueError):
        return None
    return _normalized_axis_value(axis, rank)


def _expand_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    target_shape = _expand_target_shape(node.attrs, node_values)
    if target_shape is None:
        return "dynamic_shape_transform_requires_ps"
    if any(dim < 0 for dim in target_shape):
        return "shape_mismatch:expand_negative_dim"
    broadcast_shape = _broadcast_shape_result(node_values[0].shape, target_shape)
    if broadcast_shape != target_shape:
        return "shape_mismatch:expand"
    return None


def _expand_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values or _expand_shape_failure(node, node_values) is not None:
        return None
    return _expand_target_shape(node.attrs, node_values)


def _expand_target_shape(node_attrs: dict[str, Any], node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    sources = [node_values[1].array] if len(node_values) >= 2 and node_values[1].array is not None else ()
    return _expand_target_shape_from_sources(node_attrs, sources)


def _expand_target_shape_from_sources(attrs: dict[str, Any], sources: Sequence[Any]) -> Optional[tuple[int, ...]]:
    shape = attrs.get("shape", sources[0] if sources else None)
    if shape is None:
        return None
    sizes = _int_list_from_source(shape)
    if sizes is None:
        return None
    return tuple(int(dim) for dim in sizes)


def _tile_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 2:
        return "shape_mismatch:tile_input_count"
    if _tile_repeats_from_value(node_values[1], len(node_values[0].shape)) is None:
        return "shape_mismatch:tile_repeats"
    return None


def _tile_repeats_from_value(value: NNTensorValue, rank: int) -> Optional[tuple[int, ...]]:
    if value.array is None:
        return None
    try:
        repeats = tuple(int(item) for item in _numpy().asarray(value.array).reshape(-1).tolist())
    except (TypeError, ValueError):
        return None
    if len(repeats) != int(rank):
        return None
    if any(rep < 0 for rep in repeats):
        return None
    return repeats


def _split_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    rank = len(input_shape)
    axis = _normalized_axis_value(node.attrs.get("axis", 0), rank)
    if axis is None:
        return "shape_mismatch:split_axis"
    output_count = len(node.outputs)
    if output_count <= 0:
        return "shape_mismatch:split_outputs"
    sizes = _split_sizes_from_attrs_or_value(
        node.attrs,
        [value.array for value in node_values[1:]],
        axis_dim=input_shape[axis],
        output_count=output_count,
    )
    if sizes is None:
        return "shape_mismatch:split_sizes"
    if len(sizes) != output_count:
        return "shape_mismatch:split_output_count"
    if any(size <= 0 for size in sizes):
        return "shape_mismatch:split_sizes"
    if sum(sizes) != input_shape[axis]:
        return "shape_mismatch:split_total"
    return None


def _split_sizes_from_attrs_or_value(
    attrs: dict[str, Any],
    sources: Sequence[Any],
    *,
    axis_dim: int,
    output_count: int,
) -> Optional[list[int]]:
    if output_count <= 0:
        return None
    split_source = attrs.get("split", sources[0] if sources else None)
    if split_source is not None:
        return _int_list_from_source(split_source)
    try:
        num_outputs = int(attrs.get("num_outputs", output_count))
    except (TypeError, ValueError):
        return None
    if num_outputs != output_count or axis_dim % output_count != 0:
        return None
    return [axis_dim // output_count for _ in range(output_count)]


def _split_output_shapes(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[dict[str, tuple[int, ...]]]:
    if not node_values:
        return None
    if _split_shape_failure(node, node_values) is not None:
        return None
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    axis = _normalized_axis_value(node.attrs.get("axis", 0), len(input_shape))
    if axis is None:
        return None
    sizes = _split_sizes_from_attrs_or_value(
        node.attrs,
        [value.array for value in node_values[1:]],
        axis_dim=input_shape[axis],
        output_count=len(node.outputs),
    )
    if sizes is None or len(sizes) != len(node.outputs):
        return None
    output_shapes: dict[str, tuple[int, ...]] = {}
    for output_name, size in zip(node.outputs, sizes):
        shape = list(input_shape)
        shape[axis] = int(size)
        output_shapes[output_name] = tuple(shape)
    return output_shapes


def _normalized_axis_value(axis_value: Any, rank: int) -> Optional[int]:
    try:
        axis = int(axis_value)
    except (TypeError, ValueError):
        return None
    if axis < 0:
        axis += int(rank)
    if axis < 0 or axis >= int(rank):
        return None
    return axis


def _reduction_axes_from_sources(
    attrs: dict[str, Any],
    sources: Sequence[Any],
    *,
    rank: int,
) -> Optional[list[int]]:
    axes_source = attrs.get("axes", sources[0] if sources else None)
    if axes_source is None:
        return list(range(rank))
    axes = _int_list_from_source(axes_source)
    if axes is None:
        return None
    if not axes and _truthy_int(attrs.get("noop_with_empty_axes", 0)):
        return []
    return [axis + rank if axis < 0 else axis for axis in axes]


def _slice_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    rank = len(node_values[0].shape)
    params = _slice_params_from_sources(node.attrs, [value.array for value in node_values[1:]], rank=rank)
    if params is None:
        return "dynamic_shape_transform_requires_ps"
    starts, ends, axes, steps = params
    if not (len(starts) == len(ends) == len(axes) == len(steps)):
        return "shape_mismatch:slice_params"
    seen_axes: set[int] = set()
    for axis, step in zip(axes, steps):
        if axis < -rank or axis >= rank:
            return "shape_mismatch:slice_axis"
        normalized_axis = axis + rank if axis < 0 else axis
        if normalized_axis in seen_axes:
            return "shape_mismatch:slice_duplicate_axis"
        seen_axes.add(normalized_axis)
        if int(step) == 0:
            return "shape_mismatch:slice_step"
    return None


def _slice_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values:
        return None
    if _slice_shape_failure(node, node_values) is not None:
        return None
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    params = _slice_params_from_sources(node.attrs, [value.array for value in node_values[1:]], rank=len(input_shape))
    if params is None:
        return None
    starts, ends, axes, steps = params
    output_shape = list(input_shape)
    for start, end, axis, step in zip(starts, ends, axes, steps):
        dim = input_shape[int(axis)]
        normalized = slice(int(start), int(end), int(step)).indices(dim)
        output_shape[int(axis)] = len(range(*normalized))
    return tuple(output_shape)


def _shape_op_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 1:
        return "shape_mismatch:shape_input_count"
    if _shape_op_range(node.attrs, len(node_values[0].shape)) is None:
        return "shape_mismatch:shape_bounds"
    return None


def _range_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 3:
        return "shape_mismatch:range_input_count"
    if any(_shape_element_count(value.shape) != 1 for value in node_values):
        return "shape_mismatch:range_scalar"
    if _range_output_length(node_values) is None:
        return "shape_mismatch:range"
    return None


def _cumsum_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) not in {1, 2}:
        return "shape_mismatch:cumsum_input_count"
    rank = len(node_values[0].shape)
    if rank == 0:
        return "shape_mismatch:cumsum_axis"
    axis = _axis_from_attrs_or_value(node.attrs, [value.array for value in node_values[1:]], rank=rank)
    if axis is None:
        return "shape_mismatch:cumsum_axis"
    return None


def _constant_of_shape_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) != 1:
        return "shape_mismatch:constant_of_shape_input_count"
    if _constant_of_shape_output_shape(node_values[0]) is None:
        return "shape_mismatch:constant_of_shape"
    return None


def _trilu_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) not in {1, 2}:
        return "shape_mismatch:trilu_input_count"
    if len(node_values[0].shape) < 2:
        return "shape_mismatch:trilu_rank"
    if len(node_values) >= 2:
        k = _scalar_int_from_value(node_values[1])
        if k is None:
            return "shape_mismatch:trilu_k"
    return None


def _pad_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if len(node_values) < 1 or len(node_values) > 3:
        return "shape_mismatch:pad_input_count"
    if _pad_mode(node.attrs) != "constant":
        return "unsupported_pad_mode"
    if "axes" in node.attrs or len(node.inputs) > 3:
        return "unsupported_pad_axes"
    pads = _pad_values_from_sources(node.attrs, [value.array for value in node_values[1:2]], rank=len(node_values[0].shape))
    if pads is None:
        return "shape_mismatch:pad_pads"
    if any(value < 0 for value in pads):
        return "shape_mismatch:pad_negative"
    if len(node_values) >= 3 and _pad_constant_from_sources(node.attrs, [node_values[2].array]) is None:
        return "shape_mismatch:pad_value"
    return None


def _pad_output_shape(node: NNNode, input_shape: Sequence[int], sources: Sequence[Any]) -> Optional[tuple[int, ...]]:
    pads = _pad_values_from_sources(node.attrs, sources, rank=len(input_shape))
    if pads is None or any(value < 0 for value in pads):
        return None
    rank = len(input_shape)
    return tuple(int(dim) + int(pads[index]) + int(pads[index + rank]) for index, dim in enumerate(input_shape))


def _transpose_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    rank = len(node_values[0].shape)
    if _transpose_perm(node.attrs, rank) is None:
        return "shape_mismatch:transpose_perm"
    return None


def _transpose_perm(attrs: dict[str, Any], rank: int) -> Optional[list[int]]:
    perm = attrs.get("perm")
    if perm is None:
        return list(reversed(range(rank)))
    try:
        axes = [int(axis) for axis in list(perm)]
    except (TypeError, ValueError):
        return None
    if len(axes) != rank or sorted(axes) != list(range(rank)):
        return None
    return axes


def _transpose_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values:
        return None
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    perm = _transpose_perm(node.attrs, len(input_shape))
    if perm is None:
        return None
    return tuple(input_shape[axis] for axis in perm)


def _flatten_axis(attrs: dict[str, Any], rank: int) -> Optional[int]:
    try:
        axis = int(attrs.get("axis", 1))
    except (TypeError, ValueError):
        return None
    if axis < 0:
        axis += rank
    if axis < 0 or axis > rank:
        return None
    return axis


def _flatten_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values:
        return None
    input_shape = tuple(int(dim) for dim in node_values[0].shape)
    axis = _flatten_axis(node.attrs, len(input_shape))
    if axis is None:
        return None
    leading = _shape_element_count(input_shape[:axis])
    trailing = _shape_element_count(input_shape[axis:])
    return (leading, trailing)


def _slice_params_from_sources(
    attrs: dict[str, Any],
    sources: Sequence[Any],
    *,
    rank: int,
) -> Optional[tuple[list[int], list[int], list[int], list[int]]]:
    starts_source = attrs.get("starts", sources[0] if len(sources) >= 1 else None)
    ends_source = attrs.get("ends", sources[1] if len(sources) >= 2 else None)
    if starts_source is None or ends_source is None:
        return None
    starts = _int_list_from_source(starts_source)
    ends = _int_list_from_source(ends_source)
    if starts is None or ends is None:
        return None
    axes_source = attrs.get("axes", sources[2] if len(sources) >= 3 else None)
    steps_source = attrs.get("steps", sources[3] if len(sources) >= 4 else None)
    axes = _int_list_from_source(axes_source) if axes_source is not None else list(range(len(starts)))
    steps = _int_list_from_source(steps_source) if steps_source is not None else [1] * len(starts)
    if axes is None or steps is None:
        return None
    if len(axes) != len(starts) or len(steps) != len(starts):
        return starts, ends, axes, steps
    normalized_axes = [axis + rank if axis < 0 else axis for axis in axes]
    return starts, ends, normalized_axes, steps


def _int_list_from_source(source: Any) -> Optional[list[int]]:
    if source is None:
        return None
    try:
        arr = _numpy().asarray(source).reshape(-1)
        return [int(item) for item in arr.tolist()]
    except (TypeError, ValueError):
        return None


def _linear_operand_shape(shape: Sequence[int], transpose: Any) -> tuple[int, ...]:
    dims = tuple(int(dim) for dim in shape)
    if not _truthy_int(transpose) or len(dims) < 2:
        return dims
    return (*dims[:-2], dims[-1], dims[-2])


def _matmul_output_shape(left: Sequence[int], right: Sequence[int]) -> Optional[tuple[int, ...]]:
    if len(left) < 1 or len(right) < 2 or int(left[-1]) != int(right[-2]):
        return None
    try:
        np = _numpy()
        batch_shape = np.broadcast_shapes(tuple(int(dim) for dim in left[:-2]), tuple(int(dim) for dim in right[:-2]))
    except ValueError:
        return None
    if len(left) == 1:
        return (*tuple(int(dim) for dim in batch_shape), int(right[-1]))
    return (*tuple(int(dim) for dim in batch_shape), int(left[-2]), int(right[-1]))


def _attention_score_output_shape(node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if len(node_values) < 2:
        return None
    query_shape = tuple(int(dim) for dim in node_values[0].shape)
    key_shape = tuple(int(dim) for dim in node_values[1].shape)
    if len(query_shape) < 2 or len(key_shape) < 2:
        return None
    transposed_key_shape = (*key_shape[:-2], key_shape[-1], key_shape[-2])
    return _matmul_output_shape(query_shape, transposed_key_shape)


def _attention_value_output_shape(node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if len(node_values) < 2:
        return None
    return _matmul_output_shape(node_values[0].shape, node_values[1].shape)


def _einsum_equation(node: NNNode) -> Optional[str]:
    equation = node.attrs.get("equation")
    if isinstance(equation, bytes):
        equation = equation.decode("utf-8", errors="replace")
    if not isinstance(equation, str) or not equation.strip():
        return None
    return equation.strip()


def _einsum_output_shape(node: NNNode, input_shapes: Sequence[Sequence[int]]) -> Optional[tuple[int, ...]]:
    equation = _einsum_equation(node)
    if equation is None:
        return None
    try:
        np = _numpy()
        dummy_inputs = [
            np.zeros(tuple(int(dim) for dim in shape), dtype=np.float32)
            for shape in input_shapes
        ]
        return tuple(int(dim) for dim in np.einsum(equation, *dummy_inputs).shape)
    except (TypeError, ValueError):
        return None


def _einsum_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    if _einsum_equation(node) is None:
        return "shape_mismatch:einsum_equation"
    output_shape = _einsum_output_shape(node, [value.shape for value in node_values])
    if output_shape is None:
        return "shape_mismatch:einsum"
    return None


def _shape_op_range(attrs: dict[str, Any], rank: int) -> Optional[tuple[int, int]]:
    try:
        start = int(attrs.get("start", 0))
        end = int(attrs.get("end", rank))
    except (TypeError, ValueError):
        return None
    rank = int(rank)
    if rank < 0:
        return None
    if start < -rank or start > rank or end < -rank or end > rank:
        return None
    if start < 0:
        start += rank
    if end < 0:
        end += rank
    if start < 0 or start > rank or end < 0 or end > rank or end < start:
        return None
    return start, end


def _range_output_length(node_values: Sequence[NNTensorValue]) -> Optional[int]:
    scalars = _range_scalar_values(node_values)
    if scalars is None:
        return None
    start, limit, delta = scalars
    try:
        start_f = float(start)
        limit_f = float(limit)
        delta_f = float(delta)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(start_f) and math.isfinite(limit_f) and math.isfinite(delta_f)) or delta_f == 0.0:
        return None
    span = (limit_f - start_f) / delta_f
    if span <= 0.0:
        return 0
    return max(0, int(math.ceil(span)))


def _range_scalar_values(node_values: Sequence[NNTensorValue]) -> Optional[tuple[Any, Any, Any]]:
    if len(node_values) != 3:
        return None
    scalars: list[Any] = []
    for value in node_values:
        if value.array is None:
            return None
        arr = _numpy().asarray(value.array).reshape(-1)
        if int(arr.size) != 1:
            return None
        scalars.append(arr[0].item())
    return scalars[0], scalars[1], scalars[2]


def _scalar_int_from_value(value: NNTensorValue) -> Optional[int]:
    if value.array is None:
        return None
    try:
        arr = _numpy().asarray(value.array).reshape(-1)
    except (TypeError, ValueError):
        return None
    if int(arr.size) != 1:
        return None
    try:
        return int(arr[0].item())
    except (TypeError, ValueError):
        return None


def _constant_of_shape_value_array(attrs: dict[str, Any]) -> Optional[Any]:
    np = _numpy()
    value = attrs.get("value", 0.0)
    dtype = attrs.get("dtype")
    if "value" not in attrs and dtype is None:
        dtype = "float32"
    if isinstance(value, dict):
        value_dtype = value.get("dtype", dtype or "float32")
        try:
            return np.asarray(value.get("data", value.get("value", 0)), dtype=_numpy_dtype(str(value_dtype))).reshape(-1)
        except (TypeError, ValueError):
            return None
    try:
        arr = np.asarray(value, dtype=(_numpy_dtype(str(dtype)) if dtype else None)).reshape(-1)
    except (TypeError, ValueError):
        return None
    if int(arr.size) == 0:
        return None
    return arr


def _constant_of_shape_scalar(attrs: dict[str, Any]) -> Optional[Any]:
    value = _constant_of_shape_value_array(attrs)
    if value is None or int(value.size) != 1:
        return None
    return value[0]


def _constant_of_shape_output_shape(shape_value: NNTensorValue) -> Optional[tuple[int, ...]]:
    if shape_value.array is None:
        return None
    try:
        dims = [int(dim) for dim in _numpy().asarray(shape_value.array).reshape(-1).tolist()]
    except (TypeError, ValueError):
        return None
    if any(dim < 0 for dim in dims):
        return None
    return tuple(dims)


def _truthy_int(value: Any) -> bool:
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return False


def _is_static_reshape_shape(shape: Any) -> bool:
    if not isinstance(shape, (list, tuple)):
        return False
    negative_one_count = 0
    for dim in shape:
        if not isinstance(dim, int):
            return False
        if dim == -1:
            negative_one_count += 1
        elif dim < 0:
            return False
    return negative_one_count <= 1


def _reshape_target_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    shape = node.attrs.get("shape")
    if shape is None and len(node_values) >= 2:
        shape_value = node_values[1]
        if shape_value.array is None:
            return None
        shape = shape_value.array.reshape(-1).tolist()
    if not _is_static_reshape_shape(shape):
        return None
    return tuple(int(dim) for dim in shape)


def _verify_reshape_target_shape(input_shape: Sequence[int], target_shape: Sequence[int]) -> Optional[str]:
    input_elements = _shape_element_count(input_shape)
    known_elements = 1
    negative_one_count = 0
    for dim in target_shape:
        dim = int(dim)
        if dim == -1:
            negative_one_count += 1
            continue
        if dim < 0:
            return "shape_mismatch:reshape_negative_dim"
        known_elements *= dim
    if negative_one_count > 1:
        return "shape_mismatch:reshape_multiple_inferred_dims"
    if negative_one_count == 1:
        if known_elements == 0 or input_elements % known_elements != 0:
            return "shape_mismatch:reshape_element_count"
        return None
    if known_elements != input_elements:
        return "shape_mismatch:reshape_element_count"
    return None


def _reshape_output_shape(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    if not node_values:
        return None
    target_shape = _reshape_target_shape(node, node_values)
    if target_shape is None or _verify_reshape_target_shape(node_values[0].shape, target_shape) is not None:
        return None
    if -1 not in target_shape:
        return tuple(int(dim) for dim in target_shape)
    input_elements = _shape_element_count(node_values[0].shape)
    known_elements = 1
    for dim in target_shape:
        if int(dim) != -1:
            known_elements *= int(dim)
    if known_elements == 0 or input_elements % known_elements != 0:
        return None
    inferred_dim = input_elements // known_elements
    return tuple(int(inferred_dim) if int(dim) == -1 else int(dim) for dim in target_shape)


def _shape_element_count(shape: Sequence[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _conv_shape_failure(node: NNNode, node_values: Sequence[NNTensorValue]) -> Optional[str]:
    x_shape = node_values[0].shape
    w_shape = node_values[1].shape
    if len(x_shape) != 4:
        return f"shape_mismatch:{node.op}_input_rank"
    if len(w_shape) != 4:
        return f"shape_mismatch:{node.op}_weight_rank"
    attrs = _conv_normalized_attrs(node.attrs)
    if attrs["error"] is not None:
        return attrs["error"]
    strides = attrs["strides"]
    pads = attrs["pads"]
    dilations = attrs["dilations"]
    group = int(attrs["group"])
    c_in = int(x_shape[1])
    c_out = int(w_shape[0])
    if node.op == "conv2d":
        if group != 1:
            return "unsupported_grouped_conv2d"
        if int(w_shape[1]) != c_in:
            return "shape_mismatch:conv2d_channels"
    else:
        if int(w_shape[1]) != 1:
            return "shape_mismatch:depthwise_conv2d_weight_channels"
        if group not in {1, c_in}:
            return "shape_mismatch:depthwise_conv2d_group"
        if c_out % c_in != 0:
            return "shape_mismatch:depthwise_conv2d_multiplier"
    if len(node_values) >= 3:
        bias_shape = node_values[2].shape
        if len(bias_shape) != 1 or int(bias_shape[0]) != c_out:
            return f"shape_mismatch:{node.op}_bias"
    out_h = _conv_output_dim(int(x_shape[2]), int(w_shape[2]), pads[0], pads[2], strides[0], dilations[0])
    out_w = _conv_output_dim(int(x_shape[3]), int(w_shape[3]), pads[1], pads[3], strides[1], dilations[1])
    if out_h <= 0 or out_w <= 0:
        return f"shape_mismatch:{node.op}_output"
    return None


def _conv_output_dim(input_size: int, kernel: int, pad_before: int, pad_after: int, stride: int, dilation: int) -> int:
    effective_kernel = dilation * (kernel - 1) + 1
    return (input_size + pad_before + pad_after - effective_kernel) // stride + 1


def _conv_normalized_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    auto_pad = str(attrs.get("auto_pad", "NOTSET")).upper()
    if auto_pad not in {"", "NOTSET"}:
        return {"error": "unsupported_conv_auto_pad", "strides": (1, 1), "pads": (0, 0, 0, 0), "dilations": (1, 1), "group": 1}
    strides = _int_tuple_attr(attrs.get("strides", (1, 1)), expected=2, default=(1, 1))
    pads = _int_tuple_attr(attrs.get("pads", (0, 0, 0, 0)), expected=4, default=(0, 0, 0, 0))
    dilations = _int_tuple_attr(attrs.get("dilations", (1, 1)), expected=2, default=(1, 1))
    group = _safe_int(attrs.get("group", 1), default=1)
    if strides is None or any(value <= 0 for value in strides):
        return {"error": "invalid_conv_strides", "strides": (1, 1), "pads": (0, 0, 0, 0), "dilations": (1, 1), "group": group}
    if pads is None or any(value < 0 for value in pads):
        return {"error": "invalid_conv_pads", "strides": (1, 1), "pads": (0, 0, 0, 0), "dilations": (1, 1), "group": group}
    if dilations is None or any(value <= 0 for value in dilations):
        return {"error": "invalid_conv_dilations", "strides": (1, 1), "pads": (0, 0, 0, 0), "dilations": (1, 1), "group": group}
    if group <= 0:
        return {"error": "invalid_conv_group", "strides": strides, "pads": pads, "dilations": dilations, "group": group}
    return {"error": None, "strides": strides, "pads": pads, "dilations": dilations, "group": group}


def _int_tuple_attr(value: Any, *, expected: int, default: tuple[int, ...]) -> Optional[tuple[int, ...]]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError:
            return None
    if len(items) != expected:
        return None
    try:
        return tuple(int(item) for item in items)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _constraint_failure(reason: str, *, node: NNNode, checks: Sequence[str], inputs: Optional[Sequence[str]] = None) -> dict[str, Any]:
    return {
        "schema": "pan_nn_constraint_verification_v1",
        "ok": False,
        "reason": reason,
        "node_id": node.node_id,
        "op": node.op,
        "inputs": list(inputs or node.inputs),
        "checks": list(checks),
    }


def _select_region_action(
    region: NNRegion,
    *,
    backend: str,
    values: dict[str, NNTensorValue],
    graph: NNGraph,
    foundry_auto_queue: bool,
    foundry_catalog: Optional[str] = None,
    foundry_target: str = "kv260",
    policy: Optional[FabricPolicyHead] = None,
) -> dict[str, Any]:
    constraints = _verify_region_constraints(region, values, graph)
    if not constraints["ok"]:
        action = _reject_action(region, str(constraints["reason"]))
        action["constraint_verification"] = constraints
        return action
    action = _select_region_rule_action(
        region,
        backend=backend,
        values=values,
        foundry_auto_queue=foundry_auto_queue,
    )
    action["constraint_verification"] = constraints
    action = _apply_catalog_hint_for_region(
        region,
        action,
        backend=backend,
        graph=graph,
        values=values,
        foundry_catalog=foundry_catalog,
        foundry_target=foundry_target,
    )
    if policy is None:
        return action
    return _apply_nn_policy(
        region,
        action,
        backend=backend,
        values=values,
        foundry_auto_queue=foundry_auto_queue,
        policy=policy,
    )


def _select_region_rule_action(
    region: NNRegion,
    *,
    backend: str,
    values: dict[str, NNTensorValue],
    foundry_auto_queue: bool,
) -> dict[str, Any]:
    ps_supported = all(op in PS_SUPPORTED_OPS for op in region.ops)
    sim_eligible = all(op in FABRIC_SIM_ELIGIBLE_OPS for op in region.ops)
    session_eligible = all(op in FABRIC_SESSION_ELIGIBLE_OPS for op in region.ops)
    input_residencies = sorted({values[name].residency for node in region.nodes for name in node.inputs if name in values})
    if backend == "ps":
        if not ps_supported:
            return _reject_action(region, "ps_oracle_unsupported_op")
        return _placement_action(region, "run_ps_now", "ps_authority", input_residencies)
    if backend == "auto":
        if (
            session_eligible
            and _region_elements(region, values) >= 1024
            and "daemon" in input_residencies
        ):
            return _placement_action(region, "run_daemon_resident_now", "auto_resident_inputs", input_residencies)
        if ps_supported:
            action = _placement_action(region, "run_ps_now", "auto_ps_fallback", input_residencies)
            if foundry_auto_queue and sim_eligible and _region_should_queue_foundry(region, values):
                action["queued_followup_action"] = "compile_better_for_next_time"
            return action
        return _reject_action(region, "unsupported_region")
    if backend == "sim":
        if not sim_eligible:
            return _reject_action(region, "explicit_sim_unsupported_region")
        action = _placement_action(region, "run_daemon_resident_now", "simulated_fabric_region", input_residencies)
        if foundry_auto_queue and sim_eligible:
            action["queued_followup_action"] = "compile_better_for_next_time"
        return action
    if backend == "fabric-session":
        if not session_eligible:
            return _reject_action(region, "explicit_fabric_session_unsupported_region")
        action = _placement_action(region, "run_daemon_resident_now", "daemon_session_region", input_residencies)
        if foundry_auto_queue:
            action["queued_followup_action"] = "compile_better_for_next_time"
        return action
    return _reject_action(region, f"unknown_backend:{backend}")


def _region_should_queue_foundry(region: NNRegion, values: dict[str, NNTensorValue]) -> bool:
    if region.reason in {
        "fused_mlp_region",
        "fused_gated_mlp_region",
        "fused_decode_output_region",
        "fused_decode_sampling_region",
    }:
        return True
    if _region_elements(region, values) >= 1024:
        return True
    return any(op in LLM_FOUNDRY_SEED_OPS for op in region.ops)


def _apply_catalog_hint_for_region(
    region: NNRegion,
    action: dict[str, Any],
    *,
    backend: str,
    graph: NNGraph,
    values: dict[str, NNTensorValue],
    foundry_catalog: Optional[str],
    foundry_target: str,
) -> dict[str, Any]:
    if backend != "auto" or not foundry_catalog or action.get("selected_action") == "reject_fail_closed":
        return action
    fabric_graph = _foundry_graph_for_nn_region(region, values, graph=graph)
    hint = scheduler_hint_for_graph(fabric_graph, catalog=FabricFoundryCatalog(foundry_catalog), target=foundry_target)
    action["scheduler_hint"] = hint
    if hint.get("selected_action") == "run_daemon_resident_now" and _region_can_lower_to_fabric_session(region):
        daemon_action = _placement_action(
            region,
            "run_daemon_resident_now",
            "measured_resident_daemon",
            action.get("input_residencies", []),
        )
        daemon_action["rule_selected_action"] = action.get("selected_action")
        daemon_action["rule_reason"] = action.get("reason")
        daemon_action["scheduler_hint"] = hint
        daemon_action["constraint_verification"] = action.get("constraint_verification")
        if hint.get("compile_better_for_next_time") or "compile_better_for_next_time" in set(hint.get("action_plan") or ()):
            daemon_action["queued_followup_action"] = "compile_better_for_next_time"
        return daemon_action
    if hint.get("selected_action") != "run_catalog_core_now":
        return action
    catalog_action = _placement_action(
        region,
        "run_catalog_core_now",
        "measured_catalog_core",
        action.get("input_residencies", []),
    )
    catalog_action["rule_selected_action"] = action.get("selected_action")
    catalog_action["rule_reason"] = action.get("reason")
    catalog_action["scheduler_hint"] = hint
    catalog_action["constraint_verification"] = action.get("constraint_verification")
    if action.get("queued_followup_action"):
        catalog_action["queued_followup_action"] = action["queued_followup_action"]
    return catalog_action


def _apply_nn_policy(
    region: NNRegion,
    action: dict[str, Any],
    *,
    backend: str,
    values: dict[str, NNTensorValue],
    foundry_auto_queue: bool,
    policy: FabricPolicyHead,
) -> dict[str, Any]:
    features = _nn_policy_features_for_region(region, action, values)
    decision = {
        "schema": "pan_nn_policy_decision_v1",
        "policy": _policy_summary(policy),
        "score": float(policy.score(features)),
        "features": features,
        "placement_action": _policy_majority_class(policy, "placement_action"),
        "preferred_residency": (
            _policy_majority_class(policy, "preferred_residency")
            or _policy_majority_class(policy, "output_residency")
        ),
        "materialization_needed": _policy_materialization_needed(policy),
        "compile_priority": _policy_majority_class(policy, "compile_priority"),
        "latency_estimate": _policy_numeric_head(policy, "latency_estimate"),
        "transfer_estimate": _policy_numeric_head(policy, "transfer_estimate"),
        "energy_estimate": _policy_numeric_head(policy, "energy_estimate"),
        "resource_estimate": _policy_numeric_head(policy, "resource_estimate"),
        "reconfiguration_estimate": _policy_numeric_head(policy, "reconfiguration_estimate"),
        "transfer_avoidance_estimate": _policy_numeric_head(policy, "transfer_avoidance_estimate"),
        "validation_risk": _policy_numeric_head(policy, "validation_risk"),
        "override_allowed": bool(policy.can_override_rule_scheduler()),
        "override_applied": False,
        "confidence": _policy_confidence(policy),
    }
    action["policy_decision"] = decision
    action["policy_score"] = decision["score"]
    action["neural_policy_override"] = False
    if not policy.can_override_rule_scheduler():
        action["candidate_policy"] = decision["policy"]
        decision["rejected_reason"] = "policy_not_promoted"
        return action
    if backend != "auto":
        decision["rejected_reason"] = "explicit_backend_authority"
        return action
    if action.get("selected_action") == "reject_fail_closed":
        decision["rejected_reason"] = "rule_rejected_region"
        return action
    if decision["confidence"] < float(policy.min_confidence):
        decision["rejected_reason"] = "policy_confidence_below_threshold"
        return action
    risk = decision.get("validation_risk")
    if isinstance(risk, dict) and float(risk.get("risk_rate") or 0.0) > 0.0:
        decision["rejected_reason"] = "policy_validation_risk_nonzero"
        return action

    ps_supported = all(op in PS_SUPPORTED_OPS for op in region.ops)
    sim_eligible = all(op in FABRIC_SIM_ELIGIBLE_OPS for op in region.ops)
    input_residencies = sorted(
        {values[name].residency for node in region.nodes for name in node.inputs if name in values}
    )
    desired = decision.get("placement_action")
    if desired == "run_ps_now":
        if not ps_supported:
            decision["rejected_reason"] = "policy_requested_ps_unsupported"
            return action
        _override_action(action, "run_ps_now", "neural_policy_promoted")
        decision["override_applied"] = True
        action["neural_policy_override"] = True
    elif desired == "run_daemon_resident_now":
        if not sim_eligible:
            decision["rejected_reason"] = "policy_requested_daemon_unsupported"
            return action
        if not _region_can_lower_to_fabric_session(region):
            decision["rejected_reason"] = "policy_requested_daemon_unavailable"
            decision["required_lowering"] = "fabric-session"
            return action
        if "daemon" not in input_residencies:
            decision["rejected_reason"] = "policy_requested_daemon_inputs_not_resident"
            decision["required_input_residency"] = "daemon"
            decision["promotion_scope_note"] = (
                "neural daemon authority is scoped to resident-input (hot-cache) regimes; "
                "cold runs fall through to the rule path"
            )
            return action
        _override_action(action, "run_daemon_resident_now", "neural_policy_promoted")
        decision["override_applied"] = True
        action["neural_policy_override"] = True
    elif desired == "run_catalog_core_now":
        hint = action.get("scheduler_hint")
        if not isinstance(hint, dict) or hint.get("selected_action") != "run_catalog_core_now":
            decision["rejected_reason"] = "policy_requested_catalog_core_unavailable"
            decision["required_measurement"] = "measured_or_promoted_catalog_core"
            return action
        _override_action(action, "run_catalog_core_now", "neural_policy_promoted_catalog_core")
        decision["override_applied"] = True
        action["neural_policy_override"] = True
    elif desired in {None, "unknown", "none"}:
        decision["rejected_reason"] = "policy_no_supported_placement"
    elif desired == "compile_better_for_next_time":
        decision["placement_note"] = "compile_priority_is_followup_not_runtime_action"
    else:
        decision["rejected_reason"] = f"policy_action_unavailable:{desired}"

    if decision.get("compile_priority") == "compile_better_for_next_time":
        if foundry_auto_queue:
            action["queued_followup_action"] = "compile_better_for_next_time"
            decision["compile_followup_applied"] = True
        else:
            decision["compile_followup_rejected_reason"] = "foundry_auto_queue_disabled"
    return action


def _override_action(action: dict[str, Any], selected_action: str, reason: str) -> None:
    action.setdefault("rule_selected_action", action.get("selected_action"))
    action.setdefault("rule_reason", action.get("reason"))
    action["selected_action"] = selected_action
    action["reason"] = reason


def _policy_summary(policy: FabricPolicyHead) -> dict[str, Any]:
    head_keys = {
        "kind",
        "classes",
        "positive_count",
        "failure_count",
        "sample_count",
        "mean_seconds",
        "mean_bytes",
        "risk_rate",
    }
    return {
        "policy_kind": policy.policy_kind,
        "policy_role": policy.policy_role,
        "promotion_status": policy.promotion_status,
        "can_override_rule_scheduler": policy.can_override_rule_scheduler(),
        "min_confidence": policy.min_confidence,
        "heads": {
            name: {key: value for key, value in dict(head).items() if key in head_keys}
            for name, head in sorted(policy.heads.items())
            if isinstance(head, dict)
        },
    }


def _policy_majority_class(policy: FabricPolicyHead, head_name: str) -> Optional[str]:
    head = policy.heads.get(head_name)
    if not isinstance(head, dict):
        return None
    classes = head.get("classes")
    if not isinstance(classes, dict) or not classes:
        return None
    ordered = sorted(((str(name), int(count)) for name, count in classes.items()), key=lambda item: (-item[1], item[0]))
    return ordered[0][0]


def _policy_materialization_needed(policy: FabricPolicyHead) -> Optional[bool]:
    head = policy.heads.get("materialization_needed")
    if not isinstance(head, dict):
        return None
    sample_count = int(head.get("sample_count") or 0)
    if sample_count <= 0:
        return None
    positive_count = int(head.get("positive_count") or 0)
    return positive_count >= (sample_count / 2.0)


def _policy_numeric_head(policy: FabricPolicyHead, head_name: str) -> Optional[dict[str, Any]]:
    head = policy.heads.get(head_name)
    if not isinstance(head, dict):
        return None
    return {
        key: value
        for key, value in head.items()
        if key in {
            "kind",
            "mean_seconds",
            "mean_bytes",
            "mean_avoided_bytes",
            "mean_joules",
            "mean_lut",
            "mean_dsp",
            "mean_bram",
            "mean_uram",
            "mean_milliseconds",
            "failure_count",
            "risk_rate",
            "sample_count",
        }
    }


def _policy_confidence(policy: FabricPolicyHead) -> float:
    value = policy.heldout_metrics.get("confidence")
    if value is None:
        evaluation = policy.heldout_metrics.get("promotion_evaluation")
        if isinstance(evaluation, dict):
            metrics = evaluation.get("metrics")
            if isinstance(metrics, dict):
                value = metrics.get("confidence")
    if value is None:
        return 1.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _nn_policy_features_for_region(
    region: NNRegion,
    action: dict[str, Any],
    values: dict[str, NNTensorValue],
) -> dict[str, float]:
    input_names = sorted({name for node in region.nodes for name in node.inputs if name in values})
    input_residencies = [values[name].residency for name in input_names]
    tensor_elements = [values[name].elements for name in input_names]
    ops = region.ops
    tensor_features = _nn_tensor_metadata_features(_nn_region_tensor_metadata(region, values))
    foundry_features = _nn_policy_foundry_estimate_features(region, values)
    mlp_features = _mlp_feature_counts(ops, reason=region.reason)
    attention_features = _attention_feature_counts(ops, reason=region.reason)
    decode_features = _decode_feature_counts(ops)
    conv_region = region.reason == "fused_conv_region" or any(op in {"conv2d", "depthwise_conv2d"} for op in ops)
    return {
        "op_count": float(len(ops)),
        "matmul_count": float(sum(1 for op in ops if op in {"matmul", "gemm"})),
        "conv_count": float(sum(1 for op in ops if op in {"conv2d", "depthwise_conv2d"})),
        "conv_region_count": float(conv_region),
        "activation_count": float(
            sum(1 for op in ops if op in {"relu", "gelu", "bias_gelu", "quick_gelu", "silu", "sigmoid", "tanh", "erf", "sin", "cos", "softmax", "log_softmax"})
        ),
        "spf_count": float(sum(1 for op in ops if op.startswith("spf"))),
        **mlp_features,
        **attention_features,
        **decode_features,
        "max_tensor_elements": float(max(tensor_elements) if tensor_elements else 0),
        "total_tensor_elements": float(sum(tensor_elements)),
        "input_daemon_residency_count": float(sum(1 for value in input_residencies if value == "daemon")),
        "input_ps_residency_count": float(sum(1 for value in input_residencies if value == "ps")),
        "output_daemon_residency_count": 0.0,
        "output_ps_residency_count": 0.0,
        "elapsed_seconds": 0.0,
        "setup_transfer_bytes": 0.0,
        "timed_transfer_bytes": 0.0,
        "forced_materialization_bytes": 0.0,
        "foundry_queued": 0.0,
        "selected_run_ps": float(action.get("selected_action") == "run_ps_now"),
        "selected_daemon_resident": float(action.get("selected_action") == "run_daemon_resident_now"),
        **foundry_features,
        **tensor_features,
    }


def _nn_policy_foundry_estimate_features(region: NNRegion, values: dict[str, NNTensorValue]) -> dict[str, float]:
    try:
        graph = _foundry_graph_for_nn_region(region, values)
        ops = tuple(op.op for op in graph.ops)
        if not ops:
            return _nn_foundry_estimate_features(None)
        elements = int(graph.ops[0].elements)
        region_seed = json.dumps({"ops": ops, "elements": elements, "signature": region.signature()}, sort_keys=True)
        foundry_region = FabricRegion(
            "policy_estimate_" + hashlib.sha256(region_seed.encode("utf-8")).hexdigest()[:12],
            ops,
            elements,
            graph=graph,
        )
        candidates = generate_candidates([foundry_region])
    except Exception:
        return _nn_foundry_estimate_features(None)
    if not candidates:
        return _nn_foundry_estimate_features(None)
    return _nn_foundry_estimate_features({"candidate": candidates[0].to_json()})


def _placement_action(region: NNRegion, selected_action: str, reason: str, input_residencies: Sequence[str]) -> dict[str, Any]:
    return {
        "schema": "pan_nn_placement_action_v1",
        "selected_action": selected_action,
        "reason": reason,
        "region_signature": region.signature(),
        "ops": list(region.ops),
        "input_residencies": list(input_residencies),
        "confidence": 1.0,
    }


def _reject_action(region: NNRegion, reason: str) -> dict[str, Any]:
    return {
        "schema": "pan_nn_placement_action_v1",
        "selected_action": "reject_fail_closed",
        "reason": reason,
        "region_signature": region.signature(),
        "ops": list(region.ops),
        "input_residencies": [],
        "confidence": 1.0,
    }


def _execute_region_ps(
    graph: NNGraph,
    values: dict[str, NNTensorValue],
    value_store: NNTensorValueStore,
    region: NNRegion,
    *,
    action: dict[str, Any],
    spf32_engine: Optional[_PSOpEngine] = None,
) -> None:
    arrays = {name: value.array for name, value in values.items() if value.array is not None}
    for node in region.nodes:
        _execute_node(arrays, node, tensor_specs=graph.tensors, tensor_values=values, spf32_engine=spf32_engine)
        for output_name in node.outputs:
            arr = arrays[output_name]
            if action["selected_action"] == "run_daemon_resident_now":
                residency = "daemon"
            elif action["selected_action"] == "run_catalog_core_now":
                residency = "catalog_core"
            else:
                residency = "ps"
            materialized = action["selected_action"] == "run_ps_now"
            values[output_name] = value_store.add_array(
                output_name,
                arr,
                dtype=str(arr.dtype),
                residency=residency,
                materialized=materialized,
                producer=node.node_id,
                metadata={
                    "region_id": region.region_id,
                    "selected_action": action["selected_action"],
                    "quantization": _output_quantization_for_node(node, values, output_name=output_name, arrays=arrays),
                    **_shape_metadata_for_array(graph.tensors.get(output_name), arr, role="output"),
                },
            )


def _region_can_lower_to_fabric_session(region: NNRegion) -> bool:
    return bool(region.nodes) and all(node.op in FABRIC_SESSION_ELIGIBLE_OPS for node in region.nodes)


def _output_quantization_for_node(
    node: NNNode,
    values: dict[str, NNTensorValue],
    *,
    output_name: Optional[str] = None,
    arrays: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    if node.op == "topk" and len(node.outputs) >= 2 and output_name == node.outputs[1]:
        return {}
    if node.op == "top_p_filter" and len(node.outputs) >= 2 and output_name == node.outputs[1]:
        return {}
    if isinstance(node.attrs.get("output_quantization"), dict):
        return dict(node.attrs["output_quantization"])
    if node.op == "dequantize_linear":
        return {}
    if node.op == "quantize_linear":
        return _quantization_from_qlinear_node(node, values)
    if node.op == "qlinear_matmul":
        return _quantization_from_qlinear_matmul_node(node, values)
    if node.op == "dynamic_quantize_linear":
        return _quantization_from_dynamic_quantize_linear_node(node, arrays or {}, output_name=output_name)
    if node.op in QUANTIZED_LINEAR_OPS:
        return {}
    for input_name in node.inputs:
        value = values.get(input_name)
        if value is None:
            continue
        quantization = value.metadata.get("quantization")
        if isinstance(quantization, dict) and quantization:
            return dict(quantization)
    return {}


def _quantization_from_qlinear_node(node: NNNode, values: dict[str, NNTensorValue]) -> dict[str, Any]:
    if len(node.inputs) < 2:
        return {}
    scale_value = values.get(node.inputs[1])
    zero_value = values.get(node.inputs[2]) if len(node.inputs) >= 3 else None
    if scale_value is None or scale_value.array is None:
        return {}
    np = _numpy()
    scale = np.asarray(scale_value.array)
    zero = np.asarray(0 if zero_value is None or zero_value.array is None else zero_value.array)
    dtype = str(zero.dtype) if zero_value is not None and zero_value.array is not None else str(node.attrs.get("dtype", "uint8"))
    if scale.size != 1 or zero.size not in {0, 1}:
        return {
            "scheme": "per_axis_affine",
            "axis": int(node.attrs.get("axis", 1)),
            "dtype": dtype,
            "scale_elements": int(scale.size),
            "zero_point_elements": int(zero.size),
        }
    return {
        "scheme": "affine",
        "scale": float(scale.reshape(-1)[0]),
        "zero_point": int(zero.reshape(-1)[0]) if zero.size else 0,
        "dtype": dtype,
    }


def _quantization_from_qlinear_matmul_node(node: NNNode, values: dict[str, NNTensorValue]) -> dict[str, Any]:
    if len(node.inputs) < 8:
        return {}
    scale_value = values.get(node.inputs[6])
    zero_value = values.get(node.inputs[7])
    if scale_value is None or scale_value.array is None or zero_value is None or zero_value.array is None:
        return {}
    np = _numpy()
    scale = np.asarray(scale_value.array)
    zero = np.asarray(zero_value.array)
    if scale.size != 1 or zero.size != 1:
        return {}
    return {
        "scheme": "affine",
        "scale": float(scale.reshape(-1)[0]),
        "zero_point": int(zero.reshape(-1)[0]),
        "dtype": str(zero.dtype),
    }


def _quantization_from_dynamic_quantize_linear_node(
    node: NNNode,
    arrays: dict[str, Any],
    *,
    output_name: Optional[str],
) -> dict[str, Any]:
    if len(node.outputs) < 3 or output_name != node.outputs[0]:
        return {}
    scale = arrays.get(node.outputs[1])
    zero = arrays.get(node.outputs[2])
    if scale is None or zero is None:
        return {}
    np = _numpy()
    scale_arr = np.asarray(scale)
    zero_arr = np.asarray(zero)
    if scale_arr.size != 1 or zero_arr.size != 1:
        return {}
    return {
        "scheme": "affine",
        "scale": float(scale_arr.reshape(-1)[0]),
        "zero_point": int(zero_arr.reshape(-1)[0]),
        "dtype": "uint8",
    }


def _execute_region_fabric_session(
    graph: NNGraph,
    values: dict[str, NNTensorValue],
    value_store: NNTensorValueStore,
    region: NNRegion,
    *,
    fabric_daemon: str,
    workdir: Optional[str],
    xclbin_dirs: Optional[Sequence[str]],
    active_app: str,
    fabric_backend: str,
    daemon_buffer_storage: str,
    materialize_outputs: bool,
    materialization_reason: Optional[str],
    fabric_session_cache: Optional[dict[str, FabricRuntimeSession]] = None,
    fabric_plan_cache: Optional[dict[str, _NNFabricPlanCacheEntry]] = None,
    resolved_active_app: Optional[str] = None,
    selected_action: str = "run_daemon_resident_now",
) -> dict[str, Any]:
    fabric_graph, initial_inputs, output_shapes = _fabric_graph_for_spf_region(
        region,
        values,
        materialize_outputs=materialize_outputs,
    )
    planning_started = time.perf_counter()
    resolved_app = str(resolved_active_app or (active_xmutil_app() if active_app == "auto" else active_app) or "unknown")
    plan_key = _nn_fabric_plan_cache_key(
        fabric_graph,
        workdir=workdir,
        xclbin_dirs=xclbin_dirs,
        active_app=resolved_app,
        fabric_backend=fabric_backend,
        daemon_buffer_storage=daemon_buffer_storage,
        fabric_daemon=fabric_daemon,
        materialize_outputs=materialize_outputs,
    )
    plan_cache_hit = False
    entry = fabric_plan_cache.get(plan_key) if fabric_plan_cache is not None else None
    if entry is None:
        registry = FabricRegistry.discover(
            workdir=workdir or DEFAULT_FABRIC_WORKDIR,
            xclbin_dirs=xclbin_dirs,
        )
        planner = FabricGraphPlanner(
            registry,
            backend=fabric_backend,
            active_app=resolved_app,
        )
        entry = _NNFabricPlanCacheEntry(
            fabric_graph=fabric_graph,
            plan=planner.plan(fabric_graph),
            output_shapes=output_shapes,
        )
        if fabric_plan_cache is not None:
            fabric_plan_cache[plan_key] = entry
    else:
        plan_cache_hit = True
    planning_elapsed = time.perf_counter() - planning_started
    plan = entry.plan
    output_shapes = entry.output_shapes
    if fabric_session_cache is not None:
        session = _cached_fabric_runtime_session(
            fabric_session_cache,
            plan=plan,
            fabric_daemon=fabric_daemon,
            daemon_buffer_storage=daemon_buffer_storage,
            workdir=workdir,
            materialize_outputs=materialize_outputs,
        )
        payload = _execute_region_cached_fabric_session(
            graph,
            values,
            value_store,
            region,
            fabric_graph=fabric_graph,
            fabric_session=session,
            initial_inputs=initial_inputs,
            output_shapes=output_shapes,
            materialize_outputs=materialize_outputs,
            materialization_reason=materialization_reason,
            selected_action=selected_action,
        )
        _annotate_nn_fabric_plan_cache(
            payload,
            cache_key=plan_key,
            cache_hit=plan_cache_hit,
            planning_elapsed_seconds=planning_elapsed,
            resolved_active_app=resolved_app,
        )
        return payload
    executor = FabricGraphExecutor(
        plan,
        execution_backend="fabric-session",
        initial_inputs=initial_inputs,
        daemon_socket=fabric_daemon,
        daemon_buffer_storage=daemon_buffer_storage,
        materialize_outputs=materialize_outputs,
        workdir=workdir,
    )
    result = executor.execute(repeats=1)
    if not result.ok:
        raise RuntimeError(result.error or "fabric-session NN region execution failed")
    for output_name in fabric_graph.outputs:
        fabric_value = dict(result.fabric_values.get(output_name) or {})
        output_shape = output_shapes[output_name]
        if fabric_value.get("residency") == "daemon" and fabric_value.get("daemon_handle"):
            values[output_name] = value_store.add_daemon(
                output_name,
                dict(fabric_value["daemon_handle"]),
                dtype=str(fabric_value.get("dtype") or "uint32"),
                shape=output_shape,
                producer=region.nodes[-1].node_id,
                validation=fabric_value.get("validation"),
                metadata={
                    "region_id": region.region_id,
                    "selected_action": selected_action,
                    "fabric_value_id": fabric_value.get("value_id"),
                    "fabric_execution_summary": result.summary(),
                },
            )
            continue
        digest = dict(result.outputs.get(output_name) or fabric_value.get("ps_digest") or {})
        if not digest:
            raise RuntimeError(f"fabric-session output {output_name!r} did not include a digest or daemon handle")
        values[output_name] = value_store.add_digest_value(
            output_name,
            digest,
            dtype=str(fabric_value.get("dtype") or digest.get("dtype") or "uint32"),
            shape=output_shape,
            residency=str(fabric_value.get("residency") or "ps"),
            materialized=bool(fabric_value.get("materialized", True)),
            producer=region.nodes[-1].node_id,
            validation=fabric_value.get("validation"),
            materialization_reason=materialization_reason,
            daemon_handle=fabric_value.get("daemon_handle"),
            metadata={
                "region_id": region.region_id,
                "selected_action": selected_action,
                "fabric_value_id": fabric_value.get("value_id"),
                "fabric_execution_summary": result.summary(),
            },
        )
    payload = result.to_json(include_trace=True)
    _annotate_nn_fabric_plan_cache(
        payload,
        cache_key=plan_key,
        cache_hit=plan_cache_hit,
        planning_elapsed_seconds=planning_elapsed,
        resolved_active_app=resolved_app,
    )
    return payload


def _cached_fabric_runtime_session(
    cache: dict[str, FabricRuntimeSession],
    *,
    plan: Any,
    fabric_daemon: str,
    daemon_buffer_storage: str,
    workdir: Optional[str],
    materialize_outputs: bool,
) -> FabricRuntimeSession:
    key_payload = {
        "schema": "pan_nn_fabric_daemon_session_cache_key_v1",
        "daemon": str(fabric_daemon),
        "daemon_buffer_storage": str(daemon_buffer_storage),
        "workdir": str(workdir or ""),
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()
    session = cache.get(key)
    executor = FabricGraphExecutor(
        plan,
        execution_backend="fabric-session",
        initial_inputs={},
        daemon_socket=fabric_daemon,
        daemon_buffer_storage=daemon_buffer_storage,
        materialize_outputs=materialize_outputs,
        workdir=workdir,
    )
    if session is not None:
        session.executor = executor
        return session
    session = FabricRuntimeSession(executor, session_id=f"nn-fabric-{key[:12]}")
    session.__enter__()
    cache[key] = session
    return session


def _execute_region_cached_fabric_session(
    graph: NNGraph,
    values: dict[str, NNTensorValue],
    value_store: NNTensorValueStore,
    region: NNRegion,
    *,
    fabric_graph: FabricGraph,
    fabric_session: FabricRuntimeSession,
    initial_inputs: dict[str, Any],
    output_shapes: dict[str, tuple[int, ...]],
    materialize_outputs: bool,
    materialization_reason: Optional[str],
    selected_action: str = "run_daemon_resident_now",
) -> dict[str, Any]:
    del graph
    trace_start = len(fabric_session.trace)
    packet_inputs = {
        name: _fabric_session_input_for_tensor(fabric_session, values[name])
        for name in initial_inputs
    }
    _, daemon_result, output_values = fabric_session.execute_graph_packet(
        packet_inputs,
        output_names=fabric_graph.outputs,
        repeat_index=0,
        step_index=0,
        phase="timed",
        materialize_outputs=materialize_outputs,
        note=f"nn_region:{region.region_id}",
        materialization_reason=materialization_reason if materialize_outputs else None,
    )
    new_trace = fabric_session.trace[trace_start:]
    for output_name in fabric_graph.outputs:
        fabric_value = output_values[output_name]
        output_shape = output_shapes[output_name]
        if fabric_value.is_daemon_resident():
            values[output_name] = value_store.add_daemon(
                output_name,
                dict(fabric_value.daemon_handle or {}),
                dtype=str(fabric_value.dtype or "uint32"),
                shape=output_shape,
                producer=region.nodes[-1].node_id,
                validation=fabric_value.validation,
                metadata={
                    "region_id": region.region_id,
                    "selected_action": selected_action,
                    "fabric_runtime": _fabric_runtime_metadata(fabric_session, fabric_value, owned=True),
                    "fabric_execution_summary": _fabric_trace_summary(new_trace),
                },
            )
            continue
        if fabric_value.ps_array is None:
            raise RuntimeError(f"fabric-session output {output_name!r} was neither daemon-resident nor materialized")
        values[output_name] = value_store.add_array(
            output_name,
            fabric_value.ps_array,
            dtype=str(fabric_value.dtype or "uint32"),
            residency=fabric_value.residency,
            materialized=True,
            producer=region.nodes[-1].node_id,
            validation=fabric_value.validation,
            materialization_reason=materialization_reason,
            daemon_handle=fabric_value.daemon_handle,
            metadata={
                "region_id": region.region_id,
                "selected_action": selected_action,
                "fabric_runtime": _fabric_runtime_metadata(fabric_session, fabric_value, owned=True),
                "fabric_execution_summary": _fabric_trace_summary(new_trace),
            },
        )
    return {
        "schema": "pan_fabric_execution_v1",
        "ok": True,
        "execution_backend": "fabric-session",
        "repeats": 1,
        "elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0.0) for row in new_trace),
        "summary": _fabric_trace_summary(new_trace),
        "graph": fabric_graph.to_json(),
        "outputs": {
            name: (tensor_digest(value.ps_array) if value.ps_array is not None else {})
            for name, value in output_values.items()
        },
        "fabric_values": {
            name: value.to_json()
            for name, value in output_values.items()
        },
        "value_store": fabric_session.value_store.to_json(),
        "trace": list(new_trace),
        "daemon_result": daemon_result.to_json(),
    }


def _fabric_session_input_for_tensor(session: FabricRuntimeSession, value: NNTensorValue) -> Any:
    _validate_nn_value_residency(
        value.name,
        residency=value.residency,
        daemon_handle=value.daemon_handle,
        shape=value.shape,
        materialized=bool(value.array is not None or value.materialized),
    )
    metadata = value.metadata.get("fabric_runtime")
    if isinstance(metadata, dict) and metadata.get("session_id") == session.value_store.session_id:
        fabric_value_id = metadata.get("value_id")
        if fabric_value_id:
            try:
                session.value_store.get(str(fabric_value_id))
                return str(fabric_value_id)
            except KeyError:
                pass
    if value.daemon_handle is not None and value.residency == "daemon":
        fabric_value = session.value_store.create_daemon(
            value.name,
            dict(value.daemon_handle),
            producer=f"nn_input:{value.value_id}",
            validation=value.validation,
            metadata={"nn_value_id": value.value_id, "external_daemon_handle": True},
        )
        value.metadata["fabric_runtime"] = _fabric_runtime_metadata(session, fabric_value, owned=False)
        return fabric_value.value_id
    if value.array is None:
        raise ValueError(f"NN tensor {value.name!r} is not materialized and has no daemon handle")
    fabric_value = session.import_ps_value(
        value.name,
        value.array,
        producer=f"nn_input:{value.value_id}",
        phase="setup",
    )
    value.residency = "daemon"
    value.daemon_handle = dict(fabric_value.daemon_handle or {})
    value.metadata["fabric_runtime"] = _fabric_runtime_metadata(session, fabric_value, owned=True)
    return fabric_value.value_id


def _fabric_runtime_metadata(session: FabricRuntimeSession, fabric_value: Any, *, owned: bool) -> dict[str, Any]:
    return {
        "schema": "pan_nn_fabric_runtime_ref_v1",
        "session_id": session.value_store.session_id,
        "value_id": fabric_value.value_id,
        "owned": bool(owned),
    }


def _nn_fabric_plan_cache_key(
    fabric_graph: FabricGraph,
    *,
    workdir: Optional[str],
    xclbin_dirs: Optional[Sequence[str]],
    active_app: str,
    fabric_backend: str,
    daemon_buffer_storage: str,
    fabric_daemon: str,
    materialize_outputs: bool,
) -> str:
    payload = {
        "schema": "pan_nn_fabric_plan_cache_key_v1",
        "graph": fabric_graph.to_json(),
        "workdir": str(workdir or DEFAULT_FABRIC_WORKDIR),
        "xclbin_dirs": [str(path) for path in (xclbin_dirs or ())],
        "active_app": str(active_app),
        "fabric_backend": str(fabric_backend),
        "daemon_buffer_storage": str(daemon_buffer_storage),
        "fabric_daemon": str(fabric_daemon),
        "materialize_outputs": bool(materialize_outputs),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _annotate_nn_fabric_plan_cache(
    payload: dict[str, Any],
    *,
    cache_key: str,
    cache_hit: bool,
    planning_elapsed_seconds: float,
    resolved_active_app: str,
) -> None:
    payload["plan_cache"] = {
        "schema": "pan_nn_fabric_plan_cache_v1",
        "key": str(cache_key),
        "hit": bool(cache_hit),
        "planning_elapsed_seconds": float(planning_elapsed_seconds),
        "resolved_active_app": str(resolved_active_app),
    }
    summary = dict(payload.get("summary") or {})
    summary["planning_elapsed_seconds"] = float(planning_elapsed_seconds)
    summary["plan_cache_hit_count"] = 1 if cache_hit else 0
    summary["plan_cache_miss_count"] = 0 if cache_hit else 1
    payload["summary"] = summary


def _fabric_trace_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    timed_rows = [row for row in rows if row.get("phase") == "timed" or row.get("timed")]
    setup_rows = [row for row in rows if row.get("phase") == "setup"]
    materialize_rows = [row for row in rows if row.get("phase") == "materialize"]
    timed_input_bytes = sum(
        int(((row.get("daemon_result") or {}).get("input_bytes")) or 0)
        for row in timed_rows
    )
    timed_output_bytes = sum(
        int(((row.get("daemon_result") or {}).get("output_bytes")) or 0)
        for row in timed_rows
    )
    return {
        "step_count": len(rows),
        "run_daemon_graph_count": sum(1 for row in rows if row.get("action") == "run_daemon_graph"),
        "setup_transfer_bytes": sum(int(row.get("observed_transfer_bytes") or 0) for row in setup_rows),
        "timed_transfer_bytes": timed_input_bytes,
        "final_transfer_bytes": timed_output_bytes
        + sum(int(row.get("observed_transfer_bytes") or 0) for row in materialize_rows),
        "observed_transfer_bytes": sum(int(row.get("observed_transfer_bytes") or 0) for row in rows),
        "timed_graph_packets": len(timed_rows),
        "timed_elapsed_seconds": sum(float(row.get("elapsed_seconds") or 0.0) for row in timed_rows),
        "timed_daemon_elapsed_seconds": sum(float(row.get("daemon_elapsed_seconds") or 0.0) for row in timed_rows),
        "observed_run_daemon_seconds": sum(
            float(row.get("elapsed_seconds") or 0.0)
            for row in rows
            if row.get("action") == "run_daemon_graph"
        ),
    }


def _action_requests_foundry(action: dict[str, Any]) -> bool:
    return str(action.get("queued_followup_action") or "") == "compile_better_for_next_time"


def _queue_foundry_for_nn_region(
    graph: NNGraph,
    region: NNRegion,
    values: dict[str, NNTensorValue],
    action: dict[str, Any],
    *,
    catalog_root: str,
    action_ledger_path: Optional[str],
    target: str,
    elapsed_seconds: float,
    fabric_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    if fabric_result is not None and isinstance(fabric_result.get("graph"), dict):
        fabric_graph = FabricGraph.from_json(fabric_result["graph"])
        summary = dict(fabric_result.get("summary") or {})
        source_path = "nn-run:fabric-session"
        observed_seconds = float(summary.get("timed_elapsed_seconds") or fabric_result.get("elapsed_seconds") or elapsed_seconds)
        observed_transfer_bytes = int(summary.get("timed_transfer_bytes") or 0)
        execution_backend = "daemon_session_resident"
        timed_graph_packets = int(summary.get("timed_graph_packets") or 0)
        observed_run_daemon_seconds = float(summary.get("observed_run_daemon_seconds") or 0.0)
    else:
        fabric_graph = _foundry_graph_for_nn_region(region, values, graph=graph)
        source_path = f"nn-run:{action.get('selected_action', 'unknown')}"
        observed_seconds = float(elapsed_seconds)
        observed_transfer_bytes = 0
        execution_backend = str(action.get("selected_action") or "unknown")
        timed_graph_packets = 0
        observed_run_daemon_seconds = 0.0
    return _queue_foundry_graph(
        fabric_graph,
        catalog_root=catalog_root,
        action_ledger_path=action_ledger_path,
        target=target,
        graph_name=graph.name,
        nn_region=region,
        source_path=source_path,
        observed_seconds=observed_seconds,
        observed_transfer_bytes=observed_transfer_bytes,
        execution_backend=execution_backend,
        timed_graph_packets=timed_graph_packets,
        observed_run_daemon_seconds=observed_run_daemon_seconds,
        metadata={
            "nn_region": region.to_json(),
            "nn_tensors": _nn_region_tensor_metadata(region, values),
            "placement_action": dict(action),
        },
    )


def _foundry_graph_for_nn_region(
    region: NNRegion,
    values: dict[str, NNTensorValue],
    *,
    graph: Optional[NNGraph] = None,
) -> FabricGraph:
    buffers: dict[str, FabricBufferSpec] = {}
    ops: list[FabricGraphOp] = []
    local_values = dict(values)
    for index, node in enumerate(region.nodes):
        input_values = [local_values[name] for name in node.inputs if name in local_values]
        inferred_outputs = _foundry_infer_node_outputs(node, input_values, graph=graph)
        output_names = tuple(node.outputs or (f"{node.node_id}_out",))
        primary_output = output_names[0]
        primary_shape = inferred_outputs.get(primary_output, (input_values[0].shape if input_values else ()))
        elements = int(_shape_element_count(primary_shape) if primary_shape else (input_values[0].elements if input_values else 0))
        for name in node.inputs:
            value = local_values.get(name)
            buffers.setdefault(
                name,
                FabricBufferSpec(
                    name,
                    int(value.elements if value is not None else elements),
                    "pl" if value is not None and value.residency in {"daemon", "xrt_bo", "dma_buf", "catalog_core"} else "ps",
                ),
            )
        for output_name in output_names:
            output_shape = inferred_outputs.get(output_name, primary_shape)
            buffers[output_name] = FabricBufferSpec(output_name, int(_shape_element_count(output_shape) if output_shape else elements), "unknown")
        ops.append(
            FabricGraphOp(
                node.node_id or f"nn_op{index}",
                node.op,
                node.inputs,
                primary_output,
                elements,
                output_names[1:],
            )
        )
        for output_name in output_names:
            output_shape = inferred_outputs.get(output_name, primary_shape)
            local_values[output_name] = NNTensorValue(
                value_id=f"foundry:{node.node_id}:{output_name}",
                name=output_name,
                dtype=_foundry_output_dtype(node, input_values, output_name),
                shape=tuple(int(dim) for dim in output_shape),
                residency="unknown",
                materialized=False,
                producer=node.node_id,
            )
    outputs = region.outputs or ((ops[-1].output,) if ops else ())
    return FabricGraph(buffers=buffers, ops=tuple(ops), outputs=outputs, output_residency="ps")


def _foundry_infer_node_outputs(
    node: NNNode,
    input_values: Sequence[NNTensorValue],
    *,
    graph: Optional[NNGraph],
) -> dict[str, tuple[int, ...]]:
    inferred: dict[str, tuple[int, ...]] = {}
    if graph is not None:
        for output_name in node.outputs:
            spec = graph.tensors.get(output_name)
            if spec is None:
                continue
            shape = _static_spec_shape(spec)
            if shape is not None:
                inferred[output_name] = shape
    if len(inferred) == len(node.outputs):
        return inferred
    if node.op == "attention_qkv" and len(input_values) >= 2 and len(node.outputs) == 3:
        left = tuple(int(dim) for dim in input_values[0].shape)
        right = tuple(int(dim) for dim in input_values[1].shape)
        if left and len(right) >= 2 and right[-1] % 3 == 0:
            output_shape = (*left[:-1], int(right[-1]) // 3)
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "attention_score" and len(input_values) >= 2:
        output_shape = _attention_score_output_shape(input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "attention_value" and len(input_values) >= 2:
        output_shape = _attention_value_output_shape(input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "dynamic_quantize_linear" and input_values:
        if len(node.outputs) >= 1:
            inferred.setdefault(node.outputs[0], input_values[0].shape)
        for output_name in node.outputs[1:3]:
            inferred.setdefault(output_name, ())
        return inferred
    if node.op == "shape" and input_values:
        shape_range = _shape_op_range(node.attrs, len(input_values[0].shape))
        if shape_range is not None:
            start, end = shape_range
            for output_name in node.outputs:
                inferred.setdefault(output_name, (int(end) - int(start),))
            return inferred
    if node.op == "range" and input_values:
        output_length = _range_output_length(input_values)
        if output_length is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, (int(output_length),))
            return inferred
    if node.op == "constant_of_shape" and input_values:
        output_shape = _constant_of_shape_output_shape(input_values[0])
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "flatten" and input_values:
        output_shape = _flatten_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "reshape" and input_values:
        output_shape = _reshape_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "transpose" and input_values:
        output_shape = _transpose_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "unsqueeze" and input_values:
        output_shape = _unsqueeze_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "squeeze" and input_values:
        output_shape = _squeeze_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "expand" and input_values:
        output_shape = _expand_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op in BROADCAST_OUTPUT_NN_OPS and input_values:
        output_shape = _broadcast_output_shape(input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "not" and input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[0].shape)
        return inferred
    if node.op == "clip" and input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[0].shape)
        return inferred
    if node.op == "dropout" and input_values:
        for output_name in node.outputs[:2]:
            inferred.setdefault(output_name, input_values[0].shape)
        return inferred
    if node.op in {"cumsum", "trilu"} and input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[0].shape)
        return inferred
    if node.op == "batchnorm" and input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[0].shape)
        return inferred
    if node.op == "concat" and input_values:
        output_shape = _concat_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "pad" and input_values:
        output_shape = _pad_output_shape(node, input_values[0].shape, [value.array for value in input_values[1:2]])
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "slice" and input_values:
        output_shape = _slice_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "split" and input_values:
        output_shapes = _split_output_shapes(node, input_values)
        if output_shapes is not None:
            for output_name, output_shape in output_shapes.items():
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "tile" and len(input_values) >= 2:
        repeats = _tile_repeats_from_value(input_values[1], len(input_values[0].shape))
        if repeats is not None:
            output_shape = tuple(int(dim) * int(rep) for dim, rep in zip(input_values[0].shape, repeats))
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "size" and input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, ())
        return inferred
    if node.op in {"reduce_mean", "reduce_sum", "reduce_min", "reduce_max", "reduce_prod"} and input_values:
        axes = _reduction_axes_from_sources(node.attrs, [value.array for value in input_values[1:]], rank=len(input_values[0].shape))
        if axes is not None:
            output_shape = _reduction_output_shape(node, input_values[0].shape, axes)
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "einsum" and input_values:
        output_shape = _einsum_output_shape(node, [value.shape for value in input_values])
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op in {"matmul", "gemm", "qlinear_matmul", "matmul_integer"} and len(input_values) >= 2:
        left_index, right_index = (0, 3) if node.op == "qlinear_matmul" else (0, 1)
        if len(input_values) > right_index:
            left = _linear_operand_shape(input_values[left_index].shape, node.attrs.get("transA", 0) if node.op == "gemm" else 0)
            right = _linear_operand_shape(input_values[right_index].shape, node.attrs.get("transB", 0) if node.op == "gemm" else 0)
            output_shape = _matmul_output_shape(left, right)
            if output_shape is not None:
                for output_name in node.outputs:
                    inferred.setdefault(output_name, output_shape)
                return inferred
    if node.op == "topk" and input_values:
        rank = len(input_values[0].shape)
        axis = _normalized_axis_value(node.attrs.get("axis", -1), rank)
        k = _topk_k_from_attrs_or_value(node.attrs, [value.array for value in input_values[1:]])
        if axis is not None and k is not None:
            output_shape = list(input_values[0].shape)
            output_shape[axis] = int(k)
            for output_name in node.outputs:
                inferred.setdefault(output_name, tuple(output_shape))
            return inferred
    if node.op in {"argmax", "argmin"} and input_values:
        output_shape = _argmax_output_shape(node, input_values[0].shape)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "gather" and len(input_values) >= 2:
        output_shape = _gather_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "gather_elements" and len(input_values) >= 2:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[1].shape)
        return inferred
    if node.op == "gather_nd" and len(input_values) >= 2:
        output_shape = _gather_nd_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "scatter_elements" and len(input_values) >= 3:
        output_shape = _scatter_elements_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "scatter_nd" and len(input_values) >= 3:
        output_shape = _scatter_nd_output_shape(node, input_values)
        if output_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, output_shape)
            return inferred
    if node.op == "top_p_filter" and input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[0].shape)
        return inferred
    if node.op == "kv_cache_append" and len(input_values) >= 2:
        rank = len(input_values[0].shape)
        axis = _kv_cache_axis(node.attrs, rank)
        if axis is not None:
            output_shape = list(input_values[0].shape)
            output_shape[axis] = int(input_values[0].shape[axis]) + int(input_values[1].shape[axis])
            for output_name in node.outputs:
                inferred.setdefault(output_name, tuple(output_shape))
            return inferred
    if node.op in {"conv2d", "depthwise_conv2d"} and len(input_values) >= 2:
        conv_shape = _infer_conv_output_shape(node, input_values)
        if conv_shape is not None:
            for output_name in node.outputs:
                inferred.setdefault(output_name, conv_shape)
            return inferred
    if input_values:
        for output_name in node.outputs:
            inferred.setdefault(output_name, input_values[0].shape)
    return inferred


def _infer_conv_output_shape(node: NNNode, input_values: Sequence[NNTensorValue]) -> Optional[tuple[int, ...]]:
    x_shape = tuple(int(dim) for dim in input_values[0].shape)
    w_shape = tuple(int(dim) for dim in input_values[1].shape)
    if len(x_shape) != 4 or len(w_shape) != 4:
        return None
    attrs = _conv_normalized_attrs(node.attrs)
    if attrs["error"] is not None:
        return None
    strides = attrs["strides"]
    pads = attrs["pads"]
    dilations = attrs["dilations"]
    out_h = _conv_output_dim(x_shape[2], w_shape[2], pads[0], pads[2], strides[0], dilations[0])
    out_w = _conv_output_dim(x_shape[3], w_shape[3], pads[1], pads[3], strides[1], dilations[1])
    if out_h <= 0 or out_w <= 0:
        return None
    return (x_shape[0], w_shape[0], out_h, out_w)


def _foundry_output_dtype(node: NNNode, input_values: Sequence[NNTensorValue], output_name: str) -> str:
    if node.op in {"argmax", "argmin"} or (node.op == "topk" and len(node.outputs) >= 2 and output_name == node.outputs[1]):
        return "int64"
    if node.op in BOOL_OUTPUT_NN_OPS:
        return "bool"
    if node.op == "where" and len(input_values) >= 2:
        return input_values[1].dtype
    if node.op == "top_p_filter" and len(node.outputs) >= 2 and output_name == node.outputs[1]:
        return "bool"
    if node.op == "dropout" and len(node.outputs) >= 2 and output_name == node.outputs[1]:
        return "bool"
    if node.op == "qlinear_matmul" and len(input_values) >= 8:
        return input_values[7].dtype
    if node.op == "matmul_integer":
        return "int32"
    if node.op == "dynamic_quantize_linear":
        if len(node.outputs) >= 2 and output_name == node.outputs[1]:
            return "float32"
        return "uint8"
    if node.op in {"shape", "size"}:
        return "int64"
    if node.op == "cast_like" and len(input_values) >= 2:
        return input_values[1].dtype
    if node.op == "range" and input_values:
        return input_values[0].dtype
    if node.op == "constant_of_shape":
        value = _constant_of_shape_value_array(node.attrs)
        if value is not None:
            return str(value.dtype)
        return str(node.attrs.get("dtype", "float32"))
    if input_values:
        return input_values[0].dtype
    return "unknown"


def _nn_region_tensor_metadata(region: NNRegion, values: dict[str, NNTensorValue]) -> dict[str, Any]:
    names = sorted({name for node in region.nodes for name in (*node.inputs, *node.outputs)})
    metadata: dict[str, Any] = {}
    for name in names:
        if name not in values:
            continue
        value = values[name]
        item = {
            "value_id": value.value_id,
            "name": value.name,
            "dtype": values[name].dtype,
            "shape": list(values[name].shape),
            "elements": values[name].elements,
            "residency": values[name].residency,
            "materialized": values[name].materialized,
            "producer": value.producer,
            "refcount": int(value.refcount),
            "materialization_reason": value.materialization_reason,
        }
        if value.validation is not None:
            item["validation"] = value.validation
        if value.daemon_handle is not None:
            item["daemon_handle"] = dict(value.daemon_handle)
        for key in (
            "spec_shape",
            "shape_constraints",
            "shape_bindings",
            "quantization",
            "weight",
            "retained",
            "retained_feedback",
            "feedback_input_name",
            "feedback_output_name",
        ):
            if key in value.metadata:
                item[key] = value.metadata[key]
        metadata[name] = item
    return metadata


def _queue_foundry_for_fabric_result(
    fabric_result: dict[str, Any],
    *,
    catalog_root: str,
    action_ledger_path: Optional[str] = None,
    target: str,
    graph_name: str,
    region_id: str,
) -> dict[str, Any]:
    graph_payload = fabric_result.get("graph")
    if not isinstance(graph_payload, dict):
        return {"ok": False, "reason": "missing_fabric_graph"}
    fabric_graph = FabricGraph.from_json(graph_payload)
    if not fabric_graph.ops:
        return {"ok": False, "reason": "empty_fabric_graph"}
    summary = dict(fabric_result.get("summary") or {})
    return _queue_foundry_graph(
        fabric_graph,
        catalog_root=catalog_root,
        action_ledger_path=action_ledger_path,
        target=target,
        graph_name=graph_name,
        nn_region=None,
        nn_region_id=region_id,
        source_path="nn-run:fabric-session",
        observed_seconds=float(summary.get("timed_elapsed_seconds") or fabric_result.get("elapsed_seconds") or 0.0),
        observed_transfer_bytes=int(summary.get("timed_transfer_bytes") or 0),
        execution_backend="daemon_session_resident",
        timed_graph_packets=int(summary.get("timed_graph_packets") or 0),
        observed_run_daemon_seconds=float(summary.get("observed_run_daemon_seconds") or 0.0),
        metadata={},
    )


def _queue_foundry_graph(
    fabric_graph: FabricGraph,
    *,
    catalog_root: str,
    action_ledger_path: Optional[str],
    target: str,
    graph_name: str,
    nn_region: Optional[NNRegion] = None,
    nn_region_id: Optional[str] = None,
    source_path: str,
    observed_seconds: float,
    observed_transfer_bytes: int,
    execution_backend: str,
    timed_graph_packets: int,
    observed_run_daemon_seconds: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if not fabric_graph.ops:
        return {"ok": False, "reason": "empty_fabric_graph"}
    ops = tuple(op.op for op in fabric_graph.ops)
    elements = int(fabric_graph.ops[0].elements)
    region_key = nn_region_id or (nn_region.region_id if nn_region is not None else "region")
    region_seed = {
        "source": "nn-run",
        "graph": graph_name,
        "region_id": region_key,
        "ops": ops,
        "elements": elements,
        "tensor_metadata": metadata.get("nn_tensors", {}),
    }
    foundry_region = FabricRegion(
        region_id="region_" + hashlib.sha256(json.dumps(region_seed, sort_keys=True).encode("utf-8")).hexdigest()[:16],
        ops=ops,
        elements=elements,
        occurrence_count=max(1, int(timed_graph_packets or 1)),
        observed_seconds=float(observed_seconds),
        observed_transfer_bytes=int(observed_transfer_bytes),
        source_paths=(source_path,),
        graph=fabric_graph,
    )
    catalog = FabricFoundryCatalog(catalog_root, action_ledger_path=action_ledger_path)
    catalog.put_region(foundry_region)
    candidates = generate_candidates([foundry_region], target=target)
    if not candidates:
        return {
            "ok": False,
            "reason": "no_candidate_generated",
            "region": foundry_region.to_json(),
        }
    candidate = candidates[0]
    catalog.put_candidate(candidate)
    measurement = {
        "schema": "pan_fabric_measurement_v1",
        "candidate_id": candidate.candidate_id,
        "source_path": source_path,
        "ok": True,
        "execution_backend": execution_backend,
        "target": target,
        "ops": list(ops),
        "elements": elements,
        "elapsed_seconds": float(observed_seconds),
        "observed_run_daemon_seconds": float(observed_run_daemon_seconds),
        "observed_run_pl_seconds": 0.0,
        "observed_transfer_bytes": int(observed_transfer_bytes),
        "resident": execution_backend == "daemon_session_resident",
        "timed_graph_packets": int(timed_graph_packets),
        "graph_name": graph_name,
        "nn_region_id": region_key,
        "nn_metadata": dict(metadata),
        "ingested_at": time.time(),
    }
    catalog.append_measurement(measurement)
    catalog.put_core_manifest(candidate, artifact_status="queued", measurement=measurement)
    queued = catalog.append_queue(candidate, reason="nn_region_compile_better_for_next_time")
    return {
        "ok": True,
        "region": foundry_region.to_json(),
        "candidate": candidate.to_json(),
        "queued": queued.to_json(),
        "measurement": measurement,
    }


def _fabric_graph_for_spf_region(
    region: NNRegion,
    values: dict[str, NNTensorValue],
    *,
    materialize_outputs: bool,
) -> tuple[FabricGraph, dict[str, Any], dict[str, tuple[int, ...]]]:
    buffers: dict[str, FabricBufferSpec] = {}
    ops: list[FabricGraphOp] = []
    initial_inputs: dict[str, Any] = {}
    output_shapes: dict[str, tuple[int, ...]] = {}
    local_values = dict(values)
    produced: set[str] = set()
    for index, node in enumerate(region.nodes):
        input_values = _spf32_node_input_values(node, local_values)
        for tensor_name, tensor_value in zip(node.inputs, input_values):
            buffers.setdefault(
                tensor_name,
                FabricBufferSpec(
                    tensor_name,
                    int(tensor_value.elements),
                    "pl" if tensor_value.residency == "daemon" and tensor_value.daemon_handle else "ps",
                ),
            )
            if tensor_name not in produced and tensor_name not in initial_inputs:
                initial_inputs[tensor_name] = _fabric_input_for_tensor(tensor_value)
        output = node.outputs[0]
        buffers[output] = FabricBufferSpec(output, int(input_values[0].elements), "unknown")
        output_shapes[output] = input_values[0].shape
        ops.append(
            FabricGraphOp(
                op_id=node.node_id or f"nn_{node.op.replace('.', '_')}_{index}",
                op=node.op,
                inputs=node.inputs,
                output=output,
                elements=int(input_values[0].elements),
            )
        )
        produced.add(output)
        local_values.update(_inferred_verifier_outputs(node, input_values))
    return (
        FabricGraph(
            buffers=buffers,
            ops=tuple(ops),
            outputs=region.outputs,
            output_residency="ps" if materialize_outputs else "pl",
        ),
        initial_inputs,
        output_shapes,
    )


def _spf32_node_input_values(node: NNNode, values: dict[str, NNTensorValue]) -> tuple[NNTensorValue, ...]:
    if node.op in FABRIC_BINARY_NN_OPS:
        if len(node.inputs) != 2 or len(node.outputs) != 1:
            raise ValueError(f"{node.op} NN lowering requires exactly two inputs and one output")
    elif node.op in SPF32_UNARY_NN_OPS:
        if len(node.inputs) != 1 or len(node.outputs) != 1:
            raise ValueError(f"{node.op} NN lowering requires exactly one input and one output")
    else:
        raise ValueError(f"fabric-session NN lowering does not support {node.op!r}")
    input_values = tuple(values[name] for name in node.inputs)
    first = input_values[0]
    for value in input_values[1:]:
        if value.elements != first.elements:
            raise ValueError(f"{node.op} NN lowering requires equal input element counts")
        if value.shape != first.shape:
            raise ValueError(f"{node.op} NN lowering requires equal input shapes")
    if node.op == "spf32.fft64" and int(first.elements) != 64:
        raise ValueError("spf32.fft64 NN lowering requires exactly 64 elements")
    return input_values


def _fabric_input_for_tensor(value: NNTensorValue) -> Any:
    _validate_nn_value_residency(
        value.name,
        residency=value.residency,
        daemon_handle=value.daemon_handle,
        shape=value.shape,
        materialized=bool(value.array is not None or value.materialized),
    )
    if value.daemon_handle is not None and value.residency == "daemon":
        return value.daemon_handle
    if value.array is None:
        raise ValueError(f"NN tensor {value.name!r} is not materialized and has no daemon handle")
    return value.array


def _dequantized_node_input(
    name: str,
    array: Any,
    *,
    tensor_specs: Optional[dict[str, NNTensorSpec]],
    tensor_values: Optional[dict[str, NNTensorValue]],
) -> Any:
    quantization: dict[str, Any] = {}
    if tensor_values and name in tensor_values:
        quantization = dict(tensor_values[name].metadata.get("quantization") or {})
    if not quantization and tensor_specs and name in tensor_specs:
        quantization = dict(tensor_specs[name].quantization or {})
    if not quantization:
        return array
    if not _is_affine_quantization(quantization):
        raise RuntimeError(f"unsupported quantization for tensor {name!r}: affine scale/zero_point required")
    np = _numpy()
    scale = float(quantization["scale"])
    zero_point = int(quantization.get("zero_point", 0))
    return (np.asarray(array).astype(np.float32) - float(zero_point)) * scale


def _requantize_array(array: Any, quantization: dict[str, Any]) -> Any:
    if not _is_affine_quantization(quantization):
        raise RuntimeError("output_quantization requires affine scale/zero_point")
    np = _numpy()
    scale = float(quantization["scale"])
    zero_point = int(quantization.get("zero_point", 0))
    dtype = str(quantization.get("dtype", "int8"))
    rounded = np.rint(np.asarray(array, dtype=np.float32) / scale + float(zero_point))
    return _clip_quantized_array(rounded, dtype).astype(_numpy_dtype(dtype))


def _qlinear_params(inputs: Sequence[Any], attrs: dict[str, Any], *, input_rank: int, default_zero_dtype: str) -> tuple[Any, Any]:
    np = _numpy()
    if len(inputs) < 2:
        raise RuntimeError("QuantizeLinear/DequantizeLinear requires scale input")
    scale = np.asarray(inputs[1], dtype=np.float32)
    if scale.size != 1:
        axis = _normalized_axis_value(attrs.get("axis", 1), int(input_rank))
        if axis is None:
            raise RuntimeError("QuantizeLinear/DequantizeLinear axis is out of range")
        shape = [1] * int(input_rank)
        shape[axis] = int(scale.reshape(-1).shape[0])
        scale = scale.reshape(tuple(shape))
    zero_point_dtype = _numpy_dtype(default_zero_dtype)
    zero_point = np.asarray(inputs[2] if len(inputs) >= 3 else 0, dtype=zero_point_dtype)
    if zero_point.size != 1:
        axis = _normalized_axis_value(attrs.get("axis", 1), int(input_rank))
        if axis is None:
            raise RuntimeError("QuantizeLinear/DequantizeLinear zero-point axis is out of range")
        shape = [1] * int(input_rank)
        shape[axis] = int(zero_point.reshape(-1).shape[0])
        zero_point = zero_point.reshape(tuple(shape))
    return scale, zero_point


def _matmul_integer_zero_point_array(zero_point: Any, target: Any, *, side: str) -> Any:
    np = _numpy()
    if zero_point is None:
        return np.asarray(0, dtype=np.int32)
    arr = np.asarray(zero_point).astype(np.int32)
    if arr.size == 1:
        return arr.reshape(())
    target_shape = tuple(int(dim) for dim in np.asarray(target).shape)
    if len(target_shape) < 2:
        raise RuntimeError("MatMulInteger requires matrix-compatible inputs")
    if arr.ndim == 1:
        if side == "a":
            if int(arr.shape[0]) != int(target_shape[-2]):
                raise RuntimeError("MatMulInteger A zero point does not match row dimension")
            return arr.reshape((1,) * (len(target_shape) - 2) + (target_shape[-2], 1))
        if int(arr.shape[0]) != int(target_shape[-1]):
            raise RuntimeError("MatMulInteger B zero point does not match column dimension")
        return arr.reshape((1,) * (len(target_shape) - 2) + (1, target_shape[-1]))
    try:
        return np.broadcast_to(arr, target_shape).astype(np.int32)
    except ValueError as exc:
        raise RuntimeError(f"MatMulInteger {side.upper()} zero point is not broadcastable") from exc


def _execute_dynamic_quantize_linear(array: Any) -> tuple[Any, Any, Any]:
    np = _numpy()
    arr = np.asarray(array, dtype=np.float32)
    if arr.size == 0:
        raise RuntimeError("DynamicQuantizeLinear requires a non-empty input tensor")
    qmin = 0.0
    qmax = 255.0
    x_min = min(0.0, float(np.min(arr)))
    x_max = max(0.0, float(np.max(arr)))
    scale = (x_max - x_min) / (qmax - qmin)
    if not math.isfinite(scale) or scale <= 0.0:
        scale = 1.0
        zero = 0
    else:
        zero = int(np.rint(np.clip(qmin - x_min / scale, qmin, qmax)))
    y_scale = np.asarray(scale, dtype=np.float32)
    y_zero = np.asarray(zero, dtype=np.uint8)
    rounded = np.rint(arr / float(y_scale)) + y_zero.astype(np.float32)
    y = _clip_quantized_array(rounded, "uint8").astype(np.uint8)
    return y, y_scale, y_zero


def _clip_quantized_array(array: Any, dtype: str) -> Any:
    np = _numpy()
    target = _numpy_dtype(dtype)
    if target.kind not in {"i", "u"}:
        raise RuntimeError(f"quantized output dtype must be integer, got {dtype!r}")
    info = np.iinfo(target)
    return np.clip(array, info.min, info.max)


def _rotate_half_array(array: Any, *, interleaved: bool) -> Any:
    np = _numpy()
    arr = np.asarray(array)
    if arr.shape[-1] % 2 != 0:
        raise RuntimeError("rotary_embedding requires an even last dimension")
    if interleaved:
        rotated = np.empty_like(arr)
        rotated[..., 0::2] = -arr[..., 1::2]
        rotated[..., 1::2] = arr[..., 0::2]
        return rotated
    half = int(arr.shape[-1]) // 2
    return np.concatenate((-arr[..., half:], arr[..., :half]), axis=-1)


def _execute_rotary_embedding(array: Any, cos: Any, sin: Any, *, interleaved: bool) -> Any:
    np = _numpy()
    arr = np.asarray(array)
    cos_arr = np.asarray(cos)
    sin_arr = np.asarray(sin)
    try:
        return arr * np.broadcast_to(cos_arr, arr.shape) + _rotate_half_array(arr, interleaved=interleaved) * np.broadcast_to(sin_arr, arr.shape)
    except ValueError:
        pass
    half = int(arr.shape[-1]) // 2
    if interleaved:
        even = arr[..., 0::2]
        odd = arr[..., 1::2]
        target_shape = even.shape
        cos_half = np.broadcast_to(cos_arr, target_shape)
        sin_half = np.broadcast_to(sin_arr, target_shape)
        out = np.empty_like(arr)
        out[..., 0::2] = even * cos_half - odd * sin_half
        out[..., 1::2] = even * sin_half + odd * cos_half
        return out
    left = arr[..., :half]
    right = arr[..., half:]
    target_shape = left.shape
    cos_half = np.broadcast_to(cos_arr, target_shape)
    sin_half = np.broadcast_to(sin_arr, target_shape)
    return np.concatenate((left * cos_half - right * sin_half, left * sin_half + right * cos_half), axis=-1)


def _execute_cumsum(array: Any, *, axis: int, exclusive: bool, reverse: bool) -> Any:
    np = _numpy()
    arr = np.asarray(array)
    work = np.flip(arr, axis=axis) if reverse else arr
    summed = np.cumsum(work, axis=axis).astype(arr.dtype, copy=False)
    if exclusive:
        shifted = np.zeros_like(summed)
        destination = [slice(None)] * summed.ndim
        source = [slice(None)] * summed.ndim
        destination[axis] = slice(1, None)
        source[axis] = slice(None, -1)
        shifted[tuple(destination)] = summed[tuple(source)]
        summed = shifted
    return np.flip(summed, axis=axis) if reverse else summed


def _clip_bounds_from_sources(attrs: dict[str, Any], sources: Sequence[Any]) -> tuple[Any, Any]:
    np = _numpy()
    min_source = sources[0] if len(sources) >= 1 else attrs.get("min")
    max_source = sources[1] if len(sources) >= 2 else attrs.get("max")
    min_value = None if min_source is None else np.asarray(min_source).reshape(-1)[0].item()
    max_value = None if max_source is None else np.asarray(max_source).reshape(-1)[0].item()
    return min_value, max_value


def _dropout_ratio_from_sources(attrs: dict[str, Any], sources: Sequence[Any]) -> Optional[float]:
    source = sources[0] if sources else attrs.get("ratio", 0.5)
    try:
        arr = _numpy().asarray(source).reshape(-1)
    except (TypeError, ValueError):
        return None
    if int(arr.size) != 1:
        return None
    try:
        return float(arr[0].item())
    except (TypeError, ValueError):
        return None


def _dropout_training_mode_from_sources(attrs: dict[str, Any], sources: Sequence[Any]) -> Optional[bool]:
    source = sources[0] if sources else attrs.get("training_mode", False)
    try:
        arr = _numpy().asarray(source).reshape(-1)
    except (TypeError, ValueError):
        return None
    if int(arr.size) != 1:
        return None
    item = arr[0].item()
    if isinstance(item, (bool, int, float)):
        return bool(item)
    return None


def _pad_mode(attrs: dict[str, Any]) -> str:
    mode = attrs.get("mode", "constant")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8", errors="replace")
    return str(mode).strip().lower()


def _pad_values_from_sources(attrs: dict[str, Any], sources: Sequence[Any], *, rank: int) -> Optional[tuple[int, ...]]:
    source = sources[0] if sources else attrs.get("pads")
    if source is None:
        return None
    pads = _int_list_from_source(source)
    if pads is None or len(pads) != 2 * int(rank):
        return None
    return tuple(int(value) for value in pads)


def _pad_constant_from_sources(attrs: dict[str, Any], sources: Sequence[Any]) -> Any:
    source = sources[0] if sources else attrs.get("value", attrs.get("constant_value", 0))
    try:
        arr = _numpy().asarray(source).reshape(-1)
    except (TypeError, ValueError):
        return None
    if int(arr.size) != 1:
        return None
    return arr[0].item()


def _execute_pad(array: Any, pads: Sequence[int], constant_value: Any) -> Any:
    np = _numpy()
    arr = np.asarray(array)
    rank = int(arr.ndim)
    pad_width = tuple((int(pads[index]), int(pads[index + rank])) for index in range(rank))
    return np.pad(arr, pad_width, mode="constant", constant_values=constant_value)


def _execute_node(
    values: dict[str, Any],
    node: NNNode,
    *,
    tensor_specs: Optional[dict[str, NNTensorSpec]] = None,
    tensor_values: Optional[dict[str, NNTensorValue]] = None,
    spf32_engine: Optional[_PSOpEngine] = None,
) -> None:
    np = _numpy()
    op = node.op
    inputs = [values[name] for name in node.inputs if name in values]
    qinputs = [
        _dequantized_node_input(name, values[name], tensor_specs=tensor_specs, tensor_values=tensor_values)
        for name in node.inputs
        if name in values
    ]
    if op not in PS_SUPPORTED_OPS:
        raise RuntimeError(f"PS oracle does not support NN op {op!r}")
    if op == "attention_qkv":
        if len(qinputs) < 2:
            raise RuntimeError("attention_qkv requires input and packed weight tensors")
        projected = np.matmul(qinputs[0], qinputs[1])
        if len(qinputs) >= 3:
            projected = projected + qinputs[2]
        if not node.outputs or len(node.outputs) != 3:
            raise RuntimeError("attention_qkv requires exactly three outputs")
        if projected.shape[-1] % 3 != 0:
            raise RuntimeError("attention_qkv packed projection dimension must be divisible by three")
        for output_name, arr in zip(node.outputs, np.split(projected, 3, axis=-1)):
            values[output_name] = arr
        return
    if op == "matmul":
        out = np.matmul(qinputs[0], qinputs[1])
    elif op == "einsum":
        equation = _einsum_equation(node)
        if equation is None:
            raise RuntimeError("Einsum requires an equation attribute")
        out = np.einsum(equation, *inputs)
    elif op == "gemm":
        alpha = float(node.attrs.get("alpha", 1.0))
        beta = float(node.attrs.get("beta", 1.0))
        left = np.swapaxes(qinputs[0], -1, -2) if _truthy_int(node.attrs.get("transA", 0)) else qinputs[0]
        right = np.swapaxes(qinputs[1], -1, -2) if _truthy_int(node.attrs.get("transB", 0)) else qinputs[1]
        out = alpha * np.matmul(left, right)
        if len(qinputs) >= 3:
            out = out + beta * qinputs[2]
    elif op == "qlinear_matmul":
        if len(inputs) < 8:
            raise RuntimeError("QLinearMatMul requires eight inputs")
        a = (np.asarray(inputs[0]).astype(np.float32) - np.asarray(inputs[2]).astype(np.float32)) * np.asarray(inputs[1], dtype=np.float32)
        b = (np.asarray(inputs[3]).astype(np.float32) - np.asarray(inputs[5]).astype(np.float32)) * np.asarray(inputs[4], dtype=np.float32)
        y_scale = np.asarray(inputs[6], dtype=np.float32)
        y_zero = np.asarray(inputs[7])
        target_dtype = str(y_zero.dtype)
        rounded = np.rint(np.matmul(a, b) / y_scale.astype(np.float32) + y_zero.astype(np.float32))
        out = _clip_quantized_array(rounded, target_dtype).astype(_numpy_dtype(target_dtype))
    elif op == "matmul_integer":
        if len(inputs) < 2:
            raise RuntimeError("MatMulInteger requires two inputs")
        a_zero = _matmul_integer_zero_point_array(inputs[2] if len(inputs) >= 3 else None, inputs[0], side="a")
        b_zero = _matmul_integer_zero_point_array(inputs[3] if len(inputs) >= 4 else None, inputs[1], side="b")
        a = np.asarray(inputs[0]).astype(np.int32) - a_zero
        b = np.asarray(inputs[1]).astype(np.int32) - b_zero
        out = np.matmul(a, b).astype(np.int32)
    elif op == "identity":
        out = np.asarray(inputs[0]).copy()
    elif op == "dropout":
        ratio = _dropout_ratio_from_sources(node.attrs, inputs[1:2])
        training = _dropout_training_mode_from_sources(node.attrs, inputs[2:3])
        if ratio is None or not math.isfinite(ratio) or ratio < 0.0 or ratio >= 1.0:
            raise RuntimeError("Dropout ratio must be a finite value in [0, 1)")
        if training is None or training:
            raise RuntimeError("Dropout training mode is unsupported")
        if not node.outputs:
            raise RuntimeError("Dropout requires at least one output")
        values[node.outputs[0]] = np.asarray(inputs[0]).copy()
        if len(node.outputs) >= 2:
            values[node.outputs[1]] = np.ones_like(inputs[0], dtype=np.bool_)
        return
    elif op == "sum":
        if not inputs:
            raise RuntimeError("Sum requires at least one input")
        out = inputs[0]
        for value in inputs[1:]:
            out = out + value
    elif op == "mean":
        if not inputs:
            raise RuntimeError("Mean requires at least one input")
        out = inputs[0]
        for value in inputs[1:]:
            out = out + value
        out = (out / np.asarray(len(inputs), dtype=np.asarray(inputs[0]).dtype)).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "add":
        out = inputs[0] + inputs[1]
    elif op == "sub":
        out = inputs[0] - inputs[1]
    elif op == "mul":
        out = inputs[0] * inputs[1]
    elif op == "div":
        out = inputs[0] / inputs[1]
    elif op == "neg":
        out = np.negative(inputs[0])
    elif op == "abs":
        out = np.abs(inputs[0])
    elif op == "pow":
        out = np.power(inputs[0], inputs[1])
    elif op == "mod":
        out = np.mod(inputs[0], inputs[1])
    elif op == "min":
        out = inputs[0]
        for value in inputs[1:]:
            out = np.minimum(out, value)
    elif op == "max":
        out = inputs[0]
        for value in inputs[1:]:
            out = np.maximum(out, value)
    elif op == "clip":
        min_value, max_value = _clip_bounds_from_sources(node.attrs, inputs[1:])
        out = np.clip(inputs[0], min_value, max_value)
    elif op == "concat":
        out = np.concatenate(inputs, axis=int(node.attrs.get("axis", 0)))
    elif op in SPF32_BINARY_NN_OPS:
        out = _execute_spf32_node(op, inputs, spf32_engine=spf32_engine)
    elif op in SPF32_UNARY_NN_OPS:
        out = _execute_spf32_node(op, inputs, spf32_engine=spf32_engine)
    elif op == "relu":
        out = np.maximum(inputs[0], 0)
    elif op == "gelu":
        out = 0.5 * inputs[0] * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (inputs[0] + 0.044715 * np.power(inputs[0], 3))))
    elif op == "bias_gelu":
        biased = inputs[0] + inputs[1]
        out = 0.5 * biased * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (biased + 0.044715 * np.power(biased, 3))))
    elif op == "quick_gelu":
        out = inputs[0] / (1.0 + np.exp(-1.702 * inputs[0]))
    elif op == "silu":
        out = inputs[0] / (1.0 + np.exp(-inputs[0]))
    elif op == "sigmoid":
        out = 1.0 / (1.0 + np.exp(-inputs[0]))
    elif op == "tanh":
        out = np.tanh(inputs[0])
    elif op == "erf":
        arr = np.asarray(inputs[0])
        out = np.vectorize(math.erf, otypes=[np.float64])(arr).astype(arr.dtype, copy=False)
    elif op == "sin":
        out = np.sin(inputs[0])
    elif op == "cos":
        out = np.cos(inputs[0])
    elif op == "floor":
        out = np.floor(inputs[0]).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "ceil":
        out = np.ceil(inputs[0]).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "layernorm":
        axis = _normalized_axis_value(node.attrs.get("axis", -1), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("layernorm axis is out of range")
        epsilon = float(node.attrs.get("epsilon", 1.0e-5))
        axes = tuple(range(axis, inputs[0].ndim))
        mean = np.mean(inputs[0], axis=axes, keepdims=True)
        var = np.var(inputs[0], axis=axes, keepdims=True)
        out = (inputs[0] - mean) / np.sqrt(var + epsilon)
        if len(inputs) >= 2:
            out = out * inputs[1]
        if len(inputs) >= 3:
            out = out + inputs[2]
    elif op == "rmsnorm":
        rank = inputs[0].ndim
        axis = int(node.attrs.get("axis", -1))
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise RuntimeError("rmsnorm axis is out of range")
        epsilon = float(node.attrs.get("epsilon", node.attrs.get("eps", 1.0e-5)))
        axes = tuple(range(axis, rank))
        mean_square = np.mean(np.square(inputs[0]), axis=axes, keepdims=True)
        out = inputs[0] / np.sqrt(mean_square + epsilon)
        if len(inputs) >= 2:
            out = out * inputs[1]
        if len(inputs) >= 3:
            out = out + inputs[2]
    elif op == "batchnorm":
        if len(inputs) != 5:
            raise RuntimeError("BatchNormalization requires X, scale, bias, mean, and variance")
        out = _execute_batchnorm(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon=float(node.attrs.get("epsilon", 1.0e-5)))
    elif op == "rotary_embedding":
        if len(inputs) < 3:
            raise RuntimeError("rotary_embedding requires input, cos, and sin tensors")
        out = _execute_rotary_embedding(inputs[0], inputs[1], inputs[2], interleaved=_truthy_int(node.attrs.get("interleaved", 0)))
    elif op == "exp":
        out = np.exp(inputs[0]).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "log":
        out = np.log(inputs[0]).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "reciprocal":
        out = np.reciprocal(inputs[0]).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "sqrt":
        out = np.sqrt(inputs[0])
    elif op in {"reduce_mean", "reduce_sum", "reduce_min", "reduce_max", "reduce_prod"}:
        axes = _reduction_axes_from_sources(node.attrs, inputs[1:], rank=inputs[0].ndim)
        if axes is None:
            raise RuntimeError(f"{op} requires static axes")
        if not axes and _truthy_int(node.attrs.get("noop_with_empty_axes", 0)):
            out = inputs[0]
        elif op == "reduce_sum":
            out = np.sum(inputs[0], axis=tuple(axes), keepdims=bool(int(node.attrs.get("keepdims", 1)))).astype(np.asarray(inputs[0]).dtype, copy=False)
        elif op == "reduce_min":
            out = np.min(inputs[0], axis=tuple(axes), keepdims=bool(int(node.attrs.get("keepdims", 1)))).astype(np.asarray(inputs[0]).dtype, copy=False)
        elif op == "reduce_max":
            out = np.max(inputs[0], axis=tuple(axes), keepdims=bool(int(node.attrs.get("keepdims", 1)))).astype(np.asarray(inputs[0]).dtype, copy=False)
        elif op == "reduce_prod":
            out = np.prod(inputs[0], axis=tuple(axes), keepdims=bool(int(node.attrs.get("keepdims", 1)))).astype(np.asarray(inputs[0]).dtype, copy=False)
        else:
            out = np.mean(inputs[0], axis=tuple(axes), keepdims=bool(int(node.attrs.get("keepdims", 1))))
    elif op == "softmax":
        axis = _normalized_axis_value(node.attrs.get("axis", -1), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("softmax axis is out of range")
        shifted = inputs[0] - np.max(inputs[0], axis=axis, keepdims=True)
        exp = np.exp(shifted)
        out = (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "log_softmax":
        axis = _normalized_axis_value(node.attrs.get("axis", -1), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("log_softmax axis is out of range")
        shifted = inputs[0] - np.max(inputs[0], axis=axis, keepdims=True)
        logsum = np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))
        out = (shifted - logsum).astype(np.asarray(inputs[0]).dtype, copy=False)
    elif op == "argmax":
        axis = _normalized_axis_value(node.attrs.get("axis", 0), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("argmax axis is out of range")
        if _truthy_int(node.attrs.get("select_last_index", 0)):
            flipped = np.flip(inputs[0], axis=axis)
            out = int(inputs[0].shape[axis]) - 1 - np.argmax(flipped, axis=axis)
        else:
            out = np.argmax(inputs[0], axis=axis)
        if _truthy_int(node.attrs.get("keepdims", 1)):
            out = np.expand_dims(out, axis=axis)
        out = np.asarray(out, dtype=np.int64)
    elif op == "argmin":
        axis = _normalized_axis_value(node.attrs.get("axis", 0), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("argmin axis is out of range")
        if _truthy_int(node.attrs.get("select_last_index", 0)):
            flipped = np.flip(inputs[0], axis=axis)
            out = int(inputs[0].shape[axis]) - 1 - np.argmin(flipped, axis=axis)
        else:
            out = np.argmin(inputs[0], axis=axis)
        if _truthy_int(node.attrs.get("keepdims", 1)):
            out = np.expand_dims(out, axis=axis)
        out = np.asarray(out, dtype=np.int64)
    elif op == "topk":
        if len(node.outputs) < 2:
            raise RuntimeError("topk requires value and index outputs")
        k = _topk_k_from_attrs_or_value(node.attrs, inputs[1:])
        axis = _normalized_axis_value(node.attrs.get("axis", -1), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("topk axis is out of range")
        if k is None or k <= 0 or k > int(inputs[0].shape[axis]):
            raise RuntimeError("topk k is out of range")
        values_out, indices_out = _execute_topk(
            inputs[0],
            k,
            axis=axis,
            largest=_truthy_int(node.attrs.get("largest", 1)),
        )
        values[node.outputs[0]] = values_out
        values[node.outputs[1]] = indices_out
        return
    elif op == "top_p_filter":
        p = _top_p_from_attrs_or_value(node.attrs, inputs[1:])
        axis = _normalized_axis_value(node.attrs.get("axis", -1), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("top_p_filter axis is out of range")
        if p is None or not math.isfinite(p) or p <= 0.0 or p > 1.0:
            raise RuntimeError("top_p_filter p is out of range")
        min_tokens = _safe_int(node.attrs.get("min_tokens_to_keep", 1), default=1)
        masked_value = float(node.attrs.get("masked_value", -1.0e30))
        out, mask = _execute_top_p_filter(
            inputs[0],
            p,
            axis=axis,
            min_tokens_to_keep=min_tokens,
            masked_value=masked_value,
        )
        if len(node.outputs) >= 2:
            values[node.outputs[0]] = out
            values[node.outputs[1]] = mask
            return
    elif op == "kv_cache_append":
        axis = _kv_cache_axis(node.attrs, inputs[0].ndim)
        if axis is None:
            raise RuntimeError("kv_cache_append axis is out of range")
        out = np.concatenate((inputs[0], inputs[1]), axis=axis)
    elif op == "dequantize_linear":
        scale, zero_point = _qlinear_params(inputs, node.attrs, input_rank=inputs[0].ndim, default_zero_dtype=str(inputs[0].dtype))
        out = (np.asarray(inputs[0]).astype(np.float32) - zero_point.astype(np.float32)) * scale.astype(np.float32)
    elif op == "quantize_linear":
        target_dtype = str(np.asarray(inputs[2]).dtype) if len(inputs) >= 3 else str(_numpy_dtype(node.attrs.get("dtype", "uint8")))
        scale, zero_point = _qlinear_params(inputs, node.attrs, input_rank=inputs[0].ndim, default_zero_dtype=target_dtype)
        rounded = np.rint(np.asarray(inputs[0], dtype=np.float32) / scale.astype(np.float32) + zero_point.astype(np.float32))
        out = _clip_quantized_array(rounded, target_dtype).astype(_numpy_dtype(target_dtype))
    elif op == "dynamic_quantize_linear":
        if len(node.outputs) != 3:
            raise RuntimeError("DynamicQuantizeLinear requires three outputs")
        y, y_scale, y_zero = _execute_dynamic_quantize_linear(inputs[0])
        values[node.outputs[0]] = y
        values[node.outputs[1]] = y_scale
        values[node.outputs[2]] = y_zero
        return
    elif op == "cast":
        target = _cast_target_dtype(node.attrs)
        if target is None:
            raise RuntimeError("cast requires a supported target dtype")
        out = np.asarray(inputs[0]).astype(_numpy_dtype(target))
    elif op == "cast_like":
        if len(inputs) != 2:
            raise RuntimeError("CastLike requires input and dtype-source tensors")
        out = np.asarray(inputs[0]).astype(_numpy_dtype(str(np.asarray(inputs[1]).dtype)))
    elif op == "size":
        if len(inputs) != 1:
            raise RuntimeError("Size requires one input tensor")
        out = np.asarray(inputs[0].size, dtype=np.int64)
    elif op == "shape":
        shape_range = _shape_op_range(node.attrs, inputs[0].ndim)
        if shape_range is None:
            raise RuntimeError("shape start/end bounds are invalid")
        start, end = shape_range
        out = np.asarray(tuple(int(dim) for dim in inputs[0].shape)[start:end], dtype=np.int64)
    elif op == "range":
        if len(inputs) != 3:
            raise RuntimeError("range requires start, limit, and delta tensors")
        start_arr = np.asarray(inputs[0]).reshape(-1)
        limit_arr = np.asarray(inputs[1]).reshape(-1)
        delta_arr = np.asarray(inputs[2]).reshape(-1)
        if int(start_arr.size) != 1 or int(limit_arr.size) != 1 or int(delta_arr.size) != 1:
            raise RuntimeError("range inputs must be scalar tensors")
        delta_value = delta_arr[0].item()
        try:
            if not math.isfinite(float(delta_value)) or float(delta_value) == 0.0:
                raise RuntimeError("range delta must be finite and nonzero")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("range delta must be finite and nonzero") from exc
        out = np.arange(start_arr[0].item(), limit_arr[0].item(), delta_value, dtype=start_arr.dtype)
    elif op == "constant_of_shape":
        if len(inputs) != 1:
            raise RuntimeError("ConstantOfShape requires one shape tensor")
        shape_value = np.asarray(inputs[0]).reshape(-1)
        try:
            output_shape = tuple(int(dim) for dim in shape_value.tolist())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("ConstantOfShape requires integer shape values") from exc
        if any(dim < 0 for dim in output_shape):
            raise RuntimeError("ConstantOfShape dimensions must be nonnegative")
        scalar = _constant_of_shape_scalar(node.attrs)
        if scalar is None:
            raise RuntimeError("ConstantOfShape value must be scalar")
        out = np.full(output_shape, scalar, dtype=np.asarray(scalar).dtype)
    elif op == "cumsum":
        axis = _axis_from_attrs_or_value(node.attrs, inputs[1:], rank=inputs[0].ndim)
        if axis is None:
            raise RuntimeError("CumSum axis is out of range")
        out = _execute_cumsum(
            inputs[0],
            axis=axis,
            exclusive=_truthy_int(node.attrs.get("exclusive", 0)),
            reverse=_truthy_int(node.attrs.get("reverse", 0)),
        )
    elif op == "pad":
        if _pad_mode(node.attrs) != "constant":
            raise RuntimeError("Pad v1 supports only constant mode")
        pads = _pad_values_from_sources(node.attrs, inputs[1:2], rank=inputs[0].ndim)
        constant_value = _pad_constant_from_sources(node.attrs, inputs[2:3])
        if pads is None or any(value < 0 for value in pads):
            raise RuntimeError("Pad requires nonnegative pads with length 2 * rank")
        if constant_value is None:
            raise RuntimeError("Pad constant value must be scalar")
        out = _execute_pad(inputs[0], pads, constant_value)
    elif op == "flatten":
        axis = int(node.attrs.get("axis", 1))
        rank = inputs[0].ndim
        if axis < 0:
            axis += rank
        leading = int(np.prod(inputs[0].shape[:axis], dtype=np.int64))
        trailing = int(np.prod(inputs[0].shape[axis:], dtype=np.int64))
        out = np.reshape(inputs[0], (leading, trailing))
    elif op == "unsqueeze":
        axes = _axes_from_attrs_or_value(node.attrs, inputs[1:])
        if axes is None:
            raise RuntimeError("unsqueeze requires static axes")
        output_rank = inputs[0].ndim + len(axes)
        out = inputs[0]
        for axis in sorted(axis + output_rank if axis < 0 else axis for axis in axes):
            out = np.expand_dims(out, axis=axis)
    elif op == "squeeze":
        axes = _axes_from_attrs_or_value(node.attrs, inputs[1:])
        if axes is None:
            out = np.squeeze(inputs[0])
        else:
            normalized_axes = sorted((axis + inputs[0].ndim if axis < 0 else axis) for axis in axes)
            out = np.squeeze(inputs[0], axis=tuple(normalized_axes))
    elif op == "expand":
        shape = _expand_target_shape_from_sources(node.attrs, inputs[1:])
        if shape is None:
            raise RuntimeError("expand requires static target shape")
        out = np.broadcast_to(inputs[0], shape)
    elif op == "tile":
        if len(inputs) != 2:
            raise RuntimeError("Tile requires data and repeats tensors")
        try:
            repeats = tuple(int(item) for item in np.asarray(inputs[1]).reshape(-1).tolist())
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Tile repeats must be integer values") from exc
        if len(repeats) != inputs[0].ndim or any(rep < 0 for rep in repeats):
            raise RuntimeError("Tile repeats must match input rank and be nonnegative")
        out = np.tile(inputs[0], repeats)
    elif op == "trilu":
        k = 0
        if len(inputs) >= 2:
            k_arr = np.asarray(inputs[1]).reshape(-1)
            if int(k_arr.size) != 1:
                raise RuntimeError("Trilu k must be a scalar tensor")
            k = int(k_arr[0].item())
        out = np.triu(inputs[0], k=k) if _truthy_int(node.attrs.get("upper", 1)) else np.tril(inputs[0], k=k)
    elif op == "split":
        if not node.outputs:
            raise RuntimeError("split requires at least one output")
        axis = _normalized_axis_value(node.attrs.get("axis", 0), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("split axis is out of range")
        sizes = _split_sizes_from_attrs_or_value(
            node.attrs,
            inputs[1:],
            axis_dim=int(inputs[0].shape[axis]),
            output_count=len(node.outputs),
        )
        if sizes is None:
            raise RuntimeError("split requires static split sizes")
        split_points = np.cumsum(sizes[:-1]).tolist()
        for output_name, arr in zip(node.outputs, np.split(inputs[0], split_points, axis=axis)):
            values[output_name] = arr
        return
    elif op == "where":
        out = np.where(inputs[0], inputs[1], inputs[2])
    elif op == "equal":
        out = np.equal(inputs[0], inputs[1])
    elif op == "less":
        out = np.less(inputs[0], inputs[1])
    elif op == "less_equal":
        out = np.less_equal(inputs[0], inputs[1])
    elif op == "greater":
        out = np.greater(inputs[0], inputs[1])
    elif op == "greater_equal":
        out = np.greater_equal(inputs[0], inputs[1])
    elif op == "not":
        out = np.logical_not(inputs[0])
    elif op == "and":
        out = np.logical_and(inputs[0], inputs[1])
    elif op == "or":
        out = np.logical_or(inputs[0], inputs[1])
    elif op == "gather":
        axis = int(node.attrs.get("axis", 0))
        out = np.take(inputs[0], np.asarray(inputs[1], dtype=np.int64), axis=axis)
    elif op == "gather_elements":
        axis = _normalized_axis_value(node.attrs.get("axis", 0), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("GatherElements axis is out of range")
        out = np.take_along_axis(inputs[0], np.asarray(inputs[1], dtype=np.int64), axis=axis)
    elif op == "gather_nd":
        if _truthy_int(node.attrs.get("batch_dims", 0)):
            raise RuntimeError("GatherND batch_dims is unsupported")
        indices = np.asarray(inputs[1], dtype=np.int64)
        if indices.ndim < 1:
            raise RuntimeError("GatherND indices rank must be at least one")
        index_depth = int(indices.shape[-1])
        if index_depth <= 0 or index_depth > inputs[0].ndim:
            raise RuntimeError("GatherND index depth is out of range")
        out = inputs[0][tuple(indices[..., dim] for dim in range(index_depth))]
    elif op == "scatter_elements":
        reduction = node.attrs.get("reduction", "none")
        if isinstance(reduction, bytes):
            reduction = reduction.decode("utf-8", errors="replace")
        if str(reduction).lower() not in {"", "none"}:
            raise RuntimeError("ScatterElements reduction is unsupported")
        axis = _normalized_axis_value(node.attrs.get("axis", 0), inputs[0].ndim)
        if axis is None:
            raise RuntimeError("ScatterElements axis is out of range")
        out = np.asarray(inputs[0]).copy()
        indices = np.asarray(inputs[1], dtype=np.int64)
        updates = np.asarray(inputs[2])
        if indices.shape != updates.shape or indices.ndim != out.ndim:
            raise RuntimeError("ScatterElements shape mismatch")
        seen_targets: set[tuple[int, ...]] = set()
        axis_dim = int(out.shape[axis])
        for index_coord in np.ndindex(indices.shape):
            target = list(index_coord)
            axis_index = int(indices[index_coord])
            if axis_index < 0:
                axis_index += axis_dim
            if axis_index < 0 or axis_index >= axis_dim:
                raise RuntimeError("ScatterElements index out of range")
            target[axis] = axis_index
            target_tuple = tuple(int(dim) for dim in target)
            if target_tuple in seen_targets:
                raise RuntimeError("ScatterElements duplicate index")
            seen_targets.add(target_tuple)
            out[target_tuple] = updates[index_coord]
    elif op == "scatter_nd":
        reduction = node.attrs.get("reduction", "none")
        if isinstance(reduction, bytes):
            reduction = reduction.decode("utf-8", errors="replace")
        if str(reduction).lower() not in {"", "none"}:
            raise RuntimeError("ScatterND reduction is unsupported")
        out = np.asarray(inputs[0]).copy()
        indices = np.asarray(inputs[1], dtype=np.int64)
        updates = np.asarray(inputs[2])
        if indices.ndim < 1:
            raise RuntimeError("ScatterND indices rank must be at least one")
        index_depth = int(indices.shape[-1])
        if index_depth <= 0 or index_depth > out.ndim:
            raise RuntimeError("ScatterND index depth is out of range")
        expected_updates_shape = tuple(indices.shape[:-1]) + tuple(out.shape[index_depth:])
        if tuple(updates.shape) != expected_updates_shape:
            raise RuntimeError("ScatterND updates shape mismatch")
        seen_targets: set[tuple[int, ...]] = set()
        for index_coord in np.ndindex(indices.shape[:-1]):
            raw_index = np.asarray(indices[index_coord], dtype=np.int64).reshape(-1)
            target = []
            for dim_index, raw_value in enumerate(raw_index.tolist()):
                axis_index = int(raw_value)
                dim_size = int(out.shape[dim_index])
                if axis_index < 0:
                    axis_index += dim_size
                if axis_index < 0 or axis_index >= dim_size:
                    raise RuntimeError("ScatterND index out of range")
                target.append(axis_index)
            target_tuple = tuple(target)
            if target_tuple in seen_targets:
                raise RuntimeError("ScatterND duplicate index")
            seen_targets.add(target_tuple)
            out[target_tuple] = updates[index_coord]
    elif op == "slice":
        params = _slice_params_from_sources(node.attrs, inputs[1:], rank=inputs[0].ndim)
        if params is None:
            raise RuntimeError("slice requires static starts and ends")
        starts, ends, axes, steps = params
        slices = [slice(None)] * inputs[0].ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            slices[axis] = slice(start, end, step)
        out = inputs[0][tuple(slices)]
    elif op == "attention_score":
        scale = node.attrs.get("scale")
        if scale is None:
            scale = 1.0 / math.sqrt(float(qinputs[0].shape[-1]))
        out = np.matmul(qinputs[0], np.swapaxes(qinputs[1], -1, -2)) * float(scale)
    elif op == "attention_value":
        out = np.matmul(qinputs[0], qinputs[1])
    elif op == "reshape":
        shape = tuple(int(dim) for dim in node.attrs.get("shape", inputs[1].tolist() if len(inputs) > 1 else ()))
        out = np.reshape(inputs[0], shape)
    elif op == "transpose":
        perm = node.attrs.get("perm")
        out = np.transpose(inputs[0], axes=None if perm is None else tuple(int(dim) for dim in perm))
    elif op == "conv2d":
        out = _conv2d_nchw(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None, node.attrs)
    elif op == "depthwise_conv2d":
        out = _depthwise_conv2d_nchw(inputs[0], inputs[1], inputs[2] if len(inputs) > 2 else None, node.attrs)
    else:
        raise RuntimeError(f"unimplemented NN op {op!r}")
    output_quantization = node.attrs.get("output_quantization")
    if output_quantization:
        out = _requantize_array(out, dict(output_quantization))
    if not node.outputs:
        raise RuntimeError(f"NN node {node.node_id} has no output")
    values[node.outputs[0]] = out


def _execute_topk(array: Any, k: int, *, axis: int, largest: bool) -> tuple[Any, Any]:
    np = _numpy()
    arr = np.asarray(array)
    order_source = -arr if largest else arr
    indices = np.argsort(order_source, axis=axis, kind="stable")
    slices = [slice(None)] * arr.ndim
    slices[axis] = slice(0, int(k))
    top_indices = np.asarray(indices[tuple(slices)], dtype=np.int64)
    top_values = np.take_along_axis(arr, top_indices, axis=axis)
    return top_values, top_indices


def _execute_top_p_filter(
    array: Any,
    p: float,
    *,
    axis: int,
    min_tokens_to_keep: int,
    masked_value: float,
) -> tuple[Any, Any]:
    np = _numpy()
    logits = np.asarray(array)
    order = np.argsort(-logits, axis=axis, kind="stable")
    sorted_logits = np.take_along_axis(logits, order, axis=axis)
    shifted = sorted_logits - np.max(sorted_logits, axis=axis, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities = probabilities / np.sum(probabilities, axis=axis, keepdims=True)
    cumulative = np.cumsum(probabilities, axis=axis)
    keep_sorted = (cumulative - probabilities) < float(p)
    if min_tokens_to_keep > 0:
        rank = np.arange(logits.shape[axis])
        shape = [1] * logits.ndim
        shape[axis] = logits.shape[axis]
        keep_sorted = np.logical_or(keep_sorted, rank.reshape(shape) < int(min_tokens_to_keep))
    keep = np.zeros_like(keep_sorted, dtype=bool)
    np.put_along_axis(keep, order, keep_sorted, axis=axis)
    filtered = np.where(keep, logits, np.asarray(masked_value, dtype=logits.dtype))
    return filtered, keep


def _execute_spf32_node(op: str, inputs: Sequence[Any], *, spf32_engine: Optional[_PSOpEngine]) -> Any:
    np = _numpy()
    source = np.asarray(inputs[0], dtype=np.uint32)
    elements = int(source.size)
    engine = spf32_engine or _new_spf32_ps_engine(None)
    flat_inputs = [np.ascontiguousarray(value, dtype=np.uint32).reshape(-1) for value in inputs]
    out = engine.compute(op, flat_inputs, elements)
    return np.ascontiguousarray(out, dtype=np.uint32).reshape(source.shape)


def _conv2d_nchw(x: Any, w: Any, bias: Any, attrs: dict[str, Any]) -> Any:
    np = _numpy()
    x = np.asarray(x)
    w = np.asarray(w)
    normalized = _conv_normalized_attrs(attrs)
    if normalized["error"] is not None:
        raise RuntimeError(str(normalized["error"]))
    if int(normalized["group"]) != 1:
        raise RuntimeError("conv2d PS oracle does not support grouped convolution")
    strides = normalized["strides"]
    pads = normalized["pads"]
    dilations = normalized["dilations"]
    n, c_in, h, width = x.shape
    c_out, _, kh, kw = w.shape
    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])), mode="constant")
    out_h = _conv_output_dim(h, kh, pads[0], pads[2], strides[0], dilations[0])
    out_w = _conv_output_dim(width, kw, pads[1], pads[3], strides[1], dilations[1])
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.result_type(x, w))
    for batch in range(n):
        for oc in range(c_out):
            for oy in range(out_h):
                for ox in range(out_w):
                    total = 0
                    base_y = oy * strides[0]
                    base_x = ox * strides[1]
                    for ic in range(c_in):
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = base_y + ky * dilations[0]
                                ix = base_x + kx * dilations[1]
                                total += padded[batch, ic, iy, ix] * w[oc, ic, ky, kx]
                    out[batch, oc, oy, ox] = total
            if bias is not None:
                out[batch, oc] += bias[oc]
    return out


def _execute_batchnorm(x: Any, scale: Any, bias: Any, mean: Any, variance: Any, *, epsilon: float) -> Any:
    np = _numpy()
    arr = np.asarray(x)
    if arr.ndim < 2:
        raise RuntimeError("BatchNormalization requires rank >= 2 input")
    channel_count = int(arr.shape[1])
    params = [np.asarray(item, dtype=arr.dtype).reshape(-1) for item in (scale, bias, mean, variance)]
    if any(int(param.shape[0]) != channel_count for param in params):
        raise RuntimeError("BatchNormalization parameter length must match channel count")
    reshape = (1, channel_count, *([1] * (arr.ndim - 2)))
    scale_arr, bias_arr, mean_arr, variance_arr = [param.reshape(reshape) for param in params]
    return (scale_arr * (arr - mean_arr) / np.sqrt(variance_arr + np.asarray(epsilon, dtype=arr.dtype)) + bias_arr).astype(arr.dtype, copy=False)


def _depthwise_conv2d_nchw(x: Any, w: Any, bias: Any, attrs: dict[str, Any]) -> Any:
    np = _numpy()
    x = np.asarray(x)
    w = np.asarray(w)
    normalized = _conv_normalized_attrs(attrs)
    if normalized["error"] is not None:
        raise RuntimeError(str(normalized["error"]))
    strides = normalized["strides"]
    pads = normalized["pads"]
    dilations = normalized["dilations"]
    n, channels, h, width = x.shape
    c_out, _, kh, kw = w.shape
    if c_out % channels != 0:
        raise RuntimeError("depthwise_conv2d output channels must be a multiple of input channels")
    channel_multiplier = c_out // channels
    padded = np.pad(x, ((0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])), mode="constant")
    out_h = _conv_output_dim(h, kh, pads[0], pads[2], strides[0], dilations[0])
    out_w = _conv_output_dim(width, kw, pads[1], pads[3], strides[1], dilations[1])
    out = np.zeros((n, c_out, out_h, out_w), dtype=np.result_type(x, w))
    for batch in range(n):
        for ch in range(channels):
            for multiplier in range(channel_multiplier):
                oc = ch * channel_multiplier + multiplier
                for oy in range(out_h):
                    for ox in range(out_w):
                        total = 0
                        base_y = oy * strides[0]
                        base_x = ox * strides[1]
                        for ky in range(kh):
                            for kx in range(kw):
                                iy = base_y + ky * dilations[0]
                                ix = base_x + kx * dilations[1]
                                total += padded[batch, ch, iy, ix] * w[oc, 0, ky, kx]
                        out[batch, oc, oy, ox] = total
                if bias is not None:
                    out[batch, oc] += bias[oc]
    return out


def _trace_row(
    graph: NNGraph,
    region: NNRegion,
    *,
    region_index: int,
    action: dict[str, Any],
    elapsed_seconds: float,
    residency_before: dict[str, str],
    residency_after: dict[str, str],
    output_names: Sequence[str],
    fabric_result: Optional[dict[str, Any]] = None,
    foundry_result: Optional[dict[str, Any]] = None,
    tensor_metadata: Optional[dict[str, Any]] = None,
    released_values: Optional[Sequence[dict[str, Any]]] = None,
    runtime_context: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    fabric_summary = dict((fabric_result or {}).get("summary") or {})
    timed_transfer = int(fabric_summary.get("timed_transfer_bytes") or 0)
    setup_transfer = 0
    if fabric_summary:
        setup_transfer = int(fabric_summary.get("setup_transfer_bytes") or 0)
    forced_materialization = int(fabric_summary.get("final_transfer_bytes") or 0)
    return {
        "schema": PAN_NN_TRACE_SCHEMA,
        "graph": graph.name,
        "region_index": int(region_index),
        "region": region.to_json(),
        "selected_action": action["selected_action"],
        "placement_action": action,
        "elapsed_seconds": float(elapsed_seconds),
        "residency_before": dict(sorted(residency_before.items())),
        "residency_after": dict(sorted(residency_after.items())),
        "outputs": list(output_names),
        "setup_transfer_bytes": setup_transfer,
        "timed_transfer_bytes": timed_transfer,
        "forced_materialization_bytes": forced_materialization,
        "validation": None,
        "fabric_execution": fabric_result,
        "foundry": foundry_result,
        "tensor_metadata": dict(tensor_metadata or {}),
        "released_values": list(released_values or ()),
        "runtime_context": dict(runtime_context or {}),
        "features": _nn_trace_features(
            region,
            action,
            residency_before=residency_before,
            residency_after=residency_after,
            tensor_metadata=dict(tensor_metadata or {}),
            elapsed_seconds=elapsed_seconds,
            setup_transfer_bytes=setup_transfer,
            timed_transfer_bytes=timed_transfer,
            forced_materialization_bytes=forced_materialization,
            fabric_result=fabric_result,
            foundry_result=foundry_result,
        ),
    }


def _training_row(row: dict[str, Any]) -> dict[str, Any]:
    region = row["region"]
    labels = _nn_training_labels_from_row(row)
    return {
        "schema": PAN_NN_TRAINING_SCHEMA,
        "graph": row["graph"],
        "region_id": region["region_id"],
        "region_signature": region["signature"],
        "ops": list(region["ops"]),
        "selected_action": row["selected_action"],
        "output_residency": labels["preferred_residency"],
        "materialization_need": labels["materialization_needed"],
        "compile_priority": 1.0 if labels["compile_priority"] == "compile_better_for_next_time" else 0.0,
        "elapsed_seconds": float(row.get("elapsed_seconds") or 0.0),
        "setup_transfer_bytes": int(row.get("setup_transfer_bytes") or 0),
        "timed_transfer_bytes": int(row.get("timed_transfer_bytes") or 0),
        "forced_materialization_bytes": int(row.get("forced_materialization_bytes") or 0),
        "validation_result": dict(row.get("validation") or {}),
        "validation_ok": labels["validation_ok"],
        "negative_control": not bool(labels["validation_ok"]),
        "negative_reason": None if labels["validation_ok"] else "validation_failed",
        "runtime_context": dict(row.get("runtime_context") or {}),
        "board_state": _nn_board_state_from_row(row),
        "daemon_elapsed_seconds": _nn_daemon_seconds_from_row(row),
        "transfer_estimate_bytes": _nn_transfer_bytes_from_row(row),
        "features": dict(row.get("features") or _nn_training_features_from_row(row)),
        "labels": labels,
        "tensor_metadata": dict(row.get("tensor_metadata") or {}),
        "materialization_action": dict(row.get("materialization_action") or {}),
    }


def _annotate_materialization_actions(
    trace: list[dict[str, Any]],
    *,
    graph_outputs: Sequence[str],
    include_output_digest: bool,
    materialization_reason: Optional[str],
    policy_materialization: bool = False,
) -> None:
    graph_output_set = set(graph_outputs)
    for row in trace:
        outputs = [str(name) for name in row.get("outputs") or ()]
        visible_outputs = [name for name in outputs if name in graph_output_set]
        if visible_outputs:
            explicit_materialize_reason = (
                None if materialization_reason in {"neural_policy", "neural_policy_keep_resident"} else materialization_reason
            )
            materialize, reason, policy_status = _resolve_materialization_policy(
                dict(row.get("placement_action") or {}),
                visible_outputs=visible_outputs,
                default_include_output_digest=include_output_digest,
                validate_output=bool((row.get("validation") or {}).get("ok") is not None),
                materialize_reason=explicit_materialize_reason,
                policy_materialization=policy_materialization,
            )
            row["materialization_action"] = {
                "schema": "pan_nn_materialization_action_v1",
                "action": "materialize" if materialize else "keep_resident",
                "reason": reason,
                "outputs": visible_outputs,
                "requires_user_visible_output": True,
                "policy_materialization": policy_status,
            }
            continue
        row["materialization_action"] = {
            "schema": "pan_nn_materialization_action_v1",
            "action": "keep_resident",
            "reason": "internal_activation",
            "outputs": outputs,
            "requires_user_visible_output": False,
            "policy_materialization": {
                "schema": "pan_nn_policy_materialization_v1",
                "enabled": bool(policy_materialization),
                "applied": False,
                "rejected_reason": "internal_activation",
            },
        }


def _resolve_trace_output_materialization(
    trace: Sequence[dict[str, Any]],
    *,
    graph_outputs: Sequence[str],
    default_include_output_digest: bool,
    validate_output: bool,
    materialize_reason: Optional[str],
    policy_materialization: bool,
) -> tuple[bool, Optional[str], dict[str, Any]]:
    graph_output_set = set(graph_outputs)
    selected: Optional[tuple[bool, Optional[str], dict[str, Any]]] = None
    for row in trace:
        outputs = [str(name) for name in row.get("outputs") or ()]
        visible_outputs = [name for name in outputs if name in graph_output_set]
        if not visible_outputs:
            continue
        resolved = _resolve_materialization_policy(
            dict(row.get("placement_action") or {}),
            visible_outputs=visible_outputs,
            default_include_output_digest=default_include_output_digest,
            validate_output=validate_output,
            materialize_reason=materialize_reason,
            policy_materialization=policy_materialization,
        )
        if resolved[0]:
            return resolved
        selected = resolved
    if selected is not None:
        return selected
    return _default_materialization_resolution(
        default_include_output_digest=default_include_output_digest,
        validate_output=validate_output,
        materialize_reason=materialize_reason,
        policy_status={
            "schema": "pan_nn_policy_materialization_v1",
            "enabled": bool(policy_materialization),
            "applied": False,
            "rejected_reason": "no_graph_output_region",
        },
    )


def _resolve_materialization_policy(
    action: dict[str, Any],
    *,
    visible_outputs: Sequence[str],
    default_include_output_digest: bool,
    validate_output: bool,
    materialize_reason: Optional[str],
    policy_materialization: bool,
) -> tuple[bool, Optional[str], dict[str, Any]]:
    policy_status = _policy_materialization_status(
        action,
        enabled=policy_materialization,
        visible_outputs=visible_outputs,
        validate_output=validate_output,
        materialize_reason=materialize_reason,
    )
    if policy_status.get("applied"):
        requested = bool(policy_status.get("requested_materialization"))
        return requested, "neural_policy" if requested else "neural_policy_keep_resident", policy_status
    return _default_materialization_resolution(
        default_include_output_digest=default_include_output_digest,
        validate_output=validate_output,
        materialize_reason=materialize_reason,
        policy_status=policy_status,
    )


def _default_materialization_resolution(
    *,
    default_include_output_digest: bool,
    validate_output: bool,
    materialize_reason: Optional[str],
    policy_status: dict[str, Any],
) -> tuple[bool, Optional[str], dict[str, Any]]:
    include = bool(validate_output or default_include_output_digest)
    reason = materialize_reason or ("validation" if validate_output else ("user_output" if include else "no_output_digest"))
    return include, reason, policy_status


def _policy_materialization_status(
    action: dict[str, Any],
    *,
    enabled: bool,
    visible_outputs: Sequence[str],
    validate_output: bool,
    materialize_reason: Optional[str],
) -> dict[str, Any]:
    status: dict[str, Any] = {
        "schema": "pan_nn_policy_materialization_v1",
        "enabled": bool(enabled),
        "applied": False,
        "outputs": list(visible_outputs),
    }
    decision = action.get("policy_decision")
    if isinstance(decision, dict):
        status["decision_materialization_needed"] = decision.get("materialization_needed")
        status["policy"] = decision.get("policy")
    if not enabled:
        status["rejected_reason"] = "policy_materialization_disabled"
        return status
    if validate_output:
        status["rejected_reason"] = "validation_requires_materialization"
        return status
    if not visible_outputs:
        status["rejected_reason"] = "no_visible_outputs"
        return status
    if not isinstance(decision, dict):
        status["rejected_reason"] = "no_policy_decision"
        return status
    if not decision.get("override_allowed"):
        status["rejected_reason"] = "policy_not_promoted"
        return status
    hard_rejection = str(decision.get("rejected_reason") or "")
    if hard_rejection:
        status["rejected_reason"] = hard_rejection
        return status
    if materialize_reason:
        status["rejected_reason"] = "explicit_materialization_reason"
        return status
    requested = decision.get("materialization_needed")
    if not isinstance(requested, bool):
        status["rejected_reason"] = "policy_no_materialization_head"
        return status
    status["applied"] = True
    status["requested_materialization"] = bool(requested)
    return status


def _nn_trace_features(
    region: NNRegion,
    action: dict[str, Any],
    *,
    residency_before: dict[str, str],
    residency_after: dict[str, str],
    tensor_metadata: dict[str, Any],
    elapsed_seconds: float,
    setup_transfer_bytes: int,
    timed_transfer_bytes: int,
    forced_materialization_bytes: int,
    fabric_result: Optional[dict[str, Any]],
    foundry_result: Optional[dict[str, Any]],
) -> dict[str, float]:
    residencies_before = list(residency_before.values())
    residencies_after = list(residency_after.values())
    tensor_elements = [int((row or {}).get("elements") or 0) for row in tensor_metadata.values()]
    ops = region.ops
    fabric_dispatch = _nn_fabric_dispatch_features(fabric_result, op_count=len(ops))
    foundry_features = _nn_foundry_estimate_features(foundry_result)
    tensor_features = _nn_tensor_metadata_features(tensor_metadata)
    mlp_features = _mlp_feature_counts(ops, reason=region.reason)
    attention_features = _attention_feature_counts(ops, reason=region.reason)
    decode_features = _decode_feature_counts(ops)
    conv_region = region.reason == "fused_conv_region" or any(op in {"conv2d", "depthwise_conv2d"} for op in ops)
    return {
        "op_count": float(len(ops)),
        "matmul_count": float(sum(1 for op in ops if op in {"matmul", "gemm"})),
        "conv_count": float(sum(1 for op in ops if op in {"conv2d", "depthwise_conv2d"})),
        "conv_region_count": float(conv_region),
        "activation_count": float(
            sum(1 for op in ops if op in {"relu", "gelu", "bias_gelu", "quick_gelu", "silu", "sigmoid", "tanh", "erf", "sin", "cos", "softmax", "log_softmax"})
        ),
        "spf_count": float(sum(1 for op in ops if op.startswith("spf"))),
        **mlp_features,
        **attention_features,
        **decode_features,
        **fabric_dispatch,
        "max_tensor_elements": float(max(tensor_elements) if tensor_elements else 0),
        "total_tensor_elements": float(sum(tensor_elements)),
        "input_daemon_residency_count": float(sum(1 for value in residencies_before if value == "daemon")),
        "input_ps_residency_count": float(sum(1 for value in residencies_before if value == "ps")),
        "output_daemon_residency_count": float(sum(1 for value in residencies_after if value == "daemon")),
        "output_ps_residency_count": float(sum(1 for value in residencies_after if value == "ps")),
        "elapsed_seconds": float(elapsed_seconds),
        "setup_transfer_bytes": float(setup_transfer_bytes),
        "timed_transfer_bytes": float(timed_transfer_bytes),
        "forced_materialization_bytes": float(forced_materialization_bytes),
        "foundry_queued": float((foundry_result or {}).get("queued") is not None),
        "selected_run_ps": float(action.get("selected_action") == "run_ps_now"),
        "selected_daemon_resident": float(action.get("selected_action") == "run_daemon_resident_now"),
        **foundry_features,
        **tensor_features,
    }


def _nn_training_features_from_row(row: dict[str, Any]) -> dict[str, float]:
    region = dict(row.get("region") or {})
    ops = tuple(str(op) for op in region.get("ops") or ())
    reason = str(region.get("reason") or "")
    metadata = dict(row.get("tensor_metadata") or {})
    tensor_elements = [int((item or {}).get("elements") or 0) for item in metadata.values()]
    before = dict(row.get("residency_before") or {})
    after = dict(row.get("residency_after") or {})
    fabric_dispatch = _nn_fabric_dispatch_features(row.get("fabric_execution"), op_count=len(ops))
    foundry_features = _nn_foundry_estimate_features(row.get("foundry"))
    tensor_features = _nn_tensor_metadata_features(metadata)
    mlp_features = _mlp_feature_counts(ops, reason=reason)
    attention_features = _attention_feature_counts(ops, reason=reason)
    decode_features = _decode_feature_counts(ops)
    conv_region = reason == "fused_conv_region" or any(op in {"conv2d", "depthwise_conv2d"} for op in ops)
    return {
        "op_count": float(len(ops)),
        "matmul_count": float(sum(1 for op in ops if op in {"matmul", "gemm"})),
        "conv_count": float(sum(1 for op in ops if op in {"conv2d", "depthwise_conv2d"})),
        "conv_region_count": float(conv_region),
        "activation_count": float(
            sum(1 for op in ops if op in {"relu", "gelu", "bias_gelu", "quick_gelu", "silu", "sigmoid", "tanh", "erf", "sin", "cos", "softmax", "log_softmax"})
        ),
        "spf_count": float(sum(1 for op in ops if op.startswith("spf"))),
        **mlp_features,
        **attention_features,
        **decode_features,
        **fabric_dispatch,
        "max_tensor_elements": float(max(tensor_elements) if tensor_elements else 0),
        "total_tensor_elements": float(sum(tensor_elements)),
        "input_daemon_residency_count": float(sum(1 for value in before.values() if value == "daemon")),
        "input_ps_residency_count": float(sum(1 for value in before.values() if value == "ps")),
        "output_daemon_residency_count": float(sum(1 for value in after.values() if value == "daemon")),
        "output_ps_residency_count": float(sum(1 for value in after.values() if value == "ps")),
        "elapsed_seconds": float(row.get("elapsed_seconds") or 0.0),
        "setup_transfer_bytes": float(row.get("setup_transfer_bytes") or 0),
        "timed_transfer_bytes": float(row.get("timed_transfer_bytes") or 0),
        "forced_materialization_bytes": float(row.get("forced_materialization_bytes") or 0),
        "foundry_queued": float((row.get("foundry") or {}).get("queued") is not None),
        "selected_run_ps": float(row.get("selected_action") == "run_ps_now"),
        "selected_daemon_resident": float(row.get("selected_action") == "run_daemon_resident_now"),
        **foundry_features,
        **tensor_features,
    }


def _nn_tensor_metadata_features(metadata: dict[str, Any]) -> dict[str, float]:
    features = {
        "static_shape_dim_count": 0.0,
        "symbolic_shape_dim_count": 0.0,
        "dynamic_shape_dim_count": 0.0,
        "shape_binding_count": 0.0,
        "float_tensor_count": 0.0,
        "float16_tensor_count": 0.0,
        "float32_tensor_count": 0.0,
        "float64_tensor_count": 0.0,
        "low_precision_float_tensor_count": 0.0,
        "integer_tensor_count": 0.0,
        "bool_tensor_count": 0.0,
        "unknown_dtype_tensor_count": 0.0,
        "quantized_tensor_count": 0.0,
        "weight_tensor_count": 0.0,
        "retained_weight_tensor_count": 0.0,
        "resident_weight_tensor_count": 0.0,
        "retained_feedback_tensor_count": 0.0,
        "resident_feedback_tensor_count": 0.0,
        "kv_cache_feedback_tensor_count": 0.0,
        "resident_kv_cache_feedback_tensor_count": 0.0,
    }
    for item in metadata.values():
        if not isinstance(item, dict):
            continue
        residency = str(item.get("residency"))
        is_weight = bool(item.get("weight"))
        if is_weight:
            features["weight_tensor_count"] += 1.0
            if bool(item.get("retained")):
                features["retained_weight_tensor_count"] += 1.0
            if residency in {"daemon", "xrt_bo", "dma_buf", "catalog_core"}:
                features["resident_weight_tensor_count"] += 1.0
        is_feedback = bool(item.get("retained_feedback"))
        if is_feedback:
            features["retained_feedback_tensor_count"] += 1.0
            if residency in {"daemon", "xrt_bo", "dma_buf", "catalog_core"}:
                features["resident_feedback_tensor_count"] += 1.0
            feedback_name = str(item.get("feedback_input_name") or item.get("name") or "").lower()
            if "cache" in feedback_name or feedback_name in {"k", "v"}:
                features["kv_cache_feedback_tensor_count"] += 1.0
                if residency in {"daemon", "xrt_bo", "dma_buf", "catalog_core"}:
                    features["resident_kv_cache_feedback_tensor_count"] += 1.0
        for constraint in item.get("shape_constraints") or ():
            if not isinstance(constraint, dict):
                continue
            kind = str(constraint.get("kind", "unknown"))
            if kind == "static":
                features["static_shape_dim_count"] += 1.0
            elif kind == "symbolic":
                features["symbolic_shape_dim_count"] += 1.0
            elif kind == "dynamic":
                features["dynamic_shape_dim_count"] += 1.0
        bindings = item.get("shape_bindings") or {}
        if isinstance(bindings, dict):
            features["shape_binding_count"] += float(len(bindings))
        quantization = item.get("quantization") or {}
        if isinstance(quantization, dict) and quantization:
            features["quantized_tensor_count"] += 1.0
        try:
            dtype_kind = _numpy_dtype(str(item.get("dtype", ""))).kind
        except (TypeError, ValueError):
            dtype_kind = "?"
        if dtype_kind == "f":
            features["float_tensor_count"] += 1.0
            dtype_name = str(_numpy_dtype(str(item.get("dtype", ""))))
            if dtype_name in {"float16", "float32", "float64"}:
                features[f"{dtype_name}_tensor_count"] += 1.0
            if dtype_name == "float16":
                features["low_precision_float_tensor_count"] += 1.0
        elif dtype_kind in {"i", "u"}:
            features["integer_tensor_count"] += 1.0
        elif dtype_kind == "b":
            features["bool_tensor_count"] += 1.0
        else:
            features["unknown_dtype_tensor_count"] += 1.0
    return features


def _nn_fabric_dispatch_features(fabric_result: Any, *, op_count: int) -> dict[str, float]:
    if not isinstance(fabric_result, dict):
        return {
            "fabric_graph_op_count": 0.0,
            "daemon_graph_packet_count": 0.0,
            "ops_per_daemon_packet": 0.0,
            "dispatch_amortized_op_count": 0.0,
        }
    graph = dict(fabric_result.get("graph") or {})
    graph_ops = graph.get("ops") or ()
    graph_op_count = len(graph_ops) if graph_ops else int(op_count)
    summary = dict(fabric_result.get("summary") or {})
    packet_count = int(summary.get("timed_graph_packets") or summary.get("run_daemon_graph_count") or 0)
    ops_per_packet = float(graph_op_count) / float(packet_count) if packet_count > 0 else 0.0
    return {
        "fabric_graph_op_count": float(graph_op_count),
        "daemon_graph_packet_count": float(packet_count),
        "ops_per_daemon_packet": ops_per_packet,
        "dispatch_amortized_op_count": float(max(0, int(graph_op_count) - int(packet_count))),
    }


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _nn_foundry_estimate_features(foundry_result: Any) -> dict[str, float]:
    features = {
        "foundry_estimated_latency_s": 0.0,
        "foundry_estimated_energy_j": 0.0,
        "foundry_estimated_lut": 0.0,
        "foundry_estimated_dsp": 0.0,
        "foundry_estimated_bram": 0.0,
        "foundry_estimated_uram": 0.0,
        "foundry_estimated_transfer_avoidance_bytes": 0.0,
        "foundry_estimated_transfer_avoidance_s": 0.0,
        "foundry_estimated_reconfig_ms": 0.0,
        "foundry_estimated_reconfig_s": 0.0,
        "foundry_estimated_compile_minutes": 0.0,
    }
    if not isinstance(foundry_result, dict):
        return features
    candidate = foundry_result.get("candidate")
    if not isinstance(candidate, dict):
        return features
    estimate = candidate.get("estimate")
    if not isinstance(estimate, dict):
        return features
    resource = estimate.get("resource_estimate") if isinstance(estimate.get("resource_estimate"), dict) else estimate
    energy = estimate.get("energy_estimate") if isinstance(estimate.get("energy_estimate"), dict) else estimate
    transfer = estimate.get("transfer_avoidance") if isinstance(estimate.get("transfer_avoidance"), dict) else estimate
    reconfig = estimate.get("reconfiguration") if isinstance(estimate.get("reconfiguration"), dict) else estimate
    compile_estimate = estimate.get("compile_estimate") if isinstance(estimate.get("compile_estimate"), dict) else estimate
    latency = estimate.get("latency_estimate") if isinstance(estimate.get("latency_estimate"), dict) else estimate
    features.update(
        {
            "foundry_estimated_latency_s": _safe_float(latency.get("seconds", estimate.get("latency_s"))),
            "foundry_estimated_energy_j": _safe_float(energy.get("joules", estimate.get("energy_j"))),
            "foundry_estimated_lut": _safe_float(resource.get("lut")),
            "foundry_estimated_dsp": _safe_float(resource.get("dsp")),
            "foundry_estimated_bram": _safe_float(resource.get("bram")),
            "foundry_estimated_uram": _safe_float(resource.get("uram")),
            "foundry_estimated_transfer_avoidance_bytes": _safe_float(
                transfer.get("avoided_bytes", estimate.get("transfer_avoidance_bytes"))
            ),
            "foundry_estimated_transfer_avoidance_s": _safe_float(
                transfer.get("seconds", estimate.get("transfer_avoidance_s"))
            ),
            "foundry_estimated_reconfig_ms": _safe_float(reconfig.get("milliseconds", estimate.get("reconfig_ms"))),
            "foundry_estimated_reconfig_s": _safe_float(reconfig.get("seconds")),
            "foundry_estimated_compile_minutes": _safe_float(compile_estimate.get("minutes", estimate.get("compile_minutes"))),
        }
    )
    return features


def _nn_training_labels_from_row(row: dict[str, Any]) -> dict[str, Any]:
    compile_requested = (
        row.get("placement_action", {}).get("queued_followup_action") == "compile_better_for_next_time"
        or (row.get("foundry") or {}).get("queued") is not None
        or float(row.get("compile_priority") or 0.0) > 0.0
    )
    output_residency = str(row.get("output_residency") or _output_residency_from_trace(row))
    materialization_action = dict(row.get("materialization_action") or {})
    materialization_needed = (
        materialization_action.get("action") == "materialize"
        or bool(row.get("materialization_need"))
        or bool(row.get("forced_materialization_bytes"))
    )
    return {
        "placement_action": str(row.get("selected_action") or "unknown"),
        "preferred_residency": output_residency,
        "output_residency": output_residency,
        "materialization_needed": bool(materialization_needed),
        "compile_priority": "compile_better_for_next_time" if compile_requested else "none",
        "validation_ok": bool((row.get("validation") or {}).get("ok", True)),
    }


def _nn_policy_heads(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [dict(row.get("labels") or _nn_training_labels_from_row(row)) for row in rows]
    latency_values = [float(row.get("elapsed_seconds") or 0.0) for row in rows]
    transfer_values = [float(_nn_transfer_bytes_from_row(row)) for row in rows]
    feature_rows = [dict(row.get("features") or _nn_training_features_from_row(row)) for row in rows]
    energy_values = _nn_feature_values(feature_rows, "foundry_estimated_energy_j")
    lut_values = _nn_feature_values(feature_rows, "foundry_estimated_lut")
    dsp_values = _nn_feature_values(feature_rows, "foundry_estimated_dsp")
    bram_values = _nn_feature_values(feature_rows, "foundry_estimated_bram")
    uram_values = _nn_feature_values(feature_rows, "foundry_estimated_uram")
    reconfig_ms_values = _nn_feature_values(feature_rows, "foundry_estimated_reconfig_ms")
    transfer_avoidance_values = _nn_feature_values(feature_rows, "foundry_estimated_transfer_avoidance_bytes")
    return {
        "placement_action": {
            "kind": "linear_latency_score",
            "classes": _nn_label_counts(labels, "placement_action"),
        },
        "preferred_residency": {
            "kind": "categorical_majority",
            "classes": _nn_label_counts(labels, "preferred_residency"),
        },
        "materialization_needed": {
            "kind": "binary_rate",
            "positive_count": sum(1 for label in labels if bool(label.get("materialization_needed"))),
            "sample_count": len(labels),
        },
        "compile_priority": {
            "kind": "categorical_priority",
            "classes": _nn_label_counts(labels, "compile_priority"),
        },
        "latency_estimate": {
            "kind": "regression_mean",
            "mean_seconds": float(sum(latency_values) / len(latency_values)) if latency_values else 0.0,
            "sample_count": len(rows),
        },
        "transfer_estimate": {
            "kind": "regression_mean",
            "mean_bytes": float(sum(transfer_values) / len(transfer_values)) if transfer_values else 0.0,
            "sample_count": len(rows),
        },
        "energy_estimate": {
            "kind": "regression_mean",
            "mean_joules": _mean(energy_values),
            "sample_count": len(rows),
        },
        "resource_estimate": {
            "kind": "resource_mean",
            "mean_lut": _mean(lut_values),
            "mean_dsp": _mean(dsp_values),
            "mean_bram": _mean(bram_values),
            "mean_uram": _mean(uram_values),
            "sample_count": len(rows),
        },
        "reconfiguration_estimate": {
            "kind": "regression_mean",
            "mean_milliseconds": _mean(reconfig_ms_values),
            "sample_count": len(rows),
        },
        "transfer_avoidance_estimate": {
            "kind": "regression_mean",
            "mean_avoided_bytes": _mean(transfer_avoidance_values),
            "sample_count": len(rows),
        },
        "validation_risk": {
            "kind": "binary_failure_rate",
            "failure_count": sum(1 for label in labels if not bool(label.get("validation_ok", True))),
            "sample_count": len(labels),
            "risk_rate": (
                float(sum(1 for label in labels if not bool(label.get("validation_ok", True)))) / float(len(labels))
                if labels
                else 0.0
            ),
        },
    }


def _nn_feature_values(rows: Sequence[dict[str, Any]], key: str) -> list[float]:
    return [_safe_float(row.get(key)) for row in rows]


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _nn_label_counts(labels: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        value = str(label.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _output_residency_from_trace(row: dict[str, Any]) -> str:
    after = row.get("residency_after", {})
    for output in row.get("outputs", []):
        if output in after:
            return str(after[output])
    return "unknown"


def _nn_board_state_from_row(row: dict[str, Any]) -> dict[str, Any]:
    context = dict(row.get("runtime_context") or {})
    placement = dict(row.get("placement_action") or {})
    hint = dict(placement.get("scheduler_hint") or {})
    quarantine = hint.get("quarantine")
    return {
        "target": context.get("target", "unknown"),
        "active_app": context.get("active_app", "unknown"),
        "active_app_requested": context.get("active_app_requested", context.get("active_app", "unknown")),
        "fabric_backend": context.get("fabric_backend", "unknown"),
        "quarantine_state": "active" if quarantine else "none",
        "quarantine": quarantine,
    }


def _nn_daemon_seconds_from_row(row: dict[str, Any]) -> float:
    summary = dict((row.get("fabric_execution") or {}).get("summary") or {})
    return float(summary.get("observed_run_daemon_seconds") or 0.0)


def _nn_transfer_bytes_from_row(row: dict[str, Any]) -> int:
    return (
        int(row.get("setup_transfer_bytes") or 0)
        + int(row.get("timed_transfer_bytes") or 0)
        + int(row.get("forced_materialization_bytes") or 0)
    )


def _resolve_validation_policy(
    graph: NNGraph,
    *,
    validation_rtol: Optional[float],
    validation_atol: Optional[float],
    validation_sample_count: Optional[int],
) -> dict[str, Any]:
    declared = _declared_validation_policy(graph)
    rtol, rtol_source = _resolve_validation_float(
        "rtol",
        explicit=validation_rtol,
        declared=declared.get("rtol"),
        default=DEFAULT_NN_VALIDATION_RTOL,
        declared_source=str(declared.get("rtol_source") or "model_default"),
    )
    atol, atol_source = _resolve_validation_float(
        "atol",
        explicit=validation_atol,
        declared=declared.get("atol"),
        default=DEFAULT_NN_VALIDATION_ATOL,
        declared_source=str(declared.get("atol_source") or "model_default"),
    )
    sample_count, sample_source = _resolve_validation_sample_count(
        explicit=validation_sample_count,
        declared=declared.get("sample_count"),
        declared_source=str(declared.get("sample_count_source") or "model_default"),
    )
    return {
        "schema": "pan_nn_validation_policy_v1",
        "rtol": rtol,
        "atol": atol,
        "sample_count": sample_count,
        "sources": {
            "rtol": rtol_source,
            "atol": atol_source,
            "sample_count": sample_source,
        },
        "declared": declared.get("declared", []),
    }


def _declared_validation_policy(graph: NNGraph) -> dict[str, Any]:
    result: dict[str, Any] = {"declared": []}
    for source, payload in _validation_policy_payloads(graph):
        rtol = _optional_validation_float("rtol", payload.get("rtol"), source=source)
        atol = _optional_validation_float("atol", payload.get("atol"), source=source)
        sample_count = _optional_validation_sample_count(payload.get("sample_count"), source=source)
        declaration = {"source": source}
        if rtol is not None:
            declaration["rtol"] = rtol
            if result.get("rtol") is None or float(rtol) > float(result["rtol"]):
                result["rtol"] = rtol
                result["rtol_source"] = source
        if atol is not None:
            declaration["atol"] = atol
            if result.get("atol") is None or float(atol) > float(result["atol"]):
                result["atol"] = atol
                result["atol_source"] = source
        if sample_count is not None:
            declaration["sample_count"] = sample_count
            if result.get("sample_count") is None:
                result["sample_count"] = sample_count
                result["sample_count_source"] = source
        if len(declaration) > 1:
            result["declared"].append(declaration)
    return result


def _validation_policy_payloads(graph: NNGraph) -> list[tuple[str, dict[str, Any]]]:
    payloads: list[tuple[str, dict[str, Any]]] = []
    metadata_payload = _extract_validation_payload(graph.metadata)
    if metadata_payload:
        payloads.append(("graph_metadata", metadata_payload))
    for node in graph.nodes:
        payload = _extract_validation_payload(node.attrs)
        if payload:
            payloads.append((f"node:{node.node_id}:{node.op}", payload))
    return payloads


def _extract_validation_payload(source: Any) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    for key in ("validation", "validation_policy", "validation_tolerance", "tolerance"):
        payload = source.get(key)
        if isinstance(payload, dict):
            nested = dict(payload)
            tolerance = nested.get("tolerance")
            if isinstance(tolerance, dict):
                nested.update(tolerance)
            return nested
    payload = {}
    for key in ("rtol", "atol", "sample_count"):
        if key in source:
            payload[key] = source[key]
    return payload


def _resolve_validation_float(
    name: str,
    *,
    explicit: Optional[float],
    declared: Any,
    default: float,
    declared_source: str,
) -> tuple[float, str]:
    if explicit is not None:
        return _required_validation_float(name, explicit, source="explicit"), "explicit"
    declared_value = _optional_validation_float(name, declared, source=declared_source)
    if declared_value is not None:
        return declared_value, declared_source
    return float(default), "runtime_default"


def _optional_validation_float(name: str, value: Any, *, source: str) -> Optional[float]:
    if value is None:
        return None
    return _required_validation_float(name, value, source=source)


def _required_validation_float(name: str, value: Any, *, source: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_validation_{name}:{source}") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"invalid_validation_{name}:{source}")
    return parsed


def _resolve_validation_sample_count(
    *,
    explicit: Optional[int],
    declared: Any,
    declared_source: str,
) -> tuple[Optional[int], str]:
    if explicit is not None:
        return _required_validation_sample_count(explicit, source="explicit"), "explicit"
    declared_value = _optional_validation_sample_count(declared, source=declared_source)
    if declared_value is not None:
        return declared_value, declared_source
    return None, "runtime_default"


def _optional_validation_sample_count(value: Any, *, source: str) -> Optional[int]:
    if value is None:
        return None
    return _required_validation_sample_count(value, source=source)


def _required_validation_sample_count(value: Any, *, source: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid_validation_sample_count:{source}") from exc
    if parsed < 0:
        raise ValueError(f"invalid_validation_sample_count:{source}")
    return parsed


def _validate_outputs(
    values: dict[str, NNTensorValue],
    outputs: Sequence[str],
    expected: dict[str, Any],
    *,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-5,
    sample_count: Optional[int] = None,
    seed: int = 1,
) -> dict[str, Any]:
    np = _numpy()
    requested_sample_count = None if sample_count is None else max(0, int(sample_count))
    payload: dict[str, Any] = {
        "oracle": "numpy_ps",
        "ok": True,
        "mode": "full" if requested_sample_count in {None, 0} else "sampled",
        "rtol": float(rtol),
        "atol": float(atol),
        "sample_count": requested_sample_count,
        "outputs": {},
    }
    for name in outputs:
        value = values[name]
        want = np.asarray(expected[name])
        actual_digest = value.ps_digest
        if value.array is not None:
            actual = np.asarray(value.array)
            if actual.shape != want.shape:
                ok = False
                mismatch_count = max(int(actual.size), int(want.size))
                checked_elements = 0
                sampled_indices = None
            else:
                actual_checked, expected_checked, sampled_indices = _validation_slices(
                    actual,
                    want,
                    sample_count=requested_sample_count,
                    seed=seed,
                )
                close = np.isclose(actual_checked, expected_checked, rtol=float(rtol), atol=float(atol))
                ok = bool(np.all(close))
                mismatch_count = int(np.size(close) - int(np.count_nonzero(close)))
                checked_elements = int(np.size(actual_checked))
            actual_digest = tensor_digest(actual)
        elif actual_digest is not None:
            expected_digest = tensor_digest(want)
            ok = actual_digest.get("sha256") == expected_digest.get("sha256")
            mismatch_count = 0 if ok else 1
            sampled_indices = None
        else:
            ok = False
            actual_digest = {"error": "output_not_materialized_for_validation"}
            mismatch_count = 1
            sampled_indices = None
        payload["outputs"][name] = {
            "ok": ok,
            "actual": actual_digest,
            "expected": tensor_digest(want),
            "rtol": float(rtol),
            "atol": float(atol),
            "checked_elements": checked_elements if value.array is not None else None,
            "mismatch_count": mismatch_count,
            "sample_indices": sampled_indices,
        }
        payload["ok"] = bool(payload["ok"] and ok)
    return payload


def _validation_slices(
    actual: Any,
    expected: Any,
    *,
    sample_count: Optional[int],
    seed: int,
) -> tuple[Any, Any, Optional[list[int]]]:
    np = _numpy()
    actual_arr = np.asarray(actual)
    expected_arr = np.asarray(expected)
    if actual_arr.shape != expected_arr.shape:
        return actual_arr.reshape(-1), expected_arr.reshape(-1), None
    flat_actual = actual_arr.reshape(-1)
    flat_expected = expected_arr.reshape(-1)
    if sample_count is None or int(sample_count) <= 0 or int(sample_count) >= int(flat_actual.size):
        return flat_actual, flat_expected, None
    rng = np.random.default_rng(int(seed))
    indices = sorted(int(index) for index in rng.choice(int(flat_actual.size), size=int(sample_count), replace=False))
    return flat_actual[indices], flat_expected[indices], indices


def _output_summary(value: NNTensorValue, *, include_digest: bool, materialization_reason: Optional[str]) -> dict[str, Any]:
    if include_digest and not materialization_reason:
        raise RuntimeError("materialization_requires_reason")
    payload = {
        "value": value.to_json(),
        "materialized": bool(include_digest and (value.array is not None or value.ps_digest is not None)),
        "residency": value.residency,
        "materialization_reason": materialization_reason,
    }
    if include_digest and value.array is not None:
        payload["digest"] = tensor_digest(value.array)
    elif include_digest and value.ps_digest is not None:
        payload["digest"] = dict(value.ps_digest)
    return payload


def _region_elements(region: NNRegion, values: dict[str, NNTensorValue]) -> int:
    for output in region.outputs:
        if output in values:
            return values[output].elements
    for node in region.nodes:
        for name in node.inputs:
            if name in values:
                return values[name].elements
    return 0


def _spec_array(spec: NNTensorSpec) -> Any:
    if spec.data is None:
        raise ValueError(f"initializer {spec.name!r} has no data")
    arr = _array_from_payload(spec.data, dtype=spec.dtype)
    shape = _concrete_shape(spec.shape)
    if shape:
        arr = arr.reshape(shape)
    return arr


def _array_from_payload(payload: Any, *, dtype: str) -> Any:
    np = _numpy()
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    return np.asarray(payload, dtype=_numpy_dtype(dtype))


def _generated_input(spec: Optional[NNTensorSpec], rng: Any) -> Any:
    np = _numpy()
    if spec is None:
        return rng.standard_normal((1, 1), dtype=np.float32)
    dtype = _numpy_dtype(spec.dtype)
    shape = _concrete_shape(spec.shape)
    if not shape:
        shape = (1,)
    if np.dtype(dtype).kind in {"u", "i"}:
        return rng.integers(0, 16, size=shape, dtype=dtype)
    return rng.standard_normal(shape).astype(dtype)


def _concrete_shape(shape: Sequence[int | str]) -> tuple[int, ...]:
    concrete = []
    for dim in shape:
        if isinstance(dim, str):
            if not dim.isdigit():
                raise ValueError(f"dynamic NN shape dimension {dim!r} requires explicit input data")
            concrete.append(int(dim))
        else:
            concrete.append(int(dim))
    return tuple(concrete)


def _produced_names(nodes: Sequence[NNNode]) -> set[str]:
    return {name for node in nodes for name in node.outputs}


def _numpy_dtype(dtype: str) -> Any:
    np = _numpy()
    normalized = _canonical_nn_dtype(dtype)
    return np.dtype(normalized)


def _canonical_nn_dtype(dtype: str) -> str:
    normalized = NN_DTYPE_ALIASES.get(str(dtype).lower(), str(dtype).lower())
    if normalized not in SUPPORTED_NN_DTYPES:
        raise ValueError(f"unsupported_nn_dtype:{dtype}")
    return normalized


def _cast_target_dtype(attrs: dict[str, Any]) -> Optional[str]:
    target = attrs.get("dtype", attrs.get("to"))
    if target is None:
        return None
    if isinstance(target, str):
        return target
    try:
        return _onnx_elem_type_dtype(int(target))
    except (TypeError, ValueError):
        return None


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for PAN NN runtime") from exc
    return np


def _load_onnx_model(source: Path) -> NNGraph:
    try:
        import onnx  # type: ignore
        from onnx import numpy_helper  # type: ignore
    except ImportError as exc:
        raise RuntimeError("onnx is required to load .onnx models; use pan_nn_graph_v1 JSON in this environment") from exc
    model = onnx.load(str(source))
    graph = model.graph
    tensors: dict[str, NNTensorSpec] = {}
    initializer_names = {init.name for init in graph.initializer}
    input_names: list[str] = []
    for value in graph.input:
        if value.name in initializer_names:
            continue
        input_names.append(value.name)
        tensors[value.name] = NNTensorSpec(
            name=value.name,
            dtype=_onnx_value_dtype(value),
            shape=_onnx_value_shape(value),
            initializer=False,
        )
    for init in graph.initializer:
        array = numpy_helper.to_array(init)
        tensors[init.name] = NNTensorSpec(
            name=init.name,
            dtype=str(array.dtype),
            shape=tuple(int(dim) for dim in array.shape),
            initializer=True,
            data=array.tolist(),
        )
    for value in [*getattr(graph, "value_info", ()), *graph.output]:
        if value.name in tensors:
            continue
        tensors[value.name] = NNTensorSpec(
            name=value.name,
            dtype=_onnx_value_dtype(value),
            shape=_onnx_value_shape(value),
            initializer=False,
        )
    nodes_list: list[NNNode] = []
    for index, node in enumerate(graph.node):
        node_id = node.name or f"node{index}"
        raw_node_inputs = tuple(str(name) for name in node.input)
        attrs = _onnx_attrs(node)
        attrs = _onnx_postprocess_attrs(str(node.op_type), attrs, numpy_helper)
        constant_spec = _onnx_constant_tensor_spec(node, attrs, numpy_helper)
        if constant_spec is not None:
            tensors[constant_spec.name] = constant_spec
            continue
        node_inputs = _onnx_node_inputs(str(node.op_type), raw_node_inputs, node_id=node_id, tensors=tensors)
        nodes_list.append(
            NNNode(
                node_id=node_id,
                op=_normalize_onnx_op(str(node.op_type), attrs, node_inputs, tensors),
                inputs=node_inputs,
                outputs=tuple(str(name) for name in node.output if name),
                attrs=attrs,
            )
        )
    nodes = tuple(nodes_list)
    outputs = tuple(value.name for value in graph.output)
    return NNGraph(
        name=graph.name or source.name,
        inputs=tuple(input_names),
        outputs=outputs,
        tensors=tensors,
        nodes=nodes,
        metadata={"source": str(source), "format": "onnx"},
    )


def _onnx_node_inputs(
    op_type: str,
    raw_inputs: Sequence[str],
    *,
    node_id: str,
    tensors: dict[str, NNTensorSpec],
) -> tuple[str, ...]:
    if str(op_type).lower() != "matmulinteger":
        return tuple(str(name) for name in raw_inputs if name)
    names = tuple(str(name) for name in raw_inputs)
    if len(names) >= 4 and not names[2] and names[3]:
        a_name = names[0] if names else ""
        a_dtype = tensors[a_name].dtype if a_name in tensors else "uint8"
        zero_name = f"{node_id}__a_zero_point_default"
        suffix = 1
        while zero_name in tensors:
            suffix += 1
            zero_name = f"{node_id}__a_zero_point_default_{suffix}"
        tensors[zero_name] = NNTensorSpec(
            name=zero_name,
            dtype=a_dtype,
            shape=(),
            initializer=True,
            data=0,
        )
        return (names[0], names[1], zero_name, names[3])
    return tuple(name for name in names if name)


def _onnx_value_dtype(value: Any) -> str:
    elem_type = int(value.type.tensor_type.elem_type)
    return _onnx_elem_type_dtype(elem_type)


def _onnx_elem_type_dtype(elem_type: int) -> str:
    mapping = {
        1: "float32",
        10: "float16",
        2: "uint8",
        3: "int8",
        5: "int16",
        6: "int32",
        7: "int64",
        9: "bool",
        12: "uint32",
        13: "uint64",
        16: "bfloat16",
    }
    return mapping.get(elem_type, f"onnx_elem_type_{elem_type}")


def _onnx_value_shape(value: Any) -> tuple[int | str, ...]:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_value:
            dims.append(int(dim.dim_value))
        elif dim.dim_param:
            dims.append(str(dim.dim_param))
        else:
            dims.append("dynamic")
    return tuple(dims)


def _onnx_attrs(node: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    try:
        from onnx import helper  # type: ignore
    except ImportError:
        return attrs
    for attr in node.attribute:
        attrs[attr.name] = helper.get_attribute_value(attr)
        if isinstance(attrs[attr.name], bytes):
            attrs[attr.name] = attrs[attr.name].decode("utf-8", errors="replace")
    return attrs


def _onnx_postprocess_attrs(op_type: str, attrs: dict[str, Any], numpy_helper: Any) -> dict[str, Any]:
    if str(op_type).lower() != "constantofshape" or "value" not in attrs:
        return attrs
    updated = dict(attrs)
    value = updated["value"]
    try:
        array = numpy_helper.to_array(value)
    except Exception:
        array = value
    np = _numpy()
    arr = np.asarray(array).reshape(-1)
    if int(arr.size) == 1:
        updated["value"] = arr.tolist()[0]
        updated["dtype"] = str(arr.dtype)
    else:
        updated["value"] = arr.tolist()
        updated["dtype"] = str(arr.dtype)
    return updated


def _onnx_constant_tensor_spec(node: Any, attrs: dict[str, Any], numpy_helper: Any) -> Optional[NNTensorSpec]:
    if str(getattr(node, "op_type", "")).lower() != "constant":
        return None
    outputs = [str(name) for name in getattr(node, "output", ()) if name]
    if len(outputs) != 1:
        return None
    array = _onnx_constant_array(attrs, numpy_helper)
    if array is None:
        return None
    np = _numpy()
    arr = np.asarray(array)
    return NNTensorSpec(
        name=outputs[0],
        dtype=str(arr.dtype),
        shape=tuple(int(dim) for dim in arr.shape),
        initializer=True,
        data=arr.tolist(),
    )


def _onnx_constant_array(attrs: dict[str, Any], numpy_helper: Any) -> Any:
    if "value" in attrs:
        value = attrs["value"]
        try:
            return numpy_helper.to_array(value)
        except Exception:
            return value
    if "value_float" in attrs:
        return _numpy().asarray(attrs["value_float"], dtype=_numpy().float32)
    if "value_floats" in attrs:
        return _numpy().asarray(list(attrs["value_floats"]), dtype=_numpy().float32)
    if "value_int" in attrs:
        return _numpy().asarray(attrs["value_int"], dtype=_numpy().int64)
    if "value_ints" in attrs:
        return _numpy().asarray(list(attrs["value_ints"]), dtype=_numpy().int64)
    return None


def _normalize_onnx_op(
    op_type: str,
    attrs: dict[str, Any],
    inputs: Sequence[str],
    tensors: dict[str, NNTensorSpec],
) -> str:
    op = str(op_type).lower()
    aliases = {
        "matmul": "matmul",
        "einsum": "einsum",
        "gemm": "gemm",
        "identity": "identity",
        "dropout": "dropout",
        "sum": "sum",
        "mean": "mean",
        "add": "add",
        "sub": "sub",
        "mul": "mul",
        "div": "div",
        "neg": "neg",
        "abs": "abs",
        "pow": "pow",
        "mod": "mod",
        "min": "min",
        "max": "max",
        "clip": "clip",
        "floor": "floor",
        "ceil": "ceil",
        "exp": "exp",
        "log": "log",
        "reciprocal": "reciprocal",
        "sqrt": "sqrt",
        "reducemean": "reduce_mean",
        "reducesum": "reduce_sum",
        "reduce_sum": "reduce_sum",
        "reducemin": "reduce_min",
        "reduce_min": "reduce_min",
        "reducemax": "reduce_max",
        "reduce_max": "reduce_max",
        "reduceprod": "reduce_prod",
        "reduce_prod": "reduce_prod",
        "concat": "concat",
        "relu": "relu",
        "gelu": "gelu",
        "biasgelu": "bias_gelu",
        "bias_gelu": "bias_gelu",
        "quickgelu": "quick_gelu",
        "quick_gelu": "quick_gelu",
        "silu": "silu",
        "swish": "silu",
        "sigmoid": "sigmoid",
        "tanh": "tanh",
        "erf": "erf",
        "sin": "sin",
        "cos": "cos",
        "softmax": "softmax",
        "logsoftmax": "log_softmax",
        "log_softmax": "log_softmax",
        "cast": "cast",
        "castlike": "cast_like",
        "cast_like": "cast_like",
        "size": "size",
        "shape": "shape",
        "range": "range",
        "constantofshape": "constant_of_shape",
        "constant_of_shape": "constant_of_shape",
        "cumsum": "cumsum",
        "cum_sum": "cumsum",
        "pad": "pad",
        "flatten": "flatten",
        "unsqueeze": "unsqueeze",
        "squeeze": "squeeze",
        "expand": "expand",
        "tile": "tile",
        "trilu": "trilu",
        "split": "split",
        "gather": "gather",
        "gatherelements": "gather_elements",
        "gather_elements": "gather_elements",
        "gathernd": "gather_nd",
        "gather_nd": "gather_nd",
        "scatterelements": "scatter_elements",
        "scatter_elements": "scatter_elements",
        "scatternd": "scatter_nd",
        "scatter_nd": "scatter_nd",
        "slice": "slice",
        "where": "where",
        "equal": "equal",
        "less": "less",
        "lessorequal": "less_equal",
        "lessequal": "less_equal",
        "less_equal": "less_equal",
        "greater": "greater",
        "greaterorequal": "greater_equal",
        "greaterequal": "greater_equal",
        "greater_equal": "greater_equal",
        "not": "not",
        "and": "and",
        "or": "or",
        "reshape": "reshape",
        "transpose": "transpose",
        "layernormalization": "layernorm",
        "layer_norm": "layernorm",
        "layernorm": "layernorm",
        "rmsnormalization": "rmsnorm",
        "rms_norm": "rmsnorm",
        "rmsnorm": "rmsnorm",
        "simplifiedlayernormalization": "rmsnorm",
        "rotaryembedding": "rotary_embedding",
        "rotary_embedding": "rotary_embedding",
        "rope": "rotary_embedding",
        "batchnormalization": "batchnorm",
        "batch_norm": "batchnorm",
        "batchnorm": "batchnorm",
        "argmax": "argmax",
        "argmin": "argmin",
        "topk": "topk",
        "top_k": "topk",
        "top_p_filter": "top_p_filter",
        "toppfilter": "top_p_filter",
        "kv_cache_append": "kv_cache_append",
        "kvcacheappend": "kv_cache_append",
        "quantizelinear": "quantize_linear",
        "dequantizelinear": "dequantize_linear",
        "qlinearmatmul": "qlinear_matmul",
        "qlinear_matmul": "qlinear_matmul",
        "matmulinteger": "matmul_integer",
        "matmul_integer": "matmul_integer",
        "dynamicquantizelinear": "dynamic_quantize_linear",
        "dynamic_quantize_linear": "dynamic_quantize_linear",
    }
    if op != "conv":
        return aliases.get(op, op)
    group = _safe_int(attrs.get("group", 1), default=1)
    if group <= 1:
        attrs["group"] = 1
        return "conv2d"
    attrs["group"] = group
    input_shape = tensors.get(inputs[0]).shape if len(inputs) >= 1 and inputs[0] in tensors else ()
    weight_shape = tensors.get(inputs[1]).shape if len(inputs) >= 2 and inputs[1] in tensors else ()
    if _onnx_conv_is_depthwise(input_shape, weight_shape, group):
        return "depthwise_conv2d"
    return "conv2d"


def _onnx_conv_is_depthwise(
    input_shape: Sequence[int | str],
    weight_shape: Sequence[int | str],
    group: int,
) -> bool:
    if len(weight_shape) != 4 or group <= 1:
        return False
    weight_channels = weight_shape[1]
    if isinstance(weight_channels, str) or int(weight_channels) != 1:
        return False
    if len(input_shape) == 4 and not isinstance(input_shape[1], str):
        return int(input_shape[1]) == int(group)
    return True
