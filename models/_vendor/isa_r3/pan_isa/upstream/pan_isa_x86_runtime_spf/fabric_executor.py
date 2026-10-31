"""Resident graph executor for PAN Fabric Runtime.

The planner decides where typed SPF graph nodes should run. This module makes
that plan executable while keeping explicit buffer residency state, so training
data captures the real cost of crossing PS/PL boundaries.
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from .fabric_daemon import FabricDaemonGraphSession, _is_daemon_bo_descriptor, execute_fabric_graph_daemon
from .fabric_graph import FabricGraph, FabricGraphPlan, FabricGraphPlanStep, FabricPolicyHead
from .fabric_runtime import (
    FabricKernelSpec,
    KV260FabricSession,
    _find_libspf32,
    _numpy,
    _scalar_n_arg,
    _xrt_export_bo,
    _xrt_import_bo,
    _xrt_read_array,
    _xrt_sync_bo,
    _xrt_write_array,
    fabric_xrt_report,
    load_xmutil_app,
)


FABRIC_EXECUTION_SCHEMA = "pan_fabric_execution_v1"
FABRIC_TRAINING_SCHEMA = "pan_fabric_training_record_v1"
FABRIC_VALUE_SCHEMA = "pan_fabric_value_v1"
FABRIC_VALUE_STORE_SCHEMA = "pan_fabric_value_store_v1"
FABRIC_VALUE_RESIDENCIES = frozenset({"ps", "daemon", "xrt_bo", "dma_buf", "catalog_core", "unknown"})
SUPPORTED_DAEMON_GRAPH_OPS = frozenset({"spf32.mul", "spf32.add", "spf32.div", "spf32.abs2", "spf32.fft64", "mlp.gemm_k16"})


@dataclass
class FabricBufferState:
    name: str
    elements: int
    ps_array: Any = None
    pl_array: Any = None
    pl_bo: Any = None
    pl_group_id: Optional[int] = None
    pl_session_key: Optional[str] = None
    pl_bo_nbytes: Optional[int] = None
    pl_bo_export: Optional[dict[str, Any]] = None

    def residency(self) -> tuple[str, ...]:
        values = []
        if self.ps_array is not None:
            values.append("ps")
        if self.pl_array is not None or self.pl_bo is not None or (self.pl_bo_export or {}).get("ok"):
            values.append("pl")
        return tuple(values)

    def bo_descriptor(self) -> Optional[dict[str, Any]]:
        if self.pl_bo is None and self.pl_bo_export is None:
            return None
        backend = "daemon" if self.pl_session_key == "daemon" else "xrt"
        return {
            "name": self.name,
            "backend": backend,
            "elements": int(self.elements),
            "session_key": self.pl_session_key,
            "group_id": self.pl_group_id,
            "nbytes": self.pl_bo_nbytes,
            "resident": self.pl_bo is not None or backend == "daemon",
            "export": self.pl_bo_export,
        }


@dataclass(frozen=True)
class FabricValue:
    """Runtime-visible value that may be materialized on PS or resident in pan-fabricd."""

    value_id: str
    name: str
    dtype: str
    elements: int
    residency: str
    materialized: bool
    ps_array: Any = None
    daemon_handle: Optional[dict[str, Any]] = None
    producer: Optional[str] = None
    validation: Optional[dict[str, Any]] = None
    refcount: int = 1
    materialization_reason: Optional[str] = None
    ps_digest: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_buffer_state(
        cls,
        state: FabricBufferState,
        *,
        value_id: Optional[str] = None,
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
    ) -> "FabricValue":
        descriptor = state.bo_descriptor()
        daemon_handle = (
            dict(descriptor["export"])
            if descriptor and descriptor.get("backend") == "daemon" and descriptor.get("export")
            else None
        )
        if daemon_handle is not None:
            residency = "daemon"
        elif state.pl_bo is not None:
            residency = "xrt_bo"
        elif state.ps_array is not None:
            residency = "ps"
        else:
            residency = "unknown"
        return cls(
            value_id=value_id or f"{producer or 'buffer'}:{state.name}",
            name=state.name,
            dtype="uint32",
            elements=int(state.elements),
            residency=residency,
            materialized=state.ps_array is not None,
            ps_array=state.ps_array,
            daemon_handle=daemon_handle,
            producer=producer,
            validation=validation,
            ps_digest=_array_digest(state.ps_array) if state.ps_array is not None else None,
        )

    def is_daemon_resident(self) -> bool:
        return self.residency == "daemon" and self.daemon_handle is not None

    def to_json(self) -> dict[str, Any]:
        payload = {
            "schema": FABRIC_VALUE_SCHEMA,
            "value_id": self.value_id,
            "name": self.name,
            "dtype": self.dtype,
            "elements": int(self.elements),
            "residency": self.residency,
            "materialized": bool(self.materialized),
            "daemon_handle": self.daemon_handle,
            "producer": self.producer,
            "validation": self.validation,
            "refcount": int(self.refcount),
            "materialization_reason": self.materialization_reason,
            "metadata": dict(self.metadata),
        }
        digest = self.ps_digest
        if digest is None and self.ps_array is not None:
            digest = _array_digest(self.ps_array)
        if digest is not None:
            payload["ps_digest"] = digest
        return payload


class FabricValueStore:
    """Session-local value table for PAN fabric runtime values."""

    def __init__(self, *, session_id: Optional[str] = None):
        self.session_id = session_id or f"fabric-session-{time.time_ns()}"
        self._values: dict[str, FabricValue] = {}
        self._latest_by_name: dict[str, str] = {}
        self._counter = 0
        self.materialization_count = 0
        self.release_count = 0

    def create_ps(
        self,
        name: str,
        array: Any,
        *,
        dtype: str = "uint32",
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
        materialization_reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FabricValue:
        np = _numpy()
        arr = np.ascontiguousarray(array, dtype=np.uint32).reshape(-1)
        if materialization_reason:
            self.materialization_count += 1
        return self.add(
            FabricValue(
                value_id=self._next_id(name),
                name=str(name),
                dtype=dtype,
                elements=int(arr.size),
                residency="ps",
                materialized=True,
                ps_array=arr,
                daemon_handle=None,
                producer=producer,
                validation=validation,
                materialization_reason=materialization_reason,
                ps_digest=_array_digest(arr),
                metadata=dict(metadata or {}),
            )
        )

    def create_daemon(
        self,
        name: str,
        handle: dict[str, Any],
        *,
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FabricValue:
        if not _is_daemon_bo_descriptor(handle):
            raise ValueError("daemon FabricValue requires a pan_daemon_bo_v1 descriptor")
        return self.add(
            FabricValue(
                value_id=self._next_id(name),
                name=str(name),
                dtype=str(handle.get("dtype") or "uint32"),
                elements=int(handle.get("elements") or 0),
                residency="daemon",
                materialized=False,
                daemon_handle=dict(handle),
                producer=producer,
                validation=validation,
                metadata=dict(metadata or {}),
            )
        )

    def create_metadata(
        self,
        name: str,
        *,
        elements: int,
        dtype: str = "runtime_region",
        residency: str = "ps",
        materialized: bool = False,
        producer: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
        materialization_reason: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> FabricValue:
        if materialization_reason:
            self.materialization_count += 1
        return self.add(
            FabricValue(
                value_id=self._next_id(name),
                name=str(name),
                dtype=dtype,
                elements=int(elements),
                residency=residency,
                materialized=bool(materialized),
                producer=producer,
                validation=validation,
                materialization_reason=materialization_reason,
                metadata=dict(metadata or {}),
            )
        )

    def add(self, value: FabricValue) -> FabricValue:
        if value.residency not in FABRIC_VALUE_RESIDENCIES:
            raise ValueError(f"unsupported FabricValue residency {value.residency!r}")
        self._values[value.value_id] = value
        self._latest_by_name[value.name] = value.value_id
        return value

    def get(self, value_id: str) -> FabricValue:
        try:
            return self._values[str(value_id)]
        except KeyError as exc:
            raise KeyError(f"unknown FabricValue id {value_id!r}") from exc

    def latest(self, name: str) -> FabricValue:
        try:
            return self.get(self._latest_by_name[str(name)])
        except KeyError as exc:
            raise KeyError(f"no FabricValue exists for buffer {name!r}") from exc

    def retain(self, value_id: str) -> FabricValue:
        value = self.get(value_id)
        retained = FabricValue(
            **{
                **value.__dict__,
                "refcount": int(value.refcount) + 1,
            }
        )
        self._values[value_id] = retained
        return retained

    def release(self, value_id: str) -> Optional[FabricValue]:
        value = self.get(value_id)
        self.release_count += 1
        if value.refcount <= 1:
            del self._values[value_id]
            if self._latest_by_name.get(value.name) == value_id:
                self._latest_by_name.pop(value.name, None)
            return None
        released = FabricValue(
            **{
                **value.__dict__,
                "refcount": int(value.refcount) - 1,
            }
        )
        self._values[value_id] = released
        return released

    def resolve_input(self, source: Any) -> Any:
        if isinstance(source, FabricValue):
            return self._runtime_value(source)
        if isinstance(source, str) and source in self._values:
            return self._runtime_value(self.get(source))
        if isinstance(source, dict) and "value_id" in source:
            return self._runtime_value(self.get(str(source["value_id"])))
        return source

    def value_id_for_source(self, source: Any) -> Optional[str]:
        if isinstance(source, FabricValue):
            return source.value_id
        if isinstance(source, str) and source in self._values:
            return source
        if isinstance(source, dict) and "value_id" in source:
            return str(source["value_id"])
        return None

    def materialize(
        self,
        value_id: str,
        array: Any,
        *,
        reason: str,
        validation: Optional[dict[str, Any]] = None,
    ) -> FabricValue:
        prior = self.get(value_id)
        self.materialization_count += 1
        np = _numpy()
        arr = np.ascontiguousarray(array, dtype=np.uint32).reshape(-1)
        materialized = FabricValue(
            value_id=value_id,
            name=prior.name,
            dtype=prior.dtype,
            elements=int(arr.size),
            residency="ps",
            materialized=True,
            ps_array=arr,
            daemon_handle=prior.daemon_handle,
            producer=prior.producer,
            validation=validation if validation is not None else prior.validation,
            refcount=prior.refcount,
            materialization_reason=reason,
            ps_digest=_array_digest(arr),
            metadata=dict(prior.metadata),
        )
        self._values[value_id] = materialized
        self._latest_by_name[prior.name] = value_id
        return materialized

    def update_validation(self, value_id: str, validation: dict[str, Any]) -> FabricValue:
        value = self.get(value_id)
        updated = FabricValue(
            **{
                **value.__dict__,
                "validation": dict(validation),
            }
        )
        self._values[value_id] = updated
        self._latest_by_name[value.name] = value_id
        return updated

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": FABRIC_VALUE_STORE_SCHEMA,
            "session_id": self.session_id,
            "value_count": len(self._values),
            "latest_by_name": dict(sorted(self._latest_by_name.items())),
            "materialization_count": int(self.materialization_count),
            "release_count": int(self.release_count),
            "values": {
                value_id: value.to_json()
                for value_id, value in sorted(self._values.items())
            },
        }

    def _runtime_value(self, value: FabricValue) -> Any:
        if value.daemon_handle is not None and value.residency == "daemon":
            return value.daemon_handle
        if value.ps_array is not None:
            return value.ps_array
        raise ValueError(f"FabricValue {value.value_id!r} is not usable as a runtime input")

    def _next_id(self, name: str) -> str:
        self._counter += 1
        return f"{self.session_id}:{name}:{self._counter}"


class FabricRuntimeSession:
    """Long-lived execution context for resident fabric graph packets."""

    def __init__(self, executor: "FabricGraphExecutor", *, session_id: Optional[str] = None):
        self.executor = executor
        self.value_store = FabricValueStore(session_id=session_id)
        self.daemon: Optional[FabricDaemonGraphSession] = None
        self.trace: list[dict[str, Any]] = []
        self.training_records: list[dict[str, Any]] = []

    def __enter__(self) -> "FabricRuntimeSession":
        self.daemon = FabricDaemonGraphSession(self.executor.daemon_socket)
        self.daemon.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.daemon is not None:
            self.daemon.close()
            self.daemon = None

    def execute_graph_packet(
        self,
        inputs: dict[str, Any],
        *,
        output_names: Optional[Sequence[str]] = None,
        repeat_index: int,
        step_index: int,
        phase: str,
        materialize_outputs: bool,
        note: str,
        materialization_reason: Optional[str] = None,
        validation: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], Any, dict[str, FabricValue]]:
        if self.daemon is None:
            raise RuntimeError("FabricRuntimeSession is not connected")
        resolved_inputs = {
            name: self.value_store.resolve_input(value)
            for name, value in inputs.items()
        }
        input_value_ids = {
            name: self.value_store.value_id_for_source(value)
            for name, value in inputs.items()
            if self.value_store.value_id_for_source(value) is not None
        }
        result = self.daemon.execute_graph(
            self.executor.plan,
            resolved_inputs,
            output_names=output_names or self.executor.graph.outputs,
            buffer_storage=self.executor.daemon_buffer_storage,
            materialize_outputs=materialize_outputs,
        )
        if not result.ok:
            raise RuntimeError(f"fabric daemon session packet failed: {result.error or result.status}")
        output_values: dict[str, FabricValue] = {}
        for name, value in result.outputs.items():
            if _is_daemon_bo_descriptor(value):
                output_values[name] = self.value_store.create_daemon(
                    name,
                    value,
                    producer=note,
                    validation=validation,
                    metadata={"phase": phase},
                )
            else:
                output_values[name] = self.value_store.create_ps(
                    name,
                    value,
                    producer=note,
                    validation=validation,
                    materialization_reason=materialization_reason,
                    metadata={"phase": phase},
                )
        row = self._trace_row(
            result,
            repeat_index=repeat_index,
            step_index=step_index,
            phase=phase,
            note=note,
            input_value_ids=input_value_ids,
            output_value_ids={name: value.value_id for name, value in output_values.items()},
            materialization_reason=materialization_reason,
        )
        self.trace.append(row)
        if phase == "timed":
            self.training_records.append(_training_record(row, self.executor.plan, repeat_index=repeat_index))
        return row, result, output_values

    def import_ps_value(
        self,
        name: str,
        array: Any,
        *,
        producer: str = "initial_input",
        phase: str = "setup",
    ) -> FabricValue:
        if self.daemon is None:
            raise RuntimeError("FabricRuntimeSession is not connected")
        result = self.daemon.import_u32_value(array)
        if not result.ok:
            raise RuntimeError(f"fabric daemon value import failed: {result.error or result.status}")
        descriptor = _daemon_bo_descriptor_from_value_result(result, name=name)
        value = self.value_store.create_daemon(
            name,
            descriptor,
            producer=producer,
            metadata={"phase": phase, "import_roundtrip_seconds": result.roundtrip_seconds},
        )
        self.trace.append(
            self._value_trace_row(
                result,
                phase=phase,
                action="import_value",
                note=producer,
                output_value_id=value.value_id,
                materialization_reason=None,
            )
        )
        return value

    def materialize_value(
        self,
        value: FabricValue,
        *,
        reason: str,
        validation: Optional[dict[str, Any]] = None,
    ) -> FabricValue:
        if self.daemon is None:
            raise RuntimeError("FabricRuntimeSession is not connected")
        if not value.daemon_handle:
            if value.ps_array is None:
                raise ValueError(f"FabricValue {value.value_id!r} cannot be materialized")
            return value
        result = self.daemon.materialize_value(int(value.daemon_handle["handle"]), elements=int(value.elements))
        if not result.ok:
            raise RuntimeError(f"fabric daemon materialize failed: {result.error or result.status}")
        materialized = self.value_store.materialize(
            value.value_id,
            result.materialized_array(),
            reason=reason,
            validation=validation,
        )
        self.trace.append(
            self._value_trace_row(
                result,
                phase="materialize",
                action="materialize",
                note="materialize_value",
                input_value_id=value.value_id,
                output_value_id=materialized.value_id,
                materialization_reason=reason,
            )
        )
        return materialized

    def release_value(self, value: FabricValue) -> None:
        if value.daemon_handle and self.daemon is not None:
            result = self.daemon.release_value(int(value.daemon_handle["handle"]))
            self.trace.append(
                self._value_trace_row(
                    result,
                    phase="release",
                    action="release_value",
                    note="release_value",
                    input_value_id=value.value_id,
                    materialization_reason=None,
                )
            )
        try:
            self.value_store.release(value.value_id)
        except KeyError:
            pass

    def execute_resident_graph_chain(self, *, repeats: int) -> dict[str, Any]:
        self.executor._validate_fabric_session_plan()
        graph = self.executor.graph
        output_name = graph.outputs[0]
        first_input = graph.ops[0].inputs[0]
        if int(repeats) > 1 and graph.buffers[first_input].elements != graph.buffers[output_name].elements:
            raise ValueError("fabric-session repeated graph chaining requires final output size to match the first input")
        buffers = self.executor._initial_buffers(0)
        input_names = self.executor._graph_input_names()
        input_values: dict[str, FabricValue] = {}
        ps_inputs: dict[str, Any] = {}
        for name in input_names:
            source = self.executor._daemon_input_value(buffers, name)
            if _is_daemon_bo_descriptor(source):
                input_values[name] = self.value_store.create_daemon(name, source, producer="initial_input")
            else:
                ps_inputs[name] = source
                input_values[name] = self.import_ps_value(name, source, producer="initial_input")

        current_value = input_values[first_input]
        prior_timed_values: list[FabricValue] = []
        for repeat_index in range(int(repeats)):
            packet_inputs = {
                name: (current_value.value_id if name == first_input else value.value_id)
                for name, value in input_values.items()
            }
            _, _, outputs = self.execute_graph_packet(
                packet_inputs,
                repeat_index=repeat_index,
                step_index=repeat_index,
                phase="timed",
                materialize_outputs=False,
                note="resident_graph_packet",
            )
            if current_value.value_id not in {input_values[first_input].value_id}:
                prior_timed_values.append(current_value)
            current_value = outputs[output_name]

        validation: Optional[dict[str, Any]] = None
        final_value = current_value
        if self.executor.materialize_outputs:
            materialized = self.materialize_value(
                current_value,
                reason="final_output_digest_or_validation",
            )
            validation = self.executor._validate_fabric_session_output(
                actual=materialized.ps_array,
                initial_inputs=ps_inputs,
                repeats=int(repeats),
            )
            if validation.get("ok") is False:
                raise RuntimeError("fabric-session validation failed against PS oracle")
            final_value = self.value_store.update_validation(materialized.value_id, validation)
        keep_ids = {final_value.value_id}
        for value in list(input_values.values()) + prior_timed_values:
            if value.value_id not in keep_ids:
                self.release_value(value)
        return {
            "buffers": buffers,
            "output_name": output_name,
            "output_value": final_value,
            "validation": validation,
            "initial_inputs": ps_inputs,
        }

    def _trace_row(
        self,
        result: Any,
        *,
        repeat_index: int,
        step_index: int,
        phase: str,
        note: str,
        input_value_ids: dict[str, str],
        output_value_ids: dict[str, str],
        materialization_reason: Optional[str],
    ) -> dict[str, Any]:
        observed_transfer_bytes = result.input_bytes + result.output_bytes
        return {
            "schema": FABRIC_EXECUTION_SCHEMA,
            "repeat_index": repeat_index,
            "step_index": step_index,
            "phase": phase,
            "timed": phase == "timed",
            "action": "run_daemon_graph",
            "backend": "daemon_session_resident",
            "op_id": "fabric_session",
            "op": "fabric.graph",
            "graph_ops": [op.op for op in self.executor.graph.ops],
            "buffer": None,
            "src": None,
            "dst": None,
            "kernel": None,
            "planned_latency_s": self.executor.plan.total_latency_s,
            "planned_energy_j": self.executor.plan.total_energy_j,
            "planned_transfer_bytes": self.executor.plan.transfer_bytes,
            "observed_transfer_bytes": observed_transfer_bytes,
            "elapsed_seconds": result.roundtrip_seconds,
            "daemon_elapsed_seconds": result.daemon_elapsed_seconds,
            "daemon_socket": self.executor.daemon_socket,
            "daemon_buffer_storage": self.executor.daemon_buffer_storage,
            "session_id": self.value_store.session_id,
            "session_note": note,
            "input_value_ids": dict(sorted(input_value_ids.items())),
            "output_value_ids": dict(sorted(output_value_ids.items())),
            "materialization_reason": materialization_reason,
            "placement_action": "run_daemon_resident_now",
            "features": {
                "daemon_graph_ops": float(result.graph_op_count),
                "daemon_input_bytes": float(result.input_bytes),
                "daemon_output_bytes": float(result.output_bytes),
                "daemon_resident_outputs": float(result.resident_output_count),
                "daemon_avoided_host_materializations": float(result.avoided_host_materialization_count),
                "latency_s": self.executor.plan.total_latency_s,
                "energy_j": self.executor.plan.total_energy_j,
                "transfer_bytes": float(observed_transfer_bytes),
                "value_store_size": float(len(self.value_store._values)),
                "unsupported": 0.0,
            },
            "scheduler_hint": self.executor.scheduler_hint or None,
            "daemon_result": result.to_json(),
        }

    def _value_trace_row(
        self,
        result: Any,
        *,
        phase: str,
        action: str,
        note: str,
        input_value_id: Optional[str] = None,
        output_value_id: Optional[str] = None,
        materialization_reason: Optional[str],
    ) -> dict[str, Any]:
        if action == "import_value":
            observed_transfer_bytes = int(getattr(result, "elements", 0) or 0) * 4
        elif action == "materialize":
            observed_transfer_bytes = int(getattr(result, "payload", b"") and len(result.payload) or 0)
        else:
            observed_transfer_bytes = 0
        return {
            "schema": FABRIC_EXECUTION_SCHEMA,
            "repeat_index": -1,
            "step_index": -1,
            "phase": phase,
            "timed": False,
            "action": action,
            "backend": "daemon_session_resident",
            "op_id": "fabric_value",
            "op": "fabric.value",
            "buffer": None,
            "src": None,
            "dst": None,
            "kernel": None,
            "planned_latency_s": 0.0,
            "planned_energy_j": 0.0,
            "planned_transfer_bytes": observed_transfer_bytes,
            "observed_transfer_bytes": observed_transfer_bytes,
            "elapsed_seconds": result.roundtrip_seconds,
            "daemon_elapsed_seconds": 0.0,
            "daemon_socket": self.executor.daemon_socket,
            "daemon_buffer_storage": "value",
            "session_id": self.value_store.session_id,
            "session_note": note,
            "input_value_ids": {"value": input_value_id} if input_value_id else {},
            "output_value_ids": {"value": output_value_id} if output_value_id else {},
            "materialization_reason": materialization_reason,
            "placement_action": "materialize" if action == "materialize" else "keep_value_resident",
            "features": {
                "latency_s": 0.0,
                "energy_j": 0.0,
                "transfer_bytes": float(observed_transfer_bytes),
                "value_store_size": float(len(self.value_store._values)),
                "unsupported": 0.0,
            },
            "scheduler_hint": self.executor.scheduler_hint or None,
            "daemon_value_result": result.to_json(),
        }


@dataclass(frozen=True)
class FabricExecutionResult:
    ok: bool
    graph: FabricGraph
    plan: FabricGraphPlan
    execution_backend: str
    repeats: int
    elapsed_seconds: float
    trace: tuple[dict[str, Any], ...]
    training_records: tuple[dict[str, Any], ...]
    outputs: dict[str, dict[str, Any]]
    final_residency: dict[str, tuple[str, ...]]
    output_bos: dict[str, dict[str, Any]] = field(default_factory=dict)
    fabric_values: dict[str, dict[str, Any]] = field(default_factory=dict)
    value_store: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    schema: str = FABRIC_EXECUTION_SCHEMA

    def to_json(self, *, include_trace: bool = True) -> dict[str, Any]:
        payload = {
            "schema": self.schema,
            "ok": self.ok,
            "execution_backend": self.execution_backend,
            "repeats": self.repeats,
            "elapsed_seconds": self.elapsed_seconds,
            "summary": self.summary(),
            "graph": self.graph.to_json(),
            "plan": self.plan.to_json(),
            "outputs": self.outputs,
            "output_bos": self.output_bos,
            "fabric_values": self.fabric_values,
            "value_store": self.value_store,
            "final_residency": {name: list(values) for name, values in sorted(self.final_residency.items())},
            "error": self.error,
        }
        if include_trace:
            payload["trace"] = list(self.trace)
        return payload

    def summary(self) -> dict[str, Any]:
        return {
            "step_count": len(self.trace),
            "run_ps_count": sum(1 for row in self.trace if row.get("action") == "run_ps"),
            "run_pl_count": sum(1 for row in self.trace if row.get("action") == "run_pl"),
            "run_daemon_graph_count": sum(1 for row in self.trace if row.get("action") == "run_daemon_graph"),
            "transfer_count": sum(1 for row in self.trace if row.get("action") == "transfer"),
            "observed_transfer_bytes": sum(int(row.get("observed_transfer_bytes") or 0) for row in self.trace),
            "setup_transfer_bytes": sum(
                int(row.get("observed_transfer_bytes") or 0)
                for row in self.trace
                if row.get("phase") == "setup"
            ),
            "timed_transfer_bytes": sum(
                int(row.get("observed_transfer_bytes") or 0)
                for row in self.trace
                if row.get("phase") == "timed" or row.get("timed")
            ),
            "final_transfer_bytes": sum(
                int(row.get("observed_transfer_bytes") or 0)
                for row in self.trace
                if row.get("phase") == "materialize"
            ),
            "timed_graph_packets": sum(1 for row in self.trace if row.get("phase") == "timed" or row.get("timed")),
            "timed_elapsed_seconds": sum(
                float(row.get("elapsed_seconds") or 0.0)
                for row in self.trace
                if row.get("phase") == "timed" or row.get("timed")
            ),
            "timed_daemon_elapsed_seconds": sum(
                float(row.get("daemon_elapsed_seconds") or 0.0)
                for row in self.trace
                if row.get("phase") == "timed" or row.get("timed")
            ),
            "observed_bo_sync_bytes": sum(
                int(event.get("bytes") or 0)
                for row in self.trace
                for event in row.get("bo_events") or ()
            ),
            "bo_event_count": sum(len(row.get("bo_events") or ()) for row in self.trace),
            "bo_export_count": sum(
                1
                for row in self.trace
                for event in row.get("bo_events") or ()
                if event.get("action") == "bo_export"
            ),
            "bo_import_count": sum(
                1
                for row in self.trace
                for event in row.get("bo_events") or ()
                if event.get("action") == "bo_import"
            ),
            "bo_reuse_count": sum(
                1
                for row in self.trace
                for event in row.get("bo_events") or ()
                if event.get("action") == "bo_reuse"
            ),
            "observed_run_ps_seconds": sum(
                float(row.get("elapsed_seconds") or 0.0) for row in self.trace if row.get("action") == "run_ps"
            ),
            "observed_run_pl_seconds": sum(
                float(row.get("elapsed_seconds") or 0.0) for row in self.trace if row.get("action") == "run_pl"
            ),
            "observed_run_daemon_seconds": sum(
                float(row.get("elapsed_seconds") or 0.0)
                for row in self.trace
                if row.get("action") == "run_daemon_graph"
            ),
        }


class FabricGraphExecutor:
    """Execute a planned typed SPF graph with explicit PS/PL buffer state."""

    def __init__(
        self,
        plan: FabricGraphPlan,
        *,
        execution_backend: str = "auto",
        seed: int = 1,
        workdir: Optional[str] = None,
        libspf32_path: Optional[str] = None,
        libspf16_path: Optional[str] = None,
        load_app: bool = False,
        initial_inputs: Optional[dict[str, Any]] = None,
        daemon_socket: str = "/tmp/pan-fabricd.sock",
        daemon_buffer_storage: str = "inline",
        materialize_outputs: bool = True,
        enable_bo_import: Optional[bool] = None,
        allow_xrt_app_switch: Optional[bool] = None,
        scheduler_hint: Optional[dict[str, Any]] = None,
    ):
        if execution_backend not in {"auto", "sim", "xrt", "daemon", "fabric-session"}:
            raise ValueError(f"unsupported fabric execution backend {execution_backend!r}")
        self.plan = plan
        self.graph = plan.graph
        self.seed = int(seed)
        self.workdir = workdir or "/home/ubuntu/spfr"
        self.load_app = bool(load_app)
        self.materialize_outputs = bool(materialize_outputs)
        self.enable_bo_import = (
            bool(int(os.environ.get("PAN_FABRIC_ENABLE_BO_IMPORT", "0")))
            if enable_bo_import is None
            else bool(enable_bo_import)
        )
        self.allow_xrt_app_switch = (
            bool(int(os.environ.get("PAN_FABRIC_ALLOW_XRT_APP_SWITCH", "0")))
            if allow_xrt_app_switch is None
            else bool(allow_xrt_app_switch)
        )
        self.initial_inputs = dict(initial_inputs or {})
        self.daemon_socket = str(daemon_socket)
        self.daemon_buffer_storage = str(daemon_buffer_storage)
        self.scheduler_hint = dict(scheduler_hint or {})
        self.execution_backend = self._resolve_execution_backend(execution_backend)
        self.ps = _PSOpEngine(workdir=self.workdir, libspf32_path=libspf32_path, libspf16_path=libspf16_path)
        self._sessions: dict[str, KV260FabricSession] = {}
        self._active_app: Optional[str] = None
        self._bo_events: list[dict[str, Any]] = []

    def execute(self, *, repeats: int = 1) -> FabricExecutionResult:
        if self.execution_backend == "fabric-session":
            return self._execute_fabric_session(repeats=repeats)
        repeats = max(1, int(repeats))
        trace: list[dict[str, Any]] = []
        training: list[dict[str, Any]] = []
        outputs: dict[str, dict[str, Any]] = {}
        output_bos: dict[str, dict[str, Any]] = {}
        fabric_values: dict[str, dict[str, Any]] = {}
        error = None
        started = time.perf_counter()
        try:
            for repeat_index in range(repeats):
                buffers = self._initial_buffers(repeat_index)
                if self.execution_backend == "daemon":
                    rows = self._execute_daemon_graph(buffers, repeat_index=repeat_index)
                    trace.extend(rows)
                    for row in rows:
                        if row["action"] in {"run_daemon_graph", "run_ps", "run_pl"}:
                            training.append(_training_record(row, self.plan, repeat_index=repeat_index))
                else:
                    for step_index, step in enumerate(self.plan.steps):
                        row = self._execute_step(buffers, step, step_index=step_index, repeat_index=repeat_index)
                        trace.append(row)
                        if row["action"] in {"run_ps", "run_pl"}:
                            training.append(_training_record(row, self.plan, repeat_index=repeat_index))
                if repeat_index == repeats - 1:
                    if self.materialize_outputs:
                        outputs = {
                            name: _array_digest(self._ensure_ps(buffers, name))
                            for name in self.graph.outputs
                        }
                    else:
                        outputs = {
                            name: _buffer_nonmaterialized_summary(buffers[name])
                            for name in self.graph.outputs
                        }
                    output_bos = {
                        name: descriptor
                        for name in self.graph.outputs
                        if (descriptor := buffers[name].bo_descriptor()) is not None
                    }
                    fabric_values = {
                        name: FabricValue.from_buffer_state(
                            buffers[name],
                            value_id=f"{self.execution_backend}:{name}:repeat{repeat_index}",
                            producer=self.execution_backend,
                        ).to_json()
                        for name in self.graph.outputs
                    }
                    final_residency = {
                        name: state.residency()
                        for name, state in sorted(buffers.items())
                    }
        except Exception as exc:
            error = str(exc)
            outputs = {}
            output_bos = {}
            final_residency = {}
        elapsed = time.perf_counter() - started
        return FabricExecutionResult(
            ok=error is None,
            graph=self.graph,
            plan=self.plan,
            execution_backend=self.execution_backend,
            repeats=repeats,
            elapsed_seconds=elapsed,
            trace=tuple(trace),
            training_records=tuple(training),
            outputs=outputs,
            final_residency=final_residency,
            output_bos=output_bos,
            fabric_values=fabric_values,
            error=error,
        )

    def _resolve_execution_backend(self, requested: str) -> str:
        if requested != "auto":
            return requested
        has_pl_steps = any(step.action == "run_pl" for step in self.plan.steps)
        if not has_pl_steps:
            return "sim"
        report = fabric_xrt_report()
        return "xrt" if report.get("pyxrt_imported") and report.get("device_visible") else "sim"

    def _execute_daemon_graph(
        self,
        buffers: dict[str, FabricBufferState],
        *,
        repeat_index: int,
    ) -> tuple[dict[str, Any], ...]:
        self._validate_daemon_plan()
        produced = {op.output for op in self.graph.ops}
        inputs = {
            name: self._daemon_input_value(buffers, name)
            for name, spec in self.graph.buffers.items()
            if name not in produced and spec.elements > 0
        }
        before = {name: state.residency() for name, state in sorted(buffers.items())}
        result = execute_fabric_graph_daemon(
            self.daemon_socket,
            self.plan,
            inputs,
            output_names=self.graph.outputs,
            buffer_storage=self.daemon_buffer_storage,
            materialize_outputs=self.materialize_outputs,
        )
        if not result.ok:
            raise RuntimeError(f"fabric daemon graph execution failed: {result.error or result.status}")
        for name, value in result.outputs.items():
            state = buffers[name]
            if _is_daemon_bo_descriptor(value):
                state.ps_array = None
                state.pl_bo_export = {"ok": True, **dict(value)}
                state.pl_bo_nbytes = int(value.get("nbytes") or int(state.elements) * 4)
            else:
                state.ps_array = value
                state.pl_bo_export = None
                state.pl_bo_nbytes = None
            state.pl_array = None
            state.pl_bo = None
            state.pl_group_id = None
            state.pl_session_key = "daemon"
        after = {name: state.residency() for name, state in sorted(buffers.items())}
        return (
            {
                "schema": FABRIC_EXECUTION_SCHEMA,
                "repeat_index": repeat_index,
                "step_index": -1,
                "action": "run_daemon_graph",
                "backend": "daemon",
                "op_id": "daemon_graph",
                "op": "fabric.graph",
                "buffer": None,
                "src": None,
                "dst": None,
                "kernel": None,
                "planned_latency_s": self.plan.total_latency_s,
                "planned_energy_j": self.plan.total_energy_j,
                "planned_transfer_bytes": self.plan.transfer_bytes,
                "observed_transfer_bytes": result.input_bytes + result.output_bytes,
                "elapsed_seconds": result.roundtrip_seconds,
                "daemon_elapsed_seconds": result.daemon_elapsed_seconds,
                "daemon_socket": self.daemon_socket,
                "daemon_buffer_storage": self.daemon_buffer_storage,
                "features": {
                    "daemon_graph_ops": float(result.graph_op_count),
                    "daemon_input_bytes": float(result.input_bytes),
                    "daemon_output_bytes": float(result.output_bytes),
                    "daemon_resident_outputs": float(result.resident_output_count),
                    "daemon_avoided_host_materializations": float(result.avoided_host_materialization_count),
                    "latency_s": self.plan.total_latency_s,
                    "energy_j": self.plan.total_energy_j,
                    "transfer_bytes": float(result.input_bytes + result.output_bytes),
                    "unsupported": 0.0,
                },
                "residency_before": {name: list(values) for name, values in before.items()},
                "residency_after": {name: list(values) for name, values in after.items()},
                "daemon_result": result.to_json(),
            },
        )

    def _validate_daemon_plan(self) -> None:
        if any(step.action == "run_ps" for step in self.plan.steps):
            raise ValueError("daemon execution requires a PL-only graph plan; use --fabric-backend pl or sim/xrt")
        run_pl = [step for step in self.plan.steps if step.action == "run_pl"]
        if len(run_pl) != len(self.graph.ops):
            raise ValueError("daemon execution requires every graph op to be a PL run step")
        unsupported = [
            step
            for step in run_pl
            if step.kernel is None
            or step.kernel.call_abi not in {"binary_u32_n", "unary_u32_n", "fft64_u32"}
            or step.op not in SUPPORTED_DAEMON_GRAPH_OPS
        ]
        if unsupported:
            names = ", ".join(str(step.op) for step in unsupported)
            raise ValueError(f"daemon graph v1 does not support these graph ops: {names}")

    def _execute_fabric_session(self, *, repeats: int) -> FabricExecutionResult:
        repeats = max(1, int(repeats))
        outputs: dict[str, dict[str, Any]] = {}
        output_bos: dict[str, dict[str, Any]] = {}
        fabric_values: dict[str, dict[str, Any]] = {}
        value_store_json: dict[str, Any] = {}
        final_residency: dict[str, tuple[str, ...]] = {}
        error = None
        started = time.perf_counter()
        runtime_session = FabricRuntimeSession(self)
        try:
            self._validate_fabric_session_plan()
            with runtime_session:
                session_result = runtime_session.execute_resident_graph_chain(repeats=repeats)
            buffers = session_result["buffers"]
            output_name = str(session_result["output_name"])
            output_value: FabricValue = session_result["output_value"]
            output_state = buffers[output_name]
            output_state.ps_array = output_value.ps_array
            output_state.pl_array = None
            output_state.pl_bo = None
            output_state.pl_group_id = None
            output_state.pl_session_key = "daemon" if output_value.daemon_handle is not None else None
            output_state.pl_bo_export = (
                {"ok": True, **dict(output_value.daemon_handle)}
                if output_value.daemon_handle is not None
                else None
            )
            output_state.pl_bo_nbytes = (
                int(output_value.daemon_handle.get("nbytes") or int(output_value.elements) * 4)
                if output_value.daemon_handle is not None
                else None
            )
            if output_value.ps_array is not None:
                outputs = {output_name: _array_digest(output_value.ps_array)}
            else:
                outputs = {
                    output_name: {
                        "elements": int(output_value.elements),
                        "materialized": False,
                        "residency": [output_value.residency],
                        "bo": _daemon_bo_descriptor_for_value(output_value),
                        "fabric_value": output_value.to_json(),
                    }
                }
                descriptor = _daemon_bo_descriptor_for_value(output_value)
                output_bos = {output_name: descriptor} if descriptor is not None else {}
            fabric_values = {output_name: output_value.to_json()}
            value_store_json = runtime_session.value_store.to_json()
            final_residency = {name: state.residency() for name, state in sorted(buffers.items())}
        except Exception as exc:
            error = str(exc)
            outputs = {}
            output_bos = {}
            fabric_values = {}
            value_store_json = runtime_session.value_store.to_json()
            final_residency = {}
        elapsed = time.perf_counter() - started
        return FabricExecutionResult(
            ok=error is None,
            graph=self.graph,
            plan=self.plan,
            execution_backend=self.execution_backend,
            repeats=repeats,
            elapsed_seconds=elapsed,
            trace=tuple(runtime_session.trace),
            training_records=tuple(runtime_session.training_records),
            outputs=outputs,
            final_residency=final_residency,
            output_bos=output_bos,
            fabric_values=fabric_values,
            value_store=value_store_json,
            error=error,
        )

    def _validate_fabric_session_plan(self) -> None:
        self._validate_daemon_plan()
        if len(self.graph.outputs) != 1:
            raise ValueError("fabric-session v1 supports exactly one graph output")
        if not self.graph.ops or not self.graph.ops[0].inputs:
            raise ValueError("fabric-session v1 requires at least one graph op with an input")

    def _fabric_session_packet(
        self,
        session: FabricDaemonGraphSession,
        inputs: dict[str, Any],
        *,
        repeat_index: int,
        step_index: int,
        phase: str,
        materialize_outputs: bool,
        note: str,
    ) -> tuple[dict[str, Any], Any]:
        result = session.execute_graph(
            self.plan,
            inputs,
            output_names=self.graph.outputs,
            buffer_storage=self.daemon_buffer_storage,
            materialize_outputs=materialize_outputs,
        )
        if not result.ok:
            raise RuntimeError(f"fabric daemon session packet failed: {result.error or result.status}")
        observed_transfer_bytes = result.input_bytes + result.output_bytes
        row = {
            "schema": FABRIC_EXECUTION_SCHEMA,
            "repeat_index": repeat_index,
            "step_index": step_index,
            "phase": phase,
            "timed": phase == "timed",
            "action": "run_daemon_graph",
            "backend": "daemon_session_resident",
            "op_id": "fabric_session",
            "op": "spf32.mul",
            "buffer": None,
            "src": None,
            "dst": None,
            "kernel": None,
            "planned_latency_s": self.plan.total_latency_s,
            "planned_energy_j": self.plan.total_energy_j,
            "planned_transfer_bytes": self.plan.transfer_bytes,
            "observed_transfer_bytes": observed_transfer_bytes,
            "elapsed_seconds": result.roundtrip_seconds,
            "daemon_elapsed_seconds": result.daemon_elapsed_seconds,
            "daemon_socket": self.daemon_socket,
            "daemon_buffer_storage": self.daemon_buffer_storage,
            "session_note": note,
            "features": {
                "daemon_graph_ops": float(result.graph_op_count),
                "daemon_input_bytes": float(result.input_bytes),
                "daemon_output_bytes": float(result.output_bytes),
                "daemon_resident_outputs": float(result.resident_output_count),
                "daemon_avoided_host_materializations": float(result.avoided_host_materialization_count),
                "latency_s": self.plan.total_latency_s,
                "energy_j": self.plan.total_energy_j,
                "transfer_bytes": float(observed_transfer_bytes),
                "unsupported": 0.0,
            },
            "scheduler_hint": self.scheduler_hint or None,
            "daemon_result": result.to_json(),
        }
        return row, result

    def _validate_fabric_session_output(
        self,
        *,
        actual: Any,
        initial_inputs: dict[str, Any],
        repeats: int,
    ) -> dict[str, Any]:
        input_names = self._graph_input_names()
        if any(name not in initial_inputs for name in input_names):
            return {
                "oracle": "spf_graph_chain",
                "ok": None,
                "reason": "daemon_input_not_materialized_for_ps_oracle",
            }
        values = {
            name: _coerce_input(initial_inputs[name], self.graph.buffers[name].elements)
            for name in input_names
        }
        first_input = self.graph.ops[0].inputs[0]
        output_name = self.graph.outputs[0]
        current = values[first_input]
        for _ in range(int(repeats)):
            values[first_input] = current
            for op in self.graph.ops:
                inputs = [values[name] for name in op.inputs]
                values[op.output] = self.ps.compute(op.op, inputs, op.elements)
            current = values[output_name]
        actual_digest = _array_digest(actual)
        expected_digest = _array_digest(current)
        return {
            "oracle": "spf_graph_chain",
            "graph_ops": [op.op for op in self.graph.ops],
            "ok": actual_digest["sha256"] == expected_digest["sha256"],
            "actual_sha256": actual_digest["sha256"],
            "expected_sha256": expected_digest["sha256"],
            "repeats": int(repeats),
        }

    def _graph_input_names(self) -> tuple[str, ...]:
        produced = {op.output for op in self.graph.ops}
        return tuple(
            name
            for name, spec in self.graph.buffers.items()
            if name not in produced and int(spec.elements) > 0
        )

    def _daemon_input_value(self, buffers: dict[str, FabricBufferState], name: str) -> Any:
        state = buffers[name]
        if state.pl_session_key == "daemon" and state.pl_bo_export and _is_daemon_bo_descriptor(state.pl_bo_export):
            return state.pl_bo_export
        return self._ensure_ps(buffers, name)

    def _initial_buffers(self, repeat_index: int) -> dict[str, FabricBufferState]:
        buffers = {
            name: FabricBufferState(name=name, elements=spec.elements)
            for name, spec in self.graph.buffers.items()
        }
        produced = {op.output for op in self.graph.ops}
        for name, state in buffers.items():
            spec = self.graph.buffers[name]
            if name in produced and spec.residency == "unknown":
                continue
            source = self.initial_inputs.get(name)
            if source is not None and _is_daemon_bo_descriptor(source):
                if self.execution_backend not in {"daemon", "fabric-session"}:
                    raise ValueError(f"daemon BO input {name!r} requires daemon or fabric-session execution")
                if int(source.get("elements", spec.elements)) != int(spec.elements):
                    raise ValueError(
                        f"daemon BO input {name!r} expected {spec.elements} words, got {source.get('elements')}"
                    )
                state.pl_bo_export = {"ok": True, **dict(source)}
                state.pl_bo_nbytes = int(source.get("nbytes") or int(spec.elements) * 4)
                state.pl_session_key = "daemon"
            else:
                state.ps_array = _coerce_input(source, spec.elements) if source is not None else _seed_words(
                    self.seed,
                    repeat_index,
                    name,
                    spec.elements,
                )
            if spec.residency in {"pl", "both"} and self.execution_backend == "sim":
                state.pl_array = state.ps_array.copy()
        return buffers

    def _execute_step(
        self,
        buffers: dict[str, FabricBufferState],
        step: FabricGraphPlanStep,
        *,
        step_index: int,
        repeat_index: int,
    ) -> dict[str, Any]:
        before = {name: state.residency() for name, state in sorted(buffers.items())}
        bo_event_start = len(self._bo_events)
        started = time.perf_counter()
        observed_transfer_bytes = 0
        if step.action == "transfer":
            observed_transfer_bytes = self._execute_transfer(buffers, step)
        elif step.action == "run_ps":
            self._execute_ps(buffers, step)
        elif step.action == "run_pl":
            if self.execution_backend == "xrt":
                observed_transfer_bytes = self._execute_pl_xrt(buffers, step)
            else:
                self._execute_pl_sim(buffers, step)
        else:
            raise ValueError(f"unsupported fabric plan action {step.action!r}")
        elapsed = time.perf_counter() - started
        after = {name: state.residency() for name, state in sorted(buffers.items())}
        bo_events = self._bo_events[bo_event_start:]
        return {
            "schema": FABRIC_EXECUTION_SCHEMA,
            "repeat_index": repeat_index,
            "step_index": step_index,
            "action": step.action,
            "backend": step.backend,
            "op_id": step.op_id,
            "op": step.op,
            "buffer": step.buffer,
            "src": step.src,
            "dst": step.dst,
            "kernel": None if step.kernel is None else step.kernel.to_json(),
            "planned_latency_s": step.latency_s,
            "planned_energy_j": step.energy_j,
            "planned_transfer_bytes": step.transfer_bytes,
            "observed_transfer_bytes": observed_transfer_bytes,
            "elapsed_seconds": elapsed,
            "features": dict(sorted(step.features.items())),
            "bo_events": bo_events,
            "residency_before": {name: list(values) for name, values in before.items()},
            "residency_after": {name: list(values) for name, values in after.items()},
        }

    def _execute_transfer(self, buffers: dict[str, FabricBufferState], step: FabricGraphPlanStep) -> int:
        if step.buffer is None or step.src is None or step.dst is None:
            raise ValueError("transfer step is missing buffer/src/dst")
        state = buffers[step.buffer]
        if step.dst == "ps":
            self._ensure_ps(buffers, step.buffer)
        elif step.dst == "pl":
            if self.execution_backend == "sim":
                self._ensure_pl_sim(buffers, step.buffer)
            else:
                # The actual BO bank is kernel-argument-specific, so XRT mode
                # materializes this lazily at the following PL run.
                self._ensure_ps(buffers, step.buffer)
        else:
            raise ValueError(f"unsupported transfer destination {step.dst!r}")
        return int(state.elements) * 4

    def _execute_ps(self, buffers: dict[str, FabricBufferState], step: FabricGraphPlanStep) -> None:
        op = self._graph_op(step)
        inputs = [self._ensure_ps(buffers, name) for name in op.inputs]
        buffers[op.output].ps_array = self.ps.compute(op.op, inputs, op.elements)
        buffers[op.output].pl_array = None
        buffers[op.output].pl_bo = None
        buffers[op.output].pl_group_id = None
        buffers[op.output].pl_session_key = None
        buffers[op.output].pl_bo_nbytes = None
        buffers[op.output].pl_bo_export = None

    def _execute_pl_sim(self, buffers: dict[str, FabricBufferState], step: FabricGraphPlanStep) -> None:
        if step.kernel is None:
            raise ValueError(f"PL-sim step {step.op_id} has no kernel")
        op = self._graph_op(step)
        inputs = [self._ensure_pl_sim(buffers, name) for name in op.inputs]
        buffers[op.output].pl_array = self.ps.compute(op.op, inputs, op.elements)
        buffers[op.output].ps_array = None
        buffers[op.output].pl_bo = None
        buffers[op.output].pl_group_id = None
        buffers[op.output].pl_session_key = "sim"
        buffers[op.output].pl_bo_nbytes = None
        buffers[op.output].pl_bo_export = None

    def _execute_pl_xrt(self, buffers: dict[str, FabricBufferState], step: FabricGraphPlanStep) -> int:
        if step.kernel is None:
            raise ValueError(f"PL step {step.op_id} has no kernel")
        spec = step.kernel
        op = self._graph_op(step)
        if spec.fixed_elements is not None and spec.fixed_elements != op.elements:
            raise ValueError(f"{spec.op} expects {spec.fixed_elements} words, got {op.elements}")
        if self.load_app and self._active_app != spec.app_name:
            if self._active_app is not None and not self.allow_xrt_app_switch:
                raise RuntimeError(
                    "xrt_app_switch_requires_process_boundary: "
                    f"cannot switch from {self._active_app!r} to {spec.app_name!r} "
                    "inside this Python XRT process; use a fused foundry core, a daemon process boundary, "
                    "or set PAN_FABRIC_ALLOW_XRT_APP_SWITCH=1 to force the experimental path"
                )
            self._prepare_for_app_switch(buffers, spec.app_name)
            report = load_xmutil_app(spec.app_name)
            if not report.get("ok"):
                raise RuntimeError(f"xmutil load failed for {spec.app_name}: {report}")
            self._active_app = spec.app_name
        session = self._session(spec)
        kernel = session.kernel(spec.kernel_name)
        nbytes = int(op.elements) * 4
        started_bytes = 0
        if spec.call_abi == "binary_u32_n":
            bo_a, bytes_a = self._ensure_pl_bo(buffers, op.inputs[0], session, kernel.group_id(0), nbytes)
            bo_b, bytes_b = self._ensure_pl_bo(buffers, op.inputs[1], session, kernel.group_id(1), nbytes)
            bo_out = self._new_output_bo(buffers, op.output, session, kernel.group_id(2), nbytes)
            run = kernel(bo_a, bo_b, bo_out, int(op.elements))
            started_bytes = bytes_a + bytes_b
        elif spec.call_abi == "unary_u32_n":
            bo_in, started_bytes = self._ensure_pl_bo(buffers, op.inputs[0], session, kernel.group_id(0), nbytes)
            bo_out = self._new_output_bo(buffers, op.output, session, kernel.group_id(1), nbytes)
            run = kernel(bo_in, bo_out, int(op.elements))
        elif spec.call_abi == "fft64_u32":
            bo_in, started_bytes = self._ensure_pl_bo(buffers, op.inputs[0], session, kernel.group_id(0), nbytes)
            bo_out = self._new_output_bo(buffers, op.output, session, kernel.group_id(1), nbytes)
            run = kernel(bo_in, bo_out, _scalar_n_arg(spec, int(op.elements))) if spec.requires_scalar_n else kernel(bo_in, bo_out)
        else:
            raise ValueError(f"unsupported fabric call ABI {spec.call_abi!r}")
        run.wait()
        return started_bytes

    def _session(self, spec: FabricKernelSpec) -> KV260FabricSession:
        key = spec.xclbin_path
        if key not in self._sessions:
            self._sessions[key] = KV260FabricSession(spec.xclbin_path)
        return self._sessions[key]

    def _ensure_ps(self, buffers: dict[str, FabricBufferState], name: str) -> Any:
        state = buffers[name]
        if state.ps_array is not None:
            return state.ps_array
        if state.pl_array is not None:
            state.ps_array = state.pl_array.copy()
            return state.ps_array
        if state.pl_bo is not None:
            session = self._sessions.get(state.pl_session_key or "")
            if session is None:
                raise RuntimeError(f"missing XRT session for PL buffer {name}")
            nbytes = int(state.elements) * 4
            _xrt_sync_bo(state.pl_bo, session.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, nbytes)
            state.ps_array = _xrt_read_array(state.pl_bo, state.elements)
            self._record_bo_event("bo_sync_from_device", name=name, session_key=state.pl_session_key, nbytes=nbytes)
            return state.ps_array
        raise ValueError(f"buffer {name!r} has no materialized value")

    def _ensure_pl_sim(self, buffers: dict[str, FabricBufferState], name: str) -> Any:
        state = buffers[name]
        if state.pl_array is None:
            state.pl_array = self._ensure_ps(buffers, name).copy()
        return state.pl_array

    def _ensure_pl_bo(
        self,
        buffers: dict[str, FabricBufferState],
        name: str,
        session: KV260FabricSession,
        group_id: int,
        nbytes: int,
    ) -> tuple[Any, int]:
        state = buffers[name]
        session_key = session.xclbin_path
        if (
            state.pl_bo is not None
            and state.pl_session_key == session_key
            and state.pl_bo_nbytes == int(nbytes)
        ):
            self._record_bo_event(
                "bo_reuse",
                name=name,
                session_key=session_key,
                from_group_id=state.pl_group_id,
                to_group_id=int(group_id),
                nbytes=nbytes,
            )
            return state.pl_bo, 0
        if self.enable_bo_import and state.pl_bo_export and state.pl_bo_export.get("ok"):
            try:
                bo = _xrt_import_bo(session, state.pl_bo_export, int(group_id), int(nbytes))
            except Exception as exc:
                self._record_bo_event(
                    "bo_import_failed",
                    name=name,
                    session_key=session_key,
                    group_id=group_id,
                    nbytes=nbytes,
                    error=str(exc),
                )
            else:
                state.pl_bo = bo
                state.pl_group_id = int(group_id)
                state.pl_session_key = session_key
                state.pl_bo_nbytes = int(nbytes)
                self._record_bo_event("bo_import", name=name, session_key=session_key, group_id=group_id, nbytes=nbytes)
                return bo, 0
        arr = self._ensure_ps(buffers, name)
        bo = _new_xrt_bo(session, int(group_id), int(nbytes))
        _xrt_write_array(bo, arr)
        _xrt_sync_bo(bo, session.xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, nbytes)
        state.pl_bo = bo
        state.pl_group_id = int(group_id)
        state.pl_session_key = session_key
        state.pl_bo_nbytes = int(nbytes)
        state.pl_bo_export = None
        self._record_bo_event("bo_sync_to_device", name=name, session_key=session_key, group_id=group_id, nbytes=nbytes)
        return bo, int(nbytes)

    def _new_output_bo(
        self,
        buffers: dict[str, FabricBufferState],
        name: str,
        session: KV260FabricSession,
        group_id: int,
        nbytes: int,
    ) -> Any:
        state = buffers[name]
        bo = _new_xrt_bo(session, int(group_id), int(nbytes))
        state.ps_array = None
        state.pl_array = None
        state.pl_bo = bo
        state.pl_group_id = int(group_id)
        state.pl_session_key = session.xclbin_path
        state.pl_bo_nbytes = int(nbytes)
        state.pl_bo_export = None
        return bo

    def _prepare_for_app_switch(self, buffers: dict[str, FabricBufferState], next_app: str) -> None:
        released = 0
        for name, state in sorted(buffers.items()):
            if state.pl_bo is None:
                continue
            if state.pl_bo_export is None:
                state.pl_bo_export = _xrt_export_bo(state.pl_bo)
                self._record_bo_event(
                    "bo_export",
                    name=name,
                    session_key=state.pl_session_key,
                    group_id=state.pl_group_id,
                    nbytes=state.pl_bo_nbytes,
                    export=state.pl_bo_export,
                    next_app=next_app,
                )
            if not self.enable_bo_import and state.ps_array is None:
                self._ensure_ps(buffers, name)
            elif state.pl_bo_export is not None and not state.pl_bo_export.get("ok") and state.ps_array is None:
                self._ensure_ps(buffers, name)
            state.pl_bo = None
            released += 1
        if released:
            self._sessions.clear()
            gc.collect()
            self._record_bo_event("xrt_session_reset", next_app=next_app, released_bo_count=released)

    def _record_bo_event(self, action: str, **fields: Any) -> None:
        nbytes = fields.pop("nbytes", None)
        row = {
            "action": action,
            "bytes": 0 if nbytes is None else int(nbytes),
            **fields,
        }
        self._bo_events.append(row)

    def _graph_op(self, step: FabricGraphPlanStep) -> Any:
        for op in self.graph.ops:
            if op.op_id == step.op_id:
                return op
        raise ValueError(f"unknown graph op id {step.op_id!r}")


class _PSOpEngine:
    def __init__(self, *, workdir: str, libspf32_path: Optional[str], libspf16_path: Optional[str] = None):
        self.workdir = str(workdir)
        self.lib = None
        try:
            self.lib = ctypes.CDLL(_find_libspf32(libspf32_path, workdir=workdir))
        except Exception:
            self.lib = None
        self.libspf16 = None
        try:
            self.libspf16 = ctypes.CDLL(_find_libspf16(libspf16_path, workdir=workdir))
        except Exception:
            self.libspf16 = None
        self._configured: set[str] = set()

    def compute(self, op: str, inputs: Sequence[Any], elements: int) -> Any:
        if op.startswith("spf16."):
            return self._compute_spf16_packed(op, inputs, elements)
        if op in {"spf32.fft64", "spf32.fft64_fb7", "spf32.fft64_fb7_fields"}:
            return self._compute_spf32_fft(op, inputs, elements)
        if op.startswith("spf32.fb7_bittrick"):
            raise RuntimeError(f"{op} is quarantined as unsafe; no validation oracle is exposed")
        if self.lib is not None and self._can_call_lib(op, len(inputs)):
            try:
                return self._compute_lib(op, inputs, elements)
            except Exception:
                pass
        return _compute_raw_u32(op, inputs, elements)

    def _can_call_lib(self, op: str, arity: int) -> bool:
        if not op.startswith("spf32."):
            return False
        name = op.split(".", 1)[1]
        fn = f"spf32_{name}_array"
        return hasattr(self.lib, fn) and arity in {1, 2}

    def _compute_lib(self, op: str, inputs: Sequence[Any], elements: int) -> Any:
        np = _numpy()
        name = op.split(".", 1)[1]
        fn_name = f"spf32_{name}_array"
        fn = getattr(self.lib, fn_name)
        arity = len(inputs)
        if fn_name not in self._configured:
            if arity == 1:
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                ]
            else:
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_size_t,
                ]
            fn.restype = None
            self._configured.add(fn_name)
        arrays = [np.ascontiguousarray(value, dtype=np.uint32).reshape(-1) for value in inputs]
        out = np.empty(int(elements), dtype=np.uint32)
        if arity == 1:
            fn(
                arrays[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_size_t(elements),
            )
        else:
            fn(
                arrays[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                arrays[1].ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                ctypes.c_size_t(elements),
            )
        return out

    def _compute_spf16_packed(self, op: str, inputs: Sequence[Any], elements: int) -> Any:
        if self.libspf16 is None:
            raise RuntimeError("typed SPF16 oracle unavailable; set SPF16_LIBRARY or SPF_LIBSPF16_PATH")
        np = _numpy()
        name = op.split(".", 1)[1]
        fn_name = f"spf16_{name}_array"
        if not hasattr(self.libspf16, fn_name):
            raise RuntimeError(f"typed SPF16 oracle missing symbol {fn_name}")
        fn = getattr(self.libspf16, fn_name)
        arity = len(inputs)
        if arity not in {1, 2}:
            raise ValueError(f"{op} expects one or two packed SPF16 inputs")
        config_key = f"spf16:{fn_name}:{arity}"
        if config_key not in self._configured:
            if arity == 1:
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.c_size_t,
                ]
            else:
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.POINTER(ctypes.c_uint16),
                    ctypes.c_size_t,
                ]
            fn.restype = None
            self._configured.add(config_key)
        arrays = [
            np.ascontiguousarray(value, dtype=np.uint32).reshape(-1).view(np.uint16)
            for value in inputs
        ]
        if any(int(arr.size) != int(elements) * 2 for arr in arrays):
            raise ValueError(f"{op} packed SPF16 input size mismatch")
        out16 = np.empty(int(elements) * 2, dtype=np.uint16)
        if arity == 1:
            fn(
                arrays[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_size_t(out16.size),
            )
        else:
            fn(
                arrays[0].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                arrays[1].ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                out16.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                ctypes.c_size_t(out16.size),
            )
        return np.ascontiguousarray(out16.view(np.uint32))

    def _compute_spf32_fft(self, op: str, inputs: Sequence[Any], elements: int) -> Any:
        if len(inputs) != 1:
            raise ValueError(f"{op} expects one SPF32 input")
        np = _numpy()
        arr = np.ascontiguousarray(inputs[0], dtype=np.uint32).reshape(-1)
        if int(arr.size) != int(elements):
            raise ValueError(f"{op} input size mismatch")
        _install_spf_python_paths(self.workdir)
        if op == "spf32.fft64":
            try:
                from spf.fft import spf32_fft  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"typed SPF32 FFT oracle unavailable: {exc}") from exc
            return np.ascontiguousarray(spf32_fft.fft(arr), dtype=np.uint32)
        if op == "spf32.fft64_fb7":
            try:
                from spf.native import fb7  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"typed SPF32 FB7 FFT oracle unavailable: {exc}") from exc
            return np.ascontiguousarray(fb7.fft64(arr, mode="balanced"), dtype=np.uint32)
        if op == "spf32.fft64_fb7_fields":
            try:
                from spf.fft import fb7_fields  # type: ignore
            except Exception as exc:
                raise RuntimeError(f"typed SPF32 FB7 fields oracle unavailable: {exc}") from exc
            return np.ascontiguousarray(fb7_fields.fft_fb7_fields(arr, return_words=True), dtype=np.uint32)
        raise ValueError(f"unsupported SPF32 FFT op {op!r}")


def train_fabric_policy_from_traces(
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
                if row.get("schema") == FABRIC_TRAINING_SCHEMA and row.get("features"):
                    if not isinstance(row.get("labels"), dict):
                        row["labels"] = _training_labels(row)
                    rows.append(row)
    if not rows:
        raise ValueError("no fabric training records found")
    np = _numpy()
    feature_names = sorted({name for row in rows for name in row["features"]})
    x = np.ones((len(rows), len(feature_names) + 1), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)
    for row_index, row in enumerate(rows):
        for feature_index, name in enumerate(feature_names):
            x[row_index, feature_index + 1] = float(row["features"].get(name, 0.0))
        y[row_index] = float(row.get("observed_seconds") or row.get("elapsed_seconds") or 0.0)
    reg = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    coef = np.linalg.pinv(x.T @ x + reg) @ x.T @ y
    return FabricPolicyHead(
        bias=float(coef[0]),
        weights={name: float(coef[index + 1]) for index, name in enumerate(feature_names)},
        policy_kind="linear_neural_fabric_head_v1_trained_from_trace",
        policy_role="neural_candidate",
        promotion_status="candidate",
        can_override_rules=False,
        min_confidence=0.85,
        heads=_fabric_policy_heads(rows),
        heldout_metrics={
            "status": "unmeasured",
            "heads_trained": [
                "placement_action",
                "output_residency",
                "materialization_needed",
                "compile_priority",
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


def load_fabric_inputs(path: Optional[str | Path]) -> Optional[dict[str, Any]]:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fabric input JSON must be an object keyed by buffer name")
    return payload


def write_jsonl(rows: Sequence[dict[str, Any]], path: Optional[str | Path]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _training_record(row: dict[str, Any], plan: FabricGraphPlan, *, repeat_index: int) -> dict[str, Any]:
    labels = _training_labels(row)
    return {
        "schema": FABRIC_TRAINING_SCHEMA,
        "repeat_index": repeat_index,
        "action": row.get("action"),
        "backend": row.get("backend"),
        "op_id": row.get("op_id"),
        "op": row.get("op"),
        "features": dict(row.get("features") or {}),
        "planned_latency_s": row.get("planned_latency_s"),
        "planned_energy_j": row.get("planned_energy_j"),
        "planned_transfer_bytes": row.get("planned_transfer_bytes"),
        "observed_seconds": row.get("elapsed_seconds"),
        "observed_transfer_bytes": row.get("observed_transfer_bytes"),
        "labels": labels,
        "policy": plan.policy.to_json(),
    }


def _training_labels(row: dict[str, Any]) -> dict[str, Any]:
    hint = dict(row.get("scheduler_hint") or {})
    action_plan = list(hint.get("action_plan") or ())
    placement_action = row.get("placement_action") or _placement_label_from_row(row)
    return {
        "placement_action": placement_action,
        "output_residency": _output_residency_label(row),
        "materialization_needed": bool(row.get("materialization_reason") or placement_action == "materialize"),
        "compile_priority": "compile_better_for_next_time" if (
            hint.get("compile_better_for_next_time") or "compile_better_for_next_time" in action_plan
        ) else "none",
    }


def _placement_label_from_row(row: dict[str, Any]) -> str:
    action = str(row.get("action") or "")
    backend = str(row.get("backend") or "")
    if action == "run_ps":
        return "run_ps_now"
    if action == "run_pl":
        return "run_existing_pl_now"
    if action == "run_daemon_graph" and backend == "daemon_session_resident":
        return "run_daemon_resident_now"
    if action == "materialize":
        return "materialize"
    if action in {"import_value", "release_value"}:
        return "keep_resident"
    return action or "unknown"


def _output_residency_label(row: dict[str, Any]) -> str:
    if row.get("action") == "materialize":
        return "ps"
    after = dict(row.get("residency_after") or {})
    values = []
    for residency in after.values():
        if isinstance(residency, (list, tuple)):
            values.extend(str(item) for item in residency)
        elif residency is not None:
            values.append(str(residency))
    if "daemon" in values:
        return "daemon"
    if "pl" in values or "xrt_bo" in values:
        return "pl"
    if "ps" in values:
        return "ps"
    return "unknown"


def _fabric_policy_heads(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    labels = [dict(row.get("labels") or {}) for row in rows]
    return {
        "placement_action": {
            "kind": "linear_latency_score",
            "classes": _label_counts(labels, "placement_action"),
        },
        "output_residency": {
            "kind": "categorical_majority",
            "classes": _label_counts(labels, "output_residency"),
        },
        "materialization_needed": {
            "kind": "binary_rate",
            "positive_count": sum(1 for label in labels if bool(label.get("materialization_needed"))),
            "sample_count": len(labels),
        },
        "compile_priority": {
            "kind": "categorical_priority",
            "classes": _label_counts(labels, "compile_priority"),
        },
    }


def _label_counts(labels: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for label in labels:
        value = str(label.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _coerce_input(value: Any, elements: int) -> Any:
    np = _numpy()
    if isinstance(value, dict) and "words" in value:
        value = value["words"]
    arr = np.asarray(value, dtype=np.uint32).reshape(-1)
    if int(arr.size) != int(elements):
        raise ValueError(f"input length mismatch: expected {elements}, got {arr.size}")
    return np.ascontiguousarray(arr, dtype=np.uint32)


def _buffer_nonmaterialized_summary(state: FabricBufferState) -> dict[str, Any]:
    descriptor = state.bo_descriptor()
    return {
        "elements": int(state.elements),
        "materialized": False,
        "residency": list(state.residency()),
        "bo": descriptor,
        "fabric_value": FabricValue.from_buffer_state(
            state,
            value_id=f"buffer:{state.name}",
            producer="buffer",
        ).to_json(),
    }


def _daemon_bo_descriptor_for_value(value: FabricValue) -> Optional[dict[str, Any]]:
    if value.daemon_handle is None:
        return None
    return {
        "name": value.name,
        "backend": "daemon",
        "elements": int(value.elements),
        "session_key": "daemon",
        "group_id": None,
        "nbytes": value.daemon_handle.get("nbytes"),
        "resident": True,
        "export": dict(value.daemon_handle),
    }


def _daemon_bo_descriptor_from_value_result(result: Any, *, name: str) -> dict[str, Any]:
    payload = result.json_payload() if getattr(result, "payload", b"") else {}
    handle = int(payload.get("handle", result.handle))
    elements = int(payload.get("elements", result.elements))
    nbytes = int(payload.get("nbytes", elements * 4))
    return {
        "schema": "pan_daemon_bo_v1",
        "backend": "daemon",
        "name": str(name),
        "handle": handle,
        "elements": elements,
        "nbytes": nbytes,
        "dtype": str(payload.get("dtype") or "uint32"),
        "resident": True,
    }


def _single_daemon_output(result: Any) -> dict[str, Any]:
    value = next(iter(result.outputs.values()))
    if not _is_daemon_bo_descriptor(value):
        raise RuntimeError("expected daemon-resident BO output")
    return value


def _single_materialized_output(result: Any) -> Any:
    value = next(iter(result.outputs.values()))
    if _is_daemon_bo_descriptor(value):
        raise RuntimeError("expected materialized daemon output")
    return value


def _seed_words(seed: int, repeat_index: int, name: str, elements: int) -> Any:
    np = _numpy()
    h = hashlib.sha256(f"{seed}:{repeat_index}:{name}".encode("utf-8")).digest()
    rng_seed = int.from_bytes(h[:8], "little") & ((1 << 63) - 1)
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 2**32, size=int(elements), dtype=np.uint32)


def _compute_raw_u32(op: str, inputs: Sequence[Any], elements: int) -> Any:
    np = _numpy()
    arrays = [np.ascontiguousarray(value, dtype=np.uint32).reshape(-1) for value in inputs]
    if not arrays:
        raise ValueError(f"{op} needs input buffers")
    if any(int(arr.size) != int(elements) for arr in arrays):
        raise ValueError(f"{op} input size mismatch")
    name = op.split(".", 1)[1] if "." in op else op
    if name in {"add"} and len(arrays) == 2:
        return np.ascontiguousarray(arrays[0] + arrays[1], dtype=np.uint32)
    if name in {"mul"} and len(arrays) == 2:
        return np.ascontiguousarray(arrays[0] * arrays[1], dtype=np.uint32)
    if name in {"div"} and len(arrays) == 2:
        denom = np.where(arrays[1] == 0, np.uint32(1), arrays[1])
        return np.ascontiguousarray(arrays[0] // denom, dtype=np.uint32)
    if name in {"neg"} and len(arrays) == 1:
        return np.ascontiguousarray(-arrays[0], dtype=np.uint32)
    if name in {"conj"} and len(arrays) == 1:
        return np.ascontiguousarray(arrays[0] ^ np.uint32(0x80000000), dtype=np.uint32)
    if name in {"abs2"} and len(arrays) == 1:
        return np.ascontiguousarray(arrays[0] * arrays[0], dtype=np.uint32)
    if name.startswith("fft64") and len(arrays) == 1:
        return np.ascontiguousarray(np.roll(arrays[0], 1) ^ np.uint32(0x9E3779B9), dtype=np.uint32)
    if name.startswith("fb7_bittrick") and len(arrays) == 1:
        return np.ascontiguousarray(((arrays[0] << np.uint32(7)) | (arrays[0] >> np.uint32(25))), dtype=np.uint32)
    raise ValueError(f"unsupported PS/sim fabric op {op!r}")


def _array_digest(array: Any) -> dict[str, Any]:
    np = _numpy()
    arr = np.ascontiguousarray(array, dtype=np.uint32).reshape(-1)
    return {
        "elements": int(arr.size),
        "sha256": hashlib.sha256(memoryview(arr).cast("B")).hexdigest(),
        "first_words": [f"0x{int(value):08x}" for value in arr[:8]],
    }


def _find_libspf16(explicit: Optional[str], *, workdir: str) -> str:
    if explicit:
        return explicit
    for env_name in ("SPF_LIBSPF16_PATH", "SPF16_LIBRARY"):
        env = os.environ.get(env_name)
        if env:
            return env
    candidates = (
        Path(workdir) / "lib" / "libspf16.so",
        Path(workdir) / "native" / "libspf16.so",
        Path("/home/ubuntu/spfr/lib/libspf16.so"),
        Path("/home/ubuntu/spfr/native/libspf16.so"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("libspf16.so not found; set SPF16_LIBRARY or SPF_LIBSPF16_PATH")


def _install_spf_python_paths(workdir: str) -> None:
    candidates = (
        Path(workdir),
        Path(workdir) / "src",
        Path(workdir) / "python",
        Path("/home/ubuntu/spfr"),
        Path("/home/ubuntu/spfr/src"),
    )
    for candidate in reversed(candidates):
        if candidate.is_dir():
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)


def _new_xrt_bo(session: KV260FabricSession, group_id: int, nbytes: int) -> Any:
    return session.xrt.bo(session.device, int(nbytes), session._bo_flag, int(group_id))
