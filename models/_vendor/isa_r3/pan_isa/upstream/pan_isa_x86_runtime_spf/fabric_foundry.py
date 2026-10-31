"""PAN fabric foundry control plane.

The foundry is the cold/warm path beside the runtime scheduler. It mines
observed graph regions, proposes hardware candidates, estimates whether they
are worth building, queues artifact-producing work, catalogs built cores, and
feeds scheduler hints back into normal planning.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from pan_hardware_resolver.ledger import write_ledger_jsonl

from .action_ledger import foundry_action_ledger_row, foundry_action_packet
from .fabric_graph import FabricGraph
from .fabric_runtime import UNSAFE_FABRIC_OPS


FOUNDRY_SCHEMA = "pan_fabric_foundry_v1"
REGION_SCHEMA = "pan_fabric_region_v1"
CANDIDATE_SCHEMA = "pan_fabric_candidate_v1"
BUILD_QUEUE_SCHEMA = "pan_fabric_build_queue_entry_v1"
CORE_MANIFEST_SCHEMA = "pan_fabric_core_manifest_v1"
MEASUREMENT_SCHEMA = "pan_fabric_measurement_v1"
SCHEDULER_HINT_SCHEMA = "pan_fabric_scheduler_hint_v1"
CONTROL_DECISION_SCHEMA = "pan_fabric_control_decision_v1"
HEALTH_EVENT_SCHEMA = "pan_fabric_health_event_v1"
QUARANTINE_SCHEMA = "pan_fabric_quarantine_v1"
MANIFEST_STATES = frozenset({"candidate", "queued", "building", "built", "measured", "promoted", "failed", "quarantined"})

DEFAULT_FOUNDRY_ROOT = ".pan_fabric_foundry"

OP_RESOURCE_TABLE: dict[str, dict[str, float]] = {
    "spf32.mul": {"lut": 1800, "dsp": 4, "bram": 2, "uram": 0, "words_s": 68_000_000, "energy_j_per_word": 1.8e-8},
    "spf32.add": {"lut": 900, "dsp": 0, "bram": 1, "uram": 0, "words_s": 95_000_000, "energy_j_per_word": 8.0e-9},
    "spf32.conj": {"lut": 450, "dsp": 0, "bram": 1, "uram": 0, "words_s": 120_000_000, "energy_j_per_word": 4.0e-9},
    "spf32.abs2": {"lut": 1500, "dsp": 2, "bram": 2, "uram": 0, "words_s": 60_000_000, "energy_j_per_word": 1.6e-8},
    "spf32.fft64": {"lut": 6200, "dsp": 16, "bram": 18, "uram": 0, "words_s": 12_000_000, "energy_j_per_word": 8.0e-7},
    "matmul": {"lut": 5200, "dsp": 32, "bram": 12, "uram": 0, "words_s": 28_000_000, "energy_j_per_word": 6.5e-8},
    "einsum": {"lut": 6400, "dsp": 40, "bram": 16, "uram": 0, "words_s": 22_000_000, "energy_j_per_word": 8.5e-8},
    "gemm": {"lut": 5600, "dsp": 32, "bram": 14, "uram": 0, "words_s": 26_000_000, "energy_j_per_word": 7.0e-8},
    "qlinear_matmul": {"lut": 4600, "dsp": 18, "bram": 10, "uram": 0, "words_s": 42_000_000, "energy_j_per_word": 3.8e-8},
    "matmul_integer": {"lut": 3900, "dsp": 14, "bram": 8, "uram": 0, "words_s": 52_000_000, "energy_j_per_word": 2.9e-8},
    "dynamic_quantize_linear": {"lut": 2400, "dsp": 2, "bram": 4, "uram": 0, "words_s": 65_000_000, "energy_j_per_word": 1.8e-8},
    "qkv_projection": {"lut": 9800, "dsp": 64, "bram": 26, "uram": 2, "words_s": 19_000_000, "energy_j_per_word": 1.4e-7},
    "attention_qkv": {"lut": 9800, "dsp": 64, "bram": 26, "uram": 2, "words_s": 19_000_000, "energy_j_per_word": 1.4e-7},
    "attention_score": {"lut": 7200, "dsp": 48, "bram": 18, "uram": 1, "words_s": 17_000_000, "energy_j_per_word": 1.2e-7},
    "attention_value": {"lut": 7600, "dsp": 48, "bram": 18, "uram": 1, "words_s": 17_000_000, "energy_j_per_word": 1.2e-7},
    "fused_attention": {"lut": 14200, "dsp": 96, "bram": 36, "uram": 4, "words_s": 12_000_000, "energy_j_per_word": 2.4e-7},
    "softmax": {"lut": 6800, "dsp": 8, "bram": 16, "uram": 0, "words_s": 11_000_000, "energy_j_per_word": 1.1e-7},
    "rmsnorm": {"lut": 4200, "dsp": 12, "bram": 8, "uram": 0, "words_s": 34_000_000, "energy_j_per_word": 5.0e-8},
    "layernorm": {"lut": 4800, "dsp": 14, "bram": 8, "uram": 0, "words_s": 30_000_000, "energy_j_per_word": 5.8e-8},
    "rotary_embedding": {"lut": 3600, "dsp": 8, "bram": 8, "uram": 0, "words_s": 38_000_000, "energy_j_per_word": 4.2e-8},
    "kv_cache_append": {"lut": 1800, "dsp": 0, "bram": 8, "uram": 1, "words_s": 70_000_000, "energy_j_per_word": 1.2e-8},
    "topk": {"lut": 6200, "dsp": 4, "bram": 14, "uram": 0, "words_s": 8_000_000, "energy_j_per_word": 9.0e-8},
    "top_p_filter": {"lut": 7200, "dsp": 6, "bram": 16, "uram": 0, "words_s": 7_000_000, "energy_j_per_word": 1.0e-7},
    "conv2d": {"lut": 7600, "dsp": 48, "bram": 20, "uram": 0, "words_s": 22_000_000, "energy_j_per_word": 1.1e-7},
    "depthwise_conv2d": {"lut": 4400, "dsp": 12, "bram": 12, "uram": 0, "words_s": 36_000_000, "energy_j_per_word": 5.5e-8},
    "batchnorm": {"lut": 2200, "dsp": 4, "bram": 4, "uram": 0, "words_s": 58_000_000, "energy_j_per_word": 2.4e-8},
    "relu": {"lut": 550, "dsp": 0, "bram": 1, "uram": 0, "words_s": 160_000_000, "energy_j_per_word": 2.0e-9},
    "gelu": {"lut": 4200, "dsp": 6, "bram": 6, "uram": 0, "words_s": 32_000_000, "energy_j_per_word": 4.5e-8},
    "silu": {"lut": 3800, "dsp": 5, "bram": 5, "uram": 0, "words_s": 36_000_000, "energy_j_per_word": 4.0e-8},
    "sigmoid": {"lut": 3200, "dsp": 4, "bram": 5, "uram": 0, "words_s": 42_000_000, "energy_j_per_word": 3.5e-8},
    "tanh": {"lut": 3300, "dsp": 4, "bram": 5, "uram": 0, "words_s": 40_000_000, "energy_j_per_word": 3.8e-8},
    "add": {"lut": 700, "dsp": 0, "bram": 1, "uram": 0, "words_s": 140_000_000, "energy_j_per_word": 3.0e-9},
    "mul": {"lut": 1100, "dsp": 2, "bram": 1, "uram": 0, "words_s": 100_000_000, "energy_j_per_word": 7.0e-9},
    "div": {"lut": 4200, "dsp": 4, "bram": 6, "uram": 0, "words_s": 18_000_000, "energy_j_per_word": 6.5e-8},
    "argmax": {"lut": 3600, "dsp": 0, "bram": 8, "uram": 0, "words_s": 16_000_000, "energy_j_per_word": 4.2e-8},
    "mlp.gemm_k16": {"lut": 1200, "dsp": 8, "bram": 4, "uram": 0, "words_s": 48_000_000, "energy_j_per_word": 2.5e-8},
}

DEFAULT_OP_RESOURCE_ROW = {"lut": 1600, "dsp": 2, "bram": 2, "uram": 0, "words_s": 50_000_000, "energy_j_per_word": 2.0e-8}
ESTIMATED_TRANSFER_BANDWIDTH_BYTES_S = 1_100_000_000.0
ESTIMATED_RECONFIG_WATTS = 2.8


@dataclass(frozen=True)
class FabricRegion:
    region_id: str
    ops: tuple[str, ...]
    elements: int
    occurrence_count: int = 1
    observed_seconds: float = 0.0
    observed_transfer_bytes: int = 0
    source_paths: tuple[str, ...] = ()
    graph: Optional[FabricGraph] = None
    schema: str = REGION_SCHEMA

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FabricRegion":
        graph_payload = payload.get("graph")
        return cls(
            region_id=str(payload["region_id"]),
            ops=tuple(str(op) for op in payload.get("ops") or ()),
            elements=int(payload.get("elements") or 0),
            occurrence_count=int(payload.get("occurrence_count") or 1),
            observed_seconds=float(payload.get("observed_seconds") or 0.0),
            observed_transfer_bytes=int(payload.get("observed_transfer_bytes") or 0),
            source_paths=tuple(str(path) for path in payload.get("source_paths") or ()),
            graph=FabricGraph.from_json(graph_payload) if isinstance(graph_payload, dict) else None,
            schema=str(payload.get("schema", REGION_SCHEMA)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "region_id": self.region_id,
            "ops": list(self.ops),
            "elements": self.elements,
            "occurrence_count": self.occurrence_count,
            "observed_seconds": self.observed_seconds,
            "observed_transfer_bytes": self.observed_transfer_bytes,
            "source_paths": list(self.source_paths),
            "graph": None if self.graph is None else self.graph.to_json(),
        }


@dataclass(frozen=True)
class FabricCandidate:
    candidate_id: str
    region_id: str
    ops: tuple[str, ...]
    elements: int
    target: str = "kv260"
    strategy: str = "fused_stream_pr_core"
    estimate: dict[str, Any] = field(default_factory=dict)
    priority: float = 0.0
    status: str = "candidate"
    artifact_cache_key: str = ""
    artifact_dir: Optional[str] = None
    schema: str = CANDIDATE_SCHEMA

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FabricCandidate":
        return cls(
            candidate_id=str(payload["candidate_id"]),
            region_id=str(payload["region_id"]),
            ops=tuple(str(op) for op in payload.get("ops") or ()),
            elements=int(payload.get("elements") or 0),
            target=str(payload.get("target", "kv260")),
            strategy=str(payload.get("strategy", "fused_stream_pr_core")),
            estimate=dict(payload.get("estimate") or {}),
            priority=float(payload.get("priority") or 0.0),
            status=str(payload.get("status", "candidate")),
            artifact_cache_key=str(payload.get("artifact_cache_key") or ""),
            artifact_dir=None if payload.get("artifact_dir") is None else str(payload["artifact_dir"]),
            schema=str(payload.get("schema", CANDIDATE_SCHEMA)),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "candidate_id": self.candidate_id,
            "region_id": self.region_id,
            "ops": list(self.ops),
            "elements": self.elements,
            "target": self.target,
            "strategy": self.strategy,
            "estimate": dict(self.estimate),
            "priority": self.priority,
            "status": self.status,
            "artifact_cache_key": self.artifact_cache_key,
            "artifact_dir": self.artifact_dir,
        }


@dataclass(frozen=True)
class BuildQueueEntry:
    build_id: str
    candidate_id: str
    status: str
    priority: float
    created_at: float
    artifact_dir: str
    reason: str = "compile_better_for_next_time"
    schema: str = BUILD_QUEUE_SCHEMA

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "build_id": self.build_id,
            "candidate_id": self.candidate_id,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
            "artifact_dir": self.artifact_dir,
            "reason": self.reason,
        }


class FabricFoundryCatalog:
    def __init__(self, root: str | Path = DEFAULT_FOUNDRY_ROOT, *, action_ledger_path: str | Path | None = None):
        self.root = Path(root)
        self.regions_dir = self.root / "regions"
        self.candidates_dir = self.root / "candidates"
        self.manifests_dir = self.root / "core_manifests"
        self.artifacts_dir = self.root / "artifacts"
        self.queue_path = self.root / "build_queue.jsonl"
        self.measurements_path = self.root / "measurements.jsonl"
        self.decisions_path = self.root / "decisions.jsonl"
        self.health_path = self.root / "health.jsonl"
        self.quarantine_path = self.root / "quarantine.jsonl"
        self.action_ledger_path = None if action_ledger_path is None else Path(action_ledger_path)

    def ensure(self) -> None:
        for path in (self.regions_dir, self.candidates_dir, self.manifests_dir, self.artifacts_dir):
            path.mkdir(parents=True, exist_ok=True)

    def put_region(self, region: FabricRegion) -> Path:
        self.ensure()
        path = self.regions_dir / f"{region.region_id}.json"
        _write_json(path, region.to_json())
        self._append_action_ledger_row(
            "mine_graph_region",
            region.to_json(),
            before_status=None,
            after_status="region",
            target="kv260",
            source_path=str(path),
        )
        return path

    def put_candidate(self, candidate: FabricCandidate) -> Path:
        self.ensure()
        path = self.candidates_dir / f"{candidate.candidate_id}.json"
        _write_json(path, candidate.to_json())
        self._append_action_ledger_row(
            "generate_candidate",
            candidate.to_json(),
            before_status="region",
            after_status="candidate",
            target=candidate.target,
            source_path=str(path),
        )
        return path

    def append_queue(self, candidate: FabricCandidate, *, reason: str = "compile_better_for_next_time") -> BuildQueueEntry:
        self.ensure()
        artifact_dir = self.artifacts_dir / candidate.artifact_cache_key
        artifact_dir.mkdir(parents=True, exist_ok=True)
        entry = BuildQueueEntry(
            build_id=f"build_{_short_hash(candidate.candidate_id + ':' + str(time.time_ns()))}",
            candidate_id=candidate.candidate_id,
            status="queued",
            priority=float(candidate.priority),
            created_at=time.time(),
            artifact_dir=str(artifact_dir),
            reason=reason,
        )
        with self.queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_json(), sort_keys=True) + "\n")
        subject = dict(candidate.to_json())
        subject["build"] = entry.to_json()
        self._append_action_ledger_row(
            "queue_build",
            subject,
            before_status="candidate",
            after_status="queued",
            target=candidate.target,
            validation={"queue_reason": reason},
            source_path=str(self.queue_path),
        )
        return entry

    def put_core_manifest(
        self,
        candidate: FabricCandidate,
        *,
        artifact_status: str = "queued",
        artifact_path: Optional[str] = None,
        measurement: Optional[dict[str, Any]] = None,
    ) -> Path:
        self.ensure()
        if artifact_status not in MANIFEST_STATES:
            raise ValueError(f"unsupported core manifest artifact status {artifact_status!r}")
        previous_status = self._manifest_status(candidate.candidate_id)
        payload = {
            "schema": CORE_MANIFEST_SCHEMA,
            "core_id": candidate.candidate_id,
            "candidate_id": candidate.candidate_id,
            "region_id": candidate.region_id,
            "ops": list(candidate.ops),
            "target": candidate.target,
            "strategy": candidate.strategy,
            "input_abi": "pan_buffer_u32",
            "output_abi": "pan_buffer_u32",
            "graph_abi": "pan_fabric_graph_v1",
            "toolchain": {
                "kind": "external_vitis",
                "owned_by_pan": False,
            },
            "allowed_states": sorted(MANIFEST_STATES),
            "artifact": {
                "status": artifact_status,
                "path": artifact_path,
                "cache_key": candidate.artifact_cache_key,
                "cache_dir": str(self.artifacts_dir / candidate.artifact_cache_key),
            },
            "build_worker_contract": {
                "schema": "pan_fabric_external_build_contract_v1",
                "candidate_id": candidate.candidate_id,
                "target": candidate.target,
                "graph_abi": "pan_fabric_graph_v1",
                "input_abi": "pan_buffer_u32",
                "output_abi": "pan_buffer_u32",
                "expected_artifacts": ["xclbin", "manifest", "reports"],
            },
            "estimate": dict(candidate.estimate),
            "measurement": measurement,
        }
        path = self.manifests_dir / f"{candidate.candidate_id}.json"
        _write_json(path, payload)
        self._append_core_manifest_action(
            candidate,
            artifact_status=artifact_status,
            previous_status=previous_status,
            artifact_path=artifact_path,
            measurement=measurement,
            source_path=str(path),
        )
        return path

    def append_measurement(self, measurement: dict[str, Any]) -> None:
        self.ensure()
        _append_jsonl(self.measurements_path, measurement)

    def append_decision(self, decision: dict[str, Any]) -> None:
        self.ensure()
        _append_jsonl(self.decisions_path, decision)

    def append_health_event(self, event: dict[str, Any]) -> None:
        self.ensure()
        _append_jsonl(self.health_path, event)

    def append_quarantine(self, record: dict[str, Any]) -> None:
        self.ensure()
        _append_jsonl(self.quarantine_path, record)
        self._append_action_ledger_row(
            "quarantine_artifact",
            dict(record),
            before_status=str(record.get("before_status") or record.get("artifact_status") or "failed"),
            after_status="quarantined",
            target=str(record.get("target") or "kv260"),
            validation={"reason": record.get("reason"), "severity": record.get("severity")},
            source_path=str(self.quarantine_path),
        )

    def load_regions(self) -> list[FabricRegion]:
        if not self.regions_dir.is_dir():
            return []
        return [FabricRegion.from_json(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.regions_dir.glob("*.json"))]

    def load_candidates(self) -> list[FabricCandidate]:
        if not self.candidates_dir.is_dir():
            return []
        return [FabricCandidate.from_json(json.loads(path.read_text(encoding="utf-8"))) for path in sorted(self.candidates_dir.glob("*.json"))]

    def load_measurements(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.measurements_path)

    def load_decisions(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.decisions_path)

    def load_health_events(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.health_path)

    def load_quarantines(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.quarantine_path)

    def load_action_ledger_rows(self) -> list[dict[str, Any]]:
        if self.action_ledger_path is None:
            return []
        return _read_jsonl(self.action_ledger_path)

    def active_quarantines_for_graph(self, graph: FabricGraph, *, target: str = "kv260") -> list[dict[str, Any]]:
        ops = tuple(op.op for op in graph.ops)
        elements = int(graph.ops[0].elements) if graph.ops else 0
        now = time.time()
        matches = []
        for record in self.load_quarantines():
            if str(record.get("status") or "active") != "active":
                continue
            expires_at = record.get("expires_at")
            if expires_at is not None and float(expires_at) <= now:
                continue
            if str(record.get("target") or target) != target:
                continue
            rec_ops = tuple(str(op) for op in (record.get("ops") or ()))
            rec_elements = int(record.get("elements") or 0)
            if rec_ops == ops and (rec_elements == 0 or rec_elements == elements):
                matches.append(dict(record))
        return matches

    def artifact_xclbin_dirs(self) -> list[str]:
        dirs: list[str] = []
        for manifest_path in sorted(self.manifests_dir.glob("*.json")) if self.manifests_dir.is_dir() else []:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            artifact = dict(manifest.get("artifact") or {})
            status = str(artifact.get("status") or "")
            if status not in {"measured", "promoted"}:
                continue
            artifact_path = artifact.get("path")
            if artifact_path:
                dirs.append(str(Path(str(artifact_path)).parent))
            cache_dir = artifact.get("cache_dir")
            if cache_dir:
                dirs.append(str(cache_dir))
        return sorted(dict.fromkeys(dirs))

    def _manifest_status(self, candidate_id: str) -> Optional[str]:
        path = self.manifests_dir / f"{candidate_id}.json"
        if not path.is_file():
            return None
        payload = json.loads(path.read_text(encoding="utf-8"))
        return str((payload.get("artifact") or {}).get("status") or "")

    def _append_core_manifest_action(
        self,
        candidate: FabricCandidate,
        *,
        artifact_status: str,
        previous_status: Optional[str],
        artifact_path: Optional[str],
        measurement: Optional[dict[str, Any]],
        source_path: str,
    ) -> None:
        action_name, before_status, after_status = _manifest_action_transition(artifact_status, previous_status)
        if action_name is None:
            return
        subject = dict(candidate.to_json())
        subject["artifact"] = {
            "status": artifact_status,
            "path": artifact_path,
            "cache_key": candidate.artifact_cache_key,
            "previous_status": previous_status,
        }
        if measurement is not None:
            subject["measurement"] = dict(measurement)
        self._append_action_ledger_row(
            action_name,
            subject,
            before_status=before_status,
            after_status=after_status,
            target=candidate.target,
            validation=_manifest_action_validation(artifact_status, measurement),
            measurement={} if measurement is None else dict(measurement),
            source_path=source_path,
        )

    def _append_action_ledger_row(
        self,
        selected_action: str,
        subject: dict[str, Any],
        *,
        before_status: str | None,
        after_status: str | None,
        target: str,
        validation: Optional[dict[str, Any]] = None,
        measurement: Optional[dict[str, Any]] = None,
        source_path: str,
    ) -> None:
        if self.action_ledger_path is None:
            return
        action = foundry_action_packet(
            selected_action,
            subject,
            before_status=before_status,
            after_status=after_status,
            target=target,
            validation=validation,
        )
        row = foundry_action_ledger_row(
            {
                "catalog_root": str(self.root),
                "source_path": source_path,
                "subject_id": _foundry_catalog_subject_id(subject),
                "before_status": before_status,
            },
            {
                "kind": selected_action,
                "subject_id": _foundry_catalog_subject_id(subject),
                "after_status": after_status,
            },
            {"target": target},
            action,
            measurement=measurement or {},
            lineage={
                "producer": "pan_isa_x86_runtime_spf.fabric_foundry.FabricFoundryCatalog",
                "catalog_root": str(self.root),
                "source_path": source_path,
            },
        )
        write_ledger_jsonl([row], self.action_ledger_path)


def mine_regions(
    paths: Sequence[str | Path],
    *,
    max_ops: int = 6,
    min_count: int = 1,
) -> list[FabricRegion]:
    buckets: dict[tuple[tuple[str, ...], int], dict[str, Any]] = {}
    for path in paths:
        for graph, observed_seconds, transfer_bytes in _iter_graph_observations(path):
            ops = tuple(op.op for op in graph.ops)
            if not ops:
                continue
            elements = int(graph.ops[0].elements)
            for start in range(len(ops)):
                for end in range(start + 1, min(len(ops), start + int(max_ops)) + 1):
                    window = ops[start:end]
                    key = (window, elements)
                    row = buckets.setdefault(
                        key,
                        {
                            "count": 0,
                            "seconds": 0.0,
                            "transfer": 0,
                            "sources": set(),
                            "graph": _subgraph_from_ops(graph, start, end),
                        },
                    )
                    row["count"] += 1
                    row["seconds"] += float(observed_seconds)
                    row["transfer"] += int(transfer_bytes)
                    row["sources"].add(str(path))
    regions = []
    for (ops, elements), row in buckets.items():
        if int(row["count"]) < int(min_count):
            continue
        region_id = "region_" + _short_hash(json.dumps({"ops": ops, "elements": elements}, sort_keys=True))
        regions.append(
            FabricRegion(
                region_id=region_id,
                ops=ops,
                elements=elements,
                occurrence_count=int(row["count"]),
                observed_seconds=float(row["seconds"]),
                observed_transfer_bytes=int(row["transfer"]),
                source_paths=tuple(sorted(row["sources"])),
                graph=row["graph"],
            )
        )
    regions.sort(key=lambda item: (-item.occurrence_count, item.ops, item.elements))
    return regions


def generate_candidates(
    regions: Sequence[FabricRegion],
    *,
    target: str = "kv260",
) -> list[FabricCandidate]:
    candidates: list[FabricCandidate] = []
    for region in regions:
        if _contains_unsafe_op(region.ops):
            continue
        strategy = "single_pr_core" if len(region.ops) == 1 else "fused_stream_pr_core"
        estimate = estimate_candidate(region, target=target, strategy=strategy)
        candidate_seed = {
            "region_id": region.region_id,
            "ops": region.ops,
            "elements": region.elements,
            "target": target,
            "strategy": strategy,
            "graph_abi": "pan_fabric_graph_v1",
            "buffer_abi": "pan_buffer_u32",
            "toolchain": "external_vitis",
        }
        candidate_id = "core_" + _short_hash(json.dumps(candidate_seed, sort_keys=True))
        artifact_cache_key = _short_hash(
            json.dumps(
                {
                    "candidate": candidate_seed,
                    "estimate": estimate,
                    "kernel_strategy": strategy,
                    "toolchain": "external_vitis",
                },
                sort_keys=True,
            )
        )
        candidates.append(
            FabricCandidate(
                candidate_id=candidate_id,
                region_id=region.region_id,
                ops=region.ops,
                elements=region.elements,
                target=target,
                strategy=strategy,
                estimate=estimate,
                priority=float(estimate["benefit_score"]),
                artifact_cache_key=artifact_cache_key,
                artifact_dir=str(Path("artifacts") / artifact_cache_key),
            )
        )
    candidates.sort(key=lambda item: (-item.priority, item.candidate_id))
    return candidates


def estimate_candidate(region: FabricRegion, *, target: str = "kv260", strategy: str = "fused_stream_pr_core") -> dict[str, Any]:
    if _contains_unsafe_op(region.ops):
        return {
            "target": target,
            "strategy": strategy,
            "unsafe": True,
            "unsafe_ops": sorted(set(region.ops) & set(UNSAFE_FABRIC_OPS)),
            "benefit_score": -1.0e9,
        }
    lut = 1400.0
    dsp = 0.0
    bram = 4.0
    uram = 0.0
    throughput_words_s = 180_000_000.0
    energy_j_per_word = 0.0
    op_resource_rows: list[dict[str, Any]] = []
    for op in region.ops:
        row = OP_RESOURCE_TABLE.get(op, DEFAULT_OP_RESOURCE_ROW)
        op_resource_rows.append({"op": op, **{key: row[key] for key in ("lut", "dsp", "bram", "uram", "words_s", "energy_j_per_word")}})
        lut += float(row["lut"])
        dsp += float(row["dsp"])
        bram += float(row["bram"])
        uram += float(row["uram"])
        throughput_words_s = min(throughput_words_s, float(row["words_s"]))
        energy_j_per_word += float(row["energy_j_per_word"])
    if len(region.ops) > 1:
        lut *= 0.88
        bram += len(region.ops) * 2
        throughput_words_s *= 1.15
        energy_j_per_word *= 0.82
    dispatch_seconds = 6.0e-5
    compute_seconds = float(region.elements) / throughput_words_s
    latency_s = dispatch_seconds + compute_seconds
    energy_j = energy_j_per_word * float(region.elements)
    observed = float(region.observed_seconds or 0.0)
    ps_baseline_s = observed if observed > 0 else sum(_ps_baseline_seconds(op, region.elements) for op in region.ops)
    speedup = ps_baseline_s / latency_s if latency_s > 0 else 0.0
    compile_minutes = 18.0 + 4.5 * len(region.ops) + 0.002 * (lut / 100.0)
    reconfig_ms = 8.0 + 0.012 * (lut / 100.0)
    reconfig_seconds = reconfig_ms / 1000.0
    estimated_intermediate_transfer_bytes = max(0, len(region.ops) - 1) * max(0, int(region.elements)) * 4
    observed_transfer_bytes = max(0, int(region.observed_transfer_bytes))
    transfer_avoidance_bytes = observed_transfer_bytes if observed_transfer_bytes > 0 else estimated_intermediate_transfer_bytes
    transfer_avoidance_seconds = transfer_avoidance_bytes / ESTIMATED_TRANSFER_BANDWIDTH_BYTES_S
    amortized_compile_penalty_s = compile_minutes * 60.0 / max(1, region.occurrence_count)
    benefit_score = (
        (ps_baseline_s + transfer_avoidance_seconds - latency_s) * max(1, region.occurrence_count)
        - (0.002 * compile_minutes)
        - reconfig_seconds
    )
    resource_estimate = {
        "lut": int(round(lut)),
        "dsp": int(round(dsp)),
        "bram": int(round(bram)),
        "uram": int(round(uram)),
        "op_resource_rows": op_resource_rows,
    }
    latency_estimate = {
        "dispatch_seconds": dispatch_seconds,
        "compute_seconds": compute_seconds,
        "seconds": latency_s,
        "throughput_words_s": throughput_words_s,
        "baseline_seconds": ps_baseline_s,
        "speedup_vs_observed_or_ps": speedup,
    }
    transfer_avoidance = {
        "observed_transfer_bytes": observed_transfer_bytes,
        "estimated_intermediate_bytes": estimated_intermediate_transfer_bytes,
        "avoided_bytes": transfer_avoidance_bytes,
        "seconds": transfer_avoidance_seconds,
        "bandwidth_bytes_s": ESTIMATED_TRANSFER_BANDWIDTH_BYTES_S,
    }
    reconfiguration = {
        "milliseconds": reconfig_ms,
        "seconds": reconfig_seconds,
        "energy_j": reconfig_seconds * ESTIMATED_RECONFIG_WATTS,
    }
    compile_estimate = {
        "minutes": compile_minutes,
        "amortized_penalty_seconds": amortized_compile_penalty_s,
    }
    energy_estimate = {
        "joules": energy_j,
        "joules_per_word": energy_j_per_word,
        "reconfiguration_joules": reconfiguration["energy_j"],
    }
    return {
        "target": target,
        "strategy": strategy,
        "estimate_contract": "pan_foundry_estimate_v2",
        "latency_s": latency_s,
        "energy_j": energy_j,
        "lut": resource_estimate["lut"],
        "dsp": resource_estimate["dsp"],
        "bram": resource_estimate["bram"],
        "uram": resource_estimate["uram"],
        "compile_minutes": compile_minutes,
        "reconfig_ms": reconfig_ms,
        "throughput_words_s": throughput_words_s,
        "speedup_vs_observed_or_ps": speedup,
        "amortized_compile_penalty_s": amortized_compile_penalty_s,
        "transfer_avoidance_bytes": transfer_avoidance_bytes,
        "transfer_avoidance_s": transfer_avoidance_seconds,
        "resource_estimate": resource_estimate,
        "latency_estimate": latency_estimate,
        "energy_estimate": energy_estimate,
        "transfer_avoidance": transfer_avoidance,
        "reconfiguration": reconfiguration,
        "compile_estimate": compile_estimate,
        "op_resource_rows": op_resource_rows,
        "benefit_score": benefit_score,
    }


def ingest_measurements(
    paths: Sequence[str | Path],
    *,
    catalog: FabricFoundryCatalog,
    candidate_id: Optional[str] = None,
    artifact_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    measurements = []
    candidates = {candidate.candidate_id: candidate for candidate in catalog.load_candidates()}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        measurement = _measurement_from_payload(payload, source_path=str(path), candidate_id=candidate_id)
        measurements.append(measurement)
        catalog.append_measurement(measurement)
        cid = measurement.get("candidate_id")
        if cid and cid in candidates:
            catalog.put_core_manifest(
                candidates[cid],
                artifact_status="measured" if measurement.get("ok") else "failed",
                artifact_path=artifact_path,
                measurement=measurement,
            )
    return measurements


def record_control_decision(
    catalog: FabricFoundryCatalog,
    graph: FabricGraph,
    *,
    payload: dict[str, Any],
    hint: Optional[dict[str, Any]] = None,
    target: str = "kv260",
    source_path: str = "fabric-exec",
) -> dict[str, Any]:
    """Persist the runtime decision, health event, and quarantine if needed."""

    ops = tuple(op.op for op in graph.ops)
    elements = int(graph.ops[0].elements) if graph.ops else 0
    summary = dict(payload.get("summary") or {})
    hint_payload = dict(hint or payload.get("foundry_hint") or {})
    action = str(hint_payload.get("selected_action") or hint_payload.get("action") or _placement_action_from_payload(payload))
    decision = {
        "schema": CONTROL_DECISION_SCHEMA,
        "decision_id": "decision_" + _short_hash(json.dumps({
            "ops": ops,
            "elements": elements,
            "source": source_path,
            "time": time.time_ns(),
        }, sort_keys=True)),
        "created_at": time.time(),
        "source_path": source_path,
        "target": target,
        "graph_ops": list(ops),
        "elements": elements,
        "selected_action": action,
        "execution_backend": payload.get("execution_backend"),
        "ok": bool(payload.get("ok")),
        "scheduler_hint": hint_payload or None,
        "transfer_buckets": {
            "setup": int(summary.get("setup_transfer_bytes") or 0),
            "timed_graph_execution": int(summary.get("timed_transfer_bytes") or 0),
            "forced_materialization": int(summary.get("final_transfer_bytes") or 0),
            "observed_total": int(summary.get("observed_transfer_bytes") or 0),
        },
        "timing": {
            "elapsed_seconds": float(payload.get("elapsed_seconds") or 0.0),
            "timed_elapsed_seconds": float(summary.get("timed_elapsed_seconds") or 0.0),
            "timed_daemon_elapsed_seconds": float(summary.get("timed_daemon_elapsed_seconds") or 0.0),
        },
        "value_store": _value_store_summary(payload),
        "validation": _validation_summary(payload),
        "error": payload.get("error"),
    }
    catalog.append_decision(decision)

    health = {
        "schema": HEALTH_EVENT_SCHEMA,
        "event_id": "health_" + _short_hash(decision["decision_id"] + ":health"),
        "created_at": time.time(),
        "source_path": source_path,
        "target": target,
        "ops": list(ops),
        "elements": elements,
        "status": "healthy" if _payload_health_ok(payload) else "unhealthy",
        "reason": "execution_ok" if _payload_health_ok(payload) else _health_failure_reason(payload),
        "execution_backend": payload.get("execution_backend"),
        "decision_id": decision["decision_id"],
        "details": {
            "error": payload.get("error"),
            "summary": summary,
            "validation": _validation_summary(payload),
        },
    }
    catalog.append_health_event(health)

    quarantine = None
    if health["status"] == "unhealthy":
        quarantine = {
            "schema": QUARANTINE_SCHEMA,
            "quarantine_id": "quarantine_" + _short_hash(decision["decision_id"] + ":quarantine"),
            "created_at": time.time(),
            "status": "active",
            "target": target,
            "ops": list(ops),
            "elements": elements,
            "reason": health["reason"],
            "severity": _quarantine_severity(payload),
            "source_path": source_path,
            "decision_id": decision["decision_id"],
            "execution_backend": payload.get("execution_backend"),
            "artifact_id": (hint_payload.get("candidate") or {}).get("candidate_id") if isinstance(hint_payload.get("candidate"), dict) else None,
            "details": health["details"],
        }
        catalog.append_quarantine(quarantine)
    return {
        "schema": "pan_fabric_control_record_v1",
        "decision": decision,
        "health": health,
        "quarantine": quarantine,
    }


def scheduler_hint_for_graph(
    graph: FabricGraph,
    *,
    catalog: FabricFoundryCatalog,
    target: str = "kv260",
) -> dict[str, Any]:
    ops = tuple(op.op for op in graph.ops)
    elements = int(graph.ops[0].elements) if graph.ops else 0
    active_quarantines = catalog.active_quarantines_for_graph(graph, target=target)
    if active_quarantines:
        return _scheduler_hint(
            {
            "schema": SCHEDULER_HINT_SCHEMA,
            "target": target,
            "graph_ops": list(ops),
            "elements": elements,
            "action": "reject_fail_closed",
            "use_existing_now": False,
            "compile_better_for_next_time": False,
            "candidate": None,
            "reason": "active_quarantine",
            "quarantine": active_quarantines[0],
            "quarantine_count": len(active_quarantines),
            },
            selected_action="reject_fail_closed",
        )
    if _contains_unsafe_op(ops):
        return _scheduler_hint(
            {
            "schema": SCHEDULER_HINT_SCHEMA,
            "target": target,
            "graph_ops": list(ops),
            "elements": elements,
            "action": "stay_on_existing_runtime",
            "use_existing_now": True,
            "compile_better_for_next_time": False,
            "candidate": None,
            "reason": "unsafe_op_quarantined",
            "unsafe_ops": sorted(set(ops) & set(UNSAFE_FABRIC_OPS)),
            },
            selected_action="run_ps_now",
        )
    candidates = catalog.load_candidates()
    compatible = [
        candidate
        for candidate in candidates
        if candidate.target == target and candidate.ops == ops and candidate.elements == elements
    ]
    built = [
        candidate
        for candidate in compatible
        if _manifest_status(catalog, candidate.candidate_id) in {"measured", "promoted"}
    ]
    if built:
        selected = max(built, key=lambda item: item.priority)
        return _scheduler_hint(
            {
            "schema": SCHEDULER_HINT_SCHEMA,
            "target": target,
            "graph_ops": list(ops),
            "elements": elements,
            "action": "use_catalog_core",
            "use_existing_now": False,
            "compile_better_for_next_time": False,
            "candidate": selected.to_json(),
            "reason": "built_core_available",
            },
            selected_action="run_catalog_core_now",
        )
    resident_measurement = _best_resident_measurement(
        catalog,
        ops=ops,
        elements=elements,
        target=target,
        compatible_candidates=compatible,
    )
    if resident_measurement is not None:
        selected = max(compatible, key=lambda item: item.priority) if compatible else None
        if selected is None:
            region_id = "region_" + _short_hash(json.dumps({"ops": ops, "elements": elements}, sort_keys=True))
            generated = generate_candidates(
                [FabricRegion(region_id=region_id, ops=ops, elements=elements, graph=graph)],
                target=target,
            )
            selected = generated[0] if generated else None
        return _scheduler_hint(
            {
            "schema": SCHEDULER_HINT_SCHEMA,
            "target": target,
            "graph_ops": list(ops),
            "elements": elements,
            "action": "use_resident_daemon_session_compile_better_for_next_time",
            "use_existing_now": True,
            "existing_runtime": {
                "backend": "daemon_session_resident",
                "measurement": resident_measurement,
            },
            "compile_better_for_next_time": bool(selected and selected.priority > 0.0),
            "candidate": None if selected is None else selected.to_json(),
            "reason": "resident_daemon_measurement_available",
            },
            selected_action="run_daemon_resident_now",
            extra_actions=("keep_resident", "compile_better_for_next_time") if selected and selected.priority > 0.0 else ("keep_resident",),
        )
    if compatible:
        selected = max(compatible, key=lambda item: item.priority)
        return _scheduler_hint(
            {
            "schema": SCHEDULER_HINT_SCHEMA,
            "target": target,
            "graph_ops": list(ops),
            "elements": elements,
            "action": "use_existing_now_compile_better_for_next_time",
            "use_existing_now": True,
            "compile_better_for_next_time": selected.priority > 0.0,
            "candidate": selected.to_json(),
            "reason": "candidate_available_not_built",
            },
            selected_action="run_ps_now",
            extra_actions=("compile_better_for_next_time",) if selected.priority > 0.0 else (),
        )
    region_id = "region_" + _short_hash(json.dumps({"ops": ops, "elements": elements}, sort_keys=True))
    region = FabricRegion(region_id=region_id, ops=ops, elements=elements, graph=graph)
    generated = generate_candidates([region], target=target)
    candidate = generated[0] if generated else None
    return _scheduler_hint(
        {
        "schema": SCHEDULER_HINT_SCHEMA,
        "target": target,
        "graph_ops": list(ops),
        "elements": elements,
        "action": "use_existing_now_compile_better_for_next_time",
        "use_existing_now": True,
        "compile_better_for_next_time": bool(candidate and candidate.priority > 0.0),
        "candidate": None if candidate is None else candidate.to_json(),
        "reason": "no_catalog_core_for_graph",
        },
        selected_action="run_ps_now",
        extra_actions=("compile_better_for_next_time",) if candidate and candidate.priority > 0.0 else (),
    )


def _scheduler_hint(
    payload: dict[str, Any],
    *,
    selected_action: str,
    extra_actions: Sequence[str] = (),
) -> dict[str, Any]:
    actions = [selected_action]
    for action in extra_actions:
        if action not in actions:
            actions.append(action)
    payload["selected_action"] = selected_action
    payload["action_plan"] = actions
    return payload


def catalog_summary(catalog: FabricFoundryCatalog) -> dict[str, Any]:
    regions = catalog.load_regions()
    candidates = catalog.load_candidates()
    queue_count = 0
    if catalog.queue_path.is_file():
        queue_count = sum(1 for line in catalog.queue_path.read_text(encoding="utf-8").splitlines() if line.strip())
    measurements = catalog.load_measurements()
    decisions = catalog.load_decisions()
    health_events = catalog.load_health_events()
    quarantines = catalog.load_quarantines()
    action_ledger_rows = catalog.load_action_ledger_rows()
    return {
        "schema": FOUNDRY_SCHEMA,
        "catalog_root": str(catalog.root),
        "region_count": len(regions),
        "candidate_count": len(candidates),
        "queue_count": queue_count,
        "measurement_count": len(measurements),
        "decision_count": len(decisions),
        "health_event_count": len(health_events),
        "quarantine_count": len(quarantines),
        "active_quarantine_count": sum(1 for record in quarantines if str(record.get("status") or "active") == "active"),
        "action_ledger_path": None if catalog.action_ledger_path is None else str(catalog.action_ledger_path),
        "action_ledger_row_count": len(action_ledger_rows),
        "artifact_xclbin_dirs": catalog.artifact_xclbin_dirs(),
        "regions": [region.to_json() for region in regions],
        "candidates": [candidate.to_json() for candidate in candidates],
        "measurements": measurements,
        "decisions": decisions,
        "health_events": health_events,
        "quarantines": quarantines,
    }


def _manifest_action_transition(artifact_status: str, previous_status: Optional[str]) -> tuple[str | None, str | None, str | None]:
    if artifact_status == "built":
        return "ingest_built_artifact", previous_status if previous_status in {"queued", "building"} else "queued", "built"
    if artifact_status in {"measured", "failed"}:
        return "measure_artifact", previous_status if previous_status == "built" else "built", artifact_status
    if artifact_status == "promoted":
        return "promote_artifact", previous_status if previous_status == "measured" else "measured", "promoted"
    if artifact_status == "quarantined":
        before = previous_status if previous_status in {"candidate", "queued", "building", "built", "measured", "failed", "promoted"} else "failed"
        return "quarantine_artifact", before, "quarantined"
    return None, None, None


def _manifest_action_validation(artifact_status: str, measurement: Optional[dict[str, Any]]) -> dict[str, Any]:
    payload = dict(measurement or {})
    measurement_ok = bool(payload.get("ok")) if payload else False
    validation_failures = int(payload.get("validation_failures") or 0)
    if artifact_status == "failed":
        measurement_ok = False
    if artifact_status in {"measured", "promoted"} and payload and "validation_failures" not in payload:
        validation_failures = 0
    return {
        "measurement_ok": measurement_ok,
        "validation_failures": validation_failures,
        "artifact_status": artifact_status,
    }


def _best_resident_measurement(
    catalog: FabricFoundryCatalog,
    *,
    ops: tuple[str, ...],
    elements: int,
    target: str,
    compatible_candidates: Sequence[FabricCandidate],
) -> Optional[dict[str, Any]]:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in catalog.load_candidates()}
    compatible_ids = {candidate.candidate_id for candidate in compatible_candidates}
    matches = []
    for measurement in catalog.load_measurements():
        if not measurement.get("ok"):
            continue
        backend = str(measurement.get("execution_backend") or "")
        if backend != "daemon_session_resident" and not bool(measurement.get("resident")):
            continue
        if not _measurement_matches_region(
            measurement,
            ops=ops,
            elements=elements,
            target=target,
            candidate_by_id=candidate_by_id,
            compatible_ids=compatible_ids,
        ):
            continue
        matches.append(dict(measurement))
    if not matches:
        return None
    matches.sort(
        key=lambda item: (
            float(item.get("timed_mwords_per_s") or 0.0),
            -float(item.get("elapsed_seconds") or 0.0),
        ),
        reverse=True,
    )
    return matches[0]


def _measurement_matches_region(
    measurement: dict[str, Any],
    *,
    ops: tuple[str, ...],
    elements: int,
    target: str,
    candidate_by_id: dict[str, FabricCandidate],
    compatible_ids: set[str],
) -> bool:
    measured_ops = measurement.get("ops")
    measured_elements = measurement.get("elements")
    measured_target = measurement.get("target")
    if measured_ops is not None or measured_elements is not None:
        return (
            tuple(str(op) for op in (measured_ops or ())) == ops
            and int(measured_elements or 0) == int(elements)
            and (measured_target is None or str(measured_target) == target)
        )
    candidate_id = measurement.get("candidate_id")
    if candidate_id in compatible_ids:
        return True
    candidate = candidate_by_id.get(str(candidate_id)) if candidate_id else None
    return bool(
        candidate
        and candidate.target == target
        and candidate.ops == ops
        and int(candidate.elements) == int(elements)
    )


def _iter_graph_observations(path: str | Path) -> Iterable[tuple[FabricGraph, float, int]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        rows_by_repeat: dict[int, list[dict[str, Any]]] = {}
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not _is_fabric_row(row):
                    continue
                if isinstance(row.get("graph"), dict):
                    yield (
                        FabricGraph.from_json(row["graph"]),
                        float(row.get("elapsed_seconds") or row.get("observed_seconds") or 0.0),
                        int(row.get("observed_transfer_bytes") or 0),
                    )
                    continue
                rows_by_repeat.setdefault(int(row.get("repeat_index") or 0), []).append(row)
        for rows in rows_by_repeat.values():
            rows.sort(key=lambda item: int(item.get("step_index") or 0))
            graph = _graph_from_rows(rows)
            if graph.ops:
                yield graph, sum(float(row.get("elapsed_seconds") or row.get("observed_seconds") or 0.0) for row in rows), sum(int(row.get("observed_transfer_bytes") or 0) for row in rows)
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") == "pan_fabric_execution_v1" and isinstance(payload.get("graph"), dict):
        graph = FabricGraph.from_json(payload["graph"])
        summary = dict(payload.get("summary") or {})
        seconds = float(summary.get("observed_run_daemon_seconds") or summary.get("observed_run_pl_seconds") or payload.get("elapsed_seconds") or 0.0)
        transfer = int(summary.get("observed_transfer_bytes") or 0)
        yield graph, seconds, transfer
    elif isinstance(payload.get("graph"), dict):
        yield FabricGraph.from_json(payload["graph"]), 0.0, 0


def _graph_from_rows(rows: Sequence[dict[str, Any]]) -> FabricGraph:
    ops = []
    buffers = {}
    previous = "input0"
    elements = 0
    side_index = 1
    for index, row in enumerate(rows):
        op = str(row.get("op") or "")
        if not op.startswith("spf"):
            continue
        row_elements = _infer_elements(row)
        elements = elements or row_elements
        arity = 2 if op in {"spf32.mul", "spf32.add", "spf16.mul", "spf16.div"} else 1
        inputs = [previous]
        if arity == 2:
            side = f"input{side_index}"
            side_index += 1
            inputs.append(side)
            buffers.setdefault(side, {"elements": row_elements, "residency": "ps"})
        output = f"tmp{index}"
        buffers.setdefault(previous, {"elements": row_elements, "residency": "ps"})
        buffers[output] = {"elements": row_elements, "residency": "unknown"}
        ops.append({"id": f"mined{index}", "op": op, "inputs": inputs, "output": output, "elements": row_elements})
        previous = output
    if not ops:
        return FabricGraph.from_json({"buffers": {}, "ops": [], "outputs": []})
    return FabricGraph.from_json({"buffers": buffers, "ops": ops, "outputs": [previous], "output_residency": "ps"})


def _subgraph_from_ops(graph: FabricGraph, start: int, end: int) -> FabricGraph:
    ops = graph.ops[start:end]
    if not ops:
        return FabricGraph.from_json({"buffers": {}, "ops": [], "outputs": []})
    elements = int(ops[0].elements)
    return FabricGraph.linear([op.op for op in ops], elements=elements, input_residency="ps", output_residency="ps")


def _measurement_from_payload(payload: dict[str, Any], *, source_path: str, candidate_id: Optional[str]) -> dict[str, Any]:
    if payload.get("schema") == "pan_fabric_daemon_chain_bench_v1":
        elements = int(payload.get("elements") or 0)
        ops = tuple(str(op) for op in payload.get("ops") or ("spf32.mul",))
        return {
            "schema": MEASUREMENT_SCHEMA,
            "candidate_id": candidate_id,
            "source_path": source_path,
            "ok": bool(payload.get("ok", True)),
            "execution_backend": "daemon_session_resident",
            "target": str(payload.get("target", "kv260")),
            "ops": list(ops),
            "elements": elements,
            "elapsed_seconds": float(payload.get("timed_elapsed_seconds") or 0.0),
            "observed_run_daemon_seconds": float(payload.get("timed_daemon_elapsed_seconds") or 0.0),
            "observed_run_pl_seconds": 0.0,
            "observed_transfer_bytes": int(payload.get("timed_transfer_bytes") or 0),
            "output_sha256": payload.get("final_sha256"),
            "timed_mwords_per_s": float(payload.get("timed_mwords_per_s") or 0.0),
            "timed_daemon_mwords_per_s": float(payload.get("timed_daemon_mwords_per_s") or 0.0),
            "resident": True,
            "timed_graph_packets": int(payload.get("timed_graph_packets") or 0),
            "ingested_at": time.time(),
        }
    summary = dict(payload.get("summary") or {})
    outputs = dict(payload.get("outputs") or {})
    first_output = next(iter(outputs.values()), {}) if outputs else {}
    graph_payload = payload.get("graph")
    graph = FabricGraph.from_json(graph_payload) if isinstance(graph_payload, dict) else None
    ops = tuple(op.op for op in graph.ops) if graph is not None else ()
    elements = int(graph.ops[0].elements) if graph is not None and graph.ops else None
    foundry_hint = dict(payload.get("foundry_hint") or {})
    target = str(foundry_hint.get("target") or payload.get("target") or "kv260")
    timed_packets = int(summary.get("timed_graph_packets") or payload.get("repeats") or 0)
    timed_seconds = float(summary.get("timed_elapsed_seconds") or payload.get("elapsed_seconds") or 0.0)
    daemon_seconds = float(summary.get("timed_daemon_elapsed_seconds") or 0.0)
    words = int(elements or 0) * timed_packets
    return {
        "schema": MEASUREMENT_SCHEMA,
        "candidate_id": candidate_id,
        "source_path": source_path,
        "ok": bool(payload.get("ok", True)),
        "execution_backend": payload.get("execution_backend"),
        "target": target,
        "ops": list(ops),
        "elements": elements,
        "elapsed_seconds": float(payload.get("elapsed_seconds") or summary.get("observed_run_daemon_seconds") or summary.get("observed_run_pl_seconds") or 0.0),
        "observed_run_daemon_seconds": float(summary.get("observed_run_daemon_seconds") or 0.0),
        "observed_run_pl_seconds": float(summary.get("observed_run_pl_seconds") or 0.0),
        "observed_transfer_bytes": int(summary.get("observed_transfer_bytes") or 0),
        "output_sha256": first_output.get("sha256"),
        "timed_mwords_per_s": (words / timed_seconds / 1e6) if timed_seconds and words else 0.0,
        "timed_daemon_mwords_per_s": (words / daemon_seconds / 1e6) if daemon_seconds and words else 0.0,
        "resident": str(payload.get("execution_backend") or "") == "fabric-session",
        "timed_graph_packets": timed_packets,
        "ingested_at": time.time(),
    }


def _manifest_status(catalog: FabricFoundryCatalog, candidate_id: str) -> Optional[str]:
    path = catalog.manifests_dir / f"{candidate_id}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str((payload.get("artifact") or {}).get("status") or "")


def _is_fabric_row(row: dict[str, Any]) -> bool:
    if row.get("schema") in {"pan_fabric_runtime_trace_v1", "pan_fabric_runtime_training_record_v1"}:
        return isinstance(row.get("graph"), dict)
    return row.get("schema") in {"pan_fabric_execution_v1", "pan_fabric_training_record_v1"} and str(row.get("op") or "").startswith("spf")


def _infer_elements(row: dict[str, Any]) -> int:
    if row.get("elements"):
        return int(row["elements"])
    transfer = int(row.get("observed_transfer_bytes") or row.get("planned_transfer_bytes") or 0)
    if transfer > 0:
        return max(1, transfer // 4)
    return 4096


def _ps_baseline_seconds(op: str, elements: int) -> float:
    table = {
        "spf32.mul": 1.0 / 29_544_624.27017326,
        "spf32.add": 1.0 / 35_000_000.0,
        "spf32.conj": 1.0 / 50_000_000.0,
        "spf32.abs2": 1.0 / 24_000_000.0,
        "spf32.fft64": 1.2e-5,
    }
    return float(elements) * table.get(op, 8.0e-8)


def _contains_unsafe_op(ops: Sequence[str]) -> bool:
    return any(op in UNSAFE_FABRIC_OPS for op in ops)


def _foundry_catalog_subject_id(subject: dict[str, Any]) -> Optional[str]:
    for key in ("candidate_id", "region_id", "build_id", "artifact_id", "quarantine_id"):
        if key in subject:
            return str(subject[key])
    build = subject.get("build")
    if isinstance(build, dict) and build.get("build_id"):
        return str(build["build_id"])
    artifact = subject.get("artifact")
    if isinstance(artifact, dict) and artifact.get("cache_key"):
        return str(artifact["cache_key"])
    return None


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def _placement_action_from_payload(payload: dict[str, Any]) -> str:
    backend = str(payload.get("execution_backend") or "")
    if backend == "fabric-session":
        return "run_daemon_resident_now"
    if backend == "daemon":
        return "run_daemon_now"
    if backend == "xrt":
        return "run_existing_pl_now"
    if backend == "sim":
        return "run_simulated_fabric_now"
    if backend == "ps":
        return "run_ps_now"
    return "run_existing_runtime_now"


def _value_store_summary(payload: dict[str, Any]) -> dict[str, Any]:
    store = dict(payload.get("value_store") or {})
    values = dict(store.get("values") or {})
    residencies: dict[str, int] = {}
    for value in values.values():
        residency = str((value or {}).get("residency") or "unknown")
        residencies[residency] = residencies.get(residency, 0) + 1
    return {
        "session_id": store.get("session_id"),
        "value_count": int(store.get("value_count") or len(values)),
        "materialization_count": int(store.get("materialization_count") or 0),
        "release_count": int(store.get("release_count") or 0),
        "residencies": residencies,
    }


def _validation_summary(payload: dict[str, Any]) -> dict[str, Any]:
    values = dict(payload.get("fabric_values") or {})
    records = {}
    ok_values: list[bool] = []
    for name, value in values.items():
        if not isinstance(value, dict):
            continue
        validation = value.get("validation")
        if not isinstance(validation, dict):
            continue
        records[str(name)] = dict(validation)
        if "ok" in validation:
            ok_values.append(bool(validation.get("ok")))
    if "sha_matches_expected" in payload:
        ok_values.append(bool(payload.get("sha_matches_expected")))
        records["final_output"] = {
            "ok": bool(payload.get("sha_matches_expected")),
            "final_sha256": payload.get("final_sha256"),
            "expected_sha256": payload.get("expected_sha256"),
        }
    return {
        "ok": all(ok_values) if ok_values else None,
        "record_count": len(records),
        "records": records,
    }


def _payload_health_ok(payload: dict[str, Any]) -> bool:
    if not bool(payload.get("ok")):
        return False
    if payload.get("error"):
        return False
    validation = _validation_summary(payload)
    if validation.get("ok") is False:
        return False
    return True


def _health_failure_reason(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        error = str(payload.get("error"))
        lower = error.lower()
        if any(token in lower for token in ("timeout", "timed out", "hang")):
            return "execution_timeout"
        if any(token in lower for token in ("xrt", "zocl", "xclbin", "kernel")):
            return "fabric_runtime_error"
        return "execution_error"
    if _validation_summary(payload).get("ok") is False:
        return "validation_failed"
    if not bool(payload.get("ok")):
        return "execution_not_ok"
    return "unknown_unhealthy_state"


def _quarantine_severity(payload: dict[str, Any]) -> str:
    text = json.dumps({"error": payload.get("error"), "backend": payload.get("execution_backend")}, sort_keys=True).lower()
    if any(token in text for token in ("timeout", "hang", "reset", "wedged", "zocl", "xrt", "kernel")):
        return "critical"
    if _validation_summary(payload).get("ok") is False:
        return "high"
    return "medium"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
