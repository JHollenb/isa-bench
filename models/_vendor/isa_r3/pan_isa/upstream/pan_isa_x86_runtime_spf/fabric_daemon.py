"""Client for the resident PAN fabric daemon."""
from __future__ import annotations

import json
import os
import socket
import struct
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


REQUEST_MAGIC = b"PANFD1\0\0"
RESPONSE_MAGIC = b"PANFR1\0\0"
VERSION = 1
OP_SPF32_MUL_BENCH = 1

REQUEST_STRUCT = struct.Struct("<8sHHIIIIIIQ")
RESPONSE_STRUCT = struct.Struct("<8sHHIQIIQQQQQdQQQd8I192s")

GRAPH_REQUEST_MAGIC = b"PANFG1\0\0"
GRAPH_RESPONSE_MAGIC = b"PANFGR1\0"
GRAPH_COMMAND_EXECUTE = 2
GRAPH_FLAG_KEEP_OUTPUTS_RESIDENT = 1 << 0
GRAPH_RESPONSE_FLAG_RESIDENT_BOS = 1 << 0
GRAPH_STORAGE_EMPTY = 0
GRAPH_STORAGE_INLINE_U32 = 1
GRAPH_STORAGE_FD_U32 = 2
GRAPH_STORAGE_DAEMON_BO_U32 = 3
GRAPH_DTYPE_U32 = 1
GRAPH_STATUS_OK = 0
GRAPH_STATUS_ERROR = 1
GRAPH_STATUS_UNSUPPORTED = 2

GRAPH_REQUEST_HEADER_STRUCT = struct.Struct("<8sHHIQQIIII")
GRAPH_BUFFER_STRUCT = struct.Struct("<32sHHIIQQQ")
GRAPH_OP_STRUCT = struct.Struct("<32sHHIIIIII")
GRAPH_RESPONSE_HEADER_STRUCT = struct.Struct("<8sHHIQQIIII")
GRAPH_OP_IDS = {
    "spf32.mul": 1,
    "spf32.add": 2,
    "spf32.conj": 3,
    "spf32.abs2": 4,
    "spf32.fft64": 5,
    "spf32.div": 6,
    "mlp.gemm_k16": 7,
}
GRAPH_CALL_ABI_IDS = {
    "binary_u32_n": 1,
    "unary_u32_n": 2,
    "fft64_u32": 3,
}

VALUE_REQUEST_MAGIC = b"PANFV1\0\0"
VALUE_RESPONSE_MAGIC = b"PANFVR1\0"
VALUE_COMMAND_QUERY = 1
VALUE_COMMAND_DESCRIBE = 2
VALUE_COMMAND_RELEASE = 3
VALUE_COMMAND_MATERIALIZE = 4
VALUE_COMMAND_EXPORT = 5
VALUE_COMMAND_IMPORT = 6
VALUE_COMMAND_RETAIN = 7
VALUE_STATUS_OK = 0
VALUE_STATUS_ERROR = 1
VALUE_STATUS_UNSUPPORTED = 2
VALUE_CAP_DMA_BUF_EXPORT = 1 << 0
VALUE_CAP_DMA_BUF_IMPORT = 1 << 1
VALUE_REQUEST_STRUCT = struct.Struct("<8sHHIQQII")
VALUE_RESPONSE_STRUCT = struct.Struct("<8sHHIQQIIIIII")


@dataclass(frozen=True)
class FabricDaemonBenchResult:
    ok: bool
    socket_path: str
    request_id: int
    n_words: int
    reps: int
    warmup: int
    roundtrip_seconds: float
    status: int
    min_ns: int = 0
    p50_ns: int = 0
    p90_ns: int = 0
    p99_ns: int = 0
    max_ns: int = 0
    mean_ns: float = 0.0
    sync_to_device_ns: int = 0
    sync_from_device_ns: int = 0
    daemon_elapsed_ns: int = 0
    mwords_per_s: float = 0.0
    first_words: tuple[int, ...] = ()
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "pan_fabric_daemon_bench_v1",
            "ok": self.ok,
            "socket_path": self.socket_path,
            "request_id": self.request_id,
            "op": "spf32.mul",
            "n_words": self.n_words,
            "reps": self.reps,
            "warmup": self.warmup,
            "status": self.status,
            "roundtrip_seconds": self.roundtrip_seconds,
            "daemon_elapsed_seconds": self.daemon_elapsed_ns / 1e9,
            "min_ns": self.min_ns,
            "p50_ns": self.p50_ns,
            "p90_ns": self.p90_ns,
            "p99_ns": self.p99_ns,
            "max_ns": self.max_ns,
            "mean_ns": self.mean_ns,
            "ns_per_word": (self.p50_ns / self.n_words) if self.n_words else 0.0,
            "mwords_per_s": self.mwords_per_s,
            "sync_to_device_ns": self.sync_to_device_ns,
            "sync_from_device_ns": self.sync_from_device_ns,
            "first_words": [f"0x{word:08x}" for word in self.first_words],
            "error": self.error,
        }


@dataclass(frozen=True)
class FabricDaemonGraphResult:
    ok: bool
    socket_path: str
    request_id: int
    graph_op_count: int
    output_count: int
    roundtrip_seconds: float
    daemon_elapsed_seconds: float
    input_bytes: int
    output_bytes: int
    outputs: dict[str, Any]
    status: int = GRAPH_STATUS_OK
    error: str = ""
    buffer_storage: str = "inline"
    flags: int = 0
    resident_output_count: int = 0
    avoided_host_materialization_count: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "schema": "pan_fabric_daemon_graph_result_v1",
            "ok": self.ok,
            "socket_path": self.socket_path,
            "request_id": self.request_id,
            "graph_op_count": self.graph_op_count,
            "output_count": self.output_count,
            "roundtrip_seconds": self.roundtrip_seconds,
            "daemon_elapsed_seconds": self.daemon_elapsed_seconds,
            "input_bytes": self.input_bytes,
            "output_bytes": self.output_bytes,
            "buffer_storage": self.buffer_storage,
            "status": self.status,
            "flags": self.flags,
            "resident_output_count": self.resident_output_count,
            "avoided_host_materialization_count": self.avoided_host_materialization_count,
            "outputs": {
                name: _output_digest(value)
                for name, value in sorted(self.outputs.items())
            },
            "error": self.error,
        }


@dataclass(frozen=True)
class FabricDaemonValueResult:
    ok: bool
    socket_path: str
    command: int
    request_id: int
    status: int
    handle: int = 0
    elements: int = 0
    payload: bytes = b""
    resident_count: int = 0
    capability_flags: int = 0
    error: str = ""
    roundtrip_seconds: float = 0.0

    def json_payload(self) -> dict[str, Any]:
        if not self.payload:
            return {}
        return json.loads(self.payload.decode("utf-8"))

    def materialized_array(self) -> Any:
        np = _numpy()
        return np.frombuffer(self.payload, dtype=np.uint32, count=int(self.elements)).copy()

    def to_json(self) -> dict[str, Any]:
        payload_kind = "json" if self.payload[:1] in {b"{", b"["} else ("u32_bytes" if self.payload else "empty")
        data = {
            "schema": "pan_fabric_daemon_value_result_v1",
            "ok": self.ok,
            "socket_path": self.socket_path,
            "command": self.command,
            "request_id": self.request_id,
            "status": self.status,
            "handle": self.handle,
            "elements": self.elements,
            "payload_bytes": len(self.payload),
            "payload_kind": payload_kind,
            "resident_count": self.resident_count,
            "capability_flags": self.capability_flags,
            "roundtrip_seconds": self.roundtrip_seconds,
            "error": self.error,
        }
        if payload_kind == "json":
            data["payload"] = self.json_payload()
        elif payload_kind == "u32_bytes":
            data["payload_digest"] = _array_digest(self.materialized_array())
        return data


class FabricDaemonGraphSession:
    """Persistent client connection for repeated PAN fabric graph packets."""

    def __init__(self, socket_path: str, *, timeout: float = 120.0):
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)
        self._sock: Optional[socket.socket] = None

    def __enter__(self) -> "FabricDaemonGraphSession":
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def connect(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self._sock = sock

    def close(self) -> None:
        if self._sock is None:
            return
        try:
            self._sock.close()
        finally:
            self._sock = None

    def execute_graph(
        self,
        plan: Any,
        inputs: dict[str, Any],
        *,
        output_names: Optional[Sequence[str]] = None,
        request_id: Optional[int] = None,
        buffer_storage: str = "inline",
        materialize_outputs: bool = True,
    ) -> FabricDaemonGraphResult:
        self.connect()
        assert self._sock is not None
        return _execute_fabric_graph_daemon_on_socket(
            self._sock,
            self.socket_path,
            plan,
            inputs,
            output_names=output_names,
            request_id=request_id,
            buffer_storage=buffer_storage,
            materialize_outputs=materialize_outputs,
        )

    def query_values(self, *, request_id: Optional[int] = None) -> FabricDaemonValueResult:
        return self._value_command(VALUE_COMMAND_QUERY, request_id=request_id)

    def describe_value(self, handle: int, *, request_id: Optional[int] = None) -> FabricDaemonValueResult:
        return self._value_command(VALUE_COMMAND_DESCRIBE, handle=int(handle), request_id=request_id)

    def release_value(self, handle: int, *, request_id: Optional[int] = None) -> FabricDaemonValueResult:
        return self._value_command(VALUE_COMMAND_RELEASE, handle=int(handle), request_id=request_id)

    def retain_value(self, handle: int, *, request_id: Optional[int] = None) -> FabricDaemonValueResult:
        return self._value_command(VALUE_COMMAND_RETAIN, handle=int(handle), request_id=request_id)

    def materialize_value(
        self,
        handle: int,
        *,
        elements: int = 0,
        request_id: Optional[int] = None,
    ) -> FabricDaemonValueResult:
        return self._value_command(
            VALUE_COMMAND_MATERIALIZE,
            handle=int(handle),
            elements=int(elements),
            request_id=request_id,
        )

    def export_value(self, handle: int, *, request_id: Optional[int] = None) -> FabricDaemonValueResult:
        return self._value_command(VALUE_COMMAND_EXPORT, handle=int(handle), request_id=request_id)

    def import_value(
        self,
        descriptor: dict[str, Any],
        *,
        request_id: Optional[int] = None,
    ) -> FabricDaemonValueResult:
        payload = json.dumps(descriptor, sort_keys=True).encode("utf-8")
        return self._value_command(VALUE_COMMAND_IMPORT, payload=payload, request_id=request_id)

    def import_u32_value(
        self,
        array: Any,
        *,
        request_id: Optional[int] = None,
    ) -> FabricDaemonValueResult:
        np = _numpy()
        arr = np.ascontiguousarray(array, dtype=np.uint32).reshape(-1)
        return self._value_command(
            VALUE_COMMAND_IMPORT,
            elements=int(arr.size),
            payload=memoryview(arr).cast("B").tobytes(),
            request_id=request_id,
        )

    def _value_command(
        self,
        command: int,
        *,
        handle: int = 0,
        elements: int = 0,
        payload: bytes = b"",
        request_id: Optional[int] = None,
    ) -> FabricDaemonValueResult:
        self.connect()
        assert self._sock is not None
        return _execute_daemon_value_on_socket(
            self._sock,
            self.socket_path,
            int(command),
            handle=int(handle),
            elements=int(elements),
            payload=payload,
            request_id=request_id,
        )


def fabric_daemon_bench(
    socket_path: str,
    *,
    n_words: int = 4096,
    reps: int = 2000,
    warmup: int = 20,
    seed_a: int = 3201 + ord("m"),
    seed_b: int = 4201 + ord("m"),
    request_id: Optional[int] = None,
    timeout: float = 120.0,
) -> FabricDaemonBenchResult:
    if request_id is None:
        request_id = int(time.time_ns()) & ((1 << 64) - 1)
    payload = REQUEST_STRUCT.pack(
        REQUEST_MAGIC,
        VERSION,
        OP_SPF32_MUL_BENCH,
        0,
        int(n_words),
        int(reps),
        int(warmup),
        int(seed_a),
        int(seed_b),
        int(request_id),
    )
    started = time.perf_counter()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        sock.sendall(payload)
        raw = _recv_exact(sock, RESPONSE_STRUCT.size)
    elapsed = time.perf_counter() - started
    unpacked = RESPONSE_STRUCT.unpack(raw)
    (
        magic,
        version,
        status,
        flags,
        response_id,
        resp_n,
        resp_reps,
        min_ns,
        p50_ns,
        p90_ns,
        p99_ns,
        max_ns,
        mean_ns,
        sync_to,
        sync_from,
        daemon_elapsed,
        mwords_per_s,
        *tail,
    ) = unpacked
    first_words = tuple(int(value) for value in tail[:8])
    error = bytes(tail[8]).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    if magic != RESPONSE_MAGIC:
        raise ValueError(f"bad daemon response magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported daemon response version {version}")
    return FabricDaemonBenchResult(
        ok=status == 0,
        socket_path=socket_path,
        request_id=int(response_id),
        n_words=int(resp_n),
        reps=int(resp_reps),
        warmup=int(warmup),
        roundtrip_seconds=elapsed,
        status=int(status),
        min_ns=int(min_ns),
        p50_ns=int(p50_ns),
        p90_ns=int(p90_ns),
        p99_ns=int(p99_ns),
        max_ns=int(max_ns),
        mean_ns=float(mean_ns),
        sync_to_device_ns=int(sync_to),
        sync_from_device_ns=int(sync_from),
        daemon_elapsed_ns=int(daemon_elapsed),
        mwords_per_s=float(mwords_per_s),
        first_words=first_words,
        error=error,
    )


def execute_fabric_graph_daemon(
    socket_path: str,
    plan: Any,
    inputs: dict[str, Any],
    *,
    output_names: Optional[Sequence[str]] = None,
    request_id: Optional[int] = None,
    timeout: float = 120.0,
    buffer_storage: str = "inline",
    materialize_outputs: bool = True,
) -> FabricDaemonGraphResult:
    """Execute a planned SPF graph through a resident ``pan-fabricd``.

    ``buffer_storage`` controls the client-to-daemon ABI:
    - ``inline`` embeds uint32 input/output payloads in the graph packet.
    - ``fd`` sends POSIX file descriptors over the Unix socket. The daemon maps
      them as host buffers and copies to/from XRT BOs in v1; the descriptor ABI
      is intentionally compatible with later dma-buf import.
    """

    if buffer_storage not in {"inline", "fd"}:
        raise ValueError(f"unsupported daemon buffer storage {buffer_storage!r}")
    if request_id is None:
        request_id = int(time.time_ns()) & ((1 << 64) - 1)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(socket_path)
        return _execute_fabric_graph_daemon_on_socket(
            sock,
            socket_path,
            plan,
            inputs,
            output_names=output_names,
            request_id=request_id,
            buffer_storage=buffer_storage,
            materialize_outputs=materialize_outputs,
        )


def _execute_daemon_value_on_socket(
    sock: socket.socket,
    socket_path: str,
    command: int,
    *,
    handle: int = 0,
    elements: int = 0,
    payload: bytes = b"",
    request_id: Optional[int] = None,
) -> FabricDaemonValueResult:
    if request_id is None:
        request_id = int(time.time_ns()) & ((1 << 64) - 1)
    packet = VALUE_REQUEST_STRUCT.pack(
        VALUE_REQUEST_MAGIC,
        VERSION,
        int(command),
        0,
        int(request_id),
        int(handle),
        int(elements),
        len(payload),
    ) + bytes(payload)
    started = time.perf_counter()
    sock.sendall(packet)
    raw_header = _recv_exact(sock, VALUE_RESPONSE_STRUCT.size)
    (
        magic,
        version,
        status,
        flags,
        response_id,
        response_handle,
        response_elements,
        payload_bytes,
        resident_count,
        capability_flags,
        error_bytes,
        _reserved,
    ) = VALUE_RESPONSE_STRUCT.unpack(raw_header)
    if magic != VALUE_RESPONSE_MAGIC:
        raise ValueError(f"bad value daemon response magic {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported value daemon response version {version}")
    error = ""
    if error_bytes:
        error = _recv_exact(sock, int(error_bytes)).split(b"\0", 1)[0].decode("utf-8", errors="replace")
    response_payload = _recv_exact(sock, int(payload_bytes)) if payload_bytes else b""
    elapsed = time.perf_counter() - started
    return FabricDaemonValueResult(
        ok=int(status) == VALUE_STATUS_OK,
        socket_path=socket_path,
        command=int(command),
        request_id=int(response_id),
        status=int(status),
        handle=int(response_handle),
        elements=int(response_elements),
        payload=response_payload,
        resident_count=int(resident_count),
        capability_flags=int(capability_flags),
        error=error,
        roundtrip_seconds=elapsed,
    )


def _execute_fabric_graph_daemon_on_socket(
    sock: socket.socket,
    socket_path: str,
    plan: Any,
    inputs: dict[str, Any],
    *,
    output_names: Optional[Sequence[str]],
    request_id: Optional[int],
    buffer_storage: str,
    materialize_outputs: bool,
) -> FabricDaemonGraphResult:
    if buffer_storage not in {"inline", "fd"}:
        raise ValueError(f"unsupported daemon buffer storage {buffer_storage!r}")
    if request_id is None:
        request_id = int(time.time_ns()) & ((1 << 64) - 1)
    packet, fds, fd_files, output_fd_names = _encode_graph_request(
        plan,
        inputs,
        output_names=output_names,
        request_id=int(request_id),
        buffer_storage=buffer_storage,
        materialize_outputs=materialize_outputs,
    )
    started = time.perf_counter()
    try:
        if fds:
            sent = sock.sendmsg([packet], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, _pack_fds(fds))])
            if sent < len(packet):
                sock.sendall(packet[sent:])
        else:
            sock.sendall(packet)
        raw_header = _recv_exact(sock, GRAPH_RESPONSE_HEADER_STRUCT.size)
        header = GRAPH_RESPONSE_HEADER_STRUCT.unpack(raw_header)
        (
            magic,
            version,
            status,
            flags,
            response_id,
            total_size,
            output_count,
            trace_count,
            error_bytes,
            daemon_elapsed_ns,
        ) = header
        if magic != GRAPH_RESPONSE_MAGIC:
            raise ValueError(f"bad graph daemon response magic {magic!r}")
        if version != VERSION:
            raise ValueError(f"unsupported graph daemon response version {version}")
        remaining_size = int(total_size) - GRAPH_RESPONSE_HEADER_STRUCT.size
        if remaining_size < 0:
            raise ValueError("graph daemon response total_size is shorter than header")
        body = _recv_exact(sock, remaining_size)
    finally:
        for fd in fds:
            try:
                os.close(fd)
            except OSError:
                pass
    try:
        outputs, output_bytes, error = _decode_graph_response_body(
            body,
            int(output_count),
            int(error_bytes),
            output_fd_names=output_fd_names,
        )
    finally:
        for file_obj in fd_files:
            file_obj.close()
    elapsed = time.perf_counter() - started
    resident_outputs = sum(1 for value in outputs.values() if _is_daemon_bo_descriptor(value))
    produced = {op.output for op in plan.graph.ops}
    requested = set(output_names or plan.graph.outputs)
    avoided_materializations = len(produced - requested) + (resident_outputs if not materialize_outputs else 0)
    return FabricDaemonGraphResult(
        ok=int(status) == GRAPH_STATUS_OK,
        socket_path=socket_path,
        request_id=int(response_id),
        graph_op_count=int(trace_count),
        output_count=int(output_count),
        roundtrip_seconds=elapsed,
        daemon_elapsed_seconds=int(daemon_elapsed_ns) / 1e9,
        input_bytes=sum(_input_nbytes(value) for value in inputs.values()),
        output_bytes=output_bytes,
        outputs=outputs,
        status=int(status),
        error=error,
        buffer_storage=buffer_storage,
        flags=int(flags),
        resident_output_count=resident_outputs,
        avoided_host_materialization_count=avoided_materializations,
    )


def write_json(payload: dict[str, Any], path: Optional[str]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = []
    remaining = int(size)
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EOFError("fabric daemon closed connection before response was complete")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _encode_graph_request(
    plan: Any,
    inputs: dict[str, Any],
    *,
    output_names: Optional[Sequence[str]],
    request_id: int,
    buffer_storage: str,
    materialize_outputs: bool,
) -> tuple[bytes, list[int], list[Any], dict[str, Any]]:
    np = _numpy()
    graph = plan.graph
    names = list(graph.buffers.keys())
    index_by_name = {name: index for index, name in enumerate(names)}
    produced = {op.output for op in graph.ops}
    outputs = tuple(output_names or graph.outputs)
    fds: list[int] = []
    fd_files: list[Any] = []
    output_fd_names: dict[str, Any] = {}
    inline_payloads: list[tuple[str, bytes]] = []
    buffer_entries: list[dict[str, Any]] = []
    for name in names:
        spec = graph.buffers[name]
        source = inputs.get(name)
        arr = None
        if source is not None:
            if _is_daemon_bo_descriptor(source):
                if int(source.get("elements", spec.elements)) != int(spec.elements):
                    raise ValueError(
                        f"daemon graph input {name!r} expected {spec.elements} words, got {source.get('elements')}"
                    )
            else:
                arr = np.ascontiguousarray(source, dtype=np.uint32).reshape(-1)
                if int(arr.size) != int(spec.elements):
                    raise ValueError(f"daemon graph input {name!r} expected {spec.elements} words, got {arr.size}")
        storage = GRAPH_STORAGE_EMPTY
        payload_bytes = 0
        payload_offset = 0
        fd_index = 0xFFFFFFFFFFFFFFFF
        if _is_daemon_bo_descriptor(source):
            storage = GRAPH_STORAGE_DAEMON_BO_U32
            payload_offset = int(source["handle"])
            payload_bytes = int(source.get("nbytes") or int(spec.elements) * 4)
        elif arr is not None and (name not in produced or name in inputs):
            raw = memoryview(arr).cast("B").tobytes()
            payload_bytes = len(raw)
            if buffer_storage == "fd":
                fd_index = len(fds)
                file_obj = tempfile.TemporaryFile()
                file_obj.write(raw)
                file_obj.flush()
                file_obj.seek(0)
                fd_files.append(file_obj)
                fds.append(os.dup(file_obj.fileno()))
                storage = GRAPH_STORAGE_FD_U32
            else:
                storage = GRAPH_STORAGE_INLINE_U32
                inline_payloads.append((name, raw))
        elif materialize_outputs and buffer_storage == "fd" and name in outputs:
            fd_index = len(fds)
            payload_bytes = int(spec.elements) * 4
            file_obj = tempfile.TemporaryFile()
            file_obj.truncate(payload_bytes)
            file_obj.seek(0)
            fd_files.append(file_obj)
            fds.append(os.dup(file_obj.fileno()))
            storage = GRAPH_STORAGE_FD_U32
            output_fd_names[name] = file_obj
        buffer_entries.append(
            {
                "name": name,
                "elements": int(spec.elements),
                "storage": storage,
                "payload_bytes": payload_bytes,
                "payload_offset": payload_offset,
                "fd_index": fd_index,
            }
        )
    header_size = GRAPH_REQUEST_HEADER_STRUCT.size
    desc_size = len(buffer_entries) * GRAPH_BUFFER_STRUCT.size
    op_size = len(graph.ops) * GRAPH_OP_STRUCT.size
    output_index_size = len(outputs) * 4
    payload_base = header_size + desc_size + op_size + output_index_size
    inline_by_name = {name: raw for name, raw in inline_payloads}
    payload_parts: list[bytes] = []
    for entry in buffer_entries:
        raw = inline_by_name.get(entry["name"])
        if raw is None:
            continue
        entry["payload_offset"] = payload_base + sum(len(part) for part in payload_parts)
        payload_parts.append(raw)
    total_size = payload_base + sum(len(part) for part in payload_parts)
    flags = GRAPH_FLAG_KEEP_OUTPUTS_RESIDENT if not materialize_outputs else 0
    header = GRAPH_REQUEST_HEADER_STRUCT.pack(
        GRAPH_REQUEST_MAGIC,
        VERSION,
        GRAPH_COMMAND_EXECUTE,
        flags,
        int(request_id),
        int(total_size),
        len(buffer_entries),
        len(graph.ops),
        len(outputs),
        len(fds),
    )
    buffer_blob = b"".join(
        GRAPH_BUFFER_STRUCT.pack(
            _fixed_name(entry["name"]),
            GRAPH_DTYPE_U32,
            int(entry["storage"]),
            0,
            int(entry["elements"]),
            int(entry["payload_offset"]),
            int(entry["payload_bytes"]),
            int(entry["fd_index"]),
        )
        for entry in buffer_entries
    )
    op_blob = b"".join(
        GRAPH_OP_STRUCT.pack(
            _fixed_name(op.op_id),
            GRAPH_OP_IDS.get(op.op, 0),
            GRAPH_CALL_ABI_IDS.get(getattr(_kernel_for_op(plan, op.op_id), "call_abi", "binary_u32_n"), 0),
            0,
            index_by_name[op.inputs[0]] if len(op.inputs) > 0 else 0xFFFFFFFF,
            index_by_name[op.inputs[1]] if len(op.inputs) > 1 else 0xFFFFFFFF,
            index_by_name[op.output],
            int(op.elements),
            0,
        )
        for op in graph.ops
    )
    output_blob = b"".join(struct.pack("<I", index_by_name[name]) for name in outputs)
    return header + buffer_blob + op_blob + output_blob + b"".join(payload_parts), fds, fd_files, output_fd_names


def _decode_graph_response_body(
    body: bytes,
    output_count: int,
    error_bytes: int,
    *,
    output_fd_names: dict[str, Any],
) -> tuple[dict[str, Any], int, str]:
    offset = 0
    error = ""
    if error_bytes:
        error = body[:error_bytes].split(b"\0", 1)[0].decode("utf-8", errors="replace")
        offset += error_bytes
    np = _numpy()
    outputs: dict[str, Any] = {}
    output_bytes = 0
    descs = []
    for _ in range(output_count):
        descs.append(GRAPH_BUFFER_STRUCT.unpack_from(body, offset))
        offset += GRAPH_BUFFER_STRUCT.size
    for raw_name, dtype, storage, _flags, elements, payload_offset, payload_bytes, _fd_index in descs:
        if dtype != GRAPH_DTYPE_U32:
            raise ValueError(f"unsupported graph response dtype {dtype}")
        name = _unpack_name(raw_name)
        byte_count = int(payload_bytes)
        if storage == GRAPH_STORAGE_INLINE_U32:
            output_bytes += byte_count
            start = int(payload_offset) - GRAPH_RESPONSE_HEADER_STRUCT.size
            end = start + byte_count
            raw = body[start:end]
            outputs[name] = np.frombuffer(raw, dtype=np.uint32).copy()
        elif storage == GRAPH_STORAGE_FD_U32 and name in output_fd_names:
            output_bytes += byte_count
            file_obj = output_fd_names[name]
            file_obj.seek(0)
            raw = file_obj.read(byte_count)
            outputs[name] = np.frombuffer(raw, dtype=np.uint32).copy()
        elif storage == GRAPH_STORAGE_EMPTY:
            outputs[name] = np.empty(int(elements), dtype=np.uint32)
        elif storage == GRAPH_STORAGE_DAEMON_BO_U32:
            outputs[name] = {
                "schema": "pan_daemon_bo_v1",
                "backend": "daemon",
                "name": name,
                "handle": int(payload_offset),
                "elements": int(elements),
                "nbytes": byte_count,
                "dtype": "uint32",
                "resident": True,
            }
        else:
            raise ValueError(f"unsupported graph response storage {storage} for {name!r}")
    return outputs, output_bytes, error


def _fixed_name(value: str) -> bytes:
    raw = str(value).encode("utf-8")
    if len(raw) > 31:
        raise ValueError(f"graph protocol name {value!r} is longer than 31 bytes")
    return raw + b"\0" * (32 - len(raw))


def _unpack_name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _kernel_for_op(plan: Any, op_id: str) -> Any:
    for step in plan.steps:
        if step.action == "run_pl" and step.op_id == op_id and step.kernel is not None:
            return step.kernel
    return None


def _pack_fds(fds: Sequence[int]) -> bytes:
    import array

    return array.array("i", [int(fd) for fd in fds]).tobytes()


def _numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("numpy is required for PAN fabric daemon graph packets") from exc
    return np


def _array_nbytes(value: Any) -> int:
    np = _numpy()
    return int(np.ascontiguousarray(value, dtype=np.uint32).reshape(-1).nbytes)


def _input_nbytes(value: Any) -> int:
    if _is_daemon_bo_descriptor(value):
        return 0
    return _array_nbytes(value)


def _array_digest(value: Any) -> dict[str, Any]:
    import hashlib

    np = _numpy()
    arr = np.ascontiguousarray(value, dtype=np.uint32).reshape(-1)
    return {
        "elements": int(arr.size),
        "sha256": hashlib.sha256(memoryview(arr).cast("B")).hexdigest(),
        "first_words": [f"0x{int(word):08x}" for word in arr[:8]],
    }


def _is_daemon_bo_descriptor(value: Any) -> bool:
    return isinstance(value, dict) and value.get("schema") == "pan_daemon_bo_v1" and "handle" in value


def _output_digest(value: Any) -> dict[str, Any]:
    if _is_daemon_bo_descriptor(value):
        return dict(value)
    return _array_digest(value)
