"""PANFG1 graph-request encoder for ``pan-fabricd``.

Wire layout vendored from the reference client
``pan-isa/pan_isa_x86_runtime_spf/fabric_daemon.py:23-51,741-781`` and the
daemon's own constants ``pan-isa/pan_fabricd/src/pan_fabricd.cpp:51-62``.
Vendored rather than imported because ``pan-isa`` is not an installable
package and this module must stay usable from the SPF checkout alone.

Only the inline-uint32 storage class is emitted: fd/dma-buf transport
(``GRAPH_STORAGE_FD_U32``) and daemon-BO reuse
(``GRAPH_STORAGE_DAEMON_BO_U32``) are daemon-side residency options that a
pure encoder cannot establish.
"""

from __future__ import annotations

import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

GRAPH_REQUEST_MAGIC = b"PANFG1\0\0"
GRAPH_RESPONSE_MAGIC = b"PANFGR1\0"
VERSION = 1
GRAPH_COMMAND_EXECUTE = 2
GRAPH_FLAG_KEEP_OUTPUTS_RESIDENT = 1 << 0
GRAPH_STORAGE_EMPTY = 0
GRAPH_STORAGE_INLINE_U32 = 1
GRAPH_DTYPE_U32 = 1

GRAPH_REQUEST_HEADER_STRUCT = struct.Struct("<8sHHIQQIIII")
GRAPH_BUFFER_STRUCT = struct.Struct("<32sHHIIQQQ")
GRAPH_OP_STRUCT = struct.Struct("<32sHHIIIIII")

#: ``pan_fabricd.cpp:53-59``. ``spf32.conj`` (3) has a constant and a client
#: entry but is **never** passed to ``add_graph_kernel`` (``:557-562``), so the
#: daemon answers ``GRAPH_STATUS_UNSUPPORTED`` for it. It is listed here for
#: faithfulness of the encoding and excluded from :data:`DAEMON_REGISTERED_OPS`.
GRAPH_OP_IDS: Mapping[str, int] = {
    "spf32.mul": 1,
    "spf32.add": 2,
    "spf32.conj": 3,
    "spf32.abs2": 4,
    "spf32.fft64": 5,
    "spf32.div": 6,
    "mlp.gemm_k16": 7,
}

#: Ops the daemon actually registers a kernel for (``pan_fabricd.cpp:557-562``).
DAEMON_REGISTERED_OPS = frozenset({"spf32.mul", "spf32.add", "spf32.div", "spf32.abs2", "spf32.fft64"})

GRAPH_CALL_ABI_IDS: Mapping[str, int] = {"binary_u32_n": 1, "unary_u32_n": 2, "fft64_u32": 3}

#: Which call ABI each registered op uses (``pan_fabricd.cpp:557-562``).
OP_CALL_ABI: Mapping[str, str] = {
    "spf32.mul": "binary_u32_n",
    "spf32.add": "binary_u32_n",
    "spf32.div": "binary_u32_n",
    "spf32.abs2": "unary_u32_n",
    "spf32.fft64": "fft64_u32",
}

_NAME_BYTES = 32


def _fixed_name(name: str) -> bytes:
    raw = str(name).encode("utf-8")[: _NAME_BYTES - 1]
    return raw.ljust(_NAME_BYTES, b"\0")


@dataclass(frozen=True, slots=True)
class FabricOp:
    """One PANFG1 graph node."""

    op_id: str
    op: str
    inputs: tuple[str, ...]
    output: str
    elements: int


@dataclass(frozen=True, slots=True)
class FabricGraph:
    """A PANFG1 graph: named uint32 buffers, ordered ops, requested outputs."""

    ops: tuple[FabricOp, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]

    def buffer_names(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for name in self.inputs:
            seen.setdefault(name, None)
        for op in self.ops:
            for name in (*op.inputs, op.output):
                seen.setdefault(name, None)
        for name in self.outputs:
            seen.setdefault(name, None)
        return tuple(seen)


def encode_request(
    graph: FabricGraph,
    payloads: Mapping[str, Sequence[int] | np.ndarray],
    *,
    request_id: int = 1,
    materialize_outputs: bool = True,
) -> bytes:
    """Serialize ``graph`` as a PANFG1 execute request.

    ``payloads`` supplies the inline uint32 words for every graph input; every
    other buffer is declared ``GRAPH_STORAGE_EMPTY`` and allocated daemon-side.
    Mirrors ``fabric_daemon.py:741-781`` field-for-field.
    """
    names = graph.buffer_names()
    index_by_name = {name: i for i, name in enumerate(names)}
    payload_base = (
        GRAPH_REQUEST_HEADER_STRUCT.size
        + GRAPH_BUFFER_STRUCT.size * len(names)
        + GRAPH_OP_STRUCT.size * len(graph.ops)
        + 4 * len(graph.outputs)
    )
    entries: list[dict[str, int | str]] = []
    payload_parts: list[bytes] = []
    offset = payload_base
    for name in names:
        if name in payloads:
            raw = np.ascontiguousarray(np.asarray(payloads[name], dtype=np.uint32).ravel()).tobytes()
            entries.append(
                {
                    "name": name,
                    "storage": GRAPH_STORAGE_INLINE_U32,
                    "elements": len(raw) // 4,
                    "payload_offset": offset,
                    "payload_bytes": len(raw),
                }
            )
            payload_parts.append(raw)
            offset += len(raw)
        else:
            elements = next((op.elements for op in graph.ops if op.output == name), 0)
            entries.append(
                {
                    "name": name,
                    "storage": GRAPH_STORAGE_EMPTY,
                    "elements": int(elements),
                    "payload_offset": 0,
                    "payload_bytes": 0,
                }
            )
    total_size = offset
    flags = 0 if materialize_outputs else GRAPH_FLAG_KEEP_OUTPUTS_RESIDENT
    header = GRAPH_REQUEST_HEADER_STRUCT.pack(
        GRAPH_REQUEST_MAGIC,
        VERSION,
        GRAPH_COMMAND_EXECUTE,
        flags,
        int(request_id),
        int(total_size),
        len(entries),
        len(graph.ops),
        len(graph.outputs),
        0,  # no file descriptors: inline storage only
    )
    buffer_blob = b"".join(
        GRAPH_BUFFER_STRUCT.pack(
            _fixed_name(str(e["name"])),
            GRAPH_DTYPE_U32,
            int(e["storage"]),
            0,
            int(e["elements"]),
            int(e["payload_offset"]),
            int(e["payload_bytes"]),
            0,
        )
        for e in entries
    )
    op_blob = b"".join(
        GRAPH_OP_STRUCT.pack(
            _fixed_name(op.op_id),
            GRAPH_OP_IDS[op.op],
            GRAPH_CALL_ABI_IDS[OP_CALL_ABI[op.op]],
            0,
            index_by_name[op.inputs[0]] if op.inputs else 0xFFFFFFFF,
            index_by_name[op.inputs[1]] if len(op.inputs) > 1 else 0xFFFFFFFF,
            index_by_name[op.output],
            int(op.elements),
            0,
        )
        for op in graph.ops
    )
    output_blob = b"".join(struct.pack("<I", index_by_name[n]) for n in graph.outputs)
    return header + buffer_blob + op_blob + output_blob + b"".join(payload_parts)


__all__ = [
    "DAEMON_REGISTERED_OPS",
    "GRAPH_OP_IDS",
    "FabricGraph",
    "FabricOp",
    "encode_request",
]
