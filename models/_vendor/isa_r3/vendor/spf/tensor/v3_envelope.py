"""SPF_TB_v3 archival envelope — self-describing safetensors-style container.

Same high-level shape as the v2 envelope (``struct.pack("<Q", header_len)`` +
UTF-8 JSON header + concatenated binary chunks), with v3 deltas (design §13.1 / §7):

* ``__metadata__.format = "spftensor"``, ``spftensor_major = 3``, ``spftensor_minor = 0``.
* each block carries ``dtype = "SPF_TB_v3"`` plus ``representation`` (u8) and
  ``residual_codec`` (u8), and a ``base_profile_id`` (with a serialized base tree
  that may include ``SEPARABLE_PHASE`` / ``SEPARABLE_LOGMAG`` ``(a, ω)`` vectors).

Numpy arrays anywhere in the base params / residual payload are spilled to the
binary section and referenced by a chunk id, so any registered base model or
residual codec serialises without a bespoke schema. A v3 reader accepts
``major ∈ {2, 3}`` (v2 is relabelled on the way in); the v2 reader rejects
``major == 3`` (design rule 9 / T17_3).
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import BandRange, SPFError
from .v2 import TensorBlockHeaderV2, from_spftensor_safetensors
from .v3 import (
    SPFTENSOR_DTYPE_V3,
    SPFTENSOR_V3_MAJOR,
    SPFTENSOR_V3_MINOR,
    BaseModelRef,
    RepresentationRef,
    ResidualRef,
    SPFBlockV3,
    tensorblock_to_v3,
    validate_profile_id,
)

_FORMAT = "spftensor"


# --------------------------------------------------------------------------- #
# Generic tree (de)serialisation with a binary chunk table.
# --------------------------------------------------------------------------- #


def _encode_tree(obj: Any, chunks: list[bytes]) -> Any:
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        sid = len(chunks)
        chunks.append(arr.tobytes())
        return {"__nd__": sid, "dtype": arr.dtype.str, "shape": list(obj.shape)}
    if isinstance(obj, SPFBlockV3):
        # A nested block — the composition primitive (an ITERATED_OPERATOR carries
        # its generator block). Serialised inline, sharing this envelope's chunk
        # table so its arrays spill alongside the parent's. Reconstructs to an
        # SPFBlockV3 with bit-identical planes.
        return {
            "__blk__": {
                "dims": list(obj.dims),
                "origin": list(obj.origin),
                "representation": int(obj.representation.rep_id),
                "residual_codec": int(obj.residual.codec_id),
                "base_profile_id": int(obj.base.profile_id),
                "base": _encode_tree(dict(obj.base.params), chunks),
                "residual": _encode_tree(obj.residual.payload, chunks),
                "outliers": [[int(k), int(v)] for k, v in obj.outliers.items()],
            }
        }
    if isinstance(obj, TensorBlockHeaderV2):
        return {"__hdr__": obj.to_metadata()}
    if isinstance(obj, BandRange):
        return {"__band__": [obj.dim, obj.start, obj.length, obj.L_delta, obj.T_delta, obj.effective_flags]}
    if isinstance(obj, Mapping):
        return {"__map__": [[str(k), _encode_tree(v, chunks)] for k, v in obj.items()]}
    if isinstance(obj, (list, tuple)):
        return {"__seq__": [_encode_tree(v, chunks) for v in obj]}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    raise SPFError(f"cannot serialise object of type {type(obj)!r} into SPF_TB_v3")


def _decode_tree(node: Any, payload: bytes, table: list[tuple[int, int]]) -> Any:
    if isinstance(node, dict):
        if "__nd__" in node:
            sid = int(node["__nd__"])
            if sid < 0 or sid >= len(table):
                raise SPFError("SPF_TB chunk id out of range")
            off, length = table[sid]
            if off < 0 or length < 0 or off + length > len(payload):
                raise SPFError("SPF_TB chunk offset/length out of range")
            buf = payload[off : off + length]
            try:
                dt = np.dtype(node["dtype"])
                shape = tuple(int(s) for s in node["shape"])
            except (TypeError, ValueError) as exc:
                raise SPFError("SPF_TB chunk dtype/shape malformed") from exc
            count = 1
            for s in shape:  # overflow-free product (a forged huge shape fails the length check below)
                count *= s
            if len(buf) != dt.itemsize * count:
                raise SPFError("SPF_TB chunk byte length does not match dtype·shape")
            arr = np.frombuffer(buf, dtype=dt)
            return arr.reshape(shape).copy()
        if "__blk__" in node:
            b = node["__blk__"]
            return SPFBlockV3(
                dims=tuple(b["dims"]),
                origin=tuple(b["origin"]),
                representation=RepresentationRef(int(b["representation"])),
                base=BaseModelRef(
                    int(b["base_profile_id"]),
                    params=_decode_tree(b["base"], payload, table),
                ),
                residual=ResidualRef(
                    int(b["residual_codec"]),
                    payload=_decode_tree(b["residual"], payload, table),
                ),
                outliers={int(k): int(v) for k, v in b["outliers"]},
            )
        if "__hdr__" in node:
            return TensorBlockHeaderV2.from_metadata(node["__hdr__"])
        if "__band__" in node:
            d, s, length, ld, td, fl = node["__band__"]
            return BandRange(dim=d, start=s, length=length, L_delta=ld, T_delta=td, flags=fl)
        if "__map__" in node:
            return {k: _decode_tree(v, payload, table) for k, v in node["__map__"]}
        if "__seq__" in node:
            return [_decode_tree(v, payload, table) for v in node["__seq__"]]
    return node


# --------------------------------------------------------------------------- #
# Write / read.
# --------------------------------------------------------------------------- #


def to_spftensor_v3(blocks: SPFBlockV3 | Sequence[SPFBlockV3]) -> bytes:
    """Serialise one or more :class:`SPFBlockV3` into an SPF_TB_v3 envelope."""
    if isinstance(blocks, SPFBlockV3):
        blocks = (blocks,)
    chunks: list[bytes] = []
    block_entries: list[dict[str, Any]] = []
    for blk in blocks:
        validate_profile_id(blk.base.profile_id)
        block_entries.append(
            {
                "dtype": SPFTENSOR_DTYPE_V3,
                "dims": list(blk.dims),
                "origin": list(blk.origin),
                "representation": int(blk.representation.rep_id),
                "residual_codec": int(blk.residual.codec_id),
                "base_profile_id": int(blk.base.profile_id),
                "base": _encode_tree(blk.base.params, chunks),
                "residual": _encode_tree(blk.residual.payload, chunks),
                "outliers": [[int(k), int(v)] for k, v in blk.outliers.items()],
            }
        )
    # Build the chunk offset table over the concatenated payload.
    table: list[tuple[int, int]] = []
    off = 0
    for c in chunks:
        table.append((off, len(c)))
        off += len(c)
    header = {
        "__metadata__": {
            "format": _FORMAT,
            "spftensor_major": str(SPFTENSOR_V3_MAJOR),
            "spftensor_minor": str(SPFTENSOR_V3_MINOR),
            "created_unix_ns": str(time.time_ns()),
        },
        "chunk_table": table,
        "blocks": block_entries,
    }
    head_bytes = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(head_bytes)) + head_bytes + b"".join(chunks)


def _read_v3(header: Mapping[str, Any], payload: bytes) -> tuple[SPFBlockV3, ...]:
    table = [tuple(t) for t in header["chunk_table"]]
    out: list[SPFBlockV3] = []
    for entry in header["blocks"]:
        if entry.get("dtype") != SPFTENSOR_DTYPE_V3:
            raise SPFError(f"unexpected block dtype {entry.get('dtype')!r} in SPF_TB_v3")
        base = BaseModelRef(
            int(entry["base_profile_id"]),
            params=_decode_tree(entry["base"], payload, table),
        )
        residual = ResidualRef(
            int(entry["residual_codec"]),
            payload=_decode_tree(entry["residual"], payload, table),
        )
        out.append(
            SPFBlockV3(
                dims=tuple(entry["dims"]),
                origin=tuple(entry["origin"]),
                representation=RepresentationRef(int(entry["representation"])),
                base=base,
                residual=residual,
                outliers={int(k): int(v) for k, v in entry["outliers"]},
            )
        )
    return tuple(out)


def from_spftensor_v3(data: bytes) -> SPFBlockV3 | tuple[SPFBlockV3, ...]:
    """Read an SPF_TB envelope, accepting ``major ∈ {2, 3}``.

    A ``major == 2`` envelope is read with the v2 reader and **relabelled** to v3
    (``representation = POLAR_SPF32``, ``base = STB_AFFINE_POLY``, ``residual =
    RAW_INT16``) with bit-identical reconstruction. ``major == 3`` is read
    natively. Returns a single block when the envelope holds one.
    """
    if len(data) < 8:
        raise SPFError("SPF_TB envelope too short")
    (head_len,) = struct.unpack_from("<Q", data, 0)
    if head_len > len(data) - 8:
        raise SPFError("SPF_TB header length exceeds buffer")
    try:
        header = json.loads(data[8 : 8 + head_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SPFError("SPF_TB header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise SPFError("SPF_TB header root must be a JSON object")
    payload = data[8 + head_len :]
    meta = header.get("__metadata__", {})
    if not isinstance(meta, dict) or meta.get("format") != _FORMAT:
        raise SPFError("not an SPFTensor envelope")
    try:
        major = int(meta.get("spftensor_major", SPFTENSOR_V3_MAJOR))
    except (TypeError, ValueError) as exc:
        raise SPFError("SPFTensor major version is malformed") from exc
    if major == SPFTENSOR_V3_MAJOR:
        blocks = _read_v3(header, payload)
    elif major == 2:
        v2 = from_spftensor_safetensors(data)
        v2_blocks = v2 if isinstance(v2, tuple) else (v2,)
        blocks = tuple(tensorblock_to_v3(b) for b in v2_blocks)
    else:
        raise SPFError(f"unsupported SPFTensor major version {major} for v3 reader")
    return blocks[0] if len(blocks) == 1 else blocks


__all__ = [
    "from_spftensor_v3",
    "to_spftensor_v3",
]
