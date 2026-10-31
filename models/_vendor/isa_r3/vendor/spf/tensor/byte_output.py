"""SPF-native byte output consumers.

This module contains the two small output consumers needed by the next native
scaffold cut:

* :func:`emit_copy` extracts byte identity from SPF32 unit words.  The words
  are the observed value payload selected by a preceding record read; this
  function does not know, or accept, an answer label.
* :class:`ConditionalByteHead` consumes a typed integer mass plane.  Its
  factorised byte law has eight dependent binary decisions, exact integer
  sibling sums, and no floating sigmoid/softmax path.

The conditional mass plane is a registered ``SPFBlockV3`` structural profile.
``ByteMassHeader`` is its scalar header view and ``ByteMassPlane`` is a typed
consumer view over the block's immutable header-owned integer arrays.  The
native ABI mirrors and validates that layout; byte generation remains a
host-side integer consumer until a hardware backend is admitted.

Byte identity uses the existing SPF32 word contract:

``L = 0`` and ``Theta = byte * 256``.

The only arithmetic in the output consumers is integer extraction,
comparison, multiplication of integer probability numerators, and masking.
No Cartesian decode, trigonometric operation, sigmoid, softmax, or coherent
SPF reduction is performed here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .core import SPFError, pack_words, unpack_L, unpack_theta
from .v3 import (
    BYTE_MASS_MAX_CONTEXT_LIMIT,
    BYTE_MASS_MAX_OUTPUT_LENGTH,
    BYTE_MASS_NODE_COUNT,
    BYTE_MASS_SCHEMA,
    BYTE_MASS_SCORE_MAX,
    BYTE_MASS_SCORE_MIN,
    BaseModelKind,
    BaseModelRef,
    Representation,
    RepresentationRef,
    ResidualCodecKind,
    ResidualRef,
    SPFBlockV3,
    _immutable_bytes_array,
    register_base_model,
    validate_byte_mass_params,
)

BitOrder = Literal["msb0", "lsb0"]
StopLaw = Literal["max-length", "two-mass"]

_BYTE_PHASE_SHIFT = 8
_BYTE_PHASE_STEP = 1 << _BYTE_PHASE_SHIFT
_BYTE_PHASE_MASK = (1 << 16) - 1
_BYTE_NODE_COUNT = BYTE_MASS_NODE_COUNT
_MAX_MASS_Q = 63


class ByteOutputError(SPFError):
    """Raised when a byte consumer receives an invalid typed contract."""


def _require_bool_mask(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype != np.bool_:
        raise ByteOutputError(f"{name} must be a boolean mask")
    if raw.ndim == 0:
        return np.full(shape, bool(raw), dtype=np.bool_)
    if raw.shape != shape:
        raise ByteOutputError(f"{name} must have shape {shape}, got {raw.shape}")
    return raw.astype(np.bool_, copy=False)


def _require_uint_words(words: object) -> np.ndarray:
    raw = np.asarray(words)
    if raw.dtype != np.uint32:
        raise ByteOutputError("byte output requires packed uint32 SPF32 words")
    return raw


@dataclass(frozen=True, slots=True)
class CopyResult:
    """Result of exact packed-byte copying.

    ``data`` is the exact observed byte payload.  ``trace`` records the
    integer extraction rows and ``counters`` records mechanics counts.  The
    aliases ``bytes`` and ``payload`` keep the boundary convenient for worker
    code without changing the immutable result shape.
    """

    data: bytes
    trace: tuple[Mapping[str, int | bool | str], ...]
    counters: Mapping[str, int]

    @property
    def bytes(self) -> bytes:
        """Return the emitted bytes (an explicit alias for ``data``)."""

        return self.data

    @property
    def payload(self) -> bytes:
        """Return the emitted bytes (an explicit source-payload alias)."""

        return self.data

    def __iter__(self):
        """Allow ``data, trace, counters = emit_copy(...)`` at call sites."""

        yield self.data
        yield self.trace
        yield self.counters


def pack_utf8_bytes(payload: bytes | bytearray | memoryview) -> np.ndarray:
    """Pack an observed UTF-8 byte payload into SPF32 unit words.

    This is the ingress helper for an observed record payload.  It intentionally
    accepts bytes-like values only: a serving path must not turn a semantic
    answer label or Python string into an output source implicitly.
    """

    if isinstance(payload, memoryview):
        raw = payload.tobytes()
    elif isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        raise ByteOutputError("observed byte payload must be bytes-like")
    byte_values = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    phases = (byte_values.astype(np.uint32) << _BYTE_PHASE_SHIFT).astype(np.uint16)
    return pack_words(np.zeros(phases.shape, dtype=np.int16), phases).astype(np.uint32, copy=False)


def emit_copy(words: object, valid: object = True) -> CopyResult:
    """Extract an observed byte payload from packed SPF32 words.

    ``words`` must be ``uint32`` SPF32 unit words with ``L == 0`` and a phase
    code divisible by 256.  A scalar ``valid=False`` represents a missing
    record and emits an empty payload.  A per-byte validity mask fails closed
    when any byte is invalid rather than returning a partial answer.

    The function reads only the packed ``L``/``Theta`` planes.  It never
    decodes words to Cartesian values and has no access to evaluator targets.
    """

    raw = _require_uint_words(words)
    flat = raw.reshape(-1)
    valid_mask = _require_bool_mask(valid, flat.shape, "valid")
    valid_arg = np.asarray(valid)
    explicit_missing = valid_arg.ndim == 0 and not bool(valid_arg)
    counters: dict[str, int] = {
        "word_extracts": 0,
        "phase_checks": 0,
        "log_checks": 0,
        "bytes_emitted": 0,
        "valid": 0 if explicit_missing else int(valid_mask.all()),
        "missing": int((~valid_mask).sum()),
        "invalid": 0,
    }
    if explicit_missing or not valid_mask.all():
        status = "missing" if explicit_missing or not valid_mask.any() else "invalid"
        counters[status] = counters.get(status, 0) + 1
        return CopyResult(
            b"",
            (
                {
                    "status": status,
                    "valid_words": int(valid_mask.sum()),
                    "word_count": int(flat.size),
                },
            ),
            counters,
        )

    if flat.size == 0:
        return CopyResult(
            b"",
            ({"status": "valid", "word_count": 0},),
            counters,
        )

    log_codes = unpack_L(flat)
    counters["word_extracts"] = int(flat.size)
    counters["log_checks"] = int(flat.size)
    if np.any(log_codes != 0):
        counters["invalid"] += 1
        raise ByteOutputError("byte words must have SPF32 L=0 unit magnitude")

    phase_codes = unpack_theta(flat).astype(np.uint32, copy=False)
    counters["phase_checks"] = int(flat.size)
    if np.any((phase_codes & (_BYTE_PHASE_STEP - 1)) != 0):
        counters["invalid"] += 1
        raise ByteOutputError("byte word phases must be exact multiples of 256")
    if np.any(phase_codes > _BYTE_PHASE_MASK):
        counters["invalid"] += 1
        raise ByteOutputError("byte word phase is outside the SPF32 phase ring")

    byte_codes = (phase_codes >> _BYTE_PHASE_SHIFT).astype(np.uint8, copy=False)
    data = bytes(byte_codes.tolist())
    counters["bytes_emitted"] = len(data)
    trace = tuple(
        {
            "status": "emitted",
            "index": int(index),
            "phase": int(phase),
            "byte": int(byte),
        }
        for index, (phase, byte) in enumerate(zip(phase_codes, byte_codes, strict=True))
    )
    return CopyResult(data, trace, counters)


def _mass_dtype(q: int) -> np.dtype:
    bits = int(q) + 1  # Q+1 is required to represent ν == 2**Q.
    for dtype, width in ((np.uint8, 8), (np.uint16, 16), (np.uint32, 32), (np.uint64, 64)):
        if bits <= width:
            # The native validator loads multi-byte masses in little-endian
            # order, so the host-side carrier must use the same byte order.
            return np.dtype(dtype).newbyteorder("<")
    raise ByteOutputError("Q+1 mass width exceeds the available exact uint64 plane")


@dataclass(frozen=True, slots=True)
class ByteMassHeader:
    """Exact integer contract for a conditional byte mass plane.

    This is the scalar view/constructor for the registered v3
    ``BYTE_MASS_PLANE`` header.  The numerical mass arrays are owned by the
    resulting :class:`SPFBlockV3`; this object never acts as a second carrier.
    """

    q: int
    bit_order: BitOrder = "msb0"
    rom_id: str = "unbound-mass-rom"
    score_min: int = -(1 << 15)
    score_max: int = (1 << 15) - 1
    context_limit: int = 1
    max_output_length: int = 1
    stop_law: StopLaw = "max-length"
    stop_min_mass: int = 0
    abi: str = BYTE_MASS_SCHEMA
    consumer_backend: str = "host-python-integer"
    mass_bits: int = field(init=False)
    missing_sentinel: int = field(init=False)
    invalid_sentinel: int = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)):
            raise ByteOutputError("Q must be an integer")
        q = int(self.q)
        if q < 0:
            raise ByteOutputError("Q must be non-negative")
        if q > _MAX_MASS_Q:
            raise ByteOutputError("Q overflow: exact mass planes support Q <= 63")
        if self.bit_order not in ("msb0", "lsb0"):
            raise ByteOutputError("bit_order must be 'msb0' or 'lsb0'")
        if not isinstance(self.rom_id, str) or not self.rom_id:
            raise ByteOutputError("rom_id must be a non-empty string")
        try:
            rom_bytes = self.rom_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ByteOutputError("rom_id must be valid UTF-8") from exc
        if len(rom_bytes) >= 64 or b"\x00" in rom_bytes:
            raise ByteOutputError("rom_id must fit a NUL-free native 63-byte identity field")
        if not isinstance(self.abi, str) or self.abi != BYTE_MASS_SCHEMA:
            raise ByteOutputError(f"abi must be {BYTE_MASS_SCHEMA!r}")
        if not isinstance(self.consumer_backend, str) or not self.consumer_backend:
            raise ByteOutputError("consumer_backend must be a non-empty string")
        if isinstance(self.score_min, bool) or not isinstance(self.score_min, (int, np.integer)):
            raise ByteOutputError("score_min must be an integer")
        if isinstance(self.score_max, bool) or not isinstance(self.score_max, (int, np.integer)):
            raise ByteOutputError("score_max must be an integer")
        if (
            int(self.score_min) > int(self.score_max)
            or int(self.score_min) < BYTE_MASS_SCORE_MIN
            or int(self.score_max) > BYTE_MASS_SCORE_MAX
        ):
            raise ByteOutputError("score bounds are outside the native int32 range")
        if isinstance(self.context_limit, bool) or not isinstance(self.context_limit, (int, np.integer)):
            raise ByteOutputError("context_limit must be an integer")
        if int(self.context_limit) <= 0 or int(self.context_limit) > BYTE_MASS_MAX_CONTEXT_LIMIT:
            raise ByteOutputError("context_limit is outside the native uint32 range")
        if isinstance(self.max_output_length, bool) or not isinstance(self.max_output_length, (int, np.integer)):
            raise ByteOutputError("max_output_length must be an integer")
        if int(self.max_output_length) <= 0 or int(self.max_output_length) > BYTE_MASS_MAX_OUTPUT_LENGTH:
            raise ByteOutputError("max_output_length is outside the native uint32 range")
        if self.stop_law not in ("max-length", "two-mass"):
            raise ByteOutputError("stop_law must be 'max-length' or 'two-mass'")
        full = 1 << q
        if isinstance(self.stop_min_mass, bool) or not isinstance(self.stop_min_mass, (int, np.integer)):
            raise ByteOutputError("stop_min_mass must be an integer")
        if not 0 <= int(self.stop_min_mass) <= full:
            raise ByteOutputError("stop_min_mass is outside the Q-bit mass range")
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "score_min", int(self.score_min))
        object.__setattr__(self, "score_max", int(self.score_max))
        object.__setattr__(self, "context_limit", int(self.context_limit))
        object.__setattr__(self, "max_output_length", int(self.max_output_length))
        object.__setattr__(self, "stop_min_mass", int(self.stop_min_mass))
        object.__setattr__(self, "mass_bits", q + 1)
        # These are out-of-band state sentinels.  Valid mass entries occupy
        # exactly 0..2**Q, so full-scale remains a valid value at every Q.
        object.__setattr__(self, "missing_sentinel", full + 1)
        object.__setattr__(self, "invalid_sentinel", full + 2)
        _mass_dtype(q)  # fail at construction rather than during a later read

    @property
    def full_scale(self) -> int:
        """The exact denominator ``2**Q`` for one binary decision."""

        return 1 << self.q

    @property
    def node_count(self) -> int:
        """Number of binary-tree internal nodes for one byte (255)."""

        return _BYTE_NODE_COUNT

    @property
    def mass_dtype(self) -> np.dtype:
        """Canonical unsigned dtype for the Q+1 mass field."""

        return _mass_dtype(self.q)

    def to_metadata(self) -> dict[str, object]:
        """Return a JSON-safe header description for a receipt."""

        return {
            "abi": self.abi,
            "schema": self.abi,
            "q": self.q,
            "mass_bits": self.mass_bits,
            "mass_dtype": self.mass_dtype.name,
            "full_scale": self.full_scale,
            "bit_order": self.bit_order,
            "node_count": self.node_count,
            "rom_id": self.rom_id,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "context_limit": self.context_limit,
            "max_output_length": self.max_output_length,
            "stop_law": self.stop_law,
            "stop_min_mass": self.stop_min_mass,
            "missing_sentinel": self.missing_sentinel,
            "invalid_sentinel": self.invalid_sentinel,
            "consumer_backend": self.consumer_backend,
        }

    @classmethod
    def from_block(cls, block: SPFBlockV3) -> ByteMassHeader:
        """View the immutable scalar header stored in one registered block."""

        if not isinstance(block, SPFBlockV3) or block.base.profile_id != int(BaseModelKind.BYTE_MASS_PLANE):
            raise ByteOutputError("block is not a registered BYTE_MASS_PLANE")
        params = block.base.params
        return cls(
            q=int(params["q"]),
            bit_order=params["bit_order"],
            rom_id=str(params["rom_id"]),
            score_min=int(params["score_min"]),
            score_max=int(params["score_max"]),
            context_limit=int(params["context_limit"]),
            max_output_length=int(params["max_output_length"]),
            stop_law=params["stop_law"],
            stop_min_mass=int(params["stop_min_mass"]),
            abi=str(params["schema"]),
            consumer_backend=str(params.get("consumer_backend", "host-python-integer")),
        )


def _require_integer_array(value: object, name: str, *, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer) or raw.dtype == np.bool_:
        raise ByteOutputError(f"{name} must be an integer array")
    if ndim is not None and raw.ndim != ndim:
        raise ByteOutputError(f"{name} must have rank {ndim}")
    return raw


@dataclass(frozen=True, slots=True)
class ByteMassPlane:
    """View of typed integer planes owned by one registered SPFBlockV3.

    The block's ``base.params`` contains the scalar header and immutable
    ``context_ids``, ``masses``, ``valid``, ``missing`` and optional
    ``stop_masses`` arrays.  This view carries no independent numerical copy;
    ``block`` is the only live carrier.
    """

    block: SPFBlockV3

    def __post_init__(self) -> None:
        if not isinstance(self.block, SPFBlockV3):
            raise ByteOutputError("mass plane requires a registered SPFBlockV3")
        if self.block.base.profile_id != int(BaseModelKind.BYTE_MASS_PLANE):
            raise ByteOutputError("block is not a registered BYTE_MASS_PLANE")
        try:
            validate_byte_mass_params(self.block.dims, self.block.base.params)
        except SPFError as exc:
            raise ByteOutputError(str(exc)) from exc

    @property
    def header(self) -> ByteMassHeader:
        return ByteMassHeader.from_block(self.block)

    @property
    def context_ids(self) -> np.ndarray:
        return self.block.base.params["context_ids"]

    @property
    def masses(self) -> np.ndarray:
        return self.block.base.params["masses"]

    @property
    def valid(self) -> np.ndarray:
        return self.block.base.params["valid"]

    @property
    def missing(self) -> np.ndarray:
        return self.block.base.params["missing"]

    @property
    def stop_masses(self) -> np.ndarray | None:
        return self.block.base.params.get("stop_masses")

    def to_block(self) -> SPFBlockV3:
        """Return the registered block that owns this view's planes."""

        return self.block

    def native_check(self) -> int:
        """Run the optional ABI-7 native layout/invariant guard."""

        from .native import validate_byte_mass_plane_native

        return validate_byte_mass_plane_native(self)

    def to_bytes(self) -> bytes:
        """Serialize the owning block through the immutable SPF_TB_v3 envelope."""

        from .v3_envelope import to_spftensor_v3

        return to_spftensor_v3(self.block)

    @classmethod
    def from_block(cls, block: SPFBlockV3) -> ByteMassPlane:
        """Construct a view over an already validated registered block."""

        return cls(block)

    @classmethod
    def from_bytes(cls, data: bytes) -> ByteMassPlane:
        """Deserialize one registered byte-mass block and return its view."""

        from .v3_envelope import from_spftensor_v3

        decoded = from_spftensor_v3(data)
        if isinstance(decoded, tuple):
            if len(decoded) != 1:
                raise ByteOutputError("byte mass envelope must contain exactly one block")
            decoded = decoded[0]
        return cls(decoded)

    @classmethod
    def from_masses(
        cls,
        header: ByteMassHeader,
        masses: object,
        *,
        context_ids: object | None = None,
        valid: object | None = None,
        missing: object | None = None,
        stop_masses: object | None = None,
    ) -> ByteMassPlane:
        """Construct a canonical typed plane from integer mass rows."""

        if not isinstance(header, ByteMassHeader):
            raise ByteOutputError("from_masses requires a ByteMassHeader")
        raw = _require_integer_array(masses, "masses")
        if raw.ndim == 2 and raw.shape == (header.node_count, 2):
            raw = raw[None, ...]
        expected_tail = (header.node_count, 2)
        if raw.ndim != 3 or raw.shape[1:] != expected_tail:
            raise ByteOutputError(f"masses must have shape (contexts, {expected_tail[0]}, 2)")
        if np.any(raw < 0) or np.any(raw.astype(object) > header.full_scale):
            raise ByteOutputError("mass is outside the exact Q-bit range")
        canonical = raw.astype(header.mass_dtype, copy=True)
        n_context = canonical.shape[0]
        ids = (
            np.arange(n_context, dtype=np.int64)
            if context_ids is None
            else _require_integer_array(context_ids, "context_ids", ndim=1)
        )
        valid_mask = (
            np.ones(n_context, dtype=np.bool_) if valid is None else _require_bool_mask(valid, (n_context,), "valid")
        )
        missing_mask = (
            np.zeros(n_context, dtype=np.bool_)
            if missing is None
            else _require_bool_mask(missing, (n_context,), "missing")
        )
        stop = None
        if stop_masses is not None:
            stop_raw = _require_integer_array(stop_masses, "stop_masses")
            expected = (n_context, header.max_output_length, 2)
            if stop_raw.shape != expected:
                raise ByteOutputError(f"stop_masses must have shape {expected}")
            if np.any(stop_raw < 0) or np.any(stop_raw.astype(object) > header.full_scale):
                raise ByteOutputError("stop mass is outside the exact Q-bit range")
            stop = stop_raw.astype(header.mass_dtype, copy=True)
        params = header.to_metadata()
        params.update(
            {
                "context_ids": ids,
                "masses": canonical,
                "valid": valid_mask,
                "missing": missing_mask,
                "stop_masses": stop,
            }
        )
        try:
            block = SPFBlockV3(
                dims=(n_context, header.node_count, 2),
                representation=RepresentationRef(int(Representation.STRUCTURAL_VIEW)),
                base=BaseModelRef(int(BaseModelKind.BYTE_MASS_PLANE), params),
                residual=ResidualRef(int(ResidualCodecKind.NONE), None),
            )
        except SPFError as exc:
            raise ByteOutputError(str(exc)) from exc
        return cls(block)

    @classmethod
    def from_scores(
        cls,
        header: ByteMassHeader,
        scores: object,
        rom: MassROM,
        *,
        context_ids: object | None = None,
        valid: object | None = None,
        missing: object | None = None,
        stop_scores: object | None = None,
        stop_rom: MassROM | None = None,
    ) -> ByteMassPlane:
        """Lower bounded integer scores through a total integer ``MassROM``.

        Out-of-range scores are deterministically clamped to the header's ROM
        bounds.  This is an explicit integer policy, never a floating response
        curve.  ``scores`` has shape ``(contexts, 255)`` (or ``(255,)`` for a
        single context), one score for each binary prefix node.
        """

        if not isinstance(header, ByteMassHeader) or not isinstance(rom, MassROM):
            raise ByteOutputError("from_scores requires a ByteMassHeader and MassROM")
        if rom.q != header.q or rom.rom_id != header.rom_id:
            raise ByteOutputError("MassROM identity or Q does not match the header")
        raw = _require_integer_array(scores, "scores")
        if raw.ndim == 1 and raw.shape == (header.node_count,):
            raw = raw[None, ...]
        if raw.ndim != 2 or raw.shape[1] != header.node_count:
            raise ByteOutputError(f"scores must have shape (contexts, {header.node_count})")
        clamped = np.clip(raw.astype(object), header.score_min, header.score_max)
        values = np.empty(raw.shape, dtype=header.mass_dtype)
        for index, score in np.ndenumerate(clamped):
            values[index] = rom.mass(int(score))
        full = header.full_scale
        masses = np.stack((full - values.astype(object), values.astype(object)), axis=-1)
        masses = masses.astype(header.mass_dtype)
        stop = None
        if stop_scores is not None:
            if header.stop_law != "two-mass":
                raise ByteOutputError("stop_scores require a two-mass stop law")
            if stop_rom is None:
                raise ByteOutputError("stop_scores require stop_rom")
            if (
                stop_rom.q != header.q
                or stop_rom.rom_id != header.rom_id
                or stop_rom.score_min != header.score_min
                or stop_rom.score_max != header.score_max
            ):
                raise ByteOutputError("stop ROM metadata does not exactly match the header")
            stop_raw = _require_integer_array(stop_scores, "stop_scores")
            expected = (raw.shape[0], header.max_output_length)
            if stop_raw.shape != expected:
                raise ByteOutputError(f"stop_scores must have shape {expected}")
            stop = np.empty((*expected, 2), dtype=header.mass_dtype)
            stop_clamped = np.clip(stop_raw.astype(object), header.score_min, header.score_max)
            for index, score in np.ndenumerate(stop_clamped):
                mass_one = stop_rom.mass(int(score))
                stop[(*index, 0)] = full - mass_one
                stop[(*index, 1)] = mass_one
        return cls.from_masses(
            header,
            masses,
            context_ids=context_ids,
            valid=valid,
            missing=missing,
            stop_masses=stop,
        )

    @property
    def context_count(self) -> int:
        return int(self.context_ids.size)

    def row_for_context(self, context: int) -> int:
        if isinstance(context, bool) or not isinstance(context, (int, np.integer)):
            raise ByteOutputError("context must be an integer id")
        context_id = int(context)
        if context_id < 0 or context_id >= self.header.context_limit:
            raise ByteOutputError("context is outside the declared context bound")
        rows = np.flatnonzero(self.context_ids == context_id)
        if rows.size != 1:
            raise ByteOutputError("context is not present in the mass plane")
        return int(rows[0])

    def state(self, context: int) -> str:
        """Return ``valid``, ``missing``, or ``invalid`` for one context."""

        row = self.row_for_context(context)
        if bool(self.missing[row]):
            return "missing"
        if not bool(self.valid[row]):
            return "invalid"
        return "valid"

    def sentinel(self, context: int) -> int | None:
        """Return the declared out-of-band sentinel for a non-valid context."""

        state = self.state(context)
        if state == "missing":
            return self.header.missing_sentinel
        if state == "invalid":
            return self.header.invalid_sentinel
        return None


@dataclass(frozen=True, slots=True)
class MassROM:
    """Total integer score-to-mass lookup table."""

    q: int
    score_min: int
    score_max: int
    values: np.ndarray
    rom_id: str = "mass-rom"

    def __post_init__(self) -> None:
        if isinstance(self.q, bool) or not isinstance(self.q, (int, np.integer)):
            raise ByteOutputError("MassROM q must be an integer")
        q = int(self.q)
        if q < 0 or q > _MAX_MASS_Q:
            raise ByteOutputError("MassROM q is outside the exact integer range")
        if not isinstance(self.score_min, (int, np.integer)) or isinstance(self.score_min, bool):
            raise ByteOutputError("MassROM score_min must be an integer")
        if not isinstance(self.score_max, (int, np.integer)) or isinstance(self.score_max, bool):
            raise ByteOutputError("MassROM score_max must be an integer")
        lo, hi = int(self.score_min), int(self.score_max)
        if lo > hi or lo < BYTE_MASS_SCORE_MIN or hi > BYTE_MASS_SCORE_MAX:
            raise ByteOutputError("MassROM score bounds are outside the native int32 range")
        values = _require_integer_array(self.values, "MassROM values", ndim=1)
        if values.size != hi - lo + 1:
            raise ByteOutputError("MassROM table length does not cover its declared score bounds")
        if np.any(values < 0) or np.any(values.astype(object) > (1 << q)):
            raise ByteOutputError("MassROM value is outside the Q-bit mass range")
        if not isinstance(self.rom_id, str) or not self.rom_id:
            raise ByteOutputError("MassROM rom_id must be non-empty")
        try:
            rom_bytes = self.rom_id.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ByteOutputError("MassROM rom_id must be valid UTF-8") from exc
        if len(rom_bytes) >= 64 or b"\x00" in rom_bytes:
            raise ByteOutputError("MassROM rom_id must fit a NUL-free native 63-byte identity field")
        values = _immutable_bytes_array(values, _mass_dtype(q))
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "score_min", lo)
        object.__setattr__(self, "score_max", hi)
        object.__setattr__(self, "values", values)

    @classmethod
    def constant(cls, header: ByteMassHeader, mass_one: int) -> MassROM:
        """Build a total constant ROM for mechanics and deterministic tests."""

        if isinstance(mass_one, bool) or not isinstance(mass_one, (int, np.integer)):
            raise ByteOutputError("constant mass must be an integer")
        mass = int(mass_one)
        if mass < 0 or mass > header.full_scale:
            raise ByteOutputError("constant mass is outside the Q-bit range")
        values = np.full(
            header.score_max - header.score_min + 1,
            mass,
            dtype=header.mass_dtype,
        )
        return cls(header.q, header.score_min, header.score_max, values, header.rom_id)

    def mass(self, score: int) -> int:
        if isinstance(score, bool) or not isinstance(score, (int, np.integer)):
            raise ByteOutputError("MassROM score must be an integer")
        value = int(score)
        if value < self.score_min or value > self.score_max:
            raise ByteOutputError("MassROM score is outside its declared bounds")
        return int(self.values[value - self.score_min])


@dataclass(frozen=True, slots=True)
class ByteHeadResult:
    """Output and integer trace from a conditional byte head."""

    data: bytes
    trace: tuple[Mapping[str, int | bool | str | None], ...]
    counters: Mapping[str, int]

    @property
    def bytes(self) -> bytes:
        return self.data

    @property
    def payload(self) -> bytes:
        return self.data


def _node_index(depth: int, prefix: int) -> int:
    if not 0 <= int(depth) < 8:
        raise ByteOutputError("byte decision depth must be in [0,7]")
    count = 1 << int(depth)
    if not 0 <= int(prefix) < count:
        raise ByteOutputError("byte prefix is outside the current tree level")
    return count - 1 + int(prefix)


def _bit_for_byte(byte: int, depth: int, bit_order: BitOrder) -> int:
    shift = 7 - depth if bit_order == "msb0" else depth
    return (int(byte) >> shift) & 1


def _choose_bit(mass_zero: int, mass_one: int, *, mode: str, draw: int | None, full: int) -> int:
    if mass_zero < 0 or mass_one < 0 or mass_zero + mass_one != full:
        raise ByteOutputError("conditional sibling masses do not sum to the declared full scale")
    if mode == "sample":
        if draw is None or isinstance(draw, bool) or not isinstance(draw, (int, np.integer)):
            raise ByteOutputError("sample mode requires an integer draw")
        value = int(draw)
        if value < 0 or value >= full:
            raise ByteOutputError("integer draw is outside [0, 2**Q)")
        return int(value >= mass_zero)
    if mode == "zero":
        # Choose zero whenever it has support; this makes a zero mass distinct
        # from missing/invalid state while remaining a valid deterministic law.
        return 0 if mass_zero > 0 else 1
    if mode in ("full", "one"):
        return 1 if mass_one > 0 else 0
    raise ByteOutputError("mode must be 'sample', 'zero', or 'full'")


def _require_draw(value: object) -> int:
    """Accept Python/NumPy integers while rejecting lossy coercions."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ByteOutputError("integer draws must be Python or NumPy integers")
    return int(value)


@dataclass(frozen=True, slots=True)
class ConditionalByteHead:
    """Factorised exact-integer conditional byte consumer.

    The ``plane`` is a proposed typed mass plane, not a registered SPF block.
    All serving methods are donor-free and target-free: they consume only the
    selected context and its declared integer masses.
    """

    plane: ByteMassPlane

    def __post_init__(self) -> None:
        if not isinstance(self.plane, ByteMassPlane):
            raise ByteOutputError("ConditionalByteHead requires a ByteMassPlane")

    @property
    def header(self) -> ByteMassHeader:
        return self.plane.header

    def _row_and_status(self, context: int) -> tuple[int, str]:
        row = self.plane.row_for_context(context)
        status = self.plane.state(context)
        if status != "valid":
            sentinel = self.plane.sentinel(context)
            raise ByteOutputError(f"context {int(context)} is {status} (sentinel={sentinel})")
        return row, status

    def mass_pair(self, context: int, depth: int, prefix: int) -> tuple[int, int]:
        """Read one exact sibling pair for a context/prefix."""

        row, _ = self._row_and_status(context)
        node = _node_index(depth, prefix)
        pair = self.plane.masses[row, node]
        return int(pair[0]), int(pair[1])

    def probability_numerator(self, byte: int, context: int = 0) -> int:
        """Return the exact numerator for one byte over denominator ``2**(8Q)``."""

        if isinstance(byte, bool) or not isinstance(byte, (int, np.integer)):
            raise ByteOutputError("byte must be an integer")
        value = int(byte)
        if value < 0 or value > 255:
            raise ByteOutputError("byte must be in [0,255]")
        numerator = 1
        prefix = 0
        for depth in range(8):
            mass_zero, mass_one = self.mass_pair(context, depth, prefix)
            bit = _bit_for_byte(value, depth, self.header.bit_order)
            numerator *= mass_one if bit else mass_zero
            prefix = (prefix << 1) | bit
        return int(numerator)

    def probability_denominator(self) -> int:
        """Return the exact common denominator for one byte."""

        return self.header.full_scale**8

    def all_probability_numerators(self, context: int = 0) -> tuple[int, ...]:
        """Return exact numerators for all 256 byte leaves."""

        return tuple(self.probability_numerator(byte, context) for byte in range(256))

    def probability_total(self, context: int = 0) -> tuple[int, int]:
        """Return ``(sum leaf numerators, common denominator)`` exactly."""

        values = self.all_probability_numerators(context)
        return int(sum(values)), self.probability_denominator()

    def validate_probability_law(self, context: int = 0) -> dict[str, int | bool]:
        """Audit exact sibling sums and byte-leaf normalization."""

        row, _ = self._row_and_status(context)
        full = self.header.full_scale
        sibling_pairs = self.plane.masses[row].astype(object).sum(axis=-1)
        sibling_ok = bool(np.all(sibling_pairs == full))
        numerator, denominator = self.probability_total(context)
        return {
            "context": int(context),
            "full_scale": full,
            "sibling_pairs": self.header.node_count,
            "sibling_sum_exact": sibling_ok,
            "leaf_numerator_sum": numerator,
            "leaf_denominator": denominator,
            "normalized_exact": numerator == denominator,
        }

    def sample_byte(
        self,
        context: int = 0,
        *,
        mode: Literal["sample", "zero", "full", "one"] = "zero",
        draws: Sequence[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Select one byte with integer draws or a support-respecting mode.

        ``mode='zero'`` and ``mode='full'`` are deterministic controls.  They
        select the preferred branch when it has nonzero mass and fall back to
        the sibling when it is forced to zero.  ``mode='sample'`` compares an
        integer draw in ``[0,2**Q)`` against the zero mass.
        """

        self._row_and_status(context)
        if mode not in ("sample", "zero", "full", "one"):
            raise ByteOutputError("mode must be 'sample', 'zero', 'full', or 'one'")
        if mode == "sample":
            if draws is not None and len(draws) != 8:
                raise ByteOutputError("sample draws must contain exactly eight integers")
            if draws is None and rng is None:
                raise ByteOutputError("sample mode requires draws or an integer RNG")
        elif draws is not None or rng is not None:
            raise ByteOutputError("draws/rng are only valid in sample mode")
        draw_values: list[int | None]
        if mode == "sample" and draws is not None:
            draw_values = [_require_draw(value) for value in draws]
        elif mode == "sample":
            assert rng is not None
            draw_values = [int(rng.integers(0, self.header.full_scale)) for _ in range(8)]
        else:
            draw_values = [None] * 8
        prefix = 0
        value = 0
        for depth in range(8):
            mass_zero, mass_one = self.mass_pair(context, depth, prefix)
            bit = _choose_bit(
                mass_zero,
                mass_one,
                mode="sample" if mode == "sample" else ("full" if mode in ("full", "one") else "zero"),
                draw=draw_values[depth],
                full=self.header.full_scale,
            )
            value |= bit << (7 - depth if self.header.bit_order == "msb0" else depth)
            prefix = (prefix << 1) | bit
        return value

    def _walk_byte(
        self,
        context: int,
        *,
        mode: Literal["sample", "zero", "full", "one"],
        draws: Sequence[int] | None,
        rng: np.random.Generator | None,
        byte_index: int,
    ) -> tuple[int, list[dict[str, int | bool | str | None]], int]:
        self._row_and_status(context)
        if mode not in ("sample", "zero", "full", "one"):
            raise ByteOutputError("mode must be 'sample', 'zero', 'full', or 'one'")
        if mode == "sample":
            if draws is not None and len(draws) != 8:
                raise ByteOutputError("sample draws must contain exactly eight integers")
            if draws is None and rng is None:
                raise ByteOutputError("sample mode requires draws or an integer RNG")
            draw_values: list[int | None] = (
                [_require_draw(value) for value in draws]
                if draws is not None
                else [int(rng.integers(0, self.header.full_scale)) for _ in range(8)]  # type: ignore[union-attr]
            )
        else:
            if draws is not None or rng is not None:
                raise ByteOutputError("draws/rng are only valid in sample mode")
            draw_values = [None] * 8
        prefix = 0
        value = 0
        trace: list[dict[str, int | bool | str | None]] = []
        for depth in range(8):
            node = _node_index(depth, prefix)
            mass_zero, mass_one = self.mass_pair(context, depth, prefix)
            bit = _choose_bit(
                mass_zero,
                mass_one,
                mode="sample" if mode == "sample" else ("full" if mode in ("full", "one") else "zero"),
                draw=draw_values[depth],
                full=self.header.full_scale,
            )
            value |= bit << (7 - depth if self.header.bit_order == "msb0" else depth)
            trace.append(
                {
                    "kind": "byte-bit",
                    "byte_index": int(byte_index),
                    "depth": depth,
                    "node": node,
                    "prefix": prefix,
                    "mass_zero": mass_zero,
                    "mass_one": mass_one,
                    "draw": draw_values[depth],
                    "bit": bit,
                }
            )
            prefix = (prefix << 1) | bit
        return value, trace, 8 if mode == "sample" else 0

    def emit_byte(
        self,
        context: int = 0,
        *,
        mode: Literal["sample", "zero", "full", "one"] = "zero",
        draws: Sequence[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Emit one byte as an integer, with no string/label lookup."""

        return self.sample_byte(context, mode=mode, draws=draws, rng=rng)

    def emit(
        self,
        context: int = 0,
        *,
        mode: Literal["sample", "zero", "full", "one"] = "zero",
        max_length: int | None = None,
        draws: Iterable[Sequence[int]] | None = None,
        stop_draws: Sequence[int] | None = None,
        rng: np.random.Generator | None = None,
    ) -> ByteHeadResult:
        """Generate a finite byte string under the declared stop law.

        Every result is bounded by ``header.max_output_length``.  A
        ``max-length`` header forces termination at that bound.  A ``two-mass``
        header makes an integer stop decision after each byte and still forces
        termination at the bound, so even a zero stop mass cannot create an
        infinite serving loop.
        """

        self._row_and_status(context)
        limit = self.header.max_output_length if max_length is None else max_length
        if isinstance(limit, bool) or not isinstance(limit, (int, np.integer)):
            raise ByteOutputError("max_length must be an integer")
        limit = int(limit)
        if limit <= 0 or limit > self.header.max_output_length:
            raise ByteOutputError("max_length is outside the declared finite output bound")
        if stop_draws is not None and len(stop_draws) > limit:
            raise ByteOutputError("stop_draws exceeds the finite output bound")
        if self.header.stop_law == "two-mass" and self.plane.stop_masses is None:
            raise ByteOutputError("two-mass stop law has no stop mass plane")
        draw_iter = None if draws is None else iter(draws)
        trace: list[Mapping[str, int | bool | str | None]] = []
        counters = {
            "contexts": 1,
            "mass_reads": 0,
            "integer_comparisons": 0,
            "random_draws": 0,
            "bytes_emitted": 0,
            "stop_decisions": 0,
            "forced_stops": 0,
        }
        output = bytearray()
        for byte_index in range(limit):
            byte_draws = None if draw_iter is None else next(draw_iter, None)
            if mode == "sample" and byte_draws is None and rng is None:
                raise ByteOutputError("sample mode requires one eight-draw row per byte")
            value, byte_trace, draws_used = self._walk_byte(
                context,
                mode=mode,
                draws=byte_draws,
                rng=rng,
                byte_index=byte_index,
            )
            output.append(value)
            trace.extend(byte_trace)
            counters["mass_reads"] += 8
            counters["integer_comparisons"] += 8
            counters["random_draws"] += draws_used
            counters["bytes_emitted"] += 1
            stop = False
            forced = byte_index + 1 >= limit
            if self.header.stop_law == "two-mass" and not forced:
                row = self.plane.row_for_context(context)
                assert self.plane.stop_masses is not None
                mass_zero, mass_one = (int(v) for v in self.plane.stop_masses[row, byte_index])
                if mode == "sample":
                    if stop_draws is not None and byte_index < len(stop_draws):
                        stop_draw = _require_draw(stop_draws[byte_index])
                    elif rng is not None:
                        stop_draw = int(rng.integers(0, self.header.full_scale))
                    else:
                        raise ByteOutputError("sample stop law requires stop_draws or an integer RNG")
                    stop_mode = "sample"
                    counters["random_draws"] += 1
                else:
                    stop_draw = None
                    stop_mode = "full" if mode in ("full", "one") else "zero"
                stop_bit = _choose_bit(
                    mass_zero,
                    mass_one,
                    mode=stop_mode,
                    draw=stop_draw,
                    full=self.header.full_scale,
                )
                stop = stop_bit == 1
                counters["stop_decisions"] += 1
                counters["integer_comparisons"] += 1
                trace.append(
                    {
                        "kind": "stop",
                        "byte_index": byte_index,
                        "mass_zero": mass_zero,
                        "mass_one": mass_one,
                        "draw": stop_draw,
                        "stop": stop,
                        "forced": False,
                    }
                )
            if forced:
                counters["forced_stops"] += 1
                trace.append({"kind": "stop", "byte_index": byte_index, "stop": True, "forced": True})
            if stop:
                break
        return ByteHeadResult(bytes(output), tuple(trace), counters)

    def generate(self, context: int = 0, **kwargs: object) -> ByteHeadResult:
        """Alias for :meth:`emit` used by generation-oriented callers."""

        return self.emit(context, **kwargs)  # type: ignore[arg-type]


class _ByteMassPlaneModel:
    """Registry entry for the typed header-owned byte mass profile.

    This profile intentionally has no scalar ``reconstruct_planes`` path.  Its
    consumer is :class:`ConditionalByteHead`, which reads the registered
    integer arrays directly through :class:`ByteMassPlane`.
    """

    kind = int(BaseModelKind.BYTE_MASS_PLANE)

    def reconstruct_planes(
        self,
        params: Mapping[str, object],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise SPFError("BYTE_MASS_PLANE has typed integer masses, not scalar SPF planes")

    def storage_bits(self, params: Mapping[str, object]) -> int:
        p = validate_byte_mass_params((int(np.asarray(params["masses"]).shape[0]), _BYTE_NODE_COUNT, 2), params)
        masses = np.asarray(p["masses"])
        bits = masses.nbytes * 8
        for name in ("context_ids", "valid", "missing", "stop_masses"):
            value = p.get(name)
            if isinstance(value, np.ndarray):
                bits += value.nbytes * 8
        return int(bits)


# Register the profile with the v3 registry.  The registered SPFBlockV3 is the
# sole live carrier; this module only supplies the typed host consumer.
register_base_model(_ByteMassPlaneModel())


__all__ = [
    "BitOrder",
    "ByteHeadResult",
    "ByteMassHeader",
    "ByteMassPlane",
    "ByteOutputError",
    "ConditionalByteHead",
    "CopyResult",
    "MassROM",
    "StopLaw",
    "emit_copy",
    "pack_utf8_bytes",
]
