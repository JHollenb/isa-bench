"""Phase-carried exposure and packed-record transport, SPFTensor v3.1.

Live payloads are SPF32 words. Exposure is a unit-circle SPF64 word plus an
explicit winding in the serialized base header; its tick has no attenuation
meaning. A rational law maps WHOLE elapsed spans to log codes at read time.
Persistent state contains no Cartesian coordinates. The consumer is a packed
pairwise reducer, not the AUTO Cartesian fallback. Retention is O(history).
"""

from __future__ import annotations

import numpy as np

from spf.scalar._constants import SPF64

from .core import SPF32_L_MIN_FINITE, SPF32_WORD_ZERO, SPFError, pack_words, unpack_L, unpack_theta
from .v3 import BaseModelKind, BaseModelRef, SPFBlockV3, register_base_model

PROFILE = int(BaseModelKind.PHASE_CLOCK_RECORDS)
CLOCK_MOD = 1 << 32
CLOCK_LIMIT = (1 << 62) - 1


def integer_plane(value, name, *, low=0, high=CLOCK_LIMIT):
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < low) or np.any(raw > high):
        raise SPFError(f"{name} must contain integers in [{low}, {high}]")
    return raw.astype(np.int64)


def validate_clock_block(block, *, streaming=False):
    from .sequence_mixer import _words
    if not isinstance(block, SPFBlockV3) or block.base.profile_id != PROFILE:
        raise SPFError("state must be a PHASE_CLOCK_RECORDS SPFBlockV3")
    p = block.base.params
    required = {
        "payload_words",
        "exposure_words",
        "exposure_winding",
        "rotation_words",
        "segments",
        "attenuation_numerator",
        "attenuation_denominator",
        "position",
        "consumer",
        "record_abi",
    }
    if streaming:
        required |= {
            "record_positions",
            "current_exposure_words",
            "current_exposure_winding",
            "current_rotation",
            "current_segment",
        }
    if required - set(p):
        raise SPFError(f"incomplete clock record header: {sorted(required - set(p))}")
    words = _words(p["payload_words"], "clock record payload", 3)
    if words.dtype != np.uint32 or words.shape != block.dims or words.ndim < 3:
        raise SPFError("invalid packed record payload shape/dtype")
    ticks = phase_clock_ticks(p["exposure_words"], p["exposure_winding"])
    rotation = np.asarray(p["rotation_words"])
    if (
        ticks.shape != words.shape[:-1]
        or rotation.shape != ticks.shape
        or rotation.dtype != np.uint64
        or np.any(unpack_L(rotation, SPF64) != 0)
    ):
        raise SPFError("invalid per-record phase clocks")
    if integer_plane(p["segments"], "segments").shape != words.shape[:-2]:
        raise SPFError("one segment required per record")
    position = integer_plane(p["position"], "position")
    if position.ndim != 0 or p["record_abi"] != 1 or p["consumer"] != "factored-integer-candidate":
        raise SPFError("unsupported clock record ABI/consumer")
    for name, low, high in (("attenuation_numerator", -CLOCK_LIMIT, 0), ("attenuation_denominator", 1, CLOCK_LIMIT)):
        a = integer_plane(p[name], name, low=low, high=high)
        try:
            np.broadcast_to(a, (*words.shape[:-3], words.shape[-2]))
        except ValueError:
            raise SPFError("attenuation law must broadcast to batch/state shape") from None
    if "record_positions" in p:
        positions = integer_plane(p["record_positions"], "record_positions")
        if positions.shape != words.shape[:-2] or np.any(positions >= int(position)):
            raise SPFError("record positions must precede the stream position")
    if streaming:
        current = phase_clock_ticks(p["current_exposure_words"], p["current_exposure_winding"])
        shape = (*words.shape[:-3], words.shape[-2])
        if (
            current.shape != shape
            or integer_plane(p["current_rotation"], "current_rotation", high=CLOCK_MOD - 1).shape != shape
        ):
            raise SPFError("current clock has wrong shape")
        if integer_plane(p["current_segment"], "current_segment").shape != words.shape[:-3]:
            raise SPFError("current segment has wrong shape")
    return p


def phase_clock_words(ticks):
    """Canonical unit-circle SPF64 clock words and exact winding metadata."""
    raw = np.asarray(ticks)
    if not np.issubdtype(raw.dtype, np.integer) or np.any(raw < 0) or np.any(raw > CLOCK_LIMIT):
        raise SPFError("exposure ticks must be integers in [0, 2^62-1]")
    ticks = raw.astype(np.int64)
    return pack_words(np.zeros(ticks.shape, np.int64), ticks % CLOCK_MOD, SPF64), ticks // CLOCK_MOD


def phase_clock_ticks(words, winding):
    words = np.asarray(words)
    winding = np.asarray(winding)
    if words.dtype != np.uint64 or not np.issubdtype(winding.dtype, np.integer):
        raise SPFError("clock needs canonical SPF64 words and integer winding")
    if words.shape != winding.shape or np.any(unpack_L(words, SPF64) != 0):
        raise SPFError("clock word/winding shapes must agree; clock magnitude must be one")
    if np.any(winding < 0) or np.any(winding > CLOCK_LIMIT // CLOCK_MOD):
        raise SPFError("clock winding outside exact int64 range")
    ticks = winding.astype(np.int64) * CLOCK_MOD + unpack_theta(words, SPF64).astype(np.int64)
    if np.any(ticks > CLOCK_LIMIT):
        raise SPFError("clock exceeds exact int64 range")
    return ticks


def round_ratio(numerator, denominator):
    """Integer RNE division, including negative numerators; no floating arithmetic."""
    n, d = np.broadcast_arrays(np.asarray(numerator), np.asarray(denominator))
    if not np.issubdtype(d.dtype, np.integer) or np.any(d <= 0):
        raise SPFError("rational denominator must be a positive integer")
    if n.dtype != object and not np.issubdtype(n.dtype, np.integer):
        raise SPFError("rational numerator must be an integer")
    if n.dtype == object and any(not isinstance(v, (int, np.integer)) for v in n.flat):
        raise SPFError("rational numerator must contain integers")
    # Object arithmetic prevents silent int64 overflow before the division.
    no, do = n.astype(object), d.astype(object)
    q, r = np.floor_divide(no, do), np.remainder(no, do)
    q, r = np.asarray(q, dtype=object), np.asarray(r, dtype=object)
    q += (2 * r > d) | ((2 * r == d) & (q % 2 != 0))
    if any(abs(int(v)) > CLOCK_LIMIT for v in q.flat):
        raise SPFError("rational result exceeds exact int64 range")
    return q.astype(np.int64)


def packed_pairwise(terms, axis=-1, *, profile="canonical"):
    """Existing SPF32 LOGPOLAR reducer, without statistics or Cartesian routing."""
    from .reducers import _logpolar_reduce_rows

    words = np.asarray(terms)
    if words.dtype != np.uint32:
        raise SPFError("packed reducer requires SPF32 word planes")
    moved = np.moveaxis(words, axis, -1)
    if moved.shape[-1] == 0:
        return np.full(moved.shape[:-1], SPF32_WORD_ZERO, np.uint32)
    rows = moved.reshape(-1, moved.shape[-1])
    keep = unpack_L(rows) != -32768
    if profile not in ("canonical", "factored-integer-candidate"):
        raise SPFError("unknown packed join profile")
    out = _logpolar_reduce_rows(rows, keep, integer_only=profile == "factored-integer-candidate")
    return out.reshape(moved.shape[:-1])


class PhaseClockRecordsModel:
    """Sparse record sampler: reconstruct means PAYLOAD, never a mixed clock plane."""

    kind = PROFILE

    def reconstruct_planes(self, params, dims, origin):
        indices = np.arange(int(np.prod(dims)), dtype=np.int64)
        return indices, *self.sample_planes(params, dims, origin, indices)

    def sample_planes(self, params, dims, origin, active, *, phase=True, logmag=True):
        indices = np.asarray(active, dtype=np.int64)
        words = np.asarray(params["payload_words"], dtype=np.uint32).reshape(-1)[indices]
        return unpack_L(words).astype(np.int64), (
            unpack_theta(words).astype(np.int64) if phase else np.zeros(words.shape, np.int64)
        )

    def storage_bits(self, params):
        def charge(value):
            if isinstance(value, np.ndarray):
                return value.nbytes * 8
            if isinstance(value, dict):
                return sum(charge(v) for v in value.values())
            if isinstance(value, (int, np.integer)):
                return 64
            if isinstance(value, str):
                return len(value.encode()) * 8
            raise SPFError("unaccounted clock-record metadata type")

        return charge(params)


register_base_model(PhaseClockRecordsModel())


def clocked_record_block(payload_words, exposure_ticks, rotation_ticks, segments, *, numerator, denominator, position):
    from .sequence_mixer import _words

    words = _words(payload_words, "clock record payload", 3)
    if words.dtype != np.uint32 or words.ndim < 3:
        raise SPFError("records require packed (..., records, states, payload) SPF32 words")
    e = integer_plane(exposure_ticks, "exposure_ticks")
    r = integer_plane(rotation_ticks, "rotation_ticks", high=CLOCK_MOD - 1)
    if e.shape != words.shape[:-1] or r.shape != e.shape:
        raise SPFError("one exposure and rotation clock is required per record/state")
    ew, winding = phase_clock_words(e)
    rw = pack_words(np.zeros(r.shape, np.int64), r.astype(np.uint64) % CLOCK_MOD, SPF64)
    num = np.asarray(numerator)
    den = np.asarray(denominator)
    if not np.issubdtype(num.dtype, np.integer) or np.any(num > 0):
        raise SPFError("attenuation numerator must be a nonpositive integer")
    round_ratio(np.zeros_like(num), den)
    segment = integer_plane(segments, "segments")
    pos = integer_plane(position, "position")
    if segment.shape != words.shape[:-2] or pos.ndim != 0:
        raise SPFError("invalid record segment/position shape")
    block = SPFBlockV3(
        dims=words.shape,
        base=BaseModelRef(
            PROFILE,
            {
                "payload_words": words.copy(),
                "exposure_words": ew,
                "exposure_winding": winding,
                "rotation_words": rw,
                "segments": segment.copy(),
                "attenuation_numerator": num.astype(np.int64),
                "attenuation_denominator": den.astype(np.int64),
                "position": int(position),
                "clock_tick_semantics": "exposure-not-attenuation",
                "consumer": "factored-integer-candidate",
                "record_abi": 1,
                "numerical_status": "candidate-ROM-pairwise; arbitrary-fold error not certified",
            },
        ),
    )
    validate_clock_block(block)
    return block


def read_clocked_records(
    block,
    query_ticks,
    query_rotation,
    query_segments,
    query_positions,
    c_words,
    *,
    chunk=32,
    record_mask=None,
    record_gains=None,
    counters=None,
):
    """Strict causal reads; transport each retained word once for this consumer.

    Rounding contract: SPF32 transport words followed by balanced packed sums.
    This differs from both packed-step and wide-reference accumulation.
    """
    if not 1 <= int(chunk) <= 256:
        raise SPFError("read chunk must be in [1,256]")
    p = validate_clock_block(block)
    from .sequence_mixer import _words
    wt = phase_clock_ticks(p["exposure_words"], p["exposure_winding"])
    wr = unpack_theta(p["rotation_words"], SPF64).astype(np.int64)
    qt = integer_plane(query_ticks, "query_ticks")
    qr = integer_plane(query_rotation, "query_rotation", high=CLOCK_MOD - 1)
    c = _words(c_words, "clock read gain")
    if c.dtype != np.uint32 or c.shape != qt.shape or qr.shape != qt.shape:
        raise SPFError("read clock and SPF32 gains must share (..., queries, states)")
    words = p["payload_words"]
    batch, records, states, payload = words.shape[:-3], *words.shape[-3:]
    if qt.ndim < 2 or qt.shape[:-2] != batch or qt.shape[-1] != states:
        raise SPFError("read batch/state shape differs from records")
    qs = integer_plane(query_segments, "query_segments")
    qp = integer_plane(query_positions, "query_positions")
    if qs.shape != qt.shape[:-1] or qp.shape != qs.shape:
        raise SPFError("one query position/segment is required per query")
    if record_mask is not None:
        rm = np.asarray(record_mask)
        if rm.dtype != np.bool_ or rm.shape != (*qs.shape, records):
            raise SPFError("record mask must be a query/record boolean plane")
    if record_gains is not None:
        rg = _words(record_gains, "clock record gain")
        if rg.dtype != np.uint32 or rg.shape != (*qs.shape, records):
            raise SPFError("record gains must be a query/record SPF32 plane")
    from .core import spf_mul_words

    output = np.full((*batch, qt.shape[-2], payload), SPF32_WORD_ZERO, np.uint32)
    for start in range(0, qt.shape[-2], int(chunk)):
        stop = min(start + int(chunk), qt.shape[-2])
        age = qt[..., start:stop, None, :] - wt[..., None, :, :]
        law_shape = (*batch, states)
        law_num = np.broadcast_to(p["attenuation_numerator"], law_shape)[..., None, None, :]
        denominator = np.broadcast_to(p["attenuation_denominator"], law_shape)[..., None, None, :]
        numerator = age.astype(object) * law_num * 256
        # Explicit finite saturation BEFORE int64 conversion of the quotient.
        numerator = np.maximum(numerator, SPF32_L_MIN_FINITE * denominator.astype(object))
        numerator = np.minimum(numerator, 0)
        logs = round_ratio(numerator, denominator)
        phases = round_ratio((qr[..., start:stop, None, :] - wr[..., None, :, :]) % CLOCK_MOD, 65536) % 65536
        transport = pack_words(np.clip(logs, SPF32_L_MIN_FINITE, 0).astype(np.int16), phases.astype(np.uint16))
        terms = spf_mul_words(words[..., None, :, :, :], transport[..., None])
        terms = spf_mul_words(terms, c[..., start:stop, None, :, None])
        if record_gains is not None:
            terms = spf_mul_words(terms, np.asarray(record_gains)[..., start:stop, :, None, None])
        record_positions = np.asarray(p.get("record_positions", np.arange(records)), np.int64)
        live = record_positions[..., None, :] < qp[..., start:stop, None]
        live &= p["segments"][..., None, :] == qs[..., start:stop, None]
        if record_mask is not None:
            live &= np.asarray(record_mask)[..., start:stop, :]
        terms = np.where(live[..., None, None], terms, SPF32_WORD_ZERO).astype(np.uint32)
        # Flatten only the reduction axes; payload and batch stay explicit.
        flat = np.moveaxis(terms, -1, -3).reshape(*batch, stop - start, payload, records * states)
        if counters is not None:
            counters.word_multiplications += terms.size * (2 + (record_gains is not None))
            counts = np.count_nonzero(unpack_L(flat) != -32768, axis=-1)
            counters.coherent_additions += int(np.maximum(counts - 1, 0).sum())
            counters.finalizations += int(np.maximum(counts - 1, 0).sum())
        output[..., start:stop, :] = packed_pairwise(flat, profile=p["consumer"])
    return output
