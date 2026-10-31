"""Canonical packed-record memory with phase-carried exposure clocks.

The live carrier is SPFBlockV3, not a private Cartesian accumulator. Attenuation
is an independent rational law applied to whole spans. Old Q16 inputs are an
explicit quantized ingress contract; use exposure_ticks and attenuation_* for
rates smaller than one Q16 step. Retention is O(T), reads O(T squared).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import SPF32_WORD_ZERO, SPFError, spf_mul_words
from .phase_clock import (
    CLOCK_LIMIT,
    clocked_record_block,
    packed_pairwise,
    phase_clock_ticks,
    phase_clock_words,
    read_clocked_records,
    validate_clock_block,
)
from .sequence_mixer import _resets, _words
from .v3 import SPFBlockV3

ClockedState = SPFBlockV3


@dataclass(frozen=True)
class ClockedDirectionParameters:
    log_q16: np.ndarray
    phase_q32: np.ndarray
    b: np.ndarray
    c: np.ndarray
    exposure_ticks: np.ndarray | None = None
    attenuation_numerator: object = -1
    attenuation_denominator: object = 65536


@dataclass(frozen=True)
class ClockedStateParameters:
    forward: ClockedDirectionParameters
    backward: ClockedDirectionParameters | None = None
    diagonal: np.uint32 = SPF32_WORD_ZERO


@dataclass
class ClockedStateCounters:
    word_multiplications: int = 0
    coherent_additions: int = 0
    finalizations: int = 0
    integer_additions: int = 0
    integer_multiplies: int = 0
    rom_reads: int = 0
    rebases: int = 0


@dataclass(frozen=True)
class ClockedStateResult:
    words: np.ndarray
    forward_state: SPFBlockV3
    backward_state: SPFBlockV3 | None
    counters: ClockedStateCounters = field(default_factory=ClockedStateCounters)


def _prefix(delta, initial):
    if np.any(delta < 0) or any(int(v) > CLOCK_LIMIT for v in delta.flat):
        raise SPFError("exposure increments outside exact nonnegative range")
    # Check before int64 cumsum can overflow.
    total = np.sum(delta.astype(object), axis=-2) + initial.astype(object)
    if any(int(v) > CLOCK_LIMIT for v in total.flat):
        raise SPFError("phase exposure clock overflow")
    return initial[..., None, :] + np.cumsum(delta, axis=-2, dtype=np.int64)


def _scan(values, direction, reset, state, chunk, counters, *, materialize_output=True, return_queries=False):
    b, c = _words(direction.b, "b", 2), _words(direction.c, "c", 2)
    raw_log, raw_rotation = np.asarray(direction.log_q16), np.asarray(direction.phase_q32)
    if b.shape != c.shape or raw_log.shape != b.shape or raw_rotation.shape != b.shape:
        raise SPFError("clock/gain planes must share (..., T, N)")
    if values.shape[:-1] != b.shape[:-1] or b.shape[-1] < 1:
        raise SPFError("values and clock planes must share batch/time axes")
    if not np.issubdtype(raw_log.dtype, np.integer) or np.any(raw_log > 0):
        raise SPFError("legacy log_q16 must be nonpositive integer codes")
    if not np.issubdtype(raw_rotation.dtype, np.integer):
        raise SPFError("rotation must be integer Q32 phase ticks")
    delta = -raw_log.astype(object) if direction.exposure_ticks is None else np.asarray(direction.exposure_ticks)
    if delta.shape != b.shape or (delta.dtype != object and not np.issubdtype(delta.dtype, np.integer)):
        raise SPFError("exposure_ticks must be integer clock/gain shape")
    if any(not isinstance(v, (int, np.integer)) or v < 0 or v > CLOCK_LIMIT for v in delta.flat):
        raise SPFError("invalid exposure increment")
    delta = delta.astype(np.int64)
    batch, length, n, payload = b.shape[:-2], b.shape[-2], b.shape[-1], values.shape[-1]
    if state is not None and not isinstance(state, SPFBlockV3):
        raise SPFError("initial_state must be an SPFBlockV3")
    if state is not None:
        validate_clock_block(state, streaming=True)
    position = 0 if state is None else int(state.base.params["position"])
    zeros = np.zeros((*batch, n), np.int64)
    old = {} if state is None else state.base.params
    if state is not None and (
        not isinstance(state, SPFBlockV3) or state.dims[:-3] != batch or state.dims[-2:] != (n, payload)
    ):
        raise SPFError("initial_state must be a matching packed clock-record block")
    initial = (
        zeros if state is None else phase_clock_ticks(old["current_exposure_words"], old["current_exposure_winding"])
    )
    rotation = zeros if state is None else old["current_rotation"]
    segment = np.zeros(batch, np.int64) if state is None else old["current_segment"]
    ticks = _prefix(delta, initial)
    rotations = (rotation[..., None, :] + np.cumsum(raw_rotation.astype(np.int64) & 0xFFFFFFFF, axis=-2)) & 0xFFFFFFFF
    segments = segment[..., None] + np.cumsum(reset.astype(np.int64), axis=-1)
    writes = spf_mul_words(b[..., None], values[..., None, :])
    wt, wr, seg, pos = (
        ticks,
        rotations,
        segments,
        np.broadcast_to(np.arange(position, position + length), (*batch, length)),
    )
    if state is not None:
        for key, value in (
            ("attenuation_numerator", direction.attenuation_numerator),
            ("attenuation_denominator", direction.attenuation_denominator),
        ):
            if not np.array_equal(old[key], np.asarray(value)):
                raise SPFError("stream attenuation law cannot change")
        writes = np.concatenate((old["payload_words"], writes), axis=-3)
        wt = np.concatenate((phase_clock_ticks(old["exposure_words"], old["exposure_winding"]), wt), axis=-2)
        from spf.scalar._constants import SPF64

        from .core import unpack_theta

        wr = np.concatenate((unpack_theta(old["rotation_words"], SPF64).astype(np.int64), wr), axis=-2)
        seg = np.concatenate((old["segments"], seg), axis=-1)
        pos = np.concatenate((old["record_positions"], pos), axis=-1)
    from .core import SPF32_L_ZERO, unpack_L

    keep = np.any(unpack_L(writes) != SPF32_L_ZERO, axis=(*tuple(range(len(batch))), -2, -1))
    writes, wt, wr = writes[..., keep, :, :], wt[..., keep, :], wr[..., keep, :]
    seg, pos = seg[..., keep], pos[..., keep]
    final = clocked_record_block(
        writes,
        wt,
        wr,
        seg,
        numerator=direction.attenuation_numerator,
        denominator=direction.attenuation_denominator,
        position=position + length,
    )
    params = final.base.params
    params["record_positions"] = pos
    end = initial if length == 0 else ticks[..., -1, :]
    params["current_exposure_words"], params["current_exposure_winding"] = phase_clock_words(end)
    params["current_rotation"] = rotation.copy() if length == 0 else rotations[..., -1, :].copy()
    params["current_segment"] = segment.copy() if length == 0 else segments[..., -1].copy()
    out = (
        None
        if not materialize_output
        else read_clocked_records(
            final,
            ticks,
            rotations,
            segments,
            np.broadcast_to(np.arange(position, position + length), (*batch, length)),
            c,
            chunk=min(chunk, 256),
            counters=counters,
        )
    )
    counters.word_multiplications += int(b.size * payload)
    queries = (ticks, rotations, segments, np.broadcast_to(np.arange(position, position + length), (*batch, length)))
    return (out, final, queries, keep) if return_queries else (out, final)


def execute_clocked_state(values, parameters, *, chunk=64, initial_state=None, reset_before=None):
    words = _words(values, "values", 2)
    if not 1 <= int(chunk) <= 4096:
        raise SPFError("chunk must be in [1,4096]")
    if initial_state is not None and parameters.backward is not None:
        raise SPFError("bidirectional execution requires full context")
    reset = _resets(reset_before, words.shape[:-1])
    counters = ClockedStateCounters()
    out, forward = _scan(words, parameters.forward, reset, initial_state, int(chunk), counters)
    parts, backward = [out], None
    if parameters.backward is not None:
        d = parameters.backward
        def flip(a):
            return np.asarray(a)[..., ::-1, :]
        rd = ClockedDirectionParameters(
            flip(d.log_q16),
            flip(d.phase_q32),
            flip(d.b),
            flip(d.c),
            None if d.exposure_ticks is None else flip(d.exposure_ticks),
            d.attenuation_numerator,
            d.attenuation_denominator,
        )
        rr = np.zeros_like(reset)
        rr[..., 1:] = reset[..., ::-1][..., :-1]
        rout, backward = _scan(words[..., ::-1, :], rd, rr, None, int(chunk), counters)
        parts.append(rout[..., ::-1, :])
    if np.uint32(parameters.diagonal) != SPF32_WORD_ZERO:
        parts.append(spf_mul_words(words, np.uint32(parameters.diagonal)))
        counters.word_multiplications += words.size
    if len(parts) > 1:
        out = packed_pairwise(np.stack(parts, axis=-1), profile="factored-integer-candidate")
        counters.coherent_additions += out.size * (len(parts) - 1)
    return ClockedStateResult(out, forward, backward, counters)


# Explicit compatibility references, never called by the native executor.
from .clocked_state_reference import ClockedSequenceMixer as _ReferenceGeneratedMixer  # noqa: E402
from .clocked_state_reference import finalize_words as finalize_words  # noqa: E402
from .clocked_state_reference import integer_clock as integer_clock  # noqa: E402


class ClockedSequenceMixer(_ReferenceGeneratedMixer):
    def generate(self, addresses):
        old = super().generate(addresses)
        def convert(d):
            return ClockedDirectionParameters(d.log_q16, d.phase_q32, d.b, d.c)
        return ClockedStateParameters(
            convert(old.forward), None if old.backward is None else convert(old.backward), old.diagonal
        )

    def execute(self, addresses, values, **kwargs):
        return execute_clocked_state(values, self.generate(addresses), **kwargs)
