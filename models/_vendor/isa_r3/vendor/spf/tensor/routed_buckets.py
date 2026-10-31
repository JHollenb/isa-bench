"""Sparse addressed SPF records. No Cartesian state or dense bucket expansion."""

from dataclasses import dataclass, field

import numpy as np

from .clocked_state import ClockedDirectionParameters, ClockedStateCounters, _scan
from .core import SPF32_WORD_ZERO, SPFError, pack_words
from .phase_clock import read_clocked_records, validate_clock_block
from .routed_buckets_reference import _bucket_count, _indices
from .routed_buckets_reference import bucket_addresses as bucket_addresses
from .sequence_mixer import _resets, _words
from .v3 import SPFBlockV3

RoutedBucketState = SPFBlockV3


@dataclass
class RoutedBucketCounters(ClockedStateCounters):
    bucket_writes: int = 0
    bucket_reads: int = 0


@dataclass(frozen=True)
class RoutedBucketResult:
    words: np.ndarray
    state: SPFBlockV3
    counters: RoutedBucketCounters = field(default_factory=RoutedBucketCounters)


def execute_routed_buckets(
    values,
    write_buckets,
    read_buckets=None,
    *,
    buckets,
    log_q16=None,
    phase_q32=None,
    exposure_ticks=None,
    attenuation_numerator=-1,
    attenuation_denominator=65536,
    read="bucket",
    frequencies=None,
    chunk=64,
    initial_state=None,
    reset_before=None,
):
    words = _words(values, "values", 2)
    k = _bucket_count(buckets)
    shape = words.shape[:-1]
    wb = _indices(write_buckets, "write_buckets", shape, k)
    rb = wb if read_buckets is None else _indices(read_buckets, "read_buckets", shape, k)
    if read not in ("bucket", "fourier") or (read == "bucket" and frequencies is not None):
        raise SPFError("invalid routed read contract")
    if read == "fourier" and read_buckets is not None:
        raise SPFError("Fourier read consumes all buckets; omit read_buckets")

    def plane(raw):
        if raw is None:
            return np.zeros((*shape, 1), np.int64)
        a = np.asarray(raw)
        if a.shape == shape:
            a = a[..., None]
        if a.shape != (*shape, 1) or not np.issubdtype(a.dtype, np.integer):
            raise SPFError(
                "native sparse routed clock must be shared; per-bucket clocks require the explicit reference executor"
            )
        return a

    l, rot = plane(log_q16), plane(phase_q32)
    ticks = None if exposure_ticks is None else plane(exposure_ticks)
    one = pack_words(np.zeros(l.shape, np.int16), np.zeros(l.shape, np.uint16))
    b = np.where(wb[..., None] >= 0, one, SPF32_WORD_ZERO).astype(np.uint32)
    d = ClockedDirectionParameters(l, rot, b, one, ticks, attenuation_numerator, attenuation_denominator)
    counters = RoutedBucketCounters()
    reset = _resets(reset_before, shape)
    # _scan builds the canonical retained records; query gains are zero here.
    d = ClockedDirectionParameters(l, rot, b, np.zeros_like(one), ticks, attenuation_numerator, attenuation_denominator)
    if not 1 <= int(chunk) <= 4096:
        raise SPFError("chunk must be in [1,4096]")
    if initial_state is not None:
        previous = validate_clock_block(initial_state, streaming=True)
        if previous.get("buckets") != k or "write_buckets" not in previous:
            raise SPFError("stream bucket count/address header mismatch")
        _indices(previous["write_buckets"], "stored write_buckets", initial_state.dims[:-2], k)
    _, state, queries, keep = _scan(
        words, d, reset, initial_state, int(chunk), counters, materialize_output=False, return_queries=True
    )
    p = state.base.params
    prior = (
        np.empty((*shape[:-1], 0), np.int64) if initial_state is None else initial_state.base.params["write_buckets"]
    )
    addresses = np.concatenate((prior, wb), axis=-1)[..., keep]
    p["write_buckets"], p["buckets"] = addresses, k
    qt, qr, qs, qp = queries
    if read == "bucket":
        mask = (rb[..., :, None] == addresses[..., None, :]) & (rb[..., :, None] >= 0)
        out = read_clocked_records(
            state, qt, qr, qs, qp, one, chunk=min(int(chunk), 256), record_mask=mask, counters=counters
        )
    else:
        freqs = tuple(range(k)) if frequencies is None else tuple(int(v) for v in frequencies)
        if not freqs:
            raise SPFError("Fourier read needs at least one frequency")
        terms = []
        for m in freqs:
            phase = ((m * addresses * 65536 + k // 2) // k) % 65536
            gains = pack_words(np.zeros(phase.shape, np.int16), phase.astype(np.uint16))
            gains = np.broadcast_to(gains[..., None, :], (*shape, addresses.shape[-1]))
            terms.append(
                read_clocked_records(
                    state,
                    qt,
                    qr,
                    qs,
                    qp,
                    one,
                    chunk=min(int(chunk), 256),
                    record_mask=np.broadcast_to(addresses[..., None, :] >= 0, gains.shape),
                    record_gains=gains,
                    counters=counters,
                )
            )
        out = np.stack(terms, axis=-2)
    counters.bucket_writes = int(np.count_nonzero(wb >= 0))
    counters.bucket_reads = int(np.count_nonzero(rb >= 0)) if read == "bucket" else int(np.prod(shape)) * len(freqs)
    return RoutedBucketResult(out, state, counters)


def execute_routed_buckets_step(env, inputs, outputs, params):
    unknown = set(params) - {"buckets", "chunk", "read", "frequencies"}
    if unknown or "buckets" not in params:
        raise SPFError(f"invalid routed_buckets parameters: {sorted(unknown)}; buckets is required")
    if len(inputs) not in (2, 3, 4, 5) or len(outputs) != 1:
        raise SPFError("routed_buckets takes (values, write[, read][, log_q16, phase_q32]) -> (y,)")
    read = inputs[2] if len(inputs) in (3, 5) else None
    clock = inputs[-2:] if len(inputs) >= 4 else (None, None)
    result = execute_routed_buckets(
        env[inputs[0]],
        env[inputs[1]],
        None if read is None else env[read],
        log_q16=None if clock[0] is None else env[clock[0]],
        phase_q32=None if clock[1] is None else env[clock[1]],
        **params,
    )
    return {outputs[0]: result.words}
