"""REFERENCE routed bucket memory with private Cartesian-coordinate accumulators.

Route is the direct-coordinate form of a grouped (character/Fourier) mixer. Token
``t`` writes its value into bucket ``a_t`` and reads the total of earlier writes
in bucket ``r_t`` (read before inject):

    G_{g,t} = sum_{j<t, a_j=g} exp(P_t - P_j) v_j,      y_t = G_{r_t,t}.

Packed SPF32 bucket sums round at every coherent addition, so a bucket stalls
once its total passes about 738 writes of the incoming size. Here every bucket
is a clocked-state lane (:mod:`.clocked_state`): an exact integer clock ``P``
(Q16 log decay plus 2^32-ring phase, one clock shared by all buckets or one per
bucket), writes that enter an int64 frame-relative accumulator once, a rebase
once per chunk at absolute positions, and one SPF32 rounding per read. A token
converts and adds only into its write bucket and reads only its read bucket.

The bucket read is bit-identical to :func:`execute_clocked_state` with one-hot
write gains and one-hot read gains over ``K`` states (the routed export used by
the learned comparison), and the ``fourier`` read mode is bit-identical to the
same executor with read gains ``omega^(m g)``: the Fourier mixer's state is the
DFT of the bucket totals. Coherent SPF additions: zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .clocked_state_reference import (
    _ENTRY_BITS,
    _NORM_BITS,
    _PHASE32,
    _Q,
    _SENTINEL_EXP,
    _SUM_GUARD,
    MAX_CHUNK,
    MAX_STATES,
    _align_sum,
    _check_headroom,
    _cmul_q31,
    _code_to_fixed,
    _normalize,
    _shift,
    _word_codes,
    finalize_words,
)
from .core import SPFError, unpack_theta
from .sequence_mixer import _resets, _words

_R = 65536


# ---------------------------------------------------------------- addresses


def _bucket_count(buckets: int) -> int:
    k = int(buckets)
    if k < 1:
        raise SPFError("routed buckets need at least one bucket")
    return k


def bucket_addresses(
    words: npt.ArrayLike,
    buckets: int,
    *,
    table: npt.ArrayLike | None = None,
    rounding: str = "nearest",
) -> np.ndarray:
    """Bucket index per word, read from its phase with integer arithmetic only.

    Without a table, ``nearest`` gives ``round(K theta / 65536) mod K`` (half up),
    ``floor`` gives ``floor(K theta / 65536)`` and ``exact`` rejects phases off the
    K-grid. With ``table`` (an integer sector ROM, entries in ``[-1, K)``, -1 for
    no bucket) the index is ``table[floor(theta * len(table) / 65536)]``, the
    phase-shear ROM rule.
    """
    k = _bucket_count(buckets)
    theta = unpack_theta(_words(words, "address words")).astype(np.int64)
    if table is not None:
        rom = np.asarray(table)
        if rom.ndim != 1 or not 1 <= rom.size <= _R or not np.issubdtype(rom.dtype, np.integer):
            raise SPFError("bucket table must be a 1-D integer ROM with 1..65536 entries")
        if np.any((rom < -1) | (rom >= k)):
            raise SPFError("bucket table entries must lie in [-1, buckets)")
        return rom.astype(np.int64)[(theta * rom.size) >> 16]
    if rounding == "nearest":
        return ((theta * k + _R // 2) >> 16) % k
    if rounding == "floor":
        return (theta * k) >> 16
    if rounding == "exact":
        if np.any((theta * k) % _R):
            raise SPFError("off-grid address phase: no silent nearest bucket")
        return (theta * k) >> 16
    raise SPFError("rounding must be 'nearest', 'floor' or 'exact'")


# ---------------------------------------------------------------- state and results


@dataclass
class RoutedBucketCounters:
    """Operation counts of the declared schedule, not latency."""

    rom_reads: int = 0
    integer_multiplies: int = 0
    integer_additions: int = 0
    finalizations: int = 0
    rebases: int = 0
    bucket_writes: int = 0
    bucket_reads: int = 0
    coherent_additions: int = 0


@dataclass(frozen=True)
class RoutedBucketState:
    """Streaming state: absolute position, clocks (..., C), accumulators (..., K, P).

    ``C`` is 1 for a clock shared by every bucket, ``K`` for per-bucket clocks.
    """

    position: int
    chunk: int
    clock_L: np.ndarray
    clock_T: np.ndarray
    base_L: np.ndarray
    base_T: np.ndarray
    acc_re: np.ndarray
    acc_im: np.ndarray
    acc_exp: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            int(a.nbytes)
            for a in (self.clock_L, self.clock_T, self.base_L, self.base_T, self.acc_re, self.acc_im, self.acc_exp)
        )


@dataclass(frozen=True)
class RoutedBucketResult:
    """``words`` has shape (..., T, P), or (..., T, M, P) for ``read="fourier"``."""

    words: np.ndarray
    state: RoutedBucketState
    counters: RoutedBucketCounters = field(default_factory=RoutedBucketCounters)


def _empty_state(batch: tuple[int, ...], k: int, p: int, c: int, chunk: int) -> RoutedBucketState:
    clock = np.zeros((*batch, c), dtype=np.int64)
    acc = np.zeros((*batch, k, p), dtype=np.int64)
    return RoutedBucketState(
        0,
        chunk,
        clock,
        clock.copy(),
        clock.copy(),
        clock.copy(),
        acc,
        acc.copy(),
        np.full(acc.shape, _SENTINEL_EXP, dtype=np.int64),
    )


# ---------------------------------------------------------------- validation


def _indices(value: npt.ArrayLike, name: str, shape: tuple[int, ...], k: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.shape != shape or not np.issubdtype(arr.dtype, np.integer):
        raise SPFError(f"{name} must be an integer plane with the values batch/time shape")
    if np.any((arr < -1) | (arr >= k)):
        raise SPFError(f"{name} entries must lie in [-1, buckets); -1 means no bucket")
    return arr.astype(np.int64)


def _clock_codes(log_q16, phase_q32, shape: tuple[int, ...], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Clock planes as (..., T, C): C = 1 for a shared clock, K for per-bucket clocks."""
    if log_q16 is None and phase_q32 is None:
        zero = np.zeros((*shape, 1), dtype=np.int64)
        return zero, zero.copy()
    planes = []
    for name, plane in (("log_q16", log_q16), ("phase_q32", phase_q32)):
        if plane is None:
            planes.append(None)
            continue
        arr = np.asarray(plane)
        if not np.issubdtype(arr.dtype, np.integer):
            raise SPFError(f"clock {name} must be integer codes")
        if arr.shape == shape:
            arr = arr[..., None]
        elif arr.shape != (*shape, k):
            raise SPFError(f"clock {name} must have shape (..., T) for a shared clock or (..., T, K) per bucket")
        planes.append(arr.astype(np.int64))
    ref = next(p for p in planes if p is not None)
    L = np.zeros(ref.shape, dtype=np.int64) if planes[0] is None else planes[0]
    T = np.zeros(ref.shape, dtype=np.int64) if planes[1] is None else planes[1] & _PHASE32
    if L.shape != T.shape:
        raise SPFError("clock log and phase planes must both be shared or both be per bucket")
    if np.any(L > 0):
        raise SPFError("clock transitions must not amplify magnitude (log_q16 <= 0)")
    return L, T


# ---------------------------------------------------------------- the scan


def _at(plane: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Clock value at a bucket: plane (nb, S, C), index (nb, S) -> (nb, S)."""
    if plane.shape[-1] == 1:
        return plane[..., 0]
    return np.take_along_axis(plane, np.maximum(index, 0)[..., None], -1)[..., 0]


def _base_at(base: np.ndarray, index: np.ndarray) -> np.ndarray:
    """Frame base at a bucket: base (nb, C), index (nb, S) -> (nb, S) or (nb, 1)."""
    if base.shape[-1] == 1:
        return base
    return np.take_along_axis(base, np.maximum(index, 0), 1)


def _segment_prefix(m: np.ndarray, groups: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive per-bucket prefix sums of m (nb, S, lanes) within each row, in sorted key order.

    Returns the sorted keys (nb, S) and the prefixes. Keys order rows, then
    buckets, then positions; tokens without a write sort last in their row.
    """
    nb, steps = groups.shape
    rows = np.arange(nb)[:, None]
    key = (rows * (k + 1) + groups) * steps + np.arange(steps)
    order = np.argsort(key, axis=1)
    skey = np.take_along_axis(key, order, 1)
    sm = np.take_along_axis(m, order[..., None], 1)
    total = np.cumsum(sm, axis=1)  # <= steps * 2^49 per row: no overflow
    group = skey // steps
    first = np.ones(group.shape, dtype=np.bool_)
    first[:, 1:] = group[:, 1:] != group[:, :-1]
    start = np.maximum.accumulate(np.where(first, np.arange(steps), 0), axis=1)[..., None]
    before = np.take_along_axis(total, start, 1) - np.take_along_axis(sm, start, 1)
    return skey, total - before


def _scan(words, wb, rb, Lc, Tc, reset, state: RoutedBucketState, freqs, counters: RoutedBucketCounters):
    """Flattened-batch scan: words (nb, T, P), indices (nb, T), clocks (nb, T, C)."""
    nb, length, p = words.shape
    k = state.acc_re.shape[-2]
    chunk = state.chunk
    rows = np.arange(nb)
    PL = state.clock_L[:, None, :] + np.cumsum(Lc, axis=1)
    PT = (state.clock_T[:, None, :] + np.cumsum(Tc, axis=1)) & _PHASE32
    counters.integer_additions += 2 * int(Lc.size)
    wL, wT, wzero = _word_codes(words)
    fourier = freqs is not None
    out_shape = (nb, length, len(freqs), p) if fourier else (nb, length, p)
    out_re = np.zeros(out_shape, dtype=np.int64)
    out_im = np.zeros_like(out_re)
    out_exp = np.full(out_shape, _SENTINEL_EXP, dtype=np.int64)
    pos = state.position
    acc_re, acc_im, acc_exp = state.acc_re.copy(), state.acc_im.copy(), state.acc_exp.copy()
    base_L, base_T = state.base_L.copy(), state.base_T.copy()
    clock_L, clock_T = state.clock_L.copy(), state.clock_T.copy()
    start = 0
    while start < length:
        absolute = pos + start
        if absolute > 0 and absolute % chunk == 0:  # rebase once per chunk: g *= exp(P_{t-1} - P_base)
            dL, dT = clock_L - base_L, (clock_T - base_T) & _PHASE32
            fr, fi, fe = _code_to_fixed(dL, dT, counters)
            acc_re, acc_im = _cmul_q31(acc_re, acc_im, fr[..., None], fi[..., None], counters)
            acc_re, acc_im, acc_exp = _normalize(acc_re, acc_im, acc_exp + fe[..., None], _NORM_BITS)
            base_L, base_T = clock_L.copy(), clock_T.copy()
            counters.rebases += int(acc_exp.shape[0] * k)
        stop = min(length, start + chunk - (absolute % chunk))
        sl = slice(start, stop)
        steps = stop - start
        w_idx, seg_reset = wb[:, sl], reset[:, sl]
        writes = w_idx >= 0
        # Writes enter frame-relative at their own bucket's clock: exp(P_base - P_j) v_j.
        xL = wL[:, sl] + (_base_at(base_L, w_idx) - _at(PL[:, sl], w_idx))[..., None]
        xT = wT[:, sl] + ((_base_at(base_T, w_idx) - _at(PT[:, sl], w_idx)) & _PHASE32)[..., None]
        wre, wim, wexp = _code_to_fixed(xL, xT, counters)
        live = writes[..., None] & ~wzero[:, sl]
        g = np.maximum(w_idx, 0)
        lane_exp = acc_exp[rows[:, None], g]  # (nb, S, P)
        need = np.where(live, wexp - _ENTRY_BITS, _SENTINEL_EXP)
        counters.bucket_writes += int(np.count_nonzero(writes))
        counters.integer_additions += 2 * int(np.count_nonzero(live))
        if not np.any(seg_reset) and np.all(need <= lane_exp):
            # Constant lane exponents: exact integer sums, in any order.
            s = lane_exp - wexp + _Q
            m_re = np.where(live, _shift(wre, s), 0)
            m_im = np.where(live, _shift(wim, s), 0)
            if fourier:
                onehot = (w_idx[..., None] == np.arange(k))[..., None]  # (nb, S, K, 1)
                d_re, d_im = np.where(onehot, m_re[:, :, None], 0), np.where(onehot, m_im[:, :, None], 0)
                c_re, c_im = np.cumsum(d_re, axis=1), np.cumsum(d_im, axis=1)
                hist = (acc_re[:, None] + c_re - d_re, acc_im[:, None] + c_im - d_im)
                hist_exp = np.broadcast_to(acc_exp[:, None], hist[0].shape)
                acc_re, acc_im = acc_re + c_re[:, -1], acc_im + c_im[:, -1]
            else:
                skey, prefix = _segment_prefix(np.concatenate([m_re, m_im], -1), np.where(writes, w_idx, k), k)
                p_re, p_im = prefix[..., :p], prefix[..., p:]
                r_idx = rb[:, sl]
                rd = r_idx >= 0
                query = ((rows[:, None] * (k + 1) + np.maximum(r_idx, 0)) * steps + np.arange(steps)).ravel()
                flat = skey.ravel()
                j = np.searchsorted(flat, query, side="left") - 1
                hit = (j >= 0) & (flat[np.maximum(j, 0)] // steps == query // steps)
                hit = hit.reshape(nb, steps)[..., None]
                j = np.maximum(j, 0)
                r = np.maximum(r_idx, 0)
                hist_re = acc_re[rows[:, None], r] + np.where(hit, p_re.reshape(-1, p)[j].reshape(nb, steps, p), 0)
                hist_im = acc_im[rows[:, None], r] + np.where(hit, p_im.reshape(-1, p)[j].reshape(nb, steps, p), 0)
                hist_exp = acc_exp[rows[:, None], r]
                hist = (np.where(rd[..., None], hist_re, 0), np.where(rd[..., None], hist_im, 0))
                at = np.nonzero(writes)
                np.add.at(acc_re, (at[0], w_idx[at]), m_re[at])
                np.add.at(acc_im, (at[0], w_idx[at]), m_im[at])
        else:
            hist_shape = (nb, steps, k, p) if fourier else (nb, steps, p)
            hist = (np.zeros(hist_shape, dtype=np.int64), np.zeros(hist_shape, dtype=np.int64))
            hist_exp = np.full(hist_shape, _SENTINEL_EXP, dtype=np.int64)
            acc_re, acc_im, acc_exp = acc_re.copy(), acc_im.copy(), acc_exp.copy()
            for i in range(steps):
                cut = seg_reset[:, i]
                acc_re[cut], acc_im[cut], acc_exp[cut] = 0, 0, _SENTINEL_EXP
                if fourier:
                    hist[0][:, i], hist[1][:, i], hist_exp[:, i] = acc_re, acc_im, acc_exp
                else:
                    rr = np.nonzero(rb[:, start + i] >= 0)[0]
                    gr = rb[rr, start + i]
                    hist[0][rr, i], hist[1][rr, i], hist_exp[rr, i] = acc_re[rr, gr], acc_im[rr, gr], acc_exp[rr, gr]
                wr = np.nonzero(writes[:, i])[0]
                if wr.size == 0:
                    continue
                gw = w_idx[wr, i]
                lv = live[wr, i]
                old = acc_exp[wr, gw]
                new = np.where(lv, np.maximum(old, wexp[wr, i] - _ENTRY_BITS), old)
                s = new - wexp[wr, i] + _Q
                acc_re[wr, gw] = _shift(acc_re[wr, gw], new - old) + np.where(lv, _shift(wre[wr, i], s), 0)
                acc_im[wr, gw] = _shift(acc_im[wr, gw], new - old) + np.where(lv, _shift(wim[wr, i], s), 0)
                acc_exp[wr, gw] = new
        _check_headroom(acc_re, acc_im)
        # Reads: h_t = exp(P_t - P_base) g_{<t} at the read bucket, or rotated bucket totals.
        if fourier:
            dL = np.broadcast_to(PL[:, sl] - base_L[:, None], (nb, steps, k))
            dT = np.broadcast_to((PT[:, sl] - base_T[:, None]) & _PHASE32, (nb, steps, k))
            counters.bucket_reads += int(nb * steps * k * len(freqs))
            for mi, m in enumerate(freqs):
                tick = (((m * np.arange(k)) % k) * (1 << 33) + k) // (2 * k)  # round(2^32 (m g mod K) / K)
                fr, fi, fe = _code_to_fixed(dL, dT + tick, counters)
                tre, tim = _cmul_q31(hist[0], hist[1], fr[..., None], fi[..., None], counters)
                dead = (tre == 0) & (tim == 0)
                sre, sim, sexp = _align_sum(
                    (np.where(dead, 0, tre), np.where(dead, 0, tim), hist_exp + fe[..., None]), counters, axis=-2
                )
                out_re[:, sl, mi], out_im[:, sl, mi], out_exp[:, sl, mi] = sre, sim, sexp
        else:
            r_idx = rb[:, sl]
            rd = r_idx >= 0
            rL = _at(PL[:, sl], r_idx) - _base_at(base_L, r_idx)
            rT = (_at(PT[:, sl], r_idx) - _base_at(base_T, r_idx)) & _PHASE32
            fr, fi, fe = _code_to_fixed(rL, rT, counters)
            tre, tim = _cmul_q31(hist[0], hist[1], fr[..., None], fi[..., None], counters)
            alive = rd[..., None] & ((tre != 0) | (tim != 0))
            # The one live term passes the same guard shift as a K-state readout sum.
            out_re[:, sl] = np.where(alive, _shift(tre, _SUM_GUARD), 0)
            out_im[:, sl] = np.where(alive, _shift(tim, _SUM_GUARD), 0)
            out_exp[:, sl] = np.where(alive, hist_exp + fe[..., None], _SENTINEL_EXP) + _SUM_GUARD
            counters.bucket_reads += int(np.count_nonzero(rd))
        clock_L, clock_T = PL[:, stop - 1].copy(), PT[:, stop - 1].copy()
        start = stop
    final = RoutedBucketState(pos + length, chunk, clock_L, clock_T, base_L, base_T, acc_re, acc_im, acc_exp)
    return (out_re, out_im, out_exp), final


def execute_routed_buckets(
    values: npt.ArrayLike,
    write_buckets: npt.ArrayLike,
    read_buckets: npt.ArrayLike | None = None,
    *,
    buckets: int,
    log_q16: npt.ArrayLike | None = None,
    phase_q32: npt.ArrayLike | None = None,
    read: str = "bucket",
    frequencies: Sequence[int] | None = None,
    chunk: int = 64,
    initial_state: RoutedBucketState | None = None,
    reset_before: npt.ArrayLike | None = None,
) -> RoutedBucketResult:
    """Causal routed bucket memory with clocked accumulators, one SPF32 rounding per read.

    ``values`` (..., T, P) packed words; ``write_buckets``/``read_buckets`` (..., T)
    integer bucket indices, -1 for no write/read (read buckets default to the write
    buckets). Per token: reset every bucket if requested, read, then inject.
    ``log_q16``/``phase_q32`` are clock codes, shape (..., T) for one clock shared
    by all buckets or (..., T, K) per bucket; omitted means no decay or rotation.
    ``read="fourier"`` returns ``sum_g omega^(m g) G_{g,t}`` for each frequency
    ``m`` (default all K), shape (..., T, M, P). The returned state continues a
    stream bit-exactly.
    """
    k = _bucket_count(buckets)
    words = _words(values, "values", 2)
    batch, length, p = words.shape[:-2], words.shape[-2], words.shape[-1]
    shape = (*batch, length)
    wb = _indices(write_buckets, "write_buckets", shape, k)
    if read == "bucket":
        if frequencies is not None:
            raise SPFError("frequencies apply only to read='fourier'")
        rb = wb if read_buckets is None else _indices(read_buckets, "read_buckets", shape, k)
        freqs = None
    elif read == "fourier":
        if read_buckets is not None:
            raise SPFError("read='fourier' reads every bucket; omit read_buckets")
        if k > MAX_STATES:
            raise SPFError(f"fourier reads support at most {MAX_STATES} buckets")
        freqs = tuple(range(k)) if frequencies is None else tuple(int(m) for m in frequencies)
        if not freqs:
            raise SPFError("fourier reads need at least one frequency")
        rb = wb
    else:
        raise SPFError("read must be 'bucket' or 'fourier'")
    Lc, Tc = _clock_codes(log_q16, phase_q32, shape, k)
    chunk = int(chunk)
    if not 1 <= chunk <= MAX_CHUNK:
        raise SPFError(f"chunk must be in [1, {MAX_CHUNK}]")
    reset = _resets(reset_before, shape)
    c = Lc.shape[-1]
    if initial_state is None:
        state = _empty_state(batch, k, p, c, chunk)
    else:
        if initial_state.chunk != chunk:
            raise SPFError("initial_state was produced with a different chunk")
        if initial_state.acc_re.shape != (*batch, k, p) or initial_state.clock_L.shape != (*batch, c):
            raise SPFError("initial_state must match (..., K, P) accumulators and (..., C) clocks")
        state = initial_state
    nb = int(np.prod(batch, dtype=np.int64))
    flat_state = RoutedBucketState(
        state.position,
        chunk,
        *(
            np.asarray(a).reshape(nb, *np.shape(a)[len(batch) :])
            for a in (
                state.clock_L,
                state.clock_T,
                state.base_L,
                state.base_T,
                state.acc_re,
                state.acc_im,
                state.acc_exp,
            )
        ),
    )
    counters = RoutedBucketCounters()
    (re, im, exp), final = _scan(
        words.reshape(nb, length, p),
        wb.reshape(nb, length),
        rb.reshape(nb, length),
        Lc.reshape(nb, length, c),
        Tc.reshape(nb, length, c),
        reset.reshape(nb, length),
        flat_state,
        freqs,
        counters,
    )
    out = finalize_words(*_normalize(re, im, exp, _NORM_BITS))
    counters.finalizations += int(np.count_nonzero(out))
    out_shape = (*batch, length, len(freqs), p) if freqs is not None else (*batch, length, p)
    final = RoutedBucketState(
        final.position,
        chunk,
        *(
            a.reshape(*batch, *a.shape[1:])
            for a in (
                final.clock_L,
                final.clock_T,
                final.base_L,
                final.base_T,
                final.acc_re,
                final.acc_im,
                final.acc_exp,
            )
        ),
    )
    return RoutedBucketResult(out.reshape(out_shape), final, counters)


_STEP_PARAMS = frozenset({"buckets", "chunk", "read", "frequencies"})


def execute_routed_buckets_step(
    env: dict, inputs: Sequence[str], outputs: Sequence[str], params: dict
) -> dict[str, np.ndarray]:
    """ExecutionPlan ``routed_buckets`` semantics.

    Inputs ``(values, write)``, ``(values, write, read)``, ``(values, write, log_q16,
    phase_q32)`` or ``(values, write, read, log_q16, phase_q32)``; one output, the
    read words. Params: buckets, chunk, read, frequencies.
    """
    unknown = set(params) - _STEP_PARAMS
    if unknown:
        raise SPFError(f"routed_buckets got unknown parameters {sorted(unknown)}")
    if len(inputs) not in (2, 3, 4, 5) or len(outputs) != 1:
        raise SPFError("routed_buckets takes (values, write[, read][, log_q16, phase_q32]) -> (y,)")
    names = list(inputs)
    read_name = names[2] if len(names) in (3, 5) else None
    clock = names[-2:] if len(names) >= 4 else (None, None)
    result = execute_routed_buckets(
        env[names[0]],
        env[names[1]],
        None if read_name is None else env[read_name],
        log_q16=None if clock[0] is None else env[clock[0]],
        phase_q32=None if clock[1] is None else env[clock[1]],
        **params,
    )
    return {outputs[0]: result.words}


__all__ = [
    "RoutedBucketCounters",
    "RoutedBucketResult",
    "RoutedBucketState",
    "bucket_addresses",
    "execute_routed_buckets",
    "execute_routed_buckets_step",
]
