"""Addition-free select head and clocked-ledger lookup on live SPF32 words.

A select head scores every causal candidate with integer arithmetic, picks the
top-k by exact integer comparison, and gathers the chosen value words. It never
forms a coherent sum, so no SPF adder, Cartesian carrier, or finalization runs:

    score[t, j] = sim(q_t, k_j) + forget_scale * (P_q[t] - P_k[j]) + bias[t, j]
    sim = -sum_c w_c |Theta_q[t,c] - Theta_k[j,c]|_circ      ("circular_l1")
    sim =  sum_c w_c [q[t,c] == k[j,c]]                       ("exact_match")

``candidate_index="sector"`` proposes candidates from ``select_index`` (phase
sector cells with exact integer score bounds) instead of scoring every causal
pair; it returns the same top-k bit for bit.

P is an exact integer clock (a prefix sum of nonpositive log-forget codes), so
the forget term is the log-domain form of a multiplicative decay: a weight
2^(score/S) times prod_{u=j+1..t} 2^(forget_scale*c_u/S) has log
(score + forget_scale*(P_t - P_j))/S. Taking the argmax in the log domain makes
the decay an exact additive score bias (the Forgetting-Transformer bias).

The ledger helpers are the hard limit of that decay: an exact key match and the
latest clock wins. Binding keys are phase sums on the 2^32 ring.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_NAN,
    SPF32_L_ZERO,
    SPF32_WORD_ZERO,
    SPFError,
    circular_diff_u16,
    pack_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)
from .select_index import SelectHeadIndex

NO_SCORE = np.int64(np.iinfo(np.int64).min)
LEDGER_KEY_BITS = 32
LEDGER_CLOCK_LIMIT = 1 << 62
_HALF_TURN = 1 << 15
_SELECT_BUDGET = 1 << 62
_BLOCK_ELEMENTS = 1 << 20
_IDENTITY_WORD = np.uint32(pack_words(np.int16(0), np.uint16(0)))


@dataclass
class SelectHeadCounters:
    """Elementwise integer work issued by the executor (array elements x ops).

    These are declared op-model counts, not instruction or latency measurements.
    ``coherent_additions`` is structurally zero: no SPF word is ever summed.
    """

    integer_ops: int = 0
    comparisons: int = 0
    gathers: int = 0
    word_multiplications: int = 0
    coherent_additions: int = 0
    pairs_scored: int = 0
    buckets_probed: int = 0
    bound_evaluations: int = 0


@dataclass(frozen=True)
class SelectHeadResult:
    """``indices``/``found``/``scores`` are (..., Tq, k); ``values`` adds a payload axis.

    Unfound slots carry index -1, score ``NO_SCORE``, and canonical zero words.
    ``combined`` is (..., Tq, P) for ``combine="product"``; ``combined_found``
    marks rows with at least one selected value.
    """

    indices: np.ndarray
    values: np.ndarray
    found: np.ndarray
    scores: np.ndarray
    counters: SelectHeadCounters
    combined: np.ndarray | None = None
    combined_found: np.ndarray | None = None
    score_matrix: np.ndarray | None = field(default=None, repr=False)


def _words(value: npt.ArrayLike, name: str, minimum_ndim: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype != np.uint32 or arr.ndim < minimum_ndim:
        raise SPFError(f"{name} must be packed SPF32 uint32 words with at least {minimum_ndim} axes")
    return arr


def _integers(value: npt.ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise SPFError(f"{name} must be an integer array")
    return arr.astype(np.int64)


def _batched(value: npt.ArrayLike | None, name: str, shape: tuple[int, ...], default) -> np.ndarray:
    arr = default if value is None else _integers(value, name)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError:
        raise SPFError(f"{name} must broadcast to {shape}") from None


def execute_select_head(
    query_words: npt.ArrayLike,
    key_words: npt.ArrayLike,
    value_words: npt.ArrayLike,
    *,
    channel_weights: npt.ArrayLike,
    score: str = "circular_l1",
    forget_prefix: npt.ArrayLike | tuple[npt.ArrayLike, npt.ArrayLike] | None = None,
    forget_scale: int = 1,
    forget_denominator: int = 1,
    bias: npt.ArrayLike | None = None,
    k: int = 1,
    causal: bool = True,
    strict: bool = True,
    tie: str = "latest",
    key_valid: npt.ArrayLike | None = None,
    query_positions: npt.ArrayLike | None = None,
    key_positions: npt.ArrayLike | None = None,
    min_score: int | None = None,
    combine: str | None = None,
    return_score_matrix: bool = False,
    candidate_index: str = "all",
    index_channels: npt.ArrayLike | None = None,
    sector_bits: int = 4,
    max_probes: int | None = None,
) -> SelectHeadResult:
    """Exact integer top-k selection and value gather; no coherent additions.

    Shapes: queries (..., Tq, C), keys (..., Tk, C), values (..., Tk, P) with a
    shared batch prefix. ``forget_prefix`` is either one clock (..., T) shared by
    queries and keys (requires Tq == Tk) or a ``(P_query, P_key)`` pair.
    Causality compares ``key_positions`` with ``query_positions`` (default
    ``arange``): keys at or after the query are excluded, or only after it when
    ``strict=False``. Ties on score go to the latest (or earliest) key position,
    then the larger (or smaller) index. Candidates below ``min_score`` or masked
    by ``key_valid`` are excluded; an empty set yields ``found=False``, index -1
    and zero words, never a fabricated read.

    ``candidate_index="all"`` scores every pair. ``"sector"`` builds a
    ``SelectHeadIndex`` per batch row (cells on ``index_channels`` at
    ``sector_bits`` bits each) and returns the identical top-k while scoring
    fewer pairs; it cannot bound an explicit ``bias`` and has no score matrix.
    ``max_probes`` (sector only) caps visited cells: approximate, recall unmeasured
    by the executor.
    """
    q = _words(query_words, "query_words", 2)
    kw = _words(key_words, "key_words", 2)
    v = _words(value_words, "value_words", 2)
    batch = q.shape[:-2]
    if kw.shape[:-2] != batch or v.shape[:-2] != batch:
        raise SPFError("queries, keys and values must share their batch axes")
    tq, n_channels = q.shape[-2:]
    tk = kw.shape[-2]
    if kw.shape[-1] != n_channels or v.shape[-2] != tk:
        raise SPFError("keys need the query channel count and one value row per key")
    if score not in ("circular_l1", "exact_match"):
        raise SPFError("score must be 'circular_l1' or 'exact_match'")
    if tie not in ("latest", "earliest"):
        raise SPFError("tie must be 'latest' or 'earliest'")
    if combine not in (None, "product"):
        raise SPFError("combine must be None or 'product'")
    if candidate_index not in ("all", "sector"):
        raise SPFError("candidate_index must be 'all' or 'sector'")
    if (
        isinstance(forget_denominator, (bool, np.bool_))
        or not isinstance(forget_denominator, (int, np.integer))
        or forget_denominator <= 0
    ):
        raise SPFError("forget_denominator must be a positive integer")
    if candidate_index == "sector" and forget_denominator != 1:
        raise SPFError("rational whole-span forgetting currently requires candidate_index='all'")
    if candidate_index == "sector" and (bias is not None or return_score_matrix):
        raise SPFError("the sector candidate index cannot bound an explicit bias or return a score matrix")
    if max_probes is not None and (candidate_index != "sector" or int(max_probes) < 1):
        raise SPFError("max_probes is a positive cell budget for candidate_index='sector'")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 1:
        raise SPFError("k must be a positive integer")
    weights = _integers(channel_weights, "channel_weights")
    if weights.shape != (n_channels,) or np.any(weights < 0):
        raise SPFError("channel_weights must be one nonnegative integer per channel")
    used = [int(c) for c in np.flatnonzero(weights)]
    if score == "circular_l1" and used:
        L_used = unpack_L(np.concatenate((q[..., used].ravel(), kw[..., used].ravel())))
        if np.any(L_used != 0):
            raise SPFError("circular_l1 channels must be finite nonzero unit-circle words (L=0)")
    flat = int(np.prod(batch, dtype=np.int64)) if batch else 1
    qf, kf, vf = q.reshape(flat, tq, n_channels), kw.reshape(flat, tk, n_channels), v.reshape(flat, tk, -1)
    payload = vf.shape[-1]

    qpos = _batched(query_positions, "query_positions", (*batch, tq), np.arange(tq)).reshape(flat, tq)
    kpos = _batched(key_positions, "key_positions", (*batch, tk), np.arange(tk)).reshape(flat, tk)
    if key_valid is None:
        valid = np.ones((flat, tk), dtype=np.bool_)
    else:
        mask = np.asarray(key_valid)
        if mask.dtype != np.bool_:
            raise SPFError("key_valid must be a boolean plane")
        try:
            valid = np.broadcast_to(mask, (*batch, tk)).reshape(flat, tk)
        except ValueError:
            raise SPFError("key_valid must broadcast to the key batch/time shape") from None

    forget = None
    if forget_prefix is not None:
        if isinstance(forget_scale, (bool, np.bool_)) or not isinstance(forget_scale, (int, np.integer)):
            raise SPFError("forget_scale must be an integer score-per-clock-code scale")
        if isinstance(forget_prefix, tuple):
            pq_raw, pk_raw = forget_prefix
        else:
            if tq != tk:
                raise SPFError("a shared forget_prefix needs Tq == Tk; pass (P_query, P_key)")
            pq_raw = pk_raw = forget_prefix
        pq = _batched(pq_raw, "forget_prefix", (*batch, tq), None).reshape(flat, tq)
        pk = _batched(pk_raw, "forget_prefix", (*batch, tk), None).reshape(flat, tk)
        forget = (pq, pk, int(forget_scale))
    explicit = None
    if bias is not None:
        explicit = _batched(bias, "bias", (*batch, tq, tk), None).reshape(flat, tq, tk)

    # Exact int64 range: the score bound times Tk (tie rank) must stay below 2^62.
    bound = int(weights.sum()) * (_HALF_TURN if score == "circular_l1" else 1)
    if forget is not None:
        lo = min(int(forget[0].min(initial=0)), int(forget[1].min(initial=0)))
        hi = max(int(forget[0].max(initial=0)), int(forget[1].max(initial=0)))
        bound += (abs(forget[2]) * (hi - lo) + forget_denominator - 1) // forget_denominator
    if explicit is not None and explicit.size:
        bound += int(np.abs(explicit).max())
    if min_score is not None:
        bound = max(bound, abs(int(min_score)))
    if (2 * bound + 2) * max(tk, 1) >= _SELECT_BUDGET:
        raise SPFError("score range too wide for exact int64 top-k selection")

    # Canonical tie preference: rank keys by (position, index); latest = higher rank.
    order = np.argsort(kpos, axis=-1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.broadcast_to(np.arange(tk), order.shape), axis=-1)
    pref = rank if tie == "latest" else (tk - 1 - rank)

    counters = SelectHeadCounters()
    k_eff = min(k, tk)
    indices = np.full((flat, tq, k), -1, dtype=np.int64)
    scores = np.full((flat, tq, k), NO_SCORE, dtype=np.int64)
    found = np.zeros((flat, tq, k), dtype=np.bool_)
    matrix = np.full((flat, tq, tk), NO_SCORE, dtype=np.int64) if return_score_matrix else None
    if candidate_index == "sector":
        _select_indexed(
            qf,
            kf,
            qpos,
            kpos,
            valid,
            forget,
            weights,
            score,
            tie,
            k,
            causal,
            strict,
            min_score,
            index_channels,
            sector_bits,
            max_probes,
            indices,
            scores,
            found,
            counters,
        )
    rows = max(1, _BLOCK_ELEMENTS // max(1, flat * tk))
    theta_q = unpack_theta(qf) if score == "circular_l1" and candidate_index == "all" else None
    theta_k = unpack_theta(kf) if score == "circular_l1" and candidate_index == "all" else None

    for r0 in range(0, tq if candidate_index == "all" else 0, rows):
        r1 = min(tq, r0 + rows)
        cells = flat * (r1 - r0) * tk
        counters.pairs_scored += cells
        s = np.zeros((flat, r1 - r0, tk), dtype=np.int64)
        for c in used:
            if score == "circular_l1":
                distance = np.abs(circular_diff_u16(theta_q[:, r0:r1, c, None], theta_k[:, None, :, c]))
                s -= weights[c] * distance.astype(np.int64)
                counters.integer_ops += 4 * cells  # ring subtract, wrap, abs, weighted accumulate
            else:
                s += weights[c] * (qf[:, r0:r1, c, None] == kf[:, None, :, c])
                counters.integer_ops += 2 * cells  # word compare, weighted accumulate
        if forget is not None:
            from .phase_clock import round_ratio

            span = forget[0][:, r0:r1, None].astype(object) - forget[1][:, None, :].astype(object)
            s += round_ratio(forget[2] * span, forget_denominator)
            counters.integer_ops += 3 * cells
        if explicit is not None:
            s += explicit[:, r0:r1, :]
            counters.integer_ops += cells
        allowed = np.broadcast_to(valid[:, None, :], s.shape)
        if causal:
            before = kpos[:, None, :] < qpos[:, r0:r1, None] if strict else kpos[:, None, :] <= qpos[:, r0:r1, None]
            allowed = allowed & before
            counters.comparisons += cells
        if min_score is not None:
            allowed = allowed & (s >= int(min_score))
            counters.comparisons += cells
        composite = np.where(allowed, s * tk + pref[:, None, :], NO_SCORE)
        counters.integer_ops += 2 * cells
        counters.comparisons += cells  # one candidate comparison per cell in the selection scan
        if matrix is not None:
            matrix[:, r0:r1, :] = np.where(allowed, s, NO_SCORE)
        if k_eff == 1:
            top = np.argmax(composite, axis=-1)[..., None]
        else:
            if k_eff < tk:
                part = np.argpartition(composite, tk - k_eff, axis=-1)[..., tk - k_eff :]
            else:
                part = np.broadcast_to(np.arange(tk), composite.shape).copy()
            keyed = np.take_along_axis(composite, part, axis=-1)
            top = np.take_along_axis(part, np.argsort(keyed, axis=-1, kind="stable")[..., ::-1], axis=-1)
        chosen = np.take_along_axis(composite, top, axis=-1)
        hit = chosen != NO_SCORE
        indices[:, r0:r1, :k_eff] = np.where(hit, top, -1)
        scores[:, r0:r1, :k_eff] = np.where(hit, np.take_along_axis(s, top, axis=-1), NO_SCORE)
        found[:, r0:r1, :k_eff] = hit

    gathered = vf[np.arange(flat)[:, None, None], np.clip(indices, 0, None)]
    values = np.where(found[..., None], gathered, SPF32_WORD_ZERO).astype(np.uint32)
    counters.gathers += int(np.count_nonzero(found))

    combined = combined_found = None
    if combine == "product":
        L = unpack_L(values).astype(np.int64)
        if np.any(found[..., None] & (L == SPF32_L_NAN)):
            raise SPFError("product combine requires finite selected values")
        nonzero = found[..., None] & (int(SPF32_L_ZERO) != L)
        total = np.where(nonzero, L, 0).sum(axis=-2)
        has_zero = np.any(found[..., None] & (int(SPF32_L_ZERO) == L), axis=-2)
        if np.any(~has_zero & ((total < SPF32_L_MIN_FINITE) | (total > SPF32_L_MAX_FINITE))):
            raise SPFError("product combine would saturate the SPF32 log range")
        product = np.full((flat, tq, payload), _IDENTITY_WORD, dtype=np.uint32)
        for slot in range(k):
            live = found[..., slot, None]
            step = spf_mul_words(product, values[..., slot, :])
            product = np.where(live, step, product).astype(np.uint32)
            counters.word_multiplications += int(np.count_nonzero(live)) * payload
        combined_found = found.any(axis=-1)
        combined = np.where(combined_found[..., None], product, SPF32_WORD_ZERO).astype(np.uint32)
        combined = combined.reshape((*batch, tq, payload))
        combined_found = combined_found.reshape((*batch, tq))

    return SelectHeadResult(
        indices=indices.reshape((*batch, tq, k)),
        values=values.reshape((*batch, tq, k, payload)),
        found=found.reshape((*batch, tq, k)),
        scores=scores.reshape((*batch, tq, k)),
        counters=counters,
        combined=combined,
        combined_found=combined_found,
        score_matrix=None if matrix is None else matrix.reshape((*batch, tq, tk)),
    )


def _select_indexed(
    qf,
    kf,
    qpos,
    kpos,
    valid,
    forget,
    weights,
    score,
    tie,
    k,
    causal,
    strict,
    min_score,
    index_channels,
    sector_bits,
    max_probes,
    indices,
    scores,
    found,
    counters,
) -> None:
    """Fill the top-k planes through one exact ``SelectHeadIndex`` per batch row."""
    flat, tq, tk = qf.shape[0], qf.shape[1], kf.shape[1]
    if tk == 0:
        return
    for b in range(flat):
        order = np.argsort(kpos[b], kind="stable")  # ids = tie rank, as in the all-pairs path
        index = SelectHeadIndex(
            weights,
            capacity=tk,
            score=score,
            clocked=forget is not None,
            forget_scale=forget[2] if forget is not None else 1,
            tie=tie,
            index_channels=index_channels,
            sector_bits=sector_bits,
            tail_limit=tk,
        )
        index.append(
            kf[b][order],
            positions=kpos[b][order],
            clock=None if forget is None else forget[1][b][order],
            valid=np.ascontiguousarray(valid[b][order]),
        )
        index.freeze()
        for t in range(tq):
            ids, s = index.query(
                qf[b, t],
                position=int(qpos[b, t]),
                clock=None if forget is None else int(forget[0][b, t]),
                k=k,
                causal=causal,
                strict=strict,
                min_score=min_score,
                max_probes=max_probes,
            )
            n = ids.size
            indices[b, t, :n] = order[ids]
            scores[b, t, :n] = s
            found[b, t, :n] = True
        ic = index.counters
        counters.pairs_scored += ic.pairs_scored
        counters.buckets_probed += ic.buckets_probed
        counters.bound_evaluations += ic.bound_evaluations
        counters.integer_ops += ic.integer_ops
        counters.comparisons += ic.comparisons


def _ledger_ints(value: npt.ArrayLike, name: str, limit: int) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise SPFError(f"{name} must be integers")
    arr = arr.astype(np.int64) if arr.dtype != np.uint64 else arr
    if np.any(arr < 0) or np.any(arr >= limit):
        raise SPFError(f"{name} must lie in [0, {limit})")
    return arr.astype(np.int64)


def binding_key(entity: npt.ArrayLike, role: npt.ArrayLike) -> np.ndarray:
    """Phase-sum binding key ``(entity + role) mod 2^32`` as uint32."""
    a = _ledger_ints(entity, "entity", 1 << LEDGER_KEY_BITS)
    b = _ledger_ints(role, "role", 1 << LEDGER_KEY_BITS)
    return ((a + b) & ((1 << LEDGER_KEY_BITS) - 1)).astype(np.uint32)


def _ledger_index(keys: npt.ArrayLike, clock: npt.ArrayLike, valid: npt.ArrayLike | None):
    key = _ledger_ints(keys, "keys", 1 << LEDGER_KEY_BITS)
    tick = _ledger_ints(clock, "clock", LEDGER_CLOCK_LIMIT)
    if key.ndim != 1 or tick.shape != key.shape:
        raise SPFError("a ledger is one append-only axis of keys with one clock per record")
    live = np.ones(key.shape, dtype=np.bool_) if valid is None else np.asarray(valid)
    if live.dtype != np.bool_ or live.shape != key.shape:
        raise SPFError("valid must be a boolean mask with one entry per record")
    record = np.flatnonzero(live)
    # Separate typed coordinates avoid stealing clock resolution to squeeze a
    # key and time into one int64. Clock winding now survives full phase turns.
    composite = np.empty(record.size, dtype=[("key", "<i8"), ("tick", "<i8")])
    composite["key"], composite["tick"] = key[record], tick[record]
    order = np.argsort(composite, order=("key", "tick"), kind="stable")
    return composite[order], record[order]


def _ledger_lookup(sorted_composite, sorted_record, query_key, as_of):
    target = np.empty(np.shape(query_key), dtype=sorted_composite.dtype)
    target["key"], target["tick"] = query_key, as_of
    pos = np.searchsorted(sorted_composite, target, side="right") - 1
    safe = np.clip(pos, 0, None)
    hit = (
        (pos >= 0) & (sorted_composite["key"][safe] == query_key)
        if sorted_composite.size
        else np.zeros(np.shape(target), dtype=np.bool_)
    )
    index = np.where(hit, sorted_record[safe] if sorted_record.size else -1, -1)
    return index.astype(np.int64), hit


def latest_under_key(
    keys: npt.ArrayLike,
    clock: npt.ArrayLike,
    query_keys: npt.ArrayLike,
    *,
    as_of: npt.ArrayLike | None = None,
    valid: npt.ArrayLike | None = None,
    clock_winding: npt.ArrayLike | None = None,
    as_of_winding: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Latest record whose key equals the query key, with ``clock <= as_of``.

    Exact equality on the 2^32 key ring; equal clocks go to the later write.
    Returns ``(indices, found)``; unfound queries carry index -1. Integer sort
    plus binary search: no scores are mixed and nothing is summed.
    """
    if clock_winding is not None:
        from .phase_clock import phase_clock_ticks

        clock = phase_clock_ticks(clock, clock_winding)
    if as_of_winding is not None:
        from .phase_clock import phase_clock_ticks

        as_of = phase_clock_ticks(as_of, as_of_winding)
    sc, sr = _ledger_index(keys, clock, valid)
    qk = _ledger_ints(query_keys, "query_keys", 1 << LEDGER_KEY_BITS)
    now = (
        np.full(qk.shape, LEDGER_CLOCK_LIMIT - 1, dtype=np.int64)
        if as_of is None
        else np.broadcast_to(_ledger_ints(as_of, "as_of", LEDGER_CLOCK_LIMIT), qk.shape)
    )
    return _ledger_lookup(sc, sr, qk, now)


def ledger_two_hop(
    keys: npt.ArrayLike,
    clock: npt.ArrayLike,
    named_entity: npt.ArrayLike,
    first_keys: npt.ArrayLike,
    second_role: npt.ArrayLike,
    *,
    as_of: npt.ArrayLike | None = None,
    valid: npt.ArrayLike | None = None,
    clock_winding: npt.ArrayLike | None = None,
    as_of_winding: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chain two exact reads: latest record under ``first_keys``, then the latest
    record under ``binding_key(named_entity[first], second_role)``.

    Returns ``(first_index, second_index, found)``; ``found`` needs both hops.
    """
    if clock_winding is not None:
        from .phase_clock import phase_clock_ticks

        clock = phase_clock_ticks(clock, clock_winding)
    if as_of_winding is not None:
        from .phase_clock import phase_clock_ticks

        as_of = phase_clock_ticks(as_of, as_of_winding)
    sc, sr = _ledger_index(keys, clock, valid)
    named = _ledger_ints(named_entity, "named_entity", 1 << LEDGER_KEY_BITS)
    if named.shape != np.shape(keys):
        raise SPFError("named_entity needs one entity per record")
    qk = _ledger_ints(first_keys, "first_keys", 1 << LEDGER_KEY_BITS)
    now = (
        np.full(qk.shape, LEDGER_CLOCK_LIMIT - 1, dtype=np.int64)
        if as_of is None
        else np.broadcast_to(_ledger_ints(as_of, "as_of", LEDGER_CLOCK_LIMIT), qk.shape)
    )
    first, hit1 = _ledger_lookup(sc, sr, qk, now)
    entity = np.where(hit1, named[np.clip(first, 0, None)], 0)
    second_key = binding_key(entity, np.broadcast_to(second_role, qk.shape)).astype(np.int64)
    second, hit2 = _ledger_lookup(sc, sr, second_key, now)
    found = hit1 & hit2
    return first, np.where(found, second, -1), found


__all__ = [
    "LEDGER_CLOCK_LIMIT",
    "NO_SCORE",
    "SelectHeadCounters",
    "SelectHeadResult",
    "binding_key",
    "execute_select_head",
    "latest_under_key",
    "ledger_two_hop",
]
