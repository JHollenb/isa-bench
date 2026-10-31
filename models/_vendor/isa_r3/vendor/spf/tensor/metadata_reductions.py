"""Analysis-only metadata telemetry for routed SPF reductions.

The helpers in this module do not change reducer behavior. They expose the
Stage-5 metadata route decisions, bypass masks, and planner cost/error estimates
for packed SPF row streams and attention-shaped intermediates.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .attention_ops import _SENTINEL_I32, _contract_product_terms, quantize_scores_lq
from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    AccumulationDomain,
    ReducerPolicy,
    ReducerSpec,
    RoutingPolicy,
    SPFError,
    ensure_reducer_spec,
    pack_words,
)
from .reducers import (
    ReductionStats,
    _apply_dom_bypass_mask_rows,
    compute_stats,
    domain_cost,
    domain_error,
    from_block_metadata,
    guard_cfg,
    reduce_stream,
    route,
)

if TYPE_CHECKING:
    from .v2 import BlockMetadata

__all__ = [
    "MetadataReductionAnalysis",
    "MetadataRouteComparison",
    "MetadataRouteTelemetry",
    "analyze_contract_reductions",
    "analyze_reduce_axis",
    "analyze_softmax_lse_scores",
    "compare_block_metadata_route",
    "softmax_score_words",
]


@dataclass(frozen=True, slots=True)
class MetadataRouteTelemetry:
    """Telemetry for one logical reduction row."""

    row_index: tuple[int, ...]
    stats: ReductionStats
    domain: AccumulationDomain
    n_terms: int
    n_kept: int
    kept_mask: tuple[bool, ...]
    cost_ref: float
    cost_routed: float
    error_est: float
    phase_unreliable: bool


@dataclass(frozen=True, slots=True)
class MetadataReductionAnalysis:
    """Telemetry batch matching one reduced output shape."""

    output_shape: tuple[int, ...]
    rows: tuple[MetadataRouteTelemetry, ...]

    @property
    def total_terms(self) -> int:
        return sum(row.n_terms for row in self.rows)

    @property
    def total_kept(self) -> int:
        return sum(row.n_kept for row in self.rows)

    @property
    def cost_ref(self) -> float:
        return sum(row.cost_ref for row in self.rows)

    @property
    def cost_routed(self) -> float:
        return sum(row.cost_routed for row in self.rows)

    def domain_histogram(self) -> dict[AccumulationDomain, int]:
        hist: dict[AccumulationDomain, int] = {}
        for row in self.rows:
            hist[row.domain] = hist.get(row.domain, 0) + 1
        return hist


@dataclass(frozen=True, slots=True)
class MetadataRouteComparison:
    """O(N) row stats compared with an O(1) block metadata route."""

    computed: MetadataRouteTelemetry
    header_stats: ReductionStats
    header_domain: AccumulationDomain
    domain_matches: bool
    cost_delta: float
    error_delta: float


def _coerce_reducer(reducer: ReducerSpec | ReducerPolicy | None) -> ReducerSpec:
    if reducer is None:
        return ReducerSpec(policy=ReducerPolicy.AUTO)
    spec = ensure_reducer_spec(reducer)
    if spec.policy != ReducerPolicy.AUTO:
        raise SPFError(
            "metadata route telemetry requires a ReducerSpec with policy=AUTO; "
            f"got {spec.policy.name}"
        )
    return spec


def _select_domain(stats: ReductionStats, spec: ReducerSpec) -> AccumulationDomain:
    if spec.routing == RoutingPolicy.PINNED:
        if spec.domain is None:
            raise SPFError("PINNED routing requires an explicit ReducerSpec.domain")
        return AccumulationDomain(spec.domain)
    if spec.routing == RoutingPolicy.STATIC:
        return route(stats, spec.precision_mode)
    raise SPFError("ADAPTIVE routing is deferred (ship-order step 3); use STATIC or PINNED")


def _cost_for_terms(domain: AccumulationDomain, stats: ReductionStats, n_terms: int) -> float:
    if n_terms == 0:
        return 0.0
    return domain_cost(domain, replace(stats, N=int(n_terms)))


def _error_for_domain(domain: AccumulationDomain, stats: ReductionStats) -> float:
    if stats.N == 0:
        return 0.0
    return domain_error(domain, stats)


def _rows_for_axis(terms: npt.ArrayLike, axis: int | None) -> tuple[np.ndarray, tuple[int, ...]]:
    arr = np.asarray(terms, dtype=np.uint32)
    if axis is None:
        return arr.reshape(1, -1), ()
    moved = np.moveaxis(arr, int(axis), -1)
    return moved.reshape(-1, moved.shape[-1]), tuple(int(v) for v in moved.shape[:-1])


def _row_index(flat_index: int, output_shape: tuple[int, ...]) -> tuple[int, ...]:
    if not output_shape:
        return ()
    return tuple(int(v) for v in np.unravel_index(flat_index, output_shape))


def analyze_reduce_axis(
    terms: npt.ArrayLike,
    reducer: ReducerSpec | ReducerPolicy | None = None,
    *,
    axis: int | None = -1,
    include_guard: bool = True,
) -> MetadataReductionAnalysis:
    """Analyze metadata routing for packed SPF terms without changing outputs.

    ``axis`` follows :func:`spf.tensor.reducers.reduce_axis`: each row along that
    axis is summarized with ``ReductionStats``, a route decision, the dominant
    bypass mask, and planner cost/error estimates.
    """
    spec = _coerce_reducer(reducer)
    rows, output_shape = _rows_for_axis(terms, axis)
    gcfg = guard_cfg(spec.precision_mode)
    keep_rows = _apply_dom_bypass_mask_rows(rows, gcfg.dom_bypass_gap)

    telemetry: list[MetadataRouteTelemetry] = []
    for flat_idx, row in enumerate(rows):
        stats = compute_stats(row)
        domain = _select_domain(stats, spec)
        n_kept = int(np.count_nonzero(keep_rows[flat_idx]))
        phase_unreliable = False
        if include_guard and stats.N > 0:
            phase_unreliable = bool(reduce_stream(row, spec, stats=stats).phase_unreliable)
        telemetry.append(
            MetadataRouteTelemetry(
                row_index=_row_index(flat_idx, output_shape),
                stats=stats,
                domain=domain,
                n_terms=int(stats.N),
                n_kept=n_kept,
                kept_mask=tuple(bool(v) for v in keep_rows[flat_idx].tolist()),
                cost_ref=_cost_for_terms(AccumulationDomain.CARTESIAN, stats, int(stats.N)),
                cost_routed=_cost_for_terms(domain, stats, n_kept),
                error_est=_error_for_domain(domain, stats),
                phase_unreliable=phase_unreliable,
            )
        )
    return MetadataReductionAnalysis(output_shape=output_shape, rows=tuple(telemetry))


def compare_block_metadata_route(
    words: npt.ArrayLike,
    metadata: BlockMetadata,
    reducer: ReducerSpec | ReducerPolicy | None = None,
) -> MetadataRouteComparison:
    """Compare O(N) row routing with an O(1) ``BlockMetadata`` route."""
    spec = _coerce_reducer(reducer)
    analysis = analyze_reduce_axis(words, spec, axis=None, include_guard=False)
    computed = analysis.rows[0]
    header_stats = from_block_metadata(metadata)
    header_domain = _select_domain(header_stats, spec)
    header_cost = _cost_for_terms(header_domain, header_stats, int(header_stats.N))
    header_error = _error_for_domain(header_domain, header_stats)
    return MetadataRouteComparison(
        computed=computed,
        header_stats=header_stats,
        header_domain=header_domain,
        domain_matches=header_domain == computed.domain,
        cost_delta=header_cost - computed.cost_routed,
        error_delta=header_error - computed.error_est,
    )


def analyze_contract_reductions(
    A_words: npt.ArrayLike,
    B_words: npt.ArrayLike,
    reducer: ReducerSpec | ReducerPolicy | None = None,
    *,
    transpose_b: bool = False,
    causal: bool = False,
    include_guard: bool = True,
) -> MetadataReductionAnalysis:
    """Analyze the product-row reductions used by ``execute_contract``.

    Causal-masked output positions are represented as all-zero reduction rows so
    telemetry reflects the effective output surface instead of unused reductions.
    """
    products = _contract_product_terms(
        A_words,
        B_words,
        transpose_b=transpose_b,
        causal=causal,
        apply_causal_mask=causal,
    )
    return analyze_reduce_axis(products, reducer, axis=-1, include_guard=include_guard)


def _lq_to_spf32_l(lq: np.ndarray, l_frac_bits: int) -> np.ndarray:
    q = np.asarray(lq, dtype=np.int64)
    shift = int(l_frac_bits) - 8
    if shift > 0:
        half = np.int64(1 << (shift - 1))
        return np.where(q >= 0, (q + half) >> shift, -((-q + half) >> shift))
    if shift < 0:
        return q << int(-shift)
    return q


def softmax_score_words(
    scores: npt.ArrayLike,
    *,
    causal: bool = False,
    l_frac_bits: int = 16,
    beta: float = 1.0,
    pre_quantized: bool = False,
) -> np.ndarray:
    """Convert softmax/LSE score rows to SPF-like words for metadata analysis.

    The exact softmax path remains floating-point. This helper only builds
    theta-zero SPF words that approximate the row's ``2**score`` terms so the
    existing ``ReductionStats`` router can analyze score-row shape.
    """
    f = int(l_frac_bits)
    if f < 1 or f > 24:
        raise ValueError(f"softmax_score_words: l_frac_bits must be in [1,24], got {f}")
    if pre_quantized:
        sc_in = np.asarray(scores)
        if not np.issubdtype(sc_in.dtype, np.integer):
            raise ValueError(
                "softmax_score_words: pre_quantized=True requires integer L-code scores, "
                f"got dtype {sc_in.dtype}"
            )
        if beta != 1.0:
            raise ValueError(
                "softmax_score_words: beta must already be folded into pre-quantized "
                "L-codes - pass beta=1.0"
            )
        lq = sc_in.astype(np.int32, copy=False)
    else:
        lq = quantize_scores_lq(scores, beta=beta, l_frac_bits=f)
    if lq.ndim != 2:
        raise ValueError(f"softmax_score_words: scores must be 2-D, got shape {lq.shape}")

    n_rows, row_len = lq.shape
    finite = lq != np.int32(_SENTINEL_I32)
    if causal:
        cols = np.arange(row_len)[np.newaxis, :]
        valid = cols <= np.arange(n_rows)[:, np.newaxis]
        finite = finite & valid

    lq64 = lq.astype(np.int64, copy=False)
    any_finite = np.any(finite, axis=1)
    row_max = np.max(np.where(finite, lq64, np.iinfo(np.int64).min), axis=1)
    row_max = np.where(any_finite, row_max, 0)
    lq_rel = np.where(finite, lq64 - row_max[:, np.newaxis], np.int64(_SENTINEL_I32))

    l8 = _lq_to_spf32_l(lq_rel, f)
    finite = finite & (l8 >= int(SPF32_L_MIN_FINITE))
    l8 = np.clip(l8, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE).astype(np.int16)
    l_codes = np.where(finite, l8, np.int16(int(SPF32_L_ZERO))).astype(np.int16)
    theta = np.zeros_like(l_codes, dtype=np.uint16)
    return pack_words(l_codes, theta).astype(np.uint32, copy=False)


def analyze_softmax_lse_scores(
    scores: npt.ArrayLike,
    reducer: ReducerSpec | ReducerPolicy | None = None,
    *,
    causal: bool = False,
    l_frac_bits: int = 16,
    beta: float = 1.0,
    pre_quantized: bool = False,
    include_guard: bool = True,
) -> MetadataReductionAnalysis:
    """Analyze softmax/LSE score rows through the metadata router."""
    words = softmax_score_words(
        scores,
        causal=causal,
        l_frac_bits=l_frac_bits,
        beta=beta,
        pre_quantized=pre_quantized,
    )
    return analyze_reduce_axis(words, reducer, axis=1, include_guard=include_guard)
