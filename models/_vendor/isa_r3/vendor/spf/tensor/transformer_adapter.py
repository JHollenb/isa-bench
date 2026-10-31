"""Transformer-facing SPFBlockMatrix adapter helpers.

These helpers do not add a transformer runtime. They compile common transformer
weight surfaces into typed SPFBlockMatrix kernels so the algebra layer can apply
the same proof, reducer, and lazy residual-stream rules as the lower-level API.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .core import ReducerSpec, SPFError
from .kernel import AxisSpec, SPFBlockMatrix, SPFKernel, lazy_sum, spfblock_einsum


@dataclass(frozen=True)
class TransformerQKVKernels:
    query: SPFBlockMatrix
    key: SPFBlockMatrix
    value: SPFBlockMatrix

    def apply(self, x: np.ndarray) -> tuple[SPFKernel, SPFKernel, SPFKernel]:
        return self.query.apply(x), self.key.apply(x), self.value.apply(x)


def transformer_projection_kernel(
    weights: np.ndarray,
    *,
    out_axis: str,
    in_axis: str = "d_model",
    basis: str = "standard",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    arr = np.asarray(weights)
    if arr.ndim != 2:
        raise SPFError("transformer projection weights must be rank-2 [out, in]")
    return SPFBlockMatrix.dense(
        weights,
        out_axes=(AxisSpec(out_axis, arr.shape[0], role="codomain", basis=basis),),
        in_axes=(AxisSpec(in_axis, arr.shape[1], role="domain", basis=basis),),
        reducer_policy=reducer_policy,
    )


def transformer_embedding_kernel(
    embedding_table: np.ndarray,
    *,
    vocab_axis: str = "vocab",
    out_axis: str = "d_model",
    basis: str = "standard",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    table = np.asarray(embedding_table)
    if table.ndim != 2:
        raise SPFError("transformer embedding table must be rank-2 [vocab, d_model]")
    return SPFBlockMatrix.dense(
        table.T,
        out_axes=(AxisSpec(out_axis, int(table.shape[1]), role="codomain", basis=basis),),
        in_axes=(AxisSpec(vocab_axis, int(table.shape[0]), role="domain", basis=basis),),
        reducer_policy=reducer_policy,
    )


def _require_rank2(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise SPFError(f"{name} must be rank-2 [out, in]")
    return arr


def _kernel_axis_size(kernel: SPFKernel, axis: str) -> int:
    for spec in kernel.axes:
        if spec.name == axis:
            return int(spec.size)
    raise SPFError(f"kernel is missing axis {axis!r}")


def _activation_shape(value: SPFKernel | np.ndarray, axes: tuple[str, ...]) -> tuple[int, ...]:
    if isinstance(value, SPFKernel):
        if value.sig.out_axes != axes or value.sig.in_axes:
            raise SPFError("activation kernel axes must match requested output axes and have no input axes")
        return tuple(_kernel_axis_size(value, axis) for axis in axes)
    arr = np.asarray(value)
    if arr.ndim != len(axes):
        raise SPFError("activation array rank must match requested output axes")
    return tuple(int(dim) for dim in arr.shape)


def transformer_low_rank_projection_kernel(
    u: np.ndarray,
    v: np.ndarray,
    coeff: np.ndarray | None = None,
    *,
    out_axis: str,
    in_axis: str = "d_model",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.low_rank(
        u,
        v,
        coeff,
        out_axis=out_axis,
        in_axis=in_axis,
        reducer_policy=reducer_policy,
    )


def transformer_circulant_projection_kernel(
    spectrum: np.ndarray,
    *,
    out_axis: str,
    in_axis: str = "d_model",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.circulant_spectrum(
        spectrum,
        out_axis=out_axis,
        axis=in_axis,
        reducer_policy=reducer_policy,
    )


def transformer_block_circulant_projection_kernel(
    spectrum: np.ndarray | dict[str, object],
    *,
    out_axis: str,
    in_axis: str = "d_model",
    block_axes: tuple[str, str] = ("block_out", "block_in"),
    inner_structure: str = "dense",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.block_circulant_spectrum(
        spectrum,
        out_axis=out_axis,
        in_axis=in_axis,
        block_axes=block_axes,
        inner_structure=inner_structure,
        reducer_policy=reducer_policy,
    )


def transformer_product_character_projection_kernel(
    *,
    moduli: Sequence[int],
    out_frequency: Sequence[int],
    in_frequency: Sequence[int],
    out_axis: str,
    in_axis: str = "d_model",
    scale: complex = 1.0 + 0.0j,
    exact_group_semantics: bool = True,
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.product_character(
        moduli=moduli,
        out_frequency=out_frequency,
        in_frequency=in_frequency,
        out_axis=out_axis,
        in_axis=in_axis,
        scale=scale,
        exact_group_semantics=exact_group_semantics,
        reducer_policy=reducer_policy,
    )


def transformer_rope_position_kernel(
    n_positions: int,
    head_dim: int,
    *,
    position_axis: str = "position",
    feature_axis: str = "d_model",
    base: float = 10000.0,
    position_origin: int = 0,
    exactness: str = "quantized",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.rope(
        n_positions,
        head_dim,
        out_axis=position_axis,
        in_axis=feature_axis,
        base=base,
        position_origin=position_origin,
        exactness=exactness,
        reducer_policy=reducer_policy,
    )


def transformer_clock_position_kernel(
    n_ticks: int,
    omega: Sequence[int] | np.ndarray,
    *,
    position_axis: str = "position",
    feature_axis: str = "d_model",
    tick_origin: int = 0,
    exactness: str = "quantized",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.clock(
        n_ticks,
        omega,
        out_axis=position_axis,
        in_axis=feature_axis,
        tick_origin=tick_origin,
        exactness=exactness,
        reducer_policy=reducer_policy,
    )


def transformer_jacobian_projection_kernel(
    matrix: np.ndarray,
    *,
    out_axis: str,
    in_axis: str = "d_model",
    provenance: dict[str, object] | None = None,
    residual: np.ndarray | None = None,
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.jacobian_operator(
        matrix,
        out_axis=out_axis,
        in_axis=in_axis,
        provenance=provenance,
        residual=residual,
        reducer_policy=reducer_policy,
    )


def transformer_residual_patch_kernel(
    base: SPFKernel,
    patch: SPFKernel,
    *,
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.residual_patch(
        base,
        patch,
        reducer_policy=reducer_policy,
    )


def transformer_kv_cache_kernel(
    cache_words: np.ndarray,
    *,
    position_axis: str = "position",
    feature_axis: str = "d_model",
    cache_axis: str = "kv",
    position_group: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    words = np.asarray(cache_words, dtype=np.uint32)
    if words.ndim not in {2, 3}:
        raise SPFError("KV cache words must be rank-2 [position, feature] or rank-3 [kv, position, feature]")
    if len({position_axis, feature_axis, cache_axis}) != 3:
        raise SPFError("KV cache axes must have distinct names")
    meta = {"kind": "kv_cache", **dict(metadata or {})}
    position_meta = {**meta, "role": "position"}
    if position_group is not None:
        moduli = tuple(int(x) for x in position_group.get("moduli", ()))
        if moduli and int(np.prod(moduli, dtype=np.int64)) != int(words.shape[-2]):
            raise SPFError("KV cache position product-group size must match position axis")
    position = AxisSpec(
        position_axis,
        int(words.shape[-2]),
        role="codomain",
        basis="product_group" if position_group is not None else "standard",
        group=None if position_group is None else dict(position_group),
        metadata=position_meta,
    )
    feature = AxisSpec(
        feature_axis,
        int(words.shape[-1]),
        role="codomain",
        metadata={**meta, "role": "feature"},
    )
    if words.ndim == 2:
        return SPFBlockMatrix.from_words(
            words,
            out_axes=(position, feature),
            in_axes=(),
            reducer_policy=reducer_policy,
        )
    cache = AxisSpec(
        cache_axis,
        int(words.shape[0]),
        role="batch",
        metadata={**meta, "role": "cache"},
    )
    return SPFBlockMatrix.from_words(
        words,
        batch_axes=(cache,),
        out_axes=(position, feature),
        in_axes=(),
        reducer_policy=reducer_policy,
    )


def transformer_qkv_kernels(
    q_weights: np.ndarray,
    k_weights: np.ndarray,
    v_weights: np.ndarray,
    *,
    in_axis: str = "d_model",
    q_axis: str = "q",
    k_axis: str = "k",
    v_axis: str = "v",
    reducer_policy: ReducerSpec | None = None,
) -> TransformerQKVKernels:
    q_arr = _require_rank2("Q projection weights", q_weights)
    k_arr = _require_rank2("K projection weights", k_weights)
    v_arr = _require_rank2("V projection weights", v_weights)
    if q_arr.shape[1] != k_arr.shape[1] or q_arr.shape[1] != v_arr.shape[1]:
        raise SPFError("Q/K/V projections must share the same input width")
    q = transformer_projection_kernel(q_arr, out_axis=q_axis, in_axis=in_axis, reducer_policy=reducer_policy)
    k = transformer_projection_kernel(k_arr, out_axis=k_axis, in_axis=in_axis, reducer_policy=reducer_policy)
    v = transformer_projection_kernel(v_arr, out_axis=v_axis, in_axis=in_axis, reducer_policy=reducer_policy)
    return TransformerQKVKernels(query=q, key=k, value=v)


def transformer_mlp_kernel(
    up_weights: np.ndarray,
    down_weights: np.ndarray,
    *,
    in_axis: str = "d_model",
    hidden_axis: str = "mlp_hidden",
    out_axis: str = "d_model_out",
    reducer_policy: ReducerSpec | None = None,
) -> SPFKernel:
    up_arr = _require_rank2("MLP up projection weights", up_weights)
    # This adapter represents only the linear down @ up projection pair.
    # Nonlinear address circuits belong between projections in ExecutionPlan.
    down_arr = _require_rank2("MLP down projection weights", down_weights)
    if up_arr.shape[0] != down_arr.shape[1]:
        raise SPFError("MLP down projection input width must match up projection hidden width")
    up = transformer_projection_kernel(
        up_arr,
        out_axis=hidden_axis,
        in_axis=in_axis,
        reducer_policy=reducer_policy,
    )
    down = transformer_projection_kernel(
        down_arr,
        out_axis=out_axis,
        in_axis=hidden_axis,
        reducer_policy=reducer_policy,
    )
    return down @ up


def transformer_residual_stream(
    terms: Sequence[SPFKernel],
    *,
    reducer: ReducerSpec | None = None,
) -> SPFKernel:
    return lazy_sum(tuple(terms), reducer=reducer)


def _format_adapter_einsum_labels(labels: Sequence[str]) -> str:
    return "".join(f"[{label}]" for label in labels)


def _activation_kernel(
    value: SPFKernel | np.ndarray,
    *,
    axes: tuple[str, ...],
    reducer_policy: ReducerSpec | None,
) -> SPFKernel:
    if isinstance(value, SPFKernel):
        _activation_shape(value, axes)
        return value
    _activation_shape(value, axes)
    return SPFBlockMatrix.dense(
        value,
        out_axes=axes,
        in_axes=(),
        reducer_policy=reducer_policy,
    )


def transformer_attention_scores(
    query_states: SPFKernel | np.ndarray,
    key_states: SPFKernel | np.ndarray,
    *,
    query_axis: str = "q",
    key_axis: str = "k",
    feature_axis: str = "h",
    reducer_policy: ReducerSpec | None = None,
) -> SPFKernel:
    if len({query_axis, key_axis, feature_axis}) != 3:
        raise SPFError("attention query, key, and feature axes must have distinct names")
    q_shape = _activation_shape(query_states, (query_axis, feature_axis))
    k_shape = _activation_shape(key_states, (key_axis, feature_axis))
    if q_shape[1] != k_shape[1]:
        raise SPFError("attention query/key feature widths must match")
    q = _activation_kernel(
        query_states,
        axes=(query_axis, feature_axis),
        reducer_policy=reducer_policy,
    )
    k = _activation_kernel(
        key_states,
        axes=(key_axis, feature_axis),
        reducer_policy=reducer_policy,
    )
    return spfblock_einsum(
        f"{_format_adapter_einsum_labels((query_axis, feature_axis))},"
        f"{_format_adapter_einsum_labels((key_axis, feature_axis))}->"
        f"{_format_adapter_einsum_labels((query_axis, key_axis))}",
        q,
        k,
        reducer=reducer_policy,
    )


def transformer_relative_attention(
    scores: SPFKernel,
    relative_bias: SPFKernel | np.ndarray,
    *,
    reducer: ReducerSpec | None = None,
) -> SPFKernel:
    if not isinstance(scores, SPFKernel):
        raise SPFError("relative attention scores must be an SPFKernel")
    if len(scores.sig.out_axes) != 2 or scores.sig.in_axes:
        raise SPFError("relative attention scores must be a rank-2 activation kernel")
    if isinstance(relative_bias, SPFKernel):
        if relative_bias.sig.out_axes != scores.sig.out_axes or relative_bias.sig.in_axes:
            raise SPFError("relative attention bias axes must match scores and have no input axes")
        bias = relative_bias
    else:
        arr = np.asarray(relative_bias)
        score_shape = tuple(_kernel_axis_size(scores, axis) for axis in scores.sig.out_axes)
        if arr.shape != score_shape:
            raise SPFError("relative attention bias shape must match scores")
        bias = SPFBlockMatrix.dense(
            relative_bias,
            out_axes=scores.sig.out_axes,
            in_axes=(),
            reducer_policy=scores.sig.reducer_policy,
        )
    return lazy_sum((scores, bias), reducer=reducer or scores.sig.reducer_policy)


def transformer_gauge_norm_kernel(
    scale: np.ndarray,
    *,
    axis: str = "d_model",
    out_axis: str = "d_model_out",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return SPFBlockMatrix.gauge(
        scale,
        out_axis=out_axis,
        axis=axis,
        reducer_policy=reducer_policy,
    )


def transformer_readout_kernel(
    weights: np.ndarray,
    *,
    in_axis: str = "d_model",
    out_axis: str = "vocab",
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return transformer_projection_kernel(weights, out_axis=out_axis, in_axis=in_axis, reducer_policy=reducer_policy)


def transformer_character_readout_kernel(
    *,
    moduli: Sequence[int],
    out_frequency: Sequence[int],
    in_frequency: Sequence[int],
    in_axis: str = "d_model",
    out_axis: str = "vocab",
    scale: complex = 1.0 + 0.0j,
    exact_group_semantics: bool = True,
    reducer_policy: ReducerSpec | None = None,
) -> SPFBlockMatrix:
    return transformer_product_character_projection_kernel(
        moduli=moduli,
        out_frequency=out_frequency,
        in_frequency=in_frequency,
        out_axis=out_axis,
        in_axis=in_axis,
        scale=scale,
        exact_group_semantics=exact_group_semantics,
        reducer_policy=reducer_policy,
    )


__all__ = [
    "TransformerQKVKernels",
    "transformer_attention_scores",
    "transformer_block_circulant_projection_kernel",
    "transformer_character_readout_kernel",
    "transformer_circulant_projection_kernel",
    "transformer_clock_position_kernel",
    "transformer_embedding_kernel",
    "transformer_gauge_norm_kernel",
    "transformer_jacobian_projection_kernel",
    "transformer_kv_cache_kernel",
    "transformer_low_rank_projection_kernel",
    "transformer_mlp_kernel",
    "transformer_product_character_projection_kernel",
    "transformer_projection_kernel",
    "transformer_qkv_kernels",
    "transformer_readout_kernel",
    "transformer_relative_attention",
    "transformer_residual_patch_kernel",
    "transformer_residual_stream",
    "transformer_rope_position_kernel",
]
