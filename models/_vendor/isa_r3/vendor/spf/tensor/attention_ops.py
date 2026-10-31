"""Python reference implementations for 'contract' and 'softmax_lse' ops.

These are called by :class:`~spf.tensor.v2.ExecutionPlan`'s ``execute()``
dispatcher. They mirror the C drivers in ``native/src/scalar/spf32/tensor/``
bit-exactly (when the native library is available the tests verify parity).

Both helpers operate purely on numpy arrays and import nothing from outside
the spf.tensor package.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    SPF32_PHASE_MASK,
    SPF32_WORD_ZERO,
    pack_words,
    unpack_L,
    unpack_theta,
)

__all__ = ["execute_contract", "execute_softmax_lse", "quantize_scores_lq"]

# ---------------------------------------------------------------------------
# Internal constants (mirrors of C header defines)
# ---------------------------------------------------------------------------

# Used by softmax_lse to mirror SPFT_L_SENTINEL = INT32_MIN
_SENTINEL_I32: int = -(1 << 31)
# |Lq| ceiling that maps to overflow / underflow (2^40)
_L_CLAMP: int = 1 << 40


# ---------------------------------------------------------------------------
# contract
# ---------------------------------------------------------------------------


def _contract_product_terms(
    A_words: npt.ArrayLike,
    B_words: npt.ArrayLike,
    *,
    transpose_b: bool = False,
    causal: bool = False,
    apply_causal_mask: bool = False,
) -> np.ndarray:
    """Build unreduced SPF product streams for ``execute_contract``.

    Returns a uint32 array of shape ``(M, N, K)`` where the final axis is the
    reduction stream for one output element. ``apply_causal_mask`` is for
    analysis-only callers that want masked rows represented as all-zero streams;
    ``execute_contract`` keeps its existing post-reduction mask.
    """
    A = np.asarray(A_words, dtype=np.uint32)
    B_raw = np.asarray(B_words, dtype=np.uint32)

    if A.ndim != 2:
        raise ValueError(f"contract: A must be 2-D, got shape {A.shape}")
    if B_raw.ndim != 2:
        raise ValueError(f"contract: B must be 2-D, got shape {B_raw.shape}")

    M, K = A.shape
    if transpose_b:
        # B already supplied as (N, K); use directly — B_t[j, k] = B_raw[j, k]
        B_t = B_raw  # shape (N, K)
        N = B_t.shape[0]
        if B_t.shape[1] != K:
            raise ValueError(
                f"contract: transpose_b=True but B.shape[1]={B_t.shape[1]} != K={K}"
            )
    else:
        # B is (K, N); transpose to (N, K) so row j is the j-th column of B
        if B_raw.shape[0] != K:
            raise ValueError(
                f"contract: B.shape[0]={B_raw.shape[0]} != K={K}"
            )
        B_t = B_raw.T.copy()  # (N, K), C-contiguous
        N = B_raw.shape[1]

    # Unpack planes once
    aL = unpack_L(A)          # (M, K) int16
    aT = unpack_theta(A)      # (M, K) uint16
    bL = unpack_L(B_t)        # (N, K) int16
    bT = unpack_theta(B_t)    # (N, K) uint16

    # Build product array: shape (M, N, K)
    # A[i, k] plane broadcast with B_t[j, k] plane
    aL3 = aL[:, np.newaxis, :]   # (M, 1, K)
    aT3 = aT[:, np.newaxis, :]
    bL3 = bL[np.newaxis, :, :]   # (1, N, K)
    bT3 = bT[np.newaxis, :, :]

    zero_mask = (aL3 == np.int16(int(SPF32_L_ZERO))) | (bL3 == np.int16(int(SPF32_L_ZERO)))

    # Saturating L add
    pL64 = aL3.astype(np.int64) + bL3.astype(np.int64)
    pL = np.clip(pL64, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE).astype(np.int16)
    # Modular theta add
    pT = ((aT3.astype(np.uint32) + bT3.astype(np.uint32)) & SPF32_PHASE_MASK).astype(np.uint16)

    # Zero sentinel annihilates
    pL = np.where(zero_mask, np.int16(int(SPF32_L_ZERO)), pL)
    pT = np.where(zero_mask, np.uint16(0), pT)

    if causal and apply_causal_mask:
        i_idx = np.arange(M)[:, np.newaxis, np.newaxis]
        j_idx = np.arange(N)[np.newaxis, :, np.newaxis]
        masked = j_idx > i_idx
        pL = np.where(masked, np.int16(int(SPF32_L_ZERO)), pL)
        pT = np.where(masked, np.uint16(0), pT)

    return pack_words(pL, pT)  # uint32 (M, N, K)


def execute_contract(
    A_words: npt.ArrayLike,
    B_words: npt.ArrayLike,
    reducer: object,
    *,
    transpose_b: bool = False,
    causal: bool = False,
) -> np.ndarray:
    """Python reference for the ``contract`` execution op.

    Computes ``C[i, j] = reduce_k(A[i, k] ⊗ B[k, j])``.

    Parameters
    ----------
    A_words:
        uint32 array, shape ``(M, K)`` — packed SPF-32 words.
    B_words:
        uint32 array, shape ``(K, N)`` normally, or ``(N, K)`` when
        ``transpose_b=True`` (the caller already supplies the transposed layout).
    reducer:
        A :class:`~spf.tensor.core.ReducerSpec` passed to
        :func:`~spf.tensor.reducers.reduce_axis`.
    transpose_b:
        When True B is shaped ``(N, K)`` and is used as-is; when False B is
        shaped ``(K, N)`` and is transposed internally so every inner product
        reads a contiguous K-run.
    causal:
        When True, output elements ``C[i, j]`` with ``j > i`` are forced to
        ``SPF32_WORD_ZERO`` after reduction (upper-triangular mask).

    Returns
    -------
    np.ndarray of dtype uint32, shape ``(M, N)``
    """
    from . import reducers  # local import to avoid circular at module level

    products = _contract_product_terms(
        A_words,
        B_words,
        transpose_b=transpose_b,
        causal=causal,
        apply_causal_mask=False,
    )
    M, N = products.shape[:2]

    # Reduce over axis=-1 (the K axis) → (M, N)
    C = reducers.reduce_axis(products, reducer, axis=-1)

    # Causal mask: j > i → WORD_ZERO
    if causal:
        i_idx = np.arange(M)[:, np.newaxis]   # (M, 1)
        j_idx = np.arange(N)[np.newaxis, :]   # (1, N)
        C = np.where(j_idx > i_idx, np.uint32(int(SPF32_WORD_ZERO)), C).astype(np.uint32)

    return C.astype(np.uint32)


# ---------------------------------------------------------------------------
# softmax_lse
# ---------------------------------------------------------------------------

# C llround: round-half-away from zero
def _half_away_round_scalar(x: float) -> int:
    """Scalar round-half-away (mirrors C ``llround``)."""
    if x >= 0.0:
        return math.floor(x + 0.5)
    else:
        return -math.floor(-x + 0.5)


def _half_away_round(x: npt.ArrayLike) -> np.ndarray:
    """Vectorised round-half-away (mirrors C ``llround``)."""
    a = np.asarray(x, dtype=np.float64)
    return np.sign(a) * np.floor(np.abs(a) + 0.5)


def _quantize_score(s: float, l_per_unit: float) -> int:
    """Quantize one score to an int32 L-code; mirrors spft_quantize_score."""
    if math.isnan(s) or s == -math.inf:
        return _SENTINEL_I32
    q = s * l_per_unit
    if q >= _L_CLAMP:
        return min(int(_L_CLAMP), (1 << 31) - 1)  # INT32_MAX clamp
    if q <= -_L_CLAMP:
        return _SENTINEL_I32
    v = _half_away_round_scalar(q)
    if v > (1 << 31) - 1:
        return (1 << 31) - 1
    if v < -(1 << 31) + 1:
        return _SENTINEL_I32
    return int(v)


def quantize_scores_lq(
    scores: npt.ArrayLike,
    *,
    beta: float = 1.0,
    l_frac_bits: int = 16,
) -> np.ndarray:
    """Vectorised score → int32 L-code quantizer for the softmax_lse ingress.

    Elementwise bit-identical to the scalar :func:`_quantize_score` with
    ``l_per_unit = beta * 2**l_frac_bits / ln 2`` (the scalar loop stays as the
    parity reference; tests hold the two together). The resulting codes mean
    ``lq = round_half_away(score · beta · 2^f / ln 2)`` — beta is FOLDED into
    the codes, so a downstream ``execute_softmax_lse(..., pre_quantized=True)``
    call must receive ``beta=1.0`` and the SAME ``l_frac_bits``.
    """
    sc = np.asarray(scores, dtype=np.float64)
    f = l_frac_bits
    if f < 1 or f > 24:
        raise ValueError(f"quantize_scores_lq: l_frac_bits must be in [1,24], got {f}")
    if beta == 0.0:
        beta = 1.0
    l_per_unit = beta * float(1 << f) / math.log(2.0)
    int32_max = (1 << 31) - 1

    q = sc * l_per_unit
    nan_or_neginf = np.isnan(sc) | (sc == -np.inf)
    hi = q >= float(_L_CLAMP)          # NaN compares False — safe
    lo = q <= -float(_L_CLAMP)
    special = nan_or_neginf | hi | lo
    v = _half_away_round(np.where(special, 0.0, q))
    v = np.where(v > float(int32_max), float(int32_max), v)
    v = np.where(v < float(-(1 << 31) + 1), float(_SENTINEL_I32), v)
    out = v.astype(np.int64)
    out = np.where(hi, np.int64(int32_max), out)
    out = np.where(lo | nan_or_neginf, np.int64(_SENTINEL_I32), out)
    return out.astype(np.int32)


def execute_softmax_lse(
    scores: npt.ArrayLike,
    reducer: object,
    *,
    causal: bool = False,
    l_frac_bits: int = 16,
    beta: float = 1.0,
    out: str = "f32",
    pre_quantized: bool = False,
) -> np.ndarray:
    """Python reference for the ``softmax_lse`` execution op.

    Mirrors ``spft_softmax_driver`` from ``native/src/scalar/spf32/tensor/spft_softmax.c``
    exactly (the ``SPFT_LSE_EXACT`` / ``SPFT_OUT_F32`` or ``SPFT_OUT_WORDS`` path).

    Parameters
    ----------
    scores:
        float32 or float64 array, shape ``(T_rows, T_cols)``.
    reducer:
        Required by the op contract (records the reducer boundary); not used
        for the computation, which is floating-point throughout. Passed through
        so the ExecutionPlan's reducer_boundaries reflects this step.
    causal:
        Row ``i`` has only the first ``i+1`` columns valid (causal mask).
    l_frac_bits:
        Fractional bits ``f`` (default 16; 8 = SPF-32 faithful variant).
    beta:
        Score scaling (default 1.0; the C ``opts->beta``).
    out:
        ``"f32"`` → return float32 weights; ``"words"`` → return packed uint32
        SPF-32 words (requantized with ``shift = f - 8``).
    pre_quantized:
        When True, ``scores`` must be an INTEGER array of int32 L-codes already
        quantized with the same ``l_frac_bits`` (see
        :func:`quantize_scores_lq`); the fp quantize loop is skipped and the
        LSE runs directly on the codes. ``beta`` must be 1.0 (it is folded into
        the codes by the producer — a non-unit beta here would double-count).
        Sentinel entries (``INT32_MIN``) are treated as masked. Python-only
        ingress for now: the C ``spft_softmax_driver`` has no L-code input mode
        yet.

    Returns
    -------
    np.ndarray
        Shape ``(T_rows, T_cols)``. dtype float32 for ``out="f32"``; uint32
        for ``out="words"``.
    """
    lq_arr: np.ndarray | None = None
    if pre_quantized:
        sc_in = np.asarray(scores)
        if not np.issubdtype(sc_in.dtype, np.integer):
            raise ValueError(
                "softmax_lse: pre_quantized=True requires integer L-code scores, "
                f"got dtype {sc_in.dtype}"
            )
        if beta != 1.0:
            raise ValueError(
                "softmax_lse: beta must already be folded into pre-quantized "
                "L-codes — pass beta=1.0"
            )
        lq_arr = sc_in.astype(np.int32, copy=False)
        sc = lq_arr.astype(np.float64)  # only for shape bookkeeping below
    else:
        sc = np.asarray(scores, dtype=np.float64)
    if sc.ndim != 2:
        raise ValueError(f"softmax_lse: scores must be 2-D, got shape {sc.shape}")

    n_rows, row_len = sc.shape
    f = l_frac_bits
    if f < 1 or f > 24:
        raise ValueError(f"softmax_lse: l_frac_bits must be in [1,24], got {f}")
    if beta == 0.0:
        beta = 1.0

    # l_per_unit = beta * 2^f / ln2  (same as C: beta * (1<<f) / M_LN2)
    l_per_unit: float = beta * float(1 << f) / math.log(2.0)
    inv_scale: float = 1.0 / float(1 << f)

    if out == "f32":
        result = np.zeros((n_rows, row_len), dtype=np.float32)
    elif out == "words":
        result = np.full((n_rows, row_len), int(SPF32_WORD_ZERO), dtype=np.uint32)
    else:
        raise ValueError(f"softmax_lse: out must be 'f32' or 'words', got {out!r}")

    for i in range(n_rows):
        valid_len = (i + 1) if causal else row_len

        # Quantize row (or take pre-quantized L-codes directly)
        lq = []
        for j in range(row_len):
            if j >= valid_len:
                lq.append(_SENTINEL_I32)
            elif lq_arr is not None:
                lq.append(int(lq_arr[i, j]))
            else:
                lq.append(_quantize_score(float(sc[i, j]), l_per_unit))

        # LSE (exact path: fp64 sum of 2^((lq-m)/2^f), then log2 back)
        m = _SENTINEL_I32
        for v in lq:
            if v != _SENTINEL_I32 and v > m:
                m = v
        if m == _SENTINEL_I32:
            # all-masked row: weights stay 0 / WORD_ZERO
            continue

        s_sum = 0.0
        for v in lq:
            if v != _SENTINEL_I32:
                s_sum += math.exp2(float(v - m) * inv_scale)

        corr = _half_away_round_scalar(math.log2(s_sum) * float(1 << f))
        lse64 = m + corr
        # Clamp lse to int32 range (same logic as C)
        if lse64 > (1 << 31) - 1:
            lse64 = (1 << 31) - 1
        if lse64 < -(1 << 31) + 1:
            lse64 = _SENTINEL_I32

        # Emit weights
        if out == "f32":
            for j in range(row_len):
                if lq[j] == _SENTINEL_I32 or lse64 == _SENTINEL_I32:
                    result[i, j] = 0.0
                else:
                    result[i, j] = float(
                        math.exp2(float(lq[j] - lse64) * inv_scale)
                    )
        else:  # "words"
            shift = f - 8  # requantize wide L to SPF-32 L (f >= 8)
            # SPF32_L_CODE_MIN_FINITE = -32767, SPF32_L_CODE_MAX_FINITE = 32766
            l_min = SPF32_L_MIN_FINITE  # -32767
            l_max = SPF32_L_MAX_FINITE  # 32766
            for j in range(row_len):
                if lq[j] == _SENTINEL_I32 or lse64 == _SENTINEL_I32:
                    result[i, j] = np.uint32(int(SPF32_WORD_ZERO))
                    continue
                lw = int(lq[j]) - int(lse64)  # always <= 0
                if shift > 0:
                    half = 1 << (shift - 1)
                    l8 = (lw + half) >> shift if lw >= 0 else -((-lw + half) >> shift)
                elif shift < 0:
                    l8 = lw << (-shift)
                else:
                    l8 = lw
                if l8 < l_min:
                    result[i, j] = np.uint32(int(SPF32_WORD_ZERO))
                else:
                    if l8 > l_max:
                        l8 = l_max
                    # pack: L in upper int16 bits, theta=0
                    l16 = np.int16(l8)
                    result[i, j] = pack_words(l16, np.uint16(0))

    return result
