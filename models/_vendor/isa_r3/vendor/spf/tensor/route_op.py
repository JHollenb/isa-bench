"""Fused phase-native attention ``route`` op — words in, words out, no decode.

The three stages of attention, split where the measurements split them, all
inside one op so no caller can re-introduce the fp32 score hop (the cl-0241
phase-drop at ``np.real(decode_words(scores))``):

    A — ADDRESS (pure Θ, integer):  s[i,j] = Σ_k cos(Θq[i,k] − Θk[j,k])
        ΔΘ is the u16 ring subtract; cos comes from a 65536-entry twiddle
        LUT; the fp64 coherence sum is quantized ONCE to int32 L-codes with
        beta folded in. Magnitude is never read (cl-0180/0181: magnitude
        destroys the world axis). Causal mask = sentinel L-codes, HERE.
    B — GATE (pure L, integer):     P = softmax_lse(lq, pre_quantized=True)
        The log-domain LSE runs directly on the codes; weights emerge as
        (L ≤ 0, Θ=0) SPF words. Softmax is load-bearing (cl-0180) — kept.
    C — MIX (the one coherent sum): delta = contract(P_words, v_words)
        Per-term multiply is an integer L add with Θ_V passing through
        exactly (P words are real-positive); the reduction over memory is
        the single place complex addition exists in the op. The reducer
        boundary chooses the accumulator: CARTESIAN pivoted-fp64 pairwise
        (reference) or PINNED LOGPOLAR (the native ``spf32.add`` FB7 fold).

Design constraints carried from the measurements (panr-core-route arc):
- FULL memory only — rank truncation and prototype clustering both re-lose
  the world axis (cl-0180/cl-0181). There is deliberately no rank parameter
  and no prototype parameter.
- Addressing is REAL phase concordance (``Σcos(ΔΘ) = Re(Σe^{iΔΘ})``,
  measured-by-algebra; cl-0291/0292/0295: concordance crushes magnitude and
  ties whiten+L2 at 1e-6 parity). The complex-phasor scorer was a discarded
  detour (cl-0241/0242 retracted) — do not "upgrade" stage A to complex.
- The _COS_LUT is a twiddle-style SCORER table. It is NOT inside the
  ``spf32.add`` path (the spf-add-path no-Sb-LUT rule is untouched).
- Whitening happens CALLER-side, before encode: q/k arrive as packed phasor
  words and this op holds no fp statistics and performs no decode. Canonical
  recipe: ``mu, sd = phase_stats(H_mem)``; ``q_words = phasor_words(H, mu,
  sd)``; ``k_words = phasor_words(H_mem, mu, sd)``; ``v_words =
  encode_real_f32(H_mem_out - H_mem).reshape(M, d)``.

Memory note: stage C streams one value channel at a time through
``execute_route_stage_c_streamed``. Peak product storage is ``(T, M)`` words
instead of the old ``(T, M, d_v)`` tensor. The trade is memory for Python
dispatch: this issues ``d_v`` separate ``reduce_axis`` passes rather than one
over ``(T, M, d_v)``, so callers with large ``d_v`` pay a ``d_v``x Python-loop
for the lower memory ceiling.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .attention_ops import (
    _SENTINEL_I32,
    execute_softmax_lse,
    quantize_scores_lq,
)
from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    SPF32_PHASE_MOD,
    encode_complex,
    pack_words,
    unpack_L,
    unpack_theta,
)
from .phase_route import to_phasor_pairs

__all__ = [
    "compile_route_execution_plan",
    "concordance_lq",
    "execute_route",
    "execute_route_stage_c_streamed",
    "phasor_words",
]

# Twiddle-style scorer table: cos over the u16 phase ring (fp64, 512 KB).
# Index = unsigned ring difference; cos(u·2π/2^16) == cos(signed·2π/2^16)
# up to fp64 ulp (the argument differs by exactly 2π for u > 2^15).
_COS_LUT: np.ndarray = np.cos(
    np.arange(int(SPF32_PHASE_MOD), dtype=np.float64)
    * (2.0 * np.pi / float(SPF32_PHASE_MOD))
)


def phasor_words(
    H: npt.ArrayLike, mu: npt.ArrayLike, sd: npt.ArrayLike
) -> np.ndarray:
    """Whitened halved-pair phasor words for the route addressing stage.

    Whiten exactly as :func:`~spf.tensor.phase_route.phase_concordance_scores`
    (mu/sd quantized f32 → f64), pair channels k ↔ k+d/2 into complex phasors,
    encode as packed SPF-32 words. Shape: real ``(T, d)`` → ``(T, d//2)``.

    The route op reads ONLY the Θ plane of these words (magnitude-stripped
    addressing); the L plane is present but unused. One shared whitening path
    so callers and tests cannot drift.
    """
    Ha = np.asarray(H, dtype=np.float64)
    if Ha.ndim != 2:
        raise ValueError(f"phasor_words: H must be 2-D (T, d), got {Ha.shape}")
    if Ha.shape[-1] % 2:
        raise ValueError(
            f"phasor_words: halved pairing needs even d, got d={Ha.shape[-1]}"
        )
    mu64 = np.asarray(mu, dtype=np.float32).astype(np.float64)
    sd64 = np.asarray(sd, dtype=np.float32).astype(np.float64)
    Z = to_phasor_pairs((Ha - mu64) / sd64)
    L, theta = encode_complex(Z)
    return pack_words(L, theta)


def concordance_lq(
    qT: npt.ArrayLike,
    kT: npt.ArrayLike,
    *,
    beta: float,
    l_frac_bits: int,
    causal: bool = False,
    row_block: int = 64,
) -> np.ndarray:
    """Stage A: integer-ΔΘ phase-concordance scores as int32 L-codes ``(T, M)``.

    ``s[i,j] = Σ_k cos(Θq[i,k] − Θk[j,k])`` with the ring difference taken in
    int32 then masked back to u16 (no reliance on unsigned wrap), cos from
    ``_COS_LUT``, fp64 channel sum, then ONE quantization to L-codes with beta
    folded (:func:`~spf.tensor.attention_ops.quantize_scores_lq`). No fp32
    score array exists at any point.

    ``causal=True`` writes the sentinel ``INT32_MIN`` at ``j > i`` — the mask
    lives here, pre-LSE, as the single source of truth.

    ``row_block`` bounds the ``(block, M, d/2)`` intermediates — peak scorer
    memory is ``row_block * M * (d/2)`` int32 + fp64 LUT gathers, independent
    of T.
    """
    qTa = np.asarray(qT)
    kTa = np.asarray(kT)
    if qTa.ndim != 2 or kTa.ndim != 2 or qTa.shape[1] != kTa.shape[1]:
        raise ValueError(
            f"concordance_lq: need qT (T,P), kT (M,P); got {qTa.shape}, {kTa.shape}"
        )
    if not (np.issubdtype(qTa.dtype, np.integer) and np.issubdtype(kTa.dtype, np.integer)):
        raise ValueError("concordance_lq: phase planes must be integer (u16 codes)")
    T = qTa.shape[0]
    M = kTa.shape[0]
    q32 = qTa.astype(np.int32)
    k32 = kTa.astype(np.int32)
    mask = int(SPF32_PHASE_MOD) - 1

    lq = np.empty((T, M), dtype=np.int32)
    for start in range(0, T, max(1, int(row_block))):
        stop = min(start + max(1, int(row_block)), T)
        dring = (q32[start:stop, None, :] - k32[None, :, :]) & mask
        s_blk = _COS_LUT[dring].sum(axis=-1)
        lq[start:stop] = quantize_scores_lq(s_blk, beta=beta, l_frac_bits=l_frac_bits)

    if causal:
        i_idx = np.arange(T)[:, None]
        j_idx = np.arange(M)[None, :]
        lq = np.where(j_idx > i_idx, np.int32(_SENTINEL_I32), lq)
    return lq


def execute_route(
    q_words: npt.ArrayLike,
    k_words: npt.ArrayLike,
    v_words: npt.ArrayLike,
    reducer: object,
    *,
    causal: bool = False,
    beta: float = 8.0,
    l_frac_bits: int = 16,
) -> np.ndarray:
    """Fused phase-native route: ``delta_words = softmax(β·S) ⊛ V`` in SPF words.

    Parameters
    ----------
    q_words:
        uint32 ``(T, P)`` phasor words (P = d/2 halved-pair channels). Only
        the Θ plane is read.
    k_words:
        uint32 ``(M, P)`` phasor words. Only the Θ plane is read.
    v_words:
        uint32 ``(M, d_v)`` packed SPF-32 value words (full complex words —
        both planes used in the mix).
    reducer:
        :class:`~spf.tensor.core.ReducerSpec` — the stage-C accumulation
        boundary. CARTESIAN/ACCURATE = reference; PINNED LOGPOLAR = native
        ``spf32.add`` fold.
    causal:
        Mask ``j > i`` at the SCORE matrix (T, M) via sentinel L-codes. The
        mix stage is never masked (masking (T, d_v) would zero output
        channels, not future tokens).
    beta:
        Softmax temperature, folded into the stage-A quantization. Stage B
        always runs ``beta=1.0`` (enforced by ``execute_softmax_lse``).
    l_frac_bits:
        L-code fractional bits, threaded to BOTH the quantizer and the LSE —
        the one parameter that keeps the producer/consumer scale contract.

    Returns
    -------
    uint32 ``(T, d_v)`` delta words. Decode/realify is the caller's epilogue.

    Numerics: P words are ``(L ≤ 0, Θ=0)``, so each stage-C product has phase
    exactly ``Θ_V`` and L the saturating ``L_P + L_V`` — the weights act as
    pure positive log-magnitude shifts on the value phasors, and sub-range
    weights became ``WORD_ZERO`` in the LSE (annihilating in the contract).
    No ``decode_words`` call exists anywhere in this op (rule #1, pinned by
    the monkeypatch test in test_T23_route.py).
    """
    q = np.asarray(q_words, dtype=np.uint32)
    k = np.asarray(k_words, dtype=np.uint32)
    v = np.asarray(v_words, dtype=np.uint32)
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError(
            f"execute_route: q/k/v must be 2-D, got {q.shape}, {k.shape}, {v.shape}"
        )
    if q.shape[1] != k.shape[1]:
        raise ValueError(
            f"execute_route: q P={q.shape[1]} != k P={k.shape[1]}"
        )
    if v.shape[0] != k.shape[0]:
        raise ValueError(
            f"execute_route: v rows M={v.shape[0]} != k rows M={k.shape[0]}"
        )

    # Stage A — address: integer Θ planes only, one quantization, no decode.
    lq = concordance_lq(
        unpack_theta(q),
        unpack_theta(k),
        beta=beta,
        l_frac_bits=l_frac_bits,
        causal=causal,
    )
    # Stage B — gate: log-domain LSE on the codes; sentinels carry the mask
    # (causal=False here on purpose — a second mask would be redundant).
    p_words = execute_softmax_lse(
        lq,
        reducer,
        causal=False,
        l_frac_bits=l_frac_bits,
        beta=1.0,
        out="words",
        pre_quantized=True,
    )
    # Stage C — mix: stream value channels through the reducer boundary. This
    # avoids the old (T, M, d_v) probability-by-value product tensor.
    return execute_route_stage_c_streamed(p_words, v, reducer)


def execute_route_stage_c_streamed(
    p_words: npt.ArrayLike,
    v_words: npt.ArrayLike,
    reducer: object,
) -> np.ndarray:
    """Stage C route contraction without a materialized ``(T, M, d_v)`` tensor.

    ``p_words`` are positive-real SPF probability words from Stage B, so term
    multiplication is the route-specific product ``L_P + L_V`` with ``Theta_V``
    passing through unchanged. The function forms only one ``(T, M)`` term
    slab per value channel, reduces rows with the existing reducer backend, and
    writes the ``(T, d_v)`` output column. It is bit-identical to
    ``execute_contract(p_words, v_words, transpose_b=False)`` for route Stage C
    but has peak product storage ``T*M`` words instead of ``T*M*d_v`` words.
    """
    from . import reducers  # local import to avoid circular imports

    p = np.asarray(p_words, dtype=np.uint32)
    v = np.asarray(v_words, dtype=np.uint32)
    if p.ndim != 2 or v.ndim != 2:
        raise ValueError(
            f"route Stage C: p/v must be 2-D, got {p.shape}, {v.shape}"
        )
    if p.shape[1] != v.shape[0]:
        raise ValueError(
            f"route Stage C: p memory M={p.shape[1]} != v rows M={v.shape[0]}"
        )

    p_l = unpack_L(p).astype(np.int64)
    p_zero = p_l == np.int16(int(SPF32_L_ZERO))
    v_l = unpack_L(v).astype(np.int64)
    v_t = unpack_theta(v).astype(np.uint16, copy=False)
    t_rows = p.shape[0]
    d_v = v.shape[1]
    out = np.empty((t_rows, d_v), dtype=np.uint32)

    for channel in range(d_v):
        v_l_col = v_l[:, channel]
        v_t_col = v_t[:, channel]
        v_zero = v_l_col == np.int16(int(SPF32_L_ZERO))
        zero_mask = p_zero | v_zero[np.newaxis, :]
        term_l = np.clip(
            p_l + v_l_col[np.newaxis, :],
            SPF32_L_MIN_FINITE,
            SPF32_L_MAX_FINITE,
        ).astype(np.int16)
        term_t = np.broadcast_to(v_t_col[np.newaxis, :], term_l.shape).astype(
            np.uint16,
            copy=True,
        )
        term_l = np.where(zero_mask, np.int16(int(SPF32_L_ZERO)), term_l)
        term_t = np.where(zero_mask, np.uint16(0), term_t)
        out[:, channel] = reducers.reduce_axis(pack_words(term_l, term_t), reducer, axis=-1)
    return out.astype(np.uint32)


def compile_route_execution_plan(
    *,
    beta: float = 8.0,
    l_frac_bits: int = 16,
    causal: bool = False,
    mix_domain: object | None = None,
) -> object:
    """The fused route as ExecutionPlan DATA — one step, words in, words out.

        route(q_words, k_words, v_words) -> delta_words

    The whiten/phasor-pair prologue and the realify/residual epilogue stay
    host-side (same split as :func:`compile_phase_route_execution_plan`,
    which remains untouched as the 3-step reference).

    ``mix_domain=None`` → the reference reducer (AUTO, ACCURATE, STATIC).
    ``mix_domain=AccumulationDomain.LOGPOLAR`` → (AUTO, ACCURATE, PINNED,
    domain) — the native FB7-fold mix.
    """
    from .core import (
        AccumulationDomain,
        PrecisionMode,
        ReducerPolicy,
        ReducerSpec,
        RoutingPolicy,
    )
    from .v2 import ExecutionPlan, ExecutionStep

    if mix_domain is None:
        reducer = ReducerSpec(
            policy=ReducerPolicy.AUTO,
            precision_mode=PrecisionMode.ACCURATE,
            routing=RoutingPolicy.STATIC,
        )
        domain_name = "AUTO"
    else:
        domain = AccumulationDomain(mix_domain)
        reducer = ReducerSpec(
            policy=ReducerPolicy.AUTO,
            precision_mode=PrecisionMode.ACCURATE,
            routing=RoutingPolicy.PINNED,
            domain=domain,
        )
        domain_name = domain.name
    return ExecutionPlan(
        steps=(
            ExecutionStep(
                "route",
                ("q_words", "k_words", "v_words"),
                ("delta_words",),
                reducer=reducer,
                params={
                    "causal": bool(causal),
                    "beta": float(beta),
                    "l_frac_bits": int(l_frac_bits),
                },
            ),
        ),
        factorization_metadata={
            "operator": "Route",
            "lowering": "spf_fused_route",
            "beta": float(beta),
            "l_frac_bits": int(l_frac_bits),
            "causal": bool(causal),
            "full_memory": True,  # no rank / prototypes (cl-0180/cl-0181)
            "plan_steps": 1,
            "mix_domain": domain_name,
            "reducer_precision": "ACCURATE",
            "reducer_routing": "PINNED" if mix_domain is not None else "STATIC",
        },
    )
