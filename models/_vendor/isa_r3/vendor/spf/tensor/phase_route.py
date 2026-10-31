"""Full-memory phase-route surface — PHASE_ROUTE_SPEC items 1 + 5.

Per token, retrieve an update from a phase-keyed associative memory (softmax
retrieval over magnitude-stripped direction vectors) and add it to the
residual stream:

    W(x)  = (x - mu) / sd            per-channel whiten, stats from MEMORY
    Q[i]  = W(H[i])     / ||.||2     query: unit direction (phase axis)
    K[j]  = W(Hmem[j])  / ||.||2     key:   unit direction
    V[j]  = Hmem_out[j] - Hmem[j]    value: the memory token's residual update
    A     = softmax_j(beta * <Q,K>)
    out   = H + A @ V

This is the C kernel ``spft_phase_route`` (native/src/scalar/spf32/tensor/
spft_phase_route.c) mirrored step by step; the c_bindings parity test holds
the two together. The middle three steps are the registered ExecutionPlan
ops (``contract`` -> ``softmax_lse`` -> ``contract``), so the route is also
expressible as plan DATA via :func:`compile_phase_route_execution_plan`.

Design constraints carried from the measurements (panr-core-route arc):
- FULL memory only — rank truncation and prototype clustering both re-lose
  the world axis (cl-0180/cl-0181). There is deliberately no rank parameter.
- The softmax nonlinearity is load-bearing (linear/ridge fails the world
  gate, cl-0180) — kept, in the log domain (no expf).
- Wide-dynamic-range accumulations route to CARTESIAN: the pivoted fp64
  NumPy-order PAIRWISE path. NOT Kahan — bit-parity with reducers.py is the
  gate; "pivoted-Kahan" in PHASE_ROUTE_SPEC.md is inaccurate wording.
- SPF-32 operand encode adds ~5.5e-4 RMS/term quantization: end-to-end
  agreement with an fp64 reference is ~1e-3-class, NOT the 4.8e-6 of the
  accumulate alone (that figure is the accumulator comparison on
  unquantized operands). Measured in tests/tensor/test_T22_phase_route.py.

NOTE — `phase_route` scores a real dot of magnitude-CARRYING vectors: that is
the incoherent cosine engine (cl-0241), the disproved op. The phase-native fix
for the *addressing* half is :func:`phase_address` (below): the score is a real
phase-concordance sum `Σ_k cos(ΔΘ_k)` over integer ring differences, with NO
complex cross-Gram and NO complex value contraction — addressing needs neither
(`Σcos(ΔΘ) = Re(Σ e^{iΔΘ})`, measured-by-algebra). Prefer it over reaching for
a Cartesian complex op; complex is not where SPF earns its keep here.

The WORDS-NATIVE form of phase_address is the fused ``route`` ExecutionPlan op
(:mod:`spf.tensor.route_op` + C ``spft_attention_route``): same concordance
addressing, but L-code softmax ingress and zero decode inside — use it when
operands are already SPF words. Deployment caveat (measured, world gate): the
concordance addressing is SPIKE-LAYER-selective — at non-spike layers the
whiten+L2 direction dot routes better; select layers by phase importance
(cl-0243), don't swap blindly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .attention_ops import execute_contract, execute_softmax_lse
from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    SPF32_PHASE_MOD,
    circular_diff_u16,
    encode_complex,
    pack_words,
    unpack_L,
    unpack_theta,
)

__all__ = [
    "compile_phase_route_execution_plan",
    "decode_real_f32",
    "encode_real_f32",
    "phase_address",
    "phase_concordance_scores",
    "phase_features",
    "phase_route",
    "phase_stats",
    "to_phasor_pairs",
]

_THETA_TO_RAD = 2.0 * np.pi / 65536.0


def phase_stats(H_mem: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel whitening stats over the memory rows.

    Population std (ddof=0) + 1e-8, fp64 accumulate, f32 out — mirrors
    ``spft_phase_stats`` (C sums sequentially, numpy pairwise: agreement is
    fp64-roundoff-level, absorbed by the f32 cast in practice).
    """
    Hm = np.asarray(H_mem, dtype=np.float64)
    if Hm.ndim != 2:
        raise ValueError("H_mem must be 2-D (M, d)")
    mu = Hm.mean(axis=0)
    sd = Hm.std(axis=0) + 1e-8
    return mu.astype(np.float32), sd.astype(np.float32)


def phase_features(
    H: npt.ArrayLike, mu: npt.ArrayLike, sd: npt.ArrayLike
) -> np.ndarray:
    """Whiten + per-token L2 normalize (item 1) — mirrors ``spft_phase_features``.

    The C kernel casts the whitened value to f32 *before* the normalize
    multiply but accumulates the norm in fp64 from the uncast value;
    replicated exactly here.
    """
    H64 = np.asarray(H, dtype=np.float64)
    if H64.ndim != 2:
        raise ValueError("H must be 2-D (T, d)")
    mu64 = np.asarray(mu, dtype=np.float32).astype(np.float64)
    sd64 = np.asarray(sd, dtype=np.float32).astype(np.float64)
    # C reads f32 inputs; quantize H to f32 first for exact mirroring.
    w64 = (np.asarray(H, dtype=np.float32).astype(np.float64) - mu64) / sd64
    q32 = w64.astype(np.float32)
    inv = 1.0 / (np.sqrt((w64 * w64).sum(axis=1)) + 1e-12)
    return (q32.astype(np.float64) * inv[:, None]).astype(np.float32)


def encode_real_f32(x: npt.ArrayLike) -> np.ndarray:
    """Vectorized mirror of C ``spft_encode_real_f32``.

    rint (half-even) of log2|v|*256, clipped to the finite L range; theta is
    exactly 0 (positive) or 32768 (negative). Zero/NaN -> WORD_ZERO.
    """
    v = np.asarray(x, dtype=np.float32).astype(np.float64).reshape(-1)
    out_L = np.full(v.shape, SPF32_L_ZERO, dtype=np.int64)
    out_T = np.zeros(v.shape, dtype=np.uint16)
    finite = (v != 0.0) & ~np.isnan(v)
    lq = np.rint(np.log2(np.abs(v[finite])) * 256.0).astype(np.int64)
    out_L[finite] = np.clip(lq, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE)
    out_T[finite] = np.where(v[finite] < 0.0, 32768, 0).astype(np.uint16)
    return pack_words(out_L.astype(np.int16), out_T)


def decode_real_f32(words: npt.ArrayLike) -> np.ndarray:
    """Vectorized mirror of C ``spft_decode_real_f32`` (real projection)."""
    w = np.asarray(words, dtype=np.uint32).reshape(-1)
    L = unpack_L(w).astype(np.float64)
    theta = unpack_theta(w).astype(np.float64)
    mag = np.exp2(L / 256.0)
    val = (mag * np.cos(theta * _THETA_TO_RAD)).astype(np.float32)
    return np.where(unpack_L(w) == SPF32_L_ZERO, np.float32(0.0), val)


def phase_route(
    H: npt.ArrayLike,
    H_mem: npt.ArrayLike,
    H_mem_out: npt.ArrayLike,
    reducer: object,
    *,
    beta: float = 8.0,
    l_frac_bits: int = 16,
) -> np.ndarray:
    """Item 5: the full surface. Returns H + softmax(beta*Q.K^T) @ V.

    Step-by-step mirror of ``spft_phase_route``; the two contractions and
    the softmax run through the registered op reference implementations
    (which are themselves bit-parity-tested against the C kernels).
    """
    H32 = np.ascontiguousarray(H, dtype=np.float32)
    Hm = np.ascontiguousarray(H_mem, dtype=np.float32)
    Hmo = np.ascontiguousarray(H_mem_out, dtype=np.float32)
    if H32.ndim != 2 or Hm.shape != Hmo.shape or Hm.shape[1] != H32.shape[1]:
        raise ValueError("H (T,d), H_mem (M,d), H_mem_out (M,d) required")
    T, d = H32.shape
    M = Hm.shape[0]

    mu, sd = phase_stats(Hm)
    Q = phase_features(H32, mu, sd)
    K = phase_features(Hm, mu, sd)
    V = Hmo - Hm

    Qw = encode_real_f32(Q).reshape(T, d)
    Kw = encode_real_f32(K).reshape(M, d)
    Vw = encode_real_f32(V).reshape(M, d)

    scores_w = execute_contract(Qw, Kw, reducer, transpose_b=True, causal=False)
    scores = decode_real_f32(scores_w).reshape(T, M)
    P_w = execute_softmax_lse(
        scores, reducer, causal=False, l_frac_bits=l_frac_bits, beta=beta,
        out="words",
    )
    delta_w = execute_contract(P_w, Vw, reducer, transpose_b=False, causal=False)
    delta = decode_real_f32(delta_w).reshape(T, d)
    return H32 + delta


# ---------------------------------------------------------------------------
# Phase-concordance ADDRESSING — done native, no complex (the routing half)
# ---------------------------------------------------------------------------
#
# `phase_route` above scores a real dot of magnitude-CARRYING vectors (the
# cosine/incoherent engine, cl-0241). The coherent fix could be written with a
# complex cross-Gram + complex value interference, but that is NOT where SPF
# earns its keep — the complex resultant needs a CARTESIAN fold. The addressing
# (which memory each query retrieves) is the part that is genuinely phase, and
# it does NOT need complex at all:
#
#   measured-by-algebra:  sum_k cos(ΔΘ_k) = Re( sum_k e^{iΔΘ_k} )
#
# Re distributes over the sum, so the concordance score is a REAL reduction of
# per-channel cosines — the complex resultant is never formed. ΔΘ_k is an
# integer ring subtraction (`circular_diff_u16`); cos(ΔΘ_k) is the real
# projection of a UNIT phasor (magnitude stripped). That is the whole SPF win
# for routing: phase alignment is integer, the coherence sum is plain real.
#
# The value mix `A @ V` stays a plain real reduction — it is an accumulation,
# not an SPF primitive; it is deliberately NOT dressed as a phase win.


def to_phasor_pairs(X: npt.ArrayLike) -> np.ndarray:
    """Pair real features into complex phasors via the model's halved-RoPE
    pairing (channel k <-> k+d/2): real ``[..., d]`` -> complex ``[..., d/2]``.
    Only the per-pair ANGLE is used downstream (magnitude is stripped)."""
    Xa = np.asarray(X)
    half = Xa.shape[-1] // 2
    return Xa[..., :half] + 1j * Xa[..., half:]


def phase_concordance_scores(
    H_q: npt.ArrayLike,
    H_k: npt.ArrayLike,
    mu: npt.ArrayLike,
    sd: npt.ArrayLike,
) -> np.ndarray:
    """Native phase-concordance addressing scores ``S[T, M]``.

    ``S[i, j] = sum_k cos(ΔΘ_ik - ΔΘ_jk)`` where per-channel phase is the
    halved-pair angle of the whitened features (magnitude stripped) and the
    difference is the integer ``circular_diff_u16``. This is exactly
    ``Re( sum_k e^{iΔΘ} )`` — the real coherence sum, no complex resultant
    formed, no Cartesian fold. Score is signed; ``S/d ∈ [-1, 1]``.
    """
    mu64 = np.asarray(mu, dtype=np.float32).astype(np.float64)
    sd64 = np.asarray(sd, dtype=np.float32).astype(np.float64)
    Qz = to_phasor_pairs((np.asarray(H_q, np.float64) - mu64) / sd64)
    Kz = to_phasor_pairs((np.asarray(H_k, np.float64) - mu64) / sd64)
    # magnitude stripped: keep only the SPF phase code of each pair
    _, thQ = encode_complex(Qz)              # u16 phase planes [T, d/2]
    _, thK = encode_complex(Kz)              # [M, d/2]
    # integer per-channel ring difference, broadcast to [T, M, d/2]
    dtheta = circular_diff_u16(thQ[:, None, :], thK[None, :, :])
    # cos(ΔΘ) = Re of the unit phasor at ΔΘ; sum over channels = Re(Σ e^{iΔΘ})
    cos = np.cos(dtheta.astype(np.float64) * (2.0 * np.pi / SPF32_PHASE_MOD))
    return cos.sum(-1)


def phase_address(
    H: npt.ArrayLike,
    H_mem: npt.ArrayLike,
    H_mem_out: npt.ArrayLike,
    *,
    beta: float = 8.0,
    l_frac_bits: int = 16,
) -> np.ndarray:
    """Phase-concordance addressing router: ``H + softmax(beta*S) @ V``.

    ``S`` is the native real coherence sum (:func:`phase_concordance_scores`,
    integer ΔΘ, no complex). The softmax nonlinearity runs through the SPF
    log-domain ``softmax_lse`` (it is the load-bearing op, cl-0180); the value
    mix ``A @ (H_mem_out - H_mem)`` is a PLAIN real reduction — not an SPF win.

    Contrast: :func:`phase_route` scores magnitude-carrying cosine (incoherent,
    cl-0241); this op keeps the phase win in the *addressing* while staying
    purely real. BAG-ONLY (no causal/positional variant).

    UNTESTED (asserted): whether concordance addressing closes more of the
    routing gap than the cosine op. cl-0254 says the cross-corpus part may be a
    hard floor, so the likely win is in-context addressing — measure before
    claiming.
    """
    from .core import PrecisionMode, ReducerPolicy, ReducerSpec, RoutingPolicy

    H32 = np.ascontiguousarray(H, dtype=np.float32)
    Hm = np.ascontiguousarray(H_mem, dtype=np.float32)
    Hmo = np.ascontiguousarray(H_mem_out, dtype=np.float32)
    if H32.ndim != 2 or Hm.shape != Hmo.shape or Hm.shape[1] != H32.shape[1]:
        raise ValueError("H (T,d), H_mem (M,d), H_mem_out (M,d) required")
    d = H32.shape[1]
    if d % 2:
        raise ValueError(f"addressing needs even d for halved pairing, got d={d}")

    spec = ReducerSpec(
        policy=ReducerPolicy.AUTO,
        precision_mode=PrecisionMode.ACCURATE,
        routing=RoutingPolicy.STATIC,
    )
    mu, sd = phase_stats(Hm)
    scores = phase_concordance_scores(H32, Hm, mu, sd)         # [T, M] real, native ΔΘ
    A = execute_softmax_lse(
        scores, spec, causal=False, l_frac_bits=l_frac_bits, beta=beta, out="f32",
    )
    V = Hmo - Hm
    return H32 + (A.astype(np.float64) @ V).astype(np.float32)  # plain real value mix


def compile_phase_route_execution_plan(
    *, beta: float = 8.0, l_frac_bits: int = 16
) -> object:
    """The route's dynamic middle as ExecutionPlan DATA.

    Three steps over named env words (the whiten/normalize prologue and the
    residual-add epilogue stay host-side, like the C surface):

        contract(q_words, k_words, transpose_b)  -> scores_words
        softmax_lse(scores_f32, beta, out=words) -> p_words
        contract(p_words, v_words)               -> delta_words

    The softmax step consumes fp32-decoded scores (host decode between
    steps 0 and 1) — same contract as the dynamic-hub plan in
    panr-core-fast/pan_lab/spf_attention.py.
    """
    from .core import PrecisionMode, ReducerPolicy, ReducerSpec, RoutingPolicy
    from .v2 import ExecutionPlan, ExecutionStep

    reducer = ReducerSpec(
        policy=ReducerPolicy.AUTO,
        precision_mode=PrecisionMode.ACCURATE,
        routing=RoutingPolicy.STATIC,
    )
    return ExecutionPlan(
        steps=(
            ExecutionStep(
                "contract",
                ("q_words", "k_words"),
                ("scores_words",),
                reducer=reducer,
                params={"transpose_b": True, "causal": False},
            ),
            ExecutionStep(
                "softmax_lse",
                ("scores_f32",),
                ("p_words",),
                reducer=reducer,
                params={
                    "causal": False,
                    "l_frac_bits": int(l_frac_bits),
                    "beta": float(beta),
                    "out": "words",
                },
            ),
            ExecutionStep(
                "contract",
                ("p_words", "v_words"),
                ("delta_words",),
                reducer=reducer,
                params={"transpose_b": False, "causal": False},
            ),
        ),
        factorization_metadata={
            "operator": "PhaseRoute",
            "lowering": "spf_contract_softmax_contract",
            "beta": float(beta),
            "l_frac_bits": int(l_frac_bits),
            "full_memory": True,  # no rank / prototypes (cl-0180/cl-0181)
            "plan_steps": 3,
            "reducer_precision": "ACCURATE",
            "reducer_routing": "STATIC",
        },
    )
