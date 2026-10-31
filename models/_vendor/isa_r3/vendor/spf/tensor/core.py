"""SPF OperatorBlock / TensorBlock reference implementation.

This module implements the corrected Level-4 contract:

* tensors reconstruct to explicit SPF L/T planes with inactive entries set to
  the SPF zero sentinel, never to ``+1``;
* elementwise spectral products stay in SPF integer arithmetic;
* every contraction or rank sum has an explicit reducer policy;
* synthesis/export is a terminal boundary operator that chooses spectral DAC,
  NCO, template/tile, or IFFT-compatible output without pretending dense OFDM
  samples are metadata-only.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from spf.scalar._constants import SPF16 as _SPF16
from spf.scalar._constants import SPF32 as _SPF32
from spf.scalar._constants import SPF64 as _SPF64
from spf.scalar._constants import SPF128 as _SPF128
from spf.scalar._constants import Format as _ScalarFmt

SPF32_L_FRAC_BITS = 8
SPF32_L_SCALE = 1 << SPF32_L_FRAC_BITS
SPF32_PHASE_BITS = 16
SPF32_PHASE_MOD = 1 << SPF32_PHASE_BITS
SPF32_PHASE_MASK = SPF32_PHASE_MOD - 1
SPF32_L_ZERO = np.int16(-32768)
SPF32_L_MIN_FINITE = -32767
SPF32_L_MAX_FINITE = 32766
SPF32_L_NAN = 32767
SPF32_T_NAN = 65535
SPF32_WORD_ZERO = np.uint32(0x80000000)
SPF32_WORD_NAN = np.uint32(0x7FFFFFFF)
INT32_MAX = (1 << 31) - 1
INT32_MIN = -(1 << 31)
SOB_MAGIC = 0x53504F42
SOB_HEADER_BYTES = 96


class SPFError(ValueError):
    """Raised when a Level-4 block violates its declared contract."""


class ScalarFormat(IntEnum):
    SPF16 = 1
    SPF32 = 2
    SPF64 = 3
    SPF128 = 4
    PHASE_ONLY8 = 8
    PHASE_ONLY16 = 9
    FNT32 = 32


# Map the tensor-layer ScalarFormat enum onto the scalar-layer Format descriptor
# (bit widths, scales, sentinels, dtypes). The plane/word helpers below accept a
# ``Format`` and default to SPF-32 so every existing SPF-32 call site is unchanged
# and bit-identical; SPF-16/64/128 ride the same code paths.
_SCALAR_FORMAT_TO_FMT: dict[ScalarFormat, _ScalarFmt] = {
    ScalarFormat.SPF16: _SPF16,
    ScalarFormat.SPF32: _SPF32,
    ScalarFormat.SPF64: _SPF64,
    ScalarFormat.SPF128: _SPF128,
}


def format_for(scalar_format_id: ScalarFormat | int) -> _ScalarFmt:
    """Resolve a :class:`ScalarFormat` (or its int value) to a scalar ``Format``."""
    try:
        return _SCALAR_FORMAT_TO_FMT[ScalarFormat(int(scalar_format_id))]
    except (KeyError, ValueError):
        raise SPFError(
            f"no scalar Format for ScalarFormat {scalar_format_id!r} (supported: SPF16/SPF32/SPF64/SPF128)"
        ) from None


def _is_wide_fmt(fmt: _ScalarFmt) -> bool:
    """True when the packed word exceeds 64 bits (SPF-128 → Python-int/object)."""
    return fmt.total_bits > 64


class Profile(IntEnum):
    STB_AFFINE = 1
    STB_SPARSE = 2
    SOV_DIAG = 10
    SOV_BC = 11
    SOV_LR = 12
    SOV_HDC = 13
    DEBUG_DENSE_RECONSTRUCT = 255


class ReducerPolicy(IntEnum):
    NONE = 0
    CARTESIAN_KAHAN = 1
    CARTESIAN_NEUMAIER = 2
    CARTESIAN_KULISCH = 3
    PAIRWISE_SUM_TO_PRODUCT = 4
    COTRANSFORMED = 5
    CLENSHAW_AP = 6
    PHASE_HISTOGRAM = 7
    ZECH_FNT = 8
    ANALOG_EXPORT = 9
    # Stage-5 domain-routed reduction. Opt-in token: legacy specs never carry
    # it, so existing behaviour is byte-identical. Dispatches to
    # ``spf.tensor.reducers.reduce_stream`` (TROPICAL / LOGPOLAR / CARTESIAN).
    AUTO = 10
    # Stage-5 phase-coherence split. Opt-in token (legacy specs never carry it →
    # byte-identical). Bins the finite terms by Θ into ``n_phase_bins`` contiguous
    # arcs, reduces each bin via the AUTO route()+guard path (so a coherent bin
    # stays SPF-native), then combines the ≤B bin-sum words once with CARTESIAN.
    PHASE_BINNED = 11
    # Explicit SPF v0.2 typed reducer. The legacy AUTO/PHASE_BINNED routes
    # remain unchanged; this token is the opt-in migration seam.
    V02 = 12


class AccumulationDomain(IntEnum):
    """SPF reduction accumulation domain (Stage-5 ReducerSpec)."""

    TROPICAL = 0  # max-plus: result is the dominant term. Exact when gap large.
    LOGPOLAR = 1  # FB7 ⊕ pairwise tree. Native; benign for short incoherent sums.
    CARTESIAN = 2  # decode → pivoted wide accumulate → encode once. N-independent error.


class RoutingPolicy(IntEnum):
    """How a ``ReducerSpec`` selects its accumulation domain (Stage-5)."""

    STATIC = 0  # route() at plan time from header/computed stats
    ADAPTIVE = 1  # route() at run time from probed stats (deferred — ship-order 3)
    PINNED = 2  # domain fixed by planner/user via ReducerSpec.domain


class PrecisionMode(IntEnum):
    FAST = 0
    BALANCED = 1
    ACCURATE = 2
    EXACT = 3


class ReductionOrder(IntEnum):
    LINEAR = 0
    PAIRWISE = 1
    TREE = 2


@dataclass(frozen=True, slots=True)
class ReducerSpec:
    """First-class reducer boundary for TensorBlock contractions.

    The fields after ``n_phase_bins`` mirror the v0.2 reducer certificate
    contract.  They are inert for legacy policies and are copied verbatim by
    the ``ReducerPolicy.V02`` adapter.  The older routing, quantizer, and
    reduction-order fields remain part of the compatibility boundary; the
    v0.2 reference does not implement non-default values for those fields and
    the adapter reports them explicitly instead of silently discarding them.
    """

    policy: ReducerPolicy = ReducerPolicy.CARTESIAN_KAHAN
    accumulator_dtype: str = "complex128"
    reduction_order: ReductionOrder = ReductionOrder.LINEAR
    q_dL: int = 4
    q_dTheta: int = 4
    precision_mode: PrecisionMode = PrecisionMode.BALANCED
    quantizer: str = "spf32"
    # Stage-5 domain routing. Inert unless ``policy == AUTO`` — preserves the
    # behaviour of every spec built the old way.
    routing: RoutingPolicy = RoutingPolicy.STATIC
    domain: AccumulationDomain | None = None
    # Number of contiguous phase arcs for ``policy == PHASE_BINNED``. Inert for
    # every other policy.
    n_phase_bins: int = 16
    # v0.2 certificate controls.  These names intentionally match
    # ``spf.tensor.v02.ReductionSpec`` so the migration adapter is lossless.
    error_budget_abs: float = 0.0
    error_budget_rel: float = 0.0
    allow_pruning: bool = False
    deterministic: bool = True
    accumulator_fraction_bits: int | None = None
    required_value_kind: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", ReducerPolicy(self.policy))
        object.__setattr__(self, "reduction_order", ReductionOrder(self.reduction_order))
        object.__setattr__(self, "precision_mode", PrecisionMode(self.precision_mode))
        object.__setattr__(self, "routing", RoutingPolicy(self.routing))
        if self.domain is not None:
            object.__setattr__(self, "domain", AccumulationDomain(self.domain))
        if self.accumulator_dtype not in {"complex64", "complex128", "float64", "int64"}:
            raise SPFError("unsupported reducer accumulator dtype")
        if int(self.q_dL) <= 0 or int(self.q_dTheta) <= 0:
            raise SPFError("reducer residual precisions must be positive")
        if int(self.n_phase_bins) <= 0:
            raise SPFError("reducer n_phase_bins must be positive")
        if self.routing == RoutingPolicy.PINNED and self.domain is None:
            raise SPFError("PINNED routing requires an explicit ReducerSpec.domain")
        for name in ("error_budget_abs", "error_budget_rel"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise SPFError(f"reducer {name} must be finite and non-negative")
        if self.accumulator_fraction_bits is not None and int(self.accumulator_fraction_bits) < 0:
            raise SPFError("reducer accumulator_fraction_bits must be non-negative")

    def require_contraction(self) -> None:
        if self.policy == ReducerPolicy.NONE:
            raise SPFError("contraction requires a ReducerSpec with a non-NONE policy")

    def cost(self, stats: object) -> float:
        """Planner weight: expensive-op-equivalent cost of this reduction (Stage-5 §9)."""
        from . import reducers

        return reducers.spec_cost(self, stats)

    def error(self, stats: object) -> float:
        """Planner weight: relative-error estimate of this reduction (Stage-5 §9)."""
        from . import reducers

        return reducers.spec_error(self, stats)


def ensure_reducer_spec(reducer: ReducerPolicy | ReducerSpec) -> ReducerSpec:
    if isinstance(reducer, ReducerSpec):
        reducer.require_contraction()
        return reducer
    spec = ReducerSpec(policy=ReducerPolicy(reducer))
    spec.require_contraction()
    return spec


class ActiveEncoding(IntEnum):
    DENSE_ALL = 0
    COO_U32 = 1
    BITMAP = 2
    RLE = 3
    SEPARABLE_MASK = 4


class SynthMode(IntEnum):
    SPECTRAL_DAC = 0
    NORMAL_DAC_NCO = 1
    NORMAL_DAC_IFFT = 2
    HYBRID_TILE = 3
    TEMPLATE_CODEBOOK = 4
    AUTO = 255


class OutputFormat(IntEnum):
    IQ_INT16 = 1
    REAL_INT16 = 2


class QAMDemap(IntEnum):
    LOGPROJ = 1


class ScalePolicy(IntEnum):
    BLOCK_FLOAT_BACKOFF = 0
    FIXED_BACKOFF = 1
    UNITARY_IFFT = 2


class ClipPolicy(IntEnum):
    SATURATE = 0
    ERROR = 1


ElementwiseProfiles = {
    Profile.STB_AFFINE,
    Profile.STB_SPARSE,
    Profile.SOV_DIAG,
    Profile.SOV_HDC,
    Profile.DEBUG_DENSE_RECONSTRUCT,
}


def circular_diff_u16(a: int | npt.ArrayLike, b: int | npt.ArrayLike) -> int | np.ndarray:
    """Signed circular phase difference ``a - b`` in ``[-32768, 32767]``."""
    d = (np.asarray(a, dtype=np.uint32) - np.asarray(b, dtype=np.uint32)) & SPF32_PHASE_MASK
    out = np.where(d <= 0x7FFF, d.astype(np.int32), d.astype(np.int32) - SPF32_PHASE_MOD)
    if np.ndim(out) == 0:
        return int(out)
    return out


def _sat_l(v: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    """Saturate L codes to ``fmt``'s finite range; returned as ``l_signed_dtype``.

    SPF-128 clamps in Python int (its codes outrun int64 before the clip), so the
    inputs may arrive as an ``object`` array of Python ints; the clamped result
    always fits int64."""
    lo, hi = fmt.l_code_min_finite, fmt.l_code_max_finite
    if _is_wide_fmt(fmt):
        arr = np.asarray(v)
        flat = [lo if x < lo else hi if x > hi else int(x) for x in np.atleast_1d(arr).ravel().tolist()]
        return np.array(flat, dtype=fmt.l_signed_dtype).reshape(arr.shape)
    arr = np.asarray(v, dtype=np.int64)
    return np.clip(arr, lo, hi).astype(fmt.l_signed_dtype)


def _sat_l16(v: npt.ArrayLike) -> np.ndarray:
    """Back-compat SPF-32 saturation (``int16`` codes). Prefer :func:`_sat_l`."""
    return _sat_l(v, _SPF32)


def pack_words(L: npt.ArrayLike, theta: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    """Pack logical L/T planes into packed words of ``fmt`` (default SPF-32).

    SPF-128 words have no numpy dtype, so they are returned as an ``object`` array
    of arbitrary-precision Python ints."""
    L_arr = np.asarray(L, dtype=fmt.l_signed_dtype)
    T_arr = np.asarray(theta, dtype=fmt.theta_dtype)
    if _is_wide_fmt(fmt):
        l_bits_mask = (1 << fmt.l_bits) - 1
        l_u = L_arr.astype(object) & l_bits_mask
        words = (l_u << fmt.theta_bits) | T_arr.astype(object)
        # A bare 128-bit Python int trips np.where's C-long coercion; wrap the
        # zero sentinel as a 0-d object array so the result stays object.
        zero_fill = np.empty((), dtype=object)
        zero_fill[()] = int(fmt.word_zero)
        words = np.where(L_arr == fmt.l_code_zero, zero_fill, words)
        return words.astype(object)
    l_u = L_arr.astype(fmt.l_unsigned_dtype).astype(fmt.word_dtype) & ((1 << fmt.l_bits) - 1)
    words = (l_u << fmt.theta_bits) | (T_arr.astype(fmt.word_dtype) & fmt.word_mask_theta)
    words = np.where(L_arr == fmt.l_code_zero, fmt.word_dtype(fmt.word_zero), words)
    return words.astype(fmt.word_dtype)


def unpack_L(words: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    w = np.asarray(words, dtype=fmt.word_dtype)
    if fmt.l_bits != np.dtype(fmt.l_signed_dtype).itemsize * 8:
        raw = np.asarray(w >> fmt.theta_bits, dtype=np.int64) & ((1 << fmt.l_bits) - 1)
        sign = 1 << (fmt.l_bits - 1)
        return ((raw ^ sign) - sign).astype(fmt.l_signed_dtype)
    # np.asarray re-wraps a 0-d ``object`` word (which unwraps to a Python int).
    return np.asarray(w >> fmt.theta_bits, dtype=fmt.l_unsigned_dtype).view(fmt.l_signed_dtype)


def unpack_theta(words: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    w = np.asarray(words, dtype=fmt.word_dtype)
    return np.asarray(w & fmt.word_mask_theta, dtype=fmt.theta_dtype)


def encode_complex(z: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> tuple[np.ndarray, np.ndarray]:
    """Encode complex values into logical L/T planes for ``fmt`` (default SPF-32)."""
    zz = np.asarray(z, dtype=np.complex128)
    mag = np.abs(zz)
    L = np.full(zz.shape, fmt.l_code_zero, dtype=fmt.l_signed_dtype)
    T = np.zeros(zz.shape, dtype=fmt.theta_dtype)
    nan = np.isnan(zz.real) | np.isnan(zz.imag)
    L[nan] = fmt.l_code_inf
    T[nan] = fmt.theta_nan_payload
    inf = np.isinf(mag) & ~nan
    finite_nz = (mag > 0.0) & np.isfinite(mag) & ~nan
    if np.any(finite_nz | inf):
        special = finite_nz | inf
        if np.any(finite_nz):
            lq = np.rint(np.log2(mag[finite_nz]) * fmt.l_scale)
            L[finite_nz] = _sat_l(lq, fmt)
        if np.any(inf):
            L[inf] = fmt.l_code_inf
        theta = np.angle(zz[special]) % (2.0 * np.pi)
        if _is_wide_fmt(fmt):
            scaled = (theta * fmt.theta_scale / (2.0 * np.pi)).tolist()
            T[special] = np.array([round(x) % fmt.theta_scale for x in scaled], dtype=fmt.theta_dtype)
        else:
            tq = np.rint(theta * fmt.theta_scale / (2.0 * np.pi)).astype(np.int64)
            T[special] = (tq & fmt.word_mask_theta).astype(fmt.theta_dtype)
    return L, T


def encode_words(z: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    L, T = encode_complex(z, fmt)
    return pack_words(L, T, fmt)


def decode_complex(L: npt.ArrayLike, theta: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    """Decode logical L/T planes of ``fmt`` (default SPF-32) into complex128."""
    L_arr = np.asarray(L, dtype=fmt.l_signed_dtype)
    T_arr = np.asarray(theta, dtype=fmt.theta_dtype)
    out = np.zeros(np.broadcast_shapes(L_arr.shape, T_arr.shape), dtype=np.complex128)
    Lb = np.broadcast_to(L_arr, out.shape)
    Tb = np.broadcast_to(T_arr, out.shape)
    nan = (Lb == fmt.l_code_inf) & (Tb == fmt.theta_nan_payload)
    inf = (Lb == fmt.l_code_inf) & ~nan
    finite = (Lb != fmt.l_code_zero) & ~inf & ~nan
    if np.any(inf):
        phase = Tb[inf].astype(np.float64) * (2.0 * np.pi / fmt.theta_scale)
        with np.errstate(invalid="ignore", over="ignore"):
            out[inf] = np.inf * np.exp(1j * phase)
    if np.any(finite):
        mag = np.exp2(Lb[finite].astype(np.float64) / fmt.l_scale)
        phase = Tb[finite].astype(np.float64) * (2.0 * np.pi / fmt.theta_scale)
        out[finite] = mag * np.exp(1j * phase)
    out[nan] = np.nan + 1j * np.nan
    return out


def decode_words(words: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    return decode_complex(unpack_L(words, fmt), unpack_theta(words, fmt), fmt)


def spf_mul_planes(
    aL: npt.ArrayLike,
    aT: npt.ArrayLike,
    xL: npt.ArrayLike,
    xT: npt.ArrayLike,
    fmt: _ScalarFmt = _SPF32,
) -> tuple[np.ndarray, np.ndarray]:
    """Elementwise SPF multiply: integer log add + modular phase add."""
    aL_arr = np.asarray(aL, dtype=fmt.l_signed_dtype)
    xL_arr = np.asarray(xL, dtype=fmt.l_signed_dtype)
    aT_arr = np.asarray(aT, dtype=fmt.theta_dtype)
    xT_arr = np.asarray(xT, dtype=fmt.theta_dtype)
    zero = (aL_arr == fmt.l_code_zero) | (xL_arr == fmt.l_code_zero)
    acc = object if _is_wide_fmt(fmt) else np.int64
    L = _sat_l(aL_arr.astype(acc) + xL_arr.astype(acc), fmt)
    T = ((aT_arr.astype(acc) + xT_arr.astype(acc)) & fmt.word_mask_theta).astype(fmt.theta_dtype)
    L = np.where(zero, fmt.l_code_zero, L).astype(fmt.l_signed_dtype)
    T = np.where(zero, 0, T).astype(fmt.theta_dtype)
    return L, T


def spf_mul_words(a: npt.ArrayLike, x: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    L, T = spf_mul_planes(unpack_L(a, fmt), unpack_theta(a, fmt), unpack_L(x, fmt), unpack_theta(x, fmt), fmt)
    return pack_words(L, T, fmt)


def spf_conj_words(x: npt.ArrayLike, fmt: _ScalarFmt = _SPF32) -> np.ndarray:
    L = unpack_L(x, fmt)
    T = unpack_theta(x, fmt)
    acc = object if _is_wide_fmt(fmt) else np.int64
    Tc = ((-T.astype(acc)) & fmt.word_mask_theta).astype(fmt.theta_dtype)
    Tc = np.where(fmt.l_code_zero == L, 0, Tc).astype(fmt.theta_dtype)
    return pack_words(L, Tc, fmt)


def _kahan_sum_complex(values: np.ndarray) -> complex:
    """Pairwise (numpy) complex summation in FP64.

    Vectorized replacement for the former scalar Kahan loop; see
    ``reducers._kahan_sum`` for the accuracy / golden-regen rationale.
    """
    return complex(np.sum(np.asarray(values, dtype=np.complex128)))


def reduce_words_result(words: npt.ArrayLike, reducer: ReducerPolicy | ReducerSpec) -> object:
    """Domain-routed reduction returning the full :class:`reducers.ReductionResult`.

    Flag-aware entry point: unlike :func:`reduce_words` it preserves the
    ``PHASE_UNRELIABLE`` advisory flag, the chosen accumulation domain, and the
    PHASE_BINNED FP-boundary decode count. Requires ``reducer.policy`` to be
    ``AUTO``, ``PHASE_BINNED``, or the explicit v0.2 ``V02`` migration seam.
    """
    from . import reducers

    spec = ensure_reducer_spec(reducer)
    if spec.policy is ReducerPolicy.V02:
        return reduce_words_v02(words, spec=spec, return_result=True)
    if spec.policy not in (ReducerPolicy.AUTO, ReducerPolicy.PHASE_BINNED):
        raise SPFError("reduce_words_result requires a ReducerSpec with policy=AUTO, PHASE_BINNED, or V02")
    return reducers.reduce_stream(np.asarray(words, dtype=np.uint32), spec)


def reduce_words(words: npt.ArrayLike, reducer: ReducerPolicy | ReducerSpec) -> np.uint32:
    """Reduce SPF terms into one SPF word using an explicit reducer."""
    spec = ensure_reducer_spec(reducer)
    policy = spec.policy
    ww = np.asarray(words, dtype=np.uint32)
    if ww.size == 0:
        return SPF32_WORD_ZERO
    if policy is ReducerPolicy.V02:
        return np.uint32(reduce_words_v02(ww, spec=spec))
    if policy in (ReducerPolicy.AUTO, ReducerPolicy.PHASE_BINNED):
        # Stage-5 domain router / phase-coherence split. The advisory
        # PHASE_UNRELIABLE flag and FP-boundary decode count are dropped here (no
        # spare bit in a uint32 word); use reduce_words_result to keep them.
        return np.uint32(reduce_words_result(ww, spec).word)
    if policy == ReducerPolicy.CARTESIAN_KAHAN:
        return encode_words(np.asarray(_kahan_sum_complex(decode_words(ww)))).reshape(()).astype(np.uint32)
    if policy == ReducerPolicy.PAIRWISE_SUM_TO_PRODUCT:
        return _pairwise_sum_to_product_reduce(ww)
    if policy == ReducerPolicy.PHASE_HISTOGRAM:
        return _phase_histogram_reduce(ww, bins=64)
    raise NotImplementedError(f"reducer {policy.name} is declared but not implemented in Python v0")


def reduce_words_v02(
    words: npt.ArrayLike,
    *,
    spec: object | None = None,
    return_result: bool = False,
) -> object:
    """Run the explicit v0.2 ``SPF_COMPLEX`` reducer from the tensor boundary.

    This adapter makes migration visible without silently changing the
    established reducer policies.  ``return_result=False`` matches the old
    word-returning API; callers needing certificates should request the full
    v0.2 result.

    A core :class:`ReducerSpec` is translated losslessly for the v0.2
    certificate fields (error budgets, precision, pruning, determinism,
    accumulator fraction bits, and required value kind).  The legacy
    accumulator dtype, reduction order, residual quantizers, routing/domain,
    and phase-bin controls have no v0.2 equivalent yet.  Non-default values
    for those fields raise ``NotImplementedError`` so a request cannot be
    silently downgraded.
    """

    from .v02 import ReductionSpec as V02ReductionSpec
    from .v02 import reduce_v02

    if spec is None:
        policy = V02ReductionSpec()
    elif isinstance(spec, V02ReductionSpec):
        policy = spec
    else:
        reducer_spec = ensure_reducer_spec(spec)  # type: ignore[arg-type]
        unsupported: list[str] = []
        if reducer_spec.accumulator_dtype != "complex128":
            unsupported.append("accumulator_dtype")
        if reducer_spec.reduction_order is not ReductionOrder.LINEAR:
            unsupported.append("reduction_order")
        if int(reducer_spec.q_dL) != 4:
            unsupported.append("q_dL")
        if int(reducer_spec.q_dTheta) != 4:
            unsupported.append("q_dTheta")
        if reducer_spec.quantizer != "spf32":
            unsupported.append("quantizer")
        if reducer_spec.routing is not RoutingPolicy.STATIC:
            unsupported.append("routing")
        if reducer_spec.domain is not None:
            unsupported.append("domain")
        if int(reducer_spec.n_phase_bins) != 16:
            unsupported.append("n_phase_bins")
        if unsupported:
            fields = ", ".join(unsupported)
            raise NotImplementedError(
                "v0.2 reducer adapter does not implement non-default core "
                f"ReducerSpec fields: {fields}. Use v02.ReductionSpec directly "
                "or keep those fields at their compatibility defaults."
            )
        policy = V02ReductionSpec(
            error_budget_abs=reducer_spec.error_budget_abs,
            error_budget_rel=reducer_spec.error_budget_rel,
            precision_mode=reducer_spec.precision_mode,
            allow_pruning=reducer_spec.allow_pruning,
            deterministic=reducer_spec.deterministic,
            accumulator_fraction_bits=reducer_spec.accumulator_fraction_bits,
            required_value_kind=reducer_spec.required_value_kind,
        )
    result = reduce_v02(np.asarray(words, dtype=np.uint32).reshape(-1), kind="SPF_COMPLEX", spec=policy)
    return result if return_result else np.uint32(result.word)


def _pairwise_sum_to_product_reduce(words: np.ndarray) -> np.uint32:
    """Approximate unit-phasor bundle via hierarchical phase averaging."""
    phases = unpack_theta(words).astype(np.uint32)
    live = phases.copy()
    if live.size == 0:
        return SPF32_WORD_ZERO
    while live.size > 1:
        if live.size % 2:
            tail = live[-1:]
            live = live[:-1]
        else:
            tail = np.empty(0, dtype=np.uint32)
        a = live[0::2]
        b = live[1::2]
        delta = circular_diff_u16(b, a)
        avg = (a.astype(np.int64) + np.asarray(delta, dtype=np.int64) // 2) & SPF32_PHASE_MASK
        live = np.concatenate([avg.astype(np.uint32), tail])
    return pack_words(np.asarray(0, dtype=np.int16), live.astype(np.uint16))[0]


def _phase_histogram_reduce(words: np.ndarray, *, bins: int) -> np.uint32:
    phases = unpack_theta(words).astype(np.uint32)
    if phases.size == 0:
        return SPF32_WORD_ZERO
    hist = np.bincount((phases * bins // SPF32_PHASE_MOD).astype(np.int64), minlength=bins)
    mode = int(np.argmax(hist))
    theta = np.uint16(((mode + 0.5) * SPF32_PHASE_MOD / bins) % SPF32_PHASE_MOD)
    return pack_words(
        np.asarray([0], dtype=np.int16),
        np.asarray([theta], dtype=np.uint16),
    )[0]


@dataclass(slots=True)
class OperatorHeader:
    """SOB-96 common header for Level-4 blocks."""

    profile_id: Profile
    reducer_id: ReducerPolicy = ReducerPolicy.NONE
    dims: tuple[int, ...] = (1,)
    version_major: int = 0
    version_minor: int = 1
    scalar_format_id: ScalarFormat = ScalarFormat.SPF32
    flags: int = 0
    phase_bits: int = SPF32_PHASE_BITS
    log_frac_bits: int = SPF32_L_FRAC_BITS
    payload_bytes: int = 0
    payload_offset: int = SOB_HEADER_BYTES
    strides: tuple[int, ...] | None = None
    origin: tuple[int, ...] | None = None
    global_L_bias: int = 0
    global_T_bias: int = 0
    ramp_alpha: tuple[int, ...] | None = None
    magic: int = SOB_MAGIC
    header_bytes: int = SOB_HEADER_BYTES

    def __post_init__(self) -> None:
        self.dims = tuple(int(d) for d in self.dims)
        if len(self.dims) > 4:
            raise SPFError("SOB-96 supports at most four dimensions")
        if self.strides is None:
            self.strides = _row_major_strides(self.dims)
        if self.origin is None:
            self.origin = tuple(0 for _ in self.dims)
        if self.ramp_alpha is None:
            self.ramp_alpha = tuple(0 for _ in self.dims)
        self.strides = _pad_tuple(self.strides, 4, 0)
        self.origin = _pad_tuple(self.origin, 4, 0)
        self.ramp_alpha = _pad_tuple(self.ramp_alpha, 4, 0)

    @property
    def ndims(self) -> int:
        return len(self.dims)

    @property
    def padded_dims(self) -> tuple[int, int, int, int]:
        return _pad_tuple(self.dims, 4, 0)

    @property
    def total_elements(self) -> int:
        if not self.dims:
            return 1
        return int(np.prod(self.dims, dtype=np.int64))

    def validate(self, *, containing_bytes: int | None = None) -> None:
        if self.magic != SOB_MAGIC:
            raise SPFError("bad SOB magic")
        if self.ndims > 4:
            raise SPFError("ndims exceeds v0.1 maximum")
        if any(d <= 0 for d in self.dims):
            raise SPFError("all active dims must be positive")
        if self.header_bytes < SOB_HEADER_BYTES or self.header_bytes % 16 != 0:
            raise SPFError("header size must be >=96 and 16-byte aligned")
        if self.phase_bits not in (8, 16, 32, 64):
            raise SPFError("phase_bits must be 8, 16, 32, or 64")
        if self.reducer_id == ReducerPolicy.NONE and self.profile_id not in ElementwiseProfiles:
            raise SPFError("contraction-capable profile requires a reducer")
        if containing_bytes is not None:
            if self.payload_offset < 0 or self.payload_bytes < 0:
                raise SPFError("negative payload metadata")
            if self.payload_offset + self.payload_bytes > containing_bytes:
                raise SPFError("payload extends beyond containing block")

    def shift_view(self, dim: int, delta: int) -> OperatorHeader:
        origin = list(self.origin or (0, 0, 0, 0))
        if dim < 0 or dim >= self.ndims:
            raise SPFError("shift dimension out of range")
        origin[dim] = (origin[dim] + int(delta)) % self.dims[dim]
        return replace(self, origin=tuple(origin))

    def apply_global_gain(self, delta_L: int) -> OperatorHeader:
        fmt = format_for(self.scalar_format_id)
        new_bias = int(min(max(self.global_L_bias + delta_L, fmt.l_code_min_finite), fmt.l_code_max_finite))
        return replace(self, global_L_bias=new_bias)

    def apply_global_phase(self, delta_T: int) -> OperatorHeader:
        fmt = format_for(self.scalar_format_id)
        return replace(self, global_T_bias=(self.global_T_bias + int(delta_T)) & fmt.word_mask_theta)

    def apply_phase_ramp(self, dim: int, delta_alpha: int) -> OperatorHeader:
        ramps = list(self.ramp_alpha or (0, 0, 0, 0))
        if dim < 0 or dim >= self.ndims:
            raise SPFError("ramp dimension out of range")
        ramps[dim] = int(ramps[dim]) + int(delta_alpha)
        return replace(self, ramp_alpha=tuple(ramps))

    def to_bytes(self) -> bytes:
        self.validate()
        return struct.pack(
            "<IBBBBBBBBHHII4I4i4IhH I4i".replace(" ", ""),
            self.magic,
            self.version_major,
            self.version_minor,
            int(self.scalar_format_id),
            int(self.profile_id),
            int(self.reducer_id),
            self.ndims,
            self.flags,
            self.phase_bits,
            self.log_frac_bits,
            self.header_bytes,
            self.payload_bytes,
            self.payload_offset,
            *self.padded_dims,
            *(self.strides or (0, 0, 0, 0)),
            *(self.origin or (0, 0, 0, 0)),
            self.global_L_bias,
            0,
            self.global_T_bias,
            *(self.ramp_alpha or (0, 0, 0, 0)),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> OperatorHeader:
        if len(data) < SOB_HEADER_BYTES:
            raise SPFError("not enough bytes for SOB-96")
        vals = struct.unpack("<IBBBBBBBBHHII4I4i4IhHI4i", data[:SOB_HEADER_BYTES])
        dims = tuple(int(v) for v in vals[13:17] if int(v) != 0)
        obj = cls(
            magic=vals[0],
            version_major=vals[1],
            version_minor=vals[2],
            scalar_format_id=ScalarFormat(vals[3]),
            profile_id=Profile(vals[4]),
            reducer_id=ReducerPolicy(vals[5]),
            dims=dims,
            flags=vals[7],
            phase_bits=vals[8],
            log_frac_bits=vals[9],
            header_bytes=vals[10],
            payload_bytes=vals[11],
            payload_offset=vals[12],
            strides=tuple(int(v) for v in vals[17:21]),
            origin=tuple(int(v) for v in vals[21:25]),
            global_L_bias=vals[25],
            global_T_bias=vals[27],
            ramp_alpha=tuple(int(v) for v in vals[28:32]),
        )
        obj.validate()
        return obj


@dataclass(frozen=True, slots=True)
class BandRange:
    dim: int
    start: int
    length: int
    L_delta: int = 0
    T_delta: int = 0
    flags: int | None = None

    def __post_init__(self) -> None:
        if self.dim < 0 or self.dim > 3:
            raise SPFError("band dim must be 0..3")
        if self.start < 0 or self.length <= 0:
            raise SPFError("band start/length must define a non-empty range")

    @property
    def effective_flags(self) -> int:
        if self.flags is not None:
            return int(self.flags)
        f = 0
        if self.L_delta != 0:
            f |= 0x01
        if self.T_delta != 0:
            f |= 0x02
        return f

    def covers(self, q: int) -> bool:
        return self.start <= q < self.start + self.length


def _coord_matrix(
    active: npt.ArrayLike,
    dims: Sequence[int],
    origin: Sequence[int] | None,
    order: int,
) -> np.ndarray:
    """Origin-shifted logical coordinates for ``active`` as an ``(n, order)`` int64 matrix.

    Vectorizes ``q[dim] = (origin[dim] + unravel(linear)[dim]) % dims[dim]`` over all
    active linear indices at once. ``order == 0`` yields an ``(n, 0)`` matrix so the
    downstream ``Q @ coeff`` reductions collapse to zero per element.
    """
    act = np.asarray(active, dtype=np.int64)
    if order == 0:
        return np.zeros((act.size, 0), dtype=np.int64)
    org = origin or (0,) * order
    shape = tuple(int(d) for d in dims)
    coords = np.unravel_index(act, shape)
    cols = [(int(org[d]) + coords[d].astype(np.int64)) % int(dims[d]) for d in range(order)]
    return np.stack(cols, axis=1)


def _band_lookup(
    bands: Sequence[BandRange],
    dims: Sequence[int],
    order: int,
) -> tuple[list[np.ndarray | None], list[np.ndarray | None]]:
    """Per-dim additive band-delta lookup tables ``(L, T)``; ``None`` where a dim has none.

    Mirrors ``_band_delta`` exactly: a band on dim ``d`` adds ``L_delta`` / ``T_delta``
    (gated by ``effective_flags``) across ``[start, start+length)``; multiple bands on a
    dim accumulate. Indexing the table with the origin-shifted coord reproduces the
    per-element ``covers(q)`` test as a vectorized gather.
    """
    bandL: list[np.ndarray | None] = [None] * order
    bandT: list[np.ndarray | None] = [None] * order
    for band in bands:
        d = int(band.dim)
        if d < 0 or d >= order:
            continue
        flags = band.effective_flags
        start = int(band.start)
        end = start + int(band.length)
        if flags & 0x01:
            if bandL[d] is None:
                bandL[d] = np.zeros(int(dims[d]), dtype=np.int64)
            bandL[d][start:end] += int(band.L_delta)
        if flags & 0x02:
            if bandT[d] is None:
                bandT[d] = np.zeros(int(dims[d]), dtype=np.int64)
            bandT[d][start:end] += int(band.T_delta)
    return bandL, bandT


@dataclass(slots=True)
class AffineTensorBlock:
    """STB-AFFINE tensor payload with dense or sparse active residuals."""

    header: OperatorHeader
    active_encoding: ActiveEncoding = ActiveEncoding.DENSE_ALL
    dL: npt.ArrayLike | None = None
    dT: npt.ArrayLike | None = None
    active_indices: npt.ArrayLike | None = None
    bands: Sequence[BandRange] = field(default_factory=tuple)

    def validate(self) -> None:
        self.header.validate()
        total = self.header.total_elements
        if self.active_encoding == ActiveEncoding.DENSE_ALL:
            active_count = total
        elif self.active_encoding == ActiveEncoding.COO_U32:
            if self.active_indices is None:
                raise SPFError("COO_U32 requires active_indices")
            idx = np.asarray(self.active_indices, dtype=np.int64)
            if np.any((idx < 0) | (idx >= total)):
                raise SPFError("active index out of bounds")
            if idx.size and not np.all(idx[1:] > idx[:-1]):
                raise SPFError("active indices must be strictly increasing")
            active_count = int(idx.size)
        else:
            raise NotImplementedError(f"{self.active_encoding.name} decode is not implemented yet")
        if self.dL is None or self.dT is None:
            raise SPFError("dL and dT residual streams are required")
        if len(np.asarray(self.dL)) != active_count or len(np.asarray(self.dT)) != active_count:
            raise SPFError("residual stream count mismatch")
        self._validate_bands()

    def _validate_bands(self) -> None:
        by_dim: dict[int, list[BandRange]] = {}
        for band in self.bands:
            if band.dim >= self.header.ndims:
                raise SPFError("band dimension outside tensor rank")
            by_dim.setdefault(band.dim, []).append(band)
        for dim, ranges in by_dim.items():
            ranges = sorted(ranges, key=lambda b: b.start)
            prev_end = -1
            for band in ranges:
                if band.start < prev_end:
                    raise SPFError(f"overlapping band ranges on dim {dim}")
                if band.start + band.length > self.header.dims[dim]:
                    raise SPFError(f"band extends beyond dim {dim}")
                prev_end = band.start + band.length

    def reconstruct(self) -> tuple[np.ndarray, np.ndarray]:
        """Materialize dense L/T planes. Debug/export path, not hot matvec."""
        self.validate()
        total = self.header.total_elements
        L = np.full(total, SPF32_L_ZERO, dtype=np.int16)
        T = np.zeros(total, dtype=np.uint16)
        if self.active_encoding == ActiveEncoding.DENSE_ALL:
            active = np.arange(total, dtype=np.int64)
        else:
            active = np.asarray(self.active_indices, dtype=np.int64)
        dL_arr = np.asarray(self.dL, dtype=np.int16)
        dT_arr = np.asarray(self.dT, dtype=np.int16)
        origin = self.header.origin or (0, 0, 0, 0)
        ramps = self.header.ramp_alpha or (0, 0, 0, 0)
        dims = self.header.dims
        # Vectorized over active indices (see TensorBlock.reconstruct): pure integer
        # affine arithmetic, bit-identical to the scalar loop, mask deferred to the end.
        order = len(dims)
        Q = _coord_matrix(active, dims, origin, order)
        ramps_arr = np.asarray([int(ramps[d]) for d in range(order)], dtype=np.int64)
        l_val = np.full(Q.shape[0], np.int64(self.header.global_L_bias), dtype=np.int64)
        t_val = np.int64(self.header.global_T_bias) + Q @ ramps_arr
        if self.bands:
            bandL, bandT = _band_lookup(self.bands, dims, order)
            for dim in range(order):
                if bandL[dim] is not None:
                    l_val = l_val + bandL[dim][Q[:, dim]]
                if bandT[dim] is not None:
                    t_val = t_val + bandT[dim][Q[:, dim]]
        l_val = l_val + dL_arr.astype(np.int64)
        t_val = t_val + dT_arr.astype(np.int64)
        # A resolved L of exactly the zero sentinel (0x8000) is zero, not a finite
        # magnitude — preserve it (Θ forced 0) the way pack_words / spf_mul_planes do,
        # instead of clipping it to a tiny finite L. Keeps DiagonalOperator.apply_words
        # bit-identical to spf_mul_words for zero taps in a DENSE_ALL residual stream,
        # and matches the native C spf32_op_affine_reconstruct guard.
        zero = l_val == int(SPF32_L_ZERO)
        L[active] = np.where(zero, SPF32_L_ZERO, _sat_l16(l_val)).astype(np.int16)
        T[active] = np.where(zero, 0, t_val & SPF32_PHASE_MASK).astype(np.uint16)
        return L.reshape(dims), T.reshape(dims)

    def _band_delta(self, dim: int, q: int) -> tuple[int, int]:
        l_delta = 0
        t_delta = 0
        for band in self.bands:
            if band.dim != dim or not band.covers(q):
                continue
            flags = band.effective_flags
            if flags & 0x01:
                l_delta += int(band.L_delta)
            if flags & 0x02:
                t_delta += int(band.T_delta)
        return l_delta, t_delta

    def shift_view(self, dim: int, delta: int) -> AffineTensorBlock:
        return replace(self, header=self.header.shift_view(dim, delta))

    def apply_global_gain(self, delta_L: int) -> AffineTensorBlock:
        return replace(self, header=self.header.apply_global_gain(delta_L))

    def apply_global_phase(self, delta_T: int) -> AffineTensorBlock:
        return replace(self, header=self.header.apply_global_phase(delta_T))


@dataclass(slots=True)
class DiagonalOperator:
    """SOV-DIAG elementwise spectral operator."""

    weights: AffineTensorBlock
    _planes: tuple[np.ndarray, np.ndarray] | None = field(default=None, init=False, repr=False, compare=False)

    def materialize(self) -> tuple[np.ndarray, np.ndarray]:
        """Dense ``(L, Θ)`` weight planes, memoised.

        ``weights`` is immutable for the operator's lifetime, so the
        ``reconstruct()`` pass — a debug/export-grade materialisation, explicitly
        "not hot matvec" — is computed once and reused. A hot ``apply_*`` loop then
        pays only the vectorised ``spf_mul_planes`` per call, at parity with
        ``spf_mul_words``, instead of re-reconstructing the constant operator on
        every invocation.
        """
        if self._planes is None:
            self._planes = self.weights.reconstruct()
        return self._planes

    def apply_planes(self, xL: npt.ArrayLike, xT: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        aL, aT = self.materialize()
        xL_arr = np.asarray(xL, dtype=np.int16).reshape(aL.shape)
        xT_arr = np.asarray(xT, dtype=np.uint16).reshape(aT.shape)
        return spf_mul_planes(aL, aT, xL_arr, xT_arr)

    def apply_words(self, words: npt.ArrayLike) -> np.ndarray:
        yL, yT = self.apply_planes(unpack_L(words), unpack_theta(words))
        return pack_words(yL, yT)

    def compose_header_only(self, other: DiagonalOperator) -> DiagonalOperator:
        """Compose two compatible affine-ramp diagonals by merging headers.

        This fast path is intentionally conservative: it only accepts dense
        all-active operators with no bands and zero residuals, where the
        composition is a pure header update.
        """
        a = self.weights
        b = other.weights
        if a.header.dims != b.header.dims:
            raise SPFError("diagonal composition requires matching dims")
        for block in (a, b):
            if block.active_encoding != ActiveEncoding.DENSE_ALL:
                raise SPFError("header-only composition requires dense blocks")
            if block.bands:
                raise SPFError("header-only composition does not merge bands")
            if np.any(np.asarray(block.dL, dtype=np.int16) != 0) or np.any(np.asarray(block.dT, dtype=np.int16) != 0):
                raise SPFError("header-only composition requires zero residuals")
        ramps = tuple(int(x) + int(y) for x, y in zip(a.header.ramp_alpha, b.header.ramp_alpha, strict=True))
        header = replace(
            a.header,
            global_L_bias=int(a.header.global_L_bias) + int(b.header.global_L_bias),
            global_T_bias=(int(a.header.global_T_bias) + int(b.header.global_T_bias)) & SPF32_PHASE_MASK,
            ramp_alpha=ramps,
        )
        total = header.total_elements
        return DiagonalOperator(
            AffineTensorBlock(
                header=header,
                active_encoding=ActiveEncoding.DENSE_ALL,
                dL=np.zeros(total, dtype=np.int16),
                dT=np.zeros(total, dtype=np.int16),
            )
        )


@dataclass(slots=True)
class TiledInverse:
    """Coherence-gated per-tile MIMO channel inverse for OFDM equalization.

    Wraps :class:`DiagonalOperator`: the SISO per-subcarrier case degenerates to
    a diagonal scale by Ĥ⁻¹ (built via :meth:`as_diagonal`), while the MIMO case
    groups frequency-adjacent subcarriers into **coherence tiles**, inverts one
    averaged ``n_rx × n_tx`` channel matrix per tile (Moore–Penrose
    pseudoinverse), and applies that inverse across the tile.

    **Tile-size policy (validated in tests/dsp/test_sofdm_chanest_mimo.py):** a
    *smooth* (flat / low-delay-spread) channel is coherent over many subcarriers,
    so a large tile averages noise across subcarriers without distorting the
    matrix → *larger tiles*. A *frequency-selective* channel changes per bin, so a
    shared inverse is wrong → *per-subcarrier* (tile size 1). :meth:`choose_tile`
    picks the tile size from the measured coherence bandwidth.

    **Honest scoping:** the per-tile matrix inverse is a small FP64 *add-domain
    boundary solve* run once per channel estimate at RX egress. It is **NOT** on
    the multiplier-free SPF hot path and is **NOT** counted as an SPF-native win;
    only the resulting elementwise apply (the diagonal / block scale) is.
    """

    h: npt.ArrayLike  # per-subcarrier channel: (n_sub,) SISO or (n_sub, n_rx, n_tx).
    tile_size: int = 1
    rcond: float = 1e-10

    def __post_init__(self) -> None:
        h = np.asarray(self.h, dtype=np.complex128)
        if h.ndim == 1:
            h = h[:, None, None]
        elif h.ndim != 3:
            raise SPFError("TiledInverse h must be (n_sub,) or (n_sub, n_rx, n_tx)")
        object.__setattr__(self, "h", h)
        if int(self.tile_size) < 1:
            raise SPFError("tile_size must be >= 1")
        object.__setattr__(self, "tile_size", int(self.tile_size))

    @property
    def n_sub(self) -> int:
        return int(np.asarray(self.h).shape[0])

    @property
    def n_rx(self) -> int:
        return int(np.asarray(self.h).shape[1])

    @property
    def n_tx(self) -> int:
        return int(np.asarray(self.h).shape[2])

    @staticmethod
    def coherence_bandwidth(h: npt.ArrayLike, *, threshold: float = 0.9) -> int:
        """Estimate the coherence bandwidth (in subcarriers) of a channel.

        Measured as the largest lag whose mean normalised frequency
        autocorrelation magnitude stays ``>= threshold``. A flat channel returns
        ``n_sub`` (fully coherent); a fast-varying channel returns ``1``.
        """
        h_arr = np.asarray(h, dtype=np.complex128)
        if h_arr.ndim == 1:
            h_arr = h_arr[:, None, None]
        n_sub = h_arr.shape[0]
        flat = h_arr.reshape(n_sub, -1)
        # Normalised per-channel-tap frequency autocorrelation, averaged over taps.
        # Numerator (lag-l autocorrelation) and denominator (power) are computed on
        # the SAME n_sub-lag sample set so the ratio is unbiased at large lags; using
        # the full-frame power would inflate |corr|/power and over-estimate coherence.
        coh = 1
        for lag in range(1, n_sub):
            corr = np.mean(flat[lag:] * np.conj(flat[:-lag]), axis=0)
            overlap_power = np.mean(np.abs(flat[:-lag]) ** 2, axis=0)
            overlap_power = np.where(overlap_power == 0.0, 1.0, overlap_power)
            mean_corr = float(np.mean(np.abs(corr) / overlap_power))
            if mean_corr >= threshold:
                coh = lag + 1
            else:
                break
        return int(coh)

    @classmethod
    def choose_tile(
        cls,
        h: npt.ArrayLike,
        *,
        threshold: float = 0.9,
        max_tile: int | None = None,
    ) -> int:
        """Coherence-gated tile-size policy: smooth → large tile, selective → 1.

        Returns ``min(coherence_bandwidth, max_tile)``; a selective channel whose
        coherence bandwidth is one subcarrier yields per-subcarrier inversion.
        """
        coh = cls.coherence_bandwidth(h, threshold=threshold)
        if max_tile is not None:
            coh = min(coh, int(max_tile))
        return max(1, int(coh))

    @classmethod
    def from_channel(
        cls,
        h: npt.ArrayLike,
        *,
        threshold: float = 0.9,
        max_tile: int | None = None,
        rcond: float = 1e-10,
    ) -> TiledInverse:
        """Build a ``TiledInverse`` with the tile size chosen by the policy."""
        tile = cls.choose_tile(h, threshold=threshold, max_tile=max_tile)
        return cls(h=h, tile_size=tile, rcond=rcond)

    def tile_inverses(self) -> np.ndarray:
        """Per-tile pseudoinverse matrices broadcast back to every subcarrier.

        Shape ``(n_sub, n_tx, n_rx)``: the inverse applied at each subcarrier is
        the pseudoinverse of the tile-averaged channel matrix it belongs to.
        """
        h = np.asarray(self.h, dtype=np.complex128)
        n_sub, n_rx, n_tx = h.shape
        inv = np.empty((n_sub, n_tx, n_rx), dtype=np.complex128)
        for start in range(0, n_sub, self.tile_size):
            stop = min(start + self.tile_size, n_sub)
            h_tile = np.mean(h[start:stop], axis=0)  # averaged channel for the tile.
            inv[start:stop] = np.linalg.pinv(h_tile, rcond=self.rcond)
        return inv

    def as_diagonal(self) -> DiagonalOperator:
        """SISO fast path: per-subcarrier Ĥ⁻¹ as a :class:`DiagonalOperator`.

        Valid only for the ``n_rx == n_tx == 1`` case, where the tiled inverse is
        a scalar reciprocal per subcarrier and the equalize is exactly the
        elementwise SPF multiply that ``DiagonalOperator`` performs.
        """
        if self.n_rx != 1 or self.n_tx != 1:
            raise SPFError("as_diagonal requires a SISO (n_rx == n_tx == 1) channel")
        h_inv = self.tile_inverses().reshape(self.n_sub)
        header = OperatorHeader(profile_id=Profile.SOV_DIAG, dims=(self.n_sub, 1, 1, 1))
        block = AffineTensorBlock(
            header=header,
            active_encoding=ActiveEncoding.DENSE_ALL,
            dL=np.zeros(self.n_sub, dtype=np.int16),
            dT=np.zeros(self.n_sub, dtype=np.uint16),
        )
        # Carry the actual Ĥ⁻¹ values as the residual planes via encode_words.
        inv_words = encode_words(h_inv)
        block = replace(
            block,
            dL=unpack_L(inv_words).astype(np.int16),
            dT=unpack_theta(inv_words).astype(np.uint16),
        )
        return DiagonalOperator(block)

    def apply(self, rx: npt.ArrayLike) -> np.ndarray:
        """Equalize received MIMO symbols: ``x̂[k] = Ĥ_tile⁻¹ · y[k]``.

        ``rx`` is ``(n_sub,)`` SISO or ``(n_sub, n_rx)`` MIMO; returns the
        per-subcarrier transmit estimate (``(n_sub,)`` or ``(n_sub, n_tx)``).
        """
        y = np.asarray(rx, dtype=np.complex128)
        squeeze = y.ndim == 1
        if squeeze:
            y = y[:, None]
        if y.shape[0] != self.n_sub or y.shape[1] != self.n_rx:
            raise SPFError(f"rx shape {y.shape} incompatible with channel (n_sub={self.n_sub}, n_rx={self.n_rx})")
        inv = self.tile_inverses()  # (n_sub, n_tx, n_rx)
        out = np.einsum("kij,kj->ki", inv, y)  # (n_sub, n_tx)
        return out[:, 0] if squeeze else out


@dataclass(slots=True)
class LowRankOperator:
    """SOV-LR matrix operator: A ≈ Σ_r λ_r u_r v_rᴴ."""

    u_factors: npt.ArrayLike
    v_factors: npt.ArrayLike
    lambdas: npt.ArrayLike
    reducer_id: ReducerPolicy | ReducerSpec = field(
        default_factory=lambda: ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN)
    )

    def apply_words(self, x: npt.ArrayLike) -> np.ndarray:
        from . import reducers

        reducer = ensure_reducer_spec(self.reducer_id)
        if reducer.policy not in (ReducerPolicy.CARTESIAN_KAHAN, ReducerPolicy.AUTO, ReducerPolicy.V02):
            raise NotImplementedError("Python LR reference implements CARTESIAN_KAHAN, AUTO, and V02")
        u = np.asarray(self.u_factors, dtype=np.uint32)
        v = np.asarray(self.v_factors, dtype=np.uint32)
        lam = np.asarray(self.lambdas, dtype=np.uint32)
        xw = np.asarray(x, dtype=np.uint32)
        if u.ndim != 2 or v.ndim != 2 or u.shape[0] != v.shape[0] or lam.shape[0] != u.shape[0]:
            raise SPFError("low-rank factors must have shapes [rank,out], [rank,in], [rank]")
        if xw.shape[0] != v.shape[1]:
            raise SPFError("input dimension mismatch")
        if u.shape[0] == 0:
            return np.full(u.shape[1], SPF32_WORD_ZERO, dtype=np.uint32)

        products = spf_mul_words(spf_conj_words(v), xw[None, :])
        if reducer.policy is ReducerPolicy.V02:
            inner = np.asarray([reduce_words_v02(row, spec=reducer) for row in products], dtype=np.uint32)
            scaled = spf_mul_words(lam, inner)
            out_terms = spf_mul_words(u, np.asarray(scaled, dtype=np.uint32)[:, None])
            return np.asarray(
                [reduce_words_v02(out_terms[:, column], spec=reducer) for column in range(out_terms.shape[1])],
                dtype=np.uint32,
            )
        inner = reducers.reduce_axis(products, reducer, axis=1)
        scaled = spf_mul_words(lam, np.asarray(inner, dtype=np.uint32))
        out_terms = spf_mul_words(u, np.asarray(scaled, dtype=np.uint32)[:, None])
        return np.asarray(reducers.reduce_axis(out_terms, reducer, axis=0), dtype=np.uint32)


class HDCPhaseOperator:
    """SOV-HDC phase-only bind/unbind and triangular similarity helpers."""

    @staticmethod
    def bind(a: npt.ArrayLike, b: npt.ArrayLike) -> np.ndarray:
        summed = (np.asarray(a, dtype=np.uint32) + np.asarray(b, dtype=np.uint32)) & SPF32_PHASE_MASK
        return summed.astype(np.uint16)

    @staticmethod
    def unbind(bound: npt.ArrayLike, factor: npt.ArrayLike) -> np.ndarray:
        return ((np.asarray(bound, dtype=np.uint32) - np.asarray(factor, dtype=np.uint32)) & SPF32_PHASE_MASK).astype(
            np.uint16
        )

    @staticmethod
    def conjugate(a: npt.ArrayLike) -> np.ndarray:
        return ((-np.asarray(a, dtype=np.uint32)) & SPF32_PHASE_MASK).astype(np.uint16)

    @staticmethod
    def triangular_similarity_i64(a: npt.ArrayLike, b: npt.ArrayLike) -> int:
        """Wide triangular phase similarity.

        This is not FHRR cosine similarity. It is the same monotone triangular
        phase score family as the SPF8/SPF16 HDC kernels, accumulated in int64
        for production ranking paths.
        """
        aa = np.asarray(a, dtype=np.uint16)
        bb = np.asarray(b, dtype=np.uint16)
        if aa.shape != bb.shape:
            raise SPFError("HDC similarity shape mismatch")
        delta = circular_diff_u16(aa, bb)
        abs_delta = np.abs(np.asarray(delta, dtype=np.int64))
        score = np.int64(SPF32_PHASE_MOD // 2) - abs_delta
        return int(np.sum(score, dtype=np.int64))

    @staticmethod
    def triangular_similarity(a: npt.ArrayLike, b: npt.ArrayLike) -> int:
        """Legacy int32 triangular similarity wrapper with saturation."""
        score = HDCPhaseOperator.triangular_similarity_i64(a, b)
        return int(min(max(score, INT32_MIN), INT32_MAX))

    @staticmethod
    def bundle(
        vectors: npt.ArrayLike,
        *,
        reducer: ReducerPolicy = ReducerPolicy.CARTESIAN_KAHAN,
        bins: int = 64,
    ) -> np.ndarray:
        phases = np.asarray(vectors, dtype=np.uint16)
        if phases.ndim != 2:
            raise SPFError("HDC bundle expects [n_vectors, dims]")
        policy = ReducerPolicy(reducer)
        raise SPFError(
            f"HDC bundling reducer {policy.name} is not implemented in Level-4 v0; "
            "cosine-LUT similarity and circular-mean bundling are follow-up work"
        )


@dataclass(frozen=True, slots=True)
class QAM16LogProjMetadata:
    """OperatorBlock metadata for SPF-native QAM16 log-projection demap."""

    qam_demap: QAMDemap = QAMDemap.LOGPROJ
    qam_projection_phase_bits: int = 10
    qam_projection_threshold_L: int = 0
    agc_L_offset: int = 0
    cpe_theta: int = 0
    erasure_policy: str = "mask_zeroes_bits_and_confidence"
    confidence_output: bool = True

    def validate(self) -> None:
        if QAMDemap(self.qam_demap) != QAMDemap.LOGPROJ:
            raise SPFError("only QAM16 log-projection demap metadata is supported")
        if not 1 <= int(self.qam_projection_phase_bits) <= 12:
            raise SPFError("qam_projection_phase_bits must be in [1, 12]")
        if not SPF32_L_MIN_FINITE <= int(self.qam_projection_threshold_L) <= SPF32_L_MAX_FINITE:
            raise SPFError("qam_projection_threshold_L out of SPF32 finite L range")
        if not SPF32_L_MIN_FINITE <= int(self.agc_L_offset) <= SPF32_L_MAX_FINITE:
            raise SPFError("agc_L_offset out of SPF32 finite L range")
        if not 0 <= int(self.cpe_theta) <= SPF32_PHASE_MASK:
            raise SPFError("cpe_theta must be a u16 phase")
        if self.erasure_policy not in {"mask_zeroes_bits_and_confidence", "none"}:
            raise SPFError("unsupported QAM16 erasure policy")

    def to_metadata(self) -> dict[str, int | str | bool]:
        self.validate()
        data = asdict(self)
        data["qam_demap"] = "logproj"
        return data

    @classmethod
    def from_metadata(cls, data: Mapping[str, object]) -> QAM16LogProjMetadata:
        qam_demap = data.get("qam_demap", "logproj")
        if qam_demap not in ("logproj", QAMDemap.LOGPROJ, int(QAMDemap.LOGPROJ)):
            raise SPFError("only qam_demap=logproj is supported")
        obj = cls(
            qam_projection_phase_bits=int(data.get("qam_projection_phase_bits", 10)),
            qam_projection_threshold_L=int(data.get("qam_projection_threshold_L", 0)),
            agc_L_offset=int(data.get("agc_L_offset", 0)),
            cpe_theta=int(data.get("cpe_theta", 0)),
            erasure_policy=str(data.get("erasure_policy", "mask_zeroes_bits_and_confidence")),
            confidence_output=bool(data.get("confidence_output", True)),
        )
        obj.validate()
        return obj

    def equalized_projection_planes(
        self,
        y_words: npt.ArrayLike,
        h_inv_words: npt.ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return L_eq/theta_eq and log-projection L_I/L_Q planes."""
        self.validate()
        y = np.asarray(y_words, dtype=np.uint32)
        h = np.asarray(h_inv_words, dtype=np.uint32)
        if y.shape != h.shape:
            raise SPFError("QAM16 logproj y/h_inv shape mismatch")
        yL = unpack_L(y)
        hL = unpack_L(h)
        yT = unpack_theta(y)
        hT = unpack_theta(h)
        zero = (yL == SPF32_L_ZERO) | (hL == SPF32_L_ZERO)
        L_eq = _sat_l16(yL.astype(np.int64) + hL.astype(np.int64) + int(self.agc_L_offset))
        theta_eq = ((yT.astype(np.uint32) + hT.astype(np.uint32) + int(self.cpe_theta)) & SPF32_PHASE_MASK).astype(
            np.uint16
        )
        angle = theta_eq.astype(np.float64) * (2.0 * np.pi / SPF32_PHASE_MOD)
        eps = np.finfo(np.float64).tiny
        L_I = _sat_l16(L_eq.astype(np.int64) + np.rint(np.log2(np.maximum(np.abs(np.cos(angle)), eps)) * SPF32_L_SCALE))
        L_Q = _sat_l16(L_eq.astype(np.int64) + np.rint(np.log2(np.maximum(np.abs(np.sin(angle)), eps)) * SPF32_L_SCALE))
        L_eq = np.where(zero, SPF32_L_ZERO, L_eq).astype(np.int16)
        L_I = np.where(zero, SPF32_L_ZERO, L_I).astype(np.int16)
        L_Q = np.where(zero, SPF32_L_ZERO, L_Q).astype(np.int16)
        theta_eq = np.where(zero, 0, theta_eq).astype(np.uint16)
        return L_eq, theta_eq, L_I, L_Q


@dataclass(slots=True)
class SynthStats:
    synth_mode: SynthMode
    active_tones: int = 0
    ifft_calls: int = 0
    nco_tones: int = 0
    template_hits: int = 0
    estimated_adds: int = 0
    estimated_lut_reads: int = 0
    boundary_crossings: int = 1
    clip_count: int = 0
    papr: float = 0.0
    evm_vs_reference: float = 0.0


@dataclass(slots=True)
class SynthPlan:
    """Terminal export policy for crossing from SPF spectral tensors to DAC output."""

    nfft: int
    ncp: int = 0
    n_symbols: int = 1
    sample_rate: int = 0
    phase_acc_bits: int = 32
    subcarrier_spacing_hz_q: int = 0
    center_frequency_hz_q: int = 0
    sin_lut_bits: int = 12
    amp_lut_bits: int = 12
    output_format: OutputFormat = OutputFormat.IQ_INT16
    scale_policy: ScalePolicy = ScalePolicy.BLOCK_FLOAT_BACKOFF
    clip_policy: ClipPolicy = ClipPolicy.SATURATE
    synth_mode: SynthMode = SynthMode.AUTO
    sparse_threshold: int = 96
    active_mask_offset: int = 0
    tile_map_offset: int = 0
    window_profile_id: int = 0

    def validate(self) -> None:
        if self.nfft <= 0 or self.n_symbols <= 0:
            raise SPFError("nfft and n_symbols must be positive")
        if self.ncp < 0 or self.ncp > self.nfft:
            raise SPFError("ncp must lie in [0, nfft]")
        if self.phase_acc_bits < 16 or self.phase_acc_bits > 32:
            raise SPFError("phase_acc_bits must be 16..32 for this implementation")
        if OutputFormat(self.output_format) != OutputFormat.IQ_INT16:
            raise SPFError("SynthPlan only implements IQ_INT16 normal DAC export; REAL_INT16 is unsupported")

    def choose_mode(self, *, active_count: int, total_bins: int | None = None) -> SynthMode:
        self.validate()
        total = self.nfft if total_bins is None else int(total_bins)
        if self.synth_mode != SynthMode.AUTO:
            return SynthMode(self.synth_mode)
        if active_count <= self.sparse_threshold or active_count * 4 <= total:
            return SynthMode.NORMAL_DAC_NCO
        return SynthMode.NORMAL_DAC_IFFT

    @property
    def samples_per_symbol(self) -> int:
        return self.nfft + self.ncp

    def export_spectral_dac_descriptors(
        self,
        bins: npt.ArrayLike,
        active_indices: Sequence[int] | None = None,
    ) -> dict[str, np.ndarray | int]:
        """Return amplitude/phase/FCW descriptors for a spectral DAC or NCO bank."""
        self.validate()
        words = _reshape_symbols(np.asarray(bins, dtype=np.uint32), self)
        active = _active_indices(words[0], active_indices)
        first = words[0, active]
        L = unpack_L(first).astype(np.int32)
        T = unpack_theta(first).astype(np.uint32)
        amp = np.where(int(SPF32_L_ZERO) == L, 0.0, np.exp2(L / SPF32_L_SCALE)).astype(np.float32)
        shift = self.phase_acc_bits - SPF32_PHASE_BITS
        phase_init = (T << shift).astype(np.uint32) if shift >= 0 else (T >> -shift).astype(np.uint32)
        fcw_num = active.astype(np.uint64) << self.phase_acc_bits
        fcw = (fcw_num // np.uint64(self.nfft)).astype(np.uint32)
        return {
            "mode": int(SynthMode.SPECTRAL_DAC),
            "active_indices": active.astype(np.uint32),
            "amplitude": amp,
            "phase_init": phase_init,
            "fcw": fcw,
            "nfft": self.nfft,
            "ncp": self.ncp,
            "n_symbols": self.n_symbols,
        }

    def export_normal_dac(
        self,
        bins: npt.ArrayLike,
        *,
        active_indices: Sequence[int] | None = None,
        templates: Mapping[int, npt.ArrayLike] | None = None,
    ) -> tuple[np.ndarray, SynthStats]:
        """Emit interleaved IQ int16 samples using the cheapest legal mode."""
        self.validate()
        words = _reshape_symbols(np.asarray(bins, dtype=np.uint32), self)
        active = _active_indices_for_symbols(words, active_indices)
        mode = self.choose_mode(active_count=int(active.size), total_bins=self.nfft)
        if mode == SynthMode.SPECTRAL_DAC:
            raise SPFError("SPECTRAL_DAC mode exports descriptors, not normal DAC samples")
        if mode == SynthMode.TEMPLATE_CODEBOOK:
            if templates is None:
                raise SPFError("template synth requires templates")
            samples, stats = self._template_synth(words, active, templates)
        elif mode == SynthMode.NORMAL_DAC_NCO:
            samples, stats = self._nco_synth(words, active)
        elif mode in (SynthMode.NORMAL_DAC_IFFT, SynthMode.HYBRID_TILE):
            samples, stats = self._ifft_synth(words)
            if mode == SynthMode.HYBRID_TILE:
                stats.synth_mode = SynthMode.HYBRID_TILE
        else:
            raise NotImplementedError(f"synth mode {mode.name} not implemented")
        return samples, stats

    def _nco_synth(self, words: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, SynthStats]:
        z = decode_words(words)
        nn = np.arange(self.samples_per_symbol, dtype=np.int64)
        n_view = np.where(nn < self.ncp, self.nfft - self.ncp + nn, nn - self.ncp)
        phase = np.exp(2j * np.pi * active[:, None] * n_view[None, :] / self.nfft)
        out = np.sum(z[:, active, None] * phase[None, :, :], axis=1) / self.nfft
        iq, papr, clips = _quantize_iq16(out.reshape(-1), self.clip_policy)
        stats = SynthStats(
            synth_mode=SynthMode.NORMAL_DAC_NCO,
            active_tones=int(active.size),
            nco_tones=int(active.size * self.n_symbols),
            estimated_adds=int(self.samples_per_symbol * self.n_symbols * active.size * 2),
            estimated_lut_reads=int(self.samples_per_symbol * self.n_symbols * active.size * 2),
            clip_count=clips,
            papr=papr,
        )
        return iq, stats

    def _ifft_synth(self, words: np.ndarray) -> tuple[np.ndarray, SynthStats]:
        z = decode_words(words)
        useful = np.fft.ifft(z, axis=1)
        out = np.concatenate([useful[:, -self.ncp :], useful], axis=1) if self.ncp else useful
        iq, papr, clips = _quantize_iq16(out.reshape(-1), self.clip_policy)
        stats = SynthStats(
            synth_mode=SynthMode.NORMAL_DAC_IFFT,
            active_tones=self.nfft,
            ifft_calls=self.n_symbols,
            estimated_adds=int(self.n_symbols * self.nfft * math.log2(max(self.nfft, 2))),
            clip_count=clips,
            papr=papr,
        )
        return iq, stats

    def _template_synth(
        self,
        words: np.ndarray,
        active: np.ndarray,
        templates: Mapping[int, npt.ArrayLike],
    ) -> tuple[np.ndarray, SynthStats]:
        # Minimal codebook path: a zero-copy template map keyed by a hash of the
        # active SPF words. Global gain/phase can be layered by callers before
        # export; this routine deliberately remains exact-template only.
        chunks: list[np.ndarray] = []
        hits = 0
        for m in range(self.n_symbols):
            key = int(np.bitwise_xor.reduce(words[m, active], initial=np.uint32(0)))
            if key not in templates:
                sub_plan = replace(self, n_symbols=1, synth_mode=SynthMode.NORMAL_DAC_IFFT)
                fallback, _ = sub_plan._ifft_synth(words[m : m + 1])
                chunks.append(fallback.reshape(-1, 2))
                continue
            arr = np.asarray(templates[key], dtype=np.int16)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 2)
            chunks.append(arr)
            hits += 1
        out = np.concatenate(chunks, axis=0).astype(np.int16).reshape(-1)
        return out, SynthStats(synth_mode=SynthMode.TEMPLATE_CODEBOOK, template_hits=hits)


def _row_major_strides(dims: Sequence[int]) -> tuple[int, ...]:
    strides: list[int] = []
    acc = 1
    for dim in reversed(tuple(dims)):
        strides.append(acc)
        acc *= int(dim)
    return tuple(reversed(strides))


def _pad_tuple(values: Sequence[int] | None, n: int, fill: int) -> tuple[int, ...]:
    vals = tuple(int(v) for v in (values or ()))
    if len(vals) > n:
        raise SPFError(f"expected <= {n} values")
    return vals + tuple(fill for _ in range(n - len(vals)))


def _reshape_symbols(words: np.ndarray, plan: SynthPlan) -> np.ndarray:
    if words.ndim == 1:
        expected = plan.n_symbols * plan.nfft
        if words.size == plan.nfft and plan.n_symbols == 1:
            return words.reshape(1, plan.nfft)
        if words.size != expected:
            raise SPFError(f"bins length {words.size} does not match n_symbols*nfft={expected}")
        return words.reshape(plan.n_symbols, plan.nfft)
    if words.shape != (plan.n_symbols, plan.nfft):
        raise SPFError(f"bins shape {words.shape} does not match {(plan.n_symbols, plan.nfft)}")
    return words


def _active_indices(words: np.ndarray, active_indices: Sequence[int] | None) -> np.ndarray:
    if active_indices is not None:
        active = np.asarray(active_indices, dtype=np.int64)
    else:
        active = np.flatnonzero(unpack_L(words) != SPF32_L_ZERO).astype(np.int64)
    if np.any((active < 0) | (active >= words.shape[-1])):
        raise SPFError("active index out of range")
    if active.size and not np.all(active[1:] > active[:-1]):
        raise SPFError("active_indices must be strictly increasing")
    return active.astype(np.int64)


def _active_indices_for_symbols(
    words: np.ndarray,
    active_indices: Sequence[int] | None,
) -> np.ndarray:
    if active_indices is not None:
        active = _active_indices(words[0], active_indices)
        live = np.flatnonzero(np.any(unpack_L(words) != SPF32_L_ZERO, axis=0)).astype(np.int64)
        missing = np.setdiff1d(live, active, assume_unique=False)
        if missing.size:
            raise SPFError(
                "active_indices omit nonzero bins across symbols: " + ",".join(str(int(v)) for v in missing[:8])
            )
        return active
    active_mask = np.any(unpack_L(words) != SPF32_L_ZERO, axis=0)
    return np.flatnonzero(active_mask).astype(np.int64)


def _quantize_iq16(samples: np.ndarray, clip_policy: ClipPolicy) -> tuple[np.ndarray, float, int]:
    if samples.size == 0:
        return np.empty(0, dtype=np.int16), 0.0, 0
    power = np.abs(samples) ** 2
    mean_power = float(np.mean(power))
    papr = float(np.max(power) / mean_power) if mean_power > 0 else 0.0
    peak = float(max(np.max(np.abs(samples.real)), np.max(np.abs(samples.imag)), 1e-30))
    scale = 30000.0 / peak
    q = np.empty((samples.size, 2), dtype=np.float64)
    q[:, 0] = np.rint(samples.real * scale)
    q[:, 1] = np.rint(samples.imag * scale)
    clipped = np.logical_or(q > 32767.0, q < -32768.0)
    clips = int(np.count_nonzero(clipped))
    if clips and ClipPolicy(clip_policy) == ClipPolicy.ERROR:
        raise SPFError(f"normal DAC export clipped {clips} samples")
    q = np.clip(q, -32768.0, 32767.0).astype(np.int16)
    return q.reshape(-1), papr, clips


__all__ = [
    "SPF32_L_NAN",
    "SPF32_L_ZERO",
    "SPF32_T_NAN",
    "SPF32_WORD_NAN",
    "SPF32_WORD_ZERO",
    "ActiveEncoding",
    "AffineTensorBlock",
    "BandRange",
    "ClipPolicy",
    "DiagonalOperator",
    "HDCPhaseOperator",
    "LowRankOperator",
    "OperatorHeader",
    "OutputFormat",
    "PrecisionMode",
    "Profile",
    "QAM16LogProjMetadata",
    "QAMDemap",
    "ReducerPolicy",
    "ReducerSpec",
    "ReductionOrder",
    "SPFError",
    "ScalarFormat",
    "ScalePolicy",
    "SynthMode",
    "SynthPlan",
    "SynthStats",
    "circular_diff_u16",
    "decode_complex",
    "decode_words",
    "encode_complex",
    "encode_words",
    "pack_words",
    "reduce_words",
    "spf_conj_words",
    "spf_mul_planes",
    "spf_mul_words",
    "unpack_L",
    "unpack_theta",
]
