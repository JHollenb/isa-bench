"""SPFTensor v2 compatibility model.

The v0.1 SOB-96 ``OperatorHeader`` remains the compatibility ABI. This module
adds the staged v2 Python model used by newer TensorBlock, reducer, execution,
and serialization tests.  Its historical ``ramp_R`` aggregate is kept as a
compatibility ABI; the normative v0.2 quadratic-field surface is
``spf.tensor.v02.Jet2D``, whose gradient/Hessian representation has no
column-sum ambiguity.
"""

from __future__ import annotations

import json
import struct
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt

from .core import (
    SPF32_L_MIN_FINITE,
    SPF32_L_SCALE,
    SPF32_L_ZERO,
    SPF32_PHASE_MASK,
    SPF32_PHASE_MOD,
    SPF32_WORD_ZERO,
    AccumulationDomain,
    ActiveEncoding,
    AffineTensorBlock,
    BandRange,
    LowRankOperator,
    OperatorHeader,
    PrecisionMode,
    Profile,
    ReducerPolicy,
    ReducerSpec,
    ScalarFormat,
    SPFError,
    _band_lookup,
    _coord_matrix,
    _sat_l16,
    decode_words,
    encode_words,
    format_for,
    pack_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)
from .op_registry import OP_REGISTRY, execution_ops, lifted_ops

SPFTENSOR_MAJOR = 2
SPFTENSOR_MINOR = 0
SPFTENSOR_DTYPE_V2 = "SPF_TB_v2"
SPFTENSOR_DTYPE_V1 = "SPF_TB_v1"
SPFTENSOR_MAX_ORDER = 4
# 18 words before 2026-09-17; ``phase_gradient[4]`` (8 bytes) and
# ``mixed_ramp[6]`` (12 bytes) were carved from the front of the reserved area
# so the header size and every earlier field offset are unchanged (ABI v22).
# ABI v23 carved ``log_quadratic[4]`` (int16, 8 bytes) and ``log_mixed[6]``
# (int16, 12 bytes) next, leaving 8 reserved words.
# ABI v24 carved ``cubic_mixed[2]`` (u16, 4 bytes) next, leaving 7 reserved
# words; the header is still 256 bytes and every earlier offset is unchanged.
SPFTENSOR_RESERVED_WORDS = 7
SPFTENSOR_MIXED_RAMP_MAX = 6
SPFTENSOR_LOG_MIXED_MAX = 6
#: AlgRef v0.2 §13.6 mixed cubic terms, 2-D only: ``x^2 y`` and ``x y^2``.
SPFTENSOR_CUBIC_MIXED_MAX = 2

PROFILE_TT_RANK_MIN = 0x1000
PROFILE_TT_RANK_MAX = 0x3FFF
PROFILE_CP_CLOSED = 0x4000
PROFILE_V3_NONABELIAN_MIN = 0x8000
PROFILE_V3_NONABELIAN_MAX = 0xBFFF

# Stage-4 cancellation-risk screening defaults. ``dynamic_range`` is compared in
# log2 units against the spec cancellation-wedge magnitude bound ``|ΔL| < 1.0``
# (docs/spf-formats-and-test-plans.md §T1.6/§T1.7). ``coherence`` caps the mean
# resultant length: cancellation needs phases that OPPOSE (low R̄), the block-scale
# analogue of the wedge's ``|ΔΘ − π| < 0.5`` opposite-phase requirement — high
# coherence is the aligned, reinforcing, numerically easy case. The screen
# pre-filters blocks for routing to a cotransformation reducer; it is not a proof
# of catastrophic cancellation.
BLOCK_CANCEL_DL_THRESHOLD = 1.0
BLOCK_CANCEL_COHERENCE_MAX = 0.5

# Legal execution ops are declared in :mod:`spf.tensor.op_registry` (the single
# source of truth). ``EXECUTION_OPS`` is a snapshot of the builtin set taken at
# import for back-compat; it does *not* reflect ops added later via
# ``register_op``. :class:`ExecutionStep` validates against the *live* registry,
# so a newly registered op is plan-legal without editing this whitelist — call
# :func:`~spf.tensor.op_registry.execution_ops` for a live membership check.
EXECUTION_OPS = execution_ops()

# L code for the SPF-32 inf/NaN sentinel band (int16 0x7FFF). Together with
# SPF32_L_ZERO (0x8000) these are the two non-finite L codes; ``ramp`` passes
# such words through unchanged rather than perturbing their phase.
SPF32_L_SENTINEL = np.int16(0x7FFF)


def _u16(value: int) -> int:
    return int(value) & SPF32_PHASE_MASK


def _as_int_tuple(values: Sequence[int] | None, n: int, fill: int = 0) -> tuple[int, ...]:
    vals = tuple(int(v) for v in (values or ()))
    if len(vals) > n:
        raise SPFError(f"expected <= {n} values")
    return vals + tuple(fill for _ in range(n - len(vals)))


def _normalize_u16_vector(values: Sequence[int] | None, n: int) -> tuple[int, ...]:
    return tuple(_u16(v) for v in _as_int_tuple(values, n, 0))


def _normalize_i16_vector(values: Sequence[int] | None, n: int, *, name: str) -> tuple[int, ...]:
    """Signed 16-bit L-tick vector; out of range raises.

    Used by every header field the C ABI declares ``int16_t``: ``band_exp``
    (``spf32_tensorblock_v2_header_t.band_exp[4]``) and the ABI v23
    ``log_quadratic`` / ``log_mixed``. Unlike the phase vectors these do *not*
    wrap — the log polynomial is evaluated in a widened accumulator and
    saturated once, so an out-of-range coefficient would make the Python and C
    reconstructions disagree (Python widens, C truncates to int16) instead of
    agreeing modulo anything.
    """

    vals = _as_int_tuple(values, n, 0)
    for v in vals:
        if not -32768 <= int(v) <= 32767:
            raise SPFError(f"{name} entries must fit int16 L ticks")
    return vals


def cubic_mixed_terms(order: int) -> tuple[tuple[int, int], ...]:
    """Exponent pairs ``(px, py)`` of the ``cubic_mixed`` terms, storage order.

    AlgRef v0.2 §13.6: a full cubic in two dimensions needs ``x^3``, ``x^2 y``,
    ``x y^2``, ``y^3``. The axis-only cubes live in :attr:`cubic_ramp`; the two
    MIXED cubics are ``cubic_mixed[0] * q0^2 * q1`` and
    ``cubic_mixed[1] * q0 * q1^2``.

    Only order 2 has these terms. §13.5 is explicit that axis-only cubic terms
    are not closed under rotation and that the primary fast path should be "full
    cross-term quadratics plus subdivision, rather than axis-only cubics"; §13.6
    makes full cubic support optional. Orders 0/1/3/4 therefore return ``()``
    and must leave the two words zero — extending to order 3+ would need
    ``x^2 z``, ``y^2 z``, ``xyz``, ... (7 more terms at order 3), which does not
    fit the 2 carved words and has no §13.6 mandate.
    """

    return ((2, 1), (1, 2)) if int(order) == 2 else ()


def _cubic_mixed_phase(Q: np.ndarray, cubic_mixed: Sequence[int], order: int) -> np.ndarray:
    """Vectorized ``sum_k coeff[k] * q0^px * q1^py`` over coordinate rows ``Q``."""

    total = np.zeros(Q.shape[0], dtype=np.int64)
    for k, (px, py) in enumerate(cubic_mixed_terms(order)):
        coeff = int(cubic_mixed[k]) if k < len(cubic_mixed) else 0
        if coeff:
            total = total + coeff * (Q[:, 0] ** px) * (Q[:, 1] ** py)
    return total


def mixed_ramp_pairs(order: int) -> tuple[tuple[int, int], ...]:
    """Dimension pairs ``(i, j)``, ``i < j``, in ``mixed_ramp`` storage order.

    Lexicographic: order 2 → ``((0, 1),)``; order 3 → ``((0,1),(0,2),(1,2))``;
    order 4 → six pairs. This is the packed upper triangle of the symmetric
    phase Hessian of design §12 (``H_ij`` for ``i != j``); the diagonal lives in
    ``quadratic_ramp`` as ``H_ii / 2``.
    """

    n = int(order)
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def _mixed_ramp_phase(Q: np.ndarray, mixed_ramp: Sequence[int], order: int) -> np.ndarray:
    """Vectorized ``sum_k coeff[k] * q_i * q_j`` over coordinate rows ``Q``.

    Shared by the phase (``mixed_ramp``, u16) and log-magnitude (``log_mixed``,
    i16) off-diagonal terms; both use the :func:`mixed_ramp_pairs` order.
    """

    total = np.zeros(Q.shape[0], dtype=np.int64)
    for k, (i, j) in enumerate(mixed_ramp_pairs(order)):
        coeff = int(mixed_ramp[k]) if k < len(mixed_ramp) else 0
        if coeff:
            total = total + coeff * Q[:, i] * Q[:, j]
    return total


def _ramp_theta_increment(
    shape: tuple[int, ...],
    ramp_R: Sequence[Sequence[int]],
    ramp_b: Sequence[int],
    quadratic_ramp: Sequence[int] | None,
    cubic_ramp: Sequence[int] | None,
    *,
    nonlinear: bool,
    phase_gradient: Sequence[int] | None = None,
    mixed_ramp: Sequence[int] | None = None,
    cubic_mixed: Sequence[int] | None = None,
) -> np.ndarray:
    """Per-element phase-ramp increment over the logical index grid.

    Mirrors :meth:`TensorBlockHeaderV2.ramp_phase` exactly (same row/col
    summation, ``& SPF32_PHASE_MASK``) but vectorised over an arbitrary-rank
    index grid. Integer math is order-independent modulo 2^16, so the column
    folding below is bit-identical to the reference double loop.
    """
    order = len(tuple(ramp_R))
    inc = np.zeros(shape, dtype=np.int64)
    inc += sum(int(ramp_b[row]) for row in range(order))
    if order:
        coords = np.indices(shape, dtype=np.int64)  # (order, *shape)
        grad = tuple(phase_gradient or ()) + (0,) * order
        for col in range(order):
            colsum = sum(int(ramp_R[row][col]) for row in range(order)) + int(grad[col])
            if colsum:
                inc += colsum * coords[col]
        if nonlinear:
            quad = tuple(quadratic_ramp or ()) + (0,) * order
            cube = tuple(cubic_ramp or ()) + (0,) * order
            for dim in range(order):
                c = coords[dim]
                if int(quad[dim]):
                    inc += int(quad[dim]) * c * c
                if int(cube[dim]):
                    inc += int(cube[dim]) * c * c * c
            mixed = tuple(mixed_ramp or ())
            for k, (i, j) in enumerate(mixed_ramp_pairs(order)):
                coeff = int(mixed[k]) if k < len(mixed) else 0
                if coeff:
                    inc += coeff * coords[i] * coords[j]
            # §13.6 mixed cubics (2-D only): x^2 y and x y^2.
            cmixed = tuple(cubic_mixed or ())
            for k, (px, py) in enumerate(cubic_mixed_terms(order)):
                coeff = int(cmixed[k]) if k < len(cmixed) else 0
                if coeff:
                    inc += coeff * (coords[0] ** px) * (coords[1] ** py)
    return inc & SPF32_PHASE_MASK


def _normalize_u16_matrix(
    matrix: Sequence[Sequence[int]] | npt.ArrayLike | None,
    n: int,
) -> tuple[tuple[int, ...], ...]:
    if matrix is None:
        return tuple(tuple(0 for _ in range(n)) for _ in range(n))
    arr = np.asarray(matrix, dtype=np.int64)
    if n == 0 and arr.size == 0:
        return tuple()
    if arr.shape != (n, n):
        raise SPFError(f"ramp_R must have shape {(n, n)}")
    return tuple(tuple(_u16(int(arr[r, c])) for c in range(n)) for r in range(n))


def _identity_ramp_R(dims: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    n = len(dims)
    return tuple(tuple(1 if r == c else 0 for c in range(n)) for r in range(n))


def active_dimensions(dims: Sequence[int]) -> tuple[int, ...]:
    """Dimensions participating in v2 ramp invertibility checks."""

    return tuple(i for i, dim in enumerate(dims) if int(dim) > 1)


def determinant_mod2(matrix: Sequence[Sequence[int]] | npt.ArrayLike) -> int:
    """Return ``det(matrix) mod 2`` using Gaussian elimination over GF(2)."""

    a = np.asarray(matrix, dtype=np.uint8) & np.uint8(1)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise SPFError("determinant_mod2 requires a square matrix")
    n = int(a.shape[0])
    det = 1
    rank = 0
    col = 0
    while col < n:
        pivot = None
        for r in range(rank, n):
            if int(a[r, col]) != 0:
                pivot = r
                break
        if pivot is None:
            det = 0
            break
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        for r in range(n):
            if r != rank and int(a[r, col]) != 0:
                a[r, :] ^= a[rank, :]
        rank += 1
        col += 1
    return int(det)


def ramp_determinant_mod2(
    ramp_R: Sequence[Sequence[int]] | npt.ArrayLike,
    dims: Sequence[int],
) -> int:
    active = active_dimensions(dims)
    if not active:
        return 1
    arr = np.asarray(ramp_R, dtype=np.uint16)
    sub = arr[np.ix_(active, active)]
    if not np.any(sub):
        return 1
    return determinant_mod2(sub)


@dataclass(frozen=True, slots=True)
class ProfileTerm:
    """One term in the v2 hybrid profile lattice."""

    profile_id: int | Profile
    coefficient_q16: int = 1 << 16
    rank: int = 0
    flags: int = 0
    metadata: Mapping[str, int | str | bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", int(self.profile_id))
        if int(self.rank) < 0:
            raise SPFError("profile term rank must be non-negative")
        if int(self.profile_id) == PROFILE_CP_CLOSED:
            raise SPFError("CP is not a closed contraction profile in SPFTensor v2")
        known = {int(p) for p in Profile}
        profile_id = int(self.profile_id)
        reserved_tt = PROFILE_TT_RANK_MIN <= profile_id <= PROFILE_TT_RANK_MAX
        reserved_v3 = PROFILE_V3_NONABELIAN_MIN <= profile_id <= PROFILE_V3_NONABELIAN_MAX
        if profile_id not in known and not reserved_tt and not reserved_v3:
            raise SPFError(f"unsupported v2 profile id 0x{profile_id:04x}")

    @property
    def is_tt_reserved(self) -> bool:
        return PROFILE_TT_RANK_MIN <= int(self.profile_id) <= PROFILE_TT_RANK_MAX

    @property
    def is_v3_reserved(self) -> bool:
        return PROFILE_V3_NONABELIAN_MIN <= int(self.profile_id) <= PROFILE_V3_NONABELIAN_MAX

    def to_metadata(self) -> dict[str, Any]:
        self.__post_init__()
        return {
            "profile_id": int(self.profile_id),
            "coefficient_q16": int(self.coefficient_q16),
            "rank": int(self.rank),
            "flags": int(self.flags),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_metadata(cls, data: Mapping[str, object]) -> ProfileTerm:
        return cls(
            profile_id=int(data["profile_id"]),
            coefficient_q16=int(data.get("coefficient_q16", 1 << 16)),
            rank=int(data.get("rank", 0)),
            flags=int(data.get("flags", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(frozen=True, slots=True)
class SPFScalar:
    """One encoded SPF scalar word with format metadata."""

    word: int = int(SPF32_WORD_ZERO)
    scalar_format_id: ScalarFormat = ScalarFormat.SPF32

    def __post_init__(self) -> None:
        fmt_id = ScalarFormat(self.scalar_format_id)
        object.__setattr__(self, "scalar_format_id", fmt_id)
        # Mask to this format's full word width — SPF-64/128 words are wider than
        # 32 bits, so the legacy ``& 0xFFFFFFFF`` would have truncated them.
        fmt = format_for(fmt_id)
        object.__setattr__(self, "word", int(self.word) & ((1 << fmt.total_bits) - 1))

    @classmethod
    def from_complex(cls, value: complex, scalar_format_id: ScalarFormat = ScalarFormat.SPF32) -> SPFScalar:
        fmt = format_for(scalar_format_id)
        word = int(encode_words(np.asarray([value], dtype=np.complex128), fmt)[0])
        return cls(word=word, scalar_format_id=scalar_format_id)

    def to_complex(self) -> complex:
        fmt = format_for(self.scalar_format_id)
        return complex(decode_words(np.asarray([self.word], dtype=fmt.word_dtype), fmt)[0])


@dataclass(slots=True)
class TensorBlockHeaderV2:
    """SPFTensor v2 TensorBlock header.

    ``ramp_R`` and ``ramp_b`` are stored as unsigned 16-bit residues. A zero
    active submatrix is treated as "no linear v2 ramp"; otherwise the active
    submatrix must be invertible modulo 2.
    """

    dims: tuple[int, ...] = (1,)
    version: int = SPFTENSOR_MAJOR
    order: int | None = None
    scalar_format_id: ScalarFormat = ScalarFormat.SPF32
    global_L_bias: int = 0
    global_T_bias: int = 0
    coarse_exp: int = 0
    band_exp: tuple[int, ...] | None = None
    origin: tuple[int, ...] | None = None
    ramp_alpha_v1: tuple[int, ...] | None = None
    ramp_beta_v1: tuple[int, ...] | None = None
    ramp_R: tuple[tuple[int, ...], ...] | npt.ArrayLike | None = None
    ramp_b: tuple[int, ...] | None = None
    quadratic_ramp: tuple[int, ...] | None = None
    cubic_ramp: tuple[int, ...] | None = None
    # Full-Hessian phase jet (design §12, added 2026-09-17, ABI v22).
    # ``phase_gradient[d]`` is a per-dimension linear phase term with no
    # invertibility constraint (``ramp_R`` keeps its mod-2 lattice rule);
    # ``mixed_ramp`` holds the off-diagonal Hessian terms, see ``mixed_ramp_pairs``.
    phase_gradient: tuple[int, ...] | None = None
    mixed_ramp: tuple[int, ...] | None = None
    # §13.6 mixed cubic phase terms, 2-D only (added 2026-09-18, ABI v24).
    # ``cubic_mixed[0]`` multiplies ``q0^2 q1`` and ``cubic_mixed[1]``
    # ``q0 q1^2``; see :func:`cubic_mixed_terms`. Both default to zero, so a
    # header that does not opt in is byte-identical to an ABI v23 one — that is
    # what answers design §12.6's warning against an "always-on incomplete cubic
    # header": the incomplete axis-only cubic is no longer the only cubic on
    # offer, and neither cubic costs anything until a producer sets it.
    cubic_mixed: tuple[int, ...] | None = None
    # Quadratic log-magnitude jet (design §17.4 / §27.1, added 2026-09-17, ABI
    # v23). Signed int16 L ticks: ``log_quadratic[d]`` multiplies ``q_d²`` and
    # ``log_mixed`` holds the off-diagonal pairs in ``mixed_ramp_pairs`` order.
    # Gaussian blur and splat opacity are *negative* quadratic in log magnitude.
    log_quadratic: tuple[int, ...] | None = None
    log_mixed: tuple[int, ...] | None = None
    profile_terms: tuple[ProfileTerm, ...] | None = None
    q_dL: int = 4
    q_dTheta: int = 4
    precision_mode: PrecisionMode = PrecisionMode.BALANCED
    active_encoding: ActiveEncoding = ActiveEncoding.DENSE_ALL
    payload_offsets: Mapping[str, int] = field(default_factory=dict)
    payload_bytes: int = 0
    outlier_count: int = 0
    reserved: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.dims = tuple(int(d) for d in self.dims)
        if self.order is None:
            self.order = len(self.dims)
        self.order = int(self.order)
        self.scalar_format_id = ScalarFormat(self.scalar_format_id)
        self.band_exp = _normalize_i16_vector(self.band_exp, self.order, name="band_exp")
        self.origin = _as_int_tuple(self.origin, self.order, 0)
        self.ramp_alpha_v1 = _as_int_tuple(self.ramp_alpha_v1, self.order, 0)
        self.ramp_beta_v1 = _as_int_tuple(self.ramp_beta_v1, self.order, 0)
        self.ramp_R = _normalize_u16_matrix(self.ramp_R, self.order)
        self.ramp_b = _normalize_u16_vector(self.ramp_b, self.order)
        self.quadratic_ramp = _normalize_u16_vector(self.quadratic_ramp, self.order)
        self.cubic_ramp = _normalize_u16_vector(self.cubic_ramp, self.order)
        self.phase_gradient = _normalize_u16_vector(self.phase_gradient, self.order)
        self.mixed_ramp = _normalize_u16_vector(self.mixed_ramp, len(mixed_ramp_pairs(self.order)))
        # Storage is a fixed 2 words at every order; only order 2 may use them.
        self.cubic_mixed = _normalize_u16_vector(self.cubic_mixed, SPFTENSOR_CUBIC_MIXED_MAX)
        self.log_quadratic = _normalize_i16_vector(self.log_quadratic, self.order, name="log_quadratic")
        self.log_mixed = _normalize_i16_vector(
            self.log_mixed, len(mixed_ramp_pairs(self.order)), name="log_mixed"
        )
        if not self.profile_terms:
            self.profile_terms = (ProfileTerm(Profile.STB_AFFINE),)
        else:
            self.profile_terms = tuple(
                term if isinstance(term, ProfileTerm) else ProfileTerm.from_metadata(term)
                for term in self.profile_terms
            )
        self.precision_mode = PrecisionMode(self.precision_mode)
        self.active_encoding = ActiveEncoding(self.active_encoding)
        self.payload_offsets = {str(k): int(v) for k, v in dict(self.payload_offsets).items()}
        reserved = tuple(int(v) for v in (self.reserved or ()))
        if len(reserved) > SPFTENSOR_RESERVED_WORDS and not any(reserved[SPFTENSOR_RESERVED_WORDS:]):
            # Metadata written before ABI v22 carried 18 reserved words, v22
            # carried 13, v23 carried 8; the carved-out tail was always zero, so
            # it is dropped losslessly.
            reserved = reserved[:SPFTENSOR_RESERVED_WORDS]
        self.reserved = _as_int_tuple(reserved, SPFTENSOR_RESERVED_WORDS, 0)
        self.validate()

    @property
    def total_elements(self) -> int:
        if not self.dims:
            return 1
        return int(np.prod(self.dims, dtype=np.int64))

    @property
    def active_dims(self) -> tuple[int, ...]:
        return active_dimensions(self.dims)

    def validate(self, *, containing_bytes: int | None = None) -> None:
        if int(self.version) != SPFTENSOR_MAJOR:
            raise SPFError("TensorBlockHeaderV2 requires version=2")
        if self.order != len(self.dims):
            raise SPFError("TensorBlockHeaderV2 order must match dims length")
        if self.order < 0 or self.order > SPFTENSOR_MAX_ORDER:
            raise SPFError("TensorBlockHeaderV2 supports order 0..4")
        if any(int(d) <= 0 for d in self.dims):
            raise SPFError("TensorBlockHeaderV2 dims must be positive")
        _fmt = format_for(self.scalar_format_id)
        if not _fmt.l_code_min_finite <= int(self.global_L_bias) <= _fmt.l_code_max_finite:
            raise SPFError("global_L_bias out of scalar-format finite range")
        if not _fmt.l_code_min_finite <= int(self.coarse_exp) <= _fmt.l_code_max_finite:
            raise SPFError("coarse_exp out of scalar-format finite range")
        if int(self.q_dL) <= 0 or int(self.q_dTheta) <= 0:
            raise SPFError("q_dL and q_dTheta must be positive")
        if ramp_determinant_mod2(self.ramp_R, self.dims) == 0:
            raise SPFError("active ramp_R must be invertible modulo 2")
        if not cubic_mixed_terms(self.order) and any(self.cubic_mixed or ()):
            raise SPFError(
                "cubic_mixed carries the AlgRef v0.2 §13.6 two-dimensional mixed "
                f"cubics (x^2 y, x y^2); order {self.order} has none, so both "
                "words must be zero"
            )
        if any(int(v) < 0 for v in self.payload_offsets.values()):
            raise SPFError("payload offsets must be non-negative")
        if int(self.payload_bytes) < 0 or int(self.outlier_count) < 0:
            raise SPFError("payload sizes/counts must be non-negative")
        if containing_bytes is not None:
            for name, offset in self.payload_offsets.items():
                if int(offset) > containing_bytes:
                    raise SPFError(f"payload offset {name} lies outside containing block")
            if int(self.payload_bytes) > containing_bytes:
                raise SPFError("payload_bytes exceeds containing block")

    def ramp_phase(self, logical_index: Sequence[int]) -> int:
        q = tuple(int(v) for v in logical_index)
        if len(q) != self.order:
            raise SPFError("logical index rank mismatch")
        theta = 0
        for row in range(self.order):
            row_theta = int(self.ramp_b[row])
            for col in range(self.order):
                row_theta += int(self.ramp_R[row][col]) * q[col]
            theta += row_theta
        for dim, coord in enumerate(q):
            theta += int(self.phase_gradient[dim]) * coord
            theta += int(self.quadratic_ramp[dim]) * coord * coord
            theta += int(self.cubic_ramp[dim]) * coord * coord * coord
        for k, (i, j) in enumerate(mixed_ramp_pairs(self.order)):
            theta += int(self.mixed_ramp[k]) * q[i] * q[j]
        for k, (px, py) in enumerate(cubic_mixed_terms(self.order)):
            theta += int(self.cubic_mixed[k]) * (q[0] ** px) * (q[1] ** py)
        return theta & SPF32_PHASE_MASK

    @property
    def has_nonlinear_phase(self) -> bool:
        """True when any quadratic, mixed, or cubic phase term is non-zero."""

        return bool(
            any(self.quadratic_ramp)
            or any(self.cubic_ramp)
            or any(self.mixed_ramp)
            or any(self.cubic_mixed)
        )

    def ramp_log(self, logical_index: Sequence[int]) -> int:
        """Log-magnitude increment (L ticks) of the header polynomial at ``logical_index``.

        ``band_exp·q + Σ log_quadratic[d]·q_d² + Σ log_mixed[k]·q_i·q_j``; the
        ``global_L_bias`` / ``coarse_exp`` constants are added by the block, and
        saturation to the finite L range happens there too.
        """

        q = tuple(int(v) for v in logical_index)
        if len(q) != self.order:
            raise SPFError("logical index rank mismatch")
        l_val = 0
        for dim, coord in enumerate(q):
            l_val += int(self.band_exp[dim]) * coord
            l_val += int(self.log_quadratic[dim]) * coord * coord
        for k, (i, j) in enumerate(mixed_ramp_pairs(self.order)):
            l_val += int(self.log_mixed[k]) * q[i] * q[j]
        return l_val

    @property
    def has_nonlinear_log(self) -> bool:
        """True when any quadratic or mixed log-magnitude term is non-zero."""

        return bool(any(self.log_quadratic) or any(self.log_mixed))

    def to_metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": int(self.version),
            "order": int(self.order),
            "dims": list(self.dims),
            "scalar_format_id": int(self.scalar_format_id),
            "global_L_bias": int(self.global_L_bias),
            "global_T_bias": int(self.global_T_bias),
            "coarse_exp": int(self.coarse_exp),
            "band_exp": list(self.band_exp),
            "origin": list(self.origin),
            "ramp_alpha_v1": list(self.ramp_alpha_v1),
            "ramp_beta_v1": list(self.ramp_beta_v1),
            "ramp_R": [list(row) for row in self.ramp_R],
            "ramp_b": list(self.ramp_b),
            "quadratic_ramp": list(self.quadratic_ramp),
            "cubic_ramp": list(self.cubic_ramp),
            "phase_gradient": list(self.phase_gradient),
            "mixed_ramp": list(self.mixed_ramp),
            "cubic_mixed": list(self.cubic_mixed),
            "log_quadratic": list(self.log_quadratic),
            "log_mixed": list(self.log_mixed),
            "profile_terms": [term.to_metadata() for term in self.profile_terms],
            "q_dL": int(self.q_dL),
            "q_dTheta": int(self.q_dTheta),
            "precision_mode": int(self.precision_mode),
            "active_encoding": int(self.active_encoding),
            "payload_offsets": dict(self.payload_offsets),
            "payload_bytes": int(self.payload_bytes),
            "outlier_count": int(self.outlier_count),
            "reserved": list(self.reserved),
        }

    @classmethod
    def from_metadata(cls, data: Mapping[str, object]) -> TensorBlockHeaderV2:
        version = int(data.get("version", SPFTENSOR_MAJOR))
        if version != SPFTENSOR_MAJOR:
            raise SPFError(f"unsupported TensorBlock major version {version}")
        ramp_R_raw = data.get("ramp_R")
        ramp_R = None if ramp_R_raw is None else tuple(tuple(int(x) for x in row) for row in ramp_R_raw)
        return cls(
            dims=tuple(int(v) for v in data.get("dims", (1,))),
            version=version,
            order=int(data.get("order", len(data.get("dims", (1,))))),
            scalar_format_id=ScalarFormat(int(data.get("scalar_format_id", ScalarFormat.SPF32))),
            global_L_bias=int(data.get("global_L_bias", 0)),
            global_T_bias=int(data.get("global_T_bias", 0)),
            coarse_exp=int(data.get("coarse_exp", 0)),
            band_exp=tuple(int(v) for v in data.get("band_exp", ())),
            origin=tuple(int(v) for v in data.get("origin", ())),
            ramp_alpha_v1=tuple(int(v) for v in data.get("ramp_alpha_v1", ())),
            ramp_beta_v1=tuple(int(v) for v in data.get("ramp_beta_v1", ())),
            ramp_R=ramp_R,
            ramp_b=tuple(int(v) for v in data.get("ramp_b", ())),
            quadratic_ramp=tuple(int(v) for v in data.get("quadratic_ramp", ())),
            cubic_ramp=tuple(int(v) for v in data.get("cubic_ramp", ())),
            phase_gradient=tuple(int(v) for v in data.get("phase_gradient", ())),
            mixed_ramp=tuple(int(v) for v in data.get("mixed_ramp", ())),
            cubic_mixed=tuple(int(v) for v in data.get("cubic_mixed", ())),
            log_quadratic=tuple(int(v) for v in data.get("log_quadratic", ())),
            log_mixed=tuple(int(v) for v in data.get("log_mixed", ())),
            profile_terms=tuple(
                ProfileTerm.from_metadata(term)
                for term in data.get("profile_terms", (ProfileTerm(Profile.STB_AFFINE).to_metadata(),))
            ),
            q_dL=int(data.get("q_dL", 4)),
            q_dTheta=int(data.get("q_dTheta", 4)),
            precision_mode=PrecisionMode(int(data.get("precision_mode", PrecisionMode.BALANCED))),
            active_encoding=ActiveEncoding(int(data.get("active_encoding", ActiveEncoding.DENSE_ALL))),
            payload_offsets=dict(data.get("payload_offsets", {})),
            payload_bytes=int(data.get("payload_bytes", 0)),
            outlier_count=int(data.get("outlier_count", 0)),
            reserved=tuple(int(v) for v in data.get("reserved", ())),
        )


@dataclass(slots=True)
class TensorBlock:
    """v2 TensorBlock payload with explicit active/residual streams."""

    header: TensorBlockHeaderV2
    dL: npt.ArrayLike | None = None
    dT: npt.ArrayLike | None = None
    active_indices: npt.ArrayLike | None = None
    bands: Sequence[BandRange] = field(default_factory=tuple)
    outliers: Mapping[int, int] = field(default_factory=dict)
    packed_residuals: bytes = b""

    def validate(self) -> None:
        self.header.validate()
        if self.header.scalar_format_id != ScalarFormat.SPF32:
            raise SPFError(
                "TensorBlock v2 execution requires SPF32; "
                "use an SPFBlockV3 with its explicit representation for other widths"
            )
        total = self.header.total_elements
        if self.header.active_encoding == ActiveEncoding.DENSE_ALL:
            active_count = total
        elif self.header.active_encoding == ActiveEncoding.COO_U32:
            if self.active_indices is None:
                raise SPFError("COO_U32 requires active_indices")
            idx = np.asarray(self.active_indices, dtype=np.int64)
            if np.any((idx < 0) | (idx >= total)):
                raise SPFError("active index out of bounds")
            if idx.size and not np.all(idx[1:] > idx[:-1]):
                raise SPFError("active indices must be strictly increasing")
            active_count = int(idx.size)
        else:
            raise NotImplementedError(f"{self.header.active_encoding.name} decode is not implemented yet")
        if self.dL is None or self.dT is None:
            raise SPFError("dL and dT residual streams are required")
        if len(np.asarray(self.dL)) != active_count or len(np.asarray(self.dT)) != active_count:
            raise SPFError("residual stream count mismatch")
        self._validate_bands()
        for linear in self.outliers:
            if int(linear) < 0 or int(linear) >= total:
                raise SPFError("outlier index out of bounds")

    def _validate_bands(self) -> None:
        by_dim: dict[int, list[BandRange]] = {}
        for band in self.bands:
            if band.dim >= self.header.order:
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
        self.validate()
        total = self.header.total_elements
        L = np.full(total, SPF32_L_ZERO, dtype=np.int16)
        T = np.zeros(total, dtype=np.uint16)
        if self.header.active_encoding == ActiveEncoding.DENSE_ALL:
            active = np.arange(total, dtype=np.int64)
        else:
            active = np.asarray(self.active_indices, dtype=np.int64)
        dL_arr = np.asarray(self.dL, dtype=np.int16)
        dT_arr = np.asarray(self.dT, dtype=np.int16)
        # Vectorized over all active indices: the per-element loop body is pure
        # integer arithmetic (origin-mod coords → affine/poly ramp → band deltas →
        # +dL/+dT → saturate/mask), so batching it is bit-identical to the scalar
        # form. ``ramp_phase`` masks per element; deferring the ``& MASK`` to the
        # end is mod-2^16 equivalent because every intermediate op is an integer add.
        order = int(self.header.order or 0)
        dims = self.header.dims
        Q = _coord_matrix(active, dims, self.header.origin, order)
        band_exp = np.asarray(self.header.band_exp, dtype=np.int64)
        R = np.asarray(self.header.ramp_R, dtype=np.int64).reshape(order, order)
        ramp_b = np.asarray(self.header.ramp_b, dtype=np.int64)
        quad = np.asarray(self.header.quadratic_ramp, dtype=np.int64)
        cubic = np.asarray(self.header.cubic_ramp, dtype=np.int64)
        grad = np.asarray(self.header.phase_gradient, dtype=np.int64)
        log_quad = np.asarray(self.header.log_quadratic, dtype=np.int64)
        l_val = (
            np.int64(self.header.global_L_bias)
            + np.int64(self.header.coarse_exp)
            + Q @ band_exp
            + (Q * Q) @ log_quad
            + _mixed_ramp_phase(Q, self.header.log_mixed, order)
        )
        t_val = (
            np.int64(self.header.global_T_bias)
            + ramp_b.sum()
            + Q @ (R.sum(axis=0) + grad)
            + (Q * Q) @ quad
            + (Q * Q * Q) @ cubic
            + _mixed_ramp_phase(Q, self.header.mixed_ramp, order)
            # AlgRef v0.2 §13.6 mixed cubics, 2-D only (ABI v24).
            + _cubic_mixed_phase(Q, self.header.cubic_mixed, order)
        )
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
        # instead of clipping it to a tiny finite L. Matches the native C
        # spf32_tb_v2_affine_reconstruct guard and AffineTensorBlock.reconstruct.
        zero = l_val == int(SPF32_L_ZERO)
        L[active] = np.where(zero, SPF32_L_ZERO, _sat_l16(l_val)).astype(np.int16)
        T[active] = np.where(zero, 0, t_val & SPF32_PHASE_MASK).astype(np.uint16)
        if self.outliers:
            words = pack_words(L, T)
            for linear, word in self.outliers.items():
                words[int(linear)] = np.uint32(int(word) & 0xFFFFFFFF)
            L = unpack_L(words)
            T = unpack_theta(words)
        return L.reshape(self.header.dims), T.reshape(self.header.dims)

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

    @classmethod
    def from_affine(cls, block: AffineTensorBlock) -> TensorBlock:
        return migrate_affine_to_v2(block)

    def to_words(self) -> np.ndarray:
        L, T = self.reconstruct()
        return pack_words(L, T)

    def sample_words(self, linear: npt.ArrayLike, *, validate: bool = True, phase: bool = True) -> np.ndarray:
        """Read addressed samples from the header and residual streams directly.

        No full coordinate, L/T, or word plane is constructed. ``validate=False``
        is reserved for owners that validated and froze the payload at import.
        """
        if validate:
            self.validate()
        query = np.asarray(linear, dtype=np.int64)
        flat = query.reshape(-1)
        if np.any(flat < 0) or np.any(flat >= self.header.total_elements):
            raise ValueError("TensorBlock sample outside extent")
        result = np.full(flat.size, SPF32_WORD_ZERO, dtype=np.uint32)
        if self.header.active_encoding == ActiveEncoding.DENSE_ALL:
            live = np.ones(flat.size, dtype=bool)
            positions = flat
        else:
            active = np.asarray(self.active_indices)
            positions = np.searchsorted(active, flat)
            live = positions < active.size
            live[live] &= active[positions[live]] == flat[live]
        h = self.header
        order = int(h.order or 0)
        Q = _coord_matrix(flat[live], h.dims, h.origin, order)
        L = (
            np.int64(h.global_L_bias)
            + np.int64(h.coarse_exp)
            + Q @ np.asarray(h.band_exp, dtype=np.int64)
            + (Q * Q) @ np.asarray(h.log_quadratic, dtype=np.int64)
            + _mixed_ramp_phase(Q, h.log_mixed, order)
        )
        T = np.zeros(L.shape, dtype=np.int64)
        if phase:
            R = np.asarray(h.ramp_R, dtype=np.int64).reshape(order, order)
            T = (
                np.int64(h.global_T_bias)
                + np.asarray(h.ramp_b, dtype=np.int64).sum()
                + Q @ (R.sum(axis=0) + np.asarray(h.phase_gradient, dtype=np.int64))
                + (Q * Q) @ np.asarray(h.quadratic_ramp, dtype=np.int64)
                + (Q * Q * Q) @ np.asarray(h.cubic_ramp, dtype=np.int64)
                + _mixed_ramp_phase(Q, h.mixed_ramp, order)
                # AlgRef v0.2 §13.6 mixed cubics, 2-D only (ABI v24).
                + _cubic_mixed_phase(Q, h.cubic_mixed, order)
            )
        for band in self.bands:
            covered = (Q[:, band.dim] >= band.start) & (Q[:, band.dim] < band.start + band.length)
            if band.effective_flags & 0x01:
                L[covered] += int(band.L_delta)
            if phase and band.effective_flags & 0x02:
                T[covered] += int(band.T_delta)
        L += np.asarray(self.dL, dtype=np.int16)[positions[live]].astype(np.int64)
        if phase:
            T += np.asarray(self.dT, dtype=np.int16)[positions[live]].astype(np.int64)
        zero = int(SPF32_L_ZERO) == L
        result[live] = pack_words(
            np.where(zero, SPF32_L_ZERO, _sat_l16(L)).astype(np.int16),
            np.where(zero, 0, T & SPF32_PHASE_MASK).astype(np.uint16),
        )
        apply_outliers(result, flat, self.outliers)
        return result.reshape(query.shape)

    def metadata(self) -> BlockMetadata:
        """Cheap log-domain per-block screening metadata (SPF-tensor Stage 4)."""

        return block_metadata(self)


def apply_outliers(result: np.ndarray, linear: np.ndarray, outliers: Any) -> None:
    """Join addressed samples to absolute overrides in O(Q log M), including repeats."""
    if not outliers:
        return
    addresses = np.asarray(sorted(outliers), dtype=np.int64)
    positions = np.searchsorted(addresses, linear)
    selected = positions < addresses.size
    selected[selected] &= addresses[positions[selected]] == linear[selected]
    words = np.asarray([outliers[int(i)] for i in addresses], dtype=result.dtype)
    result[selected] = words[positions[selected]]


@dataclass(frozen=True, slots=True)
class BlockMetadata:
    """Cheap log-domain per-block screening metadata (SPF-tensor Stage 4).

    Every field is an O(N) single pass over the decoded ``(L, Θ)`` planes —
    free in log-polar form, costly to recover in Cartesian. Grounded in
    tropical-norm intuition (``dynamic_range`` = max−min L is a tropical norm,
    ``dominant_index`` = argmax L is the tropical projection) and in circular
    statistics (``phase_coherence`` = mean resultant length; Fisher, Mardia–Jupp).
    Inactive elements (zero sentinel ``L == SPF32_L_ZERO``) are excluded.
    """

    finite_count: int
    dynamic_range: float
    dominant_index: int
    phase_coherence: float
    # Stage-5 routing inputs (appended; default-valued so older constructors and
    # serialized metadata stay valid). ``dominant_gap`` = max(L) − 2nd-max(L) in
    # log2 bits (the worst add's q); ``pivot_L`` = max(L) in log2 bits.
    # ``dominant_gap`` defaults to 0.0 (the *conservative* value: an absent gap
    # suppresses the lossy TROPICAL shortcut, so a partially-built metadata falls
    # to the accurate LOGPOLAR/CARTESIAN domains — never the reverse). Real values
    # come from ``block_metadata_from_planes``; ``reducers.from_block_metadata``
    # maps ``finite_count < 2`` to ``+inf`` before routing.
    dominant_gap: float = 0.0
    pivot_L: float = 0.0

    def to_metadata(self) -> dict[str, float | int]:
        return {
            "finite_count": int(self.finite_count),
            "dynamic_range": float(self.dynamic_range),
            "dominant_index": int(self.dominant_index),
            "phase_coherence": float(self.phase_coherence),
            "dominant_gap": float(self.dominant_gap),
            "pivot_L": float(self.pivot_L),
        }


def block_dynamic_range(L: npt.ArrayLike) -> float:
    """Per-block log-magnitude dynamic range ``max(L) − min(L)`` in log2 units.

    A tropical norm — free in log-polar form. Computed over finite elements
    only (zero sentinel excluded). Returns ``0.0`` for fewer than two finite
    elements.
    """

    L_arr = np.asarray(L, dtype=np.int16).reshape(-1)
    finite = L_arr != SPF32_L_ZERO
    if int(np.count_nonzero(finite)) < 2:
        return 0.0
    vals = L_arr[finite].astype(np.float64) / SPF32_L_SCALE
    return float(vals.max() - vals.min())


def block_dominant_gap(L: npt.ArrayLike) -> float:
    """Gap ``max(L) − 2nd-max(L)`` over finite elements, in log2 units.

    The q of the worst add — large means one term dominates (tropical-exact).
    Returns ``0.0`` for fewer than two finite elements (no second term), and
    ``0.0`` when the two largest magnitudes tie (no single dominant term).
    """

    L_arr = np.asarray(L, dtype=np.int16).reshape(-1)
    finite = L_arr != SPF32_L_ZERO
    if int(np.count_nonzero(finite)) < 2:
        return 0.0
    vals = L_arr[finite].astype(np.float64) / SPF32_L_SCALE
    top2 = np.partition(vals, -2)[-2:]
    return float(top2[1] - top2[0])


def block_dominant_index(L: npt.ArrayLike) -> int:
    """Flat index of the dominant (largest-magnitude) finite element.

    The tropical projection ``argmax(L)``. Ties resolve to the lowest flat
    index. Returns ``-1`` when no finite element exists.
    """

    L_arr = np.asarray(L, dtype=np.int16).reshape(-1)
    finite = L_arr != SPF32_L_ZERO
    if not np.any(finite):
        return -1
    masked = np.where(finite, L_arr.astype(np.int64), np.int64(SPF32_L_MIN_FINITE) - 1)
    return int(np.argmax(masked))


def block_phase_coherence(
    theta: npt.ArrayLike,
    *,
    finite_mask: npt.ArrayLike | None = None,
) -> float:
    """Mean resultant length ``R̄ = |Σ e^{iθ}| / N`` of the phase field, in ``[0, 1]``.

    Circular-statistics phase concentration (Fisher; Mardia–Jupp). ``R̄ → 1``
    means tightly clustered phases, ``R̄ → 0`` diffuse. Computed over the
    elements selected by ``finite_mask`` (defaults to all). Returns ``0.0`` for
    an empty selection.
    """

    T_arr = np.asarray(theta, dtype=np.uint16).reshape(-1)
    if finite_mask is None:
        sel = T_arr
    else:
        mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
        if mask.shape != T_arr.shape:
            raise SPFError("finite_mask must match the phase plane size")
        sel = T_arr[mask]
    n = int(sel.size)
    if n == 0:
        return 0.0
    angle = sel.astype(np.float64) * (2.0 * np.pi / SPF32_PHASE_MOD)
    resultant = np.sum(np.exp(1j * angle))
    return float(np.abs(resultant) / n)


def block_metadata_from_planes(L: npt.ArrayLike, theta: npt.ArrayLike) -> BlockMetadata:
    """Compute :class:`BlockMetadata` directly from decoded ``(L, Θ)`` planes."""

    L_arr = np.asarray(L, dtype=np.int16).reshape(-1)
    T_arr = np.asarray(theta, dtype=np.uint16).reshape(-1)
    if L_arr.shape != T_arr.shape:
        raise SPFError("L and theta planes must have matching size")
    finite = L_arr != SPF32_L_ZERO
    pivot_L = float(L_arr[finite].astype(np.float64).max() / SPF32_L_SCALE) if np.any(finite) else 0.0
    return BlockMetadata(
        finite_count=int(np.count_nonzero(finite)),
        dynamic_range=block_dynamic_range(L_arr),
        dominant_index=block_dominant_index(L_arr),
        phase_coherence=block_phase_coherence(T_arr, finite_mask=finite),
        dominant_gap=block_dominant_gap(L_arr),
        pivot_L=pivot_L,
    )


def block_metadata(block: TensorBlock) -> BlockMetadata:
    """Reconstruct ``block`` and compute its :class:`BlockMetadata`."""

    L, T = block.reconstruct()
    return block_metadata_from_planes(L, T)


def cancellation_risk(
    meta: BlockMetadata,
    *,
    max_coherence: float = BLOCK_CANCEL_COHERENCE_MAX,
    dynamic_range_threshold: float = BLOCK_CANCEL_DL_THRESHOLD,
) -> bool:
    """Heuristic screen: is this block's SPF reduction prone to cancellation?

    Flags near-equal magnitude (``dynamic_range < dynamic_range_threshold``, the
    spec wedge bound ``|ΔL| < 1.0``) combined with *low* phase coherence
    (``R̄ ≤ max_coherence``) — spread / near-antipodal phases, so the
    magnitude-weighted resultant ``|Σ z|`` collapses far below ``Σ|z|`` and the
    running sum loses significance. Within the near-equal-magnitude regime the
    screen gates on, ``R̄`` tracks that magnitude-weighted resultant, and the
    condition is the block-scale analogue of the spec cancellation wedge's
    opposite-phase requirement ``|ΔΘ − π| < 0.5``
    (docs/spf-formats-and-test-plans.md §T1.6/§T1.7). It routes a block to a
    cotransformation reducer (SPF-tensor Stage 4).

    Note: this intentionally inverts the source roadmap note, which framed *high*
    coherence as the risk. High coherence is aligned/reinforcing phases — the
    numerically easy case; cancellation needs phases that oppose, i.e. low ``R̄``.
    An O(N) screen, not a proof; certainty requires inspecting operand pairs.
    """

    if meta.finite_count < 2:
        return False
    return bool(meta.phase_coherence <= float(max_coherence) and meta.dynamic_range < float(dynamic_range_threshold))


@dataclass(slots=True)
class SPFBlock:
    """Container for one or more v2 TensorBlocks."""

    tensors: tuple[TensorBlock, ...]
    scalar_format_id: ScalarFormat = ScalarFormat.SPF32

    def __post_init__(self) -> None:
        self.scalar_format_id = ScalarFormat(self.scalar_format_id)
        self.tensors = tuple(self.tensors)
        for tensor in self.tensors:
            tensor.validate()


def migrate_affine_to_v2(block: AffineTensorBlock) -> TensorBlock:
    """Import the v0.1 affine TensorBlock into a v2-compatible TensorBlock."""

    block.validate()
    order = block.header.ndims
    legacy_ramp = tuple((block.header.ramp_alpha or (0, 0, 0, 0))[:order])
    direct_ramp_R = [[0 for _ in range(order)] for _ in range(order)]
    for dim in range(order):
        direct_ramp_R[dim][dim] = _u16(legacy_ramp[dim])
    if ramp_determinant_mod2(direct_ramp_R, block.header.dims) == 1:
        ramp_R = tuple(tuple(row) for row in direct_ramp_R)
    else:
        ramp_R = _identity_ramp_R(block.header.dims)
    ramp_b = [0 for _ in range(order)]
    if order:
        ramp_b[0] = _u16(block.header.global_T_bias)
    header = TensorBlockHeaderV2(
        dims=block.header.dims,
        order=order,
        scalar_format_id=block.header.scalar_format_id,
        global_L_bias=block.header.global_L_bias,
        global_T_bias=0,
        origin=tuple((block.header.origin or (0, 0, 0, 0))[:order]),
        ramp_alpha_v1=legacy_ramp,
        ramp_beta_v1=tuple(ramp_b),
        ramp_R=ramp_R,
        ramp_b=tuple(ramp_b),
        profile_terms=(ProfileTerm(block.header.profile_id),),
        q_dL=4,
        q_dTheta=4,
        precision_mode=PrecisionMode.BALANCED,
        active_encoding=block.active_encoding,
    )
    active_indices = None if block.active_indices is None else np.asarray(block.active_indices, dtype=np.uint32).copy()
    dT = np.asarray(block.dT, dtype=np.int16).copy()
    if ramp_R != tuple(tuple(row) for row in direct_ramp_R):
        if block.active_encoding == ActiveEncoding.DENSE_ALL:
            active = np.arange(block.header.total_elements, dtype=np.int64)
        else:
            active = np.asarray(block.active_indices, dtype=np.int64)
        origin = tuple((block.header.origin or (0, 0, 0, 0))[:order])
        delta_theta = np.empty(dT.shape, dtype=np.uint16)
        for r, linear in enumerate(active):
            logical = np.unravel_index(int(linear), block.header.dims)
            q = tuple(
                (int(origin[dim]) + int(coord)) % int(block.header.dims[dim]) for dim, coord in enumerate(logical)
            )
            legacy_theta = int(block.header.global_T_bias)
            legacy_theta += sum(int(legacy_ramp[dim]) * q[dim] for dim in range(order))
            v2_theta = header.ramp_phase(q)
            delta = (legacy_theta - v2_theta) & SPF32_PHASE_MASK
            delta_theta[r] = np.uint16(delta)
        dT = (
            ((dT.view(np.uint16).astype(np.uint32) + delta_theta.astype(np.uint32)) & SPF32_PHASE_MASK)
            .astype(np.uint16)
            .view(np.int16)
        )
    return TensorBlock(
        header=header,
        dL=np.asarray(block.dL, dtype=np.int16).copy(),
        dT=dT,
        active_indices=active_indices,
        bands=tuple(block.bands),
    )


@dataclass(frozen=True, slots=True)
class ExecutionStep:
    op: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    reducer: ReducerSpec | None = None
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate against the live registry, not a frozen snapshot, so an op
        # registered via ``op_registry.register_op`` is plan-legal immediately.
        spec = OP_REGISTRY.get(self.op)
        if spec is None:
            raise SPFError(f"unsupported execution op {self.op!r}")
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if self.reducer is not None and not isinstance(self.reducer, ReducerSpec):
            object.__setattr__(self, "reducer", ReducerSpec(policy=ReducerPolicy(self.reducer)))
        if spec.needs_reducer:
            if self.reducer is None:
                raise SPFError(f"{self.op} requires a ReducerSpec boundary")
            self.reducer.require_contraction()


@dataclass(slots=True)
class ExecutionPlan:
    """Executable v2 plan with explicit reducer and requantization boundaries."""

    steps: tuple[ExecutionStep, ...] = field(default_factory=tuple)
    factorization_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.steps = tuple(step if isinstance(step, ExecutionStep) else ExecutionStep(**step) for step in self.steps)
        self.factorization_metadata = dict(self.factorization_metadata)

    @property
    def backend(self) -> str | None:
        """Backend label recorded by the lowering that produced this plan."""

        value = self.factorization_metadata.get("backend")
        return None if value is None else str(value)

    @property
    def reference_only(self) -> bool:
        """Whether this plan contains or explicitly declares oracle-only work."""

        if bool(self.factorization_metadata.get("reference_only", False)):
            return True
        return any((spec := OP_REGISTRY.get(step.op)) is not None and spec.reference_only for step in self.steps)

    @property
    def deployable(self) -> bool:
        """Whether the plan is eligible for a native backend deployment."""

        return bool(self.factorization_metadata.get("deployable", not self.reference_only)) and not self.reference_only

    @property
    def capability(self) -> str | None:
        """Primary lowering capability label, when one was recorded."""

        value = self.factorization_metadata.get("capability")
        return None if value is None else str(value)

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Stable capability labels attached to the lowering."""

        raw = self.factorization_metadata.get("capabilities", ())
        if isinstance(raw, str):
            return (raw,)
        return tuple(str(value) for value in raw)

    @property
    def reducer_boundaries(self) -> tuple[int, ...]:
        return tuple(i for i, step in enumerate(self.steps) if step.reducer is not None)

    @property
    def structured_reducer_boundaries(self) -> tuple[int, ...]:
        """Indices of reference structured nodes that perform a reduction."""

        return tuple(int(index) for index in self.factorization_metadata.get("structured_reducer_boundaries", ()))

    @property
    def lifted_contractions(self) -> tuple[int, ...]:
        lifted = lifted_ops()
        return tuple(i for i, step in enumerate(self.steps) if step.op in lifted)

    @property
    def requantization_points(self) -> tuple[int, ...]:
        return tuple(
            i for i, step in enumerate(self.steps)
            if step.op in {"fft", "ifft"} or bool(step.params.get("requantize", step.reducer is not None))
        )

    @property
    def structured_requantization_points(self) -> tuple[int, ...]:
        """Indices of structured nodes that materialize or requantize values."""

        return tuple(int(index) for index in self.factorization_metadata.get("structured_requantization_points", ()))

    def execute(self, values: Mapping[str, Any]) -> dict[str, Any]:
        env: dict[str, Any] = dict(values)
        for step in self.steps:
            if step.op == "permute":
                env[step.outputs[0]] = np.transpose(env[step.inputs[0]], axes=tuple(step.params["axes"]))
            elif step.op == "gather":
                env[step.outputs[0]] = np.take(
                    env[step.inputs[0]],
                    np.asarray(step.params["indices"], dtype=np.int64),
                    axis=int(step.params.get("axis", 0)),
                )
            elif step.op in {"diagonal_scale", "elementwise_mul"}:
                env[step.outputs[0]] = spf_mul_words(env[step.inputs[0]], env[step.inputs[1]])
            elif step.op == "ramp":
                env[step.outputs[0]] = self._execute_ramp(env[step.inputs[0]], step)
            elif step.op in {"fft", "ifft"}:
                # The default remains the legacy FB7 ABI.  Plans can opt into
                # the reference FB3-Hybrid schedule with params["profile"].
                from spf.fft import spf32_fft

                fn = spf32_fft.fft if step.op == "fft" else spf32_fft.ifft
                input_words = np.asarray(env[step.inputs[0]], dtype=np.uint32)
                profile = step.params.get("profile")
                precision = step.params.get("precision", "balanced")
                if profile is None and precision == "balanced":
                    env[step.outputs[0]] = fn(input_words)
                else:
                    env[step.outputs[0]] = fn(input_words, profile=profile, precision=precision)
            elif step.op == "reduce":
                env[step.outputs[0]] = self._execute_reduce(env[step.inputs[0]], step)
            elif step.op == "low_rank_apply":
                env[step.outputs[0]] = self._execute_low_rank(env, step)
            elif step.op == "block_circulant_apply":
                env[step.outputs[0]] = self._execute_block_circulant_apply(env, step)
            elif step.op == "decoded_sum_reference":
                env[step.outputs[0]] = self._execute_decoded_sum(env, step)
            elif step.op == "packed_sum":
                from .phase_clock import packed_pairwise
                if step.reducer is None or step.reducer.domain != AccumulationDomain.LOGPOLAR:
                    raise SPFError("packed_sum requires the declared LOGPOLAR reducer")
                env[step.outputs[0]] = packed_pairwise(np.stack([env[name] for name in step.inputs], axis=-1))
            elif step.op == "decoded_pushforward_reference":
                env[step.outputs[0]] = self._execute_decoded_pushforward(env, step)
            elif step.op == "decoded_contract_reference":
                env[step.outputs[0]] = self._execute_decoded_contract(env, step)
            elif step.op == "decoded_low_rank_scaled_reference":
                x, left, right, coeff = (decode_words(np.asarray(env[name], dtype=np.uint32)) for name in step.inputs)
                scales = np.asarray(step.params["coeff_log2_scale"], dtype=np.int32)
                coeff = np.ldexp(coeff.real, scales) + 1j * np.ldexp(coeff.imag, scales)
                env[step.outputs[0]] = encode_words(left @ (coeff @ (np.conj(right).T @ x)))
            elif step.op == "decoded_materialized_reference":
                env[step.outputs[0]] = np.asarray(env[step.inputs[0]], dtype=np.uint32).copy()
            elif step.op == "contract":
                env[step.outputs[0]] = self._execute_contract(env, step)
            elif step.op == "softmax_lse":
                env[step.outputs[0]] = self._execute_softmax_lse(env, step)
            elif step.op == "route":
                env[step.outputs[0]] = self._execute_route(env, step)
            elif step.op == "gauge_norm":
                env[step.outputs[0]] = self._execute_gauge_norm(env, step)
            elif step.op == "l_gate":
                env[step.outputs[0]] = self._execute_l_gate(env, step)
            elif step.op in {"phase_character_features", "phase_shear", "phase_neuron"}:
                if len(step.inputs) != 1 or len(step.outputs) != 1:
                    raise SPFError(f"{step.op} requires one input and one output")
                from .phase_circuits import (
                    execute_phase_character_features,
                    execute_phase_neuron,
                    execute_phase_shear,
                )

                params = dict(step.params)
                n_channels = params.pop("n_channels", None)
                params.pop("n_terms", None)
                words = np.asarray(env[step.inputs[0]])
                if n_channels is not None and (words.ndim == 0 or words.shape[-1] != n_channels):
                    raise SPFError("phase circuit input channel count differs from its plan")
                fn = {
                    "phase_character_features": execute_phase_character_features,
                    "phase_shear": execute_phase_shear,
                    "phase_neuron": execute_phase_neuron,
                }[step.op]
                env[step.outputs[0]] = fn(words, **params)
            elif step.op == "monomial_scan":
                from .monomial_scan import execute_monomial_scan_step

                env.update(execute_monomial_scan_step(env, step.inputs, step.outputs, step.params))
            elif step.op == "routed_buckets":
                from .routed_buckets import execute_routed_buckets_step

                env.update(execute_routed_buckets_step(env, step.inputs, step.outputs, step.params))
            elif step.op.startswith("v02_"):
                # Structured renderer operations are an explicit v0.2 seam.
                # They are reference-only and dispatched lazily so importing
                # the legacy tensor VM does not pull in renderer code or alter
                # any historical default operation.
                if len(step.outputs) != 1:
                    raise SPFError(f"{step.op} requires exactly one output")
                from spf.renderer.v02 import execute_op

                operands = tuple(env[name] for name in step.inputs)
                result = execute_op(step.op, operands, step.params)
                from spf.renderer import v02 as structured_renderer

                structured_renderer._validate_execution_output(step.op, result, step.params)
                env[step.outputs[0]] = result
            else:  # pragma: no cover - guarded by ExecutionStep validation.
                raise SPFError(f"unsupported execution op {step.op!r}")
        return env

    def _execute_ramp(self, words: npt.ArrayLike, step: ExecutionStep) -> np.ndarray:
        """Apply a phase ramp ``Θ += ramp(q)`` over the logical index grid.

        Reference for the ``spf_ramp`` PL kernel. Linear ``ramp_R``/``ramp_b``
        only by default (the kernel's v1 datapath); set ``params["nonlinear"]``
        to also fold the quadratic/cubic terms (the kernel's stubbed extension
        point). ``L`` is untouched and every non-finite word (zero / inf / NaN
        sentinel) passes through unchanged — bit-identical to the kernel's
        sentinel guard.
        """
        arr = np.asarray(words, dtype=np.uint32)
        ramp_R = step.params.get("ramp_R")
        ramp_b = step.params.get("ramp_b")
        if ramp_R is None or ramp_b is None:
            raise SPFError("ramp step requires params['ramp_R'] and params['ramp_b']")
        order = len(tuple(ramp_R))
        if arr.ndim != order:
            raise SPFError(f"ramp expects an order-{order} array, got ndim={arr.ndim}")
        inc = _ramp_theta_increment(
            arr.shape,
            ramp_R,
            ramp_b,
            step.params.get("quadratic_ramp"),
            step.params.get("cubic_ramp"),
            nonlinear=bool(step.params.get("nonlinear", False)),
            phase_gradient=step.params.get("phase_gradient"),
            mixed_ramp=step.params.get("mixed_ramp"),
            cubic_mixed=step.params.get("cubic_mixed"),
        )
        L = unpack_L(arr)
        T = unpack_theta(arr)
        T_new = ((T.astype(np.int64) + inc.astype(np.int64)) & SPF32_PHASE_MASK).astype(np.uint16)
        out = pack_words(L, T_new)
        sentinel = (L == SPF32_L_ZERO) | (L == SPF32_L_SENTINEL)
        return np.where(sentinel, arr, out).astype(np.uint32)

    def _execute_reduce(self, words: npt.ArrayLike, step: ExecutionStep) -> np.ndarray | np.uint32:
        assert step.reducer is not None
        from . import reducers

        arr = np.asarray(words, dtype=np.uint32)
        axis = step.params.get("axis")
        if step.reducer.policy is ReducerPolicy.V02:
            from .core import reduce_words_v02

            if axis is None:
                return reduce_words_v02(arr, spec=step.reducer)
            normalized_axis = int(axis) % arr.ndim
            moved = np.moveaxis(arr, normalized_axis, -1)
            reduced_shape = moved.shape[:-1]
            result = np.empty(reduced_shape, dtype=np.uint32)
            for index in np.ndindex(reduced_shape):
                result[index] = reduce_words_v02(moved[index], spec=step.reducer)
            return result
        return reducers.reduce_axis(arr, step.reducer, None if axis is None else int(axis))

    def _execute_low_rank(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        assert step.reducer is not None
        if "operator" in step.params:
            op = step.params["operator"]
            if not isinstance(op, LowRankOperator):
                raise SPFError("low_rank_apply operator must be LowRankOperator")
            op = replace(op, reducer_id=step.reducer)
        else:
            op = LowRankOperator(
                step.params["u_factors"],
                step.params["v_factors"],
                step.params["lambdas"],
                reducer_id=step.reducer,
            )
        return op.apply_words(env[step.inputs[0]])

    def _execute_block_circulant_apply(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        assert step.reducer is not None
        if len(step.inputs) != 2:
            raise SPFError("block_circulant_apply requires vector and spectrum inputs")
        frequency_count = int(step.params["frequency_count"])
        out_blocks = int(step.params["out_blocks"])
        in_blocks = int(step.params["in_blocks"])
        if frequency_count <= 0 or out_blocks <= 0 or in_blocks <= 0:
            raise SPFError("block_circulant_apply dimensions must be positive")
        vector = decode_words(np.asarray(env[step.inputs[0]], dtype=np.uint32))
        spectrum = decode_words(np.asarray(env[step.inputs[1]], dtype=np.uint32))
        if vector.size != frequency_count * in_blocks:
            raise SPFError("block_circulant_apply vector size mismatch")
        if spectrum.size != frequency_count * out_blocks * in_blocks:
            raise SPFError("block_circulant_apply spectrum size mismatch")
        x = vector.reshape(frequency_count, in_blocks)
        spec = spectrum.reshape(frequency_count, out_blocks, in_blocks)
        X = np.fft.fft(x, axis=0)
        Y = np.einsum("foi,fi->fo", spec, X)
        y = np.fft.ifft(Y, axis=0)
        return encode_words(y.reshape(frequency_count * out_blocks))

    def _execute_decoded_contract(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        # Intentional FP64 reference path. The ReducerSpec is required (and recorded
        # as a reducer/requantization boundary by reducer_boundaries /
        # requantization_points) so the plan declares where an SPF integer reducer
        # would sit, but this oracle deliberately decodes to complex128 and contracts
        # with np.tensordot — it does NOT dispatch the reducer policy/accumulator. It
        # is the correctness reference an optimized integer reducer is validated
        # against, not the optimized path itself.
        assert step.reducer is not None
        a = decode_words(np.asarray(env[step.inputs[0]], dtype=np.uint32))
        b = decode_words(np.asarray(env[step.inputs[1]], dtype=np.uint32))
        axes = step.params.get("axes", 1)
        batch_pairs = tuple(tuple(int(x) for x in pair) for pair in step.params.get("batch_axes", ()))
        if batch_pairs:
            if isinstance(axes, int):
                if int(axes) < 0:
                    raise SPFError("decoded_contract_reference axes must be non-negative")
                left_contract = tuple(range(a.ndim - int(axes), a.ndim))
                right_contract = tuple(range(int(axes)))
            else:
                left_contract, right_contract = axes
                left_contract = tuple(int(x) for x in left_contract)
                right_contract = tuple(int(x) for x in right_contract)
            if len(left_contract) != len(right_contract):
                raise SPFError("decoded_contract_reference contraction axes must be paired")
            label_next = 0
            left_labels: list[int | None] = [None] * a.ndim
            right_labels: list[int | None] = [None] * b.ndim
            out_labels: list[int] = []

            for left_axis, right_axis in batch_pairs:
                if a.shape[left_axis] != b.shape[right_axis]:
                    raise SPFError("decoded_contract_reference batch axis size mismatch")
                label = label_next
                label_next += 1
                left_labels[left_axis] = label
                right_labels[right_axis] = label
                out_labels.append(label)
            for left_axis, right_axis in zip(left_contract, right_contract, strict=True):
                if a.shape[left_axis] != b.shape[right_axis]:
                    raise SPFError("decoded_contract_reference contracted axis size mismatch")
                label = label_next
                label_next += 1
                left_labels[left_axis] = label
                right_labels[right_axis] = label
            for idx, label in enumerate(left_labels):
                if label is None:
                    label = label_next
                    label_next += 1
                    left_labels[idx] = label
                    out_labels.append(label)
            for idx, label in enumerate(right_labels):
                if label is None:
                    label = label_next
                    label_next += 1
                    right_labels[idx] = label
                    out_labels.append(label)
            contracted = np.einsum(a, left_labels, b, right_labels, out_labels)
            return encode_words(contracted)
        contracted = np.tensordot(a, b, axes=axes)
        return encode_words(contracted)

    def _execute_decoded_sum(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        # Reference-only additive boundary for LazySum. Like decoded_contract_reference,
        # this declares the reducer/requantization point but intentionally decodes to
        # complex128 so integer reducer implementations can be tested against it.
        assert step.reducer is not None
        if not step.inputs:
            raise SPFError("decoded_sum_reference requires at least one input")
        dense = decode_words(np.asarray(env[step.inputs[0]], dtype=np.uint32))
        for name in step.inputs[1:]:
            dense = dense + decode_words(np.asarray(env[name], dtype=np.uint32))
        return encode_words(dense)

    def _execute_decoded_pushforward(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        assert step.reducer is not None
        if len(step.inputs) != 1:
            raise SPFError("decoded_pushforward_reference requires exactly one input")
        dense = decode_words(np.asarray(env[step.inputs[0]], dtype=np.uint32))
        axes = tuple(int(axis) for axis in step.params.get("axes", ()))
        if not axes:
            raise SPFError("decoded_pushforward_reference requires at least one axis")
        if len(set(axes)) != len(axes):
            raise SPFError("decoded_pushforward_reference axes must be unique")
        if any(axis < 0 or axis >= dense.ndim for axis in axes):
            raise SPFError("decoded_pushforward_reference axis out of range")
        return encode_words(np.sum(dense, axis=axes))

    def _execute_contract(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        """Dispatch the ``contract`` op to :func:`~spf.tensor.attention_ops.execute_contract`."""
        assert step.reducer is not None
        from .attention_ops import execute_contract

        A = np.asarray(env[step.inputs[0]], dtype=np.uint32)
        B = np.asarray(env[step.inputs[1]], dtype=np.uint32)
        vector = B.ndim == 1
        if vector:
            B = B[None, :] if bool(step.params.get("transpose_b", False)) else B[:, None]
        result = execute_contract(
            A,
            B,
            step.reducer,
            transpose_b=bool(step.params.get("transpose_b", False)),
            causal=bool(step.params.get("causal", False)),
        )
        return result[:, 0] if vector else result

    def _execute_softmax_lse(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        """Dispatch the ``softmax_lse`` op to :func:`~spf.tensor.attention_ops.execute_softmax_lse`."""
        assert step.reducer is not None
        from .attention_ops import execute_softmax_lse

        scores = env[step.inputs[0]]
        return execute_softmax_lse(
            scores,
            step.reducer,
            causal=bool(step.params.get("causal", False)),
            l_frac_bits=int(step.params.get("l_frac_bits", 16)),
            beta=float(step.params.get("beta", 1.0)),
            out=str(step.params.get("out", "f32")),
            pre_quantized=bool(step.params.get("pre_quantized", False)),
        )

    def _execute_route(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        """Dispatch the fused ``route`` op to :func:`~spf.tensor.route_op.execute_route`."""
        assert step.reducer is not None
        from .route_op import execute_route

        return execute_route(
            np.asarray(env[step.inputs[0]], dtype=np.uint32),
            np.asarray(env[step.inputs[1]], dtype=np.uint32),
            np.asarray(env[step.inputs[2]], dtype=np.uint32),
            step.reducer,
            causal=bool(step.params.get("causal", False)),
            beta=float(step.params.get("beta", 8.0)),
            l_frac_bits=int(step.params.get("l_frac_bits", 16)),
        )

    def _execute_gauge_norm(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        """Dispatch the ``gauge_norm`` op to :func:`~spf.tensor.norm_ops.execute_gauge_norm`."""
        from .norm_ops import execute_gauge_norm

        gamma = step.params.get("gamma_codes")
        return execute_gauge_norm(
            np.asarray(env[step.inputs[0]], dtype=np.uint32),
            tier=str(step.params.get("tier", "gml")),
            axis=int(step.params.get("axis", -1)),
            gamma_codes=None if gamma is None else np.asarray(gamma),
        )

    def _execute_l_gate(self, env: Mapping[str, Any], step: ExecutionStep) -> np.ndarray:
        """Dispatch the ``l_gate`` op to :func:`~spf.tensor.norm_ops.execute_l_gate`."""
        from .norm_ops import execute_l_gate

        if "tau_codes" not in step.params:
            raise SPFError("l_gate step requires params['tau_codes']")
        return execute_l_gate(
            np.asarray(env[step.inputs[0]], dtype=np.uint32),
            tau_codes=np.asarray(step.params["tau_codes"]),
            axis=int(step.params.get("axis", -1)),
        )


def _band_to_metadata(band: BandRange) -> dict[str, int]:
    return {
        "dim": int(band.dim),
        "start": int(band.start),
        "length": int(band.length),
        "L_delta": int(band.L_delta),
        "T_delta": int(band.T_delta),
        "flags": int(band.effective_flags),
    }


def _band_from_metadata(data: Mapping[str, object]) -> BandRange:
    return BandRange(
        dim=int(data["dim"]),
        start=int(data["start"]),
        length=int(data["length"]),
        L_delta=int(data.get("L_delta", 0)),
        T_delta=int(data.get("T_delta", 0)),
        flags=int(data.get("flags", 0)),
    )


def _block_payload(block: TensorBlock) -> tuple[bytes, dict[str, tuple[int, int]]]:
    block.validate()
    chunks: list[bytes] = []
    sections: dict[str, tuple[int, int]] = {}

    def append(name: str, payload: bytes) -> None:
        start = sum(len(chunk) for chunk in chunks)
        chunks.append(payload)
        sections[name] = (start, start + len(payload))

    if block.header.active_encoding == ActiveEncoding.COO_U32:
        active = np.ascontiguousarray(block.active_indices, dtype=np.uint32)
        append("active_indices", active.tobytes())
    dL = np.ascontiguousarray(block.dL, dtype=np.int16)
    dT = np.ascontiguousarray(block.dT, dtype=np.int16)
    append("dL", dL.tobytes())
    append("dT", dT.tobytes())
    if block.packed_residuals:
        append("packed_residuals", bytes(block.packed_residuals))
    return b"".join(chunks), sections


def to_spftensor_safetensors(
    blocks: TensorBlock | AffineTensorBlock | Sequence[TensorBlock | AffineTensorBlock],
) -> bytes:
    """Write a safetensors-style SPFTensor envelope using JSON + binary buffers."""

    if isinstance(blocks, (TensorBlock, AffineTensorBlock)):
        block_seq: tuple[TensorBlock | AffineTensorBlock, ...] = (blocks,)
    else:
        block_seq = tuple(blocks)
    v2_blocks = tuple(migrate_affine_to_v2(b) if isinstance(b, AffineTensorBlock) else b for b in block_seq)
    payload_chunks: list[bytes] = []
    header: dict[str, Any] = {
        "__metadata__": {
            "format": "spftensor",
            "spftensor_major": str(SPFTENSOR_MAJOR),
            "spftensor_minor": str(SPFTENSOR_MINOR),
            "created_unix_ns": str(time.time_ns()),
        }
    }
    cursor = 0
    for i, block in enumerate(v2_blocks):
        payload, sections = _block_payload(block)
        start = cursor
        end = start + len(payload)
        cursor = end
        payload_chunks.append(payload)
        entry_meta = {
            "header": block.header.to_metadata(),
            "sections": {name: [int(a), int(b)] for name, (a, b) in sections.items()},
            "bands": [_band_to_metadata(band) for band in block.bands],
            "outliers": {str(k): int(v) for k, v in block.outliers.items()},
        }
        header[f"blocks/{i}"] = {
            "dtype": SPFTENSOR_DTYPE_V2,
            "shape": [int(block.header.total_elements)],
            "data_offsets": [start, end],
            "metadata": entry_meta,
        }
    header_json = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return struct.pack("<Q", len(header_json)) + header_json + b"".join(payload_chunks)


def from_spftensor_safetensors(data: bytes) -> TensorBlock | tuple[TensorBlock, ...]:
    """Read a safetensors-style SPFTensor envelope."""

    if len(data) < 8:
        raise SPFError("SPFTensor envelope is shorter than safetensors header prefix")
    (header_len,) = struct.unpack("<Q", data[:8])
    if header_len > len(data) - 8:
        raise SPFError("SPFTensor header length exceeds buffer")
    try:
        header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SPFError("SPFTensor header is not valid JSON") from exc
    if not isinstance(header, dict):
        raise SPFError("SPFTensor header root must be a JSON object")
    meta = header.get("__metadata__", {})
    try:
        major = int(meta.get("spftensor_major", SPFTENSOR_MAJOR))
    except (TypeError, ValueError) as exc:
        raise SPFError("SPFTensor major version is malformed") from exc
    if major != SPFTENSOR_MAJOR:
        raise SPFError(f"unsupported SPFTensor major version {major}")
    payload = data[8 + header_len :]
    blocks: list[TensorBlock] = []

    def entry_key(name: str) -> tuple[str, int | str]:
        prefix, _, suffix = name.partition("/")
        return (prefix, int(suffix) if suffix.isdecimal() else suffix)

    for name in sorted((k for k in header if k != "__metadata__"), key=entry_key):
        entry = header[name]
        if not isinstance(entry, Mapping):
            raise SPFError(f"SPFTensor entry {name} must be an object")
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, Sequence) or len(offsets) != 2:
            raise SPFError(f"SPFTensor entry {name} has malformed data_offsets")
        start, end = int(offsets[0]), int(offsets[1])
        if start < 0 or end < start or end > len(payload):
            raise SPFError(f"SPFTensor entry {name} data_offsets out of range")
        dtype = str(entry.get("dtype"))
        entry_meta = entry.get("metadata", {})
        if not isinstance(entry_meta, Mapping):
            raise SPFError(f"SPFTensor entry {name} metadata must be an object")
        block_payload = payload[start:end]
        if dtype == SPFTENSOR_DTYPE_V2:
            blocks.append(_read_v2_block(entry_meta, block_payload))
        elif dtype == SPFTENSOR_DTYPE_V1:
            blocks.append(_read_v1_block(entry_meta, block_payload))
        else:
            raise SPFError(f"unsupported SPFTensor dtype {dtype!r}")
    if len(blocks) == 1:
        return blocks[0]
    return tuple(blocks)


def _read_section(payload: bytes, sections: Mapping[str, object], name: str) -> bytes:
    if name not in sections:
        raise SPFError(f"missing SPFTensor payload section {name}")
    bounds = sections[name]
    if not isinstance(bounds, Sequence) or len(bounds) != 2:
        raise SPFError(f"section {name} has malformed offsets")
    start, end = int(bounds[0]), int(bounds[1])
    if start < 0 or end < start or end > len(payload):
        raise SPFError(f"section {name} offsets out of range")
    return payload[start:end]


def _frombuffer_checked(raw: bytes, dtype: npt.DTypeLike, name: str) -> np.ndarray:
    """``np.frombuffer`` with an itemsize-alignment guard that raises ``SPFError``.

    A malformed/untrusted envelope can carry a section length that is not a
    multiple of the element size; raw ``np.frombuffer`` would leak a bare
    ``ValueError`` instead of the module's ``SPFError`` contract.
    """

    itemsize = np.dtype(dtype).itemsize
    if len(raw) % itemsize != 0:
        raise SPFError(f"SPFTensor section {name} length {len(raw)} is not a multiple of {itemsize}")
    return np.frombuffer(raw, dtype=dtype).copy()


def _read_v2_block(entry_meta: Mapping[str, object], payload: bytes) -> TensorBlock:
    header_meta = entry_meta.get("header")
    if not isinstance(header_meta, Mapping):
        raise SPFError("SPF_TB_v2 entry missing header metadata")
    header = TensorBlockHeaderV2.from_metadata(header_meta)
    sections = entry_meta.get("sections", {})
    if not isinstance(sections, Mapping):
        raise SPFError("SPF_TB_v2 sections metadata must be an object")
    active_indices = None
    if header.active_encoding == ActiveEncoding.COO_U32:
        active_indices = _frombuffer_checked(
            _read_section(payload, sections, "active_indices"), np.uint32, "active_indices"
        )
    dL = _frombuffer_checked(_read_section(payload, sections, "dL"), np.int16, "dL")
    dT = _frombuffer_checked(_read_section(payload, sections, "dT"), np.int16, "dT")
    bands = tuple(_band_from_metadata(band) for band in entry_meta.get("bands", ()))
    outliers = {int(k): int(v) for k, v in dict(entry_meta.get("outliers", {})).items()}
    packed = b""
    if "packed_residuals" in sections:
        packed = _read_section(payload, sections, "packed_residuals")
    block = TensorBlock(
        header=header,
        dL=dL,
        dT=dT,
        active_indices=active_indices,
        bands=bands,
        outliers=outliers,
        packed_residuals=packed,
    )
    block.validate()
    return block


def _read_v1_block(entry_meta: Mapping[str, object], payload: bytes) -> TensorBlock:
    try:
        h = OperatorHeader.from_bytes(bytes.fromhex(str(entry_meta["operator_header_hex"])))
    except KeyError as exc:
        raise SPFError("SPF_TB_v1 entry missing operator_header_hex metadata") from exc
    active_encoding = ActiveEncoding(int(entry_meta.get("active_encoding", ActiveEncoding.DENSE_ALL)))
    sections = entry_meta.get("sections", {})
    if not isinstance(sections, Mapping):
        raise SPFError("SPF_TB_v1 sections metadata must be an object")
    active_indices = None
    if active_encoding == ActiveEncoding.COO_U32:
        active_indices = _frombuffer_checked(
            _read_section(payload, sections, "active_indices"), np.uint32, "active_indices"
        )
    dL = _frombuffer_checked(_read_section(payload, sections, "dL"), np.int16, "dL")
    dT = _frombuffer_checked(_read_section(payload, sections, "dT"), np.int16, "dT")
    bands = tuple(_band_from_metadata(band) for band in entry_meta.get("bands", ()))
    return migrate_affine_to_v2(
        AffineTensorBlock(
            header=h,
            active_encoding=active_encoding,
            active_indices=active_indices,
            dL=dL,
            dT=dT,
            bands=bands,
        )
    )


__all__ = [
    "BLOCK_CANCEL_COHERENCE_MAX",
    "BLOCK_CANCEL_DL_THRESHOLD",
    "EXECUTION_OPS",
    "PROFILE_CP_CLOSED",
    "PROFILE_TT_RANK_MAX",
    "PROFILE_TT_RANK_MIN",
    "PROFILE_V3_NONABELIAN_MAX",
    "PROFILE_V3_NONABELIAN_MIN",
    "SPFTENSOR_DTYPE_V1",
    "SPFTENSOR_DTYPE_V2",
    "SPFTENSOR_MAJOR",
    "SPFTENSOR_MINOR",
    "BlockMetadata",
    "ExecutionPlan",
    "ExecutionStep",
    "ProfileTerm",
    "SPFBlock",
    "SPFScalar",
    "TensorBlock",
    "TensorBlockHeaderV2",
    "active_dimensions",
    "block_dominant_index",
    "block_dynamic_range",
    "block_metadata",
    "block_metadata_from_planes",
    "block_phase_coherence",
    "cancellation_risk",
    "determinant_mod2",
    "from_spftensor_safetensors",
    "migrate_affine_to_v2",
    "ramp_determinant_mod2",
    "to_spftensor_safetensors",
]
