"""Batched rank consumers for separable amplitude fields.

Factor callbacks keep their scalar contract. Each distinct axis coordinate is
visited once, products remain packed, and rank sums share a pivoted integer
accumulator. This is amplitude separability, never a phase-plane rank shortcut.

This module also owns the *certificate* side of §17: the conversion of a
Gaussian-chirp atom or mixture into separable factors, the bound on the
off-diagonal chirp term such a conversion drops, the residual envelope, and the
§17.2 storage ratio. The planner and the compiler both call it so that one
implementation decides when a separable candidate may be published as exact.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import numpy as np

from spf.renderer.carriers import decode_words, encode_words
from spf.tensor import v02
from spf.tensor.core import spf_mul_words

#: Which library reducer each supported value kind's rank sum runs through.
#: ``"planes"`` is the vectorized plane reducer; it carries its own certified
#: bound but is *not* word-identical to the scalar ``v02.reduce_*`` oracle for
#: ``log_signed``/``spf_complex``.  ``evaluate_grid(..., reducer="scalar")``
#: selects the scalar oracle itself.
RANK_REDUCERS: dict[str, str] = {
    "spf_complex": "spf.tensor.v02_planes.reduce_complex",
    "log_pos": "spf.tensor.v02_planes.reduce_logpos",
    "log_signed": "spf.tensor.v02_planes.reduce_logsigned",
}

#: Kinds whose plane reducer is word-identical to the scalar reducer (measured
#: by ``tests/v02/test_sfr_separable_certificate.py``).
PLANE_REDUCER_BIT_IDENTICAL: frozenset[str] = frozenset({"log_pos"})


def rank_reducer(kind: str, *, reducer: str = "planes") -> str:
    """Name the reducer :func:`evaluate_grid` uses for ``kind``."""

    kind = v02._normalize_kind_name(kind)
    if reducer == "scalar":
        return "spf.tensor.v02.reduce_v02"
    return RANK_REDUCERS.get(kind, "spf.tensor.v02.reduce_v02")


@lru_cache(maxsize=1)
def _phase_table() -> np.ndarray:
    table = np.asarray([v02._sincos_phase(i) for i in range(65536)])
    table.setflags(write=False)
    return table


def _factor_words(value: Any, kind: str) -> int:
    if kind == "log_pos":
        typed = v02._coerce_logpos(value)
        return v02.SPF32_WORD_NAN if typed.is_nan else int(v02.pack_spf32(typed.L, 0))
    if kind == "log_signed":
        signed = v02._coerce_logsigned(value)
        return v02.SPF32_WORD_NAN if signed.is_nan else int(v02.pack_spf32(signed.L, 0 if signed.sign >= 0 else 32768))
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return int(encode_words(value))
    return v02._coerce_word(value)


def evaluate_grid(
    field: v02.SeparableField, points: np.ndarray, *, kind: str, reducer: str = "planes"
) -> tuple[np.ndarray, float, float]:
    """Return typed samples and the worst rank-sum error, through an explicit pivot boundary.

    ``reducer="planes"`` (the default) runs the rank sum through the vectorized
    plane reducer named by :data:`RANK_REDUCERS`.  ``reducer="scalar"`` runs the
    same scalar reducer :func:`spf.tensor.v02.evaluate_separable` uses, so its
    output words are identical to that oracle for every kind, at 70-150x the
    cost (measured, 1k-64k points, rank 3-8).
    """
    kind = v02._normalize_kind_name(kind)
    if reducer not in {"planes", "scalar"}:
        raise ValueError("separable rank reducer must be 'planes' or 'scalar'")
    native = kind in {"native", "real", "dense", "float", "float64"}
    if (kind not in {"spf_complex", "log_pos", "log_signed"} and not native) or (reducer == "scalar" and not native):
        results = [v02.evaluate_separable(field, p, kind=kind) for p in points]
        bound = max((r.absolute_error_bound for r in results), default=0)
        phase_error = max((r.phase_error_bound for r in results), default=0)
        if kind in {"spf_complex", "log_pos", "log_signed"}:
            # Match the packed/decoded terminal type of the vectorized route so
            # the reducer choice is invisible to a caller apart from the words.
            words = np.asarray([_factor_words(r.value, kind) for r in results], dtype=np.uint32)
            return (words if kind == "spf_complex" else np.real(decode_words(words))), bound, phase_error
        return np.asarray([r.value for r in results]), bound, phase_error
    if points.shape[-1] != field.dimension:
        raise ValueError("separable evaluation point has the wrong dimension")
    coordinates = [np.unique(points[:, axis], return_inverse=True) for axis in range(field.dimension)]
    products = []
    for row in field.factors:
        product = np.ones(points.shape[0]) if native else np.zeros(points.shape[0], dtype=np.uint32)
        for factor, (axis_values, inverse) in zip(row, coordinates, strict=True):
            samples = [factor(value) for value in axis_values] if callable(factor) else [factor] * axis_values.size
            values = (
                np.asarray(samples)
                if native
                else np.asarray([_factor_words(v, kind) for v in samples], dtype=np.uint32)
            )
            selected = values[inverse]
            product = product * selected if native else spf_mul_words(product, selected)
        products.append(product)
    if native:
        return np.asarray(sum(products, np.zeros(points.shape[0]))), 0.0, 0.0
    if not products:
        values = np.full(points.shape[0], v02.SPF32_WORD_ZERO, dtype=np.uint32)
        return values if kind == "spf_complex" else np.zeros(points.shape[0]), 0.0, 0.0
    rows = np.stack(products, axis=1)
    words, bound, phase_error = _rank_reduce(rows, kind)
    if kind != "spf_complex":
        return np.real(decode_words(words)), bound, phase_error
    return words, bound, phase_error


def _rank_reduce(rows: np.ndarray, kind: str) -> tuple[np.ndarray, float, float]:
    """Rank sum over ``axis=1`` through the library plane reducers, with their certified bounds.

    ``rows`` is ``(points, rank)`` of packed product words.  ``spf_complex``
    rows reduce with :func:`spf.tensor.v02_planes.reduce_complex`; ``log_pos``
    rows reduce on the positive log plane and ``log_signed`` rows (zero or
    half-turn phases) on the signed log plane.  Each publishes its own certified
    bound, and this terminal keeps no private fixed-point accumulator.

    These are *not* word-identical to the scalar ``v02.reduce_*`` oracle:
    measured over 2000 random rows per kind, ``log_pos`` agrees 2000/2000 while
    ``log_signed`` differs on 112 and ``spf_complex`` on 502 (both reducers'
    own bounds hold on every case).  A caller that needs the scalar words calls
    :func:`evaluate_grid` with ``reducer="scalar"``; :data:`RANK_REDUCERS` and
    :func:`rank_reducer` name the reducer actually used.
    """

    from spf.tensor import v02_planes as planes

    words_in = np.asarray(rows, dtype=np.uint32)
    if words_in.ndim != 2:
        raise ValueError("rank reduction expects a (points, rank) word matrix")
    if kind == "spf_complex":
        reduction = planes.reduce_complex(words_in, axis=1)
        words = np.asarray(reduction.value, dtype=np.uint32)
    elif kind == "log_pos":
        reduction = planes.reduce_logpos(planes.LogPosPlane.from_words(words_in), axis=1)
        words = np.asarray(reduction.value.to_words(), dtype=np.uint32)
    else:
        reduction = planes.reduce_logsigned(planes.LogSignedPlane.from_words(words_in), axis=1)
        words = np.asarray(reduction.value.to_words(), dtype=np.uint32)
    bound = float(np.max(np.asarray(reduction.absolute_error_bound, dtype=np.float64), initial=0.0))
    phase = float(np.max(np.asarray(reduction.phase_error_bound, dtype=np.float64), initial=0.0))
    return words.reshape(words_in.shape[0]), bound, phase


# ---------------------------------------------------------------------------
# §17 certified conversion into separable factors
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeparableConversion:
    """Separable factors plus the bound on what they do not represent.

    ``bound`` is an absolute pointwise bound on ``|source - field|`` over the
    region the conversion was certified on.  Only ``bound == 0`` may be
    published as ``EXACT``; any other value is a real ledger entry and the
    candidate is inexact (Design §11.3 prohibits uncertified approximate
    merges in ``ACCURATE``/``EXACT``).
    """

    field: v02.SeparableField
    bound: float = 0.0
    detail: str = "exact diagonal factors"

    def __post_init__(self) -> None:
        bound = float(self.bound)
        if math.isnan(bound) or bound < 0.0:
            raise ValueError("separable conversion bound must be non-negative or infinite")
        object.__setattr__(self, "bound", bound)

    @property
    def exact(self) -> bool:
        return self.bound == 0.0


@dataclass(frozen=True, slots=True)
class SeparableStorage:
    """§17.2 storage accounting: ``prod(n_j)`` dense against ``R*sum(n_j)``."""

    dense_values: int
    separable_values: int

    @property
    def ratio(self) -> float:
        if self.separable_values <= 0:
            return math.inf
        return self.dense_values / self.separable_values


def region_box(region: Any, dimension: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(lower, upper)`` for a renderer ``Region`` or an atom region form."""

    if region is None:
        return None
    lower = getattr(region, "lower", None)
    upper = getattr(region, "upper", None)
    if lower is not None and upper is not None and not callable(lower) and not callable(upper):
        low = np.asarray([float(value) for value in lower], dtype=float)
        high = np.asarray([float(value) for value in upper], dtype=float)
    else:
        try:
            low, high = v02._box_bounds(region, dimension)
        except (TypeError, ValueError, IndexError):
            return None
    if low.shape != (dimension,) or high.shape != (dimension,):
        return None
    return low, high


def cross_term_exponent_sup(matrix: Any, region: Any) -> float:
    """Bound ``sup|E|`` over the region, where ``E = -1/2 sum_{i != j} C_ij x_i x_j``.

    ``E`` is exactly the exponent a diagonal separable form drops.  It grows
    with the square of the coordinate scale, so no absolute tolerance on
    ``C_ij`` certifies dropping it: ``C_01 = 9e-13`` at ``q = 1e6`` already
    moves the value by 146%.  All-zero off-diagonals give exactly ``0``; an
    unknown or unusable region gives ``inf``.
    """

    entries = np.asarray(matrix)
    if entries.ndim != 2 or entries.shape[0] != entries.shape[1]:
        return math.inf
    dimension = int(entries.shape[0])
    pairs = [(i, j) for i in range(dimension) for j in range(dimension) if i != j]
    magnitudes = [abs(complex(entries[i, j])) for i, j in pairs]
    if not any(magnitudes):
        return 0.0
    box = region_box(region, dimension)
    if box is None:
        return math.inf
    lower, upper = box
    extent = np.maximum(np.abs(lower), np.abs(upper))
    total = math.fsum(
        0.5 * magnitude * float(extent[i]) * float(extent[j])
        for magnitude, (i, j) in zip(magnitudes, pairs, strict=True)
    )
    return total if math.isfinite(total) else math.inf


def diagonal_relative_bound(matrix: Any, region: Any) -> float:
    """Relative bound ``|exp(sup|E|) - 1|`` of dropping the off-diagonal chirp."""

    supremum = cross_term_exponent_sup(matrix, region)
    if supremum == 0.0:
        return 0.0
    if not math.isfinite(supremum):
        return math.inf
    try:
        return math.expm1(supremum)
    except OverflowError:
        return math.inf


def residual_upper_bound(residual: Any) -> float:
    """Return ``sup|residual|``, or ``inf`` when nothing certifies one.

    A stored numeric residual certifies itself: the maximum over its stored
    values bounds it everywhere it is defined.  An object may declare an
    envelope through ``magnitude_upper_bound``/``upper_bound``/``bound``.  A
    bare callable declares nothing, and sampling it would publish a measurement
    as a bound, so it returns infinity — the same reading
    ``normalization`` already gives an "unbounded callable residual".
    """

    if residual is None:
        return 0.0
    for name in ("magnitude_upper_bound", "upper_bound", "bound"):
        declared = getattr(residual, name, None)
        if declared is None or callable(declared):
            continue
        try:
            value = abs(float(declared))
        except (TypeError, ValueError):
            continue
        return value if math.isfinite(value) else math.inf
    if callable(residual):
        return math.inf
    try:
        array = np.asarray(residual)
    except (TypeError, ValueError):
        return math.inf
    if array.size == 0:
        return 0.0
    try:
        value = float(np.max(np.abs(array)))
    except (TypeError, ValueError):
        return math.inf
    return value if math.isfinite(value) else math.inf


def constant_residual(residual: Any) -> complex | None:
    """Return a stored residual's constant value, or ``None``.

    Only a stored numeric residual can certify constancy.  Probing a callable
    at sample points is a measurement, not a proof, so a callable is refused.
    """

    if residual is None or callable(residual):
        return None
    try:
        array = np.asarray(residual)
    except (TypeError, ValueError):
        return None
    if array.size == 0:
        return None
    flat = array.reshape(-1)
    if not bool(np.all(flat == flat[0])):
        return None
    try:
        return complex(flat[0])
    except (TypeError, ValueError):
        return None


def _factor_exp(exponent: complex) -> complex:
    """Scalar complex exponential for one separable factor callback.

    Factor callbacks keep the scalar contract -- :func:`evaluate_grid` visits
    each distinct axis coordinate once and ``v02.evaluate_separable`` passes one
    coordinate at a time -- so this stays a scalar ``cmath`` call rather than a
    dense array transcendental.  An overflowing exponent saturates to an
    infinite magnitude at phase zero, which the reducer's envelope reads as
    unknown instead of raising inside a render.
    """

    try:
        return cmath.exp(exponent)
    except OverflowError:
        return complex(math.inf, 0.0)


def diagonal_atom_factors(atom: v02.GaussianChirpAtom) -> tuple[Any, ...] | None:
    """Factors of the *diagonal part* of a 2-D Gaussian-chirp atom.

    The off-diagonal cross term is not represented here; the caller certifies
    it with :func:`diagonal_relative_bound` before publishing an exactness.
    """

    if atom.dimension != 2:
        return None
    matrix = np.asarray(atom.C)
    vector = np.asarray(atom.h)
    c0, c1 = complex(matrix[0, 0]), complex(matrix[1, 1])
    h0, h1 = complex(vector[0]), complex(vector[1])
    a0 = _factor_exp(complex(atom.kappa))
    return (
        lambda x, c=c0, h=h0, a=a0: a * _factor_exp(h * x - 0.5 * c * x * x),
        lambda y, c=c1, h=h1: _factor_exp(h * y - 0.5 * c * y * y),
    )


def _constant_factors(value: complex, dimension: int) -> tuple[Any, ...]:
    """A rank-1 row whose product is the constant ``value`` in every coordinate."""

    return (lambda _q, v=value: v, *(((lambda _q: 1.0),) * (dimension - 1)))


def _diagonal_envelope(atom: v02.GaussianChirpAtom, region: Any) -> float:
    """``sup|psi_diag|`` over the region, through the audited §15.9 atom bound."""

    box = region_box(region, atom.dimension)
    if box is None:
        return math.inf
    matrix = np.asarray(atom.C)
    diagonal = np.diag(np.diag(matrix))
    try:
        reference = replace(atom, C=diagonal, allow_nondecaying=True)
        bound = float(v02.atom_bound(reference, {"lo": box[0], "hi": box[1]}))
    except (TypeError, ValueError, OverflowError, np.linalg.LinAlgError):
        return math.inf
    return bound if math.isfinite(bound) else math.inf


def separable_conversion(payload: Any, region: Any = None) -> SeparableConversion | None:
    """Certified conversion of an atom, mixture, or separable field into factors.

    ``None`` means no certificate is available and the caller must refuse the
    conversion.  A returned conversion always states a real bound: the dropped
    off-diagonal chirp, an unrepresentable mixture residual, and the mixture's
    own ``dropped_bound`` are all accounted for, so ``bound == 0`` is the only
    case that may be published as ``EXACT``.
    """

    if isinstance(payload, v02.SeparableField):
        return SeparableConversion(payload, 0.0, "already separable")
    if isinstance(payload, v02.GaussianChirpAtom):
        factors = diagonal_atom_factors(payload)
        if factors is None:
            return None
        separable = v02.SeparableField((factors,), "SPF_COMPLEX")
        relative = diagonal_relative_bound(payload.C, region)
        if relative == 0.0:
            return SeparableConversion(separable, 0.0, "diagonal atom: C has no cross term")
        bound = _diagonal_envelope(payload, region) * relative
        if not math.isfinite(bound):
            return None
        return SeparableConversion(separable, bound, f"dropped cross term, relative <= {relative:.6g}")
    if not isinstance(payload, v02.AtomMixture):
        return None
    rows: list[tuple[Any, ...]] = []
    bounds: list[float] = [float(payload.dropped_bound)]
    details: list[str] = []
    for atom in payload.atoms:
        conversion = separable_conversion(atom, region)
        if conversion is None:
            return None
        rows.extend(conversion.field.factors)
        if conversion.bound:
            bounds.append(conversion.bound)
            details.append(conversion.detail)
    dimension = payload.atoms[0].dimension if payload.atoms else len(getattr(region, "lower", ()) or ())
    if payload.residual is not None:
        constant = constant_residual(payload.residual)
        if constant is not None and dimension >= 1:
            rows.append(_constant_factors(constant, dimension))
            details.append("constant residual folded as one rank")
        else:
            residual_bound = residual_upper_bound(payload.residual)
            if not math.isfinite(residual_bound):
                return None
            bounds.append(residual_bound)
            details.append(f"residual bounded by {residual_bound:.6g}")
    if not rows:
        return None
    if payload.dropped_bound:
        details.append(f"mixture dropped_bound {float(payload.dropped_bound):.6g}")
    total = math.fsum(bounds)
    if not math.isfinite(total):
        return None
    detail = "; ".join(details) if details else "exact diagonal factors"
    return SeparableConversion(v02.SeparableField(tuple(rows), "SPF_COMPLEX"), total, detail)


def separable_storage(field: v02.SeparableField, shape: Sequence[int]) -> SeparableStorage:
    """§17.2 storage accounting for ``field`` materialized over ``shape``.

    ``128**3`` at rank 8 gives ``2_097_152`` dense values against
    ``8 * (128 + 128 + 128) = 3072`` one-dimensional values, the design's
    ``683x`` ratio.
    """

    extents = [int(n) for n in shape]
    if not extents or any(n <= 0 for n in extents):
        raise ValueError("separable storage accounting needs a positive extent per axis")
    dense = 1
    for extent in extents:
        dense *= extent
    return SeparableStorage(dense, max(1, field.rank) * sum(extents))


__all__ = [
    "PLANE_REDUCER_BIT_IDENTICAL",
    "RANK_REDUCERS",
    "SeparableConversion",
    "SeparableStorage",
    "constant_residual",
    "cross_term_exponent_sup",
    "diagonal_atom_factors",
    "diagonal_relative_bound",
    "evaluate_grid",
    "rank_reducer",
    "region_box",
    "residual_upper_bound",
    "separable_conversion",
    "separable_storage",
]
