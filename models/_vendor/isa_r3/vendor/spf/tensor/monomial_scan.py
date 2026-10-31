"""Addition-free automaton scans over monomial (permutation x phase-diagonal) words.

A monomial ``M(d, p)`` acts on a word vector ``x`` as ``(M x)[i] = d[i] * x[p[i]]``:
one gather and one SPF32 word multiply per coordinate. Monomials compose, invert
and take adjoints in closed form,

    M(da, pa) M(db, pb) = M(da * db[pa], pb[pa])
    M(d, p)^*           = M(conj(d)[q], q)          q = p^-1
    M(d, p)^-1          = M((1/d)[q], q)

so a recurrence ``x_t = M_t x_{t-1}`` has prefix states ``S_t = M_t ... M_0`` made
only of index gathers and integer log/phase additions. No coherent sum, no
finalization and no Cartesian decode occurs; every composition is exact when the
log range is certified. Exact composition is associative, so every scan method
(sequential, Hillis-Steele ``parallel``, work-efficient ``blelloch``, two-level
``chunked``) produces identical words.

The composition law holds for any index map ``p: [n] -> [n]``, not only for
permutations: ``perm = pb[pa]`` is function composition and ``diag`` is a gathered
integer sum, both associative. ``bijective=False`` admits such transformation-monoid
maps (resets, merges); they have no inverse or adjoint. ``order="right"`` forms
``S_t = S_{t-1} M_t`` instead of ``M_t S_{t-1}``; with transition maps ``f_t`` its
``perm`` plane is ``f_t o ... o f_0``, the automaton run from every start state.

The SPF32 word multiply saturates log codes at the finite range, and a clamped
product is not associative. The unit-circle domain (L = 0) is the default and never
saturates. Nonzero log magnitudes are accepted only under a conservative range
certificate: the summed per-step maximum ``|L|`` (plus any carried prefix) must fit
the finite SPF32 log range, which bounds every window product any method forms.
Inputs that cannot be certified are refused rather than silently clamped.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_NAN,
    SPF32_L_ZERO,
    SPF32_PHASE_MOD,
    SPFError,
    spf_conj_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)

_IDENTITY_WORD = np.uint32(0)  # L = 0, theta = 0


@dataclass
class MonomialCounters:
    """Actual operation counts, not latency measurements."""

    gathers: int = 0
    word_multiplications: int = 0
    integer_additions: int = 0
    coherent_additions: int = 0
    compositions: int = 0  # n-wide monomial products formed by a scan

    def charge(self, size: int, gathers_per_element: int) -> None:
        self.gathers += gathers_per_element * size
        self.word_multiplications += size
        # One SPF32 word multiply is one log-code add and one modular phase add.
        self.integer_additions += 2 * size


@dataclass(frozen=True)
class Monomial:
    """Monomial planes: ``perm`` (..., n) integer index maps, ``diag`` (..., n) SPF32 words.

    ``perm`` rows are permutations unless a caller passes ``bijective=False``.
    """

    perm: np.ndarray
    diag: np.ndarray

    def __post_init__(self) -> None:
        if self.perm.shape != self.diag.shape:
            raise SPFError("monomial perm and diag planes must share shape (..., n)")


@dataclass(frozen=True)
class MonomialScanResult:
    """Inclusive prefix monomials ``S_t`` for every step, plus operation counters."""

    prefix: Monomial
    counters: MonomialCounters = field(default_factory=MonomialCounters)

    @property
    def last(self) -> Monomial:
        """The carried prefix for streaming the next chunk (index -1 in time order)."""
        return Monomial(self.prefix.perm[..., -1, :], self.prefix.diag[..., -1, :])

    @property
    def first(self) -> Monomial:
        """The carried prefix for streaming a reverse scan's next (left) chunk."""
        return Monomial(self.prefix.perm[..., 0, :], self.prefix.diag[..., 0, :])


def _is_bijective(perm: np.ndarray) -> bool:
    n = perm.shape[-1]
    return bool(np.array_equal(np.sort(perm, axis=-1), np.broadcast_to(np.arange(n), perm.shape)))


def _perm_plane(perm: npt.ArrayLike, name: str = "perm", *, bijective: bool = True) -> np.ndarray:
    arr = np.asarray(perm)
    if arr.dtype == np.bool_ or not np.issubdtype(arr.dtype, np.integer):
        raise SPFError(f"{name} must be an integer index-map plane")
    if arr.ndim == 0 or arr.shape[-1] == 0:
        raise SPFError(f"{name} requires a nonempty coordinate axis")
    n = arr.shape[-1]
    if np.any(arr < 0) or np.any(arr >= n):
        raise SPFError(f"{name} entries must lie in [0, n)")
    perm64 = arr.astype(np.int64)
    if bijective and not _is_bijective(perm64):
        raise SPFError(
            f"{name} rows must be permutations of range(n); pass bijective=False for "
            "transformation-monoid maps such as resets"
        )
    return perm64


def _diag_plane(diag: npt.ArrayLike, *, unit_circle: bool, name: str = "diag") -> np.ndarray:
    arr = np.asarray(diag)
    if arr.dtype != np.uint32 or arr.ndim == 0:
        raise SPFError(f"{name} must be packed SPF32 uint32 words with a coordinate axis")
    L = unpack_L(arr)
    if np.any(L == SPF32_L_ZERO):
        raise SPFError(f"{name} contains a zero word; a monomial transition must be invertible")
    if np.any(L == SPF32_L_NAN):
        raise SPFError(f"{name} contains an Inf/NaN sentinel word")
    if unit_circle and np.any(L != 0):
        raise SPFError(f"{name} must be unit-circle words (L=0); pass unit_circle=False with a range certificate")
    return arr


def _monomial(m: Monomial, *, unit_circle: bool, name: str, bijective: bool = True) -> Monomial:
    perm = _perm_plane(m.perm, f"{name} perm", bijective=bijective)
    diag = _diag_plane(m.diag, unit_circle=unit_circle, name=f"{name} diag")
    if perm.shape != diag.shape:
        raise SPFError(f"{name} perm and diag planes must share shape (..., n)")
    return Monomial(perm, diag)


def _max_abs_log(diag: np.ndarray) -> np.ndarray:
    return np.max(np.abs(unpack_L(diag).astype(np.int64)), axis=-1)


def _certify_range(diag: np.ndarray, initial: Monomial | None) -> None:
    """Conservative certificate: no prefix or window log sum can leave the finite range."""
    budget = np.sum(_max_abs_log(diag), axis=-1)
    if initial is not None:
        budget = budget + _max_abs_log(initial.diag)
    if np.any(budget > SPF32_L_MAX_FINITE):
        raise SPFError(
            "monomial scan log range is not certified: summed per-step max |L| exceeds the finite "
            "SPF32 range, so a product could saturate; split the scan or rebase the magnitudes"
        )


def compose_monomials(a: Monomial, b: Monomial, counters: MonomialCounters | None = None) -> Monomial:
    """``a`` applied after ``b``: ``M(da * db[pa], pb[pa])`` (broadcast over leading axes)."""
    pa, pb = np.broadcast_arrays(a.perm, b.perm)
    da, db = np.broadcast_arrays(a.diag, b.diag)
    perm = np.take_along_axis(pb, pa, axis=-1)
    diag = spf_mul_words(da, np.take_along_axis(db, pa, axis=-1))
    if counters is not None:
        counters.charge(perm.size, gathers_per_element=2)
        counters.compositions += perm.size // perm.shape[-1]
    return Monomial(perm, diag)


def identity_monomial(shape: Sequence[int]) -> Monomial:
    """Identity ``M(1, range(n))`` with planes of shape ``shape`` = (..., n)."""
    shape = tuple(int(v) for v in shape)
    if not shape or shape[-1] <= 0:
        raise SPFError("identity monomial needs a positive coordinate axis")
    perm = np.broadcast_to(np.arange(shape[-1], dtype=np.int64), shape).copy()
    return Monomial(perm, np.full(shape, _IDENTITY_WORD, dtype=np.uint32))


def _inverse_perm(perm: np.ndarray) -> np.ndarray:
    # argsort of a permutation is its inverse: an index computation, no arithmetic.
    return np.argsort(perm, axis=-1, kind="stable").astype(np.int64)


def _bijective_operand(m: Monomial, op: str) -> Monomial:
    perm = _perm_plane(m.perm, f"{op} operand perm", bijective=False)
    if not _is_bijective(perm):
        raise SPFError(
            f"monomial {op} needs a bijective index map; a non-bijective (transformation-monoid) "
            f"map such as a reset forgets which coordinate it came from and has no {op}"
        )
    return _monomial(Monomial(perm, m.diag), unit_circle=False, name=f"{op} operand")


def monomial_adjoint(m: Monomial) -> Monomial:
    """``M(d, p)^* = M(conj(d)[q], q)`` with ``q = p^-1``. Exact; ``p`` must be a permutation."""
    m = _bijective_operand(m, "adjoint")
    q = _inverse_perm(m.perm)
    return Monomial(q, np.take_along_axis(spf_conj_words(m.diag), q, axis=-1))


def monomial_inverse(m: Monomial) -> Monomial:
    """``M(d, p)^-1 = M((1/d)[q], q)``. Refuses non-bijective maps and unrepresentable reciprocals."""
    from .kernel import _spf_words_reciprocal

    m = _bijective_operand(m, "inverse")
    L = unpack_L(m.diag).astype(np.int64)
    if np.any(-L > SPF32_L_MAX_FINITE) or np.any(-L < SPF32_L_MIN_FINITE):
        raise SPFError("monomial inverse: a reciprocal log code leaves the finite SPF32 range")
    # Existing exact log-polar reciprocal (-L, -theta); the check above rules out its clamp.
    recip = _spf_words_reciprocal(m.diag).reshape(m.diag.shape)
    q = _inverse_perm(m.perm)
    return Monomial(q, np.take_along_axis(recip, q, axis=-1))


def _time_slice(m: Monomial, index) -> Monomial:
    return Monomial(m.perm[..., index, :], m.diag[..., index, :])


def _store(dst: Monomial, index, src: Monomial) -> None:
    dst.perm[..., index, :] = src.perm
    dst.diag[..., index, :] = src.diag


# A scan combines an earlier prefix with a later element: ``op(earlier, later)``.
_Combine = Callable[[Monomial, Monomial], Monomial]


def _scan_sequential(steps: Monomial, op: _Combine) -> Monomial:
    """T - 1 dependent compositions, each vectorized over the leading axes."""
    out = Monomial(np.empty(steps.perm.shape, dtype=np.int64), np.empty(steps.diag.shape, dtype=np.uint32))
    acc = _time_slice(steps, 0)
    _store(out, 0, acc)
    for t in range(1, steps.perm.shape[-2]):
        acc = op(acc, _time_slice(steps, t))
        _store(out, t, acc)
    return out


def _scan_parallel(steps: Monomial, op: _Combine) -> Monomial:
    """Hillis-Steele inclusive scan: ceil(log2 T) passes, about T log2 T compositions."""
    perm, diag = steps.perm.copy(), steps.diag.copy()
    T = perm.shape[-2]
    k = 1
    while k < T:
        window = op(Monomial(perm[..., :-k, :], diag[..., :-k, :]), Monomial(perm[..., k:, :], diag[..., k:, :]))
        perm = np.concatenate((perm[..., :k, :], window.perm), axis=-2)
        diag = np.concatenate((diag[..., :k, :], window.diag), axis=-2)
        k <<= 1
    return Monomial(perm, diag)


def _pad_identity(steps: Monomial, pad: int) -> Monomial:
    """Append ``pad`` identity steps (copies, so callers may write in place)."""
    if pad == 0:
        return Monomial(steps.perm.copy(), steps.diag.copy())
    ident = identity_monomial((*steps.perm.shape[:-2], pad, steps.perm.shape[-1]))
    return Monomial(
        np.concatenate((steps.perm, ident.perm), axis=-2), np.concatenate((steps.diag, ident.diag), axis=-2)
    )


def _scan_blelloch(steps: Monomial, op: _Combine) -> Monomial:
    """Work-efficient (Blelloch) scan: 2 log2 P passes, 2 (P - 1) compositions, P = 2^ceil(log2 T).

    The up-sweep leaves each right child holding its subtree product. The down-sweep
    pushes exclusive prefixes back down, keeping operand order (earlier on the left of
    ``op``) because composition does not commute. Inclusive prefix ``t`` is exclusive
    prefix ``t + 1``, and the last one is the saved root, so no final pass is needed.
    """
    T = steps.perm.shape[-2]
    P = 1 << (T - 1).bit_length()
    tree = _pad_identity(steps, P - T)
    stride = 2
    while stride <= P:
        left, right = slice(stride // 2 - 1, None, stride), slice(stride - 1, None, stride)
        _store(tree, right, op(_time_slice(tree, left), _time_slice(tree, right)))
        stride <<= 1
    root = Monomial(tree.perm[..., -1:, :].copy(), tree.diag[..., -1:, :].copy())
    _store(tree, -1, identity_monomial(root.perm.shape[:-2] + root.perm.shape[-1:]))
    stride = P
    while stride >= 2:
        left, right = slice(stride // 2 - 1, None, stride), slice(stride - 1, None, stride)
        subtree = Monomial(tree.perm[..., left, :].copy(), tree.diag[..., left, :].copy())
        prefix = Monomial(tree.perm[..., right, :].copy(), tree.diag[..., right, :].copy())
        _store(tree, left, prefix)
        _store(tree, right, op(prefix, subtree))
        stride >>= 1
    perm = np.concatenate((tree.perm, root.perm), axis=-2)[..., 1 : T + 1, :]
    diag = np.concatenate((tree.diag, root.diag), axis=-2)[..., 1 : T + 1, :]
    return Monomial(perm, diag)


def _scan_chunked(steps: Monomial, op: _Combine) -> Monomial:
    """Two-level scan with chunk length c = ceil(sqrt(T)): about 2T compositions, 2 sqrt(T) passes.

    (1) sequential scan inside every chunk at once, (2) sequential scan of the chunk
    totals, (3) one fix-up pass composing each chunk's carried prefix on its earlier side.
    This is the layout a streaming kernel uses: local work stays in one chunk.
    """
    T, n = steps.perm.shape[-2:]
    lead = steps.perm.shape[:-2]
    c = math.isqrt(T - 1) + 1
    C = -(-T // c)
    padded = _pad_identity(steps, C * c - T)
    chunks = Monomial(padded.perm.reshape(*lead, C, c, n), padded.diag.reshape(*lead, C, c, n))
    local = _scan_sequential(chunks, op)
    carry = _scan_sequential(_time_slice(local, -1), op)
    if C > 1:
        fixed = op(
            Monomial(carry.perm[..., :-1, None, :], carry.diag[..., :-1, None, :]),
            Monomial(local.perm[..., 1:, :, :], local.diag[..., 1:, :, :]),
        )
        local.perm[..., 1:, :, :] = fixed.perm
        local.diag[..., 1:, :, :] = fixed.diag
    return Monomial(local.perm.reshape(*lead, C * c, n)[..., :T, :], local.diag.reshape(*lead, C * c, n)[..., :T, :])


_SCAN_METHODS = {
    "sequential": _scan_sequential,
    "parallel": _scan_parallel,
    "blelloch": _scan_blelloch,
    "chunked": _scan_chunked,
}


def execute_monomial_scan(
    perm: npt.ArrayLike,
    diag: npt.ArrayLike,
    *,
    initial: Monomial | None = None,
    method: str = "sequential",
    reverse: bool = False,
    unit_circle: bool = True,
    order: str = "left",
    bijective: bool = True,
) -> MonomialScanResult:
    """Inclusive prefix monomials for ``x_t = M_t x_{t-1}`` (or ``x_t = M_t x_{t+1}`` reversed).

    ``perm``/``diag`` have shape (..., T, n). With ``order="left"`` (default) each new
    factor enters on the left: forward ``S_t = M_t ... M_0 I``, reverse
    ``S_t = M_t ... M_{T-1} I``. With ``order="right"`` it enters on the right: forward
    ``S_t = I M_0 ... M_t``, reverse ``S_t = I M_{T-1} ... M_t``. For transition maps
    ``f_t`` the right-order forward ``perm`` plane is ``f_t o ... o f_0`` (row ``q`` is
    the state reached from start ``q``) and ``diag`` is the path weight; this is the
    order that runs automata with resets (``bijective=False``).

    ``initial`` (shape (..., n)) is the carried prefix from a previous chunk
    (``result.last`` forward, ``result.first`` reversed), giving bit-exact streaming.
    ``method`` is ``"sequential"``, ``"parallel"`` (Hillis-Steele), ``"blelloch"``
    (work-efficient) or ``"chunked"`` (two-level); all return identical words because
    certified composition is exact and associative. No coherent addition occurs.
    """
    if method not in _SCAN_METHODS:
        raise SPFError(f"monomial scan method must be one of {sorted(_SCAN_METHODS)}")
    if order not in ("left", "right"):
        raise SPFError("monomial scan order must be 'left' or 'right'")
    flags = (reverse, unit_circle, bijective)
    if not all(isinstance(flag, (bool, np.bool_)) for flag in flags):
        raise SPFError("monomial scan reverse/unit_circle/bijective flags must be booleans")
    steps = _monomial(
        Monomial(np.asarray(perm), np.asarray(diag)), unit_circle=unit_circle, name="transition", bijective=bijective
    )
    if steps.perm.ndim < 2:
        raise SPFError("monomial scan planes need shape (..., T, n)")
    lead, n = steps.perm.shape[:-2], steps.perm.shape[-1]
    start = None
    if initial is not None:
        start = _monomial(initial, unit_circle=unit_circle, name="initial", bijective=bijective)
        if start.perm.shape != (*lead, n):
            raise SPFError("initial prefix must have shape (..., n) matching the transition planes")
    if not unit_circle:
        _certify_range(steps.diag, start)
    if steps.perm.shape[-2] == 0:
        empty = Monomial(steps.perm.copy(), steps.diag.copy())
        return MonomialScanResult(empty)
    if reverse:
        steps = Monomial(np.flip(steps.perm, axis=-2), np.flip(steps.diag, axis=-2))
    counters = MonomialCounters()

    def op(earlier: Monomial, later: Monomial) -> Monomial:
        if order == "left":
            return compose_monomials(later, earlier, counters)
        return compose_monomials(earlier, later, counters)

    if start is not None:
        # Fold the carried prefix into the first step; every method then scans from identity.
        first = op(start, _time_slice(steps, 0))
        steps = Monomial(steps.perm.copy(), steps.diag.copy())
        _store(steps, 0, first)
    prefix = _SCAN_METHODS[method](steps, op)
    if reverse:
        prefix = Monomial(np.flip(prefix.perm, axis=-2).copy(), np.flip(prefix.diag, axis=-2).copy())
    return MonomialScanResult(prefix, counters)


def apply_monomial(
    prefix: Monomial,
    x_words: npt.ArrayLike,
    counters: MonomialCounters | None = None,
) -> np.ndarray:
    """Readout ``y = d * x[p]`` per step: one gather and one word multiply per coordinate.

    ``x_words`` broadcasts against ``prefix.diag``; a state vector of shape (..., n)
    applied to scan prefixes of shape (..., T, n) should be passed as ``x[..., None, :]``.
    Zero state words are allowed (they stay zero). No coherent addition occurs.
    """
    x = np.asarray(x_words)
    if x.dtype != np.uint32 or x.ndim == 0:
        raise SPFError("monomial readout state must be packed SPF32 uint32 words")
    if np.any(unpack_L(x) == SPF32_L_NAN):
        raise SPFError("monomial readout state contains an Inf/NaN sentinel word")
    if x.shape[-1] != prefix.perm.shape[-1]:
        raise SPFError("monomial readout state width differs from the prefix coordinate axis")
    perm, diag = np.broadcast_arrays(prefix.perm, prefix.diag)
    shape = np.broadcast_shapes(perm.shape, x.shape)
    perm = np.broadcast_to(perm, shape)
    gathered = np.take_along_axis(np.broadcast_to(x, shape), perm, axis=-1)
    diag = np.broadcast_to(diag, shape)
    Lx = unpack_L(gathered).astype(np.int64)
    Ly = unpack_L(diag).astype(np.int64) + Lx
    if np.any((Lx != SPF32_L_ZERO) & ((Ly > SPF32_L_MAX_FINITE) | (Ly < SPF32_L_MIN_FINITE))):
        raise SPFError("monomial readout would saturate the SPF32 log range")
    y = spf_mul_words(diag, gathered)
    if counters is not None:
        counters.charge(y.size, gathers_per_element=1)
    return y


_STEP_PARAMS = frozenset({"method", "reverse", "unit_circle", "order", "bijective"})


def execute_monomial_scan_step(
    env: dict, inputs: Sequence[str], outputs: Sequence[str], params: dict
) -> dict[str, np.ndarray]:
    """ExecutionPlan ``monomial_scan`` semantics.

    Inputs ``(perm, diag)`` produce outputs ``(prefix_perm, prefix_diag)``. Inputs
    ``(perm, diag, x)`` with ``x`` of shape (..., n) produce one output: the state
    ``S_t x`` for every step, shape (..., T, n). Params: method, reverse, unit_circle,
    order, bijective.
    """
    unknown = set(params) - _STEP_PARAMS
    if unknown:
        raise SPFError(f"monomial_scan got unknown parameters {sorted(unknown)}")
    if len(inputs) not in (2, 3) or len(outputs) != (1 if len(inputs) == 3 else 2):
        raise SPFError("monomial_scan takes (perm, diag) -> (perm, diag) or (perm, diag, x) -> (y,)")
    result = execute_monomial_scan(env[inputs[0]], env[inputs[1]], **params)
    if len(inputs) == 2:
        return {outputs[0]: result.prefix.perm, outputs[1]: result.prefix.diag}
    x = np.asarray(env[inputs[2]])
    return {outputs[0]: apply_monomial(result.prefix, x[..., None, :] if x.ndim else x)}


@dataclass(frozen=True)
class MonomialTransitionTable:
    """Transition ROM of an automaton: K stored monomials selected per token by a symbol.

    ``perms``/``diags`` have shape (K, n). Symbols are integers in ``[0, K)`` or are
    read from unit-circle feature words with ``floor(theta * K / 65536)``. With
    ``bijective=False`` the stored maps may be arbitrary functions (resets); scan them
    with ``order="right"`` to run the automaton from every start state.
    """

    perms: np.ndarray
    diags: np.ndarray
    unit_circle: bool = True
    bijective: bool = True

    def __post_init__(self) -> None:
        table = _monomial(
            Monomial(np.asarray(self.perms), np.asarray(self.diags)),
            unit_circle=self.unit_circle,
            name="transition table",
            bijective=self.bijective,
        )
        if table.perm.ndim != 2 or table.perm.shape[0] == 0:
            raise SPFError("transition table planes must have shape (K, n) with K >= 1")
        object.__setattr__(self, "perms", table.perm)
        object.__setattr__(self, "diags", table.diag)

    @property
    def n_symbols(self) -> int:
        return int(self.perms.shape[0])

    def planes(self, symbols: npt.ArrayLike) -> Monomial:
        """Gather the (..., T, n) transition planes for integer symbols of shape (..., T)."""
        sym = np.asarray(symbols)
        if sym.dtype == np.bool_ or not np.issubdtype(sym.dtype, np.integer):
            raise SPFError("automaton symbols must be integers")
        if np.any(sym < 0) or np.any(sym >= self.n_symbols):
            raise SPFError("automaton symbol out of range")
        sym = sym.astype(np.int64)
        return Monomial(self.perms[sym], self.diags[sym])

    def symbols_from_phase(self, feature_words: npt.ArrayLike) -> np.ndarray:
        """ROM index ``floor(theta * K / 65536)`` of unit-circle feature words."""
        words = np.asarray(feature_words)
        if words.dtype != np.uint32:
            raise SPFError("feature words must be packed SPF32 uint32 words")
        if np.any(unpack_L(words) != 0):
            raise SPFError("ROM-indexing feature words must be unit-circle (L=0)")
        theta = unpack_theta(words).astype(np.int64)
        return (theta * self.n_symbols) // SPF32_PHASE_MOD

    def scan(self, symbols: npt.ArrayLike, **kwargs) -> MonomialScanResult:
        planes = self.planes(symbols)
        kwargs.setdefault("unit_circle", self.unit_circle)
        kwargs.setdefault("bijective", self.bijective)
        return execute_monomial_scan(planes.perm, planes.diag, **kwargs)


__all__ = [
    "Monomial",
    "MonomialCounters",
    "MonomialScanResult",
    "MonomialTransitionTable",
    "apply_monomial",
    "compose_monomials",
    "execute_monomial_scan",
    "execute_monomial_scan_step",
    "identity_monomial",
    "monomial_adjoint",
    "monomial_inverse",
]
