"""Closed SPFBlockMatrix laws for order-2 STB header blocks (matmul post laws 1/4/5/6/7).

Backing derivation and numeric checks:
``obsidian/spf/2026-09-30-135512-spf-matmul-expanded-and-factored.md`` §1.2, §2, §4,
§5, §7 and the table in §8, with the exact header reconstruction formula from
``obsidian/spf/2026-09-30-133152-spf-mathematical-definition.md`` §10.1.

The product of two order-2 ``STB_AFFINE_POLY`` blocks with no ``k²`` term is

    C_ij = e^{ρ(i)} · F_K(0; s_ij) · e^{σ(j)},   s_ij = s0 + u·q_i + v·q_j,

a diagonally scaled *ridge / Cauchy-like* matrix (matmul post §1.2, §2.3). We read
both headers exactly into integer tick polynomials in ``(i, k)`` and ``(k, j)``
(:func:`read_stb_block`, :func:`stb_axis_decomposition`), fold the free-axis bands
into the two diagonals (**law 6**), and store the result as a lazy
:class:`GeometricRidge` node holding the integer generators ``(s0, u, v, K)`` plus
the two diagonals as SPF words. Materialization is two SPF pair-adds per entry,
reusing the exact word helper :func:`spf.tensor.kernel._spf_geometric_depth_sum_words`
(**law 1**). A ``SEPARABLE_PHASE`` Gram maps onto the same node with non-uniform
phase nodes (**law 7**, :func:`separable_phase_gram`). Residual/outlier operands are
rewritten additively into the closed pieces plus a sparse patch (**law 5**,
:func:`compose_stb_residual`). The matvec chirp correlation (**law 4**,
:func:`stb_block_matvec_reference`) is an explicit opt-in; see its docstring for why.

Every precondition failure declines (returns ``None``) so the dense fallback keeps
the product correct — never silently wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_PHASE_MASK,
    ReducerSpec,
    SPFError,
    _sat_l16,
    decode_words,
    encode_words,
    pack_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)
from .v3 import BaseModelKind, Representation, ResidualCodecKind

__all__ = [
    "GEOMETRIC_RIDGE_OP",
    "GeometricRidgeAlgebra",
    "STBBlockAlgebra",
    "StbHeaderView",
    "compose_stb_residual",
    "compose_stb_ridge",
    "geometric_ridge",
    "materialize_geometric_ridge",
    "read_stb_block",
    "separable_phase_gram",
    "stb_axis_decomposition",
    "stb_block_matvec_reference",
]

GEOMETRIC_RIDGE_OP = "GeometricRidge"


# --------------------------------------------------------------------------- #
# Header introspection: read an order-2 STB block into integer tick polynomials.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StbHeaderView:
    """An order-2 ``STB_AFFINE_POLY`` block's header, read into plain ints.

    Field names follow the §10.1 reconstruction exactly (``L`` in log ticks, ``Θ``
    in phase ticks; ``d`` indexes the two axes). ``colsum_R[d]`` is the ``ramp_R``
    column sum that enters the linear phase, ``bilinear_*`` is the single order-2
    ``(0,1)`` cross term. ``bands`` is the list of :class:`BandRange` steps.
    """

    dims: tuple[int, int]
    origin: tuple[int, int]
    # log-magnitude (L) tick coefficients
    l_const: int  # global_L_bias + coarse_exp
    l_lin: tuple[int, int]  # band_exp[d]
    l_quad: tuple[int, int]  # log_quadratic[d]
    l_bilinear: int  # log_mixed[(0,1)]
    # phase (Θ) tick coefficients
    t_const: int  # global_T_bias + sum(ramp_b)
    t_lin: tuple[int, int]  # colsum_R[d] + phase_gradient[d]
    t_quad: tuple[int, int]  # quadratic_ramp[d]
    t_cubic: tuple[int, int]  # cubic_ramp[d]
    t_bilinear: int  # mixed_ramp[(0,1)]
    t_cubic_mixed: tuple[int, int]  # cubic_mixed = (q0^2 q1, q0 q1^2)
    bands: tuple[Any, ...]
    has_residual: bool
    has_outliers: bool


def read_stb_block(kernel: Any) -> StbHeaderView | None:
    """Read a rank-2 dense ``STB_AFFINE_POLY`` :class:`SPFBlockMatrix` into ints.

    Returns ``None`` (decline) unless the kernel is a ``Block`` node backed by a
    dense, ``POLAR_SPF32`` ``STB_AFFINE_POLY`` block with a real 2-D header. The
    returned view carries every §10.1 tick coefficient plus residual/outlier
    flags; callers decide which terms a given law can absorb.
    """
    if getattr(kernel, "expr", None) is None or kernel.expr.op != "Block":
        return None
    if len(kernel.sig.out_axes) != 1 or len(kernel.sig.in_axes) != 1 or kernel.sig.batch_axes:
        return None
    block = kernel.expr.block
    if block is None:
        return None
    if int(block.representation.rep_id) != int(Representation.POLAR_SPF32):
        return None
    if int(block.base.profile_id) != int(BaseModelKind.STB_AFFINE_POLY):
        return None
    params = block.base.params
    header = params.get("header")
    if header is None or int(header.order or 0) != 2:
        return None
    dims = tuple(int(d) for d in header.dims)
    if len(dims) != 2:
        return None
    # Law 1 needs the whole grid live; a sparse active set leaves ZERO holes that
    # the closed ridge does not model, so decline to the dense fallback.
    active = np.asarray(params.get("active_indices", ()), dtype=np.int64)
    total = int(dims[0] * dims[1])
    if active.size != total or not np.array_equal(active, np.arange(total, dtype=np.int64)):
        return None

    origin = tuple(int(x) for x in (header.origin or (0, 0)))[:2]
    band_exp = tuple(int(x) for x in (header.band_exp or (0, 0)))
    log_quad = tuple(int(x) for x in (header.log_quadratic or (0, 0)))
    log_mixed = tuple(int(x) for x in (header.log_mixed or (0,)))
    ramp_R = np.asarray(header.ramp_R, dtype=np.int64).reshape(2, 2)
    colsum = ramp_R.sum(axis=0)
    grad = tuple(int(x) for x in (header.phase_gradient or (0, 0)))
    quad = tuple(int(x) for x in (header.quadratic_ramp or (0, 0)))
    cubic = tuple(int(x) for x in (header.cubic_ramp or (0, 0)))
    mixed = tuple(int(x) for x in (header.mixed_ramp or (0,)))
    cubic_mixed = tuple(int(x) for x in (header.cubic_mixed or (0, 0)))
    ramp_b = np.asarray(header.ramp_b, dtype=np.int64) if header.ramp_b is not None else np.zeros(2, np.int64)

    return StbHeaderView(
        dims=(dims[0], dims[1]),
        origin=(origin[0], origin[1]),
        l_const=int(header.global_L_bias) + int(header.coarse_exp),
        l_lin=(band_exp[0], band_exp[1]),
        l_quad=(log_quad[0], log_quad[1]),
        l_bilinear=int(log_mixed[0]) if log_mixed else 0,
        t_const=int(header.global_T_bias) + int(ramp_b.sum()),
        t_lin=(int(colsum[0]) + grad[0], int(colsum[1]) + grad[1]),
        t_quad=(quad[0], quad[1]),
        t_cubic=(cubic[0], cubic[1]),
        t_bilinear=int(mixed[0]) if mixed else 0,
        t_cubic_mixed=(cubic_mixed[0], cubic_mixed[1]),
        bands=tuple(params.get("bands", ()) or ()),
        has_residual=int(block.residual.codec_id) != int(ResidualCodecKind.NONE),
        has_outliers=bool(block.outliers),
    )


def _cyclic_coords(size: int, origin: int) -> np.ndarray:
    return (int(origin) + np.arange(int(size), dtype=np.int64)) % int(size)


@dataclass(frozen=True)
class _AxisDecomposition:
    n_free: int
    rho_words: np.ndarray  # (n_free,) diagonal e^{ρ(free)} as SPF words
    node_l: np.ndarray  # (n_free,) bilinear·q_free — the L node ticks
    node_t: np.ndarray  # (n_free,) bilinear·q_free — the Θ node ticks
    klin_l: int  # k-linear L constant (no free dependence)
    klin_t: int  # k-linear Θ constant
    eta_l: int  # k² L coefficient (must cancel for law 1)
    eta_t: int  # k² Θ coefficient


def stb_axis_decomposition(view: StbHeaderView, *, free_dim: int, contract_dim: int) -> _AxisDecomposition | None:
    """Split an STB header's exponent into ``ρ(free) + s·k + η·k²`` (matmul post §1.2).

    ``free_dim`` is the row axis of the left operand (``i``) or the column axis of
    the right operand (``j``); ``contract_dim`` is the shared ``k`` axis. Declines
    (returns ``None``) whenever the contracted axis is not a clean quadratic in
    ``k`` — a cyclic-origin wrap, a ``k³``/mixed-cubic term, or a band on ``k`` —
    because then the exponent is not polynomial in ``k`` over the contracted range.
    Free-axis bands are folded into ``ρ`` (**law 6**).
    """
    n_free = int(view.dims[free_dim])
    # Cyclic wrap on the contracted axis makes q_k non-polynomial in k over [0,K).
    if int(view.origin[contract_dim]) != 0:
        return None
    # k³ and mixed cubics (q_i² q_k, q_i q_k²) break the affine ridge in k.
    if view.t_cubic[contract_dim] != 0:
        return None
    if view.t_cubic_mixed[0] != 0 or view.t_cubic_mixed[1] != 0:
        return None
    # Bands on the contracted axis are not constant over k → decline.
    for band in view.bands:
        if int(band.dim) == contract_dim:
            return None

    q_free = _cyclic_coords(n_free, view.origin[free_dim])

    # ρ(free): the exponent terms with no k-dependence.
    l_rho = view.l_const + view.l_lin[free_dim] * q_free + view.l_quad[free_dim] * q_free * q_free
    t_rho = (
        view.t_const
        + view.t_lin[free_dim] * q_free
        + view.t_quad[free_dim] * q_free * q_free
        + view.t_cubic[free_dim] * q_free * q_free * q_free
    )
    # Free-axis bands fold into the diagonal (law 6).
    for band in view.bands:
        if int(band.dim) != free_dim:
            continue
        covered = (q_free >= int(band.start)) & (q_free < int(band.start) + int(band.length))
        if int(band.effective_flags) & 0x01:
            l_rho = l_rho + covered * int(band.L_delta)
        if int(band.effective_flags) & 0x02:
            t_rho = t_rho + covered * int(band.T_delta)

    # The dense operand saturates (or hits the ZERO sentinel) wherever its raw L
    # leaves the finite range; the closed form would then compute the product of
    # the *unclipped* field, so decline and let the dense fallback match the words.
    k_idx = np.arange(int(view.dims[contract_dim]), dtype=np.int64)[None, :]
    l_grid = (
        np.asarray(l_rho, dtype=np.int64)[:, None]
        + (view.l_bilinear * q_free[:, None] + view.l_lin[contract_dim]) * k_idx
        + view.l_quad[contract_dim] * k_idx * k_idx
    )
    if l_grid.size and (int(l_grid.min()) < int(SPF32_L_MIN_FINITE) or int(l_grid.max()) > int(SPF32_L_MAX_FINITE)):
        return None

    rho_words = pack_words(
        _sat_l16(l_rho).astype(np.int16),
        (t_rho & SPF32_PHASE_MASK).astype(np.uint16),
    )

    # k-linear coefficient s(free) = klin_const + bilinear·q_free.
    node_l = view.l_bilinear * q_free
    node_t = view.t_bilinear * q_free
    klin_l = view.l_lin[contract_dim]
    klin_t = view.t_lin[contract_dim]
    eta_l = view.l_quad[contract_dim]
    eta_t = view.t_quad[contract_dim]

    return _AxisDecomposition(
        n_free=n_free,
        rho_words=rho_words.reshape(-1),
        node_l=node_l.astype(np.int64),
        node_t=node_t.astype(np.int64),
        klin_l=int(klin_l),
        klin_t=int(klin_t),
        eta_l=int(eta_l),
        eta_t=int(eta_t),
    )


# --------------------------------------------------------------------------- #
# GeometricRidge node: Diag(ρ) · F_K(0; s_ij) · Diag(σ).
# --------------------------------------------------------------------------- #


def _ridge_word_grid(params: dict[str, Any]) -> np.ndarray:
    n, m = int(params["N"]), int(params["M"])
    s0_l, s0_t = int(params["s0_l"]), int(params["s0_t"])
    nr_l = np.asarray(params["node_row_l"], dtype=np.int64).reshape(n, 1)
    nr_t = np.asarray(params["node_row_t"], dtype=np.int64).reshape(n, 1)
    nc_l = np.asarray(params["node_col_l"], dtype=np.int64).reshape(1, m)
    nc_t = np.asarray(params["node_col_t"], dtype=np.int64).reshape(1, m)
    w_l = s0_l + nr_l + nc_l
    w_t = (s0_t + nr_t + nc_t) & SPF32_PHASE_MASK
    return pack_words(_sat_l16(w_l).astype(np.int16), w_t.astype(np.uint16))


def _ridge_saturates(params: dict[str, Any]) -> bool:
    """True when ``w`` or ``w^K`` would leave the SPF32 ±128-octave L range."""
    n, m = int(params["N"]), int(params["M"])
    k = int(params["K"])
    s0_l = int(params["s0_l"])
    nr_l = np.asarray(params["node_row_l"], dtype=np.int64)
    nc_l = np.asarray(params["node_col_l"], dtype=np.int64)
    if n == 0 or m == 0:
        return False
    w_min = s0_l + int(nr_l.min()) + int(nc_l.min())
    w_max = s0_l + int(nr_l.max()) + int(nc_l.max())
    lo = k * w_min
    hi = k * w_max
    return (
        w_min < int(SPF32_L_MIN_FINITE)
        or w_max > int(SPF32_L_MAX_FINITE)
        or lo < int(SPF32_L_MIN_FINITE)
        or hi > int(SPF32_L_MAX_FINITE)
    )


def materialize_geometric_ridge(params: dict[str, Any]) -> np.ndarray:
    """Materialize a :data:`GEOMETRIC_RIDGE_OP` node to an ``(N, M)`` word plane.

    ``F_K(0; s_ij) = (1 − w^K)/(1 − w)`` per entry via the exact word helper
    :func:`spf.tensor.kernel._spf_geometric_depth_sum_words` (two SPF pair-adds per
    general entry; ``w = 1 → K`` and character orthogonality ``w^K = 1, w ≠ 1 → 0``
    exactly), then the two diagonals fold in with one SPF multiply each.
    """
    from .kernel import _default_reducer, _spf_geometric_depth_sum_words

    n, m = int(params["N"]), int(params["M"])
    k = int(params["K"])
    reducer = params.get("reducer") or _default_reducer()
    w_words = _ridge_word_grid(params).reshape(-1)
    closed = _spf_geometric_depth_sum_words(w_words, k, reducer)
    if closed is None:
        raise SPFError("geometric ridge saturated at materialize (compose should have declined)")
    ridge = closed[0].reshape(n, m)
    rho = np.asarray(params["rho_words"], dtype=np.uint32).reshape(n, 1)
    sigma = np.asarray(params["sigma_words"], dtype=np.uint32).reshape(1, m)
    out = spf_mul_words(np.broadcast_to(rho, (n, m)).copy(), ridge)
    out = spf_mul_words(out, np.broadcast_to(sigma, (n, m)).copy())
    return np.asarray(out, dtype=np.uint32).reshape(n, m)


def geometric_ridge(
    *,
    N: int,
    M: int,
    K: int,
    s0_l: int,
    s0_t: int,
    node_row_l: np.ndarray,
    node_row_t: np.ndarray,
    node_col_l: np.ndarray,
    node_col_t: np.ndarray,
    rho_words: np.ndarray,
    sigma_words: np.ndarray,
    out_axis: Any,
    in_axis: Any,
    reducer_policy: ReducerSpec | None,
    law: str,
    notes: tuple[str, ...] = (),
) -> Any | None:
    """Construct a lazy :data:`GEOMETRIC_RIDGE_OP` kernel, or ``None`` if it saturates."""
    from .kernel import (
        AxisSpec,
        KernelSignature,
        SPFBlockExpr,
        SPFKernel,
        SPFProof,
        _default_reducer,
        _spf_words_integer_power,
        _structural_view_block,
    )

    n, m, k = int(N), int(M), int(K)
    params: dict[str, Any] = {
        "N": n,
        "M": m,
        "K": k,
        "s0_l": int(s0_l),
        "s0_t": int(s0_t) & SPF32_PHASE_MASK,
        "node_row_l": np.asarray(node_row_l, dtype=np.int64).reshape(n),
        "node_row_t": np.asarray(node_row_t, dtype=np.int64).reshape(n),
        "node_col_l": np.asarray(node_col_l, dtype=np.int64).reshape(m),
        "node_col_t": np.asarray(node_col_t, dtype=np.int64).reshape(m),
        "rho_words": np.asarray(rho_words, dtype=np.uint32).reshape(n),
        "sigma_words": np.asarray(sigma_words, dtype=np.uint32).reshape(m),
        "reducer": reducer_policy or _default_reducer(),
    }
    if _ridge_saturates(params):
        return None
    # Count the reducer boundary the closed form crosses at materialize: two pair
    # adds per general entry, none where w = 1 (sum = K) or w^K = 1 (exact zero).
    # Integer work only; the pair adds themselves run once, at materialize.
    w_words = _ridge_word_grid(params).reshape(-1)
    w_pow, in_range = _spf_words_integer_power(w_words, k)
    if not in_range:
        return None
    special = ((unpack_L(w_words) == 0) & (unpack_theta(w_words) == 0)) | (
        (unpack_L(w_pow) == 0) & (unpack_theta(w_pow) == 0)
    )
    reducer_count = 2 * int(np.count_nonzero(~special))

    oa = AxisSpec(out_axis.name, n, role="codomain", basis=out_axis.basis)
    ia = AxisSpec(in_axis.name, m, role="domain", basis=in_axis.basis)
    block = _structural_view_block((n, m), "geometric_ridge", {"N": n, "M": m, "K": k})
    return SPFKernel(
        SPFBlockExpr(
            GEOMETRIC_RIDGE_OP,
            args=(params,),
            block=block,
            proof=SPFProof(law, False, False, reducer_count, 0, notes=notes),
        ),
        (oa, ia),
        KernelSignature(
            (oa.name,),
            (ia.name,),
            basis_in=(ia.basis,),
            basis_out=(oa.basis,),
            exactness="quantized",
            reducer_policy=reducer_policy,
        ),
    )


# --------------------------------------------------------------------------- #
# Law 1 + 6: STB header block ∘ STB header block → GeometricRidge (η = 0).
# --------------------------------------------------------------------------- #


def compose_stb_ridge(a: Any, b: Any) -> Any | None:
    """Law 1 (+6): ``Block(STB, order 2, η=0) ∘ Block(STB)`` → ridge (matmul post §2).

    Reads both headers exactly, folds free-axis bands into the row/column diagonals
    (law 6), and returns ``Diag(e^ρ) · F_K(0; s_ij) · Diag(e^σ)`` as a
    :data:`GEOMETRIC_RIDGE_OP` node. Declines (dense fallback) when either operand
    carries a residual/outlier (that is law 5, :func:`compose_stb_residual`), when
    the combined ``k²`` coefficient ``η`` is non-zero (the metaplectic / Gauss case,
    handled for full-period phase by the ``QuadraticPhase`` law 2), or when an axis
    decomposition declines.
    """
    va = read_stb_block(a)
    vb = read_stb_block(b)
    if va is None or vb is None:
        return None
    if va.has_residual or va.has_outliers or vb.has_residual or vb.has_outliers:
        return None  # law 5 territory
    if int(va.dims[1]) != int(vb.dims[0]):
        raise SPFError("STB ridge compose: contracted dimensions must match")
    # A: free = row (dim 0), contract = col (dim 1). B: contract = row (dim 0),
    # free = col (dim 1).
    da = stb_axis_decomposition(va, free_dim=0, contract_dim=1)
    db = stb_axis_decomposition(vb, free_dim=1, contract_dim=0)
    if da is None or db is None:
        return None
    # η = 0 (no k² term) is law 1; a non-zero combined k² is the Gauss/metaplectic
    # regime and is declined here.
    if (da.eta_l + db.eta_l) != 0 or ((da.eta_t + db.eta_t) & SPF32_PHASE_MASK) != 0:
        return None

    from .kernel import _axis_map

    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    n, m, k = da.n_free, db.n_free, int(va.dims[1])
    s0_l = da.klin_l + db.klin_l
    s0_t = da.klin_t + db.klin_t
    return geometric_ridge(
        N=n,
        M=m,
        K=k,
        s0_l=s0_l,
        s0_t=s0_t,
        node_row_l=da.node_l,
        node_row_t=da.node_t,
        node_col_l=db.node_l,
        node_col_t=db.node_t,
        rho_words=da.rho_words,
        sigma_words=db.rho_words,
        out_axis=out_axis,
        in_axis=in_axis,
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        law="stb_block_ridge_geometric",
        notes=(f"N={n}", f"K={k}", f"M={m}", "diag_cauchy_diag"),
    )


# --------------------------------------------------------------------------- #
# Law 7: SEPARABLE_PHASE Gram (RoPE Gram) → ridge with non-uniform phase nodes.
# --------------------------------------------------------------------------- #


def _separable_single_phase_term(kernel: Any) -> tuple[dict[str, Any], tuple[int, ...]] | None:
    """The single ``a[p]·ω[c]`` phase term of a plain SEPARABLE_PHASE block + its origin."""
    if getattr(kernel, "expr", None) is None or kernel.expr.op != "Block":
        return None
    block = kernel.expr.block
    if block is None or int(block.base.profile_id) != int(BaseModelKind.SEPARABLE_PHASE):
        return None
    if int(block.representation.rep_id) != int(Representation.POLAR_SPF32):
        return None
    if int(block.residual.codec_id) != int(ResidualCodecKind.NONE) or block.outliers:
        return None
    params = block.base.params
    terms = list(params.get("phase_terms", ()) or ())
    if len(terms) != 1 or params.get("logmag_terms") or params.get("logmag_bias"):
        return None
    if params.get("phase_bias"):
        return None
    origin = tuple(int(x) for x in (block.origin or (0, 0)))
    return terms[0], origin


def separable_phase_gram(r: Any) -> Any | None:
    """Law 7 — ``Rᴴ R`` for a ``SEPARABLE_PHASE`` block ``R[p, c] = ζ^{a(p)·ω(c)}``.

    With the position ramp ``a(p) = p + c₀`` (unit stride, any offset ``c₀`` from a
    closed-form ``a_origin`` or the block origin), summed over positions ``p``::

        (Rᴴ R)[c, c'] = Σ_p ζ^{(p + c₀)(ω_{c'} − ω_c)}
                      = ζ^{−c₀ω_c} · Σ_p ζ^{p(ω_{c'} − ω_c)} · ζ^{c₀ω_{c'}},

    the Dirichlet kernel of the frequency differences (matmul post §2.3, law 7) —
    a :data:`GEOMETRIC_RIDGE_OP` with non-uniform phase nodes ``−ω_c`` / ``+ω_{c'}``
    and the offset folded into the two diagonals. Declines unless ``R``'s rows are
    the position axis (``axis_i = 0``), its columns the channel axis
    (``axis_j = 1``), and the effective ramp has unit stride.
    """
    found = _separable_single_phase_term(r)
    if found is None:
        return None
    term, origin = found
    if int(term.get("axis_i", -1)) != 0 or int(term.get("axis_j", -1)) != 1:
        return None
    dims = tuple(int(d) for d in r.expr.block.dims)
    if len(dims) != 2:
        return None
    positions, channels = dims
    if positions == 0 or channels == 0:
        return None
    p_abs = np.arange(positions, dtype=np.int64) + (origin[0] if origin else 0)
    c_abs = np.arange(channels, dtype=np.int64) + (origin[1] if len(origin) > 1 else 0)
    if term.get("closed_form"):
        a_eff = p_abs + int(term.get("a_origin", 0))
    else:
        a_vec = np.asarray(term.get("a"), dtype=np.int64).reshape(-1)
        if int(p_abs.max()) >= a_vec.size:
            return None
        a_eff = a_vec[p_abs]
    c0 = int(a_eff[0])
    if not np.array_equal(a_eff, c0 + np.arange(positions, dtype=np.int64)):
        return None  # non-unit-stride ramp: not a geometric sum over p
    omega_vec = np.asarray(term.get("omega"), dtype=np.int64).reshape(-1)
    if int(c_abs.max()) >= omega_vec.size:
        return None
    omega = omega_vec[c_abs]

    from .kernel import _axis_map

    in_spec = _axis_map(r.axes)[r.sig.in_axes[0]]
    from .kernel import AxisSpec

    out_axis = AxisSpec("c_out", channels, role="codomain", basis=in_spec.basis)
    in_axis = AxisSpec("c_in", channels, role="domain", basis=in_spec.basis)
    mod = SPF32_PHASE_MASK + 1
    zeros_l = np.zeros(channels, dtype=np.int16)
    rho_words = pack_words(zeros_l, ((-c0 * omega) % mod).astype(np.uint16))
    sigma_words = pack_words(zeros_l, ((c0 * omega) % mod).astype(np.uint16))
    zeros = np.zeros(channels, dtype=np.int64)
    return geometric_ridge(
        N=channels,
        M=channels,
        K=positions,
        s0_l=0,
        s0_t=0,
        node_row_l=zeros,
        node_row_t=(-omega) % mod,
        node_col_l=zeros,
        node_col_t=omega % mod,
        rho_words=rho_words,
        sigma_words=sigma_words,
        out_axis=out_axis,
        in_axis=in_axis,
        reducer_policy=r.sig.reducer_policy,
        law="separable_phase_gram_dirichlet",
        notes=(f"positions={positions}", f"channels={channels}", f"offset={c0}", "non_uniform_nodes"),
    )


# --------------------------------------------------------------------------- #
# Law 5: residual / outlier operands rewritten additively (matmul post §4).
# --------------------------------------------------------------------------- #


def _header_only_kernel(kernel: Any) -> Any:
    """A copy of an STB Block kernel with residual codec NONE and no outliers (``H``)."""
    from .kernel import SPFBlockMatrix, _axis_map

    block = kernel.expr.block
    from .v3 import BaseModelRef, RepresentationRef, ResidualRef, SPFBlockV3

    h_block = SPFBlockV3(
        dims=tuple(block.dims),
        origin=tuple(block.origin),
        representation=RepresentationRef(int(block.representation.rep_id)),
        base=BaseModelRef(int(block.base.profile_id), dict(block.base.params)),
        residual=ResidualRef(int(ResidualCodecKind.NONE), None),
        outliers={},
    )
    out_axes = tuple(_axis_map(kernel.axes)[n] for n in kernel.sig.out_axes)
    in_axes = tuple(_axis_map(kernel.axes)[n] for n in kernel.sig.in_axes)
    return SPFBlockMatrix.from_block(h_block, out_axes=out_axes, in_axes=in_axes)


def compose_stb_residual(a: Any, b: Any) -> Any | None:
    """Law 5 — ``A B`` when ``A`` or ``B`` carries a residual/outlier patch.

    Rewrite each operand additively ``X = H_X + S_X`` (matmul post §4): ``S`` is
    sparse (residual support + outliers). Then ``AB = H_A H_B + H_A S_B + S_A H_B +
    S_A S_B`` where ``H_A H_B`` is the closed ridge (law 1) and the three sparse
    terms are computed against closed columns of ``H_A`` / rows of ``H_B``, so the
    ``O(NKM)`` contraction is gone (dense cost drops to ``O(NK + KM + N·nnz)``).
    The sparse patch is materialized into words and added through a ``LazySum``
    reducer boundary. Declines when the header product itself has no closed ridge.
    """
    va = read_stb_block(a)
    vb = read_stb_block(b)
    if va is None or vb is None:
        return None
    if not (va.has_residual or va.has_outliers or vb.has_residual or vb.has_outliers):
        return None  # pure header product is law 1
    h_a = _header_only_kernel(a)
    h_b = _header_only_kernel(b)
    ridge = compose_stb_ridge(h_a, h_b)
    if ridge is None:
        return None  # header product has no closed ridge → dense fallback for all

    # Additive corrections S = X − H, computed once over the (sparse) support.
    a_full = decode_words(a.materialize_words())
    h_a_full = decode_words(h_a.materialize_words())
    b_full = decode_words(b.materialize_words())
    h_b_full = decode_words(h_b.materialize_words())
    s_a = a_full - h_a_full
    s_b = b_full - h_b_full

    n = a_full.shape[0]
    m = b_full.shape[1]
    correction = np.zeros((n, m), dtype=np.complex128)
    # H_A S_B: each nonzero column entry of S_B pulls one closed column of H_A.
    kb, jb = np.nonzero(s_b)
    for kk, jj in zip(kb.tolist(), jb.tolist(), strict=True):
        correction[:, jj] += h_a_full[:, kk] * s_b[kk, jj]
    # S_A H_B: each nonzero of S_A pulls one closed row of H_B.
    ia, ka = np.nonzero(s_a)
    for ii, kk in zip(ia.tolist(), ka.tolist(), strict=True):
        correction[ii, :] += s_a[ii, kk] * h_b_full[kk, :]
    # S_A S_B: sparse × sparse, accumulated over the shared contracted index k
    # (never the dense N·K·M product).
    from collections import defaultdict

    sb_by_k: dict[int, list[tuple[int, complex]]] = defaultdict(list)
    for kk, jj in zip(kb.tolist(), jb.tolist(), strict=True):
        sb_by_k[kk].append((jj, complex(s_b[kk, jj])))
    for ii, kk in zip(ia.tolist(), ka.tolist(), strict=True):
        row = sb_by_k.get(kk)
        if not row:
            continue
        sa_val = s_a[ii, kk]
        for jj, sb_val in row:
            correction[ii, jj] += sa_val * sb_val

    nnz = int(ia.size + kb.size)
    from .kernel import SPFBlockMatrix, SPFProof, _axis_map, lazy_sum

    out_axis = _axis_map(a.axes)[a.sig.out_axes[0]]
    in_axis = _axis_map(b.axes)[b.sig.in_axes[0]]
    patch = SPFBlockMatrix.dense(
        correction,
        out_axes=(out_axis,),
        in_axes=(in_axis,),
        reducer_policy=a.sig.reducer_policy or b.sig.reducer_policy,
        proof=SPFProof(
            "stb_residual_sparse_patch",
            False,
            False,
            nnz,
            1,
            notes=(f"nnz={nnz}", "closed_columns_and_rows_of_H"),
        ),
    )
    return lazy_sum((ridge, patch), reducer=a.sig.reducer_policy or b.sig.reducer_policy)


# --------------------------------------------------------------------------- #
# Law 4: STB block matvec via integer-chirp correlation (opt-in).
# --------------------------------------------------------------------------- #


def _triangular(n: np.ndarray) -> np.ndarray:
    return (n * (n - 1)) // 2


def stb_block_matvec_reference(
    operator: Any,
    x: np.ndarray,
    *,
    tile: int = 64,
) -> np.ndarray:
    r"""Law 4 — ``y = A x`` for an order-2 STB block via integer-chirp correlation.

    Using ``ik = T(i+k) − T(i) − T(k)``, ``T(n) = n(n−1)/2`` (integer ticks, no
    half-ticks), the bilinear phase becomes a chirp and the matvec is one Hankel
    correlation of length ``N + K`` plus a sparse residual patch (matmul post §5).
    A magnitude bilinear term (``log_mixed ≠ 0``) makes the chirp overflow, so we
    tile (``t = 64``); phase-only bilinear blocks take the full transform.

    **Status: opt-in reference.** The honest fast route is an FFT of complex
    numbers; here it is a counted ``REFERENCE_CARTESIAN`` path (``numpy.fft``, hence
    the ``_reference`` name, AGENTS.md rule 2) — it does *not*
    beat the dense contraction on the project's carrier rule, so it is a direct
    helper, **not** auto-registered on ``apply``. It returns the dense ``y`` words;
    the caller decides. Matches the dense product to SPF rounding (measured).
    """
    va = read_stb_block(operator)
    if va is None:
        raise SPFError("stb_block_matvec requires a dense order-2 STB block operator")
    xw = np.asarray(x, dtype=np.complex128).reshape(-1)
    n, k = int(va.dims[0]), int(va.dims[1])
    if xw.size != k:
        raise SPFError("stb_block_matvec vector length must equal the contracted dim")
    da = stb_axis_decomposition(va, free_dim=0, contract_dim=1)
    if da is None or da.eta_l != 0 or (da.eta_t & SPF32_PHASE_MASK) != 0:
        raise SPFError("stb_block_matvec needs η = 0 (no k² term) on the contracted axis")

    ln2 = np.log(2.0)
    tp = 2.0 * np.pi

    def eps(l_ticks: np.ndarray, t_ticks: np.ndarray) -> np.ndarray:
        return np.asarray(l_ticks, float) * ln2 / 256.0 + 1j * np.asarray(t_ticks, float) * tp / 65536.0

    # Diagonals: ρ(i) on rows, σ'(k) on the contracted axis (the k-linear part).
    i_idx = np.arange(n)
    k_idx = np.arange(k)
    d_row = np.exp(eps(unpack_L(da.rho_words).astype(np.int64), unpack_theta(da.rho_words).astype(np.int64)))
    # σ'(k): the k-linear part of the exponent (η = 0, so no k²). The bilinear i·k
    # coupling is carried by the chirp z^{ik} below, not here.
    sig_l = da.klin_l * k_idx
    sig_t = da.klin_t * k_idx
    d_col = np.exp(eps(sig_l, sig_t))
    z_l, z_t = int(va.l_bilinear), int(va.t_bilinear)

    tri = _triangular
    if z_l == 0:
        # Phase-only bilinear: one length N+K chirp correlation.
        nn = np.arange(n + k)
        h = np.exp(eps(z_l * tri(nn), z_t * tri(nn)))
        u = d_col * xw * np.exp(eps(-z_l * tri(k_idx), -z_t * tri(k_idx)))
        l2 = 1 << int(np.ceil(np.log2(n + 2 * k)))
        corr = np.fft.ifft(np.fft.fft(h, l2) * np.fft.fft(u[::-1], l2))[k - 1 : k - 1 + n]
        y = d_row * np.exp(eps(-z_l * tri(i_idx), -z_t * tri(i_idx))) * corr
    else:
        # Magnitude bilinear: tile so each tile's chirp stays in the float/SPF range.
        t = int(tile)
        y = np.zeros(n, dtype=np.complex128)
        for i0 in range(0, n, t):
            rows = min(t, n - i0)
            for k0 in range(0, k, t):
                cols = min(t, k - k0)
                b = np.arange(cols)
                u = (d_col[k0 : k0 + cols] * xw[k0 : k0 + cols]) * np.exp(eps(z_l * i0 * b, z_t * i0 * b))
                nn = np.arange(rows + cols)
                h = np.exp(eps(z_l * tri(nn), z_t * tri(nn)))
                v = u * np.exp(eps(-z_l * tri(b), -z_t * tri(b)))
                l2 = 1 << int(np.ceil(np.log2(rows + 2 * cols)))
                c = np.fft.ifft(np.fft.fft(h, l2) * np.fft.fft(v[::-1], l2))[cols - 1 : cols - 1 + rows]
                a_idx = np.arange(rows)
                tile_y = (
                    np.exp(eps(z_l * (i0 * k0 + a_idx * k0), z_t * (i0 * k0 + a_idx * k0)))
                    * np.exp(eps(-z_l * tri(a_idx), -z_t * tri(a_idx)))
                    * c
                )
                y[i0 : i0 + rows] += d_row[i0 : i0 + rows] * tile_y

    return np.asarray(encode_words(y), dtype=np.uint32)


# --------------------------------------------------------------------------- #
# Algebra registration hooks.
# --------------------------------------------------------------------------- #


class STBBlockAlgebra:
    """``Block ∘ Block`` closed laws: ridge (law 1/6) and residual rewrite (law 5)."""

    kind = "Block"

    def compose(self, a: Any, b: Any) -> Any | None:
        closed = compose_stb_ridge(a, b)
        if closed is not None:
            return closed
        return compose_stb_residual(a, b)

    def adjoint(self, a: Any) -> Any | None:
        return None

    def inverse(self, a: Any) -> Any | None:
        return None

    def power(self, a: Any, k: int) -> Any | None:
        return None


class GeometricRidgeAlgebra:
    """A materialized ridge has no further closed law here (folds via dense)."""

    kind = GEOMETRIC_RIDGE_OP

    def compose(self, a: Any, b: Any) -> Any | None:
        return None

    def adjoint(self, a: Any) -> Any | None:
        return None

    def inverse(self, a: Any) -> Any | None:
        return None

    def power(self, a: Any, k: int) -> Any | None:
        return None
