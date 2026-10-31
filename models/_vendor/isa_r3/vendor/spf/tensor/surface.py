"""Build a sparse 2-D character surface as a first-class SPF ``ExecutionPlan``.

A trained transformer's per-class logit surface, FFT'd over the ``(a, b)`` input
grid, concentrates on a few 2-D group characters
``χ_{(u,v)}(a,b) = exp(2πi(u·a + v·b)/N)``. PAN recovers those atoms (learned-model
analysis — *upstream*, never here); this module is the SPF-side consumer that turns
the *recovered* sparse surface ``Σ_j c_j·χ_{(u_j,v_j)}(a,b)`` into a real
``ExecutionPlan`` and lowers it, instead of the hand-built op-count dict the probe
emits.

Each class is a **bank** of ``k`` separable character binds plus a **bounded sparse
reduce**:

* separability is BCCB / relationship #8 — the 2-D character factors into two 1-D
  characters ``χ_u(a)·χ_v(b)``, so a class's ``k`` atoms are *one*
  ``elementwise_mul`` of two stacked gathers;
* the plan-time :class:`~spf.tensor.rewrite.CharacterBankBind` rule collapses each
  ``k``-atom bank to a single ``affine_bank_mod`` gather (the ``u==v`` atoms are
  sum-characters, the rest affine — the same algebra as the scalar bind);
* the per-class ``reduce`` sums the ``k`` scaled atoms; its cost scales ``∝ k``
  (``n_terms``) so a denser surface is priced honestly.

The boundary mirrors :mod:`spf.tensor.character`: recovery is model analysis,
*rewriting is algebra after the plan exists*. This module never recovers atoms.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# ReducerPolicy/ReducerSpec name the bounded sparse reduce's accumulation domain;
# LowRankOperator carries the rank-factored coefficient matrix of the vector surface.
from .core import LowRankOperator, ReducerPolicy, ReducerSpec, encode_words
from .rewrite import (
    Provenance,
    RewriteContext,
    rewrite_plan,
    rewrite_then_lower,
)
from .v2 import ExecutionPlan, ExecutionStep

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class SparseCharacterSurface:
    """A recovered sparse 2-D character surface over ``Z_modulus``.

    ``class_atoms[c]`` is class ``c``'s tuple of ``(u, v)`` character atoms; the
    matching complex coefficients are runtime words, supplied at execution (so the
    plan/structure does not bake them). Build with :meth:`from_lists`.
    """

    modulus: int
    class_atoms: tuple[tuple[tuple[int, int], ...], ...]

    @classmethod
    def from_lists(
        cls, modulus: int, class_atoms: Sequence[Sequence[tuple[int, int]]]
    ) -> SparseCharacterSurface:
        n = int(modulus)
        classes = tuple(
            tuple((int(u) % n, int(v) % n) for (u, v) in atoms) for atoms in class_atoms
        )
        if not classes:
            raise ValueError("a surface needs at least one class")
        if any(len(atoms) == 0 for atoms in classes):
            raise ValueError("every class needs at least one atom")
        return cls(modulus=n, class_atoms=classes)

    # --------------------------------------------------------------------- #
    # Plan construction
    # --------------------------------------------------------------------- #

    def to_execution_plan(self) -> tuple[ExecutionPlan, RewriteContext]:
        """Build the un-collapsed bank-bind + reduce plan and its rewrite context.

        Per class: two stacked character gathers (``index_coeff_bank`` = the atom's
        ``u``/``v``), their bind (an ``elementwise_mul`` flagged Cartesian-fallback),
        a per-atom coefficient ``diagonal_scale``, and a ``reduce`` over the atom
        axis. The provenance vouches each gather output an exact DFT-row *bank* so
        :class:`~spf.tensor.rewrite.CharacterBankBind` fires.
        """
        n = self.modulus
        steps: list[ExecutionStep] = []
        provenance: dict[str, Provenance] = {}
        for c, atoms in enumerate(self.class_atoms):
            u = [int(uu) for (uu, _vv) in atoms]
            v = [int(vv) for (_uu, vv) in atoms]
            na, nb, bound = f"chi_a_{c}", f"chi_b_{c}", f"chi_{c}"
            scaled, out = f"surf_terms_{c}", f"surface_{c}"
            steps.append(
                ExecutionStep(
                    "gather", ("char_table", "input_a"), (na,),
                    params={"axis": 0, "index_coeff_bank": u, "modulus": n},
                )
            )
            steps.append(
                ExecutionStep(
                    "gather", ("char_table", "input_b"), (nb,),
                    params={"axis": 0, "index_coeff_bank": v, "modulus": n},
                )
            )
            provenance[na] = Provenance(
                kind="dft_row_bank", N=n, index_bank=tuple(u), axis_coeff_bank=tuple(u), exact=True
            )
            provenance[nb] = Provenance(
                kind="dft_row_bank", N=n, index_bank=tuple(v), axis_coeff_bank=tuple(v), exact=True
            )
            steps.append(
                ExecutionStep(
                    "elementwise_mul", (na, nb), (bound,),
                    params={"cartesian_fallback": True, "purpose": "character_bank_bind"},
                )
            )
            steps.append(
                ExecutionStep(
                    "diagonal_scale", (bound, f"coeff_{c}"), (scaled,),
                    params={"purpose": "surface_coefficient", "class": c},
                )
            )
            steps.append(
                ExecutionStep(
                    "reduce", (scaled,), (out,),
                    reducer=ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN),
                    params={"axis": 0, "n_terms": len(atoms), "purpose": "surface_class_sum"},
                )
            )
        plan = ExecutionPlan(
            tuple(steps),
            factorization_metadata={
                "operator": "SparseCharacterSurfaceOperator",
                "modulus": n,
                "classes": len(self.class_atoms),
                "atoms_per_class": [len(atoms) for atoms in self.class_atoms],
            },
        )
        # Legacy recovered-character builders optimize the ideal group function.
        # Their rounded tables do not certify packed-step equivalence.
        return plan, RewriteContext(provenance=provenance, rounding_contract="algebraic_output")

    def primitive_character_table(self) -> np.ndarray:
        """SPF words for the primitive character row ``χ_1(m) = e^{2πi·m/N}``.

        The bank gathers index this single row at ``(u·a) mod N`` to realise every
        ``χ_u``; one table backs the whole surface.
        """
        m = np.arange(self.modulus, dtype=np.float64)
        return encode_words(np.exp(1j * _TWO_PI * m / self.modulus))


@dataclass(frozen=True)
class SparseCharacterSurfaceReport:
    """Result of building + rewriting + lowering a sparse character surface."""

    modulus: int
    n_classes: int
    atoms_per_class: tuple[int, ...]
    plan_steps_before: int
    plan_steps_after: int
    rewrites_fired: tuple[str, ...]
    banks_collapsed: int
    reduce_count: int
    reduce_terms: tuple[int, ...]
    pl_dispatch_before: int
    pl_dispatch_after: int
    host_dispatch_before: int
    host_dispatch_after: int
    profile_before: str | None
    profile_after: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "SparseCharacterSurfaceOperator",
            "modulus": self.modulus,
            "n_classes": self.n_classes,
            "atoms_per_class": list(self.atoms_per_class),
            "plan_steps_before": self.plan_steps_before,
            "plan_steps_after": self.plan_steps_after,
            "rewrites_fired": list(self.rewrites_fired),
            "banks_collapsed": self.banks_collapsed,
            "reduce_count": self.reduce_count,
            "reduce_terms": list(self.reduce_terms),
            "pl_dispatch_before": self.pl_dispatch_before,
            "pl_dispatch_after": self.pl_dispatch_after,
            "host_dispatch_before": self.host_dispatch_before,
            "host_dispatch_after": self.host_dispatch_after,
            "profile_before": self.profile_before,
            "profile_after": self.profile_after,
        }


def compile_sparse_character_surface(
    surface: SparseCharacterSurface,
) -> SparseCharacterSurfaceReport:
    """Build, rewrite, and lower a sparse character surface into a report."""
    plan, ctx = surface.to_execution_plan()
    optimized, matches, _ = rewrite_plan(plan, ctx)
    _, delta = rewrite_then_lower(plan, ctx)
    fired = tuple(m.name for m in matches)
    reduce_terms = tuple(
        int(s.params.get("n_terms", 1)) for s in optimized.steps if s.op == "reduce"
    )
    return SparseCharacterSurfaceReport(
        modulus=surface.modulus,
        n_classes=len(surface.class_atoms),
        atoms_per_class=tuple(len(atoms) for atoms in surface.class_atoms),
        plan_steps_before=len(plan.steps),
        plan_steps_after=len(optimized.steps),
        rewrites_fired=fired,
        banks_collapsed=sum(1 for n in fired if n == "character_bank_bind"),
        reduce_count=len(reduce_terms),
        reduce_terms=reduce_terms,
        pl_dispatch_before=delta.pl_dispatch_before,
        pl_dispatch_after=delta.pl_dispatch_after,
        host_dispatch_before=delta.host_dispatch_before,
        host_dispatch_after=delta.host_dispatch_after,
        profile_before=delta.profile_before,
        profile_after=delta.profile_after,
    )


# ------------------------------------------------------------------------- #
# Shared class-shift surface — the c -> -u·c class-phase law
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SharedCharacterClassShiftSurface:
    """A surface whose ``n_classes`` outputs share one atom set + a class ramp.

    PAN's ``SharedAdditionClassShiftKernel`` measures that across the class axis
    every frequency-``u`` coefficient obeys one shared law
    ``coeff_u(c) = base_u·exp(-2πi·u·c/N) = base_u·χ_{-u}(c)`` — the class index
    ``c`` enters only as a deterministic phase ramp. So the full surface is a
    *three-axis* character bind ::

        S[c, a, b] = Σ_j  base_j · χ_{u_j}(a) · χ_{v_j}(b) · χ_{(-u_j) mod N}(c)

    Where :class:`SparseCharacterSurface` stores ``n_classes`` independent banks
    (``n_classes·k`` coefficient words), this stores **one shared** ``base_j`` bank
    (``k`` words) and lets the class axis ride a third bound character — param count
    ``3k + n_classes`` vs ``n_classes·k``.

    The structural keystone is *this PR's machinery*: the ``(a, b)`` pair is one
    :class:`~spf.tensor.rewrite.CharacterBankBind` (collapses to ``affine_bank_mod``,
    identical to :class:`SparseCharacterSurface`); the class character is a third
    DFT-row bank on the coupled generator ``(-u_j) mod N``, multiplied in and reduced
    over the atom axis. No new rewrite rule.

    The coupling ``class_gen[j] ≡ (-u_j) mod N`` is what distinguishes a true
    class-shift from an arbitrary 3-character mix — it is enforced *by construction*
    and validated by :meth:`from_class_coefficients`, which declines (returns
    ``None``) when the measured ramp/amplitude law does not hold, so the caller falls
    back to per-class :class:`SparseCharacterSurface`. Recovery (FFT, fitting
    ``base_u``, measuring the ramp residual) stays upstream in PAN; this only builds /
    rewrites / lowers the *recovered* operator.
    """

    modulus: int
    n_classes: int
    atoms: tuple[tuple[int, int], ...]  # shared (u, v) across every class

    @classmethod
    def from_lists(
        cls,
        modulus: int,
        n_classes: int,
        atoms: Sequence[tuple[int, int]],
    ) -> SharedCharacterClassShiftSurface:
        n = int(modulus)
        shared = tuple((int(u) % n, int(v) % n) for (u, v) in atoms)
        if int(n_classes) < 1:
            raise ValueError("a surface needs at least one class")
        if not shared:
            raise ValueError("a shared surface needs at least one atom")
        return cls(modulus=n, n_classes=int(n_classes), atoms=shared)

    @classmethod
    def from_class_coefficients(
        cls,
        modulus: int,
        atoms: Sequence[tuple[int, int]],
        class_coeffs: np.ndarray,
        *,
        phase_tol: float = 1e-3,
        amp_tol: float = 1e-3,
    ) -> SharedCharacterClassShiftSurface | None:
        """The honesty gate: emit the shared operator *only* when the law holds.

        ``class_coeffs`` is the recovered ``(n_classes, k)`` complex coefficient
        matrix ``coeff_{u_j}(c)``. For each atom ``j`` this checks the two metrics
        PAN's kernel uses — phase **linearity** against the predicted ramp
        ``χ_{-u_j}(c)`` (max angular residual ``≤ phase_tol``) and amplitude
        **constancy** across classes (coefficient of variation ``≤ amp_tol``) — using
        the wrap-safe complex ratio ``coeff(c)/coeff(0)`` so the ``-2π·u·c/N`` slope is
        compared without unwrapping. Returns the operator when *every* atom passes,
        else ``None`` (caller falls back to :class:`SparseCharacterSurface`).
        """
        n = int(modulus)
        shared = tuple((int(u) % n, int(v) % n) for (u, v) in atoms)
        coeffs = np.asarray(class_coeffs, dtype=np.complex128)
        n_classes = coeffs.shape[0]
        if coeffs.ndim != 2 or coeffs.shape[1] != len(shared):
            raise ValueError(
                f"class_coeffs must be (n_classes, k={len(shared)}); got {coeffs.shape}"
            )
        if n_classes < 1 or len(shared) == 0:
            return None
        c_idx = np.arange(n_classes)
        for j, (u, _v) in enumerate(shared):
            col = coeffs[:, j]
            amp = np.abs(col)
            mean_amp = float(amp.mean())
            if mean_amp <= 0.0:
                return None
            if float(amp.std()) / mean_amp > amp_tol:  # amplitude_cv gate
                return None
            base = col[0]
            if base == 0:
                return None
            predicted = np.exp(-1j * _TWO_PI * (u * c_idx) / n)  # χ_{-u}(c)
            residual = np.angle((col / base) / predicted)  # wrap-safe ramp residual
            if float(np.max(np.abs(residual))) > phase_tol:  # phase_linearity gate
                return None
        return cls(modulus=n, n_classes=int(n_classes), atoms=shared)

    # --------------------------------------------------------------------- #
    # Plan construction
    # --------------------------------------------------------------------- #

    def to_execution_plan(self) -> tuple[ExecutionPlan, RewriteContext]:
        """Build the three-axis bind plan + its rewrite context.

        Emits sample-indexed gathers for ``input_a``/``input_b``/``input_c``: the
        ``(a, b)`` pair is a :class:`~spf.tensor.rewrite.CharacterBankBind` (collapses
        to one ``affine_bank_mod`` gather, exactly as :class:`SparseCharacterSurface`);
        the class character ``χ_{(-u_j) mod N}(c)`` is a third DFT-row bank multiplied
        in, then a per-atom ``base_j`` ``diagonal_scale`` and a ``reduce`` over the
        atom axis. The provenance vouches the ``(a, b)`` gathers as an exact DFT-row
        bank so the bind fires; the class gather rides the coupled generator
        ``(-u_j) mod N``.
        """
        n = self.modulus
        u = [int(uu) for (uu, _vv) in self.atoms]
        v = [int(vv) for (_uu, vv) in self.atoms]
        neg_u = [(-uu) % n for uu in u]  # class generator coupled to the a-axis
        k = len(self.atoms)
        steps: list[ExecutionStep] = []
        provenance: dict[str, Provenance] = {}
        steps.append(
            ExecutionStep(
                "gather", ("char_table", "input_a"), ("chi_a",),
                params={"axis": 0, "index_coeff_bank": u, "modulus": n},
            )
        )
        steps.append(
            ExecutionStep(
                "gather", ("char_table", "input_b"), ("chi_b",),
                params={"axis": 0, "index_coeff_bank": v, "modulus": n},
            )
        )
        provenance["chi_a"] = Provenance(
            kind="dft_row_bank", N=n, index_bank=tuple(u), axis_coeff_bank=tuple(u), exact=True
        )
        provenance["chi_b"] = Provenance(
            kind="dft_row_bank", N=n, index_bank=tuple(v), axis_coeff_bank=tuple(v), exact=True
        )
        steps.append(
            ExecutionStep(
                "elementwise_mul", ("chi_a", "chi_b"), ("chi_ab",),
                params={"cartesian_fallback": True, "purpose": "character_bank_bind"},
            )
        )
        # Third bound character: the class index on the coupled generator (-u_j).
        steps.append(
            ExecutionStep(
                "gather", ("char_table", "input_c"), ("chi_c",),
                params={"axis": 0, "index_coeff_bank": neg_u, "modulus": n},
            )
        )
        provenance["chi_c"] = Provenance(
            kind="dft_row_bank", N=n, index_bank=tuple(neg_u), axis_coeff_bank=tuple(neg_u), exact=True
        )
        steps.append(
            ExecutionStep(
                "elementwise_mul", ("chi_ab", "chi_c"), ("chi_abc",),
                params={"cartesian_fallback": True, "purpose": "class_shift_bind"},
            )
        )
        steps.append(
            ExecutionStep(
                "diagonal_scale", ("chi_abc", "base"), ("surf_terms",),
                params={"purpose": "shared_base_coefficient"},
            )
        )
        steps.append(
            ExecutionStep(
                "reduce", ("surf_terms",), ("class_shift_surface",),
                reducer=ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN),
                params={"axis": 0, "n_terms": k, "purpose": "shared_surface_sum"},
            )
        )
        plan = ExecutionPlan(
            tuple(steps),
            factorization_metadata={
                "operator": "SharedCharacterClassShiftSurfaceOperator",
                "modulus": n,
                "n_classes": self.n_classes,
                "shared_atoms": k,
                "class_generator_couple": "neg_u",
            },
        )
        # Legacy recovered-character builders optimize the ideal group function.
        # Their rounded tables do not certify packed-step equivalence.
        return plan, RewriteContext(provenance=provenance, rounding_contract="algebraic_output")

    def primitive_character_table(self) -> np.ndarray:
        """SPF words for ``χ_1(m) = e^{2πi·m/N}`` — one row backs all three axes."""
        m = np.arange(self.modulus, dtype=np.float64)
        return encode_words(np.exp(1j * _TWO_PI * m / self.modulus))


@dataclass(frozen=True)
class SharedCharacterClassShiftReport:
    """Result of building + rewriting + lowering a shared class-shift surface."""

    modulus: int
    n_classes: int
    shared_atoms: int
    plan_steps_before: int
    plan_steps_after: int
    rewrites_fired: tuple[str, ...]
    banks_collapsed: int
    reduce_terms: tuple[int, ...]
    param_words_shared: int  # 3k + n_classes (this operator)
    param_words_per_class: int  # n_classes·k (the SparseCharacterSurface fallback)
    pl_dispatch_before: int
    pl_dispatch_after: int
    host_dispatch_before: int
    host_dispatch_after: int
    profile_before: str | None
    profile_after: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "SharedCharacterClassShiftSurfaceOperator",
            "modulus": self.modulus,
            "n_classes": self.n_classes,
            "shared_atoms": self.shared_atoms,
            "plan_steps_before": self.plan_steps_before,
            "plan_steps_after": self.plan_steps_after,
            "rewrites_fired": list(self.rewrites_fired),
            "banks_collapsed": self.banks_collapsed,
            "reduce_terms": list(self.reduce_terms),
            "param_words_shared": self.param_words_shared,
            "param_words_per_class": self.param_words_per_class,
            "pl_dispatch_before": self.pl_dispatch_before,
            "pl_dispatch_after": self.pl_dispatch_after,
            "host_dispatch_before": self.host_dispatch_before,
            "host_dispatch_after": self.host_dispatch_after,
            "profile_before": self.profile_before,
            "profile_after": self.profile_after,
        }


def compile_shared_character_class_shift_surface(
    surface: SharedCharacterClassShiftSurface,
) -> SharedCharacterClassShiftReport:
    """Build, rewrite, and lower a shared class-shift surface into a report."""
    plan, ctx = surface.to_execution_plan()
    optimized, matches, _ = rewrite_plan(plan, ctx)
    _, delta = rewrite_then_lower(plan, ctx)
    fired = tuple(m.name for m in matches)
    reduce_terms = tuple(
        int(s.params.get("n_terms", 1)) for s in optimized.steps if s.op == "reduce"
    )
    k = len(surface.atoms)
    return SharedCharacterClassShiftReport(
        modulus=surface.modulus,
        n_classes=surface.n_classes,
        shared_atoms=k,
        plan_steps_before=len(plan.steps),
        plan_steps_after=len(optimized.steps),
        rewrites_fired=fired,
        banks_collapsed=sum(1 for n in fired if n == "character_bank_bind"),
        reduce_terms=reduce_terms,
        param_words_shared=3 * k + surface.n_classes,
        param_words_per_class=surface.n_classes * k,
        pl_dispatch_before=delta.pl_dispatch_before,
        pl_dispatch_after=delta.pl_dispatch_after,
        host_dispatch_before=delta.host_dispatch_before,
        host_dispatch_after=delta.host_dispatch_after,
        profile_before=delta.profile_before,
        profile_after=delta.profile_after,
    )


# ------------------------------------------------------------------------- #
# Low-rank character vector surface — the MLP / attention-head output
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LowRankCharacterVectorSurface:
    """A vector-valued character surface whose coefficient matrix is low-rank.

    PAN's highest *validated* transformer win is not a scalar surface but a
    **vector** one: an MLP / attention-head output is a ``d_out``-dim residual
    vector, and across those output dims the per-atom coefficient is **low-rank** ::

        out[d, a, b] = Σ_j  C[d, j] · χ_{(u_j,v_j)}(a, b),     C = U·diag(λ)·Vᴴ  (rank r)

    (PAN `SharedLowRankAffineCharacterVectorOperator` / `MLPOutputLowRankVectorSurface`,
    rank-16, acc 0.998904 — beats the dense transformer.) This is exactly the
    sparse-surface character **bank** (it still collapses through
    :class:`~spf.tensor.rewrite.CharacterBankBind`) composed with a
    :class:`~spf.tensor.core.LowRankOperator` on the atom axis: the recovered
    rank-``r`` coefficient stores ``r·(d_out + k)`` words instead of the dense
    ``d_out·k``.

    The plan is evaluated per input token ``(a, b)``: the ``k``-atom bank gathers to
    one ``affine_bank_mod`` row, the ``low_rank_apply`` contracts that ``k``-vector to
    the ``d_out`` output vector. The factors ``U``/``V``/``λ`` *define* the operator
    (recovered upstream in PAN), baked into the plan like the character generators of
    :class:`~spf.tensor.character.CharacterBasisOperator`. Recovery stays in PAN; this
    only builds / rewrites / lowers.
    """

    modulus: int
    atoms: tuple[tuple[int, int], ...]  # shared (u, v) across every output dim
    out_dim: int
    rank: int
    u_factors: np.ndarray  # SPF words, shape [rank, out_dim]  (LowRankOperator u)
    v_factors: np.ndarray  # SPF words, shape [rank, k]        (LowRankOperator v)
    lambdas: np.ndarray  # SPF words, shape [rank]

    @classmethod
    def from_factors(
        cls,
        modulus: int,
        atoms: Sequence[tuple[int, int]],
        u: np.ndarray,
        v: np.ndarray,
        lambdas: np.ndarray | None = None,
    ) -> LowRankCharacterVectorSurface:
        """Build from explicit complex low-rank factors.

        ``u`` is ``[out_dim, rank]`` and ``v`` is ``[rank, k]`` so the realised
        coefficient is ``C[d, j] = Σ_r λ_r·u[d, r]·conj(v[r, j])`` — the
        :class:`~spf.tensor.core.LowRankOperator` contraction. ``lambdas`` defaults to
        ones (fold the scale into ``u``).
        """
        n = int(modulus)
        shared = tuple((int(uu) % n, int(vv) % n) for (uu, vv) in atoms)
        u = np.asarray(u, dtype=np.complex128)
        v = np.asarray(v, dtype=np.complex128)
        out_dim, rank = u.shape
        if v.shape[0] != rank or v.shape[1] != len(shared):
            raise ValueError(f"v must be [rank={rank}, k={len(shared)}]; got {v.shape}")
        lam = np.ones(rank) if lambdas is None else np.asarray(lambdas, dtype=np.complex128)
        if lam.shape != (rank,):
            raise ValueError(f"lambdas must be [rank={rank}]; got {lam.shape}")
        return cls(
            modulus=n,
            atoms=shared,
            out_dim=int(out_dim),
            rank=int(rank),
            u_factors=encode_words(u.T),  # LowRankOperator wants [rank, out]
            v_factors=encode_words(v),
            lambdas=encode_words(lam),
        )

    @classmethod
    def from_coefficient_matrix(
        cls,
        modulus: int,
        atoms: Sequence[tuple[int, int]],
        coefficient_matrix: np.ndarray,
        rank: int,
    ) -> LowRankCharacterVectorSurface:
        """Truncated-SVD factor a dense ``[out_dim, k]`` coefficient matrix to rank ``r``.

        The SPF-side helper for when PAN hands the dense per-dim coefficient matrix
        rather than pre-factored ``U``/``V``: keeps the top-``r`` singular triplet so
        ``C ≈ U_r·diag(σ_r)·conj(V_r)ᴴ`` matches the
        :class:`~spf.tensor.core.LowRankOperator` contraction exactly.
        """
        c = np.asarray(coefficient_matrix, dtype=np.complex128)
        if c.ndim != 2 or c.shape[1] != len(tuple(atoms)):
            raise ValueError(f"coefficient_matrix must be [out_dim, k={len(tuple(atoms))}]; got {c.shape}")
        r = int(rank)
        if not 1 <= r <= min(c.shape):
            raise ValueError(f"rank must be in [1, {min(c.shape)}]; got {r}")
        u_full, s_full, vh_full = np.linalg.svd(c, full_matrices=False)
        u_r = u_full[:, :r]  # [out, r]
        s_r = s_full[:r]  # [r]
        vh_r = vh_full[:r, :]  # [r, k]; C[d,j] = Σ_r u[d,r]·s[r]·vh[r,j]
        return cls.from_factors(modulus, atoms, u_r, np.conj(vh_r), lambdas=s_r)

    # --------------------------------------------------------------------- #
    # Plan construction
    # --------------------------------------------------------------------- #

    def to_execution_plan(self) -> tuple[ExecutionPlan, RewriteContext]:
        """Build the bank-bind + low-rank-apply plan and its rewrite context.

        Two stacked character gathers (the ``k``-atom bank, ``index_coeff_bank`` =
        the atom ``u``/``v``) bind via :class:`~spf.tensor.rewrite.CharacterBankBind`
        into one ``affine_bank_mod`` gather; the ``low_rank_apply`` then contracts the
        ``k``-vector through ``U``/``V``/``λ`` to the ``d_out`` output vector. Evaluate
        per token: ``input_a``/``input_b`` are scalar (0-d) indices.
        """
        n = self.modulus
        u = [int(uu) for (uu, _vv) in self.atoms]
        v = [int(vv) for (_uu, vv) in self.atoms]
        op = LowRankOperator(
            u_factors=self.u_factors, v_factors=self.v_factors, lambdas=self.lambdas
        )
        steps: list[ExecutionStep] = [
            ExecutionStep(
                "gather", ("char_table", "input_a"), ("chi_a",),
                params={"axis": 0, "index_coeff_bank": u, "modulus": n},
            ),
            ExecutionStep(
                "gather", ("char_table", "input_b"), ("chi_b",),
                params={"axis": 0, "index_coeff_bank": v, "modulus": n},
            ),
            ExecutionStep(
                "elementwise_mul", ("chi_a", "chi_b"), ("chi",),
                params={"cartesian_fallback": True, "purpose": "character_bank_bind"},
            ),
            ExecutionStep(
                "low_rank_apply", ("chi",), ("vector_surface",),
                reducer=ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN),
                params={"operator": op, "purpose": "low_rank_character_coefficient"},
            ),
        ]
        provenance = {
            "chi_a": Provenance(
                kind="dft_row_bank", N=n, index_bank=tuple(u), axis_coeff_bank=tuple(u), exact=True
            ),
            "chi_b": Provenance(
                kind="dft_row_bank", N=n, index_bank=tuple(v), axis_coeff_bank=tuple(v), exact=True
            ),
        }
        plan = ExecutionPlan(
            tuple(steps),
            factorization_metadata={
                "operator": "LowRankCharacterVectorSurfaceOperator",
                "modulus": n,
                "out_dim": self.out_dim,
                "atoms": len(self.atoms),
                "rank": self.rank,
            },
        )
        # Legacy recovered-character builders optimize the ideal group function.
        # Their rounded tables do not certify packed-step equivalence.
        return plan, RewriteContext(provenance=provenance, rounding_contract="algebraic_output")

    def primitive_character_table(self) -> np.ndarray:
        """SPF words for ``χ_1(m) = e^{2πi·m/N}`` — one row backs the whole bank."""
        m = np.arange(self.modulus, dtype=np.float64)
        return encode_words(np.exp(1j * _TWO_PI * m / self.modulus))

    @property
    def param_words_low_rank(self) -> int:
        """Word count of the rank-factored coefficient: ``r·(out_dim + k) + r``."""
        return self.rank * (self.out_dim + len(self.atoms)) + self.rank

    @property
    def param_words_dense(self) -> int:
        """Word count of the dense per-dim coefficient matrix: ``out_dim·k``."""
        return self.out_dim * len(self.atoms)


@dataclass(frozen=True)
class LowRankCharacterVectorReport:
    """Result of building + rewriting + lowering a low-rank character vector surface."""

    modulus: int
    out_dim: int
    n_atoms: int
    rank: int
    plan_steps_before: int
    plan_steps_after: int
    rewrites_fired: tuple[str, ...]
    banks_collapsed: int
    param_words_low_rank: int
    param_words_dense: int
    pl_dispatch_before: int
    pl_dispatch_after: int
    host_dispatch_before: int
    host_dispatch_after: int
    profile_before: str | None
    profile_after: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "LowRankCharacterVectorSurfaceOperator",
            "modulus": self.modulus,
            "out_dim": self.out_dim,
            "n_atoms": self.n_atoms,
            "rank": self.rank,
            "plan_steps_before": self.plan_steps_before,
            "plan_steps_after": self.plan_steps_after,
            "rewrites_fired": list(self.rewrites_fired),
            "banks_collapsed": self.banks_collapsed,
            "param_words_low_rank": self.param_words_low_rank,
            "param_words_dense": self.param_words_dense,
            "pl_dispatch_before": self.pl_dispatch_before,
            "pl_dispatch_after": self.pl_dispatch_after,
            "host_dispatch_before": self.host_dispatch_before,
            "host_dispatch_after": self.host_dispatch_after,
            "profile_before": self.profile_before,
            "profile_after": self.profile_after,
        }


def compile_low_rank_character_vector_surface(
    surface: LowRankCharacterVectorSurface,
) -> LowRankCharacterVectorReport:
    """Build, rewrite, and lower a low-rank character vector surface into a report."""
    plan, ctx = surface.to_execution_plan()
    optimized, matches, _ = rewrite_plan(plan, ctx)
    _, delta = rewrite_then_lower(plan, ctx)
    fired = tuple(m.name for m in matches)
    return LowRankCharacterVectorReport(
        modulus=surface.modulus,
        out_dim=surface.out_dim,
        n_atoms=len(surface.atoms),
        rank=surface.rank,
        plan_steps_before=len(plan.steps),
        plan_steps_after=len(optimized.steps),
        rewrites_fired=fired,
        banks_collapsed=sum(1 for n in fired if n == "character_bank_bind"),
        param_words_low_rank=surface.param_words_low_rank,
        param_words_dense=surface.param_words_dense,
        pl_dispatch_before=delta.pl_dispatch_before,
        pl_dispatch_after=delta.pl_dispatch_after,
        host_dispatch_before=delta.host_dispatch_before,
        host_dispatch_after=delta.host_dispatch_after,
        profile_before=delta.profile_before,
        profile_after=delta.profile_after,
    )


# ------------------------------------------------------------------------- #
# QKV-free multi-head attention — per-head low-rank fanout, one fused operator
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LowRankAttentionHead:
    """One attention head's recovered low-rank character generator.

    The validated PAN structure (`TransformerQKVFreeLowRankAttentionHeadOutputPatch`,
    per-head rank-8, acc 0.989–0.996) replaces a head's *output* — the
    ``d_model``-vector it contributes to the residual stream — by a low-rank
    character surface that reads only the input tokens ``(a, b)``, never Q/K/V or the
    softmax. ``out_proj`` is folded into the left factor upstream, so ``u`` is already
    ``[d_model, rank]`` (the head-output form, not the ``head_dim`` context form)::

        head_out[d, a, b] = Σ_j  C_h[d, j]·χ_{(u_j,v_j)}(a, b),   C_h = U_h·diag(λ_h)·V_hᴴ
    """

    atoms: tuple[tuple[int, int], ...]
    rank: int
    u_factors: np.ndarray  # complex [d_model, rank]  (out_proj already folded)
    v_factors: np.ndarray  # complex [rank, k]
    lambdas: np.ndarray  # complex [rank]


@dataclass(frozen=True)
class MultiHeadLowRankAttentionSurface:
    """The whole QKV-free attention contribution as **one** low-rank operator.

    A multi-head attention layer's contribution to the residual stream is the sum of
    its heads' outputs plus the ``out_proj`` bias. Each head is a
    :class:`LowRankAttentionHead`; summing low-rank character operators that read the
    *same* tokens but bind *different* atoms is itself a single low-rank operator —

        Σ_h U_h diag(λ_h) V_hᴴ χ_h  =  [U_0|U_1|…] diag([λ_0;λ_1;…]) blockdiag(V_0,V_1,…)ᴴ [χ_0;χ_1;…]

    — because ``V_hᴴ`` only ever multiplies its own head's character slice, so the
    stacked ``V`` is **block-diagonal** (measured exact, rel-err 3e-16). The
    ``out_proj`` bias rides along as a rank-1 DC term on the constant ``(0,0)``
    character (``χ_{(0,0)} ≡ 1``). The result is therefore a single
    :class:`LowRankCharacterVectorSurface`: one ``affine_bank_mod`` gather over the
    concatenated atoms, one ``low_rank_apply`` to the ``d_model`` output. Recovery
    (per-head atoms + factors) stays in PAN; this only assembles / rewrites / lowers.
    """

    modulus: int
    d_model: int
    heads: tuple[LowRankAttentionHead, ...]
    bias: np.ndarray | None  # complex [d_model] out_proj bias, or None

    @classmethod
    def from_heads(
        cls,
        modulus: int,
        d_model: int,
        heads: Sequence[LowRankAttentionHead | dict[str, Any]],
        bias: np.ndarray | None = None,
    ) -> MultiHeadLowRankAttentionSurface:
        """Build from per-head generators (and the optional ``out_proj`` bias).

        Each head may be a :class:`LowRankAttentionHead` or a ``dict`` of complex
        ``atoms`` / ``u`` ``[d_model, rank]`` / ``v`` ``[rank, k]`` / optional
        ``lambdas`` — the same factor convention as
        :meth:`LowRankCharacterVectorSurface.from_factors`.
        """
        n = int(modulus)
        d = int(d_model)
        built: list[LowRankAttentionHead] = []
        for h in heads:
            if isinstance(h, LowRankAttentionHead):
                head = h
            else:
                atoms = tuple((int(uu) % n, int(vv) % n) for (uu, vv) in h["atoms"])
                u = np.asarray(h["u"], dtype=np.complex128)
                v = np.asarray(h["v"], dtype=np.complex128)
                rank = int(u.shape[1])
                lam = (
                    np.ones(rank, dtype=np.complex128)
                    if h.get("lambdas") is None
                    else np.asarray(h["lambdas"], dtype=np.complex128)
                )
                head = LowRankAttentionHead(atoms=atoms, rank=rank, u_factors=u, v_factors=v, lambdas=lam)
            u = np.asarray(head.u_factors, dtype=np.complex128)
            v = np.asarray(head.v_factors, dtype=np.complex128)
            if u.shape != (d, head.rank):
                raise ValueError(f"head u must be [d_model={d}, rank={head.rank}]; got {u.shape}")
            if v.shape != (head.rank, len(head.atoms)):
                raise ValueError(
                    f"head v must be [rank={head.rank}, k={len(head.atoms)}]; got {v.shape}"
                )
            built.append(head)
        if not built:
            raise ValueError("need at least one attention head")
        b = None if bias is None else np.asarray(bias, dtype=np.complex128).reshape(d)
        return cls(modulus=n, d_model=d, heads=tuple(built), bias=b)

    # --------------------------------------------------------------------- #
    # Assembly into the single fused low-rank operator
    # --------------------------------------------------------------------- #

    def to_low_rank_surface(self) -> LowRankCharacterVectorSurface:
        """Fuse the per-head fanout into one :class:`LowRankCharacterVectorSurface`.

        Concatenates the heads' ``U``/``λ``/atoms, lays each head's ``V`` on the
        block-diagonal, and (if present) prepends the ``out_proj`` bias as a rank-1 DC
        term on the ``(0, 0)`` atom.
        """
        d = self.d_model
        has_bias = self.bias is not None
        dc = 1 if has_bias else 0
        k_total = dc + sum(len(h.atoms) for h in self.heads)
        r_total = dc + sum(h.rank for h in self.heads)
        atoms: list[tuple[int, int]] = [(0, 0)] if has_bias else []
        u = np.zeros((d, r_total), dtype=np.complex128)
        v = np.zeros((r_total, k_total), dtype=np.complex128)
        lam = np.zeros(r_total, dtype=np.complex128)
        rcur = kcur = 0
        if has_bias:
            u[:, 0] = self.bias  # DC column carries the bias vector
            v[0, 0] = 1.0
            lam[0] = 1.0
            rcur = kcur = 1
        for h in self.heads:
            rr, kk = h.rank, len(h.atoms)
            atoms.extend(h.atoms)
            u[:, rcur : rcur + rr] = np.asarray(h.u_factors, dtype=np.complex128)
            lam[rcur : rcur + rr] = np.asarray(h.lambdas, dtype=np.complex128)
            v[rcur : rcur + rr, kcur : kcur + kk] = np.asarray(h.v_factors, dtype=np.complex128)
            rcur += rr
            kcur += kk
        return LowRankCharacterVectorSurface.from_factors(self.modulus, atoms, u, v, lam)

    def to_execution_plan(self) -> tuple[ExecutionPlan, RewriteContext]:
        """Build the fused bank-bind + low-rank-apply plan (delegates to the surface)."""
        return self.to_low_rank_surface().to_execution_plan()

    def primitive_character_table(self) -> np.ndarray:
        """SPF words for ``χ_1(m) = e^{2πi·m/N}`` — one row backs the whole bank."""
        return self.to_low_rank_surface().primitive_character_table()

    @property
    def n_heads(self) -> int:
        return len(self.heads)

    @property
    def total_atoms(self) -> int:
        """Concatenated atom count (``k_total``), including the DC bias atom."""
        return (1 if self.bias is not None else 0) + sum(len(h.atoms) for h in self.heads)

    @property
    def total_rank(self) -> int:
        """Concatenated rank (``r_total``), including the DC bias term."""
        return (1 if self.bias is not None else 0) + sum(h.rank for h in self.heads)

    @property
    def param_words_low_rank(self) -> int:
        """Words in the fused operator: ``r_total·(d_model + k_total) + r_total``."""
        return self.to_low_rank_surface().param_words_low_rank

    @property
    def param_words_dense_attention(self) -> int:
        """Words in the dense layer this replaces: ``in_proj (3·d²) + out_proj (d²)``.

        The honest baseline is the full QKV attention block the surrogate bypasses for
        the final-token output — three ``d_model×d_model`` input projections plus the
        ``d_model×d_model`` output projection (biases excluded, as in PAN's count).
        """
        d = self.d_model
        return 4 * d * d


@dataclass(frozen=True)
class MultiHeadLowRankAttentionReport:
    """Result of assembling + rewriting + lowering a QKV-free attention surface."""

    modulus: int
    d_model: int
    n_heads: int
    total_atoms: int
    total_rank: int
    has_bias: bool
    plan_steps_before: int
    plan_steps_after: int
    rewrites_fired: tuple[str, ...]
    banks_collapsed: int
    param_words_low_rank: int
    param_words_dense_attention: int
    pl_dispatch_before: int
    pl_dispatch_after: int
    host_dispatch_before: int
    host_dispatch_after: int
    profile_before: str | None
    profile_after: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "MultiHeadLowRankAttentionSurfaceOperator",
            "modulus": self.modulus,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "total_atoms": self.total_atoms,
            "total_rank": self.total_rank,
            "has_bias": self.has_bias,
            "plan_steps_before": self.plan_steps_before,
            "plan_steps_after": self.plan_steps_after,
            "rewrites_fired": list(self.rewrites_fired),
            "banks_collapsed": self.banks_collapsed,
            "param_words_low_rank": self.param_words_low_rank,
            "param_words_dense_attention": self.param_words_dense_attention,
            "pl_dispatch_before": self.pl_dispatch_before,
            "pl_dispatch_after": self.pl_dispatch_after,
            "host_dispatch_before": self.host_dispatch_before,
            "host_dispatch_after": self.host_dispatch_after,
            "profile_before": self.profile_before,
            "profile_after": self.profile_after,
        }


def compile_multi_head_low_rank_attention_surface(
    surface: MultiHeadLowRankAttentionSurface,
) -> MultiHeadLowRankAttentionReport:
    """Assemble, rewrite, and lower a QKV-free multi-head attention surface."""
    plan, ctx = surface.to_execution_plan()
    optimized, matches, _ = rewrite_plan(plan, ctx)
    _, delta = rewrite_then_lower(plan, ctx)
    fired = tuple(m.name for m in matches)
    return MultiHeadLowRankAttentionReport(
        modulus=surface.modulus,
        d_model=surface.d_model,
        n_heads=surface.n_heads,
        total_atoms=surface.total_atoms,
        total_rank=surface.total_rank,
        has_bias=surface.bias is not None,
        plan_steps_before=len(plan.steps),
        plan_steps_after=len(optimized.steps),
        rewrites_fired=fired,
        banks_collapsed=sum(1 for n in fired if n == "character_bank_bind"),
        param_words_low_rank=surface.param_words_low_rank,
        param_words_dense_attention=surface.param_words_dense_attention,
        pl_dispatch_before=delta.pl_dispatch_before,
        pl_dispatch_after=delta.pl_dispatch_after,
        host_dispatch_before=delta.host_dispatch_before,
        host_dispatch_after=delta.host_dispatch_after,
        profile_before=delta.profile_before,
        profile_after=delta.profile_after,
    )


__all__ = [
    "LowRankAttentionHead",
    "LowRankCharacterVectorReport",
    "LowRankCharacterVectorSurface",
    "MultiHeadLowRankAttentionReport",
    "MultiHeadLowRankAttentionSurface",
    "SharedCharacterClassShiftReport",
    "SharedCharacterClassShiftSurface",
    "SparseCharacterSurface",
    "SparseCharacterSurfaceReport",
    "compile_low_rank_character_vector_surface",
    "compile_multi_head_low_rank_attention_surface",
    "compile_shared_character_class_shift_surface",
    "compile_sparse_character_surface",
]
