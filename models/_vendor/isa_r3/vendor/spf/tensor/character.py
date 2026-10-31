"""Load a recovered character-basis operator from a ``.spfmodel`` and lower it.

This is the SPF-side consumer of PAN's character decompile. **Recovery stays
upstream**: extracting a ``CharacterBasisOperator`` from a trained model's phase
surfaces is learned-model analysis (2-D Fourier character recovery, calibration,
control ablations) and lives in the PAN repo. This module never recovers anything.
It takes the *already-recovered* operator metadata and does the three things the
SPF compiler owns:

1. **Build** a first-class :class:`~spf.tensor.v2.ExecutionPlan` from the operator
   — the un-collapsed sum/affine character *bind* form (one ``gather`` per input
   axis into a character table, an ``elementwise_mul`` bind, then a per-row
   ``diagonal_scale`` fanout).
2. **Rewrite** it with the #316 algebraic registry
   (:func:`spf.tensor.rewrite.rewrite_plan`). The plan-time
   :class:`~spf.tensor.rewrite.CharacterPhaseBind` rule collapses each matched
   sum-character bind ``χ_f(a)·χ_f(b) -> χ_f((a+b) mod N)`` — reproducing, as an
   *algebraic* rewrite over an existing plan, the collapse PAN performs by hand
   during recovery.
3. **Lower** both forms (:func:`spf.tensor.rewrite.rewrite_then_lower`) and emit a
   PL/host dispatch report.

The boundary is deliberate and load-bearing: recovery is model analysis; rewriting
is algebra *after* the plan exists. The recovery pass is therefore **not** a rewrite
rule — only the structural collapse is.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .io.archive import read_spfmodel
from .modular_arith import dlog_table, primitive_root
from .rewrite import (
    Provenance,
    RewriteContext,
    rewrite_plan,
    rewrite_then_lower,
)
from .v2 import ExecutionPlan, ExecutionStep

# Where a `.spfmodel` carries the recovered operator (under manifest ``meta``).
OPERATOR_META_KEY = "character_operator"


@dataclass(frozen=True)
class CharacterBasisOperator:
    """A recovered affine/sum character-basis operator over ``Z_modulus``.

    ``row r`` reads the character ``χ_{g_r}`` of the affine index
    ``s = (α·a + β·b + bias) mod N`` and applies a per-row phase offset, where
    ``g_r = unique_generators[row_to_generator[r]]``. For a sum-character circuit
    (``α == β``, the modular-addition case) both per-axis character indices coincide
    and the bind collapses; for a difference/affine circuit they differ and it does
    not — exactly what the rewrite engine should (not) fire on.
    """

    modulus: int
    rows: int
    affine_coefficients_mod: tuple[int, int]
    affine_bias_mod: int
    unique_generators: tuple[int, ...]
    row_to_generator: tuple[int, ...]
    row_phase_offsets: tuple[float, ...]
    character_structure: str = "UNKNOWN"

    # --------------------------------------------------------------------- #
    # Construction
    # --------------------------------------------------------------------- #

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> CharacterBasisOperator:
        """Build from a recovered-operator metadata dict (PAN's emitter shape)."""
        p = int(meta.get("modulus", meta.get("p")))
        rows = int(meta["rows"])
        coeffs = meta.get("affine_coefficients_mod", [1, 1])
        alpha, beta = int(coeffs[0]) % p, int(coeffs[1]) % p
        bias = int(meta.get("affine_bias_mod", 0)) % p
        gens = tuple(int(g) % p for g in meta["unique_generators"])
        r2g = tuple(int(i) for i in meta["row_to_generator"])
        if len(r2g) != rows:
            raise ValueError(f"row_to_generator has {len(r2g)} entries, expected rows={rows}")
        if any(not 0 <= i < len(gens) for i in r2g):
            raise ValueError("row_to_generator indexes outside unique_generators")
        offsets = _row_offsets(meta, rows)
        return cls(
            modulus=p,
            rows=rows,
            affine_coefficients_mod=(alpha, beta),
            affine_bias_mod=bias,
            unique_generators=gens,
            row_to_generator=r2g,
            row_phase_offsets=offsets,
            character_structure=str(meta.get("character_structure", "UNKNOWN")),
        )

    @classmethod
    def from_spfmodel(cls, path: str | Path) -> CharacterBasisOperator:
        """Load the recovered operator carried in a ``.spfmodel`` archive."""
        archive = read_spfmodel(path)
        meta = dict(archive.manifest.get("meta") or {})
        op_meta = meta.get(OPERATOR_META_KEY)
        if not isinstance(op_meta, dict):
            raise ValueError(
                f"{path}: archive carries no {OPERATOR_META_KEY!r} operator metadata "
                "(was it exported by a character-decompile run?)"
            )
        return cls.from_metadata(op_meta)

    # --------------------------------------------------------------------- #
    # Plan construction
    # --------------------------------------------------------------------- #

    def _generator_axis_indices(self, g: int) -> tuple[int, int]:
        """Per-axis character indices (u_a, u_b) for generator ``g``.

        ``χ_g(α·a + β·b + bias) = χ_{gα}(a)·χ_{gβ}(b)·const`` over ``Z_N``; the
        constant ``g·bias`` phase folds into the row offset.
        """
        alpha, beta = self.affine_coefficients_mod
        n = self.modulus
        return (alpha * g) % n, (beta * g) % n

    def to_execution_plan(self) -> tuple[ExecutionPlan, RewriteContext]:
        """Build the bind-form plan and the provenance-bearing rewrite context.

        Returns ``(plan, ctx)``; pass both to :func:`spf.tensor.rewrite.rewrite_plan`
        / :func:`rewrite_then_lower`. The context vouches each per-axis gather output
        as an exact DFT-row character so :class:`CharacterPhaseBind` can fire on the
        sum-character binds (and only those).
        """
        n = self.modulus
        alpha, beta = self.affine_coefficients_mod
        affine = not (alpha == 1 and beta == 1)
        steps: list[ExecutionStep] = []
        provenance: dict[str, Provenance] = {}
        bound_name: dict[int, str] = {}  # generator index -> bound env-var

        for gi, g in enumerate(self.unique_generators):
            ua, ub = self._generator_axis_indices(g)
            na, nb = f"chi_{gi}_a", f"chi_{gi}_b"
            if affine:
                # Both axes read the same generator-g character table at the scaled
                # index (c·input) mod N; the per-axis rows αg / βg are then distinct
                # and the bind fuses to the affine index (αa+βb) mod N. The bias
                # phase χ_g(bias) folds into the row offset (see `_bias_phase`).
                table = f"chi_gen_{gi}_table"
                steps.append(
                    ExecutionStep(
                        "gather", (table, "input_0"), (na,),
                        params={"axis": 0, "index_coeff": alpha, "modulus": n},
                    )
                )
                steps.append(
                    ExecutionStep(
                        "gather", (table, "input_1"), (nb,),
                        params={"axis": 0, "index_coeff": beta, "modulus": n},
                    )
                )
                provenance[na] = Provenance(kind="dft_row", N=n, index=ua, exact=True, axis_coeff=alpha)
                provenance[nb] = Provenance(kind="dft_row", N=n, index=ub, exact=True, axis_coeff=beta)
            else:
                steps.append(
                    ExecutionStep("gather", ("chi_a_table", "input_0"), (na,), params={"axis": 0})
                )
                steps.append(
                    ExecutionStep("gather", ("chi_b_table", "input_1"), (nb,), params={"axis": 0})
                )
                provenance[na] = Provenance(kind="dft_row", N=n, index=ua, exact=True)
                provenance[nb] = Provenance(kind="dft_row", N=n, index=ub, exact=True)
            bound = f"chi_{gi}"
            steps.append(
                ExecutionStep(
                    "elementwise_mul",
                    (na, nb),
                    (bound,),
                    params={"cartesian_fallback": True, "purpose": "character_bind"},
                )
            )
            bound_name[gi] = bound

        for r in range(self.rows):
            gi = self.row_to_generator[r]
            g = self.unique_generators[gi]
            offset = self.row_phase_offsets[r] + _bias_phase(g, self.affine_bias_mod, n)
            steps.append(
                ExecutionStep(
                    "diagonal_scale",
                    (bound_name[gi], f"row_{r}_offset"),
                    (f"mixed_row_{r}",),
                    params={
                        "purpose": "row_phase_offset",
                        "row": r,
                        "generator_index": gi,
                        "generator": g,
                        "coeff_theta": float(offset),
                    },
                )
            )

        plan = ExecutionPlan(
            tuple(steps),
            factorization_metadata={
                "operator": "CharacterBasisOperator",
                "character_structure": self.character_structure,
                "modulus": n,
                "rows": self.rows,
                "affine_coefficients_mod": list(self.affine_coefficients_mod),
                "affine_bias_mod": self.affine_bias_mod,
                "unique_generators": list(self.unique_generators),
            },
        )
        # Legacy recovered-character builders optimize the ideal group function.
        # Their rounded tables do not certify packed-step equivalence.
        return plan, RewriteContext(provenance=provenance, rounding_contract="algebraic_output")


@dataclass(frozen=True)
class CharacterLoweringReport:
    """Result of building + rewriting + lowering a recovered character operator."""

    operator: CharacterBasisOperator
    plan_steps_before: int
    plan_steps_after: int
    rewrites_fired: tuple[str, ...]
    binds_collapsed: int
    pl_dispatch_before: int
    pl_dispatch_after: int
    host_dispatch_before: int
    host_dispatch_after: int
    profile_before: str | None
    profile_after: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": "CharacterBasisOperator",
            "character_structure": self.operator.character_structure,
            "modulus": self.operator.modulus,
            "rows": self.operator.rows,
            "unique_generators": list(self.operator.unique_generators),
            "affine_coefficients_mod": list(self.operator.affine_coefficients_mod),
            "affine_bias_mod": self.operator.affine_bias_mod,
            "plan_steps_before": self.plan_steps_before,
            "plan_steps_after": self.plan_steps_after,
            "rewrites_fired": list(self.rewrites_fired),
            "binds_collapsed": self.binds_collapsed,
            "pl_dispatch_before": self.pl_dispatch_before,
            "pl_dispatch_after": self.pl_dispatch_after,
            "host_dispatch_before": self.host_dispatch_before,
            "host_dispatch_after": self.host_dispatch_after,
            "profile_before": self.profile_before,
            "profile_after": self.profile_after,
        }


def compile_character_operator(op: CharacterBasisOperator) -> CharacterLoweringReport:
    """Build, rewrite, and lower a recovered operator into a lowering report."""
    plan, ctx = op.to_execution_plan()
    # Algebraic rewrite first (CharacterPhaseBind collapses sum-character binds).
    optimized, matches, _ = rewrite_plan(plan, ctx)
    # Then lower both forms for the PL/host dispatch delta (re-runs the same
    # rewrite internally under ctx, so the after-form matches `optimized`).
    _, delta = rewrite_then_lower(plan, ctx)
    fired = tuple(m.name for m in matches)
    return CharacterLoweringReport(
        operator=op,
        plan_steps_before=len(plan.steps),
        plan_steps_after=len(optimized.steps),
        rewrites_fired=fired,
        binds_collapsed=sum(1 for n in fired if n == "character_phase_bind"),
        pl_dispatch_before=delta.pl_dispatch_before,
        pl_dispatch_after=delta.pl_dispatch_after,
        host_dispatch_before=delta.host_dispatch_before,
        host_dispatch_after=delta.host_dispatch_after,
        profile_before=delta.profile_before,
        profile_after=delta.profile_after,
    )


def compile_character_spfmodel(
    path: str | Path, *, report_path: str | Path | None = None
) -> CharacterLoweringReport:
    """Load a ``.spfmodel`` operator, lower it, and (optionally) write the report.

    The end-to-end SPF-side path:
    ``.spfmodel -> CharacterBasisOperator -> ExecutionPlan -> rewrite -> lowering``.
    """
    op = CharacterBasisOperator.from_spfmodel(path)
    report = compile_character_operator(op)
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return report


def _row_offsets(meta: dict[str, Any], rows: int) -> tuple[float, ...]:
    """Per-row phase offsets, indexed by row, from ``row_coefficients``."""
    offsets = [0.0] * rows
    for coeff in meta.get("row_coefficients", ()):
        r = int(coeff["row"])
        if 0 <= r < rows:
            offsets[r] = float(coeff.get("coeff_theta", 0.0))
    return tuple(offsets)


def _bias_phase(g: int, bias: int, modulus: int) -> float:
    """The constant ``χ_g(bias)`` phase folded into a row offset (0 when bias=0)."""
    if bias == 0:
        return 0.0

    return (2.0 * math.pi / modulus) * ((g * bias) % modulus)


# --------------------------------------------------------------------------- #
# Discrete-log character surface — the multiplicative-group sibling.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiscreteLogCharacterSurface:
    """A character surface over the multiplicative group ``Z_p^* = ⟨g0⟩``.

    The additive :class:`CharacterBasisOperator` covers ``Z_p`` residue axes
    (``mod_add``/affine). Multiplicative axes (``mod_mul``/``mod_div``) are the
    *same* sum-character circuit, but in the **exponent group** ``Z_{p−1}`` after a
    discrete-log re-index of each input: ``a·b mod p ↔ dlog a + dlog b mod (p−1)``.
    So the surface carries the ``dlog`` table (one free gather per input) plus the
    additive exponent coefficients ``(u, v)`` — ``(1, 1)`` for multiplication
    (``DLOG_SUM``) and ``(1, −1)`` for division (``DLOG_DIFF``). The ``x=0`` row sits
    outside the group and is handled by an explicit sentinel (the product is 0).

    The honesty gate tells you the *additive* basis is wrong for ``mod_mul``
    (projection energy ≈ 0); this surface supplies the *right* basis. This closes
    the ``REAL_PHASE_FALLBACK`` coverage gap — it is a coverage extension, not a
    method change: the same G1–G6 battery applies unchanged.
    """

    modulus: int
    exponent_modulus: int
    primitive_root: int
    dlog_table: tuple[int, ...]
    zero_sentinel: int
    affine_coefficients_exponent: tuple[int, int]
    character_structure: str = "DLOG_SUM"

    @classmethod
    def for_task(
        cls,
        p: int,
        *,
        structure: str = "DLOG_SUM",
        generator: int | None = None,
    ) -> DiscreteLogCharacterSurface:
        """Build the canonical surface for ``mod_mul`` (``DLOG_SUM``) / ``mod_div`` (``DLOG_DIFF``)."""
        g0 = primitive_root(p) if generator is None else int(generator)
        m = p - 1
        table = dlog_table(p, g0)
        if structure == "DLOG_SUM":
            uv = (1, 1)
        elif structure == "DLOG_DIFF":
            uv = (1, (-1) % m)
        else:
            raise ValueError(f"unknown discrete-log structure {structure!r}")
        return cls(
            modulus=int(p),
            exponent_modulus=int(m),
            primitive_root=int(g0),
            dlog_table=tuple(table),
            zero_sentinel=-1,
            affine_coefficients_exponent=uv,
            character_structure=structure,
        )

    @classmethod
    def from_metadata(cls, meta: dict[str, Any]) -> DiscreteLogCharacterSurface:
        """Build from a recovered-operator metadata dict (PAN's emitter shape)."""
        p = int(meta.get("modulus", meta.get("p")))
        m = int(meta.get("exponent_modulus", p - 1))
        # Don't eagerly compute primitive_root for the default — it raises on a
        # non-prime p, and a recovered metadata dict may already carry g0/the table.
        g0 = int(meta["primitive_root"]) if "primitive_root" in meta else primitive_root(p)
        table = meta.get("dlog_table")
        dlog = tuple(int(v) for v in table) if table is not None else tuple(dlog_table(p, g0))
        if len(dlog) != p:
            raise ValueError(f"dlog_table has {len(dlog)} entries, expected p={p}")
        uv = meta.get("affine_coefficients_exponent", [1, 1])
        return cls(
            modulus=p,
            exponent_modulus=m,
            primitive_root=g0,
            dlog_table=dlog,
            zero_sentinel=int(meta.get("zero_sentinel", -1)),
            affine_coefficients_exponent=(int(uv[0]) % m, int(uv[1]) % m),
            character_structure=str(meta.get("character_structure", "DLOG_SUM")),
        )

    # ----------------------------------------------------------------- #
    # Reference evaluator
    # ----------------------------------------------------------------- #

    def logits(
        self, a: np.ndarray | int, b: np.ndarray | int, *, freqs: tuple[int, ...] = (1,)
    ) -> np.ndarray:
        """Zero-parameter logits over ``c∈[0,p)`` via the exponent-group character sum.

        ``logit[c] = Σ_k cos(2πk·(u·dlog a + v·dlog b − dlog c)/(p−1))`` for the
        ``p−1`` group elements; the ``c=0`` column is forced low unless a zero
        operand makes the product 0 (then it is forced to win). Argmax of this is
        ``a·b mod p`` (sum) / ``a·b⁻¹ mod p`` (diff).
        """
        p, m = self.modulus, self.exponent_modulus
        u, v = self.affine_coefficients_exponent
        dlog = np.asarray(self.dlog_table, dtype=np.int64)
        av = np.asarray(a, dtype=np.int64)
        bv = np.asarray(b, dtype=np.int64)
        shape = np.broadcast(av, bv).shape
        av = np.broadcast_to(av, shape)
        bv = np.broadcast_to(bv, shape)
        da = dlog[av]
        db = dlog[bv]
        s = (u * da + v * db) % m  # exponent-group target (valid where both nonzero)
        dlog_c = dlog[np.arange(p)]  # dlog of each candidate c; c=0 -> -1
        c_exp = dlog_c.reshape(*([1] * s.ndim), p).astype(np.float64)
        sf = s[..., None].astype(np.float64)
        logits = np.zeros((*shape, p), dtype=np.float64)
        for k in freqs:
            logits += np.cos((2.0 * math.pi * k / m) * (sf - c_exp))
        # c=0 is off-group: never a winner for nonzero operands.
        logits[..., 0] = -np.inf
        # A zero operand makes the product exactly 0 — force c=0 to win there.
        zero_operand = (av == 0) | (bv == 0)
        if np.any(zero_operand):
            neg = np.full((p,), -np.inf)
            neg[0] = np.inf
            logits[zero_operand] = neg
        return logits

    def predict(
        self, a: np.ndarray | int, b: np.ndarray | int, *, freqs: tuple[int, ...] = (1,)
    ) -> np.ndarray:
        """Closed-form prediction ``argmax_c`` of :meth:`logits`."""
        return self.logits(a, b, freqs=freqs).argmax(-1)

    def accuracy(self, *, freqs: tuple[int, ...] = (1,)) -> float:
        """Exact accuracy over all ``p²`` pairs against the true multiplicative target."""
        p = self.modulus
        a = np.broadcast_to(np.arange(p).reshape(p, 1), (p, p))
        b = np.broadcast_to(np.arange(p).reshape(1, p), (p, p))
        pred = self.predict(a, b, freqs=freqs)
        if self.character_structure == "DLOG_SUM":
            true = (a * b) % p
        else:  # DLOG_DIFF — division a · b^{-1}
            inv = np.array([pow(int(x), p - 2, p) if x else 0 for x in range(p)], dtype=np.int64)
            true = (a * inv[b]) % p
        return float((pred == true).mean())

    # ----------------------------------------------------------------- #
    # Lowering
    # ----------------------------------------------------------------- #

    def lowering(self) -> dict[str, Any]:
        """SPF-free lowering declaration (G5): ``dlog_gather`` then the phase ramp add."""
        return {
            "ops": ["dlog_gather", "affine_index_mod", "phase_table_gather", "diagonal_phase_shift"],
            "reducer_count": 0,
            "cartesian_fallback_count": 0,
            "spf_free": True,
        }

    def to_execution_plan(self) -> tuple[ExecutionPlan, RewriteContext]:
        """Build the bind-form plan: ``dlog_gather`` each input, then the exponent-group bind.

        The two ``dlog_gather`` steps re-index ``a``/``b`` into the exponent group;
        the ``elementwise_mul`` binds the two character reads (``χ_u(da)·χ_v(db)``),
        and a ``diagonal_scale`` carries the row phase. No reducer, no Cartesian
        fallback — the cheap target.
        """
        p, m = self.modulus, self.exponent_modulus
        u, v = self.affine_coefficients_exponent
        steps: list[ExecutionStep] = []
        provenance: dict[str, Provenance] = {}
        # The discrete-log re-index is a free gather over the size-p dlog table
        # (one np.take per input) — the cheapest free_monomial op, so it lowers to
        # the existing ``gather`` rather than a new resident op. ``dlog=True`` tags
        # it as the re-index; ``lowering()`` names it ``dlog_gather`` as the SPF-free
        # primitive it stands for.
        steps.append(
            ExecutionStep(
                "gather", ("dlog_table", "input_0"), ("da",),
                params={"axis": 0, "dlog": True, "modulus": p},
            )
        )
        steps.append(
            ExecutionStep(
                "gather", ("dlog_table", "input_1"), ("db",),
                params={"axis": 0, "dlog": True, "modulus": p},
            )
        )
        steps.append(
            ExecutionStep(
                "gather", ("chi_a_table", "da"), ("chi_a",),
                params={"axis": 0, "index_coeff": u, "modulus": m},
            )
        )
        steps.append(
            ExecutionStep(
                "gather", ("chi_b_table", "db"), ("chi_b",),
                params={"axis": 0, "index_coeff": v, "modulus": m},
            )
        )
        provenance["chi_a"] = Provenance(kind="dft_row", N=m, index=u, exact=True, axis_coeff=u)
        provenance["chi_b"] = Provenance(kind="dft_row", N=m, index=v, exact=True, axis_coeff=v)
        steps.append(
            ExecutionStep(
                "elementwise_mul", ("chi_a", "chi_b"), ("bound",),
                params={"cartesian_fallback": True, "purpose": "character_bind"},
            )
        )
        steps.append(
            ExecutionStep(
                "diagonal_scale", ("bound", "row_offset"), ("mixed",),
                params={"purpose": "row_phase_offset", "coeff_theta": 0.0},
            )
        )
        plan = ExecutionPlan(
            tuple(steps),
            factorization_metadata={
                "operator": "DiscreteLogCharacterSurface",
                "character_structure": self.character_structure,
                "modulus": p,
                "exponent_modulus": m,
                "primitive_root": self.primitive_root,
                "affine_coefficients_exponent": list(self.affine_coefficients_exponent),
            },
        )
        # Legacy recovered-character builders optimize the ideal group function.
        # Their rounded tables do not certify packed-step equivalence.
        return plan, RewriteContext(provenance=provenance, rounding_contract="algebraic_output")


__all__ = [
    "OPERATOR_META_KEY",
    "CharacterBasisOperator",
    "CharacterLoweringReport",
    "DiscreteLogCharacterSurface",
    "compile_character_operator",
    "compile_character_spfmodel",
]
