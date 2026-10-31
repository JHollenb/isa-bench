"""Spectral operator base class + honest lowering certificate.

A *spectral operator* is anything that lowers to an :class:`ExecutionPlan`:
the character-basis / affine-character / sparse-surface forms produced upstream
(PAN recovery, ``surface.py``), the structured operators in :mod:`spf.tensor.core`
(diagonal, low-rank, HDC), and future group-irrep forms. They share one
contract — ``lower() -> ExecutionPlan`` plus a self-declared proof status — so the
optimizer, cost model, and backends treat them uniformly and the vocabulary can
grow by subclassing instead of threading a new dict through the whole stack.

The recovery/proof of *what a learned circuit is* stays upstream (the PAN↔SPF
contract in ``docs/tensor/pan-spftensor-api.md``: recover upstream, rewrite and
lower in SPF). This module's :class:`LoweringCertificate` is the SPF-side honest
artifact: not "did we prove the decompile" but "this plan uses only supported
ops, N reducers, M Cartesian fallbacks, cost X" — derived structurally from the
:mod:`spf.tensor.op_registry`.
"""

from __future__ import annotations

import abc
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from .core import AccumulationDomain
from .op_registry import OP_REGISTRY
from .rewrite import DEFAULT_COST_MODEL, OpCostModel, Provenance
from .v2 import ExecutionPlan

__all__ = [
    "LoweringCertificate",
    "PlanOperator",
    "ProofStatus",
    "SpectralOperator",
    "certify_plan",
]


class ProofStatus(StrEnum):
    """How much evidence backs an operator's equivalence claim.

    Mirrors the gate the PAN decompiler already uses: a plan is only ``VALIDATED``
    when calibrated held-out accuracy, atom agreement, and negative controls all
    pass; ``DIAGNOSTIC`` when structure is recovered but the gate refuses to
    promote it; ``FALLBACK`` when the operator is a non-SPF-cheap escape hatch.
    """

    VALIDATED = "validated"
    DIAGNOSTIC = "diagnostic"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class LoweringCertificate:
    """Structural, backend-agnostic account of an :class:`ExecutionPlan`.

    Every field is derived from the plan + op registry — nothing here asserts the
    *function* is correct (that is the upstream proof's job). It answers the
    deployment question: does this plan stay inside the SPF op vocabulary, and at
    what reducer / fallback / cost shape.
    """

    op_counts: Mapping[str, int]
    n_steps: int
    reducer_count: int
    cartesian_fallback_count: int
    unsupported_ops: tuple[str, ...]
    reference_ops: tuple[str, ...]
    cost: float
    proof_status: ProofStatus = ProofStatus.DIAGNOSTIC

    @property
    def supported_spf_ops_only(self) -> bool:
        """True iff every op is registered and none is a numpy reference oracle."""
        return not self.unsupported_ops and not self.reference_ops

    @property
    def deployable(self) -> bool:
        """True iff the plan lowers wholly to SPF backends with no host fallback.

        Strictest shape: supported ops only and zero Cartesian (host PS) reduce
        fallbacks. A ``VALIDATED`` proof status plus ``deployable`` is the
        green-light shape the decompiler suite reports for a clean compile.
        """
        return self.supported_spf_ops_only and self.cartesian_fallback_count == 0

    def summary(self) -> dict[str, object]:
        """JSON-friendly dict, matching the decompiler proof-artifact field names."""
        return {
            "op_counts": dict(self.op_counts),
            "n_steps": self.n_steps,
            "reducer_count": self.reducer_count,
            "cartesian_fallback_count": self.cartesian_fallback_count,
            "supported_spf_ops_only": self.supported_spf_ops_only,
            "deployable": self.deployable,
            "unsupported_ops": list(self.unsupported_ops),
            "reference_ops": list(self.reference_ops),
            "cost": self.cost,
            "proof_status": str(self.proof_status),
        }


def _pinned_cartesian_reduces(plan: ExecutionPlan) -> int:
    """Count reduce steps whose reducer is *pinned* to the CARTESIAN domain.

    Pre-lowering best effort: only a ``PINNED`` reducer's domain is knowable from
    the plan alone (a ``STATIC`` reducer's domain is resolved at lower time from
    stats). The authoritative count is ``len(LoweredPlan.ps_fallback)``; pass it
    in via ``cartesian_fallback_count`` when you have a lowered plan.
    """
    n = 0
    for step in plan.steps:
        spec = step.reducer
        if spec is not None and spec.domain == AccumulationDomain.CARTESIAN:
            n += 1
    return n


def certify_plan(
    plan: ExecutionPlan,
    *,
    cost_model: OpCostModel = DEFAULT_COST_MODEL,
    proof_status: ProofStatus = ProofStatus.DIAGNOSTIC,
    cartesian_fallback_count: int | None = None,
) -> LoweringCertificate:
    """Build a :class:`LoweringCertificate` for any execution plan.

    Works on plans from any source — rewrite output, PAN-recovered surfaces, the
    structured operators. ``cartesian_fallback_count`` overrides the pre-lowering
    estimate when the caller has lowered the plan (use ``len(lowered.ps_fallback)``).
    """
    counts = Counter(step.op for step in plan.steps)
    unsupported = tuple(sorted(op for op in counts if op not in OP_REGISTRY))
    reference = tuple(
        sorted(op for op in counts if op in OP_REGISTRY and OP_REGISTRY[op].reference_only)
    )
    fallback = (
        _pinned_cartesian_reduces(plan)
        if cartesian_fallback_count is None
        else int(cartesian_fallback_count)
    )
    cost = cost_model.plan_cost(plan)
    return LoweringCertificate(
        op_counts=dict(counts),
        n_steps=len(plan.steps),
        reducer_count=len(plan.reducer_boundaries),
        cartesian_fallback_count=fallback,
        unsupported_ops=unsupported,
        reference_ops=reference,
        cost=cost,
        proof_status=proof_status,
    )


class SpectralOperator(abc.ABC):
    """Base class for anything that lowers to an :class:`ExecutionPlan`.

    Subclasses implement :meth:`lower`; everything else (cost, certificate) is
    derived. ``proof_status`` and ``provenance`` default to the conservative
    values and are overridden by operators that carry stronger evidence.
    """

    @abc.abstractmethod
    def lower(self) -> ExecutionPlan:
        """Return the executable plan this operator compiles to."""

    @property
    def proof_status(self) -> ProofStatus:
        return ProofStatus.DIAGNOSTIC

    @property
    def provenance(self) -> Provenance | None:
        return None

    def cost(self, cost_model: OpCostModel = DEFAULT_COST_MODEL) -> float:
        return cost_model.plan_cost(self.lower())

    def certify(
        self,
        *,
        cost_model: OpCostModel = DEFAULT_COST_MODEL,
        cartesian_fallback_count: int | None = None,
    ) -> LoweringCertificate:
        return certify_plan(
            self.lower(),
            cost_model=cost_model,
            proof_status=self.proof_status,
            cartesian_fallback_count=cartesian_fallback_count,
        )


@dataclass(slots=True)
class PlanOperator(SpectralOperator):
    """Adapter wrapping a prebuilt :class:`ExecutionPlan` as a spectral operator.

    Lets a plan recovered upstream (PAN character surface, transformer patch) or
    produced by ``rewrite_plan`` enter the operator API uniformly — carrying its
    proof status and provenance — without forcing the producer to subclass.
    """

    plan: ExecutionPlan
    status: ProofStatus = ProofStatus.DIAGNOSTIC
    operator_provenance: Provenance | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def lower(self) -> ExecutionPlan:
        return self.plan

    @property
    def proof_status(self) -> ProofStatus:
        return self.status

    @property
    def provenance(self) -> Provenance | None:
        return self.operator_provenance
