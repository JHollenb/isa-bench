"""SPFTensor v2 decomposition-selection guidance (roadmap Stage 3).

Mathematically-grounded guidance for choosing a tensor decomposition under the
SPF cost model. Two imported results drive the policy:

* **TT / Hierarchical-Tucker quasi-best approximations provably exist.** The
  Frobenius-best approximation with bounded TT-ranks always exists, and TT-SVD
  is quasi-optimal: ``‖A − B‖ ≤ √(d−1)·‖A − A_best‖`` (Oseledets 2011, *Tensor-
  Train Decomposition*, SIAM J. Sci. Comput. 33(5), Cor. 2.4; the hierarchical-
  SVD analogue is Grasedyck/Hackbusch).
* **CP is not closed — it is unsafe as a storage target.** The best rank-``r``
  CP approximation can fail to exist (de Silva & Lim 2008, *Tensor Rank and the
  Ill-Posedness of the Best Low-Rank Approximation Problem*, SIAM J. Matrix
  Anal. Appl. 30(3); they exhibit a space where no rank-3 tensor has an optimal
  rank-2 approximation). So CP is rejected as a storage profile here, mirroring
  ``ProfileTerm``'s rejection of ``PROFILE_CP_CLOSED``.

The **SPF-specific** twist: a rank choice should minimise *additive recompression*,
not just stored bytes. In log-polar SPF, MUL/scale are O(1) integer adds but the
complex **ADD** (the ``⊕`` accumulation) is the expensive op. Every contraction
and TT-rounding sweep pays its real cost in those accumulations, so
:func:`additive_recompression_cost` counts the SPF ``⊕`` operations a TT network
incurs — the metric to minimise when picking ranks/orderings.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum

from .core import SPFError
from .v2 import PROFILE_CP_CLOSED, PROFILE_TT_RANK_MIN


class DecompositionKind(IntEnum):
    """Supported tensor-decomposition families, ordered by the profile lattice."""

    TT = 1  # Tensor-Train (Oseledets) — quasi-best exists.
    HIERARCHICAL_TUCKER = 2  # H-Tucker (Grasedyck/Hackbusch) — quasi-best exists.
    CP = 3  # CANDECOMP/PARAFAC — set not closed; unsafe as a storage target.
    DENSE = 4  # ⊤ of the profile lattice; always representable, no compression.


#: Decompositions whose best bounded-rank approximation provably exists, so they
#: are safe to commit as a stored block. CP is deliberately excluded.
STORAGE_SAFE_KINDS = frozenset(
    {DecompositionKind.TT, DecompositionKind.HIERARCHICAL_TUCKER, DecompositionKind.DENSE}
)


def is_storage_safe(kind: DecompositionKind) -> bool:
    """Whether ``kind`` has a provably-existing best bounded-rank approximation."""

    return DecompositionKind(kind) in STORAGE_SAFE_KINDS


def tt_quasi_optimality_constant(order: int) -> float:
    """TT-SVD quasi-optimality constant ``√(d−1)`` for an order-``d`` tensor.

    From Oseledets (2011) Cor. 2.4: the TT-SVD approximation ``B`` satisfies
    ``‖A − B‖_F ≤ √(d−1)·‖A − A_best‖_F``. Order ``d ≤ 1`` is exact (constant 0).
    """

    d = int(order)
    if d < 0:
        raise SPFError("tensor order must be non-negative")
    return math.sqrt(max(d - 1, 0))


@dataclass(frozen=True, slots=True)
class DecompositionRecommendation:
    """Outcome of :func:`select_decomposition`."""

    kind: DecompositionKind
    quasi_optimality_constant: float
    profile_id: int
    rationale: str

    @property
    def storage_safe(self) -> bool:
        return is_storage_safe(self.kind)


def select_decomposition(order: int, *, allow_dense: bool = True) -> DecompositionRecommendation:
    """Recommend a storage-safe decomposition for an order-``d`` tensor.

    Prefers Tensor-Train (quasi-best exists, ``√(d−1)`` bound). Falls back to
    ``DENSE`` only when ``allow_dense`` and the order is too small to compress
    (``d ≤ 1``). Never recommends CP — see module docstring.
    """

    d = int(order)
    if d < 0:
        raise SPFError("tensor order must be non-negative")
    if d <= 1:
        if not allow_dense:
            raise SPFError("order-0/1 tensors are not compressible; dense storage required")
        return DecompositionRecommendation(
            kind=DecompositionKind.DENSE,
            quasi_optimality_constant=0.0,
            profile_id=0,  # dense/⊤ sentinel
            rationale=f"order {d} has no internal mode to factor; store dense",
        )
    return DecompositionRecommendation(
        kind=DecompositionKind.TT,
        quasi_optimality_constant=tt_quasi_optimality_constant(d),
        profile_id=int(PROFILE_TT_RANK_MIN),
        rationale=(
            f"TT-SVD is quasi-optimal (√(d−1)={tt_quasi_optimality_constant(d):.4f}) and its "
            "best bounded-rank approximation provably exists (Oseledets 2011 Cor. 2.4); "
            "CP is rejected as a non-closed storage target (de Silva–Lim 2008)"
        ),
    )


def reject_cp_storage() -> None:
    """Raise the canonical error for attempting to store a CP factorization.

    Kept as a single call site so the CP ban reads identically wherever a caller
    is tempted to commit a CP decomposition (mirrors ``ProfileTerm`` rejecting
    ``PROFILE_CP_CLOSED`` = 0x%04x).
    """

    raise SPFError(
        f"CP is not a closed decomposition (profile 0x{PROFILE_CP_CLOSED:04x}); the best "
        "bounded-rank CP approximation can fail to exist (de Silva–Lim 2008). Use TT or "
        "Hierarchical-Tucker as a storage target."
    )


def additive_recompression_cost(
    mode_sizes: Sequence[int],
    tt_ranks: Sequence[int],
) -> int:
    """Count the SPF ``⊕`` accumulations in a full Tensor-Train contraction.

    Under the SPF cost model a complex MUL/scale is O(1) integer arithmetic but
    complex ADD (the ``⊕`` accumulation) is the expensive op, so the real cost of
    a TT contraction or rounding sweep is its number of accumulations — not its
    stored byte count. This counts exactly those additions for contracting an
    order-``d`` TT tensor (mode sizes ``n_1..n_d``, ranks ``r_0..r_d`` with
    ``r_0 = r_d = 1``) down to a scalar via the standard left-to-right sweep:

    * mode-``k`` reduction over ``n_k`` terms into the ``r_{k-1}×r_k`` core slab:
      ``r_{k-1}·r_k·(n_k − 1)`` additions;
    * interface vector–matrix reduction over ``r_{k-1}`` terms:
      ``r_k·(r_{k-1} − 1)`` additions.

    Minimising this over candidate rank/ordering choices is the Stage-3
    SPF-specific selection criterion (additions, not bytes).
    """

    n = [int(m) for m in mode_sizes]
    r = [int(x) for x in tt_ranks]
    d = len(n)
    if len(r) != d + 1:
        raise SPFError("tt_ranks must have len(mode_sizes) + 1 entries (r_0..r_d)")
    if d == 0:
        return 0
    if r[0] != 1 or r[-1] != 1:
        raise SPFError("boundary TT-ranks must satisfy r_0 = r_d = 1")
    if any(m <= 0 for m in n) or any(x <= 0 for x in r):
        raise SPFError("mode sizes and TT-ranks must be positive")
    adds = 0
    for k in range(d):
        r_prev, r_next, n_k = r[k], r[k + 1], n[k]
        adds += r_prev * r_next * (n_k - 1)  # reduce over mode k
        adds += r_next * (r_prev - 1)  # reduce over the interface index
    return adds


def cheapest_tt_ordering(
    candidates: Sequence[tuple[Sequence[int], Sequence[int]]],
) -> int:
    """Index of the ``(mode_sizes, tt_ranks)`` candidate with the fewest SPF adds.

    Ties resolve to the lowest index. Empty input raises.
    """

    if not candidates:
        raise SPFError("cheapest_tt_ordering requires at least one candidate")
    costs = [additive_recompression_cost(modes, ranks) for modes, ranks in candidates]
    return int(min(range(len(costs)), key=lambda i: costs[i]))


__all__ = [
    "STORAGE_SAFE_KINDS",
    "DecompositionKind",
    "DecompositionRecommendation",
    "additive_recompression_cost",
    "cheapest_tt_ordering",
    "is_storage_safe",
    "reject_cp_storage",
    "select_decomposition",
    "tt_quasi_optimality_constant",
]
