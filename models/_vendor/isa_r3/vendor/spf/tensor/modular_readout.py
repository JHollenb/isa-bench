"""Zero-parameter modular-character argmax readout (the algorithm *is* the model).

:mod:`spf.tensor.character` carries the *bind/lower* half of PAN's character
decompile — it takes a recovered :class:`~spf.tensor.character.CharacterBasisOperator`
and emits the un-collapsed bind plan. What it does **not** carry is the terminal
*readout*: turning the character bind back into a discrete prediction. That step
is closed-form and parameter-free, and it is the natural SPF boundary operator
for a modular-arithmetic circuit, so it belongs here.

For any AFFINE-modular task ``c = (α·a + β·b + γ) mod p`` the predictor

    pred(a, b) = argmax_c  Σ_{k∈freqs} cos( 2π·k·(α·a + β·b + γ − c) / p )

is **exact with zero learned parameters**. It is mechanism-faithful: every term
is a PER-INPUT phase combined by ADDITION — exactly what an untrained phase
accumulator computes (the a-encoder emits ``2π(kα)a/p``, the b-encoder
``2π(kβ)b/p``, the readout subtracts ``2πkc/p`` plus the bias), and the modular
reduction falls out of cosine periodicity. A single frequency already suffices
for the argmax (measured); the many frequencies a trained net uses are grokking
dynamics, not the function. This is the SPF-side readout that closes the
``χ_g(α·a+β·b+bias)`` character surface (spec: the T8 character suite).

**The multiplicative boundary, and its resolution.** The additive predictor
solves *additive*-group tasks; it cannot represent the multiplicative group, so
``a·b mod p`` scores near chance (the honest boundary PAN reports). SPF resolves
it with a **discrete-log change of basis**: over ``Z_p^* = ⟨g⟩``, ``dlog`` carries
multiplication to addition (``dlog(ab) = dlog(a)+dlog(b) mod (p−1)``), so the
*same* additive phase-accumulator argmax is exact in the exponent ring, then
``g^·`` maps back. Zero operands (``a=0`` or ``b=0`` → product 0) sit outside the
group and are handled explicitly. The certificate reports both the boundary and
its resolution.

Pure ``numpy``; no learned weights, no checkpoint, no float MACs in the SPF
sense — the readout is phase addition + an argmax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .character import CharacterBasisOperator
from .modular_arith import discrete_log_tables, primitive_root

__all__ = [
    "CharacterArgmaxReadout",
    "additive_predicts_multiplicative_accuracy",
    "analytic_affine_accuracy",
    "analytic_affine_argmax",
    "analytic_affine_logits",
    "analytic_modular_certificate",
    "discrete_log_tables",
    "min_frequencies_for_exact",
    "multiplicative_accuracy",
    "multiplicative_argmax",
    "primitive_root",
]

_TWO_PI = 2.0 * math.pi


# --------------------------------------------------------------------------- #
# Affine-modular character argmax (the additive group Z_p).
# --------------------------------------------------------------------------- #


def analytic_affine_logits(
    a: np.ndarray | int,
    b: np.ndarray | int,
    *,
    p: int,
    alpha: int,
    beta: int,
    gamma: int = 0,
    freqs: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Zero-parameter logits over ``c ∈ [0, p)`` for ``c = (αa+βb+γ) mod p``.

    Built from per-input phases combined by addition: the result has shape
    ``a.shape + (p,)``. ``freqs`` defaults to ``(1,)`` — one frequency is enough
    for the argmax on every affine task (see :func:`min_frequencies_for_exact`).
    """
    if freqs is None:
        freqs = (1,)
    av = np.asarray(a, dtype=np.int64)
    bv = np.asarray(b, dtype=np.int64)
    c = np.arange(p, dtype=np.int64).reshape(*([1] * av.ndim), p)
    af = av.astype(np.float64)[..., None]
    bf = bv.astype(np.float64)[..., None]
    cf = c.astype(np.float64)
    logits = np.zeros((*av.shape, p), dtype=np.float64)
    for k in freqs:
        phase = (_TWO_PI * k / p) * (alpha * af + beta * bf + gamma - cf)
        logits += np.cos(phase)
    return logits


def analytic_affine_argmax(
    a: np.ndarray | int,
    b: np.ndarray | int,
    *,
    p: int,
    alpha: int,
    beta: int,
    gamma: int = 0,
    freqs: tuple[int, ...] | None = None,
) -> np.ndarray:
    """The closed-form prediction ``argmax_c`` of :func:`analytic_affine_logits`."""
    return analytic_affine_logits(
        a, b, p=p, alpha=alpha, beta=beta, gamma=gamma, freqs=freqs
    ).argmax(-1)


def analytic_affine_accuracy(
    *, p: int, alpha: int, beta: int, gamma: int = 0, freqs: tuple[int, ...] | None = None
) -> float:
    """Exact accuracy over all ``p²`` ``(a, b)`` pairs — no sampling, no training."""
    a = np.arange(p, dtype=np.int64).reshape(p, 1)
    b = np.arange(p, dtype=np.int64).reshape(1, p)
    a_grid = np.broadcast_to(a, (p, p))
    b_grid = np.broadcast_to(b, (p, p))
    pred = analytic_affine_argmax(a_grid, b_grid, p=p, alpha=alpha, beta=beta, gamma=gamma, freqs=freqs)
    true = (alpha * a_grid + beta * b_grid + gamma) % p
    return float((pred == true).mean())


def min_frequencies_for_exact(
    *, p: int, alpha: int, beta: int, gamma: int = 0, max_k: int = 8
) -> int | None:
    """Smallest ``K`` (using ``freqs = 1..K``) giving exact accuracy; ``None`` if > ``max_k``."""
    for K in range(1, max_k + 1):
        if analytic_affine_accuracy(
            p=p, alpha=alpha, beta=beta, gamma=gamma, freqs=tuple(range(1, K + 1))
        ) >= 1.0 - 1e-9:
            return K
    return None


# --------------------------------------------------------------------------- #
# The multiplicative boundary and its discrete-log resolution (Z_p^*).
# --------------------------------------------------------------------------- #


def multiplicative_argmax(
    a: np.ndarray | int,
    b: np.ndarray | int,
    *,
    p: int,
    generator: int | None = None,
    freqs: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Exact ``a·b mod p`` via the discrete-log change of basis (additive in ``Z_{p-1}``).

    For nonzero operands the same additive phase-accumulator argmax runs in the
    exponent ring (``k = dlog a + dlog b``), then ``g^k`` maps back. Either operand
    zero ⇒ product 0 (outside ``Z_p^*``), handled explicitly. This is the SPF
    resolution of the multiplicative boundary that the bare additive predictor
    (:func:`additive_predicts_multiplicative_accuracy`) cannot cross.
    """
    av = np.asarray(a, dtype=np.int64)
    bv = np.asarray(b, dtype=np.int64)
    dlog, exp_table, _ = discrete_log_tables(p, generator)
    m = p - 1
    nz = (av != 0) & (bv != 0)
    out = np.zeros(np.broadcast(av, bv).shape, dtype=np.int64)
    if np.any(nz):
        ka = dlog[np.where(av == 0, 1, av)]
        kb = dlog[np.where(bv == 0, 1, bv)]
        # Additive character argmax in the exponent ring Z_{p-1} (mechanism-
        # faithful: exponents add), equivalent to (ka+kb) mod (p−1) but built from
        # the same cosine phase sum as the affine readout.
        k_pred = analytic_affine_argmax(ka, kb, p=m, alpha=1, beta=1, freqs=freqs)
        mapped = exp_table[k_pred % m]
        out = np.where(np.broadcast_to(nz, out.shape), mapped, out)
    return out


def multiplicative_accuracy(
    *, p: int, generator: int | None = None, freqs: tuple[int, ...] | None = None
) -> float:
    """Exact accuracy of :func:`multiplicative_argmax` over all ``p²`` pairs (≈ 1.0)."""
    a = np.arange(p, dtype=np.int64).reshape(p, 1)
    b = np.arange(p, dtype=np.int64).reshape(1, p)
    a_grid = np.broadcast_to(a, (p, p))
    b_grid = np.broadcast_to(b, (p, p))
    pred = multiplicative_argmax(a_grid, b_grid, p=p, generator=generator, freqs=freqs)
    true = (a_grid * b_grid) % p
    return float((pred == true).mean())


def additive_predicts_multiplicative_accuracy(*, p: int) -> float:
    """Honest boundary probe: the bare additive predictor scored on ``a·b mod p``.

    The additive phase model only ever predicts ``(a+b) mod p``; scored against
    the multiplicative target it lands near chance — the additive group structure
    cannot represent the multiplicative one without a discrete-log basis change.
    """
    a = np.arange(p, dtype=np.int64).reshape(p, 1)
    b = np.arange(p, dtype=np.int64).reshape(1, p)
    a_grid = np.broadcast_to(a, (p, p))
    b_grid = np.broadcast_to(b, (p, p))
    pred = analytic_affine_argmax(
        a_grid, b_grid, p=p, alpha=1, beta=1, freqs=tuple(range(1, (p - 1) // 2 + 1))
    )
    true = (a_grid * b_grid) % p
    return float((pred == true).mean())


# --------------------------------------------------------------------------- #
# The SPF readout operator + certificate.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CharacterArgmaxReadout:
    """A terminal, zero-parameter modular-character readout over ``Z_p``.

    Carries the closed-form ``c = (αa + βb + γ) mod p`` predictor. Build it
    directly, or :meth:`from_character_operator` to read it off a recovered
    :class:`~spf.tensor.character.CharacterBasisOperator` (the same modulus and
    affine coefficients the bind plan was lowered from). ``freqs`` defaults to a
    single frequency — exact for every affine task.
    """

    p: int
    alpha: int = 1
    beta: int = 1
    gamma: int = 0
    freqs: tuple[int, ...] = (1,)

    @classmethod
    def from_character_operator(cls, op: CharacterBasisOperator) -> CharacterArgmaxReadout:
        """Read the readout off a recovered character operator's affine surface."""
        alpha, beta = op.affine_coefficients_mod
        return cls(p=int(op.modulus), alpha=int(alpha), beta=int(beta), gamma=int(op.affine_bias_mod))

    def logits(self, a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        return analytic_affine_logits(
            a, b, p=self.p, alpha=self.alpha, beta=self.beta, gamma=self.gamma, freqs=self.freqs
        )

    def predict(self, a: np.ndarray | int, b: np.ndarray | int) -> np.ndarray:
        return self.logits(a, b).argmax(-1)

    def accuracy(self) -> float:
        return analytic_affine_accuracy(
            p=self.p, alpha=self.alpha, beta=self.beta, gamma=self.gamma, freqs=self.freqs
        )


def analytic_modular_certificate(p: int = 113) -> dict[str, Any]:
    """Certify zero-parameter accuracy across affine tasks + the multiplicative boundary/fix."""
    affine_tasks = [
        {"name": "mod_add", "alpha": 1, "beta": 1, "gamma": 0},
        {"name": "mod_sub", "alpha": 1, "beta": -1, "gamma": 0},
        {"name": "affine_2a+3b", "alpha": 2, "beta": 3, "gamma": 0},
        {"name": "affine_7a-7b+11", "alpha": 7, "beta": -7, "gamma": 11},
    ]
    rows = []
    for t in affine_tasks:
        acc1 = analytic_affine_accuracy(p=p, alpha=t["alpha"], beta=t["beta"], gamma=t["gamma"], freqs=(1,))
        minK = min_frequencies_for_exact(p=p, alpha=t["alpha"], beta=t["beta"], gamma=t["gamma"])
        rows.append({**t, "accuracy_single_freq": acc1, "min_freqs_for_exact": minK, "learned_parameters": 0})

    boundary_acc = additive_predicts_multiplicative_accuracy(p=p)
    dlog_acc = multiplicative_accuracy(p=p)
    g = primitive_root(p)
    return {
        "schema": "spf_modular_readout_certificate_v1",
        "p": p,
        "affine_tasks": rows,
        "all_affine_exact_single_freq": all(r["accuracy_single_freq"] >= 1.0 - 1e-9 for r in rows),
        "multiplicative_boundary": {
            "task": "mod_mul (a*b mod p)",
            "additive_predictor_accuracy_all_freqs": boundary_acc,
            "note": "additive character predictor does NOT solve the multiplicative group at chance-level",
        },
        "multiplicative_resolution": {
            "method": "discrete-log change of basis over Z_p^* = <g>",
            "generator": g,
            "accuracy": dlog_acc,
            "note": "dlog carries a*b to addition in Z_{p-1}; the same phase-accumulator argmax is exact",
        },
        "claim_boundary": (
            "exact zero-parameter accuracy over all p^2 pairs for affine-modular tasks via a "
            "closed-form Fourier-character argmax (the algorithm is the model). Multiplicative "
            "tasks need a discrete-log basis change, after which the same additive readout is exact."
        ),
    }
