"""Plan-time algebraic rewrite registry over :class:`~spf.tensor.v2.ExecutionPlan`.

SPFTensor already carries the operator vocabulary (``fft``/``ifft``,
``diagonal_scale``, ``reduce``, ``gather``/``permute``, ...). What it lacks is a
layer that recognises algebraic identities *between* those steps and emits a
cheaper, semantically equivalent plan. This module is that layer: a small
provenance-aware rule engine that runs to a fixpoint and records an auditable
cost/error trail for every rewrite it applies.

Design rules (deliberate non-goals):

* No new storage profile, no new ``AccumulationDomain``, no new execution op.
  Rewrites only delete/merge/retag existing steps.
* Orthogonality is provenance-sensitive — a reducer only sees packed words and
  cannot know a tensor is an exact DFT row. So the orthogonality rewrite lives
  here, at plan time, behind a hard provenance gate (:class:`Provenance`), never
  inside a reducer.
* ``decoded_contract_reference`` stays the FP64 oracle/fallback; rewrites may
  *eliminate* it (closed form) but never promote it to an owned dense lane.

The cost numbers are *planner weights*, not measured bounds — the same
convention as :mod:`spf.tensor.reducers` (``C_ADD``/``C_DEC``/``C_ENC``).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_NAN,
    SPF32_L_ZERO,
    SPF32_WORD_ZERO,
    AccumulationDomain,
    PrecisionMode,
    ReducerPolicy,
    RoutingPolicy,
    SPFError,
    decode_words,
    encode_words,
    spf_mul_words,
    unpack_L,
)
from .op_registry import OP_REGISTRY, cost_weights
from .v2 import ExecutionPlan, ExecutionStep

# --------------------------------------------------------------------------- #
# Cost model (planner weights, calibratable from measured kernel timings)
# --------------------------------------------------------------------------- #

# Per-step expensive-op-equivalent weight. fft/ifft dominate (≈log2 N butterfly
# passes); contractions decode-heavy; gather/permute are a cheap host pass but
# non-zero so composing two into one is a measurable win. These are *planner
# weights*, not measured bounds — the same convention as :mod:`spf.tensor.reducers`
# (``C_ADD``/``C_DEC``/``C_ENC``). Calibrate against real cycles with
# :func:`cost_model_from_timings`.
# Seeded from the op registry (the single source of truth for both the op
# whitelist and these planner weights). This is the *builtin* snapshot; an op
# registered at runtime is still costed correctly because ``OpCostModel.cost``
# falls back to the live registry before its ``default`` — see ``cost``.
_OP_COST: dict[str, float] = cost_weights()


# fft/ifft work scales as N·log₂N. The ``_OP_COST`` weight (6.0) is the cost at
# this reference N, so ``step_cost`` scales it by ``N·log₂N / (REF·log₂REF)`` —
# the weight is *inert* at N = 64 (factor 1.0) and rises for larger transforms, so
# a plan mixing a 64-pt and a 4096-pt FFT ranks the big one correctly.
_FFT_REF_N = 64
_FFT_REF_WORK = _FFT_REF_N * math.log2(_FFT_REF_N)

# Vector ops whose work is O(N) (one host/native pass over the data length). Their
# weight scales linearly in the vector length ``params["n"]`` — the elementwise
# sibling of the fft ``N·log₂N`` scaling — anchored at the same N=64 reference so a
# step that does not carry ``n`` is unscaled (every existing plan is inert).
_VECTOR_N_OPS = ("diagonal_scale", "elementwise_mul", "ramp", "gather", "permute")


@dataclass(frozen=True)
class OpCostModel:
    """Per-op planner weights.

    ``cost(op)`` resolves in three tiers: an explicit ``weights`` entry (a
    calibrated override or the builtin snapshot) wins; failing that the *live*
    op registry's declared ``cost_weight`` is used, so an op registered at
    runtime is costed at its declared weight rather than silently dropping to
    ``default``; an op unknown to both falls back to ``default``.
    """

    weights: Mapping[str, float] = field(default_factory=lambda: dict(_OP_COST))
    default: float = 1.0

    def cost(self, op: str) -> float:
        weight = self.weights.get(op)
        if weight is not None:
            return float(weight)
        spec = OP_REGISTRY.get(op)
        if spec is not None:
            return float(spec.cost_weight)
        return self.default

    def step_cost(self, step: ExecutionStep) -> float:
        """Cost of one step, length-scaling the work-proportional ops (inert at N=64).

        ``cost(op)`` stays the flat per-op weight (back-compat / unknown-N); this
        is the step-aware variant the plan cost uses so large-N steps rank above
        small ones under the same calibrated weights:

        * ``fft`` / ``ifft`` scale by ``N·log₂N`` in ``params["n"]``;
        * the O(N) vector ops (:data:`_VECTOR_N_OPS`) scale linearly in
          ``params["n"]`` — so dropping a data-length ``diagonal_scale`` is correctly
          worth ``N`` at large N, not a flat 1.0 (Speculation #9);
        * a bounded sparse ``reduce`` scales by its ``n_terms`` (atom count).

        All three are anchored at the ``N = 64`` reference and read an explicit
        length param, so a step without it keeps its flat weight (every existing
        plan is unchanged).
        """
        base = self.cost(step.op)
        if step.op == "phase_character_features":
            return base * max(1, int(step.params.get("n_terms", 0)) + len(step.params.get("characters", ())))
        if step.op == "phase_shear":
            return base * max(1, len(step.params.get("coefficients", (0, 0, 1))))
        if step.op == "phase_neuron":
            return base * max(1, int(step.params.get("n_terms", 0)) + len(step.params.get("units", ())))
        if step.op in ("fft", "ifft"):
            n = int(step.params.get("n", _FFT_REF_N))
            if n > 1:
                return base * (n * math.log2(n)) / _FFT_REF_WORK
        if step.op in _VECTOR_N_OPS:
            n = int(step.params.get("n", _FFT_REF_N))
            if n > 1:
                return base * float(n) / _FFT_REF_N
        if step.op == "reduce":
            # A bounded sparse reduce summing k terms (a k-atom character surface)
            # costs ∝ k. ``n_terms`` is the reduced-axis length; absent (a plain
            # reduce) ⇒ 1, so every existing reduce keeps its flat weight.
            n_terms = int(step.params.get("n_terms", 1))
            if n_terms > 1:
                return base * float(n_terms)
        return base

    def plan_cost(self, plan: ExecutionPlan) -> float:
        """Total step-aware cost of a plan — the one place the sum lives."""
        return float(sum(self.step_cost(step) for step in plan.steps))


DEFAULT_COST_MODEL = OpCostModel()

# Map a measured PL/native kernel name to the ExecutionPlan op(s) it backs.
_TIMING_OP_ALIASES: dict[str, tuple[str, ...]] = {
    "fft64": ("fft", "ifft"),
    "fft": ("fft", "ifft"),
    "mul": ("diagonal_scale", "elementwise_mul"),
    "ramp": ("ramp",),
    "reduce": ("reduce",),
    "low_rank": ("low_rank_apply",),
}


def calibrate_cost_model(
    measured: Mapping[str, float], *, anchor: str = "mul"
) -> OpCostModel:
    """Normalise measured per-op costs (ns / cycles) to ``anchor == 1.0`` weights.

    ``measured`` is keyed by ExecutionPlan op name. The anchor op's measured cost
    becomes weight 1.0 and every other op is expressed as a multiple of it, so the
    rewrite savings guard reflects real relative cost instead of the nominal
    defaults. Ops absent from ``measured`` keep their default weight.
    """
    base = float(measured.get(anchor, 0.0))
    if base <= 0.0:
        raise ValueError(f"anchor op {anchor!r} missing or non-positive in measured costs")
    weights = dict(_OP_COST)
    weights.update({op: float(c) / base for op, c in measured.items()})
    return OpCostModel(weights=weights)


def cost_model_from_timings(
    ns_by_kernel: Mapping[str, float], *, anchor_kernel: str = "mul"
) -> OpCostModel:
    """Build a calibrated model from kernel-keyed timings (e.g. FB7/FFT native).

    ``ns_by_kernel`` is keyed by *kernel* name (``"fft64"``, ``"mul"``, ...); each
    is mapped to the ExecutionPlan op(s) it backs via :data:`_TIMING_OP_ALIASES`
    and normalised to ``anchor_kernel``. Host-only ops (gather/permute) keep their
    default weight. This is the bridge from ``docs/timings-schema.md`` artifacts to
    the planner's savings guard.
    """
    base = float(ns_by_kernel.get(anchor_kernel, 0.0))
    if base <= 0.0:
        raise ValueError(f"anchor kernel {anchor_kernel!r} missing or non-positive")
    weights = dict(_OP_COST)
    for kernel, ns in ns_by_kernel.items():
        for op in _TIMING_OP_ALIASES.get(kernel, (kernel,)):
            weights[op] = float(ns) / base
    return OpCostModel(weights=weights)


def _step_cost(step: ExecutionStep, model: OpCostModel = DEFAULT_COST_MODEL) -> float:
    return model.step_cost(step)


def _plan_cost(plan: ExecutionPlan, model: OpCostModel = DEFAULT_COST_MODEL) -> float:
    return model.plan_cost(plan)


# SPF word for 1.0 + 0j (L code 0, Θ 0) — multiplying by it is identity.
_ONE_WORD = np.uint32(encode_words(np.asarray(1.0 + 0.0j)))


def _mul_l_saturates(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff ``spf_mul_words(a, b)`` would clamp the log-magnitude (L) plane.

    SPF multiply is an integer L-add + a *cyclic* Θ-add. Θ wraps exactly (the
    phase group ``Z/2¹⁶``), but L is a saturating signed fixed-point — ``_sat_l``
    clamps the sum to ``[L_MIN_FINITE, L_MAX_FINITE]``. When that clamp fires the
    fused coefficient is no longer the bit-exact product, so a coefficient-folding
    rule that claims ``exact`` would be lying. This predicts the clamp from the raw
    L-code sum (zero-sentinel operands excepted — those decode to 0 either way) so
    the fold can simply *refuse* on saturation rather than fold-and-misreport.
    """
    aL = unpack_L(np.asarray(a, dtype=np.uint32)).astype(np.int64)
    bL = unpack_L(np.asarray(b, dtype=np.uint32)).astype(np.int64)
    aL, bL = (x.reshape(-1) for x in np.broadcast_arrays(aL, bL))
    nz = (aL != int(SPF32_L_ZERO)) & (bL != int(SPF32_L_ZERO))
    if not bool(np.any(nz)):
        return False
    s = aL[nz] + bL[nz]
    return bool(np.any((s < SPF32_L_MIN_FINITE) | (s > SPF32_L_MAX_FINITE)))


# --------------------------------------------------------------------------- #
# Provenance / context / match records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Provenance:
    """What a plan env-var *is* — supplied out-of-band, never read from words.

    A reducer cannot recover this from packed ``(L, Θ)`` words, so the planner
    must be told. Orthogonality and other provenance-sensitive rewrites only
    fire when the caller has vouched for the tensor's structure here.
    """

    kind: str = "generic"  # "dft_row" | "phase_ramp" | "constant_modulus" | "generic"
    N: int | None = None
    index: int | None = None  # DFT row index f
    axis_coeff: int | None = None  # affine input coeff α: this row is index = α·g for a
    # shared base generator g (None ⇒ 1). Lets CharacterPhaseBind fuse an *affine*
    # bind χ_{αg}(a)·χ_{βg}(b) -> χ_g((αa+βb) mod N), not only the sum case α=β=1.
    index_bank: tuple[int, ...] | None = None  # per-atom DFT rows — a *bank* of k
    # characters stacked on one axis (kind="dft_row_bank"). The batched sibling of
    # `index`: a sparse 2-D character surface Σ_j c_j·χ_{(u_j,v_j)}(a,b) is k separable
    # binds at once. Lets CharacterBankBind fuse the whole bank in one rewrite.
    axis_coeff_bank: tuple[int, ...] | None = None  # per-atom input coeffs for the bank
    moduli: tuple[int, ...] | None = None  # per-factor moduli of a PRODUCT group
    # ∏ Z/nᵢ (kind="product_dft_row"). The character index `index` is a flat
    # mixed-radix frequency over the factors; the bind adds *componentwise* mod nᵢ
    # (no carry), which CRT cannot collapse to a single modulus for non-coprime
    # factors (Walsh-Hadamard (Z/2)^k, mixed radix). Lets ProductGroupCharacterBind fire.
    exact: bool = False  # exact constant-magnitude SEPARABLE_PHASE basis object
    conjugated: bool = False  # this env-var already holds conj(row)
    has_residual: bool = False
    windowed: bool = False
    noisy: bool = False

    def is_exact_dft_row(self) -> bool:
        """True iff this is an exact, residue-free, unwindowed, noiseless DFT row."""
        return bool(
            self.kind == "dft_row"
            and self.exact
            and self.N is not None
            and self.index is not None
            and not self.has_residual
            and not self.windowed
            and not self.noisy
        )

    def is_exact_dft_row_bank(self) -> bool:
        """True iff this is an exact bank of DFT rows (the batched character form)."""
        return bool(
            self.kind == "dft_row_bank"
            and self.exact
            and self.N is not None
            and self.index_bank is not None
            and not self.has_residual
            and not self.windowed
            and not self.noisy
        )

    def is_exact_product_dft_row(self) -> bool:
        """True iff this is an exact character of a product group ``∏ Z/nᵢ``."""
        return bool(
            self.kind == "product_dft_row"
            and self.exact
            and self.moduli is not None
            and self.index is not None
            and not self.has_residual
            and not self.windowed
            and not self.noisy
        )

    def linear_phase_shift(self) -> int | None:
        """Integer circular shift this Fourier diagonal applies, or ``None``.

        ``dft_row`` (index ``f``) and ``phase_ramp`` (shift ``s``) are the *same*
        object — an exact unit-modulus linear-phase diagonal, which between an
        ``fft`` and an ``ifft`` *is* a circular shift (DFT shift / modulation
        theorem). This is the single source of truth both tags collapse into: a
        ``phase_ramp`` carries the shift directly; an exact ``dft_row`` of index
        ``f`` is the diagonal ``exp(+i·2π·m·f/N)`` whose ``fft→·→ifft`` shifts by
        ``(-f) mod N``. Any other / inexact / under-specified provenance ⇒ ``None``.
        """
        if not self.exact or self.N is None or self.index is None:
            return None
        if self.kind == "phase_ramp":
            return int(self.index) % int(self.N)
        if self.kind == "dft_row":
            return (-int(self.index)) % int(self.N)
        return None

    def is_linear_phase_diagonal(self) -> bool:
        """True iff this is an exact unit-modulus linear-phase diagonal (a roll).

        The one "is this a circular shift?" gate every roll-aware rule should read,
        subsuming both the ``phase_ramp`` and exact-``dft_row`` spellings. See
        :meth:`linear_phase_shift`.
        """
        return self.linear_phase_shift() is not None


def phase_ramp_shift_provenance(n: int, shift: int) -> Provenance:
    """Provenance for a Fourier-domain linear-phase diagonal that *is* a roll.

    A diagonal ``h[m] = exp(-i·2π·m·s/N)`` applied to ``fft(x)`` and inverted is a
    circular shift of ``x`` by ``s`` (DFT shift theorem). Tag the coefficient with
    this so :class:`RollThroughFourier` can replace the whole
    ``fft → diagonal_scale → ifft`` triple with a single ``gather``. ``index``
    carries the integer shift ``s``.
    """
    return Provenance(
        kind="phase_ramp", N=int(n), index=int(shift) % int(n), exact=True
    )


def fit_phase_ramp(
    h_words: Sequence[int] | np.ndarray,
    *,
    modulus_tol: float = 1e-3,
    evm_tol: float = 1e-3,
) -> Provenance | None:
    """Derive roll provenance from *measured* coefficient words, or refuse.

    A Fourier-domain diagonal is an exact circular shift only when it is a
    **unit-modulus linear phase** ``h[m] = exp(-i·2π·m·s/N)`` (the DFT shift
    theorem leaves the magnitude alone, so a non-unit modulus would scale the
    output that the bare ``gather`` does not). This fits the integer shift ``s``
    from the words and returns :func:`phase_ramp_shift_provenance` only when the
    fit is within ``evm_tol`` *and* the modulus is unit within ``modulus_tol`` —
    otherwise ``None``, never a false roll. The honesty gate (refuse on residual)
    is the whole point: this is what lets :class:`RollThroughFourier` fire on a
    real channel coefficient instead of a hand-constructed one.
    """
    h = decode_words(np.asarray(h_words, dtype=np.uint32)).reshape(-1)
    n = int(h.size)
    if n < 2:
        return None
    mag = np.abs(h)
    mean_mag = float(np.mean(mag))
    # Unit, constant modulus — both required for a bare reindex to be exact.
    if mean_mag <= 0.0 or abs(mean_mag - 1.0) > modulus_tol:
        return None
    if float(np.std(mag)) > modulus_tol:
        return None
    idx = np.arange(n, dtype=np.float64)
    slope = float(np.polyfit(idx, np.unwrap(np.angle(h)), 1)[0])
    # h[m] = exp(-i·2π·m·s/N) ⇒ slope = -2π·s/N ⇒ s = -slope·N/(2π).
    s = round(-slope * n / (2.0 * math.pi)) % n
    h_fit = np.exp(-1j * 2.0 * math.pi * idx * s / n)
    err = float(np.sqrt(np.mean(np.abs(h - h_fit) ** 2)))
    if err > evm_tol:
        return None
    return phase_ramp_shift_provenance(n, s)


def provenance_from_v3(block: Any, *, phase_bits: int = 16) -> Provenance:
    """Derive :class:`Provenance` structurally from a :class:`~spf.tensor.v3.SPFBlockV3`.

    Closes the loop so callers need not hand-supply provenance: a block that is
    ``SEPARABLE_PHASE`` base × ``NONE`` residual × no outliers, with a single
    closed-form phase term over a constant ``ω`` and no log-magnitude content
    (constant modulus), is an exact DFT row when ``2^phase_bits`` is divisible by
    ``N`` and ``ω`` lands on an integer row index (no phase rounding). Otherwise a
    conservative ``generic`` provenance is returned — never a false ``exact``.
    """
    from .v3 import BaseModelKind, ResidualCodecKind

    dims = tuple(int(d) for d in getattr(block, "dims", ()) or ())
    n = int(np.prod(dims, dtype=np.int64)) if dims else 1
    generic = Provenance(kind="generic", N=n)
    if int(block.base.profile_id) != int(BaseModelKind.SEPARABLE_PHASE):
        return generic
    if int(block.residual.codec_id) != int(ResidualCodecKind.NONE):
        return generic
    if block.outliers:
        return generic
    params = block.base.params
    # Constant modulus: no log-magnitude content, no phase bias.
    if params.get("logmag_terms") or params.get("logmag_bias") or params.get("phase_bias"):
        return generic
    terms = list(params.get("phase_terms", ()))
    if len(terms) != 1:
        return generic
    term = terms[0]
    if not term.get("closed_form"):
        return generic
    omega = np.asarray(term.get("omega", ()), dtype=np.int64).reshape(-1)
    if omega.size == 0 or not bool(np.all(omega == omega.flat[0])):
        return generic
    c = int(omega.flat[0])
    mod = 1 << int(phase_bits)
    # Exact iff the per-index phase increment c == f·mod/N for an integer row f.
    if n <= 0 or mod % n != 0 or (c * n) % mod != 0:
        return generic
    f = (c * n) // mod % n
    return Provenance(kind="dft_row", N=n, index=int(f), exact=True)


def spectral_map(
    h_words: Sequence[int] | np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Spectral mapping theorem: ``f(C(h)) = C(f(ĥ))`` — the spine of the algebra.

    A circulant ``C(h)`` is diagonalised by the DFT (the isomorphism
    ``C[Z/N] ≅ Cᴺ``), so *any* analytic function of it is the circulant of the
    pointwise-mapped spectrum. Decode the spectrum, apply the scalar map ``f``
    elementwise, re-encode — the result is the kernel of ``f(C)``. The FDWM matrix
    power (``f(z)=z^k``) and the circulant reciprocal (``f(z)=1/z``) are special
    cases; this opens the rest of the functional calculus for free:

    * **inverse / deconvolution** — ``f(z)=1/z``
    * **resolvent / structured linear solve** — ``f(z)=1/(1−w·z)`` gives
      ``(I−w·C)⁻¹`` in one FFT, a pointwise reciprocal, one IFFT
    * **matrix exponential** (linear ODE / heat / diffusion) — ``f(z)=exp(t·z)``
    * **square root / whitening** — ``f(z)=sqrt(z)``
    * **geometric series / iterative-solver preconditioner** — ``f(z)=Σ wᵏzᵏ``

    Returns the packed-word kernel ``f(ĥ)``; feed it as a circulant coefficient
    (``fft → diagonal_scale(f(ĥ)) → ifft``). Exact up to SPF rounding of the mapped
    spectrum; the caller owns ``f``'s own numerical conditioning (e.g. ``ĥ`` away
    from the singularities of ``f``).
    """
    spectrum = decode_words(np.asarray(h_words, dtype=np.uint32))
    return encode_words(np.asarray(f(spectrum), dtype=np.complex128))


@dataclass(frozen=True)
class RewriteContext:
    """Inputs the rule engine needs beyond the plan itself.

    ``constants`` maps env-var names to packed-word arrays whose values are known
    at plan time (channel inverses, matched-filter spectra, twiddles). Only such
    coefficients can be folded together (e.g. diagonal-scale composition).
    """

    error_budget: float = 0.0
    mode: PrecisionMode = PrecisionMode.BALANCED
    provenance: Mapping[str, Provenance] = field(default_factory=dict)
    constants: Mapping[str, np.ndarray] = field(default_factory=dict)
    cost_model: OpCostModel = DEFAULT_COST_MODEL
    rounding_contract: str = "packed_steps"
    input_l_ranges: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    input_shapes: Mapping[str, tuple[int, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rounding_contract not in {"packed_steps", "algebraic_output"}:
            raise ValueError("rounding_contract must be packed_steps or algebraic_output")
        for name, (lo, hi) in self.input_l_ranges.items():
            if not SPF32_L_MIN_FINITE <= lo <= hi <= SPF32_L_MAX_FINITE:
                raise ValueError(f"invalid finite input L range for {name!r}")


@dataclass(frozen=True)
class RewriteMatch:
    """One applied (or applicable) rewrite, for the audit log."""

    name: str
    confidence: float
    exact: bool
    reason: str
    old_cost: float
    new_cost: float
    error_bound: float | None
    step_span: tuple[int, ...]
    annotation: bool = False  # metadata-only retag (exempt from the savings guard)
    algebraic: bool = False

    @property
    def exactness_scope(self) -> str:
        """Exact operator algebra is distinct from packed-step word parity."""
        return "algebraic_output" if self.algebraic or self.name in _ALGEBRAIC_OUTPUT_RULES else "packed_steps"


# --------------------------------------------------------------------------- #
# Dataflow helpers
# --------------------------------------------------------------------------- #


def _producer(plan: ExecutionPlan) -> dict[str, int]:
    """Map each output env-var name to the step index that writes it (SSA)."""
    out: dict[str, int] = {}
    for i, step in enumerate(plan.steps):
        for name in step.outputs:
            if name in out:
                raise ValueError(f"plan is not SSA: {name!r} written by two steps")
            out[name] = i
    return out


def _use_count(plan: ExecutionPlan, name: str) -> int:
    return sum(name in step.inputs for step in plan.steps) + int(
        name in plan.factorization_metadata.get("observed_outputs", ())
    )


def _remap_inputs(
    steps: tuple[ExecutionStep, ...], rename: Mapping[str, str]
) -> tuple[ExecutionStep, ...]:
    """Substitute input names across steps (outputs are left untouched)."""
    out: list[ExecutionStep] = []
    for step in steps:
        new_inputs = tuple(rename.get(n, n) for n in step.inputs)
        out.append(replace(step, inputs=new_inputs) if new_inputs != step.inputs else step)
    return tuple(out)


def apply_aliases(env: Mapping[str, Any], plan: ExecutionPlan) -> dict[str, Any]:
    """Resolve rewrite output aliases into an execute() result env.

    When a rewrite cancels a *terminal* output (e.g. ``ifft(fft(x))`` with ``y``
    nobody consumes), the producing steps vanish and ``y`` simply aliases ``x``.
    Such aliases are recorded in ``plan.factorization_metadata["aliases"]``; this
    fills them back in so callers can still query the original output name.
    """
    aliases: Mapping[str, str] = plan.factorization_metadata.get("aliases", {})
    out = dict(env)
    for dst in aliases:
        src = dst
        seen = set()
        while src in aliases and src not in seen:
            seen.add(src)
            src = aliases[src]
        if dst not in out and src in out:
            out[dst] = out[src]
    return out


def run_rewritten(
    plan: ExecutionPlan,
    inputs: Mapping[str, Any],
    folded: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Execute a rewritten plan the safe way — merge ``folded`` and resolve aliases.

    Two sharp edges of running a rewrite output by hand: (1) the fused/closed-form
    constants in ``folded`` must be added to the execute env, and (2) a *terminal*
    output a full cancel erased (``ifft(fft(x))`` with nobody consuming ``y``)
    survives only as an alias in ``factorization_metadata`` — ``plan.execute`` alone
    raises ``KeyError`` on it. This wraps both: ``apply_aliases(plan.execute({**inputs,
    **folded}), plan)``. Prefer it over a bare ``plan.execute`` for any plan that has
    been through :func:`rewrite_plan`.
    """
    env = {**dict(inputs), **dict(folded or {})}
    return apply_aliases(plan.execute(env), plan)


def _drop_step_alias(
    plan: ExecutionPlan, k: int, alias_src: str
) -> ExecutionPlan:
    """Drop step ``k`` whose single output equals ``alias_src``; remap + record alias.

    Shared by the no-op prune and identity rules: the dropped step's output simply
    aliases an existing value, so downstream uses are renamed and a terminal alias
    is recorded for :func:`apply_aliases`.
    """
    dropped = plan.steps[k]
    rename = {dropped.outputs[0]: alias_src}
    kept = tuple(s for i, s in enumerate(plan.steps) if i != k)
    meta = dict(plan.factorization_metadata)
    aliases = dict(meta.get("aliases", {}))
    aliases[dropped.outputs[0]] = alias_src
    meta["aliases"] = aliases
    return ExecutionPlan(steps=_remap_inputs(kept, rename), factorization_metadata=meta)


# --------------------------------------------------------------------------- #
# Rule protocol + registry driver
# --------------------------------------------------------------------------- #


@runtime_checkable
class RewriteRule(Protocol):
    """A peephole rewrite over an :class:`ExecutionPlan`.

    Invariant (load-bearing): a rule's site selector (``_find``) must be a **pure,
    deterministic** function of ``(plan, ctx)`` with no dependence on ``folded`` —
    the driver calls ``match`` then ``apply`` on the *same* plan, and both re-derive
    the site via that selector, so they must agree on which site they picked. Reading
    mutable fold state inside ``_find`` would let the two diverge. (``apply`` may of
    course *write* ``folded`` once it has the site.)
    """

    name: str

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None: ...

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan: ...


_ALGEBRAIC_OUTPUT_RULES = frozenset({
    "fft_ifft_cancel", "double_fft_reversal", "roll_through_fourier",
    "diag_low_rank_fold", "reduce_circulant_spectral", "decimation_fold",
    "reduce_permute_invariance",
    "phase_ramp_orthogonality",
    "character_phase_bind", "character_bank_bind", "product_group_character_bind",
})


def _rule_fires(m: RewriteMatch, ctx: RewriteContext) -> bool:
    """The savings/error guard: a non-annotation rewrite must lower cost honestly."""
    if m.exactness_scope == "algebraic_output" and ctx.rounding_contract != "algebraic_output":
        return False
    return m.annotation or (
        m.new_cost < m.old_cost
        and (m.exact or (m.error_bound is not None and m.error_bound <= ctx.error_budget))
    )


def _remap_boundary_metadata(old: ExecutionPlan, new: ExecutionPlan) -> ExecutionPlan:
    """Keep declared structured boundary indices aligned after deleting steps."""
    meta = dict(new.factorization_metadata)
    for key in ("structured_reducer_boundaries", "structured_requantization_points"):
        if key not in meta:
            continue
        indices: set[int] = set()
        for index in old.factorization_metadata.get(key, ()):
            outputs = set(old.steps[int(index)].outputs)
            indices.update(i for i, step in enumerate(new.steps) if outputs.intersection(step.outputs))
        meta[key] = tuple(sorted(indices))
    return ExecutionPlan(steps=new.steps, factorization_metadata=meta)


def _rewrite_first(
    plan: ExecutionPlan,
    ctx: RewriteContext,
    active_rules: Sequence[RewriteRule],
    *,
    greedy: bool = False,
) -> tuple[ExecutionPlan, list[RewriteMatch], dict[str, np.ndarray]]:
    """Single-path fixpoint: fire one rule per step until none applies.

    ``greedy=False`` fires the first applicable rule in ``active_rules`` order (the
    shipped heuristic); ``greedy=True`` fires the largest-saving applicable rule.
    Both walk *one* path to a fixpoint — :func:`_rewrite_beam` is the lookahead
    variant that escapes the local optima both of these fall into (Speculation #8).
    """
    matches: list[RewriteMatch] = []
    folded: dict[str, np.ndarray] = {}
    budget_iters = max(8, len(plan.steps) * 8)  # backstop vs a looping rule
    changed = True
    while changed and budget_iters > 0:
        changed = False
        budget_iters -= 1
        # Folded constants (fused coeffs, closed forms) become available to later
        # rewrites — so e.g. a chained circulant fully collapses to one diagonal
        # (FDWM power) as each pairwise fusion's product feeds the next.
        work_ctx = replace(ctx, constants={**ctx.constants, **folded}) if folded else ctx
        if greedy:
            # Score every applicable rule; fire the largest-saving one. Strict ``>``
            # keeps the first rule in registry order on a tie, so it is deterministic.
            best: tuple[float, RewriteRule, RewriteMatch] | None = None
            for rule in active_rules:
                m = rule.match(plan, work_ctx)
                if m is None or not _rule_fires(m, ctx):
                    continue
                saving = m.old_cost - m.new_cost
                if best is None or saving > best[0]:
                    best = (saving, rule, m)
            if best is not None:
                _, rule, m = best
                plan = _remap_boundary_metadata(plan, rule.apply(plan, work_ctx, folded))
                matches.append(m)
                changed = True
            continue
        for rule in active_rules:
            m = rule.match(plan, work_ctx)
            if m is None or not _rule_fires(m, ctx):
                continue
            plan = _remap_boundary_metadata(plan, rule.apply(plan, work_ctx, folded))
            matches.append(m)
            changed = True
            break
    return plan, matches, folded


def _rewrite_beam(
    plan: ExecutionPlan,
    ctx: RewriteContext,
    active_rules: Sequence[RewriteRule],
    *,
    beam_width: int,
) -> tuple[ExecutionPlan, list[RewriteMatch], dict[str, np.ndarray]]:
    """Beam search over reachable fixpoints; return the cheapest terminal.

    The single-path schedulers (``first``/``greedy``) take the locally-best move and
    can be trapped: e.g. RollThroughFourier is the largest *immediate* saving on a
    roll wedged between two circulants, but the ``gather`` it emits severs the
    ``fft … ifft`` chain so the surrounding circulants can no longer fuse — the
    globally cheaper plan declines that move (T8.123). Beam search keeps the
    ``beam_width`` cheapest partial plans each round and expands all of them, so the
    decline-now/win-later branch survives. The system is terminating, so every branch
    reaches a fixpoint; we return the cheapest one reached.

    Per-branch ``folded`` is copied on expansion (a rule only *adds* keys, never
    mutates an existing coefficient array, so a shallow ``dict`` copy is sound).
    """
    Node = tuple[float, ExecutionPlan, dict[str, np.ndarray], list[RewriteMatch]]
    start: Node = (_plan_cost(plan, ctx.cost_model), plan, {}, [])
    frontier: list[Node] = [start]
    best: Node = start
    rounds = max(8, len(plan.steps) * 8)
    while rounds > 0 and frontier:
        rounds -= 1
        succ: list[Node] = []
        for cost, p, folded, matches in frontier:
            work_ctx = replace(ctx, constants={**ctx.constants, **folded}) if folded else ctx
            fired = False
            for rule in active_rules:
                m = rule.match(p, work_ctx)
                if m is None or not _rule_fires(m, ctx):
                    continue
                fired = True
                f2 = dict(folded)
                p2 = _remap_boundary_metadata(p, rule.apply(p, work_ctx, f2))
                succ.append((_plan_cost(p2, ctx.cost_model), p2, f2, [*matches, m]))
            # A node with no applicable rule is a terminal (fixpoint) candidate.
            if not fired and cost < best[0]:
                best = (cost, p, folded, matches)
        if not succ:
            break
        # Keep the cheapest distinct-shape successors; op-signature dedup keeps the
        # beam diverse and the sort key (cost, signature) makes the prune deterministic.
        succ.sort(key=lambda t: (t[0], tuple(s.op for s in t[1].steps)))
        seen: set[tuple[str, ...]] = set()
        pruned: list[Node] = []
        for node in succ:
            sig = tuple(s.op for s in node[1].steps)
            if sig in seen:
                continue
            seen.add(sig)
            pruned.append(node)
            if len(pruned) >= beam_width:
                break
        frontier = pruned
        if frontier[0][0] < best[0]:
            best = frontier[0]
    _, p, folded, matches = best
    return p, matches, folded


def rewrite_plan(
    plan: ExecutionPlan,
    ctx: RewriteContext | None = None,
    *,
    rules: Sequence[RewriteRule] | None = None,
    select: str = "first",
    beam_width: int = 8,
) -> tuple[ExecutionPlan, list[RewriteMatch], dict[str, np.ndarray]]:
    """Rewrite ``plan`` to a fixpoint.

    Returns ``(plan, matches, folded)`` where ``folded`` holds new plan-time
    constants (fused coefficients, closed-form results) keyed by env-var name.
    Run the result with :func:`run_rewritten` — it merges ``folded`` into the
    execute env *and* resolves the output aliases a full cancel leaves behind (a
    terminal ``ifft(fft(x))`` erases ``y``, which a bare ``plan.execute`` would
    ``KeyError`` on). Equivalent by hand: ``apply_aliases(plan.execute({**inputs,
    **folded}), plan)``.

    ``rules`` overrides the registry order (defaults to :data:`RULES`) — the
    confluence property test drives it with shuffled permutations to assert the
    fixpoint is order-independent.

    ``select`` picks the scheduler (all three are semantics-preserving; they differ
    only in how cheap a fixpoint they reach):

    * ``"first"`` (default) — fire the first applicable rule in ``rules`` order. The
      shipped order is a hand-tuned cost preference (RollThroughFourier before
      FftIfftCancel, T8.78); T8.122 measured a shuffle can reach a strictly cheaper
      fixpoint, so it is a heuristic, not cost-minimal.
    * ``"greedy"`` — fire the largest-saving applicable rule each step. Measured
      (T8.123) to be **no better than ``first``** on the random battery: the biggest
      immediate saving (RollThroughFourier) is exactly the move that forecloses the
      bigger downstream circulant fusion, so greedy falls into the same trap.
    * ``"beam"`` — :func:`_rewrite_beam` with ``beam_width``. Lookahead escapes that
      trap; measured to match the best of 20 random shuffles on every plan and to
      strictly beat ``first`` on exactly the plans where a shuffle did (Speculation
      #8 resolved). Returns the cheaper of the beam result and the ``first`` baseline,
      so it is never worse than the default.
    """
    ctx = ctx or RewriteContext()
    active_rules = RULES if rules is None else tuple(rules)
    if select not in ("first", "greedy", "beam"):
        raise ValueError(f"unknown select strategy {select!r}")
    if select == "beam":
        base = _rewrite_first(plan, ctx, active_rules)
        beam = _rewrite_beam(plan, ctx, active_rules, beam_width=beam_width)
        # Guarantee the optimiser is never worse than the shipped heuristic.
        return beam if _plan_cost(beam[0], ctx.cost_model) < _plan_cost(base[0], ctx.cost_model) else base
    return _rewrite_first(plan, ctx, active_rules, greedy=(select == "greedy"))


# --------------------------------------------------------------------------- #
# Rule 1 — fft/ifft cancellation
# --------------------------------------------------------------------------- #


class FftIfftCancel:
    """``ifft(fft(x)) -> x`` and ``fft(ifft(x)) -> x``.

    Requires ``rounding_contract='algebraic_output'``. This is an exact operator
    identity, not parity with the rounded intermediate words of the old program.
    """

    name = "fft_ifft_cancel"
    _inverse: ClassVar[dict[str, str]] = {"fft": "ifft", "ifft": "fft"}

    def _find(self, plan: ExecutionPlan) -> tuple[int, int] | None:
        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op not in self._inverse:
                continue
            src = outer.inputs[0]
            i = prod.get(src)
            if i is None:
                continue
            inner = plan.steps[i]
            if inner.op != self._inverse[outer.op]:
                continue
            # The intermediate must feed only this step, or we can't drop it.
            if _use_count(plan, src) != 1:
                continue
            return i, j
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        i, j = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = _step_cost(plan.steps[i], ctx.cost_model) + _step_cost(plan.steps[j], ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"{plan.steps[i].op}->{plan.steps[j].op} round trip is identity; "
            "drop both (also removes round-trip rounding)",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(i, j),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        i, j = self._find(plan)  # type: ignore[misc]
        inner, outer = plan.steps[i], plan.steps[j]
        # outer.output now aliases inner.input.
        rename = {outer.outputs[0]: inner.inputs[0]}
        kept = tuple(s for k, s in enumerate(plan.steps) if k not in (i, j))
        # Record a terminal alias so a fully-cancelled output is still queryable
        # via apply_aliases() (no downstream step carries the rename otherwise).
        meta = dict(plan.factorization_metadata)
        aliases = dict(meta.get("aliases", {}))
        aliases[outer.outputs[0]] = inner.inputs[0]
        meta["aliases"] = aliases
        return ExecutionPlan(steps=_remap_inputs(kept, rename), factorization_metadata=meta)


# --------------------------------------------------------------------------- #
# Rule 1b — double-transform reversal (DFT period 4)
# --------------------------------------------------------------------------- #


class DoubleFftReversal:
    """``fft(fft(x)) -> N·reverse(x)`` and ``ifft(ifft(x)) -> (1/N)·reverse(x)``.

    Applying the *same* DFT twice is a reversal: ``fft(fft(x))[n] = N·x[(-n) mod N]``
    — so the DFT has period four (``DFT⁴ = I``). Two transform dispatches collapse
    to a host ``gather`` (the reversal permutation, ``idx = (-n) mod N``) plus one
    ``diagonal_scale`` by the constant ``N`` (resp. ``1/N``). Exact for the
    power-of-two ``N`` an FFT always uses — the scale is an integer L-shift and the
    reversal a pure reindex, no rounding. Distinct from :class:`FftIfftCancel`,
    which only cancels *opposite* transforms (``ifft∘fft``).

    Honesty gate: both transforms must carry ``params["n"]`` and agree on it — the
    reversal index set and the scale magnitude are only known when ``N`` is.
    """

    name = "double_fft_reversal"
    _scale: ClassVar[dict[str, float]] = {"fft": 1.0, "ifft": -1.0}

    def _find(self, plan: ExecutionPlan) -> tuple[int, int] | None:
        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op not in ("fft", "ifft"):
                continue
            src = outer.inputs[0]
            i = prod.get(src)
            if i is None:
                continue
            inner = plan.steps[i]
            if inner.op != outer.op:  # same direction only
                continue
            if _use_count(plan, src) != 1:
                continue
            n_in, n_out = inner.params.get("n"), outer.params.get("n")
            if n_in is None or n_out is None or int(n_in) != int(n_out):
                continue
            return i, j
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        i, j = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = (
            _step_cost(plan.steps[i], ctx.cost_model)
            + _step_cost(plan.steps[j], ctx.cost_model)
            - ctx.cost_model.cost("gather")
            - ctx.cost_model.cost("diagonal_scale")
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"{plan.steps[j].op}∘{plan.steps[i].op} is N·reverse (DFT period 4); "
            "replace two transforms with a reversal gather + constant scale",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(i, j),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        i, j = self._find(plan)  # type: ignore[misc]
        inner, outer = plan.steps[i], plan.steps[j]
        n = int(outer.params["n"])
        x = inner.inputs[0]
        out = outer.outputs[0]
        idx = ((-np.arange(n, dtype=np.int64)) % n).astype(np.int64)  # x[(-n) mod N]
        scale_val = float(n) if outer.op == "fft" else 1.0 / float(n)
        scale_name = f"{out}__fft2_scale"
        folded[scale_name] = encode_words(np.full(n, scale_val, dtype=np.complex128))
        rev_name = f"{out}__rev"
        gather = ExecutionStep("gather", (x,), (rev_name,), params={"indices": idx, "axis": -1})
        scale = ExecutionStep("diagonal_scale", (rev_name, scale_name), (out,))
        new_steps: list[ExecutionStep] = []
        for k, step in enumerate(plan.steps):
            if k == i:
                continue
            if k == j:
                new_steps.extend((gather, scale))
            else:
                new_steps.append(step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 2 — diagonal-scale composition
# --------------------------------------------------------------------------- #


class DiagonalScaleCompose:
    """``diag(diag(x, a), b) -> diag(x, a*b)`` for plan-time-constant ``a``, ``b``.

    SPF multiply is integer L-add + modular Θ-add, so the fused coefficient is
    bit-exact except where ``L`` saturates — and :meth:`_find` *refuses* a pair
    whose product would clamp the L plane (:func:`_mul_l_saturates`), so the rule
    fires only when the fold is genuinely bit-exact (``exact=True`` never lies).
    Folding two data-length multiplies into one is the real win; the coefficient
    product is a one-off precompute returned via ``folded``.
    """

    name = "diagonal_scale_compose"

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> tuple[int, int] | None:
        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op != "diagonal_scale":
                continue
            data_name, b_name = outer.inputs[0], outer.inputs[1]
            i = prod.get(data_name)
            if i is None:
                continue
            inner = plan.steps[i]
            if inner.op != "diagonal_scale":
                continue
            a_name = inner.inputs[1]
            if a_name not in ctx.constants or b_name not in ctx.constants:
                continue
            if _use_count(plan, data_name) != 1:
                continue
            # Refuse the fold when the integer L-add would saturate — then the
            # fused coeff is not the bit-exact product and `exact` would lie. The
            # two scales stay (still correct); only the optimisation is skipped.
            if _mul_l_saturates(
                np.asarray(ctx.constants[a_name]), np.asarray(ctx.constants[b_name])
            ):
                continue
            a = np.asarray(ctx.constants[a_name], dtype=np.uint32)
            b = np.asarray(ctx.constants[b_name], dtype=np.uint32)
            if np.any(unpack_L(a) == SPF32_L_NAN) or np.any(unpack_L(b) == SPF32_L_NAN):
                continue
            aL, bL = np.broadcast_arrays(unpack_L(a).astype(np.int64), unpack_L(b).astype(np.int64))
            # Same-sign rates have associative saturation. Opposing rates can
            # undo a clamp and require an input-range premise in packed mode.
            monotone = bool(np.all(((aL >= 0) & (bL >= 0)) | ((aL <= 0) & (bL <= 0))))
            x_name = inner.inputs[0]
            if ctx.rounding_contract == "packed_steps" and not np.all((aL == 0) & (bL == 0)):
                known = ctx.constants.get(x_name)
                if known is None and x_name not in ctx.input_l_ranges:
                    continue
                if known is not None and np.any(unpack_L(known) == SPF32_L_NAN):
                    continue
            if ctx.rounding_contract == "packed_steps" and not monotone:
                if x_name in ctx.constants:
                    if _mul_l_saturates(np.asarray(ctx.constants[x_name]), a):
                        continue
                elif x_name in ctx.input_l_ranges:
                    lo, hi = ctx.input_l_ranges[x_name]
                    if np.any(lo + aL < SPF32_L_MIN_FINITE) or np.any(hi + aL > SPF32_L_MAX_FINITE):
                        continue
                else:
                    continue
            return i, j
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan, ctx)
        if found is None:
            return None
        i, j = found
        a_name, b_name = plan.steps[i].inputs[1], plan.steps[j].inputs[1]
        old = _plan_cost(plan, ctx.cost_model)
        saved = _step_cost(plan.steps[i], ctx.cost_model)  # two diagonal_scale passes -> one
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"fuse constant coeffs {a_name!r}*{b_name!r}; integer add is "
            f"composed under {ctx.rounding_contract}",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(i, j),
            algebraic=ctx.rounding_contract == "algebraic_output",
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        i, j = self._find(plan, ctx)  # type: ignore[misc]
        inner, outer = plan.steps[i], plan.steps[j]
        a_name, b_name = inner.inputs[1], outer.inputs[1]
        fused_name = f"{a_name}__x__{b_name}"
        folded[fused_name] = spf_mul_words(
            np.asarray(ctx.constants[a_name]), np.asarray(ctx.constants[b_name])
        )
        x_name = inner.inputs[0]
        new_steps: list[ExecutionStep] = []
        for k, step in enumerate(plan.steps):
            if k == i:
                continue  # drop the inner scale
            if k == j:
                new_steps.append(replace(step, inputs=(x_name, fused_name)))
            else:
                new_steps.append(step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 3 — circulant / FDWM recognition (annotation only)
# --------------------------------------------------------------------------- #


class CirculantTag:
    """Recognise ``ifft(diag(fft(x), h))`` and tag it circulant (FDWM apply).

    Tag-only: the *fusion* of ``C(h1)∘C(h2) -> C(h1·h2)`` is emergent — the inner
    ``ifft(...fft...)`` is cancelled by :class:`FftIfftCancel` and the two scales
    merged by :class:`DiagonalScaleCompose` at the fixpoint. This rule just
    records the structure for auditability; it never edits steps.
    """

    name = "circulant_tag"

    def _circulants(self, plan: ExecutionPlan) -> list[dict[str, str]]:
        prod = _producer(plan)
        found: list[dict[str, str]] = []
        for outer in plan.steps:
            if outer.op != "ifft":
                continue
            mid = outer.inputs[0]
            si = prod.get(mid)
            if si is None or plan.steps[si].op != "diagonal_scale":
                continue
            scale = plan.steps[si]
            fi = prod.get(scale.inputs[0])
            if fi is None or plan.steps[fi].op != "fft":
                continue
            found.append(
                {
                    "data": plan.steps[fi].inputs[0],
                    "coeff": scale.inputs[1],
                    "out": outer.outputs[0],
                }
            )
        return found

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        circ = self._circulants(plan)
        if not circ:
            return None
        if plan.factorization_metadata.get("circulant") == circ:
            return None  # already tagged — idempotent, stops the fixpoint
        cost = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"recognised {len(circ)} circulant/FDWM apply(s) ifft(diag(fft(x),h))",
            old_cost=cost,
            new_cost=cost,
            error_bound=0.0,
            step_span=(),
            annotation=True,
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        meta = dict(plan.factorization_metadata)
        meta["circulant"] = self._circulants(plan)
        return ExecutionPlan(steps=plan.steps, factorization_metadata=meta)


# --------------------------------------------------------------------------- #
# Rule 4 — phase-ramp / DFT-row orthogonality (provenance-gated, plan-time only)
# --------------------------------------------------------------------------- #


class PhaseRampOrthogonality:
    """``contract(conj(row_f), row_g) -> N·δ_{fg}`` — only for exact DFT rows.

    Hard provenance gate: both operands must be exact, residue-free, unwindowed,
    noiseless DFT rows of equal ``N`` (:meth:`Provenance.is_exact_dft_row`). Any
    missing or false flag -> no rewrite. The contraction step is deleted and its
    closed-form result returned via ``folded`` under the step's output name.
    """

    name = "phase_ramp_orthogonality"

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> int | None:
        for k, step in enumerate(plan.steps):
            if step.op != "decoded_contract_reference" or len(step.inputs) != 2:
                continue
            pa = ctx.provenance.get(step.inputs[0])
            pb = ctx.provenance.get(step.inputs[1])
            if pa is None or pb is None:
                continue
            if not (pa.is_exact_dft_row() and pb.is_exact_dft_row()):
                continue
            if pa.N != pb.N:
                continue
            return k
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        k = self._find(plan, ctx)
        if k is None:
            return None
        old = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="both operands are exact constant-modulus SEPARABLE_PHASE DFT rows; "
            "inner product is closed-form N·delta",
            old_cost=old,
            new_cost=old - _step_cost(plan.steps[k], ctx.cost_model),
            error_bound=0.0,
            step_span=(k,),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        k = self._find(plan, ctx)  # type: ignore[arg-type]
        step = plan.steps[k]
        pa = ctx.provenance[step.inputs[0]]
        pb = ctx.provenance[step.inputs[1]]
        n = int(pa.N)  # type: ignore[arg-type]
        frequency = (-1 if pa.conjugated else 1) * int(pa.index) + (-1 if pb.conjugated else 1) * int(pb.index)
        value = float(n) if frequency % n == 0 else 0.0
        folded[step.outputs[0]] = encode_words(np.asarray(value, dtype=np.complex128))
        kept = tuple(s for j, s in enumerate(plan.steps) if j != k)
        return ExecutionPlan(steps=kept, factorization_metadata=plan.factorization_metadata)


# --------------------------------------------------------------------------- #
# Rule 4b — character phase-bind collapse
# --------------------------------------------------------------------------- #


class CharacterPhaseBind:
    """``gather(χ_{αg}, a) * gather(χ_{βg}, b) -> gather(χ_g, (αa+βb) mod N)``.

    The affine-character identity ``χ_{αg}(a)·χ_{βg}(b) = χ_g((αa+βb) mod N)`` for
    an exact DFT-row character ``χ_g`` and per-axis input coefficients ``α``, ``β``.
    Two sibling ``gather`` steps that index characters of the *same* base generator
    ``g`` (each scaled by its axis coefficient) and feed a single
    ``elementwise_mul`` collapse to one ``gather`` over the affine-combined indices
    — removing the Cartesian-fallback multiply.

    The pure modular-addition case ``α = β = 1`` (equal row index ``f = g``) is the
    sum-character special case ``χ_f(a)·χ_f(b) = χ_f((a+b) mod N)``; difference /
    affine circuits (``α ≠ β``: ``a-b``, ``a+2b``, ``7a-7b``) have distinct per-axis
    rows ``αg ≠ βg`` and fuse to ``indices_op="affine_mod"`` carrying ``coeffs``.

    Hard provenance gate (mirrors :class:`PhaseRampOrthogonality`): both gather
    outputs must be :meth:`Provenance.is_exact_dft_row` of equal ``N`` and share a
    base generator — ``index_a·β ≡ index_b·α (mod N)`` (for α=β=1 this is exactly
    ``index_a == index_b``). The axis coefficients are vouched out-of-band via
    :attr:`Provenance.axis_coeff` (``None`` ⇒ 1), never inferred from words. Any
    missing/false flag, or no shared generator, -> no rewrite.

    This is the SPF-side home of PAN's modular-addition *and* affine decompile: its
    two per-input encoders emit exactly this ``gather, gather, elementwise_mul``
    triple over a shared character channel, and the bind is what turns
    ``e^{iαga}·e^{iβgb}`` into ``e^{ig(αa+βb)}`` on one table.

    The two gathers may name *distinct* table operands (e.g. ``encoders.0`` vs
    ``encoders.1``); the provenance contract proves they carry the same character,
    so the fused gather keeps the first table and drops the second exactly. The
    fused gather carries ``indices_op="sum_mod"``/``"affine_mod"`` and is host-
    executed by the consumer (PAN's interpreter / the character loader), not the
    core FP64 oracle — so this rule is structurally validated, never added to the
    executable confluence battery. It fires only on its own triple+provenance, so
    it is inert on every existing plan in that battery.
    """

    name = "character_phase_bind"

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> tuple[int, int, int] | None:
        prod = _producer(plan)
        for k, mul in enumerate(plan.steps):
            if mul.op != "elementwise_mul" or len(mul.inputs) != 2:
                continue
            la, lb = mul.inputs
            ia, ib = prod.get(la), prod.get(lb)
            if ia is None or ib is None or ia == ib:
                continue
            ga, gb = plan.steps[ia], plan.steps[ib]
            if ga.op != "gather" or gb.op != "gather":
                continue
            if any(len(step.inputs) != 2 or step.params.get("indices_op") is not None for step in (ga, gb)):
                continue
            # Each gather output must be consumed only by this multiply.
            if _use_count(plan, la) != 1 or _use_count(plan, lb) != 1:
                continue
            pa = ctx.provenance.get(la)
            pb = ctx.provenance.get(lb)
            if pa is None or pb is None:
                continue
            if pa.conjugated != pb.conjugated:
                continue
            if not (pa.is_exact_dft_row() and pb.is_exact_dft_row()):
                continue
            if pa.N != pb.N:
                continue
            # Shared base generator g: index_a = α·g, index_b = β·g, so
            # index_a·β ≡ index_b·α (mod N). For α=β=1 this is index_a == index_b.
            alpha = 1 if pa.axis_coeff is None else int(pa.axis_coeff)
            beta = 1 if pb.axis_coeff is None else int(pb.axis_coeff)
            if (int(pa.index) * beta - int(pb.index) * alpha) % int(pa.N) != 0:
                continue
            return ia, ib, k
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan, ctx)
        if found is None:
            return None
        ia, ib, k = found
        old = _plan_cost(plan, ctx.cost_model)
        # Drop the second gather + the multiply, keep one gather (the fused
        # sum-mod gather costs the same as a plain gather).
        saved = _step_cost(plan.steps[ib], ctx.cost_model) + _step_cost(
            plan.steps[k], ctx.cost_model
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="both gathers index the same exact DFT-row character; product is "
            "the sum-character gather χ_f((a+b) mod N)",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(ia, ib, k),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        ia, ib, k = self._find(plan, ctx)  # type: ignore[misc]
        ga, gb, mul = plan.steps[ia], plan.steps[ib], plan.steps[k]
        pa, pb = ctx.provenance[mul.inputs[0]], ctx.provenance[mul.inputs[1]]
        modulus = int(pa.N)  # type: ignore[arg-type]
        alpha = 1 if pa.axis_coeff is None else int(pa.axis_coeff)
        beta = 1 if pb.axis_coeff is None else int(pb.axis_coeff)
        params: dict[str, Any] = {
            "axis": int(ga.params.get("axis", 0)),
            "modulus": modulus,
            "fused_from": "gather+gather+elementwise_mul",
        }
        # Pure modular addition keeps the simpler sum-mod index op (back-compatible
        # with the existing executor + board path); affine/difference binds carry
        # the per-axis coefficients so the fused index is (αa+βb) mod N.
        if alpha == 1 and beta == 1:
            params["indices_op"] = "sum_mod"
        else:
            params["indices_op"] = "affine_mod"
            params["coeffs"] = [alpha, beta]
        fused = ExecutionStep(
            "gather",
            inputs=(ga.inputs[0], ga.inputs[1], gb.inputs[1]),
            outputs=mul.outputs,
            params=params,
        )
        new_steps: list[ExecutionStep] = []
        for idx, step in enumerate(plan.steps):
            if idx in (ia, ib):
                continue
            new_steps.append(fused if idx == k else step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


class CharacterBankBind:
    """Batched :class:`CharacterPhaseBind` over a *bank* of k characters at once.

    A sparse 2-D character surface ``Σ_j c_j·χ_{(u_j,v_j)}(a,b)`` (the shape a
    transformer's per-class logit surface FFTs to) is k separable binds stacked on
    one axis. Each atom is a separable product ``χ_{u_j}(a)·χ_{v_j}(b)`` (BCCB /
    relationship #8: the 2-D character factors into two 1-D characters), so the
    whole bank is one ``elementwise_mul`` of two stacked gathers — and it collapses
    by the same algebra as the scalar bind, applied per atom:
    ``χ_{u_j}(a)·χ_{v_j}(b) = χ_{g_j}((u_j a + v_j b) mod N)``.

    Provenance gate: both gather outputs :meth:`Provenance.is_exact_dft_row_bank`
    of equal ``N`` and equal bank length, and every atom shares a base generator —
    ``index_a[j]·β[j] ≡ index_b[j]·α[j] (mod N)`` for all j (the per-atom form of
    the scalar cross-check). The fused gather carries ``indices_op="affine_bank_mod"``
    + per-axis coefficient vectors ``coeffs_a``/``coeffs_b``; it is host-executed by
    the surface consumer, so — like :class:`CharacterPhaseBind` — it is structurally
    validated and provably inert on the executable confluence battery (no plan there
    carries bank provenance).
    """

    name = "character_bank_bind"

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> tuple[int, int, int] | None:
        prod = _producer(plan)
        for k, mul in enumerate(plan.steps):
            if mul.op != "elementwise_mul" or len(mul.inputs) != 2:
                continue
            la, lb = mul.inputs
            ia, ib = prod.get(la), prod.get(lb)
            if ia is None or ib is None or ia == ib:
                continue
            ga, gb = plan.steps[ia], plan.steps[ib]
            if ga.op != "gather" or gb.op != "gather":
                continue
            if any(len(step.inputs) != 2 or step.params.get("indices_op") is not None for step in (ga, gb)):
                continue
            if _use_count(plan, la) != 1 or _use_count(plan, lb) != 1:
                continue
            pa, pb = ctx.provenance.get(la), ctx.provenance.get(lb)
            if pa is None or pb is None:
                continue
            if pa.conjugated != pb.conjugated:
                continue
            if not (pa.is_exact_dft_row_bank() and pb.is_exact_dft_row_bank()):
                continue
            if pa.N != pb.N:
                continue
            ua, ub = pa.index_bank, pb.index_bank
            if ua is None or ub is None or len(ua) != len(ub):
                continue
            alpha = pa.axis_coeff_bank or ua
            beta = pb.axis_coeff_bank or ub
            if len(alpha) != len(ua) or len(beta) != len(ub):
                continue
            n = int(pa.N)
            # Per-atom shared base generator: index_a[j]·β[j] ≡ index_b[j]·α[j].
            if any((ua[j] * beta[j] - ub[j] * alpha[j]) % n != 0 for j in range(len(ua))):
                continue
            return ia, ib, k
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan, ctx)
        if found is None:
            return None
        ia, ib, k = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = _step_cost(plan.steps[ib], ctx.cost_model) + _step_cost(
            plan.steps[k], ctx.cost_model
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="both gathers index the same exact DFT-row character bank; the "
            "product is the per-atom affine-character bank χ_g((αa+βb) mod N)",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(ia, ib, k),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        ia, ib, k = self._find(plan, ctx)  # type: ignore[misc]
        ga, gb, mul = plan.steps[ia], plan.steps[ib], plan.steps[k]
        pa, pb = ctx.provenance[mul.inputs[0]], ctx.provenance[mul.inputs[1]]
        modulus = int(pa.N)  # type: ignore[arg-type]
        alpha = list(pa.axis_coeff_bank or pa.index_bank)  # type: ignore[arg-type]
        beta = list(pb.axis_coeff_bank or pb.index_bank)  # type: ignore[arg-type]
        fused = ExecutionStep(
            "gather",
            inputs=(ga.inputs[0], ga.inputs[1], gb.inputs[1]),
            outputs=mul.outputs,
            params={
                "axis": int(ga.params.get("axis", 0)),
                "indices_op": "affine_bank_mod",
                "coeffs_a": alpha,
                "coeffs_b": beta,
                "modulus": modulus,
                "fused_from": "gather+gather+elementwise_mul",
            },
        )
        new_steps: list[ExecutionStep] = []
        for idx, step in enumerate(plan.steps):
            if idx in (ia, ib):
                continue
            new_steps.append(fused if idx == k else step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


class ProductGroupCharacterBind:
    """``gather(χ_F, a) * gather(χ_F, b) -> gather(χ_F, (a ⊕ b) mod (n₁,…,n_d))``.

    The product-group generalisation of :class:`CharacterPhaseBind` (algebra-structure
    #7): a character of ``∏ Z/nᵢ`` factors as ``χ_F(A) = ∏ᵢ χ_{Fᵢ}(Aᵢ)``, so the bind
    of two equal-frequency characters adds the group elements **componentwise** ::

        χ_F(a)·χ_F(b) = χ_F((a₁+b₁) mod n₁, …, (a_d+b_d) mod n_d)

    with ``a``/``b``/``F`` carried as flat mixed-radix integers (positional strides over
    the factors). For *coprime* factors this is just a single ``Z/N`` character via CRT
    (already :class:`CharacterPhaseBind`); the value here is the **non-coprime /
    mixed-radix** case — Walsh-Hadamard ``(Z/2)^k``, repeated radices — where the
    componentwise sum has no single-modulus form. The flat ``sum_mod`` would carry
    across factors and be wrong; ``product_sum_mod`` adds per factor with no carry.

    Provenance gate (mirrors :class:`CharacterPhaseBind`): both gather outputs
    :meth:`Provenance.is_exact_product_dft_row` of equal ``moduli`` and equal flat
    frequency ``index`` (the sum-character case). The fused gather carries
    ``indices_op="product_sum_mod"`` + ``moduli``; host-executed by the consumer, so —
    like the other character binds — it is structurally validated and inert on the
    executable confluence battery (no plan there carries product-group provenance).
    """

    name = "product_group_character_bind"

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> tuple[int, int, int] | None:
        prod = _producer(plan)
        for k, mul in enumerate(plan.steps):
            if mul.op != "elementwise_mul" or len(mul.inputs) != 2:
                continue
            la, lb = mul.inputs
            ia, ib = prod.get(la), prod.get(lb)
            if ia is None or ib is None or ia == ib:
                continue
            ga, gb = plan.steps[ia], plan.steps[ib]
            if ga.op != "gather" or gb.op != "gather":
                continue
            if any(len(step.inputs) != 2 or step.params.get("indices_op") is not None for step in (ga, gb)):
                continue
            if _use_count(plan, la) != 1 or _use_count(plan, lb) != 1:
                continue
            pa, pb = ctx.provenance.get(la), ctx.provenance.get(lb)
            if pa is None or pb is None:
                continue
            if pa.conjugated != pb.conjugated:
                continue
            if not (pa.is_exact_product_dft_row() and pb.is_exact_product_dft_row()):
                continue
            if pa.moduli != pb.moduli or int(pa.index) != int(pb.index):
                continue
            return ia, ib, k
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan, ctx)
        if found is None:
            return None
        ia, ib, k = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = _step_cost(plan.steps[ib], ctx.cost_model) + _step_cost(
            plan.steps[k], ctx.cost_model
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="both gathers index the same exact product-group character; product "
            "is the componentwise sum-character χ_F((a ⊕ b) mod (n₁,…,n_d))",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(ia, ib, k),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        ia, ib, k = self._find(plan, ctx)  # type: ignore[misc]
        ga, gb, mul = plan.steps[ia], plan.steps[ib], plan.steps[k]
        pa = ctx.provenance[mul.inputs[0]]
        fused = ExecutionStep(
            "gather",
            inputs=(ga.inputs[0], ga.inputs[1], gb.inputs[1]),
            outputs=mul.outputs,
            params={
                "axis": int(ga.params.get("axis", 0)),
                "indices_op": "product_sum_mod",
                "moduli": list(pa.moduli),  # type: ignore[arg-type]
                "fused_from": "gather+gather+elementwise_mul",
            },
        )
        new_steps: list[ExecutionStep] = []
        for idx, step in enumerate(plan.steps):
            if idx in (ia, ib):
                continue
            new_steps.append(fused if idx == k else step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 5 — gather / permute composition
# --------------------------------------------------------------------------- #


class GatherPermuteCompose:
    """``gather(gather(x, i1), i2) -> gather(x, i1[i2])`` and ``permute∘permute``.

    Exact host-side index algebra; removes one indexing pass.
    """

    name = "gather_permute_compose"

    def _find(self, plan: ExecutionPlan) -> tuple[int, int] | None:
        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op not in ("gather", "permute"):
                continue
            src = outer.inputs[0]
            i = prod.get(src)
            if i is None or plan.steps[i].op != outer.op:
                continue
            if _use_count(plan, src) != 1:
                continue
            # gather is composable only along a shared axis.
            if outer.op == "gather" and int(plan.steps[i].params.get("axis", 0)) != int(
                outer.params.get("axis", 0)
            ):
                continue
            return i, j
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        i, j = found
        old = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"compose two {plan.steps[j].op} passes into one",
            old_cost=old,
            new_cost=old - _step_cost(plan.steps[i], ctx.cost_model),
            error_bound=0.0,
            step_span=(i, j),
        )

    def _composed_params(
        self, inner: ExecutionStep, outer: ExecutionStep
    ) -> dict[str, Any]:
        if outer.op == "gather":
            i1 = np.asarray(inner.params["indices"], dtype=np.int64)
            i2 = np.asarray(outer.params["indices"], dtype=np.int64)
            params = dict(outer.params)
            params["indices"] = i1[i2]
            return params
        # permute: new_axes[k] = inner_axes[outer_axes[k]]
        a1 = tuple(inner.params["axes"])
        a2 = tuple(outer.params["axes"])
        params = dict(outer.params)
        params["axes"] = tuple(a1[k] for k in a2)
        return params

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        i, j = self._find(plan)  # type: ignore[misc]
        inner, outer = plan.steps[i], plan.steps[j]
        params = self._composed_params(inner, outer)
        new_steps: list[ExecutionStep] = []
        for k, step in enumerate(plan.steps):
            if k == i:
                continue
            if k == j:
                new_steps.append(replace(step, inputs=(inner.inputs[0],), params=params))
            else:
                new_steps.append(step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 6 — ramp / ramp fusion
# --------------------------------------------------------------------------- #


class RampFusion:
    """``ramp(ramp(x, R1, b1), R2, b2) -> ramp(x, R1+R2, b1+b2)``.

    Phase ramps add: the per-element increment is linear in the ramp coefficients
    and reduced mod ``2^16``, so summing coefficients is bit-exact (sentinels pass
    through identically either way). Quadratic/cubic terms fold too, gated by each
    ramp's ``nonlinear`` flag. Drops a whole data pass.
    """

    name = "ramp_fusion"

    def _find(self, plan: ExecutionPlan) -> tuple[int, int] | None:
        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op != "ramp":
                continue
            src = outer.inputs[0]
            i = prod.get(src)
            if i is None or plan.steps[i].op != "ramp":
                continue
            if _use_count(plan, src) != 1:
                continue
            return i, j
        return None

    def _fuse(self, inner: ExecutionStep, outer: ExecutionStep) -> dict[str, Any]:
        r1, r2 = tuple(inner.params["ramp_R"]), tuple(outer.params["ramp_R"])
        b1, b2 = tuple(inner.params["ramp_b"]), tuple(outer.params["ramp_b"])
        order = len(r1)
        rf = [[int(r1[r][c]) + int(r2[r][c]) for c in range(order)] for r in range(order)]
        bf = [int(b1[r]) + int(b2[r]) for r in range(order)]
        n1 = bool(inner.params.get("nonlinear", False))
        n2 = bool(outer.params.get("nonlinear", False))

        def eff(step: ExecutionStep, key: str, flag: bool) -> tuple[int, ...]:
            return tuple(int(v) for v in (step.params.get(key) or ())) if flag else ()

        def add(a: tuple[int, ...], b: tuple[int, ...]) -> list[int] | None:
            m = max(len(a), len(b))
            if m == 0:
                return None
            return [(a[k] if k < len(a) else 0) + (b[k] if k < len(b) else 0) for k in range(m)]

        params: dict[str, Any] = {"ramp_R": rf, "ramp_b": bf}
        gf = add(eff(inner, "phase_gradient", True), eff(outer, "phase_gradient", True))
        if gf is not None:
            params["phase_gradient"] = gf
        if n1 or n2:
            params["nonlinear"] = True
            qf = add(eff(inner, "quadratic_ramp", n1), eff(outer, "quadratic_ramp", n2))
            cf = add(eff(inner, "cubic_ramp", n1), eff(outer, "cubic_ramp", n2))
            mf = add(eff(inner, "mixed_ramp", n1), eff(outer, "mixed_ramp", n2))
            cmf = add(eff(inner, "cubic_mixed", n1), eff(outer, "cubic_mixed", n2))
            if qf is not None:
                params["quadratic_ramp"] = qf
            if cf is not None:
                params["cubic_ramp"] = cf
            if mf is not None:
                params["mixed_ramp"] = mf
            if cmf is not None:
                params["cubic_mixed"] = cmf
        return params

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        i, j = found
        old = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="phase ramps add; fuse R1+R2, b1+b2 (bit-exact mod 2^16)",
            old_cost=old,
            new_cost=old - _step_cost(plan.steps[i], ctx.cost_model),
            error_bound=0.0,
            step_span=(i, j),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        i, j = self._find(plan)  # type: ignore[misc]
        inner, outer = plan.steps[i], plan.steps[j]
        params = self._fuse(inner, outer)
        new_steps: list[ExecutionStep] = []
        for k, step in enumerate(plan.steps):
            if k == i:
                continue
            if k == j:
                new_steps.append(replace(step, inputs=(inner.inputs[0],), params=params))
            else:
                new_steps.append(step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 7 — no-op / identity prune
# --------------------------------------------------------------------------- #


class PhaseShearCompose:
    """Fuse compatible polynomial shears; preserve validation even for identity.

    Controller/reference coordinates are unchanged by each shear. Adding the
    polynomials is therefore exact in the phase ring, including inverse pairs.
    ROM composition is deliberately left to a future measured implementation.
    """

    name = "phase_shear_compose"

    def _find(self, plan: ExecutionPlan) -> tuple[int, int] | None:
        from .phase_circuits import _integer, canonical_polynomial

        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op != "phase_shear" or len(outer.inputs) != 1:
                continue
            i = prod.get(outer.inputs[0])
            if i is None or i >= j or plan.steps[i].op != "phase_shear" or _use_count(plan, outer.inputs[0]) != 1:
                continue
            inner = plan.steps[i]
            if len(inner.inputs) != 1 or len(inner.outputs) != 1 or len(outer.outputs) != 1:
                continue
            allowed = {"target", "control", "reference", "coefficients", "inverse", "table", "n_channels", "n_terms"}
            if any(set(step.params) - allowed for step in (inner, outer)):
                continue
            if any(inner.inputs[0] in step.outputs for step in plan.steps[i + 1:j]):
                continue
            if any(s.params.get("table") is not None for s in (inner, outer)):
                continue
            if not all(inner.params.get(k) == outer.params.get(k) for k in (
                "target", "control", "reference", "n_channels",
            )):
                continue
            if inner.params.get("target") in {inner.params.get("control"), inner.params.get("reference")}:
                continue
            if not all(isinstance(s.params.get("inverse", False), (bool, np.bool_)) for s in (inner, outer)):
                continue
            try:
                for step in (inner, outer):
                    for key in ("target", "control", "reference", "n_channels"):
                        if step.params.get(key) is not None:
                            _integer(step.params[key])
                    canonical_polynomial(step.params.get("coefficients", (0, 0, 1)))
            except SPFError:
                continue
            return i, j
        return None

    def _params(self, inner: ExecutionStep, outer: ExecutionStep) -> dict[str, Any]:
        from .phase_circuits import canonical_polynomial

        a, b = (canonical_polynomial(s.params.get("coefficients", (0, 0, 1))) for s in (inner, outer))
        sa, sb = (-1 if s.params.get("inverse", False) else 1 for s in (inner, outer))
        coeff = tuple(
            sa * (a[k] if k < len(a) else 0) + sb * (b[k] if k < len(b) else 0)
            for k in range(max(len(a), len(b)))
        )
        params = dict(outer.params)
        params.update(coefficients=canonical_polynomial(coeff), inverse=False)
        return params

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        i, j = found
        replacement = replace(plan.steps[j], params=self._params(plan.steps[i], plan.steps[j]))
        old = _plan_cost(plan, ctx.cost_model)
        saved = _step_cost(plan.steps[i], ctx.cost_model) + _step_cost(plan.steps[j], ctx.cost_model)
        return RewriteMatch(self.name, 1.0, True, "sum native shear polynomials; controllers retained",
                            old, old - saved + _step_cost(replacement, ctx.cost_model), 0.0, (i, j))

    def apply(self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]) -> ExecutionPlan:
        i, j = self._find(plan)  # type: ignore[misc]
        inner, outer = plan.steps[i], plan.steps[j]
        replacement = replace(outer, inputs=inner.inputs, params=self._params(inner, outer))
        return ExecutionPlan(steps=tuple(replacement if k == j else s for k, s in enumerate(plan.steps) if k != i),
                             factorization_metadata=plan.factorization_metadata)


class NoOpPrune:
    """Delete provably-identity steps that other rules leave between fusable ops.

    Three sound cases (no plan-time shape needed):

    * zero ramp (``R == 0``, ``b == 0``, no active quad/cubic term);
    * identity ``permute`` (``axes == (0, 1, ..., n-1)`` — identity for any shape);
    * ``diagonal_scale`` / ``elementwise_mul`` by an all-``1`` constant coefficient.

    Identity ``gather`` is deliberately *not* pruned: ``np.take`` with in-order
    indices is identity only if they cover the full axis, which is not known
    without the array length at plan time — pruning it would be unsound.
    """

    name = "noop_prune"

    def _is_noop(self, step: ExecutionStep, ctx: RewriteContext) -> bool:
        if step.op == "ramp":
            r = tuple(step.params.get("ramp_R") or ())
            b = tuple(step.params.get("ramp_b") or ())
            g = tuple(step.params.get("phase_gradient") or ())
            zero_lin = (
                all(int(v) == 0 for row in r for v in row)
                and all(int(v) == 0 for v in b)
                and all(int(v) == 0 for v in g)
            )
            if not zero_lin:
                return False
            if bool(step.params.get("nonlinear", False)):
                q = tuple(step.params.get("quadratic_ramp") or ())
                c = tuple(step.params.get("cubic_ramp") or ())
                m = tuple(step.params.get("mixed_ramp") or ())
                cm = tuple(step.params.get("cubic_mixed") or ())
                return all(int(v) == 0 for v in q + c + m + cm)
            return True
        if step.op == "permute":
            axes = tuple(int(a) for a in step.params.get("axes", ()))
            return axes == tuple(range(len(axes)))
        if step.op in {"diagonal_scale", "elementwise_mul"}:
            coeff = step.inputs[1]
            if coeff not in ctx.constants:
                return False
            w = np.asarray(ctx.constants[coeff], dtype=np.uint32)
            if not (w.size and np.all(w == _ONE_WORD)):
                return False
            data = step.inputs[0]
            known = ctx.constants.get(data)
            shape = np.shape(known) if known is not None else ctx.input_shapes.get(data)
            # Removing a broadcast multiply must preserve the output shape.
            if w.ndim and (shape is None or np.broadcast_shapes(shape, w.shape) != shape):
                return False
            # The core multiply canonicalizes nonfinite sentinels even at one.
            if ctx.rounding_contract == "packed_steps":
                return bool(
                    (known is not None and not np.any(
                        (unpack_L(known) == SPF32_L_NAN)
                        | ((unpack_L(known) == SPF32_L_ZERO) & (known != SPF32_WORD_ZERO))
                    ))
                    or data in ctx.input_l_ranges
                )
            return True
        return False

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> int | None:
        for k, step in enumerate(plan.steps):
            if self._is_noop(step, ctx):
                return k
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        k = self._find(plan, ctx)
        if k is None:
            return None
        old = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"{plan.steps[k].op} is an identity (no-op); drop it",
            old_cost=old,
            new_cost=old - _step_cost(plan.steps[k], ctx.cost_model),
            error_bound=0.0,
            step_span=(k,),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        k = self._find(plan, ctx)  # type: ignore[arg-type]
        return _drop_step_alias(plan, k, plan.steps[k].inputs[0])


# --------------------------------------------------------------------------- #
# Rule 8 — roll-through-Fourier (provenance-gated)
# --------------------------------------------------------------------------- #


class RollThroughFourier:
    """``ifft(diag(fft(x), h)) -> gather(roll x by s)`` for a linear-phase ``h``.

    A linear-phase diagonal in the Fourier basis is a circular shift (DFT shift
    theorem), which is pure host-side reindexing. Replaces a
    ``fft → diagonal_scale → ifft`` triple (≈13 cost weight) with one ``gather``
    (≈0.5) and *removes* the round-trip rounding — strictly at least as accurate.

    Hard provenance gate: ``ctx.provenance[h]`` must be an exact unit-modulus linear
    phase — :meth:`Provenance.is_linear_phase_diagonal`, the single predicate both
    the ``phase_ramp`` (``index`` *is* the shift ``s``, see
    :func:`phase_ramp_shift_provenance`) and exact-``dft_row`` (``index`` is the row
    ``f``; a DFT-row diagonal between ``fft`` and ``ifft`` is a circular shift by
    ``s = (-f) mod N`` via the modulation theorem, so :func:`provenance_from_v3`
    fires the roll too) spellings collapse into. The shift comes from
    :meth:`Provenance.linear_phase_shift`. The fft/diag intermediates must be
    single-use. The DFT acts over the last axis, so the gather does too.
    """

    name = "roll_through_fourier"

    @staticmethod
    def _shift(pv: Provenance) -> int | None:
        """Integer roll shift for an exact unit-modulus linear-phase diagonal.

        ``phase_ramp`` carries the shift directly; an exact ``dft_row`` of index
        ``f`` is the same diagonal ``exp(+i·2π·m·f/N)`` whose ``fft→·→ifft`` is a
        shift by ``(-f) mod N``. Returns ``None`` for any other provenance.
        """
        if pv is None or not pv.exact or pv.N is None or pv.index is None:
            return None
        if pv.kind == "phase_ramp":
            return int(pv.index) % int(pv.N)
        if pv.kind == "dft_row":
            return (-int(pv.index)) % int(pv.N)
        return None

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> tuple[int, int, int] | None:
        prod = _producer(plan)
        for j, ifft_s in enumerate(plan.steps):
            if ifft_s.op != "ifft":
                continue
            mid = ifft_s.inputs[0]
            si = prod.get(mid)
            if si is None or plan.steps[si].op != "diagonal_scale":
                continue
            scale = plan.steps[si]
            pv = ctx.provenance.get(scale.inputs[1])
            if pv is None or not pv.is_linear_phase_diagonal():
                continue
            data = scale.inputs[0]
            fi = prod.get(data)
            if fi is None or plan.steps[fi].op != "fft":
                continue
            if _use_count(plan, mid) != 1 or _use_count(plan, data) != 1:
                continue
            return fi, si, j
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan, ctx)
        if found is None:
            return None
        fi, si, j = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = (
            _step_cost(plan.steps[fi], ctx.cost_model)
            + _step_cost(plan.steps[si], ctx.cost_model)
            + _step_cost(plan.steps[j], ctx.cost_model)
            - ctx.cost_model.cost("gather")
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="Fourier-domain linear-phase diagonal is a circular shift; "
            "replace fft/diag/ifft with a gather (removes round-trip rounding)",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(fi, si, j),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        fi, si, j = self._find(plan, ctx)  # type: ignore[misc]
        pv = ctx.provenance[plan.steps[si].inputs[1]]
        x = plan.steps[fi].inputs[0]
        out = plan.steps[j].outputs[0]
        n, s = int(pv.N), int(pv.linear_phase_shift())  # type: ignore[arg-type]
        idx = ((np.arange(n, dtype=np.int64) - s) % n).astype(np.int64)
        gather = ExecutionStep("gather", (x,), (out,), params={"indices": idx, "axis": -1})
        new_steps: list[ExecutionStep] = []
        for k, step in enumerate(plan.steps):
            if k in (fi, si):
                continue
            new_steps.append(gather if k == j else step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 9 — diagonal · low-rank fold
# --------------------------------------------------------------------------- #


class DiagLowRankFold:
    """``diag(low_rank_apply(x), D) -> low_rank_apply(x)`` with ``D`` folded into U.

    ``D·(U diag(λ) Vᴴ x) = (D⊙U) diag(λ) Vᴴ x`` — the post-scale folds into the
    operator's left factor, so a data-length multiply disappears. Bit-exact (SPF
    multiply into U); :meth:`_find` refuses the fold when ``D⊙U`` would saturate the
    L plane, so the ``exact`` claim holds. ``D`` must be a plan-time constant.
    """

    name = "diag_low_rank_fold"

    def _find(self, plan: ExecutionPlan, ctx: RewriteContext) -> tuple[int, int] | None:
        prod = _producer(plan)
        for j, outer in enumerate(plan.steps):
            if outer.op != "diagonal_scale":
                continue
            if outer.inputs[1] not in ctx.constants:
                continue
            src = outer.inputs[0]
            i = prod.get(src)
            if i is None or plan.steps[i].op != "low_rank_apply":
                continue
            if _use_count(plan, src) != 1:
                continue
            # Refuse when folding D into U would saturate the L plane — keeps the
            # bit-exact claim honest (same gate as DiagonalScaleCompose).
            if _mul_l_saturates(
                self._u_words(plan.steps[i]), np.asarray(ctx.constants[outer.inputs[1]])[None, :]
            ):
                continue
            return i, j
        return None

    @staticmethod
    def _u_words(lr_step: ExecutionStep) -> np.ndarray:
        """The low-rank left-factor (U) words, from either step-param shape."""
        params = lr_step.params
        op = params.get("operator")
        u = op.u_factors if op is not None else params["u_factors"]
        return np.asarray(u, dtype=np.uint32)

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan, ctx)
        if found is None:
            return None
        i, j = found
        old = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="fold post-scale D into the low-rank left factor U (D⊙U); "
            "non-saturating (bit-exact)",
            old_cost=old,
            new_cost=old - _step_cost(plan.steps[j], ctx.cost_model),
            error_bound=0.0,
            step_span=(i, j),
        )

    def _folded_params(self, lr_step: ExecutionStep, d_words: np.ndarray) -> dict[str, Any]:
        params = dict(lr_step.params)
        if "operator" in params:
            op = params["operator"]
            u = np.asarray(op.u_factors, dtype=np.uint32)
            params["operator"] = replace(op, u_factors=spf_mul_words(u, d_words[None, :]))
        else:
            u = np.asarray(params["u_factors"], dtype=np.uint32)
            params["u_factors"] = spf_mul_words(u, d_words[None, :])
        return params

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        i, j = self._find(plan, ctx)  # type: ignore[misc]
        lr, outer = plan.steps[i], plan.steps[j]
        d_words = np.asarray(ctx.constants[outer.inputs[1]], dtype=np.uint32)
        new_params = self._folded_params(lr, d_words)
        new_steps: list[ExecutionStep] = []
        for k, step in enumerate(plan.steps):
            if k == i:
                new_steps.append(replace(step, outputs=outer.outputs, params=new_params))
            elif k == j:
                continue
            else:
                new_steps.append(step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# Rule 10 — reduce-after-circulant -> spectral DC bin
# --------------------------------------------------------------------------- #


class ReduceCirculantSpectral:
    """``reduce_sum(ifft(M)) -> gather(M, [0])`` over the transform axis.

    ``Σ_n ifft(M)[n] = M[0]`` (the DFT DC identity) for *any* spectrum ``M``, so a
    full-axis sum reduction of a circulant/FDWM apply ``M = diag(fft(x), h)`` needs
    neither the inverse transform nor the reduction — just the DC bin of the
    spectral product. Exact and provenance-free (holds for any ``h``); replaces the
    ``{ifft, reduce}`` pair with one host ``gather`` of index ``0`` on the last
    (transform) axis, killing an fft64-macro dispatch and the add-wall reduction.

    Gate: the reduce must be over the IFFT axis. ``axis == -1`` is the general
    per-row form (``M[..., 0]``) and is sound for *any* rank — the DC identity
    holds row-by-row. ``axis is None`` (reduce-all) is the DC bin **only when the
    spectrum is rank-1**: a rank>1 reduce-all also sums *across* rows
    (``Σ_rows M[row, 0]``), which the per-row DC gather does not reproduce. Plan
    time cannot read the operand rank, so ``axis is None`` fires only behind an
    explicit ``params["ndim"] == 1`` vouch on the reduce step — the same
    shape-at-plan-time gap that stops :class:`NoOpPrune` /
    :class:`ReducePermuteInvariance`. The single-output DC bin carries a trailing
    length-1 axis where the reduce removed it — semantically the same coherent sum.
    """

    name = "reduce_circulant_spectral"

    def _find(self, plan: ExecutionPlan) -> tuple[int, int, str] | None:
        prod = _producer(plan)
        for k, red in enumerate(plan.steps):
            if red.op != "reduce":
                continue
            if not _is_coherent_sum(red.reducer):
                continue
            axis = red.params.get("axis")
            if axis is None:
                # reduce-all == DC bin only for a vouched rank-1 spectrum.
                if int(red.params.get("ndim", 0)) != 1:
                    continue
            elif int(axis) != -1:
                continue
            src = red.inputs[0]
            ii = prod.get(src)
            if ii is None or plan.steps[ii].op != "ifft":
                continue
            if _use_count(plan, src) != 1:
                continue
            m_name = plan.steps[ii].inputs[0]
            return ii, k, m_name
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        ii, k, _ = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = (
            _step_cost(plan.steps[ii], ctx.cost_model)
            + _step_cost(plan.steps[k], ctx.cost_model)
            - ctx.cost_model.cost("gather")
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="Σ ifft(M) = M[0] (DC bin); drop the ifft and the reduce, "
            "take the DC bin of the spectral product",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(ii, k),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        ii, k, m_name = self._find(plan)  # type: ignore[misc]
        out = plan.steps[k].outputs[0]
        gather = ExecutionStep(
            "gather", (m_name,), (out,),
            params={"indices": 0, "axis": -1},
        )
        new_steps: list[ExecutionStep] = []
        for j, step in enumerate(plan.steps):
            if j == ii:
                continue
            new_steps.append(gather if j == k else step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


_SUM_REDUCERS = (
    ReducerPolicy.CARTESIAN_KAHAN,
    ReducerPolicy.CARTESIAN_NEUMAIER,
    ReducerPolicy.CARTESIAN_KULISCH,
    ReducerPolicy.AUTO,
)


def _is_coherent_sum(spec: Any) -> bool:
    if spec is None or spec.policy not in _SUM_REDUCERS:
        return False
    return spec.policy != ReducerPolicy.AUTO or (
        spec.routing == RoutingPolicy.PINNED
        and spec.domain in {AccumulationDomain.LOGPOLAR, AccumulationDomain.CARTESIAN}
    )


class DecimationFold:
    """``reduce_sum(gather(ifft(M), stride-s)) -> (1/s)·reduce_sum(gather(M, coset))``.

    The Poisson / decimation identity (algebra-structure.md #3): summing every
    ``s``-th sample of ``ifft(M)`` is a *coset sum* of the spectrum ::

        Σ_{i=0}^{m-1} ifft(M)[i·s] = (1/s)·Σ_{j=0}^{s-1} M[j·m],   N = m·s

    The full-axis sum (:class:`ReduceCirculantSpectral`, ``s = 1``) is the DC bin
    ``M[0]``; this is the strided generalisation — the decimated subsequence sums to
    ``1/s`` times the spectrum sampled on the coset ``{0, m, 2m, …, (s-1)m}`` (``s``
    bins). It drops the whole ``N``-point ``ifft`` (an fft64-macro dispatch) and turns
    the ``N``-term reduction into an ``s``-term one — a large win when ``s ≪ N``.

    Match: a sum ``reduce`` over the decimation axis whose source is a ``gather`` with
    *static* indices ``[0, s, 2s, …, (m-1)s]`` (a contiguous stride-``s`` subsequence
    from 0, ``s ≥ 2``), whose source is an ``ifft``. Honesty gate (mirrors
    :class:`DoubleFftReversal` / :class:`ReduceCirculantSpectral`): the ``ifft`` must
    carry ``params["n"]`` equal to ``m·s`` — the identity needs the gather to cover the
    *full* subsequence, which plan time can only confirm from the declared transform
    length. The reducer must be a complex sum (``CARTESIAN_KAHAN`` / ``AUTO``); a
    tropical/argmax reduce is not the coherent sum the identity assumes. The ``1/s``
    scale is a folded plan-time constant (the same SPF representation every fused
    coefficient carries), so the rewrite is exact in the algebra.
    """

    name = "decimation_fold"

    def _find(self, plan: ExecutionPlan) -> tuple[int, int, int, str, int, int] | None:
        prod = _producer(plan)
        for k, red in enumerate(plan.steps):
            if red.op != "reduce":
                continue
            if not _is_coherent_sum(red.reducer):
                continue
            axis = red.params.get("axis")
            if axis is None:
                if int(red.params.get("ndim", 0)) != 1:
                    continue
            elif int(axis) != -1:
                continue
            src = red.inputs[0]
            jj = prod.get(src)
            if jj is None or plan.steps[jj].op != "gather" or _use_count(plan, src) != 1:
                continue
            dec = plan.steps[jj]
            if "indices" not in dec.params or int(dec.params.get("axis", 0)) not in (-1, 0):
                continue
            idx = np.asarray(dec.params["indices"], dtype=np.int64).ravel()
            m = idx.shape[0]
            if m < 2 or int(idx[0]) != 0:
                continue
            s = int(idx[1] - idx[0])
            if s < 2 or not np.array_equal(idx, np.arange(m, dtype=np.int64) * s):
                continue
            isrc = dec.inputs[0]
            ii = prod.get(isrc)
            if ii is None or plan.steps[ii].op != "ifft" or _use_count(plan, isrc) != 1:
                continue
            n = plan.steps[ii].params.get("n")
            if n is None or int(n) != m * s:
                continue
            return ii, jj, k, plan.steps[ii].inputs[0], s, m

        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        ii, jj, k, _m_name, s, _m = found
        old = _plan_cost(plan, ctx.cost_model)
        saved = (
            _step_cost(plan.steps[ii], ctx.cost_model)
            + _step_cost(plan.steps[jj], ctx.cost_model)
            + _step_cost(plan.steps[k], ctx.cost_model)
            - ctx.cost_model.cost("gather")
            - ctx.cost_model.cost("reduce")
            - ctx.cost_model.cost("diagonal_scale")
        )
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason=f"Σ ifft(M)[i·{s}] = (1/{s})·Σ M over the {s}-bin coset; drop the "
            "ifft and decimation, sum the spectrum on the coset",
            old_cost=old,
            new_cost=old - saved,
            error_bound=0.0,
            step_span=(ii, jj, k),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        ii, jj, k, m_name, s, m = self._find(plan)  # type: ignore[misc]
        red = plan.steps[k]
        out = red.outputs[0]
        coset = (np.arange(s, dtype=np.int64) * m).astype(np.int64)  # {0, m, …, (s-1)m}
        coset_name = f"{out}__coset"
        summed_name = f"{out}__coset_sum"
        scale_name = f"{out}__inv_s"
        folded[scale_name] = encode_words(np.full(1, 1.0 / float(s), dtype=np.complex128))
        gather = ExecutionStep(
            "gather", (m_name,), (coset_name,), params={"indices": coset, "axis": -1}
        )
        reduce_step = ExecutionStep(
            "reduce", (coset_name,), (summed_name,),
            reducer=red.reducer,
            # Flat-priced like any reduce (no n_terms): the win is dropping the
            # N-point ifft, not re-pricing an s-term sum against the cost-scaler.
            params={"axis": -1, "purpose": "decimation_coset_sum", "coset_bins": s},
        )
        scale = ExecutionStep("diagonal_scale", (summed_name, scale_name), (out,))
        new_steps: list[ExecutionStep] = []
        for j, step in enumerate(plan.steps):
            if j in (ii, jj):
                continue
            if j == k:
                new_steps.extend((gather, reduce_step, scale))
            else:
                new_steps.append(step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


class ReducePermuteInvariance:
    """``reduce(gather(x, perm)) -> reduce(x)`` when the gather is a full-axis permutation.

    A sum over a whole axis is invariant under any bijection of that axis, so a
    permutation gather feeding a full reduction is dead — drop it and reduce the
    source directly. The honest sibling of :class:`ReduceCirculantSpectral`: that
    one kills the *spectral* reduce, this one kills the *host-index* reduce.

    Soundness needs the gather to be **onto** the reduced axis — a permutation of
    the *entire* axis, not a sub-selection — and that coverage cannot be read from
    the index *values* alone (a permutation of ``range(m)`` on a longer axis is a
    drop, which changes the sum). So the gather must carry ``params["n"]`` (the axis
    length) and ``indices`` must be a permutation of ``range(n)``; without ``n`` the
    rule refuses — the same axis-length gap that stops :class:`NoOpPrune` from
    dropping an identity gather. The reduce must be over the gathered axis
    (``axis is None`` reduce-all, or ``axis == gather axis``). Host-side, exact.
    """

    name = "reduce_permute_invariance"

    def _find(self, plan: ExecutionPlan) -> tuple[int, int] | None:
        prod = _producer(plan)
        for k, red in enumerate(plan.steps):
            if red.op != "reduce":
                continue
            src = red.inputs[0]
            gi = prod.get(src)
            if gi is None or plan.steps[gi].op != "gather":
                continue
            if _use_count(plan, src) != 1:
                continue
            g = plan.steps[gi]
            n = g.params.get("n")
            if n is None:
                continue
            n = int(n)
            g_axis = int(g.params.get("axis", 0))
            r_axis = red.params.get("axis")
            if r_axis is not None and int(r_axis) != g_axis:
                continue
            idx = np.asarray(g.params.get("indices", ()), dtype=np.int64).reshape(-1)
            if idx.size != n or not np.array_equal(np.sort(idx), np.arange(n, dtype=np.int64)):
                continue
            return gi, k
        return None

    def match(self, plan: ExecutionPlan, ctx: RewriteContext) -> RewriteMatch | None:
        found = self._find(plan)
        if found is None:
            return None
        gi, _k = found
        old = _plan_cost(plan, ctx.cost_model)
        return RewriteMatch(
            name=self.name,
            confidence=1.0,
            exact=True,
            reason="full-axis sum is invariant under a permutation gather; drop the "
            "gather and reduce the source directly",
            old_cost=old,
            new_cost=old - _step_cost(plan.steps[gi], ctx.cost_model),
            error_bound=0.0,
            step_span=(gi,),
        )

    def apply(
        self, plan: ExecutionPlan, ctx: RewriteContext, folded: dict[str, np.ndarray]
    ) -> ExecutionPlan:
        gi, k = self._find(plan)  # type: ignore[misc]
        x = plan.steps[gi].inputs[0]
        new_red = replace(plan.steps[k], inputs=(x,))
        new_steps: list[ExecutionStep] = []
        for j, step in enumerate(plan.steps):
            if j == gi:
                continue
            new_steps.append(new_red if j == k else step)
        return ExecutionPlan(
            steps=tuple(new_steps), factorization_metadata=plan.factorization_metadata
        )


# --------------------------------------------------------------------------- #
# PL-lowering bridge — measure the rewrite's hardware payoff
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class LoweringDelta:
    """Dispatch/overlay deltas of a rewrite, via :mod:`spf.tensor.pl_lowering`."""

    profile_before: str | None
    profile_after: str | None
    pl_dispatch_before: int
    pl_dispatch_after: int
    host_dispatch_before: int
    host_dispatch_after: int
    matches: tuple[RewriteMatch, ...]


def rewrite_then_lower(
    plan: ExecutionPlan,
    ctx: RewriteContext | None = None,
    *,
    lower_kwargs: Mapping[str, Any] | None = None,
) -> tuple[ExecutionPlan, LoweringDelta]:
    """Rewrite ``plan`` then lower both forms, reporting the PL/host dispatch delta.

    Round-trips the plan-step win to hardware: a collapsed circulant or a
    roll-through-Fourier removes resident-CU (``fft64``) dispatches, which can let
    ``select_profile`` pick a smaller overlay. Returns the rewritten plan and a
    :class:`LoweringDelta` with before/after PL + host dispatch counts and the
    selected bake profile each side.
    """
    from . import pl_lowering

    lk = dict(lower_kwargs or {})
    before = pl_lowering.lower(plan, **lk)
    new_plan, matches, _ = rewrite_plan(plan, ctx)
    after = pl_lowering.lower(new_plan, **lk)

    def counts(lp: Any) -> tuple[int, int]:
        pl = sum(1 for o in lp.ops if o.target == pl_lowering.PL_TARGET)
        host = sum(1 for o in lp.ops if o.target == pl_lowering.HOST_TARGET)
        return pl, host

    pl_b, host_b = counts(before)
    pl_a, host_a = counts(after)
    return new_plan, LoweringDelta(
        profile_before=before.select_profile(),
        profile_after=after.select_profile(),
        pl_dispatch_before=pl_b,
        pl_dispatch_after=pl_a,
        host_dispatch_before=host_b,
        host_dispatch_after=host_a,
        matches=tuple(matches),
    )


# Order: prune dead steps first, then the *most* reducing structural rewrites
# (roll-through-Fourier — a whole fft/diag/ifft triple → one host gather), then
# inverse cancellation, the constant/provenance folds, the other composes, and the
# circulant tag last (annotation-only, idempotent).
#
# RollThroughFourier deliberately precedes FftIfftCancel: on a chained
# roll ``C(h₁)∘C(h₂)`` the two are *non-confluent* — cancel-first merges the inner
# round-trip into a two-diagonal circulant (a gather can no longer be recognised),
# while roll-first collapses each triple to a gather and ``GatherPermuteCompose``
# fuses them into one ``roll(s₁+s₂)``. Both are semantically identical; the gather
# form is strictly cheaper, so roll wins the tie. (Roll only fires behind explicit
# ``phase_ramp`` provenance, so it never pre-empts the FDWM-power path, whose
# coefficients are plain constants — see test T8_78 for the characterised
# order-dependence.)
#
# FDWM matrix power ``C(λ)^k -> C(λ^k)`` stays emergent: a chained circulant of
# *constant* spectra collapses via FftIfftCancel + DiagonalScaleCompose, whose
# self-compose (a == b) raises the spectrum to the k-th power one fusion at a time.
#
# MAINTAINER NOTE: this rule set is *not* globally confluent — order encodes a cost
# preference (most-reducing first), characterised by T8.78. The confluence property
# test (T8.75) exercises a *battery*, not the full interaction matrix, so adding a
# rule here is not automatically safe: check it against the existing order (does it
# race RollThroughFourier / FftIfftCancel for the same site? does it change the
# fixpoint under a shuffled `rules=`?) and extend T8.75/T8.78 to cover it.
RULES: tuple[RewriteRule, ...] = (
    NoOpPrune(),
    PhaseShearCompose(),
    RollThroughFourier(),
    FftIfftCancel(),
    DoubleFftReversal(),
    RampFusion(),
    DiagonalScaleCompose(),
    DiagLowRankFold(),
    ReduceCirculantSpectral(),
    DecimationFold(),
    ReducePermuteInvariance(),
    PhaseRampOrthogonality(),
    CharacterPhaseBind(),
    CharacterBankBind(),
    ProductGroupCharacterBind(),
    GatherPermuteCompose(),
    CirculantTag(),
)
