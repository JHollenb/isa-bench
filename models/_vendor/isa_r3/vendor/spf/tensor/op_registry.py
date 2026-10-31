"""Execution-op registry — the single source of truth for the SPFTensor VM.

:class:`~spf.tensor.v2.ExecutionPlan` is a straight-line dataflow interpreter
over a *closed, algebraically meaningful* op set: every op has a known planner
cost and a known lowering story (resident PL CU, host monomial, or reference
oracle). That closure is what lets the cost model and ``pl_lowering`` reason
about a plan at all — a general control-flow VM would forfeit it.

Historically the op set was a frozen ``set`` literal in :mod:`spf.tensor.v2` and
the per-op planner weights a separate dict in :mod:`spf.tensor.rewrite`, so
adding one operator meant editing the validator, the cost dict, the executor,
and both backends in lockstep. This module collapses the *declarative* half of
that into one table: each :class:`OpSpec` carries the op's cost weight, whether
it is a contraction (needs a :class:`ReducerSpec` boundary), and how it lowers.
``v2`` derives its validator from here and ``rewrite`` derives its cost weights
from here, so a new spectral operator registers in one place.

This is a leaf module: it imports nothing from the rest of the package, so it
sits below ``v2``/``rewrite``/``operators`` in the import graph.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "OP_REGISTRY",
    "OpSpec",
    "contraction_ops",
    "cost_weights",
    "execution_ops",
    "lifted_ops",
    "op_spec",
    "register_op",
]


@dataclass(frozen=True, slots=True)
class OpSpec:
    """Declarative metadata for one :class:`ExecutionPlan` op.

    The ``category`` mirrors the PL/host split already documented in
    :mod:`spf.tensor.pl_lowering`:

    * ``free_monomial`` — pure index manipulation, stays host-side (PS).
    * ``resident_cu`` — backed by a resident PL compute unit (``mul``/``ramp``).
    * ``transform`` — FFT/IFFT; resident ``fft64`` macro or host fallback.
    * ``reduction`` — a contraction with a :class:`ReducerSpec`; PL when the
      accumulation domain is tropical/log-polar, host PS on a CARTESIAN fallback.
    * ``contraction`` — lifted contraction; host (PS) for now.
    * ``reference`` — numpy oracle, never deployed to a backend.
    """

    name: str
    cost_weight: float
    category: str
    needs_reducer: bool = False
    pl_resident: bool = False
    host_supported: bool = True
    reference_only: bool = False
    summary: str = ""

    @property
    def deployable(self) -> bool:
        """True iff this op lowers to a real backend (not a numpy oracle)."""
        return not self.reference_only

    @property
    def lifted(self) -> bool:
        """True iff this is a *lifted* contraction (host PS / oracle, not resident).

        The ``contraction`` and ``reference`` categories — ``low_rank_apply`` and
        ``decoded_contract_reference`` — are lifted out of the resident-CU plane.
        ``reduce`` (category ``reduction``) is *not* lifted: it has a resident PL
        path and only falls back to host on a CARTESIAN domain.
        """
        # Structured v0.2 reference nodes use the same declarative registry so
        # the SSA renderer can lower through ExecutionPlan, but they are not
        # legacy TensorBlock contractions.  Keep them out of the legacy
        # lowering/reducer classification until a backend-specific renderer
        # lowering exists.
        return not self.name.startswith("v02_") and self.category in ("contraction", "reference")


# Order mirrors the original ``EXECUTION_OPS`` frozenset / ``_OP_COST`` dict.
_BUILTIN_OPS: tuple[OpSpec, ...] = (
    OpSpec("permute", 0.5, "free_monomial",
           summary="axis transpose; host index work"),
    OpSpec("gather", 0.5, "free_monomial",
           summary="reindex along an axis; host index work"),
    OpSpec("diagonal_scale", 1.0, "resident_cu", pl_resident=True,
           summary="per-element SPF multiply by a diagonal"),
    OpSpec("elementwise_mul", 1.0, "resident_cu", pl_resident=True,
           summary="per-element SPF multiply of two operands"),
    OpSpec("ramp", 1.0, "resident_cu", pl_resident=True,
           summary="add a linear (and optional higher-order) phase ramp"),
    OpSpec("fft", 6.0, "transform", pl_resident=True,
           summary="forward DFT; resident fft64 macro or host fallback"),
    OpSpec("ifft", 6.0, "transform", pl_resident=True,
           summary="inverse DFT; conj/fft64/conj + 1/N or host fallback"),
    OpSpec("reduce", 3.0, "reduction", needs_reducer=True, pl_resident=True,
           summary="contraction over an axis; CARTESIAN domain falls back to host"),
    OpSpec("low_rank_apply", 6.0, "contraction", needs_reducer=True,
           summary="lifted low-rank contraction; host (PS)"),
    OpSpec("block_circulant_apply", 8.0, "reference", needs_reducer=True,
           reference_only=True,
           summary="numpy oracle block-circulant spectral apply; reference only until C/native parity"),
    OpSpec("decoded_contract_reference", 2.0, "reference", needs_reducer=True,
           reference_only=True,
           summary="numpy oracle contraction; reference only, never lowered"),
    OpSpec("decoded_sum_reference", 2.0, "reference", needs_reducer=True,
           reference_only=True,
           summary="numpy oracle addition boundary; reference only, never lowered"),
    OpSpec("packed_sum", 3.0, "native_phase", needs_reducer=True,
           summary="balanced packed SPF sum with integer factored-ROM joins"),
    OpSpec("decoded_pushforward_reference", 2.0, "reference", needs_reducer=True,
           reference_only=True,
           summary="numpy oracle pushforward over named kernel axes; reference only, never lowered"),
    OpSpec("decoded_low_rank_scaled_reference", 6.0, "reference", needs_reducer=True,
           reference_only=True,
           summary="factor contraction with wide coefficient exponent metadata; one output quantization"),
    OpSpec("decoded_materialized_reference", 0.25, "reference",
           reference_only=True,
           summary="declared reference output for closed materialized kernel views; no summation"),
    OpSpec(
        "contract",
        6.0,
        "contraction",
        needs_reducer=True,
        pl_resident=False,
        summary=(
            "data×data contraction C[i,j] = reduce_k(A[i,k] ⊗ B[k,j]) over packed SPF words. "
            "params: transpose_b (bool, B given as N×K), causal (bool, outputs j>i forced to WORD_ZERO)"
        ),
    ),
    OpSpec(
        "softmax_lse",
        4.0,
        "reduction",
        needs_reducer=True,
        pl_resident=False,
        summary=(
            "log-domain causal/dense softmax — scores quantized to wide L-codes "
            "(Lq = round_half_away(s*beta*2^f / ln2)), LSE = max + log2(sum 2^((Lq-m)/2^f)) "
            "re-quantized, weights = 2^((Lq - lse)/2^f). "
            "params: causal (bool), l_frac_bits (int, default 16), beta (float, default 1.0), "
            "out ('f32' default | 'words'), pre_quantized (bool — scores are int32 L-codes, "
            "beta must be 1.0)"
        ),
    ),
    OpSpec(
        "route",
        10.0,
        "contraction",
        needs_reducer=True,
        pl_resident=False,
        summary=(
            "fused phase-native attention: integer-ΔΘ concordance scores (cos twiddle LUT, "
            "fp64 accumulate) quantized once to L-codes, log-domain softmax_lse "
            "(pre_quantized), P⊛V mix via contract behind the reducer boundary. "
            "Words in, words out, no decode inside. Full memory — no rank/prototype params. "
            "params: causal (bool), beta (float, default 8.0), l_frac_bits (int, default 16)"
        ),
    ),
    OpSpec(
        "gauge_norm",
        1.5,
        "resident_cu",
        pl_resident=False,
        summary=(
            "log-domain gauge norm: L -= per-row statistic (+ learned per-channel γ codes), "
            "Θ untouched, sentinels pass through. The row statistic is over L PLANES "
            "(integer codes, not a coherent fold) so no ReducerSpec boundary. "
            "params: tier ('linf'|'gml'|'rms', default 'gml'), axis (int, default -1), "
            "gamma_codes (optional per-channel int L-code list)"
        ),
    ),
    OpSpec(
        "l_gate",
        0.5,
        "resident_cu",
        pl_resident=False,
        summary=(
            "tropical magnitude gate: finite words with L < tau[channel] become WORD_ZERO "
            "(comparator ReLU analogue; NaN sentinel passes through). "
            "params: tau_codes (scalar or per-channel int L-code list), axis (int, default -1)"
        ),
    ),
    # Integer address circuits execute on the host, without a resident PL CU.
    OpSpec("phase_character_features", 1.0, "native_phase",
           summary="sparse integer character bank on unit-circle SPF32 word channels; no coherent fold"),
    OpSpec("phase_shear", 1.0, "native_phase",
           summary="reversible integer polynomial/ROM phase shear; optional relative controller"),
    OpSpec("monomial_scan", 1.0, "native_phase",
           summary="addition-free prefix scan of permutation x phase-diagonal monomials; gathers + integer adds"),
    OpSpec("phase_neuron", 1.0, "native_phase",
           summary="reversible character-controlled ROM coupling: targets += C @ ROM(A theta + b); no coherent fold"),
    OpSpec("routed_buckets", 1.0, "native_phase",
           summary="causal addressed packed records with phase-clock span transport"),

    # Structured SPF v0.2 renderer operations.  These are deliberately
    # reference-only and explicitly named so they cannot alter the legacy
    # tensor/FFT defaults.  The renderer module owns their semantics; this
    # registry only makes their typed names legal in ExecutionPlan.
    OpSpec("v02_jet2d", 1.0, "reference", reference_only=True,
           summary="construct a typed v0.2 Jet2D field"),
    OpSpec("v02_jet_add", 1.0, "reference", reference_only=True,
           summary="add two compatible v0.2 quadratic jets"),
    OpSpec("v02_jet_transform", 1.0, "reference", reference_only=True,
           summary="affine-transform a typed v0.2 Jet2D field"),
    OpSpec("v02_jet_materialize", 3.0, "reference", reference_only=True,
           summary="materialize a v0.2 Jet2D field"),
    OpSpec("v02_atom_create", 1.0, "reference", reference_only=True,
           summary="construct a Gaussian-chirp atom"),
    OpSpec("v02_atom_mul", 1.0, "reference", reference_only=True,
           summary="multiply Gaussian-chirp atoms"),
    OpSpec("v02_atom_div", 1.0, "reference", reference_only=True,
           summary="divide Gaussian-chirp atoms"),
    OpSpec("v02_atom_affine", 1.0, "reference", reference_only=True,
           summary="affine-transform a Gaussian-chirp atom"),
    OpSpec("v02_atom_fourier", 3.0, "reference", reference_only=True,
           summary="Fourier-transform a Gaussian-chirp atom"),
    OpSpec("v02_atom_inverse_fourier", 3.0, "reference", reference_only=True,
           summary="inverse-Fourier-transform a Gaussian-chirp atom"),
    OpSpec("v02_atom_convolve", 5.0, "reference", reference_only=True,
           summary="convolve Gaussian-chirp atoms through Fourier closure"),
    OpSpec("v02_atom_integral", 3.0, "reference", reference_only=True,
           summary="integrate a Gaussian-chirp atom"),
    OpSpec("v02_atom_pixel_integral", 3.0, "reference", reference_only=True,
           summary="integrate an atom against a Gaussian footprint"),
    OpSpec("v02_atom_derivative", 2.0, "reference", reference_only=True,
           summary="evaluate an analytic atom gradient"),
    OpSpec("v02_atom_hessian", 2.0, "reference", reference_only=True,
           summary="evaluate an analytic atom Hessian"),
    OpSpec("v02_atom_bound", 3.0, "reference", reference_only=True,
           summary="compute a certified atom magnitude bound"),
    OpSpec("v02_atom_bound_log", 3.0, "reference", reference_only=True,
           summary="compute a logarithmic atom magnitude bound"),
    OpSpec("v02_atom_to_jet", 3.0, "reference", reference_only=True,
           summary="lower a two-dimensional atom to a quadratic jet"),
    OpSpec("v02_atom_mixture", 1.0, "reference", reference_only=True,
           summary="construct a typed Gaussian-chirp atom mixture"),
    OpSpec("v02_mixture", 1.0, "reference", reference_only=True,
           summary="alias for v02_atom_mixture"),
    OpSpec("v02_mixture_add", 1.0, "reference", reference_only=True,
           summary="concatenate symbolic atom mixtures"),
    OpSpec("v02_mixture_prune", 3.0, "reference", reference_only=True,
           summary="certified atom-mixture pruning"),
    OpSpec("v02_mixture_merge_equal", 2.0, "reference", reference_only=True,
           summary="exactly merge equal atom shapes"),
    OpSpec("v02_mixture_eval", 3.0, "reference", reference_only=True,
           summary="evaluate a Gaussian-chirp mixture"),
    OpSpec("v02_mixture_materialize", 4.0, "reference", reference_only=True,
           summary="materialize a Gaussian-chirp mixture"),
    OpSpec("v02_mixture_lower", 4.0, "reference", reference_only=True,
           summary="lower a symbolic mixture at a materialization boundary"),
    OpSpec("v02_separable", 1.0, "reference", reference_only=True,
           summary="construct a typed separable field"),
    OpSpec("v02_separable_field", 1.0, "reference", reference_only=True,
           summary="alias for v02_separable"),
    OpSpec("v02_separable_eval", 3.0, "reference", reference_only=True,
           summary="evaluate a separable v0.2 field"),
    OpSpec("v02_separable_materialize", 4.0, "reference", reference_only=True,
           summary="materialize a separable v0.2 field"),
    OpSpec("v02_hermitian_synthesis", 3.0, "reference", reference_only=True,
           summary="synthesize a Hermitian signed-real field"),
    OpSpec("v02_spherical_gaussian", 1.0, "reference", reference_only=True,
           summary="construct a spherical Gaussian"),
    OpSpec("v02_spherical_gaussian_product", 1.0, "reference", reference_only=True,
           summary="multiply spherical Gaussians"),
    OpSpec("v02_spherical_gaussian_integral", 2.0, "reference", reference_only=True,
           summary="integrate a spherical Gaussian"),
    OpSpec("v02_transport_identity", 1.0, "reference", reference_only=True,
           summary="construct the affine transport identity"),
    OpSpec("v02_transport", 1.0, "reference", reference_only=True,
           summary="construct an ordered affine Transport value"),
    OpSpec("v02_transport_compose", 2.0, "reference", reference_only=True,
           summary="compose two ordered affine transports"),
    OpSpec("v02_transport_reduce", 3.0, "reference", reference_only=True,
           summary="reduce ordered Transport values with a balanced tree"),
    OpSpec("v02_transport_apply", 2.0, "reference", reference_only=True,
           summary="apply an affine Transport to a background value"),
    OpSpec("v02_logadd_pos", 1.0, "reference", reference_only=True,
           summary="typed positive log addition"),
    OpSpec("v02_logsub_pos", 1.0, "reference", reference_only=True,
           summary="typed positive log subtraction"),
    OpSpec("v02_logadd_signed", 1.0, "reference", reference_only=True,
           summary="typed signed-real log addition"),
    OpSpec("v02_complex_pair_add", 2.0, "reference", reference_only=True,
           summary="reference complex pair sum/difference"),
    OpSpec("v02_complex_pair_select", 0.5, "reference", reference_only=True,
           summary="explicitly project one result from a complex pair"),
    OpSpec("v02_phase_shift", 0.5, "reference", reference_only=True,
           summary="apply an explicit modular phase shift to SPF_COMPLEX"),
    OpSpec("v02_promote_complex", 1.0, "reference", reference_only=True,
           summary="explicitly promote log magnitude plus phase to SPF_COMPLEX"),
    OpSpec("v02_phase_magnitude_to_complex", 1.0, "reference", reference_only=True,
           summary="promote LOG_POS and PHASE to SPF_COMPLEX"),
    OpSpec("v02_cast", 1.0, "reference", reference_only=True,
           summary="explicitly cast between declared v0.2 value kinds"),
    OpSpec("v02_extract", 0.5, "reference", reference_only=True,
           summary="explicitly extract one named result from a multi-result node"),
    OpSpec("v02_abs2", 1.0, "reference", reference_only=True,
           summary="demote SPF complex values to LOG_POS magnitude-square"),
    OpSpec("v02_materialize", 4.0, "reference", reference_only=True,
           summary="materialize a typed structured field"),
    OpSpec("v02_fft", 6.0, "reference", reference_only=True,
           summary="execute the explicit v0.2 SPF-32 FFT"),
    OpSpec("v02_ifft", 6.0, "reference", reference_only=True,
           summary="execute the explicit v0.2 SPF-32 inverse FFT"),
    OpSpec("v02_tone", 1.0, "reference", reference_only=True,
           summary="apply log-native tone mapping"),
    OpSpec("v02_display", 1.0, "reference", reference_only=True,
           summary="map log-native values to display codes"),
    # The renderer design names these operations explicitly.  They remain
    # reference-only in the Python implementation, but registering the full
    # typed vocabulary makes the IR/ISA boundary complete and prevents a
    # caller from silently falling back to an untyped legacy operation.
    OpSpec("v02_field_constant", 0.25, "reference", reference_only=True,
           summary="materialize a typed field constant"),
    OpSpec("v02_field_asset", 0.25, "reference", reference_only=True,
           summary="load a typed field asset supplied by the caller"),
    OpSpec("v02_field_reference", 0.25, "reference", reference_only=True,
           summary="forward a typed field reference supplied by the caller"),
    OpSpec("v02_atom_product", 1.0, "reference", reference_only=True,
           summary="alias for Gaussian-chirp atom multiplication"),
    OpSpec("v02_mixture_concat", 1.0, "reference", reference_only=True,
           summary="concatenate symbolic atom mixtures"),
    OpSpec("v02_mixture_cull", 3.0, "reference", reference_only=True,
           summary="alias for certified mixture pruning"),
    OpSpec("v02_mixture_bound", 3.0, "reference", reference_only=True,
           summary="compute per-atom mixture magnitude bounds"),
    OpSpec("v02_mixture_factor", 2.0, "reference", reference_only=True,
           summary="factor an exact common atom shape from a mixture"),
    OpSpec("v02_mixture_factor_select", 0.5, "reference", reference_only=True,
           summary="explicitly project factor or residual from a mixture factorization"),
    OpSpec("v02_mixture_to_jets", 4.0, "reference", reference_only=True,
           summary="lower smooth mixture atoms to quadratic jets"),
    OpSpec("v02_log_mul", 1.0, "reference", reference_only=True,
           summary="typed LOG_POS multiplication"),
    OpSpec("v02_log_div", 1.0, "reference", reference_only=True,
           summary="typed LOG_POS division"),
    OpSpec("v02_phase_add", 0.5, "reference", reference_only=True,
           summary="modular phase addition"),
    OpSpec("v02_phase_sub", 0.5, "reference", reference_only=True,
           summary="modular phase subtraction"),
    OpSpec("v02_phase_conj", 0.5, "reference", reference_only=True,
           summary="modular phase conjugation"),
    OpSpec("v02_phase_compose", 0.5, "reference", reference_only=True,
           summary="compose phase-only values"),
    OpSpec("v02_complex_mul", 1.0, "reference", reference_only=True,
           summary="typed SPF_COMPLEX multiplication"),
    OpSpec("v02_complex_div", 1.0, "reference", reference_only=True,
           summary="typed SPF_COMPLEX division"),
    OpSpec("v02_coordinate_remap", 1.0, "reference", reference_only=True,
           summary="apply an affine coordinate remap"),
    OpSpec("v02_bandlimit", 1.0, "reference", reference_only=True,
           summary="select a declared frequency band"),
    OpSpec("v02_support_intersect", 1.0, "reference", reference_only=True,
           summary="intersect axis-aligned supports"),
    OpSpec("v02_mask", 0.5, "reference", reference_only=True,
           summary="apply a boolean field mask"),
    OpSpec("v02_reduce_positive", 3.0, "reference", reference_only=True,
           summary="reduce LOG_POS values with the v0.2 pivot reducer"),
    OpSpec("v02_reduce_signed", 3.0, "reference", reference_only=True,
           summary="reduce LOG_SIGNED values with the v0.2 pivot reducer"),
    OpSpec("v02_reduce_complex", 3.0, "reference", reference_only=True,
           summary="reduce SPF_COMPLEX values with the v0.2 pivot reducer"),
    OpSpec("v02_reduce_optical_depth", 2.0, "reference", reference_only=True,
           summary="reduce OPTICAL_DEPTH values by ordinary addition"),
    OpSpec("v02_reduce_transport", 3.0, "reference", reference_only=True,
           summary="reduce ordered affine transports"),
    OpSpec("v02_reduce_tropical", 1.0, "reference", reference_only=True,
           summary="reduce a tropical min/max field"),
    OpSpec("v02_visibility_select", 1.0, "reference", reference_only=True,
           summary="select visible values from a boolean mask"),
    OpSpec("v02_depth_min", 1.0, "reference", reference_only=True,
           summary="select the minimum tropical depth"),
    OpSpec("v02_tile_cull", 2.0, "reference", reference_only=True,
           summary="cull bounded atoms against a tile threshold"),
    OpSpec("v02_sort", 1.0, "reference", reference_only=True,
           summary="deterministically sort reference values"),
    OpSpec("v02_sparse_synthesis", 3.0, "reference", reference_only=True,
           summary="synthesize sparse values into a dense reference tile"),
    OpSpec("v02_fft_forward", 6.0, "reference", reference_only=True,
           summary="alias for the v0.2 forward FFT"),
    OpSpec("v02_fft_inverse", 6.0, "reference", reference_only=True,
           summary="alias for the v0.2 inverse FFT"),
    OpSpec("v02_hermitian_synth", 3.0, "reference", reference_only=True,
           summary="alias for Hermitian synthesis"),
    OpSpec("v02_materialize_tile", 4.0, "reference", reference_only=True,
           summary="alias for typed tile materialization"),
    OpSpec("v02_display_convert", 1.0, "reference", reference_only=True,
           summary="alias for display conversion"),
)

OP_REGISTRY: dict[str, OpSpec] = {spec.name: spec for spec in _BUILTIN_OPS}


def register_op(spec: OpSpec, *, replace: bool = False) -> None:
    """Register a new execution op (the VM-extension hook).

    A new spectral operator becomes plan-legal by registering its :class:`OpSpec`
    here; :class:`ExecutionStep` validates against the live registry, so no edit
    to the executor's op whitelist is needed. The *executor*, *cost*, and
    *backend lowerings* must still be taught the op — registration declares the
    contract, it does not implement the semantics. Raises on a duplicate name
    unless ``replace=True`` so a typo cannot silently shadow a builtin.
    """
    if spec.name in OP_REGISTRY and not replace:
        raise ValueError(f"execution op {spec.name!r} already registered")
    OP_REGISTRY[spec.name] = spec


def op_spec(name: str) -> OpSpec:
    """Return the :class:`OpSpec` for ``name`` or raise ``KeyError``."""
    return OP_REGISTRY[name]


def execution_ops() -> frozenset[str]:
    """Live set of legal execution-op names (the VM's op whitelist)."""
    return frozenset(OP_REGISTRY)


def contraction_ops() -> frozenset[str]:
    """Ops that require a :class:`ReducerSpec` boundary."""
    return frozenset(name for name, spec in OP_REGISTRY.items() if spec.needs_reducer)


def lifted_ops() -> frozenset[str]:
    """Lifted contractions — host PS / oracle, not resident-CU (see :attr:`OpSpec.lifted`)."""
    return frozenset(name for name, spec in OP_REGISTRY.items() if spec.lifted)


def cost_weights() -> dict[str, float]:
    """Per-op planner weights, keyed by op name (the old ``_OP_COST`` table)."""
    return {name: spec.cost_weight for name, spec in OP_REGISTRY.items()}
