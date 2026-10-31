"""Typed SFR v0.2 compiler and executable program.

The repository already contains two useful, but deliberately separate, layers:
``spf.renderer.v02`` is the small reference VM and ``spf.renderer.model`` is
the metadata-rich field contract.  This module is the missing seam between
them.  It owns an immutable SSA graph, performs only certified rewrites, and
lowers the result back to the reference VM.  Native backends are checked at
compile time; an unavailable backend is never silently replaced by Python.

The implementation is intentionally conservative.  A rewrite either has an
explicit precondition and a certificate in the report, or it is left alone.
That makes this module useful as an executable specification while the native
GPU/SPU lowerings remain proposal-stage work.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeVar, cast

import numpy as np

from spf.renderer.backends import BackendUnavailableError, SFRBackend, get_backend
from spf.renderer.model import (
    Coherence,
    CoordinateChart,
    Domain,
    ErrorBudget,
    ErrorContract,
    Field,
    FieldOperation,
    FieldRepresentation,
    PrecisionContract,
    PrecisionMode,
    Region,
    RepresentationKind,
    Semantic,
    Support,
)
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport
from spf.renderer.v02 import (
    FieldIR,
    FieldKind,
    MultiResult,
    RenderPlan,
    TypedField,
)
from spf.renderer.v02 import (
    FieldNode as V02FieldNode,
)
from spf.tensor import v02 as arithmetic


class CompilationError(RuntimeError):
    """Base class for an SFR graph that cannot be compiled safely."""


class TypeCheckError(CompilationError, TypeError):
    """Raised when an edge crosses an algebra or metadata boundary."""


class RewritePreconditionError(CompilationError, ValueError):
    """Raised when a requested strict rewrite lacks its proof conditions."""


class BackendCapabilityError(CompilationError):
    """Raised when a graph asks a backend for an unsupported capability."""


class PassName(StrEnum):
    TYPE_CHECK_CANONICALIZE = "type_check_canonicalize"
    PROPAGATE = "chart_support_bound_propagation"
    AFFINE_FUSION = "affine_fusion"
    PHASE_FUSION = "phase_fusion"
    ATOM_PRODUCT_FUSION = "atom_product_fusion"
    MIXTURE_CONCATENATION = "symbolic_mixture_concatenation"
    COMMON_FACTOR = "common_factor_and_separable_rewrites"
    FFT_CANCELLATION = "fft_ifft_cancellation"
    HERMITIAN_LOWERING = "hermitian_lowering"
    PRUNING = "bound_certified_pruning"
    DEAD_CODE = "dead_code_elimination"
    REDUCTION_ANNOTATION = "reduction_annotation"
    MATERIALIZATION = "materialization_planning"
    CAPABILITY = "backend_capability_validation"
    LIVENESS = "liveness_memory_schedule"


def _kind(value: FieldKind | str | None) -> FieldKind:
    if isinstance(value, FieldKind):
        return value
    if value is None:
        return FieldKind.ANY
    name = str(value).strip().lower().replace("-", "_").split(".")[-1]
    aliases = {
        "jet": FieldKind.JET2D,
        "jet_2d": FieldKind.JET2D,
        "atom": FieldKind.ATOM,
        "gaussian_chirp_atom": FieldKind.ATOM,
        "mixture": FieldKind.ATOM_MIXTURE,
        "atom_mixture": FieldKind.ATOM_MIXTURE,
        "separable": FieldKind.SEPARABLE,
        "words": FieldKind.WORDS,
        "word": FieldKind.WORDS,
        "spf": FieldKind.SPF_COMPLEX,
        "complex": FieldKind.SPF_COMPLEX,
        "positive": FieldKind.LOG_POS,
        "logpos": FieldKind.LOG_POS,
        "signed": FieldKind.LOG_SIGNED,
        "logsigned": FieldKind.LOG_SIGNED,
        "depth": FieldKind.OPTICAL_DEPTH,
        "tropical": FieldKind.TROPICAL_MIN,
        "display": FieldKind.DISPLAY,
    }
    if name in aliases:
        return aliases[name]
    try:
        return FieldKind(name)
    except ValueError:
        try:
            return FieldKind[name.upper()]
        except KeyError as exc:
            raise TypeCheckError(f"unknown SFR value kind {value!r}") from exc


def _representation_kind(value: RepresentationKind | str) -> RepresentationKind:
    if isinstance(value, RepresentationKind):
        return value
    name = str(value).strip().upper().replace("-", "_").split(".")[-1]
    try:
        return RepresentationKind(name)
    except ValueError as exc:
        raise ValueError(f"unknown representation kind {value!r}") from exc


_EnumT = TypeVar("_EnumT", bound=StrEnum)


def _safe_enum(value: Any, enum: type[_EnumT], default: _EnumT) -> _EnumT:
    if isinstance(value, enum):
        return value
    if value is None:
        return default
    name = str(value).strip().upper().replace("-", "_").replace(" ", "_")
    try:
        return enum(name)
    except ValueError:
        try:
            return enum[name]
        except KeyError:
            return default


def _normalise_op(operation: str) -> str:
    name = str(operation).strip().lower().replace("-", "_")
    aliases = {
        "constant": "v02_field_constant",
        "const": "v02_field_constant",
        "asset": "v02_field_asset",
        "reference": "v02_field_reference",
        "add": "v02_logadd_pos",
        "mul": "v02_log_mul",
        "fft_forward": "v02_fft_forward",
        "fft_inverse": "v02_ifft",
        "ifft": "v02_ifft",
        "hermitian": "v02_hermitian_synthesis",
        "materialize": "v02_materialize",
    }
    return aliases.get(name, name)


_KIND_DEFAULTS: dict[FieldKind, tuple[Domain, Semantic, Coherence]] = {
    FieldKind.SPF_COMPLEX: (Domain.CUSTOM, Semantic.COMPLEX_OPTICAL_AMPLITUDE, Coherence.COHERENT_COMPLEX),
    FieldKind.WORDS: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE),
    FieldKind.RADIANCE: (Domain.CUSTOM, Semantic.RADIANCE, Coherence.INCOHERENT_POSITIVE),
    FieldKind.INCOHERENT_RADIANCE: (Domain.CUSTOM, Semantic.RADIANCE, Coherence.INCOHERENT_POSITIVE),
    FieldKind.LOG_POS: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.INCOHERENT_POSITIVE),
    FieldKind.LOG_SIGNED: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.SIGNED_REAL),
    FieldKind.PHASE: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.PHASE_ONLY),
    FieldKind.OPTICAL_DEPTH: (Domain.CUSTOM, Semantic.OPTICAL_DEPTH, Coherence.NOT_APPLICABLE),
    FieldKind.TROPICAL_MIN: (Domain.CUSTOM, Semantic.DEPTH, Coherence.NOT_APPLICABLE),
    FieldKind.TROPICAL_MAX: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE),
    FieldKind.TRANSPORT: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE),
    FieldKind.DENSE: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE),
    FieldKind.DISPLAY: (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE),
}


_OP_OUTPUT: dict[str, FieldKind] = {
    "v02_field_constant": FieldKind.ANY,
    "v02_field_asset": FieldKind.ANY,
    "v02_field_reference": FieldKind.ANY,
    "v02_jet2d": FieldKind.JET2D,
    "v02_jet_add": FieldKind.JET2D,
    "v02_jet_transform": FieldKind.JET2D,
    "v02_jet_materialize": FieldKind.DENSE,
    "v02_atom_create": FieldKind.ATOM,
    "v02_atom_product": FieldKind.ATOM,
    "v02_atom_mul": FieldKind.ATOM,
    "v02_atom_div": FieldKind.ATOM,
    "v02_atom_affine": FieldKind.ATOM,
    "v02_atom_fourier": FieldKind.ATOM,
    "v02_atom_inverse_fourier": FieldKind.ATOM,
    "v02_atom_convolve": FieldKind.ATOM,
    "v02_atom_to_jet": FieldKind.JET2D,
    "v02_mixture": FieldKind.ATOM_MIXTURE,
    "v02_atom_mixture": FieldKind.ATOM_MIXTURE,
    "v02_mixture_add": FieldKind.ATOM_MIXTURE,
    "v02_mixture_concat": FieldKind.ATOM_MIXTURE,
    "v02_mixture_prune": FieldKind.ATOM_MIXTURE,
    "v02_mixture_cull": FieldKind.ATOM_MIXTURE,
    "v02_mixture_merge_equal": FieldKind.ATOM_MIXTURE,
    "v02_mixture_factor": FieldKind.MIXTURE_FACTOR,
    "v02_mixture_to_jets": FieldKind.ANY,
    "v02_mixture_eval": FieldKind.DENSE,
    "v02_mixture_materialize": FieldKind.DENSE,
    "v02_mixture_lower": FieldKind.DENSE,
    "v02_separable": FieldKind.SEPARABLE,
    "v02_separable_field": FieldKind.SEPARABLE,
    "v02_separable_eval": FieldKind.REDUCTION,
    "v02_separable_materialize": FieldKind.DENSE,
    "v02_hermitian_synthesis": FieldKind.HERMITIAN,
    "v02_hermitian_synth": FieldKind.HERMITIAN,
    "v02_transport_identity": FieldKind.TRANSPORT,
    "v02_transport": FieldKind.TRANSPORT,
    "v02_transport_compose": FieldKind.TRANSPORT,
    "v02_transport_reduce": FieldKind.TRANSPORT,
    "v02_transport_apply": FieldKind.ANY,
    "v02_logadd_pos": FieldKind.LOG_POS,
    "v02_logsub_pos": FieldKind.LOG_POS,
    "v02_logadd_signed": FieldKind.LOG_SIGNED,
    "v02_log_mul": FieldKind.LOG_POS,
    "v02_log_div": FieldKind.LOG_POS,
    "v02_phase_add": FieldKind.PHASE,
    "v02_phase_sub": FieldKind.PHASE,
    "v02_phase_conj": FieldKind.PHASE,
    "v02_phase_compose": FieldKind.PHASE,
    "v02_complex_mul": FieldKind.SPF_COMPLEX,
    "v02_complex_div": FieldKind.SPF_COMPLEX,
    "v02_complex_pair_add": FieldKind.COMPLEX_PAIR,
    "v02_phase_shift": FieldKind.SPF_COMPLEX,
    "v02_promote_complex": FieldKind.SPF_COMPLEX,
    "v02_phase_magnitude_to_complex": FieldKind.SPF_COMPLEX,
    "v02_abs2": FieldKind.LOG_POS,
    "v02_fft": FieldKind.WORDS,
    "v02_fft_forward": FieldKind.WORDS,
    "v02_ifft": FieldKind.WORDS,
    "v02_fft_inverse": FieldKind.WORDS,
    "v02_tone": FieldKind.LOG_POS,
    "v02_display": FieldKind.DISPLAY,
    "v02_display_convert": FieldKind.DISPLAY,
    "v02_reduce_positive": FieldKind.REDUCTION,
    "v02_reduce_signed": FieldKind.REDUCTION,
    "v02_reduce_complex": FieldKind.REDUCTION,
    "v02_reduce_optical_depth": FieldKind.REDUCTION,
    "v02_reduce_transport": FieldKind.REDUCTION,
    "v02_reduce_tropical": FieldKind.REDUCTION,
    "v02_depth_min": FieldKind.REDUCTION,
    "v02_visibility_select": FieldKind.DENSE,
    "v02_mask": FieldKind.DENSE,
    "v02_coordinate_remap": FieldKind.DENSE,
    "v02_bandlimit": FieldKind.DENSE,
    "v02_support_intersect": FieldKind.DENSE,
    "v02_tile_cull": FieldKind.DENSE,
    "v02_sort": FieldKind.DENSE,
    "v02_sparse_synthesis": FieldKind.DENSE,
    "v02_materialize": FieldKind.DENSE,
    "v02_materialize_tile": FieldKind.DENSE,
}


_EXPECTED_INPUTS: dict[str, tuple[FieldKind, ...]] = {
    "v02_jet_add": (FieldKind.JET2D, FieldKind.JET2D),
    "v02_jet_transform": (FieldKind.JET2D,),
    "v02_jet_materialize": (FieldKind.JET2D,),
    "v02_atom_mul": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_product": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_div": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_convolve": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_affine": (FieldKind.ATOM,),
    "v02_atom_fourier": (FieldKind.ATOM,),
    "v02_atom_inverse_fourier": (FieldKind.ATOM,),
    "v02_atom_integral": (FieldKind.ATOM,),
    "v02_atom_bound": (FieldKind.ATOM,),
    "v02_atom_bound_log": (FieldKind.ATOM,),
    "v02_atom_to_jet": (FieldKind.ATOM,),
    "v02_mixture_add": (FieldKind.ATOM_MIXTURE, FieldKind.ATOM_MIXTURE),
    "v02_mixture_concat": (FieldKind.ATOM_MIXTURE, FieldKind.ATOM_MIXTURE),
    "v02_mixture_prune": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_cull": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_merge_equal": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_factor": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_to_jets": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_eval": (FieldKind.ATOM_MIXTURE, FieldKind.DENSE),
    "v02_mixture_materialize": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_lower": (FieldKind.ATOM_MIXTURE,),
    "v02_separable_eval": (FieldKind.SEPARABLE, FieldKind.DENSE),
    "v02_separable_materialize": (FieldKind.SEPARABLE,),
    "v02_transport_compose": (FieldKind.TRANSPORT, FieldKind.TRANSPORT),
    "v02_transport_reduce": (FieldKind.TRANSPORT,),
    "v02_transport_apply": (FieldKind.TRANSPORT,),
    "v02_logadd_pos": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_logsub_pos": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_logadd_signed": (FieldKind.LOG_SIGNED, FieldKind.LOG_SIGNED),
    "v02_log_mul": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_log_div": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_phase_add": (FieldKind.PHASE, FieldKind.PHASE),
    "v02_phase_sub": (FieldKind.PHASE, FieldKind.PHASE),
    "v02_phase_compose": (FieldKind.PHASE, FieldKind.PHASE),
    "v02_phase_conj": (FieldKind.PHASE,),
    "v02_complex_mul": (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
    "v02_complex_div": (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
    "v02_complex_pair_add": (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
    "v02_phase_shift": (FieldKind.SPF_COMPLEX,),
    "v02_abs2": (FieldKind.SPF_COMPLEX,),
    "v02_fft": (FieldKind.WORDS,),
    "v02_fft_forward": (FieldKind.WORDS,),
    "v02_ifft": (FieldKind.WORDS,),
    "v02_fft_inverse": (FieldKind.WORDS,),
    "v02_tone": (FieldKind.LOG_POS,),
    "v02_display": (FieldKind.LOG_POS,),
    "v02_display_convert": (FieldKind.LOG_POS,),
    "v02_reduce_positive": (FieldKind.LOG_POS,),
    "v02_reduce_signed": (FieldKind.LOG_SIGNED,),
    "v02_reduce_complex": (FieldKind.SPF_COMPLEX,),
    "v02_reduce_optical_depth": (FieldKind.OPTICAL_DEPTH,),
    "v02_reduce_transport": (FieldKind.TRANSPORT,),
    "v02_reduce_tropical": (FieldKind.TROPICAL_MIN,),
    "v02_depth_min": (FieldKind.TROPICAL_MIN,),
}


_REDUCTION_OPS = frozenset(
    {
        "v02_logadd_pos",
        "v02_logsub_pos",
        "v02_logadd_signed",
        "v02_complex_pair_add",
        "v02_reduce_positive",
        "v02_reduce_signed",
        "v02_reduce_complex",
        "v02_reduce_optical_depth",
        "v02_reduce_transport",
        "v02_reduce_tropical",
        "v02_depth_min",
        "v02_transport_compose",
        "v02_transport_reduce",
        "v02_mixture_eval",
        "v02_separable_eval",
    }
)

_MATERIALIZE_OPS = frozenset(
    {
        "v02_jet_materialize",
        "v02_mixture_materialize",
        "v02_mixture_lower",
        "v02_separable_materialize",
        "v02_materialize",
        "v02_materialize_tile",
        "v02_display",
        "v02_display_convert",
    }
)

_STATIC_FOLD_OPS = frozenset(
    {
        "v02_atom_create",
        "v02_atom_mul",
        "v02_atom_product",
        "v02_atom_div",
        "v02_atom_affine",
        "v02_atom_fourier",
        "v02_atom_inverse_fourier",
        "v02_atom_convolve",
        "v02_atom_to_jet",
        "v02_atom_mixture",
        "v02_mixture",
        "v02_mixture_add",
        "v02_mixture_concat",
        "v02_mixture_merge_equal",
        "v02_phase_add",
        "v02_phase_sub",
        "v02_phase_conj",
        "v02_phase_compose",
        "v02_phase_shift",
        "v02_log_mul",
        "v02_log_div",
    }
)


def _numeric_bound(value: Any) -> float:
    if value is None:
        return math.inf
    if isinstance(value, arithmetic.GaussianChirpAtom):
        return math.inf
    if isinstance(value, arithmetic.AtomMixture):
        return math.inf
    if isinstance(value, arithmetic.SPF32Value):
        return float(abs(complex(arithmetic.decode_complex(value.word))))
    try:
        array = np.asarray(value)
        if array.size == 0:
            return 0.0
        if array.dtype.kind in "iu" and np.issubdtype(array.dtype, np.unsignedinteger):
            # Packed words are not magnitudes.  They are intentionally left
            # unknown unless their declared algebra is decoded by the caller.
            return math.inf
        result = float(np.max(np.abs(array.astype(np.complex128, copy=False))))
        return result if math.isfinite(result) else math.inf
    except (TypeError, ValueError, OverflowError):
        return math.inf


def _payload_of(value: Any) -> Any:
    if isinstance(value, Field):
        return value.representation.payload
    return value


def _adapt_spectral_texture_field(value: Field) -> Field:
    """Ingress one exact packed texture block into the typed block compiler.

    ``SpectralTexture`` is a spatial asset whose one-block ``DENSE_SPF``
    representation stores the local Fourier coefficients as SPF32 words.  A
    full multi-block asset cannot be relabelled as one periodic spectrum, so
    ``typed_block_view`` deliberately returns ``None`` for that case and the
    existing selective texture consumer remains in charge.  The adapted field
    carries a Fourier domain marker because the packed block is the spectrum,
    plus a terminal-spatial marker so the one inverse transform happens at the
    requested output boundary.
    """

    from spf.renderer.assets import SpectralTexture
    from spf.renderer.blocks import BlockView

    if not isinstance(value, Field) or isinstance(value.representation.payload, BlockView):
        return value
    texture = value.representation.payload
    if not isinstance(texture, SpectralTexture):
        return value
    view = texture.typed_block_view()
    if view is None:
        return value
    representation_metadata = {
        **dict(value.representation.metadata),
        "shape": texture.shape,
        "block_shape": texture.block_shape,
        "spectral_texture": True,
        "texture_block_ingress": "packed_spf32_spectrum_to_tensor_block_view",
        "texture_block_error_bound": float(texture.error_bound),
    }
    field_metadata = {
        **dict(value.metadata),
        "shape": texture.shape,
        "spectral_texture": True,
        "sfr_texture_block_ingress": True,
        # The texture stores a Fourier block, while the public texture API's
        # terminal remains spatial.  This is a terminal contract, not a scene
        # backend capability claim.
        "sfr_terminal_spatial": True,
    }
    value_kind = value.value_kind
    coherence = value.coherence
    if value_kind is FieldKind.DENSE:
        value_kind = FieldKind.SPF_COMPLEX
        coherence = Coherence.COHERENT_COMPLEX
    return replace(
        value,
        domain=Domain.FOURIER_2D,
        coherence=coherence,
        value_kind=value_kind,
        representation=replace(
            value.representation,
            kind=RepresentationKind.DENSE_TYPED,
            payload=view,
            byte_size=view.nbytes,
            metadata=representation_metadata,
        ),
        metadata=field_metadata,
    )


def _dimension_from_payload(value: Any, params: Mapping[str, Any] | None = None) -> int:
    if isinstance(value, (arithmetic.GaussianChirpAtom, arithmetic.SphericalGaussian)):
        return int(getattr(value, "dimension", 3))
    if isinstance(value, arithmetic.AtomMixture) and value.atoms:
        return value.atoms[0].dimension
    if isinstance(value, arithmetic.Jet2D):
        return 2
    if isinstance(value, arithmetic.SeparableField):
        return value.dimension
    # Block carriers expose their lattice through ``shape``.  Keep this
    # structural query separate from array coercion: ``np.asarray(BlockView)``
    # creates a scalar object view and silently collapses the chart dimension.
    try:
        from spf.renderer.blocks import BlockView

        if isinstance(value, BlockView):
            return len(value.shape)
    except ImportError:
        pass
    if params is not None:
        for key in ("dimension", "shape", "region"):
            raw = params.get(key)
            if raw is None:
                continue
            if key == "region" and isinstance(raw, Region):
                return raw.dimension
            try:
                array = np.asarray(raw)
                if key == "shape" and array.ndim == 1:
                    return int(array.size)
                if key == "region" and array.ndim == 1 and array.size in {2, 4}:
                    return 2 if array.size == 4 else int(array.size // 2)
            except (TypeError, ValueError):
                pass
    try:
        array = np.asarray(value)
        return max(1, int(array.ndim))
    except (TypeError, ValueError):
        return 1


def _region_value(value: Any) -> Region | None:
    if isinstance(value, Region):
        return value
    if isinstance(value, Support):
        return value.region
    if isinstance(value, Mapping):
        lower = value.get("lower", value.get("lo"))
        upper = value.get("upper", value.get("hi"))
        if lower is not None and upper is not None:
            try:
                return Region(tuple(lower), tuple(upper))
            except (TypeError, ValueError):
                return None
    try:
        array = np.asarray(value, dtype=float)
        if array.ndim == 2 and 2 in array.shape:
            if array.shape[1] == 2:
                return Region(tuple(array[:, 0]), tuple(array[:, 1]))
            return Region(tuple(array[0]), tuple(array[1]))
    except (TypeError, ValueError):
        return None
    return None


def _region_for_atom(region: Region | None, dimension: int) -> Mapping[str, tuple[float, ...]]:
    if region is None:
        return {"lo": (0.0,) * dimension, "hi": (1.0,) * dimension}
    if region.dimension != dimension:
        raise RewritePreconditionError("atom support/region dimensions do not match")
    return {"lo": region.lower, "hi": region.upper}


def _atom_or_mixture_bound(value: Any, region: Region | None) -> float:
    """Ref §16.2 culling certificate: ``sup|F|`` over the region.

    A mixture is ``sum_j psi_j + residual``, so the residual's own envelope is
    part of the bound.  A residual that certifies no envelope makes the bound
    infinite rather than understating it; before the v0.2 conformance audit the
    residual was omitted entirely (region (-1,-1)-(1,1) with residual 3.0:
    published 1.0 against a true sup of 4.0).
    """

    from spf.renderer.separable_terminal import residual_upper_bound

    try:
        if isinstance(value, arithmetic.GaussianChirpAtom):
            return float(arithmetic.atom_bound(value, _region_for_atom(region, value.dimension)))
        if isinstance(value, arithmetic.AtomMixture):
            residual = residual_upper_bound(value.residual)
            if not math.isfinite(residual):
                return math.inf
            if not value.atoms:
                return float(value.dropped_bound) + residual
            return (
                float(value.dropped_bound)
                + residual
                + math.fsum(
                    float(arithmetic.atom_bound(atom, _region_for_atom(region, atom.dimension)))
                    for atom in value.atoms
                )
            )
    except (TypeError, ValueError, np.linalg.LinAlgError, OverflowError):
        return math.inf
    return _numeric_bound(value)


_BLOCK_REDUCTIONS = frozenset({"v02_reduce_positive", "v02_reduce_signed", "v02_reduce_complex"})


def _block_reduction_bound(operation: str, sources: Sequence[SFRNode | None]) -> float | None:
    """Ref §10.4: reducing a whole block sums ``N_b`` terms, not one sample.

    ``fsum`` over the per-sample envelopes of a single block source understates
    the total by up to ``N_b``.  When the reduced value is one block carrier,
    publish its stored ``(L_max, N_active)`` block bound instead.
    """

    from spf.renderer.blocks import BlockView

    if operation not in _BLOCK_REDUCTIONS or len(sources) != 1 or sources[0] is None:
        return None
    payload = _payload_of(sources[0].params.get("value"))
    if not isinstance(payload, BlockView):
        return None
    return float(payload.summed_contribution_upper_bound_L().magnitude)


def _payload_kind(value: Any) -> FieldKind:
    value = _payload_of(value)
    # SPFBlock/TensorBlock/V3 and kernel views are already packed SPF complex
    # carriers.  Their type is part of the representation contract; treating
    # them as generic dense values makes graph constants lose their addressed
    # consumer before planning begins.
    try:
        from spf.renderer.blocks import BlockView

        if isinstance(value, BlockView):
            return FieldKind.SPF_COMPLEX
    except ImportError:
        pass
    if isinstance(value, arithmetic.Jet2D):
        return FieldKind.JET2D
    if isinstance(value, arithmetic.GaussianChirpAtom):
        return FieldKind.ATOM
    if isinstance(value, arithmetic.AtomMixture):
        return FieldKind.ATOM_MIXTURE
    if isinstance(value, arithmetic.SeparableField):
        return FieldKind.SEPARABLE
    if isinstance(value, arithmetic.Transport):
        return FieldKind.TRANSPORT
    if isinstance(value, arithmetic.LogPos):
        return FieldKind.LOG_POS
    if isinstance(value, arithmetic.LogSigned):
        return FieldKind.LOG_SIGNED
    if isinstance(value, arithmetic.Phase):
        return FieldKind.PHASE
    if isinstance(value, arithmetic.SPF32Value):
        return FieldKind.SPF_COMPLEX
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "u" and value.dtype.itemsize == 4:
            return FieldKind.WORDS
        return FieldKind.DENSE
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return FieldKind.WORDS
    return FieldKind.DENSE


def _default_meta(
    value_kind: FieldKind,
    *,
    domain: Any = None,
    semantic: Any = None,
    coherence: Any = None,
    chart: CoordinateChart | None = None,
    support: Support | None = None,
    error: ErrorContract | None = None,
    precision: PrecisionContract | None = None,
    ledger: arithmetic.ErrorLedger | None = None,
    bound: float = math.inf,
    phase_error: float = 0.0,
) -> CanonicalMetadata:
    default_domain, default_semantic, default_coherence = _KIND_DEFAULTS.get(
        value_kind, (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE)
    )
    dimension = chart.dimension if chart is not None else 1
    return CanonicalMetadata(
        value_kind=value_kind,
        domain=_safe_enum(domain, Domain, default_domain),
        semantic=_safe_enum(semantic, Semantic, default_semantic),
        coherence=_safe_enum(coherence, Coherence, default_coherence),
        chart=chart or CoordinateChart.identity(dimension),
        support=support or Support.full(),
        error=error or ErrorContract(),
        precision=precision or PrecisionContract(),
        ledger=ledger or arithmetic.ErrorLedger(),
        upper_bound=float(bound),
        phase_error=float(phase_error),
    )


@dataclass(frozen=True, slots=True)
class CanonicalMetadata:
    """Canonical metadata attached to every compiler SSA value."""

    value_kind: FieldKind
    domain: Domain = Domain.CUSTOM
    semantic: Semantic = Semantic.GENERIC_FIELD
    coherence: Coherence = Coherence.NOT_APPLICABLE
    chart: CoordinateChart = field(default_factory=lambda: CoordinateChart.identity(1))
    support: Support = field(default_factory=Support.full)
    error: ErrorContract = field(default_factory=ErrorContract)
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    ledger: arithmetic.ErrorLedger = field(default_factory=arithmetic.ErrorLedger)
    upper_bound: float = math.inf
    phase_error: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value_kind", _kind(self.value_kind))
        object.__setattr__(self, "domain", _safe_enum(self.domain, Domain, Domain.CUSTOM))
        object.__setattr__(self, "semantic", _safe_enum(self.semantic, Semantic, Semantic.GENERIC_FIELD))
        object.__setattr__(self, "coherence", _safe_enum(self.coherence, Coherence, Coherence.NOT_APPLICABLE))
        if not isinstance(self.chart, CoordinateChart):
            raise TypeError("canonical metadata chart must be CoordinateChart")
        if not isinstance(self.support, Support):
            raise TypeError("canonical metadata support must be Support")
        if not isinstance(self.error, ErrorContract):
            raise TypeError("canonical metadata error must be ErrorContract")
        if not isinstance(self.precision, PrecisionContract):
            raise TypeError("canonical metadata precision must be PrecisionContract")
        if not isinstance(self.ledger, arithmetic.ErrorLedger):
            raise TypeError("canonical metadata ledger must be ErrorLedger")
        bound = float(self.upper_bound)
        if math.isnan(bound) or bound < 0.0:
            raise ValueError("canonical upper_bound must be non-negative or infinity")
        phase_error = float(self.phase_error)
        if math.isnan(phase_error) or phase_error < 0.0:
            raise ValueError("canonical phase_error must be non-negative or infinity")
        object.__setattr__(self, "upper_bound", bound)
        object.__setattr__(self, "phase_error", phase_error)

    @property
    def dimension(self) -> int:
        return self.chart.dimension

    @property
    def error_bound(self) -> float:
        return float(self.ledger.total_bound)

    def with_ledger(self, ledger: arithmetic.ErrorLedger) -> CanonicalMetadata:
        return replace(self, ledger=ledger)


@dataclass(frozen=True, slots=True)
class SFRNode:
    """One immutable, metadata-rich SSA operation."""

    op: str
    output: str
    inputs: tuple[str, ...] = ()
    metadata: CanonicalMetadata | Mapping[str, Any] | None = None
    params: Mapping[str, Any] = field(default_factory=dict)
    input_kinds: tuple[FieldKind | str, ...] = ()
    annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        operation = _normalise_op(self.op)
        output = str(self.output).strip()
        if not output:
            raise ValueError("SFR nodes require a non-empty output")
        inputs = tuple(str(value) for value in self.inputs)
        if any(not value for value in inputs):
            raise ValueError("SFR node input names must be non-empty")
        params = dict(self.params)
        declared_value: Any
        if isinstance(self.metadata, CanonicalMetadata):
            declared_value = self.metadata.value_kind
        elif isinstance(self.metadata, Mapping):
            declared_value = self.metadata.get("value_kind", params.get("value_kind"))
        else:
            declared_value = params.get("value_kind")
        declared = _kind(declared_value)
        if declared is FieldKind.ANY:
            declared = _OP_OUTPUT.get(operation, FieldKind.ANY)
        if operation in {"v02_field_constant", "v02_field_asset", "v02_field_reference"}:
            payload = params.get("value", params.get("asset", params.get("reference")))
            if declared is FieldKind.ANY and payload is not None:
                declared = _payload_kind(payload)
        if declared is FieldKind.ANY and operation not in {"v02_extract", "v02_cast"}:
            # ANY is retained for a genuinely polymorphic external boundary;
            # it is rejected on typed edges by the compiler.
            declared = _OP_OUTPUT.get(operation, FieldKind.ANY)
        if isinstance(self.metadata, CanonicalMetadata):
            canonical = replace(self.metadata, value_kind=declared)
        else:
            raw = dict(self.metadata or {})
            chart = raw.get("chart")
            if chart is None:
                dimension = _dimension_from_payload(params.get("value"), params)
                chart = CoordinateChart.identity(dimension)
            canonical = _default_meta(
                declared,
                domain=raw.get("domain", params.get("domain")),
                semantic=raw.get("semantic", params.get("semantic")),
                coherence=raw.get("coherence", params.get("coherence")),
                chart=chart,
                support=raw.get("support", Support.full()),
                error=raw.get("error", ErrorContract()),
                precision=raw.get("precision", PrecisionContract()),
                ledger=raw.get("ledger", arithmetic.ErrorLedger()),
                bound=raw.get("upper_bound", params.get("bound", math.inf)),
                phase_error=raw.get("phase_error", 0.0),
            )
        object.__setattr__(self, "op", operation)
        object.__setattr__(self, "output", output)
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "metadata", canonical)
        object.__setattr__(self, "params", MappingProxyType(params))
        object.__setattr__(self, "input_kinds", tuple(_kind(value) for value in self.input_kinds))
        object.__setattr__(self, "annotations", MappingProxyType(dict(self.annotations)))

    @property
    def value_kind(self) -> FieldKind:
        return self.field_metadata.value_kind

    @property
    def field_metadata(self) -> CanonicalMetadata:
        return cast(CanonicalMetadata, self.metadata)

    def with_updates(self, **changes: Any) -> SFRNode:
        return replace(self, **changes)

    def to_v02(self) -> V02FieldNode:
        if self.op not in _V02_REGISTERED_OPS:
            raise CompilationError(f"SFR op {self.op!r} has no v0.2 reference lowering")
        params = dict(self.params)
        if self.op == "v02_field_constant" and "value" not in params:
            raise CompilationError(f"constant node {self.output!r} has no value")
        params.setdefault("value_kind", self.value_kind.value)
        metadata = self.field_metadata
        return V02FieldNode(
            self.op,
            self.inputs,
            self.output,
            self.value_kind,
            self.input_kinds,
            params,
            metadata.domain.value,
            metadata.semantic.value,
            metadata.coherence.value,
        )


# The operation registry is intentionally read from the reference node
# constructor once.  Keeping this set local prevents importing private v02
# registry tables while still allowing the compiler to fail closed.
_V02_REGISTERED_OPS = frozenset(
    {
        name
        for name in (
            *_OP_OUTPUT.keys(),
            "v02_atom_integral",
            "v02_atom_pixel_integral",
            "v02_atom_derivative",
            "v02_atom_hessian",
            "v02_mixture_bound",
            "v02_mixture_factor_select",
            "v02_complex_pair_select",
            "v02_extract",
            "v02_cast",
            "v02_promote_complex",
            "v02_phase_magnitude_to_complex",
            "v02_spherical_gaussian",
            "v02_spherical_gaussian_product",
            "v02_spherical_gaussian_integral",
            "v02_mixture_eval",
            "v02_mixture_materialize",
            "v02_mixture_lower",
            "v02_separable_eval",
            "v02_transport_apply",
            "v02_tone",
            "v02_display_convert",
        )
    }
)


@dataclass(frozen=True, slots=True)
class FieldGraph:
    """Immutable topologically ordered SFR graph.

    ``inputs`` may be a mapping of external names to values/``Field`` objects,
    or a sequence of names when the values are supplied at execution time.
    """

    nodes: tuple[SFRNode, ...] = ()
    inputs: Mapping[str, Any] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        raw_nodes = cast(tuple[SFRNode | V02FieldNode | Mapping[str, Any], ...], self.nodes)
        normalized = tuple(item if isinstance(item, SFRNode) else _coerce_node(item) for item in raw_nodes)
        raw_inputs = cast(Mapping[str, Any] | Sequence[str], self.inputs)
        input_map = dict(raw_inputs) if isinstance(raw_inputs, Mapping) else {str(name): None for name in raw_inputs}
        if len(input_map) != len(set(input_map)) or any(not str(name) for name in input_map):
            raise ValueError("FieldGraph inputs must be unique non-empty names")
        produced: dict[str, int] = {}
        for index, item in enumerate(normalized):
            if item.output in produced:
                raise ValueError(f"duplicate SFR output {item.output!r}")
            produced[item.output] = index
        available = set(input_map)
        for index, item in enumerate(normalized):
            for name in item.inputs:
                if name in available:
                    continue
                if name in produced and produced[name] >= index:
                    raise ValueError(f"SFR input {name!r} is a forward reference")
                # Undeclared external inputs are retained for compatibility
                # with FieldIR; execution will demand their value.
                input_map[name] = None
                available.add(name)
            available.add(item.output)
        selected = tuple(str(value) for value in self.outputs) or ((normalized[-1].output,) if normalized else ())
        if any(name not in produced for name in selected):
            raise ValueError("FieldGraph outputs must be produced by a node")
        if len(selected) != len(set(selected)):
            raise ValueError("FieldGraph outputs must be unique")
        object.__setattr__(self, "nodes", normalized)
        object.__setattr__(self, "inputs", MappingProxyType({str(k): v for k, v in input_map.items()}))
        object.__setattr__(self, "outputs", selected)

    @classmethod
    def from_ir(cls, ir: FieldIR, *, inputs: Mapping[str, Any] | None = None) -> FieldGraph:
        if not isinstance(ir, FieldIR):
            raise TypeError("FieldGraph.from_ir expects FieldIR")
        nodes = tuple(
            SFRNode(
                item.op,
                item.output,
                item.inputs,
                {
                    "value_kind": item.value_kind,
                    "domain": item.domain,
                    "semantic": item.semantic,
                    "coherence": item.coherence,
                },
                item.params,
                item.input_kinds,
            )
            for item in ir.nodes
        )
        external = dict(inputs or {})
        for name in ir.inputs:
            external.setdefault(name, None)
        return cls(nodes, external, ir.outputs)

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[SFRNode | V02FieldNode | Mapping[str, Any]],
        *,
        inputs: Mapping[str, Any] | Sequence[str] = (),
        outputs: Sequence[str] = (),
    ) -> FieldGraph:
        normalized_inputs = dict(inputs) if isinstance(inputs, Mapping) else {str(name): None for name in inputs}
        return cls(cast(tuple[SFRNode, ...], tuple(nodes)), normalized_inputs, tuple(outputs))

    @classmethod
    def from_field(cls, value: Field) -> FieldGraph:
        if not isinstance(value, Field):
            raise TypeError("FieldGraph.from_field expects Field")
        value = _adapt_spectral_texture_field(value)
        from spf.renderer.normalization import _consume_representation_operations

        value, _ = _consume_representation_operations(value)
        declared_kind = _kind(value.value_kind)
        payload_kind = _payload_kind(value.representation.payload)
        # A structured payload may be carried by a semantic value kind (for
        # example an SPF_COMPLEX field whose resident value is a Gaussian
        # atom). Lowering needs the algebra carried by the payload so atom
        # operations remain typed; the original semantic kind remains on the
        # owning Field used by planning and runtime materialization.
        value_kind = (
            payload_kind
            if payload_kind
            in {
                FieldKind.ATOM,
                FieldKind.ATOM_MIXTURE,
                FieldKind.JET2D,
                FieldKind.SEPARABLE,
                FieldKind.SPHERICAL_GAUSSIAN,
                FieldKind.TRANSPORT,
            }
            else declared_kind
        )
        nodes: list[SFRNode] = [
            SFRNode(
                "v02_field_constant",
                value.id,
                metadata=CanonicalMetadata(
                    value_kind,
                    _safe_enum(value.domain, Domain, Domain.CUSTOM),
                    _safe_enum(value.semantic, Semantic, Semantic.GENERIC_FIELD),
                    _safe_enum(value.coherence, Coherence, Coherence.NOT_APPLICABLE),
                    value.chart,
                    value.support,
                    value.error,
                    value.precision,
                    value.ledger,
                    _atom_or_mixture_bound(value.representation.payload, value.support.region),
                ),
                params={"value": value.representation.payload, "value_kind": value_kind.value},
            )
        ]
        current = value.id
        for index, operation in enumerate(value.operations):
            lowered = _field_operation_lowerings(
                operation,
                nodes[-1].value_kind,
                domain=value.domain,
                payload=value.representation.payload,
            )
            inline_operand: str | None = None
            for sub_index, (op, params, kind) in enumerate(lowered):
                output = f"{value.id}__{index:03d}_{operation.op}"
                if sub_index:
                    output = f"{output}__{sub_index:02d}"
                if op == "v02_field_constant":
                    # Inline transfer atoms are explicit SSA values. This
                    # lets the normal atom-product pass prove closure instead
                    # of hiding the transfer in a pre-baked payload.
                    output = f"{output}__operand"
                    inputs: tuple[str, ...] = ()
                    inline_operand = output
                else:
                    inputs = (current, inline_operand) if inline_operand is not None else (current,)
                if op == "v02_cast" and params.get("deferred_operation"):
                    # The VM dispatches representation-closed operations on
                    # the retained block itself.  Carry the owning field
                    # domain/chart beside the typed edge so constructing a
                    # FilteredBlockView never has to inspect or materialize
                    # the original Field object.
                    params = {
                        **params,
                        "_field_domain": _safe_enum(value.domain, Domain, Domain.CUSTOM).value,
                        "_field_chart": value.chart,
                    }
                source_meta = nodes[-1].field_metadata
                operation_error = operation.error if sub_index == len(lowered) - 1 else arithmetic.ErrorLedger()
                metadata = replace(
                    source_meta,
                    value_kind=kind,
                    ledger=source_meta.ledger.plus(operation_error),
                )
                nodes.append(SFRNode(op, output, inputs, metadata, params))
                if op != "v02_field_constant":
                    current = output
                    inline_operand = None
        return cls(tuple(nodes), {}, (current,))

    def to_ir(self) -> FieldIR:
        return FieldIR.from_nodes(
            tuple(item.to_v02() for item in self.nodes), inputs=tuple(self.inputs), outputs=self.outputs
        )


def _coerce_node(value: V02FieldNode | Mapping[str, Any]) -> SFRNode:
    if isinstance(value, V02FieldNode):
        return SFRNode(
            value.op,
            value.output,
            value.inputs,
            {
                "value_kind": value.value_kind,
                "domain": value.domain,
                "semantic": value.semantic,
                "coherence": value.coherence,
            },
            value.params,
            value.input_kinds,
        )
    if isinstance(value, Mapping):
        item = dict(value)
        if "output" not in item and "name" in item:
            item["output"] = item.pop("name")
        return SFRNode(**item)
    raise TypeError(f"unsupported SFR node type {type(value).__name__}")


def _atom_scale(exposure: float, dimension: int) -> arithmetic.GaussianChirpAtom:
    """Return a constant Gaussian atom carrying an exposure gain."""

    return arithmetic.GaussianChirpAtom.from_arrays(
        float(exposure) * math.log(2.0),
        np.zeros(dimension, dtype=np.complex128),
        np.zeros((dimension, dimension), dtype=np.complex128),
    )


def _gaussian_transfer(covariance: Any, dimension: int) -> arithmetic.GaussianChirpAtom:
    """Build ``exp(-2π² kᵀΣk)`` as an explicit Fourier-domain atom."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError("Gaussian filter covariance dimension does not match the field")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Gaussian filter covariance must be symmetric")
    if np.min(np.linalg.eigvalsh(matrix)) < -1.0e-12:
        raise ValueError("Gaussian filter covariance must be positive semidefinite")
    return arithmetic.GaussianChirpAtom.from_arrays(
        0.0j,
        np.zeros(dimension, dtype=np.complex128),
        4.0 * math.pi**2 * matrix,
    )


def _spatial_gaussian_kernel(covariance: Any, dimension: int) -> arithmetic.GaussianChirpAtom:
    """Build the normalized spatial Gaussian kernel used by convolution."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise ValueError("Gaussian filter covariance dimension does not match the field")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("Gaussian filter covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(matrix)
    if np.min(eigenvalues) <= 1.0e-12:
        raise ValueError("singular spatial Gaussian filter is not analytically closed")
    normalizer = -0.5 * dimension * math.log(2.0 * math.pi) - 0.5 * math.log(float(np.linalg.det(matrix)))
    return arithmetic.GaussianChirpAtom.from_arrays(
        normalizer,
        np.zeros(dimension, dtype=np.complex128),
        np.linalg.inv(matrix),
    )


def _deferred_identity(operation: FieldOperation, source_kind: FieldKind) -> tuple[str, dict[str, Any], FieldKind]:
    """Keep an unsupported field operation visible without changing values.

    ``v02_coordinate_remap`` transforms a point list, not a field. Mapping a
    dense field translation to it was therefore both type-incorrect and
    numerically wrong. The dense/mixture runtime owns these fallbacks; this
    identity cast records the exact deferred operation in the compiler graph.
    """

    return (
        "v02_cast",
        {
            "from_kind": source_kind.value,
            "to_kind": source_kind.value,
            "deferred_operation": operation.op,
            **dict(operation.params),
        },
        source_kind,
    )


def _field_operation_lowerings(
    operation: FieldOperation,
    source_kind: FieldKind,
    *,
    domain: Domain | str | None = None,
    payload: Any = None,
) -> tuple[tuple[str, dict[str, Any], FieldKind], ...]:
    """Lower a Field operation to explicit typed SSA nodes.

    A closed Gaussian atom operation expands to a parameter-only transfer
    constant followed by the algebraic primitive that consumes it. No sample
    layout is evaluated here. Unsupported payload families remain an
    explicitly deferred identity node and are executed exactly once by the
    representation-aware runtime.
    """

    from spf.renderer.blocks import BlockView

    # Storage and semantic kinds are independent: a positive block still
    # needs a retained header update, not scalar LogPos coercion.
    if isinstance(payload, BlockView):
        return (_deferred_identity(operation, source_kind),)
    params = dict(operation.params)
    domain_value = _safe_enum(domain, Domain, Domain.CUSTOM)
    dimension = (
        int(payload.dimension)
        if payload is not None and hasattr(payload, "dimension")
        else len(tuple(params.get("offset", ())))
    )
    dimension = max(1, dimension)

    if source_kind is FieldKind.ATOM:
        if operation.op == "translate":
            offset = np.asarray(params.get("offset", ()), dtype=np.float64).reshape(-1)
            if offset.shape != (dimension,):
                raise ValueError("translation offset dimension does not match the atom")
            if domain_value in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
                return (
                    (
                        "v02_atom_affine",
                        {"A": np.eye(dimension), "b": -offset, "source_operation": operation.op},
                        FieldKind.ATOM,
                    ),
                )
            if domain_value in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
                ramp = arithmetic.GaussianChirpAtom.from_arrays(
                    0.0j,
                    -1j * 2.0 * math.pi * offset,
                    np.zeros((dimension, dimension), dtype=np.complex128),
                )
                return (
                    (
                        "v02_field_constant",
                        {"value": ramp, "value_kind": FieldKind.ATOM.value, "source_operation": operation.op},
                        FieldKind.ATOM,
                    ),
                    ("v02_atom_mul", {"source_operation": operation.op}, FieldKind.ATOM),
                )
            return (_deferred_identity(operation, source_kind),)
        if operation.op == "gaussian_filter":
            covariance = params.get("covariance")
            if domain_value in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
                transfer = _gaussian_transfer(covariance, dimension)
                primitive = "v02_atom_mul"
            elif domain_value in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
                if np.min(np.linalg.eigvalsh(np.asarray(covariance))) <= 1e-12:
                    transfer = _gaussian_transfer(covariance, dimension)
                    return (
                        ("v02_atom_fourier", {"source_operation": operation.op}, FieldKind.ATOM),
                        (
                            "v02_field_constant",
                            {"value": transfer, "value_kind": FieldKind.ATOM.value, "source_operation": operation.op},
                            FieldKind.ATOM,
                        ),
                        ("v02_atom_mul", {"source_operation": operation.op}, FieldKind.ATOM),
                        ("v02_atom_inverse_fourier", {"source_operation": operation.op}, FieldKind.ATOM),
                    )
                transfer = _spatial_gaussian_kernel(covariance, dimension)
                primitive = "v02_atom_convolve"
            else:
                return (_deferred_identity(operation, source_kind),)
            return (
                (
                    "v02_field_constant",
                    {"value": transfer, "value_kind": FieldKind.ATOM.value, "source_operation": operation.op},
                    FieldKind.ATOM,
                ),
                (primitive, {"source_operation": operation.op}, FieldKind.ATOM),
            )
        if operation.op == "exposure":
            exposure = float(params.get("ev", params.get("exposure", 0.0)))
            if not math.isfinite(exposure):
                raise ValueError("exposure must be finite")
            return (
                (
                    "v02_field_constant",
                    {
                        "value": _atom_scale(exposure, dimension),
                        "value_kind": FieldKind.ATOM.value,
                        "source_operation": operation.op,
                    },
                    FieldKind.ATOM,
                ),
                ("v02_atom_mul", {"source_operation": operation.op}, FieldKind.ATOM),
            )
        if operation.op == "lens_phase":
            focal_value = params.get("focal_length")
            wavelength_value = params.get("wavelength")
            if focal_value is None or wavelength_value is None:
                raise ValueError("lens phase requires focal_length and wavelength")
            focal = float(focal_value)
            wavelength = float(wavelength_value)
            if focal == 0.0 or wavelength <= 0.0 or not math.isfinite(focal) or not math.isfinite(wavelength):
                raise ValueError("lens phase requires finite focal length and positive wavelength")
            lens = arithmetic.GaussianChirpAtom.from_arrays(
                0.0j,
                np.zeros(dimension, dtype=np.complex128),
                1j * (2.0 * math.pi / (wavelength * focal)) * np.eye(dimension),
            )
            return (
                (
                    "v02_field_constant",
                    {"value": lens, "value_kind": FieldKind.ATOM.value, "source_operation": operation.op},
                    FieldKind.ATOM,
                ),
                ("v02_atom_mul", {"source_operation": operation.op}, FieldKind.ATOM),
            )

    if operation.op in {"abs2", "intensity", "magnitude_square"}:
        # The representation consumer owns the coherent reduction followed by
        # demotion; applying a scalar abs2 to an atom/mixture header is invalid.
        return (_deferred_identity(operation, source_kind),)
    if operation.op == "exposure":
        if source_kind in {FieldKind.LOG_POS, FieldKind.RADIANCE, FieldKind.INCOHERENT_RADIANCE}:
            return (("v02_tone", {"mode": "exposure", "exposure": params.get("ev", 0.0)}, source_kind),)
        return (_deferred_identity(operation, source_kind),)
    if operation.op == "lens_phase":
        # For a dense coherent field the physical phase parameters remain
        # deferred; a scalar ``dT`` phase shift is closed in the packed algebra.
        delta = params.get("dT", params.get("phase", 0))
        if source_kind is FieldKind.SPF_COMPLEX and "dT" in params:
            return (("v02_phase_shift", {"dT": delta, **params}, FieldKind.SPF_COMPLEX),)
        return (_deferred_identity(operation, source_kind),)
    if operation.op == "bandlimit":
        cutoff = float(params.get("cutoff", 0.0))
        return (("v02_bandlimit", {"band": (-int(abs(cutoff)), int(abs(cutoff))), **params}, source_kind),)
    if operation.op in {
        "mask",
        "translate",
        "gaussian_filter",
        "glare",
        "fresnel",
        "spectral_lens",
        "phase",
        "phase_shift",
        "lod",
        # Design §17.2 rotation and §17.3 general affine remap coordinates, so
        # they are not diagonal nodes; the representation-aware runtime owns
        # them (exact on atoms/jets, an index permutation on a packed plane, a
        # counted reference resample otherwise).
        "rotate",
        "affine",
    }:
        return (_deferred_identity(operation, source_kind),)
    return ((_normalise_op(operation.op), params, source_kind),)


def _field_operation_lowering(
    operation: FieldOperation, source_kind: FieldKind
) -> tuple[str, dict[str, Any], FieldKind]:
    """Compatibility wrapper for callers that expect a single node."""

    lowered = _field_operation_lowerings(operation, source_kind)
    if len(lowered) != 1:
        raise RewritePreconditionError(f"{operation.op} expands to multiple typed nodes")
    return lowered[0]


@dataclass(frozen=True, slots=True)
class CompilerOptions:
    backend: str | SFRBackend = "reference"
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    optimize: bool = True
    strict: bool = True
    strict_hermitian: bool = True
    allow_implicit_hermitian_endpoints: bool = False
    enable_separable_rewrites: bool = True
    enable_pruning: bool = True
    allow_fallback: bool = False
    fallback_backend: str = "reference"
    layout: Any = None
    addressed_request: bool = False
    consumer_representations: tuple[RepresentationKind, ...] | None = None
    regions: tuple[Region, ...] = ()
    retain_intermediates: bool = False
    max_pass_iterations: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend", self.backend)
        if not isinstance(self.precision, PrecisionContract):
            object.__setattr__(self, "precision", PrecisionContract.for_mode(self.precision))
        if not isinstance(self.error_budget, ErrorBudget):
            value = float(self.error_budget)
            object.__setattr__(self, "error_budget", ErrorBudget(absolute=value))
        object.__setattr__(self, "regions", tuple(self.regions))


@dataclass(frozen=True, slots=True)
class MemorySchedule:
    intervals: Mapping[str, tuple[int, int]]
    slots: Mapping[str, tuple[int, int]]
    release_after: Mapping[int, tuple[str, ...]]
    peak_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "intervals", MappingProxyType(dict(self.intervals)))
        object.__setattr__(self, "slots", MappingProxyType(dict(self.slots)))
        object.__setattr__(
            self, "release_after", MappingProxyType({int(k): tuple(v) for k, v in self.release_after.items()})
        )

    @property
    def peak_memory(self) -> int:
        return self.peak_bytes


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    """Executable compiled graph plus its proof/report artifacts."""

    graph: FieldGraph
    original_graph: FieldGraph
    backend: str
    report: RenderReport
    passes: tuple[CompilerPassRecord, ...] = ()
    memory_schedule: MemorySchedule = field(default_factory=lambda: MemorySchedule({}, {}, {}, 0))
    aliases: Mapping[str, str] = field(default_factory=dict)
    materialization_plan: Any = None
    _backend_object: SFRBackend | None = field(default=None, repr=False, compare=False)
    _source_field: Field | None = field(default=None, repr=False, compare=False)
    _execution_ir: FieldIR = field(init=False, repr=False, compare=False)
    _instructions: tuple[Any, ...] = field(default=(), init=False, repr=False, compare=False)
    _addresses: Mapping[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)
    _slot_count: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "passes", tuple(self.passes))
        object.__setattr__(self, "aliases", MappingProxyType(dict(self.aliases)))
        ir = self.graph.to_ir()
        steps = ir.to_execution_plan().steps
        addresses = {name: slot for name, (slot, _) in self.memory_schedule.slots.items()}
        next_slot = max(addresses.values(), default=-1) + 1
        for name in self.graph.inputs:
            if name not in addresses:
                addresses[name] = next_slot
                next_slot += 1
        instructions = tuple(
            (
                node.op,
                node.output,
                tuple(addresses[name] for name in node.inputs),
                addresses[node.output],
                step.params,
                tuple(
                    addresses[name]
                    for name in self.memory_schedule.release_after.get(index, ())
                    if name not in self.graph.outputs
                ),
            )
            for index, (node, step) in enumerate(zip(ir.nodes, steps, strict=True))
        )
        object.__setattr__(self, "_execution_ir", ir)
        object.__setattr__(self, "_instructions", instructions)
        object.__setattr__(self, "_addresses", MappingProxyType(addresses))
        object.__setattr__(self, "_slot_count", next_slot)

    @property
    def ir(self) -> FieldIR:
        return self._execution_ir

    @property
    def nodes(self) -> tuple[SFRNode, ...]:
        return self.graph.nodes

    @property
    def outputs(self) -> tuple[str, ...]:
        return self.graph.outputs

    @property
    def fallbacks(self) -> tuple[str, ...]:
        return self.report.fallbacks

    def _backend(self) -> SFRBackend:
        return self._backend_object or get_backend(self.backend)

    def execute(self, values: Mapping[str, Any] | None = None, *, return_environment: bool = True) -> Any:
        if (
            self._source_field is not None
            and self.report.diagnostics.get("materialization_execution") == "field_representation_runtime"
        ):
            raise CompilationError(
                "this Field has deferred representation operations; use program.materialize(request=...) "
                "to execute the named whole-field consumer"
            )
        backend = self._backend()
        backend.require_available()
        supplied = dict(values or {})
        for name, default in self.graph.inputs.items():
            if name not in supplied and default is not None:
                supplied[name] = _payload_of(default)
        for name, value in tuple(supplied.items()):
            supplied[name] = _payload_of(value)
        self._execution_ir.validate_values(supplied)
        slots: list[Any] = [None] * self._slot_count
        for name, value in supplied.items():
            if name in self._addresses:
                slots[self._addresses[name]] = value
        environment = dict(supplied) if return_environment else None
        from spf.renderer.v02 import _validate_execution_output

        try:
            for op, name, inputs, output, params, releases in self._instructions:
                slots[output] = backend.execute_operation(op, tuple(slots[index] for index in inputs), params)
                _validate_execution_output(op, slots[output], params)
                if environment is not None:
                    environment[name] = slots[output]
                for index in releases:
                    slots[index] = None
        except (TypeError, ValueError, IndexError) as exc:
            raise CompilationError(f"compiled SFR program execution failed on {self.backend!r}: {exc}") from exc
        if environment is not None:
            for alias, source in self.aliases.items():
                if source in environment:
                    environment[alias] = environment[source]
            return environment
        selected = {name: slots[self._addresses[name]] for name in self.outputs}
        return next(iter(selected.values())) if len(selected) == 1 else selected

    def render(self, values: Mapping[str, Any] | None = None) -> Any:
        if values is not None and all(name in values for name in self.outputs):
            selected = {name: values[name] for name in self.outputs}
            return next(iter(selected.values())) if len(selected) == 1 else selected
        return self.execute(values, return_environment=False)

    def materialize(self, field_id: str | None = None, request: Any = None, **options: Any) -> Any:
        if self.materialization_plan is None:
            raise CompilationError("program has no materialization plan")
        if self._source_field is not None:
            if field_id is not None and field_id != self._source_field.id and field_id not in self.outputs:
                raise CompilationError(f"compiled program has no source field {field_id!r}")
            route, _ = _field_materialization_route(self._source_field, self.graph)
            if route != "compiled_ssa":
                return self.materialization_plan.materialize(self._source_field, request, **options)
            from spf.renderer.normalization import inherit_representation_error, normalize_field
            from spf.renderer.planner import MaterializationPlanner

            # Normalization supplies support/error custody only. The sampled
            # value must come from the executed program, never the source plan's
            # preselected payload. Re-plan against that actual output carrier.
            normalized = normalize_field(self._source_field).field
            value = self.execute(return_environment=False)
            if isinstance(value, TypedField):
                value = value.value
            residual = normalized.representation.residual
            canonical_payload = normalized.representation.payload
            if self._source_field.representation.residual is not None and isinstance(
                canonical_payload, arithmetic.AtomMixture
            ):
                # The normalizer folds the separately stored residual into a
                # mixture. Keep the executed analytic header, but attach the
                # residual after exactly the same closed operation prefix.
                if isinstance(value, arithmetic.AtomMixture):
                    value = arithmetic.AtomMixture(
                        value.atoms, canonical_payload.residual,
                        max(value.dropped_bound, canonical_payload.dropped_bound),
                    )
                else:
                    residual = canonical_payload.residual
            representation_kind = {
                FieldKind.ATOM: RepresentationKind.GAUSSIAN_CHIRP_ATOM,
                FieldKind.ATOM_MIXTURE: RepresentationKind.ATOM_MIXTURE,
                FieldKind.JET2D: RepresentationKind.QUADRATIC_JET,
                FieldKind.SEPARABLE: RepresentationKind.SEPARABLE,
            }.get(_payload_kind(value), normalized.representation.kind)
            output = replace(
                normalized,
                representation=replace(
                    normalized.representation,
                    kind=representation_kind,
                    payload=value,
                    residual=residual,
                    byte_size=None,
                    consumed_operation_indices=(),
                ),
                operations=(),
            )
            output_node = next(node for node in self.graph.nodes if node.output == self.outputs[-1])
            output = replace(
                output,
                representation=inherit_representation_error(output.representation, output_node.field_metadata.ledger),
            )
            return MaterializationPlanner(backend=self._backend()).materialize(output, request, **options)
        return self.materialization_plan.materialize_from_program(
            self, field_id or self.outputs[-1], request, **options
        )


def _field_materialization_route(source: Field, graph: FieldGraph) -> tuple[str, str]:
    """Name the actual terminal consumer; deferred identities are not execution."""
    from spf.renderer.block_filter import FILTER_OPS
    from spf.renderer.blocks import BlockView
    from spf.renderer.normalization import normalize_field

    block = isinstance(source.representation.payload, BlockView)
    for node in graph.nodes:
        operation = node.params.get("deferred_operation")
        if operation and not (block and operation in FILTER_OPS | {"mask"}):
            return "field_representation_runtime", f"deferred {operation} is owned by the representation consumer"
    if normalize_field(source).field.operations:
        return "field_representation_runtime", "an open operation suffix requires the representation consumer"
    return "compiled_ssa", "materialization consumes the executed output carrier"


@dataclass
class _CompileContext:
    options: CompilerOptions
    backend: SFRBackend
    source_fields: tuple[Field, ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    rewrites: list[str] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)
    pass_records: list[CompilerPassRecord] = field(default_factory=list)

    def record(
        self, name: PassName | str, before: int, after: int, rewrites: Iterable[str] = (), notes: Iterable[str] = ()
    ) -> None:
        self.pass_records.append(CompilerPassRecord(str(name), int(before), int(after), tuple(rewrites), tuple(notes)))
        self.rewrites.extend(str(value) for value in rewrites)
        self.notes.extend(str(value) for value in notes)


def _coherence_compatible(actual: Coherence, expected: Coherence) -> bool:
    return actual is expected


def _kind_compatible(actual: FieldKind, expected: FieldKind) -> bool:
    if expected is FieldKind.ANY or actual is expected:
        return True
    if actual is FieldKind.ANY:
        return False
    if expected is FieldKind.LOG_POS and actual in {FieldKind.RADIANCE, FieldKind.INCOHERENT_RADIANCE}:
        return True
    if expected is FieldKind.WORDS and actual is FieldKind.SPF_COMPLEX:
        return True
    if expected is FieldKind.SPF_COMPLEX and actual is FieldKind.WORDS:
        return True
    return bool(expected is FieldKind.DENSE and actual in {FieldKind.RADIANCE, FieldKind.INCOHERENT_RADIANCE})


def _canonicalize(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    produced: dict[str, SFRNode] = {}
    external: dict[str, CanonicalMetadata] = {}
    for name, value in graph.inputs.items():
        if value is None:
            continue
        if isinstance(value, Field):
            value_kind = _kind(value.value_kind)
            external[name] = CanonicalMetadata(
                value_kind,
                _safe_enum(value.domain, Domain, Domain.CUSTOM),
                _safe_enum(value.semantic, Semantic, Semantic.GENERIC_FIELD),
                _safe_enum(value.coherence, Coherence, Coherence.NOT_APPLICABLE),
                value.chart,
                value.support,
                value.error,
                value.precision,
                value.ledger,
                _atom_or_mixture_bound(value.representation.payload, value.support.region),
            )
        else:
            external[name] = _default_meta(
                _payload_kind(value), chart=CoordinateChart.identity(_dimension_from_payload(value))
            )
    normalized: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        operation = _normalise_op(item.op)
        if operation not in _V02_REGISTERED_OPS:
            raise CompilationError(f"unregistered SFR operation {item.op!r}")
        params = dict(item.params)
        explicit_kind = item.value_kind
        kind = explicit_kind if explicit_kind is not FieldKind.ANY else _OP_OUTPUT.get(operation, FieldKind.ANY)
        if operation in {"v02_field_constant", "v02_field_asset", "v02_field_reference"}:
            value = params.get("value", params.get("asset", params.get("reference")))
            if kind is FieldKind.ANY and value is not None:
                kind = _payload_kind(value)
            if (
                value is not None
                and isinstance(value, (int, np.integer))
                and not isinstance(value, (bool, np.bool_))
                and kind not in {FieldKind.WORDS, FieldKind.SPF_COMPLEX}
                and not params.get("typed_proof")
            ):
                raise TypeCheckError(
                    f"raw integer constant {item.output!r} requires WORDS/SPF_COMPLEX or params['typed_proof']"
                )
        if operation in {"v02_phase_shift", "v02_phase_add", "v02_phase_sub", "v02_phase_conj"} and (
            any(
                (produced[name].field_metadata if name in produced else external.get(name)) is not None
                and (produced[name].field_metadata if name in produced else external[name]).coherence
                is not Coherence.PHASE_ONLY
                and operation != "v02_phase_shift"
                for name in item.inputs
                if name in produced or name in external
            )
            and operation != "v02_phase_shift"
        ):
            raise TypeCheckError(f"{operation} requires PHASE_ONLY inputs")
        expected = item.input_kinds or _EXPECTED_INPUTS.get(operation, ())
        if (
            expected
            and len(expected) > len(item.inputs)
            and operation
            not in {
                "v02_reduce_positive",
                "v02_reduce_signed",
                "v02_reduce_complex",
                "v02_reduce_optical_depth",
                "v02_reduce_transport",
                "v02_reduce_tropical",
            }
        ):
            raise TypeCheckError(f"{operation} has too few inputs for its typed contract")
        for position, input_name in enumerate(item.inputs):
            if position >= len(expected):
                continue
            source = produced.get(input_name)
            source_meta = source.field_metadata if source is not None else external.get(input_name)
            if source_meta is None:
                continue
            expected_kind = _kind(expected[position])
            if not _kind_compatible(source_meta.value_kind, expected_kind):
                raise TypeCheckError(
                    f"{operation} input {position} ({input_name!r}) requires {expected_kind.value}, "
                    f"received {source_meta.value_kind.value}"
                )
            if expected_kind is FieldKind.SPF_COMPLEX and source_meta.coherence is not Coherence.COHERENT_COMPLEX:
                raise TypeCheckError(
                    f"{operation} input {position} crosses a coherence boundary without an explicit cast"
                )
        if (
            operation == "v02_hermitian_synthesis"
            and context.options.strict_hermitian
            and (
                "self_conjugate" not in params
                and "endpoint_metadata" not in params
                and not context.options.allow_implicit_hermitian_endpoints
            )
        ):
            raise RewritePreconditionError("Hermitian lowering requires explicit self_conjugate/endpoint_metadata")
        source_meta = next((produced[name].field_metadata for name in item.inputs if name in produced), None)
        if source_meta is None:
            source_meta = next((external[name] for name in item.inputs if name in external), None)
        dimension = _dimension_from_payload(params.get("value"), params)
        chart = (
            item.field_metadata.chart
            if item.field_metadata.chart.dimension
            else (source_meta.chart if source_meta is not None else CoordinateChart.identity(dimension))
        )
        if (
            source_meta is not None
            and item.field_metadata.chart == CoordinateChart.identity(1)
            and source_meta.chart.dimension != 1
        ):
            chart = source_meta.chart
        default_domain, default_semantic, default_coherence = _KIND_DEFAULTS.get(
            kind, (Domain.CUSTOM, Semantic.GENERIC_FIELD, Coherence.NOT_APPLICABLE)
        )
        metadata = replace(
            item.field_metadata,
            value_kind=kind,
            domain=item.field_metadata.domain
            if item.field_metadata.domain is not Domain.CUSTOM
            else (source_meta.domain if source_meta else default_domain),
            semantic=item.field_metadata.semantic
            if item.field_metadata.semantic is not Semantic.GENERIC_FIELD
            else (source_meta.semantic if source_meta else default_semantic),
            coherence=item.field_metadata.coherence
            if item.field_metadata.coherence is not Coherence.NOT_APPLICABLE
            else (source_meta.coherence if source_meta and kind is source_meta.value_kind else default_coherence),
            chart=chart,
            precision=context.options.precision,
        )
        if kind is FieldKind.PHASE and metadata.coherence is not Coherence.PHASE_ONLY:
            metadata = replace(metadata, coherence=Coherence.PHASE_ONLY)
        if kind in {FieldKind.SPF_COMPLEX} and metadata.coherence is not Coherence.COHERENT_COMPLEX:
            if context.options.strict:
                raise TypeCheckError(f"{operation} output requires COHERENT_COMPLEX metadata")
            metadata = replace(metadata, coherence=Coherence.COHERENT_COMPLEX)
        node = SFRNode(operation, item.output, item.inputs, metadata, params, expected, item.annotations)
        normalized.append(node)
        produced[node.output] = node
        rewrites.append(f"{item.output}:canonical_metadata")
    context.record(PassName.TYPE_CHECK_CANONICALIZE, len(graph.nodes), len(normalized), rewrites=rewrites)
    return FieldGraph(tuple(normalized), graph.inputs, graph.outputs)


def _propagate(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    produced: dict[str, SFRNode] = {}
    result: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        sources = [produced.get(name) for name in item.inputs]
        source_meta = next((source.field_metadata for source in sources if source is not None), None)
        metadata = item.field_metadata
        if source_meta is not None:
            metadata = replace(
                metadata,
                chart=source_meta.chart,
                support=source_meta.support,
                error=source_meta.error,
                precision=source_meta.precision,
                ledger=source_meta.ledger,
                upper_bound=source_meta.upper_bound,
                phase_error=source_meta.phase_error,
            )
        params = dict(item.params)
        operation = item.op
        region = _region_value(params.get("region")) or metadata.support.region
        source_bounds = [source.field_metadata.upper_bound for source in sources if source is not None]
        bound = metadata.upper_bound
        if operation in {"v02_field_constant", "v02_field_asset", "v02_field_reference"}:
            value = params.get("value", params.get("asset", params.get("reference")))
            bound = (
                _atom_or_mixture_bound(value, region)
                if isinstance(value, (arithmetic.GaussianChirpAtom, arithmetic.AtomMixture))
                else _numeric_bound(value)
            )
            if (
                isinstance(value, np.ndarray)
                and value.dtype.kind == "u"
                and item.value_kind in {FieldKind.WORDS, FieldKind.SPF_COMPLEX}
            ):
                bound = math.inf
        elif (
            operation in {"v02_atom_mul", "v02_atom_product", "v02_complex_mul", "v02_log_mul", "v02_atom_convolve"}
            and len(source_bounds) >= 2
        ):
            bound = math.prod(source_bounds) if all(math.isfinite(v) for v in source_bounds) else math.inf
        elif operation in {
            "v02_logadd_pos",
            "v02_logadd_signed",
            "v02_mixture_add",
            "v02_mixture_concat",
            "v02_mixture_prune",
            "v02_reduce_positive",
            "v02_reduce_signed",
            "v02_reduce_complex",
        }:
            block_total = _block_reduction_bound(operation, sources)
            if block_total is not None:
                bound = block_total
            else:
                bound = (
                    math.fsum(source_bounds)
                    if source_bounds and all(math.isfinite(v) for v in source_bounds)
                    else bound
                )
        elif operation in {"v02_abs2"} and source_bounds and math.isfinite(source_bounds[0]):
            bound = source_bounds[0] ** 2
        elif operation in {
            "v02_phase_shift",
            "v02_phase_add",
            "v02_phase_sub",
            "v02_phase_conj",
            "v02_atom_affine",
            "v02_jet_transform",
            "v02_coordinate_remap",
        }:
            bound = source_bounds[0] if source_bounds else bound
        elif operation in {"v02_fft", "v02_fft_forward", "v02_ifft", "v02_fft_inverse"} and source_bounds:
            bound = source_bounds[0]
            current_domain = metadata.domain
            if current_domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
                domain = {
                    Domain.FOURIER_1D: Domain.SPATIAL_1D,
                    Domain.FOURIER_2D: Domain.SPATIAL_2D,
                    Domain.FOURIER_3D: Domain.SPATIAL_3D,
                }[current_domain]
            else:
                domain = {
                    Domain.SPATIAL_1D: Domain.FOURIER_1D,
                    Domain.SPATIAL_2D: Domain.FOURIER_2D,
                    Domain.SPATIAL_3D: Domain.FOURIER_3D,
                }.get(current_domain, current_domain)
            metadata = replace(metadata, domain=domain)
        elif source_bounds:
            bound = source_bounds[0]
        support = metadata.support
        if operation == "v02_support_intersect":
            supports = [source.field_metadata.support for source in sources if source is not None]
            support = supports[0] if supports else support
            for other in supports[1:]:
                support = support.intersection(other)
        elif operation in {"v02_mask", "v02_visibility_select", "v02_tile_cull"} and source_meta is not None:
            support = source_meta.support
        ledger = metadata.ledger
        extra_error = params.get("error_ledger", params.get("error"))
        if isinstance(extra_error, arithmetic.ErrorLedger):
            ledger = ledger.plus(extra_error)
        metadata = replace(metadata, support=support, upper_bound=bound, ledger=ledger)
        if metadata.error.phase_required and operation in {"v02_reduce_tropical", "v02_depth_min"}:
            raise TypeCheckError("tropical/depth reduction cannot consume a phase-required field")
        updated = replace(item, metadata=metadata)
        result.append(updated)
        produced[updated.output] = updated
        rewrites.append(f"{updated.output}:propagated")
    context.record(PassName.PROPAGATE, len(graph.nodes), len(result), rewrites=rewrites)
    return FieldGraph(tuple(result), graph.inputs, graph.outputs)


def _compose_affine(
    first: Mapping[str, Any], second: Mapping[str, Any], *, atom_semantics: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    A1 = np.asarray(first.get("A"), dtype=np.float64)
    b1 = np.asarray(first.get("b", np.zeros(A1.shape[0])), dtype=np.float64).reshape(-1)
    A2 = np.asarray(second.get("A"), dtype=np.float64)
    b2 = np.asarray(second.get("b", np.zeros(A2.shape[0])), dtype=np.float64).reshape(-1)
    if (
        A1.ndim != 2
        or A2.ndim != 2
        or A1.shape[0] != A1.shape[1]
        or A2.shape[0] != A2.shape[1]
        or A1.shape != A2.shape
        or b1.shape != (A1.shape[0],)
        or b2.shape != (A1.shape[0],)
    ):
        raise RewritePreconditionError("affine fusion requires matching square matrices and shifts")
    if atom_semantics:
        return A1 @ A2, A1 @ b2 + b1
    return A2 @ A1, A2 @ b1 + b2


def _fuse_affine(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    nodes = list(graph.nodes)
    uses = _use_counts(nodes)
    rewrites: list[str] = []
    changed = True
    while changed:
        changed = False
        for index in range(1, len(nodes)):
            first = nodes[index - 1]
            second = nodes[index]
            if first.output not in second.inputs or uses[first.output] != 1:
                continue
            affine_ops = {"v02_atom_affine", "v02_jet_transform", "v02_coordinate_remap"}
            if first.op not in affine_ops or second.op != first.op:
                continue
            atom_semantics = first.op in {"v02_atom_affine", "v02_jet_transform"}
            try:
                A, b = _compose_affine(first.params, second.params, atom_semantics=atom_semantics)
            except RewritePreconditionError:
                continue
            params = dict(second.params)
            params["A"], params["b"] = A, b
            params["fused_from"] = (first.output, second.output)
            replaced_inputs = tuple(
                source for name in second.inputs for source in (first.inputs if name == first.output else (name,))
            )
            fused = replace(second, inputs=replaced_inputs, params=params)
            nodes[index] = fused
            nodes.pop(index - 1)
            uses = _use_counts(nodes)
            changed = True
            rewrites.append(f"{first.output}+{second.output}:affine")
            break
    context.record(PassName.AFFINE_FUSION, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _phase_delta(params: Mapping[str, Any]) -> int | None:
    for key in ("dT", "phase", "delta"):
        if key in params:
            raw = params[key]
            if isinstance(raw, arithmetic.Phase):
                return int(raw.T)
            if np.isscalar(raw):
                try:
                    return int(cast(Any, raw))
                except (TypeError, ValueError):
                    return None
    return None


def _fuse_phase(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    nodes = list(graph.nodes)
    uses = _use_counts(nodes)
    rewrites: list[str] = []
    index = 1
    while index < len(nodes):
        first, second = nodes[index - 1], nodes[index]
        if first.op == second.op == "v02_phase_shift" and first.output in second.inputs and uses[first.output] == 1:
            left, right = _phase_delta(first.params), _phase_delta(second.params)
            if left is not None and right is not None:
                params = dict(second.params)
                params["dT"] = (left + right) & arithmetic.SPF32_PHASE_MASK
                params["fused_from"] = (first.output, second.output)
                params.pop("phase", None)
                params.pop("delta", None)
                fused = replace(
                    second,
                    inputs=tuple(
                        source
                        for name in second.inputs
                        for source in (first.inputs if name == first.output else (name,))
                    ),
                    params=params,
                )
                nodes[index] = fused
                nodes.pop(index - 1)
                uses = _use_counts(nodes)
                rewrites.append(f"{first.output}+{second.output}:phase_shift")
                index = max(1, index - 1)
                continue
        index += 1
    context.record(PassName.PHASE_FUSION, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _static_values(graph: FieldGraph) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for item in graph.nodes:
        if item.op in {"v02_field_constant", "v02_field_asset", "v02_field_reference"}:
            if "value" in item.params:
                values[item.output] = item.params["value"]
            elif "asset" in item.params:
                values[item.output] = item.params["asset"]
            elif "reference" in item.params:
                values[item.output] = item.params["reference"]
    return values


def _fold_static(
    graph: FieldGraph, context: _CompileContext, *, pass_name: PassName, allow_separable: bool = False
) -> FieldGraph:
    values = _static_values(graph)
    nodes: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        if item.op not in _STATIC_FOLD_OPS or any(name not in values for name in item.inputs):
            nodes.append(item)
            continue
        operands = [values[name] for name in item.inputs]
        params = dict(item.params)
        try:
            result = arithmetic_value_execute(item.op, operands, params)
        except (TypeError, ValueError, np.linalg.LinAlgError, OverflowError):
            nodes.append(item)
            continue
        if isinstance(result, MultiResult):
            nodes.append(item)
            continue
        output_kind = item.value_kind
        if output_kind is FieldKind.ANY:
            output_kind = _payload_kind(result)
        folded = replace(
            item,
            op="v02_field_constant",
            inputs=(),
            input_kinds=(),
            metadata=replace(
                item.field_metadata,
                value_kind=output_kind,
                upper_bound=_atom_or_mixture_bound(result, item.field_metadata.support.region),
            ),
            params={"value": result, "value_kind": output_kind.value, "folded_from": item.op},
            annotations={**dict(item.annotations), "constant_folded": item.op},
        )
        nodes.append(folded)
        values[item.output] = result
        rewrites.append(f"{item.output}:{item.op}->constant")
    context.record(pass_name, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def arithmetic_value_execute(operation: str, operands: Sequence[Any], params: Mapping[str, Any]) -> Any:
    """Small indirection that is easy to monkeypatch in compiler tests."""

    return arithmetic_renderer_execute(operation, operands, params)


def arithmetic_renderer_execute(operation: str, operands: Sequence[Any], params: Mapping[str, Any]) -> Any:
    from spf.renderer.v02 import execute_op

    return execute_op(operation, operands, params)


def _try_diagonal_separable(
    mixture: arithmetic.AtomMixture, region: Region | None = None
) -> arithmetic.SeparableField | None:
    """Rewrite a diagonal mixture into separable factors, or refuse.

    This pass *replaces* the mixture node with the separable constant, so it
    only fires on an exactly equivalent conversion.  The shared §17 certificate
    accounts for the off-diagonal chirp term (unbounded in coordinate scale, so
    no absolute tolerance certifies it), the residual, and ``dropped_bound``.
    """

    from spf.renderer.separable_terminal import separable_conversion

    if not mixture.atoms or any(atom.dimension != 2 for atom in mixture.atoms):
        return None
    conversion = separable_conversion(mixture, region)
    if conversion is None or not conversion.exact:
        return None
    return conversion.field


def _common_factor(atom_mixture: arithmetic.AtomMixture) -> arithmetic.GaussianChirpAtom | None:
    if not atom_mixture.atoms:
        return None
    first = atom_mixture.atoms[0]
    first_h = np.asarray(first.h, dtype=np.complex128)
    first_C = np.asarray(first.C, dtype=np.complex128)
    if any(
        np.asarray(atom.h, dtype=np.complex128).tobytes() != first_h.tobytes()
        or np.asarray(atom.C, dtype=np.complex128).tobytes() != first_C.tobytes()
        for atom in atom_mixture.atoms[1:]
    ):
        return None
    return arithmetic.GaussianChirpAtom(first.kappa, first.h, first.C, allow_nondecaying=first.allow_nondecaying)


def _factor_and_separable(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    values = _static_values(graph)
    nodes: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        current = item
        if item.op == "v02_mixture_factor" and item.inputs and item.inputs[0] in values:
            value = values[item.inputs[0]]
            if isinstance(value, arithmetic.AtomMixture):
                factor = _common_factor(value)
                if factor is not None and "factor" not in item.params:
                    params = dict(item.params)
                    params["factor"] = factor
                    params["common_factor_proven"] = True
                    current = replace(item, params=params)
                    rewrites.append(f"{item.output}:common_factor")
        if (
            context.options.enable_separable_rewrites
            and item.op in {"v02_mixture", "v02_atom_mixture"}
            and item.output in values
            and isinstance(values[item.output], arithmetic.AtomMixture)
            and item.params.get("allow_separable", False)
        ):
            separable = _try_diagonal_separable(values[item.output], item.field_metadata.support.region)
            if separable is not None:
                current = replace(
                    item,
                    op="v02_field_constant",
                    inputs=(),
                    metadata=replace(item.field_metadata, value_kind=FieldKind.SEPARABLE),
                    params={"value": separable, "value_kind": FieldKind.SEPARABLE.value, "separable_from": item.op},
                )
                values[item.output] = separable
                rewrites.append(f"{item.output}:mixture->separable")
        nodes.append(current)
    context.record(PassName.COMMON_FACTOR, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _mixture_concat(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    # v02 mixture_add/concat already has flat symbolic semantics.  Static
    # folding above performs exact concatenation; this pass additionally
    # canonicalizes both spellings so later pattern matching is deterministic.
    nodes: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        if item.op == "v02_mixture_add":
            nodes.append(replace(item, op="v02_mixture_concat", params={**dict(item.params), "symbolic_concat": True}))
            rewrites.append(f"{item.output}:mixture_add->concat")
        else:
            nodes.append(item)
    context.record(PassName.MIXTURE_CONCATENATION, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _fft_cancel(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    nodes = list(graph.nodes)
    rewrites: list[str] = []
    index = 1
    while index < len(nodes):
        first, second = nodes[index - 1], nodes[index]
        forward = first.op in {"v02_fft", "v02_fft_forward"}
        inverse = second.op in {"v02_ifft", "v02_fft_inverse"}
        if not (forward and inverse and first.output in second.inputs):
            index += 1
            continue
        uses = _use_counts(nodes)
        if uses[first.output] != 1:
            index += 1
            continue
        if first.output in graph.outputs:
            # The forward transform is independently observable, so removing
            # it would change a declared program result.
            index += 1
            continue
        if first.params.get("convention", "cycles") != second.params.get(
            "convention", first.params.get("convention", "cycles")
        ):
            index += 1
            continue
        if first.params.get("normalization", "inverse") != second.params.get("normalization", "inverse"):
            index += 1
            continue
        source = first.inputs[0] if first.inputs else None
        if source is None:
            index += 1
            continue
        if second.output in graph.outputs:
            # Keep the public SSA name through a typed zero-cost boundary.
            # FieldGraph/FieldIR outputs must be produced nodes; an out-of-band
            # alias would otherwise build an invalid graph.
            kind = second.value_kind
            nodes[index] = replace(
                second,
                op="v02_cast",
                inputs=(source,),
                params={
                    "from_kind": kind.value,
                    "to_kind": kind.value,
                    "identity_from": "fft_ifft_cancel",
                },
                input_kinds=(kind,),
            )
            nodes.pop(index - 1)
        else:
            for later in range(index + 1, len(nodes)):
                item = nodes[later]
                if second.output in item.inputs:
                    nodes[later] = replace(
                        item, inputs=tuple(source if name == second.output else name for name in item.inputs)
                    )
            nodes.pop(index)
            nodes.pop(index - 1)
        rewrites.append(f"{first.output}+{second.output}:fft_ifft_cancel")
        index = max(1, index - 1)
    context.record(PassName.FFT_CANCELLATION, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _hermitian_lower(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    nodes: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        if item.op not in {"v02_hermitian_synthesis", "v02_hermitian_synth"}:
            nodes.append(item)
            continue
        params = dict(item.params)
        if "self_conjugate" not in params and "endpoint_metadata" in params:
            endpoint = params["endpoint_metadata"]
            if isinstance(endpoint, Mapping) and "self_conjugate" in endpoint:
                params["self_conjugate"] = endpoint["self_conjugate"]
        if (
            "self_conjugate" not in params
            and context.options.strict_hermitian
            and not context.options.allow_implicit_hermitian_endpoints
        ):
            raise RewritePreconditionError("Hermitian lowering needs explicit endpoint metadata")
        params["hermitian_lowered"] = True
        params.setdefault("endpoint_metadata", {"explicit": "self_conjugate" in params})
        nodes.append(
            replace(
                item,
                op="v02_hermitian_synthesis",
                params=params,
                annotations={**dict(item.annotations), "hermitian_lowered": True},
            )
        )
        rewrites.append(f"{item.output}:hermitian_lowered")
    context.record(PassName.HERMITIAN_LOWERING, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _prune(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    if not context.options.enable_pruning:
        context.record(PassName.PRUNING, len(graph.nodes), len(graph.nodes), notes=("disabled",))
        return graph
    values = _static_values(graph)
    nodes: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        if item.op not in {"v02_mixture_prune", "v02_mixture_cull"} or not item.inputs or item.inputs[0] not in values:
            nodes.append(item)
            continue
        mixture = values[item.inputs[0]]
        if not isinstance(mixture, arithmetic.AtomMixture):
            nodes.append(item)
            continue
        if (
            cast(PrecisionMode, item.field_metadata.precision.mode) is PrecisionMode.EXACT
            or item.field_metadata.error.phase_required
        ):
            nodes.append(item)
            continue
        region = _region_value(item.params.get("region")) or item.field_metadata.support.region
        if region is None:
            nodes.append(item)
            continue
        budget = item.params.get("error_budget", item.params.get("prune_budget", context.options.error_budget.absolute))
        try:
            budget_value = float(budget)
        except (TypeError, ValueError):
            nodes.append(item)
            continue
        if not math.isfinite(budget_value) or budget_value <= 0.0:
            nodes.append(item)
            continue
        pruned = arithmetic.mixture_prune(
            mixture,
            _region_for_atom(region, mixture.atoms[0].dimension)
            if mixture.atoms
            else _region_for_atom(region, region.dimension),
            budget_value,
        )
        dropped = max(0.0, pruned.dropped_bound - mixture.dropped_bound)
        if dropped <= 0.0:
            nodes.append(item)
            continue
        values[item.output] = pruned
        ledger = item.field_metadata.ledger.plus(arithmetic.ErrorLedger(pruning=dropped))
        current = replace(
            item,
            op="v02_field_constant",
            inputs=(),
            metadata=replace(
                item.field_metadata,
                ledger=ledger,
                upper_bound=_atom_or_mixture_bound(pruned, region),
            ),
            params={
                "value": pruned,
                "value_kind": FieldKind.ATOM_MIXTURE.value,
                "pruned_from": item.op,
                "pruning_bound": dropped,
            },
            annotations={**dict(item.annotations), "bound_certified_pruning": dropped},
        )
        nodes.append(current)
        rewrites.append(f"{item.output}:{dropped:.6g} bound-pruned")
    context.record(PassName.PRUNING, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _dce(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    required = set(graph.outputs)
    keep: list[SFRNode] = []
    for item in reversed(graph.nodes):
        side_effect = item.op in _MATERIALIZE_OPS or item.params.get("side_effect", False)
        if item.output in required or side_effect or context.options.retain_intermediates:
            keep.append(item)
            required.update(item.inputs)
    keep.reverse()
    rewrites = [f"removed:{item.output}" for item in graph.nodes if item not in keep]
    context.record(PassName.DEAD_CODE, len(graph.nodes), len(keep), rewrites=rewrites)
    return _rebuild_graph(keep, graph, context)


def _annotate_reductions(graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    nodes: list[SFRNode] = []
    rewrites: list[str] = []
    for item in graph.nodes:
        if item.op not in _REDUCTION_OPS:
            nodes.append(item)
            continue
        annotation = {
            "kind": item.value_kind.value,
            "ordered": item.op in {"v02_reduce_transport", "v02_transport_reduce", "v02_transport_compose"},
            "phase_required": item.field_metadata.error.phase_required,
            "method": item.params.get("method", "reference"),
        }
        nodes.append(replace(item, annotations={**dict(item.annotations), "reduction": annotation}))
        rewrites.append(f"{item.output}:reduction={annotation['kind']}")
    context.record(PassName.REDUCTION_ANNOTATION, len(graph.nodes), len(nodes), rewrites=rewrites)
    return _rebuild_graph(nodes, graph, context)


def _use_counts(nodes: Sequence[SFRNode]) -> Counter[str]:
    result: Counter[str] = Counter()
    for item in nodes:
        result.update(item.inputs)
    return result


def _rebuild_graph(nodes: Sequence[SFRNode], graph: FieldGraph, context: _CompileContext) -> FieldGraph:
    available = {item.output for item in nodes}
    outputs = tuple(graph.outputs)
    # Removed nodes can be aliases of their original source.  FieldGraph still
    # needs produced output names, so a small pass-through constant is not
    # invented; CompiledProgram applies aliases after execution.
    for output in outputs:
        if output not in available and output not in context.aliases:
            raise CompilationError(f"rewrite removed declared output {output!r} without an alias")
    return FieldGraph(tuple(nodes), graph.inputs, outputs)


def _schedule(graph: FieldGraph) -> MemorySchedule:
    _use_counts(graph.nodes)
    last_use: dict[str, int] = {item.output: index for index, item in enumerate(graph.nodes)}
    for index, item in enumerate(graph.nodes):
        for name in item.inputs:
            last_use[name] = index
    for name in graph.outputs:
        last_use[name] = len(graph.nodes)
    intervals: dict[str, tuple[int, int]] = {}
    sizes: dict[str, int] = {}
    for index, item in enumerate(graph.nodes):
        intervals[item.output] = (index, last_use.get(item.output, index))
        value = item.params.get("value")
        if value is not None:
            # Structural carriers own their compact storage.  Looking at an
            # object-array wrapper reports only one pointer and causes the
            # scheduler to undercount retained blocks, kernels, and filters.
            structural_bytes = getattr(value, "nbytes", None)
            if structural_bytes is not None:
                try:
                    sizes[item.output] = int(structural_bytes)
                    continue
                except (TypeError, ValueError):
                    pass
            try:
                sizes[item.output] = int(np.asarray(value).nbytes)
            except (TypeError, ValueError):
                sizes[item.output] = 64
        else:
            sizes[item.output] = int(item.params.get("byte_size", 64))
    slots: dict[str, tuple[int, int]] = {}
    active: list[tuple[int, int, str]] = []
    next_slot = 0
    peak = 0
    releases: dict[int, list[str]] = defaultdict(list)
    for name, (start, end) in intervals.items():
        active = [entry for entry in active if entry[0] >= start]
        occupied = {slot for _, slot, _ in active}
        slot = 0
        while slot in occupied:
            slot += 1
        next_slot = max(next_slot, slot + 1)
        slots[name] = (slot, sizes[name])
        active.append((end, slot, name))
        releases[end].append(name)
        current = sum(sizes[live_name] for _, _, live_name in active)
        peak = max(peak, current)
    return MemorySchedule(intervals, slots, {key: tuple(value) for key, value in releases.items()}, int(peak))


def _validate_capabilities(graph: FieldGraph, context: _CompileContext) -> None:
    backend = context.backend
    try:
        backend.require_available()
    except BackendUnavailableError as exc:
        if not context.options.allow_fallback:
            raise
        fallback = get_backend(context.options.fallback_backend)
        fallback.require_available()
        if fallback.name == backend.name:
            raise
        context.fallbacks.append(f"backend {backend.name} -> {fallback.name}: {exc.reason}")
        context.backend = fallback
        backend = fallback
    missing_ops = sorted({item.op for item in graph.nodes if not backend.capabilities.supports_operation(item.op)})
    if missing_ops:
        if not context.options.allow_fallback:
            raise BackendCapabilityError(f"backend {backend.name!r} lacks operations: {', '.join(missing_ops)}")
        fallback = get_backend(context.options.fallback_backend)
        fallback.require_available()
        fallback_missing = sorted(
            {item.op for item in graph.nodes if not fallback.capabilities.supports_operation(item.op)}
        )
        if fallback_missing:
            raise BackendCapabilityError(
                f"fallback backend {fallback.name!r} lacks operations: {', '.join(fallback_missing)}"
            )
        context.fallbacks.append(
            f"backend {backend.name} -> {fallback.name}: unsupported operations {', '.join(missing_ops)}"
        )
        context.backend = fallback
        backend = fallback
    for item in graph.nodes:
        representation = item.params.get("representation")
        if representation is not None:
            try:
                if not backend.capabilities.supports_representation(_representation_kind(representation)):
                    raise BackendCapabilityError(
                        f"backend {backend.name!r} lacks representation {representation!r} for {item.output!r}"
                    )
            except ValueError as exc:
                raise BackendCapabilityError(str(exc)) from exc
    context.record(
        PassName.CAPABILITY,
        len(graph.nodes),
        len(graph.nodes),
        notes=(f"backend={backend.name}", "capabilities validated"),
    )


def _field_for_node(node: SFRNode) -> Field | None:
    if "value" not in node.params:
        return None
    payload = node.params["value"]
    kind = node.value_kind if node.value_kind is not FieldKind.ANY else _payload_kind(payload)
    representation = {
        FieldKind.ATOM: RepresentationKind.GAUSSIAN_CHIRP_ATOM,
        FieldKind.ATOM_MIXTURE: RepresentationKind.ATOM_MIXTURE,
        FieldKind.JET2D: RepresentationKind.QUADRATIC_JET,
        FieldKind.SEPARABLE: RepresentationKind.SEPARABLE,
        FieldKind.SPF_COMPLEX: RepresentationKind.DENSE_TYPED,
        FieldKind.WORDS: RepresentationKind.DENSE_TYPED,
    }.get(kind)
    if representation is None:
        from spf.renderer.model import default_dense_representation

        representation = default_dense_representation(kind, payload)
    return Field(
        id=node.output,
        domain=node.field_metadata.domain,
        semantic=node.field_metadata.semantic,
        coherence=node.field_metadata.coherence,
        value_kind=kind,
        representation=FieldRepresentation(representation, payload, error=node.field_metadata.ledger),
        chart=node.field_metadata.chart,
        support=node.field_metadata.support,
        error=node.field_metadata.error,
        precision=node.field_metadata.precision,
    )


def _plan_materializations(graph: FieldGraph, context: _CompileContext) -> Any:
    try:
        from spf.renderer.planner import MaterializationPlanner

        planner = MaterializationPlanner(
            backend=context.backend,
            precision=context.options.precision,
            error_budget=context.options.error_budget,
            allow_fallback=context.options.allow_fallback,
            candidate_kinds=context.options.consumer_representations,
        )
        # A direct ``Field`` carries deferred operations outside the SSA
        # payload. Keep that original object as the planner input so candidate
        # construction can inspect the operation chain before selecting a
        # representation. Static graph sources remain the fallback for
        # FieldGraph/FieldIR callers that have no owning Field object.
        fields = context.source_fields or tuple(
            field_value for field_value in (_field_for_node(item) for item in graph.nodes) if field_value is not None
        )
        if context.options.addressed_request and len(fields) == 1:
            # Explicit points have no regular-grid fitting contract. Keep the
            # source representation and let its addressed consumer sample it.
            planner.candidate_kinds = (fields[0].representation.kind,)
        if not fields:
            context.record(
                PassName.MATERIALIZATION,
                len(graph.nodes),
                len(graph.nodes),
                notes=("no static materialization sources",),
            )
            return None
        plan = planner.plan_fields(fields, regions=context.options.regions, layout=context.options.layout)
        context.record(
            PassName.MATERIALIZATION, len(graph.nodes), len(graph.nodes), rewrites=(f"decisions={len(plan.decisions)}",)
        )
        return plan
    except (ImportError, ValueError, TypeError) as exc:
        from spf.renderer.normalization import MaterializationError

        if isinstance(exc, MaterializationError):
            raise
        if context.options.strict:
            raise CompilationError(f"materialization planning failed: {exc}") from exc
        context.record(PassName.MATERIALIZATION, len(graph.nodes), len(graph.nodes), notes=(f"deferred: {exc}",))
        return None


def compile_graph(
    graph: FieldGraph | FieldIR | RenderPlan | Field,
    *,
    options: CompilerOptions | None = None,
    backend: str | SFRBackend | None = None,
    **option_overrides: Any,
) -> CompiledProgram:
    """Compile a FieldGraph/FieldIR/Field into an executable SFR program."""

    selected = options or CompilerOptions(backend=backend or "reference")
    if backend is not None or option_overrides:
        values = {name: getattr(selected, name) for name in selected.__dataclass_fields__}
        if backend is not None:
            values["backend"] = backend
        values.update(option_overrides)
        selected = CompilerOptions(**values)
    plan_metadata: Mapping[str, Any] = {}
    if isinstance(graph, Field):
        source_field = _adapt_spectral_texture_field(graph)
        source = FieldGraph.from_field(source_field)
    elif isinstance(graph, RenderPlan):
        source = FieldGraph.from_ir(graph.ir)
        plan_metadata = graph.metadata
    elif isinstance(graph, FieldIR):
        source = FieldGraph.from_ir(graph)
    elif isinstance(graph, FieldGraph):
        source = graph
    else:
        raise TypeError(f"unsupported compile input {type(graph).__name__}")
    backend_object = get_backend(selected.backend)
    source_fields = (source_field,) if isinstance(graph, Field) else ()
    context = _CompileContext(selected, backend_object, source_fields=source_fields)
    current = _canonicalize(source, context)
    current = _propagate(current, context)
    if selected.optimize:
        current = _fuse_affine(current, context)
        current = _fuse_phase(current, context)
        current = _fold_static(current, context, pass_name=PassName.ATOM_PRODUCT_FUSION)
        current = _fold_static(current, context, pass_name=PassName.MIXTURE_CONCATENATION)
        current = _mixture_concat(current, context)
        current = _factor_and_separable(current, context)
        current = _fft_cancel(current, context)
        current = _hermitian_lower(current, context)
        current = _prune(current, context)
        current = _dce(current, context)
    else:
        for pass_name in (
            PassName.AFFINE_FUSION,
            PassName.PHASE_FUSION,
            PassName.ATOM_PRODUCT_FUSION,
            PassName.MIXTURE_CONCATENATION,
            PassName.COMMON_FACTOR,
            PassName.FFT_CANCELLATION,
            PassName.HERMITIAN_LOWERING,
            PassName.PRUNING,
            PassName.DEAD_CODE,
        ):
            context.record(pass_name, len(current.nodes), len(current.nodes), notes=("optimization disabled",))
    current = _annotate_reductions(current, context)
    _validate_capabilities(current, context)
    backend_object = context.backend
    materialization_plan = _plan_materializations(current, context)
    schedule = _schedule(current)
    context.record(
        PassName.LIVENESS, len(current.nodes), len(current.nodes), notes=(f"peak_bytes={schedule.peak_bytes}",)
    )
    metrics = RenderMetrics(
        backend=backend_object.name,
        workload="sfr_compiled_program",
        representation="typed_ssa",
        precision=cast(PrecisionMode, selected.precision.mode).value.lower(),
        error_budget=selected.error_budget.absolute,
        field_operations=len(current.nodes),
        peak_memory=schedule.peak_bytes,
    )
    diagnostics = {
        "original_nodes": len(source.nodes),
        "optimized_nodes": len(current.nodes),
        "outputs": current.outputs,
        "aliases": dict(context.aliases),
        "backend_native": backend_object.capabilities.native,
        "backend_available": backend_object.capabilities.available,
        "requested_backend": get_backend(selected.backend).name,
        "rewrites": tuple(context.rewrites),
        "notes": tuple(context.notes),
        "liveness": dict(schedule.intervals),
        "memory_slots": dict(schedule.slots),
        "reduction_annotations": {
            item.output: dict(item.annotations["reduction"])
            for item in current.nodes
            if "reduction" in item.annotations
        },
        "render_plan_metadata": dict(plan_metadata),
    }
    if context.source_fields:
        route, reason = _field_materialization_route(context.source_fields[0], current)
        diagnostics["materialization_execution"] = route
        diagnostics["materialization_execution_reason"] = reason
    report = RenderReport(
        metrics,
        ledger=arithmetic.ErrorLedger(),
        passes=tuple(context.pass_records),
        fallbacks=tuple(context.fallbacks),
        capabilities_used=(backend_object.name,),
        diagnostics=diagnostics,
    )
    return CompiledProgram(
        graph=current,
        original_graph=source,
        backend=backend_object.name,
        report=report,
        passes=tuple(context.pass_records),
        memory_schedule=schedule,
        aliases=context.aliases,
        materialization_plan=materialization_plan,
        _backend_object=backend_object,
        _source_field=context.source_fields[0] if context.source_fields else None,
    )


def compile_field_graph(graph: FieldGraph | FieldIR | RenderPlan | Field, **kwargs: Any) -> CompiledProgram:
    return compile_graph(graph, **kwargs)


def compile_program(graph: FieldGraph | FieldIR | RenderPlan | Field, **kwargs: Any) -> CompiledProgram:
    return compile_graph(graph, **kwargs)


def compile(graph: FieldGraph | FieldIR | RenderPlan | Field, **kwargs: Any) -> CompiledProgram:
    return compile_graph(graph, **kwargs)


def _request_signature(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (value.shape, value.dtype.str, value.tobytes())
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _request_signature(v)) for k, v in value.items()))
    if is_dataclass(value) and not isinstance(value, type):
        return tuple((f.name, _request_signature(getattr(value, f.name))) for f in fields(value))
    if isinstance(value, (tuple, list)):
        return tuple(_request_signature(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class PreparedMaterialization:
    """A reusable representation plan with an explicit source-update contract."""

    source: Field
    request: Any
    program: CompiledProgram
    terminal: Any = field(default=None, init=False, repr=False, compare=False)
    terminal_report: Any = field(default=None, init=False, repr=False, compare=False)
    terminal_result: Any = field(default=None, init=False, repr=False, compare=False)

    updated_prepared: Any = field(default=None, init=False, repr=False, compare=False)
    source_updated: bool = field(default=False, init=False, repr=False, compare=False)

    def execute(self, *, field: Field | None = None, **options: Any) -> Any:
        selected = self.source if field is None else field
        if selected is not self.source:
            from spf.renderer.blocks import BlockView

            before, after = self.source.representation.payload, selected.representation.payload
            if (
                not isinstance(before, BlockView)
                or not isinstance(after, BlockView)
                or before.shape != after.shape
                or self.source.operations
                or selected.operations
                or any(
                    getattr(self.source, name) != getattr(selected, name)
                    for name in ("domain", "coherence", "value_kind", "chart", "support", "error", "precision")
                )
            ):
                raise ValueError(
                    "prepared source update requires compatible block headers; prepare other changes again"
                )
            updated = self.updated_prepared
            if updated is None or updated.source is not selected:
                from spf.renderer.terminal import prepare_terminal

                updated = PreparedMaterialization(selected, self.request, self.program)
                object.__setattr__(updated, "source_updated", True)
                object.__setattr__(updated, "terminal", prepare_terminal(selected, self.request))
                object.__setattr__(self, "updated_prepared", updated)
            return materialize_field(selected, self.request, _prepared=updated, **options)
        return materialize_field(selected, self.request, _prepared=self, **options)


_PREPARED_CACHE: OrderedDict[Any, PreparedMaterialization] = OrderedDict()


def _selected_execution_field(field: Field, program: CompiledProgram, *, updated: bool = False) -> Any:
    """Resolve consumed operations once without sampling a representation."""
    field = _adapt_spectral_texture_field(field)
    if program.materialization_plan is None or not program.materialization_plan.selected:
        raise CompilationError(f"compiler produced no materialization decision for {field.id!r}")
    planner_choices = tuple(program.materialization_plan.selected)
    choice = next((item for item in planner_choices if item.field_id == field.id), planner_choices[0])
    from spf.renderer.normalization import select_representation

    selected = select_representation(field, choice.representation)
    consumed = selected.consumed_indices
    execution_field = selected.field
    if updated:
        execution_field = replace(
            execution_field, representation=replace(field.representation, consumed_operation_indices=())
        )
    return planner_choices, choice, consumed, set(consumed), execution_field.operations, execution_field


def prepare_materialization(field: Field, request: Any = None) -> PreparedMaterialization:
    """Compile static source state once; mutable generic arrays are never cached."""
    from spf.renderer.model import MaterializationRequest, _freeze_value
    from spf.renderer.planner import ErrorBudgetExceeded
    from spf.renderer.runtime import MaterializationError

    selected = request or MaterializationRequest()
    if not isinstance(field, Field) or not isinstance(selected, MaterializationRequest):
        raise TypeError("preparation requires a Field and MaterializationRequest")
    payload = field.representation.payload
    cacheable = payload.__class__.__name__ in {
        "GaussianChirpAtom",
        "AtomMixture",
        "Jet2D",
        "TensorBlockView",
        "V3BlockView",
        "KernelBlockView",
        "FilteredBlockView",
        "SpectralTexture",
        "SpectralJetTile2D",
        "AdaptiveJetField",
    }
    key = (id(field), _request_signature(selected))
    if cacheable and key in _PREPARED_CACHE:
        prepared = _PREPARED_CACHE.pop(key)
        _PREPARED_CACHE[key] = prepared
        return prepared
    regions: tuple[Region, ...] = ()
    addressed = False
    region = selected.region
    if isinstance(region, Region):
        regions = (region,)
    elif isinstance(region, Mapping):
        points = region.get("points", region.get("directions"))
        if points is not None:
            points = np.asarray(points, dtype=np.float64)
            if points.ndim < 2 or not np.all(np.isfinite(points)):
                raise ValueError("explicit points must have a finite trailing coordinate dimension")
            flat = points.reshape(-1, points.shape[-1])
            if flat.size:
                regions = (Region(tuple(flat.min(axis=0)), tuple(flat.max(axis=0))),)
            addressed = True
        elif "lo" in region and "hi" in region:
            regions = (Region(tuple(region["lo"]), tuple(region["hi"])),)
    try:
        program = compile_graph(
            field,
            options=CompilerOptions(
                backend=selected.backend,
                precision=selected.precision,
                error_budget=selected.error_budget,
                layout=selected.layout,
                regions=regions,
                addressed_request=addressed,
                allow_fallback=True,
                strict=True,
            ),
        )
    except ErrorBudgetExceeded as exc:
        raise MaterializationError(
            f"materialization exceeds requested ErrorBudget or field ErrorContract: {exc}"
        ) from exc
    prepared = PreparedMaterialization(field, replace(selected, region=_freeze_value(selected.region)), program)
    from spf.renderer.terminal import prepare_terminal

    if cacheable:
        execution_field = _selected_execution_field(field, program)[-1]
        object.__setattr__(prepared, "terminal", prepare_terminal(execution_field, selected))
    if cacheable:
        _PREPARED_CACHE[key] = prepared
        while len(_PREPARED_CACHE) > 16:
            _PREPARED_CACHE.popitem(last=False)
    return prepared


def materialize_field(
    field: Field,
    request: Any = None,
    *,
    return_result: bool = False,
    return_certificate: bool = False,
    return_report: bool = False,
    backend: SFRBackend | str | None = None,
    _prepared: PreparedMaterialization | None = None,
) -> Any:
    """Compile, plan, and execute one public :class:`Field` boundary.

    Deferred field operations remain attached to the runtime payload because
    they may require a representation-aware whole-field fallback (notably an
    atom mixture with a residual).  The compiler nevertheless owns backend
    validation and representation planning, and the selected backend/decision
    is the one handed to the materialization runtime.  This closes the former
    public-API bypass while retaining one authoritative operation evaluator.
    """

    from dataclasses import replace as dataclass_replace

    from spf.renderer.model import MaterializationRequest

    if not isinstance(field, Field):
        raise TypeError("compiler.materialize_field expects Field")
    selected = request or MaterializationRequest()
    if not isinstance(selected, MaterializationRequest):
        raise TypeError("request must be MaterializationRequest")

    # Keep deferred operations in the compiler graph and planner input. The
    # selected representation can absorb a proven prefix; only the remaining
    # operation indices are handed to the runtime below.
    if backend is not None:
        selected = dataclass_replace(selected, backend=get_backend(backend).name)
    reused = _prepared is not None or (id(field), _request_signature(selected)) in _PREPARED_CACHE
    prepared = _prepared or prepare_materialization(field, selected)
    program = prepared.program
    if field is prepared.source and prepared.terminal is not None and prepared.terminal_report is not None:
        projected = prepared.terminal.execute()
        executed = dataclass_replace(prepared.terminal_result, values=projected.values)
        if return_report:
            return executed, prepared.terminal_report
        return executed if return_result or return_certificate else executed.values
    planner_choices, choice, consumed, consumed_set, pending_operations, execution_field = _selected_execution_field(
        field, program, updated=field is not prepared.source or prepared.source_updated
    )
    operation_count = len(field.operations)
    from spf.tensor import v02

    terminal = prepared.terminal if field is prepared.source else None
    if terminal is None:
        executed, runtime_report = program._backend().materialize_field(
            execution_field,
            selected,
            return_report=True,
        )
    else:
        projected = terminal.execute()
        runtime_report = projected.report
        bound = float(runtime_report.ledger.total_bound)
        executed = v02.MaterializationResult(
            projected.values,
            bound,
            bound / max(float(np.max(np.abs(projected.values), initial=0)), np.finfo(float).tiny),
            runtime_report.ledger,
        )
    diagnostics = dict(runtime_report.diagnostics)
    diagnostics.update(
        {
            "compiler_integrated": True,
            "execution_lane": "prepared_representation_consumer",
            "prepared_program_reused": reused,
            "compiler_original_nodes": program.report.diagnostics["original_nodes"],
            "compiler_optimized_nodes": program.report.diagnostics["optimized_nodes"],
            "planner_selected": tuple(item.kind.value for item in planner_choices),
            "planner_executed_representation": choice.kind.value,
            "planner_consumed_operation_indices": consumed,
            "planner_pending_operation_indices": tuple(
                index for index in range(operation_count) if index not in consumed_set
            ),
            "deferred_operations_executed": tuple(operation.op for operation in pending_operations),
        }
    )
    merged_report = RenderReport(
        metrics=runtime_report.metrics,
        ledger=runtime_report.ledger,
        passes=(
            *program.report.passes,
            CompilerPassRecord(
                "deferred_field_program",
                operation_count,
                len(pending_operations),
                notes=(
                    "executed by representation-aware runtime after planner selection",
                    f"consumed_indices={consumed}",
                ),
            ),
            *runtime_report.passes,
        ),
        decisions=(*program.materialization_plan.decisions, *runtime_report.decisions),
        fallbacks=(*program.report.fallbacks, *runtime_report.fallbacks),
        capabilities_used=tuple(
            dict.fromkeys(
                (
                    *program.report.capabilities_used,
                    *runtime_report.capabilities_used,
                    "compiler_planner_runtime",
                    "prepared_backend_dispatch",
                )
            )
        ),
        diagnostics=diagnostics,
    )
    if terminal is not None:
        object.__setattr__(prepared, "terminal", terminal)
        object.__setattr__(prepared, "terminal_report", merged_report)
        object.__setattr__(prepared, "terminal_result", dataclass_replace(executed, values=None))
    if return_report:
        return executed, merged_report
    if return_result or return_certificate:
        return executed
    return executed.values



# --------------------------------------------------------------------------- #
# Equality saturation (design §33.6; blog Tier 2 item 10).
#
# A small but real e-graph: e-classes joined by a union-find, hash-consed
# e-nodes, rewrite rules applied to saturation under a node budget, and
# extraction by backend cost.  The filter-chain planner uses it to hold
# ``ifft(mul(fft(F), H))`` and ``conv(F, h)`` (and the translate-as-header-ramp
# and fused-transfer forms) in one class and pick the cheapest under the
# backend's measured ``CostModel``.  Both the chosen and the rejected forms are
# reported with their costs.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ENode:
    """One operator application whose children are e-class ids."""

    op: str
    children: tuple[int, ...] = ()
    label: Any = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "children", tuple(int(c) for c in self.children))
        hash(self.label)


Expr = tuple[str, Any, tuple["Expr", ...]]


def format_expr(expr: Expr) -> str:
    op, label, children = expr
    inner = ", ".join(format_expr(child) for child in children)
    if label is None or label == ():
        return f"{op}({inner})" if children else op
    tag = ",".join(str(v) for v in label) if isinstance(label, tuple) else str(label)
    return f"{op}[{tag}]({inner})" if children else f"{op}[{tag}]"


class EGraphBudgetError(CompilationError):
    """Raised when saturation would exceed the configured node budget."""


@dataclass(frozen=True, slots=True)
class SaturationRecord:
    iterations: int
    node_count: int
    class_count: int
    saturated: bool
    budget: int
    applied: tuple[str, ...] = ()


class EGraph:
    """Union-find e-graph with hash-consing, rebuild, saturation, and cost extraction."""

    def __init__(self, *, node_budget: int = 512) -> None:
        self._parent: list[int] = []
        self.classes: dict[int, set[ENode]] = {}
        self._memo: dict[ENode, int] = {}
        self.node_budget = int(node_budget)
        self.node_count = 0

    # ---- union-find ---------------------------------------------------------

    def find(self, class_id: int) -> int:
        root = int(class_id)
        while self._parent[root] != root:
            root = self._parent[root]
        node = int(class_id)
        while self._parent[node] != root:
            self._parent[node], node = root, self._parent[node]
        return root

    def canonical(self, node: ENode) -> ENode:
        return ENode(node.op, tuple(self.find(c) for c in node.children), node.label)

    def add(self, node: ENode) -> int:
        node = self.canonical(node)
        existing = self._memo.get(node)
        if existing is not None:
            return self.find(existing)
        if self.node_count >= self.node_budget:
            raise EGraphBudgetError(f"e-graph node budget {self.node_budget} exhausted")
        class_id = len(self._parent)
        self._parent.append(class_id)
        self.classes[class_id] = {node}
        self._memo[node] = class_id
        self.node_count += 1
        return class_id

    def add_expr(self, expr: Expr) -> int:
        op, label, children = expr
        return self.add(ENode(op, tuple(self.add_expr(child) for child in children), label))

    def merge(self, a: int, b: int) -> int:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return ra
        keep, drop = (ra, rb) if ra < rb else (rb, ra)
        self._parent[drop] = keep
        self.classes[keep] |= self.classes.pop(drop)
        return keep

    def rebuild(self) -> None:
        """Re-canonicalise every e-node and merge classes that became congruent."""

        changed = True
        while changed:
            changed = False
            self._memo = {}
            for class_id in list(self.classes):
                if self.find(class_id) != class_id or class_id not in self.classes:
                    continue
                nodes = {self.canonical(n) for n in self.classes[class_id]}
                self.classes[class_id] = nodes
                for node in nodes:
                    other = self._memo.get(node)
                    if other is None:
                        self._memo[node] = class_id
                    elif self.find(other) != class_id:
                        self.merge(other, class_id)
                        changed = True
                        break
                if changed:
                    break

    def nodes(self) -> list[tuple[int, ENode]]:
        return [(class_id, node) for class_id, nodes in self.classes.items() for node in sorted(nodes, key=repr)]

    def xfer_label(self, class_id: int, op: str) -> Any:
        for node in self.classes[self.find(class_id)]:
            if node.op == op:
                return node.label
        return None

    # ---- saturation -----------------------------------------------------------

    def saturate(self, rules: Sequence[Any], *, iterations: int = 8) -> SaturationRecord:
        """Apply ``rules`` until no rule changes the graph, or the iteration/node budget ends."""

        applied: list[str] = []
        saturated = False
        done = 0
        for _ in range(iterations):
            done += 1
            before = (self.node_count, len(self.classes), len(applied))
            try:
                for rule in rules:
                    for name, left, right in rule(self):
                        if self.find(left) != self.find(right):
                            self.merge(left, right)
                            applied.append(name)
            except EGraphBudgetError:
                self.rebuild()
                break
            self.rebuild()
            if (self.node_count, len(self.classes), len(applied)) == before:
                saturated = True
                break
        return SaturationRecord(done, self.node_count, len(self.classes), saturated, self.node_budget, tuple(applied))

    # ---- extraction -----------------------------------------------------------

    def _best(self, cost_fn: Any) -> dict[int, tuple[float, ENode]]:
        """Cheapest e-node per class by bottom-up dynamic programming (cyclic classes converge)."""

        best: dict[int, tuple[float, ENode]] = {}
        changed = True
        while changed:
            changed = False
            for class_id, nodes in self.classes.items():
                for node in nodes:
                    child_costs = []
                    for child in node.children:
                        entry = best.get(self.find(child))
                        if entry is None:
                            break
                        child_costs.append(entry[0])
                    else:
                        cost = float(cost_fn(node, child_costs))
                        current = best.get(class_id)
                        if current is None or cost < current[0] - 1.0e-12:
                            best[class_id] = (cost, node)
                            changed = True
        return best

    def extract(self, root: int, cost_fn: Any) -> tuple[Expr, float]:
        """Cheapest expression for ``root`` under ``cost_fn(node, child_costs)``."""

        best = self._best(cost_fn)
        root = self.find(root)
        return self._build(root, best), best[root][0]

    def _build(self, class_id: int, best: dict[int, tuple[float, ENode]]) -> Expr:
        _, node = best[self.find(class_id)]
        return (node.op, node.label, tuple(self._build(c, best) for c in node.children))

    def alternatives(self, root: int, cost_fn: Any, *, limit: int = 16, depth: int = 4) -> list[tuple[Expr, float]]:
        """Distinct full expressions for ``root`` (all node choices to ``depth``, best below), cheapest first."""

        best = self._best(cost_fn)

        def enumerate_class(class_id: int, level: int) -> list[tuple[Expr, float]]:
            class_id = self.find(class_id)
            out: list[tuple[Expr, float]] = []
            for node in sorted(self.classes[class_id], key=repr):
                if level >= depth:
                    exprs = tuple(self._build(c, best) for c in node.children)
                    costs = [best[self.find(c)][0] for c in node.children]
                    out.append(((node.op, node.label, exprs), float(cost_fn(node, costs))))
                    continue
                combos: list[tuple[tuple[Expr, ...], tuple[float, ...]]] = [((), ())]
                for child in node.children:
                    options = enumerate_class(child, level + 1)
                    combos = [((*ex, e), (*co, c)) for ex, co in combos for e, c in options][: limit * 4]
                for ex, co in combos:
                    out.append(((node.op, node.label, ex), float(cost_fn(node, list(co)))))
            return out[: limit * 4]

        results: list[tuple[Expr, float]] = []
        seen: set[str] = set()
        for expr, cost in enumerate_class(root, 0):
            text = format_expr(expr)
            if text not in seen:
                seen.add(text)
                results.append((expr, cost))
        results.sort(key=lambda item: (item[1], format_expr(item[0])))
        return results[:limit]


# ---- filter-chain instance ----------------------------------------------------


_QUADRATIC_FILTERS = {"translate", "gaussian_filter", "exposure", "lens_phase", "fresnel"}


@dataclass(frozen=True, slots=True)
class FilterEquivalent:
    expr: str
    cost: float
    chosen: bool
    legal: bool
    kind: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "expr": self.expr,
            "cost": self.cost,
            "chosen": self.chosen,
            "legal": self.legal,
            "kind": self.kind,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class FilterSaturationPlan:
    """Outcome of saturating one filter chain: chosen form, rejected forms, costs, and the conv kernel."""

    chosen: FilterEquivalent
    equivalents: tuple[FilterEquivalent, ...]
    record: SaturationRecord
    element_count: int
    cost_model: Any
    kernel: Any = None
    kernel_reason: str = ""

    @property
    def executes_convolution(self) -> bool:
        return self.chosen.kind == "conv" and self.kernel is not None

    def pass_record(self) -> CompilerPassRecord:
        return CompilerPassRecord(
            "equality_saturation",
            self.record.node_count,
            len(self.equivalents),
            rewrites=tuple(sorted(set(self.record.applied))),
            notes=(
                f"chosen={self.chosen.expr}",
                f"saturated={self.record.saturated}",
                f"iterations={self.record.iterations}",
                f"node_budget={self.record.budget}",
            ),
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "equivalents": tuple(item.as_dict() for item in self.equivalents),
            "chosen_equivalent": self.chosen.expr,
            "chosen_equivalent_cost": self.chosen.cost,
            "equality_saturation": {
                "iterations": self.record.iterations,
                "node_count": self.record.node_count,
                "class_count": self.record.class_count,
                "saturated": self.record.saturated,
                "node_budget": self.record.budget,
                "rules_applied": tuple(sorted(set(self.record.applied))),
            },
            "convolution_kernel_taps": None if self.kernel is None else len(self.kernel.words),
            "convolution_reason": self.kernel_reason,
        }

    def cost_estimate(self) -> dict[str, Any]:
        estimate = self.cost_model.cost_estimate(
            self.chosen.cost, element_count=self.element_count, expression=self.chosen.expr
        )
        return dict(estimate)


def _filter_cost_fn(model: Any, element_count: int, op_names: Sequence[str], kernel_taps: Any) -> Any:
    n = max(1, int(element_count))
    p = model.primitives_for(n)
    log_n = max(math.log2(n), 1.0)

    def xfer_cost(label: tuple[int, ...]) -> float:
        names = [op_names[i] for i in label]
        cost = 0.0
        if any(name in _QUADRATIC_FILTERS or name == "phase" for name in names):
            cost += p.jet_setup + n * p.jet_sample
        for name in names:
            if name == "glare":
                cost += 3.0 * (p.atom_transform + n * p.atom_sample)
            elif name in {"lod", "phase"}:
                cost += n * p.dense_sample
        return cost

    def cost(node: ENode, children: Sequence[float]) -> float:
        total = float(sum(children))
        if node.op == "src":
            return 0.0
        if node.op == "xfer":
            return xfer_cost(tuple(node.label))
        if node.op == "mul":
            return float(total + n * (p.phase_add + p.log_add))
        if node.op in {"fft", "ifft"}:
            return float(total + n * log_n * p.fft_coefficient)
        if node.op == "ramp":
            return float(total + 4.0 * p.phase_add)
        if node.op == "kern":
            taps = kernel_taps(tuple(node.label))
            return math.inf if taps is None else float(taps) * float(p.jet_sample)
        if node.op == "conv":
            taps = kernel_taps(tuple(node.label))
            if taps is None:
                return math.inf
            k = float(taps)
            words = n * k * (p.phase_add + p.log_add)
            # A one-tap kernel is a gather plus one word product: no reduction.
            reduce = n * (p.pivot_setup + k * p.pair_complex + p.pivot_finalize) if k > 1 else 0.0
            return float(total + words + reduce)
        return total

    return cost


def _filter_rules(op_names: Sequence[str], compact: Any) -> list[Any]:
    def translate_only(label: tuple[int, ...]) -> bool:
        return bool(label) and all(op_names[i] == "translate" for i in label)

    def fuse(g: EGraph) -> list[tuple[str, int, int]]:
        out = []
        for cls, node in g.nodes():
            if node.op != "mul":
                continue
            inner, t2 = node.children
            for m in list(g.classes[g.find(inner)]):
                if m.op != "mul":
                    continue
                a, b = g.xfer_label(m.children[1], "xfer"), g.xfer_label(t2, "xfer")
                if a is None or b is None:
                    continue
                label = tuple(sorted(set(a) | set(b)))
                new = g.add(ENode("mul", (m.children[0], g.add(ENode("xfer", (), label)))))
                out.append(("fuse_transfers", cls, new))
        return out

    def conv(g: EGraph) -> list[tuple[str, int, int]]:
        out = []
        for cls, node in g.nodes():
            if node.op != "ifft":
                continue
            for m in list(g.classes[g.find(node.children[0])]):
                if m.op != "mul":
                    continue
                label = g.xfer_label(m.children[1], "xfer")
                if label is None or not compact(tuple(label)):
                    continue
                for f in list(g.classes[g.find(m.children[0])]):
                    if f.op != "fft":
                        continue
                    kern = g.add(ENode("kern", (), tuple(label)))
                    new = g.add(ENode("conv", (f.children[0], kern), tuple(label)))
                    out.append(("spectral_to_convolution", cls, new))
        return out

    def conv_fuse(g: EGraph) -> list[tuple[str, int, int]]:
        out = []
        for cls, node in g.nodes():
            if node.op != "conv":
                continue
            for m in list(g.classes[g.find(node.children[0])]):
                if m.op != "conv":
                    continue
                label = tuple(sorted(set(m.label) | set(node.label)))
                if not compact(label):
                    continue
                kern = g.add(ENode("kern", (), label))
                new = g.add(ENode("conv", (m.children[0], kern), label))
                out.append(("fuse_convolutions", cls, new))
        return out

    def cancel(g: EGraph) -> list[tuple[str, int, int]]:
        out = []
        for cls, node in g.nodes():
            if node.op not in {"fft", "ifft"}:
                continue
            other = "ifft" if node.op == "fft" else "fft"
            for m in list(g.classes[g.find(node.children[0])]):
                if m.op == other:
                    out.append(("fft_ifft_cancel", cls, m.children[0]))
        return out

    def ramp(g: EGraph) -> list[tuple[str, int, int]]:
        out = []
        for cls, node in g.nodes():
            if node.op != "mul":
                continue
            label = g.xfer_label(node.children[1], "xfer")
            if label is None or not translate_only(label):
                continue
            spectral = any(
                m.op == "fft" or (m.op == "src" and m.label == "fourier") for m in g.classes[g.find(node.children[0])]
            )
            if not spectral:
                continue
            new = g.add(ENode("ramp", (node.children[0],), tuple(label)))
            out.append(("translate_to_header_ramp", cls, new))
        return out

    return [fuse, conv, conv_fuse, cancel, ramp]


def saturate_filter_chain(
    chain: Any,
    output: str,
    *,
    cost_model: Any | None = None,
    node_budget: int = 512,
    iterations: int = 8,
) -> FilterSaturationPlan:
    """Saturate ``chain`` for ``output`` and pick the cheapest equivalent under ``cost_model``.

    ``chain`` is a :class:`spf.renderer.filter_chain.FilterChain`; its
    ``word_kernel(indices)`` provides the compact spatial kernel (or ``None``)
    for a set of operations and ``word_kernel_legal(kernel)`` says whether the
    kernel's certified bound fits the chain's error contract.
    """

    from spf.renderer.cost import cpu_cost_model

    model = cost_model if cost_model is not None else cpu_cost_model()
    op_names = tuple(op.name.value for op in chain.operations)
    element_count = int(math.prod(chain.shape))
    spectral_output = str(output).lower() in {"frequency", "spectrum", "spectrum_intensity", "intensity"}
    kernels: dict[tuple[int, ...], Any] = {}
    reasons: dict[tuple[int, ...], str] = {}

    def kernel(label: tuple[int, ...]) -> Any:
        if label not in kernels:
            try:
                kernels[label], reasons[label] = chain.word_kernel(label)
            except (ValueError, TypeError) as exc:
                kernels[label], reasons[label] = None, f"kernel derivation failed: {exc}"
        return kernels[label]

    def compact(label: tuple[int, ...]) -> bool:
        return kernel(label) is not None

    def taps(label: tuple[int, ...]) -> int | None:
        k = kernel(label)
        if k is None or not chain.word_kernel_legal(k):
            return None
        return len(k.words)

    # The seed program is never subject to the budget; only rewrites are.
    graph = EGraph(node_budget=1 << 30)
    root = graph.add(ENode("src", (), "fourier" if chain.source_is_fourier else "spatial"))
    if not chain.source_is_fourier:
        root = graph.add(ENode("fft", (root,)))
    for index in range(len(op_names)):
        root = graph.add(ENode("mul", (root, graph.add(ENode("xfer", (), (index,))))))
    if not spectral_output:
        root = graph.add(ENode("ifft", (root,)))
    graph.node_budget = max(int(node_budget), graph.node_count)
    record = graph.saturate(_filter_rules(op_names, compact), iterations=iterations)
    cost_fn = _filter_cost_fn(model, element_count, op_names, taps)
    chosen_expr, chosen_cost = graph.extract(root, cost_fn)
    candidates = graph.alternatives(root, cost_fn, limit=12)

    def kind_of(expr: Expr) -> str:
        return "conv" if expr[0] == "conv" else "spectral"

    def legality(expr: Expr) -> tuple[bool, str]:
        if expr[0] == "conv":
            k = kernel(tuple(expr[1]))
            if k is None:
                return False, reasons.get(tuple(expr[1]), "no compact kernel")
            if not chain.word_kernel_legal(k):
                return False, chain.word_kernel_reason(k)
            return True, "certified bound within the stated relative error budget"
        return True, ""

    chosen_text = format_expr(chosen_expr)
    equivalents = []
    seen = set()
    for expr, cost in [(chosen_expr, chosen_cost), *candidates]:
        text = format_expr(expr)
        if text in seen:
            continue
        seen.add(text)
        legal, reason = legality(expr)
        equivalents.append(FilterEquivalent(text, float(cost), text == chosen_text, legal, kind_of(expr), reason))
    if not spectral_output and not any(item.kind == "conv" for item in equivalents):
        # The rule could not fire (no compact kernel): still name the
        # convolution sibling and why it was rejected, so the report always
        # shows both forms.
        label = tuple(range(len(op_names)))
        kernel(label)
        text = f"conv[{','.join(str(i) for i in label)}](src[spatial], kern[{','.join(str(i) for i in label)}])"
        equivalents.append(FilterEquivalent(text, math.inf, False, False, "conv", reasons.get(label, "")))
    chosen = next(item for item in equivalents if item.chosen)
    chosen_kernel = kernels.get(tuple(chosen_expr[1])) if chosen.kind == "conv" else None
    return FilterSaturationPlan(
        chosen=chosen,
        equivalents=tuple(equivalents),
        record=record,
        element_count=element_count,
        cost_model=model,
        kernel=chosen_kernel,
        kernel_reason=chosen.reason if chosen.kind == "conv" else next((r for r in reasons.values() if r), ""),
    )

__all__ = [
    "BackendCapabilityError",
    "BackendUnavailableError",
    "CanonicalMetadata",
    "CompilationError",
    "CompiledProgram",
    "CompilerOptions",
    "EGraph",
    "EGraphBudgetError",
    "ENode",
    "FieldGraph",
    "FilterEquivalent",
    "FilterSaturationPlan",
    "MemorySchedule",
    "PassName",
    "PreparedMaterialization",
    "RewritePreconditionError",
    "SFRNode",
    "SaturationRecord",
    "TypeCheckError",
    "compile",
    "compile_field_graph",
    "compile_graph",
    "compile_program",
    "format_expr",
    "materialize_field",
    "prepare_materialization",
    "saturate_filter_chain",
]
