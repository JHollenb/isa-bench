"""Portable execution seam for the structured SPF v0.2 surface.

This module is intentionally a narrow vertical slice rather than a renderer
framework.  :class:`FieldIR` is an immutable, validated straight-line graph;
:class:`RenderPlan` lowers that graph to the existing tensor
``ExecutionPlan`` for the semantic oracle or through the typed compiler for
the vector CPU and cycle-modelled SPU simulator.  Physical accelerator names
remain fail-closed until their native runtimes exist, and the unprefixed
legacy tensor operations retain their historical behavior.

The operation handlers compose the already-tested primitives in
``spf.tensor.v02``: Jet2D fields, Gaussian-chirp mixtures, separable fields,
Hermitian synthesis, ordered transport, materialization, the explicit v0.2
FFT/IFFT, and log-native tone/display mapping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.tensor.op_registry import OP_REGISTRY

__all__ = [
    "FieldIR",
    "FieldKind",
    "FieldMetadata",
    "FieldNode",
    "MultiResult",
    "RenderPlan",
    "TypedField",
    "execute",
    "execute_op",
    "node",
    "render",
]


class FieldKind(StrEnum):
    """Runtime type labels carried by structured IR nodes."""

    ANY = "any"
    JET2D = "jet2d"
    ATOM = "gaussian_chirp_atom"
    ATOM_MIXTURE = "gaussian_chirp_mixture"
    SPHERICAL_GAUSSIAN = "spherical_gaussian"
    SEPARABLE = "separable_field"
    HERMITIAN = "hermitian_field"
    TRANSPORT = "transport"
    WORDS = "spf32_words"
    SPF_COMPLEX = "spf_complex"
    RADIANCE = "radiance"
    INCOHERENT_RADIANCE = "incoherent_radiance"
    LOG_POS = "log_pos"
    LOG_SIGNED = "log_signed"
    PHASE = "phase"
    OPTICAL_DEPTH = "optical_depth"
    TROPICAL_MIN = "tropical_min"
    TROPICAL_MAX = "tropical_max"
    REDUCTION = "reduction_result"
    DENSE = "dense"
    DISPLAY = "display_code"
    MATERIALIZATION_RESULT = "materialization_result"
    MULTI_RESULT = "multi_result"
    COMPLEX_PAIR = "complex_pair"
    MIXTURE_FACTOR = "mixture_factor"


_KIND_ALIASES = {
    "jet": FieldKind.JET2D,
    "jet_2d": FieldKind.JET2D,
    "atom_mixture": FieldKind.ATOM_MIXTURE,
    "mixture": FieldKind.ATOM_MIXTURE,
    "sg": FieldKind.SPHERICAL_GAUSSIAN,
    "spherical_gaussian_atom": FieldKind.SPHERICAL_GAUSSIAN,
    "separable": FieldKind.SEPARABLE,
    "transport_map": FieldKind.TRANSPORT,
    "word": FieldKind.WORDS,
    "words": FieldKind.WORDS,
    "spf": FieldKind.SPF_COMPLEX,
    "spf32": FieldKind.SPF_COMPLEX,
    "spf_complex": FieldKind.SPF_COMPLEX,
    "complex": FieldKind.SPF_COMPLEX,
    "coherent_complex": FieldKind.SPF_COMPLEX,
    "radiance": FieldKind.RADIANCE,
    "incoherent_radiance": FieldKind.INCOHERENT_RADIANCE,
    "positive_radiance": FieldKind.RADIANCE,
    "depth": FieldKind.OPTICAL_DEPTH,
    "optical_depth": FieldKind.OPTICAL_DEPTH,
    "tropical": FieldKind.TROPICAL_MIN,
    "tropical_min": FieldKind.TROPICAL_MIN,
    "tropical_max": FieldKind.TROPICAL_MAX,
    "reduction": FieldKind.REDUCTION,
    "display": FieldKind.DISPLAY,
    "pair": FieldKind.MULTI_RESULT,
    "multi_result": FieldKind.MULTI_RESULT,
}


def _kind(value: FieldKind | str) -> FieldKind:
    if isinstance(value, FieldKind):
        return value
    name = str(value).strip().lower().replace("-", "_").split(".")[-1]
    alias = _KIND_ALIASES.get(name)
    return alias if alias is not None else FieldKind(name)


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """Semantic metadata carried alongside a typed field boundary."""

    domain: str = "CUSTOM"
    semantic: str = "GENERIC_FIELD"
    coherence: str = "NOT_APPLICABLE"

    def __post_init__(self) -> None:
        for name in ("domain", "semantic", "coherence"):
            value = str(getattr(self, name)).strip().upper().replace("-", "_").replace(" ", "_")
            object.__setattr__(self, name, value)


_DEFAULT_METADATA: dict[FieldKind, FieldMetadata] = {
    FieldKind.WORDS: FieldMetadata(),
    FieldKind.SPF_COMPLEX: FieldMetadata(semantic="COMPLEX_OPTICAL_AMPLITUDE", coherence="COHERENT_COMPLEX"),
    FieldKind.RADIANCE: FieldMetadata(semantic="RADIANCE", coherence="INCOHERENT_POSITIVE"),
    FieldKind.INCOHERENT_RADIANCE: FieldMetadata(semantic="RADIANCE", coherence="INCOHERENT_POSITIVE"),
    FieldKind.LOG_POS: FieldMetadata(coherence="INCOHERENT_POSITIVE"),
    FieldKind.LOG_SIGNED: FieldMetadata(coherence="SIGNED_REAL"),
    FieldKind.PHASE: FieldMetadata(semantic="PHASE", coherence="PHASE_ONLY"),
    FieldKind.OPTICAL_DEPTH: FieldMetadata(semantic="OPTICAL_DEPTH"),
    FieldKind.TROPICAL_MIN: FieldMetadata(semantic="DEPTH"),
    FieldKind.TROPICAL_MAX: FieldMetadata(semantic="BOUND"),
    FieldKind.TRANSPORT: FieldMetadata(semantic="AFFINE_TRANSPORT"),
}


def _metadata_for(
    kind: FieldKind,
    *,
    domain: str | None = None,
    semantic: str | None = None,
    coherence: str | None = None,
) -> FieldMetadata:
    default = _DEFAULT_METADATA.get(kind, FieldMetadata())
    return FieldMetadata(
        domain=default.domain if domain is None else domain,
        semantic=default.semantic if semantic is None else semantic,
        coherence=default.coherence if coherence is None else coherence,
    )


@dataclass(frozen=True, slots=True)
class TypedField:
    """A caller-supplied value with an explicit logical kind and metadata.

    A raw ``uint32`` is only a wire carrier.  Use ``TypedField(...,
    FieldKind.SPF_COMPLEX)`` or an explicit ``v02_cast``/promotion node when
    the carrier is intended to be consumed as coherent complex data.
    """

    value: Any
    kind: FieldKind | str
    domain: str | None = None
    semantic: str | None = None
    coherence: str | None = None

    def __post_init__(self) -> None:
        normalized = _kind(self.kind)
        metadata = _metadata_for(
            normalized,
            domain=self.domain,
            semantic=self.semantic,
            coherence=self.coherence,
        )
        object.__setattr__(self, "kind", normalized)
        object.__setattr__(self, "domain", metadata.domain)
        object.__setattr__(self, "semantic", metadata.semantic)
        object.__setattr__(self, "coherence", metadata.coherence)


@dataclass(frozen=True, slots=True)
class MultiResult:
    """A typed, named multi-result that cannot masquerade as one field."""

    operation: str
    values: tuple[Any, ...]
    kind: FieldKind | str = FieldKind.MULTI_RESULT
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        values = tuple(self.values)
        names = tuple(str(name) for name in self.names)
        if names and len(names) != len(values):
            raise ValueError("multi-result names must match the result count")
        if len(set(names)) != len(names):
            raise ValueError("multi-result names must be unique")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "kind", _kind(self.kind))

    def select(self, selector: int | str) -> Any:
        if isinstance(selector, str):
            if selector not in self.names:
                raise KeyError(f"{self.operation} has no result named {selector!r}")
            return self.values[self.names.index(selector)]
        index = int(selector)
        if index < 0 or index >= len(self.values):
            raise IndexError(f"{self.operation} result index {index} is out of range")
        return self.values[index]


_OP_OUTPUT_KINDS = {
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
    "v02_atom_integral": FieldKind.DENSE,
    "v02_atom_pixel_integral": FieldKind.DENSE,
    "v02_atom_derivative": FieldKind.DENSE,
    "v02_atom_hessian": FieldKind.DENSE,
    "v02_atom_bound": FieldKind.DENSE,
    "v02_atom_bound_log": FieldKind.DENSE,
    "v02_atom_to_jet": FieldKind.JET2D,
    "v02_atom_mixture": FieldKind.ATOM_MIXTURE,
    "v02_mixture": FieldKind.ATOM_MIXTURE,
    "v02_mixture_add": FieldKind.ATOM_MIXTURE,
    "v02_mixture_concat": FieldKind.ATOM_MIXTURE,
    "v02_mixture_prune": FieldKind.ATOM_MIXTURE,
    "v02_mixture_cull": FieldKind.ATOM_MIXTURE,
    "v02_mixture_bound": FieldKind.DENSE,
    "v02_mixture_factor": FieldKind.MIXTURE_FACTOR,
    "v02_mixture_to_jets": FieldKind.ANY,
    "v02_mixture_merge_equal": FieldKind.ATOM_MIXTURE,
    "v02_mixture_eval": FieldKind.DENSE,
    "v02_mixture_materialize": FieldKind.DENSE,
    "v02_mixture_lower": FieldKind.DENSE,
    "v02_separable": FieldKind.SEPARABLE,
    "v02_separable_field": FieldKind.SEPARABLE,
    "v02_separable_eval": FieldKind.REDUCTION,
    "v02_separable_materialize": FieldKind.DENSE,
    "v02_hermitian_synthesis": FieldKind.HERMITIAN,
    "v02_hermitian_synth": FieldKind.HERMITIAN,
    "v02_spherical_gaussian": FieldKind.SPHERICAL_GAUSSIAN,
    "v02_spherical_gaussian_product": FieldKind.SPHERICAL_GAUSSIAN,
    "v02_spherical_gaussian_integral": FieldKind.DENSE,
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
    "v02_complex_pair_select": FieldKind.ANY,
    "v02_phase_shift": FieldKind.SPF_COMPLEX,
    "v02_promote_complex": FieldKind.SPF_COMPLEX,
    "v02_phase_magnitude_to_complex": FieldKind.SPF_COMPLEX,
    "v02_cast": FieldKind.ANY,
    "v02_extract": FieldKind.ANY,
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
    "v02_mixture_factor_select": FieldKind.ANY,
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


_OP_INPUT_KINDS = {
    "v02_jet_add": (FieldKind.JET2D, FieldKind.JET2D),
    "v02_jet_transform": (FieldKind.JET2D,),
    "v02_jet_materialize": (FieldKind.JET2D,),
    "v02_atom_mul": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_product": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_div": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_affine": (FieldKind.ATOM,),
    "v02_atom_fourier": (FieldKind.ATOM,),
    "v02_atom_inverse_fourier": (FieldKind.ATOM,),
    "v02_atom_convolve": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_integral": (FieldKind.ATOM,),
    "v02_atom_pixel_integral": (FieldKind.ATOM, FieldKind.ATOM),
    "v02_atom_derivative": (FieldKind.ATOM, FieldKind.DENSE),
    "v02_atom_hessian": (FieldKind.ATOM, FieldKind.DENSE),
    "v02_atom_bound": (FieldKind.ATOM,),
    "v02_atom_bound_log": (FieldKind.ATOM,),
    "v02_atom_to_jet": (FieldKind.ATOM,),
    "v02_mixture_add": (FieldKind.ATOM_MIXTURE, FieldKind.ATOM_MIXTURE),
    "v02_mixture_concat": (FieldKind.ATOM_MIXTURE, FieldKind.ATOM_MIXTURE),
    "v02_mixture_prune": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_cull": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_bound": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_factor": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_to_jets": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_merge_equal": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_eval": (FieldKind.ATOM_MIXTURE, FieldKind.DENSE),
    "v02_mixture_materialize": (FieldKind.ATOM_MIXTURE,),
    "v02_mixture_lower": (FieldKind.ATOM_MIXTURE,),
    "v02_separable_eval": (FieldKind.SEPARABLE, FieldKind.DENSE),
    "v02_separable_materialize": (FieldKind.SEPARABLE,),
    "v02_spherical_gaussian_product": (
        FieldKind.SPHERICAL_GAUSSIAN,
        FieldKind.SPHERICAL_GAUSSIAN,
    ),
    "v02_spherical_gaussian_integral": (FieldKind.SPHERICAL_GAUSSIAN,),
    "v02_transport_compose": (FieldKind.TRANSPORT, FieldKind.TRANSPORT),
    "v02_transport_reduce": (FieldKind.TRANSPORT,),
    "v02_transport_apply": (FieldKind.TRANSPORT, FieldKind.ANY),
    "v02_logadd_pos": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_logsub_pos": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_logadd_signed": (FieldKind.LOG_SIGNED, FieldKind.LOG_SIGNED),
    "v02_log_mul": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_log_div": (FieldKind.LOG_POS, FieldKind.LOG_POS),
    "v02_phase_add": (FieldKind.PHASE, FieldKind.PHASE),
    "v02_phase_sub": (FieldKind.PHASE, FieldKind.PHASE),
    "v02_phase_conj": (FieldKind.PHASE,),
    "v02_phase_compose": (FieldKind.PHASE, FieldKind.PHASE),
    "v02_complex_mul": (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
    "v02_complex_div": (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
    "v02_complex_pair_add": (FieldKind.SPF_COMPLEX, FieldKind.SPF_COMPLEX),
    "v02_phase_shift": (FieldKind.SPF_COMPLEX,),
    "v02_promote_complex": (),
    "v02_phase_magnitude_to_complex": (FieldKind.LOG_POS, FieldKind.PHASE),
    "v02_cast": (),
    "v02_extract": (FieldKind.MULTI_RESULT,),
    "v02_complex_pair_select": (FieldKind.COMPLEX_PAIR,),
    "v02_mixture_factor_select": (FieldKind.MIXTURE_FACTOR,),
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
    "v02_visibility_select": (FieldKind.ANY, FieldKind.DENSE),
    "v02_mask": (FieldKind.ANY, FieldKind.DENSE),
    "v02_coordinate_remap": (FieldKind.DENSE,),
    "v02_bandlimit": (FieldKind.ANY,),
    "v02_tile_cull": (FieldKind.ANY,),
    "v02_sort": (FieldKind.DENSE,),
    "v02_materialize": (FieldKind.ANY,),
    "v02_materialize_tile": (FieldKind.ANY,),
}


# These are semantic boundaries in the structured algebra even though the
# reference executor does not attach the legacy ``ReducerSpec`` object to a
# v02 ``ExecutionStep``.  Keeping them explicit prevents a plan consumer from
# mistaking a reference reduction or wire-format write for an ordinary unary
# node.  Native backends can later replace these metadata entries with their
# compiled reducer/quantizer implementations without changing the IR.
_STRUCTURED_REDUCER_OPS = frozenset(
    {
        "v02_mixture_eval",
        "v02_mixture_materialize",
        "v02_mixture_lower",
        "v02_mixture_bound",
        "v02_separable_eval",
        "v02_separable_materialize",
        "v02_hermitian_synthesis",
        "v02_sparse_synthesis",
        "v02_reduce_positive",
        "v02_reduce_signed",
        "v02_reduce_complex",
        "v02_reduce_optical_depth",
        "v02_reduce_transport",
        "v02_reduce_tropical",
        "v02_depth_min",
        "v02_transport_reduce",
    }
)

_STRUCTURED_REQUANTIZATION_OPS = frozenset(
    {
        "v02_jet_materialize",
        "v02_mixture_materialize",
        "v02_mixture_lower",
        "v02_separable_materialize",
        "v02_complex_mul",
        "v02_complex_div",
        "v02_complex_pair_add",
        "v02_phase_shift",
        "v02_abs2",
        "v02_fft",
        "v02_fft_forward",
        "v02_ifft",
        "v02_fft_inverse",
        "v02_sparse_synthesis",
        "v02_materialize",
        "v02_materialize_tile",
        "v02_display",
        "v02_display_convert",
    }
)


def _inferred_output_kind(operation: str, params: Mapping[str, Any]) -> FieldKind:
    """Infer a conservative output kind when a node omits one explicitly."""

    output = _OP_OUTPUT_KINDS.get(operation, FieldKind.ANY)
    if operation in {
        "v02_jet_materialize",
        "v02_mixture_materialize",
        "v02_mixture_lower",
        "v02_separable_materialize",
        "v02_materialize",
        "v02_materialize_tile",
    } and bool(params.get("return_result", params.get("return_certificate", False))):
        return FieldKind.MATERIALIZATION_RESULT
    if operation == "v02_transport_reduce" and bool(params.get("return_result", False)):
        return FieldKind.REDUCTION
    if operation == "v02_mixture_eval":
        declared_kind = params.get("value_kind", params.get("kind"))
        if declared_kind is not None:
            name = str(getattr(declared_kind, "value", declared_kind)).lower().replace("-", "_")
            if name in {"spf", "spf32", "spf_complex", "complex", "words", "word", "uint32"}:
                return FieldKind.SPF_COMPLEX
            if name in {"native", "default", "dense", "complex128"}:
                return FieldKind.DENSE
    if output is FieldKind.ANY and operation in {
        "v02_field_constant",
        "v02_field_asset",
        "v02_field_reference",
    }:
        declared_kind = params.get("value_kind", params.get("kind"))
        if declared_kind is not None:
            return _kind(declared_kind)
    if operation in {"v02_cast", "v02_promote_complex", "v02_phase_magnitude_to_complex"}:
        target = params.get("to_kind", params.get("target_kind"))
        if target is not None:
            return _kind(target)
    if operation in {
        "v02_extract",
        "v02_complex_pair_select",
        "v02_mixture_factor_select",
    }:
        selector = params.get("select", params.get("which", params.get("index")))
        source_kind = str(params.get("source_kind", "")).lower().replace("-", "_")
        if operation == "v02_mixture_factor_select" or source_kind in {
            "mixture_factor",
            "factor",
        }:
            if selector in {0, "0", "factor", "left"}:
                return FieldKind.ATOM
            if selector in {1, "1", "residual", "right"}:
                return FieldKind.ATOM_MIXTURE
        if operation == "v02_complex_pair_select" or source_kind in {
            "complex_pair",
            "pair",
        }:
            if selector in {0, "0", "sum", "plus", "left"}:
                return FieldKind.SPF_COMPLEX
            if selector in {1, "1", "difference", "diff", "minus", "right"}:
                return FieldKind.SPF_COMPLEX
    raw_format = getattr(params.get("output_format"), "value", params.get("output_format"))
    if raw_format is None or output in {FieldKind.COMPLEX_PAIR, FieldKind.MIXTURE_FACTOR}:
        return output
    name = str(raw_format).lower().replace("-", "_")
    if name in {"spf_complex", "spf", "spf32", "word", "words", "uint32"}:
        return FieldKind.WORDS
    if name in {"log_pos", "logpos", "positive", "positive_real"}:
        return FieldKind.LOG_POS
    if name in {"log_signed", "logsigned", "signed", "signed_real"}:
        return FieldKind.LOG_SIGNED
    if name in {"phase", "phase_only"}:
        return FieldKind.PHASE
    if name in {"native", "default", "complex", "complex128", "dense", "real", "float", "float64"}:
        return FieldKind.DENSE
    return output


def _operation_input_kinds(
    operation: str,
    input_count: int,
    params: Mapping[str, Any],
    explicit: Sequence[FieldKind | str] = (),
) -> tuple[FieldKind, ...]:
    """Return the operation's canonical semantic input contract.

    ``input_kinds`` is a declaration on an edge, not a way to replace the
    operation's algebra.  Build the canonical contract first, then allow an
    explicit declaration only to specialize an ``ANY`` slot (or to name one
    of the small, explicitly supported constructor overloads).
    """

    if operation in {"v02_field_constant", "v02_field_asset", "v02_field_reference"}:
        if input_count > 1:
            raise ValueError(f"{operation} accepts at most one input")
        canonical = (FieldKind.ANY,) * input_count
    elif operation in {"v02_jet2d", "v02_atom_create"}:
        if input_count > 1:
            raise ValueError(f"{operation} accepts at most one input")
        canonical_kind = FieldKind.JET2D if operation == "v02_jet2d" else FieldKind.ATOM
        canonical = (canonical_kind,) * input_count
    elif operation in {"v02_atom_mixture", "v02_mixture"}:
        # The constructor accepts one or more atoms, or no graph operands
        # when ``params['atoms']`` supplies them.  A one-input AtomMixture
        # passthrough is handled as an explicit constructor overload below.
        canonical = (FieldKind.ATOM,) * input_count
    elif operation in {"v02_separable", "v02_separable_field"}:
        if input_count > 1:
            raise ValueError(f"{operation} accepts at most one input")
        canonical = (FieldKind.ANY,) * input_count
    elif operation in {"v02_hermitian_synthesis", "v02_hermitian_synth"}:
        if input_count > 2:
            raise ValueError(f"{operation} accepts at most two inputs")
        canonical = () if input_count == 0 else (FieldKind.ANY,) if input_count == 1 else (FieldKind.ANY, FieldKind.ANY)
    elif operation == "v02_spherical_gaussian":
        if input_count > 3:
            raise ValueError(f"{operation} accepts at most three inputs")
        if input_count == 0:
            canonical = ()
        elif input_count == 1:
            canonical = (FieldKind.SPHERICAL_GAUSSIAN,)
        else:
            canonical = (FieldKind.DENSE,) * input_count
    elif operation == "v02_transport_identity":
        if input_count:
            raise ValueError(f"{operation} does not accept graph inputs")
        canonical = ()
    elif operation == "v02_transport":
        if input_count > 2:
            raise ValueError(f"{operation} accepts at most two inputs")
        # Transport components may be log-native, packed, or dense.  Their
        # representation family is checked by the Transport implementation;
        # the one-input overload is the typed Transport passthrough.
        canonical = (
            () if input_count == 0 else (FieldKind.TRANSPORT,) if input_count == 1 else (FieldKind.ANY, FieldKind.ANY)
        )
    elif operation == "v02_phase_shift":
        if input_count not in {1, 2}:
            raise ValueError(f"{operation} requires one SPF input and an optional PHASE input")
        canonical = (
            (FieldKind.SPF_COMPLEX,)
            if input_count == 1
            else (
                FieldKind.SPF_COMPLEX,
                FieldKind.PHASE,
            )
        )
    elif operation in {"v02_promote_complex", "v02_phase_magnitude_to_complex"}:
        if input_count not in {0, 1, 2}:
            raise ValueError(f"{operation} accepts zero, one, or two inputs")
        magnitude_kind = _kind(params.get("magnitude_kind", "log_pos"))
        canonical = (
            () if input_count == 0 else (magnitude_kind,) if input_count == 1 else (magnitude_kind, FieldKind.PHASE)
        )
    elif operation == "v02_cast":
        if input_count != 1:
            raise ValueError("v02_cast requires exactly one input")
        source_kind = params.get("from_kind")
        canonical = (_kind(source_kind),) if source_kind is not None else (FieldKind.ANY,)
    elif operation in {
        "v02_reduce_positive",
        "v02_reduce_signed",
        "v02_reduce_complex",
        "v02_reduce_optical_depth",
        "v02_reduce_transport",
        "v02_depth_min",
        "v02_reduce_tropical",
        "v02_transport_reduce",
    }:
        if input_count < 1:
            raise ValueError(f"{operation} requires at least one input")
        if operation == "v02_reduce_tropical" or operation == "v02_depth_min":
            mode = str(params.get("kind", params.get("mode", "tropical_min")))
            reduction_kind = (
                FieldKind.TROPICAL_MAX
                if mode.lower().replace("-", "_") in {"max", "tropical_max"}
                else FieldKind.TROPICAL_MIN
            )
        else:
            reduction_kind = {
                "v02_reduce_positive": FieldKind.LOG_POS,
                "v02_reduce_signed": FieldKind.LOG_SIGNED,
                "v02_reduce_complex": FieldKind.SPF_COMPLEX,
                "v02_reduce_optical_depth": FieldKind.OPTICAL_DEPTH,
                "v02_reduce_transport": FieldKind.TRANSPORT,
                "v02_transport_reduce": FieldKind.TRANSPORT,
            }[operation]
        canonical = (reduction_kind,) * input_count
    elif operation == "v02_support_intersect":
        if input_count < 1:
            raise ValueError("v02_support_intersect requires at least one input")
        canonical = (FieldKind.DENSE,) * input_count
    elif operation == "v02_sparse_synthesis":
        if input_count != 2:
            raise ValueError("v02_sparse_synthesis requires exactly two inputs")
        canonical = (FieldKind.DENSE, FieldKind.ANY)
    else:
        declared = _OP_INPUT_KINDS.get(operation)
        if declared is None:
            # The registry is extensible.  An unlisted v02 operation remains
            # an untyped reference extension, but its edges are still
            # represented one-for-one instead of silently disappearing.
            canonical = (FieldKind.ANY,) * input_count
        else:
            if len(declared) != input_count:
                raise ValueError(f"{operation} requires exactly {len(declared)} inputs; got {input_count}")
            canonical = declared

    if not explicit:
        return canonical
    declared_inputs = tuple(_kind(value) for value in explicit)
    if len(declared_inputs) != input_count:
        raise ValueError("input_kinds must match the node input count")
    resolved: list[FieldKind] = []
    for position, (declared_kind, expected_kind) in enumerate(zip(declared_inputs, canonical, strict=True)):
        if not _input_kind_compatible(operation, declared_kind, expected_kind, position, input_count):
            raise TypeError(
                f"{operation} input {position} declares {declared_kind.value}, but requires {expected_kind.value}"
            )
        resolved.append(declared_kind if expected_kind is FieldKind.ANY else expected_kind)
    return tuple(resolved)


def _expected_input_kinds(item: FieldNode) -> tuple[FieldKind, ...]:
    """Return the canonical input contract for a node."""

    return _operation_input_kinds(item.op, len(item.inputs), item.params, item.input_kinds)


def _kind_compatible(actual: FieldKind, expected: FieldKind) -> bool:
    """Check the small v0.2 type lattice without implicit algebra changes."""

    # ANY is a declaration of uncertainty, not a universal subtype.  It is
    # safe only when the consumer explicitly accepts any value; accepting an
    # unknown producer at a typed boundary would reinterpret storage as a
    # semantic algebra.
    if expected is FieldKind.ANY:
        return True
    if actual is FieldKind.ANY:
        return False
    if actual is expected:
        return True
    if expected is FieldKind.LOG_POS and actual in {
        FieldKind.RADIANCE,
        FieldKind.INCOHERENT_RADIANCE,
    }:
        return True
    # A semantic complex value may be consumed by a wire-format operation,
    # but an untyped wire word may not enter a coherent complex operation.
    if expected is FieldKind.WORDS and actual is FieldKind.SPF_COMPLEX:
        return True
    if expected is FieldKind.MULTI_RESULT and actual in {
        FieldKind.COMPLEX_PAIR,
        FieldKind.MIXTURE_FACTOR,
    }:
        return True
    # Dense numeric arrays are the carrier for coordinates, masks, atom
    # bounds, and reference scalar outputs.  DISPLAY is intentionally a
    # terminal numeric kind, so an object/numeric array is accepted there.
    if expected is FieldKind.DISPLAY and actual is FieldKind.DENSE:
        return True
    if expected is FieldKind.DISPLAY and actual is FieldKind.WORDS:
        return True
    return expected is FieldKind.HERMITIAN and actual in {
        FieldKind.DENSE,
        FieldKind.LOG_POS,
        FieldKind.LOG_SIGNED,
        FieldKind.SPF_COMPLEX,
    }


def _input_kind_compatible(
    operation: str,
    actual: FieldKind,
    expected: FieldKind,
    position: int,
    input_count: int,
) -> bool:
    """Check one operation edge, including its explicit constructor overloads."""

    if _kind_compatible(actual, expected):
        return True
    # ``v02_mixture`` is a constructor from atoms, but its implementation
    # also supports a one-input typed-mixture passthrough.  Keep that
    # overload explicit and bounded so a Jet or dense field can never be
    # smuggled into the mixture algebra.
    return (
        operation in {"v02_atom_mixture", "v02_mixture"}
        and input_count == 1
        and position == 0
        and expected is FieldKind.ATOM
        and actual is FieldKind.ATOM_MIXTURE
    )


def _runtime_kind(value: Any) -> FieldKind:
    """Classify a runtime value for the renderer's boundary checks."""

    if isinstance(value, TypedField):
        return cast(FieldKind, value.kind)
    if isinstance(value, MultiResult):
        return cast(FieldKind, value.kind)
    # Storage width does not determine algebra. The intensity observable is
    # positive even though its source is still stored as SPF complex words.
    try:
        from spf.renderer.blocks import BlockView

        if isinstance(value, BlockView):
            return FieldKind.LOG_POS if value.observable == "intensity" else FieldKind.SPF_COMPLEX
    except ImportError:
        pass
    implementation = _impl()
    if isinstance(value, implementation.MaterializationResult):
        return FieldKind.MATERIALIZATION_RESULT
    if isinstance(value, implementation.ReductionResult):
        return FieldKind.REDUCTION
    if isinstance(value, implementation.Jet2D):
        return FieldKind.JET2D
    if isinstance(value, implementation.GaussianChirpAtom):
        return FieldKind.ATOM
    if isinstance(value, implementation.AtomMixture):
        return FieldKind.ATOM_MIXTURE
    if isinstance(value, implementation.SphericalGaussian):
        return FieldKind.SPHERICAL_GAUSSIAN
    if isinstance(value, implementation.SeparableField):
        return FieldKind.SEPARABLE
    if isinstance(value, implementation.Transport):
        return FieldKind.TRANSPORT
    if isinstance(value, implementation.LogPos):
        return FieldKind.LOG_POS
    if isinstance(value, implementation.LogSigned):
        return FieldKind.LOG_SIGNED
    if isinstance(value, implementation.Phase):
        return FieldKind.PHASE
    if isinstance(value, implementation.SphericalGaussian):
        return FieldKind.DENSE
    if isinstance(value, implementation.SPF32Value):
        return FieldKind.SPF_COMPLEX
    if isinstance(value, np.ndarray):
        if value.dtype == np.uint32:
            return FieldKind.WORDS
        if value.dtype == object and value.size:
            flattened = value.reshape(-1)
            first = _runtime_kind(flattened[0])
            if all(_kind_compatible(_runtime_kind(item), first) for item in flattened[1:]):
                return first
        if np.issubdtype(value.dtype, np.number) or value.dtype == bool:
            return FieldKind.DENSE
        return FieldKind.ANY
    if isinstance(value, (list, tuple)):
        if not value:
            return FieldKind.ANY
        kinds = [_runtime_kind(item) for item in value]
        return kinds[0] if all(_kind_compatible(item, kinds[0]) for item in kinds[1:]) else FieldKind.ANY
    if isinstance(value, (int, np.integer)):
        return FieldKind.WORDS
    if isinstance(value, (float, complex, np.number, bool)):
        return FieldKind.DENSE
    return FieldKind.ANY


def _require_kind(value: Any, expected: FieldKind, *, context: str) -> None:
    actual = _runtime_kind(value)
    if not _kind_compatible(actual, expected):
        raise TypeError(f"{context} requires {expected.value}, received {actual.value}")


_SPF_WORD_RESULT_OPS = frozenset(
    {
        "v02_cast",
        "v02_promote_complex",
        "v02_phase_magnitude_to_complex",
        "v02_phase_shift",
        "v02_complex_mul",
        "v02_complex_div",
        "v02_complex_pair_select",
        "v02_extract",
        "v02_mixture_eval",
    }
)


def _output_kind_compatible(actual: FieldKind, expected: FieldKind, operation: str) -> bool:
    """Allow a declared semantic SPF value to retain its uint32 carrier."""

    if _kind_compatible(actual, expected):
        return True
    return expected is FieldKind.SPF_COMPLEX and actual is FieldKind.WORDS and operation in _SPF_WORD_RESULT_OPS


def _validate_execution_output(operation: str, value: Any, options: Mapping[str, Any]) -> None:
    """Validate a lowered step's declared output at the VM boundary."""

    declared = options.get("_declared_value_kind")
    if declared is None:
        return
    expected = _kind(declared)
    if expected is FieldKind.ANY:
        return
    actual = _runtime_kind(value)
    if not _output_kind_compatible(actual, expected, operation):
        raise TypeError(f"{operation} produced {actual.value}, declared {expected.value}")


def _validate_execute_inputs(
    operation: str,
    operands: Sequence[Any],
    options: Mapping[str, Any],
) -> None:
    """Guard direct executor calls as well as values entering the VM."""

    expected = _operation_input_kinds(
        operation,
        len(operands),
        options,
        tuple(options.get("_input_kinds", ())),
    )
    # ``_input_kinds`` is the consumer contract.  It is not evidence that an
    # external value has that kind.  Only a producer declaration carried in
    # ``_input_value_kinds`` can authorize a raw SPF32 carrier at a coherent
    # edge.
    trusted = tuple(options.get("_input_value_kinds", ()))
    for position, expected_kind in enumerate(expected):
        if position >= len(operands) or expected_kind is FieldKind.ANY:
            continue
        actual = _runtime_kind(operands[position])
        if _input_kind_compatible(operation, actual, expected_kind, position, len(operands)):
            continue
        # A plan's producer declaration is the semantic proof that a raw
        # uint32 carrier is coherent SPF data.  Direct calls have no such
        # proof and therefore reject the same carrier.
        if not (
            expected_kind is FieldKind.SPF_COMPLEX
            and actual is FieldKind.WORDS
            and position < len(trusted)
            and _kind(trusted[position]) is FieldKind.SPF_COMPLEX
        ):
            raise TypeError(f"{operation} input {position} requires {expected_kind.value}, received {actual.value}")


def _params(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class FieldNode:
    """One typed v0.2 operation in a topologically ordered field graph."""

    op: str
    inputs: tuple[str, ...] = ()
    output: str = ""
    value_kind: FieldKind | str = FieldKind.ANY
    input_kinds: tuple[FieldKind | str, ...] = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    domain: str | None = None
    semantic: str | None = None
    coherence: str | None = None

    def __post_init__(self) -> None:
        operation = str(self.op).strip()
        if operation.lower() in {"add", "v02_add"}:
            raise ValueError(
                "generic untyped Add is not legal in SPF v0.2 FieldIR; "
                "use a typed operation such as v02_mixture_concat, "
                "v02_logadd_pos, or v02_complex_pair_add"
            )
        if not operation.startswith("v02_"):
            raise ValueError("structured renderer nodes must use a v02_* operation")
        if operation not in OP_REGISTRY:
            raise ValueError(f"unregistered structured renderer op {operation!r}")
        if not str(self.output):
            raise ValueError("structured renderer nodes require a non-empty output name")
        input_names = tuple(str(name) for name in self.inputs)
        if any(not name for name in input_names):
            raise ValueError("structured renderer input names must be non-empty")
        input_kinds = tuple(_kind(value) for value in self.input_kinds)
        if input_kinds and len(input_kinds) != len(input_names):
            raise ValueError("input_kinds must match the node input count")
        parameters = _params(self.params)
        # Validate arity and any explicit edge declarations while the node is
        # being built.  FieldIR repeats the canonical check when it resolves
        # producer-to-consumer edges, so a standalone FieldNode cannot carry
        # an invalid contract into a later graph.
        _operation_input_kinds(operation, len(input_names), parameters, input_kinds)
        explicit_output = _kind(self.value_kind)
        inferred_output = _inferred_output_kind(operation, parameters)
        if operation in {"v02_complex_pair_add", "v02_mixture_factor"} and explicit_output not in {
            FieldKind.ANY,
            inferred_output,
        }:
            raise TypeError(
                f"{operation} is multi-result and cannot be declared as "
                f"{explicit_output.value}; use an explicit extract node"
            )
        # A node may omit its output kind and inherit the operation result,
        # but it may not relabel a known result as an unrelated algebra.
        if (
            explicit_output is not FieldKind.ANY
            and inferred_output is not FieldKind.ANY
            and not _output_kind_compatible(inferred_output, explicit_output, operation)
        ):
            raise TypeError(f"{operation} produces {inferred_output.value}, but declares {explicit_output.value}")
        declared_output = inferred_output if explicit_output is FieldKind.ANY else explicit_output
        metadata = _metadata_for(
            declared_output,
            domain=self.domain,
            semantic=self.semantic,
            coherence=self.coherence,
        )
        object.__setattr__(self, "op", operation)
        object.__setattr__(self, "inputs", input_names)
        object.__setattr__(self, "output", str(self.output))
        object.__setattr__(self, "value_kind", declared_output)
        object.__setattr__(self, "input_kinds", input_kinds)
        object.__setattr__(self, "params", parameters)
        object.__setattr__(self, "domain", metadata.domain)
        object.__setattr__(self, "semantic", metadata.semantic)
        object.__setattr__(self, "coherence", metadata.coherence)

    @property
    def metadata(self) -> FieldMetadata:
        return FieldMetadata(
            self.domain or "CUSTOM",
            self.semantic or "GENERIC_FIELD",
            self.coherence or "NOT_APPLICABLE",
        )


def node(
    op: str,
    output: str,
    *inputs: str,
    value_kind: FieldKind | str = FieldKind.ANY,
    input_kinds: Sequence[FieldKind | str] = (),
    params: Mapping[str, Any] | None = None,
    domain: str | None = None,
    semantic: str | None = None,
    coherence: str | None = None,
    **operation_params: Any,
) -> FieldNode:
    """Convenience constructor for a typed :class:`FieldNode`."""

    merged_params = dict(params or {})
    merged_params.update(operation_params)
    return FieldNode(
        op,
        tuple(inputs),
        output,
        value_kind,
        tuple(input_kinds),
        merged_params,
        domain,
        semantic,
        coherence,
    )


@dataclass(frozen=True, slots=True)
class FieldIR:
    """Immutable, validated, topologically ordered structured field IR."""

    nodes: tuple[FieldNode, ...] = ()
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_items: list[FieldNode] = []
        for item in self.nodes:
            normalized_items.append(item if isinstance(item, FieldNode) else FieldNode(**dict(item)))
        normalized_nodes = tuple(normalized_items)
        declared_inputs = tuple(str(name) for name in self.inputs)
        if len(set(declared_inputs)) != len(declared_inputs) or any(not name for name in declared_inputs):
            raise ValueError("FieldIR inputs must be unique non-empty names")

        all_outputs: dict[str, int] = {}
        for index, item in enumerate(normalized_nodes):
            if item.output in all_outputs:
                raise ValueError(f"duplicate FieldIR output {item.output!r}")
            all_outputs[item.output] = index
        if set(declared_inputs) & set(all_outputs):
            raise ValueError("FieldIR input and output names must be disjoint")

        produced: set[str] = set()
        produced_kinds: dict[str, FieldKind] = {}
        inferred_inputs = list(declared_inputs)
        declared_set = set(declared_inputs)
        for index, item in enumerate(normalized_nodes):
            for name in item.inputs:
                if name in produced or name in declared_set:
                    continue
                if name in all_outputs and all_outputs[name] >= index:
                    raise ValueError(f"FieldIR input {name!r} is a forward reference")
                inferred_inputs.append(name)
                declared_set.add(name)
            expected_kinds = _expected_input_kinds(item)
            for position, name in enumerate(item.inputs):
                if position >= len(expected_kinds):
                    continue
                expected = _kind(expected_kinds[position])
                # Names not produced by this graph are external inputs.  Their
                # runtime carrier is checked by ``validate_values``; treating
                # them as ANY here would accidentally make an unknown producer
                # satisfy a typed edge.  Produced ANY values, however, are
                # rejected immediately because they lack semantic metadata.
                actual = produced_kinds.get(name)
                if actual is None:
                    continue
                if not _input_kind_compatible(item.op, actual, expected, position, len(item.inputs)):
                    raise TypeError(
                        f"FieldIR node {item.op!r} input {position} ({name!r}) "
                        f"requires {expected.value}, received {actual.value}"
                    )
            produced.add(item.output)
            produced_kinds[item.output] = cast(FieldKind, item.value_kind)

        normalized_outputs = tuple(str(name) for name in self.outputs)
        if not normalized_outputs and normalized_nodes:
            normalized_outputs = (normalized_nodes[-1].output,)
        if any(name not in produced for name in normalized_outputs):
            raise ValueError("FieldIR outputs must be produced by a node")
        if len(set(normalized_outputs)) != len(normalized_outputs):
            raise ValueError("FieldIR outputs must be unique")
        object.__setattr__(self, "nodes", normalized_nodes)
        object.__setattr__(self, "inputs", tuple(inferred_inputs))
        object.__setattr__(self, "outputs", normalized_outputs)

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[FieldNode | Mapping[str, Any]],
        *,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
    ) -> FieldIR:
        normalized = tuple(item if isinstance(item, FieldNode) else FieldNode(**dict(item)) for item in nodes)
        return cls(normalized, tuple(inputs), tuple(outputs))

    def to_execution_plan(self) -> Any:
        """Lower to the existing tensor VM without changing legacy defaults."""

        from spf.tensor.v2 import ExecutionPlan, ExecutionStep

        produced_kinds: dict[str, FieldKind] = {}
        lowered_steps: list[ExecutionStep] = []
        for item in self.nodes:
            output_kind = cast(FieldKind, item.value_kind)
            expected = _expected_input_kinds(item)
            input_value_kinds = tuple(produced_kinds.get(name, FieldKind.ANY).value for name in item.inputs)
            lowered_steps.append(
                ExecutionStep(
                    item.op,
                    item.inputs,
                    (item.output,),
                    params={
                        **dict(item.params),
                        "_input_kinds": tuple(expected_kind.value for expected_kind in expected),
                        "_input_value_kinds": input_value_kinds,
                        "_declared_value_kind": output_kind.value,
                        "_declared_domain": item.domain,
                        "_declared_semantic": item.semantic,
                        "_declared_coherence": item.coherence,
                    },
                )
            )
            produced_kinds[item.output] = output_kind
        steps = tuple(lowered_steps)
        reducer_boundaries = tuple(index for index, item in enumerate(self.nodes) if item.op in _STRUCTURED_REDUCER_OPS)
        requantization_points = tuple(
            index
            for index, item in enumerate(self.nodes)
            if item.op in _STRUCTURED_REQUANTIZATION_OPS or bool(item.params.get("requantize", False))
        )
        reducer_specs = tuple(
            {
                "index": index,
                "op": item.op,
                "output": item.output,
                "value_kind": cast(FieldKind, item.value_kind).value,
                "spec": item.params.get("spec"),
            }
            for index, item in enumerate(self.nodes)
            if index in reducer_boundaries
        )
        return ExecutionPlan(
            steps=steps,
            factorization_metadata={
                "structured_renderer": "v02_reference",
                "backend": "reference",
                "reference_only": True,
                "deployable": False,
                "capability": "reference_only",
                "capabilities": ("reference_executor", "typed_field_ir"),
                "reference_executor": "spf.renderer.v02.execute_op",
                "lowering": "v02_reference_execution_plan",
                "field_inputs": self.inputs,
                "field_outputs": self.outputs,
                "field_kinds": {item.output: cast(FieldKind, item.value_kind).value for item in self.nodes},
                "field_metadata": {
                    item.output: {
                        "domain": item.domain,
                        "semantic": item.semantic,
                        "coherence": item.coherence,
                    }
                    for item in self.nodes
                },
                "structured_reducer_boundaries": reducer_boundaries,
                "structured_requantization_points": requantization_points,
                "structured_reducer_specs": reducer_specs,
            },
        )

    def lower(self, backend: str = "reference") -> Any:
        """Lower this IR through the declared backend boundary.

        Reference lowering returns the small tensor ``ExecutionPlan``.  The
        vector CPU and functional SPU simulator lower through the full typed
        compiler.  Physical accelerator targets remain explicit failures so
        a caller cannot mistake reference execution for native execution.
        """

        name = str(backend).lower().replace("-", "_")
        if name in {"reference", "python", "cpu_reference"}:
            return self.to_execution_plan()
        if name in {"cpu", "numpy", "spu_sim", "spu_simulator"}:
            from spf.renderer.compiler import compile_graph

            return compile_graph(self, backend=name)
        raise NotImplementedError(
            f"SPF v0.2 backend {backend!r} is not implemented as an executable lowering; "
            "use 'reference', 'cpu', or 'spu_sim'"
        )

    def validate_values(self, values: Mapping[str, Any]) -> None:
        """Validate external values against the IR's input-kind contracts."""

        external_inputs = set(self.inputs)
        for item in self.nodes:
            expected_kinds = _expected_input_kinds(item)
            for position, name in enumerate(item.inputs):
                if name not in external_inputs or name not in values or position >= len(expected_kinds):
                    continue
                expected = _kind(expected_kinds[position])
                if expected is not FieldKind.ANY:
                    _require_kind(values[name], expected, context=f"{item.op} input {position}")


@dataclass(frozen=True, slots=True)
class RenderPlan:
    """Validated v0.2 render plan backed by :class:`FieldIR`."""

    ir: FieldIR
    outputs: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.ir, FieldIR):
            raise TypeError("RenderPlan.ir must be a FieldIR")
        selected = tuple(str(name) for name in self.outputs) or self.ir.outputs
        if any(name not in self.ir.outputs for name in selected):
            raise ValueError("RenderPlan outputs must be declared by its FieldIR")
        if not selected:
            raise ValueError("RenderPlan requires at least one output")
        object.__setattr__(self, "outputs", selected)
        object.__setattr__(self, "metadata", _params(self.metadata))

    @classmethod
    def from_nodes(
        cls,
        nodes: Sequence[FieldNode | Mapping[str, Any]],
        *,
        inputs: Sequence[str] = (),
        outputs: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RenderPlan:
        return cls(FieldIR.from_nodes(nodes, inputs=inputs, outputs=outputs), metadata=metadata or {})

    def to_execution_plan(self) -> Any:
        return self.lower()

    def lower(self, backend: str = "reference") -> Any:
        """Lower the plan through the explicit v0.2 backend boundary."""

        name = str(backend).lower().replace("-", "_")
        if name in {"cpu", "numpy", "spu_sim", "spu_simulator"}:
            from spf.renderer.compiler import compile_graph

            return compile_graph(self, backend=name)
        plan = self.ir.lower(backend)
        if self.metadata:
            protected = {
                key: plan.factorization_metadata[key]
                for key in (
                    "backend",
                    "reference_only",
                    "deployable",
                    "capability",
                    "capabilities",
                    "reference_executor",
                    "lowering",
                )
                if key in plan.factorization_metadata
            }
            plan.factorization_metadata.update(dict(self.metadata))
            # User metadata can describe a consumer or experiment, but it
            # cannot relabel the actual lowering as native or deployable.
            plan.factorization_metadata.update(protected)
        return plan

    def execute(self, values: Mapping[str, Any]) -> dict[str, Any]:
        missing = [name for name in self.ir.inputs if name not in values]
        if missing:
            raise KeyError(f"render plan inputs are missing: {', '.join(missing)}")
        self.ir.validate_values(values)
        environment = cast(dict[str, Any], self.to_execution_plan().execute(values))
        for item in self.ir.nodes:
            if item.output not in environment or item.value_kind is FieldKind.ANY:
                continue
            expected_kind = cast(FieldKind, item.value_kind)
            actual = _runtime_kind(environment[item.output])
            if not _output_kind_compatible(actual, expected_kind, item.op):
                raise TypeError(f"{item.op} produced {actual.value}, declared {expected_kind.value}")
        return environment

    def render(self, values: Mapping[str, Any]) -> Any:
        environment = self.execute(values)
        selected = {name: environment[name] for name in self.outputs}
        return next(iter(selected.values())) if len(selected) == 1 else selected


def _one(operands: Sequence[Any], op: str) -> Any:
    if len(operands) != 1:
        raise ValueError(f"{op} requires exactly one input")
    return operands[0]


def _two(operands: Sequence[Any], op: str) -> tuple[Any, Any]:
    if len(operands) != 2:
        raise ValueError(f"{op} requires exactly two inputs")
    return operands[0], operands[1]


def _impl() -> Any:
    # Keep the registry leaf importable without importing the tensor v0.2
    # implementation.  This also avoids a renderer ↔ tensor.v2 import cycle.
    from spf.tensor import v02 as implementation

    return implementation


def _unwrap(value: Any) -> Any:
    return value.value if isinstance(value, TypedField) else value


def _unwrap_operands(operands: Sequence[Any]) -> tuple[Any, ...]:
    return tuple(_unwrap(value) for value in operands)


def _apply_block_deferred_operation(value: Any, operation: Any, params: Mapping[str, Any]) -> Any:
    """Execute a closed Field operation on a retained block carrier.

    Field lowering keeps representation-aware operations visible as a typed
    ``v02_cast`` until the carrier reaches the executable VM.  Dispatching
    here preserves the block/header/filter hierarchy; coercing the carrier
    through NumPy would lose its addressed consumer before materialization.
    ``None`` means that this operation is outside the closed block algebra
    and must remain on the named runtime fallback path.
    """

    from spf.renderer.blocks import BlockView

    if not isinstance(value, BlockView):
        return None
    from spf.renderer.block_filter import FILTER_OPS, FilteredBlockView
    from spf.renderer.model import CoordinateChart, FieldOperation

    operation_name = str(operation).strip().lower().replace("-", "_")
    closed = FILTER_OPS | {"mask"}
    if operation_name not in closed:
        return None
    chart = params.get("_field_chart")
    if not isinstance(chart, CoordinateChart):
        chart = CoordinateChart.identity(len(value.shape))
    domain = params.get("_field_domain", "CUSTOM")
    # Spatial/frequency filter operations need the field domain and chart to
    # construct one compact transfer view.  Header-only operations can stay
    # on the raw carrier and retain fractional offsets in wide state.
    phase_delta = params.get("dT", params.get("phase", params.get("delta", 0)))
    header_only = operation_name in {"exposure", "mask"} or (
        operation_name in {"phase", "phase_shift"} and np.asarray(phase_delta).ndim == 0
    )
    if not header_only and not isinstance(value, FilteredBlockView):
        descriptor = type("_BlockFieldDescriptor", (), {"domain": domain, "chart": chart})()
        value = FilteredBlockView.from_field(value, descriptor)
    operation_params = {
        key: item for key, item in params.items() if not str(key).startswith("_")
    }
    return value.apply(FieldOperation(operation_name, operation_params))


def _phase_code(value: Any) -> int:
    implementation = _impl()
    if isinstance(value, implementation.Phase):
        return int(value)
    return int(value)


def _promote_one(magnitude: Any, phase: Any, implementation: Any) -> int:
    """Pack one explicit log magnitude plus phase into coherent SPF32."""

    if isinstance(magnitude, implementation.LogPos):
        log_code = magnitude.L
    elif isinstance(magnitude, implementation.LogSigned):
        log_code = magnitude.L
        if magnitude.sign < 0:
            phase = _phase_code(phase) + implementation.SPF32_PHASE_HALF
    elif isinstance(magnitude, implementation.SPF32Value):
        # A typed SPF value already has a phase.  A supplied phase is still
        # applied explicitly, which makes the conversion deterministic.
        log_code = magnitude.L
        phase = _phase_code(phase) + magnitude.T
    else:
        raise TypeError("phase+magnitude promotion requires LOG_POS or LOG_SIGNED magnitudes")
    return int(implementation.pack_spf32(int(log_code), _phase_code(phase)))


def _promote_phase_magnitude(magnitude: Any, phase: Any, implementation: Any) -> Any:
    magnitude_array = np.asarray(magnitude, dtype=object)
    phase_array = np.asarray(phase, dtype=object)
    if magnitude_array.ndim == 0 and phase_array.ndim == 0:
        return _promote_one(magnitude_array.item(), phase_array.item(), implementation)
    magnitude_b, phase_b = np.broadcast_arrays(magnitude_array, phase_array)
    output = np.empty(magnitude_b.shape, dtype=np.uint32)
    for index in np.ndindex(output.shape):
        output[index] = _promote_one(magnitude_b[index], phase_b[index], implementation)
    return output


def _cast_scalar(value: Any, source: FieldKind, target: FieldKind, implementation: Any) -> Any:
    """Perform one explicit typed-algebra cast without guessing semantics."""

    if target is FieldKind.LOG_SIGNED:
        if source is FieldKind.LOG_POS:
            if isinstance(value, implementation.LogPos):
                return implementation.LogSigned.nan() if value.is_nan else implementation.LogSigned(value.L, 1)
            raise TypeError("LOG_POS cast requires typed LogPos values")
        if source in {FieldKind.RADIANCE, FieldKind.INCOHERENT_RADIANCE}:
            if isinstance(value, implementation.LogPos):
                return implementation.LogSigned.nan() if value.is_nan else implementation.LogSigned(value.L, 1)
            raise TypeError("radiance-to-LOG_SIGNED cast requires typed LogPos values")
        if source is FieldKind.SPF_COMPLEX:
            word = int(value.word if isinstance(value, implementation.SPF32Value) else value)
            unpacked = implementation.unpack_spf32(word)
            if unpacked.is_nan:
                return implementation.LogSigned.nan()
            if unpacked.is_zero:
                return implementation.LogSigned.zero()
            if unpacked.T not in {implementation.SPF32_T_ZERO, implementation.SPF32_PHASE_HALF}:
                raise ValueError("SPF_COMPLEX to LOG_SIGNED requires a certified phase of 0 or pi")
            return implementation.LogSigned(unpacked.L, 1 if unpacked.T == implementation.SPF32_T_ZERO else -1)
        raise TypeError(f"cannot cast {source.value} to {target.value} explicitly")

    if target is FieldKind.LOG_POS:
        if source is FieldKind.LOG_POS:
            return value
        if source in {FieldKind.RADIANCE, FieldKind.INCOHERENT_RADIANCE}:
            if isinstance(value, implementation.LogPos):
                return value
            word = int(value.word if isinstance(value, implementation.SPF32Value) else value)
            unpacked = implementation.unpack_spf32(word)
            if unpacked.is_nan:
                return implementation.LogPos.nan()
            if unpacked.is_zero:
                return implementation.LogPos.zero()
            if unpacked.T != implementation.SPF32_T_ZERO:
                raise ValueError("radiance-to-LOG_POS cast requires zero phase")
            return implementation.LogPos(unpacked.L)
        if source is FieldKind.SPF_COMPLEX:
            raise TypeError("SPF_COMPLEX to LOG_POS requires explicit abs2")
        if source is FieldKind.DENSE:
            return implementation._format_materialized(value, "log_pos")
        raise TypeError(f"cannot cast {source.value} to {target.value} explicitly")

    if target is FieldKind.PHASE:
        if source is FieldKind.PHASE:
            return value
        if source is FieldKind.SPF_COMPLEX:
            word = int(value.word if isinstance(value, implementation.SPF32Value) else value)
            unpacked = implementation.unpack_spf32(word)
            if unpacked.is_nan:
                raise ValueError("NaN SPF_COMPLEX has no valid PHASE cast")
            return implementation.Phase(unpacked.T)
        raise TypeError(f"cannot cast {source.value} to {target.value} explicitly")

    if target is FieldKind.SPF_COMPLEX:
        if source in {FieldKind.WORDS, FieldKind.SPF_COMPLEX}:
            return np.asarray(value, dtype=np.uint32) if isinstance(value, np.ndarray) else int(value)
        if source is FieldKind.LOG_SIGNED:
            return _promote_phase_magnitude(value, 0, implementation)
        if source in {
            FieldKind.LOG_POS,
            FieldKind.RADIANCE,
            FieldKind.INCOHERENT_RADIANCE,
        }:
            if source in {FieldKind.RADIANCE, FieldKind.INCOHERENT_RADIANCE} and isinstance(
                value, (int, np.integer, np.ndarray)
            ):
                raw = np.asarray(value, dtype=np.uint32)
                phases = np.asarray(implementation.unpack_T(raw))
                logs = np.asarray(implementation.unpack_L(raw))
                nan_mask = (logs == implementation.SPF32_NAN_L) & (phases == implementation.SPF32_NAN_T)
                if np.any((phases != implementation.SPF32_T_ZERO) & ~nan_mask):
                    raise ValueError("radiance-to-SPF_COMPLEX cast requires zero phase")
                return np.asarray(implementation.pack_spf32(implementation.unpack_L(raw), 0), dtype=np.uint32)
            return _promote_phase_magnitude(value, 0, implementation)
        raise TypeError(f"cannot cast {source.value} to {target.value} explicitly")

    if target is FieldKind.WORDS:
        if source is FieldKind.SPF_COMPLEX:
            return np.asarray(value, dtype=np.uint32) if isinstance(value, np.ndarray) else int(value)
        if source in {
            FieldKind.LOG_POS,
            FieldKind.LOG_SIGNED,
            FieldKind.PHASE,
        }:
            if source is FieldKind.PHASE:
                return implementation.pack_spf32(0, _phase_code(value))
            return implementation.pack_spf32(value)
        raise TypeError(f"cannot cast {source.value} to {target.value} explicitly")

    if target is FieldKind.DENSE:
        if source is FieldKind.LOG_POS or source is FieldKind.LOG_SIGNED:
            return value.value
        if source is FieldKind.SPF_COMPLEX:
            word = value.word if isinstance(value, implementation.SPF32Value) else int(value)
            return implementation.decode_complex(word)
        if source is FieldKind.PHASE:
            phase = _phase_code(value)
            return complex(np.exp(1j * phase * implementation._PHASE_TO_RAD))
        raise TypeError(f"cannot cast {source.value} to {target.value} explicitly")

    raise TypeError(f"unsupported explicit cast target {target.value}")


def _explicit_cast(value: Any, source: FieldKind, target: FieldKind, implementation: Any) -> Any:
    """Map :func:`_cast_scalar` over a scalar or array carrier."""

    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _cast_scalar(value.item(), source, target, implementation)
        mapped = _map_scalar_or_array(value, lambda item: _cast_scalar(item, source, target, implementation))
        if target is FieldKind.WORDS or target is FieldKind.SPF_COMPLEX:
            return np.asarray(mapped, dtype=np.uint32)
        if target is FieldKind.DENSE:
            return np.asarray(mapped)
        return mapped
    if isinstance(value, (list, tuple)):
        mapped = _map_scalar_or_array(
            np.asarray(value, dtype=object),
            lambda item: _cast_scalar(item, source, target, implementation),
        )
        if target in {FieldKind.WORDS, FieldKind.SPF_COMPLEX}:
            return np.asarray(mapped, dtype=np.uint32)
        return mapped
    return _cast_scalar(value, source, target, implementation)


def _region_params(params: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    region = params.get("region")
    shape = params.get("shape")
    if region is None and shape is not None:
        region = tuple(shape)
    if region is None:
        raise ValueError("materialization requires params['region'] or params['shape']")
    return region, params.get("output_format"), params.get("origin", (0.0, 0.0))


def _separable_materialize(field: Any, params: Mapping[str, Any]) -> Any:
    implementation = _impl()
    points = params.get("points")
    if points is None:
        region = params.get("region")
        if isinstance(region, Mapping):
            points = region.get("points")
    if points is None:
        raise ValueError("separable materialization requires params['points']")

    output_format = params.get("output_format")
    raw_format = getattr(output_format, "value", output_format)
    name = str(raw_format).lower().replace("-", "_") if raw_format is not None else None
    return_result = bool(params.get("return_result", params.get("return_certificate", False)))

    def materialized_value(result: Any) -> Any:
        """Format one typed reduction result at the renderer boundary."""

        value = result.value
        if name in {"spf_complex", "spf", "spf32", "word", "words", "uint32"}:
            if isinstance(value, implementation.SPF32Value):
                return value.word
            if isinstance(value, (implementation.LogPos, implementation.LogSigned)):
                return implementation.pack_spf32(value)
            if isinstance(value, (int, np.integer)):
                # SPF_COMPLEX reductions already return packed words.
                return int(value) & 0xFFFFFFFF
            return implementation.encode_complex(value)
        if name in {
            "log_pos",
            "logpos",
            "positive",
            "positive_real",
            "log_signed",
            "logsigned",
            "signed",
            "signed_real",
            "phase",
            "phase_only",
        }:
            return implementation._format_materialized(value, output_format)
        if name in {"native", "default", "complex", "complex128", "dense"}:
            if isinstance(value, (implementation.LogPos, implementation.LogSigned)):
                return value.value
            if isinstance(value, implementation.SPF32Value):
                return implementation.decode_complex(value.word)
            if isinstance(value, (int, np.integer)):
                return implementation.decode_complex(int(value))
            return value
        if name in {"real", "float", "float64"}:
            if isinstance(value, (implementation.LogPos, implementation.LogSigned)):
                return value.value
            if isinstance(value, implementation.SPF32Value):
                return implementation.decode_complex(value.word).real
            if isinstance(value, (int, np.integer)):
                return implementation.decode_complex(int(value)).real
            return float(np.real(value))
        if output_format is None:
            return value
        try:
            return np.asarray(value, dtype=output_format).item()
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unsupported separable output_format {output_format!r}") from exc

    def source_value(result: Any) -> complex:
        value = result.value
        if isinstance(value, implementation.LogPos):
            return complex(value.value)
        if isinstance(value, implementation.LogSigned):
            return complex(value.value)
        if isinstance(value, implementation.SPF32Value):
            return complex(implementation.decode_complex(value.word))
        if isinstance(value, implementation.Phase):
            return complex(np.exp(1j * value.T * implementation._PHASE_TO_RAD))
        if isinstance(value, (int, np.integer)) and result.value_kind == "SPF_COMPLEX":
            return complex(implementation.decode_complex(int(value)))
        return complex(value)

    def aggregate_ledger(results: Sequence[Any]) -> Any:
        ledgers = tuple(result.ledger for result in results)
        if not ledgers:
            return implementation.ErrorLedger()
        return implementation.ErrorLedger(
            source_quantization=max(ledger.source_quantization for ledger in ledgers),
            representation_fit=max(ledger.representation_fit for ledger in ledgers),
            pruning=max(ledger.pruning for ledger in ledgers),
            reducer=max(ledger.reducer for ledger in ledgers),
            transform=max(ledger.transform for ledger in ledgers),
            materialization=max(ledger.materialization for ledger in ledgers),
            display=max(ledger.display for ledger in ledgers),
            stages=ledgers,
        )

    point_array = np.asarray(points)
    if point_array.ndim == 1:
        result = implementation.evaluate_separable(
            field,
            point_array,
            kind=params.get("kind"),
            spec=params.get("spec"),
        )
        formatted = materialized_value(result)
        if not return_result:
            return formatted
        return implementation._materialization_result(
            formatted,
            source_value(result),
            output_format,
            result.ledger,
        )
    if point_array.shape[-1] != field.dimension:
        raise ValueError("separable materialization points have the wrong dimension")
    values = np.empty(point_array.shape[:-1], dtype=object)
    source_values = np.empty(point_array.shape[:-1], dtype=np.complex128)
    reduction_results: list[Any] = []
    for index in np.ndindex(values.shape):
        result = implementation.evaluate_separable(
            field,
            point_array[index],
            kind=params.get("kind"),
            spec=params.get("spec"),
        )
        values[index] = materialized_value(result)
        source_values[index] = source_value(result)
        reduction_results.append(result)
    if name in {
        "spf_complex",
        "spf",
        "spf32",
        "word",
        "words",
        "uint32",
    }:
        formatted = np.asarray(values, dtype=np.uint32)
    else:
        try:
            formatted = np.asarray(values)
        except (TypeError, ValueError):
            formatted = values
    if return_result:
        return implementation._materialization_result(
            formatted,
            source_values,
            output_format,
            aggregate_ledger(reduction_results),
        )
    return formatted


def _map_scalar_or_array(value: Any, function: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return function(value.item())
        values = np.empty(value.shape, dtype=object)
        for index in np.ndindex(value.shape):
            values[index] = function(value[index])
        return values
    if isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=object)
        return _map_scalar_or_array(array, function)
    return function(value)


def _jet_add(left: Any, right: Any) -> Any:
    implementation = _impl()
    if not isinstance(left, implementation.Jet2D) or not isinstance(right, implementation.Jet2D):
        raise TypeError("v02_jet_add requires two Jet2D values")
    if left.value_kind != right.value_kind:
        raise TypeError("v02_jet_add requires matching value kinds")
    if left.value_kind == "log_signed" and left.sign != right.sign:
        raise TypeError("LOG_SIGNED jets with opposite signs need an explicit signed reduction")

    guard_name = "guard_T" if left.value_kind == "phase" else "guard_L"
    guard = max(getattr(left, guard_name), getattr(right, guard_name))
    left_scale = 2 ** (guard - getattr(left, guard_name))
    right_scale = 2 ** (guard - getattr(right, guard_name))

    scaled_kind = left.value_kind in {"log", "log_pos", "log_signed", "phase"}

    def align(value: Any, scale: int) -> Any:
        return value * scale if scaled_kind else value

    return implementation.Jet2D(
        value=align(left.value, left_scale) + align(right.value, right_scale),
        grad_x=align(left.grad_x, left_scale) + align(right.grad_x, right_scale),
        grad_y=align(left.grad_y, left_scale) + align(right.grad_y, right_scale),
        hess_xx=align(left.hess_xx, left_scale) + align(right.hess_xx, right_scale),
        hess_xy=align(left.hess_xy, left_scale) + align(right.hess_xy, right_scale),
        hess_yy=align(left.hess_yy, left_scale) + align(right.hess_yy, right_scale),
        guard_L=max(left.guard_L, right.guard_L),
        guard_T=max(left.guard_T, right.guard_T),
        value_kind=left.value_kind,
        sign=left.sign,
    )


def _atom_from_options(operands: Sequence[Any], options: Mapping[str, Any]) -> Any:
    implementation = _impl()
    if len(operands) == 1 and isinstance(operands[0], implementation.GaussianChirpAtom):
        return operands[0]
    if isinstance(options.get("atom"), implementation.GaussianChirpAtom):
        return options["atom"]
    return implementation.GaussianChirpAtom(
        kappa=options.get("kappa", 0.0 + 0.0j),
        h=options.get("h", np.zeros(1, dtype=np.complex128)),
        C=options.get("C", np.eye(1, dtype=np.complex128)),
    )


def _reduction_values(operands: Sequence[Any]) -> Sequence[Any]:
    if len(operands) == 1 and isinstance(operands[0], np.ndarray):
        return tuple(np.asarray(operands[0]).reshape(-1))
    if len(operands) == 1 and isinstance(operands[0], (list, tuple)):
        return tuple(operands[0])
    return operands


def _reduction_spec(options: Mapping[str, Any], implementation: Any) -> Any:
    spec = options.get("spec")
    if isinstance(spec, Mapping):
        return implementation.ReductionSpec(**dict(spec))
    if spec is not None:
        return spec
    fields = {
        name: options[name]
        for name in (
            "error_budget_abs",
            "error_budget_rel",
            "precision_mode",
            "allow_pruning",
            "deterministic",
            "accumulator_fraction_bits",
            "required_value_kind",
        )
        if name in options
    }
    return implementation.ReductionSpec(**fields) if fields else None


def _phase_value(value: Any) -> Any:
    implementation = _impl()
    if isinstance(value, implementation.Phase):
        return value
    return implementation.Phase(int(value))


def _reduce_reference(operands: Sequence[Any], options: Mapping[str, Any], kind: Any) -> Any:
    implementation = _impl()
    result = implementation.reduce_v02(
        _reduction_values(operands),
        kind=kind,
        spec=_reduction_spec(options, implementation),
    )
    return result.value if bool(options.get("return_value", False)) else result


def execute_op(op: str, operands: Sequence[Any] = (), params: Mapping[str, Any] | None = None) -> Any:
    """Execute one registered structured v0.2 reference operation."""

    operation = str(op)
    if operation.lower() in {"add", "v02_add"}:
        raise ValueError("generic untyped Add is not legal in SPF v0.2; use an explicit typed operation")
    if operation not in OP_REGISTRY or not operation.startswith("v02_"):
        raise ValueError(f"unsupported structured renderer op {operation!r}")
    options = params or {}
    implementation = _impl()
    original_operands = tuple(operands)
    _validate_execute_inputs(operation, original_operands, options)
    operands = _unwrap_operands(original_operands)

    if operation in {"v02_field_constant", "v02_field_asset", "v02_field_reference"}:
        if original_operands:
            value = _one(original_operands, operation)
        else:
            value = None
            for key in ("value", "asset", "reference"):
                if key in options:
                    value = options[key]
                    break
            if value is None:
                raise ValueError(f"{operation} requires an operand or params['value']")
        declared_kind = options.get("_declared_value_kind", options.get("value_kind", options.get("kind")))
        if declared_kind is not None and _kind(declared_kind) is not FieldKind.ANY:
            metadata = _metadata_for(
                _kind(declared_kind),
                domain=options.get("_declared_domain", options.get("domain")),
                semantic=options.get("_declared_semantic", options.get("semantic")),
                coherence=options.get("_declared_coherence", options.get("coherence")),
            )
            return TypedField(
                _unwrap(value),
                _kind(declared_kind),
                metadata.domain,
                metadata.semantic,
                metadata.coherence,
            )
        return _unwrap(value)

    if operation in {"v02_log_mul", "v02_log_div"}:
        left, right = _two(operands, operation)
        function = implementation.logpos_mul if operation == "v02_log_mul" else implementation.logpos_div
        return function(left, right)

    if operation in {"v02_phase_add", "v02_phase_sub", "v02_phase_compose"}:
        left, right = _two(operands, operation)
        if operation == "v02_phase_sub":
            return implementation.Phase(int(_phase_value(left)) - int(_phase_value(right)))
        return implementation.phase_compose(_phase_value(left), _phase_value(right))

    if operation == "v02_phase_conj":
        return implementation.Phase(-int(_phase_value(_one(operands, operation))))

    if operation == "v02_phase_shift":
        if len(operands) == 2:
            value, delta = operands
        else:
            value = _one(operands, operation)
            delta = options.get("dT", options.get("phase", options.get("delta")))
        if delta is None:
            raise ValueError("v02_phase_shift requires params['dT'] or a PHASE input")
        block_value = _apply_block_deferred_operation(
            _unwrap(value), "phase_shift", {"dT": delta, **dict(options)}
        )
        if block_value is not None:
            return block_value
        return implementation.spf_phase_shift(
            value,
            _phase_code(delta),
            profile=implementation.NaNProfile.STRICT_V02,
        )

    if operation in {"v02_promote_complex", "v02_phase_magnitude_to_complex"}:
        if len(operands) == 2:
            magnitude, phase = operands
        elif len(operands) == 1:
            magnitude = _one(operands, operation)
            phase = options.get("phase", options.get("T", 0))
        else:
            magnitude = options.get("magnitude", options.get("value"))
            phase = options.get("phase", options.get("T", 0))
        if magnitude is None:
            raise ValueError("phase+magnitude promotion requires a magnitude")
        return _promote_phase_magnitude(magnitude, phase, implementation)

    if operation == "v02_cast":
        if len(original_operands) != 1:
            raise ValueError("v02_cast requires exactly one input")
        source = original_operands[0]
        value = _unwrap(source)
        deferred = options.get("deferred_operation")
        if deferred is not None:
            block_value = _apply_block_deferred_operation(value, deferred, options)
            if block_value is not None:
                # A declared positive/signed source can retain ordinary packed
                # words without an intensity conversion. Preserve that typed
                # producer proof when its storage has no intrinsic semantic tag.
                if isinstance(source, TypedField) and not _kind_compatible(
                    _runtime_kind(block_value), _kind(source.kind)
                ):
                    return TypedField(block_value, source.kind, source.domain, source.semantic, source.coherence)
                return block_value
        declared_sources = tuple(options.get("_input_value_kinds", ()))
        declared_source = declared_sources[0] if declared_sources else _runtime_kind(source)
        source_kind = _kind(options.get("from_kind", declared_source))
        target_name = options.get("to_kind", options.get("target_kind"))
        if target_name is None:
            raise ValueError("v02_cast requires params['to_kind']")
        target = _kind(target_name)
        if target is source_kind:
            return value
        return _explicit_cast(value, source_kind, target, implementation)

    if operation in {
        "v02_extract",
        "v02_complex_pair_select",
        "v02_mixture_factor_select",
    }:
        result = _one(operands, operation)
        if not isinstance(result, MultiResult):
            raise TypeError(f"{operation} requires a typed multi-result; tuple-valued fields are not implicit")
        selector = options.get("select", options.get("which", options.get("index")))
        if selector is None:
            raise ValueError(f"{operation} requires params['select'] or params['index']")
        return result.select(selector)

    if operation in {"v02_complex_mul", "v02_complex_div"}:
        left, right = _two(operands, operation)
        function = implementation.spf_mul if operation == "v02_complex_mul" else implementation.spf_div
        return function(left, right, profile=implementation.NaNProfile.STRICT_V02)

    if operation in {
        "v02_reduce_positive",
        "v02_reduce_signed",
        "v02_reduce_complex",
        "v02_reduce_optical_depth",
        "v02_reduce_transport",
    }:
        kind = {
            "v02_reduce_positive": implementation.ValueKind.LOG_POS,
            "v02_reduce_signed": implementation.ValueKind.LOG_SIGNED,
            "v02_reduce_complex": implementation.ValueKind.SPF_COMPLEX,
            "v02_reduce_optical_depth": implementation.ValueKind.OPTICAL_DEPTH,
            "v02_reduce_transport": implementation.ValueKind.AFFINE_TRANSPORT,
        }[operation]
        return _reduce_reference(operands, options, kind)

    if operation == "v02_reduce_tropical":
        kind = str(options.get("kind", options.get("mode", "tropical_min"))).lower().replace("-", "_")
        if kind in {"max", "tropical_max"}:
            kind = implementation.ValueKind.TROPICAL_MAX
        else:
            kind = implementation.ValueKind.TROPICAL_MIN
        return _reduce_reference(operands, options, kind)

    if operation == "v02_depth_min":
        return _reduce_reference(operands, options, implementation.ValueKind.TROPICAL_MIN)

    if operation == "v02_visibility_select":
        values, mask = _two(operands, operation)
        default = options.get("default", 0)
        return np.where(np.asarray(mask, dtype=bool), np.asarray(values), default)

    if operation == "v02_mask":
        values, mask = _two(operands, operation)
        default = options.get("default", 0)
        return np.where(np.asarray(mask, dtype=bool), np.asarray(values), default)

    if operation == "v02_coordinate_remap":
        points = _one(operands, operation)
        matrix = np.asarray(options.get("A"), dtype=np.float64)
        shift = np.asarray(options.get("b", np.zeros(matrix.shape[0])), dtype=np.float64).reshape(-1)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or shift.shape != (matrix.shape[0],):
            raise ValueError("coordinate remap requires square A and matching b")
        point_array = np.asarray(points, dtype=np.float64)
        if point_array.ndim == 0 or point_array.shape[-1] != matrix.shape[1]:
            raise ValueError("coordinate remap points have the wrong trailing dimension")
        return point_array @ matrix.T + shift

    if operation == "v02_bandlimit":
        values = np.asarray(_one(operands, operation))
        band = options.get("band", options.get("limits"))
        if band is None or len(tuple(band)) != 2:
            raise ValueError("bandlimit requires params['band']=(low, high)")
        low, high = (int(band[0]), int(band[1]))
        axis = int(options.get("axis", -1))
        axis = axis if axis >= 0 else values.ndim + axis
        if axis < 0 or axis >= values.ndim:
            raise ValueError("bandlimit axis is out of range")
        length = values.shape[axis]
        coordinates = np.arange(length)
        if bool(options.get("centered", False)):
            coordinates = (coordinates + length // 2) % length - length // 2
        keep = (coordinates >= low) & (coordinates <= high)
        shape = [1] * values.ndim
        shape[axis] = length
        zero = implementation.SPF32_WORD_ZERO if values.dtype == np.uint32 else 0
        return np.where(keep.reshape(shape), values, zero)

    if operation == "v02_support_intersect":
        supports = operands
        if (
            len(supports) == 1
            and isinstance(supports[0], (list, tuple))
            and supports[0]
            and not np.isscalar(supports[0][0])
        ):
            supports = tuple(supports[0])
        if not supports:
            raise ValueError("support intersection requires at least one support")

        def bounds(support: Any) -> tuple[np.ndarray, np.ndarray]:
            if isinstance(support, Mapping) and "lo" in support and "hi" in support:
                return np.asarray(support["lo"], dtype=np.float64), np.asarray(support["hi"], dtype=np.float64)
            array = np.asarray(support, dtype=np.float64)
            if array.ndim != 2 or 2 not in array.shape:
                raise ValueError("supports must have shape (d,2), (2,d), or {'lo','hi'}")
            return (array[:, 0], array[:, 1]) if array.shape[1] == 2 else (array[0], array[1])

        lower, upper = bounds(supports[0])
        for support in supports[1:]:
            other_lower, other_upper = bounds(support)
            if other_lower.shape != lower.shape:
                raise ValueError("support dimensions do not match")
            lower = np.maximum(lower, other_lower)
            upper = np.minimum(upper, other_upper)
        return np.stack((lower, upper), axis=0)

    if operation == "v02_sort":
        values = _one(operands, operation)
        axis = int(options.get("axis", -1))
        if "key" in options:
            sort_key = np.asarray(options["key"])
            order = np.argsort(sort_key, axis=axis, kind="stable")
            return np.take_along_axis(np.asarray(values), order, axis=axis)
        return np.sort(np.asarray(values), axis=axis, kind="stable")

    if operation == "v02_tile_cull":
        value = _one(operands, operation)
        region = options.get("region")
        threshold = float(options.get("threshold", options.get("min_bound", 0.0)))
        if isinstance(value, implementation.AtomMixture):
            if region is None:
                raise ValueError("tile cull requires params['region']")
            return np.asarray([float(implementation.atom_bound(atom, region)) >= threshold for atom in value.atoms])
        if isinstance(value, (int, float, np.integer, np.floating)):
            bound = float(value)
        else:
            bound = float(np.max(np.asarray(value)).item())
        return bool(bound >= threshold)

    if operation == "v02_sparse_synthesis":
        if len(operands) == 2:
            indices, sparse_values = operands
        else:
            indices = options.get("indices")
            sparse_values = options.get("values")
        sparse_shape = tuple(int(item) for item in options.get("shape", ()))
        if indices is None or sparse_values is None or not sparse_shape:
            raise ValueError("sparse synthesis requires indices, values, and shape")
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim == 1:
            index_array = index_array.reshape(-1, 1)
        if index_array.ndim != 2 or index_array.shape[1] != len(sparse_shape):
            raise ValueError("sparse synthesis indices must have shape (N, len(shape))")
        sparse_array = np.asarray(sparse_values)
        if sparse_array.reshape(-1).size != index_array.shape[0]:
            raise ValueError("sparse synthesis indices and values must have the same length")
        fill = options.get("fill", implementation.SPF32_WORD_ZERO if sparse_array.dtype == np.uint32 else 0)
        result = np.full(sparse_shape, fill, dtype=sparse_array.dtype)
        for index, item in zip(index_array, sparse_array.reshape(-1), strict=True):
            coordinate = tuple(int(value) for value in index)
            if any(value < 0 or value >= size for value, size in zip(coordinate, sparse_shape, strict=True)):
                raise IndexError("sparse synthesis index is outside the output shape")
            result[coordinate] = item
        return result

    if operation == "v02_jet2d":
        if operands:
            value = _one(operands, operation)
            if isinstance(value, implementation.Jet2D):
                return value
            raise TypeError("v02_jet2d input must be a Jet2D")
        names = ("value", "grad_x", "grad_y", "hess_xx", "hess_xy", "hess_yy", "guard_L", "guard_T", "value_kind")
        return implementation.Jet2D(**{name: options[name] for name in names if name in options})

    if operation == "v02_jet_add":
        left, right = _two(operands, operation)
        return _jet_add(left, right)

    if operation == "v02_jet_transform":
        jet = _one(operands, operation)
        return implementation.jet_transform(jet, options["A"], options["b"])

    if operation == "v02_jet_materialize":
        jet = _one(operands, operation)
        region, output_format, origin = _region_params(options)
        return implementation.jet_materialize(
            jet,
            region=region,
            output_format=output_format,
            origin=origin,
            modulo=options.get("modulo"),
            return_result=bool(options.get("return_result", False)),
            return_certificate=bool(options.get("return_certificate", False)),
        )

    if operation == "v02_atom_create":
        return _atom_from_options(operands, options)

    if operation in {"v02_atom_mul", "v02_atom_product", "v02_atom_div", "v02_atom_convolve"}:
        left, right = _two(operands, operation)
        functions = {
            "v02_atom_mul": implementation.atom_mul,
            "v02_atom_product": implementation.atom_mul,
            "v02_atom_div": implementation.atom_div,
            "v02_atom_convolve": implementation.atom_convolve,
        }
        return functions[operation](left, right)

    if operation == "v02_atom_affine":
        if len(operands) == 3:
            atom, affine_matrix, affine_shift = operands
        else:
            atom = _one(operands, operation)
            affine_matrix, affine_shift = options.get("A"), options.get("b")
        if affine_matrix is None or affine_shift is None:
            raise ValueError("atom affine transform requires A and b")
        return implementation.atom_affine(atom, affine_matrix, affine_shift)

    if operation in {"v02_atom_fourier", "v02_atom_inverse_fourier"}:
        atom = _one(operands, operation)
        function = (
            implementation.atom_fourier if operation == "v02_atom_fourier" else implementation.atom_inverse_fourier
        )
        return function(atom)

    if operation == "v02_atom_integral":
        return implementation.atom_integral(_one(operands, operation))

    if operation == "v02_atom_pixel_integral":
        atom, footprint = _two(operands, operation)
        return implementation.atom_pixel_integral(atom, footprint)

    if operation in {"v02_atom_derivative", "v02_atom_hessian"}:
        atom, point = _two(operands, operation)
        function = implementation.atom_derivative if operation == "v02_atom_derivative" else implementation.atom_hessian
        return function(atom, point)

    if operation in {"v02_atom_bound", "v02_atom_bound_log"}:
        if len(operands) == 2:
            atom, region = operands
        else:
            atom = _one(operands, operation)
            region = options.get("region")
        if region is None:
            raise ValueError("atom bound requires a region")
        function = implementation.atom_bound if operation == "v02_atom_bound" else implementation.atom_bound_log
        return function(atom, region)

    if operation == "v02_atom_to_jet":
        atom = _one(operands, operation)
        return implementation.atom_to_jet(atom, options.get("center", (0.0, 0.0)))

    if operation in {"v02_atom_mixture", "v02_mixture"}:
        if len(operands) == 1 and isinstance(operands[0], implementation.AtomMixture):
            return operands[0]
        if "atoms" in options:
            atoms = tuple(options["atoms"])
        elif len(operands) == 1 and isinstance(operands[0], (list, tuple)):
            atoms = tuple(operands[0])
        else:
            atoms = tuple(operands)
        return implementation.AtomMixture(
            atoms=atoms,
            residual=options.get("residual"),
            dropped_bound=float(options.get("dropped_bound", 0.0)),
        )

    if operation in {"v02_mixture_add", "v02_mixture_concat"}:
        left, right = _two(operands, operation)
        return implementation.mixture_add(left, right)

    if operation in {"v02_mixture_prune", "v02_mixture_cull"}:
        if len(operands) == 2:
            mixture, region = operands
        else:
            mixture = _one(operands, operation)
            region = options.get("region")
        if region is None:
            raise ValueError("mixture pruning requires a region")
        return implementation.mixture_prune(mixture, region, float(options.get("error_budget", 0.0)))

    if operation == "v02_mixture_bound":
        mixture = _one(operands, operation)
        region = options.get("region")
        if region is None:
            raise ValueError("mixture bound requires params['region']")
        return np.asarray([implementation.atom_bound(atom, region) for atom in mixture.atoms], dtype=object)

    if operation == "v02_mixture_factor":
        mixture = _one(operands, operation)
        if not mixture.atoms:
            return MultiResult(
                operation,
                (None, mixture),
                FieldKind.MIXTURE_FACTOR,
                ("factor", "residual"),
            )
        factor = options.get("factor")
        if factor is None:
            factor = implementation.GaussianChirpAtom(
                0.0j,
                mixture.atoms[0].h,
                mixture.atoms[0].C,
                allow_nondecaying=mixture.atoms[0].allow_nondecaying,
            )
        if not isinstance(factor, implementation.GaussianChirpAtom):
            raise TypeError("mixture factor must be a GaussianChirpAtom")
        residual = implementation.AtomMixture(
            tuple(implementation.atom_div(atom, factor) for atom in mixture.atoms),
            mixture.residual,
            mixture.dropped_bound,
        )
        return MultiResult(
            operation,
            (factor, residual),
            FieldKind.MIXTURE_FACTOR,
            ("factor", "residual"),
        )

    if operation == "v02_mixture_to_jets":
        mixture = _one(operands, operation)
        center = options.get("center", (0.0, 0.0))
        return tuple(implementation.atom_to_jet(atom, center) for atom in mixture.atoms)

    if operation == "v02_mixture_merge_equal":
        return implementation.mixture_merge_equal(_one(operands, operation))

    if operation == "v02_mixture_eval":
        mixture, points = _two(operands, operation)
        return mixture.evaluate(
            points,
            value_kind=options.get("value_kind"),
            spec=options.get("spec"),
        )

    if operation == "v02_mixture_materialize":
        mixture = _one(operands, operation)
        region, output_format, _ = _region_params(options)
        return mixture.materialize(
            region,
            output_format,
            spec=options.get("spec"),
            counters=options.get("counters"),
            return_result=bool(options.get("return_result", False)),
            return_certificate=bool(options.get("return_certificate", False)),
        )

    if operation == "v02_mixture_lower":
        mixture = _one(operands, operation)
        region, output_format, _ = _region_params(options)
        return mixture.materialize(
            region,
            output_format,
            spec=options.get("spec"),
            counters=options.get("counters"),
            return_result=bool(options.get("return_result", False)),
            return_certificate=bool(options.get("return_certificate", False)),
        )

    if operation in {"v02_separable", "v02_separable_field"}:
        if len(operands) == 1 and isinstance(operands[0], implementation.SeparableField):
            return operands[0]
        factors = options.get("factors", operands)
        return implementation.SeparableField(
            tuple(tuple(row) for row in factors),
            options.get("value_kind", "SPF_COMPLEX"),
        )

    if operation == "v02_separable_eval":
        field_value, point = _two(operands, operation)
        return implementation.evaluate_separable(
            field_value,
            point,
            kind=options.get("kind"),
            spec=options.get("spec"),
        )

    if operation == "v02_separable_materialize":
        return _separable_materialize(_one(operands, operation), options)

    if operation in {"v02_hermitian_synthesis", "v02_hermitian_synth"}:
        if len(operands) == 2:
            coefficients, phases = operands
        elif len(operands) == 1:
            coefficients, phases = operands[0], options.get("phases")
        else:
            coefficients, phases = options.get("coefficients"), options.get("phases")
        if coefficients is None or phases is None:
            raise ValueError("Hermitian synthesis requires coefficients and phases")
        return implementation.hermitian_synthesis(
            coefficients,
            phases,
            self_conjugate=options.get("self_conjugate"),
        )

    if operation == "v02_spherical_gaussian":
        if len(operands) == 1 and isinstance(operands[0], implementation.SphericalGaussian):
            return operands[0]
        if len(operands) >= 2:
            direction, concentration = operands[:2]
            amplitude = operands[2] if len(operands) > 2 else options.get("amplitude", 1.0 + 0.0j)
        else:
            direction = options.get("mu")
            concentration = options.get("lambda", options.get("lambda_"))
            amplitude = options.get("amplitude", 1.0 + 0.0j)
        if direction is None or concentration is None:
            raise ValueError("spherical Gaussian construction requires mu and lambda")
        return implementation.SphericalGaussian(direction, concentration, amplitude)

    if operation == "v02_spherical_gaussian_product":
        left, right = _two(operands, operation)
        return implementation.spherical_gaussian_product(left, right)

    if operation == "v02_spherical_gaussian_integral":
        return implementation.spherical_gaussian_integral(_one(operands, operation))

    if operation == "v02_transport_identity":
        return implementation.Transport(implementation.LogPos(0), implementation.LogPos.zero())

    if operation == "v02_transport":
        if len(operands) == 1 and isinstance(operands[0], implementation.Transport):
            return operands[0]
        if len(operands) == 2:
            transmittance, contribution = operands
        else:
            transmittance = options.get("T")
            contribution = options.get("C")
        if transmittance is None or contribution is None:
            raise ValueError("Transport construction requires T and C")
        return implementation.Transport(transmittance, contribution)

    if operation == "v02_transport_compose":
        front, back = _two(operands, operation)
        return implementation.transport_compose(front, back)

    if operation == "v02_transport_reduce":
        segments = operands
        if len(operands) == 1 and isinstance(operands[0], (list, tuple, np.ndarray)):
            segments = tuple(operands[0])
        if bool(options.get("return_result", False)):
            return implementation.reduce_transport(segments, spec=options.get("spec"))
        return implementation.transport_reduce_ordered(segments)

    if operation == "v02_transport_apply":
        transport, background = _two(operands, operation)
        return transport.apply(background)

    if operation in {"v02_logadd_pos", "v02_logsub_pos", "v02_logadd_signed"}:
        left, right = _two(operands, operation)
        function = {
            "v02_logadd_pos": implementation.logadd_pos,
            "v02_logsub_pos": implementation.logsub_pos,
            "v02_logadd_signed": implementation.logadd_signed,
        }[operation]
        return function(left, right)

    if operation == "v02_complex_pair_add":
        left, right = _two(operands, operation)
        result = implementation.spf_pair_addsub_hybrid(left, right, precision=options.get("precision", "balanced"))
        return MultiResult(
            operation,
            tuple(result),
            FieldKind.COMPLEX_PAIR,
            ("sum", "difference"),
        )

    if operation == "v02_abs2":
        return _map_scalar_or_array(_one(operands, operation), implementation.spf_abs2)

    if operation in {"v02_materialize", "v02_materialize_tile"}:
        value = _one(operands, operation)
        if isinstance(value, implementation.Jet2D):
            region, output_format, origin = _region_params(options)
            return implementation.jet_materialize(
                value,
                region=region,
                output_format=output_format,
                origin=origin,
                modulo=options.get("modulo"),
                return_result=bool(options.get("return_result", False)),
                return_certificate=bool(options.get("return_certificate", False)),
            )
        if isinstance(value, implementation.AtomMixture):
            region, output_format, _ = _region_params(options)
            return value.materialize(
                region,
                output_format,
                spec=options.get("spec"),
                counters=options.get("counters"),
                return_result=bool(options.get("return_result", False)),
                return_certificate=bool(options.get("return_certificate", False)),
            )
        if isinstance(value, implementation.SeparableField):
            return _separable_materialize(value, options)
        if hasattr(value, "copy"):
            return value.copy()
        return value

    if operation in {"v02_fft", "v02_fft_forward", "v02_ifft", "v02_fft_inverse"}:
        words = _one(operands, operation)
        function = (
            implementation.spf_fft_radix2
            if operation in {"v02_fft", "v02_fft_forward"}
            else implementation.spf_ifft_radix2
        )
        return function(words, precision=options.get("precision", "balanced"))

    if operation == "v02_tone":
        value = _one(operands, operation)
        mode = str(options.get("mode", "exposure")).lower().replace("-", "_")
        if mode == "exposure":
            exposure = options.get("exposure", options.get("amount", 0.0))
            return _map_scalar_or_array(value, lambda item: implementation.logpos_exposure(item, exposure))
        if mode == "power":
            power = options.get("power", options.get("amount", 1.0))
            return _map_scalar_or_array(value, lambda item: implementation.logpos_power(item, power))
        if mode in {"reinhard", "tone_reinhard"}:
            return _map_scalar_or_array(value, implementation.logpos_reinhard)
        raise ValueError(f"unsupported v0.2 tone mode {mode!r}")

    if operation in {"v02_display", "v02_display_convert"}:
        value = _one(operands, operation)
        return _map_scalar_or_array(
            value,
            lambda item: implementation.display_code_from_log(
                item,
                black_offset=int(options.get("black_offset", 0)),
                lut=options.get("lut"),
            ),
        )

    raise ValueError(f"no implementation for registered structured op {operation!r}")


def execute(plan: RenderPlan | FieldIR, values: Mapping[str, Any]) -> dict[str, Any]:
    """Execute a validated structured plan and return its full environment."""

    render_plan = plan if isinstance(plan, RenderPlan) else RenderPlan(plan)
    return render_plan.execute(values)


def render(plan: RenderPlan | FieldIR, values: Mapping[str, Any]) -> Any:
    """Execute a plan and return its declared output, or output mapping."""

    render_plan = plan if isinstance(plan, RenderPlan) else RenderPlan(plan)
    return render_plan.render(values)
