"""Canonical field materialization runtime for SFR v0.2.

The metadata-rich :mod:`spf.renderer.model` objects deliberately do not own
an evaluator.  This module is the small reference runtime at their boundary:
it keeps analytic representations analytic while the operation prefix is
closed, crosses one dense sampling boundary, and then executes the remaining
operations on that dense tile.  The latter distinction is important: a mask
or a display conversion must not cause an atom/jet/mixture to be sampled once
per operation.

The runtime is intentionally reference-oriented.  It uses the v0.2 tensor
helpers for the canonical SPF32 and typed scalar semantics, while keeping the
field-level error ledger and report visible to callers.  Backends may provide
their own primitive implementations later; the public entry point and the
materialization certificate remain the same.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import numpy as np

from spf.renderer.backends import SFRBackend, get_backend
from spf.renderer.carriers import decode_words, encode_words
from spf.renderer.model import (
    Coherence,
    Domain,
    Field,
    FieldOperation,
    MaterializationRequest,
    Region,
    RepresentationKind,
    SampleLayout,
    Support,
    SupportKind,
)
from spf.renderer.normalization import MaterializationError as MaterializationError

# Compatibility imports for existing clients of the former runtime helpers.
from spf.renderer.normalization import _apply_atom_operation as _apply_atom_operation
from spf.renderer.normalization import _apply_representation_error as _apply_representation_error
from spf.renderer.normalization import (
    _consume_representation_operations,
    _field_ledger_after_operations,
    _offset,
    _ResidualSum,
    _support_after_operations,
    complete_field_ledger,
    normalize_field,
)
from spf.renderer.normalization import _prepare_structured as _prepare_structured
from spf.renderer.normalization import _scale_ledger as _scale_ledger
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision
from spf.renderer.v02 import FieldKind
from spf.tensor import core as _core
from spf.tensor import v02

try:  # The residual module is intentionally optional for older installations.
    from spf.renderer.residuals import SparseResidual
except ImportError:  # pragma: no cover - only relevant to an incomplete install
    SparseResidual = ()  # type: ignore[assignment,misc]


_TAU = 2.0 * math.pi
_FORMAT_ALIASES = {
    "": "native",
    "none": "native",
    "default": "native",
    "native": "native",
    "dense": "native",
    "complex": "complex",
    "complex128": "native",
    "float": "real",
    "float64": "real",
    "real": "real",
    "spf": "spf32",
    "spf32": "spf32",
    "spf_complex": "spf32",
    "word": "spf32",
    "words": "spf32",
    "uint32": "spf32",
    "logpos": "log_pos",
    "positive": "log_pos",
    "positive_real": "log_pos",
    "log_pos": "log_pos",
    "logsigned": "log_signed",
    "signed": "log_signed",
    "signed_real": "log_signed",
    "log_signed": "log_signed",
    "phase_only": "phase",
    "phase": "phase",
    "display_code": "display",
    "display": "display",
}

_POSITIVE_KINDS = {
    v02.ValueKind.LOG_POS.value.lower(),
    "log_pos",
    "radiance",
    "incoherent_radiance",
}
_SIGNED_KINDS = {v02.ValueKind.LOG_SIGNED.value.lower(), "log_signed"}
_PHASE_KINDS = {v02.ValueKind.PHASE.value.lower(), "phase"}
_COMPLEX_KINDS = {v02.ValueKind.SPF_COMPLEX.value.lower(), "spf_complex", "complex"}


@dataclass(frozen=True, slots=True)
class _SamplingContext:
    points: np.ndarray
    output_shape: tuple[int, ...]
    order: str
    layout: SampleLayout | None
    region: Any
    explicit_points: bool = False
    output_target: str = "native"
    precision: str = "balanced"

    @property
    def element_count(self) -> int:
        return int(math.prod(self.output_shape))


@dataclass(slots=True)
class _Execution:
    values: Any
    source_kind: str
    source_values: Any
    analytic_operations: list[str]
    dense_operations: list[str]
    capabilities: list[str]
    fallbacks: list[str]
    notes: list[str]
    atoms_considered: int = 0
    atoms_pruned: int = 0
    reduction_operations: int = 0
    positive_logadds: int = 0
    complex_adds: int = 0
    pivot_reductions: int = 0
    fallback_conversions: int = 0
    direct_dense: bool = False
    reducer_error: float = 0.0
    reducer_phase_error: float = 0.0
    ingress_quantization: float = 0.0
    projection_quantization: float = 0.0
    transform_error: float = 0.0
    transform_phase_error: float = 0.0
    fft_count: int = 0


_SPF32_RELATIVE_QUANTIZATION = (2.0 ** (1.0 / 512.0) - 1.0) + math.pi / v02.SPF32_PHASE_MOD
_WORD_CLOSED_OPS = frozenset(
    {
        "mask",
        "exposure",
        "phase_shift",
        "phase",
        "abs2",
        "intensity",
        "magnitude_square",
        "bandlimit",
        "band_limit",
        "power",
        "tone_power",
        "reinhard",
        "tone_reinhard",
        "lens_phase",
        "display",
        "display_convert",
        "translate",
        "gaussian_filter",
        "rotate",
        "affine",
    }
)


class _WordSource:
    """Packed SPF32 word plane standing in for the complex comparison domain.

    The executed path carries words (AGENTS.md rule 1).  Anything that needs
    the Cartesian view of the same samples (a ``complex`` output request, a
    typed log/phase output, an observed-error measurement) decodes through
    :meth:`complex_reference` exactly once, at that boundary.  Scale queries
    read the L plane only and never decode.
    """

    __slots__ = ("_complex", "words")

    def __init__(self, words: np.ndarray) -> None:
        self.words = np.asarray(words, dtype=np.uint32)
        self._complex: np.ndarray | None = None

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.words.shape)

    @property
    def size(self) -> int:
        return int(self.words.size)

    @property
    def nbytes(self) -> int:
        return int(self.words.nbytes)

    @property
    def decoded(self) -> bool:
        return self._complex is not None

    def complex_reference(self) -> np.ndarray:
        if self._complex is None:
            self._complex = np.asarray(decode_words(self.words))
        return self._complex

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        array = self.complex_reference()
        return array if dtype is None else np.asarray(array, dtype=dtype)


def _words_max_abs(words: Any) -> float:
    """Magnitude envelope of a word plane from its L plane; no decode."""

    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    if packed.size == 0:
        return 0.0
    if np.any(packed == np.uint32(v02.SPF32_WORD_NAN)):
        return math.nan
    logs = _core.unpack_L(packed)
    finite = logs != v02.SPF32_L_ZERO
    if not np.any(finite):
        return 0.0
    top = int(np.max(logs[finite]))
    if top >= v02.SPF32_L_MAX_FINITE + 1:
        return math.inf
    return math.exp2(top / v02.SPF32_L_SCALE)


def _is_word_plane(value: Any) -> bool:
    return isinstance(value, np.ndarray) and value.dtype == np.uint32


def _cartesian_numeric(value: Any) -> bool:
    array = value if isinstance(value, np.ndarray) else None
    return array is not None and array.dtype.kind in "fc"


def _normal_name(value: Any) -> str:
    return str(getattr(value, "value", value)).strip().lower().replace("-", "_").replace(" ", "_")


def _format_name(value: Any) -> str:
    return _FORMAT_ALIASES.get(_normal_name(value), _normal_name(value))


def _field_kind_name(field: Field) -> str:
    return _normal_name(field.value_kind)


def _resolve_target(request: MaterializationRequest) -> str:
    """Requested output boundary format, with ``output_kind`` folded in."""

    target = _format_name(request.output_format)
    if target == "native" and request.output_kind is not None:
        requested_kind = _normal_name(request.output_kind)
        if requested_kind in {"log_pos", "radiance", "incoherent_radiance"}:
            target = "log_pos"
        elif requested_kind == "log_signed":
            target = "log_signed"
        elif requested_kind == "phase":
            target = "phase"
        elif requested_kind in {"words", "spf_complex"}:
            target = "spf32"
        elif requested_kind == "display":
            target = "display"
    return target


def _words_requested(field: Field, context: _SamplingContext) -> bool:
    """True when the output boundary is packed words (spf32, or native on an SPF-typed field)."""

    if context.output_target == "spf32":
        return True
    return context.output_target == "native" and _field_kind_name(field) in _COMPLEX_KINDS | {"words", "word", "uint32"}


def _representation_kind(value: RepresentationKind | str) -> RepresentationKind:
    if isinstance(value, RepresentationKind):
        return value
    return RepresentationKind(str(value).strip().upper().replace("-", "_").split(".")[-1])


def _is_kind(field: Field, *names: str) -> bool:
    kind = _field_kind_name(field)
    return kind in {_normal_name(name) for name in names}


def _payload_value(payload: Any, *, prefer: Sequence[str] = ("values", "value", "data", "payload")) -> Any:
    if isinstance(payload, Mapping):
        for key in prefer:
            if key in payload:
                return payload[key]
    if isinstance(payload, v02.MaterializationResult):
        return payload.values
    if (
        hasattr(payload, "value")
        and type(payload).__module__.startswith("spf")
        and isinstance(payload, (v02.LogPos, v02.LogSigned, v02.Phase, v02.SPF32Value))
    ):
        # Typed scalar wrappers expose ``value`` but should not be unwrapped
        # here; the scalar conversion helpers below know how to handle them.
        return payload
    return payload


def _as_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value if dtype is None else np.asarray(value, dtype=dtype)
    return np.asarray(value, dtype=dtype)


def _map_object(value: Any, function: Any, *, dtype: Any = object) -> np.ndarray | Any:
    array = np.asarray(value, dtype=object)
    if array.ndim == 0:
        return function(array.item())
    output = np.empty(array.shape, dtype=dtype)
    for index in np.ndindex(array.shape):
        output[index] = function(array[index])
    return output


def _typed_to_complex(value: Any, *, phase_units: str = "code") -> complex:
    if isinstance(value, v02.SPF32Value):
        return complex(decode_words(value.word))
    if isinstance(value, v02.LogPos):
        return complex(value.value)
    if isinstance(value, v02.LogSigned):
        return complex(value.value)
    if isinstance(value, v02.Phase):
        return complex(math.cos(value.radians), math.sin(value.radians))
    if isinstance(value, (int, np.integer)) and phase_units == "code":
        return complex(decode_words(int(value)))
    return complex(value)


def _reference_complex_array(value: Any, *, kind: str = "dense", phase_units: str = "code") -> np.ndarray:
    """REFERENCE_CARTESIAN conversion of a carrier to the complex comparison domain.

    AGENTS.md rule 2: this is the interop/measurement boundary, never an
    execution step.  Callers that only need a magnitude envelope, a phase
    difference or an ``abs2`` use the word-domain helpers instead
    (:func:`_words_max_abs`, :mod:`spf.tensor.v02_planes`).
    """

    name = _normal_name(kind)
    numeric = np.asarray(value)
    if numeric.dtype.kind != "O" and name in _POSITIVE_KINDS | _SIGNED_KINDS | _PHASE_KINDS:
        if name in _PHASE_KINDS:
            # A PHASE carrier is a wrapped 16-bit code plane (§6).  Pack it as
            # unit-magnitude words and let the library codec emit the phasor
            # (AGENTS.md rule 3; exact at the quarter turns, where a private
            # ``exp(1j*angle)`` is not).
            from spf.tensor import v02_planes as planes

            phase_plane = (
                planes.PhasePlane.from_radians(numeric.astype(np.float64))
                if phase_units in {"radians", "rad"}
                else planes.PhasePlane(np.asarray(numeric, dtype=np.int64))
            )
            return np.asarray(decode_words(phase_plane.to_words()))
        if numeric.dtype.kind in "iu":
            words = numeric.astype(np.uint32, copy=False)
            phases = words & 65535
            special = ((words >> 16) == 32768) | (words == v02.SPF32_WORD_NAN)
            allowed = phases == 0
            if name in _SIGNED_KINDS:
                allowed |= phases == v02.SPF32_PHASE_HALF
            if np.any(~special & ~allowed):
                raise MaterializationError("real fields require zero phase (or pi for signed-real)")
            return np.asarray(decode_words(words))
        # A real-kind payload stays in its own real carrier; only a Cartesian
        # import is checked for a stray imaginary part, and the single widening
        # happens once on the projected real plane.
        real_part = numeric
        if np.iscomplexobj(numeric):
            if np.any(np.abs(numeric.imag) > 1.0e-12 * np.maximum(1.0, np.abs(numeric.real))):
                raise MaterializationError("real fields require real-valued payloads")
            real_part = numeric.real
        return np.asarray(real_part, dtype=np.complex128)
    if name in _POSITIVE_KINDS:
        array = np.asarray(value, dtype=object)

        def positive(item: Any) -> complex:
            if isinstance(item, v02.LogPos):
                return complex(item.value)
            if isinstance(item, v02.SPF32Value):
                unpacked = item
                if not unpacked.is_zero and not unpacked.is_nan and unpacked.T != v02.SPF32_T_ZERO:
                    raise MaterializationError("positive fields require zero-phase SPF32 payloads")
                return complex(decode_words(unpacked.word))
            if isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
                unpacked = v02.unpack_spf32(int(item))
                if not unpacked.is_zero and not unpacked.is_nan and unpacked.T != v02.SPF32_T_ZERO:
                    raise MaterializationError("positive fields require zero-phase SPF32 payloads")
                return complex(decode_words(unpacked.word))
            if isinstance(item, (complex, np.complexfloating)):
                value = complex(item)
                if abs(value.imag) > 1.0e-12 * max(1.0, abs(value.real)):
                    raise MaterializationError("positive fields require real-valued payloads")
                return complex(value.real)
            return complex(float(item))

        return np.asarray(_map_object(array, positive, dtype=np.complex128))
    if name in _SIGNED_KINDS:
        array = np.asarray(value, dtype=object)

        def signed(item: Any) -> complex:
            if isinstance(item, v02.LogSigned):
                return complex(item.value)
            if isinstance(item, v02.LogPos):
                return complex(item.value)
            if isinstance(item, v02.SPF32Value):
                unpacked = item
            elif isinstance(item, (int, np.integer)) and not isinstance(item, (bool, np.bool_)):
                unpacked = v02.unpack_spf32(int(item))
            else:
                return complex(float(item))
            if unpacked.is_nan:
                return complex(math.nan, math.nan)
            if unpacked.is_zero:
                return 0.0 + 0.0j
            if unpacked.T not in {v02.SPF32_T_ZERO, v02.SPF32_PHASE_HALF}:
                raise MaterializationError("signed-real fields require zero or pi phase")
            return complex(decode_words(unpacked.word))

        return np.asarray(_map_object(array, signed, dtype=np.complex128))
    if name in _PHASE_KINDS:
        array = np.asarray(value, dtype=object)

        def phase(item: Any) -> complex:
            if isinstance(item, v02.Phase):
                return complex(math.cos(item.radians), math.sin(item.radians))
            raw = float(item)
            radians = raw if phase_units in {"radians", "rad"} else raw * _TAU / v02.SPF32_PHASE_MOD
            return complex(math.cos(radians), math.sin(radians))

        return np.asarray(_map_object(array, phase, dtype=np.complex128))
    if name in {"words", "word", "uint32"}:
        return np.asarray(decode_words(np.asarray(value, dtype=np.uint32)))
    array = np.asarray(value)
    if array.dtype == object:
        return np.asarray(
            _map_object(array, lambda item: _typed_to_complex(item, phase_units=phase_units), dtype=np.complex128)
        )
    if array.dtype.kind in "iu" and name in _COMPLEX_KINDS:
        return np.asarray(decode_words(np.asarray(array, dtype=np.uint32)))
    return np.asarray(array, dtype=np.complex128)


#: Compatibility spelling for the reference conversion above.  Kept so existing
#: callers and tests keep working; new code should say ``reference`` out loud.
_complex_array = _reference_complex_array


def _real_array(value: Any, *, kind: str, phase_units: str = "code") -> np.ndarray:
    name = _normal_name(kind)
    if name in _PHASE_KINDS:
        # A phase-only value is represented by its wrapped phase code/radian
        # carrier when requested as a real diagnostic, not by its unit phasor.
        array = np.asarray(value, dtype=object)

        def phase_value(item: Any) -> float:
            if isinstance(item, v02.Phase):
                return float(item.T if phase_units == "code" else item.radians)
            return float(item)

        return np.asarray(_map_object(array, phase_value, dtype=np.float64), dtype=np.float64)
    complex_values = _complex_array(value, kind=name, phase_units=phase_units)
    return np.asarray(np.real(complex_values), dtype=np.float64)


def _max_abs(value: Any) -> float:
    if isinstance(value, _WordSource):
        return _words_max_abs(value.words) if not value.decoded else _max_abs(value.complex_reference())
    if _is_word_plane(value):
        return _words_max_abs(value)
    array = np.asarray(value, dtype=np.complex128)
    if array.size == 0:
        return 0.0
    try:
        maximum = float(np.max(np.abs(array)))
    except (TypeError, ValueError, OverflowError):
        return math.inf
    return math.nan if math.isnan(maximum) else maximum


def _max_difference(left: Any, right: Any) -> float:
    try:
        difference = np.asarray(left, dtype=np.complex128) - np.asarray(right, dtype=np.complex128)
        if difference.size == 0:
            return 0.0
        maximum = float(np.max(np.abs(difference)))
    except (TypeError, ValueError, OverflowError):
        return math.inf
    return math.nan if math.isnan(maximum) else maximum


def _phase_difference(left: Any, right: Any) -> float:
    """Return a conservative maximum wrapped phase difference in radians."""

    first = np.asarray(left, dtype=np.complex128)
    second = np.asarray(right, dtype=np.complex128)
    if first.shape != second.shape:
        raise ValueError("phase comparison arrays must have equal shapes")
    if first.size == 0:
        return 0.0
    finite = np.isfinite(first.real) & np.isfinite(first.imag) & np.isfinite(second.real) & np.isfinite(second.imag)
    active = finite & (np.abs(first) > np.finfo(np.float64).tiny) & (np.abs(second) > np.finfo(np.float64).tiny)
    if not np.any(active):
        return 0.0 if np.all(finite) else math.nan
    values = np.angle(second[active] * np.conj(first[active]))
    return float(np.max(np.abs(values))) if values.size else 0.0


def _ledger_add(ledger: v02.ErrorLedger, **increments: float) -> v02.ErrorLedger:
    values = {
        name: float(getattr(ledger, name)) + float(increments.get(name, 0.0))
        for name in (
            "source_quantization",
            "representation_fit",
            "pruning",
            "reducer",
            "transform",
            "materialization",
            "display",
        )
    }
    return v02.ErrorLedger(**values, stages=ledger.stages)


def _support_intersects_request(support: Support, region: Any) -> bool:
    if support.kind is SupportKind.EMPTY:
        return False
    if support.kind in {SupportKind.FULL, SupportKind.UNKNOWN} or support.region is None:
        return True
    if isinstance(region, Mapping):
        if "lo" in region and "hi" in region:
            return support.region.intersects(Region(tuple(region["lo"]), tuple(region["hi"])))
        if "points" in region:
            points = np.asarray(region["points"], dtype=np.float64)
            if points.ndim == 1:
                points = points.reshape(1, -1)
            return any(support.region.contains(point) for point in points)
    return True


def _infer_dimension(field: Field, payload: Any, context_hint: Any = None) -> int:
    if field.chart.dimension:
        return field.chart.dimension
    if isinstance(payload, v02.GaussianChirpAtom):
        return payload.dimension
    if isinstance(payload, v02.AtomMixture) and payload.atoms:
        return payload.atoms[0].dimension
    if isinstance(payload, v02.SeparableField):
        return payload.dimension
    if isinstance(payload, v02.SphericalGaussian):
        return 3
    if isinstance(context_hint, np.ndarray) and context_hint.ndim:
        return int(context_hint.shape[-1]) if context_hint.ndim > 1 else int(context_hint.size)
    candidate = _payload_value(payload)
    if isinstance(candidate, np.ndarray) and candidate.ndim:
        return int(candidate.ndim)
    if isinstance(candidate, (list, tuple)) and candidate:
        return int(np.asarray(candidate).ndim)
    return 0


def _mapping_region(
    region: Any,
) -> tuple[tuple[int, ...] | None, tuple[float, ...] | None, tuple[float, ...] | None, Any]:
    if not isinstance(region, Mapping):
        return None, None, None, region
    shape = region.get("shape")
    shape_tuple = None if shape is None else tuple(int(value) for value in np.asarray(shape).reshape(-1))
    origin = region.get("origin")
    origin_tuple = None if origin is None else tuple(float(value) for value in np.asarray(origin).reshape(-1))
    spacing = region.get("spacing")
    spacing_tuple = None if spacing is None else tuple(float(value) for value in np.asarray(spacing).reshape(-1))
    return shape_tuple, origin_tuple, spacing_tuple, region


def _payload_shape(payload: Any) -> tuple[int, ...] | None:
    candidate = _payload_value(payload)
    if isinstance(candidate, np.ndarray) and candidate.ndim:
        return tuple(int(value) for value in candidate.shape)
    if isinstance(candidate, (list, tuple)) and candidate:
        array = np.asarray(candidate)
        return tuple(int(value) for value in array.shape) if array.ndim else None
    if isinstance(payload, SparseResidual):
        return tuple(payload.shape)
    if isinstance(payload, Mapping):
        for key in ("shape", "spatial_shape", "full_shape"):
            if key in payload:
                values = tuple(int(value) for value in np.asarray(payload[key]).reshape(-1))
                if values:
                    return values
    shape = getattr(payload, "shape", None)
    if shape is not None:
        values = tuple(int(value) for value in np.asarray(shape).reshape(-1))
        if values:
            return values
    return None


def _region_from_context(context: _SamplingContext) -> Any:
    if context.region is not None:
        return context.region
    if context.points.size:
        lower = np.min(np.asarray(context.points, dtype=np.float64), axis=0)
        upper = np.max(np.asarray(context.points, dtype=np.float64), axis=0)
        return {"lo": lower, "hi": upper}
    dimension = int(context.points.shape[-1]) if context.points.ndim > 1 else len(context.output_shape)
    return {"lo": np.zeros(dimension), "hi": np.zeros(dimension)}


def _renderer_layout_points(field: Field, layout: SampleLayout) -> np.ndarray:
    """Return regular-grid points in AXIS ORDER, except for texture ``(u, v)`` lookups.

    Design v0.2 §17.1 coordinate convention (fixed 2026-09-18): at the IR,
    runtime and transfer level, coordinate component ``i`` belongs to array axis
    ``i`` and carries ``layout.spacing[i]``.  Atom and mixture fields used to
    receive ``(x, y)`` points with the spacing crossed, which put them at odds
    with the dense, word-FFT and compact-kernel datapaths (audit C12); they now
    use ``SampleLayout.points``, the axis-order grid.  ``SpectralTexture.sample``
    is a texture API whose trailing pair is ``(u, v)`` with u advancing along
    columns, so texture fields keep that lookup order; it never reaches a
    translation or covariance, which stay axis order.
    """

    if len(layout.shape) != 2 or not field.metadata.get("spectral_texture"):
        return layout.points()
    height, width = layout.shape
    u_axis = layout.origin[1] + layout.spacing[1] * np.arange(width, dtype=np.float64)
    v_axis = layout.origin[0] + layout.spacing[0] * np.arange(height, dtype=np.float64)
    v_grid, u_grid = np.meshgrid(v_axis, u_axis, indexing="ij")
    grid = np.stack((u_grid, v_grid), axis=-1)
    return np.asarray(
        grid.reshape((-1, 2), order=cast(Literal["C", "F", "A"], layout.order)),
        dtype=np.float64,
    )


def _resolve_context(field: Field, request: MaterializationRequest, payload: Any) -> _SamplingContext:
    metadata = dict(field.metadata)
    metadata.update(dict(field.representation.metadata))
    explicit = metadata.get("points", metadata.get("directions"))
    region: Any = request.region
    if explicit is None and isinstance(request.region, Mapping):
        explicit = request.region.get("points", request.region.get("directions"))
    if explicit is not None:
        points = np.asarray(explicit, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] == 0:
            raise MaterializationError("explicit materialization points must have a trailing coordinate dimension")
        if not np.all(np.isfinite(points)):
            raise MaterializationError("materialization points must be finite")
        shape = points.shape[:-1]
        if not shape:
            shape = ()
        return _SamplingContext(
            points.reshape(-1, points.shape[-1]),
            tuple(shape),
            "C",
            None,
            region,
            True,
            _resolve_target(request),
            _normal_name(request.precision.mode),
        )

    shape_hint, origin_hint, spacing_hint, _raw_region = _mapping_region(request.region)
    layout = request.layout
    if layout is None and shape_hint is not None:
        layout = SampleLayout(shape_hint, origin=origin_hint or (), spacing=spacing_hint or ())
    if layout is None:
        inferred_shape = metadata.get("shape", metadata.get("resolution", metadata.get("full_shape")))
        if inferred_shape is None:
            inferred_shape = _payload_shape(payload)
        if inferred_shape is not None:
            layout = SampleLayout(tuple(int(value) for value in np.asarray(inferred_shape).reshape(-1)))
    if layout is None:
        raise MaterializationError("materialization requires a SampleLayout, region shape, or payload shape")

    raw_points = _renderer_layout_points(field, layout)
    dimension = _infer_dimension(field, payload, raw_points)
    if dimension and len(layout.shape) != dimension:
        # A zero-dimensional chart in Field.from_value is allowed for a
        # structured payload; a regular grid still determines the dimension.
        if field.chart.dimension not in {0, len(layout.shape)}:
            raise MaterializationError("sample layout dimension does not match the field chart")
        dimension = len(layout.shape)
    if raw_points.shape[1] != len(layout.shape):
        raise MaterializationError("sample layout points have the wrong dimension")
    if field.chart.dimension and field.chart.dimension == raw_points.shape[1]:
        chart = field.chart
        factors = np.asarray(chart.sample_spacing) * np.asarray(chart.scale)
        basis = np.asarray(chart.basis)
        if np.all(factors == 1) and np.array_equal(basis, np.eye(chart.dimension)) and not any(chart.origin):
            points = raw_points
        else:
            points = (raw_points * factors) @ basis.T + np.asarray(chart.origin)
    else:
        points = raw_points.astype(np.float64, copy=False)
    region = request.region
    if region is None:
        lower = np.min(points, axis=0) if points.size else np.zeros(points.shape[1])
        upper = np.max(points, axis=0) if points.size else lower
        region = {"lo": lower, "hi": upper}
    return _SamplingContext(
        points,
        layout.shape,
        layout.order,
        layout,
        region,
        False,
        _resolve_target(request),
        _normal_name(request.precision.mode),
    )


def _reshape_sample(value: Any, context: _SamplingContext, *, dtype: Any | None = None) -> Any:
    array = np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)
    if context.output_shape == ():
        return array.reshape(-1)[0] if array.size else array
    if array.ndim == 0:
        return np.full(context.output_shape, array.item(), dtype=array.dtype)
    if array.shape == context.output_shape:
        return array
    if array.size == context.element_count:
        return array.reshape(
            context.output_shape,
            order=cast(Literal["C", "F", "A"], context.order),
        )
    raise MaterializationError(
        f"sampled payload has shape {array.shape}, expected {context.output_shape} or {context.element_count} elements"
    )


def _packed_projection(payload: Any, context: _SamplingContext, *, field: Field, execution: _Execution) -> Any:
    """Project an analytic block transfer straight into packed words when words are requested.

    The transfer stays analytic (log/phase jets) until this one boundary; the
    SPF32 quantization bound of the projection goes into the ledger.  A
    Cartesian output request keeps the analytic Cartesian projection, which is
    the requested format and not an intermediate.
    """

    from spf.renderer.block_filter import FilteredBlockView

    if not isinstance(payload, FilteredBlockView) or not _words_requested(field, context):
        return None
    try:
        packed = payload.sample_packed(context.points)
    except ValueError as exc:
        raise MaterializationError(str(exc)) from exc
    if packed is None:
        return None
    words, bound = packed
    execution.projection_quantization = max(execution.projection_quantization, bound)
    execution.capabilities.append("packed_transfer_projection")
    return _reshape_sample(words, context)


def _dense_sample(payload: Any, context: _SamplingContext, *, field: Field) -> Any:
    from spf.renderer.assets import SpectralTexture
    from spf.renderer.blocks import BlockView

    if isinstance(payload, BlockView):
        return _reshape_sample(payload.sample(context.points), context)
    if isinstance(payload, SpectralTexture):
        return _reshape_sample(payload.sample(context.points, normalized=False, filter="bilinear"), context)
    reconstruct = getattr(payload, "reconstruct", None)
    value = reconstruct() if callable(reconstruct) else _payload_value(payload)
    if callable(value):
        value = value(context.points)
        return _reshape_sample(value, context)
    if isinstance(value, (v02.LogPos, v02.LogSigned, v02.Phase, v02.SPF32Value)):
        return np.full(context.output_shape, value, dtype=object)
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(context.output_shape, array.item(), dtype=array.dtype)
    if array.shape == context.output_shape:
        return array.copy()
    if array.size == context.element_count:
        return array.reshape(
            context.output_shape,
            order=cast(Literal["C", "F", "A"], context.order),
        ).copy()
    if context.layout is not None and array.ndim == len(context.layout.shape):
        indices = np.rint(context.layout.points()).astype(int)
        if all(np.all((indices[:, axis] >= 0) & (indices[:, axis] < array.shape[axis])) for axis in range(array.ndim)):
            sampled = array[tuple(indices[:, axis] for axis in range(array.ndim))]
            return sampled.reshape(context.output_shape, order=context.order)
    raise MaterializationError(f"dense payload shape {array.shape} cannot satisfy sample layout {context.output_shape}")


def _sparse_dense(payload: Any, context: _SamplingContext, *, field: Field) -> Any:
    base = None
    residual = payload
    if isinstance(payload, Mapping):
        base = payload.get("base", payload.get("background"))
        residual = payload.get("residual", payload.get("sparse", payload.get("values", payload)))
    if isinstance(residual, SparseResidual):
        dense = residual.to_dense()
    elif isinstance(residual, Mapping):
        values = residual.get("values")
        indices = residual.get("indices")
        shape = residual.get("shape", context.output_shape)
        if values is None or indices is None:
            dense = np.asarray(residual)
        else:
            shape_tuple = tuple(int(value) for value in np.asarray(shape).reshape(-1))
            dense = np.zeros(shape_tuple, dtype=np.asarray(values).dtype)
            index_array = np.asarray(indices, dtype=np.int64)
            if index_array.ndim == 1 and len(shape_tuple) == 1:
                dense.reshape(-1)[index_array] = np.asarray(values).reshape(-1)
            else:
                index_array = index_array.reshape(-1, len(shape_tuple))
                for index, value in zip(index_array, np.asarray(values).reshape(-1), strict=True):
                    dense[tuple(int(item) for item in index)] = value
    else:
        dense = np.asarray(residual)
    result = _dense_sample(dense, context, field=field)
    if base is not None:
        result = _dense_sample(base, context, field=field) + result
    return result


def _as_complex_value(value: Any) -> complex:
    if isinstance(value, v02.LogPos):
        return complex(value.value)
    if isinstance(value, v02.LogSigned):
        return complex(value.value)
    if isinstance(value, v02.SPF32Value):
        return complex(decode_words(value.word))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return complex(decode_words(int(value)))
    return complex(value)


def _hermitian_from_pairs(payload: Any, context: _SamplingContext) -> Any:
    if isinstance(payload, Mapping):
        coefficients = payload.get("coefficients", payload.get("values"))
        phases = payload.get("phases", payload.get("phase"))
        flags = payload.get("self_conjugate", payload.get("endpoints"))
        if coefficients is not None and phases is not None:
            coefficient_values = tuple(np.asarray(coefficients, dtype=object).reshape(-1).tolist())
            phase_values = np.asarray(phases, dtype=object)
            if phase_values.ndim == 0:
                phase_values = np.broadcast_to(phase_values, (len(coefficient_values),))
            result = v02.hermitian_synthesis(coefficient_values, phase_values, self_conjugate=flags)
            if isinstance(result, (v02.LogSigned, v02.LogPos, v02.SPF32Value)):
                scalar = _as_complex_value(result).real
                return np.full(context.output_shape, scalar, dtype=np.float64)
            array = np.asarray(result, dtype=object)
            physical = np.asarray(
                _map_object(array, lambda item: float(_as_complex_value(item).real), dtype=np.float64)
            )
            return _reshape_sample(physical, context)
        pairs = payload.get("pairs")
        if pairs is not None:
            terms: list[Any] = []
            for pair in pairs:
                if isinstance(pair, Mapping):
                    terms.append(
                        v02.hermitian_pair_term(
                            pair.get("coefficient", pair.get("value")),
                            pair.get("phase", 0),
                            self_conjugate=bool(pair.get("self_conjugate", False)),
                        )
                    )
                else:
                    values = tuple(pair)
                    if len(values) == 2:
                        terms.append(v02.hermitian_pair_term(values[0], values[1]))
                    elif len(values) == 3:
                        terms.append(v02.hermitian_pair_term(values[0], values[1], self_conjugate=bool(values[2])))
                    else:
                        raise MaterializationError(
                            "Hermitian pair entries must contain coefficient, phase, and optional endpoint flag"
                        )
            reduced = v02.reduce_logsigned(terms).value
            return np.full(context.output_shape, float(_as_complex_value(reduced).real), dtype=np.float64)
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return _hermitian_from_pairs({"coefficients": payload[0], "phases": payload[1]}, context)
    raise MaterializationError("Hermitian payload is not a supported coefficient/pair form")


def _full_spectrum_from_sparse(payload: Mapping[str, Any]) -> tuple[np.ndarray, tuple[int, ...]]:
    shape_raw = payload.get("shape", payload.get("full_shape", payload.get("spatial_shape")))
    if shape_raw is None:
        raise MaterializationError("Hermitian sparse spectrum requires full shape")
    shape = tuple(int(value) for value in np.asarray(shape_raw).reshape(-1))
    if not shape or any(value <= 0 for value in shape):
        raise MaterializationError("Hermitian full shape must contain positive dimensions")
    indices = payload.get("indices", payload.get("coordinates"))
    values = payload.get("values", payload.get("coefficients"))
    if indices is None or values is None:
        raise MaterializationError("Hermitian sparse spectrum requires indices and values")
    index_array = np.asarray(indices, dtype=np.int64).reshape(-1, len(shape))
    value_array = np.asarray(values, dtype=object).reshape(-1)
    if index_array.shape[0] != value_array.size:
        raise MaterializationError("Hermitian indices and values must have equal lengths")
    spectrum = np.zeros(shape, dtype=np.complex128)
    for index, value in zip(index_array, value_array, strict=True):
        coordinate = tuple(int(item) for item in index)
        if any(item < 0 or item >= size for item, size in zip(coordinate, shape, strict=True)):
            raise MaterializationError("Hermitian spectrum index is outside full shape")
        complex_value = _as_complex_value(value)
        partner = tuple((-coordinate[axis]) % shape[axis] for axis in range(len(shape)))
        spectrum[coordinate] = complex_value
        spectrum[partner] = np.conj(complex_value)
    return spectrum, shape


def _hermitian_spectrum(payload: Any, context: _SamplingContext) -> Any:
    if isinstance(payload, Mapping) and (
        payload.get("indices") is not None
        and payload.get("values", payload.get("coefficients")) is not None
        and payload.get("shape", payload.get("full_shape", payload.get("spatial_shape"))) is not None
    ):
        from spf.renderer.ocean import sparse_signed_real_synthesis

        shape = tuple(
            int(item)
            for item in np.asarray(
                payload.get("shape", payload.get("full_shape", payload.get("spatial_shape")))
            ).reshape(-1)
        )
        spatial = sparse_signed_real_synthesis(
            payload["values"] if payload.get("values") is not None else payload["coefficients"],
            payload["indices"],
            shape,
            phases=payload.get("phases", payload.get("phase")),
        )
        if context.output_shape == shape:
            return spatial
        return _dense_sample(spatial, context, field=None)  # type: ignore[arg-type]

    mapping = payload if isinstance(payload, Mapping) else {}
    spectrum = mapping.get("spectrum", mapping.get("half_spectrum"))
    if spectrum is None and isinstance(payload, Mapping) and "values" in payload:
        spectrum = payload["values"]
    shape_raw = mapping.get("shape", mapping.get("full_shape", mapping.get("spatial_shape")))
    if spectrum is None:
        raise MaterializationError("Hermitian payload requires a spectrum or coefficient/phase pairs")
    values = np.asarray(spectrum, dtype=object)
    complex_values = np.asarray(_map_object(values, _as_complex_value, dtype=np.complex128))
    shape = (
        context.output_shape if shape_raw is None else tuple(int(item) for item in np.asarray(shape_raw).reshape(-1))
    )
    if complex_values.shape == shape:
        raise MaterializationError(
            "HERMITIAN_HALF requires a nonredundant rfftn half-spectrum; "
            "a full spectrum must use an explicit dense fallback"
        )
    from spf.renderer.ocean import HermitianHalfSpectrum

    # The v0.2 public Hermitian path is a direct signed-real pair synthesis,
    # not a hidden NumPy irfftn.  HermitianHalfSpectrum validates endpoint
    # conjugacy and its sparse() method keeps only canonical pair terms.
    spatial = HermitianHalfSpectrum(complex_values, shape).sparse()
    return _reshape_sample(spatial, context)


def _sample_hermitian(payload: Any, context: _SamplingContext) -> np.ndarray:
    if isinstance(payload, Mapping) and (
        payload.get("phases", payload.get("phase")) is not None or payload.get("pairs") is not None
    ):
        return np.asarray(_hermitian_from_pairs(payload, context), dtype=np.float64)
    if isinstance(payload, (tuple, list)) and len(payload) == 2:
        return np.asarray(_hermitian_from_pairs(payload, context), dtype=np.float64)
    return np.asarray(_hermitian_spectrum(payload, context), dtype=np.float64)


def _sample_atom(atom: v02.GaussianChirpAtom, context: _SamplingContext) -> np.ndarray:
    # §17.3: the sample points are real coordinates; the atom's own evaluator
    # owns the exponent arithmetic.  Widening the coordinate plane to a
    # Cartesian pair plane first allocated a second dense plane for nothing.
    values = atom.evaluate(np.asarray(context.points, dtype=np.float64))
    return np.asarray(_reshape_sample(values, context))


def _sample_residual(residual: Any, context: _SamplingContext) -> np.ndarray:
    from spf.renderer.normalization import _ScaledResidual

    if isinstance(residual, _ScaledResidual):
        return residual.scale_values(_sample_residual(residual.residual, context))
    if isinstance(residual, SparseResidual):
        if context.output_shape == residual.shape and not context.explicit_points:
            return residual.sample_flat(np.arange(context.element_count))
        indices = np.asarray(context.points, dtype=np.float64)
        if (
            indices.shape[-1] == len(residual.shape)
            and np.all(indices == np.rint(indices))
            and np.all(indices >= 0)
            and np.all(indices < np.asarray(residual.shape))
        ):
            linear = np.ravel_multi_index(tuple(indices.astype(np.int64).T), residual.shape)
            return residual.sample_flat(linear)
    if isinstance(residual, _ResidualSum):
        # Accumulate in the residual stream's own carrier: a pre-seeded
        # Cartesian accumulator promoted every real residual to a pair plane.
        total: np.ndarray | None = None
        for item in residual.items:
            sampled = _sample_residual(item, context)
            total = sampled if total is None else total + sampled
        return np.zeros(context.element_count) if total is None else total
    if isinstance(residual, SparseResidual):
        if context.explicit_points or context.layout is None or residual.shape != context.output_shape:
            raise MaterializationError(
                "a sparse residual requires a matching regular-grid materialization before geometric transforms"
            )
        dense = np.asarray(residual.to_dense())
        if context.order == "F":
            return dense.reshape(-1, order="F")
        return dense.reshape(-1, order="C")
    value = v02._residual_value(residual, context.points)
    return np.asarray(value).reshape(-1)


def _sample_mixture(mixture: v02.AtomMixture, context: _SamplingContext, *, execution: _Execution) -> np.ndarray:
    execution.atoms_considered = max(execution.atoms_considered, len(mixture.atoms))
    from spf.renderer.mixture_terminal import evaluate_atoms

    native = evaluate_atoms(mixture.atoms, context.points)
    if native is not None:
        execution.capabilities.append("native_compensated_atom_pivot")
        execution.pivot_reductions += context.element_count
        execution.reduction_operations += context.element_count
        if mixture.residual is not None:
            native += _sample_residual(mixture.residual, context)
        return np.asarray(_reshape_sample(native, context))
    if mixture.atoms:
        # §9.4 ordered mixture reduction in the atoms' own carrier.  The former
        # pre-seeded Cartesian accumulator widened every atom plane a second
        # time; seeding from the first atom keeps one plane per term.
        accumulated: np.ndarray | None = None
        for atom in mixture.atoms:
            sampled = np.asarray(atom.evaluate(context.points))
            accumulated = sampled if accumulated is None else accumulated + sampled
        values = cast(np.ndarray, accumulated)
        execution.reduction_operations += context.element_count
        execution.complex_adds += context.element_count * max(0, len(mixture.atoms) - 1)
    else:
        values = np.zeros(context.element_count)
    if mixture.residual is not None:
        values = values + _sample_residual(mixture.residual, context)
    return np.asarray(_reshape_sample(values, context))


def _sample_jet(jet: v02.Jet2D, context: _SamplingContext) -> Any:
    if context.output_shape and len(context.output_shape) == 2 and not context.explicit_points:
        if context.element_count == 0:
            return np.empty(context.output_shape, dtype=np.asarray(jet.value).dtype)
        points = np.asarray(context.points, dtype=np.float64).reshape(
            (*context.output_shape, 2),
            order=cast(Literal["C", "F", "A"], context.order),
        )
        origin = points[0, 0]
        x_step = points[0, 1] - origin if context.output_shape[1] > 1 else np.zeros(2)
        y_step = points[1, 0] - origin if context.output_shape[0] > 1 else np.zeros(2)
        index_jet = v02.jet_transform(jet, np.column_stack((x_step, y_step)), origin)
        from spf.renderer.native import _jet_recurrence

        values = _jet_recurrence(index_jet, context.output_shape)
    else:
        points = np.asarray(context.points)
        values = (
            jet.value
            + jet.grad_x * points[:, 0]
            + jet.grad_y * points[:, 1]
            + 0.5 * jet.hess_xx * points[:, 0] ** 2
            + jet.hess_xy * points[:, 0] * points[:, 1]
            + 0.5 * jet.hess_yy * points[:, 1] ** 2
        )
    return _reshape_sample(values, context)


def _sample_separable(
    field_value: v02.SeparableField, context: _SamplingContext, *, execution: _Execution, field: Field
) -> Any:
    from spf.renderer.separable_terminal import evaluate_grid

    kind = _field_kind_name(field)
    values, bound, phase_bound = evaluate_grid(field_value, context.points, kind=kind)
    execution.reducer_error = max(execution.reducer_error, bound)
    execution.reducer_phase_error = max(execution.reducer_phase_error, phase_bound)
    execution.capabilities.append("batched_separable_rank_consumer")
    execution.reduction_operations += context.element_count
    if kind in _POSITIVE_KINDS or kind in _SIGNED_KINDS:
        execution.positive_logadds += context.element_count
    elif kind in _COMPLEX_KINDS:
        execution.complex_adds += context.element_count
    return _reshape_sample(values, context)


def _sample_payload(field: Field, payload: Any, context: _SamplingContext, execution: _Execution) -> Any:
    kind = _representation_kind(field.representation.kind)
    if isinstance(payload, v02.AtomMixture):
        kind = RepresentationKind.ATOM_MIXTURE
    if kind is RepresentationKind.GAUSSIAN_CHIRP_ATOM:
        execution.capabilities.append("atom_sample")
        return _sample_atom(payload, context)
    if kind is RepresentationKind.ATOM_MIXTURE:
        execution.capabilities.append("mixture_reduce")
        return _sample_mixture(payload, context, execution=execution)
    if kind is RepresentationKind.QUADRATIC_JET:
        execution.capabilities.append("jet_forward_difference")
        if isinstance(payload, v02.Jet2D):
            return _sample_jet(payload, context)
        if not context.explicit_points and len(context.output_shape) == 2:
            from spf.renderer.emission import emit_selected
            from spf.renderer.native import _INTENSITY_LUT, _MAGNITUDE_LUT

            emitted = emit_selected(
                payload,
                context.output_shape,
                (0, 0),
                (slice(0, context.output_shape[0]), slice(0, context.output_shape[1])),
                semantic="complex",
                squared=False,
                epsilon=0,
                magnitude_lut=_MAGNITUDE_LUT,
                intensity_lut=_INTENSITY_LUT,
                retain_codes=False,
            )
            if emitted is not None:
                execution.capabilities.append("fused_complex_jet_terminal")
                return emitted.values
        if hasattr(payload, "materialize"):
            return _reshape_sample(payload.materialize(), context)
        raise TypeError("QUADRATIC_JET payload must be Jet2D or a materializable jet tile")
    if kind is RepresentationKind.SEPARABLE:
        execution.capabilities.append("separable_reduce")
        return _sample_separable(payload, context, execution=execution, field=field)
    if kind is RepresentationKind.SPHERICAL_GAUSSIAN_ATOM:
        if not isinstance(payload, v02.SphericalGaussian):
            raise TypeError("SPHERICAL_GAUSSIAN_ATOM payload must be SphericalGaussian")
        if context.points.shape[-1] != 3:
            raise MaterializationError("spherical Gaussian materialization requires three-dimensional directions")
        execution.capabilities.append("spherical_gaussian_sample")
        return _reshape_sample(payload.evaluate(context.points), context)
    if kind is RepresentationKind.HERMITIAN_HALF:
        execution.capabilities.append("hermitian_synthesis")
        execution.reduction_operations += max(1, context.element_count)
        execution.positive_logadds += max(1, context.element_count)
        # Planner candidates store NumPy's rfftn half-spectrum as the raw
        # payload and keep its encoding/full shape in representation
        # metadata. Preserve that metadata at the runtime boundary; passing
        # only the array makes the public ``Field.materialize`` path look for
        # coefficient/phase pairs and fail closed.
        hermitian_metadata = dict(field.representation.metadata)
        if hermitian_metadata.get("encoding") == "rfftn" and not isinstance(payload, Mapping):
            payload = {"spectrum": payload, **hermitian_metadata}
        return _sample_hermitian(payload, context)
    if kind is RepresentationKind.SPARSE_RESIDUAL:
        execution.capabilities.append("sparse_synthesis")
        return _sparse_dense(payload, context, field=field)
    if kind in {RepresentationKind.DENSE_TYPED, RepresentationKind.DENSE_CARTESIAN_REFERENCE}:
        execution.direct_dense = True
        execution.capabilities.append("dense_payload")
        packed = _packed_projection(payload, context, field=field, execution=execution)
        if packed is not None:
            return packed
        return _dense_sample(payload, context, field=field)
    raise MaterializationError(f"no materializer is registered for representation {kind.value}")


def _effective_source_kind(field: Field, values: Any) -> str:
    kind = _field_kind_name(field)
    if kind in {"radiance", "incoherent_radiance"}:
        return "log_pos"
    if kind in {"words", "word", "uint32"}:
        return "words"
    if kind == "any":
        np.asarray(values)
        return "dense"
    return kind


def _semantic_source(field: Field, values: Any) -> tuple[Any, np.ndarray, str]:
    source_kind = _effective_source_kind(field, values)
    if source_kind in _COMPLEX_KINDS and isinstance(values, np.ndarray) and values.dtype == np.uint32:
        source_kind = "words"
    metadata = dict(field.metadata)
    metadata.update(dict(field.representation.metadata))
    phase_units = _normal_name(metadata.get("phase_units", "code"))
    if source_kind in _POSITIVE_KINDS:
        complex_values = _complex_array(values, kind="log_pos", phase_units=phase_units)
        imaginary = np.abs(complex_values.imag)
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(complex_values.real))
        if np.any(imaginary > tolerance):
            raise MaterializationError("LOG_POS materialization received a complex-valued source")
        real = np.asarray(complex_values.real, dtype=np.float64)
        if np.any(real < -1.0e-12):
            raise MaterializationError("LOG_POS materialization received negative values")
        real = np.where(real < 0.0, 0.0, real)
        return real, real, "log_pos"
    if source_kind in _SIGNED_KINDS:
        complex_values = _complex_array(values, kind="log_signed", phase_units=phase_units)
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(complex_values.real))
        if np.any(np.abs(complex_values.imag) > tolerance):
            raise MaterializationError("LOG_SIGNED materialization requires real-valued samples")
        real = np.asarray(complex_values.real, dtype=np.float64)
        return real, real, "log_signed"
    if source_kind in _PHASE_KINDS:
        complex_values = _complex_array(values, kind="phase", phase_units=phase_units)
        return values, complex_values, "phase"
    if source_kind in {"words", "word", "uint32"}:
        words = np.asarray(values, dtype=np.uint32)
        return words, cast(np.ndarray, _WordSource(words)), "words"
    complex_values = _complex_array(values, kind=source_kind, phase_units=phase_units)
    if field.coherence is Coherence.INCOHERENT_POSITIVE:
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(complex_values.real))
        if np.any(np.abs(complex_values.imag) > tolerance) or np.any(complex_values.real < -tolerance):
            raise MaterializationError("incoherent-positive field produced a non-positive real sample")
        real = np.where(complex_values.real < 0.0, 0.0, complex_values.real)
        return real, real, "log_pos"
    if field.coherence is Coherence.SIGNED_REAL:
        tolerance = 1.0e-12 * np.maximum(1.0, np.abs(complex_values.real))
        if np.any(np.abs(complex_values.imag) > tolerance):
            raise MaterializationError("signed-real field produced complex samples")
        real = np.asarray(complex_values.real, dtype=np.float64)
        return real, real, "log_signed"
    return np.asarray(values), complex_values, source_kind


def _zero_for_source(source_kind: str, shape: tuple[int, ...]) -> Any:
    if source_kind in {"words", "uint32", "word"}:
        return np.full(shape, v02.SPF32_WORD_ZERO, dtype=np.uint32)
    if source_kind in {"log_pos", "log_signed", "phase"}:
        return np.zeros(shape, dtype=np.float64)
    return np.zeros(shape, dtype=np.complex128)


def _broadcast_mask(mask: Any, context: _SamplingContext) -> np.ndarray:
    array = np.asarray(mask, dtype=bool)
    if array.ndim == 0:
        return np.full(context.output_shape, bool(array), dtype=bool)
    if array.shape == context.output_shape:
        return array
    if array.size == context.element_count:
        return np.asarray(
            array.reshape(
                context.output_shape,
                order=cast(Literal["C", "F", "A"], context.order),
            ),
            dtype=bool,
        )
    try:
        return np.asarray(np.broadcast_to(array, context.output_shape), dtype=bool)
    except ValueError as exc:
        raise MaterializationError("mask shape does not match the materialization layout") from exc


def _frequency_grid(shape: tuple[int, ...], spacing: Sequence[float]) -> tuple[np.ndarray, ...]:
    from spf.renderer.carriers import dft_frequency_grid

    return dft_frequency_grid(shape, spacing)


def _dense_translate(
    values: np.ndarray,
    offset: np.ndarray,
    context: _SamplingContext,
    field: Field,
) -> np.ndarray:
    if context.layout is None or values.ndim != len(offset):
        raise MaterializationError("dense translation requires a regular grid")
    if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
        coordinates = context.points.reshape(
            (*context.output_shape, len(offset)),
            order=cast(Literal["C", "F", "A"], context.order),
        )
        phase = -_TAU * np.einsum("...i,i->...", coordinates, offset)
        return np.asarray(values * np.exp(1j * phase), dtype=np.complex128)
    if field.domain not in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
        raise MaterializationError(
            f"translation is not defined for domain {getattr(field.domain, 'value', field.domain)}"
        )
    # Exact periodic band-limited translation.  Integer shifts reduce to the
    # same operation as ``roll``; fractional shifts retain the requested phase
    # instead of being silently rounded to a sample offset.
    spacing = tuple(float(step) for step in context.layout.spacing)
    frequencies = _frequency_grid(values.shape, spacing)
    phase_cycles = np.zeros(values.shape, dtype=np.float64)
    for axis_frequency, displacement in zip(frequencies, offset, strict=True):
        phase_cycles += axis_frequency * float(displacement)
    shifted = np.fft.ifftn(np.fft.fftn(values) * np.exp(-1j * _TAU * phase_cycles))
    if not np.iscomplexobj(values) and np.all(
        np.abs(np.imag(shifted)) <= 1.0e-12 * np.maximum(1.0, np.abs(np.real(shifted)))
    ):
        return np.asarray(np.real(shifted), dtype=np.float64)
    return np.asarray(shifted, dtype=np.complex128)


def _index_affine(matrix: np.ndarray, offset: np.ndarray, spacing: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Rewrite a physical map ``x -> A x + b`` in sample-index coordinates.

    With ``x = S n`` and ``S = diag(spacing)`` the index map is
    ``n -> (S^-1 A S) n + S^-1 b``.
    """

    scale = np.asarray(tuple(float(step) for step in spacing), dtype=np.float64)
    return (matrix * scale[None, :]) / scale[:, None], np.asarray(offset, dtype=np.float64) / scale


def _physical_index_spacing(field: Field, context: _SamplingContext, dimension: int) -> tuple[float, ...]:
    """Physical step per sample index along each axis, from the layout and the chart.

    A coordinate remap states ``A`` and ``b`` in the field's physical units, so
    the index-space map needs the step that one index advances.  Both the sample
    layout and a chart can carry that step (they are multiplied when the chart is
    diagonal); a non-diagonal chart does not reduce to a per-axis step and a
    coordinate remap on one is refused rather than silently mis-scaled.
    """

    steps = np.ones(dimension, dtype=np.float64)
    if context.layout is not None and context.layout.spacing:
        steps = np.asarray([float(step) for step in context.layout.spacing], dtype=np.float64)
    chart = field.chart
    if chart.dimension == dimension:
        basis = np.asarray(chart.basis, dtype=np.float64) @ np.diag(
            np.asarray(chart.scale, dtype=np.float64) * np.asarray(chart.sample_spacing, dtype=np.float64)
        )
        if np.count_nonzero(basis - np.diag(np.diagonal(basis))):
            raise MaterializationError("a coordinate remap requires an axis-aligned chart")
        steps = steps * np.abs(np.diagonal(basis))
    return tuple(float(step) for step in steps)


def _lattice_index_map(
    matrix: np.ndarray, offset: np.ndarray, shape: Sequence[int], spacing: Sequence[float]
) -> tuple[np.ndarray, ...] | None:
    """Per-axis integer index arrays when the map sends every sample to a sample.

    ``None`` when the map lands off the lattice, which is the case that needs
    interpolation.  A rotation by a multiple of a quarter turn, an axis
    exchange, a reflection and an integer scale or shift all land here, so those
    stay pure index permutations of the carrier with no arithmetic at all.
    """

    index_matrix, index_offset = _index_affine(matrix, offset, spacing)
    rounded = np.rint(index_matrix)
    rounded_offset = np.rint(index_offset)
    if not (
        np.allclose(index_matrix, rounded, rtol=0.0, atol=1.0e-9)
        and np.allclose(index_offset, rounded_offset, rtol=0.0, atol=1.0e-9)
    ):
        return None
    dims = tuple(int(n) for n in shape)
    grid = np.indices(dims)
    mapped = []
    for axis, size in enumerate(dims):
        total = np.full(dims, int(rounded_offset[axis]), dtype=np.int64)
        for source in range(len(dims)):
            total = total + int(rounded[axis, source]) * grid[source]
        mapped.append(total % size)
    return tuple(mapped)


def _reference_affine_resample(
    values: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    spacing: Sequence[float],
) -> tuple[np.ndarray, float]:
    """Design §17.2/§17.3 coordinate remap ``g(y) = f(A y + b)`` on a periodic grid.

    Returns the resampled array and a stated absolute bound on the resampling
    error.  A map that carries the sample lattice into itself is executed as a
    pure index permutation and is exact (bound 0).  Otherwise the field is read
    with periodic multilinear interpolation and the bound is the Peano remainder
    of linear interpolation, ``sum_i t_i (1 - t_i) / 2 * max |second difference
    along axis i|`` maximised over the output points; this is a bound for a
    field whose second derivative along each axis is attained on the sample
    lattice, which is the design's own sampled-field model (§7.3), and is stated
    rather than proved for content at the Nyquist limit.
    """

    dims = tuple(int(n) for n in values.shape)
    lattice = _lattice_index_map(matrix, offset, dims, spacing)
    if lattice is not None:
        return np.asarray(values[lattice]), 0.0
    index_matrix, index_offset = _index_affine(matrix, offset, spacing)
    grid = np.stack(tuple(np.indices(dims)), axis=-1).astype(np.float64)
    mapped = grid @ index_matrix.T + index_offset
    floor = np.floor(mapped)
    fraction = mapped - floor
    base = floor.astype(np.int64)
    result = np.zeros(dims, dtype=values.dtype)
    for corner in range(1 << len(dims)):
        weight = np.ones(dims, dtype=np.float64)
        index: list[np.ndarray] = []
        for axis, size in enumerate(dims):
            step = (corner >> axis) & 1
            weight = weight * (fraction[..., axis] if step else 1.0 - fraction[..., axis])
            index.append((base[..., axis] + step) % size)
        result = result + weight * values[tuple(index)]
    second = 0.0
    for axis in range(len(dims)):
        difference = np.diff(values, n=2, axis=axis, append=values.take([0, 1], axis=axis))
        second += 0.125 * float(np.max(np.abs(difference))) if difference.size else 0.0
    return np.asarray(result), float(second)


def _dense_gaussian_filter(values: np.ndarray, covariance: Any, context: _SamplingContext, field: Field) -> np.ndarray:
    dimension = values.ndim
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (dimension, dimension):
        raise MaterializationError("dense Gaussian filter covariance does not match value dimensions")
    if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
        coordinates = context.points.reshape((*context.output_shape, dimension))
        exponent = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", coordinates, matrix, coordinates)
        return np.asarray(values * np.exp(exponent))
    spacing = context.layout.spacing if context.layout is not None else (1.0,) * dimension
    grids = _frequency_grid(values.shape, spacing)
    frequency = np.stack(grids, axis=-1)
    exponent = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", frequency, matrix, frequency)
    transformed = np.fft.fftn(values) * np.exp(exponent)
    result = np.fft.ifftn(transformed)
    if np.all(np.abs(np.imag(result)) <= 1.0e-12 * np.maximum(1.0, np.abs(np.real(result)))):
        return np.asarray(np.real(result), dtype=np.float64)
    return np.asarray(result)


def _dense_lens(values: np.ndarray, params: Mapping[str, Any], context: _SamplingContext) -> np.ndarray:
    if not np.iscomplexobj(values):
        values = np.asarray(values, dtype=np.complex128)
    focal_raw = params.get("focal_length")
    wavelength_raw = params.get("wavelength")
    if focal_raw is None or wavelength_raw is None:
        raise MaterializationError("lens phase requires focal_length and wavelength")
    focal = float(focal_raw)
    wavelength = float(wavelength_raw)
    if focal == 0.0 or wavelength <= 0.0:
        raise MaterializationError("lens phase requires a finite nonzero focal length and positive wavelength")
    dimension = context.points.shape[-1]
    coordinates = context.points.reshape((*context.output_shape, dimension))
    radius2 = np.sum(np.square(coordinates[..., : min(2, dimension)]), axis=-1)
    phase = -math.pi * radius2 / (wavelength * focal)
    return np.asarray(values * np.exp(1j * phase), dtype=np.complex128)


def _unit_phasor_words(codes: np.ndarray) -> np.ndarray:
    """Unit-magnitude words carrying a per-element phase code plane."""

    theta = (np.asarray(np.rint(codes), dtype=np.int64) & v02.SPF32_PHASE_MASK).astype(np.uint16)
    return np.asarray(_core.pack_words(np.zeros(theta.shape, dtype=np.int16), theta), dtype=np.uint32)


def _log_gain_words(log2_gain: np.ndarray) -> np.ndarray:
    """Zero-phase words for a real positive gain given as log2; underflow is the zero word."""

    scaled = np.asarray(log2_gain, dtype=np.float64) * v02.SPF32_L_SCALE
    codes = np.rint(np.clip(scaled, v02.SPF32_L_MIN_FINITE - 1, v02.SPF32_L_MAX_FINITE)).astype(np.int64)
    codes = np.where(np.isnan(scaled), v02.SPF32_L_MIN_FINITE - 1, codes)
    L = np.where(codes < v02.SPF32_L_MIN_FINITE, v02.SPF32_L_ZERO, codes).astype(np.int16)
    packed = _core.pack_words(L, np.zeros(L.shape, dtype=np.uint16))
    return np.asarray(packed, dtype=np.uint32)


def _word_live(words: np.ndarray) -> np.ndarray:
    L = np.asarray(_core.unpack_L(words))
    return np.asarray((L != v02.SPF32_L_ZERO) & (words != np.uint32(v02.SPF32_WORD_NAN)), dtype=bool)


def _word_positive(words: np.ndarray) -> bool:
    theta = _core.unpack_theta(words)
    return bool(np.all((theta == 0) | ~_word_live(words)))


def _word_scale_log(words: np.ndarray, factor: float) -> np.ndarray:
    """``|w| ** factor`` for zero-phase words: an integer scale of the L plane."""

    L = _core.unpack_L(words).astype(np.int64)
    live = _word_live(words)
    scaled = np.clip(np.rint(L.astype(np.float64) * factor), v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE)
    out = _core.pack_words(scaled.astype(np.int16), np.zeros(L.shape, dtype=np.uint16))
    return np.asarray(np.where(live, out, words), dtype=np.uint32)


def _word_reinhard(words: np.ndarray, execution: _Execution) -> np.ndarray:
    """``x / (1 + x)`` on positive words through the library reducer and divide."""

    from spf.tensor import reducers

    one = np.full(words.shape, int(v02.pack_spf32(0, 0)), dtype=np.uint32)
    terms = np.stack((one, words), axis=-1)
    spec = _core.ensure_reducer_spec(_core.ReducerPolicy.AUTO)
    denominator = np.asarray(reducers.reduce_axis(terms, spec, axis=-1), dtype=np.uint32).reshape(words.shape)
    execution.reduction_operations += int(words.size)
    execution.positive_logadds += int(words.size)
    return np.asarray(v02.spf_div(words, denominator), dtype=np.uint32)


def _word_fft_nd(words: np.ndarray, *, inverse: bool, precision: str, counters: v02.PerformanceCounters) -> np.ndarray:
    """Separable N-d v0.2 FFT on a packed word plane (:mod:`spf.tensor.v02_planes`)."""

    from spf.tensor import v02_planes as planes

    transform = planes.spf_ifftn_radix2 if inverse else planes.spf_fftn_radix2
    return np.asarray(transform(words, precision, counters=counters), dtype=np.uint32)


def _word_fft_shape_ok(shape: tuple[int, ...]) -> bool:
    return bool(shape) and all(2 <= n <= v02.SPF32_INTERNAL_PHASE_MOD and (n & (n - 1)) == 0 for n in shape)


def _word_abs_sum(words: np.ndarray) -> tuple[float, float]:
    """``(sum |w|, min live |w|)`` of a word plane through the magnitude ROM (no decoder call)."""

    from spf.renderer.carriers import magnitude_table

    live = _word_live(words)
    magnitude = magnitude_table()[np.asarray(_core.unpack_L(words), dtype=np.int64) & 0xFFFF]
    total = float(math.fsum(magnitude[live].tolist())) if np.any(live) else 0.0
    smallest = float(np.min(magnitude[live])) if np.any(live) else 0.0
    return total, smallest


_FFT_PAIR_QUANTIZATION = (
    v02._LOG_QUANTIZATION_RELATIVE_ERROR
    + (1.0 + v02._LOG_QUANTIZATION_RELATIVE_ERROR) * (math.pi / v02.SPF32_PHASE_MOD)
    + math.pi / v02.SPF32_PHASE_MOD  # one twiddle rounding per stage
)


def _word_transform_suffix(
    field: Field,
    words: np.ndarray,
    operation: FieldOperation,
    context: _SamplingContext,
    execution: _Execution,
) -> np.ndarray | None:
    """Spatial translate / Gaussian filter on packed words: v0.2 FFT, word transfer, v0.2 IFFT.

    Every stage is a library word operation: :func:`spf.tensor.v02_planes.spf_fft_radix2`
    per axis, ``v02.spf_mul`` against the transfer words (a unit phasor plane for
    translation, a zero-phase log-gain plane for the Gaussian) and the inverse
    plane transform.  The error certificate follows the convention of
    ``v02._reduce_complex_pair_tree``: each output passes through one FB3 pair
    per stage, each bounded relative to the row magnitude, so with ``S`` stages
    across all axes, ``A = sum|x|``, ``B = sum|X|`` and ``q`` the per-pair
    quantization, ``|error| <= A*expm1(S*log1p(q)) + B*(2*rho + rho^2 + eps*(1+rho))/N``.
    Returns ``None`` for shapes the radix-2 transform cannot take (a non-power-of-two
    axis), which the dispatcher then counts as the reference executor.
    """

    if context.layout is None or words.ndim != len(context.output_shape) or not _word_fft_shape_ok(words.shape):
        return None
    dimension = words.ndim
    spacing = context.layout.spacing if context.layout.spacing else (1.0,) * dimension
    grids = _frequency_grid(words.shape, spacing)
    if operation.op == "translate":
        offset = _offset(operation.params, dimension)
        cycles = np.zeros(words.shape, dtype=np.float64)
        for axis_frequency, displacement in zip(grids, offset, strict=True):
            cycles -= axis_frequency * float(displacement)
        transfer = _unit_phasor_words(cycles * v02.SPF32_PHASE_MOD)
    else:
        matrix = np.asarray(operation.params.get("covariance"), dtype=np.float64)
        if matrix.shape != (dimension, dimension):
            raise MaterializationError("Gaussian filter covariance does not match value dimensions")
        frequency = np.stack(grids, axis=-1)
        exponent = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", frequency, matrix, frequency)
        transfer = _log_gain_words(exponent / math.log(2.0))
    counters = v02.PerformanceCounters()
    spectrum = _word_fft_nd(words, inverse=False, precision=context.precision, counters=counters)
    filtered = np.asarray(v02.spf_mul(spectrum, transfer), dtype=np.uint32)
    output = _word_fft_nd(filtered, inverse=True, precision=context.precision, counters=counters)

    stages = sum(int(math.log2(n)) for n in words.shape)
    eps = math.expm1(stages * math.log1p(_FFT_PAIR_QUANTIZATION))
    rho = _SPF32_RELATIVE_QUANTIZATION
    input_abs, _ = _word_abs_sum(words)
    spectrum_abs, _ = _word_abs_sum(spectrum)
    bound = input_abs * eps + spectrum_abs * (2.0 * rho + rho * rho + eps * (1.0 + rho)) / float(words.size)
    _, smallest = _word_abs_sum(output)
    phase_bound = 0.0 if bound == 0.0 else (math.pi if smallest <= bound else math.asin(bound / smallest))
    execution.transform_error += float(bound)
    execution.transform_phase_error = max(execution.transform_phase_error, float(phase_bound))
    execution.fft_count += 2 * dimension
    execution.complex_adds += int(counters.complex_pair_adds)
    execution.reduction_operations += int(counters.fft_butterflies)
    execution.capabilities.append("word_fft_v02")
    execution.notes.append(
        f"{operation.op}: v0.2 plane FFT on words ({stages} radix-2 stages, "
        f"{int(counters.fft_butterflies)} butterflies), certified transform bound {bound:.3e}"
    )
    return output


def _word_suffix_operation(
    field: Field,
    words: np.ndarray,
    operation: FieldOperation,
    context: _SamplingContext,
    execution: _Execution,
) -> np.ndarray | None:
    """Execute one suffix operation on the packed word plane, or return None.

    Every branch here stays on ``spf.tensor.core`` word arithmetic (rule 3):
    gains and phase shifts are word multiplies, ``abs2`` is ``w * conj(w)``,
    Fourier-domain translation and lens phases are unit-phasor multiplies, a
    Fourier-domain Gaussian is a zero-phase log-gain multiply.  Spatial
    translation and spatial Gaussian filtering run through the v0.2 plane FFT
    (:func:`_word_transform_suffix`) when every axis is a power of two;
    otherwise ``None`` sends them to the counted reference executor.
    """

    op, params = operation.op, operation.params
    fourier = field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}
    if op == "mask":
        mask = _broadcast_mask(params.get("mask"), context)
        return np.asarray(np.where(mask, words, np.uint32(v02.SPF32_WORD_ZERO)), dtype=np.uint32)
    if op == "exposure":
        exposure = float(params.get("ev", params.get("exposure", 0.0)))
        if not math.isfinite(exposure):
            raise MaterializationError("exposure must be finite")
        gain = _log_gain_words(np.asarray(exposure))
        return np.asarray(v02.spf_mul(words, np.uint32(gain)), dtype=np.uint32)
    if op in {"phase_shift", "phase"}:
        delta = params.get("dT", params.get("phase", params.get("delta")))
        if delta is None:
            raise MaterializationError("phase_shift requires dT, phase, or delta")
        raw = float(delta)
        codes = raw * v02.SPF32_PHASE_MOD / _TAU if params.get("phase_units") in {"radians", "rad"} else raw
        return np.asarray(v02.spf_phase_shift(words, round(codes)), dtype=np.uint32)
    if op in {"abs2", "intensity", "magnitude_square"}:
        return np.asarray(v02.spf_mul(words, v02.spf_conj(words)), dtype=np.uint32)
    if op in {"bandlimit", "band_limit"}:
        cutoff = float(params.get("cutoff", params.get("high", math.inf)))
        if context.layout is None:
            raise MaterializationError("bandlimit requires a regular sample layout")
        if fourier:
            coordinates = context.points.reshape((*context.output_shape, context.points.shape[-1]))
            keep = np.linalg.norm(coordinates, axis=-1) <= cutoff
        else:
            grids = _frequency_grid(context.output_shape, context.layout.spacing)
            keep = np.linalg.norm(np.stack(grids, axis=-1), axis=-1) <= cutoff
        return np.asarray(np.where(keep, words, np.uint32(v02.SPF32_WORD_ZERO)), dtype=np.uint32)
    if op in {"power", "tone_power"}:
        if not _word_positive(words):
            raise MaterializationError("power tone mapping requires a positive field")
        return _word_scale_log(words, float(params.get("power", params.get("amount", 1.0))))
    if op in {"reinhard", "tone_reinhard"}:
        if not _word_positive(words):
            raise MaterializationError("Reinhard tone mapping requires a positive field")
        return _word_reinhard(words, execution)
    if op == "lens_phase":
        if field.coherence is not Coherence.COHERENT_COMPLEX:
            raise MaterializationError("lens phase requires coherent-complex semantics")
        focal_raw, wavelength_raw = params.get("focal_length"), params.get("wavelength")
        if focal_raw is None or wavelength_raw is None:
            raise MaterializationError("lens phase requires focal_length and wavelength")
        focal, wavelength = float(focal_raw), float(wavelength_raw)
        if focal == 0.0 or wavelength <= 0.0:
            raise MaterializationError("lens phase requires a finite nonzero focal length and positive wavelength")
        dimension = context.points.shape[-1]
        coordinates = context.points.reshape((*context.output_shape, dimension))
        radius2 = np.sum(np.square(coordinates[..., : min(2, dimension)]), axis=-1)
        cycles = -0.5 * radius2 / (wavelength * focal)
        return np.asarray(v02.spf_mul(words, _unit_phasor_words(cycles * v02.SPF32_PHASE_MOD)), dtype=np.uint32)
    if op == "translate" and fourier:
        offset = _offset(params, len(context.output_shape))
        coordinates = context.points.reshape((*context.output_shape, len(offset)))
        cycles = -np.einsum("...i,i->...", coordinates, np.asarray(offset, dtype=np.float64))
        return np.asarray(v02.spf_mul(words, _unit_phasor_words(cycles * v02.SPF32_PHASE_MOD)), dtype=np.uint32)
    if op == "gaussian_filter" and fourier:
        dimension = len(context.output_shape)
        matrix = np.asarray(params.get("covariance"), dtype=np.float64)
        if matrix.shape != (dimension, dimension):
            raise MaterializationError("Gaussian filter covariance does not match value dimensions")
        coordinates = context.points.reshape((*context.output_shape, dimension))
        exponent = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", coordinates, matrix, coordinates)
        return np.asarray(v02.spf_mul(words, _log_gain_words(exponent / math.log(2.0))), dtype=np.uint32)
    if op in {"rotate", "affine"}:
        # A coordinate map that carries the sample lattice into itself is a pure
        # index permutation of the packed plane: the same uint32 words, moved.
        # Design §17.2 quarter turns, axis exchanges and reflections all qualify,
        # and |det A| = 1 there, so §17.3 adds no log shift.  Anything else needs
        # interpolation between words and goes to the counted reference path.
        from spf.renderer.normalization import _affine_params

        if context.layout is None:
            return None
        matrix, shift = _affine_params(params, words.ndim)
        if abs(abs(float(np.linalg.det(matrix))) - 1.0) > 1.0e-12:
            return None
        spacing = _physical_index_spacing(field, context, words.ndim)
        if fourier:
            matrix = np.asarray(np.linalg.inv(matrix).T, dtype=np.float64)
            shift = np.zeros(words.ndim, dtype=np.float64)
            if np.any(params.get("offset", np.zeros(words.ndim))):
                return None
        lattice = _lattice_index_map(matrix, shift, words.shape, spacing)
        return None if lattice is None else np.asarray(words[lattice], dtype=np.uint32)
    if op in {"translate", "gaussian_filter"} and not fourier:
        return _word_transform_suffix(field, words, operation, context, execution)
    if op in {"display", "display_convert"}:
        return words
    return None


def _apply_dense_operations(
    field: Field,
    values: Any,
    source_kind: str,
    operations: Sequence[FieldOperation],
    context: _SamplingContext,
    execution: _Execution,
) -> tuple[Any, str]:
    """Apply the suffix in order on the live carrier, without another representation pass.

    A packed word carrier executes every word-closed operation through the
    core word ops and stays packed.  The first operation without a word
    datapath decodes once (counted, named) and the rest of the suffix runs on
    the Cartesian reference executor.
    """

    if not operations:
        return values, source_kind
    if source_kind not in {"words", "word", "uint32"}:
        # Typed float/complex carrier: the suffix has no packed plane to run
        # on, so it is the counted Cartesian reference path.
        execution.fallback_conversions += 1
        execution.fallbacks.append(
            f"dense suffix after analytic prefix: {', '.join(operation.op for operation in operations)}"
        )
        return _reference_dense_operations(field, values, source_kind, operations, context, execution)
    words = np.asarray(values, dtype=np.uint32)
    for index, operation in enumerate(operations):
        if operation.op not in _WORD_CLOSED_OPS:
            raise MaterializationError(f"suffix operation {operation.op!r} has no word or reference implementation")
        result = _word_suffix_operation(field, words, operation, context, execution)
        if result is None:
            execution.fallback_conversions += 1
            execution.fallbacks.append(
                f"suffix {operation.op!r} has no word datapath in domain "
                f"{getattr(field.domain, 'value', field.domain)}; decoded to the Cartesian reference executor"
            )
            reference = np.asarray(decode_words(words))
            return _reference_dense_operations(field, reference, "dense", operations[index:], context, execution)
        execution.dense_operations.append(operation.op)
        execution.capabilities.append(f"word_{operation.op}")
        words = np.asarray(result, dtype=np.uint32).reshape(context.output_shape)
    return words, "words"


def _reference_dense_affine(
    values: np.ndarray,
    operation: FieldOperation,
    context: _SamplingContext,
    field: Field,
    execution: _Execution,
) -> np.ndarray:
    """Reference executor for §17.2 rotation and §17.3 general affine.

    A spatial field is resampled as ``g(y) = f(A y + b)``; a Fourier-domain
    field is resampled as the design writes the spectrum,
    ``G(k) = |det A|^-1 exp(i 2 pi (A^-1 b) . k) F(A^-T k)``, so the
    ``ΔL = -256 log2 |det A|`` magnitude shift lands exactly where §17.3 puts
    it.  The resampling bound enters the ledger's ``transform`` slot.
    """

    from spf.renderer.normalization import _affine_params

    if context.layout is None:
        raise MaterializationError("an affine coordinate remap requires a regular sample layout")
    dimension = values.ndim
    matrix, shift = _affine_params(operation.params, dimension)
    spacing = _physical_index_spacing(field, context, dimension)
    fourier = field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}
    if fourier:
        inverse = np.linalg.inv(matrix)
        resampled, bound = _reference_affine_resample(values, inverse.T, np.zeros(dimension), spacing)
        determinant = abs(float(np.linalg.det(matrix)))
        coordinates = context.points.reshape((*context.output_shape, context.points.shape[-1]))
        phase = _TAU * np.einsum("...i,i->...", coordinates, inverse @ shift)
        resampled = resampled * np.cos(phase) + 1.0j * (resampled * np.sin(phase))
        resampled = resampled / determinant
        bound = bound / determinant
    elif field.domain in {Domain.SPATIAL_1D, Domain.SPATIAL_2D, Domain.SPATIAL_3D}:
        resampled, bound = _reference_affine_resample(values, matrix, shift, spacing)
    else:
        raise MaterializationError(
            f"affine transform is not defined for domain {getattr(field.domain, 'value', field.domain)}"
        )
    execution.transform_error += float(bound)
    execution.notes.append(
        f"{operation.op}: reference coordinate resample, "
        f"{'exact index permutation' if bound == 0.0 else 'periodic multilinear interpolation'}, "
        f"stated bound {bound:.3e}"
    )
    return np.asarray(resampled)


def _reference_dense_operations(
    field: Field,
    values: Any,
    source_kind: str,
    operations: Sequence[FieldOperation],
    context: _SamplingContext,
    execution: _Execution,
) -> tuple[Any, str]:
    """Cartesian reference executor for suffix operations with no word datapath.

    This is the ``REFERENCE_CARTESIAN`` path (design v0.2 §7.3, §44.1): it
    decodes the carrier once, applies the remaining operations on complex128,
    and is counted in ``fallback_conversions`` by the dispatcher.  It is never
    reached for a word-closed suffix.
    """

    if not operations:
        return values, source_kind
    working = np.asarray(values)
    current_kind = source_kind
    for operation in operations:
        op = operation.op
        execution.dense_operations.append(op)
        execution.capabilities.append(f"reference_dense_{op}")
        if op == "mask":
            mask = _broadcast_mask(operation.params.get("mask"), context)
            working = np.where(mask, working, v02.SPF32_WORD_ZERO if current_kind == "words" else 0)
            continue
        if op == "translate":
            working_complex = _complex_array(working, kind=current_kind)
            working = _dense_translate(
                working_complex.reshape(context.output_shape),
                _offset(operation.params, working_complex.ndim),
                context,
                field,
            )
            current_kind = "dense"
            continue
        if op == "gaussian_filter":
            working_complex = _complex_array(working, kind=current_kind).reshape(context.output_shape)
            working = _dense_gaussian_filter(working_complex, operation.params.get("covariance"), context, field)
            current_kind = "dense"
            continue
        if op in {"rotate", "affine"}:
            working = _reference_dense_affine(
                _complex_array(working, kind=current_kind).reshape(context.output_shape),
                operation,
                context,
                field,
                execution,
            )
            current_kind = "dense"
            continue
        if op == "exposure":
            exposure = float(operation.params.get("ev", operation.params.get("exposure", 0.0)))
            if not math.isfinite(exposure):
                raise MaterializationError("exposure must be finite")
            if current_kind in _PHASE_KINDS:
                raise MaterializationError("exposure is not defined for a phase-only field")
            if current_kind == "words":
                working = _complex_array(working, kind="words") * (2.0**exposure)
                current_kind = "dense"
            else:
                working = working * (2.0**exposure)
            continue
        if op == "lens_phase":
            if field.coherence is not Coherence.COHERENT_COMPLEX:
                raise MaterializationError("lens phase requires coherent-complex semantics")
            working = _dense_lens(
                _complex_array(working, kind=current_kind).reshape(context.output_shape), operation.params, context
            )
            current_kind = "dense"
            continue
        if op in {"abs2", "intensity", "magnitude_square"}:
            working = np.square(np.abs(_complex_array(working, kind=current_kind))).astype(np.float64)
            current_kind = "log_pos"
            continue
        if op in {"phase_shift", "phase"}:
            delta = operation.params.get("dT", operation.params.get("phase", operation.params.get("delta")))
            if delta is None:
                raise MaterializationError("phase_shift requires dT, phase, or delta")
            raw = float(delta)
            radians = (
                raw if operation.params.get("phase_units") in {"radians", "rad"} else raw * _TAU / v02.SPF32_PHASE_MOD
            )
            working = _complex_array(working, kind=current_kind) * np.exp(1j * radians)
            current_kind = "dense"
            continue
        if op in {"bandlimit", "band_limit"}:
            cutoff = float(operation.params.get("cutoff", operation.params.get("high", math.inf)))
            if not math.isfinite(cutoff) and not math.isinf(cutoff):
                raise MaterializationError("bandlimit cutoff must be finite or infinity")
            if context.layout is None:
                raise MaterializationError("bandlimit requires a regular sample layout")
            if field.domain in {Domain.FOURIER_1D, Domain.FOURIER_2D, Domain.FOURIER_3D}:
                coordinates = context.points.reshape((*context.output_shape, context.points.shape[-1]))
                keep = np.linalg.norm(coordinates, axis=-1) <= cutoff
            else:
                grids = _frequency_grid(context.output_shape, context.layout.spacing)
                keep = np.linalg.norm(np.stack(grids, axis=-1), axis=-1) <= cutoff
            working = np.where(keep, working, v02.SPF32_WORD_ZERO if current_kind == "words" else 0)
            continue
        if op in {"power", "tone_power"}:
            power = float(operation.params.get("power", operation.params.get("amount", 1.0)))
            if current_kind not in _POSITIVE_KINDS:
                raise MaterializationError("power tone mapping requires a positive field")
            working = np.power(np.maximum(np.real(working), 0.0), power)
            current_kind = "log_pos"
            continue
        if op in {"reinhard", "tone_reinhard"}:
            if current_kind not in _POSITIVE_KINDS:
                raise MaterializationError("Reinhard tone mapping requires a positive field")
            working = np.real(working) / (1.0 + np.real(working))
            current_kind = "log_pos"
            continue
        if op in {"display", "display_convert"}:
            # Display is handled at the requested output boundary.  Keeping
            # the marker here preserves operation order and report visibility.
            continue
        raise MaterializationError(f"dense fallback has no implementation for operation {op!r}")
    return working, current_kind


def _typed_logpos(value: float) -> v02.LogPos:
    if math.isnan(float(value)):
        return v02.LogPos.nan()
    return v02.LogPos.from_value(max(0.0, float(value)))


def _typed_logsigned(value: float) -> v02.LogSigned:
    return v02.LogSigned.from_value(float(value))


def _typed_phase(value: complex, *, source_kind: str = "dense") -> v02.Phase:
    if source_kind in _PHASE_KINDS and isinstance(value, v02.Phase):
        return value
    number = complex(value)
    if math.isnan(number.real) or math.isnan(number.imag):
        raise MaterializationError("phase output cannot represent NaN")
    if number == 0.0:
        return v02.Phase(0)
    return v02.Phase(round(math.atan2(number.imag, number.real) * v02.SPF32_PHASE_MOD / _TAU))


def _object_output(source: Any, function: Any, shape: tuple[int, ...]) -> Any:
    array = np.asarray(source, dtype=object)
    if shape == ():
        return function(array.item())
    output = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        output[index] = function(array[index])
    return output


def _metadata_for_output(field: Field, operations: Sequence[FieldOperation]) -> dict[str, Any]:
    metadata = dict(field.metadata)
    metadata.update(dict(field.representation.metadata))
    for operation in operations:
        if operation.op in {"display", "display_convert"}:
            metadata.update(dict(operation.params))
    return metadata


def _format_output(
    field: Field,
    request: MaterializationRequest,
    values: Any,
    source_values: np.ndarray,
    source_kind: str,
    context: _SamplingContext,
    dense_operations: Sequence[FieldOperation],
) -> tuple[Any, float, float, v02.ErrorLedger, list[str]]:
    """Format one sampled field and return payload/error/phase metadata."""

    target = _resolve_target(request)

    semantic_demotions: list[str] = []
    metadata = _metadata_for_output(field, dense_operations)
    if target == "log_pos" and (
        field.coherence is Coherence.COHERENT_COMPLEX
        and not bool(metadata.get("explicit_intensity", metadata.get("allow_complex_to_positive", False)))
    ):
        raise MaterializationError("LOG_POS output from coherent-complex data requires explicit abs2/intensity")
    if (
        target == "display"
        and field.coherence is Coherence.COHERENT_COMPLEX
        and not bool(metadata.get("explicit_intensity", metadata.get("display_from_intensity", False)))
    ):
        raise MaterializationError("display output from coherent-complex data requires explicit abs2/intensity")

    word_source = source_values if isinstance(source_values, _WordSource) else None
    cartesian_declared = _field_kind_name(field) in {"dense", "any", ""}
    if word_source is not None and source_kind in {"words", "word", "uint32"}:
        words_out = np.asarray(values, dtype=np.uint32)
        if target == "spf32" or (target == "native" and not cartesian_declared):
            # The executed path is packed; the output is the same plane.  No
            # decode is needed to certify it against itself.
            return words_out.copy(), 0.0, 0.0, v02.ErrorLedger(), semantic_demotions
        if target in {"native", "complex"}:
            # Cartesian-declared field: the requested output boundary is the
            # one decode of the word plane (design §7.3, counted by the caller).
            return word_source.complex_reference(), 0.0, 0.0, v02.ErrorLedger(), semantic_demotions
    requested_compute = str(metadata.get("compute_dtype", "complex128"))
    cartesian_dtype = np.complex64 if requested_compute == "complex64" else np.complex128
    source_complex = (
        word_source.complex_reference() if word_source is not None else np.asarray(source_values, dtype=cartesian_dtype)
    )
    if target == "complex":
        return source_complex, 0.0, 0.0, v02.ErrorLedger(), semantic_demotions
    if target == "native":
        if source_kind in _PHASE_KINDS:
            output = _object_output(
                values,
                lambda item: (
                    item if isinstance(item, v02.Phase) else _typed_phase(complex(item), source_kind=source_kind)
                ),
                context.output_shape,
            )
            numeric = _complex_array(output, kind="phase")
        elif source_kind in {"words", "word", "uint32"}:
            output = np.asarray(values, dtype=np.uint32)
            numeric = np.asarray(decode_words(output))
        elif source_kind in {"log_pos", "log_signed"}:
            output = np.asarray(values, dtype=np.float64)
            # The two comparison helpers own the one Cartesian widening they
            # need; widening here allocated a second plane per output.
            numeric = output
        else:
            output = np.asarray(values)
            numeric = output
        return (
            output,
            _max_difference(numeric, source_complex),
            _phase_difference(source_complex, numeric),
            v02.ErrorLedger(),
            semantic_demotions,
        )

    if target == "spf32":
        if source_kind in {"words", "word", "uint32"} and not dense_operations:
            output = np.asarray(values, dtype=np.uint32).copy()
        else:
            output = np.asarray(encode_words(source_complex), dtype=np.uint32)
        numeric = np.asarray(decode_words(output))
        error = _max_difference(numeric, source_complex)
        ledger = v02.ErrorLedger(source_quantization=error)
        return output, error, _phase_difference(source_complex, numeric), ledger, semantic_demotions

    if target == "log_pos":
        if source_kind in _COMPLEX_KINDS or field.coherence is Coherence.COHERENT_COMPLEX:
            raise MaterializationError("LOG_POS output requires positive semantics or an explicit intensity operation")
        real = np.asarray(np.real(source_complex), dtype=np.float64)
        if np.any(real < -1.0e-12):
            raise MaterializationError("LOG_POS output cannot represent negative values")
        output = _object_output(real, _typed_logpos, context.output_shape)
        numeric = _complex_array(output, kind="log_pos")
        return (
            output,
            _max_difference(numeric, source_complex),
            0.0,
            v02.ErrorLedger(materialization=_max_difference(numeric, source_complex)),
            semantic_demotions,
        )

    if target == "log_signed":
        if np.any(np.abs(source_complex.imag) > 1.0e-12 * np.maximum(1.0, np.abs(source_complex.real))):
            raise MaterializationError("LOG_SIGNED output requires real-valued samples")
        real = np.asarray(source_complex.real, dtype=np.float64)
        output = _object_output(real, _typed_logsigned, context.output_shape)
        numeric = _complex_array(output, kind="log_signed")
        error = _max_difference(numeric, source_complex)
        return (
            output,
            error,
            _phase_difference(source_complex, numeric),
            v02.ErrorLedger(materialization=error),
            semantic_demotions,
        )

    if target == "phase":
        output = _object_output(
            source_complex, lambda item: _typed_phase(item, source_kind=source_kind), context.output_shape
        )
        numeric = _complex_array(output, kind="phase")
        semantic_demotions.append("complex_to_phase" if source_kind not in _PHASE_KINDS else "phase_requantization")
        error = _max_difference(numeric, source_complex)
        return (
            output,
            error,
            _phase_difference(source_complex, numeric),
            v02.ErrorLedger(materialization=error),
            semantic_demotions,
        )

    if target == "real":
        output = np.asarray(source_complex.real, dtype=np.float64)
        # The comparison helpers own the one Cartesian widening they need.
        numeric = output
        if source_kind in _COMPLEX_KINDS or field.coherence is Coherence.COHERENT_COMPLEX:
            semantic_demotions.append("complex_to_real")
        error = _max_difference(numeric, source_complex)
        return (
            output,
            error,
            _phase_difference(source_complex, numeric),
            v02.ErrorLedger(materialization=error),
            semantic_demotions,
        )

    if target == "display":
        real = np.asarray(source_complex.real, dtype=np.float64)
        if np.any(np.abs(source_complex.imag) > 1.0e-12 * np.maximum(1.0, np.abs(real))):
            if bool(metadata.get("display_from_intensity", metadata.get("explicit_intensity", False))):
                real = np.square(np.abs(source_complex))
                semantic_demotions.append("complex_to_intensity")
            else:
                raise MaterializationError("display output requires positive-real samples")
        if np.any(real < -1.0e-12):
            raise MaterializationError("display output requires non-negative values")
        black_offset = int(metadata.get("black_offset", 0))
        lut = metadata.get("lut")
        output = _object_output(
            real,
            lambda item: v02.display_code_from_log(_typed_logpos(float(item)), black_offset=black_offset, lut=lut),
            context.output_shape,
        )
        output = np.asarray(output, dtype=np.uint8)
        # Display is an intentionally lossy terminal code; the display ledger
        # records it without pretending that code values are field amplitudes.
        return output, 0.0, 0.0, v02.ErrorLedger(display=0.0), semantic_demotions

    raise MaterializationError(f"unsupported materialization output_format {request.output_format!r}")


def _enforce_error_budget(
    field: Field,
    request: MaterializationRequest,
    ledger: v02.ErrorLedger,
    source_values: np.ndarray,
    phase_error: float,
    *,
    scale: float | None = None,
) -> None:
    scale = _max_abs(source_values) if scale is None else scale
    if not request.error_budget.as_contract(phase_required=field.error.phase_required).accepts(
        ledger, scale=scale, phase_error=phase_error
    ):
        raise MaterializationError(
            f"materialization exceeds requested ErrorBudget (absolute={ledger.total_bound!r}, "
            f"relative={ledger.total_bound / max(scale, np.finfo(np.float64).tiny)!r}, phase={phase_error!r})"
        )
    if not field.error.accepts(ledger, scale=scale, phase_error=phase_error):
        raise MaterializationError(
            f"materialization exceeds field ErrorContract (absolute={ledger.total_bound!r}, phase={phase_error!r})"
        )


def _source_nbytes(source: Any) -> int:
    if isinstance(source, _WordSource):
        return source.nbytes
    return int(np.asarray(source).nbytes)


def _report(
    field: Field,
    request: MaterializationRequest,
    context: _SamplingContext,
    execution: _Execution,
    ledger: v02.ErrorLedger,
    output: Any,
    observed_error: float,
    phase_error: float,
    backend: SFRBackend,
) -> RenderReport:
    count = context.element_count
    potential_pass_elements = count * max(1, len(field.operations) + 1)
    output_array = np.asarray(output)
    cost = backend.cost_model.estimate_representation(
        field.representation, request.layout or SampleLayout(context.output_shape or (1,))
    )
    metrics = RenderMetrics(
        backend=backend.name,
        workload=_normal_name(field.semantic),
        resolution=context.output_shape,
        representation=_representation_kind(field.representation.kind).value,
        precision=_normal_name(request.precision.mode),
        error_budget=float(request.error_budget.absolute),
        bytes_read=int(field.representation.byte_size or 0),
        bytes_written=int(output_array.nbytes),
        peak_memory=int(output_array.nbytes + _source_nbytes(execution.source_values)),
        potential_elements=count,
        materialized_elements=count,
        atoms_considered=execution.atoms_considered,
        atoms_pruned=execution.atoms_pruned,
        positive_logadds=execution.positive_logadds,
        complex_adds=execution.complex_adds,
        pivot_reductions=execution.pivot_reductions,
        fallback_conversions=execution.fallback_conversions,
        field_operations=len(field.operations),
        reduction_operations=execution.reduction_operations,
        fft_count=execution.fft_count,
    )
    rewrites = tuple(execution.analytic_operations)
    passes = [
        CompilerPassRecord(
            "analytic_closure",
            len(field.operations),
            len(field.operations) - len(execution.dense_operations),
            rewrites=rewrites,
            notes=tuple(execution.notes),
        ),
        CompilerPassRecord(
            "materialize_once",
            1,
            1,
            notes=(f"{count} elements",),
        ),
    ]
    if execution.dense_operations:
        passes.append(
            CompilerPassRecord(
                "dense_suffix",
                len(execution.dense_operations),
                1,
                notes=("remaining operations applied after the single sampling boundary",),
            )
        )
    decision = RepresentationDecision(
        field_id=field.id,
        region=repr(context.region),
        selected=_representation_kind(field.representation.kind).value,
        candidates=(_representation_kind(field.representation.kind).value,),
        predicted_cost=float(cost.total),
        predicted_error=float(ledger.total_bound),
        reason="analytic closure before one dense materialization"
        if execution.analytic_operations
        else "dense/reference payload",
    )
    diagnostics = {
        "one_materialization": True,
        "analytic_operations": tuple(execution.analytic_operations),
        "dense_operations": tuple(execution.dense_operations),
        "output_format": _format_name(request.output_format),
        "output_observed_error": observed_error,
        "phase_error_bound": phase_error,
        "source_scale": _max_abs(execution.source_values),
        "source_kind": execution.source_kind,
        "materialized_elements": count,
        "potential_pass_elements": potential_pass_elements,
        "realized_pass_elements": count,
        "pass_avoidance": float(potential_pass_elements) / float(count) if count else 1.0,
        "byte_traffic_status": "logical_model",
        "peak_memory_status": "logical_live_set_model",
        "energy_status": "unmeasured",
    }
    return RenderReport(
        metrics,
        ledger=ledger,
        passes=tuple(passes),
        decisions=(decision,),
        fallbacks=tuple(execution.fallbacks),
        capabilities_used=tuple(dict.fromkeys(execution.capabilities)),
        diagnostics=diagnostics,
    )


def materialize_field(
    field: Field,
    request: MaterializationRequest | None = None,
    *,
    backend: SFRBackend | str | None = None,
    return_result: bool = False,
    return_certificate: bool = False,
    return_report: bool = False,
    report: bool = False,
) -> Any:
    """Materialize one typed field.

    By default the formatted payload is returned.  ``return_result`` or
    ``return_certificate`` returns a :class:`spf.tensor.v02.MaterializationResult`.
    ``return_report`` (or its short alias ``report``) returns
    ``(MaterializationResult, RenderReport)`` so the numeric certificate and
    the execution accounting travel together.
    """

    if not isinstance(field, Field):
        raise TypeError("materialize_field expects a renderer.model.Field")
    selected = request if request is not None else MaterializationRequest()
    if not isinstance(selected, MaterializationRequest):
        raise TypeError("materialize_field request must be MaterializationRequest")
    selected_backend: SFRBackend
    if backend is None:
        selected_backend = get_backend(selected.backend)
    elif isinstance(backend, SFRBackend):
        selected_backend = backend
    else:
        selected_backend = get_backend(str(backend))
    selected_backend.require_available()
    field, consumed_operation_indices = _consume_representation_operations(field)
    representation_kind = _representation_kind(field.representation.kind)
    if not selected_backend.capabilities.supports_representation(representation_kind):
        raise NotImplementedError(
            f"backend {selected_backend.name!r} does not support representation {representation_kind.value}"
        )

    context = _resolve_context(field, selected, field.representation.payload)
    region = _region_from_context(context)
    payload = field.representation.payload
    prepared_representation = field.representation
    execution_field = field
    effective_support = _support_after_operations(field)
    if not _support_intersects_request(effective_support, region):
        source_kind = _effective_source_kind(field, np.zeros(context.output_shape))
        zero_values = _zero_for_source(source_kind, context.output_shape)
        execution = _Execution(
            zero_values,
            source_kind,
            _WordSource(zero_values) if source_kind in {"words", "word", "uint32"} else np.zeros(context.output_shape),
            [],
            [],
            ["support_cull"],
            ["field support does not intersect requested region"],
            [],
        )
        sampled = execution.values
        pending: tuple[FieldOperation, ...] = ()
    else:
        static_error = _field_ledger_after_operations(field).total_bound
        request_budget = float(selected.error_budget.absolute)
        pruning_budget = request_budget if math.isinf(request_budget) else max(0.0, request_budget - static_error)
        normalized = normalize_field(field, region=region, budget=pruning_budget)
        execution_field = normalized.field
        prepared_representation = execution_field.representation
        payload, pending = prepared_representation.payload, execution_field.operations
        execution = _Execution(
            values=None,
            source_kind=_field_kind_name(field),
            source_values=np.empty(context.output_shape),
            analytic_operations=list(normalized.analytic_operations),
            dense_operations=[],
            capabilities=[],
            fallbacks=[],
            notes=list(normalized.notes),
            atoms_considered=normalized.atoms_considered,
            atoms_pruned=normalized.atoms_pruned,
        )
        structural = payload.__class__.__name__ in {
            "SpectralTexture",
            "TensorBlockView",
            "V3BlockView",
            "KernelBlockView",
            "FilteredBlockView",
        }
        if structural:
            execution.capabilities.append("addressed_block_consumer")
        sampled = _sample_payload(execution_field, payload, context, execution)
        if prepared_representation.residual is not None:
            sampled = sampled + _sample_residual(prepared_representation.residual, context)
        spf_typed = _field_kind_name(field) in _COMPLEX_KINDS | {"words", "word", "uint32"}
        if (
            representation_kind in {RepresentationKind.DENSE_CARTESIAN_REFERENCE, RepresentationKind.DENSE_TYPED}
            and not structural
            and not _is_word_plane(np.asarray(sampled) if not isinstance(sampled, np.ndarray) else sampled)
            and not (representation_kind is RepresentationKind.DENSE_TYPED and spf_typed)
        ):
            # A Cartesian (or typed-object) payload sampled directly is the
            # reference path; a packed word plane is the carrier and costs
            # no conversion.
            execution.fallback_conversions += 1
            execution.fallbacks.append("dense representation materialized directly")
        if (
            representation_kind is RepresentationKind.DENSE_TYPED
            and not structural
            and _cartesian_numeric(sampled)
            and spf_typed
        ):
            # DenseSPF32 ingress (design §7.3): a Cartesian payload bound to an
            # SPF-typed field becomes the packed word carrier once, here, with
            # the SPF32 quantization bound entered in the ledger.  Counted as
            # a conversion because it crosses the Cartesian/SPF boundary.
            scale = float(np.max(np.abs(sampled))) if np.size(sampled) else 0.0
            execution.ingress_quantization = (
                scale * _SPF32_RELATIVE_QUANTIZATION if math.isfinite(scale) else math.inf
            )
            execution.fallback_conversions += 1
            execution.fallbacks.append("Cartesian payload encoded to SPF32 words at ingress (DenseSPF32)")
            execution.capabilities.append("dense_spf32_ingress")
            sampled = np.asarray(encode_words(sampled), dtype=np.uint32)

    source_field = field
    if field.metadata.get("explicit_intensity") and any(
        op.op in {"abs2", "intensity", "magnitude_square"} for op in pending
    ):
        source_field = replace(field, coherence=Coherence.COHERENT_COMPLEX, value_kind=FieldKind.SPF_COMPLEX)
    native_values, source_values, source_kind = _semantic_source(source_field, sampled)
    if pending:
        native_values, source_kind = _apply_dense_operations(
            field,
            native_values,
            source_kind,
            pending,
            context,
            execution,
        )
        _, source_values, source_kind = _semantic_source(replace(field, value_kind=field.value_kind), native_values)
    execution.values = native_values
    execution.source_values = source_values
    output, observed_error, phase_error, format_ledger, demotions = _format_output(
        field,
        selected,
        native_values,
        source_values,
        source_kind,
        context,
        pending,
    )
    ledger = complete_field_ledger(execution_field)
    ledger = _ledger_add(
        ledger,
        reducer=execution.reducer_error,
        transform=execution.transform_error,
        source_quantization=execution.ingress_quantization + execution.projection_quantization,
    )
    phase_error = max(phase_error, execution.reducer_phase_error, execution.transform_phase_error)
    if field.metadata.get("explicit_intensity") and any(
        op.op in {"abs2", "intensity", "magnitude_square"} for op in pending
    ):
        from spf.renderer.native import _scale_ledger_for_observable

        amplitude_error = max(float(ledger.total_bound), float(field.metadata.get("intensity_input_error", 0)))
        if amplitude_error:
            if not ledger.total_bound:
                ledger = v02.ErrorLedger(source_quantization=amplitude_error)
            ledger = _scale_ledger_for_observable(ledger, 2 * _max_abs(sampled) + amplitude_error)
    # Mixture dropped bounds are part of the mathematical field's pruning
    # certificate; residual error belongs to representation fit.  Formatting
    # error is then added at the one output boundary.
    ledger = _ledger_add(
        ledger,
        source_quantization=format_ledger.source_quantization,
        materialization=format_ledger.materialization,
        display=format_ledger.display,
    )
    if demotions:
        execution.notes.extend(f"explicit semantic demotion: {item}" for item in demotions)
    _enforce_error_budget(field, selected, ledger, source_values, phase_error)
    result = v02.MaterializationResult(
        values=output,
        absolute_error_bound=max(observed_error, ledger.total_bound)
        if not math.isnan(ledger.total_bound)
        else math.nan,
        relative_error_bound=(
            math.inf
            if not math.isfinite(observed_error)
            else observed_error / max(_max_abs(source_values), np.finfo(np.float64).tiny)
        ),
        ledger=ledger,
    )
    render_report = _report(
        field,
        selected,
        context,
        execution,
        ledger,
        output,
        observed_error,
        phase_error,
        selected_backend,
    )
    if consumed_operation_indices:
        diagnostics = dict(render_report.diagnostics)
        diagnostics.update(
            {
                "planner_consumed_operation_indices": consumed_operation_indices,
                "planner_pending_operation_indices": tuple(
                    index
                    for index in range(len(field.operations) + len(consumed_operation_indices))
                    if index not in set(consumed_operation_indices)
                ),
            }
        )
        render_report = replace(render_report, diagnostics=diagnostics)
    if return_report or report:
        return result, render_report
    if return_result or return_certificate:
        return result
    return result.values


materialize = materialize_field


__all__ = ["MaterializationError", "materialize", "materialize_field"]
