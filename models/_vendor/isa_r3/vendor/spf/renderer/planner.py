"""Representation and materialization planning for SFR v0.2.

The planner treats representation choice as a per-region decision.  Every
candidate carries a cost, an error ledger, and a capability decision.  A dense
Cartesian reference conversion is retained as an explicit fallback, but it is
never selected merely because a native backend is unavailable: unavailable
backends fail at the boundary and a fallback conversion is visible in the
plan/report.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.renderer.backends import BackendUnavailableError, SFRBackend, get_backend
from spf.renderer.carriers import decode_words
from spf.renderer.cost import PlanCost
from spf.renderer.model import (
    Coherence,
    CoordinateChart,
    ErrorBudget,
    Exactness,
    Field,
    FieldRepresentation,
    MaterializationRequest,
    PrecisionContract,
    PrecisionMode,
    Region,
    RepresentationKind,
    SampleLayout,
)
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision
from spf.renderer.separable_terminal import (
    SeparableConversion,
    SeparableStorage,
    separable_conversion,
    separable_storage,
)
from spf.renderer.v02 import FieldKind
from spf.tensor import v02 as arithmetic


class PlanningError(RuntimeError):
    """Base planner error."""


class ErrorBudgetExceededError(PlanningError):
    """No legal representation satisfies the requested error contract."""


# Compatibility with the terminology used by the v0.2 design document.
ErrorBudgetExceeded = ErrorBudgetExceededError


class CapabilityError(PlanningError):
    """A representation or operation is not executable on the backend."""


def _kind(value: FieldKind | str) -> FieldKind:
    if isinstance(value, FieldKind):
        return value
    name = str(value).strip().lower().replace("-", "_").split(".")[-1]
    aliases = {
        "atom": FieldKind.ATOM,
        "gaussian_chirp_atom": FieldKind.ATOM,
        "mixture": FieldKind.ATOM_MIXTURE,
        "atom_mixture": FieldKind.ATOM_MIXTURE,
        "jet": FieldKind.JET2D,
        "jet_2d": FieldKind.JET2D,
        "separable": FieldKind.SEPARABLE,
        "words": FieldKind.WORDS,
        "word": FieldKind.WORDS,
        "spf": FieldKind.SPF_COMPLEX,
        "complex": FieldKind.SPF_COMPLEX,
        "positive": FieldKind.LOG_POS,
        "logpos": FieldKind.LOG_POS,
        "signed": FieldKind.LOG_SIGNED,
        "logsigned": FieldKind.LOG_SIGNED,
        "display": FieldKind.DISPLAY,
    }
    if name in aliases:
        return aliases[name]
    try:
        return FieldKind(name)
    except ValueError:
        return FieldKind[name.upper()]


def _region(value: Any, dimension: int) -> Region:
    if isinstance(value, Region):
        if value.dimension != dimension:
            raise ValueError("region dimension does not match field")
        return value
    if isinstance(value, Mapping):
        lower = value.get("lower", value.get("lo"))
        upper = value.get("upper", value.get("hi"))
        if lower is not None and upper is not None:
            return Region(tuple(lower), tuple(upper))
        if "shape" in value:
            shape = tuple(float(v) for v in np.asarray(value["shape"]).reshape(-1))
            origin = tuple(float(v) for v in np.asarray(value.get("origin", (0.0,) * len(shape))).reshape(-1))
            return Region(origin, tuple(o + max(0.0, s - 1.0) for o, s in zip(origin, shape, strict=True)))
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and array.size == dimension:
        return Region((0.0,) * dimension, tuple(max(0.0, v - 1.0) for v in array))
    if array.ndim == 1 and array.size == 2 * dimension:
        return Region(tuple(array[:dimension]), tuple(array[dimension:]))
    if array.ndim == 2 and array.shape == (dimension, 2):
        return Region(tuple(array[:, 0]), tuple(array[:, 1]))
    if array.ndim == 2 and array.shape == (2, dimension):
        return Region(tuple(array[0]), tuple(array[1]))
    raise ValueError("region must be a Region, a shape, or lower/upper bounds")


def _dimension(field: Field) -> int:
    if field.chart.dimension:
        return field.chart.dimension
    payload = field.representation.payload
    if isinstance(payload, arithmetic.GaussianChirpAtom):
        return payload.dimension
    if isinstance(payload, arithmetic.AtomMixture) and payload.atoms:
        return payload.atoms[0].dimension
    if isinstance(payload, arithmetic.Jet2D):
        return 2
    if isinstance(payload, arithmetic.SeparableField):
        return payload.dimension
    return max(1, int(np.asarray(payload).ndim))


def _layout_for(field: Field, layout: SampleLayout | None, region: Region) -> SampleLayout:
    if layout is not None:
        if layout.shape and len(layout.shape) != region.dimension:
            raise ValueError("sample layout dimension does not match region")
        return layout
    payload = field.representation.payload
    if isinstance(payload, np.ndarray) and payload.ndim:
        shape = tuple(int(max(1, n)) for n in payload.shape)
        if len(shape) == region.dimension:
            return SampleLayout(
                shape, origin=region.lower, spacing=field.chart.sample_spacing or (1.0,) * region.dimension
            )
    shape = tuple(max(1, round(hi - lo) + 1) for lo, hi in zip(region.lower, region.upper, strict=True))
    return SampleLayout(shape, origin=region.lower, spacing=(1.0,) * region.dimension)


#: Stored width of one one-dimensional separable factor value, per value kind.
_SEPARABLE_VALUE_BYTES: dict[str, int] = {"spf_complex": 4, "log_pos": 4, "log_signed": 4, "phase": 4}

#: Stored width of one Cartesian interop pair (two IEEE doubles) and one real.
#: Atom parameter blocks are declared in these units; the declaration is
#: arithmetic on the parameter count, never a materialized plane.
_INTEROP_PAIR_BYTES = 16
_INTEROP_REAL_BYTES = 8


def _parameter_bytes(value: Any, *, width: int) -> int:
    """Declared storage of a scalar-or-array atom parameter block.

    ``width`` is the stored width of one element.  The size is read from the
    array's own shape (the atoms normalize their own parameter arrays in
    ``__post_init__``); a Python scalar counts as one element.  Nothing is
    allocated, so this stays O(1) in the parameter count.
    """

    if isinstance(value, np.ndarray):
        return int(value.size) * int(width)
    array = np.asarray(value)
    return int(max(1, array.size)) * int(width)


def _payload_nbytes(value: Any, *, storage: SeparableStorage | None = None) -> int:
    if value is None:
        return 0
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, arithmetic.SeparableField):
        # Ref §17.2: a separable field stores ``R*sum(n_j)`` one-dimensional
        # values, never the ``prod(n_j)`` dense samples.  Without a declared
        # extent only the per-rank factor parameter blocks are known.
        element = _SEPARABLE_VALUE_BYTES.get(str(value.value_kind), 8)
        if storage is not None:
            return int(storage.separable_values * element)
        return int(max(1, value.rank) * max(1, value.dimension) * element)
    if isinstance(value, arithmetic.GaussianChirpAtom):
        # The atom is a fixed-size parameter block: one pair kappa, one
        # d-vector of pairs, and one d-by-d matrix of pairs.  ``getattr`` is
        # intentionally avoided here because Python scalar objects do not
        # expose a useful storage size.  The width is declared arithmetic on
        # the parameter count; no plane is formed.
        return int(
            _parameter_bytes(value.kappa, width=_INTEROP_PAIR_BYTES)
            + _parameter_bytes(value.h, width=_INTEROP_PAIR_BYTES)
            + _parameter_bytes(value.C, width=_INTEROP_PAIR_BYTES)
        )
    if isinstance(value, arithmetic.SphericalGaussian):
        return int(
            _parameter_bytes(value.mu, width=_INTEROP_REAL_BYTES)
            + _parameter_bytes(value.lambda_, width=_INTEROP_REAL_BYTES)
            + _parameter_bytes(value.amplitude, width=_INTEROP_PAIR_BYTES)
        )
    if isinstance(value, arithmetic.AtomMixture):
        return int(
            sum(_payload_nbytes(atom) for atom in value.atoms)
            + _payload_nbytes(value.residual)
            + _parameter_bytes(value.dropped_bound, width=_INTEROP_REAL_BYTES)
        )
    if isinstance(value, arithmetic.Jet2D):
        # Jet headers are serialized as widened scalar words by the native
        # boundary.  Include the six coefficients and the two guard fields;
        # the latter are fixed-width metadata, not a dense plane.
        coefficients = (value.value, value.grad_x, value.grad_y, value.hess_xx, value.hess_xy, value.hess_yy)
        return int(sum(np.asarray(item).nbytes or 8 for item in coefficients) + 2 * 4 + 4)
    # SpectralJetTile2D is imported lazily by the planner to avoid making the
    # low-level jet module part of the planner import cycle.  It contains
    # optional residual/source arrays; count those only when present so an
    # analytic Gaussian tile remains O(1) in its declared footprint.
    if value.__class__.__name__ == "SpectralJetTile2D":
        size = _payload_nbytes(value.log_magnitude) + _payload_nbytes(value.phase)
        size += int(np.asarray(value.shape, dtype=np.int64).nbytes)
        size += int(np.asarray(value.origin, dtype=np.float64).nbytes)
        from spf.renderer.residuals import residual_storage_bytes

        size += residual_storage_bytes(value.log_residual, value.phase_residual)
        size += _payload_nbytes(value.zero_mask)
        # Four finite diagnostic scalars are part of the candidate metadata.
        size += 4 * 8
        return int(size)
    if value.__class__.__name__ == "AdaptiveJetField":
        size = int(np.asarray(value.shape, dtype=np.int64).nbytes)
        size += sum(_payload_nbytes(tile) for tile in value.tiles)
        size += 16 * len(value.slices) + 8
        return int(size)
    if isinstance(value, (tuple, list)):
        return sum(_payload_nbytes(item) for item in value)
    return int(getattr(value, "nbytes", 0) or 0)


def _layout_points(layout: SampleLayout) -> np.ndarray:
    """Return points in the v0.2 row-major ``(x=column, y=row)`` order."""

    if len(layout.shape) == 2:
        height, width = layout.shape
        x_axis = layout.origin[0] + layout.spacing[0] * np.arange(width, dtype=np.float64)
        y_axis = layout.origin[1] + layout.spacing[1] * np.arange(height, dtype=np.float64)
        y_grid, x_grid = np.meshgrid(y_axis, x_axis, indexing="ij")
        return np.asarray(np.stack((x_grid, y_grid), axis=-1).reshape(-1, 2), dtype=np.float64)
    return layout.points()


def _typed_scalar_word(value: Any) -> int:
    """Pack one v0.2 typed scalar into an SPF32 word through the plane types."""

    from spf.tensor import v02_planes as planes

    if isinstance(value, arithmetic.LogPos):
        return int(planes.LogPosPlane(np.asarray([value.L])).to_words()[0])
    if isinstance(value, arithmetic.LogSigned):
        return int(planes.LogSignedPlane(np.asarray([value.L]), np.asarray([value.sign])).to_words()[0])
    if isinstance(value, arithmetic.Phase):
        return int(planes.PhasePlane(np.asarray([value.T])).to_words()[0])
    raise TypeError(f"not a v0.2 typed scalar: {type(value).__name__}")


def _word_native_source(field: Field, layout: SampleLayout) -> np.ndarray | None:
    """Packed SPF32 words for the source when the payload already carries them.

    §7.3 candidate carrier: an existing packed plane is shaped, never decoded,
    and a typed v0.2 scalar is packed through its plane type
    (:mod:`spf.tensor.v02_planes`).  A family whose evaluator is Cartesian (an
    analytic atom, a mixture, a separable field, an interop array) returns
    ``None`` on purpose — encoding it here would quantize the very samples a
    candidate fit needs unquantized, so those callers use
    :func:`_source_values`, the REFERENCE_CARTESIAN producer, instead.
    """

    payload = field.representation.payload
    if isinstance(payload, np.ndarray) and payload.dtype.kind == "u" and payload.dtype.itemsize == 4:
        try:
            return np.ascontiguousarray(np.broadcast_to(payload, layout.shape), dtype=np.uint32)
        except ValueError:
            if payload.size == layout.element_count:
                return np.asarray(payload.reshape(layout.shape), dtype=np.uint32)
            return None
    if isinstance(payload, arithmetic.SPF32Value):
        return np.full(layout.shape, np.uint32(payload.word), dtype=np.uint32)
    if isinstance(payload, (arithmetic.LogPos, arithmetic.LogSigned, arithmetic.Phase)):
        return np.full(layout.shape, np.uint32(_typed_scalar_word(payload)), dtype=np.uint32)
    return None


def _word_plane_scale(words: np.ndarray) -> float:
    """Magnitude envelope of a word plane from its L plane through the ROM.

    No decode: ``LogPosPlane.max_code`` picks the largest finite log code and
    the shared magnitude ROM turns it into ``2**(L/256)`` — the same number a
    decode-then-``max(abs)`` produced, because that *is* a word's magnitude.
    """

    from spf.renderer.carriers import magnitude_table
    from spf.tensor import v02_planes as planes

    if not words.size:
        return 1.0
    plane = planes.LogPosPlane.from_words(words)
    if bool(np.any(plane.is_nan)):
        return 1.0
    code = plane.max_code()
    if code == arithmetic.SPF32_L_ZERO:
        return 0.0
    maximum = float(magnitude_table()[code & 0xFFFF])
    return maximum if math.isfinite(maximum) else 1.0


def _source_values(field: Field, layout: SampleLayout, region: Region) -> np.ndarray | None:
    """REFERENCE_CARTESIAN source view: unquantized samples for candidate fitting.

    AGENTS.md rule 2: this is the fallback/measurement producer.  Candidate
    formation that only needs the word carrier calls :func:`_source_words`, and
    anything that only needs a magnitude envelope calls
    :func:`_word_plane_scale`; neither decodes a packed plane.
    """

    payload = field.representation.payload
    if payload.__class__.__name__ in {
        "SpectralTexture",
        "TensorBlockView",
        "V3BlockView",
        "KernelBlockView",
        "FilteredBlockView",
    }:
        # Existing blocks already own fitted/packed state; preserve it for
        # the addressed consumer instead of reconstructing it to refit.
        return None
    points = _layout_points(layout)
    if isinstance(payload, arithmetic.GaussianChirpAtom):
        return np.asarray(payload.evaluate(points), dtype=np.complex128).reshape(layout.shape)
    if isinstance(payload, arithmetic.AtomMixture):
        if payload.residual is not None and hasattr(payload.residual, "to_dense"):
            atom_values = np.zeros(layout.element_count, dtype=np.complex128)
            for atom in payload.atoms:
                atom_values += np.asarray(atom.evaluate(points), dtype=np.complex128).reshape(-1)
            residual_values = np.asarray(payload.residual.to_dense(), dtype=np.complex128)
            if residual_values.shape != layout.shape:
                return None
            return atom_values.reshape(layout.shape) + residual_values
        return np.asarray(payload.evaluate(points), dtype=np.complex128).reshape(layout.shape)
    if isinstance(payload, arithmetic.SeparableField):
        values = [arithmetic.evaluate_separable(payload, point).value for point in points]
        return np.asarray([_numeric_value(value) for value in values], dtype=np.complex128).reshape(layout.shape)
    if isinstance(payload, arithmetic.Jet2D):
        if layout.element_count == 0:
            return np.empty(layout.shape, dtype=np.complex128)
        values = arithmetic.jet_materialize(
            payload,
            shape=(layout.shape[0], layout.shape[1]),
            origin=(layout.origin[0], layout.origin[1]),
        )
        return np.asarray(_decode_typed(values), dtype=np.complex128)
    if hasattr(payload, "materialize") and payload.__class__.__name__ == "SpectralJetTile2D":
        return np.asarray(payload.materialize(), dtype=np.complex128)
    if payload.__class__.__name__ == "SpectralTexture" and hasattr(payload, "reconstruct"):
        source = np.asarray(payload.reconstruct())
        try:
            return np.broadcast_to(source, layout.shape).astype(np.complex128, copy=False)
        except ValueError:
            return source.reshape(layout.shape).astype(np.complex128) if source.size == layout.element_count else None
    if isinstance(payload, np.ndarray):
        array = np.asarray(payload)
        if array.dtype.kind == "u" and array.dtype.itemsize == 4:
            return np.asarray(decode_words(array), dtype=np.complex128)
        try:
            return np.broadcast_to(array, layout.shape).astype(np.complex128, copy=False)
        except ValueError:
            if array.size == layout.element_count:
                return array.reshape(layout.shape).astype(np.complex128)
            return None
    if isinstance(payload, (arithmetic.LogPos, arithmetic.LogSigned, arithmetic.Phase)):
        return np.asarray([_numeric_value(payload)] * layout.element_count, dtype=np.complex128).reshape(layout.shape)
    if np.isscalar(payload):
        return np.full(layout.shape, _numeric_value(payload), dtype=np.complex128)
    return None


def _numeric_value(value: Any) -> complex | float:
    if isinstance(value, arithmetic.LogPos):
        return value.value
    if isinstance(value, arithmetic.LogSigned):
        return value.value
    if isinstance(value, arithmetic.Phase):
        radians = int(value.T) * 2.0 * math.pi / arithmetic.SPF32_PHASE_MOD
        return complex(math.cos(radians), math.sin(radians))
    if isinstance(value, arithmetic.SPF32Value):
        return complex(decode_words(value.word))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return complex(decode_words(int(value)))
    return cast(complex | float, value)


def _dense_executable(payload: Any) -> bool:
    """True when the runtime's dense sampler can execute ``payload`` as a plane.

    A coefficient/pair mapping (Hermitian, sparse) is executable only under
    its own representation kind; labelling it dense would sample the mapping
    object itself.
    """

    if isinstance(payload, Mapping):
        return any(key in payload for key in ("values", "value", "data", "payload"))
    return True


def _is_cartesian_array(payload: Any) -> bool:
    return isinstance(payload, np.ndarray) and payload.dtype.kind in "fc"


def _spf_typed_field(field: Field) -> bool:
    kind = field.value_kind if isinstance(field.value_kind, FieldKind) else FieldKind(str(field.value_kind).lower())
    return kind in {FieldKind.SPF_COMPLEX, FieldKind.WORDS}


def _cartesian_import(field: Field, payload: Any) -> bool:
    """True when the field itself was bound to a Cartesian array (an interop import)."""

    if not _is_cartesian_array(payload):
        return False
    kind = field.value_kind if isinstance(field.value_kind, FieldKind) else FieldKind(str(field.value_kind).lower())
    return kind not in {FieldKind.SPF_COMPLEX, FieldKind.WORDS} and field.representation.kind in {
        RepresentationKind.DENSE_CARTESIAN_REFERENCE,
        RepresentationKind.DENSE_TYPED,
    }


_SPF32_RELATIVE_QUANTIZATION = (2.0 ** (1.0 / 512.0) - 1.0) + math.pi / arithmetic.SPF32_PHASE_MOD


def _dense_spf32_payload(payload: np.ndarray) -> tuple[np.ndarray, float]:
    """Encode a Cartesian payload to packed SPF32 words with its quantization bound."""

    from spf.renderer.carriers import encode_words

    scale = float(np.max(np.abs(payload))) if payload.size else 0.0
    bound = scale * _SPF32_RELATIVE_QUANTIZATION if math.isfinite(scale) else math.inf
    return np.asarray(encode_words(payload), dtype=np.uint32), bound


def _decode_typed(value: Any) -> Any:
    array = np.asarray(value)
    if array.dtype.kind == "u" and array.dtype.itemsize == 4:
        return decode_words(array)
    if array.dtype == object:
        out = np.empty(array.shape, dtype=np.complex128)
        for index in np.ndindex(array.shape):
            out[index] = _numeric_value(array[index])
        return out
    return value


def _fit_jet(
    payload: Any,
    layout: SampleLayout,
    *,
    precision: PrecisionContract | None = None,
) -> tuple[Any, float, float]:
    from spf.renderer.jets import SpectralJetTile2D

    if layout.element_count == 0 or len(layout.shape) != 2:
        raise ValueError("quadratic jet candidates require a non-empty 2-D layout")
    if isinstance(payload, arithmetic.GaussianChirpAtom):
        selected_precision = precision or PrecisionContract()
        return _analytic_atom_jet(
            payload,
            layout,
            guard_L=selected_precision.log_guard_bits,
            guard_T=selected_precision.phase_guard_bits,
        )
    source = _source_values_from_payload(payload, layout)
    if source is None:
        raise ValueError("cannot fit a jet without numeric samples")
    tile = SpectralJetTile2D.fit(source, origin=(layout.origin[0], layout.origin[1]))
    reconstructed = tile.materialize()
    error = float(np.max(np.abs(reconstructed - source))) if source.size else 0.0
    phase_error = _max_phase_error(reconstructed, source)
    return tile, error, phase_error


def _source_values_from_payload(payload: Any, layout: SampleLayout) -> np.ndarray | None:
    temp = Field.from_value(payload, id="planner_source", chart=CoordinateChart.identity(len(layout.shape)))
    return _source_values(
        temp,
        layout,
        Region(
            layout.origin,
            tuple(
                origin + max(0, size - 1) * spacing
                for origin, size, spacing in zip(layout.origin, layout.shape, layout.spacing, strict=True)
            ),
        ),
    )


def _max_phase_error(left: Any, right: Any) -> float:
    """Largest wrapped phase discrepancy between two already-Cartesian planes.

    Both call sites hand over planes that are already in the comparison domain
    (a reference source view and a decoded materialization), so this reads them
    as they are instead of widening a second copy of each.
    """

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if not left_array.size:
        return 0.0
    mask = (np.abs(left_array) > np.finfo(np.float64).tiny) & (np.abs(right_array) > np.finfo(np.float64).tiny)
    if not np.any(mask):
        return 0.0
    return float(np.max(np.abs(np.angle(left_array[mask] * np.conj(right_array[mask])))))


def _quadratic_box_range(coefficients: Sequence[float], shape: tuple[int, int]) -> tuple[float, float]:
    """Return the exact range of a 2-D quadratic on a pixel rectangle.

    The rectangle is expressed in local pixel coordinates ``0 <= x < width``
    and ``0 <= y < height``.  A quadratic can attain an extremum only at a
    corner, on an edge stationary point, or at an interior stationary point;
    all of those are a constant-size set.  This keeps the atom planner free
    of a grid allocation while still allowing it to reject SPF32 exponent
    saturation conservatively.
    """

    if len(coefficients) != 6:
        raise ValueError("quadratic range requires six coefficients")
    c, gx, gy, hxx, hxy, hyy = (float(value) for value in coefficients)
    xmax = float(max(0, int(shape[1]) - 1))
    ymax = float(max(0, int(shape[0]) - 1))

    def evaluate(x: float, y: float) -> float:
        return c + gx * x + gy * y + 0.5 * hxx * x * x + hxy * x * y + 0.5 * hyy * y * y

    points: list[tuple[float, float]] = [(0.0, 0.0), (xmax, 0.0), (0.0, ymax), (xmax, ymax)]
    tolerance = 1.0e-14 * max(1.0, abs(gx), abs(gy), abs(hxx), abs(hxy), abs(hyy))

    def add_if_inside(x: float, y: float) -> None:
        if -tolerance <= x <= xmax + tolerance and -tolerance <= y <= ymax + tolerance:
            points.append((min(xmax, max(0.0, x)), min(ymax, max(0.0, y))))

    # Interior stationary point, including a rank-deficient semidefinite
    # Hessian when its linear term lies in the Hessian range.
    hessian = np.asarray(((hxx, hxy), (hxy, hyy)), dtype=np.float64)
    gradient = np.asarray((gx, gy), dtype=np.float64)
    if float(np.linalg.norm(hessian, ord=np.inf)) > tolerance:
        try:
            stationary = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            stationary = -np.linalg.pinv(hessian) @ gradient
            if float(np.linalg.norm(hessian @ stationary + gradient, ord=np.inf)) > 32.0 * tolerance:
                stationary = None
        if stationary is not None:
            add_if_inside(float(stationary[0]), float(stationary[1]))

    # Restrict to each edge and add its (at most one) quadratic stationary
    # point.  Endpoints are already present in ``points``.
    def add_edge_root(linear: float, quadratic: float, *, axis: str, fixed: float) -> None:
        if abs(quadratic) <= tolerance:
            return
        root = -linear / quadratic
        if axis == "x":
            add_if_inside(root, fixed)
        else:
            add_if_inside(fixed, root)

    for fixed_x in (0.0, xmax):
        add_edge_root(gy + hxy * fixed_x, hyy, axis="y", fixed=fixed_x)
    for fixed_y in (0.0, ymax):
        add_edge_root(gx + hxy * fixed_y, hxx, axis="x", fixed=fixed_y)

    values = [evaluate(x, y) for x, y in points]
    return float(min(values)), float(max(values))


def _coefficient_rounding_bound(exact: np.ndarray, shape: tuple[int, int], guard_bits: int) -> tuple[np.ndarray, float]:
    """Round six analytic headers and bound the resulting stored code error."""

    rounded = np.rint(np.asarray(exact, dtype=np.float64)).astype(np.int64)
    height, width = (int(value) for value in shape)
    x = float(max(0, width - 1))
    y = float(max(0, height - 1))
    coefficient_error = np.abs(np.asarray(exact, dtype=np.float64) - rounded)
    bound = (
        coefficient_error[0]
        + coefficient_error[1] * x
        + coefficient_error[2] * y
        + 0.5 * coefficient_error[3] * x * x
        + coefficient_error[4] * x * y
        + 0.5 * coefficient_error[5] * y * y
    ) / float(1 << int(guard_bits))
    # The final storage RNE contributes half a code.  A tiny floating-point
    # cushion covers the recurrence's widened float64 carrier without turning
    # the bound into a grid-sized computation.
    bound += 0.5 + np.finfo(np.float64).eps * max(1.0, float(np.sum(np.abs(exact))))
    return rounded, float(bound)


def _atom_layout_exponents(
    atom: arithmetic.GaussianChirpAtom, layout: SampleLayout
) -> tuple[np.ndarray, np.ndarray]:
    """Six log-magnitude and six phase exponent coefficients of ``atom`` on ``layout``.

    ``GaussianChirpAtom`` is exactly quadratic in both domains, so transforming
    it by the layout spacing/origin (:func:`spf.tensor.v02.atom_affine`, the one
    §17.3 implementation) yields the complete local polynomial in O(1) — no
    samples and no ``evaluate`` call.  The atom's own parameter arrays are the
    v0.2 structured carrier; only their real/imaginary parts are read here, and
    the two returned header vectors are real float64.  Shared by the analytic
    jet candidate and the analytic field-scale bound so the parameter unpacking
    exists once.
    """

    transformed = arithmetic.atom_affine(
        atom,
        np.diag(np.asarray(layout.spacing, dtype=np.complex128)),
        np.asarray(layout.origin, dtype=np.complex128),
    )
    h = np.asarray(transformed.h)
    C = np.asarray(transformed.C)
    real_exponent = np.asarray(
        (
            np.real(transformed.kappa),
            np.real(h[0]),
            np.real(h[1]),
            -np.real(C[0, 0]),
            -np.real(C[0, 1]),
            -np.real(C[1, 1]),
        ),
        dtype=np.float64,
    )
    phase_exponent = np.asarray(
        (
            np.imag(transformed.kappa),
            np.imag(h[0]),
            np.imag(h[1]),
            -np.imag(C[0, 0]),
            -np.imag(C[0, 1]),
            -np.imag(C[1, 1]),
        ),
        dtype=np.float64,
    )
    return real_exponent, phase_exponent


def _analytic_atom_jet(
    atom: arithmetic.GaussianChirpAtom,
    layout: SampleLayout,
    *,
    guard_L: int,
    guard_T: int,
) -> tuple[Any, float, float]:
    """Construct a paired Jet2 tile from Gaussian parameters in O(1).

    ``GaussianChirpAtom`` is exactly quadratic in the log-magnitude and phase
    domains.  Transforming the atom by the layout spacing/origin therefore
    gives the complete local polynomial directly; no samples, least-squares
    solve, residual stream, or ``GaussianChirpAtom.evaluate`` call is needed.
    """

    if atom.dimension != 2 or len(layout.shape) != 2 or layout.element_count == 0:
        raise ValueError("analytic Gaussian jet candidates require a non-empty 2-D layout")
    from spf.renderer.jets import JetFitDiagnostics, SpectralJetTile2D

    real_exponent, phase_exponent = _atom_layout_exponents(atom, layout)
    log_scale = arithmetic.SPF32_L_SCALE * (1 << int(guard_L)) / math.log(2.0)
    phase_scale = arithmetic.SPF32_PHASE_MOD * (1 << int(guard_T)) / (2.0 * math.pi)
    exact_log = real_exponent * log_scale
    exact_phase = phase_exponent * phase_scale
    log_rounded, log_code_error = _coefficient_rounding_bound(exact_log, layout.shape, guard_L)
    phase_rounded, phase_code_error = _coefficient_rounding_bound(exact_phase, layout.shape, guard_T)

    # A finite Jet2 SPF32 plane cannot represent an exponent outside the
    # finite log-code interval.  Refuse the candidate rather than silently
    # converting a saturated Gaussian into a different field.
    real_min, real_max = _quadratic_box_range(real_exponent.tolist(), layout.shape)
    # ``exact_log`` is in widened guard units; the packed SPF32 plane sees
    # one storage code per ``2**guard_L`` widened units.
    exact_log_min = real_min * log_scale / float(1 << int(guard_L))
    exact_log_max = real_max * log_scale / float(1 << int(guard_L))
    if exact_log_min < arithmetic.SPF32_L_MIN_FINITE or exact_log_max > arithmetic.SPF32_L_MAX_FINITE:
        raise ValueError("analytic Gaussian jet log range exceeds finite SPF32 codes")

    phase_error = phase_code_error * (2.0 * math.pi / arithmetic.SPF32_PHASE_MOD)
    max_magnitude = math.exp(real_max)
    relative_magnitude_error = math.expm1(log_code_error * math.log(2.0) / arithmetic.SPF32_L_SCALE)
    complex_error = max_magnitude * (relative_magnitude_error + (1.0 + relative_magnitude_error) * phase_error)
    complex_error += np.finfo(np.float64).eps * max(1.0, max_magnitude) * 8.0
    diagnostics = JetFitDiagnostics(
        log_code_error / arithmetic.SPF32_L_SCALE,
        phase_error,
        float(complex_error),
        0.0,
    )
    # Coefficients are in local integer-index coordinates after the affine
    # transform.  The tile origin is consequently zero even when the physical
    # layout has a non-zero origin; this preserves non-unit spacing exactly.
    tile = SpectralJetTile2D(
        arithmetic.Jet2D(
            value=int(log_rounded[0]),
            grad_x=int(log_rounded[1]),
            grad_y=int(log_rounded[2]),
            hess_xx=int(log_rounded[3]),
            hess_xy=int(log_rounded[4]),
            hess_yy=int(log_rounded[5]),
            guard_L=int(guard_L),
            value_kind="log_pos",
        ),
        arithmetic.Jet2D(
            value=int(phase_rounded[0]),
            grad_x=int(phase_rounded[1]),
            grad_y=int(phase_rounded[2]),
            hess_xx=int(phase_rounded[3]),
            hess_xy=int(phase_rounded[4]),
            hess_yy=int(phase_rounded[5]),
            guard_T=int(guard_T),
            value_kind="phase",
        ),
        layout.shape,
        origin=(0.0, 0.0),
        diagnostics=diagnostics,
    )
    return tile, float(complex_error), float(phase_error)


def _error_contract_accepts(
    field: Field, ledger: arithmetic.ErrorLedger, *, phase_error: float, scale: float, budget: ErrorBudget
) -> bool:
    if not field.error.accepts(ledger, scale=scale, phase_error=phase_error):
        return False
    contract = budget.as_contract(phase_required=field.error.phase_required)
    return contract.accepts(ledger, scale=scale, phase_error=phase_error)


@dataclass(frozen=True, slots=True)
class CapabilityDecision:
    backend: str
    requested: str
    supported: bool
    available: bool
    native: bool
    reason: str

    @property
    def legal(self) -> bool:
        return self.supported and self.available


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    field_id: str
    region: Region
    kind: RepresentationKind
    representation: FieldRepresentation
    cost: PlanCost
    predicted_error: float
    phase_error: float
    legal: bool
    reason: str
    capability: CapabilityDecision
    fallback: bool = False

    @property
    def predicted_cost(self) -> float:
        return self.cost.total

    @property
    def error(self) -> float:
        return self.predicted_error


@dataclass(frozen=True, slots=True)
class MaterializationCertificate:
    field_id: str
    region: Region
    representation: RepresentationKind
    output_kind: FieldKind
    error_bound: float
    phase_error: float
    budget: ErrorBudget
    accepted: bool
    fallback: bool
    backend: str
    materialized_elements: int
    potential_elements: int
    bytes_read: int
    bytes_written: int
    reason: str

    @property
    def materialization_avoidance(self) -> float:
        if self.materialized_elements == 0:
            return math.inf if self.potential_elements else 1.0
        return self.potential_elements / self.materialized_elements

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_id": self.field_id,
            "region": {"lower": self.region.lower, "upper": self.region.upper},
            "representation": self.representation.value,
            "output_kind": self.output_kind.value,
            "error_bound": self.error_bound,
            "phase_error": self.phase_error,
            "accepted": self.accepted,
            "fallback": self.fallback,
            "backend": self.backend,
            "materialized_elements": self.materialized_elements,
            "potential_elements": self.potential_elements,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class MaterializedField:
    values: Any
    certificate: MaterializationCertificate

    @property
    def value(self) -> Any:
        return self.values


@dataclass(frozen=True, slots=True)
class PlannerOptions:
    backend: str | SFRBackend = "reference"
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    allow_fallback: bool = True
    # Design v0.2 §7.3 / §14.2 (amended 2026-09-17): the dense fallback is
    # DenseSPF32 (packed words), never a Cartesian complex buffer.  The
    # Cartesian reference candidate is still enumerated so that a field that
    # was *imported* as Cartesian (``Field.from_value(complex_array)``) has a
    # legal representation; it is never legal for a structured or SPF-typed
    # source outside the reference backend (see ``_candidate``).
    fallback_kind: RepresentationKind = RepresentationKind.DENSE_TYPED
    candidate_kinds: tuple[RepresentationKind, ...] = (
        RepresentationKind.GAUSSIAN_CHIRP_ATOM,
        RepresentationKind.SPHERICAL_GAUSSIAN_ATOM,
        RepresentationKind.ATOM_MIXTURE,
        RepresentationKind.QUADRATIC_JET,
        RepresentationKind.SEPARABLE,
        RepresentationKind.SPARSE_RESIDUAL,
        RepresentationKind.HERMITIAN_HALF,
        RepresentationKind.DENSE_TYPED,
        RepresentationKind.DENSE_CARTESIAN_REFERENCE,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.precision, PrecisionContract):
            object.__setattr__(self, "precision", PrecisionContract.for_mode(self.precision))
        if not isinstance(self.error_budget, ErrorBudget):
            object.__setattr__(self, "error_budget", ErrorBudget(absolute=float(self.error_budget)))
        object.__setattr__(self, "fallback_kind", _representation(self.fallback_kind))
        object.__setattr__(self, "candidate_kinds", tuple(_representation(value) for value in self.candidate_kinds))


def _representation(value: RepresentationKind | str) -> RepresentationKind:
    if isinstance(value, RepresentationKind):
        return value
    return RepresentationKind(str(value).strip().upper().replace("-", "_").split(".")[-1])


@dataclass(frozen=True, slots=True)
class MaterializationPlan:
    backend: str
    regions: tuple[Region, ...]
    candidates: tuple[RepresentationCandidate, ...]
    decisions: tuple[RepresentationDecision, ...]
    certificates: tuple[MaterializationCertificate, ...] = ()
    fallbacks: tuple[str, ...] = ()
    peak_memory: int = 0
    liveness: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    pass_records: tuple[CompilerPassRecord, ...] = ()
    report: RenderReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "decisions", tuple(self.decisions))
        object.__setattr__(self, "certificates", tuple(self.certificates))
        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))
        object.__setattr__(self, "pass_records", tuple(self.pass_records))
        object.__setattr__(self, "liveness", MappingProxyType(dict(self.liveness)))

    @property
    def selected(self) -> tuple[RepresentationCandidate, ...]:
        selected = {decision.field_id: decision.selected for decision in self.decisions}
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.legal and candidate.kind.value == selected.get(candidate.field_id)
        )

    def candidate_table(self, field_id: str | None = None) -> tuple[RepresentationCandidate, ...]:
        return tuple(item for item in self.candidates if field_id is None or item.field_id == field_id)

    def materialize_from_program(self, program: Any, field_id: str, request: Any = None, **options: Any) -> Any:
        environment = program.execute({}, return_environment=True)
        if field_id not in environment:
            raise PlanningError(f"compiled program did not produce field {field_id!r}")
        value = environment[field_id]
        selected = next((item for item in self.selected if item.field_id == field_id), None)
        if selected is None:
            return value
        field = Field.from_value(value, id=field_id, value_kind=FieldKind.DENSE)
        return _materialize_payload(field, selected, request, options)

    def materialize(self, field: Field, request: MaterializationRequest | None = None, **options: Any) -> Any:
        planner = MaterializationPlanner(backend=self.backend)
        return planner.materialize(field, request, plan=self, **options)


class MaterializationPlanner:
    """Enumerate and choose representation candidates per field and tile."""

    def __init__(
        self,
        *,
        backend: str | SFRBackend = "reference",
        precision: PrecisionContract | str | None = None,
        error_budget: ErrorBudget | float | None = None,
        allow_fallback: bool = True,
        candidate_kinds: Sequence[RepresentationKind | str] | None = None,
    ) -> None:
        self.backend = get_backend(backend)
        self.precision = (
            PrecisionContract()
            if precision is None
            else (precision if isinstance(precision, PrecisionContract) else PrecisionContract.for_mode(precision))
        )
        self.error_budget = (
            ErrorBudget()
            if error_budget is None
            else (error_budget if isinstance(error_budget, ErrorBudget) else ErrorBudget(absolute=float(error_budget)))
        )
        self.allow_fallback = bool(allow_fallback)
        self.candidate_kinds = (
            tuple(candidate_kinds) if candidate_kinds is not None else PlannerOptions().candidate_kinds
        )

    def _require_backend(self) -> None:
        try:
            self.backend.require_available()
        except BackendUnavailableError:
            raise

    def capability(self, kind: RepresentationKind | str) -> CapabilityDecision:
        selected = _representation(kind)
        supported = self.backend.capabilities.supports_representation(selected)
        reason = (
            "supported"
            if supported and self.backend.capabilities.available
            else (self.backend.capabilities.reason_unavailable or "representation unsupported")
        )
        return CapabilityDecision(
            self.backend.name,
            selected.value,
            supported,
            self.backend.capabilities.available,
            self.backend.capabilities.native,
            reason,
        )

    def enumerate_candidates(
        self,
        field: Field,
        *,
        region: Region | None = None,
        layout: SampleLayout | None = None,
        kinds: Sequence[RepresentationKind | str] | None = None,
    ) -> tuple[RepresentationCandidate, ...]:
        self._require_backend()
        if not isinstance(field, Field):
            raise TypeError("enumerate_candidates expects Field")
        dimension = _dimension(field)
        selected_region = region or field.support.region or Region((0.0,) * dimension, (1.0,) * dimension)
        selected_layout = _layout_for(field, layout, selected_region)
        selected_kinds = tuple(_representation(kind) for kind in (kinds or self.candidate_kinds))
        if field.representation.payload.__class__.__name__ in {
            "SpectralTexture",
            "TensorBlockView",
            "V3BlockView",
            "KernelBlockView",
            "FilteredBlockView",
        }:
            selected_kinds = (RepresentationKind.DENSE_TYPED,)
        from spf.renderer.normalization import normalize_field

        normalization = normalize_field(field, region=selected_region, budget=self.error_budget.absolute)
        prepared_field = normalization.field
        prepared_payload = prepared_field.representation.payload
        if isinstance(prepared_payload, arithmetic.AtomMixture) and kinds is None:
            # Fitting every raster point just to enumerate a cheaper candidate
            # defeats symbolic mixtures before the selected consumer runs.
            selected_kinds = tuple(
                k for k in selected_kinds if k in {RepresentationKind.ATOM_MIXTURE, RepresentationKind.SEPARABLE}
            )
        candidates: list[RepresentationCandidate] = []
        for kind in selected_kinds:
            candidates.append(
                self._candidate(
                    field,
                    selected_region,
                    selected_layout,
                    kind,
                    payload=prepared_payload,
                    consumed_operation_indices=normalization.consumed_indices,
                    pending_operations=prepared_field.operations,
                    prepared_error=prepared_field.representation.error,
                    prepared_residual=prepared_field.representation.residual,
                    preparation_notes=normalization.notes,
                )
            )
        return tuple(candidates)

    def _candidate(
        self,
        field: Field,
        region: Region,
        layout: SampleLayout,
        kind: RepresentationKind,
        *,
        payload: Any | None = None,
        consumed_operation_indices: Sequence[int] = (),
        pending_operations: Sequence[Any] | None = None,
        prepared_error: arithmetic.ErrorLedger | None = None,
        prepared_residual: Any = None,
        preparation_notes: Sequence[str] = (),
    ) -> RepresentationCandidate:
        capability = self.capability(kind)
        if not capability.supported:
            representation = FieldRepresentation(
                kind,
                field.representation.payload if payload is None else payload,
                error=field.ledger if prepared_error is None else prepared_error,
                byte_size=_payload_nbytes(field.representation.payload if payload is None else payload),
                consumed_operation_indices=tuple(consumed_operation_indices),
            )
            return RepresentationCandidate(
                field.id,
                region,
                kind,
                representation,
                PlanCost(error=math.inf, total=math.inf),
                math.inf,
                math.inf,
                False,
                "backend lacks representation",
                capability,
            )
        payload = field.representation.payload if payload is None else payload
        ledger = field.representation.error if prepared_error is None else prepared_error
        if pending_operations is None:
            pending_operations = field.operations
        phase_error = 0.0
        reason = "native representation"
        if preparation_notes:
            reason = f"{reason}; {'; '.join(str(note) for note in preparation_notes)}"
        fallback = False
        converted: Any
        try:
            if kind is RepresentationKind.GAUSSIAN_CHIRP_ATOM:
                if not isinstance(payload, arithmetic.GaussianChirpAtom):
                    raise ValueError("source is not a Gaussian-chirp atom")
                representation = FieldRepresentation(
                    kind,
                    payload,
                    exactness=Exactness.EXACT,
                    error=ledger,
                    evaluation_cost=1.0,
                    transform_cost=1.0,
                    byte_size=_payload_nbytes(payload),
                )
            elif kind is RepresentationKind.SPHERICAL_GAUSSIAN_ATOM:
                if not isinstance(payload, arithmetic.SphericalGaussian):
                    raise ValueError("source is not a spherical Gaussian")
                representation = FieldRepresentation(
                    kind,
                    payload,
                    exactness=Exactness.EXACT,
                    error=ledger,
                    evaluation_cost=1.0,
                    byte_size=_payload_nbytes(payload),
                )
            elif kind is RepresentationKind.ATOM_MIXTURE:
                if isinstance(payload, arithmetic.GaussianChirpAtom):
                    converted = arithmetic.AtomMixture((payload,))
                    reason = "one-atom symbolic mixture"
                elif isinstance(payload, arithmetic.AtomMixture):
                    converted = payload
                    reason = "existing symbolic mixture"
                else:
                    raise ValueError("source is not an atom or mixture")
                representation = FieldRepresentation(
                    kind,
                    converted,
                    exactness=Exactness.EXACT,
                    error=ledger,
                    evaluation_cost=max(1.0, len(converted.atoms)),
                    byte_size=_payload_nbytes(converted),
                )
            elif kind is RepresentationKind.QUADRATIC_JET:
                if isinstance(payload, arithmetic.Jet2D):
                    converted = payload
                    fit_error = 0.0
                else:
                    fit_payload = payload
                    if isinstance(payload, arithmetic.GaussianChirpAtom) and field.chart.dimension == payload.dimension:
                        chart = field.chart
                        matrix = np.asarray(chart.basis) @ np.diag(
                            np.asarray(chart.scale) * np.asarray(chart.sample_spacing)
                        )
                        fit_payload = arithmetic.atom_affine(payload, matrix, np.asarray(chart.origin))
                    converted, fit_error, phase_error = _fit_jet(
                        fit_payload,
                        layout,
                        precision=self.precision,
                    )
                ledger = ledger.plus(arithmetic.ErrorLedger(representation_fit=fit_error))
                if isinstance(payload, arithmetic.GaussianChirpAtom):
                    reason = "analytic Gaussian jet (header quantization bound)"
                else:
                    reason = "analytic jet" if fit_error == 0.0 else "certified tile fit"
                representation = FieldRepresentation(
                    kind,
                    converted,
                    exactness=Exactness.BOUNDED,
                    error=ledger,
                    evaluation_cost=0.25,
                    materialization_cost=0.15,
                    byte_size=_payload_nbytes(converted),
                    metadata={"chart_baked": isinstance(payload, arithmetic.GaussianChirpAtom)},
                )
            elif kind is RepresentationKind.SEPARABLE:
                conversion = _separable_conversion(payload, region)
                if conversion is None:
                    raise ValueError("source has no certified separable form")
                converted = conversion.field
                # §11.3: an approximate merge has to carry its bound, so only a
                # zero-bound conversion is EXACT.  The dropped off-diagonal
                # chirp, an unrepresentable residual, and the mixture's
                # dropped_bound all land in representation_fit.
                if conversion.exact:
                    exactness = Exactness.EXACT
                    reason = "exact separable factors"
                else:
                    exactness = Exactness.BOUNDED
                    ledger = ledger.plus(arithmetic.ErrorLedger(representation_fit=conversion.bound))
                    reason = f"certified separable factors ({conversion.detail})"
                shape = tuple(int(extent) for extent in layout.shape)
                # §17.2 storage ratio: prod(n_j) dense samples against the
                # R*sum(n_j) one-dimensional factor values actually stored.
                storage = separable_storage(converted, shape) if len(shape) == converted.dimension else None
                metadata: dict[str, Any] = {
                    "separable_bound": conversion.bound,
                    "separable_detail": conversion.detail,
                }
                if storage is not None:
                    metadata["storage_ratio"] = storage.ratio
                    metadata["dense_values"] = storage.dense_values
                    metadata["separable_values"] = storage.separable_values
                representation = FieldRepresentation(
                    kind,
                    converted,
                    exactness=exactness,
                    error=ledger,
                    evaluation_cost=max(1.0, converted.rank),
                    materialization_cost=0.2,
                    byte_size=_payload_nbytes(converted, storage=storage),
                    metadata=metadata,
                )
            elif kind is RepresentationKind.SPARSE_RESIDUAL:
                from spf.renderer.residuals import SparseResidual

                if isinstance(payload, SparseResidual):
                    residual = payload
                elif field.representation.residual is not None and isinstance(
                    field.representation.residual, SparseResidual
                ):
                    residual = field.representation.residual
                elif isinstance(payload, arithmetic.GaussianChirpAtom):
                    # A closed Gaussian is already an exact symbolic source;
                    # deriving a residual would require materializing every
                    # sample and would defeat analytic planning.  Require an
                    # explicitly supplied residual stream instead.
                    raise ValueError("analytic Gaussian source has no explicit sparse residual")
                else:
                    source = _source_values(field, layout, region)
                    if source is None or np.iscomplexobj(source):
                        raise ValueError("no real residual stream is available")
                    residual = SparseResidual.from_dense(np.asarray(source), threshold=0.0)
                # The retained residual owns its approximation bound. The
                # complete-field ledger accounts for that bound once.
                representation = FieldRepresentation(
                    kind,
                    residual,
                    exactness=Exactness.BOUNDED,
                    error=ledger,
                    evaluation_cost=0.2,
                    materialization_cost=0.1,
                    byte_size=residual.nbytes,
                )
                reason = "sparse residual stream"
            elif kind is RepresentationKind.HERMITIAN_HALF:
                if field.coherence is not Coherence.SIGNED_REAL:
                    raise ValueError("Hermitian half-spectrum requires signed-real field semantics")
                metadata = dict(field.representation.metadata)
                if payload.__class__.__name__ == "HermitianHalfSpectrum" and hasattr(payload, "values"):
                    # A validated half spectrum already owns its carrier.
                    converted = np.asarray(payload.values)
                    full_shape = tuple(int(value) for value in payload.full_shape)
                    metadata.update({"encoding": "rfftn", "full_shape": full_shape, "validated": True})
                    fit_error = 0.0
                    reason = "validated Hermitian half-spectrum"
                elif field.representation.kind is RepresentationKind.HERMITIAN_HALF:
                    converted = payload
                    full_shape_value = metadata.get("full_shape", metadata.get("shape"))
                    if full_shape_value is None and isinstance(payload, np.ndarray):
                        raise ValueError("Hermitian half-spectrum requires full_shape metadata")
                    if full_shape_value is not None:
                        metadata["full_shape"] = tuple(int(value) for value in full_shape_value)
                    metadata.setdefault("encoding", "rfftn")
                    fit_error = 0.0
                    reason = "existing Hermitian half-spectrum"
                else:
                    source = _source_values(field, layout, region)
                    if source is None:
                        raise ValueError("Hermitian candidate requires materializable real samples")
                    # ``_source_values`` is the reference Cartesian producer; it
                    # already hands back the comparison carrier.
                    imaginary_error = float(np.max(np.abs(source.imag))) if source.size else 0.0
                    if imaginary_error > 1.0e-12 * max(1.0, float(np.max(np.abs(source)))):
                        raise ValueError("Hermitian half-spectrum requires a real-valued field")
                    real_source = np.asarray(source.real, dtype=np.float64)
                    converted = np.fft.rfftn(real_source)
                    reconstructed = np.fft.irfftn(
                        converted,
                        s=real_source.shape,
                        axes=tuple(range(real_source.ndim)),
                    )
                    fit_error = float(np.max(np.abs(reconstructed - real_source))) if real_source.size else 0.0
                    metadata.update(
                        {
                            "encoding": "rfftn",
                            "full_shape": tuple(int(value) for value in real_source.shape),
                            "validated": True,
                        }
                    )
                    reason = "rfftn Hermitian half-spectrum"
                ledger = ledger.plus(arithmetic.ErrorLedger(representation_fit=fit_error))
                representation = FieldRepresentation(
                    kind,
                    converted,
                    exactness=Exactness.EXACT if fit_error == 0.0 else Exactness.BOUNDED,
                    error=ledger,
                    metadata=metadata,
                    evaluation_cost=0.4,
                    materialization_cost=0.4,
                    byte_size=_payload_nbytes(converted),
                )
            elif kind is RepresentationKind.DENSE_TYPED:
                if isinstance(payload, (arithmetic.GaussianChirpAtom, arithmetic.AtomMixture)):
                    raise ValueError("an analytic source requires an analytic consumer, not a dense relabel")
                if not _dense_executable(payload):
                    raise ValueError("payload is not a dense plane; it needs its own representation kind")
                structural = payload.__class__.__name__ in {
                    "SpectralTexture",
                    "TensorBlockView",
                    "V3BlockView",
                    "KernelBlockView",
                    "FilteredBlockView",
                }
                typed_payload = payload
                if structural:
                    reason = "structural sample_words consumer; model-specific lowering"
                elif _is_cartesian_array(payload):
                    if not _spf_typed_field(field):
                        # A Cartesian import keeps its honest label; DENSE_TYPED
                        # would only relabel complex128 as SPF (rule 2).
                        raise ValueError("a Cartesian payload of a dense-kind field is not a typed SPF plane")
                    typed_payload, quantization = _dense_spf32_payload(payload)
                    ledger = ledger.plus(arithmetic.ErrorLedger(source_quantization=quantization))
                    reason = "DenseSPF32 fallback: Cartesian payload encoded to packed words"
                else:
                    reason = "dense typed SPF/log plane"
                representation = FieldRepresentation(
                    kind,
                    typed_payload,
                    exactness=Exactness.BOUNDED,
                    error=ledger,
                    evaluation_cost=1.0,
                    materialization_cost=1.0,
                    byte_size=_payload_nbytes(typed_payload),
                )
            elif kind is RepresentationKind.DENSE_CARTESIAN_REFERENCE:
                if isinstance(payload, (arithmetic.GaussianChirpAtom, arithmetic.AtomMixture)):
                    raise ValueError("an unsampled analytic source is not a Cartesian array")
                if not _dense_executable(payload):
                    raise ValueError("payload is not a Cartesian array; it needs its own representation kind")
                if self.backend.name != "reference" and not _cartesian_import(field, payload):
                    # Rule 2: a structured or SPF-typed source never falls to
                    # complex128 outside the reference backend.
                    raise ValueError(
                        "DENSE_CARTESIAN_REFERENCE is a reference/interop representation; "
                        "the dense fallback for this source is DENSE_TYPED (DenseSPF32)"
                    )
                representation = FieldRepresentation(
                    kind,
                    payload,
                    exactness=Exactness.REFERENCE,
                    error=ledger,
                    evaluation_cost=2.0,
                    materialization_cost=2.0,
                    byte_size=max(_payload_nbytes(payload), layout.element_count * 16),
                )
                reason = "Cartesian reference fallback"
                fallback = True
            else:  # pragma: no cover - enum exhaustive guard
                raise ValueError(f"unsupported candidate {kind.value}")
        except (TypeError, ValueError, np.linalg.LinAlgError, OverflowError) as exc:
            representation = FieldRepresentation(
                kind,
                payload,
                exactness=Exactness.APPROXIMATE,
                error=arithmetic.ErrorLedger(source_quantization=math.inf),
                byte_size=_payload_nbytes(payload),
                consumed_operation_indices=tuple(consumed_operation_indices),
            )
            return RepresentationCandidate(
                field.id,
                region,
                kind,
                representation,
                PlanCost(error=math.inf, total=math.inf),
                math.inf,
                math.inf,
                False,
                str(exc),
                capability,
                fallback,
            )
        consumed = tuple(int(index) for index in consumed_operation_indices)
        if consumed and representation.consumed_operation_indices != consumed:
            representation = replace(representation, consumed_operation_indices=consumed)
        from spf.renderer.normalization import (
            complete_field_ledger,
            inherit_representation_error,
            select_representation,
        )

        representation = replace(representation, residual=prepared_residual)
        prepared_source = replace(
            field,
            operations=(),
            representation=replace(
                field.representation,
                payload=payload,
                residual=prepared_residual,
                error=ledger,
                consumed_operation_indices=(),
            ),
        )
        representation = inherit_representation_error(representation, complete_field_ledger(prepared_source))
        execution_field = select_representation(field, representation).field
        full_ledger = complete_field_ledger(execution_field)
        scale = max(1.0, _field_scale(field, layout, region, payload=payload))
        error_bound = float(full_ledger.total_bound)
        legal = _error_contract_accepts(
            field, full_ledger, phase_error=phase_error, scale=scale, budget=self.error_budget
        )
        if not legal:
            reason = f"error contract exceeded ({error_bound:.6g})"
        if pending_operations:
            reason = f"{reason}; pending operations={len(pending_operations)}"
        if preparation_notes:
            reason = f"{reason}; {'; '.join(str(note) for note in preparation_notes)}"
        cost = self._cost(representation, layout, region, field, error_ledger=full_ledger)
        return RepresentationCandidate(
            field.id, region, kind, representation, cost, error_bound, phase_error, legal, reason, capability, fallback
        )

    def _cost(
        self,
        representation: FieldRepresentation,
        layout: SampleLayout,
        region: Region,
        field: Field,
        *,
        error_ledger: arithmetic.ErrorLedger | None = None,
    ) -> PlanCost:
        model = self.backend.cost_model
        cost_representation = representation if error_ledger is None else replace(representation, error=error_ledger)
        cost = model.estimate_representation(cost_representation, layout)
        multiplier = 1.0
        if representation.kind is RepresentationKind.ATOM_MIXTURE:
            payload = representation.payload
            multiplier += 0.05 * len(getattr(payload, "atoms", ()))
        if representation.kind is RepresentationKind.HERMITIAN_HALF:
            multiplier *= 0.6
        if representation.kind is RepresentationKind.SPARSE_RESIDUAL:
            multiplier *= max(0.1, getattr(representation.payload, "density", 1.0))
        if representation.kind is RepresentationKind.DENSE_CARTESIAN_REFERENCE:
            multiplier *= 1.5
        total = cost.total * multiplier
        return PlanCost(
            compute=cost.compute * multiplier,
            bytes_moved=cost.bytes_moved,
            synchronization=cost.synchronization,
            reductions=cost.reductions,
            materialization=cost.materialization,
            error=cost.error,
            total=total,
            memory=cost.memory,
            divergence=cost.divergence,
        )

    def _choose(
        self, candidates: Sequence[RepresentationCandidate], field: Field, region: Region
    ) -> RepresentationCandidate:
        legal = [candidate for candidate in candidates if candidate.legal]
        if not legal:
            if self.allow_fallback:
                for fallback_kind in (RepresentationKind.DENSE_TYPED, RepresentationKind.DENSE_CARTESIAN_REFERENCE):
                    fallback = next((item for item in candidates if item.kind is fallback_kind), None)
                    if fallback is not None and fallback.legal:
                        return replace_candidate(
                            fallback, fallback=True, reason=f"explicit dense fallback selected ({fallback_kind.value})"
                        )
            raise ErrorBudgetExceeded(f"no representation for {field.id!r} in {region!r} satisfies its error contract")
        return min(legal, key=lambda item: (item.cost.total, item.kind.value, item.reason))

    def plan_field(
        self,
        field: Field,
        *,
        regions: Sequence[Region] | None = None,
        layout: SampleLayout | None = None,
        request: MaterializationRequest | None = None,
    ) -> MaterializationPlan:
        self._require_backend()
        selected_request = request
        if selected_request is not None:
            if selected_request.backend and selected_request.backend != self.backend.name:
                raise CapabilityError(
                    f"request backend {selected_request.backend!r} does not match planner backend {self.backend.name!r}"
                )
            if selected_request.layout is not None:
                layout = selected_request.layout
            if selected_request.region is not None:
                regions = (selected_request.region,)
            self.error_budget = selected_request.error_budget
            self.precision = selected_request.precision
        dimension = _dimension(field)
        selected_regions = tuple(regions or (field.support.region or Region((0.0,) * dimension, (1.0,) * dimension),))
        candidates: list[RepresentationCandidate] = []
        decisions: list[RepresentationDecision] = []
        fallbacks: list[str] = []
        for selected_region in selected_regions:
            tile_candidates = self.enumerate_candidates(field, region=selected_region, layout=layout)
            candidates.extend(tile_candidates)
            choice = self._choose(tile_candidates, field, selected_region)
            if choice.fallback:
                fallbacks.append(f"{field.id}@{_region_name(selected_region)}: {choice.reason}")
            decisions.append(
                RepresentationDecision(
                    field.id,
                    _region_name(selected_region),
                    choice.kind.value,
                    tuple(item.kind.value for item in tile_candidates),
                    choice.cost.total,
                    choice.predicted_error,
                    choice.reason,
                )
            )
        liveness = {f"{field.id}@{index}": (index, index) for index in range(len(selected_regions))}
        peak = max((_candidate_bytes(item) for item in candidates if item.legal), default=0)
        pass_records = (
            CompilerPassRecord(
                "representation_candidate_enumeration",
                len(selected_regions),
                len(selected_regions),
                notes=(f"candidates={len(candidates)}",),
            ),
            CompilerPassRecord(
                "materialization_decision",
                len(selected_regions),
                len(decisions),
                notes=(f"backend={self.backend.name}",),
            ),
            CompilerPassRecord(
                "liveness_memory_schedule", len(decisions), len(decisions), notes=(f"peak_bytes={peak}",)
            ),
        )
        metrics = RenderMetrics(
            backend=self.backend.name,
            workload="sfr_materialization",
            resolution=tuple(_layout_for(field, layout, selected_regions[0]).shape),
            representation=decisions[0].selected if decisions else "unknown",
            precision=cast(PrecisionMode, self.precision.mode).value.lower(),
            error_budget=self.error_budget.absolute,
            potential_elements=sum(_layout_for(field, layout, item).element_count for item in selected_regions),
            materialized_elements=sum(_layout_for(field, layout, item).element_count for item in selected_regions),
            peak_memory=peak,
            fallback_conversions=len(fallbacks),
        )
        report = RenderReport(
            metrics,
            ledger=field.ledger,
            passes=pass_records,
            decisions=tuple(decisions),
            fallbacks=tuple(fallbacks),
            capabilities_used=(self.backend.name,),
            diagnostics={
                "field_id": field.id,
                "regions": tuple(_region_name(item) for item in selected_regions),
                "candidate_count": len(candidates),
                "selected": tuple(decision.selected for decision in decisions),
                "budget": self.error_budget.absolute,
            },
        )
        return MaterializationPlan(
            self.backend.name,
            selected_regions,
            tuple(candidates),
            tuple(decisions),
            fallbacks=tuple(fallbacks),
            peak_memory=peak,
            liveness=liveness,
            pass_records=pass_records,
            report=report,
        )

    def plan_fields(
        self,
        fields: Sequence[Field],
        *,
        regions: Sequence[Region] | None = None,
        layout: SampleLayout | None = None,
    ) -> MaterializationPlan:
        if not fields:
            return MaterializationPlan(self.backend.name, tuple(regions or ()), (), ())
        plans = [self.plan_field(value, regions=regions, layout=layout) for value in fields]
        candidates = tuple(item for plan in plans for item in plan.candidates)
        decisions = tuple(item for plan in plans for item in plan.decisions)
        fallbacks = tuple(item for plan in plans for item in plan.fallbacks)
        passes = tuple(item for plan in plans for item in plan.pass_records)
        peak = max((plan.peak_memory for plan in plans), default=0)
        regions_all = tuple(item for plan in plans for item in plan.regions)
        report = RenderReport(
            RenderMetrics(
                backend=self.backend.name,
                workload="sfr_materialization",
                representation=decisions[0].selected if decisions else "unknown",
                precision=cast(PrecisionMode, self.precision.mode).value.lower(),
                error_budget=self.error_budget.absolute,
                potential_elements=sum(
                    _layout_for(field, layout, region).element_count
                    for field, plan in zip(fields, plans, strict=True)
                    for region in plan.regions
                ),
                materialized_elements=sum(
                    _layout_for(field, layout, region).element_count
                    for field, plan in zip(fields, plans, strict=True)
                    for region in plan.regions
                ),
                peak_memory=peak,
                fallback_conversions=len(fallbacks),
            ),
            ledger=arithmetic.ErrorLedger(),
            passes=passes,
            decisions=decisions,
            fallbacks=fallbacks,
            capabilities_used=(self.backend.name,),
            diagnostics={"field_count": len(fields), "candidate_count": len(candidates)},
        )
        return MaterializationPlan(
            self.backend.name,
            regions_all,
            candidates,
            decisions,
            fallbacks=fallbacks,
            peak_memory=peak,
            pass_records=passes,
            report=report,
        )

    def plan(
        self,
        field: Field,
        request: MaterializationRequest | None = None,
        *,
        regions: Sequence[Region] | None = None,
        layout: SampleLayout | None = None,
    ) -> MaterializationPlan:
        return self.plan_field(field, regions=regions, layout=layout, request=request)

    def materialize(
        self,
        field: Field,
        request: MaterializationRequest | None = None,
        *,
        plan: MaterializationPlan | None = None,
        return_certificate: bool = False,
        **options: Any,
    ) -> Any:
        selected_request = request or MaterializationRequest(**options)
        selected_plan = plan or self.plan_field(field, request=selected_request)
        choices = [item for item in selected_plan.selected if item.field_id == field.id]
        if not choices:
            raise PlanningError(f"materialization plan has no decision for {field.id!r}")
        choice = choices[0]
        region = selected_plan.regions[0]
        layout = selected_request.layout or _layout_for(field, None, region)
        materialized = _materialize_payload(field, choice, selected_request, {"layout": layout})
        # Reconstruct the exact source after the same analytic prefix used to
        # make the candidate.  Comparing a transformed candidate against the
        # original pre-operation atom would manufacture a false materialization
        # error (and would also let an unsupported suffix be certified).
        from spf.renderer.normalization import complete_field_ledger, normalize_field, select_representation

        normalized = normalize_field(field, region=region, budget=selected_request.error_budget.absolute).field
        source = None if normalized.operations else _source_values(normalized, layout, region)
        actual_error = 0.0
        phase_error = choice.phase_error
        if source is not None:
            actual = np.asarray(_decode_typed(materialized), dtype=np.complex128)
            # ``source`` comes from the reference producer in that same domain.
            source_array = np.asarray(source)
            if actual.shape == source_array.shape:
                actual_error = float(np.max(np.abs(actual - source_array))) if actual.size else 0.0
                phase_error = max(phase_error, _max_phase_error(actual, source_array))
        ledger = complete_field_ledger(select_representation(field, choice.representation).field)
        if actual_error > ledger.total_bound:
            ledger = ledger.plus(arithmetic.ErrorLedger(materialization=actual_error - ledger.total_bound))
        certificate = MaterializationCertificate(
            field.id,
            region,
            choice.kind,
            _kind(selected_request.output_kind),
            float(ledger.total_bound),
            phase_error,
            selected_request.error_budget,
            _error_contract_accepts(
                field,
                ledger,
                phase_error=phase_error,
                scale=max(1.0, _field_scale(field, layout, region)),
                budget=selected_request.error_budget,
            ),
            choice.fallback,
            self.backend.name,
            layout.element_count,
            layout.element_count,
            int(choice.representation.byte_size or 0),
            int(np.asarray(materialized).nbytes) if isinstance(materialized, np.ndarray) else 0,
            choice.reason,
        )
        if not certificate.accepted:
            raise ErrorBudgetExceeded(f"materialization of {field.id!r} exceeded the requested error contract")
        return MaterializedField(materialized, certificate) if return_certificate else materialized


def _region_name(region: Region) -> str:
    return f"({','.join(f'{value:g}' for value in region.lower)})->({','.join(f'{value:g}' for value in region.upper)})"


def _field_scale(
    field: Field,
    layout: SampleLayout,
    region: Region,
    *,
    payload: Any | None = None,
) -> float:
    selected_payload = field.representation.payload if payload is None else payload
    if isinstance(selected_payload, arithmetic.AtomMixture) and selected_payload.residual is None:
        return selected_payload.dropped_bound + math.fsum(
            _field_scale(field, layout, region, payload=atom) for atom in selected_payload.atoms
        )
    if isinstance(selected_payload, arithmetic.GaussianChirpAtom):
        atom = selected_payload
        if atom.dimension == len(layout.shape) == 2 and layout.element_count:
            if field.chart.dimension == atom.dimension:
                matrix = np.asarray(field.chart.basis) @ np.diag(
                    np.asarray(field.chart.scale) * np.asarray(field.chart.sample_spacing)
                )
                atom = arithmetic.atom_affine(atom, matrix, np.asarray(field.chart.origin))
            # §17.3 header coefficients, shared with the analytic jet candidate.
            real_exponent, _ = _atom_layout_exponents(atom, layout)
            _, maximum = _quadratic_box_range(tuple(real_exponent.tolist()), layout.shape)
            if math.isfinite(maximum):
                try:
                    return float(math.exp(maximum))
                except OverflowError:
                    return math.inf
    if payload is not None and payload is not field.representation.payload:
        field = replace(field, representation=replace(field.representation, payload=payload))
    # §7.3: a word-native source reports its envelope off the L plane, with no
    # decode.  Only a Cartesian family falls through to the reference producer.
    words = _word_native_source(field, layout)
    if words is not None:
        # Same value the decode-then-``max(abs)`` path reported, including the
        # all-zero plane (0.0) and the empty plane (1.0).
        return _word_plane_scale(words)
    source = _source_values(field, layout, region)
    if source is None or not source.size:
        return 1.0
    maximum = float(np.max(np.abs(source)))
    return maximum if math.isfinite(maximum) else 1.0


def _candidate_bytes(candidate: RepresentationCandidate) -> int:
    return int(candidate.representation.byte_size or 0)


def _separable_conversion(payload: Any, region: Region | None = None) -> SeparableConversion | None:
    """Certified separable conversion, delegated to the single §17 implementation.

    The conversion accounts for the off-diagonal chirp term, the mixture
    residual, and the mixture's ``dropped_bound``.  Its ``bound`` is what the
    candidate ledger must carry; only ``bound == 0`` may be tagged ``EXACT``.
    """

    return separable_conversion(payload, region)


def _to_separable(payload: Any, region: Region | None = None) -> arithmetic.SeparableField | None:
    """Return an *exact* separable form, or ``None``.

    An inexact conversion is never returned here: a caller that can publish a
    bound calls :func:`_separable_conversion` instead.  Before the v0.2
    conformance audit this function dropped ``AtomMixture.residual`` and used an
    absolute ``1e-12`` off-diagonal tolerance, and the planner tagged the result
    ``SEPARABLE EXACT`` with bound ``0``.
    """

    conversion = _separable_conversion(payload, region)
    if conversion is None or not conversion.exact:
        return None
    return conversion.field


def replace_candidate(candidate: RepresentationCandidate, *, fallback: bool, reason: str) -> RepresentationCandidate:
    return RepresentationCandidate(
        candidate.field_id,
        candidate.region,
        candidate.kind,
        candidate.representation,
        candidate.cost,
        candidate.predicted_error,
        candidate.phase_error,
        candidate.legal,
        reason,
        candidate.capability,
        fallback,
    )


def _materialize_payload(
    field: Field,
    candidate: RepresentationCandidate,
    request: MaterializationRequest | None,
    options: Mapping[str, Any] | None = None,
) -> Any:
    """Use the canonical runtime after binding a planner-selected carrier."""
    from spf.renderer.normalization import select_representation
    from spf.renderer.runtime import materialize_field

    options = dict(options or {})
    selected_request = request or MaterializationRequest(
        layout=options.get("layout"),
        region=options.get("region"),
        output_kind=options.get("output_kind", FieldKind.DENSE),
        backend="reference",
    )
    layout = _layout_for(field, selected_request.layout, candidate.region)
    if selected_request.layout is None:
        selected_request = replace(selected_request, layout=layout)
    if selected_request.output_kind is FieldKind.DENSE and selected_request.output_format in {None, "native"}:
        selected_request = replace(selected_request, output_format="complex")
    selected = select_representation(field, candidate.representation).field
    return materialize_field(selected, selected_request)


def plan_materialization(
    field: Field,
    request: MaterializationRequest | None = None,
    *,
    backend: str | SFRBackend | None = None,
    regions: Sequence[Region] | None = None,
    layout: SampleLayout | None = None,
    **options: Any,
) -> MaterializationPlan:
    selected_backend = backend or (request.backend if request is not None else "reference")
    planner = MaterializationPlanner(
        backend=selected_backend,
        precision=options.pop("precision", request.precision if request is not None else PrecisionContract()),
        error_budget=options.pop("error_budget", request.error_budget if request is not None else ErrorBudget()),
        allow_fallback=options.pop("allow_fallback", True),
        candidate_kinds=options.pop("candidate_kinds", None),
    )
    return planner.plan_field(field, regions=regions, layout=layout, request=request)


def materialize_field(
    field: Field,
    request: MaterializationRequest | None = None,
    *,
    backend: str | SFRBackend | None = None,
    return_certificate: bool = False,
    **options: Any,
) -> Any:
    planner = MaterializationPlanner(
        backend=backend or (request.backend if request is not None else "reference"),
        precision=options.pop("precision", request.precision if request is not None else PrecisionContract()),
        error_budget=options.pop("error_budget", request.error_budget if request is not None else ErrorBudget()),
        allow_fallback=options.pop("allow_fallback", True),
    )
    return planner.materialize(field, request, return_certificate=return_certificate, **options)


def plan(field: Field, request: MaterializationRequest | None = None, **kwargs: Any) -> MaterializationPlan:
    return plan_materialization(field, request, **kwargs)


__all__ = [
    "CapabilityDecision",
    "CapabilityError",
    "ErrorBudgetExceeded",
    "ErrorBudgetExceededError",
    "MaterializationCertificate",
    "MaterializationPlan",
    "MaterializationPlanner",
    "MaterializedField",
    "PlannerOptions",
    "PlanningError",
    "RepresentationCandidate",
    "materialize_field",
    "plan",
    "plan_materialization",
]
