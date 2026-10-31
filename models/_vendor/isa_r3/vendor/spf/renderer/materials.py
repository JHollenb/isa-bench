"""Typed material terms for the SFR v0.2 reference renderer.

The material surface deliberately keeps two algebras visible:

* positive factors are multiplied and divided in :class:`spf.tensor.v02.LogPos`;
* colour matrices and other sign-changing operations cross an explicit
  :class:`SignedColor` boundary.

This is a small, deterministic CPU reference implementation.  It is useful
both as a shader-lowering oracle and as the local shading path used by
``scene.Renderer``; it does not claim to be a native microfacet backend.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.model import (
    Coherence,
    CoordinateChart,
    Domain,
    Field,
    FieldRepresentation,
    RepresentationKind,
    Semantic,
)
from spf.renderer.v02 import FieldKind
from spf.tensor import v02

_EPS = np.finfo(np.float64).tiny
_PI = math.pi


def _reject_bool(value: Any, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not bool")


def _field_payload(value: Any, *, name: str) -> Any:
    """Extract a field payload while preserving its semantic boundary."""

    if not isinstance(value, Field):
        return value
    if value.coherence is Coherence.COHERENT_COMPLEX:
        raise TypeError(
            f"{name} is coherent-complex; call coherent_intensity() explicitly "
            "before using it as an incoherent material factor"
        )
    return value.representation.payload


def _block_payload(value: Any) -> tuple[Any, Field | None] | None:
    """Return an addressed renderer block and its optional Field owner."""

    owner = value if isinstance(value, Field) else None
    payload = value.representation.payload if owner is not None else value
    from spf.renderer.carriers import as_block_view

    block = as_block_view(payload)
    return None if block is None else (block, owner)


def _closed_block_payload(value: Any, *, name: str) -> tuple[Any, Field | None] | None:
    """Resolve parameter-closed Field operations before addressed sampling."""

    selected = _block_payload(value)
    if selected is None or selected[1] is None:
        return selected
    from spf.renderer.normalization import normalize_field

    normalized = normalize_field(selected[1]).field
    if normalized.operations:
        pending = ", ".join(operation.op for operation in normalized.operations)
        raise TypeError(f"{name} block source has unsupported deferred operations: {pending}")
    resolved = _block_payload(normalized)
    if resolved is not None:
        from spf.renderer.block_filter import FilteredBlockView

        if isinstance(resolved[0], FilteredBlockView) and not resolved[0].frequency_domain:
            raise TypeError(f"{name} spatial filter requires an explicit materialization boundary")
    return resolved


def _material_block_query_points(
    owner: Field | None,
    coordinates: np.ndarray,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray:
    """Return physical points for a compact material residual query.

    A compact RGB vector is often a one-dimensional channel carrier while a
    rank-three RGB block has two spatial axes plus a channel axis.  Residual
    callables belong to the owning Field chart, so neither case may pass the
    synthetic channel coordinate as a physical dimension.
    """

    if owner is None:
        return coordinates.reshape(1, -1)
    dimension = owner.chart.dimension
    if dimension == 2:
        return coordinates.reshape(1, 2)
    if dimension == 0:
        return np.empty((1, 0), dtype=np.float64)
    if dimension != len(shape):
        raise TypeError(
            f"{name} compact residual requires a chart with dimension 0, 2, or the compact rank; "
            f"received chart dimension {dimension} for rank {len(shape)}"
        )
    if not shape:
        return np.empty((1, 0), dtype=np.float64)
    addresses = np.arange(int(math.prod(shape)), dtype=np.int64)
    indices = np.asarray(np.unravel_index(addresses, shape), dtype=np.float64).T
    return np.asarray([owner.chart.coordinates(index) for index in indices], dtype=np.float64)


def _material_block_lattice_points(
    owner: Field | None,
    shape: tuple[int, ...],
    *,
    name: str,
) -> np.ndarray | None:
    """Map every compact address to its declared physical/chart point."""

    count = int(math.prod(shape)) if shape else 1
    if owner is None:
        if not shape:
            return np.empty((count, 0), dtype=np.float64)
        indices = np.asarray(np.unravel_index(np.arange(count, dtype=np.int64), shape), dtype=np.float64).T
        return indices
    dimension = owner.chart.dimension
    if dimension == 0:
        return np.empty((count, 0), dtype=np.float64)
    if dimension == len(shape):
        if not shape:
            return np.empty((count, 0), dtype=np.float64)
        indices = np.asarray(np.unravel_index(np.arange(count, dtype=np.int64), shape), dtype=np.float64).T
        return np.asarray([owner.chart.coordinates(index) for index in indices], dtype=np.float64)
    if dimension == 2 and len(shape) == 3 and shape[-1] == 3:
        indices = np.asarray(np.unravel_index(np.arange(count, dtype=np.int64), shape), dtype=np.float64).T
        spatial_indices = indices[:, :2]
        return np.asarray([owner.chart.coordinates(index) for index in spatial_indices], dtype=np.float64)
    if dimension == 2 and len(shape) < 2:
        return None
    raise TypeError(
        f"{name} compact block requires a chart with dimension 0, 2, or the compact rank; "
        f"received chart dimension {dimension} for rank {len(shape)}"
    )


def _material_block_residual_parts(residual: Any) -> tuple[Any | None, Any | None]:
    """Split lattice-addressed and physical callable residual terms."""

    from spf.renderer.normalization import _ResidualSum, _ScaledResidual

    if isinstance(residual, _ResidualSum):
        lattice_terms: list[Any] = []
        direct_terms: list[Any] = []
        for item in residual.items:
            lattice, direct = _material_block_residual_parts(item)
            if lattice is not None:
                lattice_terms.append(lattice)
            if direct is not None:
                direct_terms.append(direct)
        if not lattice_terms:
            lattice_result = None
        elif len(lattice_terms) == 1:
            lattice_result = lattice_terms[0]
        else:
            lattice_result = _ResidualSum(tuple(lattice_terms))
        if not direct_terms:
            direct_result = None
        elif len(direct_terms) == 1:
            direct_result = direct_terms[0]
        else:
            direct_result = _ResidualSum(tuple(direct_terms))
        return lattice_result, direct_result
    if isinstance(residual, _ScaledResidual):
        lattice, direct = _material_block_residual_parts(residual.residual)
        return (
            None if lattice is None else _ScaledResidual(lattice, residual.exposure),
            None if direct is None else _ScaledResidual(direct, residual.exposure),
        )
    if callable(residual):
        return None, residual
    return residual, None


def _material_block_residual_values(
    residual: Any,
    addresses: np.ndarray,
    shape: tuple[int, ...],
    points: np.ndarray,
    *,
    count: int,
    name: str,
) -> np.ndarray:
    """Read compact lattice residuals and evaluate callable residuals in-chart."""

    from spf.renderer.volume import _real_residual, _residual_lattice_or_direct

    try:
        lattice, direct = _residual_lattice_or_direct(residual, addresses, shape, points)
    except ValueError as exc:
        raise TypeError(
            f"{name} callable residual must return one scalar per declared physical/chart query"
        ) from exc
    result = np.zeros(count, dtype=np.complex128)
    if lattice is not None:
        values = np.asarray(lattice, dtype=np.complex128).reshape(-1)
        if values.size == 1:
            result += values[0]
        elif values.size == count:
            result += values
        else:
            raise TypeError(f"{name} lattice residual does not match compact addressed values")
    if direct is not None:
        values = np.asarray(direct, dtype=np.complex128).reshape(-1)
        if values.size == 1:
            result += values[0]
        elif values.size == count:
            result += values
        else:
            raise TypeError(
                f"{name} callable residual must be scalar per physical query for compact RGB material sampling"
            )
    return _real_residual(result, tolerance=1.0e-12)


def _block_real_values(value: Any, *, name: str) -> np.ndarray | None:
    """Cross an explicit dense numeric boundary by reading addressed words.

    This helper is reserved for callers that explicitly request a compact
    numeric vector/matrix.  It still allocates the requested terminal array;
    spatial sampling and structural operators must use their addressed views
    instead of routing through this boundary.
    """

    selected = _closed_block_payload(value, name=name)
    if selected is None:
        return None
    block, owner = selected
    if owner is not None and owner.coherence is Coherence.COHERENT_COMPLEX:
        raise TypeError(
            f"{name} is coherent-complex; call coherent_intensity() explicitly before using its block payload"
        )
    from spf.renderer.volume import _project_block_real

    shape = tuple(int(value) for value in block.shape)
    count = int(math.prod(shape)) if shape else 1
    addresses = np.arange(count, dtype=np.int64)
    values = _project_block_real(
        block,
        addresses,
        value_kind=None if owner is None else str(owner.value_kind),
    ).reshape(-1)
    if owner is not None:
        support_kind = str(getattr(owner.support.kind, "value", owner.support.kind)).lower()
        if support_kind == "empty":
            return np.zeros(shape if shape else (), dtype=np.float64)
        residual = owner.representation.residual
        lattice_residual, direct_residual = (
            _material_block_residual_parts(residual) if residual is not None else (None, None)
        )
        points = _material_block_lattice_points(owner, shape, name=name)
        if direct_residual is not None and points is None:
            raise TypeError(f"{name} callable residual requires an explicit physical/chart query")
        if lattice_residual is not None or direct_residual is not None:
            residual_points = points if points is not None else np.empty((count, 0), dtype=np.float64)
            values += _material_block_residual_values(
                residual,
                addresses,
                shape,
                residual_points,
                count=count,
                name=name,
            )
        if owner.support.region is not None:
            if points is None:
                raise TypeError(f"{name} support region requires an explicit physical/chart query")
            mask = np.asarray([owner.support.region.contains(point.tolist()) for point in points], dtype=bool)
            values = np.where(mask, values, 0.0)
    return values.reshape(shape if shape else ())


def _numeric_scalar(value: Any, *, name: str, allow_negative: bool = True) -> float:
    value = _field_payload(value, name=name)
    _reject_bool(value, name)
    if isinstance(value, v02.RawLogCode):
        raise TypeError(f"{name} received a raw log code; use LogPos or LogSigned explicitly")
    if isinstance(value, v02.SPF32Value):
        raise TypeError(f"{name} received a packed SPF32 word; decode it explicitly")
    if isinstance(value, (v02.LogPos, v02.LogSigned)):
        result = float(value.value)
    elif isinstance(value, (complex, np.complexfloating)):
        if value.imag != 0.0:
            raise TypeError(f"{name} cannot implicitly discard a complex component")
        result = float(value.real)
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a finite real scalar") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if not allow_negative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _numeric_array(value: Any, *, name: str, allow_negative: bool = True) -> np.ndarray:
    """Decode a real scalar/vector without guessing packed SPF32 words."""

    block_values = _block_real_values(value, name=name)
    if block_values is not None:
        result = np.asarray(block_values, dtype=np.float64)
        if not allow_negative and np.any(result < 0.0):
            raise ValueError(f"{name} must be non-negative")
        return result
    value = _field_payload(value, name=name)
    if isinstance(value, (v02.LogPos, v02.LogSigned, v02.RawLogCode, v02.SPF32Value)):
        return np.asarray([_numeric_scalar(value, name=name, allow_negative=allow_negative)], dtype=np.float64)
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        if np.any(np.imag(raw) != 0.0):
            raise TypeError(f"{name} cannot implicitly discard a complex component")
        raw = np.real(raw)
    if raw.dtype == object:
        values = np.empty(raw.shape, dtype=np.float64)
        for index in np.ndindex(raw.shape):
            values[index] = _numeric_scalar(raw[index], name=name, allow_negative=allow_negative)
        result = values
    else:
        try:
            result = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain finite real values") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    if not allow_negative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


def as_native_color(value: Any, *, name: str = "color", channels: int = 3, allow_negative: bool = False) -> np.ndarray:
    """Return a finite colour vector, broadcasting one scalar to all channels."""

    values = _numeric_array(value, name=name, allow_negative=allow_negative).reshape(-1)
    if values.size == 1:
        values = np.repeat(values, channels)
    if values.size != channels:
        raise ValueError(f"{name} must have one value or exactly {channels} channels")
    return values.astype(np.float64, copy=True)


def as_log_pos(value: Any, *, name: str = "value") -> v02.LogPos:
    """Convert one non-negative real value to a typed positive-log value.

    Packed SPF32 words and raw log codes are intentionally rejected: a material
    factor must not infer whether an integer is a wire word or a log code.
    """

    value = _field_payload(value, name=name)
    _reject_bool(value, name)
    if isinstance(value, v02.LogPos):
        return value
    if isinstance(value, v02.LogSigned):
        if value.is_nan or value.value < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return v02.LogPos(value.L)
    if isinstance(value, (v02.RawLogCode, v02.SPF32Value)):
        raise TypeError(f"{name} requires LogPos or a real value, not a raw SPF32 carrier")
    return v02.LogPos.from_value(_numeric_scalar(value, name=name, allow_negative=False))


def as_log_pos_array(value: Any, *, name: str = "value") -> np.ndarray:
    """Convert a scalar/vector of non-negative values to an object LogPos array."""

    value = _field_payload(value, name=name)
    if isinstance(value, (v02.LogPos, v02.LogSigned, v02.RawLogCode, v02.SPF32Value)):
        return np.asarray([as_log_pos(value, name=name)], dtype=object)
    raw = np.asarray(value)
    if raw.ndim == 0:
        return np.asarray([as_log_pos(raw.item(), name=name)], dtype=object)
    result = np.empty(raw.shape, dtype=object)
    for index in np.ndindex(raw.shape):
        result[index] = as_log_pos(raw[index], name=name)
    return result


def block_words(value: Any, *, name: str = "value") -> tuple[np.ndarray, Field | None] | None:
    """Addressed packed words of a compact block payload, without any decode.

    Returns ``None`` when ``value`` is not a block payload, or when the block
    cannot be read as closed words (fractional header offsets, a
    representation residual, or a support region that needs a chart query):
    those cases keep their explicit float boundary in :func:`_block_real_values`.
    """

    selected = _closed_block_payload(value, name=name)
    if selected is None:
        return None
    block, owner = selected
    if owner is not None and owner.coherence is Coherence.COHERENT_COMPLEX:
        raise TypeError(f"{name} is coherent-complex; call coherent_intensity() explicitly before reading words")
    if owner is not None and owner.representation.residual is not None:
        return None
    shape = tuple(int(item) for item in block.shape)
    count = int(math.prod(shape)) if shape else 1
    addresses = np.arange(count, dtype=np.int64)
    if owner is not None:
        support_kind = str(getattr(owner.support.kind, "value", owner.support.kind)).lower()
        if support_kind == "empty":
            return np.full(shape if shape else (), v02.SPF32_WORD_ZERO, dtype=np.uint32), owner
        if owner.support.region is not None:
            return None
    value_kind = None if owner is None else str(owner.value_kind).lower()
    positive_observable = value_kind in {"log_pos", "radiance", "incoherent_radiance", "optical_depth", "opacity"}
    words = np.asarray(block.sample_words(addresses, phase=not positive_observable))
    projected = np.asarray(block.project_words(words, addresses))
    if projected.dtype.kind not in "iu":
        return None
    return np.asarray(projected, dtype=np.uint32).reshape(shape if shape else ()), owner


def as_log_pos_plane(value: Any, *, name: str = "value", channels: int | None = None) -> tuple[Any, str]:
    """Return ``(LogPosPlane, ingress)`` for a positive scalar/vector source.

    ``ingress`` is ``"addressed_block_words"`` when the plane was read straight
    from packed block words, ``"typed"`` for ``LogPos`` objects and
    ``"native_scalar"`` for ordinary reals (encoded, never decoded).  Signed
    words with a negative sign are rejected: a positive factor cannot absorb a
    sign.
    """

    from spf.tensor.v02_planes import LogPosPlane, LogSignedPlane

    selected = block_words(value, name=name)
    if selected is not None:
        words, _ = selected
        signed = LogSignedPlane.from_words(words.reshape(-1))
        if np.any(signed.sign < 0):
            raise ValueError(f"{name} must be non-negative")
        plane = LogPosPlane(np.where(signed.sign == 0, v02.SPF32_L_ZERO, signed.L.astype(np.int64)))
        plane = LogPosPlane(np.where(signed.is_nan, v02.SPF32_NAN_L, plane.L.astype(np.int64)))
        ingress = "addressed_block_words"
    else:
        payload = _field_payload(value, name=name)
        if isinstance(payload, (v02.LogPos, v02.LogSigned)):
            plane = LogPosPlane.from_scalars([as_log_pos(payload, name=name)])
            ingress = "typed"
        else:
            raw = np.asarray(payload)
            if raw.dtype == object:
                plane = LogPosPlane.from_scalars([as_log_pos(item, name=name) for item in raw.reshape(-1)])
                ingress = "typed"
            else:
                values = _numeric_array(raw, name=name, allow_negative=False).reshape(-1)
                plane = LogPosPlane.from_values(values)
                ingress = "native_scalar"
    if channels is not None:
        if plane.size == 1 and channels > 1:
            plane = LogPosPlane(np.repeat(plane.L, channels))
        if plane.size != channels:
            raise ValueError(f"{name} must have one value or exactly {channels} channels")
    return plane, ingress


def log_product_plane(factors: Iterable[Any]) -> Any:
    """Multiply positive factor planes elementwise (``LogPosPlane.mul``)."""

    from spf.tensor.v02_planes import LogPosPlane

    result: LogPosPlane | None = None
    for index, value in enumerate(factors):
        plane = value if isinstance(value, LogPosPlane) else as_log_pos_plane(value, name=f"factor[{index}]")[0]
        result = plane if result is None else result.mul(plane)
    if result is None:
        raise ValueError("log_product_plane requires at least one factor")
    return result


def log_sum_plane(terms: Any, *, axis: int = 0) -> Any:
    """Sum positive terms along ``axis`` with the plane pivot reducer."""

    from spf.tensor.v02_planes import LogPosPlane, reduce_logpos

    plane = terms if isinstance(terms, LogPosPlane) else LogPosPlane.from_scalars(terms)
    return reduce_logpos(plane, axis=axis)


def decode_log_pos(value: Any) -> float | np.ndarray:
    """Decode a positive-log scalar or object array to ordinary reals."""

    if isinstance(value, v02.LogPos):
        return float(value.value)
    if isinstance(value, v02.LogSigned):
        return float(value.value)
    raw = np.asarray(value)
    if raw.ndim == 0:
        return float(as_log_pos(raw.item()).value)
    result = np.empty(raw.shape, dtype=np.float64)
    for index in np.ndindex(raw.shape):
        result[index] = float(as_log_pos(raw[index]).value)
    return result


def log_product(values: Iterable[Any]) -> v02.LogPos:
    """Multiply positive factors using the SPF positive-log algebra."""

    result = v02.LogPos.from_value(1.0)
    for index, value in enumerate(values):
        result = v02.logpos_mul(result, as_log_pos(value, name=f"factor[{index}]"))
    return result


def log_quotient(numerator: Any, denominator: Any) -> v02.LogPos:
    """Divide positive factors using the SPF positive-log algebra."""

    return v02.logpos_div(as_log_pos(numerator, name="numerator"), as_log_pos(denominator, name="denominator"))


def log_sum(values: Iterable[Any]) -> v02.LogPos:
    """Sum positive terms with the canonical positive-log ROM path."""

    items = list(values)
    if not items:
        return v02.LogPos.zero()
    result = as_log_pos(items[0], name="term[0]")
    for index, value in enumerate(items[1:], start=1):
        result = v02.logadd_pos(result, as_log_pos(value, name=f"term[{index}]"))
    return result


@dataclass(frozen=True, slots=True)
class SignedColor:
    """Explicit signed-real colour boundary.

    The channels remain typed ``LogSigned`` values until a caller asks for
    ordinary floating-point values through :attr:`values` or ``as_array``.
    """

    channels: tuple[v02.LogSigned, ...]

    def __post_init__(self) -> None:
        normalized = tuple(
            item if isinstance(item, v02.LogSigned) else v02.LogSigned.from_value(_numeric_scalar(item, name="channel"))
            for item in self.channels
        )
        if not normalized:
            raise ValueError("SignedColor requires at least one channel")
        object.__setattr__(self, "channels", normalized)

    @classmethod
    def from_values(cls, values: Any, *, channels: int = 3) -> SignedColor:
        real = as_native_color(values, name="signed color", channels=channels, allow_negative=True)
        return cls(tuple(v02.LogSigned.from_value(float(value)) for value in real))

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(float(item.value) for item in self.channels)

    def as_array(self) -> np.ndarray:
        return np.asarray(self.values, dtype=np.float64)

    @property
    def codes(self) -> tuple[tuple[int, int], ...]:
        return tuple((item.L, item.sign) for item in self.channels)

    def __iter__(self) -> Iterator[v02.LogSigned]:
        return iter(self.channels)

    def __len__(self) -> int:
        return len(self.channels)

    def __getitem__(self, index: int) -> v02.LogSigned:
        return self.channels[index]


def to_signed_color(value: Any, *, channels: int = 3, name: str = "color") -> SignedColor:
    """Convert a real colour to ``LogSigned`` explicitly.

    A coherent field, complex scalar, packed SPF32 word, or raw log code is
    never promoted implicitly.  Use :func:`coherent_intensity` for the
    intentional coherent-complex-to-positive boundary.
    """

    if isinstance(value, Field) and value.coherence is Coherence.COHERENT_COMPLEX:
        raise TypeError(
            f"{name} is coherent-complex; call coherent_intensity() explicitly before signed colour conversion"
        )
    values = as_native_color(value, name=name, channels=channels, allow_negative=True)
    return SignedColor(tuple(v02.LogSigned.from_value(float(item)) for item in values))


explicit_signed_color = to_signed_color
positive_to_signed_color = to_signed_color


def signed_color_values(value: SignedColor | Any, *, channels: int = 3) -> np.ndarray:
    if isinstance(value, SignedColor):
        if len(value) != channels:
            raise ValueError(f"signed color must have {channels} channels")
        return value.as_array()
    return to_signed_color(value, channels=channels).as_array()


def apply_color_matrix(matrix: Any, color: SignedColor | Any) -> SignedColor:
    """Apply a sign-changing colour matrix through the explicit signed path."""

    # Diagonal SPF kernels already carry the only non-zero entries in their
    # structural expression.  Read exactly those entries and apply them
    # elementwise; passing through ``_block_real_values`` would allocate an
    # otherwise dense n-by-n matrix at this consumer boundary.
    diagonal_values: np.ndarray | None = None
    from spf.tensor.kernel import SPFKernel

    if isinstance(matrix, SPFKernel) and matrix.expr.op == "DiagonalSPF":
        diagonal = np.asarray(matrix.expr.args[0]["diag_words"], dtype=np.uint32).reshape(-1)
        size = int(diagonal.size)
        addresses = np.arange(size, dtype=np.int64) * (size + 1)
        from spf.tensor.core import decode_words

        decoded = np.asarray(matrix.sample_words(addresses), dtype=np.uint32)
        values = np.asarray(decode_words(decoded), dtype=np.complex128)
        if np.any(~np.isfinite(values.real)) or np.any(np.abs(values.imag) > 1.0e-12):
            raise TypeError("color matrix diagonal must contain finite real values")
        diagonal_values = values.real
        transform_shape = (size, size)
    else:
        block_values = _block_real_values(matrix, name="color matrix")
        transform = np.asarray(matrix if block_values is None else block_values, dtype=np.float64)
        if transform.ndim != 2:
            raise ValueError("color matrix must be a two-dimensional matrix")
        transform_shape = transform.shape
    if transform_shape[0] == 0:
        raise ValueError("color matrix must have at least one output channel")
    values = signed_color_values(color, channels=transform_shape[1])
    result = diagonal_values * values if diagonal_values is not None else transform @ values
    return SignedColor(tuple(v02.LogSigned.from_value(float(item)) for item in result))


def coherent_intensity(value: Any) -> Any:
    """Explicitly demote coherent complex data through magnitude-square."""

    if isinstance(value, Field):
        if value.coherence is not Coherence.COHERENT_COMPLEX:
            raise TypeError("coherent_intensity requires a coherent-complex Field")
        from dataclasses import replace

        from spf.renderer.blocks import BlockView

        if isinstance(value.representation.payload, BlockView):
            from spf.renderer.normalization import normalize_field

            normalized = normalize_field(value).field
            block = normalized.representation.payload
            if normalized.operations or normalized.representation.residual is not None:
                from spf.renderer.normalization import complete_field_ledger

                return replace(
                    value.with_operation("abs2"),
                    coherence=Coherence.INCOHERENT_POSITIVE,
                    value_kind=FieldKind.LOG_POS,
                    semantic=Semantic.RADIANCE,
                    metadata={
                        **value.metadata,
                        "explicit_intensity": True,
                        "intensity_input_error": float(complete_field_ledger(value).total_bound),
                    },
                )
            from spf.renderer.native import _scale_ledger_for_observable
            from spf.renderer.normalization import complete_field_ledger

            ledger = complete_field_ledger(value)
            if ledger.total_bound:
                from spf.scalar._constants import SPF32
                from spf.tensor.core import unpack_L

                from .block_filter import FilteredBlockView, sample_source
                from .blocks import V3BlockView

                fmt = block.block.representation.format if isinstance(block, V3BlockView) else SPF32
                if isinstance(block, FilteredBlockView):
                    # Error propagation observes the actual composed amplitude;
                    # it may require the explicit FFT boundary for spatial data.
                    maximum = float(np.max(np.abs(sample_source(block)), initial=0))
                else:
                    words = block.sample_words(block.active_addresses(), phase=False)
                    logs = unpack_L(words, fmt)
                    finite = (logs != fmt.l_code_zero) & (words != fmt.word_nan)
                    with np.errstate(over="ignore"):
                        maximum = (
                            float(np.exp2(float(np.max(logs[finite])) / fmt.l_scale + block.log_offset / 256))
                            if np.any(finite)
                            else 0.0
                        )
                ledger = _scale_ledger_for_observable(ledger, 2 * maximum + ledger.total_bound)
            return replace(
                value,
                coherence=Coherence.INCOHERENT_POSITIVE,
                value_kind=FieldKind.LOG_POS,
                semantic=Semantic.RADIANCE,
                representation=replace(
                    normalized.representation, payload=block._replace(observable="intensity"), error=ledger
                ),
                operations=(),
                support=normalized.support,
            )
        if isinstance(value.representation.payload, v02.GaussianChirpAtom):
            from spf.renderer.native import _prepare_native_atom, _scale_ledger_for_observable
            from spf.renderer.normalization import complete_field_ledger

            try:
                atom, support = _prepare_native_atom(value)
            except NotImplementedError:
                pass
            else:
                atom = v02.GaussianChirpAtom.from_arrays(
                    2 * np.real(atom.kappa),
                    2 * np.real(atom.h),
                    2 * np.real(atom.C),
                    allow_nondecaying=atom.allow_nondecaying,
                )
                ledger = complete_field_ledger(value)
                if ledger.total_bound:
                    # Propagate amplitude uncertainty through |z|^2. A finite
                    # global Gaussian peak is sufficient; otherwise keep an
                    # explicitly unbounded certificate.
                    matrix = np.real(atom.C)
                    if np.min(np.linalg.eigvalsh(matrix)) > 0:
                        peak_intensity = float(
                            np.exp(
                                np.real(atom.kappa) + 0.5 * np.real(atom.h) @ np.linalg.solve(matrix, np.real(atom.h))
                            )
                        )
                        scale = 2 * np.sqrt(peak_intensity) + ledger.total_bound
                    else:
                        scale = float("inf")
                    ledger = _scale_ledger_for_observable(ledger, scale)
                return replace(
                    value,
                    coherence=Coherence.INCOHERENT_POSITIVE,
                    value_kind=FieldKind.LOG_POS,
                    semantic=Semantic.RADIANCE,
                    operations=(),
                    support=support,
                    representation=replace(value.representation, payload=atom, error=ledger),
                )
        # General coherent sums retain their phases until after interference.
        # This is a terminal semantic change; it is never distributed over a
        # mixture's atoms or moved ahead of pending propagation operations.
        from spf.renderer.normalization import complete_field_ledger

        return replace(
            value.with_operation("abs2"),
            coherence=Coherence.INCOHERENT_POSITIVE,
            value_kind=FieldKind.LOG_POS,
            semantic=Semantic.RADIANCE,
            metadata={
                **value.metadata,
                "explicit_intensity": True,
                "intensity_input_error": float(complete_field_ledger(value).total_bound),
            },
        )
    if isinstance(value, v02.SPF32Value):
        return float(v02.spf_abs2(value).value)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return float(v02.spf_abs2(int(value)).value)
    if isinstance(value, np.ndarray) and value.dtype.kind in "iu":
        result = np.empty(value.shape, dtype=np.float64)
        for index in np.ndindex(value.shape):
            result[index] = float(v02.spf_abs2(int(value[index])).value)
        return result
    raw = np.asarray(value)
    if not np.iscomplexobj(raw):
        raise TypeError("coherent_intensity requires complex values or packed SPF32 words")
    result = np.square(np.abs(raw))
    return float(result) if result.ndim == 0 else result


@dataclass(frozen=True, slots=True)
class LogMaterialFactors:
    """Positive-log factors emitted by a material evaluation."""

    base_color: tuple[v02.LogPos, ...]
    fresnel: tuple[v02.LogPos, ...]
    distribution: v02.LogPos
    geometry: v02.LogPos
    denominator: v02.LogPos

    @property
    def value(self) -> tuple[v02.LogPos, ...]:
        result: list[v02.LogPos] = []
        for base, fresnel in zip(self.base_color, self.fresnel, strict=True):
            product = log_product((base, fresnel, self.distribution, self.geometry))
            result.append(log_quotient(product, self.denominator))
        return tuple(result)

    @property
    def values(self) -> tuple[float, ...]:
        return tuple(float(item.value) for item in self.value)

    @property
    def log_values(self) -> tuple[v02.LogPos, ...]:
        return self.value

    @property
    def plane(self) -> Any:
        """The same quotient evaluated on a ``LogPosPlane`` of shape ``(channels,)``."""

        from spf.tensor.v02_planes import LogPosPlane

        channels = len(self.base_color)
        base = LogPosPlane.from_scalars(self.base_color)
        fresnel = LogPosPlane.from_scalars(self.fresnel)
        distribution = LogPosPlane.from_scalars([self.distribution] * channels)
        geometry = LogPosPlane.from_scalars([self.geometry] * channels)
        denominator = LogPosPlane.from_scalars([self.denominator] * channels)
        return log_product_plane((base, fresnel, distribution, geometry)).div(denominator)


def _unit(vector: Any, *, name: str) -> np.ndarray:
    values = _numeric_array(vector, name=name).reshape(-1)
    if values.size != 3:
        raise ValueError(f"{name} must be a three-vector")
    norm = float(np.linalg.norm(values))
    if not math.isfinite(norm) or norm <= _EPS:
        raise ValueError(f"{name} must be a finite nonzero vector")
    return values / norm


def _clamped_dot(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def fresnel_schlick(cos_theta: float, f0: float) -> float:
    cosine = float(np.clip(cos_theta, 0.0, 1.0))
    base = float(np.clip(f0, 0.0, 1.0))
    return base + (1.0 - base) * (1.0 - cosine) ** 5


def ggx_distribution(n_dot_h: float, roughness: float) -> float:
    cosine = float(np.clip(n_dot_h, 0.0, 1.0))
    alpha = max(float(roughness) ** 2, 1.0e-6)
    alpha2 = alpha * alpha
    denominator = _PI * (cosine * cosine * (alpha2 - 1.0) + 1.0) ** 2
    return max(alpha2 / max(denominator, _EPS), _EPS)


def smith_g1(n_dot_v: float, roughness: float) -> float:
    cosine = float(np.clip(n_dot_v, 0.0, 1.0))
    alpha = max(float(roughness) ** 2, 1.0e-6)
    alpha2 = alpha * alpha
    return max(2.0 * cosine / max(cosine + math.sqrt(alpha2 + (1.0 - alpha2) * cosine * cosine), _EPS), _EPS)


def _zero_factors(material: Material) -> LogMaterialFactors:
    zero = v02.LogPos.zero()
    one = v02.LogPos.from_value(1.0)
    return LogMaterialFactors(material.log_base_color(), tuple(zero for _ in material.base_color), zero, zero, one)


@dataclass(frozen=True, slots=True)
class Material:
    """A small PBR material with a typed positive-log factor path."""

    base_color: Any = (0.8, 0.8, 0.8)
    roughness: float = 0.5
    metallic: float = 0.0
    specular: float = 0.5
    emission: Any = (0.0, 0.0, 0.0)
    name: str = "material"
    model: str = "microfacet"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    base_color_source: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from spf.renderer.assets import SpectralTexture

        source: Any = None
        supplied_base = self.base_color
        if isinstance(supplied_base, SpectralTexture):
            source = supplied_base
        elif isinstance(supplied_base, Field) and isinstance(supplied_base.representation.payload, SpectralTexture):
            if supplied_base.coherence is Coherence.COHERENT_COMPLEX:
                raise TypeError("coherent texture fields require explicit intensity conversion before material use")
            source = supplied_base
        elif isinstance(supplied_base, Field):
            block = _block_payload(supplied_base)
            if block is not None:
                if supplied_base.coherence is not Coherence.INCOHERENT_POSITIVE:
                    raise TypeError("block material colors require an incoherent positive Field")
                source = supplied_base
        elif _block_payload(supplied_base) is not None:
            source = supplied_base
        if source is not None:
            if isinstance(source, SpectralTexture):
                if not source.real:
                    raise TypeError("material base-color textures must be real-valued")
            elif (
                isinstance(source, Field)
                and isinstance(source.representation.payload, SpectralTexture)
                and not source.representation.payload.real
            ):
                raise TypeError("material base-color textures must be real-valued")
            base = np.ones(3, dtype=np.float64)
        else:
            base = as_native_color(supplied_base, name="base_color", allow_negative=False)
        emission = as_native_color(self.emission, name="emission", allow_negative=False)
        roughness = float(self.roughness)
        metallic = float(self.metallic)
        specular = float(self.specular)
        if not math.isfinite(roughness) or not 0.0 <= roughness <= 1.0:
            raise ValueError("roughness must be finite and in [0, 1]")
        if not math.isfinite(metallic) or not 0.0 <= metallic <= 1.0:
            raise ValueError("metallic must be finite and in [0, 1]")
        if not math.isfinite(specular) or not 0.0 <= specular <= 1.0:
            raise ValueError("specular must be finite and in [0, 1]")
        object.__setattr__(self, "base_color", tuple(float(value) for value in base))
        object.__setattr__(self, "base_color_source", source)
        object.__setattr__(self, "emission", tuple(float(value) for value in emission))
        object.__setattr__(self, "roughness", roughness)
        object.__setattr__(self, "metallic", metallic)
        object.__setattr__(self, "specular", specular)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "model", str(self.model).lower().replace("-", "_"))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def pbr(
        cls,
        base_color: Any = (0.8, 0.8, 0.8),
        *,
        roughness: float = 0.5,
        metallic: float = 0.0,
        specular: float = 0.5,
        emission: Any = (0.0, 0.0, 0.0),
        name: str = "material",
        model: str = "microfacet",
        metadata: Mapping[str, Any] | None = None,
    ) -> Material:
        return cls(
            base_color=base_color,
            roughness=roughness,
            metallic=metallic,
            specular=specular,
            emission=emission,
            name=name,
            model=model,
            metadata={} if metadata is None else metadata,
        )

    @property
    def base_color_array(self) -> np.ndarray:
        return np.asarray(self.base_color, dtype=np.float64)

    @property
    def emission_array(self) -> np.ndarray:
        return np.asarray(self.emission, dtype=np.float64)

    @property
    def baseColor(self) -> tuple[float, ...]:  # noqa: N802 - shader notation
        return tuple(float(value) for value in self.base_color)

    @property
    def is_textured(self) -> bool:
        return self.base_color_source is not None

    def sample_base_color(
        self,
        uv: Any,
        *,
        covariance: Any | None = None,
        cache: MutableMapping[int, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Evaluate a scalar spectral texture as grayscale base colour."""

        if self.base_color_source is None:
            return self.base_color_array
        source = self.base_color_source
        block = _closed_block_payload(source, name="material color")
        if block is not None:
            block_view, owner = block
            from spf.renderer.volume import _project_block_real, _sample_block_density

            value_kind = None if owner is None else str(owner.value_kind)
            shape = tuple(int(value) for value in block_view.shape)
            coordinates = np.asarray(uv, dtype=np.float64)
            if coordinates.shape != (2,) or not np.all(np.isfinite(coordinates)):
                raise ValueError("block material UV must be a finite two-vector")
            if covariance is not None:
                raise TypeError(
                    "block material sampling has no addressed footprint consumer; "
                    "materialize or bind a texture for covariance filtering"
                )
            support_mask: np.ndarray | None = None
            if owner is not None:
                support_kind = str(getattr(owner.support.kind, "value", owner.support.kind)).lower()
                if support_kind == "empty":
                    return np.zeros(3, dtype=np.float64)
                if len(shape) > 1 and owner.chart.dimension != 2:
                    raise TypeError("spatial block material sampling requires a two-dimensional coordinate chart")
                if owner.chart.dimension == 2:
                    matrix = np.asarray(owner.chart.basis, dtype=np.float64) @ np.diag(
                        np.asarray(owner.chart.scale, dtype=np.float64)
                        * np.asarray(owner.chart.sample_spacing, dtype=np.float64)
                    )
                    if np.linalg.matrix_rank(matrix) != 2:
                        raise ValueError("material block chart must be invertible")
                    local = (coordinates - np.asarray(owner.chart.origin, dtype=np.float64)) @ np.linalg.inv(matrix).T
                else:
                    local = coordinates
            else:
                local = coordinates
            output_count = 3 if len(shape) == 3 and shape[-1] == 3 else shape[0] if len(shape) == 1 else 1
            query_points = _material_block_query_points(owner, coordinates, shape, name="material color")
            if owner is not None and owner.support.region is not None:
                support_region = owner.support.region
                if owner.chart.dimension == 2:
                    supported = support_region.contains(coordinates.tolist())
                    support_mask = np.full(output_count, supported, dtype=bool)
                elif support_region.dimension != owner.chart.dimension:
                    raise TypeError("material block support region does not match its Field chart")
                else:
                    support_mask = np.asarray(
                        [support_region.contains(point.tolist()) for point in query_points], dtype=bool
                    )
            # A raw block is addressed with normalized UV coordinates.  An
            # owning Field chart already maps physical coordinates directly to
            # lattice indices; applying the UV extent a second time would
            # distort asymmetric grids (and silently swap their axes).
            lattice_points = owner is not None
            residual = None if owner is None else owner.representation.residual
            sampled: Any
            if len(shape) == 0:
                addresses = np.asarray([0], dtype=np.int64)
                sampled = _project_block_real(block_view, addresses, value_kind=value_kind)
                if residual is not None:
                    sampled += _material_block_residual_values(
                        residual,
                        addresses,
                        shape,
                        query_points,
                        count=1,
                        name="material color",
                    )
            elif len(shape) == 1:
                if shape[0] not in {1, 3}:
                    raise ValueError("one-dimensional block material colors must contain one or three values")
                addresses = np.arange(shape[0], dtype=np.int64)
                sampled = _project_block_real(block_view, addresses, value_kind=value_kind)
                if residual is not None:
                    sampled += _material_block_residual_values(
                        residual,
                        addresses,
                        shape,
                        query_points,
                        count=shape[0],
                        name="material color",
                    )
            elif len(shape) == 2:
                point = (
                    (local[0], local[1])
                    if lattice_points
                    else (local[1] * max(shape[0] - 1, 0), local[0] * max(shape[1] - 1, 0))
                )
                sampled = _sample_block_density(
                    block_view,
                    point,
                    value_kind=value_kind,
                    boundary="clamp",
                    residual=residual,
                    residual_points=coordinates if lattice_points else None,
                )
            elif len(shape) == 3 and shape[-1] == 3:
                points = np.asarray(
                    [
                        (
                            local[0] if lattice_points else local[1] * max(shape[0] - 1, 0),
                            local[1] if lattice_points else local[0] * max(shape[1] - 1, 0),
                            float(channel),
                        )
                        for channel in range(3)
                    ],
                    dtype=np.float64,
                )
                lattice_residual, direct_residual = (
                    _material_block_residual_parts(residual) if residual is not None else (None, None)
                )
                sampled = _sample_block_density(
                    block_view,
                    points,
                    value_kind=value_kind,
                    boundary="clamp",
                    residual=lattice_residual,
                )
                if direct_residual is not None:
                    direct_values = _material_block_residual_values(
                        direct_residual,
                        np.arange(3, dtype=np.int64),
                        shape,
                        query_points,
                        count=3,
                        name="material color",
                    )
                    sampled = np.asarray(sampled, dtype=np.float64).reshape(-1) + direct_values
            else:
                raise ValueError("block material colors must be one-, two-, or RGB-channel three-dimensional")
            sampled = np.asarray(sampled, dtype=np.float64).reshape(-1)
            if support_mask is not None:
                sampled = np.where(support_mask, sampled, 0.0)
        else:
            texture = source.representation.payload if isinstance(source, Field) else source
            sampled = texture.sample(uv, covariance=covariance, cache=cache)
        return as_native_color(sampled, name="sampled base_color", allow_negative=False)

    def sampled(
        self,
        uv: Any,
        *,
        covariance: Any | None = None,
        cache: MutableMapping[int, np.ndarray] | None = None,
    ) -> Material:
        """Return a constant material specialization for one texture sample."""

        if not self.is_textured:
            return self
        return replace(self, base_color=tuple(self.sample_base_color(uv, covariance=covariance, cache=cache)))

    def log_base_color(self) -> tuple[v02.LogPos, ...]:
        return tuple(as_log_pos(value, name="base_color") for value in self.base_color)

    @property
    def base_color_log(self) -> tuple[v02.LogPos, ...]:
        return self.log_base_color()

    def log_factors(self, normal: Any, view: Any, light: Any, *, model: str | None = None) -> LogMaterialFactors:
        """Evaluate the positive factors for one light direction.

        ``light`` is the direction from the shaded point toward the light.
        ``model='lambert'`` is provided as a parity oracle; the default is the
        compact GGX-style positive chain from the v0.2 design.
        """

        n = _unit(normal, name="normal")
        view_direction = _unit(view, name="view")
        light_direction = _unit(light, name="light")
        n_dot_l = _clamped_dot(n, light_direction)
        n_dot_v = _clamped_dot(n, view_direction)
        selected_model = self.model if model is None else str(model).lower().replace("-", "_")
        if n_dot_l <= 0.0 or n_dot_v <= 0.0:
            return _zero_factors(self)
        if selected_model in {"lambert", "diffuse"}:
            return LogMaterialFactors(
                self.log_base_color(),
                tuple(v02.LogPos.from_value(1.0) for _ in self.base_color),
                v02.LogPos.from_value(1.0),
                v02.LogPos.from_value(1.0),
                v02.LogPos.from_value(_PI),
            )
        if selected_model not in {"microfacet", "ggx", "pbr"}:
            raise ValueError(f"unsupported material model {selected_model!r}")
        half = view_direction + light_direction
        half_norm = float(np.linalg.norm(half))
        if half_norm <= _EPS:
            return _zero_factors(self)
        half /= half_norm
        n_dot_h = _clamped_dot(n, half)
        f0 = 0.04 * self.specular * (1.0 - self.metallic) + self.base_color_array * self.metallic
        fresnel = tuple(
            v02.LogPos.from_value(fresnel_schlick(float(n_dot_hv), float(channel)))
            for n_dot_hv, channel in zip((float(np.dot(view_direction, half)),) * len(f0), f0, strict=True)
        )
        distribution = v02.LogPos.from_value(ggx_distribution(n_dot_h, self.roughness))
        geometry = v02.LogPos.from_value(smith_g1(n_dot_l, self.roughness) * smith_g1(n_dot_v, self.roughness))
        denominator = v02.LogPos.from_value(max(4.0 * n_dot_l * n_dot_v, _EPS))
        return LogMaterialFactors(self.log_base_color(), fresnel, distribution, geometry, denominator)

    def log_brdf(self, normal: Any, view: Any, light: Any, *, model: str | None = None) -> tuple[v02.LogPos, ...]:
        return self.log_factors(normal, view, light, model=model).value

    def brdf_plane(self, normal: Any, view: Any, light: Any, *, model: str | None = None) -> Any:
        """BRDF factors as a ``LogPosPlane`` of shape ``(channels,)``."""

        return self.log_factors(normal, view, light, model=model).plane

    def evaluate_brdf(
        self,
        normal: Any,
        view: Any,
        light: Any,
        *,
        output: str = "native",
        model: str | None = None,
    ) -> tuple[v02.LogPos, ...] | tuple[float, ...]:
        factors = self.log_factors(normal, view, light, model=model)
        normalized = str(output).lower().replace("-", "_")
        if normalized in {"log", "log_pos", "typed"}:
            return factors.value
        if normalized not in {"native", "float", "float64"}:
            raise ValueError(f"unsupported BRDF output {output!r}")
        return factors.values

    brdf = evaluate_brdf

    def as_field(self, *, id: str | None = None) -> Field:
        """Expose base colour as an explicit positive radiance field."""

        if self.base_color_source is not None:
            from spf.renderer.assets import SpectralTexture

            if isinstance(self.base_color_source, Field):
                return replace(
                    self.base_color_source,
                    id=id or self.base_color_source.id,
                    semantic=Semantic.ALBEDO,
                )
            block = _block_payload(self.base_color_source)
            if block is not None:
                from spf.renderer.carriers import carrier_shape

                shape = carrier_shape(block[0])
                dimension = len(shape)
                return Field.from_value(
                    block[0],
                    id=id or self.name,
                    value_kind=FieldKind.LOG_POS,
                    semantic=Semantic.ALBEDO,
                    chart=CoordinateChart.identity(dimension),
                )
            texture = (
                self.base_color_source.representation.payload
                if isinstance(self.base_color_source, Field)
                else self.base_color_source
            )
            if isinstance(texture, SpectralTexture):
                return Field.from_texture(
                    texture,
                    id=id or self.name,
                    value_kind=FieldKind.LOG_POS,
                    semantic=Semantic.ALBEDO,
                )

        return Field(
            id=id or self.name,
            domain=Domain.SPATIAL_2D,
            semantic=Semantic.ALBEDO,
            coherence=Coherence.INCOHERENT_POSITIVE,
            value_kind=FieldKind.LOG_POS,
            representation=FieldRepresentation(
                RepresentationKind.DENSE_TYPED,
                np.asarray(self.log_base_color(), dtype=object),
            ),
            chart=CoordinateChart.identity(2),
        )


def evaluate_brdf(
    material: Material,
    normal: Any,
    view: Any,
    light: Any,
    *,
    output: str = "native",
    model: str | None = None,
) -> tuple[v02.LogPos, ...] | tuple[float, ...]:
    return material.evaluate_brdf(normal, view, light, output=output, model=model)


def brdf_log(
    material: Material, normal: Any, view: Any, light: Any, *, model: str | None = None
) -> tuple[v02.LogPos, ...]:
    return material.log_brdf(normal, view, light, model=model)


def material_factor_field(material: Material, *, id: str | None = None) -> Field:
    return material.as_field(id=id)


__all__ = [
    "LogMaterialFactors",
    "Material",
    "SignedColor",
    "apply_color_matrix",
    "as_log_pos",
    "as_log_pos_array",
    "as_log_pos_plane",
    "as_native_color",
    "block_words",
    "brdf_log",
    "coherent_intensity",
    "decode_log_pos",
    "evaluate_brdf",
    "explicit_signed_color",
    "fresnel_schlick",
    "ggx_distribution",
    "log_product",
    "log_product_plane",
    "log_quotient",
    "log_sum",
    "log_sum_plane",
    "material_factor_field",
    "positive_to_signed_color",
    "signed_color_values",
    "smith_g1",
    "to_signed_color",
]
