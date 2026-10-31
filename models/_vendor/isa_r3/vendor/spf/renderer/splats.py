"""Reference/CPU Gaussian-splat and visibility pipelines for SFR v0.2.

The scene/control part of a splat renderer is deliberately conventional:
camera projection, covariance projection, tile binning, depth ordering, and
primitive IDs are ordinary floating point/integer data.  Once a splat has
been assigned to a screen tile, its opacity is represented as a quadratic
``log2(alpha)`` field.  The SPF path then uses the v0.2 positive-log
complement and ordered affine transport primitives for compositing.

This module is intentionally self-contained so it can be used before a full
scene graph or native backend exists.  ``rasterize_splats_reference`` is the
float64 oracle; ``rasterize_splats_spf`` uses canonical ``LogPos`` and
``LogSigned`` values at its arithmetic boundaries and returns both decoded
pixels and the packed SPF words.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from spf.renderer import _splat_cuda
from spf.renderer.carriers import decode_words
from spf.renderer.report import RenderMetrics, RenderReport, _quality_data_range, psnr, relative_l2, ssim_global
from spf.tensor import v02

if TYPE_CHECKING:
    from spf.renderer.splat_terminal import TypedWordView


_LOG2_E = math.log2(math.e)
_BLOCK_INGRESS = frozenset({"addressed_block_to_float64", "addressed_block_words"})
_SPF_RELATIVE_QUANTIZATION = 2.0 ** (1.0 / (2.0 * v02.SPF32_L_SCALE)) - 1.0


def _readonly_array(value: Any, *, dtype: Any = np.float64, shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Make a defensive immutable numeric array for frozen records."""

    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"expected array shape {shape}, received {array.shape}")
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _finite_vector(value: Any, dimension: int, name: str) -> tuple[float, ...]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (dimension,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {dimension}-vector")
    return tuple(float(item) for item in array)


def _real_value(value: Any, name: str) -> float:
    if isinstance(value, (v02.LogPos, v02.LogSigned)):
        result = value.value
    elif isinstance(value, v02.SPF32Value):
        result = complex(decode_words(value.word)).real
        if abs(complex(decode_words(value.word)).imag) > 1.0e-12:
            raise ValueError(f"{name} must be real")
    else:
        result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class _SplatSource:
    """One compact producer crossing the splat scalar boundary.

    The splat geometry and transport kernels intentionally consume ordinary
    scalar values.  This record keeps the producer Field and its complete
    error contract beside that value so the conversion cannot look like a
    lossless, block-native arithmetic path.
    """

    values: np.ndarray
    ledger: v02.ErrorLedger = dataclass_field(default_factory=v02.ErrorLedger)
    field: Any = None
    ingress: str = "native_scalar"
    conversion_count: int = 0
    # Packed SPF32 words of the source when it was read without decoding
    # (``ingress == "addressed_block_words"``); ``values`` is then the
    # geometry envelope projected through the terminal magnitude ROM.
    words: np.ndarray | None = None


def _normalise_splat_field(value: Any, *, name: str) -> tuple[Any, v02.ErrorLedger]:
    """Normalize a Field's closed prefix before crossing the splat boundary."""

    from spf.renderer.model import Coherence, Field

    if not isinstance(value, Field):
        return value, v02.ErrorLedger()
    if value.coherence is Coherence.COHERENT_COMPLEX:
        raise TypeError(
            f"{name} is coherent-complex; call coherent_intensity() explicitly before splat ingestion"
        )
    from spf.renderer.normalization import complete_field_ledger, normalize_field

    normalized = normalize_field(value).field
    if normalized.operations:
        pending = ", ".join(operation.op for operation in normalized.operations)
        raise TypeError(f"{name} has unsupported deferred field operations at the splat boundary: {pending}")
    return normalized, complete_field_ledger(value)


def _splat_block_support_mask(owner: Any, points: np.ndarray, *, name: str) -> np.ndarray:
    """Apply the owning Field support at compact addressed coordinates."""

    if owner is None:
        return np.ones(points.shape[0], dtype=bool)
    support = owner.support
    kind = str(getattr(support.kind, "value", support.kind)).lower().replace("-", "_")
    if kind == "empty":
        return np.zeros(points.shape[0], dtype=bool)
    if kind in {"full", "unknown"}:
        return np.ones(points.shape[0], dtype=bool)
    if kind == "aabb" and support.region is not None:
        return np.asarray(
            [support.region.contains(point.tolist()) for point in points],
            dtype=bool,
        )
    raise TypeError(
        f"{name} compact block support {kind!r} has no scalar addressed splat contract"
    )


def _splat_block_points(owner: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    """Map compact block indices through the Field chart for residuals/support."""

    dimension = len(shape)
    if owner is None:
        chart = None
    else:
        chart = owner.chart
        if chart.dimension not in {0, dimension}:
            raise TypeError(
                f"{name} chart dimension {chart.dimension} does not match compact block rank {dimension}"
            )
    if not shape:
        # A rank-zero TensorBlock is one addressed scalar.  ``unravel_index``
        # returns an empty tuple for it, so make the leading sample axis
        # explicit for support/residual consumers.
        return np.empty((1, 0), dtype=np.float64)
    indices = np.asarray(np.unravel_index(np.arange(math.prod(shape), dtype=np.int64), shape), dtype=np.float64).T
    if chart is None or chart.dimension == 0:
        return indices
    return np.asarray([chart.coordinates(index) for index in indices], dtype=np.float64)


def _splat_residual_values(
    residual: Any,
    addresses: np.ndarray,
    shape: tuple[int, ...],
    points: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Read only compact residual addresses and require a real scalar result."""

    from spf.renderer.volume import _residual_lattice_or_direct

    lattice, direct = _residual_lattice_or_direct(residual, addresses, shape, points)
    result = np.zeros(addresses.shape, dtype=np.complex128)
    if lattice is not None:
        result += np.asarray(lattice, dtype=np.complex128).reshape(-1)
    if direct is not None:
        result += np.asarray(direct, dtype=np.complex128).reshape(-1)
    if np.any(np.abs(result.imag) > 1.0e-12 * np.maximum(1.0, np.abs(result.real))):
        raise TypeError(f"{name} residual is complex at the real splat boundary")
    return result.real


_SPLAT_ADDRESSABLE_KERNEL_OPS = frozenset(
    {
        "Bind",
        "Block",
        "Character",
        "ClosedFormIdentity",
        "ClosedFormZero",
        "DiagonalSPF",
        "KroneckerProduct",
        "LazyHadamard",
        "LazyOuterSum",
        "LowRankOuter",
        "MaterializedWords",
        "Permutation",
        "Roll",
        "StructuralApply",
    }
)


def _require_addressed_splat_kernel(block: Any, *, name: str) -> None:
    """Reject lazy matrix expressions whose query API reconstructs a plane."""

    from spf.renderer.block_filter import FilteredBlockView
    from spf.renderer.blocks import KernelBlockView

    if isinstance(block, FilteredBlockView):
        _require_addressed_splat_kernel(block.source, name=name)
        return
    if not isinstance(block, KernelBlockView):
        return
    def check(kernel: Any) -> None:
        operation = str(kernel.expr.op)
        if operation not in _SPLAT_ADDRESSABLE_KERNEL_OPS:
            raise TypeError(
                f"{name} matrix expression {operation!r} has no addressed splat query; "
                "an explicit compact materialization is required"
            )
        # These addressed operators delegate to child query methods. A lazy
        # contraction hidden below one of them can still materialize a plane.
        if operation in {"Bind", "LazyHadamard"}:
            for child in kernel.expr.args:
                check(child)
        elif operation == "KroneckerProduct":
            for child in kernel.expr.args[0]["factors"]:
                check(child)
        elif operation == "StructuralApply":
            check(kernel.expr.args[0]["vector"])

    check(block.kernel)


def _splat_source(
    value: Any,
    *,
    name: str,
    channels: int,
    allow_negative: bool,
) -> _SplatSource:
    """Decode a scalar/RGB source, using addressed block reads when available."""

    original = value
    normalized, ledger = _normalise_splat_field(value, name=name)
    from spf.renderer.model import Field

    owner = normalized if isinstance(normalized, Field) else None
    payload = normalized.representation.payload if owner is not None else normalized

    from spf.renderer.carriers import as_block_view

    block = as_block_view(payload)
    if block is not None:
        from spf.renderer.block_filter import FilteredBlockView

        if isinstance(block, FilteredBlockView) and not block.frequency_domain:
            raise TypeError(
                f"{name} spatial block operations require an explicit materialization boundary"
            )
        _require_addressed_splat_kernel(block, name=name)
        shape = tuple(int(item) for item in block.shape)
        count = int(math.prod(shape))
        if count not in {1, channels}:
            raise ValueError(f"{name} compact block must contain one or exactly {channels} addressed values")
        addresses = np.arange(count, dtype=np.int64)
        points = _splat_block_points(owner, shape, name=name)
        from spf.renderer.materials import block_words

        closed_words = block_words(normalized, name=name)
        if closed_words is not None:
            from spf.renderer.transport_terminal import display_values
            from spf.tensor.v02_planes import LogSignedPlane

            words = np.asarray(closed_words[0], dtype=np.uint32).reshape(-1)
            support = _splat_block_support_mask(owner, points, name=name)
            words = np.where(support, words, np.uint32(v02.SPF32_WORD_ZERO))
            signed = LogSignedPlane.from_words(words)
            if np.any(signed.is_nan):
                raise ValueError(f"{name} must contain finite values")
            if not allow_negative and np.any(signed.sign < 0):
                raise ValueError(f"{name} must be non-negative")
            if count != channels:
                words = np.repeat(words, channels)
                signed = LogSignedPlane.from_words(words)
            return _SplatSource(
                values=np.asarray(display_values(signed), dtype=np.float64),
                ledger=ledger,
                field=original if owner is not None else None,
                ingress="addressed_block_words",
                conversion_count=0,
                words=words,
            )
        from spf.renderer.volume import _project_block_real

        value_kind = None if owner is None else str(owner.value_kind)
        values = np.asarray(
            _project_block_real(block, addresses, value_kind=value_kind),
            dtype=np.float64,
        ).reshape(-1)
        if owner is not None and owner.representation.residual is not None:
            values += _splat_residual_values(
                owner.representation.residual,
                addresses,
                shape,
                points,
                name=name,
            )
        values = np.where(_splat_block_support_mask(owner, points, name=name), values, 0.0)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values")
        if not allow_negative and np.any(values < 0.0):
            raise ValueError(f"{name} must be non-negative")
        values = values if count == channels else np.repeat(values, channels)
        ingress = "addressed_block_to_float64"
        return _SplatSource(
            values=np.asarray(values, dtype=np.float64),
            ledger=ledger,
            field=original if owner is not None else None,
            ingress=ingress,
            conversion_count=count,
        )

    if owner is not None and owner.representation.residual is not None:
        raise TypeError(f"{name} dense Field residual has no compact splat address contract")
    from spf.renderer.materials import as_native_color

    values = as_native_color(payload, name=name, channels=channels, allow_negative=allow_negative)
    return _SplatSource(
        values=np.asarray(values, dtype=np.float64),
        ledger=ledger,
        field=original if owner is not None else None,
        ingress="native_scalar",
        conversion_count=0,
    )


def _normalise_projection(name: str) -> str:
    selected = str(name).strip().lower().replace("-", "_")
    if selected in {"ortho", "orthographic"}:
        return "orthographic"
    if selected in {"perspective", "persp"}:
        return "perspective"
    raise ValueError("camera projection must be 'orthographic' or 'perspective'")


@dataclass(frozen=True, slots=True)
class CameraProjection:
    """Projection of one world point into pixel row/column coordinates.

    ``xy`` uses the conventional ``(column, row)`` spelling.  ``pixel`` and
    ``jacobian`` use ``(row, column)`` order because that is the storage order
    used by image tiles and by :func:`v02.jet_forward_differences` here.
    The Jacobian is the derivative of pixel ``(row, column)`` with respect to
    world ``(x, y, z)``.
    """

    xy: tuple[float, float]
    depth: float
    jacobian: np.ndarray
    visible: bool = True
    camera_coordinates: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        xy = tuple(float(value) for value in self.xy)
        if len(xy) != 2 or not all(math.isfinite(value) for value in xy):
            raise ValueError("projection xy must contain two finite values")
        matrix = _readonly_array(self.jacobian, shape=(2, 3))
        if not np.all(np.isfinite(matrix)):
            raise ValueError("projection Jacobian must be finite")
        coords = tuple(float(value) for value in self.camera_coordinates)
        if len(coords) != 3 or not all(math.isfinite(value) for value in coords):
            raise ValueError("camera coordinates must be finite")
        object.__setattr__(self, "xy", xy)
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "jacobian", matrix)
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "camera_coordinates", coords)

    @property
    def pixel(self) -> tuple[float, float]:
        """Return ``(row, column)`` pixel coordinates."""

        return (self.xy[1], self.xy[0])

    @property
    def screen_rc(self) -> tuple[float, float]:
        return self.pixel

    @property
    def pixel_jacobian(self) -> np.ndarray:
        return self.jacobian


@dataclass(frozen=True, slots=True)
class Camera:
    """Minimal orthographic/perspective camera used by the reference path.

    For orthographic projection ``ortho_height`` is the world-space height of
    the image.  Pixels are square, so both axes use ``height/ortho_height``
    pixels per world unit.  ``fov_y`` is in radians for perspective mode.
    Camera depth is positive along ``target-position``; smaller positive depth
    is nearer and is therefore the stable tile sort key.
    """

    width: int
    height: int
    projection: str = "orthographic"
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target: tuple[float, float, float] = (0.0, 0.0, 1.0)
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    ortho_height: float = 2.0
    fov_y: float = math.radians(60.0)
    near: float = 1.0e-6
    far: float = math.inf

    def __post_init__(self) -> None:
        width, height = int(self.width), int(self.height)
        if width <= 0 or height <= 0:
            raise ValueError("camera width and height must be positive")
        projection = _normalise_projection(self.projection)
        position = _finite_vector(self.position, 3, "camera position")
        target = _finite_vector(self.target, 3, "camera target")
        up = _finite_vector(self.up, 3, "camera up")
        forward = np.asarray(target) - np.asarray(position)
        if np.linalg.norm(forward) == 0.0:
            raise ValueError("camera target must differ from position")
        if np.linalg.norm(np.cross(forward, up)) <= 1.0e-14:
            raise ValueError("camera up must not be parallel to the view direction")
        ortho_height = float(self.ortho_height)
        fov_y = float(self.fov_y)
        near = float(self.near)
        far = float(self.far)
        if not math.isfinite(ortho_height) or ortho_height <= 0.0:
            raise ValueError("ortho_height must be positive and finite")
        if not math.isfinite(fov_y) or not 0.0 < fov_y < math.pi:
            raise ValueError("fov_y must lie strictly between zero and pi")
        if not math.isfinite(near) or near < 0.0:
            raise ValueError("near must be finite and non-negative")
        if math.isnan(far) or far <= near:
            raise ValueError("far must exceed near")
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "projection", projection)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "ortho_height", ortho_height)
        object.__setattr__(self, "fov_y", fov_y)
        object.__setattr__(self, "near", near)
        object.__setattr__(self, "far", far)

    @classmethod
    def orthographic(
        cls,
        width: int,
        height: int,
        *,
        scale: float = 2.0,
        **kwargs: Any,
    ) -> Camera:
        """Construct an orthographic camera using ``scale`` as image height."""

        return cls(width, height, projection="orthographic", ortho_height=scale, **kwargs)

    @classmethod
    def perspective(
        cls,
        width: int,
        height: int,
        *,
        fov_y: float = math.radians(60.0),
        **kwargs: Any,
    ) -> Camera:
        return cls(width, height, projection="perspective", fov_y=fov_y, **kwargs)

    @property
    def projection_type(self) -> str:
        return self.projection

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return right, camera-up, and forward world-space unit vectors."""

        forward = np.asarray(self.target, dtype=np.float64) - np.asarray(self.position, dtype=np.float64)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(self.up, dtype=np.float64))
        right /= np.linalg.norm(right)
        camera_up = np.cross(right, forward)
        camera_up /= np.linalg.norm(camera_up)
        return right, camera_up, forward

    def project(self, point: Sequence[float]) -> CameraProjection:
        """Project a world-space point and return its pixel Jacobian."""

        world = np.asarray(point, dtype=np.float64).reshape(-1)
        if world.shape != (3,) or not np.all(np.isfinite(world)):
            raise ValueError("camera projection point must be a finite 3-vector")
        right, camera_up, forward = self.basis()
        delta = world - np.asarray(self.position, dtype=np.float64)
        x_cam = float(right @ delta)
        y_cam = float(camera_up @ delta)
        z_cam = float(forward @ delta)
        visible = self.near < z_cam <= self.far
        if self.projection == "orthographic":
            pixels_per_unit = self.height / self.ortho_height
            column = 0.5 * self.width + pixels_per_unit * x_cam
            row = 0.5 * self.height - pixels_per_unit * y_cam
            jacobian = np.stack((-pixels_per_unit * camera_up, pixels_per_unit * right), axis=0)
        else:
            focal = 0.5 * self.height / math.tan(0.5 * self.fov_y)
            # Keep a finite record for a culled point; its visibility flag is
            # authoritative and the enormous values are never binned.
            z_for_projection = math.copysign(max(abs(z_cam), self.near), z_cam or 1.0) if z_cam <= 0.0 else z_cam
            column = 0.5 * self.width + focal * x_cam / z_for_projection
            row = 0.5 * self.height - focal * y_cam / z_for_projection
            row_gradient = -focal / z_for_projection * camera_up + focal * y_cam / (z_for_projection**2) * forward
            column_gradient = focal / z_for_projection * right - focal * x_cam / (z_for_projection**2) * forward
            jacobian = np.stack((row_gradient, column_gradient), axis=0)
        return CameraProjection(
            (column, row),
            z_cam,
            jacobian,
            visible=visible,
            camera_coordinates=(x_cam, y_cam, z_cam),
        )

    project_point = project


def _opacity_code(source: _SplatSource, opacity: float) -> int:
    if source.words is not None:
        return int(v02.LogPos(int(v02.unpack_L(int(source.words.reshape(-1)[0])))).L)
    return int(v02.LogPos.from_value(float(opacity)).L)


def _color_codes(source: _SplatSource, color: tuple[float, ...]) -> tuple[tuple[int, int], ...]:
    if source.words is not None:
        from spf.tensor.v02_planes import LogSignedPlane

        signed = LogSignedPlane.from_words(source.words.reshape(-1))
        return tuple((int(code), int(sign)) for code, sign in zip(signed.L, signed.sign, strict=True))
    codes = []
    for value in color:
        typed = v02.LogSigned.from_value(float(value))
        codes.append((int(typed.L), int(typed.sign)))
    return tuple(codes)


@dataclass(frozen=True, slots=True)
class GaussianSplat:
    """A 3-D Gaussian opacity lobe and its RGB contribution."""

    mean: tuple[float, float, float]
    covariance: Any
    opacity: Any = 1.0
    color: Any = (1.0, 1.0, 1.0)
    primitive_id: int = 0
    material_id: Any = None
    local_coordinates: Any = None
    opacity_ledger: v02.ErrorLedger = dataclass_field(
        default_factory=v02.ErrorLedger,
        init=False,
        repr=False,
        compare=False,
    )
    color_ledger: v02.ErrorLedger = dataclass_field(
        default_factory=v02.ErrorLedger,
        init=False,
        repr=False,
        compare=False,
    )
    opacity_source: Any = dataclass_field(default=None, init=False, repr=False, compare=False)
    color_source: Any = dataclass_field(default=None, init=False, repr=False, compare=False)
    source_ingress: tuple[str, ...] = dataclass_field(default=(), init=False, repr=False, compare=False)
    source_conversion_count: int = dataclass_field(default=0, init=False, repr=False, compare=False)
    # Canonical SPF32 log codes of the sources: ``opacity_code`` is the LOG_POS
    # code, ``color_codes`` are ``(L, sign)`` pairs.  Read straight from block
    # words when the source is word-native, encoded once otherwise.
    opacity_code: int = dataclass_field(default=v02.SPF32_L_ZERO, init=False, repr=False, compare=False)
    color_codes: tuple[tuple[int, int], ...] = dataclass_field(default=(), init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        mean = _finite_vector(self.mean, 3, "splat mean")
        covariance = _readonly_array(self.covariance, shape=(3, 3))
        if not np.allclose(covariance, covariance.T, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("splat covariance must be symmetric")
        minimum = float(np.min(np.linalg.eigvalsh(covariance)))
        tolerance = 1.0e-10 * max(1.0, float(np.linalg.norm(covariance, ord=2)))
        if minimum < -tolerance:
            raise ValueError("splat covariance must be positive semidefinite")
        opacity_source = _splat_source(
            self.opacity,
            name="splat opacity",
            channels=1,
            allow_negative=False,
        )
        opacity_values = opacity_source.values.reshape(-1)
        if opacity_values.size != 1:
            raise ValueError("splat opacity must be a scalar")
        opacity = float(opacity_values[0])
        if opacity < 0.0 or opacity > 1.0 + 1.0e-12:
            raise ValueError("splat opacity must lie in [0, 1]")
        opacity = min(1.0, opacity)
        color_source = _splat_source(
            self.color,
            name="splat color",
            channels=3,
            allow_negative=True,
        )
        color_values = color_source.values.reshape(-1)
        if color_values.size != 3:
            raise ValueError("splat color must be an RGB three-vector or scalar")
        if not np.all(np.isfinite(color_values)):
            raise ValueError("splat color must be finite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "covariance", covariance)
        object.__setattr__(self, "opacity", opacity)
        object.__setattr__(self, "color", tuple(float(value) for value in color_values))
        object.__setattr__(self, "primitive_id", int(self.primitive_id))
        object.__setattr__(self, "opacity_ledger", opacity_source.ledger)
        object.__setattr__(self, "color_ledger", color_source.ledger)
        object.__setattr__(self, "opacity_source", opacity_source.field)
        object.__setattr__(self, "color_source", color_source.field)
        object.__setattr__(self, "source_ingress", (opacity_source.ingress, color_source.ingress))
        object.__setattr__(
            self,
            "source_conversion_count",
            int(opacity_source.conversion_count + color_source.conversion_count),
        )
        object.__setattr__(self, "opacity_code", _opacity_code(opacity_source, opacity))
        object.__setattr__(self, "color_codes", _color_codes(color_source, self.color))

    @property
    def words_native(self) -> bool:
        """True when opacity and colour were read as packed words (no decode)."""

        return "addressed_block_words" in self.source_ingress and self.source_conversion_count == 0

    @property
    def opacity_log(self) -> v02.LogPos:
        return v02.LogPos(self.opacity_code)

    @property
    def color_is_positive(self) -> bool:
        return all(value >= 0.0 for value in self.color)

    @property
    def color_kind(self) -> str:
        return "LOG_POS" if self.color_is_positive else "LOG_SIGNED"

    @property
    def max_color_abs(self) -> float:
        return max((abs(value) for value in self.color), default=0.0)

    @property
    def opacity_error_bound(self) -> float:
        return float(self.opacity_ledger.total_bound)

    @property
    def color_error_bound(self) -> float:
        return float(self.color_ledger.total_bound)

    @property
    def source_ledger(self) -> v02.ErrorLedger:
        return self.opacity_ledger.plus(self.color_ledger)

    @property
    def source_error_bound(self) -> float:
        return float(self.source_ledger.total_bound)

    @property
    def opacity_upper_bound(self) -> float:
        """Conservative opacity maximum for support/culling decisions."""

        error = self.opacity_error_bound
        if math.isinf(error):
            return 1.0
        return float(min(1.0, max(0.0, float(self.opacity) + error)))


def _quadratic_box_minimum(
    precision: np.ndarray,
    center: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Minimize ``(x-center)^T precision (x-center)`` over a 2-D box.

    Enumerating lower/free/upper active sets is small, deterministic, and
    gives a safe upper bound for Gaussian opacity.  It also handles a
    non-axis-aligned ellipse, where simply clamping the center is not exact.
    """

    candidates: list[np.ndarray] = []
    for states in product((-1, 0, 1), repeat=2):
        active = np.asarray([axis for axis, state in enumerate(states) if state], dtype=int)
        free = np.asarray([axis for axis, state in enumerate(states) if not state], dtype=int)
        point = np.empty(2, dtype=np.float64)
        if active.size:
            point[active] = np.asarray(
                [lower[axis] if states[axis] < 0 else upper[axis] for axis in active], dtype=np.float64
            )
        if free.size:
            if active.size:
                rhs = -precision[np.ix_(free, active)] @ (point[active] - center[active])
            else:
                rhs = np.zeros(free.size, dtype=np.float64)
            matrix = precision[np.ix_(free, free)]
            try:
                point[free] = center[free] + np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                point[free] = center[free] + np.linalg.pinv(matrix) @ rhs
            if np.any(point[free] < lower[free] - 1.0e-10) or np.any(point[free] > upper[free] + 1.0e-10):
                continue
            point[free] = np.clip(point[free], lower[free], upper[free])
        candidates.append(point)
    if not candidates:
        point = np.clip(center, lower, upper)
        candidates = [point]
    values = [float((point - center) @ precision @ (point - center)) for point in candidates]
    winner = int(np.argmin(values))
    return max(0.0, values[winner]), candidates[winner]


@dataclass(frozen=True, slots=True)
class ProjectedSplat:
    """Camera-projected splat geometry before tile-local jet lowering."""

    source_index: int
    primitive_id: int
    screen_rc: tuple[float, float]
    depth: float
    covariance_2d: np.ndarray
    precision_2d: np.ndarray
    opacity: float
    color: tuple[float, float, float]
    tile_bounds: tuple[int, int, int, int] | None
    visible: bool = True
    material_id: Any = None
    local_coordinates: Any = None
    opacity_ledger: v02.ErrorLedger = dataclass_field(
        default_factory=v02.ErrorLedger,
        repr=False,
        compare=False,
    )
    color_ledger: v02.ErrorLedger = dataclass_field(
        default_factory=v02.ErrorLedger,
        repr=False,
        compare=False,
    )
    opacity_source: Any = dataclass_field(default=None, repr=False, compare=False)
    color_source: Any = dataclass_field(default=None, repr=False, compare=False)
    source_ingress: tuple[str, ...] = dataclass_field(default=(), repr=False, compare=False)
    source_conversion_count: int = dataclass_field(default=0, repr=False, compare=False)
    opacity_code: int | None = dataclass_field(default=None, repr=False, compare=False)
    color_codes: tuple[tuple[int, int], ...] | None = dataclass_field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        screen = tuple(float(value) for value in self.screen_rc)
        if len(screen) != 2 or not all(math.isfinite(value) for value in screen):
            raise ValueError("projected splat screen_rc must be finite")
        covariance = _readonly_array(self.covariance_2d, shape=(2, 2))
        precision = _readonly_array(self.precision_2d, shape=(2, 2))
        if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(precision)):
            raise ValueError("projected covariance and precision must be finite")
        opacity = float(self.opacity)
        if not 0.0 <= opacity <= 1.0 + 1.0e-12:
            raise ValueError("projected opacity must lie in [0, 1]")
        bounds = None if self.tile_bounds is None else tuple(int(value) for value in self.tile_bounds)
        if bounds is not None and len(bounds) != 4:
            raise ValueError("projected tile bounds must be (row0, col0, row1, col1)")
        if not isinstance(self.opacity_ledger, v02.ErrorLedger) or not isinstance(self.color_ledger, v02.ErrorLedger):
            raise TypeError("projected splat source ledgers must be ErrorLedger values")
        object.__setattr__(self, "source_index", int(self.source_index))
        object.__setattr__(self, "primitive_id", int(self.primitive_id))
        object.__setattr__(self, "screen_rc", screen)
        object.__setattr__(self, "depth", float(self.depth))
        object.__setattr__(self, "covariance_2d", covariance)
        object.__setattr__(self, "precision_2d", precision)
        object.__setattr__(self, "opacity", min(1.0, opacity))
        object.__setattr__(self, "color", tuple(float(value) for value in self.color))
        object.__setattr__(self, "tile_bounds", bounds)
        object.__setattr__(self, "visible", bool(self.visible))
        object.__setattr__(self, "opacity_ledger", self.opacity_ledger)
        object.__setattr__(self, "color_ledger", self.color_ledger)
        object.__setattr__(self, "source_ingress", tuple(str(value) for value in self.source_ingress))
        count = int(self.source_conversion_count)
        if count < 0:
            raise ValueError("projected splat source conversion count must be non-negative")
        object.__setattr__(self, "source_conversion_count", count)
        if self.opacity_code is None:
            object.__setattr__(self, "opacity_code", int(v02.LogPos.from_value(min(1.0, opacity)).L))
        if self.color_codes is None:
            object.__setattr__(
                self,
                "color_codes",
                tuple((int(v02.LogSigned.from_value(float(v)).L), int(v02.LogSigned.from_value(float(v)).sign))
                      for v in self.color),
            )

    @property
    def words_native(self) -> bool:
        return "addressed_block_words" in self.source_ingress and self.source_conversion_count == 0

    @property
    def mean_2d(self) -> tuple[float, float]:
        return self.screen_rc

    @property
    def center(self) -> tuple[float, float]:
        return self.screen_rc

    @property
    def covariance(self) -> np.ndarray:
        return self.covariance_2d

    @property
    def precision(self) -> np.ndarray:
        return self.precision_2d

    @property
    def opacity_log(self) -> v02.LogPos:
        return v02.LogPos(int(cast(int, self.opacity_code)))

    @property
    def max_color_abs(self) -> float:
        return max((abs(value) for value in self.color), default=0.0)

    @property
    def opacity_error_bound(self) -> float:
        return float(self.opacity_ledger.total_bound)

    @property
    def color_error_bound(self) -> float:
        return float(self.color_ledger.total_bound)

    @property
    def source_ledger(self) -> v02.ErrorLedger:
        return self.opacity_ledger.plus(self.color_ledger)

    @property
    def source_error_bound(self) -> float:
        return float(self.source_ledger.total_bound)

    @property
    def max_color_abs_bound(self) -> float:
        error = self.color_error_bound
        return math.inf if math.isinf(error) else self.max_color_abs + error

    @property
    def opacity_upper_bound(self) -> float:
        error = self.opacity_error_bound
        return 1.0 if math.isinf(error) else min(1.0, max(0.0, self.opacity + error))

    def log2_alpha(self, point_rc: Sequence[float], *, quantized_opacity: bool = False) -> float:
        point = np.asarray(point_rc, dtype=np.float64).reshape(-1)
        if point.shape != (2,):
            raise ValueError("screen point must be a two-vector")
        if self.opacity == 0.0:
            return -math.inf
        # The quantized constant term is the LOG_POS code itself: log2 of a
        # canonical SPF32 magnitude is exactly ``L / 256`` (no exp2/log2 pair).
        log2_alpha0 = self.opacity_log.L / v02.SPF32_L_SCALE if quantized_opacity else math.log2(self.opacity)
        delta = point - np.asarray(self.screen_rc)
        return float(log2_alpha0 - 0.5 * float(delta @ self.precision_2d @ delta) * _LOG2_E)

    def alpha_bound(self, bounds: Sequence[float]) -> float:
        """Return a certified maximum opacity over a pixel rectangle."""

        if self.opacity_upper_bound == 0.0:
            return 0.0
        values = np.asarray(bounds, dtype=np.float64).reshape(-1)
        if values.shape != (4,):
            raise ValueError("tile bounds must be (row0, col0, row1, col1)")
        row0, col0, row1, col1 = values.tolist()
        lower = np.asarray([row0, col0], dtype=np.float64)
        upper = np.asarray([row1, col1], dtype=np.float64)
        if np.any(upper < lower):
            raise ValueError("tile upper bounds must not precede lower bounds")
        quadratic, _ = _quadratic_box_minimum(
            self.precision_2d,
            np.asarray(self.screen_rc, dtype=np.float64),
            lower,
            upper,
        )
        try:
            return float(math.exp(-0.5 * quadratic) * self.opacity_upper_bound)
        except OverflowError:
            return 0.0

    def log2_alpha_bound(self, bounds: Sequence[float]) -> float:
        alpha = self.alpha_bound(bounds)
        return -math.inf if alpha == 0.0 else math.log2(alpha)


def _pixel_bounds(
    projected: CameraProjection,
    opacity: float,
    image_shape: tuple[int, int],
    min_alpha: float,
) -> tuple[int, int, int, int] | None:
    height, width = image_shape
    if opacity <= 0.0 or not projected.visible:
        return None
    if min_alpha > opacity:
        return None
    if min_alpha <= 0.0:
        return (0, 0, height, width)
    radius = math.sqrt(max(0.0, -2.0 * math.log(min_alpha / opacity)))
    covariance = projected.jacobian @ projected.jacobian.T
    # The camera Jacobian is not the splat covariance.  This helper is only a
    # conservative pixel extent for a unit covariance when called directly;
    # the actual extent is computed in ``project_splat`` below.
    extent = radius * np.sqrt(np.maximum(np.diag(covariance), 0.0))
    row0 = max(0, math.floor(projected.pixel[0] - extent[0]))
    col0 = max(0, math.floor(projected.pixel[1] - extent[1]))
    row1 = min(height, math.ceil(projected.pixel[0] + extent[0]) + 1)
    col1 = min(width, math.ceil(projected.pixel[1] + extent[1]) + 1)
    if row0 >= row1 or col0 >= col1:
        return None
    return row0, col0, row1, col1


def project_splat(
    splat: GaussianSplat,
    camera: Camera,
    *,
    source_index: int = 0,
    image_shape: tuple[int, int] | None = None,
    min_alpha: float = 1.0e-4,
    covariance_epsilon: float = 1.0e-8,
) -> ProjectedSplat:
    """Project one splat and conservatively compute its pixel support."""

    shape = (camera.height, camera.width) if image_shape is None else tuple(int(value) for value in image_shape)
    if len(shape) != 2 or any(value <= 0 for value in shape):
        raise ValueError("image_shape must be a positive (height, width) pair")
    if shape != (camera.height, camera.width):
        raise ValueError("image_shape must match camera height and width")
    threshold = float(min_alpha)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("min_alpha must lie in [0, 1]")
    epsilon = float(covariance_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("covariance_epsilon must be positive and finite")
    projection = camera.project(splat.mean)
    covariance_2d = projection.jacobian @ np.asarray(splat.covariance) @ projection.jacobian.T
    covariance_2d = 0.5 * (covariance_2d + covariance_2d.T)
    covariance_2d += epsilon * np.eye(2)
    try:
        precision_2d = np.linalg.inv(covariance_2d)
    except np.linalg.LinAlgError as exc:
        raise ValueError("projected splat covariance is singular") from exc
    bounds: tuple[int, int, int, int] | None
    support_opacity = splat.opacity_upper_bound
    if (
        not projection.visible
        or splat.opacity <= 0.0
        or support_opacity <= 0.0
        or threshold > support_opacity
    ):
        # An uncertain zero nominal opacity remains in the source ledger, but
        # has no finite nominal log-alpha jet to lower into a tile entry.
        bounds = None
    elif threshold <= 0.0:
        bounds = (0, 0, shape[0], shape[1])
    else:
        radius = math.sqrt(max(0.0, -2.0 * math.log(threshold / support_opacity)))
        extent = radius * np.sqrt(np.maximum(np.diag(covariance_2d), 0.0))
        row0 = max(0, math.floor(projection.pixel[0] - extent[0]))
        col0 = max(0, math.floor(projection.pixel[1] - extent[1]))
        row1 = min(shape[0], math.ceil(projection.pixel[0] + extent[0]) + 1)
        col1 = min(shape[1], math.ceil(projection.pixel[1] + extent[1]) + 1)
        bounds = None if row0 >= row1 or col0 >= col1 else (row0, col0, row1, col1)
    return ProjectedSplat(
        source_index=source_index,
        primitive_id=splat.primitive_id,
        screen_rc=projection.pixel,
        depth=projection.depth,
        covariance_2d=covariance_2d,
        precision_2d=precision_2d,
        opacity=splat.opacity,
        color=splat.color,
        tile_bounds=bounds,
        visible=projection.visible,
        material_id=splat.material_id,
        local_coordinates=splat.local_coordinates,
        opacity_ledger=splat.opacity_ledger,
        color_ledger=splat.color_ledger,
        opacity_source=splat.opacity_source,
        color_source=splat.color_source,
        source_ingress=splat.source_ingress,
        source_conversion_count=splat.source_conversion_count,
        opacity_code=splat.opacity_code,
        color_codes=splat.color_codes,
    )


def project_splats(
    splats: Iterable[GaussianSplat],
    camera: Camera,
    *,
    image_shape: tuple[int, int] | None = None,
    min_alpha: float = 1.0e-4,
    covariance_epsilon: float = 1.0e-8,
) -> tuple[ProjectedSplat, ...]:
    return tuple(
        project_splat(
            splat,
            camera,
            source_index=index,
            image_shape=image_shape,
            min_alpha=min_alpha,
            covariance_epsilon=covariance_epsilon,
        )
        for index, splat in enumerate(splats)
    )


@dataclass(frozen=True, slots=True)
class ProjectedSplatTile:
    """One projected splat lowered into one tile-local quadratic jet."""

    tile_index: tuple[int, int]
    bounds: tuple[int, int, int, int]
    projected: ProjectedSplat
    max_alpha: float
    log2_alpha_jet: v02.Jet2D
    spf_log_alpha_jet: v02.Jet2D

    @property
    def splat_index(self) -> int:
        return self.projected.source_index

    @property
    def source_index(self) -> int:
        return self.projected.source_index

    @property
    def depth(self) -> float:
        return self.projected.depth

    @property
    def alpha_bound(self) -> float:
        return self.max_alpha

    @property
    def tile_bounds(self) -> tuple[int, int, int, int]:
        return self.bounds

    def alpha_header(self, *, rounding: str = "nearest") -> tuple[Any, float]:
        """The tile's log-alpha jet as ``TensorBlockHeaderV2`` structure (ABI v23).

        Design §27.1 / §14.2: the quadratic log-opacity is block structure, not a
        renderer-private residual.  Returns ``(header, bound_ticks)`` from
        :func:`spf.tensor.v02.jet_to_tensor_block_header` with ``global_L_bias`` /
        ``band_exp`` / ``log_quadratic`` / ``log_mixed`` in whole SPF32 L ticks
        and the certified rounding bound over the tile extent (``0.0`` when the
        widened jet is header-exact).  ``SPFError`` propagates when a coefficient
        does not fit int16; consumers then keep the widened jet.
        """

        row0, col0, row1, col1 = self.bounds
        return v02.jet_to_tensor_block_header(self.spf_log_alpha_jet, (row1 - row0, col1 - col0), rounding=rounding)


def log2_alpha_jet(projected: ProjectedSplat, origin: Sequence[float]) -> v02.Jet2D:
    """Build the exact native-unit quadratic ``log2(alpha)`` jet.

    ``origin`` is in mathematical screen ``(x, y) == (column, row)`` order,
    matching the coordinate convention of :class:`v02.Jet2D`.  Image arrays
    remain ``(row, column)``; the tile materializer supplies the corresponding
    local coordinates when it emits rows.
    """

    origin_array = np.asarray(origin, dtype=np.float64).reshape(-1)
    if origin_array.shape != (2,) or not np.all(np.isfinite(origin_array)):
        raise ValueError("jet origin must be a finite two-vector")
    if projected.opacity <= 0.0:
        raise ValueError("a zero-opacity splat has no finite log-alpha jet")
    # The projected record stores image coordinates as (row, column), whereas
    # Jet2D's first variable is x and its second variable is y.  Reorder both
    # the center and precision matrix at this algebra boundary.
    center_xy = np.asarray((projected.screen_rc[1], projected.screen_rc[0]), dtype=np.float64)
    precision_xy = projected.precision_2d[np.ix_((1, 0), (1, 0))]
    delta = origin_array - center_xy
    coefficient = 0.5 * _LOG2_E
    value = math.log2(projected.opacity) - coefficient * float(delta @ precision_xy @ delta)
    gradient = -_LOG2_E * (precision_xy @ delta)
    hessian = -_LOG2_E * precision_xy
    return v02.Jet2D(
        value=value,
        grad_x=gradient[0],
        grad_y=gradient[1],
        hess_xx=hessian[0, 0],
        hess_xy=hessian[0, 1],
        hess_yy=hessian[1, 1],
        value_kind="native",
    )


def spf_log_alpha_jet(
    projected: ProjectedSplat,
    origin: Sequence[float],
    *,
    guard_L: int = 8,
) -> v02.Jet2D:
    """Build the same jet in widened SPF log-code ticks.

    ``Jet2D`` emits an SPF log code by dividing its coefficients by
    ``2**guard_L``.  One tick is therefore ``1/(256*2**guard_L)`` in
    log2(alpha), which postpones canonical SPF32 rounding until a pixel is
    emitted.
    """

    guard = int(guard_L)
    if guard < 0:
        raise ValueError("guard_L must be non-negative")
    base = log2_alpha_jet(projected, origin)
    # The SPF field starts from the canonical opacity code, while the native
    # jet above intentionally retains the caller's float64 source.  Replacing
    # only the constant term keeps the spatial quadratic exact and makes the
    # eventual alpha complement agree with the declared SPF source value.
    origin_array = np.asarray(origin, dtype=np.float64).reshape(-1)
    quantized_value = projected.log2_alpha((origin_array[1], origin_array[0]), quantized_opacity=True)
    scale = float(v02.SPF32_L_SCALE * (1 << guard))
    return v02.Jet2D(
        value=quantized_value * scale,
        grad_x=base.grad_x * scale,
        grad_y=base.grad_y * scale,
        hess_xx=base.hess_xx * scale,
        hess_xy=base.hess_xy * scale,
        hess_yy=base.hess_yy * scale,
        guard_L=guard,
        guard_T=guard,
        value_kind="log_pos",
    )


def make_log_alpha_jet(
    projected: ProjectedSplat,
    origin: Sequence[float],
    *,
    guard_L: int = 8,
    value_kind: str = "spf",
) -> v02.Jet2D:
    """Convenience constructor for either native or widened log-alpha jets."""

    selected = str(value_kind).lower().replace("-", "_")
    if selected in {"native", "log2", "float", "reference"}:
        return log2_alpha_jet(projected, origin)
    if selected in {"spf", "log", "log_pos", "widened"}:
        return spf_log_alpha_jet(projected, origin, guard_L=guard_L)
    raise ValueError("value_kind must be 'native' or 'spf'")


def _tile_entry(
    projected: ProjectedSplat,
    tile_index: tuple[int, int],
    bounds: tuple[int, int, int, int],
    *,
    guard_L: int,
) -> ProjectedSplatTile:
    # Jet2D uses (x, y)=(column, row), while tile bounds are (row0, col0,
    # row1, col1).
    origin = (float(bounds[1]), float(bounds[0]))
    return ProjectedSplatTile(
        tile_index=tile_index,
        bounds=bounds,
        projected=projected,
        max_alpha=projected.alpha_bound(bounds),
        log2_alpha_jet=log2_alpha_jet(projected, origin),
        spf_log_alpha_jet=spf_log_alpha_jet(projected, origin, guard_L=guard_L),
    )


@dataclass(frozen=True, slots=True)
class SplatTile:
    """Stable depth-sorted tile record."""

    index: tuple[int, int]
    bounds: tuple[int, int, int, int]
    entries: tuple[ProjectedSplatTile, ...]

    @property
    def tile_index(self) -> tuple[int, int]:
        return self.index

    @property
    def splats(self) -> tuple[ProjectedSplatTile, ...]:
        return self.entries

    @property
    def splat_indices(self) -> tuple[int, ...]:
        return tuple(entry.source_index for entry in self.entries)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.bounds[2] - self.bounds[0], self.bounds[3] - self.bounds[1])


@dataclass(frozen=True, slots=True)
class TileBins:
    """Deterministic row-major tile bins plus projected source records."""

    image_shape: tuple[int, int]
    tile_size: int
    projected: tuple[ProjectedSplat, ...]
    tiles: tuple[SplatTile, ...]
    min_alpha: float

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.image_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("TileBins image_shape must be positive (height, width)")
        size = int(self.tile_size)
        if size <= 0:
            raise ValueError("tile_size must be positive")
        object.__setattr__(self, "image_shape", shape)
        object.__setattr__(self, "tile_size", size)
        object.__setattr__(self, "projected", tuple(self.projected))
        object.__setattr__(self, "tiles", tuple(self.tiles))
        object.__setattr__(self, "min_alpha", float(self.min_alpha))

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def tile_map(self) -> dict[tuple[int, int], SplatTile]:
        return {tile.index: tile for tile in self.tiles}

    def tile(self, index: tuple[int, int]) -> SplatTile | None:
        key = tuple(int(value) for value in index)
        for tile in self.tiles:
            if tile.index == key:
                return tile
        return None

    def __iter__(self) -> Iterator[SplatTile]:
        return iter(self.tiles)


def bin_splats(
    splats: Iterable[GaussianSplat],
    camera: Camera,
    *,
    image_shape: tuple[int, int] | None = None,
    tile_size: int = 16,
    min_alpha: float = 1.0e-4,
    covariance_epsilon: float = 1.0e-8,
    guard_L: int = 8,
) -> TileBins:
    """Project, conservatively bin, and stably depth-sort Gaussian splats."""

    shape = (camera.height, camera.width) if image_shape is None else tuple(int(value) for value in image_shape)
    if shape != (camera.height, camera.width):
        raise ValueError("image_shape must match camera height and width")
    size = int(tile_size)
    if size <= 0:
        raise ValueError("tile_size must be positive")
    threshold = float(min_alpha)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("min_alpha must lie in [0, 1]")
    projected = project_splats(
        splats,
        camera,
        image_shape=shape,
        min_alpha=threshold,
        covariance_epsilon=covariance_epsilon,
    )
    grouped: dict[tuple[int, int], list[ProjectedSplatTile]] = {}
    for record in projected:
        if not record.visible or record.tile_bounds is None:
            continue
        row0, col0, row1, col1 = record.tile_bounds
        tile_row0, tile_col0 = row0 // size, col0 // size
        tile_row1, tile_col1 = (row1 - 1) // size, (col1 - 1) // size
        for tile_row in range(tile_row0, tile_row1 + 1):
            for tile_col in range(tile_col0, tile_col1 + 1):
                bounds = (
                    tile_row * size,
                    tile_col * size,
                    min(shape[0], (tile_row + 1) * size),
                    min(shape[1], (tile_col + 1) * size),
                )
                entry = _tile_entry(record, (tile_row, tile_col), bounds, guard_L=guard_L)
                if threshold > 0.0 and entry.max_alpha < threshold:
                    # This is the only opacity cull.  ``alpha_bound`` is the
                    # maximum of the exact quadratic over the whole tile, so
                    # this test cannot remove a pixel whose opacity is above
                    # the threshold.
                    continue
                grouped.setdefault((tile_row, tile_col), []).append(entry)
    tiles = tuple(
        SplatTile(
            index=index,
            bounds=(
                index[0] * size,
                index[1] * size,
                min(shape[0], (index[0] + 1) * size),
                min(shape[1], (index[1] + 1) * size),
            ),
            entries=tuple(sorted(grouped[index], key=lambda item: (item.depth, item.source_index))),
        )
        for index in sorted(grouped)
    )
    return TileBins(shape, size, projected, tiles, threshold)


build_tile_bins = bin_splats
project_gaussian_splat = project_splat
ProjectedSplatRecord = ProjectedSplat
SplatTileRecord = SplatTile


def _logpos_from_wide(wide_value: float, guard_L: int) -> v02.LogPos:
    code = int(np.rint(float(wide_value) / float(1 << int(guard_L))))
    # alpha0 is validated to be <=1 and the quadratic is non-positive.  A
    # round-to-nearest tie at zero may produce a tiny positive code; clamp it
    # rather than asking logsub_pos to process an invalid opacity.
    return v02.LogPos(min(0, code))


def _typed_alpha_tile(entry: ProjectedSplatTile, counters: v02.PerformanceCounters) -> np.ndarray:
    row0, col0, row1, col1 = entry.bounds
    shape = (row1 - row0, col1 - col0)
    wide = v02.jet_forward_differences(
        entry.spf_log_alpha_jet,
        shape,
        # The jet coefficients are Taylor coefficients in coordinates local
        # to this tile's origin.  ``log2_alpha_jet`` evaluates at the supplied
        # screen origin and stores that as q=(0, 0), so the materializer must
        # advance local row/column offsets rather than apply the absolute
        # origin a second time.
        origin=(0.0, 0.0),
        counters=counters,
    )
    output = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        output[index] = _logpos_from_wide(
            float(np.asarray(wide[index]).item()),
            entry.spf_log_alpha_jet.guard_L,
        )
    return output


def _typed_channel_product(alpha: v02.LogPos, color: float) -> v02.LogPos | v02.LogSigned:
    if color >= 0.0:
        return v02.logpos_mul(alpha, v02.LogPos.from_value(color))
    magnitude = v02.logpos_mul(alpha, v02.LogPos.from_value(-color))
    if magnitude.is_zero:
        return v02.LogSigned.zero()
    if magnitude.is_nan:
        return v02.LogSigned.nan()
    return v02.LogSigned(magnitude.L, -1)


def _typed_background(value: float) -> v02.LogPos | v02.LogSigned:
    if value >= 0.0:
        return v02.LogPos.from_value(value)
    return v02.LogSigned.from_value(value)


def _decode_typed(value: v02.LogPos | v02.LogSigned) -> float:
    return float(value.value)


def _compose_segments(
    segments: Sequence[v02.Transport],
    *,
    reduction: str,
    counters: v02.PerformanceCounters,
) -> v02.Transport:
    if not segments:
        return v02.Transport(v02.LogPos(0), v02.LogPos.zero())
    selected = str(reduction).lower().replace("-", "_")
    if selected == "auto":
        selected = "tree" if len(segments) > 8 else "serial"
    if selected == "tree":
        return v02.transport_reduce_ordered(segments, counters=counters)
    if selected != "serial":
        raise ValueError("reduction must be 'serial', 'tree', or 'auto'")
    current = segments[0]
    for segment in segments[1:]:
        current = v02.transport_compose(current, segment, counters=counters)
    return current


@dataclass(frozen=True, slots=True)
class EarlyTerminationCertificate:
    """Per-pixel proof record for a skipped back tail."""

    tile_index: tuple[int, int]
    pixel: tuple[int, int]
    processed: int
    remaining: int
    terminated: bool
    remaining_bound: float
    error_bound: float

    @property
    def valid(self) -> bool:
        return self.terminated and math.isfinite(self.error_bound) and self.error_bound >= self.remaining_bound


@dataclass(frozen=True, slots=True)
class SplatRenderResult:
    """Decoded render plus typed/paked values, control outputs, and report."""

    color: np.ndarray
    transmittance: np.ndarray
    alpha: np.ndarray
    depth: np.ndarray
    primitive_id: np.ndarray
    coverage: np.ndarray
    report: RenderReport
    tiles: TileBins
    color_typed: np.ndarray | TypedWordView | None = None
    transmittance_typed: np.ndarray | TypedWordView | None = None
    color_words: np.ndarray | None = None
    transmittance_words: np.ndarray | None = None
    certificates: Sequence[EarlyTerminationCertificate] = ()

    @property
    def image(self) -> np.ndarray:
        return self.color

    @property
    def values(self) -> np.ndarray:
        return self.color

    @property
    def spf(self) -> np.ndarray | None:
        return self.color_words

    @property
    def early_termination(self) -> Sequence[EarlyTerminationCertificate]:
        return self.certificates

    @property
    def tile_certificates(self) -> Sequence[EarlyTerminationCertificate]:
        return self.certificates


def _empty_outputs(shape: tuple[int, int], channels: int = 3) -> tuple[np.ndarray, ...]:
    color = np.zeros((*shape, channels), dtype=np.float64)
    transmittance = np.ones(shape, dtype=np.float64)
    alpha = np.zeros(shape, dtype=np.float64)
    depth = np.full(shape, math.inf, dtype=np.float64)
    primitive_id = np.full(shape, -1, dtype=np.int64)
    coverage = np.zeros(shape, dtype=np.float64)
    return color, transmittance, alpha, depth, primitive_id, coverage


def _reference_core(
    bins: TileBins,
    *,
    background: tuple[float, float, float],
    opaque_depth: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = bins.image_shape
    color, transmittance, alpha_out, depth_out, primitive_out, coverage = _empty_outputs(shape)
    background_array = np.asarray(background, dtype=np.float64)
    color[...] = background_array
    for tile in bins.tiles:
        row0, col0, row1, col1 = tile.bounds
        rows = np.arange(row0, row1, dtype=np.float64)[:, None]
        columns = np.arange(col0, col1, dtype=np.float64)[None, :]
        alpha_maps: list[np.ndarray] = []
        for entry in tile.entries:
            delta_row = rows - entry.projected.screen_rc[0]
            delta_col = columns - entry.projected.screen_rc[1]
            quadratic = (
                entry.projected.precision_2d[0, 0] * delta_row * delta_row
                + 2.0 * entry.projected.precision_2d[0, 1] * delta_row * delta_col
                + entry.projected.precision_2d[1, 1] * delta_col * delta_col
            )
            alpha_maps.append(np.clip(entry.projected.opacity * np.exp(-0.5 * quadratic), 0.0, 1.0))
        for local_row, row in enumerate(range(row0, row1)):
            for local_col, column in enumerate(range(col0, col1)):
                t = 1.0
                value = np.zeros_like(background_array)
                first = True
                for entry, alpha_map in zip(tile.entries, alpha_maps, strict=True):
                    if opaque_depth is not None and entry.depth >= opaque_depth[row, column]:
                        continue
                    opacity = float(alpha_map[local_row, local_col])
                    if opacity <= 0.0:
                        continue
                    if first:
                        depth_out[row, column] = entry.depth
                        primitive_out[row, column] = entry.projected.primitive_id
                        first = False
                    value += t * opacity * np.asarray(entry.projected.color)
                    t *= 1.0 - opacity
                value += t * background_array
                color[row, column] = value
                transmittance[row, column] = t
                alpha_out[row, column] = 1.0 - t
                coverage[row, column] = 1.0 - t
    return color, transmittance, alpha_out, depth_out, primitive_out, coverage


def _scale_error_ledger(ledger: v02.ErrorLedger, scale: float) -> v02.ErrorLedger:
    """Scale a source ledger through a non-negative compositing gain."""

    factor = abs(float(scale))

    def scale_bound(value: float) -> float:
        # An unknown source multiplied by an exactly zero gain contributes no
        # output uncertainty.  Avoid ``0 * inf -> nan`` in that case while
        # retaining an unbounded result for every non-zero unknown gain.
        value = float(value)
        if value == 0.0 or factor == 0.0:
            return 0.0
        if math.isinf(value) or math.isinf(factor):
            return math.inf
        return value * factor

    return v02.ErrorLedger(
        **{
            name: scale_bound(float(getattr(ledger, name)))
            for name in (
                "source_quantization",
                "representation_fit",
                "pruning",
                "reducer",
                "transform",
                "materialization",
                "display",
            )
        },
        stages=ledger.stages,
    )


def _splat_source_error_ledger(
    bins: TileBins,
    *,
    background_bound: float,
) -> v02.ErrorLedger:
    """Propagate opacity/color source bounds through final RGB compositing.

    An opacity error changes both the source contribution and the attenuated
    background.  A color error is weighted by the largest possible opacity.
    The opacity gain includes the color error so the product cross-term is
    retained.  This is a conservative terminal bound over all projected
    splats and therefore remains valid for either the reference or portable
    compositing implementation.
    """

    # An opacity perturbation also changes the attenuated suffix behind the
    # source.  Use the largest bounded radiance in the complete projected
    # scene as a conservative suffix envelope (including the background),
    # rather than only the background value.
    signal_bound = max(
        background_bound,
        max((item.max_color_abs_bound for item in bins.projected), default=0.0),
    )
    result = v02.ErrorLedger()
    for item in bins.projected:
        color_bound = item.max_color_abs_bound
        opacity_gain = color_bound + signal_bound
        color_gain = item.opacity_upper_bound
        result = result.plus(_scale_error_ledger(item.opacity_ledger, opacity_gain))
        result = result.plus(_scale_error_ledger(item.color_ledger, color_gain))
    return result


def _splat_ingress_summary(bins: TileBins) -> dict[str, Any]:
    """Summarize producer crossings once per source primitive."""

    unique = {item.source_index: item for item in bins.projected}
    counts = {"block_to_float64": 0, "native_scalar": 0, "block_words": 0}
    ingress: list[str] = []
    field_ids: list[str] = []
    field_operations: list[tuple[str, tuple[str, ...]]] = []
    chart_dimensions: list[tuple[str, int]] = []
    residual_count = 0
    for item in unique.values():
        counts["block_to_float64"] += int(item.source_conversion_count)
        source_ingress = item.source_ingress or ("native_scalar",)
        for source_kind in source_ingress:
            if source_kind == "addressed_block_to_float64":
                ingress.append(source_kind)
            elif source_kind == "addressed_block_words":
                ingress.append(source_kind)
                counts["block_words"] += 1
            else:
                ingress.append("native_scalar")
                counts["native_scalar"] += 1
        for source in (item.opacity_source, item.color_source):
            if source is None:
                continue
            field_id = str(getattr(source, "id", "field"))
            field_ids.append(field_id)
            field_operations.append((field_id, tuple(operation.op for operation in source.operations)))
            chart_dimensions.append((field_id, int(source.chart.dimension)))
            residual_count += int(source.representation.residual is not None)
    return {
        "source_ingress": tuple(sorted(set(ingress))),
        "source_block_count": sum(
            1
            for item in unique.values()
            if any(kind in _BLOCK_INGRESS for kind in (item.source_ingress or ()))
        ),
        "source_conversion_count": counts["block_to_float64"],
        "conversion_counts": counts,
        "block_native_arithmetic": False,
        "source_field_ids": tuple(field_ids),
        "source_field_operations": tuple(field_operations),
        "source_chart_dimensions": tuple(chart_dimensions),
        "source_residual_fields": residual_count,
        "source_execution_class": "python_addressed_block_projection"
        if counts["block_to_float64"]
        else "addressed_block_words"
        if counts["block_words"]
        else "native_scalar_ingress",
    }


def _report(
    *,
    backend: str,
    bins: TileBins,
    counters: v02.PerformanceCounters,
    error_budget: float,
    color: np.ndarray,
    reference: np.ndarray | None,
    certificates: Sequence[EarlyTerminationCertificate],
    representation: str,
    atoms_pruned: int = 0,
    background: Sequence[float] = (0.0, 0.0, 0.0),
) -> RenderReport:
    shape = bins.image_shape
    spatial_samples = int(math.prod(shape))
    potential_splat_samples = spatial_samples * max(1, len(bins.projected))
    considered = sum(len(tile.entries) for tile in bins.tiles)
    max_occupancy = max((len(tile.entries) for tile in bins.tiles), default=0)
    metrics = RenderMetrics(
        backend=str(backend),
        workload="gaussian_splats",
        resolution=shape,
        representation=representation,
        precision="balanced",
        error_budget=float(error_budget),
        bytes_read=int(counters.bytes_read),
        bytes_written=int(counters.bytes_written),
        peak_memory=int(color.nbytes),
        potential_elements=spatial_samples,
        materialized_elements=spatial_samples,
        atoms_considered=considered,
        atoms_pruned=int(atoms_pruned),
        positive_logadds=int(counters.positive_logadds),
        complex_adds=int(counters.complex_pair_adds),
        pivot_reductions=int(counters.pivot_reductions),
        transport_compositions=int(counters.transport_compositions),
        fft_count=0,
        fallback_conversions=int(counters.fallback_conversions),
        field_operations=considered + int(np.prod(shape, dtype=np.int64)),
        reduction_operations=int(np.prod(shape, dtype=np.int64)),
        synchronization_depth=math.ceil(math.log2(max(max_occupancy, 1))),
    )
    diagnostics: dict[str, Any] = {
        "execution_class": "reference_python" if backend == "reference" else "portable_python",
        "actual_execution_class": "reference_python" if backend == "reference" else "portable_python",
        "actual_callable": (
            "spf.renderer.splats.rasterize_splats_reference"
            if backend == "reference"
            else "spf.renderer.splats.rasterize_splats_spf"
        ),
        "hardware_native": False,
        "tiles": bins.tile_count,
        "tile_size": bins.tile_size,
        "potential_splat_samples": potential_splat_samples,
        "evaluated_tile_entries": considered,
        "byte_traffic_status": "logical_counter",
        "peak_memory_status": "logical_output_only",
        "energy_status": "unmeasured",
        "reference_comparison": reference is not None,
        "early_terminated_pixels": sum(1 for item in certificates if item.terminated),
        "certificates": len(certificates),
    }
    ingress = _splat_ingress_summary(bins)
    diagnostics.update(ingress)
    background_bound = max((abs(float(value)) for value in background), default=0.0)
    ledger = _splat_source_error_ledger(bins, background_bound=background_bound)
    if backend != "reference":
        signal_bound = max(
            background_bound,
            max((item.max_color_abs_bound for item in bins.projected), default=0.0),
        )
        # Exact front-to-back compositing is a convex weighting of source and
        # background values, hence |C| <= signal_bound.  The portable SPF path
        # quantizes alpha, complement, products, and reducer outputs.  Until a
        # tighter per-node interval ledger is exposed, bound the computed side
        # by the worst multiplicative quantizer growth and compare the two
        # bounded values.  This is intentionally conservative but independent
        # of an optional float oracle, so an empty ledger can never masquerade
        # as a proof of exactness.
        # Per contributing segment: alpha emission, complement, color product,
        # and ordered transport composition; one terminal/background stage.
        quantized_stages = 1 + 4 * max_occupancy
        # Alpha compositing is non-expansive for alpha in [0,1].  Charging one
        # worst-case relative quantizer factor per typed stage therefore bounds
        # the departure from the exact convex result without the useless 2B
        # interval bound.
        reducer_bound = signal_bound * ((1.0 + _SPF_RELATIVE_QUANTIZATION) ** quantized_stages - 1.0)
        # Binning truncates each Gaussian when its certified alpha bound falls
        # below min_alpha.  Account for both omitted source color and the
        # possible change to the entire attenuated suffix, including brighter
        # rear splats as well as the background.
        pruning_bound = sum(
            bins.min_alpha * (item.max_color_abs_bound + signal_bound)
            for item in bins.projected
            if item.visible
        )
        early_bound = max((item.error_bound for item in certificates if item.terminated), default=0.0)
        ledger = ledger.plus(v02.ErrorLedger(pruning=pruning_bound + early_bound, reducer=reducer_bound))
        diagnostics.update(
            {
                "certificate_model": "convex_signal_plus_spf_quantizer_growth",
                "signal_bound": signal_bound,
                "quantized_stage_bound": quantized_stages,
                "independent_reducer_bound": reducer_bound,
                "support_pruning_bound": pruning_bound,
            }
        )
    if reference is not None:
        data_range = _quality_data_range(np.asarray(reference), None)
        metrics.relative_l2 = relative_l2(color, reference)
        metrics.psnr = psnr(color, reference, data_range=data_range)
        metrics.ssim = ssim_global(color, reference, data_range=data_range)
        observed = float(np.max(np.abs(color - reference))) if color.size else 0.0
        if observed > ledger.total_bound:
            ledger = ledger.plus(v02.ErrorLedger(reducer=observed - ledger.total_bound))
        diagnostics["reference_max_abs_error"] = observed
        diagnostics["quality_data_range"] = data_range
    if certificates:
        diagnostics["max_early_termination_bound"] = max(item.error_bound for item in certificates)
    return RenderReport(
        metrics=metrics,
        ledger=ledger,
        fallbacks=(),
        capabilities_used=("CAMERA_PROJECTION", "JET2D", "LOG_ALPHA", "AFFINE_TRANSPORT"),
        diagnostics=diagnostics,
    )


def rasterize_splats_reference(
    splats: Iterable[GaussianSplat],
    camera: Camera,
    *,
    image_shape: tuple[int, int] | None = None,
    tile_size: int = 16,
    min_alpha: float = 1.0e-4,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    opaque_depth: Any | None = None,
    bins: TileBins | None = None,
) -> SplatRenderResult:
    """Render depth-sorted splats with direct float64 alpha compositing."""

    background_tuple = tuple(float(value) for value in background)
    if len(background_tuple) != 3 or not all(math.isfinite(value) for value in background_tuple):
        raise ValueError("background must be a finite RGB three-vector")
    selected_bins = bins or bin_splats(
        splats,
        camera,
        image_shape=image_shape,
        tile_size=tile_size,
        min_alpha=min_alpha,
    )
    depth_limit = _opaque_depth_limit(opaque_depth, selected_bins.image_shape)
    outputs = _reference_core(selected_bins, background=background_tuple, opaque_depth=depth_limit)
    counters = v02.PerformanceCounters()
    counters.fallback_conversions = int(_splat_ingress_summary(selected_bins)["source_conversion_count"])
    counters.values_materialized = int(np.prod(selected_bins.image_shape, dtype=np.int64) * 3)
    counters.bytes_written = int(outputs[0].nbytes)
    report = _report(
        backend="reference",
        bins=selected_bins,
        counters=counters,
        error_budget=0.0,
        color=outputs[0],
        reference=None,
        certificates=(),
        representation="dense_float64",
        background=background_tuple,
    )
    return SplatRenderResult(*outputs, report=report, tiles=selected_bins)


def _budget_value(error_budget: Any) -> float:
    if error_budget is None:
        return 0.0
    value = error_budget.absolute if hasattr(error_budget, "absolute") else error_budget
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("error_budget must be a finite non-negative absolute bound")
    return result


def _opaque_depth_limit(value: Any | None, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or np.any(np.isnan(result)) or np.any(result < 0.0):
        raise ValueError("opaque_depth must match image_shape and contain non-negative depths without NaNs")
    return result


def rasterize_splats_spf(
    splats: Iterable[GaussianSplat],
    camera: Camera,
    *,
    image_shape: tuple[int, int] | None = None,
    tile_size: int = 16,
    min_alpha: float = 1.0e-4,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    opaque_depth: Any | None = None,
    error_budget: Any = 0.0,
    reduction: str = "serial",
    guard_L: int = 8,
    early_terminate: bool = True,
    compare_reference: bool = False,
    bins: TileBins | None = None,
    terminal: str = "auto",
) -> SplatRenderResult:
    """Render splats through SPF log-alpha and ordered transport.

    ``terminal="planes"`` forces the word-plane compositor
    (:func:`spf.renderer.splat_terminal.rasterize_planes`); ``"cuda"`` forces
    the device compositor (:func:`spf.renderer._splat_cuda.rasterize`, CUDA
    when torch reports one, else CPU torch, words identical to the plane
    path); ``"auto"`` selects the device compositor when a CUDA runtime is
    present and every visible splat's opacity and colour were read as packed
    block words, else the plane compositor for such word-native scenes.

    The returned ``color`` is a native display projection rather than a
    Cartesian complex decode.  ``color_typed``/``color_words`` retain the
    typed output.  The float64 oracle is opt-in via ``compare_reference`` so
    it cannot accidentally become part of a production render.
    ``error_budget=0`` disables early termination; a positive budget enables
    it only when a conservative skipped-tail bound fits the budget.
    """

    background_tuple = tuple(float(value) for value in background)
    if len(background_tuple) != 3 or not all(math.isfinite(value) for value in background_tuple):
        raise ValueError("background must be a finite RGB three-vector")
    budget = _budget_value(error_budget)
    guard = int(guard_L)
    if guard < 0:
        raise ValueError("guard_L must be non-negative")
    selected_bins = bins or bin_splats(
        splats,
        camera,
        image_shape=image_shape,
        tile_size=tile_size,
        min_alpha=min_alpha,
        guard_L=guard,
    )
    shape = selected_bins.image_shape
    depth_limit = _opaque_depth_limit(opaque_depth, shape)
    if str(reduction).lower().replace("-", "_") not in {"serial", "tree", "auto"}:
        raise ValueError("reduction must be 'serial', 'tree', or 'auto'")
    # A finite source ledger is a statement about the input, not just about
    # the nominal decoded values.  The current native/portable tail tests do
    # not yet propagate that interval through skipped segments, so retain all
    # entries whenever source uncertainty is present.
    source_uncertain = any(item.source_error_bound > 0.0 for item in selected_bins.projected)
    effective_early_terminate = bool(early_terminate and not source_uncertain)
    from spf.renderer.splat_terminal import rasterize_native, rasterize_planes

    words_native = bool(selected_bins.projected) and all(
        item.words_native for item in selected_bins.projected if item.visible and item.tile_bounds is not None
    )
    # "cuda", "torch", "torch_cpu", "torch_cuda" and "device" all name the
    # torch integer compositor (resolve_terminal raises when a pinned device
    # request cannot be honoured, so it never degrades silently).
    device_terminal = _splat_cuda.resolve_terminal(terminal)
    if device_terminal is None and terminal not in {"planes", "native", "typed"}:
        raise ValueError(
            "terminal must be 'auto', 'planes', 'native', 'typed' or a device terminal "
            f"({', '.join(sorted(_splat_cuda.DEVICE_TERMINALS))})"
        )
    # The fused single-launch CUDA tile kernel (design §36.3 / Appendix B) is the
    # first choice on a CUDA host whose extension builds; its words are the plane
    # path's words.  ``auto`` falls through to the torch compositor and then to
    # the CPU planes when it cannot serve the scene.
    if device_terminal == "device_fused" or (
        terminal == "auto" and words_native and _splat_cuda.fused_available()
    ):
        fused_result = _splat_cuda.rasterize(
            selected_bins,
            background_tuple,
            depth_limit,
            budget,
            effective_early_terminate,
            compare_reference,
            fused=True,
        )
        if fused_result is not None:
            return fused_result
    if device_terminal == "device" or (terminal == "auto" and words_native and _splat_cuda.cuda_available()):
        device_result = _splat_cuda.rasterize(
            selected_bins,
            background_tuple,
            depth_limit,
            budget,
            effective_early_terminate,
            compare_reference,
        )
        if device_result is not None:
            return device_result
    if words_native or terminal == "planes":
        plane_result = rasterize_planes(
            selected_bins,
            background_tuple,
            depth_limit,
            budget,
            effective_early_terminate,
            compare_reference,
        )
        if plane_result is not None:
            return plane_result
    native = rasterize_native(
        selected_bins,
        background_tuple,
        depth_limit,
        budget,
        effective_early_terminate,
        compare_reference,
    )
    if native is not None:
        return native
    color, transmittance, alpha_out, depth_out, primitive_out, coverage = _empty_outputs(shape)
    color_typed = np.empty((*shape, 3), dtype=object)
    transmittance_typed = np.empty(shape, dtype=object)
    color_words = np.zeros((*shape, 3), dtype=np.uint32)
    transmittance_words = np.zeros(shape, dtype=np.uint32)
    counters = v02.PerformanceCounters()
    counters.fallback_conversions = int(_splat_ingress_summary(selected_bins)["source_conversion_count"])
    certificates: list[EarlyTerminationCertificate] = []
    # Initialize the complete image before visiting sparse bins.  A scene can
    # contain one occupied tile and many empty tiles; typed outputs must still
    # have the same explicit background/identity values everywhere.
    background_typed = [_typed_background(value) for value in background_tuple]
    for row, column in np.ndindex(shape):
        color_typed[row, column] = list(background_typed)
        transmittance_typed[row, column] = v02.LogPos(0)
        color[row, column] = np.asarray([_decode_typed(value) for value in background_typed])
        color_words[row, column] = np.asarray([v02.pack_spf32(value) for value in background_typed], dtype=np.uint32)
        transmittance_words[row, column] = np.uint32(v02.pack_spf32(v02.LogPos(0)))
    for tile in selected_bins.tiles:
        row0, col0, row1, col1 = tile.bounds
        tile_shape = (row1 - row0, col1 - col0)
        alpha_maps = [_typed_alpha_tile(entry, counters) for entry in tile.entries]
        alpha_float_maps = [
            np.asarray(
                [[_decode_typed(cast(v02.LogPos, alpha_maps[index][position])) for position in np.ndindex(tile_shape)]],
                dtype=np.float64,
            ).reshape(tile_shape)
            for index in range(len(alpha_maps))
        ]
        for local_row, row in enumerate(range(row0, row1)):
            for local_col, column in enumerate(range(col0, col1)):
                alpha_values = [
                    float(alpha_float_maps[index][local_row, local_col]) for index in range(len(tile.entries))
                ]
                if depth_limit is not None:
                    alpha_values = [
                        0.0 if entry.depth >= depth_limit[row, column] else alpha_value
                        for entry, alpha_value in zip(tile.entries, alpha_values, strict=True)
                    ]
                segments_by_channel: list[list[v02.Transport]] = [[], [], []]
                prefix_abs = 0.0
                first_entry: ProjectedSplatTile | None = None
                terminated = False
                processed = 0
                remaining_bound = 0.0
                for index, (entry, alpha_log) in enumerate(
                    (entry, alpha_maps[entry_index][local_row, local_col])
                    for entry_index, entry in enumerate(tile.entries)
                ):
                    if alpha_values[index] <= 0.0 or alpha_log.is_zero:
                        processed = index + 1
                        continue
                    if first_entry is None:
                        first_entry = entry
                    transmittance_log = v02.logsub_pos(v02.LogPos(0), alpha_log, counters=counters)
                    for channel, color_value in enumerate(entry.projected.color):
                        contribution = _typed_channel_product(alpha_log, color_value)
                        segments_by_channel[channel].append(v02.Transport(transmittance_log, contribution))
                    prefix_abs += alpha_values[index] * entry.projected.max_color_abs
                    processed = index + 1
                    if effective_early_terminate and budget > 0.0 and index + 1 < len(tile.entries):
                        tail = sum(
                            alpha_values[tail_index] * tile.entries[tail_index].projected.max_color_abs
                            for tail_index in range(index + 1, len(tile.entries))
                        )
                        tail += max(abs(value) for value in background_tuple)
                        # SPF's scalar quantizer is included as a conservative
                        # margin.  It intentionally makes early termination
                        # less aggressive on very small budgets.
                        quantization_margin = _SPF_RELATIVE_QUANTIZATION * max(1.0, prefix_abs + tail)
                        prefix_t = math.prod(1.0 - value for value in alpha_values[: index + 1])
                        remaining_bound = prefix_t * tail + quantization_margin
                        if remaining_bound <= budget:
                            terminated = True
                            break
                if not segments_by_channel[0]:
                    typed_t = v02.LogPos(0)
                    typed_channels: list[v02.LogPos | v02.LogSigned] = [
                        _typed_background(value) for value in background_tuple
                    ]
                else:
                    reduced = [
                        _compose_segments(segments_by_channel[channel], reduction=reduction, counters=counters)
                        for channel in range(3)
                    ]
                    typed_t = reduced[0].T
                    typed_channels = []
                    for channel, transport in enumerate(reduced):
                        typed_channels.append(
                            transport.apply(_typed_background(background_tuple[channel]), counters=counters)
                        )
                if not isinstance(typed_t, v02.LogPos):
                    # Every segment uses positive transmittance; this is a
                    # defensive check against an accidental mixed algebra.
                    raise TypeError("splat transport must retain LOG_POS transmittance")
                transmittance_typed[row, column] = typed_t
                transmittance[row, column] = _decode_typed(typed_t)
                color_typed[row, column] = typed_channels
                for channel, typed_value in enumerate(typed_channels):
                    color[row, column, channel] = _decode_typed(typed_value)
                    color_words[row, column, channel] = np.uint32(v02.pack_spf32(typed_value))
                transmittance_words[row, column] = np.uint32(v02.pack_spf32(typed_t))
                alpha_out[row, column] = 1.0 - transmittance[row, column]
                coverage[row, column] = alpha_out[row, column]
                if first_entry is not None:
                    depth_out[row, column] = first_entry.depth
                    primitive_out[row, column] = first_entry.projected.primitive_id
                certificates.append(
                    EarlyTerminationCertificate(
                        tile.index,
                        (row, column),
                        processed,
                        max(0, len(tile.entries) - processed),
                        terminated,
                        float(max(0.0, remaining_bound)),
                        float(max(0.0, remaining_bound)),
                    )
                )
    reference = None
    if compare_reference:
        reference = _reference_core(selected_bins, background=background_tuple, opaque_depth=depth_limit)[0]
    report = _report(
        backend="portable_python",
        bins=selected_bins,
        counters=counters,
        error_budget=budget,
        color=color,
        reference=reference,
        certificates=certificates,
        representation="spf_log_alpha_transport",
        background=background_tuple,
    )
    return SplatRenderResult(
        color,
        transmittance,
        alpha_out,
        depth_out,
        primitive_out,
        coverage,
        report,
        selected_bins,
        color_typed,
        transmittance_typed,
        color_words,
        transmittance_words,
        tuple(certificates),
    )


rasterize_splats = rasterize_splats_spf
rasterize_splats_cpu = rasterize_splats_spf
rasterize_splats_float64 = rasterize_splats_reference


__all__ = [
    "Camera",
    "CameraProjection",
    "EarlyTerminationCertificate",
    "GaussianSplat",
    "ProjectedSplat",
    "ProjectedSplatRecord",
    "ProjectedSplatTile",
    "SplatRenderResult",
    "SplatTile",
    "SplatTileRecord",
    "TileBins",
    "bin_splats",
    "build_tile_bins",
    "log2_alpha_jet",
    "make_log_alpha_jet",
    "project_gaussian_splat",
    "project_splat",
    "project_splats",
    "rasterize_splats",
    "rasterize_splats_cpu",
    "rasterize_splats_float64",
    "rasterize_splats_reference",
    "rasterize_splats_spf",
    "spf_log_alpha_jet",
]
