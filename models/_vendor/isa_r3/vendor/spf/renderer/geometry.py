"""Tiered geometry, the §19.5 ``VisibleSample`` contract, and §20 visibility.

Design v0.2 §19 splits geometry into four tiers and requires every one of them
to emit the same typed visibility record before the field plane starts:

====== ============================= =====================================
tier   representation                module surface
====== ============================= =====================================
0      conventional meshes           :class:`spf.renderer.scene.Mesh`
1      analytic patches (§19.2)      :class:`AnalyticPatch` implementations
2      field geometry (§19.3)        :class:`HermitianHeightField`
3      3-D Gaussian primitives       :class:`GaussianGeometry`
====== ============================= =====================================

Two rules shape this module.

*The field plane begins after visibility.*  Every tier resolves depth and
coverage first (§20.1 fixed-point ``z* = min z``, §20.2 tile culling on
geometry / support / magnitude / opacity bounds), publishes a
:class:`VisibleSampleBuffer`, and only then binds materials and lighting.  A
culled tile therefore costs no field evaluation at all, which is the whole
source of the materialization-avoidance ratio this module reports
(``M_A = potential_elements / materialized_elements``, §16.3 / §48.3).

*Structure stays on the carrier.*  Tier 1 lowers its shading-relevant
quantities to :class:`spf.tensor.v02.Jet2D` log jets (and through
:func:`spf.tensor.v02.jet_to_tensor_block_header` into ``TensorBlockHeaderV2``
structure), tier 2 keeps the Hermitian spectrum and takes derivatives by the
§28.1 quarter-turn, tier 3 delegates to the existing
:mod:`spf.renderer.splats` log-alpha route.  Shading runs on
``LogPosPlane`` word planes and materializes once.  No function here decodes a
dense complex buffer, so this file holds no ``np.fft`` / ``complex128`` /
``np.exp`` / ``np.log`` site (AGENTS.md rule 7); the float32 comparison arms
live in ``benchmarks/sfr_geometry_bench.py``.

``VisibleSampleBuffer`` is the array-structured form of the §19.5 record: one
plane per field of ``VisibleSample`` so a 4K frame does not need eight million
Python objects.  :meth:`VisibleSampleBuffer.sample` materializes the typed
record for any pixel, and :meth:`VisibleSampleBuffer.samples` iterates the
covered ones, so the typed contract remains the interface.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np

from spf.renderer.materials import Material
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision
from spf.tensor import v02
from spf.tensor.v02_planes import LogPosPlane, reduce_logpos

__all__ = [
    "DEPTH_TICKS_PER_UNIT",
    "AnalyticPatch",
    "CullRecord",
    "FourierPatch",
    "GaussianGeometry",
    "GeometryLayer",
    "GeometryTier",
    "HeightFieldPatch",
    "HermitianHeightField",
    "PatchFrame",
    "PatchHit",
    "PolynomialPatch",
    "QuadricPatch",
    "SignedDistancePatch",
    "TileCullResult",
    "TileRegion",
    "VisibleSampleBuffer",
    "cull_tiles",
    "depth_ticks",
    "rasterize_geometry",
    "render_scene",
    "resolve_depth_min",
    "shade_samples",
]

_EPS = 1.0e-12
_LOG2_E = 1.4426950408889634
_LN2 = 0.6931471805599453

# §20.1: depth selection is an integer/fixed-point compare.  One tick is
# 2**-16 world units, so a 2**47-unit scene still fits int64 without wrapping
# and two depths within a tick compare equal rather than flickering.
DEPTH_FRACTION_BITS = 16
DEPTH_TICKS_PER_UNIT = float(1 << DEPTH_FRACTION_BITS)
_DEPTH_TICK_MAX = np.int64(np.iinfo(np.int64).max)


class GeometryTier(StrEnum):
    """The four §19 geometry tiers, spelled as report/diagnostic labels."""

    MESH = "tier0_mesh"
    PATCH = "tier1_patch"
    FIELD = "tier2_field"
    GAUSSIAN = "tier3_gaussian"


# ---------------------------------------------------------------------------
# §20.1 depth selection on a fixed-point compare tree
# ---------------------------------------------------------------------------


def depth_ticks(depth: Any) -> np.ndarray:
    """Quantize depth to the int64 compare key used for ``z* = min z``.

    Non-finite or negative depth becomes the maximum key, which loses every
    comparison, so an empty sample can never win visibility.  Quantization is
    ``floor``, making the key monotone in depth: ``a < b`` in ticks implies
    ``a < b`` in world units up to one tick, and never the reverse.
    """

    values = np.asarray(depth, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        scaled = np.floor(values * DEPTH_TICKS_PER_UNIT)
    # The sentinel is assigned in integer space: float64 cannot hold
    # ``int64.max`` exactly, so casting it would overflow the very key that is
    # meant to lose every comparison.
    usable = np.isfinite(values) & (values >= 0.0) & (scaled < float(_DEPTH_TICK_MAX))
    finite = np.asarray(np.where(usable, scaled, 0.0), dtype=np.int64)
    return np.asarray(np.where(usable, finite, _DEPTH_TICK_MAX), dtype=np.int64)


def resolve_depth_min(
    depths: Sequence[Any],
    *,
    coverage: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve ``z* = min_i z_i`` across candidate layers on integer keys.

    Returns ``(winner, ticks)``: ``winner`` holds the index of the selected
    layer (``-1`` where nothing is covered) and ``ticks`` the winning
    fixed-point key.  Ties in ticks go to the lower layer index, which makes
    the result independent of evaluation order.
    """

    if not depths:
        raise ValueError("resolve_depth_min requires at least one candidate layer")
    keys = [depth_ticks(item) for item in depths]
    shape = keys[0].shape
    if any(item.shape != shape for item in keys):
        raise ValueError("candidate depth layers must share one shape")
    masks: list[np.ndarray]
    if coverage is None:
        masks = [np.ones(shape, dtype=bool) for _ in keys]
    else:
        if len(coverage) != len(keys):
            raise ValueError("coverage must supply one plane per depth layer")
        masks = [np.asarray(item, dtype=np.float64) > 0.0 for item in coverage]
        if any(item.shape != shape for item in masks):
            raise ValueError("coverage planes must share the depth shape")
    best = np.full(shape, _DEPTH_TICK_MAX, dtype=np.int64)
    winner = np.full(shape, -1, dtype=np.int64)
    for index, (key, mask) in enumerate(zip(keys, masks, strict=True)):
        eligible = mask & (key < best)
        best = np.where(eligible, key, best)
        winner = np.where(eligible, np.int64(index), winner)
    return winner, best


# ---------------------------------------------------------------------------
# §19.5 VisibleSample contract, array-structured
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VisibleSampleBuffer:
    """One plane per §19.5 ``VisibleSample`` field over an image rectangle.

    ``depth`` is ``inf`` and ``coverage`` ``0`` where no primitive is visible;
    ``primitive_id`` and ``material_id`` are ``-1`` there.  ``ticks`` carries
    the §20.1 fixed-point compare key so a resolver never re-quantizes.
    """

    tier: GeometryTier
    depth: np.ndarray
    coverage: np.ndarray
    primitive_id: np.ndarray
    material_id: np.ndarray
    local_coordinates: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    field_bindings: Mapping[str, np.ndarray] = field(default_factory=dict)
    potential_elements: int = 0

    def __post_init__(self) -> None:
        depth = np.asarray(self.depth, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError("visible sample buffers are two-dimensional (H, W)")
        shape = depth.shape
        coverage = np.asarray(self.coverage, dtype=np.float64)
        primitive_id = np.asarray(self.primitive_id, dtype=np.int64)
        material_id = np.asarray(self.material_id, dtype=np.int64)
        local = np.asarray(self.local_coordinates, dtype=np.float64)
        position = np.asarray(self.position, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        if coverage.shape != shape or primitive_id.shape != shape or material_id.shape != shape:
            raise ValueError("coverage and id planes must share the depth shape")
        if local.shape != (*shape, 2):
            raise ValueError("local coordinates must have shape (H, W, 2)")
        if position.shape != (*shape, 3) or normal.shape != (*shape, 3):
            raise ValueError("position and normal planes must have shape (H, W, 3)")
        if np.any(coverage < 0.0) or np.any(coverage > 1.0) or not np.all(np.isfinite(coverage)):
            raise ValueError("visible sample coverage must be finite and in [0, 1]")
        if np.any(depth < 0.0) or np.any(np.isnan(depth)):
            raise ValueError("visible sample depth must be non-negative or infinity")
        bindings = {str(name): np.asarray(value) for name, value in self.field_bindings.items()}
        for name, value in bindings.items():
            if value.shape[:2] != shape:
                raise ValueError(f"field binding {name!r} must share the buffer (H, W) shape")
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "primitive_id", primitive_id)
        object.__setattr__(self, "material_id", material_id)
        object.__setattr__(self, "local_coordinates", local)
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "normal", normal)
        object.__setattr__(self, "field_bindings", MappingProxyType(bindings))
        object.__setattr__(self, "potential_elements", int(self.potential_elements))

    @classmethod
    def empty(cls, shape: tuple[int, int], tier: GeometryTier, *, potential_elements: int = 0) -> VisibleSampleBuffer:
        height, width = int(shape[0]), int(shape[1])
        normal = np.zeros((height, width, 3), dtype=np.float64)
        normal[..., 2] = 1.0
        return cls(
            tier=tier,
            depth=np.full((height, width), math.inf, dtype=np.float64),
            coverage=np.zeros((height, width), dtype=np.float64),
            primitive_id=np.full((height, width), -1, dtype=np.int64),
            material_id=np.full((height, width), -1, dtype=np.int64),
            local_coordinates=np.zeros((height, width, 2), dtype=np.float64),
            position=np.zeros((height, width, 3), dtype=np.float64),
            normal=normal,
            potential_elements=potential_elements,
        )

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.depth.shape[0]), int(self.depth.shape[1]))

    @property
    def ticks(self) -> np.ndarray:
        """The §20.1 fixed-point compare key for every pixel."""

        return depth_ticks(self.depth)

    @property
    def visible_mask(self) -> np.ndarray:
        return np.asarray((self.coverage > 0.0) & np.isfinite(self.depth))

    @property
    def realized_elements(self) -> int:
        """Covered samples: what the field plane will actually evaluate."""

        return int(np.count_nonzero(self.visible_mask))

    @property
    def materialization_avoidance(self) -> float:
        realized = self.realized_elements
        if realized == 0:
            return math.inf if self.potential_elements else 1.0
        return float(self.potential_elements) / float(realized)

    def sample(self, pixel: tuple[int, int], *, material: Material | None = None) -> Any:
        """Materialize the typed §19.5 record for one pixel."""

        from spf.renderer.scene import VisibleSample

        x, y = int(pixel[0]), int(pixel[1])
        bindings = {name: plane[y, x] for name, plane in self.field_bindings.items()}
        return VisibleSample(
            pixel=(x, y),
            depth=float(self.depth[y, x]),
            coverage=float(self.coverage[y, x]),
            position=tuple(float(value) for value in self.position[y, x]),
            normal=tuple(float(value) for value in self.normal[y, x]),
            primitive_id=int(self.primitive_id[y, x]),
            material=material,
            source=str(self.tier.value),
            local_coordinates=tuple(float(value) for value in self.local_coordinates[y, x]),
            field_bindings=bindings,
            material_id=int(self.material_id[y, x]),
        )

    def samples(self, *, material: Material | None = None, limit: int | None = None) -> Iterator[Any]:
        """Iterate the typed records for covered pixels in row-major order."""

        rows, columns = np.nonzero(self.visible_mask)
        count = rows.size if limit is None else min(int(limit), rows.size)
        for index in range(count):
            yield self.sample((int(columns[index]), int(rows[index])), material=material)


# ---------------------------------------------------------------------------
# §20.2 tile culling
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TileRegion:
    """A screen tile in ``(row0, col0, row1, col1)`` half-open pixel bounds."""

    index: tuple[int, int]
    bounds: tuple[int, int, int, int]

    @property
    def pixels(self) -> int:
        row0, col0, row1, col1 = self.bounds
        return max(0, row1 - row0) * max(0, col1 - col0)

    def as_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Lower/upper corner arrays in ``(row, column)`` order."""

        row0, col0, row1, col1 = self.bounds
        return (
            np.asarray((float(row0), float(col0)), dtype=np.float64),
            np.asarray((float(row1 - 1), float(col1 - 1)), dtype=np.float64),
        )


@dataclass(frozen=True, slots=True)
class CullRecord:
    """Why one tile survived or did not, with the bounds that decided it."""

    tile: TileRegion
    culled: bool
    reason: str
    geometry_overlap: bool = True
    magnitude_bound: float = math.inf
    opacity_bound: float = 1.0
    transmittance_bound: float = 1.0

    @property
    def contribution_bound(self) -> float:
        """Certified upper bound on this tile's contribution to the frame."""

        return float(self.magnitude_bound * self.opacity_bound * self.transmittance_bound)


@dataclass(frozen=True, slots=True)
class TileCullResult:
    """Kept and culled tiles plus the avoidance ratio they imply."""

    shape: tuple[int, int]
    tile_size: int
    records: tuple[CullRecord, ...]

    @property
    def kept(self) -> tuple[CullRecord, ...]:
        return tuple(item for item in self.records if not item.culled)

    @property
    def culled(self) -> tuple[CullRecord, ...]:
        return tuple(item for item in self.records if item.culled)

    @property
    def potential_elements(self) -> int:
        return int(self.shape[0]) * int(self.shape[1])

    @property
    def realized_elements(self) -> int:
        return sum(item.tile.pixels for item in self.kept)

    @property
    def materialization_avoidance(self) -> float:
        realized = self.realized_elements
        if realized == 0:
            return math.inf if self.potential_elements else 1.0
        return float(self.potential_elements) / float(realized)

    def kept_mask(self) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        for record in self.kept:
            row0, col0, row1, col1 = record.tile.bounds
            mask[row0:row1, col0:col1] = True
        return mask


def tile_regions(shape: tuple[int, int], tile_size: int) -> tuple[TileRegion, ...]:
    """Enumerate row-major tiles covering ``shape`` exactly once."""

    height, width = int(shape[0]), int(shape[1])
    size = int(tile_size)
    if height <= 0 or width <= 0:
        raise ValueError("tile shape must be positive")
    if size <= 0:
        raise ValueError("tile_size must be positive")
    regions: list[TileRegion] = []
    for tile_row in range((height + size - 1) // size):
        for tile_col in range((width + size - 1) // size):
            regions.append(
                TileRegion(
                    (tile_row, tile_col),
                    (
                        tile_row * size,
                        tile_col * size,
                        min(height, (tile_row + 1) * size),
                        min(width, (tile_col + 1) * size),
                    ),
                )
            )
    return tuple(regions)


def cull_tiles(
    shape: tuple[int, int],
    tile_size: int,
    *,
    geometry_bounds: Sequence[Sequence[float]] = (),
    magnitude_bound: Callable[[TileRegion], float] | None = None,
    opacity_bound: Callable[[TileRegion], float] | None = None,
    transmittance_bound: Callable[[TileRegion], float] | None = None,
    atoms: Sequence[v02.GaussianChirpAtom] = (),
    min_contribution: float = 0.0,
) -> TileCullResult:
    """Cull tiles on §20.2 bounds before any field evaluation.

    Every test is a *certified upper bound* over the whole tile, so a tile is
    removed only when no pixel in it can contribute above
    ``min_contribution``:

    ``geometry_bounds``
        screen-space ``(row0, col0, row1, col1)`` rectangles, inclusive of the
        primitive's conservative extent.  A tile that meets none of them holds
        no geometry.  An empty sequence disables the test rather than culling
        everything.
    ``atoms``
        Gaussian–chirp atoms; :func:`spf.tensor.v02.atom_bound` supplies the
        certified maximum magnitude over the tile box (§11.2 atom support).
    ``magnitude_bound`` / ``opacity_bound`` / ``transmittance_bound``
        caller-supplied maxima over the tile.  They multiply into one
        contribution bound, matching §20.2's "maximum magnitude" and
        "opacity/transmittance" bounds.

    ``min_contribution = 0`` keeps every tile that holds geometry, which is
    the soundness-preserving default: nothing with a nonzero bound is removed.
    """

    threshold = float(min_contribution)
    if math.isnan(threshold) or threshold < 0.0:
        raise ValueError("min_contribution must be non-negative")
    boxes = [np.asarray(item, dtype=np.float64).reshape(-1) for item in geometry_bounds]
    if any(item.size != 4 for item in boxes):
        raise ValueError("geometry bounds must be (row0, col0, row1, col1) rectangles")
    records: list[CullRecord] = []
    for region in tile_regions(shape, tile_size):
        row0, col0, row1, col1 = region.bounds
        overlap = True
        if boxes:
            overlap = any(
                box[0] <= float(row1 - 1)
                and box[2] >= float(row0)
                and box[1] <= float(col1 - 1)
                and box[3] >= float(col0)
                for box in boxes
            )
        magnitude = math.inf if magnitude_bound is None else float(magnitude_bound(region))
        if atoms:
            lower, upper = region.as_box()
            atom_max = max(float(v02.atom_bound(atom, (lower, upper))) for atom in atoms)
            magnitude = atom_max if magnitude_bound is None else min(magnitude, atom_max)
        opacity = 1.0 if opacity_bound is None else float(opacity_bound(region))
        transmittance = 1.0 if transmittance_bound is None else float(transmittance_bound(region))
        if not overlap:
            reason, culled = "geometry_bounds", True
        elif magnitude <= 0.0:
            reason, culled = "magnitude_bound", True
        elif opacity <= 0.0:
            reason, culled = "opacity_bound", True
        elif transmittance <= 0.0:
            reason, culled = "transmittance_bound", True
        elif math.isfinite(magnitude) and magnitude * opacity * transmittance <= threshold:
            reason, culled = "contribution_below_budget", True
        else:
            reason, culled = "kept", False
        records.append(
            CullRecord(
                tile=region,
                culled=culled,
                reason=reason,
                geometry_overlap=overlap,
                magnitude_bound=magnitude,
                opacity_bound=opacity,
                transmittance_bound=transmittance,
            )
        )
    return TileCullResult(shape=(int(shape[0]), int(shape[1])), tile_size=int(tile_size), records=tuple(records))


# ---------------------------------------------------------------------------
# a scalar forward-mode 2-jet, used to derive analytic patch derivatives
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Jet2:
    """Value plus exact first and second derivatives in two variables.

    Tier 1 needs ``∂`` and ``∂²`` of quantities built from a patch's height
    function (normals, cosines, their logarithms).  Carrying the derivatives
    through the arithmetic is exact and keeps every patch type one
    implementation away from a :class:`spf.tensor.v02.Jet2D`.
    """

    v: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    dxx: float = 0.0
    dxy: float = 0.0
    dyy: float = 0.0

    @classmethod
    def constant(cls, value: float) -> _Jet2:
        return cls(float(value))

    def __add__(self, other: _Jet2 | float) -> _Jet2:
        o = other if isinstance(other, _Jet2) else _Jet2.constant(float(other))
        return _Jet2(self.v + o.v, self.dx + o.dx, self.dy + o.dy, self.dxx + o.dxx, self.dxy + o.dxy, self.dyy + o.dyy)

    def __neg__(self) -> _Jet2:
        return _Jet2(-self.v, -self.dx, -self.dy, -self.dxx, -self.dxy, -self.dyy)

    def __sub__(self, other: _Jet2 | float) -> _Jet2:
        o = other if isinstance(other, _Jet2) else _Jet2.constant(float(other))
        return self + (-o)

    def __mul__(self, other: _Jet2 | float) -> _Jet2:
        o = other if isinstance(other, _Jet2) else _Jet2.constant(float(other))
        return _Jet2(
            self.v * o.v,
            self.dx * o.v + self.v * o.dx,
            self.dy * o.v + self.v * o.dy,
            self.dxx * o.v + 2.0 * self.dx * o.dx + self.v * o.dxx,
            self.dxy * o.v + self.dx * o.dy + self.dy * o.dx + self.v * o.dxy,
            self.dyy * o.v + 2.0 * self.dy * o.dy + self.v * o.dyy,
        )

    __rmul__ = __mul__
    __radd__ = __add__

    def reciprocal(self) -> _Jet2:
        if abs(self.v) <= _EPS:
            raise ValueError("cannot invert a 2-jet through zero")
        inverse = 1.0 / self.v
        square = inverse * inverse
        cube = square * inverse
        return _Jet2(
            inverse,
            -self.dx * square,
            -self.dy * square,
            -self.dxx * square + 2.0 * self.dx * self.dx * cube,
            -self.dxy * square + 2.0 * self.dx * self.dy * cube,
            -self.dyy * square + 2.0 * self.dy * self.dy * cube,
        )

    def __truediv__(self, other: _Jet2 | float) -> _Jet2:
        o = other if isinstance(other, _Jet2) else _Jet2.constant(float(other))
        return self * o.reciprocal()

    def sqrt(self) -> _Jet2:
        if self.v <= 0.0:
            raise ValueError("cannot take the square root of a non-positive 2-jet")
        root = math.sqrt(self.v)
        half = 0.5 / root
        quarter = 0.25 / (root * self.v)
        return _Jet2(
            root,
            half * self.dx,
            half * self.dy,
            half * self.dxx - quarter * self.dx * self.dx,
            half * self.dxy - quarter * self.dx * self.dy,
            half * self.dyy - quarter * self.dy * self.dy,
        )

    def log2(self) -> _Jet2:
        if self.v <= 0.0:
            raise ValueError("cannot take log2 of a non-positive 2-jet")
        inverse = 1.0 / self.v
        scale = _LOG2_E * inverse
        return _Jet2(
            math.log2(self.v),
            scale * self.dx,
            scale * self.dy,
            scale * (self.dxx - self.dx * self.dx * inverse),
            scale * (self.dxy - self.dx * self.dy * inverse),
            scale * (self.dyy - self.dy * self.dy * inverse),
        )

    def to_jet2d(self, *, value_kind: str = "native", guard_L: int = 0) -> v02.Jet2D:
        """Convert to the library jet.  ``value_kind='log_pos'`` widens ticks.

        A ``log_pos`` jet is expressed in widened SPF32 L ticks exactly as
        :func:`spf.renderer.splats.spf_log_alpha_jet` does, so
        :func:`spf.tensor.v02.jet_forward_differences` and
        :func:`spf.tensor.v02.jet_to_tensor_block_header` accept it unchanged.
        """

        kind = str(value_kind).lower()
        scale = float(v02.SPF32_L_SCALE * (1 << int(guard_L))) if kind in {"log", "log_pos"} else 1.0
        return v02.Jet2D(
            value=self.v * scale,
            grad_x=self.dx * scale,
            grad_y=self.dy * scale,
            hess_xx=self.dxx * scale,
            hess_xy=self.dxy * scale,
            hess_yy=self.dyy * scale,
            guard_L=int(guard_L),
            guard_T=int(guard_L),
            value_kind=kind,
        )


# ---------------------------------------------------------------------------
# Tier 1 — analytic patches (§19.2)
# ---------------------------------------------------------------------------


def _screen_rect(camera: Any, points: Any, resolution: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Conservative ``(row0, col0, row1, col1)`` screen extent of world points."""

    width, height = int(resolution[0]), int(resolution[1])
    screen, _, valid = camera.project(np.asarray(points, dtype=np.float64), (width, height))
    if not np.any(valid):
        return None
    selected = np.asarray(screen)[np.asarray(valid)]
    col0 = max(0, math.floor(float(np.min(selected[:, 0]))))
    row0 = max(0, math.floor(float(np.min(selected[:, 1]))))
    col1 = min(width, math.ceil(float(np.max(selected[:, 0]))) + 1)
    row1 = min(height, math.ceil(float(np.max(selected[:, 1]))) + 1)
    if row0 >= row1 or col0 >= col1:
        return None
    return (row0, col0, row1, col1)


def _unit_vector(value: Any, *, name: str) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(values))
    if norm <= _EPS:
        raise ValueError(f"{name} must be nonzero")
    return np.asarray(values / norm, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PatchFrame:
    """Orthonormal patch frame: ``P(u, v) = origin + u·Tu + v·Tv + h·N``."""

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tangent_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    tangent_v: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        if origin.size != 3 or not np.all(np.isfinite(origin)):
            raise ValueError("patch origin must be a finite three-vector")
        u = _unit_vector(self.tangent_u, name="patch tangent_u")
        v = _unit_vector(self.tangent_v, name="patch tangent_v")
        if abs(float(u @ v)) > 1.0e-9:
            raise ValueError("patch tangents must be orthogonal")
        object.__setattr__(self, "origin", tuple(float(value) for value in origin))
        object.__setattr__(self, "tangent_u", tuple(float(value) for value in u))
        object.__setattr__(self, "tangent_v", tuple(float(value) for value in v))

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = np.asarray(self.tangent_u, dtype=np.float64)
        v = np.asarray(self.tangent_v, dtype=np.float64)
        return u, v, np.asarray(np.cross(u, v), dtype=np.float64)

    @property
    def origin_vector(self) -> np.ndarray:
        return np.asarray(self.origin, dtype=np.float64)

    def to_local(self, points: Any) -> np.ndarray:
        """Project world points to ``(u, v, h)`` patch coordinates."""

        u, v, n = self.basis
        delta = np.asarray(points, dtype=np.float64) - self.origin_vector
        return np.stack((delta @ u, delta @ v, delta @ n), axis=-1)

    def to_world(self, u: Any, v: Any, h: Any) -> np.ndarray:
        tu, tv, n = self.basis
        world = (
            self.origin_vector
            + np.asarray(u, dtype=np.float64)[..., None] * tu
            + np.asarray(v, dtype=np.float64)[..., None] * tv
            + np.asarray(h, dtype=np.float64)[..., None] * n
        )
        return np.asarray(world, dtype=np.float64)

    def world_normal(self, du: Any, dv: Any) -> np.ndarray:
        """World normal of ``h(u, v)`` from its local gradient (no buffer)."""

        tu, tv, n = self.basis
        gradient_u = np.asarray(du, dtype=np.float64)[..., None]
        gradient_v = np.asarray(dv, dtype=np.float64)[..., None]
        raw = n - gradient_u * tu - gradient_v * tv
        norm = np.linalg.norm(raw, axis=-1, keepdims=True)
        return np.asarray(raw / np.where(norm > _EPS, norm, 1.0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PatchHit:
    """Ray/patch intersection results, all arrays sharing a leading shape."""

    hit: np.ndarray
    t: np.ndarray
    position: np.ndarray
    normal: np.ndarray
    local: np.ndarray
    # Field evaluations per active ray, so a report can state realized work
    # rather than assuming the nominal step count.
    evaluations_per_ray: int = 0

    @property
    def count(self) -> int:
        return int(np.count_nonzero(self.hit))


@runtime_checkable
class AnalyticPatch(Protocol):
    """The §19.2 tier-1 surface: analytic value, analytic derivatives, a jet."""

    tier: GeometryTier
    primitive_id: int
    material_id: int

    def screen_bounds(self, camera: Any, resolution: tuple[int, int]) -> tuple[int, int, int, int] | None: ...

    def intersect(self, origins: Any, directions: Any, *, t_min: float, t_max: float) -> PatchHit: ...

    def log_cosine_jet(self, light: Any, *, origin: Sequence[float], guard_L: int = 8) -> v02.Jet2D: ...


class _LocalHeightPatch(ABC):
    """Shared machinery for patches expressed as ``h(u, v)`` over a frame."""

    tier: GeometryTier = GeometryTier.PATCH

    def __init__(
        self,
        frame: PatchFrame,
        domain: tuple[float, float, float, float],
        *,
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> None:
        u0, v0, u1, v1 = (float(value) for value in domain)
        if not (u1 > u0 and v1 > v0):
            raise ValueError("patch domain must be a non-empty (u0, v0, u1, v1) rectangle")
        self.frame = frame
        self.domain = (u0, v0, u1, v1)
        self.primitive_id = int(primitive_id)
        self.material_id = int(material_id)

    # -- analytic surface ---------------------------------------------------

    @abstractmethod
    def height(self, u: Any, v: Any) -> np.ndarray:
        """Local height ``h(u, v)``."""

    @abstractmethod
    def height_gradient(self, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
        """Analytic ``(∂h/∂u, ∂h/∂v)`` — the normal source, §19.2."""

    @abstractmethod
    def height_jet(self, u: float, v: float) -> _Jet2:
        """Exact 2-jet of ``h`` about ``(u, v)``."""

    @abstractmethod
    def height_range(self) -> tuple[float, float]:
        """Certified ``(h_min, h_max)`` over the whole domain, for §20.2."""

    def normal(self, u: Any, v: Any) -> np.ndarray:
        du, dv = self.height_gradient(u, v)
        return self.frame.world_normal(du, dv)

    def position(self, u: Any, v: Any) -> np.ndarray:
        return self.frame.to_world(u, v, self.height(u, v))

    def inside(self, u: Any, v: Any) -> np.ndarray:
        u0, v0, u1, v1 = self.domain
        uu = np.asarray(u, dtype=np.float64)
        vv = np.asarray(v, dtype=np.float64)
        return np.asarray((uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1))

    # -- visibility ---------------------------------------------------------

    def corner_points(self) -> np.ndarray:
        u0, v0, u1, v1 = self.domain
        h_min, h_max = self.height_range()
        us = np.asarray((u0, u1, u0, u1) * 2, dtype=np.float64)
        vs = np.asarray((v0, v0, v1, v1) * 2, dtype=np.float64)
        hs = np.concatenate((np.full(4, h_min), np.full(4, h_max)))
        return self.frame.to_world(us, vs, hs)

    def screen_bounds(self, camera: Any, resolution: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """Conservative screen rectangle from the patch's world bounding box."""

        return _screen_rect(camera, self.corner_points(), resolution)

    def intersect(self, origins: Any, directions: Any, *, t_min: float = 0.0, t_max: float = math.inf) -> PatchHit:
        """Ray/height intersection inside the certified height slab.

        The slab ``[h_min, h_max]`` clips the search interval in closed form
        first (§20.2 geometry bounds), so no march step is spent outside the
        patch.  A bracketed sign change of ``h_ray(t) - h(u(t), v(t))`` is then
        refined by a fixed number of bisection steps, which keeps the cost per
        ray constant and the result independent of evaluation order.
        """

        return _march_height(self, origins, directions, t_min=t_min, t_max=t_max)

    # -- field-plane lowering ----------------------------------------------

    def cosine_jet(self, light: Any, *, origin: Sequence[float]) -> _Jet2:
        """Exact 2-jet of the shading cosine ``N·L`` about ``origin``.

        The raw normal is ``n - h_u·Tu - h_v·Tv``, so the cosine is
        ``(n·L - h_u (Tu·L) - h_v (Tv·L)) / |raw|``.  Because ``h_u`` and
        ``h_v`` are themselves differentiated twice, an exact Hessian needs the
        patch's third derivatives; :meth:`height_third` supplies them (zero for
        a quadratic height, closed form for the polynomial and Fourier patches).
        """

        direction = _unit_vector(light, name="light direction")
        tu, tv, n = self.frame.basis
        u0, v0 = float(origin[0]), float(origin[1])
        h = self.height_jet(u0, v0)
        uuu, uuv, uvv, vvv = self.height_third(u0, v0)
        du = _Jet2(h.dx, h.dxx, h.dxy, uuu, uuv, uvv)
        dv = _Jet2(h.dy, h.dxy, h.dyy, uuv, uvv, vvv)
        numerator = _Jet2.constant(float(n @ direction)) - du * float(tu @ direction) - dv * float(tv @ direction)
        norm_square = _Jet2.constant(1.0) + du * du + dv * dv
        return numerator / norm_square.sqrt()

    def height_third(self, u: float, v: float) -> tuple[float, float, float, float]:
        """``(h_uuu, h_uuv, h_uvv, h_vvv)``; zero for a quadratic height."""

        return (0.0, 0.0, 0.0, 0.0)

    def log_cosine_jet(self, light: Any, *, origin: Sequence[float], guard_L: int = 8) -> v02.Jet2D:
        """``log2(N·L)`` as a widened LOG_POS :class:`spf.tensor.v02.Jet2D`.

        This is the §19.2 promise made concrete: the shading-relevant quantity
        reaches the field plane as header-expressible quadratic structure, not
        as a dense normal buffer.  Pair it with
        :func:`spf.tensor.v02.jet_to_tensor_block_header` for the ABI v23
        ``global_L_bias`` / ``band_exp`` / ``log_quadratic`` / ``log_mixed``
        form, or with :func:`spf.tensor.v02.jet_forward_differences` to emit a
        tile by additions alone.
        """

        return self.cosine_jet(light, origin=origin).log2().to_jet2d(value_kind="log_pos", guard_L=guard_L)

    def height_block_header(
        self,
        origin: Sequence[float],
        extent: Sequence[int],
        *,
        rounding: str = "nearest",
    ) -> tuple[Any, float]:
        """The height jet as ``TensorBlockHeaderV2`` log structure (ABI v23)."""

        jet = self.height_jet(float(origin[0]), float(origin[1])).to_jet2d(value_kind="log_pos", guard_L=0)
        return v02.jet_to_tensor_block_header(jet, tuple(int(value) for value in extent), rounding=rounding)


def _march_height(
    patch: _LocalHeightPatch,
    origins: Any,
    directions: Any,
    *,
    t_min: float,
    t_max: float,
    steps: int = 48,
    refinements: int = 24,
) -> PatchHit:
    origin = np.asarray(origins, dtype=np.float64)
    direction = np.asarray(directions, dtype=np.float64)
    if origin.shape[-1] != 3 or direction.shape[-1] != 3:
        raise ValueError("ray origins and directions must be three-vectors")
    origin, direction = np.broadcast_arrays(origin, direction)
    leading = origin.shape[:-1]
    tu, tv, n = patch.frame.basis
    base = patch.frame.origin_vector
    delta = origin - base
    o_h = delta @ n
    d_h = direction @ n
    h_min, h_max = patch.height_range()
    lo = np.full(leading, float(t_min), dtype=np.float64)
    hi = np.full(leading, float(t_max), dtype=np.float64)
    # Closed-form slab clip: h_min <= o_h + t d_h <= h_max.
    safe = np.abs(d_h) > _EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        t_low = (h_min - o_h) / np.where(safe, d_h, 1.0)
        t_high = (h_max - o_h) / np.where(safe, d_h, 1.0)
    enter = np.minimum(t_low, t_high)
    exit_ = np.maximum(t_low, t_high)
    inside_slab = (o_h >= h_min) & (o_h <= h_max)
    lo = np.where(safe, np.maximum(lo, enter), np.where(inside_slab, lo, np.inf))
    hi = np.where(safe, np.minimum(hi, exit_), np.where(inside_slab, hi, -np.inf))
    active = np.isfinite(lo) & (hi > lo)
    hit = np.zeros(leading, dtype=bool)
    t_hit = np.full(leading, math.inf, dtype=np.float64)
    if not np.any(active):
        return _empty_hit(leading, patch)
    lo_a, hi_a = lo[active], hi[active]
    o_a, d_a = origin[active], direction[active]

    def residual(ts: np.ndarray) -> np.ndarray:
        points = o_a[..., None, :] + ts[..., None] * d_a[..., None, :]
        local = points - base
        u = local @ tu
        v = local @ tv
        return np.asarray(local @ n - patch.height(u, v), dtype=np.float64)

    grid = np.linspace(0.0, 1.0, int(steps) + 1)
    ts = lo_a[..., None] + (hi_a - lo_a)[..., None] * grid
    values = residual(ts)
    signs = values <= 0.0
    crossing = signs[..., 1:] & ~signs[..., :-1]
    # Also accept a ray that starts already below the surface.
    below_at_entry = signs[..., 0]
    any_crossing = np.any(crossing, axis=-1)
    first = np.argmax(crossing, axis=-1)
    a = np.take_along_axis(ts, first[..., None], axis=-1)[..., 0]
    b = np.take_along_axis(ts, (first + 1)[..., None], axis=-1)[..., 0]
    a = np.where(below_at_entry, ts[..., 0], a)
    b = np.where(below_at_entry, ts[..., 0], b)
    found = any_crossing | below_at_entry
    for _ in range(int(refinements)):
        mid = 0.5 * (a + b)
        negative = residual(mid[..., None])[..., 0] <= 0.0
        b = np.where(negative, mid, b)
        a = np.where(negative, a, mid)
    t_found = 0.5 * (a + b)
    points = o_a + t_found[..., None] * d_a
    local = points - base
    u = local @ tu
    v = local @ tv
    inside = patch.inside(u, v)
    found = found & inside
    hit[active] = found
    t_hit[active] = np.where(found, t_found, math.inf)
    positions = np.zeros((*leading, 3), dtype=np.float64)
    normals = np.zeros((*leading, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    locals_ = np.zeros((*leading, 2), dtype=np.float64)
    positions[active] = points
    normals[active] = patch.normal(u, v)
    locals_[active] = np.stack((u, v), axis=-1)
    return PatchHit(hit=hit, t=t_hit, position=positions, normal=normals, local=locals_)


def _empty_hit(leading: tuple[int, ...], patch: _LocalHeightPatch) -> PatchHit:
    normals = np.zeros((*leading, 3), dtype=np.float64)
    normals[..., 2] = 1.0
    return PatchHit(
        hit=np.zeros(leading, dtype=bool),
        t=np.full(leading, math.inf, dtype=np.float64),
        position=np.zeros((*leading, 3), dtype=np.float64),
        normal=normals,
        local=np.zeros((*leading, 2), dtype=np.float64),
    )


class HeightFieldPatch(_LocalHeightPatch):
    """§19.2 height field whose height *is* a quadratic jet.

    ``h(u, v) = c + g·q + ½ qᵀHq``, which is exactly what
    :class:`spf.tensor.v02.Jet2D` stores, so :meth:`height_jet` is a
    re-centred copy rather than a fit and :meth:`height_block_header` is
    header-exact.
    """

    def __init__(
        self,
        jet: v02.Jet2D,
        *,
        frame: PatchFrame | None = None,
        domain: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0),
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> None:
        super().__init__(frame or PatchFrame(), domain, primitive_id=primitive_id, material_id=material_id)
        self.jet = jet
        self._c = float(jet.value)
        self._gx = float(jet.grad_x)
        self._gy = float(jet.grad_y)
        self._hxx = float(jet.hess_xx)
        self._hxy = float(jet.hess_xy)
        self._hyy = float(jet.hess_yy)

    def height(self, u: Any, v: Any) -> np.ndarray:
        uu = np.asarray(u, dtype=np.float64)
        vv = np.asarray(v, dtype=np.float64)
        return np.asarray(
            self._c
            + self._gx * uu
            + self._gy * vv
            + 0.5 * self._hxx * uu * uu
            + self._hxy * uu * vv
            + 0.5 * self._hyy * vv * vv,
            dtype=np.float64,
        )

    def height_gradient(self, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
        uu = np.asarray(u, dtype=np.float64)
        vv = np.asarray(v, dtype=np.float64)
        return (
            np.asarray(self._gx + self._hxx * uu + self._hxy * vv, dtype=np.float64),
            np.asarray(self._gy + self._hxy * uu + self._hyy * vv, dtype=np.float64),
        )

    def height_jet(self, u: float, v: float) -> _Jet2:
        du, dv = self.height_gradient(u, v)
        return _Jet2(float(self.height(u, v)), float(du), float(dv), self._hxx, self._hxy, self._hyy)

    def height_range(self) -> tuple[float, float]:
        u0, v0, u1, v1 = self.domain
        # A quadratic on a rectangle attains its extremes on the boundary or at
        # an interior stationary point; enumerate both, which is exact.
        candidates = [(u, v) for u in (u0, u1, 0.5 * (u0 + u1)) for v in (v0, v1, 0.5 * (v0 + v1))]
        determinant = self._hxx * self._hyy - self._hxy * self._hxy
        if abs(determinant) > _EPS:
            us = (self._hxy * self._gy - self._hyy * self._gx) / determinant
            vs = (self._hxy * self._gx - self._hxx * self._gy) / determinant
            if u0 <= us <= u1 and v0 <= vs <= v1:
                candidates.append((us, vs))
        values = [float(self.height(u, v)) for u, v in candidates]
        # Edge extrema of the restricted one-dimensional quadratics.
        for fixed_u in (u0, u1):
            if abs(self._hyy) > _EPS:
                v_star = -(self._gy + self._hxy * fixed_u) / self._hyy
                if v0 <= v_star <= v1:
                    values.append(float(self.height(fixed_u, v_star)))
        for fixed_v in (v0, v1):
            if abs(self._hxx) > _EPS:
                u_star = -(self._gx + self._hxy * fixed_v) / self._hxx
                if u0 <= u_star <= u1:
                    values.append(float(self.height(u_star, fixed_v)))
        return (min(values), max(values))


class PolynomialPatch(_LocalHeightPatch):
    """§19.2 polynomial patch: ``h(u, v) = Σ_{i,j} c_ij u^i v^j``.

    The default order is bi-quadratic (``c`` is 3×3); any square coefficient
    matrix is accepted and its derivatives stay analytic.
    """

    def __init__(
        self,
        coefficients: Any,
        *,
        frame: PatchFrame | None = None,
        domain: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0),
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> None:
        super().__init__(frame or PatchFrame(), domain, primitive_id=primitive_id, material_id=material_id)
        matrix = np.asarray(coefficients, dtype=np.float64)
        if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
            raise ValueError("polynomial patch coefficients must be a finite 2-D array")
        self.coefficients = matrix
        self._orders = (matrix.shape[0], matrix.shape[1])

    def _powers(self, values: np.ndarray, order: int, derivative: int) -> np.ndarray:
        exponents = np.arange(order, dtype=np.int64)
        scale = np.ones(order, dtype=np.float64)
        for step in range(derivative):
            scale = scale * np.maximum(exponents - step, 0.0)
        shifted = np.maximum(exponents - derivative, 0)
        table = np.power(values[..., None], shifted[None, :].reshape((1,) * values.ndim + (order,)))
        mask = exponents >= derivative
        return np.asarray(table * (scale * mask)[None, :].reshape((1,) * values.ndim + (order,)), dtype=np.float64)

    def _evaluate(self, u: Any, v: Any, *, du: int = 0, dv: int = 0) -> np.ndarray:
        uu = np.atleast_1d(np.asarray(u, dtype=np.float64))
        vv = np.atleast_1d(np.asarray(v, dtype=np.float64))
        shape = np.broadcast_shapes(uu.shape, vv.shape)
        uu = np.broadcast_to(uu, shape)
        vv = np.broadcast_to(vv, shape)
        u_table = self._powers(uu, self._orders[0], du)
        v_table = self._powers(vv, self._orders[1], dv)
        result = np.einsum("...i,ij,...j->...", u_table, self.coefficients, v_table)
        scalar = np.ndim(u) == 0 and np.ndim(v) == 0
        return np.asarray(result.reshape(()) if scalar else result, dtype=np.float64)

    def height(self, u: Any, v: Any) -> np.ndarray:
        return self._evaluate(u, v)

    def height_gradient(self, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
        return self._evaluate(u, v, du=1), self._evaluate(u, v, dv=1)

    def height_jet(self, u: float, v: float) -> _Jet2:
        return _Jet2(
            float(self._evaluate(u, v)),
            float(self._evaluate(u, v, du=1)),
            float(self._evaluate(u, v, dv=1)),
            float(self._evaluate(u, v, du=2)),
            float(self._evaluate(u, v, du=1, dv=1)),
            float(self._evaluate(u, v, dv=2)),
        )

    def height_third(self, u: float, v: float) -> tuple[float, float, float, float]:
        return (
            float(self._evaluate(u, v, du=3)),
            float(self._evaluate(u, v, du=2, dv=1)),
            float(self._evaluate(u, v, du=1, dv=2)),
            float(self._evaluate(u, v, dv=3)),
        )

    def height_range(self) -> tuple[float, float]:
        u0, v0, u1, v1 = self.domain
        # A certified envelope from the coefficient magnitudes on the domain
        # box: |h| <= Σ |c_ij| max|u|^i max|v|^j.  Conservative, never loose
        # enough to matter for a slab clip, and never wrong.
        bound_u = max(abs(u0), abs(u1))
        bound_v = max(abs(v0), abs(v1))
        total = 0.0
        for i in range(self._orders[0]):
            for j in range(self._orders[1]):
                total += abs(float(self.coefficients[i, j])) * (bound_u**i) * (bound_v**j)
        return (-total, total)


class FourierPatch(_LocalHeightPatch):
    """§19.2 local Fourier patch: ``h = Σ_m a_m cos(2π k_m·q) + b_m sin(...)``."""

    def __init__(
        self,
        frequencies: Any,
        cosines: Any,
        sines: Any,
        *,
        frame: PatchFrame | None = None,
        domain: tuple[float, float, float, float] = (-1.0, -1.0, 1.0, 1.0),
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> None:
        super().__init__(frame or PatchFrame(), domain, primitive_id=primitive_id, material_id=material_id)
        k = np.asarray(frequencies, dtype=np.float64).reshape(-1, 2)
        a = np.asarray(cosines, dtype=np.float64).reshape(-1)
        b = np.asarray(sines, dtype=np.float64).reshape(-1)
        if a.size != k.shape[0] or b.size != k.shape[0]:
            raise ValueError("Fourier patch needs one cosine and one sine weight per frequency")
        if not (np.all(np.isfinite(k)) and np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("Fourier patch parameters must be finite")
        self.frequencies = k
        self.cosines = a
        self.sines = b

    def _phase(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(
            2.0 * math.pi * (u[..., None] * self.frequencies[:, 0] + v[..., None] * self.frequencies[:, 1]),
            dtype=np.float64,
        )

    def height(self, u: Any, v: Any) -> np.ndarray:
        uu = np.asarray(u, dtype=np.float64)
        vv = np.asarray(v, dtype=np.float64)
        theta = self._phase(uu, vv)
        return np.asarray(np.sum(self.cosines * np.cos(theta) + self.sines * np.sin(theta), axis=-1), dtype=np.float64)

    def height_gradient(self, u: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
        uu = np.asarray(u, dtype=np.float64)
        vv = np.asarray(v, dtype=np.float64)
        theta = self._phase(uu, vv)
        common = -self.cosines * np.sin(theta) + self.sines * np.cos(theta)
        scale = 2.0 * math.pi
        return (
            np.asarray(np.sum(scale * self.frequencies[:, 0] * common, axis=-1), dtype=np.float64),
            np.asarray(np.sum(scale * self.frequencies[:, 1] * common, axis=-1), dtype=np.float64),
        )

    def height_jet(self, u: float, v: float) -> _Jet2:
        theta = self._phase(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        cos = np.cos(theta)
        sin = np.sin(theta)
        value = self.cosines * cos + self.sines * sin
        first = -self.cosines * sin + self.sines * cos
        scale = 2.0 * math.pi
        kx = self.frequencies[:, 0]
        ky = self.frequencies[:, 1]
        return _Jet2(
            float(np.sum(value)),
            float(np.sum(scale * kx * first)),
            float(np.sum(scale * ky * first)),
            float(np.sum(-(scale**2) * kx * kx * value)),
            float(np.sum(-(scale**2) * kx * ky * value)),
            float(np.sum(-(scale**2) * ky * ky * value)),
        )

    def height_third(self, u: float, v: float) -> tuple[float, float, float, float]:
        theta = self._phase(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64))
        first = -self.cosines * np.sin(theta) + self.sines * np.cos(theta)
        scale = 2.0 * math.pi
        kx = self.frequencies[:, 0]
        ky = self.frequencies[:, 1]
        # d³/dθ³ of (a cos θ + b sin θ) is -(first derivative).
        cube = -(scale**3)
        return (
            float(np.sum(cube * kx * kx * kx * first)),
            float(np.sum(cube * kx * kx * ky * first)),
            float(np.sum(cube * kx * ky * ky * first)),
            float(np.sum(cube * ky * ky * ky * first)),
        )

    def height_range(self) -> tuple[float, float]:
        total = float(np.sum(np.hypot(self.cosines, self.sines)))
        return (-total, total)


class QuadricPatch:
    """§19.2 implicit quadric ``xᵀA x + 2 bᵀx + c = 0`` with an exact normal.

    Ray intersection is the closed-form quadratic root, and the normal
    ``∇F = 2(Ax + b)`` needs no buffer.
    """

    tier: GeometryTier = GeometryTier.PATCH

    def __init__(
        self,
        matrix: Any,
        linear: Any = (0.0, 0.0, 0.0),
        constant: float = -1.0,
        *,
        primitive_id: int = 0,
        material_id: int = 0,
        center: Any = (0.0, 0.0, 0.0),
        radius: float = math.inf,
    ) -> None:
        a = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
        if not np.allclose(a, a.T, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("quadric matrix must be symmetric")
        b = np.asarray(linear, dtype=np.float64).reshape(-1)
        if b.size != 3 or not np.all(np.isfinite(b)) or not np.all(np.isfinite(a)):
            raise ValueError("quadric coefficients must be finite")
        self.matrix = a
        self.linear = b
        self.constant = float(constant)
        self.primitive_id = int(primitive_id)
        self.material_id = int(material_id)
        self.center = np.asarray(center, dtype=np.float64).reshape(-1)
        self.radius = float(radius)
        if self.center.size != 3:
            raise ValueError("quadric center must be a three-vector")

    def implicit(self, points: Any) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        return np.asarray(
            np.einsum("...i,ij,...j->...", x, self.matrix, x) + 2.0 * (x @ self.linear) + self.constant,
            dtype=np.float64,
        )

    def normal(self, points: Any) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        raw = 2.0 * (x @ self.matrix + self.linear)
        norm = np.linalg.norm(raw, axis=-1, keepdims=True)
        return np.asarray(raw / np.where(norm > _EPS, norm, 1.0), dtype=np.float64)

    def screen_bounds(self, camera: Any, resolution: tuple[int, int]) -> tuple[int, int, int, int] | None:
        if not math.isfinite(self.radius):
            return (0, 0, int(resolution[1]), int(resolution[0]))
        offsets = np.asarray(
            [
                (sx, sy, sz)
                for sx in (-self.radius, self.radius)
                for sy in (-self.radius, self.radius)
                for sz in (-self.radius, self.radius)
            ],
            dtype=np.float64,
        )
        return _screen_rect(camera, self.center + offsets, resolution)

    def intersect(self, origins: Any, directions: Any, *, t_min: float = 0.0, t_max: float = math.inf) -> PatchHit:
        o = np.asarray(origins, dtype=np.float64)
        d = np.asarray(directions, dtype=np.float64)
        o, d = np.broadcast_arrays(o, d)
        leading = o.shape[:-1]
        a2 = np.einsum("...i,ij,...j->...", d, self.matrix, d)
        a1 = 2.0 * (np.einsum("...i,ij,...j->...", d, self.matrix, o) + d @ self.linear)
        a0 = np.einsum("...i,ij,...j->...", o, self.matrix, o) + 2.0 * (o @ self.linear) + self.constant
        discriminant = a1 * a1 - 4.0 * a2 * a0
        quadratic = np.abs(a2) > _EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            root = np.sqrt(np.maximum(discriminant, 0.0))
            near = (-a1 - root) / (2.0 * np.where(quadratic, a2, 1.0))
            far = (-a1 + root) / (2.0 * np.where(quadratic, a2, 1.0))
            linear_t = -a0 / np.where(quadratic, 1.0, np.where(np.abs(a1) > _EPS, a1, 1.0))
        lower = np.minimum(near, far)
        upper = np.maximum(near, far)
        candidate = np.where(lower >= float(t_min), lower, upper)
        candidate = np.where(quadratic, candidate, linear_t)
        valid = np.where(quadratic, discriminant >= 0.0, np.abs(a1) > _EPS)
        hit = valid & (candidate >= float(t_min)) & (candidate <= float(t_max)) & np.isfinite(candidate)
        t = np.where(hit, candidate, math.inf)
        position = o + np.where(hit, t, 0.0)[..., None] * d
        normals = self.normal(position)
        local = position[..., :2] - self.center[:2]
        return PatchHit(
            hit=np.asarray(hit),
            t=np.asarray(t, dtype=np.float64),
            position=position,
            normal=normals,
            local=np.asarray(local, dtype=np.float64).reshape(*leading, 2),
        )

    def log_cosine_jet(self, light: Any, *, origin: Sequence[float], guard_L: int = 8) -> v02.Jet2D:
        """``log2(N·L)`` at a surface point, as a LOG_POS jet.

        The quadric's own ``(u, v)`` chart is the screen-aligned projection, so
        the cosine jet is expressed directly in the implicit gradient: exact
        value and gradient, curvature from ``A``.
        """

        direction = _unit_vector(light, name="light direction")
        point = np.asarray((origin[0], origin[1], float(origin[2]) if len(origin) > 2 else 0.0), dtype=np.float64)
        raw = 2.0 * (point @ self.matrix + self.linear)
        norm = float(np.linalg.norm(raw))
        if norm <= _EPS:
            raise ValueError("quadric gradient vanishes; no shading jet exists")
        cosine = float(raw @ direction) / norm
        if cosine <= 0.0:
            raise ValueError("quadric point faces away from the light; log2 cosine is undefined")
        # Curvature of the unit normal in the tangent plane is 2A/|∇F| projected
        # onto the light, which gives the quadratic term of log2(N·L) directly.
        curvature = 2.0 * (self.matrix @ direction) / norm
        return (
            _Jet2(
                cosine,
                float(curvature[0]),
                float(curvature[1]),
                float(2.0 * self.matrix[0, 0] / norm),
                float(2.0 * self.matrix[0, 1] / norm),
                float(2.0 * self.matrix[1, 1] / norm),
            )
            .log2()
            .to_jet2d(value_kind="log_pos", guard_L=guard_L)
        )


class SignedDistancePatch:
    """§19.2 signed distance field: a union of spheres, Lipschitz constant 1.

    ``f(x) = min_i (|x - c_i| - r_i)``.  Sphere tracing is exact because the
    field is 1-Lipschitz, and ``∇f`` is the unit radial direction of whichever
    sphere is active, so the normal is analytic.
    """

    tier: GeometryTier = GeometryTier.PATCH

    def __init__(
        self,
        centers: Any,
        radii: Any,
        *,
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> None:
        c = np.asarray(centers, dtype=np.float64).reshape(-1, 3)
        r = np.asarray(radii, dtype=np.float64).reshape(-1)
        if r.size != c.shape[0] or not np.all(np.isfinite(c)) or not np.all(np.isfinite(r)) or np.any(r <= 0.0):
            raise ValueError("signed distance patch needs one positive radius per center")
        self.centers = c
        self.radii = r
        self.primitive_id = int(primitive_id)
        self.material_id = int(material_id)

    def distance(self, points: Any) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        offsets = x[..., None, :] - self.centers
        return np.asarray(np.min(np.linalg.norm(offsets, axis=-1) - self.radii, axis=-1), dtype=np.float64)

    def gradient(self, points: Any) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        offsets = x[..., None, :] - self.centers
        lengths = np.linalg.norm(offsets, axis=-1)
        active = np.argmin(lengths - self.radii, axis=-1)
        selected = np.take_along_axis(offsets, active[..., None, None], axis=-2)[..., 0, :]
        length = np.take_along_axis(lengths, active[..., None], axis=-1)
        return np.asarray(selected / np.where(length > _EPS, length, 1.0), dtype=np.float64)

    normal = gradient

    def screen_bounds(self, camera: Any, resolution: tuple[int, int]) -> tuple[int, int, int, int] | None:
        corners = [
            center + np.asarray((sx, sy, sz), dtype=np.float64)
            for center, radius in zip(self.centers, self.radii, strict=True)
            for sx in (-radius, radius)
            for sy in (-radius, radius)
            for sz in (-radius, radius)
        ]
        return _screen_rect(camera, np.asarray(corners, dtype=np.float64), resolution)

    def intersect(
        self,
        origins: Any,
        directions: Any,
        *,
        t_min: float = 0.0,
        t_max: float = math.inf,
        steps: int = 96,
        tolerance: float = 1.0e-6,
    ) -> PatchHit:
        o = np.asarray(origins, dtype=np.float64)
        d = np.asarray(directions, dtype=np.float64)
        o, d = np.broadcast_arrays(o, d)
        leading = o.shape[:-1]
        limit = float(t_max)
        if not math.isfinite(limit):
            reach = float(np.max(np.linalg.norm(self.centers, axis=-1) + self.radii))
            limit = float(np.max(np.linalg.norm(o, axis=-1)) + 2.0 * reach + 1.0)
        t = np.full(leading, float(t_min), dtype=np.float64)
        hit = np.zeros(leading, dtype=bool)
        for _ in range(int(steps)):
            points = o + t[..., None] * d
            distance = self.distance(points)
            reached = (distance <= float(tolerance)) & ~hit & (t <= limit)
            hit |= reached
            advance = np.where(hit | (t > limit), 0.0, np.maximum(distance, float(tolerance)))
            t = t + advance
            if bool(np.all(hit | (t > limit))):
                break
        t = np.where(hit, t, math.inf)
        position = o + np.where(hit, t, 0.0)[..., None] * d
        return PatchHit(
            hit=hit,
            t=np.asarray(t, dtype=np.float64),
            position=position,
            normal=self.gradient(position),
            local=np.asarray(position[..., :2], dtype=np.float64).reshape(*leading, 2),
        )

    def log_cosine_jet(self, light: Any, *, origin: Sequence[float], guard_L: int = 8) -> v02.Jet2D:
        """``log2(N·L)`` on the sphere through ``origin``, a LOG_POS jet.

        On a sphere of radius ``r`` the unit normal is ``(x-c)/r`` exactly, so
        both derivatives of the cosine are closed form.
        """

        direction = _unit_vector(light, name="light direction")
        point = np.asarray(
            (float(origin[0]), float(origin[1]), float(origin[2]) if len(origin) > 2 else 0.0), dtype=np.float64
        )
        offsets = point - self.centers
        lengths = np.linalg.norm(offsets, axis=-1)
        active = int(np.argmin(lengths - self.radii))
        radius = float(self.radii[active])
        offset = offsets[active]
        length = float(lengths[active]) or radius
        normal = offset / length
        cosine = float(normal @ direction)
        if cosine <= 0.0:
            raise ValueError("sample faces away from the light; log2 cosine is undefined")
        return (
            _Jet2(
                cosine,
                float(direction[0] / radius),
                float(direction[1] / radius),
                float(-cosine / (radius * radius)),
                0.0,
                float(-cosine / (radius * radius)),
            )
            .log2()
            .to_jet2d(value_kind="log_pos", guard_L=guard_L)
        )


# ---------------------------------------------------------------------------
# Tier 2 — field geometry (§19.3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HermitianHeightField:
    """§19.3 displacement as ``h(x) = Re Σ_k H_k e^{i2πk·x}``.

    The stored ``amplitude_cos`` / ``amplitude_sin`` pair is the real form of
    the Hermitian spectrum: ``h = Σ_m (A_m cos θ_m - B_m sin θ_m)`` with
    ``θ_m = 2π(k_x x / Lx + k_y y / Ly)``.  Constructed from an ``rfft``-style
    spectrum by :meth:`from_spectrum`, it reproduces
    :func:`spf.renderer.ocean.sparse_signed_real_synthesis` on the grid to
    float64 rounding, and evaluates at *arbitrary* points off the grid — which
    is what a ray/height intersection needs and a dense ``irfft`` cannot give.

    Derivatives come from the spectrum itself (§28.1: ``∂̂_x h = i2πk_x H``, a
    log shift plus a quarter turn), so normals never need a buffer.
    """

    indices: np.ndarray
    amplitude_cos: np.ndarray
    amplitude_sin: np.ndarray
    grid_shape: tuple[int, int]
    extent: tuple[float, float] = (1.0, 1.0)
    base_height: float = 0.0
    primitive_id: int = 0
    material_id: int = 0

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64).reshape(-1, 2)
        cos = np.asarray(self.amplitude_cos, dtype=np.float64).reshape(-1)
        sin = np.asarray(self.amplitude_sin, dtype=np.float64).reshape(-1)
        if cos.size != indices.shape[0] or sin.size != indices.shape[0]:
            raise ValueError("one cosine and one sine amplitude per mode is required")
        if not (np.all(np.isfinite(cos)) and np.all(np.isfinite(sin))):
            raise ValueError("height-field amplitudes must be finite")
        shape = tuple(int(value) for value in self.grid_shape)
        if len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("grid_shape must be a positive (rows, columns) pair")
        extent = tuple(float(value) for value in self.extent)
        if len(extent) != 2 or any(not math.isfinite(value) or value <= 0.0 for value in extent):
            raise ValueError("extent must be a positive finite (Lx, Ly) pair")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "amplitude_cos", cos)
        object.__setattr__(self, "amplitude_sin", sin)
        object.__setattr__(self, "grid_shape", (shape[0], shape[1]))
        object.__setattr__(self, "extent", (extent[0], extent[1]))
        object.__setattr__(self, "base_height", float(self.base_height))
        object.__setattr__(self, "primitive_id", int(self.primitive_id))
        object.__setattr__(self, "material_id", int(self.material_id))

    tier = GeometryTier.FIELD

    # -- construction -------------------------------------------------------

    @classmethod
    def from_spectrum(
        cls,
        grid_shape: Sequence[int],
        indices: Any,
        coefficients: Any,
        *,
        extent: Sequence[float] | None = None,
        base_height: float = 0.0,
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> HermitianHeightField:
        """Build from unique Hermitian modes in the ``rfftn`` convention.

        ``coefficients`` may be complex, a ``(M, 2)`` real/imaginary array, or
        a sequence of ``(re, im)`` pairs.  The conjugate partner of each mode
        is folded in (factor two), except for a self-conjugate mode (every
        component in ``{0, N/2}``), matching
        :func:`spf.renderer.ocean.build_half_spectrum` and the ``1/N``
        normalization of ``irfftn``.
        """

        shape = tuple(int(value) for value in grid_shape)
        if len(shape) != 2:
            raise ValueError("HermitianHeightField is a two-dimensional field")
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1, 2) % np.asarray(shape, dtype=np.int64)
        values = np.asarray(coefficients)
        if values.dtype.kind == "c":
            real = np.asarray(values.real, dtype=np.float64).reshape(-1)
            imaginary = np.asarray(values.imag, dtype=np.float64).reshape(-1)
        else:
            pairs = np.asarray(values, dtype=np.float64).reshape(-1, 2)
            real, imaginary = pairs[:, 0], pairs[:, 1]
        count = int(np.prod(shape))
        self_conjugate = np.all((index_array == 0) | (index_array * 2 == np.asarray(shape)), axis=-1)
        weight = np.where(self_conjugate, 1.0, 2.0) / float(count)
        return cls(
            indices=index_array,
            amplitude_cos=real * weight,
            amplitude_sin=np.where(self_conjugate, 0.0, imaginary * weight),
            grid_shape=(shape[0], shape[1]),
            extent=(float(shape[1]), float(shape[0])) if extent is None else (float(extent[0]), float(extent[1])),
            base_height=base_height,
            primitive_id=primitive_id,
            material_id=material_id,
        )

    @classmethod
    def from_ocean_carrier(
        cls,
        carrier: Any,
        *,
        extent: Sequence[float] | None = None,
        base_height: float = 0.0,
        primitive_id: int = 0,
        material_id: int = 0,
    ) -> HermitianHeightField:
        """Build from an :class:`spf.renderer.ocean_carrier.OceanCarrier`.

        The carrier's own structural support is read: its mode indices plus
        either the declared coefficient values or, for a word-native carrier,
        the packed SPF32 ``(L, Θ)`` codes read as magnitude and phase.  The
        word route is a carrier read, not a Cartesian decode: no complex
        buffer is created and ``fallback_conversions`` stays zero.
        """

        shape = tuple(int(value) for value in carrier.shape)
        if len(shape) != 2:
            raise ValueError("only a two-dimensional ocean carrier is a height field")
        if carrier.coefficient_values and carrier.index_values:
            indices = np.asarray(carrier.index_values, dtype=np.int64)
            values = np.asarray([complex(item) for item in carrier.coefficient_values])
            return cls.from_spectrum(
                shape,
                indices,
                values,
                extent=extent,
                base_height=base_height,
                primitive_id=primitive_id,
                material_id=material_id,
            )
        index_array, words, _flags = carrier.materialize_sparse_packed()
        codes_L = np.asarray(v02.unpack_L(np.asarray(words, dtype=np.uint32)), dtype=np.int64)
        codes_T = np.asarray(v02.unpack_T(np.asarray(words, dtype=np.uint32)), dtype=np.int64)
        magnitude = np.asarray(
            [0.0 if int(code) == v02.SPF32_L_ZERO else v02.LogPos(int(code)).value for code in codes_L.reshape(-1)],
            dtype=np.float64,
        )
        angle = 2.0 * math.pi * np.asarray(codes_T, dtype=np.float64) / float(v02.SPF32_PHASE_MOD)
        pairs = np.stack((magnitude * np.cos(angle), magnitude * np.sin(angle)), axis=-1)
        return cls.from_spectrum(
            shape,
            np.asarray(index_array, dtype=np.int64),
            pairs,
            extent=extent,
            base_height=base_height,
            primitive_id=primitive_id,
            material_id=material_id,
        )

    # -- analytic field -----------------------------------------------------

    @property
    def mode_count(self) -> int:
        return int(self.indices.shape[0])

    def _signed_frequency(self) -> np.ndarray:
        shape = np.asarray(self.grid_shape, dtype=np.int64)
        # rows -> y, columns -> x; return (kx, ky) in cycles per world extent.
        wrapped = np.where(self.indices > shape // 2, self.indices - shape, self.indices)
        return np.stack((wrapped[:, 1], wrapped[:, 0]), axis=-1).astype(np.float64)

    def _theta(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        kx = self.indices[:, 1].astype(np.float64)
        ky = self.indices[:, 0].astype(np.float64)
        return np.asarray(
            2.0 * math.pi * (x[..., None] * kx / self.extent[0] + y[..., None] * ky / self.extent[1]),
            dtype=np.float64,
        )

    def height(self, x: Any, y: Any) -> np.ndarray:
        """``h`` at arbitrary world ``(x, y)``, realized only where asked."""

        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        theta = self._theta(xx, yy)
        return np.asarray(
            self.base_height + np.sum(self.amplitude_cos * np.cos(theta) - self.amplitude_sin * np.sin(theta), axis=-1),
            dtype=np.float64,
        )

    def gradient(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(∂h/∂x, ∂h/∂y)`` from the spectrum (§28.1 quarter turn)."""

        xx = np.asarray(x, dtype=np.float64)
        yy = np.asarray(y, dtype=np.float64)
        theta = self._theta(xx, yy)
        cos = np.cos(theta)
        sin = np.sin(theta)
        common = -self.amplitude_cos * sin - self.amplitude_sin * cos
        kx = 2.0 * math.pi * self.indices[:, 1].astype(np.float64) / self.extent[0]
        ky = 2.0 * math.pi * self.indices[:, 0].astype(np.float64) / self.extent[1]
        return (
            np.asarray(np.sum(kx * common, axis=-1), dtype=np.float64),
            np.asarray(np.sum(ky * common, axis=-1), dtype=np.float64),
        )

    def normal(self, x: Any, y: Any, *, up: Sequence[float] = (0.0, 1.0, 0.0)) -> np.ndarray:
        """Surface normal from the analytic gradient; no normal buffer."""

        dx, dy = self.gradient(x, y)
        axis = np.asarray(up, dtype=np.float64)
        vertical = int(np.argmax(np.abs(axis)))
        raw = np.zeros((*np.shape(dx), 3), dtype=np.float64)
        horizontal = [index for index in range(3) if index != vertical]
        raw[..., horizontal[0]] = -dx
        raw[..., horizontal[1]] = -dy
        raw[..., vertical] = 1.0
        norm = np.linalg.norm(raw, axis=-1, keepdims=True)
        return np.asarray(raw / np.where(norm > _EPS, norm, 1.0), dtype=np.float64)

    def magnitude_bound(self) -> float:
        """Certified ``max |h - base|`` — the §20.2 magnitude bound."""

        return float(np.sum(np.hypot(self.amplitude_cos, self.amplitude_sin)))

    def slab(self) -> tuple[float, float]:
        bound = self.magnitude_bound()
        return (self.base_height - bound, self.base_height + bound)

    def gradient_bound(self) -> float:
        """Certified ``max |∇h|`` — the Lipschitz constant for ray marching."""

        kx = 2.0 * math.pi * np.abs(self.indices[:, 1].astype(np.float64)) / self.extent[0]
        ky = 2.0 * math.pi * np.abs(self.indices[:, 0].astype(np.float64)) / self.extent[1]
        amplitude = np.hypot(self.amplitude_cos, self.amplitude_sin)
        return float(np.sum(amplitude * np.hypot(kx, ky)))

    # -- structured grid routes (reuse, never re-implement) ------------------

    def ocean_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """The ``(indices, coefficients)`` pair the ocean route consumes."""

        count = float(np.prod(self.grid_shape))
        self_conjugate = np.all((self.indices == 0) | (self.indices * 2 == np.asarray(self.grid_shape)), axis=-1)
        weight = np.where(self_conjugate, 1.0, 2.0) / count
        pairs = np.stack(
            (self.amplitude_cos / weight, np.where(self_conjugate, 0.0, self.amplitude_sin / weight)), axis=-1
        )
        return self.indices, pairs

    def synthesize_grid(self, *, output_kind: Any = "native", return_report: bool = False) -> Any:
        """Grid synthesis through the sparse Hermitian route (no FFT).

        Delegates to :func:`spf.renderer.ocean.sparse_signed_real_synthesis`
        (design §15.2 / §28.2 signed-real reduction), which is the library's
        Hermitian carrier path.
        """

        from spf.renderer.ocean import sparse_signed_real_synthesis

        indices, pairs = self.ocean_spectrum()
        coefficients = [complex(float(re), float(im)) for re, im in pairs]
        return sparse_signed_real_synthesis(
            coefficients,
            [tuple(int(value) for value in row) for row in indices],
            self.grid_shape,
            output_kind=output_kind,
            return_report=return_report,
        )

    def derivative_grid(self, axis: int = 0, *, return_report: bool = False) -> Any:
        """One spatial derivative through the §28.1 spectral quarter turn."""

        from spf.renderer.ocean import synthesize_derivative

        indices, pairs = self.ocean_spectrum()
        coefficients = [complex(float(re), float(im)) for re, im in pairs]
        spacing = (self.extent[1] / self.grid_shape[0], self.extent[0] / self.grid_shape[1])
        return synthesize_derivative(
            coefficients,
            [tuple(int(value) for value in row) for row in indices],
            self.grid_shape,
            axis=int(axis),
            spacing=spacing,
            return_report=return_report,
        )

    # -- ray/height intersection -------------------------------------------

    def intersect_rays(
        self,
        origins: Any,
        directions: Any,
        *,
        t_min: float = 0.0,
        t_max: float = math.inf,
        up: Sequence[float] = (0.0, 1.0, 0.0),
        steps: int = 64,
        refinements: int = 24,
        method: str = "scan",
        work_budget: int = 1 << 24,
    ) -> PatchHit:
        """Intersect rays with the height surface, §19.3 terrain/ocean route.

        The certified slab from :meth:`slab` clips each ray in closed form, so
        the march only ever touches the interval where an intersection can
        exist.  A ray whose slab interval is empty is never evaluated at all —
        that is where the avoidance ratio comes from for a horizon view.

        ``method="scan"`` brackets a sign change on a uniform grid and bisects
        it, which is robust for any spectrum.  ``method="lipschitz"`` uses the
        certified gradient bound instead (see :meth:`_lipschitz_march`) and
        typically needs a fraction of the evaluations; the returned
        ``PatchHit.evaluations_per_ray`` reports which cost was actually paid.

        ``work_budget`` caps ``rays × steps × modes`` per vectorized block, so a
        4K frame marches in bounded memory instead of allocating one array per
        step for the whole screen.  Blocking changes no result: every ray is
        independent.
        """

        o = np.asarray(origins, dtype=np.float64)
        d = np.asarray(directions, dtype=np.float64)
        o, d = np.broadcast_arrays(o, d)
        if len(o.shape) == 2 and int(o.shape[0]) > 0:
            per_ray = max(1, (int(steps) + 1) * max(1, self.mode_count))
            block = max(1, int(work_budget) // per_ray)
            if block < int(o.shape[0]):
                parts = [
                    self.intersect_rays(
                        o[start : start + block],
                        d[start : start + block],
                        t_min=t_min,
                        t_max=t_max,
                        up=up,
                        steps=steps,
                        refinements=refinements,
                        method=method,
                        work_budget=work_budget,
                    )
                    for start in range(0, int(o.shape[0]), block)
                ]
                return PatchHit(
                    hit=np.concatenate([part.hit for part in parts]),
                    t=np.concatenate([part.t for part in parts]),
                    position=np.concatenate([part.position for part in parts]),
                    normal=np.concatenate([part.normal for part in parts]),
                    local=np.concatenate([part.local for part in parts]),
                    evaluations_per_ray=max(part.evaluations_per_ray for part in parts),
                )
        leading = o.shape[:-1]
        axis = np.asarray(up, dtype=np.float64)
        vertical = int(np.argmax(np.abs(axis)))
        horizontal = [index for index in range(3) if index != vertical]
        h_min, h_max = self.slab()
        o_h = o[..., vertical]
        d_h = d[..., vertical]
        safe = np.abs(d_h) > _EPS
        with np.errstate(divide="ignore", invalid="ignore"):
            t_low = (h_min - o_h) / np.where(safe, d_h, 1.0)
            t_high = (h_max - o_h) / np.where(safe, d_h, 1.0)
        enter = np.minimum(t_low, t_high)
        exit_ = np.maximum(t_low, t_high)
        inside = (o_h >= h_min) & (o_h <= h_max)
        lo = np.where(safe, np.maximum(float(t_min), enter), np.where(inside, float(t_min), np.inf))
        hi = np.where(safe, np.minimum(float(t_max), exit_), np.where(inside, float(t_max), -np.inf))
        active = np.isfinite(lo) & np.isfinite(hi) & (hi > lo)
        hit = np.zeros(leading, dtype=bool)
        t_hit = np.full(leading, math.inf, dtype=np.float64)
        position = np.zeros((*leading, 3), dtype=np.float64)
        normal = np.zeros((*leading, 3), dtype=np.float64)
        normal[..., vertical] = 1.0
        local = np.zeros((*leading, 2), dtype=np.float64)
        if not np.any(active):
            return PatchHit(hit=hit, t=t_hit, position=position, normal=normal, local=local)
        o_a, d_a, lo_a, hi_a = o[active], d[active], lo[active], hi[active]

        def residual(ts: np.ndarray) -> np.ndarray:
            points = o_a[..., None, :] + ts[..., None] * d_a[..., None, :]
            return np.asarray(
                points[..., vertical] - self.height(points[..., horizontal[0]], points[..., horizontal[1]]),
                dtype=np.float64,
            )

        selected_method = str(method).lower()
        if selected_method not in {"scan", "lipschitz", "newton"}:
            raise ValueError("method must be 'scan', 'newton', or 'lipschitz'")
        if selected_method == "lipschitz":
            t_found, found, evaluations = self._lipschitz_march(
                residual,
                lo_a,
                hi_a,
                np.hypot(d_a[..., horizontal[0]], d_a[..., horizontal[1]]),
                np.abs(d_a[..., vertical]),
                steps=int(steps) + int(refinements),
            )
        else:
            grid = np.linspace(0.0, 1.0, int(steps) + 1)
            ts = lo_a[..., None] + (hi_a - lo_a)[..., None] * grid
            values = residual(ts)
            below = values <= 0.0
            crossing = below[..., 1:] & ~below[..., :-1]
            entry_below = below[..., 0]
            first = np.argmax(crossing, axis=-1)
            found = np.any(crossing, axis=-1) | entry_below
            a = np.take_along_axis(ts, first[..., None], axis=-1)[..., 0]
            b = np.take_along_axis(ts, (first + 1)[..., None], axis=-1)[..., 0]
            a = np.where(entry_below, ts[..., 0], a)
            b = np.where(entry_below, ts[..., 0], b)
            if selected_method == "newton":
                # Safeguarded Newton on the bracket using the *analytic* ray
                # derivative f'(t) = d_z - ∇h · d_horizontal.  ∇h comes from the
                # spectrum (§28.1), which a sampled height grid does not carry,
                # so this convergence rate is a property of the representation.
                t_found = 0.5 * (a + b)
                for _ in range(max(1, int(refinements))):
                    point = o_a + t_found[..., None] * d_a
                    x = point[..., horizontal[0]]
                    y = point[..., horizontal[1]]
                    value = point[..., vertical] - self.height(x, y)
                    gx, gy = self.gradient(x, y)
                    slope = d_a[..., vertical] - (gx * d_a[..., horizontal[0]] + gy * d_a[..., horizontal[1]])
                    b = np.where(value <= 0.0, t_found, b)
                    a = np.where(value <= 0.0, a, t_found)
                    step = np.where(np.abs(slope) > _EPS, value / np.where(np.abs(slope) > _EPS, slope, 1.0), 0.0)
                    candidate = t_found - step
                    inside = (candidate > a) & (candidate < b)
                    t_found = np.where(inside, candidate, 0.5 * (a + b))
                # Two height-equivalent evaluations per Newton step (value and
                # gradient share the phase table but not its sine/cosine pair).
                evaluations = int(steps) + 1 + 2 * max(1, int(refinements))
            else:
                for _ in range(int(refinements)):
                    mid = 0.5 * (a + b)
                    negative = residual(mid[..., None])[..., 0] <= 0.0
                    b = np.where(negative, mid, b)
                    a = np.where(negative, a, mid)
                t_found = 0.5 * (a + b)
                evaluations = int(steps) + 1 + int(refinements)
        points = o_a + t_found[..., None] * d_a
        hit[active] = found
        t_hit[active] = np.where(found, t_found, math.inf)
        position[active] = points
        normal[active] = self.normal(points[..., horizontal[0]], points[..., horizontal[1]], up=up)
        local[active] = np.stack((points[..., horizontal[0]], points[..., horizontal[1]]), axis=-1)
        return PatchHit(
            hit=hit,
            t=t_hit,
            position=position,
            normal=normal,
            local=local,
            evaluations_per_ray=evaluations,
        )

    def _lipschitz_march(
        self,
        residual: Callable[[np.ndarray], np.ndarray],
        lo: np.ndarray,
        hi: np.ndarray,
        horizontal_speed: np.ndarray,
        vertical_speed: np.ndarray,
        *,
        steps: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Safe march using the certified spectral gradient bound.

        ``|f'(t)| <= |d_z| + L·|d_horizontal|`` with ``L`` from
        :meth:`gradient_bound`, so stepping by ``f(t) / (|d_z| + L|d_h|)``
        cannot step past a crossing.  This is sphere tracing for a height
        field, and it needs no bracketing scan: the bound is a property of the
        Hermitian spectrum, which a sampled height grid does not carry.
        """

        lipschitz = self.gradient_bound()
        rate = np.maximum(vertical_speed + lipschitz * horizontal_speed, _EPS)
        t = lo.copy()
        found = np.zeros(lo.shape, dtype=bool)
        result = hi.copy()
        evaluations = 0
        for _ in range(max(1, int(steps))):
            value = residual(t[..., None])[..., 0]
            evaluations += 1
            crossed = (value <= 0.0) & ~found
            result = np.where(crossed, t, result)
            found = found | crossed
            done = found | (t >= hi)
            t = np.where(done, t, np.minimum(hi, t + np.maximum(value / rate, _EPS)))
            if bool(np.all(done)):
                break
        return result, found, evaluations


# ---------------------------------------------------------------------------
# Tier 3 — Gaussian primitives (§19.4, §27)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GaussianGeometry:
    """§19.4/§27 tier-3 geometry: 3-D Gaussians on the log-alpha jet route.

    This is a thin tier adapter over :mod:`spf.renderer.splats`: projection
    (§27.1), depth sort (§27.4), tile binning (§27.2) and the
    ``ProjectedSplatTile`` log-alpha jet headers are that module's, and are
    reused rather than rebuilt.  What this class adds is the tier contract —
    a :class:`VisibleSampleBuffer` and the §20.2 cull accounting.
    """

    splats: tuple[Any, ...]
    tile_size: int = 16
    min_alpha: float = 1.0e-4
    guard_bits: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "splats", tuple(self.splats))
        if int(self.tile_size) <= 0:
            raise ValueError("tile_size must be positive")
        object.__setattr__(self, "tile_size", int(self.tile_size))
        object.__setattr__(self, "min_alpha", float(self.min_alpha))
        object.__setattr__(self, "guard_bits", int(self.guard_bits))

    tier = GeometryTier.GAUSSIAN

    def project(self, camera: Any) -> Any:
        from spf.renderer.splats import bin_splats

        return bin_splats(
            self.splats,
            camera,
            tile_size=self.tile_size,
            min_alpha=self.min_alpha,
            guard_L=self.guard_bits,
        )

    def cull(self, camera: Any, *, bins: Any = None) -> TileCullResult:
        """Report §20.2 tile occupancy as a cull result for the frame report."""

        selected = bins if bins is not None else self.project(camera)
        shape = (int(selected.image_shape[0]), int(selected.image_shape[1]))
        occupied = {tile.index for tile in selected.tiles if tile.entries}
        alpha_bounds = {
            tile.index: max((entry.max_alpha for entry in tile.entries), default=0.0) for tile in selected.tiles
        }
        records = tuple(
            CullRecord(
                tile=region,
                culled=region.index not in occupied,
                reason="kept" if region.index in occupied else "opacity_bound",
                geometry_overlap=region.index in occupied,
                magnitude_bound=float(alpha_bounds.get(region.index, 0.0)),
            )
            for region in tile_regions(shape, int(selected.tile_size))
        )
        return TileCullResult(shape=shape, tile_size=int(selected.tile_size), records=records)

    def render(self, camera: Any, **options: Any) -> Any:
        """Render through the SPF log-alpha transport path (§27.3/§27.5)."""

        from spf.renderer.splats import rasterize_splats_spf

        return rasterize_splats_spf(
            self.splats,
            camera,
            tile_size=self.tile_size,
            min_alpha=self.min_alpha,
            guard_L=self.guard_bits,
            **options,
        )

    def visible_samples(self, camera: Any, **options: Any) -> tuple[VisibleSampleBuffer, Any]:
        """Render and publish the §19.5 buffer alongside the splat result."""

        result = self.render(camera, **options)
        shape = (int(result.color.shape[0]), int(result.color.shape[1]))
        material_id = np.where(result.primitive_id >= 0, result.primitive_id, -1).astype(np.int64)
        rows, columns = np.mgrid[0 : shape[0], 0 : shape[1]]
        local = np.stack((columns.astype(np.float64), rows.astype(np.float64)), axis=-1)
        normal = np.zeros((*shape, 3), dtype=np.float64)
        normal[..., 2] = 1.0
        buffer = VisibleSampleBuffer(
            tier=GeometryTier.GAUSSIAN,
            depth=np.asarray(result.depth, dtype=np.float64),
            coverage=np.asarray(result.coverage, dtype=np.float64),
            primitive_id=np.asarray(result.primitive_id, dtype=np.int64),
            material_id=material_id,
            local_coordinates=local,
            position=np.zeros((*shape, 3), dtype=np.float64),
            normal=normal,
            field_bindings={"alpha": np.asarray(result.alpha, dtype=np.float64)},
            potential_elements=int(shape[0]) * int(shape[1]) * max(1, len(self.splats)),
        )
        return buffer, result


# ---------------------------------------------------------------------------
# field plane: log-plane shading of a visible-sample buffer
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShadingResult:
    """One materialization of a shaded sample buffer plus its counters."""

    image: np.ndarray
    words: np.ndarray
    positive_logadds: int
    pivot_reductions: int
    reduction_error_bound: float
    materialized_elements: int
    lights: int


def shade_samples(
    samples: VisibleSampleBuffer,
    lights: Sequence[Any],
    *,
    material: Material,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    error_budget: float | None = None,
) -> ShadingResult:
    """Shade the covered samples on LOG_POS word planes, materializing once.

    The field plane starts here, after visibility.  Per light the term is
    ``radiance · albedo · max(N·L, 0)`` built by
    :meth:`spf.tensor.v02_planes.LogPosPlane.mul` (integer adds on L codes),
    the lights are stacked and reduced once by the positive pivot reducer
    (§23.1), and ``to_values`` is the single materialization (§16.2).  Only
    covered pixels enter the planes, so the reducer's width is the realized
    sample count, not the frame.
    """

    from spf.renderer.lighting import light_direction, sample_light

    height, width = samples.shape
    image = np.broadcast_to(np.asarray(background, dtype=np.float64).reshape(3), (height, width, 3)).copy()
    mask = samples.visible_mask
    realized = int(np.count_nonzero(mask))
    if realized == 0:
        return ShadingResult(image, np.zeros((0,), dtype=np.uint32), 0, 0, 0.0, 0, 0)
    normals = samples.normal[mask]
    albedo = LogPosPlane.from_values(np.abs(material.base_color_array).reshape(3))
    terms: list[np.ndarray] = []
    emission = np.asarray(material.emission_array, dtype=np.float64).reshape(3)
    positive_logadds = 0
    for light in lights:
        direction = np.asarray(light_direction(light, None, fallback=(0.0, 0.0, 1.0)), dtype=np.float64)
        cosine = np.maximum(normals @ direction, 0.0)
        radiance = np.abs(np.asarray(sample_light(light, direction), dtype=np.float64).reshape(3))
        cosine_plane = LogPosPlane.from_values(np.repeat(cosine[:, None], 3, axis=1))
        radiance_plane = LogPosPlane.from_values(np.broadcast_to(radiance, (realized, 3)))
        term = cosine_plane.mul(albedo).mul(radiance_plane)
        positive_logadds += 2 * term.size
        terms.append(term.L)
    if not terms:
        terms.append(LogPosPlane.from_values(np.broadcast_to(np.abs(material.base_color_array), (realized, 3))).L)
    stacked = LogPosPlane(np.stack(terms))
    reduction = reduce_logpos(stacked, axis=0, spec=error_budget)
    shaded = np.asarray(reduction.value.to_values(), dtype=np.float64) + emission
    image[mask] = shaded
    return ShadingResult(
        image=image,
        words=np.asarray(reduction.value.to_words(), dtype=np.uint32),
        positive_logadds=positive_logadds + int(reduction.counters.positive_logadds),
        pivot_reductions=int(reduction.counters.pivot_reductions),
        reduction_error_bound=float(reduction.total_error_bound),
        materialized_elements=int(shaded.size),
        lights=len(terms),
    )


# ---------------------------------------------------------------------------
# tier routing: visibility -> VisibleSample -> field plane -> one display
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeometryLayer:
    """A tier's rendered layer: image, depth, coverage, samples, report."""

    tier: GeometryTier
    image: np.ndarray
    depth: np.ndarray
    coverage: np.ndarray
    samples: VisibleSampleBuffer
    report: RenderReport
    cull: TileCullResult | None = None

    @property
    def color(self) -> np.ndarray:
        return self.image

    @property
    def materialization_avoidance(self) -> float:
        return float(self.report.metrics.materialization_avoidance)


def _camera_rays(camera: Any, resolution: tuple[int, int], mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized world rays for the pixels selected by ``mask``."""

    width, height = int(resolution[0]), int(resolution[1])
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    right, up, forward = camera.basis
    aspect = float(camera.aspect) if getattr(camera, "aspect", None) is not None else width / height
    ndc_x = ((columns.astype(np.float64) + 0.5) / width) * 2.0 - 1.0
    ndc_y = 1.0 - ((rows.astype(np.float64) + 0.5) / height) * 2.0
    if bool(getattr(camera, "orthographic", False)):
        half = float(camera.ortho_scale) * 0.5
        origins = camera.position_vector + ndc_x[:, None] * (half * aspect) * right + ndc_y[:, None] * half * up
        directions = np.broadcast_to(forward, origins.shape).copy()
        return origins, directions
    tangent = math.tan(math.radians(float(camera.fov_y)) * 0.5)
    raw = forward + ndc_x[:, None] * (tangent * aspect) * right + ndc_y[:, None] * tangent * up
    norm = np.linalg.norm(raw, axis=-1, keepdims=True)
    directions = raw / np.where(norm > _EPS, norm, 1.0)
    origins = np.broadcast_to(camera.position_vector, directions.shape).copy()
    return origins, directions


def _tier_report(
    *,
    tier: GeometryTier,
    resolution: tuple[int, int],
    representation: str,
    potential: int,
    materialized: int,
    shading: ShadingResult,
    cull: TileCullResult | None,
    ledger: v02.ErrorLedger,
    diagnostics: Mapping[str, Any],
    callable_name: str,
) -> RenderReport:
    width, height = resolution
    metrics = RenderMetrics(
        backend="reference",
        workload=str(tier.value),
        resolution=(width, height),
        representation=representation,
        precision="balanced",
        potential_elements=int(potential),
        materialized_elements=int(materialized),
        positive_logadds=int(shading.positive_logadds),
        pivot_reductions=int(shading.pivot_reductions),
        fft_count=0,
        fallback_conversions=0,
        field_operations=int(materialized),
        reduction_operations=int(shading.pivot_reductions),
        potential_intermediate_elements=int(potential),
        materialized_intermediate_elements=int(materialized),
        terminal_output_elements=int(width * height * 3),
    )
    details: dict[str, Any] = {
        "execution_class": "reference_python",
        "actual_execution_class": "reference_python",
        "actual_callable": callable_name,
        "hardware_native": False,
        "tier": str(tier.value),
        "materialization_avoidance": metrics.materialization_avoidance,
        "lights": int(shading.lights),
    }
    if cull is not None:
        details.update(
            {
                "tiles": len(cull.records),
                "tiles_kept": len(cull.kept),
                "tiles_culled": len(cull.culled),
                "tile_size": cull.tile_size,
                "tile_avoidance": cull.materialization_avoidance,
            }
        )
    details.update(dict(diagnostics))
    return RenderReport(
        metrics=metrics,
        ledger=ledger,
        passes=(
            CompilerPassRecord("tier_visibility", 1, 1, notes=("fixed-point z* = min z", "tile cull before field")),
            CompilerPassRecord("tier_field_plane", 1, 1, notes=("LOG_POS planes", "one materialization")),
        ),
        decisions=(
            RepresentationDecision(
                str(tier.value),
                "screen",
                representation,
                (representation, "dense_reference"),
                float(materialized),
                float(ledger.total_bound),
                "tier routing after visibility",
            ),
        ),
        capabilities_used=("VISIBLE_SAMPLE", "TILE_CULL", "LOG_POS_PLANE", "PIVOT_REDUCE"),
        diagnostics=details,
    )


def rasterize_patch(
    patch: Any,
    camera: Any,
    resolution: tuple[int, int],
    *,
    lights: Sequence[Any] = (),
    material: Material | None = None,
    tile_size: int = 32,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    min_contribution: float = 0.0,
    error_budget: float | None = None,
) -> GeometryLayer:
    """Tier-1 route: cull tiles, intersect, publish samples, shade once."""

    width, height = int(resolution[0]), int(resolution[1])
    shape = (height, width)
    selected_material = material or Material.pbr(base_color=(0.7, 0.7, 0.72), roughness=0.5, name="patch")
    bounds = patch.screen_bounds(camera, (width, height))
    cull = cull_tiles(
        shape,
        tile_size,
        geometry_bounds=() if bounds is None else (bounds,),
        min_contribution=min_contribution,
    )
    mask = cull.kept_mask()
    samples = VisibleSampleBuffer.empty(shape, GeometryTier.PATCH, potential_elements=height * width)
    origins, directions = _camera_rays(camera, (width, height), mask)
    if origins.shape[0]:
        hit = patch.intersect(origins, directions, t_min=float(getattr(camera, "near", 0.0)))
        rows, columns = np.nonzero(mask)
        depth = samples.depth.copy()
        coverage = samples.coverage.copy()
        primitive = samples.primitive_id.copy()
        material_ids = samples.material_id.copy()
        local = samples.local_coordinates.copy()
        position = samples.position.copy()
        normal = samples.normal.copy()
        selected = np.asarray(hit.hit)
        r, c = rows[selected], columns[selected]
        _, _, forward = camera.basis
        world = hit.position[selected]
        depth[r, c] = (world - camera.position_vector) @ forward
        coverage[r, c] = 1.0
        primitive[r, c] = int(patch.primitive_id)
        material_ids[r, c] = int(getattr(patch, "material_id", 0))
        local[r, c] = hit.local[selected]
        position[r, c] = world
        normal[r, c] = hit.normal[selected]
        samples = VisibleSampleBuffer(
            tier=GeometryTier.PATCH,
            depth=depth,
            coverage=coverage,
            primitive_id=primitive,
            material_id=material_ids,
            local_coordinates=local,
            position=position,
            normal=normal,
            potential_elements=height * width,
        )
    shading = shade_samples(
        samples, lights, material=selected_material, background=background, error_budget=error_budget
    )
    report = _tier_report(
        tier=GeometryTier.PATCH,
        resolution=(width, height),
        representation="analytic_patch_jet",
        potential=height * width,
        materialized=max(1, samples.realized_elements),
        shading=shading,
        cull=cull,
        ledger=v02.ErrorLedger(reducer=shading.reduction_error_bound),
        diagnostics={
            "patch": type(patch).__name__,
            "screen_bounds": bounds,
            "rays_traced": int(origins.shape[0]),
        },
        callable_name="spf.renderer.geometry.rasterize_patch",
    )
    return GeometryLayer(
        tier=GeometryTier.PATCH,
        image=shading.image,
        depth=samples.depth,
        coverage=samples.coverage,
        samples=samples,
        report=report,
        cull=cull,
    )


def rasterize_height_field(
    field_geometry: HermitianHeightField,
    camera: Any,
    resolution: tuple[int, int],
    *,
    lights: Sequence[Any] = (),
    material: Material | None = None,
    tile_size: int = 32,
    background: Sequence[float] = (0.0, 0.0, 0.0),
    up: Sequence[float] = (0.0, 1.0, 0.0),
    steps: int = 64,
    refinements: int = 20,
    method: str = "scan",
    error_budget: float | None = None,
) -> GeometryLayer:
    """Tier-2 route: slab-cull tiles, march only inside, shade once.

    The certified height slab is projected per tile: a tile whose whole ray
    bundle misses the slab is dropped before a single ``h`` evaluation, and
    only the survivors pay the march.  Normals come from the spectrum, so no
    normal buffer is ever built.

    The per-tile slab test is *exact*, not merely conservative.  Whether a ray
    can reach the slab at ``t >= 0`` depends only on the sign of the vertical
    ray component and on the ray's vertical origin, and both are affine in the
    pixel coordinate, so their extremes over a tile are attained at the tile's
    four corner pixels.  Testing those corners therefore cannot drop a tile
    that contains a reachable ray.
    """

    width, height = int(resolution[0]), int(resolution[1])
    shape = (height, width)
    selected_material = material or Material.pbr(base_color=(0.28, 0.42, 0.55), roughness=0.4, name="height_field")
    h_min, h_max = field_geometry.slab()
    axis = np.asarray(up, dtype=np.float64)
    vertical = int(np.argmax(np.abs(axis)))

    def tile_slab_overlap(region: TileRegion) -> float:
        row0, col0, row1, col1 = region.bounds
        corners = np.asarray([(col0, row0), (col1 - 1, row0), (col0, row1 - 1), (col1 - 1, row1 - 1)], dtype=np.float64)
        reachable = 0.0
        for pixel in corners:
            origin, direction = camera.ray(pixel, (width, height))
            o_h = float(origin[vertical])
            d_h = float(direction[vertical])
            if abs(d_h) <= _EPS:
                if h_min <= o_h <= h_max:
                    reachable = 1.0
            else:
                t_low = (h_min - o_h) / d_h
                t_high = (h_max - o_h) / d_h
                if max(min(t_low, t_high), 0.0) <= max(t_low, t_high):
                    reachable = 1.0
        return reachable

    cull = cull_tiles(shape, tile_size, magnitude_bound=tile_slab_overlap)
    mask = cull.kept_mask()
    # §16.3 potential: the same per-ray field work a dense full-screen march
    # would pay.  Realized: only the rays that survived the slab cull.
    per_ray = int(steps) + 1 + int(refinements)
    potential = height * width * per_ray
    samples = VisibleSampleBuffer.empty(shape, GeometryTier.FIELD, potential_elements=potential)
    origins, directions = _camera_rays(camera, (width, height), mask)
    march_samples = 0
    if origins.shape[0]:
        hit = field_geometry.intersect_rays(
            origins,
            directions,
            t_min=float(getattr(camera, "near", 0.0)),
            up=up,
            steps=steps,
            refinements=refinements,
            method=method,
        )
        march_samples = int(origins.shape[0]) * max(1, int(hit.evaluations_per_ray))
        rows, columns = np.nonzero(mask)
        depth = samples.depth.copy()
        coverage = samples.coverage.copy()
        primitive = samples.primitive_id.copy()
        material_ids = samples.material_id.copy()
        local = samples.local_coordinates.copy()
        position = samples.position.copy()
        normal = samples.normal.copy()
        selected = np.asarray(hit.hit)
        r, c = rows[selected], columns[selected]
        _, _, forward = camera.basis
        world = hit.position[selected]
        depth[r, c] = (world - camera.position_vector) @ forward
        coverage[r, c] = 1.0
        primitive[r, c] = int(field_geometry.primitive_id)
        material_ids[r, c] = int(field_geometry.material_id)
        local[r, c] = hit.local[selected]
        position[r, c] = world
        normal[r, c] = hit.normal[selected]
        samples = VisibleSampleBuffer(
            tier=GeometryTier.FIELD,
            depth=depth,
            coverage=coverage,
            primitive_id=primitive,
            material_id=material_ids,
            local_coordinates=local,
            position=position,
            normal=normal,
            field_bindings={"height": position[..., vertical]},
            potential_elements=potential,
        )
    shading = shade_samples(
        samples, lights, material=selected_material, background=background, error_budget=error_budget
    )
    report = _tier_report(
        tier=GeometryTier.FIELD,
        resolution=(width, height),
        representation="hermitian_height_field",
        potential=potential,
        materialized=max(1, march_samples),
        shading=shading,
        cull=cull,
        ledger=v02.ErrorLedger(reducer=shading.reduction_error_bound),
        diagnostics={
            "modes": field_geometry.mode_count,
            "slab": (h_min, h_max),
            "gradient_bound": field_geometry.gradient_bound(),
            "march_steps": int(steps),
            "march_method": str(method),
            "march_samples": march_samples,
            "evaluations_per_ray": int(per_ray),
            "rays_traced": int(origins.shape[0]),
        },
        callable_name="spf.renderer.geometry.rasterize_height_field",
    )
    return GeometryLayer(
        tier=GeometryTier.FIELD,
        image=shading.image,
        depth=samples.depth,
        coverage=samples.coverage,
        samples=samples,
        report=report,
        cull=cull,
    )


def rasterize_geometry(
    primitive: Any,
    camera: Any,
    resolution: tuple[int, int],
    *,
    lights: Sequence[Any] = (),
    material: Material | None = None,
    **options: Any,
) -> GeometryLayer:
    """Route one primitive through its §19 tier and return its layer.

    Tier 0 meshes stay with :func:`spf.renderer.scene.rasterize_mesh`; tier 1
    and tier 2 use the routes above; tier 3 delegates to
    :mod:`spf.renderer.splats`.  The dispatch is on the primitive's declared
    ``tier``, never on a name.
    """

    tier = getattr(primitive, "tier", None)
    if isinstance(primitive, HermitianHeightField) or tier is GeometryTier.FIELD:
        return rasterize_height_field(primitive, camera, resolution, lights=lights, material=material, **options)
    if tier is GeometryTier.PATCH:
        return rasterize_patch(primitive, camera, resolution, lights=lights, material=material, **options)
    if tier is GeometryTier.GAUSSIAN:
        raise TypeError("tier-3 Gaussian geometry renders through Scene.add_splat / GaussianGeometry.render")
    raise TypeError(f"{type(primitive).__name__} does not declare a GeometryTier")


def render_scene(
    scene: Any,
    camera: Any | None = None,
    *,
    resolution: int | Sequence[int] = (64, 64),
    backend: str = "reference",
    **options: Any,
) -> Any:
    """Render every tier of a scene through one public entry point (§45).

    The hybrid path is: conventional visibility for tier 0, tier-1/2 analytic
    and field geometry through :func:`rasterize_geometry`, tier-3 Gaussians
    through the splat route, all composited by nearest finite depth, then the
    field-plane materials and lighting, one materialization and display.  The
    returned :class:`spf.renderer.scene.RenderResult` carries the
    ``RenderReport`` counters (``materialized_elements``,
    ``fallback_conversions``, ``fft_count``, and ``M_A`` through
    ``metrics.materialization_avoidance``).
    """

    from spf.renderer.scene import Renderer

    renderer = Renderer(backend)
    return renderer.render(scene, camera, resolution=resolution, **options)
