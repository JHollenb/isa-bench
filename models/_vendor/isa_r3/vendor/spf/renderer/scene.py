"""CPU-reference scene orchestration for the SFR v0.2 renderer.

The scene layer owns ordinary visibility and composition.  It deliberately
keeps coherent wave optics separate from the incoherent radiance path and
uses lazy discovery for the concurrently-developed ``splats`` module.  A
missing optional component is reported as an explicit fallback rather than
silently changing its value semantics.
"""

from __future__ import annotations

import importlib
import inspect
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from spf.renderer.backends import SFRBackend, get_backend
from spf.renderer.compiler import materialize_field
from spf.renderer.lighting import evaluate_lighting
from spf.renderer.materials import Material, as_native_color
from spf.renderer.model import (
    Coherence,
    ErrorBudget,
    Field,
    MaterializationRequest,
    PrecisionContract,
    PrecisionMode,
    SampleLayout,
)
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision
from spf.renderer.runtime import MaterializationError
from spf.tensor import v02

_EPS = 1.0e-10
_DEFAULT_MATERIAL = Material.pbr(base_color=(0.8, 0.8, 0.8), roughness=0.6, name="default")
_SCENE_EXECUTION_CLASS = "scene_cpu_reference_orchestration"
_SCENE_CALLABLE = "spf.renderer.scene.Renderer.render"


def _resolution(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, (int, np.integer)):
        width = height = int(value)
    else:
        values = tuple(int(item) for item in value)
        if len(values) != 2:
            raise ValueError("resolution must be an integer or (width, height)")
        width, height = values
    if width <= 0 or height <= 0:
        raise ValueError("resolution dimensions must be positive")
    return width, height


def _unit(vector: Any, *, name: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(values))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError(f"{name} must be nonzero")
    return values / norm


@dataclass(frozen=True, slots=True)
class Camera:
    """Minimal perspective/orthographic camera shared by reference paths."""

    position: Any = (0.0, 0.0, 3.0)
    target: Any = (0.0, 0.0, 0.0)
    up: Any = (0.0, 1.0, 0.0)
    fov_y: float = 45.0
    aspect: float | None = None
    near: float = 1.0e-4
    far: float = math.inf
    orthographic: bool = False
    ortho_scale: float = 2.0

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        target = np.asarray(self.target, dtype=np.float64).reshape(-1)
        up = np.asarray(self.up, dtype=np.float64).reshape(-1)
        if position.size != 3 or target.size != 3 or up.size != 3:
            raise ValueError("camera position, target, and up must be three-vectors")
        forward = _unit(target - position, name="camera forward")
        right = np.cross(forward, _unit(up, name="camera up"))
        if np.linalg.norm(right) <= np.finfo(np.float64).tiny:
            raise ValueError("camera up must not be parallel to the viewing direction")
        fov = float(self.fov_y)
        near = float(self.near)
        far = float(self.far)
        if not math.isfinite(fov) or not 0.0 < fov < 180.0:
            raise ValueError("camera fov_y must be finite and in (0, 180) degrees")
        if not math.isfinite(near) or near <= 0.0:
            raise ValueError("camera near plane must be finite and positive")
        if math.isnan(far) or far <= near:
            raise ValueError("camera far plane must be greater than near")
        scale = float(self.ortho_scale)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("ortho_scale must be finite and positive")
        if self.aspect is not None and (not math.isfinite(float(self.aspect)) or float(self.aspect) <= 0.0):
            raise ValueError("camera aspect must be finite and positive")
        object.__setattr__(self, "position", tuple(float(item) for item in position))
        object.__setattr__(self, "target", tuple(float(item) for item in target))
        object.__setattr__(self, "up", tuple(float(item) for item in _unit(up, name="camera up")))
        object.__setattr__(self, "fov_y", fov)
        object.__setattr__(self, "near", near)
        object.__setattr__(self, "far", far)
        object.__setattr__(self, "ortho_scale", scale)

    @property
    def position_vector(self) -> np.ndarray:
        return np.asarray(self.position, dtype=np.float64)

    @property
    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = _unit(np.asarray(self.target) - np.asarray(self.position), name="camera forward")
        right = _unit(np.cross(forward, np.asarray(self.up)), name="camera right")
        true_up = _unit(np.cross(right, forward), name="camera up")
        return right, true_up, forward

    def project(self, points: Any, resolution: int | Sequence[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project world points to pixel coordinates, positive camera depth, and validity."""

        width, height = _resolution(resolution)
        values = np.asarray(points, dtype=np.float64)
        if values.shape[-1:] != (3,):
            raise ValueError("points must have a trailing dimension of three")
        original_shape = values.shape[:-1]
        flat = values.reshape(-1, 3)
        right, true_up, forward = self.basis
        relative = flat - self.position_vector
        camera_x = relative @ right
        camera_y = relative @ true_up
        depth = relative @ forward
        aspect = float(self.aspect) if self.aspect is not None else width / height
        if self.orthographic:
            half_height = self.ortho_scale * 0.5
            half_width = half_height * aspect
            ndc_x = camera_x / half_width
            ndc_y = camera_y / half_height
        else:
            tangent = math.tan(math.radians(self.fov_y) * 0.5)
            safe_depth = np.where(np.abs(depth) > _EPS, depth, np.nan)
            ndc_x = camera_x / (safe_depth * tangent * aspect)
            ndc_y = camera_y / (safe_depth * tangent)
        screen = np.stack(
            ((ndc_x + 1.0) * 0.5 * width - 0.5, (1.0 - ndc_y) * 0.5 * height - 0.5),
            axis=-1,
        )
        valid = np.isfinite(screen).all(axis=-1) & np.isfinite(depth)
        valid &= depth >= self.near
        valid &= depth <= self.far
        if not self.orthographic:
            valid &= depth > 0.0
        return screen.reshape(*original_shape, 2), depth.reshape(original_shape), valid.reshape(original_shape)

    def ray(self, pixel: Sequence[float], resolution: int | Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """Return a world-space ray through a pixel center."""

        width, height = _resolution(resolution)
        values = np.asarray(tuple(pixel), dtype=np.float64).reshape(-1)
        if values.size != 2:
            raise ValueError("pixel must be an (x, y) pair")
        x, y = values
        right, true_up, forward = self.basis
        ndc_x = ((x + 0.5) / width) * 2.0 - 1.0
        ndc_y = 1.0 - ((y + 0.5) / height) * 2.0
        aspect = float(self.aspect) if self.aspect is not None else width / height
        if self.orthographic:
            half_height = self.ortho_scale * 0.5
            origin = self.position_vector + right * ndc_x * half_height * aspect + true_up * ndc_y * half_height
            return origin, forward.copy()
        tangent = math.tan(math.radians(self.fov_y) * 0.5)
        direction = _unit(forward + right * ndc_x * tangent * aspect + true_up * ndc_y * tangent, name="ray")
        return self.position_vector.copy(), direction


@dataclass(frozen=True, slots=True)
class Triangle:
    """One conventional triangle with optional per-vertex normals/material."""

    vertices: Any
    material: Material | None = None
    normals: Any | None = None
    primitive_id: int = 0
    uvs: Any | None = None
    material_id: Any = None
    field_bindings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        if vertices.shape != (3, 3) or not np.all(np.isfinite(vertices)):
            raise ValueError("triangle vertices must be a finite (3, 3) array")
        normals = None if self.normals is None else np.asarray(self.normals, dtype=np.float64)
        if normals is not None and (normals.shape != (3, 3) or not np.all(np.isfinite(normals))):
            raise ValueError("triangle normals must have shape (3, 3)")
        uvs = None if self.uvs is None else np.asarray(self.uvs, dtype=np.float64)
        if uvs is not None and (uvs.shape != (3, 2) or not np.all(np.isfinite(uvs))):
            raise ValueError("triangle uvs must have shape (3, 2)")
        object.__setattr__(self, "vertices", vertices.copy())
        object.__setattr__(self, "normals", None if normals is None else normals.copy())
        object.__setattr__(self, "primitive_id", int(self.primitive_id))
        object.__setattr__(self, "uvs", None if uvs is None else uvs.copy())
        object.__setattr__(self, "field_bindings", MappingProxyType(dict(self.field_bindings)))

    @property
    def face_normal(self) -> np.ndarray:
        edge_a = self.vertices[1] - self.vertices[0]
        edge_b = self.vertices[2] - self.vertices[0]
        cross = np.cross(edge_a, edge_b)
        norm = float(np.linalg.norm(cross))
        return np.zeros(3, dtype=np.float64) if norm <= _EPS else cross / norm


def _obj_index(value: str, size: int, *, path: Path, line_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_number}: invalid OBJ index {value!r}") from exc
    if parsed == 0:
        raise ValueError(f"{path}:{line_number}: OBJ indices are one-based and cannot be zero")
    index = parsed - 1 if parsed > 0 else size + parsed
    if index < 0 or index >= size:
        raise ValueError(f"{path}:{line_number}: OBJ index {parsed} is out of bounds")
    return index


def _load_obj(path: Path, *, material: Material | None, name: str) -> Mesh:
    """Load the geometry subset of OBJ with deterministic fan triangulation."""

    positions: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    uvs: list[tuple[float, float]] = []
    output_positions: list[tuple[float, float, float]] = []
    output_normals: list[tuple[float, float, float] | None] = []
    output_uvs: list[tuple[float, float] | None] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.partition("#")[0].strip()
        if not line:
            continue
        fields = line.split()
        record, values = fields[0], fields[1:]
        if record == "v":
            if len(values) < 3:
                raise ValueError(f"{path}:{line_number}: OBJ vertex needs three coordinates")
            positions.append((float(values[0]), float(values[1]), float(values[2])))
        elif record == "vn":
            if len(values) < 3:
                raise ValueError(f"{path}:{line_number}: OBJ normal needs three coordinates")
            normals.append((float(values[0]), float(values[1]), float(values[2])))
        elif record == "vt":
            if len(values) < 2:
                raise ValueError(f"{path}:{line_number}: OBJ texture coordinate needs two coordinates")
            uvs.append((float(values[0]), float(values[1])))
        elif record == "f":
            if len(values) < 3:
                raise ValueError(f"{path}:{line_number}: OBJ face needs at least three vertices")
            corners: list[tuple[int, int | None, int | None]] = []
            for value in values:
                parts = value.split("/")
                position = _obj_index(parts[0], len(positions), path=path, line_number=line_number)
                uv = (
                    _obj_index(parts[1], len(uvs), path=path, line_number=line_number)
                    if len(parts) > 1 and parts[1]
                    else None
                )
                normal = (
                    _obj_index(parts[2], len(normals), path=path, line_number=line_number)
                    if len(parts) > 2 and parts[2]
                    else None
                )
                corners.append((position, uv, normal))
            for index in range(1, len(corners) - 1):
                for position_index, uv_index, normal_index in (corners[0], corners[index], corners[index + 1]):
                    output_positions.append(positions[position_index])
                    output_uvs.append(None if uv_index is None else uvs[uv_index])
                    output_normals.append(None if normal_index is None else normals[normal_index])

    if not output_positions:
        raise ValueError(f"{path}: OBJ contains no triangulatable faces")
    loaded_normals = None if any(value is None for value in output_normals) else np.asarray(output_normals)
    loaded_uvs = None if any(value is None for value in output_uvs) else np.asarray(output_uvs)
    return Mesh(
        np.asarray(output_positions),
        material=material,
        normals=loaded_normals,
        uvs=loaded_uvs,
        name=name,
    )


def _interpolate_clip_vertex(
    left: tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float],
    right: tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float],
    plane: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]:
    denominator = right[3] - left[3]
    amount = 0.0 if abs(denominator) <= _EPS else float(np.clip((plane - left[3]) / denominator, 0.0, 1.0))
    position = left[0] + amount * (right[0] - left[0])
    normal = None if left[1] is None or right[1] is None else left[1] + amount * (right[1] - left[1])
    uv = None if left[2] is None or right[2] is None else left[2] + amount * (right[2] - left[2])
    return position, normal, uv, plane


def _clip_polygon_depth(
    polygon: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]],
    plane: float,
    *,
    keep_greater: bool,
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]]:
    if not polygon:
        return []

    def inside(vertex: tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]) -> bool:
        return vertex[3] >= plane - _EPS if keep_greater else vertex[3] <= plane + _EPS

    output: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]] = []
    previous = polygon[-1]
    previous_inside = inside(previous)
    for current in polygon:
        current_inside = inside(current)
        if current_inside != previous_inside:
            output.append(_interpolate_clip_vertex(previous, current, plane))
        if current_inside:
            output.append(current)
        previous, previous_inside = current, current_inside
    return output


def _prepare_triangles(mesh: Mesh, camera: Any) -> tuple[tuple[Triangle, ...], int, int]:
    """Clip scene-camera triangles to near/far planes before projection."""

    source = tuple(mesh.iter_triangles())
    if not isinstance(camera, Camera):
        return source, 0, 0
    _, _, forward = camera.basis
    origin = camera.position_vector
    prepared: list[Triangle] = []
    clipped_count = 0
    discarded_count = 0
    for triangle in source:
        depths = (triangle.vertices - origin) @ forward
        polygon = [
            (
                triangle.vertices[index],
                None if triangle.normals is None else triangle.normals[index],
                None if triangle.uvs is None else triangle.uvs[index],
                float(depths[index]),
            )
            for index in range(3)
        ]
        needs_clipping = bool(np.any(depths < camera.near - _EPS) or np.any(depths > camera.far + _EPS))
        if needs_clipping:
            clipped_count += 1
            polygon = _clip_polygon_depth(polygon, camera.near, keep_greater=True)
            if math.isfinite(camera.far):
                polygon = _clip_polygon_depth(polygon, camera.far, keep_greater=False)
        if len(polygon) < 3:
            discarded_count += 1
            continue
        for index in range(1, len(polygon) - 1):
            vertices = (polygon[0], polygon[index], polygon[index + 1])
            normals = (
                None if any(value[1] is None for value in vertices) else np.asarray([value[1] for value in vertices])
            )
            texture_coordinates = (
                None if any(value[2] is None for value in vertices) else np.asarray([value[2] for value in vertices])
            )
            prepared.append(
                Triangle(
                    np.asarray([value[0] for value in vertices]),
                    triangle.material,
                    normals,
                    triangle.primitive_id,
                    texture_coordinates,
                    triangle.material_id,
                    triangle.field_bindings,
                )
            )
    return tuple(prepared), clipped_count, discarded_count


@dataclass(frozen=True, slots=True)
class Mesh:
    """Indexed triangle mesh accepted by the CPU reference rasterizer."""

    vertices: Any
    indices: Any | None = None
    material: Material | None = None
    normals: Any | None = None
    name: str = "mesh"
    faces: Any | None = None
    uvs: Any | None = None
    material_id: Any = None
    field_bindings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw_vertices = np.asarray(self.vertices, dtype=np.float64)
        implicit_indices: np.ndarray | None
        if raw_vertices.ndim == 3 and raw_vertices.shape[1:] == (3, 3):
            vertices = raw_vertices.reshape(-1, 3)
            implicit_indices = np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3)
        elif raw_vertices.ndim == 2 and raw_vertices.shape[1] == 3:
            vertices = raw_vertices
            implicit_indices = (
                np.arange(vertices.shape[0], dtype=np.int64).reshape(-1, 3) if vertices.shape[0] % 3 == 0 else None
            )
        else:
            raise ValueError("mesh vertices must have shape (N, 3) or (T, 3, 3)")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("mesh vertices must be finite")
        source_indices = self.indices if self.indices is not None else self.faces
        if source_indices is None:
            if implicit_indices is None:
                raise ValueError("indexed mesh requires indices/faces or a multiple of three vertices")
            indices = implicit_indices
        else:
            indices = np.asarray(source_indices)
            if indices.ndim != 2 or indices.shape[1] != 3:
                raise ValueError("mesh indices/faces must have shape (T, 3)")
            if not np.issubdtype(indices.dtype, np.integer) and not np.all(np.equal(indices, np.rint(indices))):
                raise ValueError("mesh indices must be integers")
            indices = np.asarray(indices, dtype=np.int64)
            if np.any(indices < 0) or np.any(indices >= vertices.shape[0]):
                raise ValueError("mesh indices are out of bounds")
        normals = None if self.normals is None else np.asarray(self.normals, dtype=np.float64)
        if normals is not None:
            if normals.ndim == 3 and normals.shape[1:] == (3, 3):
                normals = normals.reshape(-1, 3)
            if normals.shape != vertices.shape or not np.all(np.isfinite(normals)):
                raise ValueError("mesh normals must have shape (N, 3)")
        uvs = None if self.uvs is None else np.asarray(self.uvs, dtype=np.float64)
        if uvs is not None and (uvs.shape != (vertices.shape[0], 2) or not np.all(np.isfinite(uvs))):
            raise ValueError("mesh uvs must have shape (N, 2)")
        vertices = vertices.copy()
        indices = indices.copy()
        vertices.setflags(write=False)
        indices.setflags(write=False)
        if normals is not None:
            normals = normals.copy()
            normals.setflags(write=False)
        if uvs is not None:
            uvs = uvs.copy()
            uvs.setflags(write=False)
        object.__setattr__(self, "vertices", vertices)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "normals", normals)
        object.__setattr__(self, "uvs", uvs)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "field_bindings", MappingProxyType(dict(self.field_bindings)))

    @classmethod
    def from_triangles(
        cls,
        triangles: Iterable[Triangle],
        *,
        material: Material | None = None,
        name: str = "mesh",
    ) -> Mesh:
        items = tuple(triangles)
        if not items:
            raise ValueError("from_triangles requires at least one triangle")
        selected_material = material
        if selected_material is None:
            first_material = items[0].material
            if all(item.material is first_material for item in items):
                selected_material = first_material
            elif any(item.material is not None for item in items):
                raise ValueError("Mesh stores one material; split triangles with different materials into meshes")
        vertices = np.concatenate([item.vertices for item in items], axis=0)
        normals = None
        if all(item.normals is not None for item in items):
            normals = np.concatenate([item.normals for item in items if item.normals is not None], axis=0)
        uvs = None
        if all(item.uvs is not None for item in items):
            uvs = np.concatenate([item.uvs for item in items if item.uvs is not None], axis=0)
        material_id = items[0].material_id if all(item.material_id == items[0].material_id for item in items) else None
        bindings = items[0].field_bindings
        if not all(dict(item.field_bindings) == dict(bindings) for item in items):
            raise ValueError("Mesh stores one field-binding map; split triangles with different bindings")
        return cls(
            vertices,
            material=selected_material,
            normals=normals,
            name=name,
            uvs=uvs,
            material_id=material_id,
            field_bindings=bindings,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        material: Material | None = None,
        *,
        name: str | None = None,
    ) -> Mesh:
        """Load an OBJ-style mesh or deterministic NumPy ``.npz`` bundle.

        OBJ polygon faces are fan-triangulated and independent position/UV/
        normal indices are expanded, preserving the exact interpolation data.
        An ``.npz`` bundle must contain ``vertices`` and may contain
        ``indices``/``faces``, ``normals``, and ``uvs``.
        """

        source = Path(path).expanduser()
        selected_name = source.stem if name is None else str(name)
        if source.suffix.lower() == ".npz":
            with np.load(source, allow_pickle=False) as archive:
                if "vertices" not in archive:
                    raise ValueError(f"{source}: mesh archive requires a 'vertices' array")
                vertices = np.asarray(archive["vertices"]).copy()
                indices = None
                if "indices" in archive:
                    indices = np.asarray(archive["indices"]).copy()
                elif "faces" in archive:
                    indices = np.asarray(archive["faces"]).copy()
                normals = np.asarray(archive["normals"]).copy() if "normals" in archive else None
                uvs = np.asarray(archive["uvs"]).copy() if "uvs" in archive else None
            return cls(vertices, indices=indices, material=material, normals=normals, name=selected_name, uvs=uvs)
        return _load_obj(source, material=material, name=selected_name)

    @property
    def triangle_count(self) -> int:
        indices = self.indices
        if indices is None:  # pragma: no cover - constructor establishes indices
            raise RuntimeError("mesh indices were not initialized")
        return int(indices.shape[0])

    def iter_triangles(self) -> Iterable[Triangle]:
        indices = self.indices
        if indices is None:  # pragma: no cover - constructor establishes indices
            raise RuntimeError("mesh indices were not initialized")
        for primitive_id, index in enumerate(indices):
            normals = None if self.normals is None else self.normals[index]
            uvs = None if self.uvs is None else self.uvs[index]
            yield Triangle(
                self.vertices[index],
                self.material,
                normals,
                primitive_id,
                uvs,
                self.material_id,
                self.field_bindings,
            )


@dataclass(frozen=True, slots=True)
class VisibleSample:
    """A visible sample with explicit depth and coverage semantics."""

    pixel: tuple[int, int] = (0, 0)
    depth: float = math.inf
    coverage: float = 0.0
    color: Any = (0.0, 0.0, 0.0)
    position: Any = (0.0, 0.0, 0.0)
    normal: Any = (0.0, 0.0, 1.0)
    primitive_id: int = -1
    material: Material | None = None
    source: str = "mesh"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    uv: Any | None = None
    local_coordinates: Any | None = None
    field_bindings: Mapping[str, Any] = field(default_factory=dict)
    material_id: Any = None

    def __post_init__(self) -> None:
        pixel = tuple(int(value) for value in self.pixel)
        if len(pixel) != 2:
            raise ValueError("visible sample pixel must be an (x, y) pair")
        depth = float(self.depth)
        coverage = float(self.coverage)
        if math.isnan(depth) or depth < 0.0:
            raise ValueError("visible sample depth must be non-negative or infinity")
        if not math.isfinite(coverage) or not 0.0 <= coverage <= 1.0:
            raise ValueError("visible sample coverage must be in [0, 1]")
        color = as_native_color(self.color, name="visible sample color", allow_negative=True)
        position = np.asarray(self.position, dtype=np.float64).reshape(-1)
        normal = np.asarray(self.normal, dtype=np.float64).reshape(-1)
        uv = None if self.uv is None else np.asarray(self.uv, dtype=np.float64).reshape(-1)
        local = (
            None if self.local_coordinates is None else np.asarray(self.local_coordinates, dtype=np.float64).reshape(-1)
        )
        if position.size != 3 or not np.all(np.isfinite(position)):
            raise ValueError("visible sample position must be finite and three-dimensional")
        if normal.size != 3 or not np.all(np.isfinite(normal)):
            raise ValueError("visible sample normal must be finite and three-dimensional")
        if uv is not None and (uv.size != 2 or not np.all(np.isfinite(uv))):
            raise ValueError("visible sample uv must be a finite two-vector")
        if local is not None and not np.all(np.isfinite(local)):
            raise ValueError("visible sample local coordinates must be finite")
        norm = float(np.linalg.norm(normal))
        if norm > _EPS:
            normal = normal / norm
        object.__setattr__(self, "pixel", pixel)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(self, "color", tuple(float(value) for value in color))
        object.__setattr__(self, "position", tuple(float(value) for value in position))
        object.__setattr__(self, "normal", tuple(float(value) for value in normal))
        object.__setattr__(self, "primitive_id", int(self.primitive_id))
        object.__setattr__(self, "source", str(self.source))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        object.__setattr__(self, "uv", None if uv is None else tuple(float(value) for value in uv))
        object.__setattr__(
            self,
            "local_coordinates",
            None if local is None else tuple(float(value) for value in local),
        )
        object.__setattr__(self, "field_bindings", MappingProxyType(dict(self.field_bindings)))

    @property
    def visible(self) -> bool:
        return self.coverage > 0.0 and math.isfinite(self.depth)

    @classmethod
    def empty(cls, pixel: tuple[int, int] = (0, 0)) -> VisibleSample:
        return cls(pixel=pixel)


@dataclass(frozen=True, slots=True)
class RasterizedMesh:
    image: np.ndarray
    depth: np.ndarray
    coverage: np.ndarray
    primitive_ids: np.ndarray
    samples: tuple[VisibleSample, ...]
    clipped_triangles: int = 0
    discarded_triangles: int = 0
    positive_logadds: int = 0
    pivot_reductions: int = 0
    reduction_error_bound: float = 0.0
    # These bounds describe source radiance/material uncertainty before the
    # BRDF and display stages.  They stay separate from the image ledger.
    source_error_bound: float = 0.0
    source_ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)
    lighting_source_error_bound: float = 0.0
    lighting_source_ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)
    material_source_error_bound: float = 0.0
    material_source_ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)
    material_output_error_bound: float = 0.0
    material_source_propagated: bool = False
    material_output_error_map: np.ndarray | None = None

    def __post_init__(self) -> None:
        image = np.asarray(self.image, dtype=np.float64)
        depth = np.asarray(self.depth, dtype=np.float64)
        coverage = np.asarray(self.coverage, dtype=np.float64)
        primitive_ids = np.asarray(self.primitive_ids, dtype=np.int64)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("rasterized image must have shape (H, W, 3)")
        if depth.shape != image.shape[:2] or coverage.shape != depth.shape or primitive_ids.shape != depth.shape:
            raise ValueError("rasterized mesh buffers must share (H, W) shape")
        if int(self.clipped_triangles) < 0 or int(self.discarded_triangles) < 0:
            raise ValueError("triangle clip counters must be non-negative")
        reduction_bound = float(self.reduction_error_bound)
        if math.isnan(reduction_bound) or reduction_bound < 0.0:
            raise ValueError("mesh reduction error bound must be non-negative")
        for name in (
            "source_ledger",
            "lighting_source_ledger",
            "material_source_ledger",
        ):
            if not isinstance(getattr(self, name), v02.ErrorLedger):
                raise TypeError(f"mesh {name} must be an ErrorLedger")
        source_bound = float(self.source_error_bound)
        lighting_source_bound = float(self.lighting_source_error_bound)
        material_source_bound = float(self.material_source_error_bound)
        material_output_bound = float(self.material_output_error_bound)
        material_output_map = (
            np.zeros(depth.shape, dtype=np.float64)
            if self.material_output_error_map is None
            else np.asarray(self.material_output_error_map, dtype=np.float64)
        )
        if material_output_map.shape != depth.shape:
            raise ValueError("mesh material output error map must share the raster (H, W) shape")
        if not np.all(np.isfinite(material_output_map)) or np.any(material_output_map < 0.0):
            raise ValueError("mesh material output error map must be finite and non-negative")
        material_output_bound = max(material_output_bound, float(np.max(material_output_map, initial=0.0)))
        for name, value in (
            ("source error", source_bound),
            ("lighting source error", lighting_source_bound),
            ("material source error", material_source_bound),
            ("material output error", material_output_bound),
        ):
            if math.isnan(value) or value < 0.0:
                raise ValueError(f"mesh {name} bound must be non-negative")
        source_bound = max(source_bound, float(self.source_ledger.total_bound))
        lighting_source_bound = max(lighting_source_bound, float(self.lighting_source_ledger.total_bound))
        material_source_bound = max(material_source_bound, float(self.material_source_ledger.total_bound))
        source_bound = max(source_bound, lighting_source_bound + material_source_bound)
        object.__setattr__(self, "image", image.copy())
        object.__setattr__(self, "depth", depth.copy())
        object.__setattr__(self, "coverage", coverage.copy())
        object.__setattr__(self, "primitive_ids", primitive_ids.copy())
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "clipped_triangles", int(self.clipped_triangles))
        object.__setattr__(self, "discarded_triangles", int(self.discarded_triangles))
        object.__setattr__(self, "positive_logadds", int(self.positive_logadds))
        object.__setattr__(self, "pivot_reductions", int(self.pivot_reductions))
        object.__setattr__(self, "reduction_error_bound", reduction_bound)
        object.__setattr__(self, "source_error_bound", source_bound)
        object.__setattr__(self, "lighting_source_error_bound", lighting_source_bound)
        object.__setattr__(self, "material_source_error_bound", material_source_bound)
        object.__setattr__(self, "material_output_error_bound", material_output_bound)
        object.__setattr__(self, "material_source_propagated", bool(self.material_source_propagated))
        object.__setattr__(self, "material_output_error_map", material_output_map.copy())


def depth_min(samples: Iterable[VisibleSample], *, tie_epsilon: float = _EPS) -> VisibleSample | None:
    """Select the nearest covered sample; ties prefer greater coverage."""

    candidates = [item for item in samples if isinstance(item, VisibleSample) and item.visible]
    if not candidates:
        return None
    nearest = min(item.depth for item in candidates)
    tied = [item for item in candidates if abs(item.depth - nearest) <= tie_epsilon]
    return max(tied, key=lambda item: (item.coverage, -item.primitive_id))


nearest_visible = depth_min
select_nearest = depth_min


def coverage_mask(samples: Iterable[VisibleSample], *, threshold: float = 0.0) -> bool:
    values = tuple(samples)
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("coverage threshold must be in [0, 1]")
    return any(item.coverage > threshold for item in values if isinstance(item, VisibleSample))


def _project_with_camera(
    camera: Any, points: Any, resolution: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(camera, "project"):
        result = camera.project(points, resolution)
        if isinstance(result, tuple) and len(result) == 3:
            screen, depth, valid = result
            return (
                np.asarray(screen, dtype=np.float64),
                np.asarray(depth, dtype=np.float64),
                np.asarray(valid, dtype=bool),
            )
    fallback = Camera()
    return fallback.project(points, resolution)


def _shade_sample(
    material: Material,
    normal: np.ndarray,
    position: np.ndarray,
    camera: Any,
    lights: Sequence[Any],
    *,
    uv: Any | None = None,
    texture_cache: dict[int, np.ndarray] | None = None,
) -> tuple[np.ndarray, Any | None]:
    material = material.sampled((0.0, 0.0) if uv is None else uv, cache=texture_cache)
    if not lights:
        return material.base_color_array + material.emission_array, None
    camera_position = np.asarray(getattr(camera, "position", (0.0, 0.0, 1.0)), dtype=np.float64)
    view = _unit(camera_position - position, name="view")
    lighting = evaluate_lighting(lights, normal, view, point=position, material=material)
    return np.asarray(lighting.color + material.emission_array, dtype=np.float64), lighting


_SOURCE_LEDGER_FIELDS = (
    "source_quantization",
    "representation_fit",
    "pruning",
    "reducer",
    "transform",
    "materialization",
    "display",
)


def _max_source_ledger(left: v02.ErrorLedger, right: v02.ErrorLedger) -> v02.ErrorLedger:
    """Keep a pointwise source envelope without counting one pixel repeatedly."""

    def bound(value: float) -> float:
        return math.inf if math.isnan(value) else value

    return v02.ErrorLedger(
        **{name: max(bound(getattr(left, name)), bound(getattr(right, name))) for name in _SOURCE_LEDGER_FIELDS},
        stages=tuple(dict.fromkeys((*left.stages, *right.stages))),
    )


def _material_source_ledger(material: Material) -> v02.ErrorLedger:
    """Read material source uncertainty before its sampled BRDF specialization."""

    source = material.base_color_source
    if source is None:
        return v02.ErrorLedger()
    if isinstance(source, Field):
        from spf.renderer.normalization import complete_field_ledger, normalize_field

        # Normalize parameter-only operations before reading the ledger.  This
        # is the same source boundary consumed by Material.sample_base_color;
        # it carries source uncertainty without treating it as image error.
        return complete_field_ledger(normalize_field(source).field)
    bound = getattr(source, "error_bound", getattr(source, "source_error_bound", 0.0))
    value = float(bound)
    if math.isnan(value) or value < 0.0:
        raise ValueError("material source error bound must be non-negative")
    return v02.ErrorLedger(source_quantization=value)


def _bounded_texture_source(material: Material) -> bool:
    source = material.base_color_source
    if source is None:
        return False
    from spf.renderer.assets import SpectralTexture

    if isinstance(source, SpectralTexture):
        return True
    if isinstance(source, Field) and isinstance(source.representation.payload, SpectralTexture):
        return True
    # Addressed block fields carry the same finite source ledger and positive
    # interpolation contract as SpectralTexture.  Deferred spatial filters or
    # other unsupported prefixes still fail closed in the sampling helper.
    from spf.renderer.carriers import as_block_view

    payload = source.representation.payload if isinstance(source, Field) else source
    return as_block_view(payload) is not None


def _material_output_error_bound(
    material: Material,
    source_error: float,
    normal: np.ndarray,
    position: np.ndarray,
    camera: Any,
    lights: Sequence[Any],
    *,
    uv: Any | None,
    texture_cache: dict[int, np.ndarray],
    nominal_lighting: Any | None,
) -> float | None:
    """Bound a texture source interval after the audited mesh shading path."""

    if source_error <= 0.0 or not math.isfinite(source_error) or not _bounded_texture_source(material):
        return None
    coordinates = (0.0, 0.0) if uv is None else uv
    try:
        base = np.asarray(material.sample_base_color(coordinates, cache=texture_cache), dtype=np.float64).reshape(3)
    except (TypeError, ValueError, LookupError):
        return None
    if not np.all(np.isfinite(base)):
        return None
    lower = np.maximum(base - source_error, 0.0)
    upper = base + source_error
    lower_material = replace(material, base_color=tuple(float(value) for value in lower))
    upper_material = replace(material, base_color=tuple(float(value) for value in upper))
    if not lights:
        # This is a bound around the nominal sample, so use the larger
        # one-sided distance rather than the full interval width.
        return float(np.max(np.maximum(base - lower, upper - base), initial=0.0))
    if nominal_lighting is None or nominal_lighting.source_error_bound > 0.0:
        # A light source interval has its own pre-BRDF boundary and is not
        # folded into this material-only proof.
        return None
    camera_position = np.asarray(getattr(camera, "position", (0.0, 0.0, 1.0)), dtype=np.float64)
    view = _unit(camera_position - position, name="view")
    try:
        lower_lighting = evaluate_lighting(lights, normal, view, point=position, material=lower_material)
        upper_lighting = evaluate_lighting(lights, normal, view, point=position, material=upper_material)
    except (TypeError, ValueError, LookupError):
        return None
    nominal = np.asarray(nominal_lighting.color, dtype=np.float64)
    distance = max(
        float(np.max(np.abs(nominal - lower_lighting.color), initial=0.0)),
        float(np.max(np.abs(upper_lighting.color - nominal), initial=0.0)),
    )
    # The exact positive shading chain is monotone in each base-colour
    # channel: Lambert is affine, while the microfacet path is a product of
    # positive factors and an affine Fresnel term.  Consequently every
    # interior source value has terms below the componentwise upper endpoint.
    # The reducer certificate at that endpoint is therefore a conservative
    # envelope for the interior positive reduction; retaining the nominal and
    # lower certificates also covers the comparison to the quantized nominal
    # value used by the rasterizer.
    reduction = max(
        float(nominal_lighting.reduction_error_bound),
        float(lower_lighting.reduction_error_bound),
        float(upper_lighting.reduction_error_bound),
    )
    return distance + reduction


def _display_transform_for_bound(
    values: np.ndarray,
    exposure: float,
    tone_map: str,
    gamma: float,
    display: bool,
) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    if exposure:
        result *= 2.0**exposure
    if tone_map in {"reinhard", "tone_reinhard"}:
        result /= 1.0 + np.maximum(result, 0.0)
    if gamma != 1.0:
        np.maximum(result, 0.0, out=result)
        np.power(result, 1.0 / gamma, out=result)
    np.maximum(result, 0.0, out=result)
    if display:
        result = np.rint(np.clip(result, 0.0, 1.0) * 255.0) / 255.0
    return result


def _display_error_bound(
    image: np.ndarray,
    linear_error: float | np.ndarray,
    exposure: float,
    tone_map: str,
    gamma: float,
    display: bool,
) -> float:
    error = np.asarray(linear_error, dtype=np.float64)
    if error.ndim == 0:
        if float(error) <= 0.0:
            return 0.0
        error = np.full(image.shape[:2], float(error), dtype=np.float64)
    elif error.shape != image.shape[:2]:
        raise ValueError("display error map must share the image (H, W) shape")
    if np.any(np.isnan(error)) or np.any(np.isposinf(error)):
        return math.inf
    if np.any(error < 0.0):
        raise ValueError("display error map must be finite and non-negative")
    if not np.any(error > 0.0):
        return 0.0
    lower = _display_transform_for_bound(image - error[..., None], exposure, tone_map, gamma, display)
    upper = _display_transform_for_bound(image + error[..., None], exposure, tone_map, gamma, display)
    return float(np.max(np.abs(upper - lower), initial=0.0))


def rasterize_mesh(
    mesh: Mesh,
    camera: Any,
    resolution: int | Sequence[int],
    *,
    lights: Sequence[Any] = (),
    material: Material | None = None,
) -> RasterizedMesh:
    """Rasterize a mesh with nearest-depth visibility and unit interior coverage."""

    width, height = _resolution(resolution)
    image = np.zeros((height, width, 3), dtype=np.float64)
    depth = np.full((height, width), math.inf, dtype=np.float64)
    coverage = np.zeros((height, width), dtype=np.float64)
    primitive_ids = np.full((height, width), -1, dtype=np.int64)
    sample_by_pixel: dict[tuple[int, int], VisibleSample] = {}
    selected_material = material or mesh.material or _DEFAULT_MATERIAL
    texture_cache: dict[int, np.ndarray] = {}
    positive_logadds = 0
    pivot_reductions = 0
    reduction_error_bound = 0.0
    lighting_source_error_bound = 0.0
    lighting_source_ledger = v02.ErrorLedger()
    material_source_error_bound = 0.0
    material_source_ledger = v02.ErrorLedger()
    material_output_error_bound = 0.0
    material_output_error_map: np.ndarray = np.zeros((height, width), dtype=np.float64)
    material_source_unpropagated = False
    material_source_seen = False
    triangles, clipped_triangles, discarded_triangles = _prepare_triangles(mesh, camera)
    for triangle in triangles:
        screen, triangle_depth, valid = _project_with_camera(camera, triangle.vertices, (width, height))
        if not np.all(valid):
            # Custom camera implementations own their clip convention; an
            # invalid primitive is counted and never becomes a false sample.
            discarded_triangles += 1
            continue
        denominator = (screen[1, 1] - screen[2, 1]) * (screen[0, 0] - screen[2, 0]) + (screen[2, 0] - screen[1, 0]) * (
            screen[0, 1] - screen[2, 1]
        )
        if abs(float(denominator)) <= _EPS:
            continue
        low_x = max(0, math.floor(float(np.min(screen[:, 0]))))
        high_x = min(width - 1, math.ceil(float(np.max(screen[:, 0]))))
        low_y = max(0, math.floor(float(np.min(screen[:, 1]))))
        high_y = min(height - 1, math.ceil(float(np.max(screen[:, 1]))))
        if low_x > high_x or low_y > high_y:
            continue
        face_normal = triangle.face_normal
        # Reject coverage/depth for the whole triangle tile before constructing
        # typed visible samples or invoking material consumers.
        yy, xx = np.mgrid[low_y : high_y + 1, low_x : high_x + 1]
        px, py = xx + 0.5, yy + 0.5
        w0 = (
            (screen[1, 1] - screen[2, 1]) * (px - screen[2, 0]) + (screen[2, 0] - screen[1, 0]) * (py - screen[2, 1])
        ) / denominator
        w1 = (
            (screen[2, 1] - screen[0, 1]) * (px - screen[2, 0]) + (screen[0, 0] - screen[2, 0]) * (py - screen[2, 1])
        ) / denominator
        weights = np.stack((w0, w1, 1 - w0 - w1), axis=-1)
        inside = np.min(weights, axis=-1) >= -_EPS
        if not bool(getattr(camera, "orthographic", False)):
            reciprocal = weights / triangle_depth
            reciprocal_sum = np.sum(reciprocal, axis=-1)
            inside &= reciprocal_sum > _EPS
            denominator_safe = np.where(inside, reciprocal_sum, 1)
            weights = reciprocal / denominator_safe[..., None]
            depths = 1 / denominator_safe
        else:
            depths = weights @ triangle_depth
        winners = inside & (depths < depth[low_y : high_y + 1, low_x : high_x + 1] - _EPS)
        for row, column in np.argwhere(winners):
            pixel_y, pixel_x = int(yy[row, column]), int(xx[row, column])
            interpolation_weights = weights[row, column]
            candidate_depth = float(depths[row, column])
            position = interpolation_weights @ triangle.vertices
            if triangle.normals is None:
                normal = face_normal
            else:
                normal = _unit(
                    interpolation_weights @ triangle.normals,
                    name="interpolated normal",
                )
            uv = None if triangle.uvs is None else interpolation_weights @ triangle.uvs
            color, lighting = _shade_sample(
                selected_material,
                normal,
                position,
                camera,
                lights,
                uv=uv,
                texture_cache=texture_cache,
            )
            if lighting is not None:
                positive_logadds += lighting.positive_logadds
                pivot_reductions += lighting.pivot_reductions
                reduction_error_bound = max(reduction_error_bound, lighting.reduction_error_bound)
                lighting_source_error_bound = max(lighting_source_error_bound, lighting.source_error_bound)
                source_ledger = lighting.source_ledger
                represented = float(source_ledger.total_bound)
                if lighting.source_error_bound > represented:
                    source_ledger = source_ledger.plus(
                        v02.ErrorLedger(source_quantization=lighting.source_error_bound - represented)
                    )
                lighting_source_ledger = _max_source_ledger(lighting_source_ledger, source_ledger)
            if not material_source_seen and selected_material.base_color_source is not None:
                material_source_ledger = _material_source_ledger(selected_material)
                material_source_error_bound = float(material_source_ledger.total_bound)
                material_source_seen = True
            if material_source_error_bound > 0.0:
                output_bound = _material_output_error_bound(
                    selected_material,
                    material_source_error_bound,
                    normal,
                    position,
                    camera,
                    lights,
                    uv=uv,
                    texture_cache=texture_cache,
                    nominal_lighting=lighting,
                )
                if output_bound is not None:
                    material_output_error_bound = max(material_output_error_bound, output_bound)
                    material_output_error_map[pixel_y, pixel_x] = max(
                        material_output_error_map[pixel_y, pixel_x], output_bound
                    )
                else:
                    # A source proof must cover every visible query. One
                    # successful texture sample cannot certify another face,
                    # UV, or source binding that the audited path could not
                    # enclose.
                    material_source_unpropagated = True
            sample = VisibleSample(
                pixel=(pixel_x, pixel_y),
                depth=candidate_depth,
                coverage=1.0,
                color=color,
                position=position,
                normal=normal,
                primitive_id=triangle.primitive_id,
                material=selected_material,
                source=mesh.name,
                uv=uv,
                local_coordinates=uv if uv is not None else interpolation_weights[1:],
                field_bindings=triangle.field_bindings,
                material_id=triangle.material_id,
            )
            depth[pixel_y, pixel_x] = candidate_depth
            coverage[pixel_y, pixel_x] = 1.0
            primitive_ids[pixel_y, pixel_x] = triangle.primitive_id
            image[pixel_y, pixel_x] = color
            sample_by_pixel[(pixel_y, pixel_x)] = sample
    samples = tuple(sample_by_pixel[key] for key in sorted(sample_by_pixel))
    source_ledger = lighting_source_ledger.plus(material_source_ledger)
    source_error_bound = max(
        lighting_source_error_bound,
        material_source_error_bound,
        float(source_ledger.total_bound),
    )
    return RasterizedMesh(
        image,
        depth,
        coverage,
        primitive_ids,
        samples,
        clipped_triangles,
        discarded_triangles,
        positive_logadds,
        pivot_reductions,
        reduction_error_bound,
        source_error_bound,
        source_ledger,
        lighting_source_error_bound,
        lighting_source_ledger,
        material_source_error_bound,
        material_source_ledger,
        material_output_error_bound,
        material_source_error_bound <= 0.0 or not material_source_unpropagated,
        material_output_error_map,
    )


@dataclass(slots=True)
class Scene:
    """Hybrid scene container for meshes, splats, volumes, and field bindings."""

    meshes: tuple[Mesh, ...] = ()
    lights: tuple[Any, ...] = ()
    splats: tuple[Any, ...] = ()
    volumes: tuple[Any, ...] = ()
    # §19.2/§19.3 tier-1 analytic patches and tier-2 field geometry.  They live
    # in their own list because they resolve visibility themselves and then bind
    # the field plane, unlike a `fields` binding (which is an additive radiance
    # layer with no geometry of its own).
    geometry: tuple[Any, ...] = ()
    geometry_materials: dict[int, Material] = field(default_factory=dict)
    fields: dict[str, Any] = field(default_factory=dict)
    camera: Any | None = None
    background: Any = (0.0, 0.0, 0.0)
    exposure: float = 0.0
    tone_map: str = "none"
    gamma: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "meshes", tuple(self.meshes))
        object.__setattr__(self, "lights", tuple(self.lights))
        object.__setattr__(self, "splats", tuple(self.splats))
        object.__setattr__(self, "volumes", tuple(self.volumes))
        object.__setattr__(self, "geometry", tuple(self.geometry))
        object.__setattr__(self, "geometry_materials", dict(self.geometry_materials))
        object.__setattr__(self, "fields", dict(self.fields))
        object.__setattr__(
            self,
            "background",
            tuple(float(value) for value in as_native_color(self.background, name="background", allow_negative=False)),
        )
        exposure = float(self.exposure)
        gamma = float(self.gamma)
        if not math.isfinite(exposure) or not math.isfinite(gamma) or gamma <= 0.0:
            raise ValueError("scene exposure must be finite and gamma must be positive")
        object.__setattr__(self, "exposure", exposure)
        object.__setattr__(self, "tone_map", str(self.tone_map).lower().replace("-", "_"))
        object.__setattr__(self, "gamma", gamma)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def field_bindings(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.fields))

    def add_mesh(self, mesh: Mesh | str | Path, material: Material | None = None) -> Scene:
        """Append a mesh object or load one from the high-level path API."""

        if isinstance(mesh, (str, Path)):
            selected = Mesh.load(mesh, material)
        elif isinstance(mesh, Mesh):
            selected = mesh if material is None else replace(mesh, material=material)
        else:
            raise TypeError("Scene.add_mesh requires a Mesh or filesystem path")
        self.meshes = (*self.meshes, selected)
        return self

    def add_light(self, light: Any) -> Scene:
        self.lights = (*self.lights, light)
        return self

    def add_directional_light(self, direction: Any, intensity: Any = 1.0, **options: Any) -> Scene:
        """Construct and append the high-level directional-light surface."""

        from spf.renderer.lighting import DirectionalLight

        return self.add_light(DirectionalLight(direction=direction, intensity=intensity, **options))

    def add_spherical_gaussian_light(
        self,
        axis: Any,
        sharpness: float,
        amplitude: Any = 1.0,
        **options: Any,
    ) -> Scene:
        from spf.renderer.lighting import SphericalGaussianLight

        return self.add_light(SphericalGaussianLight(mu=axis, lambda_=sharpness, amplitude=amplitude, **options))

    def add_splat(self, splat: Any) -> Scene:
        self.splats = (*self.splats, splat)
        return self

    def add_volume(self, volume: Any) -> Scene:
        self.volumes = (*self.volumes, volume)
        return self

    def add_geometry(self, primitive: Any, material: Material | None = None) -> Scene:
        """Append a tier-1 analytic patch or tier-2 field-geometry primitive.

        The primitive must declare a ``spf.renderer.geometry.GeometryTier``;
        :func:`spf.renderer.geometry.rasterize_geometry` routes it.  ``material``
        is attached as an attribute so the field plane can find it without a
        parallel list.
        """

        from spf.renderer.geometry import GeometryTier

        tier = getattr(primitive, "tier", None)
        if tier not in {GeometryTier.PATCH, GeometryTier.FIELD}:
            raise TypeError("Scene.add_geometry accepts tier-1 patches and tier-2 field geometry")
        if material is not None:
            self.geometry_materials[id(primitive)] = material
        self.geometry = (*self.geometry, primitive)
        return self

    add_patch = add_geometry
    add_height_field = add_geometry

    def bind_field(self, name: str, value: Any) -> Scene:
        self.fields[str(name)] = value
        return self

    add_field = bind_field

    @classmethod
    def compose(cls, *scenes: Scene) -> Scene:
        if not scenes:
            return cls()
        result = cls()
        for scene in scenes:
            if not isinstance(scene, Scene):
                raise TypeError("Scene.compose accepts Scene values")
            result.meshes += scene.meshes
            result.lights += scene.lights
            result.splats += scene.splats
            result.volumes += scene.volumes
            result.geometry += scene.geometry
            result.geometry_materials.update(scene.geometry_materials)
            result.fields.update(scene.fields)
            if scene.camera is not None:
                result.camera = scene.camera
            result.background = scene.background
            result.exposure = scene.exposure
            result.tone_map = scene.tone_map
            result.gamma = scene.gamma
        return result


@dataclass(frozen=True, slots=True)
class RenderResult:
    """Image plus visibility buffers and a complete execution report."""

    image: np.ndarray
    depth: np.ndarray
    coverage: np.ndarray
    report: RenderReport
    visible: tuple[tuple[VisibleSample | None, ...], ...]

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        depth = np.asarray(self.depth, dtype=np.float64)
        coverage = np.asarray(self.coverage, dtype=np.float64)
        if image.ndim != 3 or image.shape[-1] != 3 or depth.shape != image.shape[:2] or coverage.shape != depth.shape:
            raise ValueError("render result buffers have incompatible shapes")
        object.__setattr__(self, "image", image.copy())
        object.__setattr__(self, "depth", depth.copy())
        object.__setattr__(self, "coverage", coverage.copy())
        object.__setattr__(self, "visible", tuple(tuple(row) for row in self.visible))

    @property
    def color(self) -> np.ndarray:
        return self.image

    @property
    def shape(self) -> tuple[int, ...]:
        return self.image.shape

    def __iter__(self) -> Iterator[np.ndarray | RenderReport]:
        """Permit the design's ``image, report = renderer.render(...)`` form."""

        yield self.image
        yield self.report

    @property
    def dtype(self) -> np.dtype[Any]:
        return self.image.dtype

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.image, dtype=dtype)

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, item: Any) -> Any:
        return self.image[item]


@dataclass(slots=True)
class _Layer:
    image: np.ndarray
    depth: np.ndarray
    alpha: np.ndarray
    source: str
    premultiplied: bool = False


def _decode_payload(value: Any) -> Any:
    if isinstance(value, v02.LogPos):
        return float(value.value)
    if isinstance(value, v02.LogSigned):
        return float(value.value)
    if isinstance(value, np.ndarray) and value.dtype == object:
        result = np.empty(value.shape, dtype=np.float64)
        for index in np.ndindex(value.shape):
            result[index] = float(_decode_payload(value[index]))
        return result
    return value


def _normalize_image(value: Any, resolution: tuple[int, int], *, name: str, copy: bool = True) -> np.ndarray:
    width, height = resolution
    raw = _decode_payload(value)
    if isinstance(raw, Field):
        raw = _decode_payload(raw.representation.payload)
    array = np.asarray(raw)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} is complex; use coherent_intensity() explicitly before rendering")
    if array.dtype.kind not in "fiu":
        array = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    if array.shape == (height, width):
        array = array[..., None]
    if array.ndim == 0 or array.shape in {(height, width, 1), (height, width, 3), (3,)}:
        array = np.broadcast_to(array, (height, width, 3))
    else:
        raise ValueError(f"{name} must be scalar, RGB, (H, W), or (H, W, 3)")
    if copy:
        array = np.array(array, dtype=np.float64, copy=True)
    return array


def _normalize_layer(value: Any, resolution: tuple[int, int], *, name: str) -> _Layer:
    width, height = resolution
    image = value
    depth = None
    alpha = None
    premultiplied = False
    if isinstance(value, RenderResult):
        image, depth, alpha = value.image, value.depth, value.coverage
    elif isinstance(value, Mapping):
        image = value.get("image", value.get("color", value.get("radiance")))
        depth = value.get("depth")
        alpha = value.get("alpha", value.get("coverage"))
        premultiplied = bool(value.get("premultiplied", False))
    else:
        if hasattr(value, "image"):
            image = value.image
            depth = getattr(value, "depth", None)
            alpha = getattr(value, "coverage", getattr(value, "alpha", None))
            # Splat/transport results carry a transmittance plane and their
            # color is already the accumulated contribution over black.
            premultiplied = hasattr(value, "transmittance")
    if image is None:
        raise ValueError(f"{name} did not provide an image/color payload")
    image_array = _normalize_image(image, resolution, name=name)
    if depth is None:
        depth_array = np.full((height, width), math.inf, dtype=np.float64)
    else:
        depth_array = np.asarray(depth, dtype=np.float64)
        if depth_array.ndim == 0:
            depth_array = np.full((height, width), float(depth_array), dtype=np.float64)
        if depth_array.shape != (height, width) or np.any(np.isnan(depth_array)):
            raise ValueError(f"{name} depth must have shape (H, W) and no NaNs")
    if alpha is None:
        alpha_array = np.ones((height, width), dtype=np.float64)
    else:
        alpha_array = np.asarray(alpha, dtype=np.float64)
        if alpha_array.ndim == 0:
            alpha_array = np.full((height, width), float(alpha_array), dtype=np.float64)
        if alpha_array.shape != (height, width):
            raise ValueError(f"{name} alpha/coverage must have shape (H, W)")
    if not np.all(np.isfinite(alpha_array)) or np.any(alpha_array < 0.0) or np.any(alpha_array > 1.0):
        raise ValueError(f"{name} alpha/coverage must be in [0, 1]")
    return _Layer(image_array, depth_array, alpha_array, name, premultiplied)


def _composite_layer(
    image: np.ndarray,
    depth: np.ndarray,
    coverage: np.ndarray,
    layer: _Layer,
    *,
    opaque_depth: bool = False,
) -> None:
    """Composite a layer, respecting nearer finite depth and alpha."""

    finite_layer_depth = np.isfinite(layer.depth)
    nearer = finite_layer_depth & (layer.depth < depth - _EPS)
    no_depth = ~finite_layer_depth
    eligible = (nearer | no_depth) & (layer.alpha > 0.0)
    if not np.any(eligible):
        return
    alpha = np.where(eligible, layer.alpha, 0.0)[..., None]
    foreground = layer.image if layer.premultiplied else layer.image * alpha
    image[...] = np.where(eligible[..., None], foreground + image * (1.0 - alpha), image)
    coverage[...] = np.where(eligible, np.maximum(coverage, layer.alpha), coverage)
    if opaque_depth:
        depth[...] = np.where(nearer & (layer.alpha >= 1.0 - _EPS), layer.depth, depth)


def _call_component(component: Any, *, scene: Scene, camera: Any, resolution: tuple[int, int]) -> Any:
    function: Callable[..., Any] | None = None
    if callable(component):
        function = component
    elif hasattr(component, "render") and callable(component.render):
        function = component.render
    if function is None:
        return component
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        signature = None
    kwargs = {"scene": scene, "camera": camera, "resolution": resolution}
    if signature is not None:
        accepted = {
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in accepted
            or any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values())
        }
    try:
        return function(**kwargs)
    except TypeError:
        return function(camera, resolution)


def _component_payload(value: Any) -> tuple[Any, RenderReport | None]:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], RenderReport):
        return value[0], value[1]
    report = getattr(value, "report", None)
    return value, report if isinstance(report, RenderReport) else None


def _callable_identity(value: Any) -> str:
    """Return a stable module-qualified identity for a scene component.

    Reports must identify the executable value that produced a layer without
    serializing a bound method or object ``repr`` (both can contain process
    addresses and therefore cannot be compared across frames).
    """

    function = value if callable(value) else getattr(value, "render", None)
    if callable(function):
        module = getattr(function, "__module__", None)
        qualified = getattr(function, "__qualname__", None) or getattr(function, "__name__", None)
        if qualified is None:
            qualified = type(function).__qualname__
    else:
        component_type = type(value)
        module = component_type.__module__
        qualified = component_type.__qualname__
    return f"{module}.{qualified}" if module else str(qualified)


def _append_provenance_values(target: list[str], value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            _append_provenance_values(target, item)
        return
    text = str(value).strip()
    if text:
        target.append(text)


def _record_report_provenance(
    report: RenderReport | None,
    execution_classes: list[str],
    callables: list[str],
) -> None:
    """Copy child execution identities into the enclosing frame trace."""

    if report is None:
        return
    diagnostics = report.diagnostics
    for key in ("actual_execution_class", "execution_class", "execution"):
        _append_provenance_values(execution_classes, diagnostics.get(key))
    for key in ("actual_callable", "display_callback_identity", "selected_kernel"):
        _append_provenance_values(callables, diagnostics.get(key))


def _lazy_splat_render(
    splats: Sequence[Any],
    *,
    scene: Scene,
    camera: Any,
    resolution: tuple[int, int],
    error_budget: float,
    opaque_depth: np.ndarray | None = None,
) -> Any:
    try:
        module = importlib.import_module(".splats", package=__package__)
    except ModuleNotFoundError as exc:
        if exc.name in {"spf.renderer.splats", "spf.renderer"}:
            raise LookupError("splat renderer module is not available") from exc
        raise
    del scene  # the splat workload is self-contained once the camera is fixed
    width, height = resolution
    splat_camera_type = getattr(module, "Camera", None)
    if splat_camera_type is None:
        raise LookupError("splat module has no camera model")
    if isinstance(camera, splat_camera_type):
        if (camera.width, camera.height) != (width, height):
            raise ValueError("splat camera dimensions must match the render resolution")
        splat_camera = camera
    elif isinstance(camera, Camera):
        splat_camera = splat_camera_type(
            width,
            height,
            projection="orthographic" if camera.orthographic else "perspective",
            position=camera.position,
            target=camera.target,
            up=camera.up,
            ortho_height=camera.ortho_scale,
            fov_y=math.radians(camera.fov_y),
            near=camera.near,
            far=camera.far,
        )
    else:
        raise TypeError("scene splats require a scene.Camera or splats.Camera")
    function = getattr(module, "rasterize_splats", None)
    if not callable(function):
        raise LookupError("splat module has no rasterize_splats function")
    # An infinite scene budget means exact traversal, represented by a zero
    # early-termination budget in the splat workload.
    budget = 0.0 if math.isinf(error_budget) else error_budget
    return function(
        splats,
        splat_camera,
        image_shape=(height, width),
        error_budget=budget,
        background=(0.0, 0.0, 0.0),
        opaque_depth=opaque_depth,
        # The frame composer has not assigned a separate approximation budget
        # to this layer.  Keep traversal exact; direct splat callers may opt
        # into certified tail termination with their own local budget.
        early_terminate=False,
        compare_reference=False,
    )


class Renderer:
    """High-level CPU/reference renderer with explicit backend boundaries."""

    def __init__(
        self,
        backend: str | SFRBackend = "reference",
        *,
        precision: str | PrecisionContract = "balanced",
        error_budget: float | ErrorBudget = math.inf,
    ) -> None:
        requested = getattr(backend, "name", backend)
        self.requested_backend = str(requested).strip().lower().replace("-", "_")
        self.backend = get_backend(backend)
        self.backend.require_available()
        self.precision_contract = (
            precision if isinstance(precision, PrecisionContract) else PrecisionContract.for_mode(precision)
        )
        self.precision = cast(PrecisionMode, self.precision_contract.mode).value.lower()
        self.error_contract = (
            error_budget if isinstance(error_budget, ErrorBudget) else ErrorBudget(absolute=error_budget)
        )
        self.error_budget = float(self.error_contract.absolute)
        if math.isnan(self.error_budget) or self.error_budget < 0.0:
            raise ValueError("error_budget must be non-negative")

    def _validate_scene_backend(self) -> None:
        """Fail closed before a native backend can fall through to NumPy.

        The high-level scene composer currently owns a CPU/reference execution
        path.  Dense image, depth, and coverage arrays are valid terminal
        observations at this boundary; they do not make a native GPU scene
        implementation exist.  A future native scene backend must advertise
        the explicit ``hybrid_scene`` operation before it can be selected.
        """

        self.backend.require_available("hybrid_scene")
        capabilities = self.backend.capabilities
        if capabilities.native and not capabilities.supports_operation("hybrid_scene"):
            reason = capabilities.unsupported_reasons.get("hybrid_scene")
            detail = f": {reason}" if reason else ""
            raise NotImplementedError(
                f"SFR backend {capabilities.name!r} does not support 'hybrid_scene'{detail}"
            )

    def render(
        self,
        scene: Scene,
        camera: Any | None = None,
        *,
        resolution: int | Sequence[int] = (64, 64),
        exposure: float | None = None,
        tone_map: str | None = None,
        gamma: float | None = None,
        display: bool = False,
        return_result: bool = True,
        return_report: bool = False,
    ) -> RenderResult | np.ndarray | tuple[np.ndarray, RenderReport]:
        if not isinstance(scene, Scene):
            raise TypeError("Renderer.render requires a Scene")
        self._validate_scene_backend()
        actual_execution_classes: list[str] = [_SCENE_EXECUTION_CLASS]
        actual_callables: list[str] = [_SCENE_CALLABLE]
        selected_camera = camera or scene.camera or Camera()
        size = _resolution(resolution)
        width, height = size
        image = np.broadcast_to(np.asarray(scene.background, dtype=np.float64), (height, width, 3)).copy()
        depth = np.full((height, width), math.inf, dtype=np.float64)
        coverage = np.zeros((height, width), dtype=np.float64)
        visible_grid: list[list[VisibleSample | None]] = [[None] * width for _ in range(height)] if scene.meshes else []
        clipped_triangles = 0
        discarded_triangles = 0
        lighting_reduction_error = 0.0
        lighting_source_error_bound = 0.0
        lighting_source_ledger = v02.ErrorLedger()
        material_source_error_bound = 0.0
        material_source_ledger = v02.ErrorLedger()
        material_output_error_bound = 0.0
        material_output_error_map: np.ndarray = np.zeros((height, width), dtype=np.float64)
        material_source_unpropagated = False
        fallbacks: list[str] = []
        passes: list[CompilerPassRecord] = []
        decisions: list[RepresentationDecision] = []
        capabilities = [self.backend.name, "reference_depth_min", "reference_coverage"]
        ledger = v02.ErrorLedger()
        metrics = RenderMetrics(
            backend=self.backend.name,
            workload="hybrid_scene",
            resolution=(width, height),
            representation="hybrid_mesh_splat_volume_field",
            precision=self.precision,
            error_budget=self.error_budget,
            potential_elements=width * height * (len(scene.meshes) + len(scene.volumes)),
        )
        passes.append(
            CompilerPassRecord(
                "scene_validate", 0, len(scene.meshes) + len(scene.splats) + len(scene.volumes) + len(scene.fields)
            )
        )

        for mesh_index, mesh in enumerate(scene.meshes):
            if not isinstance(mesh, Mesh):
                fallbacks.append(f"mesh[{mesh_index}]:unsupported_type")
                continue
            actual_callables.append("spf.renderer.scene.rasterize_mesh")
            raster = rasterize_mesh(mesh, selected_camera, size, lights=scene.lights)
            clipped_triangles += raster.clipped_triangles
            discarded_triangles += raster.discarded_triangles
            lighting_reduction_error = max(lighting_reduction_error, raster.reduction_error_bound)
            lighting_source_error_bound = max(lighting_source_error_bound, raster.lighting_source_error_bound)
            lighting_source_ledger = _max_source_ledger(lighting_source_ledger, raster.lighting_source_ledger)
            material_source_error_bound = max(material_source_error_bound, raster.material_source_error_bound)
            material_source_ledger = _max_source_ledger(material_source_ledger, raster.material_source_ledger)
            material_output_error_bound = max(material_output_error_bound, raster.material_output_error_bound)
            material_source_unpropagated |= (
                raster.material_source_error_bound > 0.0 and not raster.material_source_propagated
            )
            winners = raster.depth < depth - _EPS
            assert raster.material_output_error_map is not None  # Normalized by RasterizedMesh.__post_init__.
            np.copyto(material_output_error_map, raster.material_output_error_map, where=winners)
            np.copyto(image, raster.image, where=winners[..., None])
            np.copyto(depth, raster.depth, where=winners)
            np.copyto(coverage, raster.coverage, where=winners)
            for sample in raster.samples:
                pixel_x, pixel_y = sample.pixel
                if winners[pixel_y, pixel_x]:
                    visible_grid[pixel_y][pixel_x] = sample
            metrics.field_operations += mesh.triangle_count
            metrics.reduction_operations += raster.pivot_reductions
            metrics.positive_logadds += raster.positive_logadds
            metrics.pivot_reductions += raster.pivot_reductions
            metrics.materialized_elements += len(raster.samples) * 3
            mesh_indices = mesh.indices
            if mesh_indices is None:  # pragma: no cover - Mesh validates this invariant
                raise RuntimeError("mesh indices were not initialized")
            metrics.bytes_read += int(mesh.vertices.nbytes + mesh_indices.nbytes)
            decisions.append(
                RepresentationDecision(
                    mesh.name,
                    "screen",
                    "triangle_mesh",
                    ("triangle_mesh", "dense_reference"),
                    float(mesh.triangle_count),
                    0.0,
                    "CPU reference rasterization",
                )
            )
        if clipped_triangles:
            capabilities.append("near_far_triangle_clipping")
        passes.append(
            CompilerPassRecord(
                "mesh_visibility",
                len(scene.meshes),
                len(scene.meshes),
                notes=(
                    "nearest finite depth wins",
                    f"clipped_triangles={clipped_triangles}",
                    f"discarded_triangles={discarded_triangles}",
                ),
            )
        )
        if lighting_reduction_error:
            ledger = ledger.plus(v02.ErrorLedger(reducer=lighting_reduction_error))

        # §19.2/§19.3 tiers 1 and 2.  Each primitive resolves its own visibility
        # (fixed-point z*, §20.2 tile cull), publishes a VisibleSampleBuffer,
        # shades it on LOG_POS word planes, and materializes once; the layer then
        # composites against the tier-0 depth buffer by nearest finite depth.
        for geometry_index, primitive in enumerate(scene.geometry):
            try:
                from spf.renderer.geometry import rasterize_geometry

                layer = rasterize_geometry(
                    primitive,
                    selected_camera,
                    size,
                    lights=scene.lights,
                    material=scene.geometry_materials.get(id(primitive)),
                    background=(0.0, 0.0, 0.0),
                )
            except (TypeError, ValueError, LookupError) as exc:
                fallbacks.append(f"geometry[{geometry_index}]:unsupported:{exc}")
                continue
            actual_callables.append(layer.report.diagnostics.get("actual_callable", "spf.renderer.geometry"))
            _composite_layer(
                image,
                depth,
                coverage,
                _normalize_layer(layer, size, name=f"geometry[{geometry_index}]"),
                opaque_depth=True,
            )
            _record_report_provenance(layer.report, actual_execution_classes, actual_callables)
            metrics.add(layer.report.metrics)
            ledger = ledger.plus(layer.report.ledger)
            capabilities.extend(layer.report.capabilities_used)
            capabilities.append(str(layer.tier.value))
            decisions.append(
                RepresentationDecision(
                    f"geometry[{geometry_index}]",
                    "screen",
                    str(layer.report.metrics.representation),
                    (str(layer.report.metrics.representation), "dense_reference"),
                    float(layer.report.metrics.materialized_elements),
                    float(layer.report.ledger.total_bound),
                    f"{layer.tier.value} after visibility",
                )
            )
        if scene.geometry:
            passes.append(
                CompilerPassRecord(
                    "tier_geometry_composition",
                    len(scene.geometry),
                    len(scene.geometry),
                    notes=("tier 1/2 visibility before field plane", "one materialization per tier layer"),
                )
            )

        if scene.splats:
            try:
                splat_value = _lazy_splat_render(
                    scene.splats,
                    scene=scene,
                    camera=selected_camera,
                    resolution=size,
                    error_budget=self.error_budget,
                    opaque_depth=depth,
                )
                _composite_layer(
                    image, depth, coverage, _normalize_layer(splat_value, size, name="splats"), opaque_depth=False
                )
                capabilities.append("splat_renderer")
                actual_callables.extend(
                    (
                        "spf.renderer.scene._lazy_splat_render",
                        "spf.renderer.splats.rasterize_splats",
                    )
                )
                splat_report = getattr(splat_value, "report", None)
                if isinstance(splat_report, RenderReport):
                    _record_report_provenance(splat_report, actual_execution_classes, actual_callables)
                    metrics.add(splat_report.metrics)
                    ledger = ledger.plus(splat_report.ledger)
                    fallbacks.extend(f"splats:{item}" for item in splat_report.fallbacks)
                    capabilities.extend(splat_report.capabilities_used)
                metrics.atoms_considered += len(scene.splats)
                decisions.append(
                    RepresentationDecision(
                        "splats",
                        "screen",
                        "gaussian_splat",
                        ("gaussian_splat", "dense_reference"),
                        float(len(scene.splats)),
                        0.0,
                        "lazy splat backend",
                    )
                )
            except LookupError as exc:
                fallbacks.append(f"splats:unavailable:{exc}")
            except (TypeError, ValueError) as exc:
                fallbacks.append(f"splats:invalid:{exc}")
        passes.append(
            CompilerPassRecord(
                "splat_composition",
                len(scene.splats),
                len(scene.splats),
                notes=("lazy import; explicit fallback on absence",),
            )
        )

        for volume_index, volume in enumerate(scene.volumes):
            try:
                value, volume_report = _component_payload(
                    _call_component(volume, scene=scene, camera=selected_camera, resolution=size)
                )
                actual_callables.extend(("spf.renderer.scene._call_component", _callable_identity(volume)))
                _composite_layer(image, depth, coverage, _normalize_layer(value, size, name=f"volume[{volume_index}]"))
                capabilities.append("volume_layer")
                if volume_report is not None:
                    _record_report_provenance(volume_report, actual_execution_classes, actual_callables)
                    metrics.add(volume_report.metrics)
                    ledger = ledger.plus(volume_report.ledger)
                    fallbacks.extend(f"volume[{volume_index}]:{item}" for item in volume_report.fallbacks)
                    capabilities.extend(volume_report.capabilities_used)
                decisions.append(
                    RepresentationDecision(
                        f"volume[{volume_index}]",
                        "screen",
                        "volume_layer",
                        ("volume_layer", "dense_reference"),
                        1.0,
                        0.0,
                        "explicit volume layer",
                    )
                )
            except (TypeError, ValueError, LookupError) as exc:
                fallbacks.append(f"volume[{volume_index}]:unsupported:{exc}")
        passes.append(CompilerPassRecord("volume_composition", len(scene.volumes), len(scene.volumes)))

        for field_name, binding in scene.fields.items():
            try:
                if isinstance(binding, Field) and binding.coherence is Coherence.COHERENT_COMPLEX:
                    raise TypeError(f"field {field_name!r} is coherent-complex; call coherent_intensity() explicitly")
                value, source_report = _component_payload(
                    _call_component(binding, scene=scene, camera=selected_camera, resolution=size)
                )
                actual_callables.extend(("spf.renderer.scene._call_component", _callable_identity(binding)))
                field_reports = [] if source_report is None else [source_report]
                if isinstance(value, Field) and value.coherence is Coherence.COHERENT_COMPLEX:
                    raise TypeError(f"field {field_name!r} is coherent-complex; explicit intensity conversion required")
                if isinstance(value, Field):
                    actual_callables.append("spf.renderer.compiler.materialize_field")
                    materialized, materialization_report = materialize_field(
                        value,
                        MaterializationRequest(
                            layout=SampleLayout((height, width)),
                            backend=self.backend.name,
                            precision=self.precision_contract,
                            error_budget=self.error_contract,
                        ),
                        backend=self.backend,
                        return_report=True,
                    )
                    value = materialized.values
                    field_reports.append(materialization_report)
                field_image = _normalize_image(
                    value,
                    size,
                    name=f"field[{field_name}]",
                    copy=False,
                )
                image += field_image
                capabilities.append("field_binding")
                for field_report in field_reports:
                    _record_report_provenance(field_report, actual_execution_classes, actual_callables)
                    metrics.add(field_report.metrics)
                    ledger = ledger.plus(field_report.ledger)
                    fallbacks.extend(f"field[{field_name}]:{item}" for item in field_report.fallbacks)
                    capabilities.extend(field_report.capabilities_used)
                metrics.field_operations += 1
                decisions.append(
                    RepresentationDecision(
                        str(field_name),
                        "screen",
                        "field_binding",
                        ("field_binding", "dense_reference"),
                        1.0,
                        0.0,
                        "additive radiance field",
                    )
                )
            except TypeError:
                raise
            except MaterializationError:
                # A required field that cannot satisfy its declared error
                # contract is not an optional component.  Returning a black
                # or partial frame would falsely certify missing content.
                raise
            except (ValueError, LookupError) as exc:
                raise MaterializationError(f"required field {field_name!r} failed: {exc}") from exc
        passes.append(
            CompilerPassRecord(
                "field_composition",
                len(scene.fields),
                len(scene.fields),
                notes=("positive radiance bindings are additive",),
            )
        )

        selected_exposure = scene.exposure if exposure is None else float(exposure)
        selected_gamma = scene.gamma if gamma is None else float(gamma)
        selected_tone = scene.tone_map if tone_map is None else str(tone_map).lower().replace("-", "_")
        if not math.isfinite(selected_exposure) or not math.isfinite(selected_gamma) or selected_gamma <= 0.0:
            raise ValueError("exposure must be finite and gamma must be positive")

        # ``material_output_error_bound`` is an audited linear-space bound
        # after the material BRDF and before tone/display.  Keep that output
        # quantity in the frame ledger; source bounds remain pre-BRDF
        # diagnostics and are never added as if they were image error.
        if material_output_error_bound > 0.0:
            ledger = ledger.plus(v02.ErrorLedger(materialization=material_output_error_bound))
        frame_output_error_bound = _display_error_bound(
            image,
            material_output_error_map,
            selected_exposure,
            selected_tone,
            selected_gamma,
            display,
        )
        linear_scale = float(np.max(np.abs(image))) if image.size else 0.0
        frame_contract = self.error_contract.as_contract()
        source_ledger = lighting_source_ledger.plus(material_source_ledger)
        source_error_bound = max(
            lighting_source_error_bound,
            material_source_error_bound,
            float(source_ledger.total_bound),
        )
        source_uncertain = source_error_bound > 0.0
        source_unpropagated = lighting_source_error_bound > 0.0 or material_source_unpropagated
        finite_frame_budget = any(
            math.isfinite(value)
            for value in (
                self.error_contract.absolute,
                self.error_contract.relative,
                self.error_contract.phase,
            )
        )
        if math.isfinite(self.error_contract.temporal) or self.error_contract.perceptual is not None:
            raise MaterializationError(
                "scene frame certification supports only absolute, relative, and phase ErrorBudget terms; "
                "temporal/perceptual contracts require a sequence or display consumer"
            )
        # A finite terminal certificate requires both a finite propagated
        # ledger and a finite contract.  An unbounded exploratory render is
        # deliberately reported as uncertified even when every source is
        # exact.
        frame_error_certified = False
        if finite_frame_budget and source_unpropagated:
            raise MaterializationError(
                "composed scene contains unpropagated pre-BRDF source uncertainty "
                f"(source_error_bound={source_error_bound!r}); cannot certify a finite ErrorBudget"
            )
        if finite_frame_budget:
            frame_error_certified = frame_contract.accepts(ledger, scale=linear_scale, phase_error=0.0)
            if not frame_error_certified:
                raise MaterializationError(
                    "composed scene exceeds requested ErrorBudget "
                    f"(absolute={ledger.total_bound!r}, "
                    f"relative={ledger.total_bound / max(linear_scale, np.finfo(np.float64).tiny)!r})"
                )

        if selected_exposure:
            image *= 2.0**selected_exposure
        if selected_tone in {"reinhard", "tone_reinhard"}:
            image /= 1.0 + np.maximum(image, 0.0)
        elif selected_tone not in {"none", "identity", ""}:
            fallbacks.append(f"tone:{selected_tone}:unsupported")
        if selected_gamma != 1.0:
            np.maximum(image, 0.0, out=image)
            np.power(image, 1.0 / selected_gamma, out=image)
        np.maximum(image, 0.0, out=image)
        if display:
            image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
            capabilities.append("display_quantization")
        passes.append(
            CompilerPassRecord("tone_display", 1, 1, notes=(selected_tone, "display" if display else "native"))
        )

        metrics.bytes_written += int(image.nbytes + depth.nbytes + coverage.nbytes)
        metrics.peak_memory = int(image.nbytes + depth.nbytes + coverage.nbytes)
        metrics.fallback_conversions = max(metrics.fallback_conversions, len(fallbacks))
        execution_trace = tuple(dict.fromkeys(actual_execution_classes))
        callable_trace = tuple(dict.fromkeys(actual_callables))
        report = RenderReport(
            metrics=metrics,
            ledger=ledger,
            passes=tuple(passes),
            decisions=tuple(decisions),
            fallbacks=tuple(fallbacks),
            capabilities_used=tuple(dict.fromkeys(capabilities)),
            diagnostics={
                "mesh_count": len(scene.meshes),
                "light_count": len(scene.lights),
                "splat_count": len(scene.splats),
                "volume_count": len(scene.volumes),
                "field_count": len(scene.fields),
                "visible_samples": int(np.count_nonzero(coverage)),
                "clipped_triangles": clipped_triangles,
                "discarded_triangles": discarded_triangles,
                "fallback_count": len(fallbacks),
                # ``backend_*`` is retained for compatibility and describes
                # the selected backend.  The explicit requested/selected and
                # actual fields keep a semantic CPU frame from being reported
                # as native merely because a backend was selected.
                "requested_backend": self.requested_backend,
                "selected_backend": self.backend.name,
                "selected_backend_native": bool(self.backend.capabilities.native),
                "selected_backend_execution_class": self.backend.capabilities.execution_class,
                "backend_native": bool(self.backend.capabilities.native),
                "backend_execution_class": self.backend.capabilities.execution_class,
                "actual_execution_class": _SCENE_EXECUTION_CLASS,
                "actual_callable": _SCENE_CALLABLE,
                "actual_execution_classes": execution_trace,
                "actual_callables": callable_trace,
                "actual_child_execution_classes": tuple(
                    value for value in execution_trace if value != _SCENE_EXECUTION_CLASS
                ),
                "actual_child_callables": tuple(value for value in callable_trace if value != _SCENE_CALLABLE),
                "backend_execution_provenance": _SCENE_EXECUTION_CLASS,
                "frame_error_certified": frame_error_certified,
                "frame_error_scale": linear_scale,
                "frame_error_domain": "linear_radiance_before_display",
                "material_display_error_bound": frame_output_error_bound,
                "material_display_error_domain": "normalized_display" if display else "post_tone_float",
                "material_display_error_scope": "material_source_only",
                "material_output_error_bound": material_output_error_bound,
                "material_source_propagated": material_source_error_bound > 0.0 and not material_source_unpropagated,
                # These source quantities are intentionally separate from
                # ``ledger``: they are measured before BRDF/display and do
                # not claim a numeric image error bound at this boundary.
                "source_error_bound": source_error_bound,
                "source_ledger": {name: getattr(source_ledger, name) for name in _SOURCE_LEDGER_FIELDS},
                "lighting_source_error_bound": lighting_source_error_bound,
                "lighting_source_ledger": {
                    name: getattr(lighting_source_ledger, name) for name in _SOURCE_LEDGER_FIELDS
                },
                "material_source_error_bound": material_source_error_bound,
                "material_source_ledger": {
                    name: getattr(material_source_ledger, name) for name in _SOURCE_LEDGER_FIELDS
                },
                "source_error_scope": "pre_brdf_source_radiance",
                "source_error_status": (
                    "unpropagated_pre_brdf"
                    if source_unpropagated
                    else "propagated_to_output"
                    if source_uncertain
                    else "none"
                ),
            },
        )
        result = RenderResult(
            image=image,
            depth=depth,
            coverage=coverage,
            report=report,
            visible=tuple(tuple(row) for row in visible_grid) if visible_grid else ((None,) * width,) * height,
        )
        if return_report:
            return result.image, result.report
        if not return_result:
            return result.image
        return result

    def render_image(self, scene: Scene, **options: Any) -> np.ndarray:
        options["return_result"] = False
        return self.render(scene, **options)  # type: ignore[return-value]


def render(scene: Scene, **options: Any) -> RenderResult | np.ndarray | tuple[np.ndarray, RenderReport]:
    """Convenience wrapper around :class:`Renderer`."""

    return Renderer(
        options.pop("backend", "reference"),
        precision=options.pop("precision", "balanced"),
        error_budget=options.pop("error_budget", math.inf),
    ).render(scene, **options)


__all__ = [
    "Camera",
    "Mesh",
    "RasterizedMesh",
    "RenderResult",
    "Renderer",
    "Scene",
    "Triangle",
    "VisibleSample",
    "coverage_mask",
    "depth_min",
    "nearest_visible",
    "rasterize_mesh",
    "render",
    "select_nearest",
]
