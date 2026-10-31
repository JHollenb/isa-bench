"""One observable boundary for ocean geometry, typed radiance, and display.

Camera vectors, intersection, normals and reflection live in the conventional
control plane. The log profile composes positive radiance in guarded Q24 log
codes and emits pixels through a combined Q8 exposure/tone/gamma LUT. A matched
FP32 profile uses the same visibility and sampling algorithm. Native point
samplers can supply height/gradient without creating any ocean raster.
"""

from __future__ import annotations

import ctypes
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, cast

import numpy as np

from spf.renderer.compiler import FieldGraph, SFRNode
from spf.renderer.model import Coherence, CoordinateChart, Domain, Semantic
from spf.renderer.v02 import FieldKind


def _radiance_graph(profile: str, exposure: float) -> FieldGraph:
    """Build the positive field plane between control geometry and display.

    Geometry supplies positive angular/visibility factors at a query; those
    are external typed field inputs, not materialized arrays. The C lowering
    fuses this fixed program while its measured counters count each reduction.
    """
    nodes: list[SFRNode] = []
    metadata = {
        "value_kind": FieldKind.LOG_POS,
        "semantic": Semantic.RADIANCE,
        "domain": Domain.SCREEN_TILE,
        "coherence": Coherence.INCOHERENT_POSITIVE,
        "chart": CoordinateChart.identity(2),
    }

    def product(output: str, a: str, b: str) -> None:
        nodes.append(
            SFRNode(
                "v02_log_mul",
                output,
                (a, b),
                metadata,
                params={"guard_bits": 24, "consumer_profile": profile},
                annotations={"numeric_reduction": False, "materializes": False},
            )
        )

    def add(output: str, a: str, b: str) -> None:
        nodes.append(
            SFRNode(
                "v02_logadd_pos",
                output,
                (a, b),
                metadata,
                params={"reducer": "interpolated_guard_rom" if profile == "log" else "fp32", "guard_bits": 24},
                annotations={"numeric_reduction": True, "materializes": False},
            )
        )

    def power(output: str, value: str, exponent: int) -> None:
        nodes.append(
            SFRNode(
                "v02_tone",
                output,
                (value,),
                metadata,
                params={"mode": "power", "power": exponent, "guard_bits": 24},
                annotations={"numeric_reduction": False, "materializes": False},
            )
        )

    def complement(output: str, value: str) -> None:
        nodes.append(
            SFRNode(
                "v02_logsub_pos",
                output,
                ("positive_one", value),
                metadata,
                params={"reducer": "geometric_log_complement_rom" if profile == "log" else "fp32"},
                annotations={"numeric_reduction": True, "materializes": False},
            )
        )

    def sky(prefix: str) -> None:
        power(prefix + "_glow_power", prefix + "_sun_alignment", 420)
        product(prefix + "_glow_gain", prefix + "_glow_power", "glow_scale")
        power(prefix + "_disk_power", prefix + "_sun_alignment", 20000)
        product(prefix + "_disk_gain", prefix + "_disk_power", "disk_scale")
        product(prefix + "_horizon", "horizon_color", prefix + "_one_minus_blend")
        product(prefix + "_zenith", "zenith_color", prefix + "_blend")
        product(prefix + "_glow", "glow_color", prefix + "_glow_gain")
        product(prefix + "_disk", "disk_color", prefix + "_disk_gain")
        add(prefix + "_base", prefix + "_horizon", prefix + "_zenith")
        add(prefix + "_with_glow", prefix + "_base", prefix + "_glow")
        add(prefix + "_radiance", prefix + "_with_glow", prefix + "_disk")

    sky("reflection")
    sky("background")
    power("fresnel_power", "one_minus_view_cosine", 5)
    product("fresnel_tail", "fresnel_power", "fresnel_tail_scale")
    add("fresnel", "fresnel_floor", "fresnel_tail")
    complement("one_minus_fresnel", "fresnel")
    power("highlight_power", "half_vector_alignment", 220)
    product("highlight_gain", "highlight_power", "highlight_scale")
    complement("one_minus_transmission", "optical_transmission")
    product("depth_term", "depth_color", "one_minus_fresnel")
    product("reflection_term", "reflection_radiance", "fresnel")
    product("highlight_term", "highlight_gain", "highlight_color")
    add("surface_base", "depth_term", "reflection_term")
    add("surface", "surface_base", "highlight_term")
    product("attenuated_surface", "surface", "optical_transmission")
    product("haze_term", "haze_color", "one_minus_transmission")
    add("water_radiance", "attenuated_surface", "haze_term")
    # Binary visibility masks are positive fields. The native lowering
    # chooses the nonzero branch and eliminates the masked products exactly.
    product("visible_water", "water_radiance", "water_mask_gain")
    product("visible_background", "background_radiance", "sky_mask_gain")
    add("radiance", "visible_water", "visible_background")
    nodes.append(
        SFRNode(
            "v02_display_convert",
            "pixels",
            ("radiance",),
            {**metadata, "value_kind": FieldKind.DISPLAY},
            params={
                "exposure": exposure,
                "tone": "reinhard_0.65",
                "gamma": 1 / 2.2,
                "output": "uint8_rgb",
                "profile": profile,
            },
            annotations={"phase_observable": False, "materializes": True},
        )
    )
    graph = FieldGraph.from_nodes(nodes, outputs=("pixels",))
    return graph


def _lower_radiance(graph: FieldGraph) -> int:
    """Validate the fixed graph accepted by the versioned fused C lowering."""
    if not graph.nodes:
        raise ValueError("empty ocean radiance program")
    terminal = graph.nodes[-1]
    if terminal.op != "v02_display_convert" or terminal.params.get("profile") not in ("fp32", "log"):
        raise ValueError("ocean radiance program has no supported display boundary")
    expected = _radiance_graph(str(terminal.params["profile"]), float(terminal.params["exposure"]))

    def identity(node: SFRNode) -> Any:
        return node.op, node.output, node.inputs, dict(node.params)

    if [identity(n) for n in graph.nodes] != [identity(n) for n in expected.nodes]:
        raise ValueError("modified radiance graph needs a compatible native lowering")
    if any(n.field_metadata.coherence is Coherence.COHERENT_COMPLEX for n in graph.nodes):
        raise TypeError("ordinary ocean radiance cannot use coherent complex arithmetic")
    if any(
        (a.value_kind, a.field_metadata.semantic, a.field_metadata.domain, a.field_metadata.coherence)
        != (b.value_kind, b.field_metadata.semantic, b.field_metadata.domain, b.field_metadata.coherence)
        for a, b in zip(graph.nodes, expected.nodes, strict=True)
    ):
        raise TypeError("ocean radiance program metadata disagrees with its native lowering")
    return int(terminal.params["profile"] == "log")


class _DisplayStats(ctypes.Structure):
    _fields_: ClassVar[list[tuple[str, Any]]] = [
        (name, ctypes.c_uint64)
        for name in (
            "pixels",
            "water_pixels",
            "height_queries",
            "gradient_queries",
            "positive_products",
            "positive_sums",
            "display_lookups",
            "positive_complements",
            "scalar_log_conversions",
            "scalar_exp2_conversions",
            "callback_invocations",
            "scalar_callback_input_conversions",
            "scalar_callback_output_conversions",
        )
    ]


@lru_cache(maxsize=1)
def _library() -> Any:
    root = Path(__file__).resolve().parents[3]
    suffix = "dylib" if sys.platform == "darwin" else "so"
    explicit = os.environ.get("SPF_OCEAN_DISPLAY_LIBRARY")
    candidates = (
        [Path(explicit)]
        if explicit
        else [
            root / "native" / f"libspf_ocean_display.{suffix}",
            root / "native/build" / f"libspf_ocean_display.{suffix}",
        ]
    )
    for path in candidates:
        if not path.is_file():
            continue
        lib: Any = ctypes.CDLL(str(path))
        lib.spf_ocean_display_abi_version.argtypes = []
        lib.spf_ocean_display_abi_version.restype = ctypes.c_int
        abi_version = int(lib.spf_ocean_display_abi_version())
        if abi_version not in (2, 3):
            raise RuntimeError("ocean display ABI mismatch; rebuild the library")
        # ABI v2 owns an eight-counter record.  ABI v3 extends that record with
        # scalar and callback counters, so a current bridge may safely pass a
        # larger zero-initialized record to v2 while marking those fields
        # unavailable.  A v3 library must prove the complete record size before
        # ctypes points it at the extended structure.
        try:
            stats_size = lib.spf_ocean_display_stats_size
        except AttributeError:
            stats_size = None
        if abi_version == 3:
            if stats_size is None:
                raise RuntimeError("ocean display ABI v3 lacks its stats size probe; rebuild the library")
            stats_size.argtypes = []
            stats_size.restype = ctypes.c_int
            if stats_size() != ctypes.sizeof(_DisplayStats):
                raise RuntimeError("ocean display stats ABI mismatch; rebuild the library")
            scalar_counters_available = True
        else:
            if stats_size is not None:
                stats_size.argtypes = []
                stats_size.restype = ctypes.c_int
                if stats_size() != 8 * ctypes.sizeof(ctypes.c_uint64):
                    raise RuntimeError("legacy ocean display stats ABI mismatch; rebuild the library")
            scalar_counters_available = False
        lib._sfr_ocean_display_abi_version = abi_version
        lib._sfr_ocean_display_scalar_counters_available = scalar_counters_available
        lib._sfr_ocean_positive_basis = hasattr(lib, "spf_ocean_display_positive_basis_version")
        ptr = ctypes.c_void_p
        lib.spf_ocean_display.argtypes = (
            [ptr] * 3
            + [ctypes.c_int] * 2
            + [ctypes.c_float] * 2
            + [ptr] * 2
            + [ctypes.c_int] * 4
            + [ptr] * 7
            + [ctypes.POINTER(_DisplayStats)]
        )
        lib.spf_ocean_display.restype = ctypes.c_int
        return lib
    if explicit:
        raise RuntimeError(f"requested ocean display library does not exist: {explicit}")
    return None


def ocean_display_available() -> bool:
    return _library() is not None


def _callback_trace(sampler: Any) -> dict[str, Any]:
    """Describe the callable retained by a native sampler exactly.

    ``PreparedOceanImage`` has a Python method named ``points`` for the
    source-side batch API, but the display kernel invokes the C callback held
    in ``_sample`` directly.  Use the function pointer object itself when a
    sampler exposes it and keep an explicit unknown state for opaque test or
    third-party samplers instead of guessing from the source method name.
    """

    callback = getattr(sampler, "_sample", None)
    if callback is None:
        callback = getattr(sampler, "callback", None)
    name = getattr(callback, "__name__", None)
    pointer = getattr(sampler, "function_ptr", None)
    address = getattr(pointer, "value", None)
    return {
        "name": str(name) if name else None,
        "address": int(address) if address is not None else None,
        "status": "exact" if name else "opaque",
    }


def _attach_display_costs(source: Any, *, selected_sampling: str, costs: dict[str, Any]) -> Any:
    """Carry the image boundary's cost evidence onto the bound plan.

    ``OceanExecutionPlan`` is frozen and stores an immutable diagnostics map,
    so replacing the plan is the only way to retain calibration evidence for
    a later report.  Lightweight compatibility sources may expose a mutable
    plan with a different shape; those remain usable and keep the costs on
    the image object itself.
    """

    plan = getattr(source, "plan", None)
    if plan is None or not hasattr(plan, "diagnostics"):
        return plan
    diagnostics = dict(getattr(plan, "diagnostics", {}))
    diagnostics["display_selected_sampling"] = selected_sampling
    diagnostics["display_cost_model"] = dict(costs)
    model = str(costs.get("model", ""))
    diagnostics["display_cost_status"] = "measured" if model.startswith("measured_") else "heuristic"
    try:
        updated = replace(plan, diagnostics=diagnostics)
        source.plan = updated
        return updated
    except (AttributeError, TypeError, ValueError):
        # A third-party source may expose a read-only plan or a non-dataclass
        # record.  Its ``PreparedOceanImage`` diagnostics still carry the
        # complete cost record, so this metadata attachment is best effort.
        return plan


def _readonly(value: Any, dtype: Any = np.float32) -> np.ndarray:
    array = np.array(value, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _unit(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.linalg.norm(array) < 1e-8:
        raise ValueError("camera vectors must be finite nonzero 3-vectors")
    return np.asarray(array / np.linalg.norm(array), dtype=np.float32)


@dataclass(frozen=True)
class OceanCamera:
    eye: Any = (9.0, 3.0, 4.2)
    direction: Any = (0.0, 1.0, -0.115)
    up: Any = (0.0, 0.0, 1.0)
    horizontal_tangent: float = 0.7

    def __post_init__(self) -> None:
        eye = _readonly(self.eye)
        if eye.shape != (3,) or not np.all(np.isfinite(eye)):
            raise ValueError("camera eye must be a finite 3-vector")
        if not math.isfinite(self.horizontal_tangent) or self.horizontal_tangent <= 0:
            raise ValueError("camera field of view must be positive and finite")
        direction = _unit(self.direction)
        right = _unit(np.cross(direction, _unit(self.up)))
        up = _unit(np.cross(right, direction))
        object.__setattr__(self, "eye", eye)
        object.__setattr__(self, "direction", _readonly(direction))
        object.__setattr__(self, "up", _readonly(up))

    def parameters(self, resolution: tuple[int, int]) -> np.ndarray:
        height, width = resolution
        right = _unit(np.cross(self.direction, self.up))
        return _readonly(
            np.concatenate(
                (self.direction, right, self.up, [self.horizontal_tangent, self.horizontal_tangent * height / width])
            )
        )


@lru_cache(maxsize=1)
def _logadd_rom() -> np.ndarray:
    gap = np.arange(26 * 256 + 1, dtype=np.float64) / 256
    return _readonly(np.rint(np.log2(1 + np.exp2(-gap)) * 2**24), np.int32)


@lru_cache(maxsize=1)
def _logsub_rom() -> np.ndarray:
    ticks = np.exp2(np.arange(29, dtype=np.float64))[:, None] * (1 + np.arange(65)[None, :] / 64)
    logs = np.log2(-np.expm1(-ticks / 2**24 * np.log(2)))
    return _readonly(np.rint(logs * 2**24).ravel(), np.int32)


@lru_cache(maxsize=16)
def _display_lut(exposure: float = 0.0) -> np.ndarray:
    codes = np.arange(65536, dtype=np.uint16).view(np.int16).astype(np.float64)
    # Form the sigmoid stably even for extreme but finite exposure requests.
    log_radiance = codes / 256 + exposure - math.log2(0.65)
    positive = log_radiance >= 0
    mapped = np.empty_like(log_radiance)
    mapped[positive] = 1 / (1 + np.exp2(-log_radiance[positive]))
    z = np.exp2(log_radiance[~positive])
    mapped[~positive] = z / (1 + z)
    result = np.rint(mapped ** (1 / 2.2) * 255).astype(np.uint8)
    result[32768] = result[32767] = 0
    return _readonly(result, np.uint8)


@dataclass(frozen=True)
class OceanImage:
    pixels: np.ndarray
    linear: np.ndarray | None
    diagnostics: Any = field(default_factory=dict)


class PreparedOceanDisplay:
    """Owned compact camera and radiance program; never caches output pixels."""

    def __init__(
        self,
        *,
        resolution: tuple[int, int] = (640, 1024),
        length: Any = (128.0, 128.0),
        grid_origin: Any = (0.0, 0.0),
        camera: OceanCamera | None = None,
        profile: str = "log",
        exposure: float = 0.0,
        intersections: int = 7,
    ) -> None:
        if len(resolution) != 2 or any(isinstance(x, bool) or int(x) != x or x < 1 for x in resolution):
            raise ValueError("display resolution must be two positive integers")
        self.resolution = (int(resolution[0]), int(resolution[1]))
        length = np.broadcast_to(length, (2,)).astype(float)
        if not np.all(np.isfinite(length)) or np.any(length <= 0):
            raise ValueError("ocean physical lengths must be positive and finite")
        self.length = tuple(float(x) for x in length)
        self.camera = camera or OceanCamera()
        self._set_grid_origin(grid_origin)
        if profile not in ("log", "fp32"):
            raise ValueError("ocean display profile must be 'log' or 'fp32'")
        if not math.isfinite(exposure):
            raise ValueError("display exposure must be finite")
        if isinstance(intersections, bool) or int(intersections) != intersections or not 1 <= intersections <= 64:
            raise ValueError("intersection count must be in [1,64]")
        self.profile, self.exposure, self.intersections = profile, float(exposure), int(intersections)
        self.graph = _radiance_graph(profile, self.exposure)
        self._native_profile = _lower_radiance(self.graph)
        self.parameters = _readonly(
            np.concatenate((self.camera.parameters(self.resolution), np.asarray([self.exposure])))
        )
        self.logadd_rom, self.logsub_rom, self.display_lut = _logadd_rom(), _logsub_rom(), _display_lut(self.exposure)

    def _set_grid_origin(self, origin: Any) -> None:
        origin_array = np.asarray(origin, dtype=np.float64)
        if origin_array.shape != (2,) or not np.all(np.isfinite(origin_array)):
            raise ValueError("grid origin must be finite physical (y, x) coordinates")
        self.grid_origin = (float(origin_array[0]), float(origin_array[1]))
        # This shader's world-space lighting is translation invariant in x/y.
        # Translating the camera into the grid's local frame therefore gives
        # exactly (world_position - grid_origin) at every bilinear lookup.
        # Native point callbacks retain the original world-space camera.
        grid_eye = np.asarray(self.camera.eye, dtype=np.float64).copy()
        grid_eye[:2] -= origin_array[::-1]
        if np.any(np.abs(grid_eye) > np.finfo(np.float32).max):
            raise ValueError("camera relative to grid origin must fit FP32")
        self._grid_eye = _readonly(grid_eye)

    def execute(self, fields: Any = None, *, sampler: Any = None, return_linear: bool = False) -> OceanImage:
        if (fields is None) == (sampler is None):
            raise ValueError("supply exactly one joint field result or native sampler")
        # CUDA ocean samplers own a fused image terminal.  They retain the
        # packed support on device and return only RGB (plus optional linear
        # RGB), so never route them through the CPU callback ABI below.
        if sampler is not None:
            cuda_execute = getattr(sampler, "cuda_display", None)
            if callable(cuda_execute):
                return cast(OceanImage, cuda_execute(self, return_linear=return_linear))
        lib = _library()
        if lib is None:
            raise RuntimeError("native ocean display is unavailable; build native/sfr_ocean_display.c")
        conversion_elements = 0
        buffers: list[np.ndarray] = []
        rows = columns = 1
        function_ptr = context_ptr = None
        source_details = {}
        source_stats_before: dict[str, Any] = {}
        source_diagnostics_before: dict[str, Any] = {}
        callback_trace: dict[str, Any] = {"name": None, "address": None, "status": "not_applicable"}
        if sampler is not None:
            function_ptr = sampler.function_ptr
            context_ptr = sampler.context_ptr
            callback_trace = _callback_trace(sampler)
            source_details = dict(getattr(sampler, "diagnostics", {}))
            source_diagnostics_before = dict(source_details)
            source_details["display_callback"] = dict(callback_trace)
            if hasattr(getattr(sampler, "stats", None), "to_dict"):
                source_stats_before = sampler.stats.to_dict()
            if not function_ptr or not context_ptr:
                raise ValueError("native sampler must retain a non-null callback and context")
        else:
            values = (fields.height, fields.grad_y, fields.grad_x) if hasattr(fields, "height") else tuple(fields)
            if len(values) != 3:
                raise ValueError("ocean display requires height, dy, dx")
            for value in values:
                array = np.asarray(value)
                if (
                    array.ndim != 2
                    or array.dtype.kind not in "fiu"
                    or array.size == 0
                    or not np.all(np.isfinite(array))
                ):
                    raise ValueError("ocean height and slopes must be finite 2D arrays")
                if array.dtype != np.float32 or not array.flags.c_contiguous:
                    conversion_elements += array.size
                with np.errstate(over="ignore"):
                    converted = np.ascontiguousarray(array, dtype=np.float32)
                if array.dtype != np.float32 and not np.all(np.isfinite(converted)):
                    raise ValueError("ocean height and slopes must be representable as finite FP32 values")
                buffers.append(converted)
            if any(value.shape != buffers[0].shape for value in buffers):
                raise ValueError("ocean height and slope shapes must match")
            rows, columns = buffers[0].shape
        pixels = np.empty((*self.resolution, 3), np.uint8)
        linear = np.empty((*self.resolution, 3), np.float32) if return_linear else None
        stats = _DisplayStats()

        def ptr(value: np.ndarray | None) -> Any:
            return None if value is None else ctypes.c_void_p(value.ctypes.data)

        inputs = [ptr(value) for value in buffers] if buffers else [None] * 3
        status = lib.spf_ocean_display(
            *inputs,
            rows,
            columns,
            *self.length,
            ptr(self.parameters),
            ptr(self.camera.eye if sampler is not None else self._grid_eye),
            self.resolution[1],
            self.resolution[0],
            self.intersections,
            self._native_profile,
            ptr(self.logadd_rom),
            ptr(self.logsub_rom),
            ptr(self.display_lut),
            function_ptr,
            context_ptr,
            ptr(pixels),
            ptr(linear),
            ctypes.byref(stats),
        )
        if status:
            raise RuntimeError(f"native ocean display or sampler rejected execution: {status}")
        if sampler is not None:
            source_diagnostics_after = dict(getattr(sampler, "diagnostics", {}))
            source_details = dict(source_diagnostics_after)
            source_details["display_callback"] = dict(callback_trace)
            tracked_diagnostics = (
                "jet_queries",
                "sparse_queries",
                "tile_builds",
                "residual_terms_evaluated",
            )
            if any(
                name in source_diagnostics_before or name in source_diagnostics_after
                for name in tracked_diagnostics
            ):
                source_details["display_sampler_diagnostics_before"] = source_diagnostics_before
                source_details["display_sampler_diagnostics_after"] = source_diagnostics_after
                source_details["display_sampler_counter_deltas_during_display"] = {
                    name: int(source_diagnostics_after.get(name, 0)) - int(source_diagnostics_before.get(name, 0))
                    for name in tracked_diagnostics
                    if name in source_diagnostics_before or name in source_diagnostics_after
                }
            if source_stats_before:
                source_stats_after = sampler.stats.to_dict()
                source_details["native_stats_after"] = source_stats_after
                source_details["native_counter_deltas_during_display"] = {
                    name: int(value) - int(source_stats_before[name])
                    for name, value in source_stats_after.items()
                    if isinstance(value, int) and name not in {"flags", "phase_bits", "arithmetic_boundary"}
                }
                source_details["counter_scope"] = (
                    "deltas cover visible field callbacks in this display; cumulative after-state "
                    "also includes preparation and earlier frames"
                )
        scalar_counters_available = bool(
            getattr(lib, "_sfr_ocean_display_scalar_counters_available", True)
        )
        diagnostics: dict[str, Any] = {name: int(getattr(stats, name)) for name, _ in _DisplayStats._fields_}
        scalar_conversion_elements = int(
            stats.scalar_log_conversions
            + stats.scalar_exp2_conversions
            + stats.scalar_callback_input_conversions
            + stats.scalar_callback_output_conversions
        )
        positive_basis = self.profile == "log" and bool(getattr(lib, "_sfr_ocean_positive_basis", False))
        diagnostics.update(
            {
                "backend": "native_cpu",
                "display_abi_version": int(getattr(lib, "_sfr_ocean_display_abi_version", 3)),
                "profile": self.profile,
                "geometry_algebra": "fp32_control",
                "radiance_algebra": "guarded_q24_log_positive" if self.profile == "log" else "fp32_positive",
                "radiance_reducer": "shared_positive_basis_matrix" if positive_basis else
                "interpolated_logadd_rom" if self.profile == "log" else "fp32_sum",
                "positive_reduction_accumulator": "fp64_nonnegative" if positive_basis else
                "q24_log" if self.profile == "log" else "fp32",
                "positive_reducer_finalizations": 3*int(stats.pixels)+int(stats.water_pixels)
                if positive_basis else int(stats.positive_sums),
                "positive_matrix_relative_error_bound": 1e-6 if positive_basis else 0,
                "positive_matrix_absolute_pruning_bound": 7*2.0**-128 if positive_basis else 0,
                "display_boundary": "q8_log_to_u8_lut" if self.profile == "log" else "fp32_tone_gamma",
                "source": "native_queries" if sampler is not None else "materialized_height_and_slopes",
                "display_callback": dict(callback_trace),
                "display_callback_identity": callback_trace["name"],
                "display_callback_address": callback_trace["address"],
                "grid_origin_yx": self.grid_origin,
                "source_diagnostics": source_details,
                "field_ir": [
                    {
                        "op": n.op,
                        "output": n.output,
                        "inputs": n.inputs,
                        "value_kind": n.value_kind.value,
                        "effects": dict(n.annotations),
                    }
                    for n in self.graph.nodes
                ],
                "field_ir_role": "semantic_source_program",
                "field_lowering": "shared_positive_basis_to_native_cpu" if positive_basis else
                "validated_static_radiance_graph_to_native_cpu",
                "lowered_positive_reduction": {
                    "op": "v02_reduce_positive",
                    "basis_shape": [7, 3],
                    "sky_terms": 4,
                    "water_terms": 7,
                    "weight_decode": "once_per_shared_term",
                    "accumulator": "fp64_nonnegative",
                    "finalizations_per_rgb": 3,
                    "common_factors": ["fresnel", "optical_transmission"],
                    "dense_intermediate_elements": 0,
                } if positive_basis else None,
                "visibility_lowering": "eliminate_zero_mask_products_and_select_branch",
                "conversion_elements": conversion_elements,
                "array_conversion_elements": conversion_elements,
                "scalar_conversion_elements": scalar_conversion_elements,
                "scalar_conversion_counters_available": scalar_counters_available,
                "conversion_counter_scope": (
                    "array_conversion_elements counts Python field-array elements converted to contiguous FP32; "
                    "scalar counters count SFR log/exp2 values and physical y/x plus callback outputs; "
                    "FP32 profile scalar counters remain zero"
                    if scalar_counters_available
                    else "array conversion count is valid; ABI v2 has no scalar/callback conversion counters"
                ),
                "materialized_ocean_elements": 0 if sampler is not None else 3 * rows * columns,
                "materialized_display_elements": pixels.size + (0 if linear is None else linear.size),
                "dense_ray_elements": 0,
                "dense_shading_intermediates": 0,
                "display_log_quantization_max_bits": 1 / 512 if self.profile == "log" else 0,
                "logadd_interpolation_error_bits": 5e-7 if self.profile == "log" else 0,
                "log_complement_interpolation_error_bits": 4.5e-5 if self.profile == "log" else 0,
                "reference_boundary_note": (
                    "visibility/normalization uses FP32; positive geometric factors "
                    "enter through scalar logarithms; the shared positive basis reduces in FP64; "
                    "no integer-only claim"
                    if positive_basis else
                    "visibility/normalization uses FP32; positive factors enter through scalar logarithms"
                    if self.profile == "log" else "geometry and radiance use the unchanged FP32 control"
                ),
            }
        )
        return OceanImage(pixels, linear, MappingProxyType(diagnostics))

    render = execute


def prepare_ocean_display(**options: Any) -> PreparedOceanDisplay:
    return PreparedOceanDisplay(**options)


class PreparedOceanImage:
    """Bind an owned ocean timeline to its visible RGB observation boundary.

    ``sampling='queries'`` requests a native analytic observation, including
    height at intersections and slopes at the final hit. ``sampling='grid'``
    requests the complete periodic height/gradient raster and bilinear lookup.
    These sampling contracts differ; the selection and costs remain visible.
    """

    def __init__(
        self,
        source: Any,
        *,
        resolution: tuple[int, int] = (640, 1024),
        length: Any = (128.0, 128.0),
        sampling: str = "auto",
        method: str = "auto",
        backend: Any = "cpu",
        footprint: Any = None,
        query_error_budget: float = 1e-6,
        **display_options: Any,
    ) -> None:
        if sampling not in ("auto", "queries", "grid"):
            raise ValueError("ocean image sampling must be auto, queries, or grid")
        if not math.isfinite(query_error_budget) or query_error_budget < 0:
            raise ValueError("query_error_budget must be finite and nonnegative")
        self.display = PreparedOceanDisplay(resolution=resolution, length=length, **display_options)
        if not hasattr(source, "execute_fields"):
            from spf.renderer.ocean_terminal import prepare_ocean

            shape = source.shape
            spacing = tuple(distance / size for distance, size in zip(self.display.length, shape, strict=True))
            source = prepare_ocean(source, method=method, backend=backend, spacing=spacing)
        if len(source.shape) != 2:
            raise ValueError("an ocean image requires a two dimensional ocean")
        expected = tuple(distance / size for distance, size in zip(self.display.length, source.shape, strict=True))
        if hasattr(source, "spacing") and not np.allclose(source.spacing, expected, rtol=1e-12, atol=0):
            raise ValueError("prepared ocean spacing and display physical lengths disagree")
        source_layout = getattr(source, "layout", None)
        if source_layout is not None:
            if "grid_origin" in display_options and not np.allclose(
                self.display.grid_origin, source_layout.origin, rtol=0, atol=0
            ):
                raise ValueError("prepared ocean layout and display grid origins disagree")
            self.display._set_grid_origin(source_layout.origin)
        self.source = source
        self.footprint = footprint
        self.requested_sampling = sampling
        if callable(getattr(source, "bind_display_execution", None)):
            # PreparedOcean owns the request-to-executor decision.  The image
            # retains the sampling contract and modeled costs only.
            selected, bound, costs, fallback_reason = source.bind_display_execution(
                self.display.resolution,
                self.display.intersections,
                sampling=sampling,
                footprint=footprint,
                output_dtype=np.float32,
            )
            self.sampling = selected
            self.costs = dict(costs)
            self.fallback_reason = fallback_reason
            self.bound_plan = bound
        else:
            # Compatibility for lightweight field-like sources without a
            # PreparedOcean request binder.
            grid_samples = math.prod(source.shape)
            query_samples = math.prod(self.display.resolution) * (self.display.intersections + 1)
            plan = getattr(source, "plan", None)
            active = max(1, int(getattr(plan, "active_modes", getattr(plan, "mode_count", 1))))
            grid_cost = min(getattr(plan, "sparse_cost", grid_samples), getattr(plan, "dense_cost", grid_samples))
            query_cost = query_samples * active
            self.costs = {
                "grid_samples": grid_samples,
                "query_samples_upper_bound": query_samples,
                "active_modes": active,
                "grid_operation_estimate": grid_cost,
                "query_operation_estimate": query_cost,
                "model": "operation_count_heuristic_not_measured_runtime",
            }
            has_query = bool(getattr(source, "native_queries_supported", False))
            requested_method = getattr(plan, "diagnostics", {}).get("requested_method", method)
            requested_method = {
                "dense": "irfft",
                "fft": "irfft",
                "dense_irfft": "irfft",
                "packed": "fb3",
                "v02": "fb3",
            }.get(str(requested_method), str(requested_method))
            footprint_supported = getattr(source, "native_footprint_supported", lambda value: True)(footprint)
            reference_requested = str(getattr(plan, "backend", "cpu")) in {"reference", "python", "numpy"}
            query_candidate = (
                has_query
                and footprint_supported
                and not reference_requested
                and requested_method not in ("irfft", "fb3")
            )
            self.sampling = sampling
            self.fallback_reason = "native_query_footprint_unavailable" if not footprint_supported else None
            if sampling == "auto":
                self.sampling = "queries" if query_candidate and query_cost < grid_cost else "grid"
            elif sampling == "queries" and not query_candidate:
                raise ValueError("this ocean plan has no native compact point consumer")
            self.bound_plan = getattr(source, "plan", None)
        # Measure complete legal SFR routes at the explicit preparation
        # boundary. The extra absolute field error is visible and can be
        # disabled with query_error_budget=0; FP32 sources never enter it.
        from spf.renderer.ocean_adaptive import calibrate_ocean_display

        calibrated = calibrate_ocean_display(
            source, self.display, sampling=sampling, footprint=footprint,
            absolute_budget=query_error_budget,
        )
        self._adaptive_sampler = None
        if calibrated is not None:
            selected, self._adaptive_sampler, costs = calibrated
            selected, bound, _, fallback_reason = source.bind_display_execution(
                self.display.resolution, self.display.intersections,
                sampling=selected, footprint=footprint, output_dtype=np.float32,
            )
            self.sampling = selected
            self.costs = costs
            self.fallback_reason = fallback_reason
            self.bound_plan = bound
        self.bound_plan = _attach_display_costs(
            self.source,
            selected_sampling=self.sampling,
            costs=self.costs,
        )
        self._field_error_ledger = None
        self._field_error_sampler: Any = None
        self._field_error_plan: Any = None

    def advance(self, steps: int = 1) -> None:
        self.source.advance(steps)

    def field_sampler(self) -> Any:
        """Return the exact retained point consumer used by this image.

        Refresh the source timeline before selecting its display wrapper so
        callers can validate the same approximation and phase state as C.
        """
        if self.sampling != "queries":
            raise ValueError("a grid image has no continuous field sampler")
        base_sampler = self.source.native_sampler(footprint=self.footprint)
        sampler = base_sampler
        display_sampler = getattr(self.source, "display_sampler", None)
        if callable(display_sampler):
            candidate = display_sampler(footprint=self.footprint)
            if candidate is base_sampler or getattr(candidate, "source", None) is base_sampler:
                sampler = candidate
        adaptive_sampler = getattr(self, "_adaptive_sampler", None)
        if adaptive_sampler is not None and (
            adaptive_sampler is base_sampler or getattr(adaptive_sampler, "source", None) is base_sampler
        ):
            sampler = adaptive_sampler
        return sampler

    def execute(self, *, return_linear: bool = False) -> OceanImage:
        if self.sampling == "queries":
            sampler = self.field_sampler()
            result = self.display.execute(sampler=sampler, return_linear=return_linear)
        else:
            fields = self.source.execute_fields(output_dtype=np.float32, footprint=self.footprint)
            result = self.display.execute(fields, return_linear=return_linear)
        diagnostics = dict(result.diagnostics)
        if self.sampling == "grid":
            from spf.renderer._ocean_display_native import CONSUMER, DISPLAY_CONSUMER

            field_report = getattr(fields, "report", None)
            field_diagnostics = dict(getattr(field_report, "diagnostics", {}) or {})
            field_consumer = field_diagnostics.get("consumer")
            field_counters = dict(field_diagnostics.get("actual_counters", {}) or {})
            diagnostics["field_consumer"] = field_consumer
            diagnostics["fallback_conversions"] = int(field_counters.get("fallback_conversions", 0)) + (
                1 if int(diagnostics.get("array_conversion_elements", 0)) else 0
            )
            if field_consumer == CONSUMER:
                # Height and both slopes were synthesized on packed SPF32 words
                # in C and handed to the display as contiguous float32 planes:
                # no Cartesian coefficient plane and no array conversion.
                diagnostics["consumer"] = DISPLAY_CONSUMER
                diagnostics["display_kernel_identity"] = DISPLAY_CONSUMER
        if self.sampling == "queries" and getattr(sampler, "words", None) is not None:
            query_plan = getattr(sampler, "_query_block_plan", None)
            if self._field_error_sampler is not sampler or self._field_error_plan is not query_plan:
                ray_extent = max(1800.0, abs(float(self.display.camera.eye[2])) / 0.004)
                eye_yx = np.asarray(self.display.camera.eye[:2])[::-1]
                sample_extent = np.max(
                    (np.abs(eye_yx) + ray_extent + 2 * np.abs(sampler.origin)) / np.asarray(sampler.spacing)
                )
                self._field_error_ledger = sampler.error_ledger(coordinate_bound=float(sample_extent)).to_dict()
                self._field_error_sampler = sampler
                self._field_error_plan = query_plan
            diagnostics["field_error_ledger"] = self._field_error_ledger
            diagnostics["field_error_ledger_scope"] = "height_and_slopes_before_geometry_and_display"
        report = getattr(self.source, "report", None)
        if report is not None and self.sampling == "grid":
            diagnostics["ocean_report"] = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        elif self.sampling == "queries":
            # execute_fields was deliberately not invoked. Its previous
            # report is not evidence for this direct-query frame.
            plan = getattr(self.source, "plan", None)
            diagnostics["source_plan"] = plan.to_dict() if plan is not None and hasattr(plan, "to_dict") else None
            diagnostics["source_type"] = type(self.source).__name__
            sampler_details = getattr(sampler, "diagnostics", {})
            diagnostics["query_field_algebra"] = (
                "spf32_q30_preparation_and_sparse_residual_with_fp64_local_taylor"
                if sampler_details.get("representation") == "local_taylor_jet_with_sparse_residual"
                else sampler_details.get("native_precision", "cuda_spf32_native")
                if callable(getattr(sampler, "cuda_display", None))
                else
                "spf32_q30_signed"
                if getattr(sampler, "words", None) is not None
                else "fp32_cartesian"
                if getattr(sampler, "coeff_re", None) is not None
                else sampler_details.get("native_precision", "unspecified_native_callback")
            )
            diagnostics["query_lowering"] = "retained_compact_context_to_native_shared_height_and_gradients"
        diagnostics.update(
            requested_sampling=self.requested_sampling,
            selected_sampling=self.sampling,
            sampling_contract="continuous_points" if self.sampling == "queries" else "periodic_bilinear_grid",
            sampling_costs=self.costs,
            sampling_fallback_reason=self.fallback_reason,
            cached_output_pixels=0,
        )
        plan = getattr(self.source, "plan", None)
        if plan is not None:
            source_kernel = getattr(plan, "kernel", None)
            callback_identity = diagnostics.get("display_callback_identity")
            display_kernel_identity = diagnostics.get("display_kernel_identity")
            diagnostics.update(
                bound_request=(
                    getattr(plan.request, "to_dict", lambda: plan.request)()
                    if getattr(plan, "request", None) is not None
                    else None
                ),
                bound_executor=getattr(plan, "executor", None),
                source_selected_kernel=source_kernel,
                selected_kernel=(
                    display_kernel_identity
                    if display_kernel_identity
                    else callback_identity
                    if self.sampling == "queries" and callback_identity
                    else source_kernel
                ),
            )
        return OceanImage(result.pixels, result.linear, MappingProxyType(diagnostics))

    render = execute


def prepare_ocean_image(source: Any, **options: Any) -> PreparedOceanImage:
    return PreparedOceanImage(source, **options)


def render_ocean_image(source: Any, **options: Any) -> OceanImage:
    return prepare_ocean_image(source, **options).execute()


def compile_ocean_display(output: str | os.PathLike[str] | None = None) -> Path:
    root = Path(__file__).resolve().parents[3]
    suffix = "dylib" if sys.platform == "darwin" else "so"
    target = Path(output) if output is not None else root / "native" / f"libspf_ocean_display.{suffix}"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            os.environ.get("CC", "cc"),
            "-std=c11",
            "-shared",
            "-fPIC",
            "-O3",
            "-ffp-contract=off",
            str(root / "native/src/scalar/spf32/sfr_ocean_display.c"),
            "-o",
            str(target),
            "-lm",
        ],
        check=True,
    )
    _library.cache_clear()
    return target


if __name__ == "__main__":
    print(compile_ocean_display())
