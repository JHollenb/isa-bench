"""Optional CUDA consumer for compact packed-SPF ocean fields.

The module is intentionally separate from the reference and C CPU consumers.
Importing :mod:`spf.renderer` therefore remains dependency free on machines
without PyTorch or CUDA.  The extension is compiled lazily on the first
sampler construction.  Its only source lane is canonical ``uint32`` SPF32
words; a Cartesian/FP32 constructor is deliberately absent so a caller
cannot accidentally label a float control as an SPF CUDA execution.

The kernel evaluates one output sample per CUDA thread and walks the retained
addressed support.  It emits height, physical ``y`` gradient, and physical
``x`` gradient together.  Grid samples use the emitted u16 phase; point
queries compose the packed phase, wide phase, and physical coordinate before
one round-to-nearest-even u16 emission, matching the SFR ocean contract.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

_SOURCE = Path(__file__).with_name("_ocean_cuda_kernel.cu")
_EXTENSION_NAME = "spf_sfr_ocean_cuda_v1"
_BUILD_LOCK = threading.RLock()
_EXTENSION: Any | None = None
_BUILD_ERROR: BaseException | None = None
_POINT_CYCLE_LIMIT = float(2**32)
_TAU = 2.0 * math.pi


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except Exception as exc:  # pragma: no cover - depends on host extras
        raise ImportError("CUDA SFR requires the optional PyTorch CUDA runtime") from exc
    return torch


def availability() -> tuple[bool, str | None]:
    """Return whether a CUDA runtime can execute the optional SFR kernel.

    This probe does not compile or allocate a CUDA context.  Compilation is
    lazy and is part of sampler construction, where an actual compile failure
    remains visible to the caller instead of being converted to CPU fallback.
    """

    if str(os.environ.get("SPF_SFR_CUDA_DISABLE", "")).strip().lower() in {"1", "true", "yes"}:
        return False, "SPF_SFR_CUDA_DISABLE requested"
    if not _SOURCE.is_file():
        return False, f"CUDA kernel source is missing: {_SOURCE}"
    try:
        torch = _import_torch()
    except ImportError as exc:
        return False, str(exc)
    try:
        if not bool(torch.cuda.is_available()):
            return False, "PyTorch CUDA runtime is unavailable on this host"
    except Exception as exc:  # pragma: no cover - vendor runtime failure
        return False, f"CUDA runtime probe failed: {exc}"
    return True, None


def available() -> bool:
    """Boolean shorthand for :func:`availability`."""

    return availability()[0]


def runtime_info() -> dict[str, Any]:
    """Return host/runtime facts used in custody and backend diagnostics."""

    ok, reason = availability()
    info: dict[str, Any] = {
        "available": bool(ok),
        "reason": reason,
        "kernel_source": str(_SOURCE),
        "kernel_sha256": hashlib.sha256(_SOURCE.read_bytes()).hexdigest() if _SOURCE.is_file() else None,
        "query_header_sha256": {
            name: hashlib.sha256(_SOURCE.with_name(name).read_bytes()).hexdigest()
            for name in ("_ocean_cuda_jet.cuh", "_ocean_cuda_jet_checks.cuh")
            if _SOURCE.with_name(name).is_file()
        },
        "extension": _EXTENSION_NAME,
        "extension_loaded": _EXTENSION is not None,
    }
    if not ok:
        return info
    try:
        torch = _import_torch()
        index = int(torch.cuda.current_device())
        info.update(
            {
                "torch_version": str(torch.__version__),
                "cuda_version": str(getattr(torch.version, "cuda", None)),
                "device_index": index,
                "device_name": str(torch.cuda.get_device_name(index)),
                "device_capability": tuple(int(x) for x in torch.cuda.get_device_capability(index)),
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        info["runtime_info_error"] = str(exc)
    return info


def ensure_loaded() -> Any:
    """Build/load the CUDA extension and return its Python module.

    The extension lives in PyTorch's normal user cache.  No package import
    performs this work.  A failed build is retained for the process so every
    caller receives the same explicit error and no execution can be mislabeled
    as a native CUDA result.
    """

    global _EXTENSION, _BUILD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _BUILD_ERROR is not None:
        raise RuntimeError("the optional CUDA SFR extension failed to build") from _BUILD_ERROR
    ok, reason = availability()
    if not ok:
        raise RuntimeError(f"CUDA SFR backend is unavailable: {reason}")
    with _BUILD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _BUILD_ERROR is not None:
            raise RuntimeError("the optional CUDA SFR extension failed to build") from _BUILD_ERROR
        try:
            _import_torch()
            from torch.utils.cpp_extension import load  # type: ignore[import-not-found,unused-ignore]

            _EXTENSION = load(
                name=_EXTENSION_NAME,
                sources=[str(_SOURCE)],
                extra_include_paths=[str(_SOURCE.parent), str(_SOURCE.parents[3] / "native/src/scalar/spf32")],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False,
            )
        except BaseException as exc:  # keep the build failure explicit
            _BUILD_ERROR = exc
            raise RuntimeError(f"failed to compile the optional CUDA SFR extension: {exc}") from exc
    return _EXTENSION


def _pair(value: Any, *, name: str, positive: bool = False) -> tuple[float, float]:
    values: tuple[float, ...]
    if np.ndim(value) == 0:
        values = (float(value), float(value))
    else:
        values = tuple(float(x) for x in value)
        if len(values) != 2:
            raise ValueError(f"{name} must contain exactly two values")
    if any(not math.isfinite(x) for x in values):
        raise ValueError(f"{name} must be finite")
    if positive and any(x <= 0.0 for x in values):
        raise ValueError(f"{name} must be positive")
    return values[0], values[1]


def _words(value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError("CUDA SPF32 words must be a one-dimensional uint32 array")
    if raw.dtype.kind not in "iu":
        raise TypeError("CUDA SPF32 words must contain integer packed uint32 values")
    if raw.size and (np.any(raw < 0) or np.any(raw > np.iinfo(np.uint32).max)):
        raise ValueError("CUDA SPF32 words must fit the uint32 packed representation")
    result = np.array(raw, dtype=np.uint32, order="C", copy=True)
    result.setflags(write=False)
    return result


def _indices(value: Any, count: int, shape: tuple[int, int]) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (count, 2):
        raise ValueError(f"CUDA ocean indices must have shape ({count}, 2)")
    if raw.dtype.kind not in "iu":
        raise TypeError("CUDA ocean indices must contain integer values")
    if raw.dtype.kind == "u" and raw.size and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("CUDA ocean indices must fit signed int64")
    result = np.array(raw, dtype=np.int64, order="C", copy=True)
    # The native consumer accepts wrapped integer indices.  Extremely large
    # values are still valid int64 addresses and are canonicalized by CUDA.
    del shape
    result.setflags(write=False)
    return result


def _u64(value: Any, count: int, *, name: str) -> np.ndarray:
    if value is None:
        result = np.zeros(count, dtype=np.uint64)
    else:
        raw = np.asarray(value)
        if raw.shape != (count,):
            raise ValueError(f"CUDA ocean {name} must have shape ({count},)")
        if raw.dtype.kind not in "iu":
            raise TypeError(f"CUDA ocean {name} must contain integer uint64 values")
        if raw.dtype.kind == "i" and raw.size and np.any(raw < 0):
            raise ValueError(f"CUDA ocean {name} must be non-negative uint64 values")
        if raw.dtype.kind == "u" and raw.size and np.any(raw > np.iinfo(np.uint64).max):
            raise ValueError(f"CUDA ocean {name} must fit uint64")
        result = np.array(raw, dtype=np.uint64, order="C", copy=True)
    return result


def _shape(value: Any) -> tuple[int, int]:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.shape != (2,) or raw.dtype.kind not in "iu":
        raise TypeError("CUDA ocean shape must contain exactly two integer dimensions")
    if np.any(raw < 1):
        raise ValueError("CUDA ocean shape dimensions must be positive")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("CUDA ocean shape dimensions must fit signed int64")
    return int(raw[0]), int(raw[1])


def _validate_phase_bits(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or int(value) != value:
        raise TypeError("CUDA ocean phase_bits must be an integer")
    bits = int(value)
    if bits < 1 or bits > 64:
        raise ValueError("CUDA ocean phase_bits must be in [1, 64]")
    return bits


def _validate_coordinates(values: np.ndarray, spacing: tuple[float, float], *, name: str) -> None:
    if values.size and not np.all(np.isfinite(values)):
        raise ValueError(f"CUDA ocean {name} must contain finite physical coordinates")
    # The point consumer emits one u16 phase after composing coordinates.  A
    # bounded phase cycle keeps that composition exact enough for the declared
    # low-level contract and mirrors the C callback's long-double guard.
    for axis, denominator in enumerate(spacing):
        if values.size and np.any(np.abs(values[..., axis] / denominator) > _POINT_CYCLE_LIMIT):
            raise ValueError(f"CUDA ocean {name} exceeds the representable phase composition bound")


def _host_log_counts(words: np.ndarray) -> tuple[int, int, int]:
    logs = (words >> np.uint32(16)).astype(np.uint16).view(np.int16)
    zero = int(np.count_nonzero(logs == np.int16(-32768)))
    invalid = int(np.count_nonzero(logs == np.int16(32767)))
    active = int(words.size - zero - invalid)
    return active, zero, invalid


@dataclass(frozen=True, slots=True)
class CUDAOceanErrorLedger:
    """Absolute bounds for Q30 ROM factors and an FP64 signed reduction.

    Bounds are relative to the supplied packed coefficients, not the original
    unquantized spectrum. Source-transfer quantization is composed separately.
    No CPU fixed-point coefficient or int64-accumulator claim is inherited.
    """

    mode_count: int
    active_modes: int
    phase_emission_bound: float
    phase_lut_bound: float
    coordinate_composition_bound: float
    fp64_setup_relative_bound: float
    fp64_reduction_relative_bound: float
    final_float32_bound: float
    height_absolute_bound: float
    gradient_y_absolute_bound: float
    gradient_x_absolute_bound: float
    finite: bool
    arithmetic: str = "Q30 phase ROM, FP64 coefficients/products/signed sum, FP32 field output"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cuda_error_ledger(
    words: Any, indices: Any, shape: tuple[int, int], *,
    spacing: tuple[float, float] = (1.0, 1.0),
    origin: tuple[float, float] = (0.0, 0.0),
    footprint: tuple[float, float] = (0.0, 0.0),
    normalize: bool = True, coordinate_bound: float | None = None,
) -> CUDAOceanErrorLedger:
    """Host-only bound shared by the CUDA field and inline display consumer."""
    packed = _words(words)
    canonical = _indices(indices, len(packed), shape) % np.asarray(shape)
    signed = np.where(canonical <= np.asarray(shape) // 2, canonical, canonical - np.asarray(shape))
    logs = (packed >> np.uint32(16)).astype(np.uint16).view(np.int16)
    active = (logs != -32768) & (logs != 32767)
    frequency = _TAU * signed / (np.asarray(shape) * np.asarray(spacing))
    nyquist = np.zeros_like(canonical, dtype=bool)
    self_axes = canonical == 0
    for axis, size in enumerate(shape):
        if size % 2 == 0:
            nyquist[:, axis] = canonical[:, axis] == size // 2
            self_axes[:, axis] |= nyquist[:, axis]
    pair = np.where(np.all(self_axes, axis=1), 1.0, 2.0)
    attenuation_exponent = 0.5 * np.sum((frequency * np.asarray(footprint)) ** 2, axis=1)
    amplitude = np.where(active, np.exp2(logs.astype(float) / 256.0) * np.exp(-attenuation_exponent), 0.0)
    gains = np.sum((amplitude * pair)[:, None] * np.column_stack((np.ones(len(packed)), np.abs(frequency))), axis=0)
    if normalize:
        gains /= math.prod(shape)
    factors = int(np.max(1 + nyquist[active].sum(axis=1), initial=1))
    # A grid emits temporal and coordinate phases separately; queries emit
    # once per ordinary/Nyquist factor. Cover both consumers with one ledger.
    phase = max(2, factors) * math.pi / 65536.0
    lut = (1.0 + math.pi**2 / (8.0 * 1024.0**2) + 2.0**-30) ** factors - 1.0
    extent = (
        max(abs(origin[i]) / spacing[i] + shape[i] for i in range(2))
        if coordinate_bound is None else float(coordinate_bound)
    )
    if not math.isfinite(extent) or extent < 0:
        raise ValueError("coordinate_bound must be finite and non-negative")
    eps = float(np.finfo(np.float64).eps)
    coordinate = 16.0 * eps * _TAU * (extent + max(abs(origin[i]) / spacing[i] for i in range(2)))
    if extent > _POINT_CYCLE_LIMIT:
        coordinate = math.inf
    # Setup includes exp2/exp, physical frequencies, products, and normalization.
    # This conservative allowance assumes ordinary CUDA double libm accuracy;
    # it is not a proof of arbitrary vendor-library implementations.
    setup = 64.0 * eps * (1.0 + float(np.max(attenuation_exponent[active], initial=0.0)))
    rounding = (int(np.count_nonzero(active)) + 32) * eps
    reduction = rounding / (1.0 - rounding) if rounding < 1 else math.inf
    before_float = gains * (phase + lut + coordinate + setup + reduction)
    final = (gains + before_float) * 2.0**-24 + 2.0**-149
    bounds = before_float + final
    finite = bool(np.all(np.isfinite(bounds)) and not np.any(logs == 32767))
    return CUDAOceanErrorLedger(
        len(packed), int(np.count_nonzero(active)), phase, lut, coordinate,
        setup, reduction, float(np.max(final)),
        float(bounds[0]), float(bounds[1]), float(bounds[2]), finite,
    )


@dataclass(slots=True)
class CUDAOceanStats:
    """Per-call custody record for the CUDA ocean kernel."""

    samples: int = 0
    active_modes: int = 0
    zero_modes: int = 0
    phase_emissions: int = 0
    phase_updates: int = 0
    trig_evaluations: int = 0
    phase_lut_evaluations: int = 0
    visible_samples: int = 0
    skipped_samples: int = 0
    invalid_modes: int = 0
    nonfinite_outputs: int = 0
    term_evaluations: int = 0
    kernel_launches: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    source_bytes: int = 0
    output_bytes: int = 0
    term_preparations: int = 0
    preparation_exp2: int = 0
    gradient_outputs: int = 2
    phase_bits: int = 64
    arithmetic_boundary: str = "cuda_spf32_log_phase_to_float32"
    phase_quantization_bound: float = math.pi / 65536.0
    max_abs_height: float = 0.0
    max_abs_gradient: float = 0.0
    status: str = "valid"
    kernel_name: str = "spf_sfr_ocean_cuda_v1::sfr_ocean_{grid,points}"
    source_representation: str = "spf32_packed_uint32"
    derivative_operator: str = "spectral_quarter_turn_projection"
    spatial_phase: str = "cuda_shared_q30_phase_rom"
    device: str | None = None
    runtime: dict[str, Any] = field(default_factory=dict)
    error_ledger: dict[str, Any] | None = None
    display_pixels: int = 0
    display_water_pixels: int = 0
    display_height_queries: int = 0
    display_gradient_queries: int = 0
    display_lookups: int = 0
    region_stats: dict[str, int] = field(default_factory=dict)

    @property
    def phase_grid_steps(self) -> int:
        return 0

    @property
    def setup_exp2(self) -> int:
        return self.preparation_exp2

    @property
    def setup_trig(self) -> int:
        return 0

    @property
    def cartesian_source_decodes(self) -> int:
        return 0

    @property
    def flags(self) -> int:
        # Bit 4 is the explicit derivative quarter-turn lane in the C ABI;
        # the CUDA implementation performs the equivalent projection from
        # packed log/phase data in the fused kernel.
        return 8

    def to_dict(self) -> dict[str, Any]:
        values = {
            "samples": int(self.samples),
            "active_modes": int(self.active_modes),
            "source_terms": int(self.active_modes),
            "zero_modes": int(self.zero_modes),
            "phase_emissions": int(self.phase_emissions),
            "phase_updates": int(self.phase_updates),
            "trig_evaluations": int(self.trig_evaluations),
            "phase_lut_evaluations": int(self.phase_lut_evaluations),
            "phase_grid_steps": int(self.phase_grid_steps),
            "setup_exp2": int(self.setup_exp2),
            "setup_trig": int(self.setup_trig),
            "visible_samples": int(self.visible_samples),
            "skipped_samples": int(self.skipped_samples),
            "invalid_modes": int(self.invalid_modes),
            "nonfinite_outputs": int(self.nonfinite_outputs),
            "cartesian_source_decodes": int(self.cartesian_source_decodes),
            "term_preparations": int(self.term_preparations),
            "byte_accounting": "estimated logical operands, not measured DRAM traffic",
            "term_evaluations": int(self.term_evaluations),
            "kernel_launches": int(self.kernel_launches),
            "bytes_read": int(self.bytes_read),
            "bytes_written": int(self.bytes_written),
            "source_bytes": int(self.source_bytes),
            "output_bytes": int(self.output_bytes),
            "gradient_outputs": int(self.gradient_outputs),
            "phase_bits": int(self.phase_bits),
            "arithmetic_boundary": self.arithmetic_boundary,
            "phase_quantization_bound": float(self.phase_quantization_bound),
            "phase_lut_bound": float(math.pi**2 / (8.0 * 1024.0**2) + 2.0**-30),
            "max_abs_height": float(self.max_abs_height),
            "max_abs_gradient": float(self.max_abs_gradient),
            "status": self.status,
            "kernel_name": self.kernel_name,
            "source_representation": self.source_representation,
            "derivative_operator": self.derivative_operator,
            "spatial_phase": self.spatial_phase,
            "device": self.device,
            "runtime": dict(self.runtime),
            "display_pixels": int(self.display_pixels),
            "display_water_pixels": int(self.display_water_pixels),
            "display_height_queries": int(self.display_height_queries),
            "display_gradient_queries": int(self.display_gradient_queries),
            "display_lookups": int(self.display_lookups),
        }
        values["query_blocks"] = dict(self.region_stats)
        if self.error_ledger is not None:
            values["error_ledger"] = dict(self.error_ledger)
        return values


class CUDAOceanSampler:
    """Persistent compact SPF32 ocean sampler backed by a CUDA kernel."""

    def __init__(
        self,
        words: Any,
        indices: Any,
        *,
        shape: tuple[int, int] | list[int],
        phases: Any = None,
        increments: Any = None,
        phase_bits: int = 64,
        spacing: Any = (1.0, 1.0),
        origin: Any = (0.0, 0.0),
        normalize: bool = True,
        footprint: Any = (0.0, 0.0),
    ) -> None:
        self.shape = _shape(shape)
        self.spacing = _pair(spacing, name="spacing", positive=True)
        self.origin = _pair(origin, name="origin")
        self.footprint = _pair(footprint, name="footprint") if footprint is not None else (0.0, 0.0)
        if any(x < 0.0 for x in self.footprint):
            raise ValueError("CUDA ocean footprint sigmas must be non-negative")
        self.normalize = bool(normalize)
        self.phase_bits = _validate_phase_bits(phase_bits)
        self.words = _words(words)
        self.indices = _indices(indices, int(self.words.size), self.shape)
        self.phases = _u64(phases, int(self.words.size), name="phases")
        self.increments = _u64(increments, int(self.words.size), name="increments")
        self.increments.setflags(write=False)
        self._active_modes, self._zero_modes, self._invalid_modes = _host_log_counts(self.words)
        if self._invalid_modes:
            raise ValueError("CUDA SPF32 source contains reserved invalid log codes")
        ok, reason = availability()
        if not ok:
            raise RuntimeError(f"CUDA SFR backend is unavailable: {reason}")
        torch = _import_torch()
        self._torch = torch
        self._timeline_owner = "uint64_modular_cuda"
        self.device = torch.device("cuda", torch.cuda.current_device())
        self._extension = ensure_loaded()
        # ``torch.tensor`` establishes an owned device copy.  ``as_tensor``
        # may alias a read-only NumPy view and emits a warning on some torch
        # versions, which would make input custody and stream ordering unclear.
        self._words_device = torch.tensor(self.words, dtype=torch.uint32, device=self.device)
        self._indices_device = torch.tensor(self.indices, dtype=torch.int64, device=self.device)
        self._phases_device = torch.tensor(self.phases, dtype=torch.uint64, device=self.device)
        self._increments_device = torch.tensor(self.increments, dtype=torch.uint64, device=self.device)
        prepared_terms = self._extension.prepare_terms(
            self._words_device,
            self._indices_device,
            int(self.shape[0]),
            int(self.shape[1]),
            float(self.spacing[0]),
            float(self.spacing[1]),
            float(self.footprint[0]),
            float(self.footprint[1]),
        )
        self._term_f64, self._term_i64, self._term_flags, prepare_status = prepared_terms
        self._check_status(prepare_status)
        self._phase_codes_device, self._phase_temporal_device = self._extension.refresh_phase_terms(
            self._term_f64, self._phases_device, int(self.phase_bits)
        )
        self._query_block_plan: Any = None
        self._query_block_descriptor: dict[str, Any] | None = None
        self._query_approximate = torch.empty(0, dtype=torch.int64, device=self.device)
        self._query_residual = torch.empty(0, dtype=torch.int64, device=self.device)
        self._query_keys = torch.empty(0, dtype=torch.int64, device=self.device)
        self._query_coefficients = torch.empty((0, 18), dtype=torch.float64, device=self.device)
        self._ready_event = torch.cuda.Event()
        self._record_ready()
        self._stats = CUDAOceanStats(
            active_modes=self._active_modes,
            zero_modes=self._zero_modes,
            term_preparations=self.mode_count,
            preparation_exp2=self.mode_count,
            phase_emissions=self.mode_count,
            phase_bits=self.phase_bits,
            source_bytes=int(self.words.nbytes + self.indices.nbytes + self.phases.nbytes),
            device=str(self.device),
            runtime=runtime_info(),
        )

    @classmethod
    def from_spf32(cls, words: Any, indices: Any, **kwargs: Any) -> CUDAOceanSampler:
        return cls(words, indices, **kwargs)

    @property
    def mode_count(self) -> int:
        return int(self.words.size)

    @property
    def stats(self) -> CUDAOceanStats:
        return self._stats

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "backend": "cuda",
            "execution_class": "cuda_ocean_fused",
            "native_precision": "spf32_packed_coefficients_q30_phase_rom_fp64_signed_reduce",
            "source_representation": "spf32_packed_uint32",
            "kernel": _EXTENSION_NAME,
            "device": str(self.device),
            "mode_count": self.mode_count,
            "shape": self.shape,
            "phase_bits": self.phase_bits,
            "phase_update_lane": getattr(self, "_timeline_owner", "uint64_modular_cuda"),
            "phase_code_storage": "resident_uint16",
            "phase_temporal_storage": "resident_float64_for_fractional_queries",
            "phase_lut": "native_shared_q30_quarter_rom_1025_interpolated",
            "gradient_lane": "spectral_quarter_turn_projection",
        }

    def mode_state(self) -> dict[str, Any]:
        """Expose retained packed source and wide phase state for custody."""

        return {
            "words": self.words,
            "indices": self.indices,
            "phases": self.phases,
            "increments": self.increments,
            "phase_bits": self.phase_bits,
        }

    def phase_codes(self) -> np.ndarray:
        self._wait_ready()
        return np.asarray(self._phase_codes_device.detach().cpu().numpy(), dtype=np.uint16)

    def configure_query_blocks(self, absolute_budget: float = 1e-6) -> dict[str, Any] | None:
        from spf.renderer._ocean_cuda_blocks import plan_query_blocks

        plan = plan_query_blocks(self, absolute_budget)
        self._wait_ready()
        self._query_block_plan = plan
        self._query_block_descriptor = None if plan is None else plan.to_dict()
        torch = self._torch
        if plan is None:
            self._query_approximate = torch.empty(0, dtype=torch.int64, device=self.device)
            self._query_residual = torch.empty(0, dtype=torch.int64, device=self.device)
            self._query_keys = torch.empty(0, dtype=torch.int64, device=self.device)
            self._query_coefficients = torch.empty((0, 18), dtype=torch.float64, device=self.device)
        else:
            self._query_approximate = torch.tensor(plan.approximate, dtype=torch.int64, device=self.device)
            self._query_residual = torch.tensor(plan.residual, dtype=torch.int64, device=self.device)
            self._query_keys = torch.full((plan.capacity,), -1, dtype=torch.int64, device=self.device)
            self._query_coefficients = torch.empty((plan.capacity, 18), dtype=torch.float64, device=self.device)
        self._record_ready()
        return self._query_block_descriptor

    def _record_ready(self) -> None:
        self._ready_event.record(self._torch.cuda.current_stream(self.device))

    def _wait_ready(self) -> None:
        self._torch.cuda.current_stream(self.device).wait_event(self._ready_event)

    def invalidate_query_blocks(self) -> None:
        self._wait_ready()
        if self._query_block_plan is not None:
            self._query_keys.fill_(-1)
            self._record_ready()

    def refresh_phase(self) -> None:
        self.invalidate_query_blocks()
        self._phases_device.copy_(self._torch.as_tensor(self.phases, dtype=self._torch.uint64, device=self.device))
        self._phase_codes_device, self._phase_temporal_device = self._extension.refresh_phase_terms(
            self._term_f64, self._phases_device, int(self.phase_bits)
        )
        self._record_ready()
        self._stats.phase_updates += self.mode_count

    def refresh(self) -> None:
        self.refresh_phase()

    def advance(self, steps: int = 1) -> None:
        if isinstance(steps, (bool, np.bool_)) or int(steps) != steps:
            raise TypeError("CUDA ocean phase steps must be integral")
        steps = int(steps)
        if steps == 0 or self.mode_count == 0:
            return
        self._wait_ready()
        self._extension.advance_phase(
            self._phases_device,
            self._increments_device,
            int(steps),
            int(self.phase_bits),
        )
        self._phase_codes_device, self._phase_temporal_device = self._extension.refresh_phase_terms(
            self._term_f64, self._phases_device, int(self.phase_bits)
        )
        modulus = 1 << self.phase_bits if self.phase_bits < 64 else 1 << 64
        self.phases = np.asarray(
            [
                (int(phase) + int(increment) * steps) % modulus
                for phase, increment in zip(self.phases, self.increments, strict=True)
            ],
            dtype=np.uint64,
        )
        self.invalidate_query_blocks()
        self._record_ready()
        self._stats.phase_updates += abs(steps) * self.mode_count

    def _validate_points(self, points: Any) -> np.ndarray:
        result = np.asarray(points, dtype=np.float64)
        if result.ndim != 2 or result.shape[1] != 2:
            raise ValueError("CUDA ocean points must have shape (count, 2) in physical (y, x) order")
        if not result.flags.c_contiguous:
            result = np.ascontiguousarray(result)
        _validate_coordinates(result, self.spacing, name="points")
        return result

    def _stats_for_call(self, output: Any, *, count: int, visible: int, skipped: int, point: bool) -> CUDAOceanStats:
        array = np.asarray(
            output.detach().cpu().numpy() if hasattr(output, "detach") else output,
            dtype=np.float32,
        )
        finite = np.isfinite(array)
        if not bool(np.all(finite)):
            raise FloatingPointError("CUDA ocean produced nonfinite field output")
        canonical = self.indices % np.asarray(self.shape)
        active = (self.words >> np.uint32(16)) != 32768
        nyquist = np.zeros_like(canonical, dtype=bool)
        for axis, size in enumerate(self.shape):
            if size % 2 == 0:
                nyquist[:, axis] = canonical[:, axis] == size // 2
        factors = 1 + nyquist.sum(axis=1)
        emissions = nyquist.sum(axis=1) + (~np.all(nyquist, axis=1))
        self._stats = CUDAOceanStats(
            samples=int(count),
            active_modes=self._active_modes,
            zero_modes=self._zero_modes,
            phase_emissions=int(np.sum(emissions[active]) * visible if point else self._active_modes * visible),
            trig_evaluations=0,
            phase_lut_evaluations=int(np.sum(factors[active]) * visible if point else self._active_modes * visible),
            visible_samples=int(visible),
            skipped_samples=int(skipped),
            invalid_modes=self._invalid_modes,
            nonfinite_outputs=int(np.count_nonzero(~finite)),
            term_evaluations=int(self._active_modes * visible),
            kernel_launches=1,
            bytes_read=int((self.mode_count + self._active_modes * (58 + (8 if point else 0))) * visible),
            bytes_written=int(count * 3 * 4),
            source_bytes=int(self.words.nbytes + self.indices.nbytes + self.phases.nbytes),
            output_bytes=int(count * 3 * 4),
            phase_bits=self.phase_bits,
            max_abs_height=float(
                np.max(np.abs(array[:, 0]))
                if point and count
                else np.max(np.abs(array[0]))
                if count
                else 0.0
            ),
            max_abs_gradient=float(
                np.max(np.abs(array[:, 1:])) if point and count else np.max(np.abs(array[1:])) if count else 0.0
            ),
            device=str(self.device),
            runtime=runtime_info(),
        )
        return self._stats

    def _check_status(self, status: Any) -> None:
        code = int(status.detach().cpu().item())
        if code == -2:
            raise ValueError("CUDA ocean query exceeds the representable phase composition bound")
        if code == -4:
            raise ValueError("CUDA ocean source violates the Hermitian/self-conjugate phase contract")
        if code != 0:
            raise RuntimeError(f"CUDA ocean kernel rejected execution with status {code}")

    def grid(
        self, *, return_stats: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[tuple[np.ndarray, np.ndarray, np.ndarray], CUDAOceanStats]:
        self._wait_ready()
        _validate_coordinates(np.asarray([self.origin], dtype=np.float64), self.spacing, name="origin")
        output, status = self._extension.grid(
            self._term_f64,
            self._term_i64,
            self._term_flags,
            self._phase_codes_device,
            int(self.shape[0]),
            int(self.shape[1]),
            float(self.spacing[0]),
            float(self.spacing[1]),
            float(self.origin[0]),
            float(self.origin[1]),
            int(self.normalize),
        )
        self._check_status(status)
        result = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
        stats = self._stats_for_call(
            result,
            count=int(math.prod(self.shape)),
            visible=int(math.prod(self.shape)),
            skipped=0,
            point=False,
        )
        fields = (result[0].reshape(self.shape), result[1].reshape(self.shape), result[2].reshape(self.shape))
        return (fields, stats) if return_stats else fields

    def points(
        self,
        points: Any,
        *,
        visible: Any = None,
        return_stats: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, CUDAOceanStats]:
        self._wait_ready()
        array = self._validate_points(points)
        count = int(array.shape[0])
        if visible is None:
            visible_array = np.ones(count, dtype=np.uint8)
        else:
            visible_array = np.asarray(visible, dtype=np.uint8).reshape(-1)
            if visible_array.shape != (count,):
                raise ValueError(f"CUDA ocean visible mask must have shape ({count},)")
            visible_array = np.ascontiguousarray(visible_array)
        point_tensor = self._torch.as_tensor(array, dtype=self._torch.float64, device=self.device)
        visible_tensor = self._torch.as_tensor(visible_array, dtype=self._torch.uint8, device=self.device)
        output, status, jet_stats = self._extension.points(
            self._term_f64,
            self._term_i64,
            self._term_flags,
            self._phase_codes_device,
            self._phase_temporal_device,
            point_tensor,
            visible_tensor,
            self._query_approximate, self._query_residual, self._query_keys, self._query_coefficients,
            0 if self._query_block_plan is None else self._query_block_plan.tiles_per_axis,
            int(self.shape[0]),
            int(self.shape[1]),
            float(self.spacing[0]),
            float(self.spacing[1]),
            float(self.origin[0]),
            float(self.origin[1]),
            int(self.normalize),
        )
        self._check_status(status)
        result = np.asarray(output.detach().cpu().numpy(), dtype=np.float32)
        visible_count = int(np.count_nonzero(visible_array))
        stats = self._stats_for_call(
            result,
            count=count,
            visible=visible_count,
            skipped=int(np.count_nonzero(visible_array == 0)),
            point=True,
        )
        block_stats = np.asarray(jet_stats.detach().cpu().numpy(), dtype=np.int64)
        stats.region_stats = dict(zip(
            ("jet_queries", "jet_builds", "jet_term_preparations", "jet_sparse_fallback_queries"),
            (int(v) for v in block_stats[:4]), strict=True,
        ))
        stats.phase_lut_evaluations = int(block_stats[4])
        stats.phase_emissions = int(block_stats[5])
        stats.term_evaluations = (visible_count-int(block_stats[0]))*self._active_modes + int(block_stats[2])
        if self._query_block_plan is not None:
            stats.term_evaluations += int(block_stats[0])*len(self._query_block_plan.residual)
        return (result, stats) if return_stats else result

    def sample(self, y: float, x: float, *, need_gradients: bool = True) -> np.ndarray:
        result = self.points(np.asarray([[float(y), float(x)]], dtype=np.float64))[0]
        if not need_gradients:
            result = result.copy()
            result[1:] = 0.0
        return result

    def cuda_display(self, display: Any, *, return_linear: bool = False) -> Any:
        """Render a final RGB image entirely on CUDA from the retained support.

        The image kernel performs camera rays, iterative ocean intersections,
        normal/reflection geometry, positive radiance, and the supplied log
        display LUT in one device pass.  Only RGB (and optional requested
        linear RGB) crosses back to the host; no height/gradient raster is
        materialized.
        """

        from spf.renderer import _ocean_cuda_display
        source = getattr(_ocean_cuda_display, "_SOURCE", None)
        if source is None or not source.is_file():
            raise RuntimeError("CUDA ocean display source is unavailable; generic display is not advertised")
        return _ocean_cuda_display.execute_cuda_display(self, display, return_linear=return_linear)

    def error_ledger(self, *, coordinate_bound: float | None = None) -> Any:
        base = cuda_error_ledger(
            self.words, self.indices, self.shape, spacing=self.spacing,
            origin=self.origin, normalize=self.normalize, footprint=self.footprint,
            coordinate_bound=coordinate_bound,
        )
        if self._query_block_plan is not None:
            bounds = self._query_block_plan.output_bounds
            base = replace(
                base, height_absolute_bound=base.height_absolute_bound+bounds[0],
                gradient_y_absolute_bound=base.gradient_y_absolute_bound+bounds[1],
                gradient_x_absolute_bound=base.gradient_x_absolute_bound+bounds[2],
            )
        extra = getattr(self, "_source_transfer_ledger", None)
        if extra is None:
            return base
        return replace(
            base,
            height_absolute_bound=base.height_absolute_bound + float(extra.height_absolute_bound),
            gradient_y_absolute_bound=base.gradient_y_absolute_bound + float(extra.gradient_y_absolute_bound),
            gradient_x_absolute_bound=base.gradient_x_absolute_bound + float(extra.gradient_x_absolute_bound),
            finite=bool(base.finite and math.isfinite(float(extra.output_error_bound))),
        )

    def close(self) -> None:
        self._words_device = None
        self._indices_device = None
        self._phases_device = None
        self._phase_codes_device = None
        self._phase_temporal_device = None
        self._increments_device = None
        self._term_f64 = None
        self._term_i64 = None
        self._term_flags = None
        self._query_keys = None
        self._query_coefficients = None
        self._query_approximate = None
        self._query_residual = None

    def __enter__(self) -> CUDAOceanSampler:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "CUDAOceanSampler",
    "CUDAOceanStats",
    "availability",
    "available",
    "ensure_loaded",
    "runtime_info",
]
