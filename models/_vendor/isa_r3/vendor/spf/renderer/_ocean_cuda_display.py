"""Device resident SFR ocean display consumer.

The regular CUDA sampler owns the packed source and its prepared term SoA.
This module owns the display boundary so the sampler's field API does not have
to materialize a height/gradient raster.  The companion extension evaluates a
camera ray, the iterative water hit, gradients, and positive Q24 radiance in
one CUDA launch.  Only the final RGB tensor (and an explicitly requested
linear tensor) is copied to the host.

The module is optional.  Importing it does not import PyTorch or create a CUDA
context; compilation is lazy and a failed compile remains an explicit error.
It consumes the prepared-term contract exposed by ``CUDAOceanSampler``:

* ``_term_f64[:, 0:5]`` = packed phase, magnitude, raw dy, raw dx, footprint;
* ``_term_i64[:, 0:2]`` = signed ky/kx; and
* ``_term_flags`` = active, Nyquist-y, Nyquist-x, and self-conjugate bits.

The phase state remains the sampler's resident uint64 timeline plus its
emitted uint16 and float64 temporal phase tensors.  The ROMs and
the 26 fixed material codes are tiny display inputs; no ocean field buffer is
created or transferred.
"""

from __future__ import annotations

import hashlib
import math
import os
import threading
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

_SOURCE = Path(__file__).with_name("_ocean_cuda_display_kernel.cu")
_EXTENSION_NAME = "spf_sfr_ocean_cuda_display_v1"
_BUILD_LOCK = threading.RLock()
_EXTENSION: Any | None = None
_BUILD_ERROR: BaseException | None = None
_QS = 1 << 24
_LZERO = -(1 << 63)

_STAT_NAMES = (
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
    "term_evaluations",
    "phase_emissions",
    "phase_lut_evaluations",
    "nonfinite_outputs",
    "self_conjugate_rejections",
    "kernel_launches",
    "bytes_read",
    "bytes_written",
    "jet_queries", "jet_builds", "jet_term_preparations", "jet_sparse_fallback_queries",
)


def _import_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found,unused-ignore]
    except Exception as exc:  # pragma: no cover - optional dependency
        raise ImportError("CUDA SFR display requires the optional PyTorch CUDA runtime") from exc
    return torch


def availability() -> tuple[bool, str | None]:
    """Return whether the optional device display can execute on this host."""

    if str(os.environ.get("SPF_SFR_CUDA_DISABLE", "")).strip().lower() in {"1", "true", "yes"}:
        return False, "SPF_SFR_CUDA_DISABLE requested"
    if not _SOURCE.is_file():
        return False, f"CUDA display kernel source is missing: {_SOURCE}"
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
    return availability()[0]


def runtime_info() -> dict[str, Any]:
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
            }
        )
    except Exception as exc:  # pragma: no cover - diagnostics only
        info["runtime_info_error"] = str(exc)
    return info


def ensure_loaded() -> Any:
    """Build/load the display extension without changing the sampler ABI."""

    global _EXTENSION, _BUILD_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _BUILD_ERROR is not None:
        raise RuntimeError("the optional CUDA SFR display extension failed to build") from _BUILD_ERROR
    ok, reason = availability()
    if not ok:
        raise RuntimeError(f"CUDA SFR display is unavailable: {reason}")
    with _BUILD_LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        if _BUILD_ERROR is not None:
            raise RuntimeError("the optional CUDA SFR display extension failed to build") from _BUILD_ERROR
        try:
            from torch.utils.cpp_extension import load  # type: ignore[import-not-found,unused-ignore]

            _EXTENSION = load(
                name=_EXTENSION_NAME,
                sources=[str(_SOURCE)],
                extra_include_paths=[str(_SOURCE.parent), str(_SOURCE.parents[3] / "native/src/scalar/spf32")],
                extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
                verbose=False,
            )
        except BaseException as exc:  # keep compilation failure visible
            _BUILD_ERROR = exc
            raise RuntimeError(f"failed to compile the optional CUDA SFR display extension: {exc}") from exc
    return _EXTENSION


def _material_codes() -> tuple[np.ndarray, np.ndarray]:
    """Return fixed Q24 material codes and their seven positive basis rows.

    Values are evaluated from the same float32 literals used by the native
    display.  The basis is sent as float64 because the device reducer's
    positive matrix boundary is intentionally FP64, matching the SFR log
    consumer's nonnegative accumulation contract.
    """

    # Row order is deep, horizon, zenith, warm, bright, specular, haze.
    values = np.asarray(
        [
            (0.006, 0.080, 0.099),
            (0.77, 0.85, 0.91),
            (0.15, 0.39, 0.65),
            (1.0, 0.8, 0.5),
            (1.0, 0.93, 0.73),
            (1.0, 0.89, 0.68),
            (0.64, 0.77, 0.85),
        ],
        dtype=np.float32,
    )
    codes = np.rint(np.log2(values.astype(np.float64)) * _QS).astype(np.int64)
    scales = np.asarray(
        [0.75, 5.0, 0.0204, 0.9796, 0.8],
        dtype=np.float32,
    )
    scale_codes = np.rint(np.log2(scales.astype(np.float64)) * _QS).astype(np.int64)
    fixed = np.concatenate((codes.reshape(-1), scale_codes))
    basis = np.exp2(codes.astype(np.float64) / _QS)
    fixed.setflags(write=False)
    basis.setflags(write=False)
    return fixed, basis


def _device_tensor(torch: Any, value: Any, *, dtype: Any, device: Any) -> Any:
    return torch.tensor(value, dtype=dtype, device=device).contiguous()


def _cached_display_inputs(torch: Any, display: Any, device: Any) -> tuple[tuple[Any, ...], bool, int]:
    """Upload immutable display inputs once per display/device pair.

    The ocean terms and phases belong to the sampler and remain resident
    there.  Camera parameters, ROMs, LUT, and the fixed positive basis belong
    to the prepared display, so retaining them on that object avoids a host to
    device upload on every frame while keeping the cache scoped to the exact
    display preparation.
    """

    cache = getattr(display, "_cuda_display_inputs", None)
    if cache is None:
        cache = {}
        display._cuda_display_inputs = cache
    key = (str(device), tuple(int(x) for x in display.resolution), float(display.exposure))
    cached = cache.get(key)
    if cached is not None:
        return cached, True, 0
    camera = _device_tensor(torch, np.asarray(display.parameters, dtype=np.float32), dtype=torch.float32, device=device)
    eye = _device_tensor(torch, np.asarray(display.camera.eye, dtype=np.float32), dtype=torch.float32, device=device)
    logadd = _device_tensor(torch, np.asarray(display.logadd_rom, dtype=np.int32), dtype=torch.int32, device=device)
    logsub = _device_tensor(torch, np.asarray(display.logsub_rom, dtype=np.int32), dtype=torch.int32, device=device)
    lut = _device_tensor(torch, np.asarray(display.display_lut, dtype=np.uint8), dtype=torch.uint8, device=device)
    fixed_codes, basis = _material_codes()
    fixed = _device_tensor(torch, fixed_codes, dtype=torch.int64, device=device)
    # The kernel consumes one flat row-major [7, 3] matrix so its pointer
    # arithmetic stays identical for sky's four-row and water's seven-row
    # reductions.  Keep the host source two-dimensional for readability.
    basis_device = _device_tensor(torch, basis.reshape(-1), dtype=torch.float64, device=device)
    values = (camera, eye, logadd, logsub, lut, fixed, basis_device)
    cache[key] = values
    uploaded = sum(int(value.numel() * value.element_size()) for value in values)
    return values, False, uploaded


def _check_terms(sampler: Any) -> tuple[Any, Any, Any, Any, Any]:
    terms = getattr(sampler, "_term_f64", None)
    frequencies = getattr(sampler, "_term_i64", None)
    flags = getattr(sampler, "_term_flags", None)
    phase_codes = getattr(sampler, "_phase_codes_device", None)
    phase_temporal = getattr(sampler, "_phase_temporal_device", None)
    if terms is None or frequencies is None or flags is None or phase_codes is None or phase_temporal is None:
        raise RuntimeError("CUDA sampler does not expose resident prepared SFR terms")
    if terms.dim() != 2 or terms.size(1) != 5:
        raise RuntimeError("CUDA sampler prepared terms are incompatible with the display consumer")
    if frequencies.dim() != 2 or frequencies.size(0) != terms.size(0) or frequencies.size(1) != 2:
        raise RuntimeError("CUDA sampler prepared frequencies are incompatible with the display consumer")
    if flags.dim() != 1 or flags.size(0) != terms.size(0):
        raise RuntimeError("CUDA sampler prepared flags are incompatible with the display consumer")
    if phase_codes.dim() != 1 or phase_codes.size(0) != terms.size(0):
        raise RuntimeError("CUDA sampler resident u16 phase ROM is incompatible with the display consumer")
    if phase_temporal.dim() != 1 or phase_temporal.size(0) != terms.size(0):
        raise RuntimeError("CUDA sampler resident temporal phase ROM is incompatible with the display consumer")
    return terms, frequencies, flags, phase_codes, phase_temporal


def _check_status(status: Any) -> None:
    """Raise a stable Python error for a device display status code."""

    code = int(status.detach().cpu().reshape(-1)[0].item())
    if code == 0:
        return
    if code == -2:
        raise ValueError("CUDA SFR display query exceeds the representable phase composition bound")
    if code == -4:
        raise ValueError("CUDA SFR display source violates the Hermitian/self-conjugate phase contract")
    raise RuntimeError(f"CUDA SFR display kernel rejected execution with status {code}")


def execute_cuda_display(sampler: Any, display: Any, *, return_linear: bool = False) -> Any:
    """Render ``display`` from a resident CUDA SFR sampler.

    The return type is the renderer's ``OceanImage``.  Diagnostics include
    device counters from the same launch, with callback counters explicitly
    zero because no Python or C callback is reachable from this kernel.
    """

    if str(getattr(display, "profile", "log")) != "log":
        raise ValueError("device SFR display supports the positive log profile only")
    sampler._wait_ready()
    terms, frequencies, flags, phase_codes, phase_temporal = _check_terms(sampler)
    torch = getattr(sampler, "_torch", None) or _import_torch()
    extension = ensure_loaded()
    device = terms.device
    if not bool(getattr(device, "type", None) == "cuda"):
        raise ValueError("resident SFR display terms must be CUDA tensors")

    (camera, eye, logadd, logsub, lut, fixed, basis_device), cache_hit, uploaded_bytes = _cached_display_inputs(
        torch, display, device
    )
    pixels, linear, stats_tensor, status = extension.display(
        terms,
        frequencies,
        flags,
        phase_codes,
        phase_temporal,
        camera,
        eye,
        logadd,
        logsub,
        lut,
        fixed,
        basis_device,
        sampler._query_approximate, sampler._query_residual,
        sampler._query_keys, sampler._query_coefficients,
        0 if sampler._query_block_plan is None else sampler._query_block_plan.tiles_per_axis,
        int(sampler.shape[0]),
        int(sampler.shape[1]),
        float(sampler.spacing[0]),
        float(sampler.spacing[1]),
        float(sampler.origin[0]),
        float(sampler.origin[1]),
        int(bool(getattr(sampler, "normalize", True))),
        int(display.resolution[1]),
        int(display.resolution[0]),
        int(display.intersections),
        int(bool(return_linear)),
    )
    _check_status(status)
    stats_values = np.asarray(stats_tensor.detach().cpu().numpy(), dtype=np.int64).reshape(-1)
    counters = {name: int(stats_values[index]) for index, name in enumerate(_STAT_NAMES)}
    counters["trig_evaluations"] = 0
    scalar_conversion_elements = sum(
        counters[name]
        for name in (
            "scalar_log_conversions",
            "scalar_exp2_conversions",
            "scalar_callback_input_conversions",
            "scalar_callback_output_conversions",
        )
    )
    pixel_array = np.asarray(pixels.detach().cpu().numpy(), dtype=np.uint8)
    linear_array = None if not return_linear else np.asarray(linear.detach().cpu().numpy(), dtype=np.float32)
    from spf.renderer.ocean_display import OceanImage

    mode_count = int(terms.size(0))
    display_kernel_identity = _EXTENSION_NAME + "::ocean_display"
    diagnostics = {
        "backend": "cuda",
        "execution_class": "cuda_sfr_native_log_display",
        "display_profile": "log",
        "display_boundary": "q24_positive_log_to_supplied_u8_lut",
        "field_source": "resident_prepared_spf32_terms",
        "source_representation": "spf32_packed_uint32",
        "materialized_field_elements": 0,
        "materialized_ocean_elements": 0,
        "materialized_display_elements": int(pixel_array.size + (0 if linear_array is None else linear_array.size)),
        "conversion_elements": 0,
        "array_conversion_elements": 0,
        "scalar_conversion_elements": int(scalar_conversion_elements),
        "scalar_conversion_counters_available": True,
        "output_pixels": int(pixel_array.shape[0] * pixel_array.shape[1]),
        "source_terms": mode_count,
        "intersections": int(display.intersections),
        "kernel": display_kernel_identity,
        "display_kernel_identity": display_kernel_identity,
        "kernel_source": str(_SOURCE),
        "runtime": runtime_info(),
        "phase_state": "resident_uint16_phase_codes_and_float64_temporal",
        "phase_update_lane": getattr(sampler, "_timeline_owner", "uint64_modular_cuda"),
        "phase_composition": "one_round_to_even_u16_per_sparse_term_or_jet_header_term",
        "phase_rom": "native_sfr_ocean_q30_quarter_1025",
        "phase_rom_absolute_error_bound": math.pi**2 / (8.0 * 1024.0**2) + 2.0**-30,
        "phase_lut_bound": math.pi**2 / (8.0 * 1024.0**2) + 2.0**-30,
        "phase_quantization_bound": math.pi / 65536.0,
        "phase_error_scope": "phase sine/cosine only; field/image error also depends on resident term accumulation",
        "radiance_reducer": "shared_positive_basis_matrix",
        "positive_reduction_basis": "seven_shared_terms_to_three_rgb_channels",
        "positive_reducer_finalizations": 3 * int(counters["pixels"]),
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
        },
        "positive_reduction_accumulator": "fp64_nonnegative",
        "positive_matrix_relative_error_bound": 1e-6,
        "positive_matrix_absolute_pruning_bound": 7 * 2.0**-128,
        "display_callback_identity": None,
        "display_callback_address": None,
        "callback_identity": None,
        "callback_status": "none_device_inline",
        "callback_invocations": 0,
        "scalar_callback_input_conversions": 0,
        "scalar_callback_output_conversions": 0,
        "display_callback": {
            "name": None,
            "address": None,
            "status": "none_device_inline",
        },
        "stats": counters,
        "query_blocks": sampler._query_block_descriptor,
        "counter_scope": "this CUDA display launch; preparation counters are separate sampler state",
        "byte_accounting": "logical operand bytes touched by the fused display loop",
        "display_input_cache": {
            "scope": "prepared_display_and_cuda_device",
            "hit": bool(cache_hit),
            "uploaded_bytes": int(uploaded_bytes),
            "inputs": "camera, eye, logadd ROM, logsub ROM, display LUT, material codes, positive basis",
        },
        "display_input_preparation": {
            "charged_this_launch": not bool(cache_hit),
            "scalar_log_conversions": 26 if not cache_hit else 0,
            "scalar_exp2_conversions": 21 if not cache_hit else 0,
            "scope": "host material-code preparation and positive-basis upload; kernel counters exclude cached setup",
        },
    }
    # Keep the same flat counter vocabulary as the CPU display boundary while
    # retaining the grouped copy above for consumers that treat counters as a
    # record.  These values are from this kernel launch only.
    diagnostics.update(counters)
    return OceanImage(pixel_array, linear_array, MappingProxyType(diagnostics))


__all__ = ["availability", "available", "ensure_loaded", "execute_cuda_display", "runtime_info"]
