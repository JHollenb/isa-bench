"""Compiled native callback for compact analytic ocean sources.

The native display terminal consumes ``int(void*, y, x, need, out[3])``.  This
module owns the small bridge around ``sfr_ocean_analytic.c``: atom/jet
parameters are copied into a native context once per frame, and every query
then evaluates ``exp``/``cos`` algebra in C.  No Python callback, NumPy point
allocation, dense Fourier plane, or image grid is involved in the native
route.

Residuals without a native evaluator are rejected by this route.  The caller
can inspect ``native_sampling_capability`` and use ``execute_fields`` so the
prepared Python consumer evaluates that residual explicitly.
"""

from __future__ import annotations

import ctypes
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Mapping
from contextlib import suppress
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .model import Region

_F64P = ctypes.POINTER(ctypes.c_double)
_VOIDP = ctypes.c_void_p
_U64 = ctypes.c_uint64

STATUS_OK = 0
STATUS_CLOSED = 1
STATUS_CONTEXT = 2
STATUS_INVALID_QUERY = 3
STATUS_EVALUATION = 4
STATUS_NONFINITE = 5
STATUS_COMPLEX = 6

KIND_ATOM_MIXTURE = 0
KIND_JET = 1


class AnalyticNativeError(RuntimeError):
    """Raised when the compiled analytic callback rejects a query/context."""

    def __init__(self, message: str, *, status: int = STATUS_EVALUATION) -> None:
        super().__init__(message)
        self.status = int(status)


class AnalyticNativeUnavailableError(AnalyticNativeError):
    """Raised when a source needs the explicit vectorized fallback route."""


# Short spelling retained for callers that use the capability/error name
# directly; the concrete exception follows the project Error naming rule.
AnalyticNativeUnavailable = AnalyticNativeUnavailableError


class AnalyticNativeStats(ctypes.Structure):
    """Counters owned by one native analytic context."""

    _fields_ = [
        ("samples", _U64),
        ("gradient_queries", _U64),
        ("source_terms", _U64),
        ("max_abs_imaginary_leakage", ctypes.c_double),
    ]

    def to_dict(self) -> dict[str, int | float]:
        return {
            name: float(getattr(self, name)) if name.startswith("max_") else int(getattr(self, name))
            for name, *_ in self._fields_
        }


def _suffix() -> str:
    if sys.platform == "darwin":
        return "dylib"
    if sys.platform == "win32":
        return "dll"
    return "so"


def library_path() -> Path:
    """Return the optional compiled analytic library path."""

    root = Path(__file__).resolve().parents[3]
    explicit = os.environ.get("SPF_OCEAN_ANALYTIC_LIBRARY")
    if explicit:
        return Path(explicit)
    return root / "native" / f"libspf_ocean_analytic.{_suffix()}"


def _configure(library_object: Any) -> Any:
    library_object.spf_sfr_ocean_analytic_abi_version.argtypes = []
    library_object.spf_sfr_ocean_analytic_abi_version.restype = ctypes.c_int
    library_object.spf_sfr_ocean_analytic_context_create.argtypes = [
        ctypes.c_uint32,
        ctypes.c_size_t,
        _F64P,
        ctypes.c_int32,
        _F64P,
        _F64P,
        ctypes.c_double,
        ctypes.POINTER(_VOIDP),
    ]
    library_object.spf_sfr_ocean_analytic_context_create.restype = ctypes.c_int
    library_object.spf_sfr_ocean_analytic_context_stats.argtypes = [_VOIDP, ctypes.POINTER(AnalyticNativeStats)]
    library_object.spf_sfr_ocean_analytic_context_stats.restype = ctypes.c_int
    library_object.spf_sfr_ocean_analytic_context_release.argtypes = [_VOIDP]
    library_object.spf_sfr_ocean_analytic_context_release.restype = None
    library_object.spf_sfr_ocean_analytic_sample.argtypes = [
        _VOIDP,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        _F64P,
    ]
    library_object.spf_sfr_ocean_analytic_sample.restype = ctypes.c_int
    return library_object


@lru_cache(maxsize=1)
def library() -> Any:
    """Load the compiled analytic library, or return ``None`` when absent."""

    path = library_path()
    if not path.is_file():
        if os.environ.get("SPF_OCEAN_ANALYTIC_LIBRARY"):
            raise RuntimeError(f"requested analytic ocean library does not exist: {path}")
        return None
    configured = _configure(ctypes.CDLL(str(path)))
    if configured.spf_sfr_ocean_analytic_abi_version() != 1:
        raise RuntimeError("native analytic ocean ABI mismatch; rebuild the analytic target")
    return configured


def available() -> bool:
    """Return whether the optional compiled analytic library is available."""

    return library() is not None


def compile_library(output: str | os.PathLike[str] | None = None) -> Path:
    """Build the optional analytic callback library outside a render call."""

    compiler = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if compiler is None:
        raise RuntimeError("a C compiler is required to build the analytic ocean library")
    root = Path(__file__).resolve().parents[3]
    source = root / "native" / "src" / "scalar" / "analytic" / "sfr_ocean_analytic.c"
    include = root / "native" / "include"
    if not source.is_file():
        raise RuntimeError(f"analytic ocean source is missing: {source}")
    target = Path(output) if output is not None else root / "native" / f"libspf_ocean_analytic.{_suffix()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        compiler,
        "-std=c99",
        "-shared",
        "-fPIC",
        "-O3",
        "-ffp-contract=off",
        "-I",
        str(include),
        str(source),
        "-o",
        str(target),
        "-lm",
    ]
    subprocess.run(command, check=True)
    library.cache_clear()
    return target


def _complex_pair(value: Any, *, name: str) -> tuple[float, float]:
    try:
        scalar = complex(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AnalyticNativeUnavailable(f"analytic {name} is not a complex scalar") from exc
    if not math.isfinite(scalar.real) or not math.isfinite(scalar.imag):
        raise AnalyticNativeUnavailable(f"analytic {name} is non-finite", status=STATUS_NONFINITE)
    return float(scalar.real), float(scalar.imag)


def _atom_parameters(atoms: Any) -> np.ndarray:
    values: list[float] = []
    for index, atom in enumerate(tuple(atoms)):
        if getattr(atom, "dimension", None) != 2:
            raise AnalyticNativeUnavailable(f"analytic atom {index} must be two-dimensional")
        h = np.asarray(atom.h, dtype=np.complex128).reshape(-1)
        matrix = np.asarray(atom.C, dtype=np.complex128)
        if h.shape != (2,) or matrix.shape != (2, 2):
            raise AnalyticNativeUnavailable(f"analytic atom {index} must have h=(2,) and C=(2,2)")
        values.extend(_complex_pair(atom.kappa, name=f"atom[{index}].kappa"))
        for item in h:
            values.extend(_complex_pair(item, name=f"atom[{index}].h"))
        for item in matrix.reshape(-1):
            values.extend(_complex_pair(item, name=f"atom[{index}].C"))
    if not values:
        raise AnalyticNativeUnavailable("analytic atom mixture is empty")
    result = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    result.setflags(write=False)
    return result


def _jet_parameters(jet: Any, covariance: np.ndarray | None) -> np.ndarray:
    if str(getattr(jet, "value_kind", "native")).lower() not in {"native", "real", "complex"}:
        raise AnalyticNativeUnavailable(f"native analytic callback does not decode Jet2D value kind {jet.value_kind!r}")
    value = complex(jet.value)
    if covariance is not None:
        value += 0.5 * (
            complex(jet.hess_xx) * covariance[0, 0]
            + 2.0 * complex(jet.hess_xy) * covariance[0, 1]
            + complex(jet.hess_yy) * covariance[1, 1]
        )
    fields = (
        value,
        complex(jet.grad_x),
        complex(jet.grad_y),
        complex(jet.hess_xx),
        complex(jet.hess_xy),
        complex(jet.hess_yy),
    )
    values: list[float] = []
    for index, item in enumerate(fields):
        values.extend(_complex_pair(item, name=f"jet[{index}]"))
    result = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    result.setflags(write=False)
    return result


def _support_descriptor(support: Any) -> tuple[int, np.ndarray, np.ndarray]:
    """Precompute the exact AABB/empty support cases understood by C."""

    if support is None:
        return 0, np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    if isinstance(support, Region):
        region = support
        lower = np.asarray(region.lower, dtype=np.float64)
        upper = np.asarray(region.upper, dtype=np.float64)
        if lower.shape != (2,) or upper.shape != (2,):
            raise AnalyticNativeUnavailable("analytic native support must be two-dimensional")
        return 2, lower, upper
    kind = str(getattr(support, "kind", "")).upper()
    if kind.endswith("EMPTY"):
        return 1, np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    if kind.endswith("FULL") or kind.endswith("UNKNOWN") or kind.endswith("FREQUENCY_BAND"):
        return 0, np.zeros(2, dtype=np.float64), np.zeros(2, dtype=np.float64)
    if not kind.endswith("AABB"):
        raise AnalyticNativeUnavailable(f"native analytic support kind {kind!r} has no compiled point mask")
    nested_region = getattr(support, "region", None)
    if nested_region is None:
        raise AnalyticNativeUnavailable("AABB analytic native support requires a region")
    return _support_descriptor(nested_region)


def _source_parameters(prepared: Any, footprint: Any) -> tuple[int, int, np.ndarray, np.ndarray | None]:
    """Extract one compact native parameter payload from a prepared source."""

    from .ocean_analytic import _filter_atoms, _footprint_covariance

    representation = str(getattr(prepared, "representation", ""))
    atom_source = getattr(prepared, "atom_source", None)
    residual = None if atom_source is None else getattr(atom_source, "residual", None)
    if representation in {"jet", "quadratic_jet"}:
        residual = getattr(prepared, "jet_residual", None)
    if residual is not None:
        raise AnalyticNativeUnavailable(
            "analytic residual has no compiled evaluator; use execute_fields for explicit residual evaluation"
        )
    if representation in {"atom", "mixture", "gaussian_chirp_atom", "atom_mixture"}:
        if atom_source is None:
            raise AnalyticNativeUnavailable("analytic atom source is unavailable")
        atoms = tuple(atom_source.atoms)
        if footprint is not None and footprint is not getattr(prepared, "footprint", None):
            # ``atom_source.atoms`` includes a preparation footprint when one
            # was supplied.  A distinct sampler footprint starts from the
            # current unfiltered spatial frame to avoid applying two filters.
            base_atoms = tuple(getattr(prepared, "_base_spatial_atoms", ())) or atoms
            try:
                atoms = _filter_atoms(base_atoms, footprint)
            except ValueError as exc:
                raise AnalyticNativeUnavailable(f"native atom footprint convolution failed: {exc}") from exc
        return KIND_ATOM_MIXTURE, len(atoms), _atom_parameters(atoms), None
    if representation in {"jet", "quadratic_jet"}:
        jet = getattr(prepared, "jet", None)
        if jet is None:
            raise AnalyticNativeUnavailable("analytic jet source is unavailable")
        if footprint is None or footprint is getattr(prepared, "footprint", None):
            covariance = getattr(prepared, "footprint_covariance", None)
        else:
            covariance = _footprint_covariance(footprint)
        return KIND_JET, 1, _jet_parameters(jet, covariance), covariance
    raise AnalyticNativeUnavailable(f"compiled analytic callback does not consume representation {representation!r}")


def _status_message(status: int) -> str:
    return {
        STATUS_CLOSED: "analytic native context is closed",
        STATUS_CONTEXT: "analytic native context is invalid",
        STATUS_INVALID_QUERY: "analytic native query coordinates must be finite",
        STATUS_EVALUATION: "analytic native source evaluation failed",
        STATUS_NONFINITE: "analytic native source produced a non-finite value",
        STATUS_COMPLEX: "analytic native source has unresolved complex leakage",
    }.get(int(status), f"analytic native callback failed with status {int(status)}")


class AnalyticNativeSampler:
    """Persistent C callback context for one compact analytic source frame."""

    __slots__ = (
        "_closed",
        "_context",
        "_frame",
        "_function_ptr",
        "_kind",
        "_last_status",
        "_library",
        "_params",
        "_sample",
        "_source_count",
        "_support_lower",
        "_support_mode",
        "_support_upper",
        "cache_capacity",
        "footprint",
        "prepared",
        "source",
    )

    def __init__(
        self,
        prepared: Any,
        *,
        footprint: Any = None,
        cache_capacity: int = 4096,
    ) -> None:
        del cache_capacity  # Kept as an API compatibility spelling; C owns no point cache.
        if not callable(getattr(prepared, "native_sample", None)):
            raise TypeError("analytic native sampler requires a prepared analytic source")
        if not bool(getattr(prepared, "native_queries_supported", True)):
            raise AnalyticNativeUnavailable(
                "prepared analytic source has no native query capability; use execute_fields for fallback evaluation"
            )
        output_format = str(getattr(prepared, "output_format", "real")).lower()
        if output_format not in {"real", "native", "height", "signed_real"}:
            raise AnalyticNativeUnavailable("native analytic callback requires a real output format")
        self.prepared = prepared
        self.source = prepared
        self.footprint = footprint
        self.cache_capacity = 0
        self._library = _require_library()
        self._sample = self._library.spf_sfr_ocean_analytic_sample
        self._context = _VOIDP()
        self._function_ptr = ctypes.cast(self._sample, _VOIDP)
        self._params = np.empty(0, dtype=np.float64)
        self._kind = -1
        self._source_count = 0
        self._support_mode = 0
        self._support_lower = np.zeros(2, dtype=np.float64)
        self._support_upper = np.zeros(2, dtype=np.float64)
        self._frame = 0
        self._closed = False
        self._last_status = STATUS_OK
        self._build_context()

    def _build_context(self) -> None:
        kind, source_count, params, _covariance = _source_parameters(self.prepared, self.footprint)
        support_mode, lower, upper = _support_descriptor(getattr(self.prepared, "support", None))
        lower = np.ascontiguousarray(lower, dtype=np.float64)
        upper = np.ascontiguousarray(upper, dtype=np.float64)
        context = _VOIDP()
        status = self._library.spf_sfr_ocean_analytic_context_create(
            int(kind),
            int(source_count),
            params.ctypes.data_as(_F64P),
            int(support_mode),
            lower.ctypes.data_as(_F64P),
            upper.ctypes.data_as(_F64P),
            max(0.0, float(getattr(self.prepared, "tolerance", 1.0e-10))),
            ctypes.byref(context),
        )
        if int(status) != STATUS_OK:
            raise AnalyticNativeError(_status_message(int(status)), status=int(status))
        self._context = context
        self._params = params
        self._kind = int(kind)
        self._source_count = int(source_count)
        self._support_mode = int(support_mode)
        self._support_lower = lower
        self._support_upper = upper
        self._closed = False
        self._last_status = STATUS_OK

    @property
    def context_ptr(self) -> ctypes.c_void_p:
        """Opaque native context pointer for the display callback."""

        return self._context if not self._closed else _VOIDP()

    @property
    def function_ptr(self) -> ctypes.c_void_p:
        """Native C function pointer with ``(context,y,x,need,out[3])`` ABI."""

        return self._function_ptr if not self._closed else _VOIDP()

    @property
    def callback(self) -> Any:
        """Configured C callback function object."""

        return self._sample

    @property
    def frame(self) -> int:
        return self._frame

    @property
    def diagnostics(self) -> Mapping[str, Any]:
        stats = AnalyticNativeStats()
        if not self._closed:
            status = self._library.spf_sfr_ocean_analytic_context_stats(self._context, ctypes.byref(stats))
            self._last_status = int(status)
        plan = getattr(self.prepared, "plan", None)
        ledger = getattr(plan, "error", None)
        ledger_values = (
            {
                name: float(getattr(ledger, name))
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
            if ledger is not None
            else {}
        )
        return MappingProxyType(
            {
                "consumer": "analytic_native_c",
                "route": "analytic_native_callback",
                "representation": getattr(plan, "representation", "unknown"),
                "compact_source": True,
                "native_compact_parameters": True,
                "native_precision": "fp64_analytic",
                "callback_abi": "int(void*,double y,double x,int,double out_h_dy_dx[3])",
                "callback_coordinates": "physical_yx",
                "callback_output": "height,gradient_y,gradient_x",
                "chart_preprocessing": "prepared_physical_coordinates",
                "support_preprocessing": "native_context_aabb",
                "residual_policy": "explicit_vectorized_fallback",
                "real_projection": "real_part",
                "frame": self._frame,
                "source_count": self._source_count,
                "support_mode": self._support_mode,
                "error_ledger": ledger_values,
                "error_bound": None if ledger is None else float(ledger.total_bound),
                "error_bound_semantics": "representation_and_declared_approximation_only",
                "numerical_evaluation_error": "not_certified",
                "temporal_residual_bound": float(getattr(plan, "temporal_residual_bound", 0.0)),
                "stats": stats.to_dict(),
                "imaginary_leakage": float(stats.max_abs_imaginary_leakage),
                "last_status": self._last_status,
                "dense_source_materializations": 0,
                "fft_count": 0,
                "point_cache": "none",
            }
        )

    def native_sampling_capability(self) -> Mapping[str, Any]:
        """Report callback eligibility with explicit native provenance."""

        return MappingProxyType(
            {
                "supported": not self._closed,
                "representation": getattr(getattr(self.prepared, "plan", None), "representation", "unknown"),
                "route": "analytic_native_callback",
                "precision": "fp64_analytic",
                "reason": (
                    "compiled compact atom/jet context is available"
                    if not self._closed
                    else "native analytic context is closed"
                ),
            }
        )

    def _check_status(self, status: int) -> None:
        self._last_status = int(status)
        if int(status) != STATUS_OK:
            raise AnalyticNativeError(_status_message(int(status)), status=int(status))

    def sample(self, y: float, x: float, *, need_gradients: bool = True) -> np.ndarray:
        """Evaluate one physical ``(y,x)`` query through the C kernel."""

        if self._closed:
            raise AnalyticNativeError(_status_message(STATUS_CLOSED), status=STATUS_CLOSED)
        try:
            y_value, x_value = float(y), float(x)
        except (TypeError, ValueError, OverflowError) as exc:
            raise AnalyticNativeError(
                "analytic native query coordinates must be real", status=STATUS_INVALID_QUERY
            ) from exc
        output = np.empty(3, dtype=np.float64)
        status = self._sample(
            self._context,
            y_value,
            x_value,
            int(bool(need_gradients)),
            output.ctypes.data_as(_F64P),
        )
        self._check_status(int(status))
        return output

    invoke = sample

    @property
    def stats(self) -> AnalyticNativeStats:
        result = AnalyticNativeStats()
        if self._closed:
            return result
        self._check_status(int(self._library.spf_sfr_ocean_analytic_context_stats(self._context, ctypes.byref(result))))
        return result

    def refresh(self) -> None:
        """Rebuild the native context from the current prepared source frame."""

        if self._closed:
            raise AnalyticNativeError(_status_message(STATUS_CLOSED), status=STATUS_CLOSED)
        old_context = self._context
        self._library.spf_sfr_ocean_analytic_context_release(old_context)
        self._context = _VOIDP()
        try:
            self._build_context()
        except Exception:
            self._closed = True
            raise
        self._frame += 1

    begin_frame = refresh

    def close(self) -> None:
        """Release the native compact context."""

        if not self._closed and self._context:
            self._library.spf_sfr_ocean_analytic_context_release(self._context)
        self._context = _VOIDP()
        self._function_ptr = _VOIDP()
        self._closed = True

    def __enter__(self) -> AnalyticNativeSampler:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


PreparedAnalyticNativeSampler = AnalyticNativeSampler


def _require_library() -> Any:
    selected = library()
    if selected is None:
        raise RuntimeError(
            "native analytic ocean library is unavailable; build it with "
            "spf.renderer.ocean_analytic_native.compile_library()"
        )
    return selected


__all__ = [
    "KIND_ATOM_MIXTURE",
    "KIND_JET",
    "STATUS_CLOSED",
    "STATUS_COMPLEX",
    "STATUS_CONTEXT",
    "STATUS_EVALUATION",
    "STATUS_INVALID_QUERY",
    "STATUS_NONFINITE",
    "STATUS_OK",
    "AnalyticNativeError",
    "AnalyticNativeSampler",
    "AnalyticNativeStats",
    "AnalyticNativeUnavailable",
    "AnalyticNativeUnavailableError",
    "PreparedAnalyticNativeSampler",
    "available",
    "compile_library",
    "library",
    "library_path",
]
