"""One-materialization spectral filter chains.

The filter-chain path is the first SFR vertical slice.  Translation, Gaussian
footprints, exposure, LOD windows, Fresnel propagation, and thin-lens phase
are represented as multiplicative transfer functions.  The eager variant is
kept as a useful baseline; fused, dense-SPF, and structured variants expose
how many transforms, buffers, and conversions each strategy actually used.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf.renderer.assets import SpectralTexture, lod_window
from spf.renderer.carriers import decode_words, dft_frequencies, encode_words
from spf.renderer.model import ErrorBudget, PrecisionContract, PrecisionMode
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision, relative_l2
from spf.tensor import v02

if TYPE_CHECKING:
    from spf.renderer.analytic_filter_chain import PreparedAnalyticFilterChain
    from spf.renderer.atom_filter_chain import PreparedAtomFilterChain
    from spf.renderer.model import SampleLayout


class FilterVariant(StrEnum):
    EAGER_FP32 = "fp32_eager"
    FUSED_FP32 = "fp32_fused"
    DENSE_SPF = "dense_spf"
    STRUCTURED = "structured"


ExecutionVariant = FilterVariant


def _variant(value: FilterVariant | str) -> FilterVariant:
    if isinstance(value, FilterVariant):
        return value
    name = str(getattr(value, "value", value)).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "eager": FilterVariant.EAGER_FP32,
        "fp32": FilterVariant.EAGER_FP32,
        "fp32_eager": FilterVariant.EAGER_FP32,
        "fused": FilterVariant.FUSED_FP32,
        "fp32_fused": FilterVariant.FUSED_FP32,
        "spf": FilterVariant.DENSE_SPF,
        "dense": FilterVariant.DENSE_SPF,
        "dense_spf": FilterVariant.DENSE_SPF,
        "sfr": FilterVariant.STRUCTURED,
        "structured": FilterVariant.STRUCTURED,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise ValueError(f"unknown filter-chain variant {value!r}") from exc


class FilterName(StrEnum):
    TRANSLATE = "translate"
    GAUSSIAN = "gaussian_filter"
    EXPOSURE = "exposure"
    LENS = "lens_phase"
    FRESNEL = "fresnel"
    LOD = "lod"
    PHASE = "phase"
    GLARE = "glare"
    ROTATE = "rotate"
    AFFINE = "affine"


# Design §17.2/§17.3: these remap coordinates instead of multiplying a spectral
# diagonal, so they have no transfer plane and no compact convolution kernel.
_COORDINATE_FILTERS = frozenset({FilterName.ROTATE, FilterName.AFFINE})


@dataclass(frozen=True, slots=True)
class FilterOperation:
    name: FilterName
    params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(getattr(self.name, "value", self.name)).strip().lower().replace("-", "_")
        aliases = {
            "gaussian": FilterName.GAUSSIAN.value,
            "lens": FilterName.LENS.value,
            "propagate": FilterName.FRESNEL.value,
        }
        object.__setattr__(self, "name", FilterName(aliases.get(name, name)))
        object.__setattr__(self, "params", MappingProxyType(dict(self.params)))


@dataclass(frozen=True, slots=True)
class FusedTransfer:
    """A transfer function assembled without intermediate field buffers."""

    transfer: np.ndarray
    log_magnitude: np.ndarray
    phase: np.ndarray
    operations: tuple[str, ...] = ()
    phase_only: bool = False

    def __post_init__(self) -> None:
        transfer = np.asarray(self.transfer, dtype=np.complex128)
        log_magnitude = np.asarray(self.log_magnitude, dtype=np.float64)
        phase = np.asarray(self.phase, dtype=np.float64)
        if transfer.shape != log_magnitude.shape or transfer.shape != phase.shape:
            raise ValueError("fused transfer planes must share one shape")
        object.__setattr__(self, "transfer", transfer)
        object.__setattr__(self, "log_magnitude", log_magnitude)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "operations", tuple(str(value) for value in self.operations))
        object.__setattr__(self, "phase_only", bool(self.phase_only))

    @property
    def magnitude(self) -> np.ndarray:
        return np.asarray(np.abs(self.transfer))

    @property
    def shape(self) -> tuple[int, ...]:
        return self.transfer.shape

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.transfer, dtype=dtype)


@dataclass(frozen=True, slots=True)
class FilterResult:
    """Values plus the transparent report returned by :meth:`FilterChain.run`."""

    values: Any
    report: RenderReport
    variant: FilterVariant
    words: np.ndarray | None = None

    @property
    def value(self) -> Any:
        return self.values

    @property
    def data(self) -> Any:
        return self.values

    def __iter__(self) -> Iterator[Any]:
        yield self.values
        yield self.report

    def __array__(self, dtype: Any = None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


def _shape_and_source(source: Any) -> tuple[np.ndarray, tuple[int, int]]:
    from spf.renderer.carriers import carrier_shape, complex_carrier

    shape = carrier_shape(source)
    if len(shape) != 2:
        raise ValueError("filter-chain sources must be two-dimensional")
    values = complex_carrier(source.reconstruct() if isinstance(source, SpectralTexture) else source)
    return values, (shape[0], shape[1])


def _spacing(value: Sequence[float] | float) -> tuple[float, float]:
    result: tuple[float, ...]
    if isinstance(value, (int, float, np.integer, np.floating)):
        scalar = float(value)
        result = (scalar, scalar)
    else:
        result = tuple(float(item) for item in value)
    if len(result) != 2 or any(item <= 0.0 or not math.isfinite(item) for item in result):
        raise ValueError("sample spacing must be a positive scalar or two-vector")
    return result[0], result[1]


def _frequency_grid(shape: tuple[int, int], spacing: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """DFT bin frequencies per axis in AXIS ORDER: axis ``i`` is paired with ``spacing[i]``.

    Design v0.2 §17.1 coordinate convention (fixed 2026-09-18): component 0 of
    every offset, frequency vector and covariance matrix belongs to array axis 0
    (rows).  Matches ``ocean_terminal._derivative_words`` (``spacing[axis]``) and
    ``carriers.dft_frequency_grid``.
    """

    f0 = dft_frequencies(shape[0], spacing[0])
    f1 = dft_frequencies(shape[1], spacing[1])
    return np.meshgrid(f0, f1, indexing="ij")


def _translation_params(params: Mapping[str, Any]) -> tuple[float, float]:
    """Return the user-facing ``(dx, dy)`` of a translation (x = columns, y = rows)."""

    if "offset" in params:
        offset = tuple(float(item) for item in params["offset"])
    else:
        offset = tuple(float(params.get(key, 0.0)) for key in ("dx", "dy"))
    if len(offset) != 2 or any(not math.isfinite(item) for item in offset):
        raise ValueError("translation requires a finite (dx, dy) offset")
    return offset


# Row/column exchange.  ``_AXIS_SWAP @ v`` sends a user-facing ``(x, y)`` vector
# to axis order ``(axis 0, axis 1)``; ``_AXIS_SWAP @ M @ _AXIS_SWAP`` does the
# same for a matrix.  The swap is its own inverse.
_AXIS_SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])


def _covariance_param(params: Mapping[str, Any]) -> np.ndarray:
    covariance = np.asarray(params["covariance"], dtype=np.float64)
    if covariance.shape != (2, 2) or not np.allclose(covariance, covariance.T, atol=1.0e-14, rtol=0.0):
        raise ValueError("Gaussian filter covariance must be a symmetric 2x2 matrix")
    if float(np.min(np.linalg.eigvalsh(covariance))) < -1.0e-12:
        raise ValueError("Gaussian filter covariance must be positive semidefinite")
    return covariance


def _axis_order_params(name: FilterName, params: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one operation's user-facing ``(x, y)`` parameters to axis order.

    The single boundary required by the design §17.1 convention: `FilterChain`
    keeps ``dx``/``dy`` and ``(sigma_x, sigma_y)`` matrices in the reader's
    ``x = columns`` names; everything downstream (the IR, `transfers`, the
    runtime, the word datapath, the compact kernels) sees axis order only.
    """

    result = dict(params)
    if name is FilterName.TRANSLATE:
        dx, dy = _translation_params(params)
        result.pop("dx", None)
        result.pop("dy", None)
        result["offset"] = (dy, dx)
    elif name is FilterName.GAUSSIAN:
        result["covariance"] = _AXIS_SWAP @ _covariance_param(params) @ _AXIS_SWAP
    elif name in {FilterName.ROTATE, FilterName.AFFINE}:
        from spf.renderer.normalization import _affine_params

        matrix, offset = _affine_params(params, 2)
        result.pop("angle", None)
        result["matrix"] = _AXIS_SWAP @ matrix @ _AXIS_SWAP
        result["offset"] = _AXIS_SWAP @ offset
    return result


def _phase_and_log_magnitude(
    operation: FilterOperation,
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(log2 magnitude, phase radians)` for one spectral operation.

    ``operation.params`` are in AXIS ORDER here (see :func:`_axis_order_params`):
    ``offset[0]`` and ``covariance[0, 0]`` belong to array axis 0.
    """

    f0, f1 = _frequency_grid(shape, spacing)
    if operation.name is FilterName.TRANSLATE:
        offset = np.asarray(operation.params["offset"], dtype=np.float64)
        if offset.shape != (2,) or not np.all(np.isfinite(offset)):
            raise ValueError("translation requires a finite axis-order offset")
        return np.zeros(shape, dtype=np.float64), -2.0 * math.pi * (f0 * offset[0] + f1 * offset[1])
    if operation.name is FilterName.GAUSSIAN:
        covariance = _covariance_param(operation.params)
        vector = np.stack((f0, f1), axis=-1)
        log_magnitude = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", vector, covariance, vector) / math.log(2.0)
        return np.asarray(log_magnitude, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    if operation.name is FilterName.EXPOSURE:
        exposure = float(operation.params["ev"])
        if not math.isfinite(exposure):
            raise ValueError("exposure must be finite")
        return np.full(shape, exposure, dtype=np.float64), np.zeros(shape, dtype=np.float64)
    if operation.name is FilterName.GLARE:
        strength = float(operation.params["strength"])
        length = float(operation.params["length"])
        width = float(operation.params["width"])
        if not (math.isfinite(strength) and 0 <= strength <= 1):
            raise ValueError("glare strength must be finite and in [0, 1]")
        if not (math.isfinite(length) and math.isfinite(width) and length > 0 and width > 0):
            raise ValueError("glare length and width must be finite and positive")
        # Three elongated symmetric Gaussian PSFs produce six rays. The
        # identity component preserves detail; every component has unit DC.
        # Ray angles are measured in the reader's frame (x = columns = axis 1),
        # so the direction cosine multiplies ``f1`` and the sine ``f0``; this is
        # the same star `transfers.SpectralTransfer` builds in axis order.
        transfer = np.full(shape, 1 - strength, dtype=np.float64)
        for angle in (0.0, math.pi / 3, 2 * math.pi / 3):
            along = math.cos(angle) * f1 + math.sin(angle) * f0
            across = -math.sin(angle) * f1 + math.cos(angle) * f0
            transfer += (strength / 3) * np.exp(-2 * math.pi**2 * (length**2 * along**2 + width**2 * across**2))
        with np.errstate(divide="ignore"):
            return np.log2(transfer), np.zeros(shape, dtype=np.float64)
    if operation.name in {FilterName.LENS, FilterName.FRESNEL}:
        wavelength = float(operation.params["wavelength"])
        distance = float(
            operation.params.get("distance", operation.params.get("focal_length", operation.params.get("f", 0.0)))
        )
        if wavelength <= 0.0 or not math.isfinite(wavelength) or distance == 0.0 or not math.isfinite(distance):
            raise ValueError("lens/Fresnel filters require finite nonzero distance and positive wavelength")
        phase = -math.pi * wavelength * distance * (f0 * f0 + f1 * f1)
        if operation.name is FilterName.FRESNEL:
            phase = phase + (2.0 * math.pi * distance / wavelength)
        return np.zeros(shape, dtype=np.float64), phase
    if operation.name is FilterName.LOD:
        window = lod_window(
            shape,
            operation.params["cutoff"],
            window=str(operation.params.get("window", "raised_cosine")),
            centered=bool(operation.params.get("centered", False)),
        )
        with np.errstate(divide="ignore"):
            log_magnitude = np.log2(window)
        return log_magnitude, np.zeros(shape, dtype=np.float64)
    if operation.name is FilterName.PHASE:
        phase = np.asarray(operation.params["phase"], dtype=np.float64)
        phase = np.broadcast_to(phase, shape)
        return np.zeros(shape, dtype=np.float64), np.asarray(phase, dtype=np.float64)
    if operation.name in {FilterName.ROTATE, FilterName.AFFINE}:
        raise ValueError(
            f"{operation.name.value} remaps coordinates (design §17.2/§17.3); it has no diagonal transfer plane"
        )
    raise ValueError(f"unsupported filter operation {operation.name!r}")


# --------------------------------------------------------------------------- #
# Compact spatial kernels on SPF words (the ``conv(F, h)`` equivalent held by
# the equality-saturation planner, design §33.6).  A chain whose operations are
# all Gaussian footprints, exposures, scalar phases, and integer translations
# has a compact periodic kernel; the planner may execute it as a word
# convolution instead of ``ifft(mul(fft(F), H))`` when the backend cost model
# says it is cheaper and the certified bound fits the chain's stated relative
# error budget.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class WordKernel:
    """A compact periodic convolution kernel as SPF32 words with a certified bound."""

    offsets: np.ndarray  # (K, 2) integer lattice offsets (axis 0, axis 1)
    words: np.ndarray  # (K,) uint32 tap words
    values: np.ndarray  # (K,) tap values before quantization (Cartesian, plan-time only)
    representation_fit: float  # truncation + aliasing, relative to sum |k|
    tap_quantization: float  # max relative tap encode error (measured at plan time)
    reducer: float  # output word rounding, relative
    operations: tuple[str, ...]
    absolute_gain: float  # sum |k|

    def __post_init__(self) -> None:
        offsets = np.ascontiguousarray(np.asarray(self.offsets, dtype=np.int64).reshape(-1, 2))
        words = np.ascontiguousarray(np.asarray(self.words, dtype=np.uint32).reshape(-1))
        if len(offsets) != len(words) or len(words) == 0:
            raise ValueError("word kernel needs one word per tap")
        for name, array in (("offsets", offsets), ("words", words)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def taps(self) -> int:
        return len(self.words)

    @property
    def relative_bound(self) -> float:
        return float(self.representation_fit + self.tap_quantization + self.reducer)

    def ledger(self, scale: float = 1.0) -> v02.ErrorLedger:
        gain = float(scale) * self.absolute_gain
        return v02.ErrorLedger(
            source_quantization=self.tap_quantization * gain,
            representation_fit=self.representation_fit * gain,
            reducer=self.reducer * gain,
        )


def _gaussian_aliasing_bound(sigma_px: float) -> float:
    """Poisson-summation bound on |DFT(sampled Gaussian) - continuous transfer| for |f| <= 1/2."""

    if sigma_px <= 0.0:
        return math.inf
    total = 0.0
    for m in range(1, 64):
        term = math.exp(-2.0 * math.pi**2 * sigma_px**2 * (m - 0.5) ** 2)
        total += 2.0 * term
        if term < 1.0e-18:
            break
    return total


_Taps = dict[tuple[int, int], complex]


def _kernel_convolve(a: _Taps, b: _Taps) -> _Taps:
    out: dict[tuple[int, int], complex] = {}
    for (ay, ax), av in a.items():
        for (by, bx), bv in b.items():
            key = (ay + by, ax + bx)
            out[key] = out.get(key, 0.0) + av * bv
    return out


def _gaussian_taps(covariance: np.ndarray, spacing: tuple[float, float], tolerance: float) -> tuple[_Taps, float]:
    """Lattice taps of the unit-mass Gaussian ``N(0, covariance)`` and their certified relative bound."""

    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (2, 2) or not np.allclose(cov, cov.T, atol=1.0e-14, rtol=0.0):
        raise ValueError("Gaussian kernel covariance must be a symmetric 2x2 matrix")
    scale = np.diag([float(spacing[0]), float(spacing[1])])
    # Physical covariance -> pixel covariance (transfer frequencies are cycles per physical unit).
    cov_px = np.linalg.inv(scale) @ cov @ np.linalg.inv(scale)
    eig = np.linalg.eigvalsh(cov_px)
    if float(eig.min()) <= 1.0e-12:
        raise ValueError("degenerate Gaussian has no compact two-dimensional kernel")
    radius_sq = 2.0 * math.log(1.0 / tolerance)
    reach = math.ceil(math.sqrt(radius_sq * float(eig.max()))) + 1
    inverse = np.linalg.inv(cov_px)
    grid = np.arange(-reach, reach + 1, dtype=np.float64)
    qy, qx = np.meshgrid(grid, grid, indexing="ij")
    maha = inverse[0, 0] * qy * qy + 2.0 * inverse[0, 1] * qy * qx + inverse[1, 1] * qx * qx
    keep = maha <= radius_sq
    weights = 2.0 ** (-0.5 * maha[keep] / math.log(2.0))
    weights = weights / float(weights.sum())
    taps = {
        (int(y), int(x)): complex(float(w))
        for y, x, w in zip(qy[keep].astype(int), qx[keep].astype(int), weights, strict=True)
    }
    sigma_min = math.sqrt(float(eig.min()))
    aliasing = (1.0 + _gaussian_aliasing_bound(sigma_min)) ** 2 - 1.0
    truncation = 2.0 * tolerance / (1.0 - tolerance)
    return taps, aliasing + truncation


@dataclass(frozen=True, slots=True)
class FilterChain:
    """An immutable compatible spectral filter graph."""

    source: Any
    operations: tuple[FilterOperation, ...] = ()
    sample_spacing: tuple[float, float] = (1.0, 1.0)
    precision: PrecisionContract = field(default_factory=PrecisionContract)
    error_budget: ErrorBudget = field(default_factory=ErrorBudget)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _prepared: dict[
        tuple[str, SampleLayout | None],
        PreparedAnalyticFilterChain | PreparedAtomFilterChain | PreparedStructuredFilterChain,
    ] = field(default_factory=dict, init=False, repr=False, compare=False)
    _saturation: dict[str, Any] = field(default_factory=dict, init=False, repr=False, compare=False)
    _kernels: dict[tuple[int, ...], Any] = field(default_factory=dict, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        from spf.renderer.atom_filter_chain import is_atom_source
        from spf.renderer.carriers import as_block_view, carrier_shape, field_carrier
        from spf.renderer.model import Field

        if isinstance(self.source, Field):
            from spf.renderer.normalization import _field_ledger_after_operations
            from spf.renderer.runtime import _apply_representation_error

            object.__setattr__(
                self,
                "metadata",
                {
                    **self.metadata,
                    "source_shape": carrier_shape(self.source) or self.metadata.get("source_shape", ()),
                    "source_domain": self.source.domain,
                    "source_chart": self.source.chart,
                    "source_support": self.source.support,
                    "source_error_includes_payload": True,
                    "source_error": _apply_representation_error(
                        self.source, _field_ledger_after_operations(self.source)
                    ),
                },
            )
            object.__setattr__(self, "source", field_carrier(self.source))

        shape = self.metadata.get("source_shape", carrier_shape(self.source))
        if len(shape) != 2 or any(int(n) != n or n <= 0 for n in shape):
            raise ValueError("filter-chain source must be two-dimensional")
        actual_shape = carrier_shape(self.source)
        if actual_shape and tuple(shape) != tuple(actual_shape):
            raise ValueError("filter-chain shape does not match its source carrier")
        block = as_block_view(self.source)
        if block is not None:
            object.__setattr__(self, "source", block)
        elif is_atom_source(self.source):
            from spf.renderer.model import _freeze_value

            object.__setattr__(self, "source", _freeze_value(self.source))
        elif not isinstance(self.source, SpectralTexture):
            source = np.asarray(self.source)
            if source.flags.writeable:
                source = source.copy()
                source.setflags(write=False)
            object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "operations",
            tuple(
                operation if isinstance(operation, FilterOperation) else FilterOperation(operation)
                for operation in self.operations
            ),
        )
        object.__setattr__(self, "sample_spacing", _spacing(self.sample_spacing))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        if not shape:
            raise ValueError("filter-chain source cannot be empty")

    @classmethod
    def from_source(
        cls,
        source: Any,
        *,
        sample_spacing: Sequence[float] | float = (1.0, 1.0),
        precision: PrecisionContract | None = None,
        error_budget: ErrorBudget | None = None,
        metadata: Mapping[str, Any] | None = None,
        compute_dtype: str | None = None,
        shape: tuple[int, int] | None = None,
    ) -> FilterChain:
        return cls(
            source,
            (),
            _spacing(sample_spacing),
            precision or PrecisionContract(),
            error_budget or ErrorBudget(),
            {
                **(metadata or {}),
                **({"source_shape": shape} if shape is not None else {}),
                **({"compute_dtype": compute_dtype} if compute_dtype is not None else {}),
            },
        )

    from_texture = from_source

    @property
    def shape(self) -> tuple[int, int]:
        from spf.renderer.carriers import carrier_shape

        shape = self.metadata.get("source_shape", carrier_shape(self.source))
        return shape[0], shape[1]

    @property
    def filter_count(self) -> int:
        return len(self.operations)

    def _append(self, name: FilterName, **params: Any) -> FilterChain:
        return FilterChain(
            self.source,
            (*self.operations, FilterOperation(name, params)),
            self.sample_spacing,
            self.precision,
            self.error_budget,
            self.metadata,
        )

    def translate(
        self,
        *offset: float,
        dx: float | None = None,
        dy: float | None = None,
    ) -> FilterChain:
        if offset:
            if len(offset) != 2:
                raise ValueError("translation requires two coordinates")
            return self._append(FilterName.TRANSLATE, offset=tuple(float(item) for item in offset))
        return self._append(
            FilterName.TRANSLATE, dx=0.0 if dx is None else float(dx), dy=0.0 if dy is None else float(dy)
        )

    shift = translate

    def gaussian_filter(self, covariance: Any) -> FilterChain:
        return self._append(FilterName.GAUSSIAN, covariance=np.asarray(covariance, dtype=np.float64).copy())

    gaussian = gaussian_filter
    footprint = gaussian_filter

    def rotate(self, angle: float) -> FilterChain:
        """Design §17.2 rotation ``g(x) = f(Rx)``, angle in radians.

        The angle is stated in the reader's right-handed ``(x, y)`` frame with
        x along the columns.  Index space is ``(row, column)``, i.e. the same
        frame with the axes exchanged, so a positive angle turns the opposite way
        there; ``rotate(-pi/2)`` is the axis-order quarter turn.  Quarter-turn
        multiples stay exact index permutations on every representation.
        """

        return self._append(FilterName.ROTATE, angle=float(angle))

    def affine(self, matrix: Any, offset: Any = (0.0, 0.0)) -> FilterChain:
        """Design §17.3 general affine ``g(x) = f(Ax + b)`` in the reader's (x, y) frame.

        ``affine(identity, -d)`` is exactly ``translate(*d)``.  The spectrum
        picks up ``|det A|^-1``, i.e. ``ΔL = -256 log2|det A|``.
        """

        return self._append(
            FilterName.AFFINE,
            matrix=np.asarray(matrix, dtype=np.float64).copy(),
            offset=tuple(float(value) for value in np.asarray(offset, dtype=np.float64).reshape(-1)),
        )

    scale = affine

    def exposure(self, ev: float) -> FilterChain:
        return self._append(FilterName.EXPOSURE, ev=float(ev))

    expose = exposure

    def glare(self, *, strength: float = 0.15, length: float = 24.0, width: float = 1.0) -> FilterChain:
        """Apply a normalized six-ray Gaussian-mixture PSF in physical chart units."""
        return self._append(FilterName.GLARE, strength=strength, length=length, width=width)

    def lens_phase(self, *, focal_length: float, wavelength: float) -> FilterChain:
        return self._append(FilterName.LENS, focal_length=float(focal_length), wavelength=float(wavelength))

    lens = lens_phase

    def fresnel(self, distance: float, wavelength: float) -> FilterChain:
        return self._append(FilterName.FRESNEL, distance=float(distance), wavelength=float(wavelength))

    propagate = fresnel

    def phase_shift(self, phase: Any) -> FilterChain:
        return self._append(FilterName.PHASE, phase=np.asarray(phase, dtype=np.float64).copy())

    def lod(
        self, cutoff: float | Sequence[float], *, window: str = "raised_cosine", centered: bool = False
    ) -> FilterChain:
        return self._append(FilterName.LOD, cutoff=cutoff, window=str(window), centered=bool(centered))

    bandlimit = lod

    def fused_transfer(self) -> FusedTransfer:
        log_magnitude = np.zeros(self.shape, dtype=np.float64)
        phase = np.zeros(self.shape, dtype=np.float64)
        for operation in self.operations:
            delta_log, delta_phase = self._transfer_planes(operation)
            log_magnitude = log_magnitude + delta_log
            phase = phase + delta_phase
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            magnitude = np.exp2(log_magnitude)
            transfer = magnitude * np.exp(1j * phase)
        phase_only = bool(np.allclose(log_magnitude, 0.0, rtol=0.0, atol=1.0e-14))
        return FusedTransfer(
            transfer, log_magnitude, phase, tuple(operation.name.value for operation in self.operations), phase_only
        )

    fuse = fused_transfer

    def _source_values(self) -> np.ndarray:
        return _shape_and_source(self.source)[0]

    @property
    def source_is_fourier(self) -> bool:
        from spf.renderer.block_filter import FOURIER

        return self.metadata.get("source_domain") in FOURIER

    @staticmethod
    def _field_operation(operation: FilterOperation) -> Any:
        from spf.renderer.model import FieldOperation

        # The one user-facing -> axis-order boundary (design §17.1 convention).
        name, params = operation.name.value, _axis_order_params(operation.name, operation.params)
        if operation.name is FilterName.PHASE:
            params["phase_units"] = "radians"
        if operation.name is FilterName.LENS:
            name = "spectral_lens"
            params["distance"] = params.get("focal_length", params.get("f"))
        return FieldOperation(name, params)

    def _transfer_planes(self, operation: FilterOperation) -> tuple[np.ndarray, np.ndarray]:
        chart = self.metadata.get("source_chart")
        if chart is None:
            axis_operation = FilterOperation(operation.name, _axis_order_params(operation.name, operation.params))
            return _phase_and_log_magnitude(axis_operation, self.shape, self.sample_spacing)
        from spf.renderer.block_filter import chart_matrix
        from spf.renderer.model import FieldOperation
        from spf.renderer.transfers import SpectralTransfer

        matrix = chart_matrix(chart, 2)
        axes = (
            [np.arange(n) for n in self.shape] if self.source_is_fourier else [dft_frequencies(n) for n in self.shape]
        )
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        points = (
            points @ matrix.T + np.asarray(chart.origin) if self.source_is_fourier else points @ np.linalg.inv(matrix)
        )
        op = self._field_operation(operation)
        op = FieldOperation(op.op, {**op.params, "shape": self.shape})
        transfer = SpectralTransfer.identity(2).apply(op)
        logs, phase = transfer.log_phase(points.reshape(-1, 2), np.arange(math.prod(self.shape)))
        return logs.reshape(self.shape), phase.reshape(self.shape)

    def _single_transfer(self, operation: FilterOperation) -> np.ndarray:
        log_magnitude, phase = self._transfer_planes(operation)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            return np.asarray(np.exp2(log_magnitude) * np.exp(1j * phase), dtype=np.complex128)

    def _report(
        self,
        variant: FilterVariant,
        *,
        fft_count: int,
        materialized: int,
        potential: int,
        bytes_read: int,
        bytes_written: int,
        fallback_conversions: int = 0,
        ledger: v02.ErrorLedger | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> RenderReport:
        spatial_samples = int(math.prod(self.shape))
        cost_notes = {
            "variant": variant.value,
            "operations": tuple(operation.name.value for operation in self.operations),
            "filter_count": self.filter_count,
            "one_materialization": variant is not FilterVariant.EAGER_FP32,
            "materialized_buffers": 1 if variant is not FilterVariant.EAGER_FP32 else max(1, self.filter_count + 1),
            "potential_pass_elements": int(potential),
            "realized_pass_elements": int(materialized),
            "pass_avoidance": float(potential) / float(materialized) if materialized else 1.0,
            "spatial_materialization_avoidance": 1.0,
            "reference_only": True,
            "byte_traffic_status": "logical_model",
            "peak_memory_status": "logical_live_set_model",
            "energy_status": "unmeasured",
        }
        if diagnostics:
            cost_notes.update(diagnostics)
        metrics = RenderMetrics(
            backend="reference",
            workload="filter_chain",
            resolution=self.shape,
            representation=("structured_spectral_consumer" if variant is FilterVariant.STRUCTURED else variant.value),
            precision=cast(PrecisionMode, self.precision.mode).value.lower(),
            error_budget=self.error_budget.relative,
            bytes_read=int(bytes_read),
            bytes_written=int(bytes_written),
            peak_memory=int(max(materialized, 1) * 16),
            potential_elements=spatial_samples,
            materialized_elements=spatial_samples,
            fft_count=int(fft_count),
            fallback_conversions=int(fallback_conversions),
            field_operations=max(1, self.filter_count),
            reduction_operations=0,
        )
        selected = RepresentationDecision(
            field_id="filter_chain",
            region=str(self.shape),
            selected=variant.value,
            candidates=tuple(item.value for item in FilterVariant),
            predicted_cost=float(materialized + fft_count * self.shape[0] * self.shape[1]),
            predicted_error=0.0 if ledger is None else float(ledger.total_bound),
            reason="one fused spectral transfer"
            if variant is not FilterVariant.EAGER_FP32
            else "reference eager baseline",
        )
        return RenderReport(
            metrics,
            ledger or v02.ErrorLedger(),
            passes=(CompilerPassRecord("fuse_compatible_filters", self.filter_count, 1 if self.operations else 0),),
            decisions=(selected,),
            capabilities_used=(
                "FFT",
                "SPF32" if variant is FilterVariant.DENSE_SPF else "COMPLEX128",
                "structured_spectral_consumer" if variant is FilterVariant.STRUCTURED else variant.value,
            ),
            diagnostics=cost_notes,
        )

    # ---- equality-saturation planning ------------------------------------------

    def word_kernel(self, indices: Sequence[int], *, tolerance: float = 2.0**-20) -> tuple[WordKernel | None, str]:
        """Compact periodic kernel of the operations at ``indices`` as SPF32 words, or ``(None, reason)``."""

        label = tuple(int(i) for i in indices)
        if label in self._kernels:
            return cast(tuple[WordKernel | None, str], self._kernels[label])
        result = self._derive_word_kernel(label, tolerance)
        self._kernels[label] = result
        return result

    def _derive_word_kernel(self, label: tuple[int, ...], tolerance: float) -> tuple[WordKernel | None, str]:
        from spf.tensor import core

        if self.source_is_fourier:
            return None, "source is a Fourier-domain carrier; convolution needs a spatial word plane"
        taps: dict[tuple[int, int], complex] = {(0, 0): 1.0 + 0.0j}
        fit = 0.0
        names: list[str] = []
        for index in label:
            operation = self.operations[index]
            names.append(operation.name.value)
            if operation.name is FilterName.GAUSSIAN:
                # ``_gaussian_taps`` reads the covariance in axis order.
                axis_params = _axis_order_params(operation.name, operation.params)
                gaussian, bound = _gaussian_taps(axis_params["covariance"], self.sample_spacing, tolerance)
                taps = _kernel_convolve(taps, gaussian)
                fit = (1.0 + fit) * (1.0 + bound) - 1.0
            elif operation.name is FilterName.EXPOSURE:
                factor = 2.0 ** float(operation.params["ev"])
                taps = {key: value * factor for key, value in taps.items()}
            elif operation.name is FilterName.PHASE:
                phase = np.asarray(operation.params["phase"], dtype=np.float64)
                if phase.ndim != 0:
                    return None, "spatially varying phase has no compact kernel"
                factor = complex(math.cos(float(phase)), math.sin(float(phase)))
                taps = {key: value * factor for key, value in taps.items()}
            elif operation.name is FilterName.TRANSLATE:
                offset = _axis_order_params(operation.name, operation.params)["offset"]
                shift = (offset[0] / self.sample_spacing[0], offset[1] / self.sample_spacing[1])
                if any(abs(v - round(v)) > 1.0e-9 for v in shift):
                    return None, "sub-pixel translation has no compact kernel (sinc support)"
                taps = _kernel_convolve(taps, {(round(shift[0]), round(shift[1])): 1.0 + 0.0j})
            else:
                return None, f"{operation.name.value} has no compact spatial kernel"
        # The spectral form is a circular convolution, so taps wrap modulo the
        # extent; coincident wrapped taps add (this is exactly what the DFT does).
        wrapped: dict[tuple[int, int], complex] = {}
        for (y, x), value in taps.items():
            key = (y % self.shape[0], x % self.shape[1])
            wrapped[key] = wrapped.get(key, 0.0) + value
        taps = wrapped
        offsets = np.asarray(list(taps), dtype=np.int64).reshape(-1, 2)
        values = np.asarray([taps[tuple(o)] for o in offsets], dtype=complex)
        words = np.asarray(core.encode_words(values), dtype=np.uint32)
        decoded = np.asarray(core.decode_words(words))
        magnitude = np.abs(values)
        quantization = float(np.max(np.abs(decoded - values) / np.where(magnitude > 0, magnitude, 1.0)))
        # Output words are rounded once by the reducer: 2^-8 fractional L bits
        # and 2^-16 turns of phase, in relative terms.
        reducer = 2.0 ** (1 - v02.SPF32_L_FRAC_BITS) * math.log(2.0) + 2.0 * math.pi * 2.0**-16
        kernel = WordKernel(
            offsets,
            words,
            values,
            fit,
            quantization,
            reducer,
            tuple(names),
            float(np.sum(magnitude)),
        )
        return kernel, "compact periodic kernel"

    def word_kernel_legal(self, kernel: WordKernel) -> bool:
        return self.word_kernel_reason(kernel) == ""

    def word_kernel_reason(self, kernel: WordKernel) -> str:
        """Empty when the kernel's certified relative bound fits the stated budget; else why not."""

        budget = float(self.error_budget.relative)
        if not math.isfinite(budget):
            return "no finite relative error budget stated; the exact spectral form is kept"
        if kernel.relative_bound > budget:
            return f"certified relative bound {kernel.relative_bound:.3e} exceeds budget {budget:.3e}"
        return ""

    def saturation(self, output: str = "spatial", *, cost_model: Any = None) -> Any:
        """Equality-saturation plan for this chain and ``output`` (cached per output)."""

        from spf.renderer.compiler import saturate_filter_chain

        if cost_model is None:
            # ``metadata["cost_model"]`` pins the pricing (a CostModel or a
            # backend name); the default is the calibrated cpu model.
            pinned = self.metadata.get("cost_model")
            if isinstance(pinned, str):
                from spf.renderer.backends import get_backend

                cost_model = get_backend(pinned).cost_model
            else:
                cost_model = pinned
        name = str(output).lower().replace("-", "_")
        key = f"{name}:{getattr(cost_model, 'name', 'cpu')}:{id(cost_model) if cost_model is not None else 0}"
        if key not in self._saturation:
            self._saturation[key] = saturate_filter_chain(self, name, cost_model=cost_model)
        return self._saturation[key]

    @staticmethod
    def _attach_plan(report: RenderReport, plan: Any) -> RenderReport:
        if plan is None:
            return report
        return replace(
            report,
            passes=(*report.passes, plan.pass_record()),
            diagnostics={**report.diagnostics, **plan.diagnostics()},
            cost_estimate=plan.cost_estimate(),
            cost_source=plan.cost_model.calibration_source,
        )

    def prepare(
        self, *, output: str = "spatial", layout: SampleLayout | None = None
    ) -> PreparedAnalyticFilterChain | PreparedAtomFilterChain | PreparedStructuredFilterChain:
        from spf.renderer.analytic_filter_chain import PreparedAnalyticFilterChain
        from spf.renderer.assets import SpectralTexture
        from spf.renderer.atom_filter_chain import PreparedAtomFilterChain, is_atom_source
        from spf.renderer.blocks import BlockView
        from spf.renderer.model import Domain

        name = str(output).lower().replace("-", "_")
        key = (name, layout)
        if key not in self._prepared:
            prepared: Any
            if is_atom_source(self.source):
                prepared = PreparedAtomFilterChain(self, name, layout)
            else:
                typed_texture = (
                    self.source.typed_block_view()
                    if isinstance(self.source, SpectralTexture)
                    else None
                )
                block_source = typed_texture if typed_texture is not None else self.source
                block_chain = self
                if typed_texture is not None:
                    # A one-block DENSE_SPF texture is a local Fourier carrier.
                    # Keep the ordinary texture object for eager/dense controls;
                    # only the structured compiler receives this exact adapter.
                    block_chain = replace(
                        self,
                        source=typed_texture,
                        metadata={
                            **dict(self.metadata),
                            "source_domain": Domain.FOURIER_2D,
                            "source_error": v02.ErrorLedger(
                                representation_fit=float(self.source.error_bound)
                            ),
                            "texture_block_ingress": (
                                "packed_spf32_spectrum_to_tensor_block_view"
                            ),
                        },
                    )
                if isinstance(block_source, BlockView):
                    compute_dtype = np.dtype(block_chain.metadata.get("compute_dtype", "complex128"))
                    if compute_dtype not in (np.dtype("complex64"), np.dtype("complex128")):
                        raise ValueError("filter compute_dtype must be complex64 or complex128")
                    if any(operation.name in _COORDINATE_FILTERS for operation in block_chain.operations):
                        # A §17.2/§17.3 coordinate remap is outside the block
                        # filter's diagonal algebra, so it never reaches a filtered
                        # block terminal.  The typed field path keeps the packed
                        # words and lets the runtime own the remap.
                        prepared = PreparedCoordinateFilterChain(block_chain, name, layout)
                        self._prepared[key] = prepared
                        return cast(PreparedStructuredFilterChain, prepared)
                    plan = block_chain.saturation(name)
                    if plan.executes_convolution and layout is None:
                        try:
                            prepared = PreparedWordConvolutionFilterChain(block_chain, name, plan)
                        except ValueError as exc:
                            prepared = PreparedStructuredFilterChain(
                                block_chain, name, layout, plan, rejected=str(exc)
                            )
                    else:
                        prepared = PreparedStructuredFilterChain(block_chain, name, layout, plan)
                else:
                    if layout is not None:
                        raise ValueError("addressed filter layouts require a block or analytic atom source")
                    prepared = PreparedAnalyticFilterChain(self, name)
            self._prepared[key] = prepared
        return self._prepared[key]

    def run(
        self,
        variant: FilterVariant | str = FilterVariant.STRUCTURED,
        *,
        output: str = "spatial",
        reference: Any | None = None,
        layout: SampleLayout | None = None,
    ) -> FilterResult:
        """Execute one strategy and return values plus an auditable report."""

        selected = _variant(variant)
        if selected is FilterVariant.STRUCTURED:
            prepared_result = self.prepare(output=output, layout=layout).execute()
            if reference is not None:
                prepared_result.report.metrics.relative_l2 = relative_l2(prepared_result.values, reference)
            return cast(FilterResult, prepared_result)
        if layout is not None:
            raise ValueError("baseline filter variants require a full source grid")
        started = time.perf_counter()
        source = self._source_values()
        output_name = str(output).lower().replace("-", "_")
        if output_name not in {"spatial", "dense", "frequency", "spectrum", "spectrum_intensity", "intensity"}:
            raise ValueError(f"unsupported filter-chain output {output!r}")
        size = int(source.size)
        values: np.ndarray
        filtered: np.ndarray
        if selected is FilterVariant.EAGER_FP32:
            current = np.asarray(source, dtype=np.complex64)
            if self.source_is_fourier:
                for operation in self.operations:
                    current = current * self._single_transfer(operation).astype(np.complex64)
                filtered = current
                current = fft.ifftn(filtered, workers=1) if output_name in {"spatial", "dense"} else filtered
                fft_count = int(output_name in {"spatial", "dense"})
            elif not self.operations:
                spectrum = fft.fftn(current, workers=1)
                current = fft.ifftn(spectrum, workers=1).astype(np.complex64)
                filtered = spectrum
                fft_count = 2
            else:
                for operation in self.operations:
                    spectrum = fft.fftn(current, workers=1)
                    filtered = spectrum * self._single_transfer(operation).astype(np.complex64)
                    current = fft.ifftn(filtered, workers=1).astype(np.complex64)
                fft_count = 2 * self.filter_count
            if output_name in {"frequency", "spectrum"}:
                values = filtered
            elif output_name in {"spectrum_intensity", "intensity"}:
                values = np.abs(filtered) ** 2
            else:
                values = current
            report = self._report(
                selected,
                fft_count=fft_count,
                materialized=size * max(1, self.filter_count + 1),
                potential=size * max(1, 2 * self.filter_count + 1),
                bytes_read=size * 16 * max(1, self.filter_count),
                bytes_written=size * 16 * max(1, self.filter_count + 1),
            )
        elif selected is FilterVariant.FUSED_FP32:
            fused = self.fused_transfer()
            spectrum = np.asarray(source, dtype=np.complex64)
            if not self.source_is_fourier:
                spectrum = fft.fftn(spectrum, workers=1)
            filtered = spectrum * np.asarray(fused.transfer, dtype=np.complex64)
            if output_name in {"frequency", "spectrum"}:
                values = filtered
            elif output_name in {"spectrum_intensity", "intensity"}:
                values = np.abs(filtered) ** 2
            else:
                values = fft.ifftn(filtered, workers=1).astype(np.complex64)
            report = self._report(
                selected,
                fft_count=int(not self.source_is_fourier) + int(output_name in {"spatial", "dense"}),
                materialized=size,
                potential=size * max(1, self.filter_count + 1),
                bytes_read=size * 16,
                bytes_written=size * 16,
                diagnostics={"fused_transfer_terms": len(self.operations)},
            )
        elif selected is FilterVariant.DENSE_SPF:
            fused = self.fused_transfer()
            spectrum = np.asarray(source, dtype=np.complex128)
            if not self.source_is_fourier:
                spectrum = np.fft.fftn(spectrum)
            spectrum_words = np.asarray(encode_words(spectrum), dtype=np.uint32)
            transfer_words = np.asarray(encode_words(fused.transfer), dtype=np.uint32)
            from spf.tensor.core import spf_mul_words

            filtered_words = spf_mul_words(spectrum_words, transfer_words)
            if output_name in {"frequency", "spectrum"}:
                values = filtered_words
            elif output_name in {"spectrum_intensity", "intensity"}:
                from spf.renderer.continuity import project_packed_magnitude

                values = project_packed_magnitude(filtered_words, squared=True)
            else:
                values = np.fft.ifftn(decode_words(filtered_words))
            source_quantization = (
                float(np.max(np.abs(np.asarray(decode_words(spectrum_words)) - spectrum))) if size else 0.0
            )
            transfer_quantization = (
                float(np.max(np.abs(np.asarray(decode_words(transfer_words)) - fused.transfer))) if size else 0.0
            )
            ledger = v02.ErrorLedger(source_quantization=source_quantization + transfer_quantization)
            report = self._report(
                selected,
                fft_count=int(not self.source_is_fourier) + int(output_name in {"spatial", "dense"}),
                materialized=size,
                potential=size * max(1, self.filter_count + 1),
                bytes_read=size * 4 * 2,
                bytes_written=size * 4,
                fallback_conversions=2,
                ledger=ledger,
                diagnostics={"dense_spf_words": int(spectrum_words.size), "transfer_materialized_once": True},
            )
        result: np.ndarray = np.asarray(values)
        if output_name in {"spatial", "dense"} and np.all(np.abs(result.imag) <= 1.0e-12):
            result = result.real
        report = report.with_measured_cost((time.perf_counter() - started) * 1.0e3)
        if reference is not None:
            report.metrics.relative_l2 = relative_l2(result, reference)
        return FilterResult(result, report, selected)

    execute = run

    def render(
        self,
        variant: FilterVariant | str = FilterVariant.STRUCTURED,
        *,
        output: str = "spatial",
        reference: Any | None = None,
    ) -> tuple[Any, RenderReport]:
        result = self.run(variant, output=output, reference=reference)
        return result.values, result.report

    def apply(
        self,
        variant: FilterVariant | str = FilterVariant.STRUCTURED,
        *,
        output: str = "spatial",
        return_report: bool = False,
        reference: Any | None = None,
    ) -> Any:
        result = self.run(variant, output=output, reference=reference)
        return (result.values, result.report) if return_report else result.values


def build_filter_chain(source: Any, **options: Any) -> FilterChain:
    return FilterChain.from_source(source, **options)


__all__ = [
    "ExecutionVariant",
    "FilterChain",
    "FilterName",
    "FilterOperation",
    "FilterResult",
    "FilterVariant",
    "FusedTransfer",
    "PreparedCoordinateFilterChain",
    "PreparedFilterChain",
    "PreparedWordConvolutionFilterChain",
    "WordKernel",
    "build_filter_chain",
]


class PreparedFilterChain:
    """Dense complex128 reference, retained under the legacy STRUCTURED alias.

    It owns a sampled spectrum and a sampled diagonal. Formula fusion and
    preparation do not imply SPFBlock, TensorBlock, or SPFKernel execution.
    """

    def __init__(self, chain: FilterChain, output: str) -> None:
        if output not in {"spatial", "dense", "frequency", "spectrum", "spectrum_intensity", "intensity"}:
            raise ValueError(f"unsupported filter-chain output {output!r}")
        self.chain, self.output = chain, output
        self.spectrum = chain._source_values()
        if not chain.source_is_fourier:
            self.spectrum = fft.fftn(self.spectrum, workers=1)
        self.spectrum.setflags(write=False)
        fused = chain.fused_transfer()
        self.phase_eliminated = output in {"spectrum_intensity", "intensity"}
        self.transfer = np.exp2(2 * fused.log_magnitude) if self.phase_eliminated else fused.transfer.copy()
        self.transfer.setflags(write=False)
        size = math.prod(chain.shape)
        self.report = chain._report(
            FilterVariant.STRUCTURED,
            fft_count=0 if output in {"frequency", "spectrum", "spectrum_intensity", "intensity"} else 1,
            materialized=size,
            potential=size * max(1, chain.filter_count + 1),
            bytes_read=self.spectrum.nbytes + self.transfer.nbytes,
            bytes_written=size * (8 if self.phase_eliminated else 16),
            diagnostics={
                "structured": False,
                "structured_reference_dense": True,
                "reference_only": True,
                "structured_operator_formula": True,
                "source_carrier": "dense_complex128_spectrum",
                "operator_carrier": "dense_complex128_diagonal" if not self.phase_eliminated else "dense_float64_gain",
                "spf_block_tensor_matrix_used": False,
                "prepared": True,
                "preparation_fft_count": int(not chain.source_is_fourier),
                "cached_spatial_pixels": 0,
                "consumer": "spectral_diagonal",
                "terminal_phase_eliminated": self.phase_eliminated,
                "analytic_terms": tuple(op.name.value for op in chain.operations),
            },
        )
        self.report.metrics.representation = "dense_spectral_reference"

    def execute(self) -> FilterResult:
        if self.phase_eliminated:
            values = (self.spectrum.real * self.spectrum.real + self.spectrum.imag * self.spectrum.imag) * self.transfer
        else:
            filtered = self.spectrum * self.transfer
            values = filtered if self.output in {"frequency", "spectrum"} else fft.ifftn(filtered, workers=1)
        if self.output in {"spatial", "dense"} and np.all(np.abs(values.imag) <= 1e-12):
            values = values.real
        # Metrics may be annotated by a caller; do not share that mutable record.
        report = replace(self.report, metrics=replace(self.report.metrics))
        return FilterResult(values, report, FilterVariant.STRUCTURED)

    run = execute


class PreparedStructuredFilterChain:
    """Public block filter chain routed through the typed field compiler."""

    def __init__(
        self, chain: FilterChain, output: str, layout: Any = None, plan: Any = None, *, rejected: str = ""
    ) -> None:
        from spf.renderer.block_filter import FOURIER, FilteredBlockView
        from spf.renderer.model import CoordinateChart, Domain, Field, MaterializationRequest, SampleLayout
        from spf.renderer.transfers import SpectralTransfer

        if output not in {"spatial", "dense", "frequency", "spectrum", "spectrum_intensity", "intensity"}:
            raise ValueError(f"unsupported filter-chain output {output!r}")
        self.chain, self.output = chain, output
        self.plan = plan
        self.rejected_convolution = rejected
        domain = chain.metadata.get("source_domain", Domain.SPATIAL_2D)
        # Axis-order chart (design §17.1 convention): axis i carries spacing[i].
        # The former anti-diagonal basis encoded a row/column exchange inside the
        # chart; the exchange now happens once, in ``_axis_order_params``.
        chart = chain.metadata.get("source_chart") or CoordinateChart.identity(2, spacing=chain.sample_spacing)
        view = FilteredBlockView(chain.source, SpectralTransfer.identity(2), domain in FOURIER, chart)
        field = Field.from_value(view, domain=domain, chart=chart)
        field = replace(
            field,
            metadata={
                "sfr_terminal_spectral": output in {"frequency", "spectrum", "spectrum_intensity", "intensity"},
                "sfr_terminal_spatial": output in {"spatial", "dense"},
                "compute_dtype": np.dtype(chain.metadata.get("compute_dtype", "complex128")).name,
                "sfr_texture_block_ingress": bool(chain.metadata.get("texture_block_ingress")),
            },
        )
        field = replace(
            field,
            representation=replace(field.representation, error=chain.metadata.get("source_error", v02.ErrorLedger())),
        )
        for operation in chain.operations:
            op = chain._field_operation(operation)
            field = field.with_operation(op.op, **op.params)
        self.prepared = field.prepare(
            MaterializationRequest(
                layout=layout or SampleLayout(chain.shape),
                output_format="complex",
                precision=chain.precision,
                error_budget=chain.error_budget,
            )
        )
        self.terminal = self.prepared.terminal
        if self.terminal is None:
            raise RuntimeError("analytic block filter did not reach its typed terminal")
        self.phase_eliminated = output in {"spectrum_intensity", "intensity"}
        self.spectrum = None if self.terminal.periodic is None else self.terminal.periodic.spectrum
        self.transfer = self.terminal.view.transfer
        self._steady_report: RenderReport | None = None
        self._steady_real_output = False

    def execute(self) -> FilterResult:
        from spf.renderer import normalization, runtime

        started = time.perf_counter()
        spectral = self.output in {"frequency", "spectrum", "spectrum_intensity", "intensity"}
        steady_terminal = self.terminal.source_plan or (
            self.terminal.periodic
            if self.terminal.periodic is not None and self.terminal.periodic.hoisted
            else None
        )
        if self._steady_report is not None and steady_terminal is not None and not spectral:
            steady_dtype = np.dtype(
                getattr(
                    steady_terminal,
                    "dtype",
                    self.terminal.field.metadata.get("compute_dtype", "complex128"),
                )
            )
            values = np.asarray(steady_terminal.execute(), dtype=steady_dtype)
            if self._steady_real_output:
                values = values.real
            report = replace(self._steady_report, metrics=replace(self._steady_report.metrics))
            report = report.with_measured_cost((time.perf_counter() - started) * 1.0e3)
            return FilterResult(values, report, FilterVariant.STRUCTURED)
        ledger = runtime._apply_representation_error(
            self.terminal.field, normalization._field_ledger_after_operations(self.terminal.field)
        )
        if (self.terminal.periodic is not None or self.terminal.source_plan is not None) and spectral:
            executor = self.terminal.source_plan or self.terminal.periodic
            values = executor.execute(spectrum=True, spectrum_intensity=self.phase_eliminated)
            fft_count = 0
            # A forward, unnormalized DFT has maximum absolute gain N.
            # The field ledger already includes sqrt(N) for convolution.
            spectral_bound = any(
                op.name not in {FilterName.EXPOSURE, FilterName.PHASE}
                or (op.name is FilterName.PHASE and np.asarray(op.params["phase"]).ndim > 0)
                for op in self.chain.operations
            )
            factor = math.sqrt(math.prod(self.chain.shape)) if spectral_bound else math.prod(self.chain.shape)
            ledger = runtime._scale_ledger(ledger, factor)
        else:
            result, field_report = self.prepared.execute(return_report=True)
            ledger = field_report.ledger
            values = result.values
            fft_count = field_report.metrics.fft_count
            if self.terminal.view.frequency_domain and self.terminal.source_plan is None:
                if spectral:
                    if self.phase_eliminated:
                        values = np.abs(values) ** 2
                else:
                    values = fft.ifftn(values, workers=1)
                    fft_count = 1
        if self.phase_eliminated and ledger.total_bound:
            # |z|^2 error <= 2 |z| epsilon + epsilon^2, using represented output.
            ledger = runtime._scale_ledger(ledger, 2 * math.sqrt(runtime._max_abs(values)) + ledger.total_bound)
        runtime._enforce_error_budget(self.terminal.field, self.terminal.request, ledger, values, 0.0)
        real_output = self.output in {"spatial", "dense"} and np.all(np.abs(values.imag) <= 1e-12)
        if real_output:
            values = values.real
        size = values.size
        report = self.chain._report(
            FilterVariant.STRUCTURED,
            fft_count=fft_count,
            materialized=size,
            potential=size * max(1, self.chain.filter_count + 1),
            bytes_read=(
                self.spectrum.nbytes
                if self.spectrum is not None
                else self.terminal.source_plan.nbytes
                if self.terminal.source_plan is not None
                else self.terminal.words.nbytes
            )
            + self.transfer.nbytes,
            bytes_written=values.nbytes,
            ledger=ledger,
            diagnostics={
                "structured": True,
                "reference_only": False,
                "compiler_integrated": True,
                "spf_block_tensor_matrix_used": True,
                "source_carrier": type(self.chain.source).__name__,
                "operator_carrier": "GaussianChirpAtom + full log/phase jets + positive mixtures",
                "dense_transfer_elements": 0,
                "operator_bytes": self.transfer.nbytes,
                "sampled_phase_elements": sum(a.size for a in self.transfer.phase_residuals),
                "byte_traffic_status": "logical carrier I/O; excludes FFT traffic and arithmetic scratch",
                "memory_status": "logical output bytes only; peak resident memory not measured",
                "prepared": True,
                "cached_spatial_pixels": 0,
                "preparation_fft_count": (
                    self.terminal.source_plan.preparation_fft_count
                    if self.terminal.source_plan is not None
                    else 0
                    if self.terminal.periodic is None
                    else self.terminal.periodic.preparation_fft_count
                ),
                "source_plan": None if self.terminal.source_plan is None else self.terminal.source_plan.kind,
                "native_filter_consumer": bool(
                    getattr(self.terminal.source_plan, "native_consumer_used", False)
                )
                if self.terminal.source_plan is not None
                else bool(
                    self.terminal.periodic is not None and self.terminal.periodic.native.supported
                ),
                "native_filter_path": (
                    getattr(self.terminal.source_plan, "native_filter_path", None)
                    if self.terminal.source_plan is not None
                    else "PeriodicBlockExecution -> NativeTransfer.grid -> spf_sfr_filter_grid"
                    if self.terminal.periodic is not None and self.terminal.periodic.native.supported
                    else None
                ),
                "native_filter_abi": 3
                if (
                    getattr(self.terminal.source_plan, "native_consumer_used", False)
                    or (self.terminal.periodic is not None and self.terminal.periodic.native.supported)
                )
                else None,
                "source_mode_count": None
                if self.terminal.source_plan is None
                else self.terminal.source_plan.mode_count,
                "compute_dtype": (
                    np.dtype(self.terminal.field.metadata.get("compute_dtype", "complex128")).name
                    if self.terminal.source_plan is None
                    else np.dtype(
                        getattr(
                            self.terminal.source_plan,
                            "dtype",
                            self.terminal.field.metadata.get("compute_dtype", "complex128"),
                        )
                    ).name
                ),
                "consumer": "typed_block_analytic_filter",
                "terminal_phase_eliminated": self.phase_eliminated,
                "spectral_prefix_hoisted": self.terminal.periodic is not None and self.terminal.periodic.hoisted,
            },
        )
        report.metrics.representation = "block_analytic_transfer"
        report.metrics.backend = "cpu"
        report.metrics.precision = np.dtype(
            self.terminal.field.metadata.get("compute_dtype", "complex128")
        ).name
        report.metrics.resolution = values.shape
        report.metrics.materialized_elements = values.size
        report.metrics.terminal_output_elements = values.size
        report.metrics.terminal_output_bytes = values.nbytes
        report = replace(
            report,
            capabilities_used=(
                "addressed_block_consumer",
                "analytic_spectral_transfer",
                "typed_field_compiler",
                *(
                    ("native_packed_filter_consumer",)
                    if getattr(self.terminal.source_plan, "native_consumer_used", False)
                    else ()
                ),
            ),
        )
        report = self.chain._attach_plan(report, self.plan)
        if self.rejected_convolution:
            report = replace(
                report, diagnostics={**report.diagnostics, "convolution_rejected_at_prepare": self.rejected_convolution}
            )
        if steady_terminal is not None and not spectral:
            self._steady_report = report
            self._steady_real_output = bool(real_output)
        report = report.with_measured_cost((time.perf_counter() - started) * 1.0e3)
        return FilterResult(values, report, FilterVariant.STRUCTURED)

    run = execute


class PreparedCoordinateFilterChain:
    """Packed-block chain containing a §17.2/§17.3 coordinate remap.

    A remap is not a spectral diagonal, so the filtered-block terminal cannot
    absorb it.  The chain is prepared as a plain typed ``Field`` instead and the
    representation-aware runtime owns the remap: an exact index permutation of
    the packed word plane when the map carries the sample lattice into itself,
    the counted ``reference``-named resample otherwise.  Every diagonal
    operation before the remap is still absorbed analytically.
    """

    def __init__(self, chain: FilterChain, output: str, layout: Any = None) -> None:
        from spf.renderer.model import CoordinateChart, Domain, Field, MaterializationRequest, SampleLayout

        if output not in {"spatial", "dense"}:
            raise ValueError("a coordinate remap chain produces a spatial terminal only")
        self.chain, self.output = chain, output
        domain = chain.metadata.get("source_domain", Domain.SPATIAL_2D)
        chart = chain.metadata.get("source_chart") or CoordinateChart.identity(2, spacing=chain.sample_spacing)
        field = Field.from_value(chain.source, domain=domain, chart=chart)
        field = replace(
            field,
            representation=replace(field.representation, error=chain.metadata.get("source_error", v02.ErrorLedger())),
        )
        for operation in chain.operations:
            op = chain._field_operation(operation)
            field = field.with_operation(op.op, **op.params)
        self.field = field
        self.prepared = field.prepare(
            MaterializationRequest(
                layout=layout or SampleLayout(chain.shape),
                output_format="complex",
                precision=chain.precision,
                error_budget=chain.error_budget,
            )
        )

    def execute(self) -> FilterResult:
        started = time.perf_counter()
        result, field_report = self.prepared.execute(return_report=True)
        values = np.asarray(getattr(result, "values", result))
        if np.all(np.abs(values.imag) <= 1.0e-12):
            values = values.real
        size = int(values.size)
        report = self.chain._report(
            FilterVariant.STRUCTURED,
            fft_count=field_report.metrics.fft_count,
            materialized=size,
            potential=size * max(1, self.chain.filter_count + 1),
            bytes_read=size * 4,
            bytes_written=int(values.nbytes),
            fallback_conversions=int(field_report.metrics.fallback_conversions),
            ledger=field_report.ledger,
            diagnostics={
                **field_report.diagnostics,
                "structured": True,
                "reference_only": False,
                "compiler_integrated": True,
                "source_carrier": type(self.chain.source).__name__,
                "operator_carrier": "coordinate remap on the typed field path",
                "dense_transfer_elements": 0,
                "consumer": "coordinate_remap_field_path",
                "prepared": True,
                "coordinate_operations": tuple(
                    operation.name.value for operation in self.chain.operations if operation.name in _COORDINATE_FILTERS
                ),
            },
        )
        report.metrics.representation = "coordinate_remap_field_path"
        report.metrics.backend = "cpu"
        report.metrics.resolution = values.shape
        report.metrics.materialized_elements = size
        report.metrics.terminal_output_elements = size
        report.metrics.terminal_output_bytes = int(values.nbytes)
        report = replace(
            report,
            passes=(*report.passes, *field_report.passes),
            fallbacks=(*report.fallbacks, *field_report.fallbacks),
            capabilities_used=tuple(dict.fromkeys((*report.capabilities_used, *field_report.capabilities_used))),
        )
        report = report.with_measured_cost((time.perf_counter() - started) * 1.0e3)
        return FilterResult(values, report, FilterVariant.STRUCTURED)

    run = execute


class PreparedWordConvolutionFilterChain:
    """The ``conv(F, h)`` equivalent: a compact kernel applied to the source word plane.

    Chosen by :func:`spf.renderer.compiler.saturate_filter_chain` when the
    backend cost model prices it below ``ifft(mul(fft(F), H))`` and the
    kernel's certified bound fits the chain's relative error budget. Every
    arithmetic step runs on packed SPF32 words: tap products through
    ``spf.tensor.core.spf_mul_words`` and the tap sum through
    ``spf.tensor.reducers.reduce_axis``. The output is decoded once at the
    terminal boundary (counted as materialization, not as a fallback).
    """

    ROWS = 64

    def __init__(self, chain: FilterChain, output: str, plan: Any) -> None:
        if output not in {"spatial", "dense"}:
            raise ValueError("word convolution produces a spatial terminal only")
        kernel = plan.kernel
        if kernel is None:
            # Explicit construction (a caller forcing the conv form): derive
            # the whole-chain kernel; it must exist and fit the budget.
            kernel, reason = chain.word_kernel(range(len(chain.operations)))
            if kernel is None:
                raise ValueError(f"no compact kernel was derived for this chain: {reason}")
            if not chain.word_kernel_legal(kernel):
                raise ValueError(chain.word_kernel_reason(kernel))
        self.chain, self.output, self.plan = chain, output, plan
        self.kernel: WordKernel = kernel
        view = chain.source
        count = int(math.prod(chain.shape))
        linear = np.arange(count, dtype=np.int64)
        words = view.project_words(view.sample_words(linear, phase=view.observable != "intensity"), linear)
        if not isinstance(words, np.ndarray) or words.dtype != np.uint32:
            raise ValueError("source carrier does not project to an SPF32 word plane (wide header offsets)")
        self.words = np.ascontiguousarray(words.reshape(chain.shape))
        self.words.setflags(write=False)
        self.source_ledger = chain.metadata.get("source_error", v02.ErrorLedger())

    def _rows(self, start: int, stop: int) -> np.ndarray:
        from spf.tensor import reducers
        from spf.tensor.core import ReducerPolicy, ReducerSpec, spf_mul_words

        height, width = self.chain.shape
        rows = np.arange(start, stop, dtype=np.int64)[:, None]
        cols = np.arange(width, dtype=np.int64)[None, :]
        stack = np.empty((self.kernel.taps, stop - start, width), dtype=np.uint32)
        for k, (dy, dx) in enumerate(self.kernel.offsets):
            shifted = self.words[(rows - int(dy)) % height, (cols - int(dx)) % width]
            stack[k] = spf_mul_words(shifted, np.full(shifted.shape, self.kernel.words[k], dtype=np.uint32))
        if self.kernel.taps == 1:
            return np.asarray(stack[0], dtype=np.uint32)
        spec = ReducerSpec(policy=ReducerPolicy.CARTESIAN_KAHAN)
        return np.asarray(reducers.reduce_axis(stack, spec, axis=0), dtype=np.uint32)

    def execute(self) -> FilterResult:
        from spf.renderer import runtime

        started = time.perf_counter()
        height, width = self.chain.shape
        out_words = np.empty((height, width), dtype=np.uint32)
        for start in range(0, height, self.ROWS):
            stop = min(height, start + self.ROWS)
            out_words[start:stop] = self._rows(start, stop)
        values = np.asarray(decode_words(out_words))
        if np.all(np.abs(values.imag) <= 1.0e-12):
            values = values.real
        size = values.size
        scale = float(runtime._max_abs(np.asarray(decode_words(self.words))))
        ledger = runtime._ledger_add(
            self.kernel.ledger(scale),
            source_quantization=self.source_ledger.source_quantization,
            representation_fit=self.source_ledger.representation_fit,
        )
        report = self.chain._report(
            FilterVariant.STRUCTURED,
            fft_count=0,
            materialized=size,
            potential=size * max(1, self.chain.filter_count + 1),
            bytes_read=self.words.nbytes + self.kernel.words.nbytes,
            bytes_written=out_words.nbytes,
            ledger=ledger,
            diagnostics={
                "structured": True,
                "reference_only": False,
                "compiler_integrated": True,
                "spf_block_tensor_matrix_used": True,
                "source_carrier": type(self.chain.source).__name__,
                "operator_carrier": f"compact word kernel ({self.kernel.taps} taps)",
                "dense_transfer_elements": 0,
                "operator_bytes": int(self.kernel.words.nbytes),
                "word_products": int(self.kernel.taps * size),
                "word_reductions": int(size if self.kernel.taps > 1 else 0),
                "consumer": "word_convolution",
                "prepared": True,
                "preparation_fft_count": 0,
                "cached_spatial_pixels": 0,
                "terminal_phase_eliminated": False,
                "kernel_relative_bound": self.kernel.relative_bound,
                "kernel_operations": self.kernel.operations,
            },
        )
        report.metrics.representation = "block_word_convolution"
        report.metrics.backend = "cpu"
        report.metrics.resolution = values.shape
        report.metrics.materialized_elements = size
        report.metrics.terminal_output_elements = size
        report.metrics.terminal_output_bytes = values.nbytes
        report.metrics.reduction_operations = int(size if self.kernel.taps > 1 else 0)
        report.metrics.field_operations = int(self.kernel.taps * size)
        report = replace(
            report,
            capabilities_used=("addressed_block_consumer", "word_convolution", "equality_saturation"),
        )
        report = self.chain._attach_plan(report, self.plan)
        report = report.with_measured_cost((time.perf_counter() - started) * 1.0e3)
        return FilterResult(values, report, FilterVariant.STRUCTURED, words=out_words)

    run = execute
