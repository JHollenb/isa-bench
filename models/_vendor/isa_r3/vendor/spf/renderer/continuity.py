"""Producer-to-consumer SPF32 format continuity.

This module models the path that the renderer ultimately wants: a producer
emits canonical packed SPF32 words directly from native log/phase state and
compatible consumers, including the two-dimensional inverse FFT, keep those
words packed.  Magnitude and intensity terminals project the packed log field
directly and never construct a Cartesian complex array.  Legacy complex inputs
and an explicitly Cartesian spatial output remain visibly named boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.backends import get_backend
from spf.renderer.carriers import dft_frequencies
from spf.renderer.report import CompilerPassRecord, RenderMetrics, RenderReport, RepresentationDecision
from spf.tensor import v02

_TWO_PI = 2.0 * math.pi
_MIN_FINITE_LOG2 = v02.SPF32_L_MIN_FINITE / v02.SPF32_L_SCALE
_MAX_FINITE_LOG2 = v02.SPF32_L_MAX_FINITE / v02.SPF32_L_SCALE


def _readonly(array: Any, *, dtype: Any, ndim: int = 2) -> np.ndarray:
    value = np.asarray(array, dtype=dtype)
    if value.ndim != ndim:
        raise ValueError(f"packed spectral planes must be {ndim}-dimensional")
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _broadcast_mask(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    return np.broadcast_to(np.asarray(value, dtype=bool), shape)


def _quantize_finite_log2(values: Any) -> np.ndarray:
    """RNE-quantize log2 values without ever emitting a reserved log code."""

    clipped = np.clip(np.asarray(values, dtype=np.float64), _MIN_FINITE_LOG2, _MAX_FINITE_LOG2)
    return np.asarray(np.rint(clipped * v02.SPF32_L_SCALE), dtype=np.int64)


def pack_log_phase_codes(
    log_codes: Any,
    phase_codes: Any,
    *,
    zero: Any = False,
    nan: Any = False,
) -> np.ndarray:
    """Vector-pack canonical SPF32 words without a Cartesian encode step."""

    logs, phases = np.broadcast_arrays(np.asarray(log_codes, dtype=np.int64), np.asarray(phase_codes, dtype=np.int64))
    supplied_zero = _broadcast_mask(zero, logs.shape)
    supplied_nan = _broadcast_mask(nan, logs.shape)
    if np.any(supplied_zero & supplied_nan):
        raise ValueError("an SPF32 element cannot be both zero and NaN")
    wrapped_phases = phases & v02.SPF32_PHASE_MASK
    zero_mask = supplied_zero | (logs == v02.SPF32_L_ZERO)
    nan_mask = supplied_nan | ((logs == v02.SPF32_NAN_L) & (wrapped_phases == v02.SPF32_NAN_T))
    finite_logs = np.clip(logs, v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE)
    words = ((finite_logs & 0xFFFF) << v02.SPF32_PHASE_BITS) | wrapped_phases
    words = np.where(zero_mask, v02.SPF32_WORD_ZERO, words)
    words = np.where(nan_mask, v02.SPF32_WORD_NAN, words)
    return np.asarray(words, dtype=np.uint32)


def unpack_log_phase_codes(words: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical log, phase, zero, and NaN planes."""

    packed = np.asarray(words, dtype=np.uint32)
    logs = np.asarray(v02.unpack_L(packed), dtype=np.int64)
    phases = np.asarray(v02.unpack_T(packed), dtype=np.int64)
    nan = (logs == v02.SPF32_NAN_L) & (phases == v02.SPF32_NAN_T)
    reserved = (logs == v02.SPF32_NAN_L) & ~nan
    logs = np.where(reserved, v02.SPF32_L_MAX_FINITE, logs)
    zero = logs == v02.SPF32_L_ZERO
    phases = np.where(zero, v02.SPF32_T_ZERO, phases)
    return logs, phases, zero, nan


def validate_packed_spectrum(words: Any) -> dict[str, Any]:
    """Validate an untrusted packed plane and return explicit diagnostics.

    This is deliberately separate from trusted resident construction.  It is
    the full-field scan an application should run at an import/security
    boundary, not on every steady-state renderer invocation.
    """

    raw = np.asarray(words)
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError("packed spectrum words must use an integer dtype")
    if raw.ndim != 2:
        raise ValueError("packed spectrum words must be two-dimensional")
    if raw.size:
        if raw.dtype.kind == "i" and np.any(raw < 0):
            raise ValueError("packed spectrum contains negative words")
        if raw.dtype.itemsize > np.dtype(np.uint32).itemsize and np.any(raw > 0xFFFFFFFF):
            raise ValueError("packed spectrum contains words above uint32 range")
    packed = np.asarray(raw, dtype=np.uint32)
    logs, phases, zero, nan = unpack_log_phase_codes(packed)
    canonical = pack_log_phase_codes(logs, phases, zero=zero, nan=nan)
    return {
        "shape": tuple(int(value) for value in packed.shape),
        "dtype": str(raw.dtype),
        "canonical": bool(np.array_equal(packed, canonical)),
        "noncanonical_elements": int(np.count_nonzero(packed != canonical)),
        "zero_elements": int(np.count_nonzero(zero)),
        "nan_elements": int(np.count_nonzero(nan)),
        "finite_elements": int(packed.size - np.count_nonzero(zero | nan)),
    }


def decode_packed_complex(words: Any, *, dtype: Any = np.complex64) -> np.ndarray:
    """Vectorized terminal SPF32-to-Cartesian conversion through the library codec.

    The renderer keeps no decoder of its own (AGENTS.md rule 3): this is
    :func:`spf.renderer.carriers.decode_words` (``spf.tensor.v02.decode_complex``)
    with the terminal's requested dtype applied at the output boundary.
    """

    from spf.renderer.carriers import decode_words

    return np.asarray(decode_words(np.asarray(words, dtype=np.uint32)), dtype=dtype)


def packed_ifft2_stages(
    words: Any,
    *,
    precision: str = "accurate",
    chart: str = "r15",
    implementation: str = "v02",
    validate: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the packed result after each separable inverse-FFT axis.

    ``v02`` (and the compatibility spelling ``python``) selects the normative
    FB3-Hybrid profile. ``legacy_logchart`` selects the experimental native
    R1/R15 chart whose out-of-chart fallback is centered FB7; the old
    ``native`` spelling remains an explicit compatibility alias. ``auto`` now
    resolves to ``v02`` rather than silently changing the arithmetic profile
    when a native library happens to be installed. Both axes remain canonical
    ``uint32`` SPF32 from ingress through the returned spatial field.
    """

    raw = np.asarray(words)
    if raw.dtype.kind not in {"i", "u"}:
        raise TypeError("packed_ifft2 requires an integer array of packed SPF32 words")
    if raw.size:
        if raw.dtype.kind == "i" and np.any(raw < 0):
            raise ValueError("packed_ifft2 words must be in the uint32 range")
        if raw.dtype.itemsize > np.dtype(np.uint32).itemsize and np.any(raw > 0xFFFFFFFF):
            raise ValueError("packed_ifft2 words must be in the uint32 range")
    packed = np.asarray(raw, dtype=np.uint32)
    if packed.ndim != 2:
        raise ValueError("packed_ifft2 requires a two-dimensional packed field")
    if any(size < 2 or (size & (size - 1)) != 0 for size in packed.shape):
        raise ValueError("packed_ifft2 requires power-of-two dimensions of at least two")
    selected = str(implementation).strip().lower().replace("-", "_")
    aliases = {
        "auto": "v02",
        "python": "v02",
        "fb3": "v02",
        "fb3_hybrid": "v02",
        "native": "legacy_logchart",
        "native_logchart": "legacy_logchart",
    }
    selected = aliases.get(selected, selected)
    if selected not in {"v02", "legacy_logchart"}:
        raise ValueError("packed_ifft2 implementation must be 'v02' or 'legacy_logchart'")
    if validate:
        logs, phases, zero, nan = unpack_log_phase_codes(packed)
        canonical = pack_log_phase_codes(logs, phases, zero=zero, nan=nan)
        if not np.array_equal(packed, canonical):
            raise ValueError("packed_ifft2 requires canonical SPF32 words")

    use_native = selected == "legacy_logchart"
    if use_native:
        from spf.native import logchart

        if not logchart.is_available():
            raise RuntimeError("legacy native packed log-chart IFFT is unavailable")
    if use_native:
        first = logchart.ifft_rows(
            np.ascontiguousarray(packed),
            mode=precision,
            chart=chart,
        )
        second = logchart.ifft_rows(
            np.ascontiguousarray(first.T),
            mode=precision,
            chart=chart,
        )
        return np.asarray(first, dtype=np.uint32), np.ascontiguousarray(second.T, dtype=np.uint32)

    first = np.asarray(v02.spf_ifft_radix2(packed, precision=precision), dtype=np.uint32)
    second = v02.spf_ifft_radix2(np.ascontiguousarray(first.T), precision=precision)
    return first, np.ascontiguousarray(np.asarray(second, dtype=np.uint32).T)


def packed_ifft2(
    words: Any,
    *,
    precision: str = "accurate",
    chart: str = "r15",
    implementation: str = "v02",
    validate: bool = True,
) -> np.ndarray:
    """Run both separable inverse-FFT axes and return packed spatial SPF32."""

    _, spatial = packed_ifft2_stages(
        words,
        precision=precision,
        chart=chart,
        implementation=implementation,
        validate=validate,
    )
    return spatial


def project_packed_magnitude(
    words: Any,
    *,
    squared: bool = False,
    dtype: Any = np.float32,
) -> np.ndarray:
    """Project packed spatial SPF32 to magnitude without Cartesian decoding.

    Phase is intentionally never read.  The only nonlinear terminal operation
    is ``exp2(L / 256)`` (or ``exp2(2L / 256)`` for intensity), which is the
    scalar quantity requested by the display consumer rather than an
    SPF-to-complex format conversion.
    """

    from spf.renderer.carriers import decode_magnitude

    return np.asarray(decode_magnitude(words, squared=bool(squared)), dtype=dtype)


def encode_complex_adapter(values: Any, *, provenance: str = "legacy_complex_adapter") -> PackedSpectrum2D:
    """Explicit legacy ingress adapter; native producers should not call it."""

    source = np.asarray(values, dtype=np.complex128)
    if source.ndim != 2:
        raise ValueError("legacy complex spectral inputs must be two-dimensional")
    magnitude = np.abs(source)
    zero = magnitude == 0.0
    nan = np.isnan(source.real) | np.isnan(source.imag)
    safe = np.where(zero | nan, 1.0, magnitude)
    ideal_logs = np.log2(safe)
    logs = _quantize_finite_log2(ideal_logs)
    phase_radians = np.where(nan, 0.0, np.angle(source))
    phases = np.rint(np.mod(phase_radians, _TWO_PI) * (v02.SPF32_PHASE_MOD / _TWO_PI)).astype(np.int64)
    active = ~(zero | nan)
    saturated = active & ((ideal_logs < _MIN_FINITE_LOG2) | (ideal_logs > _MAX_FINITE_LOG2))
    magnitude_upper_bound = math.inf if np.any(nan) else float(np.max(magnitude, initial=0.0))
    return PackedSpectrum2D(
        pack_log_phase_codes(logs, phases, zero=zero, nan=nan),
        provenance=(provenance,),
        ingress_conversions=1,
        saturated_elements=int(np.count_nonzero(saturated)),
        magnitude_upper_bound=magnitude_upper_bound,
    )


def spf_product_relative_error_bound(factor_count: int = 1) -> float:
    """Worst relative complex error from ``factor_count`` RNE inputs."""

    if factor_count < 0:
        raise ValueError("factor_count must be non-negative")
    delta_log2 = factor_count * 0.5 / v02.SPF32_L_SCALE
    delta_phase = factor_count * math.pi / v02.SPF32_PHASE_MOD
    radius = math.exp2(delta_log2)
    return math.sqrt(max(0.0, radius * radius + 1.0 - 2.0 * radius * math.cos(delta_phase)))


@dataclass(frozen=True, slots=True)
class PackedSpectrum2D:
    """Immutable canonical SPF32 spectrum kept packed between consumers."""

    words: np.ndarray
    sample_spacing: tuple[float, float] = (1.0, 1.0)
    provenance: tuple[str, ...] = ("native_spf32",)
    ingress_conversions: int = 0
    packed_operations: int = 0
    quantized_factors: int = 1
    saturated_elements: int | None = None
    magnitude_upper_bound: float = math.inf
    _construction_mode: str = "checked_copy"
    _validate_canonical: bool = True

    def __post_init__(self) -> None:
        mode = str(self._construction_mode).strip().lower().replace("-", "_")
        if mode == "checked_copy":
            diagnostics = validate_packed_spectrum(self.words)
            words = np.asarray(self.words, dtype=np.uint32)
            logs, phases, zero, nan = unpack_log_phase_codes(words)
            packed = _readonly(pack_log_phase_codes(logs, phases, zero=zero, nan=nan), dtype=np.uint32)
            canonical_validated = bool(diagnostics["canonical"])
        elif mode in {"trusted_owned", "trusted_borrowed"}:
            raw = np.asarray(self.words)
            if raw.dtype != np.uint32 or raw.ndim != 2 or not raw.flags.c_contiguous:
                raise TypeError("trusted packed spectra require a C-contiguous two-dimensional uint32 array")
            if mode == "trusted_borrowed" and raw.flags.writeable:
                raise ValueError("a borrowed trusted packed spectrum must already be read-only")
            if self._validate_canonical:
                diagnostics = validate_packed_spectrum(raw)
                if not diagnostics["canonical"]:
                    raise ValueError("trusted packed spectrum contains noncanonical SPF32 words")
                canonical_validated = True
            else:
                canonical_validated = False
            packed = raw
            if mode == "trusted_owned" and packed.flags.writeable:
                packed.setflags(write=False)
        else:
            raise ValueError("unknown packed spectrum construction mode")
        if len(self.sample_spacing) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.sample_spacing
        ):
            raise ValueError("sample spacing must contain two positive finite values")
        object.__setattr__(self, "words", packed)
        object.__setattr__(self, "sample_spacing", tuple(float(value) for value in self.sample_spacing))
        object.__setattr__(self, "provenance", tuple(str(value) for value in self.provenance))
        object.__setattr__(self, "ingress_conversions", int(self.ingress_conversions))
        object.__setattr__(self, "packed_operations", int(self.packed_operations))
        object.__setattr__(self, "quantized_factors", int(self.quantized_factors))
        saturated = None if self.saturated_elements is None else int(self.saturated_elements)
        object.__setattr__(self, "saturated_elements", saturated)
        upper_bound = float(self.magnitude_upper_bound)
        object.__setattr__(self, "magnitude_upper_bound", upper_bound)
        object.__setattr__(self, "_construction_mode", mode)
        object.__setattr__(self, "_validate_canonical", canonical_validated)
        if (
            self.ingress_conversions < 0
            or self.packed_operations < 0
            or self.quantized_factors < 0
            or (saturated is not None and saturated < 0)
        ):
            raise ValueError("conversion, operation, factor, and saturation counts must be non-negative")
        if math.isnan(upper_bound) or upper_bound < 0.0:
            raise ValueError("magnitude upper bound must be non-negative")

    @classmethod
    def from_checked_words(cls, words: Any, **metadata: Any) -> PackedSpectrum2D:
        """Canonicalize and copy words received at an untrusted boundary."""

        return cls(words, _construction_mode="checked_copy", _validate_canonical=True, **metadata)

    @classmethod
    def from_canonical_words(
        cls,
        words: Any,
        *,
        validate: bool = False,
        copy: bool = False,
        ownership: str = "borrowed",
        **metadata: Any,
    ) -> PackedSpectrum2D:
        """Adopt already-canonical resident SPF32 memory without rescanning it.

        ``ownership='borrowed'`` requires a read-only array.  With
        ``ownership='owned'`` and ``copy=False`` ownership is transferred and
        the array is frozen in place.  ``validate=True`` opts back into the
        explicit full-field canonicality scan.
        """

        selected = str(ownership).strip().lower()
        if selected not in {"borrowed", "owned"}:
            raise ValueError("ownership must be 'borrowed' or 'owned'")
        raw = np.asarray(words)
        if copy:
            raw = np.array(raw, dtype=np.uint32, order="C", copy=True)
            selected = "owned"
        return cls(
            raw,
            _construction_mode=f"trusted_{selected}",
            _validate_canonical=bool(validate),
            **metadata,
        )

    @property
    def construction_mode(self) -> str:
        return self._construction_mode

    @property
    def canonical_validated(self) -> bool:
        return self._validate_canonical

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.words.shape[0]), int(self.words.shape[1])

    @property
    def size(self) -> int:
        return int(self.words.size)

    @property
    def nbytes(self) -> int:
        return int(self.words.nbytes)

    def multiply(self, other: PackedSpectrum2D, *, backend: str = "cpu") -> PackedSpectrum2D:
        if self.shape != other.shape:
            raise ValueError("packed spectra must have identical shapes")
        if self.sample_spacing != other.sample_spacing:
            raise ValueError("packed spectra must have identical sample spacing")
        left_logs, _, left_zero, left_nan = unpack_log_phase_codes(self.words)
        right_logs, _, right_zero, right_nan = unpack_log_phase_codes(other.words)
        active = ~(left_zero | left_nan | right_zero | right_nan)
        raw_logs = left_logs + right_logs
        operation_saturation = int(
            np.count_nonzero(active & ((raw_logs < v02.SPF32_L_MIN_FINITE) | (raw_logs > v02.SPF32_L_MAX_FINITE)))
        )
        if self.saturated_elements is None or other.saturated_elements is None:
            saturated_elements = None
        else:
            saturated_elements = self.saturated_elements + other.saturated_elements + operation_saturation
        result = np.asarray(get_backend(backend).spf_mul(self.words, other.words), dtype=np.uint32)
        if self.magnitude_upper_bound == 0.0 or other.magnitude_upper_bound == 0.0:
            magnitude_upper_bound = 0.0
        else:
            magnitude_upper_bound = self.magnitude_upper_bound * other.magnitude_upper_bound
        return PackedSpectrum2D.from_canonical_words(
            result,
            ownership="owned",
            sample_spacing=self.sample_spacing,
            provenance=(*self.provenance, *other.provenance, "spf_mul"),
            ingress_conversions=self.ingress_conversions + other.ingress_conversions,
            packed_operations=self.packed_operations + other.packed_operations + 1,
            quantized_factors=self.quantized_factors + other.quantized_factors,
            saturated_elements=saturated_elements,
            magnitude_upper_bound=magnitude_upper_bound,
        )

    def decode_terminal(self, *, dtype: Any = np.complex64) -> np.ndarray:
        return decode_packed_complex(self.words, dtype=dtype)


@dataclass(frozen=True, slots=True)
class NativeLogPhaseProducer2D:
    """Analytic producer whose native state is log magnitude plus phase."""

    log2_magnitude: np.ndarray
    phase_radians: np.ndarray
    zero: np.ndarray | bool = False
    name: str = "native_log_phase"
    sample_spacing: tuple[float, float] = (1.0, 1.0)

    def __post_init__(self) -> None:
        logs, phases = np.broadcast_arrays(
            np.asarray(self.log2_magnitude, dtype=np.float64),
            np.asarray(self.phase_radians, dtype=np.float64),
        )
        if logs.ndim != 2:
            raise ValueError("native log/phase producer planes must be two-dimensional")
        zero = _broadcast_mask(self.zero, logs.shape)
        if np.any(np.isnan(logs)) or np.any(np.isinf(logs) & ~zero) or np.any(~np.isfinite(phases)):
            raise ValueError("native producer log/phase planes must not contain NaN or infinite phase")
        logs_copy = np.array(logs, copy=True)
        phases_copy = np.array(phases, copy=True)
        zero_copy = np.array(zero, copy=True)
        logs_copy.setflags(write=False)
        phases_copy.setflags(write=False)
        zero_copy.setflags(write=False)
        if len(self.sample_spacing) != 2 or any(
            not math.isfinite(value) or value <= 0.0 for value in self.sample_spacing
        ):
            raise ValueError("sample spacing must contain two positive finite values")
        object.__setattr__(self, "log2_magnitude", logs_copy)
        object.__setattr__(self, "phase_radians", phases_copy)
        object.__setattr__(self, "zero", zero_copy)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "sample_spacing", tuple(float(value) for value in self.sample_spacing))

    @classmethod
    def gaussian_spectrum(
        cls,
        shape: tuple[int, int],
        *,
        center: tuple[float, float],
        covariance: Any,
        amplitude: float = 1.0,
        sample_spacing: tuple[float, float] = (1.0, 1.0),
        name: str = "analytic_gaussian",
    ) -> NativeLogPhaseProducer2D:
        if len(shape) != 2 or any(int(value) <= 0 for value in shape):
            raise ValueError("shape must contain two positive dimensions")
        covariance_array = np.asarray(covariance, dtype=np.float64)
        if covariance_array.shape != (2, 2) or not np.allclose(covariance_array, covariance_array.T):
            raise ValueError("covariance must be a symmetric 2x2 matrix")
        if np.min(np.linalg.eigvalsh(covariance_array)) < 0.0:
            raise ValueError("covariance must be positive semidefinite")
        if not math.isfinite(amplitude) or amplitude <= 0.0:
            raise ValueError("amplitude must be positive and finite")
        fy = dft_frequencies(int(shape[0]), float(sample_spacing[1]))
        fx = dft_frequencies(int(shape[1]), float(sample_spacing[0]))
        ky, kx = np.meshgrid(fy, fx, indexing="ij")
        quadratic = (
            covariance_array[0, 0] * kx * kx + 2.0 * covariance_array[0, 1] * kx * ky + covariance_array[1, 1] * ky * ky
        )
        log2_magnitude = math.log2(amplitude) - (2.0 * math.pi**2 / math.log(2.0)) * quadratic
        phase = -_TWO_PI * (kx * float(center[0]) + ky * float(center[1]))
        return cls(log2_magnitude, phase, name=name, sample_spacing=sample_spacing)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.log2_magnitude.shape[0]), int(self.log2_magnitude.shape[1])

    def produce(self, *, collect_diagnostics: bool = True) -> PackedSpectrum2D:
        """Emit packed SPF32 directly; no complex array is constructed."""

        safe_logs = np.where(np.asarray(self.zero), 0.0, self.log2_magnitude)
        log_codes = _quantize_finite_log2(safe_logs)
        phase_codes = np.rint(np.mod(self.phase_radians, _TWO_PI) * (v02.SPF32_PHASE_MOD / _TWO_PI)).astype(np.int64)
        diagnostics = self.quantization_diagnostics() if collect_diagnostics else None
        if diagnostics is None:
            saturated_elements = None
            magnitude_upper_bound = math.inf
        else:
            active_logs = self.log2_magnitude[~np.asarray(self.zero)]
            saturated_elements = diagnostics["low_saturation_elements"] + diagnostics["high_saturation_elements"]
            if active_logs.size == 0:
                magnitude_upper_bound = 0.0
            elif diagnostics["high_saturation_elements"]:
                magnitude_upper_bound = math.inf
            else:
                magnitude_upper_bound = 2.0 ** float(np.max(active_logs, initial=-math.inf))
        words = pack_log_phase_codes(log_codes, phase_codes, zero=self.zero)
        return PackedSpectrum2D.from_canonical_words(
            words,
            ownership="owned",
            sample_spacing=self.sample_spacing,
            provenance=(f"producer:{self.name}",),
            ingress_conversions=0,
            saturated_elements=saturated_elements,
            magnitude_upper_bound=magnitude_upper_bound,
        )

    def quantization_diagnostics(self) -> dict[str, int]:
        """Count ideal values that saturate at the finite SPF32 boundaries."""

        active = ~np.asarray(self.zero)
        return {
            "finite_elements": int(np.count_nonzero(active)),
            "low_saturation_elements": int(np.count_nonzero(active & (self.log2_magnitude < _MIN_FINITE_LOG2))),
            "high_saturation_elements": int(np.count_nonzero(active & (self.log2_magnitude > _MAX_FINITE_LOG2))),
        }

    def complex_reference(self, *, dtype: Any = np.complex64) -> np.ndarray:
        """Construct the Cartesian control from the same ideal semantics."""

        magnitude = np.exp2(self.log2_magnitude)
        values = magnitude * (np.cos(self.phase_radians) + 1j * np.sin(self.phase_radians))
        values = np.where(self.zero, 0.0 + 0.0j, values)
        return np.asarray(values, dtype=dtype)


@dataclass(frozen=True, slots=True)
class PackedMaterialization:
    values: np.ndarray
    packed: PackedSpectrum2D
    report: RenderReport


@dataclass(frozen=True, slots=True)
class PackedConsumerPipeline2D:
    """Packed producer-to-consumer path with an SPF-native spatial terminal."""

    source: PackedSpectrum2D
    transfer: PackedSpectrum2D
    backend: str = "cpu"
    metadata: Any = None
    ifft_implementation: str = "v02"

    def __post_init__(self) -> None:
        if self.source.shape != self.transfer.shape:
            raise ValueError("source and transfer spectra must have identical shapes")
        selected = str(self.ifft_implementation).strip().lower().replace("-", "_")
        aliases = {
            "auto": "v02",
            "python": "v02",
            "fb3": "v02",
            "fb3_hybrid": "v02",
            "native": "legacy_logchart",
            "native_logchart": "legacy_logchart",
        }
        selected = aliases.get(selected, selected)
        if selected not in {"v02", "legacy_logchart"}:
            raise ValueError("ifft_implementation must be 'v02' or 'legacy_logchart'")
        object.__setattr__(self, "ifft_implementation", selected)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata or {})))

    def execute_packed(self) -> PackedSpectrum2D:
        return self.source.multiply(self.transfer, backend=self.backend)

    def render(
        self,
        *,
        resolution: tuple[int, int] | None = None,
        **_: Any,
    ) -> tuple[np.ndarray, RenderReport]:
        """Render as a high-level Scene field component."""

        if resolution is not None:
            width, height = (int(value) for value in resolution)
            if self.source.shape != (height, width):
                raise ValueError(
                    f"packed spectrum shape {self.source.shape} does not match render resolution {(width, height)}"
                )
        materialized = self.materialize(output="magnitude")
        return materialized.values, materialized.report

    def materialize(self, *, output: str) -> PackedMaterialization:
        name = str(output).strip().lower().replace("-", "_")
        if name not in {
            "packed_frequency",
            "packed_spatial",
            "magnitude",
            "intensity",
            "cartesian_complex_debug",
        }:
            raise ValueError(f"unsupported packed terminal output {output!r}")
        packed = self.execute_packed()
        size = packed.size
        terminal_conversions = 0
        scalar_projections = 0
        fft_count = 0
        if name == "packed_frequency":
            values = packed.words
        elif name == "cartesian_complex_debug":
            terminal_conversions = 1
            complex_spectrum = packed.decode_terminal(dtype=np.complex64)
            spatial = np.asarray(np.fft.ifft2(complex_spectrum), dtype=np.complex64)
            fft_count = 1
            values = spatial
        else:
            spatial_words = packed_ifft2(
                packed.words,
                implementation=self.ifft_implementation,
                validate=False,
            )
            fft_count = 1
            if name == "packed_spatial":
                values = spatial_words
            else:
                scalar_projections = 1
                values = project_packed_magnitude(spatial_words, squared=name == "intensity")
        metrics = RenderMetrics(
            backend=self.backend,
            workload="producer_consumer_continuity",
            resolution=packed.shape,
            representation="packed_spf32_continuous",
            precision="spf32_q7.8_u16",
            bytes_read=size * 12,
            bytes_written=size * (4 + (8 if terminal_conversions else 4 if fft_count else 0)),
            peak_memory=size * (12 if terminal_conversions else 8 if fft_count else 4),
            potential_elements=size,
            materialized_elements=size,
            fft_count=fft_count,
            fallback_conversions=packed.ingress_conversions,
            field_operations=packed.packed_operations,
        )
        relative_bound = spf_product_relative_error_bound(packed.quantized_factors)
        native_transform = fft_count > 0 and terminal_conversions == 0
        certificate_valid = (
            packed.saturated_elements == 0 and math.isfinite(packed.magnitude_upper_bound) and not native_transform
        )
        complex_absolute_bound = relative_bound * packed.magnitude_upper_bound if certificate_valid else math.inf
        absolute_bound = complex_absolute_bound
        if name == "intensity" and certificate_valid:
            absolute_bound = complex_absolute_bound * (2.0 * packed.magnitude_upper_bound + complex_absolute_bound)
        ledger = v02.ErrorLedger(source_quantization=absolute_bound)
        report = RenderReport(
            metrics,
            ledger,
            passes=(CompilerPassRecord("preserve_packed_spf32", 2, 1, rewrites=("direct_log_phase_pack",)),),
            decisions=(
                RepresentationDecision(
                    field_id="producer_consumer_spectrum",
                    region=str(packed.shape),
                    selected="packed_spf32_continuous",
                    candidates=("complex64", "packed_spf32_continuous"),
                    predicted_cost=float(size),
                    predicted_error=absolute_bound,
                    reason="producer and compatible consumers share the canonical SPF32 ABI",
                ),
            ),
            capabilities_used=(
                "SPF32",
                "PACKED_CONTINUITY",
                (
                    "SPF_NATIVE_IFFT"
                    if fft_count and not terminal_conversions
                    else "IFFT"
                    if fft_count
                    else "PACKED_SPECTRUM"
                ),
            ),
            diagnostics={
                "conversion_counts": {
                    "ingress_complex_to_spf": packed.ingress_conversions,
                    "intermediate_spf_to_complex": 0,
                    "terminal_spf_to_complex": terminal_conversions,
                    "terminal_spf_to_scalar": scalar_projections,
                },
                "packed_operations": packed.packed_operations,
                "producer_provenance": packed.provenance,
                "terminal_output": name,
                "source_construction_mode": self.source.construction_mode,
                "transfer_construction_mode": self.transfer.construction_mode,
                "packed_ifft_implementation": (
                    self.ifft_implementation if fft_count and not terminal_conversions else None
                ),
                "error_certificate": {
                    "units": "absolute_value_domain",
                    "valid": certificate_valid,
                    "absolute_bound": absolute_bound,
                    "complex_absolute_bound": complex_absolute_bound,
                    "relative_bound": relative_bound,
                    "quantized_factors": packed.quantized_factors,
                    "saturated_elements": packed.saturated_elements,
                    "magnitude_upper_bound": packed.magnitude_upper_bound,
                    "limitation": (
                        "packed IFFT butterfly error is measured against the Cartesian oracle but is not yet "
                        "composed into this analytic product-quantization certificate"
                        if native_transform
                        else None
                    ),
                },
                **dict(self.metadata),
            },
        )
        return PackedMaterialization(np.asarray(values), packed, report)


def packed_transfer_from_log_phase(
    log2_magnitude: Any,
    phase_radians: Any,
    *,
    zero: Any = False,
    sample_spacing: tuple[float, float] = (1.0, 1.0),
    name: str = "compiled_transfer",
) -> PackedSpectrum2D:
    """Compile an analytic transfer directly into canonical packed words."""

    return NativeLogPhaseProducer2D(
        np.asarray(log2_magnitude, dtype=np.float64),
        np.asarray(phase_radians, dtype=np.float64),
        zero=zero,
        name=name,
        sample_spacing=sample_spacing,
    ).produce()


__all__ = [
    "NativeLogPhaseProducer2D",
    "PackedConsumerPipeline2D",
    "PackedMaterialization",
    "PackedSpectrum2D",
    "decode_packed_complex",
    "encode_complex_adapter",
    "pack_log_phase_codes",
    "packed_ifft2",
    "packed_ifft2_stages",
    "packed_transfer_from_log_phase",
    "project_packed_magnitude",
    "spf_product_relative_error_bound",
    "unpack_log_phase_codes",
    "validate_packed_spectrum",
]
