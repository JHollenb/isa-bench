"""Lower retained frequency transfers at the addressed ocean boundary.

``FilteredBlockView`` deliberately keeps its source and spectral operator
separate.  The native ocean ABI has a smaller contract: it consumes one
addressed SPF32 word per mode.  This module is the small lowering boundary
between those contracts.  It evaluates the shared view planes for the
addressed support, composes the log and phase once, and emits canonical SPF32
words without constructing either a transfer plane or a dense spectrum.

The returned error bounds are for the lowering step only.  They compare the
exact normalized view planes with the words emitted by this module; the native
sampler adds its own phase-table, reducer, and output bounds afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.tensor import v02


def _readonly(array: Any, *, dtype: Any | None = None) -> np.ndarray:
    result = np.asarray(array, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _signed_frequency(index: np.ndarray, size: int) -> np.ndarray:
    return np.where(index <= size // 2, index, index - size).astype(np.float64)


def _pair_weights(indices: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    self_conjugate = np.ones(indices.shape[0], dtype=bool)
    for axis, size in enumerate(shape):
        coordinate = indices[:, axis]
        self_conjugate &= (coordinate == 0) | ((size % 2 == 0) & (coordinate == size // 2))
    return np.where(self_conjugate, 1.0, 2.0)


def _phase_codes(phases: np.ndarray) -> np.ndarray:
    """Emit wrapped u16 phases with the v0.2 round-to-even rule."""

    # Inactive windows may carry an undefined phase.  They are canonicalized
    # to zero below; replace them before the integer conversion so lowering
    # never emits a spurious invalid-cast warning.
    finite_phases = np.where(np.isfinite(phases), phases, 0.0)
    with np.errstate(invalid="ignore"):
        wrapped = np.remainder(finite_phases, 2.0 * math.pi)
        codes = np.rint(wrapped * (v02.SPF32_PHASE_MOD / (2.0 * math.pi)))
    # A finite phase just below 2π can round to the wrapped zero code.  The
    # packed representation intentionally has a modular phase domain, so this
    # is the canonical result.
    return np.asarray(codes, dtype=np.int64) & int(v02.SPF32_PHASE_MASK)


def filtered_view_operations(view: Any) -> tuple[str, ...]:
    """Return all retained frequency operations, including nested views.

    A mask or a scalar header update can leave a ``FilteredBlockView`` around
    an earlier filtered view.  The outer transfer is then the identity while
    the inner transfer still carries the source operation.  Preserve that
    chain at route selection and reporting time.  Non-zero wide offsets are
    represented explicitly because ``planes`` composes them at the same
    source boundary even when no transfer atom was appended.
    """

    if not bool(getattr(view, "frequency_domain", False)):
        return ()
    source = getattr(view, "source", None)
    operations = list(filtered_view_operations(source)) if source is not None else []
    transfer = getattr(view, "transfer", None)
    operations.extend(str(value) for value in getattr(transfer, "operations", ()))
    log_offset = float(getattr(view, "log_offset", 0.0))
    phase_offset = float(getattr(view, "phase_offset", 0.0))
    if log_offset != 0.0 or phase_offset != 0.0:
        operations.append("view_offset")
    return tuple(operations)


def _view_has_payload(view: Any, name: str) -> bool:
    """Check sampled transfer payloads through a nested filtered chain."""

    if not bool(getattr(view, "frequency_domain", False)):
        return False
    transfer = getattr(view, "transfer", None)
    payload = getattr(transfer, name, ())
    if payload:
        return True
    source = getattr(view, "source", None)
    return source is not None and _view_has_payload(source, name)


@dataclass(frozen=True, slots=True)
class FilteredSourceLowering:
    """O(K) packed source produced from one retained filtered support.

    ``indices`` are full-grid ocean indices; ``source_addresses`` are indices
    in the view's own shape (which may be a half-plane shape).  ``exact_values``
    and ``emitted_values`` are support-sized diagnostics, not dense fields.
    """

    indices: np.ndarray
    source_addresses: np.ndarray
    source_words: np.ndarray
    words: np.ndarray
    exact_values: np.ndarray
    emitted_values: np.ndarray
    coefficient_errors: np.ndarray
    operations: tuple[str, ...]
    coefficient_error_bound: float
    log_quantization_bound: float
    log_saturation_bound: float
    phase_quantization_bound: float
    height_absolute_bound: float
    gradient_y_absolute_bound: float
    gradient_x_absolute_bound: float
    error_ledger: v02.ErrorLedger
    diagnostics: Mapping[str, Any]
    # The semantic planes are retained only on addressed support so a later
    # native footprint can be composed without asking the view to replay its
    # transfer chain (or allocating a dense coefficient plane).
    exact_logs: np.ndarray | None = None
    exact_phases: np.ndarray | None = None
    zero_mask: np.ndarray | None = None
    nan_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "indices",
            "source_addresses",
            "source_words",
            "words",
            "exact_values",
            "emitted_values",
            "coefficient_errors",
        ):
            value = np.asarray(getattr(self, name))
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        for name in ("exact_logs", "exact_phases", "zero_mask", "nan_mask"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value)
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        object.__setattr__(self, "operations", tuple(str(item) for item in self.operations))
        object.__setattr__(self, "diagnostics", MappingProxyType(dict(self.diagnostics)))

    @property
    def output_error_bound(self) -> float:
        """Largest output-plane bound attributable to source lowering."""

        return max(self.height_absolute_bound, self.gradient_y_absolute_bound, self.gradient_x_absolute_bound)

    @property
    def active_count(self) -> int:
        return int(self.words.size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_count": self.active_count,
            "operations": self.operations,
            "coefficient_error_bound": float(self.coefficient_error_bound),
            "log_quantization_bound": float(self.log_quantization_bound),
            "log_saturation_bound": float(self.log_saturation_bound),
            "phase_quantization_bound": float(self.phase_quantization_bound),
            "height_absolute_bound": float(self.height_absolute_bound),
            "gradient_y_absolute_bound": float(self.gradient_y_absolute_bound),
            "gradient_x_absolute_bound": float(self.gradient_x_absolute_bound),
            "output_error_bound": float(self.output_error_bound),
            "error_ledger": {
                name: float(getattr(self.error_ledger, name))
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
            **dict(self.diagnostics),
        }


def _emit_composed_spf32_planes(
    index_array: np.ndarray,
    source_addresses: np.ndarray,
    source_words_array: np.ndarray,
    logs: np.ndarray,
    phases: np.ndarray,
    zero: np.ndarray,
    nan: np.ndarray,
    *,
    shape: tuple[int, ...],
    spacing: Sequence[float] | float,
    normalize: bool,
    operations: Sequence[str],
    diagnostics: Mapping[str, Any],
) -> FilteredSourceLowering:
    """Emit one canonical SPF32 support from already-composed planes.

    Both retained ``FilteredBlockView`` transfers and native Gaussian
    footprint precomposition cross this one emitter.  Keeping the packing,
    Hermitian checks, and support-sized certificate in one place prevents the
    two native lowering paths from drifting in their rounding or residual
    accounting.
    """

    index_array = np.asarray(index_array, dtype=np.int64).reshape(-1, len(shape))
    source_addresses = np.asarray(source_addresses, dtype=np.int64).reshape(-1)
    source_words_array = np.asarray(source_words_array, dtype=np.uint32).reshape(-1)
    logs = np.asarray(logs, dtype=np.float64).reshape(-1)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    zero = np.asarray(zero, dtype=bool).reshape(-1)
    nan = np.asarray(nan, dtype=bool).reshape(-1)
    support_count = index_array.shape[0]
    if not (
        source_addresses.size
        == source_words_array.size
        == logs.size
        == phases.size
        == zero.size
        == nan.size
        == support_count
    ):
        raise ValueError("composed SPF32 planes do not match the retained support")

    # Hard windows, and a Gaussian that underflows in the diagnostic scalar
    # domain, are represented by the one canonical inactive SPF zero.  Never
    # feed their undefined phase/log values to an integer conversion.
    effective_zero = zero | np.isneginf(logs)
    active = ~effective_zero & ~nan
    if np.any(nan & ~effective_zero):
        raise ValueError("native filtered source lowering cannot consume active NaN words")
    if np.any(~np.isfinite(logs[active])) or np.any(~np.isfinite(phases[active])):
        raise ValueError("native filtered source lowering requires finite active log/phase planes")

    with np.errstate(over="ignore", invalid="ignore"):
        log_float = logs * float(v02.SPF32_L_SCALE)
    finite_log_float = np.where(np.isfinite(log_float), log_float, 0.0)
    # Finite planes outside the representable SPF32 range use endpoint
    # saturation.  The coefficient delta, including this saturation, is part
    # of the returned support certificate.
    bounded_log_float = np.clip(
        finite_log_float,
        float(v02.SPF32_L_MIN_FINITE),
        float(v02.SPF32_L_MAX_FINITE),
    )
    rounded_logs = np.rint(bounded_log_float).astype(np.int64)
    rounded_logs = np.where(effective_zero, int(v02.SPF32_L_ZERO), rounded_logs)
    rounded_logs = np.where(
        active & (rounded_logs == int(v02.SPF32_L_ZERO)), int(v02.SPF32_L_MIN_FINITE), rounded_logs
    )
    rounded_phases = _phase_codes(phases)
    rounded_phases = np.where(effective_zero, int(v02.SPF32_T_ZERO), rounded_phases)
    words = np.asarray(v02.pack_spf32(rounded_logs, rounded_phases), dtype=np.uint32).reshape(-1)
    self_conjugate = _pair_weights(index_array, shape) == 1.0
    emitted_phase_codes = np.asarray(v02.unpack_T(words), dtype=np.uint16)
    invalid_self_phase = self_conjugate & active & ~np.isin(emitted_phase_codes, (0, 32768))
    if np.any(invalid_self_phase):
        raise ValueError("filtered source transfer violates the real Hermitian self-conjugate boundary")

    exact_values = np.zeros(support_count, dtype=np.complex128)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exact_values[active] = np.exp2(logs[active]) * np.exp(1j * np.remainder(phases[active], 2.0 * math.pi))
    if np.any(~np.isfinite(exact_values[active].real)) or np.any(~np.isfinite(exact_values[active].imag)):
        raise ValueError("filtered source transfer cannot be represented as finite SPF32 values")
    emitted_values = np.asarray(v02.decode_complex(words), dtype=np.complex128).reshape(-1)
    coefficient_errors = np.abs(emitted_values - exact_values)
    coefficient_errors = np.where(effective_zero, 0.0, coefficient_errors)

    # Quantization diagnostics retain both code-domain components, while the
    # field-space bounds use the same pair weights and physical derivative
    # gains as the native packed reducer.  Summing O(K) support is conservative
    # for every output sample and remains valid near cancellation.
    active_logs = active & np.isfinite(log_float)
    log_error = np.abs(bounded_log_float - rounded_logs.astype(np.float64)) / float(v02.SPF32_L_SCALE)
    log_error = float(np.max(log_error[active_logs], initial=0.0))
    with np.errstate(over="ignore", invalid="ignore"):
        log_saturation = np.abs(log_float - bounded_log_float) / float(v02.SPF32_L_SCALE)
    log_saturation = float(np.max(log_saturation[active_logs], initial=0.0))
    phase_error = np.zeros_like(phases)
    if np.any(active):
        emitted_phase = rounded_phases.astype(np.float64) * (2.0 * math.pi / v02.SPF32_PHASE_MOD)
        phase_error[active] = np.abs(np.angle(np.exp(1j * (emitted_phase[active] - phases[active]))))
    phase_error_bound = float(np.max(phase_error[active], initial=0.0))
    weights = np.where(self_conjugate, 1.0, 2.0)
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.ndim == 0:
        spacing_array = np.full(len(shape), float(spacing_array), dtype=np.float64)
    if (
        spacing_array.shape != (len(shape),)
        or np.any(~np.isfinite(spacing_array))
        or np.any(spacing_array <= 0.0)
    ):
        raise ValueError("filtered source spacing must be finite, positive, and match the field rank")
    normalizer = 1.0 / float(math.prod(shape)) if normalize else 1.0
    absolute_errors = np.asarray(coefficient_errors, dtype=np.float64)
    height_bound = float(np.sum(weights * absolute_errors, dtype=np.float64) * normalizer)
    gradient_bounds: list[float] = []
    for axis, size in enumerate(shape):
        signed = _signed_frequency(index_array[:, axis], size)
        gain = np.abs(2.0 * math.pi * signed / (float(size) * spacing_array[axis]))
        gradient_bounds.append(float(np.sum(weights * absolute_errors * gain, dtype=np.float64) * normalizer))
    gradient_y_bound = gradient_bounds[0]
    gradient_x_bound = gradient_bounds[1] if len(gradient_bounds) > 1 else gradient_bounds[0]
    output_bound = max(height_bound, gradient_y_bound, gradient_x_bound)
    ledger = v02.ErrorLedger(source_quantization=output_bound)
    diagnostic_values = {
        "domain": "frequency",
        "representation": "spf32",
        "support_elements": int(support_count),
        "dense_intermediate_elements": 0,
        "normalization": bool(normalize),
        "log2_plane_composed_once": True,
        "phase_plane_composed_once": True,
        "hermitian_self_conjugates_checked": True,
        **dict(diagnostics),
    }
    canonical_phases = np.where(effective_zero, 0.0, phases)
    return FilteredSourceLowering(
        indices=_readonly(index_array, dtype=np.int64),
        source_addresses=_readonly(source_addresses, dtype=np.int64),
        source_words=_readonly(source_words_array, dtype=np.uint32),
        words=_readonly(words, dtype=np.uint32),
        exact_values=_readonly(exact_values, dtype=np.complex128),
        emitted_values=_readonly(emitted_values, dtype=np.complex128),
        coefficient_errors=_readonly(coefficient_errors, dtype=np.float64),
        operations=tuple(str(item) for item in operations),
        coefficient_error_bound=float(np.max(absolute_errors, initial=0.0)),
        log_quantization_bound=log_error,
        log_saturation_bound=log_saturation,
        phase_quantization_bound=phase_error_bound,
        height_absolute_bound=height_bound,
        gradient_y_absolute_bound=gradient_y_bound,
        gradient_x_absolute_bound=gradient_x_bound,
        error_ledger=ledger,
        diagnostics=MappingProxyType(diagnostic_values),
        exact_logs=_readonly(logs, dtype=np.float64),
        exact_phases=_readonly(canonical_phases, dtype=np.float64),
        zero_mask=_readonly(effective_zero, dtype=bool),
        nan_mask=_readonly(nan, dtype=bool),
    )


def lower_filtered_source(
    view: Any,
    indices: Any,
    *,
    shape: Sequence[int],
    spacing: Sequence[float] | float = 1.0,
    normalize: bool = True,
    source_words: Any | None = None,
) -> FilteredSourceLowering:
    """Lower a frequency-domain ``FilteredBlockView`` on addressed support.

    The source words are read once at ``O(K)`` addresses.  Shared
    ``FilteredBlockView.planes`` supplies source headers, transfer atoms,
    sampled residuals/windows, masks, and wide offsets; this function only
    quantizes those already-composed planes to SPF32.  Spatial filtered views,
    intensity views, active NaNs, and active non-finite planes are rejected.
    Finite log planes outside the SPF32 code range use canonical endpoint
    saturation, and the resulting coefficient delta is included in the
    support-sized error bound.
    """

    shape_tuple = tuple(int(value) for value in shape)
    if not shape_tuple or any(value <= 0 for value in shape_tuple):
        raise ValueError("filtered source lowering requires a positive field shape")
    if not bool(getattr(view, "frequency_domain", False)):
        raise ValueError("native filtered source lowering requires a frequency-domain view")
    if str(getattr(view, "observable", "complex")) != "complex":
        raise ValueError("native filtered source lowering requires a complex-valued view")

    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim == 1:
        index_array = index_array.reshape(1, -1)
    if index_array.ndim != 2 or index_array.shape[1] != len(shape_tuple):
        raise ValueError("filtered source indices must have shape (K, dimension)")
    index_array = index_array.copy()
    for axis, size in enumerate(shape_tuple):
        index_array[:, axis] %= size

    view_shape = tuple(int(value) for value in getattr(view, "shape", ()))
    if len(view_shape) != index_array.shape[1] or any(value <= 0 for value in view_shape):
        raise ValueError("filtered source view shape does not match the support dimension")
    try:
        source_addresses = np.asarray(
            np.ravel_multi_index(
                tuple(index_array[:, axis] for axis in range(index_array.shape[1])), view_shape
            ),
            dtype=np.int64,
        ).reshape(-1)
    except ValueError as exc:
        raise ValueError("filtered source support is outside the retained view shape") from exc

    if source_words is None:
        source_words_array = np.asarray(view.sample_words(source_addresses, phase=True), dtype=np.uint32).reshape(-1)
    else:
        source_words_array = np.asarray(source_words, dtype=np.uint32).reshape(-1)
    if source_words_array.size != index_array.shape[0]:
        raise ValueError("filtered source words do not match the retained support")

    # ``planes`` is the shared semantic implementation.  It composes the
    # source log/phase with every retained transfer operation once and leaves
    # residual/window/mask behavior visible through its sentinels.
    logs, phases, zero, nan = view.planes(source_words_array, source_addresses)
    logs = np.asarray(logs, dtype=np.float64).reshape(-1)
    phases = np.asarray(phases, dtype=np.float64).reshape(-1)
    zero = np.asarray(zero, dtype=bool).reshape(-1)
    nan = np.asarray(nan, dtype=bool).reshape(-1)
    if not (logs.size == phases.size == zero.size == nan.size == index_array.shape[0]):
        raise ValueError("filtered source plane result does not match the retained support")

    return _emit_composed_spf32_planes(
        index_array,
        source_addresses,
        source_words_array,
        logs,
        phases,
        zero,
        nan,
        shape=shape_tuple,
        spacing=spacing,
        normalize=normalize,
        operations=filtered_view_operations(view),
        diagnostics={
            "source_transfer_lowered": True,
            "residuals_consumed": _view_has_payload(view, "phase_residuals"),
            "windows_consumed": _view_has_payload(view, "windows"),
        },
    )


def _gaussian_log2_gain(
    index_array: np.ndarray,
    *,
    shape: tuple[int, int],
    spacing: Sequence[float] | float,
    covariance: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a support-sized Gaussian log2 attenuation and physical freq."""

    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
        raise ValueError("native Gaussian footprint covariance must be finite 2x2")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("native Gaussian footprint covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(matrix), initial=0.0)) < -1.0e-12:
        raise ValueError("native Gaussian footprint covariance must be positive semidefinite")
    spacing_array = np.asarray(spacing, dtype=np.float64)
    if spacing_array.ndim == 0:
        spacing_array = np.full(2, float(spacing_array), dtype=np.float64)
    if (
        spacing_array.shape != (2,)
        or np.any(~np.isfinite(spacing_array))
        or np.any(spacing_array <= 0.0)
    ):
        raise ValueError("native Gaussian footprint spacing must be finite and positive")
    signed_y = _signed_frequency(index_array[:, 0], shape[0])
    signed_x = _signed_frequency(index_array[:, 1], shape[1])
    # The public footprint covariance is in (x, y) order, whereas addressed
    # ocean indices and native ABI spacing are (y, x).  Keep that reversal at
    # this one physical-frequency boundary so positive and negative cross
    # terms use the same convention as the reference footprint factor.
    frequencies = np.stack(
        (
            signed_x / (float(shape[1]) * spacing_array[1]),
            signed_y / (float(shape[0]) * spacing_array[0]),
        ),
        axis=1,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        quadratic = np.einsum("ki,ij,kj->k", frequencies, matrix, frequencies)
        # A PSD matrix can have a tiny negative roundoff residue.  Clamping
        # here keeps attenuation <= 1, matching the validated reference
        # Gaussian without changing a meaningful cross-term contribution.
        quadratic = np.maximum(quadratic, 0.0)
        exponent = -2.0 * math.pi**2 * quadratic
        log2_gain = exponent / math.log(2.0)
    if np.any(np.isnan(log2_gain)) or np.any(np.isposinf(log2_gain)):
        raise ValueError("native Gaussian footprint produces non-finite attenuation")
    return np.asarray(log2_gain, dtype=np.float64), frequencies


def lower_spf32_gaussian_footprint(
    indices: Any,
    words: Any | None = None,
    *,
    shape: Sequence[int],
    spacing: Sequence[float] | float = 1.0,
    normalize: bool = True,
    covariance: Any,
    source_lowering: FilteredSourceLowering | None = None,
) -> FilteredSourceLowering:
    """Compose a full Gaussian covariance into addressed SPF32 words.

    The native C ABI exposes only separable ``(sigma_y, sigma_x)`` values.
    This support-sized lowering evaluates the complete physical-frequency
    quadratic form once, reuses the canonical SPF32 emitter used by retained
    source transfers, and lets the native sampler receive zero sigmas.  When
    ``source_lowering`` is supplied its semantic log/phase planes are reused,
    so source transfer and footprint are composed before one final rounding.
    """

    shape_tuple = tuple(int(value) for value in shape)
    if len(shape_tuple) != 2 or any(value <= 0 for value in shape_tuple):
        raise ValueError("native Gaussian footprint lowering requires a positive two-dimensional field")

    if source_lowering is not None:
        index_array = np.asarray(source_lowering.indices, dtype=np.int64).reshape(-1, 2)
        source_addresses = np.asarray(source_lowering.source_addresses, dtype=np.int64).reshape(-1)
        source_words_array = np.asarray(source_lowering.source_words, dtype=np.uint32).reshape(-1)
        if source_lowering.exact_logs is None or source_lowering.exact_phases is None:
            raise ValueError("source lowering does not retain semantic planes for footprint composition")
        logs = np.asarray(source_lowering.exact_logs, dtype=np.float64).reshape(-1)
        phases = np.asarray(source_lowering.exact_phases, dtype=np.float64).reshape(-1)
        zero = (
            np.asarray(source_lowering.zero_mask, dtype=bool).reshape(-1)
            if source_lowering.zero_mask is not None
            else np.isneginf(logs)
        )
        nan = (
            np.asarray(source_lowering.nan_mask, dtype=bool).reshape(-1)
            if source_lowering.nan_mask is not None
            else np.zeros(logs.shape, dtype=bool)
        )
        operations = (*source_lowering.operations, "gaussian_footprint")
        source_transfer_lowered = bool(source_lowering.diagnostics.get("source_transfer_lowered", False))
    else:
        index_array = np.asarray(indices, dtype=np.int64)
        if index_array.ndim == 1:
            index_array = index_array.reshape(1, -1)
        if index_array.ndim != 2 or index_array.shape[1] != 2:
            raise ValueError("native Gaussian footprint indices must have shape (K, 2)")
        index_array = index_array.copy()
        for axis, size in enumerate(shape_tuple):
            index_array[:, axis] %= size
        if words is None:
            raise ValueError("native Gaussian footprint lowering requires packed source words")
        source_words_array = np.asarray(words, dtype=np.uint32).reshape(-1)
        if source_words_array.size != index_array.shape[0]:
            raise ValueError("native Gaussian footprint words do not match the support")
        source_addresses = np.asarray(
            np.ravel_multi_index(
                tuple(index_array[:, axis] for axis in range(2)),
                shape_tuple,
            ),
            dtype=np.int64,
        ).reshape(-1)
        log_codes = np.asarray(v02.unpack_L(source_words_array), dtype=np.int64).reshape(-1)
        phase_codes = np.asarray(v02.unpack_T(source_words_array), dtype=np.uint16).reshape(-1)
        zero = log_codes == int(v02.SPF32_L_ZERO)
        # Reserved NaN/infinite log codes cannot cross the native boundary as
        # a finite coefficient certificate.  Mark every reserved-log word so
        # the shared emitter raises the same active-NaN mechanics error.
        nan = (source_words_array == np.uint32(v02.SPF32_WORD_NAN)) | (
            log_codes == int(v02.SPF32_NAN_L)
        )
        logs = np.where(zero, -np.inf, log_codes.astype(np.float64) / float(v02.SPF32_L_SCALE))
        phases = phase_codes.astype(np.float64) * (2.0 * math.pi / v02.SPF32_PHASE_MOD)
        operations = ("gaussian_footprint",)
        source_transfer_lowered = False

    if not (
        index_array.shape[0]
        == source_addresses.size
        == source_words_array.size
        == logs.size
        == phases.size
        == zero.size
        == nan.size
    ):
        raise ValueError("native Gaussian footprint planes do not match the support")
    log2_gain, frequencies = _gaussian_log2_gain(
        index_array,
        shape=(shape_tuple[0], shape_tuple[1]),
        spacing=spacing,
        covariance=covariance,
    )
    spacing_values = np.asarray(spacing, dtype=np.float64)
    if spacing_values.ndim == 0:
        spacing_values = np.full(2, float(spacing_values), dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        composed_logs = logs + log2_gain
    if np.any(np.isnan(composed_logs)) or np.any(np.isposinf(composed_logs)):
        raise ValueError("native Gaussian footprint produces non-finite source planes")
    lowering = _emit_composed_spf32_planes(
        index_array,
        source_addresses,
        source_words_array,
        composed_logs,
        phases,
        zero,
        nan,
        shape=shape_tuple,
        spacing=spacing,
        normalize=normalize,
        operations=operations,
        diagnostics={
            "source_transfer_lowered": source_transfer_lowered,
            "footprint_lowered": True,
            "footprint_composed_once": True,
            "footprint_covariance_xy": tuple(tuple(float(value) for value in row) for row in np.asarray(covariance)),
            "physical_frequency_order": "xy_from_addressed_yx",
            "physical_spacing_yx": tuple(float(value) for value in spacing_values),
            "footprint_log2_gain_min": float(np.min(log2_gain)) if log2_gain.size else 0.0,
            "footprint_log2_gain_max": float(np.max(log2_gain)) if log2_gain.size else 0.0,
            "footprint_frequency_min": tuple(float(value) for value in np.min(frequencies, axis=0))
            if frequencies.shape[0]
            else (0.0, 0.0),
            "footprint_frequency_max": tuple(float(value) for value in np.max(frequencies, axis=0))
            if frequencies.shape[0]
            else (0.0, 0.0),
        },
    )
    return lowering



__all__ = [
    "FilteredSourceLowering",
    "filtered_view_operations",
    "lower_filtered_source",
    "lower_spf32_gaussian_footprint",
]
