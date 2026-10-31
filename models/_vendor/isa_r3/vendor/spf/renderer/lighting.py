"""Positive-light terms and conservative light clustering for SFR v0.2.

Lights are ordinary incoherent radiance sources.  They are never promoted to
the coherent complex algebra by this module.  The spherical-Gaussian helpers
reuse the analytic reference implementation in ``spf.tensor.v02`` and keep a
real, non-negative light boundary around it.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, cast, runtime_checkable

import numpy as np

from spf.renderer.materials import (
    Material,
    _block_payload,
    _block_real_values,
    _closed_block_payload,
    as_log_pos_plane,
    as_native_color,
    evaluate_brdf,
    log_sum,
)
from spf.renderer.model import Field, Region, Support
from spf.tensor import v02
from spf.tensor.v02_planes import LogPosPlane, PlaneReduction, reduce_logpos

_EPS = np.finfo(np.float64).tiny


def _unit(vector: Any, *, name: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(values))
    if norm <= _EPS:
        raise ValueError(f"{name} must be nonzero")
    return values / norm


def _positive_scalar(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be numeric, not bool")
    if isinstance(value, v02.RawLogCode):
        raise TypeError(f"{name} requires a typed positive value, not RawLogCode")
    if isinstance(value, v02.SPF32Value):
        raise TypeError(f"{name} requires a real radiance value, not a packed SPF32 word")
    block = _block_payload(value)
    if block is not None:
        block_view = block[0]
        if int(np.prod(block_view.shape)) != 1:
            raise ValueError(f"{name} block must contain one scalar")
        values = _block_real_values(value, name=name)
        assert values is not None
        result = float(values.reshape(-1)[0])
    else:
        result = float(value.value) if isinstance(value, v02.LogPos) else float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _validate_block_field_source(value: Any, *, name: str) -> None:
    """Reject block Field residuals until an addressed residual terminal exists."""

    if not isinstance(value, Field) or _block_payload(value) is None:
        return
    if value.representation.residual is not None:
        raise TypeError(
            f"{name} block Field carries a representation residual; "
            "canonical addressed residual sampling is unavailable"
        )


def _source_field_ledger(value: Any, *, name: str) -> v02.ErrorLedger:
    """Return uncertainty owned by an addressed Field source."""

    if not isinstance(value, Field) or _block_payload(value) is None:
        return v02.ErrorLedger()
    _validate_block_field_source(value, name=name)
    from spf.renderer.normalization import complete_field_ledger

    return complete_field_ledger(value)


def _scaled_source_ledger(ledger: v02.ErrorLedger, scale: float) -> v02.ErrorLedger:
    """Scale source uncertainty while preserving its ledger categories."""

    factor = float(abs(scale))

    def scaled(value: float) -> float:
        value = float(value)
        if value == 0.0 or factor == 0.0:
            return 0.0
        if not math.isfinite(value) or not math.isfinite(factor):
            return math.inf
        result = value * factor
        return result if math.isfinite(result) else math.inf

    return v02.ErrorLedger(
        source_quantization=scaled(ledger.source_quantization),
        representation_fit=scaled(ledger.representation_fit),
        pruning=scaled(ledger.pruning),
        reducer=scaled(ledger.reducer),
        transform=scaled(ledger.transform),
        materialization=scaled(ledger.materialization),
        display=scaled(ledger.display),
        stages=ledger.stages,
    )


def _source_block_envelope(
    value: Any, *, name: str
) -> tuple[float, float, v02.ErrorLedger] | None:
    """Return compact block max, source error, and its category ledger."""

    _validate_block_field_source(value, name=name)
    selected = _closed_block_payload(value, name=name)
    if selected is None:
        return None
    block, _ = selected
    ledger = _source_field_ledger(value, name=name)
    ledger_bound = float(ledger.total_bound)
    # A malformed or unbounded source must retain the light. Treat NaN as an
    # unbounded screening result rather than allowing culling to proceed.
    error = ledger_bound if math.isfinite(ledger_bound) and ledger_bound >= 0.0 else math.inf
    shape = tuple(int(item) for item in block.shape)
    if len(shape) > 1:
        # Header/band/residual envelopes stay in packed storage. Unsupported
        # expressions report infinity, so screening never silently decodes
        # a source plane or discards an unbounded contribution.
        return float(block.magnitude_upper_bound), error, ledger
    from spf.renderer.materials import block_words

    words = block_words(value, name=name)
    if words is not None:
        # Envelope straight from the packed log codes through the terminal
        # magnitude ROM: no decode of the source plane.
        from spf.renderer.transport_terminal import display_values
        from spf.tensor.v02_planes import LogSignedPlane

        signed = LogSignedPlane.from_words(words[0].reshape(-1))
        if np.any(signed.is_nan) or np.any(signed.sign < 0):
            raise ValueError(f"{name} block must be finite and non-negative")
        magnitudes = display_values(signed)
        return (float(np.max(magnitudes)) if magnitudes.size else 0.0), error, ledger
    values = _block_real_values(value, name=name)
    assert values is not None
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} block must be finite and non-negative")
    return (float(np.max(values)) if values.size else 0.0), error, ledger


def _source_envelope_or_constant(
    value: Any, constant: float, *, name: str
) -> tuple[float, float, v02.ErrorLedger]:
    envelope = _source_block_envelope(value, name=name)
    if envelope is not None:
        return envelope
    return float(constant), 0.0, v02.ErrorLedger()


def _product_source_ledger(
    left_source: Any,
    left_constant: float,
    right_source: Any,
    right_constant: float,
    *,
    left_name: str,
    right_name: str,
) -> v02.ErrorLedger:
    """Propagate two multiplicative positive source uncertainties.

    For maxima ``a ± e_a`` and ``b ± e_b``, the absolute product uncertainty
    is bounded by ``e_a (b + e_b) + e_b a``. This retains each operand's
    category while including the cross term ``e_a e_b``.
    """

    left_max, _left_error, left_ledger = _source_envelope_or_constant(
        left_source, left_constant, name=left_name
    )
    right_max, right_error, right_ledger = _source_envelope_or_constant(
        right_source, right_constant, name=right_name
    )
    return _scaled_source_ledger(left_ledger, right_max + right_error).plus(
        _scaled_source_ledger(right_ledger, left_max)
    )


def _source_block_sample(value: Any, *, point: Any = None, channels: int = 3, name: str = "radiance") -> np.ndarray:
    """Sample a positive block source at one point without dense reconstruction."""

    _validate_block_field_source(value, name=name)
    selected = _closed_block_payload(value, name=name)
    if selected is None:
        return as_native_color(value, name=name, channels=channels, allow_negative=False)
    block, owner = selected
    from spf.renderer.volume import _project_block_real, _sample_block_density

    shape = tuple(int(item) for item in block.shape)
    value_kind = None if owner is None else str(owner.value_kind)
    if owner is not None and str(getattr(owner.coherence, "value", owner.coherence)) == "COHERENT_COMPLEX":
        raise TypeError(f"{name} is coherent-complex; call coherent_intensity() explicitly")
    if len(shape) == 1:
        values = _project_block_real(block, np.arange(shape[0], dtype=np.int64), value_kind=value_kind)
        return as_native_color(values, name=name, channels=channels, allow_negative=False)
    coordinates = (
        np.zeros(2, dtype=np.float64)
        if point is None
        else np.asarray(point, dtype=np.float64).reshape(-1)[:2]
    )
    if coordinates.size != 2 or not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{name} block query point must provide two finite coordinates")
    if owner is not None:
        support_kind = str(getattr(owner.support.kind, "value", owner.support.kind)).lower()
        if support_kind == "empty":
            return np.zeros(channels, dtype=np.float64)
        if owner.support.region is not None and not owner.support.region.contains(coordinates.tolist()):
            return np.zeros(channels, dtype=np.float64)
        if len(shape) > 1 and owner.chart.dimension != 2:
            raise TypeError(f"{name} spatial block sampling requires a two-dimensional coordinate chart")
    if owner is not None and owner.chart.dimension == 2:
        chart = owner.chart
        matrix = np.asarray(chart.basis, dtype=np.float64) @ np.diag(
            np.asarray(chart.scale, dtype=np.float64) * np.asarray(chart.sample_spacing, dtype=np.float64)
        )
        if np.linalg.matrix_rank(matrix) != 2:
            raise ValueError(f"{name} block chart must be invertible")
        coordinates = (coordinates - np.asarray(chart.origin, dtype=np.float64)) @ np.linalg.inv(matrix).T
    if len(shape) == 2:
        sampled = _sample_block_density(block, coordinates, value_kind=value_kind, boundary="clamp")
    elif len(shape) == 3 and shape[-1] == channels:
        sampled = _sample_block_density(
            block,
            np.asarray(
                [
                    (coordinates[0], coordinates[1], float(channel))
                    for channel in range(channels)
                ],
                dtype=np.float64,
            ),
            value_kind=value_kind,
            boundary="clamp",
        )
    else:
        raise ValueError(f"{name} block must be a vector, two-dimensional field, or HxWx{channels} field")
    return as_native_color(sampled, name=name, channels=channels, allow_negative=False)


def _source_block_max(value: Any, *, channels: int = 3, name: str = "radiance") -> float | None:
    envelope = _source_block_envelope(value, name=name)
    if envelope is None:
        return None
    maximum, error, _ = envelope
    if not math.isfinite(error):
        return math.inf
    result = maximum + error
    return result if math.isfinite(result) else math.inf


def _light_color_envelope(value: Any) -> np.ndarray:
    """Nominal float colour of a light source without decoding a block plane.

    A compact block source is read as packed words and projected through the
    terminal magnitude ROM; anything else takes the ordinary float sample.
    """

    try:
        plane, ingress = as_log_pos_plane(value, name="light color", channels=3)
    except (TypeError, ValueError):
        return _source_block_sample(value, channels=3, name="light color")
    if ingress != "addressed_block_words":
        return _source_block_sample(value, channels=3, name="light color")
    from spf.renderer.transport_terminal import display_values

    return np.asarray(display_values(plane), dtype=np.float64).reshape(3)


@runtime_checkable
class Light(Protocol):
    """Minimal structural light contract used by the scene orchestrator."""

    name: str

    def direction_at(self, point: Any = None) -> np.ndarray: ...

    def radiance_at(self, point: Any = None) -> np.ndarray: ...

    def max_radiance(self) -> float: ...


@dataclass(frozen=True, slots=True)
class DirectionalLight:
    """A distant, constant-direction positive radiance source."""

    direction: Any = (0.0, 0.0, 1.0)
    color: Any = (1.0, 1.0, 1.0)
    intensity: Any = 1.0
    name: str = "directional"
    angular_radius: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    color_source: Any = field(default=None, init=False, repr=False, compare=False)
    intensity_source: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        direction = _unit(self.direction, name="directional light direction")
        color_source = self.color if _block_payload(self.color) is not None else None
        intensity_source = self.intensity if _block_payload(self.intensity) is not None else None
        _validate_block_field_source(color_source, name="light color")
        _validate_block_field_source(intensity_source, name="light intensity")
        color_block = _block_payload(color_source)
        color_shape = tuple(int(item) for item in color_block[0].shape) if color_block is not None else ()
        if color_source is not None and not (len(color_shape) == 1 and color_shape[0] in {1, 3}):
            color = np.ones(3, dtype=np.float64)
        else:
            color = _light_color_envelope(color_source if color_source is not None else self.color)
        intensity = 1.0 if intensity_source is not None else _positive_scalar(self.intensity, name="light intensity")
        angular_radius = float(self.angular_radius)
        if not math.isfinite(angular_radius) or angular_radius < 0.0:
            raise ValueError("angular_radius must be finite and non-negative")
        object.__setattr__(self, "direction", tuple(float(value) for value in direction))
        object.__setattr__(self, "color", tuple(float(value) for value in color))
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "color_source", color_source)
        object.__setattr__(self, "intensity_source", intensity_source)
        object.__setattr__(self, "angular_radius", angular_radius)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def direction_vector(self) -> np.ndarray:
        return np.asarray(self.direction, dtype=np.float64)

    @property
    def color_array(self) -> np.ndarray:
        return np.asarray(self.color, dtype=np.float64)

    @property
    def radiance(self) -> np.ndarray:
        return self.radiance_at()

    def direction_at(self, point: Any = None) -> np.ndarray:
        del point
        return self.direction_vector.copy()

    def radiance_at(self, point: Any = None) -> np.ndarray:
        color = (
            _source_block_sample(self.color_source, point=point, channels=3, name="light color")
            if self.color_source is not None
            else self.color_array
        )
        intensity = (
            _source_block_sample(self.intensity_source, point=point, channels=1, name="light intensity")[0]
            if self.intensity_source is not None
            else self.intensity
        )
        return np.asarray(color * intensity, dtype=np.float64)

    def max_radiance(self) -> float:
        color_max = _source_block_max(self.color_source, name="light color")
        intensity_max = _source_block_max(self.intensity_source, channels=1, name="light intensity")
        if color_max is None:
            color_max = float(np.max(self.color_array))
        if intensity_max is None:
            intensity_max = self.intensity
        return float(color_max * intensity_max)

    def evaluate(self, point: Any = None) -> np.ndarray:
        return self.radiance_at(point)


@dataclass(frozen=True, slots=True)
class SphericalGaussianLight:
    """A positive spherical-Gaussian environment or area-light lobe."""

    mu: Any = (0.0, 0.0, 1.0)
    lambda_: float = 0.0
    amplitude: Any = 1.0
    color: Any = (1.0, 1.0, 1.0)
    name: str = "spherical_gaussian"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    amplitude_source: Any = field(default=None, init=False, repr=False, compare=False)
    color_source: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        direction = _unit(self.mu, name="spherical Gaussian direction")
        concentration = float(self.lambda_)
        if not math.isfinite(concentration) or concentration < 0.0:
            raise ValueError("spherical Gaussian lambda must be finite and non-negative")
        amplitude_source = self.amplitude if _block_payload(self.amplitude) is not None else None
        color_source = self.color if _block_payload(self.color) is not None else None
        _validate_block_field_source(amplitude_source, name="spherical Gaussian amplitude")
        _validate_block_field_source(color_source, name="light color")
        amplitude = 1.0 if amplitude_source is not None else _positive_scalar(
            self.amplitude, name="spherical Gaussian amplitude"
        )
        color_block = _block_payload(color_source)
        color_shape = tuple(int(item) for item in color_block[0].shape) if color_block is not None else ()
        if color_source is not None and not (len(color_shape) == 1 and color_shape[0] in {1, 3}):
            color = np.ones(3, dtype=np.float64)
        else:
            color = _light_color_envelope(color_source if color_source is not None else self.color)
        object.__setattr__(self, "mu", tuple(float(value) for value in direction))
        object.__setattr__(self, "lambda_", concentration)
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "color", tuple(float(value) for value in color))
        object.__setattr__(self, "amplitude_source", amplitude_source)
        object.__setattr__(self, "color_source", color_source)
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def lam(self) -> float:
        return self.lambda_

    @property
    def mu_vector(self) -> np.ndarray:
        return np.asarray(self.mu, dtype=np.float64)

    @property
    def color_array(self) -> np.ndarray:
        return np.asarray(self.color, dtype=np.float64)

    @property
    def lobe(self) -> v02.SphericalGaussian:
        if self.amplitude_source is not None or self.color_source is not None:
            raise TypeError("spatially varying spherical Gaussian sources do not have one global analytic lobe")
        return v02.SphericalGaussian(self.mu_vector, self.lambda_, complex(self.amplitude))

    def direction_at(self, point: Any = None) -> np.ndarray:
        del point
        return self.mu_vector.copy()

    def radiance_at(self, point: Any = None) -> np.ndarray:
        color = (
            _source_block_sample(self.color_source, point=point, channels=3, name="light color")
            if self.color_source is not None
            else self.color_array
        )
        amplitude = (
            _source_block_sample(self.amplitude_source, point=point, channels=1, name="light amplitude")[0]
            if self.amplitude_source is not None
            else self.amplitude
        )
        return np.asarray(color * amplitude, dtype=np.float64)

    def evaluate(self, directions: Any, *, point: Any = None) -> np.ndarray | float:
        values = np.asarray(directions, dtype=np.float64)
        color = (
            _source_block_sample(self.color_source, point=point, channels=3, name="light color")
            if self.color_source is not None
            else self.color_array
        )
        amplitude = (
            _source_block_sample(self.amplitude_source, point=point, channels=1, name="light amplitude")[0]
            if self.amplitude_source is not None
            else self.amplitude
        )
        if values.shape == (3,):
            if not math.isclose(float(np.linalg.norm(values)), 1.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("spherical Gaussian directions must be unit length")
            return np.asarray(
                float(amplitude * math.exp(self.lambda_ * (float(np.dot(self.mu_vector, values)) - 1.0))) * color,
                dtype=np.float64,
            )
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError("spherical Gaussian directions must have trailing dimension 3")
        norms = np.linalg.norm(values, axis=-1)
        if not np.allclose(norms, 1.0, rtol=1e-12, atol=1e-12):
            raise ValueError("spherical Gaussian directions must be unit length")
        weights = amplitude * np.exp(self.lambda_ * (np.einsum("...d,d->...", values, self.mu_vector) - 1.0))
        return np.asarray(weights[..., None] * color, dtype=np.float64)

    __call__ = evaluate

    def integral(self) -> np.ndarray:
        if self.amplitude_source is not None or self.color_source is not None:
            raise TypeError("spatially varying spherical Gaussian sources require point sampling")
        scalar = float(v02.spherical_gaussian_integral(self.lobe).real)
        return self.color_array * scalar

    def lobe_integral(self) -> float:
        return float(v02.spherical_gaussian_integral(self.lobe).real)

    def product(self, other: SphericalGaussianLight) -> v02.SphericalGaussian:
        return v02.spherical_gaussian_product(self.lobe, other.lobe)

    def max_radiance(self) -> float:
        color_max = _source_block_max(self.color_source, name="light color")
        amplitude_max = _source_block_max(self.amplitude_source, channels=1, name="light amplitude")
        if color_max is None:
            color_max = float(np.max(self.color_array))
        if amplitude_max is None:
            amplitude_max = self.amplitude
        return float(color_max * amplitude_max)


SphericalGaussian = v02.SphericalGaussian


def spherical_gaussian_light(
    mu: Any,
    lambda_: float,
    amplitude: Any = 1.0,
    *,
    color: Any = (1.0, 1.0, 1.0),
    name: str = "spherical_gaussian",
) -> SphericalGaussianLight:
    return SphericalGaussianLight(mu, lambda_, amplitude, color, name)


def spherical_gaussian_integral(value: SphericalGaussianLight | v02.SphericalGaussian) -> Any:
    """Return the analytic integral of a lobe or coloured light."""

    if isinstance(value, SphericalGaussianLight):
        return value.integral()
    return v02.spherical_gaussian_integral(value)


def spherical_gaussian_product(
    left: SphericalGaussianLight | v02.SphericalGaussian,
    right: SphericalGaussianLight | v02.SphericalGaussian,
) -> v02.SphericalGaussian:
    left_lobe = left.lobe if isinstance(left, SphericalGaussianLight) else left
    right_lobe = right.lobe if isinstance(right, SphericalGaussianLight) else right
    return v02.spherical_gaussian_product(left_lobe, right_lobe)


def _support_region(value: Region | Support | None) -> Region | None:
    if isinstance(value, Region):
        return value
    if isinstance(value, Support):
        return value.region
    return None


def _angular_tuple(value: Any) -> tuple[np.ndarray, float] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        direction = value.get("direction", value.get("mu"))
        cosine = value.get("cosine", value.get("min_cosine", -1.0))
    else:
        try:
            direction, cosine = value
        except (TypeError, ValueError) as exc:
            raise ValueError("angular_bound must be (direction, min_cosine)") from exc
    threshold = float(cosine)
    if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
        raise ValueError("angular bound cosine must be in [-1, 1]")
    return _unit(direction, name="angular bound direction"), threshold


def _point_in_region(point: Any, region: Region | None) -> bool:
    coordinates = np.asarray(point, dtype=np.float64).reshape(-1)
    return region is None or region.contains(coordinates.tolist())


def light_upper_bound(light: Any, *, point: Any = None, normal: Any = None) -> float:
    """Return a conservative scalar upper bound for one light contribution.

    The bound intentionally ignores cosine and BRDF attenuation.  It can be
    loose, but never makes a positive contribution look smaller than it is.
    """

    if point is not None and hasattr(light, "spatial_bound"):
        region = _support_region(light.spatial_bound)
        if not _point_in_region(point, region):
            return 0.0
    if hasattr(light, "max_radiance"):
        bound = float(light.max_radiance())
    elif hasattr(light, "radiance_at"):
        values = np.asarray(light.radiance_at(point), dtype=np.float64)
        bound = float(np.max(np.abs(values)))
    else:
        return math.inf
    if not math.isfinite(bound) or bound < 0.0:
        return math.inf
    del normal
    return math.nextafter(bound, math.inf) if bound > 0.0 else 0.0


def _ticks_from_linear(value: float) -> float:
    """SPF L code ticks (``256 log2 v``) for a non-negative linear magnitude."""

    if value == 0.0:
        return -math.inf
    if not math.isfinite(value):
        return math.inf
    return float(v02.SPF32_L_SCALE * math.log2(value))


def _linear_from_ticks(ticks: float) -> float:
    """Inverse of :func:`_ticks_from_linear`, rounded up so it stays a bound."""

    if ticks == -math.inf:
        return 0.0
    if not math.isfinite(ticks):
        return math.inf
    value = float(2.0 ** (ticks / v02.SPF32_L_SCALE))
    if not math.isfinite(value):
        return math.inf
    # The log round trip may land one ulp below the quantity it bounds; step up.
    return math.nextafter(value, math.inf) if value > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class LightCluster:
    """A conservative group bound used before evaluating individual lights.

    Design §22.3.  The cluster stores ``max_log_contribution`` (``L_max``, the
    largest single light contribution) **in SPF L code ticks**, i.e. in units of
    ``256 log2`` and not in plain ``log2``, together with ``active_count``
    (``N``) and the spatial/angular bounds.  The design's conservative aggregate
    is then

    ``L_U = L_max + 256 log2 N``   (:attr:`log_upper_bound`, also ticks),

    and ``2**(L_U/256)`` is :attr:`linear_upper_bound`.  The mechanically summed
    bound ``256 log2(Σ_i bound_i)`` (:attr:`log_sum_bound`, :attr:`linear_bound`)
    is retained because ``Σ_i bound_i ≤ N max_i bound_i`` makes it the tighter of
    the two certificates; culling uses it, so a cull is sound under the design
    bound as well.
    """

    lights: tuple[Any, ...]
    spatial_bound: Region | Support | None = None
    angular_bound: Any = None
    max_log_contribution: float | None = None
    active_count: int | None = None
    name: str = "light_cluster"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    log_sum_bound: float = field(default=-math.inf, init=False)
    linear_bound: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        lights = tuple(self.lights)
        if any(light is None for light in lights):
            raise ValueError("light cluster cannot contain None")
        region = _support_region(self.spatial_bound)
        angular = _angular_tuple(self.angular_bound)
        bounds = [light_upper_bound(light) for light in lights]
        linear_sum = math.fsum(bounds)
        # L_max in L ticks: the largest single contribution, per §22.3.
        max_ticks = _ticks_from_linear(max(bounds, default=0.0))
        if self.max_log_contribution is not None:
            supplied = float(self.max_log_contribution)
            if math.isnan(supplied):
                raise ValueError("max_log_contribution must not be NaN")
            # Never permit an advertised bound below the mechanically derived one.
            max_ticks = max(max_ticks, supplied)
            # max_i bound_i <= sum_i bound_i, so an advertised L_max raises the sum.
            linear_sum = max(linear_sum, _linear_from_ticks(max_ticks))
        count = len(lights) if self.active_count is None else int(self.active_count)
        if count < 0:
            raise ValueError("active_count must be non-negative")
        object.__setattr__(self, "lights", lights)
        object.__setattr__(self, "spatial_bound", region)
        object.__setattr__(self, "angular_bound", None if angular is None else (tuple(angular[0]), angular[1]))
        object.__setattr__(self, "max_log_contribution", max_ticks)
        object.__setattr__(self, "active_count", count)
        object.__setattr__(self, "log_sum_bound", _ticks_from_linear(linear_sum))
        object.__setattr__(self, "linear_bound", float(linear_sum))
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def from_lights(
        cls,
        lights: Iterable[Any],
        *,
        spatial_bound: Region | Support | None = None,
        angular_bound: Any = None,
        name: str = "light_cluster",
    ) -> LightCluster:
        return cls(tuple(lights), spatial_bound, angular_bound, name=name)

    @property
    def log_upper_bound(self) -> float:
        """Design §22.3 ``L_U = L_max + 256 log2 N`` in SPF L code ticks."""

        max_ticks = self.max_log_contribution
        count = int(self.active_count or 0)
        if max_ticks is None:
            return math.inf
        if count <= 0 or max_ticks == -math.inf:
            return -math.inf
        if not math.isfinite(max_ticks):
            return math.inf
        return float(max_ticks + v02.SPF32_L_SCALE * math.log2(count))

    @property
    def linear_upper_bound(self) -> float:
        """The §22.3 bound as an ordinary magnitude, ``2**(L_U/256)``."""

        return _linear_from_ticks(self.log_upper_bound)

    def upper_bound(self, *, point: Any = None, normal: Any = None) -> float:
        if point is not None and not _point_in_region(point, _support_region(self.spatial_bound)):
            return 0.0
        if normal is not None and self.angular_bound is not None:
            direction, threshold = self.angular_bound
            if float(np.dot(_unit(normal, name="normal"), direction)) < threshold:
                return 0.0
        # Recompute when a point-specific bound exists; otherwise the constructor
        # bound is already the sum of individual conservative bounds.  Both the
        # summed and the §22.3 aggregate bound the cluster, so report the tighter.
        if point is None and normal is None:
            return min(self.linear_bound, self.linear_upper_bound)
        return math.fsum(light_upper_bound(light, point=point, normal=normal) for light in self.lights)

    def should_cull(self, remaining_error: float, *, point: Any = None, normal: Any = None) -> bool:
        budget = float(remaining_error)
        if math.isnan(budget) or budget < 0.0:
            raise ValueError("remaining_error must be non-negative")
        bound = self.upper_bound(point=point, normal=normal)
        return math.isfinite(bound) and bound <= budget


def _light_source_ledger(light: Any) -> v02.ErrorLedger:
    """Collect source uncertainty for one light or a nested light cluster."""

    children = getattr(light, "lights", None)
    if children is not None:
        result = v02.ErrorLedger()
        for child in children:
            result = result.plus(_light_source_ledger(child))
        return result
    if isinstance(light, DirectionalLight):
        return _product_source_ledger(
            light.color_source,
            float(np.max(light.color_array)),
            light.intensity_source,
            float(light.intensity),
            left_name="light color",
            right_name="light intensity",
        )
    if isinstance(light, SphericalGaussianLight):
        return _product_source_ledger(
            light.color_source,
            float(np.max(light.color_array)),
            light.amplitude_source,
            float(light.amplitude),
            left_name="light color",
            right_name="spherical Gaussian amplitude",
        )
    # Preserve custody for compatible custom light carriers. If they expose
    # both source operands, use the same product rule; otherwise a lone source
    # is already at the radiance boundary and can be carried directly.
    color_source = getattr(light, "color_source", None)
    intensity_source = getattr(light, "intensity_source", None)
    amplitude_source = getattr(light, "amplitude_source", None)
    factor_source = intensity_source if intensity_source is not None else amplitude_source
    if color_source is not None or factor_source is not None:
        color = getattr(light, "color", 1.0)
        factor = getattr(light, "intensity", getattr(light, "amplitude", 1.0))
        color_values = np.asarray(color, dtype=np.float64).reshape(-1)
        color_constant = float(np.max(color_values)) if color_values.size else 0.0
        return _product_source_ledger(
            color_source,
            color_constant,
            factor_source,
            float(factor),
            left_name="light color",
            right_name="light intensity",
        )
    return v02.ErrorLedger()


def _lights_source_ledger(lights: Iterable[Any]) -> v02.ErrorLedger:
    result = v02.ErrorLedger()
    for light in lights:
        result = result.plus(_light_source_ledger(light))
    return result


def cluster_lights(lights: Iterable[Any], *, max_size: int | None = None) -> tuple[LightCluster, ...]:
    items = tuple(lights)
    if not items:
        return ()
    size = len(items) if max_size is None else int(max_size)
    if size <= 0:
        raise ValueError("max_size must be positive")
    return tuple(
        LightCluster.from_lights(items[start : start + size], name=f"light_cluster_{start // size}")
        for start in range(0, len(items), size)
    )


def cull_light_cluster(
    cluster: LightCluster,
    remaining_error: float,
    *,
    point: Any = None,
    normal: Any = None,
) -> bool:
    return cluster.should_cull(remaining_error, point=point, normal=normal)


def cull_lights(
    lights: Iterable[Any],
    remaining_error: float | None = None,
    *,
    error_budget: float | None = None,
    point: Any = None,
    normal: Any = None,
) -> tuple[Any, ...]:
    """Cull only terms whose conservative upper bound fits the error budget."""

    if remaining_error is not None and error_budget is not None:
        raise ValueError("pass only one of remaining_error or error_budget")
    budget_value = remaining_error if remaining_error is not None else error_budget
    if budget_value is None:
        return tuple(lights)
    budget = float(budget_value)
    if math.isnan(budget) or budget < 0.0:
        raise ValueError("remaining_error must be non-negative")
    selected, _ = _cull_lights_with_bound(tuple(lights), budget, point=point, normal=normal)
    return selected


def _cull_lights_with_bound(
    lights: tuple[Any, ...],
    budget: float,
    *,
    point: Any = None,
    normal: Any = None,
) -> tuple[tuple[Any, ...], float]:
    selected: list[Any] = []
    remaining = budget
    dropped_bound = 0.0
    for item in lights:
        if isinstance(item, LightCluster):
            bound = item.upper_bound(point=point, normal=normal)
            if math.isfinite(bound) and bound <= remaining:
                remaining -= bound
                dropped_bound += bound
            else:
                selected.extend(item.lights)
        else:
            bound = light_upper_bound(item, point=point, normal=normal)
            if math.isfinite(bound) and bound <= remaining:
                remaining -= bound
                dropped_bound += bound
            else:
                selected.append(item)
    return tuple(selected), dropped_bound


def sample_light(light: Any, direction: Any, *, point: Any = None) -> np.ndarray:
    """Sample a light in a specified unit direction as ordinary radiance."""

    if isinstance(light, DirectionalLight):
        return light.radiance_at(point)
    if isinstance(light, SphericalGaussianLight):
        values = light.evaluate(_unit(direction, name="light direction"), point=point)
        return np.asarray(values, dtype=np.float64).reshape(3)
    if hasattr(light, "evaluate"):
        values = np.asarray(light.evaluate(direction), dtype=np.float64)
    elif hasattr(light, "radiance_at"):
        values = np.asarray(light.radiance_at(point), dtype=np.float64)
    else:
        raise TypeError(f"unsupported light type {type(light).__name__}")
    if np.iscomplexobj(values):
        raise TypeError("coherent light values require an explicit intensity conversion")
    return as_native_color(values, name="sampled radiance", allow_negative=False)


def light_direction(light: Any, point: Any = None, *, fallback: Any = (0.0, 0.0, 1.0)) -> np.ndarray:
    if hasattr(light, "direction_at"):
        return _unit(light.direction_at(point), name="light direction")
    return _unit(fallback, name="light direction")


@dataclass(frozen=True, slots=True)
class LightingResult:
    color: np.ndarray
    contributions: tuple[np.ndarray, ...]
    active_lights: int
    culled_lights: int
    upper_bound: float
    culled_upper_bound: float = 0.0
    positive_logadds: int = 0
    pivot_reductions: int = 0
    reduction_error_bound: float = 0.0
    # Source-radiance uncertainty, after multiplicative light parameters and
    # before cosine/BRDF weighting. It is kept separate because a caller that
    # applies an unbounded material/angular gain cannot claim a finite final
    # pixel bound from this source-bound alone.
    source_error_bound: float = 0.0
    source_ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)

    def __post_init__(self) -> None:
        color = np.asarray(self.color, dtype=np.float64).reshape(-1)
        if color.size != 3 or not np.all(np.isfinite(color)):
            raise ValueError("lighting color must be a finite RGB vector")
        object.__setattr__(self, "color", color.copy())
        object.__setattr__(
            self, "contributions", tuple(np.asarray(item, dtype=np.float64).reshape(3) for item in self.contributions)
        )
        bound = float(self.culled_upper_bound)
        if math.isnan(bound) or bound < 0.0:
            raise ValueError("culled light bound must be non-negative")
        object.__setattr__(self, "culled_upper_bound", bound)
        reduction_bound = float(self.reduction_error_bound)
        if math.isnan(reduction_bound) or reduction_bound < 0.0:
            raise ValueError("light reduction error bound must be non-negative")
        if not isinstance(self.source_ledger, v02.ErrorLedger):
            raise TypeError("lighting source_ledger must be an ErrorLedger")
        source_bound = float(self.source_error_bound)
        ledger_bound = float(self.source_ledger.total_bound)
        if math.isnan(source_bound) or source_bound < 0.0:
            raise ValueError("lighting source error bound must be non-negative")
        if math.isnan(ledger_bound) or ledger_bound < 0.0:
            raise ValueError("lighting source_ledger must have a non-negative bound")
        source_bound = max(source_bound, ledger_bound)
        object.__setattr__(self, "positive_logadds", int(self.positive_logadds))
        object.__setattr__(self, "pivot_reductions", int(self.pivot_reductions))
        object.__setattr__(self, "reduction_error_bound", reduction_bound)
        object.__setattr__(self, "source_error_bound", source_bound)

    @property
    def ledger(self) -> v02.ErrorLedger:
        """Source-bound and positive-reduction uncertainty at their boundaries."""

        source = self.source_ledger
        represented = float(source.total_bound)
        if self.source_error_bound > represented:
            source = source.plus(v02.ErrorLedger(source_quantization=self.source_error_bound - represented))
        return source.plus(v02.ErrorLedger(reducer=self.reduction_error_bound))


# ---------------------------------------------------------------------------
# Visibility factor V_i (design §22.1 term construction, §24 composition)
# ---------------------------------------------------------------------------

FULL_VISIBILITY: tuple[v02.LogPos, v02.LogPos, v02.LogPos] = (v02.LogPos(0), v02.LogPos(0), v02.LogPos(0))
"""Fully visible: ``L(V) = 0`` per channel, the default ``V_i`` of §22.1."""

OPAQUE_VISIBILITY: tuple[v02.LogPos, v02.LogPos, v02.LogPos] = (
    v02.LogPos.zero(),
    v02.LogPos.zero(),
    v02.LogPos.zero(),
)
"""Opaque visibility: the positive zero sentinel, which §24 maps to no term."""


def visibility_channels(value: Any, *, name: str = "visibility") -> tuple[v02.LogPos, v02.LogPos, v02.LogPos]:
    """Normalize one positive visibility factor to three LOG_POS channels.

    Accepts ``None`` (fully visible, ``L = 0``), a non-negative real, a
    :class:`spf.tensor.v02.LogPos`, a one- or three-element sequence of either,
    or a one/three-element :class:`LogPosPlane`.  ``0.0`` becomes the positive
    zero sentinel, which §24 names as opaque.
    """

    if value is None:
        return FULL_VISIBILITY
    if isinstance(value, v02.LogPos):
        return (value, value, value)
    if isinstance(value, LogPosPlane):
        codes = np.asarray(value.L, dtype=np.int64).reshape(-1)
        if codes.size == 1:
            single = v02.LogPos(int(codes[0]))
            return (single, single, single)
        if codes.size != 3:
            raise ValueError(f"{name} plane must hold one or three channels")
        return (v02.LogPos(int(codes[0])), v02.LogPos(int(codes[1])), v02.LogPos(int(codes[2])))
    items: tuple[Any, ...]
    if isinstance(value, (tuple, list, np.ndarray)):
        items = tuple(np.asarray(value, dtype=object).reshape(-1))
    else:
        items = (value,)
    if len(items) not in {1, 3}:
        raise ValueError(f"{name} must be a scalar or a three-channel factor")
    channels = tuple(
        item if isinstance(item, v02.LogPos) else v02.LogPos.from_value(_positive_scalar(item, name=name))
        for item in items
    )
    if any(channel.is_nan for channel in channels):
        raise ValueError(f"{name} must be finite and non-negative")
    if len(channels) == 1:
        return (channels[0], channels[0], channels[0])
    return (channels[0], channels[1], channels[2])


def compose_visibility(*factors: Any, name: str = "visibility") -> tuple[v02.LogPos, v02.LogPos, v02.LogPos]:
    """Compose visibility/attenuation factors per §24: ``L_V = Σ_i L(V_i)``.

    Log-additive, so shadow × fog × transmission × mask is an integer add per
    channel; one opaque factor (zero sentinel) makes the product opaque.
    """

    result = FULL_VISIBILITY
    for factor in factors:
        channels = visibility_channels(factor, name=name)
        result = (
            v02.logpos_mul(result[0], channels[0]),
            v02.logpos_mul(result[1], channels[1]),
            v02.logpos_mul(result[2], channels[2]),
        )
    return result


def visibility_plane(value: Any, *, name: str = "visibility") -> LogPosPlane:
    """The §22.1 ``V_i`` slot as a three-channel :class:`LogPosPlane`."""

    channels = visibility_channels(value, name=name)
    return LogPosPlane(np.asarray([channel.L for channel in channels], dtype=np.int64))


def _visibility_for_light(spec: Any, light: Any) -> Any:
    """Resolve a per-light visibility factor from a shared visibility argument.

    ``spec`` is one factor for every light, a ``{light.name: factor}`` mapping,
    or a callable taking the light and returning a factor.
    """

    if spec is None:
        return None
    if isinstance(spec, Mapping):
        return spec.get(str(getattr(light, "name", "")))
    if callable(spec) and not isinstance(spec, (v02.LogPos, LogPosPlane, np.ndarray)):
        return spec(light)
    return spec


def evaluate_light_contribution(
    light: Any,
    normal: Any,
    view: Any,
    *,
    point: Any = None,
    material: Material | None = None,
    visibility: Any = None,
) -> np.ndarray:
    """Evaluate one positive direct-light contribution."""

    typed = _evaluate_light_contribution_log(
        light, normal, view, point=point, material=material, visibility=visibility
    )
    return np.asarray([float(value.value) for value in typed], dtype=np.float64)


def _evaluate_light_contribution_log(
    light: Any,
    normal: Any,
    view: Any,
    *,
    point: Any = None,
    material: Material | None = None,
    visibility: Any = None,
) -> tuple[v02.LogPos, ...]:
    """Evaluate one direct term while positive factors stay in log space.

    Design §22.1: ``L(T_i) = L(f_r,i) + L(L_i) + L(V_i) + L(G_i)``.  ``V_i`` is
    the visibility/attenuation factor of §24 (log-additive, LOG_POS); it
    defaults to fully visible (``L = 0``) and an opaque factor (zero sentinel)
    removes the term.  ``G_i`` is the geometric cosine term here.
    """

    n = _unit(normal, name="normal")
    view_vector = _unit(view, name="view")
    visibility_log = visibility_channels(visibility)
    direction = light_direction(light, point, fallback=n)
    cosine = max(float(np.dot(n, direction)), 0.0)
    if cosine == 0.0 or all(channel.is_zero for channel in visibility_log):
        return (v02.LogPos.zero(),) * 3
    radiance = sample_light(light, direction, point=point)
    radiance_log = tuple(v02.LogPos.from_value(float(value)) for value in radiance)
    brdf = (
        (v02.LogPos.from_value(1.0),) * 3
        if material is None
        else tuple(
            cast(v02.LogPos, value)
            for value in evaluate_brdf(material, n, view_vector, direction, output="log")
        )
    )
    cosine_log = v02.LogPos.from_value(cosine)
    return tuple(
        v02.logpos_mul(
            v02.logpos_mul(v02.logpos_mul(radiance_log[channel], brdf[channel]), cosine_log),
            visibility_log[channel],
        )
        for channel in range(3)
    )


def evaluate_lighting(
    lights: Iterable[Any],
    normal: Any,
    view: Any,
    *,
    point: Any = None,
    material: Material | None = None,
    remaining_error: float | None = None,
    visibility: Any = None,
) -> LightingResult:
    """Sum direct light terms (§22.1–22.2).

    ``visibility`` is the §22.1 ``V_i`` slot: one factor for every light, a
    ``{light.name: factor}`` mapping, or a callable of the light.  It defaults to
    fully visible (``L = 0``).
    """

    items = tuple(lights)
    source_ledger = _lights_source_ledger(items)
    source_error_bound = float(source_ledger.total_bound)
    if remaining_error is None:
        selected, dropped_bound = items, 0.0
    else:
        budget = float(remaining_error)
        if math.isnan(budget) or budget < 0.0:
            raise ValueError("remaining_error must be non-negative")
        selected, dropped_bound = _cull_lights_with_bound(items, budget, point=point, normal=normal)
    typed_contributions = tuple(
        _evaluate_light_contribution_log(
            light,
            normal,
            view,
            point=point,
            material=material,
            visibility=_visibility_for_light(visibility, light),
        )
        for light in selected
    )
    contributions = tuple(
        np.asarray([float(value.value) for value in term], dtype=np.float64) for term in typed_contributions
    )
    typed_channels = tuple(
        tuple(term[channel] for term in typed_contributions) for channel in range(3)
    )
    reductions = tuple(v02.reduce_logpos(values) for values in typed_channels)
    color = (
        np.asarray([float(reduction.value.value) for reduction in reductions], dtype=np.float64)
        if reductions
        else np.zeros(3, dtype=np.float64)
    )
    return LightingResult(
        color=np.asarray(color, dtype=np.float64),
        contributions=contributions,
        active_lights=len(selected),
        culled_lights=len(items) - len(selected),
        upper_bound=math.fsum(light_upper_bound(light, point=point, normal=normal) for light in items),
        culled_upper_bound=dropped_bound,
        positive_logadds=sum(reduction.counters.positive_logadds for reduction in reductions),
        pivot_reductions=sum(reduction.counters.pivot_reductions for reduction in reductions),
        reduction_error_bound=max(
            (
                reduction.total_error_bound
                + math.fsum(
                    abs(float(typed.value) - float(source[channel]))
                    for typed, source in zip(typed_channels[channel], contributions, strict=True)
                )
                for channel, reduction in enumerate(reductions)
            ),
            default=0.0,
        ),
        source_error_bound=source_error_bound,
        source_ledger=source_ledger,
    )


def positive_light_sum(values: Iterable[Any]) -> v02.LogPos:
    """Sum positive light terms through the canonical positive-log reducer."""

    return log_sum(values)


# ---------------------------------------------------------------------------
# Plane (word-native) lighting: design §22 on LogPosPlane carriers
# ---------------------------------------------------------------------------


def light_radiance_plane(light: Any, *, point: Any = None, direction: Any = None) -> tuple[LogPosPlane, str, int]:
    """Radiance of one light as a ``LogPosPlane`` of shape ``(3,)``.

    Returns ``(plane, ingress, conversions)``.  A :class:`DirectionalLight`
    whose colour/intensity are compact block Fields is read as packed words
    (``ingress="addressed_block_words"``, zero conversions).  Native constants
    are encoded (``"native_scalar"``).  Any other source is sampled through
    the float path and encoded, and counts one conversion per block source.
    """

    if isinstance(light, DirectionalLight):
        block_sources = [source for source in (light.color_source, light.intensity_source) if source is not None]
        try:
            color, color_ingress = (
                as_log_pos_plane(light.color_source, name="light color", channels=3)
                if light.color_source is not None
                else (LogPosPlane.from_values(light.color_array), "native_scalar")
            )
            intensity, intensity_ingress = (
                as_log_pos_plane(light.intensity_source, name="light intensity", channels=3)
                if light.intensity_source is not None
                else (LogPosPlane.from_values(np.repeat(float(light.intensity), 3)), "native_scalar")
            )
        except ValueError:
            radiance = LogPosPlane.from_values(light.radiance_at(point))
            return radiance, "addressed_block_to_float64", len(block_sources)
        word_native = all(kind != "addressed_block_to_float64" for kind in (color_ingress, intensity_ingress))
        ingress = "addressed_block_words" if block_sources and word_native else "native_scalar"
        return color.mul(intensity), ingress, 0
    values = sample_light(light, direction if direction is not None else light_direction(light, point), point=point)
    return LogPosPlane.from_values(np.asarray(values, dtype=np.float64).reshape(3)), "native_scalar", 0


@dataclass(frozen=True, slots=True)
class LightingPlaneResult:
    """Direct lighting evaluated and reduced on LOG_POS word planes."""

    color: LogPosPlane
    contributions: LogPosPlane
    reduction: PlaneReduction
    active_lights: int
    culled_lights: int
    culled_upper_bound: float
    counters: v02.PerformanceCounters
    consumer: str
    source_ingress: tuple[str, ...]
    source_error_bound: float = 0.0
    source_ledger: v02.ErrorLedger = field(default_factory=v02.ErrorLedger)

    @property
    def color_words(self) -> np.ndarray:
        return self.color.to_words()

    @property
    def fallback_conversions(self) -> int:
        return int(self.counters.fallback_conversions)

    @property
    def reduction_error_bound(self) -> float:
        return float(self.reduction.total_error_bound)

    @property
    def ledger(self) -> v02.ErrorLedger:
        return self.source_ledger.plus(v02.ErrorLedger(reducer=self.reduction_error_bound))

    def display(self) -> np.ndarray:
        """Float RGB through the terminal magnitude ROM (no decoder call)."""

        from spf.renderer.transport_terminal import display_values

        return np.asarray(display_values(self.color), dtype=np.float64).reshape(3)

    def decode(self, lights: Iterable[Any] = (), *, point: Any = None, normal: Any = None) -> LightingResult:
        """Project to the float :class:`LightingResult` (one output boundary)."""

        items = tuple(lights)
        return LightingResult(
            color=self.color.to_values().reshape(3),
            contributions=tuple(self.contributions.to_values().reshape(-1, 3)),
            active_lights=self.active_lights,
            culled_lights=self.culled_lights,
            upper_bound=math.fsum(light_upper_bound(light, point=point, normal=normal) for light in items),
            culled_upper_bound=self.culled_upper_bound,
            positive_logadds=int(self.counters.positive_logadds),
            pivot_reductions=int(self.counters.pivot_reductions),
            reduction_error_bound=self.reduction_error_bound,
            source_error_bound=self.source_error_bound,
            source_ledger=self.source_ledger,
        )


def evaluate_lighting_planes(
    lights: Iterable[Any],
    normal: Any,
    view: Any,
    *,
    point: Any = None,
    material: Material | None = None,
    remaining_error: float | None = None,
    visibility: Any = None,
) -> LightingPlaneResult:
    """Evaluate direct lighting with every term and the reduction on word planes.

    Per light the contribution is ``radiance * brdf * cosine * visibility`` as
    ``LogPosPlane.mul`` calls (design §22.1, with ``V_i`` from §24); the lights
    are stacked on a leading axis and reduced once per channel by
    :func:`spf.tensor.v02_planes.reduce_logpos` (the positive pivot reducer,
    bit-identical to ``v02.reduce_logpos``).  Block-backed lights are read as
    packed words and never decoded.  ``visibility`` takes the same forms as in
    :func:`evaluate_lighting` and defaults to fully visible (``L = 0``).
    """

    items = tuple(lights)
    source_ledger = _lights_source_ledger(items)
    source_error_bound = float(source_ledger.total_bound)
    if remaining_error is None:
        selected, dropped_bound = items, 0.0
    else:
        budget = float(remaining_error)
        if math.isnan(budget) or budget < 0.0:
            raise ValueError("remaining_error must be non-negative")
        selected, dropped_bound = _cull_lights_with_bound(items, budget, point=point, normal=normal)
    n = _unit(normal, name="normal")
    view_vector = _unit(view, name="view")
    counters = v02.PerformanceCounters()
    ingress: list[str] = []
    terms: list[LogPosPlane] = []
    for light in selected:
        direction = light_direction(light, point, fallback=n)
        cosine = max(float(np.dot(n, direction)), 0.0)
        factor = _visibility_for_light(visibility, light)
        visible = None if factor is None else visibility_plane(factor)
        if cosine == 0.0 or (visible is not None and bool(np.all(visible.is_zero))):
            terms.append(LogPosPlane.zeros((3,)))
            continue
        radiance, kind, conversions = light_radiance_plane(light, point=point, direction=direction)
        ingress.append(kind)
        counters.fallback_conversions += int(conversions)
        brdf = (
            LogPosPlane.ones((3,))
            if material is None
            else cast(LogPosPlane, material.brdf_plane(n, view_vector, direction))
        )
        cosine_plane = LogPosPlane.from_values(np.repeat(cosine, 3))
        term = radiance.mul(brdf).mul(cosine_plane)
        counters.log_ops += 6
        if visible is not None:
            term = term.mul(visible)
            counters.log_ops += 3
        terms.append(term)
    contributions = (
        LogPosPlane(np.stack([term.L for term in terms])) if terms else LogPosPlane.zeros((0, 3))
    )
    reduction = reduce_logpos(contributions, axis=0)
    counters.pivot_reductions += int(reduction.counters.pivot_reductions)
    counters.positive_logadds += int(reduction.counters.positive_logadds)
    counters.aligned_mantissas += int(reduction.counters.aligned_mantissas)
    counters.terms_seen += int(reduction.counters.terms_seen)
    return LightingPlaneResult(
        color=cast(LogPosPlane, reduction.value),
        contributions=contributions,
        reduction=reduction,
        active_lights=len(selected),
        culled_lights=len(items) - len(selected),
        culled_upper_bound=dropped_bound,
        counters=counters,
        consumer="spf.tensor.v02_planes.reduce_logpos",
        source_ingress=tuple(sorted(set(ingress))),
        source_error_bound=source_error_bound,
        source_ledger=source_ledger,
    )


__all__ = [
    "FULL_VISIBILITY",
    "OPAQUE_VISIBILITY",
    "DirectionalLight",
    "Light",
    "LightCluster",
    "LightingPlaneResult",
    "LightingResult",
    "SphericalGaussian",
    "SphericalGaussianLight",
    "cluster_lights",
    "compose_visibility",
    "cull_light_cluster",
    "cull_lights",
    "evaluate_light_contribution",
    "evaluate_lighting",
    "evaluate_lighting_planes",
    "light_direction",
    "light_radiance_plane",
    "light_upper_bound",
    "positive_light_sum",
    "sample_light",
    "spherical_gaussian_integral",
    "spherical_gaussian_light",
    "spherical_gaussian_product",
    "visibility_channels",
    "visibility_plane",
]
