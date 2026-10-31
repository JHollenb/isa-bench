"""Spectral texture assets for the SFR v0.2 reference renderer.

The asset format in this module is intentionally small and boring.  A texture
is split into spatial blocks and every block owns one selected representation
plus the certificate produced while choosing it.  The representations are
the ones used by the v0.2 design: a dense SPF spectrum, a Hermitian half
spectrum for real data, an SVD/separable factorisation, a quadratic jet, or a
Gaussian--chirp atom.

The on-disk format is a deterministic header followed by sorted NPY payloads.
It is not a pickle container and loading always uses ``allow_pickle=False``.
This makes assets suitable for caches and for byte-for-byte reproducibility
tests without making the format depend on Python object graphs.
"""

from __future__ import annotations

import io
import json
import math
import struct
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.carriers import decode_words, dft_centered, dft_frequencies, encode_words
from spf.renderer.model import (
    Coherence,
    CoordinateChart,
    Domain,
    ErrorBudget,
    ErrorContract,
    Field,
    FieldRepresentation,
    PrecisionContract,
    Region,
    RepresentationKind,
    Semantic,
    Support,
)
from spf.renderer.report import RepresentationDecision
from spf.renderer.v02 import FieldKind
from spf.tensor import v02


class TextureRepresentation(StrEnum):
    """Block representations considered by the asset planner."""

    DENSE_SPF = "DENSE_SPF"
    HERMITIAN_HALF = "HERMITIAN_HALF"
    SVD_SEPARABLE = "SVD_SEPARABLE"
    QUADRATIC_JET = "QUADRATIC_JET"
    GAUSSIAN_CHIRP = "GAUSSIAN_CHIRP"


Representation = TextureRepresentation


_REPRESENTATION_KIND: dict[TextureRepresentation, RepresentationKind] = {
    TextureRepresentation.DENSE_SPF: RepresentationKind.DENSE_TYPED,
    TextureRepresentation.HERMITIAN_HALF: RepresentationKind.HERMITIAN_HALF,
    TextureRepresentation.SVD_SEPARABLE: RepresentationKind.SEPARABLE,
    TextureRepresentation.QUADRATIC_JET: RepresentationKind.QUADRATIC_JET,
    TextureRepresentation.GAUSSIAN_CHIRP: RepresentationKind.GAUSSIAN_CHIRP_ATOM,
}

_MAGIC = b"SFRSFT2\0"
_VERSION = 1
_HEADER = struct.Struct("<8sIQ")
_LENGTH = struct.Struct("<Q")


def _representation(value: TextureRepresentation | RepresentationKind | str) -> TextureRepresentation:
    if isinstance(value, TextureRepresentation):
        return value
    if isinstance(value, RepresentationKind):
        kind_aliases: dict[RepresentationKind, TextureRepresentation] = {
            RepresentationKind.DENSE_TYPED: TextureRepresentation.DENSE_SPF,
            RepresentationKind.HERMITIAN_HALF: TextureRepresentation.HERMITIAN_HALF,
            RepresentationKind.SEPARABLE: TextureRepresentation.SVD_SEPARABLE,
            RepresentationKind.QUADRATIC_JET: TextureRepresentation.QUADRATIC_JET,
            RepresentationKind.GAUSSIAN_CHIRP_ATOM: TextureRepresentation.GAUSSIAN_CHIRP,
        }
        if value in kind_aliases:
            return kind_aliases[value]
    name = str(getattr(value, "value", value)).strip().upper().replace("-", "_").replace(" ", "_")
    name_aliases: dict[str, TextureRepresentation] = {
        "DENSE": TextureRepresentation.DENSE_SPF,
        "DENSE_TYPED": TextureRepresentation.DENSE_SPF,
        "SPF": TextureRepresentation.DENSE_SPF,
        "SPF32": TextureRepresentation.DENSE_SPF,
        "DENSE_SPF": TextureRepresentation.DENSE_SPF,
        "HERMITIAN": TextureRepresentation.HERMITIAN_HALF,
        "HERMITIAN_HALF": TextureRepresentation.HERMITIAN_HALF,
        "SVD": TextureRepresentation.SVD_SEPARABLE,
        "SEPARABLE": TextureRepresentation.SVD_SEPARABLE,
        "SVD_SEPARABLE": TextureRepresentation.SVD_SEPARABLE,
        "JET": TextureRepresentation.QUADRATIC_JET,
        "QUADRATIC_JET": TextureRepresentation.QUADRATIC_JET,
        "ATOM": TextureRepresentation.GAUSSIAN_CHIRP,
        "GAUSSIAN_CHIRP_ATOM": TextureRepresentation.GAUSSIAN_CHIRP,
        "GAUSSIAN_CHIRP": TextureRepresentation.GAUSSIAN_CHIRP,
    }
    try:
        return name_aliases[name]
    except KeyError as exc:
        raise ValueError(f"unknown texture representation {value!r}") from exc


def _complex_pair(value: complex) -> list[float]:
    number = complex(value)
    return [float(number.real), float(number.imag)]


def _complex_value(value: Any) -> complex:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 2:
        raise ValueError("complex scalar metadata must contain [real, imag]")
    return complex(float(array[0]), float(array[1]))


def _json_value(value: Any) -> Any:
    """Convert metadata to a canonical JSON-compatible value."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("texture metadata must contain finite floats")
        return number
    if isinstance(value, (complex, np.complexfloating)):
        return {"__complex__": _complex_pair(complex(value))}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "c" and value.ndim == 0:
            return {"__complex__": _complex_pair(complex(value.item()))}
        return [_json_value(item) for item in value.tolist()]
    raise TypeError(f"texture metadata value {type(value).__name__} is not serializable")


def _restore_json(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) == {"__complex__"}:
        return _complex_value(value["__complex__"])
    if isinstance(value, Mapping):
        return {str(key): _restore_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_json(item) for item in value]
    return value


def _budget_parts(
    budget: ErrorBudget | float | None,
) -> tuple[float, float]:
    if budget is None:
        # ``None`` means choose a useful representation but retain a complete
        # certificate.  It is deliberately not a license to hide error.
        return math.inf, math.inf
    if isinstance(budget, ErrorBudget):
        return float(budget.absolute), float(budget.relative)
    value = float(budget)
    if math.isnan(value) or value < 0.0:
        raise ValueError("texture error budget must be non-negative")
    return value, math.inf


def _within_budget(error: float, scale: float, budget: tuple[float, float]) -> bool:
    absolute, relative = budget
    return (not math.isfinite(absolute) or error <= absolute + 1.0e-12 * max(1.0, absolute)) and (
        not math.isfinite(relative)
        or error / max(abs(scale), np.finfo(np.float64).tiny) <= relative + 1.0e-12 * max(1.0, relative)
    )


def _nbytes(payload: Any) -> int:
    if isinstance(payload, np.ndarray):
        return int(payload.nbytes)
    if isinstance(payload, Mapping):
        return sum(_nbytes(key) + _nbytes(value) for key, value in payload.items())
    if isinstance(payload, (tuple, list)):
        return sum(_nbytes(value) for value in payload)
    return 0


@dataclass(frozen=True, slots=True)
class RepresentationCandidate:
    """One fitted representation and its a-posteriori certificate."""

    kind: TextureRepresentation
    payload: Any
    error_bound: float
    relative_error_bound: float
    byte_size: int
    evaluation_cost: float
    materialization_cost: float = 0.0
    certified: bool = True
    reason: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _representation(self.kind))
        for name in ("error_bound", "relative_error_bound", "evaluation_cost", "materialization_cost"):
            value = float(getattr(self, name))
            if math.isnan(value) or value < 0.0:
                raise ValueError(f"candidate {name} must be non-negative")
            object.__setattr__(self, name, value)
        size = int(self.byte_size)
        if size < 0:
            raise ValueError("candidate byte_size must be non-negative")
        object.__setattr__(self, "byte_size", size)
        object.__setattr__(self, "certified", bool(self.certified))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def representation_kind(self) -> RepresentationKind:
        return _REPRESENTATION_KIND[self.kind]

    @property
    def error(self) -> float:
        return self.error_bound

    @property
    def cost(self) -> float:
        return float(self.byte_size) / 64.0 + self.evaluation_cost + self.materialization_cost

    def as_field_representation(self) -> FieldRepresentation:
        return FieldRepresentation(
            self.representation_kind,
            self.payload,
            exactness="BOUNDED" if self.error_bound else "EXACT",
            byte_size=self.byte_size,
            evaluation_cost=self.evaluation_cost,
            materialization_cost=self.materialization_cost,
            metadata={"certified": self.certified, "reason": self.reason, **dict(self.metadata)},
        )


@dataclass(frozen=True, slots=True)
class TextureBlock:
    """A block-local selected representation."""

    index: tuple[int, int]
    origin: tuple[int, int]
    shape: tuple[int, int]
    representation: TextureRepresentation
    payload: Any
    error_bound: float = 0.0
    relative_error_bound: float = 0.0
    candidates: tuple[RepresentationCandidate, ...] = ()
    certified: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        index = tuple(int(value) for value in self.index)
        origin = tuple(int(value) for value in self.origin)
        shape = tuple(int(value) for value in self.shape)
        if len(index) != 2 or len(origin) != 2 or len(shape) != 2 or any(value <= 0 for value in shape):
            raise ValueError("texture blocks require two-dimensional positive shapes")
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "representation", _representation(self.representation))
        for name in ("error_bound", "relative_error_bound"):
            value = float(getattr(self, name))
            if math.isnan(value) or value < 0.0:
                raise ValueError(f"texture block {name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "certified", bool(self.certified))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def kind(self) -> TextureRepresentation:
        return self.representation

    @property
    def byte_size(self) -> int:
        return _nbytes(self.payload)

    @property
    def representation_kind(self) -> RepresentationKind:
        return _REPRESENTATION_KIND[self.representation]

    @property
    def field_representation(self) -> FieldRepresentation:
        return FieldRepresentation(
            self.representation_kind,
            self.payload,
            exactness="BOUNDED" if self.error_bound else "EXACT",
            error=v02.ErrorLedger(representation_fit=self.error_bound),
            byte_size=self.byte_size,
            metadata={"certified": self.certified, **dict(self.metadata)},
        )

    def reconstruct(self) -> np.ndarray:
        """Reconstruct this block in its native complex/real value domain."""

        shape = self.shape
        payload = self.payload
        result: np.ndarray
        if self.representation is TextureRepresentation.DENSE_SPF:
            spectrum = np.asarray(decode_words(np.asarray(payload["spectrum"], dtype=np.uint32)))
            result = np.fft.ifftn(spectrum, s=shape, axes=tuple(range(len(shape))))
        elif self.representation is TextureRepresentation.HERMITIAN_HALF:
            half = np.asarray(decode_words(np.asarray(payload["spectrum"], dtype=np.uint32)))
            result = np.fft.irfftn(half, s=shape, axes=tuple(range(len(shape))))
        elif self.representation is TextureRepresentation.SVD_SEPARABLE:
            left = np.asarray(payload["U"])
            singular = np.asarray(payload["s"])
            right = np.asarray(payload["Vh"])
            result = (left * singular[np.newaxis, :]) @ right
        elif self.representation is TextureRepresentation.QUADRATIC_JET:
            result = v02.jet_forward_differences(payload["jet"], shape)
        elif self.representation is TextureRepresentation.GAUSSIAN_CHIRP:
            points = _grid_points(shape)
            result = np.asarray(payload["atom"].evaluate(points), dtype=np.complex128).reshape(shape)
        else:  # pragma: no cover - enum validation makes this unreachable
            raise ValueError(f"unsupported texture block representation {self.representation!r}")
        result = np.asarray(result)
        if bool(self.metadata.get("real_output", False)):
            return np.asarray(result.real, dtype=np.float64)
        return np.asarray(result, dtype=np.complex128)

    materialize = reconstruct

    def to_field(self, *, field_id: str | None = None) -> Field:
        """Expose the block through the metadata-rich renderer field model."""

        identifier = field_id or f"texture_block_{self.index[0]}_{self.index[1]}"
        return Field(
            id=identifier,
            domain=Domain.SPATIAL_2D,
            semantic=Semantic.GENERIC_FIELD,
            coherence=Coherence.COHERENT_COMPLEX,
            value_kind=FieldKind.SPF_COMPLEX,
            representation=self.field_representation,
            chart=CoordinateChart.identity(2),
            support=Support.aabb(
                self.origin,
                tuple(origin + size for origin, size in zip(self.origin, self.shape, strict=False)),
            ),
            error=ErrorContract(
                absolute_max=self.error_bound,
                relative_max=self.relative_error_bound,
            ),
            precision=PrecisionContract(),
        )

    def typed_block_view(self) -> Any | None:
        """Return this block as an addressed SPF carrier when one is available.

        ``DENSE_SPF`` stores the block's already-quantised spectrum as packed
        SPF32 words.  The typed renderer can consume those words through the
        existing :class:`TensorBlockView` contract, so ingress only separates
        the packed log/phase planes; it never reconstructs a Cartesian block.
        Other texture representations remain owned by the texture/reference
        consumer until they gain a representation-specific block adapter.
        """

        from spf.renderer.carriers import as_block_view

        payload = self.payload
        candidate = payload
        if isinstance(payload, Mapping):
            for key in ("block", "carrier", "tensor_block", "value"):
                if key in payload:
                    candidate = payload[key]
                    break
        view = as_block_view(candidate)
        if view is not None:
            return view
        if self.representation is not TextureRepresentation.DENSE_SPF or not isinstance(payload, Mapping):
            return None
        spectrum = payload.get("spectrum")
        if spectrum is None:
            return None
        words = np.asarray(spectrum, dtype=np.uint32)
        if words.size != int(np.prod(self.shape, dtype=np.int64)):
            return None
        return _packed_spectrum_tensor_view(words.reshape(self.shape), self.shape)


def _grid_points(shape: tuple[int, int], origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    y, x = np.meshgrid(
        np.arange(shape[0], dtype=np.float64) + float(origin[0]),
        np.arange(shape[1], dtype=np.float64) + float(origin[1]),
        indexing="ij",
    )
    return np.stack((x, y), axis=-1)


def _packed_spectrum_tensor_view(words: np.ndarray, shape: tuple[int, int]) -> Any:
    """Adapt one packed SPF32 spectrum to the existing addressed block ABI.

    The conversion is between packed word storage and the v2 residual planes;
    it does not call ``decode_words`` or form a Cartesian value array.  The
    renderer's normal ``TensorBlockView`` then owns the carrier and serves
    addressed ``sample_words`` reads to the analytic filter terminal.
    """

    from spf.renderer.blocks import TensorBlockView
    from spf.tensor.core import unpack_L, unpack_theta
    from spf.tensor.v2 import TensorBlock, TensorBlockHeaderV2

    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    phase = np.asarray(unpack_theta(packed), dtype=np.uint16).view(np.int16)
    tensor = TensorBlock(
        TensorBlockHeaderV2(dims=tuple(int(value) for value in shape)),
        dL=np.asarray(unpack_L(packed), dtype=np.int16),
        dT=phase,
    )
    return TensorBlockView.from_block(tensor)


def _candidate_error(source: np.ndarray, reconstructed: Any) -> tuple[float, float]:
    source_array = np.asarray(source, dtype=np.complex128)
    difference = np.abs(source_array - np.asarray(reconstructed, dtype=np.complex128))
    error = float(np.max(difference)) if difference.size else 0.0
    scale = float(np.max(np.abs(source_array))) if source_array.size else 1.0
    return error, error / max(scale, np.finfo(np.float64).tiny)


def _dense_candidate(source: np.ndarray) -> RepresentationCandidate:
    spectrum = np.fft.fftn(np.asarray(source, dtype=np.complex128))
    words = np.asarray(encode_words(spectrum), dtype=np.uint32)
    axes = tuple(range(source.ndim))
    reconstructed = np.fft.ifftn(np.asarray(decode_words(words)), s=source.shape, axes=axes)
    error, relative = _candidate_error(source, reconstructed)
    return RepresentationCandidate(
        TextureRepresentation.DENSE_SPF,
        {"spectrum": words},
        error,
        relative,
        int(words.nbytes),
        evaluation_cost=float(source.size),
        materialization_cost=float(source.size),
        reason="full SPF32 spectrum",
        metadata={"real_output": bool(np.isrealobj(source))},
    )


def _hermitian_candidate(source: np.ndarray) -> RepresentationCandidate | None:
    if not np.isrealobj(source):
        return None
    half = np.fft.rfftn(np.asarray(source, dtype=np.float64))
    words = np.asarray(encode_words(half), dtype=np.uint32)
    axes = tuple(range(source.ndim))
    reconstructed = np.fft.irfftn(np.asarray(decode_words(words)), s=source.shape, axes=axes)
    error, relative = _candidate_error(source, reconstructed)
    return RepresentationCandidate(
        TextureRepresentation.HERMITIAN_HALF,
        {"spectrum": words},
        error,
        relative,
        int(words.nbytes),
        evaluation_cost=float(words.size),
        materialization_cost=float(source.size),
        reason="real input: store the non-redundant rFFT half-spectrum",
        metadata={"real_output": True, "full_spectrum_shape": list(source.shape)},
    )


def _svd_candidate(source: np.ndarray, rank: int | None = None) -> RepresentationCandidate:
    matrix = np.asarray(source, dtype=np.complex128)
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    if rank is None:
        rank = int(matrix.shape[0] if matrix.shape[0] < matrix.shape[1] else matrix.shape[1])
    rank = max(1, min(int(rank), singular.size))
    payload = {"U": left[:, :rank], "s": singular[:rank], "Vh": right[:rank, :]}
    reconstructed = (payload["U"] * payload["s"][np.newaxis, :]) @ payload["Vh"]
    error, relative = _candidate_error(source, reconstructed)
    return RepresentationCandidate(
        TextureRepresentation.SVD_SEPARABLE,
        payload,
        error,
        relative,
        _nbytes(payload),
        evaluation_cost=float(rank * matrix.shape[0] * matrix.shape[1]),
        materialization_cost=float(matrix.size),
        reason=f"rank-{rank} SVD separable field",
        metadata={"rank": rank, "real_output": bool(np.isrealobj(source))},
    )


def _fit_jet_candidate(source: np.ndarray) -> RepresentationCandidate | None:
    height, width = source.shape
    points = _grid_points((height, width)).reshape(-1, 2)
    design = np.column_stack(
        (
            np.ones(points.shape[0]),
            points[:, 0],
            points[:, 1],
            points[:, 0] ** 2,
            points[:, 0] * points[:, 1],
            points[:, 1] ** 2,
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, np.asarray(source, dtype=np.complex128).reshape(-1), rcond=None)
    jet = v02.Jet2D(
        value=coefficients[0],
        grad_x=coefficients[1],
        grad_y=coefficients[2],
        hess_xx=2.0 * coefficients[3],
        hess_xy=coefficients[4],
        hess_yy=2.0 * coefficients[5],
    )
    reconstructed = v02.jet_forward_differences(jet, source.shape)
    error, relative = _candidate_error(source, reconstructed)
    scale = max(float(np.max(np.abs(source))), 1.0)
    if not math.isfinite(error):
        return None
    return RepresentationCandidate(
        TextureRepresentation.QUADRATIC_JET,
        {"jet": jet},
        error,
        relative,
        6 * 16,
        evaluation_cost=6.0,
        materialization_cost=float(source.size),
        reason="full quadratic value/gradient/Hessian fit",
        metadata={"real_output": bool(np.isrealobj(source)), "fit_scale": scale},
    )


def _fit_atom_candidate(source: np.ndarray) -> RepresentationCandidate | None:
    if source.size < 6 or np.any(np.abs(np.asarray(source)) <= np.finfo(np.float64).tiny):
        return None
    points = _grid_points(source.shape).reshape(-1, 2)
    values = np.asarray(source, dtype=np.complex128).reshape(-1)
    amplitude = np.log(np.abs(values))
    phase = np.unwrap(np.unwrap(np.angle(values).reshape(source.shape), axis=0), axis=1).reshape(-1)
    design = np.column_stack(
        (
            np.ones(points.shape[0]),
            points[:, 0],
            points[:, 1],
            points[:, 0] ** 2,
            points[:, 0] * points[:, 1],
            points[:, 1] ** 2,
        )
    )
    amplitude_coefficients, _, _, _ = np.linalg.lstsq(design, amplitude, rcond=None)
    phase_coefficients, _, _, _ = np.linalg.lstsq(design, phase, rcond=None)
    try:
        atom = v02.GaussianChirpAtom.from_arrays(
            amplitude_coefficients[0] + 1j * phase_coefficients[0],
            np.asarray(
                [
                    amplitude_coefficients[1] + 1j * phase_coefficients[1],
                    amplitude_coefficients[2] + 1j * phase_coefficients[2],
                ],
                dtype=np.complex128,
            ),
            np.asarray(
                [
                    [
                        -2.0 * amplitude_coefficients[3] - 2j * phase_coefficients[3],
                        -amplitude_coefficients[4] - 1j * phase_coefficients[4],
                    ],
                    [
                        -amplitude_coefficients[4] - 1j * phase_coefficients[4],
                        -2.0 * amplitude_coefficients[5] - 2j * phase_coefficients[5],
                    ],
                ],
                dtype=np.complex128,
            ),
        )
    except ValueError:
        # A least-squares quadratic is not guaranteed to be an integrable
        # Gaussian.  Such a block simply has no legal atom candidate.
        return None
    if not atom.is_decaying:
        return None
    reconstructed = np.asarray(atom.evaluate(points), dtype=np.complex128).reshape(source.shape)
    error, relative = _candidate_error(source, reconstructed)
    return RepresentationCandidate(
        TextureRepresentation.GAUSSIAN_CHIRP,
        {"atom": atom},
        error,
        relative,
        ((1 + 2 + 3) * 16),
        evaluation_cost=24.0,
        materialization_cost=float(source.size),
        reason="quadratic log-magnitude and unwrapped phase fit",
        metadata={"real_output": bool(np.isrealobj(source))},
    )


def _fit_block_candidates(
    source: np.ndarray,
    *,
    error_budget: ErrorBudget | float | None,
    svd_rank: int | None = None,
) -> tuple[RepresentationCandidate, ...]:
    candidates: list[RepresentationCandidate] = [_dense_candidate(source)]
    hermitian = _hermitian_candidate(source)
    if hermitian is not None:
        candidates.append(hermitian)
    candidates.append(_svd_candidate(source, svd_rank))
    jet = _fit_jet_candidate(source)
    if jet is not None:
        candidates.append(jet)
    atom = _fit_atom_candidate(source)
    if atom is not None:
        candidates.append(atom)
    budget = _budget_parts(error_budget)
    scale = max(float(np.max(np.abs(source))) if source.size else 1.0, np.finfo(np.float64).tiny)
    normalized: list[RepresentationCandidate] = []
    for candidate in candidates:
        normalized.append(
            RepresentationCandidate(
                candidate.kind,
                candidate.payload,
                candidate.error_bound,
                candidate.relative_error_bound,
                candidate.byte_size,
                candidate.evaluation_cost,
                candidate.materialization_cost,
                certified=_within_budget(candidate.error_bound, scale, budget),
                reason=candidate.reason,
                metadata=candidate.metadata,
            )
        )
    return tuple(normalized)


def _select_candidate(
    candidates: Sequence[RepresentationCandidate],
    *,
    error_budget: ErrorBudget | float | None,
    preferred: TextureRepresentation | str | None = None,
) -> RepresentationCandidate:
    if not candidates:
        raise ValueError("at least one texture representation candidate is required")
    if preferred is not None:
        selected_kind = _representation(preferred)
        preferred_candidates = [candidate for candidate in candidates if candidate.kind is selected_kind]
        if preferred_candidates:
            certified = [candidate for candidate in preferred_candidates if candidate.certified]
            return min(certified or preferred_candidates, key=lambda candidate: candidate.cost)
    certified = [candidate for candidate in candidates if candidate.certified]
    if certified:
        return min(certified, key=lambda candidate: (candidate.cost, candidate.error_bound, candidate.kind.value))
    # No candidate can satisfy the requested contract.  Preserve the native
    # SPF fallback and its measured certificate rather than silently dropping
    # the block.
    dense = [candidate for candidate in candidates if candidate.kind is TextureRepresentation.DENSE_SPF]
    return dense[0] if dense else min(candidates, key=lambda candidate: candidate.cost)


def _block_from_candidate(
    index: tuple[int, int],
    origin: tuple[int, int],
    source: np.ndarray,
    candidate: RepresentationCandidate,
    candidates: Sequence[RepresentationCandidate],
) -> TextureBlock:
    return TextureBlock(
        index=index,
        origin=origin,
        shape=(int(source.shape[0]), int(source.shape[1])),
        representation=candidate.kind,
        payload=candidate.payload,
        error_bound=candidate.error_bound,
        relative_error_bound=candidate.relative_error_bound,
        candidates=tuple(candidates),
        certified=candidate.certified,
        metadata={"real_output": bool(np.isrealobj(source)), "reason": candidate.reason, **dict(candidate.metadata)},
    )


@dataclass(frozen=True, slots=True)
class SpectralTexture:
    """A block-structured SPF-native texture asset."""

    shape: tuple[int, int]
    block_shape: tuple[int, int]
    blocks: tuple[TextureBlock, ...]
    channel_semantics: str = "GENERIC_FIELD"
    color_space: str = "linear"
    real: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_error_bound: float = 0.0
    global_atom: v02.GaussianChirpAtom | None = None

    def __post_init__(self) -> None:
        shape = tuple(int(value) for value in self.shape)
        block_shape = tuple(int(value) for value in self.block_shape)
        if len(shape) != 2 or len(block_shape) != 2 or any(value <= 0 for value in shape + block_shape):
            raise ValueError("SpectralTexture requires positive two-dimensional shapes")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "block_shape", block_shape)
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "channel_semantics", str(self.channel_semantics))
        object.__setattr__(self, "color_space", str(self.color_space))
        object.__setattr__(self, "real", bool(self.real))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        bound = float(self.source_error_bound)
        if math.isnan(bound) or bound < 0.0:
            raise ValueError("source_error_bound must be non-negative")
        object.__setattr__(self, "source_error_bound", bound)
        if self.global_atom is not None and not isinstance(self.global_atom, v02.GaussianChirpAtom):
            raise TypeError("global_atom provenance must be a GaussianChirpAtom")

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.shape

    @property
    def block_grid(self) -> tuple[int, int]:
        rows, columns = (math.ceil(size / block) for size, block in zip(self.shape, self.block_shape, strict=True))
        return rows, columns

    @property
    def representation(self) -> tuple[TextureRepresentation, ...]:
        return tuple(block.representation for block in self.blocks)

    @property
    def representation_kinds(self) -> tuple[RepresentationKind, ...]:
        return tuple(block.representation_kind for block in self.blocks)

    @property
    def error_bound(self) -> float:
        return self.source_error_bound + math.fsum(block.error_bound for block in self.blocks)

    @property
    def byte_size(self) -> int:
        return sum(block.byte_size for block in self.blocks)

    @classmethod
    def from_array(
        cls,
        values: Any,
        *,
        block_shape: Sequence[int] | None = None,
        error_budget: ErrorBudget | float | None = None,
        representation: TextureRepresentation | str | None = None,
        svd_rank: int | None = None,
        channel_semantics: str = "GENERIC_FIELD",
        color_space: str = "linear",
        metadata: Mapping[str, Any] | None = None,
    ) -> SpectralTexture:
        """Fit an asset from a real or complex two-dimensional array.

        ``error_budget`` is used to certify compact representations.  With no
        budget the native fallback remains available and every candidate still
        carries its measured reconstruction error.
        """

        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("SpectralTexture.from_array currently requires a two-dimensional array")
        if not np.issubdtype(array.dtype, np.number):
            raise TypeError("texture values must be numeric")
        source = np.asarray(array, dtype=np.complex128 if np.iscomplexobj(array) else np.float64)
        selected_block_shape = tuple(int(value) for value in (block_shape or source.shape))
        if len(selected_block_shape) != 2 or any(value <= 0 for value in selected_block_shape):
            raise ValueError("block_shape must contain two positive dimensions")
        blocks: list[TextureBlock] = []
        rows, columns = (range(0, source.shape[axis], selected_block_shape[axis]) for axis in range(2))
        for row in rows:
            for column in columns:
                row_end = min(row + selected_block_shape[0], source.shape[0])
                column_end = min(column + selected_block_shape[1], source.shape[1])
                block_source = source[row:row_end, column:column_end]
                candidates = _fit_block_candidates(block_source, error_budget=error_budget, svd_rank=svd_rank)
                chosen = _select_candidate(candidates, error_budget=error_budget, preferred=representation)
                blocks.append(
                    _block_from_candidate(
                        (row // selected_block_shape[0], column // selected_block_shape[1]),
                        (row, column),
                        block_source,
                        chosen,
                        candidates,
                    )
                )
        return cls(
            (int(source.shape[0]), int(source.shape[1])),
            selected_block_shape,
            tuple(blocks),
            channel_semantics,
            color_space,
            bool(np.isrealobj(source)),
            metadata or {},
        )

    @classmethod
    def from_atom(
        cls,
        atom: v02.GaussianChirpAtom,
        shape: Sequence[int],
        *,
        block_shape: Sequence[int] | None = None,
        origin: Sequence[float] = (0.0, 0.0),
        channel_semantics: str = "GENERIC_FIELD",
        color_space: str = "linear",
        metadata: Mapping[str, Any] | None = None,
    ) -> SpectralTexture:
        """Create an exact atom-backed texture without fitting it again."""

        if not isinstance(atom, v02.GaussianChirpAtom) or atom.dimension != 2:
            raise TypeError("from_atom requires a two-dimensional GaussianChirpAtom")
        full_shape = tuple(int(value) for value in shape)
        if len(full_shape) != 2 or any(value <= 0 for value in full_shape):
            raise ValueError("shape must contain two positive dimensions")
        selected_block_shape = tuple(int(value) for value in (block_shape or full_shape))
        if len(selected_block_shape) != 2 or any(value <= 0 for value in selected_block_shape):
            raise ValueError("block_shape must contain two positive dimensions")
        origin_array = np.asarray(origin, dtype=np.float64).reshape(-1)
        if origin_array.shape != (2,):
            raise ValueError("origin must be a two-vector")
        blocks: list[TextureBlock] = []
        for row in range(0, full_shape[0], selected_block_shape[0]):
            for column in range(0, full_shape[1], selected_block_shape[1]):
                local_shape = (
                    min(selected_block_shape[0], full_shape[0] - row),
                    min(selected_block_shape[1], full_shape[1] - column),
                )
                # Atom coordinates are conventionally (x, y), while texture
                # block origins are stored in array order (row=y, column=x).
                local_offset = origin_array + np.asarray((column, row), dtype=np.float64)
                local_atom = v02.atom_affine(atom, np.eye(2), local_offset)
                source = np.asarray(local_atom.evaluate(_grid_points(local_shape)), dtype=np.complex128)
                candidate = RepresentationCandidate(
                    TextureRepresentation.GAUSSIAN_CHIRP,
                    {"atom": local_atom},
                    0.0,
                    0.0,
                    (1 + 2 + 3) * 16,
                    24.0,
                    float(np.prod(local_shape)),
                    True,
                    "source supplied as an analytic Gaussian--chirp atom",
                    {"real_output": False},
                )
                blocks.append(
                    _block_from_candidate(
                        (row // selected_block_shape[0], column // selected_block_shape[1]),
                        (row, column),
                        source,
                        candidate,
                        (candidate,),
                    )
                )
        global_atom = v02.atom_affine(atom, np.eye(2), origin_array)
        return cls(
            full_shape,
            selected_block_shape,
            tuple(blocks),
            channel_semantics,
            color_space,
            False,
            {"source_kind": "GAUSSIAN_CHIRP_ATOM", **dict(metadata or {})},
            global_atom=global_atom,
        )

    @classmethod
    def from_field(cls, field: Field, *, block_shape: Sequence[int] | None = None) -> SpectralTexture:
        if not isinstance(field, Field):
            raise TypeError("from_field requires the renderer Field model")
        payload = field.representation.payload
        if isinstance(payload, v02.GaussianChirpAtom):
            shape = field.metadata.get("shape")
            if shape is None:
                raise ValueError("an atom field needs metadata['shape'] to become a texture")
            return cls.from_atom(payload, shape, block_shape=block_shape)
        return cls.from_array(payload, block_shape=block_shape)

    def typed_block_view(self) -> Any | None:
        """Return a single full-size packed block for the typed compiler.

        A local texture block is not a global periodic spectrum when an asset
        has multiple blocks.  The compiler therefore accepts only the exact
        one-block/full-shape case here; all other assets retain the existing
        selective texture consumer and its explicit fallback accounting.
        """

        if len(self.blocks) != 1:
            return None
        block = self.blocks[0]
        if block.origin != (0, 0) or block.shape != self.shape:
            return None
        return block.typed_block_view()

    def reconstruct(self) -> np.ndarray:
        """Materialize the selected block representations once."""

        result = np.empty(self.shape, dtype=np.complex128)
        if self.real:
            result = np.empty(self.shape, dtype=np.float64)
        for block in self.blocks:
            row, column = block.origin
            height, width = block.shape
            result[row : row + height, column : column + width] = block.reconstruct()
        return result

    materialize = reconstruct
    to_array = reconstruct

    def _sample_pixels(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
        *,
        cache: MutableMapping[Any, Any] | None,
        counters: MutableMapping[str, int] | None,
    ) -> np.ndarray:
        """Resolve only blocks containing the requested integer pixels."""

        row_values = np.asarray(rows, dtype=np.int64)
        column_values = np.asarray(columns, dtype=np.int64)
        if row_values.shape != column_values.shape:
            raise ValueError("texture pixel row/column arrays must have identical shapes")
        full = None if cache is None else cache.get(id(self))
        if full is not None:
            return np.asarray(full)[row_values, column_values]

        dtype = np.float64 if self.real else np.complex128
        output = np.empty(row_values.shape, dtype=dtype)
        block_rows = row_values // self.block_shape[0]
        block_columns = column_values // self.block_shape[1]
        block_map = {block.index: block for block in self.blocks}
        pairs = np.stack((block_rows.reshape(-1), block_columns.reshape(-1)), axis=-1)
        for block_row, block_column in np.unique(pairs, axis=0):
            index = (int(block_row), int(block_column))
            block = block_map[index]
            key = (id(self), *index)
            values = None if cache is None else cache.get(key)
            if values is None:
                values = np.asarray(block.reconstruct())
                if cache is not None:
                    cache[key] = values
                if counters is not None:
                    counters["blocks_reconstructed"] = int(counters.get("blocks_reconstructed", 0)) + 1
                    counters["elements_reconstructed"] = int(counters.get("elements_reconstructed", 0)) + int(
                        values.size
                    )
            mask = (block_rows == block_row) & (block_columns == block_column)
            local_rows = row_values[mask] - block.origin[0]
            local_columns = column_values[mask] - block.origin[1]
            output[mask] = values[local_rows, local_columns]
        return output

    def _sample_gaussian_atom_pixels(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
        covariance: np.ndarray,
        *,
        cache: MutableMapping[Any, Any] | None,
        counters: MutableMapping[str, int] | None,
    ) -> np.ndarray:
        """Evaluate an atom-native Gaussian footprint only at requested pixels."""

        if self.global_atom is None:
            raise ValueError("analytic Gaussian footprint requires typed global-atom provenance")
        inverse = np.linalg.inv(covariance)
        normalizer = -math.log(2.0 * math.pi) - 0.5 * math.log(float(np.linalg.det(covariance)))
        kernel = v02.GaussianChirpAtom.from_arrays(normalizer, np.zeros(2), inverse)
        row_values = np.asarray(rows, dtype=np.int64)
        column_values = np.asarray(columns, dtype=np.int64)
        output = np.empty(row_values.shape, dtype=np.complex128)
        covariance_key = covariance.tobytes()
        key = (id(self), "global_gaussian_atom", covariance_key)
        filtered_atom = None if cache is None else cache.get(key)
        if filtered_atom is None:
            filtered_atom = v02.atom_convolve(self.global_atom, kernel)
            if cache is not None:
                cache[key] = filtered_atom
            if counters is not None:
                counters["analytic_blocks_evaluated"] = int(counters.get("analytic_blocks_evaluated", 0)) + 1
        points = np.stack((column_values, row_values), axis=-1).astype(np.float64)
        output[...] = np.asarray(filtered_atom.evaluate(points), dtype=np.complex128)
        if counters is not None:
            counters["elements_evaluated"] = int(counters.get("elements_evaluated", 0)) + int(points.size // 2)
        return output

    def sample(
        self,
        coordinates: Any,
        *,
        normalized: bool = True,
        filter: str = "bilinear",
        address_mode: str = "clamp",
        covariance: Any | None = None,
        lod_cutoff: float | Sequence[float] | None = None,
        cache: MutableMapping[Any, Any] | None = None,
        counters: MutableMapping[str, int] | None = None,
    ) -> Any:
        """Sample scalar texture values at ``(u, v)`` or pixel coordinates.

        The optional cache is owned by the caller (typically one raster pass),
        so the immutable asset remains safe to share.  A covariance applies the
        v0.2 anisotropic Gaussian footprint before sampling; ``lod_cutoff``
        applies support selection with the asset's smooth spectral window.
        """

        points = np.asarray(coordinates, dtype=np.float64)
        if points.ndim == 0 or points.shape[-1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError("texture coordinates must have a finite trailing (u, v) dimension")
        scalar = points.shape == (2,)
        flat = points.reshape(-1, 2)
        selected_filter = str(filter).lower().replace("-", "_")
        if selected_filter not in {"nearest", "bilinear", "linear"}:
            raise ValueError("texture filter must be 'nearest' or 'bilinear'")
        address = str(address_mode).lower().replace("-", "_")
        if address not in {"clamp", "repeat"}:
            raise ValueError("texture address_mode must be 'clamp' or 'repeat'")

        sample_cache: MutableMapping[Any, Any] = {} if cache is None else cache
        values: np.ndarray | None = None
        covariance_matrix = None if covariance is None else np.asarray(covariance, dtype=np.float64)
        atom_native_filter = bool(
            covariance_matrix is not None
            and lod_cutoff is None
            # A block fitted independently from samples is only valid on its
            # own support.  Convolving that unrestricted local atom ignores
            # energy from adjacent blocks.  Analytic footprint closure is
            # therefore legal only when every block is a coordinate view of
            # one declared global atom.  Periodic addressing additionally
            # needs image copies, so it deliberately uses the dense periodic
            # fallback.
            and self.global_atom is not None
            and address == "clamp"
            and covariance_matrix.shape == (2, 2)
            and np.allclose(covariance_matrix, covariance_matrix.T)
            and float(np.min(np.linalg.eigvalsh(covariance_matrix))) > 1.0e-12
            and all(block.representation is TextureRepresentation.GAUSSIAN_CHIRP for block in self.blocks)
        )
        if (covariance is not None or lod_cutoff is not None) and not atom_native_filter:
            covariance_key = None if covariance_matrix is None else covariance_matrix.tobytes()
            cutoff_key = None if lod_cutoff is None else tuple(np.asarray(lod_cutoff, dtype=np.float64).reshape(-1))
            dense_key = (id(self), "dense_filter_fallback", covariance_key, cutoff_key)
            cached_values = sample_cache.get(dense_key)
            if cached_values is None:
                values = np.asarray(self.reconstruct())
                transfer = np.ones(self.shape)
                if covariance is not None:
                    transfer *= self.anisotropic_footprint(covariance_matrix)
                if lod_cutoff is not None:
                    transfer *= self.lod_window(lod_cutoff)
                values = np.fft.ifftn(np.fft.fftn(values) * transfer)
                if self.real:
                    values = np.real_if_close(values, tol=1000)
                values = np.asarray(values)
                values.setflags(write=False)
                sample_cache[dense_key] = values
                if counters is not None:
                    counters["full_reconstructions"] = int(counters.get("full_reconstructions", 0)) + 1
                    counters["blocks_reconstructed"] = int(counters.get("blocks_reconstructed", 0)) + len(self.blocks)
                    counters["elements_reconstructed"] = int(counters.get("elements_reconstructed", 0)) + int(
                        values.size
                    )
                    counters["dense_filter_ffts"] = (
                        int(counters.get("dense_filter_ffts", 0))
                        + 2
                    )
            else:
                values = np.asarray(cached_values)
                if counters is not None:
                    counters["dense_filter_cache_hits"] = int(counters.get("dense_filter_cache_hits", 0)) + 1
            if counters is not None:
                counters["dense_filter_fallbacks"] = int(counters.get("dense_filter_fallbacks", 0)) + 1
        height, width = self.shape
        u = flat[:, 0]
        v = flat[:, 1]
        if normalized:
            if address == "repeat":
                x = np.mod(u, 1.0) * width
                y = np.mod(v, 1.0) * height
            else:
                x = np.clip(u, 0.0, 1.0) * max(0, width - 1)
                y = np.clip(v, 0.0, 1.0) * max(0, height - 1)
        else:
            x, y = u.copy(), v.copy()
            if address == "repeat":
                x, y = np.mod(x, width), np.mod(y, height)
            else:
                x = np.clip(x, 0.0, width - 1)
                y = np.clip(y, 0.0, height - 1)

        if selected_filter == "nearest":
            ix = np.rint(x).astype(np.int64)
            iy = np.rint(y).astype(np.int64)
            if address == "repeat":
                ix, iy = np.mod(ix, width), np.mod(iy, height)
            else:
                ix, iy = np.clip(ix, 0, width - 1), np.clip(iy, 0, height - 1)
            if values is not None:
                result = values[iy, ix]
            elif atom_native_filter:
                assert covariance_matrix is not None
                result = self._sample_gaussian_atom_pixels(
                    iy,
                    ix,
                    covariance_matrix,
                    cache=sample_cache,
                    counters=counters,
                )
            else:
                result = self._sample_pixels(iy, ix, cache=sample_cache, counters=counters)
        else:
            x0 = np.floor(x).astype(np.int64)
            y0 = np.floor(y).astype(np.int64)
            if address == "repeat":
                x1, y1 = np.mod(x0 + 1, width), np.mod(y0 + 1, height)
                x0, y0 = np.mod(x0, width), np.mod(y0, height)
            else:
                x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
            tx, ty = x - np.floor(x), y - np.floor(y)
            if values is not None:
                dense_values = np.asarray(values)

                def sample(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
                    return np.asarray(dense_values[rows, columns])

            elif atom_native_filter:
                assert covariance_matrix is not None

                def sample(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
                    return self._sample_gaussian_atom_pixels(
                        rows,
                        columns,
                        covariance_matrix,
                        cache=sample_cache,
                        counters=counters,
                    )

            else:

                def sample(rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
                    return self._sample_pixels(
                        rows,
                        columns,
                        cache=sample_cache,
                        counters=counters,
                    )

            result = (
                (1.0 - tx) * (1.0 - ty) * sample(y0, x0)
                + tx * (1.0 - ty) * sample(y0, x1)
                + (1.0 - tx) * ty * sample(y1, x0)
                + tx * ty * sample(y1, x1)
            )
        reshaped = np.asarray(result).reshape(points.shape[:-1])
        return reshaped.item() if scalar else reshaped

    def spectrum(self) -> np.ndarray:
        """Return the full dense spectrum for consumers that need it."""

        return np.fft.fftn(np.asarray(self.reconstruct(), dtype=np.complex128))

    def candidate_report(self) -> tuple[RepresentationDecision, ...]:
        decisions: list[RepresentationDecision] = []
        for block in self.blocks:
            candidates = tuple(candidate.kind.value for candidate in block.candidates)
            decisions.append(
                RepresentationDecision(
                    field_id=f"block_{block.index[0]}_{block.index[1]}",
                    region=str(
                        Region(
                            block.origin,
                            tuple(origin + size for origin, size in zip(block.origin, block.shape, strict=False)),
                        )
                    ),
                    selected=block.representation.value,
                    candidates=candidates,
                    predicted_cost=next(
                        candidate.cost for candidate in block.candidates if candidate.kind is block.representation
                    ),
                    predicted_error=block.error_bound,
                    reason=str(block.metadata.get("reason", "selected by block planner")),
                )
            )
        return tuple(decisions)

    def lod_window(
        self,
        cutoff: float | Sequence[float],
        *,
        window: str = "raised_cosine",
        centered: bool = False,
    ) -> np.ndarray:
        return lod_window(self.shape, cutoff, window=window, centered=centered)

    def apply_lod(
        self,
        cutoff: float | Sequence[float],
        *,
        window: str = "raised_cosine",
        centered: bool = False,
        block_shape: Sequence[int] | None = None,
    ) -> SpectralTexture:
        spectrum = self.spectrum() * self.lod_window(cutoff, window=window, centered=centered)
        filtered = np.fft.ifftn(spectrum)
        values = np.real_if_close(filtered, tol=1000)
        return SpectralTexture.from_array(values, block_shape=block_shape or self.block_shape, error_budget=None)

    lod = apply_lod

    def anisotropic_footprint(
        self,
        covariance: Any,
        *,
        sample_spacing: Sequence[float] = (1.0, 1.0),
    ) -> np.ndarray:
        return anisotropic_gaussian_footprint(self.shape, covariance, sample_spacing=sample_spacing)

    def save(self, path: str | Path) -> None:
        """Write a deterministic, no-pickle SFT2 asset."""

        array_blobs: dict[str, bytes] = {}
        blocks: list[dict[str, Any]] = []
        for block_index, block in enumerate(self.blocks):
            prefix = f"b{block_index:08d}"
            payload_meta: dict[str, Any] = {}
            if block.representation in {TextureRepresentation.DENSE_SPF, TextureRepresentation.HERMITIAN_HALF}:
                name = f"{prefix}_spectrum"
                array_blobs[name] = _npy_bytes(np.asarray(block.payload["spectrum"], dtype=np.uint32))
                payload_meta["spectrum"] = name
            elif block.representation is TextureRepresentation.SVD_SEPARABLE:
                for key in ("U", "s", "Vh"):
                    name = f"{prefix}_{key}"
                    array_blobs[name] = _npy_bytes(np.asarray(block.payload[key]))
                    payload_meta[key] = name
            elif block.representation is TextureRepresentation.QUADRATIC_JET:
                payload_meta["jet"] = _jet_to_json(block.payload["jet"])
            elif block.representation is TextureRepresentation.GAUSSIAN_CHIRP:
                payload_meta["atom"] = _atom_to_json(block.payload["atom"])
            else:  # pragma: no cover
                raise ValueError(f"cannot serialize representation {block.representation!r}")
            blocks.append(
                {
                    "index": list(block.index),
                    "origin": list(block.origin),
                    "shape": list(block.shape),
                    "representation": block.representation.value,
                    "error_bound": block.error_bound,
                    "relative_error_bound": block.relative_error_bound,
                    "certified": block.certified,
                    "metadata": _json_value(dict(block.metadata)),
                    "payload": payload_meta,
                }
            )
        metadata = {
            "format": "SFT2",
            "version": _VERSION,
            "shape": list(self.shape),
            "block_shape": list(self.block_shape),
            "channel_semantics": self.channel_semantics,
            "color_space": self.color_space,
            "real": self.real,
            "source_error_bound": self.source_error_bound,
            "metadata": _json_value(dict(self.metadata)),
            "blocks": blocks,
            "arrays": sorted(array_blobs),
        }
        metadata_bytes = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        output = bytearray(_HEADER.pack(_MAGIC, _VERSION, len(metadata_bytes)))
        output.extend(metadata_bytes)
        for name in sorted(array_blobs):
            name_bytes = name.encode("ascii")
            output.extend(_LENGTH.pack(len(name_bytes)))
            output.extend(name_bytes)
            blob = array_blobs[name]
            output.extend(_LENGTH.pack(len(blob)))
            output.extend(blob)
        Path(path).write_bytes(bytes(output))

    @classmethod
    def load(cls, path: str | Path) -> SpectralTexture:
        """Read an SFT2 asset without ever enabling pickle loading."""

        data = Path(path).read_bytes()
        if len(data) < _HEADER.size:
            raise ValueError("SFT2 file is truncated")
        magic, version, metadata_length = _HEADER.unpack_from(data, 0)
        if magic != _MAGIC or version != _VERSION:
            raise ValueError("unsupported SFT2 header")
        offset = _HEADER.size
        end = offset + int(metadata_length)
        if end > len(data):
            raise ValueError("SFT2 metadata is truncated")
        metadata = json.loads(data[offset:end].decode("utf-8"))
        offset = end
        arrays: dict[str, np.ndarray] = {}
        while offset < len(data):
            if offset + _LENGTH.size > len(data):
                raise ValueError("SFT2 array name length is truncated")
            name_length = int(_LENGTH.unpack_from(data, offset)[0])
            offset += _LENGTH.size
            name_end = offset + name_length
            if name_end > len(data):
                raise ValueError("SFT2 array name is truncated")
            name = data[offset:name_end].decode("ascii")
            offset = name_end
            if offset + _LENGTH.size > len(data):
                raise ValueError("SFT2 array length is truncated")
            blob_length = int(_LENGTH.unpack_from(data, offset)[0])
            offset += _LENGTH.size
            blob_end = offset + blob_length
            if blob_end > len(data):
                raise ValueError("SFT2 array payload is truncated")
            with io.BytesIO(data[offset:blob_end]) as stream:
                arrays[name] = np.asarray(np.load(stream, allow_pickle=False))
            offset = blob_end
        blocks: list[TextureBlock] = []
        for block_meta in metadata.get("blocks", []):
            kind = _representation(block_meta["representation"])
            payload_meta = block_meta["payload"]
            payload: Any
            if kind in {TextureRepresentation.DENSE_SPF, TextureRepresentation.HERMITIAN_HALF}:
                payload = {"spectrum": arrays[payload_meta["spectrum"]]}
            elif kind is TextureRepresentation.SVD_SEPARABLE:
                payload = {key: arrays[payload_meta[key]] for key in ("U", "s", "Vh")}
            elif kind is TextureRepresentation.QUADRATIC_JET:
                payload = {"jet": _jet_from_json(payload_meta["jet"])}
            elif kind is TextureRepresentation.GAUSSIAN_CHIRP:
                payload = {"atom": _atom_from_json(payload_meta["atom"])}
            else:  # pragma: no cover
                raise ValueError(f"unsupported serialized representation {kind!r}")
            blocks.append(
                TextureBlock(
                    tuple(block_meta["index"]),
                    tuple(block_meta["origin"]),
                    tuple(block_meta["shape"]),
                    kind,
                    payload,
                    block_meta.get("error_bound", 0.0),
                    block_meta.get("relative_error_bound", 0.0),
                    (),
                    block_meta.get("certified", True),
                    _restore_json(block_meta.get("metadata", {})),
                )
            )
        if offset != len(data):  # pragma: no cover - defensive parser guard
            raise ValueError("unexpected bytes after SFT2 payload")
        return cls(
            tuple(metadata["shape"]),
            tuple(metadata["block_shape"]),
            tuple(blocks),
            metadata.get("channel_semantics", "GENERIC_FIELD"),
            metadata.get("color_space", "linear"),
            metadata.get("real", True),
            _restore_json(metadata.get("metadata", {})),
            metadata.get("source_error_bound", 0.0),
        )


def _npy_bytes(array: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
    return stream.getvalue()


def _jet_to_json(jet: v02.Jet2D) -> dict[str, Any]:
    return {
        "value": _complex_pair(jet.value),
        "grad_x": _complex_pair(jet.grad_x),
        "grad_y": _complex_pair(jet.grad_y),
        "hess_xx": _complex_pair(jet.hess_xx),
        "hess_xy": _complex_pair(jet.hess_xy),
        "hess_yy": _complex_pair(jet.hess_yy),
        "guard_L": jet.guard_L,
        "guard_T": jet.guard_T,
        "value_kind": jet.value_kind,
        "sign": jet.sign,
    }


def _jet_from_json(value: Mapping[str, Any]) -> v02.Jet2D:
    return v02.Jet2D(
        value=_complex_value(value["value"]),
        grad_x=_complex_value(value["grad_x"]),
        grad_y=_complex_value(value["grad_y"]),
        hess_xx=_complex_value(value["hess_xx"]),
        hess_xy=_complex_value(value["hess_xy"]),
        hess_yy=_complex_value(value["hess_yy"]),
        guard_L=int(value.get("guard_L", 8)),
        guard_T=int(value.get("guard_T", 8)),
        value_kind=value.get("value_kind", "native"),
        sign=int(value.get("sign", 1)),
    )


def _atom_to_json(atom: v02.GaussianChirpAtom) -> dict[str, Any]:
    vector = np.asarray(atom.h, dtype=np.complex128)
    matrix = np.asarray(atom.C, dtype=np.complex128)
    return {
        "kappa": _complex_pair(atom.kappa),
        "h": [[float(value.real), float(value.imag)] for value in vector],
        "C": [[[float(value.real), float(value.imag)] for value in row] for row in matrix],
        "allow_nondecaying": atom.allow_nondecaying,
    }


def _atom_from_json(value: Mapping[str, Any]) -> v02.GaussianChirpAtom:
    h = np.asarray([complex(item[0], item[1]) for item in value["h"]], dtype=np.complex128)
    C = np.asarray([[complex(item[0], item[1]) for item in row] for row in value["C"]], dtype=np.complex128)
    return v02.GaussianChirpAtom.from_arrays(
        _complex_value(value["kappa"]),
        h,
        C,
        allow_nondecaying=bool(value.get("allow_nondecaying", False)),
    )


def _frequency_axes(shape: Sequence[int], sample_spacing: Sequence[float] = (1.0, 1.0)) -> tuple[np.ndarray, ...]:
    dimensions = tuple(int(value) for value in shape)
    spacing = tuple(float(value) for value in sample_spacing)
    if len(dimensions) != 2 or len(spacing) != 2 or any(value <= 0.0 or not math.isfinite(value) for value in spacing):
        raise ValueError("shape and sample_spacing must be positive two-vectors")
    return tuple(dft_frequencies(size, delta) for size, delta in zip(dimensions, spacing, strict=True))


def lod_window(
    shape: Sequence[int],
    cutoff: float | Sequence[float],
    *,
    window: str = "raised_cosine",
    centered: bool = False,
) -> np.ndarray:
    """Return a deterministic radial/elliptic frequency LOD window."""

    dimensions = tuple(int(value) for value in shape)
    if len(dimensions) != 2 or any(value <= 0 for value in dimensions):
        raise ValueError("LOD shape must be a positive two-vector")
    cutoffs: tuple[float, ...]
    if isinstance(cutoff, (int, float, np.integer, np.floating)):
        scalar = float(cutoff)
        cutoffs = (scalar, scalar)
    else:
        cutoffs = tuple(float(value) for value in cutoff)
        if len(cutoffs) != 2:
            raise ValueError("LOD cutoff must be a scalar or two-vector")
    if any(value <= 0.0 or not math.isfinite(value) for value in cutoffs):
        raise ValueError("LOD cutoffs must be positive and finite")
    axes = _frequency_axes(dimensions)
    fy, fx = np.meshgrid(axes[0], axes[1], indexing="ij")
    if centered:
        fy = dft_centered(fy)
        fx = dft_centered(fx)
    radius = np.sqrt((fx / cutoffs[0]) ** 2 + (fy / cutoffs[1]) ** 2)
    name = str(window).lower().replace("-", "_").replace(" ", "_")
    if name in {"hard", "box", "ideal"}:
        result = (radius <= 1.0).astype(np.float64)
    elif name in {"hann", "cosine"}:
        result = np.where(radius <= 1.0, 0.5 * (1.0 + np.cos(math.pi * radius)), 0.0)
    elif name in {"raised_cosine", "raised", "tukey"}:
        result = np.where(
            radius <= 0.8,
            1.0,
            np.where(radius < 1.0, 0.5 * (1.0 + np.cos(math.pi * (radius - 0.8) / 0.2)), 0.0),
        )
    else:
        raise ValueError(f"unsupported LOD window {window!r}")
    return np.asarray(result, dtype=np.float64)


def apply_lod_window(
    spectrum: Any,
    cutoff: float | Sequence[float],
    *,
    window: str = "raised_cosine",
    centered: bool = False,
) -> np.ndarray:
    values = np.asarray(spectrum)
    if values.ndim != 2:
        raise ValueError("LOD application currently requires a two-dimensional spectrum")
    result = values * lod_window(values.shape, cutoff, window=window, centered=centered)
    return np.asarray(result)


@dataclass(frozen=True, slots=True)
class GaussianFootprint:
    """An anisotropic Gaussian pixel footprint and its Fourier transfer."""

    covariance: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.covariance, dtype=np.float64)
        if matrix.shape != (2, 2) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-14):
            raise ValueError("GaussianFootprint covariance must be a symmetric 2x2 matrix")
        if float(np.min(np.linalg.eigvalsh(matrix))) < -1.0e-12:
            raise ValueError("GaussianFootprint covariance must be positive semidefinite")
        matrix = matrix.copy()
        matrix.setflags(write=False)
        object.__setattr__(self, "covariance", matrix)

    def transfer(self, shape: Sequence[int], *, sample_spacing: Sequence[float] = (1.0, 1.0)) -> np.ndarray:
        return anisotropic_gaussian_footprint(shape, self.covariance, sample_spacing=sample_spacing)

    def atom(self) -> v02.GaussianChirpAtom:
        return v02.GaussianChirpAtom.from_arrays(
            0.0j,
            np.zeros(2, dtype=np.complex128),
            4.0 * math.pi**2 * self.covariance.astype(np.complex128),
        )


def anisotropic_gaussian_footprint(
    shape: Sequence[int],
    covariance: Any,
    *,
    sample_spacing: Sequence[float] = (1.0, 1.0),
) -> np.ndarray:
    """Evaluate ``exp(-2π² kᵀΣk)`` on an FFT frequency grid."""

    footprint = GaussianFootprint(np.asarray(covariance, dtype=np.float64))
    fy, fx = np.meshgrid(*_frequency_axes(shape, sample_spacing), indexing="ij")
    frequencies = np.stack((fx, fy), axis=-1)
    exponent = -2.0 * math.pi**2 * np.einsum("...i,ij,...j->...", frequencies, footprint.covariance, frequencies)
    return np.asarray(np.exp(exponent), dtype=np.float64)


gaussian_footprint = anisotropic_gaussian_footprint
frequency_window = lod_window
apply_frequency_window = apply_lod_window
save_texture = SpectralTexture.save
load_texture = SpectralTexture.load


__all__ = [
    "GaussianFootprint",
    "Representation",
    "RepresentationCandidate",
    "SpectralTexture",
    "TextureBlock",
    "TextureRepresentation",
    "anisotropic_gaussian_footprint",
    "apply_frequency_window",
    "apply_lod_window",
    "frequency_window",
    "gaussian_footprint",
    "load_texture",
    "lod_window",
    "save_texture",
]
