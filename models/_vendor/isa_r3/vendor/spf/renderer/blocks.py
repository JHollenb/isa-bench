"""Immutable renderer views over the existing SPFTensor storage contract."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from spf.tensor.v2 import SPFBlock, TensorBlock

if TYPE_CHECKING:
    from spf.tensor.kernel import SPFKernel
    from spf.tensor.v3 import SPFBlockV3


@dataclass(frozen=True, slots=True)
class BlockSummedBound:
    """Ref §10.4 block cull record: stored ``(L_max, N_active)`` and ``L_Ub``.

    ``L_max`` is the block's per-sample log-code envelope in the units of its
    own observable (header offsets applied, intensity already doubled).
    ``L_upper`` is the design's ``L_Ub = L_b + RNE(256 log2 N_b)``, a bound on
    the *total* contribution of all ``N_active`` terms, which is what a culling
    decision over a whole block needs; the per-sample ``magnitude_upper_bound``
    understates that sum by up to ``N_b``.
    """

    L_max: float
    N_active: int
    L_upper: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "L_max", float(self.L_max))
        object.__setattr__(self, "N_active", int(self.N_active))
        object.__setattr__(self, "L_upper", float(self.L_upper))

    @property
    def magnitude(self) -> float:
        """``2**(L_upper/256)``, with the numeric terminal's outward guard."""

        return _magnitude_from_log_code(self.L_upper)


class BlockView:
    """Owned structural carrier consumed by address, with closed header operations."""

    shape: tuple[int, ...]
    observable: str
    log_offset: float
    phase_offset: float
    mask_indices: np.ndarray | None

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        raise NotImplementedError

    def active_addresses(self) -> np.ndarray:
        raise NotImplementedError

    def stored_log_upper_bound(self) -> float:
        """``L_b``: the stored per-sample log-code envelope, before header offsets.

        ``-inf`` means a canonically zero block and ``inf`` means unknown. The
        default is unknown; a carrier that can read its header overrides it.
        """

        return math.inf

    def active_count(self) -> int:
        """``N_b``: an upper bound on live addresses, without reading a plane."""

        return int(self.active_addresses().size)

    def summed_contribution_upper_bound_L(self) -> BlockSummedBound:  # noqa: N802 - §10.4 L_Ub spelling
        """Ref §10.4 ``L_Ub = L_b + RNE(256 log2 N_b)`` for this block.

        The record is the design's stored ``(L_max, N_active)`` pair, so a
        consumer culls a whole block without touching a sample.
        """

        from spf.tensor import v02

        if self.mask_indices is not None and self.mask_indices.size == 0:
            return BlockSummedBound(-math.inf, 0, -math.inf)
        count = max(0, self.active_count())
        stored = self.stored_log_upper_bound()
        if stored == -math.inf:
            return BlockSummedBound(-math.inf, count, -math.inf)
        if not math.isfinite(stored) or not math.isfinite(self.log_offset):
            return BlockSummedBound(math.inf, count, math.inf)
        log_max = (stored + self.log_offset) * (2 if self.observable == "intensity" else 1)
        if count == 0:
            return BlockSummedBound(log_max, 0, -math.inf)
        return BlockSummedBound(log_max, count, log_max + float(v02._rne(256.0 * math.log2(count))))

    @property
    def scalar_bits(self) -> int:
        return 32

    @property
    def nbytes(self) -> int:
        raise NotImplementedError

    @property
    def magnitude_upper_bound(self) -> float:
        """Conservative sample magnitude envelope; infinity means unknown.

        The envelope is for this carrier, before a Field's separate residual
        and uncertainty ledger. It must not reconstruct a plane to screen it.
        """
        return 0.0 if self.mask_indices is not None and self.mask_indices.size == 0 else math.inf

    def _replace(self, **updates: Any) -> BlockView:
        return cast(BlockView, replace(cast(Any, self), **updates))

    def apply(self, operation: Any) -> BlockView:
        """Compose a closed header operation, sharing the owned residual streams.

        Offsets are wide until the sample boundary. They apply to absolute
        outliers too; inactive samples and zero/NaN sentinels are invariant.
        Masks store only live addresses and therefore work for point queries.
        """
        op, params = operation.op, operation.params
        if op == "exposure":
            ev = float(params.get("ev", params.get("exposure", 0)))
            if not math.isfinite(ev):
                raise ValueError("exposure must be finite")
            return self._replace(log_offset=self.log_offset + ev * (128 if self.observable == "intensity" else 256))
        if op in {"phase", "phase_shift"}:
            if self.observable != "complex":
                raise ValueError("phase shift requires a complex block")
            phase = float(params.get("dT", params.get("phase", params.get("delta", 0))))
            if params.get("phase_units") in {"radians", "rad"}:
                phase *= 65536 / (2 * math.pi)
            if not math.isfinite(phase):
                raise ValueError("phase must be finite")
            return self._replace(phase_offset=(self.phase_offset + phase) % 65536)
        if op == "mask":
            mask = np.broadcast_to(np.asarray(params["mask"], dtype=bool), self.shape)
            indices = np.flatnonzero(mask)
            if self.mask_indices is not None:
                indices = np.intersect1d(indices, self.mask_indices, assume_unique=True)
            indices.setflags(write=False)
            return self._replace(mask_indices=indices)
        raise ValueError(f"operation {op!r} is not closed over TensorBlockView")

    def sample(self, points: Any) -> np.ndarray:
        coordinates = np.asarray(points, dtype=np.float64)
        if coordinates.ndim < 1 or coordinates.shape[-1] != len(self.shape):
            raise ValueError("block sample coordinate dimension does not match extent")
        if not np.all(np.isfinite(coordinates)) or np.any(coordinates != np.rint(coordinates)):
            raise ValueError("TensorBlock sampling requires finite integer coordinates")
        indices = coordinates.astype(np.int64)
        linear = np.ravel_multi_index(tuple(indices[..., axis] for axis in range(len(self.shape))), self.shape)
        return self.project_words(self.sample_words(linear, phase=self.observable != "intensity"), linear)

    def project_words(self, result: np.ndarray, linear: np.ndarray) -> np.ndarray:
        from spf.tensor import v02

        log_codes = (result >> 16).astype(np.int16).astype(np.float64)
        finite = (log_codes != v02.SPF32_L_ZERO) & (result != v02.SPF32_WORD_NAN)
        wide_log = log_codes + self.log_offset
        wide = self.log_offset != round(self.log_offset) or self.phase_offset != round(self.phase_offset)
        wide = wide or bool(
            np.any(finite & ((wide_log < v02.SPF32_L_MIN_FINITE) | (wide_log > v02.SPF32_L_MAX_FINITE)))
        )
        # Retain fractional and out-of-storage-range header state until the
        # requested numeric terminal. An explicit SPF32 output quantizes once
        # in the formatter, which reports the actual quantization/clipping error.
        if wide or self.observable == "intensity":
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                values = np.exp2(wide_log * (2 if self.observable == "intensity" else 1) / 256)
                values = np.where(log_codes == v02.SPF32_L_ZERO, 0.0, values)
                if self.observable != "intensity":
                    angle = ((result & 65535).astype(float) + self.phase_offset) * (2 * math.pi / 65536)
                    values = values * np.exp(1j * angle)
            values = np.where(result == v02.SPF32_WORD_NAN, np.nan, values)
            if self.mask_indices is not None:
                values = np.where(np.isin(linear, self.mask_indices), values, 0)
            return np.asarray(values)
        if self.log_offset or self.phase_offset:
            result = transform_words(result, self.log_offset, self.phase_offset)
        if self.mask_indices is not None:
            result = np.where(np.isin(linear, self.mask_indices), result, np.uint32(v02.SPF32_WORD_ZERO))
        return np.asarray(result, dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class TensorBlockView(BlockView):
    """A prepared, addressed consumer of TensorBlock/SPFBlock storage.

    One tensor retains its original shape. Several equal-shaped tensors have
    an explicit leading tensor axis; they are not implicitly added or tiled.
    Import takes ownership of a frozen copy. Header updates return a new view
    sharing the immutable residual arrays and need no dense reconstruction.
    """

    _tensors: tuple[TensorBlock, ...]
    shape: tuple[int, ...]
    observable: str = "complex"
    log_offset: float = 0.0
    phase_offset: float = 0.0
    mask_indices: np.ndarray | None = None

    @classmethod
    def from_block(cls, value: TensorBlock | SPFBlock) -> TensorBlockView:
        tensors = (value,) if isinstance(value, TensorBlock) else value.tensors
        if not tensors:
            raise ValueError("renderer SPFBlock must contain a tensor")
        owned = []
        for original in tensors:
            tensor = copy.deepcopy(original)
            tensor.validate()
            for name in ("dL", "dT", "active_indices"):
                data = getattr(tensor, name)
                if data is not None:
                    array = np.asarray(data).copy()
                    array.setflags(write=False)
                    setattr(tensor, name, array)
            owned.append(tensor)
        dims = owned[0].header.dims
        if any(t.header.dims != dims for t in owned):
            raise ValueError("renderer SPFBlock requires equal-shaped tensors; bind unequal tensors individually")
        shape = dims if len(owned) == 1 else (len(owned), *dims)
        return cls(tuple(owned), shape)

    @property
    def nbytes(self) -> int:
        return sum(
            256
            + len(t.packed_residuals)
            + 8 * len(t.outliers)
            + sum(np.asarray(a).nbytes for a in (t.dL, t.dT, t.active_indices) if a is not None)
            for t in self._tensors
        )

    @cached_property
    def magnitude_upper_bound(self) -> float:
        if self.mask_indices is not None and self.mask_indices.size == 0:
            return 0.0
        return _project_log_upper_bound(self, self.stored_log_upper_bound())

    def stored_log_upper_bound(self) -> float:
        return max((_tensor_log_upper_bound(tensor) for tensor in self._tensors), default=-math.inf)

    def active_count(self) -> int:
        """Count live addresses from the stored encoding, never from a plane."""

        from spf.tensor.core import ActiveEncoding

        total = 0
        for tensor in self._tensors:
            if tensor.header.active_encoding is ActiveEncoding.DENSE_ALL:
                total += int(tensor.header.total_elements)
                continue
            addresses = np.asarray(tensor.active_indices, dtype=np.int64)
            if tensor.outliers:
                addresses = np.union1d(addresses, np.fromiter(tensor.outliers, dtype=np.int64))
            total += int(addresses.size)
        # A mask can only remove addresses, so its own size is also a bound.
        if self.mask_indices is not None:
            total = min(total, int(self.mask_indices.size))
        return total

    def with_header(self, *, tensor_index: int = 0, **updates: Any) -> TensorBlockView:
        tensors = list(self._tensors)
        original = tensors[tensor_index]
        header = replace(original.header, **updates)
        if header.dims != original.header.dims or header.active_encoding != original.header.active_encoding:
            raise ValueError("header update cannot change extent or active encoding")
        tensor = replace(original, header=header)
        tensor.validate()
        tensors[tensor_index] = tensor
        return replace(self, _tensors=tuple(tensors))

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        linear = np.asarray(linear, dtype=np.int64)
        if len(self._tensors) == 1:
            result = self._tensors[0].sample_words(linear, validate=False, phase=phase)
        else:
            stride = self._tensors[0].header.total_elements
            result = np.empty(linear.shape, dtype=np.uint32)
            for index, tensor in enumerate(self._tensors):
                selected = linear // stride == index
                result[selected] = tensor.sample_words(linear[selected] % stride, validate=False, phase=phase)
        return result

    def active_addresses(self) -> np.ndarray:
        from spf.tensor.core import ActiveEncoding

        chunks = []
        for i, tensor in enumerate(self._tensors):
            count = tensor.header.total_elements
            addresses = (
                np.arange(count, dtype=np.int64)
                if tensor.header.active_encoding is ActiveEncoding.DENSE_ALL
                else np.asarray(tensor.active_indices, dtype=np.int64)
            )
            if tensor.outliers:
                addresses = np.union1d(addresses, np.fromiter(tensor.outliers, dtype=np.int64))
            chunks.append(addresses + i * count)
        addresses = np.concatenate(chunks)
        return np.intersect1d(addresses, self.mask_indices) if self.mask_indices is not None else addresses


def _owned(value: Any) -> Any:
    """Own all nested source arrays without expanding a structural expression."""
    from collections.abc import Mapping
    from dataclasses import fields, is_dataclass

    value = copy.deepcopy(value)
    seen: set[int] = set()

    def freeze(item: Any) -> None:
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, np.ndarray):
            item.setflags(write=False)
        elif isinstance(item, Mapping):
            for child in item.values():
                freeze(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                freeze(child)
        elif is_dataclass(item):
            for attr in fields(item):
                freeze(getattr(item, attr.name))

    freeze(value)
    return value


@dataclass(frozen=True, slots=True)
class V3BlockView(BlockView):
    block: SPFBlockV3
    shape: tuple[int, ...]
    observable: str = "complex"
    log_offset: float = 0.0
    phase_offset: float = 0.0
    mask_indices: np.ndarray | None = None

    @classmethod
    def from_block(cls, value: Any) -> V3BlockView:
        _ = value.representation.format  # reject non-polar representations explicitly
        return cls(_owned(value), value.dims)

    @property
    def nbytes(self) -> int:
        return 64 + (self.block.storage_bits() + 7) // 8

    @property
    def scalar_bits(self) -> int:
        return self.block.representation.format.total_bits

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        return self.block.sample_words(linear, phase=phase)

    def active_addresses(self) -> np.ndarray:
        indices = self.block.active_addresses()
        return np.intersect1d(indices, self.mask_indices) if self.mask_indices is not None else indices

    def project_words(self, result: np.ndarray, linear: np.ndarray) -> np.ndarray:
        if self.scalar_bits == 32:
            return BlockView.project_words(self, result, linear)
        from spf.tensor.core import unpack_L, unpack_theta

        fmt = self.block.representation.format
        logs = unpack_L(result, fmt)
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.exp2(logs.astype(float) / fmt.l_scale + self.log_offset / 256)
            values = np.where(logs == fmt.l_code_zero, 0, values)
            if self.observable == "intensity":
                values = values * values
            else:
                angle = unpack_theta(result, fmt).astype(float) * (2 * math.pi / (1 << fmt.theta_bits))
                values = values * np.exp(1j * (angle + self.phase_offset * (2 * math.pi / 65536)))
        values = np.where(result == fmt.word_nan, np.nan, values)
        if self.mask_indices is not None:
            values = np.where(np.isin(linear, self.mask_indices), values, 0)
        return values


@dataclass(frozen=True, slots=True)
class KernelBlockView(BlockView):
    kernel: SPFKernel
    shape: tuple[int, ...]
    observable: str = "complex"
    log_offset: float = 0.0
    phase_offset: float = 0.0
    mask_indices: np.ndarray | None = None

    @classmethod
    def from_kernel(cls, value: Any) -> KernelBlockView:
        from spf.tensor.kernel import _shape_for

        return cls(_owned(value), _shape_for(value.sig, value.axes))

    @property
    def nbytes(self) -> int:
        from spf.tensor.kernel import _stored_carrier_bytes

        return 64 + _stored_carrier_bytes(self.kernel)

    @cached_property
    def magnitude_upper_bound(self) -> float:
        if self.mask_indices is not None and self.mask_indices.size == 0:
            return 0.0
        return _project_log_upper_bound(self, self.stored_log_upper_bound())

    def stored_log_upper_bound(self) -> float:
        op = self.kernel.expr.op
        if op == "ClosedFormIdentity":
            return 0.0
        if op == "DiagonalSPF":
            return _packed_log_upper_bound(self.kernel.expr.args[0]["diag_words"])
        # An arbitrary matrix expression needs its own algebraic bound; reading
        # all its addresses here would defeat the screening path.
        return math.inf

    def sample_words(self, linear: Any, *, phase: bool = True) -> np.ndarray:
        return self.kernel.sample_words(linear, phase=phase)

    def active_addresses(self) -> np.ndarray:
        indices = self.kernel.active_addresses()
        return np.intersect1d(indices, self.mask_indices) if self.mask_indices is not None else indices


def _packed_log_upper_bound(words: Any) -> float:
    """Read only stored log codes, preserving zero and invalid sentinels."""
    from spf.tensor import v02

    packed = np.asarray(words, dtype=np.uint32).reshape(-1)
    if np.any(packed == v02.SPF32_WORD_NAN):
        return math.inf
    logs = (packed >> 16).astype(np.int16)
    finite = logs != v02.SPF32_L_ZERO
    return float(np.max(logs[finite])) if np.any(finite) else -math.inf


def _tensor_log_upper_bound(tensor: TensorBlock) -> float:
    """Bound the canonical v2 log header plus residual without coordinates.

    Wrapped origins permute each axis and cannot enlarge its interval. Bands
    are disjoint per axis; their maxima can be added independently. The
    residual maximum may occur elsewhere, so the resulting bound is safe but
    deliberately not claimed to be the exact pivot. Absolute outliers are
    included separately and canonical saturation occurs before wide offsets.
    """
    from spf.tensor import v02

    header = tensor.header
    residual = np.asarray(tensor.dL, dtype=np.int16)
    upper = -math.inf
    if residual.size:
        axis_terms = [max(0, int(slope) * (size - 1)) for slope, size in zip(
            header.band_exp or (), header.dims, strict=True
        )]
        band_terms = [0] * len(header.dims)
        for band in tensor.bands:
            if band.effective_flags & 0x01:
                band_terms[band.dim] = max(band_terms[band.dim], int(band.L_delta))
        raw_upper = int(header.global_L_bias) + int(header.coarse_exp)
        raw_upper += sum(axis_terms) + sum(band_terms) + int(np.max(residual))
        constant_zero = (
            raw_upper == v02.SPF32_L_ZERO
            and not any(header.band_exp or ())
            and not any(band.effective_flags & 0x01 and band.L_delta for band in tensor.bands)
            and bool(np.all(residual == residual[0]))
        )
        if not constant_zero:
            upper = float(min(v02.SPF32_L_MAX_FINITE, max(v02.SPF32_L_MIN_FINITE, raw_upper)))
    if tensor.outliers:
        upper = max(upper, _packed_log_upper_bound(list(tensor.outliers.values())))
    return upper


def _magnitude_from_log_code(code: float) -> float:
    """Project one log code to a magnitude, guarded outward for the terminal."""

    if code == -math.inf:
        return 0.0
    if not math.isfinite(code):
        return math.inf
    try:
        value = math.exp2(code / 256)
    except OverflowError:
        return math.inf
    # Outward guard includes the numeric terminal's exp2/complex projection.
    return math.nextafter(value * (1.0 + 8.0 * np.finfo(np.float64).eps), math.inf)


def _project_log_upper_bound(view: BlockView, log_bound: float) -> float:
    if log_bound == -math.inf:
        return 0.0
    if not math.isfinite(log_bound) or not math.isfinite(view.log_offset):
        return math.inf
    return _magnitude_from_log_code((log_bound + view.log_offset) * (2 if view.observable == "intensity" else 1))


def transform_words(words: Any, log_offset: float = 0, phase_offset: float = 0) -> np.ndarray:
    """Typed packed scalar transform, with RNE, modular phase and sentinel preservation."""
    from spf.tensor import v02

    words = np.asarray(words, dtype=np.uint32)
    log = (words >> 16).astype(np.int16).astype(np.float64)
    phase = (words & 65535).astype(np.float64)
    shifted = np.clip(np.rint(log + log_offset), v02.SPF32_L_MIN_FINITE, v02.SPF32_L_MAX_FINITE).astype(np.int32)
    angle = np.rint(phase + phase_offset).astype(np.int64) & 65535
    result = (shifted.astype(np.uint32) << 16) | angle.astype(np.uint32)
    special = (log == v02.SPF32_L_ZERO) | (words == v02.SPF32_WORD_NAN)
    return np.asarray(np.where(special, words, result), dtype=np.uint32)


__all__ = ["BlockSummedBound", "BlockView", "KernelBlockView", "TensorBlockView", "V3BlockView"]
