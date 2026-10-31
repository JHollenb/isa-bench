"""Sparse and narrow residual representations for SFR v0.2 fields.

Residual width is a precision-mode decision (design §52).  A
:class:`~spf.renderer.model.PrecisionContract` carries ``residual_log_bits`` and
``residual_phase_bits``; :func:`residual_quantization_bits` reads them and
:meth:`SparseResidual.from_dense` accepts the mode directly, so FAST / BALANCED /
ACCURATE choose 8/12/16-bit log residuals and 10/12/16-bit phase residuals, and
EXACT stores the float64 residual losslessly ("no uncertified pruning").
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any

import numpy as np

from spf.renderer.model import PrecisionContract, PrecisionMode

_POPCOUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


class ResidualKind(StrEnum):
    COO = "COO"
    RLE = "RLE"
    BITMASK_PACKED = "BITMASK_PACKED"
    DENSE_PATCH = "DENSE_PATCH"
    WAVELET = "WAVELET"
    FREQUENCY_BAND = "FREQUENCY_BAND"


class ResidualComponent(StrEnum):
    """Which §52 residual width applies to a stream."""

    LOG = "LOG"
    PHASE = "PHASE"


def _as_precision_contract(precision: Any) -> PrecisionContract:
    if isinstance(precision, PrecisionContract):
        return precision
    return PrecisionContract.for_mode(precision)


def residual_quantization_bits(
    precision: PrecisionContract | PrecisionMode | str,
    *,
    component: ResidualComponent | str = ResidualComponent.LOG,
) -> int | None:
    """Signed residual width for one component under a §52 precision mode.

    Returns ``None`` for EXACT, which keeps the float64 residual (the design's
    "dense/reference fallbacks permitted, no uncertified pruning").
    """

    contract = _as_precision_contract(precision)
    if contract.mode is PrecisionMode.EXACT:
        return None
    selected = component if isinstance(component, ResidualComponent) else ResidualComponent(str(component).upper())
    bits = (
        contract.residual_phase_bits if selected is ResidualComponent.PHASE else contract.residual_log_bits
    )
    return int(bits)


def _readonly(array: Any, dtype: Any | None = None) -> np.ndarray:
    value = np.asarray(array, dtype=dtype).copy()
    value.setflags(write=False)
    return value


def _signed_dtype(bits: int) -> np.dtype[Any]:
    if bits <= 8:
        return np.dtype(np.int8)
    if bits <= 16:
        return np.dtype(np.int16)
    if bits <= 32:
        return np.dtype(np.int32)
    return np.dtype(np.int64)


@dataclass(frozen=True, slots=True)
class SparseResidual:
    """One renderer residual stream with deterministic decoding.

    ``indices`` are flattened C-order positions for COO/frequency encodings.
    A BITMASK_PACKED constructor accepts indices, then stores a little-endian
    bitmask and packed active values instead. For RLE indices are run rows. Values may be real
    or complex for generic fields; narrow log/phase residuals use signed ints.
    """

    kind: ResidualKind | str
    shape: tuple[int, ...]
    values: np.ndarray
    indices: np.ndarray | None = None
    scale: float = 1.0
    error_bound: float = 0.0
    quantization_bits: int | None = None
    bitmask: np.ndarray | None = None
    # Provenance: the §52 mode whose residual width produced this stream.
    precision_mode: PrecisionMode | str | None = None
    _rank_checkpoints: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)
    max_abs_value: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        kind = self.kind if isinstance(self.kind, ResidualKind) else ResidualKind(str(self.kind).upper())
        shape = tuple(int(value) for value in self.shape)
        if not shape or any(value < 0 for value in shape):
            raise ValueError("residual shape must be non-empty and non-negative")
        scale = float(self.scale)
        bound = float(self.error_bound)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("residual scale must be positive and finite")
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("residual error bound must be finite and non-negative")
        bits = None if self.quantization_bits is None else int(self.quantization_bits)
        if bits is not None and not 1 <= bits <= 64:
            raise ValueError("residual quantization bits must be in [1, 64]")
        values = _readonly(self.values)
        indices = None if self.indices is None else _readonly(self.indices, np.int64)
        count = int(math.prod(shape))
        bitmask = None if self.bitmask is None else _readonly(self.bitmask, np.uint8)
        if kind is ResidualKind.BITMASK_PACKED and bitmask is not None:
            if indices is not None:
                raise ValueError("provide bitmask or indices, not both")
            if bitmask.shape != ((count + 7) // 8,):
                raise ValueError("bitmask byte count does not match shape")
            if count % 8 and bitmask.size and int(bitmask[-1]) >> (count % 8):
                raise ValueError("bitmask padding must be zero")
            if int(_POPCOUNT[bitmask].sum()) != values.size:
                raise ValueError("bitmask active count does not match values")
        elif bitmask is not None:
            raise ValueError("bitmask requires BITMASK_PACKED encoding")
        if kind in {ResidualKind.COO, ResidualKind.BITMASK_PACKED, ResidualKind.FREQUENCY_BAND}:
            if bitmask is None and (indices is None or indices.ndim != 1 or values.size != indices.size):
                raise ValueError(f"{kind.value} residual requires one flat index per value")
            if indices is not None:
                if np.any(indices < 0) or np.any(indices >= count):
                    raise ValueError("residual index is outside its shape")
                if np.unique(indices).size != indices.size:
                    raise ValueError("residual indices must be unique")
                order = np.argsort(indices)
                values = _readonly(values.reshape(-1)[order])
                indices = _readonly(indices[order], np.uint32 if count <= 2**32 else np.uint64)
                if kind is ResidualKind.BITMASK_PACKED:
                    bitmask = np.zeros((count + 7) // 8, dtype=np.uint8)
                    np.bitwise_or.at(bitmask, indices // 8, (1 << (indices % 8)).astype(np.uint8))
                    bitmask.setflags(write=False)
                    indices = None
        elif kind is ResidualKind.RLE:
            if indices is None or indices.ndim != 2 or indices.shape[1] != 2:
                raise ValueError("RLE residual indices must contain (start, length) rows")
            if np.any(indices[:, 0] < 0) or np.any(indices[:, 1] <= 0):
                raise ValueError("RLE starts and lengths must be valid")
            if np.any(indices[:, 0] + indices[:, 1] > count):
                raise ValueError("RLE run exceeds residual shape")
            if int(np.sum(indices[:, 1])) != values.reshape(-1).size:
                raise ValueError("RLE values must match total run length")
        elif kind in {ResidualKind.DENSE_PATCH, ResidualKind.WAVELET} and values.shape != shape:
            raise ValueError(f"{kind.value} residual values must match shape")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "error_bound", bound)
        object.__setattr__(self, "quantization_bits", bits)
        object.__setattr__(self, "bitmask", bitmask)
        object.__setattr__(
            self,
            "precision_mode",
            None if self.precision_mode is None else PrecisionMode(str(self.precision_mode).upper()),
        )
        object.__setattr__(
            self, "max_abs_value", float(np.max(np.abs(values.astype(np.complex128)), initial=0)) * scale
        )
        if bitmask is not None:
            # One rank checkpoint per 256 represented samples, not per sample.
            prefix = np.concatenate(([0], np.cumsum(_POPCOUNT[bitmask], dtype=np.uint64)))
            object.__setattr__(
                self, "_rank_checkpoints", _readonly(prefix[::32], np.uint32 if count <= 2**32 else np.uint64)
            )

    @property
    def density(self) -> float:
        count = int(math.prod(self.shape))
        if count == 0:
            return 0.0
        if self.kind in {ResidualKind.DENSE_PATCH, ResidualKind.WAVELET}:
            active = int(np.count_nonzero(self.values))
        else:
            active = int(self.values.size)
        return active / count

    @property
    def nbytes(self) -> int:
        return int(
            sum(a.nbytes for a in (self.values, self.indices, self.bitmask, self._rank_checkpoints) if a is not None)
        )

    def share_layout(self, other: SparseResidual) -> SparseResidual:
        """Share immutable active-mask/rank storage when both streams match."""
        if self.shape != other.shape or self.bitmask is None or other.bitmask is None:
            return self
        if not np.array_equal(self.bitmask, other.bitmask):
            return self
        result = replace(self)
        object.__setattr__(result, "bitmask", other.bitmask)
        object.__setattr__(result, "_rank_checkpoints", other._rank_checkpoints)
        return result

    def sample_flat(self, indices: Any) -> np.ndarray:
        """Decode only requested flat positions; never construct a dense field."""
        query = np.asarray(indices, dtype=np.int64)
        if np.any(query < 0) or np.any(query >= math.prod(self.shape)):
            raise ValueError("residual sample outside shape")
        flat = query.reshape(-1)
        out = np.zeros(flat.size, dtype=np.result_type(self.values.dtype, float))
        if not flat.size or not self.values.size:
            return out.reshape(query.shape)
        if self.kind in {ResidualKind.DENSE_PATCH, ResidualKind.WAVELET}:
            return self.values.reshape(-1)[query] * self.scale
        if self.kind is ResidualKind.BITMASK_PACKED:
            assert self.bitmask is not None and self._rank_checkpoints is not None
            byte, bit = flat // 8, flat % 8
            live = ((self.bitmask[byte] >> bit) & 1).astype(bool)
            live_byte, live_bit = byte[live], bit[live]
            start = (live_byte // 32) * 32
            rank = self._rank_checkpoints[live_byte // 32].astype(np.int64)
            # Bounded rank work, independent of field size and sparsity span.
            for offset in range(32):
                valid = start + offset < live_byte
                rank[valid] += _POPCOUNT[self.bitmask[(start + offset)[valid]]]
            rank += _POPCOUNT[self.bitmask[live_byte] & ((1 << live_bit) - 1).astype(np.uint8)]
            out[live] = self.values.reshape(-1)[rank] * self.scale
        elif self.kind is ResidualKind.RLE:
            assert self.indices is not None
            cursor = 0
            for start, length in self.indices:
                selected = (flat >= start) & (flat < start + length)
                out[selected] = self.values.reshape(-1)[cursor + flat[selected] - int(start)] * self.scale
                cursor += int(length)
        else:
            assert self.indices is not None
            positions = np.searchsorted(self.indices, flat)
            valid = positions < self.indices.size
            valid[valid] &= self.indices[positions[valid]] == flat[valid]
            out[valid] = self.values.reshape(-1)[positions[valid]] * self.scale
        return out.reshape(query.shape)

    def add_to(self, target: np.ndarray, *, region: tuple[slice, ...] | None = None) -> None:
        """Apply a residual to a caller-owned tile, without a dense residual plane."""
        if region is None:
            region = tuple(slice(0, size) for size in self.shape)
        axes = [np.arange(*part.indices(size)) for part, size in zip(region, self.shape, strict=True)]
        shape = tuple(axis.size for axis in axes)
        if target.shape != shape:
            raise ValueError("residual tile shape does not match region")
        linear = np.ravel_multi_index(np.ix_(*axes), self.shape)
        target += self.sample_flat(linear)

    def to_dense(self) -> np.ndarray:
        if self.kind in {ResidualKind.DENSE_PATCH, ResidualKind.WAVELET}:
            return np.asarray(self.values) * self.scale
        dtype = np.result_type(self.values.dtype, np.asarray(self.scale).dtype)
        flat = np.zeros(int(math.prod(self.shape)), dtype=dtype)
        values = self.values.reshape(-1) * self.scale
        if self.kind is ResidualKind.BITMASK_PACKED:
            assert self.bitmask is not None
            active = np.flatnonzero(np.unpackbits(self.bitmask, bitorder="little")[: flat.size])
            flat[active] = values
        elif self.kind is ResidualKind.RLE:
            assert self.indices is not None
            cursor = 0
            for start, length in self.indices:
                run = int(length)
                flat[int(start) : int(start) + run] = values[cursor : cursor + run]
                cursor += run
        else:
            assert self.indices is not None
            flat[self.indices] = values
        return flat.reshape(self.shape)

    def apply(self, base: Any) -> np.ndarray:
        values = np.asarray(base)
        if values.shape != self.shape:
            raise ValueError("residual base shape does not match")
        result = np.array(values, dtype=np.result_type(values.dtype, self.values.dtype, float), copy=True)
        self.add_to(result)
        return result

    @classmethod
    def from_dense(
        cls,
        residual: Any,
        *,
        threshold: float = 0.0,
        quantization_bits: int | None = None,
        kind: ResidualKind | str | None = None,
        precision: PrecisionContract | PrecisionMode | str | None = None,
        component: ResidualComponent | str = ResidualComponent.LOG,
    ) -> SparseResidual:
        """Encode a dense residual, optionally at a §52 precision mode's width.

        ``precision`` selects ``residual_log_bits`` / ``residual_phase_bits``
        from the :class:`~spf.renderer.model.PrecisionContract` for
        ``component``; EXACT keeps float64 values and is lossless at
        ``threshold=0``.  An explicit ``quantization_bits`` always wins.
        """

        values = np.asarray(residual)
        if values.ndim == 0:
            raise ValueError("residual must have at least one dimension")
        limit = float(threshold)
        if not math.isfinite(limit) or limit < 0.0:
            raise ValueError("residual threshold must be finite and non-negative")
        scale = 1.0
        bound = limit
        encoded = values
        mode = None if precision is None else _as_precision_contract(precision).mode
        if quantization_bits is None and precision is not None:
            quantization_bits = residual_quantization_bits(precision, component=component)
        bits = None if quantization_bits is None else int(quantization_bits)
        if bits is not None:
            if np.iscomplexobj(values):
                raise ValueError("narrow residual quantization currently requires real values")
            if not 2 <= bits <= 64:
                raise ValueError("signed residual quantization requires 2..64 bits")
            maximum_code = (1 << (bits - 1)) - 1
            maximum = float(np.max(np.abs(values))) if values.size else 0.0
            scale = maximum / maximum_code if maximum > 0.0 else 1.0
            encoded = np.rint(values / scale).clip(-maximum_code - 1, maximum_code).astype(_signed_dtype(bits))
            bound += 0.5 * scale
        mask = np.abs(values) > limit
        active = np.flatnonzero(mask.reshape(-1))
        index_bytes = 4 if values.size <= 2**32 else 8
        mask_bytes = (values.size + 7) // 8
        costs = {
            ResidualKind.DENSE_PATCH: values.size * encoded.dtype.itemsize,
            ResidualKind.COO: active.size * (encoded.dtype.itemsize + index_bytes),
            ResidualKind.BITMASK_PACKED: active.size * encoded.dtype.itemsize
            + mask_bytes
            + (mask_bytes // 32 + 1) * index_bytes,
        }
        selected_kind = (
            (kind if isinstance(kind, ResidualKind) else ResidualKind(str(kind).upper()))
            if kind is not None
            else min(costs, key=lambda candidate: costs[candidate])
        )
        if selected_kind in {ResidualKind.DENSE_PATCH, ResidualKind.WAVELET}:
            dense = np.where(mask, encoded, 0)
            return cls(
                selected_kind,
                values.shape,
                dense,
                scale=scale,
                error_bound=bound,
                quantization_bits=bits,
                precision_mode=mode,
            )
        if selected_kind is ResidualKind.RLE:
            flat_mask = mask.reshape(-1)
            starts: list[tuple[int, int]] = []
            index = 0
            while index < flat_mask.size:
                if not flat_mask[index]:
                    index += 1
                    continue
                start = index
                while index < flat_mask.size and flat_mask[index]:
                    index += 1
                starts.append((start, index - start))
            selected_values = (
                np.concatenate([encoded.reshape(-1)[start : start + length] for start, length in starts])
                if starts
                else np.asarray([], dtype=encoded.dtype)
            )
            return cls(
                selected_kind,
                values.shape,
                selected_values,
                np.asarray(starts, dtype=np.int64).reshape(-1, 2),
                scale,
                bound,
                bits,
                precision_mode=mode,
            )
        return cls(
            selected_kind,
            values.shape,
            encoded.reshape(-1)[active],
            active,
            scale,
            bound,
            bits,
            precision_mode=mode,
        )


def residual_storage_bytes(*streams: SparseResidual | None) -> int:
    """Count shared mask and rank arrays only once."""
    arrays = {
        id(a): a
        for stream in streams
        if stream is not None
        for a in (stream.values, stream.indices, stream.bitmask, stream._rank_checkpoints)
        if a is not None
    }
    return sum(a.nbytes for a in arrays.values())


__all__ = [
    "ResidualComponent",
    "ResidualKind",
    "SparseResidual",
    "residual_quantization_bits",
    "residual_storage_bytes",
]
