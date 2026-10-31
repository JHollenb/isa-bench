"""Shared addressed carrier ingress for renderer subsystems.

Shape inspection never realizes a source. Packed integer planes are SPF words,
not Cartesian numeric values; object adapters remain compatibility boundaries.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np

from spf.tensor import v02


def dft_frequencies(size: int, step: float = 1.0) -> np.ndarray:
    """DFT bin frequencies in cycles per unit in the standard (``fftfreq``) bin order.

    Bins ``0 .. ceil(n/2)-1`` are non-negative, the rest negative; identical to
    NumPy's layout without importing a dense FFT module into the renderer.
    """

    n = int(size)
    bins = np.arange(n, dtype=np.int64)
    bins = np.where(bins >= (n + 1) // 2, bins - n, bins)
    return np.asarray(bins, dtype=np.float64) / (float(n) * float(step))


def dft_frequency_grid(shape: Sequence[int], spacing: Sequence[float] | float = 1.0) -> tuple[np.ndarray, ...]:
    """``meshgrid(indexing="ij")`` of :func:`dft_frequencies` per axis."""

    dims = tuple(int(n) for n in shape)
    steps = np.broadcast_to(np.asarray(spacing, dtype=np.float64), (len(dims),))
    return tuple(np.meshgrid(*[dft_frequencies(n, float(d)) for n, d in zip(dims, steps, strict=True)], indexing="ij"))


def dft_centered(values: np.ndarray) -> np.ndarray:
    """Move the zero-frequency bin to the centre on every axis (``fftshift`` layout)."""

    array = np.asarray(values)
    return np.roll(array, [n // 2 for n in array.shape], axis=tuple(range(array.ndim)))


@lru_cache(maxsize=1)
def magnitude_table() -> np.ndarray:
    """``2**(L/256)`` for every int16 log code, decoded once through the library codec.

    Index is the unsigned 16-bit log code; the zero sentinel decodes to 0 and
    the reserved code saturates to the finite maximum (v0.2 rule).  Renderer
    modules read magnitudes through this table rather than through a private
    ``exp2`` (AGENTS.md rule 3).
    """

    codes = np.arange(65536, dtype=np.uint32) << np.uint32(16)
    table = np.ascontiguousarray(np.real(v02.decode_complex(codes)), dtype=np.float64)
    table.setflags(write=False)
    return table


def decode_magnitude(words: Any, *, squared: bool = False) -> np.ndarray:
    """``|w|`` (or ``|w|^2``) of packed words through the magnitude ROM; NaN words decode to NaN."""

    array = np.asarray(words, dtype=np.uint32)
    codes = (array >> np.uint32(16)).astype(np.int64) & 0xFFFF
    magnitude = magnitude_table()[codes]
    if squared:
        magnitude = magnitude * magnitude
    return np.where(array == np.uint32(v02.SPF32_WORD_NAN), np.nan, magnitude)


def as_block_view(value: Any) -> Any:
    from spf.renderer.blocks import BlockView, KernelBlockView, TensorBlockView, V3BlockView
    from spf.tensor.kernel import SPFKernel
    from spf.tensor.v2 import SPFBlock, TensorBlock
    from spf.tensor.v3 import SPFBlockV3

    if isinstance(value, BlockView):
        return value
    if isinstance(value, (TensorBlock, SPFBlock)):
        return TensorBlockView.from_block(value)
    if isinstance(value, SPFBlockV3):
        return V3BlockView.from_block(value)
    if isinstance(value, SPFKernel):
        return KernelBlockView.from_kernel(value)
    return None


def carrier_shape(value: Any) -> tuple[int, ...]:
    from spf.renderer.model import Field

    if isinstance(value, Field):
        shape = value.metadata.get("shape", value.representation.metadata.get("shape"))
        return tuple(shape) if shape is not None else carrier_shape(value.representation.payload)
    from spf.renderer.blocks import BlockView
    from spf.tensor.kernel import SPFKernel, _shape_for
    from spf.tensor.v2 import SPFBlock, TensorBlock
    from spf.tensor.v3 import SPFBlockV3

    if isinstance(value, BlockView):
        return value.shape
    if isinstance(value, TensorBlock):
        return value.header.dims
    if isinstance(value, SPFBlock):
        shapes = tuple(t.header.dims for t in value.tensors)
        if not shapes or any(s != shapes[0] for s in shapes):
            raise ValueError("renderer SPFBlock needs equal-shaped tensors")
        return shapes[0] if len(shapes) == 1 else (len(shapes), *shapes[0])
    if isinstance(value, SPFBlockV3):
        return value.dims
    if isinstance(value, SPFKernel):
        return _shape_for(value.sig, value.axes)
    if hasattr(value, "shape"):
        return tuple(int(n) for n in value.shape)
    return np.asarray(value).shape


def complex_carrier(value: Any) -> np.ndarray:
    from spf.renderer.model import Field, MaterializationRequest, SampleLayout

    if isinstance(value, Field):
        from spf.renderer.normalization import MaterializationError, normalize_field

        field = normalize_field(value).field
        shape = carrier_shape(field)
        if not shape:
            raise MaterializationError("complex field realization requires sample shape metadata")
        return np.asarray(
            field.materialize(MaterializationRequest(layout=SampleLayout(shape), output_format="complex")),
            dtype=np.complex128,
        )
    block = as_block_view(value)
    if block is not None:
        from spf.renderer.block_filter import FilteredBlockView, sample_source

        if isinstance(block, FilteredBlockView) and not block.frequency_domain:
            return sample_source(block)
        indices = block.active_addresses()
        words = block.sample_words(indices, phase=block.observable != "intensity")
        samples = block.project_words(words, indices)
        result = np.zeros(block.shape, dtype=np.complex128)
        result.flat[indices] = complex_carrier(samples)
        return result
    raw = np.asarray(value)
    if raw.dtype.kind in "iu":
        return np.asarray(decode_words(raw.astype(np.uint32, copy=False)), dtype=np.complex128)
    if raw.dtype.kind == "O":

        def decode(item: Any) -> complex:
            if isinstance(item, (v02.LogPos, v02.LogSigned)):
                return complex(item.value)
            if isinstance(item, v02.SPF32Value):
                return complex(v02.decode_complex(item.word))
            if isinstance(item, (int, np.integer)):
                return complex(v02.decode_complex(int(item)))
            return complex(item)

        return np.asarray([decode(x) for x in raw.flat], dtype=np.complex128).reshape(raw.shape)
    return np.asarray(raw, dtype=np.complex128)


def decode_words(words: Any) -> Any:
    """Decode packed SPF32 words to complex128 via :mod:`spf.tensor.core`.

    Measured 2026-09-17: ``core.decode_words`` is bit-identical to the former
    libm lookup tables for all 65,536 phase codes, so the renderer no longer
    keeps its own decoder (AGENTS.md rule 3). Sentinel policy is the v0.2 one
    from :func:`spf.tensor.v02.decode_complex`.
    """

    array = np.asarray(words, dtype=np.uint32)
    if array.ndim == 0:
        return v02.decode_complex(int(array))
    return v02.decode_complex(array)


def encode_words(values: Any) -> Any:
    """Encode complex values to packed SPF32 words via :mod:`spf.tensor.core`."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim == 0:
        return v02.encode_complex(array)
    return v02.encode_complex(array)


def field_carrier(field: Any) -> Any:
    """Use the renderer's analytic closures; realize only a pending dense suffix."""
    from spf.renderer.model import MaterializationRequest, SampleLayout
    from spf.renderer.normalization import normalize_field

    normalized = normalize_field(field).field
    payload = normalized.representation.payload
    if not normalized.operations and normalized.representation.residual is None:
        return payload
    shape = field.metadata.get("shape", field.representation.metadata.get("shape"))
    if shape is None:
        shape = carrier_shape(payload)
    if not shape:
        raise ValueError("a field with a dense operation suffix requires sample shape metadata")
    return normalized.materialize(MaterializationRequest(layout=SampleLayout(tuple(shape)), output_format="complex"))
