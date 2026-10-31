"""Structure preserving carrier ingress for ocean execution.

The ocean renderer has two very different boundaries:

* a sparse list of authored modes (where a small list is the useful carrier),
  and
* an addressed SPF block/tensor/matrix (where the header, support, residual
  stream, and outliers are the useful carrier).

This module deliberately stops at that boundary.  In particular, constructing
an :class:`OceanCarrier` never calls ``reconstruct()``, ``to_words()``,
``materialize_words()``, or ``complex_carrier()``.  Packed words are fetched
only after a terminal route has been selected, and then are kept as packed
words until the reference consumer needs a numeric value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from spf.renderer.blocks import BlockView, KernelBlockView, TensorBlockView, V3BlockView
from spf.renderer.carriers import encode_words
from spf.tensor import v02


def _shape(value: Sequence[int] | np.ndarray) -> tuple[int, ...]:
    result = tuple(int(x) for x in np.asarray(value).reshape(-1).tolist())
    if not result or any(x <= 0 for x in result):
        raise ValueError("ocean shape must contain positive dimensions")
    return result


def _half_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*shape[:-1], shape[-1] // 2 + 1)


def _normal_index(index: Sequence[int], shape: tuple[int, ...]) -> tuple[int, ...]:
    values_list: list[int] = []
    for raw in index:
        if isinstance(raw, (bool, np.bool_)):
            raise TypeError("Fourier indices must be integral coordinates")
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise TypeError("Fourier indices must be integral coordinates") from exc
        if isinstance(raw, (float, np.floating)) and (not np.isfinite(raw) or float(raw) != value):
            raise ValueError("Fourier indices must be integral coordinates")
        values_list.append(value)
    values = tuple(values_list)
    if len(values) != len(shape):
        raise ValueError("Fourier index dimension does not match the ocean shape")
    return tuple(x % n for x, n in zip(values, shape, strict=True))


def _normal_indices(indices: Any, shape: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    # Validate scalar coordinates before narrowing them.  An early integer
    # cast would silently accept fractional authored support.
    array = np.asarray(indices)
    if array.ndim == 1:
        if array.size != len(shape):
            raise ValueError("Fourier indices must have shape (K, dimension)")
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(shape):
        raise ValueError("Fourier indices must have shape (K, dimension)")
    return tuple(_normal_index(row, shape) for row in array)


def _as_tuple(value: Any, count: int | None = None) -> tuple[Any, ...]:
    if value is None:
        result: tuple[Any, ...] = ()
    elif isinstance(value, np.ndarray):
        result = tuple(value.reshape(-1).tolist())
    elif isinstance(value, (list, tuple)):
        result = tuple(value)
    else:
        result = (value,)
    if count is not None and len(result) != count:
        raise ValueError("ocean coefficient count does not match Fourier index count")
    return result


def _is_block(value: Any) -> bool:
    return isinstance(value, BlockView)


def _view_from(value: Any) -> BlockView | None:
    if isinstance(value, BlockView):
        return value
    from spf.renderer.carriers import as_block_view
    from spf.renderer.model import Field

    if isinstance(value, Field):
        value = value.representation.payload

    result = as_block_view(value)
    return result if isinstance(result, BlockView) else None


def _array_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    try:
        return int(np.asarray(value).nbytes)
    except (TypeError, ValueError):
        return 0


def _view_support(view: BlockView) -> tuple[int, str, int, int, int, int]:
    """Return (count, encoding, residual_count, residual_bytes, outliers, mask)."""

    mask_count = 0 if view.mask_indices is None else int(np.asarray(view.mask_indices).size)
    # A normalized Fourier filter is a lazy wrapper around the addressed
    # source.  Its own shape/transfer metadata must not widen a compact COO
    # certificate to the whole plane before route selection.
    source = getattr(view, "source", None)
    if type(view).__name__ == "FilteredBlockView" and source is not None:
        count, encoding, residual_count, residual_bytes, outliers, _ = _view_support(source)
        if view.mask_indices is not None:
            count = min(count, mask_count)
        return count, encoding, residual_count, residual_bytes, outliers, mask_count
    if isinstance(view, TensorBlockView):
        total = 0
        count = 0
        encodings: set[str] = set()
        residual_count = 0
        residual_bytes = 0
        outliers = 0
        from spf.tensor.core import ActiveEncoding

        for tensor in view._tensors:
            elements = int(tensor.header.total_elements)
            total += elements
            # v2 TensorBlock headers and the legacy core helpers expose
            # equivalent enum names from different modules.  Compare the
            # stable name so a COO support certificate is not widened to the
            # whole half-plane before an addressed native transfer lowering.
            encoding_name = str(getattr(tensor.header.active_encoding, "name", tensor.header.active_encoding))
            if encoding_name == ActiveEncoding.DENSE_ALL.name:
                count += elements
                encodings.add("dense_all")
            elif encoding_name == ActiveEncoding.COO_U32.name:
                count += int(np.asarray(tensor.active_indices).size)
                encodings.add("coo_u32")
            else:
                # Unsupported encodings are still structural.  Let the block
                # consumer report the explicit fallback when it is fetched.
                encodings.add(str(tensor.header.active_encoding.name).lower())
                count += elements
            residual_count += int(np.asarray(tensor.dL).size) if tensor.dL is not None else 0
            residual_count += int(np.asarray(tensor.dT).size) if tensor.dT is not None else 0
            residual_bytes += _array_nbytes(tensor.dL) + _array_nbytes(tensor.dT)
            outliers += len(tensor.outliers)
        if view.mask_indices is not None:
            # The exact intersection is deferred for a dense mask.  The count
            # remains a conservative upper bound in the plan metadata.
            count = min(count, mask_count)
        return count, "+".join(sorted(encodings)) or "unknown", residual_count, residual_bytes, outliers, mask_count
    if isinstance(view, V3BlockView):
        block = view.block
        total = int(block.total_elements)
        params = block.base.params
        active = params.get("active_indices") if block.base.profile_id == 0 else None
        if active is None:
            count = total
            encoding = "dense_model"
        else:
            count = int(np.asarray(active).size)
            encoding = "coo_u32"
        count += len(block.outliers)
        residual = block.residual.payload
        if isinstance(residual, Mapping):
            residual_count = sum(int(np.asarray(v).size) for k, v in residual.items() if str(k) in {"dL", "dT"})
            residual_bytes = sum(_array_nbytes(v) for k, v in residual.items() if str(k) in {"dL", "dT"})
        else:
            residual_count, residual_bytes = 0, _array_nbytes(residual)
        if view.mask_indices is not None:
            count = min(count, mask_count)
        return count, encoding, residual_count, residual_bytes, len(block.outliers), mask_count
    if isinstance(view, KernelBlockView):
        op = str(view.kernel.expr.op)
        if op == "ClosedFormZero":
            count = 0
            encoding = "closed_form_zero"
        elif op in {"DiagonalSPF", "Permutation", "Roll", "ClosedFormIdentity"}:
            count = int(np.prod(view.shape, dtype=np.int64) // max(1, view.shape[-1]))
            encoding = f"structural_{op.lower()}"
        elif op == "Block" and view.kernel.expr.block is not None:
            # The nested block has its own support certificate.  Avoid calling
            # KernelBlockView.active_addresses here because a generic kernel
            # may conservatively enumerate its full matrix.
            nested = V3BlockView.from_block(view.kernel.expr.block)
            count, encoding, residual_count, residual_bytes, outliers, _ = _view_support(nested)
            return count, f"kernel_block:{encoding}", residual_count, residual_bytes, outliers, mask_count
        else:
            count = int(np.prod(view.shape, dtype=np.int64))
            encoding = f"kernel_{op.lower()}"
        if view.mask_indices is not None:
            count = min(count, mask_count)
        return count, encoding, 0, 0, 0, mask_count
    total = int(np.prod(view.shape, dtype=np.int64))
    return total, "block_view", 0, 0, 0, mask_count


def _source_metadata(
    *,
    shape: tuple[int, ...],
    kind: str,
    view: BlockView | None,
    half: np.ndarray | None,
    indices: tuple[tuple[int, ...], ...],
    coefficients: tuple[Any, ...],
    modes: tuple[Any, ...],
    half_view: bool,
) -> Mapping[str, Any]:
    values: dict[str, Any] = {
        "source_kind": kind,
        "shape": shape,
        "carrier_shape": None if view is None else tuple(view.shape),
        "half_view": bool(half_view),
        "potential_elements": int(math.prod(shape)),
        "active_modes": len(indices) if indices else len(modes),
        "structure_preserved": bool(view is not None),
        "packed_storage": bool(
            view is not None or any(isinstance(x, (int, np.integer, v02.SPF32Value)) for x in coefficients)
        ),
        "source_conversion": "none",
    }
    if half is not None:
        values.update({"half_shape": tuple(half.shape), "half_dtype": str(half.dtype), "half_nbytes": int(half.nbytes)})
        if half.dtype.kind not in "iu":
            values["source_conversion"] = "cartesian_half_spectrum_compatibility"
    if view is not None:
        count, encoding, residual_count, residual_bytes, outliers, mask_count = _view_support(view)
        log_offset, phase_offset = _view_offsets(view)
        values.update(
            {
                "support_count": int(count),
                "support_encoding": encoding,
                "residual_count": int(residual_count),
                "residual_bytes": int(residual_bytes),
                "outlier_count": int(outliers),
                "mask_count": int(mask_count),
                "scalar_bits": int(view.scalar_bits),
                "carrier_nbytes": int(view.nbytes),
                "profile_id": (
                    int(view.block.base.profile_id)
                    if isinstance(view, V3BlockView)
                    else str(view.kernel.expr.op)
                    if isinstance(view, KernelBlockView)
                    else "v2_tensor"
                ),
                "source_conversion": "addressed_packed_words",
                "log_offset": log_offset,
                "phase_offset": phase_offset,
                "fractional_offsets": bool(log_offset != round(log_offset) or phase_offset != round(phase_offset)),
            }
        )
        if len(shape) == 2 and (encoding.startswith("structural_diagonal") or str(encoding).endswith("diagonalspf")):
            values["support_pattern"] = "diagonal"
        elif isinstance(view, TensorBlockView) and len(view._tensors) == 1 and len(shape) == 2:
            from spf.tensor.core import ActiveEncoding

            tensor = view._tensors[0]
            if tensor.header.active_encoding is ActiveEncoding.COO_U32 and tensor.active_indices is not None:
                addresses = np.asarray(tensor.active_indices, dtype=np.int64).reshape(-1)
                rows, columns = np.unravel_index(addresses, view.shape)
                if addresses.size and bool(np.all(rows == columns)):
                    values["support_pattern"] = "diagonal"
    return MappingProxyType(values)


def _word_from_value(value: Any) -> int:
    if isinstance(value, v02.SPF32Value):
        return int(value.word)
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        word = int(value)
        if word < 0 or word > 0xFFFFFFFF:
            raise ValueError("packed ocean coefficients must be unsigned uint32 words")
        return word
    if isinstance(value, v02.LogPos):
        return int(encode_words(complex(value.value, 0.0)))
    if isinstance(value, v02.LogSigned):
        return int(encode_words(complex(value.value, 0.0)))
    return int(encode_words(complex(value)))


def _validated_spf32_words(value: Any, *, source: str) -> np.ndarray:
    """Validate a block's packed scalar width before narrowing to uint32."""

    array = np.asarray(value)
    if array.dtype.kind not in "ui":
        raise TypeError(f"{source} must provide unsigned packed integer words")
    if np.any(array < 0) or np.any(array > np.iinfo(np.uint32).max):
        raise ValueError(f"{source} contains packed words outside the uint32 SPF32 range")
    return np.asarray(array, dtype=np.uint32)


def _view_offsets(view: BlockView) -> tuple[float, float]:
    """Return lazy block log/phase offsets in their packed scalar units."""

    log_offset = float(getattr(view, "log_offset", 0.0))
    phase_offset = float(getattr(view, "phase_offset", 0.0))
    if not np.isfinite(log_offset) or not np.isfinite(phase_offset):
        raise ValueError("ocean block offsets must be finite")
    return log_offset, phase_offset


def _view_offsets_require_numeric(words: Any, view: BlockView) -> bool:
    """Whether a view offset must remain wide until a numeric boundary.

    An integral log shift is usually exactly representable by the SPF32
    transform, but ``transform_words`` clips finite log codes at the storage
    limits.  ``BlockView.project_words`` deliberately keeps such an exposure
    wide, so detect the same condition before an ocean packed consumer applies
    the offset.  The check is support-sized and never allocates a field plane.
    """

    log_offset, phase_offset = _view_offsets(view)
    if log_offset != round(log_offset) or phase_offset != round(phase_offset):
        return True
    if log_offset == 0.0:
        return False
    result = np.asarray(words, dtype=np.uint32).reshape(-1)
    if not result.size:
        return False
    log_codes = np.asarray(v02.unpack_L(result), dtype=np.int64)
    finite = (log_codes != int(v02.SPF32_L_ZERO)) & (result != np.uint32(v02.SPF32_WORD_NAN))
    if not np.any(finite):
        return False
    shifted = log_codes[finite].astype(np.float64) + float(log_offset)
    return bool(np.any(shifted < float(v02.SPF32_L_MIN_FINITE)) or np.any(shifted > float(v02.SPF32_L_MAX_FINITE)))


def _apply_integral_view_offsets(words: Any, view: BlockView) -> np.ndarray:
    """Apply exactly representable lazy offsets while retaining SPF words.

    Fractional offsets remain metadata for the numeric reference consumer;
    silently rounding them here would make a native/packed route claim a
    different source.  Integer offsets use the canonical packed transform and
    therefore preserve zero/NaN sentinels and the source's error ledger.
    """

    result = np.asarray(words, dtype=np.uint32)
    log_offset, phase_offset = _view_offsets(view)
    # Keep both components wide when the log component would clip.  Applying
    # only the phase in that case would also disagree with BlockView.project_words,
    # which composes the complete complex factor at its numeric boundary.
    if _view_offsets_require_numeric(result, view):
        return result
    if log_offset == round(log_offset) and phase_offset == round(phase_offset) and (log_offset or phase_offset):
        from spf.renderer.blocks import transform_words

        return np.asarray(transform_words(result, log_offset, phase_offset), dtype=np.uint32)
    return result


@dataclass(slots=True)
class OceanCarrier:
    """Owned structural ocean input with lazy packed-word materialization."""

    shape: tuple[int, ...]
    source_kind: str
    source_view: BlockView | None = None
    half_spectrum: np.ndarray | None = None
    coefficient_values: tuple[Any, ...] = ()
    index_values: tuple[tuple[int, ...], ...] = ()
    phase_values: tuple[Any, ...] = ()
    modes: tuple[Any, ...] = ()
    analytic_payload: Any = None
    half_view: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _packed_coefficients: np.ndarray | None = field(default=None, init=False, repr=False)
    _indices: np.ndarray | None = field(default=None, init=False, repr=False)
    _self_flags: np.ndarray | None = field(default=None, init=False, repr=False)
    _full_words: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.shape = _shape(self.shape)
        self.coefficient_values = tuple(self.coefficient_values)
        self.index_values = tuple(_normal_index(i, self.shape) for i in self.index_values)
        self.phase_values = tuple(self.phase_values)
        self.modes = tuple(self.modes)
        if self.half_spectrum is not None:
            half = np.asarray(self.half_spectrum).copy()
            half.setflags(write=False)
            self.half_spectrum = half
        self.metadata = MappingProxyType(dict(self.metadata))

    def half_words(self) -> np.ndarray:
        """Packed SPF32 half-spectrum words for a ``half_spectrum`` source.

        An integer-typed ``half_spectrum`` is already packed SPF32 and is used
        as-is; only a complex (Cartesian compatibility) half spectrum is
        encoded. Regression 2026-09-17: the packed words used to be cast to
        ``complex128`` and re-encoded, which turned every word into a
        ~2^31 magnitude coefficient (benchmarks reported relative L2 ≈ 2e9).
        """

        assert self.half_spectrum is not None
        half = np.asarray(self.half_spectrum)
        if half.dtype.kind in "iu":
            return np.asarray(half, dtype=np.uint32)
        return np.asarray(encode_words(np.asarray(half, dtype=np.complex128)), dtype=np.uint32)

    @classmethod
    def from_inputs(
        cls,
        shape: Sequence[int],
        *,
        coefficients: Any = None,
        indices: Any = None,
        phases: Any = None,
        half_spectrum: Any = None,
        modes: Sequence[Any] = (),
        source: Any = None,
    ) -> OceanCarrier:
        from spf.renderer.model import Field
        from spf.renderer.normalization import require_closed_field

        # All entry forms cross the same semantic boundary before any adapter
        # inspects storage. A Field's operations must never be lost by unwrapping.
        coefficients = (
            require_closed_field(coefficients, consumer="ocean") if isinstance(coefficients, Field) else coefficients
        )
        source = require_closed_field(source, consumer="ocean") if isinstance(source, Field) else source
        half_spectrum = (
            require_closed_field(half_spectrum, consumer="ocean") if isinstance(half_spectrum, Field) else half_spectrum
        )
        full_shape = _shape(shape)
        candidate = source
        if candidate is None and half_spectrum is not None:
            candidate = half_spectrum
        if candidate is None and coefficients is not None:
            possible = _view_from(coefficients)
            if possible is not None:
                candidate = coefficients
        view = _view_from(candidate) if candidate is not None else None
        if view is not None:
            carrier_shape = tuple(view.shape)
            half_view = carrier_shape == _half_shape(full_shape)
            if carrier_shape != full_shape and not half_view:
                raise ValueError(
                    "addressed ocean carrier shape "
                    f"{carrier_shape} must equal {full_shape} or {_half_shape(full_shape)}"
                )
            metadata = _source_metadata(
                shape=full_shape,
                kind="block_view",
                view=view,
                half=None,
                indices=(),
                coefficients=(),
                modes=tuple(modes),
                half_view=half_view,
            )
            source_representation = getattr(candidate, "representation", None)
            if getattr(source_representation, "residual", None) is not None:
                raise ValueError("ocean block source requires an explicit representation residual consumer")
            source_error = getattr(source_representation, "error", None)
            if source_error is not None:
                metadata = MappingProxyType(
                    {
                        **dict(metadata),
                        "error_budget": float(getattr(source_error, "total_bound", 0.0)),
                        "error_ledger": source_error,
                    }
                )
            if phases is not None:
                metadata = MappingProxyType(
                    {
                        **dict(metadata),
                        "phase_state_supplied": True,
                        "phase_state_count": len(_as_tuple(phases)),
                        "phase_composition": "deferred_wide_phase_consumer",
                    }
                )
            return cls(
                full_shape,
                "block_view",
                source_view=view,
                phase_values=_as_tuple(phases) if phases is not None else (),
                half_view=half_view,
                modes=tuple(modes),
                metadata=metadata,
            )
        # Analytic atom/jet inputs are explicit carriers.  No spectrum is
        # inferred from them and no parameter is sampled while planning.
        analytic = candidate
        if analytic is None and coefficients is not None and indices is None:
            analytic = coefficients
        # A Field is a metadata envelope, not a dense array.  Keep the
        # envelope as ``analytic_payload`` so the canonical analytic consumer
        # receives its chart, support, residual, and representation ledger.
        # Block-backed Fields have already returned through ``view`` above.
        analytic_value = analytic
        field_metadata: dict[str, Any] = {}
        if isinstance(analytic, Field):
            from spf.renderer.normalization import complete_field_ledger

            analytic_value = analytic.representation.payload
            field_metadata = {
                "field_id": analytic.id,
                "field_domain": str(analytic.domain),
                "field_metadata": dict(analytic.metadata),
                "field_representation": str(analytic.representation.kind),
                "field_representation_metadata": dict(analytic.representation.metadata),
                "field_error_ledger": complete_field_ledger(analytic),
            }
        if analytic is not None and (
            hasattr(analytic_value, "ocean_fields")
            or hasattr(analytic_value, "execute_fields")
            or hasattr(analytic_value, "evaluate_with_gradients")
            or analytic_value.__class__.__name__
            in {"GaussianChirpAtom", "AtomMixture", "Jet2D", "SpectralJetTile2D", "AdaptiveJetField"}
        ):
            metadata = MappingProxyType(
                {
                    "source_kind": "analytic",
                    "shape": full_shape,
                    "potential_elements": int(math.prod(full_shape)),
                    "active_modes": 0,
                    "structure_preserved": True,
                    "packed_storage": False,
                    "source_conversion": "none",
                    "analytic_type": type(analytic_value).__name__,
                    "residual_preserved": bool(
                        getattr(analytic_value, "residual", None) is not None
                        or (
                            isinstance(analytic, Field)
                            and getattr(analytic.representation, "residual", None) is not None
                        )
                    ),
                    "error_budget": float(getattr(analytic_value, "dropped_bound", 0.0)),
                    **field_metadata,
                }
            )
            return cls(full_shape, "analytic", analytic_payload=analytic, modes=tuple(modes), metadata=metadata)
        if half_spectrum is not None:
            half = np.asarray(half_spectrum)
            expected = _half_shape(full_shape)
            if half.shape != expected:
                raise ValueError(f"half spectrum shape {half.shape} does not match expected {expected}")
            packed_input = half.dtype.kind in "iu"
            if packed_input:
                half = np.asarray(half, dtype=np.uint32)
                if np.any(half == np.uint32(v02.SPF32_WORD_NAN)):
                    raise ValueError("ocean half spectrum contains NaN SPF32 words")
            elif not np.all(np.isfinite(half)):
                raise ValueError("ocean half spectrum contains non-finite coefficients")
            metadata = _source_metadata(
                shape=full_shape,
                kind="half_spectrum",
                view=None,
                half=half,
                indices=(),
                coefficients=(),
                modes=tuple(modes),
                half_view=True,
            )
            if packed_input:
                constant_value = complex(v02.decode_complex(int(half.reshape(-1)[0]))) if half.size else 0.0j
            else:
                constant_value = complex(half.reshape(-1)[0]) if half.size else 0.0j
            if half.size and bool(np.all(half == half.reshape(-1)[0])) and abs(constant_value.imag) <= 1.0e-12:
                updated = dict(metadata)
                updated["closed_form"] = "constant_half"
                updated["closed_form_value"] = constant_value.real
                metadata = MappingProxyType(updated)
            return cls(
                full_shape, "half_spectrum", half_spectrum=half, modes=tuple(modes), half_view=True, metadata=metadata
            )
        if indices is None or coefficients is None:
            raise ValueError("ocean requires a half spectrum, addressed carrier, or sparse coefficients and indices")
        idx = _normal_indices(indices, full_shape)
        coeff = _as_tuple(coefficients, len(idx))
        pvals = _as_tuple(phases, len(idx)) if phases is not None else ()
        metadata = _source_metadata(
            shape=full_shape,
            kind="sparse_modes",
            view=None,
            half=None,
            indices=idx,
            coefficients=coeff,
            modes=tuple(modes),
            half_view=False,
        )
        values = dict(metadata)
        if len(full_shape) == 2 and idx and all(index[0] == index[1] for index in idx):
            values["support_pattern"] = "diagonal"
        values["source_conversion"] = (
            "cartesian_sparse_compatibility"
            if any(not isinstance(x, (int, np.integer, v02.SPF32Value, v02.LogPos, v02.LogSigned)) for x in coeff)
            else "packed_sparse_words"
        )
        return cls(
            full_shape,
            "sparse_modes",
            coefficient_values=coeff,
            index_values=idx,
            phase_values=pvals,
            modes=tuple(modes),
            metadata=MappingProxyType(values),
        )

    @property
    def potential_elements(self) -> int:
        return int(math.prod(self.shape))

    @property
    def active_count(self) -> int:
        if self.source_kind == "sparse_modes":
            return len(self.index_values)
        if self.source_kind == "half_spectrum" and self.half_spectrum is not None:
            words = self.half_words()
            return int(np.count_nonzero(words != np.uint32(v02.SPF32_WORD_ZERO)))
        return int(self.metadata.get("support_count", self.potential_elements))

    @property
    def structure_certificate(self) -> Mapping[str, Any]:
        return self.metadata

    @property
    def indices(self) -> np.ndarray:
        """Canonical full-grid indices, fetched only for the selected sparse route."""

        self.materialize_sparse_packed()
        assert self._indices is not None
        return self._indices.copy()

    @property
    def packed_coefficients(self) -> np.ndarray:
        """Read-only packed SPF32 words for the selected sparse route."""

        self.materialize_sparse_packed()
        assert self._packed_coefficients is not None
        return self._packed_coefficients.copy()

    @property
    def wide_phases(self) -> np.ndarray:
        if not self.modes:
            return np.empty(0, dtype=object)
        return np.asarray([m.phase_accumulator.copy() for m in self.modes], dtype=object)

    def _set_sparse_cache(self, indices: np.ndarray, words: np.ndarray, self_flags: np.ndarray) -> None:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1, len(self.shape))
        words = np.asarray(words, dtype=np.uint32).reshape(-1)
        self_flags = np.asarray(self_flags, dtype=bool).reshape(-1)
        indices.setflags(write=False)
        words.setflags(write=False)
        self_flags.setflags(write=False)
        self._indices, self._packed_coefficients, self._self_flags = indices, words, self_flags

    def materialize_sparse_packed(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch only the selected sparse support as ``(indices, words, self)``."""

        if self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None:
            return self._indices, self._packed_coefficients, self._self_flags
        if self.source_kind == "analytic":
            self._set_sparse_cache(
                np.empty((0, len(self.shape)), dtype=np.int64), np.empty(0, dtype=np.uint32), np.empty(0, dtype=bool)
            )
            assert self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None
            return self._indices, self._packed_coefficients, self._self_flags
        if self.source_kind == "sparse_modes":
            from spf.renderer import ocean as o

            values: list[int] = []
            indices: list[tuple[int, ...]] = []
            flags: list[bool] = []
            for position, index in enumerate(self.index_values):
                word = _word_from_value(self.coefficient_values[position])
                phase = self.phase_values[position] if self.phase_values else None
                if phase is not None:
                    code = o._phase_radians(phase) * (v02.SPF32_PHASE_MOD / (2 * math.pi))
                    word = int(v02.spf_phase_shift(word, round(code)))
                indices.append(index)
                values.append(word)
                flags.append(o._is_self_conjugate(index, self.shape))
            self._set_sparse_cache(
                np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.uint32), np.asarray(flags, dtype=bool)
            )
            assert self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None
            return self._indices, self._packed_coefficients, self._self_flags
        if self.source_kind == "half_spectrum":
            return self._sparse_from_half_words(self.half_words())
        assert self.source_view is not None
        if int(self.source_view.scalar_bits) != 32:
            raise TypeError(
                "ocean packed materialization requires a 32-bit SPF view; "
                f"received scalar_bits={self.source_view.scalar_bits}. "
                "Convert explicitly to SPF32 and attach its representation error ledger."
            )
        addresses = np.asarray(self.source_view.active_addresses(), dtype=np.int64).reshape(-1)
        words = _validated_spf32_words(
            self.source_view.sample_words(addresses, phase=True),
            source="addressed SPF32 view",
        ).reshape(-1)
        words = _apply_integral_view_offsets(words, self.source_view)
        if self.half_view:
            # Keep an addressed half-plane as ``O(K)`` entries.  The previous
            # implementation scattered into an ``N``-element sentinel plane
            # and then used ``argwhere``; that defeats TensorBlock/BlockView
            # structure even when the selected sparse route needs only a few
            # words.
            return self._sparse_from_half_entries(addresses, words)
        # Full-grid block views have the same addressed contract.  Build an
        # index→word map only for the supplied support, then validate any
        # explicitly supplied Hermitian partners.  A lone full-grid entry is
        # legal as a sparse authored mode and is completed by the consumer.
        return self._sparse_from_full_entries(addresses, words)

    def _sparse_from_half_entries(
        self, addresses: np.ndarray, words: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Canonicalize addressed half-plane words without a dense plane."""

        from spf.renderer import ocean as o

        half_shape = _half_shape(self.shape)
        entries = {
            tuple(int(value) for value in np.unravel_index(int(address), half_shape)): int(word)
            for address, word in zip(addresses, words, strict=True)
            if int(word) != int(v02.SPF32_WORD_ZERO)
        }
        indices: list[tuple[int, ...]] = []
        values: list[int] = []
        flags: list[bool] = []
        seen: set[tuple[int, ...]] = set()
        for index in sorted(entries):
            if index in seen:
                continue
            word = entries[index]
            partner = o._partner(index, self.shape)
            if o._endpoint_last(index, self.shape):
                partner_word = entries.get(partner)
                if partner_word is None:
                    if index != partner:
                        raise o.HermitianValidationError(f"missing Hermitian endpoint partner {index}/{partner}")
                elif partner_word != int(v02.spf_conj(word)):
                    raise o.HermitianValidationError(f"Hermitian endpoint mismatch {index}/{partner}")
            if index == partner:
                decoded = complex(v02.decode_complex(word))
                if abs(decoded.imag) > 1.0e-12:
                    raise o.HermitianValidationError(f"DC/Nyquist coefficient {index} must remain real")
            if partner in entries:
                if partner != index and entries[partner] != int(v02.spf_conj(word)):
                    raise o.HermitianValidationError(f"Hermitian endpoint mismatch {index}/{partner}")
                seen.add(partner)
            seen.add(index)
            indices.append(index)
            values.append(word)
            flags.append(o._is_self_conjugate(index, self.shape))
        self._set_sparse_cache(
            np.asarray(indices, dtype=np.int64).reshape(-1, len(self.shape)),
            np.asarray(values, dtype=np.uint32),
            np.asarray(flags, dtype=bool),
        )
        assert self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None
        return self._indices, self._packed_coefficients, self._self_flags

    def _sparse_from_full_entries(
        self, addresses: np.ndarray, words: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Canonicalize addressed full-grid words using only ``O(K)`` state."""

        from spf.renderer import ocean as o

        entries = {
            tuple(int(value) for value in np.unravel_index(int(address), self.shape)): int(word)
            for address, word in zip(addresses, words, strict=True)
            if int(word) != int(v02.SPF32_WORD_ZERO)
        }
        indices: list[tuple[int, ...]] = []
        values: list[int] = []
        flags: list[bool] = []
        seen: set[tuple[int, ...]] = set()
        for index in sorted(entries):
            if index in seen:
                continue
            word = entries[index]
            partner = o._partner(index, self.shape)
            if index == partner:
                decoded = complex(v02.decode_complex(word))
                if abs(decoded.imag) > 1.0e-12:
                    raise o.HermitianValidationError(f"self-conjugate coefficient {index} must remain real")
            elif partner in entries:
                partner_word = entries[partner]
                if partner_word != int(v02.spf_conj(word)):
                    raise o.HermitianValidationError(f"Hermitian partner mismatch {index}/{partner}")
                # Choose one stable representative when both orbit members
                # were supplied.  Keeping the lexicographically smaller
                # address matches the half-plane sparse convention.
                if partner < index:
                    seen.add(index)
                    continue
                seen.add(partner)
            seen.add(index)
            indices.append(index)
            values.append(word)
            flags.append(o._is_self_conjugate(index, self.shape))
        self._set_sparse_cache(
            np.asarray(indices, dtype=np.int64).reshape(-1, len(self.shape)),
            np.asarray(values, dtype=np.uint32),
            np.asarray(flags, dtype=bool),
        )
        assert self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None
        return self._indices, self._packed_coefficients, self._self_flags

    def _sparse_from_half_words(self, words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from spf.renderer import ocean as o

        full_shape = self.shape
        support = np.argwhere(words != np.uint32(v02.SPF32_WORD_ZERO))
        indices: list[tuple[int, ...]] = []
        values: list[int] = []
        flags: list[bool] = []
        for address in support:
            index = tuple(int(x) for x in address)
            if index[-1] > full_shape[-1] // 2:
                continue
            if o._endpoint_last(index, full_shape):
                partner = o._partner(index, full_shape)
                if index > partner:
                    continue
                if index != partner and words[partner] == np.uint32(v02.SPF32_WORD_ZERO):
                    # The endpoint partner is in the same half plane; preserve
                    # the explicit validation failure for malformed input.
                    raise o.HermitianValidationError(f"missing Hermitian endpoint partner {index}/{partner}")
                if index == partner:
                    decoded = complex(v02.decode_complex(int(words[index])))
                    if abs(decoded.imag) > 1e-12:
                        raise o.HermitianValidationError(f"DC/Nyquist coefficient {index} must remain real")
            indices.append(index)
            values.append(int(words[index]))
            flags.append(o._is_self_conjugate(index, full_shape))
        self._set_sparse_cache(
            np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.uint32), np.asarray(flags, dtype=bool)
        )
        assert self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None
        return self._indices, self._packed_coefficients, self._self_flags

    def _sparse_from_full_words(self, words: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        from spf.renderer import ocean as o

        support = np.argwhere(words != np.uint32(v02.SPF32_WORD_ZERO))
        seen: set[tuple[int, ...]] = set()
        indices: list[tuple[int, ...]] = []
        values: list[int] = []
        flags: list[bool] = []
        for address in support:
            index = tuple(int(x) for x in address)
            partner = o._partner(index, self.shape)
            orbit = min(index, partner)
            if orbit in seen:
                continue
            seen.add(orbit)
            if index != orbit:
                if words.flat[np.ravel_multi_index(partner, self.shape)] == np.uint32(v02.SPF32_WORD_ZERO):
                    raise o.HermitianValidationError(f"missing Hermitian partner {index}/{partner}")
                values.append(int(words[index]))
            else:
                values.append(int(words[index]))
            indices.append(index)
            flags.append(index == partner)
        self._set_sparse_cache(
            np.asarray(indices, dtype=np.int64), np.asarray(values, dtype=np.uint32), np.asarray(flags, dtype=bool)
        )
        assert self._indices is not None and self._packed_coefficients is not None and self._self_flags is not None
        return self._indices, self._packed_coefficients, self._self_flags

    def materialize_half_packed(self) -> np.ndarray:
        """Fetch/own an SPF32 half plane only for a dense transform route."""

        if self._full_words is not None and self._full_words.shape == _half_shape(self.shape):
            return self._full_words.copy()
        if self.source_kind == "half_spectrum":
            result = self.half_words()
            result.setflags(write=False)
            self._full_words = result
            return result.copy()
        if self.source_kind == "sparse_modes":
            from spf.renderer import ocean as o

            # This is an explicit dense route; building the Hermitian half
            # plane is charged to the route and never happens while planning.
            half = np.full(_half_shape(self.shape), np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32)
            indices, words, flags = self.materialize_sparse_packed()
            for index, word, _self_flag in zip(indices, words, flags, strict=True):
                idx = tuple(int(x) for x in index)
                partner = o._partner(idx, self.shape)
                destination = partner if idx[-1] > self.shape[-1] // 2 else idx
                half[destination] = np.uint32(v02.spf_conj(word) if idx[-1] > self.shape[-1] // 2 else word)
                if o._endpoint_last(destination, self.shape) and destination != o._partner(destination, self.shape):
                    half[o._partner(destination, self.shape)] = np.uint32(v02.spf_conj(half[destination]))
            self._full_words = half
            return half.copy()
        assert self.source_view is not None
        if int(self.source_view.scalar_bits) != 32:
            raise TypeError(
                "ocean dense SPF materialization requires a 32-bit SPF view; "
                f"received scalar_bits={self.source_view.scalar_bits}. "
                "Convert explicitly to SPF32 and attach its representation error ledger."
            )
        addresses = np.asarray(self.source_view.active_addresses(), dtype=np.int64).reshape(-1)
        words = _validated_spf32_words(
            self.source_view.sample_words(addresses, phase=True),
            source="addressed SPF32 view",
        ).reshape(-1)
        words = _apply_integral_view_offsets(words, self.source_view)
        half_destination = np.full(_half_shape(self.shape), np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32)
        if self.half_view:
            half_destination.flat[addresses] = words
        else:
            for address, word in zip(addresses, words, strict=True):
                index = np.unravel_index(int(address), self.shape)
                if index[-1] > self.shape[-1] // 2:
                    idx = tuple((-int(x)) % int(size) for x, size in zip(index, self.shape, strict=True))
                    half_destination[idx] = np.uint32(v02.spf_conj(int(word)))
                else:
                    half_destination[index] = word
        self._full_words = half_destination
        return half_destination.copy()

    def derivative_operators(self, spacing: Sequence[float] | float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return packed derivative metadata ``(ΔL, ΔΘ, zero)`` per support bin.

        ``ΔΘ`` is an exact quarter turn (or its modular negative) and ``ΔL``
        is the rounded SPF32 log shift for the physical frequency multiplier.
        The arrays are metadata for the selected addressed support; no spatial
        or Cartesian coefficient plane is created.
        """

        indices, _, _ = self.materialize_sparse_packed()
        values = np.asarray(spacing, dtype=np.float64)
        if values.ndim == 0:
            values = np.full(len(self.shape), float(values), dtype=np.float64)
        if values.shape != (len(self.shape),) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError("derivative spacing must be finite, positive, and match the shape")
        shifts = np.zeros((len(self.shape), len(indices)), dtype=np.int64)
        phases = np.zeros((len(self.shape), len(indices)), dtype=np.int64)
        zeros = np.zeros((len(self.shape), len(indices)), dtype=bool)
        for axis, (size, step) in enumerate(zip(self.shape, values, strict=True)):
            for position, index in enumerate(indices):
                frequency = int(index[axis] if index[axis] <= size // 2 else index[axis] - size)
                if size % 2 == 0 and abs(frequency) == size // 2:
                    zeros[axis, position] = True
                    continue
                magnitude = abs(2.0 * math.pi * frequency / (size * step))
                if magnitude == 0.0:
                    zeros[axis, position] = True
                    continue
                shifts[axis, position] = round(256.0 * math.log2(magnitude))
                phases[axis, position] = v02.SPF32_PHASE_QUARTER if frequency > 0 else 3 * v02.SPF32_PHASE_QUARTER
        for array in (shifts, phases, zeros):
            array.setflags(write=False)
        return shifts, phases, zeros


__all__ = ["OceanCarrier"]
