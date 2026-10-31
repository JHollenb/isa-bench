"""OFDM resource grid built on Level-4 SPF TensorBlock invariants."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum

import numpy as np
import numpy.typing as npt

from spf.tensor import (
    SPF32_L_ZERO,
    SPF32_WORD_ZERO,
    ActiveEncoding,
    AffineTensorBlock,
    OperatorHeader,
    Profile,
    ReducerPolicy,
    SPFError,
    circular_diff_u16,
    pack_words,
    unpack_L,
    unpack_theta,
)


class OFDMBinClass(IntEnum):
    """Semantic class for one spectral resource element."""

    DATA = 1
    PILOT = 2
    GUARD = 3
    DC_NULL = 4
    INACTIVE = 5


ACTIVE_BIN_CLASSES = frozenset({OFDMBinClass.DATA, OFDMBinClass.PILOT})
NULL_BIN_CLASSES = frozenset({OFDMBinClass.GUARD, OFDMBinClass.DC_NULL, OFDMBinClass.INACTIVE})


@dataclass(frozen=True, slots=True)
class OFDMResourceGrid:
    """SPF spectral resource grid with explicit OFDM bin classes.

    The physical shape is always ``[frame, symbol, subcarrier, antenna_or_beam]``.
    ``origin`` is a logical circulation offset over that shape, matching the
    Level-4 TensorBlock header contract.
    """

    words: npt.NDArray[np.uint32]
    bin_class: npt.NDArray[np.int16]
    origin: tuple[int, int, int, int] = (0, 0, 0, 0)
    reducer_id: ReducerPolicy = ReducerPolicy.NONE

    def __post_init__(self) -> None:
        words = np.ascontiguousarray(np.asarray(self.words), dtype=np.uint32)
        classes = np.ascontiguousarray(np.asarray(self.bin_class), dtype=np.int16)
        object.__setattr__(self, "words", words)
        object.__setattr__(self, "bin_class", classes)
        object.__setattr__(self, "origin", tuple(int(v) for v in self.origin))
        object.__setattr__(self, "reducer_id", ReducerPolicy(self.reducer_id))
        self.validate()

    @classmethod
    def from_trusted(
        cls,
        words: npt.NDArray[np.uint32],
        bin_class: npt.NDArray[np.int16],
        *,
        origin: tuple[int, int, int, int] = (0, 0, 0, 0),
        reducer_id: ReducerPolicy = ReducerPolicy.NONE,
    ) -> OFDMResourceGrid:
        """Construct a grid from pre-validated arrays, skipping ``validate()``.

        Callers must guarantee that ``words`` is ``uint32`` and ``bin_class`` is
        ``int16`` with matching 4-D shapes, and that the bin-class / SPF-zero
        invariants checked by :meth:`validate` already hold. Used by hot-path
        constructors (TX cascade, channel apply, FFT staging) where the
        invariants are established one line above the construction.
        """
        obj = object.__new__(cls)
        object.__setattr__(obj, "words", np.ascontiguousarray(words, dtype=np.uint32))
        object.__setattr__(
            obj, "bin_class", np.ascontiguousarray(bin_class, dtype=np.int16)
        )
        object.__setattr__(obj, "origin", tuple(int(v) for v in origin))
        object.__setattr__(obj, "reducer_id", ReducerPolicy(reducer_id))
        return obj

    @classmethod
    def from_planes(
        cls,
        L: npt.ArrayLike,
        theta: npt.ArrayLike,
        bin_class: npt.ArrayLike,
        *,
        origin: tuple[int, int, int, int] = (0, 0, 0, 0),
        reducer_id: ReducerPolicy = ReducerPolicy.NONE,
    ) -> OFDMResourceGrid:
        """Construct a grid from logical L/T planes."""
        return cls(pack_words(L, theta), np.asarray(bin_class), origin, reducer_id)

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, int, int, int],
        *,
        bin_class: OFDMBinClass = OFDMBinClass.INACTIVE,
        reducer_id: ReducerPolicy = ReducerPolicy.NONE,
    ) -> OFDMResourceGrid:
        """Create a zero-sentinel grid with one repeated bin class."""
        return cls(
            np.full(shape, SPF32_WORD_ZERO, dtype=np.uint32),
            np.full(shape, int(bin_class), dtype=np.int16),
            reducer_id=reducer_id,
        )

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return tuple(int(v) for v in self.words.shape)  # type: ignore[return-value]

    @property
    def l_plane(self) -> npt.NDArray[np.int16]:
        return unpack_L(self.words)

    @property
    def theta(self) -> npt.NDArray[np.uint16]:
        return unpack_theta(self.words)

    def validate(self) -> None:
        if self.words.ndim != 4:
            raise SPFError(
                "OFDMResourceGrid shape must be [frame][symbol][subcarrier][antenna_or_beam]"
            )
        if self.bin_class.shape != self.words.shape:
            raise SPFError("bin_class plane must match resource grid shape")
        if len(self.origin) != 4:
            raise SPFError("origin must have four dimensions")
        if any(dim <= 0 for dim in self.words.shape):
            raise SPFError("resource grid dimensions must be positive")
        if any(
            origin < 0 or origin >= dim
            for origin, dim in zip(self.origin, self.words.shape, strict=True)
        ):
            raise SPFError("origin entry outside grid dimension")
        valid = np.asarray([int(cls) for cls in OFDMBinClass], dtype=np.int16)
        if not np.isin(self.bin_class, valid).all():
            raise SPFError("bin_class contains an unknown OFDM bin class")
        l_plane = self.l_plane
        null_mask = np.isin(
            self.bin_class,
            np.asarray([int(cls) for cls in NULL_BIN_CLASSES], dtype=np.int16),
        )
        if np.any(self.words[null_mask] != SPF32_WORD_ZERO):
            raise SPFError("inactive, guard, and DC-null bins must be SPF zero sentinel")
        active_mask = np.isin(
            self.bin_class,
            np.asarray([int(cls) for cls in ACTIVE_BIN_CLASSES], dtype=np.int16),
        )
        if np.any(l_plane[active_mask] == SPF32_L_ZERO):
            raise SPFError("data and pilot bins must carry valid nonzero SPF words")
        if self.reducer_id == ReducerPolicy.NONE:
            return
        ReducerPolicy(self.reducer_id)

    def active_mask(self) -> npt.NDArray[np.bool_]:
        return np.isin(
            self.bin_class,
            np.asarray([int(cls) for cls in ACTIVE_BIN_CLASSES], dtype=np.int16),
        )

    def active_subcarriers(self) -> npt.NDArray[np.uint32]:
        """Return the union of active subcarriers across frames, symbols, and beams."""
        mask = np.isin(
            self.logical_bin_class(),
            np.asarray([int(cls) for cls in ACTIVE_BIN_CLASSES], dtype=np.int16),
        )
        active = np.flatnonzero(np.any(mask, axis=(0, 1, 3))).astype(np.uint32)
        return np.ascontiguousarray(active, dtype=np.uint32)

    def shift_view(self, dim: int, delta: int) -> OFDMResourceGrid:
        if dim < 0 or dim >= 4:
            raise SPFError("shift dimension out of range")
        origin = list(self.origin)
        origin[dim] = (origin[dim] + int(delta)) % self.shape[dim]
        return replace(self, origin=(origin[0], origin[1], origin[2], origin[3]))

    def logical_words(self) -> npt.NDArray[np.uint32]:
        """Materialize the logical view represented by ``origin``."""
        out = self.words
        for axis, offset in enumerate(self.origin):
            if offset:
                out = np.roll(out, shift=-offset, axis=axis)
        return np.ascontiguousarray(out, dtype=np.uint32)

    def logical_bin_class(self) -> npt.NDArray[np.int16]:
        out = self.bin_class
        for axis, offset in enumerate(self.origin):
            if offset:
                out = np.roll(out, shift=-offset, axis=axis)
        return np.ascontiguousarray(out, dtype=np.int16)

    def phase_delta(
        self, other: OFDMResourceGrid | npt.ArrayLike
    ) -> npt.NDArray[np.int32] | int:
        """Signed circular phase difference against another grid or theta plane."""
        rhs = other.theta if isinstance(other, OFDMResourceGrid) else other
        return circular_diff_u16(self.theta, rhs)

    def require_reducer(self, reducer_id: ReducerPolicy | None = None) -> ReducerPolicy:
        """Validate reducer policy for a contraction or summation edge."""
        policy = self.reducer_id if reducer_id is None else ReducerPolicy(reducer_id)
        if policy == ReducerPolicy.NONE:
            raise SPFError("OFDM contractions and summations require an explicit reducer_id")
        return policy

    def to_tensor_block(self) -> AffineTensorBlock:
        """Encode live resource elements as a sparse Level-4 affine TensorBlock."""
        self.validate()
        flat_words = self.words.reshape(-1)
        flat_L = unpack_L(flat_words)
        live = np.flatnonzero(flat_L != SPF32_L_ZERO).astype(np.uint32)
        return AffineTensorBlock(
            header=OperatorHeader(
                profile_id=Profile.STB_AFFINE,
                reducer_id=self.reducer_id,
                dims=self.shape,
                origin=self.origin,
            ),
            active_encoding=ActiveEncoding.COO_U32,
            active_indices=live,
            dL=flat_L[live],
            dT=unpack_theta(flat_words)[live].view(np.int16),
        )

    @classmethod
    def from_tensor_block(
        cls,
        block: AffineTensorBlock,
        bin_class: npt.ArrayLike,
        *,
        reducer_id: ReducerPolicy | None = None,
    ) -> OFDMResourceGrid:
        """Reconstruct an OFDM grid from a Level-4 affine TensorBlock."""
        L, theta = block.reconstruct()
        effective_reducer = (
            ReducerPolicy(block.header.reducer_id)
            if reducer_id is None
            else ReducerPolicy(reducer_id)
        )
        return cls.from_planes(
            L,
            theta,
            bin_class,
            origin=_header_origin4(block.header.origin),
            reducer_id=effective_reducer,
        )


__all__ = [
    "ACTIVE_BIN_CLASSES",
    "NULL_BIN_CLASSES",
    "OFDMBinClass",
    "OFDMResourceGrid",
]


def _header_origin4(origin: tuple[int, ...] | None) -> tuple[int, int, int, int]:
    if origin is None:
        return (0, 0, 0, 0)
    padded = (*tuple(int(v) for v in origin[:4]), 0, 0, 0, 0)
    return (padded[0], padded[1], padded[2], padded[3])
