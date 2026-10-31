"""Opt-in finite phase-program compilation over canonical SPF32 word planes.

Tables contain existing packed SPF signals; address indexes and counts are
integer lookup/reducer control metadata. No new numerical carrier is introduced.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from spf.scalar import spf32

from . import core, v02_planes
from .phase_circuits import _integer
from .sequence_mixer import PhaseFeatureNetwork


class FeatureCache:
    @staticmethod
    def key(network, alphabet):
        program = Path(sys.modules[network.__class__.__module__].__file__).read_bytes()
        return hashlib.sha256(
            b"SPF32-ABI24|address-cache-v1|"
            + program
            + repr(network).encode()
            + repr(network._plan).encode()
            + str(alphabet.shape).encode()
            + str(alphabet.dtype).encode()
            + alphabet.tobytes()
        ).hexdigest()

    def __init__(self, network, alphabet):
        alphabet = self._initialize_table(network, alphabet)
        self.index = {tuple(int(v) for v in row): i for i, row in enumerate(alphabet)}
        if len(self.index) != len(alphabet):
            raise ValueError("duplicate alphabet entries")

    def _initialize_table(self, network, alphabet):
        if not isinstance(network, PhaseFeatureNetwork):
            raise TypeError("PhaseFeatureNetwork required")
        alphabet = np.asarray(alphabet)
        if alphabet.dtype != np.uint32 or alphabet.ndim != 2 or alphabet.shape[1] != network.n_channels:
            raise ValueError("alphabet must be an SPF32 word matrix matching the program")
        if not len(alphabet) or np.any(core.unpack_L(alphabet) != 0):
            raise ValueError("nonempty finite unit-circle alphabet required")
        self.fingerprint = self.key(network, alphabet)
        self.table = network.execute(alphabet)
        self.table.setflags(write=False)
        return alphabet

    def execute(self, network, alphabet, inputs):
        inputs = np.asarray(inputs)
        if inputs.dtype != np.uint32 or inputs.ndim != 2 or inputs.shape[1:] != alphabet.shape[1:]:
            raise ValueError("address shape or dtype mismatch")
        key = self.key(network, alphabet)
        if key != self.fingerprint:
            raise ValueError("stale feature parameters or alphabet")
        ids = np.fromiter((self.index[tuple(int(v) for v in row)] for row in inputs), dtype=np.int64, count=len(inputs))
        return self.table[ids]


class VectorFeatureCache(FeatureCache):
    """Compiled finite alphabet with vectorized full-word address lookup."""

    def __init__(self, network, alphabet):
        super().__init__(network, alphabet)
        self.word_dtype = alphabet.dtype
        self.key_dtype = np.dtype([(f"c{i}", alphabet.dtype) for i in range(alphabet.shape[1])])
        keys = np.ascontiguousarray(alphabet).view(self.key_dtype).reshape(-1)
        self.order = np.argsort(keys)
        self.keys = keys[self.order]

    def execute(self, network, alphabet, inputs):
        if self.key(network, alphabet) != self.fingerprint:
            raise ValueError("stale feature parameters or alphabet")
        if inputs.dtype != self.word_dtype or inputs.ndim != 2 or inputs.shape[1:] != alphabet.shape[1:]:
            raise ValueError("address shape or dtype mismatch")
        keys = np.ascontiguousarray(inputs).view(self.key_dtype).reshape(-1)
        positions = np.searchsorted(self.keys, keys)
        valid = positions < len(self.keys)
        if not np.all(valid):
            raise KeyError("address outside compiled alphabet")
        if not np.array_equal(self.keys[positions], keys):
            raise KeyError("address outside compiled alphabet")
        return self.table[self.order[positions]]


class GridFeatureCache(FeatureCache):
    """Compile aligned phase axes into a guarded row-major address law.

    Each axis uses a distinct channel and a power-of-two radix <= 65536.
    The caller supplies the entire finite alphabet; compilation stores its
    feature plane. A direct grid law needs no per-address Python dictionary.
    """

    def __init__(self, network, alphabet, *, axis_sizes=None, address_channels=None):
        alphabet = np.asarray(alphabet)
        if alphabet.dtype != np.uint32 or alphabet.ndim != 2:
            raise ValueError("alphabet must be a packed SPF32 word matrix")
        sizes = (len(alphabet),) if axis_sizes is None else tuple(_integer(v) for v in axis_sizes)
        channels = (
            tuple(range(len(sizes))) if address_channels is None else tuple(_integer(v) for v in address_channels)
        )
        if not sizes or any(v < 1 or v > 65536 or v & (v - 1) for v in sizes):
            raise ValueError("each grid axis must have a power-of-two size at most 65536")
        if len(channels) != len(sizes) or len(set(channels)) != len(channels):
            raise ValueError("one distinct address channel is required per grid axis")
        if any(c < 0 or c >= alphabet.shape[1] for c in channels):
            raise ValueError("address channel outside alphabet shape")
        if math.prod(sizes) != len(alphabet):
            raise ValueError("alphabet length must equal the product of grid axis sizes")
        self.axis_sizes = sizes
        self.address_channels = channels
        self.shifts = tuple(16 - (v.bit_length() - 1) for v in sizes)
        self.strides = tuple(math.prod(sizes[i + 1 :]) for i in range(len(sizes)))
        # The compiled table stays in declared row-major order. Address laws
        # index this existing word plane; they do not enumerate an unoccupied
        # Cartesian address space on behalf of the caller.
        ids = np.arange(len(alphabet), dtype=np.int64)
        for channel, size, stride, shift in zip(channels, sizes, self.strides, self.shifts, strict=True):
            expected = ((ids // stride) % size) << shift
            if not np.array_equal(core.unpack_theta(alphabet[:, channel]), expected):
                raise ValueError("address channels must form an aligned row-major phase grid")
        # The verified full grid has distinct address tuples by construction.
        # Compile the numerical table without the generic dictionary index.
        self._initialize_table(network, alphabet)
        self.shift = self.shifts[0]  # Backward-compatible single-axis metadata.

    def execute(self, network, alphabet, inputs):
        if self.key(network, alphabet) != self.fingerprint:
            raise ValueError("stale feature parameters or alphabet")
        if inputs.dtype != alphabet.dtype or inputs.ndim != 2 or inputs.shape[1:] != alphabet.shape[1:]:
            raise ValueError("address shape or dtype mismatch")
        return self.table[self.address_ids(alphabet, inputs)]

    def address_ids(self, alphabet, inputs):
        """Guard the full word tuple after extracting its aligned phase address."""
        inputs = np.asarray(inputs)
        if inputs.dtype != np.uint32 or inputs.ndim != 2 or inputs.shape[1:] != alphabet.shape[1:]:
            raise ValueError("address shape or dtype mismatch")
        if len(self.address_channels) == 1:
            ids = core.unpack_theta(inputs[:, self.address_channels[0]]) >> self.shifts[0]
        else:
            ids = np.zeros(len(inputs), dtype=np.int64)
            for channel, stride, shift in zip(self.address_channels, self.strides, self.shifts, strict=True):
                ids += (core.unpack_theta(inputs[:, channel]).astype(np.int64) >> shift) * stride
        if not np.array_equal(alphabet[ids], inputs):
            raise KeyError("address outside compiled alphabet")
        return ids


class FinitePhaseFeatureMemory:
    """Exchangeable feature pooling over an owned aligned finite alphabet.

    This is not an ordered recurrence, arbitrary payload store, or attention
    replacement. Counts are reducer multiplicities. Read rounds each unique
    coordinate before multiplying by its count, using the existing reducer.
    ``write_strategy="sparse"`` updates selected counts and tracks occupied
    IDs; the alphabet, feature table and count array remain fully allocated.
    """

    def __init__(
        self, network: PhaseFeatureNetwork, alphabet, *, axis_sizes=None, address_channels=None, write_strategy="dense"
    ):
        self._network = copy.deepcopy(network)
        self._alphabet = np.array(alphabet, copy=True)
        self._cache = GridFeatureCache(
            self._network, self._alphabet, axis_sizes=axis_sizes, address_channels=address_channels
        )
        self._alphabet.setflags(write=False)
        if write_strategy not in ("dense", "sparse"):
            raise ValueError("write_strategy must be dense or sparse")
        self._write_strategy = write_strategy
        self._counts = np.zeros(len(self._alphabet), dtype=np.int64)
        self._occupied_ids = []

    @property
    def write_strategy(self):
        """Fixed update/read policy; changing it would invalidate occupied IDs."""
        return self._write_strategy

    @property
    def counts(self):
        """An owned snapshot of integer control state."""
        return self._counts.copy()

    @property
    def payload_array_bytes(self):
        return self._cache.table.nbytes + self._counts.nbytes

    def reset(self):
        self._counts.fill(0)
        self._occupied_ids.clear()

    def write(self, addresses):
        ids = self._cache.address_ids(self._alphabet, addresses)
        if self.write_strategy == "dense":
            increments = np.bincount(ids, minlength=len(self._counts))
            if np.any(increments > (1 << 31) - self._counts):
                raise OverflowError("multiplicity exceeds reducer's per-address bound")
            self._counts += increments
        else:
            unique, increments = np.unique(ids, return_counts=True)
            previous = self._counts[unique]
            if np.any(increments > (1 << 31) - previous):
                raise OverflowError("multiplicity exceeds reducer's per-address bound")
            new = unique[previous == 0]
            self._counts[unique] += increments
            self._occupied_ids.extend(int(v) for v in new)

    def read(self, *, F=40, frame=None):
        occupied = (
            np.flatnonzero(self._counts)
            if self.write_strategy == "dense"
            else np.asarray(self._occupied_ids, dtype=np.int64)
        )
        result = v02_planes.reduce_complex(
            self._cache.table[occupied],
            axis=0,
            F=F,
            multiplicities=self._counts[occupied, None],
        ).value
        if frame is not None:
            word = np.asarray(frame)
            if word.dtype != np.uint32 or word.ndim != 0:
                raise ValueError("frame must be one packed SPF32 word")
            result = spf32.mul(result, word)
        return result


__all__ = ["FeatureCache", "FinitePhaseFeatureMemory", "GridFeatureCache", "VectorFeatureCache"]
