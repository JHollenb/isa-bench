"""Deferred wave operators: axis phase metadata and one domain conversion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf.renderer.carriers import as_block_view, carrier_shape, complex_carrier


@dataclass(frozen=True)
class WaveCarrier:
    source: Any
    domain: str = "spatial"
    phase_y: Any = 0.0
    phase_x: Any = 0.0
    phase_constant: float = 0.0
    gain: Any = 1.0

    @classmethod
    def own(cls, source: Any, domain: str) -> WaveCarrier:
        if isinstance(source, cls):
            return source
        block = as_block_view(source)
        if block is not None:
            return cls(block, domain)
        values = np.asarray(source).copy()
        values.setflags(write=False)
        return cls(values, domain)

    @property
    def shape(self) -> tuple[int, ...]:
        return carrier_shape(self.source)

    @property
    def is_identity(self) -> bool:
        return (
            not self.phase_constant
            and not np.any(self.phase_y)
            and not np.any(self.phase_x)
            and np.ndim(self.gain) == 0
            and complex(np.asarray(self.gain).item()) == 1.0
        )

    def _block_words(self) -> np.ndarray | None:
        """Packed plane of a block source with the deferred header ops folded in.

        The axis phases and the scalar gain are word operations: a per-element
        unit phasor multiply and one gain word multiply through
        ``spf.tensor.core`` (rule 3).  Returns ``None`` for a non-block source
        or a non-scalar gain, which keep the Cartesian reference path.
        """
        from spf.tensor import core, v02

        block = as_block_view(self.source)
        if block is None or np.ndim(self.gain) != 0 or block.observable != "complex":
            return None
        indices = block.active_addresses()
        words = block.project_words(block.sample_words(indices), indices)
        if words.dtype != np.uint32:
            return None  # wide header offsets project to Cartesian values
        plane = np.full(block.shape, np.uint32(v02.SPF32_WORD_ZERO), dtype=np.uint32)
        plane.reshape(-1)[indices] = words
        if self.phase_constant or np.any(self.phase_y) or np.any(self.phase_x):
            py = np.broadcast_to(np.asarray(self.phase_y, dtype=np.float64), (self.shape[0],))
            px = np.broadcast_to(np.asarray(self.phase_x, dtype=np.float64), (self.shape[1],))
            radians = self.phase_constant + py[:, None] + px[None, :]
            codes = (np.rint(radians * (65536 / (2 * np.pi))).astype(np.int64) & 65535).astype(np.uint16)
            phasor = core.pack_words(np.zeros(codes.shape, dtype=np.int16), codes)
            plane = np.asarray(v02.spf_mul(plane, phasor), dtype=np.uint32)
        gain = complex(np.asarray(self.gain).item())
        if gain != 1.0:
            from spf.renderer.carriers import encode_words

            plane = np.asarray(v02.spf_mul(plane, np.uint32(encode_words(gain))), dtype=np.uint32)
        return plane

    def words(self, domain: str) -> np.ndarray:
        """Packed SPF32 plane in ``domain`` without a Cartesian intermediate.

        Only the carrier's own domain is available as words: a domain change
        is the wave's one transform boundary and runs through
        :meth:`materialize`.
        """
        if domain != self.domain:
            raise ValueError("a wave domain change is a transform boundary; request materialize(domain)")
        plane = self._block_words()
        if plane is not None:
            return plane
        source = np.asarray(self.source)
        if source.dtype.kind in "iu" and self.is_identity:
            return np.asarray(source, dtype=np.uint32)
        from spf.renderer.carriers import encode_words

        return np.asarray(encode_words(self.materialize(domain)), dtype=np.uint32)

    def phase_shift(self, y: Any, x: Any, constant: float = 0) -> WaveCarrier:
        py, px = np.asarray(self.phase_y + y), np.asarray(self.phase_x + x)
        py.setflags(write=False)
        px.setflags(write=False)
        return replace(self, phase_y=py, phase_x=px, phase_constant=self.phase_constant + constant)

    def multiply(self, gain: Any) -> WaveCarrier:
        values = np.asarray(self.gain * gain).copy()
        values.setflags(write=False)
        return replace(self, gain=values)

    def materialize(self, domain: str) -> np.ndarray:
        if domain == self.domain:
            plane = self._block_words()
            if plane is not None:
                # Block source: the header ops ran on words; this is the
                # single decode at the requested Cartesian boundary.
                from spf.renderer.carriers import decode_words

                return np.asarray(decode_words(plane))
        values = complex_carrier(self.source) * self.gain
        if self.phase_constant or np.any(self.phase_y) or np.any(self.phase_x):
            py = np.broadcast_to(self.phase_y, (self.shape[0],))
            px = np.broadcast_to(self.phase_x, (self.shape[1],))
            values = values * np.exp(1j * (self.phase_constant + py[:, None] + px[None, :]))
        if domain != self.domain:
            values = fft.fftn(values, workers=1) if domain == "frequency" else fft.ifftn(values, workers=1)
        return np.asarray(values)

    def to_domain(self, domain: str) -> WaveCarrier:
        return self if domain == self.domain else self.own(self.materialize(domain), domain)

    def intensity(self, domain: str) -> np.ndarray:
        if domain != self.domain:
            return np.asarray(np.abs(self.materialize(domain)) ** 2)
        block = as_block_view(self.source)
        if block is not None:
            indices = block.active_addresses()
            words = block.sample_words(indices, phase=False)
            intensity = block._replace(observable="intensity").project_words(words, indices)
            values = np.zeros(block.shape, dtype=np.float64)
            values.flat[indices] = intensity
        elif np.asarray(self.source).dtype.kind in "iu":
            from spf.renderer.continuity import project_packed_magnitude

            values = project_packed_magnitude(self.source, squared=True)
        else:
            values = np.abs(np.asarray(self.source)) ** 2
        return np.asarray(values * np.abs(self.gain) ** 2)
