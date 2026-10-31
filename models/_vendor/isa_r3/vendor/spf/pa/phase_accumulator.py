"""SPF-PA: 32-bit DDS-style phase accumulator (spec §9).

State update (spec §9.2):

    Φ_{t+1} = (Φ_t + F) mod 2^32
    angle(t) = Φ_t / 2^32 × 2π

``F`` is a frequency control word and ``Φ`` is the 32-bit phase. Advancing
by one step adds ``F`` to ``Φ`` modulo ``2^32`` — one integer add with
natural wraparound. The sinusoidal output is ``sin``/``cos`` of the angle.

Spec divergence — §12 T5.1 computes ``PA_words[i] = round(freq × 2^32)``,
which produces an angle ``2π`` larger than the reference ``pos × freq``
(because ``angle = Φ/2^32 × 2π``, so one step already carries an implicit
``2π`` factor). ``frequency_word`` and ``positional_encoding`` below use
the correct mapping ``round(freq × 2^32 / (2π))``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MASK32 = 0xFFFFFFFF
_SCALE = 1 << 32
_TWO_PI = 2.0 * math.pi


@dataclass
class PhaseAccumulator:
    """DDS phase accumulator with 32-bit frequency word and phase register."""

    freq_word: int
    phase: int = 0

    def __post_init__(self) -> None:
        self.freq_word = int(self.freq_word) & _MASK32
        self.phase = int(self.phase) & _MASK32

    def step(self, n: int = 1) -> None:
        self.phase = (self.phase + int(n) * self.freq_word) & _MASK32

    def angle(self) -> float:
        return self.phase / _SCALE * _TWO_PI

    def sincos(self) -> tuple[float, float]:
        a = self.angle()
        return math.sin(a), math.cos(a)


def frequency_word(radians_per_step: float) -> int:
    """Convert a radians-per-step advance to the nearest 32-bit control word."""
    return round(radians_per_step * _SCALE / _TWO_PI) & _MASK32


def generate_tone_reference(
    freq_word: int, phase0: int = 0, n_samples: int = 0
) -> dict[str, np.ndarray]:
    """Reference tone generator for the ``pa_tone`` workload contract.

    Emits ``n_samples`` complex samples as ``(n_samples, 2) float32`` where
    column 0 is the real part (cos θ) and column 1 is the imaginary part
    (sin θ), matching the in-memory layout of the C entrypoint
    ``spf32_pa_tone_sc``. Angles use exact double-precision
    ``math.sin``/``math.cos``; backends are expected to approximate these
    within their tolerance profile.
    """
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    fw = int(freq_word) & _MASK32
    phi = int(phase0) & _MASK32
    positions = np.arange(n_samples, dtype=np.uint64)
    phases = (positions * np.uint64(fw) + np.uint64(phi)) & np.uint64(_MASK32)
    angles = phases.astype(np.float64) / _SCALE * _TWO_PI
    out = np.empty((n_samples, 2), dtype=np.float32)
    out[:, 0] = np.cos(angles).astype(np.float32)
    out[:, 1] = np.sin(angles).astype(np.float32)
    return {"samples": out}


def positional_encoding(seq_len: int, d_model: int, base: float = 10000.0) -> np.ndarray:
    """Sinusoidal positional encoding via a PA per frequency pair.

    Returns a ``(seq_len, d_model) float32`` array matching the transformer
    convention ``PE[pos, 2i] = sin(pos / base^(2i/d))`` and
    ``PE[pos, 2i+1] = cos(pos / base^(2i/d))``.
    """
    if seq_len < 0:
        raise ValueError("seq_len must be non-negative")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError("d_model must be a positive even integer")

    i = np.arange(0, d_model, 2, dtype=np.float64)
    freqs = 1.0 / (base ** (i / d_model))
    words = (np.round(freqs * _SCALE / _TWO_PI).astype(np.uint64)) & np.uint64(_MASK32)
    positions = np.arange(seq_len, dtype=np.uint64)[:, None]
    phases_int = (positions * words[None, :]) & np.uint64(_MASK32)
    angles = phases_int.astype(np.float64) / _SCALE * _TWO_PI
    out = np.empty((seq_len, d_model), dtype=np.float32)
    out[:, 0::2] = np.sin(angles)
    out[:, 1::2] = np.cos(angles)
    return out
