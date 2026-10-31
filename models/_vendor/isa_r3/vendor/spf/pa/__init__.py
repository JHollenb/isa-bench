"""SPF-PA: DDS-style 32-bit phase accumulator."""

from spf.pa.phase_accumulator import (
    PhaseAccumulator,
    frequency_word,
    generate_tone_reference,
    positional_encoding,
)

__all__ = [
    "PhaseAccumulator",
    "frequency_word",
    "generate_tone_reference",
    "positional_encoding",
]
