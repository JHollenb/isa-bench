"""FP16 block ↔ SBF-128 conversion (spec §11, §5).

Two standard paths:

- ``fp16_to_sbf128(x)`` — same as :func:`spf.block.encode`: rfft then
  per-band E8M0 quantization, 97-byte output.
- ``sbf128_to_fp16(b)`` — same as :func:`spf.block.decode`.

Plus one non-standard path needed for test T6.1:

- ``sbf128_from_spectrum(spectrum)`` — accepts a precomputed length-65
  complex spectrum (e.g. the output of an SPF-32 FFT decoded back to
  complex) and runs the per-band quantizer directly, skipping rfft. This
  reproduces the T6.1 spec pseudocode's ``encode_sbf128_from_spectrum``
  at docs/spf-formats-and-test-plans.md:1614.

Spec cost: ~200 cycles/block for the standard paths (rfft dominates).
``sbf128_from_spectrum`` is just the quantization half.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf import block
from spf.block.sbf128 import _quantize_spectrum

__all__ = ["fp16_to_sbf128", "sbf128_from_spectrum", "sbf128_to_fp16"]


def fp16_to_sbf128(x: np.ndarray) -> bytes:
    """Encode a length-128 FP16 real block to a 97-byte SBF-128 block."""
    return block.encode(x)


def sbf128_to_fp16(b: bytes) -> np.ndarray:
    """Decode a 97-byte SBF-128 block to a length-128 FP16 real block."""
    return block.decode(b)


def sbf128_from_spectrum(spectrum: npt.ArrayLike) -> bytes:
    """Quantize a precomputed length-65 complex spectrum into an SBF-128 block.

    Unlike :func:`fp16_to_sbf128`, this skips the real-FFT and feeds the
    band quantizer directly. Used by T6.1 to measure whether an externally
    computed FFT (e.g., via SPF-32) is compatible with SBF-128's band
    allocation.
    """
    s = np.asarray(spectrum, dtype=np.complex64)
    return _quantize_spectrum(s)
