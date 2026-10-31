"""FP16 ↔ SPF-16 conversion (spec §11, narrower variant of §11.2).

Thin named wrapper over :mod:`spf.scalar.spf16`. FP16 has no native complex
dtype in numpy, so the caller passes either a complex64 array or a pair of
real float16 arrays; the decoder returns complex64 with real/imag already at
FP16 precision (the SPF-16 L/Θ quantization is the loss-dominating step).

Spec cost: ~6 cycles (same algorithm as §11.2, 8-bit Θ, Q3.4 L).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.scalar import spf16

__all__ = ["fp16_to_spf16", "spf16_to_fp16"]


def fp16_to_spf16(z: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Encode FP16/complex values into SPF-16 words."""
    return spf16.encode(z)


def spf16_to_fp16(w: npt.ArrayLike) -> npt.NDArray[np.complex64]:
    """Decode SPF-16 words to complex64 (real/imag already at FP16 precision)."""
    return spf16.decode(w).astype(np.complex64, copy=False)
