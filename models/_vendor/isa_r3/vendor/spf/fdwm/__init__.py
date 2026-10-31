"""Frequency-Domain Weight Matrix formats (FDWM-C, FDWM-BC).

These are O(n) and O(n²/k) storage formats for weight matrices that are
(approximately) circulant or block-circulant. The forward pass becomes
``y = ifft(fft(c) * fft(x))`` — a single pointwise multiply in the frequency
domain — replacing dense O(n²) matmul with O(n log n) transforms and
O(n) memory bandwidth.

Numpy/FP32 reference. The PyTorch integration layer has been removed
pending a Phase 2 revisit; this module is the only path today.
"""

from spf.fdwm.block_circulant import (
    FDWMBC_HEADER_SIZE,
    FDWMBC_MAGIC,
    fdwm_bc_decode,
    fdwm_bc_encode,
    fdwm_bc_forward,
)
from spf.fdwm.circulant import (
    FDWMC_HEADER_SIZE,
    FDWMC_MAGIC,
    fdwm_c_decode,
    fdwm_c_encode,
    fdwm_c_forward,
)
from spf.fdwm.projection import (
    project_to_block_circulant,
    project_to_circulant,
    reconstruct_block_circulant,
)

__all__ = [
    "FDWMBC_HEADER_SIZE",
    "FDWMBC_MAGIC",
    "FDWMC_HEADER_SIZE",
    "FDWMC_MAGIC",
    "fdwm_bc_decode",
    "fdwm_bc_encode",
    "fdwm_bc_forward",
    "fdwm_c_decode",
    "fdwm_c_encode",
    "fdwm_c_forward",
    "project_to_block_circulant",
    "project_to_circulant",
    "reconstruct_block_circulant",
]
