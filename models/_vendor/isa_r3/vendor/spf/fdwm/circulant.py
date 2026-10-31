"""FDWM-C — Full Circulant Weight Matrix (spec §7).

A circulant matrix ``C ∈ ℝ^(n×n)`` is determined by its first column
``c ∈ ℝ^n``. The matrix-vector product reduces to ``Cx = IFFT(FFT(c) ⊙ FFT(x))``.
FDWM-C stores the weights as ``Ĉ = FFT(c)`` (n complex values) and performs
``y = real(IFFT(Ĉ ⊙ FFT(x)))`` as the forward pass.

Block format (spec §7.2):

    bytes 0..3    magic = 'FDMC'
    bytes 4..7    n (uint32, little-endian, power of 2)
    bytes 8..11   dtype flag (0=FP16, 1=SPF-32, 2=FP32 — only FP32 in Phase 1)
    bytes 12..15  reserved (zeros)
    bytes 16..    n complex values of Ĉ

For FP32: 8 bytes per complex value (4 real + 4 imag). n=4096 → 32,784 B.
(Compression vs dense FP32 4096×4096: 32 MB → 32 KB = 1000× — the headline
number in the roadmap becomes 2048× when compared to dense FP16.)
"""

from __future__ import annotations

from typing import Final

import numpy as np

FDWMC_MAGIC: Final[bytes] = b"FDMC"
FDWMC_HEADER_SIZE: Final[int] = 16

_DTYPE_FP32: Final[int] = 2


def fdwm_c_forward(x: np.ndarray, c_hat: np.ndarray) -> np.ndarray:
    """Compute ``y = Cx`` for circulant C with stored ``c_hat = FFT(c)``.

    ``x`` and ``c_hat`` must be the same length ``n``. Output is real float32
    (complex part is zero up to FFT roundoff for real c, real x).
    """
    if x.ndim != 1:
        raise ValueError(f"x must be 1-D, got shape {x.shape}")
    if c_hat.ndim != 1:
        raise ValueError(f"c_hat must be 1-D, got shape {c_hat.shape}")
    if x.shape[0] != c_hat.shape[0]:
        raise ValueError(f"length mismatch: x {x.shape[0]}, c_hat {c_hat.shape[0]}")
    spectrum = np.fft.fft(x.astype(np.float32))
    y = np.fft.ifft(c_hat * spectrum).real
    return y.astype(np.float32)


def fdwm_c_encode(c_hat: np.ndarray) -> bytes:
    """Pack an FDWM-C block: 16-byte header + n complex-64 values."""
    if c_hat.ndim != 1:
        raise ValueError(f"c_hat must be 1-D, got shape {c_hat.shape}")
    n = int(c_hat.shape[0])
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
    header = bytearray(FDWMC_HEADER_SIZE)
    header[0:4] = FDWMC_MAGIC
    header[4:8] = n.to_bytes(4, "little")
    header[8:12] = _DTYPE_FP32.to_bytes(4, "little")
    # bytes 12..16 left as zeros (reserved).
    payload = c_hat.astype(np.complex64).tobytes()
    return bytes(header) + payload


def fdwm_c_decode(block: bytes) -> np.ndarray:
    """Unpack an FDWM-C block to ``c_hat`` (complex-64, length n)."""
    if len(block) < FDWMC_HEADER_SIZE:
        raise ValueError(f"block too short: {len(block)} < {FDWMC_HEADER_SIZE}")
    if block[0:4] != FDWMC_MAGIC:
        raise ValueError(f"bad magic: {block[0:4]!r}, expected {FDWMC_MAGIC!r}")
    n = int.from_bytes(block[4:8], "little")
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be power of 2, got {n}")
    dtype_flag = int.from_bytes(block[8:12], "little")
    if dtype_flag != _DTYPE_FP32:
        raise ValueError(f"only FP32 (flag=2) supported in Phase 1, got flag={dtype_flag}")
    expected = FDWMC_HEADER_SIZE + n * 8  # 8 bytes per complex-64
    if len(block) != expected:
        raise ValueError(f"block size {len(block)} != expected {expected} for n={n}")
    payload = np.frombuffer(block[FDWMC_HEADER_SIZE:], dtype=np.complex64)
    return payload.copy()
