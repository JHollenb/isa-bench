"""FDWM-BC — Block-Circulant Weight Matrix (spec §8).

Partitions ``W ∈ ℝ^(n×n)`` into ``(n/k)²`` blocks of size ``k×k``, each
circulant. Compression factor = ``k``. Each block is stored as its rfft
(Hermitian-reduced to ``k/2 + 1`` complex values) because its first column
is real.

Forward pass:

    y_i = Σ_j   irfft(rfft(c_ij) ⊙ rfft(x_j))

where ``x`` and ``y`` are split into ``n/k`` chunks of size ``k``.

Block format (spec §8.2):

    bytes 0..3    magic = 'FDMB'
    bytes 4..7    n (uint32, little-endian)
    bytes 8..11   k (uint32, little-endian, must divide n)
    bytes 12..15  dtype flag (2 = FP32 only in Phase 1)
    bytes 16..    (n/k)² blocks × (k/2 + 1) complex-64 values each
"""

from __future__ import annotations

from typing import Final

import numpy as np

FDWMBC_MAGIC: Final[bytes] = b"FDMB"
FDWMBC_HEADER_SIZE: Final[int] = 16

_DTYPE_FP32: Final[int] = 2


def _spectrum_shape(n: int, k: int) -> tuple[int, int, int]:
    """Return the ``(n_blocks, n_blocks, k//2 + 1)`` shape of a spectrum array."""
    if n % k != 0:
        raise ValueError(f"k={k} must divide n={n}")
    n_blocks = n // k
    return n_blocks, n_blocks, k // 2 + 1


def fdwm_bc_forward(x: np.ndarray, spectrum: np.ndarray, *, n: int, k: int) -> np.ndarray:
    """Block-circulant matrix-vector product using stored per-block rffts.

    ``spectrum`` has shape ``(n/k, n/k, k/2 + 1)`` where
    ``spectrum[i, j]`` is ``rfft(c_ij)`` of the ``(i, j)`` block's first column.
    """
    expected_shape = _spectrum_shape(n, k)
    if spectrum.shape != expected_shape:
        raise ValueError(f"spectrum shape {spectrum.shape} != expected {expected_shape}")
    if x.shape != (n,):
        raise ValueError(f"x shape {x.shape} != expected ({n},)")

    n_blocks = n // k
    x_parts = x.astype(np.float32).reshape(n_blocks, k)
    # rfft each x-chunk once: shape (n_blocks, k/2 + 1).
    x_ffts = np.fft.rfft(x_parts, axis=1)

    # For each output block i, sum across all j: irfft(spectrum[i, j] * x_ffts[j]).
    # Vectorize over j inside the sum: stack spectrum[i, :, :] and x_ffts.
    y = np.zeros((n_blocks, k), dtype=np.float32)
    for i in range(n_blocks):
        products = spectrum[i] * x_ffts  # (n_blocks, k/2 + 1)
        # Sum in frequency domain first (linearity of rfft), then one irfft.
        y[i] = np.fft.irfft(products.sum(axis=0), n=k)
    return y.reshape(n)


def fdwm_bc_encode(spectrum: np.ndarray, *, n: int, k: int) -> bytes:
    """Pack an FDWM-BC block: 16-byte header + flattened per-block rffts."""
    expected_shape = _spectrum_shape(n, k)
    if spectrum.shape != expected_shape:
        raise ValueError(f"spectrum shape {spectrum.shape} != expected {expected_shape}")
    header = bytearray(FDWMBC_HEADER_SIZE)
    header[0:4] = FDWMBC_MAGIC
    header[4:8] = n.to_bytes(4, "little")
    header[8:12] = k.to_bytes(4, "little")
    header[12:16] = _DTYPE_FP32.to_bytes(4, "little")
    payload = spectrum.astype(np.complex64).tobytes()
    return bytes(header) + payload


def fdwm_bc_decode(block: bytes) -> tuple[np.ndarray, int, int]:
    """Unpack an FDWM-BC block; return ``(spectrum, n, k)``."""
    if len(block) < FDWMBC_HEADER_SIZE:
        raise ValueError(f"block too short: {len(block)} < {FDWMBC_HEADER_SIZE}")
    if block[0:4] != FDWMBC_MAGIC:
        raise ValueError(f"bad magic: {block[0:4]!r}, expected {FDWMBC_MAGIC!r}")
    n = int.from_bytes(block[4:8], "little")
    k = int.from_bytes(block[8:12], "little")
    if k == 0 or n == 0 or n % k != 0:
        raise ValueError(f"invalid (n={n}, k={k})")
    dtype_flag = int.from_bytes(block[12:16], "little")
    if dtype_flag != _DTYPE_FP32:
        raise ValueError(f"only FP32 (flag=2) supported in Phase 1, got flag={dtype_flag}")

    n_blocks = n // k
    bins = k // 2 + 1
    n_values = n_blocks * n_blocks * bins
    expected = FDWMBC_HEADER_SIZE + n_values * 8
    if len(block) != expected:
        raise ValueError(f"block size {len(block)} != expected {expected} for (n={n}, k={k})")
    flat = np.frombuffer(block[FDWMBC_HEADER_SIZE:], dtype=np.complex64)
    return flat.reshape(n_blocks, n_blocks, bins).copy(), n, k
