"""Format-parameterized radix-2 DIT FFT driver.

Spec: docs/spf-formats-and-test-plans.md §12 T1.4.

Stacks the scalar format's ``butterfly`` op across ``log2(N)`` stages with
bit-reversed input. Twiddle factors come from the scalar module's ``twiddle``
(exact for all power-of-two N dividing ``theta_scale``). Per-stage error budget
is the butterfly's ``ε_stage = 2·(l_ulp + theta_ulp·2π)`` — 0.008 for SPF-32,
0.266 for SPF-16, 3e-5 for SPF-64.

Each per-format wrapper module (``spf16_fft``, ``spf32_fft``, ``spf64_fft``)
binds the scalar module, format descriptor, and supported size range, then
exposes ``fft`` / ``ifft`` entry points that call :func:`forward_fft` and
:func:`inverse_fft` here. Twiddle tables and bit-reversal permutations are
cached per ``(format_name, N)`` key.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from spf.scalar._constants import Format

__all__ = ["forward_fft", "inverse_fft"]

# Per-(format, N) caches — global so repeated calls amortize the O(N) setup.
_TWIDDLE_CACHE: dict[tuple[str, int], np.ndarray] = {}
_BIT_REV_CACHE: dict[int, npt.NDArray[np.intp]] = {}


def _validate_length(
    x: np.ndarray, fmt: Format, min_n: int, max_n: int
) -> int:
    """Return ``N = x.shape[-1]`` after checking shape, dtype, and power-of-two bound.

    Accepts 1-D or N-D input; batching is on the trailing axis, so shape
    ``(..., N)`` runs ``prod(...)`` independent length-N FFTs.
    """
    if x.ndim < 1:
        raise ValueError(f"FFT input must have at least 1 dimension; got shape {x.shape}")
    if x.dtype != fmt.word_dtype:
        raise ValueError(
            f"FFT input must be ndarray[{fmt.word_dtype.__name__}]; got dtype {x.dtype}"
        )
    n = int(x.shape[-1])
    if n < min_n or n > max_n or (n & (n - 1)) != 0:
        raise ValueError(
            f"{fmt.name} FFT length N must be a power of 2 in [{min_n}, {max_n}]; got N={n}"
        )
    return n


def _twiddle_table(n: int, scalar: ModuleType, fmt: Format) -> np.ndarray:
    """Return the length-N/2 forward-DFT twiddle vector ``e^{−2πi k/N}`` as SPF words.

    ``scalar.twiddle(k, N)`` encodes ``e^{+2πi k/N}`` (spec §2.4 sign convention).
    Forward DFT wants the conjugate; applied once at cache time.
    """
    key = (fmt.name, n)
    cached = _TWIDDLE_CACHE.get(key)
    if cached is not None:
        return cached
    k = np.arange(n // 2, dtype=np.int64)
    table: np.ndarray = scalar.conj(scalar.twiddle(k, n))
    _TWIDDLE_CACHE[key] = table
    return table


def _bit_reversal(n: int) -> npt.NDArray[np.intp]:
    """Return the length-N index array that bit-reverses the low ``log2(N)`` bits."""
    cached = _BIT_REV_CACHE.get(n)
    if cached is not None:
        return cached
    bits = int(n).bit_length() - 1
    idx = np.arange(n, dtype=np.intp)
    rev = np.zeros(n, dtype=np.intp)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    _BIT_REV_CACHE[n] = rev
    return rev


def forward_fft(
    x: np.ndarray, *, scalar: ModuleType, fmt: Format, min_n: int, max_n: int
) -> np.ndarray:
    """Run the DIT forward FFT for the given scalar format.

    Input shape ``(..., N)``: the trailing axis is transformed and the
    leading batch dimensions are preserved. Pure 1-D input ``(N,)`` is
    the unbatched case.
    """
    n = _validate_length(x, fmt, min_n, max_n)
    buf = np.ascontiguousarray(x[..., _bit_reversal(n)], dtype=fmt.word_dtype)
    batch_shape = buf.shape[:-1]

    table = _twiddle_table(n, scalar, fmt)
    log2n = n.bit_length() - 1
    for s in range(log2n):
        m = 2 << s
        half = m >> 1
        num_blocks = n // m
        stride = n // m
        twid = table[np.arange(half) * stride]

        view = buf.reshape((*batch_shape, num_blocks, m))
        top = view[..., :half]
        bot = view[..., half:]
        new_top, new_bot = scalar.butterfly(top, bot, twid)
        view[..., :half] = new_top
        view[..., half:] = new_bot

    return buf


def inverse_fft(
    x: np.ndarray, *, scalar: ModuleType, fmt: Format, min_n: int, max_n: int
) -> np.ndarray:
    """Run the inverse FFT via ``ifft(X) = (1/N) · conj(fft(conj(X)))``.

    Multiplication by ``1/N`` for N = 2^k is exact in SPF: it's a unit-phase
    word with ``L = −k · l_scale``.
    """
    n = _validate_length(x, fmt, min_n, max_n)
    conj_x = scalar.conj(x)
    spectrum = forward_fft(conj_x, scalar=scalar, fmt=fmt, min_n=min_n, max_n=max_n)
    result = scalar.conj(spectrum)

    log2n = n.bit_length() - 1
    l_bits_mask = (1 << fmt.l_bits) - 1
    l_code = (-log2n * fmt.l_scale) & l_bits_mask
    scale_word_int = int(l_code) << fmt.theta_bits
    scale_word: Any = fmt.word_dtype(scale_word_int)
    out: np.ndarray = scalar.mul(result, scale_word)
    return out
