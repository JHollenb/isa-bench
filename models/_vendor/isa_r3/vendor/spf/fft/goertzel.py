"""Goertzel single-bin DFT — sparse-bin FFT alternative.

Implements the classic second-order Goertzel IIR for one DFT bin plus a
naive K-bin wrapper. Lives in :mod:`spf.fft` so callers can swap it in
anywhere the full FFT is overkill (PSK tone lock, Goertzel tone detect,
pitch tracking, OFDM pilot extraction).

Spec context — this is **not** part of the SPF-32 scalar spec. It's an
algorithmic restructuring that operates in rectangular (FP32 or FP64)
arithmetic and uses SPF-32 only at the I/O boundary (encode/decode). The
whole point is that each bin needs 1 FP32 multiply and 2 FP32 adds per
input sample instead of the O(log N) Sb()-gated butterflies of the
SPF-32 FFT; on RV32I the soft-float FP path costs ~200–500 cyc/op vs
the Sb LUT's ~17 k cyc/call, so one bin beats the full FFT by a large
margin.

See :doc:`docs/research/sb-test-7-goertzel.md` for the full design memo,
accuracy analysis, and break-even K derivation.

The bin-index convention matches :mod:`spf.fft.spf32_fft`: bin ``k``
corresponds to frequency ``k/N`` cycles per sample, i.e.

    X[k] = Σ_{n=0}^{N-1} x[n] · e^{-2πi k n / N}

which is the forward DFT sign used throughout the package.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from spf.scalar import spf32 as spf32_py

__all__ = [
    "goertzel_bin_fp32",
    "goertzel_bin_fp64",
    "goertzel_bins_from_spf32",
    "goertzel_dtype_default",
]

# FP32 is the intended target (matches what the RV32I C kernel runs). FP64
# is exposed for accuracy sweeps and for platforms where the FP32 mantissa
# isn't a dominant error source.
goertzel_dtype_default = np.float32


def _goertzel_coef(k: int, n: int, dtype: np.dtype) -> tuple[float, float, float]:
    """Return the three constants used by the real-arithmetic Goertzel IIR.

    Real-arithmetic second-order Goertzel (Oppenheim & Schafer, 3e §9.2):

        ω   = 2π k / N                    (bin center frequency, rad/sample)
        c   = 2 cos(ω)                    (state-update coefficient)
        e   = cos(ω) − i sin(ω)           (final output phase rotor)

    The state recursion is ``s[n] = x[n] + c·s[n-1] − s[n-2]`` over N real
    (or complex — we keep a pair of real recursions running on the real
    and imaginary parts of x) samples; the final complex output is
    ``X[k] = s[N-1] − e · s[N-2]``.
    """
    # Match the forward-DFT convention of spf.fft: X[k] = Σ x[n] e^{-2π i k n / N}.
    two_pi_k_over_n = 2.0 * np.pi * float(k) / float(n)
    c = np.asarray(2.0 * np.cos(two_pi_k_over_n), dtype=dtype).item()
    cr = np.asarray(np.cos(two_pi_k_over_n), dtype=dtype).item()
    ci = np.asarray(np.sin(two_pi_k_over_n), dtype=dtype).item()
    return c, cr, ci


def _goertzel_real_recursion(x: np.ndarray, c: float, dtype: np.dtype) -> tuple[float, float]:
    """Run the N-tap real recursion ``s[n] = x[n] + c·s[n-1] − s[n-2]``.

    Returns ``(s[N-1], s[N-2])`` — exactly what the final combining step needs.
    Implemented as a Python-level loop so the C port can match it sample by
    sample; vectorizing would break the IIR recurrence.
    """
    s1 = dtype.type(0.0)  # s[n-1]
    s2 = dtype.type(0.0)  # s[n-2]
    c_t = dtype.type(c)
    for xn in x:
        xn_t = dtype.type(xn)
        s = xn_t + c_t * s1 - s2
        s2 = s1
        s1 = s
    return float(s1), float(s2)


def goertzel_bin_fp32(x: npt.ArrayLike, k: int) -> complex:
    """Compute ``X[k]`` for a single bin using FP32 Goertzel internally.

    Parameters
    ----------
    x : array-like
        Length-N input. Can be real (Goertzel runs once) or complex (Goertzel
        runs independently on real and imaginary lanes and the two results
        are combined).
    k : int
        Bin index in ``[0, N)``.

    Returns
    -------
    complex
        The single DFT bin value. Returned as FP64 (Python complex) for
        comparability with ``numpy.fft.fft`` and the SPF-32 reference, even
        though the recursion was run in FP32.

    Notes
    -----
    Classical Goertzel is defined for real x; for complex x we exploit DFT
    linearity: ``DFT(a + i·b)[k] = DFT(a)[k] + i · DFT(b)[k]``.
    """
    return _goertzel_bin(x, k, np.dtype(np.float32))


def goertzel_bin_fp64(x: npt.ArrayLike, k: int) -> complex:
    """FP64 variant of :func:`goertzel_bin_fp32`. Same contract, wider mantissa."""
    return _goertzel_bin(x, k, np.dtype(np.float64))


def _goertzel_bin(x: npt.ArrayLike, k: int, dtype: np.dtype) -> complex:
    """Shared implementation. Split out so fp32 and fp64 share a single code
    path — the only delta is the dtype used for state variables and coefficients."""
    arr = np.ascontiguousarray(np.asarray(x))
    if arr.ndim != 1:
        raise ValueError(f"Goertzel input must be 1-D; got shape {arr.shape}")
    n = int(arr.shape[0])
    if n < 2:
        raise ValueError(f"Goertzel requires N >= 2; got N={n}")
    if not (0 <= int(k) < n):
        raise ValueError(f"bin index k must be in [0, {n}); got k={k}")

    c, cr, ci = _goertzel_coef(int(k), n, dtype)

    # Real and imaginary lanes run independently.
    if np.iscomplexobj(arr):
        re_in = np.ascontiguousarray(arr.real, dtype=dtype)
        im_in = np.ascontiguousarray(arr.imag, dtype=dtype)
    else:
        re_in = np.ascontiguousarray(arr, dtype=dtype)
        im_in = np.zeros_like(re_in)

    s1_r, s2_r = _goertzel_real_recursion(re_in, c, dtype)
    s1_i, s2_i = _goertzel_real_recursion(im_in, c, dtype)

    # Final combining step. Standard derivation of the forward DFT (sign
    # ``e^{-iω}``) Goertzel, dropping the pseudo x[N]=0 step:
    #
    #     X[k] = y[N] = s[N] − e^{-iω} · s[N-1]
    #          = (c·s[N-1] − s[N-2]) − (cr − i·ci)·s[N-1]
    #          = cr·s[N-1] − s[N-2] + i·ci·s[N-1]
    #
    # so with the state aliases ``s1 = s[N-1]`` and ``s2 = s[N-2]``:
    #   real = cr·s1 − s2
    #   imag =      ci·s1
    out_r_from_re = cr * s1_r - s2_r
    out_i_from_re = ci * s1_r
    out_r_from_im = cr * s1_i - s2_i
    out_i_from_im = ci * s1_i

    # DFT linearity: full complex output = DFT(re) + i · DFT(im).
    out_re = out_r_from_re - out_i_from_im
    out_im = out_i_from_re + out_r_from_im
    return complex(float(out_re), float(out_im))


def goertzel_bins_from_spf32(x_spf32: npt.ArrayLike, ks: Iterable[int]) -> np.ndarray:
    """Run Goertzel over a small bin set, taking SPF-32 input at the boundary.

    Decodes ``x_spf32`` once to FP32, then runs an independent Goertzel for
    each ``k ∈ ks``. This matches the sparse-bin workload: on RV32I the
    per-sample decode cost is amortized across all bins, so the marginal
    cost of each additional bin is one Goertzel pass.

    Parameters
    ----------
    x_spf32 : array-like of uint32
        Length-N SPF-32 words.
    ks : iterable of int
        Bin indices to compute. Duplicates are kept (useful when sweeping).

    Returns
    -------
    ndarray[complex128]
        Shape ``(len(ks),)`` array of DFT bin values.
    """
    x_spf = np.ascontiguousarray(np.asarray(x_spf32), dtype=np.uint32)
    # Decode once at the boundary — the whole point of the sparse-bin pitch.
    z = spf32_py.decode(x_spf).astype(np.complex64, copy=False)
    return np.asarray(
        [goertzel_bin_fp32(z, int(k)) for k in ks],
        dtype=np.complex128,
    )
