"""Length-32 Nussbaumer-style complex convolution over F_5 = 2^32 + 1.

Python reference paired with ``native/src/fnt/fermat_ntt.c``. The functions here
have the same semantics as the C primitive; the test suite cross-
validates the two implementations (and both against NumPy's real-valued
``numpy.convolve`` wrapped to cyclic via wrap-around).

The algorithm has three layers:

1. **Length-32 Fermat-NTT** over F_5. Twiddles are powers of 2 (primitive
   32nd root of unity is ω = 2² = 4); butterflies are bit shifts + modular
   add/sub. Pure integer, zero table lookups. See
   :func:`fermat_ntt_32` / :func:`fermat_ntt_32_inv`.
2. **Pointwise Gaussian-integer product** in F_5. A+iB times C+iD gives
   (AC − BD) + i·(AD + BC). The four mulmods and two addmods per element
   are the only general (non-shift) multiplications in the whole
   pipeline.
3. **Two inverse NTTs** recover the real and imaginary parts of the
   cyclic convolution. Final scaling by N⁻¹ = 2⁻⁵ ≡ 2⁵⁹ mod F_5 is
   another pure shift (2 has order 64 in F_5).

Dynamic range: F_5 − 1 = 2³² ≈ 4.3×10⁹. For length-32 complex convolution
the output magnitude is bounded by 2N · max(|a|)·max(|b|) ≈ 64 · B². At
B = 256 (8-bit inputs), that's ~4.2 M — ~500× headroom inside F_5. We
validate only against that 8-bit range; wider inputs are unchecked.

Reference: Nussbaumer, "Fast polynomial transform algorithms for digital
convolution", IBM J. Res. Dev. 20(3) (1976); Agarwal & Burrus, "Fast
convolution using Fermat number transforms", IEEE TASSP 22 (1974).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fnt._constants import TWIDDLE_ORDER, F

__all__ = [
    "FERMAT_NTT_N",
    "addmod_F5",
    "fermat_conv_32",
    "fermat_ntt_32",
    "fermat_ntt_32_inv",
    "mulmod_F5",
    "shiftmul_F5",
    "submod_F5",
]

U64 = npt.NDArray[np.uint64]

FERMAT_NTT_N = 32

# Primitive root ω = 4 = 2². The NTT stride at block size m is
# 2 · (N/m), in units of the exponent of 2 (order 64 in F_5).
_OMEGA_LOG2 = 2

# Precomputed 5-bit reversal for N=32 — matches the C kBitReverse32 table.
_BITREV32 = np.array(
    [
        0,  16, 8,  24, 4,  20, 12, 28,
        2,  18, 10, 26, 6,  22, 14, 30,
        1,  17, 9,  25, 5,  21, 13, 29,
        3,  19, 11, 27, 7,  23, 15, 31,
    ],
    dtype=np.intp,
)

# N⁻¹ mod F_5 for N = 32 is 2⁻⁵ mod F_5. Since 2 has order 64 in F_5
# (because 2³² ≡ −1 ⇒ 2⁶⁴ ≡ 1), 2⁻⁵ ≡ 2⁵⁹ — a pure shift.
_N_INV_SHIFT = 59


def addmod_F5(a: int, b: int) -> int:  # noqa: N802  (F_5 is a math constant)
    """Return ``(a + b) mod F_5``. Inputs must be in ``[0, F_5)``."""
    s = int(a) + int(b)
    if s >= F:
        s -= F
    return s


def submod_F5(a: int, b: int) -> int:  # noqa: N802
    """Return ``(a - b) mod F_5``. Inputs must be in ``[0, F_5)``."""
    if a >= b:
        return int(a) - int(b)
    return int(a) + F - int(b)


def mulmod_F5(a: int, b: int) -> int:  # noqa: N802
    """Return ``(a · b) mod F_5`` via the identity ``2³² ≡ −1 mod F_5``.

    Split the 64-bit product into high/low 32-bit halves, reduce to
    ``lo − hi mod F_5``. Exactly one submod reduction suffices because
    ``lo`` and ``hi`` are both in ``[0, 2³²)``.
    """
    prod = int(a) * int(b)
    lo = prod & 0xFFFFFFFF
    hi = prod >> 32
    return submod_F5(lo, hi)


def shiftmul_F5(a: int, k: int) -> int:  # noqa: N802
    """Return ``(a · 2^k) mod F_5`` via bit shift + reduction.

    ``k`` is reduced mod 64 (order of 2 in F_5). For ``32 ≤ k < 64``
    apply ``2^k = −2^(k-32) mod F_5``.
    """
    k &= 63
    if k == 0:
        return int(a) % F
    eff = k
    neg = False
    if k >= 32:
        eff = k - 32
        neg = True
    s = int(a) << eff
    lo = s & 0xFFFFFFFF
    hi = s >> 32
    r = submod_F5(lo, hi)
    if neg:
        return 0 if r == 0 else F - r
    return r


def _as_u64(x: np.ndarray) -> U64:
    if x.dtype != np.uint64:
        raise ValueError(f"Fermat NTT input must be ndarray[uint64]; got dtype {x.dtype}")
    if x.shape != (FERMAT_NTT_N,):
        raise ValueError(
            f"Fermat NTT input must be shape ({FERMAT_NTT_N},); got {x.shape}"
        )
    return x


def _dit_32(a: U64, sign: int) -> U64:
    """Length-32 DIT driver, shared by forward (+1) and inverse (-1)."""
    buf = a[_BITREV32].astype(np.uint64, copy=True)
    for s in range(5):
        m = 2 << s
        half = m >> 1
        stride = _OMEGA_LOG2 * (FERMAT_NTT_N // m)
        for block in range(0, FERMAT_NTT_N, m):
            for j in range(half):
                k = (sign * j * stride) & 63
                a_val = int(buf[block + j])
                wb = shiftmul_F5(int(buf[block + j + half]), k)
                buf[block + j] = np.uint64(addmod_F5(a_val, wb))
                buf[block + j + half] = np.uint64(submod_F5(a_val, wb))
    return buf


def fermat_ntt_32(a: np.ndarray) -> U64:
    """Forward length-32 Fermat-NTT over F_5."""
    u = _as_u64(a)
    return _dit_32(u, sign=+1)


def fermat_ntt_32_inv(a: np.ndarray) -> U64:
    """Inverse length-32 Fermat-NTT; scales by N⁻¹ = 2⁵⁹ mod F_5."""
    u = _as_u64(a)
    result = _dit_32(u, sign=-1)
    for i in range(FERMAT_NTT_N):
        result[i] = np.uint64(shiftmul_F5(int(result[i]), _N_INV_SHIFT))
    return result


def fermat_conv_32(
    a_re: np.ndarray,
    a_im: np.ndarray,
    b_re: np.ndarray,
    b_im: np.ndarray,
) -> tuple[U64, U64]:
    """Cyclic length-32 complex convolution of ``a = a_re + i·a_im`` and
    ``b = b_re + i·b_im`` via four forward NTTs, pointwise Gaussian-
    integer product in F_5, and two inverse NTTs.

    Returns ``(c_re, c_im)`` as length-32 ``ndarray[uint64]`` arrays, each
    with values in ``[0, F_5)``. Signed interpretation is up to the
    caller — see :func:`to_signed` below for the convention used in
    tests / benchmarks.
    """
    A_re = fermat_ntt_32(a_re)
    A_im = fermat_ntt_32(a_im)
    B_re = fermat_ntt_32(b_re)
    B_im = fermat_ntt_32(b_im)

    C_re = np.empty(FERMAT_NTT_N, dtype=np.uint64)
    C_im = np.empty(FERMAT_NTT_N, dtype=np.uint64)
    for i in range(FERMAT_NTT_N):
        arbr = mulmod_F5(int(A_re[i]), int(B_re[i]))
        aibi = mulmod_F5(int(A_im[i]), int(B_im[i]))
        arbi = mulmod_F5(int(A_re[i]), int(B_im[i]))
        aibr = mulmod_F5(int(A_im[i]), int(B_re[i]))
        C_re[i] = np.uint64(submod_F5(arbr, aibi))
        C_im[i] = np.uint64(addmod_F5(arbi, aibr))

    c_re = fermat_ntt_32_inv(C_re)
    c_im = fermat_ntt_32_inv(C_im)
    return c_re, c_im


def to_signed(x: np.ndarray) -> npt.NDArray[np.int64]:
    """Map ``[0, F_5)`` → ``(-F_5/2, F_5/2]`` by treating values in the
    upper half as their negatives.

    This is the convention for interpreting F_5-ring output as a signed
    integer when the input magnitudes keep the true result safely inside
    the signed-representable range (see the dynamic-range note in the
    module docstring).
    """
    half = F // 2
    out = x.astype(np.int64, copy=True)
    mask = out > half
    out[mask] -= F
    return out


def _pack_signed(x: np.ndarray) -> U64:
    """Pack a signed-integer array into ``[0, F_5)``-valued ``uint64``s
    by adding F_5 to negatives (inverse of :func:`to_signed`).

    Used by tests and benches to feed signed 8-bit values into the NTT
    without the caller having to think about ring encoding.
    """
    arr = np.asarray(x, dtype=np.int64)
    if arr.min() <= -F or arr.max() >= F:
        raise ValueError("values outside F_5 signed range; |x| must be < F_5")
    out = np.where(arr < 0, arr + F, arr).astype(np.uint64)
    return out


def fermat_conv_32_signed(
    a_re: np.ndarray,
    a_im: np.ndarray,
    b_re: np.ndarray,
    b_im: np.ndarray,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Signed wrapper: takes integer arrays in the signed range, returns
    signed arrays. Convenience for the naive-``np.convolve`` comparison
    in tests.
    """
    ar_u = _pack_signed(a_re)
    ai_u = _pack_signed(a_im)
    br_u = _pack_signed(b_re)
    bi_u = _pack_signed(b_im)
    c_re_u, c_im_u = fermat_conv_32(ar_u, ai_u, br_u, bi_u)
    return to_signed(c_re_u), to_signed(c_im_u)


# Silence unused-import warnings; TWIDDLE_ORDER is documented as the
# source of the `sign * j * stride` mod used above.
_ = TWIDDLE_ORDER
