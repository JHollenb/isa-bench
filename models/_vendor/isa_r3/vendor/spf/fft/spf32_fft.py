"""Radix-2 DIT FFT driver for SPF-32.

Spec: docs/spf-formats-and-test-plans.md §12 T1.4. Roadmap: Task 1.3.

.. warning::

   **The default profile here is NOT the design's default transform.**
   Algorithm Reference v0.2 §11.2 makes ``FB3_HYBRID`` the preferred v0.2
   butterfly and §11.7 demotes FB7 to "no longer the only preferred
   implementation"; this module still defaults to ``LEGACY_FB7_POLY7``.  The
   default is deliberately *not* flipped, and the reason is measured, not
   assumed: ``spf.native.fft`` / ``spf32_fft_fb7`` / the HLS bit-trick kernels
   are FB7, and ``tests/fft/test_T1_4_fft_c.py::test_c_fft_matches_python_bitwise``
   asserts this function is **bit-identical** to the C driver.  Substituting
   FB3 changes 6 of 8 words at N=8 and 1017 of 1024 at N=1024, so the flip
   would break that contract plus ~45 other call sites
   (``spf.dsp.radar_mf``, ``spf.dsp.sofdm_codec``, ``spf.tensor.pipelines``,
   ``spf.tensor.pl_lowering``, ``spf.tensor.v2``, …).

   Callers that want the normative v0.2 transform must ask for it:
   ``spf32_fft.fft(x, profile="v02")``, or call
   :func:`spf.tensor.v02.spf_fft_radix2` /
   :func:`spf.tensor.v02_planes.spf_fft_radix2` directly.  The FB7 default is a
   *compatibility* default, pinned by
   ``tests/fft/test_section_11_7_four_way_harness.py``.

Per-stage error budget: ``ε_stage ≈ 2·(2⁻⁸ + 2⁻¹⁵) ≈ 0.008``. Supported
sizes: powers of two in [2, 1024]; the twiddle generator stays exact up to
``N = 2¹⁶`` — the upper bound is a test-surface choice, not a format limit.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from spf.fft.fused import pack_words, spf_butterfly_fb7_accurate, spf_fft_fused, unpack_words
from spf.scalar import spf32
from spf.scalar._constants import SPF32

__all__ = ["fft", "ifft"]

Word = npt.NDArray[np.uint32]

_MIN_N: int = 2
_MAX_N: int = 1024


def fft(x: Word, *, profile: str | None = None, precision: str = "balanced") -> Word:
    """Forward SPF-32 N-point DFT via radix-2 decimation-in-time.

    The default profile is ``LEGACY_FB7_POLY7``.  ``profile="v02"`` uses
    the normative v0.2 FB3-Hybrid butterfly and radix-2 schedule.
    ``profile="v02_factored"`` uses the opt-in factored tan-tanh butterfly
    (:mod:`spf.fft.factored`) — a CANDIDATE (<= 1 tick/word) alternative, not
    the conforming path; see its module docstring for the error budget.

    Routes through the C spf32_fft_fb7 array entry point when libspf32 is
    available; falls back to the pure-Python spf_fft_fused reference otherwise.
    Batch dimensions on the last axis are preserved in both paths.

    Parameters
    ----------
    x : ndarray[uint32]
        1-D or N-D input of SPF-32 words. The last axis (length N, power of 2
        in [2, 1024]) is transformed; leading batch dimensions are preserved.

    Returns
    -------
    ndarray[uint32]
        Same shape as input with the last axis replaced by the DFT bins.
    """
    if profile is not None:
        profile_name = profile.lower().replace("-", "_")
        if profile_name in {"v02", "spf_v02", "fb3_hybrid"}:
            from spf.tensor.v02 import spf_fft_radix2

            return spf_fft_radix2(x, precision=precision)
        if profile_name in {"v02_factored", "factored", "spf_v02_factored"}:
            from spf.fft.factored import spf_fft_radix2_factored

            return spf_fft_radix2_factored(x, precision=precision)
        if profile_name not in {"legacy", "legacy_fb7_poly7", "fb7", "default"}:
            raise ValueError(f"unsupported SPF-32 FFT profile {profile!r}")

    from spf.native._lib import LIBSPF_AVAILABLE

    x_arr = np.asarray(x, dtype=np.uint32)
    batch_shape = x_arr.shape[:-1]
    n = x_arr.shape[-1]
    if n < _MIN_N or n > _MAX_N or (n & (n - 1)) != 0:
        raise ValueError(f"fft requires power-of-two N in [{_MIN_N}, {_MAX_N}], got N={n}")
    rows = x_arr.reshape(-1, n)
    out_rows = np.empty_like(rows)

    if LIBSPF_AVAILABLE:
        from spf.native import fb7 as _c_fb7

        for i in range(rows.shape[0]):
            out_rows[i] = _c_fb7.fft(rows[i], mode="accurate")
    else:
        for i in range(rows.shape[0]):
            row = rows[i]
            l_raw = (row >> 16) & np.uint32(0xFFFF)
            is_sentinel = l_raw == np.uint32(0x7FFF)
            if np.any(is_sentinel):
                # Any NaN/inf input poisons all N outputs (butterfly is linear;
                # every output sums contributions from every input).
                theta_raw = row & np.uint32(0xFFFF)
                has_nan = bool(np.any(is_sentinel & (theta_raw == np.uint32(0xFFFF))))
                if has_nan:
                    out_rows[i] = np.full(n, np.uint32(0x7FFFFFFF), dtype=np.uint32)
                else:
                    first_theta = int(theta_raw[np.argmax(is_sentinel)])
                    out_rows[i] = np.full(n, np.uint32((0x7FFF << 16) | first_theta), dtype=np.uint32)
            else:
                xs = unpack_words(int(w) for w in row)
                result = spf_fft_fused(xs, bfly_fn=spf_butterfly_fb7_accurate)
                out_rows[i] = np.array(pack_words(result), dtype=np.uint32)

    return out_rows.reshape(*batch_shape, n) if batch_shape else out_rows.reshape(n)


def ifft(x: Word, *, profile: str | None = None, precision: str = "balanced") -> Word:
    """Inverse SPF-32 N-point DFT via ``(1/N) · conj(fft(conj(X)))``.

    The default profile is ``LEGACY_FB7_POLY7``.  ``profile="v02"`` uses
    the v0.2 FB3-Hybrid implementation; ``profile="v02_factored"`` uses the
    opt-in factored tan-tanh butterfly (:mod:`spf.fft.factored`).
    Scaling by ``1/N`` is applied via :func:`spf.scalar.spf32.mul` against a
    unit-phase scale word ``L = -log2(N)·L_SCALE``, which preserves the
    sentinel cascade — zero/inf/nan words round-trip unchanged.
    """
    if profile is not None:
        profile_name = profile.lower().replace("-", "_")
        if profile_name in {"v02", "spf_v02", "fb3_hybrid"}:
            from spf.tensor.v02 import spf_ifft_radix2

            return spf_ifft_radix2(x, precision=precision)
        if profile_name in {"v02_factored", "factored", "spf_v02_factored"}:
            from spf.fft.factored import spf_ifft_radix2_factored

            return spf_ifft_radix2_factored(x, precision=precision)
        if profile_name not in {"legacy", "legacy_fb7_poly7", "fb7", "default"}:
            raise ValueError(f"unsupported SPF-32 IFFT profile {profile!r}")

    x_arr = np.asarray(x, dtype=np.uint32)
    n = x_arr.shape[-1]
    if n < _MIN_N or n > _MAX_N or (n & (n - 1)) != 0:
        raise ValueError(f"ifft requires power-of-two N in [{_MIN_N}, {_MAX_N}], got N={n}")
    x_conj = spf32.conj(x_arr)
    fft_out = fft(x_conj)
    out_conj = spf32.conj(fft_out)
    log2n = round(math.log2(n))
    scale_l = np.int16(-log2n * SPF32.l_scale)
    scale_word = np.uint32(np.uint32(np.uint16(scale_l)) << np.uint32(16))
    return spf32.mul(out_conj, scale_word)
