"""Circulant / GFNet diagonal-in-Fourier storage surface (``CIRCULANT_SPECTRUM``).

A circulant operator ``C(h)`` — and the global-filter token mixers built on it
(GFNet's complex filter, FNet's Fourier mixing, Spectre's frequency gate) — is
**diagonal in the Fourier basis**: ``C(h)·x = ifft( ĥ ⊙ fft(x) )``. The whole
operator is one spectral kernel ``ĥ`` (one complex coefficient per frequency
bin), and the apply is an elementwise complex multiply — exactly SPF's strength,
since a log-polar multiply is an O(1) integer add (no float MACs).

:mod:`spf.tensor.pipelines` / :mod:`spf.tensor.rewrite` already carry the circulant
*apply plan* and the ``ifft(diag(fft(x), h))`` recognition rule. What was missing
was a first-class **storage surface**: a v3 block that stores only ``ĥ`` and
declares itself diagonal-in-Fourier. This module promotes the reserved
``BaseModelKind.CIRCULANT_SPECTRUM`` (0x0200) to a real base model and adds the
surface that builds both the v3 block and the apply :class:`ExecutionPlan`.

Three constructors cover the family:

* :meth:`CirculantSpectralSurface.from_kernel` — a time-domain first column ``h``
  (a true circulant / cyclic convolution); the spectrum is ``fft(h)``.
* :meth:`CirculantSpectralSurface.from_spectrum` /
  :meth:`gfnet_global_filter` — a frequency-domain filter stored directly
  (GFNet/FNet learn ``ĥ`` in the spectral domain; no kernel fft).
* :func:`fit_circulant_from_matrix` — approximate a dense matrix by its circulant
  part (average over circular diagonals) and report captured energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import decode_words, encode_complex, encode_words
from .v2 import ExecutionPlan, ExecutionStep
from .v3 import (
    BaseModelKind,
    BaseModelRef,
    Representation,
    RepresentationRef,
    ResidualCodecKind,
    ResidualRef,
    SPFBlockV3,
    register_base_model,
)

__all__ = [
    "CIRCULANT_SPECTRUM_BITS_PER_BIN",
    "CirculantSpectralSurface",
    "CirculantSpectrumModel",
    "fit_circulant_from_matrix",
    "gfnet_global_filter",
]

# One SPF-32 (L, Θ) word per frequency bin.
CIRCULANT_SPECTRUM_BITS_PER_BIN = 32


# --------------------------------------------------------------------------- #
# v3 base model — the stored diagonal spectrum.
# --------------------------------------------------------------------------- #


class CirculantSpectrumModel:
    """v3 base model (``CIRCULANT_SPECTRUM``) storing a diagonal Fourier spectrum.

    The block's "wavefield" is the spectral kernel itself: ``reconstruct_planes``
    returns the stored ``(L, Θ)`` of ``ĥ`` over the frequency axis (dense). Pair
    it with a ``NONE`` residual — the spectrum is the whole story — so a v3 block
    reconstructs ``ĥ`` exactly to the representation's quantisation floor. The
    *apply* (``fft → diagonal_scale(ĥ) → ifft``) is an :class:`ExecutionPlan`, not
    a reconstruction, built by :meth:`CirculantSpectralSurface.apply_plan`.
    """

    kind = int(BaseModelKind.CIRCULANT_SPECTRUM)

    def reconstruct_planes(
        self,
        params: Any,
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spectrum_L = np.asarray(params["spectrum_L"], dtype=np.int64)
        spectrum_T = np.asarray(params["spectrum_T"], dtype=np.int64)
        active = np.arange(spectrum_L.size, dtype=np.int64)
        return active, spectrum_L, spectrum_T

    def storage_bits(self, params: Any) -> int:
        n = int(np.asarray(params["spectrum_L"]).size)
        return n * int(params.get("word_bits", CIRCULANT_SPECTRUM_BITS_PER_BIN))


register_base_model(CirculantSpectrumModel())


# --------------------------------------------------------------------------- #
# The surface.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CirculantSpectralSurface:
    """A diagonal-in-Fourier operator stored as its spectral kernel ``ĥ``.

    ``spectrum`` is the complex per-bin kernel (length ``n``). Build it from a
    time-domain kernel (:meth:`from_kernel`) or directly in the spectral domain
    (:meth:`from_spectrum` / :func:`gfnet_global_filter`). :meth:`to_block`
    serialises it as a v3 ``CIRCULANT_SPECTRUM`` block; :meth:`apply_plan` emits
    the ``fft → diagonal_scale → ifft`` apply.
    """

    spectrum: np.ndarray

    @property
    def n(self) -> int:
        return int(np.asarray(self.spectrum).size)

    @classmethod
    def from_spectrum(cls, spectrum: np.ndarray) -> CirculantSpectralSurface:
        """Store a frequency-domain kernel directly (GFNet/FNet learned filter)."""
        return cls(np.asarray(spectrum, dtype=np.complex128).reshape(-1))

    @classmethod
    def from_kernel(cls, kernel: np.ndarray) -> CirculantSpectralSurface:
        """Build from a time-domain first column ``h`` — ``spectrum = fft(h)``."""
        h = np.asarray(kernel, dtype=np.complex128).reshape(-1)
        return cls(np.fft.fft(h))

    def to_block(
        self, representation: int = int(Representation.POLAR_SPF32)
    ) -> SPFBlockV3:
        """Serialise as a v3 block: ``CIRCULANT_SPECTRUM`` base + ``NONE`` residual."""
        rep = RepresentationRef(int(representation))
        fmt = rep.format
        plane_L, plane_T = encode_complex(self.spectrum, fmt)
        params = {
            "spectrum_L": np.asarray(plane_L, dtype=np.int64),
            "spectrum_T": np.asarray(plane_T, dtype=fmt.theta_dtype).astype(np.int64),
            "word_bits": int(fmt.total_bits),
        }
        return SPFBlockV3(
            dims=(self.n,),
            origin=(0,),
            representation=rep,
            base=BaseModelRef(int(BaseModelKind.CIRCULANT_SPECTRUM), params),
            residual=ResidualRef(int(ResidualCodecKind.NONE), None),
        )

    def apply_plan(
        self, x: np.ndarray
    ) -> tuple[ExecutionPlan, dict[str, np.ndarray], np.ndarray]:
        """Build the apply plan for input ``x``: ``fft → diagonal_scale(ĥ) → ifft``.

        Returns ``(plan, inputs, fp64)`` like the :mod:`spf.tensor.pipelines`
        builders: ``inputs`` are SPF words, ``fp64`` is the independent dense
        circulant oracle ``ifft(fft(x) · ĥ)`` for EVM scoring.
        """
        xx = np.asarray(x, dtype=np.complex128).reshape(-1)
        if xx.size != self.n:
            raise ValueError(f"input length {xx.size} != spectrum length {self.n}")
        x_words = encode_words(xx)
        h_words = encode_words(self.spectrum)
        plan = ExecutionPlan(
            steps=(
                ExecutionStep("fft", ("x",), ("X",), params={"n": self.n}),
                ExecutionStep("diagonal_scale", ("X", "h"), ("Y",)),
                ExecutionStep("ifft", ("Y",), ("y",), params={"n": self.n}),
            )
        )
        inputs = {"x": x_words, "h": h_words}
        fp64 = np.fft.ifft(np.fft.fft(decode_words(x_words)) * decode_words(h_words))
        return plan, inputs, fp64


def gfnet_global_filter(complex_filter: np.ndarray) -> CirculantSpectralSurface:
    """A GFNet/FNet/Spectre global filter: a frequency-domain kernel stored as-is.

    These mixers *learn* the spectral kernel directly (the filter already lives in
    the Fourier domain), so the surface stores it without a kernel fft — the
    distinction from :meth:`CirculantSpectralSurface.from_kernel`.
    """
    return CirculantSpectralSurface.from_spectrum(complex_filter)


def fit_circulant_from_matrix(
    matrix: np.ndarray, *, method: str = "vectorized"
) -> tuple[CirculantSpectralSurface, float]:
    """Approximate a dense ``n×n`` matrix by its circulant part + captured energy.

    Averages each circular diagonal ``M[i, (i−k) mod n]`` to recover the circulant
    first column ``c[k]``, then ``spectrum = fft(c)``. ``captured`` is the squared-
    energy fraction the circulant approximation explains (``1`` iff ``M`` is exactly
    circulant) — the spectral analogue of the Toeplitz-by-offset fit.

    ``method``:

    * ``"vectorized"`` (default) — index-vectorised numpy, no Python ``n²`` loop
      (~8× faster at ``n=1024``).
    * ``"loop"`` — the explicit per-diagonal reference: a 1-D mean per diagonal and
      an element-wise rebuild, in that exact summation order. The two agree to float
      epsilon (max ``|Δ| ≈ 2e-16``); pick ``"loop"`` only when you need the original
      reference summation bit-for-bit.
    """
    M = np.asarray(matrix, dtype=np.complex128)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("fit_circulant_from_matrix expects a square matrix")
    n = M.shape[0]
    if method == "vectorized":
        # offset[i, j] = (i − j) mod n is the circular-diagonal index of entry (i, j).
        # col[k] averages the k-th circular diagonal; approx gathers col back over the
        # same offsets to rebuild the circulant. Both vectorised — no Python n² loop.
        ri = np.arange(n)[:, None]
        ci = np.arange(n)[None, :]
        offset = (ri - ci) % n
        col = M[ri, offset].mean(axis=0)
        approx = col[offset]
    elif method == "loop":
        # The explicit reference: a 1-D mean per circular diagonal, then an
        # element-wise rebuild — the original summation order, kept as the oracle.
        col = np.empty(n, dtype=np.complex128)
        rows = np.arange(n)
        for k in range(n):
            col[k] = M[rows, (rows - k) % n].mean()
        approx = np.empty_like(M)
        for i in range(n):
            for j in range(n):
                approx[i, j] = col[(i - j) % n]
    else:
        raise ValueError(f"unknown method {method!r}; expected 'vectorized' or 'loop'")
    total = float(np.sum(np.abs(M) ** 2))
    residual = float(np.sum(np.abs(M - approx) ** 2))
    captured = 1.0 - residual / max(total, 1e-30)
    return CirculantSpectralSurface.from_kernel(col), captured
