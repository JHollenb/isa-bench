r"""Law 10 — exact low-bit SPF matmul in a cyclotomic ring.

Spec: ``obsidian/spf/2026-09-30-135512-spf-matmul-expanded-and-factored.md`` §6
(prototype ``scratchpad/matmul/s3_ring.py``).

For an SPF with ``d`` log steps per octave and ``M`` phases, every entry is a
signed monomial ``±2**e · y**r · x**m`` with ``y = 2**(1/d)`` and ``x = ζ_M``
(a primitive ``M``-th root of unity), ``0 <= r < d``, ``0 <= m < M/2`` and the
sign carried by ``x**(M/2) = -1``.  That is an element of the ring

    R_{d,M} = Z[1/2][y, x] / (y**d - 2, x**(M/2) + 1).

A product of two entries is a monomial again — the exponents add, with
``y**d = 2`` carried into ``e`` and ``x**(M/2) = -1`` folded into the sign — and
a sum is coefficient addition.  So a contraction ``C_ij = Σ_k A_ik B_kj`` becomes

    C_ij = 2**emin · Σ_{r<d} Σ_{m<M/2} c^{ij}_{r,m} · 2**(r/d) · ζ_M**m,
    c^{ij}_{r,m} ∈ Z,

where each MAC is two integer adds (the exponents) plus one shifted add of
``±2**(e-emin)`` into accumulator ``(r, m)``.  The accumulation is **exact**
(Python big integers; a global ``emin`` keeps every shift non-negative, so the
coefficients are ordinary integers).  Because integer addition is associative
and commutative, the coefficients — hence the result — are **order-independent**
(bit-identical under any permutation of the ``k`` axis).

The only irrational operation is the final evaluation, **once per output**,
against a shared table of ``d·M/2`` constants.  Encoding that complex value back
to an ``(L, T)`` code is one RNE rounding per output.

Break-even: ``K > d·M/2`` (the finalization cost per output).  For SPF16
(``d = 16``, ``M = 256``) that is ``d·M/2 = 2048``; for a conformal 8-bit SPF
(e.g. ``d = 4``, ``M = 16``) it is 32 — a few dozen.  For SPF32
(``d = 256``, ``M = 65536``) it is 8.4e6, so this is useless there — hence
**low-bit only**.  Nothing in the library selects this reducer by default; it is
a standalone helper usable *by* ``spfblock_einsum`` / ``contract`` for low-bit
formats, not wired in.

Redundancy remark (spec §6): ``{2**(r/d) ζ**m}`` spans but is not always a
basis — e.g. ``√2 = ζ_8 + ζ_8**-1``, so for even ``d`` and ``M >= 8`` the
coordinates are redundant.  The accumulation stays exact regardless; the
redundancy only means a smaller table is possible.

Status: **CHECKED** (``tests/tensor/test_ring_accumulator.py``): the ring result
matches a 60-dps mpmath oracle to that precision and a float64 matmul of the same
quantized entries to float64's own rounding (~1e-15); coefficients are
bit-identical under k-permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["RingAccumulator", "int64_headroom_bits"]

IntCodes = npt.NDArray[np.int64]


def int64_headroom_bits(l_span: int, k: int) -> int:
    """Bits an int64 accumulator would need for a contraction.

    ``l_span`` is the L-code span (``L_max - L_min``) across both operands and
    ``k`` the contraction length.  A coordinate magnitude is at most
    ``k · 2**(l_span // d_placeholder)`` … but the ``d`` division only shrinks it,
    so this returns the conservative ``ceil(log2 k) + l_span + 1`` (sign) bound.
    An int64 accumulator is safe iff the result is ``< 63``; otherwise use the
    exact Python-int path (the default here).
    """

    return (int(k).bit_length()) + int(l_span) + 1


@dataclass(frozen=True)
class RingAccumulator:
    """Exact contraction reducer for a low-bit SPF format ``(d, M)``.

    ``d`` = log steps per octave (``2**l_frac_bits``); ``M`` = number of phase
    codes (``2**theta_bits``), even.  Operands are integer code planes: ``L`` in
    units of ``2**(1/d)`` and ``T`` in ``[0, M)``.  All arithmetic through the
    accumulation is exact integer; only :meth:`evaluate` / :meth:`encode` round.
    """

    d: int
    M: int

    def __post_init__(self) -> None:
        if self.d < 1:
            raise ValueError(f"d must be >= 1, got {self.d}")
        if self.M < 2 or (self.M & 1):
            raise ValueError(f"M must be an even integer >= 2, got {self.M}")

    # -- geometry ------------------------------------------------------------
    @property
    def n_coords(self) -> int:
        """Number of ring coordinates per output (the shared table size)."""

        return self.d * (self.M // 2)

    @property
    def break_even_k(self) -> int:
        """Contraction length above which the reducer beats per-term evaluation."""

        return self.n_coords

    @classmethod
    def for_spf16(cls) -> RingAccumulator:
        """SPF16 (Q3.4 L, 8-bit phase): ``d = 16``, ``M = 256`` (break-even K > 2048)."""

        return cls(d=16, M=256)

    # -- exact accumulation --------------------------------------------------
    def contract_coords(
        self, la: IntCodes, ta: IntCodes, lb: IntCodes, tb: IntCodes, *, sparse: bool = False
    ) -> tuple[np.ndarray | dict[tuple[int, int, int, int], int], int]:
        r"""Exact integer ring coefficients for ``C = A @ B`` (einsum ``ik,kj->ij``).

        ``la, ta`` are the ``(N, K)`` L/T code planes of ``A``; ``lb, tb`` the
        ``(K, P)`` planes of ``B``.  Returns ``(coords, emin)`` where ``coords``
        has shape ``(N, P, d, M//2)`` of Python ints (numpy ``object`` dtype) and
        ``emin`` is the shared exponent floor.  ``coords`` is exact and
        order-independent in ``k``.
        With ``sparse=True``, only nonzero coordinates are retained in a
        dictionary keyed by ``(output_row, output_column, log_residue, phase)``.
        This does not provide a native finalizer for those coefficients.
        """

        la = np.asarray(la, dtype=np.int64)
        ta = np.asarray(ta, dtype=np.int64)
        lb = np.asarray(lb, dtype=np.int64)
        tb = np.asarray(tb, dtype=np.int64)
        if la.shape != ta.shape or lb.shape != tb.shape:
            raise ValueError("L and T planes must share a shape for each operand")
        if la.ndim != 2 or lb.ndim != 2:
            raise ValueError("operands must be 2-D (N,K) and (K,P)")
        n, k = la.shape
        k2, p = lb.shape
        if k != k2:
            raise ValueError(f"contraction mismatch: A is (N,{k}), B is ({k2},P)")
        d = self.d
        half = self.M // 2
        # Global exponent floor keeps every shift >= 0 (floor-div is monotone).
        emin = int((int(la.min()) + int(lb.min())) // d)

        coords = {} if sparse else np.zeros((n, p, d, half), dtype=object)
        ones = np.ones(k, dtype=object)
        for i in range(n):
            la_i = la[i]
            ta_i = ta[i]
            for j in range(p):
                total_l = la_i + lb[:, j]
                total_t = (ta_i + tb[:, j]) % self.M
                e, r = np.divmod(total_l, d)  # floor-div; r in [0, d)
                sign = np.where(total_t >= half, -1, 1)
                m = total_t % half
                shift = (e - emin).astype(object)
                contribs = sign.astype(object) * (ones << shift)
                if sparse:
                    for rr, mm, contribution in zip(r, m, contribs, strict=True):
                        key = (i, j, int(rr), int(mm))
                        value = coords.get(key, 0) + int(contribution)
                        if value:
                            coords[key] = value
                        else:
                            coords.pop(key, None)
                else:
                    np.add.at(coords[i, j], (r, m), contribs)
        return coords, emin

    # -- evaluation (one per output) -----------------------------------------
    def _basis_float(self) -> np.ndarray:
        r = np.arange(self.d, dtype=np.float64)[:, None]
        m = np.arange(self.M // 2, dtype=np.float64)[None, :]
        return (2.0 ** (r / self.d)) * np.exp(2j * np.pi * m / self.M)

    def evaluate(self, coords: np.ndarray, emin: int, *, dps: int | None = None) -> np.ndarray:
        """Evaluate ring coefficients to complex128 — one evaluation per output.

        ``dps=None`` uses a float64 basis (matches a float64 matmul to float64
        rounding).  ``dps=<int>`` uses an ``mpmath`` basis at that precision (the
        high-precision oracle) and returns complex128 of the rounded result.
        """

        coords = np.asarray(coords, dtype=object)
        *batch, d, half = coords.shape
        if d != self.d or half != self.M // 2:
            raise ValueError("coords shape does not match this (d, M)")
        out = np.empty(tuple(batch), dtype=np.complex128)
        if dps is None:
            basis = self._basis_float()
            scale = 2.0**emin
            for idx in np.ndindex(*batch):
                cell = coords[idx].astype(np.float64)
                out[idx] = scale * complex(np.sum(cell * basis))
            return out
        import mpmath as mp

        with mp.workdps(dps):
            basis_mp = [
                [mp.mpf(2) ** (mp.mpf(r) / self.d) * mp.expjpi(mp.mpf(2 * m) / self.M) for m in range(half)]
                for r in range(self.d)
            ]
            scale_mp = mp.mpf(2) ** emin
            for idx in np.ndindex(*batch):
                cell = coords[idx]
                acc = mp.mpc(0)
                for rr in range(self.d):
                    for mm in range(half):
                        c = cell[rr, mm]
                        if c:
                            acc += int(c) * basis_mp[rr][mm]
                acc *= scale_mp
                out[idx] = complex(acc)
        return out

    def contract(
        self,
        la: IntCodes,
        ta: IntCodes,
        lb: IntCodes,
        tb: IntCodes,
        *,
        dps: int | None = None,
    ) -> np.ndarray:
        """Exact ring contraction ``C = A @ B`` evaluated once per output.

        Convenience wrapper: :meth:`contract_coords` then :meth:`evaluate`.
        Returns a complex128 ``(N, P)`` array.
        """

        coords, emin = self.contract_coords(la, ta, lb, tb)
        return self.evaluate(coords, emin, dps=dps)

    # -- one RNE encode per output ------------------------------------------
    def encode(self, z: Any) -> tuple[np.ndarray, np.ndarray]:
        """RNE-encode complex value(s) back to ``(L, T)`` codes on the ``(d, M)`` grid.

        ``L = round(d · log2|z|)`` and ``T = round(M · arg(z) / 2π) mod M`` with
        ties-to-even.  One rounding per output; zeros map to ``(0, 0)``.
        """

        z = np.asarray(z, dtype=np.complex128)
        mag = np.abs(z)
        l_code = np.where(mag > 0, np.rint(self.d * np.log2(np.where(mag > 0, mag, 1.0))), 0)
        t_code = np.rint(self.M * (np.angle(z) / (2.0 * np.pi))).astype(np.int64) % self.M
        return l_code.astype(np.int64), t_code.astype(np.int64)

    # -- value helper (exact quantized entry value) --------------------------
    def value(self, l_code: IntCodes, t_code: IntCodes) -> np.ndarray:
        """The complex value ``2**(L/d) · exp(2πi T/M)`` of code plane(s) (float64)."""

        l_code = np.asarray(l_code, dtype=np.float64)
        t_code = np.asarray(t_code, dtype=np.float64)
        return (2.0 ** (l_code / self.d)) * np.exp(2j * np.pi * t_code / self.M)
