"""PolarSPF OFDM closed-loop codec: bits -> map -> channel -> demap -> bits.

This is the **P2** closed loop for the iso-accuracy benchmark: a full OFDM modem
expressed in SPF-native constructs and scored by **BER**
(:func:`spf.dsp.metrics.ber`), not just EVM.

Datapath (the multiplier-free SPF hot path):

* **TX map** via the PolarSPF constellation codec —
  - **QPSK/PSK**: a phase-ramp codebook (``psk_modulate_spf`` / ``twiddle``);
    symbols sit exactly on the unit circle, ``L = 0``, no Sb() / magnitude cost.
  - **QAM**: ring(L) x sector(Theta) via the frozen Gray-coded SPF LUTs in
    :mod:`spf.dsp.modulation` (``qam16/64/256_modulate_spf``).
* **Subcarrier placement** is a ``gather`` / ``permute`` (a free monomial); here
  every active subcarrier carries data, so the placement is the identity grid.
* **TX IFFT** via the SPF-32 IFFT (the ``conj -> fft -> conj -> *1/N`` macro the
  PL lowering reproduces).
* **RX FFT** via the SPF-32 FFT.
* **Equalize** is a scalar ``diagonal_scale`` by Hhat^-1 (an SPF multiply);
  P2 uses perfect CSI (Hhat = H). Pilot estimation is P3.
* **CPE** is a constant phase rotation applied as ``cpe_theta`` inside the fused
  demap kernel (here a fixed 45 deg / ``0x2000`` slicer-boundary offset for QPSK
  so the decision boundaries fall between the axis-aligned QPSK symbols).

Demap heads (decision 2 in the plan):

1. **QPSK phase-slicer** — native ``ofdm_qpsk_eq_demod_fused`` (quadrant slice).
2. **QAM16 log-projection** — :class:`spf.dsp.qam_logproj.QAM16LogProjOperator`.
3. **General min-over-codebook head** — :func:`codebook_demap`, for an arbitrary
   constellation.

**Honest scoping:** the general min-over-codebook head is an **add-domain
boundary op** — it decodes the equalized symbols to Cartesian and does a nearest-
neighbour search over the codebook. It runs **once at RX egress**, off the
multiplier-free SPF hot path, and is **NOT** counted as an SPF-native win. The
QPSK slicer and QAM16 log-projection heads, by contrast, operate directly on the
SPF (L, Theta) fields and are on the native datapath.

The AWGN + multipath channel is likewise an FP64 propagation model at the wire
boundary, not part of the SPF datapath.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from spf.dsp.channels import awgn_noise, noise_variance_for_snr, rayleigh_multipath
from spf.dsp.metrics import ber
from spf.dsp.modulation import (
    psk_demodulate_fp,
    psk_modulate_spf,
    qam16_modulate_fp,
    qam_demodulate,
    qam_modulate,
)
from spf.dsp.qam_logproj import QAM16LogProjOperator
from spf.fft import spf32_fft
from spf.native import spf32 as c_spf32
from spf.tensor.core import decode_words, encode_words, spf_mul_words

__all__ = [
    "QPSK_SLICER_CPE_THETA",
    "OFDMLoopResult",
    "codebook_demap",
    "psk_codebook",
    "qam_codebook",
    "run_ofdm_loop",
    "snr_for_target_ber",
]

# QPSK symbols from ``psk_modulate_spf`` sit on the axes (Theta in
# {0, 0x4000, 0x8000, 0xC000}). The native quadrant slicer decides on the top two
# Theta bits, so its boundaries fall *on* the symbols. Rotating the slicer by a
# half-quadrant (45 deg = 0x2000) via ``cpe_theta`` moves the boundaries into the
# gaps between symbols, making the hard decision robust to small phase jitter.
QPSK_SLICER_CPE_THETA = 0x2000

Complex = npt.NDArray[np.complex128]
Word = npt.NDArray[np.uint32]
Bits = npt.NDArray[np.uint8]


# -- codebooks ------------------------------------------------------------------


def psk_codebook(order: int) -> tuple[Complex, Bits]:
    """Return ``(points, bits)`` for an order-PSK constellation.

    ``points[s]`` is the complex symbol for index ``s``; ``bits[s]`` is the
    ``log2(order)`` MSB-first bit pattern that maps to it under
    :func:`spf.dsp.modulation.psk_modulate_spf`.
    """
    k = int(np.log2(order))
    idx = np.arange(order, dtype=np.uint8)
    bits = np.stack([(idx >> (k - 1 - i)) & 1 for i in range(k)], axis=1).astype(np.uint8)
    points = decode_words(psk_modulate_spf(bits.ravel(), order))
    return points, bits


def qam_codebook(order: int) -> tuple[Complex, Bits]:
    """Return ``(points, bits)`` for a Gray-coded QAM constellation (16/64/256)."""
    k = int(np.log2(order))
    idx = np.arange(order, dtype=np.uint16)
    bits = np.stack([(idx >> (k - 1 - i)) & 1 for i in range(k)], axis=1).astype(np.uint8)
    if order == 16:
        points = qam16_modulate_fp(bits.ravel())
    else:
        points = np.asarray(qam_modulate(bits.ravel(), order, spf=False), dtype=np.complex128)
    return points, bits


# -- general min-over-codebook demap head (RX egress, add-domain) ---------------


def codebook_demap(
    equalized_words: npt.ArrayLike,
    codebook_points: Complex,
    codebook_bits: Bits,
) -> Bits:
    """General nearest-codebook hard demap for an arbitrary constellation.

    **Add-domain boundary op (NOT an SPF-native win).** Decodes the equalized
    SPF symbols to Cartesian and picks the codebook point minimising the squared
    Euclidean distance (the ML hard decision for AWGN). Intended to run once at
    RX egress for constellations without a dedicated SPF slicer; it is deliberately
    off the multiplier-free hot path.

    Parameters
    ----------
    equalized_words : array of uint32
        Equalized SPF-32 symbols (after ``diagonal_scale`` by Hhat^-1).
    codebook_points : ndarray[complex128]
        The ``M`` constellation points.
    codebook_bits : ndarray[uint8], shape ``(M, k)``
        MSB-first bit pattern for each codebook point.
    """
    eq = decode_words(np.asarray(equalized_words, dtype=np.uint32)).ravel()
    cb = np.asarray(codebook_points, dtype=np.complex128).ravel()
    dist2 = np.abs(eq[:, None] - cb[None, :]) ** 2
    nearest = np.argmin(dist2, axis=1)
    out: Bits = np.asarray(codebook_bits, dtype=np.uint8)[nearest].ravel()
    return out


# -- TX map ---------------------------------------------------------------------


def _tx_map(bits: Bits, order: int, modulation: str) -> Word:
    """bits -> SPF-32 symbol words via the PolarSPF constellation codec."""
    if modulation == "psk":
        return psk_modulate_spf(bits, order)
    if modulation == "qam":
        return np.asarray(qam_modulate(bits, order, spf=True), dtype=np.uint32)
    raise ValueError(f"modulation must be 'psk' or 'qam', got {modulation!r}")


# -- closed loop ----------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OFDMLoopResult:
    """One closed-loop OFDM run: SPF + FP-reference BER for the same channel."""

    tx_bits: Bits
    rx_bits_spf: Bits
    rx_bits_fp: Bits
    ber_spf: float
    ber_fp: float


def run_ofdm_loop(
    snr_db: float,
    *,
    order: int = 4,
    modulation: str = "psk",
    demap: str = "auto",
    n_sub: int = 64,
    n_sym: int = 64,
    n_taps: int = 4,
    seed: int = 0,
    codebook: tuple[Complex, Bits] | None = None,
) -> OFDMLoopResult:
    """Run a full OFDM closed loop (bits -> map -> channel -> demap -> bits).

    Builds the SPF-native datapath and a parallel FP64 reference modem driven by
    the **same** channel realization and noise sample, so the two BERs are a fair
    iso-channel comparison (the P2 ``<= 0.5 dB`` SNR-gap bar).

    ``demap``:
      * ``"qpsk"``    — native ``ofdm_qpsk_eq_demod_fused`` (PSK order 4 only).
      * ``"qam16"``   — :class:`QAM16LogProjOperator` (QAM order 16 only).
      * ``"codebook"``— general min-over-codebook head (any constellation).
      * ``"auto"``    — qpsk for PSK-4, qam16 for QAM-16, else codebook.
    """
    rng = np.random.default_rng(seed)
    k = int(np.log2(order))
    n_data = n_sub * n_sym
    tx_bits = rng.integers(0, 2, size=k * n_data, dtype=np.uint8)

    # -- TX map (PolarSPF codec) + FP reference symbols.
    sym_words = _tx_map(tx_bits, order, modulation).reshape(n_sym, n_sub)
    sym_fp = decode_words(sym_words)

    # -- Subcarrier placement = identity gather (every active SC carries data).
    # (A permute/gather here would be a free monomial; identity for the dense grid.)

    # -- TX IFFT (SPF macro) -> time-domain frame.
    tx_time_words = spf32_fft.ifft(sym_words)
    tx_time = decode_words(tx_time_words)
    tx_time_fp = np.fft.ifft(sym_fp, axis=-1)

    # -- Frequency-selective channel + AWGN (FP64 wire-boundary propagation).
    h_freq = rayleigh_multipath(n_sub, n_taps, rng)  # (n_sub,)
    signal_power = float(np.mean(np.abs(tx_time) ** 2))
    noise_var = noise_variance_for_snr(signal_power, snr_db)
    noise = awgn_noise((n_sym, n_sub), float(noise_var), rng)

    def _propagate(time_signal: Complex) -> Complex:
        return np.fft.ifft(h_freq * np.fft.fft(time_signal, axis=-1), axis=-1) + noise

    rx_time = _propagate(tx_time)
    rx_time_fp = _propagate(tx_time_fp)

    # -- RX FFT (SPF) + scalar ZF equalize via diagonal_scale by Hhat^-1.
    rx_freq_words = spf32_fft.fft(encode_words(rx_time))
    h_inv = encode_words(np.broadcast_to(1.0 / h_freq, (n_sym, n_sub)).copy())

    # FP reference: RX FFT + ZF equalize (perfect CSI).
    rx_freq_fp = np.fft.fft(rx_time_fp, axis=-1)
    eq_fp = rx_freq_fp * (1.0 / h_freq)

    # -- Demap.
    mode = _resolve_demap(demap, order, modulation)
    rx_bits_spf, rx_bits_fp = _demap(
        mode, order, rx_freq_words, h_inv, eq_fp, codebook
    )

    return OFDMLoopResult(
        tx_bits=tx_bits,
        rx_bits_spf=rx_bits_spf,
        rx_bits_fp=rx_bits_fp,
        ber_spf=ber(tx_bits, rx_bits_spf),
        ber_fp=ber(tx_bits, rx_bits_fp),
    )


def _resolve_demap(demap: str, order: int, modulation: str) -> str:
    if demap != "auto":
        return demap
    if modulation == "psk" and order == 4:
        return "qpsk"
    if modulation == "qam" and order == 16:
        return "qam16"
    return "codebook"


def _demap(
    mode: str,
    order: int,
    rx_freq_words: Word,
    h_inv: Word,
    eq_fp: Complex,
    codebook: tuple[Complex, Bits] | None,
) -> tuple[Bits, Bits]:
    """Return ``(spf_bits, fp_bits)`` for the chosen demap head + FP reference."""
    rx = np.asarray(rx_freq_words, dtype=np.uint32).ravel()
    hi = np.asarray(h_inv, dtype=np.uint32).ravel()
    if mode == "qpsk":
        if order != 4:
            raise ValueError("qpsk demap requires order=4")
        spf_bits, _ = c_spf32.ofdm_qpsk_eq_demod_fused(
            rx, hi, cpe_theta=QPSK_SLICER_CPE_THETA
        )
        fp_bits = psk_demodulate_fp(eq_fp.ravel(), 4)
        return spf_bits, fp_bits
    if mode == "qam16":
        if order != 16:
            raise ValueError("qam16 demap requires order=16")
        spf_bits = QAM16LogProjOperator().demap(rx, hi).bits
        fp_bits = qam_demodulate(eq_fp.ravel(), 16, spf=False)
        return spf_bits, fp_bits
    if mode == "codebook":
        if codebook is None:
            raise ValueError("codebook demap requires a (points, bits) codebook")
        points, bits = codebook
        equalized = spf_mul_words(rx, hi)
        spf_bits = codebook_demap(equalized, points, bits)
        fp_bits = _codebook_demap_fp(eq_fp.ravel(), points, bits)
        return spf_bits, fp_bits
    raise ValueError(f"unknown demap mode {mode!r}")


def _codebook_demap_fp(eq: Complex, points: Complex, bits: Bits) -> Bits:
    """FP-reference nearest-codebook demap (mirrors :func:`codebook_demap`)."""
    cb = np.asarray(points, dtype=np.complex128).ravel()
    dist2 = np.abs(eq[:, None] - cb[None, :]) ** 2
    nearest = np.argmin(dist2, axis=1)
    out: Bits = np.asarray(bits, dtype=np.uint8)[nearest].ravel()
    return out


def snr_for_target_ber(
    target_ber: float,
    *,
    snr_grid: npt.ArrayLike,
    use_fp: bool,
    **loop_kwargs: object,
) -> float:
    """Smallest SNR on ``snr_grid`` whose mean BER is <= ``target_ber``.

    Averages over the seeds in ``loop_kwargs['seeds']`` (default ``(0,)``). Returns
    ``+inf`` if no grid point reaches the target. Used to measure the SPF-vs-FP
    SNR gap at a fixed BER (the P2 ``<= 0.5 dB`` bar).
    """
    seeds: tuple[int, ...] = tuple(loop_kwargs.pop("seeds", (0,)))  # type: ignore[arg-type]
    for snr in np.asarray(snr_grid, dtype=np.float64):
        bers: list[float] = []
        for seed in seeds:
            res = run_ofdm_loop(float(snr), seed=int(seed), **loop_kwargs)  # type: ignore[arg-type]
            bers.append(res.ber_fp if use_fp else res.ber_spf)
        if float(np.mean(bers)) <= target_ber:
            return float(snr)
    return float("inf")
