"""PSK and QAM modulation / demodulation (Gray-coded).

All SPF variants operate on ``ndarray[uint32]`` SPF-32 words.
All FP variants operate on ``complex128`` and serve as reference oracles.

Promoted from ``benchmarks/dsp/_helpers.py``.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from spf.scalar import spf32
from spf.scalar._constants import SPF32

__all__ = [
    "QAM16_SPF_LUT",
    "QAM64_SPF_LUT",
    "QAM256_SPF_LUT",
    "bits_per_symbol",
    "psk_demodulate",
    "psk_demodulate_fp",
    "psk_demodulate_spf",
    "psk_modulate",
    "psk_modulate_spf",
    "qam16_demodulate_fp",
    "qam16_demodulate_spf",
    "qam16_modulate_fp",
    "qam16_modulate_spf",
    "qam16_modulate_spf_direct",
    "qam64_demodulate_fp",
    "qam64_demodulate_spf",
    "qam64_modulate_fp",
    "qam64_modulate_spf",
    "qam64_modulate_spf_direct",
    "qam256_demodulate_fp",
    "qam256_demodulate_spf",
    "qam256_modulate_fp",
    "qam256_modulate_spf",
    "qam256_modulate_spf_direct",
    "qam_demodulate",
    "qam_max_amplitude",
    "qam_modulate",
]

Word = npt.NDArray[np.uint32]
Complex = npt.NDArray[np.complex128]

_THETA_MASK: int = SPF32.word_mask_theta  # 0xFFFF


# ── helpers ───────────────────────────────────────────────────────────────────


def bits_per_symbol(order: int) -> int:
    """Return ``log2(order)`` bits per symbol for PSK/QAM order."""
    if order < 2 or order & (order - 1):
        raise ValueError(f"order must be a power of two >= 2, got {order}")
    return int(math.log2(order))


def qam_max_amplitude(order: int) -> float:
    """Peak constellation amplitude for a given QAM/PSK order.

    Used to set int16 quantization scale so the full constellation maps to
    roughly half the int16 dynamic range, leaving 1 bit of headroom.
    """
    if order <= 8:
        return 1.0
    if order == 16:
        return 3.0
    if order == 64:
        return 7.0
    if order == 256:
        return 15.0
    raise ValueError(f"Unsupported order {order}")


# ── PSK ───────────────────────────────────────────────────────────────────────


def _check_psk_order(order: int) -> int:
    if order < 2 or order & (order - 1):
        raise ValueError(f"PSK order must be a power of two >= 2, got {order}")
    if (1 << 16) % order:
        raise ValueError(f"PSK order {order} does not divide 2^16 (Θ field width)")
    return int(math.log2(order))


def psk_modulate_spf(bits: npt.NDArray[np.uint8], order: int) -> Word:
    """PSK modulator: bits → unit-magnitude SPF-32 symbols via ``spf32.twiddle``.

    Groups ``k = log2(order)`` consecutive bits (MSB first) into a symbol
    index, then emits the SPF-32 word for ``e^(2πi·s/order)``.  L = 0 exactly
    — no Sb() cost, no magnitude error, symbols lie exactly on the unit circle.
    """
    k = _check_psk_order(order)
    if len(bits) % k:
        raise ValueError(f"bit length {len(bits)} not divisible by log2(order)={k}")
    bits_u64 = bits.astype(np.uint64, copy=False)
    groups = bits_u64.reshape(-1, k)
    weights = 1 << np.arange(k - 1, -1, -1, dtype=np.uint64)
    symbol_idx = (groups * weights).sum(axis=1)
    return spf32.twiddle(symbol_idx, order)


def psk_demodulate_spf(symbols: Word, order: int) -> npt.NDArray[np.uint8]:
    """PSK demodulator: nearest-symbol decision on the Θ field (integer only)."""
    k = _check_psk_order(order)
    theta = (symbols.astype(np.uint32) & _THETA_MASK).astype(np.int64)
    step = (1 << 16) // order
    idx = ((theta + step // 2) // step) % order
    bits = np.empty(len(symbols) * k, dtype=np.uint8)
    for i in range(k):
        bits[i::k] = ((idx >> (k - 1 - i)) & 1).astype(np.uint8)
    return bits


def psk_demodulate_fp(symbols: Complex, order: int) -> npt.NDArray[np.uint8]:
    """FP PSK demodulator: nearest-angle decision via ``np.angle``."""
    k = _check_psk_order(order)
    theta = np.mod(np.angle(symbols), 2 * np.pi)
    step = 2 * np.pi / order
    idx = np.round(theta / step).astype(np.int64) % order
    bits = np.empty(len(symbols) * k, dtype=np.uint8)
    for i in range(k):
        bits[i::k] = ((idx >> (k - 1 - i)) & 1).astype(np.uint8)
    return bits


# ── 16-QAM (Gray-coded) ───────────────────────────────────────────────────────

_QAM16_GRAY_TO_AMPL: npt.NDArray[np.int8] = np.array([-3, -1, +3, +1], dtype=np.int8)
_QAM16_AMPL_TO_GRAY: npt.NDArray[np.uint8] = np.array([0, 1, 3, 2], dtype=np.uint8)
QAM16_SPF_LUT: npt.NDArray[np.uint32] = np.asarray(
    [
        0x0216A000,
        0x01A98D1C,
        0x02166000,
        0x01A972E4,
        0x01A9B2E4,
        0x0080A000,
        0x01A94D1C,
        0x00806000,
        0x0216E000,
        0x01A9F2E4,
        0x02162000,
        0x01A90D1C,
        0x01A9CD1C,
        0x0080E000,
        0x01A932E4,
        0x00802000,
    ],
    dtype=np.uint32,
)


def qam16_modulate_fp(bits: npt.NDArray[np.uint8]) -> Complex:
    """Gray-coded 16-QAM: 4 bits → I+jQ, I,Q ∈ {±1,±3}."""
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}")
    if len(bits) % 4:
        raise ValueError(f"bit length {len(bits)} not divisible by 4 (16-QAM)")
    groups = bits.reshape(-1, 4).astype(np.uint8, copy=False)
    i_gray = (groups[:, 0] << 1) | groups[:, 1]
    q_gray = (groups[:, 2] << 1) | groups[:, 3]
    i_ampl = _QAM16_GRAY_TO_AMPL[i_gray].astype(np.float64)
    q_ampl = _QAM16_GRAY_TO_AMPL[q_gray].astype(np.float64)
    return np.asarray(i_ampl + 1j * q_ampl, dtype=np.complex128)


def qam16_modulate_spf(bits: npt.NDArray[np.uint8]) -> Word:
    """SPF-32 16-QAM: direct Gray-coded constellation lookup."""
    return qam16_modulate_spf_direct(bits)


def qam16_modulate_spf_direct(bits: npt.NDArray[np.uint8]) -> Word:
    """SPF-32 16-QAM using a bit-exact lookup table, with no FP encode."""
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}")
    if len(bits) % 4:
        raise ValueError(f"bit length {len(bits)} not divisible by 4 (16-QAM)")
    groups = bits.reshape(-1, 4).astype(np.uint8, copy=False)
    lut_idx = (groups[:, 0] << 3) | (groups[:, 1] << 2) | (groups[:, 2] << 1) | groups[:, 3]
    return np.ascontiguousarray(QAM16_SPF_LUT[lut_idx], dtype=np.uint32)


def _qam16_quantize_axis(r: npt.NDArray[np.floating]) -> npt.NDArray[np.int8]:
    sign = np.where(r >= 0.0, 1, -1).astype(np.int8)
    mag = np.where(np.abs(r) > 2.0, 3, 1).astype(np.int8)
    return (sign * mag).astype(np.int8)


def qam16_demodulate_fp(symbols: Complex) -> npt.NDArray[np.uint8]:
    """Hard-decision 16-QAM demod: quantize I/Q, Gray-decode."""
    flat = np.asarray(symbols).ravel()
    i_ampl = _qam16_quantize_axis(flat.real)
    q_ampl = _qam16_quantize_axis(flat.imag)
    i_idx = ((i_ampl.astype(np.int16) + 3) >> 1).astype(np.uint8)
    q_idx = ((q_ampl.astype(np.int16) + 3) >> 1).astype(np.uint8)
    i_gray = _QAM16_AMPL_TO_GRAY[i_idx]
    q_gray = _QAM16_AMPL_TO_GRAY[q_idx]
    bits = np.empty(flat.size * 4, dtype=np.uint8)
    bits[0::4] = (i_gray >> 1) & 1
    bits[1::4] = i_gray & 1
    bits[2::4] = (q_gray >> 1) & 1
    bits[3::4] = q_gray & 1
    return bits


def qam16_demodulate_spf(symbols: Word) -> npt.NDArray[np.uint8]:
    """SPF-32 16-QAM demod: decode to FP then hard-decide."""
    return qam16_demodulate_fp(spf32.decode(symbols))


# ── 64-QAM ────────────────────────────────────────────────────────────────────

_QAM64_GRAY_TO_AMPL: npt.NDArray[np.float64] = np.array([-7.0, -5.0, -1.0, -3.0, 7.0, 5.0, 1.0, 3.0], dtype=np.float64)
_QAM64_AMPL_TO_GRAY: npt.NDArray[np.uint8] = np.array([0, 1, 3, 2, 6, 7, 5, 4], dtype=np.uint8)


def qam64_modulate_fp(bits: npt.NDArray[np.uint8]) -> Complex:
    """Gray-coded 64-QAM: 6 bits → I+jQ, I,Q ∈ {±1,±3,±5,±7}."""
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}")
    if len(bits) % 6:
        raise ValueError(f"bit length {len(bits)} not divisible by 6 (64-QAM)")
    groups = bits.reshape(-1, 6).astype(np.uint8, copy=False)
    i_gray = (groups[:, 0] << 2) | (groups[:, 1] << 1) | groups[:, 2]
    q_gray = (groups[:, 3] << 2) | (groups[:, 4] << 1) | groups[:, 5]
    return np.asarray(_QAM64_GRAY_TO_AMPL[i_gray] + 1j * _QAM64_GRAY_TO_AMPL[q_gray], dtype=np.complex128)


# Direct bit→SPF constellation table (Gray-coded). Frozen so the modulator maps
# incoming values straight to SPF words with no runtime FP encode — bit-exact to
# ``spf32.encode(qam64_modulate_fp(bits))``; index = the 6 symbol bits (MSB first).
QAM64_SPF_LUT: npt.NDArray[np.uint32] = np.asarray(
    [
        0x034FA000, 0x031B9945, 0x02D285C8, 0x02EE907F, 0x034F6000, 0x031B66BB, 0x02D27A38, 0x02EE6F81,
        0x031BA6BB, 0x02D2A000, 0x025A880B, 0x028B9605, 0x031B5945, 0x02D26000, 0x025A77F5, 0x028B69FB,
        0x02D2BA38, 0x025AB7F5, 0x0080A000, 0x01A9B2E4, 0x02D245C8, 0x025A480B, 0x00806000, 0x01A94D1C,
        0x02EEAF81, 0x028BA9FB, 0x01A98D1C, 0x0216A000, 0x02EE507F, 0x028B5605, 0x01A972E4, 0x02166000,
        0x034FE000, 0x031BE6BB, 0x02D2FA38, 0x02EEEF81, 0x034F2000, 0x031B1945, 0x02D205C8, 0x02EE107F,
        0x031BD945, 0x02D2E000, 0x025AF7F5, 0x028BE9FB, 0x031B26BB, 0x02D22000, 0x025A080B, 0x028B1605,
        0x02D2C5C8, 0x025AC80B, 0x0080E000, 0x01A9CD1C, 0x02D23A38, 0x025A37F5, 0x00802000, 0x01A932E4,
        0x02EED07F, 0x028BD605, 0x01A9F2E4, 0x0216E000, 0x02EE2F81, 0x028B29FB, 0x01A90D1C, 0x02162000,
    ],
    dtype=np.uint32,
)


def qam64_modulate_spf_direct(bits: npt.NDArray[np.uint8]) -> Word:
    """SPF-32 64-QAM via a bit-exact lookup table, with no FP encode."""
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}")
    if len(bits) % 6:
        raise ValueError(f"bit length {len(bits)} not divisible by 6 (64-QAM)")
    g = bits.reshape(-1, 6).astype(np.uint32, copy=False)
    lut_idx = (g[:, 0] << 5) | (g[:, 1] << 4) | (g[:, 2] << 3) | (g[:, 3] << 2) | (g[:, 4] << 1) | g[:, 5]
    return np.ascontiguousarray(QAM64_SPF_LUT[lut_idx], dtype=np.uint32)


def qam64_modulate_spf(bits: npt.NDArray[np.uint8]) -> Word:
    """SPF-32 64-QAM: direct Gray-coded constellation lookup (no FP encode)."""
    return qam64_modulate_spf_direct(bits)


def _qam64_quantize_axis(r: npt.NDArray[np.floating]) -> npt.NDArray[np.int8]:
    abs_r = np.abs(r)
    mag = np.where(abs_r > 6.0, 7, np.where(abs_r > 4.0, 5, np.where(abs_r > 2.0, 3, 1))).astype(np.int8)
    return (np.where(r >= 0.0, 1, -1).astype(np.int8) * mag).astype(np.int8)


def qam64_demodulate_fp(symbols: Complex) -> npt.NDArray[np.uint8]:
    """Hard-decision 64-QAM demod: quantize I/Q, Gray-decode."""
    flat = np.asarray(symbols).ravel()
    i_ampl = _qam64_quantize_axis(flat.real)
    q_ampl = _qam64_quantize_axis(flat.imag)
    i_idx = ((i_ampl.astype(np.int16) + 7) >> 1).astype(np.uint8)
    q_idx = ((q_ampl.astype(np.int16) + 7) >> 1).astype(np.uint8)
    i_gray = _QAM64_AMPL_TO_GRAY[i_idx]
    q_gray = _QAM64_AMPL_TO_GRAY[q_idx]
    bits = np.empty(flat.size * 6, dtype=np.uint8)
    bits[0::6] = (i_gray >> 2) & 1
    bits[1::6] = (i_gray >> 1) & 1
    bits[2::6] = i_gray & 1
    bits[3::6] = (q_gray >> 2) & 1
    bits[4::6] = (q_gray >> 1) & 1
    bits[5::6] = q_gray & 1
    return bits


def qam64_demodulate_spf(symbols: Word) -> npt.NDArray[np.uint8]:
    """SPF-32 64-QAM demod: decode to FP then hard-decide."""
    return qam64_demodulate_fp(spf32.decode(symbols))


# ── 256-QAM ───────────────────────────────────────────────────────────────────

_QAM256_GRAY_TO_AMPL: npt.NDArray[np.float64] = np.array(
    [-15.0, -13.0, -9.0, -11.0, -1.0, -3.0, -7.0, -5.0, 15.0, 13.0, 9.0, 11.0, 1.0, 3.0, 7.0, 5.0],
    dtype=np.float64,
)
_QAM256_AMPL_TO_GRAY: npt.NDArray[np.uint8] = np.array(
    [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8], dtype=np.uint8
)


def qam256_modulate_fp(bits: npt.NDArray[np.uint8]) -> Complex:
    """Gray-coded 256-QAM: 8 bits → I+jQ, I,Q ∈ {±1,±3,...,±15}."""
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}")
    if len(bits) % 8:
        raise ValueError(f"bit length {len(bits)} not divisible by 8 (256-QAM)")
    g = bits.reshape(-1, 8).astype(np.uint8, copy=False)
    i_gray = (g[:, 0] << 3) | (g[:, 1] << 2) | (g[:, 2] << 1) | g[:, 3]
    q_gray = (g[:, 4] << 3) | (g[:, 5] << 2) | (g[:, 6] << 1) | g[:, 7]
    return np.asarray(_QAM256_GRAY_TO_AMPL[i_gray] + 1j * _QAM256_GRAY_TO_AMPL[q_gray], dtype=np.complex128)


# Direct bit→SPF constellation table (Gray-coded). Frozen so the modulator maps
# incoming values straight to SPF words with no runtime FP encode — bit-exact to
# ``spf32.encode(qam256_modulate_fp(bits).astype(complex64))``; index = the 8
# symbol bits (MSB first).
#
# Regenerated 2026-09-18: the four symbols at |I| = |Q| = 9 (indices 34, 42,
# 162, 170) moved L 0x03AB → 0x03AC. |z| = 9·√2 in complex64 lands on an exact
# half-tick of the Q7.8 log grid, where the retired degree-3 ``spf32_fast_log2``
# encoder broke the tie downward; AlgRef v0.2 §3.1 requires exact RNE
# (libm log2 + rint), which breaks it upward. 4/256 entries changed; QAM16 and
# QAM64 are unaffected (0/64 drift). See the 2026-09-18
# mathematical-conformance audit, deviation #10.
QAM256_SPF_LUT: npt.NDArray[np.uint32] = np.asarray(
    [
        0x0468A000, 0x04509D18, 0x04219605, 0x043899C8, 0x03E982B6, 0x03EF880B, 0x040D91CA, 0x03FC8D1C,
        0x04686000, 0x045062E8, 0x042169FB, 0x04386638, 0x03E97D4A, 0x03EF77F5, 0x040D6E36, 0x03FC72E4,
        0x0450A2E8, 0x0433A000, 0x03FC98AC, 0x04179C9D, 0x03B48321, 0x03BD893E, 0x03E29420, 0x03CD8EF6,
        0x04505D18, 0x04336000, 0x03FC6754, 0x04176363, 0x03B47CDF, 0x03BD76C2, 0x03E26BE0, 0x03CD710A,
        0x0421A9FB, 0x03FCA754, 0x03ACA000, 0x03D4A410, 0x032E8482, 0x033F8D1C, 0x03839AEF, 0x035D94A9,
        0x04215605, 0x03FC58AC, 0x03AC6000, 0x03D45BF0, 0x032E7B7E, 0x033F72E4, 0x03836511, 0x035D6B57,
        0x0438A638, 0x0417A363, 0x03D49BF0, 0x03F6A000, 0x037783B2, 0x03838AD9, 0x03B49717, 0x03989162,
        0x043859C8, 0x04175C9D, 0x03D46410, 0x03F66000, 0x03777C4E, 0x03837527, 0x03B468E9, 0x03986E9E,
        0x03E9BD4A, 0x03B4BCDF, 0x032EBB7E, 0x0377BC4E, 0x0080A000, 0x01A9B2E4, 0x02D2BA38, 0x025AB7F5,
        0x03E942B6, 0x03B44321, 0x032E4482, 0x037743B2, 0x00806000, 0x01A94D1C, 0x02D245C8, 0x025A480B,
        0x03EFB7F5, 0x03BDB6C2, 0x033FB2E4, 0x0383B527, 0x01A98D1C, 0x0216A000, 0x02EEAF81, 0x028BA9FB,
        0x03EF480B, 0x03BD493E, 0x033F4D1C, 0x03834AD9, 0x01A972E4, 0x02166000, 0x02EE507F, 0x028B5605,
        0x040DAE36, 0x03E2ABE0, 0x0383A511, 0x03B4A8E9, 0x02D285C8, 0x02EE907F, 0x034FA000, 0x031B9945,
        0x040D51CA, 0x03E25420, 0x03835AEF, 0x03B45717, 0x02D27A38, 0x02EE6F81, 0x034F6000, 0x031B66BB,
        0x03FCB2E4, 0x03CDB10A, 0x035DAB57, 0x0398AE9E, 0x025A880B, 0x028B9605, 0x031BA6BB, 0x02D2A000,
        0x03FC4D1C, 0x03CD4EF6, 0x035D54A9, 0x03985162, 0x025A77F5, 0x028B69FB, 0x031B5945, 0x02D26000,
        0x0468E000, 0x0450E2E8, 0x0421E9FB, 0x0438E638, 0x03E9FD4A, 0x03EFF7F5, 0x040DEE36, 0x03FCF2E4,
        0x04682000, 0x04501D18, 0x04211605, 0x043819C8, 0x03E902B6, 0x03EF080B, 0x040D11CA, 0x03FC0D1C,
        0x0450DD18, 0x0433E000, 0x03FCE754, 0x0417E363, 0x03B4FCDF, 0x03BDF6C2, 0x03E2EBE0, 0x03CDF10A,
        0x045022E8, 0x04332000, 0x03FC18AC, 0x04171C9D, 0x03B40321, 0x03BD093E, 0x03E21420, 0x03CD0EF6,
        0x0421D605, 0x03FCD8AC, 0x03ACE000, 0x03D4DBF0, 0x032EFB7E, 0x033FF2E4, 0x0383E511, 0x035DEB57,
        0x042129FB, 0x03FC2754, 0x03AC2000, 0x03D42410, 0x032E0482, 0x033F0D1C, 0x03831AEF, 0x035D14A9,
        0x0438D9C8, 0x0417DC9D, 0x03D4E410, 0x03F6E000, 0x0377FC4E, 0x0383F527, 0x03B4E8E9, 0x0398EE9E,
        0x04382638, 0x04172363, 0x03D41BF0, 0x03F62000, 0x037703B2, 0x03830AD9, 0x03B41717, 0x03981162,
        0x03E9C2B6, 0x03B4C321, 0x032EC482, 0x0377C3B2, 0x0080E000, 0x01A9CD1C, 0x02D2C5C8, 0x025AC80B,
        0x03E93D4A, 0x03B43CDF, 0x032E3B7E, 0x03773C4E, 0x00802000, 0x01A932E4, 0x02D23A38, 0x025A37F5,
        0x03EFC80B, 0x03BDC93E, 0x033FCD1C, 0x0383CAD9, 0x01A9F2E4, 0x0216E000, 0x02EED07F, 0x028BD605,
        0x03EF37F5, 0x03BD36C2, 0x033F32E4, 0x03833527, 0x01A90D1C, 0x02162000, 0x02EE2F81, 0x028B29FB,
        0x040DD1CA, 0x03E2D420, 0x0383DAEF, 0x03B4D717, 0x02D2FA38, 0x02EEEF81, 0x034FE000, 0x031BE6BB,
        0x040D2E36, 0x03E22BE0, 0x03832511, 0x03B428E9, 0x02D205C8, 0x02EE107F, 0x034F2000, 0x031B1945,
        0x03FCCD1C, 0x03CDCEF6, 0x035DD4A9, 0x0398D162, 0x025AF7F5, 0x028BE9FB, 0x031BD945, 0x02D2E000,
        0x03FC32E4, 0x03CD310A, 0x035D2B57, 0x03982E9E, 0x025A080B, 0x028B1605, 0x031B26BB, 0x02D22000,
    ],
    dtype=np.uint32,
)


def qam256_modulate_spf_direct(bits: npt.NDArray[np.uint8]) -> Word:
    """SPF-32 256-QAM via a bit-exact lookup table, with no FP encode."""
    if bits.ndim != 1:
        raise ValueError(f"bits must be 1-D, got shape {bits.shape}")
    if len(bits) % 8:
        raise ValueError(f"bit length {len(bits)} not divisible by 8 (256-QAM)")
    g = bits.reshape(-1, 8).astype(np.uint32, copy=False)
    lut_idx = (
        (g[:, 0] << 7) | (g[:, 1] << 6) | (g[:, 2] << 5) | (g[:, 3] << 4)
        | (g[:, 4] << 3) | (g[:, 5] << 2) | (g[:, 6] << 1) | g[:, 7]
    )
    return np.ascontiguousarray(QAM256_SPF_LUT[lut_idx], dtype=np.uint32)


def qam256_modulate_spf(bits: npt.NDArray[np.uint8]) -> Word:
    """SPF-32 256-QAM: direct Gray-coded constellation lookup (no FP encode)."""
    return qam256_modulate_spf_direct(bits)


def _qam256_quantize_axis(r: npt.NDArray[np.floating]) -> npt.NDArray[np.int8]:
    idx = np.clip(np.round((np.clip(r, -16.0, 16.0) + 15.0) / 2.0).astype(np.int32), 0, 15)
    _lookup: npt.NDArray[np.int8] = np.array(
        [-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15], dtype=np.int8
    )
    return np.asarray(_lookup[idx], dtype=np.int8)


def qam256_demodulate_fp(symbols: Complex) -> npt.NDArray[np.uint8]:
    """Hard-decision 256-QAM: quantize I/Q, Gray-decode."""
    flat = np.asarray(symbols).ravel()
    i_a = _qam256_quantize_axis(flat.real)
    q_a = _qam256_quantize_axis(flat.imag)
    i_g = _QAM256_AMPL_TO_GRAY[((i_a.astype(np.int16) + 15) >> 1).astype(np.uint8)]
    q_g = _QAM256_AMPL_TO_GRAY[((q_a.astype(np.int16) + 15) >> 1).astype(np.uint8)]
    out = np.empty(flat.size * 8, dtype=np.uint8)
    out[0::8] = (i_g >> 3) & 1
    out[1::8] = (i_g >> 2) & 1
    out[2::8] = (i_g >> 1) & 1
    out[3::8] = i_g & 1
    out[4::8] = (q_g >> 3) & 1
    out[5::8] = (q_g >> 2) & 1
    out[6::8] = (q_g >> 1) & 1
    out[7::8] = q_g & 1
    return out


def qam256_demodulate_spf(symbols: Word) -> npt.NDArray[np.uint8]:
    """SPF-32 256-QAM demod: decode to FP then hard-decide."""
    return qam256_demodulate_fp(spf32.decode(symbols))


# ── unified dispatch ──────────────────────────────────────────────────────────

_MODULATE_FP = {4: qam16_modulate_fp, 16: qam16_modulate_fp, 64: qam64_modulate_fp, 256: qam256_modulate_fp}
_MODULATE_SPF = {4: qam16_modulate_spf, 16: qam16_modulate_spf, 64: qam64_modulate_spf, 256: qam256_modulate_spf}
_DEMODULATE_FP = {4: qam16_demodulate_fp, 16: qam16_demodulate_fp, 64: qam64_demodulate_fp, 256: qam256_demodulate_fp}
_DEMODULATE_SPF = {
    4: qam16_demodulate_spf,
    16: qam16_demodulate_spf,
    64: qam64_demodulate_spf,
    256: qam256_demodulate_spf,
}


def qam_modulate(bits: npt.NDArray[np.uint8], order: int, *, spf: bool = True) -> Word | Complex:
    """Unified QAM modulator (16/64/256-QAM).

    Parameters
    ----------
    bits : ndarray[uint8]
        Bit stream; length must be a multiple of ``log2(order)``.
    order : int
        Constellation order: 16, 64, or 256.
    spf : bool
        If ``True`` (default) return SPF-32 words; else return complex128.
    """
    if order == 4:
        raise ValueError("Use psk_modulate_spf/psk_demodulate_spf for QPSK (order=4)")
    table = _MODULATE_SPF if spf else _MODULATE_FP
    if order not in table:
        raise ValueError(f"Unsupported QAM order {order}; choose 16, 64, or 256")
    return table[order](bits)


def qam_demodulate(symbols: Word | Complex, order: int, *, spf: bool | None = None) -> npt.NDArray[np.uint8]:
    """Unified QAM demodulator (16/64/256-QAM).

    Parameters
    ----------
    symbols : ndarray
        SPF-32 words (uint32) or complex128 symbols.
    order : int
        Constellation order: 16, 64, or 256.
    spf : bool or None
        If ``None`` (default), auto-detect from dtype.
    """
    if order == 4:
        raise ValueError("Use psk_demodulate_spf/psk_demodulate_fp for QPSK (order=4)")
    if order not in _DEMODULATE_SPF:
        raise ValueError(f"Unsupported QAM order {order}; choose 16, 64, or 256")
    use_spf = (np.asarray(symbols).dtype == np.uint32) if spf is None else spf
    if use_spf:
        return _DEMODULATE_SPF[order](np.asarray(symbols, dtype=np.uint32))
    return _DEMODULATE_FP[order](np.asarray(symbols, dtype=np.complex128))


psk_modulate = psk_modulate_spf
psk_demodulate = psk_demodulate_spf
