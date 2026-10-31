"""SBF-128 — Spectral Block Float codec (spec §5).

A block of 128 real FP16 activations stored as its rfft coefficients grouped
into 8 frequency bands with per-band E8M0 exponents and water-filling bit
allocation (FP16 / INT8 / INT4 / INT2). Bins 56..64 are pruned (zeroed).

Output layout (spec §5.2):

    byte 0         header (version=0, flags=0)
    bytes 1..8     8 × uint8 E8M0 band exponents (0xFF = zero-band sentinel)
    bytes 9..24    band 0: bins 0–3   → 4 complex × FP16 (16 B)
    bytes 25..40   band 1: bins 4–7   → 4 complex × FP16 (16 B)
    bytes 41..56   band 2: bins 8–15  → 8 complex × INT8 (16 B)
    bytes 57..72   band 3: bins 16–23 → 8 complex × INT8 (16 B)
    bytes 73..80   band 4: bins 24–31 → 8 complex × INT4 (8 B)
    bytes 81..88   band 5: bins 32–39 → 8 complex × INT4 (8 B)
    bytes 89..92   band 6: bins 40–47 → 8 complex × INT2 (4 B)
    bytes 93..96   band 7: bins 48–55 → 8 complex × INT2 (4 B)
    (bins 56..64 zeroed)

Total: 97 B vs. 256 B for raw FP16 (2.64× compression).

Quantization (spec §5.3): each coefficient C stored at precision p bits
decodes to ``C / (2^(p-1) - 1) × 2^(exp - 127)`` where ``exp`` is the band
exponent. For INT2 the divisor is 1 (not 3); the spec uses [-2, 1] codes so
the level step is literally 1 unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

BLOCK_SIZE: int = 97
"""Encoded block size in bytes (fixed, spec §5.2)."""

BLOCK_LENGTH: int = 128
"""Number of real samples per block (spec §5.1)."""

_SPECTRUM_LENGTH: int = BLOCK_LENGTH // 2 + 1  # 65 unique bins of rfft(128)

_Precision = Literal["fp16", "int8", "int4", "int2"]


@dataclass(frozen=True, slots=True)
class _Band:
    lo: int  # inclusive spectrum bin index
    hi: int  # exclusive spectrum bin index
    prec: _Precision
    data_bytes: int  # number of bytes of payload in the block


_BANDS: tuple[_Band, ...] = (
    _Band(0, 4, "fp16", 16),
    _Band(4, 8, "fp16", 16),
    _Band(8, 16, "int8", 16),
    _Band(16, 24, "int8", 16),
    _Band(24, 32, "int4", 8),
    _Band(32, 40, "int4", 8),
    _Band(40, 48, "int2", 4),
    _Band(48, 56, "int2", 4),
)

_ZERO_BAND_SENTINEL: int = 0xFF
_E8M0_BIAS: int = 127
_E8M0_MAX: int = 254  # 0xFF reserved as zero-band sentinel


# --------------------------------------------------------------------------- #
# Bit-packing helpers (signed two's complement INT4 and INT2).
# --------------------------------------------------------------------------- #


def _pack_int4(vals: np.ndarray) -> bytes:
    """Pack an even-length int8 array of signed 4-bit values into bytes.

    ``vals[2k]`` lands in the low nibble of byte ``k``, ``vals[2k+1]`` in the
    high nibble. Caller is responsible for clipping to ``[-8, 7]`` — we mask
    to 4 bits without checking.
    """
    if vals.ndim != 1 or vals.shape[0] % 2 != 0:
        raise ValueError(f"pack_int4 requires 1-D even-length array, got shape {vals.shape}")
    u = vals.astype(np.uint8) & np.uint8(0x0F)
    lo = u[0::2]
    hi = u[1::2]
    packed = (lo | (hi << np.uint8(4))).astype(np.uint8)
    out: bytes = packed.tobytes()
    return out


def _unpack_int4(buf: bytes) -> np.ndarray:
    """Unpack packed INT4 bytes into an int8 array with sign extension."""
    b = np.frombuffer(buf, dtype=np.uint8)
    lo = b & np.uint8(0x0F)
    hi = (b >> np.uint8(4)) & np.uint8(0x0F)
    out = np.empty(2 * b.shape[0], dtype=np.int8)
    out[0::2] = _sign_extend(lo, bits=4)
    out[1::2] = _sign_extend(hi, bits=4)
    return out


def _pack_int2(vals: np.ndarray) -> bytes:
    """Pack a length-multiple-of-4 int8 array of signed 2-bit values into bytes.

    Byte ``k`` holds (low-to-high): ``vals[4k]``, ``vals[4k+1]``, ``vals[4k+2]``,
    ``vals[4k+3]``. Caller is responsible for clipping to ``[-2, 1]``.
    """
    if vals.ndim != 1 or vals.shape[0] % 4 != 0:
        raise ValueError(
            f"pack_int2 requires 1-D array with length multiple of 4, got shape {vals.shape}"
        )
    u = vals.astype(np.uint8) & np.uint8(0x03)
    packed = (
        u[0::4]
        | (u[1::4] << np.uint8(2))
        | (u[2::4] << np.uint8(4))
        | (u[3::4] << np.uint8(6))
    ).astype(np.uint8)
    out: bytes = packed.tobytes()
    return out


def _unpack_int2(buf: bytes) -> np.ndarray:
    """Unpack packed INT2 bytes into an int8 array with sign extension."""
    b = np.frombuffer(buf, dtype=np.uint8)
    out = np.empty(4 * b.shape[0], dtype=np.int8)
    out[0::4] = _sign_extend(b & np.uint8(0x03), bits=2)
    out[1::4] = _sign_extend((b >> np.uint8(2)) & np.uint8(0x03), bits=2)
    out[2::4] = _sign_extend((b >> np.uint8(4)) & np.uint8(0x03), bits=2)
    out[3::4] = _sign_extend((b >> np.uint8(6)) & np.uint8(0x03), bits=2)
    return out


def _sign_extend(values: np.ndarray, *, bits: int) -> np.ndarray:
    """Sign-extend an unsigned n-bit field held in a uint8 array to int8."""
    sign_bit = 1 << (bits - 1)
    modulus = 1 << bits
    signed = values.astype(np.int16)
    signed = np.where(signed >= sign_bit, signed - modulus, signed)
    return signed.astype(np.int8)


# --------------------------------------------------------------------------- #
# Encode / decode.
# --------------------------------------------------------------------------- #


def encode(x: np.ndarray) -> bytes:
    """Encode a length-128 float16 block as a 97-byte SBF-128 block (spec §5.4)."""
    if x.shape != (BLOCK_LENGTH,):
        raise ValueError(f"encode expects shape ({BLOCK_LENGTH},), got {x.shape}")
    if x.dtype != np.float16:
        raise ValueError(f"encode expects dtype float16, got {x.dtype}")

    spectrum = np.fft.rfft(x.astype(np.float32))
    return _quantize_spectrum(spectrum)


def _quantize_spectrum(spectrum: np.ndarray) -> bytes:
    """Quantize a length-65 complex spectrum into a 97-byte SBF-128 block.

    This is the rfft-free half of ``encode`` — it's factored out so the
    ``convert`` layer can build SBF-128 blocks directly from a precomputed
    spectrum (e.g., the output of an SPF-32 FFT, per T6.1).
    """
    if spectrum.shape != (_SPECTRUM_LENGTH,):
        raise ValueError(
            f"_quantize_spectrum expects shape ({_SPECTRUM_LENGTH},), got {spectrum.shape}"
        )

    header = bytes([0x00])
    band_exps = bytearray(8)
    band_data = bytearray()

    for idx, band in enumerate(_BANDS):
        coeffs = spectrum[band.lo : band.hi]
        max_abs = float(np.max(np.abs(coeffs)))
        if max_abs == 0.0:
            band_exps[idx] = _ZERO_BAND_SENTINEL
            band_data.extend(bytes(band.data_bytes))
            continue

        # Spec §5.4 literally reads ``floor(log2(max_abs)) + 127``, but §5.3
        # cites OCP MX semantics: the scale must satisfy ``max_abs / scale ≤ 1``
        # so INT8/INT4/INT2 quantization doesn't clip the peak coefficient by
        # ~2×. We use ``ceil`` (the MX-compliant choice) and accept the
        # one-line divergence from §5.4's pseudocode.
        raw_exp = int(np.ceil(np.log2(max_abs))) + _E8M0_BIAS
        exp = max(0, min(_E8M0_MAX, raw_exp))
        band_exps[idx] = exp
        scale = 2.0 ** (exp - _E8M0_BIAS)

        real = np.asarray(coeffs.real / scale, dtype=np.float64)
        imag = np.asarray(coeffs.imag / scale, dtype=np.float64)
        band_data.extend(_quantize_band(real, imag, band.prec))

    assert len(band_data) == sum(b.data_bytes for b in _BANDS)  # 88 bytes
    out = header + bytes(band_exps) + bytes(band_data)
    assert len(out) == BLOCK_SIZE
    return out


def decode(block: bytes) -> np.ndarray:
    """Decode a 97-byte SBF-128 block to a length-128 float16 array (spec §5.5)."""
    spectrum = get_spectrum(block)
    x = np.fft.irfft(spectrum, n=BLOCK_LENGTH)
    return x.astype(np.float16)


def get_spectrum(block: bytes) -> np.ndarray:
    """Return the dequantized spectrum (65 complex bins) without inverse FFT.

    Bins 56..64 are always zero (spectral pruning). This is the API surface
    used by the T2.3 convolution-theorem test.
    """
    if len(block) != BLOCK_SIZE:
        raise ValueError(f"block must be {BLOCK_SIZE} bytes, got {len(block)}")

    band_exps = block[1:9]
    spectrum = np.zeros(_SPECTRUM_LENGTH, dtype=np.complex64)
    offset = 9

    for idx, band in enumerate(_BANDS):
        raw = block[offset : offset + band.data_bytes]
        offset += band.data_bytes
        exp = band_exps[idx]
        if exp == _ZERO_BAND_SENTINEL:
            continue
        scale = 2.0 ** (exp - _E8M0_BIAS)
        real, imag = _dequantize_band(raw, band)
        spectrum[band.lo : band.hi] = ((real + 1j * imag) * scale).astype(np.complex64)

    return spectrum


# --------------------------------------------------------------------------- #
# Per-band quantize / dequantize.
# --------------------------------------------------------------------------- #


def _quantize_band(real: np.ndarray, imag: np.ndarray, prec: _Precision) -> bytes:
    if prec == "fp16":
        return real.astype(np.float16).tobytes() + imag.astype(np.float16).tobytes()
    if prec == "int8":
        r = np.clip(np.round(real * 127.0), -128, 127).astype(np.int8)
        i = np.clip(np.round(imag * 127.0), -128, 127).astype(np.int8)
        return r.tobytes() + i.tobytes()
    if prec == "int4":
        r = np.clip(np.round(real * 7.0), -8, 7).astype(np.int8)
        i = np.clip(np.round(imag * 7.0), -8, 7).astype(np.int8)
        interleaved = np.stack([r, i], axis=1).reshape(-1)
        return _pack_int4(interleaved)
    # int2
    r = np.clip(np.round(real * 1.0), -2, 1).astype(np.int8)
    i = np.clip(np.round(imag * 1.0), -2, 1).astype(np.int8)
    interleaved = np.stack([r, i], axis=1).reshape(-1)
    return _pack_int2(interleaved)


def _dequantize_band(raw: bytes, band: _Band) -> tuple[np.ndarray, np.ndarray]:
    n = band.hi - band.lo
    if band.prec == "fp16":
        r = np.frombuffer(raw[: n * 2], dtype=np.float16).astype(np.float32)
        i = np.frombuffer(raw[n * 2 : n * 4], dtype=np.float16).astype(np.float32)
        return r, i
    if band.prec == "int8":
        r = np.frombuffer(raw[:n], dtype=np.int8).astype(np.float32) / 127.0
        i = np.frombuffer(raw[n : n * 2], dtype=np.int8).astype(np.float32) / 127.0
        return r, i
    if band.prec == "int4":
        interleaved = _unpack_int4(raw).astype(np.float32) / 7.0
        return interleaved[0::2], interleaved[1::2]
    # int2
    interleaved = _unpack_int2(raw).astype(np.float32) / 1.0
    return interleaved[0::2], interleaved[1::2]
