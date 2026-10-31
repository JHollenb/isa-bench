"""
SPF-FB7 field-mode FFT primitives.

This module extends fused_butterfly.py with an unpacked internal field
representation for FFT/radar/OFDM pipelines.

Why this exists
---------------
The packed SPF32 word is the ABI. It is *not* the best internal pipeline state.
A radix-2 FFT repeatedly unpacks, normalizes, adds/subtracts, and repacks the
same lanes. Field mode does this instead:

    load packed SPF32  -> unpack once into internal fields with guard bits
    all FFT stages     -> operate on fields
    store output       -> round/pack once

That gives two wins:
  1. less pack/unpack/classify work in HLS inner loops;
  2. internal L/phase guard bits, especially useful for half-angle FB7.

The wire format remains unchanged:
    word = [signed int16 L_q7.8][uint16 theta]
    zero = 0x80000000

Promoted to spf.fft.fb7_fields in Phase 2 (SPFv2 promotion) once the HLS
field kernel, OFDM, and radar tests passed.
"""
from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .fused import (
    L_CODE_INF,
    L_CODE_MAX_FINITE,
    L_CODE_MIN_FINITE,
    L_CODE_ZERO,
    SPF,
    SPF32_WORD_NAN,
    SPF32_WORD_ZERO,
    SPF_L_SCALE,
    SPF_PHASE_MASK,
    atan2_approx_yal,
    atan_approx_yal,
    decode_complex,
    encode_complex,
    pack_spf32,
    sin_poly7_first_quadrant,
    unpack_spf32,
)

TAU = 2.0 * math.pi

CLASS_NORMAL = 0
CLASS_NAN = 1
CLASS_INF = 2
CLASS_ZERO = 4


def _classify_spf(x: SPF) -> int:
    if x.L_q78 == L_CODE_ZERO:
        return CLASS_ZERO
    if x.L_q78 == L_CODE_INF:
        return CLASS_INF
    return CLASS_NORMAL


def _round_shift_signed(x: int, shift: int) -> int:
    if shift <= 0:
        return int(x)
    half = 1 << (shift - 1)
    if x >= 0:
        return (x + half) >> shift
    return -(((-x) + half) >> shift)


def _clamp_l_q78(x: int) -> int:
    if x <= L_CODE_ZERO:
        return L_CODE_ZERO
    return max(L_CODE_MIN_FINITE, min(L_CODE_MAX_FINITE, int(x)))


@dataclass(slots=True)
class SPFField:
    """Unpacked internal SPF lane with optional guard bits.

    L_int is log2(amplitude) in Q7.(8+l_guard).
    theta_int is phase in unsigned BAM with 16+t_guard bits.
    cls is CLASS_NORMAL/CLASS_ZERO/CLASS_INF/CLASS_NAN.
    """

    L_int: int
    theta_int: int
    cls: int = CLASS_NORMAL
    l_guard: int = 2
    t_guard: int = 2

    @property
    def l_scale(self) -> int:
        return SPF_L_SCALE << self.l_guard

    @property
    def phase_bits(self) -> int:
        return 16 + self.t_guard

    @property
    def phase_mod(self) -> int:
        return 1 << self.phase_bits

    @property
    def phase_mask(self) -> int:
        return self.phase_mod - 1

    def copy(self) -> SPFField:
        return SPFField(self.L_int, self.theta_int, self.cls, self.l_guard, self.t_guard)

    def normalized(self) -> SPFField:
        if self.cls & CLASS_NAN:
            return SPFField(0, 0, CLASS_NAN, self.l_guard, self.t_guard)
        if self.cls & CLASS_ZERO:
            return SPFField(L_CODE_ZERO << self.l_guard, 0, CLASS_ZERO, self.l_guard, self.t_guard)
        if self.cls & CLASS_INF:
            return SPFField(
            L_CODE_INF << self.l_guard, self.theta_int & self.phase_mask,
            CLASS_INF, self.l_guard, self.t_guard,
        )
        return SPFField(
            self.L_int, self.theta_int & self.phase_mask,
            CLASS_NORMAL, self.l_guard, self.t_guard,
        )


def field_from_spf(x: SPF, *, l_guard: int = 2, t_guard: int = 2) -> SPFField:
    cls = _classify_spf(x)
    if cls & CLASS_ZERO:
        return SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
    return SPFField(
        int(x.L_q78) << l_guard, int(x.theta_u16) << t_guard, cls, l_guard, t_guard
    ).normalized()


def field_from_word(w: int, *, l_guard: int = 2, t_guard: int = 2) -> SPFField:
    if int(w) == SPF32_WORD_NAN:
        return SPFField(0, 0, CLASS_NAN, l_guard, t_guard)
    return field_from_spf(unpack_spf32(w), l_guard=l_guard, t_guard=t_guard)


def field_to_spf(f: SPFField) -> SPF:
    f = f.normalized()
    if f.cls & CLASS_NAN:
        # Python SPF dataclass has no NaN lane; callers needing exact word use field_to_word.
        return SPF(L_CODE_ZERO, 0)
    if f.cls & CLASS_ZERO:
        return SPF(L_CODE_ZERO, 0)
    L_q78 = _clamp_l_q78(_round_shift_signed(f.L_int, f.l_guard))
    if L_q78 == L_CODE_ZERO:
        return SPF(L_CODE_ZERO, 0)
    if f.t_guard:
        theta = ((f.theta_int + (1 << (f.t_guard - 1))) >> f.t_guard) & SPF_PHASE_MASK
    else:
        theta = f.theta_int & SPF_PHASE_MASK
    return SPF(L_q78, theta)


def field_to_word(f: SPFField) -> int:
    f = f.normalized()
    if f.cls & CLASS_NAN:
        return SPF32_WORD_NAN
    if f.cls & CLASS_ZERO:
        return SPF32_WORD_ZERO
    if f.cls & CLASS_INF:
        if f.t_guard:
            theta = ((f.theta_int + (1 << (f.t_guard - 1))) >> f.t_guard) & SPF_PHASE_MASK
        else:
            theta = f.theta_int & SPF_PHASE_MASK
        return ((L_CODE_INF & 0xFFFF) << 16) | theta
    return pack_spf32(field_to_spf(f))


def decode_field(f: SPFField) -> complex:
    f = f.normalized()
    if f.cls & CLASS_ZERO:
        return 0.0 + 0.0j
    if f.cls & CLASS_NAN:
        return complex(float("nan"), float("nan"))
    if f.cls & CLASS_INF:
        theta = (f.theta_int & f.phase_mask) * TAU / f.phase_mod
        return complex(float("inf") * math.cos(theta), float("inf") * math.sin(theta))
    r = 2.0 ** (f.L_int / float(f.l_scale))
    theta = (f.theta_int & f.phase_mask) * TAU / f.phase_mod
    return complex(r * math.cos(theta), r * math.sin(theta))


def encode_field(z: complex, *, l_guard: int = 2, t_guard: int = 2) -> SPFField:
    mag = abs(z)
    if mag == 0.0:
        return SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
    L_int = round(math.log2(mag) * (SPF_L_SCALE << l_guard))
    phase_mod = 1 << (16 + t_guard)
    theta = round(math.atan2(z.imag, z.real) * phase_mod / TAU) & (phase_mod - 1)
    return SPFField(L_int, theta, CLASS_NORMAL, l_guard, t_guard).normalized()


def field_neg(f: SPFField) -> SPFField:
    f = f.normalized()
    if f.cls & (CLASS_ZERO | CLASS_NAN):
        return f
    return SPFField(
        f.L_int, (f.theta_int + (f.phase_mod >> 1)) & f.phase_mask,
        f.cls, f.l_guard, f.t_guard,
    )


def field_twiddle(f: SPFField, twiddle_theta: int) -> SPFField:
    """Multiply by unit twiddle.

    twiddle_theta is in the same internal phase scale as f.theta_int.
    """
    f = f.normalized()
    if f.cls & (CLASS_ZERO | CLASS_NAN):
        return f
    return SPFField(
        f.L_int, (f.theta_int + int(twiddle_theta)) & f.phase_mask,
        f.cls, f.l_guard, f.t_guard,
    )


def phase_delta_signed(theta_b: int, theta_a: int, phase_bits: int) -> int:
    mask = (1 << phase_bits) - 1
    half = 1 << (phase_bits - 1)
    return ((int(theta_b) - int(theta_a) + half) & mask) - half


def _sincos_halfangle_guarded(delta: int, phase_bits: int) -> tuple[float, float, float]:
    """Return h_rad, sin(h), cos(h) for h=delta/2 with phase guard bits."""
    phase_mod = 1 << phase_bits
    half = phase_mod >> 1
    quarter = phase_mod >> 2
    if delta == 0:
        return 0.0, 0.0, 1.0
    if delta == -half:
        return -math.pi / 2.0, -1.0, 0.0
    if delta == quarter:
        s = math.sqrt(0.5)
        return math.pi / 4.0, s, s
    if delta == -quarter:
        s = math.sqrt(0.5)
        return -math.pi / 4.0, -s, s

    h = 0.5 * float(delta) * TAU / float(phase_mod)
    ah = abs(h)
    s = sin_poly7_first_quadrant(ah)
    c = sin_poly7_first_quadrant((math.pi / 2.0) - ah)
    if h < 0.0:
        s = -s
    return h, s, c


def _half_phase_ext_from_delta(delta: int) -> int:
    return int(delta)


def _theta_base_ext(theta: int, h_ext: int) -> int:
    return (int(theta) << 1) + int(h_ext)


def _round_phase_ext(theta_ext: float, phase_bits: int) -> int:
    return round(theta_ext * 0.5) & ((1 << phase_bits) - 1)


def _local_atan_signed(y: float, x: float, *, accurate_finish: bool) -> float:
    if x == 0.0:
        if y == 0.0:
            return 0.0
        return math.pi / 2.0 if y > 0.0 else -math.pi / 2.0
    z = abs(y) / x
    a = math.atan(z) if accurate_finish else atan_approx_yal(z)
    return a if y >= 0.0 else -a


def _finish_field(
    local_re: float, local_im: float, L_base: int, theta_base_rad: float, *,
    l_guard: int, t_guard: int, accurate_finish: bool = True,
) -> SPFField:
    inner = local_re * local_re + local_im * local_im
    if inner <= 2.0 ** -120:
        return SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
    dL = 0.5 * math.log2(inner)
    dtheta = math.atan2(local_im, local_re) if accurate_finish else atan2_approx_yal(local_im, local_re)
    L_out = round(float(L_base) + dL * float(SPF_L_SCALE << l_guard))
    phase_mod = 1 << (16 + t_guard)
    theta_out = round((theta_base_rad + dtheta) * phase_mod / TAU) & (phase_mod - 1)
    return SPFField(L_out, theta_out, CLASS_NORMAL, l_guard, t_guard).normalized()


def _finish_field_power(
    local_re: float,
    local_im: float,
    power: float,
    L_base: int,
    theta_base_ext: int,
    *,
    l_guard: int,
    t_guard: int,
    accurate_finish: bool = True,
) -> SPFField:
    if power <= 2.0 ** -120:
        return SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
    dL = 0.5 * math.log2(power)
    dtheta = _local_atan_signed(local_im, local_re, accurate_finish=accurate_finish)
    L_out = round(float(L_base) + dL * float(SPF_L_SCALE << l_guard))
    phase_bits = 16 + t_guard
    phase_ext_mod = 1 << (phase_bits + 1)
    theta_ext = float(theta_base_ext) + dtheta * phase_ext_mod / TAU
    theta_out = _round_phase_ext(theta_ext, phase_bits)
    return SPFField(L_out, theta_out, CLASS_NORMAL, l_guard, t_guard).normalized()


def _axis_finish_field(
    L_base: int,
    theta: int,
    log2_factor: float,
    *,
    l_guard: int,
    t_guard: int,
) -> SPFField:
    if not math.isfinite(log2_factor):
        return SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
    L_out = round(float(L_base) + log2_factor * float(SPF_L_SCALE << l_guard))
    return SPFField(L_out, theta, CLASS_NORMAL, l_guard, t_guard).normalized()


def butterfly_fields_fb7(
    u: SPFField,
    v: SPFField,
    twiddle_theta: int = 0,
    *,
    dominant_gap_q78: int | None = 12 * SPF_L_SCALE,
    accurate_finish: bool = True,
) -> tuple[SPFField, SPFField]:
    """Fused FB7 butterfly on unpacked internal fields.

    Returns (u + Wv, u - Wv). The sign correction is handled when the
    larger local-frame operand is Wv instead of u.
    """
    u = u.normalized()
    vt = field_twiddle(v, twiddle_theta).normalized()

    l_guard = u.l_guard
    t_guard = u.t_guard
    if vt.l_guard != l_guard or vt.t_guard != t_guard:
        raise ValueError("u and v must use same guard configuration")

    if (u.cls | vt.cls) & CLASS_NAN:
        n = SPFField(0, 0, CLASS_NAN, l_guard, t_guard)
        return n, n
    if (u.cls | vt.cls) & CLASS_INF:
        # Experimental path: delegate infinities to the stable scalar path later.
        n = SPFField(0, 0, CLASS_NAN, l_guard, t_guard)
        return n, n
    if (u.cls & CLASS_ZERO) and (vt.cls & CLASS_ZERO):
        z = SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
        return z, z
    if vt.cls & CLASS_ZERO:
        return u.copy(), u.copy()
    if u.cls & CLASS_ZERO:
        return vt.copy(), field_neg(vt)

    a_is_u = u.L_int >= vt.L_int
    A = u if a_is_u else vt
    B = vt if a_is_u else u

    gap_int = A.L_int - B.L_int
    gap_q78 = _round_shift_signed(gap_int, l_guard)

    phase_bits = 16 + t_guard
    dtheta = phase_delta_signed(B.theta_int, A.theta_int, phase_bits)
    raw_delta = (B.theta_int - A.theta_int) & ((1 << phase_bits) - 1)
    phase_mod = 1 << phase_bits
    half_phase = phase_mod >> 1

    # q = |B|/|A| = 2^(-gap)
    q = 2.0 ** (-(gap_int / float(SPF_L_SCALE << l_guard)))
    if raw_delta == 0:
        plus = _axis_finish_field(
            A.L_int, A.theta_int, math.log1p(q) / math.log(2.0),
            l_guard=l_guard, t_guard=t_guard,
        )
        if q >= 1.0:
            minus_ab = SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
        else:
            minus_ab = _axis_finish_field(
                A.L_int, A.theta_int, math.log1p(-q) / math.log(2.0),
                l_guard=l_guard, t_guard=t_guard,
            )
        minus = minus_ab if a_is_u else field_neg(minus_ab)
        return plus, minus
    if raw_delta == half_phase:
        if q >= 1.0:
            plus = SPFField(L_CODE_ZERO << l_guard, 0, CLASS_ZERO, l_guard, t_guard)
        else:
            plus = _axis_finish_field(
                A.L_int, A.theta_int, math.log1p(-q) / math.log(2.0),
                l_guard=l_guard, t_guard=t_guard,
            )
        minus_ab = _axis_finish_field(
            A.L_int, A.theta_int, math.log1p(q) / math.log(2.0),
            l_guard=l_guard, t_guard=t_guard,
        )
        minus = minus_ab if a_is_u else field_neg(minus_ab)
        return plus, minus

    # Dominant bypass: plus ≈ A and A-B ≈ A. If A is vt, u-vt = -(A-B).
    if dominant_gap_q78 is not None and gap_q78 >= dominant_gap_q78:
        plus = A.copy()
        minus = A.copy() if a_is_u else field_neg(A)
        return plus, minus

    h_rad, sh, ch = _sincos_halfangle_guarded(dtheta, phase_bits)
    del h_rad
    h_ext = _half_phase_ext_from_delta(dtheta)
    theta_base_ext = _theta_base_ext(A.theta_int, h_ext)

    if gap_int == 0:
        plus = _finish_field_power(
            2.0 * ch, 0.0, 4.0 * ch * ch, A.L_int, theta_base_ext,
            l_guard=l_guard, t_guard=t_guard, accurate_finish=accurate_finish,
        )
        minus_ab = _finish_field_power(
            0.0, -2.0 * sh, 4.0 * sh * sh, A.L_int, theta_base_ext,
            l_guard=l_guard, t_guard=t_guard, accurate_finish=accurate_finish,
        )
        minus = minus_ab if a_is_u else field_neg(minus_ab)
        return plus, minus

    p = 1.0 + q
    m = 1.0 - q
    cos_delta = ch * ch - sh * sh
    q2 = q * q
    qc = q * cos_delta
    plus_power = 1.0 + q2 + 2.0 * qc
    minus_power = 1.0 + q2 - 2.0 * qc

    plus = _finish_field_power(
        p * ch, -m * sh, plus_power, A.L_int, theta_base_ext,
        l_guard=l_guard, t_guard=t_guard, accurate_finish=accurate_finish,
    )
    minus_ab = _finish_field_power(
        m * ch, -p * sh, minus_power, A.L_int, theta_base_ext,
        l_guard=l_guard, t_guard=t_guard, accurate_finish=accurate_finish,
    )
    minus = minus_ab if a_is_u else field_neg(minus_ab)
    return plus, minus


def _bit_reverse(i: int, bits: int) -> int:
    out = 0
    for _ in range(bits):
        out = (out << 1) | (i & 1)
        i >>= 1
    return out


def fft_fb7_fields(
    values: Sequence[SPF | int | complex],
    *,
    inverse: bool = False,
    l_guard: int = 2,
    t_guard: int = 2,
    dominant_gap_q78: int | None = 12 * SPF_L_SCALE,
    accurate_finish: bool = True,
    return_words: bool = False,
) -> list[SPFField] | list[int]:
    """Radix-2 FFT using field-mode FB7 butterflies.

    Accepts SPF objects, packed words, or complex numbers. The transform is
    unscaled, matching numpy.fft.fft convention for forward FFT.
    """
    n = len(values)
    if n == 0 or (n & (n - 1)):
        raise ValueError("length must be a power of two")
    logn = int(math.log2(n))

    fields: list[SPFField] = []
    for x in values:
        if isinstance(x, SPFField):
            fields.append(x.normalized())
        elif isinstance(x, SPF):
            fields.append(field_from_spf(x, l_guard=l_guard, t_guard=t_guard))
        elif isinstance(x, int) or hasattr(x, '__index__'):
            fields.append(field_from_word(operator.index(x), l_guard=l_guard, t_guard=t_guard))
        else:
            fields.append(encode_field(complex(x), l_guard=l_guard, t_guard=t_guard))

    buf = [fields[_bit_reverse(i, logn)].copy() for i in range(n)]
    phase_mod = 1 << (16 + t_guard)
    sign = 1 if inverse else -1

    m = 2
    while m <= n:
        half = m // 2
        step = phase_mod // m
        for block in range(0, n, m):
            for j in range(half):
                tw = (sign * j * step) & (phase_mod - 1)
                out0, out1 = butterfly_fields_fb7(
                    buf[block + j],
                    buf[block + j + half],
                    tw,
                    dominant_gap_q78=dominant_gap_q78,
                    accurate_finish=accurate_finish,
                )
                buf[block + j] = out0
                buf[block + j + half] = out1
        m *= 2

    if inverse:
        # Divide by n: subtract log2(n) from amplitude logs.
        dL = logn * (SPF_L_SCALE << l_guard)
        for f in buf:
            if f.cls == CLASS_NORMAL:
                f.L_int -= dL

    if return_words:
        return [field_to_word(f) for f in buf]
    return buf


def fft_reference(values: Sequence[SPF | int | complex], *, inverse: bool = False) -> list[SPF]:
    import numpy as np

    z = []
    for x in values:
        if isinstance(x, SPF):
            z.append(decode_complex(x))
        elif isinstance(x, int):
            z.append(decode_complex(unpack_spf32(x)))
        else:
            z.append(complex(x))
    y = np.fft.ifft(z) if inverse else np.fft.fft(z)
    return [encode_complex(complex(v)) for v in y]


def evm_vs_reference(values: Sequence[SPF | int | complex], **kwargs: Any) -> float:
    ref = fft_reference(values, inverse=kwargs.get("inverse", False))
    got = fft_fb7_fields(values, **kwargs)
    if got and isinstance(got[0], int):
        got_c = [decode_complex(unpack_spf32(w)) for w in got]  # type: ignore[arg-type,unused-ignore]
    else:
        got_c = [decode_field(f) for f in got]  # type: ignore[arg-type]
    ref_c = [decode_complex(x) for x in ref]
    num = sum(abs(a - b) ** 2 for a, b in zip(got_c, ref_c, strict=False))
    den = sum(abs(b) ** 2 for b in ref_c) or 1.0
    return math.sqrt(num / den)


__all__ = [
    "SPFField",
    "butterfly_fields_fb7",
    "decode_field",
    "encode_field",
    "evm_vs_reference",
    "fft_fb7_fields",
    "fft_reference",
    "field_from_spf",
    "field_from_word",
    "field_neg",
    "field_to_spf",
    "field_to_word",
    "field_twiddle",
]
