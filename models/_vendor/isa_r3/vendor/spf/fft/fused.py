"""
SPF-FB7: fused half-angle radix-2 butterfly and shared fast SPF32 helpers.

Stable path: src/spf/fft/fused.py
Promoted from src/spf_experimental/fft/fused_butterfly.py in Phase 2
(SPFv2 promotion) once HLS csim, synthesis, board tests, and end-to-end
OFDM/radar tolerances passed.

Representation used here matches the current native/HLS SPF32 ABI:
    word = [signed int16 L_q78][uint16 theta]
    L_q78 = round(256 * log2(|z|))
    theta = round(65536 * angle(z)/(2*pi)) mod 65536
    ZERO = 0x80000000, NAN = 0x7fffffff, INF = L == 0x7fff and theta != 0xffff

Key design rule:
    pack/unpack is just bitfield movement and should remain exact.
    encode/decode are nonlinear conversion operations; they can reuse the same
    table-free polynomial sin/cos, atan2, log2, and exp2 units as FB7 for a
    future Cartesian-I/O overlay, but they should not be mixed into FFT kernels
    unless the overlay genuinely ingests/exports Cartesian I/Q.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

SPF_L_FRAC_BITS = 8
SPF_L_SCALE = 1 << SPF_L_FRAC_BITS
SPF_PHASE_BITS = 16
SPF_PHASE_MOD = 1 << SPF_PHASE_BITS
SPF_PHASE_MASK = SPF_PHASE_MOD - 1
SPF_PHASE_HALF = 1 << (SPF_PHASE_BITS - 1)
SPF_PHASE_QUARTER = 1 << (SPF_PHASE_BITS - 2)
SPF_PHASE_EXT_BITS = SPF_PHASE_BITS + 1
SPF_PHASE_EXT_MOD = 1 << SPF_PHASE_EXT_BITS

SPF32_WORD_ZERO = 0x80000000
SPF32_WORD_NAN = 0x7FFFFFFF
L_CODE_ZERO = -0x8000
L_CODE_INF = 0x7FFF
L_CODE_MAX_FINITE = 0x7FFE
L_CODE_MIN_FINITE = -0x7FFF

TAU = 2.0 * math.pi
PHASE_TO_RAD = TAU / SPF_PHASE_MOD
RAD_TO_PHASE = SPF_PHASE_MOD / TAU
PHASE_EXT_TO_RAD = TAU / SPF_PHASE_EXT_MOD
RAD_TO_PHASE_EXT = SPF_PHASE_EXT_MOD / TAU
INNER_ZERO_EPS = 2.0 ** -120

DOMINANT_GAP_FAST_Q78 = 8 * SPF_L_SCALE
DOMINANT_GAP_BALANCED_Q78 = 12 * SPF_L_SCALE
DOMINANT_GAP_ACCURATE_Q78: int | None = None


@dataclass(frozen=True, slots=True)
class SPF:
    """Unpacked SPF32 scalar, finite/zero-focused for experimental kernels."""
    L_q78: int
    theta_u16: int

    def normalized(self) -> SPF:
        if self.L_q78 <= L_CODE_ZERO:
            return SPF(L_CODE_ZERO, 0)
        return SPF(clamp_l_q78(self.L_q78), self.theta_u16 & SPF_PHASE_MASK)


def clamp_l_q78(L_q78: int) -> int:
    if L_q78 <= L_CODE_ZERO:
        return L_CODE_ZERO
    return max(L_CODE_MIN_FINITE, min(L_CODE_MAX_FINITE, int(L_q78)))


def wrap_phase_u16(theta: int) -> int:
    return int(theta) & SPF_PHASE_MASK


def phase_delta_i16(theta_b: int, theta_a: int) -> int:
    """Signed BAM delta theta_b - theta_a in [-32768, 32767]."""
    return ((int(theta_b) - int(theta_a) + SPF_PHASE_HALF) & SPF_PHASE_MASK) - SPF_PHASE_HALF


def quantize_phase_from_rad(theta_rad: float) -> int:
    return round(theta_rad * RAD_TO_PHASE) & SPF_PHASE_MASK


# ---------------------------------------------------------------------------
# Exact pack/unpack and vector bitfield helpers
# ---------------------------------------------------------------------------

def pack_spf32(x: SPF) -> int:
    """Exact SPF32 bitfield pack. No approximation belongs here."""
    x = x.normalized()
    return ((x.L_q78 & 0xFFFF) << 16) | (x.theta_u16 & 0xFFFF)


def unpack_spf32(w: int) -> SPF:
    """Exact SPF32 bitfield unpack."""
    raw_L = (int(w) >> 16) & 0xFFFF
    L = raw_L - 0x10000 if raw_L & 0x8000 else raw_L
    theta = int(w) & 0xFFFF
    return SPF(L, theta).normalized()


def pack_words(xs: Iterable[SPF]) -> list[int]:
    return [pack_spf32(x) for x in xs]


def unpack_words(ws: Iterable[int]) -> list[SPF]:
    return [unpack_spf32(w) for w in ws]


# ---------------------------------------------------------------------------
# Shared table-free approximation units
# ---------------------------------------------------------------------------

def sin_poly7_first_quadrant(x: float) -> float:
    """Constrained odd degree-7 sine approximation on [0, pi/2].

    Odd form gives sin(0)=0 exactly, which is critical near cancellation.
    The coefficients are fixed-point-friendly and accurate enough for the
    FB7 half-angle path; final hardware coefficients should be re-tuned after
    fixed-point quantization.
    """
    if x <= 0.0:
        return 0.0
    if x >= math.pi / 2.0:
        return 1.0
    t = x * x
    return x * (
        0.999996936
        + t * (-0.166649224 + t * (0.0083069279 + t * (-0.000183722708)))
    )


def sincos_bam_poly7(theta_u16: int) -> tuple[float, float]:
    """Table-free sin/cos for a full 16-bit BAM phase.

    Uses quadrant folding and the same degree-7 first-quadrant sine polynomial
    as FB7. This is for fast decode/Cartesian export overlays, not for the
    FB7 butterfly itself, which only needs a half-angle in [-pi/2, pi/2].
    """
    t = int(theta_u16) & SPF_PHASE_MASK

    # Exact axis snaps protect DC, Nyquist, and pure tones.
    if t == 0:
        return 0.0, 1.0
    if t == SPF_PHASE_QUARTER:
        return 1.0, 0.0
    if t == SPF_PHASE_HALF:
        return 0.0, -1.0
    if t == (SPF_PHASE_HALF + SPF_PHASE_QUARTER):
        return -1.0, 0.0

    quad = t >> 14
    rem = t & 0x3FFF
    x = (rem / SPF_PHASE_QUARTER) * (math.pi / 2.0)
    sq = sin_poly7_first_quadrant(x)
    cq = sin_poly7_first_quadrant((math.pi / 2.0) - x)

    if quad == 0:
        return sq, cq
    if quad == 1:
        return cq, -sq
    if quad == 2:
        return -sq, -cq
    return -cq, sq


def sincos_halfangle_poly7(delta_i16: int) -> tuple[float, float, float]:
    """Return (h_rad, sin(h), cos(h)) for h = delta/2.

    delta is signed u16/BAM ticks in [-32768, 32767], so h lies in roughly
    [-pi/2, pi/2]. That bounded half-angle is the reason FB7 is better than
    scalar local-frame add inside FFT.
    """
    if delta_i16 == 0:
        return 0.0, 0.0, 1.0
    if delta_i16 == -SPF_PHASE_HALF:
        return -math.pi / 2.0, -1.0, 0.0
    if delta_i16 == SPF_PHASE_QUARTER:
        h = math.pi / 4.0
        s = math.sqrt(0.5)
        return h, s, s
    if delta_i16 == -SPF_PHASE_QUARTER:
        h = -math.pi / 4.0
        s = math.sqrt(0.5)
        return h, -s, s

    h = 0.5 * float(delta_i16) * PHASE_TO_RAD
    ah = abs(h)
    s = sin_poly7_first_quadrant(ah)
    c = sin_poly7_first_quadrant((math.pi / 2.0) - ah)
    if h < 0.0:
        s = -s
    return h, s, c


def atan2_approx_yal(y: float, x: float) -> float:
    """Table-free atan2 approximation with octant folding.

    This is suitable for approximation studies. The current acceptance backend
    keeps exact atan2/log2 in the finish path because rough atan2 can dominate
    the poly-preview error budget.
    """
    if x == 0.0 and y == 0.0:
        return 0.0

    def atan_approx(z: float) -> float:
        az = abs(z)
        return (math.pi / 4.0) * z - z * (az - 1.0) * (0.2447 + 0.0663 * az)

    ax, ay = abs(x), abs(y)
    if ax >= ay:
        z = y / x
        a = atan_approx(z)
        if x < 0.0:
            a += math.pi if y >= 0.0 else -math.pi
        return a
    z = x / y
    return (math.pi / 2.0 if y >= 0.0 else -math.pi / 2.0) - atan_approx(z)


def atan2_poly_degree4(y: float, x: float) -> float:
    """§10.3 degree-4 atan polynomial matching fb7_atan_poly_degree4 in C.

    Replaces atan2_approx_yal in encode_complex_fast for better phase accuracy.
    """
    if x == 0.0 and y == 0.0:
        return 0.0

    def atan_poly(z: float) -> float:
        t = z * z
        return z * (0.9999374 + t * (-0.33130809
                    + t * (0.18405074 + t * (-0.09074328 + t * 0.02350681))))

    ax, ay = abs(x), abs(y)
    if ax >= ay:
        z = y / x
        a = atan_poly(z)
        if x < 0.0:
            a += math.pi if y >= 0.0 else -math.pi
        return a
    z = x / y
    return (math.pi / 2.0 if y >= 0.0 else -math.pi / 2.0) - atan_poly(z)


def log2_poly_pos(x: float) -> float:
    """Table-free positive log2 using frexp + atanh polynomial.

    This mirrors the hardware leading-one + mantissa-poly path. It is much
    more accurate than a linear Quake-style mantissa reinterpret.
    """
    if x <= 0.0:
        return -math.inf
    m, e = math.frexp(x)  # x = m * 2**e, m in [0.5, 1)
    m *= 2.0
    e -= 1
    if m > math.sqrt(2.0):
        m *= 0.5
        e += 1
    y = (m - 1.0) / (m + 1.0)
    y2 = y * y
    frac = (2.0 / math.log(2.0)) * y * (
        1.0 + y2 * (1.0 / 3.0 + y2 * (1.0 / 5.0 + y2 * (1.0 / 7.0)))
    )
    return e + frac


def exp2_poly(x: float) -> float:
    """Table-free exp2 split into integer shift + fractional polynomial."""
    if x < -64.0:
        return 0.0
    if x > 64.0:
        return math.inf
    n = math.floor(x)
    f = x - n
    ln2 = math.log(2.0)
    f2 = f * f
    p = (1.0 + ln2 * f + (ln2 * ln2 * 0.5) * f2
         + (ln2 ** 3 / 6.0) * f2 * f + (ln2 ** 4 / 24.0) * f2 * f2)
    return math.ldexp(p, int(n))


# ---------------------------------------------------------------------------
# Encode/decode reference and fast approximation paths
# ---------------------------------------------------------------------------

def decode_complex_reference(x: SPF) -> complex:
    x = x.normalized()
    if x.L_q78 == L_CODE_ZERO:
        return 0.0 + 0.0j
    r = 2.0 ** (x.L_q78 / SPF_L_SCALE)
    theta = x.theta_u16 * PHASE_TO_RAD
    return complex(r * math.cos(theta), r * math.sin(theta))


def decode_complex_fast(x: SPF) -> complex:
    """Approximate decode using shared FB7 table-free exp2 and BAM sin/cos."""
    x = x.normalized()
    if x.L_q78 == L_CODE_ZERO:
        return 0.0 + 0.0j
    r = exp2_poly(x.L_q78 / SPF_L_SCALE)
    s, c = sincos_bam_poly7(x.theta_u16)
    return complex(r * c, r * s)


def encode_complex_reference(z: complex) -> SPF:
    mag = abs(z)
    if mag == 0.0:
        return SPF(L_CODE_ZERO, 0)
    L_q78 = round(math.log2(mag) * SPF_L_SCALE)
    if L_q78 <= L_CODE_ZERO:
        return SPF(L_CODE_ZERO, 0)
    theta_u16 = quantize_phase_from_rad(math.atan2(z.imag, z.real))
    return SPF(clamp_l_q78(L_q78), theta_u16)


def encode_complex_fast(z: complex) -> SPF:
    """Approximate encode using hypot^2 + log2_poly and atan2_approx.

    This belongs in conversion/Cartesian-I/O overlays. FFT kernels that already
    consume SPF should not decode and re-encode between butterflies.
    """
    re = float(z.real)
    im = float(z.imag)
    r2 = re * re + im * im
    if r2 == 0.0:
        return SPF(L_CODE_ZERO, 0)
    L_q78 = round(0.5 * log2_poly_pos(r2) * SPF_L_SCALE)
    if L_q78 <= L_CODE_ZERO:
        return SPF(L_CODE_ZERO, 0)
    theta_u16 = quantize_phase_from_rad(atan2_poly_degree4(im, re))
    return SPF(clamp_l_q78(L_q78), theta_u16)


# Backward-compatible aliases from the first patchkit.
decode_complex = decode_complex_reference
encode_complex = encode_complex_reference


def decode_vector(xs: Iterable[SPF], *, fast: bool = False) -> list[complex]:
    fn = decode_complex_fast if fast else decode_complex_reference
    return [fn(x) for x in xs]


def encode_vector(zs: Iterable[complex], *, fast: bool = False) -> list[SPF]:
    fn = encode_complex_fast if fast else encode_complex_reference
    return [fn(complex(z)) for z in zs]


# ---------------------------------------------------------------------------
# SPF native ops and FB7 butterfly
# ---------------------------------------------------------------------------

def spf_neg(x: SPF) -> SPF:
    x = x.normalized()
    if x.L_q78 == L_CODE_ZERO:
        return x
    return SPF(x.L_q78, wrap_phase_u16(x.theta_u16 ^ SPF_PHASE_HALF))


def spf_mul_unit_twiddle(x: SPF, twiddle_theta_u16: int) -> SPF:
    x = x.normalized()
    if x.L_q78 == L_CODE_ZERO:
        return x
    return SPF(x.L_q78, wrap_phase_u16(x.theta_u16 + twiddle_theta_u16))


def _finish_local(
    local_re: float,
    local_im: float,
    L_base_q78: int,
    theta_base_rad: float,
    *,
    log2_fn: Callable[[float], float],
    atan2_fn: Callable[[float, float], float],
) -> SPF:
    inner = local_re * local_re + local_im * local_im
    if inner <= INNER_ZERO_EPS:
        return SPF(L_CODE_ZERO, 0)
    dL_q78 = round(0.5 * log2_fn(inner) * SPF_L_SCALE)
    dtheta = atan2_fn(local_im, local_re)
    return SPF(
        clamp_l_q78(L_base_q78 + dL_q78),
        wrap_phase_u16(quantize_phase_from_rad(theta_base_rad + dtheta)),
    )


def atan_approx_yal(z: float) -> float:
    """Table-free atan approximation used by the polynomial FB7 finish."""
    if z > 1.0:
        return math.pi / 2.0 - atan_approx_yal(1.0 / z)
    if z < -1.0:
        return -math.pi / 2.0 - atan_approx_yal(1.0 / z)
    az = abs(z)
    return (math.pi / 4.0) * z - z * (az - 1.0) * (0.2447 + 0.0663 * az)


def _atan_from_atan2_fn(
    atan2_fn: Callable[[float, float], float],
) -> Callable[[float], float]:
    if atan2_fn is math.atan2:
        return math.atan
    if atan2_fn is atan2_approx_yal:
        return atan_approx_yal
    return lambda z: atan2_fn(z, 1.0)


def half_phase_ext_from_delta(delta_i16: int) -> int:
    """Exact half-angle in W+1 BAM ticks for a signed W-bit delta."""
    return int(delta_i16)


def theta_base_ext(theta_u16: int, h_ext: int) -> int:
    """Dominant phase plus exact half-angle in W+1 BAM ticks."""
    return ((int(theta_u16) & SPF_PHASE_MASK) << 1) + int(h_ext)


def round_phase_ext_to_u16(theta_ext: float) -> int:
    """Round a W+1 phase accumulator to the public W-bit phase boundary."""
    return round(theta_ext * 0.5) & SPF_PHASE_MASK


def _local_atan_signed(
    y: float,
    x: float,
    *,
    atan_fn: Callable[[float], float],
) -> float:
    """Local FB7 phase correction for the half-angle frame where x >= 0."""
    if x == 0.0:
        if y == 0.0:
            return 0.0
        return math.pi / 2.0 if y > 0.0 else -math.pi / 2.0
    a = atan_fn(abs(y) / x)
    return a if y >= 0.0 else -a


def _finish_local_fb7(
    local_re: float,
    local_im: float,
    power: float,
    L_base_q78: int,
    theta_base_ext_ticks: int,
    *,
    log2_fn: Callable[[float], float],
    atan_fn: Callable[[float], float],
) -> SPF:
    if power <= INNER_ZERO_EPS:
        return SPF(L_CODE_ZERO, 0)
    dL_q78 = round(0.5 * log2_fn(power) * SPF_L_SCALE)
    dtheta = _local_atan_signed(local_im, local_re, atan_fn=atan_fn)
    theta_ext = float(theta_base_ext_ticks) + dtheta * RAD_TO_PHASE_EXT
    return SPF(
        clamp_l_q78(L_base_q78 + dL_q78),
        round_phase_ext_to_u16(theta_ext),
    )


def _axis_finish(
    L_base_q78: int,
    theta_u16: int,
    log2_factor: float,
) -> SPF:
    if not math.isfinite(log2_factor):
        return SPF(L_CODE_ZERO, 0)
    dL_q78 = round(log2_factor * SPF_L_SCALE)
    return SPF(clamp_l_q78(L_base_q78 + dL_q78), theta_u16)


def spf_add_reference(a: SPF, b: SPF) -> SPF:
    return encode_complex_reference(decode_complex_reference(a) + decode_complex_reference(b))


def spf_butterfly_reference(u: SPF, v: SPF, twiddle_theta_u16: int = 0) -> tuple[SPF, SPF]:
    vt = spf_mul_unit_twiddle(v, twiddle_theta_u16)
    return spf_add_reference(u, vt), spf_add_reference(u, spf_neg(vt))


def spf_butterfly_fused(
    u: SPF,
    v: SPF,
    twiddle_theta_u16: int = 0,
    *,
    sincos_halfangle_fn: Callable[[int], tuple[float, float, float]] = sincos_halfangle_poly7,
    log2_fn: Callable[[float], float] = log2_poly_pos,
    atan2_fn: Callable[[float, float], float] = atan2_poly_degree4,
    atan_fn: Callable[[float], float] | None = None,
    exp2_fn: Callable[[float], float] = exp2_poly,
    dominant_gap_q78: int | None = DOMINANT_GAP_BALANCED_Q78,
) -> tuple[SPF, SPF]:
    """Compute (u + twiddle*v, u - twiddle*v) with one shared local frame."""
    u = u.normalized()
    vt = spf_mul_unit_twiddle(v, twiddle_theta_u16).normalized()

    if u.L_q78 == L_CODE_ZERO and vt.L_q78 == L_CODE_ZERO:
        return SPF(L_CODE_ZERO, 0), SPF(L_CODE_ZERO, 0)
    if vt.L_q78 == L_CODE_ZERO:
        return u, u
    if u.L_q78 == L_CODE_ZERO:
        return vt, spf_neg(vt)

    a_is_u = u.L_q78 >= vt.L_q78
    A, B = (u, vt) if a_is_u else (vt, u)

    gap_q78 = A.L_q78 - B.L_q78
    raw_delta = (B.theta_u16 - A.theta_u16) & SPF_PHASE_MASK
    dtheta = phase_delta_i16(B.theta_u16, A.theta_u16)
    if atan_fn is None:
        atan_fn = _atan_from_atan2_fn(atan2_fn)

    if raw_delta == 0:
        q = exp2_fn(-gap_q78 / SPF_L_SCALE)
        plus = _axis_finish(A.L_q78, A.theta_u16, log2_fn(1.0 + q))
        minus_ab = SPF(L_CODE_ZERO, 0) if q >= 1.0 else _axis_finish(A.L_q78, A.theta_u16, log2_fn(1.0 - q))
    elif raw_delta == SPF_PHASE_HALF:
        q = exp2_fn(-gap_q78 / SPF_L_SCALE)
        plus = SPF(L_CODE_ZERO, 0) if q >= 1.0 else _axis_finish(A.L_q78, A.theta_u16, log2_fn(1.0 - q))
        minus_ab = _axis_finish(A.L_q78, A.theta_u16, log2_fn(1.0 + q))
    elif gap_q78 == 0:
        h_ext = half_phase_ext_from_delta(dtheta)
        _, sh, ch = sincos_halfangle_fn(dtheta)
        base_ext = theta_base_ext(A.theta_u16, h_ext)
        plus = _finish_local_fb7(
            2.0 * ch, 0.0, 4.0 * ch * ch, A.L_q78, base_ext,
            log2_fn=log2_fn, atan_fn=atan_fn,
        )
        minus_ab = _finish_local_fb7(
            0.0, -2.0 * sh, 4.0 * sh * sh, A.L_q78, base_ext,
            log2_fn=log2_fn, atan_fn=atan_fn,
        )
    elif dominant_gap_q78 is not None and gap_q78 >= dominant_gap_q78:
        # Fast/balanced mode: skip the smaller vector when it is far below the
        # larger one. Make the threshold a hardware register/profile knob.
        plus = A
        minus_ab = A
    else:
        q = exp2_fn(-gap_q78 / SPF_L_SCALE)
        p = 1.0 + q
        m = 1.0 - q

        h_rad, sh, ch = sincos_halfangle_fn(dtheta)
        del h_rad
        h_ext = half_phase_ext_from_delta(dtheta)
        base_ext = theta_base_ext(A.theta_u16, h_ext)

        # A+B = |A| e^(i(thetaA+h)) [(1+q)cos h - i(1-q)sin h]
        plus_re = p * ch
        plus_im = -m * sh

        # A-B = |A| e^(i(thetaA+h)) [(1-q)cos h - i(1+q)sin h]
        minus_re = m * ch
        minus_im = -p * sh
        plus_power = plus_re * plus_re + plus_im * plus_im
        minus_power = minus_re * minus_re + minus_im * minus_im

        plus = _finish_local_fb7(
            plus_re, plus_im, plus_power, A.L_q78, base_ext,
            log2_fn=log2_fn, atan_fn=atan_fn,
        )
        minus_ab = _finish_local_fb7(
            minus_re, minus_im, minus_power, A.L_q78, base_ext,
            log2_fn=log2_fn, atan_fn=atan_fn,
        )

    minus = minus_ab if a_is_u else spf_neg(minus_ab)
    return plus.normalized(), minus.normalized()


def spf_butterfly_fb7_fast(u: SPF, v: SPF, twiddle_theta_u16: int = 0) -> tuple[SPF, SPF]:
    """FAST mode: all-poly half-angle finish, G=8 dominant bypass."""
    return spf_butterfly_fused(
        u, v, twiddle_theta_u16,
        sincos_halfangle_fn=sincos_halfangle_poly7,
        log2_fn=log2_poly_pos,
        atan2_fn=atan2_poly_degree4,
        exp2_fn=exp2_poly,
        dominant_gap_q78=DOMINANT_GAP_FAST_Q78,
    )


def spf_butterfly_fb7_balanced(u: SPF, v: SPF, twiddle_theta_u16: int = 0) -> tuple[SPF, SPF]:
    """BALANCED mode: all-poly half-angle finish, G=12 dominant bypass."""
    return spf_butterfly_fused(
        u, v, twiddle_theta_u16,
        sincos_halfangle_fn=sincos_halfangle_poly7,
        log2_fn=log2_poly_pos,
        atan2_fn=atan2_poly_degree4,
        exp2_fn=exp2_poly,
        dominant_gap_q78=DOMINANT_GAP_BALANCED_Q78,
    )


def spf_butterfly_fb7_accurate(u: SPF, v: SPF, twiddle_theta_u16: int = 0) -> tuple[SPF, SPF]:
    """ACCURATE mode: all-poly finish with dominant-gap bypass disabled."""
    return spf_butterfly_fused(
        u, v, twiddle_theta_u16,
        sincos_halfangle_fn=sincos_halfangle_poly7,
        log2_fn=log2_poly_pos,
        atan2_fn=atan2_poly_degree4,
        exp2_fn=exp2_poly,
        dominant_gap_q78=None,
    )


def spf_butterfly_fb7_poly_preview(u: SPF, v: SPF, twiddle_theta_u16: int = 0) -> tuple[SPF, SPF]:
    """Backward-compatible alias for the all-poly BALANCED backend."""
    return spf_butterfly_fused(
        u, v, twiddle_theta_u16,
        sincos_halfangle_fn=sincos_halfangle_poly7,
        log2_fn=log2_poly_pos,
        atan2_fn=atan2_poly_degree4,
        exp2_fn=exp2_poly,
        dominant_gap_q78=DOMINANT_GAP_BALANCED_Q78,
    )


def bit_reverse_indices(n: int) -> list[int]:
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError("n must be a positive power of two")
    bits = int(math.log2(n))
    out: list[int] = []
    for i in range(n):
        r = 0
        x = i
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        out.append(r)
    return out


def twiddle_theta_for_stage(j: int, m: int) -> int:
    """Forward DFT twiddle phase for radix-2 DIT stage width m."""
    if m <= 0 or (m & (m - 1)) != 0:
        raise ValueError("m must be a power of two")
    return (-j * (SPF_PHASE_MOD // m)) & SPF_PHASE_MASK


def spf_fft_fused(
    xs: Sequence[SPF] | Iterable[SPF],
    *,
    bfly_fn: Callable[[SPF, SPF, int], tuple[SPF, SPF]] = spf_butterfly_fb7_balanced,
) -> list[SPF]:
    xs = list(xs)
    n = len(xs)
    if n == 0 or (n & (n - 1)) != 0:
        raise ValueError("FFT length must be a nonzero power of two")
    br = bit_reverse_indices(n)
    a = [xs[i].normalized() for i in br]
    m = 2
    while m <= n:
        half = m // 2
        for base in range(0, n, m):
            for j in range(half):
                tw = twiddle_theta_for_stage(j, m)
                out0, out1 = bfly_fn(a[base + j], a[base + j + half], tw)
                a[base + j] = out0
                a[base + j + half] = out1
        m *= 2
    return a


def spf_fft_reference(xs: Sequence[SPF] | Iterable[SPF]) -> list[SPF]:
    return spf_fft_fused(xs, bfly_fn=spf_butterfly_reference)
