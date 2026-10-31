"""REFERENCE fixed-point clocked-state executor; not a canonical SPF carrier.

A diagonal recurrence ``h_t = A_t h_{t-1} + w_t`` rewrites its state every step.
Executed in packed SPF32 words, that rewrite rounds the state at every step:
writes far below the running state are lost and per-step decays below half a log
step vanish. This module places the rounding elsewhere:

* the transition is an exact integer **clock** — prefix sums of Q16 log codes and
  of 2^32-ring phase ticks, so any span's transition product is an integer
  difference ``P_t - P_j``;
* each write enters a **frame-relative** fixed-point complex accumulator once,
  ``g = sum_j exp(P_base - P_j) w_j``, and is never rewritten;
* the frame is **rebased** once per chunk at absolute positions
  (``t % chunk == 0``), so split calls are bit-exact;
* a read ``h_t = exp(P_t - P_base) g`` is formed in fixed point, summed with the
  other states, the reverse scan and ``D v``, and rounded to one SPF32 word.

Code-to-fixed-point conversion uses integer ROM tables (the committed §8.5
exp2 ROM plus a fine table and two-level trig tables), and the final rounding is
an integer finalizer (bit length + threshold search). Tables are built once from
float/mpmath; the live path is integer-only. Coherent SPF additions: zero.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import numpy.typing as npt

from .core import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_ZERO,
    SPF32_PHASE_MASK,
    SPF32_WORD_ZERO,
    SPFError,
    pack_words,
    spf_mul_words,
    unpack_L,
    unpack_theta,
)
from .exp2_rom import EXP2_NEG_GAP_ROM_BITS
from .sequence_mixer import PhaseFeatureNetwork, PhaseMixerDirection, _resets, _words

_Q = 31  # ROM mantissa fraction bits
_HALF = np.int64(1 << (_Q - 1))
_LO_MASK = np.int64((1 << _Q) - 1)
_PHASE32 = np.int64(0xFFFFFFFF)
_ENTRY_BITS = 49  # a write enters the accumulator at <= 2^49 units
_NORM_BITS = 50  # rebase normalizes the accumulator top bit to 2^50
_SUM_GUARD = 13  # extra right shift before summing up to 4096 readout terms
_FINAL_BITS = 30  # finalizer works on <= 30-bit components
_SENTINEL_EXP = np.int64(-(1 << 40))  # exponent of an empty accumulator
_GUARD = np.int64(1 << 62)
MAX_CHUNK = 4096  # 2^50 + MAX_CHUNK * 2^49 < 2^62: one chunk of writes cannot overflow
MAX_STATES = 4096  # readout terms <= 2^62 >> 13, summed over <= 2^12 states


# ---------------------------------------------------------------- ROM tables


@dataclass(frozen=True)
class _Tables:
    exp_coarse: np.ndarray  # 2^(-k/256) at Q31, k < 256 (committed §8.5 ROM)
    exp_fine: np.ndarray  # 2^(-k/65536) at Q31, k < 256
    cos_hi: np.ndarray  # cos(2*pi*k/2^16) at Q31
    sin_hi: np.ndarray
    cos_lo: np.ndarray  # cos(2*pi*k/2^32) at Q31, k < 2^16
    sin_lo: np.ndarray
    log_thresholds: np.ndarray  # ceil(2^60 * 2^((k+0.5)/128)), k < 128
    tan_thresholds: np.ndarray  # ceil(2^31 * tan((k+0.5)*2*pi/65536)), k < 8192


@lru_cache(maxsize=1)
def _tables() -> _Tables:
    """Build every table once. Float/mpmath here only; never on the live path."""
    import mpmath

    coarse = [struct.unpack("<d", struct.pack("<Q", bits))[0] for bits in EXP2_NEG_GAP_ROM_BITS]
    exp_coarse = np.array([round(v * (1 << _Q)) for v in coarse], dtype=np.int64)
    with mpmath.workdps(40):
        exp_fine = np.array(
            [int(mpmath.nint(mpmath.mpf(2) ** (mpmath.mpf(-k) / 65536 + _Q))) for k in range(256)], dtype=np.int64
        )
        log_thr = np.array(
            [int(mpmath.ceil(mpmath.mpf(2) ** (60 + (mpmath.mpf(k) + 0.5) / 128))) for k in range(128)],
            dtype=np.int64,
        )
        tan_thr = np.array(
            [
                int(mpmath.ceil(mpmath.tan((mpmath.mpf(k) + 0.5) * 2 * mpmath.pi / 65536) * (1 << _Q)))
                for k in range(8192)
            ],
            dtype=np.int64,
        )
    hi = np.arange(1 << 16, dtype=np.float64) * (2.0 * np.pi / 65536.0)
    lo = np.arange(1 << 16, dtype=np.float64) * (2.0 * np.pi / 4294967296.0)
    scale = float(1 << _Q)
    return _Tables(
        exp_coarse,
        exp_fine,
        np.rint(np.cos(hi) * scale).astype(np.int64),
        np.rint(np.sin(hi) * scale).astype(np.int64),
        np.rint(np.cos(lo) * scale).astype(np.int64),
        np.rint(np.sin(lo) * scale).astype(np.int64),
        log_thr,
        tan_thr,
    )


# ---------------------------------------------------------------- counters


@dataclass
class ClockedStateCounters:
    """Operation counts of the declared schedule, not latency."""

    rom_reads: int = 0
    integer_multiplies: int = 0
    integer_additions: int = 0
    word_multiplications: int = 0
    finalizations: int = 0
    rebases: int = 0
    coherent_additions: int = 0


# ---------------------------------------------------------------- integer helpers


def _check_headroom(*arrays: np.ndarray) -> None:
    for arr in arrays:
        if arr.size and (np.any(arr >= _GUARD) or np.any(arr <= -_GUARD)):
            raise SPFError("clocked-state fixed-point value exceeds the 2^62 headroom guard")


def _shift(x: np.ndarray, s: np.ndarray) -> np.ndarray:
    """x * 2^-s with round-half-up for s > 0, exact left shift for s < 0."""
    x = np.asarray(x, dtype=np.int64)
    s = np.broadcast_to(np.asarray(s, dtype=np.int64), x.shape)
    right = np.clip(s, 0, 62)
    rounded = (x + np.where(right > 0, np.int64(1) << np.maximum(right - 1, 0), 0)) >> right
    left = np.clip(-s, 0, 62)
    return np.where(s >= 0, rounded, x << left)


def _mul_q31(g: np.ndarray, f: np.ndarray) -> np.ndarray:
    """round(g * f / 2^31) exactly, |g| < 2^62 and |f| <= 2^31, via a 31-bit split."""
    hi = g >> _Q
    lo = g & _LO_MASK
    return hi * f + ((lo * f + _HALF) >> _Q)


def _cmul_q31(gr, gi, fr, fi, counters: ClockedStateCounters):
    _check_headroom(gr, gi)
    counters.integer_multiplies += 8 * int(np.size(gr))
    counters.integer_additions += 2 * int(np.size(gr))
    return _mul_q31(gr, fr) - _mul_q31(gi, fi), _mul_q31(gr, fi) + _mul_q31(gi, fr)


def _bit_length(x: np.ndarray) -> np.ndarray:
    """Bit length of nonnegative int64 values by comparison search."""
    y = np.asarray(x, dtype=np.int64).copy()
    n = np.zeros(y.shape, dtype=np.int64)
    for step in (32, 16, 8, 4, 2, 1):
        big = y >= (np.int64(1) << step)
        n = np.where(big, n + step, n)
        y = np.where(big, y >> step, y)
    return n + (y > 0)


def _code_to_fixed(L_q16: np.ndarray, T_q32: np.ndarray, counters: ClockedStateCounters):
    """(L in Q16 octaves, phase on the 2^32 ring) -> (re, im) * 2^(E - 31), integer-only."""
    tab = _tables()
    L_q16 = np.asarray(L_q16, dtype=np.int64)
    T_q32 = np.asarray(T_q32, dtype=np.int64) & _PHASE32
    E = -((-L_q16) >> 16)  # ceil(L / 65536)
    gap = E * 65536 - L_q16  # [0, 65536): magnitude 2^(-gap/65536) in (0.5, 1]
    mag = (tab.exp_coarse[gap >> 8] * tab.exp_fine[gap & 255] + _HALF) >> _Q
    hi = T_q32 >> 16
    lo = T_q32 & 0xFFFF
    c1, s1, c2, s2 = tab.cos_hi[hi], tab.sin_hi[hi], tab.cos_lo[lo], tab.sin_lo[lo]
    cos = (c1 * c2 - s1 * s2 + _HALF) >> _Q
    sin = (c1 * s2 + s1 * c2 + _HALF) >> _Q
    counters.rom_reads += 6 * int(L_q16.size)
    counters.integer_multiplies += 8 * int(L_q16.size)
    counters.integer_additions += 4 * int(L_q16.size)
    return (mag * cos + _HALF) >> _Q, (mag * sin + _HALF) >> _Q, E


def _word_codes(words: np.ndarray):
    """SPF32 word -> (Q16 log code, 2^32-ring phase, zero mask) by integer shifts."""
    L = unpack_L(words).astype(np.int64)
    return L << 8, unpack_theta(words).astype(np.int64) << 16, int(SPF32_L_ZERO) == L


def _normalize(re, im, exp, top: int):
    """Shift (re, im) so the larger magnitude has ``top`` bits; empty lanes get the sentinel."""
    big = np.maximum(np.abs(re), np.abs(im))
    shift = np.where(big > 0, _bit_length(big) - top, 0)
    zero = big == 0
    return _shift(re, shift), _shift(im, shift), np.where(zero, _SENTINEL_EXP, exp + shift)


def _align_sum(parts, counters: ClockedStateCounters, axis=None):
    """Sum fixed-point parts with exponents after aligning to the largest live exponent."""
    if axis is not None:
        re, im, exp = parts
        live = (re != 0) | (im != 0)
        exp = np.where(live, exp, _SENTINEL_EXP)
        top = np.max(exp, axis=axis, keepdims=True) + _SUM_GUARD
        shift = top - exp
        counters.integer_additions += 2 * int(re.size)
        return (
            np.sum(_shift(re, shift), axis=axis),
            np.sum(_shift(im, shift), axis=axis),
            np.squeeze(top, axis=axis),
        )
    stacked = [np.stack(p) for p in zip(*parts, strict=True)]
    return _align_sum(tuple(stacked), counters, axis=0)


def finalize_words(re: npt.ArrayLike, im: npt.ArrayLike, exp: npt.ArrayLike) -> np.ndarray:
    """Round ``(re + i im) * 2^exp`` (int64 planes) to SPF32 words with integer arithmetic.

    L = RNE(256 log2|z|) and Theta = RNE(65536 arg z / 2pi) are decided by threshold
    search against exact (mpmath-built) boundary tables after reducing each component
    to 30 bits; decisions can differ from exact rounding only within ~2^-30 of a
    boundary. Magnitudes outside the finite range clamp like ``spf32.encode``.
    """
    tab = _tables()
    re = np.asarray(re, dtype=np.int64)
    im = np.asarray(im, dtype=np.int64)
    exp = np.broadcast_to(np.asarray(exp, dtype=np.int64), re.shape)
    zero = (re == 0) & (im == 0)
    re, im, exp = _normalize(re, im, exp, _FINAL_BITS)
    a, b = np.abs(re), np.abs(im)
    m = a * a + b * b  # <= 2^61
    n = _bit_length(m) - 1
    x = np.where(n >= 60, m >> np.clip(n - 60, 0, 1), m << np.clip(60 - n, 0, 62))
    k = np.searchsorted(tab.log_thresholds, x, side="right")
    L = 128 * (n + 2 * exp) + k
    L = np.clip(L, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE)
    swapped = b > a
    big, small = np.where(swapped, b, a), np.where(swapped, a, b)
    ratio = (small << _Q) // np.maximum(big, 1)
    code = np.searchsorted(tab.tan_thresholds, ratio, side="right").astype(np.int64)
    code = np.where(swapped, 16384 - code, code)
    theta = (
        np.where(
            re >= 0,
            np.where(im >= 0, code, 65536 - code),
            np.where(im >= 0, 32768 - code, 32768 + code),
        )
        & SPF32_PHASE_MASK
    )
    words = pack_words(L.astype(np.int16), theta.astype(np.uint16))
    return np.where(zero, SPF32_WORD_ZERO, words).astype(np.uint32)


def integer_clock(
    log_q16: npt.ArrayLike,
    phase_q32: npt.ArrayLike,
    *,
    reset_before: npt.ArrayLike | None = None,
    reverse: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact prefix (or suffix) sums along axis -2 of shape (..., T, N).

    Forward ``P_t = sum_{u<=t}``; reverse ``S_t = sum_{u>=t}``. A reset before t
    restarts the forward sum at t, and the reverse sum at t-1 (the boundary lies
    between t-1 and t). Log codes are Q16 octaves; phase is on the 2^32 ring.
    """
    L = np.asarray(log_q16, dtype=np.int64)
    T = np.asarray(phase_q32, dtype=np.int64) & _PHASE32
    if L.ndim < 2 or L.shape != T.shape:
        raise SPFError("integer_clock needs matching (..., T, N) log and phase planes")
    reset = _resets(reset_before, L.shape[:-1])
    if reverse:
        L, T = L[..., ::-1, :], T[..., ::-1, :]
        shifted = np.zeros_like(reset)
        shifted[..., 1:] = reset[..., ::-1][..., :-1]
        reset = shifted
    PL, PT = np.cumsum(L, axis=-2), np.cumsum(T, axis=-2)
    if np.any(reset):
        index = np.arange(L.shape[-2])
        start = np.maximum.accumulate(np.where(reset, index, 0), axis=-1)
        prior = start - 1
        take = np.maximum(prior, 0)[..., None]
        offL = np.where((prior >= 0)[..., None], np.take_along_axis(PL, np.broadcast_to(take, PL.shape), -2), 0)
        offT = np.where((prior >= 0)[..., None], np.take_along_axis(PT, np.broadcast_to(take, PT.shape), -2), 0)
        PL, PT = PL - offL, PT - offT
    PT = PT & _PHASE32
    if reverse:
        PL, PT = PL[..., ::-1, :], PT[..., ::-1, :]
    return PL, PT.astype(np.uint32)


# ---------------------------------------------------------------- parameters and state


@dataclass(frozen=True)
class ClockedDirectionParameters:
    """Transition clock codes and packed write/read planes, shape (..., T, N)."""

    log_q16: np.ndarray
    phase_q32: np.ndarray
    b: np.ndarray
    c: np.ndarray


@dataclass(frozen=True)
class ClockedStateParameters:
    forward: ClockedDirectionParameters
    backward: ClockedDirectionParameters | None = None
    diagonal: np.uint32 = SPF32_WORD_ZERO


@dataclass(frozen=True)
class ClockedState:
    """Streaming state: absolute position, clocks (..., N), accumulator (..., N, P)."""

    position: int
    chunk: int
    clock_L: np.ndarray
    clock_T: np.ndarray
    base_L: np.ndarray
    base_T: np.ndarray
    acc_re: np.ndarray
    acc_im: np.ndarray
    acc_exp: np.ndarray

    @property
    def nbytes(self) -> int:
        return sum(
            int(a.nbytes)
            for a in (self.clock_L, self.clock_T, self.base_L, self.base_T, self.acc_re, self.acc_im, self.acc_exp)
        )


@dataclass(frozen=True)
class ClockedStateResult:
    words: np.ndarray
    forward_state: ClockedState
    backward_state: ClockedState | None
    counters: ClockedStateCounters = field(default_factory=ClockedStateCounters)


def _empty_state(batch: tuple[int, ...], n: int, p: int, chunk: int) -> ClockedState:
    clock = np.zeros((*batch, n), dtype=np.int64)
    acc = np.zeros((*batch, n, p), dtype=np.int64)
    return ClockedState(
        0,
        chunk,
        clock,
        clock.copy(),
        clock.copy(),
        clock.copy(),
        acc,
        acc.copy(),
        np.full(acc.shape, _SENTINEL_EXP, dtype=np.int64),
    )


def _validate(params: ClockedStateParameters, values: np.ndarray):
    fwd = params.forward
    L = np.asarray(fwd.log_q16)
    if L.ndim < 2 or L.shape[-1] == 0:
        raise SPFError("clocked parameters need shape (..., T, N) with N >= 1")
    if L.shape[-1] > MAX_STATES:
        raise SPFError(f"clocked readout supports at most {MAX_STATES} states")
    for direction in (params.forward, params.backward):
        if direction is None:
            continue
        for name in ("log_q16", "phase_q32", "b", "c"):
            plane = np.asarray(getattr(direction, name))
            if plane.shape != L.shape:
                raise SPFError("all clocked parameter planes must share shape (..., T, N)")
        if not np.issubdtype(np.asarray(direction.log_q16).dtype, np.integer):
            raise SPFError("clock log codes must be integer Q16 octaves")
        if np.any(np.asarray(direction.log_q16) > 0):
            raise SPFError("clock transitions must not amplify magnitude (log_q16 <= 0)")
        if not np.issubdtype(np.asarray(direction.phase_q32).dtype, np.integer):
            raise SPFError("clock phases must be integer 2^32-ring ticks")
        _words(direction.b, "b", 2)
        _words(direction.c, "c", 2)
    if values.shape[:-1] != L.shape[:-1]:
        raise SPFError("values must share parameter batch/time axes, with a payload channel axis")
    diagonal = _words(params.diagonal, "diagonal")
    if diagonal.ndim != 0:
        raise SPFError("clocked diagonal must be scalar")
    return L.shape[:-2], L.shape[-2], L.shape[-1]


# ---------------------------------------------------------------- the scan


def _scan(direction: ClockedDirectionParameters, values, reset, state: ClockedState, counters):
    """One causal clocked scan. Returns per-token readout (re, im, exp) planes (..., T, P) and state."""
    chunk = state.chunk
    Lc = np.asarray(direction.log_q16, dtype=np.int64)
    Tc = np.asarray(direction.phase_q32, dtype=np.int64) & _PHASE32
    length = Lc.shape[-2]
    PL = state.clock_L[..., None, :] + np.cumsum(Lc, axis=-2)
    PT = (state.clock_T[..., None, :] + np.cumsum(Tc, axis=-2)) & _PHASE32
    counters.integer_additions += 2 * int(Lc.size)
    w = spf_mul_words(np.asarray(direction.b)[..., :, :, None], values[..., :, None, :])  # (..., T, N, P)
    counters.word_multiplications += int(w.size)
    wL, wT, wzero = _word_codes(w)
    cL, cT, czero = _word_codes(np.asarray(direction.c))
    out_re = np.zeros((*values.shape[:-2], length, values.shape[-1]), dtype=np.int64)
    out_im = np.zeros_like(out_re)
    out_exp = np.full(out_re.shape, _SENTINEL_EXP, dtype=np.int64)
    pos = state.position
    acc_re, acc_im, acc_exp = state.acc_re.copy(), state.acc_im.copy(), state.acc_exp.copy()
    base_L, base_T = state.base_L.copy(), state.base_T.copy()
    clock_L, clock_T = state.clock_L.copy(), state.clock_T.copy()
    start = 0
    while start < length:
        absolute = pos + start
        if absolute > 0 and absolute % chunk == 0:  # rebase once per chunk: g *= exp(P_{t-1} - P_base)
            dL, dT = clock_L - base_L, (clock_T - base_T) & _PHASE32
            fr, fi, fe = _code_to_fixed(dL, dT, counters)
            acc_re, acc_im = _cmul_q31(acc_re, acc_im, fr[..., None], fi[..., None], counters)
            acc_re, acc_im, acc_exp = _normalize(acc_re, acc_im, acc_exp + fe[..., None], _NORM_BITS)
            base_L, base_T = clock_L.copy(), clock_T.copy()
            counters.rebases += int(base_L.size)
        stop = min(length, start + chunk - (absolute % chunk))
        sl = slice(start, stop)
        # Writes enter frame-relative: exp(P_base - P_t) w_t.
        xL = wL[..., sl, :, :] + (base_L[..., None, :] - PL[..., sl, :])[..., None]
        xT = wT[..., sl, :, :] + ((base_T[..., None, :] - PT[..., sl, :]) & _PHASE32)[..., None]
        wre, wim, wexp = _code_to_fixed(xL, xT, counters)
        live = ~wzero[..., sl, :, :]
        need = np.where(live, wexp - _ENTRY_BITS, _SENTINEL_EXP)
        seg_reset = reset[..., sl]
        steps = stop - start
        hist_re = np.empty((steps, *acc_re.shape), dtype=np.int64)
        hist_im = np.empty_like(hist_re)
        hist_exp = np.empty_like(hist_re)
        rises = np.maximum.accumulate(np.concatenate([acc_exp[None], np.moveaxis(need, -3, 0)]), axis=0)[1:]
        fast = not np.any(seg_reset) and np.all(rises == acc_exp[None])
        if fast:  # constant exponent: exact integer prefix sums
            s = acc_exp[..., None, :, :] - wexp + _Q
            mre = np.where(live, _shift(wre, s), 0)
            mim = np.where(live, _shift(wim, s), 0)
            cre = np.cumsum(np.moveaxis(mre, -3, 0), axis=0)
            cim = np.cumsum(np.moveaxis(mim, -3, 0), axis=0)
            hist_re[0], hist_im[0] = acc_re, acc_im
            hist_re[1:] = acc_re[None] + cre[:-1]
            hist_im[1:] = acc_im[None] + cim[:-1]
            hist_exp[:] = acc_exp[None]
            acc_re, acc_im = acc_re + cre[-1], acc_im + cim[-1]
            counters.integer_additions += 2 * int(mre.size)
        else:
            for i in range(steps):
                r = seg_reset[..., i][..., None, None]
                acc_re = np.where(r, 0, acc_re)
                acc_im = np.where(r, 0, acc_im)
                acc_exp = np.where(r, _SENTINEL_EXP, acc_exp)
                hist_re[i], hist_im[i], hist_exp[i] = acc_re, acc_im, acc_exp
                lv = live[..., i, :, :]
                new_exp = np.where(lv, np.maximum(acc_exp, wexp[..., i, :, :] - _ENTRY_BITS), acc_exp)
                acc_re = _shift(acc_re, new_exp - acc_exp)
                acc_im = _shift(acc_im, new_exp - acc_exp)
                acc_exp = new_exp
                s = acc_exp - wexp[..., i, :, :] + _Q
                acc_re = acc_re + np.where(lv, _shift(wre[..., i, :, :], s), 0)
                acc_im = acc_im + np.where(lv, _shift(wim[..., i, :, :], s), 0)
                counters.integer_additions += 2 * int(lv.size)
        _check_headroom(acc_re, acc_im)
        # Reads: h_t = C_t * exp(P_t - P_base) g_{<t}, summed over states.
        rL = cL[..., sl, :] + (PL[..., sl, :] - base_L[..., None, :])
        rT = cT[..., sl, :] + ((PT[..., sl, :] - base_T[..., None, :]) & _PHASE32)
        fr, fi, fe = _code_to_fixed(rL, rT, counters)
        gre = np.moveaxis(hist_re, 0, -3)
        gim = np.moveaxis(hist_im, 0, -3)
        gexp = np.moveaxis(hist_exp, 0, -3)
        tre, tim = _cmul_q31(gre, gim, fr[..., None], fi[..., None], counters)
        texp = gexp + fe[..., None]
        dead = czero[..., sl, :, None] | ((tre == 0) & (tim == 0))
        tre, tim = np.where(dead, 0, tre), np.where(dead, 0, tim)
        sre, sim, sexp = _align_sum((tre, tim, texp), counters, axis=-2)
        out_re[..., sl, :], out_im[..., sl, :], out_exp[..., sl, :] = sre, sim, sexp
        clock_L, clock_T = PL[..., stop - 1, :].copy(), PT[..., stop - 1, :].copy()
        start = stop
    final = ClockedState(pos + length, chunk, clock_L, clock_T, base_L, base_T, acc_re, acc_im, acc_exp)
    return (out_re, out_im, out_exp), final


def execute_clocked_state(
    values: npt.ArrayLike,
    parameters: ClockedStateParameters,
    *,
    chunk: int = 64,
    initial_state: ClockedState | None = None,
    reset_before: npt.ArrayLike | None = None,
) -> ClockedStateResult:
    """Clocked read-before-inject scans plus ``D v``, one SPF32 rounding per output.

    Semantics match :func:`spf.tensor.execute_sequence_mixer` (reset, read, inject;
    optional reverse scan; ``D v`` once; bidirectional calls need full context),
    but forward, reverse and diagonal terms are combined in fixed point and
    rounded once. The returned forward state continues a causal stream bit-exactly.
    """
    words = _words(values, "values", 2)
    batch, length, n = _validate(parameters, words)
    chunk = int(chunk)
    if not 1 <= chunk <= MAX_CHUNK:
        raise SPFError(f"chunk must be in [1, {MAX_CHUNK}]")
    reset = _resets(reset_before, (*batch, length))
    p = words.shape[-1]
    if initial_state is not None:
        if parameters.backward is not None:
            raise SPFError("streaming initial_state is supported only for a causal clocked scan")
        if initial_state.chunk != chunk:
            raise SPFError("initial_state was produced with a different chunk")
        if initial_state.acc_re.shape != (*batch, n, p):
            raise SPFError("initial_state must have accumulator shape (..., N, payload_channels)")
        state = initial_state
    else:
        state = _empty_state(batch, n, p, chunk)
    counters = ClockedStateCounters()
    parts = []
    fwd, forward_state = _scan(parameters.forward, words, reset, state, counters)
    parts.append(fwd)
    backward_state = None
    if parameters.backward is not None:
        bw = parameters.backward
        flip = lambda a: np.asarray(a)[..., ::-1, :]  # noqa: E731
        flipped = ClockedDirectionParameters(flip(bw.log_q16), flip(bw.phase_q32), flip(bw.b), flip(bw.c))
        rreset = np.zeros_like(reset)
        rreset[..., 1:] = reset[..., ::-1][..., :-1]
        (bre, bim, bexp), backward_state = _scan(
            flipped, words[..., ::-1, :], rreset, _empty_state(batch, n, p, chunk), counters
        )
        parts.append((bre[..., ::-1, :], bim[..., ::-1, :], bexp[..., ::-1, :]))
    diagonal = np.uint32(parameters.diagonal)
    if diagonal != SPF32_WORD_ZERO:
        dv = spf_mul_words(diagonal, words)
        counters.word_multiplications += int(dv.size)
        dL, dT, dzero = _word_codes(dv)
        dre, dim, dexp = _code_to_fixed(dL, dT, counters)
        parts.append((np.where(dzero, 0, dre), np.where(dzero, 0, dim), dexp - _Q))
    normalized = [_normalize(*part, _NORM_BITS) for part in parts]
    re, im, exp = normalized[0] if len(normalized) == 1 else _align_sum(normalized, counters)
    out = finalize_words(re, im, exp)
    counters.finalizations += int(np.count_nonzero(out != SPF32_WORD_ZERO))
    return ClockedStateResult(out, forward_state, backward_state, counters)


# ---------------------------------------------------------------- feature-generated parameters


@dataclass(frozen=True)
class ClockedSequenceMixer:
    """Selective decay/rotation clocks generated from native phase features.

    Each state's transition is read from ROM tables indexed by a selected feature
    phase, ``index = floor(theta * len(table) / 65536)`` (the phase-shear ROM rule):
    ``decay_tables`` hold nonpositive Q16 log codes and ``rotation_tables`` 2^32-ring
    ticks. One table is shared by every state, or one table per state. Forward and
    reverse share the tables and select their own features (Hydra's shared A).
    """

    features: PhaseFeatureNetwork
    forward: PhaseMixerDirection
    decay_tables: tuple[tuple[int, ...], ...]
    rotation_tables: tuple[tuple[int, ...], ...] | None = None
    backward: PhaseMixerDirection | None = None
    diagonal: int = int(SPF32_WORD_ZERO)

    def __post_init__(self) -> None:
        n = len(self.forward.transition_features)
        m = len(self.features.characters)

        def tables(raw, name, check):
            raw = tuple(tuple(int(v) for v in table) for table in raw)
            if len(raw) not in (1, n) or any(not 1 <= len(t) <= 65536 for t in raw):
                raise SPFError(f"{name} needs one shared table or one table per state, each 1..65536 entries")
            check(raw)
            return raw

        def nonpositive(raw):
            if any(v > 0 for t in raw for v in t):
                raise SPFError("decay tables must hold nonpositive Q16 log codes")

        object.__setattr__(self, "decay_tables", tables(self.decay_tables, "decay_tables", nonpositive))
        if self.rotation_tables is not None:
            rot = tuple(tuple(int(v) & 0xFFFFFFFF for v in t) for t in self.rotation_tables)
            object.__setattr__(self, "rotation_tables", tables(rot, "rotation_tables", lambda raw: None))
        for direction in (self.forward, self.backward):
            if direction is None:
                continue
            for selectors in (direction.transition_features, direction.write_features, direction.read_features):
                if len(selectors) != n or any(not 0 <= v < m for v in selectors):
                    raise SPFError("each direction needs one valid feature selector per state and field")
            for logs in (direction.write_log_codes, direction.read_log_codes):
                if logs is not None and (len(logs) != n or any(not -32768 <= v <= 32766 for v in logs)):
                    raise SPFError("write/read gains require one finite or zero log code per state")
        _words(np.asarray(int(self.diagonal), dtype=np.uint32), "diagonal")

    def _lookup(self, tables, theta: np.ndarray, dtype) -> np.ndarray:
        n = theta.shape[-1]
        out = np.empty(theta.shape, dtype=np.int64)
        for state in range(n):
            table = np.asarray(tables[state if len(tables) > 1 else 0], dtype=np.int64)
            out[..., state] = table[(theta[..., state] * len(table)) >> 16]
        return out.astype(dtype)

    def generate(self, addresses: npt.ArrayLike) -> ClockedStateParameters:
        """Integer table lookups only: no coherent addition, no complex decode."""
        features = self.features.execute(addresses)

        def gains(selectors, logs):
            selected = features[..., list(selectors)]
            if logs is None:
                return selected.copy()
            return pack_words(np.broadcast_to(np.asarray(logs, dtype=np.int16), selected.shape), unpack_theta(selected))

        def direction(spec: PhaseMixerDirection) -> ClockedDirectionParameters:
            theta = unpack_theta(features[..., list(spec.transition_features)]).astype(np.int64)
            log = self._lookup(self.decay_tables, theta, np.int64)
            phase = (
                np.zeros(theta.shape, dtype=np.uint32)
                if self.rotation_tables is None
                else self._lookup(self.rotation_tables, theta, np.uint32)
            )
            return ClockedDirectionParameters(
                log,
                phase,
                gains(spec.write_features, spec.write_log_codes),
                gains(spec.read_features, spec.read_log_codes),
            )

        return ClockedStateParameters(
            direction(self.forward),
            None if self.backward is None else direction(self.backward),
            np.uint32(self.diagonal),
        )

    def execute(self, addresses: npt.ArrayLike, values: npt.ArrayLike, **kwargs) -> ClockedStateResult:
        return execute_clocked_state(values, self.generate(addresses), **kwargs)


__all__ = [
    "ClockedDirectionParameters",
    "ClockedSequenceMixer",
    "ClockedState",
    "ClockedStateCounters",
    "ClockedStateParameters",
    "ClockedStateResult",
    "execute_clocked_state",
    "finalize_words",
    "integer_clock",
]
