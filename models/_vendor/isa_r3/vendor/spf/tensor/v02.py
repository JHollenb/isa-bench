"""Pure-Python SPF v0.2 reference algorithms.

This module is deliberately additive.  The older :mod:`spf.tensor.core` and
:mod:`spf.tensor.reducers` surfaces remain the compatibility implementation for
their existing callers; this module provides the typed, structured v0.2 model
described by the SPF algorithm reference.

The implementation favours explicit reference semantics over backend tricks:
SPF32 words are canonical at the boundary, positive and signed arithmetic is
performed in log-code space, wide reductions use one pivot and one final
conversion, and structured fields stay analytic until a consumer asks for
samples.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum, StrEnum
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt

from . import core as _core
from .core import PrecisionMode
from .exp2_rom import (
    EXP2_NEG_GAP_ROM,
    EXP2_NEG_GAP_ROM_MASK,
    EXP2_NEG_GAP_ROM_SHA256,
    EXP2_NEG_GAP_ROM_SIZE,
    EXP2_NEG_GAP_ROM_VERSION,
    EXP2_NEG_GAP_SHIFT_BITS,
)
from .logadd_rom import (
    LOGADD_G_ZERO,
    LOGADD_MINUS,
    LOGADD_PLUS,
    LOGADD_ROM_SHA256,
    LOGADD_ROM_VERSION,
)

# ---------------------------------------------------------------------------
# Canonical SPF32 scalar contract
# ---------------------------------------------------------------------------

SPF32_L_FRAC_BITS = 8
SPF32_L_SCALE = 1 << SPF32_L_FRAC_BITS
SPF32_PHASE_BITS = 16
SPF32_PHASE_MOD = 1 << SPF32_PHASE_BITS
SPF32_PHASE_MASK = SPF32_PHASE_MOD - 1
SPF32_PHASE_HALF = SPF32_PHASE_MOD // 2
SPF32_PHASE_QUARTER = SPF32_PHASE_MOD // 4
SPF32_INTERNAL_PHASE_BITS = 32
SPF32_INTERNAL_PHASE_MOD = 1 << SPF32_INTERNAL_PHASE_BITS
SPF32_INTERNAL_PHASE_SCALE = 1 << (SPF32_INTERNAL_PHASE_BITS - SPF32_PHASE_BITS)
SPF32_L_ZERO = -32768
SPF32_L_MIN_FINITE = -32767
SPF32_L_MAX_FINITE = 32766
SPF32_NAN_L = 32767
SPF32_NAN_T = 65535
SPF32_WORD_ZERO = 0x80000000
SPF32_WORD_NAN = 0x7FFFFFFF
SPF32_T_ZERO = 0

# The scalar contract above is the executable spelling of Algorithm Reference
# §2.3.  ``spf.tensor.core`` is the bit-level authority for the same words;
# the two must never drift, so the shared constants are cross-checked here.
assert int(_core.SPF32_L_ZERO) == SPF32_L_ZERO
assert int(_core.SPF32_L_MIN_FINITE) == SPF32_L_MIN_FINITE
assert int(_core.SPF32_L_MAX_FINITE) == SPF32_L_MAX_FINITE
assert int(_core.SPF32_L_NAN) == SPF32_NAN_L
assert int(_core.SPF32_T_NAN) == SPF32_NAN_T
assert int(_core.SPF32_WORD_ZERO) == SPF32_WORD_ZERO
assert int(_core.SPF32_WORD_NAN) == SPF32_WORD_NAN
assert int(_core.SPF32_PHASE_MASK) == SPF32_PHASE_MASK

_TAU = 2.0 * math.pi
_PHASE_TO_RAD = _TAU / SPF32_PHASE_MOD
_RAD_TO_PHASE = SPF32_PHASE_MOD / _TAU
_LOG_QUANTIZATION_RELATIVE_ERROR = 2.0 ** (1.0 / (2.0 * SPF32_L_SCALE)) - 1.0
PHASE_RELIABILITY_MAX_ERROR_RAD = math.pi / 4.0


class NaNProfile(StrEnum):
    """Named NaN behavior profiles from the v0.2 algorithm reference."""

    FROZEN_COMPAT = "FROZEN_COMPAT"
    STRICT_V02 = "STRICT_V02"


class ValueKind(StrEnum):
    """Typed value kinds used by v0.2 routing and reduction contracts."""

    LOG_POS = "LOG_POS"
    LOG_SIGNED = "LOG_SIGNED"
    PHASE = "PHASE"
    SPF_COMPLEX = "SPF_COMPLEX"
    OPTICAL_DEPTH = "OPTICAL_DEPTH"
    TROPICAL_MIN = "TROPICAL_MIN"
    TROPICAL_MAX = "TROPICAL_MAX"
    AFFINE_TRANSPORT = "AFFINE_TRANSPORT"


def _canonical_l(code: int) -> int:
    """Return a finite log code without creating alternate sentinel words."""

    value = int(code)
    if value == SPF32_L_ZERO:
        return SPF32_L_ZERO
    value = max(SPF32_L_MIN_FINITE, min(SPF32_L_MAX_FINITE, value))
    if value == SPF32_L_ZERO:
        return SPF32_L_MIN_FINITE
    return value


def _clamp_finite_l(code: int) -> int:
    """Saturate an arithmetic result while reserving ``L_ZERO`` for zero."""

    value = max(SPF32_L_MIN_FINITE, min(SPF32_L_MAX_FINITE, int(code)))
    return SPF32_L_MIN_FINITE if value == SPF32_L_ZERO else value


def _pack_scalar(L: int, T: int) -> int:
    log_code = int(L)
    phase = int(T) & SPF32_PHASE_MASK
    if log_code == SPF32_L_ZERO:
        return SPF32_WORD_ZERO
    if log_code == SPF32_NAN_L and phase == SPF32_NAN_T:
        return SPF32_WORD_NAN
    log_code = _canonical_l(log_code)
    return ((log_code & 0xFFFF) << SPF32_PHASE_BITS) | phase


def _raw_word(value: Any) -> int:
    """Read a word without canonicalizing its reserved log-code bits."""

    if isinstance(value, SPF32Value):
        return value.word
    if isinstance(value, (int, np.integer)):
        return int(value) & 0xFFFFFFFF
    return _coerce_word(value)


def _pack_frozen_phase(raw_word: int, phase: int) -> int:
    """Apply the frozen pack_words rule used by the compatibility profile."""

    raw = int(raw_word) & 0xFFFFFFFF
    raw_log = (raw >> SPF32_PHASE_BITS) & 0xFFFF
    if raw_log == (SPF32_L_ZERO & 0xFFFF):
        return SPF32_WORD_ZERO
    return (raw_log << SPF32_PHASE_BITS) | (int(phase) & SPF32_PHASE_MASK)


def _pack_frozen_arithmetic(log_code: int, phase: int) -> int:
    """Pack a frozen arithmetic result after the frozen int16 saturation."""

    value = max(SPF32_L_MIN_FINITE, min(SPF32_L_MAX_FINITE, int(log_code)))
    return ((value & 0xFFFF) << SPF32_PHASE_BITS) | (int(phase) & SPF32_PHASE_MASK)


@dataclass(frozen=True, slots=True)
class SPF32Value:
    """Unpacked canonical SPF32 value used by the reference helpers."""

    L: int
    T: int = SPF32_T_ZERO

    def __post_init__(self) -> None:
        log_code = int(self.L)
        phase = int(self.T) & SPF32_PHASE_MASK
        if log_code == SPF32_L_ZERO:
            phase = SPF32_T_ZERO
        elif log_code == SPF32_NAN_L and phase == SPF32_NAN_T:
            phase = SPF32_NAN_T
        else:
            log_code = _canonical_l(log_code)
        object.__setattr__(self, "L", log_code)
        object.__setattr__(self, "T", phase)

    @property
    def word(self) -> int:
        return _pack_scalar(self.L, self.T)

    @property
    def is_zero(self) -> bool:
        return self.L == SPF32_L_ZERO

    @property
    def is_nan(self) -> bool:
        return self.L == SPF32_NAN_L and self.T == SPF32_NAN_T

    @classmethod
    def zero(cls) -> SPF32Value:
        return cls(SPF32_L_ZERO, SPF32_T_ZERO)

    @classmethod
    def nan(cls) -> SPF32Value:
        return cls(SPF32_NAN_L, SPF32_NAN_T)


@dataclass(frozen=True, slots=True)
class RawLogCode:
    """Explicit wrapper for an integer log-code input to a typed reducer.

    A Python ``int`` is deliberately *not* accepted as an implicit log code by
    the positive and signed reducers: the same object could be a packed
    SPF32 word.  Callers that really have raw log codes can use this wrapper
    (or pass ``raw_log_codes=True`` to the reducer) so the representation is
    unambiguous at the API boundary.
    """

    code: int
    sign: int = 1

    def __post_init__(self) -> None:
        sign = int(self.sign)
        if sign not in {-1, 1}:
            raise ValueError("RawLogCode.sign must be +1 or -1")
        object.__setattr__(self, "code", int(self.code))
        object.__setattr__(self, "sign", sign)

    @property
    def L(self) -> int:
        """Compatibility spelling matching :class:`LogPos`/``LogSigned``."""

        return self.code


@dataclass(frozen=True, slots=True)
class LogPos:
    """A non-negative real value represented by an SPF32 log-magnitude code."""

    L: int

    def __post_init__(self) -> None:
        value = int(self.L)
        object.__setattr__(self, "L", SPF32_NAN_L if value == SPF32_NAN_L else _canonical_l(value))

    @property
    def is_zero(self) -> bool:
        return self.L == SPF32_L_ZERO

    @property
    def is_nan(self) -> bool:
        return self.L == SPF32_NAN_L

    @property
    def value(self) -> float:
        """Decode to float64 through the SAME ufunc the vectorized decode uses.

        ``np.exp2`` and ``math.exp2`` are not the same function. Measured on
        glibc x86 (beast): they differ by up to 1 ulp on 644 of the 9143 SPF32
        log codes exercised; on Apple libm they are bit-identical. Since
        :func:`spf.tensor.core.decode_complex` — the vectorized decode that the
        renderer, the reducers and :func:`decode_complex` all go through — uses
        ``np.exp2``, this scalar property must too, or scalar and plane decode of
        the same word disagree in the last bit on Linux and agree on macOS.
        ``self.L / SPF32_L_SCALE`` is an exact power-of-two division on both
        sides, so the ufunc is the only degree of freedom. Words are unaffected:
        re-encoding the decoded value reproduces the same word either way.
        """

        if self.is_zero:
            return 0.0
        if self.is_nan:
            return math.nan
        return float(np.exp2(self.L / SPF32_L_SCALE))

    def to_float(self) -> float:
        """Return the decoded positive value for compatibility callers."""

        return self.value

    @classmethod
    def zero(cls) -> LogPos:
        return cls(SPF32_L_ZERO)

    @classmethod
    def nan(cls) -> LogPos:
        return cls(SPF32_NAN_L)

    @classmethod
    def from_value(cls, value: float) -> LogPos:
        number = float(value)
        if math.isnan(number):
            return cls.nan()
        if number == 0.0:
            return cls.zero()
        if number < 0.0:
            raise ValueError("LogPos requires a non-negative value")
        if math.isinf(number):
            return cls(SPF32_L_MAX_FINITE)
        return cls(_clamp_finite_l(_rne(SPF32_L_SCALE * math.log2(number))))

    def __int__(self) -> int:
        return self.L


@dataclass(frozen=True, slots=True)
class LogSigned:
    """A signed real value ``sign * 2**(L/256)``."""

    L: int
    sign: int = 1

    def __post_init__(self) -> None:
        log_code = int(self.L)
        sign = int(self.sign)
        if log_code == SPF32_NAN_L:
            sign = 0
        elif log_code == SPF32_L_ZERO or sign == 0:
            log_code = SPF32_L_ZERO
            sign = 0
        else:
            log_code = _canonical_l(log_code)
            sign = 1 if sign > 0 else -1
        object.__setattr__(self, "L", log_code)
        object.__setattr__(self, "sign", sign)

    @property
    def is_zero(self) -> bool:
        return not self.is_nan and (self.sign == 0 or self.L == SPF32_L_ZERO)

    @property
    def is_nan(self) -> bool:
        return self.L == SPF32_NAN_L

    @property
    def value(self) -> float:
        """Decode to float64 through ``np.exp2`` — see :attr:`LogPos.value`.

        The ``sign`` factor is exactly ``±1.0``, so the magnitude ufunc is the
        only place scalar and vector decode could diverge.
        """

        if self.is_zero:
            return 0.0
        if self.is_nan:
            return math.nan
        return float(self.sign) * float(np.exp2(self.L / SPF32_L_SCALE))

    def to_float(self) -> float:
        """Return the decoded signed value for compatibility callers."""

        return self.value

    @classmethod
    def zero(cls) -> LogSigned:
        return cls(SPF32_L_ZERO, 0)

    @classmethod
    def nan(cls) -> LogSigned:
        return cls(SPF32_NAN_L, 0)

    @classmethod
    def from_value(cls, value: float) -> LogSigned:
        number = float(value)
        if math.isnan(number):
            return cls.nan()
        if number == 0.0:
            return cls.zero()
        if math.isinf(number):
            return cls(SPF32_L_MAX_FINITE, 1 if number > 0.0 else -1)
        return cls(_clamp_finite_l(_rne(SPF32_L_SCALE * math.log2(abs(number)))), 1 if number > 0 else -1)


def _coerce_reduction_logpos(value: Any, *, raw_log_codes: bool) -> LogPos:
    """Coerce one explicitly typed positive reduction input.

    Reducer inputs are stricter than the scalar arithmetic helpers.  In
    particular, a bare integer is rejected unless the caller explicitly
    opts into the raw-code compatibility mode.
    """

    if isinstance(value, LogPos):
        return value
    if isinstance(value, RawLogCode):
        if value.sign != 1:
            raise TypeError("LOG_POS reduction cannot contain a negative RawLogCode")
        return LogPos(value.code)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("LOG_POS reduction values must be typed real values, not bool")
    if isinstance(value, (int, np.integer)):
        if not raw_log_codes:
            raise TypeError(
                "LOG_POS reduction received a bare integer (possibly a packed SPF32 word); "
                "use LogPos/RawLogCode "
                "or pass raw_log_codes=True"
            )
        return LogPos(int(value))
    if isinstance(value, (float, np.floating)):
        return LogPos.from_value(float(value))
    raise TypeError(
        "LOG_POS reduction expects LogPos, a real scalar, or RawLogCode; "
        f"received {type(value).__name__}"
    )


def _coerce_reduction_logsigned(value: Any, *, raw_log_codes: bool) -> LogSigned:
    """Coerce one explicitly typed signed reduction input."""

    if isinstance(value, LogSigned):
        return value
    if isinstance(value, RawLogCode):
        return LogSigned(value.code, value.sign)
    if isinstance(value, (bool, np.bool_)):
        raise TypeError("LOG_SIGNED reduction values must be typed real values, not bool")
    if isinstance(value, (int, np.integer)):
        if not raw_log_codes:
            raise TypeError(
                "LOG_SIGNED reduction received a bare integer (possibly a packed SPF32 word); "
                "use LogSigned/RawLogCode "
                "or pass raw_log_codes=True"
            )
        return LogSigned(int(value), 1)
    if isinstance(value, (float, np.floating)):
        return LogSigned.from_value(float(value))
    raise TypeError(
        "LOG_SIGNED reduction expects LogSigned, a real scalar, or RawLogCode; "
        f"received {type(value).__name__}"
    )


@dataclass(frozen=True, slots=True)
class Phase:
    """A wrapped SPF phase code without a log-magnitude plane."""

    T: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "T", int(self.T) & SPF32_PHASE_MASK)

    @property
    def theta(self) -> int:
        return self.T

    @property
    def radians(self) -> float:
        return self.T * _PHASE_TO_RAD

    def __int__(self) -> int:
        return self.T


def phase_compose(a: Phase | int, b: Phase | int) -> Phase:
    """Compose two phase-only values by modular addition."""

    left = a.T if isinstance(a, Phase) else int(a)
    right = b.T if isinstance(b, Phase) else int(b)
    return Phase(left + right)


def reduce_phase(values: Iterable[Phase | int]) -> Phase:
    """Reduce a phase sequence without materializing a log-magnitude field."""

    result = Phase(0)
    for value in values:
        result = phase_compose(result, value)
    return result


def tropical_min(values: Iterable[Any]) -> Any:
    """Return the ordered minimum used by the ``TROPICAL_MIN`` algebra."""

    items = list(values)
    if not items:
        raise ValueError("tropical_min requires at least one value")
    if any(_contains_nan(item) for item in items):
        return math.nan
    return min(items)


def tropical_max(values: Iterable[Any]) -> Any:
    """Return the ordered maximum used by the ``TROPICAL_MAX`` algebra."""

    items = list(values)
    if not items:
        raise ValueError("tropical_max requires at least one value")
    if any(_contains_nan(item) for item in items):
        return math.nan
    return max(items)


def optical_depth_add(a: float, b: float) -> float:
    """Combine non-negative optical depths in their ordinary additive algebra."""

    left, right = float(a), float(b)
    if left < 0.0 or right < 0.0:
        raise ValueError("optical depth must be non-negative")
    return left + right


def reduce_optical_depth(values: Iterable[float]) -> float:
    """Reduce non-negative optical depths with a stable linear sum."""

    items = [float(value) for value in values]
    if any(value < 0.0 for value in items):
        raise ValueError("optical depth must be non-negative")
    return math.fsum(items)


def _rne(value: float) -> int:
    """Round-to-nearest, ties-to-even, matching the SPF reference."""

    return round(float(value))


def _canonical_words(words: npt.ArrayLike) -> np.ndarray:
    """Vectorized :func:`_pack_scalar` over raw packed words.

    Zero keeps only its sentinel, the single NaN word is preserved, and every
    other log code (including the reserved ``NAN_L`` with a non-NaN phase) is
    saturated into the finite range.  Word planes come from and go back
    through :mod:`spf.tensor.core`; this function only applies the v0.2
    canonicalization rule on top of them.
    """

    raw = np.asarray(words, dtype=np.uint32)
    L = _core.unpack_L(raw).astype(np.int64)
    T = _core.unpack_theta(raw).astype(np.int64)
    zero = L == SPF32_L_ZERO
    nan = (L == SPF32_NAN_L) & (T == SPF32_NAN_T)
    L = np.where(zero, SPF32_L_ZERO, np.clip(L, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE))
    L = np.where(nan, SPF32_NAN_L, L)
    T = np.where(zero, SPF32_T_ZERO, T)
    return np.asarray(_core.pack_words(L.astype(np.int16), T.astype(np.uint16)), dtype=np.uint32)


def _plane_words(L: npt.ArrayLike, T: npt.ArrayLike) -> np.ndarray:
    """Pack canonical ``(L, T)`` planes with the v0.2 rule via ``core.pack_words``."""

    l_arr = np.asarray(L, dtype=np.int64)
    t_arr = np.asarray(T, dtype=np.int64)
    l_arr, t_arr = np.broadcast_arrays(l_arr, t_arr)
    t_arr = t_arr & SPF32_PHASE_MASK
    zero = l_arr == SPF32_L_ZERO
    nan = (l_arr == SPF32_NAN_L) & (t_arr == SPF32_NAN_T)
    l_can = np.where(zero, SPF32_L_ZERO, np.clip(l_arr, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE))
    l_can = np.where(nan, SPF32_NAN_L, l_can)
    t_can = np.where(zero, SPF32_T_ZERO, t_arr)
    return np.asarray(_core.pack_words(l_can.astype(np.int16), t_can.astype(np.uint16)), dtype=np.uint32)


def pack_spf32(
    L: int | SPF32Value | LogPos | LogSigned | npt.ArrayLike,
    T: int | npt.ArrayLike | None = None,
) -> int | np.ndarray:
    """Pack canonical SPF32 words from an ``(L, T)`` pair.

    Scalar calls return a Python ``int``.  Array-like inputs broadcast and
    return a ``uint32`` array.  Zero always becomes ``0x80000000`` regardless
    of the supplied phase, and the only NaN encoding is ``0x7fffffff``.
    """

    if T is None:
        if isinstance(L, SPF32Value):
            return _pack_scalar(L.L, L.T)
        if isinstance(L, LogPos):
            return _pack_scalar(L.L, SPF32_NAN_T if L.is_nan else SPF32_T_ZERO)
        if isinstance(L, LogSigned):
            if L.is_nan:
                return SPF32_WORD_NAN
            return _pack_scalar(L.L, SPF32_T_ZERO if L.sign >= 0 else SPF32_PHASE_HALF)
        if isinstance(L, (tuple, list)) and len(L) == 2:
            L, T = L
        else:
            raise TypeError("pack_spf32 expects (L, T) or a typed SPF value")

    l_arr = np.asarray(L)
    t_arr = np.asarray(T)
    if l_arr.ndim == 0 and t_arr.ndim == 0:
        return _pack_scalar(int(l_arr), int(t_arr))
    return _plane_words(l_arr, t_arr)


def unpack_L(word: int | npt.ArrayLike) -> int | np.ndarray:
    """Unpack the signed upper 16-bit SPF32 log code."""

    arr = np.asarray(word)
    if arr.ndim == 0:
        raw = (int(arr) >> SPF32_PHASE_BITS) & 0xFFFF
        return raw - 0x10000 if raw & 0x8000 else raw
    return np.asarray(_core.unpack_L(np.asarray(arr, dtype=np.uint32)), dtype=np.int64)


def unpack_T(word: int | npt.ArrayLike) -> int | np.ndarray:
    """Unpack the wrapped lower 16-bit SPF32 phase code."""

    arr = np.asarray(word)
    if arr.ndim == 0:
        return int(arr) & SPF32_PHASE_MASK
    return np.asarray(_core.unpack_theta(np.asarray(arr, dtype=np.uint32)), dtype=np.uint16)


unpack_theta = unpack_T


def unpack_spf32(word: int) -> SPF32Value:
    """Return one unpacked canonical SPF32 value."""

    return SPF32Value(int(unpack_L(word)), int(unpack_T(word)))


def encode_complex(z: npt.ArrayLike) -> np.ndarray | int:
    """Encode complex values into canonical SPF32 words.

    The arithmetic is :func:`spf.tensor.core.encode_words` (round-to-nearest
    ``256·log2|z|`` and ``65536·arg(z)/2π``).  The v0.2 profile differs from
    the frozen core only at the sentinels: an infinite magnitude saturates to
    the largest finite log code instead of becoming a directional infinity,
    because the v0.2 encoder has exactly one NaN word and no infinity words.
    """

    array = np.asarray(z, dtype=np.complex128)
    words = np.asarray(_core.encode_words(array), dtype=np.uint32)
    L = _core.unpack_L(words).astype(np.int64)
    T = _core.unpack_theta(words).astype(np.int64)
    inf = (L == SPF32_NAN_L) & (T != SPF32_NAN_T)
    if np.any(inf):
        L = np.where(inf, SPF32_L_MAX_FINITE, L)
        words = np.asarray(_core.pack_words(L.astype(np.int16), T.astype(np.uint16)), dtype=np.uint32)
    if array.ndim == 0:
        return int(words.reshape(-1)[0])
    return words


def decode_complex(word: int | npt.ArrayLike) -> np.ndarray | complex:
    """Decode canonical SPF32 words into complex values.

    Words are canonicalized first (a reserved-``L`` word with a non-NaN phase
    saturates to the finite maximum rather than becoming a second,
    directional-infinity format); the numeric decode is then
    :func:`spf.tensor.core.decode_words`, so the renderer, the reducers, and
    this module all emit bit-identical complex values for the same word.

    The scalar branch is the SAME call with a 0-d input — it is deliberately not
    a ``math.exp2``/``cmath.exp`` shortcut, because ``np.exp2`` and
    ``math.exp2`` differ by up to 1 ulp on glibc x86 (measured: 644 of 9143
    SPF32 log codes on beast) while agreeing on Apple libm. Keep it that way;
    :attr:`LogPos.value` documents the same rule for the typed kinds.
    """

    array = np.asarray(word, dtype=np.uint32)
    values = np.asarray(_core.decode_words(_canonical_words(array)), dtype=np.complex128)
    if array.ndim == 0:
        return complex(values.reshape(-1)[0])
    return values


encode_words = encode_complex
decode_words = decode_complex
encode = encode_complex
decode = decode_complex


def canonicalize(L: int, T: int = SPF32_T_ZERO) -> tuple[int, int]:
    """Canonicalize an unpacked scalar pair, in §2.4's test order.

    §2.4 tests the raw pair: zero first, then ``(L, T) == (NAN_L, NAN_T)``,
    then clamp ``L`` and reduce ``T`` mod 65536.  Routing through
    :func:`_pack_scalar` reduced ``T`` before the NaN test, so ``(32767,
    131071)`` — a phase that is NaN only after reduction — canonicalized to
    NaN instead of to the clamped finite pair.  The packer keeps its own order
    (it takes already-reduced phases from the word paths); this entry point is
    the one the design specifies, so it does the test on the raw input.
    """

    log_code = int(L)
    phase = int(T)
    if log_code == SPF32_L_ZERO:
        return SPF32_L_ZERO, SPF32_T_ZERO
    if log_code == SPF32_NAN_L and phase == SPF32_NAN_T:
        return SPF32_NAN_L, SPF32_NAN_T
    return _canonical_l(log_code), phase & SPF32_PHASE_MASK


def circular_diff_u16(a: int, b: int) -> int:
    """Return the signed circular difference ``a - b`` in ``[-32768, 32767]``."""

    difference = (int(a) - int(b)) & SPF32_PHASE_MASK
    return difference if difference <= SPF32_PHASE_HALF - 1 else difference - SPF32_PHASE_MOD


def is_zero(word: int | SPF32Value) -> bool:
    value = word if isinstance(word, SPF32Value) else unpack_spf32(int(word))
    return value.is_zero


def is_nan(word: int | SPF32Value) -> bool:
    value = word if isinstance(word, SPF32Value) else unpack_spf32(int(word))
    return value.is_nan


def _coerce_word(
    value: int | float | complex | SPF32Value | tuple[int, int] | LogPos | LogSigned,
) -> int:
    if isinstance(value, SPF32Value):
        return value.word
    if isinstance(value, LogPos):
        return _pack_scalar(value.L, SPF32_NAN_T if value.is_nan else SPF32_T_ZERO)
    if isinstance(value, LogSigned):
        if value.is_nan:
            return SPF32_WORD_NAN
        return _pack_scalar(value.L, SPF32_T_ZERO if value.sign >= 0 else SPF32_PHASE_HALF)
    if isinstance(value, tuple) and len(value) == 2:
        return _pack_scalar(value[0], value[1])
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        encoded = encode_complex(value)
        return int(np.asarray(encoded, dtype=np.uint32).reshape(-1)[0])
    raw = int(value) & 0xFFFFFFFF
    return _pack_scalar(int(unpack_L(raw)), int(unpack_T(raw)))


def _word_array(value: Any, *, preserve_raw: bool) -> np.ndarray:
    """Coerce an array-like of words/typed values to a ``uint32`` word array."""

    arr = np.asarray(value)
    if arr.dtype.kind in "iu":
        raw = np.asarray(arr, dtype=np.int64).astype(np.uint64) & np.uint64(0xFFFFFFFF)
        raw32 = raw.astype(np.uint32)
        return raw32 if preserve_raw else _canonical_words(raw32)
    coerce = _raw_word if preserve_raw else _coerce_word
    out = np.empty(arr.shape, dtype=np.uint32)
    for index in np.ndindex(arr.shape):
        out[index] = coerce(arr[index])
    return out


def _binary_plane_words(
    a: Any,
    b: Any,
    *,
    sign: int,
    profile: NaNProfile,
) -> np.ndarray:
    """Vectorized ``mul`` (``sign=+1``) / ``div`` (``sign=-1``) on word planes.

    Plane arithmetic is the frozen core rule (saturated log add/sub, modular
    phase add/sub).  The v0.2 sentinel policy is applied as masks on top:
    STRICT_V02 propagates the single NaN word and canonicalizes inputs first;
    FROZEN_COMPAT reads raw words and only reserves the zero sentinel.
    """

    frozen = profile is NaNProfile.FROZEN_COMPAT
    wa = _word_array(a, preserve_raw=frozen)
    wb = _word_array(b, preserve_raw=frozen)
    wa, wb = np.broadcast_arrays(wa, wb)
    la = _core.unpack_L(wa).astype(np.int64)
    ta = _core.unpack_theta(wa).astype(np.int64)
    lb = _core.unpack_L(wb).astype(np.int64)
    tb = _core.unpack_theta(wb).astype(np.int64)
    L = np.clip(la + sign * lb, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE)
    T = (ta + sign * tb) & SPF32_PHASE_MASK
    out = np.asarray(_core.pack_words(L.astype(np.int16), T.astype(np.uint16)), dtype=np.uint32)
    zero_a = la == SPF32_L_ZERO
    zero_b = lb == SPF32_L_ZERO
    if sign > 0:
        out = np.where(zero_a | zero_b, np.uint32(SPF32_WORD_ZERO), out)
    else:
        out = np.where(zero_b, np.uint32(SPF32_WORD_NAN), out)
        out = np.where(zero_a, np.uint32(SPF32_WORD_ZERO), out)
    if not frozen:
        nan = (wa == np.uint32(SPF32_WORD_NAN)) | (wb == np.uint32(SPF32_WORD_NAN))
        out = np.where(nan, np.uint32(SPF32_WORD_NAN), out)
    return np.asarray(out, dtype=np.uint32)


def _unary_phase_words(word: Any, *, delta: int | None, negate: bool, profile: NaNProfile) -> np.ndarray:
    """Vectorized conj (``negate``) / phase shift (``delta``) on word planes."""

    frozen = profile is NaNProfile.FROZEN_COMPAT
    w = _word_array(word, preserve_raw=frozen)
    L = _core.unpack_L(w).astype(np.int64)
    T = _core.unpack_theta(w).astype(np.int64)
    T_new = (-T if negate else T + int(delta or 0)) & SPF32_PHASE_MASK
    out = np.asarray(_core.pack_words(L.astype(np.int16), T_new.astype(np.uint16)), dtype=np.uint32)
    zero = L == SPF32_L_ZERO
    out = np.where(zero, w, out)
    if not frozen:
        out = np.where(w == np.uint32(SPF32_WORD_NAN), w, out)
    return np.asarray(out, dtype=np.uint32)


def _map_binary_words(a: Any, b: Any, operation: Any, *, preserve_raw: bool = False) -> int | np.ndarray:
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    coerce = _raw_word if preserve_raw else _coerce_word
    if a_arr.ndim == 0 and b_arr.ndim == 0:
        return operation(coerce(a), coerce(b))
    a_b, b_b = np.broadcast_arrays(a_arr, b_arr)
    out = np.empty(a_b.shape, dtype=np.uint32)
    for index in np.ndindex(a_b.shape):
        out[index] = operation(coerce(a_b[index]), coerce(b_b[index]))
    return out


def _word_parts(word: int) -> tuple[int, int]:
    return int(unpack_L(word)), int(unpack_T(word))


def _strict_nan_or_zero(a: int, b: int) -> int | None:
    if is_nan(a) or is_nan(b):
        return SPF32_WORD_NAN
    if is_zero(a) or is_zero(b):
        return SPF32_WORD_ZERO
    return None


def _finite_arithmetic_l(value: int) -> int:
    return _clamp_finite_l(value)


def spf_mul(
    a: Any,
    b: Any,
    profile: NaNProfile = NaNProfile.STRICT_V02,
) -> int | np.ndarray:
    """Multiply SPF words under the selected NaN behavior profile."""

    mode = NaNProfile(profile)

    def multiply(wa: int, wb: int) -> int:
        if mode is NaNProfile.STRICT_V02:
            special = _strict_nan_or_zero(wa, wb)
            if special is not None:
                return special
        elif is_zero(wa) or is_zero(wb):
            return SPF32_WORD_ZERO
        la, ta = _word_parts(wa)
        lb, tb = _word_parts(wb)
        if mode is NaNProfile.FROZEN_COMPAT:
            return _pack_frozen_arithmetic(la + lb, ta + tb)
        return _pack_scalar(_finite_arithmetic_l(la + lb), ta + tb)

    if np.ndim(a) == 0 and np.ndim(b) == 0:
        return _map_binary_words(a, b, multiply, preserve_raw=mode is NaNProfile.FROZEN_COMPAT)
    return _binary_plane_words(a, b, sign=1, profile=mode)


def spf_div(
    a: Any,
    b: Any,
    profile: NaNProfile = NaNProfile.STRICT_V02,
) -> int | np.ndarray:
    """Divide SPF words under the selected NaN behavior profile."""

    mode = NaNProfile(profile)

    def divide(wa: int, wb: int) -> int:
        if mode is NaNProfile.STRICT_V02 and (is_nan(wa) or is_nan(wb)):
            return SPF32_WORD_NAN
        if is_zero(wa):
            return SPF32_WORD_ZERO
        if is_zero(wb):
            return SPF32_WORD_NAN
        la, ta = _word_parts(wa)
        lb, tb = _word_parts(wb)
        if mode is NaNProfile.FROZEN_COMPAT:
            return _pack_frozen_arithmetic(la - lb, ta - tb)
        return _pack_scalar(_finite_arithmetic_l(la - lb), ta - tb)

    if np.ndim(a) == 0 and np.ndim(b) == 0:
        return _map_binary_words(a, b, divide, preserve_raw=mode is NaNProfile.FROZEN_COMPAT)
    return _binary_plane_words(a, b, sign=-1, profile=mode)


def spf_conj(
    word: Any,
    profile: NaNProfile = NaNProfile.STRICT_V02,
) -> int | np.ndarray:
    """Conjugate an SPF word under the selected NaN behavior profile."""

    mode = NaNProfile(profile)

    arr = np.asarray(word)
    if arr.ndim == 0:
        value = _raw_word(word) if mode is NaNProfile.FROZEN_COMPAT else _coerce_word(word)
        if is_zero(value) or (mode is NaNProfile.STRICT_V02 and is_nan(value)):
            return value
        _, T = _word_parts(value)
        if mode is NaNProfile.FROZEN_COMPAT:
            return _pack_frozen_phase(value, -T)
        L, _ = _word_parts(value)
        return _pack_scalar(L, -T)
    return _unary_phase_words(arr, delta=None, negate=True, profile=mode)


def spf_neg(
    word: Any,
    profile: NaNProfile = NaNProfile.STRICT_V02,
) -> int | np.ndarray:
    """Negate an SPF word under the selected NaN behavior profile."""

    mode = NaNProfile(profile)

    arr = np.asarray(word)
    if arr.ndim == 0:
        value = _raw_word(word) if mode is NaNProfile.FROZEN_COMPAT else _coerce_word(word)
        if is_zero(value) or (mode is NaNProfile.STRICT_V02 and is_nan(value)):
            return value
        _, T = _word_parts(value)
        if mode is NaNProfile.FROZEN_COMPAT:
            return _pack_frozen_phase(value, T + SPF32_PHASE_HALF)
        L, _ = _word_parts(value)
        return _pack_scalar(L, T + SPF32_PHASE_HALF)
    return _unary_phase_words(arr, delta=SPF32_PHASE_HALF, negate=False, profile=mode)


def spf_phase_shift(
    word: Any,
    dT: int,
    profile: NaNProfile = NaNProfile.STRICT_V02,
) -> int | np.ndarray:
    """Shift an SPF phase under the selected NaN behavior profile."""

    mode = NaNProfile(profile)

    arr = np.asarray(word)
    if arr.ndim == 0:
        value = _raw_word(word) if mode is NaNProfile.FROZEN_COMPAT else _coerce_word(word)
        if is_zero(value) or (mode is NaNProfile.STRICT_V02 and is_nan(value)):
            return value
        _, T = _word_parts(value)
        if mode is NaNProfile.FROZEN_COMPAT:
            return _pack_frozen_phase(value, T + int(dT))
        L, _ = _word_parts(value)
        return _pack_scalar(L, T + int(dT))
    return _unary_phase_words(arr, delta=int(dT), negate=False, profile=mode)


def v02_phase_shift(
    value: Any,
    dT: int | Phase,
    profile: NaNProfile = NaNProfile.STRICT_V02,
) -> Phase | np.uint16 | int | np.ndarray:
    """Apply a modular v0.2 phase shift to a phase or packed SPF32 value.

    ``Phase`` and ``uint16`` inputs use the phase-only carrier and return the
    corresponding wrapped phase.  Packed SPF32 inputs retain their log plane
    and use the normal SPF sentinel behavior.  The two carriers are kept
    explicit so a phase code is never mistaken for a dense angle in radians.
    """

    delta = dT.T if isinstance(dT, Phase) else int(dT)
    if isinstance(value, Phase):
        return Phase(value.T + delta)
    if isinstance(value, np.uint16):
        return np.uint16((int(value) + delta) & SPF32_PHASE_MASK)
    array = np.asarray(value)
    if array.ndim > 0 and array.dtype == np.uint16:
        return ((array.astype(np.uint32) + delta) & SPF32_PHASE_MASK).astype(np.uint16)
    return spf_phase_shift(value, delta, profile)


def spf_abs2(word: int | SPF32Value) -> LogPos:
    """Return ``|z|²`` as a typed positive-real value."""

    value = unpack_spf32(_coerce_word(word))
    if value.is_nan:
        return LogPos.nan()
    if value.is_zero:
        return LogPos.zero()
    return LogPos(_finite_arithmetic_l(2 * value.L))


# ---------------------------------------------------------------------------
# Deterministic positive/signed log arithmetic
# ---------------------------------------------------------------------------


def _coerce_logpos(value: LogPos | int | float) -> LogPos:
    if isinstance(value, LogPos):
        return value
    if isinstance(value, (float, np.floating)):
        return LogPos.from_value(float(value))
    return LogPos(int(value))


def logpos_mul(
    a: LogPos | int | float,
    b: LogPos | int | float,
    *,
    counters: PerformanceCounters | None = None,
) -> LogPos:
    """Multiply two positive log values by adding their log codes."""

    if counters is not None:
        counters.log_ops += 1
    left, right = _coerce_logpos(a), _coerce_logpos(b)
    if left.is_nan or right.is_nan:
        return LogPos.nan()
    if left.is_zero or right.is_zero:
        return LogPos.zero()
    return LogPos(_finite_arithmetic_l(left.L + right.L))


def logpos_div(
    a: LogPos | int | float,
    b: LogPos | int | float,
    *,
    counters: PerformanceCounters | None = None,
) -> LogPos:
    """Divide two positive log values by subtracting their log codes."""

    if counters is not None:
        counters.log_ops += 1
    left, right = _coerce_logpos(a), _coerce_logpos(b)
    if left.is_nan or right.is_nan:
        return LogPos.nan()
    if left.is_zero:
        return LogPos.zero()
    if right.is_zero:
        return LogPos.nan()
    return LogPos(_finite_arithmetic_l(left.L - right.L))


def logadd_pos(
    a: LogPos | int | float,
    b: LogPos | int | float,
    *,
    counters: PerformanceCounters | None = None,
) -> LogPos:
    """Exact-to-SPF32 positive log addition using ``LOGADD_PLUS``."""

    if counters is not None:
        counters.positive_logadds += 1
    left, right = _coerce_logpos(a), _coerce_logpos(b)
    if left.is_nan or right.is_nan:
        return LogPos.nan()
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    if left.L < right.L:
        left, right = right, left
    gap = left.L - right.L
    correction = 0 if gap >= LOGADD_G_ZERO else LOGADD_PLUS[gap]
    return LogPos(_finite_arithmetic_l(left.L + correction))


def logsub_pos(
    a: LogPos | int | float,
    b: LogPos | int | float,
    *,
    counters: PerformanceCounters | None = None,
) -> LogPos:
    """Exact-to-SPF32 positive subtraction, requiring ``a >= b >= 0``."""

    if counters is not None:
        counters.positive_logadds += 1
    left, right = _coerce_logpos(a), _coerce_logpos(b)
    if left.is_nan or right.is_nan:
        return LogPos.nan()
    if right.is_zero:
        return left
    if left.is_zero:
        # 0 - b is outside the positive-real domain for b>0.  The v0.2
        # typed algebra reports that invalid result as its canonical NaN;
        # only an ordering violation between two nonzero operands is a
        # caller-visible domain error.
        return LogPos.nan()
    if left.L < right.L:
        raise ValueError("logsub_pos requires a >= b >= 0")
    gap = left.L - right.L
    if gap == 0:
        return LogPos.zero()
    correction = 0 if gap >= LOGADD_G_ZERO else LOGADD_MINUS[gap]
    return LogPos(_finite_arithmetic_l(left.L + correction))


def _coerce_logsigned(value: LogSigned | int | float) -> LogSigned:
    if isinstance(value, LogSigned):
        return value
    if isinstance(value, (float, np.floating)):
        return LogSigned.from_value(float(value))
    return LogSigned(int(value), 1)


def logadd_signed(
    a: LogSigned | int | float,
    b: LogSigned | int | float,
    *,
    counters: PerformanceCounters | None = None,
) -> LogSigned:
    """Add signed-real values with positive log-add/sub lookup tables."""

    if counters is not None:
        counters.signed_logadds += 1
    left, right = _coerce_logsigned(a), _coerce_logsigned(b)
    if left.is_nan or right.is_nan:
        return LogSigned.nan()
    if left.is_zero:
        return right
    if right.is_zero:
        return left
    if left.sign == right.sign:
        return LogSigned(logadd_pos(left.L, right.L, counters=counters).L, left.sign)
    if left.L == right.L:
        return LogSigned.zero()
    if left.L > right.L:
        return LogSigned(logsub_pos(left.L, right.L, counters=counters).L, left.sign)
    return LogSigned(logsub_pos(right.L, left.L, counters=counters).L, right.sign)


def log1p_pos(value: LogPos | int | float) -> LogPos:
    """Return ``1 + value`` in positive log form."""

    return logadd_pos(LogPos(0), value)


def log1m_pos(value: LogPos | int | float) -> LogPos:
    """Return ``1 - value`` in positive log form."""

    return logsub_pos(LogPos(0), value)


# Alpha stores the positive log value of the covered fraction; transport
# stores its complement as transmittance.  Keep this named adapter beside the
# primitive because it is the typed boundary used by compositing callers.
logpos_complement = log1m_pos
alpha_to_transmittance = log1m_pos


# ---------------------------------------------------------------------------
# Complex pair reference and hybrid add/sub
# ---------------------------------------------------------------------------


def _sincos_phase(T: int) -> tuple[float, float]:
    phase = int(T) & SPF32_PHASE_MASK
    if phase == 0:
        return 0.0, 1.0
    if phase == SPF32_PHASE_QUARTER:
        return 1.0, 0.0
    if phase == SPF32_PHASE_HALF:
        return 0.0, -1.0
    if phase == 3 * SPF32_PHASE_QUARTER:
        return -1.0, 0.0
    angle = phase * _PHASE_TO_RAD
    return math.sin(angle), math.cos(angle)


def _word_from_local(
    base_L: int,
    base_T: int,
    relative_magnitude: float,
    relative_phase: float,
) -> int:
    if relative_magnitude == 0.0:
        return SPF32_WORD_ZERO
    if not math.isfinite(relative_magnitude) or relative_magnitude < 0.0:
        return SPF32_WORD_NAN
    delta_L = _rne(SPF32_L_SCALE * math.log2(relative_magnitude))
    return _pack_scalar(_clamp_finite_l(base_L + delta_L), base_T + _rne(relative_phase * _RAD_TO_PHASE))


def _rne_half_ticks(total_half_ticks: int) -> int:
    """Round a phase given in half-ticks to whole ticks, §2.1 ties-to-even.

    ``total_half_ticks`` is ``2 * Theta_exact``.  An odd value is an exact
    half-integer tie, which §2.1 resolves on the total phase (the even
    neighbour), not on a relative delta.
    """

    if total_half_ticks % 2 == 0:
        return total_half_ticks // 2
    floor_ticks = total_half_ticks >> 1  # arithmetic shift: floor for negatives too
    return floor_ticks if floor_ticks % 2 == 0 else floor_ticks + 1


def _word_from_local_exact_phase(base_L: int, phase_ticks: int, relative_magnitude: float) -> int:
    """:func:`_word_from_local` when the output phase is already exact in ticks."""

    if relative_magnitude == 0.0:
        return SPF32_WORD_ZERO
    if not math.isfinite(relative_magnitude) or relative_magnitude < 0.0:
        return SPF32_WORD_NAN
    delta_L = _rne(SPF32_L_SCALE * math.log2(relative_magnitude))
    return _pack_scalar(_clamp_finite_l(base_L + delta_L), phase_ticks)


def exp2_neg_gap(gap: int) -> float:
    """§8.5 negative-gap exponential ``q = 2^(-gap/256)`` as shift + 256-entry ROM.

    AlgRef v0.2 §8.5 specifies the SPU form directly: high bits for a shift, the
    low 8 bits for a 256-entry fractional ``exp2`` ROM.  The index math is pure
    integer (``>>`` and ``&``) and the reconstruction is one :func:`math.ldexp`,
    so there is no floating-point division and no ``libm`` exponential on the
    path.  The ROM lives in :mod:`spf.tensor.exp2_rom` (mpmath, 100 digits,
    rounded once to float64) and the C mirror is
    ``native/include/spf32_exp2_neg_gap_rom_v02.h``.

    Scaling a normal float64 by a power of two is exact and rounding commutes
    with an exact scaling, so this is bit-identical to a correctly-rounded
    ``exp2(-gap/256)`` for every ``gap`` in ``[0, 65535]`` — the only range the
    §7 pair adders produce, since ``gap = A_L - B_L`` with ``A_L >= B_L`` over
    the 16-bit L range.  ``tests/v02/test_v02_exp2_neg_gap_rom.py`` measures that
    on all 65536 gaps against :func:`math.exp2`.

    Gaps beyond the ROM's exact regime (``gap`` so large that ``2^-(gap>>8)``
    would be subnormal, i.e. ``gap >= 261632``) are not reachable from a pair of
    SPF32 words; the shift is still applied and :func:`math.ldexp` handles the
    gradual underflow the same way ``exp2`` would.
    """

    index = int(gap)
    if index < 0:
        raise ValueError("exp2_neg_gap expects a non-negative log-code gap")
    return math.ldexp(
        EXP2_NEG_GAP_ROM[index & EXP2_NEG_GAP_ROM_MASK],
        -(index >> EXP2_NEG_GAP_SHIFT_BITS),
    )


def spf_add_complex_reference(a: int | SPF32Value, b: int | SPF32Value) -> int:
    """Reference complex pair addition using a larger-magnitude local frame."""

    wa, wb = _coerce_word(a), _coerce_word(b)
    if is_nan(wa) or is_nan(wb):
        return SPF32_WORD_NAN
    if is_zero(wa):
        return wb if not is_zero(wb) else SPF32_WORD_ZERO
    if is_zero(wb):
        return wa
    la, ta = _word_parts(wa)
    lb, tb = _word_parts(wb)
    if la < lb:
        la, lb, ta, tb = lb, la, tb, ta
    gap = la - lb
    dT = circular_diff_u16(tb, ta)
    if gap == 0 and dT == -SPF32_PHASE_HALF:
        return SPF32_WORD_ZERO
    q = exp2_neg_gap(gap)
    angle = dT * _PHASE_TO_RAD
    x = 1.0 + q * math.cos(angle)
    y = q * math.sin(angle)
    magnitude = math.hypot(x, y)
    if magnitude == 0.0:
        return SPF32_WORD_ZERO
    return _word_from_local(la, ta, magnitude, math.atan2(y, x))


def _precision_name(precision: Any) -> str:
    if precision is None:
        return "balanced"
    if isinstance(precision, PrecisionMode):
        return precision.name.lower()
    if isinstance(precision, Enum):
        return str(precision.name).lower()
    if isinstance(precision, int):
        names = {0: "fast", 1: "balanced", 2: "accurate", 3: "exact"}
        return names.get(int(precision), "balanced")
    return str(precision).lower()


def _near_cancellation(gap: int, dT: int, operation: int, precision: Any) -> bool:
    mode = _precision_name(precision)
    gap_limit = {"fast": 64, "balanced": 128, "accurate": 256, "exact": 512}.get(mode, 128)
    if gap > gap_limit:
        return False
    distance = abs(dT if operation < 0 else circular_diff_u16(dT, SPF32_PHASE_HALF))
    return distance <= 2048


def _pair_local(A_L: int, A_T: int, B_L: int, B_T: int, operation: int, precision: Any) -> int:
    """Return ``A + operation*B`` where A has the larger magnitude."""

    gap = A_L - B_L
    dT = circular_diff_u16(B_T, A_T)
    if operation > 0 and dT == 0:
        return _pack_scalar(logadd_pos(A_L, B_L).L, A_T)
    if operation < 0 and dT == 0:
        if gap == 0:
            return SPF32_WORD_ZERO
        return _pack_scalar(logsub_pos(A_L, B_L).L, A_T)
    if dT == -SPF32_PHASE_HALF:
        if operation > 0:
            if gap == 0:
                return SPF32_WORD_ZERO
            return _pack_scalar(logsub_pos(A_L, B_L).L, A_T)
        return _pack_scalar(logadd_pos(A_L, B_L).L, A_T)

    angle = dT * _PHASE_TO_RAD
    if gap == 0:
        # §7.3 equal-magnitude class.  ``1 + e^{i delta} = 2 cos(delta/2)
        # e^{i delta/2}`` and ``1 - e^{i delta} = -2i sin(delta/2)
        # e^{i delta/2}``, so the exact output phase in ticks is
        # ``Theta_A + dT/2`` (minus a quarter turn for the difference, plus one
        # when dT < 0).  For odd ``dT`` that is an exact half-integer, and
        # §2.1 requires RNE on the TOTAL phase: rounding the relative delta
        # first (``Theta_A + RNE(dT/2)``) is a different number whenever
        # ``Theta_A`` is odd.  Resolve the tie in integer half-ticks; the
        # magnitude path is unchanged.
        half = angle / 2.0
        total_half_ticks = 2 * A_T + dT
        if operation > 0:
            relative_magnitude = 2.0 * abs(math.cos(half))
        else:
            relative_magnitude = 2.0 * abs(math.sin(half))
            total_half_ticks += -SPF32_PHASE_HALF if dT > 0 else SPF32_PHASE_HALF
        return _word_from_local_exact_phase(A_L, _rne_half_ticks(total_half_ticks), relative_magnitude)

    if abs(dT) == SPF32_PHASE_QUARTER:
        q = exp2_neg_gap(gap)
        x = 1.0
        y = operation * q * math.sin(angle)
        relative_magnitude = math.sqrt(1.0 + q * q)
        return _word_from_local(A_L, A_T, relative_magnitude, math.atan2(y, x))

    if _near_cancellation(gap, dT, operation, precision):
        q = exp2_neg_gap(gap)
        if operation > 0:
            epsilon_ticks = circular_diff_u16(dT, SPF32_PHASE_HALF)
            epsilon = epsilon_ticks * _PHASE_TO_RAD
            # δ is near π.  Use expm1 and 1-cos to avoid subtracting 1-q.
            x = -math.expm1(-gap * math.log(2.0) / SPF32_L_SCALE) + q * (1.0 - math.cos(epsilon))
            y = -q * math.sin(epsilon)
        else:
            epsilon = dT * _PHASE_TO_RAD
            x = -math.expm1(-gap * math.log(2.0) / SPF32_L_SCALE) + q * (1.0 - math.cos(epsilon))
            y = -q * math.sin(epsilon)
        return _word_from_local(A_L, A_T, math.hypot(x, y), math.atan2(y, x))

    q = exp2_neg_gap(gap)
    c = math.cos(angle)
    s = math.sin(angle)
    x = 1.0 + operation * q * c
    y = operation * q * s
    return _word_from_local(A_L, A_T, math.hypot(x, y), math.atan2(y, x))


def _pair_shared_direct(
    A_L: int,
    A_T: int,
    B_L: int,
    B_T: int,
    *,
    counters: PerformanceCounters | None = None,
) -> tuple[int, int]:
    """Evaluate the ordinary FB3 path with one shared ``q, sin, cos`` set.

    The two outputs use the same local frame.  The shared products are
    explicitly materialized as ``q²``, ``q cos(delta)``, and ``q sin(delta)``;
    this is the distinction between a fused FB3 implementation and two
    independently evaluated pair additions.
    """

    gap = A_L - B_L
    dT = circular_diff_u16(B_T, A_T)
    q = exp2_neg_gap(gap)
    sine, cosine = _sincos_phase(dT)
    q2 = q * q
    qcos = q * cosine
    qsin = q * sine
    if counters is not None:
        counters.exp2_ops += 1
        counters.trig_ops += 1
        counters.atan2_ops += 2
        counters.log_ops += 2
        counters.fb3_shared_products += 3

    plus_x, plus_y = 1.0 + qcos, qsin
    minus_x, minus_y = 1.0 - qcos, -qsin
    plus_radius = math.sqrt(max(0.0, 1.0 + q2 + 2.0 * qcos))
    minus_radius = math.sqrt(max(0.0, 1.0 + q2 - 2.0 * qcos))
    return (
        _word_from_local(A_L, A_T, plus_radius, math.atan2(plus_y, plus_x)),
        _word_from_local(A_L, A_T, minus_radius, math.atan2(minus_y, minus_x)),
    )


def spf_pair_addsub_hybrid(
    u: int | SPF32Value,
    t: int | SPF32Value,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> tuple[int, int]:
    """Return ``(u+t, u-t)`` with exact phase classes and stable cancellation."""

    if counters is not None:
        counters.complex_pair_adds += 1
    wu, wt = _coerce_word(u), _coerce_word(t)
    if is_nan(wu) or is_nan(wt):
        return SPF32_WORD_NAN, SPF32_WORD_NAN
    if is_zero(wu):
        return wt, spf_neg(wt)
    if is_zero(wt):
        return wu, wu
    lu, tu = _word_parts(wu)
    lt, tt = _word_parts(wt)
    if lu >= lt:
        A_L, A_T, B_L, B_T = lu, tu, lt, tt
        u_is_A = True
    else:
        A_L, A_T, B_L, B_T = lt, tt, lu, tu
        u_is_A = False
    gap = A_L - B_L
    dT = circular_diff_u16(B_T, A_T)
    ordinary_direct = (
        gap > 0
        and dT not in {0, -SPF32_PHASE_HALF, SPF32_PHASE_QUARTER, -SPF32_PHASE_QUARTER}
        and not _near_cancellation(gap, dT, 1, precision)
        and not _near_cancellation(gap, dT, -1, precision)
    )
    if ordinary_direct:
        plus, minus_local = _pair_shared_direct(A_L, A_T, B_L, B_T, counters=counters)
    else:
        plus = _pair_local(A_L, A_T, B_L, B_T, 1, precision)
        minus_local = _pair_local(A_L, A_T, B_L, B_T, -1, precision)
    minus = minus_local if u_is_A else int(spf_neg(minus_local))
    return plus, minus


spf_add = spf_add_complex_reference
complex_add_reference = spf_add_complex_reference
pair_addsub_hybrid = spf_pair_addsub_hybrid


_PAIR_PHASE_QUANTIZATION = math.pi / SPF32_PHASE_MOD


@dataclass(frozen=True, slots=True)
class PairAdderDiagnostics:
    """§7.5 ``phase_reliable``/``phase_error`` flags for one pair addition.

    ``phase_reliable``, ``phase_error_bound`` and ``near_cancellation`` are the
    conservative combination over the two outputs (both reliable, the larger
    bound, either cancelled).  The per-output fields carry the detail.

    ``near_cancellation`` means the output magnitude fell below ``2**-12`` of
    the aligned input magnitudes, i.e. the phase of that output is not
    certified; ``stable_branch`` records whether §7.4's stable form was the
    branch actually evaluated.
    """

    phase_reliable: bool
    phase_error_bound: float
    near_cancellation: bool
    plus_phase_reliable: bool = True
    minus_phase_reliable: bool = True
    plus_phase_error_bound: float = 0.0
    minus_phase_error_bound: float = 0.0
    stable_branch: bool = False


def _pair_output_flags(radius: float, aligned: float, source_error: float) -> tuple[bool, float, bool]:
    """Return ``(reliable, phase_error_bound, near_cancellation)`` for one output."""

    if radius <= 0.0 or source_error >= radius:
        return False, math.pi, True
    phase_error = _PAIR_PHASE_QUANTIZATION + math.asin(min(1.0, source_error / radius))
    cancelled = radius < 2.0**-12 * aligned
    return (not cancelled) and _phase_reliability(phase_error), phase_error, cancelled


def spf_pair_addsub_hybrid_diagnostics(
    u: int | SPF32Value,
    t: int | SPF32Value,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> tuple[int, int, PairAdderDiagnostics]:
    """``(u+t, u-t)`` plus the §7.5 reliability flags.

    §7.5's pseudocode ends with "attach phase_reliable/error flags"; the bare
    :func:`spf_pair_addsub_hybrid` returns words only and keeps its signature.
    The bound is the input quantization ``(1+q)(dr/r + pi/65536)`` carried into
    the output phase as ``asin(err/r)``, plus the output phase tick.
    """

    plus, minus = spf_pair_addsub_hybrid(u, t, precision, counters=counters)
    wu, wt = _coerce_word(u), _coerce_word(t)
    if is_nan(wu) or is_nan(wt):
        return plus, minus, PairAdderDiagnostics(False, math.nan, False, False, False, math.nan, math.nan)
    if is_zero(wu) or is_zero(wt):
        tick = _PAIR_PHASE_QUANTIZATION
        return plus, minus, PairAdderDiagnostics(True, tick, False, True, True, tick, tick)
    lu, tu = _word_parts(wu)
    lt, tt = _word_parts(wt)
    if lu >= lt:
        A_L, A_T, B_L, B_T = lu, tu, lt, tt
    else:
        A_L, A_T, B_L, B_T = lt, tt, lu, tu
    gap = A_L - B_L
    dT = circular_diff_u16(B_T, A_T)
    q = exp2_neg_gap(gap)
    sine, cosine = _sincos_phase(dT)
    radius_plus = math.hypot(1.0 + q * cosine, q * sine)
    radius_minus = math.hypot(1.0 - q * cosine, q * sine)
    aligned = 1.0 + q
    source_error = aligned * (_LOG_QUANTIZATION_RELATIVE_ERROR + _PAIR_PHASE_QUANTIZATION)
    plus_reliable, plus_error, plus_cancelled = _pair_output_flags(radius_plus, aligned, source_error)
    minus_reliable, minus_error, minus_cancelled = _pair_output_flags(radius_minus, aligned, source_error)
    stable = _near_cancellation(gap, dT, 1, precision) or _near_cancellation(gap, dT, -1, precision)
    return (
        plus,
        minus,
        PairAdderDiagnostics(
            plus_reliable and minus_reliable,
            max(plus_error, minus_error),
            plus_cancelled or minus_cancelled,
            plus_reliable,
            minus_reliable,
            plus_error,
            minus_error,
            stable,
        ),
    )


pair_addsub_hybrid_diagnostics = spf_pair_addsub_hybrid_diagnostics


def spf_butterfly_reference(
    a: int | SPF32Value,
    b: int | SPF32Value,
    twiddle_T: int | Phase,
) -> tuple[int, int]:
    """Run the independent reference radix-2 butterfly from Algorithm §11.1."""

    phase = twiddle_T.T if isinstance(twiddle_T, Phase) else int(twiddle_T)
    rotated = spf_phase_shift(b, phase)
    return (
        spf_add_complex_reference(a, rotated),
        spf_add_complex_reference(a, spf_neg(rotated)),
    )


butterfly_reference = spf_butterfly_reference


def phase_code_from_wide(phase: int) -> int:
    """Round a wider full-turn phase accumulator to a canonical u16 phase."""

    return _rne(int(phase) / SPF32_INTERNAL_PHASE_SCALE) & SPF32_PHASE_MASK


def spf_butterfly_fb3(
    a: int | SPF32Value,
    b: int | SPF32Value,
    twiddle_T: int | Phase,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> tuple[int, int]:
    """Run the v0.2 FB3-Hybrid radix-2 butterfly.

    The twiddle is phase-only, so its log-magnitude field is never multiplied
    into ``b``. The two outputs share the hybrid local-frame work. ``counters``
    makes the fused operation visible to performance reports without changing
    the scalar result API.
    """

    if counters is not None:
        counters.fft_butterflies += 1
        counters.phase_ops += 1
    phase = twiddle_T.T if isinstance(twiddle_T, Phase) else int(twiddle_T)
    rotated = spf_phase_shift(b, phase)
    return spf_pair_addsub_hybrid(a, rotated, precision, counters=counters)


butterfly_fb3 = spf_butterfly_fb3
fb3_hybrid_butterfly = spf_butterfly_fb3
spf_butterfly_fb3_hybrid = spf_butterfly_fb3


def _bit_reverse_indices(size: int) -> np.ndarray:
    """Return deterministic radix-2 bit-reversal indices."""

    bits = int(math.log2(size))
    return np.asarray(
        [int(f"{index:0{bits}b}"[::-1], 2) if bits else 0 for index in range(size)],
        dtype=np.int64,
    )


def _fft_row_v02(
    row: np.ndarray,
    *,
    precision: Any,
    counters: PerformanceCounters | None,
) -> np.ndarray:
    """Transform one packed row with the normative FB3 radix-2 schedule."""

    n = int(row.size)
    working = np.asarray(row, dtype=np.uint32)[_bit_reverse_indices(n)].copy()
    block = 2
    while block <= n:
        half = block // 2
        twiddle_step_wide = -(SPF32_INTERNAL_PHASE_MOD // block)
        for base in range(0, n, block):
            twiddle_wide = 0
            for offset in range(half):
                first = base + offset
                second = first + half
                y0, y1 = spf_butterfly_fb3(
                    int(working[first]),
                    int(working[second]),
                    phase_code_from_wide(twiddle_wide),
                    precision,
                    counters=counters,
                )
                working[first] = y0
                working[second] = y1
                twiddle_wide += twiddle_step_wide
        block *= 2
    return working


def spf_fft_radix2(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Forward SPF-32 radix-2 FFT using the v0.2 FB3-Hybrid butterfly.

    The last axis is transformed and leading batch axes are retained.  This is
    the explicit v0.2 profile; the legacy :mod:`spf.fft` entry point remains
    available for regression comparisons.  Arrays run on word planes through
    :func:`spf.tensor.v02_planes.spf_fft_radix2` (every row and every
    butterfly of a stage as one plane operation); the scalar schedule is kept
    as :func:`spf_fft_radix2_reference` and is the bit-identity oracle.
    """

    from .v02_planes import spf_fft_radix2 as _planes_fft

    return _planes_fft(values, precision, counters=counters)


def spf_fft_radix2_reference(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Scalar (per-word Python) forward FFT; the oracle for the plane path."""

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_fft_radix2 requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_fft_radix2 requires power-of-two N, got N={n}")
    if n > SPF32_INTERNAL_PHASE_MOD:
        raise ValueError("spf_fft_radix2 requires N <= the internal phase accumulator modulus")
    rows = array.reshape(-1, n)
    output = np.empty_like(rows)
    for index, row in enumerate(rows):
        output[index] = _fft_row_v02(row, precision=precision, counters=counters)
    return output.reshape(array.shape)


def _scale_ifft_word(word: int, log_shift: int) -> int:
    if is_zero(word) or is_nan(word):
        return int(word)
    log_code, phase = _word_parts(word)
    return _pack_scalar(_clamp_finite_l(log_code - log_shift), phase)


def spf_ifft_radix2(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Inverse v0.2 radix-2 FFT via conjugation and an exact log shift.

    Runs on word planes (:func:`spf.tensor.v02_planes.spf_ifft_radix2`); the
    scalar schedule is :func:`spf_ifft_radix2_reference`.
    """

    from .v02_planes import spf_ifft_radix2 as _planes_ifft

    return _planes_ifft(values, precision, counters=counters)


def spf_ifft_radix2_reference(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Scalar inverse FFT; the oracle for the plane path."""

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_ifft_radix2 requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_ifft_radix2 requires power-of-two N, got N={n}")
    conjugated = np.asarray(spf_conj(array), dtype=np.uint32)
    transformed = spf_fft_radix2_reference(conjugated, precision=precision, counters=counters)
    log_shift = int(math.log2(n)) * SPF32_L_SCALE
    output = np.empty_like(transformed)
    for index in np.ndindex(transformed.shape):
        output[index] = _scale_ifft_word(int(spf_conj(transformed[index])), log_shift)
    return output


# ---------------------------------------------------------------------------
# Reduction statistics, routing, and pivot accumulators
# ---------------------------------------------------------------------------

# A bound evaluated in float64 has to absorb its own evaluation error to stay a
# bound.  The §10.2/§10.3 tail bounds cost at most three roundings (each
# ``exp2``, the summation, and one product), so 2**-50 of relative slack covers
# them with a 1e-15 inflation, three orders of magnitude below the 1.354720e-3
# magnitude quantization the same certificates already carry.  Without it the
# equal-phase two-term case (where the triangle inequality is exactly tight)
# reports a bound one ulp under the true dropped contribution.
_BOUND_EVALUATION_SLACK = 1.0 + 2.0**-50


def _inflate_bound(value: float) -> float:
    """Round a float64-evaluated error bound away from zero past its own error."""

    if not math.isfinite(value) or value <= 0.0:
        return value
    return value * _BOUND_EVALUATION_SLACK


@dataclass(frozen=True, slots=True)
class ReductionStats:
    """Cheap reduction metadata with explicit N-aware tail diagnostics."""

    N: int
    pivot_L: float
    dLrange: float
    dom_gap: float
    Rbar: float
    weighted_coherence: float
    tail_bound_ratio: float
    tail_bound_log2: float
    effective_gap: float
    coarse_tail_ratio: float
    total_magnitude: float
    has_nan: bool = False

    @property
    def magnitude_weighted_coherence(self) -> float:
        return self.weighted_coherence

    @property
    def weighted_phase_coherence(self) -> float:
        return self.weighted_coherence

    @property
    def tail_ratio(self) -> float:
        return self.tail_bound_ratio

    def positive_tail_error(self) -> float:
        return _inflate_bound(self.tail_bound_ratio / (1.0 + self.tail_bound_ratio))

    def complex_tail_error(self) -> float:
        if self.tail_bound_ratio >= 1.0:
            return math.inf
        return _inflate_bound(self.tail_bound_ratio / max(1.0 - self.tail_bound_ratio, 1e-300))


def _as_sequence(values: Any) -> list[Any]:
    if isinstance(values, np.ndarray):
        return list(values.reshape(-1))
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, (str, bytes)):
        return [values]
    try:
        return list(values)
    except TypeError:
        return [values]


def _word_terms(values: Any) -> tuple[list[tuple[int, int]], bool]:
    arr = np.asarray(values) if isinstance(values, np.ndarray) else None
    if arr is not None and arr.dtype == np.uint32:
        raw_words = [int(item) for item in arr.reshape(-1)]
    else:
        raw_words = _as_sequence(values)
    # Normalize even uint32 arrays.  The wire format has reserved raw L=0x7fff
    # patterns, and v0.2 reductions operate on canonical words rather than
    # accidentally treating 0x7fff0000 as a directional infinity.
    words = [_coerce_word(item) for item in raw_words]
    terms: list[tuple[int, int]] = []
    has_nan = False
    for word in words:
        L, T = _word_parts(word)
        if is_nan(word):
            has_nan = True
        elif L != SPF32_L_ZERO:
            terms.append((L, T))
    return terms, has_nan


def _typed_log_terms(
    values: Any,
    signed: bool,
    *,
    raw_log_codes: bool = False,
) -> tuple[list[tuple[int, int]], bool]:
    terms: list[tuple[int, int]] = []
    has_nan = False
    for item in _as_sequence(values):
        if signed:
            value = _coerce_reduction_logsigned(item, raw_log_codes=raw_log_codes)
            if value.is_nan:
                has_nan = True
            elif not value.is_zero:
                terms.append((value.L, value.sign))
        else:
            value = _coerce_reduction_logpos(item, raw_log_codes=raw_log_codes)
            if value.is_nan:
                has_nan = True
            elif not value.is_zero:
                terms.append((value.L, 1))
    return terms, has_nan


def _stats_from_terms(terms: list[tuple[int, int]], has_nan: bool, *, complex_terms: bool) -> ReductionStats:
    if not terms:
        return ReductionStats(
            N=0,
            pivot_L=-math.inf,
            dLrange=0.0,
            dom_gap=math.inf,
            Rbar=0.0,
            weighted_coherence=0.0,
            tail_bound_ratio=0.0,
            tail_bound_log2=-math.inf,
            effective_gap=math.inf,
            coarse_tail_ratio=0.0,
            total_magnitude=0.0,
            has_nan=has_nan,
        )
    logs = [item[0] / SPF32_L_SCALE for item in terms]
    pivot = max(logs)
    minimum = min(logs)
    gaps = sorted(logs, reverse=True)
    dom_gap = math.inf if len(gaps) < 2 else gaps[0] - gaps[1]
    magnitudes = [math.exp2(log_code - pivot) for log_code in logs]
    total = math.fsum(magnitudes)
    # §10.3: the reported tail is the sum of the non-pivot magnitudes, i.e. the
    # triangle-inequality bound U_D on the contribution a tropical selection
    # drops.  Forming it as ``fsum(all) - max`` cancels: once the tail sits
    # below the float64 ulp of the pivot the subtraction returns exactly 0
    # (L = [0, -14000] at equal phase reported 0.0 where the dropped term is
    # 2**-54.6875 * r0 = 3.447e-17 * r0), so the published field was not a
    # bound.  Sum the tail directly; all terms are non-negative, so there is
    # no cancellation and the result is >= the true dropped contribution.
    peak_index = max(range(len(magnitudes)), key=magnitudes.__getitem__)
    tail_ratio = math.fsum(
        magnitude for index, magnitude in enumerate(magnitudes) if index != peak_index
    )
    coarse_tail = 0.0 if len(terms) < 2 else (len(terms) - 1) * math.exp2(-dom_gap)
    effective_gap = math.inf if len(terms) < 2 else dom_gap - math.log2(len(terms) - 1)
    if complex_terms:
        vectors = [m * complex(*_sincos_phase(T)[::-1]) for m, (_, T) in zip(magnitudes, terms, strict=True)]
        vector_sum = complex(math.fsum(item.real for item in vectors), math.fsum(item.imag for item in vectors))
        unweighted = sum(complex(*_sincos_phase(T)[::-1]) for _, T in terms)
        weighted = abs(vector_sum) / total if total else 0.0
        Rbar = abs(unweighted) / len(terms)
    else:
        signed_sum = math.fsum(m * (1.0 if sign > 0 else -1.0) for m, (_, sign) in zip(magnitudes, terms, strict=True))
        Rbar = 1.0 if not any(sign < 0 for _, sign in terms) else abs(signed_sum) / total
        weighted = Rbar
    return ReductionStats(
        N=len(terms),
        pivot_L=pivot,
        dLrange=pivot - minimum,
        dom_gap=dom_gap,
        Rbar=float(Rbar),
        weighted_coherence=float(weighted),
        tail_bound_ratio=float(tail_ratio),
        tail_bound_log2=-math.inf if tail_ratio == 0.0 else math.log2(tail_ratio),
        effective_gap=effective_gap,
        coarse_tail_ratio=float(coarse_tail),
        total_magnitude=float(total),
        has_nan=has_nan,
    )


def _opaque_stats(count: int, *, has_nan: bool = False) -> ReductionStats:
    """Create routing metadata for ordered/associative non-SPF algebras."""

    return ReductionStats(
        N=int(count),
        pivot_L=0.0,
        dLrange=0.0,
        dom_gap=math.inf,
        Rbar=1.0,
        weighted_coherence=1.0,
        tail_bound_ratio=0.0,
        tail_bound_log2=-math.inf,
        effective_gap=math.inf,
        coarse_tail_ratio=0.0,
        total_magnitude=float(count),
        has_nan=has_nan,
    )


def _contains_nan(value: Any) -> bool:
    """Detect NaN payloads across the small typed-value/container surface."""

    if isinstance(value, (LogPos, LogSigned, SPF32Value)):
        return value.is_nan
    if isinstance(value, np.ndarray):
        return any(_contains_nan(item) for item in value.reshape(-1))
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return math.isnan(float(np.real(value))) or math.isnan(float(np.imag(value)))
    if hasattr(value, "T") and hasattr(value, "C"):
        return _contains_nan(value.T) or _contains_nan(value.C)
    return False


def _normalize_kind_name(kind: str | ValueKind) -> str:
    """Normalize public value-kind aliases to one dispatch spelling."""

    name = (kind.value if isinstance(kind, ValueKind) else str(kind)).lower().replace("-", "_")
    return {
        "logpos": "log_pos",
        "positive": "log_pos",
        "positive_real": "log_pos",
        "logsigned": "log_signed",
        "signed": "log_signed",
        "signed_real": "log_signed",
        "spf": "spf_complex",
        "complex": "spf_complex",
        "complex128": "spf_complex",
        "phase_only": "phase",
        "depth": "optical_depth",
        "transport": "affine_transport",
    }.get(name, name)


def reduction_stats(
    values: Any,
    kind: str | ValueKind = "complex",
    *,
    raw_log_codes: bool = False,
) -> ReductionStats:
    """Compute v0.2 statistics for typed values or packed SPF32 words.

    The ``raw_log_codes`` escape hatch is intentionally explicit.  Without it
    an integer in a ``LOG_POS``/``LOG_SIGNED`` reduction is rejected instead
    of being guessed to be either a log code or a packed word.
    """

    name = _normalize_kind_name(kind)
    if name in {"log_pos", "logpos", "positive", "positive_real"}:
        terms, has_nan = _typed_log_terms(values, signed=False, raw_log_codes=raw_log_codes)
        return _stats_from_terms(terms, has_nan, complex_terms=False)
    if name in {"log_signed", "logsigned", "signed", "signed_real"}:
        terms, has_nan = _typed_log_terms(values, signed=True, raw_log_codes=raw_log_codes)
        return _stats_from_terms(terms, has_nan, complex_terms=False)
    if name in {"phase", "phase_only"}:
        terms: list[tuple[int, int]] = []
        has_nan = False
        for item in _as_sequence(values):
            if _contains_nan(item):
                has_nan = True
            else:
                phase = item.T if isinstance(item, Phase) else int(item)
                terms.append((0, phase))
        return _stats_from_terms(terms, has_nan, complex_terms=True)
    if name in {"optical_depth", "depth", "tropical", "tropical_min", "tropical_max", "min", "max"}:
        items = _as_sequence(values)
        has_nan = any(_contains_nan(item) for item in items)
        return _opaque_stats(len(items), has_nan=has_nan)
    if name in {"affine_transport", "transport"}:
        items = _as_sequence(values)
        return _opaque_stats(len(items), has_nan=any(_contains_nan(item) for item in items))
    terms, has_nan = _word_terms(values)
    return _stats_from_terms(terms, has_nan, complex_terms=True)


compute_stats = reduction_stats
compute_v02_stats = reduction_stats


@dataclass(frozen=True, slots=True)
class ReductionSpec:
    """Error and determinism policy consumed by the v0.2 reducers."""

    error_budget_abs: float = 0.0
    error_budget_rel: float = 0.0
    precision_mode: Any = "balanced"
    allow_pruning: bool = False
    deterministic: bool = True
    accumulator_fraction_bits: int | None = None
    required_value_kind: str | None = None

    def __post_init__(self) -> None:
        for name in ("error_budget_abs", "error_budget_rel"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")

    @property
    def error_budget(self) -> float:
        return max(float(self.error_budget_abs), float(self.error_budget_rel))


@dataclass(slots=True)
class PerformanceCounters:
    """Transparent operation counters attached to every v0.2 reduction."""

    terms_seen: int = 0
    terms_retained: int = 0
    terms_dropped: int = 0
    terms_pruned: int = 0
    aligned_mantissas: int = 0
    positive_logadds: int = 0
    signed_logadds: int = 0
    complex_pair_adds: int = 0
    pivot_reductions: int = 0
    log_ops: int = 0
    atan2_ops: int = 0
    exp2_ops: int = 0
    trig_ops: int = 0
    complex_finalizations: int = 0
    values_materialized: int = 0
    phase_ops: int = 0
    fb3_shared_products: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    fft_butterflies: int = 0
    atom_evaluations: int = 0
    jet_samples: int = 0
    transport_compositions: int = 0
    fallback_conversions: int = 0

    @property
    def spf_word_ops(self) -> int:
        return self.terms_seen

    def as_dict(self) -> dict[str, int]:
        counts = {name: int(getattr(self, name)) for name in self.__dataclass_fields__}
        counts["spf_word_ops"] = self.spf_word_ops
        return counts

    def to_dict(self) -> dict[str, int]:
        """Return the counter mapping used by benchmark/reporting adapters."""

        return self.as_dict()


@dataclass(frozen=True, slots=True)
class ErrorLedger:
    """Additive error ledger shared by reducers and materializers.

    The components follow the v0.2 renderer contract.  Bounds are absolute
    pointwise bounds in the value domain; ``total_bound`` is their conservative
    sum.  ``stages`` retains per-stage reducer certificates without making
    callers reconstruct the reduction tree.
    """

    source_quantization: float = 0.0
    representation_fit: float = 0.0
    pruning: float = 0.0
    reducer: float = 0.0
    transform: float = 0.0
    materialization: float = 0.0
    display: float = 0.0
    stages: tuple[ErrorLedger, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_quantization",
            "representation_fit",
            "pruning",
            "reducer",
            "transform",
            "materialization",
            "display",
        ):
            value = float(getattr(self, name))
            if not math.isnan(value) and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "stages", tuple(self.stages))

    @property
    def total_bound(self) -> float:
        values = (
            self.source_quantization,
            self.representation_fit,
            self.pruning,
            self.reducer,
            self.transform,
            self.materialization,
            self.display,
        )
        if any(math.isnan(value) for value in values):
            return math.nan
        return math.fsum(values)

    @property
    def total(self) -> float:
        """Short alias for :attr:`total_bound`."""

        return self.total_bound

    @property
    def absolute_error_bound(self) -> float:
        """Certificate spelling used by reduction callers."""

        return self.total_bound

    @property
    def error_bound(self) -> float:
        return self.total_bound

    def plus(self, other: ErrorLedger) -> ErrorLedger:
        """Add two ledgers while retaining both stage histories."""

        if not isinstance(other, ErrorLedger):
            raise TypeError("ErrorLedger.plus expects another ErrorLedger")
        return ErrorLedger(
            source_quantization=self.source_quantization + other.source_quantization,
            representation_fit=self.representation_fit + other.representation_fit,
            pruning=self.pruning + other.pruning,
            reducer=self.reducer + other.reducer,
            transform=self.transform + other.transform,
            materialization=self.materialization + other.materialization,
            display=self.display + other.display,
            stages=self.stages + other.stages,
        )


ReductionCertificate = ErrorLedger


@dataclass(frozen=True, slots=True)
class MaterializationResult:
    """Materialized payload plus an optional v0.2 error certificate."""

    values: Any
    absolute_error_bound: float = 0.0
    relative_error_bound: float = 0.0
    ledger: ErrorLedger = field(default_factory=ErrorLedger)

    def __post_init__(self) -> None:
        absolute = float(self.absolute_error_bound)
        relative = float(self.relative_error_bound)
        if not math.isnan(absolute) and absolute < 0.0:
            raise ValueError("absolute_error_bound must be non-negative")
        if not math.isnan(relative) and relative < 0.0:
            raise ValueError("relative_error_bound must be non-negative")
        if not isinstance(self.ledger, ErrorLedger):
            raise TypeError("ledger must be an ErrorLedger")
        ledger_bound = self.ledger.total_bound
        if not math.isnan(ledger_bound) and (math.isnan(absolute) or ledger_bound > absolute):
            absolute = ledger_bound
        object.__setattr__(self, "absolute_error_bound", absolute)
        object.__setattr__(self, "relative_error_bound", relative)

    @property
    def value(self) -> Any:
        """Compatibility alias for scalar-oriented consumers."""

        return self.values

    @property
    def data(self) -> Any:
        return self.values

    @property
    def certificate(self) -> ErrorLedger:
        return self.ledger

    @property
    def error_ledger(self) -> ErrorLedger:
        return self.ledger

    @property
    def total_error_bound(self) -> float:
        return max(self.absolute_error_bound, self.ledger.total_bound)

    @property
    def total_bound(self) -> float:
        return self.total_error_bound

    @property
    def error_bound(self) -> float:
        return self.total_error_bound


@dataclass(frozen=True, slots=True)
class ReductionResult:
    """A typed reduction value plus its error certificate and counters."""

    value: Any
    value_kind: str
    method: str
    absolute_error_bound: float = 0.0
    relative_error_bound: float = 0.0
    phase_error_bound: float = 0.0
    phase_reliable: bool = True
    retained_terms: int = 0
    dropped_terms: int = 0
    accumulator_fraction_bits: int = 0
    saturated: bool = False
    counters: PerformanceCounters = field(default_factory=PerformanceCounters)
    ledger: ErrorLedger = field(default_factory=ErrorLedger)

    def __post_init__(self) -> None:
        if not isinstance(self.ledger, ErrorLedger):
            raise TypeError("ledger must be an ErrorLedger")
        bound = float(self.absolute_error_bound)
        ledger_bound = self.ledger.total_bound
        # Existing constructors predate ErrorLedger and populate only the
        # scalar bound.  Preserve those call sites while exposing the same
        # bound through the reducer ledger.
        if self.ledger == ErrorLedger() and (math.isnan(bound) or bound > 0.0):
            object.__setattr__(
                self,
                "ledger",
                ErrorLedger(reducer=bound),
            )
            ledger_bound = bound
        if not math.isnan(ledger_bound) and (math.isnan(bound) or ledger_bound > bound):
            object.__setattr__(self, "absolute_error_bound", ledger_bound)

    @property
    def word(self) -> int:
        if self.value_kind == "SPF_COMPLEX":
            return _coerce_word(self.value)
        raise TypeError("ReductionResult.value is not an SPF_COMPLEX word")

    @property
    def n_log_adds(self) -> int:
        return self.counters.positive_logadds + self.counters.signed_logadds

    @property
    def performance(self) -> PerformanceCounters:
        return self.counters

    @property
    def certificate(self) -> ErrorLedger:
        return self.ledger

    @property
    def error_ledger(self) -> ErrorLedger:
        return self.ledger

    @property
    def stage_certificates(self) -> tuple[ErrorLedger, ...]:
        return self.ledger.stages

    @property
    def total_error_bound(self) -> float:
        return max(self.absolute_error_bound, self.ledger.total_bound)

    @property
    def total_bound(self) -> float:
        return self.total_error_bound


def _phase_reliability(error_bound: float) -> bool:
    """Classify phase only when its explicit bound is finite and bounded."""

    return math.isfinite(error_bound) and error_bound <= PHASE_RELIABILITY_MAX_ERROR_RAD


V02ReductionStats = ReductionStats
V02ReductionResult = ReductionResult
OperationCounters = PerformanceCounters


def _coerce_spec(spec: ReductionSpec | float | None, error_budget: float | None) -> ReductionSpec:
    if isinstance(spec, ReductionSpec):
        if error_budget is None:
            return spec
        return ReductionSpec(
            error_budget_abs=spec.error_budget_abs,
            error_budget_rel=float(error_budget),
            precision_mode=spec.precision_mode,
            allow_pruning=spec.allow_pruning,
            deterministic=spec.deterministic,
            accumulator_fraction_bits=spec.accumulator_fraction_bits,
            required_value_kind=spec.required_value_kind,
        )
    if spec is not None:
        return ReductionSpec(error_budget_rel=float(spec if error_budget is None else error_budget))
    return ReductionSpec(error_budget_rel=0.0 if error_budget is None else float(error_budget))


def _validate_reduction_value_kind(
    policy: ReductionSpec,
    expected: str,
    declared: str | ValueKind | None = None,
) -> None:
    """Validate a direct reducer's declared logical value kind."""

    requested = declared if declared is not None else policy.required_value_kind
    if requested is None:
        return
    normalized = _normalize_kind_name(requested)
    if normalized != expected:
        raise TypeError(f"reducer requires {expected.upper()} values, received {requested!r}")


def _fraction_bits(spec: ReductionSpec, n: int, explicit: int | None) -> int:
    if explicit is not None:
        return max(0, int(explicit))
    if spec.accumulator_fraction_bits is not None:
        return max(0, int(spec.accumulator_fraction_bits))
    mode = _precision_name(spec.precision_mode)
    if mode == "fast":
        selected = max(12, math.ceil(math.log2(max(n, 1))) + 10)
    elif mode == "accurate":
        selected = max(24, math.ceil(math.log2(max(n, 1))) + 18)
    elif mode == "exact":
        selected = 53
    else:
        selected = max(16, math.ceil(math.log2(max(n, 1))) + 12)

    # A declared budget is a precision request, not just a pruning hint. The
    # aligned-mantissa error is at most N·2**(-F-1), so derive the minimum F
    # needed by a relative budget and use full reference precision whenever an
    # absolute budget is supplied without a scale hint.
    if spec.error_budget_rel > 0.0:
        required = math.ceil(math.log2(max(n, 1) / (2.0 * spec.error_budget_rel)))
        selected = max(selected, required)
    if spec.error_budget_abs > 0.0:
        selected = max(selected, 53)
    return selected


def _positive_prune_indices(logs: list[int], spec: ReductionSpec) -> tuple[list[int], int]:
    if not spec.allow_pruning or (spec.error_budget_abs <= 0.0 and spec.error_budget_rel <= 0.0):
        return list(range(len(logs))), 0
    order = sorted(range(len(logs)), key=lambda index: (-logs[index], index))
    pivot = max(logs)
    normalized = [math.exp2((logs[index] - pivot) / SPF32_L_SCALE) for index in order]
    prefix = 0.0
    chosen = len(order)
    for count, magnitude in enumerate(normalized, start=1):
        prefix += magnitude
        tail = math.fsum(normalized[count:])
        abs_error = tail * math.exp2(pivot / SPF32_L_SCALE)
        rel_error = tail / (prefix + tail) if prefix + tail else 0.0
        abs_ok = spec.error_budget_abs <= 0.0 or abs_error <= spec.error_budget_abs
        rel_ok = spec.error_budget_rel <= 0.0 or rel_error <= spec.error_budget_rel
        if abs_ok and rel_ok:
            chosen = count
            break
    selected = order[:chosen]
    return selected, len(order) - chosen


def reduce_logpos(
    values: Iterable[LogPos | int | float],
    spec: ReductionSpec | float | None = None,
    *,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    raw_log_codes: bool = False,
    value_kind: str | ValueKind | None = None,
) -> ReductionResult:
    """Reduce positive log values with one pivoted fixed-point accumulator.

    Bare integers are rejected by default because they are ambiguous with
    packed SPF32 words.  Use :class:`LogPos`, :class:`RawLogCode`, or the
    explicit ``raw_log_codes=True`` compatibility mode.
    """

    policy = _coerce_spec(spec, error_budget)
    _validate_reduction_value_kind(policy, ValueKind.LOG_POS.value.lower(), value_kind)
    items = _as_sequence(values)
    typed, has_nan = _typed_log_terms(items, signed=False, raw_log_codes=raw_log_codes)
    counters = PerformanceCounters(
        terms_seen=len(items), terms_retained=len(typed), bytes_read=len(items) * 2, bytes_written=2
    )
    if has_nan:
        counters.values_materialized = 1
        return ReductionResult(
            LogPos.nan(),
            "LOG_POS",
            "pivot_logpos",
            math.nan,
            math.nan,
            0.0,
            False,
            len(typed),
            0,
            0,
            False,
            counters,
        )
    if not typed:
        return ReductionResult(LogPos.zero(), "LOG_POS", "zero", 0.0, 0.0, 0.0, True, 0, 0, 0, False, counters)
    logs = [item[0] for item in typed]
    keep, dropped = _positive_prune_indices(logs, policy)
    kept_logs = [logs[index] for index in keep]
    counters.terms_retained = len(kept_logs)
    counters.terms_dropped = dropped
    counters.terms_pruned = dropped
    pivot = max(kept_logs)
    bits = _fraction_bits(policy, len(kept_logs), F if F is not None else fraction_bits)
    scale = 1 << bits
    accumulator = 0
    for log_code in kept_logs:
        accumulator += _rne(math.exp2((log_code - pivot) / SPF32_L_SCALE) * scale)
    counters.aligned_mantissas = len(kept_logs)
    counters.pivot_reductions = 1
    counters.log_ops = 1
    counters.exp2_ops = len(kept_logs)
    if accumulator == 0:
        return ReductionResult(
            LogPos.zero(),
            "LOG_POS",
            "pivot_logpos",
            0.0,
            1.0,
            0.0,
            True,
            len(kept_logs),
            dropped,
            bits,
            False,
            counters,
        )
    raw_output_L = int(pivot + _rne(SPF32_L_SCALE * math.log2(accumulator / scale)))
    output_L = _clamp_finite_l(raw_output_L)
    value = LogPos(output_L)
    kept_norm = math.fsum(math.exp2((log_code - pivot) / SPF32_L_SCALE) for log_code in kept_logs)
    # §10.2: sum the pruned terms directly.  ``full_norm - kept_norm`` cancels
    # to exactly 0 whenever the pruned tail is below the float64 ulp of the
    # retained sum, which publishes 0 as a bound on a nonzero dropped mass.
    kept_indices = set(keep)
    dropped_norm = math.fsum(
        math.exp2((log_code - pivot) / SPF32_L_SCALE)
        for index, log_code in enumerate(logs)
        if index not in kept_indices
    )
    full_norm = math.fsum([kept_norm, dropped_norm])
    linear_scale = math.exp2(pivot / SPF32_L_SCALE)
    kept_sum = linear_scale * kept_norm
    dropped_bound = linear_scale * dropped_norm
    # Each aligned mantissa is rounded to at most half an accumulator unit.
    # Bound that error in the original magnitude before adding the dropped-tail
    # bound.  Using the retained output as the sole scale would understate the
    # error when a nonzero tail is pruned.
    mantissa_error = linear_scale * len(kept_logs) * 0.5 / scale
    quantization_error = (
        abs(value.value - kept_sum)
        if not math.isfinite(value.value)
        else abs(value.value) * _LOG_QUANTIZATION_RELATIVE_ERROR
    )
    absolute_error = mantissa_error + quantization_error + dropped_bound
    full_sum = linear_scale * full_norm
    saturated = raw_output_L < SPF32_L_MIN_FINITE or raw_output_L > SPF32_L_MAX_FINITE
    if saturated or not math.isfinite(absolute_error) or not math.isfinite(full_sum):
        absolute_error = math.inf
        relative_error = 1.0
    else:
        relative_error = min(1.0, absolute_error / max(full_sum, 1e-300))
    return ReductionResult(
        value,
        "LOG_POS",
        "pivot_logpos",
        absolute_error,
        relative_error,
        0.0,
        True,
        len(kept_logs),
        dropped,
        bits,
        saturated,
        counters,
    )


def reduce_logsigned(
    values: Iterable[LogSigned | int | float],
    spec: ReductionSpec | float | None = None,
    *,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    raw_log_codes: bool = False,
    value_kind: str | ValueKind | None = None,
) -> ReductionResult:
    """Reduce signed-real log values with a pivoted signed accumulator.

    As with :func:`reduce_logpos`, raw integer codes require an explicit
    wrapper or ``raw_log_codes=True``.
    """

    policy = _coerce_spec(spec, error_budget)
    _validate_reduction_value_kind(policy, ValueKind.LOG_SIGNED.value.lower(), value_kind)
    items = _as_sequence(values)
    typed, has_nan = _typed_log_terms(items, signed=True, raw_log_codes=raw_log_codes)
    counters = PerformanceCounters(
        terms_seen=len(items), terms_retained=len(typed), bytes_read=len(items) * 3, bytes_written=3
    )
    if has_nan:
        return ReductionResult(
            LogSigned.nan(),
            "LOG_SIGNED",
            "pivot_logsigned",
            math.nan,
            math.nan,
            0.0,
            False,
            len(typed),
            0,
            0,
            False,
            counters,
        )
    if not typed:
        return ReductionResult(LogSigned.zero(), "LOG_SIGNED", "zero", 0.0, 0.0, 0.0, True, 0, 0, 0, False, counters)
    pivot = max(item[0] for item in typed)
    bits = _fraction_bits(policy, len(typed), F if F is not None else fraction_bits)
    scale = 1 << bits
    positive = 0
    negative = 0
    for log_code, sign in typed:
        mantissa = _rne(math.exp2((log_code - pivot) / SPF32_L_SCALE) * scale)
        if sign > 0:
            positive += mantissa
        else:
            negative += mantissa
    accumulator = positive - negative
    counters.aligned_mantissas = len(typed)
    counters.pivot_reductions = 1
    counters.log_ops = 1
    counters.exp2_ops = len(typed)
    linear_scale = math.exp2(pivot / SPF32_L_SCALE)
    accumulator_rounding_error = linear_scale * len(typed) * 0.5 / scale
    if accumulator == 0:
        return ReductionResult(
            LogSigned.zero(),
            "LOG_SIGNED",
            "pivot_logsigned",
            accumulator_rounding_error,
            math.inf,
            0.0,
            False,
            len(typed),
            0,
            bits,
            False,
            counters,
        )
    raw_output_L = int(pivot + _rne(SPF32_L_SCALE * math.log2(abs(accumulator) / scale)))
    output_L = _clamp_finite_l(raw_output_L)
    value = LogSigned(output_L, 1 if accumulator > 0 else -1)
    accumulator_magnitude = linear_scale * abs(accumulator) / scale
    quantization_error = accumulator_magnitude * _LOG_QUANTIZATION_RELATIVE_ERROR
    absolute_error = accumulator_rounding_error + quantization_error
    saturated = raw_output_L < SPF32_L_MIN_FINITE or raw_output_L > SPF32_L_MAX_FINITE
    lower_bound = accumulator_magnitude - accumulator_rounding_error
    relative_error = math.inf if lower_bound <= 0.0 else absolute_error / lower_bound
    if saturated:
        absolute_error = math.inf
        relative_error = math.inf
    # §9.3 ``result.phase_reliable = abs(A) > cancellation_threshold``.  The
    # sign of a signed-real accumulator is the LOG_SIGNED analogue of a phase,
    # so the threshold is the same cancellation ratio ``reduce_complex`` uses:
    # the surviving magnitude against the aligned absolute sum, gated at
    # 2**-12.  Hard-coding True certified the sign of 64 alternating +-1 terms
    # whose accumulator is one code of slack.
    cancellation_ratio = abs(accumulator) / max(float(positive + negative), 1.0)
    phase_reliable = cancellation_ratio >= 2.0**-12 and lower_bound > 0.0
    return ReductionResult(
        value,
        "LOG_SIGNED",
        "pivot_logsigned",
        absolute_error,
        relative_error,
        0.0,
        phase_reliable,
        len(typed),
        0,
        bits,
        saturated,
        counters,
    )


def _validate_complex_reduction_inputs(values: Sequence[Any]) -> None:
    """Reject values that cannot be identified as SPF complex inputs."""

    for item in values:
        if isinstance(item, RawLogCode):
            raise TypeError(
                "SPF_COMPLEX reduction received RawLogCode; use a packed SPF32 "
                "word/SPF32Value or the matching LOG_POS/LOG_SIGNED reducer"
            )
        if isinstance(item, (SPF32Value, LogPos, LogSigned)):
            continue
        if isinstance(item, (bool, np.bool_)):
            raise TypeError("SPF_COMPLEX reduction values must not be bool")
        if isinstance(item, (int, np.integer, float, np.floating, complex, np.complexfloating)):
            continue
        raise TypeError(
            "SPF_COMPLEX reduction expects packed words, SPF32Value, typed log "
            f"values, or numeric complex values; received {type(item).__name__}"
        )


def reduce_complex(
    values: Iterable[int | float | complex | SPF32Value | LogPos | LogSigned],
    spec: ReductionSpec | float | None = None,
    *,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    value_kind: str | ValueKind | None = None,
) -> ReductionResult:
    """Reduce SPF complex values with one pivoted Cartesian accumulator.

    Bare integers have one unambiguous meaning here: packed SPF32 words.  Raw
    integer log codes belong to ``LOG_POS``/``LOG_SIGNED`` and must use their
    explicit typed wrappers.
    """

    policy = _coerce_spec(spec, error_budget)
    _validate_reduction_value_kind(policy, ValueKind.SPF_COMPLEX.value.lower(), value_kind)
    items = _as_sequence(values)
    _validate_complex_reduction_inputs(items)
    terms, has_nan = _word_terms(items)
    counters = PerformanceCounters(
        terms_seen=len(items), terms_retained=len(terms), bytes_read=len(items) * 4, bytes_written=4
    )
    if has_nan:
        return ReductionResult(
            SPF32_WORD_NAN,
            "SPF_COMPLEX",
            "pivot_complex",
            math.nan,
            math.nan,
            math.nan,
            False,
            len(terms),
            0,
            0,
            False,
            counters,
        )
    if not terms:
        return ReductionResult(SPF32_WORD_ZERO, "SPF_COMPLEX", "zero", 0.0, 0.0, 0.0, True, 0, 0, 0, False, counters)
    pivot = max(item[0] for item in terms)
    bits = _fraction_bits(policy, len(terms), F if F is not None else fraction_bits)
    scale = 1 << bits
    X = 0
    Y = 0
    absolute_sum = 0
    for log_code, phase in terms:
        mantissa = _rne(math.exp2((log_code - pivot) / SPF32_L_SCALE) * scale)
        sine, cosine = _sincos_phase(phase)
        X += _rne(mantissa * cosine)
        Y += _rne(mantissa * sine)
        absolute_sum += abs(mantissa)
    counters.aligned_mantissas = len(terms)
    counters.pivot_reductions = 1
    counters.complex_finalizations = 1
    counters.phase_ops = len(terms)
    counters.log_ops = 1
    counters.atan2_ops = 1
    counters.exp2_ops = len(terms)
    counters.trig_ops = len(terms)
    linear_scale = math.exp2(pivot / SPF32_L_SCALE)
    # Each aligned mantissa is rounded before the two Cartesian projections,
    # and each projection is rounded again.  Per term, the Euclidean error is
    # bounded by ``(1/2 + sqrt(2)/2) / scale``: the first term is the shared
    # mantissa error and the second is the two independent projection errors.
    # The older certificate accounted only for the latter and was unsound for
    # deliberately small accumulator precisions.
    coordinate_error = linear_scale * len(terms) * (1.0 + math.sqrt(2.0)) * 0.5 / scale
    phase_quantization = math.pi / SPF32_PHASE_MOD
    radius = math.hypot(float(X), float(Y))
    if radius == 0.0:
        return ReductionResult(
            SPF32_WORD_ZERO,
            "SPF_COMPLEX",
            "pivot_complex",
            coordinate_error,
            math.inf,
            math.pi,
            False,
            len(terms),
            0,
            bits,
            False,
            counters,
        )
    raw_output_L = int(pivot + _rne(SPF32_L_SCALE * math.log2(radius / scale)))
    output_L = _clamp_finite_l(raw_output_L)
    output_T = _rne(math.atan2(float(Y), float(X)) * _RAD_TO_PHASE) & SPF32_PHASE_MASK
    word = _pack_scalar(output_L, output_T)
    accumulated_magnitude = linear_scale * radius / scale
    quantization_error = accumulated_magnitude * (_LOG_QUANTIZATION_RELATIVE_ERROR + phase_quantization)
    absolute_error = coordinate_error + quantization_error
    lower_bound = accumulated_magnitude - coordinate_error
    relative_error = math.inf if lower_bound <= 0.0 else absolute_error / lower_bound
    saturated = raw_output_L < SPF32_L_MIN_FINITE or raw_output_L > SPF32_L_MAX_FINITE
    if saturated:
        absolute_error = math.inf
        relative_error = math.inf
    phase_ratio = radius / max(float(absolute_sum), 1.0)
    coordinate_phase_error = (
        math.pi
        if coordinate_error >= accumulated_magnitude
        else math.asin(min(1.0, coordinate_error / max(accumulated_magnitude, 1e-300)))
    )
    phase_error = phase_quantization + coordinate_phase_error
    phase_reliable = phase_ratio >= 2.0**-12 and _phase_reliability(phase_error)
    return ReductionResult(
        word,
        "SPF_COMPLEX",
        "pivot_complex",
        absolute_error,
        relative_error,
        phase_error,
        phase_reliable,
        len(terms),
        0,
        bits,
        saturated,
        counters,
    )


class ReductionRoute(StrEnum):
    NAN = "NAN"
    ZERO = "ZERO"
    PRUNED_POSITIVE_PIVOT = "PRUNED_POSITIVE_PIVOT"
    POSITIVE_PIVOT = "POSITIVE_PIVOT"
    SIGNED_PAIR_TREE = "SIGNED_PAIR_TREE"
    SIGNED_PIVOT = "SIGNED_PIVOT"
    TROPICAL = "TROPICAL"
    HYBRID_PAIR_TREE = "HYBRID_PAIR_TREE"
    COMPLEX_PIVOT = "COMPLEX_PIVOT"
    PHASE_ACCUMULATOR = "PHASE_ACCUMULATOR"
    OPTICAL_DEPTH_SUM = "OPTICAL_DEPTH_SUM"
    INTEGER_COMPARE_TREE = "INTEGER_COMPARE_TREE"
    ORDERED_ASSOCIATIVE_TREE = "ORDERED_ASSOCIATIVE_TREE"
    REFERENCE_FALLBACK = "REFERENCE_FALLBACK"


Route = ReductionRoute
ReductionMethod = ReductionRoute


def _budget_value(error_budget: float | ReductionSpec | None) -> float:
    if isinstance(error_budget, ReductionSpec):
        return max(error_budget.error_budget_abs, error_budget.error_budget_rel)
    return 0.0 if error_budget is None else max(0.0, float(error_budget))


def _budget_parts(error_budget: float | ReductionSpec | None) -> tuple[float, float, bool]:
    if isinstance(error_budget, ReductionSpec):
        return (
            max(0.0, float(error_budget.error_budget_abs)),
            max(0.0, float(error_budget.error_budget_rel)),
            bool(error_budget.allow_pruning),
        )
    return 0.0, _budget_value(error_budget), True


def _tail_absolute_bound(stats: ReductionStats) -> float:
    if stats.tail_bound_ratio == 0.0:
        return 0.0
    try:
        return _inflate_bound(stats.tail_bound_ratio * math.exp2(stats.pivot_L))
    except OverflowError:
        return math.inf


def positive_tail_certified(stats: ReductionStats, error_budget: float | ReductionSpec | None) -> bool:
    if stats.has_nan:
        return False
    if stats.N <= 1:
        return True
    absolute_budget, relative_budget, allow_pruning = _budget_parts(error_budget)
    if not allow_pruning or (absolute_budget == 0.0 and relative_budget == 0.0):
        return False
    absolute_ok = absolute_budget == 0.0 or _tail_absolute_bound(stats) <= absolute_budget
    relative_ok = relative_budget == 0.0 or stats.positive_tail_error() <= relative_budget
    return absolute_ok and relative_ok


def complex_tail_certified(stats: ReductionStats, error_budget: float | ReductionSpec | None) -> bool:
    if stats.has_nan:
        return False
    if stats.N <= 1:
        return True
    # Dropping the tail and retaining the largest term is certified only when
    # the triangle-inequality tail bound is small relative to that term.
    absolute_budget, relative_budget, allow_pruning = _budget_parts(error_budget)
    if not allow_pruning or (absolute_budget == 0.0 and relative_budget == 0.0):
        return False
    absolute_ok = absolute_budget == 0.0 or _tail_absolute_bound(stats) <= absolute_budget
    relative_ok = relative_budget == 0.0 or stats.tail_bound_ratio <= relative_budget / (1.0 + relative_budget)
    return absolute_ok and relative_ok


positive_tail_certificate = positive_tail_certified
complex_tail_certificate = complex_tail_certified


def route_reduction(
    kind: str | ReductionStats,
    stats: ReductionStats | None = None,
    error_budget: float | ReductionSpec | None = None,
    *,
    precision_mode: Any = "balanced",
    small_tree_limit: int | None = None,
) -> ReductionRoute:
    """Choose a typed reducer using budget, precision mode, and N-aware tails.

    ``small_tree_limit`` remains an explicit override for callers that have a
    measured backend preference.  When omitted, the precision mode controls
    how much work may use pairwise nonlinear arithmetic; this keeps routing
    policy from silently collapsing to one hard-coded threshold.
    """

    if isinstance(kind, ReductionStats):
        stats = kind
        kind = "complex"
    if stats is None:
        raise TypeError("route_reduction requires ReductionStats")
    if stats.has_nan:
        return ReductionRoute.NAN
    if stats.N == 0:
        return ReductionRoute.ZERO
    n_terms = int(stats.N)
    mode = _precision_name(precision_mode)
    default_limits = {"fast": 16, "balanced": 8, "accurate": 4, "exact": 0}
    default_coherence = {"fast": 0.125, "balanced": 0.25, "accurate": 0.5, "exact": 0.75}
    if small_tree_limit is None:
        tree_limit = default_limits.get(mode, default_limits["balanced"])
    else:
        tree_limit = int(small_tree_limit)
    coherence_floor = default_coherence.get(mode, default_coherence["balanced"])
    name = _normalize_kind_name(kind)
    if name in {"log_pos", "logpos", "positive", "positive_real"}:
        return (
            ReductionRoute.PRUNED_POSITIVE_PIVOT
            if positive_tail_certified(stats, error_budget)
            else ReductionRoute.POSITIVE_PIVOT
        )
    if name in {"log_signed", "logsigned", "signed", "signed_real"}:
        if n_terms <= tree_limit and stats.weighted_coherence > coherence_floor:
            return ReductionRoute.SIGNED_PAIR_TREE
        return ReductionRoute.SIGNED_PIVOT
    if name in {"phase", "phase_only"}:
        return ReductionRoute.PHASE_ACCUMULATOR
    if name in {"optical_depth", "depth"}:
        return ReductionRoute.OPTICAL_DEPTH_SUM
    if name in {"tropical", "tropical_min", "tropical_max", "min", "max"}:
        return ReductionRoute.INTEGER_COMPARE_TREE
    if name in {"affine_transport", "transport"}:
        return ReductionRoute.ORDERED_ASSOCIATIVE_TREE
    if name in {"spf_complex", "complex", "complex128"}:
        if complex_tail_certified(stats, error_budget):
            return ReductionRoute.TROPICAL
        if n_terms <= tree_limit and stats.weighted_coherence > coherence_floor:
            return ReductionRoute.HYBRID_PAIR_TREE
        return ReductionRoute.COMPLEX_PIVOT
    return ReductionRoute.REFERENCE_FALLBACK


select_route = route_reduction
route = route_reduction


def _reduce_complex_pair_tree(items: Sequence[Any], *, precision: Any = "balanced") -> ReductionResult:
    """Reduce a short complex sequence with the shared FB3 pair operation."""

    current = [_coerce_word(item) for item in items]
    counters = PerformanceCounters(
        terms_seen=len(current),
        terms_retained=len(current),
        bytes_read=len(current) * 4,
        bytes_written=4,
    )
    if not current:
        return ReductionResult(SPF32_WORD_ZERO, "SPF_COMPLEX", ReductionRoute.ZERO.value, counters=counters)
    terms, has_nan = _word_terms(items)
    if has_nan:
        return ReductionResult(
            SPF32_WORD_NAN,
            "SPF_COMPLEX",
            ReductionRoute.NAN.value,
            absolute_error_bound=math.nan,
            relative_error_bound=math.nan,
            phase_error_bound=math.nan,
            phase_reliable=False,
            retained_terms=len(terms),
            counters=counters,
        )
    while len(current) > 1:
        next_level: list[int] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
                continue
            plus, _ = spf_pair_addsub_hybrid(
                current[index], current[index + 1], precision=precision, counters=counters
            )
            next_level.append(plus)
            counters.complex_finalizations += 1
        current = next_level
    total_abs = math.fsum(math.exp2(log_code / SPF32_L_SCALE) for log_code, _ in terms)
    pair_phase_quantization = math.pi / SPF32_PHASE_MOD
    pair_quantization = (
        _LOG_QUANTIZATION_RELATIVE_ERROR + (1.0 + _LOG_QUANTIZATION_RELATIVE_ERROR) * pair_phase_quantization
    )
    depth = max(len(items) - 1, 0)
    # A deliberately conservative certificate: every quantized pair result is
    # bounded relative to its subtree magnitude, and the N-1 exponent is a
    # safe upper bound for an arbitrary balanced/odd tree shape.
    accumulation_error = total_abs * math.expm1(depth * math.log1p(pair_quantization))
    decoded_magnitude = abs(complex(decode_complex(current[0])))
    lower_bound = decoded_magnitude - accumulation_error
    relative_error = math.inf if lower_bound <= 0.0 else accumulation_error / lower_bound
    stats = _stats_from_terms(terms, False, complex_terms=True)
    phase_error = (
        math.pi
        if lower_bound <= 0.0
        else min(
            math.pi,
            math.asin(min(1.0, accumulation_error / max(decoded_magnitude, 1e-300))) + depth * pair_phase_quantization,
        )
    )
    phase_reliable = lower_bound > 0.0 and stats.weighted_coherence >= 2.0**-12 and _phase_reliability(phase_error)
    return ReductionResult(
        current[0],
        "SPF_COMPLEX",
        ReductionRoute.HYBRID_PAIR_TREE.value,
        absolute_error_bound=accumulation_error,
        relative_error_bound=relative_error,
        phase_error_bound=phase_error,
        phase_reliable=phase_reliable,
        retained_terms=len(items),
        counters=counters,
    )


def _reduce_logsigned_pair_tree(items: Sequence[Any]) -> ReductionResult:
    """Reduce a short signed sequence with the exact-to-SPF32 ROM path."""

    current = [_coerce_logsigned(item) for item in items]
    counters = PerformanceCounters(
        terms_seen=len(current),
        terms_retained=len(current),
        bytes_read=len(current) * 3,
        bytes_written=3,
    )
    if not current:
        return ReductionResult(LogSigned.zero(), "LOG_SIGNED", ReductionRoute.ZERO.value, counters=counters)
    while len(current) > 1:
        next_level: list[LogSigned] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
            else:
                next_level.append(logadd_signed(current[index], current[index + 1], counters=counters))
        current = next_level
    total_abs = math.fsum(abs(_coerce_logsigned(item).value) for item in items)
    accumulation_error = total_abs * ((1.0 + _LOG_QUANTIZATION_RELATIVE_ERROR) ** max(len(items) - 1, 0) - 1.0)
    return ReductionResult(
        current[0],
        "LOG_SIGNED",
        ReductionRoute.SIGNED_PAIR_TREE.value,
        absolute_error_bound=accumulation_error,
        relative_error_bound=math.inf,
        phase_reliable=False,
        retained_terms=len(items),
        counters=counters,
    )


def reduce_v02(
    values: Iterable[Any],
    kind: str | ValueKind = ValueKind.SPF_COMPLEX,
    spec: ReductionSpec | float | None = None,
    *,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    raw_log_codes: bool = False,
) -> Any:
    """Route a typed reduction to its matching executable implementation."""

    items = _as_sequence(values)
    kind_name = _normalize_kind_name(kind)
    policy = _coerce_spec(spec, error_budget)
    if policy.required_value_kind is not None:
        requested = _normalize_kind_name(str(policy.required_value_kind))
        if requested != kind_name:
            raise TypeError(f"reduction requires {policy.required_value_kind}, received {kind}")
    stats = reduction_stats(items, kind_name, raw_log_codes=raw_log_codes)
    selected = route_reduction(kind_name, stats, policy, precision_mode=policy.precision_mode)
    if kind_name in {"log_pos", "logpos", "positive", "positive_real", "optical_depth"}:
        if kind_name == "optical_depth":
            counters = PerformanceCounters(terms_seen=len(items), bytes_read=len(items) * 8, bytes_written=8)
            return ReductionResult(
                reduce_optical_depth(items),
                ValueKind.OPTICAL_DEPTH.value,
                selected.value,
                retained_terms=len(items),
                counters=counters,
            )
        result = reduce_logpos(
            items,
            policy,
            F=F,
            fraction_bits=fraction_bits,
            raw_log_codes=raw_log_codes,
        )
        return replace(result, method=selected.value)
    if kind_name in {"log_signed", "logsigned", "signed", "signed_real"}:
        if selected is ReductionRoute.SIGNED_PAIR_TREE:
            return _reduce_logsigned_pair_tree(items)
        result = reduce_logsigned(
            items,
            policy,
            F=F,
            fraction_bits=fraction_bits,
            raw_log_codes=raw_log_codes,
        )
        return replace(result, method=selected.value)
    if kind_name in {"spf_complex", "complex", "complex128"}:
        if selected is ReductionRoute.HYBRID_PAIR_TREE:
            return _reduce_complex_pair_tree(items, precision=policy.precision_mode)
        if selected is ReductionRoute.TROPICAL:
            words = [_coerce_word(item) for item in items]
            winner = max(words, key=lambda word: _word_parts(word)[0])
            counters = PerformanceCounters(
                terms_seen=len(words),
                terms_retained=1,
                terms_dropped=max(0, len(words) - 1),
                terms_pruned=max(0, len(words) - 1),
                bytes_read=len(words) * 4,
                bytes_written=4,
            )
            return ReductionResult(
                winner,
                ValueKind.SPF_COMPLEX.value,
                selected.value,
                absolute_error_bound=_tail_absolute_bound(stats),
                relative_error_bound=stats.complex_tail_error(),
                phase_error_bound=(
                    math.pi if stats.tail_bound_ratio >= 1.0 else math.asin(min(1.0, stats.tail_bound_ratio))
                ),
                phase_reliable=_phase_reliability(
                    math.pi if stats.tail_bound_ratio >= 1.0 else math.asin(min(1.0, stats.tail_bound_ratio))
                ),
                retained_terms=1,
                dropped_terms=max(0, len(words) - 1),
                counters=counters,
            )
        result = reduce_complex(items, policy, F=F, fraction_bits=fraction_bits)
        return replace(result, method=selected.value)
    if kind_name in {"phase", "phase_only"}:
        counters = PerformanceCounters(terms_seen=len(items), values_materialized=len(items), bytes_read=len(items) * 2)
        return ReductionResult(reduce_phase(items), ValueKind.PHASE.value, selected.value, counters=counters)
    if kind_name in {"tropical_min", "min"}:
        return ReductionResult(tropical_min(items), ValueKind.TROPICAL_MIN.value, selected.value)
    if kind_name in {"tropical_max", "max", "tropical"}:
        return ReductionResult(tropical_max(items), ValueKind.TROPICAL_MAX.value, selected.value)
    if kind_name in {"affine_transport", "transport"}:
        return reduce_transport(items, policy)
    raise ValueError(f"no v0.2 reducer is registered for value kind {kind!r}")


reduce = reduce_v02
reduce_auto = reduce_v02
reduce_values = reduce_v02

# Names used by the renderer design document.  Keep the more descriptive
# ``reduce_log*`` spellings as the implementation names and expose these as
# thin aliases so both API vocabularies share the same validation contract.
reduce_positive = reduce_logpos
reduce_signed = reduce_logsigned


# ---------------------------------------------------------------------------
# Quadratic Jet2D fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Jet2D:
    """A scalar quadratic jet ``c + gᵀq + 1/2 qᵀHq``.

    ``value_kind`` is ``"native"`` for ordinary floating/complex jets.  For
    ``"log_pos"`` and ``"phase"`` jets, coefficients are held in widened
    fixed-point units and quantized only by :func:`jet_emit_spf32`.
    """

    value: Any = 0.0
    grad_x: Any = 0.0
    grad_y: Any = 0.0
    hess_xx: Any = 0.0
    hess_xy: Any = 0.0
    hess_yy: Any = 0.0
    guard_L: int = 8
    guard_T: int = 8
    value_kind: str = "native"
    sign: int = 1

    def __post_init__(self) -> None:
        if int(self.guard_L) < 0 or int(self.guard_T) < 0:
            raise ValueError("Jet2D guard bits must be non-negative")
        kind = str(self.value_kind).lower().replace("-", "_")
        if kind not in {"native", "real", "complex", "log", "log_pos", "log_signed", "phase"}:
            raise ValueError(f"unsupported Jet2D value_kind {self.value_kind!r}")
        signed_sign = int(self.sign)
        if kind == "log_signed" and signed_sign not in {-1, 1}:
            raise ValueError("LOG_SIGNED Jet2D sign must be +1 or -1")
        object.__setattr__(self, "guard_L", int(self.guard_L))
        object.__setattr__(self, "guard_T", int(self.guard_T))
        object.__setattr__(self, "value_kind", kind)
        object.__setattr__(self, "sign", signed_sign)

    @property
    def signed_sign(self) -> int:
        """Compatibility alias for the explicit LOG_SIGNED sign metadata."""

        return self.sign

    @property
    def c(self) -> Any:
        return self.value

    @property
    def gx(self) -> Any:
        return self.grad_x

    @property
    def gy(self) -> Any:
        return self.grad_y

    @property
    def Hxx(self) -> Any:
        return self.hess_xx

    @property
    def Hxy(self) -> Any:
        return self.hess_xy

    @property
    def Hyy(self) -> Any:
        return self.hess_yy

    def evaluate(self, x: float, y: float) -> Any:
        return jet_eval(self, x, y)

    def transform(self, A: npt.ArrayLike, b: npt.ArrayLike) -> Jet2D:
        return jet_transform(self, A, b)

    def materialize(
        self,
        shape: tuple[int, int],
        origin: tuple[float, float] = (0.0, 0.0),
        *,
        modulo: int | None = None,
        output_format: Any = None,
        counters: PerformanceCounters | None = None,
        return_result: bool = False,
        return_certificate: bool = False,
    ) -> Any:
        return jet_materialize(
            self,
            shape=shape,
            output_format=output_format,
            origin=origin,
            modulo=modulo,
            counters=counters,
            return_result=return_result,
            return_certificate=return_certificate,
        )


QuadraticJet2D = Jet2D


@dataclass(frozen=True, slots=True)
class Jet2DCubic:
    """AlgRef v0.2 §13.6 full two-dimensional cubic jet.

    §13.6: "A full cubic in two dimensions requires terms x^3, x^2 y, x y^2,
    y^3.  Axis-only cubic terms are not closed under rotation.  Full cubic
    support is optional and should use third-order finite differences."

    The polynomial is the quadratic jet plus the four plain cubic monomials — no
    factorial normalisation on the cubic part, unlike the Hessian's ``1/2`` on the
    diagonal::

        P(x, y) = c + gx x + gy y
                + 1/2 Hxx x^2 + Hxy x y + 1/2 Hyy y^2
                + Cxxx x^3 + Cxxy x^2 y + Cxyy x y^2 + Cyyy y^3

    so every cubic coefficient maps to one header tick word with no division:
    ``cubic_ramp = (Cyyy, Cxxx)`` (header dim 0 is the row/``y`` axis) and
    ``cubic_mixed = (Cxyy, Cxxy)`` — ``cubic_mixed[0]`` multiplies ``q0^2 q1 =
    y^2 x`` and ``cubic_mixed[1]`` multiplies ``q0 q1^2 = y x^2``.

    This is deliberately NOT a subclass of :class:`Jet2D`: every consumer that
    tests ``isinstance(payload, Jet2D)`` implements the §13.2 *quadratic* affine
    transform and the 3-accumulator §13.3 recurrence, and would silently drop the
    cubic part. :func:`jet_eval`, :func:`jet_forward_differences` and the
    header bridge accept it explicitly; :func:`jet_transform` refuses it (§13.2's
    ``H' = AᵀHA`` has no degree-3 companion here).
    """

    value: Any = 0.0
    grad_x: Any = 0.0
    grad_y: Any = 0.0
    hess_xx: Any = 0.0
    hess_xy: Any = 0.0
    hess_yy: Any = 0.0
    cubic_xxx: Any = 0.0
    cubic_xxy: Any = 0.0
    cubic_xyy: Any = 0.0
    cubic_yyy: Any = 0.0
    guard_L: int = 8
    guard_T: int = 8
    value_kind: str = "native"
    sign: int = 1

    def __post_init__(self) -> None:
        # Same validation contract as Jet2D; run it through a throwaway Jet2D so
        # the two cannot drift.
        probe = Jet2D(
            self.value,
            self.grad_x,
            self.grad_y,
            self.hess_xx,
            self.hess_xy,
            self.hess_yy,
            self.guard_L,
            self.guard_T,
            self.value_kind,
            self.sign,
        )
        object.__setattr__(self, "guard_L", probe.guard_L)
        object.__setattr__(self, "guard_T", probe.guard_T)
        object.__setattr__(self, "value_kind", probe.value_kind)
        object.__setattr__(self, "sign", probe.sign)

    @property
    def quadratic_part(self) -> Jet2D:
        """The §13.1 quadratic jet with the cubic terms dropped."""

        return Jet2D(
            self.value,
            self.grad_x,
            self.grad_y,
            self.hess_xx,
            self.hess_xy,
            self.hess_yy,
            self.guard_L,
            self.guard_T,
            self.value_kind,
            self.sign,
        )

    @property
    def cubic_coefficients(self) -> tuple[Any, Any, Any, Any]:
        """``(Cxxx, Cxxy, Cxyy, Cyyy)`` in the jet's widened units."""

        return (self.cubic_xxx, self.cubic_xxy, self.cubic_xyy, self.cubic_yyy)

    @property
    def is_cubic(self) -> bool:
        """True when any cubic coefficient is non-zero."""

        return any(complex(v) != 0 for v in self.cubic_coefficients)

    def evaluate(self, x: float, y: float) -> Any:
        return jet_eval(self, x, y)


CubicJet2D = Jet2DCubic

#: Jets the §13.3 recurrence and the header bridge accept.
AnyJet2D = (Jet2D, Jet2DCubic)


def _jet_cubic_terms(jet: Jet2D | Jet2DCubic) -> tuple[Any, Any, Any, Any]:
    """``(Cxxx, Cxxy, Cxyy, Cyyy)``; all zero for a plain quadratic jet."""

    if isinstance(jet, Jet2DCubic):
        return jet.cubic_coefficients
    return (0.0, 0.0, 0.0, 0.0)


def _jet_arrays(jet: Jet2D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(jet.value)
    dtype = (
        np.complex128
        if any(
            np.iscomplexobj(item) for item in (jet.value, jet.grad_x, jet.grad_y, jet.hess_xx, jet.hess_xy, jet.hess_yy)
        )
        else np.float64
    )
    c = np.asarray(jet.value, dtype=dtype)
    g = np.asarray((jet.grad_x, jet.grad_y), dtype=dtype)
    H = np.asarray(((jet.hess_xx, jet.hess_xy), (jet.hess_xy, jet.hess_yy)), dtype=dtype)
    if value.ndim != 0 or g.ndim != 1 or H.shape != (2, 2):
        raise ValueError("Jet2D coefficients must be scalar")
    return c, g, H


def jet_transform(jet: Jet2D, A: npt.ArrayLike, b: npt.ArrayLike) -> Jet2D:
    """Apply ``x = A y + b`` to a quadratic jet (AlgRef v0.2 §13.2)."""

    if isinstance(jet, Jet2DCubic) and jet.is_cubic:
        raise ValueError(
            "jet_transform implements the AlgRef v0.2 §13.2 quadratic affine "
            "transform (H' = A^T H A, g' = A^T(g + H b)); a §13.6 cubic jet has "
            "no degree-3 companion here. Transform jet.quadratic_part and carry "
            "the cubic residual separately, or subdivide (§13.5)."
        )
    c, g, H = _jet_arrays(jet)
    matrix = np.asarray(A, dtype=H.dtype)
    shift = np.asarray(b, dtype=H.dtype).reshape(-1)
    if matrix.shape != (2, 2) or shift.shape != (2,):
        raise ValueError("Jet2D affine transform requires A=(2,2) and b=(2,)")
    new_H = matrix.T @ H @ matrix
    new_g = matrix.T @ (g + H @ shift)
    new_c = c + g @ shift + 0.5 * (shift @ H @ shift)
    return Jet2D(
        new_c.item(),
        new_g[0].item(),
        new_g[1].item(),
        new_H[0, 0].item(),
        new_H[0, 1].item(),
        new_H[1, 1].item(),
        jet.guard_L,
        jet.guard_T,
        jet.value_kind,
        jet.sign,
    )


def jet_eval(jet: Jet2D | Jet2DCubic, x: float | Sequence[float], y: float | None = None) -> Any:
    """Evaluate a quadratic (or §13.6 cubic) jet at ``(x, y)``."""

    if y is None:
        point = np.asarray(x)
        if point.shape != (2,):
            raise ValueError("jet_eval expects x, y or a length-2 point")
        x_value, y_value = point[0], point[1]
    else:
        x_value, y_value = x, y
    qx = np.asarray(x_value)
    qy = np.asarray(y_value)
    quadratic = (
        jet.value
        + jet.grad_x * qx
        + jet.grad_y * qy
        + 0.5 * jet.hess_xx * qx * qx
        + jet.hess_xy * qx * qy
        + 0.5 * jet.hess_yy * qy * qy
    )
    if not isinstance(jet, Jet2DCubic):
        # Kept as a separate expression so the quadratic path is byte-identical
        # to what it evaluated before §13.6 cubic jets existed.
        return quadratic
    return (
        quadratic
        + jet.cubic_xxx * qx * qx * qx
        + jet.cubic_xxy * qx * qx * qy
        + jet.cubic_xyy * qx * qy * qy
        + jet.cubic_yyy * qy * qy * qy
    )


def _jet_forward_differences_cubic(
    jet: Jet2DCubic,
    height: int,
    width: int,
    x0: float,
    y0: float,
    dtype: Any,
    modulo: int | None,
) -> np.ndarray:
    """§13.3 extended to third order for a §13.6 full 2-D cubic jet.

    §13.6: "Full cubic support ... should use third-order finite differences."
    A cubic in ``x`` at fixed ``y`` needs four accumulators along the row
    (``value``, ``dx``, ``d2x``, ``d3x`` with ``d3x = 6 Cxxx`` constant), and
    their row-start values are themselves polynomials in ``y`` of degree 3, 2, 1
    and 0 — so the row update needs four, three, two and one accumulator
    respectively. Every seed is read off :func:`jet_eval` on the integer lattice
    (the standard Newton forward-difference table), exactly as the quadratic path
    seeds ``row_dx`` / ``row_dy``; the recurrence is then closed and exact.
    """

    out = np.empty((height, width), dtype=dtype)
    if height == 0 or width == 0:
        return out

    def value_at(i: int, j: int) -> Any:
        return jet_eval(jet, x0 + float(j), y0 + float(i))

    # Newton forward differences along x, at rows 0..2.
    def dx_at(i: int) -> Any:
        return value_at(i, 1) - value_at(i, 0)

    def d2x_at(i: int) -> Any:
        return value_at(i, 2) - 2 * value_at(i, 1) + value_at(i, 0)

    row_value = value_at(0, 0)
    ry = value_at(1, 0) - value_at(0, 0)
    ryy = value_at(2, 0) - 2 * value_at(1, 0) + value_at(0, 0)
    ryyy = value_at(3, 0) - 3 * value_at(2, 0) + 3 * value_at(1, 0) - value_at(0, 0)

    row_dx = dx_at(0)
    qy = dx_at(1) - dx_at(0)
    qyy = dx_at(2) - 2 * dx_at(1) + dx_at(0)

    row_d2x = d2x_at(0)
    sy = d2x_at(1) - d2x_at(0)

    d3x = value_at(0, 3) - 3 * value_at(0, 2) + 3 * value_at(0, 1) - value_at(0, 0)

    if modulo is not None:
        row_value %= modulo
        ry %= modulo
        ryy %= modulo
        ryyy %= modulo
        row_dx %= modulo
        qy %= modulo
        qyy %= modulo
        row_d2x %= modulo
        sy %= modulo
        d3x %= modulo

    for row in range(height):
        value = row_value
        dx = row_dx
        d2x = row_d2x
        for column in range(width):
            out[row, column] = value
            value = value + dx
            dx = dx + d2x
            d2x = d2x + d3x
            if modulo is not None:
                # §13.3: reduce every accumulator at every step — the recurrence
                # is linear so reduction commutes with it, and an unreduced
                # third-order accumulator outruns 2**53 fast.
                value %= modulo
                dx %= modulo
                d2x %= modulo
        row_value = row_value + ry
        ry = ry + ryy
        ryy = ryy + ryyy
        row_dx = row_dx + qy
        qy = qy + qyy
        row_d2x = row_d2x + sy
        if modulo is not None:
            row_value %= modulo
            ry %= modulo
            ryy %= modulo
            row_dx %= modulo
            qy %= modulo
            row_d2x %= modulo
    return out


def jet_forward_differences(
    jet: Jet2D | Jet2DCubic,
    shape: tuple[int, int],
    origin: tuple[float, float] = (0.0, 0.0),
    *,
    modulo: int | None = None,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Materialize a quadratic (or §13.6 cubic) jet with the v0.2 recurrence."""

    height, width = (int(shape[0]), int(shape[1]))
    if height < 0 or width < 0:
        raise ValueError("jet materialization shape must be non-negative")
    if counters is not None:
        counters.jet_samples += height * width
        counters.values_materialized += height * width
    x0, y0 = float(origin[0]), float(origin[1])
    dtype = (
        np.complex128
        if any(
            np.iscomplexobj(item)
            for item in (
                jet.value,
                jet.grad_x,
                jet.grad_y,
                jet.hess_xx,
                jet.hess_xy,
                jet.hess_yy,
                *_jet_cubic_terms(jet),
            )
        )
        else np.float64
    )
    if modulo is None and jet.value_kind == "phase":
        modulo = SPF32_PHASE_MOD << jet.guard_T
    if isinstance(jet, Jet2DCubic):
        return _jet_forward_differences_cubic(jet, height, width, x0, y0, dtype, modulo)
    out = np.empty((height, width), dtype=dtype)
    row_value = jet_eval(jet, x0, y0)
    row_dx = jet_eval(jet, x0 + 1.0, y0) - row_value
    row_dy = jet_eval(jet, x0, y0 + 1.0) - row_value
    if modulo is not None:
        # §13.3: reduce every accumulator at every step, not only at row
        # boundaries.  The recurrence is linear, so reduction commutes with it;
        # leaving the intra-row accumulator unreduced let it grow past 2**53
        # and lose low bits (T = 65536 with Hxx = 2*(2**24+1) was 128 ticks
        # off).  Reduced, both accumulators stay below the modulus and the
        # float64 adds are exact.
        row_value %= modulo
        row_dx %= modulo
        row_dy %= modulo
    for row in range(height):
        value = row_value
        dx = row_dx
        for column in range(width):
            out[row, column] = value
            value = value + dx
            dx = dx + jet.hess_xx
            if modulo is not None:
                value %= modulo
                dx %= modulo
        row_value = row_value + row_dy
        row_dy = row_dy + jet.hess_yy
        row_dx = row_dx + jet.hess_xy
        if modulo is not None:
            row_value %= modulo
            row_dy %= modulo
            row_dx %= modulo
    return out


def jet_emit_spf32(values: npt.ArrayLike, jet: Jet2D) -> np.ndarray:
    """Quantize a widened jet plane to canonical SPF32 words once at emission."""

    array = np.asarray(values)
    kind = jet.value_kind
    if kind in {"log", "log_pos", "log_signed"}:
        wide = np.asarray(np.real(array), dtype=np.float64)
        log_codes = np.rint(wide / (1 << jet.guard_L)).astype(np.int64)
        if kind == "log_signed":
            phase = np.full(log_codes.shape, SPF32_PHASE_HALF if jet.sign < 0 else 0, dtype=np.int64)
        else:
            phase = np.zeros(log_codes.shape, dtype=np.int64)
        return np.asarray(pack_spf32(log_codes, phase), dtype=np.uint32)
    if kind == "phase":
        wide = np.asarray(np.real(array), dtype=np.float64)
        phase = np.rint(wide / (1 << jet.guard_T)).astype(np.int64) & SPF32_PHASE_MASK
        return np.asarray(pack_spf32(np.zeros(phase.shape, dtype=np.int64), phase), dtype=np.uint32)
    return np.asarray(encode_complex(np.asarray(array, dtype=np.complex128)), dtype=np.uint32)


def _format_jet_materialized(values: Any, jet: Jet2D, output_format: Any) -> Any:
    if isinstance(output_format, str) and output_format.lower().replace("-", "_") in {
        "spf",
        "spf32",
        "word",
        "words",
        "uint32",
    }:
        return jet_emit_spf32(values, jet)
    return _format_materialized(values, output_format)


def _format_materialized(values: Any, output_format: Any) -> Any:
    """Apply the small documented output-format vocabulary at the boundary."""

    if output_format is None:
        return values
    raw_format = getattr(output_format, "value", output_format)
    name = str(raw_format).lower().replace("-", "_")

    def map_values(function: Any) -> Any:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            return function(array.item())
        result = np.empty(array.shape, dtype=object)
        for index in np.ndindex(array.shape):
            result[index] = function(array[index])
        return result

    def to_complex(value: Any) -> complex:
        if isinstance(value, LogPos):
            return complex(value.value)
        if isinstance(value, LogSigned):
            return complex(value.value)
        if isinstance(value, SPF32Value):
            return complex(decode_complex(value.word))
        return complex(value)

    def to_log_pos(value: Any) -> LogPos:
        if isinstance(value, LogPos):
            return value
        if isinstance(value, LogSigned):
            raise ValueError("LOG_POS conversion of LOG_SIGNED values requires an explicit abs/abs2 operation")
        if isinstance(value, (SPF32Value, complex, np.complexfloating)):
            raise ValueError("LOG_POS conversion of complex values requires an explicit abs/abs2 operation")
        if isinstance(value, (bool, int, np.integer, float, np.floating)):
            number = float(value)
            if math.isnan(number):
                return LogPos.nan()
            if number < 0.0:
                raise ValueError("LOG_POS conversion requires real non-negative values")
            return LogPos.from_value(number)
        raise ValueError("LOG_POS conversion requires a LogPos or real non-negative scalar")

    def to_log_signed(value: Any) -> LogSigned:
        if isinstance(value, LogSigned):
            return value
        if isinstance(value, LogPos):
            return LogSigned(value.L, 1)
        if isinstance(value, SPF32Value):
            # §4.2: certify the LOG_SIGNED conversion of an SPF_COMPLEX word on
            # the word, exactly as ``v02_planes.LogSignedPlane.from_words``
            # does.  An absolute ``|imag| < 1e-12`` test is scale-dependent, so
            # it silently accepted every small-magnitude word with an arbitrary
            # phase (0xE0C00001, L = -8000, Theta = 1, |imag| = 3.7e-14) while
            # rejecting the same phase at L = 0.
            word = value.word
            if is_nan(word):
                return LogSigned.nan()
            if is_zero(word):
                return LogSigned.zero()
            L, phase = _word_parts(word)
            if phase == SPF32_T_ZERO:
                return LogSigned(L, 1)
            if phase == SPF32_PHASE_HALF:
                return LogSigned(L, -1)
            raise ValueError(
                "LOG_SIGNED materialization requires signed-real words (phase 0 or 2**15); "
                f"received phase {phase}"
            )
        complex_value = to_complex(value)
        if not math.isfinite(complex_value.imag) or abs(complex_value.imag) > 1e-12:
            raise ValueError("LOG_SIGNED materialization requires real values")
        return LogSigned.from_value(float(complex_value.real))

    def to_phase(value: Any) -> Phase:
        if isinstance(value, Phase):
            return value
        complex_value = to_complex(value)
        if math.isnan(complex_value.real) or math.isnan(complex_value.imag):
            raise ValueError("PHASE materialization cannot represent NaN")
        if complex_value == 0.0:
            return Phase(0)
        return Phase(_rne(math.atan2(complex_value.imag, complex_value.real) * _RAD_TO_PHASE))

    if name in {"native", "default", "complex", "complex128", "dense"}:
        return np.asarray(map_values(to_complex), dtype=np.complex128)
    if name in {"real", "float", "float64"}:
        return np.asarray(map_values(lambda value: float(to_complex(value).real)), dtype=np.float64)
    if name in {"log_pos", "logpos", "positive", "positive_real"}:
        return map_values(to_log_pos)
    if name in {"log_signed", "logsigned", "signed", "signed_real"}:
        return map_values(to_log_signed)
    if name in {"phase", "phase_only"}:
        return map_values(to_phase)
    if name in {"spf_complex", "spf", "spf32", "word", "words", "uint32"}:
        return np.asarray(encode_complex(np.asarray(map_values(to_complex), dtype=np.complex128)), dtype=np.uint32)
    try:
        return np.asarray(values, dtype=output_format)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported structured-field output_format {output_format!r}") from exc


def _output_format_name(output_format: Any) -> str:
    if output_format is None:
        return ""
    return str(getattr(output_format, "value", output_format)).lower().replace("-", "_")


def _materialized_numeric(values: Any, output_format: Any) -> Any:
    """Decode a formatted payload into the comparison value domain."""

    name = _output_format_name(output_format)
    if name in {"spf", "spf_complex", "spf32", "word", "words", "uint32"}:
        return np.asarray(decode_complex(np.asarray(values, dtype=np.uint32)), dtype=np.complex128)
    if name in {"log_pos", "logpos", "positive", "positive_real"}:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            item = array.item()
            return item.value if isinstance(item, LogPos) else float(item)
        output = np.empty(array.shape, dtype=np.float64)
        for index in np.ndindex(array.shape):
            item = array[index]
            output[index] = item.value if isinstance(item, LogPos) else float(item)
        return output.astype(np.complex128)
    if name in {"log_signed", "logsigned", "signed", "signed_real"}:
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            item = array.item()
            return item.value if isinstance(item, LogSigned) else float(item)
        output = np.empty(array.shape, dtype=np.float64)
        for index in np.ndindex(array.shape):
            item = array[index]
            output[index] = item.value if isinstance(item, LogSigned) else float(item)
        return output.astype(np.complex128)
    if name in {"phase", "phase_only"}:
        # ``np.exp(1j * phase)`` is the expression spf.tensor.core.decode_complex
        # uses for a unit-magnitude word, so a phase-only payload decodes to the
        # same complex value as the same phase carried in an SPF32 word. The
        # per-element ``complex(math.cos(...), math.sin(...))`` this replaced
        # could differ from it by an ulp, platform-dependently.
        array = np.asarray(values, dtype=object)
        if array.ndim == 0:
            item = array.item()
            phase = item.T if isinstance(item, Phase) else int(item)
            return complex(np.exp(1j * (phase * _PHASE_TO_RAD)))
        output = np.empty(array.shape, dtype=np.complex128)
        for index in np.ndindex(array.shape):
            item = array[index]
            phase = item.T if isinstance(item, Phase) else int(item)
            output[index] = np.exp(1j * (phase * _PHASE_TO_RAD))
        return output
    return np.asarray(values, dtype=np.complex128)


def _max_abs_difference(left: Any, right: Any) -> float:
    try:
        difference = np.asarray(left, dtype=np.complex128) - np.asarray(right, dtype=np.complex128)
        if difference.size == 0:
            return 0.0
        maximum = float(np.max(np.abs(difference)))
    except (TypeError, ValueError, OverflowError):
        return math.inf
    return math.inf if math.isnan(maximum) else maximum


def _materialization_result(
    values: Any,
    source_values: Any,
    output_format: Any,
    ledger: ErrorLedger,
) -> MaterializationResult:
    """Attach the residual conversion error after all upstream ledger terms."""

    name = _output_format_name(output_format)
    prior_bound = ledger.total_bound
    if name in {"", "native", "default", "complex", "complex128", "dense"}:
        observed = 0.0
    else:
        observed = _max_abs_difference(_materialized_numeric(values, output_format), source_values)
    residual = observed if math.isnan(prior_bound) else max(0.0, observed - prior_bound)
    if residual:
        ledger = replace(ledger, materialization=ledger.materialization + residual)
    scale = max(_max_abs_numeric(source_values), 1e-300)
    relative = math.inf if not math.isfinite(observed) else observed / scale
    return MaterializationResult(
        values=values,
        absolute_error_bound=max(observed, ledger.total_bound) if not math.isnan(ledger.total_bound) else math.nan,
        relative_error_bound=relative,
        ledger=ledger,
    )


def _jet_source_values(values: Any, jet: Jet2D) -> Any:
    """Return the physical complex values represented by a widened jet plane."""

    array = np.asarray(values, dtype=np.float64 if not np.iscomplexobj(values) else np.complex128)
    if jet.value_kind in {"log", "log_pos", "log_signed"}:
        log_value = np.real(array) / float(1 << jet.guard_L)
        magnitude = np.exp2(log_value / SPF32_L_SCALE)
        if jet.value_kind == "log_signed" and jet.sign < 0:
            magnitude = -magnitude
        return np.asarray(magnitude, dtype=np.complex128)
    if jet.value_kind == "phase":
        phase = np.real(array) / float(1 << jet.guard_T)
        return np.exp(1j * phase * _PHASE_TO_RAD)
    return np.asarray(array, dtype=np.complex128)


def jet_materialize(
    jet: Jet2D,
    region: Any = None,
    output_format: Any = None,
    *,
    shape: tuple[int, int] | None = None,
    origin: tuple[float, float] = (0.0, 0.0),
    modulo: int | None = None,
    counters: PerformanceCounters | None = None,
    return_result: bool = False,
    return_certificate: bool = False,
) -> Any:
    """Materialize a jet using the suggested ``(region, output_format)`` API.

    ``region`` accepts ``(x0, y0, width, height)``, a mapping containing
    ``origin`` and ``shape``, or the legacy ``(height, width)`` shape.  The
    two-integer call ``jet_materialize(jet, width, height)`` is retained for
    compatibility with early reference callers.
    """

    if shape is not None:
        if region is not None:
            raise TypeError("provide either region or shape, not both")
        region = shape
    if region is None:
        raise TypeError("jet_materialize requires region or shape")
    if isinstance(region, (int, np.integer)) and isinstance(output_format, (int, np.integer)):
        shape_value = (int(output_format), int(region))
        format_value = None
    elif isinstance(region, Mapping):
        if "shape" not in region:
            raise ValueError("jet materialization region mapping requires shape")
        shape_value = tuple(int(value) for value in np.asarray(region["shape"]).reshape(-1))
        if len(shape_value) != 2:
            raise ValueError("jet materialization shape must have two dimensions")
        region_origin = region.get("origin", origin)
        origin = tuple(float(value) for value in np.asarray(region_origin).reshape(-1))
        if len(origin) != 2:
            raise ValueError("jet materialization origin must have two dimensions")
        format_value = output_format
    else:
        values = np.asarray(region)
        if values.ndim != 1:
            raise ValueError("jet materialization region must be a 1D shape or four-value rectangle")
        if values.size == 4:
            x0, y0, width, height = values.tolist()
            origin = (float(x0), float(y0))
            shape_value = (int(height), int(width))
            format_value = output_format
        elif values.size == 2:
            if isinstance(output_format, (tuple, list, np.ndarray)):
                legacy_origin = np.asarray(output_format).reshape(-1)
                if legacy_origin.size != 2:
                    raise ValueError("legacy jet origin must have two dimensions")
                origin = (float(legacy_origin[0]), float(legacy_origin[1]))
                format_value = None
            else:
                format_value = output_format
            shape_value = (int(values[0]), int(values[1]))
        else:
            raise ValueError("jet materialization region must have 2 or 4 values")
    result = jet_forward_differences(jet, shape_value, origin, modulo=modulo, counters=counters)
    formatted = _format_jet_materialized(result, jet, format_value)
    if not (return_result or return_certificate):
        return formatted
    return _materialization_result(
        formatted,
        _jet_source_values(result, jet),
        format_value,
        ErrorLedger(),
    )


jet_forward_difference_materialize = jet_forward_differences


# ---------------------------------------------------------------------------
# Gaussian–chirp atoms and mixtures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GaussianChirpAtom:
    """The structured atom ``exp(kappa + hᵀx - 1/2 xᵀCx)``.

    A formal atom may have a non-decaying real quadratic.  This is needed for
    the exact division/product algebra: division subtracts ``C`` matrices and
    is still a valid pointwise formula even when the result is not integrable.
    Direct construction remains positive-semidefinite by default; algebraic
    operations set ``allow_nondecaying`` on a formal quotient when needed.
    Operations that require a whole-space Gaussian integral (Fourier,
    convolution, and integral) enforce positive-definite ``Re(C)`` at their
    boundary with :func:`_require_integrable_atom`.
    """

    kappa: complex = 0.0 + 0.0j
    h: npt.ArrayLike = field(default_factory=lambda: np.zeros(1, dtype=np.complex128))
    C: npt.ArrayLike = field(default_factory=lambda: np.eye(1, dtype=np.complex128))
    allow_nondecaying: bool = False

    def __post_init__(self) -> None:
        vector = np.asarray(self.h, dtype=np.complex128).reshape(-1)
        matrix = np.asarray(self.C, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != vector.size:
            raise ValueError("GaussianChirpAtom requires C=(d,d) and h=(d,)")
        if not np.allclose(matrix, matrix.T):
            raise ValueError("GaussianChirpAtom C must be symmetric (not Hermitian)")
        kappa = complex(self.kappa)
        if not np.isfinite(kappa.real) or not np.isfinite(kappa.imag):
            raise ValueError("GaussianChirpAtom parameters must be finite")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(vector)):
            raise ValueError("GaussianChirpAtom parameters must be finite")
        formal = bool(self.allow_nondecaying)
        if matrix.size and not formal:
            real_part = np.real(matrix)
            minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(real_part)))
            tolerance = 1e-12 * max(1.0, float(np.linalg.norm(real_part, ord=2)))
            if minimum_eigenvalue < -tolerance:
                raise ValueError("GaussianChirpAtom requires positive-semidefinite Re(C)")
        object.__setattr__(self, "kappa", kappa)
        object.__setattr__(self, "h", vector)
        object.__setattr__(self, "C", matrix)
        object.__setattr__(self, "allow_nondecaying", formal)

    @property
    def dimension(self) -> int:
        return int(self.h.size)

    @property
    def is_decaying(self) -> bool:
        """Whether the real envelope is positive semidefinite."""

        if not self.C.size:
            return True
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(np.real(self.C))))
        tolerance = 1e-12 * max(1.0, float(np.linalg.norm(np.real(self.C), ord=2)))
        return minimum_eigenvalue >= -tolerance

    @classmethod
    def from_arrays(
        cls,
        kappa: complex,
        h: npt.ArrayLike,
        C: npt.ArrayLike,
        *,
        allow_nondecaying: bool = False,
    ) -> GaussianChirpAtom:
        """Construct an atom from explicit vector and matrix arrays."""

        return cls(
            complex(kappa),
            np.asarray(h, dtype=np.complex128),
            np.asarray(C, dtype=np.complex128),
            allow_nondecaying=allow_nondecaying,
        )

    def evaluate(self, x: npt.ArrayLike, *, counters: PerformanceCounters | None = None) -> complex | np.ndarray:
        points = np.asarray(x, dtype=np.complex128)
        if points.shape == (self.dimension,):
            if counters is not None:
                counters.atom_evaluations += 1
            exponent = self.kappa + points @ self.h - 0.5 * (points @ self.C @ points)
            return complex(np.exp(exponent))
        if points.ndim == 0 or points.shape[-1] != self.dimension:
            raise ValueError("atom evaluation points must have trailing dimension d")
        if counters is not None:
            counters.atom_evaluations += int(np.prod(points.shape[:-1], dtype=np.int64))
        exponent = (
            self.kappa
            + np.einsum("...d,d->...", points, self.h)
            - 0.5 * np.einsum("...i,ij,...j->...", points, self.C, points)
        )
        return np.exp(exponent)

    __call__ = evaluate

    def product(self, other: GaussianChirpAtom) -> GaussianChirpAtom:
        return atom_mul(self, other)

    def divide(self, other: GaussianChirpAtom) -> GaussianChirpAtom:
        return atom_div(self, other)

    def affine(self, A: npt.ArrayLike, b: npt.ArrayLike) -> GaussianChirpAtom:
        return atom_affine(self, A, b)

    def fourier(self) -> GaussianChirpAtom:
        return atom_fourier(self)

    def inverse_fourier(self) -> GaussianChirpAtom:
        return atom_inverse_fourier(self)

    def convolve(self, other: GaussianChirpAtom) -> GaussianChirpAtom:
        return atom_convolve(self, other)

    def integral(self) -> complex:
        return atom_integral(self)

    def derivative(self, x: npt.ArrayLike) -> np.ndarray:
        return atom_derivative(self, x)

    def hessian(self, x: npt.ArrayLike) -> np.ndarray:
        return atom_hessian(self, x)

    def bound(self, region: Any) -> float:
        return atom_bound(self, region)


@dataclass(frozen=True, slots=True)
class PackedGaussianChirpAtom:
    """A Gaussian--chirp header whose scalar coefficients are SPF32 words.

    The atom algebra is closed over the three coefficient blocks ``kappa``,
    ``h`` and ``C``.  Keeping those blocks as words exercises the v0.2 carrier
    without pretending that the analytic Fourier map itself is a word-plane
    operation: :meth:`to_atom` is the explicit boundary at which the
    structured transform consumes decoded coefficients.
    """

    kappa_word: int
    h_words: tuple[int, ...]
    C_words: tuple[tuple[int, ...], ...]
    allow_nondecaying: bool = False

    def __post_init__(self) -> None:
        kappa = int(self.kappa_word) & 0xFFFFFFFF
        h = tuple(int(word) & 0xFFFFFFFF for word in self.h_words)
        C = tuple(tuple(int(word) & 0xFFFFFFFF for word in row) for row in self.C_words)
        if len(C) != len(h) or any(len(row) != len(h) for row in C):
            raise ValueError("PackedGaussianChirpAtom requires C=(d,d) and h=(d,)")
        object.__setattr__(self, "kappa_word", int(pack_spf32(unpack_spf32(kappa))))
        object.__setattr__(self, "h_words", tuple(int(pack_spf32(unpack_spf32(word))) for word in h))
        object.__setattr__(
            self,
            "C_words",
            tuple(tuple(int(pack_spf32(unpack_spf32(word))) for word in row) for row in C),
        )
        object.__setattr__(self, "allow_nondecaying", bool(self.allow_nondecaying))

    @property
    def dimension(self) -> int:
        return len(self.h_words)

    @classmethod
    def from_atom(cls, atom: GaussianChirpAtom) -> PackedGaussianChirpAtom:
        """Encode a structured atom header once at an explicit producer boundary."""

        if not isinstance(atom, GaussianChirpAtom):
            raise TypeError("PackedGaussianChirpAtom.from_atom requires GaussianChirpAtom")
        encoded = encode_complex(
            np.concatenate(
                (
                    np.asarray((atom.kappa,), dtype=np.complex128),
                    np.asarray(atom.h, dtype=np.complex128),
                    np.asarray(atom.C, dtype=np.complex128).reshape(-1),
                )
            )
        )
        words = tuple(int(word) for word in np.asarray(encoded, dtype=np.uint32).reshape(-1))
        offset = 1
        h_words = words[offset : offset + atom.dimension]
        offset += atom.dimension
        matrix_words = words[offset:]
        C_words = tuple(
            matrix_words[row * atom.dimension : (row + 1) * atom.dimension]
            for row in range(atom.dimension)
        )
        return cls(words[0], h_words, C_words, atom.allow_nondecaying)

    def to_atom(self) -> GaussianChirpAtom:
        """Decode the packed header at an analytic-consumer boundary."""

        values = np.asarray(
            [self.kappa_word, *self.h_words, *(word for row in self.C_words for word in row)],
            dtype=np.uint32,
        )
        decoded = np.asarray(decode_complex(values), dtype=np.complex128)
        dimension = self.dimension
        offset = 1
        h = decoded[offset : offset + dimension]
        offset += dimension
        C = decoded[offset:].reshape(dimension, dimension)
        return GaussianChirpAtom(complex(decoded[0]), h, C, self.allow_nondecaying)

    def product(self, other: PackedGaussianChirpAtom) -> PackedGaussianChirpAtom:
        """Multiply headers through SPF32 complex word addition."""

        if not isinstance(other, PackedGaussianChirpAtom) or self.dimension != other.dimension:
            raise ValueError("packed atom product requires matching dimensions")
        add = spf_pair_addsub_hybrid
        kappa, _ = add(self.kappa_word, other.kappa_word)
        h_words = tuple(add(left, right)[0] for left, right in zip(self.h_words, other.h_words, strict=True))
        C_words = tuple(
            tuple(add(left, right)[0] for left, right in zip(row_left, row_right, strict=True))
            for row_left, row_right in zip(self.C_words, other.C_words, strict=True)
        )
        return type(self)(
            kappa,
            h_words,
            C_words,
            self.allow_nondecaying or other.allow_nondecaying,
        )

    def divide(self, other: PackedGaussianChirpAtom) -> PackedGaussianChirpAtom:
        """Divide headers through the same stable SPF32 pair adder."""

        if not isinstance(other, PackedGaussianChirpAtom) or self.dimension != other.dimension:
            raise ValueError("packed atom division requires matching dimensions")
        subtract = spf_pair_addsub_hybrid
        kappa, _ = subtract(self.kappa_word, other.kappa_word)
        h_words = tuple(subtract(left, right)[1] for left, right in zip(self.h_words, other.h_words, strict=True))
        C_words = tuple(
            tuple(subtract(left, right)[1] for left, right in zip(row_left, row_right, strict=True))
            for row_left, row_right in zip(self.C_words, other.C_words, strict=True)
        )
        return type(self)(kappa, h_words, C_words, True)


def packed_atom_mul(a: PackedGaussianChirpAtom, b: PackedGaussianChirpAtom) -> PackedGaussianChirpAtom:
    """Packed SPF32 Gaussian-header product."""

    return a.product(b)


def packed_atom_div(a: PackedGaussianChirpAtom, b: PackedGaussianChirpAtom) -> PackedGaussianChirpAtom:
    """Packed SPF32 Gaussian-header quotient."""

    return a.divide(b)


def atom_mul(a: GaussianChirpAtom, b: GaussianChirpAtom) -> GaussianChirpAtom:
    """Pointwise Gaussian–chirp product."""

    if a.dimension != b.dimension:
        raise ValueError("atom product requires matching dimensions")
    return GaussianChirpAtom(
        a.kappa + b.kappa,
        a.h + b.h,
        a.C + b.C,
        allow_nondecaying=a.allow_nondecaying or b.allow_nondecaying,
    )


def atom_div(a: GaussianChirpAtom, b: GaussianChirpAtom) -> GaussianChirpAtom:
    """Pointwise Gaussian–chirp division."""

    if a.dimension != b.dimension:
        raise ValueError("atom division requires matching dimensions")
    return GaussianChirpAtom(a.kappa - b.kappa, a.h - b.h, a.C - b.C, allow_nondecaying=True)


def atom_affine(a: GaussianChirpAtom, A: npt.ArrayLike, b: npt.ArrayLike) -> GaussianChirpAtom:
    """Transform an atom under ``x = A y + b``."""

    matrix = np.asarray(A, dtype=np.complex128)
    shift = np.asarray(b, dtype=np.complex128).reshape(-1)
    if matrix.shape != (a.dimension, a.dimension) or shift.shape != (a.dimension,):
        raise ValueError("atom affine transform dimensions do not match")
    new_C = matrix.T @ a.C @ matrix
    new_h = matrix.T @ (a.h - a.C @ shift)
    new_kappa = a.kappa + a.h @ shift - 0.5 * (shift @ a.C @ shift)
    return GaussianChirpAtom(new_kappa, new_h, new_C, allow_nondecaying=a.allow_nondecaying)


def atom_fourier(a: GaussianChirpAtom) -> GaussianChirpAtom:
    """Return the analytic forward Fourier-transform atom under ``2π``."""

    return _atom_fourier_with_sign(a, -1.0)


def _atom_fourier_with_sign(a: GaussianChirpAtom, sign: float) -> GaussianChirpAtom:
    """Transform an atom with kernel ``exp(sign*i*2π*kᵀx)``."""

    _require_integrable_atom(a)
    inverse = np.linalg.inv(a.C)
    log_determinant = _continuous_logdet_positive_real(a.C)
    dimension = a.dimension
    new_C = (2.0 * math.pi) ** 2 * inverse
    new_h = sign * 1j * 2.0 * math.pi * (inverse @ a.h)
    new_kappa = (
        a.kappa
        + (dimension / 2.0) * math.log(2.0 * math.pi)
        - 0.5 * log_determinant
        + 0.5 * (a.h @ inverse @ a.h)
    )
    return GaussianChirpAtom(complex(new_kappa), new_h, new_C)


def atom_inverse_fourier(a: GaussianChirpAtom) -> GaussianChirpAtom:
    """Return the analytic inverse Fourier-transform atom under ``2π``."""

    return _atom_fourier_with_sign(a, 1.0)


def atom_convolve(a: GaussianChirpAtom, b: GaussianChirpAtom) -> GaussianChirpAtom:
    """Convolve two Gaussian–chirp atoms through Fourier closure."""

    return atom_inverse_fourier(atom_mul(atom_fourier(a), atom_fourier(b)))


def atom_pixel_integral(a: GaussianChirpAtom, footprint: GaussianChirpAtom) -> complex:
    """Integrate an atom against an atom-compatible pixel footprint."""

    return atom_integral(atom_mul(a, footprint))


atom_pixel_footprint = atom_pixel_integral


def atom_integral(a: GaussianChirpAtom) -> complex:
    """Analytic whole-space integral of a convergent atom."""

    _require_integrable_atom(a)
    inverse = np.linalg.inv(a.C)
    log_determinant = _continuous_logdet_positive_real(a.C)
    log_integral = (
        a.kappa
        + (a.dimension / 2.0) * math.log(2.0 * math.pi)
        - 0.5 * log_determinant
        + 0.5 * (a.h @ inverse @ a.h)
    )
    return complex(np.exp(log_integral))


def _continuous_logdet_positive_real(matrix: np.ndarray) -> complex:
    """Return the analytic log-determinant on ``Re(matrix) > 0``.

    ``log(det(C))`` takes the principal branch *after* multiplying the
    eigenvalues.  In dimensions greater than one that determinant can cross
    the negative real axis even though every eigenvalue remains in the open
    right half-plane, introducing a spurious ``2πi`` jump.  The integrable
    Gaussian domain is connected and has a unique branch continued from real
    positive-definite matrices.  Summing each eigenvalue's principal log
    computes that branch without forming a potentially over/underflowing
    determinant.
    """

    eigenvalues = np.linalg.eigvals(np.asarray(matrix, dtype=np.complex128))
    if not np.all(np.isfinite(eigenvalues)) or np.any(np.real(eigenvalues) <= 0.0):
        raise ValueError("Gaussian--chirp logdet requires eigenvalues in the open right half-plane")
    return complex(np.sum(np.log(eigenvalues), dtype=np.complex128))


def _require_integrable_atom(a: GaussianChirpAtom) -> None:
    """Require a strictly decaying real envelope for whole-space integrals."""

    real_part = np.real(a.C)
    if not real_part.size:
        return
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(real_part)))
    tolerance = 1e-12 * max(1.0, float(np.linalg.norm(real_part, ord=2)))
    if minimum_eigenvalue <= tolerance:
        raise ValueError("whole-space atom transform/integral requires positive-definite Re(C)")


def atom_derivative(a: GaussianChirpAtom, x: npt.ArrayLike) -> np.ndarray:
    """Analytic gradient ``(h - Cx) ψ(x)``."""

    point = np.asarray(x, dtype=np.complex128).reshape(-1)
    if point.size != a.dimension:
        raise ValueError("atom derivative point has the wrong dimension")
    return (a.h - a.C @ point) * a.evaluate(point)


def atom_hessian(a: GaussianChirpAtom, x: npt.ArrayLike) -> np.ndarray:
    """Analytic Hessian ``((h-Cx)(h-Cx)ᵀ-C) ψ(x)``."""

    point = np.asarray(x, dtype=np.complex128).reshape(-1)
    factor = a.h - a.C @ point
    return (np.outer(factor, factor) - a.C) * a.evaluate(point)


def atom_to_jet(
    atom: GaussianChirpAtom,
    center: npt.ArrayLike = (0.0, 0.0),
) -> Jet2D:
    """Lower a two-dimensional atom to its second-order Taylor jet.

    The returned jet is expressed in the local coordinate ``q`` around
    ``center``: ``atom(center + q)`` is approximated by the value, gradient,
    and Hessian stored in the jet.  This is the reference counterpart of the
    renderer's ``ATOM_TO_JET`` lowering; higher-dimensional atoms must be
    reduced or sampled by a caller before this 2-D operation is requested.
    """

    if atom.dimension != 2:
        raise ValueError("atom_to_jet currently requires a two-dimensional atom")
    point = np.asarray(center, dtype=np.complex128).reshape(-1)
    if point.shape != (2,):
        raise ValueError("atom_to_jet center must be a two-vector")
    value = atom.evaluate(point)
    gradient = atom.derivative(point)
    hessian = atom.hessian(point)
    return Jet2D(
        value=value,
        grad_x=gradient[0],
        grad_y=gradient[1],
        hess_xx=hessian[0, 0],
        hess_xy=hessian[0, 1],
        hess_yy=hessian[1, 1],
    )


class AtomBound(float):
    """Float-compatible atom bound carrying the maximizing point and log bound."""

    def __new__(cls, value: float, log_value: float, point: npt.ArrayLike) -> AtomBound:
        instance = float.__new__(cls, value)
        instance.log_value = float(log_value)
        instance.point = np.asarray(point, dtype=float)
        return instance


def _box_bounds(region: Any, dimension: int) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(region, Mapping):
        if "lo" in region and "hi" in region:
            lower = np.asarray(region["lo"], dtype=float)
            upper = np.asarray(region["hi"], dtype=float)
        elif "lower" in region and "upper" in region:
            lower = np.asarray(region["lower"], dtype=float)
            upper = np.asarray(region["upper"], dtype=float)
        else:
            raise ValueError("region mapping requires 'lo'/'hi' or 'lower'/'upper'")
    else:
        array = np.asarray(region, dtype=float)
        # A pair of Python sequences is conventionally (lower, upper).  A
        # NumPy matrix with one interval per row is conventionally (d, 2).
        # This explicit split resolves the otherwise irreducibly ambiguous
        # (2, 2) case while retaining both documented forms.
        if isinstance(region, (tuple, list)) and len(region) == 2:
            candidate_lower = np.asarray(region[0], dtype=float)
            candidate_upper = np.asarray(region[1], dtype=float)
            if candidate_lower.shape == (dimension,) and candidate_upper.shape == (dimension,):
                lower, upper = candidate_lower, candidate_upper
            else:
                raise ValueError("region bounds must each have shape (d,)")
        elif array.shape == (dimension, 2):
            lower, upper = array[:, 0], array[:, 1]
        elif array.shape == (2, dimension):
            lower, upper = array[0], array[1]
        else:
            raise ValueError("region must have shape (d,2), (2,d), or {'lo','hi'}")
    if lower.shape != (dimension,) or upper.shape != (dimension,):
        raise ValueError("region dimensions do not match atom")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("region bounds must be finite")
    return np.minimum(lower, upper), np.maximum(lower, upper)


def _real_exponent(a: GaussianChirpAtom, point: np.ndarray) -> float:
    real_C = np.real(a.C)
    real_h = np.real(a.h)
    return float(np.real(a.kappa) + real_h @ point - 0.5 * (point @ real_C @ point))


def _quadratic_box_max(a: GaussianChirpAtom, lower: np.ndarray, upper: np.ndarray) -> tuple[float, np.ndarray]:
    """Maximize the real quadratic exponent over a closed axis-aligned box.

    A quadratic has a stationary point on every active face that can contain
    an interior maximizer. Enumerating all ``lower/free/upper`` active-set
    patterns therefore covers corners, edges, and higher-dimensional faces;
    the old implementation only enumerated the latter explicitly in 2-D.
    This is intentionally a reference implementation for the small dimensions
    used by the atom API, not a high-dimensional optimizer.
    """

    dimension = a.dimension
    real_C = np.real(a.C)
    real_h = np.real(a.h)
    candidates: list[np.ndarray] = []
    coordinate_scale = max(
        1.0,
        float(np.max(np.abs(lower))) if lower.size else 0.0,
        float(np.max(np.abs(upper))) if upper.size else 0.0,
    )
    tolerance = 1e-10 * coordinate_scale
    # state=-1 fixes a coordinate at the lower face, state=1 at the upper
    # face, and state=0 leaves it free for the stationary-face solve.
    for state in product((-1, 0, 1), repeat=dimension):
        active = np.asarray([index for index, value in enumerate(state) if value], dtype=int)
        free = np.asarray([index for index, value in enumerate(state) if not value], dtype=int)
        point = np.empty(dimension, dtype=float)
        if active.size:
            point[active] = np.where(np.asarray(state, dtype=int)[active] < 0, lower[active], upper[active])
        if free.size:
            matrix = real_C[np.ix_(free, free)]
            rhs = real_h[free]
            if active.size:
                rhs = rhs - real_C[np.ix_(free, active)] @ point[active]
            try:
                free_values = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError:
                free_values, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
                residual = np.linalg.norm(matrix @ free_values - rhs, ord=np.inf)
                if residual > tolerance:
                    continue
            if np.any(free_values < lower[free] - tolerance) or np.any(free_values > upper[free] + tolerance):
                continue
            point[free] = np.clip(free_values, lower[free], upper[free])
        candidates.append(point)
    values = [_real_exponent(a, point) for point in candidates]
    winner = int(np.argmax(values))
    return values[winner], candidates[winner]


def atom_bound(a: GaussianChirpAtom, region: Any) -> AtomBound:
    """Return a float-compatible certified magnitude upper bound over a box."""

    lower, upper = _box_bounds(region, a.dimension)
    log_value, point = _quadratic_box_max(a, lower, upper)
    try:
        bound = math.exp(log_value)
    except OverflowError:
        bound = math.inf
    return AtomBound(bound, log_value, point)


def atom_bound_log(a: GaussianChirpAtom, region: Any) -> float:
    return float(atom_bound(a, region).log_value)


@dataclass(frozen=True, slots=True)
class AtomMixture:
    """A deferred symbolic sum of Gaussian–chirp atoms."""

    atoms: tuple[GaussianChirpAtom, ...] = ()
    residual: Any = None
    dropped_bound: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "atoms", tuple(self.atoms))
        try:
            bound = float(self.dropped_bound)
        except (TypeError, ValueError) as exc:
            raise ValueError("dropped_bound must be a finite non-negative number") from exc
        if not math.isfinite(bound) or bound < 0.0:
            raise ValueError("dropped_bound must be a finite non-negative number")
        object.__setattr__(self, "dropped_bound", bound)

    def add(self, other: AtomMixture) -> AtomMixture:
        return mixture_add(self, other)

    def prune(self, region: Any, error_budget: float) -> AtomMixture:
        return mixture_prune(self, region, error_budget)

    def merge_equal(self) -> AtomMixture:
        return mixture_merge_equal(self)

    def evaluate(
        self,
        points: npt.ArrayLike,
        *,
        value_kind: str | ValueKind | None = None,
        spec: ReductionSpec | float | None = None,
        counters: PerformanceCounters | None = None,
    ) -> complex | np.ndarray:
        return mixture_evaluate(self, points, value_kind=value_kind, spec=spec, counters=counters)

    def materialize(
        self,
        region: Any,
        output_format: Any = None,
        *,
        spec: ReductionSpec | float | None = None,
        counters: PerformanceCounters | None = None,
        return_result: bool = False,
        return_certificate: bool = False,
    ) -> Any:
        return mixture_materialize(
            self,
            region,
            output_format,
            spec=spec,
            counters=counters,
            return_result=return_result,
            return_certificate=return_certificate,
        )


def _residual_value(residual: Any, points: Any) -> Any:
    """Evaluate/broadcast a residual to the shape implied by ``points``."""

    value = residual(points) if callable(residual) else residual
    point_array = np.asarray(points)
    output_shape = point_array.shape[:-1] if point_array.ndim > 1 else ()
    if not output_shape:
        array = np.asarray(value)
        if array.ndim == 0:
            return array.item()
        if array.size == 1:
            return array.reshape(-1)[0]
        raise ValueError("residual does not broadcast to a single evaluation point")
    array = np.asarray(value)
    try:
        return np.broadcast_to(array, output_shape)
    except ValueError:
        # Callers commonly retain a singleton channel/coordinate axis, e.g.
        # ``(N, 1)`` for one residual per point.  Remove only singleton axes
        # before retrying NumPy's normal broadcast rules; non-singleton shape
        # mismatches still fail explicitly.
        try:
            return np.broadcast_to(np.squeeze(array), output_shape)
        except ValueError as exc:
            raise ValueError("residual does not broadcast to evaluation shape") from exc


def _add_residual(left: Any, right: Any) -> Any:
    if left is None:
        return right
    if right is None:
        return left
    if callable(left) and callable(right):
        return lambda points: left(points) + right(points)
    if callable(left) or callable(right):
        return lambda points: _residual_value(left, points) + _residual_value(right, points)
    return left + right


def mixture_add(a: AtomMixture, b: AtomMixture) -> AtomMixture:
    """Concatenate atom lists and defer numeric addition."""

    return AtomMixture(a.atoms + b.atoms, _add_residual(a.residual, b.residual), a.dropped_bound + b.dropped_bound)


def mixture_prune(mixture: AtomMixture, region: Any, error_budget: float) -> AtomMixture:
    """Drop atoms whose summed tile bounds fit within ``error_budget``."""

    budget = max(0.0, float(error_budget))
    if budget == 0.0 or not mixture.atoms:
        return mixture
    existing_bound = float(mixture.dropped_bound)
    if math.isnan(existing_bound) or existing_bound < 0.0:
        raise ValueError("mixture dropped_bound must be a non-negative number")
    if math.isinf(existing_bound) or existing_bound >= budget:
        return mixture
    remaining_budget = budget - existing_bound
    bounds = [float(atom_bound(atom, region)) for atom in mixture.atoms]
    order = sorted(range(len(bounds)), key=lambda index: (bounds[index], index))
    dropped: set[int] = set()
    cumulative_bounds: list[float] = []
    for index in order:
        if not math.isfinite(bounds[index]):
            continue
        candidate_bounds = [*cumulative_bounds, bounds[index]]
        candidate = math.fsum(candidate_bounds)
        if candidate > remaining_budget:
            break
        dropped.add(index)
        cumulative_bounds.append(bounds[index])
    kept = tuple(atom for index, atom in enumerate(mixture.atoms) if index not in dropped)
    cumulative = math.fsum(cumulative_bounds)
    return AtomMixture(kept, mixture.residual, mixture.dropped_bound + cumulative)


def _complex_logsum(values: Sequence[complex]) -> complex | None:
    """Return a stable complex log of a finite coefficient sum."""

    if not values:
        return None
    scale = max(value.real for value in values)
    scaled = sum((cmath.exp(value - scale) for value in values), 0.0 + 0.0j)
    if scaled == 0.0:
        return None
    return complex(scale + cmath.log(scaled))


def _kappa_coefficients_cancel(left: complex, right: complex) -> bool:
    """Recognize exact opposite exponential coefficients from their logs.

    Computing ``exp(i*pi)`` in binary floating point leaves a tiny imaginary
    residue.  The atom parameters retain the logarithmic coefficients, so an
    exact real-part match and a half-turn phase difference can be recognized
    without introducing a tolerance-based merge.
    """

    if left.real != right.real:
        return False
    return abs(math.remainder(left.imag - right.imag, _TAU)) == math.pi


def mixture_merge_equal(mixture: AtomMixture) -> AtomMixture:
    """Exactly merge atoms with identical ``(h, C)`` parameters.

    Coefficients are combined as complex exponentials.  No tolerance-based
    grouping is performed, and exactly cancelling groups are removed.
    """

    groups: dict[tuple[Any, ...], tuple[GaussianChirpAtom, list[complex], list[complex]]] = {}
    order: list[tuple[Any, ...]] = []
    for atom in mixture.atoms:
        key = (atom.dimension, atom.h.dtype.str, atom.h.tobytes(), atom.C.dtype.str, atom.C.tobytes())
        if key not in groups:
            groups[key] = (atom, [], [])
            order.append(key)
        groups[key][1].append(atom.kappa)
        groups[key][2].append(atom.kappa)
    merged: list[GaussianChirpAtom] = []
    for key in order:
        atom, coefficients, kappas = groups[key]
        cancelled: set[int] = set()
        for left_index, left in enumerate(kappas):
            if left_index in cancelled:
                continue
            for right_index in range(left_index + 1, len(kappas)):
                if right_index in cancelled:
                    continue
                if _kappa_coefficients_cancel(left, kappas[right_index]):
                    cancelled.add(left_index)
                    cancelled.add(right_index)
                    break
        if cancelled:
            coefficients = [coefficient for index, coefficient in enumerate(coefficients) if index not in cancelled]
        kappa = _complex_logsum(coefficients)
        if kappa is not None:
            merged.append(
                GaussianChirpAtom(
                    kappa,
                    atom.h,
                    atom.C,
                    allow_nondecaying=atom.allow_nondecaying,
                )
            )
    return AtomMixture(tuple(merged), mixture.residual, mixture.dropped_bound)


def mixture_evaluate(
    mixture: AtomMixture,
    points: npt.ArrayLike,
    *,
    value_kind: str | ValueKind | None = None,
    spec: ReductionSpec | float | None = None,
    counters: PerformanceCounters | None = None,
) -> complex | np.ndarray:
    """Evaluate a mixture at one point or an array of points."""

    if value_kind is not None and str(value_kind).lower().replace("-", "_") in {"spf", "spf_complex", "word", "words"}:
        return mixture_evaluate_spf(mixture, points, spec=spec, counters=counters)

    point_array = np.asarray(points)
    if not mixture.atoms:
        shape = point_array.shape[:-1] if point_array.ndim > 1 else ()
        result: complex | np.ndarray = np.zeros(shape, dtype=np.complex128)
    else:
        terms = [np.asarray(atom.evaluate(points, counters=counters), dtype=np.complex128) for atom in mixture.atoms]
        if point_array.ndim == 1:
            result = sum((complex(term) for term in terms), 0.0 + 0.0j)
        else:
            stacked = np.stack(terms, axis=0)
            result = np.asarray(
                np.sum(stacked.real, axis=0, dtype=np.float64) + 1j * np.sum(stacked.imag, axis=0, dtype=np.float64),
                dtype=np.complex128,
            )
    if mixture.residual is not None:
        result = result + _residual_value(mixture.residual, points)
    return result


def mixture_evaluate_spf(
    mixture: AtomMixture,
    points: npt.ArrayLike,
    *,
    spec: ReductionSpec | float | None = None,
    counters: PerformanceCounters | None = None,
) -> np.ndarray | int:
    """Evaluate and route the final mixture sum through ``SPF_COMPLEX``."""

    point_array = np.asarray(points)
    terms = [np.asarray(atom.evaluate(points, counters=counters), dtype=np.complex128) for atom in mixture.atoms]
    residual = None
    if mixture.residual is not None:
        residual = np.asarray(_residual_value(mixture.residual, points), dtype=np.complex128)

    def reduce_point(index: tuple[int, ...] | None) -> int:
        words: list[int] = []
        for term in terms:
            value = term if index is None or term.ndim == 0 else term[index]
            words.append(int(encode_complex(value)))
        if residual is not None:
            value = residual if index is None or residual.ndim == 0 else residual[index]
            words.append(int(encode_complex(value)))
        result = reduce_v02(words, kind=ValueKind.SPF_COMPLEX, spec=spec)
        if counters is not None:
            for name in result.counters.__dataclass_fields__:
                setattr(counters, name, getattr(counters, name) + getattr(result.counters, name))
        return int(result.value)

    if point_array.ndim == 1:
        return reduce_point(None)
    output = np.empty(point_array.shape[:-1], dtype=np.uint32)
    for index in np.ndindex(output.shape):
        output[index] = reduce_point(index)
    return output


def _region_grid_points(region: Any, dimension: int) -> np.ndarray:
    """Convert a structured-field region into a point grid."""

    origin: tuple[float, ...]
    shape: tuple[int, ...]
    if isinstance(region, Mapping):
        if "points" in region:
            points = np.asarray(region["points"], dtype=np.complex128)
            if points.shape[-1] != dimension:
                raise ValueError("materialization points do not match atom dimension")
            return points
        if "shape" not in region:
            raise ValueError("materialization region mapping requires shape or points")
        shape = tuple(int(value) for value in np.asarray(region["shape"]).reshape(-1))
        origin = tuple(float(value) for value in np.asarray(region.get("origin", (0.0,) * dimension)).reshape(-1))
    else:
        values = np.asarray(region)
        if (
            values.ndim == 1
            and values.size == dimension
            and np.all(np.equal(values, np.rint(values)))
            and np.all(values >= 0)
        ):
            shape = tuple(int(value) for value in values.tolist())
            origin = (0.0,) * dimension
        elif values.ndim >= 1 and values.shape[-1] == dimension:
            return np.asarray(values, dtype=np.complex128)
        if dimension == 2 and values.ndim == 1 and values.size == 4:
            x0, y0, width, height = values.tolist()
            origin = (float(x0), float(y0))
            shape = (int(height), int(width))
        elif values.ndim == 1 and values.size == dimension:
            shape = tuple(int(value) for value in values.tolist())
            origin = (0.0,) * dimension
        else:
            raise ValueError("materialization region must be points, a shape, or a four-value rectangle")
    if len(shape) != dimension or len(origin) != dimension or any(value < 0 for value in shape):
        raise ValueError("materialization region shape and origin must match atom dimension")
    axes = [float(origin[index]) + np.arange(shape[index], dtype=float) for index in range(dimension)]
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).astype(np.complex128)


def _mixture_source_values(
    mixture: AtomMixture,
    points: np.ndarray,
    *,
    counters: PerformanceCounters | None = None,
) -> tuple[list[np.ndarray], np.ndarray | None, Any]:
    """Evaluate mixture terms once and return the unquantized source field."""

    point_array = np.asarray(points)
    terms = [np.asarray(atom.evaluate(points, counters=counters), dtype=np.complex128) for atom in mixture.atoms]
    residual = None
    if mixture.residual is not None:
        residual = np.asarray(_residual_value(mixture.residual, points), dtype=np.complex128)
    if point_array.ndim == 1:
        source: Any = sum((complex(term) for term in terms), 0.0 + 0.0j)
        if residual is not None:
            source += complex(residual)
        return terms, residual, source
    if terms:
        source = np.asarray(
            np.sum(np.stack(terms, axis=0).real, axis=0, dtype=np.float64)
            + 1j * np.sum(np.stack(terms, axis=0).imag, axis=0, dtype=np.float64),
            dtype=np.complex128,
        )
    else:
        source = np.zeros(point_array.shape[:-1], dtype=np.complex128)
    if residual is not None:
        source = source + residual
    return terms, residual, source


def _sample_array_value(value: Any, index: tuple[int, ...] | None) -> Any:
    array = np.asarray(value)
    if index is None or array.ndim == 0:
        return array.item() if array.ndim == 0 else value
    return array[index]


def _mixture_spf_materialization(
    mixture: AtomMixture,
    points: np.ndarray,
    *,
    spec: ReductionSpec | float | None = None,
    counters: PerformanceCounters | None = None,
) -> tuple[Any, Any, ErrorLedger]:
    """Reduce encoded mixture terms while retaining source/reducer bounds."""

    terms, residual, source = _mixture_source_values(mixture, points, counters=counters)
    point_array = np.asarray(points)
    scalar = point_array.ndim == 1
    output_shape = () if scalar else point_array.shape[:-1]
    output: Any = np.uint32(0) if scalar else np.empty(output_shape, dtype=np.uint32)
    source_bound = 0.0
    reducer_bound = 0.0
    stages: list[ErrorLedger] = []

    def reduce_point(index: tuple[int, ...] | None) -> int:
        nonlocal source_bound, reducer_bound
        words: list[int] = []
        point_source_bound = 0.0
        for term in terms:
            value = complex(_sample_array_value(term, index))
            word = int(encode_complex(value))
            words.append(word)
            point_source_bound += abs(complex(decode_complex(word)) - value)
        if residual is not None:
            value = complex(_sample_array_value(residual, index))
            word = int(encode_complex(value))
            words.append(word)
            point_source_bound += abs(complex(decode_complex(word)) - value)
        result = reduce_v02(words, kind=ValueKind.SPF_COMPLEX, spec=spec)
        if counters is not None:
            for name in result.counters.__dataclass_fields__:
                setattr(counters, name, getattr(counters, name) + getattr(result.counters, name))
        source_bound = max(source_bound, point_source_bound)
        result_bound = float(result.absolute_error_bound)
        reducer_bound = max(reducer_bound, result_bound)
        stages.append(result.ledger)
        return int(result.value)

    if scalar:
        output = np.uint32(reduce_point(None))
    else:
        for index in np.ndindex(output_shape):
            output[index] = reduce_point(index)
    ledger = ErrorLedger(
        source_quantization=source_bound,
        pruning=max(0.0, float(mixture.dropped_bound)),
        reducer=reducer_bound,
        stages=tuple(stages),
    )
    return output, source, ledger


def mixture_materialize(
    mixture: AtomMixture,
    region: Any,
    output_format: Any = None,
    *,
    spec: ReductionSpec | float | None = None,
    counters: PerformanceCounters | None = None,
    return_result: bool = False,
    return_certificate: bool = False,
) -> Any:
    """Materialize a mixture using the suggested ``(region, output_format)`` API.

    The default return remains the payload.  ``return_result`` or
    ``return_certificate`` returns :class:`MaterializationResult` with source
    quantization, pruning, reduction, and output-conversion terms.
    """

    if mixture.atoms:
        dimension = mixture.atoms[0].dimension
        if any(atom.dimension != dimension for atom in mixture.atoms):
            raise ValueError("all mixture atoms must have the same dimension")
    else:
        # An empty mixture has no atom from which to infer dimensionality.
        # Prefer explicit mapping metadata, then interpret a one-dimensional
        # shape as a tuple of extents and an N-D array as explicit points.
        if isinstance(region, Mapping):
            if "points" in region:
                point_values = np.asarray(region["points"])
                if point_values.ndim == 0:
                    raise ValueError("materialization points must have a trailing dimension")
                dimension = int(point_values.shape[-1])
            elif "shape" in region:
                dimension = int(np.asarray(region["shape"]).size)
            else:
                raise ValueError("materialization region mapping requires shape or points")
        else:
            values = np.asarray(region)
            if values.ndim == 1:
                dimension = 2 if values.size == 4 else int(values.size)
            elif values.ndim >= 2:
                dimension = int(values.shape[-1])
            else:
                raise ValueError("empty-mixture materialization needs a shaped region or points")
        if dimension <= 0:
            raise ValueError("empty-mixture materialization needs a positive dimension")
    points = _region_grid_points(region, dimension)
    is_spf_output = _output_format_name(output_format) in {
        "spf",
        "spf32",
        "word",
        "words",
        "uint32",
    }
    if is_spf_output:
        values, source, ledger = _mixture_spf_materialization(
            mixture,
            points,
            spec=spec,
            counters=counters,
        )
        if not (return_result or return_certificate):
            return values
        return _materialization_result(values, source, "spf32", ledger)
    _, _, source = _mixture_source_values(mixture, points, counters=counters)
    values = _format_materialized(source, output_format)
    if not (return_result or return_certificate):
        return values
    ledger = ErrorLedger(pruning=max(0.0, float(mixture.dropped_bound)))
    return _materialization_result(values, source, output_format, ledger)


# ---------------------------------------------------------------------------
# Separable / low-rank fields
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SeparableField:
    """A low-rank field ``sum_r prod_j f[r][j](q[j])``."""

    factors: tuple[tuple[Any, ...], ...]
    value_kind: str | ValueKind = ValueKind.SPF_COMPLEX

    def __post_init__(self) -> None:
        rows = tuple(tuple(row) for row in self.factors)
        dimensions = {len(row) for row in rows}
        if len(dimensions) > 1:
            raise ValueError("all separable ranks must have the same dimension")
        object.__setattr__(self, "factors", rows)
        object.__setattr__(self, "value_kind", _normalize_kind_name(self.value_kind))

    @property
    def rank(self) -> int:
        return len(self.factors)

    @property
    def dimension(self) -> int:
        return len(self.factors[0]) if self.factors else 0

    def evaluate(
        self,
        q: npt.ArrayLike,
        *,
        spec: ReductionSpec | float | None = None,
        counters: PerformanceCounters | None = None,
    ) -> Any:
        return evaluate_separable(self, q, spec=spec, counters=counters)


def _separable_factor(factor: Any, coordinate: Any) -> Any:
    if callable(factor):
        return factor(coordinate)
    return factor


def evaluate_separable(
    field: SeparableField,
    q: npt.ArrayLike,
    *,
    kind: str | ValueKind | None = None,
    spec: ReductionSpec | float | None = None,
    counters: PerformanceCounters | None = None,
) -> ReductionResult:
    """Evaluate rank products and route the explicit rank sum by value kind."""

    coordinates = np.asarray(q)
    if coordinates.ndim != 1 or coordinates.size != field.dimension:
        raise ValueError("separable evaluation point has the wrong dimension")
    kind_name = _normalize_kind_name(field.value_kind if kind is None else kind)
    terms: list[Any] = []
    for factors in field.factors:
        if kind_name == "spf_complex":
            term = _pack_scalar(0, 0)
        elif kind_name == "log_pos":
            term = LogPos(0)
        elif kind_name == "log_signed":
            term = LogSigned(0, 1)
        elif kind_name in {"native", "real", "dense", "float", "float64"}:
            term = 1.0
        elif kind_name == "phase":
            term = Phase(0)
        else:
            raise ValueError(f"no v0.2 separable algebra is registered for value kind {kind!r}")
        for dimension, factor in enumerate(factors):
            factor_value = _separable_factor(factor, coordinates[dimension])
            if kind_name == "spf_complex":
                if isinstance(factor_value, (float, np.floating, complex, np.complexfloating)):
                    factor_value = encode_complex(factor_value)
                term = spf_mul(term, _coerce_word(factor_value))
            elif kind_name == "log_pos":
                term = logpos_mul(term, factor_value, counters=counters)
            elif kind_name == "log_signed":
                term = _value_mul(term, _coerce_logsigned(factor_value), counters=counters)
            elif kind_name == "phase":
                phase = factor_value if isinstance(factor_value, Phase) else Phase(int(factor_value))
                term = phase_compose(term, phase)
            elif kind_name in {"native", "real", "dense", "float", "float64"}:
                term = term * factor_value
        terms.append(term)
    if kind_name in {"native", "real", "dense", "float", "float64"}:
        operation_counters = counters if counters is not None else PerformanceCounters()
        operation_counters.terms_seen += len(terms)
        operation_counters.terms_retained += len(terms)
        operation_counters.values_materialized += len(terms)
        operation_counters.bytes_read += len(terms) * 8
        operation_counters.bytes_written += 8
        value = sum(terms, 0.0)
        return ReductionResult(
            value=value,
            value_kind="DENSE",
            method="NATIVE_SUM",
            retained_terms=len(terms),
            counters=operation_counters,
        )
    return reduce_v02(terms, kind=kind_name, spec=spec)


separable_evaluate = evaluate_separable


# ---------------------------------------------------------------------------
# Hermitian real-field synthesis
# ---------------------------------------------------------------------------


def _angle_input(value: int | float | Phase) -> tuple[float, int | None]:
    if isinstance(value, Phase):
        return value.radians, value.T
    if isinstance(value, (int, np.integer)):
        phase = int(value) & SPF32_PHASE_MASK
        return phase * _PHASE_TO_RAD, phase
    return float(value), None


def _logabs_trig(value: int | float | Phase, sine: bool) -> tuple[int, int]:
    angle, phase_code = _angle_input(value)
    if phase_code is not None:
        axis = phase_code & SPF32_PHASE_MASK
        if sine and axis in {0, SPF32_PHASE_HALF}:
            return 0, SPF32_L_ZERO
        if not sine and axis in {SPF32_PHASE_QUARTER, 3 * SPF32_PHASE_QUARTER}:
            return 0, SPF32_L_ZERO
    # Floating-point callers may supply an equivalent angle outside the
    # principal interval (for example ``2*pi``).  Reduce modulo 2*pi before
    # snapping exact axes, while leaving genuinely small angles nonzero.
    if phase_code is None:
        angle = math.remainder(angle, _TAU)
    # ``sin(pi)`` and ``cos(pi/2)`` are not exactly zero in binary floating
    # point.  Recognize only the canonical named axes here; a genuinely small
    # numeric angle (for example 1e-16) must remain a finite nonzero value.
    if sine and angle in {0.0, math.pi, -math.pi}:
        return 0, SPF32_L_ZERO
    if not sine and angle in {
        math.pi / 2.0,
        -math.pi / 2.0,
        3.0 * math.pi / 2.0,
        -3.0 * math.pi / 2.0,
    }:
        return 0, SPF32_L_ZERO
    raw = math.sin(angle) if sine else math.cos(angle)
    if raw == 0.0:
        return 0, SPF32_L_ZERO
    return (1 if raw > 0.0 else -1), _canonical_l(_rne(SPF32_L_SCALE * math.log2(abs(raw))))


def logabs_sin(value: int | float | Phase) -> tuple[int, int]:
    """Return ``(sign, log2|sin| code)`` with exact axis zeros."""

    return _logabs_trig(value, True)


def logabs_cos(value: int | float | Phase) -> tuple[int, int]:
    """Return ``(sign, log2|cos| code)`` with exact axis zeros."""

    return _logabs_trig(value, False)


def _coefficient_L_T(coefficient: Any) -> tuple[int, int]:
    if isinstance(coefficient, LogPos):
        return coefficient.L, 0
    if isinstance(coefficient, LogSigned):
        return coefficient.L, 0 if coefficient.sign >= 0 else SPF32_PHASE_HALF
    return _word_parts(_coerce_word(coefficient))


def _self_conjugate_term(coefficient: Any, phase_at_x: int | float | Phase | None = None) -> LogSigned:
    """Convert a DC/Nyquist coefficient to its direct signed-real term.

    §14.2: a self-conjugate bin is individually real, so it carries no factor
    of two, but it is still evaluated at ``x``.  Its accumulated phase is
    ``Phi_k(x) = theta_k + 2*pi*k*x`` with ``theta_k`` in ``{0, pi}``, so
    ``cos(Phi_k(x))`` is exactly ``+-1``: DC contributes ``+-|F_0|`` for every
    ``x`` while Nyquist alternates, ``F_{N/2}(-1)^n``.  Ignoring ``phase_at_x``
    dropped that alternation and made every even-N reconstruction wrong.

    The magnitude is left at ``|F_k|``: unlike the conjugate-pair term of
    §14.1 there is no ``256*log2|cos Phi|`` correction to apply, because the
    cosine of a self-conjugate bin's phase is a unit sign.  A ``phase_at_x``
    that is not on the real axis is off-contract; its nearest sign is used,
    which keeps the DC/Nyquist term exact instead of attenuating it.
    """

    base = _self_conjugate_base(coefficient)
    if phase_at_x is None or base.is_nan or base.is_zero:
        return base
    sign, logabs = logabs_cos(phase_at_x)
    if logabs == SPF32_L_ZERO:
        # cos Phi == 0 exactly: the term vanishes (§14.1 same convention).
        return LogSigned.zero()
    return LogSigned(base.L, sign)


def _self_conjugate_base(coefficient: Any) -> LogSigned:
    """Read a self-conjugate coefficient as a signed-real magnitude."""

    if isinstance(coefficient, LogSigned):
        return coefficient
    if isinstance(coefficient, LogPos):
        if coefficient.is_nan:
            return LogSigned.nan()
        if coefficient.is_zero:
            return LogSigned.zero()
        return LogSigned(coefficient.L, 1)
    if isinstance(coefficient, (float, np.floating, complex, np.complexfloating)):
        value = complex(coefficient)
        if math.isnan(value.real) or math.isnan(value.imag):
            return LogSigned.nan()
        if value.imag != 0.0:
            raise ValueError("self-conjugate Hermitian coefficients must be real")
        return LogSigned.from_value(value.real)
    word = _coerce_word(coefficient)
    L, phase = _word_parts(word)
    if is_nan(word):
        return LogSigned.nan()
    if is_zero(word):
        return LogSigned.zero()
    if phase == 0:
        return LogSigned(L, 1)
    if phase == SPF32_PHASE_HALF:
        return LogSigned(L, -1)
    raise ValueError("self-conjugate Hermitian coefficients must have phase 0 or pi")


def hermitian_pair_term(
    coefficient: Any,
    phase_at_x: int | float | Phase | None = None,
    *,
    self_conjugate: bool = False,
) -> LogSigned:
    """Convert one conjugate pair or self-conjugate endpoint.

    ``self_conjugate=True`` is required for DC/Nyquist bins. Those bins are
    real coefficients and must not receive the factor of two used by a
    non-self-conjugate pair.
    """

    if self_conjugate:
        return _self_conjugate_term(coefficient, phase_at_x)

    L, coefficient_phase = _coefficient_L_T(coefficient)
    if L == SPF32_L_ZERO:
        return LogSigned.zero()
    if L == SPF32_NAN_L:
        return LogSigned.nan()
    phase = coefficient_phase if phase_at_x is None else phase_at_x
    sign, logabs = logabs_cos(phase)
    if logabs == SPF32_L_ZERO:
        return LogSigned.zero()
    return LogSigned(_clamp_finite_l(L + SPF32_L_SCALE + logabs), sign)


def _hermitian_self_conjugate_flags(coefficients: Sequence[Any], metadata: Any) -> list[bool]:
    if metadata is not None:
        if isinstance(metadata, (bool, np.bool_)):
            return [bool(metadata)] * len(coefficients)
        flags = [bool(value) for value in np.asarray(metadata).reshape(-1).tolist()]
        if len(flags) != len(coefficients):
            raise ValueError("self_conjugate metadata must match coefficient count")
        return flags
    # Endpoint status belongs to Fourier-index metadata, not to a typed real
    # coefficient.  A LOG_POS/LOG_SIGNED value at a non-endpoint frequency
    # still contributes a conjugate pair and therefore gets doubled.  Preserve
    # the historical raw-word inference for callers that predate the explicit
    # metadata argument; new structured callers should pass the flags returned
    # by ``hermitian_pair_indices``.
    flags: list[bool] = []
    for coefficient in coefficients:
        if isinstance(coefficient, (LogSigned, LogPos, float, np.floating, complex, np.complexfloating)):
            flags.append(False)
        else:
            try:
                _, phase = _word_parts(_coerce_word(coefficient))
            except (TypeError, ValueError):
                flags.append(False)
            else:
                flags.append(phase in {0, SPF32_PHASE_HALF})
    return flags


def hermitian_synthesis(
    coefficients: Sequence[Any],
    phases: npt.ArrayLike,
    *,
    self_conjugate: Sequence[bool] | npt.ArrayLike | bool | None = None,
) -> np.ndarray | LogSigned:
    """Synthesize Hermitian terms with explicit DC/Nyquist metadata."""

    coeffs = list(coefficients)
    flags = _hermitian_self_conjugate_flags(coeffs, self_conjugate)
    phase_array = np.asarray(phases)
    if phase_array.ndim == 0:
        terms = [
            hermitian_pair_term(coefficient, phase_array.item(), self_conjugate=is_endpoint)
            for coefficient, is_endpoint in zip(coeffs, flags, strict=True)
        ]
        return reduce_logsigned(terms).value
    if phase_array.shape[0] != len(coeffs):
        raise ValueError("phases first dimension must match coefficient count")
    output_shape = phase_array.shape[1:]
    out = np.empty(output_shape, dtype=object)
    for index in np.ndindex(output_shape):
        terms = [
            hermitian_pair_term(
                coefficient,
                phase_array[(term_index, *index)],
                self_conjugate=is_endpoint,
            )
            for term_index, (coefficient, is_endpoint) in enumerate(zip(coeffs, flags, strict=True))
        ]
        out[index] = reduce_logsigned(terms).value
    return out.item() if output_shape == () else out


hermitian_synthesize = hermitian_synthesis


def hermitian_pair_indices(shape: Sequence[int]) -> tuple[tuple[tuple[int, ...], tuple[int, ...], bool], ...]:
    """Enumerate canonical ``k``/``-k`` pairs for an N-D Fourier layout."""

    dimensions = tuple(int(size) for size in shape)
    if not dimensions or any(size <= 0 for size in dimensions):
        raise ValueError("Hermitian shape must contain positive dimensions")
    pairs: list[tuple[tuple[int, ...], tuple[int, ...], bool]] = []
    visited: set[tuple[int, ...]] = set()
    for index in np.ndindex(dimensions):
        if index in visited:
            continue
        partner = tuple((-index[axis]) % dimensions[axis] for axis in range(len(dimensions)))
        representative = min(index, partner)
        other = max(index, partner)
        self_conjugate = representative == other
        pairs.append((representative, other, self_conjugate))
        visited.add(index)
        visited.add(partner)
    return tuple(pairs)


hermitian_pairs = hermitian_pair_indices


# ---------------------------------------------------------------------------
# Spherical-Gaussian rendering profile and log-native display operations
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SphericalGaussian:
    """A spherical Gaussian ``a * exp(lambda * (mu·v - 1))``."""

    mu: npt.ArrayLike
    lambda_: float
    amplitude: complex = 1.0 + 0.0j

    def __post_init__(self) -> None:
        raw_direction = np.asarray(self.mu)
        if np.iscomplexobj(raw_direction) and np.any(np.imag(raw_direction) != 0.0):
            raise ValueError("SphericalGaussian mu must be real")
        direction = np.asarray(raw_direction, dtype=np.float64).reshape(-1)
        if direction.shape != (3,):
            raise ValueError("SphericalGaussian mu must be a three-vector")
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm == 0.0:
            raise ValueError("SphericalGaussian mu must be a finite nonzero direction")
        if not math.isclose(norm, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("SphericalGaussian mu must be unit length")
        concentration = float(self.lambda_)
        if not math.isfinite(concentration) or concentration < 0.0:
            raise ValueError("SphericalGaussian lambda must be finite and non-negative")
        coefficient = complex(self.amplitude)
        if not math.isfinite(coefficient.real) or not math.isfinite(coefficient.imag):
            raise ValueError("SphericalGaussian amplitude must be finite")
        object.__setattr__(self, "mu", direction)
        object.__setattr__(self, "lambda_", concentration)
        object.__setattr__(self, "amplitude", coefficient)

    @property
    def lam(self) -> float:
        return self.lambda_

    def evaluate(self, directions: npt.ArrayLike) -> complex | np.ndarray:
        raw_values = np.asarray(directions)
        if np.iscomplexobj(raw_values) and np.any(np.imag(raw_values) != 0.0):
            raise ValueError("spherical directions must be real")
        values = np.asarray(raw_values, dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("spherical directions must be finite")
        if values.shape == (3,):
            if not math.isclose(float(np.linalg.norm(values)), 1.0, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("spherical directions must be unit length")
            return complex(self.amplitude * np.exp(self.lambda_ * (self.mu @ values - 1.0)))
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError("spherical directions must have trailing dimension 3")
        norms = np.linalg.norm(values, axis=-1)
        if not np.all(np.isclose(norms, 1.0, rtol=1e-12, atol=1e-12)):
            raise ValueError("spherical directions must be unit length")
        return self.amplitude * np.exp(self.lambda_ * (np.einsum("...d,d->...", values, self.mu) - 1.0))

    __call__ = evaluate

    def product(self, other: SphericalGaussian) -> SphericalGaussian:
        return spherical_gaussian_product(self, other)

    def integral(self) -> complex:
        return spherical_gaussian_integral(self)


def spherical_gaussian_product(left: SphericalGaussian, right: SphericalGaussian) -> SphericalGaussian:
    """Multiply two spherical Gaussians in closed form."""

    resultant = left.lambda_ * left.mu + right.lambda_ * right.mu
    concentration = float(np.linalg.norm(resultant))
    direction = np.asarray([0.0, 0.0, 1.0]) if concentration == 0.0 else resultant / concentration
    amplitude = left.amplitude * right.amplitude * math.exp(concentration - left.lambda_ - right.lambda_)
    return SphericalGaussian(direction, concentration, amplitude)


spherical_gaussian_mul = spherical_gaussian_product


def spherical_gaussian_integral(value: SphericalGaussian) -> complex:
    """Integrate a spherical Gaussian over the unit sphere."""

    if value.lambda_ == 0.0:
        return complex(4.0 * math.pi * value.amplitude)
    return complex(2.0 * math.pi * value.amplitude * (-math.expm1(-2.0 * value.lambda_)) / value.lambda_)


def _coerce_logpos_for_tone(value: LogPos | int | float) -> LogPos:
    return _coerce_logpos(value)


def logpos_exposure(value: LogPos | int | float, exposure: float) -> LogPos:
    """Apply ``2**exposure`` by a widened log-code shift."""

    current = _coerce_logpos_for_tone(value)
    if not math.isfinite(float(exposure)):
        raise ValueError("exposure must be finite")
    if current.is_nan or current.is_zero:
        return current
    return LogPos(_clamp_finite_l(current.L + _rne(SPF32_L_SCALE * float(exposure))))


def logpos_power(value: LogPos | int | float, power: float) -> LogPos:
    """Apply a positive-real power in widened log-code space."""

    current = _coerce_logpos_for_tone(value)
    exponent = float(power)
    if not math.isfinite(exponent):
        raise ValueError("power must be finite")
    if current.is_nan:
        return current
    if current.is_zero:
        if exponent > 0.0:
            return current
        if exponent == 0.0:
            return LogPos(0)
        return LogPos.nan()
    return LogPos(_clamp_finite_l(_rne(exponent * current.L)))


def logpos_reinhard(value: LogPos | int | float) -> LogPos:
    """Apply ``x/(1+x)`` without leaving log-positive representation."""

    current = _coerce_logpos_for_tone(value)
    if current.is_nan or current.is_zero:
        return current
    denominator = logadd_pos(LogPos(0), current)
    return LogPos(_clamp_finite_l(current.L - denominator.L))


tone_exposure = logpos_exposure
tone_power = logpos_power
tone_reinhard = logpos_reinhard


def display_code_from_log(
    value: LogPos | int,
    *,
    black_offset: int = 0,
    lut: Sequence[int] | npt.ArrayLike | None = None,
) -> int:
    """Map a finite log code through a direct display LUT."""

    current = value if isinstance(value, LogPos) else LogPos(int(value))
    if current.is_zero:
        return 0
    if current.is_nan:
        return 0
    index = int(current.L) - int(black_offset)
    if lut is None:
        return max(0, min(255, index))
    table = np.asarray(lut).reshape(-1)
    if not 0 <= index < table.size:
        raise IndexError("log code is outside the display LUT")
    return int(table[index])


display_lut = display_code_from_log


# ---------------------------------------------------------------------------
# Ordered affine transport monoid
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Transport:
    """An affine transport map ``x ↦ C + T x``."""

    T: Any
    C: Any
    error_bound_T: float = field(default=0.0, kw_only=True)
    error_bound_C: float = field(default=0.0, kw_only=True)
    ledger: ErrorLedger = field(default_factory=ErrorLedger, kw_only=True)

    def __post_init__(self) -> None:
        for name in ("error_bound_T", "error_bound_C"):
            value = float(getattr(self, name))
            if not math.isnan(value) and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not isinstance(self.ledger, ErrorLedger):
            raise TypeError("Transport.ledger must be an ErrorLedger")

    @property
    def transmittance(self) -> Any:
        return self.T

    @property
    def contribution(self) -> Any:
        return self.C

    @property
    def absolute_error_bound(self) -> float:
        """Coefficient error bound for a unit-magnitude background."""

        return self.error_bound_T + self.error_bound_C

    @property
    def stage_certificates(self) -> tuple[ErrorLedger, ...]:
        return self.ledger.stages

    def apply(self, background: Any, *, counters: PerformanceCounters | None = None) -> Any:
        return _value_add(self.C, _value_mul(self.T, background), counters=counters)


AffineTransport = Transport


def _logpos_as_signed(value: LogPos) -> LogSigned:
    """Promote a positive-real value through the normative typed edge."""

    if value.is_nan:
        return LogSigned.nan()
    if value.is_zero:
        return LogSigned.zero()
    return LogSigned(value.L, 1)


def _logsigned_mul(left: LogSigned, right: LogSigned, *, counters: PerformanceCounters | None = None) -> LogSigned:
    if counters is not None:
        counters.log_ops += 1
    if left.is_nan or right.is_nan:
        return LogSigned.nan()
    if left.is_zero or right.is_zero:
        return LogSigned.zero()
    return LogSigned(_clamp_finite_l(left.L + right.L), left.sign * right.sign)


def _value_mul(left: Any, right: Any, *, counters: PerformanceCounters | None = None) -> Any:
    if isinstance(left, LogPos) and isinstance(right, LogPos):
        return logpos_mul(left, right, counters=counters)
    if isinstance(left, LogSigned) and isinstance(right, LogSigned):
        return _logsigned_mul(left, right, counters=counters)
    if isinstance(left, LogPos) and isinstance(right, LogSigned):
        return _logsigned_mul(_logpos_as_signed(left), right, counters=counters)
    if isinstance(left, LogSigned) and isinstance(right, LogPos):
        return _logsigned_mul(left, _logpos_as_signed(right), counters=counters)
    if isinstance(left, (SPF32Value, int, np.integer)) and isinstance(right, (SPF32Value, int, np.integer)):
        return spf_mul(_coerce_word(left), _coerce_word(right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_arr, right_arr = np.broadcast_arrays(np.asarray(left, dtype=object), np.asarray(right, dtype=object))
        out = np.empty(left_arr.shape, dtype=object)
        for index in np.ndindex(left_arr.shape):
            out[index] = _value_mul(left_arr[index], right_arr[index], counters=counters)
        return out
    if isinstance(left, (LogPos, LogSigned)) or isinstance(right, (LogPos, LogSigned)):
        raise TypeError("transport values must use matching typed SPF algebras")
    return left * right


def _value_add(left: Any, right: Any, *, counters: PerformanceCounters | None = None) -> Any:
    if isinstance(left, LogPos) and isinstance(right, LogPos):
        return logadd_pos(left, right, counters=counters)
    if isinstance(left, LogSigned) and isinstance(right, LogSigned):
        return logadd_signed(left, right, counters=counters)
    if isinstance(left, LogPos) and isinstance(right, LogSigned):
        return logadd_signed(_logpos_as_signed(left), right, counters=counters)
    if isinstance(left, LogSigned) and isinstance(right, LogPos):
        return logadd_signed(left, _logpos_as_signed(right), counters=counters)
    if isinstance(left, (SPF32Value, int, np.integer)) and isinstance(right, (SPF32Value, int, np.integer)):
        return spf_add_complex_reference(_coerce_word(left), _coerce_word(right))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        left_arr, right_arr = np.broadcast_arrays(np.asarray(left, dtype=object), np.asarray(right, dtype=object))
        out = np.empty(left_arr.shape, dtype=object)
        for index in np.ndindex(left_arr.shape):
            out[index] = _value_add(left_arr[index], right_arr[index], counters=counters)
        return out
    if isinstance(left, (LogPos, LogSigned)) or isinstance(right, (LogPos, LogSigned)):
        raise TypeError("transport values must use matching typed SPF algebras")
    return left + right


def _coerce_transport(value: Transport | tuple[Any, Any]) -> Transport:
    if isinstance(value, Transport):
        return value
    if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
        raise TypeError("transport segments must be Transport or a (T, C) pair")
    return Transport(value[0], value[1])


def _transport_component_family(value: Any) -> str:
    """Classify a transport component without guessing across algebras."""

    if isinstance(value, RawLogCode):
        raise TypeError("RawLogCode is not a transport value; use LogPos/LogSigned")
    if isinstance(value, (LogPos, LogSigned)):
        return "typed_log"
    if isinstance(value, (SPF32Value, int, np.integer)):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("transport components must not be bool")
        return "spf_complex"
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "iu":
            return "spf_complex"
        if value.dtype.kind in "fc":
            return "native"
        families = {_transport_component_family(item) for item in value.reshape(-1)}
        if not families:
            return "native"
        if len(families) != 1:
            raise TypeError("transport arrays must contain one value representation")
        return families.pop()
    if isinstance(value, (float, np.floating, complex, np.complexfloating)):
        return "native"
    raise TypeError(
        "transport components must be LogPos/LogSigned, packed SPF32 values, "
        f"or numeric scalars/arrays; received {type(value).__name__}"
    )


def _validate_transport_inputs(items: Sequence[Any]) -> list[Transport]:
    transports = [_coerce_transport(item) for item in items]
    families: set[str] = set()
    for transport in transports:
        t_family = _transport_component_family(transport.T)
        c_family = _transport_component_family(transport.C)
        if t_family != c_family and {t_family, c_family} != {"typed_log"}:
            raise TypeError(
                "each transport segment must use one representation family for T and C"
            )
        families.add(t_family)
    if len(families) > 1:
        raise TypeError("all transport segments must use the same value representation")
    return transports


def _numeric_transport_value(value: Any) -> Any:
    """Decode a transport component into its mathematical scalar/array form."""

    if isinstance(value, LogPos):
        return value.value
    if isinstance(value, LogSigned):
        return value.value
    if isinstance(value, SPF32Value):
        return complex(decode_complex(value.word))
    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return complex(decode_complex(int(value)))
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "iu":
            return np.asarray(decode_complex(value), dtype=np.complex128)
        if value.dtype != object:
            return value
        output = np.empty(value.shape, dtype=np.complex128)
        for index in np.ndindex(value.shape):
            output[index] = _numeric_transport_value(value[index])
        return output
    return value


def _max_abs_numeric(value: Any) -> float:
    array = np.asarray(value)
    if array.size == 0:
        return 0.0
    try:
        maximum = float(np.max(np.abs(array.astype(np.complex128, copy=False))))
    except (TypeError, ValueError):
        maximum = float(np.max(np.abs(array)))
    return math.inf if math.isnan(maximum) else maximum


def _transport_operation_error(left: Any, right: Any, result: Any, operation: str) -> float:
    """Bound one quantized transport multiply/add against its exact inputs."""

    left_numeric = _numeric_transport_value(left)
    right_numeric = _numeric_transport_value(right)
    result_numeric = _numeric_transport_value(result)
    exact = left_numeric * right_numeric if operation == "mul" else left_numeric + right_numeric
    actual = _max_abs_numeric(np.asarray(result_numeric) - np.asarray(exact))
    quantized = any(
        isinstance(item, (LogPos, LogSigned, SPF32Value, int, np.integer))
        and not isinstance(item, (bool, np.bool_))
        for item in (left, right, result)
    )
    scale = max(_max_abs_numeric(exact), _max_abs_numeric(result_numeric), 1.0 if actual == 0.0 else 0.0)
    if quantized:
        # The log code is rounded once per typed operation.  Include the phase
        # radial contribution for packed complex components.  ``actual`` is
        # retained because the complex reference adder can incur cancellation
        # error beyond the scalar output quantizer.
        quantization = scale * (_LOG_QUANTIZATION_RELATIVE_ERROR + math.pi / SPF32_PHASE_MOD)
    else:
        quantization = np.finfo(np.float64).eps * scale
    return actual + quantization


def transport_compose(
    front: Transport | tuple[Any, Any],
    back: Transport | tuple[Any, Any],
    *,
    counters: PerformanceCounters | None = None,
) -> Transport:
    """Compose front and back affine transports in ordered front-to-back order."""

    first, second = _coerce_transport(front), _coerce_transport(back)
    if counters is not None:
        counters.transport_compositions += 1
    transmittance = _value_mul(first.T, second.T, counters=counters)
    attenuated_contribution = _value_mul(first.T, second.C, counters=counters)
    contribution = _value_add(first.C, attenuated_contribution, counters=counters)

    first_T = _max_abs_numeric(_numeric_transport_value(first.T))
    second_T = _max_abs_numeric(_numeric_transport_value(second.T))
    second_C = _max_abs_numeric(_numeric_transport_value(second.C))
    first_t_error = float(first.error_bound_T)
    second_t_error = float(second.error_bound_T)
    first_c_error = float(first.error_bound_C)
    second_c_error = float(second.error_bound_C)
    multiply_T_error = _transport_operation_error(first.T, second.T, transmittance, "mul")
    multiply_C_error = _transport_operation_error(first.T, second.C, attenuated_contribution, "mul")
    add_C_error = _transport_operation_error(first.C, attenuated_contribution, contribution, "add")

    total_T_error = (
        multiply_T_error
        + second_T * first_t_error
        + first_T * second_t_error
        + first_t_error * second_t_error
    )
    attenuated_error = (
        multiply_C_error
        + second_C * first_t_error
        + first_T * second_c_error
        + first_t_error * second_c_error
    )
    total_C_error = add_C_error + first_c_error + attenuated_error
    stage = ErrorLedger(reducer=total_T_error + total_C_error)
    ledger = ErrorLedger(
        reducer=total_T_error + total_C_error,
        stages=first.ledger.stages + second.ledger.stages + (stage,),
    )
    return Transport(
        transmittance,
        contribution,
        error_bound_T=total_T_error,
        error_bound_C=total_C_error,
        ledger=ledger,
    )


def transport_reduce_ordered(
    segments: Sequence[Transport | tuple[Any, Any]],
    *,
    counters: PerformanceCounters | None = None,
) -> Transport:
    """Reduce an ordered transport sequence with an order-preserving tree."""

    current = [_coerce_transport(segment) for segment in segments]
    if not current:
        return Transport(LogPos(0), LogPos.zero())
    while len(current) > 1:
        next_level: list[Transport] = []
        for index in range(0, len(current), 2):
            if index + 1 == len(current):
                next_level.append(current[index])
            else:
                next_level.append(transport_compose(current[index], current[index + 1], counters=counters))
        current = next_level
    return current[0]


def reduce_transport(
    segments: Sequence[Transport | tuple[Any, Any]],
    spec: ReductionSpec | float | None = None,
    *,
    counters: PerformanceCounters | None = None,
    value_kind: str | ValueKind | None = None,
) -> ReductionResult:
    """Reduce ordered affine transports and return a quantization certificate.

    Segment components must belong to one representation family.  In
    particular, bare integers are interpreted as packed SPF32 words here;
    positive/signed log transport must use :class:`LogPos`/:class:`LogSigned`
    so a raw log code cannot be mistaken for a wire word.
    """

    policy = _coerce_spec(spec, None)
    required_kind = _normalize_kind_name(value_kind) if value_kind is not None else (
        None
        if policy.required_value_kind is None
        else _normalize_kind_name(policy.required_value_kind)
    )
    if required_kind not in (None, ValueKind.AFFINE_TRANSPORT.value.lower()):
        raise TypeError("reduce_transport requires AFFINE_TRANSPORT values")
    items = list(segments)
    transports = _validate_transport_inputs(items)
    operation_counters = counters if counters is not None else PerformanceCounters()
    operation_counters.terms_seen = max(operation_counters.terms_seen, len(items))
    operation_counters.terms_retained = max(operation_counters.terms_retained, len(items))
    operation_counters.bytes_read += len(items) * 8
    operation_counters.bytes_written += 8
    value = transport_reduce_ordered(transports, counters=operation_counters)

    # Evaluate the same ordered composition in the decoded mathematical
    # domain.  The tree is algebraically equivalent there; the difference is
    # solely the staged SPF/log quantization captured by ``value``.
    exact_T: Any = 1.0
    exact_C: Any = 0.0
    for segment in transports:
        numeric_T = _numeric_transport_value(segment.T)
        numeric_C = _numeric_transport_value(segment.C)
        exact_C = exact_C + exact_T * numeric_C
        exact_T = exact_T * numeric_T
    observed_T_error = _max_abs_numeric(_numeric_transport_value(value.T) - np.asarray(exact_T))
    observed_C_error = _max_abs_numeric(_numeric_transport_value(value.C) - np.asarray(exact_C))
    absolute_error = max(
        float(value.error_bound_T) + float(value.error_bound_C),
        observed_T_error + observed_C_error,
    )
    exact_scale = max(_max_abs_numeric(exact_T), _max_abs_numeric(exact_C))
    relative_error = math.inf if exact_scale == 0.0 else absolute_error / exact_scale
    ledger = value.ledger
    if ledger.total_bound < absolute_error or math.isnan(ledger.total_bound):
        ledger = replace(ledger, reducer=absolute_error)
    return ReductionResult(
        value=value,
        value_kind=ValueKind.AFFINE_TRANSPORT.value,
        method=ReductionRoute.ORDERED_ASSOCIATIVE_TREE.value,
        absolute_error_bound=absolute_error,
        relative_error_bound=relative_error,
        retained_terms=len(items),
        counters=operation_counters,
        ledger=ledger,
    )


# ---------------------------------------------------------------------------
# Jet2D <-> TensorBlockHeaderV2 bridge (design §12 / §14.2, amended 2026-09-17)
# ---------------------------------------------------------------------------


def _jet_coefficients_in_header_ticks(
    jet: Jet2D | Jet2DCubic, rounding: str
) -> tuple[dict[str, int], float]:
    """Return header-tick integer coefficients and a certified rounding bound.

    Jet coefficients are held in widened fixed point (``guard_T`` / ``guard_L``
    sub-tick bits, see :func:`jet_emit_spf32`).  The header stores whole
    ticks, so each coefficient is divided by ``2**guard``; ``rounding="exact"``
    requires the quotients (and ``H_ii / 2``) to be integers.  The returned
    bound is the sum of absolute rounding residuals per coefficient, in ticks,
    to be scaled by the caller's region monomials.
    """

    kind = jet.value_kind
    if kind == "phase":
        guard = 1 << int(jet.guard_T)
    elif kind in {"log", "log_pos", "log_signed"}:
        guard = 1 << int(jet.guard_L)
    else:
        raise ValueError(f"jet_to_tensor_block_header requires a phase or log jet, got {jet.value_kind!r}")
    raw = {
        "c": complex(jet.value).real / guard,
        "gx": complex(jet.grad_x).real / guard,
        "gy": complex(jet.grad_y).real / guard,
        "hxx2": complex(jet.hess_xx).real / (2 * guard),
        "hyy2": complex(jet.hess_yy).real / (2 * guard),
        "hxy": complex(jet.hess_xy).real / guard,
    }
    coefficients = (jet.value, jet.grad_x, jet.grad_y, jet.hess_xx, jet.hess_xy, jet.hess_yy)
    if isinstance(jet, Jet2DCubic):
        # §13.6 cubic monomials carry no factorial normalisation, so each is one
        # header tick word divided by the guard scale (like hxy).
        cxxx, cxxy, cxyy, cyyy = jet.cubic_coefficients
        raw["cxxx"] = complex(cxxx).real / guard
        raw["cxxy"] = complex(cxxy).real / guard
        raw["cxyy"] = complex(cxyy).real / guard
        raw["cyyy"] = complex(cyyy).real / guard
        coefficients = (*coefficients, cxxx, cxxy, cxyy, cyyy)
    if any(np.iscomplexobj(v) and complex(v).imag for v in coefficients):
        raise ValueError("jet_to_tensor_block_header requires real coefficients")
    ticks: dict[str, int] = {}
    residual: dict[str, float] = {}
    for name, value in raw.items():
        nearest = int(_rne(value))
        residual[name] = abs(value - nearest)
        if rounding == "exact" and residual[name] != 0.0:
            raise ValueError(
                f"jet coefficient {name} = {value} header ticks is not an integer; "
                "pass rounding='nearest' to accept a certified rounding error"
            )
        ticks[name] = nearest
    return ticks, sum(residual.values())


def jet_to_tensor_block_header(
    jet: Jet2D | Jet2DCubic,
    dims: Sequence[int],
    *,
    origin: Sequence[int] | None = None,
    rounding: str = "exact",
    global_L_bias: int = 0,
    **header_kwargs: Any,
) -> tuple[Any, float]:
    """Express a 2D ``Jet2D``/``Jet2DCubic`` as a ``TensorBlockHeaderV2`` (ABI v22/v23/v24).

    A ``phase`` jet ``P(x, y) = c + g·q + ½ qᵀHq`` becomes header phase state:
    ``global_T_bias = c``, ``phase_gradient = (g_y, g_x)``,
    ``quadratic_ramp = (H_yy/2, H_xx/2)``, ``mixed_ramp = (H_xy,)``.  Header
    dimension 0 is the row (``y``) axis and dimension 1 the column (``x``)
    axis, matching :func:`jet_forward_differences`.  A :class:`Jet2DCubic` adds
    the AlgRef v0.2 §13.6 cubic terms: ``cubic_ramp = (C_yyy, C_xxx)`` and
    ``cubic_mixed = (C_xyy, C_xxy)`` (ABI v24 — ``cubic_mixed[0]`` multiplies
    ``q0² q1 = y² x`` and ``cubic_mixed[1]`` multiplies ``q0 q1² = y x²``).  The
    cubic terms are PHASE-ONLY: there is no ``log_cubic`` header field, so a
    cubic ``log`` jet is refused rather than silently truncated.  A ``log`` jet maps onto
    ``global_L_bias = c``, ``band_exp = (g_y, g_x)``,
    ``log_quadratic = (H_yy/2, H_xx/2)`` and ``log_mixed = (H_xy,)`` (ABI v23,
    signed int16 L ticks).

    Range discipline differs between the two kinds because the two header
    polynomials do.  Phase is a residue mod ``2**16``, so every integer tick
    value is exactly representable and nothing can overflow.  The log fields are
    *signed int16 stored, widened when evaluated* (C: ``int16_t band_exp[4]``,
    ``int16_t log_quadratic[4]``, ``int16_t log_mixed[6]``; ``global_L_bias`` is
    a finite L code), so an out-of-range coefficient does not wrap — it would
    make the Python and C reconstructions disagree outright.  Such a
    coefficient therefore either raises ``SPFError`` (a ``ValueError``) naming
    the jet term (``rounding="exact"``: the header cannot represent the jet)
    or is *clamped*
    to the field range with the clamping error charged to the returned bound
    (``rounding="nearest"``).

    Returns ``(header, error_bound_ticks)``, a conservative upper bound on
    ``|P_header(q) - P(q)|`` over the block extent, by the triangle inequality

    ``Σ_k (|raw_k - stored_k| + slack_k) · max_q |monomial_k(q)|``,

    accumulated in exact rational arithmetic and rounded *up* into float64 (see
    ``_round_up``) so that no replay can exceed it, not even one that evaluates
    the polynomial in exact rationals.  ``slack_k`` is half an ulp of the
    supplied coefficient on terms that had to round and zero on terms that did
    not, so ``P`` may be read either as the jet's own float64 polynomial or as
    the real-valued polynomial those floats round to, while a coefficient the
    header stores verbatim still contributes exactly nothing.  The bound is
    therefore ``0.0`` exactly when the header reproduces the jet term for term —
    always so under ``rounding="exact"``, and consumers rely on that reading
    (``spf.renderer.splat_terminal`` takes the header route only at bound 0).
    ``P_header`` is the *unsaturated* header polynomial; saturation of the summed
    L code into the finite range happens in the block and is outside this bound.
    """

    from fractions import Fraction

    from .core import ScalarFormat, SPFError, format_for
    from .v2 import TensorBlockHeaderV2

    extent = tuple(int(d) for d in dims)
    if len(extent) != 2:
        raise ValueError("jet_to_tensor_block_header expects a two-dimensional block")
    if rounding not in {"exact", "nearest"}:
        raise ValueError("rounding must be 'exact' or 'nearest'")
    ticks, _ = _jet_coefficients_in_header_ticks(jet, rounding)
    rows, cols = extent
    qy_max, qx_max = max(rows - 1, 0), max(cols - 1, 0)
    kind = jet.value_kind
    guard = 1 << (int(jet.guard_T) if kind == "phase" else int(jet.guard_L))

    def _exact(value: Any, denominator: int) -> Fraction:
        real = complex(value).real
        if not math.isfinite(real):
            raise ValueError("jet_to_tensor_block_header requires finite coefficients")
        return Fraction(real) / denominator

    def _round_up(value: Fraction) -> float:
        """Smallest float64 that is >= ``value`` (one nextafter, at most)."""

        out = float(value)
        if Fraction(out) < value:
            out = math.nextafter(out, math.inf)
        return out

    def _half_ulp(value: Any, denominator: int, residual: Fraction) -> Fraction:
        """Half an ulp of a supplied coefficient, in header ticks.

        Charged only where the coefficient did *not* already land on a whole
        header tick.  A term that rounded is approximate either way, so also
        covering the float64 representation of the caller's coefficient costs
        nothing and makes the bound hold for a replay that evaluates the real
        polynomial those floats round to, in exact rationals.  A term that did
        not round is reproduced by the header float for float, and a zero bound
        is load-bearing: consumers read ``bound == 0.0`` as "this header is word
        identical to the recurrence on the same jet" and take the header route on
        the strength of it (``spf.renderer.splat_terminal._header_alpha_codes``).
        """

        if residual == 0:
            return Fraction(0)
        return Fraction(math.ulp(complex(value).real)) / (2 * denominator)

    raw = {
        "c": _exact(jet.value, guard),
        "gx": _exact(jet.grad_x, guard),
        "gy": _exact(jet.grad_y, guard),
        "hxx2": _exact(jet.hess_xx, 2 * guard),
        "hyy2": _exact(jet.hess_yy, 2 * guard),
        "hxy": _exact(jet.hess_xy, guard),
    }
    residual = {name: abs(raw[name] - ticks[name]) for name in raw}
    slack = {
        "c": _half_ulp(jet.value, guard, residual["c"]),
        "gx": _half_ulp(jet.grad_x, guard, residual["gx"]),
        "gy": _half_ulp(jet.grad_y, guard, residual["gy"]),
        "hxx2": _half_ulp(jet.hess_xx, 2 * guard, residual["hxx2"]),
        "hyy2": _half_ulp(jet.hess_yy, 2 * guard, residual["hyy2"]),
        "hxy": _half_ulp(jet.hess_xy, guard, residual["hxy"]),
    }
    # ``TensorBlock.reconstruct`` evaluates the header polynomial at
    # ``(origin + index) % dim``, so every coordinate stays in ``[0, dim)``
    # whatever the origin: the largest monomial is fixed by the extent alone.
    monomial = {
        "c": 1,
        "gx": qx_max,
        "gy": qy_max,
        "hxx2": qx_max * qx_max,
        "hyy2": qy_max * qy_max,
        "hxy": qx_max * qy_max,
    }
    cubic = isinstance(jet, Jet2DCubic)
    if cubic:
        cxxx, cxxy, cxyy, cyyy = jet.cubic_coefficients
        raw["cxxx"] = _exact(cxxx, guard)
        raw["cxxy"] = _exact(cxxy, guard)
        raw["cxyy"] = _exact(cxyy, guard)
        raw["cyyy"] = _exact(cyyy, guard)
        for name, value in (
            ("cxxx", cxxx),
            ("cxxy", cxxy),
            ("cxyy", cxyy),
            ("cyyy", cyyy),
        ):
            residual[name] = abs(raw[name] - ticks[name])
            slack[name] = _half_ulp(value, guard, residual[name])
        monomial["cxxx"] = qx_max**3
        monomial["cxxy"] = qx_max * qx_max * qy_max
        monomial["cxyy"] = qx_max * qy_max * qy_max
        monomial["cyyy"] = qy_max**3
    error = sum(((residual[name] + slack[name]) * monomial[name] for name in raw), Fraction(0))
    if kind == "phase":
        header = TensorBlockHeaderV2(
            dims=extent,
            origin=None if origin is None else tuple(int(v) for v in origin),
            global_L_bias=int(global_L_bias),
            global_T_bias=ticks["c"] & SPF32_PHASE_MASK,
            phase_gradient=(ticks["gy"] & SPF32_PHASE_MASK, ticks["gx"] & SPF32_PHASE_MASK),
            quadratic_ramp=(ticks["hyy2"] & SPF32_PHASE_MASK, ticks["hxx2"] & SPF32_PHASE_MASK),
            mixed_ramp=(ticks["hxy"] & SPF32_PHASE_MASK,),
            # §13.6: cubic_ramp is (dim0 = y, dim1 = x) like every other
            # per-axis phase vector; cubic_mixed is (q0^2 q1, q0 q1^2).
            cubic_ramp=(
                (ticks["cyyy"] & SPF32_PHASE_MASK, ticks["cxxx"] & SPF32_PHASE_MASK)
                if cubic
                else (0, 0)
            ),
            cubic_mixed=(
                (ticks["cxyy"] & SPF32_PHASE_MASK, ticks["cxxy"] & SPF32_PHASE_MASK)
                if cubic
                else (0, 0)
            ),
            **header_kwargs,
        )
        return header, _round_up(error)
    if cubic and any(
        complex(v) != 0 for v in jet.cubic_coefficients
    ):
        raise ValueError(
            "TensorBlockHeaderV2 carries cubic terms on the PHASE polynomial only "
            "(cubic_ramp, cubic_mixed); the log-magnitude jet stops at the "
            "quadratic (band_exp, log_quadratic, log_mixed — ABI v23). Emit the "
            "log cubic as a residual, or subdivide (AlgRef v0.2 §13.5)."
        )
    fmt = format_for(header_kwargs.get("scalar_format_id", ScalarFormat.SPF32))
    i16 = (-32768, 32767)
    field_range = {
        "c": (int(fmt.l_code_min_finite), int(fmt.l_code_max_finite)),
        "gx": i16,
        "gy": i16,
        "hxx2": i16,
        "hyy2": i16,
        "hxy": i16,
    }
    term_name = {
        "c": "value",
        "gx": "grad_x",
        "gy": "grad_y",
        "hxx2": "hess_xx/2",
        "hyy2": "hess_yy/2",
        "hxy": "hess_xy",
    }
    stored: dict[str, int] = {}
    for name, (low, high) in field_range.items():
        value = int(ticks[name])
        clipped = low if value < low else (high if value > high else value)
        if clipped != value:
            if rounding == "exact":
                raise SPFError(
                    f"log jet {term_name[name]} = {value} header L ticks does not fit the header "
                    f"field range [{low}, {high}]; pass rounding='nearest' to clamp it and charge "
                    "the clamping error to the returned bound"
                )
            error += Fraction(abs(value - clipped)) * monomial[name]
        stored[name] = clipped
    header = TensorBlockHeaderV2(
        dims=extent,
        origin=None if origin is None else tuple(int(v) for v in origin),
        global_L_bias=stored["c"],
        band_exp=(stored["gy"], stored["gx"]),
        log_quadratic=(stored["hyy2"], stored["hxx2"]),
        log_mixed=(stored["hxy"],),
        **header_kwargs,
    )
    return header, _round_up(error)


def jet_from_tensor_block_header(
    header: Any, *, kind: str = "phase", guard: int | None = None
) -> Jet2D | Jet2DCubic:
    """Recover a 2D ``Jet2D``/``Jet2DCubic`` from ``TensorBlockHeaderV2`` state.

    Inverse of :func:`jet_to_tensor_block_header` for ``rounding="exact"``.
    For ``kind="phase"`` the linear term is ``phase_gradient`` plus the column
    sums of ``ramp_R`` (the two are additive in ``ramp_phase``) and
    ``ramp_b``'s sum folds into the constant.  A header carrying any AlgRef v0.2
    §13.6 cubic term (``cubic_ramp`` or ``cubic_mixed``, ABI v24) returns a
    :class:`Jet2DCubic`; an all-zero cubic state returns a plain :class:`Jet2D`,
    so a v23-era header round-trips to exactly the object it used to.  That
    default-zero behaviour is what answers design §12.6's preference for "full
    quadratic + adaptive subdivision + residual over an always-on incomplete
    cubic header": the cubic words cost nothing and change nothing until a
    producer sets them, and when set they are the COMPLETE 2-D cubic, not the
    rotation-unstable axis-only subset §13.6 warns about.

    ``kind="log"`` has no cubic state (§13 gives the log jet only up to the
    quadratic, ABI v23) and always returns a :class:`Jet2D`.

    Coefficients are returned in the jet's widened units (``guard`` bits,
    default 8) as signed values in ``(-2^15, 2^15]`` ticks.
    """

    if int(header.order or 0) != 2:
        raise ValueError("jet_from_tensor_block_header expects an order-2 header")
    guard_bits = 8 if guard is None else int(guard)
    scale = 1 << guard_bits

    def signed(value: int) -> int:
        v = int(value) & SPF32_PHASE_MASK
        return v - SPF32_PHASE_MOD if v >= SPF32_PHASE_HALF else v

    if kind == "phase":
        R = np.asarray(header.ramp_R, dtype=np.int64).reshape(2, 2)
        colsum = R.sum(axis=0)
        gy = signed(int(header.phase_gradient[0]) + int(colsum[0]))
        gx = signed(int(header.phase_gradient[1]) + int(colsum[1]))
        c = signed(int(header.global_T_bias) + int(sum(int(v) for v in header.ramp_b)))
        hyy = 2 * signed(header.quadratic_ramp[0])
        hxx = 2 * signed(header.quadratic_ramp[1])
        hxy = signed(header.mixed_ramp[0]) if len(header.mixed_ramp) else 0
        cubic_ramp = tuple(int(v) for v in (getattr(header, "cubic_ramp", ()) or ()))
        cubic_mixed = tuple(int(v) for v in (getattr(header, "cubic_mixed", ()) or ()))
        if any(cubic_ramp) or any(cubic_mixed):
            cyyy = signed(cubic_ramp[0]) if len(cubic_ramp) > 0 else 0
            cxxx = signed(cubic_ramp[1]) if len(cubic_ramp) > 1 else 0
            cxyy = signed(cubic_mixed[0]) if len(cubic_mixed) > 0 else 0
            cxxy = signed(cubic_mixed[1]) if len(cubic_mixed) > 1 else 0
            return Jet2DCubic(
                value=c * scale,
                grad_x=gx * scale,
                grad_y=gy * scale,
                hess_xx=hxx * scale,
                hess_xy=hxy * scale,
                hess_yy=hyy * scale,
                cubic_xxx=cxxx * scale,
                cubic_xxy=cxxy * scale,
                cubic_xyy=cxyy * scale,
                cubic_yyy=cyyy * scale,
                guard_T=guard_bits,
                value_kind="phase",
            )
        return Jet2D(
            value=c * scale,
            grad_x=gx * scale,
            grad_y=gy * scale,
            hess_xx=hxx * scale,
            hess_xy=hxy * scale,
            hess_yy=hyy * scale,
            guard_T=guard_bits,
            value_kind="phase",
        )
    if kind in {"log", "log_pos", "log_signed"}:
        c = int(header.global_L_bias) + int(header.coarse_exp)
        gy, gx = (int(v) for v in header.band_exp)
        hyy = 2 * int(header.log_quadratic[0])
        hxx = 2 * int(header.log_quadratic[1])
        hxy = int(header.log_mixed[0]) if len(header.log_mixed) else 0
        return Jet2D(
            value=c * scale,
            grad_x=gx * scale,
            grad_y=gy * scale,
            hess_xx=hxx * scale,
            hess_xy=hxy * scale,
            hess_yy=hyy * scale,
            guard_L=guard_bits,
            value_kind=kind,
        )
    raise ValueError(f"unsupported jet kind {kind!r}")


__all__ = [
    "EXP2_NEG_GAP_ROM",
    "EXP2_NEG_GAP_ROM_SHA256",
    "EXP2_NEG_GAP_ROM_SIZE",
    "EXP2_NEG_GAP_ROM_VERSION",
    "EXP2_NEG_GAP_SHIFT_BITS",
    "LOGADD_G_ZERO",
    "LOGADD_MINUS",
    "LOGADD_PLUS",
    "LOGADD_ROM_SHA256",
    "LOGADD_ROM_VERSION",
    "PHASE_RELIABILITY_MAX_ERROR_RAD",
    "SPF32_INTERNAL_PHASE_BITS",
    "SPF32_INTERNAL_PHASE_MOD",
    "SPF32_INTERNAL_PHASE_SCALE",
    "SPF32_L_FRAC_BITS",
    "SPF32_L_MAX_FINITE",
    "SPF32_L_MIN_FINITE",
    "SPF32_L_SCALE",
    "SPF32_L_ZERO",
    "SPF32_NAN_L",
    "SPF32_NAN_T",
    "SPF32_PHASE_BITS",
    "SPF32_PHASE_HALF",
    "SPF32_PHASE_MASK",
    "SPF32_PHASE_MOD",
    "SPF32_PHASE_QUARTER",
    "SPF32_T_ZERO",
    "SPF32_WORD_NAN",
    "SPF32_WORD_ZERO",
    "AffineTransport",
    "AtomBound",
    "AtomMixture",
    "CubicJet2D",
    "ErrorLedger",
    "GaussianChirpAtom",
    "Jet2D",
    "Jet2DCubic",
    "LogPos",
    "LogSigned",
    "MaterializationResult",
    "NaNProfile",
    "OperationCounters",
    "PackedGaussianChirpAtom",
    "PairAdderDiagnostics",
    "PerformanceCounters",
    "Phase",
    "PrecisionMode",
    "QuadraticJet2D",
    "RawLogCode",
    "ReductionCertificate",
    "ReductionMethod",
    "ReductionResult",
    "ReductionRoute",
    "ReductionSpec",
    "ReductionStats",
    "Route",
    "SPF32Value",
    "SeparableField",
    "SphericalGaussian",
    "Transport",
    "V02ReductionResult",
    "V02ReductionStats",
    "ValueKind",
    "alpha_to_transmittance",
    "atom_affine",
    "atom_bound",
    "atom_bound_log",
    "atom_convolve",
    "atom_derivative",
    "atom_div",
    "atom_fourier",
    "atom_hessian",
    "atom_integral",
    "atom_inverse_fourier",
    "atom_mul",
    "atom_pixel_footprint",
    "atom_pixel_integral",
    "atom_to_jet",
    "butterfly_fb3",
    "butterfly_reference",
    "canonicalize",
    "circular_diff_u16",
    "complex_add_reference",
    "complex_tail_certificate",
    "complex_tail_certified",
    "compute_stats",
    "compute_v02_stats",
    "decode",
    "decode_complex",
    "decode_words",
    "display_code_from_log",
    "display_lut",
    "encode",
    "encode_complex",
    "encode_words",
    "evaluate_separable",
    "exp2_neg_gap",
    "fb3_hybrid_butterfly",
    "hermitian_pair_indices",
    "hermitian_pair_term",
    "hermitian_pairs",
    "hermitian_synthesis",
    "hermitian_synthesize",
    "is_nan",
    "is_zero",
    "jet_emit_spf32",
    "jet_eval",
    "jet_forward_difference_materialize",
    "jet_forward_differences",
    "jet_from_tensor_block_header",
    "jet_materialize",
    "jet_to_tensor_block_header",
    "jet_transform",
    "log1m_pos",
    "log1p_pos",
    "logabs_cos",
    "logabs_sin",
    "logadd_pos",
    "logadd_signed",
    "logpos_complement",
    "logpos_div",
    "logpos_exposure",
    "logpos_mul",
    "logpos_power",
    "logpos_reinhard",
    "logsub_pos",
    "mixture_add",
    "mixture_evaluate",
    "mixture_evaluate_spf",
    "mixture_materialize",
    "mixture_merge_equal",
    "mixture_prune",
    "optical_depth_add",
    "pack_spf32",
    "packed_atom_div",
    "packed_atom_mul",
    "pair_addsub_hybrid",
    "phase_code_from_wide",
    "phase_compose",
    "positive_tail_certificate",
    "positive_tail_certified",
    "reduce_auto",
    "reduce_complex",
    "reduce_logpos",
    "reduce_logsigned",
    "reduce_optical_depth",
    "reduce_phase",
    "reduce_positive",
    "reduce_signed",
    "reduce_transport",
    "reduce_v02",
    "reduce_values",
    "reduction_stats",
    "route",
    "route_reduction",
    "select_route",
    "separable_evaluate",
    "spf_abs2",
    "spf_add",
    "spf_add_complex_reference",
    "spf_butterfly_fb3",
    "spf_butterfly_fb3_hybrid",
    "spf_butterfly_reference",
    "spf_conj",
    "spf_div",
    "spf_fft_radix2",
    "spf_ifft_radix2",
    "spf_mul",
    "spf_neg",
    "spf_pair_addsub_hybrid",
    "spf_pair_addsub_hybrid_diagnostics",
    "spf_phase_shift",
    "spherical_gaussian_integral",
    "spherical_gaussian_mul",
    "spherical_gaussian_product",
    "tone_exposure",
    "tone_power",
    "tone_reinhard",
    "transport_compose",
    "transport_reduce_ordered",
    "tropical_max",
    "tropical_min",
    "unpack_L",
    "unpack_T",
    "unpack_spf32",
    "unpack_theta",
    "v02_phase_shift",
]
