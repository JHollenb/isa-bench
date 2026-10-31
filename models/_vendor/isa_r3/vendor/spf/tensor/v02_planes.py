"""Plane-typed v0.2 kinds: ``LogPos``/``LogSigned``/``Phase`` over word planes.

Design v0.2 §6 defines LOG_POS, LOG_SIGNED and PHASE as *performance types*
over SPF32 word planes.  :mod:`spf.tensor.v02` holds the scalar (Python-int)
reference objects; this module holds their vectorized counterparts.  Every
plane operation is the same arithmetic applied elementwise, delegating word
packing, unpacking, multiplication, division, conjugation and phase shift to
:mod:`spf.tensor.core` (through the ``v02`` array paths) and the log-add /
log-sub corrections to the shared ``LOGADD_PLUS``/``LOGADD_MINUS`` ROM.  The
scalar objects remain the oracle: ``tests/v02/test_v02_planes_oracle.py``
asserts bit identity of every plane op against the scalar function applied
elementwise, sentinels included.

Reductions over an axis (:func:`reduce_logpos`, :func:`reduce_logsigned`,
:func:`reduce_complex`) run the same pivoted fixed-point accumulator as the
scalar reducers, one pivot per output element, and return a
:class:`PlaneReduction` carrying per-element certificates plus the aggregate
:class:`~spf.tensor.v02.ErrorLedger`/``PerformanceCounters``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from . import core as _core
from . import exp2_rom as _exp2_rom
from . import v02
from .logadd_rom import LOGADD_G_ZERO, LOGADD_MINUS, LOGADD_PLUS
from .v02 import (
    SPF32_L_MAX_FINITE,
    SPF32_L_MIN_FINITE,
    SPF32_L_SCALE,
    SPF32_L_ZERO,
    SPF32_NAN_L,
    SPF32_NAN_T,
    SPF32_PHASE_HALF,
    SPF32_PHASE_MASK,
    SPF32_PHASE_MOD,
    SPF32_PHASE_QUARTER,
    SPF32_WORD_NAN,
    SPF32_WORD_ZERO,
    ErrorLedger,
    LogPos,
    LogSigned,
    NaNProfile,
    PerformanceCounters,
    Phase,
    ReductionSpec,
)

_PHASE_TO_RAD = 2.0 * math.pi / SPF32_PHASE_MOD
_RAD_TO_PHASE = SPF32_PHASE_MOD / (2.0 * math.pi)
_LOG_QUANTIZATION_RELATIVE_ERROR = 2.0 ** (1.0 / (2.0 * SPF32_L_SCALE)) - 1.0
_PAIR_PHASE_QUANTIZATION = math.pi / SPF32_PHASE_MOD
_LOGADD_PLUS = np.asarray(LOGADD_PLUS, dtype=np.int64)
_LOGADD_MINUS = np.asarray(LOGADD_MINUS, dtype=np.int64)
_LOGADD_PLUS.setflags(write=False)
_LOGADD_MINUS.setflags(write=False)


# ---------------------------------------------------------------------------
# Native plane kernels (Design §37 CPU/ALU/RAM backend)
# ---------------------------------------------------------------------------
#
# ``libspf32_planes`` executes the same plane operations on the same words and
# is asserted bit-identical to the code below by
# ``tests/v02/test_v02_planes_native_parity.py``.  Dispatch is therefore purely
# a performance choice: it never changes an output word or an error bound.
# ``set_native(False)`` (or ``SPF_PLANES_DISABLE=1``) forces this module's own
# NumPy path, which is what the oracle tests grade against.
#
# Elementwise kernels only take the native path above a small size threshold so
# a scalar-sized call is not dominated by the ctypes boundary; the reducers,
# the pair adder and the transport folds always prefer it because their Python
# path runs a per-row loop.

_NATIVE_MIN_ELEMENTS = 256
_NATIVE_OVERRIDE: bool | None = None


def _native_module() -> Any:
    try:
        from spf.native import _libspf32_planes as module
    except Exception:  # pragma: no cover - optional build
        return None
    try:
        return module if module.available() else None
    except OSError:  # pragma: no cover - unloadable build
        return None


def native_available() -> bool:
    """True when ``libspf32_planes`` is loadable (never compiles anything)."""

    return _native_module() is not None


def set_native(enabled: bool | None) -> None:
    """Force (``True``) / forbid (``False``) native dispatch, or restore auto (``None``)."""

    global _NATIVE_OVERRIDE
    _NATIVE_OVERRIDE = None if enabled is None else bool(enabled)


def native_enabled() -> bool:
    """Whether plane ops currently dispatch to the native kernels."""

    if _NATIVE_OVERRIDE is False:
        return False
    return native_available()


def _native(*, size: int | None = None) -> Any:
    """The native module when it should be used for a call of ``size`` elements."""

    if _NATIVE_OVERRIDE is False:
        return None
    if size is not None and size < _NATIVE_MIN_ELEMENTS:
        return None
    return _native_module()


# ---------------------------------------------------------------------------
# Elementwise helpers (each mirrors one scalar rule in ``spf.tensor.v02``)
# ---------------------------------------------------------------------------


def _rint(values: npt.ArrayLike) -> np.ndarray:
    """Round-to-nearest-even, the array spelling of ``v02._rne``."""

    return np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)


def _clamp_finite(values: npt.ArrayLike) -> np.ndarray:
    """``v02._clamp_finite_l`` over an int64 array (never yields ``L_ZERO``)."""

    return np.clip(np.asarray(values, dtype=np.int64), SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE)


def _canonical_logpos_L(values: npt.ArrayLike) -> np.ndarray:
    """``LogPos.__post_init__`` over an array: NaN code kept, else canonical."""

    L = np.asarray(values, dtype=np.int64)
    nan = L == SPF32_NAN_L
    zero = L == SPF32_L_ZERO
    out = np.clip(L, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE)
    out = np.where(zero, SPF32_L_ZERO, out)
    return np.where(nan, SPF32_NAN_L, out)


def _logadd_correction(gap: np.ndarray, *, minus: bool) -> np.ndarray:
    table = _LOGADD_MINUS if minus else _LOGADD_PLUS
    safe = np.minimum(gap, LOGADD_G_ZERO - 1)
    correction = table[safe]
    return np.where(gap >= LOGADD_G_ZERO, 0, correction)


def _logadd_pos_L(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``v02.logadd_pos`` on canonical positive log codes (int64 arrays)."""

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    library = _native(size=max(a.size, b.size))
    if library is not None:
        return library.logadd_pos(a.astype(np.int16), b.astype(np.int16)).astype(np.int64)
    nan = (a == SPF32_NAN_L) | (b == SPF32_NAN_L)
    a_zero = a == SPF32_L_ZERO
    b_zero = b == SPF32_L_ZERO
    left = np.maximum(a, b)
    gap = np.abs(a - b)
    result = _clamp_finite(left + _logadd_correction(gap, minus=False))
    result = np.where(a_zero, b, np.where(b_zero, a, result))
    return np.where(nan, SPF32_NAN_L, result)


def _logsub_pos_L(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``v02.logsub_pos`` on canonical positive log codes; ``a >= b`` required."""

    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    nan = (a == SPF32_NAN_L) | (b == SPF32_NAN_L)
    a_zero = a == SPF32_L_ZERO
    b_zero = b == SPF32_L_ZERO
    both = ~(nan | a_zero | b_zero)
    if np.any(both & (a < b)):
        raise ValueError("logsub_pos requires a >= b >= 0")
    library = _native(size=max(a.size, b.size))
    if library is not None:
        return library.logsub_pos(a.astype(np.int16), b.astype(np.int16)).astype(np.int64)
    gap = a - b
    result = _clamp_finite(a + _logadd_correction(np.maximum(gap, 0), minus=True))
    result = np.where(gap == 0, SPF32_L_ZERO, result)
    result = np.where(a_zero, SPF32_NAN_L, result)
    result = np.where(b_zero, a, result)
    return np.where(nan, SPF32_NAN_L, result)


# ---------------------------------------------------------------------------
# LOG_POS plane
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogPosPlane:
    """A plane of non-negative reals stored as canonical int16 SPF32 log codes.

    Elementwise semantics equal :class:`spf.tensor.v02.LogPos`: ``L_ZERO`` is
    zero, ``NAN_L`` (32767) is NaN, every other code is finite and
    ``2**(L/256)``.
    """

    L: np.ndarray

    def __post_init__(self) -> None:
        codes = _canonical_logpos_L(np.asarray(self.L)).astype(np.int16)
        codes.setflags(write=False)
        object.__setattr__(self, "L", codes)

    # -- construction -----------------------------------------------------
    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> LogPosPlane:
        return cls(np.full(shape, SPF32_L_ZERO, dtype=np.int16))

    @classmethod
    def ones(cls, shape: tuple[int, ...]) -> LogPosPlane:
        return cls(np.zeros(shape, dtype=np.int16))

    @classmethod
    def from_codes(cls, codes: npt.ArrayLike) -> LogPosPlane:
        return cls(np.asarray(codes))

    @classmethod
    def from_words(cls, words: npt.ArrayLike) -> LogPosPlane:
        """``LogPos(unpack_L(word))`` elementwise via ``core.unpack_L``."""

        raw = np.asarray(words, dtype=np.uint32)
        return cls(np.asarray(_core.unpack_L(raw), dtype=np.int64))

    @classmethod
    def from_values(cls, values: npt.ArrayLike) -> LogPosPlane:
        """``LogPos.from_value`` elementwise, encoded through ``core``."""

        array = np.asarray(values, dtype=np.float64)
        if np.any(array < 0.0):
            raise ValueError("LogPos requires a non-negative value")
        words = np.asarray(v02.encode_complex(array.astype(np.complex128)), dtype=np.uint32).reshape(array.shape)
        return cls.from_words(words)

    @classmethod
    def from_scalars(cls, values: Any) -> LogPosPlane:
        """Stack scalar ``LogPos`` objects (or reals) into a plane."""

        array = np.asarray(values, dtype=object)
        codes = np.empty(array.shape, dtype=np.int64)
        for index in np.ndindex(array.shape):
            item = array[index]
            codes[index] = item.L if isinstance(item, LogPos) else LogPos.from_value(float(item)).L
        return cls(codes)

    # -- views ------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape)

    @property
    def size(self) -> int:
        return int(self.L.size)

    @property
    def is_zero(self) -> np.ndarray:
        return self.L == SPF32_L_ZERO

    @property
    def is_nan(self) -> np.ndarray:
        return self.L == SPF32_NAN_L

    def to_words(self) -> np.ndarray:
        """``pack_spf32(LogPos)`` elementwise via ``core.pack_words``."""

        T = np.where(self.is_nan, SPF32_NAN_T, 0).astype(np.uint16)
        return np.asarray(_core.pack_words(self.L, T), dtype=np.uint32)

    def to_values(self) -> np.ndarray:
        """Decode to float64 through ``core.decode_words`` (one output boundary)."""

        return np.asarray(np.real(v02.decode_complex(self.to_words())), dtype=np.float64).reshape(self.shape)

    def scalar(self, index: Any) -> LogPos:
        return LogPos(int(self.L[index]))

    def __getitem__(self, index: Any) -> LogPosPlane:
        return LogPosPlane(np.asarray(self.L[index]))

    def reshape(self, shape: tuple[int, ...]) -> LogPosPlane:
        return LogPosPlane(self.L.reshape(shape))

    # -- arithmetic (delegating to the v02/core array paths) -------------
    def mul(self, other: LogPosPlane) -> LogPosPlane:
        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.logpos_mul(self.L, other.L))
        return LogPosPlane.from_words(v02.spf_mul(self.to_words(), other.to_words(), NaNProfile.STRICT_V02))

    def div(self, other: LogPosPlane) -> LogPosPlane:
        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.logpos_div(self.L, other.L))
        return LogPosPlane.from_words(v02.spf_div(self.to_words(), other.to_words(), NaNProfile.STRICT_V02))

    def pow_int(self, exponent: int) -> LogPosPlane:
        """``logpos_power`` with an integer exponent (exact log-code scaling)."""

        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.logpos_pow_int(self.L, int(exponent)))
        n = int(exponent)
        L = self.L.astype(np.int64)
        result = _clamp_finite(n * L)
        if n > 0:
            result = np.where(self.is_zero, SPF32_L_ZERO, result)
        elif n == 0:
            result = np.where(self.is_zero, 0, result)
        else:
            result = np.where(self.is_zero, SPF32_NAN_L, result)
        return LogPosPlane(np.where(self.is_nan, SPF32_NAN_L, result))

    def add(self, other: LogPosPlane) -> LogPosPlane:
        """``logadd_pos`` elementwise through the ``LOGADD_PLUS`` ROM."""

        return LogPosPlane(_logadd_pos_L(self.L, other.L))

    def sub(self, other: LogPosPlane) -> LogPosPlane:
        """``logsub_pos`` elementwise; requires ``self >= other`` where both finite."""

        return LogPosPlane(_logsub_pos_L(self.L, other.L))

    def log1p(self) -> LogPosPlane:
        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.log1p(self.L))
        return LogPosPlane.ones(self.shape).add(self)

    def log1m(self) -> LogPosPlane:
        """§18.3 complement ``1 - x`` through the ``LOGADD_MINUS`` ROM."""

        library = _native(size=self.size)
        if library is not None:
            finite = ~(self.is_zero | self.is_nan)
            if np.any(finite & (self.L > 0)):
                raise ValueError("logsub_pos requires a >= b >= 0")
            return LogPosPlane(library.log1m(self.L))
        return LogPosPlane.ones(self.shape).sub(self)

    complement = log1m
    alpha_to_transmittance = log1m

    def exposure(self, ev: float) -> LogPosPlane:
        if not math.isfinite(float(ev)):
            raise ValueError("exposure must be finite")
        shift = int(v02._rne(SPF32_L_SCALE * float(ev)))
        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.logpos_exposure(self.L, shift))
        keep = self.is_zero | self.is_nan
        return LogPosPlane(np.where(keep, self.L, _clamp_finite(self.L.astype(np.int64) + shift)))

    def power(self, exponent: float) -> LogPosPlane:
        p = float(exponent)
        if not math.isfinite(p):
            raise ValueError("power must be finite")
        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.logpos_power(self.L, p))
        result = _clamp_finite(_rint(p * self.L.astype(np.float64)))
        if p > 0.0:
            result = np.where(self.is_zero, SPF32_L_ZERO, result)
        elif p == 0.0:
            result = np.where(self.is_zero, 0, result)
        else:
            result = np.where(self.is_zero, SPF32_NAN_L, result)
        return LogPosPlane(np.where(self.is_nan, SPF32_NAN_L, result))

    def reinhard(self) -> LogPosPlane:
        library = _native(size=self.size)
        if library is not None:
            return LogPosPlane(library.logpos_reinhard(self.L))
        denominator = self.log1p()
        keep = self.is_zero | self.is_nan
        result = _clamp_finite(self.L.astype(np.int64) - denominator.L.astype(np.int64))
        return LogPosPlane(np.where(keep, self.L, result))

    def as_signed(self) -> LogSignedPlane:
        sign = np.where(self.is_zero | self.is_nan, 0, 1).astype(np.int8)
        return LogSignedPlane(self.L, sign)

    def reduce(self, axis: int | None = None, spec: ReductionSpec | float | None = None, **kw: Any) -> PlaneReduction:
        return reduce_logpos(self, axis=axis, spec=spec, **kw)

    def max_code(self) -> int:
        finite = self.L[~(self.is_zero | self.is_nan)]
        return int(finite.max()) if finite.size else SPF32_L_ZERO


# ---------------------------------------------------------------------------
# LOG_SIGNED plane
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LogSignedPlane:
    """A plane of signed reals ``sign * 2**(L/256)`` (``LogSigned`` elementwise).

    ``sign`` is ``-1``/``+1`` for finite nonzero values and ``0`` for both
    zero and NaN, exactly as :class:`spf.tensor.v02.LogSigned` stores it.
    """

    L: np.ndarray
    sign: np.ndarray

    def __post_init__(self) -> None:
        L = np.asarray(self.L, dtype=np.int64)
        sign = np.asarray(self.sign, dtype=np.int64)
        L, sign = np.broadcast_arrays(L, sign)
        nan = L == SPF32_NAN_L
        zero = (L == SPF32_L_ZERO) | (sign == 0)
        codes = np.clip(L, SPF32_L_MIN_FINITE, SPF32_L_MAX_FINITE)
        codes = np.where(zero, SPF32_L_ZERO, codes)
        codes = np.where(nan, SPF32_NAN_L, codes)
        signs = np.where(sign > 0, 1, -1)
        signs = np.where(zero | nan, 0, signs)
        codes = codes.astype(np.int16)
        signs = signs.astype(np.int8)
        codes.setflags(write=False)
        signs.setflags(write=False)
        object.__setattr__(self, "L", codes)
        object.__setattr__(self, "sign", signs)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> LogSignedPlane:
        return cls(np.full(shape, SPF32_L_ZERO, dtype=np.int16), np.zeros(shape, dtype=np.int8))

    @classmethod
    def from_words(cls, words: npt.ArrayLike) -> LogSignedPlane:
        """Signed-real words: phase 0 is ``+``, phase ``2**15`` is ``-``."""

        raw = np.asarray(words, dtype=np.uint32)
        L = np.asarray(_core.unpack_L(raw), dtype=np.int64)
        T = np.asarray(_core.unpack_theta(raw), dtype=np.int64)
        nan = raw == np.uint32(SPF32_WORD_NAN)
        zero = L == SPF32_L_ZERO
        if np.any(~nan & ~zero & (T != 0) & (T != SPF32_PHASE_HALF)):
            raise ValueError("LogSignedPlane requires signed-real words (phase 0 or 2**15)")
        sign = np.where(T == SPF32_PHASE_HALF, -1, 1)
        sign = np.where(zero | nan, 0, sign)
        return cls(np.where(nan, SPF32_NAN_L, L), sign)

    @classmethod
    def from_values(cls, values: npt.ArrayLike) -> LogSignedPlane:
        array = np.asarray(values, dtype=np.float64)
        words = np.asarray(v02.encode_complex(array.astype(np.complex128)), dtype=np.uint32).reshape(array.shape)
        return cls.from_words(words)

    @classmethod
    def from_scalars(cls, values: Any) -> LogSignedPlane:
        array = np.asarray(values, dtype=object)
        codes = np.empty(array.shape, dtype=np.int64)
        signs = np.empty(array.shape, dtype=np.int64)
        for index in np.ndindex(array.shape):
            item = array[index]
            if isinstance(item, LogPos):
                item = LogSigned(item.L, 1) if not item.is_nan else LogSigned.nan()
            elif not isinstance(item, LogSigned):
                item = LogSigned.from_value(float(item))
            codes[index], signs[index] = item.L, item.sign
        return cls(codes, signs)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.L.shape)

    @property
    def is_nan(self) -> np.ndarray:
        return self.L == SPF32_NAN_L

    @property
    def is_zero(self) -> np.ndarray:
        return ~self.is_nan & (self.sign == 0)

    def to_words(self) -> np.ndarray:
        T = np.where(self.sign < 0, SPF32_PHASE_HALF, 0)
        T = np.where(self.is_nan, SPF32_NAN_T, T).astype(np.uint16)
        return np.asarray(_core.pack_words(self.L, T), dtype=np.uint32)

    def to_values(self) -> np.ndarray:
        return np.asarray(np.real(v02.decode_complex(self.to_words())), dtype=np.float64).reshape(self.shape)

    def scalar(self, index: Any) -> LogSigned:
        return LogSigned(int(self.L[index]), int(self.sign[index]))

    def __getitem__(self, index: Any) -> LogSignedPlane:
        return LogSignedPlane(np.asarray(self.L[index]), np.asarray(self.sign[index]))

    def reshape(self, shape: tuple[int, ...]) -> LogSignedPlane:
        return LogSignedPlane(self.L.reshape(shape), self.sign.reshape(shape))

    def abs(self) -> LogPosPlane:
        return LogPosPlane(self.L)

    def neg(self) -> LogSignedPlane:
        return LogSignedPlane(self.L, -self.sign.astype(np.int64))

    def mul(self, other: LogSignedPlane | LogPosPlane) -> LogSignedPlane:
        """``v02._logsigned_mul`` elementwise (sign product, saturated log add)."""

        rhs = other.as_signed() if isinstance(other, LogPosPlane) else other
        library = _native(size=self.L.size)
        if library is not None:
            codes, signs = library.logsigned_mul(self.L, self.sign, rhs.L, rhs.sign)
            return LogSignedPlane(codes, signs)
        nan = self.is_nan | rhs.is_nan
        zero = self.is_zero | rhs.is_zero
        L = _clamp_finite(self.L.astype(np.int64) + rhs.L.astype(np.int64))
        sign = self.sign.astype(np.int64) * rhs.sign.astype(np.int64)
        L = np.where(zero, SPF32_L_ZERO, L)
        L = np.where(nan, SPF32_NAN_L, L)
        return LogSignedPlane(L, np.where(zero | nan, 0, sign))

    def add(self, other: LogSignedPlane | LogPosPlane) -> LogSignedPlane:
        """``v02.logadd_signed`` elementwise through the plus/minus ROMs."""

        rhs = other.as_signed() if isinstance(other, LogPosPlane) else other
        library = _native(size=self.L.size)
        if library is not None:
            codes, signs = library.logadd_signed(self.L, self.sign, rhs.L, rhs.sign)
            return LogSignedPlane(codes, signs)
        a, sa = self.L.astype(np.int64), self.sign.astype(np.int64)
        b, sb = rhs.L.astype(np.int64), rhs.sign.astype(np.int64)
        nan = self.is_nan | rhs.is_nan
        a_zero = self.is_zero
        b_zero = rhs.is_zero
        same = sa == sb
        big = np.maximum(a, b)
        gap = np.abs(a - b)
        plus = _clamp_finite(big + _logadd_correction(gap, minus=False))
        minus = _clamp_finite(big + _logadd_correction(gap, minus=True))
        L = np.where(same, plus, np.where(gap == 0, SPF32_L_ZERO, minus))
        sign = np.where(same, sa, np.where(gap == 0, 0, np.where(a > b, sa, sb)))
        L = np.where(b_zero, a, np.where(a_zero, b, L))
        sign = np.where(b_zero, sa, np.where(a_zero, sb, sign))
        L = np.where(nan, SPF32_NAN_L, L)
        sign = np.where(nan, 0, sign)
        return LogSignedPlane(L, sign)

    def as_signed(self) -> LogSignedPlane:
        return self

    def reduce(self, axis: int | None = None, spec: ReductionSpec | float | None = None, **kw: Any) -> PlaneReduction:
        return reduce_logsigned(self, axis=axis, spec=spec, **kw)


# ---------------------------------------------------------------------------
# PHASE plane
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhasePlane:
    """A plane of wrapped uint16 SPF phase codes (``Phase`` elementwise)."""

    T: np.ndarray

    def __post_init__(self) -> None:
        codes = (np.asarray(self.T, dtype=np.int64) & SPF32_PHASE_MASK).astype(np.uint16)
        codes.setflags(write=False)
        object.__setattr__(self, "T", codes)

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> PhasePlane:
        return cls(np.zeros(shape, dtype=np.uint16))

    @classmethod
    def from_words(cls, words: npt.ArrayLike) -> PhasePlane:
        return cls(np.asarray(_core.unpack_theta(np.asarray(words, dtype=np.uint32)), dtype=np.int64))

    @classmethod
    def from_radians(cls, angles: npt.ArrayLike) -> PhasePlane:
        return cls(_rint(np.asarray(angles, dtype=np.float64) * _RAD_TO_PHASE))

    @classmethod
    def from_scalars(cls, values: Any) -> PhasePlane:
        array = np.asarray(values, dtype=object)
        codes = np.empty(array.shape, dtype=np.int64)
        for index in np.ndindex(array.shape):
            item = array[index]
            codes[index] = item.T if isinstance(item, Phase) else int(item)
        return cls(codes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.T.shape)

    @property
    def theta(self) -> np.ndarray:
        return self.T

    @property
    def radians(self) -> np.ndarray:
        return self.T.astype(np.float64) * _PHASE_TO_RAD

    def to_words(self, log_codes: npt.ArrayLike | None = None) -> np.ndarray:
        """Pack as unit-magnitude words (``L=0``) or onto the supplied L plane."""

        L = np.zeros(self.shape, dtype=np.int16) if log_codes is None else np.asarray(log_codes, dtype=np.int16)
        return np.asarray(_core.pack_words(L, self.T), dtype=np.uint32)

    def scalar(self, index: Any) -> Phase:
        return Phase(int(self.T[index]))

    def compose(self, other: PhasePlane | int) -> PhasePlane:
        if isinstance(other, PhasePlane):
            library = _native(size=self.T.size)
            if library is not None and self.T.shape == other.T.shape:
                return PhasePlane(library.phase_compose(self.T, other.T))
            delta: Any = other.T.astype(np.int64)
        else:
            delta = int(other)
        return PhasePlane(self.T.astype(np.int64) + delta)

    add = compose
    shift = compose

    def neg(self) -> PhasePlane:
        return PhasePlane(-self.T.astype(np.int64))

    conj = neg

    def reduce(self, axis: int | None = None) -> PhasePlane:
        """``reduce_phase``: modular sum over ``axis``."""

        total = np.sum(self.T.astype(np.int64), axis=axis, dtype=np.int64)
        return PhasePlane(np.asarray(total))


# ---------------------------------------------------------------------------
# Complex word-plane helpers (packed SPF32 words stay the carrier)
# ---------------------------------------------------------------------------


def words_mul(a: npt.ArrayLike, b: npt.ArrayLike, profile: NaNProfile = NaNProfile.STRICT_V02) -> np.ndarray:
    left = np.asarray(a, dtype=np.uint32)
    right = np.asarray(b, dtype=np.uint32)
    library = _native(size=max(left.size, right.size))
    if library is not None:
        return library.words_mul(left, right, frozen=profile is NaNProfile.FROZEN_COMPAT)
    return np.asarray(v02.spf_mul(left, right, profile))


def words_div(a: npt.ArrayLike, b: npt.ArrayLike, profile: NaNProfile = NaNProfile.STRICT_V02) -> np.ndarray:
    left = np.asarray(a, dtype=np.uint32)
    right = np.asarray(b, dtype=np.uint32)
    library = _native(size=max(left.size, right.size))
    if library is not None:
        return library.words_div(left, right, frozen=profile is NaNProfile.FROZEN_COMPAT)
    return np.asarray(v02.spf_div(left, right, profile))


def words_conj(a: npt.ArrayLike, profile: NaNProfile = NaNProfile.STRICT_V02) -> np.ndarray:
    words = np.asarray(a, dtype=np.uint32)
    library = _native(size=words.size)
    if library is not None:
        return library.words_conj(words, frozen=profile is NaNProfile.FROZEN_COMPAT)
    return np.asarray(v02.spf_conj(words, profile))


def words_neg(a: npt.ArrayLike, profile: NaNProfile = NaNProfile.STRICT_V02) -> np.ndarray:
    words = np.asarray(a, dtype=np.uint32)
    library = _native(size=words.size)
    if library is not None:
        return library.words_neg(words, frozen=profile is NaNProfile.FROZEN_COMPAT)
    return np.asarray(v02.spf_neg(words, profile))


def words_phase_shift(
    a: npt.ArrayLike, delta: npt.ArrayLike, profile: NaNProfile = NaNProfile.STRICT_V02
) -> np.ndarray:
    """``spf_phase_shift`` with a per-element phase delta plane."""

    words = np.asarray(a, dtype=np.uint32)
    shift = np.asarray(delta, dtype=np.int64)
    library = _native(size=words.size)
    if library is not None:
        return library.words_phase_shift(words, shift, frozen=profile is NaNProfile.FROZEN_COMPAT)
    if shift.ndim == 0:
        return np.asarray(v02.spf_phase_shift(words, int(shift), profile))
    frozen = profile is NaNProfile.FROZEN_COMPAT
    raw = words if frozen else v02._canonical_words(words)
    L = np.asarray(_core.unpack_L(raw), dtype=np.int64)
    T = np.asarray(_core.unpack_theta(raw), dtype=np.int64)
    out = np.asarray(_core.pack_words(L.astype(np.int16), ((T + shift) & SPF32_PHASE_MASK).astype(np.uint16)))
    out = np.where(L == SPF32_L_ZERO, raw, out)
    if not frozen:
        out = np.where(raw == np.uint32(SPF32_WORD_NAN), raw, out)
    return np.asarray(out, dtype=np.uint32)


def words_abs2(a: npt.ArrayLike) -> LogPosPlane:
    """``spf_abs2`` elementwise: ``|z|**2`` as a LOG_POS plane."""

    words = np.asarray(a, dtype=np.uint32)
    library = _native(size=words.size)
    if library is not None:
        return LogPosPlane(library.words_abs2(words))
    raw = v02._canonical_words(words)
    L = np.asarray(_core.unpack_L(raw), dtype=np.int64)
    nan = raw == np.uint32(SPF32_WORD_NAN)
    zero = L == SPF32_L_ZERO
    out = _clamp_finite(2 * L)
    out = np.where(zero, SPF32_L_ZERO, out)
    return LogPosPlane(np.where(nan, SPF32_NAN_L, out))


def words_planes(a: npt.ArrayLike) -> tuple[LogPosPlane, PhasePlane]:
    raw = v02._canonical_words(np.asarray(a, dtype=np.uint32))
    return LogPosPlane.from_words(raw), PhasePlane.from_words(raw)


# ---------------------------------------------------------------------------
# Pivoted reductions over an axis
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlaneReduction:
    """Per-element reduction certificates for one axis reduction of a plane."""

    value: Any
    value_kind: str
    method: str
    absolute_error_bound: np.ndarray
    relative_error_bound: np.ndarray
    phase_error_bound: np.ndarray
    phase_reliable: np.ndarray
    retained_terms: np.ndarray
    dropped_terms: np.ndarray
    accumulator_fraction_bits: np.ndarray
    saturated: np.ndarray
    counters: PerformanceCounters = field(default_factory=PerformanceCounters)
    ledger: ErrorLedger = field(default_factory=ErrorLedger)
    #: True when the pivot accumulator ran in ``libspf32_planes`` (Design §37).
    #: The words and every bound are identical either way; this is provenance,
    #: not a semantic flag.
    native: bool = False

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(np.shape(self.absolute_error_bound))

    @property
    def total_error_bound(self) -> float:
        bounds = np.asarray(self.absolute_error_bound, dtype=np.float64).reshape(-1)
        if bounds.size == 0:
            return 0.0
        if np.any(np.isnan(bounds)):
            return math.nan
        return float(np.max(bounds))

    def result(self, index: Any = ()) -> v02.ReductionResult:
        """Scalar :class:`~spf.tensor.v02.ReductionResult` for one output element."""

        if self.value_kind == "SPF_COMPLEX":
            value: Any = int(np.asarray(self.value)[index])
        else:
            value = self.value.scalar(index)
        return v02.ReductionResult(
            value,
            self.value_kind,
            self.method,
            float(np.asarray(self.absolute_error_bound)[index]),
            float(np.asarray(self.relative_error_bound)[index]),
            float(np.asarray(self.phase_error_bound)[index]),
            bool(np.asarray(self.phase_reliable)[index]),
            int(np.asarray(self.retained_terms)[index]),
            int(np.asarray(self.dropped_terms)[index]),
            int(np.asarray(self.accumulator_fraction_bits)[index]),
            bool(np.asarray(self.saturated)[index]),
        )


def _native_spec(policy: ReductionSpec, explicit: int | None) -> dict[str, Any]:
    """``ReductionSpec`` as the C ``fraction_bits`` arguments."""

    accumulator = policy.accumulator_fraction_bits
    return {
        "precision": v02._precision_name(policy.precision_mode),
        "explicit_bits": None if explicit is None else max(0, int(explicit)),
        "accumulator_fraction_bits": None if accumulator is None else max(0, int(accumulator)),
        "budget_abs": float(policy.error_budget_abs),
        "budget_rel": float(policy.error_budget_rel),
    }


def _native_reduction(
    result: dict[str, np.ndarray],
    out_shape: tuple[int, ...],
    *,
    kind: str,
    terms_seen: int,
    count: int,
) -> PlaneReduction:
    """Assemble a :class:`PlaneReduction` from the native per-row certificates."""

    kept = result["retained"]
    has_nan = result["has_nan"].astype(bool)
    live = np.where(has_nan, 0, kept)
    absolute = result["absolute"]
    element_bytes = {"LOG_POS": 2, "LOG_SIGNED": 3, "SPF_COMPLEX": 4}[kind]
    counters = PerformanceCounters(
        terms_seen=terms_seen,
        terms_retained=int(kept.sum()),
        bytes_read=terms_seen * element_bytes,
        bytes_written=element_bytes * count,
        aligned_mantissas=int(live.sum()),
        pivot_reductions=int(np.count_nonzero(live)),
        log_ops=int(np.count_nonzero(live)),
        exp2_ops=int(live.sum()),
    )
    if kind == "LOG_POS":
        value: Any = LogPosPlane(result["value_L"].reshape(out_shape) if out_shape else result["value_L"])
        counters.values_materialized = int(np.count_nonzero(has_nan))
        method = "pivot_logpos" if np.all(kept > 0) else "zero" if not np.any(kept) else "pivot_logpos"
    elif kind == "LOG_SIGNED":
        codes = result["value_L"]
        signs = result["value_sign"]
        value = LogSignedPlane(
            codes.reshape(out_shape) if out_shape else codes, signs.reshape(out_shape) if out_shape else signs
        )
        method = "pivot_logsigned" if np.any(kept) else "zero"
    else:
        value = result["value_word"].reshape(out_shape)
        counters.complex_finalizations = int(np.count_nonzero(live))
        counters.phase_ops = int(live.sum())
        counters.atan2_ops = int(np.count_nonzero(live))
        counters.trig_ops = int(live.sum())
        method = "pivot_complex" if np.any(kept) else "zero"
    return PlaneReduction(
        value,
        kind,
        method,
        absolute.reshape(out_shape),
        result["relative"].reshape(out_shape),
        result["phase_error"].reshape(out_shape),
        result["phase_reliable"].astype(bool).reshape(out_shape),
        kept.reshape(out_shape),
        result["dropped"].reshape(out_shape),
        result["bits"].astype(np.int64).reshape(out_shape),
        result["saturated"].astype(bool).reshape(out_shape),
        counters,
        _ledger_from_bounds(absolute),
        True,
    )


def _rows(values: np.ndarray, axis: int | None) -> tuple[np.ndarray, tuple[int, ...]]:
    array = np.asarray(values)
    if axis is None:
        return array.reshape(1, -1), ()
    moved = np.moveaxis(array, int(axis), -1)
    # An explicit row count preserves empty reduction axes (reshape(-1, 0) cannot).
    return moved.reshape(math.prod(moved.shape[:-1]), moved.shape[-1]), moved.shape[:-1]


def _ledger_from_bounds(bounds: np.ndarray) -> ErrorLedger:
    flat = np.asarray(bounds, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return ErrorLedger()
    if np.any(np.isnan(flat)):
        return ErrorLedger(reducer=math.nan)
    return ErrorLedger(reducer=float(np.max(flat)))


def _pruning_requested(spec: ReductionSpec) -> bool:
    return bool(spec.allow_pruning and (spec.error_budget_abs > 0.0 or spec.error_budget_rel > 0.0))


def _mantissas(rows_L: np.ndarray, pivot: np.ndarray, scale: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``rne(exp2((L - pivot)/256) * scale)`` for kept terms, zero elsewhere."""

    exponent = (rows_L.astype(np.float64) - pivot[:, None]) / SPF32_L_SCALE
    exponent = np.where(keep, exponent, 0.0)
    mantissa = _rint(np.exp2(exponent) * scale[:, None])
    return np.where(keep, mantissa, 0)


def _sum_rows(values: np.ndarray, bits: np.ndarray) -> np.ndarray:
    """Exact integer row sums; falls back to Python ints when int64 could overflow."""

    n = values.shape[1]
    if n and int(np.max(bits)) + math.ceil(math.log2(n)) >= 62:
        return np.asarray([sum(int(v) for v in row) for row in values], dtype=object)
    return np.sum(values, axis=1, dtype=np.int64)


def _finalize_log(pivot: np.ndarray, accumulator: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``pivot + rne(256*log2(acc/scale))`` per row with scalar ``math.log2``."""

    out = np.zeros(pivot.shape, dtype=np.int64)
    for index in range(pivot.shape[0]):
        acc = int(accumulator[index]) if isinstance(accumulator[index], (int, np.integer)) else accumulator[index]
        if acc == 0:
            continue
        out[index] = int(pivot[index]) + v02._rne(SPF32_L_SCALE * math.log2(abs(acc) / float(scale[index])))
    return out


def _row_norms(rows_L: np.ndarray, pivot: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """``fsum(exp2((L - pivot)/256))`` per row over kept terms, via the §8.5 ROM.

    ``pivot`` is the max over kept terms, so the exponent is a non-positive
    integer gap and :func:`_exp2_neg_gap_planes` evaluates it exactly (see that
    function and :func:`_exp2_code_planes` for why ``np.exp2`` is not used).
    """

    gap = np.where(keep, pivot[:, None] - rows_L.astype(np.int64), 0)
    magnitudes = np.where(keep, _exp2_neg_gap_planes(gap), 0.0)
    return np.asarray([math.fsum(row.tolist()) for row in magnitudes], dtype=np.float64)


def reduce_logpos(
    plane: LogPosPlane | npt.ArrayLike,
    *,
    axis: int | None = None,
    spec: ReductionSpec | float | None = None,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    use_native: bool | None = None,
) -> PlaneReduction:
    """Pivoted LOG_POS reduction over ``axis``; bit-identical to ``v02.reduce_logpos`` per output.

    Pruning (``spec.allow_pruning`` with a budget) sorts terms per output and
    is evaluated through the scalar reducer per row.  ``use_native=False``
    forces this module's NumPy accumulator (the oracle); the native §9.2
    accumulator produces the same words and the same bounds.
    """

    source = plane if isinstance(plane, LogPosPlane) else LogPosPlane(np.asarray(plane))
    policy = v02._coerce_spec(spec, error_budget)
    rows, out_shape = _rows(source.L.astype(np.int64), axis)
    explicit = F if F is not None else fraction_bits
    count, n = rows.shape
    if _pruning_requested(policy):
        results = [v02.reduce_logpos([LogPos(int(v)) for v in row], policy, F=explicit) for row in rows]
        return _stack_scalar_results(results, out_shape, kind="LOG_POS")
    library = None if use_native is False else _native()
    if library is not None:
        try:
            result = library.reduce_logpos(rows.astype(np.int16), **_native_spec(policy, explicit))
        except library.PlaneUnsupportedError:
            result = None
        if result is not None:
            return _native_reduction(result, out_shape, kind="LOG_POS", terms_seen=int(rows.size), count=count)
    has_nan = np.any(rows == SPF32_NAN_L, axis=1)
    keep = (rows != SPF32_L_ZERO) & (rows != SPF32_NAN_L)
    kept = np.sum(keep, axis=1)
    pivot = (
        np.max(np.where(keep, rows, np.int64(SPF32_L_ZERO)), axis=1, initial=SPF32_L_ZERO)
        if n
        else np.full(count, SPF32_L_ZERO, dtype=np.int64)
    )
    bits = np.asarray([v02._fraction_bits(policy, int(k), explicit) if k else 0 for k in kept], dtype=np.int64)
    bits = np.where(has_nan, 0, bits)
    scale = np.exp2(bits.astype(np.float64))
    mantissa = _mantissas(rows, pivot, scale, keep)
    accumulator = _sum_rows(mantissa, bits)
    output_raw = _finalize_log(pivot, accumulator, scale)
    nonzero = np.asarray([acc != 0 for acc in accumulator], dtype=bool) & (kept > 0)
    output_L = np.where(nonzero, _clamp_finite(output_raw), SPF32_L_ZERO)
    output_L = np.where(has_nan, SPF32_NAN_L, output_L)
    saturated = nonzero & ((output_raw < SPF32_L_MIN_FINITE) | (output_raw > SPF32_L_MAX_FINITE))
    norm = _row_norms(rows, pivot, keep)
    linear_scale = _exp2_code_planes(pivot)
    value_plane = LogPosPlane(output_L)
    values = np.where(nonzero, _exp2_code_planes(output_L), 0.0)
    kept_sum = linear_scale * norm
    mantissa_error = linear_scale * kept * 0.5 / scale
    quantization = np.where(
        np.isfinite(values), np.abs(values) * _LOG_QUANTIZATION_RELATIVE_ERROR, np.abs(values - kept_sum)
    )
    absolute = mantissa_error + quantization
    full_sum = linear_scale * norm
    bad = saturated | ~np.isfinite(absolute) | ~np.isfinite(full_sum)
    absolute = np.where(bad, math.inf, absolute)
    relative = np.where(bad, 1.0, np.minimum(1.0, absolute / np.maximum(full_sum, 1e-300)))
    absolute = np.where(nonzero, absolute, 0.0)
    relative = np.where(nonzero, relative, np.where(kept > 0, 1.0, 0.0))
    absolute = np.where(has_nan, math.nan, absolute)
    relative = np.where(has_nan, math.nan, relative)
    method = np.where(kept > 0, "pivot_logpos", "zero")
    live = np.where(has_nan, 0, kept)
    counters = PerformanceCounters(
        terms_seen=int(rows.size),
        terms_retained=int(kept.sum()),
        bytes_read=int(rows.size) * 2,
        bytes_written=2 * count,
        aligned_mantissas=int(live.sum()),
        pivot_reductions=int(np.count_nonzero(live)),
        log_ops=int(np.count_nonzero(live)),
        exp2_ops=int(live.sum()),
        values_materialized=int(np.count_nonzero(has_nan)),
    )
    reduction = PlaneReduction(
        value_plane.reshape(out_shape) if out_shape else value_plane,
        "LOG_POS",
        "pivot_logpos" if np.all(method == "pivot_logpos") else "zero" if not np.any(kept) else "pivot_logpos",
        absolute.reshape(out_shape),
        relative.reshape(out_shape),
        np.zeros(out_shape, dtype=np.float64),
        ~has_nan.reshape(out_shape),
        kept.reshape(out_shape),
        # §9.1: this branch is the unpruned accumulator, so nothing is dropped.
        # The pruned branch above returns through ``_stack_scalar_results``.
        np.zeros(out_shape, dtype=np.int64),
        bits.reshape(out_shape),
        saturated.reshape(out_shape),
        counters,
        _ledger_from_bounds(absolute),
    )
    return reduction


def reduce_logsigned(
    plane: LogSignedPlane,
    *,
    axis: int | None = None,
    spec: ReductionSpec | float | None = None,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    use_native: bool | None = None,
) -> PlaneReduction:
    """Pivoted LOG_SIGNED reduction over ``axis``; bit-identical to ``v02.reduce_logsigned``."""

    policy = v02._coerce_spec(spec, error_budget)
    rows, out_shape = _rows(plane.L.astype(np.int64), axis)
    signs, _ = _rows(plane.sign.astype(np.int64), axis)
    explicit = F if F is not None else fraction_bits
    count, n = rows.shape
    library = None if use_native is False else _native()
    if library is not None:
        try:
            result = library.reduce_logsigned(
                rows.astype(np.int16), signs.astype(np.int8), **_native_spec(policy, explicit)
            )
        except library.PlaneUnsupportedError:
            result = None
        if result is not None:
            return _native_reduction(result, out_shape, kind="LOG_SIGNED", terms_seen=int(rows.size), count=count)
    has_nan = np.any(rows == SPF32_NAN_L, axis=1)
    keep = (signs != 0) & (rows != SPF32_NAN_L)
    kept = np.sum(keep, axis=1)
    pivot = (
        np.max(np.where(keep, rows, np.int64(SPF32_L_ZERO)), axis=1, initial=SPF32_L_ZERO)
        if n
        else np.full(count, SPF32_L_ZERO, dtype=np.int64)
    )
    bits = np.asarray([v02._fraction_bits(policy, int(k), explicit) if k else 0 for k in kept], dtype=np.int64)
    bits = np.where(has_nan, 0, bits)
    scale = np.exp2(bits.astype(np.float64))
    mantissa = _mantissas(rows, pivot, scale, keep)
    positive = _sum_rows(np.where(signs > 0, mantissa, 0), bits)
    negative = _sum_rows(np.where(signs < 0, mantissa, 0), bits)
    accumulator = np.asarray([int(p) - int(m) for p, m in zip(positive, negative, strict=True)], dtype=object)
    output_raw = _finalize_log(pivot, accumulator, scale)
    nonzero = np.asarray([acc != 0 for acc in accumulator], dtype=bool) & (kept > 0)
    output_sign = np.asarray([1 if acc > 0 else -1 if acc < 0 else 0 for acc in accumulator], dtype=np.int64)
    output_L = np.where(nonzero, _clamp_finite(output_raw), SPF32_L_ZERO)
    output_L = np.where(has_nan, SPF32_NAN_L, output_L)
    output_sign = np.where(has_nan | ~nonzero, 0, output_sign)
    saturated = nonzero & ((output_raw < SPF32_L_MIN_FINITE) | (output_raw > SPF32_L_MAX_FINITE))
    linear_scale = _exp2_code_planes(pivot)
    rounding = linear_scale * kept * 0.5 / scale
    magnitude = linear_scale * np.asarray([abs(int(acc)) for acc in accumulator], dtype=np.float64) / scale
    quantization = magnitude * _LOG_QUANTIZATION_RELATIVE_ERROR
    absolute = rounding + quantization
    lower = magnitude - rounding
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(lower <= 0.0, math.inf, absolute / lower)
    absolute = np.where(saturated, math.inf, absolute)
    relative = np.where(saturated, math.inf, relative)
    absolute = np.where(nonzero, absolute, np.where(kept > 0, rounding, 0.0))
    relative = np.where(nonzero, relative, np.where(kept > 0, math.inf, 0.0))
    absolute = np.where(has_nan, math.nan, absolute)
    relative = np.where(has_nan, math.nan, relative)
    live = np.where(has_nan, 0, kept)
    counters = PerformanceCounters(
        terms_seen=int(rows.size),
        terms_retained=int(kept.sum()),
        bytes_read=int(rows.size) * 3,
        bytes_written=3 * count,
        aligned_mantissas=int(live.sum()),
        pivot_reductions=int(np.count_nonzero(live)),
        log_ops=int(np.count_nonzero(live)),
        exp2_ops=int(live.sum()),
    )
    value_plane = LogSignedPlane(output_L, output_sign)
    # §9.3 cancellation threshold, mirroring ``v02.reduce_logsigned``: the sign
    # is certified only when the surviving accumulator is at least 2**-12 of
    # the aligned absolute sum.
    aligned_absolute = np.asarray(
        [float(int(p) + int(m)) for p, m in zip(positive, negative, strict=True)], dtype=np.float64
    )
    accumulator_magnitude_int = np.asarray([abs(int(acc)) for acc in accumulator], dtype=np.float64)
    cancellation_ratio = accumulator_magnitude_int / np.maximum(aligned_absolute, 1.0)
    reliable = (cancellation_ratio >= 2.0**-12) & (lower > 0.0) & nonzero
    reliable = np.where(kept == 0, True, reliable) & ~has_nan
    return PlaneReduction(
        value_plane.reshape(out_shape) if out_shape else value_plane,
        "LOG_SIGNED",
        "pivot_logsigned" if np.any(kept) else "zero",
        absolute.reshape(out_shape),
        relative.reshape(out_shape),
        np.zeros(out_shape, dtype=np.float64),
        reliable.reshape(out_shape),
        kept.reshape(out_shape),
        np.zeros(out_shape, dtype=np.int64),
        bits.reshape(out_shape),
        saturated.reshape(out_shape),
        counters,
        _ledger_from_bounds(absolute),
    )


def _sincos_planes(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``v02._sincos_phase`` elementwise with the exact quarter-turn cases."""

    phase = np.asarray(T, dtype=np.int64) & SPF32_PHASE_MASK
    angle = phase.astype(np.float64) * _PHASE_TO_RAD
    sine = np.sin(angle)
    cosine = np.cos(angle)
    for code, (s, c) in (
        (0, (0.0, 1.0)),
        (SPF32_PHASE_QUARTER, (1.0, 0.0)),
        (SPF32_PHASE_HALF, (0.0, -1.0)),
        (3 * SPF32_PHASE_QUARTER, (-1.0, 0.0)),
    ):
        exact = phase == code
        sine = np.where(exact, s, sine)
        cosine = np.where(exact, c, cosine)
    return sine, cosine


def reduce_complex(
    words: npt.ArrayLike,
    *,
    axis: int | None = None,
    spec: ReductionSpec | float | None = None,
    F: int | None = None,
    fraction_bits: int | None = None,
    error_budget: float | None = None,
    use_native: bool | None = None,
    multiplicities: npt.ArrayLike | None = None,
) -> PlaneReduction:
    """Pivoted SPF_COMPLEX reduction over ``axis``; bit-identical to ``v02.reduce_complex``.

    Input is a packed SPF32 word plane; the output is a packed word plane.
    ``core.reduce_words_result(row, ReducerSpec(policy=V02))`` evaluates the
    same reducer one row at a time.
    Optional nonnegative integer ``multiplicities`` reuse each aligned integer
    coordinate that many times. This opt-in path uses NumPy, excludes zero
    counts, and reports the distinct input count in its work counters. At the
    same fraction-bit setting it implements the expanded coordinate sum, not
    repeated packed addition.
    """

    policy = v02._coerce_spec(spec, error_budget)
    raw = v02._canonical_words(np.asarray(words, dtype=np.uint32))
    rows, out_shape = _rows(raw, axis)
    weights = None
    if multiplicities is not None:
        supplied = np.asarray(multiplicities)
        if supplied.dtype.kind not in "iu" or np.any(supplied < 0) or np.any(supplied > (1 << 31)):
            raise ValueError("multiplicities must be nonnegative bounded integer counts")
        weights, _ = _rows(np.broadcast_to(supplied, raw.shape), axis)
        weights = weights.astype(np.int64)
    explicit = F if F is not None else fraction_bits
    count, n = rows.shape
    library = None if use_native is False or weights is not None else _native()
    if library is not None:
        try:
            result = library.reduce_complex(rows, **_native_spec(policy, explicit))
        except library.PlaneUnsupportedError:
            result = None
        if result is not None:
            return _native_reduction(result, out_shape, kind="SPF_COMPLEX", terms_seen=int(rows.size), count=count)
    L = np.asarray(_core.unpack_L(rows), dtype=np.int64)
    T = np.asarray(_core.unpack_theta(rows), dtype=np.int64)
    has_nan = np.any((rows == np.uint32(SPF32_WORD_NAN)) & (True if weights is None else weights > 0), axis=1)
    keep = (L != SPF32_L_ZERO) & (rows != np.uint32(SPF32_WORD_NAN))
    if weights is not None:
        keep &= weights > 0
    kept = np.sum(keep, axis=1)
    pivot = (
        np.max(np.where(keep, L, np.int64(SPF32_L_ZERO)), axis=1, initial=SPF32_L_ZERO)
        if n
        else np.full(count, SPF32_L_ZERO, dtype=np.int64)
    )
    equivalent = kept if weights is None else np.sum(np.where(keep, weights, 0), axis=1)
    bits = np.asarray([v02._fraction_bits(policy, int(k), explicit) if k else 0 for k in equivalent], dtype=np.int64)
    bits = np.where(has_nan, 0, bits)
    scale = np.exp2(bits.astype(np.float64))
    mantissa = _mantissas(L, pivot, scale, keep)
    sine, cosine = _sincos_planes(T)
    x_terms = np.where(keep, _rint(mantissa.astype(np.float64) * cosine), 0)
    y_terms = np.where(keep, _rint(mantissa.astype(np.float64) * sine), 0)
    sum_bits = bits
    if weights is not None:
        headroom = math.ceil(math.log2(max(1, int(weights.max(initial=0)))))
        sum_bits = bits + headroom
        if int(sum_bits.max(initial=0)) + math.ceil(math.log2(max(1, n))) >= 62:
            x_terms, y_terms = x_terms.astype(object), y_terms.astype(object)
            mantissa = mantissa.astype(object)
        x_terms, y_terms = x_terms * weights, y_terms * weights
        mantissa = mantissa * weights
    X = _sum_rows(x_terms, sum_bits)
    Y = _sum_rows(y_terms, sum_bits)
    radius = np.asarray([math.hypot(float(x), float(y)) for x, y in zip(X, Y, strict=True)], dtype=np.float64)
    nonzero = (radius != 0.0) & (kept > 0)
    output_raw = np.zeros(count, dtype=np.int64)
    output_T = np.zeros(count, dtype=np.int64)
    for index in range(count):
        if not nonzero[index]:
            continue
        output_raw[index] = int(pivot[index]) + v02._rne(SPF32_L_SCALE * math.log2(radius[index] / float(scale[index])))
        output_T[index] = v02._rne(math.atan2(float(Y[index]), float(X[index])) * _RAD_TO_PHASE) & SPF32_PHASE_MASK
    output_L = np.where(nonzero, _clamp_finite(output_raw), SPF32_L_ZERO)
    out_words = np.asarray(_core.pack_words(output_L.astype(np.int16), output_T.astype(np.uint16)), dtype=np.uint32)
    out_words = np.where(has_nan, np.uint32(SPF32_WORD_NAN), out_words)
    saturated = nonzero & ((output_raw < SPF32_L_MIN_FINITE) | (output_raw > SPF32_L_MAX_FINITE))
    linear_scale = _exp2_code_planes(pivot)
    coordinate_error = linear_scale * equivalent * (1.0 + math.sqrt(2.0)) * 0.5 / scale
    phase_quantization = math.pi / SPF32_PHASE_MOD
    magnitude = linear_scale * radius / scale
    quantization = magnitude * (_LOG_QUANTIZATION_RELATIVE_ERROR + phase_quantization)
    absolute = coordinate_error + quantization
    lower = magnitude - coordinate_error
    absolute_sum = _sum_rows(np.abs(mantissa), sum_bits).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(lower <= 0.0, math.inf, absolute / lower)
        coordinate_phase = np.where(
            coordinate_error >= magnitude,
            math.pi,
            np.arcsin(np.minimum(1.0, coordinate_error / np.maximum(magnitude, 1e-300))),
        )
        phase_ratio = radius / np.maximum(absolute_sum, 1.0)
    phase_error = phase_quantization + coordinate_phase
    reliable = (phase_ratio >= 2.0**-12) & np.isfinite(phase_error)
    reliable &= phase_error <= v02.PHASE_RELIABILITY_MAX_ERROR_RAD
    absolute = np.where(saturated, math.inf, absolute)
    relative = np.where(saturated, math.inf, relative)
    absolute = np.where(nonzero, absolute, np.where(kept > 0, coordinate_error, 0.0))
    relative = np.where(nonzero, relative, np.where(kept > 0, math.inf, 0.0))
    phase_error = np.where(nonzero, phase_error, np.where(kept > 0, math.pi, 0.0))
    reliable = np.where(kept == 0, True, reliable & nonzero)
    absolute = np.where(has_nan, math.nan, absolute)
    relative = np.where(has_nan, math.nan, relative)
    phase_error = np.where(has_nan, math.nan, phase_error)
    reliable = reliable & ~has_nan
    live = np.where(has_nan, 0, kept)
    counters = PerformanceCounters(
        terms_seen=int(rows.size),
        terms_retained=int(kept.sum()),
        bytes_read=int(rows.size) * 4 + (0 if weights is None else int(weights.nbytes)),
        bytes_written=4 * count,
        aligned_mantissas=int(live.sum()),
        pivot_reductions=int(np.count_nonzero(live)),
        complex_finalizations=int(np.count_nonzero(live)),
        phase_ops=int(live.sum()),
        log_ops=int(np.count_nonzero(live)),
        atan2_ops=int(np.count_nonzero(live)),
        exp2_ops=int(live.sum()),
        trig_ops=int(live.sum()),
    )
    return PlaneReduction(
        out_words.reshape(out_shape),
        "SPF_COMPLEX",
        "pivot_complex" if np.any(kept) else "zero",
        absolute.reshape(out_shape),
        relative.reshape(out_shape),
        phase_error.reshape(out_shape),
        reliable.reshape(out_shape),
        kept.reshape(out_shape),
        np.zeros(out_shape, dtype=np.int64),
        bits.reshape(out_shape),
        saturated.reshape(out_shape),
        counters,
        _ledger_from_bounds(absolute),
    )


def _stack_scalar_results(
    results: list[v02.ReductionResult], out_shape: tuple[int, ...], *, kind: str
) -> PlaneReduction:
    """Assemble a :class:`PlaneReduction` from per-row scalar reducer results."""

    if kind == "LOG_POS":
        value: Any = LogPosPlane(np.asarray([r.value.L for r in results], dtype=np.int64).reshape(out_shape))
    elif kind == "LOG_SIGNED":
        value = LogSignedPlane(
            np.asarray([r.value.L for r in results], dtype=np.int64).reshape(out_shape),
            np.asarray([r.value.sign for r in results], dtype=np.int64).reshape(out_shape),
        )
    else:
        value = np.asarray([r.word for r in results], dtype=np.uint32).reshape(out_shape)
    counters = PerformanceCounters()
    for r in results:
        for name in counters.__dataclass_fields__:
            setattr(counters, name, getattr(counters, name) + getattr(r.counters, name))
    absolute = np.asarray([r.absolute_error_bound for r in results], dtype=np.float64).reshape(out_shape)
    return PlaneReduction(
        value,
        kind,
        results[0].method if results else "zero",
        absolute,
        np.asarray([r.relative_error_bound for r in results], dtype=np.float64).reshape(out_shape),
        np.asarray([r.phase_error_bound for r in results], dtype=np.float64).reshape(out_shape),
        np.asarray([r.phase_reliable for r in results], dtype=bool).reshape(out_shape),
        np.asarray([r.retained_terms for r in results], dtype=np.int64).reshape(out_shape),
        # §9.1 dropped_terms: the pruned row count the scalar reducer reports.
        np.asarray([r.dropped_terms for r in results], dtype=np.int64).reshape(out_shape),
        np.asarray([r.accumulator_fraction_bits for r in results], dtype=np.int64).reshape(out_shape),
        np.asarray([r.saturated for r in results], dtype=bool).reshape(out_shape),
        counters,
        _ledger_from_bounds(absolute),
    )


# ---------------------------------------------------------------------------
# FB3-hybrid butterfly and radix-2 FFT on word planes
# ---------------------------------------------------------------------------


def _word_from_local_planes(
    base_L: np.ndarray, base_T: np.ndarray, magnitude: np.ndarray, phase: np.ndarray
) -> np.ndarray:
    """``v02._word_from_local`` elementwise."""

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_L = _rint(SPF32_L_SCALE * np.log2(np.where(magnitude > 0.0, magnitude, 1.0)))
    L = _clamp_finite(base_L + delta_L)
    T = (base_T + _rint(phase * _RAD_TO_PHASE)) & SPF32_PHASE_MASK
    out = np.asarray(_core.pack_words(L.astype(np.int16), T.astype(np.uint16)), dtype=np.uint32)
    out = np.where(magnitude == 0.0, np.uint32(SPF32_WORD_ZERO), out)
    invalid = ~np.isfinite(magnitude) | (magnitude < 0.0)
    return np.where(invalid, np.uint32(SPF32_WORD_NAN), out)


def _rne_half_ticks_planes(total_half_ticks: np.ndarray) -> np.ndarray:
    """``v02._rne_half_ticks`` elementwise: ties-to-even on the total phase."""

    half_ticks = np.asarray(total_half_ticks, dtype=np.int64)
    floor_ticks = half_ticks >> 1  # arithmetic shift: floor for negatives too
    tie = (half_ticks & 1) != 0
    rounded = np.where(tie & ((floor_ticks & 1) != 0), floor_ticks + 1, floor_ticks)
    return np.asarray(rounded, dtype=np.int64)


def _word_from_local_exact_phase_planes(
    base_L: np.ndarray, phase_ticks: np.ndarray, magnitude: np.ndarray
) -> np.ndarray:
    """``v02._word_from_local_exact_phase`` elementwise."""

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_L = _rint(SPF32_L_SCALE * np.log2(np.where(magnitude > 0.0, magnitude, 1.0)))
    L = _clamp_finite(base_L + delta_L)
    T = np.asarray(phase_ticks, dtype=np.int64) & SPF32_PHASE_MASK
    out = np.asarray(_core.pack_words(L.astype(np.int16), T.astype(np.uint16)), dtype=np.uint32)
    out = np.where(magnitude == 0.0, np.uint32(SPF32_WORD_ZERO), out)
    invalid = ~np.isfinite(magnitude) | (magnitude < 0.0)
    return np.where(invalid, np.uint32(SPF32_WORD_NAN), out)


def _circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = (np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)) & SPF32_PHASE_MASK
    return np.where(d <= SPF32_PHASE_HALF - 1, d, d - SPF32_PHASE_MOD)


def _near_cancellation_planes(gap: np.ndarray, dT: np.ndarray, operation: int, precision: Any) -> np.ndarray:
    mode = v02._precision_name(precision)
    gap_limit = {"fast": 64, "balanced": 128, "accurate": 256, "exact": 512}.get(mode, 128)
    distance = np.abs(dT) if operation < 0 else np.abs(_circular_diff(dT, SPF32_PHASE_HALF))
    return (gap <= gap_limit) & (distance <= 2048)


_EXP2_NEG_GAP_ROM_ARRAY = np.asarray(_exp2_rom.EXP2_NEG_GAP_ROM, dtype=np.float64)


def _exp2_neg_gap_planes(gap: np.ndarray) -> np.ndarray:
    """Vectorized AlgRef §8.5 ``2^(-gap/256)``: ROM on the low 8 bits, shift on the rest."""

    gap64 = gap.astype(np.int64)
    return np.ldexp(_EXP2_NEG_GAP_ROM_ARRAY[gap64 & _exp2_rom.EXP2_NEG_GAP_ROM_MASK], -(gap64 >> 8))


def _exp2_code_planes(code: npt.ArrayLike) -> np.ndarray:
    """Vectorized AlgRef §8.5 ``2^(code/256)`` for a signed log code.

    Same committed 256-entry ROM as :func:`_exp2_neg_gap_planes`: the positive
    side needs no second table because ``2^(f/256) = 2 · 2^(-(256-f)/256)`` for
    ``f`` in ``[1, 255]``, one exact doubling, and :func:`numpy.ldexp` supplies
    the octave.  Every step is exact, so the result is the correctly-rounded
    ``2^(code/256)`` on every host.

    This replaces ``np.exp2(code / 256.0)`` in the reducers' certificate
    arithmetic.  NumPy's *vectorized* ``exp2`` loop is not correctly rounded on
    glibc: measured on beast (numpy 2.4.4, glibc 2.39) it differs from this ROM,
    from ``math.exp2`` and from mpmath-at-50-dps on 4608 / 65536 codes by 1 ulp,
    which is what made the §9.2 error bounds of the NumPy reducer disagree with
    the C reducer (``libspf32_planes``, scalar ``exp2``) on Linux while they
    agreed on macOS.  The ROM form is host-independent, so the two paths now
    agree by construction rather than by libm coincidence.
    """

    code64 = np.asarray(code, dtype=np.int64)
    fraction = code64 & _exp2_rom.EXP2_NEG_GAP_ROM_MASK
    base = np.where(
        fraction == 0,
        1.0,
        2.0 * _EXP2_NEG_GAP_ROM_ARRAY[(_exp2_rom.EXP2_NEG_GAP_ROM_SIZE - fraction) & _exp2_rom.EXP2_NEG_GAP_ROM_MASK],
    )
    return np.ldexp(base, code64 >> _exp2_rom.EXP2_NEG_GAP_SHIFT_BITS)


def _pair_local_planes(
    A_L: np.ndarray, A_T: np.ndarray, B_L: np.ndarray, B_T: np.ndarray, operation: int, precision: Any
) -> np.ndarray:
    """``v02._pair_local`` elementwise: ``A + operation*B`` with ``|A| >= |B|``."""

    gap = A_L - B_L
    dT = _circular_diff(B_T, A_T)
    angle = dT.astype(np.float64) * _PHASE_TO_RAD
    # §8.5: shift + 256-entry fractional exp2 ROM, bit-identical to
    # np.exp2(-gap/256) on every gap (verified 0/65536) and to v02.exp2_neg_gap.
    q = _exp2_neg_gap_planes(gap)
    ln2_gap = -np.expm1(-gap.astype(np.float64) * math.log(2.0) / SPF32_L_SCALE)

    def packed(L: np.ndarray) -> np.ndarray:
        return np.asarray(_core.pack_words(L.astype(np.int16), A_T.astype(np.uint16)), dtype=np.uint32)

    aligned = packed(_logadd_pos_L(A_L, B_L))
    opposed = np.where(gap == 0, np.uint32(SPF32_WORD_ZERO), packed(_logsub_pos_L(A_L, np.minimum(B_L, A_L))))
    # gap == 0 branch (dT not in the exact classes).  §7.3 with the §2.1 tie
    # rule: the exact output phase in ticks is ``Theta_A + dT/2`` (-+ a quarter
    # turn for the difference), which is an exact half-integer for odd ``dT``.
    # Resolve it on the total in integer half-ticks, never on the relative
    # delta.  Mirrors ``v02._rne_half_ticks``.
    half = angle / 2.0
    total_half_ticks = 2 * A_T + dT
    if operation > 0:
        eq_mag = 2.0 * np.abs(np.cos(half))
    else:
        eq_mag = 2.0 * np.abs(np.sin(half))
        total_half_ticks = total_half_ticks + np.where(dT > 0, -SPF32_PHASE_HALF, SPF32_PHASE_HALF)
    equal = _word_from_local_exact_phase_planes(A_L, _rne_half_ticks_planes(total_half_ticks), eq_mag)
    # |dT| == quarter branch
    y_q = operation * q * np.sin(angle)
    quarter = _word_from_local_planes(A_L, A_T, np.sqrt(1.0 + q * q), np.arctan2(y_q, np.ones_like(y_q)))
    # near-cancellation branch
    epsilon = _circular_diff(dT, SPF32_PHASE_HALF).astype(np.float64) * _PHASE_TO_RAD if operation > 0 else angle
    x_n = ln2_gap + q * (1.0 - np.cos(epsilon))
    y_n = -q * np.sin(epsilon)
    near = _word_from_local_planes(A_L, A_T, np.hypot(x_n, y_n), np.arctan2(y_n, x_n))
    # default branch
    x_d = 1.0 + operation * q * np.cos(angle)
    y_d = operation * q * np.sin(angle)
    default = _word_from_local_planes(A_L, A_T, np.hypot(x_d, y_d), np.arctan2(y_d, x_d))

    out = default
    out = np.where(_near_cancellation_planes(gap, dT, operation, precision), near, out)
    out = np.where(np.abs(dT) == SPF32_PHASE_QUARTER, quarter, out)
    out = np.where(gap == 0, equal, out)
    if operation > 0:
        out = np.where(dT == -SPF32_PHASE_HALF, opposed, out)
        out = np.where(dT == 0, aligned, out)
    else:
        out = np.where(dT == -SPF32_PHASE_HALF, aligned, out)
        out = np.where(dT == 0, opposed, out)
    return np.asarray(out, dtype=np.uint32)


def _pair_shared_direct_planes(
    A_L: np.ndarray, A_T: np.ndarray, B_L: np.ndarray, B_T: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    gap = A_L - B_L
    dT = _circular_diff(B_T, A_T)
    q = _exp2_neg_gap_planes(gap)  # §8.5 ROM: platform-independent on a word path
    sine, cosine = _sincos_planes(dT)
    q2 = q * q
    qcos = q * cosine
    qsin = q * sine
    plus_radius = np.sqrt(np.maximum(0.0, 1.0 + q2 + 2.0 * qcos))
    minus_radius = np.sqrt(np.maximum(0.0, 1.0 + q2 - 2.0 * qcos))
    plus = _word_from_local_planes(A_L, A_T, plus_radius, np.arctan2(qsin, 1.0 + qcos))
    minus = _word_from_local_planes(A_L, A_T, minus_radius, np.arctan2(-qsin, 1.0 - qcos))
    return plus, minus


def _pair_branch_masks(wu: np.ndarray, wt: np.ndarray, precision: Any) -> tuple[np.ndarray, np.ndarray]:
    """``(ordinary, counted)`` branch masks for one pair plane (integer only).

    ``ordinary`` selects the §7.5 shared-frame branch (the one that costs an
    ``exp2``, a ``sincos`` and two ``atan2``/``log2`` finalizers); ``counted``
    excludes the sentinel short circuits.  Used to fill
    :class:`PerformanceCounters` without re-evaluating the branches.
    """

    lu = np.asarray(_core.unpack_L(wu), dtype=np.int64)
    tu = np.asarray(_core.unpack_theta(wu), dtype=np.int64)
    lt = np.asarray(_core.unpack_L(wt), dtype=np.int64)
    tt = np.asarray(_core.unpack_theta(wt), dtype=np.int64)
    nan = (wu == np.uint32(SPF32_WORD_NAN)) | (wt == np.uint32(SPF32_WORD_NAN))
    u_zero = lu == SPF32_L_ZERO
    t_zero = lt == SPF32_L_ZERO
    u_is_A = lu >= lt
    A_L = np.where(u_is_A, lu, lt)
    A_T = np.where(u_is_A, tu, tt)
    B_L = np.where(u_is_A, lt, lu)
    B_T = np.where(u_is_A, tt, tu)
    gap = A_L - B_L
    dT = _circular_diff(B_T, A_T)
    exact_class = (dT == 0) | (dT == -SPF32_PHASE_HALF) | (np.abs(dT) == SPF32_PHASE_QUARTER)
    ordinary = (
        (gap > 0)
        & ~exact_class
        & ~_near_cancellation_planes(gap, dT, 1, precision)
        & ~_near_cancellation_planes(gap, dT, -1, precision)
    )
    return ordinary, ~(nan | u_zero | t_zero)


def _count_pair_ops(wu: np.ndarray, wt: np.ndarray, precision: Any, counters: PerformanceCounters) -> None:
    ordinary, counted = _pair_branch_masks(wu, wt, precision)
    counters.complex_pair_adds += int(wu.size)
    direct = int(np.count_nonzero(ordinary & counted))
    counters.exp2_ops += direct
    counters.trig_ops += direct
    counters.atan2_ops += 2 * direct
    counters.log_ops += 2 * direct
    counters.fb3_shared_products += 3 * direct


def spf_pair_addsub_hybrid(
    u: npt.ArrayLike,
    t: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
    use_native: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``v02.spf_pair_addsub_hybrid`` on word planes: ``(u+t, u-t)`` elementwise."""

    wu = v02._canonical_words(np.asarray(u, dtype=np.uint32))
    wt = v02._canonical_words(np.asarray(t, dtype=np.uint32))
    wu, wt = np.broadcast_arrays(wu, wt)
    library = None if use_native is False else _native()
    if library is not None and wu.size:
        plus, minus = library.pair_addsub(
            np.ascontiguousarray(wu), np.ascontiguousarray(wt), v02._precision_name(precision)
        )
        if counters is not None:
            _count_pair_ops(wu, wt, precision, counters)
        return plus, minus
    lu = np.asarray(_core.unpack_L(wu), dtype=np.int64)
    tu = np.asarray(_core.unpack_theta(wu), dtype=np.int64)
    lt = np.asarray(_core.unpack_L(wt), dtype=np.int64)
    tt = np.asarray(_core.unpack_theta(wt), dtype=np.int64)
    nan = (wu == np.uint32(SPF32_WORD_NAN)) | (wt == np.uint32(SPF32_WORD_NAN))
    u_zero = lu == SPF32_L_ZERO
    t_zero = lt == SPF32_L_ZERO
    u_is_A = lu >= lt
    A_L = np.where(u_is_A, lu, lt)
    A_T = np.where(u_is_A, tu, tt)
    B_L = np.where(u_is_A, lt, lu)
    B_T = np.where(u_is_A, tt, tu)
    gap = A_L - B_L
    dT = _circular_diff(B_T, A_T)
    exact_class = (dT == 0) | (dT == -SPF32_PHASE_HALF) | (np.abs(dT) == SPF32_PHASE_QUARTER)
    ordinary = (
        (gap > 0)
        & ~exact_class
        & ~_near_cancellation_planes(gap, dT, 1, precision)
        & ~_near_cancellation_planes(gap, dT, -1, precision)
    )
    plus_direct, minus_direct = _pair_shared_direct_planes(A_L, A_T, B_L, B_T)
    plus_local = _pair_local_planes(A_L, A_T, B_L, B_T, 1, precision)
    minus_local = _pair_local_planes(A_L, A_T, B_L, B_T, -1, precision)
    plus = np.where(ordinary, plus_direct, plus_local)
    minus_l = np.where(ordinary, minus_direct, minus_local)
    minus = np.where(u_is_A, minus_l, words_neg(minus_l))
    # sentinel classes
    plus = np.where(t_zero, wu, plus)
    minus = np.where(t_zero, wu, minus)
    plus = np.where(u_zero, wt, plus)
    minus = np.where(u_zero, words_neg(wt), minus)
    plus = np.where(nan, np.uint32(SPF32_WORD_NAN), plus)
    minus = np.where(nan, np.uint32(SPF32_WORD_NAN), minus)
    if counters is not None:
        counted = ~(nan | u_zero | t_zero)
        counters.complex_pair_adds += int(wu.size)
        direct = int(np.count_nonzero(ordinary & counted))
        counters.exp2_ops += direct
        counters.trig_ops += direct
        counters.atan2_ops += 2 * direct
        counters.log_ops += 2 * direct
        counters.fb3_shared_products += 3 * direct
    return np.asarray(plus, dtype=np.uint32), np.asarray(minus, dtype=np.uint32)


@dataclass(frozen=True, slots=True)
class PairAdderDiagnostics:
    """§7.5 reliability flags for a plane of pair additions (array fields).

    Same contract as :class:`v02.PairAdderDiagnostics`: the first three fields
    are the conservative combination over the two outputs.
    """

    phase_reliable: np.ndarray
    phase_error_bound: np.ndarray
    near_cancellation: np.ndarray
    plus_phase_reliable: np.ndarray
    minus_phase_reliable: np.ndarray
    plus_phase_error_bound: np.ndarray
    minus_phase_error_bound: np.ndarray
    stable_branch: np.ndarray


def _pair_output_flags_planes(
    radius: np.ndarray, aligned: np.ndarray, source_error: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``v02._pair_output_flags`` elementwise."""

    degenerate = (radius <= 0.0) | (source_error >= radius)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(degenerate, 1.0, source_error / np.where(radius > 0.0, radius, 1.0))
    phase_error = np.where(degenerate, math.pi, _PAIR_PHASE_QUANTIZATION + np.arcsin(np.minimum(1.0, ratio)))
    cancelled = degenerate | (radius < 2.0**-12 * aligned)
    reliable = ~cancelled & np.isfinite(phase_error) & (phase_error <= v02.PHASE_RELIABILITY_MAX_ERROR_RAD)
    return reliable, phase_error, cancelled


def spf_pair_addsub_hybrid_diagnostics(
    u: npt.ArrayLike,
    t: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
    use_native: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, PairAdderDiagnostics]:
    """``v02.spf_pair_addsub_hybrid_diagnostics`` on word planes (§7.5)."""

    wu = v02._canonical_words(np.asarray(u, dtype=np.uint32))
    wt = v02._canonical_words(np.asarray(t, dtype=np.uint32))
    wu, wt = np.broadcast_arrays(wu, wt)
    library = None if use_native is False else _native()
    if library is not None and wu.size:
        plus, minus, flags = library.pair_addsub_diagnostics(
            np.ascontiguousarray(wu), np.ascontiguousarray(wt), v02._precision_name(precision)
        )
        if counters is not None:
            _count_pair_ops(wu, wt, precision, counters)
        plus_reliable = flags["plus_reliable"].astype(bool)
        minus_reliable = flags["minus_reliable"].astype(bool)
        cancelled = flags["plus_cancelled"].astype(bool) | flags["minus_cancelled"].astype(bool)
        return (
            plus,
            minus,
            PairAdderDiagnostics(
                plus_reliable & minus_reliable,
                np.maximum(flags["plus_error"], flags["minus_error"]),
                cancelled,
                plus_reliable,
                minus_reliable,
                flags["plus_error"],
                flags["minus_error"],
                flags["stable_branch"].astype(bool),
            ),
        )
    plus, minus = spf_pair_addsub_hybrid(u, t, precision, counters=counters, use_native=use_native)
    lu = np.asarray(_core.unpack_L(wu), dtype=np.int64)
    tu = np.asarray(_core.unpack_theta(wu), dtype=np.int64)
    lt = np.asarray(_core.unpack_L(wt), dtype=np.int64)
    tt = np.asarray(_core.unpack_theta(wt), dtype=np.int64)
    nan = (wu == np.uint32(SPF32_WORD_NAN)) | (wt == np.uint32(SPF32_WORD_NAN))
    sentinel_zero = (lu == SPF32_L_ZERO) | (lt == SPF32_L_ZERO)
    u_is_A = lu >= lt
    A_L = np.where(u_is_A, lu, lt)
    A_T = np.where(u_is_A, tu, tt)
    B_L = np.where(u_is_A, lt, lu)
    B_T = np.where(u_is_A, tt, tu)
    gap = A_L - B_L
    dT = _circular_diff(B_T, A_T)
    q = _exp2_neg_gap_planes(gap)  # §8.5 ROM: platform-independent on a word path
    sine, cosine = _sincos_planes(dT)
    radius_plus = np.hypot(1.0 + q * cosine, q * sine)
    radius_minus = np.hypot(1.0 - q * cosine, q * sine)
    aligned = 1.0 + q
    source_error = aligned * (_LOG_QUANTIZATION_RELATIVE_ERROR + _PAIR_PHASE_QUANTIZATION)
    plus_reliable, plus_error, plus_cancelled = _pair_output_flags_planes(radius_plus, aligned, source_error)
    minus_reliable, minus_error, minus_cancelled = _pair_output_flags_planes(radius_minus, aligned, source_error)
    stable = _near_cancellation_planes(gap, dT, 1, precision) | _near_cancellation_planes(gap, dT, -1, precision)
    tick = np.full(wu.shape, _PAIR_PHASE_QUANTIZATION, dtype=np.float64)
    false_flag = np.zeros(wu.shape, dtype=bool)
    # sentinel classes: a zero operand copies the other word through, a NaN
    # operand certifies nothing.
    plus_reliable = np.where(nan, False, np.where(sentinel_zero, True, plus_reliable))
    minus_reliable = np.where(nan, False, np.where(sentinel_zero, True, minus_reliable))
    plus_error = np.where(nan, math.nan, np.where(sentinel_zero, tick, plus_error))
    minus_error = np.where(nan, math.nan, np.where(sentinel_zero, tick, minus_error))
    plus_cancelled = np.where(nan | sentinel_zero, False, plus_cancelled)
    minus_cancelled = np.where(nan | sentinel_zero, False, minus_cancelled)
    stable = np.where(nan | sentinel_zero, false_flag, stable)
    return (
        plus,
        minus,
        PairAdderDiagnostics(
            plus_reliable & minus_reliable,
            np.maximum(plus_error, minus_error),
            plus_cancelled | minus_cancelled,
            plus_reliable,
            minus_reliable,
            plus_error,
            minus_error,
            stable,
        ),
    )


def _stage_indices(n: int, block: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = block // 2
    index = np.arange(n, dtype=np.int64)
    first = index[(index % block) < half]
    second = first + half
    offsets = first % block
    step = -(v02.SPF32_INTERNAL_PHASE_MOD // block)
    wide = offsets * step
    twiddle = np.asarray([v02.phase_code_from_wide(int(w)) for w in wide], dtype=np.int64)
    return first, second, twiddle


def spf_fft_radix2(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Vectorized v0.2 FB3-hybrid radix-2 FFT over the last axis of a word plane.

    Same schedule as the scalar :func:`spf.tensor.v02._fft_row_v02` (bit
    reversal, then stages of size 2, 4, ...); all rows and all butterflies of
    one stage run as one plane operation.
    """

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_fft_radix2 requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_fft_radix2 requires power-of-two N, got N={n}")
    if n > v02.SPF32_INTERNAL_PHASE_MOD:
        raise ValueError("spf_fft_radix2 requires N <= the internal phase accumulator modulus")
    rows = array.reshape(-1, n)
    working = rows[:, v02._bit_reverse_indices(n)].copy()
    block = 2
    while block <= n:
        first, second, twiddle = _stage_indices(n, block)
        a = working[:, first]
        b = words_phase_shift(working[:, second], twiddle[None, :])
        if counters is not None:
            counters.fft_butterflies += int(a.size)
            counters.phase_ops += int(a.size)
        y0, y1 = spf_pair_addsub_hybrid(a, b, precision, counters=counters)
        working[:, first] = y0
        working[:, second] = y1
        block *= 2
    return working.reshape(array.shape)


def spf_ifft_radix2(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Inverse of :func:`spf_fft_radix2`: conjugate, forward, conjugate, exact ``1/N`` log shift."""

    array = np.asarray(values, dtype=np.uint32)
    if array.ndim == 0:
        raise ValueError("spf_ifft_radix2 requires a one-dimensional or batched array")
    n = int(array.shape[-1])
    if n < 1 or (n & (n - 1)) != 0:
        raise ValueError(f"spf_ifft_radix2 requires power-of-two N, got N={n}")
    transformed = words_conj(spf_fft_radix2(words_conj(array), precision, counters=counters))
    log_shift = int(math.log2(n)) * SPF32_L_SCALE
    L = np.asarray(_core.unpack_L(transformed), dtype=np.int64)
    T = np.asarray(_core.unpack_theta(transformed), dtype=np.int64)
    keep = (L == SPF32_L_ZERO) | (transformed == np.uint32(SPF32_WORD_NAN))
    scaled = np.asarray(_core.pack_words(_clamp_finite(L - log_shift).astype(np.int16), T.astype(np.uint16)))
    return np.asarray(np.where(keep, transformed, scaled), dtype=np.uint32)


def _apply_along_axes(values: Any, transform: Any, precision: Any, axes: Any, counters: Any) -> np.ndarray:
    working = np.asarray(values, dtype=np.uint32)
    # ``axes=None`` runs the last axis first, which is the order the C packed
    # path (``spf32_fb3_ifft2``) and ``renderer.continuity.packed_ifft2`` use.
    # Word arithmetic is not associative across axes, so running 0, 1 instead
    # differed from the C words on 3265 of 4096 outputs at 64x64.
    selected = tuple(reversed(range(working.ndim))) if axes is None else tuple(int(a) % working.ndim for a in axes)
    for axis in selected:
        moved = np.ascontiguousarray(np.moveaxis(working, axis, -1))
        moved = np.asarray(transform(moved, precision, counters=counters), dtype=np.uint32)
        working = np.moveaxis(moved, -1, axis)
    return np.ascontiguousarray(working)


def spf_fftn_radix2(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    axes: Sequence[int] | None = None,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Separable N-d v0.2 FFT on a packed word plane: one :func:`spf_fft_radix2` pass per axis.

    Every transformed axis must be a power of two.  ``axes=None`` transforms
    all of them **last axis first**, matching the C packed transform
    (``spf32_fb3_ifft2``) and :func:`spf.renderer.continuity.packed_ifft2`;
    pass ``axes`` explicitly for any other order.  The per-axis passes are the
    plane radix-2 schedule, so the result is the same word sequence a
    row-by-row application of the scalar oracle would produce.
    """

    return _apply_along_axes(values, spf_fft_radix2, precision, axes, counters)


def spf_ifftn_radix2(
    values: npt.ArrayLike,
    precision: Any = "balanced",
    *,
    axes: Sequence[int] | None = None,
    counters: PerformanceCounters | None = None,
) -> np.ndarray:
    """Separable N-d inverse of :func:`spf_fftn_radix2` (exact ``1/N`` log shift per axis).

    ``axes=None`` runs the last axis first, so a 2-D call is word-identical to
    :func:`spf.renderer.continuity.packed_ifft2` and to the C packed path.
    """

    return _apply_along_axes(values, spf_ifft_radix2, precision, axes, counters)


__all__ = [
    "LogPosPlane",
    "LogSignedPlane",
    "PairAdderDiagnostics",
    "PhasePlane",
    "PlaneReduction",
    "native_available",
    "native_enabled",
    "reduce_complex",
    "reduce_logpos",
    "reduce_logsigned",
    "set_native",
    "spf_fft_radix2",
    "spf_fftn_radix2",
    "spf_ifft_radix2",
    "spf_ifftn_radix2",
    "spf_pair_addsub_hybrid",
    "spf_pair_addsub_hybrid_diagnostics",
    "words_abs2",
    "words_conj",
    "words_div",
    "words_mul",
    "words_neg",
    "words_phase_shift",
    "words_planes",
]
