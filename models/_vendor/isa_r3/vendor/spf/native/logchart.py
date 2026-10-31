"""Optional ctypes bindings for the SPF-32 logchart kernels.

The logchart API is deliberately kept out of :mod:`spf.native._libspf32`.
That loader registers the stable SPF-32 ABI and must continue to load an older
``libspf32`` while a new native build is being staged.  This module resolves
and registers the logchart symbols lazily, so importing it is safe with either
an older library or no native library at all.  A call that needs a missing
symbol raises :class:`NotImplementedError` with the loaded library path.

The public functions operate on packed SPF-32 ``uint32`` words.  They mirror
the native API while doing the validation that is otherwise easy to lose at a
ctypes boundary: chart and base mode names, power-of-two FFT sizes, inverse
flags, one-dimensional input for :func:`fft`, and two-dimensional input for
the batched :func:`fft_rows` entry point.
"""

from __future__ import annotations

import ctypes
import operator
from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from spf.native._libspf32 import LIBSPF_PATH, load_library

__all__ = [
    "CHART_R1",
    "CHART_R15",
    "LOGCHART_AVAILABLE",
    "MAX_N",
    "MIN_N",
    "MODE_ACCURATE",
    "MODE_BALANCED",
    "MODE_FAST",
    "R1",
    "R15",
    "add",
    "butterfly",
    "chart_from_name",
    "fft",
    "fft_rows",
    "fft_rows_profiled",
    "ifft",
    "ifft_rows",
    "ifft_rows_profiled",
    "is_available",
    "mode_from_name",
    "region",
]

Word = npt.NDArray[np.uint32]

MIN_N = 2
MAX_N = 1 << 16

# These values are part of the native logchart ABI.  Keep aliases because the
# short names are convenient in Python and the CHART_* spelling is clearer in
# code that serializes a configuration.
R1 = 0
R15 = 1
CHART_R1 = R1
CHART_R15 = R15

MODE_FAST = 0
MODE_BALANCED = 1
MODE_ACCURATE = 2

_CHART_NAMES: Mapping[str, int] = {
    "r1": R1,
    "r15": R15,
}
_MODE_NAMES: Mapping[str, int] = {
    "fast": MODE_FAST,
    "balanced": MODE_BALANCED,
    "accurate": MODE_ACCURATE,
}

_SYMBOLS: Mapping[str, str] = {
    "add": "spf32_add_logchart",
    "butterfly": "spf32_butterfly_logchart",
    "region": "spf32_logchart_region",
    "fft": "spf32_fft_logchart",
    "fft_rows": "spf32_fft_logchart_rows",
    "fft_rows_profiled": "spf32_fft_logchart_rows_profiled",
}

# Keep the CDLL handle alive alongside the bound callables.  Caching solely by
# ``id(lib)`` would allow a later handle to reuse an old id after garbage
# collection and inherit a stale registration.
_BOUND_LIBS: dict[int, tuple[Any, dict[str, Any]]] = {}


class _FFTStats(ctypes.Structure):
    _fields_ = [
        ("rows", ctypes.c_uint64),
        ("bit_reversal_values", ctypes.c_uint64),
        ("stages", ctypes.c_uint64),
        ("butterflies", ctypes.c_uint64),
        ("coherent_hits", ctypes.c_uint64),
        ("opposed_hits", ctypes.c_uint64),
        ("exact_special_hits", ctypes.c_uint64),
        ("fb7_fallback_hits", ctypes.c_uint64),
        ("inverse_normalized_values", ctypes.c_uint64),
        ("inverse_normalization_saturations", ctypes.c_uint64),
    ]

    def as_dict(self) -> dict[str, int]:
        return {field[0]: int(getattr(self, field[0])) for field in self._fields_}


def mode_from_name(mode: str | int) -> int:
    """Resolve one of the three base native modes.

    Logchart does not accept the FB7 composite mode byte.  Integers are
    therefore restricted to exactly ``0``, ``1``, and ``2``.
    """

    if isinstance(mode, str):
        try:
            return _MODE_NAMES[mode.strip().lower()]
        except KeyError:
            expected = ", ".join(sorted(_MODE_NAMES))
            raise ValueError(f"unknown logchart mode {mode!r}; expected one of {expected}") from None
    if isinstance(mode, (bool, np.bool_)):
        raise ValueError("logchart mode must be one of 0, 1, or 2; bool is not a mode")
    try:
        value = operator.index(mode)
    except TypeError:
        raise TypeError(f"logchart mode must be a name or integer 0..2, got {type(mode).__name__}") from None
    if value not in (MODE_FAST, MODE_BALANCED, MODE_ACCURATE):
        raise ValueError(f"logchart mode {value!r} is unsupported; expected base mode 0, 1, or 2")
    return int(value)


def chart_from_name(chart: str | int) -> int:
    """Resolve the native R1/R15 chart enum from a name or integer."""

    if isinstance(chart, str):
        try:
            return _CHART_NAMES[chart.strip().lower()]
        except KeyError:
            raise ValueError(f"unknown logchart chart {chart!r}; expected 'r1' or 'r15'") from None
    if isinstance(chart, (bool, np.bool_)):
        raise ValueError("logchart chart must be R1/R15 (0/1); bool is not a chart")
    try:
        value = operator.index(chart)
    except TypeError:
        raise TypeError(f"logchart chart must be a name or integer 0..1, got {type(chart).__name__}") from None
    if value not in (R1, R15):
        raise ValueError(f"logchart chart {value!r} is unsupported; expected R1=0 or R15=1")
    return int(value)


def _library_label(lib: Any) -> str:
    path = LIBSPF_PATH
    if path is not None:
        return str(path)
    return repr(lib)


def _symbol(lib: Any, operation: str) -> Any:
    """Resolve an operation symbol and turn a missing export into a clear error."""

    name = _SYMBOLS[operation]
    try:
        return getattr(lib, name), name
    except AttributeError:
        pass
    raise NotImplementedError(
        f"native logchart operation {operation!r} is unavailable: loaded library "
        f"{_library_label(lib)} does not export {name}; rebuild libspf32 with "
        "the logchart API"
    )


def _bind_signatures(lib: Any) -> dict[str, Any]:
    """Bind logchart signatures on *lib* and return operation callables.

    This is intentionally separate from ``_libspf32._register``.  ctypes
    defaults are unsafe for pointer arguments, so registration happens before
    every first use of a logchart operation, but an older stable library is
    never rejected during normal SPF import.
    """

    key = id(lib)
    cached = _BOUND_LIBS.get(key)
    if cached is not None and cached[0] is lib:
        return cached[1]

    add, _ = _symbol(lib, "add")
    add.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int, ctypes.c_int]
    add.restype = ctypes.c_uint32

    butterfly, _ = _symbol(lib, "butterfly")
    butterfly.argtypes = [
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
    ]
    butterfly.restype = None

    region, _ = _symbol(lib, "region")
    region.argtypes = [ctypes.c_uint32, ctypes.c_uint32, ctypes.c_int]
    region.restype = ctypes.c_int

    fft, _ = _symbol(lib, "fft")
    fft.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fft.restype = ctypes.c_int

    fft_rows, _ = _symbol(lib, "fft_rows")
    fft_rows.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
    ]
    fft_rows.restype = ctypes.c_int

    fft_rows_profiled, _ = _symbol(lib, "fft_rows_profiled")
    fft_rows_profiled.argtypes = [
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(_FFTStats),
    ]
    fft_rows_profiled.restype = ctypes.c_int

    bound = {
        "add": add,
        "butterfly": butterfly,
        "region": region,
        "fft": fft,
        "fft_rows": fft_rows,
        "fft_rows_profiled": fft_rows_profiled,
    }
    _BOUND_LIBS[key] = (lib, bound)
    return bound


def is_available() -> bool:
    """Return whether all logchart exports are present in the loaded library."""

    try:
        lib = load_library()
    except RuntimeError:
        return False
    return all(hasattr(lib, candidate) for candidate in _SYMBOLS.values())


def _u32_scalar(value: int | np.integer, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an SPF-32 uint32 word, not bool")
    try:
        result = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer packed uint32 word") from None
    if result < 0 or result > 0xFFFFFFFF:
        raise ValueError(f"{name}={result} is outside packed uint32 range 0..0xFFFFFFFF")
    return int(result)


def _words(value: npt.ArrayLike, *, name: str) -> Word:
    """Convert packed words without silently wrapping signed/out-of-range input."""

    raw = np.asarray(value)
    if raw.dtype.kind not in "biu":
        raise TypeError(f"{name} must contain packed uint32 words; got dtype {raw.dtype}")
    if raw.size:
        if np.any(raw < 0):
            raise ValueError(f"{name} contains negative values, not packed uint32 words")
        if np.any(raw > np.iinfo(np.uint32).max):
            raise ValueError(f"{name} contains values above packed uint32 range")
    return np.ascontiguousarray(raw, dtype=np.uint32)


def _validate_n(n: int) -> int:
    if isinstance(n, (bool, np.bool_)):
        raise TypeError("FFT length n must be an integer power of two")
    try:
        value = operator.index(n)
    except TypeError:
        raise TypeError("FFT length n must be an integer power of two") from None
    if value < MIN_N or value > MAX_N or (value & (value - 1)) != 0:
        raise ValueError(f"FFT length n must be a power of two in [{MIN_N}, {MAX_N}]; got {value}")
    return int(value)


def _validate_inverse(inverse: int | bool) -> int:
    if isinstance(inverse, (bool, np.bool_)):
        return int(inverse)
    try:
        value = operator.index(inverse)
    except TypeError:
        raise TypeError("inverse must be 0 or 1") from None
    if value not in (0, 1):
        raise ValueError(f"inverse must be 0 or 1; got {value}")
    return int(value)


def _word_ptr(arr: np.ndarray, *, allow_null: bool = False) -> Any:
    if arr.size == 0 and allow_null:
        return ctypes.POINTER(ctypes.c_uint32)()
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def add(
    a: int | np.integer,
    b: int | np.integer,
    mode: str | int = "accurate",
    chart: str | int = "r1",
) -> int:
    """Add two packed SPF-32 words using a selected logchart and base mode."""

    mode_i = mode_from_name(mode)
    chart_i = chart_from_name(chart)
    lib = load_library()
    fn = _bind_signatures(lib)["add"]
    return int(fn(_u32_scalar(a, name="a"), _u32_scalar(b, name="b"), mode_i, chart_i))


def butterfly(
    u: int | np.integer,
    v: int | np.integer,
    w: int | np.integer,
    mode: str | int = "accurate",
    chart: str | int = "r1",
) -> tuple[int, int]:
    """Return ``(u + w·v, u − w·v)`` as packed SPF-32 words."""

    mode_i = mode_from_name(mode)
    chart_i = chart_from_name(chart)
    lib = load_library()
    fn = _bind_signatures(lib)["butterfly"]
    top = ctypes.c_uint32()
    bot = ctypes.c_uint32()
    fn(
        _u32_scalar(u, name="u"),
        _u32_scalar(v, name="v"),
        _u32_scalar(w, name="w"),
        ctypes.byref(top),
        ctypes.byref(bot),
        mode_i,
        chart_i,
    )
    return int(top.value), int(bot.value)


def region(
    a: int | np.integer,
    b: int | np.integer,
    chart: str | int = "r1",
) -> int:
    """Return the native region classification for two packed words."""

    chart_i = chart_from_name(chart)
    lib = load_library()
    fn = _bind_signatures(lib)["region"]
    return int(fn(_u32_scalar(a, name="a"), _u32_scalar(b, name="b"), chart_i))


def _run_fft(
    arr: Word,
    *,
    inverse: int,
    mode: int,
    chart: int,
) -> Word:
    n = _validate_n(arr.shape[-1])
    out = np.empty_like(arr)
    lib = load_library()
    fn = _bind_signatures(lib)["fft"]
    rc = int(
        fn(
            _word_ptr(arr),
            _word_ptr(out),
            ctypes.c_int(n),
            ctypes.c_int(inverse),
            ctypes.c_int(mode),
            ctypes.c_int(chart),
        )
    )
    if rc != 0:
        raise RuntimeError(f"spf32_fft_logchart returned {rc} for n={n}")
    return out


def fft(
    x: npt.ArrayLike,
    mode: str | int = "accurate",
    chart: str | int = "r1",
) -> Word:
    """Forward native logchart FFT of one packed SPF-32 row."""

    arr = _words(x, name="x")
    if arr.ndim != 1:
        raise ValueError(f"logchart fft expects a 1-D packed uint32 row; got shape {arr.shape}")
    return _run_fft(
        arr,
        inverse=0,
        mode=mode_from_name(mode),
        chart=chart_from_name(chart),
    )


def ifft(
    x: npt.ArrayLike,
    mode: str | int = "accurate",
    chart: str | int = "r1",
) -> Word:
    """Inverse native logchart FFT of one packed SPF-32 row."""

    arr = _words(x, name="x")
    if arr.ndim != 1:
        raise ValueError(f"logchart ifft expects a 1-D packed uint32 row; got shape {arr.shape}")
    return _run_fft(
        arr,
        inverse=1,
        mode=mode_from_name(mode),
        chart=chart_from_name(chart),
    )


def _run_fft_rows(
    arr: Word,
    *,
    inverse: int,
    mode: int,
    chart: int,
) -> Word:
    if arr.ndim != 2:
        raise ValueError(f"logchart fft_rows expects shape (rows, n) of packed uint32 words; got {arr.shape}")
    n = _validate_n(arr.shape[1])
    rows = int(arr.shape[0])
    out = np.empty_like(arr)
    lib = load_library()
    fn = _bind_signatures(lib)["fft_rows"]
    rc = int(
        fn(
            _word_ptr(arr, allow_null=True),
            _word_ptr(out, allow_null=True),
            ctypes.c_int(n),
            ctypes.c_size_t(rows),
            ctypes.c_int(inverse),
            ctypes.c_int(mode),
            ctypes.c_int(chart),
        )
    )
    if rc != 0:
        raise RuntimeError(f"spf32_fft_logchart_rows returned {rc} for n={n}, rows={rows}")
    return out


def _run_fft_rows_profiled(
    arr: Word,
    *,
    inverse: int,
    mode: int,
    chart: int,
) -> tuple[Word, dict[str, int]]:
    if arr.ndim != 2:
        raise ValueError(f"logchart fft_rows expects a 2-D packed uint32 batch; got shape {arr.shape}")
    n = _validate_n(arr.shape[-1])
    out = np.empty_like(arr)
    stats = _FFTStats()
    lib = load_library()
    fn = _bind_signatures(lib)["fft_rows_profiled"]
    rc = int(
        fn(
            _word_ptr(arr, allow_null=True),
            _word_ptr(out, allow_null=True),
            ctypes.c_int(n),
            ctypes.c_size_t(arr.shape[0]),
            ctypes.c_int(inverse),
            ctypes.c_int(mode),
            ctypes.c_int(chart),
            ctypes.byref(stats),
        )
    )
    if rc != 0:
        raise RuntimeError(f"spf32_fft_logchart_rows_profiled returned {rc} for n={n}, rows={arr.shape[0]}")
    return out, stats.as_dict()


def fft_rows(
    x: npt.ArrayLike,
    mode: str | int = "accurate",
    chart: str | int = "r1",
    *,
    inverse: int | bool = False,
) -> Word:
    """Native logchart FFT over every row of a packed word matrix.

    ``inverse`` is keyword-only so the common forward call remains compact;
    :func:`ifft_rows` is provided as the explicit inverse spelling.
    """

    arr = _words(x, name="x")
    return _run_fft_rows(
        arr,
        inverse=_validate_inverse(inverse),
        mode=mode_from_name(mode),
        chart=chart_from_name(chart),
    )


def ifft_rows(
    x: npt.ArrayLike,
    mode: str | int = "accurate",
    chart: str | int = "r1",
) -> Word:
    """Inverse native logchart FFT over every row of a packed word matrix."""

    return fft_rows(x, mode=mode, chart=chart, inverse=True)


def fft_rows_profiled(
    x: npt.ArrayLike,
    mode: str | int = "accurate",
    chart: str | int = "r1",
    *,
    inverse: int | bool = False,
) -> tuple[Word, dict[str, int]]:
    """Run a row batch and return native mechanism counters."""

    arr = _words(x, name="x")
    return _run_fft_rows_profiled(
        arr,
        inverse=_validate_inverse(inverse),
        mode=mode_from_name(mode),
        chart=chart_from_name(chart),
    )


def ifft_rows_profiled(
    x: npt.ArrayLike,
    mode: str | int = "accurate",
    chart: str | int = "r1",
) -> tuple[Word, dict[str, int]]:
    """Profile an inverse native logchart FFT over every packed row."""

    return fft_rows_profiled(x, mode=mode, chart=chart, inverse=True)


# Report availability as a convenience without making it authoritative: a
# process that builds or replaces SPF_LIBRARY after import should use
# ``is_available()`` for a fresh probe.
LOGCHART_AVAILABLE = is_available()
