"""SPFArray — NumPy-compatible wrapper for packed SPF words.

Provides an ergonomic array object around raw ``uint32`` SPF-32 words so
callers can write::

    x = spf.asarray(z, dtype=spf.spf32_fmt)
    Y = spf.numpy.fft.fft(x) / H
    out = Y.to_numpy()

instead of manually managing packed words through the scalar API.

Backend dispatch
----------------
The ``backend=`` kwarg selects an execution backend:

- ``"reference"``  — pure-Python/NumPy SPF-32 (always available).
- ``"native"``     — C-accelerated path via ``libspf32`` ctypes bindings.
                     Requires ``make -C native``; raises ``RuntimeError`` if
                     the library is not built.
- ``"kv260_pl"``   — FPGA fabric dispatch via SSH+pyxrt to KV260.
                     Requires ``SPF_KRIA_HOST``; raises ``RuntimeError`` if
                     unset when a PL op is invoked. PL ops: mul, div, add,
                     sub (two calls: neg+add), abs2, conj, neg, fft N=64.
                     Host ops: encode, decode, ifft (no xclbin built).
                     Host path is ``native`` if ``libspf32`` is present,
                     else ``reference``. ``SPFArray.backend_info`` shows
                     the exact dispatch location per op at runtime.

The backend key and ``backend_info`` provenance map are preserved across all
arithmetic and FFT operations.
"""

from __future__ import annotations

import types as _types
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from spf.backends.kria.pl_client import PersistentPLSession

import numpy as np
import numpy.typing as npt

__all__ = [
    "SPFArray",
    "SPFBackend",
    "asarray",
    "register_backend",
]


def _normalize_profile(profile: object | None) -> str:
    """Normalize an array execution profile without importing the v0.2 module."""

    if profile is None:
        return "legacy"
    raw = getattr(profile, "value", profile)
    name = str(raw).strip().lower().replace("-", "_")
    aliases = {
        "legacy": "legacy",
        "v01": "legacy",
        "frozen_compat": "legacy",
        "v02": "v02",
        "spf_v02": "v02",
        "strict_v02": "v02",
        "fb3_hybrid": "v02",
    }
    try:
        return aliases[name]
    except KeyError:
        raise ValueError(
            f"Unsupported SPFArray profile {profile!r}; expected 'legacy' or 'v02'"
        ) from None


def _normalize_value_kind(value_kind: object | None) -> str:
    """Normalize the typed value kind carried by an SPFArray.

    Packed SPF-32 words currently have two supported declarations: ordinary
    complex SPF values and positive log values.  Other v0.2 kinds use a
    different carrier and must not be silently treated as complex words.
    """

    if value_kind is None:
        return "SPF_COMPLEX"
    raw = getattr(value_kind, "value", value_kind)
    name = str(raw).strip().upper().replace("-", "_")
    aliases = {
        "COMPLEX": "SPF_COMPLEX",
        "COMPLEX128": "SPF_COMPLEX",
        "SPF_COMPLEX": "SPF_COMPLEX",
        "LOGPOS": "LOG_POS",
        "LOG_POS": "LOG_POS",
        "POSITIVE": "LOG_POS",
        "POSITIVE_REAL": "LOG_POS",
    }
    try:
        return aliases[name]
    except KeyError:
        raise ValueError(
            f"Unsupported SPFArray value_kind {value_kind!r}; expected "
            "'SPF_COMPLEX' or 'LOG_POS'"
        ) from None


def _coerce_log_pos_values(value: object) -> npt.NDArray[np.complex128]:
    """Validate and normalize real non-negative values for a LOG_POS array."""

    from spf.tensor import v02

    array = np.asarray(value, dtype=object)
    result = np.empty(array.shape, dtype=np.complex128)
    for index in np.ndindex(array.shape):
        item = array[index]
        if isinstance(item, v02.LogPos):
            number = complex(item.value)
        else:
            try:
                number = complex(item)
            except (TypeError, ValueError) as exc:
                raise TypeError("LOG_POS inputs must be real non-negative values") from exc
        if np.isnan(number.real) or np.isnan(number.imag):
            result[index] = complex(np.nan, np.nan)
            continue
        if number.imag != 0.0 or number.real < 0.0:
            raise ValueError("LOG_POS inputs must be real and non-negative")
        result[index] = number
    return result


def _validate_log_pos_words(words: npt.ArrayLike) -> None:
    """Reject packed words whose finite LOG_POS phase is not zero."""

    from spf.tensor import v02

    array = np.asarray(words, dtype=np.uint32)
    phases = np.asarray(v02.unpack_T(array), dtype=np.uint16)
    logs = np.asarray(v02.unpack_L(array), dtype=np.int64)
    allowed_nan = (logs == v02.SPF32_NAN_L) & (phases == v02.SPF32_NAN_T)
    invalid = ((phases != v02.SPF32_T_ZERO) & ~allowed_nan) | (
        (logs == v02.SPF32_NAN_L) & ~allowed_nan
    )
    if np.any(invalid):
        raise ValueError("LOG_POS SPF words must have zero phase (except canonical NaN)")

# ── Backend protocol ─────────────────────────────────────────────────────────


@runtime_checkable
class SPFBackend(Protocol):
    """Execution backend for SPFArray operations.

    All methods accept and return ``ndarray[uint32]`` SPF words except
    ``encode`` (complex → words) and ``decode`` (words → complex128).
    Broadcasting follows NumPy semantics.
    """

    def encode(self, z: npt.ArrayLike) -> npt.NDArray[np.uint32]: ...
    def decode(self, w: npt.NDArray[np.uint32]) -> npt.NDArray[np.complex128]: ...
    def mul(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]: ...
    def div(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]: ...
    def add(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]: ...
    def sub(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]: ...
    def conj(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]: ...
    def abs2(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]: ...
    def neg(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]: ...
    def fft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]: ...
    def ifft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]: ...


class _ReferenceBackend:
    """Pure-Python/NumPy SPF-32 reference backend."""

    _availability: ClassVar[str] = "available"

    def encode(self, z: npt.ArrayLike) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.encode(z)

    def decode(self, w: npt.NDArray[np.uint32]) -> npt.NDArray[np.complex128]:
        from spf.scalar import spf32
        return spf32.decode(w)

    def mul(self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.mul(a, b)

    def div(self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.div(a, b)

    def add(self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.add(a, b)

    def sub(self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.sub(a, b)

    def conj(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.conj(a)

    def abs2(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.abs2(a)

    def neg(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.scalar import spf32
        return spf32.neg(a)

    def fft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.fft import fft_spf32
        return fft_spf32(x)

    def ifft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.fft import ifft_spf32
        return ifft_spf32(x)


class _NativeBackend:
    """C-accelerated SPF-32 backend via ``libspf32`` ctypes bindings.

    All ops dispatch to the vectorised C entry points in ``spf.native.spf32``
    and ``spf.native.fft``.  Raises ``RuntimeError`` at call time if ``libspf32``
    is not built — import is always safe.
    """

    @staticmethod
    def _check() -> None:
        from spf.native._lib import LIBSPF_AVAILABLE
        if not LIBSPF_AVAILABLE:
            raise RuntimeError(
                "backend='native' requires libspf32. "
                "Build with `make -C native`. "
                "Use backend='reference' for the pure-Python path."
            )

    def encode(self, z: npt.ArrayLike) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.encode(z)

    def decode(self, w: npt.NDArray[np.uint32]) -> npt.NDArray[np.complex128]:
        self._check()
        from spf.native import spf32 as _c
        return _c.decode(w)

    def mul(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.mul(a, b)

    def div(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.div(a, b)

    def add(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.add(a, b)

    def sub(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.sub(a, b)

    def conj(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.conj(a)

    def abs2(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.abs2(a)

    def neg(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        self._check()
        from spf.native import spf32 as _c
        return _c.neg(a)

    def fft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        # Batched path loops in Python (one ctypes call per row).
        # No vectorised C batch kernel exists yet — benchmark batch timings
        # reflect O(n_rows) ctypes overhead, not a single C dispatch.
        self._check()
        from spf.native import fft as _c_fft
        arr = np.asarray(x, dtype=np.uint32)
        if arr.ndim == 1:
            return _c_fft.fft(arr)
        shape = arr.shape
        rows = arr.reshape(-1, shape[-1])
        return np.stack([_c_fft.fft(rows[i]) for i in range(len(rows))]).reshape(shape)

    def ifft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        # Batched path loops in Python — same caveat as fft above.
        self._check()
        from spf.native import fft as _c_fft
        arr = np.asarray(x, dtype=np.uint32)
        if arr.ndim == 1:
            return _c_fft.ifft(arr)
        shape = arr.shape
        rows = arr.reshape(-1, shape[-1])
        return np.stack([_c_fft.ifft(rows[i]) for i in range(len(rows))]).reshape(shape)


class _KV260PLBackend:
    """KV260 PL-fabric backend.

    PL fabric dispatch for all ops with synthesized xclbins on the KV260.

    On PL  (xclbin exists): mul, div, add, sub (via neg+add), abs2, conj, neg,
                             fft N=64.
    On host (encode/decode/ifft): these require floating-point or inverse FFT
                                  which have no xclbin in the current build.

    All PL ops require ``SPF_KRIA_HOST`` (SSH path) or a connected
    ``PersistentPLSession`` (daemon path); raises ``RuntimeError`` if neither
    is available.  ``SPFArray.backend_info`` maps each op to its execution
    location at runtime.
    """

    def __init__(self, session: PersistentPLSession | None = None) -> None:
        self._session = session

    @property
    def backend_info(self) -> _types.MappingProxyType[str, str]:
        """Per-op execution location, resolved at call time.

        Host ops (encode/decode/ifft) show ``"host_native"`` when
        ``libspf32`` is built, ``"host_reference"`` otherwise.
        """
        from spf.native._lib import LIBSPF_AVAILABLE
        host = "host_native" if LIBSPF_AVAILABLE else "host_reference"
        return _types.MappingProxyType({
            "fft_N64": "kv260_pl",
            "mul": "kv260_pl",
            "div": "kv260_pl",
            "add": "kv260_pl",
            "sub": "kv260_pl",   # two dispatch calls: neg(b) then add(a, neg_b)
            "abs2": "kv260_pl",
            "conj": "kv260_pl",
            "neg": "kv260_pl",
            "encode": host,
            "decode": host,
            "ifft": host,
        })

    @staticmethod
    def _host() -> SPFBackend:
        # Re-read LIBSPF_AVAILABLE on every call (not cached) so tests can
        # monkeypatch it without invalidating a cached reference.
        from spf.native._lib import LIBSPF_AVAILABLE
        return _BACKEND_REGISTRY["native"] if LIBSPF_AVAILABLE else _BACKEND_REGISTRY["reference"]

    @staticmethod
    def _kria_host() -> str:
        import os
        host = os.environ.get("SPF_KRIA_HOST", "")
        if not host:
            raise RuntimeError(
                "backend='kv260_pl' requires SPF_KRIA_HOST "
                "(e.g. export SPF_KRIA_HOST=kv260). "
                "Use backend='native' or backend='reference' for host-only paths."
            )
        return host

    def encode(self, z: npt.ArrayLike) -> npt.NDArray[np.uint32]:
        return self._host().encode(z)

    def decode(self, w: npt.NDArray[np.uint32]) -> npt.NDArray[np.complex128]:
        return self._host().decode(w)

    def mul(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        from spf.backends.kria.pl_dispatch import mul_batch
        if self._session is not None:
            return np.asarray(mul_batch(a, b, session=self._session), dtype=np.uint32)
        return np.asarray(mul_batch(a, b, host=self._kria_host()), dtype=np.uint32)

    def div(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        from spf.backends.kria.pl_dispatch import div_batch
        if self._session is not None:
            return np.asarray(div_batch(a, b, session=self._session), dtype=np.uint32)
        return np.asarray(div_batch(a, b, host=self._kria_host()), dtype=np.uint32)

    def add(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        from spf.backends.kria.pl_dispatch import add_batch
        if self._session is not None:
            return np.asarray(add_batch(a, b, session=self._session), dtype=np.uint32)
        return np.asarray(add_batch(a, b, host=self._kria_host()), dtype=np.uint32)

    def sub(
        self, a: npt.NDArray[np.uint32], b: npt.NDArray[np.uint32]
    ) -> npt.NDArray[np.uint32]:
        # No sub xclbin — implement as add(a, neg(b)) via two PL dispatch calls.
        from spf.backends.kria.pl_dispatch import add_batch, neg_batch
        if self._session is not None:
            neg_b = np.asarray(neg_batch(b, session=self._session), dtype=np.uint32)
            return np.asarray(add_batch(a, neg_b, session=self._session), dtype=np.uint32)
        h = self._kria_host()
        neg_b = np.asarray(neg_batch(b, host=h), dtype=np.uint32)
        return np.asarray(add_batch(a, neg_b, host=h), dtype=np.uint32)

    def conj(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.backends.kria.pl_dispatch import conj_batch
        if self._session is not None:
            return np.asarray(conj_batch(a, session=self._session), dtype=np.uint32)
        return np.asarray(conj_batch(a, host=self._kria_host()), dtype=np.uint32)

    def abs2(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.backends.kria.pl_dispatch import abs2_batch
        if self._session is not None:
            return np.asarray(abs2_batch(a, session=self._session), dtype=np.uint32)
        return np.asarray(abs2_batch(a, host=self._kria_host()), dtype=np.uint32)

    def neg(self, a: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        from spf.backends.kria.pl_dispatch import neg_batch
        if self._session is not None:
            return np.asarray(neg_batch(a, session=self._session), dtype=np.uint32)
        return np.asarray(neg_batch(a, host=self._kria_host()), dtype=np.uint32)

    def fft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        # Only 1-D (N,) and 2-D (batch, N) supported — the PL fabric is a
        # single 64-point FFT kernel; higher-rank arrays need native backend.
        arr = np.ascontiguousarray(x, dtype=np.uint32)
        if arr.ndim not in (1, 2):
            raise ValueError(
                f"kv260_pl fft: expected 1-D or 2-D input, got ndim={arr.ndim}. "
                "Use backend='native' for higher-rank arrays."
            )
        # Resolve dispatch destination *before* validating shape so a missing
        # SPF_KRIA_HOST surfaces as the canonical "configure the backend"
        # error, not a confusing kernel-shape rejection.  With a session, the
        # host is irrelevant and we skip straight to the shape check.
        host: str | None = None
        if self._session is None:
            host = self._kria_host()
        n = arr.shape[-1]
        if n != 64:
            raise NotImplementedError(
                f"kv260_pl fft64 kernel is fixed N=64; got N={n}. "
                "Use backend='native' or backend='reference' for other FFT sizes."
            )
        if self._session is not None:
            from spf.backends.kria.pl_dispatch import fft64_via_session
            return np.asarray(fft64_via_session(self._session, arr), dtype=np.uint32)
        from spf.backends.kria.pl_dispatch import fft64_batch
        return np.asarray(fft64_batch(arr, host=host), dtype=np.uint32)

    def ifft(self, x: npt.NDArray[np.uint32]) -> npt.NDArray[np.uint32]:
        return self._host().ifft(x)


# ── Backend registry ─────────────────────────────────────────────────────────

_BACKEND_REGISTRY: dict[str, SPFBackend] = {
    "reference": _ReferenceBackend(),
    "native": _NativeBackend(),
    "kv260_pl": _KV260PLBackend(),
}


def register_backend(name: str, backend: SPFBackend) -> None:
    """Register a custom backend under ``name``.

    The backend must satisfy the :class:`SPFBackend` protocol.  After
    registration, ``SPFArray(..., backend=name)`` uses it.
    """
    if not isinstance(backend, SPFBackend):
        raise TypeError(f"backend must implement SPFBackend protocol, got {type(backend)}")
    _BACKEND_REGISTRY[name] = backend


def _get_backend(name: str) -> SPFBackend:
    try:
        return _BACKEND_REGISTRY[name]
    except KeyError:
        known = ", ".join(repr(k) for k in _BACKEND_REGISTRY)
        raise ValueError(f"Unknown backend {name!r}. Known: {known}") from None


# ── __array_function__ dispatch table ────────────────────────────────────────

_HANDLED_FUNCTIONS: dict[Any, Any] = {}

_F = TypeVar("_F", bound=Callable[..., Any])


def _implements(np_function: Callable[..., Any]) -> Callable[[_F], _F]:
    def decorator(func: _F) -> _F:
        _HANDLED_FUNCTIONS[np_function] = func
        return func
    return decorator


# ── SPFArray ─────────────────────────────────────────────────────────────────


class SPFArray:
    """NumPy-compatible array of packed SPF-32 words.

    Wraps a ``uint32`` ndarray of SPF-32 encoded samples and exposes
    arithmetic operators, shape properties, and decoded views so DSP
    code reads like normal complex NumPy without manual word packing.

    Parameters
    ----------
    words : array_like of uint32
        Raw packed SPF-32 words.
    fmt : str
        Format identifier.  Currently ``"spf32"``; reserved for future
        ``"spf16"`` / ``"spf64"`` support.
    backend : str
        Execution backend key; see module docstring.
    profile : str, optional
        Algorithm profile.  ``"legacy"`` is the compatibility default;
        ``"v02"`` selects the v0.2 reference for supported operations.
    value_kind : str or v0.2 ValueKind, optional
        Typed carrier declaration for the packed words.
    """

    def __init__(
        self,
        words: npt.ArrayLike,
        fmt: str = "spf32",
        backend: str = "reference",
        _backend_obj: SPFBackend | None = None,
        *,
        profile: str | None = None,
        value_kind: object | None = None,
    ) -> None:
        self._words: npt.NDArray[np.uint32] = np.ascontiguousarray(
            words, dtype=np.uint32
        )
        self._fmt = fmt
        self._backend_key = backend
        self._profile = _normalize_profile(profile)
        self._value_kind = _normalize_value_kind(value_kind)
        if self._profile == "v02" and self._value_kind == "LOG_POS":
            _validate_log_pos_words(self._words)
        # _backend_obj lets asarray() inject a session-aware backend without
        # registering a new registry key.  _wrap() propagates it so that all
        # derived arrays (results of arithmetic, FFT, etc.) carry the same
        # session-aware backend instance.
        self._backend: SPFBackend = (
            _backend_obj if _backend_obj is not None else _get_backend(backend)
        )

    # ── read-only properties ─────────────────────────────────────────────

    @property
    def words(self) -> npt.NDArray[np.uint32]:
        """Raw packed SPF words (uint32 ndarray, read-only view)."""
        view = self._words.view()
        view.flags.writeable = False
        return view

    @property
    def shape(self) -> tuple[int, ...]:
        return self._words.shape

    @property
    def ndim(self) -> int:
        return self._words.ndim

    @property
    def size(self) -> int:
        return self._words.size

    @property
    def format(self) -> str:
        return self._fmt

    @property
    def backend(self) -> str:
        return self._backend_key

    @property
    def profile(self) -> str:
        """Algorithm profile carried by this array (``legacy`` or ``v02``)."""

        return self._profile

    @property
    def value_kind(self) -> str:
        """Typed v0.2 value kind carried by the packed words."""

        return self._value_kind

    @property
    def backend_info(self) -> _types.MappingProxyType[str, str] | None:
        """Per-op execution location for backends with mixed dispatch.

        Returns a read-only mapping for ``kv260_pl`` showing ``"kv260_pl"``
        or ``"host_native"`` / ``"host_reference"`` per op, or ``None`` for
        backends where all ops run in one place.
        """
        return getattr(self._backend, "backend_info", None)

    # ── decode ───────────────────────────────────────────────────────────

    def to_numpy(self) -> npt.NDArray[np.complex128]:
        """Decode to complex128 ndarray (same shape as ``self.words``)."""
        if self._profile == "v02":
            v02 = self._require_v02_reference()
            return np.asarray(v02.decode_complex(self._words), dtype=np.complex128)
        return self._backend.decode(self._words)

    def decode(self) -> npt.NDArray[np.complex128]:
        """Alias for :meth:`to_numpy`."""
        return self.to_numpy()

    # ── internal helpers ─────────────────────────────────────────────────

    def _wrap(
        self,
        words: npt.ArrayLike,
        *,
        profile: str | None = None,
        value_kind: object | None = None,
    ) -> SPFArray:
        return SPFArray(
            words,
            self._fmt,
            self._backend_key,
            _backend_obj=self._backend,
            profile=self._profile if profile is None else profile,
            value_kind=self._value_kind if value_kind is None else value_kind,
        )

    def _require_v02_reference(self) -> Any:
        """Return the v0.2 reference module for an explicit v0.2 operation."""

        if self._fmt != "spf32":
            raise NotImplementedError(
                "SPFArray profile='v02' currently supports only format='spf32'"
            )
        if self._backend_key != "reference":
            raise NotImplementedError(
                "SPFArray profile='v02' is reference-only; the selected backend "
                f"{self._backend_key!r} has no v0.2 implementation"
            )
        from spf.tensor import v02

        return v02

    def _coerce_words(self, other: object) -> npt.NDArray[np.uint32]:
        if isinstance(other, SPFArray):
            if other._backend_key != self._backend_key:
                raise ValueError(
                    f"mixed backend: {self._backend_key!r} and {other._backend_key!r}. "
                    "Use asarray(x, backend=...) to align backends before operating."
                )
            if other._profile != self._profile:
                raise ValueError(
                    f"mixed profile: {self._profile!r} and {other._profile!r}. "
                    "Use asarray(x, profile=...) to align profiles before operating."
                )
            if other._value_kind != self._value_kind:
                raise ValueError(
                    f"mixed value kind: {self._value_kind!r} and {other._value_kind!r}. "
                    "Use asarray(x, value_kind=...) to align typed operands."
                )
            return other._words
        if self._profile == "v02":
            v02 = self._require_v02_reference()
            if self._value_kind == "LOG_POS":
                other_values = _coerce_log_pos_values(other)
                return np.asarray(v02.encode_complex(other_values), dtype=np.uint32)
            return np.asarray(
                v02.encode_complex(np.asarray(other, dtype=np.complex128)),
                dtype=np.uint32,
            )
        return self._backend.encode(np.asarray(other, dtype=np.complex128))

    @staticmethod
    def _v02_logpos_from_word(v02: Any, word: int) -> Any:
        value = v02.unpack_spf32(int(word))
        if value.is_nan:
            return v02.LogPos.nan()
        if value.is_zero:
            return v02.LogPos.zero()
        if value.T != v02.SPF32_T_ZERO:
            raise ValueError("LOG_POS SPF words must have zero phase")
        return v02.LogPos(value.L)

    def _v02_binary(
        self,
        other: object,
        operation: str,
        *,
        reverse: bool = False,
    ) -> npt.NDArray[np.uint32]:
        v02 = self._require_v02_reference()
        other_words = self._coerce_words(other)
        left, right = (
            (other_words, self._words) if reverse else (self._words, other_words)
        )
        left_b, right_b = np.broadcast_arrays(
            np.asarray(left, dtype=np.uint32), np.asarray(right, dtype=np.uint32)
        )
        result = np.empty(left_b.shape, dtype=np.uint32)

        for index in np.ndindex(left_b.shape):
            a_word = int(left_b[index])
            b_word = int(right_b[index])
            if self._value_kind == "LOG_POS":
                a_value = self._v02_logpos_from_word(v02, a_word)
                b_value = self._v02_logpos_from_word(v02, b_word)
                if operation == "mul":
                    value = v02.logpos_mul(a_value, b_value)
                elif operation == "div":
                    value = v02.logpos_div(a_value, b_value)
                elif operation == "add":
                    value = v02.logadd_pos(a_value, b_value)
                elif operation == "sub":
                    value = v02.logsub_pos(a_value, b_value)
                else:
                    raise NotImplementedError(
                        f"v0.2 operation {operation!r} is not defined for LOG_POS"
                    )
                result[index] = np.uint32(v02.pack_spf32(value))
                continue

            if self._value_kind != "SPF_COMPLEX":
                raise NotImplementedError(
                    f"v0.2 arithmetic does not support value_kind={self._value_kind!r}"
                )
            if operation == "mul":
                value = v02.spf_mul(a_word, b_word, profile=v02.NaNProfile.STRICT_V02)
            elif operation == "div":
                value = v02.spf_div(a_word, b_word, profile=v02.NaNProfile.STRICT_V02)
            elif operation == "add":
                value = v02.spf_add(a_word, b_word)
            elif operation == "sub":
                value = v02.spf_add_complex_reference(
                    a_word, v02.spf_neg(b_word, profile=v02.NaNProfile.STRICT_V02)
                )
            else:
                raise NotImplementedError(f"unknown v0.2 arithmetic operation {operation!r}")
            result[index] = np.uint32(value)
        return result

    def _transform(
        self,
        *,
        inverse: bool,
        profile: str | None = None,
        precision: object = "balanced",
    ) -> SPFArray:
        selected = self._profile if profile is None else _normalize_profile(profile)
        if selected == "legacy":
            if precision != "balanced":
                raise ValueError("precision is only supported with profile='v02'")
            words = self._backend.ifft(self._words) if inverse else self._backend.fft(self._words)
            return self._wrap(words, profile="legacy")
        if self._value_kind != "SPF_COMPLEX":
            raise TypeError(
                "FFT/IFFT requires value_kind='SPF_COMPLEX', "
                f"got {self._value_kind!r}"
            )
        v02 = self._require_v02_reference()
        transform = v02.spf_ifft_radix2 if inverse else v02.spf_fft_radix2
        words = transform(self._words, precision=precision)
        return self._wrap(words, profile="v02")

    # ── arithmetic operators ─────────────────────────────────────────────

    def __mul__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "mul"))
        return self._wrap(self._backend.mul(self._words, self._coerce_words(other)))

    def __rmul__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "mul", reverse=True))
        return self._wrap(self._backend.mul(self._coerce_words(other), self._words))

    def __truediv__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "div"))
        return self._wrap(self._backend.div(self._words, self._coerce_words(other)))

    def __rtruediv__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "div", reverse=True))
        return self._wrap(self._backend.div(self._coerce_words(other), self._words))

    def __add__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "add"))
        return self._wrap(self._backend.add(self._words, self._coerce_words(other)))

    def __radd__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "add", reverse=True))
        return self._wrap(self._backend.add(self._coerce_words(other), self._words))

    def __sub__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "sub"))
        return self._wrap(self._backend.sub(self._words, self._coerce_words(other)))

    def __rsub__(self, other: object) -> SPFArray:
        if self._profile == "v02":
            return self._wrap(self._v02_binary(other, "sub", reverse=True))
        return self._wrap(self._backend.sub(self._coerce_words(other), self._words))

    def __neg__(self) -> SPFArray:
        if self._profile == "v02":
            if self._value_kind == "LOG_POS":
                raise TypeError("LOG_POS values cannot be negated")
            v02 = self._require_v02_reference()
            return self._wrap(
                v02.spf_neg(self._words, profile=v02.NaNProfile.STRICT_V02)
            )
        return self._wrap(self._backend.neg(self._words))

    def conj(self) -> SPFArray:
        """Complex conjugate (L unchanged, Θ ↦ −Θ mod 2¹⁶)."""
        if self._profile == "v02":
            if self._value_kind == "LOG_POS":
                return self._wrap(self._words.copy())
            v02 = self._require_v02_reference()
            return self._wrap(
                v02.spf_conj(self._words, profile=v02.NaNProfile.STRICT_V02)
            )
        return self._wrap(self._backend.conj(self._words))

    def abs2(self) -> SPFArray:
        """|z|² as a positive-real SPF word (L ↦ 2L, Θ = 0)."""
        if self._profile == "v02":
            v02 = self._require_v02_reference()
            if self._value_kind == "LOG_POS":
                return self._wrap(
                    self._v02_binary(self, "mul"), value_kind="LOG_POS"
                )
            words = np.empty(self._words.shape, dtype=np.uint32)
            for index in np.ndindex(self._words.shape):
                words[index] = np.uint32(
                    v02.pack_spf32(v02.spf_abs2(int(self._words[index])))
                )
            return self._wrap(words, value_kind="LOG_POS")
        return self._wrap(self._backend.abs2(self._words))

    # ── NumPy ufunc protocol ─────────────────────────────────────────────

    _UFUNC_BINARY: ClassVar[dict[Any, tuple[str, str]]] = {
        np.multiply: ("__mul__", "__rmul__"),
        np.divide: ("__truediv__", "__rtruediv__"),
        np.true_divide: ("__truediv__", "__rtruediv__"),
        np.add: ("__add__", "__radd__"),
        np.subtract: ("__sub__", "__rsub__"),
    }
    _UFUNC_UNARY: ClassVar[dict[Any, str]] = {
        np.negative: "__neg__",
        np.conjugate: "conj",
    }

    def __array_ufunc__(
        self,
        ufunc: np.ufunc,
        method: str,
        *inputs: object,
        **kwargs: object,
    ) -> object:
        if method != "__call__":
            return NotImplemented
        if kwargs:
            return NotImplemented
        if ufunc in self._UFUNC_UNARY:
            op = self._UFUNC_UNARY[ufunc]
            return getattr(self, op)()
        if ufunc in self._UFUNC_BINARY:
            fwd, rev = self._UFUNC_BINARY[ufunc]
            if len(inputs) != 2:
                return NotImplemented
            a, b = inputs
            if isinstance(a, SPFArray):
                return getattr(a, fwd)(b)
            return getattr(b, rev)(a)
        return NotImplemented

    # ── NumPy array_function protocol ───────────────────────────────────

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> object:
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
        if not all(issubclass(t, SPFArray) for t in types):
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    # ── reshape / shape ops ──────────────────────────────────────────────

    def reshape(self, *shape: Any) -> SPFArray:
        newshape = shape[0] if len(shape) == 1 and not isinstance(shape[0], int) else shape
        return self._wrap(self._words.reshape(newshape))

    def transpose(self, axes: Any = None) -> SPFArray:
        return self._wrap(self._words.transpose(axes))

    def flatten(self) -> SPFArray:
        return self._wrap(self._words.flatten())

    def ravel(self) -> SPFArray:
        return self._wrap(self._words.ravel())

    # ── dunder utils ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._words)

    def __repr__(self) -> str:
        return (
            f"SPFArray(shape={self.shape}, format={self._fmt!r}, "
            f"backend={self._backend_key!r}, profile={self._profile!r}, "
            f"value_kind={self._value_kind!r})"
        )

    # __hash__ intentionally absent: defining __eq__ makes instances unhashable
    # by Python default, which is correct for mutable array semantics.
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SPFArray):
            return NotImplemented
        return (
            np.array_equal(self._words, other._words)
            and self._fmt == other._fmt
            and self._backend_key == other._backend_key
            and self._profile == other._profile
            and self._value_kind == other._value_kind
        )

    def __getitem__(self, key: Any) -> SPFArray:
        return self._wrap(self._words[key])


# ── __array_function__ implementations ───────────────────────────────────────


@_implements(np.copy)
def _spf_copy(
    a: SPFArray,
    order: Literal["K", "A", "C", "F"] = "K",
    subok: bool = False,
) -> SPFArray:
    return SPFArray(a._words.copy(order=order), a._fmt, a._backend_key,
                   _backend_obj=a._backend, profile=a._profile,
                   value_kind=a._value_kind)


@_implements(np.reshape)
def _spf_reshape(
    a: SPFArray,
    newshape: int | Sequence[int],
    order: Literal["A", "C", "F"] = "C",
) -> SPFArray:
    return SPFArray(a._words.reshape(newshape, order=order), a._fmt, a._backend_key,
                   _backend_obj=a._backend, profile=a._profile,
                   value_kind=a._value_kind)


@_implements(np.transpose)
def _spf_transpose(a: SPFArray, axes: Any = None) -> SPFArray:
    return SPFArray(a._words.transpose(axes), a._fmt, a._backend_key,
                   _backend_obj=a._backend, profile=a._profile,
                   value_kind=a._value_kind)


# ── top-level encode helper ───────────────────────────────────────────────────


def asarray(
    z: npt.ArrayLike | SPFArray,
    dtype: str | None = None,
    backend: str = "reference",
    session: PersistentPLSession | None = None,
    *,
    profile: str | None = None,
    value_kind: object | None = None,
) -> SPFArray:
    """Encode *z* as an :class:`SPFArray`.

    If *z* is already an ``SPFArray``, it is returned unchanged (no copy)
    unless *dtype*, *backend*, *profile*, or *value_kind* differ from the
    existing array.

    Parameters
    ----------
    z : array_like
        Complex samples or an existing ``SPFArray``.
    dtype : str, optional
        Format identifier string; defaults to ``"spf32"``.  Use
        ``spf.spf32_fmt`` or the literal ``"spf32"``.
    backend : str
        Execution backend; default ``"reference"``.
    profile : str, optional
        Algorithm profile carried by the array.  ``"legacy"`` is the
        compatibility default; ``"v02"`` selects the pure-Python v0.2
        reference for supported operations.
    value_kind : str or v0.2 ValueKind, optional
        Typed carrier declaration.  Packed SPF-32 arrays currently support
        ``"SPF_COMPLEX"`` and ``"LOG_POS"``.
    session : PersistentPLSession, optional
        If given, all PL ops on the resulting array are dispatched through
        this persistent board session instead of spawning a new SSH process
        per call.  Requires ``backend='kv260_pl'``.  Use
        :func:`kria_session` to create one.
    """
    if dtype is not None and not isinstance(dtype, str):
        raise TypeError(
            f"dtype must be a string like 'spf32' or None, got {type(dtype).__name__!r}. "
            "Use spf.spf32_fmt instead of spf.spf32."
        )
    fmt = dtype if dtype is not None else "spf32"
    selected_profile = (
        z.profile if isinstance(z, SPFArray) and profile is None else _normalize_profile(profile)
    )
    selected_value_kind = (
        z.value_kind if isinstance(z, SPFArray) and value_kind is None else _normalize_value_kind(value_kind)
    )

    if selected_profile == "v02" and backend != "reference" and not isinstance(z, SPFArray):
        raise NotImplementedError(
            "SPFArray profile='v02' is reference-only; the selected backend "
            f"{backend!r} has no v0.2 encoder"
        )

    if session is not None:
        from spf.backends.kria.pl_client import PersistentPLSession as _PSess
        if not isinstance(session, _PSess):
            raise TypeError(
                f"session must be a PersistentPLSession, got {type(session).__name__!r}"
            )
        if backend != "kv260_pl":
            raise ValueError(
                f"session= is only valid with backend='kv260_pl', got {backend!r}"
            )
        backend_obj: SPFBackend = _KV260PLBackend(session=session)
        words = z._words if isinstance(z, SPFArray) else backend_obj.encode(np.asarray(z, dtype=np.complex128))
        return SPFArray(
            words,
            fmt,
            "kv260_pl",
            _backend_obj=backend_obj,
            profile=selected_profile,
            value_kind=selected_value_kind,
        )

    if isinstance(z, SPFArray):
        if (
            z.format == fmt
            and z.backend == backend
            and z.profile == selected_profile
            and z.value_kind == selected_value_kind
        ):
            return z
        return SPFArray(
            z._words,
            fmt,
            backend,
            profile=selected_profile,
            value_kind=selected_value_kind,
        )
    if selected_profile == "v02":
        if backend != "reference":
            raise NotImplementedError(
                "SPFArray profile='v02' is reference-only; the selected backend "
            f"{backend!r} has no v0.2 encoder"
            )
        from spf.tensor import v02

        values = (
            _coerce_log_pos_values(z)
            if selected_value_kind == "LOG_POS"
            else np.asarray(z, dtype=np.complex128)
        )
        words = np.asarray(v02.encode_complex(values), dtype=np.uint32)
    else:
        b = _get_backend(backend)
        words = b.encode(np.asarray(z, dtype=np.complex128))
    return SPFArray(
        words,
        fmt,
        backend,
        profile=selected_profile,
        value_kind=selected_value_kind,
    )


def kria_session(
    host: str | None = None,
    port: int | None = None,
    timeout: float = 120.0,
) -> PersistentPLSession:
    """Create a :class:`~spf.backends.kria.pl_client.PersistentPLSession`.

    Use as a context manager to dispatch all PL ops through a persistent
    board daemon, eliminating the ~200–400 ms SSH + Python startup cost::

        with spf.kria_session() as sess:
            x = spf.asarray(z, backend="kv260_pl", session=sess)
            y = x * x          # dispatched over TCP, no SSH fork
            out = y.to_numpy()

    Parameters
    ----------
    host : str, optional
        TCP endpoint for the daemon.  Defaults to
        ``$SPF_KRIA_SESSION_HOST`` if set, else ``"127.0.0.1"`` to match
        the recommended SSH-tunnel setup (``pl_server.py`` binds to
        loopback by default).  ``$SPF_KRIA_HOST`` is the *SSH* target and
        is **not** used here — keeping the two split lets the same
        ``SPF_KRIA_HOST=user@board`` env power both per-call SSH dispatch
        and the local-tunnel session endpoint.
    port : int, optional
        TCP port on the board.  Defaults to ``$SPF_KRIA_SESSION_PORT`` if
        set, else 9999.
    timeout : float
        Socket send/recv timeout in seconds (default 120).

    Notes
    -----
    Recommended setup uses an SSH tunnel because ``pl_server.py`` binds
    to ``127.0.0.1`` on the board by default::

        # one-shot on the board (or via systemd):
        ssh kv260 'sudo systemctl start spf-pl-server'

        # tunnel from the host (background, exits on ssh logout):
        ssh -fNT -L 9999:127.0.0.1:9999 kv260
        # then call kria_session() with no args — defaults are 127.0.0.1:9999

    Direct LAN binds require launching the daemon with
    ``--allow-unsafe-bind --bind 0.0.0.0`` and are intended only for
    trusted-network development.  In that case set
    ``SPF_KRIA_SESSION_HOST=<board-ip>`` (or pass ``host=`` explicitly).
    """
    import os

    from spf.backends.kria.pl_client import PersistentPLSession
    h = host or os.environ.get("SPF_KRIA_SESSION_HOST", "127.0.0.1")
    p = port if port is not None else int(os.environ.get("SPF_KRIA_SESSION_PORT", "9999"))
    return PersistentPLSession(h, p, timeout)
