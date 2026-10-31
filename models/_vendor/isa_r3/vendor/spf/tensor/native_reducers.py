"""Native fast path for the SPFTensor domain-routed reducer.

Provides :func:`native_reduce_axis` and :func:`maybe_native_reduce_axis` as a
drop-in fast path for :func:`spf.tensor.reducers.reduce_axis` when
``libspftensor`` is available.

Design constraints (DO NOT violate):
- This module does NOT modify ``reducers.py`` or ``core.py``.
- This module is standalone: callers import it directly.
- When ``LIBSPFTENSOR_AVAILABLE`` is False the pure-Python fallback in
  ``reducers.reduce_axis`` is used transparently.

Usage::

    from spf.tensor.native_reducers import maybe_native_reduce_axis
    from spf.tensor.reducers import reduce_axis

    # Transparent fallback:
    result = maybe_native_reduce_axis(terms, spec, axis=1)
    # Explicit native (raises if not available):
    result = native_reduce_axis(terms, spec, axis=1)
"""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from spf.native._libspftensor import (
    LIBSPFTENSOR_AVAILABLE,
    CReducerSpec,
)
from spf.native.spftensor import (
    SPFT_DOMAIN_NONE,
    SPFT_FMT_SPF32,
    SPFT_MODE_ACCURATE,
    SPFT_MODE_BALANCED,
    SPFT_MODE_EXACT,
    SPFT_MODE_FAST,
    SPFT_POLICY_AUTO,
    SPFT_ROUTE_PINNED,
    SPFT_ROUTE_STATIC,
    make_reducer_spec,
)

if TYPE_CHECKING:
    from spf.tensor.core import ReducerSpec

__all__ = [
    "maybe_native_reduce_axis",
    "native_reduce_axis",
    "py_spec_to_c_spec",
]


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

_MODE_MAP = {
    0: SPFT_MODE_FAST,       # PrecisionMode.FAST
    1: SPFT_MODE_BALANCED,   # PrecisionMode.BALANCED
    2: SPFT_MODE_ACCURATE,   # PrecisionMode.ACCURATE
    3: SPFT_MODE_EXACT,      # PrecisionMode.EXACT
}

_ROUTE_MAP = {
    0: SPFT_ROUTE_STATIC,    # RoutingPolicy.STATIC
    2: SPFT_ROUTE_PINNED,    # RoutingPolicy.PINNED
}


def py_spec_to_c_spec(spec: ReducerSpec) -> CReducerSpec:
    """Translate a Python :class:`~spf.tensor.core.ReducerSpec` to a :class:`CReducerSpec`.

    Raises
    ------
    ValueError
        If the spec uses a policy / routing that the C library does not support
        (e.g. ReducerPolicy != AUTO, or RoutingPolicy.ADAPTIVE).
    """
    from spf.tensor.core import ReducerPolicy, RoutingPolicy

    if spec.policy != ReducerPolicy.AUTO:
        raise ValueError(
            f"native_reduce_axis only supports ReducerPolicy.AUTO; got {spec.policy!r}"
        )
    routing_int = int(spec.routing)
    if routing_int == int(RoutingPolicy.ADAPTIVE):
        raise ValueError("RoutingPolicy.ADAPTIVE is not supported by the C library")

    c_routing = _ROUTE_MAP.get(routing_int)
    if c_routing is None:
        raise ValueError(f"Unknown routing policy int {routing_int}")

    domain_int = int(spec.domain) if spec.domain is not None else SPFT_DOMAIN_NONE

    return make_reducer_spec(
        fmt=SPFT_FMT_SPF32,
        policy=SPFT_POLICY_AUTO,
        precision_mode=_MODE_MAP.get(int(spec.precision_mode), SPFT_MODE_BALANCED),
        routing=c_routing,
        domain=domain_int,
    )


# ---------------------------------------------------------------------------
# Core reduce implementation
# ---------------------------------------------------------------------------


def _native_reduce_rows(rows: np.ndarray, c_spec: CReducerSpec) -> np.ndarray:
    """Call ``spft_reduce_rows`` on a 2-D ``(n_rows, row_len)`` word matrix."""
    from spf.native.spftensor import _check

    lib_mod = __import__("spf.native._libspftensor", fromlist=["_HANDLE"])
    lib = lib_mod._HANDLE
    if lib is None:
        raise RuntimeError("libspftensor is not available")

    rows_c = np.ascontiguousarray(rows, dtype=np.uint32)
    n_rows, row_len = rows_c.shape
    out = np.empty(n_rows, dtype=np.uint32)

    rc = lib.spft_reduce_rows(
        rows_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        ctypes.c_size_t(n_rows),
        ctypes.c_size_t(row_len),
        ctypes.c_size_t(row_len),  # dense stride
        ctypes.byref(c_spec),
        out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        None,  # stats_out
        None,  # domain_out
        None,  # telemetry
    )
    _check(rc, "spft_reduce_rows")
    return out


def native_reduce_axis(
    terms: npt.ArrayLike,
    spec: ReducerSpec,
    axis: int | None = None,
) -> np.ndarray | np.uint32:
    """Reduce packed SPF words over ``axis`` using the C native backend.

    Semantically equivalent to :func:`spf.tensor.reducers.reduce_axis` for
    ``ReducerPolicy.AUTO`` specs with STATIC or PINNED routing.

    Raises ``RuntimeError`` if ``LIBSPFTENSOR_AVAILABLE`` is False or if the
    spec is not supported by the C library.

    Parameters
    ----------
    terms : array-like of uint32
    spec : ReducerSpec with policy == AUTO
    axis : int or None
        Reduction axis (same semantics as ``reduce_axis``).
    """
    if not LIBSPFTENSOR_AVAILABLE:
        raise RuntimeError(
            "native_reduce_axis requires libspftensor; "
            "build with `make -C native` or set SPFTENSOR_LIBRARY."
        )

    c_spec = py_spec_to_c_spec(spec)
    arr = np.asarray(terms, dtype=np.uint32)

    if axis is None:
        rows = arr.reshape(1, -1)
        out_shape: tuple[int, ...] = ()
    else:
        ax = int(axis)
        moved = np.moveaxis(arr, ax, -1)
        out_shape = moved.shape[:-1]
        rows = moved.reshape(-1, moved.shape[-1])

    flat = _native_reduce_rows(rows, c_spec)

    if axis is None:
        return np.uint32(flat[0])
    return flat.reshape(out_shape)


def maybe_native_reduce_axis(
    terms: npt.ArrayLike,
    spec: ReducerSpec,
    axis: int | None = None,
) -> np.ndarray | np.uint32:
    """Reduce packed SPF words with a transparent native-or-Python fallback.

    Uses :func:`native_reduce_axis` when ``LIBSPFTENSOR_AVAILABLE`` is True and
    the spec is supported. Falls back to :func:`spf.tensor.reducers.reduce_axis`
    otherwise — transparently, with no warning.

    Parameters
    ----------
    terms : array-like of uint32
    spec : ReducerSpec
    axis : int or None
    """
    if LIBSPFTENSOR_AVAILABLE:
        try:
            return native_reduce_axis(terms, spec, axis=axis)
        except (ValueError, RuntimeError):
            # Unsupported spec (non-AUTO policy, ADAPTIVE routing, etc.) —
            # fall through to the Python path silently.
            pass

    from spf.tensor.reducers import reduce_axis

    return reduce_axis(terms, spec, axis=axis)
