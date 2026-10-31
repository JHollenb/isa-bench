"""``spf.numpy`` — NumPy-like SPF DSP namespace.

Drop-in evaluation layer for complex-DSP Python code.  Import as::

    import spf.numpy as snp

    x = snp.asarray(complex_samples, dtype=snp.spf32)
    H = snp.asarray(channel_estimate, dtype=snp.spf32)
    Y = snp.fft.fft(x) / H
    out = Y.to_numpy()

This namespace covers the complex-DSP subset only — not all of NumPy.
Unsupported operations raise ``NotImplementedError`` rather than silently
decoding to complex, so callers know exactly which ops execute in SPF.

Symbols
-------
``spf32``    — dtype sentinel: pass as ``dtype=snp.spf32`` to :func:`asarray`.
``asarray``  — encode complex array as :class:`~spf.array.SPFArray`.
``array``    — alias for :func:`asarray`.
``encode``   — alias for :func:`asarray`.
``decode``   — decode :class:`~spf.array.SPFArray` back to complex128 ndarray.
``conj``     — complex conjugate.
``abs2``     — magnitude squared as SPF word.
``multiply`` — element-wise SPF multiply (``*``).
``divide``   — element-wise SPF divide (``/``).
``add``      — element-wise SPF add via Sb().
``subtract`` — element-wise SPF subtract.
``fft``      — ``spf.numpy.fft`` submodule (``fft``, ``ifft``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as _np
import numpy.typing as _npt

from spf.array import SPFArray
from spf.array import asarray as _asarray

if TYPE_CHECKING:
    from spf.backends.kria.pl_client import PersistentPLSession

__all__ = [
    "SPFArray",
    "abs2",
    "add",
    "array",
    "asarray",
    "conj",
    "decode",
    "divide",
    "encode",
    "fft",
    "multiply",
    "spf32",
    "subtract",
]

# ── dtype sentinels ───────────────────────────────────────────────────────────

spf32: str = "spf32"
"""Dtype sentinel for SPF-32.  Pass as ``dtype=snp.spf32`` to :func:`asarray`."""


# ── array construction ────────────────────────────────────────────────────────


def asarray(
    z: _npt.ArrayLike | SPFArray,
    dtype: str | None = None,
    backend: str = "reference",
    session: PersistentPLSession | None = None,
) -> SPFArray:
    """Encode *z* as a :class:`~spf.array.SPFArray`.

    Parameters
    ----------
    z :
        Complex samples or an existing :class:`~spf.array.SPFArray`.
    dtype :
        Format string; pass ``snp.spf32`` or ``"spf32"``.  Default ``"spf32"``.
    backend :
        Execution backend (``"reference"``, ``"native"``, ``"kv260_pl"``).
    session :
        Optional :class:`~spf.backends.kria.pl_client.PersistentPLSession`.
        Only valid with ``backend="kv260_pl"``; routes PL ops on the
        resulting array through the open TCP session instead of spawning
        an SSH+pyxrt process per call.
    """
    return _asarray(z, dtype=dtype, backend=backend, session=session)


array = asarray
encode = asarray


def decode(w: SPFArray | _npt.NDArray[_np.uint32]) -> _npt.NDArray[_np.complex128]:
    """Decode *w* to a complex128 ndarray.

    Accepts an :class:`~spf.array.SPFArray` or a raw ``uint32`` ndarray
    (decoded with the reference backend).  Any other type raises ``TypeError``
    to avoid silently treating arbitrary data as SPF words.
    """
    if isinstance(w, SPFArray):
        return w.to_numpy()
    arr = _np.asarray(w)
    if arr.dtype != _np.uint32:
        raise TypeError(
            f"snp.decode expects an SPFArray or ndarray[uint32], "
            f"got {type(w).__name__} with dtype={getattr(arr, 'dtype', 'N/A')}"
        )
    from spf.scalar import spf32 as _spf32
    return _spf32.decode(arr)


# ── element-wise operations ───────────────────────────────────────────────────


def conj(x: SPFArray) -> SPFArray:
    """Complex conjugate (L unchanged, Θ ↦ −Θ mod 2¹⁶)."""
    if not isinstance(x, SPFArray):
        raise TypeError(f"Expected SPFArray, got {type(x).__name__}")
    return x.conj()


def abs2(x: SPFArray) -> SPFArray:
    """|x|² as a positive-real SPF word (L ↦ 2L, Θ = 0)."""
    if not isinstance(x, SPFArray):
        raise TypeError(f"Expected SPFArray, got {type(x).__name__}")
    return x.abs2()


def multiply(x: SPFArray | _npt.ArrayLike, y: SPFArray | _npt.ArrayLike) -> SPFArray:
    """Element-wise SPF multiply (integer adds on L and Θ — no Sb())."""
    if isinstance(x, SPFArray):
        return x * y
    if isinstance(y, SPFArray):
        return y.__rmul__(x)
    raise TypeError("At least one operand must be an SPFArray")


def divide(x: SPFArray | _npt.ArrayLike, y: SPFArray | _npt.ArrayLike) -> SPFArray:
    """Element-wise SPF divide (integer subtracts on L and Θ — no Sb())."""
    if isinstance(x, SPFArray):
        return x / y
    if isinstance(y, SPFArray):
        return y.__rtruediv__(x)
    raise TypeError("At least one operand must be an SPFArray")


def add(x: SPFArray | _npt.ArrayLike, y: SPFArray | _npt.ArrayLike) -> SPFArray:
    """Element-wise SPF add via Sb() LUT."""
    if isinstance(x, SPFArray):
        return x + y
    if isinstance(y, SPFArray):
        return y.__radd__(x)
    raise TypeError("At least one operand must be an SPFArray")


def subtract(x: SPFArray | _npt.ArrayLike, y: SPFArray | _npt.ArrayLike) -> SPFArray:
    """Element-wise SPF subtract via Sb() LUT."""
    if isinstance(x, SPFArray):
        return x - y
    if isinstance(y, SPFArray):
        return y.__rsub__(x)
    raise TypeError("At least one operand must be an SPFArray")


# ── fft submodule ─────────────────────────────────────────────────────────────

from spf.numpy import fft  # noqa: E402  (must follow SPFArray import)
