"""``spf.numpy.fft`` — SPF FFT submodule.

Mirrors the ``numpy.fft`` interface for :class:`~spf.array.SPFArray` inputs::

    import spf.numpy as snp

    X = snp.fft.fft(x)       # legacy-compatible forward DFT
    X02 = snp.fft.fft(x, profile="v02")
    x_rec = snp.fft.ifft(X)  # inverse DFT

Both functions accept an :class:`~spf.array.SPFArray` and return an
:class:`~spf.array.SPFArray` of the same shape, format, and backend.
The underlying transform is the radix-2 SPF-32 DIT FFT from
:mod:`spf.fft`; batched leading axes are supported (shape ``(..., N)``).

Only trailing-axis transforms are implemented (``axis=-1`` semantics).
For other axes, transpose before calling.
"""

from __future__ import annotations

from spf.array import SPFArray

__all__ = ["fft", "ifft"]


def fft(
    x: SPFArray,
    axis: int = -1,
    norm: None = None,
    *,
    profile: str | None = None,
    precision: object = "balanced",
) -> SPFArray:
    """Forward SPF-32 DFT along the trailing axis.

    Parameters
    ----------
    x : SPFArray
        Input in SPF-32 encoding.  Shape ``(..., N)`` where ``N`` is a
        power of two in ``[2, 1024]``.
    axis : int
        Transform axis; only ``-1`` (trailing) is supported.
    norm : None
        Reserved for future NumPy-compatible normalisation; must be ``None``.
    profile : str, optional
        ``"legacy"`` (the default) uses the selected backend's established
        transform.  ``"v02"`` selects the v0.2 reference transform and is
        currently available for ``backend="reference"`` SPF-32 arrays.
    precision : object
        v0.2 precision mode.  Non-default values require ``profile="v02"``.

    Returns
    -------
    SPFArray
        DFT of *x*, same shape and backend.
    """
    if not isinstance(x, SPFArray):
        raise TypeError(f"Expected SPFArray, got {type(x).__name__}")
    if norm is not None:
        raise NotImplementedError("norm != None not supported in spf.numpy.fft.fft")
    effective_axis = axis if axis >= 0 else x.ndim + axis
    if effective_axis != x.ndim - 1:
        raise NotImplementedError(
            f"Only trailing-axis FFT is supported (axis=-1 / axis={x.ndim - 1}). "
            f"Got axis={axis}. Transpose the array first for other axes."
        )
    return x._transform(profile=profile, precision=precision, inverse=False)


def ifft(
    x: SPFArray,
    axis: int = -1,
    norm: None = None,
    *,
    profile: str | None = None,
    precision: object = "balanced",
) -> SPFArray:
    """Inverse SPF-32 DFT along the trailing axis.

    Parameters and return value mirror :func:`fft`.
    """
    if not isinstance(x, SPFArray):
        raise TypeError(f"Expected SPFArray, got {type(x).__name__}")
    if norm is not None:
        raise NotImplementedError("norm != None not supported in spf.numpy.fft.ifft")
    effective_axis = axis if axis >= 0 else x.ndim + axis
    if effective_axis != x.ndim - 1:
        raise NotImplementedError(
            f"Only trailing-axis IFFT is supported (axis=-1 / axis={x.ndim - 1}). "
            f"Got axis={axis}. Transpose the array first for other axes."
        )
    return x._transform(profile=profile, precision=precision, inverse=True)
