"""Length-32 bounded-integer cyclic complex convolution operator.

Wraps the existing FNT/Nussbaumer implementation:

* Python: :func:`spf.fnt.fermat_conv_32_signed` (always available).
* C:      :func:`spf.native.fnt.fermat_conv_32` (used when ``libfermat_ntt``
  is built — :data:`spf.native.fnt.LIBFERMAT_NTT_AVAILABLE`).

The operator's contract is narrower than the underlying primitive's
dynamic range — see ``src/spf/fnt/nussbaumer.py`` module docstring.
The 8-bit signed bound here matches the regime that is property-tested
against the naive O(N²) reference, and keeps the F_5 ring well clear
of saturation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from spf.fnt.nussbaumer import fermat_conv_32_signed

CYCLIC_CONV32_LENGTH: int = 32
CYCLIC_CONV32_BMAX: int = 127  # signed 8-bit


def cyclic_conv32(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    *,
    unchecked: bool = False,
) -> npt.NDArray[np.int32]:
    """Length-32 cyclic complex convolution of ``a`` and ``b`` over the
    Fermat ring F_5 = 2³² + 1, returned as signed int32 (re, im) pairs.

    Parameters
    ----------
    a, b
        Length-32 array-likes of complex (or 2-channel real) integer
        samples. Each ``re`` / ``im`` component must satisfy
        ``|x| ≤ CYCLIC_CONV32_BMAX`` (signed 8-bit). Inputs are cast
        to ``int64`` internally; floats are accepted only if they
        round to integers within the bound (the precondition check
        rejects non-integer floats).
    unchecked
        Skip the length + bound checks. Intended for benchmark loops
        where the caller has already validated the input shape and
        magnitude. Passing inputs that violate the preconditions with
        ``unchecked=True`` is undefined behavior — output may wrap
        modulo F_5 instead of being the true cyclic convolution.

    Returns
    -------
    np.ndarray
        Shape ``(32, 2)`` int32 array. Column 0 is the real part of
        each output lane; column 1 is the imaginary part.

    Raises
    ------
    ValueError
        If ``unchecked`` is False and either input is not length 32,
        is not real-integer-valued, or has any ``|re|``/``|im|``
        exceeding :data:`CYCLIC_CONV32_BMAX`.

    Notes
    -----
    The algorithm is bit-exact against a naive O(N²) cyclic-convolution
    reference under the documented bound; tested via
    ``tests/fnt/test_cyclic_conv32.py`` and ``tests/fnt/test_T4_2_convolution.py``.
    There is no Sb()/LUT/libm dependency on the hot path — the
    convolution is realised by Fermat-NTT butterflies (bit shifts +
    modular add/sub) plus pointwise Gaussian-integer products in F_5.
    """
    a_re, a_im = _split_complex(a, "a", unchecked=unchecked)
    b_re, b_im = _split_complex(b, "b", unchecked=unchecked)
    c_re, c_im = fermat_conv_32_signed(a_re, a_im, b_re, b_im)
    out = np.empty((CYCLIC_CONV32_LENGTH, 2), dtype=np.int32)
    out[:, 0] = c_re.astype(np.int32, copy=False)
    out[:, 1] = c_im.astype(np.int32, copy=False)
    return out


def _split_complex(
    arr: npt.ArrayLike, name: str, *, unchecked: bool
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """Decompose ``arr`` into (re, im) int64 arrays of length 32.

    Accepts complex arrays, 2-channel real arrays of shape ``(32, 2)``,
    or already-split ``(re, im)`` tuples of length-32 reals. Bound /
    length checks fire here so the wrapped FNT call sees only valid
    input.
    """
    np_arr = np.asarray(arr)
    if np.iscomplexobj(np_arr):
        re = np_arr.real
        im = np_arr.imag
    elif np_arr.ndim == 2 and np_arr.shape[-1] == 2:
        re = np_arr[..., 0]
        im = np_arr[..., 1]
    else:
        raise ValueError(
            f"{name}: expected complex array of shape ({CYCLIC_CONV32_LENGTH},) "
            f"or real array of shape ({CYCLIC_CONV32_LENGTH}, 2); "
            f"got dtype={np_arr.dtype} shape={np_arr.shape}"
        )

    if not unchecked:
        if re.shape != (CYCLIC_CONV32_LENGTH,):
            raise ValueError(
                f"{name}: length must be {CYCLIC_CONV32_LENGTH}; "
                f"got {re.shape}"
            )
        for label, part in (("re", re), ("im", im)):
            if not np.all(np.equal(np.mod(part, 1), 0)):
                raise ValueError(
                    f"{name}.{label}: cyclic_conv32 takes integer-valued inputs"
                )
            # Promote to int64 before |x| so the bound check survives
            # signed-integer minimums (e.g. np.int8(-128) where np.abs
            # would otherwise stay -128 and silently pass `> 127`).
            wide = np.asarray(part, dtype=np.int64)
            max_abs = int(np.max(np.abs(wide))) if wide.size else 0
            if max_abs > CYCLIC_CONV32_BMAX:
                raise ValueError(
                    f"{name}.{label}: max(|x|)={max_abs} exceeds "
                    f"CYCLIC_CONV32_BMAX={CYCLIC_CONV32_BMAX} "
                    "(operator regime is signed 8-bit)"
                )

    return re.astype(np.int64, copy=False), im.astype(np.int64, copy=False)
