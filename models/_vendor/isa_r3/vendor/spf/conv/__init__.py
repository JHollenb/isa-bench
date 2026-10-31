"""Bounded cyclic-convolution operators.

This namespace exposes operator-default cyclic-convolution surfaces.
Members here are the public API for callers who want a convolution
*operator* and accept the operator's preconditions; they are not a
replacement for arbitrary convolution or for ``spf.fft.spf32_fft``.

Today the only member is :func:`cyclic_conv32`, which routes length-32
bounded-integer cyclic complex convolution through the existing
Fermat-NTT / Nussbaumer pipeline (``spf.fnt.fermat_conv_32_signed`` /
``spf.native.fnt.fermat_conv_32``). It is the first ``sparse_dsp``-profile
operator default per ``docs/research/default-promotion-policy.md``.

A generic length-aware ``cyclic(a, b, *, profile=...)`` dispatcher is
intentionally not provided yet — there is only one implementation, and
callers should commit to the precondition set explicitly via the
length-pinned operator name. When a second cyclic-conv impl lands
(e.g. length-64 NTT), a dispatcher can be added here that resolves
length + profile to a concrete operator.
"""

from __future__ import annotations

from spf.conv._cyclic_conv32 import (
    CYCLIC_CONV32_BMAX,
    CYCLIC_CONV32_LENGTH,
    cyclic_conv32,
)

__all__ = [
    "CYCLIC_CONV32_BMAX",
    "CYCLIC_CONV32_LENGTH",
    "cyclic_conv32",
]
