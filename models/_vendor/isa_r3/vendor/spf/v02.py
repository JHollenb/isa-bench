"""Public SPF v0.2 reference surface.

The implementation lives under :mod:`spf.tensor.v02` so tensor users can opt
into it without changing the established v1 namespace.  This module is the
short public import path: ``from spf.v02 import reduce_complex``.
"""

from __future__ import annotations

from .tensor import v02 as _v02

__all__ = _v02.__all__
globals().update({name: getattr(_v02, name) for name in __all__})
