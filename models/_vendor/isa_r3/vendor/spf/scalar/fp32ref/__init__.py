"""Scalar fp32 reference oracle (re-export shim).

The canonical implementation lives at :mod:`spf.native.fp32ref` (libfp32ref
ctypes bindings). This shim exposes the same surface under the new
`<arch>/<type>/` layout. New code may import either path; the legacy
`spf.native.fp32ref` import keeps working for downstream callers.
"""

from spf.native import fp32ref as _fp32ref
from spf.native.fp32ref import *  # noqa: F403

__all__ = [name for name in dir(_fp32ref) if not name.startswith("_")]
