"""GDB remote protocol re-export for the SPF runtime."""
from __future__ import annotations

from .compat import ensure_scripts_importable

ensure_scripts_importable()

from pan_x86_isa.gdbstub import GdbRemoteProtocol, GdbRemoteServer  # noqa: E402,F401

__all__ = ["GdbRemoteProtocol", "GdbRemoteServer"]
