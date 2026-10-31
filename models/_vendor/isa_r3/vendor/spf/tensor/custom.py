"""Custom-named operator registry — the user extension hook (a deliberate stub).

The spectral surfaces in this package (character argmax readout, latent phase
projection, circulant/GFNet spectrum) cover the families recovered so far. A
user with a *new* operator — a learned mixer, a bespoke phase law — needs to
name it and slot it in without editing the library. This module is that seam:
:func:`register_custom_operator` reserves a name now; the ``builder`` that
actually constructs the operator can be supplied at registration or filled in
later. Calling an unbuilt name raises a clear stub error rather than failing
obscurely.

This is intentionally minimal: it carries the *name → builder* binding and
nothing about semantics. A custom operator that needs to be plan-legal still
registers an :class:`~spf.tensor.op_registry.OpSpec` via
:func:`spf.tensor.op_registry.register_op`, and a custom v3 base model still
registers via :func:`spf.tensor.v3.register_base_model`; this registry is the
human-facing index that ties a chosen name to whichever of those a user wires up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CUSTOM_OPERATOR_REGISTRY",
    "CustomOperatorSpec",
    "custom_operator",
    "custom_operators",
    "get_custom_operator",
    "register_custom_operator",
]


@dataclass(frozen=True, slots=True)
class CustomOperatorSpec:
    """A user-registered operator name, its family tag, and an optional builder.

    ``builder`` is any callable returning the operator/surface object; when it is
    ``None`` the name is reserved but not yet implemented, and :func:`custom_operator`
    raises a stub error pointing at it.
    """

    name: str
    family: str = "custom"
    builder: Callable[..., Any] | None = None
    summary: str = ""

    @property
    def implemented(self) -> bool:
        return self.builder is not None


CUSTOM_OPERATOR_REGISTRY: dict[str, CustomOperatorSpec] = {}


def register_custom_operator(
    name: str,
    *,
    family: str = "custom",
    builder: Callable[..., Any] | None = None,
    summary: str = "",
    replace: bool = False,
) -> CustomOperatorSpec:
    """Register (or reserve) a custom operator name.

    Pass ``builder`` to make the name immediately constructible via
    :func:`custom_operator`; omit it to reserve the name as a stub and supply the
    builder later (call again with ``replace=True``). Raises on a duplicate name
    unless ``replace=True`` so a typo cannot silently shadow an existing entry.
    """
    if not name or not isinstance(name, str):
        raise ValueError("custom operator name must be a non-empty string")
    if name in CUSTOM_OPERATOR_REGISTRY and not replace:
        raise ValueError(f"custom operator {name!r} already registered (pass replace=True to override)")
    spec = CustomOperatorSpec(name=name, family=family, builder=builder, summary=summary)
    CUSTOM_OPERATOR_REGISTRY[name] = spec
    return spec


def get_custom_operator(name: str) -> CustomOperatorSpec:
    """Return the :class:`CustomOperatorSpec` for ``name`` or raise ``KeyError``."""
    return CUSTOM_OPERATOR_REGISTRY[name]


def custom_operator(name: str, /, *args: Any, **kwargs: Any) -> Any:
    """Construct a registered custom operator by name via its builder.

    Raises ``KeyError`` for an unknown name and ``NotImplementedError`` for a
    reserved-but-unbuilt name (the stub case) — the message names what to wire up.
    """
    try:
        spec = CUSTOM_OPERATOR_REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"no custom operator registered as {name!r}; "
            f"call register_custom_operator({name!r}, builder=...) first"
        ) from None
    if spec.builder is None:
        raise NotImplementedError(
            f"custom operator {name!r} is reserved but has no builder yet — "
            f"register_custom_operator({name!r}, builder=..., replace=True) to implement it"
        )
    return spec.builder(*args, **kwargs)


def custom_operators() -> frozenset[str]:
    """Live set of registered custom operator names."""
    return frozenset(CUSTOM_OPERATOR_REGISTRY)
