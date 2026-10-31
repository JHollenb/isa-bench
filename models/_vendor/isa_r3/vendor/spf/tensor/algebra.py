"""Algebra registry for SPFBlockMatrix kernel laws.

The registry is intentionally thin: storage and scalar semantics remain in
``SPFBlockV3`` / ``SPFTensor``. This module owns only dispatch between typed
kernel expressions and structural algebra laws.
"""

from __future__ import annotations

from typing import Any, Protocol


class BaseModelAlgebra(Protocol):
    """Optional algebra hooks for one base model or structural expression kind."""

    kind: int | str

    def pullback(self, expr: Any, axis_map: dict[str, str]) -> Any | None: ...

    def bind(self, a: Any, b: Any) -> Any | None: ...

    def pushforward(self, expr: Any, reduce_axes: tuple[str, ...]) -> Any | None: ...

    def compose(self, a: Any, b: Any) -> Any | None: ...

    def adjoint(self, a: Any) -> Any | None: ...

    def inverse(self, a: Any) -> Any | None: ...

    def power(self, a: Any, k: int) -> Any | None: ...


BASE_MODEL_ALGEBRA_REGISTRY: dict[int, BaseModelAlgebra] = {}
STRUCTURAL_ALGEBRA_REGISTRY: dict[str, BaseModelAlgebra] = {}


def register_base_model_algebra(algebra: BaseModelAlgebra) -> BaseModelAlgebra:
    BASE_MODEL_ALGEBRA_REGISTRY[int(algebra.kind)] = algebra
    return algebra


def register_structural_algebra(algebra: BaseModelAlgebra) -> BaseModelAlgebra:
    STRUCTURAL_ALGEBRA_REGISTRY[str(algebra.kind)] = algebra
    return algebra


def _maybe_call(algebra: BaseModelAlgebra, method_name: str, *args: Any, **kwargs: Any) -> Any | None:
    method = getattr(algebra, method_name, None)
    if method is None:
        return None
    return method(*args, **kwargs)


def pullback_with_registry(expr: Any, axis_map: dict[str, str], **kwargs: Any) -> Any | None:
    """Try registered structural/base-model pullback laws."""

    algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(str(expr.expr.op))
    if algebra is not None:
        out = _maybe_call(algebra, "pullback", expr, axis_map, **kwargs)
        if out is not None:
            return out
    if expr.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(expr.expr.block.base.profile_id))
        if algebra is not None:
            out = _maybe_call(algebra, "pullback", expr, axis_map, **kwargs)
            if out is not None:
                return out
    return None


def bind_with_registry(a: Any, b: Any, **kwargs: Any) -> Any | None:
    """Try registered structural/base-model bind laws."""

    for key in (f"{a.expr.op}:{b.expr.op}", str(a.expr.op), str(b.expr.op)):
        algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(key)
        if algebra is not None:
            out = _maybe_call(algebra, "bind", a, b, **kwargs)
            if out is not None:
                return out
    if a.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(a.expr.block.base.profile_id))
        if algebra is not None:
            out = _maybe_call(algebra, "bind", a, b, **kwargs)
            if out is not None:
                return out
    return None


def pushforward_with_registry(expr: Any, reduce_axes: tuple[str, ...], **kwargs: Any) -> Any | None:
    """Try registered structural/base-model pushforward laws."""

    algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(str(expr.expr.op))
    if algebra is not None:
        out = _maybe_call(algebra, "pushforward", expr, reduce_axes, **kwargs)
        if out is not None:
            return out
    if expr.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(expr.expr.block.base.profile_id))
        if algebra is not None:
            out = _maybe_call(algebra, "pushforward", expr, reduce_axes, **kwargs)
            if out is not None:
                return out
    return None


def compose_with_registry(a: Any, b: Any) -> Any | None:
    """Try registered structural/base-model composition laws."""

    for key in (f"{a.expr.op}:{b.expr.op}", str(a.expr.op), str(b.expr.op)):
        algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(key)
        if algebra is not None:
            out = _maybe_call(algebra, "compose", a, b)
            if out is not None:
                return out
    if a.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(a.expr.block.base.profile_id))
        if algebra is not None:
            out = _maybe_call(algebra, "compose", a, b)
            if out is not None:
                return out
    return None


def adjoint_with_registry(a: Any) -> Any | None:
    for key in (str(a.expr.op),):
        algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(key)
        if algebra is not None:
            out = _maybe_call(algebra, "adjoint", a)
            if out is not None:
                return out
    if a.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(a.expr.block.base.profile_id))
        if algebra is not None:
            return _maybe_call(algebra, "adjoint", a)
    return None


def inverse_with_registry(a: Any) -> Any | None:
    for key in (str(a.expr.op),):
        algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(key)
        if algebra is not None:
            out = _maybe_call(algebra, "inverse", a)
            if out is not None:
                return out
    if a.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(a.expr.block.base.profile_id))
        if algebra is not None:
            return _maybe_call(algebra, "inverse", a)
    return None


def power_with_registry(a: Any, k: int) -> Any | None:
    for key in (str(a.expr.op),):
        algebra = STRUCTURAL_ALGEBRA_REGISTRY.get(key)
        if algebra is not None:
            out = _maybe_call(algebra, "power", a, int(k))
            if out is not None:
                return out
    if a.expr.block is not None:
        algebra = BASE_MODEL_ALGEBRA_REGISTRY.get(int(a.expr.block.base.profile_id))
        if algebra is not None:
            return _maybe_call(algebra, "power", a, int(k))
    return None


__all__ = [
    "BASE_MODEL_ALGEBRA_REGISTRY",
    "STRUCTURAL_ALGEBRA_REGISTRY",
    "BaseModelAlgebra",
    "adjoint_with_registry",
    "bind_with_registry",
    "compose_with_registry",
    "inverse_with_registry",
    "power_with_registry",
    "pullback_with_registry",
    "pushforward_with_registry",
    "register_base_model_algebra",
    "register_structural_algebra",
]
