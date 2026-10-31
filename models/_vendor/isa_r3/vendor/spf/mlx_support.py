"""Small lazy helpers for optional MLX acceleration."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_mlx_available() -> bool:
    try:
        import mlx.core  # type: ignore[import-not-found]  # noqa: F401
    except Exception:
        return False
    return True


def require_mlx() -> Any:
    try:
        import mlx.core as mlx
    except Exception as exc:  # pragma: no cover - exercised only without MLX
        raise RuntimeError(
            "MLX support requires `mlx` on Apple Silicon. Install with `uv sync --extra mlx`."
        ) from exc
    return mlx


def to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def to_mlx_f32(value: Any) -> Any:
    mlx = require_mlx()
    if value.__class__.__module__.startswith("mlx."):
        return value.astype(mlx.float32) if getattr(value, "dtype", None) != mlx.float32 else value
    return mlx.array(np.asarray(value), dtype=mlx.float32)


def synchronize(*values: Any) -> None:
    mlx = require_mlx()
    if hasattr(mlx, "eval") and values:
        mlx.eval(*values)
