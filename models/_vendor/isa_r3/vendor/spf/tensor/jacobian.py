"""SPFTensor v3 Jacobian-operator block (Axis 2) — a linear-operator view.

A Jacobian ``J`` (``∂fᵢ/∂xⱼ``) is stored as a named base model that carries
**parameterized metadata** — where the Jacobian was evaluated, what its rows and
columns mean, its declared structure, and interpretation hints (units, scale).
The matrix *values* ride the existing ``RAW_INT16`` residual over a **zero
base**, exactly like a dense block; the base model's job is to

1. carry the metadata in ``base.params``, and
2. select the block's active set from the ``structure`` hint so a *diagonal*
   Jacobian stores only its diagonal (off-diagonal reconstructs to exact 0).

This is an Axis-2 addition only: ``Representation`` stays ``POLAR_SPF32`` and the
residual codec stays ``RAW_INT16``. Registration happens at import;
:mod:`spf.tensor.v3` triggers it lazily via ``_ensure_registries``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .core import SPFError, encode_complex
from .v3 import (
    BaseModelKind,
    BaseModelRef,
    Representation,
    RepresentationRef,
    ResidualCodecKind,
    ResidualRef,
    SPFBlockV3,
    register_base_model,
)

# Structures whose active-set logic is implemented today. The rest are reserved
# names — accepted as a vocabulary but rejected at construction so we never ship
# a half-packed active set (a diagonal store with no diagonal logic, etc.).
_SUPPORTED_STRUCTURES = ("dense", "diagonal")
_RESERVED_STRUCTURES = ("sparse", "low-rank", "symmetric")


def _active_indices(structure: str, dims: tuple[int, ...]) -> np.ndarray:
    """Active linear indices for a Jacobian of shape ``dims`` and given structure.

    Single source of truth shared by :func:`build_jacobian_block` and
    :class:`JacobianOperatorModel` so the stored residual length and the
    reconstructed active set can never drift (a mismatch would silently corrupt
    the planes via numpy broadcasting in ``SPFBlockV3.reconstruct``).
    """
    if len(dims) != 2:
        raise SPFError(f"Jacobian block requires a 2-D matrix, got dims={tuple(dims)!r}")
    m, n = int(dims[0]), int(dims[1])
    if structure == "dense":
        return np.arange(m * n, dtype=np.int64)
    if structure == "diagonal":
        idx = np.arange(min(m, n), dtype=np.int64)
        return idx * n + idx
    if structure in _RESERVED_STRUCTURES:
        raise SPFError(
            f"Jacobian structure {structure!r} is reserved but not yet supported "
            f"(supported: {_SUPPORTED_STRUCTURES})"
        )
    raise SPFError(f"unknown Jacobian structure {structure!r}")


class JacobianOperatorModel:
    """Axis-2 base model for a Jacobian / linear-operator matrix view.

    ``params`` (the parameterized metadata; only ``structure`` affects the
    reconstructed planes, the rest are informational and ride the wire):

      * ``structure``  — ``"dense"`` or ``"diagonal"`` (drives the active set).
      * ``eval_point`` — int64 coords the Jacobian was evaluated at (optional).
      * ``order``      — derivative order (1 = Jacobian; informational).
      * ``in_labels`` / ``out_labels`` — column / row names (optional).
      * ``units`` / ``scale`` — interpretation hints (optional).

    The base contributes **zero** planes; the matrix values ride the residual
    (``RAW_INT16``). Reconstruction therefore yields the residual values on the
    active set and the zero sentinel elsewhere.
    """

    kind = int(BaseModelKind.JACOBIAN_OPERATOR)

    def reconstruct_planes(
        self,
        params: Mapping[str, Any],
        dims: tuple[int, ...],
        origin: tuple[int, ...],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        active = _active_indices(str(params.get("structure", "dense")), tuple(dims))
        zeros = np.zeros(active.size, dtype=np.int64)
        return active, zeros, zeros

    def storage_bits(self, params: Mapping[str, Any]) -> int:
        # Metadata only — the matrix values are charged by the residual codec in
        # SPFBlockV3.storage_bits, so do not count them here.
        bits = 16  # order (~8b) + structure tag (~8b)
        eval_point = params.get("eval_point")
        if eval_point is not None:
            bits += int(np.asarray(eval_point).size) * 64
        for key in ("in_labels", "out_labels"):
            labels = params.get(key)
            if labels:
                bits += sum(len(str(s)) for s in labels) * 8
        if params.get("units"):
            bits += len(str(params["units"])) * 8
        if params.get("scale") is not None:
            bits += 64
        return bits


def _normalise_labels(labels: Sequence[Any] | None) -> list[str] | None:
    """Coerce labels to ``list[str]`` (the envelope collapses tuple→list anyway)."""
    if labels is None:
        return None
    return [str(s) for s in labels]


def build_jacobian_block(
    J: Any,
    *,
    eval_point: Any = None,
    in_labels: Sequence[Any] | None = None,
    out_labels: Sequence[Any] | None = None,
    order: int = 1,
    structure: str = "dense",
    units: Any = None,
    scale: float | None = 1.0,
    representation: int = int(Representation.POLAR_SPF32),
) -> SPFBlockV3:
    """Build a v3 Jacobian-operator block from a dense 2-D matrix + metadata.

    ``J`` is a 2-D (complex or real) matrix. ``structure`` selects which entries
    are stored: ``"dense"`` keeps all ``m·n`` entries, ``"diagonal"`` keeps only
    the ``min(m, n)`` diagonal entries (off-diagonal reconstructs to exact 0).
    The remaining keyword arguments are informational metadata carried verbatim
    on the wire and returned by :func:`jacobian_meta`.
    """
    zz = np.asarray(J, dtype=np.complex128)
    if zz.ndim != 2:
        raise SPFError(f"Jacobian block requires a 2-D matrix, got ndim={zz.ndim}")
    dims = (int(zz.shape[0]), int(zz.shape[1]))
    # Validate the structure and compute the active set once (shared with the
    # model so the residual length matches active.size exactly).
    active = _active_indices(structure, dims)
    plane_L, plane_T = encode_complex(zz.reshape(-1)[active])
    dL = np.asarray(plane_L, dtype=np.int64)
    dT = np.asarray(plane_T, dtype=np.int64)

    meta: dict[str, Any] = {
        "structure": str(structure),
        "order": int(order),
        "scale": None if scale is None else float(scale),
    }
    if eval_point is not None:
        meta["eval_point"] = np.asarray(eval_point, dtype=np.int64)
    if in_labels is not None:
        meta["in_labels"] = _normalise_labels(in_labels)
    if out_labels is not None:
        meta["out_labels"] = _normalise_labels(out_labels)
    if units is not None:
        meta["units"] = str(units)

    base = BaseModelRef(int(BaseModelKind.JACOBIAN_OPERATOR), meta)
    residual = ResidualRef(int(ResidualCodecKind.RAW_INT16), {"dL": dL, "dT": dT})
    return SPFBlockV3(
        dims=dims,
        origin=(0, 0),
        representation=RepresentationRef(int(representation)),
        base=base,
        residual=residual,
    )


def jacobian_meta(block: SPFBlockV3) -> dict[str, Any]:
    """Return the parameterized metadata dict carried by a Jacobian block."""
    if int(block.base.profile_id) != int(BaseModelKind.JACOBIAN_OPERATOR):
        raise SPFError("block is not a JACOBIAN_OPERATOR block")
    return dict(block.base.params)


register_base_model(JacobianOperatorModel())


__all__ = [
    "JacobianOperatorModel",
    "build_jacobian_block",
    "jacobian_meta",
]
