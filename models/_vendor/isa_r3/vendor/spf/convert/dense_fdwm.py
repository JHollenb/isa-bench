"""Dense FP16/FP32 matrix → FDWM-C / FDWM-BC storage (spec §11, §7.4, §8).

FDWM stores a circulant or block-circulant matrix by its first column's FFT
spectrum. The nearest-circulant projection (diagonal averaging) is implemented
in :mod:`spf.fdwm.projection`; this module just re-exports those functions
under convert-layer names so the boundary-crossing API is discoverable from a
single namespace.

Spec cost: ``O(n log n)``, one-time per weight matrix. These are not hot-path
conversions.
"""

from __future__ import annotations

import numpy as np

from spf.fdwm.projection import project_to_block_circulant, project_to_circulant

__all__ = ["dense_to_fdwm_bc", "dense_to_fdwm_c"]


def dense_to_fdwm_c(W: np.ndarray) -> np.ndarray:
    """Project a dense ``(n, n)`` matrix onto the nearest circulant and return
    the FFT of its first column (the FDWM-C stored form).
    """
    return project_to_circulant(W)


def dense_to_fdwm_bc(W: np.ndarray, *, k: int) -> np.ndarray:
    """Project a dense ``(n, n)`` matrix onto a block-circulant form with
    ``k×k`` blocks and return the rfft spectrum of each block's first column.
    """
    return project_to_block_circulant(W, k=k)
