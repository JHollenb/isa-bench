"""Bounded regional Jet2 planning for a retained CUDA Hermitian field.

No dense field is fitted. The planner reads addressed frequency/magnitude
metadata and applies the analytic Fourier-character Taylor remainder. The
GPU constructs headers only for regions actually requested by its consumer.
These are real output jets, distinct from SpectralJetTile2D's log/phase jets.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CUDAQueryBlockPlan:
    tiles_per_axis: int
    approximate: np.ndarray
    residual: np.ndarray
    output_bounds: tuple[float, float, float]
    capacity: int = 131072

    @property
    def resident_bytes(self) -> int:
        return self.capacity * (8 + 18*8) + self.approximate.nbytes + self.residual.nbytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": "regional_real_jet2_with_addressed_spf_residual",
            "header_layout": ("value", "grad_x", "grad_y", "hess_xx", "hess_xy", "hess_yy"),
            "observables": ("height", "dy", "dx"),
            "tiles_per_axis": self.tiles_per_axis,
            "approximate_modes": int(self.approximate.size),
            "residual_modes": int(self.residual.size),
            "additional_absolute_bounds": self.output_bounds,
            "cache_capacity": self.capacity,
            "resident_bytes": self.resident_bytes,
            "cached_output_pixels": 0,
            "coefficient_materialization": "visible_regions_on_demand",
            "concurrency": "nonblocking_publication_with_identical_private_header_on_contention",
        }


def plan_query_blocks(
    source: Any, absolute_budget: float, *, max_tiles: int = 4096,
) -> CUDAQueryBlockPlan | None:
    budget = float(absolute_budget)
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("query error budget must be finite and nonnegative")
    if budget == 0 or not len(source.words):
        return None
    words = np.asarray(source.words, dtype=np.uint32)
    indices = np.asarray(source.indices, dtype=np.int64) % np.asarray(source.shape)
    logs = (words >> np.uint32(16)).astype(np.uint16).view(np.int16)
    active = (logs != -32768) & (logs != 32767)
    signed = np.where(indices <= np.asarray(source.shape)//2, indices, indices-np.asarray(source.shape))
    frequency = 2*math.pi*signed/(np.asarray(source.shape)*np.asarray(source.spacing))
    nyquist = np.zeros_like(indices, dtype=bool)
    for axis, size in enumerate(source.shape):
        if size % 2 == 0:
            nyquist[:, axis] = indices[:, axis] == size//2
    self_conjugate = np.all((indices == 0) | nyquist, axis=1)
    amplitude = np.where(active, np.exp2(logs.astype(float)/256), 0.0)
    amplitude *= np.exp(-0.5*np.sum((frequency*np.asarray(source.footprint))**2, axis=1))
    amplitude *= np.where(self_conjugate, 1.0, 2.0)
    if source.normalize:
        amplitude /= math.prod(source.shape)
    observable_gain = amplitude[:, None]*np.column_stack((np.ones(len(words)), np.abs(frequency)))
    rounding = (len(words)+128)*np.finfo(np.float64).eps
    rounding /= 1-rounding
    best = None
    unavoidable = int(np.count_nonzero(active & (np.any(nyquist, axis=1) | self_conjugate)))
    tiles = 32
    while tiles <= max_tiles:
        radius = math.pi*np.sum(np.abs(signed), axis=1)/tiles
        remainder = radius**3/6
        polynomial_norm = 1+radius+radius*radius/2
        error = observable_gain*(remainder*(1+1e-4) + 4e6*np.finfo(float).eps*radius +
                                 rounding*polynomial_norm)[:, None]
        eligible = active & ~np.any(nyquist, axis=1) & ~self_conjugate
        eligible &= np.all(np.isfinite(error) & (error <= budget/max(1, int(np.count_nonzero(active)))), axis=1)
        approximate = np.flatnonzero(eligible).astype(np.int64)
        residual = np.flatnonzero(active & ~eligible).astype(np.int64)
        if len(approximate) > len(residual):
            bounds = tuple(float(x) for x in np.sum(error[eligible], axis=0))
            best = CUDAQueryBlockPlan(
                tiles, approximate, residual, (bounds[0], bounds[1], bounds[2]),
                capacity=min(tiles*tiles, 1 << 20),
            )
            if len(residual) <= unavoidable + max(1, (int(np.count_nonzero(active))-unavoidable)//10):
                break
        tiles *= 2
    if best is not None:
        best.approximate.setflags(write=False)
        best.residual.setflags(write=False)
    return best
