"""OFDM resource-grid exports to normal IQ16 DAC samples.

``export_iq16_ifft`` is the canonical compatibility path used by the
current cross-link 2×2 proof — conventional OFDM IFFT producing IQ16
boundary samples. ``export_iq16_nco`` is kept in tree as a **future
optimization path** for sparse-DAC / direct-NCO synthesis (hardware
promotion target). It is parity-tested but is not part of the current
hardware-compatibility headline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import numpy.typing as npt

from spf.dsp.ofdm_grid import OFDMResourceGrid
from spf.tensor import SPFError, SynthMode, SynthPlan, SynthStats


def export_iq16_ifft(
    grid: OFDMResourceGrid,
    plan: SynthPlan,
    *,
    validate: bool = True,
) -> tuple[npt.NDArray[np.int16], SynthStats]:
    """Export a resource grid through the compatibility IFFT boundary.

    Validates ``grid`` via :meth:`OFDMResourceGrid.validate` before export.
    Use :func:`export_iq16_ifft_trusted` on the hot path when the grid
    was produced by a trusted constructor.
    """
    words = _grid_words_for_plan(grid, plan, validate=validate)
    ifft_plan = replace(plan, synth_mode=SynthMode.NORMAL_DAC_IFFT)
    return ifft_plan.export_normal_dac(words)


def export_iq16_ifft_trusted(
    grid: OFDMResourceGrid,
    plan: SynthPlan,
) -> tuple[npt.NDArray[np.int16], SynthStats]:
    """Like :func:`export_iq16_ifft` but skips ``grid.validate()``.

    The caller asserts ``grid`` was produced by a trusted constructor (e.g.
    :meth:`OFDMResourceGrid.from_trusted`, :func:`apply_diagonal_active_words`
    with ``trusted=True``) so the per-call validate is redundant. The
    ``SynthPlan`` is still validated and the grid shape is still checked
    against the plan.
    """
    words = _grid_words_for_plan(grid, plan, validate=False)
    ifft_plan = replace(plan, synth_mode=SynthMode.NORMAL_DAC_IFFT)
    return ifft_plan.export_normal_dac(words)


def export_iq16_nco(
    grid: OFDMResourceGrid,
    plan: SynthPlan,
    *,
    active_indices: Sequence[int] | None = None,
) -> tuple[npt.NDArray[np.int16], SynthStats]:
    """Export a resource grid through sparse NCO synthesis.

    Future optimization path; not part of the current hardware-compatibility
    proof. See the module docstring.
    """
    words = _grid_words_for_plan(grid, plan, validate=True)
    nco_plan = replace(plan, synth_mode=SynthMode.NORMAL_DAC_NCO)
    active = tuple(int(v) for v in grid.active_subcarriers()) if active_indices is None else active_indices
    return nco_plan.export_normal_dac(words, active_indices=active)


def _grid_words_for_plan(
    grid: OFDMResourceGrid,
    plan: SynthPlan,
    *,
    validate: bool = True,
) -> npt.NDArray[np.uint32]:
    if validate:
        grid.validate()
    plan.validate()
    if grid.shape[0] != 1:
        raise SPFError("SynthPlan grid export currently supports one frame at a time")
    if grid.shape[3] != 1:
        raise SPFError("SynthPlan grid export currently supports one antenna_or_beam at a time")
    if grid.shape[1] != plan.n_symbols or grid.shape[2] != plan.nfft:
        raise SPFError(
            "grid symbol/subcarrier shape does not match SynthPlan "
            f"({grid.shape[1]}, {grid.shape[2]}) != ({plan.n_symbols}, {plan.nfft})"
        )
    return np.ascontiguousarray(grid.logical_words()[0, :, :, 0], dtype=np.uint32)


__all__ = [
    "export_iq16_ifft",
    "export_iq16_ifft_trusted",
    "export_iq16_nco",
]
