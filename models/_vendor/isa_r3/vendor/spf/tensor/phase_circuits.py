"""Sparse nonlinear address circuits over the existing SPF32 word planes.

Programs are ExecutionPlans, not new carriers or linear SPFKernel matrices.
No group enumeration, coherent reduction, or Cartesian conversion occurs here.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from .core import SPF32_PHASE_MASK, SPFError, pack_words, unpack_L, unpack_theta

Character = tuple[tuple[int, int], ...]


def _integer(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise SPFError("phase circuit parameters must be integers, not bools")
    try:
        return operator.index(value)
    except TypeError:
        raise SPFError("phase circuit parameters must be integers") from None


def canonical_character(row: Sequence[Sequence[int]], n_channels: int) -> Character:
    """Merge duplicate channels and reduce exponents in the exact phase ring."""
    terms: dict[int, int] = {}
    for term in row:
        if len(term) != 2:
            raise SPFError("each character term must be (channel, exponent)")
        channel, exponent = (_integer(v) for v in term)
        if not 0 <= channel < n_channels:
            raise SPFError("character channel out of range")
        terms[channel] = (terms.get(channel, 0) + exponent) & SPF32_PHASE_MASK
    return tuple((channel, exponent) for channel, exponent in sorted(terms.items()) if exponent)


def canonical_polynomial(coefficients: Sequence[int]) -> tuple[int, ...]:
    """Ascending coefficients reduced modulo 2^16, with redundant degree removed."""
    coeff = [_integer(v) & SPF32_PHASE_MASK for v in coefficients]
    while coeff and coeff[-1] == 0:
        coeff.pop()
    return tuple(coeff)


def _word_plane(words: npt.ArrayLike) -> np.ndarray:
    arr = np.asarray(words)
    if arr.dtype != np.uint32 or arr.ndim == 0:
        raise SPFError("phase circuits require SPF32 uint32 words with a channel axis")
    return arr


def _require_unit_channels(words: np.ndarray, channels: Sequence[int]) -> None:
    if channels and np.any(unpack_L(words[..., list(channels)]) != 0):
        raise SPFError("used phase circuit channels must be finite nonzero unit-circle words (L=0)")


def execute_phase_character_features(
    words: npt.ArrayLike,
    *,
    characters: Sequence[Sequence[Sequence[int]]],
    biases: Sequence[int] | None = None,
) -> np.ndarray:
    """Evaluate M sparse characters of live D-channel addresses in O(nnz + M).

    Output shape is words.shape[:-1]+(M,), with L=0. Only used inputs require
    L=0. Negative exponents are conjugate powers. Gauge charge is the sum of
    exponents: arbitrary features are not common-gauge invariant.
    """
    arr = _word_plane(words)
    rows = tuple(canonical_character(row, arr.shape[-1]) for row in characters)
    offsets = (0,) * len(rows) if biases is None else tuple(_integer(v) & SPF32_PHASE_MASK for v in biases)
    if len(offsets) != len(rows):
        raise SPFError("one phase bias is required per character")
    channels = tuple(sorted({channel for row in rows for channel, _ in row}))
    _require_unit_channels(arr, channels)
    theta = {channel: unpack_theta(arr[..., channel]).astype(np.int64) for channel in channels}
    out = np.empty((*arr.shape[:-1], len(rows)), dtype=np.uint16)
    for index, (row, bias) in enumerate(zip(rows, offsets, strict=True)):
        value = np.full(arr.shape[:-1], bias, dtype=np.int64)
        for channel, exponent in row:
            value = (value + exponent * theta[channel]) & SPF32_PHASE_MASK
        out[..., index] = value
    return pack_words(np.zeros(out.shape, dtype=np.int16), out)


def execute_phase_shear(
    words: npt.ArrayLike,
    *,
    target: int,
    control: int,
    reference: int | None = None,
    coefficients: Sequence[int] = (0, 0, 1),
    inverse: bool = False,
    table: Sequence[int] | None = None,
) -> np.ndarray:
    """Theta[target] += f(control-reference), exactly reversible by subtraction.

    Used coordinates require L=0; spectators are preserved bit-for-bit. The
    target cannot be its controller/reference. Relative control gives common-
    gauge equivariance; raw control needs a declared frame. ROM indexing is
    floor(theta*len(table)/65536), with outputs in phase ticks. Squaring can
    amplify one-tick perturbations; native does not imply smooth.
    """
    arr = _word_plane(words)
    target, control = _integer(target), _integer(control)
    reference = None if reference is None else _integer(reference)
    channels = (target, control) if reference is None else (target, control, reference)
    if any(not 0 <= channel < arr.shape[-1] for channel in channels):
        raise SPFError("phase shear channel out of range")
    if target in (control, reference):
        raise SPFError("phase shear target must differ from its control and reference")
    if not isinstance(inverse, (bool, np.bool_)):
        raise SPFError("phase shear inverse must be boolean")
    _require_unit_channels(arr, channels)
    theta = {channel: unpack_theta(arr[..., channel]).astype(np.int64) for channel in set(channels)}
    value = theta[control]
    if reference is not None:
        value = (value - theta[reference]) & SPF32_PHASE_MASK
    if table is not None:
        if not 1 <= len(table) <= SPF32_PHASE_MASK + 1:
            raise SPFError("phase shear ROM must have between 1 and 65536 entries")
        rom = np.asarray([_integer(v) & SPF32_PHASE_MASK for v in table], dtype=np.int64)
        delta = rom[(value * len(rom)) >> 16]
    else:
        delta = np.zeros(value.shape, dtype=np.int64)
        for coefficient in reversed(canonical_polynomial(coefficients)):
            delta = (delta * value + coefficient) & SPF32_PHASE_MASK
    output = arr.copy()
    new_phase = (theta[target] + (-delta if inverse else delta)) & SPF32_PHASE_MASK
    output[..., target] = pack_words(unpack_L(arr[..., target]), new_phase.astype(np.uint16))
    return output


Unit = tuple[Character, int, "tuple[int, ...] | None"]


def _rom_table(table: Sequence[int] | None) -> tuple[int, ...] | None:
    if table is None:
        return None
    if isinstance(table, (str, bytes)) or not 1 <= len(table) <= SPF32_PHASE_MASK + 1:
        raise SPFError("phase neuron ROM must have between 1 and 65536 entries")
    return tuple(_integer(v) & SPF32_PHASE_MASK for v in table)


def canonical_phase_neuron(
    units: Sequence[Any],
    coupling: Sequence[Sequence[int]] | npt.ArrayLike,
    targets: Sequence[int],
    n_channels: int,
    *,
    require_gauge_neutral: bool = False,
) -> tuple[tuple[Unit, ...], tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Normalize (character, bias, ROM) units, the integer coupling C and targets.

    Rejects any target that is also a control channel of some unit, so the
    update is a coupling layer with an exact subtractive inverse. A unit's
    gauge charge is the sum of its exponents modulo 65536; neutral rows make
    the pre-activation invariant under a common phase shift.
    """
    if not isinstance(require_gauge_neutral, (bool, np.bool_)):
        raise SPFError("require_gauge_neutral must be boolean")
    n_channels = _integer(n_channels)
    targets = tuple(_integer(t) for t in targets)
    if any(not 0 <= t < n_channels for t in targets) or len(set(targets)) != len(targets):
        raise SPFError("phase neuron targets must be distinct in-range channels")
    rows: list[Unit] = []
    for unit in units:
        if isinstance(unit, dict):
            unit = (unit.get("character", ()), unit.get("bias", 0), unit.get("table"))
        if len(unit) != 3:
            raise SPFError("each phase neuron unit must be (character, bias, table)")
        row = canonical_character(unit[0], n_channels)
        if require_gauge_neutral and sum(e for _, e in row) & SPF32_PHASE_MASK:
            raise SPFError("phase neuron unit has nonzero gauge charge")
        if {channel for channel, _ in row} & set(targets):
            raise SPFError("phase neuron targets must be disjoint from every control channel")
        rows.append((row, _integer(unit[1]) & SPF32_PHASE_MASK, _rom_table(unit[2])))
    matrix = tuple(tuple(_integer(v) & SPF32_PHASE_MASK for v in line) for line in coupling)
    if len(matrix) != len(targets) or any(len(line) != len(rows) for line in matrix):
        raise SPFError("phase neuron coupling must have shape (targets, units)")
    return tuple(rows), matrix, targets


def execute_phase_neuron(
    words: npt.ArrayLike,
    *,
    units: Sequence[Any],
    coupling: Sequence[Sequence[int]] | npt.ArrayLike,
    targets: Sequence[int],
    inverse: bool = False,
    require_gauge_neutral: bool = False,
) -> np.ndarray:
    """Theta[targets] += C @ ROM(A Theta + b) mod 65536, exactly reversible.

    Unit m reads a sparse integer character of its controls, indexes its ROM
    with floor(pre * len(table) / 65536) (``table=None`` is the identity, an
    affine unit), and the integer coupling adds phase ticks to the targets:
    a product of unit phasors, with no coherent sum. Targets are disjoint from
    every control, so inverse=True subtracts the same update. Used channels
    require L=0; spectators are preserved bit-for-bit. A single unit with row
    e_j - e_k and C=[[1]] is ``execute_phase_shear``.
    """
    arr = _word_plane(words)
    if not isinstance(inverse, (bool, np.bool_)):
        raise SPFError("phase neuron inverse must be boolean")
    rows, matrix, targets = canonical_phase_neuron(
        units, coupling, targets, arr.shape[-1], require_gauge_neutral=require_gauge_neutral
    )
    controls = sorted({channel for row, _, _ in rows for channel, _ in row})
    _require_unit_channels(arr, sorted(set(controls) | set(targets)))
    theta = {channel: unpack_theta(arr[..., channel]).astype(np.int64) for channel in controls}
    activations = []
    for row, bias, table in rows:
        pre = np.full(arr.shape[:-1], bias, dtype=np.int64)
        for channel, exponent in row:
            pre = (pre + exponent * theta[channel]) & SPF32_PHASE_MASK
        if table is not None:
            pre = np.asarray(table, dtype=np.int64)[(pre * len(table)) >> 16]
        activations.append(pre)
    output = arr.copy()
    for target, line in zip(targets, matrix, strict=True):
        delta = np.zeros(arr.shape[:-1], dtype=np.int64)
        for coefficient, activation in zip(line, activations, strict=True):
            if coefficient:
                delta = (delta + coefficient * activation) & SPF32_PHASE_MASK
        phase = unpack_theta(arr[..., target]).astype(np.int64)
        phase = (phase - delta if inverse else phase + delta) & SPF32_PHASE_MASK
        output[..., target] = pack_words(unpack_L(arr[..., target]), phase.astype(np.uint16))
    return output


def phase_neuron_plan(
    units: Sequence[Any],
    coupling: Sequence[Sequence[int]] | npt.ArrayLike,
    targets: Sequence[int],
    *,
    n_channels: int,
    inverse: bool = False,
    require_gauge_neutral: bool = False,
    input_name: str = "x",
    output_name: str = "out",
):
    """Emit a one-step ExecutionPlan for a phase neuron, with integer-only counts."""
    from .v2 import ExecutionPlan, ExecutionStep

    if input_name == output_name:
        raise SPFError("distinct input/output names are required")
    if not isinstance(inverse, (bool, np.bool_)):
        raise SPFError("phase neuron inverse must be boolean")
    rows, matrix, targets = canonical_phase_neuron(
        units, coupling, targets, n_channels, require_gauge_neutral=require_gauge_neutral
    )
    nnz = sum(len(row) for row, _, _ in rows)
    coupling_terms = sum(1 for line in matrix for v in line if v)
    step = ExecutionStep(
        "phase_neuron",
        (input_name,),
        (output_name,),
        params={
            "units": rows,
            "coupling": matrix,
            "targets": targets,
            "inverse": bool(inverse),
            "require_gauge_neutral": bool(require_gauge_neutral),
            "n_channels": _integer(n_channels),
            "n_terms": nnz + coupling_terms,
        },
    )
    return ExecutionPlan(
        steps=(step,),
        factorization_metadata={
            "phase_circuit": {
                "integer_only": True,
                "coherent_additions": 0,
                "finalizations": 0,
                "units": len(rows),
                "targets": len(targets),
                "sparse_terms": nnz,
                "coupling_terms": coupling_terms,
                "rom_reads": sum(1 for _, _, table in rows if table is not None),
                "rom_entries": sum(len(table) for _, _, table in rows if table is not None),
                "invertible": True,
            },
        },
    )


def phase_character_plan(
    characters: Sequence[Sequence[Sequence[int]]],
    *,
    n_channels: int,
    biases: Sequence[int] | None = None,
    input_name: str = "x",
    output_name: str = "out",
):
    """Emit an ExecutionPlan sharing identical normalized features, without pruning."""
    from .v2 import ExecutionPlan, ExecutionStep

    n_channels = _integer(n_channels)
    if n_channels < 0 or input_name == output_name:
        raise SPFError("nonnegative channel count and distinct input/output names are required")
    rows = tuple(canonical_character(row, n_channels) for row in characters)
    offsets = (0,) * len(rows) if biases is None else tuple(_integer(v) & SPF32_PHASE_MASK for v in biases)
    if len(rows) != len(offsets):
        raise SPFError("one phase bias is required per character")
    unique: dict[tuple[Character, int], int] = {}
    indices = []
    for key in zip(rows, offsets, strict=True):
        if key not in unique:
            unique[key] = len(unique)
        indices.append(unique[key])
    scratch = f"{output_name}__unique"
    while scratch == input_name:
        scratch += "_"
    shared = len(unique) < len(rows)
    nnz = sum(len(row) for row, _ in unique)
    step = ExecutionStep(
        "phase_character_features",
        (input_name,),
        (scratch if shared else output_name,),
        params={
            "characters": tuple(row for row, _ in unique),
            "biases": tuple(bias for _, bias in unique),
            "n_channels": n_channels,
            "n_terms": nnz,
        },
    )
    steps = (step,)
    if shared:
        steps += (ExecutionStep("gather", (scratch,), (output_name,), params={"indices": indices, "axis": -1}),)
    return ExecutionPlan(
        steps=steps,
        factorization_metadata={
            "phase_circuit": {
                "integer_only": True,
                "coherent_additions": 0,
                "finalizations": 0,
                "requested_features": len(rows),
                "unique_features": len(unique),
                "sparse_terms": nnz,
            },
        },
    )


__all__ = [
    "canonical_phase_neuron",
    "execute_phase_character_features",
    "execute_phase_neuron",
    "execute_phase_shear",
    "phase_character_plan",
    "phase_neuron_plan",
]
