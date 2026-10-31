"""REFERENCE_CARTESIAN diagnostics for the ideal conditioned sequence operator.

This module materializes O(T^2) matrices for small verification cases only.
It does not reproduce the recurrence's intermediate SPF quantization/saturation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .core import SPFError, decode_words
from .sequence_mixer import SequenceMixerParameters, _resets, _validate_parameters, _words


@dataclass(frozen=True)
class SequenceMixerReferenceResult:
    values: np.ndarray
    fallback_conversions: int
    execution_kind: str = "REFERENCE_CARTESIAN"


def sequence_mixing_matrix_reference(
    parameters: SequenceMixerParameters,
    *,
    reset_before: npt.ArrayLike | None = None,
    max_length: int = 256,
) -> SequenceMixerReferenceResult:
    """Materialize the conditional ideal matrix, including segment boundaries.

    Lower M[i,j] = sum_n C[i,n] B[j,n] product_{u=j+1..i} A[u,n].
    Upper M[i,j] = sum_n C_rev[i,n] B_rev[j,n] product_{u=i..j-1} A_rev[u,n].
    The diagonal is D. Parameters are frozen for this diagnostic: this matrix
    is neither the full block Jacobian nor a bit-exact packed operator.
    """
    shape, n = _validate_parameters(parameters)
    length = shape[-1]
    if length > max_length:
        raise SPFError("reference matrix exceeds max_length; use the recurrence for long sequences")
    reset = _resets(reset_before, shape)
    matrix = np.zeros((*shape[:-1], length, length), dtype=np.complex128)
    conversions = 1
    diagonal = decode_words(parameters.diagonal)
    for t in range(length):
        matrix[..., t, t] = diagonal
    for direction, reverse in ((parameters.forward, False), (parameters.backward, True)):
        if direction is None:
            continue
        a, b, c = (decode_words(plane) for plane in (direction.a, direction.b, direction.c))
        conversions += a.size + b.size + c.size
        for j in range(length):
            product = np.ones((*shape[:-1], n), dtype=np.complex128)
            active = np.ones(shape[:-1], dtype=np.bool_)
            targets = range(j - 1, -1, -1) if reverse else range(j + 1, length)
            for i in targets:
                active &= ~reset[..., i + 1 if reverse else i]
                product *= a[..., i, :]
                coefficient = np.sum(c[..., i, :] * product * b[..., j, :], axis=-1)
                matrix[..., i, j] = np.where(active, coefficient, 0)
    return SequenceMixerReferenceResult(matrix, int(conversions))


def apply_sequence_mixer_reference(
    values: npt.ArrayLike,
    parameters: SequenceMixerParameters,
    *,
    reset_before: npt.ArrayLike | None = None,
    max_length: int = 256,
) -> SequenceMixerReferenceResult:
    """Apply the ideal conditional matrix to decoded values (diagnostic only)."""
    words = _words(values, "values", 2)
    shape, _ = _validate_parameters(parameters)
    if words.shape[:-1] != shape:
        raise SPFError("reference values must share parameter batch/time axes")
    matrix = sequence_mixing_matrix_reference(parameters, reset_before=reset_before, max_length=max_length)
    return SequenceMixerReferenceResult(matrix.values @ decode_words(words), matrix.fallback_conversions + words.size)


__all__ = ["SequenceMixerReferenceResult", "apply_sequence_mixer_reference", "sequence_mixing_matrix_reference"]
