"""Circular phase distance as an exact finite-ring Fourier dot product.

The authoritative score remains integer circular distance on an SPF phase ring.
This module exposes a Cartesian Fourier factorization for analysis and bounded
approximation.  It never decodes an SPF carrier to complex values.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from spf.tensor import SPFBlockV3


def make_spf64_phase_block(phase_codes: np.ndarray) -> SPFBlockV3:
    """Put an arbitrary uint32 phase plane in a first-class SPF-64 block.

    The proof-of-concept uses exact outliers for arbitrary random codes.  This is
    deliberately a mechanics carrier, not a storage-efficiency claim.
    """

    from spf.scalar._constants import SPF64
    from spf.tensor import (
        BaseModelKind,
        BaseModelRef,
        Representation,
        RepresentationRef,
        SPFBlockV3,
        pack_words,
    )

    theta = np.asarray(phase_codes, dtype=np.uint32)
    logmag = np.zeros(theta.shape, dtype=np.int32)
    words = pack_words(logmag, theta, SPF64).reshape(-1)
    return SPFBlockV3(
        dims=theta.shape,
        representation=RepresentationRef(int(Representation.POLAR_SPF64)),
        base=BaseModelRef(int(BaseModelKind.SEPARABLE_PHASE), {}),
        outliers={i: int(word) for i, word in enumerate(words.tolist())},
    )


def spf64_phase_codes(block: SPFBlockV3) -> np.ndarray:
    """Recover the integer phase plane from an SPF-64 block without complex decode."""

    from spf.tensor import Representation

    if block.representation.rep_id != int(Representation.POLAR_SPF64):
        raise ValueError("expected a POLAR_SPF64 block")
    _, theta = block.reconstruct()
    return np.asarray(theta, dtype=np.uint32)


def exact_circular_costs(
    query_block: SPFBlockV3,
    key_block: SPFBlockV3,
    gate_codes: np.ndarray,
) -> np.ndarray:
    """Return exact integer costs ``sum_c gate[c] * circular_distance(q[c], k[c])``.

    Queries have shape ``(B, C)``, keys ``(B, M, C)``, and nonnegative gate
    codes ``(B, C)``.  The established arbitrary-width signed phase-delta
    primitive supplies each distance; accumulation uses Python integers and is
    range-checked before conversion to int64.
    """

    from spf.fft.fb7_fields import phase_delta_signed
    from spf.scalar._constants import SPF64

    queries = spf64_phase_codes(query_block)
    keys = spf64_phase_codes(key_block)
    gates = np.asarray(gate_codes, dtype=np.int64)
    if queries.ndim != 2 or keys.ndim != 3:
        raise ValueError("queries must be (B,C) and keys must be (B,M,C)")
    if keys.shape[0] != queries.shape[0] or keys.shape[2] != queries.shape[1]:
        raise ValueError("query/key shapes do not agree")
    if gates.shape != queries.shape or np.any(gates < 0):
        raise ValueError("gate_codes must be nonnegative with shape (B,C)")

    batch, n_keys, channels = keys.shape
    costs = np.empty((batch, n_keys), dtype=np.int64)
    i64_max = np.iinfo(np.int64).max
    for b in range(batch):
        for m in range(n_keys):
            total = 0
            for c in range(channels):
                delta = phase_delta_signed(int(keys[b, m, c]), int(queries[b, c]), SPF64.theta_bits)
                total += int(gates[b, c]) * abs(delta)
            if total > i64_max:
                raise OverflowError("circular score exceeds int64; declare a wider accumulator")
            costs[b, m] = total
    return costs


def odd_harmonics(count: int) -> np.ndarray:
    """Return the first ``count`` positive odd harmonics."""

    if count <= 0:
        raise ValueError("count must be positive")
    return 2 * np.arange(count, dtype=np.uint64) + 1


def ring_cosine_coefficients(harmonics: np.ndarray, phase_bits: int = 32) -> np.ndarray:
    """Real-pair coefficients for the exact centered finite-ring kernel.

    For ``R = 2**phase_bits`` and positive odd ``n < R/2``, the coefficient of
    ``cos(2*pi*n*(q-k)/R)`` in ``1/4 - delta_R(q,k)/R`` is
    ``2 / (R**2 * sin(pi*n/R)**2)``.
    """

    if phase_bits < 2 or phase_bits > 32:
        raise ValueError("this float64 feature adapter supports phase_bits in [2, 32]")
    h = np.asarray(harmonics, dtype=np.uint64)
    ring = 1 << phase_bits
    if h.ndim != 1 or np.any((h & 1) == 0) or np.any(h >= ring // 2):
        raise ValueError("harmonics must be a 1-D array of positive odd values below R/2")
    angle = np.pi * h.astype(np.float64) / float(ring)
    return 2.0 / (float(ring) ** 2 * np.sin(angle) ** 2)


def ring_tail_bound(n_harmonics: int, phase_bits: int = 32) -> float:
    """Uniform per-unit-weight error bound for truncating the exact ring expansion."""

    coeff = ring_cosine_coefficients(odd_harmonics(n_harmonics), phase_bits)
    return max(0.0, 0.25 - float(coeff.sum()))


def _harmonic_angles(codes: np.ndarray, harmonics: np.ndarray, phase_bits: int) -> np.ndarray:
    """Raise unit phases to integer harmonics by exact ring multiplication."""

    if phase_bits > 32:
        raise ValueError("uint64 vector lowering supports at most a 32-bit phase ring")
    ring = 1 << phase_bits
    mask = np.uint64(ring - 1)
    ticks = (np.asarray(codes, dtype=np.uint64)[..., None] * harmonics) & mask
    return ticks.astype(np.float64) * (2.0 * np.pi / float(ring))


def truncated_dot_scores(
    query_codes: np.ndarray,
    key_codes: np.ndarray,
    channel_weights: np.ndarray,
    n_harmonics: int,
    *,
    phase_bits: int = 32,
) -> np.ndarray:
    """Approximate negative circular-distance scores with Fourier dot products.

    Inputs have shapes ``query=(B,C)``, ``keys=(B,M,C)``, and
    ``channel_weights=(B,C)``.  The returned score approximates
    ``-sum_c weight[c] * delta_R(q[c], k[c]) / R``.  Query-dependent weights
    live entirely on the query feature side, so key features can be cached.
    """

    queries = np.asarray(query_codes, dtype=np.uint64)
    keys = np.asarray(key_codes, dtype=np.uint64)
    weights = np.asarray(channel_weights, dtype=np.float64)
    if queries.ndim != 2 or keys.ndim != 3:
        raise ValueError("queries must be (B,C) and keys must be (B,M,C)")
    if keys.shape[0] != queries.shape[0] or keys.shape[2] != queries.shape[1]:
        raise ValueError("query/key shapes do not agree")
    if weights.shape != queries.shape or np.any(weights < 0.0):
        raise ValueError("channel_weights must be nonnegative with shape (B,C)")

    harmonics = odd_harmonics(n_harmonics)
    coeff = ring_cosine_coefficients(harmonics, phase_bits)
    root = np.sqrt(coeff)
    q_angle = _harmonic_angles(queries, harmonics, phase_bits)
    k_angle = _harmonic_angles(keys, harmonics, phase_bits)

    q_scale = weights[..., None] * root
    k_scale = root
    q_cos = np.cos(q_angle) * q_scale
    q_sin = np.sin(q_angle) * q_scale
    k_cos = np.cos(k_angle) * k_scale
    k_sin = np.sin(k_angle) * k_scale
    centered = np.einsum("bch,bmch->bm", q_cos, k_cos) + np.einsum("bch,bmch->bm", q_sin, k_sin)
    return centered - 0.25 * weights.sum(axis=-1, keepdims=True)


def continuous_tail_asymptotic(n_harmonics: int) -> float:
    """Simple large-ring upper bound ``1 / (2*pi**2*K)`` for interpretation."""

    if n_harmonics <= 0:
        raise ValueError("n_harmonics must be positive")
    return 1.0 / (2.0 * math.pi**2 * n_harmonics)
