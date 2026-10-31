"""Finite factorization of the v3 spectral address primitive.

Every hard address comparison in the default v3 model has the same channelwise
form::

    -relu(Lq + Lk + w) * circular_distance(theta_q, theta_k)

Attention supplies a dynamic key, the lookup MLP supplies a learned table key,
and the readout supplies a vocabulary key.  The phase factor is circulant and
has the exact odd-harmonic expansion implemented by :mod:`circular_kernel`.
On any finite magnitude alphabet the ReLU factor is a Hankel matrix.  Factoring
the two matrices and taking their row-wise Kronecker product turns their
Hadamard product into an ordinary asymmetric dot product.

This module is an analysis/reference Cartesian lifting.  Direct integer
circular distance remains the SPF-native authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .circular_kernel import odd_harmonics, ring_cosine_coefficients


@dataclass(frozen=True)
class AddressFeatureFactorization:
    """Asymmetric features whose matrix product is the direct address score."""

    query_features: np.ndarray
    key_features: np.ndarray
    gate_ranks: tuple[int, ...]

    @property
    def feature_dim(self) -> int:
        return int(self.query_features.shape[-1])

    def scores(self) -> np.ndarray:
        return self.query_features @ self.key_features.T


def _validate_address_inputs(
    query_log_codes: np.ndarray,
    query_phase_codes: np.ndarray,
    key_log_codes: np.ndarray,
    key_phase_codes: np.ndarray,
    gate_bias_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ql = np.asarray(query_log_codes, dtype=np.int64)
    qp = np.asarray(query_phase_codes, dtype=np.uint64)
    kl = np.asarray(key_log_codes, dtype=np.int64)
    kp = np.asarray(key_phase_codes, dtype=np.uint64)
    bias = np.asarray(gate_bias_codes, dtype=np.int64)
    if ql.ndim != 2 or kl.ndim != 2:
        raise ValueError("query/key log codes must be rank-two (items, channels)")
    if qp.shape != ql.shape or kp.shape != kl.shape:
        raise ValueError("phase-code shapes must match their log-code shapes")
    if ql.shape[1] != kl.shape[1] or bias.shape != (ql.shape[1],):
        raise ValueError("query, key, and bias channel counts must agree")
    return ql, qp, kl, kp, bias


def direct_gated_circular_scores(
    query_log_codes: np.ndarray,
    query_phase_codes: np.ndarray,
    key_log_codes: np.ndarray,
    key_phase_codes: np.ndarray,
    gate_bias_codes: np.ndarray,
    *,
    phase_bits: int,
    log_fraction_bits: int = 8,
) -> np.ndarray:
    """Evaluate the shared address score from integer codes.

    The returned value is in model units:

    ``-sum_c max(Lq_code + Lk_code + w_code, 0) * delta_code / (Q * R)``.

    The small-ring analysis path uses int64 accumulation and rejects geometries
    whose conservative bound could overflow it.
    """

    ql, qp, kl, kp, bias = _validate_address_inputs(
        query_log_codes,
        query_phase_codes,
        key_log_codes,
        key_phase_codes,
        gate_bias_codes,
    )
    if phase_bits < 2 or phase_bits > 32:
        raise ValueError("phase_bits must be in [2, 32]")
    if log_fraction_bits < 0 or log_fraction_bits > 30:
        raise ValueError("log_fraction_bits must be in [0, 30]")
    ring = 1 << phase_bits
    gates = np.maximum(ql[:, None, :] + kl[None, :, :] + bias[None, None, :], 0)
    delta = (qp[:, None, :] - kp[None, :, :]) & np.uint64(ring - 1)
    distance = np.minimum(delta, np.uint64(ring) - delta).astype(np.int64)
    max_term = int(gates.max(initial=0)) * (ring // 2) * ql.shape[1]
    if max_term > np.iinfo(np.int64).max:
        raise OverflowError("address score may exceed int64; use a wider declared accumulator")
    numerator = (gates * distance).sum(axis=-1, dtype=np.int64)
    return -numerator.astype(np.float64) / float((1 << log_fraction_bits) * ring)


def complete_ring_features(codes: np.ndarray, *, phase_bits: int) -> np.ndarray:
    """Return the complete real feature map for the centered circular kernel.

    This is intentionally limited to small rings.  The exact v3 ring has 32,768
    real coordinates per phase channel, and SPF-64 has 2**31; direct distance is
    the sensible exact implementation at those widths.
    """

    if phase_bits < 2 or phase_bits > 12:
        raise ValueError("complete feature materialization is limited to phase_bits in [2, 12]")
    ring = 1 << phase_bits
    harmonics = odd_harmonics(ring // 4)
    coeff = ring_cosine_coefficients(harmonics, phase_bits)
    ticks = (np.asarray(codes, dtype=np.uint64)[..., None] * harmonics) & np.uint64(ring - 1)
    angle = ticks.astype(np.float64) * (2.0 * np.pi / float(ring))
    root = np.sqrt(coeff)
    return np.concatenate((np.cos(angle) * root, np.sin(angle) * root), axis=-1)


def gate_hankel_matrix(
    query_codes: np.ndarray,
    key_codes: np.ndarray,
    bias_code: int,
) -> np.ndarray:
    """Return ``relu(query + key + bias)`` on two finite code alphabets."""

    query = np.asarray(query_codes, dtype=np.int64)
    key = np.asarray(key_codes, dtype=np.int64)
    if query.ndim != 1 or key.ndim != 1:
        raise ValueError("query_codes and key_codes must be one-dimensional")
    return np.maximum(query[:, None] + key[None, :] + int(bias_code), 0)


def factorized_gated_circular_features(
    query_log_codes: np.ndarray,
    query_phase_codes: np.ndarray,
    key_log_codes: np.ndarray,
    key_phase_codes: np.ndarray,
    gate_bias_codes: np.ndarray,
    *,
    phase_bits: int,
    log_fraction_bits: int = 8,
) -> AddressFeatureFactorization:
    """Factor a finite address panel into asymmetric Cartesian features.

    For channel ``c``, let ``G_c`` be the finite ReLU gate matrix and let
    ``K_c`` be the centered circular kernel.  If ``G_c = A_c B_c.T`` and
    ``K_c = P_c Q_c.T``, then

    ``G_c * K_c = (A_c row-kron P_c) (B_c row-kron Q_c).T``.

    The additional asymmetric gate features subtract ``G_c / 4``, leaving
    exactly ``-G_c * circular_distance / R``.  SVD is used only to expose the
    finite magnitude rank; no singular value above the numerical matrix-rank
    tolerance is discarded.
    """

    ql, qp, kl, kp, bias = _validate_address_inputs(
        query_log_codes,
        query_phase_codes,
        key_log_codes,
        key_phase_codes,
        gate_bias_codes,
    )
    q_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []
    ranks: list[int] = []
    inv_root_q = 1.0 / np.sqrt(float(1 << log_fraction_bits))

    for channel in range(ql.shape[1]):
        gate = gate_hankel_matrix(ql[:, channel], kl[:, channel], int(bias[channel])).astype(np.float64)
        u, singular, vh = np.linalg.svd(gate, full_matrices=False)
        if singular.size == 0 or singular[0] == 0.0:
            ranks.append(0)
            continue
        tolerance = np.finfo(np.float64).eps * max(gate.shape) * singular[0]
        rank = int(np.count_nonzero(singular > tolerance))
        ranks.append(rank)
        root_s = np.sqrt(singular[:rank])
        q_gate = u[:, :rank] * root_s
        k_gate = vh[:rank].T * root_s

        q_phase = complete_ring_features(qp[:, channel], phase_bits=phase_bits)
        k_phase = complete_ring_features(kp[:, channel], phase_bits=phase_bits)
        q_product = np.einsum("qr,qf->qrf", q_gate, q_phase).reshape(ql.shape[0], -1)
        k_product = np.einsum("kr,kf->krf", k_gate, k_phase).reshape(kl.shape[0], -1)
        q_parts.extend((q_product * inv_root_q, q_gate * (0.5 * inv_root_q)))
        k_parts.extend((k_product * inv_root_q, -k_gate * (0.5 * inv_root_q)))

    if not q_parts:
        return AddressFeatureFactorization(
            query_features=np.zeros((ql.shape[0], 0), dtype=np.float64),
            key_features=np.zeros((kl.shape[0], 0), dtype=np.float64),
            gate_ranks=tuple(ranks),
        )
    return AddressFeatureFactorization(
        query_features=np.concatenate(q_parts, axis=-1),
        key_features=np.concatenate(k_parts, axis=-1),
        gate_ranks=tuple(ranks),
    )
