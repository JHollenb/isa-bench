"""Nearest-circulant projection for dense and block weight matrices.

For a circulant with the §7.1 convention ``C[i, j] = c[(i − j) mod n]``, the
first-column ``c*`` minimizing ``||W − C||_F`` is the average along each
wraparound diagonal:

    c*[k] = (1/n) Σᵢ W[i, (i − k) mod n]

(Spec §7.4's pseudocode averages ``W[j, (j + k) mod n]``; that applies to the
first-*row* convention ``C[i, j] = c[(j − i) mod n]`` and returns ``c``
time-reversed for the §7.1 convention. The spec's own T3.1 test uses the
§7.1 convention, so we use the formula that's consistent with it.)

For block-circulant, each ``k×k`` block is projected independently with the
same rule.
"""

from __future__ import annotations

import numpy as np


def project_to_circulant(W: np.ndarray) -> np.ndarray:
    """Return ``FFT(c*)`` where ``c*`` is the first column of the nearest
    circulant matrix to ``W`` in Frobenius norm (spec §7.4).

    Parameters
    ----------
    W : ndarray of shape (n, n), real-valued.

    Returns
    -------
    c_hat : complex64 array of length n — the FDWM-C stored weights.
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be square, got shape {W.shape}")
    n = W.shape[0]
    W_f = W.astype(np.float32)
    c = np.empty(n, dtype=np.float32)
    i = np.arange(n)
    for k in range(n):
        c[k] = W_f[i, (i - k) % n].mean()
    return np.fft.fft(c).astype(np.complex64)


def project_to_block_circulant(W: np.ndarray, *, k: int) -> np.ndarray:
    """Return the block-wise rfft spectrum for the nearest block-circulant
    approximation of ``W`` (spec §8).

    ``W`` may be square ``(n, n)`` or rectangular ``(m, n)``; ``k`` must
    divide both dimensions. The square case is the spec's canonical FDWM-BC
    encoding; the rectangular case is needed for Phase 2 weight matrices
    like GPT-2's MLP projections where the dense weight is ``(d, 4d)`` and
    block size ``k`` still cleanly divides both.

    Parameters
    ----------
    W : (m, n) real-valued, with ``k`` dividing both ``m`` and ``n``.

    Returns
    -------
    spectrum : complex64 array of shape ``(m/k, n/k, k/2 + 1)``.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2-D, got shape {W.shape}")
    m, n = W.shape
    if m % k != 0:
        raise ValueError(f"k={k} must divide m={m}")
    if n % k != 0:
        raise ValueError(f"k={k} must divide n={n}")
    m_blocks, n_blocks = m // k, n // k
    spectrum = np.empty((m_blocks, n_blocks, k // 2 + 1), dtype=np.complex64)
    W_f = W.astype(np.float32)
    for i in range(m_blocks):
        for j in range(n_blocks):
            block = W_f[i * k : (i + 1) * k, j * k : (j + 1) * k]
            c = _block_nearest_circulant_column(block)
            spectrum[i, j] = np.fft.rfft(c).astype(np.complex64)
    return spectrum


def reconstruct_block_circulant(spectrum: np.ndarray, *, k: int) -> np.ndarray:
    """Reconstruct the dense ``(m, n)`` block-circulant matrix from its rfft
    spectrum (inverse of :func:`project_to_block_circulant`).

    Every ``k×k`` block is a circulant matrix whose first column is
    ``irfft(spectrum[i, j], n=k)`` and whose ``(i, j)``-th entry follows
    the spec §7.1 convention ``C[i, j] = c[(i − j) mod k]``.

    Used by Phase 2 quality evals (Task 2.2) to materialize the nearest
    block-circulant approximation and measure ``||W − W_bc||_F``.
    """
    if spectrum.ndim != 3:
        raise ValueError(f"spectrum must be 3-D, got shape {spectrum.shape}")
    m_blocks, n_blocks, bins = spectrum.shape
    if bins != k // 2 + 1:
        raise ValueError(f"bins={bins} != k/2 + 1 = {k // 2 + 1} for k={k}")
    m, n = m_blocks * k, n_blocks * k
    W = np.empty((m, n), dtype=np.float32)
    idx = np.arange(k)
    row = (idx[:, None] - idx[None, :]) % k  # (k, k) circulant indexing
    for i in range(m_blocks):
        for j in range(n_blocks):
            c = np.fft.irfft(spectrum[i, j], n=k).astype(np.float32)
            W[i * k : (i + 1) * k, j * k : (j + 1) * k] = c[row]
    return W


def _block_nearest_circulant_column(block: np.ndarray) -> np.ndarray:
    """First column of the nearest circulant to a ``k×k`` block."""
    k = block.shape[0]
    c = np.empty(k, dtype=np.float32)
    i = np.arange(k)
    for kk in range(k):
        c[kk] = block[i, (i - kk) % k].mean()
    return c
