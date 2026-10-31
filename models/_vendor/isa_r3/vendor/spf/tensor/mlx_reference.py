"""Opt-in MLX reference kernels for SPF tensor experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from spf.mlx_support import require_mlx, to_mlx_f32, to_numpy
from spf.tensor.core import SPF32_PHASE_MOD, decode_words, encode_words

_TWO_PI = np.float32(2.0 * np.pi)
_PHASE_SCALE = np.float32(float(SPF32_PHASE_MOD) / (2.0 * np.pi))


def _atan2(y: Any, x: Any) -> Any:
    mlx = require_mlx()
    if hasattr(mlx, "atan2"):
        return mlx.atan2(y, x)
    return mlx.arctan2(y, x)


def _round(x: Any) -> Any:
    mlx = require_mlx()
    if hasattr(mlx, "round"):
        return mlx.round(x)
    return mlx.floor(x + np.float32(0.5))


def _phase_codes(z_real: Any, z_imag: Any) -> Any:
    theta = _atan2(z_imag, z_real) % _TWO_PI
    return _round(theta * _PHASE_SCALE) % np.float32(SPF32_PHASE_MOD)


def phase_concordance_scores_mlx(
    H_q: Any,
    H_k: Any,
    mu: Any,
    sd: Any,
    *,
    return_numpy: bool = True,
) -> Any:
    """MLX twin of spf.tensor.phase_route.phase_concordance_scores."""
    mlx = require_mlx()
    q = (to_mlx_f32(H_q) - to_mlx_f32(mu)) / to_mlx_f32(sd)
    k = (to_mlx_f32(H_k) - to_mlx_f32(mu)) / to_mlx_f32(sd)
    if q.ndim != 2 or k.ndim != 2 or int(q.shape[1]) != int(k.shape[1]):
        raise ValueError(f"need H_q (T,d) and H_k (M,d) with matching d, got {tuple(q.shape)} and {tuple(k.shape)}")
    half = int(q.shape[1]) // 2
    if half * 2 != int(q.shape[1]):
        raise ValueError(f"phase concordance needs even d, got {int(q.shape[1])}")

    th_q = _phase_codes(q[:, :half], q[:, half:])
    th_k = _phase_codes(k[:, :half], k[:, half:])
    dtheta = (mlx.expand_dims(th_q, 1) - mlx.expand_dims(th_k, 0)) % np.float32(SPF32_PHASE_MOD)
    signed = mlx.where(dtheta <= np.float32(0x7FFF), dtheta, dtheta - np.float32(SPF32_PHASE_MOD))
    scores = mlx.sum(mlx.cos(signed * np.float32(2.0 * np.pi / float(SPF32_PHASE_MOD))), axis=-1)
    return to_numpy(scores) if return_numpy else scores


def phase_address_value_mix_mlx(
    probabilities: Any,
    H: Any,
    H_mem: Any,
    H_mem_out: Any,
    *,
    return_numpy: bool = True,
) -> Any:
    """Accelerate only the plain real value-mix half of phase_address."""
    p = to_mlx_f32(probabilities)
    h = to_mlx_f32(H)
    value = to_mlx_f32(H_mem_out) - to_mlx_f32(H_mem)
    out = h + p @ value
    return to_numpy(out) if return_numpy else out


def has_mlx_fft() -> bool:
    mlx = require_mlx()
    return hasattr(mlx, "fft") and hasattr(mlx.fft, "fft") and hasattr(mlx.fft, "ifft")


def block_circulant_apply_mlx(
    vector_words: Any,
    spectrum_words: Any,
    frequency_count: int,
    out_blocks: int,
    in_blocks: int,
    *,
    return_words: bool = True,
) -> Any:
    """MLX twin of ExecutionPlan._execute_block_circulant_apply."""
    mlx = require_mlx()
    if not has_mlx_fft():
        raise RuntimeError("MLX FFT support is unavailable in this mlx.core build")
    if frequency_count <= 0 or out_blocks <= 0 or in_blocks <= 0:
        raise ValueError("block_circulant_apply dimensions must be positive")
    vector = decode_words(np.asarray(vector_words, dtype=np.uint32))
    spectrum = decode_words(np.asarray(spectrum_words, dtype=np.uint32))
    if vector.size != frequency_count * in_blocks:
        raise ValueError("block_circulant_apply vector size mismatch")
    if spectrum.size != frequency_count * out_blocks * in_blocks:
        raise ValueError("block_circulant_apply spectrum size mismatch")

    x = mlx.array(vector.reshape(frequency_count, in_blocks).astype(np.complex64))
    spec = mlx.array(spectrum.reshape(frequency_count, out_blocks, in_blocks).astype(np.complex64))
    X = mlx.fft.fft(x, axis=0)
    Y = mlx.squeeze(spec @ mlx.expand_dims(X, -1), axis=-1)
    y = mlx.fft.ifft(Y, axis=0)
    dense = to_numpy(y).reshape(frequency_count * out_blocks)
    return encode_words(dense) if return_words else dense
