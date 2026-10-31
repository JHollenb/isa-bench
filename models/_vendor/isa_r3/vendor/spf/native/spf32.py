"""C-backed SPF-32 scalar ops.

Mirrors the public surface of :mod:`spf.scalar.spf32`. Pure-Python code
should prefer that module; this one exists so benchmarks and cross-
validation tests can exercise the compiled C path. Raises at call time if
``libspf32`` isn't available — import alone is safe.

Array inputs are dispatched through the vectorized ``*_array`` C entry
points: one Python→C transition per call instead of per element, so
benchmarks don't pay ~200 ns of ctypes overhead for every scalar op.
"""

from __future__ import annotations

import ctypes
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from spf.native._libspf32 import load_library

__all__ = [
    "abs2",
    "add",
    "add_cotransform",
    "add_poly7",
    "add_sumprod",
    "aoa_lut",
    "aoa_pair_phase",
    "beamform_fused",
    "butterfly",
    "candidate_fft_bins_fb7",
    "channel_estimate_ls",
    "channel_inverse_guard",
    "conj",
    "constellation_lut_demap_fused",
    "constellation_lut_llr_fused",
    "decode",
    "decode_fast",
    "differential_preamble_match",
    "div",
    "doppler_phase_diff_estimate",
    "dot",
    "dot_nokahan",
    "encode",
    "encode_fast",
    "fast_log2",
    "fb7_reduce_tree",
    "gain_phase_apply",
    "log_os_cfar",
    "log_power",
    "logpower_threshold",
    "mul",
    "neg",
    "ofdm_cpe_estimate_from_pilots",
    "ofdm_mpsk_eq_demod_fused",
    "ofdm_qpsk_eq_demod_fused",
    "ofdm_qpsk_rx_hotpath",
    "ofdm_zf_equalize",
    "pa_pe",
    "pa_pe_slice",
    "pa_tone",
    "pairwise_phase_diff",
    "phase_diff",
    "phase_ramp_apply",
    "phase_ramp_nd_apply",
    "phase_rotate_const",
    "phase_slope_estimate",
    "qam16_logproj_eq_demod_fused",
    "qpsk_phase_demod",
    "qpsk_phase_demod_conf",
    "runtime_aoa_pair_lut",
    "runtime_radar_sparse_doppler",
    "runtime_rf_native_ingress",
    "spectral_matched_filter_fb7",
    "sub",
    "sub_poly7",
    "theta_sub",
    "twiddle",
]


def _u32_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))


def _u16_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


def _u8_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def _i8_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))


def _i16_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))


def _i32_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def _f64_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def _f32_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


def _i64_ptr(arr: np.ndarray) -> Any:
    return arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))


def _prep_word(x: npt.ArrayLike) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x), dtype=np.uint32)


def _prep_u8_optional(x: npt.ArrayLike | None, n: int) -> np.ndarray | None:
    if x is None:
        return None
    arr = np.ascontiguousarray(np.asarray(x), dtype=np.uint8).ravel()
    if arr.size != n:
        raise ValueError(f"uint8 mask length {arr.size} != expected length {n}")
    return arr


def _u8_optional_ptr(arr: np.ndarray | None) -> Any:
    if arr is None:
        return ctypes.POINTER(ctypes.c_uint8)()
    return _u8_ptr(arr)


def _constellation_lut_expected_bins(
    l_shift: int, theta_shift: int, l_min: int, l_max: int
) -> int:
    if l_shift < 0 or theta_shift < 0:
        raise ValueError(
            f"l_shift and theta_shift must be non-negative, got {l_shift}, {theta_shift}"
        )
    if l_max < l_min:
        raise ValueError(f"l_max {l_max} must be >= l_min {l_min}")
    ls = min(int(l_shift), 30)
    ts = min(int(theta_shift), 15)
    n_l_bins = ((int(l_max) - int(l_min)) >> ls) + 1
    n_theta_bins = 1 << (16 - ts)
    return n_l_bins * n_theta_bins


def _unary_word(entry: str, w: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    lib = load_library()
    arr = _prep_word(w)
    out = np.empty_like(arr)
    getattr(lib, entry)(_u32_ptr(arr), _u32_ptr(out), ctypes.c_size_t(arr.size))
    return out


def _binary_word_prebroadcast(
    entry: str, a_arr: np.ndarray, b_arr: np.ndarray
) -> npt.NDArray[np.uint32]:
    """Call a binary C entry point on arrays that are already conformed (same
    shape, contiguous, dtype=uint32). Skips broadcast + ascontiguousarray."""
    lib = load_library()
    out = np.empty_like(a_arr)
    getattr(lib, entry)(
        _u32_ptr(a_arr), _u32_ptr(b_arr), _u32_ptr(out), ctypes.c_size_t(a_arr.size)
    )
    return out


def _binary_word(
    entry: str, a: npt.ArrayLike, b: npt.ArrayLike
) -> npt.NDArray[np.uint32]:
    a_arr, b_arr = np.broadcast_arrays(
        _prep_word(a), _prep_word(b)
    )
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    return _binary_word_prebroadcast(entry, a_arr, b_arr)


def fast_log2(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """C port of :func:`spf.scalar._fast_log.fast_log2`."""
    lib = load_library()
    arr = np.ascontiguousarray(np.asarray(x), dtype=np.float64)
    out = np.empty_like(arr)
    lib.spf32_fast_log2_array(_f64_ptr(arr), _f64_ptr(out), ctypes.c_size_t(arr.size))
    return out


def encode(z: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """C port of :func:`spf.scalar.spf32.encode`."""
    lib = load_library()
    arr = np.ascontiguousarray(np.asarray(z), dtype=np.complex128)
    # complex128 stores interleaved (re, im); view as float64 for the C loop.
    pairs = arr.view(np.float64).reshape(-1)
    out = np.empty(arr.shape, dtype=np.uint32)
    lib.spf32_encode_array(_f64_ptr(pairs), _u32_ptr(out), ctypes.c_size_t(arr.size))
    return out


def decode(w: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """C port of :func:`spf.scalar.spf32.decode`."""
    lib = load_library()
    arr = _prep_word(w)
    out = np.empty(arr.shape, dtype=np.complex128)
    pairs = out.view(np.float64).reshape(-1)
    lib.spf32_decode_array(_u32_ptr(arr), _f64_ptr(pairs), ctypes.c_size_t(arr.size))
    return out


def encode_fast(z: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Fast Cartesian→SPF using shared poly7/BAM helpers (matches FB7 path).

    Accuracy envelope vs :func:`encode`: max |ΔL| ≤ 1 tick, max |ΔΘ| ≈ 0.088°
    on a (-256, 256)² Cartesian grid. Use only at the Cartesian boundary —
    never inside an SPF-native pipeline.
    """
    lib = load_library()
    arr = np.ascontiguousarray(np.asarray(z), dtype=np.complex128)
    pairs = arr.view(np.float64).reshape(-1)
    out = np.empty(arr.shape, dtype=np.uint32)
    lib.spf32_encode_fast_array(_f64_ptr(pairs), _u32_ptr(out), ctypes.c_size_t(arr.size))
    return out


def decode_fast(w: npt.ArrayLike) -> npt.NDArray[np.complex128]:
    """Fast SPF→Cartesian using shared poly7/BAM helpers.

    Drift vs :func:`decode` ≈ 0.028%; roundtrip ``decode_fast ∘ encode_fast``
    EVM ≈ 0.136%.
    """
    lib = load_library()
    arr = _prep_word(w)
    out = np.empty(arr.shape, dtype=np.complex128)
    pairs = out.view(np.float64).reshape(-1)
    lib.spf32_decode_fast_array(_u32_ptr(arr), _f64_ptr(pairs), ctypes.c_size_t(arr.size))
    return out


def mul(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _binary_word("spf32_mul_array", a, b)


def div(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _binary_word("spf32_div_array", a, b)


def conj(a: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _unary_word("spf32_conj_array", a)


def abs2(a: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _unary_word("spf32_abs2_array", a)


def neg(a: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _unary_word("spf32_neg_array", a)


def twiddle(k: npt.ArrayLike, n: int) -> npt.NDArray[np.uint32]:
    if n <= 0:
        raise ValueError(f"N must be positive, got {n}")
    lib = load_library()
    k_arr = np.ascontiguousarray(np.asarray(k), dtype=np.int64)
    out = np.empty(k_arr.shape, dtype=np.uint32)
    lib.spf32_twiddle_array(
        _i64_ptr(k_arr), ctypes.c_int32(int(n)), _u32_ptr(out), ctypes.c_size_t(k_arr.size)
    )
    return out


def add(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _binary_word("spf32_add_array", a, b)


def add_sumprod(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Sum-to-product ADD variant (Task 3.4). Not spec-normative — bench only."""
    return _binary_word("spf32_add_sumprod_array", a, b)


def add_cotransform(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Cotransform-wedge ADD variant (Task 3.4). Not spec-normative — bench only."""
    return _binary_word("spf32_add_cotransform_array", a, b)


def sub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    return _binary_word("spf32_sub_array", a, b)


def theta_sub(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint32]:
    """Phase-only subtraction: L=0, Theta=(Theta_a-Theta_b) mod 2^16."""
    return _binary_word("spf32_theta_sub_array", a, b)


def ofdm_zf_equalize(
    y: npt.ArrayLike, h_inv: npt.ArrayLike
) -> npt.NDArray[np.uint32]:
    """SPF-native OFDM zero-forcing equalizer: ``X_hat[k] = Y[k] * H_inv[k]``."""
    return _binary_word("spf32_ofdm_zf_equalize_array", y, h_inv)


def phase_diff(x: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Adjacent modular theta differences: ``theta(x[i+1]) - theta(x[i])``."""
    lib = load_library()
    arr = _prep_word(x).ravel()
    out = np.empty(max(arr.size - 1, 0), dtype=np.uint16)
    if out.size > 0:
        lib.spf32_phase_diff_array(
            _u32_ptr(arr),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.c_size_t(arr.size),
        )
    return out


def phase_rotate_const(x: npt.ArrayLike, dtheta: int) -> npt.NDArray[np.uint32]:
    """Rotate SPF words by a constant modular theta offset."""
    lib = load_library()
    arr = _prep_word(x)
    out = np.empty_like(arr)
    lib.spf32_phase_rotate_const_array(
        _u32_ptr(arr.ravel()),
        ctypes.c_uint16(int(dtheta) & 0xFFFF),
        _u32_ptr(out.ravel()),
        ctypes.c_size_t(arr.size),
    )
    return out


def phase_ramp_apply(
    x: npt.ArrayLike, theta0: int, dtheta: int
) -> npt.NDArray[np.uint32]:
    """Apply ``theta += theta0 + i * dtheta`` to an SPF word array."""
    lib = load_library()
    arr = _prep_word(x)
    out = np.empty_like(arr)
    lib.spf32_phase_ramp_apply_array(
        _u32_ptr(arr.ravel()),
        ctypes.c_uint16(int(theta0) & 0xFFFF),
        ctypes.c_uint16(int(dtheta) & 0xFFFF),
        _u32_ptr(out.ravel()),
        ctypes.c_size_t(arr.size),
    )
    return out


def phase_ramp_nd_apply(
    x: npt.ArrayLike,
    ramp_R: Sequence[Sequence[int]],
    ramp_b: Sequence[int],
    *,
    quadratic_ramp: Sequence[int] | None = None,
    cubic_ramp: Sequence[int] | None = None,
    nonlinear: bool = False,
    phase_gradient: Sequence[int] | None = None,
    mixed_ramp: Sequence[int] | None = None,
    cubic_mixed: Sequence[int] | None = None,
) -> npt.NDArray[np.uint32]:
    """Apply an N-d separable phase ramp (linear + optional quad/cubic/mixed fold).

    The native datapath for the ``spf_ramp`` PL kernel. ``x`` is an order-``N``
    SPF word array; its rank fixes the ramp order. ``ramp_R`` is ``N x N``
    row-major and ``ramp_b`` length ``N``. ``phase_gradient`` (length ``N``,
    ABI v23) is a linear term folded unconditionally, like the ``ramp_R``
    column sums. With ``nonlinear=True`` the length-``N`` ``quadratic_ramp`` /
    ``cubic_ramp`` terms and the ``N(N-1)/2`` ``mixed_ramp`` off-diagonal terms
    (``spf.tensor.v2.mixed_ramp_pairs`` order) are folded too, as are the two
    AlgRef v0.2 §13.6 mixed cubics ``cubic_mixed`` — ``q0^2 q1`` and ``q0 q1^2``
    — which exist only at order 2 (ABI v24). Bit-identical to the host
    executor's ``_execute_ramp``.
    """
    lib = load_library()
    arr = _prep_word(x)
    order = arr.ndim
    out = np.empty_like(arr)
    dims = np.ascontiguousarray(arr.shape, dtype=np.uint32)
    rr = np.ascontiguousarray(
        np.asarray(ramp_R, dtype=np.uint16).reshape(order, order)
        if order
        else np.zeros((0,), dtype=np.uint16)
    )
    rb = np.ascontiguousarray(np.asarray(ramp_b, dtype=np.uint16).reshape(order))
    quad_ptr = None
    cube_ptr = None
    grad_ptr = None
    mixed_ptr = None
    cubic_mixed_ptr = None
    if nonlinear:
        if quadratic_ramp is not None:
            quad = np.ascontiguousarray(
                np.asarray(quadratic_ramp, dtype=np.uint16).reshape(order)
            )
            quad_ptr = _u16_ptr(quad)
        if cubic_ramp is not None:
            cube = np.ascontiguousarray(
                np.asarray(cubic_ramp, dtype=np.uint16).reshape(order)
            )
            cube_ptr = _u16_ptr(cube)
        if mixed_ramp is not None:
            n_pairs = order * (order - 1) // 2
            mixed_vals = [int(v) & 0xFFFF for v in mixed_ramp]
            if len(mixed_vals) > n_pairs:
                raise ValueError(f"mixed_ramp has {len(mixed_vals)} entries, order {order} allows {n_pairs}")
            mixed_vals += [0] * (n_pairs - len(mixed_vals))
            if n_pairs:
                mixed = np.ascontiguousarray(np.asarray(mixed_vals, dtype=np.uint16))
                mixed_ptr = _u16_ptr(mixed)
        if cubic_mixed is not None:
            cm_vals = [int(v) & 0xFFFF for v in cubic_mixed]
            if len(cm_vals) > 2:
                raise ValueError(f"cubic_mixed has {len(cm_vals)} entries, 2 allowed")
            cm_vals += [0] * (2 - len(cm_vals))
            if order != 2 and any(cm_vals):
                raise ValueError(
                    "cubic_mixed carries the AlgRef v0.2 §13.6 two-dimensional "
                    f"mixed cubics; order is {order}, so both words must be zero"
                )
            cmixed = np.ascontiguousarray(np.asarray(cm_vals, dtype=np.uint16))
            cubic_mixed_ptr = _u16_ptr(cmixed)
    if phase_gradient is not None:
        grad_vals = [int(v) & 0xFFFF for v in phase_gradient]
        if len(grad_vals) > order:
            raise ValueError(f"phase_gradient has {len(grad_vals)} entries, order is {order}")
        grad_vals += [0] * (order - len(grad_vals))
        if order:
            grad = np.ascontiguousarray(np.asarray(grad_vals, dtype=np.uint16))
            grad_ptr = _u16_ptr(grad)
    lib.spf32_phase_ramp_nd_apply_array(
        _u32_ptr(arr.ravel()),
        _u32_ptr(dims),
        ctypes.c_uint8(int(order)),
        _u16_ptr(rr.ravel()),
        _u16_ptr(rb),
        quad_ptr,
        cube_ptr,
        grad_ptr,
        mixed_ptr,
        cubic_mixed_ptr,
        ctypes.c_int(1 if nonlinear else 0),
        _u32_ptr(out.ravel()),
        ctypes.c_size_t(arr.size),
    )
    return out


def gain_phase_apply(
    x: npt.ArrayLike, dL: int, dtheta: int
) -> npt.NDArray[np.uint32]:
    """Apply a constant log-magnitude and phase offset to SPF words."""
    lib = load_library()
    arr = _prep_word(x)
    out = np.empty_like(arr)
    lib.spf32_gain_phase_apply_array(
        _u32_ptr(arr.ravel()),
        ctypes.c_int16(int(dL)),
        ctypes.c_uint16(int(dtheta) & 0xFFFF),
        _u32_ptr(out.ravel()),
        ctypes.c_size_t(arr.size),
    )
    return out


def pairwise_phase_diff(a: npt.ArrayLike, b: npt.ArrayLike) -> npt.NDArray[np.uint16]:
    """Pairwise modular theta differences: ``theta(a[i]) - theta(b[i])``."""
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(a), _prep_word(b))
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    out = np.empty(a_arr.shape, dtype=np.uint16)
    lib.spf32_pairwise_phase_diff_array(
        _u32_ptr(a_arr.ravel()),
        _u32_ptr(b_arr.ravel()),
        _u16_ptr(out.ravel()),
        ctypes.c_size_t(a_arr.size),
    )
    return out


def log_power(x: npt.ArrayLike) -> npt.NDArray[np.int32]:
    """Return log2 power in Q7.8 ticks from SPF words."""
    lib = load_library()
    arr = _prep_word(x)
    out = np.empty(arr.shape, dtype=np.int32)
    lib.spf32_log_power_array(
        _u32_ptr(arr.ravel()), _i32_ptr(out.ravel()), ctypes.c_size_t(arr.size)
    )
    return out


def channel_estimate_ls(
    y_pilot: npt.ArrayLike, x_pilot: npt.ArrayLike
) -> npt.NDArray[np.uint32]:
    """Least-squares SPF channel estimate: ``H = Y / X`` over pilot lanes."""
    return _binary_word("spf32_channel_estimate_ls_array", y_pilot, x_pilot)


def channel_inverse_guard(
    h: npt.ArrayLike,
    *,
    min_channel_L: int,
    max_inverse_gain_L: int,
    return_erasure: bool = True,
) -> npt.NDArray[np.uint32] | tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint8]]:
    """Invert channel words while emitting/using erasures for unsafe lanes."""
    lib = load_library()
    arr = _prep_word(h)
    out = np.empty_like(arr)
    erasure = np.empty(arr.shape, dtype=np.uint8) if return_erasure else None
    lib.spf32_channel_inverse_guard_array(
        _u32_ptr(arr.ravel()),
        _u32_ptr(out.ravel()),
        _u8_optional_ptr(None if erasure is None else erasure.ravel()),
        ctypes.c_int16(int(min_channel_L)),
        ctypes.c_int16(int(max_inverse_gain_L)),
        ctypes.c_size_t(arr.size),
    )
    if erasure is None:
        return out
    return out, erasure


def ofdm_cpe_estimate_from_pilots(
    y_pilot: npt.ArrayLike,
    h_inv_pilot: npt.ArrayLike,
    x_pilot_known: npt.ArrayLike,
    pilot_mask: npt.ArrayLike | None = None,
) -> int:
    """Estimate common phase error from SPF pilot bins."""
    lib = load_library()
    y_arr, h_arr, x_arr = np.broadcast_arrays(
        _prep_word(y_pilot), _prep_word(h_inv_pilot), _prep_word(x_pilot_known)
    )
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    x_arr = np.ascontiguousarray(x_arr).ravel()
    mask = _prep_u8_optional(pilot_mask, y_arr.size)
    return int(
        lib.spf32_ofdm_cpe_estimate_from_pilots(
            _u32_ptr(y_arr),
            _u32_ptr(h_arr),
            _u32_ptr(x_arr),
            _u8_optional_ptr(mask),
            ctypes.c_size_t(y_arr.size),
        )
    )


def ofdm_qpsk_eq_demod_fused(
    y: npt.ArrayLike,
    h_inv: npt.ArrayLike,
    cpe_theta: int = 0,
    *,
    quadrant_to_bits_lut: npt.ArrayLike | None = None,
    erasure_mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Fused SPF OFDM equalize+CPE+QPSK hard-decision demap."""
    lib = load_library()
    y_arr, h_arr = np.broadcast_arrays(_prep_word(y), _prep_word(h_inv))
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    qlut = (
        np.arange(4, dtype=np.uint8)
        if quadrant_to_bits_lut is None
        else np.ascontiguousarray(np.asarray(quadrant_to_bits_lut), dtype=np.uint8)
    )
    if qlut.size != 4:
        raise ValueError(f"quadrant_to_bits_lut must have 4 entries, got {qlut.size}")
    erasure = _prep_u8_optional(erasure_mask, y_arr.size)
    bits = np.empty(y_arr.size * 2, dtype=np.uint8)
    confidence = np.empty(y_arr.size, dtype=np.uint8)
    lib.spf32_ofdm_qpsk_eq_demod_fused_array(
        _u32_ptr(y_arr),
        _u32_ptr(h_arr),
        ctypes.c_uint16(int(cpe_theta) & 0xFFFF),
        _u8_ptr(qlut),
        _u8_ptr(bits),
        _u8_ptr(confidence),
        _u8_optional_ptr(erasure),
        ctypes.c_size_t(y_arr.size),
    )
    return bits, confidence


def ofdm_mpsk_eq_demod_fused(
    y: npt.ArrayLike,
    h_inv: npt.ArrayLike,
    bits_per_symbol: int,
    symbol_to_bits_lut: npt.ArrayLike,
    cpe_theta: int = 0,
    *,
    erasure_mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Fused SPF OFDM equalize+CPE+M-PSK hard-decision demap."""
    if bits_per_symbol <= 0 or bits_per_symbol > 8:
        raise ValueError(f"bits_per_symbol must be in [1, 8], got {bits_per_symbol}")
    lib = load_library()
    y_arr, h_arr = np.broadcast_arrays(_prep_word(y), _prep_word(h_inv))
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    lut = np.ascontiguousarray(np.asarray(symbol_to_bits_lut), dtype=np.uint8)
    if lut.size < (1 << bits_per_symbol):
        raise ValueError(
            f"symbol_to_bits_lut has {lut.size} entries, expected at least {1 << bits_per_symbol}"
        )
    erasure = _prep_u8_optional(erasure_mask, y_arr.size)
    bits = np.empty(y_arr.size * bits_per_symbol, dtype=np.uint8)
    confidence = np.empty(y_arr.size, dtype=np.uint8)
    lib.spf32_ofdm_mpsk_eq_demod_fused_array(
        _u32_ptr(y_arr),
        _u32_ptr(h_arr),
        ctypes.c_uint16(int(cpe_theta) & 0xFFFF),
        ctypes.c_uint8(int(bits_per_symbol)),
        _u8_ptr(lut),
        _u8_ptr(bits),
        _u8_ptr(confidence),
        _u8_optional_ptr(erasure),
        ctypes.c_size_t(y_arr.size),
    )
    return bits, confidence


def constellation_lut_demap_fused(
    y: npt.ArrayLike,
    h_inv: npt.ArrayLike,
    lut_symbol: npt.ArrayLike,
    lut_confidence: npt.ArrayLike,
    *,
    agc_L_offset: int = 0,
    cpe_theta: int = 0,
    l_shift: int,
    theta_shift: int,
    l_min: int,
    l_max: int,
    erasure_mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Fused equalize+CPE+2D constellation LUT hard-decision demap."""
    lib = load_library()
    y_arr, h_arr = np.broadcast_arrays(_prep_word(y), _prep_word(h_inv))
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    sym_lut = np.ascontiguousarray(np.asarray(lut_symbol), dtype=np.uint8)
    conf_lut = np.ascontiguousarray(np.asarray(lut_confidence), dtype=np.uint8)
    if sym_lut.size != conf_lut.size:
        raise ValueError(
            f"lut_symbol length {sym_lut.size} != lut_confidence length {conf_lut.size}"
        )
    expected_bins = _constellation_lut_expected_bins(l_shift, theta_shift, l_min, l_max)
    if sym_lut.size != expected_bins:
        raise ValueError(
            f"lut_symbol/lut_confidence length {sym_lut.size} != expected {expected_bins} "
            f"for l_shift={l_shift}, theta_shift={theta_shift}, l_min={l_min}, l_max={l_max}"
        )
    erasure = _prep_u8_optional(erasure_mask, y_arr.size)
    symbols = np.empty(y_arr.size, dtype=np.uint8)
    confidence = np.empty(y_arr.size, dtype=np.uint8)
    lib.spf32_constellation_lut_demap_fused_array(
        _u32_ptr(y_arr),
        _u32_ptr(h_arr),
        ctypes.c_int16(int(agc_L_offset)),
        ctypes.c_uint16(int(cpe_theta) & 0xFFFF),
        _u8_ptr(sym_lut),
        _u8_ptr(conf_lut),
        ctypes.c_uint8(int(l_shift)),
        ctypes.c_uint8(int(theta_shift)),
        ctypes.c_int16(int(l_min)),
        ctypes.c_int16(int(l_max)),
        _u8_ptr(symbols),
        _u8_ptr(confidence),
        _u8_optional_ptr(erasure),
        ctypes.c_size_t(y_arr.size),
    )
    return symbols, confidence


def qam16_logproj_eq_demod_fused(
    y: npt.ArrayLike,
    h_inv: npt.ArrayLike,
    cpe_theta: int = 0,
    *,
    agc_L_offset: int = 0,
    threshold_L: int = 256,
    phase_bits: int = 10,
    erasure_mask: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    """Fused SPF equalize+CPE+QAM16 hard-decision demap.

    The native kernel uses equalized log magnitude and phase fields directly;
    it does not decode each symbol to Cartesian coordinates.
    """
    lib = load_library()
    y_arr, h_arr = np.broadcast_arrays(_prep_word(y), _prep_word(h_inv))
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    erasure = _prep_u8_optional(erasure_mask, y_arr.size)
    bits = np.empty(y_arr.size * 4, dtype=np.uint8)
    confidence = np.empty(y_arr.size, dtype=np.uint8)
    lib.spf32_qam16_logproj_eq_demod_fused_array(
        _u32_ptr(y_arr),
        _u32_ptr(h_arr),
        ctypes.c_int16(int(agc_L_offset)),
        ctypes.c_uint16(int(cpe_theta) & 0xFFFF),
        ctypes.c_int16(int(threshold_L)),
        ctypes.c_uint8(int(phase_bits)),
        _u8_ptr(bits),
        _u8_ptr(confidence),
        _u8_optional_ptr(erasure),
        ctypes.c_size_t(y_arr.size),
    )
    return bits, confidence


def pilot_interp_loglinear(
    pilot_words: npt.ArrayLike,
    pilot_indices: npt.ArrayLike,
    n_fft: int,
    active_indices: npt.ArrayLike,
) -> npt.NDArray[np.uint32]:
    """Native log-linear pilot interpolation (RX-6b).

    Runs ``spf32_pilot_interp_loglinear_array``: linear interp on the L int16
    field and signed-circular linear interp on the u16 theta field, with no
    Cartesian decode of the pilot magnitudes. Inputs must be strictly
    ascending; pilots outside ``active_indices`` are clamped to the nearest
    pilot value.
    """
    lib = load_library()
    pilot_arr = np.ascontiguousarray(np.asarray(pilot_words, dtype=np.uint32)).ravel()
    pilot_idx = np.ascontiguousarray(
        np.asarray(pilot_indices, dtype=np.uint32)
    ).ravel()
    active_idx = np.ascontiguousarray(
        np.asarray(active_indices, dtype=np.uint32)
    ).ravel()
    if pilot_arr.size != pilot_idx.size:
        raise ValueError(
            f"pilot_words size {pilot_arr.size} != pilot_indices size {pilot_idx.size}"
        )
    if pilot_idx.size and not np.all(pilot_idx[1:] > pilot_idx[:-1]):
        raise ValueError("pilot_indices must be strictly ascending")
    if active_idx.size and not np.all(active_idx[1:] > active_idx[:-1]):
        raise ValueError("active_indices must be strictly ascending")
    out = np.empty(active_idx.size, dtype=np.uint32)
    lib.spf32_pilot_interp_loglinear_array(
        _u32_ptr(pilot_arr),
        _u32_ptr(pilot_idx),
        ctypes.c_size_t(pilot_arr.size),
        ctypes.c_uint32(int(n_fft)),
        _u32_ptr(active_idx),
        ctypes.c_size_t(active_idx.size),
        _u32_ptr(out),
    )
    return out


def constellation_lut_llr_fused(
    y: npt.ArrayLike,
    h_inv: npt.ArrayLike,
    lut_llr: npt.ArrayLike,
    bits_per_symbol: int,
    *,
    agc_L_offset: int = 0,
    cpe_theta: int = 0,
    l_shift: int,
    theta_shift: int,
    l_min: int,
    l_max: int,
    erasure_mask: npt.ArrayLike | None = None,
) -> npt.NDArray[np.int8]:
    """Fused equalize+CPE+2D constellation LUT LLR demap."""
    if bits_per_symbol <= 0:
        raise ValueError(f"bits_per_symbol must be positive, got {bits_per_symbol}")
    lib = load_library()
    y_arr, h_arr = np.broadcast_arrays(_prep_word(y), _prep_word(h_inv))
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    llr_lut = np.ascontiguousarray(np.asarray(lut_llr), dtype=np.int8)
    expected_bins = _constellation_lut_expected_bins(l_shift, theta_shift, l_min, l_max)
    expected_llr = expected_bins * int(bits_per_symbol)
    if llr_lut.size != expected_llr:
        raise ValueError(
            f"lut_llr length {llr_lut.size} != expected {expected_llr} "
            f"({expected_bins} bins * {bits_per_symbol} bits_per_symbol)"
        )
    erasure = _prep_u8_optional(erasure_mask, y_arr.size)
    llr = np.empty(y_arr.size * bits_per_symbol, dtype=np.int8)
    lib.spf32_constellation_lut_llr_fused_array(
        _u32_ptr(y_arr),
        _u32_ptr(h_arr),
        ctypes.c_int16(int(agc_L_offset)),
        ctypes.c_uint16(int(cpe_theta) & 0xFFFF),
        _i8_ptr(llr_lut),
        ctypes.c_uint8(int(bits_per_symbol)),
        ctypes.c_uint8(int(l_shift)),
        ctypes.c_uint8(int(theta_shift)),
        ctypes.c_int16(int(l_min)),
        ctypes.c_int16(int(l_max)),
        _i8_ptr(llr),
        _u8_optional_ptr(erasure),
        ctypes.c_size_t(y_arr.size),
    )
    return llr


def phase_slope_estimate(
    samples: npt.ArrayLike,
    *,
    n_streams: int,
    stride: int,
    n_per_stream: int,
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint16], npt.NDArray[np.uint8]]:
    """Estimate per-stream phase slope, intercept, and confidence."""
    lib = load_library()
    arr = _prep_word(samples).ravel()
    if n_streams < 0 or stride < 0 or n_per_stream < 0:
        raise ValueError("n_streams, stride, and n_per_stream must be non-negative")
    required = 0
    if n_streams > 0 and n_per_stream > 0:
        required = (n_streams - 1) * stride + n_per_stream
    if arr.size < required:
        raise ValueError(
            f"samples length {arr.size} is too short for n_streams={n_streams}, "
            f"stride={stride}, n_per_stream={n_per_stream}; need at least {required}"
        )
    slope = np.empty(n_streams, dtype=np.uint16)
    intercept = np.empty(n_streams, dtype=np.uint16)
    confidence = np.empty(n_streams, dtype=np.uint8)
    lib.spf32_phase_slope_estimate_array(
        _u32_ptr(arr),
        _u16_ptr(slope),
        _u16_ptr(intercept),
        _u8_ptr(confidence),
        ctypes.c_size_t(n_streams),
        ctypes.c_size_t(stride),
        ctypes.c_size_t(n_per_stream),
    )
    return slope, intercept, confidence


def differential_preamble_match(
    rx: npt.ArrayLike,
    expected_delta_theta: npt.ArrayLike,
    tolerance: int,
    *,
    preamble_len: int | None = None,
) -> npt.NDArray[np.uint16]:
    """Score differential phase preamble matches at every possible offset."""
    lib = load_library()
    rx_arr = _prep_word(rx).ravel()
    expected = np.ascontiguousarray(np.asarray(expected_delta_theta), dtype=np.uint16).ravel()
    plen = expected.size + 1 if preamble_len is None else int(preamble_len)
    if plen <= 1:
        raise ValueError(f"preamble_len must be > 1, got {plen}")
    if plen > expected.size + 1:
        raise ValueError(
            f"preamble_len {plen} exceeds expected_delta_theta length {expected.size} + 1"
        )
    n_scores = max(rx_arr.size - plen + 1, 0)
    scores = np.zeros(n_scores, dtype=np.uint16)
    if n_scores:
        lib.spf32_differential_preamble_match_array(
            _u32_ptr(rx_arr),
            _u16_ptr(expected),
            ctypes.c_uint16(int(tolerance) & 0xFFFF),
            _u16_ptr(scores),
            ctypes.c_size_t(rx_arr.size),
            ctypes.c_size_t(plen),
        )
    return scores


def doppler_phase_diff_estimate(
    slow_time_samples: npt.ArrayLike,
    *,
    n_range_bins: int,
    n_chirps: int,
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint8]]:
    """Estimate Doppler phase per range bin from adjacent slow-time deltas."""
    lib = load_library()
    arr = _prep_word(slow_time_samples).ravel()
    expected = int(n_range_bins) * int(n_chirps)
    if arr.size != expected:
        raise ValueError(
            f"slow_time_samples length {arr.size} != n_range_bins*n_chirps {expected}"
        )
    doppler = np.empty(int(n_range_bins), dtype=np.uint16)
    confidence = np.empty(int(n_range_bins), dtype=np.uint8)
    lib.spf32_doppler_phase_diff_estimate_array(
        _u32_ptr(arr),
        _u16_ptr(doppler),
        _u8_ptr(confidence),
        ctypes.c_size_t(int(n_range_bins)),
        ctypes.c_size_t(int(n_chirps)),
    )
    return doppler, confidence


def aoa_pair_phase(
    ant_a: npt.ArrayLike, ant_b: npt.ArrayLike
) -> tuple[npt.NDArray[np.uint16], npt.NDArray[np.uint8]]:
    """Pairwise AoA phase delta ``theta(ant_b) - theta(ant_a)`` and confidence."""
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(ant_a), _prep_word(ant_b))
    a_arr = np.ascontiguousarray(a_arr).ravel()
    b_arr = np.ascontiguousarray(b_arr).ravel()
    delta = np.empty(a_arr.size, dtype=np.uint16)
    confidence = np.empty(a_arr.size, dtype=np.uint8)
    lib.spf32_aoa_pair_phase_array(
        _u32_ptr(a_arr),
        _u32_ptr(b_arr),
        _u16_ptr(delta),
        _u8_ptr(confidence),
        ctypes.c_size_t(a_arr.size),
    )
    return delta, confidence


def aoa_lut(
    phase_delta: npt.ArrayLike,
    confidence: npt.ArrayLike,
    theta_to_angle_lut: npt.ArrayLike,
) -> tuple[npt.NDArray[np.int16], npt.NDArray[np.uint8]]:
    """Map AoA phase deltas through the native 256-entry angle LUT."""
    lib = load_library()
    delta = np.ascontiguousarray(np.asarray(phase_delta), dtype=np.uint16).ravel()
    conf = np.ascontiguousarray(np.asarray(confidence), dtype=np.uint8).ravel()
    lut = np.ascontiguousarray(np.asarray(theta_to_angle_lut), dtype=np.int16).ravel()
    if conf.size != delta.size:
        raise ValueError(f"confidence length {conf.size} != phase_delta length {delta.size}")
    if lut.size != 256:
        raise ValueError(f"theta_to_angle_lut must have 256 entries, got {lut.size}")
    angles = np.empty(delta.size, dtype=np.int16)
    valid = np.empty(delta.size, dtype=np.uint8)
    lib.spf32_aoa_lut_array(
        _u16_ptr(delta),
        _u8_ptr(conf),
        _i16_ptr(lut),
        _i16_ptr(angles),
        _u8_ptr(valid),
        ctypes.c_size_t(delta.size),
    )
    return angles, valid


def logpower_threshold(
    x: npt.ArrayLike, threshold_log2_power_q78: int
) -> npt.NDArray[np.uint8]:
    """Threshold SPF log-power and return a uint8 detection mask."""
    lib = load_library()
    arr = _prep_word(x)
    mask = np.empty(arr.shape, dtype=np.uint8)
    lib.spf32_logpower_threshold_array(
        _u32_ptr(arr.ravel()),
        ctypes.c_int32(int(threshold_log2_power_q78)),
        _u8_ptr(mask.ravel()),
        ctypes.c_size_t(arr.size),
    )
    return mask


def log_os_cfar(
    cells: npt.ArrayLike,
    *,
    threshold_offset_q78: int,
    guard: int,
    train: int,
    rank: int,
) -> npt.NDArray[np.uint8]:
    """Run ordered-statistic CFAR over SPF log-power cells."""
    lib = load_library()
    arr = _prep_word(cells)
    detections = np.empty(arr.shape, dtype=np.uint8)
    lib.spf32_log_os_cfar_array(
        _u32_ptr(arr.ravel()),
        _u8_ptr(detections.ravel()),
        ctypes.c_int16(int(threshold_offset_q78)),
        ctypes.c_size_t(int(arr.size)),
        ctypes.c_size_t(int(guard)),
        ctypes.c_size_t(int(train)),
        ctypes.c_uint(int(rank)),
    )
    return detections


def fb7_reduce_tree(x: npt.ArrayLike, mode: str | int = "balanced") -> np.uint32:
    """Coherently reduce an SPF word vector using the native FB7 reduce tree."""
    from spf.native.fb7 import mode_from_name

    lib = load_library()
    arr = _prep_word(x).ravel()
    out = np.empty(1, dtype=np.uint32)
    lib.spf32_fb7_reduce_tree_array(
        _u32_ptr(arr),
        _u32_ptr(out),
        ctypes.c_size_t(arr.size),
        ctypes.c_int(mode_from_name(mode)),
    )
    return np.uint32(out[0])


def spectral_matched_filter_fb7(
    x_fft: npt.ArrayLike,
    h_conj_fft: npt.ArrayLike,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """Run native spectral matched filter FB7 helper."""
    from spf.native.fb7 import mode_from_name

    lib = load_library()
    x_arr, h_arr = np.broadcast_arrays(_prep_word(x_fft), _prep_word(h_conj_fft))
    x_arr = np.ascontiguousarray(x_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    out = np.empty_like(x_arr)
    scratch = np.empty_like(x_arr)
    rc = lib.spf32_spectral_matched_filter_fb7(
        _u32_ptr(x_arr),
        _u32_ptr(h_arr),
        _u32_ptr(out),
        _u32_ptr(scratch),
        ctypes.c_int(int(x_arr.size)),
        ctypes.c_int(mode_from_name(mode)),
    )
    if rc != 0:
        raise ValueError(f"spf32_spectral_matched_filter_fb7 failed with rc={rc}")
    return out


def candidate_fft_bins_fb7(
    x: npt.ArrayLike,
    candidate_bins: npt.ArrayLike,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """Compute selected FFT bins through the native FB7 candidate-bin helper."""
    from spf.native.fb7 import mode_from_name

    lib = load_library()
    arr = _prep_word(x).ravel()
    bins = np.ascontiguousarray(np.asarray(candidate_bins), dtype=np.uint16).ravel()
    out = np.empty(bins.size, dtype=np.uint32)
    scratch = np.empty(arr.size, dtype=np.uint32)
    rc = lib.spf32_candidate_fft_bins_fb7(
        _u32_ptr(arr),
        _u16_ptr(bins),
        _u32_ptr(out),
        _u32_ptr(scratch),
        ctypes.c_size_t(arr.size),
        ctypes.c_size_t(bins.size),
        ctypes.c_int(mode_from_name(mode)),
    )
    if rc != 0:
        raise ValueError(f"spf32_candidate_fft_bins_fb7 failed with rc={rc}")
    return out


def beamform_fused(
    antenna_words: npt.ArrayLike,
    *,
    gain_l_offsets: npt.ArrayLike | None = None,
    phase_offsets: npt.ArrayLike | None = None,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """Fused per-antenna gain/phase apply plus FB7 reduce per output sample.

    ``antenna_words`` must be antenna-major with shape
    ``(n_antennas, n_samples)``.
    """
    from spf.native.fb7 import mode_from_name

    lib = load_library()
    words = np.ascontiguousarray(np.asarray(antenna_words), dtype=np.uint32)
    if words.ndim != 2:
        raise ValueError(
            f"antenna_words must have shape (n_antennas, n_samples), got {words.shape}"
        )
    n_antennas, n_samples = words.shape
    gain = (
        np.zeros(n_antennas, dtype=np.int16)
        if gain_l_offsets is None
        else np.ascontiguousarray(np.asarray(gain_l_offsets), dtype=np.int16).ravel()
    )
    phase = (
        np.zeros(n_antennas, dtype=np.uint16)
        if phase_offsets is None
        else np.ascontiguousarray(np.asarray(phase_offsets), dtype=np.uint16).ravel()
    )
    if gain.size != n_antennas:
        raise ValueError(f"gain_l_offsets length {gain.size} != n_antennas {n_antennas}")
    if phase.size != n_antennas:
        raise ValueError(f"phase_offsets length {phase.size} != n_antennas {n_antennas}")
    out = np.empty(n_samples, dtype=np.uint32)
    lib.spf32_beamform_fused_array(
        _u32_ptr(words.ravel()),
        _i16_ptr(gain),
        _u16_ptr(phase),
        _u32_ptr(out),
        ctypes.c_size_t(n_antennas),
        ctypes.c_size_t(n_samples),
        ctypes.c_int(mode_from_name(mode)),
    )
    return out


def runtime_radar_sparse_doppler(
    slow_time_words: npt.ArrayLike,
    *,
    n_range_bins: int,
    n_chirps: int,
    threshold_log2_power_q78: int = 0,
    detection_mode: str | int = "threshold",
    range_profile_words: npt.ArrayLike | None = None,
    cfar_threshold_offset_q78: int = 0,
    cfar_guard: int = 0,
    cfar_train: int = 1,
    cfar_rank: int = 1,
    fallback_confidence_threshold: int = 1,
    scalar: bool = False,
) -> tuple[
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint16],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    """Run the C product ABI for sparse radar detection plus Doppler."""
    lib = load_library()
    cube = _prep_word(slow_time_words).ravel()
    expected = int(n_range_bins) * int(n_chirps)
    if cube.size != expected:
        raise ValueError(f"slow_time_words length {cube.size} != n_range_bins*n_chirps {expected}")
    profile = None
    if range_profile_words is not None:
        profile = _prep_word(range_profile_words).ravel()
        if profile.size != int(n_range_bins):
            raise ValueError(
                f"range_profile_words length {profile.size} != n_range_bins {n_range_bins}"
            )
    if isinstance(detection_mode, str):
        mode_map = {"threshold": 0, "cfar": 1}
        if detection_mode not in mode_map:
            raise ValueError(f"unknown detection_mode {detection_mode!r}")
        mode = mode_map[detection_mode]
    else:
        mode = int(detection_mode)
    detections = np.empty(int(n_range_bins), dtype=np.uint8)
    doppler = np.empty(int(n_range_bins), dtype=np.uint16)
    confidence = np.empty(int(n_range_bins), dtype=np.uint8)
    fallback_mask = np.empty(int(n_range_bins), dtype=np.uint8)
    entry = "runtime_radar_sparse_doppler_scalar" if scalar else "runtime_radar_sparse_doppler"
    getattr(lib, entry)(
        _u32_ptr(cube),
        _u32_ptr(profile) if profile is not None else ctypes.POINTER(ctypes.c_uint32)(),
        ctypes.c_uint8(mode),
        ctypes.c_int32(int(threshold_log2_power_q78)),
        ctypes.c_int16(int(cfar_threshold_offset_q78)),
        ctypes.c_size_t(int(cfar_guard)),
        ctypes.c_size_t(int(cfar_train)),
        ctypes.c_uint(int(cfar_rank)),
        ctypes.c_uint8(int(fallback_confidence_threshold)),
        _u8_ptr(detections),
        _u16_ptr(doppler),
        _u8_ptr(confidence),
        _u8_ptr(fallback_mask),
        ctypes.c_size_t(int(n_range_bins)),
        ctypes.c_size_t(int(n_chirps)),
    )
    return detections, doppler, confidence, fallback_mask


def runtime_aoa_pair_lut(
    ant_a_words: npt.ArrayLike,
    ant_b_words: npt.ArrayLike,
    theta_to_angle_lut: npt.ArrayLike,
    *,
    calibration_phase_offset: int = 0,
    min_confidence: int = 1,
    scalar: bool = False,
) -> tuple[
    npt.NDArray[np.uint16],
    npt.NDArray[np.int16],
    npt.NDArray[np.uint8],
    npt.NDArray[np.uint8],
]:
    """Run the C product ABI for pairwise AoA phase plus angle LUT."""
    lib = load_library()
    ant_a, ant_b = np.broadcast_arrays(_prep_word(ant_a_words), _prep_word(ant_b_words))
    ant_a = np.ascontiguousarray(ant_a).ravel()
    ant_b = np.ascontiguousarray(ant_b).ravel()
    lut = np.ascontiguousarray(np.asarray(theta_to_angle_lut), dtype=np.int16).ravel()
    if lut.size != 256:
        raise ValueError(f"theta_to_angle_lut must have 256 entries, got {lut.size}")
    phase_delta = np.empty(ant_a.size, dtype=np.uint16)
    angle = np.empty(ant_a.size, dtype=np.int16)
    valid = np.empty(ant_a.size, dtype=np.uint8)
    confidence = np.empty(ant_a.size, dtype=np.uint8)
    entry = "runtime_aoa_pair_lut_scalar" if scalar else "runtime_aoa_pair_lut"
    getattr(lib, entry)(
        _u32_ptr(ant_a),
        _u32_ptr(ant_b),
        _i16_ptr(lut),
        ctypes.c_uint16(int(calibration_phase_offset) & 0xFFFF),
        ctypes.c_uint8(int(min_confidence)),
        _u16_ptr(phase_delta),
        _i16_ptr(angle),
        _u8_ptr(valid),
        _u8_ptr(confidence),
        ctypes.c_size_t(ant_a.size),
    )
    return phase_delta, angle, valid, confidence


def runtime_rf_native_ingress(
    l_code: npt.ArrayLike,
    theta: npt.ArrayLike,
    *,
    scalar: bool = False,
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.int32]]:
    """Run the C product ABI for RF-native polar ingress."""
    lib = load_library()
    l_arr, t_arr = np.broadcast_arrays(
        np.asarray(l_code, dtype=np.int16),
        np.asarray(theta, dtype=np.uint16),
    )
    l_arr = np.ascontiguousarray(l_arr, dtype=np.int16).ravel()
    t_arr = np.ascontiguousarray(t_arr, dtype=np.uint16).ravel()
    words = np.empty(l_arr.size, dtype=np.uint32)
    log_power_q78 = np.empty(l_arr.size, dtype=np.int32)
    entry = "runtime_rf_native_ingress_scalar" if scalar else "runtime_rf_native_ingress"
    getattr(lib, entry)(
        _i16_ptr(l_arr),
        _u16_ptr(t_arr),
        _u32_ptr(words),
        _i32_ptr(log_power_q78),
        ctypes.c_size_t(l_arr.size),
    )
    return words, log_power_q78


def qpsk_phase_demod(x_hat: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """QPSK hard decisions from SPF theta, returning two bit bytes per word."""
    lib = load_library()
    arr = _prep_word(x_hat).ravel()
    bits = np.empty(arr.size * 2, dtype=np.uint8)
    lib.spf32_qpsk_phase_demod_array(
        _u32_ptr(arr),
        bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        ctypes.c_size_t(arr.size),
    )
    return bits


def qpsk_phase_demod_conf(
    x_hat: npt.ArrayLike,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]]:
    """QPSK hard decisions plus phase-boundary confidence from SPF theta.

    Confidence is measured in theta ticks: 0 at a quadrant decision boundary
    and 8192 at the center of a quadrant.
    """
    lib = load_library()
    arr = _prep_word(x_hat).ravel()
    bits = np.empty(arr.size * 2, dtype=np.uint8)
    confidence = np.empty(arr.size, dtype=np.uint16)
    lib.spf32_qpsk_phase_demod_conf_array(
        _u32_ptr(arr),
        bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        confidence.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        ctypes.c_size_t(arr.size),
    )
    return bits, confidence


def ofdm_qpsk_rx_hotpath(
    y: npt.ArrayLike,
    h_inv: npt.ArrayLike,
    cpe_theta: int = 0,
    *,
    diagnostics: bool = False,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint16]] | tuple[
    npt.NDArray[np.uint8], npt.NDArray[np.uint16], npt.NDArray[np.uint32]
]:
    """Fused SPF-native OFDM QPSK receiver hot path.

    Computes ``theta(Y[k]) + theta(H_inv[k]) + cpe_theta`` and emits bits and
    confidence directly. Set ``diagnostics=True`` to also materialize the
    equalized SPF words for BER/EVM reports.
    """
    lib = load_library()
    y_arr, h_arr = np.broadcast_arrays(_prep_word(y), _prep_word(h_inv))
    y_arr = np.ascontiguousarray(y_arr).ravel()
    h_arr = np.ascontiguousarray(h_arr).ravel()
    bits = np.empty(y_arr.size * 2, dtype=np.uint8)
    confidence = np.empty(y_arr.size, dtype=np.uint16)
    diag = np.empty(y_arr.size, dtype=np.uint32) if diagnostics else None
    diag_ptr = (
        _u32_ptr(diag)
        if diag is not None
        else ctypes.POINTER(ctypes.c_uint32)()
    )
    lib.spf32_ofdm_qpsk_rx_hotpath_array(
        _u32_ptr(y_arr),
        _u32_ptr(h_arr),
        ctypes.c_uint16(int(cpe_theta) & 0xFFFF),
        bits.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        confidence.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        diag_ptr,
        ctypes.c_size_t(y_arr.size),
    )
    if diag is not None:
        return bits, confidence, diag
    return bits, confidence


def add_poly7(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """Poly7 ADD: Sb-free complex addition via FB7 polynomial kernels (§4.3).

    Unlike :func:`add` (which uses the Sb() LUT), this path uses the same
    shared sin/cos/log2/exp2/atan2 polynomial approximations as the FB7
    butterfly. The ``mode`` parameter selects the dominant-gap bypass
    threshold: ``"fast"`` (gap ≥ 8·L_SCALE), ``"balanced"`` (gap ≥ 12·L_SCALE),
    or ``"accurate"`` (no bypass; same polynomial finish).
    """
    from spf.native.fb7 import mode_from_name
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(a), _prep_word(b))
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    out = np.empty_like(a_arr)
    lib.spf32_add_poly7_array(
        _u32_ptr(a_arr), _u32_ptr(b_arr), _u32_ptr(out),
        ctypes.c_size_t(a_arr.size),
        ctypes.c_int(mode_from_name(mode)),
    )
    return out


def sub_poly7(
    a: npt.ArrayLike,
    b: npt.ArrayLike,
    mode: str | int = "balanced",
) -> npt.NDArray[np.uint32]:
    """Poly7 SUB: equivalent to ``add_poly7(a, neg(b), mode)``.

    See :func:`add_poly7` for mode semantics.
    """
    from spf.native.fb7 import mode_from_name
    lib = load_library()
    a_arr, b_arr = np.broadcast_arrays(_prep_word(a), _prep_word(b))
    a_arr = np.ascontiguousarray(a_arr)
    b_arr = np.ascontiguousarray(b_arr)
    out = np.empty_like(a_arr)
    lib.spf32_sub_poly7_array(
        _u32_ptr(a_arr), _u32_ptr(b_arr), _u32_ptr(out),
        ctypes.c_size_t(a_arr.size),
        ctypes.c_int(mode_from_name(mode)),
    )
    return out


def butterfly(
    a: npt.ArrayLike, b: npt.ArrayLike, w: npt.ArrayLike
) -> tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
    """C port of :func:`spf.scalar.spf32.butterfly`.

    Preserves the scalar composition ``a ± W*b`` by running the two adds
    through the vectorized add/sub paths on the precomputed ``W*b``. Use
    :mod:`spf.native.fb7` for the fused FB7 butterfly modes.
    """
    wb = mul(w, b)
    return add(a, wb), sub(a, wb)


def _dot_entry(entry: str, a: npt.ArrayLike, b: npt.ArrayLike) -> np.uint32:
    lib = load_library()
    a_arr = _prep_word(a).ravel()
    b_arr = _prep_word(b).ravel()
    if a_arr.shape != b_arr.shape:
        raise ValueError(
            f"dot: shape mismatch, got {a_arr.shape} and {b_arr.shape}"
        )
    return np.uint32(
        getattr(lib, entry)(_u32_ptr(a_arr), _u32_ptr(b_arr), ctypes.c_size_t(a_arr.size))
    )


def dot(a: npt.ArrayLike, b: npt.ArrayLike) -> np.uint32:
    """C port of :func:`spf.scalar.spf32.dot`. Kahan-compensated FP64 accum."""
    return _dot_entry("spf32_dot", a, b)


def dot_nokahan(a: npt.ArrayLike, b: npt.ArrayLike) -> np.uint32:
    """C port of :func:`spf.scalar.spf32.dot_nokahan`. Baseline for accuracy bench."""
    return _dot_entry("spf32_dot_nokahan", a, b)


def pa_tone(
    freq_word: int, n: int, phase0: int = 0
) -> npt.NDArray[np.complex64]:
    """Generate N complex samples of a tone via the SPF-PA C path.

    ``freq_word`` and ``phase0`` are 32-bit unsigned. Output is
    ``cos(θ) + 1j·sin(θ)`` per step with ``θ = Φ/2³² · 2π``.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    lib = load_library()
    re = np.empty(n, dtype=np.float32)
    im = np.empty(n, dtype=np.float32)
    if n > 0:
        lib.spf32_pa_tone_sc(
            _f32_ptr(re),
            _f32_ptr(im),
            ctypes.c_uint32(int(freq_word) & 0xFFFFFFFF),
            ctypes.c_uint32(int(phase0) & 0xFFFFFFFF),
            ctypes.c_size_t(n),
        )
    out = np.empty(n, dtype=np.complex64)
    out.real = re
    out.imag = im
    return out


def pa_pe(
    seq_len: int, d_model: int, base: float = 10000.0
) -> npt.NDArray[np.float32]:
    """Sinusoidal positional encoding via the SPF-PA C path.

    Returns a ``(seq_len, d_model)`` float32 array in transformer layout
    (``[:, 0::2] = sin``, ``[:, 1::2] = cos``), matching
    :func:`spf.pa.phase_accumulator.positional_encoding` up to LUT accuracy.
    """
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError(f"d_model must be a positive even integer, got {d_model}")
    lib = load_library()
    out = np.empty((seq_len, d_model), dtype=np.float32)
    if seq_len > 0:
        lib.spf32_pa_pe_f32(
            _f32_ptr(out),
            ctypes.c_uint32(int(seq_len)),
            ctypes.c_uint32(int(d_model)),
            ctypes.c_float(float(base)),
        )
    return out


def pa_pe_slice(
    seq_start: int,
    seq_chunk: int,
    d_model: int,
    base: float = 10000.0,
) -> npt.NDArray[np.float32]:
    """Position-sliced variant of :func:`pa_pe`.

    Returns the ``seq_chunk`` rows of the PE table at absolute positions
    ``[seq_start, seq_start + seq_chunk)``. Concatenating disjoint slices
    reproduces ``pa_pe(seq_start + seq_chunk, d_model, base)[seq_start:]``
    byte-for-byte.
    """
    if seq_start < 0:
        raise ValueError(f"seq_start must be non-negative, got {seq_start}")
    if seq_chunk < 0:
        raise ValueError(f"seq_chunk must be non-negative, got {seq_chunk}")
    if d_model <= 0 or d_model % 2 != 0:
        raise ValueError(f"d_model must be a positive even integer, got {d_model}")
    lib = load_library()
    out = np.empty((seq_chunk, d_model), dtype=np.float32)
    if seq_chunk > 0:
        lib.spf32_pa_pe_f32_slice(
            _f32_ptr(out),
            ctypes.c_uint32(int(seq_start)),
            ctypes.c_uint32(int(seq_chunk)),
            ctypes.c_uint32(int(d_model)),
            ctypes.c_float(float(base)),
        )
    return out
