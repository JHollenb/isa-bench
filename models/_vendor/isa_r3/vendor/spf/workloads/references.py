"""Python reference implementations for workload contracts.

Every function takes the case parameters directly and returns
``dict[str, np.ndarray]`` keyed by the contract's output names.

When a workload has array inputs, the case supplies a ``seed`` and a size
parameter; inputs are drawn from ``numpy.random.default_rng(seed)`` so a
backend re-seeded the same way produces byte-identical inputs — no
separate input-fixture artifact is needed.
"""

from __future__ import annotations

import math

import numpy as np

from spf.dsp.aoa_runtime import estimate_pair
from spf.dsp.beam_runtime import beamform_reduce
from spf.dsp.radar_runtime import sparse_doppler
from spf.dsp.rf_native import PolarRFBuffer
from spf.fft import spf32_fft
from spf.native import spf32 as c_spf32
from spf.pa.phase_accumulator import positional_encoding
from spf.scalar import spf32
from spf.scalar._ops import _sb_exact as sb_exact


def _random_complex(rng: np.random.Generator, n: int) -> np.ndarray:
    re = rng.standard_normal(n, dtype=np.float64)
    im = rng.standard_normal(n, dtype=np.float64)
    return re + 1j * im


def _build_circulant(c: np.ndarray) -> np.ndarray:
    """Expand a first-column vector to the full dense circulant matrix."""
    n = c.shape[0]
    idx = (np.arange(n)[:, None] - np.arange(n)[None, :]) % n
    out: np.ndarray = c[idx].astype(np.float32)
    return out


def _random_int_complex(
    rng: np.random.Generator, n: int, b_max: int
) -> np.ndarray:
    """Draw n integer-complex samples uniformly from [-b_max, b_max].

    Fixture for the bounded-integer convolution operator. Returns a
    complex128 array whose ``.real`` and ``.imag`` are int-valued
    floats in [-b_max, b_max] (NOT a re-cast of complex normal noise —
    the cyclic_conv32 operator's regime is integer arithmetic).
    """
    re = rng.integers(-b_max, b_max + 1, size=n, dtype=np.int64)
    im = rng.integers(-b_max, b_max + 1, size=n, dtype=np.int64)
    return re.astype(np.float64) + 1j * im.astype(np.float64)


def generate_pe_reference(
    seq_len: int, d_model: int, base: float = 10000.0
) -> dict[str, np.ndarray]:
    """pa_pe workload: sinusoidal positional encoding via PA per freq pair."""
    encoding = positional_encoding(seq_len=seq_len, d_model=d_model, base=base)
    return {"encoding": encoding}


def generate_mul_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf32_mul workload: element-wise SPF-32 multiply of two random inputs."""
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    b = _random_complex(rng, n)
    a_w = spf32.encode(a)
    b_w = spf32.encode(b)
    result = spf32.mul(a_w, b_w)
    return {"result": result}


def generate_div_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf32_div workload: element-wise SPF-32 divide of two random inputs.

    The divisor avoids zero by adding a small offset to its magnitude so
    ``L_b`` never hits the zero sentinel; SPF-32 divide on a zero word is
    undefined in the spec.
    """
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    b = _random_complex(rng, n) + (0.5 + 0.0j)  # lift magnitudes clear of zero
    a_w = spf32.encode(a)
    b_w = spf32.encode(b)
    result = spf32.div(a_w, b_w)
    return {"result": result}


def generate_conj_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf32_conj workload: element-wise SPF-32 complex conjugate."""
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    a_w = spf32.encode(a)
    result = spf32.conj(a_w)
    return {"result": result}


def generate_abs2_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf32_abs2 workload: element-wise SPF-32 squared magnitude."""
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    a_w = spf32.encode(a)
    result = spf32.abs2(a_w)
    return {"result": result}


def generate_neg_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf32_neg workload: element-wise SPF-32 unary negation."""
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    a_w = spf32.encode(a)
    result = spf32.neg(a_w)
    return {"result": result}


def generate_add_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf32_add workload: element-wise SPF-32 complex add via Sb()."""
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    b = _random_complex(rng, n)
    a_w = spf32.encode(a)
    b_w = spf32.encode(b)
    result = spf32.add(a_w, b_w)
    return {"result": result}


def generate_twiddle_reference(
    seed: int, n: int, n_fft: int
) -> dict[str, np.ndarray]:
    """spf32_twiddle workload: DFT twiddle factors W_N^k.

    Draws n uniform int64 values of k in [0, n_fft) and returns the
    unit-magnitude SPF-32 encodings of W_N^k for the PL kernel to
    reproduce bit-for-bit (exact when n_fft divides 2¹⁶).
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(0, n_fft, size=n, dtype=np.int64)
    result = spf32.twiddle(k, n_fft)
    return {"result": result}


def generate_cyclic_conv32_reference(
    seed: int, b_max: int = 127
) -> dict[str, np.ndarray]:
    """cyclic_conv32 workload: length-32 bounded-integer cyclic complex
    convolution via Fermat-NTT / Nussbaumer.

    Returns a single ``result`` array of shape ``(32, 2)`` int32 with
    columns ``(re, im)`` per the contract's outputs block.
    """
    from spf.conv import CYCLIC_CONV32_LENGTH, cyclic_conv32

    rng = np.random.default_rng(seed)
    a = _random_int_complex(rng, CYCLIC_CONV32_LENGTH, b_max)
    b = _random_int_complex(rng, CYCLIC_CONV32_LENGTH, b_max)
    return {"result": cyclic_conv32(a, b)}


def generate_butterfly_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """butterfly workload: radix-2 DIT butterflies, n independent inputs.

    For each i in [0, n), draws ``a``, ``b`` (random complex) and a twiddle
    ``w = W_n^i`` (unit-magnitude). Returns ``top = a + w·b``, ``bot = a − w·b``.
    """
    rng = np.random.default_rng(seed)
    a = _random_complex(rng, n)
    b = _random_complex(rng, n)
    a_w = spf32.encode(a)
    b_w = spf32.encode(b)
    w = spf32.twiddle(np.arange(n, dtype=np.int64), max(n, 1))
    top, bot = spf32.butterfly(a_w, b_w, w)
    return {"top": top, "bot": bot}


def _generate_fft_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """Shared N-point forward SPF-32 DFT reference used by fft64 / 256 / 1024."""
    rng = np.random.default_rng(seed)
    x = _random_complex(rng, n)
    x_w = spf32.encode(x)
    spectrum = spf32_fft.fft(x_w)
    return {"spectrum": spectrum}


def generate_fft64_reference(seed: int, n: int = 64) -> dict[str, np.ndarray]:
    """fft64 workload: N-point SPF-32 forward DFT (default N=64)."""
    return _generate_fft_reference(seed, n)


def generate_fft64_fb7_reference(seed: int, n: int = 64) -> dict[str, np.ndarray]:
    """fft64_fb7 workload reference — same spec output as fft64.

    The FB7 fused half-angle butterfly is a hardware substitution that
    keeps the SPF-32 spectrum contract; CPU/python backends grade
    against the same libspf32 reference. The FB7 polynomial drift only
    surfaces on the kria_pl backend.
    """
    return _generate_fft_reference(seed, n)


def generate_fft64_fb7_fields_reference(
    seed: int, n: int = 64
) -> dict[str, np.ndarray]:
    """fft64_fb7_fields workload reference — same as fft64."""
    return _generate_fft_reference(seed, n)


def generate_fft64_fb7_fast_reference(
    seed: int, n: int = 64
) -> dict[str, np.ndarray]:
    """fft64_fb7_fast workload reference — same spec output as fft64.

    The FAST PL backend diverges from this reference by the dominant-gap
    threshold and all-poly approximation envelope; grading happens via
    spf32_fft64_fast_normalised_l2 tolerance, not bit-exact match against
    this reference.
    """
    return _generate_fft_reference(seed, n)


def generate_fft64_fb7_fields_fast_reference(
    seed: int, n: int = 64
) -> dict[str, np.ndarray]:
    """fft64_fb7_fields_fast workload reference — same as fft64."""
    return _generate_fft_reference(seed, n)


def generate_fft256_reference(seed: int, n: int = 256) -> dict[str, np.ndarray]:
    """fft256 workload: N-point SPF-32 forward DFT (default N=256)."""
    return _generate_fft_reference(seed, n)


def generate_fft1024_reference(seed: int, n: int = 1024) -> dict[str, np.ndarray]:
    """fft1024 workload: N-point SPF-32 forward DFT (default N=1024)."""
    return _generate_fft_reference(seed, n)


def _random_complex64(rng: np.random.Generator, n: int) -> np.ndarray:
    """complex64 input draw matching the board agent's _seeded_complex64.

    Same float64 standard_normal pair as ``_random_complex`` cast to
    complex64 — keeps the FP32 baseline workloads consuming bit-identical
    inputs to their SPF-32 counterparts (e.g. spf32_mul ↔ fp32_mul) so
    the customer demo can frame both as comparable hardware runs.
    """
    return _random_complex(rng, n).astype(np.complex64)


def generate_fp32_mul_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """fp32_mul workload: element-wise FP32 complex multiply.

    Reference is np.complex128 truth cast to complex64 — backends
    (libfp32ref host or board) graded under the ``fp32_complex_strict``
    envelope. Inputs match the spf32_mul draw cast to complex64.
    """
    rng = np.random.default_rng(seed)
    a = _random_complex64(rng, n)
    b = _random_complex64(rng, n)
    result = (a.astype(np.complex128) * b.astype(np.complex128)).astype(np.complex64)
    return {"result": result}


def _generate_fp32_fft_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """Shared N-point forward FP32 DFT reference for fp32_fft{64,256,1024}.

    Reference is numpy.fft on np.complex128 cast to complex64 — backends
    graded under the ``fp32_complex_loose`` envelope. Stage-accumulated
    drift across libfp32ref's radix-2 DIT plus libm sin/cos differences
    fits inside the budget.
    """
    rng = np.random.default_rng(seed)
    x = _random_complex64(rng, n)
    spectrum = np.fft.fft(x.astype(np.complex128)).astype(np.complex64)
    return {"spectrum": spectrum}


def generate_fp32_fft64_reference(seed: int, n: int = 64) -> dict[str, np.ndarray]:
    """fp32_fft64 workload: 64-point FP32 forward DFT."""
    return _generate_fp32_fft_reference(seed, n)


def generate_fp32_fft256_reference(seed: int, n: int = 256) -> dict[str, np.ndarray]:
    """fp32_fft256 workload: 256-point FP32 forward DFT."""
    return _generate_fp32_fft_reference(seed, n)


def generate_fp32_fft1024_reference(seed: int, n: int = 1024) -> dict[str, np.ndarray]:
    """fp32_fft1024 workload: 1024-point FP32 forward DFT."""
    return _generate_fp32_fft_reference(seed, n)


def generate_liquid_fft_reference(seed: int, n: int = 64) -> dict[str, np.ndarray]:
    """liquid_fft workload: N-point FP32 forward DFT.

    liquid-dsp is the customer-facing FP32 baseline for the SPF-32
    comparison demo (``domains/dsp/liquid-dsp-comparison/``). Truth
    here is numpy.fft on complex128 cast to complex64 — identical to
    ``fp32_fft64`` so a single comparator grades both libfp32ref's and
    liquid-dsp's libliquid output against the same envelope.
    """
    return _generate_fp32_fft_reference(seed, n)


def _pack_words(l_code: np.ndarray, theta: np.ndarray) -> np.ndarray:
    l_u16 = l_code.astype(np.int16).astype(np.uint16).astype(np.uint32)
    theta_u16 = theta.astype(np.uint16).astype(np.uint32)
    return np.ascontiguousarray((l_u16 << 16) | theta_u16, dtype=np.uint32)


def generate_radar_sparse_doppler_reference(
    seed: int,
    n_range_bins: int,
    n_chirps: int,
    detection_stride: int,
    threshold_log2_power_q78: int,
) -> dict[str, np.ndarray]:
    """radar_sparse_doppler workload: sparse threshold + phase Doppler path."""
    rng = np.random.default_rng(seed)
    l_code = np.zeros((n_range_bins, n_chirps), dtype=np.int16)
    theta0 = rng.integers(0, 0x10000, size=(n_range_bins, 1), dtype=np.uint16)
    slope = rng.integers(0, 0x1000, size=(n_range_bins, 1), dtype=np.uint16)
    chirps = np.arange(n_chirps, dtype=np.uint32)[None, :]
    theta = (theta0.astype(np.uint32) + slope.astype(np.uint32) * chirps).astype(
        np.uint16
    )
    l_code[::detection_stride, :] = 512
    result = sparse_doppler(
        _pack_words(l_code, theta).ravel(),
        n_range_bins=n_range_bins,
        n_chirps=n_chirps,
        threshold_log2_power_q78=threshold_log2_power_q78,
    )
    return {
        "detections": result.detections,
        "doppler_theta": result.doppler_theta,
        "confidence": result.confidence,
        "fallback_mask": result.fallback_mask,
    }


def generate_aoa_pair_lut_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """aoa_pair_lut workload: pairwise phase delta plus native angle LUT."""
    rng = np.random.default_rng(seed)
    theta_a = rng.integers(0, 0x10000, size=n, dtype=np.uint16)
    delta = rng.integers(0, 0x10000, size=n, dtype=np.uint16)
    theta_b = (theta_a.astype(np.uint32) + delta.astype(np.uint32)).astype(np.uint16)
    l_a = np.zeros(n, dtype=np.int16)
    l_b = np.zeros(n, dtype=np.int16)
    l_b[::17] = -512
    lut = np.linspace(-90, 90, 256, dtype=np.int16)
    result = estimate_pair(_pack_words(l_a, theta_a), _pack_words(l_b, theta_b), lut)
    return {
        "phase_delta": result.phase_delta,
        "angle": result.angle,
        "valid": result.valid,
        "confidence": result.confidence,
    }


def generate_beamform_fb7_reduce_reference(
    seed: int,
    n_antennas: int,
    n_samples: int,
) -> dict[str, np.ndarray]:
    """beamform_fb7_reduce workload: fused gain/phase apply plus FB7 reduce."""
    rng = np.random.default_rng(seed)
    l_code = rng.integers(-256, 512, size=(n_antennas, n_samples), dtype=np.int16)
    theta = rng.integers(0, 0x10000, size=(n_antennas, n_samples), dtype=np.uint16)
    gain = rng.integers(-64, 65, size=n_antennas, dtype=np.int16)
    phase = rng.integers(0, 0x10000, size=n_antennas, dtype=np.uint16)
    result = beamform_reduce(
        _pack_words(l_code, theta),
        gain_l_offsets=gain,
        phase_offsets=phase,
        mode="accurate",
    )
    assert result.log_power_q78 is not None
    return {"beam_words": result.beam_words, "log_power_q78": result.log_power_q78}


def generate_rf_native_ingress_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """rf_native_ingress workload: packed polar source to SPFBuffer words."""
    rng = np.random.default_rng(seed)
    l_code = rng.integers(-1024, 1024, size=n, dtype=np.int16)
    theta = rng.integers(0, 0x10000, size=n, dtype=np.uint16)
    polar = PolarRFBuffer.from_l_theta(l_code, theta, sample_rate_hz=1.0e6)
    buf = polar.to_spf_buffer()
    return {"words": buf.words, "log_power_q78": c_spf32.log_power(buf.words)}



def generate_encode_fast_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """encode_fast workload: accurate spf32.encode() as the golden reference.

    The c backend (spf32_encode_fast_array) is graded against this golden
    under spf32_encode_fast_acceptance (ΔL ≤ 4 ULP, ΔΘ ≤ 27 ticks).
    """
    rng = np.random.default_rng(seed)
    z = _random_complex(rng, n)
    result = spf32.encode(z)
    return {"result": result}


def generate_spf16_add_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf16_add workload: element-wise SPF-16 addition."""
    from spf.scalar import spf16

    rng = np.random.default_rng(seed)
    a_c = _random_complex(rng, n)
    b_c = _random_complex(rng, n)
    a_w = spf16.encode(a_c)
    b_w = spf16.encode(b_c)
    result = spf16.add(a_w, b_w)
    return {"result": result}


def generate_spf64_add_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """spf64_add workload: element-wise SPF-64 addition."""
    from spf.scalar import spf64

    rng = np.random.default_rng(seed)
    a_c = _random_complex(rng, n)
    b_c = _random_complex(rng, n)
    a_w = spf64.encode(a_c)
    b_w = spf64.encode(b_c)
    result = spf64.add(a_w, b_w)
    return {"result": result}


def generate_fdwm_c_forward_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """fdwm_c_forward workload: independent dense circulant golden."""
    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    result = (_build_circulant(c) @ x).astype(np.float32)
    return {"result": result}


def run_fdwm_c_forward_python_backend(seed: int, n: int) -> dict[str, np.ndarray]:
    """fdwm_c_forward workload: execute the optimized FDWM forward path."""
    from spf.fdwm import fdwm_c_forward

    rng = np.random.default_rng(seed)
    c = rng.standard_normal(n).astype(np.float32)
    x = rng.standard_normal(n).astype(np.float32)
    c_hat = np.fft.fft(c).astype(np.complex64)
    result = fdwm_c_forward(x, c_hat).astype(np.float32)
    return {"result": result}


def generate_sb_core_reference(seed: int, n: int) -> dict[str, np.ndarray]:
    """sb_core workload: FP64 Sb(ΔL, ΔΘ) over a random sweep.

    Inputs are sampled from ``ΔL ∈ [0, 8)`` and ``ΔΘ ∈ (−π, π]``, the
    range covered by the runtime Sb() LUT. Outputs are ``(Δ′L, Δ′Θ)`` in
    log₂ / radian units.
    """
    rng = np.random.default_rng(seed)
    dl = rng.uniform(0.0, 8.0, size=n).astype(np.float64)
    dtheta = rng.uniform(-math.pi, math.pi, size=n).astype(np.float64)
    delta_l, delta_theta = sb_exact(dl, dtheta)
    return {
        "delta_l": np.ascontiguousarray(delta_l),
        "delta_theta": np.ascontiguousarray(delta_theta),
    }
