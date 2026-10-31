"""Workload reference entrypoints.

Each function here is the Python source-of-truth for a workload contract
under ``contracts/workloads/<name>.yaml``. It takes the case's parameters,
builds inputs deterministically from the case's ``seed`` where the inputs
are random arrays, and returns the named output tensors a golden artifact
is composed of.

Keep these pure reference math — no I/O, no logging. The generator
script and Kria host runner both invoke them as library calls.
"""

from spf.workloads.references import (
    generate_abs2_reference,
    generate_add_reference,
    generate_aoa_pair_lut_reference,
    generate_beamform_fb7_reduce_reference,
    generate_butterfly_reference,
    generate_conj_reference,
    generate_cyclic_conv32_reference,
    generate_div_reference,
    generate_encode_fast_reference,
    generate_fdwm_c_forward_reference,
    generate_fft64_fb7_fast_reference,
    generate_fft64_fb7_fields_fast_reference,
    generate_fft64_fb7_fields_reference,
    generate_fft64_fb7_reference,
    generate_fft64_reference,
    generate_fft256_reference,
    generate_fft1024_reference,
    generate_fp32_fft64_reference,
    generate_fp32_fft256_reference,
    generate_fp32_fft1024_reference,
    generate_fp32_mul_reference,
    generate_liquid_fft_reference,
    generate_mul_reference,
    generate_neg_reference,
    generate_pe_reference,
    generate_radar_sparse_doppler_reference,
    generate_rf_native_ingress_reference,
    generate_sb_core_reference,
    generate_spf16_add_reference,
    generate_spf64_add_reference,
    generate_twiddle_reference,
    run_fdwm_c_forward_python_backend,
)

__all__ = [
    "generate_abs2_reference",
    "generate_add_reference",
    "generate_aoa_pair_lut_reference",
    "generate_beamform_fb7_reduce_reference",
    "generate_butterfly_reference",
    "generate_conj_reference",
    "generate_cyclic_conv32_reference",
    "generate_div_reference",
    "generate_encode_fast_reference",
    "generate_fdwm_c_forward_reference",
    "generate_fft64_fb7_fast_reference",
    "generate_fft64_fb7_fields_fast_reference",
    "generate_fft64_fb7_fields_reference",
    "generate_fft64_fb7_reference",
    "generate_fft64_reference",
    "generate_fft256_reference",
    "generate_fft1024_reference",
    "generate_fp32_fft64_reference",
    "generate_fp32_fft256_reference",
    "generate_fp32_fft1024_reference",
    "generate_fp32_mul_reference",
    "generate_liquid_fft_reference",
    "generate_mul_reference",
    "generate_neg_reference",
    "generate_pe_reference",
    "generate_radar_sparse_doppler_reference",
    "generate_rf_native_ingress_reference",
    "generate_sb_core_reference",
    "generate_spf16_add_reference",
    "generate_spf64_add_reference",
    "generate_twiddle_reference",
    "run_fdwm_c_forward_python_backend",
]
