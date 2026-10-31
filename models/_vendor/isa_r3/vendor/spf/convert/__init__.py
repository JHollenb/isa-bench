"""Cross-format conversion paths (spec §11, Task 1.8).

Single namespace collecting every boundary-crossing path a real inference
pipeline needs:

- ``fp32_to_spf32`` / ``spf32_to_fp32`` — SPF-32 scalar encode/decode.
- ``fp16_to_spf16`` / ``spf16_to_fp16`` — SPF-16 scalar encode/decode.
- ``fp16_to_sbf128`` / ``sbf128_to_fp16`` — SBF-128 block encode/decode.
- ``sbf128_from_spectrum`` — build an SBF-128 block from a precomputed
  length-65 spectrum, skipping rfft (T6.1 Path B).
- ``dense_to_fdwm_c`` / ``dense_to_fdwm_bc`` — nearest-circulant projection
  of a dense weight matrix (spec §7.4, §8).
- ``spf32_to_fnt32`` / ``fnt32_to_spf32`` — real-projecting bridge between
  SPF-32 complex words and integer residues in ``Z_{2^32+1}``.

Re-exports reuse the existing format implementations; only the SPF-32 ↔
FNT-32 bridge and ``sbf128_from_spectrum`` are new logic.
"""

from spf.convert.dense_fdwm import dense_to_fdwm_bc, dense_to_fdwm_c
from spf.convert.fp16_sbf128 import fp16_to_sbf128, sbf128_from_spectrum, sbf128_to_fp16
from spf.convert.fp16_spf16 import fp16_to_spf16, spf16_to_fp16
from spf.convert.fp32_spf32 import fp32_to_spf32, spf32_to_fp32
from spf.convert.spf32_fnt32 import fnt32_to_spf32, spf32_to_fnt32
from spf.convert.spf32_spf16 import spf16_to_spf32, spf32_to_spf16

__all__ = [
    "dense_to_fdwm_bc",
    "dense_to_fdwm_c",
    "fnt32_to_spf32",
    "fp16_to_sbf128",
    "fp16_to_spf16",
    "fp32_to_spf32",
    "sbf128_from_spectrum",
    "sbf128_to_fp16",
    "spf16_to_fp16",
    "spf16_to_spf32",
    "spf32_to_fnt32",
    "spf32_to_fp32",
    "spf32_to_spf16",
]
