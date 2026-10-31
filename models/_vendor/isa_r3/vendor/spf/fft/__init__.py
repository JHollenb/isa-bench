"""SPF Fast Fourier Transform.

Radix-2 Cooley-Tukey (decimation-in-time) FFT driver for every SPF scalar
format. Default public surface (:func:`fft`, :func:`ifft`) is the SPF-32
binding, preserved for backward compatibility. Per-format entry points:

- SPF-16: :func:`fft_spf16`, :func:`ifft_spf16` (N ≤ 256).
- SPF-32: :func:`fft_spf32`, :func:`ifft_spf32` (N ≤ 1024; same as ``fft``).
- SPF-64: :func:`fft_spf64`, :func:`ifft_spf64` (N ≤ 1024).

FB7 fused-butterfly surface (Phase 2 promotion, normative since Phase 5):

- :func:`spf_fft_fused` — pure-Python radix-2 FFT using the FB7 butterfly.
- :func:`spf_butterfly_fb7_fast` — poly7 half-angle, G=8 dominant-gap bypass.
- :func:`spf_butterfly_fb7_balanced` — poly7 half-angle, G=12 dominant-gap bypass.
- :func:`spf_butterfly_fb7_accurate` — poly7 half-angle, no bypass.
- :class:`SPF` / :func:`pack_spf32` / :func:`unpack_spf32` — scalar helpers
  for building inputs and reading outputs.
- :mod:`spf.fft.fused` — full FB7 module (encode/decode overlays, etc.).
- :mod:`spf.fft.fb7_fields` — field-mode FFT primitives.

The explicit v0.2 profile is available as :func:`fft_spf32_v02` and
:func:`ifft_spf32_v02`; the default ``fft_spf32``/``ifft_spf32`` names retain
the legacy FB7 regression path.  The opt-in factored tan-tanh profile is
:func:`fft_spf32_v02_factored` / :func:`ifft_spf32_v02_factored` (also
reachable as ``fft(x, profile="v02_factored")``) — a CANDIDATE adder, not the
conforming path; see :mod:`spf.fft.factored`.

Spec: docs/spf-formats-and-test-plans.md §12 T1.4.
"""

from typing import Any

from spf.fft.fused import (
    SPF,
    decode_complex,
    decode_complex_fast,
    decode_complex_reference,
    decode_vector,
    encode_complex,
    encode_complex_fast,
    encode_complex_reference,
    encode_vector,
    pack_spf32,
    pack_words,
    spf_butterfly_fb7_accurate,
    spf_butterfly_fb7_balanced,
    spf_butterfly_fb7_fast,
    spf_butterfly_reference,
    spf_fft_fused,
    spf_fft_reference,
    unpack_spf32,
    unpack_words,
)
from spf.fft.spf16_fft import fft as fft_spf16
from spf.fft.spf16_fft import ifft as ifft_spf16
from spf.fft.spf32_fft import fft, ifft
from spf.fft.spf32_fft import fft as fft_spf32
from spf.fft.spf32_fft import ifft as ifft_spf32
from spf.fft.spf64_fft import fft as fft_spf64
from spf.fft.spf64_fft import ifft as ifft_spf64
from spf.tensor.v02 import spf_fft_radix2 as fft_spf32_v02
from spf.tensor.v02 import spf_ifft_radix2 as ifft_spf32_v02

__all__ = [
    "SPF",
    "decode_complex",
    "decode_complex_fast",
    "decode_complex_reference",
    "decode_vector",
    "encode_complex",
    "encode_complex_fast",
    "encode_complex_reference",
    "encode_vector",
    "fft",
    "fft_spf16",
    "fft_spf32",
    "fft_spf32_v02",
    "fft_spf32_v02_factored",
    "fft_spf64",
    "ifft",
    "ifft_spf16",
    "ifft_spf32",
    "ifft_spf32_v02",
    "ifft_spf32_v02_factored",
    "ifft_spf64",
    "pack_spf32",
    "pack_words",
    "spf_butterfly_fb7_accurate",
    "spf_butterfly_fb7_balanced",
    "spf_butterfly_fb7_fast",
    "spf_butterfly_reference",
    "spf_fft_fused",
    "spf_fft_reference",
    "unpack_spf32",
    "unpack_words",
]


def __getattr__(name: str) -> Any:  # PEP 562: keep the factored ROM off the import path
    if name == "fft_spf32_v02_factored":
        from spf.fft.factored import spf_fft_radix2_factored

        return spf_fft_radix2_factored
    if name == "ifft_spf32_v02_factored":
        from spf.fft.factored import spf_ifft_radix2_factored

        return spf_ifft_radix2_factored
    raise AttributeError(f"module 'spf.fft' has no attribute {name!r}")
