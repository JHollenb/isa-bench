"""Matched filter controls around the production SFR filter API."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy import fft  # type: ignore[import-untyped]
from scipy.ndimage import zoom  # type: ignore[import-untyped]

from spf.renderer import filter_chain as fc
from spf.tensor import core

codec: Any = fc  # Runtime hooks intentionally observe the production codec boundary.

VARIANTS = (
    "fp32-eager",
    "fp32-fused",
    "fp32-optimized",
    "spf-eager",
    "spf-dense-matched",
    "spf-dense",
    "sfr-fused",
    "fp32-prepared",
    "fp32-precomposed",
    "fp64-precomposed",
    "sfr-prepared",
    "sfr-fused-fp32",
    "sfr-prepared-fp32",
    "fp16-storage",
)


@dataclass(frozen=True)
class FilterSpec:
    translate: tuple[float, float] = (0.37, 0.22)
    blur: float = 8.0
    exposure: float = 1.5
    glare: str = "star6"
    glare_strength: float = 0.15

    @property
    def covariance(self) -> np.ndarray:
        theta = np.pi / 6
        rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        covariance = rotation @ np.diag([self.blur**2, (self.blur / 2) ** 2]) @ rotation.T
        return cast(np.ndarray, (covariance + covariance.T) * 0.5)

    def chain(self, source: Any, *, compute_dtype: str | None = None) -> fc.FilterChain:
        chain = (
            fc.FilterChain.from_source(source, compute_dtype=compute_dtype)
            .translate(*self.translate)
            .gaussian_filter(self.covariance)
        )
        chain = chain.exposure(self.exposure)
        return chain.glare(strength=self.glare_strength) if self.glare == "star6" else chain


def fixture(name: str, shape: tuple[int, int], seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    input_meta = {}
    y = np.arange(shape[0], dtype=np.float32)[:, None] / shape[0]
    x = np.arange(shape[1], dtype=np.float32)[None, :] / shape[1]
    if name == "smooth":
        source = 0.8 + 0.3 * np.cos(2 * np.pi * (3 * x + y)) + 0.15 * np.sin(2 * np.pi * (x - 2 * y))
    elif name == "sparse":
        source = np.zeros(shape, dtype=np.float32)
        for yy, xx, value in [(0.2, 0.3, 80), (0.7, 0.6, 120), (0.4, 0.8, 40)]:
            source[int(yy * shape[0]), int(xx * shape[1])] = value
    elif name == "chaotic":
        source = np.random.default_rng(seed).lognormal(-0.5, 1.2, shape).astype(np.float32)
    else:
        import cv2  # type: ignore[import-not-found,unused-ignore]

        decoded = cv2.imread(name, cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3 or decoded.dtype != np.float32:
            raise ValueError("HDR input must decode to a floating RGB image")
        source = np.asarray(decoded, dtype=np.float32)
        input_meta = {
            "source_file_sha256": hashlib.sha256(Path(name).read_bytes()).hexdigest(),
            "opencv": cv2.__version__,
            "decoded_shape": list(source.shape),
        }
        source = source[..., 0] * 0.0722 + source[..., 1] * 0.7152 + source[..., 2] * 0.2126
        source = zoom(source, (shape[0] / source.shape[0], shape[1] / source.shape[1]), order=1)
        # One source-derived normalization, shared by all variants and references.
        source /= max(float(np.mean(source)), 1e-20)
    source = np.asarray(source, dtype=np.float32)
    if source.shape != shape or not np.all(np.isfinite(source)):
        raise ValueError("fixture does not match its finite shape contract")
    return source, {
        **input_meta,
        "input": name,
        "seed": seed,
        "channels": 1,
        "interpretation": "periodic scalar field; HDR inputs use linear Rec.709 luminance, bilinear resize, unit mean",
    }


def independent_transfer(shape: tuple[int, int], spec: FilterSpec) -> np.ndarray:
    """FP64 oracle, intentionally independent of production transfer construction."""
    ky, kx = np.fft.fftfreq(shape[0])[:, None], np.fft.fftfreq(shape[1])[None, :]
    covariance = spec.covariance
    quadratic = covariance[0, 0] * kx**2 + 2 * covariance[0, 1] * kx * ky + covariance[1, 1] * ky**2
    result = np.exp(-2j * np.pi * (kx * spec.translate[0] + ky * spec.translate[1]))
    result *= np.exp(-2 * np.pi**2 * quadratic) * 2**spec.exposure
    if spec.glare == "star6":
        glare = np.full(shape, 1 - spec.glare_strength)
        for angle in [0, np.pi / 3, 2 * np.pi / 3]:
            a = kx * np.cos(angle) + ky * np.sin(angle)
            b = -kx * np.sin(angle) + ky * np.cos(angle)
            glare += spec.glare_strength / 3 * np.exp(-2 * np.pi**2 * (24**2 * a * a + b * b))
        result *= glare
    return result


def reference(source: np.ndarray, spec: FilterSpec) -> np.ndarray:
    return cast(
        np.ndarray,
        fft.ifftn(
            fft.fftn(source.astype(np.complex128), workers=1) * independent_transfer(source.shape, spec), workers=1
        ),
    )


def transfer_fp32(shape: tuple[int, int], spec: FilterSpec) -> np.ndarray:
    """Independent compact FP32 construction; avoids production FP64 scratch grids."""
    kx = fft.fftfreq(shape[1]).astype(np.float32)[None, :]
    ky = fft.fftfreq(shape[0]).astype(np.float32)[:, None]
    covariance = spec.covariance.astype(np.float32)
    quadratic = covariance[0, 0] * kx * kx + np.float32(2) * covariance[0, 1] * kx * ky
    quadratic += covariance[1, 1] * ky * ky
    magnitude = np.exp(np.float32(-2 * np.pi**2) * quadratic)
    magnitude *= np.float32(2**spec.exposure)
    phase = np.float32(-2 * np.pi) * (kx * np.float32(spec.translate[0]) + ky * np.float32(spec.translate[1]))
    transfer = np.empty(shape, dtype=np.complex64)
    transfer.real = magnitude * np.cos(phase)
    transfer.imag = magnitude * np.sin(phase)
    if spec.glare == "star6":
        glare = np.full(shape, 1 - spec.glare_strength, dtype=np.float32)
        for angle in [0, np.pi / 3, 2 * np.pi / 3]:
            a = kx * np.float32(np.cos(angle)) + ky * np.float32(np.sin(angle))
            b = -kx * np.float32(np.sin(angle)) + ky * np.float32(np.cos(angle))
            glare += np.float32(spec.glare_strength / 3) * np.exp(
                np.float32(-2 * np.pi**2) * (np.float32(576) * a * a + b * b)
            )
        transfer *= glare
    return cast(np.ndarray, transfer)


def make_runner(
    source: np.ndarray, spec: FilterSpec, variant: str
) -> tuple[Callable[[], np.ndarray], dict[str, Any], tuple[np.ndarray, ...]]:
    """Fresh variants construct operator state inside the measured call."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    meta: dict[str, Any] = {
        "scope": "fresh-public-call",
        "implementation": "production FilterChain",
        "hardware_native_spf": False,
        "analytic_sublinear": False,
        "spf_block_tensor_matrix_used": False,
    }
    if variant.startswith("sfr-"):
        meta["implementation"] = "production compact analytic FilterChain with explicit dense FFT boundary"
        meta["representation"] = "analytic_transfer_dense_source"
        meta["compute_dtype"] = "complex64" if variant.endswith("-fp32") else "complex128"
    if variant == "fp32-optimized":

        def run_optimized() -> np.ndarray:
            transfer = transfer_fp32(source.shape, spec)
            return cast(np.ndarray, fft.ifftn(fft.fftn(source.astype(np.complex64), workers=1) * transfer, workers=1))

        meta["implementation"] = "independent compact FP32 transfer construction and SciPy complex64 FFT"
        meta["scope"] = "fresh-kernel-control"
        return run_optimized, meta, ()
    if variant in {"fp32-precomposed", "fp64-precomposed"}:
        import time

        start = time.perf_counter_ns()
        dtype = np.complex64 if variant == "fp32-precomposed" else np.complex128
        transfer = (
            transfer_fp32(source.shape, spec) if dtype == np.complex64 else independent_transfer(source.shape, spec)
        )
        spectrum = fft.fftn(source.astype(dtype), workers=1) * transfer
        spectrum.setflags(write=False)
        del transfer

        def run_precomposed() -> np.ndarray:
            return cast(np.ndarray, fft.ifftn(spectrum, workers=1))

        meta.update(
            implementation="independent fully precomposed spectral prefix and fresh SciPy inverse FFT",
            scope="prepared-execution",
            spectral_prefix_hoisted=True,
            cached_terminal_values=0,
            prepare_ms=(time.perf_counter_ns() - start) / 1e6,
            prepared_state_bytes=spectrum.nbytes,
        )
        return run_precomposed, meta, (spectrum,)
    if variant in {"fp32-prepared", "sfr-prepared", "sfr-prepared-fp32"}:
        import time

        from spf.renderer.analytic_filter_chain import PreparedAnalyticFilterChain

        start = time.perf_counter_ns()
        chain = spec.chain(source, compute_dtype="complex64" if variant.endswith("-fp32") else None)
        arrays: tuple[np.ndarray, ...]
        extra_state_bytes = 0
        if variant == "fp32-prepared":
            spectrum = fft.fftn(source.astype(np.complex64), workers=1)
            transfer = chain.fused_transfer().transfer.astype(np.complex64)

            def run() -> np.ndarray:
                return cast(np.ndarray, fft.ifftn(spectrum * transfer, workers=1))

            arrays = (spectrum, transfer)
        else:
            prepared = chain.prepare()
            assert isinstance(prepared, PreparedAnalyticFilterChain)

            def run() -> np.ndarray:
                return cast(np.ndarray, prepared.execute().values)

            arrays = (prepared.chain.source, prepared.spectrum, prepared.native.coefficients, prepared.native.seeds)
            meta["spectral_prefix_hoisted"] = prepared.hoisted
            extra_state_bytes = prepared.transfer.nbytes + prepared.native.groups.nbytes
        meta.update(
            scope="prepared-execution",
            prepare_ms=(time.perf_counter_ns() - start) / 1e6,
            prepared_state_bytes=sum(a.nbytes for a in arrays) + extra_state_bytes,
        )
        return run, meta, arrays
    if variant == "sfr-fused-fp32":
        return lambda: spec.chain(source, compute_dtype="complex64").run().values, meta, ()
    if variant in {"spf-eager", "spf-dense-matched"}:

        def run_spf() -> np.ndarray:
            chain = spec.chain(source)
            if variant == "spf-dense-matched":
                spectrum = fft.fftn(source.astype(np.complex64), workers=1)
                words = core.spf_mul_words(
                    codec.encode_words(spectrum), codec.encode_words(chain.fused_transfer().transfer)
                )
                return cast(np.ndarray, fft.ifftn(codec.decode_words(words).astype(np.complex64), workers=1))
            current = source.astype(np.complex64)
            for operation in chain.operations:
                spectrum = fft.fftn(current, workers=1)
                words = core.spf_mul_words(
                    codec.encode_words(spectrum), codec.encode_words(chain._single_transfer(operation))
                )
                current = fft.ifftn(codec.decode_words(words).astype(np.complex64), workers=1)
            return cast(np.ndarray, current)

        meta["implementation"] = "benchmark control using production array codec/multiply and SciPy complex64 FFT"
        return run_spf, meta, ()
    if variant == "fp16-storage":

        def run_half() -> np.ndarray:
            chain = spec.chain(source)
            transfer = chain.fused_transfer().transfer
            transfer = transfer.real.astype(np.float16).astype(np.float32) + 1j * transfer.imag.astype(
                np.float16
            ).astype(np.float32)
            return cast(
                np.ndarray,
                fft.ifftn(fft.fftn(source.astype(np.float16).astype(np.complex64), workers=1) * transfer, workers=1),
            )

        meta["implementation"] = (
            "FP16 input/transfer storage control; FFT and arithmetic complex64, not native FP16 compute"
        )
        return run_half, meta, ()
    mapping = {
        "fp32-eager": fc.FilterVariant.EAGER_FP32,
        "fp32-fused": fc.FilterVariant.FUSED_FP32,
        "spf-dense": fc.FilterVariant.DENSE_SPF,
        "sfr-fused": fc.FilterVariant.STRUCTURED,
    }
    return lambda: spec.chain(source).run(mapping[variant]).values, meta, ()
