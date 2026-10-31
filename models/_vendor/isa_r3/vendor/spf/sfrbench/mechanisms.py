"""Matched ocean, volume, reducer, and block probes with explicit scope labels."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy import fft  # type: ignore[import-untyped]

from spf import renderer as sfr
from spf.renderer.carriers import decode_words, encode_words
from spf.renderer.separable_terminal import _rank_reduce
from spf.renderer.transport_terminal import reduce_logpos
from spf.renderer.volume import _reduce_transport_arrays
from spf.tensor import core, v02
from spf.tensor.v2 import ActiveEncoding, TensorBlock, TensorBlockHeaderV2  # type: ignore[attr-defined]

ROW_LOG: Path | None = None


def measure(run: Any, reference: Any, *, repeats: int, reset: Any = None, **metadata: Any) -> Any:
    try:
        row = _measure(run, reference, repeats=repeats, reset=reset, **metadata)
    except Exception as exc:
        row = {**metadata, "mechanics_status": "execution_failed", "reason": repr(exc)}
    if ROW_LOG is not None:
        with ROW_LOG.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    return row


def _measure(run: Any, reference: Any, *, repeats: int, reset: Any = None, **metadata: Any) -> Any:
    if reset:
        reset()
    run()
    samples = []
    for _ in range(repeats):
        if reset:
            reset()
        start = time.perf_counter_ns()
        value = run()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    value = np.asarray(value)
    expected = np.asarray(reference)
    if value.shape != expected.shape or not np.all(np.isfinite(value)):
        raise ValueError("mechanism output violated finite shape contract")
    if np.issubdtype(value.dtype, np.integer):
        if not np.array_equal(value, expected):
            raise ValueError("integer state or packed-code parity failed")
        return {
            **metadata,
            "runtime_ms": statistics.median(samples),
            "samples_ms": samples,
            "state_exact_match": True,
            "precision": str(value.dtype),
            "mechanics_status": "valid",
            "relative_l2": 0.0,
            "max_abs_error": 0.0,
            "trend_status": "observation",
            "terminal_status": "not_certified",
        }
    norm = np.linalg.norm(expected.ravel())
    return {
        **metadata,
        "runtime_ms": statistics.median(samples),
        "samples_ms": samples,
        "relative_l2": float(np.linalg.norm((value - expected).ravel()) / norm)
        if norm
        else float(np.linalg.norm(value)),
        "max_abs_error": float(np.max(np.abs(value - expected))),
        "precision": str(value.dtype),
        "bytes_read": None,
        "bytes_written": None,
        "energy_j": None,
        "mechanics_status": "valid",
        "trend_status": "observation",
        "terminal_status": "not_certified",
    }


def phase_view(state: Any) -> Any:
    top = state >> np.uint64(48)
    rem = state & np.uint64((1 << 48) - 1)
    midpoint = np.uint64(1 << 47)
    return ((top + ((rem > midpoint) | ((rem == midpoint) & ((top & np.uint64(1)) != 0)))) & np.uint64(65535)).astype(
        np.uint32
    )


def ocean(size: Any, repeats: Any) -> Any:
    rng = np.random.default_rng(20260913)
    half = fft.rfftn(rng.normal(size=(size, size)).astype(np.float32))
    half[:, 0] = 0
    half[:, -1] = 0  # exclude endpoint modes in this evolving-spectrum fixture
    words = encode_words(half)
    initial = decode_words(words).astype(np.complex64)
    initial_phase = (words & 65535).astype(np.uint64) << np.uint64(48)
    signed = rng.integers(-(1 << 57), 1 << 57, half.shape, dtype=np.int64)
    increment = signed.view(np.uint64)
    angle = signed.astype(np.float64) * (2 * np.pi / 2**64)
    rotator = np.exp(1j * angle).astype(np.complex64)
    phase, state = initial_phase.copy(), initial.copy()
    zero = words == v02.SPF32_WORD_ZERO

    def reset() -> Any:
        phase[:] = initial_phase
        state[:] = initial

    def fp_update() -> Any:
        np.multiply(state, rotator, out=state)
        return state

    def spf_update() -> Any:
        np.add(phase, increment, out=phase)
        return phase

    def emit() -> Any:
        return np.where(zero, v02.SPF32_WORD_ZERO, (words & np.uint32(0xFFFF0000)) | phase_view(phase)).astype(
            np.uint32
        )

    expected = initial.astype(np.complex128) * np.exp(1j * angle)
    rows = [
        measure(
            fp_update,
            expected,
            repeats=repeats,
            reset=reset,
            workload="ocean-update",
            variant="fp32-complex-multiply",
            modes=half.size,
        )
    ]
    # Integer state correctness is measured against the exact modular recurrence.
    exact_ticks = initial_phase + increment
    rows.append(
        measure(
            spf_update,
            exact_ticks,
            repeats=repeats,
            reset=reset,
            workload="ocean-update",
            variant="spf-wide-u64-add",
            modes=half.size,
        )
    )
    rows[-1]["quality_scope"] = "exact integer state parity; decoded field quality is reported separately"
    frequency = fft.rfftfreq(size).astype(np.float32)[None, :]
    frequency[:, -1] = 0
    multiplier = np.asarray(2j * np.pi * frequency, dtype=np.complex64)
    delta = encode_words(np.broadcast_to(multiplier, half.shape))
    derivative_reference = initial.astype(np.complex128) * multiplier.astype(np.complex128)
    rows.append(
        measure(
            lambda: initial * multiplier,
            derivative_reference,
            repeats=repeats,
            workload="ocean-derivative",
            variant="fp32-diagonal",
            modes=half.size,
        )
    )
    # Packed derivative output is decoded outside the timed multiply.
    packed_reference = core.spf_mul_words(words, delta)
    rows.append(
        measure(
            lambda: core.spf_mul_words(words, delta),
            packed_reference,
            repeats=repeats,
            workload="ocean-derivative",
            variant="spf-packed-diagonal",
            modes=half.size,
        )
    )
    rows[-1]["decoded_relative_l2"] = float(
        np.linalg.norm(decode_words(packed_reference) - derivative_reference) / np.linalg.norm(derivative_reference)
    )

    def fp_total() -> Any:
        fp_update()
        return fft.irfftn(state, s=(size, size), workers=1)

    def spf_total() -> Any:
        spf_update()
        return sfr.synthesize_ocean((size, size), half_spectrum=emit(), method="irfft")

    image_reference = fft.irfftn(expected, s=(size, size), workers=1)
    rows.append(
        measure(
            fp_total,
            image_reference,
            repeats=repeats,
            reset=reset,
            workload="ocean-realization",
            variant="fp32-update-and-ifft",
            modes=half.size,
        )
    )
    rows.append(
        measure(
            spf_total,
            image_reference,
            repeats=repeats,
            reset=reset,
            workload="ocean-realization",
            variant="spf-update-emit-public-sfr",
            modes=half.size,
        )
    )
    temporal = []
    reset()
    for frame in range(1, 257):
        fp_update()
        spf_update()
        if frame in {1, 8, 64, 256}:
            oracle = initial.astype(np.complex128) * np.exp(1j * angle * frame)
            fp_error = float(np.linalg.norm(state - oracle) / np.linalg.norm(oracle))
            spf_error = float(np.linalg.norm(decode_words(emit()) - oracle) / np.linalg.norm(oracle))
            temporal.append({"frame": frame, "fp32_relative_l2": fp_error, "spf_relative_l2": spf_error})
    return rows, temporal


def volume(repeats: Any, rays: Any = 256) -> Any:
    rows = []
    for kind in ["homogeneous", "heterogeneous"]:
        for count in [16, 64, 256, 1024]:
            rng = np.random.default_rng(333 + count)
            sigma = (
                np.ones((rays, count), np.float32) * 0.8
                if kind == "homogeneous"
                else rng.uniform(0.1, 1.5, (rays, count)).astype(np.float32)
            )
            ds = np.float32(1 / count)
            attenuation = np.exp(-sigma * ds)
            q8 = np.rint(-sigma.astype(np.float64) * float(ds) / np.log(2) * 256).astype(np.int32)
            q16 = np.rint(-sigma.astype(np.float64) * float(ds) / np.log(2) * 65536).astype(np.int32)
            expected = np.exp(-sigma.astype(np.float64).sum(axis=1) * float(ds))

            def fp_product(attenuation: Any = attenuation) -> Any:
                return np.prod(attenuation, axis=1, dtype=np.float32)

            def fp_depth(sigma: Any = sigma, ds: Any = ds) -> Any:
                return np.exp(-np.sum(sigma, axis=1, dtype=np.float32) * ds)

            def q8_depth(q8: Any = q8) -> Any:
                return np.exp2(np.sum(q8, axis=1, dtype=np.int64) / 256)

            def q16_depth(q16: Any = q16) -> Any:
                return np.exp2(np.sum(q16, axis=1, dtype=np.int64) / 65536)

            runs = [
                ("fp32-product", fp_product),
                ("fp32-optical-depth", fp_depth),
                ("spf-q8-per-segment", q8_depth),
                ("sfr-guard-q16", q16_depth),
            ]
            for variant, run in runs:
                rows.append(
                    measure(
                        run,
                        expected,
                        repeats=repeats,
                        workload="volume-attenuation",
                        field=kind,
                        variant=variant,
                        samples_per_ray=count,
                        rays=rays,
                        scope="prepared attenuation/optical-depth arrays; terminal decode included",
                        implementation="matched NumPy arithmetic probe; guard q16 is not a GPU or public-volume claim",
                    )
                )
            # Ordered emission/attenuation reduction, using production native and LogPos consumers.
            t = attenuation.T.copy()
            c = (1 - t) * np.linspace(0.2, 0.9, rays, dtype=np.float32)[None, :]
            zeros = np.zeros(count)

            def fp_tree(dtype: Any = np.float32, t: Any = t, c: Any = c) -> Any:
                tt, cc = t.astype(dtype), c.astype(dtype)
                while len(tt) > 1:
                    cc = cc[::2] + tt[::2] * cc[1::2]
                    tt = tt[::2] * tt[1::2]
                return cc[0] + tt[0] * 0.1

            expected_color = fp_tree(np.float64)
            logs_t = np.rint(256 * np.log2(t.astype(float))).astype(np.int64)
            logs_c = np.rint(256 * np.log2(c.astype(float))).astype(np.int64)

            def native(t: Any = t, c: Any = c, zeros: Any = zeros) -> Any:
                value = _reduce_transport_arrays(t.astype(float), c.astype(float), zeros, zeros)
                return np.asarray(value.C) + np.asarray(value.T) * 0.1

            def logpos(logs_t: Any = logs_t, logs_c: Any = logs_c, zeros: Any = zeros) -> Any:
                value = reduce_logpos(logs_t, logs_c, zeros, zeros)
                return np.array([x.value for x in value.C]) + np.array([x.value for x in value.T]) * 0.1

            for variant, run in [
                ("fp32-ordered-tree", fp_tree),
                ("sfr-native-tree", native),
                ("sfr-logpos-tree", logpos),
            ]:
                rows.append(
                    measure(
                        run,
                        expected_color,
                        repeats=repeats,
                        workload="volume-scatter-reduction",
                        field=kind,
                        variant=variant,
                        samples_per_ray=count,
                        rays=rays,
                        scope="prepared segment arrays; ordered tree, background and decode included",
                        transport_compositions=rays * (count - 1),
                    )
                )
    return rows


def reducers(repeats: Any, count: Any = 8192) -> Any:
    rows = []
    rng = np.random.default_rng(445)
    for rank in [1, 2, 4, 8, 16, 32]:
        for field in ["coherent", "cancelling"]:
            phase = (
                rng.normal(0, 0.02, (count, rank)) if field == "coherent" else rng.uniform(-np.pi, np.pi, (count, rank))
            )
            a = encode_words(np.exp(1j * phase))
            b = encode_words(rng.uniform(0.5, 1.5, (count, rank)).astype(complex))
            aa, bb = decode_words(a).astype(np.complex64), decode_words(b).astype(np.complex64)
            expected = np.sum(aa.astype(np.complex128) * bb.astype(np.complex128), axis=1)

            def fp(aa: Any = aa, bb: Any = bb) -> Any:
                return np.sum(aa * bb, axis=1, dtype=np.complex64)

            def spf(a: Any = a, b: Any = b) -> Any:
                return decode_words(_rank_reduce(core.spf_mul_words(a, b), "spf_complex")[0])

            for variant, run in [("fp32-multiply-reduce", fp), ("spf-multiply-reduce", spf)]:
                rows.append(
                    measure(
                        run,
                        expected,
                        repeats=repeats,
                        workload="reducer-density",
                        field=field,
                        variant=variant,
                        rank=rank,
                        field_operations=count * rank,
                        reduction_operations=count * (rank - 1),
                        reduction_density=(rank - 1) / rank,
                        scope="prepared matched quantized operands; product + sum + terminal decode",
                        count_scope=(
                            "logical complex products and binary reduction edges; not machine instruction counts"
                        ),
                    )
                )
    return rows


def blocks(size: Any, repeats: Any) -> Any:
    rows = []
    for fraction in [1, 0.25, 0.0625, 0.015625]:
        total = size * size
        active = max(1, int(total * fraction))
        indices = np.arange(active, dtype=np.uint32) * (total // active)
        block = TensorBlock(
            TensorBlockHeaderV2(dims=(size, size), global_L_bias=-512, active_encoding=ActiveEncoding.COO_U32),
            np.zeros(active, np.int16),
            np.zeros(active, np.int16),
            active_indices=indices,
        )
        field = sfr.coherent_intensity(sfr.Field.from_value(block, domain=sfr.Domain.SPATIAL_2D))
        request = sfr.MaterializationRequest(layout=sfr.SampleLayout((size, size)), output_format="real", backend="cpu")
        prepared = field.prepare(request)
        values = np.zeros(total, np.complex64)
        values[indices] = 0.25
        values = values.reshape(size, size)
        expected = values.real**2 + values.imag**2

        def fp_dense(values: Any = values) -> Any:
            return values.real**2 + values.imag**2

        def fp_sparse(total: Any = total, indices: Any = indices) -> Any:
            out = np.zeros(total, np.float32)
            out[indices] = 0.0625
            return out.reshape(size, size)

        def run_block(prepared: Any = prepared) -> Any:
            return np.asarray(prepared.execute())

        def fp_header(total: Any = total, indices: Any = indices, active: Any = active) -> Any:
            if active == total:
                return np.full((size, size), 0.0625, dtype=np.float32)
            out = np.zeros(total, np.float32)
            out[indices] = 0.0625
            return out.reshape(size, size)

        for variant, run in [
            ("fp32-dense-intensity", fp_dense),
            ("fp32-sparse-intensity", fp_sparse),
            ("fp32-header-intensity", fp_header),
            ("sfr-tensorblock-intensity", run_block),
        ]:
            rows.append(
                measure(
                    run,
                    expected,
                    repeats=repeats,
                    workload="block-materialization",
                    variant=variant,
                    active_fraction=fraction,
                    active_elements=active,
                    output_elements=total,
                    support_avoidance=total / active,
                    scope="prepared representation; fresh dense intensity output",
                    note="Support-read avoidance is distinct from output materialization: every arm writes N outputs.",
                )
            )
    return rows


def main() -> Any:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--section", choices=["all", "blocks"], default="all")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    global ROW_LOG
    ROW_LOG = args.output / "rows.jsonl"
    rows = []
    temporal = []
    tasks: list[tuple[str, Callable[[], Any]]] = [
        ("ocean", lambda: ocean(args.size, args.repeats)),
        ("volume", lambda: volume(args.repeats)),
        ("reducers", lambda: reducers(args.repeats)),
        ("blocks", lambda: blocks(args.size, args.repeats)),
    ]
    for name, run in tasks:
        if args.section == "blocks" and name != "blocks":
            continue
        print("START " + name, flush=True)
        try:
            result = run()
            if name == "ocean":
                result, temporal = result
        except Exception as exc:
            result = [
                {
                    "workload": name,
                    "mechanics_status": "execution_failed",
                    "reason": repr(exc),
                    "evidence_note": "Completed per-arm observations retained in rows.jsonl",
                }
            ]
        rows.extend(result)
        (args.output / "results.json").write_text(
            json.dumps(
                dict(
                    rows=rows,
                    temporal=temporal,
                    host=platform.node(),
                    job_id=os.environ.get("MRUN_JOB_ID"),
                    numpy=np.__version__,
                    instrument_status={
                        "runtime": "CPU wall-clock median",
                        "dram": "not_measured",
                        "energy": "not_measured",
                    },
                    terminal_status="not_certified",
                ),
                indent=2,
            )
            + "\n"
        )
        print("DONE " + name + " " + str(len(result)) + " rows", flush=True)


if __name__ == "__main__":
    main()
