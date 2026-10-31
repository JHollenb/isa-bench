"""Reproducible, isolated-process SFR benchmark entry point."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy  # type: ignore[import-untyped]

from .fields import VARIANTS, FilterSpec, fixture, make_runner, reference
from .metrics import display, quality, sample_rss
from .trace import BoundaryTrace, ExecutionTrace


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def run_filter(case: dict[str, Any], out: Path) -> dict[str, Any]:
    shape = tuple(case["resolution"])
    spec = FilterSpec(**case["spec"])
    source, input_meta = fixture(case["field"], shape, case["seed"])
    source_digest = digest(source)
    runner, meta, state = make_runner(source, spec, case["variant"])
    result: Any
    if case.get("trace_only"):
        trace = BoundaryTrace((source, *state))
        execution = ExecutionTrace()
        with trace.installed(), execution.installed():
            result = runner()
            trace.observe(result, "terminal")
        return {
            **case,
            "schema": "sfrbench-boundary-correction-v1",
            "workload": "filter-chain",
            "input_sha256": source_digest,
            "output_sha256": digest(result),
            "execution": meta,
            "execution_trace": execution.summary(),
            "mechanics_status": "valid",
            **trace.summary(),
        }
    start = time.perf_counter_ns()
    result = runner()
    first = (time.perf_counter_ns() - start) / 1e6
    for _ in range(case["warmups"] - 1):
        result = runner()
    samples = []
    for _ in range(case["repeats"]):
        result = None
        start = time.perf_counter_ns()
        result = runner()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    output_digest = digest(result)
    result = None
    result, memory = sample_rss(runner)
    # Instrumentation is outside timing and separate from the RSS observation.
    trace = BoundaryTrace((source, *state))
    execution = ExecutionTrace()
    with trace.installed(), execution.installed():
        traced_result = runner()
        trace.observe(traced_result, "terminal")
    if digest(traced_result) != digest(result):
        raise ValueError("boundary instrumentation changed the output")
    del traced_result
    expected = reference(source, spec)
    row = {
        "schema": "sfrbench-layered-v1",
        "workload": "filter-chain",
        **case,
        "backend": "numpy-fft-cpu" if case["variant"] == "spf-dense" else "scipy-cpu",
        "runtime_ms": statistics.median(samples),
        "samples_ms": samples,
        "first_call_ms": first,
        "warmup_policy": "first call is one warmup and separately reported; all warmups excluded from samples",
        "precision": str(result.dtype),
        "energy_j": None,
        "energy_instrument": "not_available",
        "arithmetic_operations": None,
        "reduction_operations": None,
        "temporal_error": None,
        "input_sha256": source_digest,
        "output_sha256": output_digest,
        "input_metadata": input_meta,
        "execution": meta,
        "execution_trace": execution.summary(),
        "execution_scope": meta["scope"],
        "prepare_ms": meta.get("prepare_ms"),
        "prepared_state_bytes": meta.get("prepared_state_bytes"),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "job_id": os.environ.get("MRUN_JOB_ID"),
        "thread_environment": {
            k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
        },
        "mechanics_status": "valid",
        "trend_status": "observation",
        "terminal_status": "not_certified",
        **memory,
        **trace.summary(),
        **quality(result, expected),
    }
    # Independent display previews are diagnostics, not replacements for full-size quality.
    from PIL import Image

    stride = max(1, max(shape) // 512)
    da, db = display(result)[::stride, ::stride], display(expected)[::stride, ::stride]
    np.savez_compressed(out.with_suffix(".npz"), actual=da, reference=db)
    Image.fromarray(np.rint(np.clip(da, 0, 1) * 255).astype(np.uint8)).save(out.with_suffix(".png"))
    return row


def write_summary(out: Path, rows: list[dict[str, Any]]) -> None:
    eager = {
        (r["field"], tuple(r["resolution"])): r
        for r in rows
        if r.get("variant") == "fp32-eager" and r.get("mechanics_status") == "valid"
    }
    fused = {
        (r["field"], tuple(r["resolution"])): r
        for r in rows
        if r.get("variant") == "fp32-fused" and r.get("mechanics_status") == "valid"
    }
    for row in rows:
        key = (row["field"], tuple(row["resolution"]))
        if key in eager and row.get("materialized_elements"):
            row["potential_boundary_elements"] = eager[key]["materialized_elements"]
            row["materialization_avoidance_boundary"] = (
                eager[key]["materialized_elements"] / row["materialized_elements"]
            )
        if key in fused and row.get("runtime_ms"):
            row["speedup_vs_fresh_fused_fp32"] = fused[key]["runtime_ms"] / row["runtime_ms"]
    (out / "results.json").write_text(
        json.dumps({"schema": "sfrbench-layered-v1", "rows": rows}, indent=2, allow_nan=False) + "\n"
    )
    columns = [
        "field",
        "resolution",
        "variant",
        "backend",
        "precision",
        "runtime_ms",
        "first_call_ms",
        "execution_scope",
        "prepare_ms",
        "prepared_state_bytes",
        "bytes_read",
        "bytes_written",
        "logical_bytes_read",
        "logical_bytes_written",
        "peak_memory",
        "materialized_elements",
        "materialization_avoidance_boundary",
        "fft_count",
        "relative_l2",
        "psnr",
        "ssim",
        "lpips",
        "energy_j",
        "mechanics_status",
    ]
    with (out / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    child = commands.add_parser("_worker")
    child.add_argument("--case", type=Path, required=True)
    child.add_argument("--output", type=Path, required=True)
    filters = commands.add_parser("filter-chain")
    filters.add_argument("--resolution", default="4096x4096")
    filters.add_argument("--field", default="smooth,chaotic")
    filters.add_argument("--variants", default=",".join(VARIANTS))
    filters.add_argument("--translate", default="0.37,0.22")
    filters.add_argument("--blur", type=float, default=8)
    filters.add_argument("--exposure", type=float, default=1.5)
    filters.add_argument("--glare", choices=["star6", "none"], default="star6")
    filters.add_argument("--seed", type=int, default=20260913)
    filters.add_argument("--warmups", type=int, default=1)
    filters.add_argument("--repeats", type=int, default=5)
    filters.add_argument("--output", type=Path, required=True)
    filters.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()
    if args.command == "_worker":
        case = json.loads(args.case.read_text())
        row = run_filter(case, args.output)
        args.output.write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in ["field", "variant", "runtime_ms", "relative_l2", "ssim", "peak_memory", "fft_count"]
                }
            ),
            flush=True,
        )
        return
    shape = tuple(int(x) for x in args.resolution.lower().split("x"))
    if len(shape) != 2 or min(shape) < 16 or args.repeats < 1 or args.warmups < 1:
        parser.error("resolution must be HxW >=16; warmups and repeats must be positive")
    shift = tuple(float(x) for x in args.translate.split(","))
    if len(shift) != 2 or not np.all(np.isfinite((*shift, args.blur, args.exposure))) or args.blur < 0:
        parser.error("finite translation dx,dy and nonnegative blur are required")
    spec = asdict(FilterSpec(translate=shift, blur=args.blur, exposure=args.exposure, glare=args.glare))
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be empty; preserve earlier runs in their own directories")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for field in args.field.split(","):
        for variant in args.variants.split(","):
            if variant not in VARIANTS:
                parser.error(f"unknown variant {variant}")
            case = dict(
                field=field,
                variant=variant,
                resolution=shape,
                spec=spec,
                seed=args.seed,
                warmups=args.warmups,
                repeats=args.repeats,
                trace_only=args.trace_only,
            )
            stem = f"{len(rows):03d}-{Path(field).stem}-{variant}"
            case_path = args.output / (stem + ".case.json")
            result_path = args.output / (stem + ".json")
            case_path.write_text(json.dumps(case, indent=2) + "\n")
            env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spf.sfrbench",
                    "_worker",
                    "--case",
                    str(case_path),
                    "--output",
                    str(result_path),
                ],
                env=env,
            )
            if completed.returncode:
                rows.append({**case, "mechanics_status": "execution_failed", "returncode": completed.returncode})
            else:
                rows.append(json.loads(result_path.read_text()))
            # A failed optional arm never discards other rows or completed evidence.
            write_summary(args.output, rows)
    print(f"Saved {len(rows)} rows to {args.output / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
