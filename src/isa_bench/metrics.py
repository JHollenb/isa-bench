"""Aggregate cell records into per-(model, suite) metrics and a markdown summary."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

MIN_TIMED_STEPS = 1000
STATUSES = ("success", "wrong_output", "abstain", "refused", "fault", "step_limit", "wall_limit", "timeout", "error")


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def load_cells(eval_dir: Path) -> dict:
    cells = {}
    for path in sorted((Path(eval_dir) / "cells").glob("*.json")):
        cells[path.stem] = json.loads(path.read_text())
    return cells


def native_index(reference_dir: Path) -> dict:
    ref = json.loads((Path(reference_dir) / "reference.json").read_text())
    return ref["programs"]


def cell_row(tag: str, cell: dict, native: dict) -> dict:
    o = cell.get("outcome") or {}
    proc = cell.get("process") or {}
    status = o.get("status") or ("timeout" if proc.get("returncode") is None and "timeout_s" in proc else "error")
    samples = cell.get("samples") or []
    cold = samples[0]["usage"] if samples else {}
    warm = [s["usage"] for s in samples[1:]] or ([cold] if cold else [])
    native_walls = [t["wall_s"] for t in native.get("native_timing", [])]
    warm_wall = _median([u.get("wall_s") for u in warm])
    native_wall = _median(native_walls)
    steps = native.get("steps") or 0
    prefix = o.get("exact_prefix_steps") or 0
    return {
        "program": tag, "status": status, "native_steps": steps, "steps_run": o.get("steps_run"),
        "exact_prefix_steps": prefix, "exact_steps": o.get("exact_steps"),
        "coverage": prefix / max(1, steps - 1) if steps else 0.0,
        "endpoint_ok": bool(o.get("endpoint_ok")), "trace_exact": bool(o.get("trace_exact")),
        "stop": o.get("stop"), "first_divergence": o.get("first_divergence"),
        "cold_wall_s": cold.get("wall_s"), "cold_cpu_s": cold.get("cpu_s"), "warm_wall_s": warm_wall,
        "us_per_step": _median([u.get("us_per_step") for u in warm]),
        "peak_rss_mb": max([u.get("peak_rss_mb") or 0 for u in [cold, *warm]] or [0]),
        "native_wall_s": native_wall,
        "slowdown_vs_native": (warm_wall / native_wall) if warm_wall and native_wall else None,
        "deterministic": cell.get("deterministic"), "error": cell.get("error"),
    }


def summarize(model: str, suite: str, eval_dir: Path, reference_dir: Path) -> dict:
    natives = native_index(reference_dir)
    cells = load_cells(eval_dir)
    rows = [cell_row(tag, cells[tag], natives.get(tag, {})) for tag in sorted(cells)]
    meta = next((c.get("metadata") for c in cells.values() if c.get("metadata")), {}) or {}
    load = next((c.get("load") for c in cells.values() if c.get("load")), {}) or {}
    n = len(rows)
    total_native = sum(max(0, r["native_steps"] - 1) for r in rows)
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in STATUSES}
    counts["other"] = n - sum(counts.values())
    metrics = {
        "programs": n,
        "success": counts["success"],
        "success_rate": counts["success"] / n if n else 0.0,
        "trace_exact": sum(1 for r in rows if r["trace_exact"]),
        "step_coverage": sum(r["exact_prefix_steps"] for r in rows) / total_native if total_native else 0.0,
        "mean_program_coverage": statistics.fmean([r["coverage"] for r in rows]) if rows else 0.0,
        "status_counts": counts,
        # per-step cost only where a run is long enough to amortize setup (early stops measure setup, not steps)
        "us_per_step_median": _median([r["us_per_step"] for r in rows if (r["steps_run"] or 0) >= MIN_TIMED_STEPS]),
        "slowdown_vs_native_median": _median([r["slowdown_vs_native"] for r in rows if r["status"] == "success"]),
        "peak_rss_mb_max": max([r["peak_rss_mb"] for r in rows] or [0]),
        # one cold run of every program (warm repeats excluded), plus the one-off model load
        "wall_s_total": sum(r["cold_wall_s"] or 0 for r in rows),
        "cpu_s_total": sum(r["cold_cpu_s"] or 0 for r in rows),
        "steps_run_total": sum(r["steps_run"] or 0 for r in rows),
        "load_wall_s": load.get("wall_s"),
        "parameter_count": meta.get("parameter_count"),
        "learned_records": meta.get("learned_records") or meta.get("learned_record_count"),
        "artifact_bytes": meta.get("artifact_bytes"),
        "artifact_sha256": meta.get("artifact_sha256"),
        "nondeterministic_cells": sum(1 for r in rows if r["deterministic"] is False),
        "errors": sum(1 for r in rows if r["error"]),
    }
    return {"model": model, "suite": suite, "metrics": metrics, "rows": rows, "metadata": meta}


def flat_metrics(summary: dict) -> dict:
    """Numeric-only metrics for MLflow / DVC (prefixed by suite)."""
    m, s = summary["metrics"], summary["suite"]
    out = {}
    for k, v in m.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)) and v is not None:
            out[f"{s}.{k}"] = float(v)
    for k, v in m["status_counts"].items():
        out[f"{s}.status.{k}"] = float(v)
    return out


def fmt(v, digits=3):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}g}"
    return str(v)


def markdown(summary: dict) -> str:
    m = summary["metrics"]
    lines = [f"# {summary['model']} on {summary['suite']}", "",
             f"Success {m['success']}/{m['programs']}, trace-exact {m['trace_exact']}, "
             f"step coverage {m['step_coverage']:.1%} (exact prefix steps / native steps), "
             f"median {fmt(m['us_per_step_median'])} µs/step, params {fmt(m['parameter_count'])}, "
             f"artifact {fmt(m['artifact_bytes'])} B.", "",
             "| program | status | exact prefix / native | first divergence | warm wall s | µs/step | ×native |",
             "|---|---|---:|---|---:|---:|---:|"]
    for r in summary["rows"]:
        div = r["first_divergence"]
        dtxt = "—" if not div else f"step {div['step']} @ {div['rip']}: {', '.join(list(div['diff'])[:4])}"
        if r["status"] in ("abstain", "refused", "fault") and r["stop"]:
            dtxt = (dtxt + f"; stop {r['stop'].get('kind')}") if div else f"stop {r['stop'].get('kind')} {r['stop'].get('rip', r['stop'].get('address', ''))}"
        lines.append(f"| {r['program']} | {r['status']} | {r['exact_prefix_steps']}/{max(0, r['native_steps'] - 1)} | {dtxt} | "
                     f"{fmt(r['warm_wall_s'])} | {fmt(r['us_per_step'])} | {fmt(r['slowdown_vs_native'])} |")
    return "\n".join(lines) + "\n"
