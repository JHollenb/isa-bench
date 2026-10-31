"""Canonical timings + accuracy schema (v1).

Single source of truth shared by the Python per-variant harness
(``tests/variants/``) and the cross-tree aggregator
(``scripts/aggregate_timings.py``). Mirrors ``docs/timings-schema.md`` and the C
header ``native/include/spf_timings_schema.h`` — the column order here MUST match
both byte-for-byte.
"""

from __future__ import annotations

import datetime as _dt
import enum
import json
from collections.abc import Iterable, Mapping
from typing import Any

SCHEMA_VERSION = 1

# Column order — must match SPF_TIMINGS_CSV_HEADER and docs/timings-schema.md.
CANONICAL_COLUMNS: tuple[str, ...] = (
    "schema_version",
    "family",
    "group",
    "op",
    "variant_id",
    "mode",
    "format",
    "backend",
    "oracle",
    "acceptance",
    "n",
    "trials",
    "warmup",
    "elems_per_op",
    "latency_min_ns_per_op",
    "latency_avg_ns_per_op",
    "latency_max_ns_per_op",
    "latency_avg_ns_per_elem",
    "accuracy_status",
    "accuracy_metric",
    "accuracy_errors",
    "accuracy_total",
    "accuracy_max_abs",
    "accuracy_p99_abs",
    "accuracy_mean_abs",
    "is_core_op",
    "core_op_ok",
    "hot_path",
    "notes",
)

CSV_HEADER = ",".join(CANONICAL_COLUMNS)


class AccuracyStatus(enum.StrEnum):
    NA = "na"
    PASS = "pass"
    FAIL = "fail"
    DRIFT = "drift"
    SCALAR_ONLY = "scalar_only"


def status_from_passed(passed: bool | None) -> str:
    if passed is None:
        return AccuracyStatus.NA.value
    return (AccuracyStatus.PASS if passed else AccuracyStatus.FAIL).value


def row(
    *,
    family: str,
    op: str,
    mode: str = "na",
    n: int,
    latency_avg_ns_per_op: float,
    accuracy_status: str,
    group: str | None = None,
    variant_id: str = "",
    format: str = "fp64",
    backend: str = "python",
    oracle: str = "fp64",
    acceptance: str = "",
    trials: int = 1,
    warmup: int = 0,
    elems_per_op: int = 1,
    latency_min_ns_per_op: float | None = None,
    latency_max_ns_per_op: float | None = None,
    latency_avg_ns_per_elem: float | None = None,
    accuracy_metric: str = "max_abs",
    accuracy_errors: int = 0,
    accuracy_total: int = 0,
    accuracy_max_abs: float | str = "",
    accuracy_p99_abs: float | str = "",
    accuracy_mean_abs: float | str = "",
    is_core_op: bool = False,
    core_op_ok: str = "na",
    hot_path: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Build one canonical row dict. Unset latency stats default to the avg."""
    avg = latency_avg_ns_per_op
    return {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "group": group if group is not None else family,
        "op": op,
        "variant_id": variant_id,
        "mode": mode,
        "format": format,
        "backend": backend,
        "oracle": oracle,
        "acceptance": acceptance,
        "n": int(n),
        "trials": int(trials),
        "warmup": int(warmup),
        "elems_per_op": int(elems_per_op),
        "latency_min_ns_per_op": float(latency_min_ns_per_op if latency_min_ns_per_op is not None else avg),
        "latency_avg_ns_per_op": float(avg),
        "latency_max_ns_per_op": float(latency_max_ns_per_op if latency_max_ns_per_op is not None else avg),
        "latency_avg_ns_per_elem": float(
            latency_avg_ns_per_elem
            if latency_avg_ns_per_elem is not None
            else avg / max(elems_per_op, 1)
        ),
        "accuracy_status": accuracy_status,
        "accuracy_metric": accuracy_metric,
        "accuracy_errors": int(accuracy_errors),
        "accuracy_total": int(accuracy_total),
        "accuracy_max_abs": accuracy_max_abs,
        "accuracy_p99_abs": accuracy_p99_abs,
        "accuracy_mean_abs": accuracy_mean_abs,
        "is_core_op": 1 if is_core_op else 0,
        "core_op_ok": core_op_ok,
        "hot_path": hot_path,
        "notes": notes,
    }


def _cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_md(rows: Iterable[Mapping[str, Any]], *, title: str, headline: str = "") -> str:
    """Human-readable per-family table. Schema lives in the csv/json siblings."""
    cols = ("op", "mode", "backend", "n", "latency_avg_ns_per_op",
            "accuracy_status", "accuracy_metric", "accuracy_max_abs",
            "is_core_op", "core_op_ok")
    head = ("op", "mode", "backend", "n", "ns/op", "acc", "metric",
            "max_abs", "core?", "core_ok")
    lines = [f"# {title}", ""]
    if headline:
        lines += [headline, ""]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join("---" for _ in head) + "|")
    for r in rows:
        lines.append("| " + " | ".join(_cell(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def to_json(rows: Iterable[Mapping[str, Any]], *, family: str | None = None,
            meta: Mapping[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "report": "spf_timings",
        "schema_version": SCHEMA_VERSION,
    }
    if family is not None:
        payload["family"] = family
    if meta:
        payload["meta"] = dict(meta)
    payload["results"] = [dict(r) for r in rows]
    return json.dumps(payload, indent=2) + "\n"


def _utc_now_iso() -> str:
    return _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_readme(meta: Mapping[str, Any], rows: Iterable[Mapping[str, Any]]) -> str:
    """Run-level README satisfying the data-runs-layout REQUIRED-README rule."""
    rows = list(rows)
    families: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        families.setdefault(str(r.get("family", "unknown")), []).append(r)
    lines = [
        f"# {meta.get('title', 'SPF timings run')}",
        "",
        f"- Generated at: `{meta.get('generated_at', _utc_now_iso())}`",
        f"- Generator: `{meta.get('generator', 'python')}`",
        f"- Schema version: `{SCHEMA_VERSION}` (see `docs/timings-schema.md`)",
    ]
    for key in ("commit", "python", "numpy", "host", "command"):
        if meta.get(key):
            lines.append(f"- {key.capitalize()}: `{meta[key]}`")
    lines += ["", "## Families", "", "| family | rows | passed | failed |",
              "| --- | ---: | ---: | ---: |"]
    for fam, frows in sorted(families.items()):
        passed = sum(1 for r in frows if r.get("accuracy_status") == "pass")
        failed = sum(1 for r in frows if r.get("accuracy_status") == "fail")
        lines.append(f"| `{fam}` | {len(frows)} | {passed} | {failed} |")
    if meta.get("caveats"):
        lines += ["", "## Caveats", "", str(meta["caveats"])]
    lines += ["", "## Layout", "",
              "- `timings.{csv,md,json}` — canonical per-row results.",
              "- `manifest.json` — machine index of this run."]
    return "\n".join(lines) + "\n"


def make_manifest(meta: Mapping[str, Any], rows: Iterable[Mapping[str, Any]],
                  files: Iterable[str] = ()) -> str:
    rows = list(rows)
    families: dict[str, int] = {}
    for r in rows:
        fam = str(r.get("family", "unknown"))
        families[fam] = families.get(fam, 0) + 1
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": meta.get("generated_at", _utc_now_iso()),
        "generator": meta.get("generator", "python"),
        "title": meta.get("title", ""),
        "commit": meta.get("commit", ""),
        "rows": len(rows),
        "families": families,
        "files": list(files),
    }
    return json.dumps(payload, indent=2) + "\n"
