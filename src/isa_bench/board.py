"""Model x suite board of correctness and compute (stage=board).

  python -m isa_bench stage=board                    every registered model's `latest-eval` version, from MLflow
  python -m isa_bench stage=board board.source=local  outputs/eval/*/*/summary.json (unregistered work too)

Writes outputs/report/board.{md,html,json}. The HTML page is self-contained: sortable columns, suite filter.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from .adapters import REPO

COLUMNS = [  # (key, header, kind)
    ("model", "model", "text"), ("suite", "suite", "text"), ("success", "success", "frac"),
    ("trace_exact", "trace-exact", "int"), ("step_coverage", "step coverage", "pct"),
    ("us_per_step_median", "µs/step", "num"), ("slowdown_vs_native_median", "× native", "num"),
    ("load_wall_s", "load s", "num"), ("wall_s_total", "wall s", "num"), ("cpu_s_total", "CPU s", "num"),
    ("steps_run_total", "steps run", "int"), ("peak_rss_mb_max", "peak RSS MB", "num"),
    ("parameter_count", "params", "int"), ("learned_records", "records", "int"), ("artifact_mb", "artifact MB", "num"),
    ("abstain", "abstain", "int"), ("refused", "refused", "int"), ("wrong_output", "wrong", "int"),
    ("version", "version", "text"), ("git_sha", "git", "text"),
]


def _row(model: str, suite: str, m: dict, version=None, git_sha=None) -> dict:
    counts = m.get("status_counts") or {}
    r = {"model": model, "suite": suite, "programs": m.get("programs"), "version": version, "git_sha": (git_sha or "")[:7]}
    for k, _, _ in COLUMNS:
        if k in m:
            r[k] = m[k]
    r["artifact_mb"] = (m.get("artifact_bytes") or 0) / 2**20 or None
    for k in ("abstain", "refused", "wrong_output"):
        r[k] = counts.get(k, m.get(f"status.{k}"))
    return r


def from_local() -> list[dict]:
    from .cli import resummarize

    resummarize()  # summaries are derived from cells; refresh them so new metrics appear
    rows = []
    for path in sorted((REPO / "outputs/eval").glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        rows.append(_row(s["model"], s["suite"], s["metrics"], version="local"))
    return rows


def from_mlflow(tracking: dict) -> list[dict]:
    from mlflow.tracking import MlflowClient

    from .tracking import _env

    _env(tracking)
    client = MlflowClient(tracking_uri=tracking["tracking_uri"])
    prefix = tracking.get("registry_prefix", "isa-bench") + "-"
    rows = []
    for rm in client.search_registered_models(filter_string=f"name LIKE '{prefix}%'", max_results=500):
        try:
            v = client.get_model_version_by_alias(rm.name, "latest-eval")
        except Exception:
            continue
        run = client.get_run(v.run_id)
        by_suite: dict[str, dict] = {}
        for key, value in run.data.metrics.items():
            suite, _, metric = key.partition(".")
            by_suite.setdefault(suite, {})[metric] = value
        for suite, m in sorted(by_suite.items()):
            for k in ("success", "programs", "trace_exact", "steps_run_total", "parameter_count", "learned_records"):
                if m.get(k) is not None:
                    m[k] = int(m[k])
            rows.append(_row(rm.name[len(prefix):], suite, m, version=f"v{v.version}", git_sha=run.data.tags.get("git_sha")))
    return rows


def _fmt(r: dict, key: str, kind: str) -> str:
    v = r.get(key)
    if key == "success":
        return "—" if v is None else f"{v}/{r.get('programs', '?')}"
    if v is None or v == "":
        return "—"
    if kind == "pct":
        return f"{v:.1%}"
    if kind == "int":
        return f"{int(v):,}"
    if kind == "num":
        return f"{v:,.3g}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def markdown(rows: list[dict], source: str) -> str:
    lines = ["# isa-bench board", "",
             f"Source: {source}. wall/CPU s = one cold run of every program in the suite; µs/step = median over runs "
             f"of at least 1,000 steps; × native = median warm wall over the native binary's wall, successes only.", "",
             "| " + " | ".join(h for _, h, _ in COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r, k, kind) for k, _, kind in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>isa-bench board</title><style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mute:#6b6b66;--line:#e3e2de;--head:#f1f0ec;--good:#1f7a4d;--bad:#b3261e}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#e8e7e3;--mute:#9a9993;--line:#2c2c2a;--head:#1f1f1e;--good:#5cc28d;--bad:#f2867d}}
body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.45 ui-sans-serif,system-ui,sans-serif}
main{padding:20px 16px;max-width:1600px;margin:auto}h1{font-size:18px;margin:0 0 4px}p{color:var(--mute);margin:0 0 12px}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:6px}table{border-collapse:collapse;width:100%}
th,td{padding:6px 9px;border-bottom:1px solid var(--line);white-space:nowrap;text-align:right;font-variant-numeric:tabular-nums}
th{background:var(--head);cursor:pointer;position:sticky;top:0;font-weight:600}th:nth-child(-n+2),td:nth-child(-n+2){text-align:left}
td.full{color:var(--good);font-weight:600}td.zero{color:var(--mute)}select{font:inherit;margin:0 0 10px;padding:3px}
</style></head><body><main><h1>isa-bench board</h1><p>__NOTE__</p>
<label>suite <select id="suite"><option value="">all</option>__OPTIONS__</select></label>
<div class="wrap"><table id="t"><thead><tr>__HEAD__</tr></thead><tbody>__BODY__</tbody></table></div></main>
<script>
const t=document.getElementById('t'),tb=t.tBodies[0];let dir={};
t.tHead.onclick=e=>{const th=e.target.closest('th');if(!th)return;const i=th.cellIndex;dir[i]=-(dir[i]||-1);
[...tb.rows].sort((a,b)=>{const x=a.cells[i].dataset.v,y=b.cells[i].dataset.v,nx=parseFloat(x),ny=parseFloat(y);
return (isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)*dir[i]}).forEach(r=>tb.appendChild(r))};
document.getElementById('suite').onchange=e=>{for(const r of tb.rows)r.hidden=e.target.value&&r.cells[1].textContent!==e.target.value};
</script></body></html>"""


def page(rows: list[dict], note: str) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for _, h, _ in COLUMNS)
    body = []
    for r in rows:
        cells = []
        for k, _, kind in COLUMNS:
            v = r.get(k)
            sort = v if isinstance(v, (int, float)) else (v or "")
            cls = ""
            if k == "success" and v is not None:
                cls = "full" if v == r.get("programs") else ("zero" if v == 0 else "")
                sort = v / max(1, r.get("programs") or 1)
            cells.append(f'<td class="{cls}" data-v="{html.escape(str(sort))}">{html.escape(_fmt(r, k, kind))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    options = "".join(f"<option>{html.escape(s)}</option>" for s in sorted({r['suite'] for r in rows}))
    return (PAGE.replace("__NOTE__", html.escape(note)).replace("__HEAD__", head).replace("__BODY__", "\n".join(body))
            .replace("__OPTIONS__", options))


def run_board(cfg: dict) -> dict:
    source = cfg.get("board", {}).get("source", "mlflow")
    rows = from_local() if source == "local" else from_mlflow(cfg["tracking"])
    rows.sort(key=lambda r: (r["suite"], -(r.get("success") or 0) / max(1, r.get("programs") or 1), r["model"]))
    label = "local outputs/eval" if source == "local" else f"MLflow registry ({cfg['tracking']['tracking_uri']}, alias latest-eval)"
    out = REPO / cfg["paths"]["report"]
    out.mkdir(parents=True, exist_ok=True)
    md = markdown(rows, label)
    (out / "board.md").write_text(md)
    (out / "board.json").write_text(json.dumps(rows, indent=1, sort_keys=True, default=str) + "\n")
    (out / "board.html").write_text(page(rows, f"Source: {label}. {len(rows)} model×suite rows. Click a header to sort."))
    print(md)
    print(f"wrote {Path(out / 'board.html').relative_to(REPO)}")
    return {"rows": len(rows)}
