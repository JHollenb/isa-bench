"""Debug one failing (model, program) cell: first divergence, native context, and an MDB replay recipe.

    python -m isa_bench stage=debug model=de_mamba suite=c_apps debug.program=small_checksum-Os

Writes ``debug/<model>/<suite>/<program>/``:
  report.md       stop/divergence, expected vs predicted state, disassembly around RIP
  bundle.json     the exact inputs of the divergent step (window, registers, flags)
  root-recipe/    MDB discovery replay recipe of that forward (direct-effect models; neural selectors: the
                  runtime head's forward at the refusing instruction)
Then:  just mdb debug/.../root-recipe
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .adapters import REPO, load_adapter

WORKSPACE = REPO.parent
MDB_SRC = WORKSPACE / "saturn-pub/packages/mdb/src"
R3_MAMBA_SHA = "9399e921776d7a5437c5bbd51bfc1928f09d2ddea444fe2aa1c9cf282baf7f1e"


def pick_cell(eval_dir: Path, program: str | None) -> tuple[str, dict]:
    cells = {p.stem: json.loads(p.read_text()) for p in sorted((eval_dir / "cells").glob("*.json"))}
    if not cells:
        raise FileNotFoundError(f"no cell records under {eval_dir}/cells (run stage=eval first)")
    if program:
        return program, cells[program]
    for tag, cell in cells.items():
        if (cell.get("outcome") or {}).get("status") != "success":
            return tag, cell
    raise SystemExit("every program succeeded; pass debug.program=<tag> to inspect one anyway")


def disasm_context(ref_prog: Path, rip: str | None, radius: int = 6) -> str:
    path = ref_prog / "disasm.txt"
    if not path.exists():
        return "(disassembly unavailable locally)"
    if not rip:
        return "(no divergence or stop address: the run ended normally)"
    # objdump's header names the Beast job path of the ELF; keep only code lines
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and "file format" not in ln]
    want = rip.lower().removeprefix("0x")
    for i, line in enumerate(lines):
        m = re.match(r"\s*([0-9a-f]+):", line)
        if m and m.group(1) == want:
            lo, hi = max(0, i - radius), min(len(lines), i + radius + 1)
            return "\n".join(("=> " if j == i else "   ") + lines[j] for j in range(lo, hi))
    return f"(rip {rip} not found in disassembly)"


def mdb_recipe(adapter, model_cfg: dict, bundle: dict, out: Path, case_id: str, dbg: dict) -> tuple[dict | None, object]:
    """Record the bundle's forward with MDB's native recorder and save a discovery replay recipe.

    Returns (recipe info, the forward's output), or (None, None) when the family has no torch forward for it.
    """
    fwd = adapter.debug_forward(bundle)
    if fwd is None:
        return None, None
    if str(MDB_SRC) not in sys.path:
        sys.path.insert(0, str(MDB_SRC))
    from mdb.discovery_cli import save_capture
    from mdb.forward_recording import record_forward

    checkpoint = fwd["model_identity"]
    capture = record_forward(
        fwd["module"], fwd["args"], model_identity=checkpoint,
        execution={"backend": "cpu-fp32", "kind": fwd["kind"], "program_step": bundle["step"]},
        input_identity=fwd["input_identity"],
        metadata={"case_id": case_id, "checkpoint_sha256": checkpoint, "program_step": bundle["step"]},
        capture_ports=bool(dbg.get("capture_ports")))  # `mdb ports` needs it; it also renumbers every vNNNNNN
    root = out / "root-recipe"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    save_capture(capture, root)
    return ({"root_recipe": str(root.relative_to(REPO)), "checkpoint_sha256": checkpoint,
             "symbol_table": (model_cfg.get("debug") or {}).get("symbol_table") if checkpoint == R3_MAMBA_SHA else None,
             "operations": len(capture.trace.get("events", []))}, capture.output)


def explain_lines(model_cfg: dict, bundle: dict, explain: dict) -> list[str]:
    """report.md section for a family's debug_explain (selectors: chosen vs typed family, or the route lookup)."""
    typed = str(bundle.get("typed_family"))
    lines = ["", "## Selector", f"Step {bundle['step']} at {bundle['rip']}, fetch window `{bundle['bytes']}`.", ""]
    if typed.startswith("(typed adapter"):
        lines += [f"The supplied PAN consumer has no family for this instruction {typed[1:-1]}, so no selector "
                  "could pass here: a consumer coverage stop, not a selection error.", ""]
    else:
        lines += [f"The typed adapter executes **{typed}**; the selector has to name the same family.", ""]
    if "top" in explain:
        lines += ["| rank | family | probability |", "|---:|---|---:|"]
        lines += [f"| {i} | {lab} | {p:.3g} |" for i, (lab, p) in enumerate(explain["top"], 1)]
        t = explain.get("typed")
        if t:
            lines += ["", f"Typed family {t['label']}: rank {t['rank']}, probability {t['prob']:.3g} "
                      "(the runtime refuses below 0.5 or on any disagreement)."]
    elif "route_key" in explain:
        fields = ", ".join(f"{n}={v}" for n, v in zip(explain["route_fields"], explain["route_key"]))
        lines += [f"Route key ({fields}): **{explain['route_label'] or 'missing'}** "
                  f"among {explain['routes']} learned routes.", "",
                  f"Learned routes with opcode {explain['route_key'][1]:#x}:"]
        lines += [f"- `{k}` -> {lab}" for k, lab in explain["same_opcode_routes"]] or ["- none"]
    return lines


def run_debug(cfg: dict) -> dict:
    model_cfg, suite = cfg["model"], cfg["suite"]["name"]
    eval_dir = REPO / cfg["paths"]["eval"]
    tag, cell = pick_cell(eval_dir, (cfg.get("debug") or {}).get("program"))
    out = REPO / "debug" / model_cfg["name"] / suite / tag
    out.mkdir(parents=True, exist_ok=True)
    o = cell.get("outcome") or {}
    div = o.get("first_divergence")
    stop = o.get("stop") or {}
    rip = (div or {}).get("rip") or stop.get("rip") or stop.get("address") or (cell.get("debug_bundle") or {}).get("rip")
    ref_prog = REPO / cfg["paths"]["reference"] / "programs" / tag
    bundle = cell.get("debug_bundle")
    info = {"model": model_cfg["name"], "suite": suite, "program": tag, "status": o.get("status"),
            "stop": o.get("stop"), "exact_prefix_steps": o.get("exact_prefix_steps"),
            "native_steps": o.get("native_steps"), "first_divergence": div, "error": cell.get("error")}
    dbg, explain = model_cfg.get("debug") or {}, None
    if bundle:
        (out / "bundle.json").write_text(json.dumps(bundle, indent=1, sort_keys=True))
    if bundle and (dbg.get("mdb") or dbg.get("explain")):
        adapter = load_adapter(model_cfg, REPO)  # one local forward at most: the adapter, not a cell run
        adapter.load()
        output = None
        if dbg.get("mdb"):
            info["mdb"], output = mdb_recipe(adapter, model_cfg, bundle, out, f"{model_cfg['name']}-{tag}-{bundle['step']}",
                                         {**dbg, **(cfg.get("debug") or {})})
        explain = info["explain"] = adapter.debug_explain(bundle, output)
    elif not bundle and dbg.get("explain") and o.get("status") != "success":
        info["note"] = "no debug_bundle in this cell (evaluated before the adapter recorded one): rerun stage=eval"
    lines = [f"# Debug: {model_cfg['name']} on {tag}", "",
             f"Status **{o.get('status')}**; exact prefix {o.get('exact_prefix_steps')} of {o.get('native_steps')} native steps.", "",
             "## Stop", "```json", json.dumps(o.get("stop"), indent=1), "```", ""]
    if div:
        lines += ["## First divergence", f"Step {div['step']} at {div['rip']}, window `{div.get('window')}`.", "",
                  "| field | model | native |", "|---|---|---|"]
        lines += [f"| {k} | {v[0]} | {v[1]} |" for k, v in div["diff"].items()]
        extra = {k: v for k, v in div.items() if k not in ("step", "rip", "window", "diff")}
        if extra:
            lines += ["", "```json", json.dumps(extra, indent=1)[:3000], "```"]
    if explain:
        lines += explain_lines(model_cfg, bundle, explain)
    if info.get("note"):
        lines += ["", f"_{info['note']}_"]
    lines += ["", "## Native code around RIP", "```", disasm_context(ref_prog, rip), "```"]
    if info.get("mdb"):
        m = info["mdb"]
        lines += ["", "## MDB", f"Replay recipe: `{m['root_recipe']}` ({m['operations']} recorded operations).", "",
                  "```sh", f"just mdb {m['root_recipe']}",
                  "```", "Console commands: see saturn-pub/packages/mdb/docs/DISCOVERY-DEBUGGER.md "
                  "(breakpoints on module:post.N, capture/inspect, fork, restore)."]
        if m.get("symbol_table"):
            lines += ["", f"This checkpoint is the R3 Mamba of the recovered symbol table: `{m['symbol_table']}` "
                      "(see ../isa/outputs/moonshot-r3-mamba/symbol-table.md)."]
    if cell.get("error"):
        lines += ["", "## Error", "```", cell["error"].get("traceback", "")[-3000:], "```"]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "info.json").write_text(json.dumps(info, indent=1, sort_keys=True, default=str))
    print(json.dumps({"debug_dir": str(out.relative_to(REPO)), **{k: info[k] for k in ("status", "exact_prefix_steps")},
                      "mdb": info.get("mdb"), "explain": info.get("explain")}, default=str))
    return info
