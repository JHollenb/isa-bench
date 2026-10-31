"""Debug one failing (model, program) cell: first divergence, native context, and an MDB replay recipe.

    python -m isa_bench stage=debug model=de_mamba suite=c_apps debug.program=small_checksum-Os

Writes ``debug/<model>/<suite>/<program>/``:
  report.md       stop/divergence, expected vs predicted state, disassembly around RIP
  bundle.json     the exact inputs of the divergent step (window, registers, flags)
  root-recipe/    (direct-effect models) MDB discovery replay recipe of that forward
Then:  python -m mdb discovery replay debug/.../root-recipe --interactive --output debug/.../console
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .adapters import REPO, load_adapter, sha256_file

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
    if not rip or not path.exists():
        return "(disassembly unavailable locally)"
    lines = path.read_text().splitlines()
    want = rip.lower().removeprefix("0x")
    for i, line in enumerate(lines):
        m = re.match(r"\s*([0-9a-f]+):", line)
        if m and m.group(1) == want:
            lo, hi = max(0, i - radius), min(len(lines), i + radius + 1)
            return "\n".join(("=> " if j == i else "   ") + lines[j] for j in range(lo, hi))
    return f"(rip {rip} not found in disassembly)"


def mdb_recipe(model_cfg: dict, bundle: dict, out: Path, case_id: str) -> dict:
    """Record the divergent forward with MDB's native recorder and save a discovery replay recipe."""
    if str(MDB_SRC) not in sys.path:
        sys.path.insert(0, str(MDB_SRC))
    import torch
    from mdb.discovery_cli import save_capture
    from mdb.forward_recording import record_forward

    adapter = load_adapter(model_cfg, REPO)
    adapter.load()
    before = bundle["before"]
    regs = [int(before["regs"][n], 16) for n in ("RAX", "RCX", "RDX", "RBX", "RSP", "RBP", "RSI", "RDI")]
    flags = [int(before["flags"][n]) for n in ("CF", "ZF", "SF", "OF")]
    window = bytes.fromhex(bundle["window"]).ljust(15, b"\0")[:15]
    win = torch.tensor([[(v >> i) & 1 for v in window for i in range(8)]], dtype=torch.float32)
    reg = torch.tensor([[[(v >> i) & 1 for i in range(64)] for v in regs]], dtype=torch.float32)
    flg = torch.tensor([flags], dtype=torch.float32)
    checkpoint = sha256_file(adapter.artifact / "model.pt")
    capture = record_forward(
        adapter.model, (win, reg, flg), model_identity=checkpoint,
        execution={"backend": "cpu-fp32", "kind": "isa-bench-first-divergence", "program_step": bundle["step"]},
        input_identity={"window_hex": window.hex(), "register_words": regs, "flags_CF_ZF_SF_OF": flags},
        metadata={"case_id": case_id, "checkpoint_sha256": checkpoint, "program_step": bundle["step"]})
    root = out / "root-recipe"
    if root.exists():
        import shutil

        shutil.rmtree(root)
    save_capture(capture, root)
    return {"root_recipe": str(root.relative_to(REPO)), "checkpoint_sha256": checkpoint,
            "symbol_table": (model_cfg.get("debug") or {}).get("symbol_table") if checkpoint == R3_MAMBA_SHA else None,
            "operations": len(capture.trace.get("events", []))}


def run_debug(cfg: dict) -> dict:
    model_cfg, suite = cfg["model"], cfg["suite"]["name"]
    eval_dir = REPO / cfg["paths"]["eval"]
    tag, cell = pick_cell(eval_dir, (cfg.get("debug") or {}).get("program"))
    out = REPO / "debug" / model_cfg["name"] / suite / tag
    out.mkdir(parents=True, exist_ok=True)
    o = cell.get("outcome") or {}
    div = o.get("first_divergence")
    rip = (div or {}).get("rip") or (o.get("stop") or {}).get("rip")
    ref_prog = REPO / cfg["paths"]["reference"] / "programs" / tag
    bundle = cell.get("debug_bundle")
    info = {"model": model_cfg["name"], "suite": suite, "program": tag, "status": o.get("status"),
            "stop": o.get("stop"), "exact_prefix_steps": o.get("exact_prefix_steps"),
            "native_steps": o.get("native_steps"), "first_divergence": div, "error": cell.get("error")}
    if bundle:
        (out / "bundle.json").write_text(json.dumps(bundle, indent=1, sort_keys=True))
        if model_cfg["family"] == "direct_effect" and (model_cfg.get("debug") or {}).get("mdb"):
            info["mdb"] = mdb_recipe(model_cfg, bundle, out, f"{model_cfg['name']}-{tag}-{bundle['step']}")
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
    lines += ["", "## Native code around RIP", "```", disasm_context(ref_prog, rip), "```"]
    if info.get("mdb"):
        m = info["mdb"]
        lines += ["", "## MDB", f"Replay recipe: `{m['root_recipe']}` ({m['operations']} recorded operations).", "",
                  "```sh", f"python -m mdb discovery replay {m['root_recipe']} --interactive --output {out.relative_to(REPO)}/console",
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
                      "mdb": info.get("mdb")}, default=str))
    return info
