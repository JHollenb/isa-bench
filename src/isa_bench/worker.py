"""Inside-the-lease entry point (Beast). Reads ``job.json`` from the payload root, writes ``result/``.

    python -m isa_bench.worker            # cwd = payload root

Stages:
  reference  compile every program variant, run it natively (timing) and trace it with ptrace
  eval       run each listed model on its suite's traced programs (models in sequence, one fresh
             subprocess per program); result/<model>/<suite>/{eval.json,cells/}
  train      run a model family's ``train.py`` (direct-effect executors)
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "result"


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


def run(cmd, timeout=600, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout, **kw)


def host() -> dict:
    cpu = None
    try:
        cpu = next((ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                    if ln.startswith("model name")), None)
    except OSError:
        pass
    gcc = run(["gcc", "--version"]).stdout.splitlines()[:1]
    libc = run(["ldd", "--version"]).stdout.splitlines()[:1]
    flags = []
    try:  # ISA extensions glibc's ifunc/CPUID dispatch keys on (static glibc picks string routines from these)
        line = next(ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("flags"))
        flags = sorted(set(line.split(":", 1)[1].split()) & {"sse4_2", "avx", "avx2", "avx512f", "avx512bw", "avx512vl",
                                                                "bmi1", "bmi2", "erms", "fsrm", "movbe", "lzcnt", "popcnt"})
    except (OSError, StopIteration):
        pass
    return {"node": platform.node(), "cpu": cpu, "python": sys.version.split()[0], "gcc": gcc, "libc": libc,
            "cpu_isa_flags": flags, "uname": platform.release()}


def mark(tag: str, payload: dict) -> None:
    print(f"ISA_BENCH_{tag}:" + json.dumps(payload, sort_keys=True, default=str), flush=True)


# ------------------------------------------------------------------ reference
def reference(job: dict) -> None:
    import hashlib

    import numpy as np

    suite = job["suite"]
    tools = OUT / "tools"
    tools.mkdir(parents=True)
    for name in ("trace_full", "native_measure"):
        b = run(["cc", "-O2", "-Wall", ROOT / f"programs/tools/{name}.c", "-o", tools / name])
        if b.returncode:
            raise RuntimeError(f"tool build failed {name}: {b.stderr[-800:]}")
    summary = {"suite": suite, "host": host(), "programs": {}}
    src_dir = ROOT / suite["dir"]
    # a suite may pin the traced program's environment (glibc suites: a clean env plus GLIBC_TUNABLES that steer
    # CPUID-based ifunc dispatch); None inherits the worker's environment as before
    penv = {k: str(v) for k, v in suite["env"].items()} if suite.get("env") is not None else None
    for name in suite["programs"]:
        for variant, flags in suite["variants"].items():
            tag = f"{name}-{variant}"
            pdir = OUT / "programs" / tag
            pdir.mkdir(parents=True)
            elf = pdir / "prog.elf"
            t0 = time.perf_counter()
            b = run(["gcc", *flags, src_dir / f"{name}.c", "-o", elf], timeout=120)
            rec = {"tag": tag, "source": f"{suite['dir']}/{name}.c", "flags": flags, "build_returncode": b.returncode,
                   "build_s": time.perf_counter() - t0, "build_stderr": b.stderr[-800:]}
            if b.returncode:
                summary["programs"][tag] = rec
                mark("BUILD_FAIL", rec)
                continue
            rec["elf_sha256"] = hashlib.sha256(elf.read_bytes()).hexdigest()
            rec["elf_bytes"] = elf.stat().st_size
            sizes = {}
            for line in run(["size", "-A", elf]).stdout.splitlines():
                f = line.split()
                if len(f) >= 2 and f[0].startswith(".") and f[1].isdigit():
                    sizes[f[0]] = int(f[1])
            rec["section_bytes"] = sizes
            dis = run(["objdump", "-d", "--no-show-raw-insn", elf]).stdout
            (pdir / "disasm.txt").write_text(dis)
            rec["static_instructions"] = sum(1 for ln in dis.splitlines() if ln[:1] == " " and ":\t" in ln)
            timing = []
            for _ in range(int(job.get("native_repeats", 7))):
                m = run([tools / "native_measure", elf, pdir / "stdout.native", pdir / "stderr.native"], timeout=60, env=penv)
                timing.append(json.loads(m.stdout))
            rec["native_timing"] = timing
            tr = run([tools / "trace_full", elf, pdir / "stdout", pdir / "stderr", pdir / "trace.bin", pdir / "mem.bin",
                      suite.get("max_trace_steps", 5_000_000)], timeout=7200, env=penv)
            status = json.loads(tr.stdout.strip().splitlines()[-1]) if tr.stdout.strip() else {"kind": "none"}
            raw = np.fromfile(pdir / "trace.bin", dtype=np.uint64).reshape(-1, 18)
            np.savez_compressed(pdir / "trace.npz", steps=raw)
            (pdir / "trace.bin").unlink()
            stdout = (pdir / "stdout").read_bytes()
            rec.update(tracer=status, steps=int(raw.shape[0]), stdout_hex=stdout.hex(),
                       exit_code=status.get("exit_code"), complete=status.get("kind") == "exit",
                       native_stdout_matches_trace=(pdir / "stdout.native").read_bytes() == stdout,
                       native_exit_matches_trace=all(t["exit_code"] == status.get("exit_code") for t in timing))
            write(pdir / "native.json", rec)
            summary["programs"][tag] = rec
            mark("TRACED", {"tag": tag, "steps": rec["steps"], "exit": rec["exit_code"], "complete": rec["complete"]})
    write(OUT / "reference.json", summary)


# ------------------------------------------------------------------ eval
def evaluate_one(model: dict, settings: dict, ref: Path, programs: list, out_dir: Path) -> dict:
    write(out_dir / "model.json", model)
    env = dict(os.environ, OMP_NUM_THREADS=str(settings["threads"]), MKL_NUM_THREADS=str(settings["threads"]),
               OPENBLAS_NUM_THREADS=str(settings["threads"]), PYTHONPATH=str(ROOT / "src"))
    cells = {}
    started = time.perf_counter()
    for tag in programs:
        pdir = ref / "programs" / tag
        native = json.loads((pdir / "native.json").read_text())
        if not native.get("complete"):
            cells[tag] = {"skipped": "native trace incomplete"}
            continue
        budget = min(int(native["steps"] * settings["max_steps_factor"]) + 64, int(settings["max_steps_cap"]))
        out = out_dir / "cells" / f"{tag}.json"
        cmd = [sys.executable, "-m", "isa_bench.cell", "--model", out_dir / "model.json", "--program", pdir, "--out", out,
               "--root", ROOT, "--repeats", settings["warm_repeats"], "--max-steps", budget,
               "--threads", settings["threads"], "--run-wall-s", settings.get("run_wall_s", 240),
               "--repeat-budget-s", settings.get("repeat_budget_s", 20)]
        t0 = time.perf_counter()
        try:
            proc = run(cmd, timeout=settings["cell_timeout_s"], env=env, cwd=ROOT)
            info = {"returncode": proc.returncode, "stderr_tail": proc.stderr[-1500:]}
        except subprocess.TimeoutExpired:
            info = {"returncode": None, "timeout_s": settings["cell_timeout_s"]}
        info["wall_s"] = time.perf_counter() - t0
        record = json.loads(out.read_text()) if out.exists() else {"model": model["name"], "program": tag,
                                                                   "complete": False}
        record["process"] = info
        write(out, record)
        o = record.get("outcome") or {}
        cells[tag] = {"status": o.get("status", "timeout" if info["returncode"] is None else "error"),
                      "exact_prefix_steps": o.get("exact_prefix_steps"), "native_steps": native["steps"]}
        mark("CELL", {"model": model["name"], "program": tag, **cells[tag], "wall_s": round(info["wall_s"], 2)})
    summary = {"model": model["name"], "host": host(), "cells": cells, "wall_s": time.perf_counter() - started,
               "reference_dir": str(ref), "run": settings}
    write(out_dir / "eval.json", summary)
    return summary


def evaluate(job: dict) -> None:
    """One lease, models in sequence: ``job["evals"]`` = [{model, suite, reference_dir, programs}]."""
    for item in job["evals"]:
        evaluate_one(item["model"], job["run"], Path(item["reference_dir"]), item["programs"],
                     OUT / item["model"]["name"] / item["suite"])


# ------------------------------------------------------------------ train
def train(job: dict) -> None:
    import importlib.util

    model = job["model"]
    path = ROOT / model["code"] / "train.py"
    spec = importlib.util.spec_from_file_location("isa_bench_train", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "src"))
    spec.loader.exec_module(module)
    module.main(model, ROOT, OUT / "artifact", emit=lambda s: print(s, flush=True))


def main() -> int:
    job = json.loads((ROOT / "job.json").read_text())
    OUT.mkdir(exist_ok=False)
    t0 = time.perf_counter()
    mark("START", {"stage": job["stage"], "host": platform.node()})
    {"reference": reference, "eval": evaluate, "train": train}[job["stage"]](job)
    mark("DONE", {"stage": job["stage"], "wall_s": round(time.perf_counter() - t0, 1)})
    shutil.copy2(ROOT / "job.json", OUT / "job.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
