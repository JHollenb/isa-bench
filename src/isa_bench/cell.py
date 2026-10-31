"""One (model, program) cell in a fresh process: load, run cold + warm repeats, measure, write JSON.

    python -m isa_bench.cell --model model.json --program <program_dir> --out cell.json \
        --root <payload root> --repeats 2 --max-steps N
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

ENTERED = time.perf_counter()


class Usage:
    """Wall/CPU time and sampled RSS of this process (psutil when present, else /proc or ru_maxrss)."""

    def __init__(self, sample_s: float = 0.005):
        self.sample_s = sample_s

    def _rss(self) -> int:
        try:
            with open("/proc/self/statm") as f:
                return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
        except OSError:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    def _loop(self):
        while not self.done.wait(self.sample_s):
            self.peak = max(self.peak, self._rss())

    def __enter__(self):
        self.done = threading.Event()
        self.start_rss = self.peak = self._rss()
        self.t0, self.c0 = time.perf_counter(), time.process_time()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.wall, self.cpu = time.perf_counter() - self.t0, time.process_time() - self.c0
        self.peak = max(self.peak, self._rss())
        self.done.set()
        self.thread.join()
        self.result = {"wall_s": self.wall, "cpu_s": self.cpu, "rss_start_mb": self.start_rss / 2**20,
                       "peak_rss_mb": self.peak / 2**20, "peak_delta_mb": (self.peak - self.start_rss) / 2**20}


def outcome_digest(o: dict) -> str:
    keys = ("status", "stdout_hex", "exit_code", "steps_run", "exact_prefix_steps", "exact_steps", "first_divergence")
    return hashlib.sha256(json.dumps({k: o.get(k) for k in keys}, sort_keys=True, default=str).encode()).hexdigest()


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--program", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--max-steps", type=int, required=True)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--run-wall-s", type=float, default=240.0, help="wall cap for one run; stop kind wall_limit")
    p.add_argument("--repeat-budget-s", type=float, default=20.0, help="skip warm repeats when the cold run was slower")
    args = p.parse_args(argv)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    sys.path.insert(0, str(args.root / "src"))
    from isa_bench.adapters import load_adapter
    from isa_bench.state import Trace

    cfg = json.loads(args.model.read_text())
    cfg.setdefault("threads", args.threads)
    record = {"model": cfg["name"], "program": args.program.name, "complete": False}
    try:
        trace = Trace(args.program)
        adapter = load_adapter(cfg, args.root)
        with Usage() as load:
            adapter.load()
        record.update(metadata=adapter.metadata, load=load.result)
        samples, outcome = [], None
        for repeat in range(args.repeats + 1):
            if repeat and samples[0]["usage"]["wall_s"] > args.repeat_budget_s:
                break
            gc.collect()
            adapter.deadline = time.monotonic() + args.run_wall_s
            with Usage() as use:
                o = adapter.run(args.program, trace, args.max_steps)
            o = dict(o)
            usage = dict(use.result)
            steps = max(1, int(o.get("steps_run") or 1))
            usage["us_per_step"] = 1e6 * usage["wall_s"] / steps
            samples.append({"repeat": repeat, "usage": usage, "digest": outcome_digest(o)})
            if outcome is None:
                outcome = o
                record["first_response_s"] = time.perf_counter() - ENTERED
        record["outcome"] = outcome
        record["samples"] = samples
        record["deterministic"] = len({s["digest"] for s in samples}) == 1
        bundle = adapter.debug_bundle(outcome)
        if bundle:
            record["debug_bundle"] = bundle
        record["complete"] = True
    except Exception as error:  # failures are evidence: keep them
        record["error"] = {"type": type(error).__name__, "message": str(error)[:2000],
                           "traceback": traceback.format_exc()[-4000:]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1, sort_keys=True, default=str))
    return 0 if record["complete"] else 1


if __name__ == "__main__":
    sys.exit(main())
