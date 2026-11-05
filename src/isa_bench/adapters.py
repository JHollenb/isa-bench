"""Model adapter contract and loader.

Each model family lives in ``models/<family>/`` with an ``adapter.py`` that defines
``Adapter(Base)``. A model *instance* is a Hydra config ``conf/model/<name>.yaml``
naming its family, artifact and comparison profile. The harness never imports a
family's code except through its adapter, inside a fresh cell subprocess.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

from .state import ArchState, Comparator, Trace

REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_bytes(path: Path | None) -> int:
    if path is None or not Path(path).exists():
        return 0
    p = Path(path)
    return p.stat().st_size if p.is_file() else sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def tree_sha256(path: Path | None) -> str | None:
    if path is None or not Path(path).exists():
        return None
    p = Path(path)
    if p.is_file():
        return sha256_file(p)
    h = hashlib.sha256()
    for f in sorted(x for x in p.rglob("*") if x.is_file()):
        h.update(str(f.relative_to(p)).encode() + b"\0" + sha256_file(f).encode() + b"\n")
    return h.hexdigest()


class Outcome(dict):
    """Result of one program run (JSON-serialisable dict).

    Keys: status, stop, stdout_hex, exit_code, steps_run, plus the Comparator summary.
    ``status`` is one of: success (exact stdout+exit), wrong_output, abstain, refused,
    fault, step_limit, error.
    """


class Base:
    """Adapter base. Subclasses implement ``load`` and ``run``."""

    family = "base"

    def __init__(self, cfg: dict, root: Path):
        self.cfg = cfg
        self.root = Path(root)                  # payload/repo root (where models/ lives)
        self.artifact = None if cfg.get("artifact") is None else self.root / cfg["artifact"]
        self.metadata: dict = {}
        self.deadline: float | None = None      # monotonic wall-clock cap for one run (set by the cell)

    def time_up(self, step: int) -> bool:
        import time

        return self.deadline is not None and step % 512 == 0 and time.monotonic() > self.deadline

    # -- paths -------------------------------------------------------------
    def code_dir(self) -> Path:
        return self.root / self.cfg["code"]

    def vendor_dir(self) -> Path:
        return self.root / "models/_vendor/isa_r3"

    def add_paths(self, *paths: Path) -> None:
        for p in reversed(paths):
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)

    # -- contract ----------------------------------------------------------
    def load(self) -> None:
        raise NotImplementedError

    def run(self, program_dir: Path, trace: Trace, max_steps: int) -> Outcome:
        raise NotImplementedError

    def comparator(self, trace: Trace) -> Comparator:
        compare = self.cfg.get("compare") or {}
        return Comparator(trace, gprs=int(compare.get("gprs", 16)), flags=tuple(compare.get("flags", ("CF", "ZF", "SF", "OF"))))

    def debug_bundle(self, outcome: Outcome) -> dict | None:
        """Inputs needed to reproduce the first divergent step offline (MDB, record lookup)."""
        return None

    def debug_forward(self, bundle: dict) -> dict | None:
        """The bundle's step as one torch forward for MDB: {module, args, model_identity, input_identity, kind}."""
        return None

    def debug_explain(self, bundle: dict, output=None) -> dict | None:
        """Family-specific reading of the bundle (and of the recorded forward's output, if any) for report.md."""
        return None

    def base_metadata(self) -> dict:
        return {"family": self.family, "artifact_bytes": tree_bytes(self.artifact),
                "artifact_sha256": tree_sha256(self.artifact)}


def finish(outcome: dict, comparator: Comparator, trace: Trace) -> Outcome:
    """Classify an outcome against the native endpoint and attach the comparison."""
    stop = outcome.get("stop") or {}
    kind = stop.get("kind")
    stdout = bytes.fromhex(outcome.get("stdout_hex", ""))
    if kind == "exit":
        ok = stdout == trace.expected_stdout and stop.get("exit_code") == trace.expected_exit
        status = "success" if ok else "wrong_output"
    elif kind in ("abstain", "abstain_predicate", "missing_record"):
        status = "abstain"
    elif kind and ("unsupported" in kind or kind.startswith("refused") or kind in ("selection_stop", "coverage_stop")):
        status = "refused"
    elif kind in ("step_limit", "timeout", "max_steps"):
        status = "step_limit"
    elif kind in ("fault", "unmapped_access", "store_fault", "fetch_fault"):
        status = "fault"
    else:
        status = "error" if kind in (None, "python_exception") else kind
    out = Outcome(outcome)
    out.update(comparator.summary())
    out["status"] = status
    out["endpoint_ok"] = status == "success"
    out["trace_exact"] = status == "success" and comparator.exact_prefix >= len(trace) - 1
    return out


def load_adapter(cfg: dict, root: Path) -> Base:
    family = cfg["family"]
    path = Path(root) / "models" / family / "adapter.py"
    spec = importlib.util.spec_from_file_location(f"isa_bench_family_{family}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Adapter(cfg, root)


def linux_boundary(regs: list, rip_after: int, memory, stdout: bytearray, *, rflags: int | None = None):
    """Supplied Linux write/exit at a SYSCALL. Kernel effects: RAX = result, RCX = return RIP, R11 = RFLAGS.

    ``regs`` is the 16-entry GPR list *before* the syscall (None for registers the
    model lacks). Returns (stop or None, regs_after).
    """
    rax, rdi, rsi, rdx = regs[0], regs[7], regs[6], regs[2]
    if rax in (60, 231):
        return {"kind": "exit", "exit_code": rdi & 0xFF}, list(regs)
    if rax == 1 and rdi in (1, 2) and rdx < (1 << 20):
        data = memory.read(rsi, rdx)
        if rdi == 1:
            stdout.extend(data)
        after = list(regs)
        after[0] = rdx
        after[1] = rip_after
        if len(after) > 11 and after[11] is not None and rflags is not None:
            after[11] = rflags
        return None, after
    return {"kind": "unsupported_syscall", "rax": rax}, list(regs)


__all__ = ["ArchState", "Base", "Outcome", "finish", "load_adapter", "linux_boundary", "sha256_file", "tree_bytes",
           "tree_sha256", "REPO"]
