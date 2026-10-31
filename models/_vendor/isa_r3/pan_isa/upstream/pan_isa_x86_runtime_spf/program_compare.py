"""Whole-program comparison for the standalone SPF runtime."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .runtime import SPFPrimaryPanRuntime, runtime_scope


def unsupported_program_report(program: Path, reason: str) -> Dict[str, Any]:
    return {
        "ok": False,
        "program": str(program),
        "steps": 0,
        "stop_reason": {
            "kind": "unsupported_runtime_feature",
            "signal": None,
            "exit_code": None,
            "detail": reason,
            "address": None,
        },
        "runtime_scope": runtime_scope(),
    }


def run_pan_program(
    program: Path,
    argv: Optional[List[str]] = None,
    max_steps: int = 100000,
    *,
    uid: Optional[int] = None,
    euid: Optional[int] = None,
    gid: Optional[int] = None,
    egid: Optional[int] = None,
    model: object | None = None,
    profile: str = "long64-ring3",
    backend: str = "auto",
    pl_xclbin: str | None = None,
    pl_kernel_name: str = "pan_simd_v1",
    pl_min_instructions: int = 8,
    pl_min_groups: int = 2,
) -> Dict[str, Any]:
    try:
        runtime = SPFPrimaryPanRuntime.from_elf(
            program,
            argv=argv,
            uid=uid,
            euid=euid,
            gid=gid,
            egid=egid,
            model=model,
            profile=profile,
            backend=backend,
            pl_xclbin=pl_xclbin,
            pl_kernel_name=pl_kernel_name,
            pl_min_instructions=pl_min_instructions,
            pl_min_groups=pl_min_groups,
        )
    except ValueError as exc:
        return unsupported_program_report(program, str(exc))
    runtime.run(max_steps=max_steps)
    return runtime.summary()


def run_qemu_user(program: Path, argv: Optional[List[str]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    binary = shutil.which("qemu-x86_64")
    if not binary:
        return {
            "ok": False,
            "engine": "qemu-x86_64",
            "blocker": {
                "kind": "missing_oracle_binary",
                "reason": "qemu-x86_64 is not installed or not on PATH",
            },
        }
    command = [binary, str(program), *list(argv or [])[1:]]
    started = time.time()
    proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    elapsed = time.time() - started
    return {
        "ok": True,
        "engine": "qemu-x86_64",
        "command": command,
        "elapsed_seconds": elapsed,
        "exit_code": proc.returncode,
        "stdout_hex": proc.stdout.hex(),
        "stdout_text": proc.stdout.decode("utf-8", errors="replace"),
        "stderr_hex": proc.stderr.hex(),
    }


def compare_program(
    program: Path,
    argv: Optional[List[str]] = None,
    max_steps: int = 100000,
    *,
    uid: Optional[int] = None,
    euid: Optional[int] = None,
    gid: Optional[int] = None,
    egid: Optional[int] = None,
    model: object | None = None,
    profile: str = "long64-ring3",
    backend: str = "auto",
    pl_xclbin: str | None = None,
    pl_kernel_name: str = "pan_simd_v1",
    pl_min_instructions: int = 8,
    pl_min_groups: int = 2,
) -> Dict[str, Any]:
    pan = run_pan_program(
        program,
        argv=argv,
        max_steps=max_steps,
        uid=uid,
        euid=euid,
        gid=gid,
        egid=egid,
        model=model,
        profile=profile,
        backend=backend,
        pl_xclbin=pl_xclbin,
        pl_kernel_name=pl_kernel_name,
        pl_min_instructions=pl_min_instructions,
        pl_min_groups=pl_min_groups,
    )
    qemu = run_qemu_user(program, argv=argv)
    comparable = bool(pan.get("ok") and qemu.get("ok"))
    exact = bool(
        comparable
        and pan.get("exit_code") == qemu.get("exit_code")
        and pan.get("stdout_hex") == qemu.get("stdout_hex")
        and pan.get("stderr_hex") == qemu.get("stderr_hex")
    )
    return {
        "ok": exact,
        "program": str(program),
        "comparison_scope": "Linux user ELF stdout/stderr/exit-code plus PAN SPF state summary",
        "pan": pan,
        "qemu": qemu,
        "exact_user_visible_result": exact,
        "accepted_blocker": None if qemu.get("ok") else qemu.get("blocker"),
    }
