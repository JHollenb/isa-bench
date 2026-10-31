#!/usr/bin/env python3
"""Fetch and build Intel XED under the experiment artifact tree."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, ResourceMonitor, write_json


XED_URL = "https://github.com/intelxed/xed.git"
MBUILD_URL = "https://github.com/intelxed/mbuild.git"


def run_cmd(cmd: Sequence[str], cwd: Optional[Path] = None, timeout_s: Optional[float] = None) -> Dict[str, Any]:
    started = time.time()
    try:
        run = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            capture_output=True,
            timeout=timeout_s,
        )
        return {
            "cmd": list(cmd),
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": run.returncode,
            "ok": run.returncode == 0,
            "elapsed_s": round(time.time() - started, 4),
            "stdout_tail": run.stdout[-8000:],
            "stderr_tail": run.stderr[-8000:],
        }
    except Exception as exc:
        return {
            "cmd": list(cmd),
            "cwd": str(cwd) if cwd is not None else None,
            "returncode": None,
            "ok": False,
            "elapsed_s": round(time.time() - started, 4),
            "error": f"{type(exc).__name__}: {exc}",
        }


def git_commit(repo: Path) -> Optional[str]:
    if not (repo / ".git").exists():
        return None
    rec = run_cmd(["git", "rev-parse", "HEAD"], cwd=repo)
    if not rec["ok"]:
        return None
    return str(rec["stdout_tail"]).strip()


def clone_or_fetch(url: str, dest: Path, branch: str) -> Dict[str, Any]:
    if (dest / ".git").exists():
        recs = [
            run_cmd(["git", "fetch", "--depth", "1", "origin", branch], cwd=dest, timeout_s=300),
            run_cmd(["git", "checkout", "FETCH_HEAD"], cwd=dest, timeout_s=120),
        ]
        return {"action": "fetch_existing", "url": url, "dest": str(dest), "steps": recs, "ok": all(r["ok"] for r in recs)}
    dest.parent.mkdir(parents=True, exist_ok=True)
    rec = run_cmd(["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)], timeout_s=600)
    return {"action": "clone", "url": url, "dest": str(dest), "steps": [rec], "ok": rec["ok"]}


def executable_probe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "ok": False}
    rec = run_cmd([str(path), "-version"], timeout_s=30)
    if not rec["ok"]:
        rec = run_cmd([str(path), "--version"], timeout_s=30)
    rec["path"] = str(path)
    rec["exists"] = True
    return rec


def find_xed_binaries(root: Path) -> List[str]:
    bins = []
    if not root.exists():
        return bins
    for path in root.rglob("xed"):
        if path.is_file():
            bins.append(str(path))
    return sorted(bins)


def inspect_xed_kit(install_dir: Path) -> Dict[str, Any]:
    include_dir = install_dir / "include" / "xed"
    lib_dir = install_dir / "lib"
    misc_dir = install_dir / "misc"
    required = {
        "libxed": lib_dir / "libxed.a",
        "libxed_ild": lib_dir / "libxed-ild.a",
        "iform_enum": include_dir / "xed-iform-enum.h",
        "iclass_enum": include_dir / "xed-iclass-enum.h",
        "idata": misc_dir / "idata.txt",
    }
    return {
        "install_dir": str(install_dir),
        "required": {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else None,
            }
            for name, path in required.items()
        },
        "kit_ok": all(path.exists() for path in required.values()),
    }


def build_xed(args: argparse.Namespace) -> Dict[str, Any]:
    out_dir = args.out_dir or default_out_dir(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = out_dir / "src"
    xed_dir = src_dir / "xed"
    mbuild_dir = src_dir / "mbuild"
    install_dir = out_dir / "install"
    monitor = ResourceMonitor(out_dir, interval_s=args.monitor_interval)
    monitor.start()
    report: Dict[str, Any] = {
        "created": DEFAULT_DATE,
        "run_name": args.run_name,
        "out_dir": str(out_dir),
        "xed_url": XED_URL,
        "mbuild_url": MBUILD_URL,
        "branch": args.branch,
        "full_x86_goal_complete": False,
        "steps": {},
    }
    try:
        monitor.set_phase("clone")
        report["steps"]["clone_xed"] = clone_or_fetch(XED_URL, xed_dir, args.branch)
        report["steps"]["clone_mbuild"] = clone_or_fetch(MBUILD_URL, mbuild_dir, args.branch)
        report["xed_commit"] = git_commit(xed_dir)
        report["mbuild_commit"] = git_commit(mbuild_dir)

        if not report["steps"]["clone_xed"]["ok"] or not report["steps"]["clone_mbuild"]["ok"]:
            report["ok"] = False
            report["error"] = "clone_or_fetch_failed"
            write_json(out_dir / "xed_build_report.json", report)
            return report

        monitor.set_phase("inspect")
        report["steps"]["mfile_help"] = run_cmd([sys.executable, "mfile.py", "--help"], cwd=xed_dir, timeout_s=120)

        monitor.set_phase("build")
        install_dir.mkdir(parents=True, exist_ok=True)
        build_attempts = [
            [
                sys.executable,
                "mfile.py",
                "install",
                f"--install-dir={install_dir}",
                "--host-os=mac",
                "--ar=/usr/bin/ar",
                "--security-level=1",
                "--no-werror",
            ],
            [
                sys.executable,
                "mfile.py",
                "install",
                f"--install-dir={install_dir}",
                "--ar=/usr/bin/ar",
                "--security-level=1",
                "--no-werror",
            ],
            [
                sys.executable,
                "mfile.py",
                "--shared",
                "install",
                f"--install-dir={install_dir}",
                "--host-os=mac",
                "--ar=/usr/bin/ar",
                "--security-level=1",
                "--no-werror",
            ],
        ]
        build_results = []
        for cmd in build_attempts[: max(1, args.max_build_attempts)]:
            clean_cmd = [
                sys.executable,
                "mfile.py",
                "-c",
                "--host-os=mac",
                "--ar=/usr/bin/ar",
                "--security-level=1",
                "--no-werror",
            ]
            build_results.append(run_cmd(clean_cmd, cwd=xed_dir, timeout_s=120))
            rec = run_cmd(cmd, cwd=xed_dir, timeout_s=args.build_timeout_s)
            build_results.append(rec)
            if rec["ok"]:
                break
        report["steps"]["build_attempts"] = build_results

        monitor.set_phase("probe")
        report["xed_kit"] = inspect_xed_kit(install_dir)
        binaries = find_xed_binaries(out_dir)
        report["xed_binaries"] = binaries
        report["xed_binary_probe"] = executable_probe(Path(binaries[0])) if binaries else {"ok": False, "error": "no xed binary found"}
        report["kit_ok"] = bool(report["xed_kit"].get("kit_ok"))
        report["cli_ok"] = bool(binaries) and bool(report["xed_binary_probe"].get("ok"))
        report["ok"] = report["kit_ok"]
        if not report["cli_ok"]:
            report["cli_error"] = "xed_binary_probe_failed"
        if not report["kit_ok"]:
            report["error"] = "xed_kit_incomplete"
        monitor.set_phase("finalize")
        write_json(out_dir / "xed_build_report.json", report)
        return report
    finally:
        monitor.stop()


def default_out_dir(run_name: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / "source_deps" / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-name", default=f"xed-{int(time.time())}")
    p.add_argument("--branch", default="main")
    p.add_argument("--monitor-interval", type=float, default=0.5)
    p.add_argument("--build-timeout-s", type=float, default=900.0)
    p.add_argument("--max-build-attempts", type=int, default=2)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_xed(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
