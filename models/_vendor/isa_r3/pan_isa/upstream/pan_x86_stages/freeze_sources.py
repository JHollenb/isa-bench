#!/usr/bin/env python3
"""Freeze local source/tool/oracle provenance for the PAN x86 experiment."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import psutil
import torch

from pan_x86_stages.stage0_x86_pan import DEFAULT_ARTIFACT_ROOT, DEFAULT_DATE, EXP_ROOT, ResourceMonitor, write_json

PAN_ARTIFACT_ROOT = DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa"


def artifact_root_from_out_dir(out_dir: Path) -> Path:
    if out_dir.parent.name == "source_freeze":
        return out_dir.parent.parent
    return PAN_ARTIFACT_ROOT


def sha256_path(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def command_probe(cmd: Sequence[str], timeout_s: float = 3.0) -> Dict[str, Any]:
    path = shutil.which(cmd[0])
    rec: Dict[str, Any] = {
        "command": list(cmd),
        "path": path,
        "available": path is not None,
    }
    if path is None:
        return rec
    try:
        run = subprocess.run(list(cmd), text=True, capture_output=True, timeout=timeout_s)
    except Exception as exc:
        rec.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return rec
    rec.update({
        "ok": run.returncode == 0,
        "returncode": run.returncode,
        "stdout_tail": run.stdout[-1000:],
        "stderr_tail": run.stderr[-1000:],
    })
    return rec


def classify_xed() -> Dict[str, Any]:
    rec = command_probe(["xed", "--help"])
    text = (rec.get("stdout_tail") or "") + (rec.get("stderr_tail") or "")
    if "xcode-select" in text or "tool 'xed' requires Xcode" in text:
        rec["classification"] = "apple_xcode_editor_shim_not_intel_xed"
    elif "Intel" in text or "XED" in text:
        rec["classification"] = "possibly_intel_xed"
    elif rec.get("available"):
        rec["classification"] = "unknown_xed_binary"
    else:
        rec["classification"] = "not_found"
    return rec


def python_module_probe(extra_path: Optional[Path], name: str) -> Dict[str, Any]:
    inserted = False
    if extra_path is not None and extra_path.exists() and str(extra_path) not in sys.path:
        sys.path.insert(0, str(extra_path))
        inserted = True
    try:
        module = __import__(name)
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(module, "__version__", "unknown")
        rec = {
            "available": True,
            "version": version,
            "path": getattr(module, "__file__", None),
            "extra_path_used": inserted,
        }
    except Exception as exc:
        rec = {
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "extra_path_used": inserted,
        }
    return rec


def build_freeze(args: argparse.Namespace, out_dir: Path) -> Dict[str, Any]:
    artifact_root = artifact_root_from_out_dir(out_dir)
    oracle_dep_path = artifact_root / "python-deps" / "unicorn"
    script_paths = [
        EXP_ROOT / "pan_x86_stages" / "stage0_x86_pan.py",
        EXP_ROOT / "pan_x86_stages" / "stage1_memory_pan.py",
        EXP_ROOT / "pan_x86_stages" / "stage2_control_flow_pan.py",
        EXP_ROOT / "pan_x86_stages" / "stage3_complex_scalar_pan.py",
        EXP_ROOT / "pan_x86_stages" / "stage4_simd_x87_pan.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage1_rosetta.py",
        EXP_ROOT / "pan_x86_stages" / "freeze_sources.py",
        EXP_ROOT / "pan_x86_stages" / "enumerate_stage_coverage.py",
        EXP_ROOT / "pan_x86_stages" / "fetch_build_xed.py",
        EXP_ROOT / "pan_x86_stages" / "enumerate_xed_iforms.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage0_qemu.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage1_qemu.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage2_qemu.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage2_bochs.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage3_emulator.py",
        EXP_ROOT / "pan_x86_stages" / "validate_stage4_emulator.py",
        EXP_ROOT / "pan_x86_stages" / "audit_stage4_fp.py",
        EXP_ROOT / "schemas" / "trace_record.schema.json",
        EXP_ROOT / "configs" / "stage_matrix.yaml",
    ]
    tools = {
        "clang": command_probe(["clang", "--version"]),
        "arch_x86_64": command_probe(["/usr/bin/arch", "-x86_64", "/usr/bin/true"]),
        "qemu_x86_64": command_probe(["qemu-x86_64", "--version"]),
        "qemu_system_x86_64": command_probe(["qemu-system-x86_64", "--version"], timeout_s=10.0),
        "qemu_system_i386": command_probe(["qemu-system-i386", "--version"], timeout_s=10.0),
        "bochs": command_probe(["bochs", "--help"], timeout_s=10.0),
        "xed": classify_xed(),
    }
    data = {
        "created": DEFAULT_DATE,
        "run_name": args.run_name,
        "full_x86_goal_complete": False,
        "scope": {
            "architecture": "x86 family",
            "first_executable_target": "x86-64 long mode user space",
            "current_executable_stages": [
                "stage0_x86_lite_alu",
                "stage1_address_memory",
                "stage2_control_flow",
                "stage3_complex_scalar",
                "stage4_simd_x87",
            ],
            "xed_iform_coverage_complete": False,
            "second_oracle_status": (
                "qemu_spot_checks_for_stage0_stage1_stage2_stage3_stage4_bochs_stage2_stage3_stage4_and_rosetta_x86_64_stage1_subset"
                if tools["qemu_system_x86_64"].get("ok") and tools["qemu_system_i386"].get("ok") and tools["bochs"].get("ok")
                else "missing"
            ),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "python_packages": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "psutil": psutil.__version__,
            "unicorn_default_path": str(oracle_dep_path),
            "unicorn": python_module_probe(oracle_dep_path, "unicorn"),
            "jsonschema": python_module_probe(None, "jsonschema"),
        },
        "tools": tools,
        "references": {
            "intel_sdm": {
                "status": "not_downloaded_or_hashed_by_this_script",
                "url": "https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html",
            },
            "intel_xed": {
                "status": tools["xed"].get("classification"),
                "built_kit": str(artifact_root / "source_deps" / "intel-xed-main-2026-06-16" / "xed_build_report.json"),
                "note": "The shell xed command is classified before trusting it as Intel XED; the local Intel XED kit is tracked separately.",
            },
            "amd_apm": {
                "status": "not_downloaded_or_hashed_by_this_script",
                "url": "https://docs.amd.com/v/u/en-US/40332_4.09_APM_PUB",
            },
        },
        "local_sources": {
            str(path.relative_to(EXP_ROOT)): {
                "path": str(path),
                "sha256": sha256_path(path),
            }
            for path in script_paths
        },
        "artifact_inputs": {
            "stage0_learned": str(artifact_root / "stage0_x86_lite_alu" / "neural-vector-heldout-w8-unicorn-selector"),
            "stage1_learned": str(artifact_root / "stage1_address_memory" / "stage1-memory-indexed-neural-unicorn"),
            "stage2_learned": str(artifact_root / "stage2_control_flow" / "stage2-control-neural-unicorn-v3"),
            "stage3_learned": str(artifact_root / "stage3_complex_scalar" / "stage3-complex-scalar-base-alu-v21-neural"),
            "stage4_learned": str(artifact_root / "stage4_simd_x87" / "stage4-sse-aligned-exception-v31-neural"),
        },
    }
    write_json(out_dir / "source_freeze.json", data)
    return data


def default_out_dir(run_name: str) -> Path:
    return DEFAULT_ARTIFACT_ROOT / DEFAULT_DATE / "pan-x86-isa" / "source_freeze" / run_name


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--run-name", default=f"freeze-{int(time.time())}")
    p.add_argument("--monitor-interval", type=float, default=0.5)
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out_dir = args.out_dir or default_out_dir(args.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(out_dir, interval_s=args.monitor_interval)
    monitor.start()
    try:
        monitor.set_phase("probe")
        data = build_freeze(args, out_dir)
        monitor.set_phase("finalize")
        print(json.dumps(data, indent=2, sort_keys=True))
    finally:
        monitor.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
