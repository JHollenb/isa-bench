"""Source and tool provenance freezing for full-ISA automation."""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pan_x86_automation.common import maybe_json, write_json


FULL_ISA_DIR = "full_isa"
SOURCE_FREEZE_PATH = Path(FULL_ISA_DIR) / "source_freeze" / "source_freeze.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return "sha256:" + h.hexdigest()


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def file_record(name: str, paths: Iterable[Path], *, required: bool = True) -> Dict[str, Any]:
    searched = [path for path in paths]
    found = first_existing(searched)
    if found is None:
        return {
            "name": name,
            "required": required,
            "status": "missing" if required else "not_configured",
            "searched": [str(path) for path in searched],
        }
    return {
        "name": name,
        "required": required,
        "status": "frozen",
        "path": str(found),
        "sha256": sha256_file(found),
        "bytes": found.stat().st_size,
    }


def tool_record(binary: str, version_args: Iterable[str] = ("--version",), *, required: bool = True) -> Dict[str, Any]:
    path = shutil.which(binary)
    if path is None:
        return {"name": binary, "required": required, "status": "missing" if required else "not_configured"}
    version = None
    ok = False
    try:
        run = subprocess.run(
            [path, *version_args],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
        version = (run.stdout or run.stderr).strip().splitlines()[:5]
        ok = run.returncode == 0
    except Exception as exc:  # pragma: no cover - defensive around local tool behavior
        version = [f"version probe failed: {exc}"]
    return {
        "name": binary,
        "required": required,
        "status": "frozen" if ok else "probe_failed",
        "path": path,
        "version": version,
    }


def build_source_freeze(root: Path, matrix: Dict[str, Any]) -> Dict[str, Any]:
    xed_dir = root / matrix["xed_source_deps"]
    xed_idata = xed_dir / "install" / "misc" / "idata.txt"
    xed_ledger = root / matrix["latest_xed_iform_ledger"] / "xed_iform_ledger.json"
    xed_build = xed_dir / "xed_build_report.json"
    xed_build_report = maybe_json(xed_build) or {}
    refs = {
        "intel_sdm": file_record(
            "intel_sdm",
            [
                root / "source_deps" / "intel-sdm" / "intel-sdm.pdf",
                root / "source_deps" / "intel-sdm" / "325462-sdm-vol-1-2abcd-3abcd.pdf",
            ],
        ),
        "amd_apm": file_record(
            "amd_apm",
            [
                root / "source_deps" / "amd-apm" / "amd-apm.pdf",
                root / "source_deps" / "amd-apm" / "amd64-architecture-programmers-manual.pdf",
            ],
        ),
        "xed_idata": file_record("xed_idata", [xed_idata, xed_ledger]),
        "xed_build_report": file_record("xed_build_report", [xed_build], required=False),
    }
    tools = {
        "qemu-system-x86_64": tool_record("qemu-system-x86_64"),
        "bochs": tool_record("bochs", ("--help",)),
        "cc": tool_record("cc", ("--version",), required=False),
        "as": tool_record("as", ("--version",), required=False),
        "xed": tool_record("xed", ("--version",), required=False),
        "sde": tool_record("sde", ("-version",), required=False),
    }
    required_refs_ok = all(record.get("status") == "frozen" for record in refs.values() if record.get("required"))
    required_tools_ok = all(record.get("status") == "frozen" for record in tools.values() if record.get("required"))
    return {
        "schema_version": "pan-x86-full-isa-source-freeze-v1",
        "created_unix": int(time.time()),
        "target": "all",
        "artifact_root": str(root),
        "ok": required_refs_ok and required_tools_ok,
        "references": refs,
        "tools": tools,
        "xed_build_report": xed_build_report,
        "gate_failures": [
            name
            for name, failed in {
                "required_references": not required_refs_ok,
                "required_tools": not required_tools_ok,
            }.items()
            if failed
        ],
    }


def write_source_freeze(root: Path, freeze: Dict[str, Any]) -> None:
    write_json(root / SOURCE_FREEZE_PATH, freeze)


def load_source_freeze(root: Path) -> Dict[str, Any]:
    return maybe_json(root / SOURCE_FREEZE_PATH) or {}
