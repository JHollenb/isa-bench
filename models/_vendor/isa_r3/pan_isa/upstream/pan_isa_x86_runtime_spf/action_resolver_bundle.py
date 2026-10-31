from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .action_resolver_demo import build_action_resolver_demo


BUNDLE_SCHEMA = "pan-action-resolver-demo-bundle-v1"


def build_action_resolver_bundle(artifact_dir: str | Path, *, source_commit: str | None = None) -> dict[str, Any]:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    summary = build_action_resolver_demo(root)
    manifest_path = root / "bundle_manifest.json"
    archive_path = root / "action_resolver_demo_bundle.tar.gz"

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "ok": bool(summary["ok"]),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "artifact_dir": str(root),
        "summary_json": str(root / "summary.json"),
        "ledger_jsonl": summary["ledger_jsonl"],
        "replay_json": summary["replay_json"],
        "archive": str(archive_path),
        "summary": summary,
        "environment": _environment(source_commit=source_commit),
        "commands": _commands(root, source_commit=source_commit),
        "files": _file_manifest(root, exclude={manifest_path.name, archive_path.name}),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_archive(root, archive_path, exclude={archive_path.name})
    manifest["archive_sha256"] = _sha256_file(archive_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _environment(*, source_commit: str | None) -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git_commit": source_commit or _git_commit(),
        "kv260_model": _kv260_model(),
    }


def _commands(root: Path, *, source_commit: str | None) -> dict[str, str]:
    artifact = str(root)
    source_arg = f" --source-commit {source_commit}" if source_commit else ""
    return {
        "local": f"python3 -m pan_isa_x86_runtime_spf.action_resolver_bundle --artifact-dir {artifact}{source_arg}",
        "kv260": (
            "ssh kv260 'cd /home/ubuntu/pan-isa-kv260-router/pan-isa && "
            "export PYTHONPATH=$PWD && "
            f"python3 -m pan_isa_x86_runtime_spf.action_resolver_bundle --artifact-dir {artifact}{source_arg}'"
        ),
        "replay": (
            "python3 -m pan_hardware_resolver.replay "
            f"{artifact}/action_ledger.jsonl --write {artifact}/replay_cli.json"
        ),
    }


def _file_manifest(root: Path, *, exclude: set[str]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in exclude:
            continue
        rows.append(
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return rows


def _write_archive(root: Path, archive_path: Path, *, exclude: set[str]) -> None:
    with tarfile.open(archive_path, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.name in exclude:
                continue
            archive.add(path, arcname=str(path.relative_to(root)))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _kv260_model() -> str | None:
    model_path = Path("/proc/device-tree/model")
    if not model_path.exists():
        return None
    try:
        return model_path.read_bytes().replace(b"\x00", b"").decode("utf-8", errors="replace").strip() or None
    except OSError:
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a packaged PAN action resolver demo artifact bundle.")
    parser.add_argument("--artifact-dir", default="/tmp/pan-action-resolver-bundle")
    parser.add_argument("--source-commit", help="Source commit hash to record when .git metadata is unavailable.")
    args = parser.parse_args(argv)
    manifest = build_action_resolver_bundle(args.artifact_dir, source_commit=args.source_commit)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
