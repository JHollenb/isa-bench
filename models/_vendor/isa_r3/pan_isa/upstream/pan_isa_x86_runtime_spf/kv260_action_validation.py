from __future__ import annotations

import argparse
import json
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


VALIDATION_SCHEMA = "pan-kv260-action-resolver-ps-validation-v1"


@dataclass(frozen=True)
class RemoteStep:
    name: str
    command: str
    artifacts: tuple[str, ...] = ()
    safe_scope: str = "ps-only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "command": self.command,
            "artifacts": list(self.artifacts),
            "safe_scope": self.safe_scope,
        }


def validation_steps(
    *,
    remote_repo: str,
    remote_artifact_dir: str,
    expected_model: str = "KV260",
    source_commit: str | None = None,
) -> list[RemoteStep]:
    repo = shlex.quote(remote_repo)
    art = shlex.quote(remote_artifact_dir)
    py = f"cd {repo} && env PYTHONPATH=$PWD"
    commit_arg = f" --source-commit {shlex.quote(source_commit)}" if source_commit else ""
    def timed(name: str, command: str) -> str:
        return f"/usr/bin/time -v -o {art}/{name}.time.txt {command}"

    hardware_unittest = timed("hardware_unittest", "python3 -m unittest discover -s tests -p test_pan_hardware_resolver.py")
    mutable_hil_unittest = timed("mutable_hil_unittest", "python3 -m unittest discover -s tests -p 'test_mutable_hil*.py'")
    kv260_smoke = timed("kv260_smoke", "python3 -m pan_hardware_resolver.kv260_smoke")
    replay = "python3 -m pan_hardware_resolver.replay"
    kv260_smoke_replay = timed("kv260_smoke_replay", replay)
    hil_both = timed("hil_both", "python3 -m pan_hardware_resolver.hil")
    hil_both_replay = timed("hil_both_replay", replay)
    mutable_hil_runner = timed("mutable_hil_runner", "python3 -m pan_hardware_resolver.mutable_hil_runner")
    mutable_hil_runner_replay = timed("mutable_hil_runner_replay", replay)
    action_bundle = timed("action_bundle", "python3 -m pan_isa_x86_runtime_spf.action_resolver_bundle")
    action_bundle_replay = timed("action_bundle_replay", replay)

    return [
        RemoteStep(
            "board_identity",
            (
                f"mkdir -p {art} && "
                f"(cat /proc/device-tree/model 2>/dev/null || hostname) > {art}/board_model.txt && "
                f"uname -a > {art}/uname.txt && "
                f"python3 --version > {art}/python_version.txt 2>&1 && "
                f"(xmutil listapps > {art}/xmutil_listapps.txt 2>&1 || true)"
            ),
            ("board_model.txt", "uname.txt", "python_version.txt", "xmutil_listapps.txt"),
        ),
        RemoteStep(
            "hardware_resolver_unittest",
            f"{py} {hardware_unittest} > {art}/hardware_unittest.log 2>&1",
            ("hardware_unittest.log", "hardware_unittest.time.txt"),
        ),
        RemoteStep(
            "mutable_hil_unittest",
            f"{py} {mutable_hil_unittest} > {art}/mutable_hil_unittest.log 2>&1",
            ("mutable_hil_unittest.log", "mutable_hil_unittest.time.txt"),
        ),
        RemoteStep(
            "kv260_safe_smoke",
            (
                f"{py} {kv260_smoke} "
                f"--expect-model {shlex.quote(expected_model)} "
                f"--write-ledger-jsonl {art}/kv260_smoke_ledger.jsonl "
                f"> {art}/kv260_smoke.json 2> {art}/kv260_smoke.stderr"
            ),
            ("kv260_smoke.json", "kv260_smoke.stderr", "kv260_smoke_ledger.jsonl", "kv260_smoke.time.txt"),
        ),
        RemoteStep(
            "kv260_safe_smoke_replay",
            (
                f"{py} {kv260_smoke_replay} "
                f"{art}/kv260_smoke_ledger.jsonl --write {art}/kv260_smoke_replay.json "
                f"> {art}/kv260_smoke_replay.stdout 2> {art}/kv260_smoke_replay.stderr"
            ),
            ("kv260_smoke_replay.json", "kv260_smoke_replay.stdout", "kv260_smoke_replay.stderr", "kv260_smoke_replay.time.txt"),
        ),
        RemoteStep(
            "benign_hil_suite",
            (
                f"{py} {hil_both} "
                f"--target both --expect-model {shlex.quote(expected_model)} "
                f"--write {art}/hil_both.json --write-ledger-jsonl {art}/hil_both_ledger.jsonl "
                f"> {art}/hil_both.stdout 2> {art}/hil_both.stderr"
            ),
            ("hil_both.json", "hil_both_ledger.jsonl", "hil_both.stdout", "hil_both.stderr", "hil_both.time.txt"),
        ),
        RemoteStep(
            "benign_hil_suite_replay",
            (
                f"{py} {hil_both_replay} "
                f"{art}/hil_both_ledger.jsonl --write {art}/hil_both_replay.json "
                f"> {art}/hil_both_replay.stdout 2> {art}/hil_both_replay.stderr"
            ),
            ("hil_both_replay.json", "hil_both_replay.stdout", "hil_both_replay.stderr", "hil_both_replay.time.txt"),
        ),
        RemoteStep(
            "synthetic_mutable_hil_runner",
            (
                f"{py} {mutable_hil_runner} "
                f"--write-template {art}/mutable_hil_runner_playbook.json "
                f"--arm-synthetic --expect-model {shlex.quote(expected_model)} "
                f"--write {art}/mutable_hil_runner_report.json "
                f"--write-ledger-jsonl {art}/mutable_hil_runner_ledger.jsonl "
                f"> {art}/mutable_hil_runner.stdout 2> {art}/mutable_hil_runner.stderr"
            ),
            (
                "mutable_hil_runner_playbook.json",
                "mutable_hil_runner_report.json",
                "mutable_hil_runner_ledger.jsonl",
                "mutable_hil_runner.stdout",
                "mutable_hil_runner.stderr",
                "mutable_hil_runner.time.txt",
            ),
        ),
        RemoteStep(
            "synthetic_mutable_hil_runner_replay",
            (
                f"{py} {mutable_hil_runner_replay} "
                f"{art}/mutable_hil_runner_ledger.jsonl --write {art}/mutable_hil_runner_replay.json "
                f"> {art}/mutable_hil_runner_replay.stdout 2> {art}/mutable_hil_runner_replay.stderr"
            ),
            ("mutable_hil_runner_replay.json", "mutable_hil_runner_replay.stdout", "mutable_hil_runner_replay.stderr", "mutable_hil_runner_replay.time.txt"),
        ),
        RemoteStep(
            "action_resolver_bundle",
            (
                f"{py} {action_bundle} "
                f"--artifact-dir {art}/action_bundle{commit_arg} "
                f"> {art}/action_bundle.stdout 2> {art}/action_bundle.stderr"
            ),
            ("action_bundle", "action_bundle.stdout", "action_bundle.stderr", "action_bundle.time.txt"),
        ),
        RemoteStep(
            "action_resolver_bundle_replay",
            (
                f"{py} {action_bundle_replay} "
                f"{art}/action_bundle/action_ledger.jsonl --write {art}/action_bundle/replay_cli.json "
                f"> {art}/action_bundle_replay.stdout 2> {art}/action_bundle_replay.stderr"
            ),
            ("action_bundle/replay_cli.json", "action_bundle_replay.stdout", "action_bundle_replay.stderr", "action_bundle_replay.time.txt"),
        ),
    ]


def build_manifest(
    *,
    host: str,
    remote_repo: str,
    remote_artifact_dir: str,
    source_commit: str | None,
    expected_model: str,
    dry_run: bool,
    sync: bool,
    step_results: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    steps = validation_steps(
        remote_repo=remote_repo,
        remote_artifact_dir=remote_artifact_dir,
        expected_model=expected_model,
        source_commit=source_commit,
    )
    results = list(step_results or [])
    return {
        "schema": VALIDATION_SCHEMA,
        "ok": bool((dry_run and not results) or (results and all(result["returncode"] == 0 for result in results))),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "host": host,
        "remote_repo": remote_repo,
        "remote_artifact_dir": remote_artifact_dir,
        "source_commit": source_commit,
        "expected_model": expected_model,
        "dry_run": bool(dry_run),
        "synced_committed_tree": bool(sync and not dry_run),
        "local_environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "safety": {
            "ps_side_only": True,
            "loads_pl_app": False,
            "raw_mmio": False,
            "real_device_mutable_writes": False,
            "synthetic_mutable_runner_only": True,
        },
        "steps": [step.to_dict() for step in steps],
        "step_results": results,
        "artifacts": sorted({artifact for step in steps for artifact in step.artifacts} | {"validation_manifest.json"}),
    }


def run_validation(
    *,
    host: str,
    remote_repo: str,
    remote_artifact_dir: str,
    source_commit: str | None = None,
    expected_model: str = "KV260",
    dry_run: bool = False,
    sync: bool = True,
) -> dict[str, Any]:
    if dry_run:
        return build_manifest(
            host=host,
            remote_repo=remote_repo,
            remote_artifact_dir=remote_artifact_dir,
            source_commit=source_commit,
            expected_model=expected_model,
            dry_run=True,
            sync=sync,
        )
    if sync:
        _sync_committed_tree(host, remote_repo)
    step_results = []
    for step in validation_steps(
        remote_repo=remote_repo,
        remote_artifact_dir=remote_artifact_dir,
        expected_model=expected_model,
        source_commit=source_commit,
    ):
        step_results.append(_run_remote_step(host, step))
        if step_results[-1]["returncode"] != 0:
            break
    manifest = build_manifest(
        host=host,
        remote_repo=remote_repo,
        remote_artifact_dir=remote_artifact_dir,
        source_commit=source_commit,
        expected_model=expected_model,
        dry_run=False,
        sync=sync,
        step_results=step_results,
    )
    _write_remote_json(host, f"{remote_artifact_dir.rstrip('/')}/validation_manifest.json", manifest)
    return manifest


def write_json(payload: dict[str, Any], path: str | Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path is None:
        print(text, end="")
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")


def _sync_committed_tree(host: str, remote_repo: str) -> None:
    archive = subprocess.Popen(["git", "archive", "--format=tar", "HEAD"], stdout=subprocess.PIPE)
    try:
        remote = subprocess.run(
            ["ssh", host, f"mkdir -p {shlex.quote(remote_repo)} && tar -xf - -C {shlex.quote(remote_repo)}"],
            stdin=archive.stdout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False,
            check=False,
        )
    finally:
        if archive.stdout is not None:
            archive.stdout.close()
    archive_rc = archive.wait()
    if archive_rc != 0:
        raise RuntimeError(f"git archive failed with exit code {archive_rc}")
    if remote.returncode != 0:
        stderr = remote.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"remote archive extract failed with exit code {remote.returncode}: {stderr}")


def _run_remote_step(host: str, step: RemoteStep) -> dict[str, Any]:
    proc = subprocess.run(
        ["ssh", host, step.command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    return {
        "name": step.name,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": step.command,
    }


def _write_remote_json(host: str, remote_path: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    parent = str(Path(remote_path).parent)
    proc = subprocess.run(
        ["ssh", host, f"mkdir -p {shlex.quote(parent)} && cat > {shlex.quote(remote_path)}"],
        input=text,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"failed to write remote manifest: {proc.stderr}")


def _git_commit() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the KV260 PS-side PAN action-resolver validation suite.")
    parser.add_argument("--host", default="kv260")
    parser.add_argument("--remote-repo", default="/home/ubuntu/pan-isa-kv260-router/pan-isa")
    parser.add_argument("--remote-artifact-dir", required=True)
    parser.add_argument("--source-commit", help="Source commit to record; defaults to local HEAD.")
    parser.add_argument("--expect-model", default="KV260")
    parser.add_argument("--no-sync", action="store_true", help="Do not sync git archive HEAD to the remote repository before running.")
    parser.add_argument("--dry-run", action="store_true", help="Write the planned manifest without contacting the board.")
    parser.add_argument("--write", help="Write local manifest JSON; stdout when omitted.")
    args = parser.parse_args(argv)
    manifest = run_validation(
        host=args.host,
        remote_repo=args.remote_repo,
        remote_artifact_dir=args.remote_artifact_dir,
        source_commit=args.source_commit or _git_commit(),
        expected_model=args.expect_model,
        dry_run=bool(args.dry_run),
        sync=not bool(args.no_sync),
    )
    write_json(manifest, args.write)
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
