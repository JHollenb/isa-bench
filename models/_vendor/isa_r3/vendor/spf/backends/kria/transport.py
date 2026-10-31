"""Shared SSH/SCP transport for KV260 board dispatch.

Two dispatch styles converge here so the ssh + scp + remote-tmp + cleanup
choreography lives in exactly one place instead of being re-implemented per
caller:

* :func:`run_inline_script` — push input ``.npy`` files, run a Python script
  piped over ``ssh`` stdin (the per-call pyxrt path used by
  :mod:`spf.backends.kria.pl_dispatch`), pull output ``.npy`` files back.
* :func:`run_agent_job` — pipe a JSON job to ``agent.py`` on the board, parse
  its JSON reply, and (optionally) pull the ``.npy`` outputs it wrote (the
  contract-driven path used by ``hw/platforms/kria/host/runner.py`` and the
  latency bench).

Both honour the ``SPF_KRIA_HOST`` / ``SPF_KRIA_WORKDIR`` env contract via
:func:`resolve_host` / :func:`resolve_workdir`. Every helper is wrapped so a
unit test can monkeypatch :mod:`subprocess` and assert the choreography
without an attached board.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_WORKDIR = "/home/ubuntu/spfr"
DEFAULT_IO_TIMEOUT = 120.0
_CLEANUP_TIMEOUT = 30.0


def resolve_host(host: str | None = None) -> str:
    """Return the board ssh target, falling back to ``$SPF_KRIA_HOST``.

    Raises ``RuntimeError`` (not a bare ``KeyError``) so callers surface the
    bring-up hint rather than a stack trace when the board is not configured.
    """
    resolved = host or os.environ.get("SPF_KRIA_HOST")
    if not resolved:
        raise RuntimeError(
            "SPF_KRIA_HOST is not set; point it at 'user@board-ip' after "
            "flashing the board per hw/platforms/kria/docs/board-bringup.md."
        )
    return resolved


def resolve_workdir(workdir: str | None = None) -> str:
    """Return the remote workdir, falling back to ``$SPF_KRIA_WORKDIR``."""
    return workdir or os.environ.get("SPF_KRIA_WORKDIR", DEFAULT_WORKDIR)


def _ssh(
    host: str,
    command: str,
    *,
    check: bool,
    timeout: float | None,
    stdin: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["ssh", host, command],
        input=stdin,
        check=check,
        capture_output=True,
        timeout=timeout,
    )


def _scp(sources: Sequence[str], dest: str, *, timeout: float | None) -> None:
    subprocess.run(["scp", "-q", *sources, dest], check=True, timeout=timeout)


def run_inline_script(
    host: str,
    script: bytes,
    *,
    infiles: Mapping[str, np.ndarray],
    outfiles: Sequence[str],
    extra_args: Sequence[str] = (),
    timeout: float,
    io_timeout: float = DEFAULT_IO_TIMEOUT,
    prefix: str = "spf-pl-",
) -> dict[str, np.ndarray]:
    """Run a board-side Python script over ssh, shuttling ``.npy`` files.

    ``infiles`` are saved as ``uint32`` arrays into a fresh remote temp dir;
    the script is piped to ``python3 - <remote_tmp> <extra_args...>`` on the
    board; ``outfiles`` are pulled back and returned keyed by filename. The
    remote and local temp dirs are always cleaned up.

    ``timeout`` bounds the script run (callers scale it with batch size);
    ``io_timeout`` bounds the cheaper mkdir/scp steps.
    """
    local_tmp = Path(tempfile.mkdtemp(prefix=prefix))
    remote_tmp = f"/tmp/{prefix}{uuid.uuid4().hex[:8]}"
    try:
        for name, arr in infiles.items():
            np.save(local_tmp / name, np.ascontiguousarray(arr, dtype=np.uint32))
        _ssh(host, f"mkdir -p {remote_tmp}", check=True, timeout=io_timeout)
        _scp(
            [str(local_tmp / name) for name in infiles],
            f"{host}:{remote_tmp}/",
            timeout=io_timeout,
        )
        argv = " ".join(shlex.quote(a) for a in (remote_tmp, *extra_args))
        _ssh(host, f"python3 - {argv}", check=True, timeout=timeout, stdin=script)
        _scp(
            [f"{host}:{remote_tmp}/{name}" for name in outfiles],
            str(local_tmp),
            timeout=io_timeout,
        )
        return {name: np.load(local_tmp / name) for name in outfiles}
    finally:
        shutil.rmtree(local_tmp, ignore_errors=True)
        _ssh(host, f"rm -rf {remote_tmp}", check=False, timeout=_CLEANUP_TIMEOUT)


def run_agent_job(
    host: str,
    workdir: str,
    job: Mapping[str, Any],
    *,
    fetch_outputs: bool = False,
    timeout: float | None = None,
    label: str = "kria agent",
    ssh_opts: Sequence[str] = (),
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Dispatch a JSON job to ``agent.py`` on the board and parse its reply.

    Pipes ``job`` as JSON to ``python3 <workdir>/agent.py`` over ssh, parses
    the single JSON document the agent prints on stdout (tolerating preceding
    warnings), and raises ``RuntimeError`` with ``label`` context on ssh
    failure, unparseable output, or a non-``ok`` status. When
    ``fetch_outputs`` is set, the ``.npy`` files named in ``result["outputs"]``
    are scp'd back and returned keyed by their name minus the ``.npy`` suffix.
    The remote ``job["out_dir"]`` is always cleaned up after a dispatch.

    ``ssh_opts`` are extra ssh flags inserted before the host (e.g.
    ``("-o", "BatchMode=yes")`` for an unattended CI gate that must fail fast
    rather than block on an auth prompt). They apply to both the dispatch and
    the cleanup ssh calls.
    """
    remote_out = str(job["out_dir"])
    ssh_cmd = f"mkdir -p {remote_out} && python3 {workdir}/agent.py"
    try:
        proc = subprocess.run(
            ["ssh", *ssh_opts, host, ssh_cmd],
            input=json.dumps(job),
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"{label}: ssh {host} failed (rc={exc.returncode})\nstderr:\n{exc.stderr}"
        ) from exc

    try:
        last_line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            result: dict[str, Any] = json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"{label}: agent returned unparseable stdout:\n{proc.stdout}"
            ) from exc
        if result.get("status") != "ok":
            raise RuntimeError(f"{label}: agent reported error: {result}")

        outputs: dict[str, np.ndarray] = {}
        if fetch_outputs:
            outputs = _fetch_agent_outputs(host, remote_out, result["outputs"])
        return result, outputs
    finally:
        subprocess.run(
            ["ssh", *ssh_opts, host, f"rm -rf {remote_out}"],
            check=False,
            capture_output=True,
        )


def _fetch_agent_outputs(
    host: str, remote_out: str, names: Sequence[str]
) -> dict[str, np.ndarray]:
    local_tmp = Path(tempfile.mkdtemp(prefix="spf-kria-"))
    try:
        _scp(
            [f"{host}:{remote_out}/{name}" for name in names],
            str(local_tmp),
            timeout=None,
        )
        outputs: dict[str, np.ndarray] = {}
        for name in names:
            key = name[: -len(".npy")] if name.endswith(".npy") else name
            outputs[key] = np.load(local_tmp / name)
        return outputs
    finally:
        shutil.rmtree(local_tmp, ignore_errors=True)
