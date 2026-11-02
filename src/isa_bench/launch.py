"""Stage a payload, submit it to mrun on Beast, wait, and fetch the result.

Payload layout (the job's cwd on Beast):
  src/isa_bench/          harness (worker + cell)
  programs/               C sources + tools        (reference stage)
  models/<family>/        adapter (+ train.py)     (eval / train)
  models/_vendor/isa_r3/  frozen shared sources    (selector / direct_effect)
  <artifact>              the model artifact       (eval)
  job.json                stage request

Reference traces stay on Beast (``store_root``); eval jobs read them in place.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .adapters import REPO, tree_sha256

WORKSPACE = Path(os.environ.get("ISA_BENCH_WORKSPACE", REPO.parent))  # set it when running from a git worktree
STORE = "/mnt/hdd8tb/uv-archive-stage/isa-bench"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")


def stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%dT%H%M%S")


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str))


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stage_payload(run_dir: Path, job: dict, *, programs: bool = False, families: tuple = (), vendor: bool = False,
                  artifacts: tuple = ()) -> Path:
    payload = run_dir / "payload"
    payload.mkdir(parents=True)
    shutil.copytree(REPO / "src/isa_bench", payload / "src/isa_bench", ignore=IGNORE)
    if programs:
        shutil.copytree(REPO / "programs", payload / "programs", ignore=IGNORE)
    for family in families:
        shutil.copytree(REPO / "models" / family, payload / "models" / family, ignore=IGNORE, symlinks=False)
    if vendor:
        shutil.copytree(REPO / "models/_vendor/isa_r3", payload / "models/_vendor/isa_r3", ignore=IGNORE)
    for artifact in dict.fromkeys(a for a in artifacts if a):
        src = REPO / artifact
        if not src.exists():
            raise FileNotFoundError(f"artifact missing locally: {artifact} (run `just pull` or `dvc pull`)")
        (payload / artifact).parent.mkdir(parents=True, exist_ok=True)
        (shutil.copytree if src.is_dir() else shutil.copy2)(src, payload / artifact)
    write(payload / "job.json", job)
    roots = sorted(p.name for p in payload.iterdir())
    write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result"], "hashed_roots": roots})
    files = {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    write(payload / "source-manifest.json", files)
    return payload


def _saturn():
    sys.path.insert(0, str(WORKSPACE / "isa"))
    from isa_lab.dependencies import namespace  # the isa repo's explicit source mount helper

    namespace("saturn", WORKSPACE / "saturn/src/saturn")
    from saturn import job_support

    return job_support


def tunnel_env() -> None:
    """Point Python clients at the SSH tunnels (`just tunnels`); macOS blocks Python from the LAN directly."""
    os.environ.setdefault("MRUN_URL", os.environ.get("ISA_BENCH_MRUN_URL", "http://127.0.0.1:19025"))
    if os.environ.get("MRUN_URL", "").startswith("http://192.168."):
        os.environ["MRUN_URL"] = os.environ.get("ISA_BENCH_MRUN_URL", "http://127.0.0.1:19025")


def submit(run_dir: Path, payload: Path, job: dict, *, executor: dict, experiment: str, task: str, note: str,
           ram_mb: int, cpu_threads: int = 2, gpu: bool = False, vram_mb: int = 0, est_wall_s: int = 600,
           timeout_s: int | None = None, model_processes: int = 1) -> str:
    tunnel_env()
    js = _saturn()
    needs = {"cuda": True} if gpu else {"cpu": True}
    request = {"schema": "isa-bench-request-v1", "task_family": task, "device": "cuda" if gpu else "cpu",
               "needs": needs, "config": {k: v for k, v in job.items() if k != "model"},
               "model_name": job.get("model", {}).get("name"),
               "source_manifest_sha256": sha(payload / "source-manifest.json"),
               "scheduling_contract": js.one_lease_sequential_models_scheduling_contract(
                   model_processes=model_processes, task_family=task, max_local_work_items=max(8, model_processes),
                   checkpoint_policy="isa-bench: one frozen model per fresh cell subprocess; no fitting during eval")
               if model_processes >= 2 else js.one_lease_scheduling_contract()}
    env = ["/usr/bin/env", "UV_CACHE_DIR=/tmp/isa-uv-cache", "UV_OFFLINE=1", "PYTHONPATH=src", "OMP_NUM_THREADS=1",
           "MKL_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1"]
    if not gpu:
        env.append("CUDA_VISIBLE_DEVICES=")
    cmd = [*env, "uv", "run", "--project", "{env}"]
    for w in executor.get("uv_with", []):
        cmd += ["--with", w]
    cmd += ["python", "-m", "isa_bench.worker"]
    write(run_dir / "request.json", {"request": request, "command": cmd})
    job_id = js.guarded_mrun_launch(
        cmd, request=request, experiment=experiment, task=task, model=None, payload=payload,
        prefer_host=executor.get("host", "beast"), pin=executor.get("host", "beast"), needs=needs, gpu=gpu,
        env_alias=executor.get("env_alias", "mrun"), preflight=True, retry_on_kill=False,
        timeout_s=int(timeout_s or max(600, est_wall_s * 2)), ram_mb=ram_mb, vram_mb=vram_mb if gpu else 0,
        cpu_threads=cpu_threads, est_wall_s=est_wall_s, priority=int(executor.get("priority", 50)), note=note)
    write(run_dir / "submission.json", {"job_id": job_id, "submitted": stamp()})
    print(json.dumps({"job_id": job_id, "run_dir": str(run_dir)}), flush=True)
    return job_id


def wait(job_id: str, run_dir: Path, *, max_wait_s: float = 21600) -> dict:
    tunnel_env()
    js = _saturn()
    current, meta, log = js.terminal_job(job_id, wait=True, max_wait_s=max_wait_s)
    (run_dir / "job.log").write_text(log)
    write(run_dir / "job.json", meta)
    state = str(meta.get("state"))
    print(json.dumps({"job_id": current, "state": state}), flush=True)
    return meta


def fetch(job_id: str, dest: Path, *, executor: dict) -> Path:
    """rsync the job's result/ directory back (mrun keeps job dirs under remote_work)."""
    remote = f"{executor.get('host', 'beast')}:{executor.get('remote_work', '/mnt/hdd8tb/mrun-work')}/{job_id}/result/"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--delete", remote, str(dest) + "/"], check=True, timeout=3600)
    return dest


def remote_result(job_id: str, executor: dict) -> str:
    return f"{executor.get('remote_work', '/mnt/hdd8tb/mrun-work')}/{job_id}/result"


def publish_store(job_id: str, kind: str, key: str, *, executor: dict) -> str:
    """Copy a finished job's result into the Beast store (stable path that later jobs read in place)."""
    target = f"{STORE}/{kind}/{key}"
    src = remote_result(job_id, executor)
    host = executor.get("host", "beast")
    subprocess.run(["ssh", host, f"mkdir -p {STORE}/{kind} && rm -rf {target}.tmp && cp -a {src} {target}.tmp "
                    f"&& rm -rf {target} && mv {target}.tmp {target}"], check=True, timeout=1800)
    return target


def run_job(stage: str, cfg: dict, job: dict, *, name: str, payload_kwargs: dict, resources: dict, note: str,
            fetch_to: Path | None, model_processes: int = 1) -> dict:
    executor = cfg["executor"]
    run_dir = REPO / executor.get("stage_dir", "runs") / stage / name / stamp()
    run_dir.mkdir(parents=True)
    payload = stage_payload(run_dir, job, **payload_kwargs)
    job_id = submit(run_dir, payload, job, executor=executor, experiment=f"isa-bench-{stage}", task=resources["task"],
                    note=note, ram_mb=resources["ram_mb"], cpu_threads=resources.get("cpu_threads", 2),
                    gpu=resources.get("gpu", False), vram_mb=resources.get("vram_mb", 0),
                    est_wall_s=resources.get("est_wall_s", 600), timeout_s=resources.get("timeout_s"),
                    model_processes=model_processes)
    receipt = {"job_id": job_id, "run_dir": str(run_dir), "payload_sha256": tree_sha256(payload)}
    if not executor.get("wait", True):
        return receipt
    meta = wait(job_id, run_dir, max_wait_s=executor.get("max_wait_s", 21600))
    receipt["state"] = meta.get("state")
    if meta.get("state") != "succeeded":
        write(run_dir / "receipt.json", receipt)
        raise RuntimeError(f"mrun job {job_id} ended {meta.get('state')}: see {run_dir}/job.log")
    if fetch_to is not None:
        fetch(job_id, fetch_to, executor=executor)
        receipt["fetched_to"] = str(fetch_to)
    write(run_dir / "receipt.json", receipt)
    shutil.copy2(payload / "source-manifest.json", run_dir / "source-manifest.json")
    shutil.rmtree(payload, ignore_errors=True)  # payload is content-addressed in source-manifest + mrun custody
    return receipt
