"""Two new Saturn learners and a matched six-model/native ELF timing cohort."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cases_at(root):
    return json.loads((Path(root) / "cases.json").read_text())


def counter_program(steps):
    if steps < 3:
        raise ValueError("counter requires XOR, MOV and SYSCALL")
    # Explicit initialization works both under the supplied loader and Linux.
    return bytes.fromhex("31ff") + bytes.fromhex("4883c701") * (steps - 3) + bytes.fromhex("b83c0000000f05")


def child(args):
    entered = time.perf_counter()
    import torch
    from execution_study import execute

    from isa_lab.data import digest
    from isa_lab.full_training import restore
    from isa_lab.measurement import Usage
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    config = json.loads(Path("study.json").read_text())
    cases = cases_at(".")
    artifact = json.loads(args.artifact.read_text())
    with Usage() as loading:
        model = restore(artifact)
    first_case = next(x for x in cases if x["id"] == f"counter-{max(config['study']['lengths'])}")
    first_response = json.loads(execute(model, first_case))
    first = {"child_entry_to_response_s": time.perf_counter() - entered,
             "loading": loading.result, "response": first_response}
    # The first request warms the model. Warm each remaining program once.
    for case in cases:
        execute(model, case)
    rows = []
    for repeat in range(config["study"]["repeats"]):
        shift = (repeat + config["seed"]) % len(cases)
        for case in cases[shift:] + cases[:shift]:
            phases = {}
            with Usage() as usage:
                response_text = execute(model, case, phases)
            response = json.loads(response_text)
            rows.append({"case": case["id"], "steps_requested": case["steps"], "repeat": repeat,
                         **usage.result, **phases, "response": response,
                         "response_sha256": digest(response), "response_bytes": len(response_text.encode())})
    if digest(model.export()) != digest(artifact["payload"]):
        raise ValueError("serving mutated the frozen model")
    write(args.out, {"name": args.name, "seed": config["seed"], "artifact_sha256": artifact["sha256"],
                     "suite_sha256": artifact["suite_sha256"], "first": first, "samples": rows,
                     "environment": {"host": platform.node(), "platform": platform.platform(),
                                     "python": platform.python_version(), "torch": torch.__version__,
                                     "backend": "cpu", "threads": 1, "gpu": False},
                     "scope": "resident model: ELF setup, predicted steps, state/page hashes and JSON; model load/imports excluded; scoring afterward"})


def native(out, config):
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("native benchmark requires Linux x86_64")
    rows = []
    for case in cases_at("."):
        path = Path(case["elf_path"])
        path.chmod(0o755)
        command = [str(path.resolve())]
        for repeat in range(-1, config["study"]["native_repeats"]):
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            started = time.perf_counter()
            completed = subprocess.run(command, capture_output=True, timeout=5, check=False)
            wall = time.perf_counter() - started
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            row = {"case": case["id"], "steps_requested": case["steps"], "repeat": repeat,
                   "wall_s": wall, "cpu_s": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                   "exit_code": completed.returncode, "stdout": completed.stdout.hex(),
                   "stderr": completed.stderr.hex(), "elf_sha256": sha(path),
                   "correct": completed.returncode == case["expected_exit"]
                              and completed.stdout.hex() == case["expected_stdout"] and not completed.stderr}
            if repeat >= 0:
                rows.append(row)
    source = Path("native_asm/kernel.c")
    binary = out / "native-kernel"
    compiler = subprocess.run(["cc", "--version"], text=True, capture_output=True, check=True).stdout
    command = ["cc", "-O3", "-std=c11", str(source), "-o", str(binary)]
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    build = subprocess.run(command, capture_output=True, text=True, check=True)
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    output = subprocess.run([str(binary.resolve())], capture_output=True, text=True, check=True)
    kernel = [json.loads(line) for line in output.stdout.splitlines()]
    write(out / "native.json", {"samples": rows, "resident_kernel": kernel,
          "build": {"compiler": compiler, "command": command, "wall_s": wall,
                    "cpu_s": after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
                    "binary_sha256": sha(binary), "binary_bytes": binary.stat().st_size,
                    "source_sha256": sha(source), "stdout": build.stdout, "stderr": build.stderr},
          "environment": {"host": platform.node(), "platform": platform.platform()},
          "scope": "same ELF bytes executed by Linux; subprocess creation, exec, stdout capture and wait included; warm kernel separate"})


def worker():
    from campaign_worker import verify
    from execution_study import execute
    from mrun.artifacts import fetch_object

    from isa_lab.full_custody import pack, publish
    verify()
    config = json.loads(Path("study.json").read_text())
    seed = config["seed"]
    out = Path("result")
    out.mkdir(exist_ok=False)
    processes = []
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               CUDA_VISIBLE_DEVICES="-1", ISA_MRUN_WORKER="1", PYTHONPATH=str(ROOT))

    def invoke(command, key):
        started = time.perf_counter()
        print("ISA_STAGE:" + json.dumps({"key": key, "phase": "start"}), flush=True)
        completed = subprocess.run(command, env=env, check=False, timeout=config["study"]["child_timeout_s"])
        row = {"key": key, "returncode": completed.returncode, "process_wall_s": time.perf_counter() - started}
        processes.append(row)
        print("ISA_STAGE:" + json.dumps({**row, "phase": "complete"}), flush=True)
        if completed.returncode:
            raise RuntimeError(f"isolated process {key} exited {completed.returncode}; retained, no retry")

    try:
        pointer = json.loads(Path("specimen-archive.json").read_text())
        archive_path = fetch_object(pointer["sha256"], Path("baseline.tar.gz"))
        with tarfile.open(archive_path, "r:gz") as archive:
            names = [f"suite-{seed}.json"]
            for model in config["study"]["models"]:
                if model not in config["study"]["new_models"]:
                    names += [f"{model}-{seed}/{filename}" for filename in
                              ("model.json", "resources.json", "heldout.json", "training.json", "continuation-validation.json")]
            for name in names:
                target = out / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(name).read())
        suite = out / f"suite-{seed}.json"
        if sha(suite) != config["original_suite_file_sha256"]:
            raise ValueError("the frozen baseline suite bytes changed")
        # Oracle and expected observable checks are outside all model requests.
        references = {}
        for case in cases_at("."):
            reference = json.loads(execute(None, case))
            if not reference["completed"] or reference["steps"] != case["steps"]:
                raise ValueError(f"native loader instrument incomplete: {case['id']}")
            if reference["stdout"] != case["expected_stdout"]:
                raise ValueError("native loader stdout instrument disagrees")
            references[case["id"]] = reference
        write(out / "references.json", references)
        write(out / "cases.json", cases_at("."))
        native(out, config)
        publish((out / "native.json",), "isa-native-binary")
        for name in config["study"]["new_models"]:
            target = out / f"{name}-{seed}"
            invoke([sys.executable, "full_child.py", "--mode", "train", "--name", name,
                    "--suite", str(suite), "--seed", str(seed), "--out", str(target),
                    "--intervals", str(config["study"]["intervals"]), "--steps", str(config["study"]["steps"]),
                    "--validation-steps", str(config["study"]["validation_steps"])], f"train-{name}-{seed}")
            publish(tuple(target.rglob("*.json")), f"trained-{name}-{seed}")
        names = config["study"]["models"]
        shift = config["order_offset"] % len(names)
        for name in names[shift:] + names[:shift]:
            target = out / f"{name}-{seed}/timing.json"
            invoke([sys.executable, "ssm_campaign.py", "child", "--name", name,
                    "--artifact", str(target.with_name("model.json")), "--out", str(target)], f"timing-{name}-{seed}")
            publish((target,), f"timing-{name}-{seed}")
        verify()
    finally:
        write(out / "processes.json", processes)
        write(out / "study.json", config)
        pack(out)


def seal(run, config):
    run.mkdir(parents=True, exist_ok=False)
    payload = run / "source"
    frozen = ROOT / f"outputs/seed{config['seed']}/source"
    shutil.copytree(frozen, payload)
    for name in ("source-manifest.json", "preflight.v1.json"):
        (payload / name).unlink()
    # All original classes/functions keep identical AST bodies. Only the factory
    # receives new branches, and the name list is additive.
    def bodies(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name != "create"}
    old = bodies(frozen / "isa_lab/full_models.py")
    new = bodies(ROOT / "isa_lab/full_models.py")
    if old != new:
        raise ValueError("original model implementation changed; cannot reuse baseline accuracy")
    def original_factory(path):
        node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "create")
        node.body = [n for n in node.body if not (isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                     and isinstance(n.test.comparators[0], ast.Constant) and n.test.comparators[0].value in ("mamba", "hydra"))]
        return ast.dump(node, include_attributes=False)
    if original_factory(frozen / "isa_lab/full_models.py") != original_factory(ROOT / "isa_lab/full_models.py"):
        raise ValueError("original model factory changed")
    original_validation = (frozen / "isa_lab/full_validation.py").read_text()
    fixed_validation = (original_validation.replace('as stream:', 'as writer:')
                        .replace('json.dump(program, stream,', 'json.dump(program, writer,')
                        .replace('stream.write(json.dumps', 'writer.write(json.dumps'))
    if fixed_validation != (ROOT / "isa_lab/full_validation.py").read_text():
        raise ValueError("validation changed beyond the diagnostic stream-name correction")
    write(payload / "contract-parity.json", {"original_model_ast_equal": True,
          "original_factory_ast_equal": True,
          "validation_patch": "gzip writer variable renamed so first-divergence reports retain input-only stream",
          "validation_original_sha256": sha(frozen / "isa_lab/full_validation.py"),
          "validation_fixed_sha256": sha(ROOT / "isa_lab/full_validation.py"),
          "base_job": config["baseline_job"], "frozen_source": str(frozen.relative_to(ROOT)),
          "original_suite_file_sha256": config["original_suite_file_sha256"],
          "scope": "all original selector definitions and original native consumer/training/vendor bytes preserved"})
    for name in ("full_models.py", "ssm.py", "full_validation.py"):
        shutil.copy2(ROOT / "isa_lab" / name, payload / "isa_lab" / name)
    for name in ("mamba_isa", "hydra_isa", "native_asm"):
        shutil.copytree(ROOT / name, payload / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("ssm_campaign.py", "execution_study.py"):
        shutil.copy2(ROOT / "scripts" / name, payload / name)
    from execution_study import stdout_program
    from mrun.artifacts import ledger

    from isa_lab.full_validation import elf_bytes
    pointer = [x for x in ledger(config["baseline_job"]) if x.get("key") == "isa-full-final-archive"][-1]["value"]["archive"]
    write(payload / "specimen-archive.json", pointer)
    programs = payload / "programs"
    programs.mkdir()
    cases = []
    for steps in config["study"]["lengths"]:
        stdout_code, expected = stdout_program(steps)
        for name, code, stdout, exit_code in (("counter", counter_program(steps), b"", (steps - 3) & 255),
                                            ("stdout", stdout_code, expected, 0)):
            path = f"programs/{name}-{steps}.elf"
            (payload / path).write_bytes(elf_bytes(code + bytes(32)))
            (payload / f"programs/{name}-{steps}.S").write_text(
                ".global _start\n.text\n_start:\n  .byte " + ",".join(f"0x{x:02x}" for x in code) + "\n")
            cases.append({"id": f"{name}-{steps}", "kind": "elf", "steps": steps, "elf_path": path,
                          "elf_sha256": sha(payload / path), "expected_stdout": stdout.hex(), "expected_exit": exit_code})
    write(payload / "cases.json", cases)
    write(payload / "study.json", config)
    roots = ["isa_lab", "nn", "pan_isa", "fieldlm", "spf_transformer", "mamba_isa", "hydra_isa", "native_asm",
             "vendor", "programs", "ssm_campaign.py", "execution_study.py", "full_child.py", "campaign_worker.py",
             "cases.json", "study.json", "specimen-archive.json", "contract-parity.json"]
    write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result", "result.tar.gz"], "hashed_roots": roots})
    write(payload / "source-manifest.json", {str(p.relative_to(payload)): sha(p) for p in sorted(payload.rglob("*")) if p.is_file()})
    return payload


def launch(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        config = OmegaConf.to_container(compose(config_name="ssm_campaign"), resolve=True)
    base = ROOT / f"outputs/seed{args.seed}"
    config.update({"seed": args.seed, "order_offset": args.order_offset,
                   "baseline_job": json.loads((base / "submission.json").read_text())["job_id"],
                   "original_suite_file_sha256": sha(base / f"result/suite-{args.seed}.json"), "smoke": args.smoke})
    if args.smoke:
        config["study"].update({"intervals": 1, "steps": 8, "validation_steps": 64,
                                "lengths": [16], "repeats": 1, "native_repeats": 3})
    from isa_lab.data import digest
    config["config_sha256"] = digest(config)
    payload = seal(args.run_dir, config)
    write(args.run_dir / "resolved-config.json", config)
    from isa_lab.dependencies import WORKSPACE, namespace
    namespace("saturn", WORKSPACE / "saturn/src/saturn")
    from saturn.job_support import guarded_mrun_launch, one_lease_sequential_models_scheduling_contract
    request = {"task_family": "training", "device": "cpu", "needs": {"cpu": True}, "config": config,
        "scheduling_contract": one_lease_sequential_models_scheduling_contract(model_processes=8,
            task_family="training", checkpoint_policy="two Saturn-controlled learners, four reused frozen learners, six fresh-process inference passes"),
        "measured_resource_exception": {"source_job": "job-f487f2dcf33a", "tree_peak_rss_mb": 1600,
            "reservation_ram_mb": 3072, "reason": "same serial parent/child CPU1 closure; smoke verifies extra tiny SSM buffers"}}
    command = ["/usr/bin/env", "UV_CACHE_DIR=/tmp/isa-uv-cache", "uv", "run", "--project", "{env}",
               "--with", "unicorn==2.1.4", "--with", "psutil>=5.9", "python", "ssm_campaign.py", "worker"]
    write(args.run_dir / "request.json", {"request": request, "command": command})
    job = guarded_mrun_launch(command, request=request, experiment="isa-ssm-native-cpu", task="training", model=None,
        payload=payload, prefer_host="beast", gpu=False, needs=request["needs"], env_alias="mrun", preflight=True,
        retry_on_kill=False, timeout_s=config["study"]["timeout_s"], ram_mb=3072, vram_mb=0, cpu_threads=1,
        est_wall_s=100 if args.smoke else 650,
        note=f"ISA seed {args.seed}: two selective-state CPU1 learners, {config['study']['intervals'] * config['study']['steps']} updates each; "
             "Saturn feedback and retained futures; six frozen-model matched ELF timings + native Linux and compiled C/asm. "
             "Eight serial model children, no GPU/retries/nested jobs. Stop at heldout/continuation/timing archive.")
    write(args.run_dir / "submission.json", {"job_id": job})
    print(json.dumps({"job_id": job, "run_dir": str(args.run_dir)}), flush=True)


def run(args):
    import datetime

    from mrun.client.api import Api
    args.run_dir = ROOT / "runs" / ("ssm-" + str(args.seed) + "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    launch(args)
    job = json.loads((args.run_dir / "submission.json").read_text())["job_id"]
    api = Api()
    while True:
        receipt = api.json("GET", f"/api/jobs/{job}")
        if receipt["state"] in {"succeeded", "failed", "timeout", "cancelled", "lost", "killed_ram", "killed_vram", "killed"}:
            break
        time.sleep(30)
    subprocess.run([sys.executable, str(ROOT / "scripts/campaign_collect.py"), str(args.run_dir)], check=True)
    if receipt["state"] != "succeeded":
        raise RuntimeError(f"study ended {receipt['state']}; raw evidence retained at {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("source", "result"):
        shutil.copytree(args.run_dir / name, args.out / name)
    for name in ("resolved-config.json", "request.json", "submission.json", "receipt.json", "job.log"):
        shutil.copy2(args.run_dir / name, args.out / name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("launch", "worker", "child", "run"))
    p.add_argument("--seed", type=int, default=673)
    p.add_argument("--order-offset", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--name")
    p.add_argument("--artifact", type=Path)
    args = p.parse_args()
    {"launch": lambda: launch(args), "worker": worker, "child": lambda: child(args), "run": lambda: run(args)}[args.mode]()


if __name__ == "__main__":
    main()
