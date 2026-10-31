"""Frozen-model causal input sweep and complete multi-step response measurements."""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def stdout_program(total_steps):
    """Count with ADD, write the computed integer to stdout, exit zero."""
    if total_steps < 11:
        raise ValueError("stdout program requires eleven fixed instructions")
    increments = total_steps - 11
    code = (bytes.fromhex("31c0") + bytes.fromhex("4883c001") * increments
            + bytes.fromhex("4883ec0848890424bf010000004889e6ba08000000b8010000000f0531ffb83c0000000f05"))
    return code, increments.to_bytes(8, "little")


def execute(model, case, timings=None):
    """Input-only request to complete JSON response; no scoring or labels."""
    started = time.perf_counter()
    from isa_lab.full_evaluate import runtime_snapshot
    if case["kind"] == "stage3":
        from isa_lab.full_scope import Consumer
        state = copy.deepcopy(case["initial"])
        consumer = Consumer()
        setup_done = time.perf_counter()
        for instruction in case["instructions"]:
            state, _ = consumer.step(model, 3, instruction, state)
        stopped = time.perf_counter()
        response = {"state": state, "steps": len(case["instructions"]), "completed": True}
    else:
        from pan_isa.runtime_adapter import SharedRuntimeModel, runtime_fixture
        adapter = SharedRuntimeModel(model) if model is not None else None
        if case["kind"] == "elf":
            from pan_isa.native import mount
            mount()
            from pan_isa_x86_runtime_spf.runtime import SPFPrimaryPanRuntime
            runtime = SPFPrimaryPanRuntime.from_elf(Path(case["elf_path"]), model=adapter, backend="ps")
        else:
            runtime = runtime_fixture(case["fixture"])
            runtime.router.backend = "ps"
            if adapter is not None:
                runtime.model = adapter
        attempted = 0
        setup_done = time.perf_counter()
        for _ in range(case["steps"]):
            attempted += 1
            reason = runtime.step()
            if reason.kind != "running":
                break
        stopped = time.perf_counter()
        response = {**runtime_snapshot(runtime), "steps": attempted,
                    "completed": runtime.stop_reason is not None and runtime.stop_reason.kind == "exit"}
    # Full architectural state, mapped-page digests, stdout, stderr and stop reason.
    # JSON serialization is inside the response timer; printing/network transport is not.
    text = json.dumps(response, sort_keys=True, separators=(",", ":"))
    if timings is not None:
        timings.update({"setup_s": setup_done - started, "execution_s": stopped - setup_done,
                        "program_stop_s": stopped - started,
                        "diagnostic_finalize_s": time.perf_counter() - stopped,
                        "diagnostic_response_s": time.perf_counter() - started})
    return text


def literal_panel(model, inputs):
    import torch

    from isa_lab.full_models import TrainableSelectors
    from pan_isa.native import runtime_module
    labels = runtime_module().RUNTIME_NEURAL_LABELS
    if isinstance(model, TrainableSelectors):
        with torch.inference_mode():
            logits = model.heads["runtime"](torch.tensor([x["raw"] for x in inputs]),
                                             torch.tensor([x["numeric"] for x in inputs]))
            confidence, predicted = logits.softmax(-1).max(-1)
        return [{"label": labels[int(i)], "confidence": float(c)}
                for i, c in zip(predicted, confidence, strict=True)]
    return [{"label": name, "confidence": confidence}
            for row in inputs for name, confidence in [model.runtime_predict(row["raw"], row["numeric"])]]


def child(args):
    first_started = time.perf_counter()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from isa_lab.data import digest
    from isa_lab.full_training import restore
    config = json.loads(Path("study.json").read_text())["study"]
    cases = json.loads(Path(f"inputs/cases-{args.seed}.json").read_text())
    artifact_path = Path(f"specimens/{args.name}-{args.seed}/model.json")
    load_started = time.perf_counter()
    artifact = json.loads(artifact_path.read_text())
    model = restore(artifact)
    loading_s = time.perf_counter() - load_started
    # A fresh-process first request, same 1,024-instruction ELF for all models.
    first_case = next(x for x in cases if x["id"] == "elf-stdout-1024")
    first_response = execute(model, first_case)
    first = {"child_entry_to_response_s": time.perf_counter() - first_started,
             "loading_s": loading_s, "response": json.loads(first_response)}
    print("ISA_FIRST_RESPONSE:" + json.dumps({"model": args.name, "seed": args.seed,
          "completed": first["response"]["completed"], "child_entry_to_response_s": first["child_entry_to_response_s"]}), flush=True)
    for case in cases:
        execute(model, case)
    rows = []
    for repeat in range(config["repeats"]):
        shift = (repeat + args.seed) % len(cases)
        for case in cases[shift:] + cases[:shift]:
            t, c = time.perf_counter(), time.process_time()
            response_text = execute(model, case)
            wall, cpu = time.perf_counter() - t, time.process_time() - c
            response = json.loads(response_text)
            rows.append({"case": case["id"], "kind": case["kind"], "requested_steps": case["steps"],
                         "repeat": repeat, "wall_s": wall, "cpu_s": cpu, "response": response,
                         "response_sha256": digest(response), "response_bytes": len(response_text.encode())})
    inputs = json.loads(Path("inputs/literals.json").read_text())
    panels = {"final": literal_panel(model, inputs)}
    if args.name in ("nn", "pan_isa"):
        for future in range(5):
            value = json.loads(artifact_path.with_name(f"future-{future:03d}-model.json").read_text())
            candidate = restore(value)
            panels[f"updates-{future * 128}"] = literal_panel(candidate, inputs)
            del candidate
    if digest(model.export()) != digest(artifact["payload"]):
        raise ValueError("frozen serving mutated learned state")
    write(args.out, {"name": args.name, "seed": args.seed, "artifact_sha256": artifact["sha256"],
                     "first": first, "samples": rows, "literal_panels": panels,
                     "environment": {"host": os.uname().nodename, "backend": "cpu", "threads": 1,
                                     "python": sys.version, "torch": torch.__version__},
                     "scope": "resident model, request initialization/load, all predicted steps, full state/page digests and JSON response; no scoring, imports, model load, network transport"})


def worker():
    from campaign_worker import verify

    from isa_lab.full_custody import pack, publish
    from pan_isa.native import runtime_module
    from pan_isa.runtime_adapter import runtime_fixture
    verify()
    config = json.loads(Path("study.json").read_text())["study"]
    out = Path("result")
    out.mkdir(exist_ok=False)
    try:
        # Weights already live in mrun's CAS. Ship only source, input manifests and
        # archive identities through the scheduler's deliberately small payload.
        from mrun.artifacts import fetch_object
        archives = json.loads(Path("specimen-archives.json").read_text())
        Path("specimen-archives").mkdir()
        for seed in config["seeds"]:
            pointer = archives[str(seed)]
            archive_path = fetch_object(pointer["sha256"], Path(f"specimen-archives/{seed}.tar.gz"))
            with tarfile.open(archive_path, "r:gz") as archive:
                for name in config["models"]:
                    filenames = ["model.json"]
                    if name in ("nn", "pan_isa"):
                        filenames += [f"future-{i:03d}-model.json" for i in range(5)]
                    for filename in filenames:
                        target = Path(f"specimens/{name}-{seed}/{filename}")
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.extractfile(f"{name}-{seed}/{filename}").read())
        # Generate controlled inputs through the unchanged native parser and encoder.
        module = runtime_module()
        literals = []
        for form, opcode, width in (("eax32", "b8", 4), ("edi32", "bf", 4),
                                    ("rax64", "48b8", 8), ("rdi64", "48bf", 8)):
            for value in range(config["literal_values"]):
                fixture = {"code": opcode + value.to_bytes(width, "little").hex(), "seed": 40001, "options": {}}
                instance = runtime_fixture(fixture)
                instruction = module.decode_current(instance)
                raw, numeric = module.encode_runtime_features(instance, instruction)
                label = instance.model.propose(instance, instruction).instruction_name
                if label != "MOV_IMM_REG":
                    raise ValueError("literal sweep instrument changed native family")
                literals.append({"form": form, "value": value, "raw": raw, "numeric": numeric})
        write("inputs/literals.json", literals)
        write(out / "literal-inputs.json", literals)
        references = {}
        for seed in config["seeds"]:
            cases = json.loads(Path(f"inputs/cases-{seed}.json").read_text())
            references[str(seed)] = {}
            for case in cases:
                if case["kind"] != "stage3":
                    reference = json.loads(execute(None, case))
                    write(out / f"native-references/{seed}-{case['id']}.json", reference)
                    if not reference["completed"] or reference["steps"] != case["steps"]:
                        raise ValueError(f"native program instrument did not complete: {case['id']}")
                    if case["kind"] == "elf" and reference["stdout"] != case["expected_stdout"]:
                        raise ValueError("stdout arithmetic/loader instrument failed")
                    references[str(seed)][case["id"]] = reference
            write(out / f"cases-{seed}.json", cases)
        write(out / "references.json", references)
        process_rows = []
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="-1", ISA_MRUN_WORKER="1", PYTHONPATH=str(ROOT))
        for index, seed in enumerate(config["seeds"]):
            names = config["models"]
            names = names[index % 4:] + names[:index % 4]
            for name in names:
                command = [sys.executable, "execution_study.py", "child", "--name", name, "--seed", str(seed),
                           "--out", str(out / f"{name}-{seed}.json")]
                started = time.perf_counter()
                process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                timer = threading.Timer(config["child_timeout_s"], process.kill)
                timer.start()
                first_s = None
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        if line.startswith("ISA_FIRST_RESPONSE:"):
                            first_s = time.perf_counter() - started
                    status = process.wait()
                finally:
                    timer.cancel()
                process_rows.append({"name": name, "seed": seed, "process_wall_s": time.perf_counter() - started,
                                     "process_to_first_response_s": first_s, "returncode": status})
                if status or first_s is None:
                    raise RuntimeError(f"study child failed: {name}-{seed}")
                publish((out / f"{name}-{seed}.json",), f"study-{name}-{seed}")
        write(out / "processes.json", process_rows)
        write(out / "study.json", json.loads(Path("study.json").read_text()))
        verify()
    finally:
        pack(out)


def seal(run, config):
    """Use the exact prior frozen scientific source and weight artifacts."""
    run.mkdir(parents=True, exist_ok=False)
    payload = run / "source"
    shutil.copytree(ROOT / "outputs/seed673/source", payload)
    for filename in ("source-manifest.json", "preflight.v1.json"):
        (payload / filename).unlink()
    shutil.copy2(Path(__file__), payload / "execution_study.py")
    (payload / "inputs").mkdir()
    (payload / "programs").mkdir()
    from mrun.artifacts import ledger

    from isa_lab.full_scope import serving
    from isa_lab.full_validation import elf_bytes
    archives = {}
    for seed in config["study"]["seeds"]:
        base = ROOT / f"outputs/seed{seed}/result"
        job = json.loads((base.parent / "submission.json").read_text())["job_id"]
        entries = [x for x in ledger(job) if x.get("key") == "isa-full-final-archive"]
        archives[str(seed)] = entries[-1]["value"]["archive"]
        with gzip.open(base / f"nn-{seed}/uninterrupted-program.json.gz", "rt") as stream:
            program = json.load(stream)
        initial = program["initial_stage3"]
        cases = []
        for length in config["study"]["stage3_lengths"]:
            steps = program["steps"][:length]
            cases.append({"id": f"stage3-{length}", "kind": "stage3", "steps": length,
                          "initial": {k: initial[k] for k in ("regs", "flags", "memory")},
                          "instructions": [serving(3, step) for step in steps]})
        for length in config["study"]["elf_lengths"]:
            code, expected = stdout_program(length)
            path = f"programs/stdout-{length}.elf"
            # The unchanged native decoder fetches a 15-byte window. Map padding
            # beyond the exit so the final window can cross a page boundary.
            (payload / path).write_bytes(elf_bytes(code + bytes(32)))
            cases.append({"id": f"elf-stdout-{length}", "kind": "elf", "steps": length,
                          "elf_path": path, "elf_sha256": hashlib.sha256((payload / path).read_bytes()).hexdigest(),
                          "expected_stdout": expected.hex()})
        suite = json.loads((base / f"suite-{seed}.json").read_text())
        for row in suite["splits"]["heldout"]["runtime"]:
            if row.get("case", "").startswith("program-"):
                cases.append({"id": row["case"], "kind": "fixture", "steps": row["max_steps"],
                              "fixture": {k: row[k] for k in ("code", "seed", "options")}})
        write(payload / f"inputs/cases-{seed}.json", cases)
    write(payload / "specimen-archives.json", archives)
    write(payload / "study.json", config)
    write(payload / "preflight.v1.json", {"path_token_allowlist": ["{env}", "result", "result.tar.gz"],
                                           "hashed_roots": ["isa_lab", "nn", "pan_isa", "fieldlm", "spf_transformer",
                                                            "vendor", "specimen-archives.json", "inputs", "programs", "study.json", "execution_study.py"]})
    manifest = {str(p.relative_to(payload)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(payload.rglob("*")) if p.is_file() and p.name != "source-manifest.json"}
    write(payload / "source-manifest.json", manifest)
    return payload


def launch(args):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base="1.3"):
        config = OmegaConf.to_container(compose(config_name="execution_study"), resolve=True)
    from isa_lab.data import digest
    config["config_sha256"] = digest(config)
    payload = seal(args.run_dir, config)
    write(args.run_dir / "resolved-config.json", config)
    if not args.execute:
        print(json.dumps({"status": "sealed", "run_dir": str(args.run_dir)}))
        return
    from isa_lab.dependencies import WORKSPACE, namespace
    namespace("saturn", WORKSPACE / "saturn/src/saturn")
    from saturn.job_support import guarded_mrun_launch, one_lease_sequential_models_scheduling_contract
    request = {"task_family": "evaluation", "device": "cpu", "needs": {"cpu": True}, "config": config,
               "scheduling_contract": one_lease_sequential_models_scheduling_contract(model_processes=12,
                     task_family="evaluation", checkpoint_policy="frozen prior artifacts and Saturn futures; no training"),
               "measured_resource_exception": {"source_job": "job-f487f2dcf33a", "tree_peak_rss_mb": 1600,
                                                "reservation_ram_mb": 3072, "reason": "same serial CPU1 model/parent closure"}}
    command = ["/usr/bin/env", "UV_CACHE_DIR=/mnt/big/llm-models/.uv-cache", "uv", "run", "--project", "{env}",
               "--with", "unicorn==2.1.4", "--with", "psutil>=5.9", "python", "execution_study.py", "worker"]
    write(args.run_dir / "request.json", {"request": request, "command": command})
    job = guarded_mrun_launch(command, request=request, experiment="isa-program-execution-study", task="evaluation",
          model=None, payload=payload, prefer_host="beast", gpu=False, needs=request["needs"], env_alias="mrun",
          preflight=True, retry_on_kill=False, timeout_s=config["study"]["timeout_s"], ram_mb=3072, vram_mb=0,
          cpu_threads=1, est_wall_s=360,
          note="Frozen four-model, three-seed end-to-end program study; <=12 serial CPU1 children; "
               "3 repeats of 16/128/1024 ELF stdout programs and <=8192 Stage3 continuations, "
               "controlled MOV literals across retained Saturn futures. No training/GPU/retries. Stop after raw evidence archive.")
    write(args.run_dir / "submission.json", {"job_id": job})
    print(json.dumps({"job_id": job, "run_dir": str(args.run_dir)}))


def run(args):
    import datetime

    from mrun.client.api import Api
    args.run_dir = ROOT / "runs" / ("execution-study-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args.execute = True
    launch(args)
    job = json.loads((args.run_dir / "submission.json").read_text())["job_id"]
    api = Api()
    terminal = {"succeeded", "failed", "timeout", "cancelled", "lost", "killed_ram", "killed_vram", "killed"}
    while True:
        receipt = api.json("GET", f"/api/jobs/{job}")
        if receipt["state"] in terminal:
            break
        time.sleep(30)
    subprocess.run([sys.executable, str(ROOT / "scripts/campaign_collect.py"), str(args.run_dir)], check=True)
    if receipt["state"] != "succeeded":
        raise RuntimeError(f"study ended {receipt['state']}; raw attempt retained at {args.run_dir}")
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ("source", "result"):
        shutil.copytree(args.run_dir / name, args.out / name)
    for name in ("resolved-config.json", "request.json", "submission.json", "receipt.json", "job.log"):
        shutil.copy2(args.run_dir / name, args.out / name)
    subprocess.run([sys.executable, str(ROOT / "scripts/execution_analyze.py"), str(args.out),
                    "--metrics-out", str(ROOT / "metrics/execution-study.json")], check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("launch", "worker", "child", "run"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    {"launch": lambda: launch(args), "worker": worker, "child": lambda: child(args),
     "run": lambda: run(args)}[args.mode]()


if __name__ == "__main__":
    main()
