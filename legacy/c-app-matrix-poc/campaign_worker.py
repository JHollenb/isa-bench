"""Finite serial model campaign; no nested scheduler submissions or automatic retries."""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

root = Path(__file__).resolve().parent
if root.name == "scripts":
    root = root.parent
sys.path.insert(0, str(root))

from isa_lab.data import generate
from isa_lab.models import MODEL_NAMES


def verify():
    manifest = json.loads(Path("source-manifest.json").read_text())
    for name, expected in manifest.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"sealed source changed: {name}")


def pack_result(out):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(out)), recursive=False)
    raw = buffer.getvalue()
    Path("result.tar.gz").write_bytes(raw)
    # Direct requested-job collection remains available if artifact publication is offline.
    encoded = base64.b64encode(raw).decode()
    for index in range(0, len(encoded), 60000):
        print(f"ISA_ARCHIVE_CHUNK:{index // 60000}:" + encoded[index:index + 60000], flush=True)
    print("ISA_ARCHIVE_SHA256:" + hashlib.sha256(raw).hexdigest(), flush=True)
    try:
        from mrun.outbox import default
        box = default()
        blob = box.put_file("result.tar.gz", "result.tar.gz")
        box.record("isa-campaign-result", {"archive": blob, "sha256": hashlib.sha256(raw).hexdigest()})
    except Exception as exc:  # noqa: BLE001 - direct archive survives unavailable publication
        print("ISA_PUBLICATION_DIAGNOSTIC:" + json.dumps({"error": type(exc).__name__}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("campaign.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    verify()
    out = Path("result")
    out.mkdir(exist_ok=False)
    attempts = []
    child_env = dict(os.environ)
    child_env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                      "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(root), "ISA_MRUN_WORKER": "1"})

    def invoke(command, key):
        started = time.perf_counter()
        print("ISA_STAGE:" + json.dumps({"key": key, "phase": "start"}), flush=True)
        # stdout inherits the mrun log so checkpoint events can be tracked live.
        status = subprocess.run(command, env=child_env, check=False, timeout=config["child_timeout_s"]).returncode
        attempts.append({"key": key, "returncode": status, "process_wall_s": time.perf_counter() - started})
        print("ISA_STAGE:" + json.dumps({"key": key, "phase": "complete", **attempts[-1]}), flush=True)
        if status:
            raise RuntimeError(f"isolated process {key} exited {status}; no retry")

    try:
        for seed_index, seed in enumerate(config["seeds"]):
            suite = generate(seed, config["examples_per_form"], config["traces_per_split"], oracle="unicorn")
            suite_path = out / f"suite-{seed}.json"
            suite_path.write_text(json.dumps(suite, indent=2, sort_keys=True) + "\n")
            order = MODEL_NAMES[seed_index % 3:] + MODEL_NAMES[:seed_index % 3]
            for name in order:
                target = out / f"{name}-{seed}"
                invoke([sys.executable, "campaign_child.py", "--mode", "train", "--model", name,
                        "--seed", str(seed), "--suite", str(suite_path), "--out", str(target),
                        "--intervals", str(config["intervals"]), "--steps", str(config["steps"])],
                       f"train-{name}-{seed}")
            for round_index in range(config["timing_rounds"]):
                shift = (seed_index + round_index) % 3
                order = MODEL_NAMES[shift:] + MODEL_NAMES[:shift]
                for position, name in enumerate(order):
                    target = out / f"{name}-{seed}"
                    output = target / f"timing-{round_index}.json"
                    invoke([sys.executable, "campaign_child.py", "--mode", "infer", "--model", name,
                            "--seed", str(seed), "--suite", str(suite_path), "--artifact", str(target / "model.json"),
                            "--out", str(output), "--round", str(round_index), "--warmups", str(config["warmups"]),
                            "--repeats", str(config["timing_repeats"])], f"infer-{name}-{seed}-{round_index}")
                    data = json.loads(output.read_text())
                    data["rotated_order_position"] = position
                    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        verify()
    finally:
        (out / "campaign.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        (out / "processes.json").write_text(json.dumps(attempts, indent=2, sort_keys=True) + "\n")
        pack_result(out)


if __name__ == "__main__":
    main()
