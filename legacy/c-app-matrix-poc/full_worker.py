"""One finite lease, four serial model processes and rotated frozen timing."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).resolve().parent
if root.name == "scripts":
    root = root.parent
sys.path.insert(0, str(root))
from campaign_worker import verify

from isa_lab.data import digest
from isa_lab.full_custody import pack, publish
from isa_lab.full_models import MODEL_NAMES
from isa_lab.full_scope import generate, inputs_only


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path("campaign.json"))
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    verify()
    out = Path("result")
    out.mkdir(exist_ok=False)
    attempts = []
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "CUDA_VISIBLE_DEVICES": "-1", "PYTHONPATH": str(root), "ISA_MRUN_WORKER": "1"})
    os.environ["ISA_MRUN_WORKER"] = "1"

    def invoke(command, key):
        t = time.perf_counter()
        print("ISA_STAGE:" + json.dumps({"key": key, "phase": "start"}), flush=True)
        proc = subprocess.run(command, env=env, check=False, timeout=config["child_timeout_s"])
        attempts.append({"key": key, "returncode": proc.returncode, "process_wall_s": time.perf_counter() - t})
        if proc.returncode:
            raise RuntimeError(f"{key} exited {proc.returncode}; no retry")

    try:
        for si, seed in enumerate(config["seeds"]):
            print("ISA_STAGE:" + json.dumps({"key": f"suite-{seed}", "phase": "generate"}), flush=True)
            suite = generate(seed, config["examples"], config["trace_length"])
            path = out / f"suite-{seed}.json"
            path.write_text(json.dumps(suite, sort_keys=True) + "\n")
            input_manifest = {"suite_sha256": suite["sha256"], "inputs": inputs_only(suite["splits"]["heldout"])}
            input_manifest["sha256"] = digest(input_manifest)
            input_path = out / f"serving-inputs-{seed}.json"
            input_path.write_text(json.dumps(input_manifest, sort_keys=True) + "\n")
            publish((path, input_path), f"suite-{seed}")
            offset = (si + config.get("order_offset", 0)) % 4
            for name in MODEL_NAMES[offset:] + MODEL_NAMES[:offset]:
                target = out / f"{name}-{seed}"
                invoke([sys.executable, "full_child.py", "--mode", "train", "--name", name,
                        "--suite", str(path), "--seed", str(seed), "--out", str(target),
                        "--intervals", str(config["intervals"]), "--steps", str(config["steps"]),
                        "--validation-steps", str(config.get("validation_steps", 0))], f"train-{name}-{seed}")
                publish(target.rglob("*"), f"model-{name}-{seed}")
            for ri in range(config["timing_rounds"]):
                shift = (offset + ri) % 4
                for name in MODEL_NAMES[shift:] + MODEL_NAMES[:shift]:
                    target = out / f"{name}-{seed}"
                    invoke([sys.executable, "full_child.py", "--mode", "infer", "--name", name,
                            "--suite", str(input_path), "--seed", str(seed), "--out", str(target / f"timing-{ri}.json"),
                            "--artifact", str(target / "model.json"), "--repeats", str(config["repeats"])], f"infer-{name}-{seed}-{ri}")
                    publish((target / f"timing-{ri}.json",), f"timing-{name}-{seed}-{ri}")
        verify()
    finally:
        (out / "campaign.json").write_text(json.dumps(config, indent=2) + "\n")
        (out / "processes.json").write_text(json.dumps(attempts, indent=2) + "\n")
        pack(out)


if __name__ == "__main__":
    main()
