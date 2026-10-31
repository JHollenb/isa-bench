"""An isolated resident model process in the single full-scope mrun lease."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if root.name == "scripts":
    root = root.parent
sys.path.insert(0, str(root))


def emit(value):
    print("ISA_EVENT:" + json.dumps(value, sort_keys=True, allow_nan=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("train", "infer"), required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--artifact", type=Path)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--intervals", type=int, default=4)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--validation-steps", type=int, default=0)
    args = p.parse_args()
    from isa_lab.measurement import Usage
    with Usage() as imports:
        import torch

        from isa_lab.data import digest
        from isa_lab.full_evaluate import evaluate, scores, serve
        from isa_lab.full_scope import Consumer
        from isa_lab.full_training import fit, model_artifact, restore
        from isa_lab.training import write_json
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    suite = json.loads(args.suite.read_text())
    if digest({k: v for k, v in suite.items() if k != "sha256"}) != suite["sha256"]:
        raise ValueError("suite identity mismatch")
    environment = {"host": platform.node(), "python": platform.python_version(), "torch": torch.__version__,
                   "cpu_threads": 1, "backend": "cpu", "gpu": False}
    if args.mode == "train":
        with Usage() as usage:
            model, consumer, _summary = fit(args.name, suite["splits"]["discovery"], suite["splits"]["feedback"],
                 seed=args.seed, suite_hash=suite["sha256"], out=args.out, intervals=args.intervals, steps=args.steps, emit=emit)
        identity = model_artifact(model, args.seed, suite["sha256"])["sha256"]
        with Usage() as testing:
            heldout = evaluate(model, suite["splits"]["heldout"], consumer)
        if model_artifact(model, args.seed, suite["sha256"])["sha256"] != identity:
            raise ValueError("evaluation mutated learned state")
        write_json(args.out / "heldout.json", {"suite_sha256": suite["sha256"], "artifact_sha256": identity,
                                              "environment": environment, "measurement": heldout})
        write_json(args.out / "serving-receipt.json", {"suite_sha256": suite["sha256"], "artifact_sha256": identity,
                                                     "serving_digest": heldout["serving_digest"], "environment": environment})
        write_json(args.out / "resources.json", {"training": usage.result, "heldout": testing.result,
              "imports": imports.result, "environment": environment,
              "artifact_bytes": (args.out / "model.json").stat().st_size,
              "learned_state": {k: v for k, v in model.export().items()
                                if k in ("parameter_count", "learned_record_count", "packed_plane_bytes")}})
        if args.validation_steps:
            from isa_lab.full_validation import static_elf, uninterrupted
            validation = {"static_elf": static_elf(model, args.seed + 80000),
                          "uninterrupted": uninterrupted(model, consumer, args.seed + 90000, args.validation_steps, out=args.out)}
            write_json(args.out / "continuation-validation.json", validation)
            emit({"event": "continuation", "name": args.name, "seed": args.seed,
                  "elf_correct": validation["static_elf"]["correct"],
                  "continuous_correct": validation["uninterrupted"]["correct"],
                  "continuous_steps": args.validation_steps,
                  "instrument_status": validation["uninterrupted"]["instrument_status"]})
        emit({"event": "trained", "name": args.name, "seed": args.seed, **scores(heldout), **usage.result})
    else:
        with Usage() as loading:
            artifact = json.loads(args.artifact.read_text())
            model = restore(artifact)
            consumer = Consumer()
        inputs = suite["inputs"]
        if artifact["suite_sha256"] != suite["suite_sha256"]:
            raise ValueError("serving inputs/model suite lineage mismatch")
        warmed = serve(model, inputs, consumer)
        reference_digest = digest(warmed)
        samples = []
        component_samples = []
        for repeat in range(args.repeats):
            with Usage() as usage:
                actual = serve(model, inputs, consumer)
            if digest(actual) != reference_digest:
                raise ValueError("warm scalar output parity drift")
            total = (sum(len(t) for ts in actual["stages"].values() for t in ts)
                     + sum(r.get("executed_steps", 1) for r in actual["runtime"])
                     + (len(actual["cumulative"]) if isinstance(actual.get("cumulative"), list) else 0))
            attempted = (sum(len(t["steps"]) for ts in inputs["stages"].values() for t in ts)
                         + sum(r.get("max_steps", 1) for r in inputs["runtime"])
                         + len(inputs.get("cumulative", {}).get("operations", [])))
            samples.append({**usage.result, "instructions": attempted, "executed_instructions": total,
                            "us_per_instruction": usage.result["wall_s"] * 1e6 / attempted,
                            "executed_us_per_instruction": usage.result["wall_s"] * 1e6 / total})
        from isa_lab.full_benchmark import components
        component_samples.append(components(model, inputs, consumer, warmed))
        frozen = json.loads((args.artifact.parent / "serving-receipt.json").read_text())
        if frozen["serving_digest"] != reference_digest:
            raise ValueError("fresh process predictions differ from heldout capture")
        if model_artifact(model, args.seed, suite["suite_sha256"])["sha256"] != artifact["sha256"]:
            raise ValueError("serving mutated learned state")
        write_json(args.out, {"name": args.name, "seed": args.seed, "suite_sha256": suite["suite_sha256"],
              "artifact_sha256": artifact["sha256"], "serving_digest": reference_digest,
              "environment": environment, "imports": imports.result, "loading": loading.result,
              "samples": samples, "component_samples": component_samples,
              "scope": "scalar native selection/semantics/typed runtime commit; no gold scoring; runtime fixture construction included"})
        emit({"event": "timed", "name": args.name, "seed": args.seed, "samples": samples})


if __name__ == "__main__":
    main()
