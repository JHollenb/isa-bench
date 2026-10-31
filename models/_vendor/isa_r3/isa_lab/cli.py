from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

from .contracts import PROFILE, TRACK
from .data import SPLITS, digest, generate, validate_suite
from .evaluate import evaluate
from .models import MODEL_NAMES, restore


def load(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Shared ISA model comparison")
    sub = parser.add_subparsers(dest="command", required=True)
    data = sub.add_parser("generate")
    data.add_argument("--out", type=Path, required=True)
    data.add_argument("--seed", type=int, default=17)
    data.add_argument("--oracle", choices=("unicorn", "python"), default="unicorn")
    data.add_argument("--examples-per-form", type=int, default=32)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=MODEL_NAMES, required=True)
    train.add_argument("--suite", type=Path, required=True)
    train.add_argument("--out", type=Path, required=True)
    train.add_argument("--seed", type=int, default=17)
    train.add_argument("--intervals", type=int, default=4)
    train.add_argument("--steps", type=int, default=32)
    train.add_argument("--mechanics-only", action="store_true")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--model", type=Path, required=True)
    ev.add_argument("--suite", type=Path, required=True)
    ev.add_argument("--out", type=Path, required=True)
    ev.add_argument("--split", choices=SPLITS, default="heldout")
    comp = sub.add_parser("compare")
    comp.add_argument("reports", type=Path, nargs="+")
    comp.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "generate":
        from .training import write_json
        suite = generate(args.seed, args.examples_per_form, oracle=args.oracle)
        write_json(args.out, suite)
        print(json.dumps({"suite": str(args.out), "sha256": suite["sha256"], "oracle": suite["oracle"]}))
    elif args.command == "train":
        # Full training is fleet-only. A tightly capped local check verifies mechanics.
        if os.environ.get("ISA_MRUN_WORKER") != "1" and not args.mechanics_only:
            parser.error("submit through scripts/launch.py; local execution requires --mechanics-only")
        suite = validate_suite(load(args.suite))
        if args.mechanics_only and (args.intervals != 1 or args.steps != 1
                                   or len(suite["splits"]["discovery"]["observations"]) > 88):
            parser.error("local mechanics cap: 88 discovery examples, one interval, one update")
        from .training import fit
        _, summary = fit(args.model, suite["splits"]["discovery"], suite["splits"]["feedback"],
                         suite_sha256=suite["sha256"], seed=args.seed, out=args.out,
                         intervals=args.intervals, steps=args.steps)
        print(json.dumps({"run": str(args.out), "model": args.model,
                          "accepted_updates": summary["accepted_updates"]}))
    elif args.command == "evaluate":
        from .training import contract_fingerprint, write_json
        suite = validate_suite(load(args.suite))
        artifact = load(args.model)
        if artifact.get("sha256") != digest({k: v for k, v in artifact.items() if k != "sha256"}):
            parser.error("model artifact hash mismatch")
        if artifact["suite_sha256"] != suite["sha256"]:
            parser.error("model and evaluation suite identity differ")
        model = restore(artifact)
        report = {"schema": "isa-report-v1", "profile": PROFILE, "track": TRACK, "model": artifact["model"],
                  "seed": artifact["seed"], "suite_sha256": suite["sha256"], "split": args.split,
                  "oracle": suite["oracle"], "contract_sha256": contract_fingerprint(),
                  "artifact_sha256": artifact["sha256"], "artifact_bytes": args.model.stat().st_size,
                  "backend": "cpu", "terminal_status": "not-certified",
                  "environment": {"python": platform.python_version(), "machine": platform.machine(),
                                  "platform": platform.platform()},
                  "measurement": evaluate(model, suite["splits"][args.split])}
        write_json(args.out, report)
        print(json.dumps({"report": str(args.out), "model": model.name, "split": args.split}))
    else:
        from .comparison import compare
        text = compare([load(path) for path in args.reports])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(text)
        print(text)


if __name__ == "__main__":
    main()
