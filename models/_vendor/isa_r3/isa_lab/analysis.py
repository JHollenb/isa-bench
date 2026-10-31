"""Artifact-only aggregation of accuracy, resource use and rotated timing samples."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .data import digest
from .evaluate import ratio
from .models import MODEL_NAMES


def analyze(result_dir):
    result_dir = Path(result_dir)
    config = json.loads((result_dir / "campaign.json").read_text())
    summary = {"schema": "isa-comparison-summary-v1", "profile": config["profile"], "track": config["track"],
               "seeds": config["seeds"], "models": {}, "terminal_status": "not-certified", "mlflow": "skipped"}
    paired = {}
    environments = []
    for seed in config["seeds"]:
        reports = [json.loads((result_dir / f"{name}-{seed}/heldout.json").read_text()) for name in MODEL_NAMES]
        for key in ("suite_sha256", "profile", "track", "contract_sha256", "oracle", "split"):
            if len({report[key] for report in reports}) != 1:
                raise ValueError(f"comparison identity mismatch for seed {seed}: {key}")
        if len({digest(report["environment"]) for report in reports}) != 1:
            raise ValueError("training execution environments differ")
    for name in MODEL_NAMES:
        total = {lane: {"correct": 0, "total": 0} for lane in ("isolated", "rollout", "trace_endpoints")}
        per_seed = []
        timing = []
        cpu = []
        peaks = []
        all_resources = []
        physical_examples = []
        artifact_bytes = []
        for seed in config["seeds"]:
            root = result_dir / f"{name}-{seed}"
            report = json.loads((root / "heldout.json").read_text())
            resources = json.loads((root / "resources.json").read_text())
            train = json.loads((root / "training.json").read_text())
            environment = resources["environment"]
            if digest(environment) != digest(report["environment"]):
                raise ValueError("resource and accuracy environments differ")
            environments.append(environment)
            artifact_bytes.append(report["artifact_bytes"])
            for lane, counts in total.items():
                for key in counts:
                    counts[key] += report["measurement"][lane][key]
            physical_examples.append(train["physical_updates"] * train["discovery_examples"])
            replay_digest = digest([row["predicted"] for row in report["measurement"]["rows"]
                                    if row["lane"] in ("isolated", "rollout")])
            per_seed.append({"seed": seed, "accuracy": {lane: ratio(report["measurement"][lane]) for lane in total},
                             "physical_updates": train["physical_updates"], "accepted_updates": train["accepted_updates"],
                             "train_full": resources["training_full"], "train_phases": train["timing"],
                             "artifact_bytes": report["artifact_bytes"], "learned_payload": resources["learned_payload"]})
            all_resources.append(resources)
            for path in sorted(root.glob("timing-*.json")):
                trial = json.loads(path.read_text())
                if trial["suite_sha256"] != report["suite_sha256"] or trial["artifact_sha256"] != report["artifact_sha256"]:
                    raise ValueError("timing and model/suite identity differ")
                if digest(trial["environment"]) != digest(environment):
                    raise ValueError("timing execution environment differs")
                samples = trial["samples"]
                if any(row["outputs_sha256"] != replay_digest for row in samples):
                    raise ValueError("fresh-process serving differs from captured heldout predictions")
                timing.extend(row["us_per_instruction"] for row in samples)
                cpu.extend(row["cpu_s"] / row["instructions"] * 1e6 for row in samples)
                peaks.extend(row["sampled_peak_rss_mb"] for row in samples)
                paired[(name, seed, trial["round"])] = statistics.median(row["us_per_instruction"] for row in samples)
        entry = {"accuracy_counts": total, "accuracy": {lane: ratio(counts) for lane, counts in total.items()},
                 "per_seed": per_seed, "timed_samples": len(timing),
                 "warm_inference_us_median": statistics.median(timing),
                 "warm_inference_us_min": min(timing), "warm_inference_us_max": max(timing),
                 "warm_cpu_us_per_instruction_median": statistics.median(cpu),
                 "inference_sampled_peak_rss_mb_max": max(peaks),
                 "training_full_wall_s_mean": statistics.mean(r["training_full"]["wall_s"] for r in all_resources),
                 "training_full_cpu_s_mean": statistics.mean(r["training_full"]["cpu_s"] for r in all_resources),
                 "training_sampled_peak_rss_mb_max": max(r["training_full"]["sampled_peak_rss_mb"] for r in all_resources),
                 "training_process_peak_rss_mb_max": max(r["training_full"]["process_lifetime_peak_rss_mb"] for r in all_resources),
                 "update_wall_s_mean": statistics.mean(r["training_phases"]["updates"]["wall_s"] for r in all_resources),
                 "label_prepare_wall_s_mean": statistics.mean(r["training_phases"]["label_prepare"]["wall_s"] for r in all_resources),
                 "physical_training_examples_mean": statistics.mean(physical_examples),
                 "artifact_bytes_mean": statistics.mean(artifact_bytes)}
        summary["models"][name] = entry
    if len({digest(env) for env in environments}) != 1:
        raise ValueError("campaign host/runtime differs across seeds/models")
    summary["environment"] = environments[0]
    summary["paired_timing_ratios"] = {}
    for numerator, denominator in (("fieldlm", "nn"), ("spf_transformer", "nn"), ("fieldlm", "spf_transformer")):
        rows = [{"seed": seed, "round": index,
                 "ratio": paired[(numerator, seed, index)] / paired[(denominator, seed, index)]}
                for seed in config["seeds"] for index in range(config["timing_rounds"])]
        summary["paired_timing_ratios"][f"{numerator}/{denominator}"] = {
            "median": statistics.median(row["ratio"] for row in rows), "rows": rows}
    return summary


def markdown(summary):
    env = summary["environment"]
    lines = ["# ISA training and inference comparison", "",
             f"Profile: `{summary['profile']}`. Track: `{summary['track']}`. Seeds: {summary['seeds']}.",
             f"Host: `{env['host']}`; CPU, one Torch/BLAS thread, scalar inference. MLflow was skipped.", "",
             "| Model | Isolated exact | Rollout step exact | Final state exact | Warm inference µs/instruction |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, row in summary["models"].items():
        a = row["accuracy"]
        lines.append(f"| {name} | {a['isolated']:.3%} | {a['rollout']:.3%} | {a['trace_endpoints']:.3%} | "
                     f"{row['warm_inference_us_median']:.2f} |")
    lines.extend(["", "| Model | Training wall s | Training CPU s | Update wall s | Train peak RSS MiB | Infer peak RSS MiB | Artifact KiB |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for name, row in summary["models"].items():
        lines.append(f"| {name} | {row['training_full_wall_s_mean']:.3f} | {row['training_full_cpu_s_mean']:.3f} | "
                     f"{row['update_wall_s_mean']:.3f} | {row['training_sampled_peak_rss_mb_max']:.1f} | "
                     f"{row['inference_sampled_peak_rss_mb_max']:.1f} | {row['artifact_bytes_mean'] / 1024:.2f} |")
    lines.extend(["", ("Training wall/CPU are means over seeds for the complete fit, including model preparation, label inference, "
                  "Saturn snapshots, feedback replay and artifact writing. Runtime imports and cold artifact loads are separately "
                  "recorded in resources/timing JSON. Update time isolates the learner callback; shared label preparation is also separate."), "",
                  ("Inference medians use rotated model order and repeated warm trials in isolated serial processes. The timed path "
                  "includes model routing, supplied candidate semantics, typed commit and returned state construction; gold scoring and "
                  "artifact writes occur outside it. Output fingerprints must match captured heldout predictions in every fresh process, warmup and repetition."), "",
                  ("RSS is sampled every 10 ms and reported in MiB; process-lifetime high-water marks are retained separately. All models "
                  "import Torch for Saturn's state boundary, so process RSS includes the common Python/Torch runtime. No GPU is used. "
                  "CPU seconds are measured process consumption; energy and hardware FLOP counters are unmeasured."), "",
                  ("This task learns operation selection on eleven known x86 forms. Candidate arithmetic, operand boundaries and "
                  "record/address structure are supplied. FieldLM/SPF perform one structural fit; NN uses the declared bounded "
                  "gradient budget. These results do not rank full FieldLM/SPFTransformer architectures or learned arithmetic."), "",
                  ("Raw scores, checkpoint futures, decisions, resources and timing rows remain in the captured result directory. "
                  "Terminal certification was not run."), ""])
    return "\n".join(lines)
