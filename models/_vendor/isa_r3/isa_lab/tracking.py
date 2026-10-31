"""MLflow lifecycle through common's credentials/configuration seam."""
from __future__ import annotations

import json
import time
from pathlib import Path


def client(uri):
    from common_harness.env import configure_environment
    configure_environment()
    from mrun.mlops.env import ensure_s3_env
    ensure_s3_env()  # common's artifact configuration, before importing MLflow
    from mlflow.tracking import MlflowClient
    return MlflowClient(tracking_uri=uri)


def initialize(run_dir, uri):
    run_dir = Path(run_dir)
    path = run_dir / "mlflow.json"
    if path.exists():
        return json.loads(path.read_text())
    submitted = json.loads((run_dir / "submission.json").read_text())
    request = json.loads((run_dir / "request.json").read_text())["request"]
    ml = client(uri)
    experiment = ml.get_experiment_by_name("isa-model-comparison")
    experiment_id = experiment.experiment_id if experiment else ml.create_experiment("isa-model-comparison")
    parent = ml.create_run(experiment_id, tags={"mlflow.runName": run_dir.name,
             "mrun.job_id": submitted["job_id"], "track": request["track"], "profile": request["profile"],
             "status_scope": "exploratory; terminal certification disabled"})
    children = {}
    for seed in request["seeds"]:
        for model in ("nn", "fieldlm", "spf_transformer"):
            run = ml.create_run(experiment_id, tags={"mlflow.runName": f"{model}-seed{seed}",
                                "mlflow.parentRunId": parent.info.run_id, "mrun.job_id": submitted["job_id"],
                                "model.family": model, "profile": request["profile"], "track": request["track"]})
            run_id = run.info.run_id
            for key, value in {"seed": seed, "cpu_threads": 1, "backend": "cpu", "oracle": request["oracle"],
                               "examples_per_form": request["examples_per_form"],
                               "intervals": request["intervals"] if model == "nn" else 1,
                               "updates_per_interval": request["steps"] if model == "nn" else 1,
                               "timing_rounds": request["timing_rounds"],
                               "timing_repeats": request["timing_repeats"]}.items():
                ml.log_param(run_id, key, value)
            children[f"{model}-{seed}"] = run_id
    value = {"tracking_uri": uri, "experiment_id": experiment_id, "parent_run_id": parent.info.run_id,
             "runs": children, "job_id": submitted["job_id"], "log_offset": 0, "pending_line": ""}
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return value


def log_event(ml, mapping, event):
    run_id = mapping["runs"][f"{event['model']}-{event['seed']}"]
    step = int(event.get("step", event.get("round", 0)))
    timestamp = int(time.time() * 1000)

    def metric(key, value):
        ml.log_metric(run_id, key, float(value), timestamp=timestamp, step=step)

    kind = event["event"]
    if kind == "feedback":
        for key in ("isolated_exact", "rollout_exact", "final_exact"):
            metric("feedback." + key, event[key])
    elif kind == "update":
        if "loss" in event:
            metric("train.loss", event["loss"])
        metric("train.update_wall_s", event["cost"]["wall_s"])
        metric("train.update_cpu_s", event["cost"]["cpu_s"])
    elif kind == "decision":
        metric("continuation.promoted", int(event["promoted"]))
    elif kind == "trained":
        for key in ("heldout_isolated_exact", "heldout_rollout_exact", "heldout_final_exact",
                    "training_wall_s", "training_cpu_s", "training_peak_rss_mb"):
            metric(key, event[key])
    elif kind == "inference":
        metric("inference.artifact_load_wall_s", event["artifact_load_wall_s"])
        for row in event["samples"]:
            sample_step = event["round"] * len(event["samples"]) + row["repeat"]
            for key in ("us_per_instruction", "cpu_s", "sampled_peak_rss_mb", "instructions_per_s"):
                ml.log_metric(run_id, "inference." + key, float(row[key]), timestamp=timestamp, step=sample_step)


def poll(run_dir):
    from mrun.client.api import Api
    run_dir = Path(run_dir)
    mapping = json.loads((run_dir / "mlflow.json").read_text())
    ml = client(mapping["tracking_uri"])
    api = Api()
    status, raw, _ = api.request("GET", f"/api/jobs/{mapping['job_id']}/logs?offset={mapping['log_offset']}&wait_s=1")
    if status != 200:
        raise RuntimeError("could not read requested campaign logs")
    lines = (mapping["pending_line"] + raw.decode()).split("\n")
    mapping["pending_line"] = lines.pop()
    observed = 0
    for line in lines:
        if "ISA_EVENT:" in line:
            event = json.loads(line.split("ISA_EVENT:", 1)[1])
            log_event(ml, mapping, event)
            observed += 1
    mapping["log_offset"] += len(raw)
    receipt = api.json("GET", f"/api/jobs/{mapping['job_id']}")
    for key in ("peak_rss_mb", "peak_vram_mb", "elapsed_s"):
        value = (receipt.get("result") or {}).get(key)
        if value is not None:
            ml.log_metric(mapping["parent_run_id"], "mrun." + key, float(value))
    (run_dir / "mlflow.json").write_text(json.dumps(mapping, indent=2, sort_keys=True) + "\n")
    return {"job_id": mapping["job_id"], "state": receipt["state"], "events_logged": observed}


def finish(run_dir):
    run_dir = Path(run_dir)
    mapping = json.loads((run_dir / "mlflow.json").read_text())
    ml = client(mapping["tracking_uri"])
    receipt = json.loads((run_dir / "receipt.json").read_text())
    result = run_dir / "result"
    for key, run_id in mapping["runs"].items():
        path = result / key
        if not (path / "heldout.json").exists():
            ml.set_terminated(run_id, "FAILED")
            continue
        ml.log_artifacts(run_id, str(path), artifact_path="execution")
        summary = json.loads((path / "training.json").read_text())
        for name in ("physical_updates", "accepted_updates", "discovery_examples", "feedback_evaluations"):
            ml.log_metric(run_id, "train." + name, summary[name])
        for phase, values in summary["timing"].items():
            for name, value in values.items():
                ml.log_metric(run_id, "train." + phase + "." + name, value)
        for label in ("profile", "track", "oracle", "suite_sha256", "artifact_sha256", "contract_sha256"):
            report = json.loads((path / "heldout.json").read_text())
            ml.set_tag(run_id, label, str(report[label]))
        ml.set_terminated(run_id, "FINISHED")
    for name in ("request.json", "receipt.json", "mlflow.json", "result.tar.gz"):
        ml.log_artifact(mapping["parent_run_id"], str(run_dir / name), artifact_path="campaign")
    for path in sorted(result.glob("suite-*.json")):
        ml.log_artifact(mapping["parent_run_id"], str(path), artifact_path="suites")
    ml.set_terminated(mapping["parent_run_id"], "FINISHED" if receipt["state"] == "succeeded" else "FAILED")
    return mapping
