"""MLflow tracking + model registry (Beast). Results travel with the model version.

For each model the register stage creates one MLflow run in experiment ``isa-bench``:
  params   resolved model config, git revision, artifact sha256, suites
  metrics  <suite>.<metric> for every evaluated suite
  artifacts
    model/       the artifact files + model config + the family's code (adapter, train)
    results/     per-suite summary.json, summary.md and every cell record
then registers ``<prefix>-<model>`` with a new version whose source is ``runs:/<id>/model``,
tagged with headline metrics, and moves the ``latest`` alias to it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .adapters import REPO


def _env(tracking: dict) -> None:
    if tracking.get("s3_endpoint"):
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = tracking["s3_endpoint"]
    common_env = REPO.parent / "common/.env"
    if common_env.exists():  # credentials for the artifact store; never printed
        for line in common_env.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"):
                os.environ.setdefault(key, value.strip().strip('"'))


def git_revision() -> dict:
    def g(*args):
        return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()

    return {"git_sha": g("rev-parse", "HEAD") or "uncommitted", "git_dirty": bool(g("status", "--porcelain"))}


def _flatten(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = json.dumps(value) if isinstance(value, (list, tuple)) else value


def register(model_cfg: dict, summaries: list[dict], *, tracking: dict, eval_dirs: list[Path]) -> dict:
    if not tracking.get("enabled"):
        return {"skipped": "tracking disabled"}
    _env(tracking)
    import mlflow
    from mlflow.tracking import MlflowClient

    from .metrics import flat_metrics

    mlflow.set_tracking_uri(tracking["tracking_uri"])
    mlflow.set_experiment(tracking.get("experiment", "isa-bench"))
    client = MlflowClient()
    name = f"{tracking.get('registry_prefix', 'isa-bench')}-{model_cfg['name']}"
    rev = git_revision()
    params = {}
    _flatten("model", model_cfg, params)
    params.update({"suites": ",".join(s["suite"] for s in summaries), **rev})
    if summaries:
        params["artifact_sha256"] = summaries[0]["metrics"].get("artifact_sha256")
    metrics = {}
    for s in summaries:
        metrics.update(flat_metrics(s))
    with mlflow.start_run(run_name=f"{model_cfg['name']}") as run:
        mlflow.set_tags({"isa_bench.model": model_cfg["name"], "isa_bench.family": model_cfg["family"], **rev})
        mlflow.log_params({k: str(v)[:500] for k, v in params.items()})
        mlflow.log_metrics(metrics)
        with tempfile.TemporaryDirectory(prefix="isa-bench-reg-") as tmp:
            stage = Path(tmp) / "model"
            stage.mkdir()
            if model_cfg.get("artifact"):
                src = REPO / model_cfg["artifact"]
                (shutil.copytree if src.is_dir() else shutil.copy2)(src, stage / "artifact")
            (stage / "model.json").write_text(json.dumps(model_cfg, indent=1, sort_keys=True))
            code = REPO / model_cfg["code"]
            for f in ("adapter.py", "train.py", "README.md"):
                if (code / f).exists():
                    shutil.copy2(code / f, stage / f)
            mlflow.log_artifacts(str(stage), artifact_path="model")
            for s, d in zip(summaries, eval_dirs):
                res = Path(tmp) / "results" / s["suite"]
                res.mkdir(parents=True)
                (res / "summary.json").write_text(json.dumps(s, indent=1, sort_keys=True, default=str))
                for f in ("summary.md", "eval.json"):
                    if (Path(d) / f).exists():
                        shutil.copy2(Path(d) / f, res / f)
                if (Path(d) / "cells").exists():
                    shutil.copytree(Path(d) / "cells", res / "cells")
            mlflow.log_artifacts(str(Path(tmp) / "results"), artifact_path="results")
        run_id = run.info.run_id
    try:
        client.get_registered_model(name)
    except Exception:
        client.create_registered_model(name, description=model_cfg.get("description", ""),
                                       tags={"isa_bench.family": model_cfg["family"]})
    headline = {}
    for s in summaries:
        m = s["metrics"]
        headline[f"{s['suite']}.success"] = f"{m['success']}/{m['programs']}"
        headline[f"{s['suite']}.step_coverage"] = f"{m['step_coverage']:.4f}"
        headline[f"{s['suite']}.us_per_step_median"] = str(m.get("us_per_step_median"))
    version = client.create_model_version(name, source=f"runs:/{run_id}/model", run_id=run_id,
                                          tags={**headline, **{k: str(v) for k, v in rev.items()}},
                                          description=f"isa-bench eval of {model_cfg['name']} on "
                                                      + ", ".join(s["suite"] for s in summaries))
    try:
        client.set_registered_model_alias(name, "latest-eval", version.version)
    except Exception:
        pass
    return {"run_id": run_id, "registered_model": name, "version": version.version,
            "tracking_uri": tracking["tracking_uri"]}
