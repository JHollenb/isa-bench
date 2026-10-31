"""Hydra entry point. One command per pipeline stage:

  python -m isa_bench stage=reference suite=c_apps                 compile + native trace on Beast
  python -m isa_bench stage=eval model=de_mamba suite=c_apps       one model on one suite (one mrun lease)
  python -m isa_bench stage=matrix models=[unicorn,m5d] suites=[c_apps,dev]   many evals, one lease
  python -m isa_bench stage=report                                  matrix table over every local eval
  python -m isa_bench stage=register model=de_mamba                 MLflow run + registry version with results
  python -m isa_bench stage=train model=de_mamba model.name=de_mamba_s674 model.train.seed=674
  python -m isa_bench stage=debug model=de_mamba suite=c_apps [debug.program=tiny_hello-Os]
  python -m isa_bench stage=status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from . import launch
from .adapters import REPO, tree_sha256

CONF = REPO / "conf"


def _plain(node) -> dict:
    return OmegaConf.to_container(node, resolve=True)


def compose(group: str, name: str) -> dict:
    return _plain(OmegaConf.load(CONF / group / f"{name}.yaml"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=1, sort_keys=True, default=str) + "\n")


# ------------------------------------------------------------------ reference
def suite_key(suite: dict) -> str:
    """Content key of a suite's reference: sources, tools, flags. Changing any program re-traces."""
    src = tree_sha256(REPO / suite["dir"])
    tools = tree_sha256(REPO / "programs/tools")
    worker = tree_sha256(REPO / "src/isa_bench/worker.py")
    blob = json.dumps({"src": src, "tools": tools, "worker": worker, "suite": suite}, sort_keys=True).encode()
    import hashlib

    return hashlib.sha256(blob).hexdigest()[:16]


def stage_reference(cfg: dict) -> dict:
    suite = cfg["suite"]
    key = f"{suite['name']}-{suite_key(suite)}"
    out = REPO / cfg["paths"]["reference"]
    job = {"stage": "reference", "suite": suite, "native_repeats": 7}
    receipt = launch.run_job("reference", cfg, job, name=suite["name"], payload_kwargs={"programs": True},
                             resources={"task": "isa-bench-reference", "ram_mb": 4000, "cpu_threads": 2,
                                        "est_wall_s": 900, "timeout_s": 7200},
                             note=f"isa-bench reference {suite['name']}", fetch_to=out)
    store = launch.publish_store(receipt["job_id"], "reference", key, executor=cfg["executor"])
    ref = json.loads((out / "reference.json").read_text())
    write(out / "store.json", {"store_path": store, "key": key, "job_id": receipt["job_id"],
                               "programs": sorted(t for t, p in ref["programs"].items() if p.get("complete"))})
    print(json.dumps({"reference": suite["name"], "store": store, "programs": {
        t: {"steps": p.get("steps"), "complete": p.get("complete")} for t, p in ref["programs"].items()}}))
    return receipt


# ------------------------------------------------------------------ eval
def eval_item(model: dict, suite: dict, cfg: dict) -> dict:
    store = REPO / "outputs/reference" / suite["name"] / "store.json"
    if not store.exists():
        raise SystemExit(f"no reference for suite {suite['name']}: run stage=reference suite={suite['name']} first")
    s = json.loads(store.read_text())
    if s["key"] != f"{suite['name']}-{suite_key(suite)}":
        print(f"warning: suite {suite['name']} sources changed since its reference ({s['key']}); re-run stage=reference",
              file=sys.stderr)
    only = cfg.get("only")
    programs = [p for p in s["programs"] if not only or p in only]
    return {"model": model, "suite": suite["name"], "reference_dir": s["store_path"], "programs": programs}


MAX_CELLS_PER_LEASE = 256  # saturn scheduling contract: MAX_LOCAL_WORK_ITEMS model processes per lease


def run_evals(cfg: dict, models: list[dict], suites: list[dict], *, name: str) -> list[dict]:
    """Evaluate models x suites; split into leases of at most MAX_CELLS_PER_LEASE cells (models kept whole)."""
    items = [eval_item(m, s, cfg) for m in models for s in suites]
    chunks, chunk, cells = [], [], 0
    for item in items:
        n = len(item["programs"])
        if chunk and cells + n > MAX_CELLS_PER_LEASE:
            chunks.append(chunk)
            chunk, cells = [], 0
        chunk.append(item)
        cells += n
    chunks.append(chunk)
    return [run_eval_lease(cfg, c, name=name if len(chunks) == 1 else f"{name}-part{k}")
            for k, c in enumerate(chunks)]


def run_eval_lease(cfg: dict, items: list[dict], *, name: str) -> dict:
    models = list({i["model"]["name"]: i["model"] for i in items}.values())
    families = tuple(dict.fromkeys(m["family"] for m in models))
    vendor = any(f in ("selector", "direct_effect") for f in families)
    n_cells = sum(len(i["programs"]) for i in items)
    ram = max(int(m["resources"]["ram_mb"]) for m in models) + 1000
    est = max(300, 45 * n_cells)
    job = {"stage": "eval", "run": cfg["run"], "evals": items}
    stage_root = REPO / "runs" / "results" / name
    receipt = launch.run_job("eval", cfg, job, name=name,
                             payload_kwargs={"families": families, "vendor": vendor,
                                             "artifacts": tuple(m.get("artifact") for m in models)},
                             resources={"task": "isa-bench-eval", "ram_mb": ram, "cpu_threads": 2, "est_wall_s": est,
                                        "timeout_s": int(cfg["run"]["cell_timeout_s"]) * n_cells + 600},
                             note=f"isa-bench eval {name}", fetch_to=stage_root, model_processes=n_cells)
    from .metrics import markdown, summarize

    for item in items:
        model, suite = item["model"]["name"], item["suite"]
        dest = REPO / "outputs/eval" / model / suite
        src = stage_root / model / suite
        if dest.exists():
            import shutil

            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dest)
        summary = summarize(model, suite, dest, REPO / "outputs/reference" / suite)
        summary["job_id"] = receipt["job_id"]
        write(dest / "summary.json", summary)
        (dest / "summary.md").write_text(markdown(summary))
        m = summary["metrics"]
        print(f"{model:14s} {suite:10s} success {m['success']}/{m['programs']}  step_coverage {m['step_coverage']:.4f}  "
              f"us/step {m['us_per_step_median']}  status {({k: v for k, v in m['status_counts'].items() if v})}")
    import shutil

    shutil.rmtree(stage_root, ignore_errors=True)
    for model in dict.fromkeys(i["model"]["name"] for i in items):
        write_model_metrics(model)
    return receipt


def write_model_metrics(model: str) -> None:
    """outputs/metrics/<model>.json: every evaluated suite's numeric metrics, suite-prefixed (DVC metrics)."""
    from .metrics import flat_metrics

    flat = {}
    for path in sorted((REPO / "outputs/eval" / model).glob("*/summary.json")):
        flat.update(flat_metrics(json.loads(path.read_text())))
    write(REPO / "outputs/metrics" / f"{model}.json", flat)


# ------------------------------------------------------------------ report
def resummarize() -> None:
    """Recompute every summary from its cell records (cells are the evidence; summaries are derived)."""
    from .metrics import markdown, summarize

    for dest in sorted(p.parent for p in (REPO / "outputs/eval").glob("*/*/cells")):
        old = dest / "summary.json"
        job_id = json.loads(old.read_text()).get("job_id") if old.exists() else None
        summary = summarize(dest.parent.name, dest.name, dest, REPO / "outputs/reference" / dest.name)
        summary["job_id"] = job_id
        write(old, summary)
        (dest / "summary.md").write_text(markdown(summary))
    for model in sorted(p.name for p in (REPO / "outputs/eval").iterdir() if p.is_dir()):
        write_model_metrics(model)


def stage_report(cfg: dict) -> dict:
    from .metrics import fmt

    resummarize()
    rows = []
    for path in sorted((REPO / "outputs/eval").glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        rows.append({"model": s["model"], "suite": s["suite"], **{k: v for k, v in s["metrics"].items()}})
    suites = sorted({r["suite"] for r in rows})
    models = sorted({r["model"] for r in rows})
    by = {(r["model"], r["suite"]): r for r in rows}
    lines = ["# isa-bench matrix", "",
             "Cell = success/programs · step coverage (exact-prefix steps over native steps) · median µs/step.", "",
             "| model | " + " | ".join(suites) + " |", "|---|" + "---|" * len(suites)]
    for m in models:
        cells = []
        for s in suites:
            r = by.get((m, s))
            cells.append("—" if r is None else f"{r['success']}/{r['programs']} · {r['step_coverage']:.1%} · "
                                               f"{fmt(r['us_per_step_median'])}")
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    out = REPO / cfg["paths"]["report"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "matrix.md").write_text("\n".join(lines) + "\n")
    write(out / "matrix.json", rows)
    print("\n".join(lines))
    return {"rows": len(rows)}


# ------------------------------------------------------------------ register
def stage_register(cfg: dict) -> dict:
    from .tracking import register

    model = cfg["model"]
    eval_dirs = sorted(p.parent for p in (REPO / "outputs/eval" / model["name"]).glob("*/summary.json"))
    if not eval_dirs:
        raise SystemExit(f"no evals for {model['name']}: run stage=eval first")
    summaries = [json.loads((d / "summary.json").read_text()) for d in eval_dirs]
    result = register(model, summaries, tracking=cfg["tracking"], eval_dirs=eval_dirs)
    print(json.dumps(result))
    return result


# ------------------------------------------------------------------ train
def stage_train(cfg: dict) -> dict:
    model = cfg["model"]
    spec = model.get("train")
    if not spec:
        raise SystemExit(f"model {model['name']} has no train recipe")
    out = REPO / cfg["paths"]["train"]
    job = {"stage": "train", "model": model}
    receipt = launch.run_job("train", cfg, job, name=model["name"],
                             payload_kwargs={"families": (model["family"],), "vendor": True},
                             resources={"task": "isa-bench-train", "ram_mb": int(spec.get("ram_mb", 24000)),
                                        "cpu_threads": int(spec.get("cpu_threads", 4)), "gpu": bool(spec.get("cuda")),
                                        "vram_mb": int(spec.get("vram_mb", 4000)),
                                        "est_wall_s": int(spec.get("est_wall_s", 7200))},
                             note=f"isa-bench train {model['name']}", fetch_to=out)
    print(json.dumps({"trained": model["name"], "artifact": str((out / "artifact").relative_to(REPO)),
                      "next": f"python -m isa_bench stage=eval model={cfg['model_conf']} model.name={model['name']} "
                              f"model.artifact={(out / 'artifact').relative_to(REPO)}"}))
    return receipt


# ------------------------------------------------------------------ status
def stage_status(cfg: dict) -> dict:
    refs = {p.parent.name: json.loads(p.read_text()) for p in (REPO / "outputs/reference").glob("*/store.json")}
    for suite, s in sorted(refs.items()):
        print(f"reference {suite:10s} {len(s['programs'])} programs  {s['store_path']}")
    for path in sorted((REPO / "outputs/eval").glob("*/*/summary.json")):
        s = json.loads(path.read_text())
        m = s["metrics"]
        print(f"eval {s['model']:14s} {s['suite']:10s} success {m['success']}/{m['programs']} "
              f"coverage {m['step_coverage']:.4f} job {s.get('job_id')}")
    frozen = sorted(p.name for p in (REPO / "artifacts/frozen").iterdir() if p.is_dir()) if (REPO / "artifacts/frozen").exists() else []
    print("frozen artifacts:", ", ".join(frozen) or "none (run `just pull`)")
    return {}


@hydra.main(version_base="1.3", config_path=str(CONF), config_name="config")
def main(raw: DictConfig) -> None:
    cfg = _plain(raw)
    cfg["model_conf"] = hydra.core.hydra_config.HydraConfig.get().runtime.choices.get("model")
    stage = cfg["stage"]
    if stage == "reference":
        stage_reference(cfg)
    elif stage == "eval":
        run_evals(cfg, [cfg["model"]], [cfg["suite"]], name=f"{cfg['model']['name']}-{cfg['suite']['name']}")
    elif stage == "matrix":
        models = [compose("model", n) for n in cfg.get("models") or []]
        suites = [compose("suite", n) for n in cfg.get("suites") or []]
        if not models or not suites:
            raise SystemExit("stage=matrix needs models=[...] suites=[...]")
        run_evals(cfg, models, suites, name=f"matrix-{len(models)}x{len(suites)}")
    elif stage == "report":
        stage_report(cfg)
    elif stage == "register":
        stage_register(cfg)
    elif stage == "train":
        stage_train(cfg)
    elif stage == "debug":
        from .debug import run_debug

        run_debug(cfg)
    elif stage == "status":
        stage_status(cfg)
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
