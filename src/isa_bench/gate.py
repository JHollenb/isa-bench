"""Proof gate over an isa-bench suite (stage=gate).

The gate itself lives in the isa repo's proof-gate worktree (hybrid/m4gate.py, scripts/hybrid_launch.py, branch
feat/m4-proof-gate). This stage drives it end to end against a suite's isa-bench reference:

  1. gate view: <store>/../gate-view/<key>/<program>/{trace.npz, mem.bin -> reference, native.json in the gate's schema}
  2. m4-prove   Z3 proves each static instruction's learned record equal to SLEIGH (plus the hybrid microcode)
  3. stage-m4-cert   freeze the certificate on Beast (sha256 custody)
  4. m4-run     gated runs: learned machine where proved, hybrid microcode elsewhere, stop otherwise; contrast arms
  5. fetch both results and write outputs/gate/<label>/<suite>/{gate.json, summary.md}

The suite is passed as the launcher's --fresh-root, so its programs report under the split "heldout" and the M4
held-out twelve count as development programs (they are always proved and run alongside).
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from . import launch
from .adapters import REPO

VIEW = "/mnt/hdd8tb/uv-archive-stage/isa-bench/gate-view"

# Builds the gate's program layout from an isa-bench reference store, on Beast. trace.npz/mem.bin are symlinks (the
# gate hashes them through the link); native.json carries the two fields the gate compares against.
VIEW_SCRIPT = r"""
import json, os, sys
store, view = sys.argv[1], sys.argv[2]
ref = json.load(open(os.path.join(store, "reference.json")))["programs"]
os.makedirs(view, exist_ok=True)
done = []
for tag, p in sorted(ref.items()):
    if not p.get("complete"):
        continue
    src, dst = os.path.join(store, "programs", tag), os.path.join(view, tag)
    os.makedirs(dst, exist_ok=True)
    for f in ("trace.npz", "mem.bin"):
        link = os.path.join(dst, f)
        if not os.path.islink(link):
            os.symlink(os.path.join(src, f), link)
    stdout = open(os.path.join(src, "stdout.native"), "rb").read().decode("utf-8", "replace")  # as the gate decodes
    json.dump({"native_stdout": stdout, "native_exit": p["exit_code"], "source": src},
              open(os.path.join(dst, "native.json"), "w"), sort_keys=True)
    done.append(tag)
print(json.dumps(done))
"""


def gate_view(store: dict, host: str) -> tuple[str, list[str]]:
    view = f"{VIEW}/{store['key']}"
    out = subprocess.run(["ssh", host, "python3", "-", store["store_path"], view], input=VIEW_SCRIPT,
                         capture_output=True, text=True, check=True).stdout
    return view, json.loads(out.strip().splitlines()[-1])


def hybrid(gcfg: dict, *args: str) -> dict:
    """Run scripts/hybrid_launch.py in the proof-gate worktree; return its last JSON line."""
    root = (REPO / gcfg["hybrid_root"]).resolve()
    launch.tunnel_env()  # the dotenv points MRUN_URL at the LAN; the laptop's Python only reaches it through the tunnel
    env = {**os.environ, "PYTHONPATH": "."}
    cmd = [str((launch.WORKSPACE / "common/bin/common").resolve()), "exec", "python", "scripts/hybrid_launch.py", *args]
    print("+ (isa-hybrid) hybrid_launch.py " + " ".join(args), flush=True)
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=3600)
    if proc.returncode:
        raise SystemExit(f"hybrid_launch {args[0]} failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-3000:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    return json.loads(lines[-1]) if lines else {}


def wait_fetch(job_id: str, run_dir: Path, executor: dict) -> Path:
    meta = launch.wait(job_id, run_dir)
    if meta.get("state") != "succeeded":
        raise SystemExit(f"{job_id} ended {meta.get('state')}; log in {run_dir / 'job.log'}")
    return launch.fetch(job_id, run_dir / "result", executor=executor)


def summarize(prove: Path | None, runs: Path, split: str = "heldout") -> dict:
    summ = json.loads((runs / "m4gate_runs.json").read_text())["summary"]
    arms = {}
    for key, s in summ.items():
        arm, _, sp = key.partition("|")
        if sp != split:
            continue
        arms[arm] = {"programs": s["programs"], "complete": s["all_steps_exact"], "stdout_ok": s["stdout_ok"],
                     "exact_steps": s["exact_steps"], "trace_steps": s["trace_steps"], "paths": s.get("paths", {}),
                     "uops": s.get("uops")}
    per = {}
    for line in (runs / "runs.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r["program"].startswith(split + "/"):
            continue
        per.setdefault(r["program"].split("/", 1)[1], {})[r["arm"]] = {
            k: r.get(k) for k in ("status", "steps_run", "trace_steps", "exact_steps", "mismatch_steps", "paths",
                                  "all_steps_exact", "stdout_ok", "first_divergence")}
    out = {"split": split, "arms": arms, "programs": per}
    if prove:
        p = json.loads((prove / "m4_proof_summary.json").read_text())
        out["proofs"] = p["by_split"].get(split)
        out["proofs_total"] = p["total"]
        out["refuted"] = [r for r in p.get("refuted", []) if str(r.get("program", "")).startswith(split)][:50]
    return out


def markdown(g: dict, suite: str, label: str) -> str:
    lines = [f"# Proof gate: {label} on {suite}", "",
             f"Prove job {g.get('prove_job')}, run job {g.get('run_job')}, certificate sha256 {str(g.get('cert_sha256'))[:12]}.",
             ""]
    pr = g.get("proofs")
    if pr:
        st = pr.get("static", {})
        lines += [f"Static instructions: proved {st.get('proved', 0)}, refuted {st.get('refuted', 0)}, "
                  f"no learned record {st.get('no_record', 0)}, OS boundary {st.get('os_boundary', 0)}.", ""]
    lines += ["| arm | complete (all steps exact) | stdout ok | exact steps / native | learned steps | hybrid steps |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, a in sorted(g["arms"].items()):
        paths = a["paths"]
        lines.append(f"| {arm} | {a['complete']}/{a['programs']} | {a['stdout_ok']}/{a['programs']} | "
                     f"{a['exact_steps']}/{a['trace_steps']} | {paths.get('m4', 0)} | {paths.get('hybrid', 0)} |")
    lines += ["", "| program | " + " | ".join(sorted(g["arms"])) + " |", "|---|" + "---|" * len(g["arms"])]
    for prog, by in sorted(g["programs"].items()):
        lines.append(f"| {prog} | " + " | ".join((by.get(a) or {}).get("status", "—") for a in sorted(g["arms"])) + " |")
    return "\n".join(lines) + "\n"


def run_gate(cfg: dict) -> dict:
    gcfg, suite, executor = cfg["gate"], cfg["suite"], cfg["executor"]
    store_file = REPO / "outputs/reference" / suite["name"] / "store.json"
    if not store_file.exists():
        raise SystemExit(f"no reference for suite {suite['name']}: run stage=reference suite={suite['name']} first")
    store = json.loads(store_file.read_text())
    host = executor.get("host", "beast")
    view, programs = gate_view(store, host)
    if not programs:
        raise SystemExit("no complete programs in the reference")
    label = gcfg["label"]
    run_dir = REPO / "runs/gate" / f"{label}-{suite['name']}" / launch.stamp()
    run_dir.mkdir(parents=True)
    common = ["--machine", gcfg["machine"], "--de-ref", gcfg["de_ref"], "--fresh-root", view]
    record = {"suite": suite["name"], "label": label, "view": view, "programs": programs, **{k: gcfg[k] for k in
              ("machine", "de_ref", "workers", "run_workers", "timeout_ms", "arms", "mutations")}}
    cert_manifest = gcfg.get("cert_manifest")
    prove_res = None
    if not cert_manifest:
        sub = hybrid(gcfg, "m4-prove", "--run-dir", str(run_dir / "prove"), "--workers", str(gcfg["workers"]),
                     "--timeout-ms", str(gcfg["timeout_ms"]), "--ram", str(gcfg["prove_ram_mb"]),
                     "--est-wall", str(gcfg["prove_est_wall_s"]), *common)
        record["prove_job"] = sub["job_id"]
        prove_res = wait_fetch(sub["job_id"], run_dir / "prove", executor)
        cert_manifest = str(run_dir / "cert-manifest.json")
        staged = hybrid(gcfg, "stage-m4-cert", "--prove-job", sub["job_id"], "--cert-manifest", cert_manifest)
        record["cert_sha256"] = staged.get("sha256")
    else:
        cm = json.loads(Path(cert_manifest).read_text())
        record["cert_sha256"], record["prove_job"] = cm.get("sha256"), cm.get("prove_job")
        earlier = Path(cert_manifest).parent / "prove/result"  # the prove run that made this certificate, if local
        prove_res = earlier if (earlier / "m4_proof_summary.json").exists() else None
    record["cert_manifest"] = cert_manifest
    sub = hybrid(gcfg, "m4-run", "--run-dir", str(run_dir / "run"), "--workers", str(gcfg["run_workers"]),
                 "--cert-manifest", cert_manifest, "--mutations", str(gcfg["mutations"]), "--ram", str(gcfg["run_ram_mb"]),
                 "--est-wall", str(gcfg["run_est_wall_s"]), "--arms-m4", *gcfg["arms"], *common)
    record["run_job"] = sub["job_id"]
    runs = wait_fetch(sub["job_id"], run_dir / "run", executor)
    g = {**record, **summarize(prove_res, runs)}
    out = REPO / "outputs/gate" / label / suite["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "gate.json").write_text(json.dumps(g, indent=1, sort_keys=True, default=str) + "\n")
    (out / "summary.md").write_text(markdown(g, suite["name"], label))
    print((out / "summary.md").read_text())
    return g

