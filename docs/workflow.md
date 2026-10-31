# isa-bench workflows

Each workflow below is a series of commands you can copy and run, plus what to expect. They assume the repo root as the working directory and `just` installed. Every command that runs a model goes to Beast through mrun. The laptop only stages jobs, waits, fetches, reports and talks to MLflow.

Times marked *measured* come from the mrun job named next to them: time on Beast, excluding queueing and staging. Wall times from the 2026-10-30 test run of this doc are end to end on the laptop. A busy queue can double either.

| workflow | what you get | time |
|---|---|---|
| [0. Session setup](#0-session-setup) | tunnels, artifacts, status | < 1 min |
| [1. Update a model](#1-update-a-model-and-see-the-effect) | smoke → eval → diff → register | seconds (unicorn, c_apps) to ~20 min (m5d on dev + heldout) |
| [2. Train a model](#2-train-a-model) | a new direct-effect checkpoint, evaluated and registered | 25–75 min of GPU time |
| [3. Run the proof gate](#3-run-the-proof-gate) | M5d v3 proof-gated over any suite | 7–8 min (~3 min reusing a certificate) |
| [4. Custom C program](#4-run-models-against-your-own-c-program) | your program traced natively, run on any model, gated, debugged | ~3 min (+8 min with the gate) |
| [5. Full matrix](#5-run-the-full-matrix) | every model × every routine suite, report, board, registry | ~25 min |
| [6. Look at results](#6-look-at-results) | matrix, compute board, MLflow UI | seconds |
| [7. Debug a failure](#7-debug-a-failure) | first divergence, MDB recording | 1–2 min |
| [8. Publish](#8-publish) | git + DVC + registry in sync | 1–2 min |

Rules that apply everywhere (full list in `CLAUDE.md`):

- Comparison is always against the native trace.
- Unicorn must stay 100% trace-exact on every suite. If it isn't, the harness broke.
- `fresh1`/`fresh2` are the proof gate's held-out sets. Never put them in routine runs, and never tune anything on them.
- At most one heavy Python process on the laptop at a time.

---

## 0. Session setup

```sh
just tunnels        # ssh forwards: mrun 19025, MLflow 15000, MinIO 19010 (the laptop's Python can't reach 192.168.68.x)
just pull           # frozen artifacts via dvc pull from Beast MinIO (falls back to rsync from Beast origins)
just status         # references on Beast, every local eval, frozen artifacts
```

`just status` prints one line per reference suite (`reference c_apps 4 programs /mnt/hdd8tb/...`) and one per eval (`eval m5d dev success 7/10 coverage 0.7434 job job-...`).

If `just tunnels` says a port is already in use, the tunnels are already up; check with `nc -z 127.0.0.1 19025`.

---

## 1. Update a model and see the effect

A model is family code (`models/<family>/`) plus an instance config (`conf/model/<name>.yaml`). The harness never imports family code. Only `models/<family>/adapter.py` does, inside a fresh subprocess per (model, program) cell.

```sh
# 1. edit the family code or the instance config
$EDITOR models/m5d/adapter.py            # or models/<family>/src/..., conf/model/m5d.yaml

# 2. smoke: tiny_hello only, no warm repeats, one short lease (measured: 4 s for unicorn; m5d adds a ~4 s model load)
just smoke m5d

# 3. the real eval: one model, one suite (measured: unicorn c_apps 5 s; m5d on dev ~10-20 min, since programs run to 1M steps)
#    run eval after smoke before committing: smoke writes metrics for a single program
just eval m5d c_apps
just eval m5d dev

# 4. what moved: outputs/metrics/m5d.json against the last commit
../common/bin/common exec dvc metrics diff
just report                               # outputs/report/matrix.md
just board local                          # outputs/report/board.{md,html}: correctness + compute

# 5. keep it: a new MLflow run + registry version (alias latest-eval) carrying artifact, code and results
git commit -am "m5d: <what changed>"      # commit first so the registry records a clean git sha
just register m5d
```

What a cell measures: status, exact-prefix steps (instructions that matched native before the first divergence), the first divergence (step, RIP, 15-byte window, differing registers and flags), cold and warm wall time, µs/step, CPU s, peak RSS, slowdown against the native binary, and whether repeats agreed.

**Noise.** Re-running an eval always moves timing metrics a little (µs/step, CPU s) and rewrites tracked files (`outputs/metrics/<model>.json`, and `outputs/report/*` after `report`/`board`). `dvc metrics diff` shows the size of the change. `git checkout -- outputs/metrics outputs/report` discards a run you don't want to keep.

**Reading a cell.** A successful run reports exact prefix `N-1` of `N` native steps. The native trace stores `N` states (the initial state plus one per instruction), which is `N-1` instructions, so `87445 of 87446` is a complete, exact run. Step coverage divides by `N-1`.

**A changed artifact is a new model, not an overwrite.** Copy the config and change what differs:

```sh
cp conf/model/de_mamba.yaml conf/model/de_mamba_v2.yaml
$EDITOR conf/model/de_mamba_v2.yaml        # name, artifact, description (and artifact_sha256 where the config has one, e.g. m5d)
$EDITOR params.yaml                        # add de_mamba_v2: {family: direct_effect, artifact: ...} so `dvc repro` knows it
../common/bin/common exec dvc add artifacts/frozen/de_mamba_v2 && just push
just eval de_mamba_v2 c_apps
```

**Through DVC instead of `just eval`:** `just repro eval@m5d` reruns m5d on every routine suite, and only if its inputs changed. The inputs are `src/isa_bench`, `models/m5d`, `conf/model/m5d.yaml`, its artifact, the references, and the `run:` block of `conf/config.yaml`. Editing `src/isa_bench` reruns every model's eval.

---

## 2. Train a model

The frozen artifacts are already the latest version of every model, so nothing needs retraining before an eval. Train only to make a new variant.

**Direct-effect models (`de_nn`, `de_mamba`, `de_hydra`)** have a training recipe in their config's `train:` block, which ports R3's `direct_effect_v2`. It needs one GPU lease on Beast.

| arch | R3 training time, seed 673 (*measured*, `isa/outputs/direct-effect-r3-report/report.json`) |
|---|---|
| nn | 23 min |
| mamba | 48 min |
| hydra | 71 min |

Add a few minutes of data generation. The config's `est_wall_s: 7500` is the lease reservation, not an estimate.

```sh
# a new seed of the Mamba executor
just train de_mamba model.name=de_mamba_s674 model.train.seed=674
#  -> outputs/train/de_mamba_s674/artifact/{model.pt,result.json}; prints the eval command to run next

# evaluate the fresh checkpoint without making a config yet
just eval de_mamba c_apps model.name=de_mamba_s674 model.artifact=outputs/train/de_mamba_s674/artifact

# keep it: freeze + config + register
mkdir -p artifacts/frozen/de_mamba_s674 && cp -r outputs/train/de_mamba_s674/artifact/* artifacts/frozen/de_mamba_s674/
cp conf/model/de_mamba.yaml conf/model/de_mamba_s674.yaml   # set name, artifact, train.seed
../common/bin/common exec dvc add artifacts/frozen/de_mamba_s674 && just push
just eval de_mamba_s674 c_apps && just register de_mamba_s674
```

`result.json` carries the held-out panels (`iid`, `heldout_literal`, `heldout_pairs`), so you see instruction-level exactness before running a single program.

**M5d** has no wired training. Its learning lease is in the isa repo (`.worktrees/isa-spf-fieldlm`, `scripts/m4.py launch-devcegis`). That refit is not monotone across iterations: keep every `machine_iter{N}.pkl` and freeze the best, not the last. A new M5d machine enters isa-bench as a new config pointing at its pickle (`conf/model/m5e.yaml`, family `m5d`), then goes through workflow 1 and, before any claim, workflow 3.

**Selectors (`sel_*`, `pan_reference`)** are frozen June artifacts and have no training path here.

---

## 3. Run the proof gate

The gate makes the learned machine provable one instruction at a time.

1. **Prove.** For every static instruction a program executes, Z3 tries to prove that M5d's learned record computes exactly what Ghidra's SLEIGH x86-64 spec says (flags per the Intel SDM). The same is tried for the hybrid microcode, where a certified 8-bit neural ALU does all the arithmetic.
2. **Certify.** The proofs are frozen into a certificate keyed by (program, RIP, 15 instruction bytes, effect hash) and bound to the machine's sha256 and runtime commit.
3. **Run gated.** M5d executes an instruction only if its record is proved there. Otherwise the hybrid microcode executes it, if proved. Otherwise the run stops. It never guesses.

```sh
just gate c_apps                         # prove -> freeze certificate -> gated runs, over the c_apps reference
cat outputs/gate/m5d_v3/c_apps/summary.md
```

`stage=gate` drives the proof-gate code in the isa repo, whose worktree is `../.worktrees/isa-hybrid` (branch `feat/m4-proof-gate`). It sets that code up against the suite's isa-bench reference:

- It builds a gate view of the reference on Beast (`/mnt/hdd8tb/uv-archive-stage/isa-bench/gate-view/<key>/`).
- It passes that view as the launcher's `--fresh-root`, so the suite's programs report under the split `heldout`.
- The 22 dev + held-out M4 programs are always proved and run alongside as development data.

The two jobs and the fetched results land in `runs/gate/m5d_v3-<suite>/<stamp>/`. The summary goes to `outputs/gate/m5d_v3/<suite>/{gate.json,summary.md}`.

Arms, in `gate.arms` of `conf/config.yaml`:

| arm | executes | reading |
|---|---|---|
| `G0-m4+program` | M5d where proved, hybrid microcode elsewhere | the gated machine; the headline |
| `M4-proved-only` | M5d where proved, stops elsewhere | how far proved learned records alone get |
| `M4-unguarded` | M5d everywhere, no proof | M5d alone (what `just eval m5d` measures) |
| `H0-proved` | hybrid microcode only | the all-hybrid contrast |

Knobs (Hydra overrides after the suite):

```sh
just gate dev gate.workers=14 gate.mutations=300      # also run the 300-mutant certificate audit
just gate c_apps gate.cert_manifest=runs/gate/m5d_v3-c_apps/<stamp>/cert-manifest.json   # reuse a certificate, skip proving
just gate c_apps gate.machine=/mnt/hdd8tb/uv-archive-stage/<new>/machine_full.pkl gate.de_ref=<runtime commit> gate.label=m5e
```

Time (*measured*, 2026-10-30): about 7 min end to end on c_apps and 8 min on a 2-program custom suite (prove job-ddf0312e20a8, run job-2f281a5a51b4). Proving took 188 s on Beast (job-f43c311f9544) and the gated runs 140 s (job-5f9ac1a33a55), plus staging. Most of the proving time goes to the 22 M4 programs that always come along; a small custom suite adds little. Reusing a certificate (`gate.cert_manifest=...`) skips proving: about 3 min. The certificate must already cover the suite's static instructions; a program the prove step never saw has no proved path and simply stops.

Result on c_apps (same jobs): the gated machine completed 3 of 4 programs, step-exact and with correct stdout. M5d alone completed 1 of 4, and the hybrid alone 2 of 4. 215 static instructions were proved, 0 refuted; 17 have no learned record. `large_records-Os` stops at an `inc` that neither path has proved.

Memory: 10 run workers at 16 GB was killed_ram (job-b47d3ef0df05). The defaults are now 8 workers at 20 GB.

For comparison, the M5d v3 held-out gate (*measured*) took 451 s to prove and 577 s to run with the mutation audit (job-c0612ee9bdb4, job-a57af3da69a8).

Held-out protocol: a set the gate has never seen gets exactly one look. Run it once, report the arms separately, and from then on it is development data. `fresh2` has already had its look.

---

## 4. Run models against your own C program

Programs are freestanding x86-64 Linux C: no libc, with `_start`, `write` and `exit` done by inline `syscall`. Copy the runtime from `programs/dev/crt.h` (`_start` calls `cmain`; it provides `put`, `put_u64`, `memcpy` and `memset`) or write `_start` yourself as in `programs/c_apps/tiny_hello.c`. The harness supplies only `write(1, ...)` and `exit`. Other syscalls stop the run.

```sh
mkdir -p programs/mine
cp programs/dev/crt.h programs/mine/
cat > programs/mine/collatz.c <<'EOF'
#include "crt.h"
static u64 steps(u64 n) { u64 k = 0; while (n != 1) { n = (n & 1) ? 3 * n + 1 : n / 2; k++; } return k; }
int cmain(void) { u64 best = 0, arg = 0; for (u64 i = 1; i < 300; i++) { u64 s = steps(i); if (s > best) { best = s; arg = i; } } put_u64(arg); put_u64(best); return 0; }
EOF

cat > conf/suite/mine.yaml <<'EOF'
name: mine
description: "my programs"
dir: programs/mine
programs: [collatz]
variants:
  O0: ["-O0", "-static", "-nostdlib", "-ffreestanding", "-fno-builtin", "-fno-tree-loop-distribute-patterns", "-fno-stack-protector", "-fno-pie", "-no-pie", "-fcf-protection=none"]
  O2: ["-O2", "-static", "-nostdlib", "-ffreestanding", "-fno-builtin", "-fno-tree-loop-distribute-patterns", "-fno-stack-protector", "-fno-pie", "-no-pie", "-fcf-protection=none"]
max_trace_steps: 5000000
EOF

just reference mine                       # compile + native ptrace trace on Beast (measured: 7 s for collatz)
just matrix "[unicorn,m5d,de_mamba]" "[mine]"   # any set of models, one lease (measured: 2 min 18 s)
just report && just board local
just gate mine                            # is it provable? where does M5d need the hybrid? (measured: 8 min)
just debug m5d mine collatz-O0            # why m5d stopped: abstain at the `test` at 0x4011ca, record 10004
```

What the test run of this example measured (2026-10-30):

| | collatz-O0 (166,484 steps) | collatz-O2 (87,446 steps) |
|---|---|---|
| unicorn | success | success |
| m5d | abstain at step 44 (predicate, record 10004) | success, step-exact |
| de_mamba | wrong_output | wrong_output |
| gated (G0) | success: hybrid covers the `test` | success |

`debug` is only informative for cells that fail; a successful cell has no divergence to show.

Program tags are `<program>-<variant>` (`collatz-O0`, `collatz-O2`). If `just reference` reports `complete: false`, the program hit `max_trace_steps` or made a syscall other than write/exit.

`stage=reference` keys the suite by the sha of its sources, the tools, the worker and the suite config. Editing the program and rerunning re-traces it. Evals warn if a suite changed since its reference.

To keep the suite in the routine pipeline, add it to `suites` and `suite_names` in `params.yaml`, then `just repro`.

---

## 5. Run the full matrix

The routine suites are `c_apps` (4 programs), `dev` (10) and `heldout_m4` (12). With 12 models that's 312 cells. Saturn's scheduling contract allows at most 256 cells per lease, so `run_evals` splits the matrix into leases and keeps each model's cells together.

```sh
just matrix "[unicorn,m5d,de_nn,de_mamba,de_hydra,pan_reference,sel_pan_isa,sel_nn,sel_mamba,sel_hydra,sel_spf,sel_fieldlm]" "[c_apps,dev,heldout_m4]"
just report                                # outputs/report/matrix.md
just register-all                          # one registry version per model
just board                                 # board from the registry
```

Or through DVC, which also records `dvc.lock`:

```sh
just repro              # reference@<suite> -> eval@<model> -> report; only what changed
just push               # outputs and artifacts to Beast MinIO (s3://experiments/dvc/isa-bench)
```

Time (*measured*, 2026-10-30): the long lease of the last full matrix took 21.5 min: unicorn, m5d and the 3 DE models on all three suites (job-21d9cd97ed2e). The 7 selector models on all three suites took 4–5 min (job-355c0d592319). All 12 models on c_apps alone took 1.7 min (job-a00538e93aeb). Unicorn and M5d dominate, because they actually run the 0.1–1.4M-step programs. Models that stop at step 1 finish in milliseconds.

Quick checks that the pipeline works (measured: 3 min 20 s for the 4×4 matrix, job-5fd4aec809c8):

```sh
just matrix "[unicorn,m5d,de_mamba,sel_spf]" "[c_apps]"   # 4 models x 4 programs
just smoke unicorn
```

---

## 6. Look at results

| what | command / place |
|---|---|
| correctness matrix | `just report` → `outputs/report/matrix.md` (success/programs · step coverage · µs/step per model × suite) |
| correctness + compute board | `just board` (registry) or `just board local` (working tree) → `outputs/report/board.{md,html,json}`. Columns: success, trace-exact, step coverage, µs/step, × native, load s, wall s, CPU s, steps run, peak RSS, params, learned records, artifact MB, abstain/refused/wrong, registry version, git sha. Open the HTML for sortable columns and a suite filter. The registry board shows versions registered after 2026-10-30; earlier versions lack wall/CPU/steps totals. The local board has no registry version or git sha. |
| one model × suite | `outputs/eval/<model>/<suite>/summary.md` (per program: status, exact prefix, first divergence, wall, µs/step, × native) and `cells/<program>.json` (raw evidence) |
| gate | `outputs/gate/<label>/<suite>/summary.md` |
| MLflow UI | `open http://127.0.0.1:15000`. Experiment `isa-bench`: select runs → Compare for a metric table. Models → `isa-bench-<model>` → alias `latest-eval`. The version tags carry the headline numbers; the run's artifacts hold `model/` (artifact, adapter, config) and `results/<suite>/` (cells, summaries). |
| metric history | `../common/bin/common exec dvc metrics show --md outputs/metrics/m5d.json` (one model; the unfiltered table is very wide) / `dvc metrics diff HEAD~1` |

---

## 7. Debug a failure

```sh
just debug de_mamba c_apps medium_sort-Os     # report.md, bundle.json, info.json + MDB recording of the divergent forward
just mdb debug/de_mamba/c_apps/medium_sort-Os/root-recipe   # interactive MDB console
```

`report.md` gives the stop, the first divergence, model vs native registers and flags, and the disassembly around RIP. For direct-effect models it also records that single forward with MDB. `de_mamba` is the checkpoint with the recovered symbol table. Tours: [debugging-with-mdb.md](debugging-with-mdb.md).

For M5d and the selectors, `report.md` names the record, abstention reason or missing route at the stop. MDB recording applies to the neural models.

---

## 8. Publish

```sh
just test                                  # harness unit tests (laptop, seconds)
git add -A && git commit -m "..."          # code, configs, dvc.lock, outputs/metrics, outputs/report
just push                                  # DVC: artifacts + pipeline outputs to Beast MinIO
just register <model>                      # or just register-all: MLflow records the committed git sha
git push                                   # github.com/JHollenb/isa-bench (public: code only)
```

The GitHub repo is public. Weights, eval outputs and the MLflow registry stay on Beast, behind the LAN. Never commit `.env` or anything from `../common/.env`.
