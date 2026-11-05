# isa-bench

Run compiled C programs on our learned x86-64 executors, check them against native hardware step by step, measure correctness and compute, and register each model version with its results.

```
programs/<suite>/*.c ──gcc──> prog.elf ──ptrace──> native trace (RIP, 16 GPRs, EFLAGS per step) + memory image
                                                        │
models/<family>/adapter.py + artifacts/frozen/<model> ──┴──> per-step comparison ──> outputs/eval/<model>/<suite>/
                                                                                          │
                                                       outputs/report/matrix.md  <────────┤
                                                       MLflow run + registry version <────┘
```

Everything that runs a model happens on Beast through mrun. The laptop stages payloads, waits for the job, fetches results, writes reports and talks to MLflow.

## Quick start

```sh
just tunnels                   # ssh forwards: mrun 19025, MLflow 15000, MinIO 19010
just pull                      # frozen artifacts (dvc pull; first time: rsync from Beast origins)
just reference c_apps          # compile + native-trace a suite on Beast
just eval m5d c_apps           # one model on one suite (one lease)
just matrix "[unicorn,m5d]" "[c_apps,dev]"   # many models x suites in one lease
just report                    # outputs/report/matrix.md
just register m5d              # MLflow run + new registry version carrying the results
just debug de_mamba c_apps     # first divergence + MDB replay recipe
just board                     # correctness + compute table from the MLflow registry (outputs/report/board.html)
just gate c_apps               # proof-gated M5d over a suite (Z3 vs SLEIGH certificate, then gated runs)
dvc repro                      # the whole pipeline; reruns only what changed
```

Step-by-step workflows (update, train, gate, custom C program, full matrix, publish): [docs/workflow.md](docs/workflow.md). MDB tour: [docs/debugging-with-mdb.md](docs/debugging-with-mdb.md).

All recipes are thin wrappers over `python -m isa_bench stage=<stage> ...` (Hydra), so any config value can be overridden on the command line.

## Layout

| path | what |
|---|---|
| `conf/` | Hydra configs. `model/<name>.yaml` is one model instance (family, artifact, comparison profile, resources, training recipe, debug hooks). `suite/<name>.yaml` is a program set plus compiler variants. `executor/`, `tracking/` select where things run and where results go. |
| `models/<family>/` | Code for one model family: `adapter.py` (load + run a program under the harness contract), optional `train.py`, the family's sources under `src/`. Families: `unicorn` (emulator control), `selector` (June PAN-ISA runtime + the six learned selectors), `direct_effect` (R3 NN/Mamba/Hydra executors), `m5d` (M5d-final v3 structured machine). |
| `models/_vendor/isa_r3/` | Frozen shared sources from the compiled-C PoC lease job-007fe5d65603 (`PROVENANCE.json` has every file's sha256). |
| `artifacts/frozen/<model>/` | Model artifacts, DVC-tracked (remote `beast-cold` = `s3://experiments/dvc/isa-bench` on Beast MinIO :9010). |
| `programs/` | C sources: `c_apps` (PoC apps), `dev` and `heldout_m4` (M4/M5 sets), `fresh1`/`fresh2` (proof-gate held-out sets, run deliberately), `tools/` (tracer + native timer). |
| `src/isa_bench/` | Harness: `worker.py` (in-lease entry), `cell.py` (one model x program in a fresh process), `state.py` (traces, memory, comparator), `adapters.py` (contract), `metrics.py`, `launch.py` (mrun), `tracking.py` (MLflow), `board.py` (compute board), `gate.py` (proof gate driver), `debug.py` (MDB), `cli.py`. |
| `outputs/` | `reference/<suite>/` (traces; `store.json` points at the Beast copy eval jobs read), `eval/<model>/<suite>/` (cells + `summary.{json,md}`), `metrics/<model>.json` (DVC metrics), `report/`. |
| `runs/` | Local receipts per submission (request, mrun job id, log, source manifest). |
| `legacy/c-app-matrix-poc/` | The throwaway PoC this repo replaces, kept for provenance. |

## What is measured

Every model starts from the native process state at `_start` (registers, flags, the full initial memory image) and runs until exit, a stop, or a step budget of 1.25x the native step count. After each instruction its architectural state is compared with the native trace.

- **status**: `success` (exact stdout and exit code), `wrong_output`, `abstain` (the model declined), `refused` (unsupported instruction or route), `fault` (bad memory access), `step_limit`, `wall_limit`, `error`.
- **exact prefix steps**: how many instructions matched native before the first divergence. Step coverage = exact prefix / native steps, summed over a suite.
- **first divergence**: step, RIP, the 15-byte window, and which registers or flags differed. A divergence is still recorded when the endpoint is right.
- **Undefined flags** (Intel SDM: e.g. SF/ZF/AF/PF after `imul`, AF after logic ops) are not compared until an instruction defines them again. Mismatches on them are counted in `masked_undefined_flag_diffs`. Without this, Unicorn itself "diverges" at the first `imul`.
- **Compute**: load wall time, cold and warm wall time per program, µs per instruction, CPU time, peak RSS, slowdown against the native binary, parameter count, artifact bytes and sha256, determinism across repeats.

What each family represents differs; the comparison profile in its model config says which registers and flags count. The direct-effect executors have 8 GPRs and CF/ZF/SF/OF and no memory-read input, so programs that load from memory beyond the fetch window cannot succeed by construction. The harness supplies fetch, the memory image, store commit and Linux `write`/`exit` to every model.

## Changing a model and seeing the result

1. Edit the family code (`models/<family>/...`) or the instance config (`conf/model/<name>.yaml`).
2. `just smoke <model>` runs it on `tiny_hello` only (one short lease) to catch breakage.
3. `dvc repro eval@<model>` (or `just eval <model> <suite>`) re-runs that model only; `dvc metrics diff` shows the change against the last commit.
4. `just register <model>` records the version in MLflow with its results.

A new variant is a new config: copy `conf/model/de_mamba.yaml` to `conf/model/de_mamba_s674.yaml`, change what differs, add it to `params.yaml`. To train one: `just train de_mamba model.name=de_mamba_s674 model.train.seed=674` (GPU lease), then evaluate with `model.artifact=outputs/train/de_mamba_s674/artifact`.

## Debugging with MDB and the symbol table

`just debug <model> <suite> [program]` writes `debug/<model>/<suite>/<program>/`:

- `report.md` (stop, first divergence, model vs native state, the disassembly around RIP);
- `bundle.json` (the exact inputs of the divergent step);
- for direct-effect models and the four neural selectors, `root-recipe/`, an MDB discovery recording of that single forward (selectors: the runtime head at the refusing instruction, with a top-5 vs typed-family table in the report; FieldLM/SPF get the route lookup instead). Add `debug.capture_ports=true` for a recipe `mdb ports` can read.

Open the recipe with `just mdb debug/<...>/root-recipe` (or `python -m mdb discovery replay <recipe> --interactive --output <dir>`). The `de_mamba` checkpoint (sha256 `9399e921…`) is the R3 Mamba whose symbol table was recovered in `../isa/outputs/moonshot-r3-mamba/` (`symbol-table.md`, `catalog.json`). The debug report links it whenever the checkpoint hash matches.

## Services

| service | where | used for |
|---|---|---|
| mrun | Beast :9025 (tunnel 19025) | every reference, eval and train job |
| MLflow | Beast :5000 (tunnel 15000), SQLite + MinIO bucket `mlflow` | runs, metrics, model registry (`isa-bench-<model>`, alias `latest-eval`) |
| MinIO cold | Beast :9010 (tunnel 19010) | DVC remote `beast-cold`, MLflow artifacts |
| Beast store | `/mnt/hdd8tb/uv-archive-stage/isa-bench/reference/<suite>-<key>` | native traces that eval jobs read in place |
| Trackio | Zima :7860 | optional (`tracking.trackio`); Zima's docker daemon was down on 2026-10-30 |

Credentials come from `../common/.env` (loaded by `just` and by `tracking.py`); they are never printed.
