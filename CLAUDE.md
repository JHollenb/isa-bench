# isa-bench agent guide

Pipeline: compiled C programs -> native ptrace reference on Beast -> each learned x86 executor run per program in a fresh process -> per-step comparison + compute metrics -> report -> MLflow registry. README.md has the full picture; this file is the rules.

## Rules
1. All model execution runs on Beast through mrun (`python -m isa_bench stage=...`). The laptop only stages, waits, fetches, reports, and records single MDB forwards. Never run eval cells or training locally.
2. Open tunnels first (`just tunnels`): the laptop's Python cannot reach 192.168.68.x. mrun 19025, MLflow 15000, MinIO 19010.
3. Do not edit files under `runs/<stage>/.../payload` while a submission is in flight; edit `src/` freely once the job id is printed.
4. A model's family code is only imported through its `models/<family>/adapter.py`, inside a cell subprocess. The harness (`src/isa_bench`) must not import family code.
5. Comparison is against the native trace, never against Unicorn. Unicorn is a control model and must stay 100% trace-exact on every suite; if it is not, the harness is wrong, not the models.
6. Architecturally undefined flags are masked (`state.FLAG_EFFECTS`); extend that table from the Intel SDM, never to make a model look better.
7. `fresh1`/`fresh2`/`fresh3` are the proof-gate held-out sets (all spent; the next one is sealed outside this repo until its one look). Do not add them to `params.yaml` suites or tune anything on them.
8. Frozen artifacts are content-addressed (DVC). A changed artifact is a new model config (`conf/model/<new>.yaml`), not an in-place overwrite.
9. Saturn's scheduling contract caps a lease at 256 cells; `run_evals` splits bigger matrices.
10. Report numbers from `outputs/eval/*/*/summary.json` with the mrun job id; label anything not measured as asserted.

## Adding a model family
`models/<family>/adapter.py` defines `Adapter(Base)` with `load()` and `run(program_dir, trace, max_steps) -> finish(...)`. Start from the native `trace.state(0)` and `mem.bin`; call `comparator.observe(i, state_after, window=..., before=...)` after every instruction; check `self.time_up(i)` each step; return stop kinds the `finish` classifier knows (exit, abstain, refused*, unsupported*, *fault, step_limit, wall_limit). Implement `debug_bundle` to expose the divergent step's inputs; optionally `debug_forward(bundle)` (one torch forward for MDB) and `debug_explain(bundle, output)` (family-specific report section), so `stage=debug` never imports family code. Then add `conf/model/<name>.yaml` and a `params.yaml` entry.

## Useful
- `just smoke <model>`: tiny_hello only, no warm repeats.
- `just debug <model> <suite> [program]`: `debug/<model>/<suite>/<program>/report.md` + MDB recipe.
- Beast job dirs: `/mnt/hdd8tb/mrun-work/<job>/`; reference store: `/mnt/hdd8tb/uv-archive-stage/isa-bench/reference/`.
