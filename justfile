# isa-bench recipes. Compute runs on Beast through mrun; the laptop stages, waits, fetches, reports.
set dotenv-load := true
set dotenv-path := "../common/.env"

py := "PYTHONPATH=src ../common/bin/common exec python -m isa_bench"
dvc := "../common/bin/common exec dvc"

# SSH tunnels: mrun 19025, MLflow 15000, MinIO cold 19010 (macOS blocks Python from the LAN)
tunnels:
    scripts/tunnels.sh

# frozen artifacts: dvc pull from Beast MinIO, or seed them from Beast origins if the remote lacks them
pull:
    {{dvc}} pull artifacts/frozen/*.dvc || scripts/pull_artifacts.sh

status:
    {{py}} stage=status

reference suite="c_apps":
    {{py}} stage=reference suite={{suite}}

# one model on one suite; extra Hydra overrides after the suite, e.g. `just eval de_mamba c_apps model.compare.gprs=4`
eval model suite="c_apps" *overrides:
    {{py}} stage=eval model={{model}} suite={{suite}} {{overrides}}

# several models x suites in one lease: just matrix "[unicorn,m5d]" "[c_apps,dev]"
matrix models suites="[c_apps]":
    {{py}} stage=matrix 'models={{models}}' 'suites={{suites}}'

# first program only, no warm repeats (new adapter code)
smoke model suite="c_apps":
    {{py}} stage=matrix 'models=[{{model}}]' 'suites=[{{suite}}]' run.warm_repeats=0 'only=[tiny_hello-Os]'

report:
    {{py}} stage=report

# correctness + compute table for every registered model (or `just board local`): outputs/report/board.{md,html}
board source="mlflow":
    {{py}} stage=board board.source={{source}}

# proof-gated M5d over a suite: prove (Z3 vs SLEIGH) -> freeze certificate -> gated runs; outputs/gate/<label>/<suite>
gate suite="c_apps" *overrides:
    {{py}} stage=gate suite={{suite}} {{overrides}}

# MLflow run + registry version carrying the model's artifact and every suite's results
register model:
    {{py}} stage=register model={{model}}

register-all:
    for m in $(ls conf/model | sed 's/.yaml$//'); do {{py}} stage=register model=$m || true; done

# first divergence/refusal report + MDB root recipe (direct-effect and neural selector models) for one cell;
# Hydra overrides after the program, e.g. `just debug de_mamba dev fib-O2 debug.capture_ports=true`
debug model suite="c_apps" program="" *overrides:
    {{py}} stage=debug model={{model}} suite={{suite}} {{ if program != "" { "debug.program=" + program } else { "" } }} {{overrides}}

# open the MDB discovery console on a debug recipe
mdb recipe:
    PYTHONPATH=../saturn-pub/packages/mdb/src ../common/bin/common exec python -m mdb discovery replay {{recipe}} --interactive --output {{recipe}}/../console-$(date +%Y%m%dT%H%M%S)

train model *overrides:
    {{py}} stage=train model={{model}} {{overrides}}

repro *targets:
    {{dvc}} repro {{targets}}

push:
    {{dvc}} push

test:
    PYTHONPATH=src ../common/bin/common exec python -m pytest -q tests
