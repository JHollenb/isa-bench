#!/usr/bin/env bash
# Seed artifacts/frozen/<model>/ from their Beast origins (one-time; afterwards `dvc pull`).
# Origins: compiled-C PoC lease job-007fe5d65603 (selectors + R3 direct-effect checkpoints) and the
# M5d-final v3 archive (machine_iter11 of recovery lease job-ab5cc831e7d8).
set -euo pipefail
cd "$(dirname "$0")/.."
POC=/mnt/hdd8tb/mrun-work/job-007fe5d65603/artifacts
M5D=/mnt/hdd8tb/uv-archive-stage/isa-m5d-v3
pull() { mkdir -p "$(dirname "$2")"; rsync -a "beast:$1" "$2"; }
for sel in nn pan_isa fieldlm mamba hydra; do
  mkdir -p artifacts/frozen/sel_$sel && pull "$POC/$sel.json" artifacts/frozen/sel_$sel/model.json
done
mkdir -p artifacts/frozen/sel_spf && pull "$POC/spf_transformer.json" artifacts/frozen/sel_spf/model.json
for arch in nn mamba hydra; do
  mkdir -p artifacts/frozen/de_$arch && pull "$POC/effect_$arch/" artifacts/frozen/de_$arch/
done
mkdir -p artifacts/frozen/m5d
pull "$M5D/machine_full.pkl" artifacts/frozen/m5d/machine_full.pkl
pull "$M5D/PROVENANCE.txt" artifacts/frozen/m5d/PROVENANCE.txt
shasum -a 256 artifacts/frozen/*/* | sed "s#  # #"
