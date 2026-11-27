#!/usr/bin/env bash
# Experiment 1 (Figure 2 left): success vs amount of random play, per maze family and seed.
# Usage: bash scripts/data_scaling.sh [family] [seeds...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
FAMILY=${1:-two_rooms}; shift || true
SEEDS=${@:-0 1 2 3 4}
for N in 10000 30000 100000 300000 1000000 3000000; do
  for S in $SEEDS; do
    TAG="${FAMILY}_${N}_s${S}"
    O="seed=$S data.family=$FAMILY data.n_transitions=$N data.path=data/$TAG.npz run_dir=runs/$TAG"
    [ -f "data/$TAG.npz" ] || python -m mazeplan.collect --config configs/default.yaml $O
    [ -f "runs/$TAG/world_model.pt" ] || python -m mazeplan.train --config configs/default.yaml $O
    python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=l2
  done
done
