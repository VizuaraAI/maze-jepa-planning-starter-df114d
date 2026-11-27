#!/usr/bin/env bash
# Tiny end-to-end smoke run (CPU friendly, a few minutes): collect, train, evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
O="run_dir=runs/smoke data.path=data/smoke.npz data.n_transitions=5000 train.steps=200 train.num_workers=0 td.steps=100 eval.layout_seeds=[9001] eval.episodes_per_layout=3 eval.max_steps=60 eval.diag_pairs=300 plan.n_samples=100 plan.n_elites=10"
python -m mazeplan.collect --config configs/default.yaml $O
python -m mazeplan.train --config configs/default.yaml $O
python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=l2
