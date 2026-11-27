#!/usr/bin/env bash
# Experiment 3 (Table 1): baseline vs subgoals vs longer horizon vs temporal-distance cost vs oracle.
# Assumes runs/$TAG exists from data_scaling.sh. Usage: bash scripts/compare_fixes.sh large_1000000_s0 large
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
TAG=${1:-two_rooms_100000_s0}; FAMILY=${2:-two_rooms}
O="data.family=$FAMILY data.path=data/$TAG.npz run_dir=runs/$TAG"
python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=l2 plan.horizon=16
python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=subgoal plan.horizon=16
python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=l2 plan.horizon=64
python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=td plan.horizon=16
python -m mazeplan.evaluate --config configs/default.yaml $O eval.cost=oracle
