# maze-jepa-planning-starter

Starter code for **"Why Latent World Models Walk Into Walls: Diagnosing and Fixing the Goal-Distance Cost for Planning in Mazes from Reward-Free Play"** (Gaurav Salkar, Vizuara).

The question: a JEPA world model trained only on random play can plan in open rooms, but in mazes it presses against walls. Is the cause the latent goal-distance cost (which tracks straight-line distance instead of path length), and what is the cheapest fix?

This repo gives you a small, complete pipeline you can read in one sitting:

1. **Grid mazes** (`two_rooms`, `medium`, `large`) with a continuous point agent, 64x64 pixel rendering and exact BFS distances.
2. **Random-play collection** (positions and actions only; frames rendered on the fly, so no big files).
3. **A LeWorldModel-style JEPA**: CNN encoder, residual action-conditioned predictor, multi-step latent prediction loss and SIGReg.
4. **CEM model-predictive control** in latent space with four costs: `l2` (baseline), `subgoal`, `td` (learned temporal distance) and `oracle` (BFS upper bound). The longer-horizon fix is `plan.horizon=64`.
5. **The diagnosis**: Spearman correlation of latent distance with BFS path length and with straight-line distance.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
```

Works on CPU for smoke tests; use a GPU (CUDA) or Apple MPS for real runs (`device: auto`).

## How to run

```bash
# 1. Smoke test: collect 5k transitions, train 200 steps, evaluate (a few minutes on CPU)
bash scripts/quickstart.sh

# 2. Experiment 1, data scaling (Figure 2 left): 10k to 3M transitions x 5 seeds
bash scripts/data_scaling.sh two_rooms 0 1 2 3 4
bash scripts/data_scaling.sh medium 0 1 2
bash scripts/data_scaling.sh large 0 1 2

# 3. Experiment 3, compare fixes (Table 1) on a trained run
bash scripts/compare_fixes.sh large_1000000_s0 large
```

Single commands, with any config key overridden on the command line:

```bash
export PYTHONPATH=src
python -m mazeplan.collect  --config configs/default.yaml data.family=large data.n_transitions=300000 data.path=data/large_300k.npz
python -m mazeplan.train    --config configs/default.yaml data.family=large data.path=data/large_300k.npz run_dir=runs/large_300k
python -m mazeplan.evaluate --config configs/default.yaml data.family=large data.path=data/large_300k.npz run_dir=runs/large_300k eval.cost=td
```

Each evaluation writes `runs/<run>/eval_<cost>_H<horizon>.json` with the success rate, planning time, the diagnosis correlations and per-episode records (start, goal, BFS distance, final cell). Use the final cells to draw the "where the planner gets stuck" heatmaps.

## Layout

```
configs/default.yaml      all hyperparameters (data, model, train, td, plan, eval)
src/mazeplan/maze.py      layouts, dynamics, rendering, BFS
src/mazeplan/data.py      random-play collection, PlayDataset
src/mazeplan/model.py     Encoder, Predictor, SIGReg, WorldModel
src/mazeplan/planner.py   CEM planner, L2 / TD costs, subgoal graph search
src/mazeplan/tdcost.py    temporal-distance head (Fix 3)
src/mazeplan/train.py     training entry point
src/mazeplan/evaluate.py  planning evaluation + diagnosis entry point
scripts/                  quickstart, data scaling, fix comparison
experiments/log_template.md  experiment log mapped to the paper's Table 1
```

## Upstream codebases

This scaffold is a deliberately small re-implementation, so you understand every line. For the paper's headline numbers, port or swap in the official components:

- **LeWorldModel** (Maes et al., 2026): paper https://arxiv.org/abs/2603.19312, project page https://le-wm.github.io (follow the official code link there).
- **PLDM** (Sobal et al., NeurIPS 2025): https://github.com/vladisai/PLDM, for the Two-Rooms environment, the VICReg baseline and their random-data protocol.
- **LeJEPA / SIGReg** reference implementation: https://github.com/rbalestr-lab/lejepa

A good sanity check before porting: reproduce PLDM's Two-Rooms random-data success with both this code and theirs.
