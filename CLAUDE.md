# CLAUDE.md

Instructions for the coding agent working on this repository with Gaurav.

## Project goal

Research project (target: a 4 to 8 page workshop paper, later possibly main track): explain why a LeWorldModel-style JEPA trained on reward-free random play fails to plan in mazes, and compare cheap fixes.

Core claim to test: the latent L2 goal cost correlates with straight-line (Euclidean) distance more than with BFS shortest-path length, and that mismatch predicts where the planner gets stuck. Fixes compared: subgoals (graph search over play latents), longer horizon, learned temporal-distance cost. Every fix is scored on success rate, planning time and extra data or training.

The paper draft lives outside this repo; its Table 1 rows map to `experiments/log_template.md`.

## Layout

- `src/mazeplan/maze.py`: layouts (`two_rooms`, `medium`, `large`), continuous point agent, rendering, BFS.
- `src/mazeplan/data.py`: random play collection (stores pos/act only), `PlayDataset` renders frames on the fly.
- `src/mazeplan/model.py`: `Encoder`, `Predictor`, `sigreg`, `WorldModel.loss`.
- `src/mazeplan/planner.py`: `cem_plan`, `l2_cost`, `make_td_cost`, `subgoal_sequence`.
- `src/mazeplan/tdcost.py`: `TDHead`, `train_td_head`.
- `src/mazeplan/{collect,train,evaluate}.py`: CLI entry points; all take `--config` plus `key=value` overrides.
- `configs/default.yaml`: single source of hyperparameters. `scripts/`: experiment drivers.

## Conventions

- Run modules with `PYTHONPATH=src python -m mazeplan.<entry>` or after `pip install -e .`.
- Every new hyperparameter goes into `configs/default.yaml` with a comment; no magic numbers in code.
- Never commit `data/` or `runs/`. Datasets are generated, not downloaded.
- Each run directory is self-describing: checkpoint stores the full config; eval writes JSON with summary and per-episode rows.
- Seeds: report mean and 95% bootstrap CI over at least 3 (ideally 5) seeds. Eval layouts (`eval.layout_seeds`) must stay disjoint from training layouts.
- Keep it small: Gaurav has about 8 hours a week. Prefer the simplest change that answers the research question. Ask before large refactors.
- Before any long sweep, run `bash scripts/quickstart.sh` to confirm nothing is broken.
- After each experiment, append a row to `experiments/log_template.md` (copy to `experiments/log.md`).

## Known simplifications (candidates for improvement)

- The encoder is a small CNN, not the official LeWorldModel ViT; port the official encoder/predictor for headline numbers.
- `tdcost.py` regresses embedding distance onto sqrt(time offset). Try a contrastive or quasimetric (QRL / TMD style) objective next.
- The subgoal graph uses latent L2 radius edges, so it inherits some of the L2 failure; consider TD-cost edges.
- The SIGReg statistic scales with batch size; tune `train.sigreg_weight` (0.01 to 1.0) and log both loss terms.

## Next steps (in order)

1. Run the quickstart, then train a Two-Rooms 100k model and check `l2` success is clearly above zero; tune `train.steps`, `sigreg_weight`.
2. Compare against PLDM's released code on Two-Rooms (https://github.com/vladisai/PLDM) to validate the setup.
3. Data-scaling sweep (`scripts/data_scaling.sh`) on all three families; produce Figure 2 (left) with a plotting script in `scripts/plot_scaling.py`.
4. Diagnosis: add a plotting script for latent distance vs BFS and vs Euclidean scatter, plus stuck-cell heatmaps from eval JSON.
5. Fix comparison (`scripts/compare_fixes.sh`) and Table 1, including a VICReg backbone variant for the PLDM row.
