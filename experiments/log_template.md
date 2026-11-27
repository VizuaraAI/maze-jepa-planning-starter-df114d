# Experiment log

Copy this file to `experiments/log.md` and add one row per finished run. Rows map directly to the paper.

## Table 1: fix comparison (success %, held-out layouts, 1M transitions)

| Run ID | Date | Backbone | Cost | Horizon | Two-Rooms | Medium | Large | Time (s/ep) | Extra data | Seeds | Commit | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T1-01 | | VICReg (PLDM) | Latent L2 | 16 | XX.X | XX.X | XX.X | XX.X | none | 5 | | |
| T1-02 | | LeWM (SIGReg) | Latent L2 | 16 | XX.X | XX.X | XX.X | XX.X | none | 5 | | |
| T1-03 | | LeWM (SIGReg) | + Subgoals | 16 | XX.X | XX.X | XX.X | XX.X | graph build | 5 | | |
| T1-04 | | LeWM (SIGReg) | + Longer horizon | 64 | XX.X | XX.X | XX.X | XX.X | none | 5 | | |
| T1-05 | | LeWM (SIGReg) | + Temporal-distance | 16 | XX.X | XX.X | XX.X | XX.X | head training | 5 | | |
| T1-06 | | Oracle | BFS distance | n/a | XX.X | XX.X | XX.X | XX.X | ground truth | 5 | | |

## Figure 2 (left): data scaling (success %, latent L2 cost)

| Family | 10k | 30k | 100k | 300k | 1M | 3M | Seeds |
|---|---|---|---|---|---|---|---|
| two_rooms | | | | | | | |
| medium | | | | | | | |
| large | | | | | | | |

## Section 5: diagnosis (Spearman rho, averaged over eval layouts)

| Run ID | Family | rho(L2, BFS) | rho(L2, Euclid) | rho(TD, BFS) | rho(TD, Euclid) | % failures stuck at separating wall |
|---|---|---|---|---|---|---|
| | | | | | | |

## Ablations

| Run ID | Change (e.g. sigreg_weight=1.0) | Family | Success % | Notes |
|---|---|---|---|---|
| | | | | |

## Free-form notes

- Date, what you tried, what surprised you, what to try next.
