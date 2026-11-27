"""Entry point: planning success rate and the latent-distance diagnosis.

python -m mazeplan.evaluate --config configs/default.yaml eval.cost=l2
eval.cost in {l2, td, subgoal, oracle}; use plan.horizon=64 for the longer-horizon fix.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from .data import PlayDataset, to_tensor
from .maze import MazeEnv, bfs_distances, free_cells, make_layout
from .model import WorldModel
from .planner import cem_plan, l2_cost, make_td_cost, subgoal_sequence
from .tdcost import TDHead
from .utils import get_device, load_config, set_seed


def load_models(cfg: dict, dev: str):
    run_dir = Path(cfg["run_dir"])
    ckpt = torch.load(run_dir / "world_model.pt", map_location=dev)
    m = cfg["model"]
    wm = WorldModel(m["latent_dim"], 2, m["enc_width"], m["pred_hidden"]).to(dev)
    wm.load_state_dict(ckpt["model"])
    wm.eval()
    head = None
    if (run_dir / "td_head.pt").exists():
        head = TDHead(m["latent_dim"]).to(dev)
        head.load_state_dict(torch.load(run_dir / "td_head.pt", map_location=dev))
        head.eval()
    return wm, head


@torch.no_grad()
def diagnose(wm: WorldModel, env: MazeEnv, head: TDHead | None, n_pairs: int, rng, dev: str) -> dict:
    """Spearman correlation of latent distance with BFS path length and with straight-line distance."""
    cells = free_cells(env.walls)
    imgs = np.stack([env.render_cell(tuple(c)) for c in cells])
    z = wm.encoder(to_tensor(imgs, dev))
    i, j = rng.integers(len(cells), size=(2, n_pairs))
    d_lat = (z[i] - z[j]).norm(dim=-1).cpu().numpy()
    d_euc = np.linalg.norm(cells[i] - cells[j], axis=1)
    bfs = {}
    d_bfs = np.array([bfs.setdefault(tuple(cells[a]), bfs_distances(env.walls, tuple(cells[a])))[tuple(cells[b])]
                      for a, b in zip(i, j)])
    out = {"rho_l2_bfs": spearmanr(d_lat, d_bfs).statistic, "rho_l2_euc": spearmanr(d_lat, d_euc).statistic}
    if head is not None:
        d_td = head(z[i], z[j]).cpu().numpy()
        out |= {"rho_td_bfs": spearmanr(d_td, d_bfs).statistic, "rho_td_euc": spearmanr(d_td, d_euc).statistic}
    return {k: float(v) for k, v in out.items()}


@torch.no_grad()
def run_episode(wm, head, env, start, goal, cfg, dev, oracle_dist=None, bank=None) -> dict:
    p, e = cfg["plan"], cfg["eval"]
    z_goal = wm.encoder(to_tensor(env.render_cell(goal), dev))[0]
    obs = env.reset(start)
    cost_name = e["cost"]
    cost_fn = make_td_cost(head) if cost_name == "td" else l2_cost
    waypoints = None
    if cost_name == "subgoal":
        z0 = wm.encoder(to_tensor(obs, dev))[0]
        waypoints = subgoal_sequence(bank, z0, z_goal, radius=e["subgoal_radius"])
    t0, mean = time.time(), None
    for t in range(e["max_steps"]):
        if max(abs(env.pos[0] - goal[0] - 0.5), abs(env.pos[1] - goal[1] - 0.5)) <= e["success_radius"]:
            return {"success": 1, "steps": t, "time": time.time() - t0, "final": env.cell}
        z = wm.encoder(to_tensor(obs, dev))[0]
        target = z_goal
        if waypoints:
            while len(waypoints) > 1 and (z - waypoints[0]).norm() < e["subgoal_reach"]:
                waypoints.pop(0)
            target = waypoints[0]
        if cost_name == "oracle":
            # ground-truth upper bound: greedy step toward the BFS-closer neighbour cell
            r, c = env.cell
            nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)) if oracle_dist[r + dr, c + dc] >= 0]
            nr, nc = min(nbrs, key=lambda q: oracle_dist[q])
            action = np.array([nr + 0.5 - env.pos[0], nc + 0.5 - env.pos[1]]) / env.max_step
        else:
            plan = cem_plan(wm.predictor, z, target, cost_fn, horizon=p["horizon"], n_samples=p["n_samples"],
                            n_elites=p["n_elites"], n_iters=p["n_iters"], init_mean=mean)
            action = plan[0].cpu().numpy()
            mean = torch.cat([plan[1:], torch.zeros_like(plan[:1])])  # warm start
        obs = env.step(action)
    return {"success": 0, "steps": e["max_steps"], "time": time.time() - t0, "final": env.cell}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg["seed"])
    dev = get_device(cfg["device"])
    e = cfg["eval"]
    rng = np.random.default_rng(cfg["seed"] + 123)
    wm, head = load_models(cfg, dev)
    if e["cost"] == "td" and head is None:
        raise SystemExit("eval.cost=td needs td_head.pt; train with td.enabled=true")

    bank = None
    if e["cost"] == "subgoal":
        ds = PlayDataset(cfg["data"]["path"], seq_len=1, img_size=cfg["data"]["img_size"])
        idx = rng.choice(len(ds.pos), size=min(len(ds.pos), 400), replace=False)
        imgs = np.stack([ds.envs[ds.layout_id[i]].render(ds.pos[i, t]) for i in idx for t in (0, 50, 100)])
        bank = wm.encoder(to_tensor(imgs, dev))

    results, diag = [], []
    for layout_seed in e["layout_seeds"]:
        env = MazeEnv(make_layout(cfg["data"]["family"], int(layout_seed)), img_size=cfg["data"]["img_size"],
                      max_step=cfg["data"]["max_step"])
        diag.append(diagnose(wm, env, head, e["diag_pairs"], rng, dev))
        cells = free_cells(env.walls)
        for _ in range(e["episodes_per_layout"]):
            while True:
                s, g = (tuple(int(x) for x in cells[k]) for k in rng.integers(len(cells), size=2))
                dist = bfs_distances(env.walls, g)
                if dist[s] >= e["min_bfs_distance"]:
                    break
            r = run_episode(wm, head, env, s, g, cfg, dev, oracle_dist=dist, bank=bank)
            r |= {"layout": int(layout_seed), "start": s, "goal": g, "bfs": int(dist[s]),
                  "euc": float(np.hypot(s[0] - g[0], s[1] - g[1]))}
            results.append(r)

    summary = {
        "cost": e["cost"], "horizon": cfg["plan"]["horizon"], "family": cfg["data"]["family"],
        "success_rate": 100 * float(np.mean([r["success"] for r in results])),
        "time_per_episode": float(np.mean([r["time"] for r in results])),
        "diagnosis": {k: float(np.mean([d[k] for d in diag])) for k in diag[0]},
    }
    out = Path(cfg["run_dir"]) / f"eval_{e['cost']}_H{cfg['plan']['horizon']}.json"
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
