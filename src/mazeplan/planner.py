"""CEM model-predictive control in latent space with pluggable goal costs."""
from __future__ import annotations

from typing import Callable

import torch

CostFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]  # (z_traj [N,H,D], z_goal [D]) -> [N]


def l2_cost(z_traj: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
    """Baseline: Euclidean distance between the final predicted latent and the goal latent."""
    return (z_traj[:, -1] - z_goal).norm(dim=-1)


def make_td_cost(td_head: torch.nn.Module) -> CostFn:
    """Fix 3: learned temporal-distance cost on top of frozen latents."""
    def cost(z_traj: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return td_head(z_traj[:, -1], z_goal.expand_as(z_traj[:, -1]))
    return cost


@torch.no_grad()
def cem_plan(predictor: torch.nn.Module, z0: torch.Tensor, z_goal: torch.Tensor, cost_fn: CostFn,
             horizon: int = 16, n_samples: int = 300, n_elites: int = 30, n_iters: int = 3,
             action_dim: int = 2, init_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Return the optimized action sequence [H, A]. z0, z_goal: [D]."""
    dev = z0.device
    mean = torch.zeros(horizon, action_dim, device=dev) if init_mean is None else init_mean.clone()
    std = torch.ones(horizon, action_dim, device=dev)
    for _ in range(n_iters):
        acts = (mean + std * torch.randn(n_samples, horizon, action_dim, device=dev)).clamp(-1, 1)
        z_traj = predictor.rollout(z0.expand(n_samples, -1), acts)
        costs = cost_fn(z_traj, z_goal)
        elite = acts[costs.topk(n_elites, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0) + 1e-3
    return mean


@torch.no_grad()
def subgoal_sequence(bank: torch.Tensor, z_start: torch.Tensor, z_goal: torch.Tensor,
                     radius: float, max_nodes: int = 2000) -> list[torch.Tensor]:
    """Fix 1: shortest path over a k-NN graph of play latents (SoRB-style). Returns latent waypoints.

    ``bank`` holds latents of states seen in play [M, D]. Edges connect nodes whose
    latent L2 distance is below ``radius``; Dijkstra on edge length.
    """
    import heapq

    nodes = torch.cat([z_start[None], bank[:max_nodes], z_goal[None]], dim=0)
    dist = torch.cdist(nodes, nodes)
    n = len(nodes)
    adj = (dist < radius).cpu()
    d = dist.cpu()
    best = [float("inf")] * n
    prev = [-1] * n
    best[0] = 0.0
    heap = [(0.0, 0)]
    while heap:
        cur, u = heapq.heappop(heap)
        if u == n - 1:
            break
        if cur > best[u]:
            continue
        for v in torch.nonzero(adj[u]).flatten().tolist():
            nd = cur + float(d[u, v])
            if nd < best[v]:
                best[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, v))
    if prev[n - 1] < 0:
        return [z_goal]
    path, u = [], n - 1
    while u > 0:
        path.append(nodes[u])
        u = prev[u]
    return path[::-1]
