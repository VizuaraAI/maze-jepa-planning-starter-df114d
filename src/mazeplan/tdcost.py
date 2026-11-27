"""Fix 3: a learned temporal-distance cost on frozen world-model latents.

Starting point: regress the embedding distance ||e(z) - e(z')|| onto the
square root of the time offset between two states of the same random-play
trajectory (random-walk diffusion makes sqrt(k) roughly proportional to the
distance travelled along corridors). Contrastive or quasimetric variants
(Myers et al. 2024; Wang et al. 2023) are natural next steps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import PlayDataset


class TDHead(nn.Module):
    def __init__(self, latent_dim: int = 64, hidden: int = 256, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(latent_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, out_dim))

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        return (self.net(z) - self.net(z_goal)).norm(dim=-1)


def sample_pairs(ds: PlayDataset, batch: int, max_offset: int, rng: np.random.Generator):
    """Sample (frame_i, frame_j, offset) pairs from the same trajectory."""
    n_traj, t_len = ds.pos.shape[0], ds.pos.shape[1]
    ti = rng.integers(n_traj, size=batch)
    k = rng.integers(0, max_offset + 1, size=batch)
    t0 = rng.integers(0, np.maximum(1, t_len - k))
    f0, f1 = [], []
    for i, a, kk in zip(ti, t0, k):
        env = ds.envs[ds.layout_id[i]]
        f0.append(env.render(ds.pos[i, a]))
        f1.append(env.render(ds.pos[i, min(a + kk, t_len - 1)]))
    to_t = lambda xs: torch.from_numpy(np.stack(xs)).permute(0, 3, 1, 2).float() / 255.0
    return to_t(f0), to_t(f1), torch.from_numpy(k.astype(np.float32))


def train_td_head(encoder: nn.Module, ds: PlayDataset, latent_dim: int, steps: int = 2000,
                  batch: int = 256, max_offset: int = 64, lr: float = 3e-4, device: str = "cpu",
                  seed: int = 0) -> TDHead:
    rng = np.random.default_rng(seed)
    head = TDHead(latent_dim).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    encoder.eval()
    for step in range(steps):
        x0, x1, k = sample_pairs(ds, batch, max_offset, rng)
        with torch.no_grad():
            z0, z1 = encoder(x0.to(device)), encoder(x1.to(device))
        loss = ((head(z0, z1) - k.to(device).sqrt()) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"[td] step {step:5d} loss {loss.item():.4f}")
    return head
