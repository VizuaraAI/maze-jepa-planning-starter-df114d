"""A compact LeWorldModel-style JEPA: pixel encoder, action-conditioned predictor, SIGReg.

This is a small, readable re-implementation for 64x64 mazes. For the paper's
main numbers, port the official LeWorldModel encoder/predictor (see README).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = 64, width: int = 32):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(3, w, 4, 2, 1), nn.GELU(),          # 32
            nn.Conv2d(w, 2 * w, 4, 2, 1), nn.GELU(),      # 16
            nn.Conv2d(2 * w, 4 * w, 4, 2, 1), nn.GELU(),  # 8
            nn.Conv2d(4 * w, 8 * w, 4, 2, 1), nn.GELU(),  # 4
            nn.Flatten(),
            nn.Linear(8 * w * 16, 256), nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Predictor(nn.Module):
    """z_{t+1} = z_t + MLP([z_t, a_t])."""

    def __init__(self, latent_dim: int = 64, action_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))

    def rollout(self, z0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """z0: [B, D], actions: [B, H, A] -> latents [B, H, D]."""
        zs, z = [], z0
        for t in range(actions.shape[1]):
            z = self(z, actions[:, t])
            zs.append(z)
        return torch.stack(zs, dim=1)


def sigreg(z: torch.Tensor, n_proj: int = 256, n_t: int = 17, t_max: float = 5.0) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (Epps-Pulley statistic on random 1D projections).

    Projects embeddings onto random unit directions and penalizes the weighted
    squared gap between the empirical characteristic function of each projection
    and that of N(0, 1). See Balestriero and LeCun (2025), LeJEPA.
    """
    n, d = z.shape
    dirs = torch.randn(d, n_proj, device=z.device, dtype=z.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    x = z @ dirs                                                   # [N, P]
    t = torch.linspace(0.0, t_max, n_t, device=z.device, dtype=z.dtype)  # symmetric, so use t >= 0
    xt = x.unsqueeze(-1) * t                                       # [N, P, T]
    ecf_re, ecf_im = torch.cos(xt).mean(0), torch.sin(xt).mean(0)  # [P, T]
    target = torch.exp(-0.5 * t ** 2)
    weight = torch.exp(-0.5 * t ** 2)
    err = ((ecf_re - target) ** 2 + ecf_im ** 2) * weight
    stat = torch.trapezoid(err, t, dim=-1) * 2.0 * n               # integrate over R via symmetry
    return stat.mean()


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = 64, action_dim: int = 2, enc_width: int = 32, pred_hidden: int = 512):
        super().__init__()
        self.encoder = Encoder(latent_dim, enc_width)
        self.predictor = Predictor(latent_dim, action_dim, pred_hidden)

    def loss(self, frames: torch.Tensor, actions: torch.Tensor, sigreg_weight: float) -> dict[str, torch.Tensor]:
        """frames: [B, K+1, 3, H, W], actions: [B, K, A]. Multi-step latent prediction + SIGReg."""
        b, k1 = frames.shape[:2]
        z = self.encoder(frames.flatten(0, 1)).view(b, k1, -1)     # [B, K+1, D]
        pred = self.predictor.rollout(z[:, 0], actions)            # [B, K, D]
        pred_loss = ((pred - z[:, 1:]) ** 2).sum(-1).mean()
        reg = sigreg(z.flatten(0, 1))
        return {"loss": pred_loss + sigreg_weight * reg, "pred": pred_loss, "sigreg": reg}
