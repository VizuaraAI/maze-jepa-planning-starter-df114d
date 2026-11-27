"""Reward-free random play collection and a PyTorch dataset of sub-trajectories.

We store positions and actions only (a few MB even for 3M transitions) and
render frames on the fly, so no large image files are written to disk.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .maze import MazeEnv, free_cells, make_layout


def collect_random_play(family: str, n_transitions: int, traj_len: int, n_layouts: int,
                        seed: int, out_path: str | Path, max_step: float = 0.5) -> Path:
    """Collect uniform random-action trajectories over ``n_layouts`` layouts of a family."""
    rng = np.random.default_rng(seed)
    n_traj = max(1, n_transitions // traj_len)
    layout_seeds = np.arange(n_layouts) + 1000 * seed
    pos = np.zeros((n_traj, traj_len + 1, 2), dtype=np.float32)
    act = np.zeros((n_traj, traj_len, 2), dtype=np.float32)
    layout_id = np.zeros(n_traj, dtype=np.int32)
    envs = [MazeEnv(make_layout(family, int(s)), max_step=max_step) for s in layout_seeds]
    for i in range(n_traj):
        k = int(rng.integers(n_layouts))
        env = envs[k]
        cells = free_cells(env.walls)
        env.reset(tuple(cells[rng.integers(len(cells))]))
        pos[i, 0] = env.pos
        for t in range(traj_len):
            a = rng.uniform(-1.0, 1.0, size=2).astype(np.float32)
            env.step(a)
            act[i, t] = a
            pos[i, t + 1] = env.pos
        layout_id[i] = k
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, pos=pos, act=act, layout_id=layout_id,
                        layout_seeds=layout_seeds, family=family, max_step=max_step)
    return out_path


class PlayDataset(Dataset):
    """Returns (frames[k+1, 3, H, W] float in [0,1], actions[k, 2], traj_idx, t0)."""

    def __init__(self, path: str | Path, seq_len: int = 4, img_size: int = 64):
        d = np.load(path, allow_pickle=False)
        self.pos, self.act, self.layout_id = d["pos"], d["act"], d["layout_id"]
        family = str(d["family"])
        self.envs = [MazeEnv(make_layout(family, int(s)), img_size=img_size, max_step=float(d["max_step"]))
                     for s in d["layout_seeds"]]
        self.seq_len = seq_len
        self.traj_len = self.act.shape[1]
        self.per_traj = self.traj_len - seq_len + 1

    def __len__(self) -> int:
        return len(self.pos) * self.per_traj

    def __getitem__(self, idx: int):
        i, t0 = divmod(idx, self.per_traj)
        env = self.envs[self.layout_id[i]]
        frames = np.stack([env.render(self.pos[i, t]) for t in range(t0, t0 + self.seq_len + 1)])
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        actions = torch.from_numpy(self.act[i, t0:t0 + self.seq_len])
        return frames, actions, i, t0


def to_tensor(img: np.ndarray, device: str | torch.device = "cpu") -> torch.Tensor:
    """HWC uint8 image (or batch) to NCHW float tensor."""
    x = torch.from_numpy(np.ascontiguousarray(img)).float() / 255.0
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x.permute(0, 3, 1, 2).to(device)
