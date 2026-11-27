"""2D grid mazes with a continuous point agent, pixel rendering and exact BFS distances.

A maze is a boolean grid ``walls[r, c]`` (True = wall). The agent lives in
continuous (row, col) coordinates measured in cells; cell (r, c) covers
[r, r+1) x [c, c+1). Actions are 2D velocities clipped to ``max_step`` cells.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------- layouts

def two_rooms(size: int = 9, door_width: int = 1, rng: np.random.Generator | None = None) -> np.ndarray:
    """Two rooms separated by a vertical wall with one door (PLDM-style)."""
    rng = rng or np.random.default_rng()
    walls = np.zeros((size, size), dtype=bool)
    walls[0, :] = walls[-1, :] = walls[:, 0] = walls[:, -1] = True
    mid = size // 2
    walls[:, mid] = True
    door = rng.integers(1, size - 1 - door_width + 1)
    walls[door:door + door_width, mid] = False
    return walls


def backtracker_maze(n_cells: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Perfect maze via recursive backtracking. Grid size is 2 * n_cells + 1."""
    rng = rng or np.random.default_rng()
    size = 2 * n_cells + 1
    walls = np.ones((size, size), dtype=bool)
    visited = np.zeros((n_cells, n_cells), dtype=bool)
    stack = [(0, 0)]
    visited[0, 0] = True
    walls[1, 1] = False
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= r + dr < n_cells and 0 <= c + dc < n_cells and not visited[r + dr, c + dc]]
        if not nbrs:
            stack.pop()
            continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        walls[2 * nr + 1, 2 * nc + 1] = False
        walls[r + nr + 1, c + nc + 1] = False  # knock down wall between the two cells
        visited[nr, nc] = True
        stack.append((nr, nc))
    return walls


def make_layout(family: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if family == "two_rooms":
        return two_rooms(9, rng=rng)
    if family == "medium":
        return backtracker_maze(4, rng=rng)  # 9 x 9 grid
    if family == "large":
        return backtracker_maze(7, rng=rng)  # 15 x 15 grid
    raise ValueError(f"unknown maze family: {family}")


# ---------------------------------------------------------------- distances

def free_cells(walls: np.ndarray) -> np.ndarray:
    return np.argwhere(~walls)


def bfs_distances(walls: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    """Shortest path length (in cells, 4-connected) from ``start`` to every cell; -1 if unreachable."""
    dist = -np.ones(walls.shape, dtype=np.int32)
    dist[start] = 0
    q = deque([start])
    while q:
        r, c = q.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < walls.shape[0] and 0 <= nc < walls.shape[1] and not walls[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


# ---------------------------------------------------------------- environment

@dataclass
class MazeEnv:
    walls: np.ndarray
    img_size: int = 64
    max_step: float = 0.5

    def __post_init__(self) -> None:
        self.pos = np.zeros(2, dtype=np.float32)
        self._base = self._render_walls()

    # -- dynamics
    def _is_free(self, p: np.ndarray) -> bool:
        r, c = int(np.floor(p[0])), int(np.floor(p[1]))
        if r < 0 or c < 0 or r >= self.walls.shape[0] or c >= self.walls.shape[1]:
            return False
        return not self.walls[r, c]

    def reset(self, cell: tuple[int, int]) -> np.ndarray:
        self.pos = np.array([cell[0] + 0.5, cell[1] + 0.5], dtype=np.float32)
        return self.render()

    def step(self, action: np.ndarray) -> np.ndarray:
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0) * self.max_step
        # sub-steps avoid tunnelling through walls; slide along walls axis by axis
        for _ in range(4):
            for axis in (0, 1):
                trial = self.pos.copy()
                trial[axis] += a[axis] / 4
                if self._is_free(trial):
                    self.pos = trial
        return self.render()

    @property
    def cell(self) -> tuple[int, int]:
        return int(self.pos[0]), int(self.pos[1])

    # -- rendering
    def _render_walls(self) -> np.ndarray:
        g = self.walls.shape[0]
        idx = (np.arange(self.img_size) * g // self.img_size)
        wall_img = self.walls[np.ix_(idx, idx)]
        img = np.full((self.img_size, self.img_size, 3), 255, dtype=np.uint8)
        img[wall_img] = 0
        return img

    def render(self, pos: np.ndarray | None = None) -> np.ndarray:
        pos = self.pos if pos is None else pos
        img = self._base.copy()
        scale = self.img_size / self.walls.shape[0]
        cy, cx = pos * scale
        rad = max(1.5, 0.35 * scale)
        yy, xx = np.ogrid[: self.img_size, : self.img_size]
        mask = (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 <= rad ** 2
        img[mask] = (220, 40, 40)
        return img

    def render_cell(self, cell: tuple[int, int]) -> np.ndarray:
        return self.render(np.array([cell[0] + 0.5, cell[1] + 0.5], dtype=np.float32))
