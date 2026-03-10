from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
import numpy as np


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    obstacle_p: float = 0.22
    seed: int | None = None
    min_cost: int = 1
    max_cost: int = 9


class WeightedGrid:
    def __init__(self, spec: GridSpec):
        self.spec = spec
        rng = np.random.default_rng(spec.seed)

        self.obstacles = rng.random((spec.rows, spec.cols)) < spec.obstacle_p

        # Positive movement cost for stepping into a cell.
        self.cost = rng.integers(
            spec.min_cost,
            spec.max_cost + 1,
            size=(spec.rows, spec.cols),
            endpoint=False,
            dtype=np.int16,
        )

        self._build_neighbor_tables()

    def _build_neighbor_tables(self) -> None:
        rows, cols = self.spec.rows, self.spec.cols

        neigh_r = np.full((rows, cols, 4), -1, dtype=np.int32)
        neigh_c = np.full((rows, cols, 4), -1, dtype=np.int32)

        # Up
        neigh_r[1:, :, 0] = np.arange(rows - 1, dtype=np.int32)[:, None]
        neigh_c[1:, :, 0] = np.arange(cols, dtype=np.int32)[None, :]

        # Down
        neigh_r[:-1, :, 1] = np.arange(1, rows, dtype=np.int32)[:, None]
        neigh_c[:-1, :, 1] = np.arange(cols, dtype=np.int32)[None, :]

        # Left
        neigh_r[:, 1:, 2] = np.arange(rows, dtype=np.int32)[:, None]
        neigh_c[:, 1:, 2] = np.arange(cols - 1, dtype=np.int32)[None, :]

        # Right
        neigh_r[:, :-1, 3] = np.arange(rows, dtype=np.int32)[:, None]
        neigh_c[:, :-1, 3] = np.arange(1, cols, dtype=np.int32)[None, :]

        valid = neigh_r >= 0
        passable_mask = np.zeros_like(valid, dtype=bool)
        passable_mask[valid] = ~self.obstacles[neigh_r[valid], neigh_c[valid]]

        self.neigh_r = neigh_r
        self.neigh_c = neigh_c
        self.neigh_valid = valid & passable_mask

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.spec.rows and 0 <= c < self.spec.cols

    def passable(self, r: int, c: int) -> bool:
        return not self.obstacles[r, c]

    def neighbors4(self, r: int, c: int) -> Iterator[tuple[int, int]]:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if self.in_bounds(rr, cc) and self.passable(rr, cc):
                yield rr, cc

    def neighbors4_vectorized(self, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return valid neighbors for one cell as NumPy arrays.
        """
        mask = self.neigh_valid[r, c]
        return self.neigh_r[r, c, mask], self.neigh_c[r, c, mask]

    def step_cost(self, r: int, c: int) -> float:
        return float(self.cost[r, c])

    def ensure_clear(self, start: tuple[int, int], goal: tuple[int, int]) -> None:
        sr, sc = start
        gr, gc = goal
        self.obstacles[sr, sc] = False
        self.obstacles[gr, gc] = False
        self._build_neighbor_tables()