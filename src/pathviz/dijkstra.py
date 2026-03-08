from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterator, Tuple

import numpy as np

from .grid import WeightedGrid

Pos = Tuple[int, int]

@dataclass
class SearchStats:
    algorithm: str
    mode: str
    found: bool
    final_cost: float | None
    nodes_popped: int
    relaxations: int
    pushes: int
    max_frontier_size: int
    visited_count: int
    path_length: int
    elapsed_ms: float = 0.0


def reconstruct_path(came_from: Dict[Pos, Pos], start: Pos, goal: Pos) -> list[Pos]:
    if goal not in came_from and goal != start:
        return []
    cur = goal
    path = [cur]
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path


def reconstruct_path_arrays(parent_r: np.ndarray, parent_c: np.ndarray, start: Pos, goal: Pos) -> list[Pos]:
    sr, sc = start
    gr, gc = goal
    if (gr, gc) != (sr, sc) and parent_r[gr, gc] < 0:
        return []

    path: list[Pos] = []
    r, c = gr, gc
    while True:
        path.append((int(r), int(c)))
        if (r, c) == (sr, sc):
            break
        pr = int(parent_r[r, c])
        pc = int(parent_c[r, c])
        if pr < 0 or pc < 0:
            return []
        r, c = pr, pc
    path.reverse()
    return path


def dist_array_to_dict(dist_arr: np.ndarray) -> dict[Pos, float]:
    coords = np.argwhere(np.isfinite(dist_arr))
    return {(int(r), int(c)): float(dist_arr[r, c]) for r, c in coords}


def dijkstra_steps(grid: WeightedGrid, start: Pos, goal: Pos) -> Iterator[dict]:
    pq: list[tuple[float, Pos]] = [(0.0, start)]
    dist: Dict[Pos, float] = {start: 0.0}
    came_from: Dict[Pos, Pos] = {}
    visited: set[Pos] = set()

    nodes_popped = 0
    relaxations = 0
    pushes = 1
    max_frontier_size = 1

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        nodes_popped += 1

        yield {"type": "pop", "node": u, "dist": d}

        if u == goal:
            path = reconstruct_path(came_from, start, goal)
            stats = SearchStats(
                algorithm="dijkstra",
                mode="baseline",
                found=True,
                final_cost=d,
                nodes_popped=nodes_popped,
                relaxations=relaxations,
                pushes=pushes,
                max_frontier_size=max_frontier_size,
                visited_count=len(visited),
                path_length=len(path),
            )
            yield {"type": "done", "found": True, "dist": d, "came_from": came_from, "dist_map": dist, "stats": stats}
            return

        ur, uc = u
        for v in grid.neighbors4(ur, uc):
            if v in visited:
                continue
            relaxations += 1
            vr, vc = v
            nd = d + grid.step_cost(vr, vc)
            if nd < dist.get(v, math.inf):
                dist[v] = nd
                came_from[v] = u
                heapq.heappush(pq, (nd, v))
                pushes += 1
                max_frontier_size = max(max_frontier_size, len(pq))
                yield {"type": "relax", "from": u, "to": v, "new_dist": nd}

    stats = SearchStats(
        algorithm="dijkstra",
        mode="baseline",
        found=False,
        final_cost=None,
        nodes_popped=nodes_popped,
        relaxations=relaxations,
        pushes=pushes,
        max_frontier_size=max_frontier_size,
        visited_count=len(visited),
        path_length=0,
    )
    yield {"type": "done", "found": False, "dist": None, "came_from": came_from, "dist_map": dist, "stats": stats}


def dijkstra_steps_vectorized(grid: WeightedGrid, start: Pos, goal: Pos) -> Iterator[dict]:
    rows, cols = grid.spec.rows, grid.spec.cols
    pq: list[tuple[float, Pos]] = [(0.0, start)]
    dist_arr = np.full((rows, cols), np.inf, dtype=np.float64)
    dist_arr[start] = 0.0
    visited = np.zeros((rows, cols), dtype=bool)
    came_from: Dict[Pos, Pos] = {}

    nodes_popped = 0
    relaxations = 0
    pushes = 1
    max_frontier_size = 1

    while pq:
        d, u = heapq.heappop(pq)
        ur, uc = u
        if visited[ur, uc]:
            continue
        visited[ur, uc] = True
        nodes_popped += 1

        yield {"type": "pop", "node": u, "dist": d}

        if u == goal:
            path = reconstruct_path(came_from, start, goal)
            stats = SearchStats(
                algorithm="dijkstra",
                mode="vectorized",
                found=True,
                final_cost=float(d),
                nodes_popped=nodes_popped,
                relaxations=relaxations,
                pushes=pushes,
                max_frontier_size=max_frontier_size,
                visited_count=int(visited.sum()),
                path_length=len(path),
            )
            yield {"type": "done", "found": True, "dist": float(d), "came_from": came_from, "dist_map": dist_array_to_dict(dist_arr), "stats": stats}
            return

        nr, nc = grid.neighbors4_vectorized(ur, uc)
        if nr.size == 0:
            continue

        open_mask = ~visited[nr, nc]
        nr = nr[open_mask]
        nc = nc[open_mask]
        if nr.size == 0:
            continue

        relaxations += int(nr.size)
        candidate = d + grid.cost[nr, nc].astype(np.float64)
        current = dist_arr[nr, nc]
        improved = candidate < current
        if not np.any(improved):
            continue

        nr2 = nr[improved]
        nc2 = nc[improved]
        cand2 = candidate[improved]
        dist_arr[nr2, nc2] = cand2

        for rr, cc, nd in zip(nr2.tolist(), nc2.tolist(), cand2.tolist()):
            v = (rr, cc)
            came_from[v] = u
            heapq.heappush(pq, (float(nd), v))
            pushes += 1
            yield {"type": "relax", "from": u, "to": v, "new_dist": float(nd)}

        max_frontier_size = max(max_frontier_size, len(pq))

    stats = SearchStats(
        algorithm="dijkstra",
        mode="vectorized",
        found=False,
        final_cost=None,
        nodes_popped=nodes_popped,
        relaxations=relaxations,
        pushes=pushes,
        max_frontier_size=max_frontier_size,
        visited_count=int(visited.sum()),
        path_length=0,
    )
    yield {"type": "done", "found": False, "dist": None, "came_from": came_from, "dist_map": dist_array_to_dict(dist_arr), "stats": stats}


def dijkstra_steps_array(grid: WeightedGrid, start: Pos, goal: Pos) -> Iterator[dict]:
    rows, cols = grid.spec.rows, grid.spec.cols
    pq: list[tuple[float, int, int]] = [(0.0, start[0], start[1])]

    dist = np.full((rows, cols), np.inf, dtype=np.float64)
    dist[start] = 0.0
    visited = np.zeros((rows, cols), dtype=bool)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    nodes_popped = 0
    relaxations = 0
    pushes = 1
    max_frontier_size = 1

    while pq:
        d, ur, uc = heapq.heappop(pq)
        if visited[ur, uc]:
            continue
        if d != dist[ur, uc]:
            continue

        visited[ur, uc] = True
        nodes_popped += 1
        u = (ur, uc)
        yield {"type": "pop", "node": u, "dist": float(d)}

        if (ur, uc) == goal:
            path = reconstruct_path_arrays(parent_r, parent_c, start, goal)
            stats = SearchStats(
                algorithm="dijkstra",
                mode="array",
                found=True,
                final_cost=float(d),
                nodes_popped=nodes_popped,
                relaxations=relaxations,
                pushes=pushes,
                max_frontier_size=max_frontier_size,
                visited_count=int(visited.sum()),
                path_length=len(path),
            )
            came_from = {
                (int(r), int(c)): (int(parent_r[r, c]), int(parent_c[r, c]))
                for r, c in np.argwhere(parent_r >= 0)
            }
            yield {"type": "done", "found": True, "dist": float(d), "came_from": came_from, "dist_map": dist_array_to_dict(dist), "stats": stats}
            return

        for vr, vc in grid.neighbors4(ur, uc):
            if visited[vr, vc]:
                continue
            relaxations += 1
            nd = d + float(grid.cost[vr, vc])
            if nd < dist[vr, vc]:
                dist[vr, vc] = nd
                parent_r[vr, vc] = ur
                parent_c[vr, vc] = uc
                heapq.heappush(pq, (nd, vr, vc))
                pushes += 1
                max_frontier_size = max(max_frontier_size, len(pq))
                yield {"type": "relax", "from": u, "to": (vr, vc), "new_dist": float(nd)}

    stats = SearchStats(
        algorithm="dijkstra",
        mode="array",
        found=False,
        final_cost=None,
        nodes_popped=nodes_popped,
        relaxations=relaxations,
        pushes=pushes,
        max_frontier_size=max_frontier_size,
        visited_count=int(visited.sum()),
        path_length=0,
    )
    came_from = {
        (int(r), int(c)): (int(parent_r[r, c]), int(parent_c[r, c]))
        for r, c in np.argwhere(parent_r >= 0)
    }
    yield {"type": "done", "found": False, "dist": None, "came_from": came_from, "dist_map": dist_array_to_dict(dist), "stats": stats}
