from __future__ import annotations

import heapq
import math
from typing import Dict, Iterator, Tuple

import numpy as np

from .grid import WeightedGrid
from .dijkstra import (
    SearchStats,
    reconstruct_path,
    reconstruct_path_arrays,
    dist_array_to_dict,
)

Pos = Tuple[int, int]


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar_steps(grid: WeightedGrid, start: Pos, goal: Pos) -> Iterator[dict]:
    pq: list[tuple[float, float, Pos]] = [(0.0, 0.0, start)]
    g_score: Dict[Pos, float] = {start: 0.0}
    came_from: Dict[Pos, Pos] = {}
    closed: set[Pos] = set()

    nodes_popped = 0
    relaxations = 0
    pushes = 1
    max_frontier_size = 1

    while pq:
        f, g, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        nodes_popped += 1

        yield {"type": "pop", "node": u, "dist": g}

        if u == goal:
            path = reconstruct_path(came_from, start, goal)
            stats = SearchStats(
                algorithm="astar",
                mode="baseline",
                found=True,
                final_cost=g,
                nodes_popped=nodes_popped,
                relaxations=relaxations,
                pushes=pushes,
                max_frontier_size=max_frontier_size,
                visited_count=len(closed),
                path_length=len(path),
            )
            yield {"type": "done", "found": True, "dist": g, "came_from": came_from, "dist_map": g_score, "stats": stats}
            return

        ur, uc = u
        for v in grid.neighbors4(ur, uc):
            if v in closed:
                continue
            relaxations += 1
            vr, vc = v
            tentative_g = g + grid.step_cost(vr, vc)
            if tentative_g < g_score.get(v, math.inf):
                g_score[v] = tentative_g
                came_from[v] = u
                f_score = tentative_g + manhattan(v, goal)
                heapq.heappush(pq, (f_score, tentative_g, v))
                pushes += 1
                max_frontier_size = max(max_frontier_size, len(pq))
                yield {"type": "relax", "from": u, "to": v, "new_dist": tentative_g}

    stats = SearchStats(
        algorithm="astar",
        mode="baseline",
        found=False,
        final_cost=None,
        nodes_popped=nodes_popped,
        relaxations=relaxations,
        pushes=pushes,
        max_frontier_size=max_frontier_size,
        visited_count=len(closed),
        path_length=0,
    )
    yield {"type": "done", "found": False, "dist": None, "came_from": came_from, "dist_map": g_score, "stats": stats}


def astar_steps_vectorized(grid: WeightedGrid, start: Pos, goal: Pos) -> Iterator[dict]:
    rows, cols = grid.spec.rows, grid.spec.cols
    pq: list[tuple[float, float, Pos]] = [(0.0, 0.0, start)]
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[start] = 0.0
    closed = np.zeros((rows, cols), dtype=bool)
    came_from: Dict[Pos, Pos] = {}

    nodes_popped = 0
    relaxations = 0
    pushes = 1
    max_frontier_size = 1
    min_step = float(grid.cost[~grid.obstacles].min()) if np.any(~grid.obstacles) else 1.0

    while pq:
        f, g, u = heapq.heappop(pq)
        ur, uc = u
        if closed[ur, uc]:
            continue
        closed[ur, uc] = True
        nodes_popped += 1

        yield {"type": "pop", "node": u, "dist": g}

        if u == goal:
            path = reconstruct_path(came_from, start, goal)
            stats = SearchStats(
                algorithm="astar",
                mode="vectorized",
                found=True,
                final_cost=float(g),
                nodes_popped=nodes_popped,
                relaxations=relaxations,
                pushes=pushes,
                max_frontier_size=max_frontier_size,
                visited_count=int(closed.sum()),
                path_length=len(path),
            )
            yield {"type": "done", "found": True, "dist": float(g), "came_from": came_from, "dist_map": dist_array_to_dict(g_score), "stats": stats}
            return

        nr, nc = grid.neighbors4_vectorized(ur, uc)
        if nr.size == 0:
            continue

        open_mask = ~closed[nr, nc]
        nr = nr[open_mask]
        nc = nc[open_mask]
        if nr.size == 0:
            continue

        relaxations += int(nr.size)
        tentative = g + grid.cost[nr, nc].astype(np.float64)
        current = g_score[nr, nc]
        improved = tentative < current
        if not np.any(improved):
            continue

        nr2 = nr[improved]
        nc2 = nc[improved]
        tg2 = tentative[improved]
        g_score[nr2, nc2] = tg2
        heur = (np.abs(nr2 - goal[0]) + np.abs(nc2 - goal[1])).astype(np.float64) * min_step
        f2 = tg2 + heur

        for rr, cc, ng, nf in zip(nr2.tolist(), nc2.tolist(), tg2.tolist(), f2.tolist()):
            v = (rr, cc)
            came_from[v] = u
            heapq.heappush(pq, (float(nf), float(ng), v))
            pushes += 1
            yield {"type": "relax", "from": u, "to": v, "new_dist": float(ng)}

        max_frontier_size = max(max_frontier_size, len(pq))

    stats = SearchStats(
        algorithm="astar",
        mode="vectorized",
        found=False,
        final_cost=None,
        nodes_popped=nodes_popped,
        relaxations=relaxations,
        pushes=pushes,
        max_frontier_size=max_frontier_size,
        visited_count=int(closed.sum()),
        path_length=0,
    )
    yield {"type": "done", "found": False, "dist": None, "came_from": came_from, "dist_map": dist_array_to_dict(g_score), "stats": stats}


def astar_steps_array(grid: WeightedGrid, start: Pos, goal: Pos) -> Iterator[dict]:
    rows, cols = grid.spec.rows, grid.spec.cols
    pq: list[tuple[float, float, int, int]] = [(float(manhattan(start, goal)), 0.0, start[0], start[1])]

    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[start] = 0.0
    closed = np.zeros((rows, cols), dtype=bool)
    parent_r = np.full((rows, cols), -1, dtype=np.int32)
    parent_c = np.full((rows, cols), -1, dtype=np.int32)

    nodes_popped = 0
    relaxations = 0
    pushes = 1
    max_frontier_size = 1
    min_step = float(grid.cost[~grid.obstacles].min()) if np.any(~grid.obstacles) else 1.0

    while pq:
        f, g, ur, uc = heapq.heappop(pq)
        if closed[ur, uc]:
            continue
        if g != g_score[ur, uc]:
            continue

        closed[ur, uc] = True
        nodes_popped += 1
        u = (ur, uc)
        yield {"type": "pop", "node": u, "dist": float(g)}

        if (ur, uc) == goal:
            path = reconstruct_path_arrays(parent_r, parent_c, start, goal)
            stats = SearchStats(
                algorithm="astar",
                mode="array",
                found=True,
                final_cost=float(g),
                nodes_popped=nodes_popped,
                relaxations=relaxations,
                pushes=pushes,
                max_frontier_size=max_frontier_size,
                visited_count=int(closed.sum()),
                path_length=len(path),
            )
            came_from = {
                (int(r), int(c)): (int(parent_r[r, c]), int(parent_c[r, c]))
                for r, c in np.argwhere(parent_r >= 0)
            }
            yield {"type": "done", "found": True, "dist": float(g), "came_from": came_from, "dist_map": dist_array_to_dict(g_score), "stats": stats}
            return

        for vr, vc in grid.neighbors4(ur, uc):
            if closed[vr, vc]:
                continue
            relaxations += 1
            ng = g + float(grid.cost[vr, vc])
            if ng < g_score[vr, vc]:
                g_score[vr, vc] = ng
                parent_r[vr, vc] = ur
                parent_c[vr, vc] = uc
                h = (abs(vr - goal[0]) + abs(vc - goal[1])) * min_step
                heapq.heappush(pq, (ng + h, ng, vr, vc))
                pushes += 1
                max_frontier_size = max(max_frontier_size, len(pq))
                yield {"type": "relax", "from": u, "to": (vr, vc), "new_dist": float(ng)}

    stats = SearchStats(
        algorithm="astar",
        mode="array",
        found=False,
        final_cost=None,
        nodes_popped=nodes_popped,
        relaxations=relaxations,
        pushes=pushes,
        max_frontier_size=max_frontier_size,
        visited_count=int(closed.sum()),
        path_length=0,
    )
    came_from = {
        (int(r), int(c)): (int(parent_r[r, c]), int(parent_c[r, c]))
        for r, c in np.argwhere(parent_r >= 0)
    }
    yield {"type": "done", "found": False, "dist": None, "came_from": came_from, "dist_map": dist_array_to_dict(g_score), "stats": stats}
