from __future__ import annotations

from typing import Iterable, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from .grid import WeightedGrid
from .dijkstra import reconstruct_path

Pos = Tuple[int, int]


def _base_image(grid: WeightedGrid, start: Pos, goal: Pos) -> np.ndarray:
    rows, cols = grid.spec.rows, grid.spec.cols
    img = np.zeros((rows, cols), dtype=np.float32)

    # obstacles as -1
    img[grid.obstacles] = -1.0

    # normalize costs into (0..1)
    cost_norm = (grid.cost.astype(np.float32) - grid.cost.min()) / max(1.0, float(grid.cost.max() - grid.cost.min()))
    img[~grid.obstacles] = 0.15 + 0.75 * cost_norm[~grid.obstacles]

    sr, sc = start
    gr, gc = goal
    img[sr, sc] = 1.25
    img[gr, gc] = 1.35
    return img

def animate_search(
    grid: WeightedGrid,
    steps,
    start: Pos,
    goal: Pos,
    title: str,
    interval_ms: int = 15,
    frame_skip: int = 1,
) -> None:
    base = _base_image(grid, start, goal)
    overlay = np.zeros_like(base, dtype=np.float32)

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(base, interpolation="nearest", cmap="gray", vmin=-1.0, vmax=1.5)

    im_over = ax.imshow(
        overlay,
        interpolation="nearest",
        cmap="plasma",
        vmin=0.0,
        vmax=1.0,
        alpha=0.65
    )

    plt.ion()
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    came_from = {}
    done = False
    found = False
    final_dist = None

    i = 0
    for ev in steps:
        t = ev["type"]

        if t == "pop":
            r, c = ev["node"]
            overlay[r, c] = 0.85  # visited
        elif t == "relax":
            r, c = ev["to"]
            # only paint if not already visited-looking
            overlay[r, c] = max(overlay[r, c], 0.45)  # frontier
        elif t == "done":
            done = True
            found = ev["found"]
            final_dist = ev["dist"]
            came_from = ev.get("came_from", {})
        else:
            continue

        i += 1
        if i % frame_skip == 0 or done:
            im_over.set_data(overlay)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(interval_ms / 1000.0)

        if done:
            break

    if found:
        path = reconstruct_path(came_from, start, goal)
        for r, c in path:
            overlay[r, c] = 1.0  # path
        im_over.set_data(overlay)
        ax.set_title(f"{title} — path found (cost={final_dist:.2f})")
    else:
        ax.set_title(f"{title} — no path")

    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.ioff()
    plt.show()