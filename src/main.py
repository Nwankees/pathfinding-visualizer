from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

from src.pathviz.astar import astar_steps, astar_steps_vectorized, astar_steps_array
from src.pathviz.dijkstra import dijkstra_steps, dijkstra_steps_vectorized, dijkstra_steps_array, SearchStats
from src.pathviz.grid import GridSpec, WeightedGrid


def consume_steps(steps) -> dict:
    last = None
    for ev in steps:
        last = ev
    if last is None or last.get("type") != "done":
        raise RuntimeError("Search generator finished without a done event.")
    return last


def select_steps(algorithm: str, mode: str, grid: WeightedGrid, start: tuple[int, int], goal: tuple[int, int]):
    if algorithm == "dijkstra":
        if mode == "baseline":
            return dijkstra_steps(grid, start, goal)
        if mode == "vectorized":
            return dijkstra_steps_vectorized(grid, start, goal)
        return dijkstra_steps_array(grid, start, goal)

    if mode == "baseline":
        return astar_steps(grid, start, goal)
    if mode == "vectorized":
        return astar_steps_vectorized(grid, start, goal)
    return astar_steps_array(grid, start, goal)


def run_once(
    algorithm: str,
    mode: str,
    spec: GridSpec,
    start: tuple[int, int],
    goal: tuple[int, int],
    visualize: bool,
    frame_skip: int,
    interval_ms: int,
) -> SearchStats:
    grid = WeightedGrid(spec)
    grid.ensure_clear(start, goal)
    steps = select_steps(algorithm, mode, grid, start, goal)
    title = f"{algorithm.upper()} ({mode})"

    t0 = time.perf_counter()
    if visualize:
        from src.pathviz.visualize import animate_search
        animate_search(grid, steps, start, goal, title=title, frame_skip=frame_skip, interval_ms=interval_ms)
        grid = WeightedGrid(spec)
        grid.ensure_clear(start, goal)
        steps = select_steps(algorithm, mode, grid, start, goal)

    done = consume_steps(steps)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    stats: SearchStats = done["stats"]
    stats.elapsed_ms = elapsed_ms
    return stats


def benchmark(args) -> None:
    start = (args.start_row, args.start_col)
    goal = (args.goal_row if args.goal_row is not None else args.rows - 3,
            args.goal_col if args.goal_col is not None else args.cols - 3)

    modes = [m.strip() for m in args.benchmark_modes.split(",") if m.strip()]
    if len(modes) < 2:
        raise ValueError("--benchmark-modes must contain at least two modes")

    rows = []
    time_buckets = {mode: [] for mode in modes}

    for trial in range(args.trials):
        spec = GridSpec(
            rows=args.rows,
            cols=args.cols,
            obstacle_p=args.obstacle_p,
            seed=args.seed + trial,
            min_cost=args.min_cost,
            max_cost=args.max_cost,
        )

        results = {
            mode: run_once(args.algorithm, mode, spec, start, goal, False, args.frame_skip, args.interval_ms)
            for mode in modes
        }

        ref = results[modes[0]]
        for mode in modes[1:]:
            cur = results[mode]
            if ref.found != cur.found:
                raise RuntimeError(f"{modes[0]} and {mode} disagree on path existence.")
            if ref.found and abs((ref.final_cost or 0.0) - (cur.final_cost or 0.0)) > 1e-9:
                raise RuntimeError(f"{modes[0]} and {mode} disagree on final shortest-path cost.")

        row = {
            "trial": trial + 1,
            "seed": spec.seed,
            "algorithm": args.algorithm,
            "rows": args.rows,
            "cols": args.cols,
            "obstacle_p": args.obstacle_p,
            "found": ref.found,
            "final_cost": None if ref.final_cost is None else round(ref.final_cost, 3),
        }
        for mode, stats in results.items():
            time_buckets[mode].append(stats.elapsed_ms)
            row[f"{mode}_ms"] = round(stats.elapsed_ms, 3)
            row[f"{mode}_nodes_popped"] = stats.nodes_popped
            row[f"{mode}_relaxations"] = stats.relaxations
            row[f"{mode}_path_length"] = stats.path_length
        if "baseline" in results:
            for mode in modes:
                if mode != "baseline":
                    denom = results[mode].elapsed_ms
                    row[f"baseline_vs_{mode}_speedup_x"] = round(results["baseline"].elapsed_ms / denom, 3) if denom > 0 else None
        rows.append(row)

    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Algorithm:           {args.algorithm}")
    print(f"Grid:                {args.rows} x {args.cols}")
    print(f"Obstacle density:    {args.obstacle_p}")
    print(f"Trials:              {args.trials}")
    print(f"Modes:               {', '.join(modes)}")
    for mode in modes:
        print(f"Mean {mode:10s} ms: {statistics.mean(time_buckets[mode]):.3f}")
        print(f"Median {mode:8s} ms: {statistics.median(time_buckets[mode]):.3f}")
    if "baseline" in time_buckets:
        for mode in modes:
            if mode != "baseline":
                avg_speedup = statistics.mean(time_buckets["baseline"]) / statistics.mean(time_buckets[mode])
                print(f"Mean speedup baseline/{mode}: {avg_speedup:.3f}x")

    if args.csv_out and rows:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved trial results to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grid pathfinding visualizer and benchmark harness")
    parser.add_argument("--algorithm", choices=["dijkstra", "astar"], default="dijkstra")
    parser.add_argument("--mode", choices=["baseline", "vectorized", "array"], default="array")
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--obstacle-p", type=float, default=0.28)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-cost", type=int, default=1)
    parser.add_argument("--max-cost", type=int, default=9)
    parser.add_argument("--start-row", type=int, default=2)
    parser.add_argument("--start-col", type=int, default=2)
    parser.add_argument("--goal-row", type=int, default=None)
    parser.add_argument("--goal-col", type=int, default=None)
    parser.add_argument("--frame-skip", type=int, default=30)
    parser.add_argument("--interval-ms", type=int, default=15)
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--benchmark-modes", type=str, default="baseline,array,vectorized")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--csv-out", type=str, default="benchmarks/pathfinding_benchmark.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.goal_row is None:
        args.goal_row = args.rows - 3
    if args.goal_col is None:
        args.goal_col = args.cols - 3

    start = (args.start_row, args.start_col)
    goal = (args.goal_row, args.goal_col)

    if args.benchmark:
        benchmark(args)
        return

    spec = GridSpec(
        rows=args.rows,
        cols=args.cols,
        obstacle_p=args.obstacle_p,
        seed=args.seed,
        min_cost=args.min_cost,
        max_cost=args.max_cost,
    )

    stats = run_once(
        algorithm=args.algorithm,
        mode=args.mode,
        spec=spec,
        start=start,
        goal=goal,
        visualize=not args.no_viz,
        frame_skip=args.frame_skip,
        interval_ms=args.interval_ms,
    )

    print("\n=== RUN SUMMARY ===")
    print(f"Algorithm:           {stats.algorithm}")
    print(f"Mode:                {stats.mode}")
    print(f"Found path:          {stats.found}")
    print(f"Final cost:          {stats.final_cost}")
    print(f"Nodes popped:        {stats.nodes_popped}")
    print(f"Relaxations:         {stats.relaxations}")
    print(f"Pushes:              {stats.pushes}")
    print(f"Max frontier size:   {stats.max_frontier_size}")
    print(f"Visited count:       {stats.visited_count}")
    print(f"Path length:         {stats.path_length}")
    print(f"Elapsed ms:          {stats.elapsed_ms:.3f}")


if __name__ == "__main__":
    main()
