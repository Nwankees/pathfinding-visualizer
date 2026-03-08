# Pathfinding Visualizer

A Python project that implements and visualizes shortest-path search on weighted grids using **Dijkstra** and **A*** algorithms.  
The project also includes a benchmarking mode to compare different implementation strategies.

## Project Structure

```text
pathfinding-visualizer/
├── benchmarks/
│   ├── astar_results.csv
│   └── pathfinding_benchmark.csv
├── src/
│   ├── main.py
│   └── pathviz/
│       ├── astar.py
│       ├── dijkstra.py
│       ├── grid.py
│       └── visualize.py
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Visualization

Run Dijkstra:

```bash
python -m src.main --algorithm dijkstra --mode baseline
```

Run A*:

```bash
python -m src.main --algorithm astar --mode baseline
```

Run without visualization:

```bash
python -m src.main --algorithm dijkstra --mode array --no-viz
```

## Benchmark Example

```bash
python -m src.main --algorithm dijkstra --benchmark --rows 300 --cols 300 --trials 10 --no-viz --benchmark-modes baseline,array,vectorized
```

Results are written to:

```text
benchmarks/pathfinding_benchmark.csv
```

## Technologies

- Python
- NumPy
- Matplotlib

---
